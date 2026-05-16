//! Model weight export utilities.
//!
//! Converts a collection of named `f32` tensors into a GGUF byte stream,
//! optionally quantizing each tensor to Q1\_0\_g128 or INT8 per-channel format
//! while respecting a configurable list of layers that must stay in FP32.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use oxibonsai_model::export::{ExportConfig, ExportFormat, WeightTensor, export_to_gguf};
//!
//! let tensors = vec![WeightTensor::new("blk.0.attn_q.weight", vec![1.0; 512], vec![8, 64])];
//! let config = ExportConfig::new(ExportFormat::Float32, "my-model");
//! let bytes = export_to_gguf(&tensors, &config, &[]).expect("export failed");
//! assert!(!bytes.is_empty());
//! ```

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

use crate::quantize::{q1_0_g128_size_bytes, quantize_q1_0_g128};
use crate::quantize_int8::quantize_per_channel;

// ─── Export format ────────────────────────────────────────────────────────────

/// The target quantization format for an export operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    /// Keep weights as IEEE 754 single precision floats.
    Float32,
    /// Quantize to Q1\_0\_g128 (1-bit sign + FP16 scale per 128-element group).
    Q1_0G128,
    /// Quantize to INT8 per output channel.
    Int8PerChannel,
    /// Ternary quantization: {-1, 0, +1} weights packed as TQ2_0_g128 (34 B / 128 weights).
    ///
    /// Embedding and LM-head tensors are ternary-encoded (unlike Q1_0G128 which keeps them
    /// FP16). Only RMS-norm weights remain FP32.
    TernaryG128,
    /// FP8 E4M3FN per-block quantization: 32 weights × 1 byte + FP16 scale (34 B / 32 weights).
    ///
    /// Uses the E4M3FN format (bias=7, no infinity, NaN at 0x7f/0xff). Provides
    /// approximately 8.5 bits per weight (34 bytes × 8 bits ÷ 32 weights). Maps
    /// to GGUF type ID 43 (PrismML extension). Only RMS-norm weights remain FP32.
    FP8E4M3,
    /// FP8 E5M2 per-block quantization: 32 weights × 1 byte + FP16 scale (34 B / 32 weights).
    ///
    /// Uses the E5M2 format (bias=15, has infinity). Provides higher dynamic range
    /// than E4M3 at the cost of mantissa precision. Maps to GGUF type ID 44
    /// (PrismML extension). Only RMS-norm weights remain FP32.
    FP8E5M2,
    /// Q4_0 quantization: 4-bit weights, 32 per block, FP16 scale (18 bytes/32 weights).
    ///
    /// Maps to GGML type ID 2. Each block stores a FP16 scale `d` and 32 nibbles
    /// packed 2-per-byte. Dequant: `w[j] = d × (nibble[j] − 8)`.
    Q4_0,
    /// Q8_0 quantization: 8-bit (int8) weights, 32 per block, FP16 scale (34 bytes/32 weights).
    ///
    /// Maps to GGML type ID 8. Each block stores a FP16 scale `d` and 32 int8 values.
    /// Dequant: `w[j] = d × qs[j]`. High fidelity — approximately 8.5 bits/weight.
    Q8_0,
    /// Q4_K quantization: 4-bit K-quant, 256 per super-block, 6-bit sub-scales (144 bytes/256 weights).
    ///
    /// Maps to GGML type ID 12. Each super-block stores FP16 `d`/`dmin`, 12 bytes of
    /// 6-bit sub-block scale/min pairs, and 128 bytes of packed 4-bit nibbles.
    Q4K,
    /// Q5_K quantization: 5-bit K-quant, 256 per super-block, 6-bit sub-scales (176 bytes/256 weights).
    ///
    /// Maps to GGML type ID 13. Extends Q4_K with an additional high-bit plane (32 bytes)
    /// stored in `qh`. Provides near-Q6 fidelity at reduced storage.
    Q5K,
    /// Q6_K quantization: 6-bit K-quant, 256 per super-block, int8 sub-scales (210 bytes/256 weights).
    ///
    /// Maps to GGML type ID 14. Each super-block stores 128 bytes of low nibbles, 64 bytes
    /// of high 2-bit pairs, 16 int8 sub-block scales, and a FP16 super-block scale.
    Q6K,
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Parameters that control how a model is exported.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Target weight format.
    pub format: ExportFormat,
    /// Human-readable model name, written into the GGUF `general.name` field.
    pub model_name: String,
    /// Version string (e.g. `"1.0.0"`), written into `general.version`.
    pub model_version: String,
    /// Optional free-text description placed in `general.description`.
    pub description: Option<String>,
    /// When `Some`, only quantize layers whose names appear in this list.
    /// When `None`, all eligible layers are quantized.
    pub quantize_layers: Option<Vec<String>>,
    /// Layer names that must remain in FP32 even when the global format is
    /// a quantized type.
    pub fp32_layers: Vec<String>,
}

impl ExportConfig {
    /// Create a minimal config with sensible defaults.
    pub fn new(format: ExportFormat, model_name: &str) -> Self {
        Self {
            format,
            model_name: model_name.to_string(),
            model_version: "1.0.0".to_string(),
            description: None,
            quantize_layers: None,
            fp32_layers: Vec::new(),
        }
    }

    /// Override the list of FP32 exception layers.
    pub fn with_fp32_layers(mut self, layers: Vec<String>) -> Self {
        self.fp32_layers = layers;
        self
    }

    /// Attach a free-text description to the GGUF `general.description` field.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Default set of layer name prefixes that should stay in FP32 when
    /// quantizing the rest of the model.
    ///
    /// Includes token embedding, output projection, and final normalization.
    pub fn default_fp32_exceptions() -> Vec<String> {
        vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
            "output.weight".to_string(),
        ]
    }
}

// ─── Weight tensor ────────────────────────────────────────────────────────────

/// A named `f32` weight tensor ready for export.
pub struct WeightTensor {
    /// Layer name used as the tensor name in the GGUF file.
    pub name: String,
    /// Flat weight data in row-major order.
    pub data: Vec<f32>,
    /// Shape `[d0, d1, …]`; `d0` is treated as the channel (output) dimension.
    pub shape: Vec<usize>,
}

impl WeightTensor {
    /// Construct a named weight tensor.
    pub fn new(name: &str, data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            data,
            shape,
        }
    }

    /// Total number of elements (product of shape dimensions).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Memory occupied by the raw `f32` data in bytes.
    pub fn memory_bytes_f32(&self) -> usize {
        self.data.len() * 4
    }
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur during a model export operation.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// A quantization step failed for the named tensor.
    #[error("Quantization error for tensor '{name}': {reason}")]
    QuantizeError { name: String, reason: String },

    /// The GGUF writer encountered an error.
    #[error("GGUF write error: {0}")]
    WriteError(String),

    /// The tensor list is empty — nothing to export.
    #[error("No tensors to export")]
    Empty,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Determine whether a tensor should be kept in FP32.
fn should_keep_fp32(name: &str, config: &ExportConfig) -> bool {
    // Explicit FP32 exception list takes priority.
    if config.fp32_layers.iter().any(|exc| name == exc.as_str()) {
        return true;
    }
    // If a quantize allowlist is active, only quantize tensors on it.
    if let Some(ref allowed) = config.quantize_layers {
        if !allowed.iter().any(|a| name == a.as_str()) {
            return true;
        }
    }
    false
}

/// Build the raw bytes and `TensorType` for a single weight tensor.
fn encode_tensor(
    tensor: &WeightTensor,
    config: &ExportConfig,
) -> Result<(Vec<u8>, TensorType), ExportError> {
    // Decide effective format for this tensor.
    let effective_format = if should_keep_fp32(&tensor.name, config) {
        ExportFormat::Float32
    } else {
        config.format
    };

    match effective_format {
        ExportFormat::Float32 => {
            let bytes: Vec<u8> = tensor.data.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok((bytes, TensorType::F32))
        }

        ExportFormat::Q1_0G128 => {
            // Pad to a multiple of GROUP_SIZE if necessary.
            use crate::quantize::GROUP_SIZE;
            let remainder = tensor.data.len() % GROUP_SIZE;
            let bytes = if remainder == 0 {
                quantize_q1_0_g128(&tensor.data).map_err(|e| ExportError::QuantizeError {
                    name: tensor.name.clone(),
                    reason: e.to_string(),
                })?
            } else {
                let mut padded = tensor.data.clone();
                padded.resize(tensor.data.len() + GROUP_SIZE - remainder, 0.0);
                quantize_q1_0_g128(&padded).map_err(|e| ExportError::QuantizeError {
                    name: tensor.name.clone(),
                    reason: e.to_string(),
                })?
            };
            Ok((bytes, TensorType::Q1_0G128))
        }

        ExportFormat::Int8PerChannel => {
            // Use the first shape dimension as the number of channels.
            let num_channels = tensor.shape.first().copied().unwrap_or(1).max(1);
            let int8 = quantize_per_channel(&tensor.data, num_channels).map_err(|e| {
                ExportError::QuantizeError {
                    name: tensor.name.clone(),
                    reason: e.to_string(),
                }
            })?;
            // Serialise: raw i8 data followed by f32 scales.
            let mut bytes: Vec<u8> = Vec::with_capacity(int8.data.len() + int8.scales.len() * 4);
            for &q in &int8.data {
                bytes.push(q as u8);
            }
            for &s in &int8.scales {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            // We store INT8 as F32 type in GGUF (custom packing) since there is
            // no INT8 type code in the current TensorType enum. The format field
            // in the GGUF metadata conveys the actual quantization used.
            Ok((bytes, TensorType::F32))
        }

        ExportFormat::TernaryG128 => {
            // Pad to a multiple of TERNARY_GROUP_SIZE inside quantize_tq2_0_g128
            // if necessary (it handles the padding internally and emits a tracing::warn).
            let bytes =
                crate::quantize_ternary::quantize_tq2_0_g128(&tensor.data).map_err(|e| {
                    ExportError::QuantizeError {
                        name: tensor.name.clone(),
                        reason: e.to_string(),
                    }
                })?;
            Ok((bytes, TensorType::TQ2_0_g128))
        }

        ExportFormat::FP8E4M3 => {
            // Pad to a multiple of QK_FP8 (32) if necessary.
            use oxibonsai_core::quant_fp8::{BlockFP8E4M3, QK_FP8};
            let remainder = tensor.data.len() % QK_FP8;
            let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
                std::borrow::Cow::Borrowed(&tensor.data)
            } else {
                let pad = QK_FP8 - remainder;
                let mut v = tensor.data.clone();
                v.resize(tensor.data.len() + pad, 0.0_f32);
                std::borrow::Cow::Owned(v)
            };
            let blocks =
                BlockFP8E4M3::quantize(&padded).map_err(|e| ExportError::QuantizeError {
                    name: tensor.name.clone(),
                    reason: e.to_string(),
                })?;
            // Serialize blocks to raw bytes via zero-copy pointer cast.
            // SAFETY: BlockFP8E4M3 is #[repr(C)] with compile-time size assert of 34 bytes.
            // The struct contains [u8; 32] + f16 (u16 layout), alignment is u8-compatible.
            let byte_len = blocks.len() * oxibonsai_core::quant_fp8::BLOCK_FP8_BYTES;
            let block_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
            Ok((block_bytes.to_vec(), TensorType::F8_E4M3))
        }

        ExportFormat::FP8E5M2 => {
            // Pad to a multiple of QK_FP8 (32) if necessary.
            use oxibonsai_core::quant_fp8::{BlockFP8E5M2, QK_FP8};
            let remainder = tensor.data.len() % QK_FP8;
            let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
                std::borrow::Cow::Borrowed(&tensor.data)
            } else {
                let pad = QK_FP8 - remainder;
                let mut v = tensor.data.clone();
                v.resize(tensor.data.len() + pad, 0.0_f32);
                std::borrow::Cow::Owned(v)
            };
            let blocks =
                BlockFP8E5M2::quantize(&padded).map_err(|e| ExportError::QuantizeError {
                    name: tensor.name.clone(),
                    reason: e.to_string(),
                })?;
            // SAFETY: same as FP8E4M3 above — BlockFP8E5M2 is #[repr(C)], 34 bytes.
            let byte_len = blocks.len() * oxibonsai_core::quant_fp8::BLOCK_FP8_BYTES;
            let block_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
            Ok((block_bytes.to_vec(), TensorType::F8_E5M2))
        }

        ExportFormat::Q4_0 => {
            // Pad to a multiple of QK_Q4_0 (32) if necessary.
            use oxibonsai_core::quant_std::{BlockQ4_0, BLOCK_Q4_0_BYTES, QK_Q4_0};
            let remainder = tensor.data.len() % QK_Q4_0;
            let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
                std::borrow::Cow::Borrowed(&tensor.data)
            } else {
                let pad = QK_Q4_0 - remainder;
                let mut v = tensor.data.clone();
                v.resize(tensor.data.len() + pad, 0.0_f32);
                std::borrow::Cow::Owned(v)
            };
            let blocks = BlockQ4_0::quantize(&padded).map_err(|e| ExportError::QuantizeError {
                name: tensor.name.clone(),
                reason: e.to_string(),
            })?;
            // SAFETY: BlockQ4_0 is #[repr(C)] with compile-time size assert of 18 bytes.
            // Contains f16 (u16 layout, 2 bytes) + [u8; 16]; alignment is 2 bytes.
            let byte_len = blocks.len() * BLOCK_Q4_0_BYTES;
            let block_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
            Ok((block_bytes.to_vec(), TensorType::Q4_0))
        }

        ExportFormat::Q8_0 => {
            // Pad to a multiple of QK_Q8_0 (32) if necessary.
            use oxibonsai_core::quant_std::{BlockQ8_0, BLOCK_Q8_0_BYTES, QK_Q8_0};
            let remainder = tensor.data.len() % QK_Q8_0;
            let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
                std::borrow::Cow::Borrowed(&tensor.data)
            } else {
                let pad = QK_Q8_0 - remainder;
                let mut v = tensor.data.clone();
                v.resize(tensor.data.len() + pad, 0.0_f32);
                std::borrow::Cow::Owned(v)
            };
            let blocks = BlockQ8_0::quantize(&padded).map_err(|e| ExportError::QuantizeError {
                name: tensor.name.clone(),
                reason: e.to_string(),
            })?;
            // SAFETY: BlockQ8_0 is #[repr(C)] with compile-time size assert of 34 bytes.
            // Contains f16 (2 bytes) + [i8; 32]; alignment is 2 bytes (from f16).
            let byte_len = blocks.len() * BLOCK_Q8_0_BYTES;
            let block_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
            Ok((block_bytes.to_vec(), TensorType::Q8_0))
        }

        ExportFormat::Q4K => {
            // Pad to a multiple of QK_K (256) if necessary.
            use oxibonsai_core::quant_k::{BlockQ4K, BLOCK_Q4_K_BYTES, QK_K};
            let remainder = tensor.data.len() % QK_K;
            let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
                std::borrow::Cow::Borrowed(&tensor.data)
            } else {
                let pad = QK_K - remainder;
                let mut v = tensor.data.clone();
                v.resize(tensor.data.len() + pad, 0.0_f32);
                std::borrow::Cow::Owned(v)
            };
            let blocks = BlockQ4K::quantize(&padded).map_err(|e| ExportError::QuantizeError {
                name: tensor.name.clone(),
                reason: e.to_string(),
            })?;
            // SAFETY: BlockQ4K is #[repr(C)] with compile-time size assert of 144 bytes.
            // Contains two f16 fields (d, dmin), [u8; 12] scales, [u8; 128] qs; alignment 2.
            let byte_len = blocks.len() * BLOCK_Q4_K_BYTES;
            let block_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
            Ok((block_bytes.to_vec(), TensorType::Q4_K))
        }

        ExportFormat::Q5K => {
            // Pad to a multiple of QK_K (256) if necessary.
            use oxibonsai_core::quant_k::QK_K;
            use oxibonsai_core::quant_k_ext::{BlockQ5K, BLOCK_Q5K_BYTES};
            let remainder = tensor.data.len() % QK_K;
            let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
                std::borrow::Cow::Borrowed(&tensor.data)
            } else {
                let pad = QK_K - remainder;
                let mut v = tensor.data.clone();
                v.resize(tensor.data.len() + pad, 0.0_f32);
                std::borrow::Cow::Owned(v)
            };
            let blocks = BlockQ5K::quantize(&padded).map_err(|e| ExportError::QuantizeError {
                name: tensor.name.clone(),
                reason: e.to_string(),
            })?;
            // SAFETY: BlockQ5K is #[repr(C)] with compile-time size assert of 176 bytes.
            // Contains two f16 fields, [u8; 12] scales, [u8; 32] qh, [u8; 128] qs; alignment 2.
            let byte_len = blocks.len() * BLOCK_Q5K_BYTES;
            let block_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
            Ok((block_bytes.to_vec(), TensorType::Q5_K))
        }

        ExportFormat::Q6K => {
            // Pad to a multiple of QK_K (256) if necessary.
            use oxibonsai_core::quant_k::QK_K;
            use oxibonsai_core::quant_k_ext::{BlockQ6K, BLOCK_Q6K_BYTES};
            let remainder = tensor.data.len() % QK_K;
            let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
                std::borrow::Cow::Borrowed(&tensor.data)
            } else {
                let pad = QK_K - remainder;
                let mut v = tensor.data.clone();
                v.resize(tensor.data.len() + pad, 0.0_f32);
                std::borrow::Cow::Owned(v)
            };
            let blocks = BlockQ6K::quantize(&padded).map_err(|e| ExportError::QuantizeError {
                name: tensor.name.clone(),
                reason: e.to_string(),
            })?;
            // SAFETY: BlockQ6K is #[repr(C)] with compile-time size assert of 210 bytes.
            // Contains [u8; 128] ql, [u8; 64] qh, [i8; 16] scales, f16 d; alignment 2.
            let byte_len = blocks.len() * BLOCK_Q6K_BYTES;
            let block_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) };
            Ok((block_bytes.to_vec(), TensorType::Q6_K))
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Export a list of weight tensors to a GGUF byte buffer.
///
/// # Arguments
///
/// * `tensors` – ordered list of named weight tensors.
/// * `config`  – export configuration (format, name, FP32 exceptions, …).
/// * `arch_metadata` – additional architecture-specific metadata KV pairs
///   (e.g. context length, number of layers) to embed in the file.
///
/// # Errors
///
/// Returns [`ExportError::Empty`] if `tensors` is empty.
/// Returns [`ExportError::QuantizeError`] if quantization of any tensor fails.
/// Returns [`ExportError::WriteError`] if the GGUF writer encounters an I/O error.
pub fn export_to_gguf(
    tensors: &[WeightTensor],
    config: &ExportConfig,
    arch_metadata: &[(String, MetadataWriteValue)],
) -> Result<Vec<u8>, ExportError> {
    if tensors.is_empty() {
        return Err(ExportError::Empty);
    }

    let mut writer = GgufWriter::new();

    // ── Standard metadata ──────────────────────────────────────────────────
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str(config.model_name.clone()),
    );
    writer.add_metadata(
        "general.version",
        MetadataWriteValue::Str(config.model_version.clone()),
    );
    if let Some(ref desc) = config.description {
        writer.add_metadata("general.description", MetadataWriteValue::Str(desc.clone()));
    }
    // Record the quantization format used.
    let quant_str = match config.format {
        ExportFormat::Float32 => "F32",
        ExportFormat::Q1_0G128 => "Q1_0G128",
        ExportFormat::Int8PerChannel => "INT8_PER_CHANNEL",
        ExportFormat::TernaryG128 => "TQ2_0_g128",
        ExportFormat::FP8E4M3 => "F8_E4M3",
        ExportFormat::FP8E5M2 => "F8_E5M2",
        ExportFormat::Q4_0 => "Q4_0",
        ExportFormat::Q8_0 => "Q8_0",
        ExportFormat::Q4K => "Q4_K",
        ExportFormat::Q5K => "Q5_K",
        ExportFormat::Q6K => "Q6_K",
    };
    writer.add_metadata(
        "general.quantization_version",
        MetadataWriteValue::Str(quant_str.to_string()),
    );

    // ── Architecture-specific metadata ─────────────────────────────────────
    for (key, val) in arch_metadata {
        writer.add_metadata(key, val.clone());
    }

    // ── Tensors ────────────────────────────────────────────────────────────
    for tensor in tensors {
        if tensor.data.is_empty() {
            // Skip empty tensors silently.
            continue;
        }

        let (bytes, tensor_type) = encode_tensor(tensor, config)?;

        // GGUF shape convention: outermost (slowest-varying) dimension first.
        let shape: Vec<u64> = if config.format == ExportFormat::Int8PerChannel
            && !should_keep_fp32(&tensor.name, config)
        {
            // For INT8 the serialized blob is flat (i8 data + scales), so we
            // report the element count as a 1-D shape to satisfy the writer's
            // size check for F32 type (4 bytes each).
            vec![(bytes.len() / 4) as u64]
        } else {
            tensor.shape.iter().map(|&d| d as u64).collect()
        };

        writer.add_tensor(TensorEntry {
            name: tensor.name.clone(),
            shape,
            tensor_type,
            data: bytes,
        });
    }

    writer
        .to_bytes()
        .map_err(|e| ExportError::WriteError(e.to_string()))
}

// ─── Size estimation ──────────────────────────────────────────────────────────

/// Estimate the total exported byte count without actually encoding anything.
///
/// This is an approximation — metadata and tensor info headers are not included.
pub fn estimate_export_size(tensors: &[WeightTensor], config: &ExportConfig) -> usize {
    tensors
        .iter()
        .map(|t| {
            if t.data.is_empty() {
                return 0;
            }
            let effective_format = if should_keep_fp32(&t.name, config) {
                ExportFormat::Float32
            } else {
                config.format
            };
            match effective_format {
                ExportFormat::Float32 => t.data.len() * 4,
                ExportFormat::Q1_0G128 => q1_0_g128_size_bytes(t.data.len()),
                ExportFormat::Int8PerChannel => {
                    let num_channels = t.shape.first().copied().unwrap_or(1).max(1);
                    // i8 data + f32 scales
                    t.data.len() + num_channels * 4
                }
                ExportFormat::TernaryG128 => {
                    crate::quantize_ternary::tq2_0_g128_size_bytes(t.data.len())
                }
                ExportFormat::FP8E4M3 | ExportFormat::FP8E5M2 => {
                    // 34 bytes per 32 weights (32 × u8 qs + 2 bytes FP16 scale).
                    // Use ceiling division so partially-filled final blocks count.
                    let num_blocks = t.data.len().div_ceil(oxibonsai_core::quant_fp8::QK_FP8);
                    num_blocks * oxibonsai_core::quant_fp8::BLOCK_FP8_BYTES
                }
                ExportFormat::Q4_0 => {
                    // 18 bytes per 32 weights (2-byte f16 scale + 16 bytes nibble-packed).
                    let num_blocks = t.data.len().div_ceil(oxibonsai_core::quant_std::QK_Q4_0);
                    num_blocks * oxibonsai_core::quant_std::BLOCK_Q4_0_BYTES
                }
                ExportFormat::Q8_0 => {
                    // 34 bytes per 32 weights (2-byte f16 scale + 32 i8 weights).
                    let num_blocks = t.data.len().div_ceil(oxibonsai_core::quant_std::QK_Q8_0);
                    num_blocks * oxibonsai_core::quant_std::BLOCK_Q8_0_BYTES
                }
                ExportFormat::Q4K => {
                    // 144 bytes per 256 weights.
                    let num_blocks = t.data.len().div_ceil(oxibonsai_core::quant_k::QK_K);
                    num_blocks * oxibonsai_core::quant_k::BLOCK_Q4_K_BYTES
                }
                ExportFormat::Q5K => {
                    // 176 bytes per 256 weights.
                    let num_blocks = t.data.len().div_ceil(oxibonsai_core::quant_k::QK_K);
                    num_blocks * oxibonsai_core::quant_k_ext::BLOCK_Q5K_BYTES
                }
                ExportFormat::Q6K => {
                    // 210 bytes per 256 weights.
                    let num_blocks = t.data.len().div_ceil(oxibonsai_core::quant_k::QK_K);
                    num_blocks * oxibonsai_core::quant_k_ext::BLOCK_Q6K_BYTES
                }
            }
        })
        .sum()
}

// ─── Export statistics ────────────────────────────────────────────────────────

/// Summary statistics produced after an export operation.
#[derive(Debug, Clone)]
pub struct ExportStats {
    /// Total number of tensors considered.
    pub num_tensors: usize,
    /// Tensors that were quantized.
    pub quantized_tensors: usize,
    /// Tensors kept in FP32.
    pub fp32_tensors: usize,
    /// Sum of original `f32` sizes in bytes.
    pub original_bytes: usize,
    /// Estimated exported size in bytes.
    pub exported_bytes: usize,
    /// `original_bytes / exported_bytes`.
    pub compression_ratio: f32,
}

/// Compute export statistics without performing the actual export.
pub fn export_stats(tensors: &[WeightTensor], config: &ExportConfig) -> ExportStats {
    let mut quantized = 0usize;
    let mut fp32_count = 0usize;
    let mut original_bytes = 0usize;

    for t in tensors {
        original_bytes += t.data.len() * 4;
        if should_keep_fp32(&t.name, config) || config.format == ExportFormat::Float32 {
            fp32_count += 1;
        } else {
            quantized += 1;
        }
    }

    let exported_bytes = estimate_export_size(tensors, config);
    let compression_ratio = if exported_bytes == 0 {
        1.0
    } else {
        original_bytes as f32 / exported_bytes as f32
    };

    ExportStats {
        num_tensors: tensors.len(),
        quantized_tensors: quantized,
        fp32_tensors: fp32_count,
        original_bytes,
        exported_bytes,
        compression_ratio,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── export_config_default_fp32_exceptions ─────────────────────────────

    #[test]
    fn test_export_config_default_fp32_exceptions() {
        let exceptions = ExportConfig::default_fp32_exceptions();
        assert!(exceptions.contains(&"token_embd.weight".to_string()));
        assert!(exceptions.contains(&"output_norm.weight".to_string()));
        assert!(exceptions.contains(&"output.weight".to_string()));
        assert_eq!(exceptions.len(), 3);
    }

    // ── weight_tensor_num_elements ────────────────────────────────────────

    #[test]
    fn test_weight_tensor_num_elements() {
        let t = WeightTensor::new("test", vec![0.0; 256], vec![16, 16]);
        assert_eq!(t.num_elements(), 256);
        assert_eq!(t.memory_bytes_f32(), 1024);
    }

    // ── estimate_export_size_fp32 ─────────────────────────────────────────

    #[test]
    fn test_estimate_export_size_fp32() {
        let tensors = vec![WeightTensor::new("w", vec![1.0; 256], vec![256])];
        let config = ExportConfig::new(ExportFormat::Float32, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(size, 256 * 4);
    }

    // ── estimate_export_size_q1_0 ─────────────────────────────────────────

    #[test]
    fn test_estimate_export_size_q1_0() {
        // 256 weights → 2 groups → 2 * 18 = 36 bytes
        let tensors = vec![WeightTensor::new("w", vec![1.0; 256], vec![256])];
        let config = ExportConfig::new(ExportFormat::Q1_0G128, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(
            size,
            2 * 18,
            "Q1_0 size for 256 weights should be {}",
            2 * 18
        );
    }

    // ── export_stats_compression_ratio ────────────────────────────────────

    #[test]
    fn test_export_stats_compression_ratio() {
        // 512 weights in Q1_0: 4 blocks × 18 = 72 bytes; original: 512*4 = 2048.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512])];
        let config = ExportConfig::new(ExportFormat::Q1_0G128, "m");
        let stats = export_stats(&tensors, &config);
        assert!(
            stats.compression_ratio > 1.0,
            "Q1_0 should compress better than FP32"
        );
        assert_eq!(stats.quantized_tensors, 1);
        assert_eq!(stats.fp32_tensors, 0);
    }

    // ── export_to_gguf_basic ──────────────────────────────────────────────

    #[test]
    fn test_export_to_gguf_basic() {
        // 128 weights → 1 Q1_0 block (18 bytes)
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            vec![1.0; 128],
            vec![128],
        )];
        let config =
            ExportConfig::new(ExportFormat::Q1_0G128, "test-model").with_description("unit test");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
        // Must start with GGUF magic: ASCII "GGUF" = bytes [0x47,0x47,0x55,0x46] → LE u32 0x46554747
        let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("slice"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
    }

    // ── export_fp32_tensor_unchanged ──────────────────────────────────────

    #[test]
    fn test_export_fp32_tensor_unchanged() {
        let data: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let tensors = vec![WeightTensor::new("w", data.clone(), vec![4])];
        let config = ExportConfig::new(ExportFormat::Float32, "m");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
        // The GGUF file should contain the f32 data somewhere in its body.
        // Find the 4-byte LE encoding of 3.0f32 = 0x40400000.
        let needle = 3.0_f32.to_le_bytes();
        let found = bytes.windows(4).any(|w| w == needle.as_slice());
        assert!(found, "float 3.0 should be present in the exported bytes");
    }

    // ── export_skips_empty_tensors ────────────────────────────────────────

    #[test]
    fn test_export_skips_empty_tensors() {
        let tensors = vec![
            WeightTensor::new("good", vec![1.0; 128], vec![128]),
            WeightTensor::new("empty", vec![], vec![0]),
        ];
        let config = ExportConfig::new(ExportFormat::Float32, "m");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
        // Tensor count in GGUF header (bytes 8..16 as u64) should be 1.
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().expect("slice"));
        assert_eq!(tensor_count, 1, "empty tensor should be skipped");
    }

    // ── TernaryG128 export ────────────────────────────────────────────────

    #[test]
    fn test_estimate_export_size_ternary_g128() {
        // 128 weights → 1 TQ2_0_g128 block → 34 bytes
        let tensors = vec![WeightTensor::new("w", vec![1.0; 128], vec![128])];
        let config = ExportConfig::new(ExportFormat::TernaryG128, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(
            size, 34,
            "128-weight tensor in TernaryG128 should be 34 bytes"
        );
    }

    #[test]
    fn test_estimate_export_size_ternary_g128_two_blocks() {
        // 256 weights → 2 TQ2_0_g128 blocks → 68 bytes
        let tensors = vec![WeightTensor::new("w", vec![1.0; 256], vec![256])];
        let config = ExportConfig::new(ExportFormat::TernaryG128, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(
            size, 68,
            "256-weight tensor in TernaryG128 should be 68 bytes"
        );
    }

    #[test]
    fn test_export_stats_ternary_g128_compression() {
        // 512 weights in TernaryG128: 4 blocks × 34 = 136 bytes; original: 512*4 = 2048.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512])];
        let config = ExportConfig::new(ExportFormat::TernaryG128, "m");
        let stats = export_stats(&tensors, &config);
        assert!(
            stats.compression_ratio > 1.0,
            "TernaryG128 should compress better than FP32"
        );
        assert_eq!(stats.quantized_tensors, 1);
        assert_eq!(stats.fp32_tensors, 0);
    }

    #[test]
    fn test_export_to_gguf_ternary_g128_basic() {
        // 128 weights → 1 TQ2_0_g128 block → valid GGUF with magic header.
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            vec![1.0; 128],
            vec![128],
        )];
        let config = ExportConfig::new(ExportFormat::TernaryG128, "ternary-model");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
        let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("slice"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
    }

    #[test]
    fn test_ternary_g128_fp32_exception_tensors_stay_fp32() {
        // output_norm.weight should stay F32 even under TernaryG128.
        let tensors = vec![
            WeightTensor::new("blk.0.attn_q.weight", vec![1.0; 128], vec![128]),
            WeightTensor::new("output_norm.weight", vec![1.0; 128], vec![128]),
        ];
        let config = ExportConfig::new(ExportFormat::TernaryG128, "m")
            .with_fp32_layers(vec!["output_norm.weight".to_string()]);
        let stats = export_stats(&tensors, &config);
        assert_eq!(stats.fp32_tensors, 1, "output_norm.weight should stay FP32");
        assert_eq!(
            stats.quantized_tensors, 1,
            "attn_q.weight should be ternary-quantized"
        );
    }

    // ── FP8E4M3 export ────────────────────────────────────────────────────────

    #[test]
    fn test_export_fp8_e4m3_roundtrip() {
        // 128 weights (4 FP8 blocks × 32 weights each) → 4 × 34 = 136 bytes of FP8 data.
        // The GGUF tensor data section must contain exactly that many bytes.
        let n_weights = 128usize;
        let n_blocks = n_weights / oxibonsai_core::quant_fp8::QK_FP8;
        let expected_bytes = n_blocks * oxibonsai_core::quant_fp8::BLOCK_FP8_BYTES;
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            vec![1.0; n_weights],
            vec![n_weights],
        )];
        let config = ExportConfig::new(ExportFormat::FP8E4M3, "fp8-e4m3-model");
        let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("FP8E4M3 export");
        // Verify GGUF magic.
        let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic slice"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
        // The raw tensor bytes must appear somewhere in the output; their length
        // is 4 × 34 = 136 bytes. We verify the total file size is at least that.
        assert!(
            gguf_bytes.len() >= expected_bytes,
            "GGUF file too small: {} < {}",
            gguf_bytes.len(),
            expected_bytes,
        );
    }

    #[test]
    fn test_export_fp8_e5m2_roundtrip() {
        // 64 weights (2 FP8 blocks × 32 weights each) → 2 × 34 = 68 bytes of FP8 data.
        let n_weights = 64usize;
        let n_blocks = n_weights / oxibonsai_core::quant_fp8::QK_FP8;
        let expected_bytes = n_blocks * oxibonsai_core::quant_fp8::BLOCK_FP8_BYTES;
        let tensors = vec![WeightTensor::new(
            "blk.0.ffn_gate.weight",
            vec![2.0; n_weights],
            vec![n_weights],
        )];
        let config = ExportConfig::new(ExportFormat::FP8E5M2, "fp8-e5m2-model");
        let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("FP8E5M2 export");
        let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic slice"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");
        assert!(
            gguf_bytes.len() >= expected_bytes,
            "GGUF file too small: {} < {}",
            gguf_bytes.len(),
            expected_bytes,
        );
    }

    #[test]
    fn test_export_fp8_size_estimate() {
        // 32 weights → 1 FP8 block → 34 bytes.
        let tensors_32 = vec![WeightTensor::new("w", vec![1.0; 32], vec![32])];
        let config_e4m3 = ExportConfig::new(ExportFormat::FP8E4M3, "m");
        let config_e5m2 = ExportConfig::new(ExportFormat::FP8E5M2, "m");
        assert_eq!(
            estimate_export_size(&tensors_32, &config_e4m3),
            34,
            "32 weights in FP8E4M3 → 1 block → 34 bytes"
        );
        assert_eq!(
            estimate_export_size(&tensors_32, &config_e5m2),
            34,
            "32 weights in FP8E5M2 → 1 block → 34 bytes"
        );

        // 256 weights → 8 blocks → 272 bytes.
        let tensors_256 = vec![WeightTensor::new("w", vec![1.0; 256], vec![256])];
        assert_eq!(
            estimate_export_size(&tensors_256, &config_e4m3),
            8 * 34,
            "256 weights → 8 blocks → 272 bytes"
        );

        // Verify compression ratio > 1 (FP8 is 34/32 bytes/weight ≈ 1.0625 vs 4.0 for FP32).
        let stats = export_stats(&tensors_256, &config_e4m3);
        assert!(
            stats.compression_ratio > 1.0,
            "FP8E4M3 should compress better than FP32"
        );
        // Expected ratio: 256*4 / (8*34) = 1024 / 272 ≈ 3.76
        assert!(
            stats.compression_ratio > 3.0,
            "FP8E4M3 compression ratio should be > 3.0, got {}",
            stats.compression_ratio
        );
        assert_eq!(stats.quantized_tensors, 1);
        assert_eq!(stats.fp32_tensors, 0);
    }

    #[test]
    fn test_fp8_fp32_exception_tensors_stay_fp32() {
        // output_norm.weight should stay F32 even under FP8E4M3 and FP8E5M2.
        let tensors = vec![
            WeightTensor::new("blk.0.attn_q.weight", vec![1.0; 64], vec![64]),
            WeightTensor::new("output_norm.weight", vec![1.0; 64], vec![64]),
        ];
        let config = ExportConfig::new(ExportFormat::FP8E4M3, "m")
            .with_fp32_layers(vec!["output_norm.weight".to_string()]);
        let stats = export_stats(&tensors, &config);
        assert_eq!(stats.fp32_tensors, 1, "output_norm.weight should stay FP32");
        assert_eq!(
            stats.quantized_tensors, 1,
            "attn_q.weight should be FP8-quantized"
        );
    }

    // ── Q4_0 export tests ─────────────────────────────────────────────────────

    #[test]
    fn test_export_q4_0_roundtrip() {
        // 64 floats → 2 Q4_0 blocks × 18 bytes = 36 bytes of quantized data.
        use oxibonsai_core::quant_std::{BlockQ4_0, BLOCK_Q4_0_BYTES, QK_Q4_0};
        let n = 64usize;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 8.0).collect();
        let config = ExportConfig::new(ExportFormat::Q4_0, "q4-0-model");
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            input.clone(),
            vec![n],
        )];
        let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4_0 export");

        // Validate GGUF magic.
        let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

        // Validate exported byte count covers at least 2 blocks × 18 bytes.
        let expected_raw = (n / QK_Q4_0) * BLOCK_Q4_0_BYTES;
        assert!(
            gguf_bytes.len() >= expected_raw,
            "GGUF file ({} bytes) must cover at least {} raw data bytes",
            gguf_bytes.len(),
            expected_raw,
        );

        // Verify roundtrip error is acceptable (Q4_0 is 4-bit, error < 10% of max range).
        let blocks = BlockQ4_0::quantize(&input).expect("Q4_0 quantize");
        assert_eq!(blocks.len(), n / QK_Q4_0, "block count matches");
        let mut output = vec![0.0f32; n];
        BlockQ4_0::dequant(&blocks, &mut output).expect("Q4_0 dequant");
        let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_err = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let threshold = max_range * 0.15;
        assert!(
            max_err <= threshold,
            "Q4_0 roundtrip max error {max_err} > threshold {threshold} (max_range={max_range})"
        );
    }

    #[test]
    fn test_export_q8_0_roundtrip() {
        // 64 floats → 2 Q8_0 blocks × 34 bytes = 68 bytes of quantized data.
        use oxibonsai_core::quant_std::{BlockQ8_0, BLOCK_Q8_0_BYTES, QK_Q8_0};
        let n = 64usize;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 16.0).collect();
        let config = ExportConfig::new(ExportFormat::Q8_0, "q8-0-model");
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            input.clone(),
            vec![n],
        )];
        let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q8_0 export");

        let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

        let expected_raw = (n / QK_Q8_0) * BLOCK_Q8_0_BYTES;
        assert!(
            gguf_bytes.len() >= expected_raw,
            "GGUF file ({} bytes) must cover at least {} raw Q8_0 data bytes",
            gguf_bytes.len(),
            expected_raw,
        );

        // Verify roundtrip error is < 1% of max range (Q8_0 is high fidelity).
        let blocks = BlockQ8_0::quantize(&input).expect("Q8_0 quantize");
        assert_eq!(blocks.len(), n / QK_Q8_0, "block count matches");
        let mut output = vec![0.0f32; n];
        BlockQ8_0::dequant(&blocks, &mut output).expect("Q8_0 dequant");
        let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_err = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let threshold = max_range * 0.01;
        assert!(
            max_err <= threshold,
            "Q8_0 roundtrip max error {max_err} > threshold {threshold} (max_range={max_range})"
        );
    }

    // ── Q4K export tests ──────────────────────────────────────────────────────

    #[test]
    fn test_export_q4k_roundtrip() {
        // 512 floats → 2 Q4_K super-blocks × 144 bytes = 288 bytes of quantized data.
        use oxibonsai_core::quant_k::{BlockQ4K, BLOCK_Q4_K_BYTES, QK_K};
        let n = 512usize;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1 - 25.6).sin()).collect();
        let config = ExportConfig::new(ExportFormat::Q4K, "q4k-model");
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            input.clone(),
            vec![n],
        )];
        let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4K export");

        let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

        let expected_raw = (n / QK_K) * BLOCK_Q4_K_BYTES;
        assert!(
            gguf_bytes.len() >= expected_raw,
            "GGUF file ({} bytes) must cover at least {} raw Q4K data bytes",
            gguf_bytes.len(),
            expected_raw,
        );

        // Verify roundtrip error < 5% of max range.
        let blocks = BlockQ4K::quantize(&input).expect("Q4K quantize");
        assert_eq!(blocks.len(), n / QK_K, "Q4K block count matches");
        let mut output = vec![0.0f32; n];
        BlockQ4K::dequant(&blocks, &mut output).expect("Q4K dequant");
        let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_err = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let threshold = (max_range * 0.08).max(0.1);
        assert!(
            max_err <= threshold,
            "Q4K roundtrip max error {max_err} > threshold {threshold}"
        );
    }

    // ── Q5K export tests ──────────────────────────────────────────────────────

    #[test]
    fn test_export_q5k_roundtrip() {
        // 512 floats → 2 Q5_K super-blocks × 176 bytes = 352 bytes of quantized data.
        use oxibonsai_core::quant_k::QK_K;
        use oxibonsai_core::quant_k_ext::{BlockQ5K, BLOCK_Q5K_BYTES};
        let n = 512usize;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07 - 17.9).cos()).collect();
        let config = ExportConfig::new(ExportFormat::Q5K, "q5k-model");
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            input.clone(),
            vec![n],
        )];
        let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q5K export");

        let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

        let expected_raw = (n / QK_K) * BLOCK_Q5K_BYTES;
        assert!(
            gguf_bytes.len() >= expected_raw,
            "GGUF file ({} bytes) must cover at least {} raw Q5K data bytes",
            gguf_bytes.len(),
            expected_raw,
        );

        // Verify roundtrip error < 5% of max range.
        let blocks = BlockQ5K::quantize(&input).expect("Q5K quantize");
        assert_eq!(blocks.len(), n / QK_K, "Q5K block count matches");
        let mut output = vec![0.0f32; n];
        BlockQ5K::dequant(&blocks, &mut output).expect("Q5K dequant");
        let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_err = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let threshold = (max_range * 0.08).max(0.1);
        assert!(
            max_err <= threshold,
            "Q5K roundtrip max error {max_err} > threshold {threshold}"
        );
    }

    // ── Q6K export tests ──────────────────────────────────────────────────────

    #[test]
    fn test_export_q6k_roundtrip() {
        // 512 floats → 2 Q6_K super-blocks × 210 bytes = 420 bytes of quantized data.
        use oxibonsai_core::quant_k::QK_K;
        use oxibonsai_core::quant_k_ext::{BlockQ6K, BLOCK_Q6K_BYTES};
        let n = 512usize;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 12.8).collect();
        let config = ExportConfig::new(ExportFormat::Q6K, "q6k-model");
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            input.clone(),
            vec![n],
        )];
        let gguf_bytes = export_to_gguf(&tensors, &config, &[]).expect("Q6K export");

        let magic = u32::from_le_bytes(gguf_bytes[0..4].try_into().expect("magic"));
        assert_eq!(magic, 0x4655_4747, "expected GGUF magic");

        let expected_raw = (n / QK_K) * BLOCK_Q6K_BYTES;
        assert!(
            gguf_bytes.len() >= expected_raw,
            "GGUF file ({} bytes) must cover at least {} raw Q6K data bytes",
            gguf_bytes.len(),
            expected_raw,
        );

        // Verify roundtrip error < 3% of max range (Q6K is high fidelity).
        let blocks = BlockQ6K::quantize(&input).expect("Q6K quantize");
        assert_eq!(blocks.len(), n / QK_K, "Q6K block count matches");
        let mut output = vec![0.0f32; n];
        BlockQ6K::dequant(&blocks, &mut output).expect("Q6K dequant");
        let max_range = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_err = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let threshold = (max_range * 0.05).max(0.1);
        assert!(
            max_err <= threshold,
            "Q6K roundtrip max error {max_err} > threshold {threshold}"
        );
    }

    // ── Size estimation tests ─────────────────────────────────────────────────

    #[test]
    fn test_estimate_export_size_q4_0() {
        // 64 elements → 2 blocks × 18 bytes = 36 bytes.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 64], vec![64])];
        let config = ExportConfig::new(ExportFormat::Q4_0, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(size, 2 * 18, "Q4_0: 64 weights → 2 blocks → 36 bytes");
    }

    #[test]
    fn test_estimate_export_size_q8_0() {
        // 64 elements → 2 blocks × 34 bytes = 68 bytes.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 64], vec![64])];
        let config = ExportConfig::new(ExportFormat::Q8_0, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(size, 2 * 34, "Q8_0: 64 weights → 2 blocks → 68 bytes");
    }

    #[test]
    fn test_estimate_export_size_q4k() {
        // 512 elements → 2 super-blocks × 144 bytes = 288 bytes.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512])];
        let config = ExportConfig::new(ExportFormat::Q4K, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(
            size,
            2 * 144,
            "Q4K: 512 weights → 2 super-blocks → 288 bytes"
        );
    }

    #[test]
    fn test_estimate_export_size_q5k() {
        // 512 elements → 2 super-blocks × 176 bytes = 352 bytes.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512])];
        let config = ExportConfig::new(ExportFormat::Q5K, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(
            size,
            2 * 176,
            "Q5K: 512 weights → 2 super-blocks → 352 bytes"
        );
    }

    #[test]
    fn test_estimate_export_size_q6k() {
        // 512 elements → 2 super-blocks × 210 bytes = 420 bytes.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 512], vec![512])];
        let config = ExportConfig::new(ExportFormat::Q6K, "m");
        let size = estimate_export_size(&tensors, &config);
        assert_eq!(
            size,
            2 * 210,
            "Q6K: 512 weights → 2 super-blocks → 420 bytes"
        );
    }

    // ── GGUF type name tests ──────────────────────────────────────────────────

    #[test]
    fn test_export_format_type_name_q4_0() {
        // Verify that the quant_str for Q4_0 matches the expected GGUF string.
        // We check by inspecting the metadata written into the GGUF file.
        let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 64], vec![64])];
        let config = ExportConfig::new(ExportFormat::Q4_0, "m");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4_0 export");
        // "Q4_0" string should appear somewhere in the metadata section.
        let needle = b"Q4_0";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "GGUF metadata should contain \"Q4_0\" quantization string"
        );
    }

    #[test]
    fn test_export_format_type_name_q4k() {
        // Verify that the quant_str for Q4K emits "Q4_K" in the GGUF metadata.
        let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256])];
        let config = ExportConfig::new(ExportFormat::Q4K, "m");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q4K export");
        let needle = b"Q4_K";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "GGUF metadata should contain \"Q4_K\" quantization string"
        );
    }

    #[test]
    fn test_export_format_type_name_q5k() {
        // Verify that Q5K emits "Q5_K" in the GGUF metadata.
        let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256])];
        let config = ExportConfig::new(ExportFormat::Q5K, "m");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q5K export");
        let needle = b"Q5_K";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "GGUF metadata should contain \"Q5_K\" quantization string"
        );
    }

    #[test]
    fn test_export_format_type_name_q6k() {
        // Verify that Q6K emits "Q6_K" in the GGUF metadata.
        let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 256], vec![256])];
        let config = ExportConfig::new(ExportFormat::Q6K, "m");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q6K export");
        let needle = b"Q6_K";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "GGUF metadata should contain \"Q6_K\" quantization string"
        );
    }

    #[test]
    fn test_export_format_type_name_q8_0() {
        // Verify that Q8_0 emits "Q8_0" in the GGUF metadata.
        let tensors = vec![WeightTensor::new("blk.0.w", vec![1.0; 64], vec![64])];
        let config = ExportConfig::new(ExportFormat::Q8_0, "m");
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("Q8_0 export");
        let needle = b"Q8_0";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "GGUF metadata should contain \"Q8_0\" quantization string"
        );
    }

    // ── Compression sanity tests ──────────────────────────────────────────────

    #[test]
    fn test_q4_0_produces_smaller_output_than_float32() {
        // 32 elements: Q4_0 = 1 block × 18 bytes; Float32 = 32 × 4 = 128 bytes.
        let tensors = vec![WeightTensor::new("w", vec![1.0; 32], vec![32])];
        let config_q4 = ExportConfig::new(ExportFormat::Q4_0, "m");
        let config_f32 = ExportConfig::new(ExportFormat::Float32, "m");
        let q4_size = estimate_export_size(&tensors, &config_q4);
        let f32_size = estimate_export_size(&tensors, &config_f32);
        assert_eq!(q4_size, 18, "Q4_0 32 weights = 18 bytes");
        assert_eq!(f32_size, 128, "Float32 32 weights = 128 bytes");
        assert!(
            q4_size < f32_size,
            "Q4_0 ({q4_size} bytes) must be smaller than Float32 ({f32_size} bytes)"
        );
    }

    #[test]
    fn test_q8_0_compression_vs_float32() {
        // Q8_0: 32 weights → 34 bytes (8.5 bits/weight vs 32 bits/weight).
        let tensors = vec![WeightTensor::new("w", vec![0.5f32; 32], vec![32])];
        let config_q8 = ExportConfig::new(ExportFormat::Q8_0, "m");
        let config_f32 = ExportConfig::new(ExportFormat::Float32, "m");
        let q8_size = estimate_export_size(&tensors, &config_q8);
        let f32_size = estimate_export_size(&tensors, &config_f32);
        assert!(
            q8_size < f32_size,
            "Q8_0 ({q8_size} bytes) must be smaller than Float32 ({f32_size} bytes)"
        );
    }

    #[test]
    fn test_new_formats_fp32_exception_respected() {
        // output_norm.weight must stay FP32 regardless of Q4_0 / Q8_0 / K-quant format.
        let tensors = vec![
            WeightTensor::new("blk.0.attn_q.weight", vec![1.0; 64], vec![64]),
            WeightTensor::new("output_norm.weight", vec![1.0; 64], vec![64]),
        ];
        let fp32_exceptions = vec!["output_norm.weight".to_string()];
        for fmt in &[
            ExportFormat::Q4_0,
            ExportFormat::Q8_0,
            ExportFormat::Q4K,
            ExportFormat::Q5K,
            ExportFormat::Q6K,
        ] {
            let config = ExportConfig::new(*fmt, "m").with_fp32_layers(fp32_exceptions.clone());
            let stats = export_stats(&tensors, &config);
            assert_eq!(
                stats.fp32_tensors, 1,
                "output_norm.weight must stay FP32 for format {fmt:?}"
            );
            assert_eq!(
                stats.quantized_tensors, 1,
                "attn_q.weight must be quantized for format {fmt:?}"
            );
        }
    }
}
