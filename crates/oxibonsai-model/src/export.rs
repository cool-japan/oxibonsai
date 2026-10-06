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

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use oxibonsai_core::gguf::metadata::{MetadataStore, MetadataValue};
use oxibonsai_core::gguf::tensor_info::keys;
use oxibonsai_core::gguf::writer::{
    GgufWriter, MetadataWriteValue, TensorEntry, TensorProducer, TensorSource, TensorStream,
    TensorType,
};

use crate::convert::meta::{
    arch_metadata_keys, filter_carried_metadata, ggml_file_type, write_arch_metadata,
    write_general_metadata, ArchMetadata, GeneralMetadata,
};
use crate::convert::qwen35::{
    is_never_quantized_qwen35, qwen35_metadata_keys, write_hadamard_metadata,
    write_qwen35_metadata, HadamardError, HadamardSpec, Qwen35Error, Qwen35Metadata,
};
use crate::convert::tokenizer_meta::TokenizerMetadata;
use crate::quantize::ScaleRule;

// ─── Export format ────────────────────────────────────────────────────────────

/// The target quantization format for an export operation.
///
/// A variant names a *wire format*, not a policy: which tensors actually get
/// it is decided per tensor by [`keep_fp32_by_kind`] plus the
/// [`ExportConfig`]'s exception list and allowlist. The variant docs used to
/// state their own, mutually inconsistent rules for `token_embd` / `output` /
/// norms ("Only RMS-norm weights remain FP32", "unlike Q1_0G128 which keeps
/// them FP16"), none of which matched what the encoder did (CQ-18).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    /// Keep weights as IEEE 754 single precision floats.
    Float32,
    /// Quantize to Q1\_0\_g128 (1-bit sign + FP16 scale per 128-element group).
    Q1_0G128,
    /// Quantize to INT8 per output channel.
    ///
    /// **Export-only / not loadable.** There is no GGUF tensor-type id for
    /// packed per-channel INT8 in the current [`TensorType`] enum and no
    /// corresponding loader arm anywhere in this workspace. Attempting to
    /// use this format with [`export_to_gguf`] returns
    /// [`ExportError::NoLoaderForFormat`] rather than silently writing bytes
    /// tagged `TensorType::F32` that a standard loader would misinterpret as
    /// raw floats. Use [`crate::quantize_int8::quantize_per_channel`]
    /// directly if you only need the in-memory quantized representation
    /// (e.g. for offline error analysis), not a round-trippable GGUF file.
    Int8PerChannel,
    /// **Legacy** ternary quantization: {-1, 0, +1} weights packed as
    /// OxiBonsai's `TQ2_0_g128` (ggml id 42, 34 B / 128 weights, `qs` first
    /// and `d` last).
    ///
    /// This byte order exists in no other ggml consumer: mainline `Q2_0` (id
    /// 42) is 64 weights in 18 bytes with `d` first, and PrismML's 128-wide
    /// block is `PQ2_0` (id 142), also `d` first. `llama.cpp` therefore
    /// refuses a file written in this format — it validates each tensor
    /// offset against the running padded sum and hard-errors
    /// (`ggml/src/gguf.cpp:780`), and the Prism fork prints a dedicated hint
    /// naming exactly these files.
    ///
    /// It is kept because the shipped `models/*.gguf` are in it and every
    /// OxiBonsai kernel reads it. **New files should use
    /// [`ExportFormat::PQ2_0`]**, which carries the same ternary data in the
    /// interoperable layout.
    TernaryG128,
    /// PrismML `PQ2_0` (ggml id 142): ternary {-1, 0, +1}, 128 weights in
    /// 34 bytes, FP16 scale **first**.
    ///
    /// The interoperable spelling of group-128 ternary and the format the
    /// Bonsai 2 27 B files use. Byte-for-byte compatible with the PrismML
    /// `llama.cpp` fork.
    PQ2_0,
    /// PrismML `PTQ1_0` (ggml id 143): ternary packed base-3, 128 weights in
    /// 28 bytes (`qs[24]` + `qh[2]` + FP16 scale **last**).
    ///
    /// ~1.75 bits/weight — the densest Bonsai 2 format (5.95 GB for the 27 B
    /// against 7.21 GB for `PQ2_0`).
    PTQ1_0,
    /// Mainline `block_q2_0` (ggml id 42): ternary, **64** weights in 18
    /// bytes, FP16 scale first.
    ///
    /// The upstream group-64 spelling of id 42, as distinct from
    /// [`ExportFormat::TernaryG128`]'s group-128 reuse of the same id.
    Q2_0G64,
    /// FP8 E4M3FN per-block quantization: 32 weights × 1 byte + FP16 scale (34 B / 32 weights).
    ///
    /// Uses the E4M3FN format (bias=7, no infinity, NaN at 0x7f/0xff). Provides
    /// approximately 8.5 bits per weight (34 bytes × 8 bits ÷ 32 weights). Maps
    /// to GGUF type ID 43 (PrismML extension).
    FP8E4M3,
    /// FP8 E5M2 per-block quantization: 32 weights × 1 byte + FP16 scale (34 B / 32 weights).
    ///
    /// Uses the E5M2 format (bias=15, has infinity). Provides higher dynamic range
    /// than E4M3 at the cost of mantissa precision. Maps to GGUF type ID 44
    /// (PrismML extension).
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
    /// Q2_K quantization: 2-bit K-quant, 256 per super-block, 4-bit packed
    /// sub-scale/min pairs (84 bytes/256 weights).
    ///
    /// Maps to GGML type ID 10. Each super-block stores 16 bytes of packed
    /// 4-bit sub-block scale/min pairs, 64 bytes of packed 2-bit codes, and
    /// FP16 `d`/`dmin` super-block scales. Backed by `encode_quantized_tensor`'s
    /// `TensorType::Q2_K` arm.
    Q2K,
    /// Q3_K quantization: 3-bit K-quant, 256 per super-block, 6-bit packed
    /// sub-scales plus a high-bit mask (110 bytes/256 weights).
    ///
    /// Maps to GGML type ID 11. Each super-block stores a 32-byte high-bit
    /// mask, 64 bytes of packed 2-bit low codes, 12 bytes of packed 6-bit
    /// sub-block scales, and one FP16 super-block scale.
    Q3K,
    /// Q8_K quantization: 8-bit K-quant, 256 per super-block, f32 scale plus
    /// per-sub-block `i16` sums (292 bytes/256 weights).
    ///
    /// Maps to GGML type ID 15. Each super-block stores an f32 scale, 256
    /// `i8` codes, and 16 `i16` sub-block sums (used by K-quant dot-product
    /// kernels to skip a re-sum pass) — the highest-fidelity K-quant format,
    /// intended as an intermediate/activation format more than a shipped
    /// weight format, but exposed here for completeness (design parity with
    /// `Q2K`/`Q3K`/`Q4K`/`Q5K`/`Q6K`).
    Q8K,
}

impl ExportFormat {
    /// The GGUF tensor type this format writes, or `None` for
    /// [`ExportFormat::Int8PerChannel`], which has no GGUF tensor-type id.
    pub fn tensor_type(self) -> Option<TensorType> {
        Some(match self {
            Self::Float32 => TensorType::F32,
            Self::Q1_0G128 => TensorType::Q1_0G128,
            Self::Int8PerChannel => return None,
            Self::TernaryG128 => TensorType::TQ2_0_g128,
            Self::PQ2_0 => TensorType::PQ2_0,
            Self::PTQ1_0 => TensorType::PTQ1_0,
            Self::Q2_0G64 => TensorType::Q2_0G64,
            Self::FP8E4M3 => TensorType::F8_E4M3,
            Self::FP8E5M2 => TensorType::F8_E5M2,
            Self::Q4_0 => TensorType::Q4_0,
            Self::Q8_0 => TensorType::Q8_0,
            Self::Q4K => TensorType::Q4_K,
            Self::Q5K => TensorType::Q5_K,
            Self::Q6K => TensorType::Q6_K,
            Self::Q2K => TensorType::Q2_K,
            Self::Q3K => TensorType::Q3_K,
            Self::Q8K => TensorType::Q8_K,
        })
    }

    /// Short label written to the non-normative
    /// [`crate::convert::meta::OXIBONSAI_QUANT_FORMAT`] key.
    pub fn label(self) -> &'static str {
        match self {
            Self::Float32 => "F32",
            Self::Q1_0G128 => "Q1_0_G128",
            Self::Int8PerChannel => "INT8_PER_CHANNEL",
            Self::TernaryG128 => "TQ2_0_G128",
            Self::PQ2_0 => "PQ2_0",
            Self::PTQ1_0 => "PTQ1_0",
            Self::Q2_0G64 => "Q2_0_G64",
            Self::FP8E4M3 => "F8_E4M3",
            Self::FP8E5M2 => "F8_E5M2",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q8K => "Q8_K",
        }
    }

    /// Whether this format encodes ternary `{-1, 0, +1}` weights and so
    /// honours [`ScaleRule`].
    pub fn is_ternary(self) -> bool {
        matches!(
            self,
            Self::TernaryG128 | Self::PQ2_0 | Self::PTQ1_0 | Self::Q2_0G64
        )
    }

    /// Every format, for exhaustive tests.
    pub const ALL: [ExportFormat; 17] = [
        Self::Float32,
        Self::Q1_0G128,
        Self::Int8PerChannel,
        Self::TernaryG128,
        Self::PQ2_0,
        Self::PTQ1_0,
        Self::Q2_0G64,
        Self::FP8E4M3,
        Self::FP8E5M2,
        Self::Q4_0,
        Self::Q8_0,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q2K,
        Self::Q3K,
        Self::Q8K,
    ];
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
    ///
    /// This is an **additional** user list; it does not replace the
    /// structural rule in [`keep_fp32_by_kind`], which keeps every 1-D
    /// tensor and every `*norm.weight` in FP32 unconditionally.
    pub fp32_layers: Vec<String>,
    /// `general.architecture` and the prefix of the `<arch>.*` key block.
    ///
    /// `None` means the file gets no architecture block, which is only
    /// acceptable for a non-model artifact; [`export_to_gguf`] warns.
    pub architecture: Option<String>,
    /// The `<arch>.*` hyper-parameter block (CQ-01).
    pub arch_metadata: Option<ArchMetadata>,
    /// The `tokenizer.ggml.*` block (CQ-08).
    pub tokenizer: Option<TokenizerMetadata>,
    /// The hybrid-only `qwen35.*` keys, for a Bonsai 2 export (CQ-06).
    pub qwen35: Option<Qwen35Metadata>,
    /// The `prism.hadamard.*` contract, for a folded export (CQ-06).
    pub hadamard: Option<HadamardSpec>,
    /// Metadata copied verbatim from a source GGUF.
    ///
    /// Populated by [`ExportConfig::with_source_metadata`]; keys the writer
    /// owns (`general.alignment`, `general.file_type`,
    /// `general.quantization_version`) are filtered out there.
    pub carried_metadata: Vec<(String, MetadataWriteValue)>,
    /// Which scale estimator the ternary / 1-bit encoders use (CQ-03).
    pub scale_rule: ScaleRule,
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
            architecture: None,
            arch_metadata: None,
            tokenizer: None,
            qwen35: None,
            hadamard: None,
            carried_metadata: Vec::new(),
            scale_rule: ScaleRule::AbsMax,
        }
    }

    /// Set `general.architecture` and the `<arch>.*` block.
    pub fn with_architecture(mut self, architecture: &str, meta: ArchMetadata) -> Self {
        self.architecture = Some(architecture.to_string());
        self.arch_metadata = Some(meta);
        self
    }

    /// Attach the tokenizer to embed (CQ-08).
    pub fn with_tokenizer(mut self, tokenizer: TokenizerMetadata) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Attach the hybrid `qwen35.*` keys.
    pub fn with_qwen35(mut self, meta: Qwen35Metadata) -> Self {
        self.qwen35 = Some(meta);
        self
    }

    /// Attach the `prism.hadamard.*` contract.
    pub fn with_hadamard(mut self, spec: HadamardSpec) -> Self {
        self.hadamard = Some(spec);
        self
    }

    /// Select the ternary / 1-bit scale estimator (CQ-03).
    ///
    /// Use [`ScaleRule::AbsMean`] whenever the source is continuous (raw
    /// HuggingFace weights); the default [`ScaleRule::AbsMax`] is correct and
    /// lossless for a source that is already ternary.
    pub fn with_scale_rule(mut self, rule: ScaleRule) -> Self {
        self.scale_rule = rule;
        self
    }

    /// Carry the architecture, tokenizer and provenance metadata of a source
    /// GGUF into the output (the `quantize` path).
    ///
    /// This is what makes `oxibonsai quantize` produce a *loadable* file
    /// (CQ-01): the source already holds the complete `<arch>.*` and
    /// `tokenizer.ggml.*` blocks, so re-deriving them would be both redundant
    /// and lossy — `oxibonsai-model` cannot even build a tokenizer from
    /// scratch.
    ///
    /// Three key groups are deliberately **not** copied:
    ///
    /// * `general.alignment` — [`GgufWriter`] injects its own; a second,
    ///   possibly contradictory value makes the reader compute the wrong data
    ///   offset.
    /// * `general.file_type` and `general.quantization_version` — both
    ///   describe the *output* encoding and are recomputed from
    ///   [`ExportConfig::format`].
    /// * `oxibonsai.*` — provenance of the previous export.
    ///
    /// # Errors
    ///
    /// Returns [`ExportError::MissingArchitecture`] when the source has no
    /// `general.architecture`, because the architecture is the prefix of
    /// every hyper-parameter key: without it the output would silently carry
    /// no usable architecture block at all.
    pub fn with_source_metadata(mut self, metadata: &MetadataStore) -> Result<Self, ExportError> {
        let architecture = metadata
            .get_string(keys::GENERAL_ARCHITECTURE)
            .map_err(|_| ExportError::MissingArchitecture)?
            .to_string();

        // `general.version` / `general.description` are on the writer-owned
        // list (otherwise they would be written twice and the reader rejects
        // a duplicate key), so lift the source's values into the config here
        // instead — otherwise a re-quantize would overwrite the real
        // `general.version` with the default "1.0.0" and drop the
        // description entirely. The real 27 B files carry `v5` here.
        if let Ok(version) = metadata.get_string("general.version") {
            self.model_version = version.to_string();
        }
        if let Ok(description) = metadata.get_string("general.description") {
            self.description = Some(description.to_string());
        }

        let mut carried: Vec<(String, MetadataWriteValue)> = Vec::new();
        let mut names: Vec<&String> = metadata.iter().map(|(k, _)| k).collect();
        names.sort();
        for key in names {
            if is_writer_owned_key(key) {
                continue;
            }
            let Some(value) = metadata.get(key) else {
                continue;
            };
            match convert_metadata_value(value) {
                Some(v) => carried.push((key.clone(), v)),
                None => tracing::warn!(
                    key = key.as_str(),
                    "source metadata value has no writable representation; dropping"
                ),
            }
        }

        self.architecture = Some(architecture);
        self.carried_metadata = carried;
        Ok(self)
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

    /// Default set of layer names that should stay in FP32 when quantizing
    /// the rest of the model.
    ///
    /// Covers the token embedding and the output projection — the two large
    /// tensors whose quantization costs the most accuracy. Norm tensors are
    /// **not** listed here because they are handled structurally by
    /// [`keep_fp32_by_kind`], which no config can switch off; the old list
    /// named `output_norm.weight` and silently quantized all 200-odd
    /// per-block norms (CQ-02). `output_norm.weight` is kept in the list for
    /// source compatibility with callers that inspect it.
    pub fn default_fp32_exceptions() -> Vec<String> {
        vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
            "output.weight".to_string(),
        ]
    }
}

/// Whether a tensor must stay in FP32 because of **what it is**, regardless
/// of the requested format or any user list.
///
/// Two rules, both of which `llama.cpp`'s own quantizer applies:
///
/// * `shape.len() == 1` — ggml never quantizes a 1-D tensor. The reader's
///   shape is a `Vec<u64>` with exactly `n_dims` entries
///   (`tensor_info.rs:217-236`), so this is an exact test, not a heuristic.
/// * `name` ends with `norm.weight` — RMSNorm gains are a handful of values
///   per layer whose *relative* magnitudes carry the whole normalisation, and
///   a 1-bit or 2-bit encoding of them measured up to **910 % relative error**
///   on the shipped 1.7 B (CQ-02). This clause also catches a norm stored
///   with a redundant trailing dimension.
///
/// The predicate is applied by `encode_tensor`, `estimate_export_size` **and**
/// `export_stats`, so the reported compression ratio always describes the file
/// that was actually written.
///
/// The spec (CQ-02) states the second rule as `name.contains("norm")`; this
/// deliberately narrows it to `name.ends_with("norm.weight")`. Every real
/// norm gain in this workspace's supported architectures (qwen3, qwen35) is
/// 1-D and named `*norm.weight`, so the narrower test has no reachable
/// divergence today, but it is a real narrowing, not just a rewording: a
/// hypothetical 2-D tensor named `*_norm.bias` or `*norm_scale` would slip
/// past this specific rule (a norm gain itself would still be caught by the
/// `shape.len() == 1` rule, since it is always 1-D). Widen this back to
/// `contains("norm")` if a future architecture ever needs it.
pub fn keep_fp32_by_kind(name: &str, shape: &[usize]) -> bool {
    shape.len() == 1 || name.ends_with("norm.weight")
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

    /// The requested [`ExportFormat`] has no corresponding GGUF tensor-type
    /// id and no loader anywhere in this workspace can read it back.
    /// Exporting would silently produce a file that a standard loader
    /// misinterprets (e.g. packed INT8 bytes tagged as raw `F32`), so the
    /// export is refused instead.
    #[error(
        "tensor '{name}': {format:?} has no matching GGUF tensor-type id or loader in this \
         workspace — refusing to export a file that would be silently misread"
    )]
    NoLoaderForFormat {
        /// The tensor the export was refused on.
        name: String,
        /// The format that has no GGUF tensor-type id.
        format: ExportFormat,
    },

    /// The source GGUF has no `general.architecture`, so the output could not
    /// be given a usable architecture block (CQ-01).
    #[error(
        "source model has no `general.architecture`; refusing to write a GGUF whose \
         hyper-parameter keys would have no namespace and which no loader could read back"
    )]
    MissingArchitecture,

    /// The `qwen35.*` block is structurally inconsistent.
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),

    /// The `prism.hadamard.*` contract is inconsistent.
    #[error(transparent)]
    Hadamard(#[from] HadamardError),
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Whether a tensor must stay in FP32 for this particular export.
///
/// Combines the structural rule ([`keep_fp32_by_kind`], which no config can
/// switch off) with the caller's explicit exception list and optional
/// quantize allowlist.
fn should_keep_fp32(name: &str, shape: &[usize], config: &ExportConfig) -> bool {
    // Structural rule: 1-D tensors and norm gains are never quantized.
    if keep_fp32_by_kind(name, shape) {
        return true;
    }
    // Hybrid-architecture policy: a handful of qwen35 scalars/conv weights
    // are excluded from folding and quantization even though their shape
    // (2-D, block-aligned `ne0`) passes every generic rule above — e.g. the
    // real 27B's `ssm_alpha.weight` / `ssm_beta.weight` are `[5120, 48]`
    // (CQ-06 residue). Applied unconditionally rather than gated on
    // `config.qwen35.is_some()`: these exact tensor names never occur in any
    // other supported architecture, and the primary CQ-01 requantize path
    // (`with_source_metadata` alone) never populates `config.qwen35` at all.
    if is_never_quantized_qwen35(name) {
        return true;
    }
    // Explicit FP32 exception list.
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

/// The tensor type a given tensor will actually be written as.
///
/// Resolution order:
///
/// 1. [`should_keep_fp32`] → `F32`.
/// 2. The requested format's tensor type, if its block width divides the
///    tensor's first dimension.
/// 3. Otherwise `F32` again, with a `tracing::info!` — a tensor whose `ne0`
///    is not a block multiple has **no** valid encoding in that format, and
///    `llama.cpp`'s own quantizer keeps such a tensor in its source precision
///    rather than failing the whole run (core-gguf-N1).
///    The previous behaviour — zero-padding the flattened tensor — made every
///    row after the first straddle a group boundary (CQ-14).
///
/// [`ExportFormat::Int8PerChannel`] has no tensor type at all and is reported
/// as `None` so the caller can refuse it with a specific error.
fn effective_tensor_type(name: &str, shape: &[usize], config: &ExportConfig) -> Option<TensorType> {
    if should_keep_fp32(name, shape, config) {
        return Some(TensorType::F32);
    }
    let target = config.format.tensor_type()?;
    let ne0 = shape.first().copied().unwrap_or(0);
    if crate::quantize::row_is_block_aligned(ne0, target) {
        Some(target)
    } else {
        tracing::info!(
            tensor = name,
            ne0,
            block_size = target.block_size(),
            format = config.format.label(),
            "first dimension is not a block multiple — keeping this tensor in F32"
        );
        Some(TensorType::F32)
    }
}

/// Encode one tensor's f32 data into `tensor_type`'s wire bytes.
fn encode_tensor_as(
    name: &str,
    data: &[f32],
    shape: &[usize],
    tensor_type: TensorType,
    rule: ScaleRule,
) -> Result<Vec<u8>, ExportError> {
    let ne0 = shape.first().copied().unwrap_or(data.len());
    crate::quantize::encode_quantized_tensor(data, ne0, tensor_type, rule).map_err(|e| {
        ExportError::QuantizeError {
            name: name.to_string(),
            reason: e.with_tensor(name).to_string(),
        }
    })
}

// ─── Metadata assembly ────────────────────────────────────────────────────────

/// Keys the writer owns and must never inherit from a source file.
///
/// Three groups:
///
/// * `general.alignment` — [`GgufWriter`] injects its own when it uses a
///   non-default alignment, and a second, possibly contradictory value makes
///   the reader compute the wrong data offset.
/// * `general.file_type` / `general.quantization_version` — both describe the
///   *output* encoding and are recomputed from the target format.
/// * Everything [`write_general_metadata`] emits unconditionally, plus the
///   `oxibonsai.*` provenance of the previous export. Carrying these through
///   as well would write each key twice, and the GGUF reader rejects a
///   duplicate key outright.
fn is_writer_owned_key(key: &str) -> bool {
    matches!(
        key,
        "general.alignment"
            | "general.file_type"
            | "general.quantization_version"
            | "general.architecture"
            | "general.name"
            | "general.version"
            | "general.description"
    ) || key.starts_with("oxibonsai.")
}

/// Translate a *read* metadata value into its *write* counterpart.
///
/// Returns `None` for a value with no writable representation (a nested or
/// mixed-type array), which the caller reports rather than silently dropping.
fn convert_metadata_value(value: &MetadataValue) -> Option<MetadataWriteValue> {
    Some(match value {
        MetadataValue::Uint8(v) => MetadataWriteValue::U8(*v),
        MetadataValue::Int8(v) => MetadataWriteValue::I8(*v),
        MetadataValue::Uint16(v) => MetadataWriteValue::U16(*v),
        MetadataValue::Int16(v) => MetadataWriteValue::I16(*v),
        MetadataValue::Uint32(v) => MetadataWriteValue::U32(*v),
        MetadataValue::Int32(v) => MetadataWriteValue::I32(*v),
        MetadataValue::Float32(v) => MetadataWriteValue::F32(*v),
        MetadataValue::Float64(v) => MetadataWriteValue::F64(*v),
        MetadataValue::Uint64(v) => MetadataWriteValue::U64(*v),
        // This used to be
        // `MetadataWriteValue::U64(u64::try_from(*v).ok()?)`, which silently
        // dropped the whole key for any negative source value (`u64::try_from`
        // fails, `.ok()?` short-circuits to `None`) instead of writing it back
        // as the signed type the GGUF spec requires. `MetadataWriteValue::I64`
        // exists precisely for this (see its doc in `gguf/writer.rs`).
        MetadataValue::Int64(v) => MetadataWriteValue::I64(*v),
        MetadataValue::Bool(v) => MetadataWriteValue::Bool(*v),
        MetadataValue::String(v) => MetadataWriteValue::Str(v.clone()),
        MetadataValue::Array(items) => convert_metadata_array(items)?,
    })
}

/// Translate a homogeneous metadata array.
fn convert_metadata_array(items: &[MetadataValue]) -> Option<MetadataWriteValue> {
    // An empty array has no element type to preserve; `arr[str]` with zero
    // entries is the least surprising encoding and round-trips as empty.
    let Some(first) = items.first() else {
        return Some(MetadataWriteValue::ArrayStr(Vec::new()));
    };
    Some(match first {
        MetadataValue::String(_) => MetadataWriteValue::ArrayStr(
            items
                .iter()
                .map(|v| match v {
                    MetadataValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?,
        ),
        MetadataValue::Int32(_) | MetadataValue::Int16(_) | MetadataValue::Int8(_) => {
            MetadataWriteValue::ArrayI32(
                items
                    .iter()
                    .map(MetadataValue::as_i32)
                    .collect::<Option<Vec<_>>>()?,
            )
        }
        MetadataValue::Uint32(_) | MetadataValue::Uint16(_) | MetadataValue::Uint8(_) => {
            MetadataWriteValue::ArrayU32(
                items
                    .iter()
                    .map(MetadataValue::as_u32)
                    .collect::<Option<Vec<_>>>()?,
            )
        }
        MetadataValue::Uint64(_) => MetadataWriteValue::ArrayU64(
            items
                .iter()
                .map(MetadataValue::as_u64)
                .collect::<Option<Vec<_>>>()?,
        ),
        // Same fix as the scalar `Int64` arm above, one level down: an
        // `arr[i64]` containing a negative value must keep its sign, not be
        // funnelled through `as_u64()` (which fails, and thus drops the
        // whole array, for any negative element).
        MetadataValue::Int64(_) => MetadataWriteValue::ArrayI64(
            items
                .iter()
                .map(MetadataValue::as_i64)
                .collect::<Option<Vec<_>>>()?,
        ),
        MetadataValue::Float32(_) | MetadataValue::Float64(_) => MetadataWriteValue::ArrayF32(
            items
                .iter()
                .map(MetadataValue::as_f32)
                .collect::<Option<Vec<_>>>()?,
        ),
        MetadataValue::Bool(_) => MetadataWriteValue::ArrayBool(
            items
                .iter()
                .map(|v| match v {
                    MetadataValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?,
        ),
        MetadataValue::Array(_) => return None,
    })
}

/// Write every metadata block the config describes.
///
/// Order is `general.*`, then carried source metadata, then the architecture
/// blocks, then the tokenizer, then the caller's extra pairs — later writes
/// are what a reader sees last, so the explicitly configured blocks win over
/// anything carried from a source file.
fn write_all_metadata(
    writer: &mut GgufWriter<'_>,
    config: &ExportConfig,
    dominant: TensorType,
    tensor_names: &BTreeSet<String>,
    extra: &[(String, MetadataWriteValue)],
) -> Result<(), ExportError> {
    let architecture = config.architecture.clone().unwrap_or_else(|| {
        tracing::warn!(
            "export config carries no architecture; the output will have no \
             general.architecture and no <arch>.* block"
        );
        String::new()
    });

    let general = GeneralMetadata {
        architecture: architecture.clone(),
        name: config.model_name.clone(),
        version: Some(config.model_version.clone()),
        description: config.description.clone(),
        file_type: ggml_file_type(dominant),
        quant_format: Some(config.format.label().to_string()),
        scale_rule: Some(format!("{:?}", config.scale_rule)),
    };
    write_general_metadata(writer, &general);

    // `rope.dimension_count` / `attention.value_length` live in the shared
    // `<arch>.*` namespace; folding them in here — before anything below
    // consults `merged_arch` — is what keeps the hybrid block from emitting
    // a second copy, which the reader rejects as a duplicate key.
    let merged_arch = if architecture.is_empty() {
        None
    } else {
        config.arch_metadata.clone().map(|mut arch| {
            if let Some(ref hybrid) = config.qwen35 {
                hybrid.apply_to_arch(&mut arch);
            }
            arch
        })
    };

    // Drop every carried key that one of the explicit blocks below is about
    // to write a second time — the reader rejects a duplicate key outright
    // (core-gguf-02 / CQ-01 regression, reintroduced through this builder).
    // `<arch>.*` / `qwen35.*` need an exact key match rather than a
    // namespace prefix: when the architecture itself is `qwen35`, the two
    // are split across `write_arch_metadata` and `write_qwen35_metadata`
    // under one shared literal prefix (see `arch_metadata_keys`'s doc).
    let mut exact_exclude: BTreeSet<String> = BTreeSet::new();
    if let Some(ref arch) = merged_arch {
        exact_exclude.extend(arch_metadata_keys(&architecture, arch));
    }
    if config.qwen35.is_some() {
        exact_exclude.extend(qwen35_metadata_keys());
    }
    let mut prefix_exclude: Vec<&str> = Vec::new();
    if config.tokenizer.is_some() {
        prefix_exclude.push("tokenizer.");
    }
    if config.hadamard.is_some() {
        prefix_exclude.push("prism.hadamard.");
    }
    let carried =
        filter_carried_metadata(&config.carried_metadata, &exact_exclude, &prefix_exclude);
    for (key, value) in &carried {
        writer.add_metadata(key, value.clone());
    }

    if let Some(ref arch) = merged_arch {
        write_arch_metadata(writer, &architecture, arch);
    }
    if let Some(ref hybrid) = config.qwen35 {
        hybrid.validate().map_err(ExportError::Qwen35)?;
        write_qwen35_metadata(writer, hybrid);
    }
    if let Some(ref hadamard) = config.hadamard {
        write_hadamard_metadata(writer, hadamard, tensor_names).map_err(ExportError::Hadamard)?;
    }
    if let Some(ref tokenizer) = config.tokenizer {
        tokenizer.write(writer);
    }

    for (key, value) in extra {
        writer.add_metadata(key, value.clone());
    }
    Ok(())
}

/// The tensor type most of the file will carry, used for
/// `general.file_type`.
fn dominant_tensor_type(config: &ExportConfig) -> TensorType {
    config.format.tensor_type().unwrap_or(TensorType::F32)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Export a list of weight tensors to a GGUF byte buffer.
///
/// # Arguments
///
/// * `tensors` – ordered list of named weight tensors.
/// * `config`  – export configuration (format, name, metadata blocks, FP32
///   exceptions, …).
/// * `arch_metadata` – extra metadata KV pairs appended verbatim after
///   everything the config describes. Prefer
///   [`ExportConfig::with_architecture`] / [`ExportConfig::with_tokenizer`]:
///   this parameter predates them and is kept so existing callers compile.
///
/// # Memory
///
/// This buffers the whole file in RAM. For anything larger than a toy model
/// use [`export_to_gguf_streaming`], which never holds more than one tensor
/// at a time (CQ-17).
///
/// # Errors
///
/// Returns [`ExportError::Empty`] if `tensors` is empty,
/// [`ExportError::QuantizeError`] if quantization of any tensor fails,
/// [`ExportError::NoLoaderForFormat`] for a format with no GGUF tensor type,
/// and [`ExportError::WriteError`] on a writer error.
pub fn export_to_gguf(
    tensors: &[WeightTensor],
    config: &ExportConfig,
    arch_metadata: &[(String, MetadataWriteValue)],
) -> Result<Vec<u8>, ExportError> {
    let non_empty: Vec<&WeightTensor> = tensors.iter().filter(|t| !t.data.is_empty()).collect();
    if non_empty.is_empty() {
        return Err(ExportError::Empty);
    }
    // No GGUF tensor-type id exists for this format at all (today, only
    // `Int8PerChannel`): refuse up front rather than only when a non-exempt
    // tensor happens to need it. Without this, an export whose tensors are
    // all 1-D / norms / off the quantize allowlist quietly "succeeds" in a
    // format documented as never able to (CQ-18 residue).
    if config.format.tensor_type().is_none() {
        return Err(ExportError::NoLoaderForFormat {
            name: non_empty[0].name.clone(),
            format: config.format,
        });
    }

    // Encoded straight from the caller's `&[f32]` — deliberately *not* via
    // `export_to_gguf_streaming`'s callback, which would have to hand over an
    // owned `Vec<f32>` and so clone every tensor. On a real 8 B that is a
    // 2.5 GB clone for `token_embd.weight` alone, on top of the caller's own
    // copy, which is precisely the peak core-gguf-18 exists to bring down.
    let mut writer = GgufWriter::new();
    let tensor_names: BTreeSet<String> = non_empty.iter().map(|t| t.name.clone()).collect();
    write_all_metadata(
        &mut writer,
        config,
        dominant_tensor_type(config),
        &tensor_names,
        arch_metadata,
    )?;

    for tensor in non_empty {
        let tensor_type = effective_tensor_type(&tensor.name, &tensor.shape, config).ok_or(
            ExportError::NoLoaderForFormat {
                name: tensor.name.clone(),
                format: config.format,
            },
        )?;
        let data = encode_tensor_as(
            &tensor.name,
            &tensor.data,
            &tensor.shape,
            tensor_type,
            config.scale_rule,
        )?;
        writer.add_tensor(TensorEntry {
            name: tensor.name.clone(),
            shape: tensor.shape.iter().map(|&d| d as u64).collect(),
            tensor_type,
            data,
        });
    }

    writer
        .to_bytes()
        .map_err(|e| ExportError::WriteError(e.to_string()))
}

/// A tensor that is going to be exported, described without its data.
///
/// The name and shape are enough to decide the output tensor type and its
/// exact byte size, which is what lets [`export_to_gguf_streaming`] write the
/// whole tensor-info directory before touching a single weight — no seek-back
/// and no need to hold the model in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlan {
    /// GGUF tensor name.
    pub name: String,
    /// GGUF shape, first (fastest-varying) dimension first.
    pub shape: Vec<usize>,
}

impl TensorPlan {
    /// Describe a tensor by name and GGUF shape.
    pub fn new(name: &str, shape: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            shape,
        }
    }

    /// Total element count.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Export tensors to `out`, pulling each tensor's data only when it is about
/// to be written.
///
/// # Why this exists (CQ-17 / core-gguf-18)
///
/// The batch API materialises every tensor as `f32` *and* every encoded
/// tensor as `Vec<u8>` *and* the whole file — for a 27 B model that is well
/// over 100 GB of live heap on a 24 GB machine. Here the caller supplies a
/// plan (names and shapes, both known from a GGUF header without reading any
/// weights) plus a `load` callback; the writer emits the header and the
/// complete tensor-info directory from the plan alone, then calls `load` once
/// per tensor, encodes it, writes it, and drops it. Peak additional memory is
/// **one tensor**.
///
/// `load` is invoked exactly once per planned tensor, in order.
///
/// # Errors
///
/// Propagates whatever `load` returns, plus the usual
/// [`ExportError::Empty`] / [`ExportError::QuantizeError`] /
/// [`ExportError::NoLoaderForFormat`] / [`ExportError::WriteError`].
pub fn export_to_gguf_streaming<W, F>(
    plan: &[TensorPlan],
    mut load: F,
    config: &ExportConfig,
    arch_metadata: &[(String, MetadataWriteValue)],
    out: &mut W,
) -> Result<ExportStats, ExportError>
where
    W: std::io::Write,
    F: FnMut(&TensorPlan) -> Result<Vec<f32>, ExportError>,
{
    if plan.is_empty() {
        return Err(ExportError::Empty);
    }
    // See the matching check in `export_to_gguf` (CQ-18 residue): refuse a
    // format with no GGUF tensor-type id up front, rather than only when a
    // non-exempt tensor happens to need it.
    if config.format.tensor_type().is_none() {
        return Err(ExportError::NoLoaderForFormat {
            name: plan[0].name.clone(),
            format: config.format,
        });
    }

    // ── 1. Resolve every tensor's output type from the plan alone. ─────────
    let mut resolved: Vec<(usize, TensorType)> = Vec::with_capacity(plan.len());
    for (idx, entry) in plan.iter().enumerate() {
        let tensor_type = effective_tensor_type(&entry.name, &entry.shape, config).ok_or(
            ExportError::NoLoaderForFormat {
                name: entry.name.clone(),
                format: config.format,
            },
        )?;
        resolved.push((idx, tensor_type));
    }

    let tensor_names: BTreeSet<String> = plan.iter().map(|p| p.name.clone()).collect();

    // ── 2. Metadata. ───────────────────────────────────────────────────────
    let mut writer = GgufWriter::new();
    write_all_metadata(
        &mut writer,
        config,
        dominant_tensor_type(config),
        &tensor_names,
        arch_metadata,
    )?;

    // ── 3. Queue one streaming source per tensor. ──────────────────────────
    // Every callback needs `&mut` access to the same loader, so the loader
    // lives behind a `RefCell` that only one callback borrows at a time (the
    // writer drives them strictly sequentially). Typed errors are stashed
    // rather than flattened into `io::Error`, so the caller still gets the
    // real `ExportError`.
    let state = Rc::new(RefCell::new(StreamState {
        load: &mut load,
        failure: None,
    }));

    let mut stats = ExportStats {
        num_tensors: plan.len(),
        quantized_tensors: 0,
        fp32_tensors: 0,
        original_bytes: 0,
        exported_bytes: 0,
        compression_ratio: 1.0,
    };

    for (idx, tensor_type) in &resolved {
        let entry = &plan[*idx];
        let shape: Vec<u64> = entry.shape.iter().map(|&d| d as u64).collect();
        let declared = tensor_type.row_bytes(&shape);

        if *tensor_type == TensorType::F32 {
            stats.fp32_tensors += 1;
        } else {
            stats.quantized_tensors += 1;
        }
        stats.original_bytes += entry.num_elements() * 4;
        stats.exported_bytes += declared as usize;

        let state = Rc::clone(&state);
        let plan_entry = entry.clone();
        let tensor_type = *tensor_type;
        let rule = config.scale_rule;
        let produce: TensorProducer<'_> = Box::new(move |sink: &mut dyn std::io::Write| {
            let mut guard = state.borrow_mut();
            let data = match (guard.load)(&plan_entry) {
                Ok(d) => d,
                Err(e) => return Err(stash(&mut guard, e)),
            };
            let bytes = match encode_tensor_as(
                &plan_entry.name,
                &data,
                &plan_entry.shape,
                tensor_type,
                rule,
            ) {
                Ok(b) => b,
                Err(e) => return Err(stash(&mut guard, e)),
            };
            drop(data);
            sink.write_all(&bytes)?;
            Ok(bytes.len() as u64)
        });

        writer.add_tensor_stream(TensorStream {
            name: entry.name.clone(),
            shape,
            tensor_type,
            source: TensorSource::Callback(produce, declared),
        });
    }

    // ── 4. Drive the write. ────────────────────────────────────────────────
    let write_result = writer.write_streaming(out);

    // A stashed typed error is always more informative than the `io::Error`
    // the writer surfaces for it.
    if let Some(err) = state.borrow_mut().failure.take() {
        return Err(err);
    }
    write_result.map_err(|e| ExportError::WriteError(e.to_string()))?;

    stats.compression_ratio = if stats.exported_bytes == 0 {
        1.0
    } else {
        stats.original_bytes as f32 / stats.exported_bytes as f32
    };
    Ok(stats)
}

/// Loader state shared by every streaming callback.
struct StreamState<'f> {
    load: &'f mut dyn FnMut(&TensorPlan) -> Result<Vec<f32>, ExportError>,
    failure: Option<ExportError>,
}

/// Record a typed failure and return the placeholder `io::Error` the writer
/// expects.
fn stash(state: &mut StreamState<'_>, err: ExportError) -> std::io::Error {
    let message = err.to_string();
    state.failure.get_or_insert(err);
    std::io::Error::other(message)
}

// ─── Size estimation ──────────────────────────────────────────────────────────

/// Estimate the total exported byte count without actually encoding anything.
///
/// This is an approximation — metadata and tensor-info headers are not
/// included — but the *per-tensor* figure is exact: it applies the same
/// `effective_tensor_type` resolution the writer does (including the FP32
/// carve-outs of CQ-02 and the block-alignment fallback), then asks the type
/// for its real per-row byte count. Before this, the estimate reported the
/// quantized size of tensors the writer kept in FP32, so `oxibonsai quantize`
/// printed a compression ratio the file did not have.
///
/// Note: for [`ExportFormat::Int8PerChannel`] this reports the theoretical
/// packed size for planning / comparison purposes only; [`export_to_gguf`]
/// refuses to actually produce a file in that format (see the format's docs).
pub fn estimate_export_size(tensors: &[WeightTensor], config: &ExportConfig) -> usize {
    tensors
        .iter()
        .map(|t| {
            if t.data.is_empty() {
                return 0;
            }
            match effective_tensor_type(&t.name, &t.shape, config) {
                Some(tensor_type) => {
                    let shape: Vec<u64> = t.shape.iter().map(|&d| d as u64).collect();
                    let shape = if shape.is_empty() {
                        vec![t.data.len() as u64]
                    } else {
                        shape
                    };
                    usize::try_from(tensor_type.row_bytes(&shape)).unwrap_or(usize::MAX)
                }
                // `Int8PerChannel`: i8 data + one f32 scale per channel.
                // `WeightTensor::shape[0]` is the channel (output) dimension,
                // matching `quantize_int8::quantize_per_channel`.
                None => {
                    let num_channels = t.shape.first().copied().unwrap_or(1).max(1);
                    t.data.len() + num_channels * 4
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
///
/// Uses the same `effective_tensor_type` resolution as the writer, so the
/// quantized / FP32 split it reports is the split the file will have.
pub fn export_stats(tensors: &[WeightTensor], config: &ExportConfig) -> ExportStats {
    let mut quantized = 0usize;
    let mut fp32_count = 0usize;
    let mut original_bytes = 0usize;

    for t in tensors {
        original_bytes += t.data.len() * 4;
        match effective_tensor_type(&t.name, &t.shape, config) {
            Some(TensorType::F32) | None => fp32_count += 1,
            Some(_) => quantized += 1,
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

// Split out per the 2000-line file-length policy -- see `export_tests.rs`'s own module doc.
#[cfg(test)]
#[path = "export_tests.rs"]
mod tests;
