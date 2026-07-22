//! HuggingFace ONNX (MatMulNBits, bits=2) → OxiBonsai GGUF conversion.
//!
//! Mirrors the safetensors pipeline in [`crate::convert`] but sources
//! projection weights from `com.microsoft::MatMulNBits` nodes whose packed
//! 2-bit codes are dequantized to f32 in memory before being re-quantized to
//! the requested output format (TQ2_0_g128 by default, or Q1_0_g128).
//!
//! # Pipeline summary
//!
//! 1. Parse the `.onnx` protobuf with `oxionnx-proto::parser::parse_model`.
//! 2. Memory-map the `.onnx_data` sidecar (if any) via [`reader::OnnxReader`].
//! 3. Read sibling `config.json` (typically found two levels up) for
//!    Qwen3 hyper-parameters.
//! 4. Enumerate graph initializers and classify them:
//!    * Norm tensors (f32 or f16 → f32) flow straight through.
//!    * `model.embed_tokens.weight` → `token_embd.weight` (re-quantized).
//!    * `lm_head.weight` (present iff `tie_word_embeddings = false`) →
//!      `output.weight` (re-quantized).
//! 5. Enumerate MatMulNBits nodes and for each one:
//!    * Look up the packed, scales, and zero-points initializers via the
//!      node's `inputs[1..=3]`.
//!    * Dequantize them with [`dequant::dequantize_matmul_nbits`] into a
//!      row-major `[N, K]` `Vec<f32>`.
//!    * Emit a pending tensor with `gguf_shape = [K, N]` (reversed) and
//!      `gguf_name` derived from the HF-style base name via
//!      [`role_map::matmul_node_to_gguf`].
//! 6. Handle `tie_word_embeddings` (duplicate token_embd as output.weight
//!    if lm_head is absent).
//! 7. Sort by GGUF name, pad to the 128-element group size, re-quantize
//!    norms as f32 / weights as TQ2_0_g128 or Q1_0_g128 (per `quant`), and
//!    write the GGUF file.

pub mod dequant;
pub mod error;
pub mod reader;
pub mod role_map;

use std::collections::BTreeMap;
use std::io::BufWriter;
use std::path::Path;

use serde_json::Value;

use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;
use oxionnx_proto::types::{NodeProto, TensorProto};

use crate::convert::common::{
    blocks_to_bytes, pad_to_multiple_of_128, read_config_json, write_metadata, ConvertStats,
};
use crate::quantize::quantize_q1_0_g128;

pub use self::error::{DequantError, OnnxImportError};
pub use self::role_map::OnnxRole;

/// Convert a HuggingFace MatMulNBits-quantized ONNX model into an OxiBonsai
/// GGUF file.
///
/// # Arguments
///
/// * `onnx_path` — full path to the `.onnx` file. The sibling `.onnx_data`
///   sidecar (if any) is located automatically via the `external_data`
///   `"location"` entries of the individual initializers.
/// * `to_path` — destination GGUF file path.
/// * `quant` — target quantisation format: `"tq2_0_g128"` (ternary, default)
///   or `"q1_0_g128"` (1-bit sign + FP16 group scale). Both use 128-element
///   groups; norm/embedding-role tensors are always kept FP32.
///
/// # Errors
///
/// Returns an [`OnnxImportError`] on any I/O, parse, unsupported `quant`
/// value, or conversion failure.
pub fn convert_onnx_to_gguf(
    onnx_path: &Path,
    to_path: &Path,
    quant: &str,
) -> Result<ConvertStats, OnnxImportError> {
    if quant != "tq2_0_g128" && quant != "q1_0_g128" {
        return Err(OnnxImportError::Other(format!(
            "unsupported quantisation format '{quant}'; supported formats are 'tq2_0_g128' and \
             'q1_0_g128'"
        )));
    }

    // ── 1. Parse ONNX and memory-map sidecar on demand ──────────────────────
    let mut reader = reader::OnnxReader::open(onnx_path)?;

    // ── 2. Load sibling config.json ─────────────────────────────────────────
    let config_path = reader::locate_config_json(onnx_path)?;
    let config = read_config_json(&config_path).map_err(|e| {
        // read_config_json returns anyhow::Error. Wrap into Other for a uniform
        // error type.
        OnnxImportError::Other(format!("reading {:?}: {e}", config_path))
    })?;

    // ── 3. Build GGUF writer + write metadata ───────────────────────────────
    let mut writer = GgufWriter::new();
    let model_name = onnx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    write_metadata(&mut writer, &config, model_name, quant)
        .map_err(|e| OnnxImportError::Other(format!("writing metadata: {e}")))?;

    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let num_hidden_layers = config
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            OnnxImportError::Other(
                "config.json is missing required field 'num_hidden_layers'".to_string(),
            )
        })? as usize;

    let vocab_size = config
        .get("vocab_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            OnnxImportError::Other("config.json is missing required field 'vocab_size'".to_string())
        })? as usize;

    let hidden_size = config
        .get("hidden_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            OnnxImportError::Other(
                "config.json is missing required field 'hidden_size'".to_string(),
            )
        })? as usize;

    // ── 4. Collect pending tensors deterministically by GGUF name ───────────
    let mut gguf_entries: BTreeMap<String, PendingTensor> = BTreeMap::new();

    // ── 4a. FP initializers (norms, embed_tokens, lm_head) ──────────────────
    // We walk a snapshot of initializer names so we can still borrow the
    // reader mutably when resolving their bytes.
    let init_names: Vec<String> = reader
        .model
        .graph
        .initializers
        .iter()
        .map(|t| t.name.clone())
        .collect();

    for name in &init_names {
        let Some(role) = role_map::classify_initializer(name, num_hidden_layers) else {
            continue;
        };
        match role {
            OnnxRole::NormFp { gguf_name } => {
                let (f32_data, shape_onnx) = read_fp_initializer(&mut reader, name)?;
                // Norm tensors are 1-D; GGUF keeps the same dimension ordering.
                let gguf_shape: Vec<u64> = shape_onnx.iter().rev().map(|&d| d as u64).collect();
                gguf_entries.insert(
                    gguf_name.clone(),
                    PendingTensor {
                        gguf_name,
                        kind: TensorKind::Norm,
                        gguf_shape,
                        f32_data,
                    },
                );
            }
            OnnxRole::EmbeddingFp => {
                let (f32_data, shape_onnx) = read_fp_initializer(&mut reader, name)?;
                let gguf_shape: Vec<u64> = shape_onnx.iter().rev().map(|&d| d as u64).collect();
                gguf_entries.insert(
                    "token_embd.weight".to_string(),
                    PendingTensor {
                        gguf_name: "token_embd.weight".to_string(),
                        kind: TensorKind::Weight,
                        gguf_shape,
                        f32_data,
                    },
                );
            }
            OnnxRole::LmHeadFp => {
                let (f32_data, shape_onnx) = read_fp_initializer(&mut reader, name)?;
                let gguf_shape: Vec<u64> = shape_onnx.iter().rev().map(|&d| d as u64).collect();
                gguf_entries.insert(
                    "output.weight".to_string(),
                    PendingTensor {
                        gguf_name: "output.weight".to_string(),
                        kind: TensorKind::Weight,
                        gguf_shape,
                        f32_data,
                    },
                );
            }
            // MatMulPacked / Scales / ZeroPoints: handled by MatMulNBits node
            // traversal (which is the authoritative source). Skip here.
            OnnxRole::MatMulPacked { .. }
            | OnnxRole::MatMulScales { .. }
            | OnnxRole::MatMulZeroPoints { .. } => {
                continue;
            }
        }
    }

    // ── 4b. MatMulNBits nodes: dequantize each into row-major [N,K] f32 ─────
    // Snapshot the node metadata we need, so we can then borrow `reader`
    // mutably below when reading each initializer's bytes.
    let matmul_snapshot: Vec<MatMulNbitsMeta> = reader
        .model
        .graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "MatMulNBits")
        .map(collect_matmul_meta)
        .collect::<Result<Vec<_>, _>>()?;

    for meta in &matmul_snapshot {
        let gguf_name = role_map::matmul_node_to_gguf(&meta.node_name)?;

        // Resolve packed, scales, (optional) zero_points bytes. The packed
        // input can come through a Reshape node (lm_head tied-embed case), so
        // we walk at most one Reshape indirection to find the real initializer.
        let packed_tensor = resolve_matmul_input_tensor(&reader, &meta.packed_name)?;
        let scales_tensor = resolve_matmul_input_tensor(&reader, &meta.scales_name)?;
        let zp_tensor: Option<TensorProto> = match meta.zero_points_name.as_ref() {
            Some(name) => Some(resolve_matmul_input_tensor(&reader, name)?),
            None => None,
        };

        let packed_bytes: Vec<u8> = reader.initializer_bytes(&packed_tensor)?.to_vec();
        let scales_bytes: Vec<u8> = reader.initializer_bytes(&scales_tensor)?.to_vec();
        let zp_bytes_opt: Option<Vec<u8>> = if let Some(zp) = zp_tensor.as_ref() {
            Some(reader.initializer_bytes(zp)?.to_vec())
        } else {
            None
        };

        let scales_f32 =
            reader::bytes_to_f32(&scales_bytes, scales_tensor.data_type, &meta.scales_name)?;

        let f32_row_major = dequant::dequantize_matmul_nbits(
            &packed_bytes,
            &scales_f32,
            zp_bytes_opt.as_deref(),
            meta.n,
            meta.k,
            meta.bits,
            meta.block_size,
        )
        .map_err(|e| OnnxImportError::Dequant {
            node: meta.node_name.clone(),
            source: e,
        })?;

        // Conceptual shape [N, K] row-major in f32_row_major.  GGUF reverses
        // dimensions so the on-disk shape reads [K, N] — identical to the
        // safetensors path.
        let gguf_shape = vec![meta.k as u64, meta.n as u64];

        gguf_entries.insert(
            gguf_name.clone(),
            PendingTensor {
                gguf_name,
                kind: TensorKind::Weight,
                gguf_shape,
                f32_data: f32_row_major,
            },
        );
    }

    // ── 4c. GatherBlockQuantized embedding (8B path, `tie=false`) ──────────
    //
    // The onnx-community 8B export does NOT route `embed_tokens` through a
    // MatMulNBits node. Instead it uses a Microsoft `GatherBlockQuantized`
    // contrib op backed by three independent initializers:
    //
    //   * `model_embed_tokens_weight_quant`   u8  [N, K/4]
    //   * `model_embed_tokens_weight_scales`  f16 [N, n_blocks]
    //   * `model_embed_tokens_weight_zp_4b`   u8  [N, n_blocks / 2]
    //
    // The packed bytes are interpreted as *2-bit* codes (4 codes/byte,
    // LSB-first), identical to MatMulNBits. Although the GBQ op carries a
    // `bits=4` attribute, the zero-point buffer stores 4-bit nibbles whose
    // values are all `≤ 3` — i.e. the real quantisation is 2-bit ternary
    // and the `bits=4` attribute is misleading. We hard-code `bits=2`,
    // `block_size=128` here; the re-pack helper enforces the nibble-range
    // invariant.
    //
    // When the primary `*_quant` initializer is absent (e.g. on the 1.7B
    // tied-embedding model), the block is skipped entirely — no regression.
    let mut token_embd_emitted = false;
    if let Some(quant_tensor) = reader
        .find_initializer("model_embed_tokens_weight_quant")
        .cloned()
    {
        let scales_tensor = reader
            .find_initializer("model_embed_tokens_weight_scales")
            .cloned()
            .ok_or_else(|| OnnxImportError::MissingNamedInitializer {
                name: "model_embed_tokens_weight_scales".to_string(),
            })?;
        let zp_tensor = reader
            .find_initializer("model_embed_tokens_weight_zp_4b")
            .cloned()
            .ok_or_else(|| OnnxImportError::MissingNamedInitializer {
                name: "model_embed_tokens_weight_zp_4b".to_string(),
            })?;

        // Dimensions we expect, derived from config.json + the ternary
        // block size the whole pipeline assumes.
        let n = vocab_size;
        let k = hidden_size;
        let block_size = dequant::EXPECTED_BLOCK_SIZE; // 128
        let n_blocks = k.div_ceil(block_size);
        let expected_quant_dims = [n as i64, (k / 4) as i64];
        let expected_scales_dims = [n as i64, n_blocks as i64];
        let expected_zp_dims = [n as i64, (n_blocks / 2) as i64];

        if quant_tensor.dims.as_slice() != expected_quant_dims {
            return Err(OnnxImportError::Other(format!(
                "GBQ embed 'model_embed_tokens_weight_quant' has dims {:?}, expected {:?} \
                 (N=vocab_size={}, K/4={})",
                quant_tensor.dims,
                expected_quant_dims,
                n,
                k / 4
            )));
        }
        if scales_tensor.dims.as_slice() != expected_scales_dims {
            return Err(OnnxImportError::Other(format!(
                "GBQ embed 'model_embed_tokens_weight_scales' has dims {:?}, expected {:?} \
                 (N=vocab_size={}, n_blocks={})",
                scales_tensor.dims, expected_scales_dims, n, n_blocks
            )));
        }
        if zp_tensor.dims.as_slice() != expected_zp_dims {
            return Err(OnnxImportError::Other(format!(
                "GBQ embed 'model_embed_tokens_weight_zp_4b' has dims {:?}, expected {:?} \
                 (N=vocab_size={}, n_blocks/2={})",
                zp_tensor.dims,
                expected_zp_dims,
                n,
                n_blocks / 2
            )));
        }

        // Resolve raw bytes for all three initializers. `initializer_bytes`
        // borrows the reader mutably, so fetch each slice and drop the
        // borrow before the next call.
        let quant_bytes: Vec<u8> = reader.initializer_bytes(&quant_tensor)?.to_vec();
        let scales_bytes: Vec<u8> = reader.initializer_bytes(&scales_tensor)?.to_vec();
        let zp_bytes: Vec<u8> = reader.initializer_bytes(&zp_tensor)?.to_vec();

        let scales_f32 =
            reader::bytes_to_f32(&scales_bytes, scales_tensor.data_type, &scales_tensor.name)?;

        // Re-pack the 4-bit-packed zero-points into the 2-bit layout that
        // `dequantize_matmul_nbits` consumes.
        let zp_repacked =
            dequant::repack_4bit_zp_to_2bit(&zp_bytes, n * n_blocks).map_err(|e| {
                OnnxImportError::Dequant {
                    node: "GatherBlockQuantized(embed_tokens)".to_string(),
                    source: e,
                }
            })?;

        // Dequantize to row-major [N, K] f32. `bits` and `block_size` are
        // intentionally hard-coded — the GBQ `bits=4` attribute is a lie.
        let f32_row_major = dequant::dequantize_matmul_nbits(
            &quant_bytes,
            &scales_f32,
            Some(&zp_repacked),
            n,
            k,
            2,
            block_size,
        )
        .map_err(|e| OnnxImportError::Dequant {
            node: "GatherBlockQuantized(embed_tokens)".to_string(),
            source: e,
        })?;

        // GGUF stores the embedding as [K, N] — same convention as the
        // MatMulNBits `output.weight` path (see loop 4b above).
        let gguf_shape = vec![k as u64, n as u64];

        gguf_entries.insert(
            "token_embd.weight".to_string(),
            PendingTensor {
                gguf_name: "token_embd.weight".to_string(),
                kind: TensorKind::Weight,
                gguf_shape,
                f32_data: f32_row_major,
            },
        );
        token_embd_emitted = true;

        tracing::info!(
            "GBQ embed detected: N={}, K={}, emitted token_embd.weight via 2-bit re-pack (bits=4 attribute overridden)",
            n,
            k
        );
    }

    // ── 5. Tied embeddings fix-up ───────────────────────────────────────────
    //
    // onnx-community exports drop the `model.embed_tokens.weight` initializer
    // (instead using a `GatherBlockQuantized` node over the *same* quantized
    // table that feeds `lm_head`). When `tie_word_embeddings=true` and only
    // the `lm_head` MatMulNBits produced an `output.weight`, clone that
    // tensor back into `token_embd.weight`.
    //
    // We also cover the mirror case (classic HF: `token_embd` present,
    // `output.weight` absent) for robustness.
    //
    // When step 4c already emitted `token_embd.weight` from an independent
    // GBQ source (8B, `tie=false` path), this fix-up MUST be skipped — the
    // two tables are *not* byte-identical and cloning `output.weight` over
    // the just-emitted tensor would silently corrupt the embedding.
    if tie_word_embeddings && !token_embd_emitted {
        match (
            gguf_entries.contains_key("token_embd.weight"),
            gguf_entries.contains_key("output.weight"),
        ) {
            (false, true) => {
                if let Some(source) = gguf_entries.get("output.weight") {
                    let cloned = PendingTensor {
                        gguf_name: "token_embd.weight".to_string(),
                        kind: source.kind,
                        gguf_shape: source.gguf_shape.clone(),
                        f32_data: source.f32_data.clone(),
                    };
                    tracing::info!(
                        "tie_word_embeddings=true: duplicating output.weight as token_embd.weight"
                    );
                    gguf_entries.insert("token_embd.weight".to_string(), cloned);
                }
            }
            (true, false) => {
                if let Some(source) = gguf_entries.get("token_embd.weight") {
                    let cloned = PendingTensor {
                        gguf_name: "output.weight".to_string(),
                        kind: source.kind,
                        gguf_shape: source.gguf_shape.clone(),
                        f32_data: source.f32_data.clone(),
                    };
                    tracing::info!(
                        "tie_word_embeddings=true: duplicating token_embd.weight as output.weight"
                    );
                    gguf_entries.insert("output.weight".to_string(), cloned);
                }
            }
            _ => {}
        }
    }

    // ── 6. Emit tensors ─────────────────────────────────────────────────────
    let mut stats = ConvertStats::default();

    for pending in gguf_entries.values() {
        let (raw_bytes, tensor_type) = match pending.kind {
            TensorKind::Norm => {
                let raw: Vec<u8> = pending
                    .f32_data
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                (raw, TensorType::F32)
            }
            TensorKind::Weight if quant == "q1_0_g128" => {
                // Q1_0_g128: 1-bit sign + FP16 group scale, same 128-element
                // group size and padding as TQ2_0_g128. Uses the canonical
                // sign convention shared with
                // `oxibonsai_core::tensor::BlockQ1_0G128` (see
                // `crate::quantize` module docs): bit=1 -> +scale, bit=0 -> -scale.
                let padded = pad_to_multiple_of_128(&pending.f32_data);
                let raw = quantize_q1_0_g128(&padded).map_err(|e| OnnxImportError::Requantize {
                    tensor: pending.gguf_name.clone(),
                    msg: format!("{e}"),
                })?;
                (raw, TensorType::Q1_0G128)
            }
            TensorKind::Weight => {
                let padded = pad_to_multiple_of_128(&pending.f32_data);
                let blocks = BlockTQ2_0_g128::quantize(&padded).map_err(|e| {
                    OnnxImportError::Requantize {
                        tensor: pending.gguf_name.clone(),
                        msg: format!("{e}"),
                    }
                })?;
                let raw = blocks_to_bytes(&blocks);
                (raw, TensorType::TQ2_0_g128)
            }
        };

        println!(
            "  converting {} {:?} -> {}",
            pending.gguf_name,
            pending.gguf_shape,
            match pending.kind {
                TensorKind::Norm => "F32",
                TensorKind::Weight if quant == "q1_0_g128" => "Q1_0_g128",
                TensorKind::Weight => "TQ2_0_g128",
            }
        );

        writer.add_tensor(TensorEntry {
            name: pending.gguf_name.clone(),
            shape: pending.gguf_shape.clone(),
            tensor_type,
            data: raw_bytes,
        });

        match pending.kind {
            TensorKind::Norm => stats.n_fp32 += 1,
            TensorKind::Weight => stats.n_ternary += 1,
        }
        stats.n_tensors += 1;
    }

    // ── 7. Write GGUF file ──────────────────────────────────────────────────
    let out_file = std::fs::File::create(to_path).map_err(|e| OnnxImportError::Io {
        path: to_path.to_path_buf(),
        source: e,
    })?;
    let mut buf_writer = BufWriter::new(out_file);
    let bytes_written = writer
        .write(&mut buf_writer)
        .map_err(|e| OnnxImportError::GgufWrite(format!("{e}")))?;

    stats.output_bytes = bytes_written;
    Ok(stats)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Tag describing how a pending tensor should be serialised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorKind {
    /// FP32 norm tensor (never quantized).
    Norm,
    /// Re-quantized TQ2_0_g128 weight tensor.
    Weight,
}

/// Accumulator for a tensor that is about to be emitted.
struct PendingTensor {
    gguf_name: String,
    kind: TensorKind,
    gguf_shape: Vec<u64>,
    f32_data: Vec<f32>,
}

/// Read one floating-point (f32/f16/bf16) initializer's bytes and shape.
fn read_fp_initializer(
    reader: &mut reader::OnnxReader,
    name: &str,
) -> Result<(Vec<f32>, Vec<i64>), OnnxImportError> {
    let tensor = reader.find_initializer(name).cloned().ok_or_else(|| {
        OnnxImportError::MissingNamedInitializer {
            name: name.to_string(),
        }
    })?;
    let bytes = reader.initializer_bytes(&tensor)?.to_vec();
    let f32_data = reader::bytes_to_f32(&bytes, tensor.data_type, &tensor.name)?;
    Ok((f32_data, tensor.dims.clone()))
}

/// Metadata harvested from a single MatMulNBits node.
struct MatMulNbitsMeta {
    node_name: String,
    packed_name: String,
    scales_name: String,
    zero_points_name: Option<String>,
    n: usize,
    k: usize,
    bits: u32,
    block_size: usize,
}

/// Extract the bits/block_size/N/K attributes and input tensor names from a
/// single MatMulNBits node.
///
/// The target GGUF tensor name is derived from `node.name` by
/// [`role_map::matmul_node_to_gguf`], so we no longer strip any `_quantized`
/// suffix here — `onnx-community` exports use `_quant` and sometimes pipe the
/// packed weight through a `Reshape` before feeding it into `MatMulNBits`.
fn collect_matmul_meta(node: &NodeProto) -> Result<MatMulNbitsMeta, OnnxImportError> {
    let node_name = if node.name.is_empty() {
        "<anon>".to_string()
    } else {
        node.name.clone()
    };

    let packed_name = node
        .inputs
        .get(1)
        .ok_or_else(|| OnnxImportError::MissingInitializer {
            node: node_name.clone(),
            index: 1,
            name: "<missing>".to_string(),
        })?
        .clone();
    let scales_name = node
        .inputs
        .get(2)
        .ok_or_else(|| OnnxImportError::MissingInitializer {
            node: node_name.clone(),
            index: 2,
            name: "<missing>".to_string(),
        })?
        .clone();
    let zero_points_name = node.inputs.get(3).cloned().filter(|s| !s.is_empty());

    let bits_i =
        reader::attr_int(&node.attributes, "bits").ok_or(OnnxImportError::MissingAttribute {
            node: node_name.clone(),
            attr: "bits",
        })?;
    let block_size_i = reader::attr_int(&node.attributes, "block_size").ok_or(
        OnnxImportError::MissingAttribute {
            node: node_name.clone(),
            attr: "block_size",
        },
    )?;
    let n_i = reader::attr_int(&node.attributes, "N").ok_or(OnnxImportError::MissingAttribute {
        node: node_name.clone(),
        attr: "N",
    })?;
    let k_i = reader::attr_int(&node.attributes, "K").ok_or(OnnxImportError::MissingAttribute {
        node: node_name.clone(),
        attr: "K",
    })?;
    if bits_i <= 0 || block_size_i <= 0 || n_i <= 0 || k_i <= 0 {
        return Err(OnnxImportError::Other(format!(
            "MatMulNBits node '{node_name}' has non-positive attribute(s): bits={bits_i} block_size={block_size_i} N={n_i} K={k_i}"
        )));
    }

    Ok(MatMulNbitsMeta {
        node_name,
        packed_name,
        scales_name,
        zero_points_name,
        n: n_i as usize,
        k: k_i as usize,
        bits: bits_i as u32,
        block_size: block_size_i as usize,
    })
}

/// Resolve a MatMulNBits input (packed / scales / zero-points) to a
/// `TensorProto`, following at most one `Reshape` indirection.
///
/// In `onnx-community` exports the `lm_head` MatMulNBits receives its packed
/// weight through a `Reshape` node rather than a direct initializer, e.g.
///
/// ```text
/// model_embed_tokens_weight_quant  (initializer [V, K/4])
///   ↓ via /lm_head/MatMul_ReshapeQuant (op Reshape)
/// model_embed_tokens_weight_quant_matmul (Reshape output)
///   ↓ (MatMulNBits input[1])
/// ```
///
/// When `name` resolves directly to an initializer we return that. When it
/// does not, we look for the node that *produces* `name` as an output; if
/// that node is a `Reshape` whose `inputs[0]` is itself an initializer, we
/// return that original initializer.
fn resolve_matmul_input_tensor(
    reader: &reader::OnnxReader,
    name: &str,
) -> Result<TensorProto, OnnxImportError> {
    if let Some(t) = reader.find_initializer(name) {
        return Ok(t.clone());
    }

    // Search for a node whose outputs include `name`.
    let producer = reader
        .model
        .graph
        .nodes
        .iter()
        .find(|n| n.outputs.iter().any(|o| o == name));

    if let Some(node) = producer {
        if node.op_type == "Reshape" {
            if let Some(src) = node.inputs.first() {
                if let Some(t) = reader.find_initializer(src) {
                    return Ok(t.clone());
                }
            }
        }
        return Err(OnnxImportError::Other(format!(
            "MatMulNBits input '{name}' is produced by node '{}' (op '{}') whose inputs are not a resolvable initializer",
            node.name, node.op_type
        )));
    }

    Err(OnnxImportError::MissingNamedInitializer {
        name: name.to_string(),
    })
}
