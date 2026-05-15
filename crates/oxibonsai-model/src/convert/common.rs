//! Shared helpers for HuggingFace / ONNX → GGUF conversion pipelines.
//!
//! Both the safetensors path (`convert::convert_hf_to_gguf`) and the ONNX path
//! (`convert::onnx::convert_onnx_to_gguf`) emit the same GGUF layout and share:
//!
//! * Parsing of the sibling `config.json`.
//! * Writing Qwen3 metadata (architecture, dimensions, norm epsilon, rope base).
//! * Padding `f32` weights to a multiple of the TQ2_0_g128 block size.
//! * Serialising `BlockTQ2_0_g128` blocks into raw GGUF tensor bytes.
//! * A single `ConvertStats` result struct so callers can report progress
//!   uniformly.

use std::path::Path;

use anyhow::Context;
use serde_json::Value;

use oxibonsai_core::gguf::tensor_info::keys;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};
use oxibonsai_core::quant_ternary::{BlockTQ2_0_g128, BLOCK_TQ2_0_G128_BYTES};

/// Statistics returned after a successful conversion.
///
/// Shared between the safetensors (`convert_hf_to_gguf`) and ONNX
/// (`convert_onnx_to_gguf`) pipelines so CLI callers can report a single
/// struct regardless of source format.
#[derive(Debug, Clone, Default)]
pub struct ConvertStats {
    /// Total number of tensors written to the GGUF file.
    pub n_tensors: usize,
    /// Number of tensors quantized to TQ2_0_g128.
    pub n_ternary: usize,
    /// Number of tensors stored as FP32.
    pub n_fp32: usize,
    /// Total size of the output GGUF file in bytes.
    pub output_bytes: usize,
}

/// Read and parse `config.json` at the given path.
///
/// Callers are responsible for locating the file (the safetensors path uses
/// `from_dir.join("config.json")`; the ONNX path may need to search the ONNX
/// parent and grandparent directories).
pub fn read_config_json(config_path: &Path) -> anyhow::Result<Value> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {:?}", config_path))?;
    let value: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {:?}", config_path))?;
    Ok(value)
}

/// Write Qwen3 metadata from `config.json` into a GGUF writer.
///
/// The caller provides the human-readable model name; for HF this is usually
/// the directory basename, and for ONNX the `.onnx` file stem or repository
/// identifier.
pub fn write_metadata(
    writer: &mut GgufWriter,
    config: &Value,
    model_name: &str,
) -> anyhow::Result<()> {
    // Architecture constant
    writer.add_metadata(
        keys::GENERAL_ARCHITECTURE,
        MetadataWriteValue::Str("qwen3".to_string()),
    );

    // Human-readable model name
    writer.add_metadata(
        keys::GENERAL_NAME,
        MetadataWriteValue::Str(model_name.to_string()),
    );

    // Quantisation version string
    writer.add_metadata(
        "general.quantization_version",
        MetadataWriteValue::Str("TQ2_0_G128".to_string()),
    );

    // Integer keys (u32) - required.
    let u32_keys = [
        (keys::LLM_BLOCK_COUNT, "num_hidden_layers"),
        (keys::LLM_EMBEDDING_LENGTH, "hidden_size"),
        (keys::LLM_FEED_FORWARD_LENGTH, "intermediate_size"),
        (keys::LLM_ATTENTION_HEAD_COUNT, "num_attention_heads"),
        (keys::LLM_ATTENTION_HEAD_COUNT_KV, "num_key_value_heads"),
        (keys::LLM_CONTEXT_LENGTH, "max_position_embeddings"),
        (keys::LLM_VOCAB_SIZE, "vocab_size"),
    ];
    for (gguf_key, json_key) in &u32_keys {
        if let Some(val) = config.get(*json_key).and_then(Value::as_u64) {
            writer.add_metadata(gguf_key, MetadataWriteValue::U32(val as u32));
        } else {
            tracing::warn!(json_key, "missing or non-u64 field in config.json");
        }
    }

    // head_dim is optional in config.json: Qwen3 (1.7B/4B/8B/14B) sets it
    // explicitly because head_dim is decoupled from hidden_size/num_heads
    // (notably Qwen3-4B has hidden=2560, heads=32, head_dim=128, so the
    // derived hidden/heads=80 is wrong). Older Qwen2 configs omit it and
    // the reader falls back to the hidden/heads derivation.
    if let Some(val) = config.get("head_dim").and_then(Value::as_u64) {
        writer.add_metadata(
            keys::LLM_ATTENTION_KEY_LENGTH,
            MetadataWriteValue::U32(val as u32),
        );
    }

    // rms_norm_eps → F32
    if let Some(eps) = config.get("rms_norm_eps").and_then(Value::as_f64) {
        writer.add_metadata(
            keys::LLM_ATTENTION_LAYER_NORM_RMS_EPSILON,
            MetadataWriteValue::F32(eps as f32),
        );
    }

    // rope_theta → F32 (default 10000.0 if absent)
    //
    // Resolution order:
    //   1. `config["rope_theta"]`                    (top-level, legacy Qwen2 layout)
    //   2. `config["rope_parameters"]["rope_theta"]` (nested, Qwen3 ONNX/newer layout)
    //   3. 10000.0 fallback (with `tracing::warn!`)
    let rope_theta = resolve_rope_theta(config);
    writer.add_metadata(
        keys::LLM_ROPE_FREQ_BASE,
        MetadataWriteValue::F32(rope_theta as f32),
    );

    // If the nested `rope_parameters` block indicates YARN scaling, note it.
    //
    // The existing native `Ternary-Bonsai-1.7B.gguf` only carries
    // `llm.rope.freq_base` — no `llm.rope.scaling.*` keys — so for now we log
    // the YARN parameters at info level and rely on architecture defaults
    // rather than inventing new metadata keys.
    if let Some(rp) = config.get("rope_parameters").and_then(Value::as_object) {
        let rope_type = rp.get("rope_type").and_then(Value::as_str).unwrap_or("");
        if rope_type.eq_ignore_ascii_case("yarn") {
            let factor = rp.get("factor").and_then(Value::as_f64);
            let original_max_pos = rp
                .get("original_max_position_embeddings")
                .and_then(Value::as_u64);
            tracing::info!(
                ?factor,
                ?original_max_pos,
                "YARN rope_parameters detected; GGUF YARN metadata not plumbed, relying on architecture defaults"
            );
        }
    }

    Ok(())
}

/// Resolve `rope_theta` from a HuggingFace `config.json` value.
///
/// Looks in this order:
///   1. Top-level `rope_theta` (legacy Qwen2 layout).
///   2. Nested `rope_parameters.rope_theta` (Qwen3 ONNX/newer layout).
///   3. Fallback `10000.0` with a `tracing::warn!` describing the absence.
fn resolve_rope_theta(config: &Value) -> f64 {
    if let Some(v) = config.get("rope_theta").and_then(Value::as_f64) {
        return v;
    }
    if let Some(v) = config
        .get("rope_parameters")
        .and_then(|rp| rp.get("rope_theta"))
        .and_then(Value::as_f64)
    {
        return v;
    }
    tracing::warn!(
        "config.json missing both `rope_theta` and `rope_parameters.rope_theta`; \
         falling back to default 10000.0"
    );
    10000.0
}

/// Pad an f32 slice to a multiple of 128 elements for TQ2_0_g128 quantisation.
///
/// If the length is already block-aligned, the slice is copied verbatim.
/// Otherwise the tail is zero-padded up to the next multiple of 128.
pub fn pad_to_multiple_of_128(f32_data: &[f32]) -> Vec<f32> {
    let len = f32_data.len();
    let remainder = len % 128;
    if remainder == 0 {
        f32_data.to_vec()
    } else {
        let padded_len = len + (128 - remainder);
        let mut padded = f32_data.to_vec();
        padded.resize(padded_len, 0.0_f32);
        padded
    }
}

/// Serialise a slice of `BlockTQ2_0_g128` blocks into raw bytes.
///
/// Each block is 34 bytes: 32 bytes of packed `qs` + 2 bytes of FP16 `d`.
///
/// # Safety
///
/// `BlockTQ2_0_g128` is `#[repr(C)]` with a compile-time size assertion of
/// exactly 34 bytes. The cast is safe because we size the output slice using
/// `blocks.len() * BLOCK_TQ2_0_G128_BYTES`.
pub fn blocks_to_bytes(blocks: &[BlockTQ2_0_g128]) -> Vec<u8> {
    let total = blocks.len() * BLOCK_TQ2_0_G128_BYTES;
    // SAFETY: repr(C) layout with compile-time size check; byte length verified.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total) };
    bytes.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pad_aligned_is_identity() {
        let v = vec![1.0_f32; 128];
        assert_eq!(pad_to_multiple_of_128(&v), v);
    }

    #[test]
    fn pad_extends_to_next_block() {
        let v = vec![1.0_f32; 130];
        let padded = pad_to_multiple_of_128(&v);
        assert_eq!(padded.len(), 256);
        assert_eq!(&padded[..130], &v[..]);
        assert!(padded[130..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn empty_input_stays_empty() {
        let v: Vec<f32> = Vec::new();
        assert!(pad_to_multiple_of_128(&v).is_empty());
    }

    #[test]
    fn rope_theta_top_level_wins() {
        // Legacy Qwen2 layout: `rope_theta` at the top level.
        let cfg = json!({
            "rope_theta": 500000.0,
        });
        assert_eq!(resolve_rope_theta(&cfg), 500000.0);
    }

    #[test]
    fn rope_theta_nested_under_rope_parameters() {
        // Qwen3 ONNX layout: nested under `rope_parameters`.
        let cfg = json!({
            "rope_parameters": {
                "factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_theta": 1_000_000.0,
                "rope_type": "yarn",
            },
        });
        assert_eq!(resolve_rope_theta(&cfg), 1_000_000.0);
    }

    #[test]
    fn rope_theta_top_level_takes_precedence_over_nested() {
        // If both are present, the top-level value wins.
        let cfg = json!({
            "rope_theta": 250000.0,
            "rope_parameters": {
                "rope_theta": 1_000_000.0,
                "rope_type": "yarn",
            },
        });
        assert_eq!(resolve_rope_theta(&cfg), 250000.0);
    }

    #[test]
    fn rope_theta_fallback_when_missing() {
        // Neither key present: fall back to the default 10000.0.
        let cfg = json!({
            "hidden_size": 2048,
        });
        assert_eq!(resolve_rope_theta(&cfg), 10000.0);
    }
}
