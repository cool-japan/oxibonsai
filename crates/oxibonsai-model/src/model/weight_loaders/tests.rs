//! Tests for `weight_loaders.rs`.
//!
//! Split into its own sibling file purely for size (`weight_loaders.rs`
//! was 2050/2000 lines with the PrismML additions) -- mirrors the
//! `weight_loaders/wiring.rs` split already in this directory and the
//! `model/types/mod.rs` + `model/types/tests.rs` pattern elsewhere in this
//! crate. A child module of `weight_loaders` (declared `#[cfg(test)] mod
//! tests;` there), so `use super::*;` below still resolves exactly as it
//! did when this was an inline `mod tests { .. }` block -- no caller-visible
//! change, no test renamed or altered.

use super::*;
use crate::export::{export_to_gguf, ExportConfig, ExportFormat, WeightTensor};

// Regression tests for a gap where `load_f32_tensor` (used to load
// TOKEN_EMBD / OUTPUT_NORM) could not dequantize Q4_K, F8_E4M3, or
// F8_E5M2 tensors, even though `export.rs`'s public `with_fp32_layers`
// override lets a caller quantize exactly those tensor names into those
// formats (by supplying an FP32-exception list that omits them). The
// export<->load pair must be closed under round-trip for every format
// `encode_tensor` can actually produce for these tensor names.

#[test]
fn load_f32_tensor_dequantizes_q4_k() {
    let n = 512usize;
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05 - 12.8).sin()).collect();
    let tensors = vec![WeightTensor::new(
        "token_embd.weight",
        data.clone(),
        vec![n],
    )];
    // Empty fp32_layers list: token_embd.weight is NOT protected here, so
    // it gets quantized to Q4_K like every other tensor under this format.
    let config = ExportConfig::new(ExportFormat::Q4K, "m").with_fp32_layers(vec![]);
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export Q4_K");

    let gguf = GgufFile::parse(&bytes).expect("parse exported GGUF");
    let loaded =
        load_f32_tensor(&gguf, "token_embd.weight").expect("load_f32_tensor should support Q4_K");
    assert_eq!(loaded.len(), n);

    let max_range = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_err = data
        .iter()
        .zip(loaded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_range * 0.08).max(0.1);
    assert!(
        max_err <= threshold,
        "Q4_K roundtrip max error {max_err} > threshold {threshold}"
    );
}

#[test]
fn load_f32_tensor_dequantizes_fp8_e4m3() {
    let n = 64usize;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.2).collect();
    let tensors = vec![WeightTensor::new(
        "output_norm.weight",
        data.clone(),
        vec![n],
    )];
    let config = ExportConfig::new(ExportFormat::FP8E4M3, "m").with_fp32_layers(vec![]);
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export FP8E4M3");

    let gguf = GgufFile::parse(&bytes).expect("parse exported GGUF");
    let loaded = load_f32_tensor(&gguf, "output_norm.weight")
        .expect("load_f32_tensor should support F8_E4M3");
    assert_eq!(loaded.len(), n);

    let max_err = data
        .iter()
        .zip(loaded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err <= 0.5,
        "FP8 E4M3 roundtrip max error {max_err} unexpectedly large"
    );
}

#[test]
fn load_f32_tensor_dequantizes_fp8_e5m2() {
    let n = 64usize;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 3.2).collect();
    let tensors = vec![WeightTensor::new(
        "output_norm.weight",
        data.clone(),
        vec![n],
    )];
    let config = ExportConfig::new(ExportFormat::FP8E5M2, "m").with_fp32_layers(vec![]);
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export FP8E5M2");

    let gguf = GgufFile::parse(&bytes).expect("parse exported GGUF");
    let loaded = load_f32_tensor(&gguf, "output_norm.weight")
        .expect("load_f32_tensor should support F8_E5M2");
    assert_eq!(loaded.len(), n);
}

#[test]
fn load_f32_tensor_unsupported_type_still_errors_clearly() {
    // `export_to_gguf`/`ExportFormat` cannot itself produce a tensor of a
    // genuinely non-executable type (see `dequant_any_covers_every_...`
    // below for that, driven directly rather than via a GGUF round
    // trip), so this test's job is narrower: confirm the happy path for
    // the PrismML `TQ2_0_g128` extension (id 42, qs-first/34B) — which
    // `dequant_any` DOES support, alongside mainline `TQ2_0` (id 35),
    // which `load_f32_tensor` did not handle before but now
    // does too — still round-trips correctly alongside the new arms.
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        vec![1.0_f32; 256],
        vec![256],
    )];
    let config = ExportConfig::new(ExportFormat::TernaryG128, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export ternary");
    let gguf = GgufFile::parse(&bytes).expect("parse exported GGUF");
    // TQ2_0_g128 (the PrismML extension actually produced) IS supported;
    // sanity-check the happy path still works alongside the new arms.
    let loaded =
        load_f32_tensor(&gguf, "blk.0.attn_q.weight").expect("TQ2_0_g128 should still load");
    assert_eq!(loaded.len(), 256);
}

// ── ssm_a must be rejected at load if any
// element is positive (or NaN), not silently logged ────────────────────

#[test]
fn load_f32_tensor_accepts_a_negative_ssm_a() {
    let tensors = vec![WeightTensor::new(
        "blk.0.ssm_a",
        vec![-0.1_f32, -2.0, -0.5],
        vec![3],
    )];
    // Protect it from quantization so the values round-trip exactly.
    let config =
        ExportConfig::new(ExportFormat::Q4K, "m").with_fp32_layers(vec!["blk.0.ssm_a".into()]);
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    let gguf = GgufFile::parse(&bytes).expect("parse exported GGUF");
    let loaded =
        load_f32_tensor(&gguf, "blk.0.ssm_a").expect("all-negative ssm_a must be accepted");
    assert_eq!(loaded.len(), 3);
}

#[test]
fn load_f32_tensor_rejects_a_positive_ssm_a() {
    let tensors = vec![WeightTensor::new(
        "blk.3.ssm_a",
        vec![-0.1_f32, 0.2, -0.5],
        vec![3],
    )];
    let config =
        ExportConfig::new(ExportFormat::Q4K, "m").with_fp32_layers(vec!["blk.3.ssm_a".into()]);
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    let gguf = GgufFile::parse(&bytes).expect("parse exported GGUF");
    let err = load_f32_tensor(&gguf, "blk.3.ssm_a")
        .expect_err("a positive ssm_a element must be rejected, not logged");
    assert_eq!(err.error_code(), "INVALID_TENSOR");
    assert!(
        matches!(err, ModelError::InvalidTensor(_)),
        "expected InvalidTensor, got {err:?}"
    );
}

#[test]
fn load_f32_tensor_ignores_ssm_a_check_for_unrelated_tensor_names() {
    // A tensor that merely CONTAINS positive values but is not named
    // `blk.N.ssm_a` must load normally — the check is name-scoped.
    let tensors = vec![WeightTensor::new(
        "blk.0.ssm_alpha.weight",
        vec![0.7_f32, 0.9],
        vec![2],
    )];
    let config = ExportConfig::new(ExportFormat::Q4K, "m")
        .with_fp32_layers(vec!["blk.0.ssm_alpha.weight".into()]);
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    let gguf = GgufFile::parse(&bytes).expect("parse exported GGUF");
    let loaded = load_f32_tensor(&gguf, "blk.0.ssm_alpha.weight")
        .expect("non-ssm_a tensors must not be subject to the negativity check");
    assert_eq!(loaded.len(), 2);
}

// ── M-09: unified `dequant_any` dispatch table ──────────────────────────

/// Table test (ACCEPTANCE): `dequant_any` must accept every type
/// [`GgufTensorType::is_executable`] reports executable — decoding a
/// single all-zero block to exactly `block_size()` elements — and reject
/// every other type with a named [`BonsaiError::NonExecutableQuantType`]
/// error carrying that type's own id. No `_ =>` wildcard exists in
/// `dequant_any`, so this loop exercises literally every arm.
#[test]
fn dequant_any_covers_every_executable_type_and_names_the_rest() {
    for &ty in GgufTensorType::ALL {
        let group = ty.block_size();
        let data = vec![0u8; ty.block_bytes()];
        let result = dequant_any(ty, &data, group);
        if ty.is_executable() {
            let out = match result {
                Ok(out) => out,
                Err(e) => panic!("dequant_any must accept executable type {ty}: {e}"),
            };
            assert_eq!(
                out.len(),
                group,
                "dequant_any({ty}) returned {} elements, expected {group}",
                out.len()
            );
        } else {
            let err = match result {
                Err(e) => e,
                Ok(_) => panic!("dequant_any must reject non-executable type {ty}"),
            };
            match err {
                ModelError::Core(BonsaiError::NonExecutableQuantType { type_id, .. }) => {
                    assert_eq!(
                        type_id,
                        ty.wire_id(),
                        "{ty}'s rejection must name its own id"
                    );
                }
                other => {
                    panic!("{ty} must be rejected with NonExecutableQuantType, got: {other}")
                }
            }
        }
    }
}

/// `dequant_any` must reject a byte length that disagrees with the
/// caller's declared element count `n`, rather than returning a `Vec`
/// whose length silently differs from what was asked for.
#[test]
fn dequant_any_rejects_n_disagreeing_with_the_byte_length() {
    // One Q4_0 block (18 bytes) decodes to 32 elements, not 31.
    let data = vec![0u8; GgufTensorType::Q4_0.block_bytes()];
    let err = dequant_any(GgufTensorType::Q4_0, &data, 31)
        .expect_err("31 disagrees with what 18 bytes of Q4_0 decodes to");
    assert!(matches!(err, ModelError::ShapeMismatch { .. }));
}

/// End-to-end regression for M-09/K-12: `load_f32_tensor` must dequantize
/// BF16 (previously missing even though `GgufTensorType::BF16` was
/// already declared supported), reusing the exact widen-by-shift the
/// crate already ships in `convert::mlx_image::pack::bf16_to_f32`. The
/// GGUF is hand-built via `GgufWriter` directly (not `export_to_gguf`,
/// whose `ExportFormat` has no BF16 variant) so this exercises the real
/// `GgufFile::parse` → `tensor_data` → `dequant_any` path, not just the
/// dispatch table in isolation.
#[test]
fn load_f32_tensor_dequantizes_bf16_via_real_gguf() {
    use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
    use oxibonsai_core::MetadataWriteValue;

    let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let mut data = Vec::with_capacity(values.len() * 2);
    let mut expected = Vec::with_capacity(values.len());
    for &v in &values {
        // Truncate (not round-to-nearest) so `expected` matches exactly
        // what `bf16_to_f32` (an exact left-shift, no rounding) produces
        // for these bits — the test is about the load *plumbing*, not
        // bf16's own rounding behavior (already covered by `pack.rs`'s
        // own `bf16_to_f32_roundtrip` test).
        let bf16_bits = (v.to_bits() >> 16) as u16;
        data.extend_from_slice(&bf16_bits.to_le_bytes());
        expected.push(f32::from_bits((bf16_bits as u32) << 16));
    }

    let mut writer = GgufWriter::new();
    writer.add_metadata("general.name", MetadataWriteValue::Str("m".to_string()));
    writer.add_metadata(
        "general.version",
        MetadataWriteValue::Str("1.0.0".to_string()),
    );
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![values.len() as u64],
        tensor_type: TensorType::BF16,
        data,
    });
    let bytes = writer.to_bytes().expect("write BF16 GGUF");

    let gguf = GgufFile::parse(&bytes).expect("parse BF16 GGUF");
    let loaded = load_f32_tensor(&gguf, "output_norm.weight")
        .expect("load_f32_tensor must now support BF16");
    assert_eq!(
        loaded, expected,
        "BF16 load must reuse the exact bit-shift widen, no rounding drift"
    );
}

// ── M-Missed-2: tied embeddings ─────────────────────────────────────────

/// ACCEPTANCE: a fixture GGUF with no `output.weight` must load the LM
/// head via the tied `token_embd.weight` path, exactly as llama.cpp does
/// for the many Qwen3-family checkpoints that never write a separate
/// output projection.
#[test]
fn load_output_weight_ties_to_token_embd_when_output_weight_is_absent() {
    let hidden = 8usize;
    let vocab = 5usize;
    let data: Vec<f32> = (0..hidden * vocab)
        .map(|i| (i as f32) * 0.01 - 0.2)
        .collect();
    let tensors = vec![WeightTensor::new(
        "token_embd.weight",
        data.clone(),
        vec![hidden, vocab],
    )];
    let config = ExportConfig::new(ExportFormat::Float32, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export tied fixture");
    let gguf = GgufFile::parse(&bytes).expect("parse tied fixture");
    assert!(
        gguf.tensors.get(tensor_names::OUTPUT).is_none(),
        "fixture must have no output.weight tensor"
    );

    let config = Qwen3Config {
        hidden_size: hidden,
        intermediate_size: hidden,
        num_layers: 1,
        num_attention_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        value_length: hidden,
        vocab_size: vocab,
        max_context_length: 128,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10_000.0,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "qwen3".to_string(),
        model_name: "test".to_string(),
    };
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf).expect("resolve id 42");
    let output_weight = load_output_weight(&gguf, &config, &kernel, resolved_42)
        .expect("must fall back to token_embd.weight when output.weight is absent");
    match output_weight {
        OutputWeight::Fp32 {
            weights,
            out_features,
            in_features,
        } => {
            assert_eq!(in_features, hidden);
            assert_eq!(out_features, vocab);
            assert_eq!(weights, data, "tied weight bytes must be reused unchanged");
        }
        _ => panic!("expected a tied Fp32 output weight"),
    }
}

/// M-10 (applied to `load_output_weight` too): a **present** tensor of a
/// non-executable type must be reported by name
/// (`BonsaiError::NonExecutableQuantType`), not misreported as
/// [`ModelError::MissingTensor`] — the tensor was found; its type is
/// simply not one this build can execute.
#[test]
fn load_output_weight_names_a_non_executable_type_instead_of_missing_tensor() {
    use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
    use oxibonsai_core::MetadataWriteValue;

    let mut writer = GgufWriter::new();
    writer.add_metadata("general.name", MetadataWriteValue::Str("m".to_string()));
    writer.add_metadata(
        "general.version",
        MetadataWriteValue::Str("1.0.0".to_string()),
    );
    // MXFP4: parseable (`GgufTensorType::from_id` succeeds) but
    // `is_executable() == false` — no kernel anywhere in this build.
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![32],
        tensor_type: TensorType::MXFP4,
        data: vec![0u8; GgufTensorType::MXFP4.block_bytes()],
    });
    let bytes = writer.to_bytes().expect("write fixture GGUF");
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");

    let config = Qwen3Config {
        hidden_size: 32,
        intermediate_size: 32,
        num_layers: 1,
        num_attention_heads: 1,
        num_kv_heads: 1,
        head_dim: 32,
        value_length: 32,
        vocab_size: 1,
        max_context_length: 128,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10_000.0,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "qwen3".to_string(),
        model_name: "test".to_string(),
    };
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf).expect("resolve id 42");
    match load_output_weight(&gguf, &config, &kernel, resolved_42) {
        Err(ModelError::Core(BonsaiError::NonExecutableQuantType { type_id, .. })) => {
            assert_eq!(type_id, GgufTensorType::MXFP4.wire_id());
        }
        Err(other) => panic!("expected NonExecutableQuantType, got: {other}"),
        Ok(_) => panic!("MXFP4 output.weight must not silently succeed"),
    }
}

// ── M-10: no `_ =>` arm silently selects a decoder ──────────────────────

/// Build a minimal one-layer GGUF fixture whose `attn_q.weight` (and
/// thus `load_transformer_block`'s sampled type) is `attn_q_type`, with
/// `attn_q_shape`/`attn_q_data` sized correctly for that type's own
/// block geometry (the caller picks these so `validate_row_blocking`
/// accepts the file). Every other required tensor is a trivial FP32
/// norm — only the 4 norm tensors plus the sample tensor are written,
/// since `load_transformer_block` must reject an unsupported
/// `attn_q_type` before it ever tries to load attn_k/v/o or any ffn_*
/// tensor.
fn one_layer_fixture(
    attn_q_type: oxibonsai_core::gguf::writer::TensorType,
    attn_q_shape: Vec<u64>,
    attn_q_data: Vec<u8>,
) -> Vec<u8> {
    use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
    use oxibonsai_core::MetadataWriteValue;

    let mut writer = GgufWriter::new();
    writer.add_metadata("general.name", MetadataWriteValue::Str("m".to_string()));
    writer.add_metadata(
        "general.version",
        MetadataWriteValue::Str("1.0.0".to_string()),
    );

    let norm: Vec<f32> = vec![1.0; 4];
    let norm_bytes: Vec<u8> = norm.iter().flat_map(|f| f.to_le_bytes()).collect();
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
    ] {
        writer.add_tensor(TensorEntry {
            name: name.to_string(),
            shape: vec![4],
            tensor_type: TensorType::F32,
            data: norm_bytes.clone(),
        });
    }
    writer.add_tensor(TensorEntry {
        name: "blk.0.attn_q.weight".to_string(),
        shape: attn_q_shape,
        tensor_type: attn_q_type,
        data: attn_q_data,
    });
    writer.to_bytes().expect("write fixture GGUF")
}

fn tiny_qwen3_config() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 4,
        intermediate_size: 4,
        num_layers: 1,
        num_attention_heads: 1,
        num_kv_heads: 1,
        head_dim: 4,
        value_length: 4,
        vocab_size: 8,
        max_context_length: 128,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10_000.0,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "qwen3".to_string(),
        model_name: "test".to_string(),
    }
}

/// Regression for M-10: `load_transformer_block`'s
/// dispatch used to be a boolean ladder ending in a bare `else` that decoded
/// ANY unmatched quantization type as `Q1_0_g128` with no error. Feeding
/// it an `attn_q.weight` of a type with no `Linear*` wrapper — BF16,
/// executable for `dequant_any` but not wired into any transformer block
/// here — must now produce a named error, not silently-wrong weights.
///
/// This must NOT be `NonExecutableQuantType`: its generated "executable
/// types are …" list would name BF16 itself, since `dequant_any` really
/// can decode it — only this call site has no kernel wrapper for it —
/// so `ModelError::Internal` carries the honest, site-specific message.
#[test]
fn load_transformer_block_rejects_decodable_but_unwired_type() {
    use oxibonsai_core::gguf::writer::TensorType;

    // BF16 block_size is 1, so any element count is a whole number of
    // "blocks"; 4 elements = 8 bytes.
    let bytes = one_layer_fixture(TensorType::BF16, vec![4], vec![0u8; 8]);
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let config = tiny_qwen3_config();
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf).expect("resolve id 42");
    // `TransformerBlock` has no `Debug` impl, so match instead of
    // `.expect_err(..)` (which would need to `Debug`-print the `Ok` case).
    match load_transformer_block(&gguf, &config, 0, &kernel, resolved_42) {
        Err(ModelError::Internal(msg)) => {
            assert!(msg.contains("BF16"), "message must name the type: {msg}");
            assert!(
                msg.contains(&GgufTensorType::BF16.wire_id().to_string()),
                "message must name the id: {msg}"
            );
        }
        Err(other) => panic!("expected ModelError::Internal, got: {other}"),
        Ok(_) => panic!("BF16 attn_q.weight must not silently decode as Q1_0_g128"),
    }
}

/// Build a full 1-layer GGUF fixture (all 7 weight tensors + 4 norms) of
/// a single quant type at a 128×128 shape (one PQ2_0/PTQ1_0 block-width,
/// or two Q2_0G64 block-widths, per row) — big enough to actually
/// exercise `load_transformer_block`'s PrismML arms end to end,
/// unlike `one_layer_fixture` (which only ever writes `attn_q`, enough
/// for the type-dispatch tests above but not a full block).
fn full_layer_fixture_uniform(
    tensor_type: oxibonsai_core::gguf::writer::TensorType,
    block_bytes: impl Fn() -> Vec<u8>,
    blocks_per_matrix: usize,
) -> Vec<u8> {
    use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry};
    use oxibonsai_core::MetadataWriteValue;

    let mut writer = GgufWriter::new();
    writer.add_metadata("general.name", MetadataWriteValue::Str("m".to_string()));
    writer.add_metadata(
        "general.version",
        MetadataWriteValue::Str("1.0.0".to_string()),
    );

    let norm: Vec<f32> = vec![1.0; 128];
    let norm_bytes: Vec<u8> = norm.iter().flat_map(|f| f.to_le_bytes()).collect();
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
    ] {
        writer.add_tensor(TensorEntry {
            name: name.to_string(),
            shape: vec![128],
            tensor_type: oxibonsai_core::gguf::writer::TensorType::F32,
            data: norm_bytes.clone(),
        });
    }

    let matrix_bytes: Vec<u8> = (0..blocks_per_matrix).flat_map(|_| block_bytes()).collect();
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ] {
        writer.add_tensor(TensorEntry {
            name: name.to_string(),
            shape: vec![128, 128],
            tensor_type,
            data: matrix_bytes.clone(),
        });
    }
    writer.to_bytes().expect("write full-layer fixture GGUF")
}

/// A `128×128` `Qwen3Config` matching [`full_layer_fixture_uniform`]'s
/// shapes: `hidden_size = intermediate_size = head_dim = 128`, one Q
/// head, one KV head (so every one of the 7 weight matrices is exactly
/// `128×128`).
fn square_128_qwen3_config() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 128,
        intermediate_size: 128,
        num_layers: 1,
        num_attention_heads: 1,
        num_kv_heads: 1,
        head_dim: 128,
        value_length: 128,
        vocab_size: 8,
        max_context_length: 128,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10_000.0,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "qwen3".to_string(),
        model_name: "test".to_string(),
    }
}

/// A `PQ2_0`-quantized transformer block loads
/// end to end through `load_transformer_block` (it used to hit the "no
/// Linear* wrapper" `Internal` error moved off this arm by this
/// package).
#[test]
fn load_transformer_block_wires_pq2_0() {
    use oxibonsai_core::gguf::writer::TensorType;

    let bytes = full_layer_fixture_uniform(
        TensorType::PQ2_0,
        || {
            let mut v = half::f16::from_f32(1.0).to_le_bytes().to_vec();
            v.extend_from_slice(&[0xFFu8; 32]); // every 2-bit code = 3 -> value +2
            v
        },
        128, // 128 out_features x (128 in_features / 128) = 128 blocks
    );
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let config = square_128_qwen3_config();
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf).expect("resolve id 42");
    match load_transformer_block(&gguf, &config, 0, &kernel, resolved_42) {
        Ok(_) => {}
        Err(e) => panic!("PQ2_0 transformer block must now load, got: {e}"),
    }
}

/// See [`load_transformer_block_wires_pq2_0`]; `PTQ1_0` variant.
#[test]
fn load_transformer_block_wires_ptq1_0() {
    use oxibonsai_core::gguf::writer::TensorType;

    let bytes = full_layer_fixture_uniform(
        TensorType::PTQ1_0,
        || {
            // All-zero qs/qh decode to code 0 (value -1) everywhere.
            let mut v = vec![0u8; 24];
            v.extend_from_slice(&[0u8; 2]); // qh
            v.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d LAST
            v
        },
        128,
    );
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let config = square_128_qwen3_config();
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf).expect("resolve id 42");
    match load_transformer_block(&gguf, &config, 0, &kernel, resolved_42) {
        Ok(_) => {}
        Err(e) => panic!("PTQ1_0 transformer block must now load, got: {e}"),
    }
}

/// See [`load_transformer_block_wires_pq2_0`]; mainline group-64 `Q2_0`
/// variant (128 in_features = 2 group-64 blocks per row).
#[test]
fn load_transformer_block_wires_q2_0_g64() {
    use oxibonsai_core::gguf::writer::TensorType;

    let bytes = full_layer_fixture_uniform(
        TensorType::Q2_0G64,
        || {
            let mut v = half::f16::from_f32(1.0).to_le_bytes().to_vec();
            v.extend_from_slice(&[0xFFu8; 16]); // every 2-bit code = 3 -> value +2
            v
        },
        256, // 128 out_features x (128 in_features / 64) = 256 blocks
    );
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let config = square_128_qwen3_config();
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf)
        .expect("resolve id 42")
        .expect("fixture has a wire-id-42 attn_q.weight tensor");
    assert_eq!(
        resolved_42,
        GgufTensorType::Q2_0G64,
        "the genuinely group-64-shaped fixture bytes must resolve as Q2_0G64, not TQ2_0_g128"
    );
    match load_transformer_block(&gguf, &config, 0, &kernel, Some(resolved_42)) {
        Ok(_) => {}
        Err(e) => panic!("Q2_0G64 transformer block must now load, got: {e}"),
    }
}

/// The other half of M-10's split: a type that is genuinely
/// non-executable (`is_executable() == false`) must still take the
/// `NonExecutableQuantType` path — the split above must not blur into
/// reporting every unmatched type as `Internal`.
#[test]
fn load_transformer_block_names_a_genuinely_non_executable_type() {
    use oxibonsai_core::gguf::writer::TensorType;

    // MXFP4 block_size is 32, so the shape must be a multiple of it —
    // one full block (32 elements) needs exactly `block_bytes()` bytes.
    let bytes = one_layer_fixture(
        TensorType::MXFP4,
        vec![32],
        vec![0u8; GgufTensorType::MXFP4.block_bytes()],
    );
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let config = tiny_qwen3_config();
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf).expect("resolve id 42");
    match load_transformer_block(&gguf, &config, 0, &kernel, resolved_42) {
        Err(ModelError::Core(BonsaiError::NonExecutableQuantType { type_id, .. })) => {
            assert_eq!(type_id, GgufTensorType::MXFP4.wire_id());
        }
        Err(other) => panic!("expected NonExecutableQuantType, got: {other}"),
        Ok(_) => panic!("MXFP4 attn_q.weight must not silently succeed"),
    }
}

// ══════════════════════════════════════════════════════════════════════
// CQ-14 — a mis-declared PQ2_0 tensor is rejected at LOAD
// ══════════════════════════════════════════════════════════════════════

/// One `TQ2_0_g128` block (34 bytes: 32 code bytes then an f16 scale).
/// `plant_plus_two` sets the reserved `0b11` code in the first lane.
fn one_ternary_block(plant_plus_two: bool) -> Vec<u8> {
    let mut data = vec![0u8; 32];
    // Lane values 0b00/0b01/0b10 are the legal ternary codes.
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = (((i % 3) as u8) << 6) | (((i % 3) as u8) << 4) | ((i % 3) as u8) << 2;
    }
    if plant_plus_two {
        data[0] |= 0b11;
    }
    data.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    data
}

/// [`one_layer_fixture`], specialised for a single-block `TQ2_0_g128`
/// `attn_q.weight` (the CQ-14 tests below).
///
/// Two differences from `one_layer_fixture`, both needed only because a
/// single 34-byte block is small enough that the id-42 resolver's own
/// evidence sources go degenerate on it:
///
/// * ggml's 32-byte default inter-tensor alignment pads BOTH a 34-byte
///   (group-128) and a 36-byte (group-64) reading of one block up to the
///   same 64 bytes, erasing the very size distinction
///   `resolve_type_42_with_sample`'s offset replay depends on. A 2-byte
///   alignment removes that collision (34 and 36 are already even).
/// * `one_ternary_block(true)` deliberately plants the reserved `0b11`
///   code the CQ-14 screen exists to catch, which also means the ONE
///   sampled block is not "clean" under the sniff's own definition for
///   ANY byte-order hypothesis — exactly the genuinely-undecidable case
///   [`oxibonsai_core::gguf::quant_resolve::resolve_type_42_with_sample`]
///   falls back to `general.quantization_version`'s legacy string tag
///   for, which every real OxiBonsai-written file of this format
///   carries (`ExportFormat::TernaryG128` sets it too).
fn one_layer_ternary_fixture(attn_q_data: Vec<u8>) -> Vec<u8> {
    use oxibonsai_core::gguf::quant_resolve::LEGACY_QVER_STRING;
    use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};
    use oxibonsai_core::MetadataWriteValue;

    let mut writer = GgufWriter::new();
    writer.set_alignment(2);
    writer.add_metadata("general.name", MetadataWriteValue::Str("m".to_string()));
    writer.add_metadata(
        "general.version",
        MetadataWriteValue::Str("1.0.0".to_string()),
    );
    writer.add_metadata(
        "general.quantization_version",
        MetadataWriteValue::Str(LEGACY_QVER_STRING.to_string()),
    );

    let norm: Vec<f32> = vec![1.0; 4];
    let norm_bytes: Vec<u8> = norm.iter().flat_map(|f| f.to_le_bytes()).collect();
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
    ] {
        writer.add_tensor(TensorEntry {
            name: name.to_string(),
            shape: vec![4],
            tensor_type: TensorType::F32,
            data: norm_bytes.clone(),
        });
    }
    writer.add_tensor(TensorEntry {
        name: "blk.0.attn_q.weight".to_string(),
        shape: vec![128],
        tensor_type: TensorType::TQ2_0_g128,
        data: attn_q_data,
    });
    writer.to_bytes().expect("write ternary fixture GGUF")
}

#[test]
fn a_reserved_plus_two_code_is_rejected_at_load_with_the_tensor_name() {
    let bytes = one_layer_ternary_fixture(one_ternary_block(true));
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");

    let err = load_f32_tensor(&gguf, "blk.0.attn_q.weight")
        .expect_err("a reserved 0b11 code must be rejected at load");
    assert_eq!(err.error_code(), "INVALID_TENSOR", "{err}");
    assert!(
        err.to_string().contains("blk.0.attn_q.weight"),
        "the message must name the tensor: {err}"
    );
}

/// The zero-copy weight path is deliberately NOT screened, and a future
/// reader must be able to see that it is a decision rather than an
/// oversight. `0b11` in a ternary weight is already rejected downstream by
/// the Metal SoA upload (MET-11); screening it *here* would additionally
/// reject synthetic fixtures built from raw random bytes outside this
/// crate, which today fall back to the CPU decoder instead.
#[test]
fn the_zero_copy_ternary_weight_path_is_not_screened() {
    let bytes = one_layer_ternary_fixture(one_ternary_block(true));
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let blocks = load_ternary_blocks(&gguf, "blk.0.attn_q.weight")
        .expect("the zero-copy path must not screen; MET-11 guards it downstream");
    assert_eq!(blocks.len(), 1);
}

#[test]
fn a_clean_ternary_tensor_still_loads() {
    let bytes = one_layer_ternary_fixture(one_ternary_block(false));
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let blocks =
        load_ternary_blocks(&gguf, "blk.0.attn_q.weight").expect("clean ternary must load");
    assert_eq!(blocks.len(), 1);
    assert_eq!(
        load_f32_tensor(&gguf, "blk.0.attn_q.weight")
            .expect("clean ternary must dequantize")
            .len(),
        128
    );
}

/// `PQ2_0` and `Q2_0_g64` define `0b11` as `+2`, so the screen must leave
/// them alone — screening them would reject valid PrismML files.
#[test]
fn prism_two_bit_layouts_are_exempt_from_the_screen() {
    // `PQ2_0` is `d` FIRST then 32 code bytes; plant `0b11` in the codes.
    let mut data = half::f16::from_f32(0.5).to_le_bytes().to_vec();
    data.extend_from_slice(&[0xFFu8; 32]);
    assert!(
        screen_ternary_codes("t", &data, GgufTensorType::PQ2_0).is_ok(),
        "PQ2_0 legitimately encodes +2 as 0b11"
    );
    assert!(screen_ternary_codes("t", &data, GgufTensorType::Q2_0G64).is_ok());
    assert!(screen_ternary_codes("t", &data, GgufTensorType::Q1_0_g128).is_ok());
}

// ══════════════════════════════════════════════════════════════════════
// A genuinely inconclusive id-42 resolution falls
// back to the legacy reading instead of erroring the whole load
// ══════════════════════════════════════════════════════════════════════

/// `one_layer_fixture`'s single 34-byte `TQ2_0_g128` block, at the
/// default 32-byte alignment and with no `general.quantization_version`
/// tag, is exactly the shape many pre-existing synthetic ternary
/// fixtures elsewhere in the workspace use (built before this
/// package's id-42 resolver existed): too little data for the offset
/// replay to settle a unique group size on its own (`34` and `36`
/// bytes both round up to the same 64-byte padded size), so
/// `resolve_type_42_with_sample` reports `AmbiguousQuantType`.
/// `resolve_id42_once` must absorb that specific error into `Ok(None)`
/// (== "no override, use the legacy reading") rather than propagate it,
/// or every one of those fixtures fails to load under
/// `BonsaiModel::from_gguf` the moment any layer's weights are touched
/// -- confirmed empirically against `oxibonsai-runtime`'s
/// `cross_backend_determinism_tests.rs` /
/// `generate_pipeline_tests.rs` / `metal_greedy_cpu_fallback_tests.rs`,
/// none of which set the tag.
#[test]
fn resolve_id42_once_falls_back_to_legacy_when_genuinely_ambiguous() {
    let bytes = one_layer_fixture(
        oxibonsai_core::gguf::writer::TensorType::TQ2_0_g128,
        vec![128],
        one_ternary_block(false),
    );
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let resolved =
        resolve_id42_once(&gguf).expect("a genuinely ambiguous sample must not hard-error");
    assert_eq!(
        resolved, None,
        "an inconclusive resolution must report 'no override' (legacy fallback), not fail \
             the whole load"
    );
    // The fallback must actually let the tensor load, decoded under the
    // legacy qs-first `TQ2_0_g128` reading -- not merely avoid an error
    // while leaving the tensor unusable.
    let loaded = load_f32_tensor(&gguf, "blk.0.attn_q.weight")
        .expect("the ambiguous-but-falls-back-to-legacy tensor must still load");
    assert_eq!(loaded.len(), 128);
}

/// [`resolve_id42_once_falls_back_to_legacy_when_genuinely_ambiguous`],
/// but threaded through [`load_transformer_block`]'s `resolved_42`
/// parameter exactly as `model/types/mod.rs` passes it -- the real
/// production call shape, not just the resolver in isolation. Reuses
/// [`one_layer_fixture`]'s single-tensor shape, so `attn_k`/`attn_v`/…
/// are intentionally absent: it only has to prove `resolved_42: None`
/// flows through to the correct (legacy) match arm, which happens
/// before any of those tensors are touched.
#[test]
fn load_transformer_block_arm_selection_falls_back_to_legacy_when_id42_is_ambiguous() {
    let bytes = one_layer_fixture(
        oxibonsai_core::gguf::writer::TensorType::TQ2_0_g128,
        vec![128],
        one_ternary_block(false),
    );
    let gguf = GgufFile::parse(&bytes).expect("parse fixture GGUF");
    let config = tiny_qwen3_config();
    let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let resolved_42 = resolve_id42_once(&gguf).expect("must not hard-error");
    assert_eq!(
        resolved_42, None,
        "this fixture must reproduce the ambiguous shape"
    );
    // The TQ2_0_g128 arm is selected (not the "no wrapper" error an
    // unresolved id-42 tensor would otherwise fall through to) and only
    // then fails on the missing `attn_k.weight` this minimal fixture
    // never wrote -- proof `resolved_42: None` -> `apply_resolved_type`
    // -> `TQ2_0_g128` reached the match, not a resolution error.
    match load_transformer_block(&gguf, &config, 0, &kernel, resolved_42) {
        Err(ModelError::Core(BonsaiError::TensorNotFound { name })) => {
            assert!(
                name.contains("attn_k"),
                "expected attn_k.weight, got: {name}"
            );
        }
        Err(other) => panic!(
            "expected the TQ2_0_g128 arm to be reached and fail on the missing attn_k.weight \
                 tensor, got: {other}"
        ),
        Ok(_) => panic!(
            "expected the TQ2_0_g128 arm to fail on the missing attn_k.weight tensor this \
                 minimal fixture never wrote"
        ),
    }
}
