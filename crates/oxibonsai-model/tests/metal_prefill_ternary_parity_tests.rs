//! Parity tests for the batched Metal ternary (TQ2_0_g128) prefill path.
//!
//! These tests verify that the new
//! [`oxibonsai_kernels::try_metal_full_forward_prefill_ternary`] path produces
//! the **same** logits as the per-position fused ternary forward
//! ([`oxibonsai_kernels::try_metal_prefill_ternary`] called once per token),
//! and that chunked prefill yields identical results to a single-shot prefill.
//!
//! The fixture is a synthetic 2-layer fully-ternary GGUF model assembled in
//! `std::env::temp_dir()` — small enough to load quickly, large enough to
//! exercise the QKV / attn-output / gate+up / down GEMM paths.

#![cfg(all(feature = "metal", target_os = "macos"))]

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic ternary fixture
// ─────────────────────────────────────────────────────────────────────────────

/// Build a TQ2_0_g128 weight blob with deterministic per-block patterns.
///
/// Each block is 34 bytes: 32 bytes of 2-bit codes (4 weights/byte, LSB-first)
/// followed by a 2-byte FP16 scale. The encoding is
/// `00→-1, 01→0, 10→+1, 11→0`, matching the existing GEMV reference kernel.
///
/// To get a reasonably "interesting" weight matrix we vary both the qs pattern
/// and the scale across blocks based on a 64-bit linear-congruential PRNG seed.
fn tq2_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 34);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_blocks {
        // 32 bytes of qs (128 weights × 2 bits).
        for _ in 0..32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
        // FP16 scale in (0.25, 0.75] so RMSNorm output stays in a sane range.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        let scale_bytes = f16::from_f32(scale_f32).to_le_bytes();
        data.extend_from_slice(&scale_bytes);
    }
    data
}

/// Build an FP32 tensor whose values vary with the index — keeps the embedding
/// path away from degenerate identity inputs while still being deterministic.
fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let phase = (i as f32) * 0.013_f32;
        let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// Build a synthetic fully-ternary GGUF for `BonsaiModel::from_gguf`.
///
/// All projection matrices (attention QKV, attention output, FFN gate/up/down,
/// LM head) are stored as `TQ2_0_g128` so the model exercises the all-ternary
/// GPU path end-to-end. RMSNorm weights and token embeddings stay FP32.
fn build_synthetic_ternary_gguf() -> Vec<u8> {
    let h: usize = 128; // hidden_size — must be ≥ 128 for TQ2_0_g128
    let inter: usize = 256; // intermediate_size — must be multiple of 128
    let num_layers: usize = 2;
    let nq: usize = 4;
    let nkv: usize = 2;
    let hd: usize = 32; // head_dim = h / nq
    let vocab: usize = 32;

    let mut writer = GgufWriter::new();

    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("PrefillTernaryParityTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(num_layers as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(nq as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(nkv as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(inter as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
    writer.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    // Token embedding: vocab × hidden (FP32, varied pattern so different
    // tokens produce different hidden states).
    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(vocab * h, 0.5),
    });

    // Output norm.
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![h as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(h, 1.0),
    });

    // Output projection (LM head): TQ2_0_g128.
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: tq2_0_g128_pattern(vocab * h, 0xCAFE_BABE),
    });

    for layer in 0..num_layers {
        let pfx = format!("blk.{layer}");

        // RMSNorm weights (FP32).
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_norm.weight"),
            shape: vec![h as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(h, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_norm.weight"),
            shape: vec![h as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(h, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_q_norm.weight"),
            shape: vec![hd as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(hd, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_k_norm.weight"),
            shape: vec![hd as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(hd, 1.0),
        });

        // Attention QKV / output: all TQ2_0_g128.
        let layer_seed = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_q.weight"),
            shape: vec![h as u64, (nq * hd) as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(nq * hd * h, layer_seed),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_k.weight"),
            shape: vec![h as u64, (nkv * hd) as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(nkv * hd * h, layer_seed.wrapping_add(1)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_v.weight"),
            shape: vec![h as u64, (nkv * hd) as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(nkv * hd * h, layer_seed.wrapping_add(2)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_output.weight"),
            shape: vec![(nq * hd) as u64, h as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(h * nq * hd, layer_seed.wrapping_add(3)),
        });

        // FFN gate / up / down: all TQ2_0_g128.
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_gate.weight"),
            shape: vec![h as u64, inter as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(inter * h, layer_seed.wrapping_add(4)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_up.weight"),
            shape: vec![h as u64, inter as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(inter * h, layer_seed.wrapping_add(5)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_down.weight"),
            shape: vec![inter as u64, h as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(h * inter, layer_seed.wrapping_add(6)),
        });
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

/// Write the synthetic GGUF to a unique file under `temp_dir()` and return
/// the path. The caller is responsible for deleting the file when done
/// (we use a stable filename keyed by a per-test suffix so re-running the
/// test suite does not pile up garbage).
fn write_temp_gguf(suffix: &str) -> std::path::PathBuf {
    let bytes = build_synthetic_ternary_gguf();
    let mut path = std::env::temp_dir();
    path.push(format!("oxibonsai_metal_prefill_ternary_{suffix}.gguf"));
    std::fs::write(&path, &bytes).expect("write synthetic GGUF to temp dir");
    path
}

/// Parse the synthetic ternary GGUF, returning the parser so the caller can
/// borrow from it when constructing a [`BonsaiModel`].
fn parse_synthetic_gguf(gguf_bytes: &[u8]) -> GgufFile<'_> {
    GgufFile::parse(gguf_bytes).expect("GgufFile::parse synthetic")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Logits from a single 8-token batched prefill must match the logits
/// produced by feeding the same 8 tokens through the per-position fused
/// ternary forward path one position at a time — within FP32 noise.
///
/// We compare only the **last** position because the batched prefill
/// returns only the final token's logits.
#[test]
fn test_batched_ternary_prefill_matches_per_position() {
    let path = write_temp_gguf("parity_8");
    let gguf_bytes = std::fs::read(&path).expect("read synthetic GGUF");
    let _ = std::fs::remove_file(&path);
    let kernel = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));

    let token_ids: Vec<u32> = (0..8_u32).map(|i| (i * 3 + 1) % 32).collect();

    // ── Reference path: sequential per-position forward ─────────────────
    let ref_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let mut model =
            BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf reference");
        let mut last = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            last = model
                .forward(tid, i, kernel.as_ref())
                .expect("sequential forward");
        }
        last
    };

    // ── Batched ternary prefill — STRICT path (no fallback masking) ──────
    let prefill_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf prefill");
        // Note: we deliberately bypass `forward_prefill` to avoid its silent
        // fallback to per-token sequential forward (which would let GPU
        // failures masquerade as parity wins).
        model
            .try_metal_prefill_with_lm_head_ternary(&token_ids, 0)
            .expect("strict ternary prefill")
    };

    assert_eq!(ref_logits.len(), prefill_logits.len(), "logit dim mismatch");
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (a, b)) in ref_logits.iter().zip(prefill_logits.iter()).enumerate() {
        assert!(a.is_finite(), "ref_logits[{i}] not finite");
        assert!(b.is_finite(), "prefill_logits[{i}] not finite");
        let abs_err = (a - b).abs();
        let rel_err = abs_err / (a.abs().max(b.abs()).max(1e-3_f32));
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if rel_err > max_rel {
            max_rel = rel_err;
        }
        assert!(
            abs_err < 1e-3 || rel_err < 1e-3,
            "logit[{i}] mismatch: ref={a:.6}, prefill={b:.6}, abs_err={abs_err:.3e}, rel_err={rel_err:.3e}"
        );
    }
    eprintln!(
        "test_batched_ternary_prefill_matches_per_position: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
    );
}

/// `forward_prefill_verify` (per-position argmax over the batch) must agree
/// with the per-position greedy argmax computed from the sequential forward
/// path — even where the relative logit gap is small.
#[test]
fn test_batched_ternary_prefill_verify_greedy_match() {
    let path = write_temp_gguf("verify_8");
    let gguf_bytes = std::fs::read(&path).expect("read synthetic GGUF");
    let _ = std::fs::remove_file(&path);
    let kernel = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));

    let token_ids: Vec<u32> = (0..8_u32).map(|i| (i * 5 + 7) % 32).collect();

    // Reference: per-position argmax via sequential forward.
    let ref_token_ids: Vec<u32> = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf ref");
        let mut out = Vec::with_capacity(token_ids.len());
        for (i, &tid) in token_ids.iter().enumerate() {
            let logits = model
                .forward(tid, i, kernel.as_ref())
                .expect("sequential forward");
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (j, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = j as u32;
                }
            }
            out.push(best_idx);
        }
        out
    };

    // Batched verify path — STRICT (no silent fallback).
    let prefill_token_ids = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf verify");
        model
            .try_metal_prefill_verify_ternary_path(&token_ids, 0)
            .expect("strict ternary prefill verify")
    };

    assert_eq!(
        ref_token_ids, prefill_token_ids,
        "greedy token IDs disagree between sequential and batched ternary prefill"
    );
}

/// Logits parity at batch_size=12 — exercises the kernel's multi-chunk
/// outer loop (`for col_base in 0..batch by 8u`) which only kicks in when
/// `batch_size > 8`. The Q1 V7 kernel silently caps at 8 columns; this
/// test guarantees the new TQ2 GEMM does not inherit that bug.
#[test]
fn test_batched_ternary_prefill_matches_per_position_batch12() {
    let path = write_temp_gguf("parity_12");
    let gguf_bytes = std::fs::read(&path).expect("read synthetic GGUF");
    let _ = std::fs::remove_file(&path);
    let kernel = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));

    let token_ids: Vec<u32> = (0..12_u32).map(|i| (i * 3 + 1) % 32).collect();

    let ref_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf ref");
        let mut last = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            last = model
                .forward(tid, i, kernel.as_ref())
                .expect("sequential forward");
        }
        last
    };

    let prefill_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf prefill");
        // Strict path: no silent fallback to sequential.
        model
            .try_metal_prefill_with_lm_head_ternary(&token_ids, 0)
            .expect("strict ternary prefill batch12")
    };

    assert_eq!(ref_logits.len(), prefill_logits.len());
    let mut max_abs = 0.0_f32;
    for (i, (a, b)) in ref_logits.iter().zip(prefill_logits.iter()).enumerate() {
        let abs_err = (a - b).abs();
        let rel_err = abs_err / (a.abs().max(b.abs()).max(1e-3_f32));
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        assert!(
            abs_err < 1e-3 || rel_err < 1e-3,
            "logit[{i}] mismatch (batch=12): ref={a:.6} prefill={b:.6} abs_err={abs_err:.3e}"
        );
    }
    eprintln!("test_batched_ternary_prefill_matches_per_position_batch12: max_abs={max_abs:.3e}");
}

/// Splitting an 8-token prompt into 4+4 chunks must yield the same final
/// logits as a single 8-token prefill — both consumed by the new ternary
/// path. (Chunked prefill walks the model with `forward_prefill` once per
/// chunk; here we drive it manually so we can compare logits.)
#[test]
fn test_batched_ternary_prefill_chunked() {
    let path = write_temp_gguf("chunked_8");
    let gguf_bytes = std::fs::read(&path).expect("read synthetic GGUF");
    let _ = std::fs::remove_file(&path);

    let token_ids: Vec<u32> = (0..8_u32).map(|i| (i * 11 + 3) % 32).collect();

    // Single-shot prefill on the full batch — strict path.
    let single_shot_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf single");
        model
            .try_metal_prefill_with_lm_head_ternary(&token_ids, 0)
            .expect("strict single-shot ternary prefill")
    };

    // Two-chunk prefill: 4 + 4 — strict path.
    let chunked_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf chunked");
        let _ = model
            .try_metal_prefill_with_lm_head_ternary(&token_ids[0..4], 0)
            .expect("strict first chunk prefill");
        model
            .try_metal_prefill_with_lm_head_ternary(&token_ids[4..8], 4)
            .expect("strict second chunk prefill")
    };

    assert_eq!(
        single_shot_logits.len(),
        chunked_logits.len(),
        "logit dim mismatch between single-shot and chunked"
    );
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (a, b)) in single_shot_logits
        .iter()
        .zip(chunked_logits.iter())
        .enumerate()
    {
        let abs_err = (a - b).abs();
        let rel_err = abs_err / (a.abs().max(b.abs()).max(1e-3_f32));
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if rel_err > max_rel {
            max_rel = rel_err;
        }
        assert!(
            abs_err < 1e-3 || rel_err < 1e-3,
            "logit[{i}] mismatch single-shot={a:.6} chunked={b:.6} abs_err={abs_err:.3e}"
        );
    }
    eprintln!("test_batched_ternary_prefill_chunked: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}");
}
