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
use oxibonsai_kernels::{MetalGraph, MetalGraphError, SessionScope};
// B5: the uncached binding types are nameable from outside the crate.
use oxibonsai_model::model::{BonsaiModel, TernaryGpuBinding, TernaryTailBinding};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic ternary fixture
// ─────────────────────────────────────────────────────────────────────────────

/// Build a TQ2_0_g128 weight blob with deterministic per-block patterns.
///
/// Each block is 34 bytes: 32 bytes of 2-bit codes (4 weights/byte, LSB-first)
/// followed by a 2-byte FP16 scale. **Only the three ternary codes are ever
/// emitted:** `00→-1, 01→0, 10→+1`. The fourth code `0b11` is reserved — in a
/// tensor declared `TQ2_0_g128` it is the `PQ2_0` (+2) encoding leaking in, so
/// genuine ternary data never contains it and the Metal SoA upload path
/// rejects any block that does (`metal_graph::reformat::validate_tq2_ternary_codes`,
/// reached from `upload_tq2_weight_soa`). An earlier revision of this helper
/// pushed raw PRNG bytes as `qs`, which made ~25 % of its lanes `0b11` and
/// relied on the old, tolerant `11→0` decode; that contract is gone, so the
/// fixture — not the validation — was wrong.
///
/// To keep the weight matrix "interesting" both the code pattern and the scale
/// vary across blocks, driven by the same 64-bit linear-congruential PRNG seed,
/// so the fixture stays fully deterministic. The helper asserts the result is
/// non-degenerate: all three codes must actually occur, otherwise a
/// seed/modulo change could silently reduce the fixture to a constant matrix
/// and every parity assertion below would still pass while testing nothing.
fn tq2_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 34);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut code_seen = [0usize; 3];
    for _ in 0..num_blocks {
        // 32 bytes of qs = 128 weights × 2 bits, four lanes per byte, LSB
        // first. Each lane is drawn independently and folded into {0, 1, 2}.
        for _ in 0..32 {
            let mut byte = 0u8;
            for lane in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let code = ((state >> 33) % 3) as u8;
                code_seen[code as usize] += 1;
                byte |= code << (2 * lane);
            }
            data.push(byte);
        }
        // FP16 scale in (0.25, 0.75] so RMSNorm output stays in a sane range.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        let scale_bytes = f16::from_f32(scale_f32).to_le_bytes();
        data.extend_from_slice(&scale_bytes);
    }
    assert!(
        code_seen.iter().all(|&n| n > 0),
        "degenerate ternary fixture: code histogram {code_seen:?} (counts of \
         -1/0/+1) — every ternary code must occur or the parity assertions \
         would pass over a near-constant weight matrix"
    );
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

/// Give this test its own Metal session (`MET-08`).
///
/// Every test here drives a `BonsaiModel` through the fused Metal path. That
/// path used to go via a **process-global** `MetalGraph` singleton — one
/// device, one command queue, one pooled buffer set and one GPU KV cache for
/// the whole process — and `cargo test` runs the tests of one integration
/// binary on parallel threads, so two of them interleaved on that single
/// shared graph and one prefill observed another's KV state. It measured as a
/// 3-in-8 failure rate for `test_batched_ternary_prefill_chunked` (`logit[0]`
/// single-shot vs chunked diverging by ~8e-2, well above the 1e-3 tolerance),
/// against 0-in-8 with `--test-threads=1`, and it was held off with a
/// file-local `gpu_serial()` mutex that ran every GPU test one at a time.
///
/// `MET-08` removed the singleton: the device, its pipelines and its weight
/// cache are shared, but the KV cache and every scratch buffer now belong to a
/// **session**. One line per test — this guard — gives each test its own, and
/// the suite passes with full parallelism and **no serialisation at all**.
/// That is the acceptance evidence for `MET-08`; if these tests ever need a
/// lock again, the session split has regressed.
///
/// The guard is inert on a host without a Metal device (`Err`), where every
/// test below early-returns anyway.
fn gpu_session() -> Result<SessionScope, MetalGraphError> {
    MetalGraph::bind_new_session()
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
    let _gpu = gpu_session();
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
    let _gpu = gpu_session();
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
    let _gpu = gpu_session();
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
    let _gpu = gpu_session();
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

// ─────────────────────────────────────────────────────────────────────────────
// Context-length guard regression tests (P1 no-panic-on-long-prompt)
//
// Every Metal prefill / greedy entry point must reject a prompt that overruns
// the model's `max_seq_len` with a clean `Err` rather than panicking on an
// out-of-bounds `RopeTable::cos_at` slice (mirrors the single-token `forward()`
// guard and the CUDA prefill guards). We build the model with a deliberately
// tiny context so a modest batch/position overflows it.
// ─────────────────────────────────────────────────────────────────────────────

/// Tiny context so a 16-token batch (or `pos == max_seq`) overflows it.
const GUARD_MAX_SEQ: usize = 8;

#[test]
fn test_ternary_prefill_context_guard_returns_err() {
    let _gpu = gpu_session();
    let gguf_bytes = build_synthetic_ternary_gguf();
    let gguf = parse_synthetic_gguf(&gguf_bytes);
    let model = BonsaiModel::from_gguf(&gguf, GUARD_MAX_SEQ).expect("BonsaiModel::from_gguf");
    // 16 tokens at pos 0 → 16 > max_seq (8): must be rejected, not panic.
    let token_ids: Vec<u32> = (0..16_u32).map(|i| i % 32).collect();
    let err = model
        .try_metal_prefill_with_lm_head_ternary(&token_ids, 0)
        .expect_err("over-long ternary prefill must return Err, not Ok/panic");
    assert!(
        err.to_string().contains("too long"),
        "expected a context-length error, got: {err}"
    );
}

#[test]
fn test_ternary_prefill_verify_context_guard_returns_err() {
    let _gpu = gpu_session();
    let gguf_bytes = build_synthetic_ternary_gguf();
    let gguf = parse_synthetic_gguf(&gguf_bytes);
    let model = BonsaiModel::from_gguf(&gguf, GUARD_MAX_SEQ).expect("BonsaiModel::from_gguf");
    let token_ids: Vec<u32> = (0..16_u32).map(|i| i % 32).collect();
    let err = model
        .try_metal_prefill_verify_ternary_path(&token_ids, 0)
        .expect_err("over-long ternary prefill-verify must return Err, not Ok/panic");
    assert!(
        err.to_string().contains("too long"),
        "expected a context-length error, got: {err}"
    );
}

#[test]
fn test_ternary_greedy_gpu_context_guard_returns_err() {
    let _gpu = gpu_session();
    let gguf_bytes = build_synthetic_ternary_gguf();
    let gguf = parse_synthetic_gguf(&gguf_bytes);
    let model = BonsaiModel::from_gguf(&gguf, GUARD_MAX_SEQ).expect("BonsaiModel::from_gguf");
    // Valid positions are 0..GUARD_MAX_SEQ; pos == GUARD_MAX_SEQ is out of range.
    let err = model
        .forward_greedy_gpu(1, GUARD_MAX_SEQ)
        .expect_err("greedy decode at pos == max_seq must return Err, not panic");
    assert!(
        err.to_string().contains("too long"),
        "expected a context-length error, got: {err}"
    );
}

/// A within-context greedy call is NOT rejected by the guard: `pos == max_seq-1`
/// is the last valid position, so the guard must let it through (it may still
/// fail later for GPU-availability reasons, which is a *different* error class).
#[test]
fn test_ternary_greedy_gpu_last_valid_pos_not_guarded() {
    let _gpu = gpu_session();
    let gguf_bytes = build_synthetic_ternary_gguf();
    let gguf = parse_synthetic_gguf(&gguf_bytes);
    let model = BonsaiModel::from_gguf(&gguf, GUARD_MAX_SEQ).expect("BonsaiModel::from_gguf");
    if let Err(e) = model.forward_greedy_gpu(1, GUARD_MAX_SEQ - 1) {
        assert!(
            !e.to_string().contains("too long"),
            "last valid position must not trip the context-length guard, got: {e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MET-03 on the real model: the cached ternary shape is byte-identical
// ─────────────────────────────────────────────────────────────────────────────

/// Context the real-model run is loaded with (small: only the device KV cache
/// scales with it, and every sequence below fits).
const REAL_MAX_SEQ: usize = 256;

/// Teacher-forced token ids for the real-model run: deterministic, spread over
/// the vocabulary, never a special-token-only prefix.
fn real_model_tokens(vocab: usize, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| ((i * 7919 + 1013) % vocab.max(1)) as u32)
        .collect()
}

/// `MET-03` acceptance on the real **Ternary-Bonsai-1.7B**
/// (`OXI_MODEL=<path to Ternary-Bonsai-1.7B.gguf>`): the cached ternary shape
/// (`CachedTernaryWeights` — eight GPU handles per layer bound directly) is
/// **byte-identical** to the uncached path it replaces (eight weight-cache
/// lookups per layer per token over a freshly rebuilt
/// `FullForwardLayerParamsTernary`), for
///
/// 1. 12 teacher-forced single-token decode steps — every logit, bit for bit;
/// 2. the greedy token of each step (cached GPU argmax == argmax of those
///    logits);
/// 3. a 9-token batched prefill (last-position logits) and its verify twin
///    (every position's greedy id);
///
/// and building / using the cache uploads **no** GPU buffer beyond the ones
/// the uncached path already made resident.
///
/// The uncached arm is the explicit `*_gpu_ternary_uncached` reference entry
/// of each shape. The model's own ternary routes — `forward_greedy_gpu`,
/// `try_metal_prefill_with_lm_head_ternary` and
/// `try_metal_prefill_verify_ternary_path` — dispatch through the cached
/// entries, so each is also checked against the cached
/// result: the routing change is bit-exact on the real model.
///
/// Every run gets its own Metal session (`MET-08`), so the device KV caches of
/// the "before" and "after" runs cannot interact — and all of those sessions
/// sit on one **isolated** device, so the residency assertions see only this
/// test's uploads even while the synthetic tests above run in parallel. Skips
/// with a capability report when `OXI_MODEL` is unset or the host has no
/// Metal device.
#[test]
fn real_model_cached_ternary_path_is_byte_identical_to_the_uncached_path() {
    let Some(path) = std::env::var_os("OXI_MODEL") else {
        eprintln!(
            "real_model_cached_ternary_path_is_byte_identical_to_the_uncached_path: \
             OXI_MODEL not set — skipping (set OXI_MODEL=<Ternary-Bonsai-1.7B.gguf> to run \
             the MET-03 real-model parity check)"
        );
        return;
    };
    let Ok(device) = oxibonsai_kernels::MetalDevice::isolated() else {
        eprintln!(
            "real_model_cached_ternary_path_is_byte_identical_to_the_uncached_path: no Metal \
             device — skipping"
        );
        return;
    };
    // Run `f` in a fresh session on the isolated device.
    let in_session = |f: &mut dyn FnMut()| {
        let session = MetalGraph::new_session_on(&device);
        MetalGraph::with_session(&session, f);
    };
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&path))
        .expect("mmap OXI_MODEL");
    let gguf = GgufFile::parse(&mmap).expect("parse OXI_MODEL");
    let model = BonsaiModel::from_gguf(&gguf, REAL_MAX_SEQ).expect("load OXI_MODEL");
    in_session(&mut || {
        model
            .get_or_create_gpu_cache()
            .expect("build the GPU weight cache (OXI_MODEL must be a ternary GGUF)");
    });

    let vocab = model.config().vocab_size;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let probe = MetalGraph::new_session_on(&device);
    let resident_before = probe.cached_weight_count().expect("count");
    let bytes_before = probe.bytes_uploaded();

    // 1 + 2: teacher-forced decode.
    let tokens = real_model_tokens(vocab, 12);
    let mut uncached: Vec<Vec<f32>> = Vec::new();
    in_session(&mut || {
        for (pos, &t) in tokens.iter().enumerate() {
            uncached.push(
                model
                    .forward_logits_gpu_ternary_uncached(t, pos)
                    .expect("uncached ternary decode"),
            );
        }
    });
    let mut cached: Vec<Vec<f32>> = Vec::new();
    in_session(&mut || {
        for (pos, &t) in tokens.iter().enumerate() {
            cached.push(
                model
                    .forward_logits_gpu_ternary_cached(t, pos)
                    .expect("cached ternary decode"),
            );
        }
    });
    let mut greedy: Vec<u32> = Vec::new();
    in_session(&mut || {
        for (pos, &t) in tokens.iter().enumerate() {
            greedy.push(
                model
                    .forward_greedy_gpu_ternary_cached(t, pos)
                    .expect("cached ternary greedy"),
            );
        }
    });
    // B4 (a): the model's greedy decode route is the cached entry.
    let mut greedy_routed: Vec<u32> = Vec::new();
    in_session(&mut || {
        for (pos, &t) in tokens.iter().enumerate() {
            greedy_routed.push(
                model
                    .forward_greedy_gpu(t, pos)
                    .expect("routed ternary greedy decode"),
            );
        }
    });
    assert_eq!(
        greedy_routed, greedy,
        "forward_greedy_gpu must decode exactly like the cached ternary entry"
    );
    let mut max_abs_logit = 0f32;
    for (pos, (before, after)) in uncached.iter().zip(&cached).enumerate() {
        assert_eq!(before.len(), vocab, "step {pos}: logits length");
        assert!(
            before.iter().all(|x| x.is_finite()),
            "step {pos}: non-finite logit"
        );
        assert_eq!(
            bits(before),
            bits(after),
            "step {pos}: the cached ternary path diverged from the uncached one"
        );
        let mut best = 0usize;
        for (i, v) in before.iter().enumerate() {
            max_abs_logit = max_abs_logit.max(v.abs());
            if *v > before[best] {
                best = i;
            }
        }
        assert_eq!(
            greedy[pos] as usize, best,
            "step {pos}: cached GPU argmax != argmax of the logits"
        );
    }

    // 3: batched prefill + verify.
    let prompt = real_model_tokens(vocab, 9);
    let mut prefill_before = Vec::new();
    in_session(&mut || {
        prefill_before = model
            .prefill_logits_gpu_ternary_uncached(&prompt, 0)
            .expect("uncached batched prefill");
    });
    let mut prefill_after = Vec::new();
    in_session(&mut || {
        prefill_after = model
            .prefill_logits_gpu_ternary_cached(&prompt, 0)
            .expect("cached batched prefill");
    });
    assert_eq!(prefill_before.len(), vocab);
    assert_eq!(
        bits(&prefill_before),
        bits(&prefill_after),
        "batched prefill: the cached path diverged from the uncached one"
    );
    // B4 (d): the model's strict batched-prefill route is the cached entry.
    let mut prefill_routed = Vec::new();
    in_session(&mut || {
        prefill_routed = model
            .try_metal_prefill_with_lm_head_ternary(&prompt, 0)
            .expect("routed batched prefill");
    });
    assert_eq!(
        bits(&prefill_routed),
        bits(&prefill_after),
        "try_metal_prefill_with_lm_head_ternary must match the cached entry bit for bit"
    );
    let mut verify_before = Vec::new();
    in_session(&mut || {
        verify_before = model
            .prefill_verify_gpu_ternary_uncached(&prompt, 0)
            .expect("uncached verify");
    });
    let mut verify_after = Vec::new();
    in_session(&mut || {
        verify_after = model
            .prefill_verify_gpu_ternary_cached(&prompt, 0)
            .expect("cached verify");
    });
    assert_eq!(verify_before.len(), prompt.len());
    assert_eq!(verify_before, verify_after, "verify ids diverged");
    // B4 (e): the model's strict verify route is the cached entry.
    let mut verify_routed = Vec::new();
    in_session(&mut || {
        verify_routed = model
            .try_metal_prefill_verify_ternary_path(&prompt, 0)
            .expect("routed verify");
    });
    assert_eq!(
        verify_routed, verify_after,
        "try_metal_prefill_verify_ternary_path must match the cached entry"
    );

    let resident_after = probe.cached_weight_count().expect("count");
    assert_eq!(
        resident_after, resident_before,
        "the cached and uncached paths must bind the same resident buffers (no second upload)"
    );
    assert_eq!(probe.bytes_uploaded(), bytes_before);
    eprintln!(
        "MET-03 real-model parity: {} decode steps + {}-token prefill/verify byte-identical \
         (vocab {vocab}, max |logit| {max_abs_logit:.3}); {} resident weight buffers = \
         {:.2} MB of GPU weights, unchanged by every run",
        tokens.len(),
        prompt.len(),
        resident_after,
        bytes_before as f64 / 1e6,
    );
    in_session(&mut || {
        let _ = model.release_metal_weights();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// B5 + C1: the uncached binding is public and keyed on the mapping epoch
// ─────────────────────────────────────────────────────────────────────────────

/// The uncached ternary binding — the decode-throughput A/B's uncached arm —
/// is nameable from outside the crate as
/// `oxibonsai_model::model::{TernaryGpuBinding, TernaryTailBinding}`, and
/// everything it binds is keyed under the model's GGUF-mapping epoch, never
/// the legacy one: every layer's `model_epoch`, and a tail over the model's
/// own final norm and LM head.
#[test]
fn the_uncached_ternary_binding_is_public_and_keyed_on_the_mapping_epoch() {
    let Ok(_gpu) = gpu_session() else {
        eprintln!("skip: no Metal device on this host");
        return;
    };
    let gguf_bytes = build_synthetic_ternary_gguf();
    let gguf = parse_synthetic_gguf(&gguf_bytes);
    let model = BonsaiModel::from_gguf(&gguf, 64).expect("BonsaiModel::from_gguf");
    let epoch = model.gpu_mapping_epoch();
    assert_ne!(
        epoch,
        oxibonsai_kernels::LEGACY_MODEL_EPOCH,
        "a loaded model keys its GPU buffers under its own mapping epoch"
    );
    {
        let binding: TernaryGpuBinding<'_> = model
            .ternary_gpu_binding()
            .expect("a ternary model binds on a Metal host");
        assert_eq!(binding.layer_params.len(), model.num_layers());
        for (layer, params) in binding.layer_params.iter().enumerate() {
            assert_eq!(
                params.model_epoch, epoch,
                "layer {layer} must be keyed under the mapping epoch"
            );
        }
        let tail: &TernaryTailBinding<'_> = binding
            .tail
            .as_ref()
            .expect("a ternary LM head binds the GPU tail");
        let config = model.config();
        assert_eq!(tail.lm_head_out_features, config.vocab_size);
        assert_eq!(tail.final_norm_bytes.len(), config.hidden_size);
        assert_eq!(
            tail.lm_head_bytes.len(),
            config.vocab_size * config.hidden_size / 128 * 34,
            "the tail borrows the whole TQ2_0_g128 LM head"
        );
    }
    let _ = model.release_metal_weights();
}

// ─────────────────────────────────────────────────────────────────────────────
// C2 (the M-21 residue) on the real model: what `upload_weights_to_gpu` keeps
// ─────────────────────────────────────────────────────────────────────────────

/// The seven projections `TransformerBlock::upload_to_gpu` uploads one by one.
const BLOCK_PROJECTIONS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

/// On the real **Ternary-Bonsai-1.7B**
/// (`OXI_MODEL=<path to Ternary-Bonsai-1.7B.gguf>`): on Metal,
/// `BonsaiModel::upload_weights_to_gpu` keeps exactly the model's own tensors
/// resident in the kernel's weight cache — the seven projections of every
/// layer plus the LM head, byte for byte — and no longer the per-layer Q‖K‖V
/// and gate‖up concatenations, which nothing on Metal reads (the block's
/// fused arms build their own copy in `MetalGraph`'s cache, the one the model
/// cache shares). The upload puts nothing in `MetalGraph`'s cache.
///
/// "Before" is measured, not estimated: replaying on top of it the exact two
/// `upload_weights_ternary` calls per layer that the pre-C2 `upload_to_gpu`
/// made grows the resident bytes by exactly the concatenations' size — what
/// every Metal caller of `upload_weights_to_gpu` used to pay. Everything is
/// uploaded inside one attribution scope and released at the end, so the
/// process-wide cache returns to where it started.
///
/// Skips with a capability report when `OXI_MODEL` is unset, the host has no
/// accelerated GPU backend or Metal device, or the model is not ternary.
#[test]
fn real_model_upload_weights_to_gpu_keeps_no_ternary_concatenation_on_metal() {
    use oxibonsai_core::BlockTQ2_0_g128;
    use oxibonsai_kernels::gpu_backend::{
        next_gpu_model_epoch, release_model_weights, resident_weight_bytes, GpuUploadScope,
    };
    use oxibonsai_kernels::TernaryKernel;
    use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

    const TEST: &str = "metal_prefill_ternary_parity_tests::\
                        real_model_upload_weights_to_gpu_keeps_no_ternary_concatenation_on_metal";
    let Some(path) = std::env::var_os("OXI_MODEL").filter(|p| !p.is_empty()) else {
        eprintln!(
            "capability report: {TEST} SKIPPED -- $OXI_MODEL is not set (point it at \
             models/Ternary-Bonsai-1.7B.gguf to run the C2 real-model upload check)"
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };
    let Ok(kernel) = KernelDispatcher::try_with_tier(KernelTier::Gpu) else {
        eprintln!("capability report: {TEST} SKIPPED -- no accelerated GPU backend");
        record_skipped(Capability::Metal, TEST);
        return;
    };
    let Ok(device) = oxibonsai_kernels::MetalDevice::isolated() else {
        eprintln!("capability report: {TEST} SKIPPED -- no Metal device");
        record_skipped(Capability::Metal, TEST);
        return;
    };
    let gate_start = std::time::Instant::now();
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&path))
        .expect("mmap OXI_MODEL");
    let gguf = GgufFile::parse(&mmap).expect("parse OXI_MODEL");
    let mut model = BonsaiModel::from_gguf(&gguf, REAL_MAX_SEQ).expect("load OXI_MODEL");
    if model.dominant_quant_type() != oxibonsai_core::GgufTensorType::TQ2_0_g128 {
        eprintln!(
            "capability report: {TEST} SKIPPED -- $OXI_MODEL is not a TQ2_0_g128 model (point it \
             at models/Ternary-Bonsai-1.7B.gguf)"
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }
    let tensor = |name: &str| -> &[u8] {
        gguf.tensor_data(name)
            .unwrap_or_else(|e| panic!("OXI_MODEL tensor {name}: {e}"))
    };
    let n_layers = model.num_layers();
    let mut own_tensors = tensor("output.weight").len() as u64;
    let mut concatenations = 0u64;
    for layer in 0..n_layers {
        for name in BLOCK_PROJECTIONS {
            own_tensors += tensor(&format!("blk.{layer}.{name}.weight")).len() as u64;
        }
        for name in ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"] {
            concatenations += tensor(&format!("blk.{layer}.{name}.weight")).len() as u64;
        }
    }

    let epoch = next_gpu_model_epoch();
    let session = MetalGraph::new_session_on(&device);
    let resident_start = resident_weight_bytes(&kernel);
    let (after_c2, before_c2, upload_stats) = {
        let scope = GpuUploadScope::enter(epoch);
        MetalGraph::with_session(&session, || model.upload_weights_to_gpu(&kernel));
        let upload_stats = scope.stats();
        let after_c2 = resident_weight_bytes(&kernel) - resident_start;

        // Replay what the pre-C2 `upload_to_gpu` additionally uploaded.
        let blocks = |name: String| -> &[BlockTQ2_0_g128] {
            BlockTQ2_0_g128::slice_from_bytes(tensor(&name))
                .unwrap_or_else(|e| panic!("{name} as TQ2_0_g128 blocks: {e}"))
        };
        for layer in 0..n_layers {
            let part = |name: &str| blocks(format!("blk.{layer}.{name}.weight"));
            let qkv = [part("attn_q"), part("attn_k"), part("attn_v")].concat();
            let gate_up = [part("ffn_gate"), part("ffn_up")].concat();
            kernel
                .upload_weights_ternary(&qkv)
                .expect("replayed pre-C2 Q‖K‖V upload");
            kernel
                .upload_weights_ternary(&gate_up)
                .expect("replayed pre-C2 gate‖up upload");
        }
        let before_c2 = resident_weight_bytes(&kernel) - resident_start;
        (after_c2, before_c2, upload_stats)
    };

    assert_eq!(
        upload_stats.fresh_buffers,
        n_layers * BLOCK_PROJECTIONS.len() + 1,
        "upload_weights_to_gpu uploads the seven projections per layer and the LM head — no \
         concatenation"
    );
    assert_eq!(
        after_c2, own_tensors,
        "upload_weights_to_gpu must keep exactly the model's own tensors resident"
    );
    assert_eq!(upload_stats.fresh_bytes, own_tensors);
    assert_eq!(
        session.bytes_uploaded(),
        0,
        "upload_weights_to_gpu puts nothing in MetalGraph's cache"
    );
    assert_eq!(
        before_c2 - after_c2,
        concatenations,
        "the pre-C2 concatenation uploads cost exactly the Q‖K‖V + gate‖up bytes"
    );
    eprintln!(
        "C2 real-model upload_weights_to_gpu ({n_layers} layers): kernel weight cache resident \
         +{:.2} MB after C2 vs +{:.2} MB before C2 (the {:.2} MB of per-layer Q‖K‖V + gate‖up \
         concatenations are no longer uploaded on Metal); MetalGraph resident +{} B",
        after_c2 as f64 / 1e6,
        before_c2 as f64 / 1e6,
        concatenations as f64 / 1e6,
        session.bytes_uploaded(),
    );

    let released = release_model_weights(&kernel, epoch).expect("release the test's uploads");
    assert_eq!(released, n_layers * (BLOCK_PROJECTIONS.len() + 2) + 1);
    assert_eq!(
        resident_weight_bytes(&kernel),
        resident_start,
        "releasing the scope's epoch frees everything this test uploaded"
    );
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, TEST, elapsed);
    record_executed_timed(Capability::LegacyModels, TEST, elapsed);
}
