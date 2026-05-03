//! Parity tests for the batched Metal **Q1_0_g128** prefill path.
//!
//! These tests verify that the batched Q1 prefill GEMM kernels
//! (`MSL_GEMM_Q1_G128_V7`, `MSL_GEMM_Q1_G128_V7_RESIDUAL` and
//! `MSL_FUSED_GATE_UP_SWIGLU_GEMM_Q1`) produce the **same** logits as the
//! per-position single-token Metal forward — across **all** positions, for
//! arbitrary batch sizes.
//!
//! ## Background
//!
//! Versions 0.1.0–0.1.4 of the Q1 batched prefill kernels silently capped
//! `cols = min(batch_size, 8u)`, then iterated columns `0..cols` in their
//! inner reduction loop and final write-back. With
//! `ChunkedPrefillConfig::chunk_size = 512` the GPU was therefore asked to
//! process up to 512 columns per chunk, but only the first 8 columns
//! received any contribution — columns 8..N were left as the zero-initialised
//! buffer values, so any prompt longer than 8 tokens silently produced wrong
//! logits.
//!
//! [`test_batched_q1_prefill_matches_per_position_batch12`] is the
//! regression test that would have caught this bug. The 8-token sanity
//! variant exists because the cap was non-pathological at exactly 8.
//!
//! ## Strict path
//!
//! Like the TQ2 parity suite (see
//! `metal_prefill_ternary_parity_tests.rs`), these tests deliberately
//! bypass [`BonsaiModel::forward_prefill`] and invoke
//! [`BonsaiModel::try_metal_prefill_with_lm_head`] directly so any GPU
//! dispatch failure is surfaced as a hard test failure rather than masked
//! by the silent fallback to per-token sequential forward.

#![cfg(all(feature = "metal", target_os = "macos"))]

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::KernelDispatcher;
use oxibonsai_model::model::BonsaiModel;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic Q1_0_g128 fixture
// ─────────────────────────────────────────────────────────────────────────────

/// Build a Q1_0_g128 weight blob with deterministic per-block patterns.
///
/// Each block is **18 bytes**, in the on-disk AoS layout that the GGUF
/// reader expects (matching `BlockQ1_0G128 { d: f16, qs: [u8; 16] }`):
/// `[scale 2B FP16 LE][qs 16B (128 sign bits, LSB-first)]`.
///
/// `bit==1 → +d`, `bit==0 → -d` (see
/// [`oxibonsai_core::tensor::BlockQ1_0G128`]).
///
/// To get a reasonably "interesting" weight matrix we vary both the qs
/// pattern and the scale across blocks based on a 64-bit linear-congruential
/// PRNG seed.
fn q1_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 18);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_blocks {
        // FP16 scale in (0.25, 0.75] so RMSNorm output stays in a sane range.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        let scale_bytes = f16::from_f32(scale_f32).to_le_bytes();
        data.extend_from_slice(&scale_bytes);
        // 16 bytes of qs (128 sign bits, 1 bit per weight, LSB-first byte order).
        for _ in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
    }
    data
}

/// Build an FP32 tensor whose values vary with the index — keeps the
/// embedding path away from degenerate identity inputs while still being
/// deterministic.
fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let phase = (i as f32) * 0.013_f32;
        let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// Build a synthetic fully-Q1 GGUF for `BonsaiModel::from_gguf`.
///
/// All projection matrices (attention QKV, attention output, FFN gate/up/down,
/// LM head) are stored as `Q1_0G128` so the model exercises the all-1-bit
/// GPU path end-to-end. RMSNorm weights and token embeddings stay FP32.
fn build_synthetic_q1_gguf() -> Vec<u8> {
    let h: usize = 128; // hidden_size — must be ≥ 128 for Q1_0_g128
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
        MetadataWriteValue::Str("PrefillQ1ParityTest".to_string()),
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

    // Token embedding: vocab × hidden (FP32, varied pattern).
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

    // Output projection (LM head): Q1_0G128.
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::Q1_0G128,
        data: q1_0_g128_pattern(vocab * h, 0xCAFE_BABE),
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

        // Attention QKV / output: all Q1_0G128.
        let layer_seed = 0x2000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_q.weight"),
            shape: vec![h as u64, (nq * hd) as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(nq * hd * h, layer_seed),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_k.weight"),
            shape: vec![h as u64, (nkv * hd) as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(nkv * hd * h, layer_seed.wrapping_add(1)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_v.weight"),
            shape: vec![h as u64, (nkv * hd) as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(nkv * hd * h, layer_seed.wrapping_add(2)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_output.weight"),
            shape: vec![(nq * hd) as u64, h as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(h * nq * hd, layer_seed.wrapping_add(3)),
        });

        // FFN gate / up / down: all Q1_0G128.
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_gate.weight"),
            shape: vec![h as u64, inter as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(inter * h, layer_seed.wrapping_add(4)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_up.weight"),
            shape: vec![h as u64, inter as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(inter * h, layer_seed.wrapping_add(5)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_down.weight"),
            shape: vec![inter as u64, h as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(h * inter, layer_seed.wrapping_add(6)),
        });
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

/// Write the synthetic GGUF to a unique file under `temp_dir()` and return
/// the path. The caller is responsible for deleting the file when done
/// (we use a stable filename keyed by a per-test suffix so re-running the
/// test suite does not pile up garbage).
fn write_temp_gguf(suffix: &str) -> std::path::PathBuf {
    let bytes = build_synthetic_q1_gguf();
    let mut path = std::env::temp_dir();
    path.push(format!("oxibonsai_metal_prefill_q1_{suffix}.gguf"));
    std::fs::write(&path, &bytes).expect("write synthetic GGUF to temp dir");
    path
}

/// Parse the synthetic Q1 GGUF, returning the parser so the caller can
/// borrow from it when constructing a [`BonsaiModel`].
fn parse_synthetic_gguf(gguf_bytes: &[u8]) -> GgufFile<'_> {
    GgufFile::parse(gguf_bytes).expect("GgufFile::parse synthetic")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Logits parity at `batch_size = 8` — sanity check.
///
/// At exactly 8 tokens the buggy `cols = min(batch_size, 8u)` was
/// non-pathological (it equalled `batch_size`), so this case used to pass
/// even with the bug present. We keep it as a regression guard against
/// regressions on the boundary value.
#[test]
fn test_batched_q1_prefill_matches_per_position_batch8() {
    let path = write_temp_gguf("parity_8");
    let gguf_bytes = std::fs::read(&path).expect("read synthetic GGUF");
    let _ = std::fs::remove_file(&path);
    // The Q1 prefill path requires GPU-resident weight handles (uploaded via
    // `BonsaiModel::upload_weights_to_gpu`), which in turn require a
    // dispatcher constructed with a live GPU backend — `auto_detect` is the
    // only public constructor that wires `Scirs2BackendHandle` (the real
    // Metal backend on macOS). If no GPU is available the test is a
    // no-op-equivalent because the upload silently produces no handles and
    // the strict prefill path returns an error early; we surface that as a
    // failure rather than silently skipping.
    let kernel = Arc::new(KernelDispatcher::auto_detect());

    let token_ids: Vec<u32> = (0..8_u32).map(|i| (i * 3 + 1) % 32).collect();

    // ── Reference path: sequential per-position forward ──────────────────
    let ref_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let mut model =
            BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf reference");
        model.upload_weights_to_gpu(kernel.as_ref());
        let mut last = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            last = model
                .forward(tid, i, kernel.as_ref())
                .expect("sequential forward");
        }
        last
    };

    // ── Batched Q1 prefill — STRICT path (no fallback masking) ────────────
    let prefill_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let mut model =
            BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf prefill");
        model.upload_weights_to_gpu(kernel.as_ref());
        model
            .try_metal_prefill_with_lm_head(&token_ids, 0)
            .expect("strict q1 prefill batch8")
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
            "logit[{i}] mismatch (batch=8): ref={a:.6}, prefill={b:.6}, abs_err={abs_err:.3e}, rel_err={rel_err:.3e}"
        );
    }
    eprintln!(
        "test_batched_q1_prefill_matches_per_position_batch8: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
    );
}

/// Logits parity at `batch_size = 12` — **regression test** for the
/// cap-of-8 bug shipped in 0.1.0–0.1.4.
///
/// With 12 prompt tokens, the buggy kernel iterated only columns 0..7 in
/// its weight-tiled inner loop and final `simd_sum` write-back, leaving
/// columns 8..11 as the zero-initialised buffer values from the previous
/// dispatch (== silently wrong logits). With the fix in place, the kernel
/// makes a second outer iteration (`col_base = 8`) covering columns 8..11
/// and the final logits match the per-position reference.
///
/// The 12-token prompt deliberately exceeds the 8-column simdgroup chunk
/// size to exercise both `col_base = 0u` (cols=8) and `col_base = 8u`
/// (cols=4) iterations of the new outer loop.
#[test]
fn test_batched_q1_prefill_matches_per_position_batch12() {
    let path = write_temp_gguf("parity_12");
    let gguf_bytes = std::fs::read(&path).expect("read synthetic GGUF");
    let _ = std::fs::remove_file(&path);
    // See `test_batched_q1_prefill_matches_per_position_batch8` for why we
    // use `auto_detect` + `upload_weights_to_gpu` rather than the reference
    // tier here.
    let kernel = Arc::new(KernelDispatcher::auto_detect());

    let token_ids: Vec<u32> = (0..12_u32).map(|i| (i * 3 + 1) % 32).collect();

    let ref_logits = {
        let gguf = parse_synthetic_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf ref");
        model.upload_weights_to_gpu(kernel.as_ref());
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
        let mut model = BonsaiModel::from_gguf(&gguf, 512).expect("BonsaiModel::from_gguf prefill");
        model.upload_weights_to_gpu(kernel.as_ref());
        model
            .try_metal_prefill_with_lm_head(&token_ids, 0)
            .expect("strict q1 prefill batch12")
    };

    assert_eq!(ref_logits.len(), prefill_logits.len());
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
            "logit[{i}] mismatch (batch=12): ref={a:.6} prefill={b:.6} abs_err={abs_err:.3e} rel_err={rel_err:.3e}"
        );
    }
    eprintln!(
        "test_batched_q1_prefill_matches_per_position_batch12: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
    );
}
