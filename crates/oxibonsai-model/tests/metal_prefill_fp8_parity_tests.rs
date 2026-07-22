//! Parity tests for the hybrid Metal FP8 (E4M3 / E5M2) batch-prefill path.
//!
//! These tests verify that
//! [`BonsaiModel::try_metal_prefill_with_lm_head_fp8`] — which batches the FP8
//! linear projections through the Phase 28 `metal_fp8_prefill` GEMM kernels
//! while running attention + the K/V store on the CPU against `self.kv_cache`
//! — produces logits that match the **sequential per-token** FP8 forward, and,
//! crucially, that it leaves real (non-zero) prompt K/V in the *same*
//! `self.kv_cache` the per-token decode path reads. This is the guard against
//! the split-KV-cache trap that keeps the CUDA FP8 prefill disabled.
//!
//! The fixture is a synthetic 2-layer fully-FP8 GGUF model assembled in
//! `std::env::temp_dir()`, small enough to load quickly and large enough to
//! exercise the QKV / attn-output / gate+up / down GEMM paths for both FP8
//! numeric formats.

#![cfg(all(feature = "metal", target_os = "macos"))]

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic FP8 fixture
// ─────────────────────────────────────────────────────────────────────────────

const H: usize = 128; // hidden_size (multiple of 32 for FP8 blocks)
const INTER: usize = 256; // intermediate_size
const NUM_LAYERS: usize = 2;
const NQ: usize = 4;
const NKV: usize = 2;
const HD: usize = 32; // head_dim = H / NQ
const VOCAB: usize = 32;
const CTX: usize = 512;

/// Deterministic small f32 weights in `[-0.1, 0.1)`, varied by index + seed.
///
/// Small magnitudes keep FP8 quantization well away from the E4M3/E5M2
/// saturation range while still producing an "interesting" (non-degenerate)
/// weight matrix so different tokens map to distinct hidden states.
fn weight_pattern_f32(num_weights: usize, seed: u64) -> Vec<f32> {
    let mut vals = Vec::with_capacity(num_weights);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_weights {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = ((state >> 40) as u32 as f32) / ((1u32 << 24) as f32); // [0,1)
        vals.push((u - 0.5) * 0.2);
    }
    vals
}

/// Reinterpret quantized FP8 E4M3 blocks as their raw 34-byte-per-block bytes.
fn e4m3_blocks_to_bytes(blocks: &[oxibonsai_core::BlockFP8E4M3]) -> Vec<u8> {
    // SAFETY: `BlockFP8E4M3` is `#[repr(C)]` with `size_of == BLOCK_FP8_BYTES`.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
        )
    };
    bytes.to_vec()
}

fn e5m2_blocks_to_bytes(blocks: &[oxibonsai_core::BlockFP8E5M2]) -> Vec<u8> {
    // SAFETY: `BlockFP8E5M2` is `#[repr(C)]` with `size_of == BLOCK_FP8_BYTES`.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr().cast::<u8>(),
            blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
        )
    };
    bytes.to_vec()
}

/// Quantize a deterministic weight pattern into FP8 block bytes for the GGUF.
fn fp8_pattern(is_e4m3: bool, num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(num_weights % 32, 0, "num_weights must be a multiple of 32");
    let vals = weight_pattern_f32(num_weights, seed);
    if is_e4m3 {
        let blocks =
            oxibonsai_core::BlockFP8E4M3::quantize(&vals).expect("FP8 E4M3 quantize fixture");
        e4m3_blocks_to_bytes(&blocks)
    } else {
        let blocks =
            oxibonsai_core::BlockFP8E5M2::quantize(&vals).expect("FP8 E5M2 quantize fixture");
        e5m2_blocks_to_bytes(&blocks)
    }
}

/// Token-embedding table `[vocab × hidden]` with a strong **per-token**
/// signature so different prompt tokens (and generated tokens) drive the
/// model to distinct hidden states — otherwise a tiny random model collapses
/// to a constant greedy trajectory, making the continuation test a weak
/// discriminator.
fn token_embd_pattern(vocab: usize, hidden: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(vocab * hidden * 4);
    for tok in 0..vocab {
        for j in 0..hidden {
            let tf = tok as f32;
            let jf = j as f32;
            // Distinct frequency + phase per token so rows are well separated.
            let val = 0.9_f32 * ((0.30 * jf + 0.7 * tf).sin() + 0.5 * (0.11 * tf * jf).cos());
            v.extend_from_slice(&val.to_le_bytes());
        }
    }
    v
}

/// FP32 tensor whose values vary with the index (embeddings / norm weights).
fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let phase = (i as f32) * 0.013_f32;
        let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// Build a synthetic fully-FP8 GGUF for `BonsaiModel::from_gguf`.
///
/// All projection matrices (attention QKV, attention output, FFN gate/up/down,
/// LM head) are stored as F8_E4M3 or F8_E5M2 so the model exercises the FP8
/// prefill path end-to-end. RMSNorm weights and token embeddings stay FP32.
fn build_synthetic_fp8_gguf(is_e4m3: bool) -> Vec<u8> {
    let ttype = if is_e4m3 {
        TensorType::F8_E4M3
    } else {
        TensorType::F8_E5M2
    };

    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("PrefillFp8ParityTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(H as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(NUM_LAYERS as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(NQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(NKV as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(INTER as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
    writer.add_metadata("qwen3.context_length", MetadataWriteValue::U32(CTX as u32));
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![H as u64, VOCAB as u64],
        tensor_type: TensorType::F32,
        data: token_embd_pattern(VOCAB, H),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![H as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(H, 1.0),
    });
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![H as u64, VOCAB as u64],
        tensor_type: ttype,
        data: fp8_pattern(is_e4m3, VOCAB * H, 0xCAFE_BABE),
    });

    for layer in 0..NUM_LAYERS {
        let pfx = format!("blk.{layer}");
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_norm.weight"),
            shape: vec![H as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(H, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_norm.weight"),
            shape: vec![H as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(H, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_q_norm.weight"),
            shape: vec![HD as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(HD, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_k_norm.weight"),
            shape: vec![HD as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(HD, 1.0),
        });

        let seed = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_q.weight"),
            shape: vec![H as u64, (NQ * HD) as u64],
            tensor_type: ttype,
            data: fp8_pattern(is_e4m3, NQ * HD * H, seed),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_k.weight"),
            shape: vec![H as u64, (NKV * HD) as u64],
            tensor_type: ttype,
            data: fp8_pattern(is_e4m3, NKV * HD * H, seed.wrapping_add(1)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_v.weight"),
            shape: vec![H as u64, (NKV * HD) as u64],
            tensor_type: ttype,
            data: fp8_pattern(is_e4m3, NKV * HD * H, seed.wrapping_add(2)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.attn_output.weight"),
            shape: vec![(NQ * HD) as u64, H as u64],
            tensor_type: ttype,
            data: fp8_pattern(is_e4m3, H * NQ * HD, seed.wrapping_add(3)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_gate.weight"),
            shape: vec![H as u64, INTER as u64],
            tensor_type: ttype,
            data: fp8_pattern(is_e4m3, INTER * H, seed.wrapping_add(4)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_up.weight"),
            shape: vec![H as u64, INTER as u64],
            tensor_type: ttype,
            data: fp8_pattern(is_e4m3, INTER * H, seed.wrapping_add(5)),
        });
        writer.add_tensor(TensorEntry {
            name: format!("{pfx}.ffn_down.weight"),
            shape: vec![INTER as u64, H as u64],
            tensor_type: ttype,
            data: fp8_pattern(is_e4m3, H * INTER, seed.wrapping_add(6)),
        });
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

fn parse_gguf(bytes: &[u8]) -> GgufFile<'_> {
    GgufFile::parse(bytes).expect("GgufFile::parse synthetic FP8")
}

fn cpu_kernel() -> Arc<KernelDispatcher> {
    Arc::new(KernelDispatcher::with_tier(KernelTier::Reference))
}

/// Cosine similarity between two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (j, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = j as u32;
        }
    }
    best_idx
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) Logit parity: CPU-sequential vs Metal-batch FP8 prefill
// ─────────────────────────────────────────────────────────────────────────────

fn run_logit_parity(is_e4m3: bool, batch: usize) {
    let gguf_bytes = build_synthetic_fp8_gguf(is_e4m3);
    let kernel = cpu_kernel();
    let token_ids: Vec<u32> = (0..batch as u32)
        .map(|i| (i * 3 + 1) % VOCAB as u32)
        .collect();

    // Reference: sequential per-token forward on a CPU kernel.
    let ref_logits = {
        let gguf = parse_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, CTX).expect("from_gguf ref");
        let mut last = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            last = model
                .forward(tid, i, kernel.as_ref())
                .expect("sequential forward");
        }
        last
    };

    // Metal FP8 batch prefill — STRICT path (no fallback masking).
    let prefill_logits = {
        let gguf = parse_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, CTX).expect("from_gguf prefill");
        model
            .try_metal_prefill_with_lm_head_fp8(&token_ids, 0, is_e4m3)
            .expect("strict FP8 metal prefill")
    };

    assert_eq!(ref_logits.len(), prefill_logits.len(), "logit dim mismatch");
    for (i, (a, b)) in ref_logits.iter().zip(prefill_logits.iter()).enumerate() {
        assert!(a.is_finite(), "ref_logits[{i}] not finite");
        assert!(b.is_finite(), "prefill_logits[{i}] not finite");
    }
    let cos = cosine(&ref_logits, &prefill_logits);
    assert!(
        cos >= 0.999,
        "FP8 (e4m3={is_e4m3}, batch={batch}) prefill logits cos={cos:.6} < 0.999"
    );
    assert_eq!(
        argmax(&ref_logits),
        argmax(&prefill_logits),
        "FP8 (e4m3={is_e4m3}, batch={batch}) last-position argmax disagrees"
    );
    eprintln!("run_logit_parity e4m3={is_e4m3} batch={batch}: cos={cos:.6}");
}

#[test]
fn fp8_e4m3_prefill_logits_match_sequential() {
    run_logit_parity(true, 20);
}

#[test]
fn fp8_e5m2_prefill_logits_match_sequential() {
    run_logit_parity(false, 20);
}

/// Cap-of-8 discriminator: batch = 12 (> 8) exercises the kernel's multi-chunk
/// outer column loop, guaranteeing the FP8 GEMM does not silently cap at 8.
#[test]
fn fp8_e4m3_prefill_logits_match_sequential_batch12() {
    run_logit_parity(true, 12);
}

#[test]
fn fp8_e5m2_prefill_logits_match_sequential_batch12() {
    run_logit_parity(false, 12);
}

// ─────────────────────────────────────────────────────────────────────────────
// (a) KV-cache population: every prompt position has non-zero K after prefill
// ─────────────────────────────────────────────────────────────────────────────

fn run_kv_populated(is_e4m3: bool) {
    let gguf_bytes = build_synthetic_fp8_gguf(is_e4m3);
    let batch = 24usize;
    let token_ids: Vec<u32> = (0..batch as u32)
        .map(|i| (i * 7 + 3) % VOCAB as u32)
        .collect();

    let gguf = parse_gguf(&gguf_bytes);
    let mut model = BonsaiModel::from_gguf(&gguf, CTX).expect("from_gguf kv");
    model
        .try_metal_prefill_with_lm_head_fp8(&token_ids, 0, is_e4m3)
        .expect("strict FP8 metal prefill");

    // Every prompt position, every layer, every KV head must have a non-zero
    // key vector in `self.kv_cache` — the same cache the decode path reads.
    let cache = model.kv_cache();
    for layer in 0..NUM_LAYERS {
        for head in 0..NKV {
            let keys = cache.keys_for(layer, head, batch);
            assert_eq!(keys.len(), batch * HD, "keys_for length");
            for pos in 0..batch {
                let k = &keys[pos * HD..(pos + 1) * HD];
                let norm: f32 = k.iter().map(|x| x * x).sum();
                assert!(
                    norm > 0.0 && norm.is_finite(),
                    "layer {layer} head {head} pos {pos}: zero/non-finite K (norm={norm})"
                );
            }
        }
    }
    eprintln!("run_kv_populated e4m3={is_e4m3}: {batch} positions × {NUM_LAYERS} layers OK");
}

#[test]
fn fp8_e4m3_prefill_populates_kv_cache() {
    run_kv_populated(true);
}

#[test]
fn fp8_e5m2_prefill_populates_kv_cache() {
    run_kv_populated(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) Greedy continuation: CPU-sequential vs Metal-batch prefill must decode
//     identical tokens for ≥ 16 steps (the prefilled `self.kv_cache` is read
//     by the same per-token decode path in both runs).
// ─────────────────────────────────────────────────────────────────────────────

fn run_greedy_continuation(is_e4m3: bool) {
    let gguf_bytes = build_synthetic_fp8_gguf(is_e4m3);
    let kernel = cpu_kernel();
    let prompt_len = 20usize;
    let decode_steps = 16usize;
    let token_ids: Vec<u32> = (0..prompt_len as u32)
        .map(|i| (i * 3 + 1) % VOCAB as u32)
        .collect();

    // Reference: sequential prefill, then greedy decode.
    let ref_gen: Vec<u32> = {
        let gguf = parse_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, CTX).expect("from_gguf ref gen");
        let mut logits = Vec::new();
        for (i, &tid) in token_ids.iter().enumerate() {
            logits = model.forward(tid, i, kernel.as_ref()).expect("seq forward");
        }
        let mut gen = Vec::with_capacity(decode_steps);
        let mut next = argmax(&logits);
        for step in 0..decode_steps {
            gen.push(next);
            let pos = prompt_len + step;
            logits = model
                .forward(next, pos, kernel.as_ref())
                .expect("seq decode");
            next = argmax(&logits);
        }
        gen
    };

    // Metal-batch prefill, then greedy decode (same CPU decode path).
    let metal_gen: Vec<u32> = {
        let gguf = parse_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, CTX).expect("from_gguf metal gen");
        let logits = model
            .try_metal_prefill_with_lm_head_fp8(&token_ids, 0, is_e4m3)
            .expect("strict FP8 metal prefill");
        let mut gen = Vec::with_capacity(decode_steps);
        let mut next = argmax(&logits);
        let mut logits = logits;
        for step in 0..decode_steps {
            gen.push(next);
            let pos = prompt_len + step;
            logits = model
                .forward(next, pos, kernel.as_ref())
                .expect("metal decode");
            next = argmax(&logits);
        }
        let _ = logits;
        gen
    };

    assert_eq!(
        ref_gen, metal_gen,
        "FP8 (e4m3={is_e4m3}) greedy continuation diverged: seq={ref_gen:?} metal={metal_gen:?}"
    );
    eprintln!(
        "run_greedy_continuation e4m3={is_e4m3}: {decode_steps} tokens identical {ref_gen:?}"
    );
}

#[test]
fn fp8_e4m3_prefill_greedy_matches_sequential() {
    run_greedy_continuation(true);
}

#[test]
fn fp8_e5m2_prefill_greedy_matches_sequential() {
    run_greedy_continuation(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunked prefill: two chunks that write `self.kv_cache` must match a single
// prefill's final logits (exercises `pos_start > 0` + cross-chunk attention).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn fp8_e4m3_prefill_chunked_matches_single_shot() {
    let gguf_bytes = build_synthetic_fp8_gguf(true);
    let token_ids: Vec<u32> = (0..12_u32).map(|i| (i * 5 + 2) % VOCAB as u32).collect();

    let single = {
        let gguf = parse_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, CTX).expect("from_gguf single");
        model
            .try_metal_prefill_with_lm_head_fp8(&token_ids, 0, true)
            .expect("single-shot FP8 prefill")
    };
    let chunked = {
        let gguf = parse_gguf(&gguf_bytes);
        let mut model = BonsaiModel::from_gguf(&gguf, CTX).expect("from_gguf chunked");
        let _ = model
            .try_metal_prefill_with_lm_head_fp8(&token_ids[0..6], 0, true)
            .expect("first chunk");
        model
            .try_metal_prefill_with_lm_head_fp8(&token_ids[6..12], 6, true)
            .expect("second chunk")
    };

    assert_eq!(single.len(), chunked.len());
    let cos = cosine(&single, &chunked);
    assert!(cos >= 0.999, "chunked vs single-shot cos={cos:.6} < 0.999");
    assert_eq!(
        argmax(&single),
        argmax(&chunked),
        "chunked argmax disagrees"
    );
    eprintln!("fp8_e4m3_prefill_chunked_matches_single_shot: cos={cos:.6}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Context-length guard: an over-long prompt must return a clean `Err`
// ("too long"), never panic on an out-of-bounds RoPE slice.
// ─────────────────────────────────────────────────────────────────────────────

const GUARD_MAX_SEQ: usize = 8;

#[test]
fn fp8_e4m3_prefill_context_guard_returns_err() {
    let gguf_bytes = build_synthetic_fp8_gguf(true);
    let gguf = parse_gguf(&gguf_bytes);
    let mut model = BonsaiModel::from_gguf(&gguf, GUARD_MAX_SEQ).expect("from_gguf guard");
    let token_ids: Vec<u32> = (0..16_u32).map(|i| i % VOCAB as u32).collect();
    let err = model
        .try_metal_prefill_with_lm_head_fp8(&token_ids, 0, true)
        .expect_err("over-long FP8 prefill must return Err, not Ok/panic");
    assert!(
        err.to_string().contains("too long"),
        "expected a context-length error, got: {err}"
    );
}

#[test]
fn fp8_e5m2_prefill_context_guard_returns_err() {
    let gguf_bytes = build_synthetic_fp8_gguf(false);
    let gguf = parse_gguf(&gguf_bytes);
    let mut model = BonsaiModel::from_gguf(&gguf, GUARD_MAX_SEQ).expect("from_gguf guard");
    let token_ids: Vec<u32> = (0..16_u32).map(|i| i % VOCAB as u32).collect();
    let err = model
        .try_metal_prefill_with_lm_head_fp8(&token_ids, 0, false)
        .expect_err("over-long FP8 prefill must return Err, not Ok/panic");
    assert!(
        err.to_string().contains("too long"),
        "expected a context-length error, got: {err}"
    );
}
