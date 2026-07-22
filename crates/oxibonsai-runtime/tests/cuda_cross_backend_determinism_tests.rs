//! Cross-backend (CPU vs CUDA) determinism guard — the CUDA sibling of
//! `cross_backend_determinism_tests.rs`'s Metal guard.
//!
//! README.md's "CPU and GPU produce byte-identical output at temperature 0 /
//! seed 42" contract is written against Metal, but the same [`KernelTier::Gpu`]
//! dispatch mechanism backs CUDA on this platform, so we pin the equivalent
//! guard here: the *same* [`InferenceEngine`] entry point
//! (`InferenceEngine::from_model_with_tier` → `engine.generate`) is driven once
//! on `KernelTier::Reference` (scalar CPU forward) and once on `KernelTier::Gpu`
//! (the fused CUDA ternary/Q1 forward), and the generated greedy token-id
//! vectors must match exactly.
//!
//! This is deliberately **engine-level** (goes through the sampler /
//! `generate()` decode loop exactly as a real server request would), which
//! complements `cuda_synthetic_prefill_parity.rs` (added alongside the CUDA
//! prefill→decode KV-handoff fix): that file drives `BonsaiModel::forward` /
//! `forward_prefill` directly and asserts logit-level closeness plus a
//! from-prefill greedy walk for a single >16-token prompt. This file instead
//! asserts *exact* token-sequence equality via the full engine/sampler path,
//! and exercises **both** sides of the batch-prefill boundary (prompts
//! `<= 16` tokens, which take the sequential per-token CUDA fast path, and
//! prompts `>= 17` tokens, which route through the fused CUDA batch-prefill
//! path whose CPU‑vs‑CUDA KV-cache handoff was the historical bug) in the same
//! suite, on two independently-shaped fully-quantised fixtures (ternary
//! `TQ2_0_g128` and binary `Q1_0G128`).
//!
//! Gracefully skips (passes) when no CUDA device is present, so it is safe to
//! run unattended on GPU-less CI while giving byte-identity coverage on any
//! `--features native-cuda` box that does have a GPU (this A4000 included).

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;

// ── Synthetic model dimensions (CUDA-friendly: hidden a multiple of 128,
//    head_dim=64, all quantised tensors a multiple of 128 weights). ──────────
const MAX_SEQ: usize = 256;
const H: usize = 128; // hidden_size (= nq * head_dim, multiple of 128 for g128)
const INTER: usize = 256; // intermediate_size (multiple of 128)
const NUM_LAYERS: usize = 2;
const NQ: usize = 2;
const NKV: usize = 1;
const HD: usize = 64; // head_dim
const VOCAB: usize = 128;

/// Returns `true` when a CUDA device is accessible on this machine.
fn cuda_available() -> bool {
    oxibonsai_kernels::CudaGraph::global().is_ok()
}

/// Deterministic FP32 tensor whose values vary with the index.
fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let phase = (i as f32) * 0.013_f32;
        let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// Build a `TQ2_0_g128` weight blob (34 bytes/block: 32B of 2-bit codes + FP16 scale).
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
        for _ in 0..32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale_f32).to_le_bytes());
    }
    data
}

/// Build a `Q1_0G128` weight blob (18 bytes/block: FP16 scale + 16B of 128 sign bits).
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
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale_f32).to_le_bytes());
        for _ in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
    }
    data
}

/// Emit one quantised projection tensor in the requested format.
fn quant_tensor(
    name: String,
    shape: Vec<u64>,
    num_weights: usize,
    seed: u64,
    ternary: bool,
) -> TensorEntry {
    if ternary {
        TensorEntry {
            name,
            shape,
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(num_weights, seed),
        }
    } else {
        TensorEntry {
            name,
            shape,
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(num_weights, seed),
        }
    }
}

/// Build a fully-quantised synthetic GGUF (`ternary=true` → TQ2, else Q1).
fn build_synthetic_gguf(ternary: bool) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("CudaCrossBackendDeterminismTest".to_string()),
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
    writer.add_metadata(
        "qwen3.context_length",
        MetadataWriteValue::U32(MAX_SEQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![H as u64, VOCAB as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(VOCAB * H, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![H as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(H, 1.0),
    });
    writer.add_tensor(quant_tensor(
        "output.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        VOCAB * H,
        0xCAFE_BABE,
        ternary,
    ));

    for layer in 0..NUM_LAYERS {
        let pfx = format!("blk.{layer}");
        for (suffix, dim) in [
            ("attn_norm", H),
            ("ffn_norm", H),
            ("attn_q_norm", HD),
            ("attn_k_norm", HD),
        ] {
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}.weight"),
                shape: vec![dim as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(dim, 1.0),
            });
        }
        let s = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_q.weight"),
            vec![H as u64, (NQ * HD) as u64],
            NQ * HD * H,
            s,
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_k.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(1),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_v.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(2),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_output.weight"),
            vec![(NQ * HD) as u64, H as u64],
            H * NQ * HD,
            s.wrapping_add(3),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_gate.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(4),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_up.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(5),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_down.weight"),
            vec![INTER as u64, H as u64],
            H * INTER,
            s.wrapping_add(6),
            ternary,
        ));
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine driver
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy sampling params: temperature 0 routes `Sampler::sample` to argmax, so
/// the seed feeds an unused RNG and the output is a deterministic argmax chain.
fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

/// Parse `gguf_bytes`, build a [`BonsaiModel`], pin an engine to `tier`, and
/// greedily generate `n` tokens from `prompt` at seed 42.
fn run(gguf_bytes: &[u8], tier: KernelTier, prompt: &[u32], n: usize) -> Vec<u32> {
    let gguf = GgufFile::parse(gguf_bytes).expect("GgufFile::parse synthetic");
    let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    let mut engine = InferenceEngine::from_model_with_tier(model, tier, greedy_params(), 42);
    engine.generate(prompt, n).expect("engine.generate")
}

/// Drive one CPU-vs-CUDA determinism check on `gguf_bytes` with `prompt`,
/// asserting the *entire* greedy token sequence matches exactly.
fn assert_cpu_cuda_agree(gguf_bytes: &[u8], prompt: &[u32], n: usize, label: &str) {
    let cpu = run(gguf_bytes, KernelTier::Reference, prompt, n);
    let cuda = run(gguf_bytes, KernelTier::Gpu, prompt, n);

    assert!(!cpu.is_empty(), "{label}: CPU reference produced no tokens");
    assert!(!cuda.is_empty(), "{label}: CUDA produced no tokens");
    assert_eq!(
        cpu,
        cuda,
        "{label}: CPU(Reference) and CUDA(Gpu) greedy output diverged at \
         temperature 0 / seed 42 (prompt_len={}).\n  cpu  = {cpu:?}\n  cuda = {cuda:?}",
        prompt.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// THE GUARD (always-on given a CUDA device): short prompt (<= 16 tokens)
/// takes the sequential per-token CUDA decode fast path end to end (no batch
/// prefill involved even for the prompt itself, since `forward_prefill`
/// dispatches single-token-at-a-time below the batch threshold). CPU and CUDA
/// greedy sequences must match exactly.
#[test]
fn cuda_ternary_short_prompt_matches_cpu_greedy_seed42() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device available");
        return;
    }
    let gguf = build_synthetic_gguf(true);
    // 6-token prompt: well under the 16-token batch-prefill boundary.
    let prompt: Vec<u32> = vec![1, 4, 7, 10, 13, 16];
    assert_cpu_cuda_agree(&gguf, &prompt, 24, "ternary short-prompt (<=16)");
}

/// THE GUARD (always-on given a CUDA device): prompt >= 17 tokens crosses the
/// batch-prefill boundary, routing through the fused CUDA batch-prefill path
/// whose KV-cache handoff to per-token decode was the historical bug (decode
/// would attend over stale/zero KV for any prompt longer than 16 tokens).
#[test]
fn cuda_ternary_long_prompt_matches_cpu_greedy_seed42() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device available");
        return;
    }
    let gguf = build_synthetic_gguf(true);
    // 20-token prompt: past the 16-token batch-prefill boundary.
    let prompt: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    assert_cpu_cuda_agree(&gguf, &prompt, 24, "ternary long-prompt (>=17)");
}

/// Same short-prompt guard on the binary `Q1_0G128` fixture — the two
/// quantised formats take different CUDA kernel paths (see
/// `oxibonsai-model/src/model/types/forward_cuda/{ternary,q1}.rs`), so both
/// need independent coverage.
#[test]
fn cuda_q1_short_prompt_matches_cpu_greedy_seed42() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device available");
        return;
    }
    let gguf = build_synthetic_gguf(false);
    let prompt: Vec<u32> = vec![2, 5, 9, 12, 15, 18];
    assert_cpu_cuda_agree(&gguf, &prompt, 24, "Q1 short-prompt (<=16)");
}

/// Same long-prompt (batch-prefill boundary) guard on the binary `Q1_0G128`
/// fixture.
#[test]
fn cuda_q1_long_prompt_matches_cpu_greedy_seed42() {
    if !cuda_available() {
        eprintln!("skip: no CUDA device available");
        return;
    }
    let gguf = build_synthetic_gguf(false);
    let prompt: Vec<u32> = (0..20u32).map(|i| (i * 11 + 5) % VOCAB as u32).collect();
    assert_cpu_cuda_agree(&gguf, &prompt, 24, "Q1 long-prompt (>=17)");
}
