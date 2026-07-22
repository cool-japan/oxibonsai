//! Regression guard: `generate_greedy_gpu`'s Metal→CPU mid-stream fallback must
//! be **coherent**, not corrupting.
//!
//! The Metal greedy decode path (`BonsaiModel::forward_greedy_gpu`) maintains
//! only the GPU-resident KV cache and never writes the CPU-side `kv_cache`. If a
//! GPU dispatch fails mid-generation and the engine naively re-ran
//! `BonsaiModel::forward` on the CPU, that forward would attend over an all-zero
//! CPU cache and silently emit a corrupted continuation (the split-KV-cache bug
//! class). The fix rebuilds the CPU KV cache from the committed sequence once and
//! then continues on the CPU (`InferenceEngine::greedy_decode_token_with_fallback`).
//!
//! This test drives the fallback deterministically without a real GPU fault via
//! the `OXIBONSAI_FORCE_CPU_DECODE_AFTER` debug seam: after `k` committed tokens
//! the greedy loop is forced onto the CPU path (rebuild + continue). The forced
//! run must produce the **same** token-id sequence as the all-GPU run — if the
//! CPU rebuild were skipped/stale the continuation would diverge into garbage.
//!
//! The fixture is the same synthetic 2-layer fully-ternary GGUF used by
//! `oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs` and
//! `cross_backend_determinism_tests.rs` (h=128, inter=256, 2 layers, vocab=32,
//! all projections + LM head `TQ2_0_g128`), reproduced verbatim so it stays
//! known-good. vocab=32 keeps every argmax token in `[0,32)`, so the (Qwen3)
//! EOS id never fires and generation runs the full token budget.

#![cfg(all(feature = "metal", target_os = "macos"))]

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;

/// KV-cache / context budget for the synthetic model.
const MAX_SEQ: usize = 512;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic ternary fixture (verbatim copy of the shared known-good fixture).
// ─────────────────────────────────────────────────────────────────────────────

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
        let scale_bytes = f16::from_f32(scale_f32).to_le_bytes();
        data.extend_from_slice(&scale_bytes);
    }
    data
}

fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let phase = (i as f32) * 0.013_f32;
        let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

fn build_synthetic_ternary_gguf() -> Vec<u8> {
    let h: usize = 128;
    let inter: usize = 256;
    let num_layers: usize = 2;
    let nq: usize = 4;
    let nkv: usize = 2;
    let hd: usize = 32;
    let vocab: usize = 32;

    let mut writer = GgufWriter::new();

    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("MetalGreedyCpuFallbackTest".to_string()),
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

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(vocab * h, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![h as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(h, 1.0),
    });
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::TQ2_0_g128,
        data: tq2_0_g128_pattern(vocab * h, 0xCAFE_BABE),
    });

    for layer in 0..num_layers {
        let pfx = format!("blk.{layer}");
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

// ─────────────────────────────────────────────────────────────────────────────
// Drivers
// ─────────────────────────────────────────────────────────────────────────────

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

/// Build a `KernelTier::Gpu` engine and greedily generate `n` tokens via the
/// GPU greedy path. `force_cpu_after` (when `Some`) sets the debug seam so the
/// decode is forced onto the coherent CPU fallback after that many tokens.
fn run_greedy_gpu(
    gguf_bytes: &[u8],
    prompt: &[u32],
    n: usize,
    force_cpu_after: Option<usize>,
) -> Vec<u32> {
    match force_cpu_after {
        Some(k) => std::env::set_var("OXIBONSAI_FORCE_CPU_DECODE_AFTER", k.to_string()),
        None => std::env::remove_var("OXIBONSAI_FORCE_CPU_DECODE_AFTER"),
    }
    let gguf = GgufFile::parse(gguf_bytes).expect("GgufFile::parse synthetic");
    let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    let mut engine =
        InferenceEngine::from_model_with_tier(model, KernelTier::Gpu, greedy_params(), 42);
    let out = engine
        .generate_greedy_gpu(prompt, n)
        .expect("generate_greedy_gpu");
    std::env::remove_var("OXIBONSAI_FORCE_CPU_DECODE_AFTER");
    out
}

/// Pure-CPU greedy reference via the scalar `KernelTier::Reference` forward.
fn run_cpu_reference(gguf_bytes: &[u8], prompt: &[u32], n: usize) -> Vec<u32> {
    let gguf = GgufFile::parse(gguf_bytes).expect("GgufFile::parse synthetic");
    let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    let mut engine =
        InferenceEngine::from_model_with_tier(model, KernelTier::Reference, greedy_params(), 42);
    engine.generate(prompt, n).expect("engine.generate")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// THE GUARD: forcing the greedy loop onto the CPU fallback mid-stream (after
/// `k=3` GPU-decoded tokens) must yield the **same** token sequence as the pure
/// all-GPU run. A stale/zero CPU cache would corrupt tokens `3..` and diverge.
#[test]
fn metal_greedy_cpu_fallback_matches_all_gpu() {
    let gguf = build_synthetic_ternary_gguf();
    let prompt: Vec<u32> = vec![1, 4, 7, 10, 13, 16];
    let n = 20;

    let all_gpu = run_greedy_gpu(&gguf, &prompt, n, None);
    let mid_stream_fallback = run_greedy_gpu(&gguf, &prompt, n, Some(3));

    assert!(!all_gpu.is_empty(), "all-GPU greedy produced no tokens");
    assert_eq!(
        all_gpu.len(),
        n,
        "synthetic model has vocab=32 (EOS never fires); expected the full token budget"
    );
    assert_eq!(
        all_gpu, mid_stream_fallback,
        "Metal→CPU mid-stream fallback CORRUPTED the continuation.\n  all_gpu  = {all_gpu:?}\n  fallback = {mid_stream_fallback:?}"
    );
}

/// Forcing the CPU fallback from the very first decode token (`k=1`, so the CPU
/// KV cache is rebuilt from just prompt+first_token) must also match all-GPU —
/// exercising the rebuild's shortest committed-sequence case.
#[test]
fn metal_greedy_cpu_fallback_from_first_token_matches_all_gpu() {
    let gguf = build_synthetic_ternary_gguf();
    let prompt: Vec<u32> = vec![2, 5, 9, 14];
    let n = 16;

    let all_gpu = run_greedy_gpu(&gguf, &prompt, n, None);
    let immediate_fallback = run_greedy_gpu(&gguf, &prompt, n, Some(1));

    assert_eq!(
        all_gpu, immediate_fallback,
        "Metal→CPU fallback from the first decode token diverged from all-GPU.\n  all_gpu  = {all_gpu:?}\n  fallback = {immediate_fallback:?}"
    );
}

/// The coherent CPU fallback also matches the canonical pure-CPU
/// (`KernelTier::Reference`) greedy reference — anchoring that the rebuilt cache
/// reproduces exactly what a from-scratch CPU generation would compute.
#[test]
fn metal_greedy_cpu_fallback_matches_cpu_reference() {
    let gguf = build_synthetic_ternary_gguf();
    let prompt: Vec<u32> = vec![1, 4, 7, 10, 13, 16];
    let n = 20;

    let cpu_reference = run_cpu_reference(&gguf, &prompt, n);
    let mid_stream_fallback = run_greedy_gpu(&gguf, &prompt, n, Some(3));

    assert!(
        !cpu_reference.is_empty(),
        "CPU reference produced no tokens"
    );
    assert_eq!(
        cpu_reference, mid_stream_fallback,
        "forced-CPU-fallback greedy diverged from the pure-CPU reference.\n  cpu      = {cpu_reference:?}\n  fallback = {mid_stream_fallback:?}"
    );
}

/// FAITHFUL real-model guard: on the shipped 1.7B ternary GGUF, the Metal greedy
/// GPU path (`generate_greedy_gpu`, the CLI's temperature-0 path) must produce a
/// byte-identical ≥64-token sequence to the pure-CPU reference, AND forcing the
/// mid-stream CPU fallback must not change a single token. Ignored by default
/// (needs a multi-hundred-MB model file + a dev Mac with Metal).
///
/// Run with:
/// ```text
/// OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf \
///   cargo test -p oxibonsai-runtime --features metal \
///   --test metal_greedy_cpu_fallback_tests \
///   real_model_greedy_gpu_fallback_byte_identical -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires OXI_MODEL real ternary GGUF; run on dev Mac"]
fn real_model_greedy_gpu_fallback_byte_identical() {
    let Some(path) = std::env::var_os("OXI_MODEL") else {
        eprintln!(
            "real_model_greedy_gpu_fallback_byte_identical: OXI_MODEL not set — skipping. \
             Set OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf to run."
        );
        return;
    };
    let gguf = std::fs::read(&path).expect("read OXI_MODEL gguf");

    // Realistic Qwen3 chat-template prefix.
    let prompt: Vec<u32> = vec![151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198];
    let n = 64;

    let all_gpu = run_greedy_gpu(&gguf, &prompt, n, None);
    let fallback = run_greedy_gpu(&gguf, &prompt, n, Some(8));
    let cpu_reference = run_cpu_reference(&gguf, &prompt, n);

    assert!(
        all_gpu.len() >= 32,
        "expected a substantial GPU greedy output"
    );
    assert_eq!(
        all_gpu, fallback,
        "real-model Metal→CPU mid-stream fallback CORRUPTED the continuation.\n  all_gpu  = {all_gpu:?}\n  fallback = {fallback:?}"
    );
    assert_eq!(
        all_gpu, cpu_reference,
        "real-model CPU vs Metal greedy byte-parity VIOLATED (generate_greedy_gpu).\n  cpu   = {cpu_reference:?}\n  metal = {all_gpu:?}"
    );
    eprintln!(
        "real_model_greedy_gpu_fallback_byte_identical: {} tokens byte-identical across \
         all-GPU / forced-CPU-fallback / pure-CPU",
        all_gpu.len()
    );
}
