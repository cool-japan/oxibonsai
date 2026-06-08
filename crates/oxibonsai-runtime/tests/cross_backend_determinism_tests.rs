//! Cross-backend (CPU vs Metal) determinism guard for the README promise.
//!
//! README.md states: *"at `--temperature 0 --seed 42`, CPU and Metal produce
//! byte-identical output."* This test pins that contract: the same model bytes
//! are driven once on [`KernelTier::Reference`] (scalar CPU forward) and once on
//! [`KernelTier::Gpu`] (the fused Metal ternary forward), at temperature 0 /
//! seed 42, and the generated token-id vectors must be identical.
//!
//! Why this exercises Metal end-to-end with no live GPU backend: the engine's
//! backend is chosen purely by the [`KernelTier`] passed to it
//! ([`InferenceEngine::from_model_with_tier`] →
//! [`oxibonsai_kernels::KernelDispatcher::with_tier`]). `KernelTier::Gpu` gates
//! the fused ternary forward in `BonsaiModel::forward`/`forward_prefill`, and
//! that forward **self-uploads** the CPU-side ternary weight blocks (it owns its
//! own Metal device/cache), so a `KernelTier::Gpu` engine runs Metal without any
//! `upload_weights_to_gpu` call or live `gpu_backend`. `auto_detect()` on this
//! Mac picks NEON, never Gpu — Metal is only reachable by explicitly naming the
//! Gpu tier, which is exactly what this guard does.
//!
//! Greedy semantics: `Sampler::sample` routes `temperature < 1e-6` to argmax, so
//! the seed feeds an unused RNG and the output is a deterministic argmax chain —
//! hence we assert *identical* token-id vectors, not approximate logits.
//!
//! The fixture is the same synthetic 2-layer fully-ternary GGUF used by
//! `oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs` (h=128,
//! inter=256, 2 layers, vocab=32, all projections + LM head stored as
//! `TQ2_0_g128`), reproduced here verbatim so the fixture stays known-good.

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
// Synthetic ternary fixture
//
// Copied verbatim from
// `oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs` so the fixture
// is identical to the one the Metal prefill parity suite already validates.
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
        MetadataWriteValue::Str("CrossBackendDeterminismTest".to_string()),
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

// ─────────────────────────────────────────────────────────────────────────────
// Engine driver
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy sampling params: temperature 0 routes `Sampler::sample` to argmax.
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// THE GUARD (always-on): on the synthetic fully-ternary fixture, the CPU
/// scalar reference forward and the fused Metal ternary forward must emit the
/// **same** greedy token-id vector at temperature 0 / seed 42. This is the
/// README's "CPU and Metal produce byte-identical output" contract, exercised
/// on every `--features metal` test run.
#[test]
fn cpu_reference_and_metal_agree_greedy_temp0_seed42() {
    let gguf = build_synthetic_ternary_gguf();
    // Fixed 6-token prompt, all in [0, vocab=32).
    let prompt: Vec<u32> = vec![1, 4, 7, 10, 13, 16];
    let n = 24;

    let cpu = run(&gguf, KernelTier::Reference, &prompt, n);
    let metal = run(&gguf, KernelTier::Gpu, &prompt, n);

    assert!(!cpu.is_empty(), "CPU reference produced no tokens");
    assert!(!metal.is_empty(), "Metal produced no tokens");
    // NOTE: full-seq vs first-token tradeoff. We assert the *entire* greedy
    // sequence (the strict README contract). If a future fixture change makes
    // this flaky from a benign near-tied argmax that rounds differently between
    // CPU-scalar and Metal-FP32 and then cascades, downgrade to comparing only
    // `cpu[0] == metal[0]` (pure prefill argmax, no cascade). It was NOT flaky
    // when this guard was written — the full sequence matched exactly.
    assert_eq!(
        cpu, metal,
        "README determinism gate VIOLATED: CPU(Reference) and Metal(Gpu) greedy \
         output diverged at temperature 0 / seed 42.\n  cpu   = {cpu:?}\n  metal = {metal:?}"
    );
}

/// FAITHFUL guard against a real staged ternary GGUF. Validates the README
/// promise literally on the shipped 1.7B ternary model rather than a synthetic
/// fixture. Ignored by default (needs a multi-hundred-MB model file + a dev Mac
/// with Metal).
///
/// Run with:
/// ```text
/// OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf \
///   cargo test -p oxibonsai-runtime --features metal \
///   --test cross_backend_determinism_tests \
///   real_model_cpu_metal_byte_identical -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires OXI_MODEL real ternary GGUF; run on dev Mac"]
fn real_model_cpu_metal_byte_identical() {
    let Some(path) = std::env::var_os("OXI_MODEL") else {
        eprintln!(
            "real_model_cpu_metal_byte_identical: OXI_MODEL not set — skipping. \
             Set OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf to run."
        );
        return;
    };

    let gguf = std::fs::read(&path).expect("read OXI_MODEL gguf");

    // Realistic Qwen3 chat-template prefix:
    //   <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
    let prompt: Vec<u32> = vec![151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198];
    let n = 32;

    let cpu = run(&gguf, KernelTier::Reference, &prompt, n);
    let metal = run(&gguf, KernelTier::Gpu, &prompt, n);

    assert!(!cpu.is_empty(), "CPU reference produced no tokens");
    assert!(!metal.is_empty(), "Metal produced no tokens");

    if cpu != metal {
        // Surface the first divergence for diagnosis — a real mismatch here is
        // an important finding against the README promise.
        let first = cpu
            .iter()
            .zip(metal.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| cpu.len().min(metal.len()));
        panic!(
            "README determinism gate VIOLATED on real model: CPU(Reference) vs \
             Metal(Gpu) greedy output diverged at temperature 0 / seed 42 \
             (first divergence at index {first}).\n  cpu   = {cpu:?}\n  metal = {metal:?}"
        );
    }
}

// CUDA variant: same shape, gate on native-cuda + linux/windows, compares
// Reference vs Gpu(CUDA). Deferred (cap-of-8 bug, needs hw).
