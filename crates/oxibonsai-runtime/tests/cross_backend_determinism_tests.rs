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
//! `upload_weights_to_gpu` call or live `gpu_backend`. The guard names both
//! tiers explicitly rather than relying on `auto_detect()`, so the CPU side is
//! always the scalar reference and the Metal side runs whether or not
//! auto-detection would have picked it on this host.
//!
//! Greedy semantics: `Sampler::sample` routes `temperature < 1e-6` to argmax, so
//! the seed feeds an unused RNG and the output is a deterministic argmax chain —
//! hence we assert *identical* token-id vectors, not approximate logits.
//!
//! The fixture is the same synthetic 2-layer fully-ternary GGUF used by
//! `oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs` (h=128,
//! inter=256, 2 layers, vocab=32, all projections + LM head stored as
//! `TQ2_0_g128`), reproduced here verbatim so the fixture stays known-good.
//!
//! ## `OXIBONSAI_KERNEL_TIER`
//!
//! The opt-in INT8 tier (K-14) changes the CPU tiers' native-format GEMV/GEMM
//! results bit for bit, so the byte-identity contract above is a contract
//! about the **default** configuration. Every test here therefore takes a
//! [`TierEnvGuard`], which clears the variable for the test's lifetime (an
//! ambient export, or a gate that sets it for this binary's throughput leg,
//! cannot leak into the CPU-vs-Metal comparison) and restores it afterwards.
//! `metal_greedy_output_ignores_the_int8_tier_selector` pins the other half:
//! a `KernelTier::Gpu` engine is never diverted, whatever the variable says.

#![cfg(all(feature = "metal", target_os = "macos"))]

use half::f16;
use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_kernels::dispatch_int8::{Int8Tier, KERNEL_TIER_ENV};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_seam::Backend;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::gguf_fixture::Lcg;
use oxibonsai_testkit::workspace::find_model;

/// Serializes every test in this binary: each one owns
/// `OXIBONSAI_KERNEL_TIER` for its whole run through a [`TierEnvGuard`]
/// (`std::env::set_var` is `unsafe` because a concurrent read on any key can
/// observe a torn `environ`), and the real-model tests must not overlap
/// either — two multi-hundred-MB models resident at once, or a throughput
/// measurement sharing the cores with a Metal run, would be wrong for other
/// reasons.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII owner of `OXIBONSAI_KERNEL_TIER` for one test: takes [`ENV_LOCK`],
/// snapshots and **clears** the variable, and restores the snapshot on drop
/// (also while unwinding from a failed assertion).
struct TierEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

impl TierEnvGuard {
    fn cleared() -> Self {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` is held for the lifetime of the returned guard and
        // serializes every reader and writer of the variable in this binary.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        Self { _lock: lock, prior }
    }

    /// The value the variable held before this guard cleared it.
    fn ambient(&self) -> Option<&str> {
        self.prior.as_deref()
    }

    /// Set (`Some`) or clear (`None`) the tier selector.
    fn select(&self, name: Option<&str>) {
        // SAFETY: `self._lock` is held (see `cleared`).
        unsafe {
            match name {
                Some(n) => std::env::set_var(KERNEL_TIER_ENV, n),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

impl Drop for TierEnvGuard {
    fn drop(&mut self) {
        // SAFETY: `self._lock` is held for the entire body of `drop`.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(KERNEL_TIER_ENV, v),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

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
/// `00→-1, 01→0, 10→+1`, matching the existing GEMV reference kernel.
///
/// To get a reasonably "interesting" weight matrix we vary both the qs pattern
/// and the scale across blocks based on a 64-bit linear-congruential PRNG seed.
///
/// CQ-14: `11` (`0b11`) is a *reserved* code, not a fourth value —
/// `screen_ternary_codes` in `oxibonsai-model/src/weight_loaders.rs` rejects
/// it outright, so this fixture must never emit it; each lane is folded into
/// `{0, 1, 2}` before packing, exactly as
/// `crates/oxibonsai-model/src/model/types/gpu_cache.rs::tq2_pattern` does.
///
/// T-07: the bytes come from
/// `oxibonsai_testkit::gguf_fixture::Lcg::next_valid_tq2_byte`; `Lcg::new(s)`
/// stores `s` as its state directly, and the same golden-ratio constant is
/// pre-added before the first `next_u64()`, so the byte sequence is the one
/// this fixture has always produced.
fn tq2_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 34);
    let mut lcg = Lcg::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
    for _ in 0..num_blocks {
        // 32 bytes of qs (128 weights × 2 bits, 4 lanes/byte).
        for _ in 0..32 {
            data.push(lcg.next_valid_tq2_byte());
        }
        // FP16 scale in (0.25, 0.75] so RMSNorm output stays in a sane range.
        let scale_f32 =
            0.25_f32 + ((lcg.next_u64() >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
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
    let _env = TierEnvGuard::cleared();
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
    println!(
        "synthetic fixture: CPU(Reference) and Metal(Gpu) greedy outputs byte-identical \
         ({} tokens)",
        cpu.len()
    );
}

/// A `KernelTier::Gpu` engine is never diverted onto the opt-in INT8 tier:
/// with `OXIBONSAI_KERNEL_TIER` naming one, the Metal greedy chain is
/// byte-identical to the one it produces with the variable unset.
#[test]
fn metal_greedy_output_ignores_the_int8_tier_selector() {
    let env = TierEnvGuard::cleared();
    let gguf = build_synthetic_ternary_gguf();
    let prompt: Vec<u32> = vec![2, 5, 8, 11, 14, 17, 20];
    let n = 16;

    let unset = run(&gguf, KernelTier::Gpu, &prompt, n);
    let tier = Int8Tier::best_available();
    env.select(Some(tier.name()));
    let selected = run(&gguf, KernelTier::Gpu, &prompt, n);
    env.select(None);

    assert!(!unset.is_empty(), "Metal produced no tokens");
    assert_eq!(
        unset, selected,
        "the Metal engine's greedy output changed when {KERNEL_TIER_ENV}={tier} was set — \
         a Gpu-tier dispatcher must never be diverted onto the INT8 tier"
    );
    println!(
        "Metal(Gpu) greedy output byte-identical with {KERNEL_TIER_ENV} unset and ={tier} \
         ({} tokens)",
        unset.len()
    );
}

/// FAITHFUL guard against a real staged ternary GGUF. Validates the README
/// promise literally on the shipped 1.7B ternary model rather than a synthetic
/// fixture. Needs a dev Mac with Metal and resolves the model from `OXI_MODEL`
/// when set, else the testkit `models/`/`$OXIBONSAI_MODELS_DIR` fallback;
/// self-skips with a `Capability::LegacyModels` record when neither locates
/// one.
///
/// Run with:
/// ```text
/// OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf \
///   cargo test -p oxibonsai-runtime --features metal \
///   --test cross_backend_determinism_tests \
///   real_model_cpu_metal_byte_identical -- --nocapture
/// ```
#[test]
fn real_model_cpu_metal_byte_identical() {
    let _env = TierEnvGuard::cleared();
    let test_name =
        "oxibonsai-runtime::cross_backend_determinism_tests::real_model_cpu_metal_byte_identical";
    let Some(path) = std::env::var_os("OXI_MODEL")
        .map(std::path::PathBuf::from)
        .or_else(|| find_model("Ternary-Bonsai-1.7B.gguf"))
    else {
        eprintln!(
            "real_model_cpu_metal_byte_identical: OXI_MODEL not set and \
             Ternary-Bonsai-1.7B.gguf not found under {:?} — skipping. Set OXI_MODEL or \
             OXIBONSAI_MODELS_DIR to run.",
            oxibonsai_testkit::workspace::models_dir()
        );
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };
    let start = std::time::Instant::now();

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
    println!(
        "real model: CPU(Reference) and Metal(Gpu) greedy outputs byte-identical ({} tokens)",
        cpu.len()
    );
    record_executed_timed(Capability::LegacyModels, test_name, start.elapsed());
}

/// The 1/5/15-minute load average, for the throughput line.
fn load_average() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            s.trim()
                .trim_matches(|c| c == '{' || c == '}')
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// First-index argmax (the greedy sampler's tie-break).
fn argmax_first(values: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Decode steps the throughput measurement times.
const DECODE_STEPS: usize = 32;

/// Greedy decode of [`DECODE_STEPS`] tokens through `engine`'s own
/// prefill/decode entry points; returns the tokens and the decode time.
fn timed_greedy_decode(
    engine: &mut InferenceEngine<'_>,
    prompt: &[u32],
) -> (Vec<u32>, std::time::Duration) {
    engine.reset();
    let logits = engine.prefill_from_pos(prompt, 0).expect("prefill");
    let mut tokens = vec![argmax_first(&logits)];
    let start = std::time::Instant::now();
    for step in 0..DECODE_STEPS {
        let next = engine
            .decode_step(tokens[step], prompt.len() + step)
            .expect("decode step");
        tokens.push(argmax_first(&next));
    }
    (tokens, start.elapsed())
}

/// CPU decode throughput of [`InferenceEngine`] on the real ternary model in
/// `OXI_MODEL`, with the opt-in INT8 tier (K-14) off and on.
///
/// The engine is built on [`Backend::Cpu`], so the model's own per-layer
/// dispatchers are CPU-tier and the tier genuinely applies (a `Gpu`-tier
/// engine is never diverted). The tier used is the ambient
/// `OXIBONSAI_KERNEL_TIER` when one is set — a gate can pick it — else the
/// best this CPU supports. Reports decode tok/s both ways, the machine's
/// load, and how far the two greedy chains agree (int8 activation
/// quantization may legitimately flip a near-tie, so agreement is evidence,
/// not a gate). Resolves the model from `OXI_MODEL` when set, else the
/// testkit `models/`/`$OXIBONSAI_MODELS_DIR` fallback; self-skips with a
/// capability record when neither locates one.
#[test]
fn real_model_cpu_decode_tok_s_with_and_without_the_int8_tier() {
    let env = TierEnvGuard::cleared();
    let test_name = "oxibonsai-runtime::cross_backend_determinism_tests::\
                     real_model_cpu_decode_tok_s_with_and_without_the_int8_tier";
    let Some(path) = std::env::var_os("OXI_MODEL")
        .map(std::path::PathBuf::from)
        .or_else(|| find_model("Ternary-Bonsai-1.7B.gguf"))
    else {
        eprintln!(
            "skip: OXI_MODEL not set and Ternary-Bonsai-1.7B.gguf not found under {:?} (set \
             OXI_MODEL or OXIBONSAI_MODELS_DIR)",
            oxibonsai_testkit::workspace::models_dir()
        );
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };
    let tier = env
        .ambient()
        .and_then(Int8Tier::from_name)
        .map(Int8Tier::clamp_to_cpu)
        .unwrap_or_else(Int8Tier::best_available);

    let mmap = mmap_gguf_file(std::path::Path::new(&path)).expect("mmap OXI_MODEL");
    let gguf = GgufFile::parse(&mmap).expect("parse OXI_MODEL");
    let mut engine =
        InferenceEngine::from_gguf_with_backend(&gguf, greedy_params(), 42, MAX_SEQ, Backend::Cpu)
            .expect("CPU-backend engine");
    // <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
    let prompt: Vec<u32> = vec![151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198];

    let start = std::time::Instant::now();
    // Warm both configurations once (page-in, Rayon pool), then measure.
    let _ = timed_greedy_decode(&mut engine, &prompt);
    let (f32_tokens, f32_time) = timed_greedy_decode(&mut engine, &prompt);
    env.select(Some(tier.name()));
    let _ = timed_greedy_decode(&mut engine, &prompt);
    let (int8_tokens, int8_time) = timed_greedy_decode(&mut engine, &prompt);
    env.select(None);

    let f32_tok_s = DECODE_STEPS as f64 / f32_time.as_secs_f64().max(1e-12);
    let int8_tok_s = DECODE_STEPS as f64 / int8_time.as_secs_f64().max(1e-12);
    let agree = f32_tokens
        .iter()
        .zip(int8_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "InferenceEngine CPU decode on {}: {f32_tok_s:.2} tok/s f32 default -> {int8_tok_s:.2} \
         tok/s with {KERNEL_TIER_ENV}={tier} ({:.2}x, {DECODE_STEPS} greedy steps, release={}); \
         greedy chains agree on the first {agree}/{} tokens; load average {}",
        std::path::Path::new(&path)
            .file_name()
            .map_or_else(|| "OXI_MODEL".into(), |n| n.to_string_lossy()),
        int8_tok_s / f32_tok_s.max(1e-12),
        !cfg!(debug_assertions),
        f32_tokens.len(),
        load_average()
    );
    assert_eq!(f32_tokens.len(), DECODE_STEPS + 1);
    assert_eq!(int8_tokens.len(), DECODE_STEPS + 1);
    record_executed_timed(Capability::LegacyModels, test_name, start.elapsed());
}

// CUDA variant: same shape, gate on native-cuda + linux/windows, compares
// Reference vs Gpu(CUDA). Deferred (cap-of-8 bug, needs hw).
