//! Legacy 1.7B/8B greedy CPU<->Metal/NEON parity gate (design §7.4) — the
//! real-model-weights half this crate can actually reach.
//!
//! ## Background: the divergence this test is the permanent regression guard for
//!
//! A pre-fix regression found that `oxibonsai run --temperature 0` was
//! **not greedy on CPU**: the CLI
//! hardcoded `repetition_penalty: 1.1` regardless of `--temperature 0`,
//! while the Metal path (`generate_greedy_gpu`) never applied any penalty at
//! all. CPU and Metal are numerically equivalent at true greedy decode on
//! every real model this project ships (per-step logits agree to ~2e-5
//! relative) — the bug was a CLI/engine *configuration* asymmetry, not a
//! kernel defect, and the existing CPU-vs-Metal parity tests
//! (`cross_backend_determinism_tests.rs`,
//! `metal_greedy_cpu_fallback_tests.rs`) never drove the real legacy GGUFs.
//!
//! This file drives the real, shipped legacy models (`Ternary-Bonsai-1.7B`,
//! `Ternary-Bonsai-8B`, `Bonsai-8B`) end to end at true greedy settings
//! (`temperature=0.0`, `repetition_penalty=1.0` — no penalty, matching what
//! `--temperature 0` must mean on every backend) across every kernel tier
//! this host can reach, and classifies every per-step logit disagreement.
//!
//! ## Scope cut: numeric prompts, not the golden text (see the deviations
//! footer for the exact missing piece and what it needs)
//!
//! The design also asks for a check against `legacy_golden.json`'s captured
//! **decoded text**, reached through "the SAME public entry points the CLI
//! uses". Both need a real tokenizer/detokenizer, and `--temperature 0`'s
//! no-penalty construction (`src/cli/util.rs::build_sampling_params`,
//! `pub(crate)` inside the `oxibonsai-cli` binary crate) lives on
//! `oxibonsai_runtime::sampling::SamplingParams`. Neither
//! `oxibonsai-tokenizer` nor `oxibonsai-runtime` is a dependency (not even a
//! dev-dependency) of `oxibonsai-model` today — from inside this crate
//! there is no way to turn real English prompt text into token ids, or
//! generated ids back into text, at all. This file therefore drives the
//! real model weights with **deterministic numeric prompt ids** (the same
//! convention `crates/oxibonsai-runtime/tests/cuda_ternary_forward_parity.rs`
//! already uses for the same reason on the CUDA side) instead of the golden
//! prompts/text, and greedy params are constructed directly on the
//! low-level dispatch path (no `SamplingParams`/`Sampler` needed for pure
//! argmax) rather than through `build_sampling_params`. This still
//! delivers the design's core, most valuable claim — CPU and every
//! GPU/SIMD tier agree, end to end, on the actual shipped model weights —
//! just not the CLI-text-output half, which
//! `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs` covers instead
//! (that crate has the tokenizer/runtime dependencies this one lacks).
//!
//! ## What this gate asserts (design decision D-6, 2026-09-22)
//!
//! 1. **The greedy TOKEN CHAIN is the hard gate**, unconditionally, for every
//!    tier pair — it used to be asserted only when a near-tie classifier
//!    counted zero argmax flips.
//! 2. **`KernelTier::Reference` vs the auto-detected CPU tier must be
//!    BIT-EXACT** on every step's logits (measured exactly 0.0 over 64
//!    self-generated steps, all three prompt slots, on
//!    `Ternary-Bonsai-1.7B.gguf`).
//! 3. Only the Reference-vs-GPU (Metal) pair takes a numeric tolerance, and
//!    it is RELATIVE: `|Δ| <= max(5e-3, 5e-3 · max(1, max|reference|))` per
//!    step. See [`per_step_bound`] for why the combinator is `max` and not
//!    `&&`, and [`REL_TOL`] for the six measured (model, slot) points it is
//!    calibrated against. The same 5e-3 as an ABSOLUTE ceiling instead of a
//!    relative one misreports a real regression at step 51 of a
//!    self-generated chain whose every KV entry came from the diverging
//!    backend — that shape, not the kernels, is what moves the reading.
//! 4. These tests are **not** `#[ignore]`d: they run whenever the models are
//!    present (`OXIBONSAI_MODELS_DIR`, else `<workspace>/models`) and
//!    otherwise self-skip into the capability manifest with
//!    `executed: false`. `executed: true` is written only after the whole
//!    tier matrix has actually run. **Invoke them with `--test-threads=1`** —
//!    see [`gpu_serial`].
//! 5. The GPU arm is the PRODUCTION Metal route, not a GPU-tier dispatcher
//!    over CPU fallbacks: it is built the way `InferenceEngine::from_gguf`
//!    builds a Metal engine (auto-detected dispatcher, weight upload where
//!    that route needs it, eager GPU cache), and every GPU-arm prefill must
//!    run on the fused Metal batch path — asserted, see
//!    [`generate_with_logits`].

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelDispatcher, KernelTier};
use oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

// ── OXIBONSAI_KERNEL_TIER scrub ─────────────────────────────────────────────
//
// The opt-in INT8 tier (K-14) diverts the CPU tiers' native-format GEMV/GEMM
// but never Metal, so an exported `OXIBONSAI_KERNEL_TIER` would make this
// gate's CPU arms compute something its Metal arm does not, and the greedy
// token chains could flip. Every test below that runs a model (or re-execs
// this binary to run one) therefore owns the variable for its whole run
// through a [`TierEnvGuard`]; re-exec'd children inherit the cleared value.

/// Serializes every [`TierEnvGuard`] in this binary: `std::env::set_var` /
/// `remove_var` are `unsafe` because a concurrent read of any key can observe
/// a torn `environ`.
static TIER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take [`TIER_ENV_LOCK`], recovering it if a failed test poisoned it.
fn lock_tier_env() -> std::sync::MutexGuard<'static, ()> {
    TIER_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// RAII owner of `OXIBONSAI_KERNEL_TIER` for one test: holds
/// [`TIER_ENV_LOCK`], clears the variable, and restores its previous value on
/// drop (also while unwinding from a failed assertion).
struct TierEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

impl TierEnvGuard {
    fn cleared() -> Self {
        Self::cleared_under(lock_tier_env())
    }

    /// [`Self::cleared`] for a caller that already holds the lock.
    fn cleared_under(lock: std::sync::MutexGuard<'static, ()>) -> Self {
        let prior = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` is held for the lifetime of the guard and serializes
        // every reader and writer of the variable in this binary.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        Self { _lock: lock, prior }
    }
}

impl Drop for TierEnvGuard {
    fn drop(&mut self) {
        // SAFETY: `self._lock` is still held (fields drop after this body).
        unsafe {
            match &self.prior {
                Some(value) => std::env::set_var(KERNEL_TIER_ENV, value),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

/// The guard clears a selector that is set, keeps every other guard out
/// while it lives, and puts the previous value (or its absence) back on drop.
#[test]
fn the_tier_env_guard_clears_the_selector_and_restores_it() {
    let ambient = {
        let lock = lock_tier_env();
        let ambient = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` is held.
        unsafe {
            std::env::set_var(KERNEL_TIER_ENV, "neon-dot");
        }
        let guard = TierEnvGuard::cleared_under(lock);
        assert!(
            std::env::var(KERNEL_TIER_ENV).is_err(),
            "the selector is cleared while the guard is alive"
        );
        assert!(
            TIER_ENV_LOCK.try_lock().is_err(),
            "the guard holds the process-wide lock"
        );
        drop(guard);
        assert_eq!(
            std::env::var(KERNEL_TIER_ENV).as_deref(),
            Ok("neon-dot"),
            "the previous value is restored on drop"
        );
        ambient
    };
    {
        let lock = lock_tier_env();
        // SAFETY: `lock` is held.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        drop(TierEnvGuard::cleared_under(lock));
        assert!(
            std::env::var(KERNEL_TIER_ENV).is_err(),
            "a selector that was unset stays unset"
        );
    }
    // Leave the process as it was found.
    let _lock = lock_tier_env();
    // SAFETY: `_lock` is held.
    unsafe {
        match ambient {
            Some(value) => std::env::set_var(KERNEL_TIER_ENV, value),
            None => std::env::remove_var(KERNEL_TIER_ENV),
        }
    }
}

/// Generation budget: long enough to reach a repetition-inducing tail on
/// every prompt below.
const MAX_TOKENS: usize = 64;

/// Context budget: generous headroom over `MAX_TOKENS` plus the longest
/// prompt below, well inside all three legacy models' trained context.
const MAX_SEQ: usize = 512;

/// One legacy model under test.
struct LegacyModel {
    /// GGUF filename under `models/`.
    file_name: &'static str,
}

const LEGACY_MODELS: [LegacyModel; 3] = [
    LegacyModel {
        file_name: "Ternary-Bonsai-1.7B.gguf",
    },
    LegacyModel {
        file_name: "Ternary-Bonsai-8B.gguf",
    },
    LegacyModel {
        file_name: "Bonsai-8B.gguf",
    },
];

/// Deterministic, varied prompt token-id sequences (see this file's header
/// doc comment for why these are numeric ids rather than the golden
/// prompts' real text) — three different lengths/patterns, mirroring
/// `cuda_ternary_forward_parity.rs`'s `(0..plen).map(|i| 1000 + i * 37)`
/// convention, chosen distinctly per slot so the three "prompts" are not
/// trivially related to each other.
fn synthetic_prompt(slot: usize) -> Vec<u32> {
    match slot {
        0 => (0..24u32).map(|i| 1000 + i * 37).collect(),
        1 => (0..18u32).map(|i| 2000 + i * 53).collect(),
        _ => (0..30u32).map(|i| 500 + i * 19).collect(),
    }
}

// ── T-05 capability-report producer ─────────────────────────────────────────
//
// T-07: `oxibonsai-model` takes `oxibonsai-testkit` as a dev-dependency
// (imported above) and calls its shared `capability` module directly,
// rather than keeping its own inline copy of the same reporting logic.

/// `models/` resolved relative to this crate's own (fixed at compile time)
/// `CARGO_MANIFEST_DIR` — every workspace member lives exactly two
/// directories below the workspace root as `crates/<name>`, so
/// `../../models` is stable regardless of which crate's test binary calls
/// in (same technique as `oxibonsai_testkit::workspace::models_dir`, which
/// this crate cannot yet depend on — see this file's header deviation).
fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OXIBONSAI_MODELS_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
}

fn find_model(file_name: &str) -> Option<PathBuf> {
    let path = models_dir().join(file_name);
    match std::fs::metadata(&path) {
        Ok(meta) if meta.is_file() && meta.len() > 0 => Some(path),
        _ => None,
    }
}

/// Whether `oxibonsai_runtime::engine_control::gguf_fused_metal_route(gguf)
/// .gpu_weight_upload_redundant()` holds — mirrored here because this crate
/// cannot depend on `oxibonsai-runtime`: the LM head and every blocked
/// `blk.*` / `token_embd` matrix are ternary, so every ternary Metal entry
/// point binds its own cached weight set and `InferenceEngine::from_gguf`
/// skips the scirs2 weight upload. The one-bit route is the opposite: its
/// fused paths need the uploaded handles.
fn gpu_weight_upload_redundant(gguf: &GgufFile<'_>) -> bool {
    let ternary_head = gguf
        .tensors
        .get("output.weight")
        .is_some_and(|head| head.tensor_type.is_ternary());
    ternary_head
        && gguf.tensors.iter().all(|(name, info)| {
            let blocked_matrix = (name.starts_with("blk.") && name.ends_with(".weight"))
                || name == "token_embd.weight";
            !blocked_matrix || info.tensor_type.block_size() <= 1 || info.tensor_type.is_ternary()
        })
}

/// Greedily decode `max_tokens` steps for `tier` at true-greedy settings (no
/// sampling, no penalty — argmax is unaffected by temperature/top-k/top-p
/// once `SamplingParams` is out of the picture entirely), returning both the
/// generated token ids and every step's full logit vector (needed for the
/// cross-tier classifier).
///
/// A CPU tier is pinned with `KernelDispatcher::with_tier`. The GPU tier is
/// built the way `InferenceEngine::from_gguf` builds a production Metal
/// engine: the auto-detected dispatcher (the only constructor wired to a live
/// GPU backend), the weight upload `from_gguf` performs — skipped exactly
/// where it skips it ([`gpu_weight_upload_redundant`]) — and the eager GPU
/// cache build. The GPU arm therefore runs the fused Metal batch prefill and
/// the fused Metal decode production runs, and this function asserts it did:
/// no `prefill_dispatch` fallback warning, and the device-KV latch
/// (`gpu_path_active`, which the CPU prefill fallback clears) set after the
/// prefill.
fn generate_with_logits(
    gguf_bytes: &[u8],
    tier: KernelTier,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    let kernel = if tier == KernelTier::Gpu {
        let kernel = KernelDispatcher::auto_detect();
        assert_eq!(
            kernel.tier(),
            KernelTier::Gpu,
            "KernelDispatcher::auto_detect() did not select the GPU tier for the GPU arm"
        );
        if !gpu_weight_upload_redundant(&gguf) {
            model.upload_weights_to_gpu(&kernel);
        }
        model
            .get_or_create_gpu_cache()
            .unwrap_or_else(|e| panic!("GPU weight cache build for the GPU arm: {e}"));
        kernel
    } else {
        KernelDispatcher::with_tier(tier)
    };

    let (prefill, fell_back, fallback_reason) =
        m18_run_detecting_warn(|| model.forward_prefill(prompt_ids, 0, &kernel));
    let mut logits = prefill.expect("prefill");
    if tier == KernelTier::Gpu {
        assert!(
            !fell_back && model.gpu_path_active(),
            "the GPU arm's prefill did not run on the fused Metal batch path (fallback warning: \
             {fell_back}, reason {fallback_reason:?}; device-KV latch set: {})",
            model.gpu_path_active()
        );
    }
    let mut generated = Vec::with_capacity(max_tokens);
    let mut all_logits = Vec::with_capacity(max_tokens);
    for step in 0..max_tokens {
        let tok = argmax_first(&logits);
        generated.push(tok);
        all_logits.push(logits.clone());
        let pos = prompt_ids.len() + step;
        logits = model.forward(tok, pos, &kernel).expect("decode step");
    }
    (generated, all_logits)
}

// ── Cross-tier per-step comparison (shared with the runtime-crate sibling) ─
//
// The comparison logic (token-chain-first, then bit-exact-CPU /
// relative-bound-GPU per step) lives in `oxibonsai_testkit::parity`, shared
// with `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs` rather than
// duplicated — see that module's doc comment for the full predicate and the
// measured (model, slot) points `REL_TOL` is calibrated against.
use oxibonsai_testkit::parity::{
    argmax_first, compare_against_reference, gpu_serial, run_named_test_in_child, TierRun,
};

/// Is this pair's non-reference side a CPU tier? Only an accelerator pair
/// gets a numeric tolerance; every CPU tier must be bit-exact against
/// `KernelTier::Reference`. Kept local (not in `oxibonsai_testkit::parity`,
/// which never depends on `oxibonsai-kernels`) and passed to
/// `compare_against_reference` as a closure.
///
/// Written as an exhaustive match over the named CPU tiers, deliberately
/// NOT as `!matches!(tier, KernelTier::Gpu)`: `KernelTier` is not
/// `#[non_exhaustive]`, so a future accelerator variant added under the
/// negated form would silently inherit the bit-exactness assertion and go
/// red for a non-bug, whereas this form fails to COMPILE here and forces
/// the classification to be decided.
fn is_cpu_tier(tier: KernelTier) -> bool {
    match tier {
        KernelTier::Reference => true,
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx2 | KernelTier::Avx512 => true,
        #[cfg(target_arch = "aarch64")]
        KernelTier::Neon => true,
        KernelTier::Gpu => false,
    }
}

/// Runs the full parity gate for one legacy model: greedily decode
/// `KernelTier::Reference`, the auto-detected CPU tier (NEON on this Apple
/// Silicon host), and `KernelTier::Gpu` (Metal) on three deterministic
/// synthetic prompts, and compare every tier against the scalar Reference
/// tier under D-6 — token chain first (hard, unconditional), then bit-exact
/// logits for the CPU pair / the relative bound for the GPU pair.
fn run_legacy_model_gate(model: &LegacyModel, test_name: &str) {
    let _tier_env = TierEnvGuard::cleared();
    let gate_start = std::time::Instant::now();
    // Serialized against this binary's other real-model gates — see
    // [`gpu_serial`]. Taken before the fixture probe so the whole gate,
    // including its capability bookkeeping, is one critical section.
    let _gpu = gpu_serial();

    // D-6(4): this test is NOT `#[ignore]`d — it runs
    // whenever the models are present and self-skips (reported, never green)
    // when they are not. This is a missing-MODEL-FILE skip, not a
    // missing-Metal-hardware skip, hence `Capability::LegacyModels`:
    // recording it as `Metal` let an unrelated Metal-hardware record
    // elsewhere in the manifest (e.g. `metal_k_quant_gemv_parity.rs`) satisfy
    // the release gate's `metal` check regardless of whether this gate ever
    // ran. `find_model` is one `fs::metadata` call, so a models-less host
    // pays nothing for the `#[ignore]` removal.
    let Some(model_path) = find_model(model.file_name) else {
        eprintln!(
            "skip: {} not found under {:?}",
            model.file_name,
            models_dir()
        );
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };

    let gguf_bytes = std::fs::read(&model_path).expect("read real legacy GGUF");

    let tiers: Vec<KernelTier> = {
        let mut v = vec![KernelTier::Reference, cpu_kernel_tier(), KernelTier::Gpu];
        v.dedup();
        v
    };

    for slot in 0..3 {
        let prompt_ids = synthetic_prompt(slot);

        let mut per_tier: Vec<TierRun<KernelTier>> = Vec::with_capacity(tiers.len());
        for &tier in &tiers {
            let (tokens, logits) = generate_with_logits(&gguf_bytes, tier, &prompt_ids, MAX_TOKENS);
            per_tier.push(TierRun {
                tier,
                tokens,
                logits,
            });
        }

        let (reference, others) = per_tier
            .split_first()
            .expect("the tier matrix always starts with KernelTier::Reference");
        let context = format!("{} prompt slot {slot}", model.file_name);
        for other in others {
            compare_against_reference(&context, reference, other, is_cpu_tier);
        }
    }

    // T-05 honesty: the `executed: true` records are written HERE, once the
    // whole tier matrix — every slot, every tier — has actually completed.
    // Writing them on model-file PRESENCE instead would claim coverage for
    // work that may still panic, which is the exact false-green class the
    // capability manifest exists to eliminate. Any assertion above panics
    // out of this function and neither record is written.
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, test_name, elapsed);
    record_executed_timed(Capability::LegacyModels, test_name, elapsed);
}

#[test]
fn ternary_1_7b_greedy_parity_across_tiers() {
    run_legacy_model_gate(
        &LEGACY_MODELS[0],
        "oxibonsai-model::legacy_parity_tests::ternary_1_7b_greedy_parity_across_tiers",
    );
}

#[test]
fn ternary_8b_greedy_parity_across_tiers() {
    run_legacy_model_gate(
        &LEGACY_MODELS[1],
        "oxibonsai-model::legacy_parity_tests::ternary_8b_greedy_parity_across_tiers",
    );
}

#[test]
fn bonsai_8b_greedy_parity_across_tiers() {
    run_legacy_model_gate(
        &LEGACY_MODELS[2],
        "oxibonsai-model::legacy_parity_tests::bonsai_8b_greedy_parity_across_tiers",
    );
}

// ─── M-18: prefill chunk-size measurement on a real long prompt ────────────
//
// `BonsaiModel::prefill_chunk_tokens()` defaults to
// `DEFAULT_PREFILL_CHUNK_TOKENS = 4096` (`chunked_prefill.rs`). This drives
// a real long prompt through the Metal `forward_prefill` path on
// `Bonsai-8B.gguf` at several chunk-token settings and measures each. The
// fused Metal batch-prefill kernel
// (`oxibonsai_kernels::gpu_backend::metal_prefill::functions_2::try_metal_full_forward_prefill`
// -> `metal_graph::buffers::commit_and_wait` -> `[MTLCommandBuffer
// waitUntilCompleted]`) is measurably expensive per call on this host, so
// EVERY arm runs in its OWN re-exec'd CHILD PROCESS: this gives each arm an
// identical fresh-process condition (no possible GPU-state carryover
// between arms), and the PARENT enforces its own wall-clock deadline per
// child, so a stall fails this test cleanly instead of hanging the gate.
//
// The measurement, and the decision it supports. The fused path is slower
// than sequential decode at every measured size on this hardware; the chunk
// size is not the lever. At `M18_PROMPT_LEN = 256` a measured run printed
// `chunk_tokens=128 prefill_us_per_token=61969.7 decode_us_per_token=52331.4`
// and `chunk_tokens=0 prefill_us_per_token=84313.8
// decode_us_per_token=53363.4` (1.18x slower than sequential single-token
// GPU decode at chunk=128, 1.58x one-shot), and the per-token cost RISES
// with the tokens of a single fused call (roughly 52.8k us at one 64-token
// call, 62.0k averaged over two 128-token calls, 84.3k at one 256-token
// call, 142.1k at one 512-token call; one 4352-token call does not finish
// within 900 s). A fixed per-call cost cannot explain that — it would make
// the per-token rate FALL as the call grows, and two 128-token calls would
// not beat one 256-token call, which they do (15.9 s vs 21.6 s). Smaller
// chunks only limit the loss, they never remove it, so
// `DEFAULT_PREFILL_CHUNK_TOKENS` stays 4096 and the fix belongs to the fused
// Metal prefill path itself. This test asserts only that chunk size never
// changes the numerics; it prints the rates every run so the measurement
// stays visible, and it gates nothing on them.
const M18_CHILD_ENV: &str = "OXIBONSAI_M18_CHILD_CHUNK";

/// Per-child deadline: `max(5x the slowest observed per-arm child time,
/// 120s)`. A measured run on this host observed child wall times of 17.57s
/// (`chunk_tokens=128`) and 27.74s (`chunk_tokens=0`, which also runs the
/// positive control) — `5 * 27.74 = 138.7`, so `139` rounds that up to a
/// whole second.
const M18_CHILD_DEADLINE_SECS: u64 = 139;

/// Chunk-token settings to measure: 128 (spans exactly 2 dispatches at
/// `M18_PROMPT_LEN`) and 0 (one-shot). Deliberately far below
/// `DEFAULT_PREFILL_CHUNK_TOKENS`-scale chunks: a single fused call at 512
/// tokens measures ~73s on this host, and a single fused call at 4352
/// tokens does not complete within a 900s bounded wait (GPU utilization
/// 100% and the thread inside `[MTLCommandBuffer waitUntilCompleted]` at
/// that point) — a dozens-of-chunks-at-4096-scale test would cost minutes
/// per child at best.
const M18_ARM_CHUNK_SIZES: [usize; 2] = [128, 0];
/// Exceeds `M18_ARM_CHUNK_SIZES`'s only non-zero value (128), so that arm
/// spans exactly two dispatches — the invariant this test exists to
/// measure.
const M18_PROMPT_LEN: usize = 256;
const M18_MAX_SEQ: usize = 4608;
const M18_CONTINUE_TOKENS: usize = 8;

fn m18_long_synthetic_prompt(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| 1 + (i % 1999)).collect()
}

/// The two highest values in `logits`, largest first — a diagnostic for a
/// token-chain divergence: a near-zero margin is benign FP noise near a
/// decision boundary, a wide margin is a real behavioural change.
fn m18_top2_margin(logits: &[f32]) -> (f32, f32) {
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    (
        sorted.first().copied().unwrap_or(f32::NAN),
        sorted.get(1).copied().unwrap_or(f32::NAN),
    )
}

/// Captures the `error` field of the one event [`M18WarnDetector`] matches.
struct M18ErrorFieldVisitor<'a>(&'a mut Option<String>);
impl tracing::field::Visit for M18ErrorFieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "error" {
            *self.0 = Some(format!("{value:?}"));
        }
    }
}

/// Minimal `tracing::Subscriber` that records whether a WARN-or-worse event
/// fired from `prefill_dispatch` (module-path filtered) while installed,
/// plus its `error` field — a hand-rolled subscriber, not
/// `tracing-subscriber`'s `fmt`, because `oxibonsai-model`'s `Cargo.toml`
/// does not carry that crate as a dev-dependency and `tracing` alone is
/// enough to implement one.
struct M18WarnDetector {
    saw_warn: std::sync::Arc<std::sync::atomic::AtomicBool>,
    reason: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}
impl tracing::Subscriber for M18WarnDetector {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.level() <= &tracing::Level::WARN
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let metadata = event.metadata();
        let is_prefill_dispatch_warn = *metadata.level() <= tracing::Level::WARN
            && metadata
                .module_path()
                .is_some_and(|m| m.contains("prefill_dispatch"));
        if is_prefill_dispatch_warn {
            self.saw_warn
                .store(true, std::sync::atomic::Ordering::SeqCst);
            let mut reason = None;
            event.record(&mut M18ErrorFieldVisitor(&mut reason));
            if let Some(reason) = reason {
                *self.reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason);
            }
        }
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

fn m18_run_detecting_warn<T>(f: impl FnOnce() -> T) -> (T, bool, Option<String>) {
    let saw_warn = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reason = std::sync::Arc::new(std::sync::Mutex::new(None));
    let detector = M18WarnDetector {
        saw_warn: saw_warn.clone(),
        reason: reason.clone(),
    };
    let result = tracing::subscriber::with_default(detector, f);
    let fell_back = saw_warn.load(std::sync::atomic::Ordering::SeqCst);
    let captured_reason = reason.lock().unwrap_or_else(|e| e.into_inner()).clone();
    (result, fell_back, captured_reason)
}

/// Run ONE M-18 arm (this same test, env-gated onto its child branch for
/// `chunk_tokens`) in a fresh child process under the per-child deadline,
/// through the shared [`run_named_test_in_child`]. `None` means "killed at
/// the deadline", distinct from `Some(status)` with `!status.success()` — a
/// real failure inside the child; the two are never reported with the same
/// message.
fn m18_run_one_arm_in_child(chunk_tokens: usize) -> (Option<std::process::ExitStatus>, String) {
    let chunk = chunk_tokens.to_string();
    let run = run_named_test_in_child(
        "prefill_chunk_size_is_a_dispatch_knob_not_a_numerics_knob_on_a_real_long_prompt",
        &[(M18_CHILD_ENV, chunk.as_str())],
        std::time::Duration::from_secs(M18_CHILD_DEADLINE_SECS),
    )
    .unwrap_or_else(|e| panic!("spawning the M-18 chunk_tokens={chunk_tokens} child: {e}"));
    (run.status, run.output)
}

#[test]
fn prefill_chunk_size_is_a_dispatch_knob_not_a_numerics_knob_on_a_real_long_prompt() {
    const TEST: &str = "oxibonsai-model::legacy_parity_tests::\
                        prefill_chunk_size_is_a_dispatch_knob_not_a_numerics_knob_on_a_real_long_prompt";
    let _tier_env = TierEnvGuard::cleared();

    // Child branch: this exact process was re-exec'd to run ONE arm (chunk
    // size from `M18_CHILD_ENV`), print machine-readable lines, and exit —
    // its own pass/fail becomes the parent's per-child `status.success()`.
    if let Ok(chunk_str) = std::env::var(M18_CHILD_ENV) {
        let chunk_tokens: usize = chunk_str
            .parse()
            .unwrap_or_else(|e| panic!("{M18_CHILD_ENV}={chunk_str:?} not a valid usize: {e}"));
        run_m18_child_arm(chunk_tokens);
        return;
    }

    // Serialized against this binary's other real-model GPU gates (see
    // `gpu_serial`'s own doc comment) for the whole multi-child sequence.
    let _gpu = gpu_serial();
    let gate_start = std::time::Instant::now();

    if find_model("Bonsai-8B.gguf").is_none() {
        eprintln!("skip: Bonsai-8B.gguf not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }

    // Enforce the invariant `M18_ARM_CHUNK_SIZES`'s own doc comment
    // describes: at least two DISTINCT values (an all-equal array would
    // compare a pair that proves nothing about chunking), and every
    // non-zero chunk size strictly smaller than the prompt (so each
    // chunked arm spans at least two dispatches).
    assert!(
        M18_ARM_CHUNK_SIZES
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "M18_ARM_CHUNK_SIZES = {M18_ARM_CHUNK_SIZES:?} has fewer than two distinct values"
    );
    for &c in &M18_ARM_CHUNK_SIZES {
        assert!(
            c == 0 || c < M18_PROMPT_LEN,
            "chunk_tokens={c} in M18_ARM_CHUNK_SIZES is >= M18_PROMPT_LEN={M18_PROMPT_LEN}"
        );
    }

    let mut reference: Option<Vec<u32>> = None;
    let mut measurements: Vec<(usize, bool)> = Vec::with_capacity(M18_ARM_CHUNK_SIZES.len());

    for &chunk_tokens in &M18_ARM_CHUNK_SIZES {
        let (status, output) = m18_run_one_arm_in_child(chunk_tokens);
        eprint!("{output}");
        let status = status.unwrap_or_else(|| {
            panic!(
                "M-18 arm chunk_tokens={chunk_tokens} did not complete within \
                 {M18_CHILD_DEADLINE_SECS}s and was killed -- \
                 oxibonsai_kernels::gpu_backend::metal_graph::buffers::commit_and_wait (a Metal \
                 command buffer wait) did not return in time, NOT a numerics bug in this test. \
                 Child output was:\n{output}"
            )
        });
        assert!(
            status.success(),
            "M-18 arm chunk_tokens={chunk_tokens} exited with {status:?} (a real failure \
             inside the child, not a timeout). Child output was:\n{output}"
        );

        let fell_back = output.contains("M18_FELL_BACK=true");
        measurements.push((chunk_tokens, fell_back));
        if chunk_tokens != 0 {
            assert!(
                !fell_back,
                "prefill_chunk_tokens={chunk_tokens} fell back off the fused Metal batch \
                 prefill path in its own child. Child output was:\n{output}"
            );
        } else {
            // The child only reaches its positive control after its own
            // arm succeeds, so requiring this line here also enforces
            // that `M18WarnDetector` was proven to work in this run, not
            // only printed as a side effect the parent never checks.
            assert!(
                output.contains("M18_CONTROL fell_back=true"),
                "chunk_tokens=0 child output has no \"M18_CONTROL fell_back=true\" line -- the \
                 positive control did not run or did not prove the detector works in this \
                 process. Child output was:\n{output}"
            );
        }
        if fell_back {
            continue; // not compared against the fused-path reference below
        }

        let tokens_line = output
            .lines()
            .find_map(|l| l.strip_prefix("M18_TOKENS="))
            .unwrap_or_else(|| panic!("child output has no M18_TOKENS= line:\n{output}"));
        let tokens: Vec<u32> = tokens_line
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("M18_TOKENS entry not a valid u32"))
            .collect();
        match &reference {
            None => reference = Some(tokens),
            Some(ref_tokens) => {
                assert_eq!(
                    &tokens, ref_tokens,
                    "prefill_chunk_tokens={chunk_tokens} produced a DIFFERENT greedy \
                     continuation than the reference (chunk size must be a pure \
                     dispatch-granularity knob, never a numerics knob). This run's child \
                     output:\n{output}"
                );
            }
        }
    }

    eprintln!("M18_SUMMARY {measurements:?}");
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, TEST, elapsed);
    record_executed_timed(Capability::LegacyModels, TEST, elapsed);
}

/// One arm's real work, run only inside its own re-exec'd child: load the
/// model fresh, upload weights, prefill `M18_PROMPT_LEN` tokens at
/// `chunk_tokens`, decode `M18_CONTINUE_TOKENS` more, print the result.
/// Every arm pays its own first-use pipeline-compilation cost, since each
/// arm is its own fresh process with no shared warmup — the printed
/// per-arm timings below include that cost, not only steady-state work.
fn run_m18_child_arm(chunk_tokens: usize) {
    let Some(model_path) = find_model("Bonsai-8B.gguf") else {
        panic!("M-18 arm child could not find Bonsai-8B.gguf though the parent just did");
    };
    let gguf_bytes = std::fs::read(&model_path).expect("read real Bonsai-8B.gguf");
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse real legacy GGUF");
    // `auto_detect`, not `with_tier(KernelTier::Gpu)`: the fused Metal
    // batch path needs GPU-resident weight handles from
    // `upload_weights_to_gpu`, which in turn needs a dispatcher wired to a
    // live GPU backend (`Scirs2BackendHandle`) -- `auto_detect` is the
    // only public constructor that does that
    // (`metal_prefill_q1_parity_tests.rs`'s own batched-Q1-prefill tests
    // document and rely on the same distinction). `with_tier(Gpu)` alone
    // gives "missing GPU handle" here.
    let kernel = KernelDispatcher::auto_detect();
    assert_eq!(
        kernel.tier(),
        KernelTier::Gpu,
        "KernelDispatcher::auto_detect() did not select the GPU tier in an M-18 arm child"
    );

    let mut model = BonsaiModel::from_gguf(&gguf, M18_MAX_SEQ).expect("BonsaiModel::from_gguf");
    // Required before the fused Metal batch path has real GPU handles to
    // read -- `upload_weights_to_gpu`'s own doc comment: "should be called
    // once after model loading and before the first forward pass";
    // production code does this in `InferenceEngine::from_gguf`,
    // `engine.rs:874`.
    model.upload_weights_to_gpu(&kernel);
    model.set_prefill_chunk_tokens(chunk_tokens);
    let prompt = m18_long_synthetic_prompt(M18_PROMPT_LEN);

    let prefill_start = std::time::Instant::now();
    let (prefill_result, fell_back, fallback_reason) =
        m18_run_detecting_warn(|| model.forward_prefill(&prompt, 0, &kernel));
    let mut logits = prefill_result.unwrap_or_else(|e| {
        panic!(
            "prefill at chunk_tokens={chunk_tokens} on a {}-token prompt: {e}",
            prompt.len()
        )
    });
    let prefill_elapsed = prefill_start.elapsed();
    println!(
        "M18_MEASURED chunk_tokens={chunk_tokens} prompt_len={} prefill_elapsed_ms={} \
         M18_FELL_BACK={fell_back} reason={fallback_reason:?}",
        prompt.len(),
        prefill_elapsed.as_millis()
    );

    let mut continuation = Vec::with_capacity(M18_CONTINUE_TOKENS);
    let mut margins = Vec::with_capacity(M18_CONTINUE_TOKENS);
    let decode_start = std::time::Instant::now();
    for step in 0..M18_CONTINUE_TOKENS {
        let tok = argmax_first(&logits);
        continuation.push(tok);
        // The chunked arm (two 128-token dispatches) and the one-shot arm
        // (one 256-token dispatch) reduce in different orders, so a
        // near-tie argmax flip between them is possible without being a
        // real chunking bug. Printed alongside the tokens (both children's
        // full output lands in a mismatch panic via the parent) so that
        // possibility is diagnosable rather than only asserted away.
        let (top1, top2) = m18_top2_margin(&logits);
        margins.push(top1 - top2);
        let pos = prompt.len() + step;
        logits = model.forward(tok, pos, &kernel).expect("decode step");
    }
    let decode_elapsed = decode_start.elapsed();
    let prefill_us_per_token = prefill_elapsed.as_micros() as f64 / prompt.len() as f64;
    let decode_us_per_token = decode_elapsed.as_micros() as f64 / M18_CONTINUE_TOKENS as f64;
    println!(
        "M18_RATE chunk_tokens={chunk_tokens} prefill_us_per_token={prefill_us_per_token:.1} \
         decode_us_per_token={decode_us_per_token:.1} (fused batched prefill vs sequential \
         single-token decode; printed for comparison, not asserted)"
    );
    println!(
        "M18_TOKENS={}",
        continuation
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "M18_MARGINS={}",
        margins
            .iter()
            .map(|m| format!("{m:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    // Positive control LAST, only for the chunk_tokens==0 arm's child (any
    // one arm proves the detector works in a process of this exact shape;
    // running it in every arm's child would just repeat the same proof).
    if chunk_tokens == 0 {
        let mut control_model =
            BonsaiModel::from_gguf(&gguf, M18_MAX_SEQ).expect("BonsaiModel::from_gguf (control)");
        let (control_result, control_fell_back, control_reason) = m18_run_detecting_warn(|| {
            control_model.forward_prefill(&synthetic_prompt(0), 0, &kernel)
        });
        control_result.expect("positive control prefill should succeed via the fallback");
        assert!(
            control_fell_back,
            "positive control failed: a freshly loaded model with no uploaded GPU weight \
             handles did not trigger forward_prefill's fallback warning"
        );
        println!("M18_CONTROL fell_back=true reason={control_reason:?}");
    }
}

// ─── M-08: YaRN wiring on the real Bonsai-8B.gguf (fast control) ───────────
//
// `BonsaiModel::from_gguf`'s real load path honours the file's
// `qwen3.rope.scaling.*` keys (`model/types/constructors.rs`'s
// `from_gguf_with_embd_and_cap` calls the SCALED `build_rope_table`, tagged
// M-08). This control test proves, in seconds, that the scaling really
// changes the model's output: an UNSCALED counterpart is loaded from a
// private in-memory copy of `models/Bonsai-8B.gguf` whose one
// `qwen3.rope.scaling.factor` value is patched from 4.0 to 1.0
// (`oxibonsai_testkit::gguf_fixture::patch_gguf_f32_metadata`, which checks
// the key's length prefix, its FLOAT32 type tag and its current value before
// writing anything), and the two arms' logit signatures are compared over a
// short prompt, each arm in its own fresh child process (the Metal decode
// path is a process-global singleton).
//
// factor=1.0, not a removed key, is deliberate:
// `RopeTable::new_with_scaling`'s own
// `new_with_scaling_yarn_factor_one_matches_plain_new` test
// (`crates/oxibonsai-model/src/layers/rope.rs`) proves `factor: 1.0` is
// accepted (a sub-unity factor errors) and reproduces plain `RopeTable::new`
// to within f32/f64-pipeline rounding, so both arms go through the same
// `from_gguf` -> `Qwen3Config::from_metadata` -> `build_rope_table` ->
// `RopeScalingStrategy::Yarn` path and differ only in the one patched value.
// The usable context is unaffected: `max_context_length` comes from the
// independent `<arch>.context_length` key
// (`crates/oxibonsai-core/src/config.rs`), never from `rope.scaling.*`, and
// each arm's effective context is asserted at runtime as well.
//
// The long-context half of M-08 — the same scaled-vs-unscaled comparison
// after a 20000-token natural-text prompt, past the 16384-token original
// context — needs the real tokenizer, which this crate does not depend on,
// so it lives in `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs`
// (`m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens`, opt-in
// via `OXIBONSAI_M08_RUN_LONG=1`).
const M08_CHILD_ARM_ENV: &str = "OXIBONSAI_M08_CHILD_ARM";
const M08_CONTROL_TEST_FN: &str = "m08_yarn_scaling_wiring_takes_effect_on_the_real_bonsai_8b_file";
const M08_FACTOR_KEY: &str = "qwen3.rope.scaling.factor";
const M08_SHIPPED_FACTOR: f32 = 4.0;

/// Context budget each arm is loaded with: past 20000 positions, so the
/// effective-context assertion also shows the patch leaves the usable
/// context of a long-context run intact.
const M08_MAX_SEQ: usize = 20_064;
const M08_CONTROL_LEN: usize = 8;

/// Hang guard per control child, not a target: a real control child takes
/// about 3 s (measured ~2.9 s and ~2.5 s for the two arms).
const M08_CONTROL_CHILD_DEADLINE_SECS: u64 = 300;

/// Deterministic pseudo-random prompt ids well below the real vocabulary
/// size — enough for the control, which compares logit signatures position
/// by position rather than any continuation.
fn m08_control_prompt(len: usize, vocab_ceiling: u32) -> Vec<u32> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    (0..len as u32)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            1 + ((state >> 33) as u32 % (vocab_ceiling - 1))
        })
        .collect()
}

/// `arm`'s GGUF bytes: the shipped file, or a private copy with
/// `qwen3.rope.scaling.factor` patched from 4.0 to 1.0.
fn m08_load_gguf_bytes(arm: &str) -> Vec<u8> {
    let model_path = find_model("Bonsai-8B.gguf")
        .unwrap_or_else(|| panic!("M-08 {arm} child could not find Bonsai-8B.gguf"));
    let mut bytes = std::fs::read(&model_path).expect("read real Bonsai-8B.gguf");
    match arm {
        "scaled" => {}
        "unscaled" => oxibonsai_testkit::gguf_fixture::patch_gguf_f32_metadata(
            &mut bytes,
            M08_FACTOR_KEY,
            M08_SHIPPED_FACTOR,
            1.0,
        )
        .unwrap_or_else(|e| panic!("patch {M08_FACTOR_KEY} to 1.0: {e}")),
        other => panic!("unknown M-08 arm {other:?}"),
    }
    bytes
}

/// Loads `arm`'s model on the auto-detected GPU tier with its weights
/// uploaded, asserts the effective context, and returns it ready to decode.
/// `Box::leak`: this runs only inside a short-lived child process dedicated
/// to one arm, which exits right after finishing.
fn m08_load_arm(arm: &str) -> (BonsaiModel<'static>, KernelDispatcher) {
    let bytes: &'static [u8] = Box::leak(m08_load_gguf_bytes(arm).into_boxed_slice());
    let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
        GgufFile::parse(bytes).expect("parse real legacy GGUF"),
    ));

    let kernel = KernelDispatcher::auto_detect();
    assert_eq!(
        kernel.tier(),
        KernelTier::Gpu,
        "KernelDispatcher::auto_detect() did not select the GPU tier in an M-08 {arm} child"
    );
    let mut model = BonsaiModel::from_gguf(gguf, M08_MAX_SEQ).expect("BonsaiModel::from_gguf");
    model.upload_weights_to_gpu(&kernel);

    let effective_context = model.max_context();
    println!("M08_EFFECTIVE_CONTEXT arm={arm} value={effective_context}");
    assert!(
        effective_context >= M08_MAX_SEQ,
        "M-08 {arm} child's effective context ({effective_context}) is below M08_MAX_SEQ \
         ({M08_MAX_SEQ}) -- the byte patch may have (incorrectly) capped the usable context"
    );

    (model, kernel)
}

/// A short, cheap signature of a logits vector: the argmax index/value
/// (what greedy decoding actually reads) plus the sum over all logits (a
/// single number sensitive to almost any change anywhere in the vector,
/// including one that happens not to move the argmax). Printed instead of
/// the full vector, which for a real vocab is far too large for a log
/// line.
fn m08_logit_signature(logits: &[f32]) -> (u32, f32, f32) {
    let argmax = argmax_first(logits);
    let argmax_val = logits[argmax as usize];
    let sum: f32 = logits.iter().sum();
    (argmax, argmax_val, sum)
}

fn run_m08_control_child(arm: &str) {
    let (mut model, kernel) = m08_load_arm(arm);
    let prompt = m08_control_prompt(M08_CONTROL_LEN, 100_000);
    let mut logits = model
        .forward(prompt[0], 0, &kernel)
        .expect("M-08 control decode step 0");
    for pos in 0..M08_CONTROL_LEN {
        let (argmax, argmax_val, sum) = m08_logit_signature(&logits);
        println!(
            "M08_CONTROL_SIGNATURE arm={arm} pos={pos} argmax={argmax} argmax_val={argmax_val} \
             sum={sum}"
        );
        if pos + 1 < M08_CONTROL_LEN {
            logits = model
                .forward(prompt[pos + 1], pos + 1, &kernel)
                .unwrap_or_else(|e| panic!("M-08 control decode step {}: {e}", pos + 1));
        }
    }
}

#[test]
fn m08_yarn_scaling_wiring_takes_effect_on_the_real_bonsai_8b_file() {
    const TEST: &str =
        "oxibonsai-model::legacy_parity_tests::m08_yarn_scaling_wiring_takes_effect_on_the_real_bonsai_8b_file";
    let _tier_env = TierEnvGuard::cleared();

    if let Ok(arm) = std::env::var(M08_CHILD_ARM_ENV) {
        run_m08_control_child(&arm);
        return;
    }

    let _gpu = gpu_serial();
    let gate_start = std::time::Instant::now();

    if find_model("Bonsai-8B.gguf").is_none() {
        eprintln!("skip: Bonsai-8B.gguf not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }

    let mut signatures: std::collections::HashMap<&str, Vec<(u32, f32, f32)>> =
        std::collections::HashMap::new();
    for arm in ["scaled", "unscaled"] {
        let run = run_named_test_in_child(
            M08_CONTROL_TEST_FN,
            &[(M08_CHILD_ARM_ENV, arm)],
            std::time::Duration::from_secs(M08_CONTROL_CHILD_DEADLINE_SECS),
        )
        .unwrap_or_else(|e| panic!("spawning the M-08 control child for arm={arm}: {e}"));
        let output = run.output;
        eprint!("{output}");
        let status = run.status.unwrap_or_else(|| {
            panic!(
                "M-08 control child for arm={arm} did not complete within \
                 {M08_CONTROL_CHILD_DEADLINE_SECS}s. Output was:\n{output}"
            )
        });
        assert!(
            status.success(),
            "M-08 control child for arm={arm} exited with {status:?}. Output was:\n{output}"
        );

        let effective_context: usize = output
            .lines()
            .find_map(|l| l.strip_prefix(&format!("M08_EFFECTIVE_CONTEXT arm={arm} value=")))
            .unwrap_or_else(|| panic!("no M08_EFFECTIVE_CONTEXT line for arm={arm}:\n{output}"))
            .trim()
            .parse()
            .expect("M08_EFFECTIVE_CONTEXT value not a valid usize");
        assert!(
            effective_context >= M08_MAX_SEQ,
            "arm={arm} effective_context={effective_context} < M08_MAX_SEQ={M08_MAX_SEQ}"
        );

        let prefix = format!("M08_CONTROL_SIGNATURE arm={arm} pos=");
        let mut sigs = Vec::with_capacity(M08_CONTROL_LEN);
        for line in output.lines().filter(|l| l.starts_with(&prefix)) {
            let argmax: u32 = line
                .split("argmax=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .expect("argmax field")
                .parse()
                .expect("argmax not a valid u32");
            let argmax_val: f32 = line
                .split("argmax_val=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .expect("argmax_val field")
                .parse()
                .expect("argmax_val not a valid f32");
            let sum: f32 = line
                .split("sum=")
                .nth(1)
                .expect("sum field")
                .trim()
                .parse()
                .expect("sum not a valid f32");
            sigs.push((argmax, argmax_val, sum));
        }
        assert_eq!(
            sigs.len(),
            M08_CONTROL_LEN,
            "arm={arm} produced {} M08_CONTROL_SIGNATURE lines, expected {M08_CONTROL_LEN}:\n{output}",
            sigs.len()
        );
        signatures.insert(arm, sigs);
    }

    let scaled = &signatures["scaled"];
    let unscaled = &signatures["unscaled"];

    // Position 0: softmax over a single cached key is 1 regardless of any
    // RoPE angle, so both arms must agree here -- a difference here would
    // mean something OTHER than RoPE scaling changed between the two
    // loads (e.g. a non-deterministic kernel), not evidence the patch
    // "worked".
    assert!(
        (scaled[0].2 - unscaled[0].2).abs() < 1.0,
        "position-0 logit sums differ between arms (scaled={}, unscaled={}) -- position 0 \
         should be scaling-invariant by construction; something other than RoPE changed",
        scaled[0].2,
        unscaled[0].2
    );

    // Position >=1: the actual proof the patch took effect.
    let any_differ_from_pos1 = scaled[1..]
        .iter()
        .zip(unscaled[1..].iter())
        .any(|(s, u)| (s.2 - u.2).abs() > 1.0 || s.0 != u.0);
    assert!(
        any_differ_from_pos1,
        "scaled and unscaled arms produced IDENTICAL logit signatures at every position from 1 \
         to {} -- the qwen3.rope.scaling.factor byte patch did not change the model's real \
         output. scaled={scaled:?} unscaled={unscaled:?}",
        M08_CONTROL_LEN - 1
    );

    eprintln!("M08_CONTROL_SUMMARY scaled={scaled:?} unscaled={unscaled:?}");
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, TEST, elapsed);
    record_executed_timed(Capability::LegacyModels, TEST, elapsed);
}

// ─── M-02: measured RSS delta from lazy embeddings, on a real model ────────
//
// `EmbeddingTable`'s row-wise dequantization (`model/types/embedding.rs`)
// keeps a quantized `token_embd.weight` borrowed zero-copy (from the memory
// map in real production use, in this test from a `Vec<u8>` returned by
// `std::fs::read` — see below) and only decodes the rows a forward pass
// actually looks up. `BonsaiModel::footprint_bytes()` -- "the number the
// M-02 / M-33 memory acceptance tests read" per its own doc comment -- is
// exercised today only against synthetic fixtures (`model/types/tests.rs`).
// This is the same invariant against a real model file, plus a real
// OS-reported RSS delta: the retired `Index<Range>` escape hatch
// materialized 1.16 / 2.31 / 4.74 GiB of dense FP32 for 1.7B / 8B / 27B
// (see `embedding.rs`'s own header table; the 2.31 GiB figure is keyed to
// this model FAMILY's embedding shape, `[4096, 151669]`, shared by both
// `Bonsai-8B.gguf` and `Ternary-Bonsai-8B.gguf` regardless of their
// different on-disk file sizes).
//
// The RSS measurement runs in a FRESH CHILD PROCESS (re-exec of this same
// test binary, gated by `RSS_CHILD_ENV`), not inline: libtest runs this
// binary's tests alphabetically, so `bonsai_8b_greedy_parity_across_tiers`
// has already loaded a model at the GPU tier in this same process by the
// time this test runs, and its retained Metal/allocator state would
// contaminate an in-process RSS baseline. The child reads the GGUF with
// `std::fs::read` into a `Vec<u8>`
// (heap-resident, not a memory map) and parses it BEFORE the first RSS
// checkpoint, so the file's own bytes are excluded from every delta below —
// only `BonsaiModel::from_gguf` ("load") and the fused-path prefill call
// ("prefill") are being measured. Three checkpoints are taken: before
// `from_gguf`, after it, and after the prefill call.
//
// The hard assertion is on `BonsaiModel::footprint_bytes()` after the
// prefill — the model's own resident-memory accounting, deterministic and
// excluding everything borrowed from the file — against a budget far below
// the 2.31 GiB a materialized dense embedding would add. Every RSS number
// (the load delta, the prefill delta and the total) is a printed diagnostic
// only: on this unified-memory host the prefill delta is measured NEGATIVE
// (RSS falls by roughly 527 MiB across the prefill call), so a `< budget`
// assertion on it would be vacuous — see the comment at the assertion.
const RSS_CHILD_ENV: &str = "OXIBONSAI_RSS_CHILD";

fn process_rss_kib() -> u64 {
    let pid = std::process::id();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("spawning `ps -o rss= -p <pid>` should not itself fail to launch");
    assert!(
        output.status.success(),
        "ps -o rss= -p {pid} exited non-zero"
    );
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap_or_else(|e| {
            panic!(
                "ps -o rss= printed non-numeric output {:?}: {e}",
                String::from_utf8_lossy(&output.stdout)
            )
        })
}

#[test]
fn footprint_and_measured_rss_confirm_the_embedding_table_stays_unmaterialized_bonsai_8b() {
    const TEST: &str = "oxibonsai-model::legacy_parity_tests::\
                        footprint_and_measured_rss_confirm_the_embedding_table_stays_unmaterialized_bonsai_8b";
    let _tier_env = TierEnvGuard::cleared();

    // Fresh-process branch: measure and print, make no assertions (the
    // PARENT invocation below does), keep no state past `return`.
    if std::env::var(RSS_CHILD_ENV).is_ok() {
        let Some(model_path) = find_model("Bonsai-8B.gguf") else {
            panic!(
                "RSS child could not find Bonsai-8B.gguf though the parent invocation just did \
                 (environment changed between the two processes?)"
            );
        };
        // The file's own bytes are read and parsed BEFORE the first
        // checkpoint, so no delta below counts the GGUF's on-disk size —
        // only what `BonsaiModel::from_gguf` and the prefill call
        // themselves add.
        let gguf_bytes = std::fs::read(&model_path).expect("read real Bonsai-8B.gguf");
        let gguf = GgufFile::parse(&gguf_bytes).expect("parse real legacy GGUF");
        // `auto_detect`, not `with_tier(KernelTier::Gpu)` -- see
        // `run_m18_child_arm`'s identical comment for why.
        let kernel = KernelDispatcher::auto_detect();
        // Same reasoning as M-18's identical assertion: `auto_detect` is
        // not pinned to GPU, and `try_metal_prefill_with_lm_head` below
        // does not itself check the dispatcher's tier, so a silent
        // CPU-tier selection here would not even fail loudly -- it would
        // just measure the wrong thing.
        assert_eq!(
            kernel.tier(),
            KernelTier::Gpu,
            "KernelDispatcher::auto_detect() did not select the GPU tier on this host in the \
             RSS child -- the measurement below would silently be of the wrong path"
        );

        let rss_before_load_kib = process_rss_kib();
        let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
        // Required before the fused Metal batch path has real GPU handles
        // to read (see `run_m18_child_arm`'s identical comment for the
        // doc-comment citation and production call site); counted in
        // `load_delta`, not `prefill_delta`, since it is not what M-02 is
        // measuring.
        model.upload_weights_to_gpu(&kernel);
        let rss_after_load_kib = process_rss_kib();
        let prompt = synthetic_prompt(0);
        // Call the fused Metal batch path directly rather than through
        // `forward_prefill` (whose dispatcher silently falls back to a
        // sequential path — that never calls the batched `copy_rows`
        // gather this test is about — on any error and still returns
        // `Ok`). `try_metal_prefill_with_lm_head` is `pub` specifically
        // "so parity tests can invoke this strict path directly, bypassing
        // the silent fallback" (its own doc comment); its `Err` propagates
        // here instead of being swallowed, so there is no silent-fallback
        // gap to guard against separately.
        let _logits = model
            .try_metal_prefill_with_lm_head(&prompt, 0)
            .unwrap_or_else(|e| {
                panic!("try_metal_prefill_with_lm_head (the fused Metal batch path): {e}")
            });
        let rss_after_prefill_kib = process_rss_kib();
        println!("OXIBONSAI_RSS_BEFORE_LOAD_KIB={rss_before_load_kib}");
        println!("OXIBONSAI_RSS_AFTER_LOAD_KIB={rss_after_load_kib}");
        println!("OXIBONSAI_RSS_AFTER_PREFILL_KIB={rss_after_prefill_kib}");
        println!("OXIBONSAI_FOOTPRINT_BYTES={}", model.footprint_bytes());
        return;
    }

    let _gpu = gpu_serial();
    let gate_start = std::time::Instant::now();

    // Skip (and record) BEFORE paying for a re-exec, so a models-less host
    // still gets the usual skip bookkeeping.
    if find_model("Bonsai-8B.gguf").is_none() {
        eprintln!("skip: Bonsai-8B.gguf not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }

    let exe = std::env::current_exe().expect("current_exe for a fresh-process RSS re-exec");
    let output = std::process::Command::new(&exe)
        .args([
            "footprint_and_measured_rss_confirm_the_embedding_table_stays_unmaterialized_bonsai_8b",
            "--exact",
            "--nocapture",
        ])
        .env(RSS_CHILD_ENV, "1")
        .output()
        .expect("spawning a fresh child process of this same test binary for a clean RSS baseline");
    assert!(
        output.status.success(),
        "fresh-process RSS child failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parse_u64_after = |marker: &str| -> u64 {
        stdout
            .lines()
            .find_map(|l| l.strip_prefix(marker))
            .unwrap_or_else(|| panic!("child stdout missing a {marker}* line:\n{stdout}"))
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("{marker}* was not a valid integer: {e}\n{stdout}"))
    };
    let rss_before_load_kib = parse_u64_after("OXIBONSAI_RSS_BEFORE_LOAD_KIB=");
    let rss_after_load_kib = parse_u64_after("OXIBONSAI_RSS_AFTER_LOAD_KIB=");
    let rss_after_prefill_kib = parse_u64_after("OXIBONSAI_RSS_AFTER_PREFILL_KIB=");
    let footprint_bytes = parse_u64_after("OXIBONSAI_FOOTPRINT_BYTES=") as usize;
    let load_delta_kib = rss_after_load_kib.saturating_sub(rss_before_load_kib);
    let prefill_delta_kib = rss_after_prefill_kib.saturating_sub(rss_after_load_kib);
    let total_delta_kib = rss_after_prefill_kib.saturating_sub(rss_before_load_kib);

    eprintln!(
        "M-02 measured (fresh child process): Bonsai-8B.gguf RSS before_load={rss_before_load_kib} \
         KiB after_load={rss_after_load_kib} KiB ({:.3} GiB delta) after_prefill=\
         {rss_after_prefill_kib} KiB ({:.3} GiB delta) total={total_delta_kib} KiB \
         ({:.3} GiB); BonsaiModel::footprint_bytes()={footprint_bytes} bytes \
         ({:.6} GiB); old eager dense-embedding baseline for this model family \
         (hidden=4096) was 2.31 GiB (embedding.rs header table)",
        load_delta_kib as f64 / (1024.0 * 1024.0),
        prefill_delta_kib as f64 / (1024.0 * 1024.0),
        total_delta_kib as f64 / (1024.0 * 1024.0),
        footprint_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // The SOLE hard gate: `footprint_bytes` sums this model's own
    // `resident_bytes()` accounting (KV cache + RoPE table + scratch
    // buffers + any dense weight copy), explicitly EXCLUDING everything
    // borrowed from the memory map -- deterministic, unlike RSS on this
    // unified-memory host (see below), and exactly the API `embedding.rs`'s
    // own doc comment names as "the number the M-02 / M-33 memory
    // acceptance tests read". Budget: the measured value on this real
    // model (38,010,880 bytes) plus a 256 MiB margin -- comfortably
    // under the 2.31 GiB (~2,481,651,712-byte) old dense-embedding
    // baseline for this model family, since `footprint_bytes` is
    // deterministic here (a KV cache + RoPE table + scratch buffers at
    // MAX_SEQ=512 does not vary run to run), so a tight-relative-to-the-old-
    // baseline bound is still safe.
    const FOOTPRINT_BUDGET_BYTES: usize = 306_446_336; // 38_010_880 + 256 MiB
    assert!(
        footprint_bytes < FOOTPRINT_BUDGET_BYTES,
        "BonsaiModel::footprint_bytes() = {footprint_bytes} bytes after loading + one prefill \
         on Bonsai-8B.gguf, at or above the {FOOTPRINT_BUDGET_BYTES}-byte budget -- looks like a \
         dense weight copy (embedding table or otherwise) got materialized"
    );

    // RSS is deliberately NOT hard-gated: on this unified-memory host,
    // `prefill_delta_kib` is measured as a NEGATIVE delta (RSS falls by
    // roughly 527 MiB across the prefill call itself --
    // plausibly transient `upload_weights_to_gpu` staging settling out, or
    // OS-level page accounting around Metal buffer allocation/
    // deallocation timing). `saturating_sub` clamps that to a printed 0,
    // which would make any `< budget` assertion on it vacuously true
    // regardless of a real regression -- a re-introduced dense embedding
    // materialization could land inside the same net decrease and this
    // gate would never see it. `footprint_bytes` above does not have this
    // problem (it is this model's own accounting, not an OS-level
    // post-hoc sample), so it carries the real assertion; every RSS number
    // above remains a printed diagnostic only, honestly labelled as such.

    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, TEST, elapsed);
    record_executed_timed(Capability::LegacyModels, TEST, elapsed);
}

// ── Notes ────────────────────────────────────────────────────────────────
//
// 1. SCOPE CUT (see this file's header doc comment for the full rationale):
//    this test drives real model weights with deterministic *numeric*
//    prompts and asserts cross-tier agreement, not "matches the golden CLI
//    text" — `oxibonsai-model` has no tokenizer/detokenizer dependency to
//    turn real English prompts into ids or ids back into text. The
//    golden-text half lives in
//    `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs` instead, which
//    depends on `oxibonsai-model` and has `TokenizerBridge` +
//    `sampling::SamplingParams`: it embeds `legacy_golden.json`'s exact 9
//    `(model, prompt, golden Metal text)` triples (`backend == "metal"`) and
//    asserts `decode(generate(prompt))`'s first 32 tokens (the fork's own
//    capture budget) EXACTLY equal the golden text. This file's
//    numeric-prompt, tokenizer-free cross-tier gate remains valuable in its
//    own right (it still runs with no tokenizer fixture at all).
// 1a. The three real-model tests are not `#[ignore]`d: they run whenever the
//     models are present and record a `LegacyModels`-capability
//     `executed: false` self-skip (below) when they are not, so a missing
//     model reads as SKIPPED rather than silently PASSED. Same shape in the
//     sibling `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs`.
// 1b. The D-6 comparison block (`compare_against_reference` and its
//     supporting types/constants) is shared with that sibling through
//     `oxibonsai_testkit::parity` rather than duplicated.
// 1c. Cross-PROCESS serialization: [`gpu_serial`] only serializes these
//     gates inside one test binary, which is what `cargo test`'s in-process
//     thread pool needs — `cargo nextest` runs each test in its own process,
//     so this alone could not stop a `--workspace` nextest run from
//     scheduling several multi-GB real-model gates concurrently.
//     `.config/nextest.toml` pins this whole binary to its single-threaded
//     `real-model-gate` test group, so a nextest run executes these cases
//     one at a time and never alongside another real-model case;
//     `scripts/release-gate.sh`'s stage 1b also invokes the binary
//     directly, serialized, in release, with `--test-threads=1`, which is
//     where the release gate collects its evidence.
// 2. `build_sampling_params`-equivalence: true greedy (temperature 0, no
//    penalty) does not need `SamplingParams` at all at this crate's level —
//    `argmax_first` on raw logits *is* the CLI's post-fix greedy contract
//    exactly. If a `SamplingParams` re-enters the picture here, construct it
//    with `repetition_penalty: 1.0` (not `SamplingParams::default()`'s
//    `1.1`) to match `src/cli/util.rs::build_sampling_params`'s documented
//    behaviour with no explicit `--repetition-penalty` flag.
// 3. The "models present" branch is runtime-verified on the real GGUFs via
//    `OXIBONSAI_MODELS_DIR=<checkout>/models cargo test --release -p
//    oxibonsai-model --features metal --test legacy_parity_tests --
//    --test-threads=1`, which is also how the release gate invokes it.
