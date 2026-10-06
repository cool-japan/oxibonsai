//! Product-level CPU/NEON/Metal greedy parity gate for the legacy models,
//! driving the real tokenizer and the real `SamplingParams` construction the
//! CLI uses. A CPU/Metal greedy divergence fixed on 2026-09-21 is the reason
//! this file exists as a permanent regression guard.
//!
//! ## File layout
//!
//! This file holds what every gate shares (the `OXIBONSAI_KERNEL_TIER`
//! guard) and the module documentation; the gates themselves live in
//! `legacy_parity_tests/`, one file per gate family, each under 1200 lines:
//!
//! - `cross_tier.rs` — the cross-tier greedy gates (numeric per-step
//!   comparison and golden text) for the three legacy models;
//! - `m08_long_context.rs` — the opt-in 20000-token YaRN gate;
//! - `cli_convert.rs` — the ONNX-export round trip and the `oxibonsai eval
//!   --task mmlu` subprocess gate;
//! - `speculative.rs` — the engine's n-gram speculation and the two-engine
//!   adaptive-lookahead decoder against plain greedy on the ternary Metal
//!   route.
//!
//! ## Why this file lives in `oxibonsai-runtime`, not next to the model-side one
//!
//! `crates/oxibonsai-model/tests/legacy_parity_tests.rs` already drives the
//! real legacy GGUFs cross-tier, but `oxibonsai-model` has no
//! `oxibonsai-tokenizer`/`oxibonsai-runtime` dependency (not even a
//! dev-dependency), so it has no way to turn real English prompt text into
//! token ids or generated ids back into text, and no way to construct a
//! `SamplingParams` (that type lives in `oxibonsai-runtime`). This crate
//! already depends on both `oxibonsai-model` and `oxibonsai-tokenizer` (via
//! [`oxibonsai_runtime::tokenizer_bridge::TokenizerBridge`], which — unlike
//! `oxibonsai_runtime::api_extensions` — is **not** `#[cfg(feature =
//! "server")]`; it and [`oxibonsai_runtime::sampling`] are always compiled
//! in), so the product-level half of the gate lives here.
//!
//! ## What this file adds over the model-side file
//! 1. Real English prompts — `legacy_golden.json`'s 9 `(model, prompt,
//!    metal text)` triples — tokenized/detokenized through the exact
//!    [`TokenizerBridge`] the CLI uses (`GOLDEN_CASES` in `cross_tier.rs`).
//! 2. A `SamplingParams` built the same way
//!    `src/cli/util.rs::build_sampling_params(0.0, 40, 0.9, 1.0)` does for
//!    `--temperature 0` (`greedy_sampling_params` in `cross_tier.rs`). That function is
//!    `pub(crate)` inside the `oxibonsai-cli` **binary** crate — there is no
//!    `lib.rs` there for another crate to depend on, so it cannot literally
//!    be `use`d from here — this reproduces its one-line body
//!    character-for-character (`cmd_eval.rs`'s own
//!    `build_sampling_params(0.0, 40, 0.9, 1.0)` call for its evaluation
//!    harness is the same convention for a temperature-0 greedy config; keep
//!    the two in sync by hand).
//! 3. Greedy generation through [`InferenceEngine::generate`] — the same
//!    method the CLI's decode paths both ultimately reduce to at
//!    temperature 0/no-penalty (`sample_with_history` routes to argmax
//!    internally; see `sampling.rs`) — driven by one engine per kernel tier
//!    this host can reach (`engine_for_tier`), and the decoded text
//!    checked against the golden Metal capture for exact equality.
//!
//! ## The GPU arm is the production Metal route
//!
//! A CPU tier is pinned with [`InferenceEngine::from_model_with_tier`]. The
//! GPU arm is built the way [`InferenceEngine::from_gguf`] builds a Metal
//! engine — the auto-detected dispatcher, the weight upload wherever
//! `from_gguf` performs it, the eager GPU cache — so its prefill runs on the
//! fused Metal batch path and its decode on the fused Metal decode path, and
//! both passes assert the prefill did not fall back
//! (`assert_fused_gpu_prefill`).
//!
//! ## Scope note: this does NOT drive `generate_greedy_gpu`'s 4-byte fast path
//!
//! `InferenceEngine::greedy_gpu_eligible` additionally requires
//! `self.uses_fused_gpu_decode()`, which `from_model_with_kernel` (via
//! `assemble(..., false)`) always sets to `false` — only
//! `from_gguf`/`from_gguf_path` resolve that flag from the GGUF's fused-route
//! metadata, and they also resolve the GGUF's own EOS set, which would make
//! the GPU arm's stopping rule differ from the CPU arms'. So the GPU engine
//! built here computes every forward pass through the fused Metal kernels
//! but samples via the general `Sampler::sample_with_history` path instead
//! of the specialized GPU-resident 4-byte-argmax-download optimization.
//! `crates/oxibonsai-runtime/tests/metal_greedy_cpu_fallback_tests.rs` is
//! the test that exercises
//! `generate_greedy_gpu` itself directly; this file's job is the tokenizer
//! and `SamplingParams` and golden-text half, not a
//! second copy of that coverage.
//!
//! ## Two independent passes per tier (deliberately not sharing one engine)
//! - **Text pass** (`run_text_pass`): a fresh engine per tier drives
//!   `generate()` end to end; exactly the first `GOLDEN_TOKEN_BUDGET`
//!   tokens (the fork's own capture budget) are detokenized and compared
//!   against the golden Metal text for EXACT equality, and an empty/short
//!   decode is rejected by `MIN_DECODED_CHARS` rather than trivially
//!   "passing". For the three fork-oracle cases the same decode must also be
//!   a prefix of the PrismML fork's own longer capture
//!   (`BONSAI_8B_FORK_CAPTURES`).
//! - **Logit pass** (`run_logit_pass`): a second, fresh engine per tier
//!   drives [`InferenceEngine::prefill_from_pos`]/[`InferenceEngine::decode_step`]
//!   manually with a first-index argmax (`oxibonsai_testkit::parity::argmax_first`),
//!   bypassing `Sampler` entirely — the same decode loop as
//!   `oxibonsai-model/tests/legacy_parity_tests.rs`'s `generate_with_logits`
//!   (the two crates share the comparison, `oxibonsai_testkit::parity`, not
//!   the engine construction) — so the same cross-tier comparison
//!   (`oxibonsai_testkit::parity::compare_against_reference`) applies to
//!   these real tokenized prompts too.
//!
//! ## What this gate asserts (design decision D-6, 2026-09-22)
//!
//! 1. **The greedy TOKEN CHAIN is the hard gate**, unconditionally, for every
//!    tier pair — it used to be asserted only when a near-tie classifier
//!    counted zero argmax flips.
//! 2. **`KernelTier::Reference` vs the auto-detected CPU tier must be
//!    BIT-EXACT** on every step's logits (measured exactly 0.0 over 64
//!    self-generated steps).
//! 3. Only the Reference-vs-GPU pair takes a numeric tolerance, and it is
//!    RELATIVE: `|Δ| <= max(5e-3, 5e-3 · max(1, max|reference|))` per step.
//!    See `oxibonsai_testkit::parity::per_step_bound` for why the combinator
//!    is `max` and not `&&`, and `REL_TOL` there for the six measured
//!    (model, slot) points it is calibrated against.
//! 4. These tests are **not** `#[ignore]`d: they run whenever the models are
//!    present (`OXIBONSAI_MODELS_DIR`, else `<workspace>/models`) and
//!    otherwise self-skip into the capability manifest with
//!    `executed: false`. `executed: true` is written only after the whole
//!    matrix has actually run. **Invoke them with `--test-threads=1`** — see
//!    `oxibonsai_testkit::parity::gpu_serial`.
//!
//! The golden-TEXT half (the text pass) and the NUMERIC half (the logit
//! pass) are separable in the output by design: every message is prefixed
//! `TEXT PASS` or `LOGIT PASS`, because a token-chain match against the METAL
//! goldens with only a numeric bound tripping is a very different release
//! posture from a text divergence.
//!
//! Using one engine for both would let `generate()`'s own `Sampler`-driven
//! KV-cache advance interfere with the manual `decode_step` loop's position
//! bookkeeping — two fresh engines per tier keeps the two claims cleanly
//! separated.
//!
//! `EosTokenSet` note: `from_model_with_tier` and `from_model_with_kernel`
//! (every arm's constructor) fall back to
//! `EosTokenSet::single(engine::EOS_TOKEN_ID)` rather than resolving the
//! GGUF's own `tokenizer.ggml.eos_token_id` (only `InferenceEngine::from_gguf`
//! does that — see above), so every tier stops on the same id. A
//! wrong fallback EOS id is only a real risk if it collides with a normal
//! content token inside the first `MAX_TOKENS` of one of these specific
//! prompts; the golden captures show it does not.
//!
//! `--seed`/`seed` is inert at `temperature == 0.0` on every backend (no RNG
//! is ever consulted before the argmax fast path returns), so the exact
//! value used below has no effect on the outcome.

#![cfg(all(feature = "metal", target_os = "macos"))]

use oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV;

// The gates live in sibling files under `legacy_parity_tests/`, one per gate
// family, so no file nears the 2000-line ceiling. `#[path]` keeps the
// directory next to this file (a crate-root file otherwise resolves `mod`
// declarations against `tests/`). Every capability `TEST_NAME` keeps its bare
// function name — the release gate requires the gates by the names their
// capability records carry; the libtest names are module-qualified
// (`cross_tier::...`, `m08_long_context::...`, `cli_convert::...`), which is
// what a `cargo test` name filter or a nextest `test(...)` expression sees.
//
// - `cross_tier`: the cross-tier greedy parity gates (numeric per-step
//   comparison and golden text) for the three legacy models, plus the
//   tiny-fixture logit check.
// - `m08_long_context`: the opt-in 20000-token YaRN gate.
// - `cli_convert`: the ONNX-export round trip and the `oxibonsai eval
//   --task mmlu` subprocess gate.
// - `speculative`: n-gram and two-engine adaptive speculative decoding
//   against plain greedy on the ternary Metal route.
#[path = "legacy_parity_tests/cli_convert.rs"]
mod cli_convert;
#[path = "legacy_parity_tests/cross_tier.rs"]
mod cross_tier;
#[path = "legacy_parity_tests/m08_long_context.rs"]
mod m08_long_context;
#[path = "legacy_parity_tests/speculative.rs"]
mod speculative;

// ── OXIBONSAI_KERNEL_TIER scrub ─────────────────────────────────────────────
//
// The opt-in INT8 tier (K-14) diverts the CPU tiers' native-format GEMV/GEMM
// but never Metal, so an exported `OXIBONSAI_KERNEL_TIER` would make this
// gate's CPU arms compute something its Metal arm does not, and the greedy
// chains could flip. Every test below that runs a model (or spawns a process
// that does) owns the variable for its whole run through a [`TierEnvGuard`];
// children inherit the cleared value.

/// Serializes every [`TierEnvGuard`]: `std::env::set_var` / `remove_var` are
/// `unsafe` because a concurrent read of any key can observe a torn `environ`.
static TIER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_tier_env() -> std::sync::MutexGuard<'static, ()> {
    TIER_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// RAII owner of `OXIBONSAI_KERNEL_TIER` for one test: holds the lock, clears
/// the variable, and restores its previous value on drop (also when unwinding
/// from a failed assertion).
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
        // SAFETY: `lock` serializes every reader and writer of the variable in
        // this binary and is held for the guard's whole life.
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

/// The guard clears a selector that is set, keeps every other guard out while
/// it lives, and puts the previous value back on drop.
#[test]
fn the_tier_env_guard_clears_the_selector_and_restores_it() {
    let lock = lock_tier_env();
    let ambient = std::env::var(KERNEL_TIER_ENV).ok();
    // SAFETY: `lock` is held.
    unsafe {
        std::env::set_var(KERNEL_TIER_ENV, "neon-dot");
    }
    let guard = TierEnvGuard::cleared_under(lock);
    assert!(
        std::env::var(KERNEL_TIER_ENV).is_err(),
        "cleared while held"
    );
    assert!(
        TIER_ENV_LOCK.try_lock().is_err(),
        "the guard holds the lock"
    );
    drop(guard);
    assert_eq!(
        std::env::var(KERNEL_TIER_ENV).as_deref(),
        Ok("neon-dot"),
        "the previous value is restored on drop"
    );
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

// ── Notes ────────────────────────────────────────────────────────────────
//
// 1. The engine constructors' `EosTokenSet`/`uses_fused_gpu_decode` scope
//    cut — see this file's header doc comment.
// 2. The comparison block (`compare_against_reference` and its supporting
//    types/constants) is shared with
//    `crates/oxibonsai-model/tests/legacy_parity_tests.rs` through
//    `oxibonsai_testkit::parity` rather than duplicated.
// 3. All nine (model, prompt) golden-TEXT comparisons pass.
//    `Bonsai-8B.gguf` (Q1_0, 1-bit) prompt 3 ("Once upon a time, in a small
//    village by the sea,") once diverged from a stale, pre-YaRN golden
//    capture at greedy step 14 (id 22208 " curious" vs the stale golden's id
//    2948 " love", top-2 gap 8.05e-2) — NOT a cross-tier parity break: the
//    logit pass for this exact case, run before the text pass, completed
//    bit-exact-CPU/relative-bound-GPU on all three tiers first, so Reference,
//    the auto-detected CPU tier and Metal all agreed with EACH OTHER and
//    only differed from the stale capture. Root cause: `Bonsai-8B.gguf`
//    declares YaRN rope scaling (factor 4, `original_context_length`
//    16384); the stale capture predated this engine honouring it. Once
//    honoured (matching llama.cpp exactly), the PrismML llama.cpp fork
//    itself emits the "curious nature" text byte-for-byte — `GOLDEN_CASES`
//    in `cross_tier.rs` carries the re-captured text, matching the fork
//    oracle, not the stale capture.
// 4. Cross-PROCESS serialization: `oxibonsai_testkit::parity::gpu_serial`
//    only serializes these gates inside one test binary, which is what
//    `cargo test`'s in-process thread pool needs — `cargo nextest` runs each
//    test in its own process, so this alone could not stop a `--workspace`
//    nextest run from scheduling several multi-GB real-model gates (this
//    file's and the model crate's) concurrently. `.config/nextest.toml` pins
//    this whole binary to its single-threaded `real-model-gate` test group,
//    so a nextest run executes these cases one at a time and never alongside
//    another real-model case (the M-08 long-context gate additionally gets
//    its own, longer slow-timeout there); `scripts/release-gate.sh`'s stage
//    1b also invokes the binary directly, serialized, in release, with
//    `--test-threads=1`, which is where the release gate collects its
//    evidence.
// 5. Every gate that runs the `oxibonsai` binary (`cli_convert.rs`) takes it
//    from `oxibonsai_testkit::cli_bin::resolve_cli_binary`, so the release
//    gate's one pre-built `--all-features` binary (`OXIBONSAI_CLI_BIN`) is
//    the one that runs; no gate compiles anything while it holds a
//    real-model critical section.
