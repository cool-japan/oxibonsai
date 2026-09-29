//! Product-level CPU/NEON/Metal greedy parity gate for the legacy models,
//! driving the real tokenizer and the real `SamplingParams` construction the
//! CLI uses. A pre-fix regression, fixed 2026-09-21, is the reason this
//! file exists as a permanent regression guard.
//!
//! ## Why this file lives in `oxibonsai-runtime`, not next to the model-side one
//!
//! `crates/oxibonsai-model/tests/legacy_parity_tests.rs` already drives the
//! real legacy GGUFs cross-tier, but by its own documented scope cut it
//! cannot reach the addendum's full ask: `oxibonsai-model` has no
//! `oxibonsai-tokenizer`/`oxibonsai-runtime` dependency (not even a
//! dev-dependency), so it has no way to turn real English prompt text into
//! token ids or generated ids back into text, and no way to construct a
//! `SamplingParams` (that type lives in `oxibonsai-runtime`). This crate
//! already depends on both `oxibonsai-model` and `oxibonsai-tokenizer` (via
//! [`oxibonsai_runtime::tokenizer_bridge::TokenizerBridge`], which — unlike
//! `oxibonsai_runtime::api_extensions` — is **not** `#[cfg(feature =
//! "server")]`; it and [`oxibonsai_runtime::sampling`] are always compiled
//! in), so it is the file the model-side deviation note names as the way to
//! close the gap.
//!
//! ## What this file adds over the model-side file
//! 1. Real English prompts — `legacy_golden.json`'s 9 `(model, prompt,
//!    metal text)` triples — tokenized/detokenized through the exact
//!    [`TokenizerBridge`] the CLI uses ([`GOLDEN_CASES`] below).
//! 2. A `SamplingParams` built the same way
//!    `src/cli/util.rs::build_sampling_params(0.0, 40, 0.9, 1.0)` does for
//!    `--temperature 0` ([`greedy_sampling_params`]). That function is
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
//!    this host can reach ([`engine_for_tier`]), and the decoded text
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
//! ([`assert_fused_gpu_prefill`]).
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
//! - **Text pass** ([`run_text_pass`]): a fresh engine per tier drives
//!   `generate()` end to end; exactly the first [`GOLDEN_TOKEN_BUDGET`]
//!   tokens (the fork's own capture budget) are detokenized and compared
//!   against the golden Metal text for EXACT equality, and an empty/short
//!   decode is rejected by [`MIN_DECODED_CHARS`] rather than trivially
//!   "passing". For the three fork-oracle cases the same decode must also be
//!   a prefix of the PrismML fork's own longer capture
//!   ([`BONSAI_8B_FORK_CAPTURES`]).
//! - **Logit pass** ([`run_logit_pass`]): a second, fresh engine per tier
//!   drives [`InferenceEngine::prefill_from_pos`]/[`InferenceEngine::decode_step`]
//!   manually with a first-index argmax ([`argmax_first`]), bypassing
//!   `Sampler` entirely — the same decode loop as
//!   `oxibonsai-model/tests/legacy_parity_tests.rs`'s `generate_with_logits`
//!   (the two crates share the comparison, `oxibonsai_testkit::parity`, not
//!   the engine construction) — so the same cross-tier comparison
//!   ([`compare_against_reference`]) applies to these real tokenized prompts
//!   too.
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
//!    See [`per_step_bound`] for why the combinator is `max` and not `&&`,
//!    and [`REL_TOL`] for the six measured (model, slot) points it is
//!    calibrated against.
//! 4. These tests are **not** `#[ignore]`d: they run whenever the models are
//!    present (`OXIBONSAI_MODELS_DIR`, else `<workspace>/models`) and
//!    otherwise self-skip into the capability manifest with
//!    `executed: false`. `executed: true` is written only after the whole
//!    matrix has actually run. **Invoke them with `--test-threads=1`** — see
//!    [`gpu_serial`].
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

use std::path::PathBuf;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_control::gguf_fused_metal_route;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::sampling_advanced::apply_repetition_penalty;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::gguf_fixture::tiny_dense_qwen3_gguf;

/// Generation budget for the LOGIT pass's cross-tier comparison: long
/// enough to reach a repetition-inducing tail on every golden prompt, so a
/// backend divergence has room to show up even when the first several
/// steps agree everywhere.
const MAX_TOKENS: usize = 64;

/// The PrismML llama.cpp fork's own `-n 32` capture budget: every string in
/// [`GOLDEN_CASES`] is the fork's greedy continuation for EXACTLY this many
/// tokens, so the TEXT pass below decodes exactly this many of its own
/// (longer, [`MAX_TOKENS`]-token) generation and checks it for EXACT
/// equality against the golden — never a shared/truncated-prefix
/// comparison, which cannot distinguish "the golden capture legitimately
/// ends here" from "the golden string itself is missing its last
/// character" (the defect one golden string actually had: a dropped
/// trailing newline that a prefix check let pass silently).
const GOLDEN_TOKEN_BUDGET: usize = 32;

/// Context budget: generous headroom over `MAX_TOKENS` plus the longest
/// prompt below, well inside all three legacy models' trained context.
const MAX_SEQ: usize = 512;

/// A deterministic seed. Inert at `temperature == 0.0` (see header doc).
const SEED: u64 = 42;

/// Below this many decoded characters, an "equals the golden text"
/// assertion would be vacuous (T-09 is this very package's own charter: a
/// real assertion, not one that trivially passes on an empty or
/// near-empty string).
const MIN_DECODED_CHARS: usize = 20;

// ── Cross-tier per-step comparison (shared with the model-crate sibling) ───
//
// The comparison logic (token-chain-first, then bit-exact-CPU /
// relative-bound-GPU per step) lives in `oxibonsai_testkit::parity`, shared
// with `crates/oxibonsai-model/tests/legacy_parity_tests.rs` rather than
// duplicated — see that module's doc comment for the full predicate and the
// measured (model, slot) points `REL_TOL` is calibrated against.
use oxibonsai_testkit::parity::{
    argmax_first, compare_against_reference, gpu_serial, per_step_bound, top_k, top_two, TierRun,
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

// ── Golden fixtures (embedded from the vendored golden capture, legacy_golden.json) ─

/// One `(model, prompt, golden Metal continuation text)` triple.
struct GoldenCase {
    /// GGUF filename under `models/`.
    model_file: &'static str,
    /// The exact prompt text the golden capture was taken from.
    prompt: &'static str,
    /// The captured Metal-backend greedy continuation (does NOT include the
    /// prompt itself), from `legacy_golden.json`'s `backend == "metal"`
    /// rows — the design's chosen greedy truth.
    metal_text: &'static str,
    /// Whether `metal_text` is independently confirmed byte-identical to
    /// the PrismML llama.cpp fork's own greedy output (temp 0) on the same
    /// file and prompt, not only to an earlier OxiBonsai capture of itself.
    /// True for the three `Bonsai-8B.gguf` cases (see the doc comment on
    /// the last of them): a change that matches this text is checked
    /// against llama.cpp's own semantics, not only against this project's
    /// history.
    fork_oracle: bool,
}

/// The 9 `backend == "metal"` rows of the vendored golden capture,
/// `legacy_golden.json` (3 legacy models x 3 prompts each), embedded
/// verbatim.
const GOLDEN_CASES: [GoldenCase; 9] = [
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: "\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        metal_text: " there lived a young girl named Lila. She was known for her kindness and her love for the sea. One day, she discovered a mysterious shell that gl",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: " if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2) print(fibonacci(10)) \n\nThe",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        metal_text: " there lived a young girl named Lila. She was known for her curiosity and love for the ocean. One day, while exploring the beach, she discovered a",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital",
        fork_oracle: true,
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: "\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef fibonacci(n):\n",
        fork_oracle: true,
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        // RE-CAPTURED 2026-09-23: Bonsai-8B.gguf declares `rope.scaling.type = yarn`
        // (factor 4, original_context_length 16384). OxiBonsai <= 0.2.4 ignored it; the
        // engine now honours it (M-08), and this text is byte-identical to the PrismML
        // llama.cpp fork (greedy, temp 0) on the same file.
        metal_text: " there lived a young girl named Lila. She was known for her curious nature and her love of the sea. Lila often spent her days exploring the cliffs",
        fork_oracle: true,
    },
];

/// The PrismML llama.cpp fork's own greedy continuation of each
/// `Bonsai-8B.gguf` golden prompt, captured independently of
/// [`GOLDEN_CASES`] and LONGER than its 32-token budget: `llama-completion
/// -n 48 --temp 0 --top-k 1 -s 42` on Metal (fork commit 9a9394a), with the
/// blank line the tool itself prints after every generation removed. The
/// text pass requires every tier's 32-token decode of these prompts to be a
/// prefix of this capture, so the gate stays pinned to llama.cpp's own
/// output even if [`GOLDEN_CASES`] is ever re-captured.
const BONSAI_8B_FORK_CAPTURES: [(&str, &str); 3] = [
    (
        "The capital of Japan is",
        " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Brazil is Brasília. The capital of Canada is Ottawa. The capital of",
    ),
    (
        "def fibonacci(n):",
        "\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return",
    ),
    (
        "Once upon a time, in a small village by the sea,",
        " there lived a young girl named Lila. She was known for her curious nature and her love of the sea. Lila often spent her days exploring the cliffs, listening to the waves, and collecting seashells. One day, while",
    ),
];

/// The fork's own capture for a fork-oracle case's prompt, if any.
fn fork_capture_for(prompt: &str) -> Option<&'static str> {
    BONSAI_8B_FORK_CAPTURES
        .iter()
        .find(|(p, _)| *p == prompt)
        .map(|(_, capture)| *capture)
}

/// The distinct model filenames in [`GOLDEN_CASES`], in first-appearance
/// order (drives the three top-level `#[test]` functions below).
const LEGACY_MODEL_FILES: [&str; 3] = [
    "Ternary-Bonsai-1.7B.gguf",
    "Ternary-Bonsai-8B.gguf",
    "Bonsai-8B.gguf",
];

/// Fixture-authoring guard: exactly the three `Bonsai-8B.gguf` cases carry
/// `fork_oracle = true`, matching the doc comment on that field, and each
/// one's golden text is a byte prefix of the fork's own longer capture
/// ([`BONSAI_8B_FORK_CAPTURES`]) — so a golden string edited or re-captured
/// away from llama.cpp's output fails here, with no model file needed.
#[test]
fn fork_oracle_rows_are_exactly_the_three_bonsai_8b_cases() {
    for case in &GOLDEN_CASES {
        assert_eq!(
            case.fork_oracle,
            case.model_file == "Bonsai-8B.gguf",
            "{} {:?}: fork_oracle must be true iff the model is Bonsai-8B.gguf",
            case.model_file,
            case.prompt,
        );
        if case.fork_oracle {
            let capture = fork_capture_for(case.prompt)
                .unwrap_or_else(|| panic!("{:?}: no fork capture for this prompt", case.prompt));
            assert!(
                capture.starts_with(case.metal_text) && capture.len() > case.metal_text.len(),
                "{:?}: the golden text is not a strict prefix of the fork's own capture\n  \
                 golden: {:?}\n  fork:   {capture:?}",
                case.prompt,
                case.metal_text
            );
        }
    }
    assert_eq!(
        GOLDEN_CASES.iter().filter(|c| c.fork_oracle).count(),
        3,
        "expected exactly 3 fork-oracle rows"
    );
    assert_eq!(BONSAI_8B_FORK_CAPTURES.len(), 3, "one fork capture per row");
}

// ── T-05 capability-report producer ─────────────────────────────────────────
//
// T-07: `oxibonsai-runtime` takes `oxibonsai-testkit` as a dev-dependency
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

/// Mirrors `src/cli/util.rs::build_sampling_params(0.0, 40, 0.9, 1.0)` (see
/// this file's header doc comment for why it cannot literally be `use`d).
/// `repetition_penalty: 1.0`, not `SamplingParams::default()`'s `1.1`, is
/// exactly the fix for the CPU/Metal divergence this file guards against.
fn greedy_sampling_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    }
}

/// Asserts `decoded` is a plausible, non-vacuous match for `golden`:
/// long enough to rule out a truncated-but-not-empty generation (at least
/// [`MIN_DECODED_CHARS`], AND at least `golden`'s own length capped at 40
/// chars — a spurious early EOS a few tokens in must still fail this, not
/// just a fully empty decode), and EXACTLY equal to `golden` byte for
/// byte — never a shared-prefix-of-whichever-is-shorter comparison. `decoded`
/// here is already sliced by the caller ([`run_text_pass`]) to exactly
/// [`GOLDEN_TOKEN_BUDGET`] tokens, the same budget the fork's own capture
/// used, so "equal" and "the fork's own greedy continuation for this many
/// steps" are the same claim. A shared/truncated-prefix comparison cannot
/// tell "the golden capture legitimately ends here" apart from "the golden
/// string itself lost a trailing character" — the second is a real defect
/// one embedded golden string had (a dropped trailing newline) that such a
/// comparison let pass silently; exact equality at a fixed, known budget
/// closes that hole.
///
/// T-09: a bug that made `decoded` empty (an
/// immediate EOS, a broken tokenizer round-trip, ...) — or merely short,
/// from an EOS a handful of tokens in — must fail this assertion, not
/// vacuously satisfy "is a prefix of everything".
///
/// `logit_pass_run` is this same case/tier's already-computed [`TierRun`]
/// from [`run_logit_pass`] (now run BEFORE the text pass in
/// [`run_model_gate`], see that function's doc comment), threaded through
/// purely for diagnostics on a mismatch: the logit pass is an independent
/// decode (fresh engine, manual argmax loop — see the module doc's "Two
/// independent passes per tier"), so its token ids are used only to locate
/// the first differing TOKEN and its logits are printed as diagnostic
/// context, never asserted equal to this pass's own tokens.
fn assert_exact_match_golden(
    decoded: &str,
    golden: &str,
    context: &str,
    tokenizer: &TokenizerBridge,
    generated_tokens: &[u32],
    logit_pass_run: Option<&TierRun<KernelTier>>,
) {
    let min_chars = MIN_DECODED_CHARS.max(golden.chars().count().min(40));
    assert!(
        decoded.chars().count() >= min_chars,
        "{context}: decoded continuation is implausibly short ({} chars, wanted >= {min_chars}): {decoded:?}",
        decoded.chars().count()
    );
    if decoded == golden {
        eprintln!("{context}: text OK (exact, {GOLDEN_TOKEN_BUDGET}-token budget): {decoded:?}");
        return;
    }

    // Tokenize the golden continuation and diff it against this tier's own
    // generated ids to find the first differing TOKEN (not just byte), then
    // — when this case/tier's logit pass already ran — print that step's
    // top-2 gap and top-5 so a near-tie is distinguishable from a real
    // disagreement. `top_two`/`top_k` already exist for this; nothing new
    // is invented here.
    let golden_tokens = tokenizer.encode(golden).unwrap_or_default();
    let token_detail = match generated_tokens
        .iter()
        .zip(golden_tokens.iter())
        .position(|(a, b)| a != b)
    {
        Some(i) => {
            let this_id = generated_tokens[i];
            let golden_id = golden_tokens[i];
            let this_str = tokenizer
                .decode(&[this_id])
                .unwrap_or_else(|e| format!("<decode failed: {e}>"));
            let golden_str = tokenizer
                .decode(&[golden_id])
                .unwrap_or_else(|e| format!("<decode failed: {e}>"));
            let logit_note = match logit_pass_run.and_then(|r| r.logits.get(i)) {
                Some(logits) => {
                    let top2 = top_two(logits);
                    format!(
                        " | this tier's LOGIT-PASS step {i} (an independent decode — diagnostic \
                         only, not asserted equal to this pass): top-2 gap={:e} top-5={}",
                        top2.gap(),
                        top_k(logits, 5),
                    )
                }
                None => String::new(),
            };
            format!(
                "\n  first differing TOKEN index {i}: this tier picked id {this_id} \
                 ({this_str:?}), golden picked id {golden_id} ({golden_str:?}){logit_note}"
            )
        }
        None => "\n  every zipped token position matches; the two token sequences differ only \
                 in LENGTH (one is a prefix of the other)"
            .to_string(),
    };

    panic!(
        "{context}: decoded text does not EXACTLY match the golden Metal capture over the \
         {GOLDEN_TOKEN_BUDGET}-token budget both were captured at\n  decoded: {decoded:?}\n  \
         golden:  {golden:?}{token_detail}"
    );
}

/// Builds the engine for one tier of the matrix. A CPU tier is pinned with
/// [`InferenceEngine::from_model_with_tier`]. The GPU tier is built the way
/// [`InferenceEngine::from_gguf`] builds a production Metal engine: the
/// auto-detected dispatcher (the only constructor wired to a live GPU
/// backend), the weight upload `from_gguf` performs — skipped exactly where
/// it skips it, on the ternary fused route, whose Metal paths bind their own
/// cached weight set — and the eager GPU cache build. The GPU arm therefore
/// runs the fused Metal batch prefill and decode production runs; the
/// callers assert it ([`assert_fused_gpu_prefill`]).
fn engine_for_tier<'a>(gguf: &'a GgufFile<'a>, tier: KernelTier) -> InferenceEngine<'a> {
    let mut model = BonsaiModel::from_gguf(gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    if tier != KernelTier::Gpu {
        return InferenceEngine::from_model_with_tier(model, tier, greedy_sampling_params(), SEED);
    }
    let kernel = KernelDispatcher::auto_detect();
    assert_eq!(
        kernel.tier(),
        KernelTier::Gpu,
        "KernelDispatcher::auto_detect() did not select the GPU tier for the GPU arm"
    );
    if !gguf_fused_metal_route(gguf).gpu_weight_upload_redundant() {
        model.upload_weights_to_gpu(&kernel);
    }
    model
        .get_or_create_gpu_cache()
        .unwrap_or_else(|e| panic!("GPU weight cache build for the GPU arm: {e}"));
    InferenceEngine::from_model_with_kernel(model, kernel, greedy_sampling_params(), SEED)
}

/// Captures the `error` field of the WARN event [`PrefillFallbackDetector`]
/// matches.
struct ErrorFieldVisitor<'a>(&'a mut Option<String>);

impl tracing::field::Visit for ErrorFieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "error" {
            *self.0 = Some(format!("{value:?}"));
        }
    }
}

/// A minimal `tracing` subscriber recording the WARN that
/// `BonsaiModel::forward_prefill` emits (from its `prefill_dispatch` module)
/// when a GPU batch prefill fails and it silently falls back to a CPU path,
/// with that event's `error` field.
struct PrefillFallbackDetector {
    reason: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl tracing::Subscriber for PrefillFallbackDetector {
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
        if *metadata.level() <= tracing::Level::WARN
            && metadata
                .module_path()
                .is_some_and(|m| m.contains("prefill_dispatch"))
        {
            let mut reason = None;
            event.record(&mut ErrorFieldVisitor(&mut reason));
            *self.reason.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(reason.unwrap_or_else(|| "<no error field>".to_string()));
        }
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Runs `f` with [`PrefillFallbackDetector`] installed on this thread,
/// returning `f`'s result and the fallback reason, if a fallback happened.
fn run_detecting_prefill_fallback<T>(f: impl FnOnce() -> T) -> (T, Option<String>) {
    let reason = std::sync::Arc::new(std::sync::Mutex::new(None));
    let detector = PrefillFallbackDetector {
        reason: reason.clone(),
    };
    let result = tracing::subscriber::with_default(detector, f);
    let fallback = reason.lock().unwrap_or_else(|e| e.into_inner()).clone();
    (result, fallback)
}

/// The GPU arm's prefill ran on the fused Metal batch path: no fallback
/// warning, and the model's device-KV latch (`gpu_path_active`, which the
/// CPU prefill fallback clears) is set.
fn assert_fused_gpu_prefill(engine: &InferenceEngine<'_>, fallback: Option<&str>, context: &str) {
    let device_kv = engine
        .dense_model()
        .is_some_and(BonsaiModel::gpu_path_active);
    assert!(
        fallback.is_none() && device_kv,
        "{context}: the GPU arm's prefill did not run on the fused Metal batch path \
         (fallback reason {fallback:?}; device-KV latch set: {device_kv})"
    );
}

/// Runs the "text pass" for one `(tier, case)` pair: a fresh engine drives
/// `generate()` for [`MAX_TOKENS`] steps, and exactly the first
/// [`GOLDEN_TOKEN_BUDGET`] of those tokens (the fork's own capture budget)
/// are detokenized and checked against the golden text for EXACT equality.
/// `logit_pass_run` is threaded through to [`assert_exact_match_golden`]
/// purely for diagnostics — see its doc comment.
fn run_text_pass(
    gguf_bytes: &[u8],
    tier: KernelTier,
    tokenizer: &TokenizerBridge,
    case: &GoldenCase,
    logit_pass_run: Option<&TierRun<KernelTier>>,
) {
    let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
    let mut engine = engine_for_tier(&gguf, tier);

    let prompt_ids = tokenizer.encode(case.prompt).expect("encode golden prompt");
    let (generated, fallback) =
        run_detecting_prefill_fallback(|| engine.generate(&prompt_ids, MAX_TOKENS));
    let generated = generated.expect("engine.generate");
    if tier == KernelTier::Gpu {
        assert_fused_gpu_prefill(
            &engine,
            fallback.as_deref(),
            &format!("TEXT PASS {} prompt {:?}", case.model_file, case.prompt),
        );
    }
    assert!(
        generated.len() >= GOLDEN_TOKEN_BUDGET,
        "{} prompt {:?} tier {tier}: generate() returned only {} tokens, fewer than the \
         {GOLDEN_TOKEN_BUDGET}-token golden budget -- an early EOS or a generation-loop defect, \
         not a text mismatch to diagnose below",
        case.model_file,
        case.prompt,
        generated.len(),
    );
    let golden_budget_tokens = &generated[..GOLDEN_TOKEN_BUDGET];
    let decoded = tokenizer
        .decode(golden_budget_tokens)
        .expect("decode generated ids");

    assert_exact_match_golden(
        &decoded,
        case.metal_text,
        // The GOLDEN-TEXT half is reported separately from the numeric
        // gate — a token-chain match against the METAL goldens with only a
        // numeric bound tripping is a very different release posture from
        // a text divergence, so every message says which half it came from.
        &format!(
            "TEXT PASS {} prompt {:?} tier {tier} ({tier:?})",
            case.model_file, case.prompt
        ),
        tokenizer,
        &generated,
        logit_pass_run,
    );

    // ORACLE: the fork-oracle cases' decode must also continue llama.cpp's
    // own (longer) capture, independently of `GOLDEN_CASES`.
    if case.fork_oracle {
        let capture = fork_capture_for(case.prompt).unwrap_or_else(|| {
            panic!("{:?}: fork-oracle case without a fork capture", case.prompt)
        });
        assert!(
            capture.starts_with(decoded.as_str()),
            "TEXT PASS {} prompt {:?} tier {tier}: the {GOLDEN_TOKEN_BUDGET}-token decode is not \
             a prefix of the PrismML fork's own greedy capture\n  decoded: {decoded:?}\n  \
             fork:    {capture:?}",
            case.model_file,
            case.prompt
        );
        eprintln!(
            "TEXT PASS {} prompt {:?} tier {tier}: fork oracle OK (prefix of the fork's own \
             capture)",
            case.model_file, case.prompt
        );
    }
}

/// Runs the "logit pass" for one case across every tier in `tiers`: a fresh
/// engine per tier drives a manual, pure-argmax decode loop (bypassing
/// `Sampler`) and returns each tier's `(token ids, per-step logits)`.
fn run_logit_pass(
    gguf_bytes: &[u8],
    tiers: &[KernelTier],
    prompt_ids: &[u32],
) -> Vec<TierRun<KernelTier>> {
    let mut per_tier = Vec::with_capacity(tiers.len());
    for &tier in tiers {
        let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
        let mut engine = engine_for_tier(&gguf, tier);

        let (prefill, fallback) =
            run_detecting_prefill_fallback(|| engine.prefill_from_pos(prompt_ids, 0));
        let mut logits = prefill.expect("prefill_from_pos");
        if tier == KernelTier::Gpu {
            assert_fused_gpu_prefill(&engine, fallback.as_deref(), "LOGIT PASS");
        }
        let mut generated = Vec::with_capacity(MAX_TOKENS);
        let mut all_logits = Vec::with_capacity(MAX_TOKENS);
        for step in 0..MAX_TOKENS {
            let tok = argmax_first(&logits);
            generated.push(tok);
            all_logits.push(logits.clone());
            let pos = prompt_ids.len() + step;
            logits = engine.decode_step(tok, pos).expect("decode_step");
        }
        per_tier.push(TierRun {
            tier,
            tokens: generated,
            logits: all_logits,
        });
    }
    per_tier
}

/// Runs the full parity gate for one legacy model: for each of its golden
/// cases, the LOGIT pass (D-6's cross-tier comparison against the Reference
/// tier — token chain first, then bit-exact for a CPU tier / the relative
/// bound for the GPU tier) and then the TEXT pass (per tier, vs. the golden
/// Metal capture).
///
/// LOGIT BEFORE TEXT, deliberately: `tiers` always starts with
/// `KernelTier::Reference`, so a text-then-logit order would let a text-pass
/// mismatch on the very first tier panic out of this function before the
/// logit pass — the cross-tier numeric verdict for that case — ever ran.
/// Running the logit pass first means a cross-tier numeric claim is always
/// either earned (the logit pass actually completes and its verdict is
/// real) or superseded by a logit-pass panic that reports the true failure
/// — never asserted without having been checked.
fn run_model_gate(model_file: &'static str, test_name: &str) {
    let gate_start = std::time::Instant::now();
    // Serialized against this binary's other real-model gates — see
    // [`gpu_serial`]. Taken before the fixture probes so the whole gate,
    // including its capability bookkeeping, is one critical section.
    let _gpu = gpu_serial();

    // D-6(4): these tests are NOT `#[ignore]`d — they
    // run whenever the models are present and self-skip (reported, never
    // green) when they are not. Both probes below are missing-FIXTURE skips,
    // not missing-Metal-hardware skips, hence `Capability::LegacyModels`:
    // recording them as `Metal` let an unrelated Metal-hardware record
    // elsewhere in the manifest (e.g. `metal_k_quant_gemv_parity.rs`) satisfy
    // the release gate's `metal` check regardless of whether this gate ever
    // ran. `find_model` is one `fs::metadata` call, so a models-less host
    // pays nothing for the `#[ignore]` removal.
    let Some(model_path) = find_model(model_file) else {
        eprintln!("skip: {model_file} not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };

    let tokenizer_path_str = tokenizer_path
        .to_str()
        .expect("models/ path is valid UTF-8");
    let tokenizer =
        TokenizerBridge::from_file(tokenizer_path_str).expect("load real tokenizer.json");
    let gguf_bytes = std::fs::read(&model_path).expect("read real legacy GGUF");

    let tiers: Vec<KernelTier> = {
        let mut v = vec![KernelTier::Reference, cpu_kernel_tier(), KernelTier::Gpu];
        v.dedup();
        v
    };

    for case in GOLDEN_CASES.iter().filter(|c| c.model_file == model_file) {
        // ---- Logit pass FIRST: D-6's cross-tier comparison (token chain
        // first, then bit-exact for CPU / relative bound for GPU) over the
        // real tokenized prompt. See this function's doc comment for why
        // this now runs before the text pass. ----
        let prompt_ids = tokenizer.encode(case.prompt).expect("encode golden prompt");
        let per_tier = run_logit_pass(&gguf_bytes, &tiers, &prompt_ids);
        {
            let (reference, others) = per_tier
                .split_first()
                .expect("the tier matrix always starts with KernelTier::Reference");
            let context = format!("LOGIT PASS {model_file} prompt {:?}", case.prompt);
            for other in others {
                compare_against_reference(&context, reference, other, is_cpu_tier);
            }
        }

        // ---- Text pass: real tokenizer + real SamplingParams, per tier.
        // `per_tier` (just computed above) is threaded through so a text
        // mismatch's panic can report the matching tier's own logit-pass
        // step data (top-2 gap / top-5) instead of only the decoded
        // strings — see `assert_exact_match_golden`. ----
        for &tier in &tiers {
            let logit_run = per_tier.iter().find(|t| t.tier == tier);
            run_text_pass(&gguf_bytes, tier, &tokenizer, case, logit_run);
        }
    }

    // T-05 honesty: the `executed: true` records are written HERE, once the
    // whole matrix — every golden case, every tier, both passes — has
    // actually completed. Writing them on fixture PRESENCE instead would
    // claim coverage for work that may still panic, which is the exact
    // false-green class the capability manifest exists to eliminate. Any
    // assertion above panics out of this function and neither record is
    // written.
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, test_name, elapsed);
    record_executed_timed(Capability::LegacyModels, test_name, elapsed);
}

#[test]
fn ternary_1_7b_greedy_text_matches_golden_across_tiers() {
    run_model_gate(
        LEGACY_MODEL_FILES[0],
        "oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_greedy_text_matches_golden_across_tiers",
    );
}

#[test]
fn ternary_8b_greedy_text_matches_golden_across_tiers() {
    run_model_gate(
        LEGACY_MODEL_FILES[1],
        "oxibonsai-runtime::legacy_parity_tests::ternary_8b_greedy_text_matches_golden_across_tiers",
    );
}

#[test]
fn bonsai_8b_greedy_text_matches_golden_across_tiers() {
    run_model_gate(
        LEGACY_MODEL_FILES[2],
        "oxibonsai-runtime::legacy_parity_tests::bonsai_8b_greedy_text_matches_golden_across_tiers",
    );
}

// ═════════════════════════════════════════════════════════════════════════
//  A non-degenerate tiny fixture for logit-shape settings
// ═════════════════════════════════════════════════════════════════════════

/// `Qwen3Config::tiny_test()` combined with an all-zero-weight model
/// produces an all-zero logit vector, under which `apply_repetition_penalty`
/// (`logit >= 0 ? logit / penalty : logit * penalty`) is the identity at
/// every logit regardless of `penalty` — a setting test built on that
/// fixture alone cannot observe the setting do anything. This test instead
/// loads `oxibonsai_testkit::gguf_fixture::tiny_dense_qwen3_gguf` — the
/// same size budget as `tiny_test()`, but with real, non-zero, deterministic
/// weights, through the real `BonsaiModel::from_gguf` — and checks both that
/// its logits are genuinely non-degenerate and that the real repetition
/// penalty actually moves them, needing no model file and no real weights.
#[test]
fn tiny_dense_qwen3_fixture_produces_non_degenerate_logits_the_repetition_penalty_can_move() {
    let bytes = tiny_dense_qwen3_gguf(0xF00D_BEEF).expect("build tiny dense qwen3 fixture");
    let gguf = GgufFile::parse(&bytes).expect("parse tiny dense qwen3 fixture");
    let mut model = BonsaiModel::from_gguf(&gguf, 64).expect("load tiny dense qwen3 fixture");
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);

    let logits = model.forward(0, 0, &kernel).expect("forward");
    assert!(
        logits.iter().any(|&v| v != 0.0),
        "the tiny dense qwen3 fixture's logits must not be all-zero"
    );
    let first = logits[0];
    assert!(
        logits.iter().any(|&v| v != first),
        "the fixture's logits must not be a degenerate constant vector either"
    );

    // Covering every vocabulary index guarantees the penalty is applied to
    // at least one non-zero logit, which a penalty != 1.0 always moves.
    let history: Vec<u32> = (0..logits.len() as u32).collect();
    let mut penalized = logits.clone();
    apply_repetition_penalty(&mut penalized, &history, 1.3);
    assert_ne!(
        logits, penalized,
        "the repetition penalty must change at least one logit on a non-degenerate vector"
    );
}

// ═════════════════════════════════════════════════════════════════════════
//  A real ONNX -> GGUF conversion, then the real tokenizer round-trip
// ═════════════════════════════════════════════════════════════════════════

/// The real `oxibonsai_model::convert_onnx_to_gguf` on the real
/// `Ternary-Bonsai-1.7B-ONNX` export, then the real
/// `oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata` on the converted
/// file's own embedded tokenizer metadata: the whole ONNX-import ->
/// GGUF-write -> tokenizer-load chain, exercised end to end rather than only
/// asserting the converted metadata is present and well-typed.
#[test]
fn onnx_converted_gguf_loads_through_the_real_tokenizer_round_trip() {
    const TEST: &str = "oxibonsai-runtime::legacy_parity_tests::onnx_converted_gguf_loads_through_the_real_tokenizer_round_trip";
    let require_real_files = std::env::var("OXI_REQUIRE_MODEL_FILES")
        .map(|v| v == "1")
        .unwrap_or(false);
    let onnx_dir = models_dir().join("Ternary-Bonsai-1.7B-ONNX");
    let onnx_path = onnx_dir.join("onnx").join("model_q2.onnx");
    if !onnx_path.is_file() {
        assert!(
            !require_real_files,
            "OXI_REQUIRE_MODEL_FILES=1: {TEST} needs {onnx_path:?}"
        );
        eprintln!("skip {TEST}: {onnx_path:?} not found");
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }
    let gate_start = std::time::Instant::now();

    let out_path = oxibonsai_testkit::temp_path::unique_path("b2-18-onnx-convert", ".gguf");
    let stats = oxibonsai_model::convert_onnx_to_gguf(&onnx_path, &out_path, "tq2_0_g128")
        .unwrap_or_else(|e| panic!("convert_onnx_to_gguf({onnx_path:?}): {e}"));
    eprintln!("{TEST}: converted {stats:?}");

    let bytes = std::fs::read(&out_path).expect("read the converted GGUF back");
    let _ = std::fs::remove_file(&out_path);
    let gguf = GgufFile::parse(&bytes).expect("the converted GGUF parses");

    // The real tokenizer constructor, on the converted file's own embedded
    // `tokenizer.ggml.*` metadata -- not a key-presence/well-typedness
    // stand-in.
    let tokenizer = oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata(&gguf.metadata)
        .expect("OxiTokenizer::from_gguf_metadata on the converted file");

    let text = "The capital of Japan is Tokyo.";
    let ids = tokenizer
        .encode(text)
        .expect("encode through the real tokenizer");
    assert!(!ids.is_empty(), "encoding real text must produce real ids");
    let decoded = tokenizer
        .decode(&ids)
        .expect("decode through the real tokenizer");
    assert_eq!(
        decoded, text,
        "encode -> decode through the converted file's own tokenizer must round-trip exactly"
    );

    record_executed_timed(Capability::LegacyModels, TEST, gate_start.elapsed());
}

// ─── CLI-CORE: `oxibonsai eval --task mmlu` against a real model ───────────
//
// CLI-CORE: the logit-based eval tasks (mmlu, arc-easy/challenge,
// hellaswag, winogrande, boolq, truthfulqa-mc1/mc2, calibration) and
// teacher-forced perplexity compile and pass clippy, but none of
// `oxibonsai-eval`'s own tests exercises `score_choices_logprob` at
// runtime against a real model (that crate has no GGUF of its own).
// `score_choices_logprob` lives in `src/cli/cmd_eval.rs`, a bin-only crate
// with no `[lib]` target — same shape as `bonsai2_runtime_tests.rs`'s G9
// gate for `oxibonsai info` — so this drives the real, shipped `oxibonsai
// eval --task mmlu` subcommand as a subprocess against a real legacy GGUF
// and a tiny in-memory MMLU-shaped fixture, and asserts the JSON report it
// writes is well-formed and reports a real, in-range score — proof the
// logit-based path actually executes rather than only compiling.
fn eval_target_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        if !dir.is_empty() {
            return std::path::PathBuf::from(dir);
        }
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target")
}

/// A 2-question, MMLU-shaped fixture (`{"id","question","choices",
/// "correct_answer","subject"}` JSONL, per `EvalTask::Mmlu`'s own doc
/// comment; `correct_answer` is the 0-based index into `choices`, per
/// `oxibonsai_eval::dataset::MultipleChoiceQuestion`), written under
/// `std::env::temp_dir()` so this test needs no checked-in fixture asset
/// and no absolute path baked into the repository.
fn write_mmlu_fixture() -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "oxibonsai-cli-core-mmlu-fixture-{}-{unique}.jsonl",
        std::process::id()
    ));
    let jsonl = concat!(
        r#"{"id":"q1","question":"What is the capital of France?","choices":["#,
        r#""London","Paris","Berlin","Madrid"],"correct_answer":1,"subject":"geography"}"#,
        "\n",
        r#"{"id":"q2","question":"Which planet is known as the Red Planet?","choices":["#,
        r#""Venus","Mars","Jupiter","Saturn"],"correct_answer":1,"subject":"astronomy"}"#,
        "\n",
    );
    std::fs::write(&path, jsonl).expect("write mmlu fixture jsonl to std::env::temp_dir()");
    path
}

#[test]
fn eval_cli_scores_a_real_mmlu_style_dataset_through_score_choices_logprob() {
    const TEST: &str = "oxibonsai-runtime::legacy_parity_tests::\
                        eval_cli_scores_a_real_mmlu_style_dataset_through_score_choices_logprob";
    let gate_start = std::time::Instant::now();

    let Some(model_path) = find_model("Ternary-Bonsai-1.7B.gguf") else {
        eprintln!(
            "skip: Ternary-Bonsai-1.7B.gguf not found under {:?}",
            models_dir()
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let build_status = std::process::Command::new(&cargo)
        .args([
            "build",
            "--release",
            "-p",
            "oxibonsai-cli",
            "--bin",
            "oxibonsai",
            "--features",
            "eval",
        ])
        .status()
        .expect(
            "spawning `cargo build -p oxibonsai-cli --bin oxibonsai --features eval` should not \
             itself fail to launch",
        );
    assert!(
        build_status.success(),
        "cargo build -p oxibonsai-cli --bin oxibonsai --features eval failed (exit {:?})",
        build_status.code()
    );

    let dataset_path = write_mmlu_fixture();
    let report_path = std::env::temp_dir().join(format!(
        "oxibonsai-cli-core-mmlu-report-{}.json",
        std::process::id()
    ));
    // Clean up whatever a previous, possibly-panicked run of this same test
    // process id/host left behind; the write below always replaces it, so
    // this is a courtesy, not a correctness requirement.
    let _ = std::fs::remove_file(&report_path);

    let bin = eval_target_dir().join("release").join("oxibonsai");
    let output = std::process::Command::new(&bin)
        .arg("eval")
        .arg("--task")
        .arg("mmlu")
        .arg("--model")
        .arg(&model_path)
        .arg("--tokenizer")
        .arg(tokenizer_path)
        .arg("--dataset")
        .arg(&dataset_path)
        .arg("--report-json")
        .arg(&report_path)
        .output()
        .unwrap_or_else(|e| panic!("spawning {} failed: {e}", bin.display()));

    let cleanup = || {
        let _ = std::fs::remove_file(&dataset_path);
        let _ = std::fs::remove_file(&report_path);
    };

    if !output.status.success() {
        cleanup();
        panic!(
            "{} eval --task mmlu --model {} exited with {:?}\nstdout:\n{}\nstderr:\n{}",
            bin.display(),
            model_path.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let report_bytes = std::fs::read(&report_path).unwrap_or_else(|e| {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        cleanup();
        panic!(
            "reading eval report at {}: {e}\nstderr:\n{stderr}",
            report_path.display()
        )
    });
    let report: serde_json::Value = serde_json::from_slice(&report_bytes).unwrap_or_else(|e| {
        cleanup();
        panic!(
            "eval --report-json produced non-JSON output: {e}\n{}",
            String::from_utf8_lossy(&report_bytes)
        )
    });
    cleanup();

    let results = report["results"]
        .as_array()
        .unwrap_or_else(|| panic!("eval report has no \"results\" array: {report}"));
    assert!(
        !results.is_empty(),
        "eval --task mmlu produced an empty \"results\" array: {report}"
    );
    // `cmd_eval.rs`'s MMLU arm reports `metric: "mmlu_accuracy"`,
    // `value: result.accuracy_pct` (a PERCENT, 0..=100, `unit: "%"`) and
    // `notes: Some(format!("n={}", result.total))` — NOT the generic
    // `"accuracy"` fraction `EvalReport::add_accuracy` writes for other
    // tasks. Match the real field this task actually writes.
    let accuracy_entry = results
        .iter()
        .find(|r| r["metric"].as_str() == Some("mmlu_accuracy"))
        .unwrap_or_else(|| panic!("no \"mmlu_accuracy\" metric entry in eval report: {report}"));
    let accuracy_pct = accuracy_entry["value"].as_f64().unwrap_or_else(|| {
        panic!("mmlu_accuracy entry has no numeric \"value\": {accuracy_entry}")
    });
    assert!(
        accuracy_pct.is_finite() && (0.0..=100.0).contains(&accuracy_pct),
        "score_choices_logprob produced an out-of-range mmlu_accuracy {accuracy_pct} \
         (report: {report})"
    );
    let notes = accuracy_entry["notes"].as_str().unwrap_or("");
    assert!(
        notes.contains("n=2"),
        "expected the 2-question fixture to score as n=2, got notes={notes:?} (report: {report})"
    );
    eprintln!(
        "CLI-CORE measured: oxibonsai eval --task mmlu on Ternary-Bonsai-1.7B.gguf, 2 real \
         questions through score_choices_logprob -> mmlu_accuracy={accuracy_pct}% ({notes})"
    );

    record_executed_timed(Capability::LegacyModels, TEST, gate_start.elapsed());
}

// ═════════════════════════════════════════════════════════════════════════
//  M-08: YaRN on the real Bonsai-8B.gguf at a >= 20000-token natural prompt
// ═════════════════════════════════════════════════════════════════════════
//
// `models/Bonsai-8B.gguf` declares `qwen3.rope.scaling.{type=yarn,
// factor=4.0, original_context_length=16384}`, and `BonsaiModel::from_gguf`
// builds its RoPE table from those keys (M-08). This gate proves, on the
// real file, that the scaling reaches the model's output at a prompt past
// the original context: the SCALED arm loads the file as shipped, the
// UNSCALED arm loads a private in-memory copy whose one
// `qwen3.rope.scaling.factor` value is patched from 4.0 to 1.0
// (`oxibonsai_testkit::gguf_fixture::patch_gguf_f32_metadata`, which checks
// the key, its type and its current value before writing). Factor 1.0 is
// accepted by `RopeTable::new_with_scaling`'s YaRN branch and reproduces the
// plain table (`rope.rs`'s `new_with_scaling_yarn_factor_one_matches_plain_new`),
// so both arms go through the same `from_gguf` -> `build_rope_table` path
// and differ only in that value; the usable context comes from the
// independent `qwen3.context_length` key, identical in both.
//
// How it runs. The prompt is 20000 tokens of NATURAL English: eleven
// public-domain passages (`M08_PASSAGES`), repeated in a different
// deterministic order each round (`m08_passage_round`) and tokenized once
// by the real `tokenizer.json`. Each arm runs in its own fresh child process
// (the Metal decode path is a process-global singleton), decodes the prompt
// one token at a time through the fused Metal path (the fused batch prefill
// is slower than sequential decode at every measured prompt size on this
// hardware — see the M-18 gate in `oxibonsai-model`'s
// `legacy_parity_tests.rs`), then greedily continues `M08_CONTINUE_TOKENS`
// steps. Each child writes the full logit vector at every continuation step
// and at `M08_SAMPLED_POSITIONS` (past the 16384-token original context) to
// a dump file; the parent compares the two arms' dumps.
//
// What it asserts. At every continuation step the two arms' logits must
// differ by more than `M08_DIVERGENCE_NOISE_MULTIPLE` times the numeric
// noise band the cross-backend parity gate above accepts for two
// computations of the SAME model (`per_step_bound` of the scaled step), so
// the divergence cannot be float noise. Step 0 is the clean comparison: both
// arms have consumed exactly the same 20000 tokens and differ only in the
// RoPE table. The per-step top-2 margins, the max-abs logit difference and
// whether the two continuations are still on the same history are printed
// for every row BEFORE any assertion, so a failure is diagnosable from its
// own output, and the dumps are kept on failure.
//
// Token inequality is a recorded soft check, not the assertion: both arms
// can pick the same greedy tokens while their logits differ well beyond
// noise — with a prompt of random token ids both arms fall into the same
// digit-repetition loop, which is why the prompt is natural text — so
// "the continuations differ" is reported with its margins, never required.
//
// Scope of the comparison: this is scaled-vs-unscaled inside OxiBonsai, not
// a comparison against a captured llama.cpp RoPE golden. Any later
// comparison of a real long-context run against the PrismML fork must use
// the fork's own f32 ITERATIVE RoPE cache (`theta *= theta_scale`) as the
// reference, not an f64 oracle: f32 angle quantisation alone moves a RoPE
// value by ~1.8e-3 absolute at position 30000 (ulp ~2e-3 at theta ~18139
// rad) — expected, not a bug. That gap cannot arise between this gate's two
// arms, which share one table builder.
//
// Opt-in: one run costs roughly an hour per arm on an M3 (the per-token rate
// grows with position as attention scans the whole history), so the test
// self-skips (recorded `executed: false`) unless `OXIBONSAI_M08_RUN_LONG=1`.
// `scripts/release-gate.sh` requires it by name whenever that variable is
// set, and its usage text makes that run a mandatory, separate release step.

/// Opt-in switch for [`m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens`].
const M08_RUN_LONG_ENV: &str = "OXIBONSAI_M08_RUN_LONG";
/// Child-process switches, set only by the parent test.
const M08_CHILD_ARM_ENV: &str = "OXIBONSAI_M08_CHILD_ARM";
const M08_CHILD_PROMPT_ENV: &str = "OXIBONSAI_M08_CHILD_PROMPT_IDS";
const M08_CHILD_DUMP_ENV: &str = "OXIBONSAI_M08_CHILD_DUMP";
/// This test's own name, for the child re-exec and the capability record.
const M08_TEST_FN: &str = "m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens";

/// Prompt length: past `qwen3.rope.scaling.original_context_length` (16384).
const M08_PROMPT_TOKENS: usize = 20_000;
/// Greedy continuation steps compared after the prompt.
const M08_CONTINUE_TOKENS: usize = 16;
/// Context budget: the prompt plus the continuation, with headroom.
const M08_MAX_SEQ: usize = M08_PROMPT_TOKENS + 64;
/// Prompt positions whose logits are also compared (recorded, not
/// asserted): the last position inside the original 16384-token context and
/// four past it. Each must be below `M08_PROMPT_TOKENS - 1`; the logits at
/// the final prompt position are continuation step 0.
const M08_SAMPLED_POSITIONS: [usize; 5] = [16_383, 16_384, 17_408, 18_432, 19_456];
/// Progress-line cadence inside a child.
const M08_PROGRESS_EVERY: usize = 1_000;
/// Hang guard per child — roughly three times the measured per-arm time.
const M08_CHILD_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10_800);
/// The YaRN factor key the unscaled arm patches, and its shipped value.
const M08_FACTOR_KEY: &str = "qwen3.rope.scaling.factor";
const M08_SHIPPED_FACTOR: f32 = 4.0;

/// The stated divergence threshold: at every continuation step the two
/// arms' max-abs logit difference must exceed this many times
/// [`per_step_bound`] of the scaled step — the largest disagreement the
/// cross-backend parity gate accepts as float noise between two
/// computations of the same model (`5e-3 * max(1, max|logit|)`, itself 4x
/// the worst cross-backend deviation measured on the real models, see
/// `REL_TOL`). 5x that bound is ~20x the worst measured float noise.
///
/// Calibrated on this file and this prompt: at a 3072-token prefix of the
/// same natural prompt, the 16 continuation steps measured 15.2x-43.0x the
/// bound (max-abs logit difference 1.61-5.18 against bounds of 0.094-0.157,
/// identical greedy tokens in both arms), so 5x keeps 3x headroom below the
/// smallest divergence observed.
const M08_DIVERGENCE_NOISE_MULTIPLE: f32 = 5.0;

/// Dump-file layout: this magic, a `u32` vocabulary size, then records of
/// `kind: u8`, `index: u32` (a prompt position or a continuation step) and
/// `vocab` `f32` logits, all little-endian.
const M08_DUMP_MAGIC: &[u8; 8] = b"OXM08LG1";
const M08_KIND_PROMPT: u8 = b'P';
const M08_KIND_CONTINUATION: u8 = b'C';

/// Eleven public-domain passages (a prime count, so every stride in
/// `1..11` yields a full permutation in [`m08_passage_round`]): the US
/// Declaration of Independence (1776), the Gettysburg Address (1863), the
/// Preamble to the US Constitution (1787), and the openings of A Tale of Two
/// Cities (1859), Pride and Prejudice (1813), Moby-Dick (1851), Walden
/// (1854), Lincoln's Second Inaugural Address (1865) and Alice's Adventures
/// in Wonderland (1865).
const M08_PASSAGES: [&str; 11] = [
    "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.",
    "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed.",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war.",
    "But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here.",
    "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.",
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
    "I went to the woods because I wished to live deliberately, to front only the essential facts of life, and see if I could not learn what it had to teach, and not, when I came to die, discover that I had not lived. I did not wish to live what was not life, living is so dear; nor did I wish to practise resignation, unless it was quite necessary.",
    "With malice toward none, with charity for all, with firmness in the right as God gives us to see the right, let us strive on to finish the work we are in, to bind up the nation's wounds, to care for him who shall have borne the battle and for his widow and his orphan, to do all which may achieve and cherish a just and lasting peace among ourselves and with all nations.",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'",
];

/// Round `round` of the prompt: all eleven passages, each exactly once, in
/// the order `i -> (stride * i + offset) mod 11` with a stride and offset
/// that change every round, joined by blank lines. Deterministic.
fn m08_passage_round(round: usize) -> String {
    let n = M08_PASSAGES.len();
    let stride = 1 + round % (n - 1);
    let offset = (3 * round) % n;
    (0..n)
        .map(|i| M08_PASSAGES[(stride * i + offset) % n])
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The gate's prompt: enough rounds of [`m08_passage_round`] to pass
/// [`M08_PROMPT_TOKENS`] once tokenized as ONE text, cut to exactly that
/// many tokens.
fn m08_natural_prompt_ids(tokenizer: &TokenizerBridge) -> Vec<u32> {
    let per_round = tokenizer
        .encode(&m08_passage_round(0))
        .expect("encode one round of the M-08 passages")
        .len();
    assert!(per_round > 0, "one round of passages encoded to no tokens");
    let rounds = M08_PROMPT_TOKENS / per_round + 2;
    let text = (0..rounds)
        .map(m08_passage_round)
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut ids = tokenizer.encode(&text).expect("encode the M-08 prompt");
    assert!(
        ids.len() >= M08_PROMPT_TOKENS,
        "{rounds} rounds of passages encoded to only {} tokens, fewer than {M08_PROMPT_TOKENS}",
        ids.len()
    );
    ids.truncate(M08_PROMPT_TOKENS);
    ids
}

/// Every round of the M-08 prompt holds each passage exactly once, and the
/// order really does change from round to round — the property the prompt
/// relies on to be natural text that is not one short period repeated. No
/// model file needed.
#[test]
fn m08_prompt_rounds_are_permutations_of_every_passage() {
    let first_rounds: Vec<String> = (0..10).map(m08_passage_round).collect();
    for (round, text) in first_rounds.iter().enumerate() {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        assert_eq!(paragraphs.len(), M08_PASSAGES.len(), "round {round}");
        for passage in &M08_PASSAGES {
            assert_eq!(
                paragraphs.iter().filter(|p| *p == passage).count(),
                1,
                "round {round} must hold every passage exactly once"
            );
        }
    }
    let distinct: std::collections::HashSet<&String> = first_rounds.iter().collect();
    assert_eq!(
        distinct.len(),
        first_rounds.len(),
        "the first ten rounds must all use different passage orders"
    );
    for pos in M08_SAMPLED_POSITIONS {
        assert!(
            pos + 1 < M08_PROMPT_TOKENS,
            "sampled position {pos} must precede the final prompt position"
        );
    }
}

/// Writes one child's logit records (see [`M08_DUMP_MAGIC`] for the layout).
struct M08DumpWriter {
    file: std::io::BufWriter<std::fs::File>,
    vocab: Option<usize>,
}

impl M08DumpWriter {
    fn create(path: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self {
            file: std::io::BufWriter::new(std::fs::File::create(path)?),
            vocab: None,
        })
    }

    fn record(&mut self, kind: u8, index: usize, logits: &[f32]) -> std::io::Result<()> {
        use std::io::Write;
        match self.vocab {
            None => {
                self.file.write_all(M08_DUMP_MAGIC)?;
                self.file.write_all(&u32_le(logits.len()))?;
                self.vocab = Some(logits.len());
            }
            Some(vocab) if vocab != logits.len() => {
                return Err(std::io::Error::other(format!(
                    "logit vector length changed from {vocab} to {}",
                    logits.len()
                )));
            }
            Some(_) => {}
        }
        self.file.write_all(&[kind])?;
        self.file.write_all(&u32_le(index))?;
        for &value in logits {
            self.file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }
}

/// `value` as 4 little-endian bytes; every count this file writes is far
/// below `u32::MAX`, which the conversion checks.
fn u32_le(value: usize) -> [u8; 4] {
    u32::try_from(value)
        .unwrap_or_else(|_| panic!("{value} does not fit in a u32"))
        .to_le_bytes()
}

/// One logit record read back from a child's dump.
struct M08Record {
    kind: u8,
    index: usize,
    logits: Vec<f32>,
}

fn m08_read_dump(path: &std::path::Path) -> Vec<M08Record> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("read M-08 dump {}: {e}", path.display()));
    let read_u32 = |at: usize| -> usize {
        let chunk: [u8; 4] = bytes
            .get(at..at + 4)
            .and_then(|s| s.try_into().ok())
            .unwrap_or_else(|| panic!("M-08 dump {} truncated at byte {at}", path.display()));
        u32::from_le_bytes(chunk) as usize
    };
    assert!(
        bytes.starts_with(M08_DUMP_MAGIC),
        "M-08 dump {} does not start with the expected magic",
        path.display()
    );
    let vocab = read_u32(M08_DUMP_MAGIC.len());
    let mut at = M08_DUMP_MAGIC.len() + 4;
    let mut records = Vec::new();
    while at < bytes.len() {
        let kind = bytes[at];
        let index = read_u32(at + 1);
        let start = at + 5;
        let end = start + 4 * vocab;
        let payload = bytes.get(start..end).unwrap_or_else(|| {
            panic!(
                "M-08 dump {} truncated inside record {}",
                path.display(),
                records.len()
            )
        });
        let logits = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        records.push(M08Record {
            kind,
            index,
            logits,
        });
        at = end;
    }
    records
}

fn m08_read_ids(path: &std::path::Path) -> Vec<u32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read M-08 prompt ids {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "M-08 prompt ids file is not whole u32s");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Loads `arm`'s model — the shipped file, or the copy with the YaRN factor
/// patched to 1.0 — on the auto-detected GPU tier with its weights
/// uploaded, as `InferenceEngine::from_gguf` does for this one-bit file.
/// The bytes and the parsed file are leaked: this runs only inside a
/// short-lived child process dedicated to one arm.
fn m08_load_arm(arm: &str) -> (BonsaiModel<'static>, KernelDispatcher) {
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
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
        GgufFile::parse(bytes).expect("parse Bonsai-8B.gguf"),
    ));
    let factor = gguf
        .metadata
        .get_f32(M08_FACTOR_KEY)
        .expect("the YaRN factor key is present");
    println!("M08_ARM arm={arm} {M08_FACTOR_KEY}={factor}");

    let kernel = KernelDispatcher::auto_detect();
    assert_eq!(
        kernel.tier(),
        KernelTier::Gpu,
        "KernelDispatcher::auto_detect() did not select the GPU tier in the M-08 {arm} child"
    );
    let mut model = BonsaiModel::from_gguf(gguf, M08_MAX_SEQ).expect("BonsaiModel::from_gguf");
    model.upload_weights_to_gpu(&kernel);
    let effective_context = model.max_context();
    println!("M08_EFFECTIVE_CONTEXT arm={arm} value={effective_context}");
    assert!(
        effective_context >= M08_MAX_SEQ,
        "M-08 {arm} child's effective context {effective_context} is below {M08_MAX_SEQ}"
    );
    (model, kernel)
}

/// One arm's work, inside its own child process: decode the prompt token by
/// token, record the sampled positions and every continuation step, print
/// the human-readable lines.
fn run_m08_child(arm: &str) {
    let prompt_path = std::env::var(M08_CHILD_PROMPT_ENV)
        .unwrap_or_else(|_| panic!("{M08_CHILD_PROMPT_ENV} is not set in the M-08 {arm} child"));
    let dump_path = std::env::var(M08_CHILD_DUMP_ENV)
        .unwrap_or_else(|_| panic!("{M08_CHILD_DUMP_ENV} is not set in the M-08 {arm} child"));
    let prompt = m08_read_ids(std::path::Path::new(&prompt_path));
    assert!(
        prompt.len() >= 2,
        "the M-08 prompt needs at least two tokens"
    );

    let (mut model, kernel) = m08_load_arm(arm);
    let mut dump = M08DumpWriter::create(std::path::Path::new(&dump_path))
        .unwrap_or_else(|e| panic!("create M-08 dump {dump_path}: {e}"));

    let start = std::time::Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        logits = model
            .forward(token, pos, &kernel)
            .unwrap_or_else(|e| panic!("M-08 {arm} prompt position {pos}: {e}"));
        if pos == 0 {
            assert!(
                model.gpu_path_active(),
                "M-08 {arm}: the first token did not run on the fused Metal decode path"
            );
        }
        if pos + 1 < prompt.len() && M08_SAMPLED_POSITIONS.contains(&pos) {
            dump.record(M08_KIND_PROMPT, pos, &logits)
                .unwrap_or_else(|e| panic!("write M-08 dump: {e}"));
            let top = top_two(&logits);
            println!(
                "M08_SAMPLE arm={arm} pos={pos} argmax={} top2_margin={:.6}",
                top.index(),
                top.gap()
            );
        }
        if pos > 0 && pos % M08_PROGRESS_EVERY == 0 {
            let elapsed = start.elapsed();
            println!(
                "M08_PROGRESS arm={arm} pos={pos} elapsed_ms={} us_per_token={:.1}",
                elapsed.as_millis(),
                elapsed.as_micros() as f64 / pos as f64
            );
        }
    }
    println!(
        "M08_PROMPT_DONE arm={arm} tokens={} elapsed_ms={}",
        prompt.len(),
        start.elapsed().as_millis()
    );

    let mut generated = Vec::with_capacity(M08_CONTINUE_TOKENS);
    for step in 0..M08_CONTINUE_TOKENS {
        let top = top_two(&logits);
        let token = argmax_first(&logits);
        dump.record(M08_KIND_CONTINUATION, step, &logits)
            .unwrap_or_else(|e| panic!("write M-08 dump: {e}"));
        println!(
            "M08_STEP arm={arm} step={step} token={token} top2_margin={:.6}",
            top.gap()
        );
        generated.push(token);
        if step + 1 < M08_CONTINUE_TOKENS {
            logits = model
                .forward(token, prompt.len() + step, &kernel)
                .unwrap_or_else(|e| panic!("M-08 {arm} continuation step {step}: {e}"));
        }
    }
    assert!(
        model.gpu_path_active(),
        "M-08 {arm}: the decode finished off the fused Metal path"
    );
    dump.finish()
        .unwrap_or_else(|e| panic!("flush M-08 dump {dump_path}: {e}"));
    println!(
        "M08_TOKENS arm={arm} tokens={}",
        generated
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
}

/// One compared row: a sampled prompt position or a continuation step.
struct M08Row {
    label: String,
    continuation: bool,
    same_history: bool,
    scaled_argmax: usize,
    scaled_margin: f32,
    unscaled_argmax: usize,
    unscaled_margin: f32,
    max_abs_diff: f32,
    noise_bound: f32,
}

impl M08Row {
    fn threshold(&self) -> f32 {
        M08_DIVERGENCE_NOISE_MULTIPLE * self.noise_bound
    }
}

/// M-08: YaRN scaling changes the real `Bonsai-8B.gguf`'s logits beyond
/// float noise at every greedy continuation step after a 20000-token natural
/// prompt (see this section's header comment for the design, the stated
/// threshold and why token inequality is only a recorded soft check).
/// Opt-in via `OXIBONSAI_M08_RUN_LONG=1`; otherwise self-skips with an
/// `executed: false` record.
#[test]
fn m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens() {
    const TEST: &str = "oxibonsai-runtime::legacy_parity_tests::\
                        m08_yarn_scaling_moves_real_bonsai_8b_logits_at_20000_tokens";

    if let Ok(arm) = std::env::var(M08_CHILD_ARM_ENV) {
        run_m08_child(&arm);
        return;
    }

    let _gpu = gpu_serial();
    let gate_start = std::time::Instant::now();
    if find_model("Bonsai-8B.gguf").is_none() {
        eprintln!("skip: Bonsai-8B.gguf not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };
    if std::env::var(M08_RUN_LONG_ENV).as_deref() != Ok("1") {
        eprintln!(
            "skip: {M08_RUN_LONG_ENV}=1 is not set -- this real {M08_PROMPT_TOKENS}-token decode \
             takes about an hour per arm and is a separate release step"
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }

    let tokenizer = TokenizerBridge::from_file(
        tokenizer_path
            .to_str()
            .expect("models/ path is valid UTF-8"),
    )
    .expect("load real tokenizer.json");
    let prompt = m08_natural_prompt_ids(&tokenizer);
    let excerpt = |ids: &[u32]| tokenizer.decode(ids).unwrap_or_else(|e| format!("<{e}>"));
    eprintln!(
        "M08_PROMPT tokens={} head={:?} tail={:?}",
        prompt.len(),
        excerpt(&prompt[..48]),
        excerpt(&prompt[prompt.len() - 48..])
    );
    let prompt_bytes: Vec<u8> = prompt.iter().flat_map(|id| id.to_le_bytes()).collect();
    let prompt_file = oxibonsai_testkit::temp_path::TempFile::write(
        "oxibonsai-m08-prompt",
        ".ids",
        &prompt_bytes,
    )
    .expect("write the M-08 prompt ids");
    let prompt_path = prompt_file
        .path()
        .to_str()
        .expect("temp dir path is valid UTF-8")
        .to_string();

    let mut dumps = Vec::new();
    let mut tokens_by_arm = Vec::new();
    for arm in ["scaled", "unscaled"] {
        let dump_path = oxibonsai_testkit::temp_path::unique_path(
            &format!("oxibonsai-m08-{arm}-logits"),
            ".bin",
        );
        let dump_str = dump_path
            .to_str()
            .expect("temp dir path is valid UTF-8")
            .to_string();
        let run = oxibonsai_testkit::parity::run_named_test_in_child(
            M08_TEST_FN,
            &[
                (M08_CHILD_ARM_ENV, arm),
                (M08_CHILD_PROMPT_ENV, prompt_path.as_str()),
                (M08_CHILD_DUMP_ENV, dump_str.as_str()),
            ],
            M08_CHILD_DEADLINE,
        )
        .unwrap_or_else(|e| panic!("spawn the M-08 {arm} child: {e}"));
        eprint!("{}", run.output);
        match run.status {
            None => panic!(
                "the M-08 {arm} child did not finish within {M08_CHILD_DEADLINE:?} and was killed"
            ),
            Some(status) => assert!(
                status.success(),
                "the M-08 {arm} child exited with {status:?}; its output is above"
            ),
        }
        let prefix = format!("M08_TOKENS arm={arm} tokens=");
        let tokens: Vec<u32> = run
            .output
            .lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .unwrap_or_else(|| panic!("the M-08 {arm} child printed no M08_TOKENS line"))
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("an M08_TOKENS entry is a u32"))
            .collect();
        assert_eq!(
            tokens.len(),
            M08_CONTINUE_TOKENS,
            "{arm} continuation length"
        );
        tokens_by_arm.push(tokens);
        dumps.push(dump_path);
    }

    let scaled = m08_read_dump(&dumps[0]);
    let unscaled = m08_read_dump(&dumps[1]);
    let expected_records = M08_SAMPLED_POSITIONS.len() + M08_CONTINUE_TOKENS;
    assert_eq!(scaled.len(), expected_records, "scaled dump record count");
    assert_eq!(
        unscaled.len(),
        expected_records,
        "unscaled dump record count"
    );

    let (scaled_tokens, unscaled_tokens) = (&tokens_by_arm[0], &tokens_by_arm[1]);
    let mut rows = Vec::with_capacity(expected_records);
    for (s, u) in scaled.iter().zip(&unscaled) {
        assert!(
            s.kind == u.kind && s.index == u.index && s.logits.len() == u.logits.len(),
            "the two M-08 dumps are not aligned record for record"
        );
        let continuation = s.kind == M08_KIND_CONTINUATION;
        let (label, same_history) = if continuation {
            (
                format!("step {}", s.index),
                scaled_tokens[..s.index] == unscaled_tokens[..s.index],
            )
        } else {
            (format!("pos {}", s.index), true)
        };
        let max_abs_diff = s
            .logits
            .iter()
            .zip(&u.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scaled_absmax = s.logits.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let (st, ut) = (top_two(&s.logits), top_two(&u.logits));
        rows.push(M08Row {
            label,
            continuation,
            same_history,
            scaled_argmax: st.index(),
            scaled_margin: st.gap(),
            unscaled_argmax: ut.index(),
            unscaled_margin: ut.gap(),
            max_abs_diff,
            noise_bound: per_step_bound(scaled_absmax),
        });
    }

    eprintln!(
        "M08_TABLE row | history | scaled argmax (top2 margin) | unscaled argmax (top2 margin) \
         | max|dlogit| | noise bound | x bound"
    );
    for row in &rows {
        eprintln!(
            "M08_ROW {} | {} | {} ({:.4}) | {} ({:.4}) | {:.4} | {:.4} | {:.1}",
            row.label,
            if row.same_history { "same" } else { "diverged" },
            row.scaled_argmax,
            row.scaled_margin,
            row.unscaled_argmax,
            row.unscaled_margin,
            row.max_abs_diff,
            row.noise_bound,
            row.max_abs_diff / row.noise_bound
        );
    }
    let decode = |ids: &[u32]| tokenizer.decode(ids).unwrap_or_else(|e| format!("<{e}>"));
    match scaled_tokens
        .iter()
        .zip(unscaled_tokens)
        .position(|(a, b)| a != b)
    {
        Some(step) => eprintln!(
            "M08_SOFT_CHECK the greedy continuations diverge at step {step}: scaled={:?} \
             unscaled={:?}",
            decode(scaled_tokens),
            decode(unscaled_tokens)
        ),
        None => eprintln!(
            "M08_SOFT_CHECK the greedy continuations are identical ({:?}); the logit rows above \
             carry the divergence and each step's top-2 margins",
            decode(scaled_tokens)
        ),
    }

    let below: Vec<&M08Row> = rows
        .iter()
        .filter(|row| row.continuation && row.max_abs_diff <= row.threshold())
        .collect();
    if !below.is_empty() {
        panic!(
            "YaRN scaling did not move the real Bonsai-8B.gguf's logits beyond \
             {M08_DIVERGENCE_NOISE_MULTIPLE}x the numeric noise band at {} continuation step(s) \
             after a {M08_PROMPT_TOKENS}-token prompt (first: {}, max|dlogit| {:.4} <= threshold \
             {:.4}). Dumps kept at {:?}; the full table is above.",
            below.len(),
            below[0].label,
            below[0].max_abs_diff,
            below[0].threshold(),
            dumps
        );
    }

    for dump in &dumps {
        // Best effort: a leftover dump in the temp dir is harmless.
        let _ = std::fs::remove_file(dump);
    }
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, TEST, elapsed);
    record_executed_timed(Capability::LegacyModels, TEST, elapsed);
}

// ── Notes ────────────────────────────────────────────────────────────────
//
// 1. The engine constructors' `EosTokenSet`/`uses_fused_gpu_decode` scope
//    cut — see this file's header doc comment.
// 2. The D-6 comparison block (`compare_against_reference` and its
//    supporting types/constants) is shared with
//    `crates/oxibonsai-model/tests/legacy_parity_tests.rs` through
//    `oxibonsai_testkit::parity` rather than duplicated.
// 3. RESOLVED: all nine
//    (model, prompt) golden-TEXT comparisons pass.
//    `Bonsai-8B.gguf` (Q1_0, 1-bit) prompt 3 ("Once upon a time, in a small
//    village by the sea,") once diverged from a stale, pre-YaRN-fix golden
//    capture at greedy step 14 (id 22208 " curious" vs the stale golden's id
//    2948 " love", top-2 gap 8.05e-2) — NOT a cross-tier parity break: the
//    logit pass for this exact case, run before the text pass, completed
//    bit-exact-CPU/relative-bound-GPU on all three tiers first, so Reference,
//    the auto-detected CPU tier and Metal all agreed with EACH OTHER and
//    only differed from the stale capture. Root cause: `Bonsai-8B.gguf`
//    declares YaRN rope scaling (factor 4, `original_context_length`
//    16384); the stale capture predated this engine honouring it. Once
//    honoured (matching llama.cpp exactly), the PrismML llama.cpp fork
//    itself emits the "curious nature" text byte-for-byte — GOLDEN_CASES
//    above carries the re-captured text, matching the fork oracle, not the
//    stale pre-fix capture.
// 4. Cross-PROCESS serialization: [`gpu_serial`] only serializes these gates
//    inside one test binary, which is what `cargo test`'s in-process thread
//    pool needs — `cargo nextest` runs each test in its own process, so
//    this alone could not stop a `--workspace` nextest run from scheduling
//    several multi-GB real-model gates (this file's and the model crate's)
//    concurrently. `.config/nextest.toml` pins this whole binary to its
//    single-threaded `real-model-gate` test group, so a nextest run executes
//    these cases one at a time and never alongside another real-model case
//    (the M-08 long-context gate additionally gets its own, longer
//    slow-timeout there); `scripts/release-gate.sh`'s stage 1b also invokes
//    the binary directly, serialized, in release, with `--test-threads=1`,
//    which is where the release gate collects its evidence.
