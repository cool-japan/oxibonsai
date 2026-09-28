//! Product-level CPU/NEON/Metal greedy parity gate for the legacy models,
//! driving the real tokenizer and the real `SamplingParams` construction the
//! CLI uses (orchestrator P0 addendum, 2026-09-21; see
//! `scratchpad/findings/legacy_cpu_metal_divergence.md`; verifier finding
//! that requested this exact file, wave 3).
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
//!    internally; see `sampling.rs`) — driven by an engine explicitly pinned
//!    to each of this host's kernel tiers via
//!    [`InferenceEngine::from_model_with_tier`], and the decoded text
//!    checked against the golden Metal capture as a common prefix.
//!
//! ## Scope note: this does NOT drive `generate_greedy_gpu`'s 4-byte fast path
//!
//! `InferenceEngine::greedy_gpu_eligible` additionally requires
//! `self.uses_fused_gpu_decode()`, which `from_model_with_tier` (via
//! `from_model_with_kernel` → `assemble(..., false)`) always sets to
//! `false` — only `from_gguf`/`from_gguf_path` (which always
//! *auto-detect* the kernel tier, so they cannot be pinned to an explicit
//! tier for a controlled tier-vs-tier comparison) resolve that flag from
//! the GGUF's actual fused-route metadata. So a `KernelTier::Gpu` engine
//! built here still computes every forward pass through the real Metal
//! ternary/K-quant kernels (`self.kernel` is genuinely pinned to `Gpu`), it
//! just samples via the general `Sampler::sample_with_history` path instead
//! of the specialized GPU-resident 4-byte-argmax-download optimization.
//! `crates/oxibonsai-runtime/tests/metal_greedy_cpu_fallback_tests.rs`
//! (already owned by this package) is the test that exercises
//! `generate_greedy_gpu` itself directly; this file's job is the tokenizer
//! and `SamplingParams` and golden-text half the addendum asked for, not a
//! second copy of that coverage.
//!
//! ## Two independent passes per tier (deliberately not sharing one engine)
//! - **Text pass** ([`run_text_pass`]): a fresh engine per tier drives
//!   `generate()` end to end; the result is detokenized and compared
//!   against the golden Metal text as a common prefix (never exact-length
//!   equality — the golden capture's own token budget may differ from
//!   [`MAX_TOKENS`], and an empty/short decode is explicitly rejected by
//!   [`MIN_DECODED_CHARS`] rather than trivially "passing").
//! - **Logit pass** ([`run_logit_pass`]): a second, fresh engine per tier
//!   drives [`InferenceEngine::prefill_from_pos`]/[`InferenceEngine::decode_step`]
//!   manually with a first-index argmax ([`argmax_first`]), bypassing
//!   `Sampler` entirely — mirroring
//!   `oxibonsai-model/tests/legacy_parity_tests.rs`'s own
//!   `generate_with_logits` verbatim (duplicated, not shared — see this
//!   file's deviations footer) so the same cross-tier comparison
//!   ([`compare_against_reference`]) applies to these real tokenized prompts
//!   too.
//!
//! ## What this gate asserts, after orchestrator ruling D-6 (2026-09-22)
//!
//! 1. **The greedy TOKEN CHAIN is the hard gate**, unconditionally, for every
//!    tier pair — it used to be asserted only when a near-tie classifier
//!    counted zero argmax flips.
//! 2. **`KernelTier::Reference` vs the auto-detected CPU tier must be
//!    BIT-EXACT** on every step's logits (measured exactly 0.0 over 64
//!    self-generated steps; wave3.5.md §2.2).
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
//! `EosTokenSet` note: `from_model_with_tier` falls back to
//! `EosTokenSet::single(engine::EOS_TOKEN_ID)` rather than resolving the
//! GGUF's own `tokenizer.ggml.eos_token_id` (only `InferenceEngine::from_gguf`
//! does that, and it cannot be pinned to an explicit tier — see above). A
//! wrong fallback EOS id is only a real risk if it collides with a normal
//! content token inside the first `MAX_TOKENS` of one of these specific
//! prompts; the golden captures show it does not.
//!
//! `--seed`/`seed` is inert at `temperature == 0.0` on every backend (no RNG
//! is ever consulted before the argmax fast path returns — see
//! `legacy_cpu_metal_divergence.md` §1), so the exact value used below has
//! no effect on the outcome.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};

/// Generation budget: "≥64 tokens on repetition-inducing prompts" per the
/// orchestrator's P0 addendum.
const MAX_TOKENS: usize = 64;

/// Context budget: generous headroom over `MAX_TOKENS` plus the longest
/// prompt below, well inside all three legacy models' trained context.
const MAX_SEQ: usize = 512;

/// A deterministic seed. Inert at `temperature == 0.0` (see header doc).
const SEED: u64 = 42;

/// Below this many decoded characters, a "common prefix matches the golden
/// text" assertion would be vacuous (T-09 is this very package's own
/// charter: a real assertion, not one that trivially passes on an empty or
/// near-empty string).
const MIN_DECODED_CHARS: usize = 20;

// ── Per-step cross-tier classifier (mirrored from the model-side file) ─────
//
// `crates/oxibonsai-model/tests/legacy_parity_tests.rs` carries a
// byte-identical copy of everything from `MAX_PER_STEP_LOGIT_DELTA` to
// `gpu_serial` below; keep the two in sync by hand. Duplicated rather than
// shared because `oxibonsai-testkit` has no parity module and this package
// does not own one — see this file's deviations footer.

/// First-index argmax tie-break, matching
/// `oxibonsai_runtime::engine_greedy::argmax_first`'s documented contract
/// (mirrored here, not imported — same cross-crate-wiring note as above).
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

// ── Cross-tier per-step comparison (orchestrator ruling D-6, 2026-09-22) ───
//
// D-6 re-shapes three things about the pre-3.5 classifier, in this priority
// order (full measurement table: `scratchpad/pkg/wave3.5.md` §2):
//
//  1. TOKEN-CHAIN EQUALITY IS THE HARD GATE — unconditional, every pair. It
//     used to fire only when the near-tie classifier counted zero flips, so a
//     run could diverge in output and still pass. Token equality is the
//     product promise (README's "CPU and Metal produce byte-identical output
//     at --temperature 0"), so it is asserted FIRST: a numeric panic can no
//     longer pre-empt the verdict that actually matters.
//  2. The Reference-vs-CPU-tier pair is asserted BIT-EXACT. Measured exactly
//     0.0 over 64 self-generated greedy steps on all three prompt slots of
//     `Ternary-Bonsai-1.7B.gguf` (wave3.5.md §2.2), with the same batched
//     prefill this gate uses. That is far stronger than the 5e-3 absolute
//     bound it replaces, and it is what would actually catch a CPU-kernel or
//     CPU-prefill regression.
//  3. Only the Reference-vs-GPU pair takes a numeric tolerance, and that
//     tolerance is RELATIVE — see [`per_step_bound`].
//
// The pre-3.5 near-tie BUDGET is gone, and that is a strengthening, not a
// weakening: with token equality asserted unconditionally the budget was
// unreachable (`tokens[step]` *is* `argmax(logits[step])` by construction, so
// "a near-tie argmax flip" and "a token-chain divergence" are the same event),
// and D-6(1) rules that a flip is a hard failure rather than something to
// absorb. The near-tie numbers are still COMPUTED and printed inside the
// token-divergence panic (D-6(3)'s "print the top-2 gap so a near-tie is
// distinguishable from a defect"), so the old signal survives as diagnosis
// instead of as tolerance.

/// Lower bound on what counts as a per-step breach for the Reference-vs-GPU
/// pair — **not** an additional ceiling (see [`per_step_bound`]). It exists so
/// a fixture whose logits all sit near zero cannot turn every float wobble
/// into a failure.
const MAX_PER_STEP_LOGIT_DELTA: f32 = 5e-3;

/// Relative per-step tolerance for the Reference-vs-GPU pair (D-6(3)).
///
/// 4× headroom over the worst RELATIVE cross-backend deviation measured
/// anywhere on the real models — wave3.5.md §2.1/§2.2, 2026-09-22, 64
/// self-generated greedy steps per slot, `max|Δ| / max(1, max|reference|)`:
///
/// | model | slot | worst \|Δ\| | relative |
/// |---|---|---|---|
/// | `Ternary-Bonsai-1.7B` | 0 | 6.375e-3 @step51 | 2.41e-4 |
/// | `Ternary-Bonsai-1.7B` | 1 | 5.853e-3 @step6  | 4.40e-4 |
/// | `Ternary-Bonsai-1.7B` | 2 | 2.806e-2 @step51 | **1.237e-3** (worst) |
/// | `Ternary-Bonsai-8B`   | 0 | 4.321e-3 @step19 | 2.47e-4 |
/// | `Ternary-Bonsai-8B`   | 1 | 7.793e-3 @step61 | 3.47e-4 |
/// | `Ternary-Bonsai-8B`   | 2 | 4.981e-3 @step36 | 3.62e-4 |
///
/// Token chains were identical on all six (model, slot) pairs, and
/// `Bonsai-8B.gguf` cleared even the old absolute bound outright. The
/// evidence base these numbers extend (`legacy_cpu_metal_divergence.md` §3.4)
/// is itself RELATIVE — 2.31e-5 at one teacher-forced step — so a deviation
/// that compounds to 1.24e-3 over 51 self-generated steps is consistent with
/// it, not a regression signature. An ABSOLUTE bound cannot distinguish a
/// 6e-3 deviation on a logit of 26 from one on a logit of 1, which is why
/// D-6 re-shaped the bound instead of raising it.
const REL_TOL: f32 = 5e-3;

/// Context for the near-tie numbers printed inside a token-divergence panic:
/// the pre-3.5 classifier tolerated an argmax flip whose reference-side
/// top-to-runner-up gap was below `max(1e-2, 32 × delta)`. These two
/// constants no longer gate anything (see the D-6 note above); they exist so
/// a failure report can still say whether the flip *looked* like a near-tie.
const NEAR_TIE_ABS_FLOOR: f32 = 1e-2;
const NEAR_TIE_REL_MULT: f32 = 32.0;

/// D-6(3)'s per-step bound for the Reference-vs-GPU pair, written out
/// literally because the combinator is the one thing an implementer can
/// invert while believing they followed the spec: it is `max`, **not** `&&`.
///
/// `MAX_PER_STEP_LOGIT_DELTA` is a FLOOR on what counts as a breach, never a
/// second ceiling. A step with `ref_absmax = 22.68` and `delta = 2.806e-2` is
/// GREEN here (bound = 0.113) — that is one of the six measured points in
/// [`REL_TOL`]'s table, so an `&&` formulation would fail a case the wave-3.5
/// triage declared correct.
///
/// D-6's own text writes the floor as `FLOOR = 1e-4` where wave3.5.md §2.4
/// writes `MAX_PER_STEP_LOGIT_DELTA` (5e-3). The two are the SAME predicate:
/// `REL_TOL * max(1, ref_absmax) >= REL_TOL = 5e-3 > 1e-4` for every possible
/// `ref_absmax`, so the floor is subsumed by the relative term either way.
/// The larger, more conservative of the two constants is the one used here.
///
/// `ref_absmax` is the step's `max|reference logit|`, and the `delta` it is
/// compared against is the step's `max|Δlogit|` — i.e. this is a per-step
/// bound on a per-step maximum, which is what reproduces D-6's own worked
/// example (`5e-3 × 22.68 = 0.113`). The per-ELEMENT relative deviation is
/// computed too, but only ever printed: the measurement base behind these
/// numbers is a per-step maximum, so asserting on a per-element ratio would
/// assert something nobody measured.
///
/// D-6(3)'s own addendum text describes the bound as "per element, per
/// step", which reads as a stricter, literally-per-element predicate than
/// the one implemented here. A wave-3.5 verifier probe measured
/// `StepStats::worst_elem_rel` (this file's diagnostic-only per-element
/// ratio) directly: on `Ternary-Bonsai-1.7B.gguf`, Reference vs Metal, 64
/// self-generated steps, the worst per-element relative deviation on each
/// of the three prompt slots was 5.754e-3 / 5.813e-3 / 1.567e-2 — ALL THREE
/// exceed the naive per-element floor of 5e-3 (since `max(1, |ref_i|) >= 1`
/// for every `i`, a literal per-element reading's bound is exactly `5e-3`
/// there). A literal per-element assertion would therefore keep this gate
/// RED on the real models permanently; the per-STEP predicate this function
/// implements — which item 3(a) of the spec writes out literally
/// (`ref_absmax = max|reference logit|`) — is the only one that both
/// reproduces D-6's own worked example and passes on measured real-model
/// data. RATIFICATION REQUESTED, not self-approved: this file cannot edit
/// the addendum's own "per element, per step" wording (it lives in
/// `scratchpad/pkg/FIX3-PARITY.json`'s spec, not a file this package owns);
/// the request is to correct that wording to match item 3(a)'s literal
/// (and already-shipped) per-step predicate, so the next reader does not
/// re-open this.
fn per_step_bound(ref_absmax: f32) -> f32 {
    f32::max(
        MAX_PER_STEP_LOGIT_DELTA,
        REL_TOL * f32::max(1.0, ref_absmax),
    )
}

/// One tier's complete greedy run for one prompt: the token chain it
/// generated and every step's full logit vector.
struct TierRun {
    tier: KernelTier,
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

/// The winner and runner-up of one logit vector (first-index tie-break,
/// matching [`argmax_first`]).
#[derive(Debug, Clone, Copy)]
struct TopTwo {
    idx: usize,
    top: f32,
    second: f32,
}

impl TopTwo {
    fn gap(self) -> f32 {
        self.top - self.second
    }
}

fn top_two(v: &[f32]) -> TopTwo {
    let mut idx = 0usize;
    let mut top = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > top {
            second = top;
            top = x;
            idx = i;
        } else if x > second {
            second = x;
        }
    }
    TopTwo { idx, top, second }
}

/// Everything one step of one tier pair measured. Collected for all
/// [`MAX_TOKENS`] steps BEFORE anything is asserted, so a failure report can
/// print the whole series (TESTS-INFRA blocking 1's second half: the old
/// message said only `step N`, which is how the wave-3 verifier misattributed
/// the failing tier pair).
struct StepStats {
    /// `max_i |reference_i - other_i|`.
    delta: f32,
    /// The index that attained `delta`.
    delta_at: usize,
    /// `max_i |reference_i|` — the scale D-6(3)'s bound is relative to.
    ref_absmax: f32,
    /// Diagnostic only, never asserted (see [`per_step_bound`]):
    /// `max_i |Δ_i| / max(1, |reference_i|)`.
    worst_elem_rel: f32,
    /// How many elements are not bit-identical (the CPU pair's assertion).
    differing: usize,
    /// Non-finite logits on either side. Always a hard failure: `f32::max`
    /// silently swallows a NaN operand, so a NaN could otherwise hide inside
    /// `delta` without ever tripping a `<=` bound.
    non_finite: usize,
    ref_top: TopTwo,
    other_top: TopTwo,
}

impl StepStats {
    fn measure(reference: &[f32], other: &[f32]) -> Self {
        let mut delta = 0.0f32;
        let mut delta_at = 0usize;
        let mut ref_absmax = 0.0f32;
        let mut worst_elem_rel = 0.0f32;
        let mut differing = 0usize;
        let mut non_finite = 0usize;
        for (i, (&a, &b)) in reference.iter().zip(other.iter()).enumerate() {
            if !a.is_finite() || !b.is_finite() {
                non_finite += 1;
            }
            if a != b {
                differing += 1;
            }
            let d = (a - b).abs();
            if d > delta {
                delta = d;
                delta_at = i;
            }
            let m = a.abs();
            if m > ref_absmax {
                ref_absmax = m;
            }
            let rel = d / f32::max(1.0, m);
            if rel > worst_elem_rel {
                worst_elem_rel = rel;
            }
        }
        Self {
            delta,
            delta_at,
            ref_absmax,
            worst_elem_rel,
            differing,
            non_finite,
            ref_top: top_two(reference),
            other_top: top_two(other),
        }
    }

    /// `delta / max(1, ref_absmax)` — the quantity [`REL_TOL`]'s table lists.
    fn relative(&self) -> f32 {
        self.delta / f32::max(1.0, self.ref_absmax)
    }
}

/// The `k` largest entries of `v` as `(index, value)`, largest first — one
/// side of "print each side's top-5 and argmax at the breaching step".
fn top_k(v: &[f32], k: usize) -> String {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    let parts: Vec<String> = idx.iter().map(|&i| format!("({i}, {:e})", v[i])).collect();
    format!("[{}]", parts.join(", "))
}

/// The full per-step delta series (D-6 item 3(d)): one line per step, so the
/// SHAPE of a divergence over the chain is visible at a glance instead of
/// being inferred from a single number.
fn delta_series(stats: &[StepStats]) -> String {
    let mut out =
        String::from("      step |        max|Δ| |  max|ref| |   relative |      bound | breach\n");
    for (step, s) in stats.iter().enumerate() {
        let bound = per_step_bound(s.ref_absmax);
        out.push_str(&format!(
            "      {step:>4} | {:>13e} | {:>9.4} | {:>10e} | {:>10e} | {}\n",
            s.delta,
            s.ref_absmax,
            s.relative(),
            bound,
            if s.delta > bound { "YES" } else { "" }
        ));
    }
    out
}

/// The "what happened at this exact step" half of every failure message:
/// both sides' argmax, top-2 gap and top-5, plus the near-tie context the
/// pre-3.5 classifier used to decide with.
fn step_detail(step: usize, s: &StepStats, reference: &[f32], other: &[f32]) -> String {
    let near_tie_bound = NEAR_TIE_ABS_FLOOR.max(NEAR_TIE_REL_MULT * s.delta);
    format!(
        "    step {step}: max|Δ|={:e} at index {} | max|ref|={:.6} | relative={:e} | bound={:e} \
         | differing elements={} of {} | non-finite={}\n      \
         reference: argmax={} top={:.6} second={:.6} top-2 gap={:e}\n      \
         other:     argmax={} top={:.6} second={:.6} top-2 gap={:e}\n      \
         reference top-5: {}\n      \
         other     top-5: {}\n      \
         near-tie context (pre-3.5 classifier, diagnostic only now): reference gap {:e} vs \
         near-tie bound {:e} -> {}\n      \
         worst per-element relative deviation this step (diagnostic, not asserted): {:e}",
        s.delta,
        s.delta_at,
        s.ref_absmax,
        s.relative(),
        per_step_bound(s.ref_absmax),
        s.differing,
        reference.len(),
        s.non_finite,
        s.ref_top.idx,
        s.ref_top.top,
        s.ref_top.second,
        s.ref_top.gap(),
        s.other_top.idx,
        s.other_top.top,
        s.other_top.second,
        s.other_top.gap(),
        top_k(reference, 5),
        top_k(other, 5),
        s.ref_top.gap(),
        near_tie_bound,
        if s.ref_top.gap() <= near_tie_bound {
            "would have LOOKED like a near-tie"
        } else {
            "NOT a near-tie -- a real disagreement"
        },
        s.worst_elem_rel,
    )
}

/// Is this pair's non-reference side a CPU tier? Only an accelerator pair
/// gets a numeric tolerance (D-6(3)); every CPU tier must be BIT-EXACT
/// against `KernelTier::Reference` (D-6(2)).
///
/// Written as an exhaustive match over the named CPU tiers, deliberately
/// NOT as `!matches!(tier, KernelTier::Gpu)`. D-6(1) speaks of "CUDA when
/// present", and `KernelTier` is not `#[non_exhaustive]`: a future wave that
/// adds an accelerator variant would silently inherit the bit-exactness
/// assertion under the negated form and go red for a non-bug, whereas this
/// form fails to COMPILE here and forces the classification to be decided.
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

/// Compare one tier's whole run against the Reference tier's, in D-6's
/// priority order. `context` names the model file and the prompt slot/text —
/// the wave-3 verifier misattributed this gate's failing tier pair precisely
/// because the old message carried none of that.
fn compare_against_reference(context: &str, reference: &TierRun, other: &TierRun) {
    let pair = format!(
        "{context}: tier {} ({:?}) vs {} ({:?})",
        other.tier, other.tier, reference.tier, reference.tier
    );
    assert_eq!(
        reference.logits.len(),
        other.logits.len(),
        "{pair}: step-count mismatch"
    );
    let stats: Vec<StepStats> = reference
        .logits
        .iter()
        .zip(other.logits.iter())
        .map(|(r, o)| {
            assert_eq!(r.len(), o.len(), "{pair}: logit vector length mismatch");
            StepStats::measure(r, o)
        })
        .collect();

    // ── (1) PRIMARY, unconditional: the greedy token chains must match ────
    // D-6(1). Checked before any numeric bound, so a tolerance panic can
    // never hide — or be mistaken for — an output divergence.
    if other.tokens != reference.tokens {
        // COSMETIC (wave-3.5 verifier): `.unwrap_or(0)` would silently point
        // at step 0 if the two chains ever differed only in LENGTH (a
        // prefix relationship) instead of at some shared index -- inert
        // today (both chains are always MAX_TOKENS long by construction)
        // but a silent wrong-index guess is the wrong failure mode for a
        // length mismatch, which `other.tokens != reference.tokens` above
        // already proved must exist. Fail loudly and specifically instead.
        let first = match reference
            .tokens
            .iter()
            .zip(other.tokens.iter())
            .position(|(a, b)| a != b)
        {
            Some(i) => i,
            None => panic!(
                "{pair}: token chains differ only in LENGTH ({} vs {} tokens) -- every \
                 shared-length position matches, which should be impossible by construction \
                 (both chains are always {MAX_TOKENS} tokens long)\n  reference chain: {:?}\n  \
                 other  chain: {:?}",
                reference.tokens.len(),
                other.tokens.len(),
                reference.tokens,
                other.tokens,
            ),
        };
        let detail = step_detail(
            first,
            &stats[first],
            &reference.logits[first],
            &other.logits[first],
        );
        panic!(
            "{pair}: GREEDY TOKEN CHAINS DIVERGED (D-6(1): the hard gate — the product promises \
             byte-identical output at --temperature 0)\n  first divergence at step {first}: \
             reference picked {}, other picked {}\n{detail}\n  reference chain: {:?}\n  other  \
             chain: {:?}\n  per-step delta series:\n{}",
            reference.tokens[first],
            other.tokens[first],
            reference.tokens,
            other.tokens,
            delta_series(&stats),
        );
    }

    // ── (2) numeric: bit-exact for a CPU pair, relative bound for GPU ─────
    if is_cpu_tier(other.tier) {
        // D-6(2). `assert_eq!` on the raw vectors would dump ~150 000 floats
        // twice on failure and say nothing useful; this asserts the same
        // predicate (every element bit-identical) and reports what differs.
        for (step, s) in stats.iter().enumerate() {
            assert!(
                s.differing == 0,
                "{pair}: CPU-TIER LOGITS ARE NOT BIT-EXACT at step {step} (D-6(2) asserts exact \
                 equality here: measured exactly 0.0 over 64 steps x 3 slots on \
                 Ternary-Bonsai-1.7B.gguf — a failure here is a REAL FINDING to report, not a \
                 bound to loosen)\n{}\n  per-step delta series:\n{}",
                step_detail(step, s, &reference.logits[step], &other.logits[step]),
                delta_series(&stats),
            );
        }
    } else {
        for (step, s) in stats.iter().enumerate() {
            assert!(
                s.non_finite == 0,
                "{pair}: {} NON-FINITE logits at step {step} — a NaN/inf is never a tolerance \
                 question\n{}\n  per-step delta series:\n{}",
                s.non_finite,
                step_detail(step, s, &reference.logits[step], &other.logits[step]),
                delta_series(&stats),
            );
            let bound = per_step_bound(s.ref_absmax);
            assert!(
                s.delta <= bound,
                "{pair}: max|Δlogit|={:e} at step {step} exceeds the D-6(3) relative bound {:e} = \
                 max({MAX_PER_STEP_LOGIT_DELTA:e}, {REL_TOL:e} * max(1, {:.6}))\n{}\n  per-step \
                 delta series:\n{}",
                s.delta,
                bound,
                s.ref_absmax,
                step_detail(step, s, &reference.logits[step], &other.logits[step]),
                delta_series(&stats),
            );
        }
    }
}

/// Serializes the real-model gates inside one test binary.
///
/// Every gate below drives `KernelTier::Gpu`. MET-08 (METAL-CONCURRENCY,
/// landed wave 4) split the old single mutable `GLOBAL_METAL_GRAPH` into a
/// process-shared, immutable `MetalDevice` and a per-session `MetalGraph`
/// (its own command queue, its own device KV cache) bound to a thread via
/// `SessionScope` — so N sessions *can* now overlap on the GPU. None of the
/// gates below bind a session, though, so `MetalGraph::global()` falls
/// through to the single process-default session (`session.rs`'s own
/// documented fallback, quoted verbatim: "A process that never binds
/// therefore behaves exactly as it did before this split: one session, one
/// queue, one KV cache, byte-identical output"), and
/// two of these gates running on `cargo test`'s default in-process thread
/// pool would still share it. Same pattern as
/// `crates/oxibonsai-runtime/tests/metal_greedy_cpu_fallback_tests.rs`'s
/// `gpu_serial()`, which was added for exactly this reason after the wave-3
/// verifier reproduced a 3-of-3 failure under default parallelism.
///
/// This is an in-binary guard only. `cargo nextest` runs each test in its own
/// PROCESS, where the shared default session is not an issue but the MEMORY
/// is: three real models (0.5 GB + 2.2 GB + 1.2 GB on disk, several GB
/// resident each) decoded concurrently drove this 8-core/24 GB M3 to load
/// average 96 during the wave-3.5 triage. **The release gate must invoke
/// these tests with `--test-threads=1`** — `scripts/release-gate.sh` does;
/// `.config/nextest.toml`'s `real-model-gate` test-group additionally pins
/// this binary to one thread under `cargo nextest`.
fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ── Golden fixtures (embedded from scratchpad/golden_legacy/legacy_golden.json) ─

/// One `(model, prompt, golden Metal continuation text)` triple.
struct GoldenCase {
    /// GGUF filename under `models/`.
    model_file: &'static str,
    /// The exact prompt text the golden capture was taken from.
    prompt: &'static str,
    /// The captured Metal-backend greedy continuation (does NOT include the
    /// prompt itself), from `legacy_golden.json`'s `backend == "metal"`
    /// rows — the orchestrator P0 addendum's chosen greedy truth.
    metal_text: &'static str,
}

/// The 9 `backend == "metal"` rows of
/// `scratchpad/golden_legacy/legacy_golden.json` (3 legacy models x 3
/// prompts each), embedded verbatim.
const GOLDEN_CASES: [GoldenCase; 9] = [
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital",
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: "\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n",
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        metal_text: " there lived a young girl named Lila. She was known for her kindness and her love for the sea. One day, she discovered a mysterious shell that gl",
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital",
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: " if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2) print(fibonacci(10)) \n\nThe",
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        metal_text: " there lived a young girl named Lila. She was known for her curiosity and love for the ocean. One day, while exploring the beach, she discovered a",
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital",
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: "\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef fibonacci(n):",
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        // RE-CAPTURED 2026-09-23: Bonsai-8B.gguf declares `rope.scaling.type = yarn`
        // (factor 4, original_context_length 16384). OxiBonsai <= 0.2.4 ignored it; the
        // engine now honours it (M-08), and this text is byte-identical to the PrismML
        // llama.cpp fork (greedy, temp 0) on the same file. See the session ruling
        // `golden_legacy/RULING_bonsai8b_yarn.md`.
        metal_text: " there lived a young girl named Lila. She was known for her curious nature and her love of the sea. Lila often spent her days exploring the cliffs",
    },
];

/// The distinct model filenames in [`GOLDEN_CASES`], in first-appearance
/// order (drives the three top-level `#[test]` functions below).
const LEGACY_MODEL_FILES: [&str; 3] = [
    "Ternary-Bonsai-1.7B.gguf",
    "Ternary-Bonsai-8B.gguf",
    "Bonsai-8B.gguf",
];

// ── T-05 capability-report producer ─────────────────────────────────────────
//
// T-07 FIX (verifier wave 3): this used to be an inline copy of
// `oxibonsai_testkit::capability::record`; `oxibonsai-runtime` now takes
// `oxibonsai-testkit` as a dev-dependency (imported above), so the copy is
// deleted in favour of the shared implementation.

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
/// exactly the orchestrator P0 addendum's fix for the CPU/Metal divergence
/// this file guards against.
fn greedy_sampling_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    }
}

/// The length of the shared byte range of `a` and `b` (i.e.
/// `a.len().min(b.len())`), walked back to the largest value that is a
/// valid UTF-8 char boundary in BOTH strings (a hand-rolled, dual-string
/// `str::floor_char_boundary` — not yet stable on this workspace's
/// `rust-version = "1.89"`), so slicing `&a[..n]`/`&b[..n]` can never panic
/// on a multi-byte character straddling the cut point.
fn shared_compare_len(a: &str, b: &str) -> usize {
    let mut n = a.len().min(b.len());
    while n > 0 && !(a.is_char_boundary(n) && b.is_char_boundary(n)) {
        n -= 1;
    }
    n
}

/// Asserts `decoded` is a plausible, non-vacuous match for `golden`:
/// long enough to rule out a truncated-but-not-empty generation (at least
/// [`MIN_DECODED_CHARS`], AND at least `golden`'s own length capped at 40
/// chars — a spurious early EOS a few tokens in must still fail this, not
/// just a fully empty decode), and byte-identical to `golden` over their
/// shared length (never full-length equality — the golden capture's own
/// token budget may differ from [`MAX_TOKENS`]).
///
/// T-09 (this package's own charter): a bug that made `decoded` empty (an
/// immediate EOS, a broken tokenizer round-trip, ...) — or merely short,
/// from an EOS a handful of tokens in — must fail this assertion, not
/// vacuously satisfy "is a prefix of everything".
///
/// `logit_pass_run` is this same case/tier's already-computed [`TierRun`]
/// from [`run_logit_pass`] (now run BEFORE the text pass in
/// [`run_model_gate`], see that function's doc comment), threaded through
/// purely for diagnostics on a mismatch (wave-3.5 verifier, "TEXT-HALF
/// FAILURE MESSAGE carries no logit diagnostics"): the logit pass is an
/// independent decode (fresh engine, manual argmax loop — see the module
/// doc's "Two independent passes per tier"), so its token ids are used only
/// to locate the first differing TOKEN and its logits are printed as
/// diagnostic context, never asserted equal to this pass's own tokens.
fn assert_prefix_matches_golden(
    decoded: &str,
    golden: &str,
    context: &str,
    tokenizer: &TokenizerBridge,
    generated_tokens: &[u32],
    logit_pass_run: Option<&TierRun>,
) {
    let min_chars = MIN_DECODED_CHARS.max(golden.chars().count().min(40));
    assert!(
        decoded.chars().count() >= min_chars,
        "{context}: decoded continuation is implausibly short ({} chars, wanted >= {min_chars}): {decoded:?}",
        decoded.chars().count()
    );
    let n = shared_compare_len(decoded, golden);
    if decoded[..n] == golden[..n] {
        return;
    }

    // Tokenize the golden continuation and diff it against this tier's own
    // generated ids to find the first differing TOKEN (not just byte), then
    // — when this case/tier's logit pass already ran — print that step's
    // top-2 gap and top-5 so a near-tie is distinguishable from a real
    // disagreement, exactly as D-6(3) asks for the numeric half. `top_two`/
    // `top_k` already exist for this; nothing new is invented here.
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
        "{context}: decoded text does not match the golden Metal capture over their shared \
         {n}-byte prefix\n  decoded: {decoded:?}\n  golden:  {golden:?}{token_detail}"
    );
}

/// Runs the "text pass" for one `(tier, case)` pair: a fresh engine drives
/// `generate()` end to end, and the detokenized result is checked against
/// the golden text as a common prefix. `logit_pass_run` is threaded through
/// to [`assert_prefix_matches_golden`] purely for diagnostics — see its doc
/// comment.
fn run_text_pass(
    gguf_bytes: &[u8],
    tier: KernelTier,
    tokenizer: &TokenizerBridge,
    case: &GoldenCase,
    logit_pass_run: Option<&TierRun>,
) {
    let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
    let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    let mut engine =
        InferenceEngine::from_model_with_tier(model, tier, greedy_sampling_params(), SEED);

    let prompt_ids = tokenizer.encode(case.prompt).expect("encode golden prompt");
    let generated = engine
        .generate(&prompt_ids, MAX_TOKENS)
        .expect("engine.generate");
    let decoded = tokenizer.decode(&generated).expect("decode generated ids");

    assert_prefix_matches_golden(
        &decoded,
        case.metal_text,
        // FIX3-PARITY item 5: the GOLDEN-TEXT half is reported separately
        // from the numeric gate — a token-chain match against the METAL
        // goldens with only a numeric bound tripping is a very different
        // release posture from a text divergence, so every message says
        // which half it came from.
        &format!(
            "TEXT PASS {} prompt {:?} tier {tier} ({tier:?})",
            case.model_file, case.prompt
        ),
        tokenizer,
        &generated,
        logit_pass_run,
    );
}

/// Runs the "logit pass" for one case across every tier in `tiers`: a fresh
/// engine per tier drives a manual, pure-argmax decode loop (bypassing
/// `Sampler`) and returns each tier's `(token ids, per-step logits)`.
fn run_logit_pass(gguf_bytes: &[u8], tiers: &[KernelTier], prompt_ids: &[u32]) -> Vec<TierRun> {
    let mut per_tier = Vec::with_capacity(tiers.len());
    for &tier in tiers {
        let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
        let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
        let mut engine =
            InferenceEngine::from_model_with_tier(model, tier, greedy_sampling_params(), SEED);

        let mut logits = engine
            .prefill_from_pos(prompt_ids, 0)
            .expect("prefill_from_pos");
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
/// LOGIT BEFORE TEXT, deliberately (wave-3.5 verifier finding, "OVERCLAIM in
/// the escalation footnote"): `tiers` always starts with `KernelTier::
/// Reference`, so under the previous text-then-logit order a text-pass
/// mismatch on the very first tier panicked out of this function before the
/// logit pass — the cross-tier numeric verdict for that case — ever ran.
/// This file's own deviations footer used to claim "this file's own logit
/// pass ... passes on Bonsai-8B.gguf" for exactly the case where that pass
/// had never executed. Running the logit pass first means that claim is now
/// either earned (the logit pass actually completes and its verdict is
/// real) or superseded by a logit-pass panic that reports the true failure
/// — never asserted without having been checked.
fn run_model_gate(model_file: &'static str, test_name: &str) {
    // Serialized against this binary's other real-model gates — see
    // [`gpu_serial`]. Taken before the fixture probes so the whole gate,
    // including its capability bookkeeping, is one critical section.
    let _gpu = gpu_serial();

    // D-6(4) / FIX3-PARITY item 1: these tests are NOT `#[ignore]`d — they
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
                compare_against_reference(&context, reference, other);
            }
        }

        // ---- Text pass: real tokenizer + real SamplingParams, per tier.
        // `per_tier` (just computed above) is threaded through so a text
        // mismatch's panic can report the matching tier's own logit-pass
        // step data (top-2 gap / top-5) instead of only the decoded
        // strings — see `assert_prefix_matches_golden`. ----
        for &tier in &tiers {
            let logit_run = per_tier.iter().find(|t| t.tier == tier);
            run_text_pass(&gguf_bytes, tier, &tokenizer, case, logit_run);
        }
    }

    // T-05 honesty (wave-1 addendum (1)(b), FIX3-PARITY item 2(b)): the
    // `executed: true` records are written HERE, once the whole matrix —
    // every golden case, every tier, both passes — has actually completed.
    // Writing them on fixture PRESENCE (as this file used to, at the top of
    // the function) claims coverage for work that may still panic, which is
    // the exact false-green class the capability manifest exists to
    // eliminate. Any assertion above panics out of this function and neither
    // record is written.
    record_executed(Capability::Metal, test_name);
    record_executed(Capability::LegacyModels, test_name);
}

// The `legacy_parity_capability_probe` that used to live here is DELETED
// (FIX3-PARITY item 2(a)). It existed only because the three gates below were
// `#[ignore]`d, so their bodies never ran under a plain `cargo test` and the
// manifest could not distinguish "models absent" from "never instrumented" —
// and it wrote `executed: <fixture present>`, i.e. it claimed execution on
// fixture PRESENCE. With the `#[ignore]`s gone (item 1) the gates themselves
// always run and record: `executed: false` from their self-skip branch,
// `executed: true` only after the whole matrix has actually completed.

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

// ── Deviations (see this package's structured output for the full list) ────
//
// 1. `from_model_with_tier`'s `EosTokenSet`/`uses_fused_gpu_decode` scope cut
//    — see this file's header doc comment.
// 2. The D-6 comparison block (`MAX_PER_STEP_LOGIT_DELTA` .. `gpu_serial`) is
//    duplicated byte-for-byte in
//    `crates/oxibonsai-model/tests/legacy_parity_tests.rs`. Sharing it would
//    mean a new `oxibonsai_testkit::parity` module; `crates/oxibonsai-testkit
//    /src/parity.rs` is not in this package's owned files, so the copy stays
//    and the two must be kept in sync by hand.
// 3. ESCALATED, NOT FIXED (FIX3-PARITY item 5; wave-3.5 verifier RE-CONFIRMED this by an
//    independent run and root-caused it further; this pass re-confirmed it a THIRD time, from
//    this file's own reordered gate, with no separate probe needed). As of 2026-09-23 this gate
//    is RED on exactly one of its nine (model, prompt) golden-TEXT comparisons —
//    `Bonsai-8B.gguf` (Q1_0, 1-bit) with the prompt "Once upon a time, in a small village by the
//    sea,". The spec's instruction for a text divergence is to ESCALATE it, never to fix or
//    weaken it locally, so the assertion stands as written. This is the exact panic this tree
//    produces, verbatim (`OXIBONSAI_MODELS_DIR=<checkout>/models cargo test --release -p
//    oxibonsai-runtime --features metal --test legacy_parity_tests -- --include-ignored
//    --test-threads=1 --nocapture`, 2026-09-23, this file's reordered `run_model_gate` — logit
//    pass before text pass, see item below on the OVERCLAIM fix):
//    RESOLVED 2026-09-23 (orchestrator ruling, golden_legacy/RULING_bonsai8b_yarn.md): the
//    divergence recorded below was NOT a defect. Bonsai-8B.gguf declares YaRN rope scaling
//    (factor 4, original_context_length 16384); the pre-session baseline ignored it, the
//    engine now honours it (M-08) exactly like llama.cpp, and the PrismML llama.cpp fork
//    emits the "curious nature" text byte-for-byte. GOLDEN_CASES above was re-captured.
//    Historical record of the escalation (pre-ruling):
//      TEXT PASS Bonsai-8B.gguf prompt "Once upon a time, in a small village by the sea," tier
//      reference (Reference): decoded text does not match the golden Metal capture over their
//      shared 141-byte prefix
//        decoded: " there lived a young girl named Lila. She was known for her curious nature
//        and her love of the sea. Lila often spent her days exploring the cliffs, listening to
//        the waves, and collecting seashells. One day, while collecting seashells, she found a
//        small, weathered box. Inside,"
//        golden:  " there lived a young girl named Lila. She was known for her love of the sea
//        and her ability to speak with the waves. One day, while exploring"
//        first differing TOKEN index 14: this tier picked id 22208 (" curious"), golden picked
//        id 2948 (" love") | this tier's LOGIT-PASS step 14 (an independent decode — diagnostic
//        only): top-2 gap=8.047676e-2 top-5=[(22208, 1.40623e1), (2948, 1.3981823e1), (40228,
//        1.3925102e1), (44872, 1.3344414e1), (42020, 1.2410843e1)]
//    That last line is this file's OWN diagnostics (the MINOR "TEXT-HALF FAILURE MESSAGE" fix
//    below), and it reproduces the wave-3.5 verifier's separately-built probe to every digit
//    (their measurement: "top-2 gap 8.048e-2", top-5
//    `[(22208,14.0623),(2948,13.9818),(40228,13.9251),(44872,13.3444),(42020,12.4108)]`) — this
//    gate is now self-diagnosing; no external probe was needed to produce the paragraph above.
//    ROOT-CAUSE (unchanged from the verifier's disposition, now reproduced by this file itself):
//    at Reference-tier greedy step 14 of this exact prompt, this tree's TOP candidate is id 22208
//    (" curious", logit 14.0623) and the OLD (pre-session, HEAD 4a6a266) golden's choice, id 2948
//    (" love", logit 13.9818 in THIS tree), is the RUNNER-UP — a top-2 gap of 8.048e-2 (5.7e-3
//    relative to |logit|~14). Somewhere in the wave-1..3 diffs to the Q1_0/Q1_0_g128 CPU
//    dequant/dispatch path (`crates/oxibonsai-core/src/quant_std.rs`,
//    `crates/oxibonsai-kernels/src/{dispatch.rs,dispatch_std_quant.rs,gemm_onebit.rs}` —
//    substantially reworked across these waves for the new PrismML PTQ1_0/PQ2_0 quant types, see
//    CONTEXT.md's "NEW TARGET" section; NONE are in this package's `owned_files`) an ~8e-2
//    absolute logit shift flipped this one near-tie.
//    EARNED, not merely asserted (this is the fix for the separate MINOR "OVERCLAIM in the
//    escalation footnote" finding — `run_model_gate` now runs the LOGIT pass before the TEXT
//    pass for every case, specifically so a claim like the next sentence is either earned or
//    superseded by a `LOGIT PASS` panic reporting the true failure): it is NOT a cross-tier
//    parity break in THIS tree — no `LOGIT PASS` panic precedes the `TEXT PASS` panic printed
//    above, which is only possible because case 3's logit pass, which unconditionally runs
//    before its text pass, completed for all three tiers without panicking; therefore Reference
//    is bit-exact to the auto-detected CPU tier and the Metal pair clears the D-6(3) relative
//    bound, with identical token chains, on this exact prompt, in this exact run. CPU and Metal
//    agree with each other in this tree; both differ from the pre-session baseline.
//    Also still true, re-verified: the divergence is NOT a harness artifact (the pre-session
//    binary, HEAD 4a6a266, re-run on the same GGUF today still prints the golden exactly); it is
//    NOT this file's `MAX_SEQ = 512` (the CLI's 4096 agrees with this file's 512; both differ
//    from the golden). The other eight (model, prompt) pairs, OBSERVED in this same run (not
//    carried over from an earlier report): `Bonsai-8B.gguf` prompts 1 and 2 passed (the loop
//    processes `GOLDEN_CASES` in order and only case 3 panicked, so cases 1-2's text AND logit
//    passes, all tiers, already completed clean before the panic above);
//    `ternary_1_7b_greedy_text_matches_golden_across_tiers` (all 3 prompts) — PASSED (`test
//    ternary_1_7b_greedy_text_matches_golden_across_tiers ... ok`);
//    `ternary_8b_greedy_text_matches_golden_across_tiers` (all 3 prompts) — this comment was
//    written while that test was STILL EXECUTING in this run (host load average spiked past 27
//    on this 8-core M3 from other concurrent sessions during this pass, well past this test's
//    normal wall-clock budget), so this run's own verdict for it is not recorded here; its tier
//    matrix does not depend on the Bonsai-8B case above, and the ORIGINAL FIX3-PARITY dispatch
//    that this package is answering states the SAME test file's independent re-run as "2 passed
//    / 1 FAILED" with `bonsai_8b_greedy_text_matches_golden_across_tiers` the ONE named failure —
//    i.e. `ternary_8b_greedy_text_matches_golden_across_tiers` passed in that run, by elimination
//    (only 3 tests exist in this file). See this package's structured output for whether this
//    run's own copy finished and confirmed that by the time it was reported.
//    ORCHESTRATOR ACTION (unchanged disposition, now with sharper, self-produced evidence):
//    bisect the wave-1..3 diffs to the three CPU dequant/dispatch files above for the source of
//    the ~8e-2 shift, then RULE whether to accept this tree's output (and re-capture
//    `scratchpad/golden_legacy/legacy_golden.json` plus this file's embedded `GOLDEN_CASES` entry
//    from today's build) or to treat the shift as a defect to fix upstream. This package's
//    `owned_files` cannot reach that fix either way — the assertion is NOT weakened or deleted to
//    work around it.
// 4. Cross-PROCESS serialization of these gates is not fixable from here.
//    [`gpu_serial`] serializes them inside one test binary, which is what
//    `cargo test`'s in-process thread pool needs; `cargo nextest` runs each
//    test in its own process, so the three real-model gates in this file
//    (plus the three in the model crate) can still be scheduled concurrently
//    by a `--workspace` nextest run. ORCHESTRATOR ACTION ITEM, UNOWNED FILE
//    (wave-3.5 verifier, re-confirmed): the fix belongs in
//    `.config/nextest.toml`, which has NO wave-3.5 owner (wave3.5.md SS6 —
//    its only prior owners, CI-GATE (wave 1) and FIX-02-CI (wave 1.5), are
//    both closed, and this exact gap was not among the files SS6 grants
//    forward). This is not a case of "forgot to check": `ownership_index
//    .json` and wave3.5.md SS6's grant table were both read while preparing
//    this fix pass, and `scripts/release-gate.sh` — this package's own file
//    — already carries the drafted recipe verbatim (its "THE REAL-MODEL
//    PARITY STAGE MUST BE SERIALIZED" section), including a debug-vs-release
//    timing caveat measured on this host and a `default-filter`-based
//    alternative confirmed supported by the installed cargo-nextest
//    (0.9.108). Self-granting this file by analogy to how
//    `scripts/release-gate.sh` itself was granted (wave3.5.md SS6: "granted
//    to the package that discovered it") would be the same self-approval
//    move D-6's sequencing note and wave3.5.md S-5 ("a manifest comment is
//    not a sign-off") exist to prevent — grants flow through the planning
//    artifact BEFORE dispatch, not by the implementer deciding its own scope
//    is too narrow. This did not bite wave 3.5 only because this worktree's
//    `models/` holds a `.gitkeep`, not the real GGUFs; on a host with the
//    real `models/` populated (this repo's, not this worktree's), a plain
//    `cargo nextest run --workspace [--all-features]` reintroduces exactly
//    the load-average-96 concurrent-decode risk stage 1b above exists to
//    avoid, DEBUG-built. `scripts/release-gate.sh` (owned here) invokes
//    these gates directly with `--test-threads=1` in RELEASE and is
//    unaffected either way.
