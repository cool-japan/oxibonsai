//! Legacy 1.7B/8B greedy CPU↔Metal/NEON parity gate (B2-01 design §7.4,
//! orchestrator P0 addendum 2026-09-21) — the real-model-weights half this
//! crate can actually reach.
//!
//! ## Background: the divergence this test is the permanent regression guard for
//!
//! `scratchpad/findings/legacy_cpu_metal_divergence.md` (this session) found
//! that `oxibonsai run --temperature 0` was **not greedy on CPU**: the CLI
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
//! ## Scope cut vs. the addendum's full design (see the deviation at the
//! bottom for the exact missing piece and what it needs)
//!
//! The addendum asks this test to reach "the SAME public entry points the
//! CLI uses" and check output against `legacy_golden.json`'s captured
//! **decoded text**. Both need a real tokenizer/detokenizer, and `--
//! temperature 0`'s no-penalty construction (`src/cli/util.rs
//! ::build_sampling_params`, `pub(crate)` inside the `oxibonsai-cli` binary
//! crate) lives on `oxibonsai_runtime::sampling::SamplingParams`. Neither
//! `oxibonsai-tokenizer` nor `oxibonsai-runtime` is a dependency (not even a
//! dev-dependency) of `oxibonsai-model` today (`crates/oxibonsai-model
//! /Cargo.toml`, not owned by this package) — from inside this crate there
//! is no way to turn real English prompt text into token ids, or generated
//! ids back into text, at all. This file therefore drives the real model
//! weights with **deterministic numeric prompt ids** (the same convention
//! `crates/oxibonsai-runtime/tests/cuda_ternary_forward_parity.rs` already
//! uses for the same reason on the CUDA side) instead of the golden
//! prompts/text, and greedy params are constructed directly on the
//! low-level dispatch path (no `SamplingParams`/`Sampler` needed for pure
//! argmax) rather than through `build_sampling_params`. This still
//! delivers the addendum's core, most valuable claim — CPU and every
//! GPU/SIMD tier agree, end to end, on the actual shipped model weights —
//! just not the CLI-text-output half.
//!
//! ## What this gate asserts, after orchestrator ruling D-6 (2026-09-22)
//!
//! 1. **The greedy TOKEN CHAIN is the hard gate**, unconditionally, for every
//!    tier pair — it used to be asserted only when a near-tie classifier
//!    counted zero argmax flips.
//! 2. **`KernelTier::Reference` vs the auto-detected CPU tier must be
//!    BIT-EXACT** on every step's logits (measured exactly 0.0 over 64
//!    self-generated steps, all three prompt slots, on
//!    `Ternary-Bonsai-1.7B.gguf`; wave3.5.md §2.2).
//! 3. Only the Reference-vs-GPU (Metal) pair takes a numeric tolerance, and
//!    it is RELATIVE: `|Δ| <= max(5e-3, 5e-3 · max(1, max|reference|))` per
//!    step. See [`per_step_bound`] for why the combinator is `max` and not
//!    `&&`, and [`REL_TOL`] for the six measured (model, slot) points it is
//!    calibrated against. The pre-3.5 bound was the same 5e-3 as an ABSOLUTE
//!    ceiling, applied at step 51 of a self-generated chain whose every KV
//!    entry came from the diverging backend; that shape, not the kernels, is
//!    what turned the gate red for a whole wave.
//! 4. These tests are **not** `#[ignore]`d: they run whenever the models are
//!    present (`OXIBONSAI_MODELS_DIR`, else `<workspace>/models`) and
//!    otherwise self-skip into the capability manifest with
//!    `executed: false`. `executed: true` is written only after the whole
//!    tier matrix has actually run. **Invoke them with `--test-threads=1`** —
//!    see [`gpu_serial`].

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};

/// Generation budget: "≥64 tokens on repetition-inducing prompts" per the
/// orchestrator's P0 addendum.
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
// T-07 FIX (verifier wave 3): this used to be an inline copy of
// `oxibonsai_testkit::capability::record`; `oxibonsai-model` now takes
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

/// First-index argmax tie-break, matching
/// `oxibonsai_runtime::engine_greedy::argmax_first`'s documented contract
/// (mirrored here, not imported: see this file's header deviation).
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

/// Greedily decode `max_tokens` steps for `tier` at true-greedy settings (no
/// sampling, no penalty — argmax is unaffected by temperature/top-k/top-p
/// once `SamplingParams` is out of the picture entirely), returning both the
/// generated token ids and every step's full logit vector (needed for the
/// cross-tier classifier).
fn generate_with_logits(
    gguf_bytes: &[u8],
    tier: KernelTier,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    let kernel = KernelDispatcher::with_tier(tier);

    let mut logits = model
        .forward_prefill(prompt_ids, 0, &kernel)
        .expect("prefill");
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

// ── Per-step cross-tier classifier (mirrored in the runtime-side file) ────
//
// `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs` carries a
// byte-identical copy of everything from `MAX_PER_STEP_LOGIT_DELTA` to
// `gpu_serial` below; keep the two in sync by hand. Duplicated rather than
// shared because `oxibonsai-testkit` has no parity module and this package
// does not own one — see this file's deviations footer.

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

/// Runs the full parity gate for one legacy model: greedily decode
/// `KernelTier::Reference`, the auto-detected CPU tier (NEON on this Apple
/// Silicon host), and `KernelTier::Gpu` (Metal) on three deterministic
/// synthetic prompts, and compare every tier against the scalar Reference
/// tier under D-6 — token chain first (hard, unconditional), then bit-exact
/// logits for the CPU pair / the relative bound for the GPU pair.
fn run_legacy_model_gate(model: &LegacyModel, test_name: &str) {
    // Serialized against this binary's other real-model gates — see
    // [`gpu_serial`]. Taken before the fixture probe so the whole gate,
    // including its capability bookkeeping, is one critical section.
    let _gpu = gpu_serial();

    // D-6(4) / FIX3-PARITY item 1: this test is NOT `#[ignore]`d — it runs
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

        let mut per_tier: Vec<TierRun> = Vec::with_capacity(tiers.len());
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
            compare_against_reference(&context, reference, other);
        }
    }

    // T-05 honesty (wave-1 addendum (1)(b), FIX3-PARITY item 2(b)): the
    // `executed: true` records are written HERE, once the whole tier matrix —
    // every slot, every tier — has actually completed. Writing them on model
    // -file PRESENCE (as this file used to, at the top of the function)
    // claims coverage for work that may still panic, which is the exact
    // false-green class the capability manifest exists to eliminate. Any
    // assertion above panics out of this function and neither record is
    // written.
    record_executed(Capability::Metal, test_name);
    record_executed(Capability::LegacyModels, test_name);
}

// The `legacy_parity_capability_probe` that used to live here is DELETED
// (FIX3-PARITY item 2(a)). It existed only because the three gates below were
// `#[ignore]`d, so their bodies never ran under a plain `cargo test` and the
// manifest could not distinguish "models absent" from "never instrumented" —
// and it wrote `executed: <model file present>`, i.e. it claimed execution on
// fixture PRESENCE. With the `#[ignore]`s gone (item 1) the gates themselves
// always run and record: `executed: false` from their self-skip branch,
// `executed: true` only after the whole tier matrix has actually completed.

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

// ── Deviations (see this package's structured output for the full list) ────
//
// 1. SCOPE CUT (see this file's header doc comment for the full rationale):
//    this test drives real model weights with deterministic *numeric*
//    prompts and asserts cross-tier agreement, not "matches the golden CLI
//    text" — `oxibonsai-model` has no tokenizer/detokenizer dependency
//    (`oxibonsai-tokenizer`/`oxibonsai-runtime`, neither a dependency nor a
//    dev-dependency of `crates/oxibonsai-model/Cargo.toml`, not owned by
//    this package) to turn real English prompts into ids or ids back into
//    text. CLOSED (verifier wave 3): the golden-text half now lives in
//    `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs`, which already
//    depends on `oxibonsai-model` and has `TokenizerBridge` +
//    `sampling::SamplingParams` — it embeds `legacy_golden.json`'s exact 9
//    `(model, prompt, golden Metal text)` triples (`backend == "metal"`) and
//    asserts `decode(generate(prompt))`'s text starts with the golden text
//    (prefix, not exact match, since the golden capture's token budget may
//    differ from that file's own `MAX_TOKENS`). This file's numeric-prompt,
//    tokenizer-free cross-tier gate remains valuable in its own right (it
//    still runs with no tokenizer fixture at all) and was NOT removed.
// 1a. CLOSED (FIX3-PARITY, wave 3.5): the three real-model tests are no
//     longer `#[ignore]`d at all, and the `legacy_parity_capability_probe`
//     companion that stood in for them is deleted. `#[ignore]` reproduced the
//     original hole — the gate never ran even on a host that HAS the models,
//     which is why a red bound went undetected for a whole wave — so the
//     record-and-return self-skip below is the shape D-6(4) requires. Same
//     change in the sibling
//     `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs`.
// 1b. The D-6 comparison block (`MAX_PER_STEP_LOGIT_DELTA` .. `gpu_serial`) is
//     duplicated byte-for-byte in that sibling. Sharing it would mean a new
//     `oxibonsai_testkit::parity` module; `crates/oxibonsai-testkit/src
//     /parity.rs` is not in this package's owned files, so the copy stays and
//     the two must be kept in sync by hand.
// 1c. Cross-PROCESS serialization is not fixable from here. [`gpu_serial`]
//     serializes these gates inside one test binary, which is what `cargo
//     test`'s in-process thread pool needs; `cargo nextest` runs each test in
//     its own process, so a `--workspace` nextest run can still schedule
//     several multi-GB real-model gates concurrently. ORCHESTRATOR ACTION
//     ITEM, UNOWNED FILE (wave-3.5 verifier, re-confirmed): the fix belongs
//     in `.config/nextest.toml`, which has NO wave-3.5 owner (wave3.5.md
//     SS6 — its only prior owners, CI-GATE (wave 1) and FIX-02-CI (wave
//     1.5), are both closed, and this exact gap is not among the files SS6
//     grants forward). `scripts/release-gate.sh` — this package's own file
//     — carries the drafted recipe verbatim (its "THE REAL-MODEL PARITY
//     STAGE MUST BE SERIALIZED" section): a `default-filter` exclusion
//     (preferred; confirmed supported by the installed cargo-nextest
//     0.9.108) or a `test-group` with `max-threads = 1`, plus a
//     debug-vs-release timing caveat measured on this host. Self-granting
//     this file by analogy to how `scripts/release-gate.sh` itself was
//     granted (wave3.5.md SS6: "granted to the package that discovered it")
//     would be the self-approval move D-6's sequencing note and wave3.5.md
//     S-5 exist to prevent — grants flow through the planning artifact
//     before dispatch. `scripts/release-gate.sh` (owned here) invokes these
//     gates directly with `--test-threads=1` in RELEASE and is unaffected
//     either way.
// 2. `build_sampling_params`-equivalence: true greedy (temperature 0, no
//    penalty) does not need `SamplingParams` at all at this crate's level —
//    `argmax_first` on raw logits *is* the CLI's post-fix greedy contract
//    exactly. If item 1 lands and `SamplingParams` re-enters the picture,
//    construct it with `repetition_penalty: 1.0` (not
//    `SamplingParams::default()`'s `1.1`) to match
//    `src/cli/util.rs::build_sampling_params`'s documented behaviour with no
//    explicit `--repetition-penalty` flag.
// 3. CLOSED (FIX3-PARITY, wave 3.5): the "models present" branch is now
//    runtime-verified on the real GGUFs via
//    `OXIBONSAI_MODELS_DIR=<checkout>/models cargo test --release -p
//    oxibonsai-model --features metal --test legacy_parity_tests --
//    --test-threads=1`, which is also how the release gate invokes it.
