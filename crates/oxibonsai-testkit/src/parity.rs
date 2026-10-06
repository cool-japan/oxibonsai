//! Shared cross-tier per-step logit/token comparison for a real-model greedy
//! parity gate (design decision D-6, 2026-09-22).
//!
//! `crates/oxibonsai-model/tests/legacy_parity_tests.rs` and
//! `crates/oxibonsai-runtime/tests/legacy_parity_tests.rs` each drive the
//! real legacy GGUFs across every kernel tier a host can reach and need the
//! identical comparison logic to classify a per-step disagreement; this
//! module is that logic, generic over the caller's own tier type (`T`, kept
//! abstract here — this crate never depends on `oxibonsai-kernels`) so both
//! test files import one implementation instead of hand-keeping two
//! byte-identical copies in sync.
//!
//! # What a comparison asserts, in priority order
//!
//! 1. **The greedy TOKEN CHAIN is the hard gate**, unconditionally, for
//!    every tier pair: checked first, so a numeric-bound panic can never
//!    hide — or be mistaken for — an output divergence. This is the product
//!    promise ("CPU and Metal produce byte-identical output at
//!    `--temperature 0`").
//! 2. A **CPU-tier pair must be BIT-EXACT** on every step's logits — the
//!    caller says which of its own tier values count as a CPU tier via the
//!    `is_cpu_tier` closure [`compare_against_reference`] takes. Measured
//!    exactly 0.0 over 64 self-generated steps, all three prompt slots, on
//!    `Ternary-Bonsai-1.7B.gguf` between `KernelTier::Reference` and the
//!    auto-detected CPU tier.
//! 3. Only a non-CPU (accelerator) pair takes a numeric tolerance, and it is
//!    RELATIVE: `|Δ| <= max(`[`MAX_PER_STEP_LOGIT_DELTA`]`, `[`REL_TOL`]` *
//!    max(1, max|reference|))` per step — see [`per_step_bound`] for why the
//!    combinator is `max` and not `&&`, and [`REL_TOL`]'s doc comment for the
//!    measured (model, slot) points it is calibrated against.
//!
//! # `per_step_bound` is a per-STEP predicate, not per-element
//!
//! The design text that motivated [`per_step_bound`] describes the relative
//! bound as applying "per element, per step". The shipped predicate
//! implemented here is per-**step** only: `ref_absmax` is the step's
//! `max|reference logit|` over the WHOLE step, and `delta` is the step's
//! `max|Δlogit|` — a per-step bound on a per-step maximum. A literal
//! per-element reading would keep this gate permanently RED on the real
//! models: measured worst per-ELEMENT relative deviations
//! (`StepStats::worst_elem_rel`, diagnostic-only here, never asserted) on
//! `Ternary-Bonsai-1.7B.gguf`, Reference vs Metal, 64 self-generated steps,
//! were 5.754e-3 / 5.813e-3 / 1.567e-2 across the three prompt slots — ALL
//! THREE exceed a naive per-element floor of 5e-3 (since `max(1, |ref_i|) >=
//! 1` for every `i`, a literal per-element bound is exactly 5e-3 there). The
//! per-step form is the one that both reproduces the worked example
//! (`5e-3 * 22.68 = 0.113`) and passes on measured real-model data. Framed
//! differently: the Metal leg's own numeric tolerance is up to ~5.6x looser,
//! at the worst observed point, than the flat 5e-3 absolute bound it
//! replaced — a real loosening, made explicit here rather than left
//! implicit in a constant's value.
//!
//! `MAX_PER_STEP_LOGIT_DELTA` (5e-3) is used as the floor rather than a
//! smaller `1e-4`: the two describe the SAME predicate
//! (`REL_TOL * max(1, ref_absmax) >= REL_TOL = 5e-3 > 1e-4` for every
//! `ref_absmax`, so a `1e-4` floor is subsumed by the relative term either
//! way) and the larger constant is the more conservative of the two to ship.

use std::fmt::{Debug, Display};

/// First-index argmax tie-break: the contract every greedy sampler in this
/// workspace follows (`oxibonsai_runtime::engine_greedy::argmax_first`,
/// mirrored here so a parity test can replay it without pulling in the
/// runtime crate's full `Sampler`).
#[must_use]
pub fn argmax_first(values: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    u32::try_from(best_i).unwrap_or(u32::MAX)
}

/// One tier's complete greedy run for one prompt: the token chain it
/// generated and every step's full logit vector. `T` is the caller's own
/// kernel-tier type (this crate never depends on `oxibonsai-kernels`).
pub struct TierRun<T> {
    /// The tier this run was pinned to.
    pub tier: T,
    /// The greedy token ids this tier produced, one per decode step.
    pub tokens: Vec<u32>,
    /// Every decode step's full logit vector, in the same order as `tokens`.
    pub logits: Vec<Vec<f32>>,
}

/// The winner and runner-up of one logit vector (first-index tie-break,
/// matching [`argmax_first`]).
#[derive(Debug, Clone, Copy)]
pub struct TopTwo {
    idx: usize,
    top: f32,
    second: f32,
}

impl TopTwo {
    /// The top-1/top-2 decision margin.
    #[must_use]
    pub fn gap(self) -> f32 {
        self.top - self.second
    }

    /// Index of the winner (first-index tie-break, as [`argmax_first`]).
    #[must_use]
    pub fn index(self) -> usize {
        self.idx
    }

    /// The winning value.
    #[must_use]
    pub fn top(self) -> f32 {
        self.top
    }

    /// The runner-up value.
    #[must_use]
    pub fn second(self) -> f32 {
        self.second
    }
}

/// The winner and runner-up of `v` (first-index tie-break).
#[must_use]
pub fn top_two(v: &[f32]) -> TopTwo {
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

/// The `k` largest entries of `v` as `(index, value)`, largest first — one
/// side of "print each side's top-k and argmax at the breaching step".
#[must_use]
pub fn top_k(v: &[f32], k: usize) -> String {
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

/// Lower bound on what counts as a per-step breach for a non-CPU (GPU)
/// pair — **not** an additional ceiling (see [`per_step_bound`]). It exists
/// so a fixture whose logits all sit near zero cannot turn every float
/// wobble into a failure.
pub const MAX_PER_STEP_LOGIT_DELTA: f32 = 5e-3;

/// Relative per-step tolerance for a non-CPU (GPU) pair.
///
/// 4x headroom over the worst RELATIVE cross-backend deviation measured
/// anywhere on the real models — measured 2026-09-22, 64 self-generated
/// greedy steps per slot, `max|Δ| / max(1, max|reference|)`:
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
/// `Bonsai-8B.gguf` cleared even the flat absolute bound this replaced
/// outright. An ABSOLUTE bound cannot distinguish a 6e-3 deviation on a
/// logit of 26 from one on a logit of 1, which is why this bound is
/// relative instead of larger.
pub const REL_TOL: f32 = 5e-3;

/// Context for the near-tie numbers printed inside a token-divergence
/// panic: an earlier classifier tolerated an argmax flip whose
/// reference-side top-to-runner-up gap was below `max(1e-2, 32 * delta)`.
/// These two constants gate nothing here; they exist so a failure report can
/// still say whether a flip *looked* like a near-tie.
pub const NEAR_TIE_ABS_FLOOR: f32 = 1e-2;
/// See [`NEAR_TIE_ABS_FLOOR`].
pub const NEAR_TIE_REL_MULT: f32 = 32.0;

/// The per-step bound for a non-CPU (GPU) pair, written out literally
/// because the combinator is the one thing an implementer can invert while
/// believing they followed the spec: it is `max`, **not** `&&`.
///
/// [`MAX_PER_STEP_LOGIT_DELTA`] is a FLOOR on what counts as a breach, never
/// a second ceiling. A step with `ref_absmax = 22.68` and `delta = 2.806e-2`
/// is GREEN here (bound = 0.113) — that is one of the six measured points in
/// [`REL_TOL`]'s table, so an `&&` formulation would fail a case measured to
/// be correct.
///
/// See this module's own doc comment ("`per_step_bound` is a per-STEP
/// predicate, not per-element") for why the bound is computed over each
/// step's own maxima rather than per logit element.
///
/// The design text this predicate implements writes the floor as
/// `FLOOR = 1e-4`; this file's own constant is [`MAX_PER_STEP_LOGIT_DELTA`]
/// (5e-3). The two describe the SAME predicate:
/// `REL_TOL * max(1, ref_absmax) >= REL_TOL = 5e-3 > 1e-4` for every
/// possible `ref_absmax`, so the smaller floor is subsumed by the relative
/// term either way — [`MAX_PER_STEP_LOGIT_DELTA`] is simply the larger,
/// more conservative of the two equivalent constants, kept as the one this
/// function actually uses.
#[must_use]
pub fn per_step_bound(ref_absmax: f32) -> f32 {
    f32::max(
        MAX_PER_STEP_LOGIT_DELTA,
        REL_TOL * f32::max(1.0, ref_absmax),
    )
}

/// Everything one step of one tier pair measured. Collected for every step
/// BEFORE anything is asserted, so a failure report can print the whole
/// series rather than only the first breaching step.
pub struct StepStats {
    /// `max_i |reference_i - other_i|`.
    pub delta: f32,
    /// The index that attained `delta`.
    pub delta_at: usize,
    /// `max_i |reference_i|` — the scale the relative bound is relative to.
    pub ref_absmax: f32,
    /// Diagnostic only, never asserted (see [`per_step_bound`]'s module
    /// doc): `max_i |Δ_i| / max(1, |reference_i|)`.
    pub worst_elem_rel: f32,
    /// How many elements are not bit-identical (the CPU-pair assertion).
    pub differing: usize,
    /// Non-finite logits on either side. Always a hard failure: `f32::max`
    /// silently swallows a NaN operand, so a NaN could otherwise hide inside
    /// `delta` without ever tripping a `<=` bound.
    pub non_finite: usize,
    /// The reference side's top-2.
    pub ref_top: TopTwo,
    /// The other side's top-2.
    pub other_top: TopTwo,
}

impl StepStats {
    /// Measure one step's `reference` logits against `other`'s.
    #[must_use]
    pub fn measure(reference: &[f32], other: &[f32]) -> Self {
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

    /// `delta / max(1, ref_absmax)` — the quantity [`REL_TOL`]'s table
    /// lists.
    #[must_use]
    pub fn relative(&self) -> f32 {
        self.delta / f32::max(1.0, self.ref_absmax)
    }
}

/// The full per-step delta series: one line per step, so the SHAPE of a
/// divergence over the chain is visible at a glance instead of being
/// inferred from a single number.
#[must_use]
pub fn delta_series(stats: &[StepStats]) -> String {
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
/// both sides' argmax, top-2 gap and top-5, plus the near-tie context an
/// earlier classifier used to decide with.
#[must_use]
pub fn step_detail(step: usize, s: &StepStats, reference: &[f32], other: &[f32]) -> String {
    let near_tie_bound = NEAR_TIE_ABS_FLOOR.max(NEAR_TIE_REL_MULT * s.delta);
    format!(
        "    step {step}: max|Δ|={:e} at index {} | max|ref|={:.6} | relative={:e} | bound={:e} \
         | differing elements={} of {} | non-finite={}\n      \
         reference: argmax={} top={:.6} second={:.6} top-2 gap={:e}\n      \
         other:     argmax={} top={:.6} second={:.6} top-2 gap={:e}\n      \
         reference top-5: {}\n      \
         other     top-5: {}\n      \
         near-tie context (diagnostic only, not asserted): reference gap {:e} vs \
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

/// Compare one tier's whole run against the reference tier's, in the
/// priority order this module's doc comment states: token chain first
/// (unconditional hard gate), then bit-exact for a CPU-tier pair or the
/// relative bound for an accelerator pair.
///
/// `is_cpu_tier` classifies `other.tier`: only the caller knows which of its
/// own tier values are CPU tiers (this crate has no `oxibonsai-kernels`
/// dependency to call `KernelTier` by name), so it is passed in rather than
/// hardcoded here.
///
/// # Panics
///
/// Panics with a detailed, multi-line report on the first violation this
/// function's own priority order reaches: a token-chain divergence, a
/// non-bit-exact CPU pair, a non-finite logit, or a relative-bound breach.
pub fn compare_against_reference<T: Copy + Debug + Display>(
    context: &str,
    reference: &TierRun<T>,
    other: &TierRun<T>,
    is_cpu_tier: impl Fn(T) -> bool,
) {
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
    if other.tokens != reference.tokens {
        let first = match reference
            .tokens
            .iter()
            .zip(other.tokens.iter())
            .position(|(a, b)| a != b)
        {
            Some(i) => i,
            None => panic!(
                "{pair}: token chains differ only in LENGTH ({} vs {} tokens) -- every \
                 shared-length position matches\n  reference chain: {:?}\n  other  chain: {:?}",
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
            "{pair}: GREEDY TOKEN CHAINS DIVERGED (the hard gate — the product promises \
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

    // ── (2) numeric: bit-exact for a CPU pair, relative bound otherwise ───
    if is_cpu_tier(other.tier) {
        for (step, s) in stats.iter().enumerate() {
            assert!(
                s.differing == 0,
                "{pair}: CPU-TIER LOGITS ARE NOT BIT-EXACT at step {step} (measured exactly 0.0 \
                 over 64 steps x 3 slots on Ternary-Bonsai-1.7B.gguf — a failure here is a REAL \
                 FINDING to report, not a bound to loosen)\n{}\n  per-step delta series:\n{}",
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
                "{pair}: max|Δlogit|={:e} at step {step} exceeds the relative bound {:e} = \
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

/// Serializes real-model gates that share process-global state (e.g. a
/// single Metal decode session) inside one test binary. `cargo test`'s
/// default in-process thread pool can otherwise run two such gates
/// concurrently and have them observe each other's half-finished GPU state.
///
/// This is an in-binary guard only, and deliberately a plain function (not
/// generic): each test binary that calls it links its own copy of the
/// `static` inside, exactly as if the function were still defined locally —
/// sharing the source does not share the lock across the separate OS
/// processes `cargo nextest` runs each test binary as.
pub fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What [`run_named_test_in_child`] observed.
#[derive(Debug)]
pub struct ChildRun {
    /// The child's exit status, or `None` when it was still running at the
    /// deadline and was killed — a different finding from a child that
    /// exited non-zero on its own, and reported differently by every caller.
    pub status: Option<std::process::ExitStatus>,
    /// Everything the child wrote to stdout and stderr, interleaved.
    pub output: String,
}

impl ChildRun {
    /// Whether the child exited on its own with status 0.
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.status.is_some_and(|status| status.success())
    }
}

/// A path under `std::env::temp_dir()` for one child's combined
/// stdout/stderr log, unique across every call in this process and across
/// processes.
///
/// The name carries the parent's process id, a process-wide atomic counter
/// and a wall-clock timestamp (see [`crate::temp_path::unique_path`]). The
/// counter is what makes it collision-free: two threads of one process that
/// start children for the *same* `test_name` share the process id and the
/// sanitised test-name tag, and a timestamp alone cannot tell them apart
/// when the platform clock ticks in microseconds. Two children writing one
/// log truncate each other's output, and whichever parent finishes first
/// removes the file the other is still reading.
fn child_log_path(test_name: &str) -> std::path::PathBuf {
    let tag: String = test_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    crate::temp_path::unique_path(&format!("child-{tag}"), ".log")
}

/// Removes the child's log file when dropped, so every early return of
/// [`run_named_test_in_child`] (a failed clone, spawn or poll) cleans up too.
struct RemoveOnDrop(std::path::PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        // Best effort: a leftover log in the temp dir is harmless.
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Re-execute the CURRENT test binary so it runs exactly one test,
/// `test_name` (`<test_name> --exact --nocapture`), in a fresh process with
/// `envs` added to its environment, and wait at most `deadline` for it —
/// killing it when the deadline passes.
///
/// `test_name` is the name libtest lists the test under, which is its path
/// from the test crate's root (`module::test_fn`), because `--exact` matches
/// the whole path: a bare function name selects nothing when the test lives
/// in a module, and the child then exits 0 having run no test.
///
/// The real-model gates compare two configurations of the Metal decode path
/// this way, one fresh process per arm, because that path is a
/// process-global singleton (one device KV cache per process): two arms in
/// one process would share it.
///
/// The child's stdout and stderr go to one file under
/// `std::env::temp_dir()` rather than a pipe: a pipe can deadlock once the
/// child fills the OS buffer before the parent reads it, and a killed
/// child's output must still be readable afterwards. The file is created
/// exclusively at a path no other call can hold (process id, a process-wide
/// counter and a timestamp are all in its name) and is removed once read.
/// The parent's environment is never modified: `envs` reach only the child.
///
/// # Errors
///
/// Any I/O error locating the test binary, creating the output file, or
/// spawning, polling or reaping the child. A child still running when
/// polling fails is killed before the error is returned.
pub fn run_named_test_in_child(
    test_name: &str,
    envs: &[(&str, &str)],
    deadline: std::time::Duration,
) -> std::io::Result<ChildRun> {
    let exe = std::env::current_exe()?;
    let log_path = child_log_path(test_name);
    let log = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&log_path)?;
    let _remove_log = RemoveOnDrop(log_path.clone());
    let log_for_stderr = log.try_clone()?;

    let mut command = std::process::Command::new(&exe);
    command
        .args([test_name, "--exact", "--nocapture"])
        .stdout(log)
        .stderr(log_for_stderr);
    for (key, value) in envs {
        command.env(key, value);
    }
    let mut child = command.spawn()?;

    let give_up_at = std::time::Instant::now() + deadline;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(e) => {
                // Best effort: the poll error is what gets reported.
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        }
        if std::time::Instant::now() >= give_up_at {
            // Best effort: the child is being abandoned either way, and the
            // caller reports `status: None` as a timeout.
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    };

    let output = std::fs::read(&log_path)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default();
    Ok(ChildRun { status, output })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_first_breaks_ties_leftmost() {
        assert_eq!(argmax_first(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_first(&[5.0]), 0);
    }

    #[test]
    fn top_two_and_top_k_agree_on_the_winner() {
        let v = [1.0, 9.0, 4.0, 9.5, 2.0];
        let t2 = top_two(&v);
        assert!((t2.gap() - 0.5).abs() < 1e-6);
        assert!(top_k(&v, 2).contains("(3,"));
    }

    #[test]
    fn per_step_bound_is_the_floor_when_reference_is_small() {
        assert!((per_step_bound(0.1) - MAX_PER_STEP_LOGIT_DELTA).abs() < 1e-9);
    }

    #[test]
    fn per_step_bound_scales_with_reference_magnitude() {
        let bound = per_step_bound(22.68);
        assert!((bound - REL_TOL * 22.68).abs() < 1e-4);
        assert!(bound > MAX_PER_STEP_LOGIT_DELTA);
    }

    #[test]
    fn step_stats_measure_reports_zero_delta_for_identical_vectors() {
        let v = [1.0f32, 2.0, 3.0];
        let s = StepStats::measure(&v, &v);
        assert_eq!(s.delta, 0.0);
        assert_eq!(s.differing, 0);
        assert_eq!(s.non_finite, 0);
    }

    #[test]
    fn step_stats_measure_flags_non_finite_logits() {
        let s = StepStats::measure(&[1.0, f32::NAN], &[1.0, 2.0]);
        assert_eq!(s.non_finite, 1);
    }

    #[test]
    fn compare_against_reference_accepts_identical_bit_exact_runs() {
        let reference = TierRun {
            tier: "reference",
            tokens: vec![1, 2, 3],
            logits: vec![vec![0.1, 0.2], vec![0.3, 0.1], vec![0.2, 0.4]],
        };
        let other = TierRun {
            tier: "cpu",
            tokens: vec![1, 2, 3],
            logits: reference.logits.clone(),
        };
        compare_against_reference("test", &reference, &other, |t| t == "cpu");
    }

    #[test]
    #[should_panic(expected = "GREEDY TOKEN CHAINS DIVERGED")]
    fn compare_against_reference_rejects_a_token_chain_divergence() {
        let reference = TierRun {
            tier: "reference",
            tokens: vec![1, 2, 3],
            logits: vec![vec![0.1, 0.9], vec![0.3, 0.1], vec![0.2, 0.4]],
        };
        let other = TierRun {
            tier: "gpu",
            tokens: vec![1, 5, 3],
            logits: reference.logits.clone(),
        };
        compare_against_reference("test", &reference, &other, |t| t == "cpu");
    }

    #[test]
    #[should_panic(expected = "NOT BIT-EXACT")]
    fn compare_against_reference_rejects_a_non_bit_exact_cpu_pair() {
        let reference = TierRun {
            tier: "reference",
            tokens: vec![1],
            logits: vec![vec![0.1, 0.2]],
        };
        let other = TierRun {
            tier: "cpu",
            tokens: vec![1],
            logits: vec![vec![0.1, 0.200_000_2]],
        };
        compare_against_reference("test", &reference, &other, |t| t == "cpu");
    }

    #[test]
    #[should_panic(expected = "exceeds the relative bound")]
    fn compare_against_reference_rejects_a_gpu_pair_past_the_relative_bound() {
        let reference = TierRun {
            tier: "reference",
            tokens: vec![1],
            logits: vec![vec![0.1, 20.0]],
        };
        let other = TierRun {
            tier: "gpu",
            tokens: vec![1],
            logits: vec![vec![0.1, 20.5]],
        };
        compare_against_reference("test", &reference, &other, |t| t == "cpu");
    }

    #[test]
    fn gpu_serial_can_be_acquired_and_released() {
        let guard = gpu_serial();
        drop(guard);
        let _guard2 = gpu_serial();
    }

    #[test]
    fn top_two_accessors_report_winner_and_runner_up() {
        let t2 = top_two(&[1.0, 9.0, 4.0, 9.5, 2.0]);
        assert_eq!(t2.index(), 3);
        assert_eq!(t2.top(), 9.5);
        assert_eq!(t2.second(), 9.0);
    }

    /// Set by [`run_named_test_in_child`]'s own tests to tell
    /// [`child_runner_probe`] what to do in the re-executed child.
    const CHILD_PROBE_ENV: &str = "OXIBONSAI_TESTKIT_CHILD_PROBE";

    /// The child side of the runner tests: a no-op in a normal run, and in a
    /// re-executed child it prints a marker, sleeps past a deadline, or
    /// fails, as asked.
    #[test]
    fn child_runner_probe() {
        match std::env::var(CHILD_PROBE_ENV).as_deref() {
            Ok("print") => println!("CHILD_PROBE_MARKER pid={}", std::process::id()),
            Ok("sleep") => std::thread::sleep(std::time::Duration::from_secs(30)),
            Ok("fail") => panic!("CHILD_PROBE_FAILURE requested"),
            _ => {}
        }
    }

    /// The libtest name of [`child_runner_probe`], as `--exact` needs it.
    const PROBE_TEST: &str = "parity::tests::child_runner_probe";

    /// Serialises every test that starts a child through
    /// [`run_named_test_in_child`], so two of them never overlap under
    /// `cargo test`'s in-process thread pool. Each one re-executes this test
    /// binary and waits on it, so overlapping them only multiplies the load
    /// on a busy host and adds nothing a single child does not exercise; the
    /// one test that needs two children at once ([`concurrent_children_keep_separate_logs`])
    /// starts them itself, under this same lock.
    static CHILD_RUNNER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn child_runner_serial() -> std::sync::MutexGuard<'static, ()> {
        CHILD_RUNNER_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn run_named_test_in_child_captures_a_successful_child() {
        let _serial = child_runner_serial();
        let run = run_named_test_in_child(
            PROBE_TEST,
            &[(CHILD_PROBE_ENV, "print")],
            std::time::Duration::from_secs(120),
        )
        .unwrap_or_else(|e| panic!("spawning the probe child: {e}"));
        assert!(run.succeeded(), "probe child failed: {run:?}");
        assert!(
            run.output.contains("CHILD_PROBE_MARKER"),
            "probe child's output was not captured: {:?}",
            run.output
        );
    }

    /// The log path is unique per call even when many threads ask for the
    /// same `test_name` in the same instant: it carries a process-wide
    /// counter, not only the process id and a timestamp (the collision that
    /// let two concurrent children share one log file). None of the paths
    /// exists yet — the runner creates it exclusively — and all of them live
    /// under the OS temp directory.
    #[test]
    fn child_log_paths_are_unique_across_threads_for_one_test_name() {
        const THREADS: usize = 8;
        const PATHS_PER_THREAD: usize = 512;
        let all: Vec<std::path::PathBuf> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        (0..PATHS_PER_THREAD)
                            .map(|_| child_log_path(PROBE_TEST))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect()
        });
        assert_eq!(all.len(), THREADS * PATHS_PER_THREAD);
        let distinct: std::collections::HashSet<&std::path::PathBuf> = all.iter().collect();
        assert_eq!(
            distinct.len(),
            all.len(),
            "child_log_path handed the same path to two calls"
        );
        assert!(
            all.iter()
                .all(|p| p.starts_with(std::env::temp_dir()) && !p.exists()),
            "every child log path must be a fresh path under the OS temp directory"
        );
    }

    /// Two children started at the same moment for the same test name each
    /// get their own log: both parents read back their own child's marker
    /// (each child prints its own pid, so a shared or clobbered log shows up
    /// as a missing or duplicated marker).
    #[test]
    fn concurrent_children_keep_separate_logs() {
        let _serial = child_runner_serial();
        let outputs: Vec<ChildRun> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        run_named_test_in_child(
                            PROBE_TEST,
                            &[(CHILD_PROBE_ENV, "print")],
                            std::time::Duration::from_secs(120),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| panic!("a runner thread panicked"))
                        .unwrap_or_else(|e| panic!("spawning a probe child: {e}"))
                })
                .collect()
        });
        let mut pids = Vec::new();
        for run in &outputs {
            assert!(run.succeeded(), "probe child failed: {run:?}");
            let markers: Vec<&str> = run
                .output
                .lines()
                .filter(|l| l.starts_with("CHILD_PROBE_MARKER pid="))
                .collect();
            assert_eq!(
                markers.len(),
                1,
                "each child's log must hold exactly its own marker: {:?}",
                run.output
            );
            pids.push(markers[0].to_string());
        }
        assert_ne!(
            pids[0], pids[1],
            "the two runs read the same child's marker: the logs were shared"
        );
    }

    /// A finished run leaves no log behind in the temp directory.
    #[test]
    fn run_named_test_in_child_removes_its_log() {
        let _serial = child_runner_serial();
        let prefix = format!(
            "oxibonsai-testkit-child-{}-{}-",
            PROBE_TEST.replace([':', '_'], "_"),
            std::process::id()
        );
        let leftovers = || -> Vec<String> {
            std::fs::read_dir(std::env::temp_dir())
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .filter(|name| name.starts_with(&prefix))
                        .collect()
                })
                .unwrap_or_default()
        };
        assert!(leftovers().is_empty(), "stale logs before the run");
        let run = run_named_test_in_child(
            PROBE_TEST,
            &[(CHILD_PROBE_ENV, "print")],
            std::time::Duration::from_secs(120),
        )
        .unwrap_or_else(|e| panic!("spawning the probe child: {e}"));
        assert!(run.succeeded(), "probe child failed: {run:?}");
        assert_eq!(leftovers(), Vec::<String>::new(), "the log was not removed");
    }

    #[test]
    fn run_named_test_in_child_reports_failure_and_timeout_distinctly() {
        let _serial = child_runner_serial();
        let failed = run_named_test_in_child(
            PROBE_TEST,
            &[(CHILD_PROBE_ENV, "fail")],
            std::time::Duration::from_secs(120),
        )
        .unwrap_or_else(|e| panic!("spawning the probe child: {e}"));
        assert!(
            failed.status.is_some() && !failed.succeeded(),
            "a panicking child must report Some(non-success), got {failed:?}"
        );
        assert!(failed.output.contains("CHILD_PROBE_FAILURE"));

        let timed_out = run_named_test_in_child(
            PROBE_TEST,
            &[(CHILD_PROBE_ENV, "sleep")],
            std::time::Duration::from_secs(1),
        )
        .unwrap_or_else(|e| panic!("spawning the probe child: {e}"));
        assert!(
            timed_out.status.is_none(),
            "a child past its deadline must report None, got {timed_out:?}"
        );
    }
}
