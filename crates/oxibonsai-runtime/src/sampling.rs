//! Sampling strategies for text generation.
//!
//! Supports temperature scaling, top-k filtering, top-p (nucleus) filtering,
//! and — via [`Sampler::sample_with_history`] — repetition, frequency, and
//! presence penalties applied over the generated-token history. The base
//! [`Sampler::sample`] converts a logit vector into a single token ID using
//! these strategies in order:
//!
//! 1. **Temperature scaling** — divide logits by temperature (0 = greedy argmax)
//! 2. **Top-k** — keep only the k highest-probability candidates
//! 3. **Softmax** — convert scaled logits to probabilities
//! 4. **Top-p** — keep the smallest set of tokens whose cumulative probability exceeds p
//! 5. **Weighted random selection** — sample from the filtered distribution
//!
//! [`Sampler::sample_with_history`] additionally applies the repetition penalty
//! (from [`SamplingParams::repetition_penalty`]) and the frequency / presence
//! penalties (from [`PenaltyParams`]) to the logit vector *before* step 1, so
//! the penalties influence both stochastic sampling and greedy (argmax)
//! decoding. When no penalty is active it is bit-identical to
//! [`Sampler::sample`].

use std::cmp::Ordering;

use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling_advanced::apply_repetition_penalty;

/// Sampling parameters.
///
/// **Do not add fields to this struct without auditing every
/// `SamplingParams { .. }` struct-literal construction workspace-wide.**
/// Several construct it field-by-field without `..SamplingParams::default()`
/// (e.g. `engine_greedy.rs`, `engine_control.rs`, `builders.rs` — verified by
/// a full `cargo check --workspace --all-features --all-targets` while
/// developing this fix, which failed at `builders.rs:105` with `missing
/// field min_p` the moment a field was added here), so a new field is a
/// breaking change to files this package does not own. [`Sampler`] carries
/// [`Sampler::min_p`]/[`Sampler::set_min_p`] as a *sampler-level* setting
/// instead, precisely to add min-p support without this hazard — see its
/// doc comment.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax scaling. 0.0 = greedy.
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) threshold (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Maximum number of new tokens to generate per request.
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            // `RT-24`/gatekeeper-REQUIRED#1(a): this used to be `1.1`, which
            // meant EVERY caller that builds a per-request `SamplingParams`
            // via `..SamplingParams::default()` (the OpenAI server's chat,
            // completions and extended handlers, `async_engine.rs`,
            // `server/blocking.rs`, `prefix_cache_engine.rs`) silently
            // applied a repetition penalty even for a `temperature: 0`
            // ("greedy") request, and `InferenceEngine::greedy_gpu_eligible`
            // requires `repetition_penalty == 1.0` to take the fused
            // GPU-argmax fast path — so a plain greedy request never used
            // the GPU route and never matched the CPU/Metal parity goldens.
            // `1.0` (no-op) is the correct default for a field the caller
            // must opt into explicitly, matching every other penalty in
            // this struct (`top_p: 1.0`-disabled is the same convention,
            // and `PenaltyParams::default()` is all-zero/disabled).
            repetition_penalty: 1.0,
            max_tokens: 128,
        }
    }
}

/// OpenAI-style frequency and presence penalties applied over the
/// generated-token history before sampling.
///
/// For each token id that has been generated `c > 0` times, the penalty
/// subtracted from that token's logit is:
///
/// ```text
/// presence_penalty * I(c > 0) + frequency_penalty * c
/// ```
///
/// Both default to `0.0` (disabled). The repetition penalty is configured
/// separately via [`SamplingParams::repetition_penalty`]; together with this
/// struct it forms the complete penalty seam consumed by
/// [`Sampler::sample_with_history`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PenaltyParams {
    /// Subtracted once from the logit of every token seen at least once
    /// (OpenAI `presence_penalty`). `0.0` disables it.
    pub presence_penalty: f32,
    /// Subtracted proportional to a token's generated count
    /// (OpenAI `frequency_penalty`). `0.0` disables it.
    pub frequency_penalty: f32,
}

impl PenaltyParams {
    /// Construct penalty parameters from OpenAI `frequency_penalty` /
    /// `presence_penalty` values.
    pub fn new(frequency_penalty: f32, presence_penalty: f32) -> Self {
        Self {
            presence_penalty,
            frequency_penalty,
        }
    }

    /// `true` when at least one of the two penalties is non-zero.
    pub fn is_active(&self) -> bool {
        self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }
}

/// Apply OpenAI-style frequency and presence penalties to `logits` in place.
///
/// A count histogram is built over `recent_tokens`; for each token id present
/// with count `c`, `presence_penalty + frequency_penalty * c` is subtracted
/// from its logit. This is a no-op when both penalties are `0.0` or
/// `recent_tokens` is empty.
///
/// The result is independent of histogram iteration order — each distinct
/// token id maps to a distinct logit index that is decremented exactly once —
/// so the operation is deterministic.
pub fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    recent_tokens: &[u32],
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    if (frequency_penalty == 0.0 && presence_penalty == 0.0) || recent_tokens.is_empty() {
        return;
    }
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &id in recent_tokens {
        *counts.entry(id).or_insert(0) += 1;
    }
    for (id, count) in counts {
        let idx = id as usize;
        if idx < logits.len() {
            logits[idx] -= presence_penalty + frequency_penalty * count as f32;
        }
    }
}

/// Token sampler.
///
/// Owns a reusable `probs_buf` that is grown on first use and then reused across
/// all subsequent `sample()` calls, eliminating the ~1.8 MB per-call heap
/// allocation that a fresh `Vec` would require for a 151 936-token vocabulary.
#[derive(Debug)]
pub struct Sampler {
    params: SamplingParams,
    /// Frequency / presence penalties applied by
    /// [`Sampler::sample_with_history`]. Default-off, so the base
    /// [`Sampler::sample`] path is unaffected.
    penalties: PenaltyParams,
    /// Min-p (probabilistic nucleus) threshold (RT-23), `0.0` = disabled.
    ///
    /// Deliberately **not** a field on [`SamplingParams`] — see that
    /// struct's doc comment. Set via [`Sampler::set_min_p`], read via
    /// [`Sampler::min_p`]; applied in [`Sampler::sample_core`] after top-k,
    /// before top-p, matching llama.cpp / vLLM ordering.
    min_p: f32,
    rng_state: u64,
    /// Reusable working buffer for `(token_index, scaled_logit)` pairs.
    ///
    /// After `select_nth_unstable_by` + `drain` the buffer holds only the top-k
    /// candidates (capacity stays at `vocab_size`).  `clear()` on the next call
    /// resets length to zero without freeing the backing store, so subsequent
    /// `extend()` calls never reallocate.
    probs_buf: Vec<(usize, f32)>,
    /// Reusable scratch copy of the raw logits used by
    /// [`Sampler::sample_with_history`] when a penalty is active. Grown on
    /// first penalised call and reused thereafter (never allocated on the
    /// no-penalty fast path).
    penalty_buf: Vec<f32>,
}

/// Mixes a raw `seed` through SplitMix64 to obtain xorshift64's initial
/// state (RT-25).
///
/// The previous implementation used `seed` directly as `rng_state`; `0` is
/// xorshift64's only fixed point, so a `0` seed made [`Sampler::next_u64`]
/// return `0` forever, collapsing the weighted-selection step to always pick
/// whichever candidate the cumulative sum reaches first (`rand_val` fixed at
/// `0.0`). This is a live, user-reachable path — `PipelineBuilder::new`
/// documents `seed: 0` as its default — not a hypothetical.
///
/// SplitMix64 is a bijection on `u64`, so it already maps `seed == 0` (and
/// every other input) to a well-mixed, effectively-arbitrary non-zero value;
/// the explicit zero-check below only matters for the one specific `seed`
/// (out of 2⁶⁴) whose SplitMix64 image is exactly `0` — substituting a fixed
/// non-zero constant there makes "the resulting state is never `0`" hold
/// unconditionally, for every possible `u64` input, rather than merely being
/// astronomically likely.
fn seed_to_rng_state(seed: u64) -> u64 {
    // Standard SplitMix64 (as used by e.g. `java.util.SplittableRandom`).
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        z
    }
}

impl Sampler {
    /// Create a new sampler with the given parameters and seed.
    ///
    /// `seed` is mixed through [`seed_to_rng_state`] (SplitMix64, RT-25)
    /// before becoming the xorshift64 PRNG's initial state, so every `u64`
    /// seed — including `0`, which callers use routinely (e.g.
    /// `PipelineBuilder`'s documented default) — produces a well-distributed,
    /// non-degenerate stream. This is the *only* place `rng_state` is ever
    /// set, so every seeding path (including
    /// [`crate::engine::InferenceEngine::generate_with_seed`], which builds a
    /// fresh `Sampler` per call) goes through the fix automatically.
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            penalties: PenaltyParams::default(),
            min_p: 0.0,
            rng_state: seed_to_rng_state(seed),
            probs_buf: Vec::new(),
            penalty_buf: Vec::new(),
        }
    }

    /// Simple xorshift64 PRNG — no external dependency needed.
    ///
    /// Its only fixed point is state `0`; [`seed_to_rng_state`] guarantees
    /// `rng_state` is never `0` at construction, and xorshift64's update is a
    /// bijection on `u64`, so a non-zero state can never reach `0` on any
    /// later call either (RT-25).
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Sample a token index from logits.
    ///
    /// This is the base sampling path (temperature → top-k → softmax → top-p →
    /// weighted select). It applies **no** penalties; use
    /// [`Sampler::sample_with_history`] to apply the repetition / frequency /
    /// presence penalties over a token history.
    #[tracing::instrument(skip(self, logits), fields(vocab_size = logits.len()), level = "debug")]
    pub fn sample(&mut self, logits: &[f32]) -> RuntimeResult<u32> {
        self.sample_core(logits)
    }

    /// Sample a token, first applying the repetition penalty (from
    /// [`SamplingParams::repetition_penalty`]) and any configured frequency /
    /// presence penalties ([`PenaltyParams`]) over `recent_tokens`.
    ///
    /// `recent_tokens` is the sequence of previously-generated token ids. The
    /// penalties are applied to a scratch copy of `logits`, so the caller's
    /// slice is never mutated. Penalties are applied before temperature /
    /// top-k / top-p, so they affect both stochastic and greedy (argmax)
    /// decoding, matching OpenAI semantics.
    ///
    /// When no penalty is active (repetition penalty `== 1.0` and both
    /// frequency and presence penalties `== 0.0`) this delegates directly to
    /// [`Sampler::sample`] with no copy and identical RNG consumption, so the
    /// default no-penalty path stays bit-identical.
    #[tracing::instrument(
        skip(self, logits, recent_tokens),
        fields(vocab_size = logits.len(), history = recent_tokens.len()),
        level = "debug"
    )]
    pub fn sample_with_history(
        &mut self,
        logits: &[f32],
        recent_tokens: &[u32],
    ) -> RuntimeResult<u32> {
        let rep = self.params.repetition_penalty;
        let rep_active = rep != 1.0;
        if !rep_active && !self.penalties.is_active() {
            // Fast path: no penalties → identical to `sample`.
            return self.sample_core(logits);
        }

        // Move the reusable scratch buffer out of `self` so the subsequent
        // `sample_core(&buf)` immutable borrow does not clash with the
        // `&mut self` it needs for `probs_buf` / RNG.
        let mut buf = std::mem::take(&mut self.penalty_buf);
        buf.clear();
        buf.extend_from_slice(logits);
        if rep_active {
            apply_repetition_penalty(&mut buf, recent_tokens, rep);
        }
        apply_frequency_presence_penalty(
            &mut buf,
            recent_tokens,
            self.penalties.frequency_penalty,
            self.penalties.presence_penalty,
        );
        let result = self.sample_core(&buf);
        // Return the buffer for reuse on the next penalised call.
        self.penalty_buf = buf;
        result
    }

    /// Core sampling implementation shared by [`Sampler::sample`] and
    /// [`Sampler::sample_with_history`]. Operates on an already
    /// penalty-adjusted (or raw) logit slice.
    fn sample_core(&mut self, logits: &[f32]) -> RuntimeResult<u32> {
        if logits.is_empty() {
            return Ok(0);
        }

        // Greedy if temperature is ~0
        if self.params.temperature < 1e-6 {
            return Ok(argmax(logits) as u32);
        }

        // Populate the reusable buffer with temperature-scaled logits.
        // On the first call this allocates `vocab_size × 12` bytes; every
        // subsequent call reuses the existing backing store (len is reset to 0
        // by `clear()`, capacity is preserved from the previous call).
        self.probs_buf.clear();
        self.probs_buf.extend(
            logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v / self.params.temperature)),
        );

        // Top-k filtering — O(n) average via partial selection rather than O(n log n) full sort.
        // `select_nth_unstable_by` rearranges `probs_buf` so that element at index `cutoff` is in
        // its fully-sorted position, all elements before it are ≤ it (lower scaled logits), and all
        // elements after it are ≥ it (higher scaled logits).  Draining the prefix leaves exactly
        // the top-k elements in arbitrary order, which is sufficient for softmax + sampling.
        if self.params.top_k > 0 && self.params.top_k < self.probs_buf.len() {
            let k = self.params.top_k;
            let cutoff = self.probs_buf.len() - k;
            self.probs_buf.select_nth_unstable_by(cutoff, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            });
            self.probs_buf.drain(..cutoff);
        }

        // Softmax
        let max_val = self
            .probs_buf
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (_, v) in self.probs_buf.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        // RT-21: with finite logits, `sum` is always >= 1.0 — the element
        // equal to `max_val` always contributes exactly `exp(0) == 1`. A
        // non-finite or non-positive `sum` therefore only arises from a
        // genuinely degenerate candidate set: every logit masked to `-inf`
        // (nothing left to sample), or a `NaN` reaching the sampler (e.g.
        // from an unstable forward pass). Dividing by such a `sum` would
        // poison *every* probability with `NaN` below — including
        // candidates whose own logit was perfectly valid — so the final
        // fallback would then return an arbitrary token instead of the
        // actual best one. Recover deterministically instead: argmax over
        // the untouched input (which safely ignores `NaN`/`-inf` entries;
        // see `argmax`'s doc comment), matching what greedy decoding would
        // have done, and warn once so the condition stays visible without
        // flooding the log on every affected token.
        if !sum.is_finite() || sum <= 0.0 {
            warn_zero_mass_once(logits.len(), sum);
            return Ok(argmax(logits) as u32);
        }

        for (_, v) in self.probs_buf.iter_mut() {
            *v /= sum;
        }

        // Min-p filtering (RT-23): inserted after top-k, before top-p, per
        // the established sampling order (penalties -> temperature -> top-k
        // -> min-p -> top-p) used by llama.cpp / vLLM. `min_p` lives on
        // `Sampler` itself, not `SamplingParams` — see that struct's doc
        // comment for why.
        if self.min_p > 0.0 {
            truncate_to_min_p(&mut self.probs_buf, self.min_p);
        }

        // Top-p (nucleus) filtering.
        if self.params.top_p < 1.0 {
            truncate_to_top_p(&mut self.probs_buf, self.params.top_p);
        }

        // Pre-compute random value before the immutable borrow of `probs_buf`
        // to satisfy the borrow checker: `next_u64` takes `&mut self` which
        // would conflict with an active `&self.probs_buf` borrow.
        let rand_val = (self.next_u64() as f64 / u64::MAX as f64) as f32;

        // Weighted random selection
        let mut cum = 0.0f32;
        for &(idx, p) in &self.probs_buf {
            cum += p;
            if rand_val <= cum {
                return Ok(idx as u32);
            }
        }

        // Fallback for the residual floating-point-rounding case where the
        // cumulative sum falls just short of `rand_val` on the very last
        // element.
        pick_highest_probability(&self.probs_buf)
    }

    /// Get current parameters.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Current frequency / presence penalty parameters.
    pub fn penalties(&self) -> &PenaltyParams {
        &self.penalties
    }

    /// Replace the frequency / presence penalty parameters in place, preserving
    /// the PRNG state and reusable buffers.
    ///
    /// These are consumed only by [`Sampler::sample_with_history`]; the base
    /// [`Sampler::sample`] path ignores them.
    pub fn set_penalties(&mut self, penalties: PenaltyParams) {
        self.penalties = penalties;
    }

    /// Replace the sampling parameters in place, preserving the PRNG state and
    /// the reusable `probs_buf` allocation.
    ///
    /// Unlike constructing a fresh [`Sampler`], this leaves `rng_state`
    /// untouched, so a caller can temporarily adjust (e.g.) the temperature for
    /// one request without perturbing the RNG sequence that subsequent requests
    /// on the same engine would observe.
    pub fn set_params(&mut self, params: SamplingParams) {
        self.params = params;
    }

    /// Current min-p (probabilistic nucleus) threshold (RT-23). `0.0` means
    /// disabled.
    pub fn min_p(&self) -> f32 {
        self.min_p
    }

    /// Replace the min-p threshold in place, preserving the PRNG state and
    /// every other setting.
    ///
    /// Values outside `[0.0, 1.0]` are clamped by [`truncate_to_min_p`] at
    /// sample time, not here, so this setter never fails; `0.0` (or any
    /// non-positive value) disables the filter.
    pub fn set_min_p(&mut self, min_p: f32) {
        self.min_p = min_p;
    }
}

/// Return the index of the maximum element, breaking ties toward the FIRST
/// (lowest) index (RT-22).
///
/// This matches llama.cpp and the Bonsai 2 golden reference set, which
/// matters for exact greedy-decoding parity: the previous implementation
/// (`Iterator::max_by`) breaks ties toward the *last* equally-maximum
/// element, per its documented semantics, so the two backends could disagree
/// on the chosen token whenever two logits tied exactly — surfacing as a
/// phantom kernel bug at the parity gate rather than the tie-break mismatch
/// it actually is.
///
/// Also robust to non-finite input (RT-21): a `NaN` is never preferred over
/// a finite value, because both `NaN > x` and `x > NaN` are `false` for any
/// `x`, so a `NaN` entry can never overwrite a better running maximum (nor
/// can it "win" the very first comparison, since the running maximum starts
/// at `NEG_INFINITY`, not the first element). If every element is
/// non-finite (all `NaN`, all `-inf`, or a mix of the two — i.e. nothing is
/// left to prefer), index `0` is returned: a safe, in-bounds, deterministic
/// answer rather than an arbitrary or nonsensical one. An empty slice also
/// returns `0`, matching the previous implementation's contract; callers
/// must not index into an empty slice with the result (`sample_core` never
/// does, since it returns early for empty `logits`).
fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Returns the index of the highest-probability candidate in `buf`, or an
/// error if `buf` is empty (RT-21).
///
/// This is `sample_core`'s final fallback for the residual
/// floating-point-rounding case where the weighted-selection loop's
/// cumulative sum falls just short of `rand_val` on the very last element.
/// It deliberately does *not* assume `buf` is sorted by probability — when
/// top-p filtering is skipped (`top_p == 1.0`), `buf` is left in
/// top-k/insertion order, so its first element is not necessarily the most
/// likely one — nor that it is non-empty: the `top_k < len` guard in
/// `sample_core` means it always is today, but returning an error rather
/// than indexing keeps this safe even if that invariant is ever broken by a
/// future refactor, and is preferable to silently returning token id `0`,
/// which a caller could mistake for a real (if unlikely) choice.
fn pick_highest_probability(buf: &[(usize, f32)]) -> RuntimeResult<u32> {
    buf.iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|&(idx, _)| idx as u32)
        .ok_or_else(|| {
            RuntimeError::Config(
                "sample_core: candidate buffer was empty after top-k/top-p filtering".to_string(),
            )
        })
}

/// Emits the RT-21 zero-mass warning at most once per process.
///
/// A degenerate candidate set (every logit masked to `-inf`, or a `NaN`
/// reaching the sampler) can otherwise recur on every decode step once it
/// starts — e.g. a constrained-decoding session with no legal continuation,
/// or a model that has diverged — which would flood the log even though the
/// (still fully functional, argmax-based) fallback keeps generation going.
fn warn_zero_mass_once(vocab_size: usize, sum: f32) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            vocab_size,
            sum,
            "sampler: softmax candidate set had zero or non-finite probability mass \
             (every candidate was masked to -inf, or a NaN reached the sampler); \
             falling back to argmax over the raw logits instead of returning NaN or an \
             arbitrary token (this warning is logged once per process)"
        );
    });
}

/// Truncates `buf` — a set of `(token_index, probability)` pairs whose
/// probabilities are non-negative — down to the nucleus (top-p) set: the
/// smallest, highest-probability subset whose cumulative probability
/// exceeds `top_p`, then re-normalizes it so the retained probabilities sum
/// to `1.0`.
///
/// This selects exactly the same *set* of indices as sorting `buf` by
/// descending probability and taking the shortest prefix whose running sum
/// exceeds `top_p` (see this module's `top_p_fast_path_matches_reference_full_sort`
/// proptest for the equivalence check) — but avoids an O(n log n) sort of
/// the *entire* candidate set (perf-12): for an unfiltered vocabulary
/// (`top_k == 0`), a full sort over 248 320 entries (Bonsai 2) measured
/// +3.2 ms/token.
///
/// Instead, an expanding window of the `window` highest-probability elements
/// is found via `select_nth_unstable_by` (an O(n) partial selection) and
/// grown geometrically (32, 128, 512, …) until its probability mass
/// provably exceeds `top_p`. At that point the true nucleus is provably
/// contained within it:
///
/// - Let `M` be the true nucleus size (the smallest count whose top-`M`
///   probabilities sum to more than `top_p`).
/// - If the top-`window` sum exceeds `top_p` for some `window`, then
///   `window` is *a* count satisfying that threshold, and since `M` is the
///   *smallest* such count, `M <= window`.
/// - The top-`M` elements are therefore a subset of the top-`window`
///   elements, so sorting just the (typically tiny, for a peaked
///   post-softmax distribution) `window`-sized slice and re-running the
///   cumulative scan on it finds exactly the same `M` elements a full sort
///   would have.
///
/// Worst case (a near-uniform distribution needing nearly the whole
/// vocabulary to reach `top_p`) this still costs O(n) amortized across
/// `O(log(n))` probes — strictly better than a full sort — and the loop
/// always terminates because the window is clamped to (and the sum check is
/// skipped in favour of taking everything at) `n`.
fn truncate_to_top_p(buf: &mut Vec<(usize, f32)>, top_p: f32) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    const INITIAL_WINDOW: usize = 32;
    let mut window = INITIAL_WINDOW.min(n);
    let cutoff = loop {
        let cutoff = n - window;
        buf.select_nth_unstable_by(cutoff, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });
        let window_sum: f32 = buf[cutoff..].iter().map(|(_, p)| *p).sum();
        if window_sum > top_p || window >= n {
            break cutoff;
        }
        window = window.saturating_mul(4).min(n);
    };

    // The nucleus is provably contained in `buf[cutoff..]` (see doc comment
    // above); sort just that window, descending by probability, to find it
    // exactly.
    buf[cutoff..].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut cum = 0.0f32;
    let keep = buf[cutoff..]
        .iter()
        .position(|&(_, p)| {
            cum += p;
            cum > top_p
        })
        // Only reachable when `window == n` (i.e. `cutoff == 0`) and even the
        // *entire* candidate set's mass doesn't exceed `top_p` — keep
        // everything, matching the reference full-sort behaviour.
        .unwrap_or(window - 1)
        + 1;

    buf.drain(..cutoff);
    buf.truncate(keep);

    // Re-normalize over the retained nucleus.
    let sum: f32 = buf.iter().map(|(_, p)| *p).sum();
    for (_, p) in buf.iter_mut() {
        *p /= sum;
    }
}

/// Min-p (probabilistic nucleus) filtering (RT-23).
///
/// `buf` holds `(token_index, probability)` pairs that already sum to `1.0`
/// (the post-softmax, post-top-k distribution). Keeps every candidate `i`
/// with `p_i >= min_p * max(p)`, then re-normalizes the survivors so they
/// again sum to `1.0` — matching [`truncate_to_top_p`]'s contract so the two
/// filters compose regardless of order or whether either is a no-op.
///
/// `min_p` is clamped to `[0.0, 1.0]` before use, which guarantees the
/// threshold never exceeds `max(p)` and so at least the highest-probability
/// candidate always survives — the distribution is never emptied out from
/// under the caller.
///
/// A no-op when `buf` has zero or one entries (nothing to filter), or when
/// `max(p)` is non-finite (a degenerate input the caller should not have
/// produced; left untouched rather than discarding everything).
fn truncate_to_min_p(buf: &mut Vec<(usize, f32)>, min_p: f32) {
    if buf.len() <= 1 {
        return;
    }

    let max_p = buf
        .iter()
        .map(|(_, p)| *p)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_p.is_finite() {
        return;
    }
    let threshold = min_p.clamp(0.0, 1.0) * max_p;

    buf.retain(|&(_, p)| p >= threshold);

    let sum: f32 = buf.iter().map(|(_, p)| *p).sum();
    if sum > 0.0 {
        for (_, p) in buf.iter_mut() {
            *p /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn greedy_sampling() {
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(params, 42);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sampler.sample(&logits).expect("sampling should succeed");
        assert_eq!(token, 3); // index of 0.9
    }

    #[test]
    fn sampling_returns_valid_index() {
        let params = SamplingParams::default();
        let mut sampler = Sampler::new(params, 12345);
        let logits = vec![0.0f32; 100];
        for _ in 0..50 {
            let token = sampler.sample(&logits).expect("sampling should succeed");
            assert!(token < 100);
        }
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn frequency_presence_penalty_formula() {
        // logit -= presence*I(count>0) + frequency*count
        let mut logits = vec![0.0f32; 5];
        // token 1 appears twice, token 3 once.
        let history = [1u32, 1, 3];
        apply_frequency_presence_penalty(&mut logits, &history, 0.5, 2.0);
        // token 1: -(2.0 + 0.5*2) = -3.0
        assert!((logits[1] - (-3.0)).abs() < 1e-6, "logits[1]={}", logits[1]);
        // token 3: -(2.0 + 0.5*1) = -2.5
        assert!((logits[3] - (-2.5)).abs() < 1e-6, "logits[3]={}", logits[3]);
        // untouched tokens stay 0
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[2], 0.0);
        assert_eq!(logits[4], 0.0);
    }

    #[test]
    fn frequency_presence_penalty_noop_when_zero() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let before = logits.clone();
        apply_frequency_presence_penalty(&mut logits, &[0, 1, 2], 0.0, 0.0);
        assert_eq!(logits, before, "zero penalties must be a no-op");
    }

    #[test]
    fn frequency_presence_penalty_out_of_range_ids_ignored() {
        let mut logits = vec![0.0f32; 3];
        // id 99 is out of range and must be silently skipped (no panic).
        apply_frequency_presence_penalty(&mut logits, &[99, 1], 1.0, 1.0);
        assert_eq!(logits[0], 0.0);
        assert!((logits[1] - (-2.0)).abs() < 1e-6);
        assert_eq!(logits[2], 0.0);
    }

    #[test]
    fn sample_with_history_fast_path_is_bit_identical() {
        // With rep=1.0 and no freq/presence, sample_with_history must equal sample.
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
            max_tokens: 128,
        };
        let logits: Vec<f32> = (0..200).map(|i| (i as f32 * 0.013).sin()).collect();
        let history = [3u32, 3, 7, 12];

        let mut a = Sampler::new(params.clone(), 777);
        let mut b = Sampler::new(params, 777);
        for _ in 0..30 {
            let ta = a.sample(&logits).expect("sample");
            let tb = b
                .sample_with_history(&logits, &history)
                .expect("sample_with_history");
            assert_eq!(ta, tb, "fast path must match base sample exactly");
        }
    }

    #[test]
    fn repetition_penalty_shifts_greedy_choice() {
        // Greedy (temp=0): the top logit is index 4, but penalising it heavily
        // should move the argmax to the next-best unpenalised token.
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 4.0,
            max_tokens: 128,
        };
        let logits = vec![0.1f32, 0.2, 0.3, 0.9, 1.0];
        let mut sampler = Sampler::new(params, 42);
        // Without history, greedy picks index 4 (highest).
        assert_eq!(
            sampler
                .sample_with_history(&logits, &[])
                .expect("no history"),
            4
        );
        // Penalising index 4 (seen) drops 1.0 → 0.25, so index 3 (0.9) wins.
        let token = sampler
            .sample_with_history(&logits, &[4])
            .expect("with history");
        assert_eq!(token, 3, "penalised argmax should move off token 4");
    }

    #[test]
    fn presence_penalty_shifts_greedy_choice() {
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        };
        let logits = vec![0.0f32, 0.0, 0.0, 0.5, 1.0];
        let mut sampler = Sampler::new(params, 42);
        sampler.set_penalties(PenaltyParams::new(0.0, 2.0)); // presence=2.0
                                                             // token 4 (1.0) penalised by 2.0 → -1.0, so token 3 (0.5) wins.
        let token = sampler
            .sample_with_history(&logits, &[4])
            .expect("with presence penalty");
        assert_eq!(token, 3);
    }

    #[test]
    fn buffer_reuse_across_calls() {
        // Verify the probs_buf is correctly reused without incorrect state leaking.
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 5,
            top_p: 1.0, // disable top-p so we control exactly
            repetition_penalty: 1.0,
            max_tokens: 128,
        };
        let mut sampler = Sampler::new(params, 99);
        let logits: Vec<f32> = (0..200).map(|i| i as f32 * 0.01).collect();
        for _ in 0..20 {
            let token = sampler.sample(&logits).expect("sampling should succeed");
            // Top-k=5 on ascending logits: only the last 5 indices (195-199) are valid
            assert!(token >= 195, "expected token ≥ 195, got {token}");
        }
    }

    // ── Reproductions for RT-21 / RT-22 / RT-25 (findings/VERIFIED.md §9) ──
    //
    // These pin the *current* buggy behaviour so a regression is caught; they
    // are written to fail against the unpatched implementation and pass once
    // the corresponding fix lands.

    #[test]
    fn repro_rt22_greedy_ties_break_to_first_index() {
        // RT-22: greedy `argmax` must break ties toward the FIRST index (this
        // matches llama.cpp and the Bonsai 2 golden set); a last-index
        // tie-break is exactly the class of divergence the parity gate trips
        // on.
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(params, 42);
        let logits = vec![1.0f32, 1.0, 1.0];
        let token = sampler.sample(&logits).expect("greedy sample");
        assert_eq!(token, 0, "greedy tie-break must return the FIRST index");
    }

    #[test]
    fn repro_rt25_seed_zero_is_not_degenerate() {
        // RT-25: seeding the xorshift64 PRNG directly from `seed` makes 0 a
        // fixed point (`next_u64()` returns 0 forever), so `rand_val` is
        // always 0.0 and the weighted-selection loop always returns whatever
        // token sits first in `probs_buf` — regardless of the actual logits.
        // `PipelineBuilder::new` documents seed 0 as its default, so this is
        // live, user-reachable behaviour, not a hypothetical.
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        };
        let mut sampler = Sampler::new(params, 0);
        // Token 9 completely dominates; if the sampler is degenerate it will
        // nonetheless always return token 0 (first insertion-order entry).
        let mut logits = vec![-50.0f32; 16];
        logits[9] = 10.0;
        let saw_other_than_zero = (0..32).any(|_| sampler.sample(&logits).expect("sample") != 0);
        assert!(
            saw_other_than_zero,
            "seed=0 must not collapse the sampler to a fixed always-token-0 stream"
        );
    }

    #[test]
    fn repro_rt21_zero_mass_recovers_argmax_signal() {
        // RT-21: a single NaN logit poisons the softmax `sum` to NaN, which
        // (before the fix) corrupts every probability in `probs_buf` — losing
        // the fact that a perfectly valid candidate (index 42) had by far the
        // highest logit. The fix must recover that signal via argmax over the
        // raw logits rather than silently returning an arbitrary token.
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 128,
        };
        let mut sampler = Sampler::new(params, 42);
        let mut logits = vec![0.0f32; 100];
        logits[42] = 50.0; // true best candidate
        logits[7] = f32::NAN; // poisons the softmax sum
        let token = sampler.sample(&logits).expect("sample must not error");
        assert_eq!(
            token, 42,
            "zero/NaN-mass fallback must recover the true argmax, not an arbitrary token"
        );
    }

    // ── argmax: tie-breaking and non-finite robustness (RT-22 / RT-21) ──

    #[test]
    fn argmax_ties_prefer_first_index() {
        assert_eq!(argmax(&[1.0, 1.0, 1.0]), 0);
        assert_eq!(argmax(&[0.0, 5.0, 5.0, 5.0, 1.0]), 1);
    }

    #[test]
    fn argmax_ignores_nan() {
        assert_eq!(argmax(&[f32::NAN, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[1.0, f32::NAN, 5.0, f32::NAN]), 2);
    }

    #[test]
    fn argmax_all_nan_returns_first_index_not_panic() {
        assert_eq!(argmax(&[f32::NAN, f32::NAN, f32::NAN]), 0);
    }

    #[test]
    fn argmax_all_neg_infinity_returns_first_index() {
        assert_eq!(
            argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            0
        );
    }

    // ── pick_highest_probability (RT-21's final-fallback fix) ──

    #[test]
    fn pick_highest_probability_errors_on_empty_buffer() {
        // Not reachable through the public API today (`sample_core`'s
        // `top_k < len` guard means `probs_buf` is never fully drained), but
        // must stay a reported error rather than a panic or a silently
        // returned `0` if that invariant is ever broken by a future
        // refactor.
        let result = pick_highest_probability(&[]);
        assert!(
            result.is_err(),
            "an empty candidate buffer must error, not panic or default silently"
        );
    }

    #[test]
    fn pick_highest_probability_finds_max_regardless_of_order() {
        // Deliberately unsorted and not starting at index 0 — exercises the
        // `top_p == 1.0` case where `probs_buf` is left in top-k/insertion
        // order, which the previous `probs_buf[0]` fallback got wrong.
        let buf = vec![(3usize, 0.1f32), (1, 0.6), (0, 0.3)];
        let idx = pick_highest_probability(&buf).expect("non-empty buffer must succeed");
        assert_eq!(
            idx, 1,
            "must return the index of the actual highest probability"
        );
    }

    // ── seed_to_rng_state (RT-25) ──

    #[test]
    fn seed_to_rng_state_never_yields_the_xorshift64_fixed_point() {
        for seed in 0u64..=1024 {
            assert_ne!(
                seed_to_rng_state(seed),
                0,
                "seed {seed} produced the xorshift64 fixed point"
            );
        }
        // A few large / boundary seeds too.
        for &seed in &[u64::MAX, u64::MAX - 1, 1u64 << 63, 0x9E37_79B9_7F4A_7C15] {
            assert_ne!(
                seed_to_rng_state(seed),
                0,
                "seed {seed:#x} produced the xorshift64 fixed point"
            );
        }
    }

    #[test]
    fn seed_to_rng_state_seed_zero_is_well_mixed() {
        // The previous implementation used `seed` directly, so seed 0 mapped
        // to state 0 (the bug). It must now map to some well-mixed non-zero
        // value.
        let state = seed_to_rng_state(0);
        assert_ne!(state, 0);
    }

    // ── truncate_to_top_p (perf-12) ──

    #[test]
    fn top_p_flat_distribution_near_one_keeps_everything() {
        // top_p close to 1.0 on a perfectly flat distribution: the
        // cumulative sum only crosses top_p at (or very near) the very last
        // element in sorted order — exactly the `window == n` /
        // `unwrap_or(window - 1)` edge in `truncate_to_top_p`.
        let n = 50usize;
        let p = 1.0f32 / n as f32;
        let mut buf: Vec<(usize, f32)> = (0..n).map(|i| (i, p)).collect();
        truncate_to_top_p(&mut buf, 0.999);
        assert_eq!(
            buf.len(),
            n,
            "top_p=0.999 on a flat distribution must retain every candidate"
        );
        let sum: f32 = buf.iter().map(|(_, p)| *p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "retained probabilities must renormalize to ~1.0, got {sum}"
        );
    }

    #[test]
    fn top_p_keeps_only_dominant_candidate() {
        let mut buf = vec![(0usize, 0.001f32), (1, 0.001), (2, 0.997), (3, 0.001)];
        truncate_to_top_p(&mut buf, 0.5);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].0, 2);
        assert!((buf[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_p_single_element_buffer_is_a_no_op() {
        let mut buf = vec![(7usize, 1.0f32)];
        truncate_to_top_p(&mut buf, 0.1);
        assert_eq!(buf, vec![(7usize, 1.0f32)]);
    }

    // ── truncate_to_min_p / Sampler::min_p (RT-23) ──────────────────────

    #[test]
    fn min_p_keeps_only_candidates_within_fraction_of_the_max() {
        // max = 0.6; threshold at min_p=0.5 is 0.3, so only 0.6 and 0.35
        // survive (0.02, 0.02, 0.01 are all below 0.3).
        let mut buf = vec![(0usize, 0.6f32), (1, 0.35), (2, 0.02), (3, 0.02), (4, 0.01)];
        truncate_to_min_p(&mut buf, 0.5);
        let mut ids: Vec<usize> = buf.iter().map(|(i, _)| *i).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1]);
        let sum: f32 = buf.iter().map(|(_, p)| *p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "must renormalize to 1.0, got {sum}"
        );
    }

    #[test]
    fn min_p_zero_is_a_no_op() {
        let mut buf = vec![(0usize, 0.9f32), (1, 0.1)];
        let before = buf.clone();
        truncate_to_min_p(&mut buf, 0.0);
        assert_eq!(buf, before);
    }

    #[test]
    fn min_p_always_keeps_at_least_the_best_candidate() {
        // A degenerate min_p > 1.0 (never expected from a caller that
        // validates the request field, but the sampler itself must stay
        // safe) must not empty the distribution: the element(s) at max_p
        // always satisfy `p >= min_p.clamp(0,1) * max_p`.
        let mut buf = vec![(0usize, 0.7f32), (1, 0.3)];
        truncate_to_min_p(&mut buf, 5.0);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].0, 0);
        assert!((buf[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn min_p_single_element_buffer_is_a_no_op() {
        let mut buf = vec![(3usize, 1.0f32)];
        truncate_to_min_p(&mut buf, 0.9);
        assert_eq!(buf, vec![(3usize, 1.0f32)]);
    }

    #[test]
    fn sampler_min_p_accessor_round_trips() {
        let mut sampler = Sampler::new(SamplingParams::default(), 1);
        assert_eq!(sampler.min_p(), 0.0, "disabled by default");
        sampler.set_min_p(0.1);
        assert!((sampler.min_p() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn sampler_with_min_p_disabled_matches_plain_sample() {
        // min_p = 0.0 (the default) must not perturb `sample_core`'s
        // existing top-k/top-p behavior at all — same RNG draw, same result.
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 8,
        };
        let logits = vec![1.0f32, 2.0, 0.5, 3.0, 0.1];
        let mut with_default = Sampler::new(params.clone(), 42);
        let mut with_min_p_zero = Sampler::new(params, 42);
        with_min_p_zero.set_min_p(0.0);
        for _ in 0..20 {
            let a = with_default.sample(&logits).expect("sample");
            let b = with_min_p_zero.sample(&logits).expect("sample");
            assert_eq!(a, b, "min_p=0.0 must be bit-identical to no min_p");
        }
    }

    #[test]
    fn sampler_min_p_narrows_the_candidate_set_over_many_draws() {
        // A very aggressive min_p (close to 1.0) should collapse sampling to
        // (almost) always pick the highest-probability candidate, across
        // many draws with a high enough temperature that the unfiltered
        // sampler would otherwise pick other tokens too.
        let params = SamplingParams {
            temperature: 2.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            max_tokens: 8,
        };
        let logits = vec![0.0f32, 0.1, 5.0, 0.2, 0.0];
        let mut sampler = Sampler::new(params, 7);
        sampler.set_min_p(0.9);
        let mut counts = std::collections::HashMap::new();
        for _ in 0..200 {
            let tok = sampler.sample(&logits).expect("sample");
            *counts.entry(tok).or_insert(0u32) += 1;
        }
        // Index 2 has by far the largest logit; with min_p=0.9 almost every
        // draw must land on it.
        let dominant = *counts.get(&2).unwrap_or(&0);
        assert!(
            dominant >= 190,
            "min_p=0.9 should overwhelmingly pick the dominant token, got counts {counts:?}"
        );
    }

    /// Reference (pre-perf-12) top-p implementation: a full descending sort
    /// followed by a linear cumulative-sum scan — exactly what
    /// `truncate_to_top_p` replaced. Used only to check that the
    /// windowed-selection implementation always retains the same *set* of
    /// indices; the two need not agree on exact renormalized probability
    /// values (floating-point summation is not associative, so summing in a
    /// different order can differ in the last bit or two).
    fn reference_top_p_set(buf: &[(usize, f32)], top_p: f32) -> std::collections::BTreeSet<usize> {
        let mut sorted = buf.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut cum = 0.0f32;
        let cutoff = sorted
            .iter()
            .position(|&(_, p)| {
                cum += p;
                cum > top_p
            })
            .unwrap_or(sorted.len().saturating_sub(1));
        sorted[..=cutoff].iter().map(|&(idx, _)| idx).collect()
    }

    proptest! {
        #[test]
        fn top_p_fast_path_matches_reference_full_sort(
            // Values are distinct by construction (a strictly increasing
            // per-index offset is added on top of a random base), so the
            // nucleus boundary is never ambiguous between two
            // equally-probable tokens — f32 has ~7 decimal digits of
            // precision at this magnitude, far finer than the 1e-3 spacing
            // added here, so no rounding collision can reintroduce a tie.
            raw in prop::collection::vec(0.0f32..1.0, 2..600),
            top_p in 0.0f32..1.0,
        ) {
            let buf: Vec<(usize, f32)> = raw
                .iter()
                .enumerate()
                .map(|(i, &r)| (i, r + i as f32 * 1e-3))
                .collect();

            let expected = reference_top_p_set(&buf, top_p);

            let mut fast = buf.clone();
            truncate_to_top_p(&mut fast, top_p);
            let actual: std::collections::BTreeSet<usize> =
                fast.iter().map(|&(idx, _)| idx).collect();

            prop_assert_eq!(actual, expected);
        }
    }
}
