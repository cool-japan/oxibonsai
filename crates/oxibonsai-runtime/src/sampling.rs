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

use crate::error::RuntimeResult;
use crate::sampling_advanced::apply_repetition_penalty;

/// Sampling parameters.
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
            repetition_penalty: 1.1,
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

impl Sampler {
    /// Create a new sampler with the given parameters and seed.
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            penalties: PenaltyParams::default(),
            rng_state: seed,
            probs_buf: Vec::new(),
            penalty_buf: Vec::new(),
        }
    }

    /// Simple xorshift64 PRNG — no external dependency needed.
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
        for (_, v) in self.probs_buf.iter_mut() {
            *v /= sum;
        }

        // Top-p filtering
        if self.params.top_p < 1.0 {
            self.probs_buf
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            let mut cum = 0.0f32;
            let cutoff = self
                .probs_buf
                .iter()
                .position(|&(_, p)| {
                    cum += p;
                    cum > self.params.top_p
                })
                .unwrap_or(self.probs_buf.len().saturating_sub(1));
            self.probs_buf.truncate(cutoff + 1);

            // Re-normalize
            let sum: f32 = self.probs_buf.iter().map(|(_, p)| p).sum();
            for (_, p) in self.probs_buf.iter_mut() {
                *p /= sum;
            }
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

        // Fallback: return the highest probability token
        Ok(self.probs_buf[0].0 as u32)
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
}

/// Return the index of the maximum element.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
