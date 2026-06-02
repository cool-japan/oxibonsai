//! Sampling strategies for text generation.
//!
//! Supports temperature scaling, top-k filtering, top-p (nucleus) filtering,
//! and repetition penalty. The [`Sampler`] converts a logit vector into a
//! single token ID using these strategies in order:
//!
//! 1. **Temperature scaling** — divide logits by temperature (0 = greedy argmax)
//! 2. **Top-k** — keep only the k highest-probability candidates
//! 3. **Softmax** — convert scaled logits to probabilities
//! 4. **Top-p** — keep the smallest set of tokens whose cumulative probability exceeds p
//! 5. **Weighted random selection** — sample from the filtered distribution

use std::cmp::Ordering;

use crate::error::RuntimeResult;

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

/// Token sampler.
///
/// Owns a reusable `probs_buf` that is grown on first use and then reused across
/// all subsequent `sample()` calls, eliminating the ~1.8 MB per-call heap
/// allocation that a fresh `Vec` would require for a 151 936-token vocabulary.
#[derive(Debug)]
pub struct Sampler {
    params: SamplingParams,
    rng_state: u64,
    /// Reusable working buffer for `(token_index, scaled_logit)` pairs.
    ///
    /// After `select_nth_unstable_by` + `drain` the buffer holds only the top-k
    /// candidates (capacity stays at `vocab_size`).  `clear()` on the next call
    /// resets length to zero without freeing the backing store, so subsequent
    /// `extend()` calls never reallocate.
    probs_buf: Vec<(usize, f32)>,
}

impl Sampler {
    /// Create a new sampler with the given parameters and seed.
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            rng_state: seed,
            probs_buf: Vec::new(),
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
    #[tracing::instrument(skip(self, logits), fields(vocab_size = logits.len()), level = "debug")]
    pub fn sample(&mut self, logits: &[f32]) -> RuntimeResult<u32> {
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
