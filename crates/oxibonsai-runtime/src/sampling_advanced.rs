//! Advanced sampling algorithms for text generation.
//!
//! This module extends the basic [`crate::sampling`] module with state-of-the-art
//! sampling strategies used in modern LLM inference:
//!
//! - **[`MirostatV1Sampler`]** — feedback-controlled perplexity targeting (Baktash et al. 2020)
//! - **[`MirostatV2Sampler`]** — simplified, more stable mirostat variant
//! - **[`TypicalSampler`]** — locally typical sampling (Meister et al. 2023)
//! - **[`EtaSampler`]** — entropy-adaptive cutoff sampling
//! - **[`MinPSampler`]** — probabilistic nucleus based on min fraction of top token
//! - **[`SamplerChain`]** — composable sampling pipeline with named presets
//! - **[`LcgRng`]** — deterministic LCG pseudo-random number generator (no external deps)
//!
//! ## Helper functions
//!
//! Module-level helpers: [`softmax_inplace`], [`log_softmax`], [`entropy`],
//! [`perplexity`], [`top_k_indices`], [`apply_temperature`], [`apply_repetition_penalty`].

// ─────────────────────────────────────────────────────────────────────────────
// LCG RNG
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Congruential Generator — deterministic pseudo-random number generator.
///
/// Uses the multiplier and increment from Knuth's MMIX:
/// `state = state * 6364136223846793005 + 1442695040888963407`
///
/// No external crate dependencies; suitable for reproducible sampling.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG seeded with `seed`. Identical seeds produce identical streams.
    pub fn new(seed: u64) -> Self {
        // Mix the seed so that seed=0 doesn't get stuck near zero.
        let state = seed
            .wrapping_add(1442695040888963407)
            .wrapping_mul(6364136223846793005);
        Self { state }
    }

    /// Advance the generator and return the next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Return a sample in `[0.0, 1.0)`.
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for f32 mantissa precision.
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }

    /// Return a sample in `0..n` (exclusive). Panics if `n == 0`.
    pub fn next_usize_below(&mut self, n: usize) -> usize {
        assert!(n > 0, "n must be greater than zero");
        (self.next_u64() % n as u64) as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Apply softmax in-place, subtracting the max for numerical stability.
///
/// ## NaN safety (RT-21 sibling)
///
/// When every logit is non-finite (most commonly `f32::NEG_INFINITY`,
/// reachable now that constraint masking uses an exact `-inf` sentinel
/// rather than a large-but-finite one -- see `pipeline.rs`'s
/// `apply_constraint_mask` doc comment, RT-20), `max` is also `-inf` and
/// `(*v - max)` becomes `(-inf) - (-inf) = NaN` for every element,
/// poisoning `sum` and leaving the whole buffer as NaN once the old
/// `sum > 0.0` guard (correctly) declined to divide by it. Rather than
/// leave that NaN in the caller's buffer -- which a downstream sampler
/// cannot recover from except via an arbitrary fallback -- fall back to a
/// first-wins one-hot distribution over the *pre-softmax* logits,
/// mirroring `sampling.rs::sample_core`'s zero-mass fallback and RT-22's
/// first-wins tie-break convention. The same fallback also covers a stray
/// `NaN` reaching this function directly (poisons `sum` immediately,
/// regardless of `max`) and the (unreachable in practice, but defensive)
/// case of every weight underflowing to exactly `0.0`.
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    // First-wins, NaN-robust argmax over the *original* logits (RT-22's
    // fold shape), computed in the same pass as `max` -- before anything
    // below overwrites `logits` in place -- so the fallback below always
    // has a well-defined index to fall back to, without needing a second
    // scan or a snapshot copy of the buffer.
    let mut argmax_idx = 0usize;
    let mut argmax_val = f32::NEG_INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        max = f32::max(max, v);
        if v > argmax_val {
            argmax_val = v;
            argmax_idx = i;
        }
    }
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum.is_finite() && sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    } else {
        // `sum` is NaN or non-positive: every input was non-finite (or
        // every exponentiated weight underflowed to zero). Write a
        // deterministic one-hot distribution instead of leaving
        // NaN/garbage in the buffer for the caller to propagate.
        for (i, v) in logits.iter_mut().enumerate() {
            *v = if i == argmax_idx { 1.0 } else { 0.0 };
        }
    }
}

/// Compute log-softmax for a slice of logits (numerically stable).
///
/// ## NaN safety (RT-21 sibling; mirrors [`softmax_inplace`])
///
/// When every logit is non-finite (most commonly `f32::NEG_INFINITY`, from
/// exhaustive constraint masking, RT-20), `max` is also `-inf` and
/// `(v - max)` becomes `(-inf) - (-inf) = NaN` for every element,
/// poisoning `log_sum_exp` (`.ln()` of a NaN sum is still NaN) and the
/// whole returned vector. Rather than leave that NaN in the caller's
/// buffer -- reachable today via [`TypicalSampler::sample`], which calls
/// this function directly -- fall back to a first-wins one-hot
/// LOG-distribution over the *pre-softmax* logits (`0.0`, i.e. probability
/// 1, at the argmax; `f32::NEG_INFINITY` everywhere else), the same
/// convention `softmax_inplace` uses in probability space and RT-22's
/// first-wins tie-break rule. The same fallback also covers a stray NaN
/// reaching this function directly and any other input for which
/// `log_sum_exp` fails to come out finite.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    // First-wins argmax over the original logits (RT-22's fold shape),
    // computed alongside `max` so the fallback below always has a
    // well-defined index without a second scan.
    let mut argmax_idx = 0usize;
    let mut argmax_val = f32::NEG_INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        max = f32::max(max, v);
        if v > argmax_val {
            argmax_val = v;
            argmax_idx = i;
        }
    }
    let log_sum_exp = logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
    if !log_sum_exp.is_finite() {
        return (0..logits.len())
            .map(|i| {
                if i == argmax_idx {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
    }
    logits.iter().map(|&v| v - log_sum_exp).collect()
}

/// Compute the Shannon entropy (in nats) of a probability distribution.
///
/// Assumes `probs` sums to 1. Skips zero entries to avoid `ln(0) = -inf`.
pub fn entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Compute perplexity from a slice of log-probabilities (natural log).
///
/// `perplexity = exp(mean(-log_prob))`
pub fn perplexity(log_probs: &[f32]) -> f32 {
    if log_probs.is_empty() {
        return 1.0;
    }
    let mean_neg_log: f32 = log_probs.iter().map(|&lp| -lp).sum::<f32>() / log_probs.len() as f32;
    mean_neg_log.exp()
}

/// Return the indices of the top-k highest logit values, sorted descending.
pub fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    if k == 0 || logits.is_empty() {
        return Vec::new();
    }
    let k = k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Divide all logits by `temp`. If `temp <= 0`, this is a no-op (caller should handle greedy).
pub fn apply_temperature(logits: &mut [f32], temp: f32) {
    if temp > 0.0 {
        for v in logits.iter_mut() {
            *v /= temp;
        }
    }
}

/// Apply repetition penalty to logits for previously-seen token ids.
///
/// Tokens with positive logits are divided by `penalty`; negative logits are multiplied.
/// `penalty` should be > 1.0 to discourage repetition.
///
/// ## Once per **distinct** token
///
/// The penalty is applied exactly once per distinct id present in
/// `token_ids`, no matter how often it occurs — HuggingFace's
/// `RepetitionPenaltyLogitsProcessor` (a gather/scatter over unique ids) and
/// llama.cpp behave the same way, and count-proportional discouragement is
/// what [`apply_frequency_presence_penalty`](crate::sampling::apply_frequency_presence_penalty)
/// exists for.
///
/// This used to compound per *occurrence* (`logit / penalty^count`), which
/// was measured changing real decisions: at the first diverging step of the
/// CPU-vs-Metal investigation, token `13` occurred twice in the window and
/// `18.892262 / 1.1 / 1.1 = 15.613` lost to an untouched `16.709`, flipping
/// the argmax; a single application leaves `17.175` and no flip. Because the
/// penalty is applied before the temperature-0 argmax, that turned
/// `--temperature 0` into a *penalised* greedy decode whose output depended
/// on how many times a token had already been emitted.
///
/// The result is independent of the order of `token_ids`, and of the
/// de-duplication's own iteration order: each distinct id maps to a distinct
/// logit index that is updated exactly once.
pub fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    if penalty == 1.0 || token_ids.is_empty() {
        return;
    }
    // De-duplicate as we go. The history handed in by `sample_with_history`
    // is the whole generated sequence, so an O(n²) "did I already see this"
    // scan would grow with the completion length; a set keeps it linear.
    let mut applied: std::collections::HashSet<u32> =
        std::collections::HashSet::with_capacity(token_ids.len());
    for &id in token_ids {
        let idx = id as usize;
        if idx >= logits.len() {
            continue;
        }
        if !applied.insert(id) {
            continue;
        }
        if logits[idx] >= 0.0 {
            logits[idx] /= penalty;
        } else {
            logits[idx] *= penalty;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weighted categorical draw from a probability slice
// ─────────────────────────────────────────────────────────────────────────────

/// Draw an index from `probs` (must sum to 1) using the given RNG.
/// Falls back to index 0 if no threshold is crossed (floating-point edge case).
fn categorical_sample(probs: &[(usize, f32)], rng: &mut LcgRng) -> usize {
    let u = rng.next_f32();
    let mut cumsum = 0.0_f32;
    for &(idx, p) in probs {
        cumsum += p;
        if u < cumsum {
            return idx;
        }
    }
    // Fallback — return highest-probability token.
    probs.first().map(|&(i, _)| i).unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Mirostat v1
// ─────────────────────────────────────────────────────────────────────────────

/// Mirostat v1 sampling — maintains target perplexity via feedback control.
///
/// Reference: Baktash et al., "Mirostat: A Neural Text Decoding Algorithm that
/// Directly Controls Perplexity" (2020), <https://arxiv.org/abs/2007.14966>.
///
/// The algorithm:
/// 1. Truncates the vocabulary to the top-`m` tokens.
/// 2. Estimates the cross-entropy of the chosen token.
/// 3. Updates `mu` (current estimate of target surprise) via `eta`.
#[derive(Debug, Clone)]
pub struct MirostatV1Sampler {
    /// Target surprise level (bits). Default: `5.0`.
    pub tau: f32,
    /// Learning rate for the feedback loop. Default: `0.1`.
    pub eta: f32,
    /// Number of top candidates to consider. Typically `vocab_size / 2`.
    pub m: usize,
    /// Running estimate of the surprise level (initialised to `2 * tau`).
    mu: f32,
}

impl MirostatV1Sampler {
    /// Create a new v1 sampler.
    pub fn new(tau: f32, eta: f32, m: usize) -> Self {
        Self {
            tau,
            eta,
            m,
            mu: 2.0 * tau,
        }
    }

    /// Sample a token index from raw logits, updating internal state.
    pub fn sample(&mut self, logits: &[f32], rng: &mut LcgRng) -> usize {
        if logits.is_empty() {
            return 0;
        }

        // Collect (index, logit) and sort descending.
        let mut candidates: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to top-m.
        let m = self.m.min(candidates.len()).max(1);
        candidates.truncate(m);

        // Softmax over the truncated set.
        let max_v = candidates[0].1;
        let mut sum = 0.0_f32;
        for (_, v) in candidates.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for (_, v) in candidates.iter_mut() {
                *v /= sum;
            }
        }

        // Filter to tokens whose estimated surprise <= mu.
        // Surprise of token i: -log2(p_i).
        let filtered: Vec<(usize, f32)> = candidates
            .iter()
            .cloned()
            .filter(|&(_, p)| p > 0.0 && (-p.log2()) <= self.mu)
            .collect();

        let pool = if filtered.is_empty() {
            &candidates
        } else {
            &filtered
        };

        // Re-normalise the pool.
        let pool_sum: f32 = pool.iter().map(|(_, p)| p).sum();
        let normalised: Vec<(usize, f32)> = if pool_sum > 0.0 {
            pool.iter().map(|&(i, p)| (i, p / pool_sum)).collect()
        } else {
            pool.to_vec()
        };

        // Sample.
        let chosen = categorical_sample(&normalised, rng);

        // Compute observed surprise and update mu.
        if let Some(&(_, p)) = normalised.iter().find(|&&(i, _)| i == chosen) {
            if p > 0.0 {
                let surprise = -p.log2();
                self.mu -= self.eta * (surprise - self.tau);
            }
        }

        chosen
    }

    /// Reset the internal state to the initial value.
    pub fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mirostat v2
// ─────────────────────────────────────────────────────────────────────────────

/// Mirostat v2 sampling — simpler and more stable than v1.
///
/// Rather than pre-truncating to top-m, v2 dynamically computes a probability
/// threshold from `mu`, discards tokens below it, then samples from the rest.
#[derive(Debug, Clone)]
pub struct MirostatV2Sampler {
    /// Target surprise level (bits). Default: `5.0`.
    pub tau: f32,
    /// Learning rate for the feedback loop. Default: `0.1`.
    pub eta: f32,
    /// Running surprise estimate (initialised to `2 * tau`).
    mu: f32,
}

impl MirostatV2Sampler {
    /// Create a new v2 sampler.
    pub fn new(tau: f32, eta: f32) -> Self {
        Self {
            tau,
            eta,
            mu: 2.0 * tau,
        }
    }

    /// Sample a token index from raw logits, updating internal state.
    pub fn sample(&mut self, logits: &[f32], rng: &mut LcgRng) -> usize {
        if logits.is_empty() {
            return 0;
        }

        // Full softmax.
        let mut probs: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        {
            let max_v = probs
                .iter()
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for (_, v) in probs.iter_mut() {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for (_, v) in probs.iter_mut() {
                    *v /= sum;
                }
            }
        }

        // The threshold probability corresponding to self.mu bits of surprise:
        // p_threshold = 2^{-mu}
        let threshold = (-self.mu * std::f32::consts::LN_2).exp();

        let mut pool: Vec<(usize, f32)> = probs
            .iter()
            .cloned()
            .filter(|&(_, p)| p >= threshold)
            .collect();

        if pool.is_empty() {
            // Fallback: keep top-1 token.
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            pool.push(probs[0]);
        }

        // Re-normalise pool.
        let pool_sum: f32 = pool.iter().map(|(_, p)| p).sum();
        if pool_sum > 0.0 {
            for (_, p) in pool.iter_mut() {
                *p /= pool_sum;
            }
        }

        let chosen = categorical_sample(&pool, rng);

        // Update mu from observed surprise.
        if let Some(&(_, p)) = pool.iter().find(|&&(i, _)| i == chosen) {
            if p > 0.0 {
                let surprise = -p.log2();
                self.mu -= self.eta * (surprise - self.tau);
            }
        }

        chosen
    }

    /// Reset the internal state to the initial value.
    pub fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
    }

    /// Current mu value (for diagnostics / tests).
    pub fn mu(&self) -> f32 {
        self.mu
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Locally Typical Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Locally Typical sampling (Meister et al., "Locally Typical Sampling", 2023).
///
/// Keeps the smallest set of tokens whose information content is closest to the
/// conditional entropy of the distribution, summing to at least `p` probability mass.
#[derive(Debug, Clone)]
pub struct TypicalSampler {
    /// Cumulative probability mass to retain. Default: `0.9`.
    pub p: f32,
    /// Minimum number of candidates to keep regardless of `p`. Default: `1`.
    pub min_keep: usize,
}

impl TypicalSampler {
    /// Create a new typical sampler.
    pub fn new(p: f32, min_keep: usize) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            min_keep: min_keep.max(1),
        }
    }

    /// Sample a token index from raw logits.
    pub fn sample(&self, logits: &[f32], rng: &mut LcgRng) -> usize {
        if logits.is_empty() {
            return 0;
        }

        // Compute log-softmax → log-probs and probs.
        let log_probs = log_softmax(logits);
        let probs: Vec<f32> = log_probs.iter().map(|&lp| lp.exp()).collect();

        // Conditional entropy H = -sum_i p_i * log(p_i).
        let h = entropy(&probs);

        // Compute |log(p_i) - H| for each token — how "typical" it is.
        let mut candidates: Vec<(usize, f32, f32)> = log_probs
            .iter()
            .cloned()
            .zip(probs.iter().cloned())
            .enumerate()
            .map(|(i, (lp, p))| {
                let typicality = (-lp - h).abs();
                (i, p, typicality)
            })
            .collect();

        // Sort ascending by typicality (most typical first).
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Keep tokens until we accumulate >= p probability mass.
        let mut cumsum = 0.0_f32;
        let mut keep = 0;
        for (k, &(_, p, _)) in candidates.iter().enumerate() {
            cumsum += p;
            keep = k + 1;
            if cumsum >= self.p && keep >= self.min_keep {
                break;
            }
        }
        keep = keep.max(self.min_keep).min(candidates.len());
        candidates.truncate(keep);

        // Re-normalise and sample.
        let total: f32 = candidates.iter().map(|(_, p, _)| p).sum();
        let normalised: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&(i, p, _)| (i, if total > 0.0 { p / total } else { p }))
            .collect();

        categorical_sample(&normalised, rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Eta Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Eta sampling — adaptively selects a probability cutoff based on distribution entropy.
///
/// The cutoff is `max(epsilon, sqrt(exp(-H(p))) * delta)` where `H` is the entropy.
/// Tokens below the cutoff are discarded.
#[derive(Debug, Clone)]
pub struct EtaSampler {
    /// Minimum token probability (floor). Default: `0.0009`.
    pub epsilon: f32,
    /// Entropy scaling factor for adaptive threshold. Default: `0.07`.
    pub delta: f32,
}

impl EtaSampler {
    /// Create a new eta sampler.
    pub fn new(epsilon: f32, delta: f32) -> Self {
        Self { epsilon, delta }
    }

    /// Sample a token index from raw logits.
    pub fn sample(&self, logits: &[f32], rng: &mut LcgRng) -> usize {
        if logits.is_empty() {
            return 0;
        }

        let mut probs: Vec<f32> = logits.to_vec();
        softmax_inplace(&mut probs);

        // Adaptive threshold.
        let h = entropy(&probs);
        let eta_threshold = (self.epsilon).max((-h).exp().sqrt() * self.delta);

        let mut candidates: Vec<(usize, f32)> = probs
            .iter()
            .cloned()
            .enumerate()
            .filter(|&(_, p)| p >= eta_threshold)
            .collect();

        if candidates.is_empty() {
            // Fallback: take argmax.
            let best = probs
                .iter()
                .cloned()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            return best;
        }

        // Re-normalise.
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        if total > 0.0 {
            for (_, p) in candidates.iter_mut() {
                *p /= total;
            }
        }

        categorical_sample(&candidates, rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Min-P Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Min-P sampling — probabilistic nucleus based on a minimum fraction of the top-token probability.
///
/// Keeps all tokens `i` where `p_i >= min_p * max(p)`.
#[derive(Debug, Clone)]
pub struct MinPSampler {
    /// Minimum fraction of the maximum probability. Default: `0.05`.
    pub min_p: f32,
    /// Minimum candidates to keep regardless of the threshold. Default: `1`.
    pub min_keep: usize,
}

impl MinPSampler {
    /// Create a new Min-P sampler.
    pub fn new(min_p: f32, min_keep: usize) -> Self {
        Self {
            min_p: min_p.clamp(0.0, 1.0),
            min_keep: min_keep.max(1),
        }
    }

    /// Sample a token index from raw logits.
    pub fn sample(&self, logits: &[f32], rng: &mut LcgRng) -> usize {
        if logits.is_empty() {
            return 0;
        }

        let mut probs: Vec<f32> = logits.to_vec();
        softmax_inplace(&mut probs);

        let max_p = probs.iter().cloned().fold(0.0_f32, f32::max);
        let threshold = self.min_p * max_p;

        let mut candidates: Vec<(usize, f32)> = probs
            .iter()
            .cloned()
            .enumerate()
            .filter(|&(_, p)| p >= threshold)
            .collect();

        // Ensure min_keep.
        if candidates.len() < self.min_keep {
            // Sort all probs descending and take top min_keep.
            let mut all: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates = all.into_iter().take(self.min_keep).collect();
        }

        // Re-normalise.
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        if total > 0.0 {
            for (_, p) in candidates.iter_mut() {
                *p /= total;
            }
        }

        categorical_sample(&candidates, rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampler Chain
// ─────────────────────────────────────────────────────────────────────────────

/// A single step in a [`SamplerChain`] pipeline.
///
/// Steps are applied in order to the logit vector before final sampling.
#[derive(Debug, Clone)]
pub enum SamplerStep {
    /// Divide logits by temperature. Values near 0 produce near-greedy output.
    Temperature(f32),
    /// Penalise previously-seen tokens to reduce repetition.
    RepetitionPenalty {
        /// Penalty multiplier (>1.0 discourages repetition).
        penalty: f32,
        /// Number of recent tokens to consider (window).
        last_n: usize,
        /// The recent token ids to penalise.
        tokens: Vec<u32>,
    },
    /// Keep only the top-k highest-logit candidates.
    TopK(usize),
    /// Nucleus (top-p) filtering.
    TopP(f32),
    /// Min-P filtering (min fraction of top token probability).
    MinP(f32),
    /// Locally typical sampling with probability mass `p`.
    Typical(f32),
    /// Mirostat v2 with given tau and eta.
    Mirostat2 {
        /// Target surprise (bits).
        tau: f32,
        /// Learning rate.
        eta: f32,
    },
    /// Always pick the argmax (no randomness).
    Greedy,
}

/// Composable sampling pipeline.
///
/// Steps are applied sequentially to the logit vector. The first `Greedy` or
/// `Mirostat2` step that yields a token terminates the pipeline. All other steps
/// modify the logit/probability vector in place.
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::sampling_advanced::{SamplerChain, SamplerStep};
///
/// let mut chain = SamplerChain::default_chat(42);
/// let mut logits = vec![1.0_f32, 5.0, 2.0, 3.0];
/// let token = chain.sample(&mut logits);
/// assert!(token < 4);
/// ```
#[derive(Debug, Clone)]
pub struct SamplerChain {
    steps: Vec<SamplerStep>,
    rng: LcgRng,
    /// Persistent Mirostat v2 state (one per chain).
    mirostat2: Option<MirostatV2Sampler>,
}

impl SamplerChain {
    /// Create an empty chain with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            steps: Vec::new(),
            rng: LcgRng::new(seed),
            mirostat2: None,
        }
    }

    /// Append a step to the chain (builder pattern).
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, step: SamplerStep) -> Self {
        // If Mirostat2 step is added, initialise persistent state.
        if let SamplerStep::Mirostat2 { tau, eta } = step {
            self.mirostat2 = Some(MirostatV2Sampler::new(tau, eta));
        }
        self.steps.push(step);
        self
    }

    /// Sample from the given logits, applying all steps in order.
    ///
    /// `logits` is consumed/mutated during processing.
    pub fn sample(&mut self, logits: &mut Vec<f32>) -> usize {
        if logits.is_empty() {
            return 0;
        }

        for step in &self.steps {
            match step {
                SamplerStep::Temperature(temp) => {
                    if *temp < 1e-6 {
                        // Treat as greedy immediately.
                        return argmax_slice(logits);
                    }
                    apply_temperature(logits, *temp);
                }

                SamplerStep::RepetitionPenalty {
                    penalty,
                    last_n,
                    tokens,
                } => {
                    let window = if *last_n == 0 {
                        tokens.as_slice()
                    } else {
                        let start = tokens.len().saturating_sub(*last_n);
                        &tokens[start..]
                    };
                    apply_repetition_penalty(logits, window, *penalty);
                }

                SamplerStep::TopK(k) => {
                    if *k > 0 && *k < logits.len() {
                        let indices = top_k_indices(logits, *k);
                        let mut mask = vec![f32::NEG_INFINITY; logits.len()];
                        for i in indices {
                            mask[i] = logits[i];
                        }
                        *logits = mask;
                    }
                }

                SamplerStep::TopP(p) => {
                    if *p < 1.0 {
                        apply_top_p(logits, *p, &mut self.rng);
                        // top_p returns early — but we continue to let sampling happen below.
                    }
                }

                SamplerStep::MinP(min_p) => {
                    let sampler = MinPSampler::new(*min_p, 1);
                    return sampler.sample(logits, &mut self.rng);
                }

                SamplerStep::Typical(p) => {
                    let sampler = TypicalSampler::new(*p, 1);
                    return sampler.sample(logits, &mut self.rng);
                }

                SamplerStep::Mirostat2 { .. } => {
                    // Use persistent state stored in self.mirostat2.
                    if let Some(ref mut ms) = self.mirostat2 {
                        return ms.sample(logits, &mut self.rng);
                    }
                }

                SamplerStep::Greedy => {
                    return argmax_slice(logits);
                }
            }
        }

        // Default: softmax then weighted sample.
        softmax_inplace(logits);
        let probs: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        categorical_sample(&probs, &mut self.rng)
    }

    // ── Presets ──────────────────────────────────────────────────────────────

    /// Greedy decoding — always picks the token with the highest logit.
    pub fn greedy() -> Self {
        Self::new(0).add(SamplerStep::Greedy)
    }

    /// Default chat preset: temperature(0.7) → top_p(0.9) → min_p(0.05).
    pub fn default_chat(seed: u64) -> Self {
        Self::new(seed)
            .add(SamplerStep::Temperature(0.7))
            .add(SamplerStep::TopP(0.9))
            .add(SamplerStep::MinP(0.05))
    }

    /// Creative preset: temperature(1.0) → mirostat_v2(tau=5.0, eta=0.1).
    pub fn creative(seed: u64) -> Self {
        Self::new(seed)
            .add(SamplerStep::Temperature(1.0))
            .add(SamplerStep::Mirostat2 { tau: 5.0, eta: 0.1 })
    }

    /// Precise preset: temperature(0.3) → top_k(40) → top_p(0.9).
    pub fn precise(seed: u64) -> Self {
        Self::new(seed)
            .add(SamplerStep::Temperature(0.3))
            .add(SamplerStep::TopK(40))
            .add(SamplerStep::TopP(0.9))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the index of the maximum element (ties broken by lowest index).
///
/// First-wins tie-break and NaN-robust (RT-22 sibling): the previous
/// `.max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))` returned the LAST of
/// several equal maxima -- contradicting this function's own doc comment --
/// and let a NaN encountered after the true maximum evict it, since
/// `partial_cmp` against a NaN is `None` and `max_by` resolves a tie (which
/// `unwrap_or(Equal)` turns an incomparable pair into) in favour of the
/// *newer* element. This is the same fold shape as
/// `pipeline.rs::argmax_logits` and `sampling.rs::argmax`, so all four
/// greedy paths reachable from this crate's public API now agree on exact
/// ties -- this one backs both `SamplerStep::Greedy` and the
/// `Temperature < 1e-6` short-circuit in `SamplerChain::sample`, so
/// `SamplerChain::greedy()`, `PipelineBuilder::sampling(...)` and
/// `ConstrainedSampler` all inherit the fix.
fn argmax_slice(values: &[f32]) -> usize {
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

/// Apply top-p (nucleus) filtering to a logit vector in-place.
///
/// Tokens outside the nucleus are set to `NEG_INFINITY` so they are excluded
/// by a subsequent softmax + sample step.
fn apply_top_p(logits: &mut [f32], p: f32, _rng: &mut LcgRng) {
    // Compute softmax probabilities.
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, (v - max_v).exp()))
        .collect();
    let total: f32 = probs.iter().map(|(_, v)| v).sum();
    if total > 0.0 {
        for (_, v) in probs.iter_mut() {
            *v /= total;
        }
    }

    // Sort descending by probability.
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find nucleus boundary.
    let mut cumsum = 0.0_f32;
    let mut nucleus_end = 0;
    for (k, &(_, prob)) in probs.iter().enumerate() {
        cumsum += prob;
        nucleus_end = k;
        if cumsum >= p {
            break;
        }
    }

    // Collect nucleus indices.
    let nucleus_indices: std::collections::HashSet<usize> =
        probs[..=nucleus_end].iter().map(|&(i, _)| i).collect();

    // Mask out non-nucleus tokens.
    for (i, v) in logits.iter_mut().enumerate() {
        if !nucleus_indices.contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (module-internal)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Repetition penalty: once per DISTINCT token ──────────────────────

    /// Reproduction of the exact decision point from the CPU-vs-Metal
    /// divergence investigation (§3.4), on the real 1.7B's logits at the
    /// first diverging step.
    ///
    /// History at that step contains token `13` twice and token `624` not at
    /// all. Applying the penalty once per **occurrence** gives
    /// `18.892262 / 1.1 / 1.1 = 15.613`, which loses to `624`'s `16.709` and
    /// flips the argmax. HuggingFace's `RepetitionPenaltyLogitsProcessor` and
    /// llama.cpp both divide once per distinct token present in the window
    /// (count-proportional behaviour is what `frequency_penalty` is for, and
    /// this crate implements that separately), giving `17.175` — which keeps
    /// the argmax on `13`.
    #[test]
    fn repetition_penalty_applies_once_per_distinct_token() {
        let mut logits = vec![0.0f32; 1000];
        logits[13] = 18.892262;
        logits[624] = 16.708843;
        logits[382] = 16.314722;

        // The 15-token common prefix from the report: token 13 twice.
        let history = [
            26194u32, 13, 576, 6722, 315, 6323, 374, 26194, 13, 576, 6722, 315, 6323, 374, 26194,
        ];
        apply_repetition_penalty(&mut logits, &history, 1.1);

        let expected = 18.892262f32 / 1.1;
        assert!(
            (logits[13] - expected).abs() < 1e-4,
            "token 13 must be penalised exactly once (expected {expected}, got {})",
            logits[13]
        );
        assert!(
            logits[13] > logits[624],
            "the argmax must NOT flip: 13={} vs 624={}",
            logits[13],
            logits[624]
        );
    }

    #[test]
    fn repetition_penalty_negative_logits_multiply_once() {
        let mut logits = vec![-2.0f32, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 0, 0], 2.0);
        assert!(
            (logits[0] - (-4.0)).abs() < 1e-6,
            "a negative logit is multiplied once, not once per occurrence: {}",
            logits[0]
        );
        assert_eq!(logits[1], 1.0, "unseen tokens are untouched");
    }

    #[test]
    fn repetition_penalty_is_order_independent() {
        let mut a = vec![3.0f32, 3.0, 3.0];
        let mut b = a.clone();
        apply_repetition_penalty(&mut a, &[0, 1, 0, 2, 1], 1.5);
        apply_repetition_penalty(&mut b, &[2, 1, 1, 0, 0], 1.5);
        assert_eq!(a, b, "the result must not depend on history order");
        let once = 3.0f32 / 1.5;
        for (i, v) in a.iter().enumerate() {
            assert!((v - once).abs() < 1e-6, "logits[{i}] = {v}");
        }
    }

    #[test]
    fn repetition_penalty_ignores_out_of_range_ids() {
        let mut logits = vec![1.0f32, 2.0];
        apply_repetition_penalty(&mut logits, &[7, 7, 1], 2.0);
        assert_eq!(logits[0], 1.0);
        assert!((logits[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lcg_rng_produces_values() {
        let mut rng = LcgRng::new(1);
        let v = rng.next_f32();
        assert!((0.0..1.0).contains(&v), "f32 out of range: {v}");
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    // ── RT-21 sibling: softmax_inplace must never leave NaN in the buffer ──

    #[test]
    fn softmax_inplace_all_neg_infinity_produces_no_nan_and_is_deterministic() {
        // Reachable now that RT-20 switched constraint masking from a
        // finite `-1e9` sentinel to an exact `-inf`: an all-disallowed
        // mask feeds an all-`-inf` slice in here. `max` becomes `-inf`, so
        // `(*v - max).exp()` computes `(-inf) - (-inf) = NaN` for every
        // element, poisoning `sum`; the old `if sum > 0.0` guard correctly
        // declines to divide by a NaN `sum`, but leaves the NaN-filled
        // buffer untouched instead of normalising to anything sane.
        let mut logits = vec![f32::NEG_INFINITY; 5];
        softmax_inplace(&mut logits);
        assert!(
            logits.iter().all(|v| !v.is_nan()),
            "softmax_inplace must never leave NaN in the buffer: {logits:?}"
        );
        // First-wins one-hot fallback (mirrors RT-22's tie-break
        // convention): index 0 gets all the mass.
        assert_eq!(logits, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < f32::EPSILON,
            "fallback distribution must still sum to 1.0, got {sum}"
        );

        // categorical_sample must deterministically resolve to the one-hot
        // index for any draw, not an arbitrary tie-break over NaN weights.
        let pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        for seed in [0u64, 1, 42, 999_999] {
            let mut rng = LcgRng::new(seed);
            let chosen = categorical_sample(&pairs, &mut rng);
            assert_eq!(
                chosen, 0,
                "categorical_sample must pick the one-hot index deterministically, \
                 got {chosen} for seed {seed}"
            );
        }
    }

    // ── RT-21 sibling: log_softmax must never leave NaN in the buffer ──────

    #[test]
    fn log_softmax_all_neg_infinity_produces_no_nan_and_is_deterministic() {
        // Same defect class as `softmax_inplace` (fixed above), reachable
        // through the identical `-inf` constraint-masking path: `max` is
        // `-inf`, so `(v - max).exp()` computes `(-inf) - (-inf) = NaN` for
        // every element, poisoning `log_sum_exp` (`.ln()` of a NaN sum,
        // plus `-inf`, is still NaN) and every returned entry. This is live
        // via `TypicalSampler::sample`, which calls `log_softmax` directly.
        let logits = vec![f32::NEG_INFINITY; 5];
        let log_probs = log_softmax(&logits);
        assert!(
            log_probs.iter().all(|v| !v.is_nan()),
            "log_softmax must never leave NaN in the buffer: {log_probs:?}"
        );
        // First-wins one-hot LOG-distribution (mirrors `softmax_inplace`'s
        // probability-space convention and RT-22's tie-break rule): `0.0`
        // (i.e. probability 1) at index 0, `-inf` (probability 0) elsewhere.
        assert_eq!(
            log_probs,
            vec![
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY
            ]
        );
        // Exponentiating back must give a clean, NaN-free one-hot
        // probability distribution that still sums to 1.
        let probs: Vec<f32> = log_probs.iter().map(|&lp| lp.exp()).collect();
        assert!(probs.iter().all(|p| !p.is_nan()));
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < f32::EPSILON,
            "fallback distribution must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn log_softmax_partial_neg_infinity_is_unaffected() {
        // A partial mask (only *some* entries `-inf`) must keep behaving
        // exactly as before: this is the numerically well-defined case
        // (`max` is finite, `(-inf - finite) = -inf`, never NaN), so the
        // fallback must not fire here.
        let logits = vec![1.0f32, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
        let log_probs = log_softmax(&logits);
        assert!(log_probs.iter().all(|v| !v.is_nan()));
        assert_eq!(log_probs[1], f32::NEG_INFINITY);
        assert_eq!(log_probs[3], f32::NEG_INFINITY);
        // The two finite entries must still sum to 1 in probability space.
        let sum: f32 = (log_probs[0].exp()) + (log_probs[2].exp());
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn mirostat_v2_returns_valid_index() {
        let logits = vec![1.0_f32, 5.0, 2.0, 3.0];
        let mut sampler = MirostatV2Sampler::new(5.0, 0.1);
        let mut rng = LcgRng::new(99);
        let idx = sampler.sample(&logits, &mut rng);
        assert!(idx < logits.len());
    }

    #[test]
    fn sampler_chain_greedy_preset() {
        let mut chain = SamplerChain::greedy();
        let mut logits = vec![0.1_f32, 5.0, 0.2, 0.3];
        let tok = chain.sample(&mut logits);
        assert_eq!(tok, 1); // index of 5.0
    }

    // ── RT-22 sibling: argmax_slice must agree with the crate's other ───────
    // ── three greedy paths on first-wins ties and NaN-robustness ────────────

    #[test]
    fn sampler_chain_greedy_breaks_ties_first_wins() {
        // `argmax_slice` backs both `SamplerStep::Greedy` and the
        // `Temperature < 1e-6` short-circuit in `SamplerChain::sample` --
        // i.e. `SamplerChain::greedy()`, `PipelineBuilder::sampling(...)`
        // and `ConstrainedSampler` all funnel through it. Its own doc
        // comment already claims "ties broken by lowest index", but
        // `Iterator::max_by` returns the *last* of several equal maxima,
        // so a plateau used to resolve to the highest index instead --
        // silently disagreeing with `sampling.rs::argmax` and
        // `pipeline.rs::argmax_logits` (both RT-22 first-wins) on the
        // exact same input, which matters for temp-0 byte-parity against
        // a reference implementation.
        let mut chain = SamplerChain::greedy();
        let mut logits = vec![5.0_f32, 5.0, 5.0, 1.0];
        let tok = chain.sample(&mut logits);
        assert_eq!(
            tok, 0,
            "a plateau of equal maxima must resolve to the first (lowest) index"
        );
    }

    #[test]
    fn sampler_chain_temperature_near_zero_short_circuit_breaks_ties_first_wins() {
        // Same `argmax_slice` call, reached via the other call site
        // (`Temperature < 1e-6`) rather than `SamplerStep::Greedy`.
        let mut chain = SamplerChain::new(0).add(SamplerStep::Temperature(1e-9));
        let mut logits = vec![2.0_f32, 9.0, 9.0, 9.0, 0.0];
        let tok = chain.sample(&mut logits);
        assert_eq!(
            tok, 1,
            "near-zero temperature must resolve a plateau to the first (lowest) index"
        );
    }

    #[test]
    fn sampler_chain_greedy_a_nan_after_the_true_max_does_not_displace_it() {
        // NaN-robustness half of RT-22. `partial_cmp(&NaN)` is `None`, and
        // `Iterator::max_by`'s left-fold treats an incomparable (or equal)
        // pair as a tie *won by the newer element* -- so the old
        // `max_by(|a,b| a.partial_cmp(b).unwrap_or(Equal))` let a NaN
        // encountered *after* the true maximum evict it: comparing
        // (running max = 7.0) against (next = NaN) yields `Equal` (via
        // `unwrap_or`), and a tie is resolved in favour of the *newer*
        // element, so the fold's running max became NaN, then the
        // following real value (3.0) evicted *that* via the same
        // Equal-favours-newer rule -- silently returning index 2 (value
        // 3.0) despite 7.0 (index 0) being the true, unambiguous maximum.
        let mut chain = SamplerChain::greedy();
        let mut logits = vec![7.0_f32, f32::NAN, 3.0];
        let tok = chain.sample(&mut logits);
        assert_eq!(
            tok, 0,
            "a NaN must never displace a real maximum that precedes it"
        );
    }
}
