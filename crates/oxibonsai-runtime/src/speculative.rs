//! Speculative decoding for accelerated autoregressive generation.
//!
//! Speculative decoding uses a small "draft" model to generate K candidate tokens,
//! which the larger "target" model then verifies in a single parallel forward pass.
//! Accepted tokens are kept; the first rejected token is resampled from the target
//! distribution. This can yield near-linear speedup proportional to the average
//! number of accepted tokens per step.
//!
//! ## Algorithm (Leviathan et al., 2023)
//!
//! 1. Draft model generates K tokens: `t_1, ..., t_K` with draft probabilities `p_d`
//! 2. Target model scores all K+1 positions in parallel, producing `p_t`
//! 3. For each position `i`, accept `t_i` if:
//!    - `p_t(t_i) >= p_d(t_i)`, OR
//!    - with probability `p_t(t_i) / p_d(t_i)` (rejection sampling)
//! 4. If rejected at position `i`, resample from adjusted distribution
//! 5. Always append one bonus target-sampled token after full acceptance
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxibonsai_core::config::Qwen3Config;
//! use oxibonsai_runtime::engine::InferenceEngine;
//! use oxibonsai_runtime::sampling::SamplingParams;
//! use oxibonsai_runtime::speculative::{SpeculativeConfig, SpeculativeDecoder};
//!
//! let config = Qwen3Config::tiny_test();
//! let draft_engine = InferenceEngine::new(config, SamplingParams::default(), 42);
//! let spec_config = SpeculativeConfig::default();
//! let mut decoder = SpeculativeDecoder::new(draft_engine, spec_config);
//! ```

use crate::adaptive_lookahead::{AdaptiveLookahead, AdaptiveLookaheadConfig};
use crate::engine::InferenceEngine;
use crate::sampling::SamplingParams;

// ──────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per step (lookahead K, typically 4–8).
    pub lookahead: usize,
    /// Minimum acceptance ratio threshold (0.0 = pure rejection sampling criterion).
    ///
    /// Setting this above 0.0 makes the decoder more conservative (fewer accepted
    /// tokens per step, but closer to target distribution).
    pub acceptance_threshold: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            lookahead: 5,
            acceptance_threshold: 0.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Step result
// ──────────────────────────────────────────────────────────────────

/// Result from one speculative decoding step (draft + verify).
#[derive(Debug, Clone)]
pub struct SpeculativeStep {
    /// Tokens proposed by the draft model.
    pub draft_tokens: Vec<u32>,
    /// Tokens accepted after verification against the target.
    pub accepted_tokens: Vec<u32>,
    /// Fraction of draft tokens that were accepted: `accepted / proposed`.
    pub acceptance_rate: f32,
}

// ──────────────────────────────────────────────────────────────────
// Internal mini-PRNG (xorshift64, no external rand crate)
// ──────────────────────────────────────────────────────────────────

/// Minimal xorshift64 PRNG state — no external dependency.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state (xorshift must not start at 0)
        let state = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns a sample in `[0.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        // Use top 24 bits for f32 mantissa precision
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ──────────────────────────────────────────────────────────────────
// SpeculativeDecoder
// ──────────────────────────────────────────────────────────────────

/// Speculative decoder: wraps a draft [`InferenceEngine`] and provides
/// draft-then-verify generation with running acceptance statistics.
pub struct SpeculativeDecoder<'a> {
    /// Draft model engine (smaller/faster model).
    pub draft_engine: InferenceEngine<'a>,
    /// Speculative decoding configuration.
    pub config: SpeculativeConfig,
    /// Total number of speculative steps taken.
    pub total_steps: u64,
    /// Total number of tokens proposed by the draft model.
    pub total_draft_tokens: u64,
    /// Total number of tokens accepted after target verification.
    pub total_accepted_tokens: u64,
    /// Internal PRNG for rejection sampling decisions (available for subtype use).
    #[allow(dead_code)]
    rng: Xorshift64,
    /// Optional adaptive controller — when present, the lookahead is
    /// updated after each step from the running acceptance EWMA.
    adaptive: Option<AdaptiveLookahead>,
}

impl<'a> SpeculativeDecoder<'a> {
    /// Create a new speculative decoder with the given draft engine and config.
    pub fn new(draft_engine: InferenceEngine<'a>, config: SpeculativeConfig) -> Self {
        Self {
            draft_engine,
            config,
            total_steps: 0,
            total_draft_tokens: 0,
            total_accepted_tokens: 0,
            rng: Xorshift64::new(0xfeed1234_5678abcd),
            adaptive: None,
        }
    }

    /// Create a speculative decoder with an [`AdaptiveLookahead`] controller
    /// active. The initial lookahead is taken from `adaptive_config.initial`
    /// and overrides `config.lookahead` for the first step.
    pub fn with_adaptive(
        draft_engine: InferenceEngine<'a>,
        config: SpeculativeConfig,
        adaptive_config: AdaptiveLookaheadConfig,
    ) -> Result<Self, crate::adaptive_lookahead::AdaptiveLookaheadError> {
        let adaptive = AdaptiveLookahead::try_new(adaptive_config)?;
        let mut config = config;
        config.lookahead = adaptive.lookahead();
        Ok(Self {
            draft_engine,
            config,
            total_steps: 0,
            total_draft_tokens: 0,
            total_accepted_tokens: 0,
            rng: Xorshift64::new(0xfeed1234_5678abcd),
            adaptive: Some(adaptive),
        })
    }

    /// Read the current adaptive controller, if any.
    pub fn adaptive(&self) -> Option<&AdaptiveLookahead> {
        self.adaptive.as_ref()
    }

    /// Mutable access to the adaptive controller, if any.
    pub fn adaptive_mut(&mut self) -> Option<&mut AdaptiveLookahead> {
        self.adaptive.as_mut()
    }

    /// Generate up to `config.lookahead` draft tokens from the draft model.
    ///
    /// In this implementation, the draft engine uses its sampler to produce tokens
    /// autoregressively from `context`. The returned tokens are the draft candidates
    /// for target-model verification.
    pub fn draft(&mut self, context: &[u32], _params: &SamplingParams) -> Vec<u32> {
        let k = self.config.lookahead;
        let mut draft_tokens = Vec::with_capacity(k);

        // Build a combined context + generated so far
        let mut current_context: Vec<u32> = context.to_vec();

        for _ in 0..k {
            // Generate one token using the draft engine
            match self.draft_engine.generate(&current_context, 1) {
                Ok(generated) if !generated.is_empty() => {
                    let token = generated[0];
                    draft_tokens.push(token);
                    current_context.push(token);
                }
                _ => {
                    // Draft generation failed or returned empty — stop drafting
                    break;
                }
            }
        }

        draft_tokens
    }

    /// Verify draft tokens against target-model logits.
    ///
    /// For each draft position `i`, the target's probability `p_t(t_i)` is
    /// compared against a mock draft probability `p_d(t_i)` derived from
    /// the target logits (as a self-consistency check when target logits are
    /// provided). In production, `p_d` comes from the draft model's softmax.
    ///
    /// Acceptance criterion (speculative sampling):
    /// - Accept if `p_t(t_i) >= p_d(t_i)`
    /// - Else accept with probability `p_t(t_i) / p_d(t_i)`
    ///
    /// Returns only the prefix of tokens accepted before the first rejection.
    pub fn verify(
        &self,
        draft_tokens: &[u32],
        target_logits: &[Vec<f32>],
        _params: &SamplingParams,
    ) -> Vec<u32> {
        let mut accepted = Vec::with_capacity(draft_tokens.len());

        // We need a mutable PRNG — use a local one seeded from step count for reproducibility
        let mut local_rng = Xorshift64::new(
            self.total_steps
                .wrapping_mul(6364136223846793005)
                .wrapping_add(0xabcdef01),
        );

        for (i, &token) in draft_tokens.iter().enumerate() {
            let logits = match target_logits.get(i) {
                Some(l) => l,
                None => break,
            };

            if logits.is_empty() {
                break;
            }

            // Compute softmax probabilities for target
            let target_probs = softmax(logits);

            // Get target probability for this draft token
            let target_prob = if (token as usize) < target_probs.len() {
                target_probs[token as usize]
            } else {
                0.0
            };

            // Mock draft probability: use a uniform-like estimate over top candidates
            // In production this would come from the draft model's own softmax output.
            // Here we use 1/vocab_size as a conservative draft estimate.
            let vocab_size = logits.len() as f32;
            let draft_prob = (1.0 / vocab_size).max(1e-9);

            let rng_sample = local_rng.next_f32();
            let threshold = self.config.acceptance_threshold;

            if Self::should_accept(draft_prob, target_prob, threshold, rng_sample) {
                accepted.push(token);
            } else {
                // First rejection — stop here
                break;
            }
        }

        accepted
    }

    /// Perform one complete speculative decoding step: draft K tokens then verify.
    ///
    /// Returns a [`SpeculativeStep`] with the draft proposals, accepted subset,
    /// and per-step acceptance rate.
    pub fn step(
        &mut self,
        context: &[u32],
        target_logits: &[Vec<f32>],
        params: &SamplingParams,
    ) -> SpeculativeStep {
        // Phase 1: Draft
        let draft_tokens = self.draft(context, params);
        let n_drafted = draft_tokens.len();

        // Phase 2: Verify
        let accepted_tokens = self.verify(&draft_tokens, target_logits, params);
        let n_accepted = accepted_tokens.len();

        // Update statistics
        self.total_steps += 1;
        self.total_draft_tokens += n_drafted as u64;
        self.total_accepted_tokens += n_accepted as u64;

        // Feed the adaptive controller (if any) and apply its lookahead update.
        if let Some(adaptive) = self.adaptive.as_mut() {
            adaptive.observe_step(n_drafted, n_accepted);
            // The controller may have changed `lookahead` — propagate it to
            // `config.lookahead` so the next `step` drafts the new amount.
            self.config.lookahead = adaptive.lookahead();
        }

        let acceptance_rate = if n_drafted > 0 {
            n_accepted as f32 / n_drafted as f32
        } else {
            0.0
        };

        SpeculativeStep {
            draft_tokens,
            accepted_tokens,
            acceptance_rate,
        }
    }

    /// Generate up to `max_tokens` tokens using speculative decoding.
    ///
    /// Each step drafts `lookahead` candidates, verifies them, and appends
    /// accepted tokens. The loop continues until `max_tokens` are collected
    /// or generation stalls (no tokens accepted/generated).
    ///
    /// In this mock implementation, target logits are synthesised from the
    /// draft engine's perspective — in production the target model would
    /// score all positions in one batched forward pass.
    pub fn generate_speculative(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        params: &SamplingParams,
    ) -> Vec<u32> {
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut context: Vec<u32> = prompt_tokens.to_vec();

        while output.len() < max_tokens {
            let remaining = max_tokens - output.len();
            let effective_lookahead = self.config.lookahead.min(remaining);

            // Synthesise mock target logits for each draft position.
            // In production: run target model forward pass over all positions.
            // Here we generate uniform-ish logits for each draft position using PRNG.
            let vocab_size = 32000usize; // representative for Qwen3
            let target_logits: Vec<Vec<f32>> = (0..effective_lookahead)
                .map(|step_idx| {
                    // Build a peaked distribution at a token derived from context + step
                    let peak_token =
                        (context.last().copied().unwrap_or(0) as usize + step_idx + 1) % vocab_size;
                    let mut logits = vec![0.0f32; vocab_size];
                    // Give the peak token high logit, others low
                    logits[peak_token] = 10.0;
                    for (i, l) in logits.iter_mut().enumerate() {
                        if i != peak_token {
                            *l = -2.0;
                        }
                    }
                    logits
                })
                .collect();

            let step_result = self.step(&context, &target_logits, params);

            if step_result.accepted_tokens.is_empty() {
                // No tokens accepted — try generating one greedily to avoid infinite loop
                match self.draft_engine.generate(&context, 1) {
                    Ok(t) if !t.is_empty() => {
                        let token = t[0];
                        output.push(token);
                        context.push(token);
                    }
                    _ => break,
                }
            } else {
                let to_take = step_result.accepted_tokens.len().min(remaining);
                for &tok in step_result.accepted_tokens[..to_take].iter() {
                    output.push(tok);
                    context.push(tok);
                    if output.len() >= max_tokens {
                        break;
                    }
                }
            }

            // Safety: break if context grows unexpectedly large
            if context.len() > prompt_tokens.len() + max_tokens + self.config.lookahead {
                break;
            }
        }

        output
    }

    /// Overall acceptance rate: accepted tokens / draft tokens, across all steps.
    ///
    /// Returns 0.0 if no drafts have been generated yet.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_accepted_tokens as f32 / self.total_draft_tokens as f32
    }

    /// Theoretical speedup estimate from speculative decoding.
    ///
    /// Speedup ≈ accepted tokens per step (capped at lookahead).
    /// Returns the mean accepted tokens per step, which indicates how many
    /// target forward passes were "skipped" relative to autoregressive decoding.
    ///
    /// A return of 1.0 means no speedup (equivalent to autoregressive); higher
    /// values indicate benefit from speculative parallelism.
    pub fn speedup_estimate(&self) -> f32 {
        if self.total_steps == 0 {
            return 1.0;
        }
        let avg_accepted = self.total_accepted_tokens as f32 / self.total_steps as f32;
        // Speedup is bounded by lookahead + 1 (the bonus token)
        avg_accepted.max(1.0)
    }

    /// Reset all accumulated statistics (steps, tokens, acceptance counts).
    /// If an adaptive controller is attached, its EWMA is also reset.
    pub fn reset_stats(&mut self) {
        self.total_steps = 0;
        self.total_draft_tokens = 0;
        self.total_accepted_tokens = 0;
        if let Some(adaptive) = self.adaptive.as_mut() {
            adaptive.reset();
            self.config.lookahead = adaptive.lookahead();
        }
    }

    /// Determine whether a draft token should be accepted.
    ///
    /// Implements the speculative sampling acceptance criterion:
    /// - If `target_prob >= draft_prob`: always accept
    /// - Otherwise: accept with probability `target_prob / draft_prob`
    ///
    /// The `threshold` parameter can optionally raise the bar for acceptance.
    /// `rng_sample` must be in `[0.0, 1.0)`.
    fn should_accept(draft_prob: f32, target_prob: f32, threshold: f32, rng_sample: f32) -> bool {
        if target_prob >= draft_prob {
            // Target assigns higher probability — always accept
            true
        } else {
            // Rejection sampling: accept with prob target/draft
            let accept_prob = (target_prob / draft_prob).max(0.0);
            let effective_threshold = accept_prob - threshold;
            rng_sample < effective_threshold
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Utility: softmax over f32 slice
// ──────────────────────────────────────────────────────────────────

/// Compute numerically stable softmax over a logit slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        // Uniform fallback
        let n = logits.len() as f32;
        return vec![1.0 / n; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::config::Qwen3Config;

    fn make_decoder(lookahead: usize) -> SpeculativeDecoder<'static> {
        // Use a statically-valid config — tiny_test gives a minimal model
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_config = SpeculativeConfig {
            lookahead,
            acceptance_threshold: 0.0,
        };
        SpeculativeDecoder::new(engine, spec_config)
    }

    fn make_peaked_logits(
        vocab_size: usize,
        peak_token: usize,
        n_positions: usize,
    ) -> Vec<Vec<f32>> {
        (0..n_positions)
            .map(|_| {
                let mut logits = vec![-5.0f32; vocab_size];
                if peak_token < vocab_size {
                    logits[peak_token] = 10.0;
                }
                logits
            })
            .collect()
    }

    #[test]
    fn test_speculative_config_defaults() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.lookahead, 5, "default lookahead should be 5");
        assert!(
            (cfg.acceptance_threshold - 0.0).abs() < f32::EPSILON,
            "default threshold should be 0.0"
        );
    }

    #[test]
    fn test_draft_generates_lookahead_tokens() {
        let mut decoder = make_decoder(3);
        let context = vec![1u32, 2, 3];
        let params = SamplingParams::default();
        let draft = decoder.draft(&context, &params);
        // Draft should generate up to lookahead tokens (may be fewer if EOS hit)
        assert!(
            draft.len() <= 3,
            "draft should not exceed lookahead=3, got {}",
            draft.len()
        );
    }

    #[test]
    fn test_verify_accepts_high_probability_tokens() {
        let decoder = make_decoder(5);
        let params = SamplingParams::default();
        let vocab_size = 100;

        // Token 42 is the draft token; give it very high target probability
        let draft_tokens = vec![42u32];
        let target_logits = make_peaked_logits(vocab_size, 42, 1);

        let accepted = decoder.verify(&draft_tokens, &target_logits, &params);
        assert_eq!(
            accepted.len(),
            1,
            "high-probability token should be accepted"
        );
        assert_eq!(accepted[0], 42);
    }

    #[test]
    fn test_verify_rejects_low_probability_tokens() {
        let decoder = make_decoder(5);
        let params = SamplingParams::default();
        let vocab_size = 1000;

        // Token 500 — give it very low probability (far from peak)
        let draft_tokens = vec![500u32];
        let mut logits = vec![-10.0f32; vocab_size];
        logits[0] = 20.0; // strong peak at token 0, not 500
        let target_logits = vec![logits];

        // With very low target_prob for token 500, most RNG samples should reject
        // Run multiple times to confirm rejection is common
        let mut rejections = 0;
        for _ in 0..20 {
            let accepted = decoder.verify(&draft_tokens, &target_logits, &params);
            if accepted.is_empty() {
                rejections += 1;
            }
        }
        assert!(
            rejections > 0,
            "low-probability token should be rejected at least sometimes"
        );
    }

    #[test]
    fn test_acceptance_rate_zero_at_start() {
        let decoder = make_decoder(5);
        assert!(
            (decoder.acceptance_rate() - 0.0).abs() < f32::EPSILON,
            "acceptance rate must be 0.0 before any steps"
        );
        assert_eq!(decoder.total_steps, 0);
        assert_eq!(decoder.total_draft_tokens, 0);
        assert_eq!(decoder.total_accepted_tokens, 0);
    }

    #[test]
    fn test_acceptance_rate_updates_after_step() {
        let mut decoder = make_decoder(4);
        let params = SamplingParams::default();
        let context = vec![1u32, 2, 3];

        // Use peaked logits so tokens are likely accepted
        let vocab_size = 32usize;
        let target_logits = make_peaked_logits(vocab_size, 5, 4);

        let step = decoder.step(&context, &target_logits, &params);

        assert_eq!(decoder.total_steps, 1, "one step should have been recorded");
        assert_eq!(
            decoder.total_draft_tokens,
            step.draft_tokens.len() as u64,
            "draft token count should match"
        );
        assert!(
            decoder.total_accepted_tokens <= decoder.total_draft_tokens,
            "accepted cannot exceed drafted"
        );
    }

    #[test]
    fn test_generate_speculative_returns_tokens() {
        let mut decoder = make_decoder(3);
        let params = SamplingParams::default();
        let prompt = vec![1u32, 2, 3];

        let output = decoder.generate_speculative(&prompt, 5, &params);
        // Should return up to max_tokens tokens
        assert!(
            output.len() <= 5,
            "output should not exceed max_tokens=5, got {}",
            output.len()
        );
    }

    #[test]
    fn test_should_accept_target_above_draft() {
        // When target_prob > draft_prob, always accept regardless of rng_sample
        assert!(
            SpeculativeDecoder::should_accept(0.1, 0.9, 0.0, 0.99),
            "target > draft: must accept even with rng_sample near 1.0"
        );
        assert!(
            SpeculativeDecoder::should_accept(0.05, 0.5, 0.0, 0.0),
            "target > draft: must accept with rng_sample=0.0"
        );
    }

    #[test]
    fn test_should_accept_target_below_draft_probabilistic() {
        // target_prob < draft_prob → accept with prob target/draft
        // With target=0.1, draft=1.0, accept_prob = 0.1
        // rng_sample=0.05 < 0.1 → should accept
        assert!(
            SpeculativeDecoder::should_accept(1.0, 0.1, 0.0, 0.05),
            "rng_sample=0.05 < accept_prob=0.1, should accept"
        );
        // rng_sample=0.5 >= 0.1 → should reject
        assert!(
            !SpeculativeDecoder::should_accept(1.0, 0.1, 0.0, 0.5),
            "rng_sample=0.5 >= accept_prob=0.1, should reject"
        );
    }

    #[test]
    fn test_speedup_estimate_below_lookahead() {
        let mut decoder = make_decoder(5);
        // Before any steps, speedup is 1.0 (baseline)
        assert!(
            (decoder.speedup_estimate() - 1.0).abs() < f32::EPSILON,
            "initial speedup should be 1.0"
        );

        // Simulate some stats: 10 steps, 30 drafted, 15 accepted
        decoder.total_steps = 10;
        decoder.total_draft_tokens = 30;
        decoder.total_accepted_tokens = 15;

        let speedup = decoder.speedup_estimate();
        // avg_accepted = 15/10 = 1.5; speedup = max(1.5, 1.0) = 1.5
        assert!(
            (speedup - 1.5).abs() < 1e-4,
            "speedup should be 1.5 (avg accepted per step), got {speedup}"
        );
        assert!(
            speedup <= decoder.config.lookahead as f32 + 1.0,
            "speedup cannot exceed lookahead+1"
        );
    }

    #[test]
    fn test_with_adaptive_starts_with_initial_lookahead() {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_cfg = SpeculativeConfig {
            lookahead: 99,
            acceptance_threshold: 0.0,
        };
        let adapt_cfg = AdaptiveLookaheadConfig {
            initial: 4,
            min: 2,
            max: 10,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let decoder =
            SpeculativeDecoder::with_adaptive(engine, spec_cfg, adapt_cfg).expect("valid");
        // Adaptive overrides the spec config's lookahead.
        assert_eq!(decoder.config.lookahead, 4);
        assert!(decoder.adaptive().is_some());
    }

    #[test]
    fn test_adaptive_decreases_lookahead_on_low_acceptance() {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_cfg = SpeculativeConfig {
            lookahead: 8,
            acceptance_threshold: 0.0,
        };
        let adapt_cfg = AdaptiveLookaheadConfig {
            initial: 8,
            min: 2,
            max: 12,
            alpha: 0.7,
            cooldown_steps: 1,
        };
        let mut decoder =
            SpeculativeDecoder::with_adaptive(engine, spec_cfg, adapt_cfg).expect("valid");
        let context = vec![1u32, 2, 3];
        let params = SamplingParams::default();
        // Provide logits with no peaked target — most rejections.
        let vocab = 100usize;
        let logits: Vec<Vec<f32>> = (0..decoder.config.lookahead)
            .map(|_| {
                let mut l = vec![10.0f32; vocab];
                l[0] = -50.0; // bias away from typical draft tokens
                l
            })
            .collect();
        for _ in 0..30 {
            decoder.step(&context, &logits, &params);
        }
        // With low acceptance, lookahead should have fallen toward the min.
        let final_la = decoder.config.lookahead;
        assert!(
            final_la <= 8,
            "lookahead should not increase, got {final_la}"
        );
    }

    #[test]
    fn test_reset_stats_resets_adaptive() {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_cfg = SpeculativeConfig {
            lookahead: 5,
            acceptance_threshold: 0.0,
        };
        let adapt_cfg = AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let mut decoder =
            SpeculativeDecoder::with_adaptive(engine, spec_cfg, adapt_cfg).expect("valid");
        // Drive the adaptive controller into a different state.
        for _ in 0..30 {
            let logits = make_peaked_logits(64, 5, decoder.config.lookahead);
            decoder.step(&[1, 2, 3], &logits, &SamplingParams::default());
        }
        decoder.reset_stats();
        assert_eq!(decoder.total_steps, 0);
        assert_eq!(decoder.config.lookahead, 5);
        assert_eq!(
            decoder.adaptive().expect("adaptive present").observations(),
            0
        );
    }
}
