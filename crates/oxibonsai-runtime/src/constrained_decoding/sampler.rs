//! [`ConstrainedSampler`] and its builder.
//!
//! Wraps a [`crate::sampling_advanced::SamplerChain`] with a
//! [`TokenConstraint`] and applies the mask to logits before sampling.

use super::error_trait::{NoConstraint, TokenConstraint};
use super::json::JsonConstraint;
use super::regex::RegexConstraint;
use crate::constrained_decoding::ConstraintError;

// ─────────────────────────────────────────────────────────────────────────────
// ConstrainedSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`crate::sampling_advanced::SamplerChain`] with a [`TokenConstraint`].
///
/// Before each sampling step the logits for disallowed tokens are masked to
/// `-1e9` so they are effectively excluded from the distribution.
pub struct ConstrainedSampler {
    inner: crate::sampling_advanced::SamplerChain,
    constraint: Box<dyn TokenConstraint>,
    generated: Vec<u32>,
    vocab_size: usize,
}

impl ConstrainedSampler {
    /// Create a new `ConstrainedSampler`.
    pub fn new(
        sampler: crate::sampling_advanced::SamplerChain,
        constraint: Box<dyn TokenConstraint>,
        vocab_size: usize,
    ) -> Self {
        Self {
            inner: sampler,
            constraint,
            generated: Vec::new(),
            vocab_size,
        }
    }

    /// Sample the next token, masking logits for disallowed tokens first.
    ///
    /// Steps:
    /// 1. Query the constraint for an allowed-token mask.
    /// 2. Set `logits[i] = -1e9` for every `false` entry in the mask.
    /// 3. Delegate to the inner sampler chain.
    /// 4. Call `constraint.advance(token)`.
    /// 5. Track the token in `self.generated`.
    pub fn sample(&mut self, logits: &mut Vec<f32>) -> u32 {
        // Apply constraint mask.
        if let Some(mask) = self
            .constraint
            .allowed_tokens(&self.generated, self.vocab_size)
        {
            for (i, allowed) in mask.iter().enumerate() {
                if i < logits.len() && !allowed {
                    logits[i] = -1e9;
                }
            }
        }
        let token = self.inner.sample(logits) as u32;
        self.constraint.advance(token);
        self.generated.push(token);
        token
    }

    /// Returns `true` if the constraint considers the current output complete.
    pub fn is_complete(&self) -> bool {
        self.constraint.is_complete()
    }

    /// Reset both the inner sampler state and the constraint.
    pub fn reset(&mut self) {
        self.generated.clear();
        self.constraint.reset();
    }

    /// Number of tokens generated so far.
    pub fn generated_text_len(&self) -> usize {
        self.generated.len()
    }

    /// Human-readable name of the active constraint.
    pub fn constraint_name(&self) -> &str {
        self.constraint.name()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstrainedSamplerBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Ergonomic builder for [`ConstrainedSampler`].
pub struct ConstrainedSamplerBuilder {
    vocab_size: usize,
    seed: u64,
}

impl ConstrainedSamplerBuilder {
    /// Create a new builder.
    pub fn new(vocab_size: usize, seed: u64) -> Self {
        Self { vocab_size, seed }
    }

    fn default_chain(&self) -> crate::sampling_advanced::SamplerChain {
        crate::sampling_advanced::SamplerChain::new(self.seed)
    }

    /// Build a `ConstrainedSampler` with a `JsonConstraint`.
    pub fn with_json_constraint(self) -> ConstrainedSampler {
        ConstrainedSampler::new(
            self.default_chain(),
            Box::new(JsonConstraint::new()),
            self.vocab_size,
        )
    }

    /// Build a `ConstrainedSampler` with a `RegexConstraint`.
    pub fn with_regex_constraint(
        self,
        pattern: &str,
    ) -> Result<ConstrainedSampler, ConstraintError> {
        let constraint = RegexConstraint::new(pattern)?;
        let chain = self.default_chain();
        Ok(ConstrainedSampler::new(
            chain,
            Box::new(constraint),
            self.vocab_size,
        ))
    }

    /// Build an unconstrained `ConstrainedSampler` (passthrough).
    pub fn unconstrained(self) -> ConstrainedSampler {
        ConstrainedSampler::new(
            self.default_chain(),
            Box::new(NoConstraint),
            self.vocab_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_sampler_masks_logits() {
        // vocab_size = 4; mask allows only tokens 0 and 2
        struct AllowEvens;
        impl TokenConstraint for AllowEvens {
            fn allowed_tokens(&self, _: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
                Some((0..vocab_size).map(|i| i % 2 == 0).collect())
            }
            fn advance(&mut self, _: u32) -> bool {
                true
            }
            fn is_complete(&self) -> bool {
                true
            }
            fn reset(&mut self) {}
            fn name(&self) -> &str {
                "AllowEvens"
            }
        }

        let chain = crate::sampling_advanced::SamplerChain::greedy();
        let mut sampler = ConstrainedSampler::new(chain, Box::new(AllowEvens), 4);
        // Make token 1 have highest logit; after masking token 0 should win.
        let mut logits = vec![2.0_f32, 10.0, 1.0, 0.5];
        // token 1 is masked → token 0 wins (highest among allowed)
        let tok = sampler.sample(&mut logits);
        assert_eq!(tok, 0);
    }

    #[test]
    fn constrained_sampler_greedy_json() {
        let chain = crate::sampling_advanced::SamplerChain::greedy();
        let mut sampler = ConstrainedSampler::new(chain, Box::new(JsonConstraint::new()), 256);
        assert!(!sampler.is_complete());
        // Feed '{' then '}'
        let mut logits_open = vec![0.0_f32; 256];
        logits_open['{' as usize] = 100.0;
        sampler.sample(&mut logits_open);

        let mut logits_close = vec![0.0_f32; 256];
        logits_close['}' as usize] = 100.0;
        sampler.sample(&mut logits_close);

        assert!(sampler.is_complete());
        assert_eq!(sampler.generated_text_len(), 2);
    }

    #[test]
    fn constrained_sampler_reset() {
        let chain = crate::sampling_advanced::SamplerChain::greedy();
        let mut sampler = ConstrainedSampler::new(chain, Box::new(JsonConstraint::new()), 256);
        let mut logits = vec![0.0_f32; 256];
        logits['{' as usize] = 100.0;
        sampler.sample(&mut logits);
        assert_eq!(sampler.generated_text_len(), 1);
        sampler.reset();
        assert_eq!(sampler.generated_text_len(), 0);
    }

    #[test]
    fn constrained_sampler_builder_json() {
        let sampler = ConstrainedSamplerBuilder::new(256, 42).with_json_constraint();
        assert_eq!(sampler.constraint_name(), "JsonConstraint");
    }

    #[test]
    fn constrained_sampler_builder_unconstrained() {
        let sampler = ConstrainedSamplerBuilder::new(256, 42).unconstrained();
        assert_eq!(sampler.constraint_name(), "NoConstraint");
        assert!(sampler.is_complete());
    }
}
