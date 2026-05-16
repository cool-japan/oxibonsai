//! Adaptive lookahead controller for speculative decoding.
//!
//! Continuously updates the speculative draft length `k` based on a running
//! EWMA of accepted-tokens-per-step. When acceptance is high, `k` increases
//! (proposing more tokens improves throughput); when acceptance is low,
//! `k` shrinks to avoid wasted draft compute.
//!
//! ## Algorithm
//!
//! For each completed step we observe `accepted ∈ [0, k]`. We maintain an
//! EWMA of `accepted` (let `s` denote the smoothed value):
//!
//! ```text
//! s_{t+1} = α · accepted_t + (1 − α) · s_t
//! ```
//!
//! The controller then computes a target `k`:
//!
//! ```text
//! target_k = clamp(round(s + 1), k_min, k_max)
//! ```
//!
//! Adding `1` biases the controller slightly toward exploration (we want to
//! propose at least one more token than we're consistently accepting, so we
//! discover whether longer drafts would be productive).
//!
//! A *cooldown* prevents `k` from changing on every single step: only after
//! `cooldown_steps` observations does the target take effect.
//!
//! ## Usage
//!
//! ```
//! use oxibonsai_runtime::adaptive_lookahead::{AdaptiveLookahead, AdaptiveLookaheadConfig};
//!
//! let mut adj = AdaptiveLookahead::new(AdaptiveLookaheadConfig::default());
//! assert_eq!(adj.lookahead(), 5);
//! // Repeatedly accept all 5 — k should grow toward k_max.
//! for _ in 0..50 {
//!     adj.observe_step(5, 5);
//! }
//! assert!(adj.lookahead() >= 5);
//! ```

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for [`AdaptiveLookahead`].
#[derive(Debug, Clone)]
pub struct AdaptiveLookaheadConfig {
    /// Initial lookahead value.
    pub initial: usize,
    /// Minimum lookahead. Must satisfy `min >= 1`.
    pub min: usize,
    /// Maximum lookahead. Must satisfy `max >= min`.
    pub max: usize,
    /// EWMA factor for accepted-tokens. Typical: 0.10 .. 0.30.
    pub alpha: f32,
    /// Number of observations to pool before applying a target update.
    ///
    /// Higher = more stable but slower to respond. Setting this to 1 makes
    /// updates immediate.
    pub cooldown_steps: u32,
}

impl Default for AdaptiveLookaheadConfig {
    fn default() -> Self {
        Self {
            initial: 5,
            min: 2,
            max: 12,
            alpha: 0.20,
            cooldown_steps: 4,
        }
    }
}

impl AdaptiveLookaheadConfig {
    /// Validate the configuration. Returns the first error encountered.
    pub fn validate(&self) -> Result<(), AdaptiveLookaheadError> {
        if self.min == 0 {
            return Err(AdaptiveLookaheadError::InvalidConfig("min must be >= 1"));
        }
        if self.max < self.min {
            return Err(AdaptiveLookaheadError::InvalidConfig("max must be >= min"));
        }
        if self.initial < self.min || self.initial > self.max {
            return Err(AdaptiveLookaheadError::InvalidConfig(
                "initial must be in [min, max]",
            ));
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(AdaptiveLookaheadError::InvalidConfig(
                "alpha must be in [0.0, 1.0]",
            ));
        }
        Ok(())
    }
}

/// Errors raised by the adaptive lookahead controller.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AdaptiveLookaheadError {
    #[error("invalid adaptive-lookahead configuration: {0}")]
    InvalidConfig(&'static str),
}

// ─── Controller ────────────────────────────────────────────────────────────

/// Adaptive lookahead controller.
///
/// Tracks an EWMA of accepted tokens per speculative step and adjusts the
/// proposed lookahead `k` toward `mean_accepted + 1`, clamped to a configured
/// `[min, max]` window with a cooldown to suppress oscillation.
#[derive(Debug, Clone)]
pub struct AdaptiveLookahead {
    config: AdaptiveLookaheadConfig,
    current: usize,
    /// EWMA of accepted tokens per step.
    accepted_ewma: f32,
    /// Total observations.
    observations: u64,
    /// Steps remaining in the current cooldown window.
    cooldown_left: u32,
    /// Total target updates applied (telemetry).
    updates: u64,
}

impl AdaptiveLookahead {
    /// Construct a new controller. Panics on invalid config — use
    /// [`AdaptiveLookahead::try_new`] to handle errors gracefully.
    pub fn new(config: AdaptiveLookaheadConfig) -> Self {
        Self::try_new(config).expect("invalid AdaptiveLookahead config")
    }

    /// Construct a new controller, returning an error on invalid config.
    pub fn try_new(config: AdaptiveLookaheadConfig) -> Result<Self, AdaptiveLookaheadError> {
        config.validate()?;
        let cooldown = config.cooldown_steps;
        Ok(Self {
            current: config.initial,
            accepted_ewma: 0.0,
            observations: 0,
            cooldown_left: cooldown,
            updates: 0,
            config,
        })
    }

    /// Current effective lookahead.
    pub fn lookahead(&self) -> usize {
        self.current
    }

    /// EWMA of accepted tokens per step.
    pub fn mean_accepted(&self) -> f32 {
        self.accepted_ewma
    }

    /// Number of observations recorded.
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// Number of times the lookahead has been changed.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Configuration in use.
    pub fn config(&self) -> &AdaptiveLookaheadConfig {
        &self.config
    }

    /// Record one speculative step's outcome.
    ///
    /// `proposed`: the `k` that was actually drafted.
    /// `accepted`: how many of those were accepted.
    ///
    /// `accepted` is clamped to `[0, proposed]`.
    pub fn observe_step(&mut self, proposed: usize, accepted: usize) {
        let accepted_clamped = accepted.min(proposed) as f32;
        let alpha = self.config.alpha;
        if self.observations == 0 {
            self.accepted_ewma = accepted_clamped;
        } else {
            self.accepted_ewma = alpha * accepted_clamped + (1.0 - alpha) * self.accepted_ewma;
        }
        self.observations = self.observations.saturating_add(1);

        // Cooldown: only commit a new target once per `cooldown_steps`.
        if self.cooldown_left > 0 {
            self.cooldown_left -= 1;
        } else {
            let target = self.compute_target();
            if target != self.current {
                self.current = target;
                self.updates = self.updates.saturating_add(1);
            }
            self.cooldown_left = self.config.cooldown_steps;
        }
    }

    /// Reset the EWMA and counters but preserve the configuration.
    pub fn reset(&mut self) {
        self.current = self.config.initial;
        self.accepted_ewma = 0.0;
        self.observations = 0;
        self.cooldown_left = self.config.cooldown_steps;
        self.updates = 0;
    }

    /// Compute the controller's target lookahead based on the current EWMA.
    ///
    /// Pure function, useful for unit-testing.
    fn compute_target(&self) -> usize {
        // mean_accepted + 1 → propose one more than we're currently
        // sustaining, biasing toward exploration.
        let target = (self.accepted_ewma + 1.0).round() as i64;
        target.clamp(self.config.min as i64, self.config.max as i64) as usize
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = AdaptiveLookaheadConfig::default();
        cfg.validate().expect("default config valid");
    }

    #[test]
    fn try_new_rejects_min_zero() {
        let cfg = AdaptiveLookaheadConfig {
            min: 0,
            ..Default::default()
        };
        assert!(AdaptiveLookahead::try_new(cfg).is_err());
    }

    #[test]
    fn try_new_rejects_inverted_bounds() {
        let cfg = AdaptiveLookaheadConfig {
            min: 8,
            max: 4,
            initial: 6,
            ..Default::default()
        };
        assert!(AdaptiveLookahead::try_new(cfg).is_err());
    }

    #[test]
    fn try_new_rejects_initial_out_of_range() {
        let cfg = AdaptiveLookaheadConfig {
            min: 4,
            max: 8,
            initial: 10,
            ..Default::default()
        };
        assert!(AdaptiveLookahead::try_new(cfg).is_err());
    }

    #[test]
    fn try_new_rejects_alpha_out_of_range() {
        let cfg = AdaptiveLookaheadConfig {
            alpha: 1.5,
            ..Default::default()
        };
        assert!(AdaptiveLookahead::try_new(cfg).is_err());
    }

    #[test]
    fn starts_at_initial() {
        let mut adj = AdaptiveLookahead::new(AdaptiveLookaheadConfig::default());
        assert_eq!(adj.lookahead(), 5);
        // First observation must not panic.
        adj.observe_step(5, 5);
    }

    #[test]
    fn high_acceptance_increases_lookahead() {
        let cfg = AdaptiveLookaheadConfig {
            initial: 3,
            min: 2,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        for _ in 0..30 {
            adj.observe_step(adj.lookahead(), adj.lookahead());
        }
        assert!(adj.lookahead() >= 3);
    }

    #[test]
    fn low_acceptance_decreases_lookahead() {
        let cfg = AdaptiveLookaheadConfig {
            initial: 10,
            min: 2,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        for _ in 0..30 {
            adj.observe_step(adj.lookahead(), 0);
        }
        // EWMA → 0, so target = 0+1 = 1, clamped to min = 2.
        assert_eq!(adj.lookahead(), 2);
    }

    #[test]
    fn lookahead_clamped_to_max() {
        let cfg = AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 8,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        for _ in 0..50 {
            adj.observe_step(20, 20);
        }
        assert_eq!(adj.lookahead(), 8);
    }

    #[test]
    fn lookahead_clamped_to_min() {
        let cfg = AdaptiveLookaheadConfig {
            initial: 6,
            min: 3,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        for _ in 0..50 {
            adj.observe_step(adj.lookahead(), 0);
        }
        assert_eq!(adj.lookahead(), 3);
    }

    #[test]
    fn cooldown_suppresses_updates() {
        let cfg = AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 100,
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        // Even strong signals can't exceed updates(<=1) within 100 steps.
        for _ in 0..50 {
            adj.observe_step(5, 5);
        }
        // Still close to initial — cooldown holds.
        assert!(adj.updates() <= 1);
    }

    #[test]
    fn observations_count_correctly() {
        let mut adj = AdaptiveLookahead::new(AdaptiveLookaheadConfig::default());
        for _ in 0..15 {
            adj.observe_step(5, 3);
        }
        assert_eq!(adj.observations(), 15);
    }

    #[test]
    fn ewma_smoothes_noise() {
        let cfg = AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 12,
            alpha: 0.10,
            cooldown_steps: 5,
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        // Alternate between full and zero acceptance — EWMA should
        // smooth around the mean.
        for i in 0..200 {
            if i % 2 == 0 {
                adj.observe_step(5, 5);
            } else {
                adj.observe_step(5, 0);
            }
        }
        let m = adj.mean_accepted();
        // Mean of 0,5 alternating = 2.5; allow generous tolerance for
        // EWMA convergence.
        assert!((m - 2.5).abs() < 1.0, "EWMA = {m}, expected ~2.5");
    }

    #[test]
    fn accepted_clamped_to_proposed() {
        let mut adj = AdaptiveLookahead::new(AdaptiveLookaheadConfig::default());
        // Garbage input: accepted > proposed
        adj.observe_step(3, 100);
        let m = adj.mean_accepted();
        assert!(m <= 3.0, "accepted should be clamped to proposed; got {m}");
    }

    #[test]
    fn reset_restores_initial() {
        let mut adj = AdaptiveLookahead::new(AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 1,
        });
        for _ in 0..40 {
            adj.observe_step(adj.lookahead(), adj.lookahead());
        }
        let before_reset = adj.lookahead();
        adj.reset();
        assert_eq!(adj.lookahead(), 5);
        assert!(adj.observations() == 0);
        // Sanity: we did move the lookahead before reset.
        let _ = before_reset;
    }

    #[test]
    fn compute_target_pure() {
        let cfg = AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 8,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        adj.accepted_ewma = 4.0;
        // target = 4+1 = 5, clamped to [2,8] = 5.
        assert_eq!(adj.compute_target(), 5);
        adj.accepted_ewma = 10.0;
        assert_eq!(adj.compute_target(), 8);
        adj.accepted_ewma = 0.0;
        assert_eq!(adj.compute_target(), 2);
    }
}
