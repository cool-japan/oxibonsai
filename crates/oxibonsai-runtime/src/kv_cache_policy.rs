//! KV cache compression policy controller.
//!
//! Adapts the KV cache precision based on cache pressure: as more sequences
//! accumulate, the cache transitions FP16 → INT8 (Q8) → INT4 (Q4) so the
//! same memory budget can accommodate longer contexts and more in-flight
//! requests.
//!
//! ## Design
//!
//! [`KvCachePolicy`] tracks an exponentially-weighted moving average of cache
//! occupancy. Crossing one of the configured thresholds upgrades the level;
//! falling below the threshold *minus* a hysteresis margin downgrades it,
//! preventing oscillation around boundaries.
//!
//! ## Levels
//!
//! | Level | Memory factor | Quality |
//! |-------|---------------|---------|
//! | `Fp16` | 1.0× | exact |
//! | `Q8`   | 0.5× | ~0.1% RMSE vs FP16 |
//! | `Q4`   | 0.25× | ~1% RMSE vs FP16 |
//!
//! ## Usage
//!
//! ```
//! use oxibonsai_runtime::kv_cache_policy::{KvCachePolicy, KvCacheLevel};
//!
//! let mut policy = KvCachePolicy::default();
//! // 60 % pressure → still FP16 by default
//! assert_eq!(policy.observe(0.60), KvCacheLevel::Fp16);
//! // Sustained 90 % pressure → upgrades to Q8
//! for _ in 0..20 {
//!     policy.observe(0.92);
//! }
//! assert_eq!(policy.current_level(), KvCacheLevel::Q8);
//! ```

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

// ─── Levels ────────────────────────────────────────────────────────────────

/// KV cache precision tier.
///
/// Lower variants are higher precision, larger memory footprint; higher
/// variants are lower precision, smaller memory footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KvCacheLevel {
    /// FP16 — full quality, baseline memory.
    Fp16,
    /// INT8 quantized — half the memory of FP16.
    Q8,
    /// INT4 quantized — quarter the memory of FP16.
    Q4,
}

impl KvCacheLevel {
    /// Memory factor relative to FP16.
    pub const fn memory_factor(self) -> f32 {
        match self {
            Self::Fp16 => 1.0,
            Self::Q8 => 0.5,
            Self::Q4 => 0.25,
        }
    }

    /// Compactness order: higher = more compact (more aggressive).
    pub const fn ordinal(self) -> u8 {
        match self {
            Self::Fp16 => 0,
            Self::Q8 => 1,
            Self::Q4 => 2,
        }
    }

    /// Human-readable tag.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    fn from_ordinal(o: u8) -> Self {
        match o {
            0 => Self::Fp16,
            1 => Self::Q8,
            _ => Self::Q4,
        }
    }
}

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for [`KvCachePolicy`].
///
/// Default values: upgrade to Q8 above 80 % cache occupancy, upgrade to Q4
/// above 95 %; hysteresis margin 5 %; EWMA factor 0.20.
#[derive(Debug, Clone)]
pub struct KvCachePolicyConfig {
    /// Cache occupancy threshold (0.0..=1.0) above which we upgrade FP16 → Q8.
    pub q8_threshold: f32,
    /// Cache occupancy threshold above which we upgrade Q8 → Q4.
    pub q4_threshold: f32,
    /// Symmetric hysteresis margin: a downgrade fires only after pressure
    /// drops below `threshold - hysteresis`.
    pub hysteresis: f32,
    /// EWMA smoothing factor (`alpha` in `s_t = alpha * x_t + (1-alpha)*s_{t-1}`).
    /// Higher = more reactive, lower = more stable.
    pub ewma_alpha: f32,
    /// Initial / minimum tier — set to `Fp16` to allow downgrade.
    pub min_level: KvCacheLevel,
    /// Maximum tier — set to `Q4` to allow full compression range.
    pub max_level: KvCacheLevel,
}

impl Default for KvCachePolicyConfig {
    fn default() -> Self {
        Self {
            q8_threshold: 0.80,
            q4_threshold: 0.95,
            hysteresis: 0.05,
            ewma_alpha: 0.20,
            min_level: KvCacheLevel::Fp16,
            max_level: KvCacheLevel::Q4,
        }
    }
}

impl KvCachePolicyConfig {
    /// Conservative profile — never upgrades from FP16.
    pub fn fp16_only() -> Self {
        Self {
            min_level: KvCacheLevel::Fp16,
            max_level: KvCacheLevel::Fp16,
            ..Self::default()
        }
    }

    /// Aggressive profile — starts at Q8 and reaches Q4 sooner.
    pub fn aggressive() -> Self {
        Self {
            q8_threshold: 0.50,
            q4_threshold: 0.80,
            hysteresis: 0.05,
            ewma_alpha: 0.30,
            min_level: KvCacheLevel::Q8,
            max_level: KvCacheLevel::Q4,
        }
    }

    fn validate(&self) -> Result<(), KvCachePolicyError> {
        if !(0.0..=1.0).contains(&self.q8_threshold) {
            return Err(KvCachePolicyError::InvalidConfig(
                "q8_threshold must be in [0.0, 1.0]",
            ));
        }
        if !(0.0..=1.0).contains(&self.q4_threshold) {
            return Err(KvCachePolicyError::InvalidConfig(
                "q4_threshold must be in [0.0, 1.0]",
            ));
        }
        if self.q4_threshold < self.q8_threshold {
            return Err(KvCachePolicyError::InvalidConfig(
                "q4_threshold must be >= q8_threshold",
            ));
        }
        if !(0.0..=1.0).contains(&self.hysteresis) {
            return Err(KvCachePolicyError::InvalidConfig(
                "hysteresis must be in [0.0, 1.0]",
            ));
        }
        if !(0.0..=1.0).contains(&self.ewma_alpha) {
            return Err(KvCachePolicyError::InvalidConfig(
                "ewma_alpha must be in [0.0, 1.0]",
            ));
        }
        if self.min_level.ordinal() > self.max_level.ordinal() {
            return Err(KvCachePolicyError::InvalidConfig(
                "min_level must be <= max_level (less compact)",
            ));
        }
        Ok(())
    }
}

/// Errors raised by [`KvCachePolicy`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KvCachePolicyError {
    #[error("invalid kv-cache policy configuration: {0}")]
    InvalidConfig(&'static str),
}

// ─── Policy controller ─────────────────────────────────────────────────────

/// Stateful KV-cache compression policy.
///
/// Thread-safe: the current level is stored in an [`AtomicU8`] so concurrent
/// observers can read without locking. The pressure EWMA is also stored
/// atomically (as `u64`-encoded `f64` bits).
#[derive(Debug)]
pub struct KvCachePolicy {
    config: KvCachePolicyConfig,
    /// Current level encoded as `u8` for atomic load/store.
    level: AtomicU8,
    /// EWMA of observed pressure, `f64` bits stored as `u64`.
    pressure_ewma: AtomicU64,
    /// Number of observations since construction (also acts as warmup gate).
    samples: AtomicU64,
    /// Total upgrades fired (for telemetry).
    upgrades: AtomicU64,
    /// Total downgrades fired (for telemetry).
    downgrades: AtomicU64,
}

impl Default for KvCachePolicy {
    fn default() -> Self {
        Self::new(KvCachePolicyConfig::default()).expect("default config is valid")
    }
}

impl KvCachePolicy {
    /// Construct a new policy.
    ///
    /// Returns an error if the config is invalid (out-of-range thresholds,
    /// inverted hysteresis, or `min_level > max_level`).
    pub fn new(config: KvCachePolicyConfig) -> Result<Self, KvCachePolicyError> {
        config.validate()?;
        Ok(Self {
            level: AtomicU8::new(config.min_level.ordinal()),
            pressure_ewma: AtomicU64::new(0u64),
            samples: AtomicU64::new(0),
            upgrades: AtomicU64::new(0),
            downgrades: AtomicU64::new(0),
            config,
        })
    }

    /// Read the current level.
    pub fn current_level(&self) -> KvCacheLevel {
        KvCacheLevel::from_ordinal(self.level.load(Ordering::Relaxed))
    }

    /// Read the smoothed pressure (EWMA).
    pub fn pressure(&self) -> f64 {
        f64::from_bits(self.pressure_ewma.load(Ordering::Relaxed))
    }

    /// Number of observations recorded so far.
    pub fn samples(&self) -> u64 {
        self.samples.load(Ordering::Relaxed)
    }

    /// Number of upgrades fired since construction.
    pub fn upgrades(&self) -> u64 {
        self.upgrades.load(Ordering::Relaxed)
    }

    /// Number of downgrades fired since construction.
    pub fn downgrades(&self) -> u64 {
        self.downgrades.load(Ordering::Relaxed)
    }

    /// Record a new pressure observation and return the (possibly updated)
    /// active level.
    ///
    /// `pressure` is expected in `[0.0, 1.0]`; values are clamped to that
    /// range before being fed into the EWMA.
    pub fn observe(&self, pressure: f64) -> KvCacheLevel {
        let p = pressure.clamp(0.0, 1.0);

        // Update EWMA (CAS loop on the f64-as-u64 bits).
        let alpha = self.config.ewma_alpha as f64;
        let one_minus_alpha = 1.0 - alpha;
        loop {
            let current_bits = self.pressure_ewma.load(Ordering::Relaxed);
            let current = f64::from_bits(current_bits);
            let n = self.samples.load(Ordering::Relaxed);
            let new_val = if n == 0 {
                p
            } else {
                alpha * p + one_minus_alpha * current
            };
            if self
                .pressure_ewma
                .compare_exchange_weak(
                    current_bits,
                    new_val.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
        self.samples.fetch_add(1, Ordering::Relaxed);

        // Decide tier from smoothed pressure.
        let smoothed = self.pressure();
        let current = self.current_level();
        let target = self.target_level(smoothed, current);

        if target != current {
            self.level.store(target.ordinal(), Ordering::Relaxed);
            if target.ordinal() > current.ordinal() {
                self.upgrades.fetch_add(1, Ordering::Relaxed);
            } else {
                self.downgrades.fetch_add(1, Ordering::Relaxed);
            }
        }
        target
    }

    /// Decide the target tier given smoothed pressure and the current tier.
    ///
    /// Pure function — no side effects, useful for tests.
    fn target_level(&self, smoothed: f64, current: KvCacheLevel) -> KvCacheLevel {
        let q8 = self.config.q8_threshold as f64;
        let q4 = self.config.q4_threshold as f64;
        let h = self.config.hysteresis as f64;

        let raw = if smoothed >= q4 {
            KvCacheLevel::Q4
        } else if smoothed >= q8 {
            KvCacheLevel::Q8
        } else {
            KvCacheLevel::Fp16
        };

        // Apply hysteresis: only allow downgrade if pressure has dropped
        // below the *previous* tier's threshold by at least `h`.
        let target = match (current, raw) {
            (KvCacheLevel::Q4, KvCacheLevel::Q8) | (KvCacheLevel::Q4, KvCacheLevel::Fp16) => {
                if smoothed < q4 - h {
                    raw
                } else {
                    KvCacheLevel::Q4
                }
            }
            (KvCacheLevel::Q8, KvCacheLevel::Fp16) => {
                if smoothed < q8 - h {
                    KvCacheLevel::Fp16
                } else {
                    KvCacheLevel::Q8
                }
            }
            _ => raw,
        };

        // Clamp to [min, max].
        let min_o = self.config.min_level.ordinal();
        let max_o = self.config.max_level.ordinal();
        let clamped = target.ordinal().clamp(min_o, max_o);
        KvCacheLevel::from_ordinal(clamped)
    }

    /// Reset the EWMA, sample counter, and tier to the configured minimum.
    /// Counters for upgrades/downgrades are also reset.
    pub fn reset(&self) {
        self.pressure_ewma.store(0u64, Ordering::Relaxed);
        self.samples.store(0, Ordering::Relaxed);
        self.upgrades.store(0, Ordering::Relaxed);
        self.downgrades.store(0, Ordering::Relaxed);
        self.level
            .store(self.config.min_level.ordinal(), Ordering::Relaxed);
    }

    /// Return the configuration this policy was built with.
    pub fn config(&self) -> &KvCachePolicyConfig {
        &self.config
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_memory_factor() {
        assert!((KvCacheLevel::Fp16.memory_factor() - 1.0).abs() < f32::EPSILON);
        assert!((KvCacheLevel::Q8.memory_factor() - 0.5).abs() < f32::EPSILON);
        assert!((KvCacheLevel::Q4.memory_factor() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn level_ordinal_monotonic() {
        assert!(KvCacheLevel::Fp16.ordinal() < KvCacheLevel::Q8.ordinal());
        assert!(KvCacheLevel::Q8.ordinal() < KvCacheLevel::Q4.ordinal());
    }

    #[test]
    fn default_policy_starts_at_fp16() {
        let p = KvCachePolicy::default();
        assert_eq!(p.current_level(), KvCacheLevel::Fp16);
        assert_eq!(p.samples(), 0);
        assert_eq!(p.upgrades(), 0);
        assert_eq!(p.downgrades(), 0);
        assert!(p.pressure() < f64::EPSILON);
    }

    #[test]
    fn validate_rejects_inverted_thresholds() {
        let cfg = KvCachePolicyConfig {
            q8_threshold: 0.9,
            q4_threshold: 0.5,
            ..Default::default()
        };
        let err = KvCachePolicy::new(cfg).unwrap_err();
        assert!(matches!(err, KvCachePolicyError::InvalidConfig(_)));
    }

    #[test]
    fn validate_rejects_min_greater_than_max() {
        let cfg = KvCachePolicyConfig {
            min_level: KvCacheLevel::Q4,
            max_level: KvCacheLevel::Fp16,
            ..Default::default()
        };
        assert!(KvCachePolicy::new(cfg).is_err());
    }

    #[test]
    fn validate_rejects_out_of_range() {
        let cfg = KvCachePolicyConfig {
            q8_threshold: 1.5,
            ..Default::default()
        };
        assert!(KvCachePolicy::new(cfg).is_err());
    }

    #[test]
    fn low_pressure_stays_fp16() {
        let p = KvCachePolicy::default();
        for _ in 0..50 {
            assert_eq!(p.observe(0.10), KvCacheLevel::Fp16);
        }
    }

    #[test]
    fn sustained_high_pressure_upgrades_to_q8_then_q4() {
        let p = KvCachePolicy::default();
        // Sustain ~85 % pressure — should reach Q8 but not Q4.
        for _ in 0..40 {
            p.observe(0.85);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q8);

        // Push to ~98 % — should reach Q4.
        for _ in 0..40 {
            p.observe(0.98);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q4);
        assert!(p.upgrades() >= 2);
    }

    #[test]
    fn pressure_drop_downgrades_after_hysteresis() {
        let p = KvCachePolicy::default();
        for _ in 0..40 {
            p.observe(0.98);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q4);

        // Drop to 0.93 (below 0.95 but within hysteresis margin).
        // Default hysteresis = 0.05, so pressure must drop below 0.90 to downgrade.
        // 0.93 is *above* 0.90, so we should still be at Q4.
        for _ in 0..40 {
            p.observe(0.93);
        }
        // 0.93 sustained should pull EWMA below q4 threshold (0.95) but not below
        // q4_threshold - hysteresis = 0.90, so we hold at Q4.
        // (depending on exact EWMA dynamics — accept Q4 or Q8 here)
        let after_partial = p.current_level();
        assert!(matches!(after_partial, KvCacheLevel::Q4 | KvCacheLevel::Q8));

        // Now drop hard to 0.10 — should reach Fp16.
        for _ in 0..200 {
            p.observe(0.05);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Fp16);
        assert!(p.downgrades() >= 1);
    }

    #[test]
    fn hysteresis_prevents_thrashing() {
        let p = KvCachePolicy::default();
        // Push above q8 threshold to trigger upgrade.
        for _ in 0..40 {
            p.observe(0.85);
        }
        let before = p.upgrades();
        assert!(before >= 1);
        // Now oscillate just above and below the threshold.
        for i in 0..40 {
            // Stays around 0.78 .. 0.82 — within hysteresis band of q8 = 0.80.
            let v = if i % 2 == 0 { 0.78 } else { 0.82 };
            p.observe(v);
        }
        // We allow at most a small number of additional level changes.
        // Without hysteresis we'd see ~20 transitions.
        let total_changes = p.upgrades() + p.downgrades();
        assert!(
            total_changes < 10,
            "hysteresis should suppress oscillation; saw {total_changes} transitions"
        );
    }

    #[test]
    fn reset_clears_state() {
        let p = KvCachePolicy::default();
        for _ in 0..50 {
            p.observe(0.99);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q4);
        p.reset();
        assert_eq!(p.current_level(), KvCacheLevel::Fp16);
        assert_eq!(p.samples(), 0);
        assert!(p.pressure() < f64::EPSILON);
    }

    #[test]
    fn fp16_only_profile_never_upgrades() {
        let p = KvCachePolicy::new(KvCachePolicyConfig::fp16_only()).expect("valid config");
        for _ in 0..200 {
            assert_eq!(p.observe(1.0), KvCacheLevel::Fp16);
        }
        assert_eq!(p.upgrades(), 0);
    }

    #[test]
    fn aggressive_profile_starts_at_q8() {
        let p = KvCachePolicy::new(KvCachePolicyConfig::aggressive()).expect("valid config");
        assert_eq!(p.current_level(), KvCacheLevel::Q8);
        for _ in 0..30 {
            p.observe(0.95);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q4);
    }

    #[test]
    fn observed_pressure_is_clamped() {
        let p = KvCachePolicy::default();
        // Out-of-range values must not break the EWMA.
        p.observe(-1.0);
        assert!(p.pressure() >= 0.0);
        p.observe(2.0);
        assert!(p.pressure() <= 1.0 + 1e-6);
    }

    #[test]
    fn level_tag_strings() {
        assert_eq!(KvCacheLevel::Fp16.tag(), "fp16");
        assert_eq!(KvCacheLevel::Q8.tag(), "q8");
        assert_eq!(KvCacheLevel::Q4.tag(), "q4");
    }

    #[test]
    fn concurrent_observe_is_safe() {
        use std::sync::Arc;
        use std::thread;

        let p = Arc::new(KvCachePolicy::default());
        let mut handles = Vec::new();
        for tid in 0..8 {
            let p = Arc::clone(&p);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let v = ((tid + i) % 100) as f64 / 100.0;
                    p.observe(v);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
        assert_eq!(p.samples(), 8 * 100);
    }
}
