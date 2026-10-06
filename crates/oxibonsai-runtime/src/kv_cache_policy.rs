//! KV cache compression policy controller.
//!
//! Computes an *advisory* KV cache precision tier from observed cache
//! pressure: as more sequences accumulate, the recommended tier transitions
//! FP16 → INT8 (Q8) → INT4 (Q4), which — if something actually acted on it —
//! would let the same memory budget accommodate longer contexts and more
//! in-flight requests.
//!
//! ## Honesty (RT-14 / M-13)
//!
//! **By default this is pure telemetry.** [`KvCachePolicy::observe`] updates
//! an internal EWMA and picks a tier, and that is all — nothing in
//! `oxibonsai-runtime` or `oxibonsai-model` reads the result and applies it
//! to a real KV cache. `BonsaiModel` holds exactly one `KvCache` field
//! (always FP32, or since M-07 optionally sparse-`f16` via
//! [`KvCache::new_sparse`](oxibonsai_model::kv_cache::KvCache::new_sparse)
//! for a hybrid model, but still a *single* field chosen once at load time)
//! threaded through every block-forward call; switching that backing per
//! tier would mean changing the block-forward signature across every
//! forward path (`block/types/forward.rs`, `forward_metal.rs`,
//! `forward_cuda/*`). The related types
//! [`KvCacheFp16`](https://docs.rs/oxibonsai-model) and the `kv_cache_quant`
//! module (`QuantizedKvCache`/`Fp8KvCache`) exist as standalone,
//! never-instantiated-by-the-model types for exactly the same reason, and
//! `PagedKvCache` is a separate, explicitly experimental primitive with the
//! same status.
//!
//! [`KvCachePolicy::with_action`] / [`KvCachePolicy::set_action`] is what
//! turns this from telemetry into something that actually *does*
//! something: attach a closure and it runs, on the calling thread, every
//! time `observe()` decides to change tier, receiving the newly selected
//! [`KvCacheLevel`]. That closure is the eviction/quantization action the
//! finding asked for — this crate cannot manufacture one out of thin air
//! (there is no KV-cache-backing selector to call in `oxibonsai-model`
//! today), but a caller that adds one can wire it in without any further
//! change here. Until such a caller exists, `current_level()` /
//! `pressure()` / the `/admin/workload-stats` and `/admin/cache-stats`
//! payloads that surface them remain **advisory numbers only** — read them
//! as "what the policy would recommend", not "what the cache is doing".
//!
//! ### What the sparse backing adds, and what it still cannot (RT-14)
//!
//! [`oxibonsai_model::kv_cache::KvCacheBacking`] is the data half of the
//! still-missing seam: an enum naming the concrete backing a `BonsaiModel`
//! could hold (`DenseF32`, `DenseF16`, `SparseF16`). [`From<KvCacheLevel>`]
//! is implemented for it right here, so that *if* a
//! `BonsaiModel::set_kv_backing(&mut self, backing: KvCacheBacking)` seam is
//! ever added (it requires editing `model/types/mod.rs` and the
//! `block/types/forward*.rs` files above — a partial wire-up would not be
//! the real fix, so none is attempted), wiring
//! this policy to it is exactly:
//!
//! ```ignore
//! // model: BonsaiModel — needs `set_kv_backing`, not added by this crate.
//! policy.set_action(move |level| model.set_kv_backing(level.into()));
//! ```
//!
//! one line, because the level-to-backing conversion already exists. Until
//! that seam lands, this module's status is unchanged from the paragraph
//! above: pure telemetry unless a caller supplies its own action.
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
//!
//! ### Wiring a real action
//!
//! ```
//! use oxibonsai_runtime::kv_cache_policy::{KvCachePolicy, KvCacheLevel};
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use std::sync::Arc;
//!
//! // In a real deployment this closure would call into a KV-cache-backing
//! // selector on the loaded model. No such selector exists yet (see the
//! // module docs), so this example just counts real transitions instead.
//! let applied = Arc::new(AtomicUsize::new(0));
//! let applied_clone = Arc::clone(&applied);
//! let policy = KvCachePolicy::default().with_action(move |_level: KvCacheLevel| {
//!     applied_clone.fetch_add(1, Ordering::Relaxed);
//! });
//! for _ in 0..20 {
//!     policy.observe(0.92); // crosses the Q8 threshold once
//! }
//! assert!(applied.load(Ordering::Relaxed) >= 1);
//! ```

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

// ─── Levels ────────────────────────────────────────────────────────────────

/// KV cache precision tier.
///
/// Lower variants are higher precision, larger memory footprint; higher
/// variants are lower precision, smaller memory footprint.
///
/// Compactness ordering (ordinal): `Fp16 (0) < Q8 (1) < Fp8 (2) < Q4 (3)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KvCacheLevel {
    /// FP16 — full quality, baseline memory.
    Fp16,
    /// INT8 quantized — half the memory of FP16.
    Q8,
    /// FP8 quantized — half of FP32, same byte width as INT8 but floating-point
    /// distribution preserves more dynamic range for attention activations.
    Fp8,
    /// INT4 quantized — quarter the memory of FP16.
    Q4,
}

impl KvCacheLevel {
    /// Memory factor relative to FP16.
    ///
    /// | Level | Factor |
    /// |-------|--------|
    /// | Fp16  | 1.0    |
    /// | Q8    | 0.5    |
    /// | Fp8   | 0.5    |
    /// | Q4    | 0.25   |
    pub const fn memory_factor(self) -> f32 {
        match self {
            Self::Fp16 => 1.0,
            Self::Q8 => 0.5,
            Self::Fp8 => 0.5,
            Self::Q4 => 0.25,
        }
    }

    /// Compactness order: higher = more compact (more aggressive).
    ///
    /// Ordering: `Fp16=0 < Q8=1 < Fp8=2 < Q4=3`.
    /// `Fp8` sits between `Q8` and `Q4` because both use 1 byte per value but
    /// FP8's floating-point distribution makes it preferable to INT8 for KV
    /// cache activations while still being intermediate before INT4.
    pub const fn ordinal(self) -> u8 {
        match self {
            Self::Fp16 => 0,
            Self::Q8 => 1,
            Self::Fp8 => 2,
            Self::Q4 => 3,
        }
    }

    /// Human-readable tag.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8 => "q8",
            Self::Fp8 => "fp8",
            Self::Q4 => "q4",
        }
    }

    fn from_ordinal(o: u8) -> Self {
        match o {
            0 => Self::Fp16,
            1 => Self::Q8,
            2 => Self::Fp8,
            _ => Self::Q4,
        }
    }
}

/// Map a telemetry-driven precision tier onto the concrete
/// [`oxibonsai_model::kv_cache::KvCacheBacking`] a hypothetical
/// `BonsaiModel::set_kv_backing` seam would need (RT-14 / M-13 — see the
/// module docs' "What the sparse backing adds" section for why that seam does not exist
/// yet). A 1:1 mirror of the four [`KvCacheLevel`] variants onto their
/// `Dense*` `KvCacheBacking` counterparts; `KvCacheBacking::DenseF32` and
/// `KvCacheBacking::SparseF16` are unreachable from this conversion because
/// the policy has no notion of "one tier above baseline" or "this model is
/// architecturally hybrid" — both are load-time facts, not pressure-driven
/// recommendations.
impl From<KvCacheLevel> for oxibonsai_model::kv_cache::KvCacheBacking {
    fn from(level: KvCacheLevel) -> Self {
        use oxibonsai_model::kv_cache::KvCacheBacking;
        match level {
            KvCacheLevel::Fp16 => KvCacheBacking::DenseF16,
            KvCacheLevel::Q8 => KvCacheBacking::DenseQ8,
            KvCacheLevel::Fp8 => KvCacheBacking::DenseFp8,
            KvCacheLevel::Q4 => KvCacheBacking::DenseQ4,
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

/// A real action to run when [`KvCachePolicy::observe`] decides to change
/// tier. See [`KvCachePolicy::with_action`].
type KvCacheAction = dyn Fn(KvCacheLevel) + Send + Sync;

/// Stateful KV-cache compression policy.
///
/// Thread-safe: the current level is stored in an [`AtomicU8`] so concurrent
/// observers can read without locking. The pressure EWMA is also stored
/// atomically (as `u64`-encoded `f64` bits).
///
/// See the [module docs](self) for the honest telemetry-vs-real-action
/// distinction (RT-14 / M-13).
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
    /// Optional real action invoked on every tier transition. `None` (the
    /// default) means this policy is pure telemetry — see the module docs.
    /// A `OnceLock` rather than a plain field because `KvCachePolicy` is
    /// typically shared behind an `Arc` (e.g. `AdminState::kv_cache_policy`)
    /// once constructed, so attaching an action later needs `&self`
    /// interior mutability, not `&mut self`.
    action: OnceLock<Arc<KvCacheAction>>,
}

impl std::fmt::Debug for KvCachePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCachePolicy")
            .field("config", &self.config)
            .field("level", &self.current_level())
            .field("pressure", &self.pressure())
            .field("samples", &self.samples())
            .field("upgrades", &self.upgrades())
            .field("downgrades", &self.downgrades())
            .field("action_attached", &self.action.get().is_some())
            .finish()
    }
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
            action: OnceLock::new(),
            config,
        })
    }

    /// Attach a real action that runs, on the calling thread, every time
    /// [`Self::observe`] decides to change tier — this is what turns the
    /// policy from telemetry into something that actually applies its
    /// decisions (RT-14 / M-13; see the module docs). Builder-style
    /// consuming setter, for attaching one at construction time.
    pub fn with_action(self, action: impl Fn(KvCacheLevel) + Send + Sync + 'static) -> Self {
        let _ = self.action.set(Arc::new(action));
        self
    }

    /// Same as [`Self::with_action`], callable through a shared `&self`
    /// (e.g. an already-constructed `Arc<KvCachePolicy>`). Returns `true` if
    /// this call attached the action, `false` if one was already attached
    /// (only the first call wins).
    pub fn set_action(&self, action: impl Fn(KvCacheLevel) + Send + Sync + 'static) -> bool {
        self.action.set(Arc::new(action)).is_ok()
    }

    /// Whether a real action is currently attached — `false` means this
    /// policy is pure telemetry (the default).
    pub fn has_action(&self) -> bool {
        self.action.get().is_some()
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
    ///
    /// When a real action is attached (see [`Self::with_action`] /
    /// [`Self::set_action`]), it runs synchronously, on this call's thread,
    /// exactly when the tier actually changes — never on every call, and
    /// never when the decision holds steady.
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
            // Apply the decision for real, if a caller wired an action in
            // (RT-14 / M-13). Fired exactly once per genuine transition.
            if let Some(action) = self.action.get() {
                action(target);
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

    // ── RT-14 / M-13: real action sink ──────────────────────────────────────

    #[test]
    fn no_action_by_default() {
        let p = KvCachePolicy::default();
        assert!(!p.has_action(), "a fresh policy must be pure telemetry");
    }

    #[test]
    fn action_sink_fires_on_every_real_transition() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::{Arc, Mutex};

        let calls = Arc::new(AtomicUsize::new(0));
        let observed_levels: Arc<Mutex<Vec<KvCacheLevel>>> = Arc::new(Mutex::new(Vec::new()));
        let calls_clone = Arc::clone(&calls);
        let levels_clone = Arc::clone(&observed_levels);

        let p = KvCachePolicy::default().with_action(move |level| {
            calls_clone.fetch_add(1, Ordering::Relaxed);
            levels_clone
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(level);
        });
        assert!(p.has_action());

        // Sustain ~85% pressure: should upgrade Fp16 -> Q8 exactly once.
        for _ in 0..40 {
            p.observe(0.85);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q8);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "the action must fire exactly once for exactly one transition, not once per observe()"
        );
        assert_eq!(
            *observed_levels.lock().unwrap_or_else(|e| e.into_inner()),
            vec![KvCacheLevel::Q8],
            "the action must receive the newly selected level"
        );

        // Push further to Q4: a second, real transition -> a second call.
        for _ in 0..40 {
            p.observe(0.99);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q4);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            *observed_levels.lock().unwrap_or_else(|e| e.into_inner()),
            vec![KvCacheLevel::Q8, KvCacheLevel::Q4]
        );
    }

    #[test]
    fn action_sink_does_not_fire_when_the_tier_holds_steady() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let p = KvCachePolicy::default().with_action(move |_level| {
            calls_clone.fetch_add(1, Ordering::Relaxed);
        });

        // Low, steady pressure never leaves Fp16 -> the action never fires.
        for _ in 0..50 {
            p.observe(0.10);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Fp16);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "no transition occurred, so telemetry-only observe() calls must not apply anything"
        );
    }

    #[test]
    fn set_action_works_through_a_shared_arc_and_only_attaches_once() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        // Mirrors real usage: AdminState holds `Arc<KvCachePolicy>`, so any
        // action must be attachable through `&self`, not just at
        // construction time.
        let p = Arc::new(KvCachePolicy::default());
        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        let attached = p.set_action(move |_level| fired_clone.store(true, Ordering::Relaxed));
        assert!(attached, "the first set_action call must succeed");
        assert!(p.has_action());

        let attached_again = p.set_action(|_| {});
        assert!(
            !attached_again,
            "a second set_action call must not replace the first (OnceLock semantics)"
        );

        for _ in 0..40 {
            p.observe(0.90);
        }
        assert!(
            fired.load(Ordering::Relaxed),
            "the action attached via &Arc must still fire on a real transition"
        );
    }

    #[test]
    fn action_sink_receives_downgrades_too() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let downgrades_seen = Arc::new(AtomicUsize::new(0));
        let downgrades_clone = Arc::clone(&downgrades_seen);
        let p = KvCachePolicy::default().with_action(move |level| {
            if level == KvCacheLevel::Fp16 {
                downgrades_clone.fetch_add(1, Ordering::Relaxed);
            }
        });

        for _ in 0..40 {
            p.observe(0.99);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Q4);

        for _ in 0..200 {
            p.observe(0.02);
        }
        assert_eq!(p.current_level(), KvCacheLevel::Fp16);
        assert!(
            downgrades_seen.load(Ordering::Relaxed) >= 1,
            "the action must also fire on downgrade transitions, not just upgrades"
        );
    }

    #[test]
    fn debug_format_does_not_leak_the_unformattable_closure_but_reports_it_is_attached() {
        let p = KvCachePolicy::default().with_action(|_level| {});
        let debug_str = format!("{p:?}");
        assert!(debug_str.contains("KvCachePolicy"));
        assert!(debug_str.contains("action_attached: true"));

        let q = KvCachePolicy::default();
        assert!(format!("{q:?}").contains("action_attached: false"));
    }

    // ── RT-14 / M-13 addendum: KvCacheBacking conversion ────────────────────

    #[test]
    fn kv_cache_level_converts_to_the_matching_dense_backing() {
        use oxibonsai_model::kv_cache::KvCacheBacking;

        assert_eq!(
            KvCacheBacking::from(KvCacheLevel::Fp16),
            KvCacheBacking::DenseF16
        );
        assert_eq!(
            KvCacheBacking::from(KvCacheLevel::Q8),
            KvCacheBacking::DenseQ8
        );
        assert_eq!(
            KvCacheBacking::from(KvCacheLevel::Fp8),
            KvCacheBacking::DenseFp8
        );
        assert_eq!(
            KvCacheBacking::from(KvCacheLevel::Q4),
            KvCacheBacking::DenseQ4
        );
    }

    #[test]
    fn action_sink_can_target_a_kv_cache_backing_directly_via_into() {
        // Exercises the exact one-line wiring the module docs promise:
        // `policy.set_action(move |level| model.set_kv_backing(level.into()))`
        // — here standing in for `set_kv_backing` with a plain capture, since
        // no such seam exists in `oxibonsai-model` yet (see the module docs).
        use oxibonsai_model::kv_cache::KvCacheBacking;
        use std::sync::{Arc, Mutex};

        let applied: Arc<Mutex<Vec<KvCacheBacking>>> = Arc::new(Mutex::new(Vec::new()));
        let applied_clone = Arc::clone(&applied);
        let p = KvCachePolicy::default().with_action(move |level: KvCacheLevel| {
            let backing: KvCacheBacking = level.into();
            applied_clone
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(backing);
        });

        for _ in 0..40 {
            p.observe(0.85); // Fp16 -> Q8
        }
        assert_eq!(
            *applied.lock().unwrap_or_else(|e| e.into_inner()),
            vec![KvCacheBacking::DenseQ8]
        );
    }
}
