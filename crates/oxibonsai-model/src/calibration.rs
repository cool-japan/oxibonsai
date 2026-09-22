//! Post-Training Quantization (PTQ) calibration pipeline.
//!
//! PTQ calibration:
//! 1. Run calibration data through the model (no gradient)
//! 2. Collect activation statistics per layer (min, max, percentiles)
//! 3. Compute optimal quantization scales
//! 4. Store scales for use during static-quantized inference
//!
//! Supported calibration methods:
//! - MinMax: scale = max(|x|) / clip_val (simple, fast)
//! - Percentile: scale = p99(|x|) / clip_val (robust to outliers)
//! - ACIQ: Analytical Clipping for Integer Quantization (Bell et al. 2019)
//!   optimal_clip = 2.83 * std_dev for normal distributions
//! - MSE: Mean squared error minimization (approximated via grid search)
//!
//! ## Reachability
//!
//! As of this writing, nothing in the CLI or the export/quantize pipeline
//! (`export.rs`, `quantize.rs`) constructs a [`CalibrationDb`] or reads its
//! scales — there is no `--calib <db>` (or equivalent) entry point yet that
//! would let a measured calibration pass influence a real quantization run.
//! This module is a tested, documented primitive for that future work, not
//! a wired feature.

use std::collections::HashMap;

// ─── Calibration method ───────────────────────────────────────────────────────

/// Calibration method for computing quantization scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibMethod {
    /// Use the global maximum absolute value.
    MinMax,
    /// Use a high percentile of absolute values.
    Percentile(f32), // e.g., 0.9999
    /// ACIQ: analytical optimal clipping based on std_dev.
    Aciq,
    /// Mean squared error (MSE) minimization (approximated).
    Mse,
}

// ─── Layer calibration statistics ─────────────────────────────────────────────

/// Statistics collected from calibration activations for one layer.
#[derive(Debug, Clone)]
pub struct LayerCalibStats {
    pub layer_name: String,
    pub num_samples: usize,
    pub running_min: f32,
    pub running_max: f32,
    pub running_mean: f32,
    pub running_var: f32, // Welford online variance accumulator (M2)
    /// Histogram of |x| values (for percentile computation), 256 bins.
    histogram: Vec<u64>,
    histogram_max: f32,
}

impl LayerCalibStats {
    /// Create a new empty stats collector for the given layer.
    pub fn new(layer_name: impl Into<String>) -> Self {
        Self {
            layer_name: layer_name.into(),
            num_samples: 0,
            running_min: f32::INFINITY,
            running_max: f32::NEG_INFINITY,
            running_mean: 0.0,
            running_var: 0.0,
            histogram: vec![0u64; 256],
            histogram_max: 0.0,
        }
    }

    /// Update statistics with a new batch of activations.
    ///
    /// Uses Welford's online algorithm for numerically stable mean/variance.
    pub fn update(&mut self, activations: &[f32]) {
        for &x in activations {
            if !x.is_finite() {
                continue;
            }

            // Update min/max
            if x < self.running_min {
                self.running_min = x;
            }
            if x > self.running_max {
                self.running_max = x;
            }

            // Welford online mean/variance
            self.num_samples += 1;
            let n = self.num_samples as f32;
            let delta = x - self.running_mean;
            self.running_mean += delta / n;
            let delta2 = x - self.running_mean;
            self.running_var += delta * delta2;

            // Update histogram max (using abs value)
            let abs_x = x.abs();
            if abs_x > self.histogram_max {
                // Rebuild histogram with new max
                self.rebuild_histogram_max(abs_x);
            }
        }

        // Re-insert all new activations into histogram
        for &x in activations {
            if !x.is_finite() {
                continue;
            }
            let abs_x = x.abs();
            self.insert_histogram(abs_x);
        }
    }

    /// Rebuild histogram bins when a new maximum is encountered.
    fn rebuild_histogram_max(&mut self, new_max: f32) {
        // Scale existing histogram to new range
        if self.histogram_max > 0.0 && new_max > self.histogram_max {
            let scale = self.histogram_max / new_max;
            let mut new_hist = vec![0u64; 256];
            for (old_bin, &count) in self.histogram.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                // Map old bin center to new bin
                let old_frac = (old_bin as f32 + 0.5) / 256.0;
                let new_frac = old_frac * scale;
                let new_bin = (new_frac * 256.0) as usize;
                let new_bin = new_bin.min(255);
                new_hist[new_bin] += count;
            }
            self.histogram = new_hist;
        }
        self.histogram_max = new_max;
    }

    /// Insert a single absolute value into the histogram.
    fn insert_histogram(&mut self, abs_x: f32) {
        if self.histogram_max <= 0.0 {
            return;
        }
        let frac = abs_x / self.histogram_max;
        let bin = (frac * 256.0) as usize;
        let bin = bin.min(255);
        self.histogram[bin] += 1;
    }

    /// Standard deviation computed from Welford variance accumulator.
    pub fn std_dev(&self) -> f32 {
        if self.num_samples < 2 {
            return 0.0;
        }
        let variance = self.running_var / self.num_samples as f32;
        variance.max(0.0).sqrt()
    }

    /// Percentile of absolute values (0.0 - 1.0).
    ///
    /// Uses the histogram for efficient O(256) computation.
    pub fn percentile_abs(&self, p: f32) -> f32 {
        if self.num_samples == 0 || self.histogram_max <= 0.0 {
            return 0.0;
        }

        let p_clamped = p.clamp(0.0, 1.0);
        let target_count = (p_clamped * self.num_samples as f32).ceil() as u64;
        let target_count = target_count.max(1);

        let mut cumulative = 0u64;
        for (bin_idx, &count) in self.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                // Return upper edge of this bin
                let upper = (bin_idx as f32 + 1.0) / 256.0 * self.histogram_max;
                return upper.min(self.histogram_max);
            }
        }

        self.histogram_max
    }

    /// ACIQ optimal clipping: 2.83 * std_dev (Laplacian assumption).
    ///
    /// Reference: Bell et al. 2019, "Accurate Post Training Quantization".
    pub fn aciq_clip(&self) -> f32 {
        2.83 * self.std_dev()
    }

    /// Compute final quantization scale for INT8 (clip_val = 127).
    pub fn compute_scale(&self, method: CalibMethod) -> f32 {
        const CLIP_VAL: f32 = 127.0;
        self.compute_scale_with_clip(method, CLIP_VAL)
    }

    /// Compute final quantization scale for INT4 (clip_val = 7).
    pub fn compute_scale_int4(&self, method: CalibMethod) -> f32 {
        const CLIP_VAL: f32 = 7.0;
        self.compute_scale_with_clip(method, CLIP_VAL)
    }

    /// Internal: compute scale for the given clip value.
    fn compute_scale_with_clip(&self, method: CalibMethod, clip_val: f32) -> f32 {
        if self.num_samples == 0 {
            return 0.0;
        }

        let abs_max = self.running_min.abs().max(self.running_max.abs());

        let clipping_value = match method {
            CalibMethod::MinMax => abs_max,
            CalibMethod::Percentile(p) => {
                let pv = self.percentile_abs(p);
                if pv <= 0.0 {
                    abs_max
                } else {
                    pv
                }
            }
            CalibMethod::Aciq => {
                let aciq = self.aciq_clip();
                if aciq <= 0.0 {
                    abs_max
                } else {
                    aciq.min(abs_max)
                }
            }
            CalibMethod::Mse => {
                // Grid search: try fractions of abs_max, pick the one minimizing MSE proxy.
                // We approximate MSE using histogram statistics.
                self.mse_optimal_clip(abs_max, clip_val)
            }
        };

        if clipping_value <= 0.0 {
            return 0.0;
        }

        clipping_value / clip_val
    }

    /// Approximate MSE-optimal clipping via grid search on the histogram.
    ///
    /// For each candidate clip value, computes:
    ///   MSE ≈ clipping_error + rounding_error
    fn mse_optimal_clip(&self, abs_max: f32, clip_val: f32) -> f32 {
        if abs_max <= 0.0 || self.num_samples == 0 {
            return abs_max;
        }

        let n_steps = 100usize;
        let mut best_clip = abs_max;
        let mut best_mse = f32::INFINITY;

        for step in 1..=n_steps {
            let frac = step as f32 / n_steps as f32;
            let candidate_clip = abs_max * frac;
            if candidate_clip <= 0.0 {
                continue;
            }

            // Rounding error variance: (clip / clip_val)^2 / 3
            let scale = candidate_clip / clip_val;
            let rounding_var = scale * scale / 3.0;

            // Clipping error: estimate fraction of values > candidate_clip
            // from histogram, compute their average squared distance.
            let clip_frac = candidate_clip / self.histogram_max;
            let clip_bin = (clip_frac * 256.0) as usize;

            let mut clip_error_sq = 0.0f32;
            for (bin_idx, &count) in self.histogram.iter().enumerate().skip(clip_bin.min(255)) {
                if count == 0 {
                    continue;
                }
                // Bin center value
                let bin_center = (bin_idx as f32 + 0.5) / 256.0 * self.histogram_max;
                let excess = (bin_center - candidate_clip).max(0.0);
                clip_error_sq += count as f32 * excess * excess;
            }

            let n = self.num_samples as f32;
            let clip_mse = if n > 0.0 { clip_error_sq / n } else { 0.0 };
            let total_mse = rounding_var + clip_mse;

            if total_mse < best_mse {
                best_mse = total_mse;
                best_clip = candidate_clip;
            }
        }

        best_clip
    }

    /// Summary statistics as a struct.
    pub fn summary(&self) -> CalibSummary {
        let p99 = self.percentile_abs(0.99);
        let p9999 = self.percentile_abs(0.9999);
        let int8_scale = self.compute_scale(CalibMethod::Percentile(0.9999));

        let (min, max) = if self.num_samples == 0 {
            (0.0, 0.0)
        } else {
            (self.running_min, self.running_max)
        };

        CalibSummary {
            layer_name: self.layer_name.clone(),
            num_samples: self.num_samples,
            min,
            max,
            mean: self.running_mean,
            std_dev: self.std_dev(),
            p99,
            p9999,
            suggested_int8_scale: int8_scale,
        }
    }
}

// ─── Calibration summary ──────────────────────────────────────────────────────

/// Summary statistics for a single layer's calibration.
#[derive(Debug, Clone)]
pub struct CalibSummary {
    pub layer_name: String,
    pub num_samples: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub p99: f32,
    pub p9999: f32,
    pub suggested_int8_scale: f32,
}

impl CalibSummary {
    /// One-line human-readable summary.
    pub fn summary_line(&self) -> String {
        format!(
            "[{}] n={} min={:.4} max={:.4} mean={:.4} std={:.4} p99={:.4} p9999={:.4} scale_int8={:.6}",
            self.layer_name,
            self.num_samples,
            self.min,
            self.max,
            self.mean,
            self.std_dev,
            self.p99,
            self.p9999,
            self.suggested_int8_scale,
        )
    }
}

// ─── Calibration database ─────────────────────────────────────────────────────

/// Calibration database: stores stats for all layers.
pub struct CalibrationDb {
    layers: HashMap<String, LayerCalibStats>,
    method: CalibMethod,
}

impl CalibrationDb {
    /// Create a new calibration database with the given method.
    pub fn new(method: CalibMethod) -> Self {
        Self {
            layers: HashMap::new(),
            method,
        }
    }

    /// Create with MinMax calibration.
    pub fn new_minmax() -> Self {
        Self::new(CalibMethod::MinMax)
    }

    /// Create with Percentile calibration.
    pub fn new_percentile(p: f32) -> Self {
        Self::new(CalibMethod::Percentile(p.clamp(0.0, 1.0)))
    }

    /// Record activations for a named layer.
    pub fn record(&mut self, layer_name: &str, activations: &[f32]) {
        let stats = self
            .layers
            .entry(layer_name.to_owned())
            .or_insert_with(|| LayerCalibStats::new(layer_name));
        stats.update(activations);
    }

    /// Get stats for a layer.
    pub fn get_stats(&self, layer_name: &str) -> Option<&LayerCalibStats> {
        self.layers.get(layer_name)
    }

    /// Compute scale for a layer using the configured method.
    pub fn scale_for_layer(&self, layer_name: &str) -> Option<f32> {
        self.layers
            .get(layer_name)
            .map(|s| s.compute_scale(self.method))
    }

    /// Export all scales as a map: layer_name → scale.
    pub fn export_scales(&self) -> HashMap<String, f32> {
        self.layers
            .iter()
            .map(|(name, stats)| (name.clone(), stats.compute_scale(self.method)))
            .collect()
    }

    /// Export summaries for all layers.
    pub fn summaries(&self) -> Vec<CalibSummary> {
        let mut summaries: Vec<CalibSummary> = self.layers.values().map(|s| s.summary()).collect();
        summaries.sort_by(|a, b| a.layer_name.cmp(&b.layer_name));
        summaries
    }

    /// Total number of calibrated layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Generate a calibration report.
    pub fn report(&self) -> String {
        let method_str = match self.method {
            CalibMethod::MinMax => "MinMax".to_owned(),
            CalibMethod::Percentile(p) => format!("Percentile({:.4})", p),
            CalibMethod::Aciq => "ACIQ".to_owned(),
            CalibMethod::Mse => "MSE".to_owned(),
        };

        let mut lines = Vec::new();
        lines.push("=== PTQ Calibration Report ===".to_string());
        lines.push(format!("Method: {}", method_str));
        lines.push(format!("Layers: {}", self.layers.len()));
        lines.push(String::new());

        let summaries = self.summaries();
        for s in &summaries {
            lines.push(s.summary_line());
        }

        lines.push(String::new());
        lines.push("=== Scales ===".to_string());
        let scales = self.export_scales();
        let mut scale_entries: Vec<_> = scales.iter().collect();
        scale_entries.sort_by_key(|(k, _)| k.as_str());
        for (name, scale) in scale_entries {
            lines.push(format!("  {}: {:.8}", name, scale));
        }

        lines.join("\n")
    }
}

// ─── Simulation ───────────────────────────────────────────────────────────────

/// Linear Congruential Generator for deterministic pseudo-random f32 in [-1, 1].
fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 32) as f32) / (u32::MAX as f32 + 1.0) * 2.0 - 1.0
}

/// Synthetic calibration-data generators for tests — **not** real
/// measurements.
///
/// Everything in this module produces fabricated activation statistics via
/// a seeded LCG, never data measured from a real model. It is kept under
/// `calibration::testing` (a `#[doc(hidden)]` alias also resolves the
/// legacy `calibration::simulate_calibration` path) rather than re-exported
/// from the crate root alongside the real calibration API, precisely so it
/// cannot be mistaken for one.
pub mod testing {
    use super::{lcg_f32, CalibrationDb};

    /// Simulate a calibration pass (for testing without a real model).
    ///
    /// Generates synthetic activation patterns for each layer using a seeded LCG.
    /// Each layer gets `samples_per_layer` activation values drawn from [-1, 1],
    /// scaled by a layer-specific amplitude to simulate natural inter-layer variation.
    ///
    /// **This produces entirely fabricated data**, not statistics measured from
    /// any real model. It exists so calibration-pipeline code (this module's
    /// own tests, and the integration tests under
    /// `crates/oxibonsai-model/tests/`) can be exercised without a real model
    /// and real activations. A `CalibrationDb` populated this way must
    /// never be used to quantize or ship a real model; [`validate_calibration`][super::validate_calibration]
    /// cannot distinguish synthetic statistics from measured ones, so that
    /// check is the caller's responsibility.
    pub fn simulate_calibration(
        db: &mut CalibrationDb,
        layer_names: &[&str],
        samples_per_layer: usize,
        seed: u64,
    ) {
        let mut state = seed;

        for (layer_idx, &layer_name) in layer_names.iter().enumerate() {
            // Layer-specific amplitude to mimic real network activation diversity.
            let amplitude = 1.0 + layer_idx as f32 * 0.5;
            let mut activations = Vec::with_capacity(samples_per_layer);

            for _ in 0..samples_per_layer {
                let v = lcg_f32(&mut state) * amplitude;
                activations.push(v);
            }

            db.record(layer_name, &activations);
        }
    }
}

/// Legacy path for [`testing::simulate_calibration`], kept resolvable for
/// existing callers (this module's own tests and the integration tests
/// under `crates/oxibonsai-model/tests/`). `#[doc(hidden)]` and not
/// re-exported from the crate root — see [`testing`]'s module docs for why
/// this is a synthetic-data generator, not part of the real calibration API.
#[doc(hidden)]
pub use testing::simulate_calibration;

// ─── Validation ───────────────────────────────────────────────────────────────

/// Validation result for a single layer's calibration scale.
#[derive(Debug, Clone)]
pub struct CalibValidation {
    pub layer_name: String,
    pub scale: f32,
    pub issues: Vec<String>,
    pub is_valid: bool,
}

impl CalibValidation {
    /// Validate the calibration scale for a layer.
    pub fn validate(layer_name: &str, stats: &LayerCalibStats, scale: f32) -> Self {
        let mut issues = Vec::new();

        // Check: scale must be positive and finite
        if !scale.is_finite() {
            issues.push(format!("scale is not finite: {}", scale));
        } else if scale <= 0.0 {
            issues.push(format!("scale is non-positive: {}", scale));
        }

        // Check: no samples recorded
        if stats.num_samples == 0 {
            issues.push("no calibration samples recorded".to_owned());
        }

        // Check: scale plausibility — should be at most abs_max / 1.0
        // (a scale larger than abs_max / 1 would map all values to near-zero)
        if scale.is_finite() && scale > 0.0 && stats.num_samples > 0 {
            let abs_max = stats.running_min.abs().max(stats.running_max.abs());
            if abs_max > 0.0 {
                // If scale is far too large (> 10x the naive minmax scale), warn
                let minmax_scale = abs_max / 127.0;
                if scale > minmax_scale * 10.0 {
                    issues.push(format!(
                        "scale {:.6} is >10x the MinMax scale {:.6} (possible overflow)",
                        scale, minmax_scale
                    ));
                }
                // If scale is far too small (< 1/1000 of minmax), warn
                if scale < minmax_scale / 1000.0 {
                    issues.push(format!(
                        "scale {:.8} is <1/1000 of MinMax scale {:.6} (possible underflow)",
                        scale, minmax_scale
                    ));
                }
            } else {
                // abs_max == 0 means all-zero activations
                issues.push("all activations are zero".to_owned());
            }
        }

        // Check: min <= max sanity
        if stats.num_samples > 0 && stats.running_min > stats.running_max {
            issues.push(format!(
                "running_min ({}) > running_max ({})",
                stats.running_min, stats.running_max
            ));
        }

        let is_valid = issues.is_empty();

        Self {
            layer_name: layer_name.to_owned(),
            scale,
            issues,
            is_valid,
        }
    }
}

/// Validate all layers in a database using its configured method.
pub fn validate_calibration(db: &CalibrationDb) -> Vec<CalibValidation> {
    let mut results: Vec<CalibValidation> = db
        .layers
        .iter()
        .map(|(name, stats)| {
            let scale = stats.compute_scale(db.method);
            CalibValidation::validate(name, stats, scale)
        })
        .collect();
    results.sort_by(|a, b| a.layer_name.cmp(&b.layer_name));
    results
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_calib_stats_new() {
        let stats = LayerCalibStats::new("layer0");
        assert_eq!(stats.layer_name, "layer0");
        assert_eq!(stats.num_samples, 0);
        assert_eq!(stats.running_var, 0.0);
        assert_eq!(stats.running_mean, 0.0);
    }

    #[test]
    fn layer_calib_stats_update_single() {
        let mut stats = LayerCalibStats::new("layer0");
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01 - 0.5).collect();
        stats.update(&data);
        assert_eq!(stats.num_samples, data.len());
    }

    #[test]
    fn layer_calib_stats_running_min_max() {
        let mut stats = LayerCalibStats::new("layer0");
        stats.update(&[-3.0, 1.0, 2.5]);
        stats.update(&[0.0, 5.0, -1.0]);
        assert!((stats.running_min - (-3.0)).abs() < 1e-6);
        assert!((stats.running_max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn layer_calib_stats_std_dev() {
        let mut stats = LayerCalibStats::new("layer0");
        // Non-constant data should have positive std dev
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.01).collect();
        stats.update(&data);
        let sd = stats.std_dev();
        assert!(
            sd > 0.0,
            "std_dev should be positive for non-constant data, got {sd}"
        );
    }

    #[test]
    fn layer_calib_stats_percentile_abs() {
        let mut stats = LayerCalibStats::new("layer0");
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        stats.update(&data);
        // p=1.0 should return near max(|x|) = 100.0
        let p100 = stats.percentile_abs(1.0);
        assert!(
            p100 >= 99.0,
            "p=1.0 percentile should be near 100, got {p100}"
        );
    }

    #[test]
    fn layer_calib_stats_aciq_clip() {
        let mut stats = LayerCalibStats::new("layer0");
        // Normal-ish data
        let data: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) * 0.1).collect();
        stats.update(&data);
        let sd = stats.std_dev();
        let aciq = stats.aciq_clip();
        let expected = 2.83 * sd;
        assert!(
            (aciq - expected).abs() < 1e-4,
            "aciq_clip={aciq}, expected 2.83*std_dev={expected}"
        );
    }

    #[test]
    fn layer_calib_stats_compute_scale_minmax() {
        let mut stats = LayerCalibStats::new("layer0");
        let data: Vec<f32> = vec![-2.54, 1.0, 0.5, -0.3, std::f32::consts::PI];
        stats.update(&data);
        let scale = stats.compute_scale(CalibMethod::MinMax);
        assert!(scale > 0.0, "MinMax scale must be positive, got {scale}");
    }

    #[test]
    fn layer_calib_stats_compute_scale_percentile() {
        let mut stats = LayerCalibStats::new("layer0");
        // Data with one large outlier
        let mut data: Vec<f32> = (0..999).map(|i| (i as f32) * 0.001).collect();
        data.push(100.0); // outlier
        stats.update(&data);

        let scale_minmax = stats.compute_scale(CalibMethod::MinMax);
        let scale_p99 = stats.compute_scale(CalibMethod::Percentile(0.99));
        // Percentile-based scale should be <= MinMax scale for data with outliers
        assert!(
            scale_p99 <= scale_minmax,
            "Percentile scale {scale_p99} should be <= MinMax scale {scale_minmax}"
        );
    }

    #[test]
    fn calib_summary_summary_line_nonempty() {
        let mut stats = LayerCalibStats::new("fc1");
        stats.update(&[0.1, -0.2, 0.3]);
        let summary = stats.summary();
        let line = summary.summary_line();
        assert!(!line.is_empty(), "summary_line should not be empty");
        assert!(
            line.contains("fc1"),
            "summary_line should contain layer name"
        );
    }

    #[test]
    fn calib_db_new_minmax() {
        let db = CalibrationDb::new_minmax();
        // Verify method is MinMax by checking that scale_for_layer uses it
        // (internal field not public, but we can verify behavior via record/scale)
        assert_eq!(db.num_layers(), 0);
    }

    #[test]
    fn calib_db_record_creates_layer() {
        let mut db = CalibrationDb::new_minmax();
        assert_eq!(db.num_layers(), 0);
        db.record("attn.q", &[0.1, -0.2, 0.5]);
        assert_eq!(db.num_layers(), 1);
        db.record("attn.k", &[0.3, 0.4]);
        assert_eq!(db.num_layers(), 2);
        // Recording to same layer should not increase count
        db.record("attn.q", &[0.9, -0.1]);
        assert_eq!(db.num_layers(), 2);
    }

    #[test]
    fn calib_db_scale_for_unknown_layer() {
        let db = CalibrationDb::new_minmax();
        let result = db.scale_for_layer("nonexistent_layer");
        assert!(result.is_none(), "Should return None for unknown layer");
    }

    #[test]
    fn calib_db_export_scales_all_layers() {
        let mut db = CalibrationDb::new_minmax();
        db.record("layer_a", &[1.0, 2.0, 3.0]);
        db.record("layer_b", &[-1.0, 0.5]);
        db.record("layer_c", &[0.1, 0.2]);
        let scales = db.export_scales();
        assert_eq!(
            scales.len(),
            db.num_layers(),
            "export_scales count should match num_layers"
        );
        assert!(scales.contains_key("layer_a"));
        assert!(scales.contains_key("layer_b"));
        assert!(scales.contains_key("layer_c"));
    }

    #[test]
    fn calib_db_report_nonempty() {
        let mut db = CalibrationDb::new_percentile(0.999);
        db.record("block0.ffn", &[0.1, -0.5, 0.3, 0.9]);
        let report = db.report();
        assert!(
            !report.is_empty(),
            "report() should return a non-empty string"
        );
        assert!(
            report.contains("block0.ffn"),
            "report should contain layer name"
        );
    }

    #[test]
    fn simulate_calibration_fills_db() {
        let mut db = CalibrationDb::new_minmax();
        let layer_names = ["layer0", "layer1", "layer2", "layer3"];
        simulate_calibration(&mut db, &layer_names, 256, 42);
        assert_eq!(db.num_layers(), layer_names.len());
        for &name in &layer_names {
            let stats = db.get_stats(name).expect("layer should exist");
            assert_eq!(stats.num_samples, 256);
        }
    }

    #[test]
    fn simulate_calibration_deterministic() {
        let layer_names = ["attn.q", "attn.k", "attn.v", "ffn.up"];
        let seed = 12345u64;

        let mut db1 = CalibrationDb::new_minmax();
        simulate_calibration(&mut db1, &layer_names, 128, seed);
        let scales1 = db1.export_scales();

        let mut db2 = CalibrationDb::new_minmax();
        simulate_calibration(&mut db2, &layer_names, 128, seed);
        let scales2 = db2.export_scales();

        for name in &layer_names {
            let s1 = scales1[*name];
            let s2 = scales2[*name];
            assert!(
                (s1 - s2).abs() < 1e-8,
                "scales should be identical for same seed: {name}: {s1} vs {s2}"
            );
        }
    }

    #[test]
    fn validate_calibration_valid_layer() {
        let mut stats = LayerCalibStats::new("block0.attn");
        // Typical activation values
        let data: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.05).collect();
        stats.update(&data);
        let scale = stats.compute_scale(CalibMethod::MinMax);
        let val = CalibValidation::validate("block0.attn", &stats, scale);
        assert!(
            val.is_valid,
            "should be valid for reasonable data: issues={:?}",
            val.issues
        );
        assert!(val.issues.is_empty());
    }

    #[test]
    fn validate_calibration_all_valid() {
        let mut db = CalibrationDb::new_minmax();
        let layer_names = [
            "embed",
            "block0.attn.q",
            "block0.attn.k",
            "block0.ffn.up",
            "block0.ffn.down",
            "lm_head",
        ];
        simulate_calibration(&mut db, &layer_names, 512, 999);
        let validations = validate_calibration(&db);
        assert_eq!(validations.len(), layer_names.len());
        for v in &validations {
            assert!(
                v.is_valid,
                "layer '{}' should be valid after simulate_calibration: issues={:?}",
                v.layer_name, v.issues
            );
        }
    }
}
