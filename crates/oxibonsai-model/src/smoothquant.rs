//! SmoothQuant per-channel FP8 calibrator and channel-aware quantization.
//!
//! SmoothQuant (Xiao et al. 2022) addresses the quantization difficulty mismatch
//! between activations (which have large per-channel outliers) and weights (which
//! are relatively smooth). It migrates the quantization difficulty from activations
//! to weights by computing per-channel smoothing factors:
//!
//!   `s_j = max(|A_j|)^α / max(|W_j|)^(1−α)`
//!
//! then rescaling: `Ã[i,j] = A[i,j] / s_j`, `W̃[i,j] = W[i,j] × s_j`.
//!
//! This module provides:
//! - [`SmoothQuantCalibrator`]: online per-channel max-abs accumulator.
//! - [`quantize_fp8_e4m3_smooth`]: quantize smoothed weights into E4M3FN blocks.
//! - [`quantize_fp8_e5m2_smooth`]: quantize smoothed weights into E5M2 blocks.

use std::collections::HashMap;

use crate::dynamic_quant::{compute_smooth_factors, smooth_weights, DynQuantError, SmoothQuantConfig};
use oxibonsai_core::quant_fp8::{BlockFP8E4M3, BlockFP8E5M2};

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the SmoothQuant calibrator and channel-aware quantization.
#[derive(Debug, Clone)]
pub enum SmoothQuantError {
    /// Calibrator has no recorded layers.
    EmptyCalibrator,
    /// The requested layer has not been recorded.
    LayerNotFound(String),
    /// The supplied `in_features` doesn't match what was originally recorded.
    InFeaturesMismatch { expected: usize, got: usize },
    /// An underlying quantization operation failed.
    QuantizationError(String),
}

impl std::fmt::Display for SmoothQuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCalibrator => write!(f, "SmoothQuant calibrator has no recorded layers"),
            Self::LayerNotFound(name) => {
                write!(f, "SmoothQuant calibrator: layer '{name}' not found")
            }
            Self::InFeaturesMismatch { expected, got } => write!(
                f,
                "SmoothQuant calibrator: in_features mismatch — expected {expected}, got {got}"
            ),
            Self::QuantizationError(msg) => {
                write!(f, "SmoothQuant quantization error: {msg}")
            }
        }
    }
}

impl std::error::Error for SmoothQuantError {}

// ─── Internal per-layer channel statistics ────────────────────────────────────

/// Per-layer per-channel running statistics used by the calibrator.
struct ChannelStats {
    /// Number of input features (columns) for this layer.
    in_features: usize,
    /// Running maximum of |activation| across all batches, one slot per column.
    running_max_abs: Vec<f32>,
    /// Total number of activation batches recorded for this layer.
    sample_count: usize,
}

impl ChannelStats {
    fn new(in_features: usize) -> Self {
        Self {
            in_features,
            running_max_abs: vec![0.0_f32; in_features],
            sample_count: 0,
        }
    }

    /// Update running per-channel max-abs from one batch of activations.
    ///
    /// `activations` is `[num_tokens × in_features]` row-major.
    fn update(&mut self, activations: &[f32], in_features: usize) {
        debug_assert_eq!(in_features, self.in_features);
        let num_tokens = activations.len() / in_features;
        for t in 0..num_tokens {
            for (j, slot) in self.running_max_abs.iter_mut().enumerate() {
                let idx = t * in_features + j;
                if idx < activations.len() {
                    let v = activations[idx].abs();
                    if v > *slot {
                        *slot = v;
                    }
                }
            }
        }
        self.sample_count += 1;
    }
}

// ─── SmoothQuantCalibrator ────────────────────────────────────────────────────

/// Online per-channel activation calibrator for SmoothQuant.
///
/// Feed batches of activations for each named linear layer via
/// [`record_activation`][Self::record_activation], then call
/// [`smooth_factors`][Self::smooth_factors] to obtain the SmoothQuant
/// smoothing vector for that layer's weight matrix.
pub struct SmoothQuantCalibrator {
    layers: HashMap<String, ChannelStats>,
    config: SmoothQuantConfig,
}

impl SmoothQuantCalibrator {
    /// Create a new calibrator using the given SmoothQuant config.
    pub fn new(config: SmoothQuantConfig) -> Self {
        Self {
            layers: HashMap::new(),
            config,
        }
    }

    /// Record one batch of activations for a named layer.
    ///
    /// `activations` is a `[num_tokens × in_features]` row-major flat slice.
    /// If this is the first call for `layer_name`, a new per-channel accumulator
    /// is created. For subsequent calls the running per-channel max-abs is updated.
    ///
    /// # Panics
    ///
    /// Panics if `in_features` changes between calls for the same `layer_name`.
    pub fn record_activation(
        &mut self,
        layer_name: &str,
        activations: &[f32],
        in_features: usize,
    ) {
        if in_features == 0 || activations.is_empty() {
            return;
        }

        let stats = self
            .layers
            .entry(layer_name.to_owned())
            .or_insert_with(|| ChannelStats::new(in_features));

        if stats.in_features != in_features {
            panic!(
                "SmoothQuantCalibrator::record_activation: in_features mismatch for layer '{}' \
                 — expected {}, got {}",
                layer_name, stats.in_features, in_features
            );
        }

        stats.update(activations, in_features);
    }

    /// Compute SmoothQuant smoothing factors for a named layer.
    ///
    /// Uses the running per-channel activation max accumulated via
    /// [`record_activation`][Self::record_activation] together with the supplied
    /// weight matrix to derive per-input-feature smoothing factors.
    ///
    /// `weights` is `[out_features × in_features]` row-major.
    ///
    /// Returns a `Vec<f32>` of length `in_features`.
    pub fn smooth_factors(
        &self,
        layer_name: &str,
        weights: &[f32],
        out_features: usize,
    ) -> Result<Vec<f32>, SmoothQuantError> {
        let stats = self
            .layers
            .get(layer_name)
            .ok_or_else(|| SmoothQuantError::LayerNotFound(layer_name.to_owned()))?;

        let in_features = stats.in_features;

        // We pass running_max_abs as a synthetic single-row "activation" matrix
        // (shape [1 × in_features]).  compute_smooth_factors will compute the
        // per-column max across that one row, which gives exactly running_max_abs[j].
        let factors = compute_smooth_factors(
            &stats.running_max_abs,
            weights,
            in_features,
            1,         // tokens = 1 (running_max_abs is already the global max)
            out_features,
            &self.config,
        );

        Ok(factors)
    }

    /// Number of distinct layers recorded by this calibrator.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Whether the calibrator has any recorded data for the given layer name.
    pub fn has_layer(&self, name: &str) -> bool {
        self.layers.contains_key(name)
    }
}

// ─── Channel-aware FP8 quantization ──────────────────────────────────────────

/// Quantize a weight matrix using SmoothQuant scaling into FP8 E4M3FN blocks.
///
/// The smoothing factors obtained from [`SmoothQuantCalibrator::smooth_factors`]
/// are applied to the weight matrix in-place (W̃ = W × s_j for column j), then
/// the smoothed weights are quantized into [`BlockFP8E4M3`] blocks.
///
/// `weights` is `[out_features × in_features]` row-major.
/// `smooth_factors` must have length `in_features`.
///
/// Returns the resulting FP8 block vector; total elements must be a multiple of
/// `QK_FP8` (32), so `out_features × in_features` must be divisible by 32.
pub fn quantize_fp8_e4m3_smooth(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    smooth_factors: &[f32],
) -> Result<Vec<BlockFP8E4M3>, SmoothQuantError> {
    if smooth_factors.len() != in_features {
        return Err(SmoothQuantError::InFeaturesMismatch {
            expected: in_features,
            got: smooth_factors.len(),
        });
    }

    // Clone and apply SmoothQuant weight scaling: W̃[i,j] = W[i,j] * s_j
    let mut smoothed = weights.to_vec();
    smooth_weights(&mut smoothed, smooth_factors, out_features, in_features)
        .map_err(|e: DynQuantError| SmoothQuantError::QuantizationError(e.to_string()))?;

    // Quantize the smoothed weights into FP8 E4M3FN blocks.
    BlockFP8E4M3::quantize(&smoothed)
        .map_err(|e| SmoothQuantError::QuantizationError(e.to_string()))
}

/// Quantize a weight matrix using SmoothQuant scaling into FP8 E5M2 blocks.
///
/// Mirrors [`quantize_fp8_e4m3_smooth`] for the E5M2 format.
///
/// `weights` is `[out_features × in_features]` row-major.
/// `smooth_factors` must have length `in_features`.
pub fn quantize_fp8_e5m2_smooth(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    smooth_factors: &[f32],
) -> Result<Vec<BlockFP8E5M2>, SmoothQuantError> {
    if smooth_factors.len() != in_features {
        return Err(SmoothQuantError::InFeaturesMismatch {
            expected: in_features,
            got: smooth_factors.len(),
        });
    }

    let mut smoothed = weights.to_vec();
    smooth_weights(&mut smoothed, smooth_factors, out_features, in_features)
        .map_err(|e: DynQuantError| SmoothQuantError::QuantizationError(e.to_string()))?;

    BlockFP8E5M2::quantize(&smoothed)
        .map_err(|e| SmoothQuantError::QuantizationError(e.to_string()))
}
