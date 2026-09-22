//! Model merging utilities: linear interpolation, SLERP, TIES, and task-vector merging.
//!
//! These methods allow creating new model variants by combining weights from
//! multiple source models — useful for model fusion, continual learning, and
//! experimental model architectures.
//!
//! ## Available Methods
//!
//! - **Linear**: simple weighted average `(1-α)*A + α*B`
//! - **SLERP**: spherical linear interpolation (preserves direction on the unit hypersphere)
//! - **TIES**: sign-majority election with magnitude trimming (from the TIES-Merging paper)
//! - **Task Vector**: `base + α*(finetuned - base)`, adds or subtracts fine-tuning direction
//! - **DARE**: random dropout of task-vector elements with rescaling for sparse merging
//!
//! ## Reachability
//!
//! As of this writing, nothing in the CLI calls [`merge_models`] or
//! [`merge_tensors`] — there is no `oxibonsai merge`-style entry point that
//! runs this pipeline against real checkpoints. This module is a tested,
//! documented primitive for that future work, not a wired feature.
//!
//! ## Example
//!
//! ```rust
//! use oxibonsai_model::model_merge::{WeightTensor, MergeConfig, MergeMethod, merge_models};
//!
//! let base = vec![
//!     WeightTensor::new("embed.weight", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
//! ];
//! let other = vec![
//!     WeightTensor::new("embed.weight", vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]),
//! ];
//! let config = MergeConfig { method: MergeMethod::Linear, alpha: 0.5, ..Default::default() };
//! let merged = merge_models(&base, &other, &config).expect("merge failed");
//! assert_eq!(merged.len(), 1);
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ──────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────

/// Errors that can occur during model merging operations.
#[derive(Debug, Error)]
pub enum MergeError {
    /// Shape mismatch between the two tensors being merged.
    #[error("shape mismatch for tensor '{name}': {a:?} vs {b:?}")]
    ShapeMismatch {
        name: String,
        a: Vec<usize>,
        b: Vec<usize>,
    },
    /// Tensor has no elements (empty data slice or shape contains zero).
    #[error("empty tensor: '{0}'")]
    EmptyTensor(String),
    /// Alpha coefficient is outside the valid range `[0.0, 1.0]`.
    #[error("invalid alpha {0}: must be in [0.0, 1.0]")]
    InvalidAlpha(f32),
    /// Density is outside the valid range `(0.0, 1.0]`.
    #[error("invalid density {0}: must be in (0.0, 1.0]")]
    InvalidDensity(f32),
    /// SLERP was attempted on a zero-norm vector.
    #[error("SLERP failed: zero vector")]
    SlerpZeroVector,
}

// ──────────────────────────────────────────────────────────────────
// WeightTensor
// ──────────────────────────────────────────────────────────────────

/// A named weight tensor — a flat `f32` slice with shape metadata.
///
/// The `data` field is stored in row-major (C-contiguous) order.
/// The product of all shape dimensions must equal `data.len()`.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    /// Unique name identifying this tensor within a model checkpoint.
    pub name: String,
    /// Raw weight data in row-major order.
    pub data: Vec<f32>,
    /// N-dimensional shape (e.g., `[4096, 4096]` for a square linear layer).
    pub shape: Vec<usize>,
}

impl WeightTensor {
    /// Construct a weight tensor from its components.
    pub fn new(name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            data,
            shape,
        }
    }

    /// Construct an all-zeros tensor with the given name and shape.
    pub fn zeros(name: impl Into<String>, shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self {
            name: name.into(),
            data: vec![0.0f32; n],
            shape,
        }
    }

    /// Number of scalar elements: product of all shape dimensions.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Euclidean (L2) norm: `sqrt(sum(x_i^2))`.
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Cosine similarity with `other`: `dot(a, b) / (|a| * |b|)`.
    ///
    /// Returns `Err(MergeError::ShapeMismatch)` when element counts differ,
    /// `Err(MergeError::EmptyTensor)` when either tensor is empty.
    pub fn cosine_similarity(&self, other: &WeightTensor) -> Result<f32, MergeError> {
        let n = self.element_count();
        if n == 0 {
            return Err(MergeError::EmptyTensor(self.name.clone()));
        }
        if other.element_count() == 0 {
            return Err(MergeError::EmptyTensor(other.name.clone()));
        }
        if n != other.element_count() {
            return Err(MergeError::ShapeMismatch {
                name: self.name.clone(),
                a: self.shape.clone(),
                b: other.shape.clone(),
            });
        }

        let dot: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();
        let denom = norm_a * norm_b;
        if denom == 0.0 {
            // At least one zero vector — cosine similarity is conventionally 0.
            return Ok(0.0);
        }
        Ok(dot / denom)
    }

    /// Element-wise addition: `self + other`.
    pub fn add(&self, other: &WeightTensor) -> Result<WeightTensor, MergeError> {
        check_compatible(self, other)?;
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(WeightTensor::new(
            self.name.clone(),
            data,
            self.shape.clone(),
        ))
    }

    /// Element-wise subtraction: `self - other`.
    pub fn sub(&self, other: &WeightTensor) -> Result<WeightTensor, MergeError> {
        check_compatible(self, other)?;
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(WeightTensor::new(
            self.name.clone(),
            data,
            self.shape.clone(),
        ))
    }

    /// Scalar multiplication: `self * alpha`.
    pub fn scale(&self, alpha: f32) -> WeightTensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * alpha).collect();
        WeightTensor::new(self.name.clone(), data, self.shape.clone())
    }

    /// Linear interpolation: `(1 - t)*self + t*other`.
    ///
    /// `t` is not validated here — use [`merge_tensors`] for validated entry points.
    pub fn lerp(&self, other: &WeightTensor, t: f32) -> Result<WeightTensor, MergeError> {
        check_compatible(self, other)?;
        let data = linear_merge(&self.data, &other.data, t);
        Ok(WeightTensor::new(
            self.name.clone(),
            data,
            self.shape.clone(),
        ))
    }
}

// ──────────────────────────────────────────────────────────────────
// MergeMethod / MergeConfig
// ──────────────────────────────────────────────────────────────────

/// Available merge strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeMethod {
    /// Simple weighted average: `result = (1-α)*A + α*B`.
    Linear,
    /// Spherical linear interpolation — preserves direction on the unit hypersphere.
    Slerp,
    /// TIES-Merging: sign-majority election with magnitude trimming.
    Ties,
    /// Task-vector merging: `base + α*(finetuned - base)`.
    ///
    /// When `α=1.0` this is identical to returning `finetuned`;
    /// when `α=0.0` it is identical to returning `base`.
    TaskVector,
    /// DARE (Drop And REscale): deterministically drop task-vector elements, then rescale.
    Dare {
        /// Seed for the deterministic LCG random number generator.
        seed: u64,
        /// Fraction of task-vector elements to zero out (0.0 = keep all, 1.0 = drop all).
        dropout_rate: f32,
    },
}

/// Configuration for a single merge pass.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Which merging algorithm to use.
    pub method: MergeMethod,
    /// Interpolation coefficient: 0.0 → model A only, 1.0 → model B only.
    pub alpha: f32,
    /// Normalize each weight tensor to unit L2 norm before merging.
    pub normalize: bool,
    /// For TIES: fraction of weights to retain by magnitude (0.0–1.0].
    pub density: f32,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            method: MergeMethod::Linear,
            alpha: 0.5,
            normalize: false,
            density: 0.5,
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// MergeStats
// ──────────────────────────────────────────────────────────────────

/// Statistics collected during a model merge operation.
#[derive(Debug, Clone)]
pub struct MergeStats {
    /// Number of tensors that were present in both models and were merged.
    pub tensors_merged: usize,
    /// Number of tensors that existed only in `base` and were copied unchanged.
    pub tensors_copied: usize,
    /// Total number of scalar parameters in the output model.
    pub total_params: usize,
    /// Average cosine similarity between corresponding base/other tensors.
    pub mean_cosine_similarity: f32,
    /// The merge method that was used.
    pub method: MergeMethod,
}

impl MergeStats {
    /// Human-readable one-line summary of the merge statistics.
    pub fn summary(&self) -> String {
        format!(
            "method={:?} merged={} copied={} total_params={} mean_cosine_sim={:.4}",
            self.method,
            self.tensors_merged,
            self.tensors_copied,
            self.total_params,
            self.mean_cosine_similarity,
        )
    }
}

// ──────────────────────────────────────────────────────────────────
// Low-level primitive functions
// ──────────────────────────────────────────────────────────────────

/// Linear interpolation element-wise: `result[i] = (1-α)*a[i] + α*b[i]`.
///
/// `a` and `b` must be the same length; if they differ the shorter slice
/// determines the output length (extra elements from the longer slice are dropped).
pub fn linear_merge(a: &[f32], b: &[f32], alpha: f32) -> Vec<f32> {
    let one_minus_alpha = 1.0 - alpha;
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| one_minus_alpha * ai + alpha * bi)
        .collect()
}

/// Spherical linear interpolation (SLERP) between two real-valued vectors.
///
/// Both vectors are normalized to unit length, and the SLERP coefficients
/// are applied to those *normalized* vectors — not the raw, differently
/// scaled inputs — so the result actually lies on the great-circle arc
/// between the two directions instead of being skewed toward whichever
/// input happens to have the larger magnitude.
///
/// If either input has zero norm, this infallible wrapper falls back to
/// ordinary linear interpolation over the raw vectors (there is no
/// direction to interpolate for a zero vector); use [`slerp_checked`]
/// instead if you need to detect that case rather than silently getting a
/// linear blend. If the two directions are nearly parallel (`cos_theta >
/// 0.9995`), both functions fall back to linear interpolation to avoid
/// numerical instability from dividing by a near-zero `sin(theta)`.
///
/// If `a` and `b` have different lengths, only the shared prefix
/// `[0..min(a.len(), b.len())]` is interpolated (mirroring
/// [`linear_merge`]) — the norms used for normalization, the dot product,
/// and the output are all computed over that same truncated prefix, so
/// ragged inputs are never normalized against a magnitude the
/// interpolation itself doesn't use. Concretely,
/// `slerp(a, b, t) == slerp(&a[..n], &b[..n], t)` where `n =
/// min(a.len(), b.len())`.
///
/// ## Formula
///
/// `result = sin((1-t)*θ)/sin(θ) * a_norm + sin(t*θ)/sin(θ) * b_norm`
///
/// where `θ = acos(dot(a_norm, b_norm))` and `a_norm = a / ||a||`, `b_norm
/// = b / ||b||`.
pub fn slerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    slerp_checked(a, b, t).unwrap_or_else(|_| {
        let n = a.len().min(b.len());
        linear_merge(&a[..n], &b[..n], t)
    })
}

/// Checked variant of [`slerp`] that reports a zero-norm input as an error
/// instead of silently falling back to linear interpolation.
///
/// Applies the same shared-prefix truncation as [`slerp`] for ragged
/// inputs (see its docs). Nearly-parallel directions (`cos_theta >
/// 0.9995`) still fall back to `Ok(linear_merge(..))` — that is a
/// legitimate numerical-stability choice, not an error condition.
///
/// # Errors
///
/// Returns [`MergeError::SlerpZeroVector`] if `a` or `b` (restricted to
/// the shared prefix) has zero (or near-zero, `< f32::EPSILON`) norm —
/// there is no direction to interpolate for a zero vector.
pub fn slerp_checked(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>, MergeError> {
    let n = a.len().min(b.len());
    if n == 0 {
        return Ok(Vec::new());
    }
    let a = &a[..n];
    let b = &b[..n];

    // Compute norms over the same truncated prefix the dot product and
    // output use — computing them over the full, possibly longer slices
    // (as an earlier version of this function did) mis-normalizes ragged
    // inputs against a magnitude the interpolation never actually uses.
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Zero-norm inputs have no direction to interpolate: this is an error,
    // not something to paper over with a silent fallback.
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return Err(MergeError::SlerpZeroVector);
    }

    // Dot product of normalized vectors
    let cos_theta: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai / norm_a) * (bi / norm_b))
        .sum::<f32>()
        .clamp(-1.0, 1.0);

    // Nearly parallel: fall back to linear (numerical-stability choice,
    // not an error — `Ok` on purpose).
    if cos_theta > 0.9995 {
        return Ok(linear_merge(a, b, t));
    }

    let theta = cos_theta.acos();
    let sin_theta = theta.sin();

    // Safety: sin_theta should be > 0 here since |cos_theta| < 0.9995
    if sin_theta.abs() < f32::EPSILON {
        return Ok(linear_merge(a, b, t));
    }

    let coeff_a = ((1.0 - t) * theta).sin() / sin_theta;
    let coeff_b = (t * theta).sin() / sin_theta;

    // Apply the coefficients to the *normalized* vectors — using the raw
    // `ai`/`bi` here was the CQ-M6 bug: it skewed the result toward
    // whichever input had the larger magnitude instead of interpolating
    // pure direction.
    Ok(a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| coeff_a * (ai / norm_a) + coeff_b * (bi / norm_b))
        .collect())
}

/// TIES-Merging: magnitude-based trimming followed by sign-majority election.
///
/// ## Algorithm
///
/// 1. Compute delta vectors: `δa = a - mean(a)`, `δb = b - mean(b)` — but for
///    checkpoint merging we treat `a` and `b` directly as task vectors.
/// 2. **Trim**: for each of `a` and `b` independently, zero out the bottom
///    `(1 - density)` fraction of weights by absolute magnitude.
/// 3. **Elect**: for each position, keep the delta whose magnitude is larger;
///    if they agree in sign use their average scaled by `alpha`, otherwise use
///    the one with larger magnitude scaled by `alpha`.
/// 4. **Result**: `0.5*(a_trimmed + b_trimmed)` weighted by alpha on the
///    agreement region; positions where signs conflict take the dominant sign.
///
/// `density` should be in `(0.0, 1.0]`; 1.0 keeps all weights (no trimming).
pub fn ties_merge(a: &[f32], b: &[f32], alpha: f32, density: f32) -> Vec<f32> {
    let n = a.len().min(b.len());
    if n == 0 {
        return Vec::new();
    }

    // --- Trim step ---
    let trimmed_a = trim_by_magnitude(a, density);
    let trimmed_b = trim_by_magnitude(b, density);

    // --- Sign-majority + magnitude-dominant election ---
    trimmed_a
        .iter()
        .zip(trimmed_b.iter())
        .map(|(va, vb)| {
            let sign_a = va.signum(); // -1.0, 0.0, or 1.0
            let sign_b = vb.signum();
            let abs_a = va.abs();
            let abs_b = vb.abs();

            if sign_a == sign_b {
                // Agree in sign: use weighted average (biased by alpha)
                (1.0 - alpha) * va + alpha * vb
            } else if abs_a >= abs_b {
                // Disagree: dominant sign wins (scale by alpha for blending)
                va * (1.0 - alpha)
            } else {
                vb * alpha
            }
        })
        .collect()
}

/// Task-vector merge: `result[i] = base[i] + alpha * (finetuned[i] - base[i])`.
///
/// - `α = 0.0` → returns `base` unchanged
/// - `α = 1.0` → returns `finetuned` unchanged
/// - Intermediate values apply a fraction of the fine-tuning direction
pub fn task_vector_merge(base: &[f32], finetuned: &[f32], alpha: f32) -> Vec<f32> {
    base.iter()
        .zip(finetuned.iter())
        .map(|(b, f)| b + alpha * (f - b))
        .collect()
}

/// DARE merge: deterministic drop-and-rescale of task-vector elements.
///
/// ## Algorithm
///
/// 1. Compute task vector `δ[i] = finetuned[i] - base[i]`.
/// 2. For each element, use an LCG RNG (seeded with `seed`) to decide whether to
///    zero it out (with probability `dropout_rate`).
/// 3. Rescale surviving elements by `1 / (1 - dropout_rate)` to preserve expected
///    magnitude (analogous to dropout rescaling in neural networks).
/// 4. Return `base[i] + alpha * δ_sparse[i]`.
///
/// The LCG is deterministic: same `seed` always produces the same sparsity mask.
pub fn dare_merge(
    base: &[f32],
    finetuned: &[f32],
    alpha: f32,
    dropout_rate: f32,
    seed: u64,
) -> Vec<f32> {
    let mut state = seed;
    let rescale = if dropout_rate < 1.0 {
        1.0 / (1.0 - dropout_rate)
    } else {
        0.0
    };

    base.iter()
        .zip(finetuned.iter())
        .map(|(b, f)| {
            let rand_val = lcg_next(&mut state);
            let delta = f - b;
            let sparse_delta = if rand_val < dropout_rate {
                0.0
            } else {
                delta * rescale
            };
            b + alpha * sparse_delta
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────
// High-level tensor-level API
// ──────────────────────────────────────────────────────────────────

/// Merge two [`WeightTensor`]s using the configured method.
///
/// Both tensors must have the same element count. If `config.normalize` is set,
/// each tensor is scaled to unit L2 norm before merging.
pub fn merge_tensors(
    base: &WeightTensor,
    other: &WeightTensor,
    config: &MergeConfig,
) -> Result<WeightTensor, MergeError> {
    validate_config(config)?;
    check_compatible(base, other)?;
    if base.element_count() == 0 {
        return Err(MergeError::EmptyTensor(base.name.clone()));
    }

    let (a_data, b_data) = if config.normalize {
        let norm_a = base.l2_norm();
        let norm_b = other.l2_norm();
        let a_norm = if norm_a > f32::EPSILON {
            base.data.iter().map(|x| x / norm_a).collect()
        } else {
            base.data.clone()
        };
        let b_norm = if norm_b > f32::EPSILON {
            other.data.iter().map(|x| x / norm_b).collect()
        } else {
            other.data.clone()
        };
        (a_norm, b_norm)
    } else {
        (base.data.clone(), other.data.clone())
    };

    let merged_data = apply_merge_method(&a_data, &b_data, config)?;
    Ok(WeightTensor::new(
        base.name.clone(),
        merged_data,
        base.shape.clone(),
    ))
}

/// Merge a full model (collection of named tensors) from two sources.
///
/// - Tensors present in **both** models are merged using `config`.
/// - Tensors present only in `base` are copied unchanged into the output.
/// - Tensors present only in `other` are silently ignored.
///
/// The output preserves the ordering of `base`.
pub fn merge_models(
    base: &[WeightTensor],
    other: &[WeightTensor],
    config: &MergeConfig,
) -> Result<Vec<WeightTensor>, MergeError> {
    let (merged, _stats) = merge_models_with_stats(base, other, config)?;
    Ok(merged)
}

/// Merge a full model with statistics collection.
///
/// Returns both the merged weight tensors and a [`MergeStats`] summary.
pub fn merge_models_with_stats(
    base: &[WeightTensor],
    other: &[WeightTensor],
    config: &MergeConfig,
) -> Result<(Vec<WeightTensor>, MergeStats), MergeError> {
    validate_config(config)?;

    // Build a name → index lookup for `other`
    let other_map: HashMap<&str, &WeightTensor> =
        other.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut result = Vec::with_capacity(base.len());
    let mut tensors_merged = 0usize;
    let mut tensors_copied = 0usize;
    let mut total_params = 0usize;
    let mut cosine_sum = 0.0f32;
    let mut cosine_count = 0usize;

    for base_tensor in base {
        total_params += base_tensor.element_count();
        if let Some(other_tensor) = other_map.get(base_tensor.name.as_str()) {
            // Accumulate cosine similarity for stats (best-effort; skip on error)
            if let Ok(sim) = base_tensor.cosine_similarity(other_tensor) {
                cosine_sum += sim;
                cosine_count += 1;
            }

            let merged_tensor = merge_tensors(base_tensor, other_tensor, config)?;
            result.push(merged_tensor);
            tensors_merged += 1;
        } else {
            result.push(base_tensor.clone());
            tensors_copied += 1;
        }
    }

    let mean_cosine_similarity = if cosine_count > 0 {
        cosine_sum / cosine_count as f32
    } else {
        0.0
    };

    let stats = MergeStats {
        tensors_merged,
        tensors_copied,
        total_params,
        mean_cosine_similarity,
        method: config.method.clone(),
    };

    Ok((result, stats))
}

// ──────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────

/// Validate that `config.alpha` and `config.density` are in their allowed ranges.
fn validate_config(config: &MergeConfig) -> Result<(), MergeError> {
    if !(0.0..=1.0).contains(&config.alpha) {
        return Err(MergeError::InvalidAlpha(config.alpha));
    }
    if config.density <= 0.0 || config.density > 1.0 {
        return Err(MergeError::InvalidDensity(config.density));
    }
    Ok(())
}

/// Return `Err(ShapeMismatch)` when two tensors have incompatible element counts.
fn check_compatible(a: &WeightTensor, b: &WeightTensor) -> Result<(), MergeError> {
    if a.element_count() != b.element_count() {
        return Err(MergeError::ShapeMismatch {
            name: a.name.clone(),
            a: a.shape.clone(),
            b: b.shape.clone(),
        });
    }
    Ok(())
}

/// Dispatch to the appropriate primitive merge function based on `config.method`.
fn apply_merge_method(a: &[f32], b: &[f32], config: &MergeConfig) -> Result<Vec<f32>, MergeError> {
    match &config.method {
        MergeMethod::Linear => Ok(linear_merge(a, b, config.alpha)),
        // `slerp_checked` performs its own zero-vector check (returning
        // `Err(MergeError::SlerpZeroVector)`), so no separate pre-check is
        // needed here — this also picks up the normalized-coefficient fix
        // for every caller that goes through `merge_tensors`/`merge_models`.
        MergeMethod::Slerp => slerp_checked(a, b, config.alpha),
        MergeMethod::Ties => Ok(ties_merge(a, b, config.alpha, config.density)),
        MergeMethod::TaskVector => Ok(task_vector_merge(a, b, config.alpha)),
        MergeMethod::Dare { seed, dropout_rate } => {
            Ok(dare_merge(a, b, config.alpha, *dropout_rate, *seed))
        }
    }
}

/// Trim the bottom `(1 - density)` fraction of elements by absolute magnitude.
///
/// Elements with magnitude below the computed threshold are zeroed out.
/// With `density = 1.0` all elements are kept (no-op). With `density = 0.5`
/// the lower half by magnitude is zeroed.
fn trim_by_magnitude(data: &[f32], density: f32) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    if density >= 1.0 {
        return data.to_vec();
    }

    // Collect absolute values and sort to find threshold
    let mut abs_sorted: Vec<f32> = data.iter().map(|x| x.abs()).collect();
    abs_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    // The index below which we trim: trim (1-density) fraction.
    // Use round() to avoid float-precision edge cases (e.g. 0.4*5 = 1.9999... → 1 without round).
    let trim_count = ((1.0 - density) * abs_sorted.len() as f32).round() as usize;
    let threshold = if trim_count < abs_sorted.len() {
        abs_sorted[trim_count]
    } else {
        f32::MAX
    };

    data.iter()
        .map(|x| if x.abs() < threshold { 0.0 } else { *x })
        .collect()
}

/// Deterministic LCG (linear congruential generator) producing values in `[0.0, 1.0)`.
///
/// Uses the Knuth/MMIX parameters:
/// `state = state * 6364136223846793005 + 1442695040888963407`
///
/// The upper 32 bits of the new state are extracted and mapped to `[0, 1)` by
/// dividing by `2^32` (i.e., `u32::MAX + 1`). This gives a full-range uniform
/// distribution over the 32-bit space.
#[inline]
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Extract upper 32 bits and scale to [0, 1)
    let bits = (*state >> 32) as u32;
    (bits as f32) / (u32::MAX as f32 + 1.0)
}

// ──────────────────────────────────────────────────────────────────
// Unit tests (in-module smoke tests; full suite in tests/)
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// CQ-M6 regression: `slerp` used to compute the normalizing norms over
    /// the *full* `a`/`b` slices while the dot product and output only used
    /// the shared `[..n]` prefix, so a ragged call was interpolated against
    /// the wrong magnitude. `slerp(a, b, t)` on ragged inputs must be
    /// byte-for-byte identical to calling it on the already-truncated
    /// prefixes.
    #[test]
    fn slerp_ragged_inputs_match_pre_truncated_equivalent() {
        let a = vec![1.0f32, 0.0, 100.0];
        let b = vec![0.0f32, 1.0];
        let ragged = slerp(&a, &b, 0.5);
        let truncated = slerp(&a[..2], &b, 0.5);
        assert_eq!(
            ragged, truncated,
            "slerp on ragged inputs must equal slerp on the pre-truncated prefix"
        );
    }

    /// Same regression, checked against the closed-form expectation: with
    /// the bug, the huge `100.0` tail element inflated `norm_a`, which
    /// suppressed `cos_theta` far below the true 2-D prefix cosine (0.0 for
    /// these orthogonal unit vectors) and produced a materially different
    /// interpolation than the fixed, prefix-consistent computation.
    #[test]
    fn slerp_ragged_inputs_orthogonal_prefix_matches_expected_midpoint() {
        let a = vec![1.0f32, 0.0, 100.0];
        let b = vec![0.0f32, 1.0];
        let result = slerp(&a, &b, 0.5);
        let expected = 1.0_f32 / 2.0_f32.sqrt();
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "expected x ~= 1/sqrt(2) = {expected}, got {}",
            result[0]
        );
        assert!(
            (result[1] - expected).abs() < 1e-5,
            "expected y ~= 1/sqrt(2) = {expected}, got {}",
            result[1]
        );
    }

    /// CQ-M6 regression: the SLERP coefficients must be applied to the
    /// *normalized* vectors, not the raw inputs. `a = [3, 0]` and `b = [0,
    /// 1]` are orthogonal, so their true angular midpoint at `t = 0.5` is
    /// exactly 45 degrees off the x-axis regardless of `a`'s magnitude.
    /// With the bug (coefficients applied to raw `a`/`b`), the result was
    /// skewed toward the larger-magnitude `a` and landed at ~18.435
    /// degrees instead.
    #[test]
    fn slerp_normalizes_before_applying_coefficients() {
        let a = vec![3.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let result = slerp(&a, &b, 0.5);
        let angle_deg = result[1].atan2(result[0]).to_degrees();
        assert!(
            (angle_deg - 45.0).abs() < 1e-3,
            "slerp midpoint of orthogonal [3,0]/[0,1] must be at 45 degrees \
             regardless of magnitude, got {angle_deg} degrees (result={result:?})"
        );
    }

    /// CQ-M6: a zero-norm input has no direction to interpolate, so
    /// [`slerp_checked`] must report it as an error rather than silently
    /// falling back to a linear blend.
    #[test]
    fn slerp_checked_errors_on_zero_norm_input() {
        let zero = vec![0.0f32, 0.0];
        let unit = vec![0.0f32, 1.0];
        let err = slerp_checked(&zero, &unit, 0.5)
            .expect_err("zero-norm input must be rejected, not silently linearised");
        assert!(matches!(err, MergeError::SlerpZeroVector));

        // The infallible `slerp` wrapper still degrades gracefully to the
        // documented linear fallback for the same input.
        let fallback = slerp(&zero, &unit, 0.5);
        assert_eq!(fallback, linear_merge(&zero, &unit, 0.5));
    }

    #[test]
    fn lcg_produces_values_in_unit_interval() {
        let mut state = 42u64;
        for _ in 0..1000 {
            let v = lcg_next(&mut state);
            assert!((0.0..=1.0).contains(&v), "lcg value {v} out of [0,1]");
        }
    }

    #[test]
    fn trim_by_magnitude_density_one_noop() {
        let data = vec![0.1, 0.5, -0.3, 0.9, -0.7];
        let trimmed = trim_by_magnitude(&data, 1.0);
        assert_eq!(trimmed, data);
    }

    #[test]
    fn trim_by_magnitude_zeros_smallest() {
        // density = 0.6 → keep top 60%, trim bottom 40%
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trimmed = trim_by_magnitude(&data, 0.6);
        // Bottom 40% of 5 = 2 elements (1.0 and 2.0) should be zeroed
        assert_eq!(trimmed[0], 0.0, "1.0 should be trimmed");
        assert_eq!(trimmed[1], 0.0, "2.0 should be trimmed");
        assert!(trimmed[2] != 0.0, "3.0 should be kept");
    }

    #[test]
    fn validate_config_rejects_bad_alpha() {
        let config = MergeConfig {
            alpha: 1.5,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_zero_density() {
        let config = MergeConfig {
            density: 0.0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }
}
