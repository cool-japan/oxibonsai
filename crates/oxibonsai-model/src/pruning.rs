//! Weight importance analysis and structured/unstructured pruning.
//!
//! Pruning reduces model size and inference cost by zeroing or removing
//! less-important weights. This module implements:
//! - Magnitude-based unstructured pruning
//! - Structured pruning (entire rows/columns)
//! - Importance scoring (L1, L2, gradient sensitivity approximation)
//! - Sparsity analysis and reporting

use crate::model_merge::WeightTensor;
use thiserror::Error;

// ──────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────

/// Errors that can occur during pruning operations.
#[derive(Debug, Error)]
pub enum PruningError {
    #[error("sparsity {0} must be in [0.0, 1.0)")]
    InvalidSparsity(f32),
    #[error("empty tensor: '{0}'")]
    EmptyTensor(String),
    #[error("structured pruning requires 2D tensor, got shape {0:?}")]
    NotTwoDimensional(Vec<usize>),
    #[error("cannot prune below min_nonzero={0} with {1} total elements")]
    BelowMinNonzero(usize, usize),
}

// ──────────────────────────────────────────────────────────────────
// ImportanceMetric
// ──────────────────────────────────────────────────────────────────

/// How to compute weight importance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImportanceMetric {
    /// L1 norm of each weight (|w|).
    L1Magnitude,
    /// L2 norm of each weight (w^2).
    L2Magnitude,
    /// Taylor first-order approximation: |w * gradient|.
    /// Since we don't run gradients at inference time, uses |w| * |w| as a proxy.
    TaylorProxy,
    /// Random importance (for baseline/ablation).
    /// Uses a seeded LCG for reproducibility.
    Random { seed: u64 },
}

// ──────────────────────────────────────────────────────────────────
// PruningGranularity
// ──────────────────────────────────────────────────────────────────

/// Pruning granularity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningGranularity {
    /// Zero individual weights.
    Unstructured,
    /// Zero entire rows (output neurons).
    StructuredRow,
    /// Zero entire columns (input features).
    StructuredColumn,
}

// ──────────────────────────────────────────────────────────────────
// PruningConfig
// ──────────────────────────────────────────────────────────────────

/// Configuration for a pruning pass.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Target fraction of zeros (0.0 - 1.0).
    pub sparsity: f32,
    /// Which metric to use for computing importance scores.
    pub metric: ImportanceMetric,
    /// Whether to prune individual weights or entire rows/columns.
    pub granularity: PruningGranularity,
    /// Minimum number of non-zero elements to keep (safety floor).
    pub min_nonzero: usize,
}

impl PruningConfig {
    /// Create a new pruning config with the given parameters.
    pub fn new(sparsity: f32, metric: ImportanceMetric, granularity: PruningGranularity) -> Self {
        Self {
            sparsity,
            metric,
            granularity,
            min_nonzero: 1,
        }
    }

    /// Convenience: unstructured L1-magnitude pruning at the given sparsity.
    pub fn unstructured_l1(sparsity: f32) -> Self {
        Self::new(
            sparsity,
            ImportanceMetric::L1Magnitude,
            PruningGranularity::Unstructured,
        )
    }

    /// Convenience: structured row pruning using L2 norm at the given sparsity.
    pub fn structured_row_l2(sparsity: f32) -> Self {
        Self::new(
            sparsity,
            ImportanceMetric::L2Magnitude,
            PruningGranularity::StructuredRow,
        )
    }
}

// ──────────────────────────────────────────────────────────────────
// ScoreStats
// ──────────────────────────────────────────────────────────────────

/// Summary statistics over importance scores.
#[derive(Debug, Clone)]
pub struct ScoreStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
}

// ──────────────────────────────────────────────────────────────────
// ImportanceScores
// ──────────────────────────────────────────────────────────────────

/// Importance score for each element (or row/column for structured).
#[derive(Debug, Clone)]
pub struct ImportanceScores {
    /// One score per element (unstructured) or row/column (structured).
    pub scores: Vec<f32>,
    /// Score below which elements are pruned.
    pub threshold: f32,
    /// The metric used to generate these scores.
    pub metric: ImportanceMetric,
}

impl ImportanceScores {
    /// Fraction of scores at or below the threshold.
    pub fn sparsity(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let below = self.scores.iter().filter(|&&s| s <= self.threshold).count();
        below as f32 / self.scores.len() as f32
    }

    /// Return the top-k scores in descending order.
    pub fn top_k(&self, k: usize) -> Vec<f32> {
        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }

    /// Compute summary statistics over all scores.
    pub fn stats(&self) -> ScoreStats {
        if self.scores.is_empty() {
            return ScoreStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
            };
        }

        let n = self.scores.len();
        let min = self.scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self
            .scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean = self.scores.iter().sum::<f32>() / n as f32;

        let variance = self.scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();

        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        ScoreStats {
            min,
            max,
            mean,
            median,
            std_dev,
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// SparsityReport
// ──────────────────────────────────────────────────────────────────

/// Analyze sparsity of a tensor.
#[derive(Debug, Clone)]
pub struct SparsityReport {
    pub name: String,
    pub total_params: usize,
    pub nonzero_params: usize,
    pub sparsity: f32,
    pub shape: Vec<usize>,
}

impl SparsityReport {
    /// Compute a sparsity report for the given tensor.
    pub fn compute(tensor: &WeightTensor) -> Self {
        let total_params = tensor.data.len();
        let nonzero_params = tensor.data.iter().filter(|&&x| x != 0.0).count();
        let sparsity = if total_params == 0 {
            0.0
        } else {
            1.0 - nonzero_params as f32 / total_params as f32
        };
        Self {
            name: tensor.name.clone(),
            total_params,
            nonzero_params,
            sparsity,
            shape: tensor.shape.clone(),
        }
    }

    /// Fraction of zeros — same as `sparsity`.
    pub fn zero_fraction(&self) -> f32 {
        self.sparsity
    }

    /// Fraction of non-zero elements.
    pub fn density(&self) -> f32 {
        1.0 - self.sparsity
    }

    /// Human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "tensor='{}' shape={:?} total={} nonzero={} sparsity={:.4}",
            self.name, self.shape, self.total_params, self.nonzero_params, self.sparsity,
        )
    }
}

// ──────────────────────────────────────────────────────────────────
// ModelSparsitySummary
// ──────────────────────────────────────────────────────────────────

/// Aggregate sparsity across all layers.
pub struct ModelSparsitySummary {
    pub layer_reports: Vec<SparsityReport>,
    pub total_params: usize,
    pub total_nonzero: usize,
    pub overall_sparsity: f32,
}

impl ModelSparsitySummary {
    /// Build a summary from a slice of weight tensors.
    pub fn from_model(tensors: &[WeightTensor]) -> Self {
        let layer_reports: Vec<SparsityReport> =
            tensors.iter().map(SparsityReport::compute).collect();
        let total_params: usize = layer_reports.iter().map(|r| r.total_params).sum();
        let total_nonzero: usize = layer_reports.iter().map(|r| r.nonzero_params).sum();
        let overall_sparsity = if total_params == 0 {
            0.0
        } else {
            1.0 - total_nonzero as f32 / total_params as f32
        };
        Self {
            layer_reports,
            total_params,
            total_nonzero,
            overall_sparsity,
        }
    }

    /// Human-readable summary of the entire model's sparsity.
    pub fn summary(&self) -> String {
        format!(
            "layers={} total_params={} total_nonzero={} overall_sparsity={:.4}",
            self.layer_reports.len(),
            self.total_params,
            self.total_nonzero,
            self.overall_sparsity,
        )
    }
}

// ──────────────────────────────────────────────────────────────────
// Public API functions
// ──────────────────────────────────────────────────────────────────

/// Compute importance scores for a weight tensor.
///
/// For unstructured metrics (L1, L2, TaylorProxy, Random), one score per element.
/// The threshold field is set to 0.0 (no pruning decision is made here).
pub fn compute_importance(tensor: &WeightTensor, metric: ImportanceMetric) -> ImportanceScores {
    let scores = match metric {
        ImportanceMetric::L1Magnitude => tensor.data.iter().map(|x| x.abs()).collect(),
        ImportanceMetric::L2Magnitude => tensor.data.iter().map(|x| x * x).collect(),
        ImportanceMetric::TaylorProxy => tensor.data.iter().map(|x| x * x).collect(),
        ImportanceMetric::Random { seed } => {
            let mut state = seed;
            tensor.data.iter().map(|_| lcg_next(&mut state)).collect()
        }
    };
    ImportanceScores {
        scores,
        threshold: 0.0,
        metric,
    }
}

/// Prune a tensor: zero out low-importance weights.
///
/// Returns the pruned tensor and a mask (1.0 = kept, 0.0 = pruned).
pub fn prune_tensor(
    tensor: &WeightTensor,
    config: &PruningConfig,
) -> Result<(WeightTensor, Vec<f32>), PruningError> {
    let mut cloned = tensor.clone();
    let mask = prune_tensor_inplace(&mut cloned, config)?;
    Ok((cloned, mask))
}

/// Prune a tensor in-place, returning only the mask.
pub fn prune_tensor_inplace(
    tensor: &mut WeightTensor,
    config: &PruningConfig,
) -> Result<Vec<f32>, PruningError> {
    validate_sparsity(config.sparsity)?;

    let n = tensor.data.len();
    if n == 0 {
        return Err(PruningError::EmptyTensor(tensor.name.clone()));
    }

    match config.granularity {
        PruningGranularity::Unstructured => prune_unstructured(tensor, config),
        PruningGranularity::StructuredRow => prune_structured(tensor, config, true),
        PruningGranularity::StructuredColumn => prune_structured(tensor, config, false),
    }
}

/// Prune a full model (all tensors) with a shared sparsity config.
pub fn prune_model(
    tensors: &[WeightTensor],
    config: &PruningConfig,
) -> Result<Vec<WeightTensor>, PruningError> {
    tensors
        .iter()
        .map(|t| {
            let (pruned, _mask) = prune_tensor(t, config)?;
            Ok(pruned)
        })
        .collect()
}

/// Compute sparsity reports for all tensors in a model.
pub fn model_sparsity_report(tensors: &[WeightTensor]) -> Vec<SparsityReport> {
    tensors.iter().map(SparsityReport::compute).collect()
}

// ──────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────

/// Deterministic LCG producing values in `[0.0, 1.0)`.
#[inline]
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 32) as u32;
    (bits as f32) / (u32::MAX as f32 + 1.0)
}

fn validate_sparsity(sparsity: f32) -> Result<(), PruningError> {
    if !(0.0..1.0).contains(&sparsity) {
        return Err(PruningError::InvalidSparsity(sparsity));
    }
    Ok(())
}

/// Compute element-wise importance scores as a flat Vec<f32>.
fn compute_element_scores(data: &[f32], metric: ImportanceMetric) -> Vec<f32> {
    match metric {
        ImportanceMetric::L1Magnitude => data.iter().map(|x| x.abs()).collect(),
        ImportanceMetric::L2Magnitude => data.iter().map(|x| x * x).collect(),
        ImportanceMetric::TaylorProxy => data.iter().map(|x| x * x).collect(),
        ImportanceMetric::Random { seed } => {
            let mut state = seed;
            data.iter().map(|_| lcg_next(&mut state)).collect()
        }
    }
}

/// Unstructured pruning: zero individual elements below threshold.
fn prune_unstructured(
    tensor: &mut WeightTensor,
    config: &PruningConfig,
) -> Result<Vec<f32>, PruningError> {
    let n = tensor.data.len();
    let scores = compute_element_scores(&tensor.data, config.metric);

    // Determine how many elements to prune
    let num_to_prune = (config.sparsity * n as f32).floor() as usize;
    // Ensure min_nonzero constraint
    let max_to_prune = n.saturating_sub(config.min_nonzero);
    if config.min_nonzero > n {
        return Err(PruningError::BelowMinNonzero(config.min_nonzero, n));
    }
    let num_to_prune = num_to_prune.min(max_to_prune);

    if num_to_prune == 0 {
        // No pruning needed — full mask of ones
        return Ok(vec![1.0f32; n]);
    }

    // Find threshold: sort scores to find the num_to_prune-th smallest
    let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = indexed[num_to_prune - 1].1;

    // Build mask: prune the num_to_prune lowest-scoring elements
    let mut mask = vec![1.0f32; n];
    let mut pruned_count = 0usize;
    for (orig_idx, score) in &indexed {
        if pruned_count >= num_to_prune {
            break;
        }
        if *score <= threshold {
            mask[*orig_idx] = 0.0;
            tensor.data[*orig_idx] = 0.0;
            pruned_count += 1;
        }
    }

    Ok(mask)
}

/// Structured pruning: zero entire rows or columns.
fn prune_structured(
    tensor: &mut WeightTensor,
    config: &PruningConfig,
    prune_rows: bool,
) -> Result<Vec<f32>, PruningError> {
    if tensor.shape.len() != 2 {
        return Err(PruningError::NotTwoDimensional(tensor.shape.clone()));
    }

    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    let (num_units, unit_size) = if prune_rows {
        (rows, cols)
    } else {
        (cols, rows)
    };

    // Compute per-unit (row or column) importance score
    let unit_scores: Vec<f32> = (0..num_units)
        .map(|u| {
            let slice: Vec<f32> = if prune_rows {
                tensor.data[u * cols..(u + 1) * cols].to_vec()
            } else {
                // column: gather every cols-th element
                (0..rows).map(|r| tensor.data[r * cols + u]).collect()
            };
            match config.metric {
                ImportanceMetric::L1Magnitude => slice.iter().map(|x| x.abs()).sum::<f32>(),
                ImportanceMetric::L2Magnitude => slice.iter().map(|x| x * x).sum::<f32>().sqrt(),
                ImportanceMetric::TaylorProxy => slice.iter().map(|x| x * x).sum::<f32>().sqrt(),
                ImportanceMetric::Random { seed } => {
                    let mut state = seed.wrapping_add(u as u64);
                    lcg_next(&mut state)
                }
            }
        })
        .collect();

    let num_to_prune = (config.sparsity * num_units as f32).floor() as usize;
    let max_to_prune = num_units.saturating_sub(config.min_nonzero.div_ceil(unit_size));
    if config.min_nonzero > num_units * unit_size {
        return Err(PruningError::BelowMinNonzero(
            config.min_nonzero,
            num_units * unit_size,
        ));
    }
    let num_to_prune = num_to_prune.min(max_to_prune);

    if num_to_prune == 0 {
        return Ok(vec![1.0f32; tensor.data.len()]);
    }

    // Sort units by score ascending
    let mut indexed: Vec<(usize, f32)> = unit_scores.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Mark units to prune
    let mut units_to_prune = std::collections::HashSet::new();
    for (unit_idx, _score) in indexed.iter().take(num_to_prune) {
        units_to_prune.insert(*unit_idx);
    }

    // Build mask and zero tensor
    let total = tensor.data.len();
    let mut mask = vec![1.0f32; total];

    for (idx, slot) in mask.iter_mut().enumerate().take(total) {
        let unit = if prune_rows { idx / cols } else { idx % cols };
        if units_to_prune.contains(&unit) {
            *slot = 0.0;
            tensor.data[idx] = 0.0;
        }
    }

    Ok(mask)
}

// ──────────────────────────────────────────────────────────────────
// In-module smoke tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(name: &str, data: Vec<f32>, shape: Vec<usize>) -> WeightTensor {
        WeightTensor::new(name, data, shape)
    }

    #[test]
    fn lcg_values_in_unit_interval() {
        let mut state = 12345u64;
        for _ in 0..1000 {
            let v = lcg_next(&mut state);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn compute_importance_l1_basic() {
        let t = make_tensor("w", vec![-2.0, 1.0, -0.5], vec![3]);
        let scores = compute_importance(&t, ImportanceMetric::L1Magnitude);
        assert!((scores.scores[0] - 2.0).abs() < 1e-6);
        assert!((scores.scores[1] - 1.0).abs() < 1e-6);
        assert!((scores.scores[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unstructured_prune_zeroes_smallest() {
        let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let t = make_tensor("w", data, vec![10]);
        let config = PruningConfig::unstructured_l1(0.3);
        let (pruned, mask) = prune_tensor(&t, &config).expect("prune ok");
        // 3 lowest elements (1,2,3) should be zero
        assert_eq!(pruned.data[0], 0.0);
        assert_eq!(pruned.data[1], 0.0);
        assert_eq!(pruned.data[2], 0.0);
        assert!(pruned.data[9] != 0.0);
        assert!(mask.iter().all(|&m| m == 0.0 || m == 1.0));
    }
}
