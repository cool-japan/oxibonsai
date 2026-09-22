//! Model compression pipeline: prune → quantize → report.
//!
//! Combines pruning and quantization stages into a unified pipeline that can
//! be configured and applied to a collection of weight tensors. Each stage
//! records statistics that are aggregated into a final `CompressionResult`.
//!
//! ## Reachability
//!
//! As of this writing, nothing in the CLI or the export pipeline calls
//! [`compress_model`] — there is no `oxibonsai compress`-style entry point
//! that runs this pipeline against a real model. This module is a tested,
//! documented primitive for that future work, not a wired feature.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_model::compression::{CompressionConfig, compress_model};
//! use oxibonsai_model::model_merge::WeightTensor;
//!
//! let tensors = vec![
//!     WeightTensor::new("layer.weight", vec![1.0, -0.5, 0.3, 2.0], vec![2, 2]),
//! ];
//! let config = CompressionConfig::prune_then_quantize(0.5);
//! let result = compress_model(&tensors, &config).expect("compression failed");
//! println!("{}", result.summary());
//! ```

use crate::model_merge::WeightTensor;
use crate::pruning::{prune_tensor, PruningConfig, PruningError};

// ──────────────────────────────────────────────────────────────────
// CompressionStage
// ──────────────────────────────────────────────────────────────────

/// A compression stage to apply in sequence.
#[derive(Debug, Clone)]
pub enum CompressionStage {
    /// Prune weights to target sparsity using the provided configuration.
    Prune(PruningConfig),
    /// Quantize to INT8 (per-tensor, simulate INT8 precision loss while
    /// keeping `f32` storage in memory). [`StageStats::memory_after_bytes`]
    /// honestly reports that the tensors are still `f32` after this stage
    /// (unchanged from `memory_before_bytes`); the theoretical INT8-encoded
    /// size (`original * 0.25`) is reported separately as
    /// [`StageStats::encoded_bytes_after`], and that is what
    /// [`StageStats::compression_ratio`] is computed against.
    QuantizeInt8,
    /// Apply magnitude-based weight clipping: zero out weights whose absolute value
    /// falls below the given percentile of absolute values in each tensor.
    Clip {
        /// Fraction in `(0.0, 1.0]` — e.g. `0.1` zeros the bottom 10% of weights.
        percentile: f32,
    },
}

impl CompressionStage {
    /// Human-readable name for this stage.
    pub fn name(&self) -> &'static str {
        match self {
            CompressionStage::Prune(_) => "prune",
            CompressionStage::QuantizeInt8 => "quantize_int8",
            CompressionStage::Clip { .. } => "clip",
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// CompressionConfig
// ──────────────────────────────────────────────────────────────────

/// Full pipeline configuration: ordered list of stages and global options.
#[derive(Debug, Clone, Default)]
pub struct CompressionConfig {
    /// Ordered sequence of compression stages to apply.
    pub stages: Vec<CompressionStage>,
    /// When `true`, tensors whose name starts with `"embed"` or `"token"` are
    /// skipped (not passed through any compression stage).
    pub skip_embedding_layers: bool,
}

impl CompressionConfig {
    /// Create a new, empty compression config (no stages).
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            skip_embedding_layers: false,
        }
    }

    /// Append a stage and return `self` for chaining.
    pub fn add_stage(mut self, stage: CompressionStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Convenience: L1-unstructured pruning at `sparsity`, followed by INT8 quantization.
    pub fn prune_then_quantize(sparsity: f32) -> Self {
        let prune_cfg = PruningConfig::unstructured_l1(sparsity);
        Self::new()
            .add_stage(CompressionStage::Prune(prune_cfg))
            .add_stage(CompressionStage::QuantizeInt8)
    }

    /// Convenience: INT8 quantization only.
    pub fn quantize_only() -> Self {
        Self::new().add_stage(CompressionStage::QuantizeInt8)
    }

    /// Convenience: L1-unstructured pruning only at `sparsity`.
    pub fn prune_only(sparsity: f32) -> Self {
        let prune_cfg = PruningConfig::unstructured_l1(sparsity);
        Self::new().add_stage(CompressionStage::Prune(prune_cfg))
    }
}

// ──────────────────────────────────────────────────────────────────
// StageStats
// ──────────────────────────────────────────────────────────────────

/// Per-stage compression statistics.
#[derive(Debug, Clone)]
pub struct StageStats {
    /// Name of the stage that produced these stats.
    pub stage_name: String,
    /// Number of tensors that were processed by this stage.
    pub tensors_processed: usize,
    /// Number of tensors skipped (e.g. embedding layers with `skip_embedding_layers`).
    pub tensors_skipped: usize,
    /// Total number of parameters (elements) entering this stage.
    pub params_before: usize,
    /// Number of non-zero parameters after this stage.
    pub nonzero_params_after: usize,
    /// Total memory (bytes) of all processed tensors before this stage.
    pub memory_before_bytes: usize,
    /// **Real, measured** memory (bytes) the processed tensors occupy in
    /// this process's memory *after* this stage — their actual current
    /// `f32` storage footprint. This pipeline never changes a tensor's
    /// element count or its in-memory representation (even
    /// [`CompressionStage::QuantizeInt8`] dequantizes back to `f32` to
    /// simulate precision loss), so today this is always equal to
    /// `memory_before_bytes`. It is reported honestly rather than assuming
    /// the stage's target on-disk encoding — see
    /// [`encoded_bytes_after`][Self::encoded_bytes_after] for that.
    pub memory_after_bytes: usize,
    /// Size (bytes) the processed tensors *would* occupy if actually
    /// re-encoded into this stage's target format (e.g. 1 byte/element for
    /// [`CompressionStage::QuantizeInt8`]). This is a computed estimate,
    /// not a measurement of anything currently resident in memory; it is
    /// what [`compression_ratio`][Self::compression_ratio] is computed
    /// against. For [`CompressionStage::Prune`] and
    /// [`CompressionStage::Clip`], which do not change the on-disk
    /// encoding, this equals `memory_after_bytes`.
    pub encoded_bytes_after: usize,
}

impl StageStats {
    /// Ratio of `memory_before_bytes / encoded_bytes_after`. Returns `1.0`
    /// if `encoded_bytes_after` is zero to avoid division by zero.
    ///
    /// This is computed against the *target-encoding* estimate, not the
    /// real post-stage memory footprint (`memory_after_bytes`), since this
    /// pipeline keeps `f32` storage in memory throughout — see
    /// [`encoded_bytes_after`][Self::encoded_bytes_after].
    pub fn compression_ratio(&self) -> f32 {
        if self.encoded_bytes_after == 0 {
            return 1.0;
        }
        self.memory_before_bytes as f32 / self.encoded_bytes_after as f32
    }

    /// Fraction of parameters that are zero after this stage.
    pub fn sparsity(&self) -> f32 {
        if self.params_before == 0 {
            return 0.0;
        }
        let zeros = self.params_before.saturating_sub(self.nonzero_params_after);
        zeros as f32 / self.params_before as f32
    }
}

// ──────────────────────────────────────────────────────────────────
// CompressionResult
// ──────────────────────────────────────────────────────────────────

/// The outcome of running the full compression pipeline.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// The compressed (and possibly sparsified) tensors in the original order.
    pub compressed_tensors: Vec<WeightTensor>,
    /// One `StageStats` entry per stage in the pipeline.
    pub stage_stats: Vec<StageStats>,
}

impl CompressionResult {
    /// Total number of parameters across all compressed tensors.
    pub fn total_params(&self) -> usize {
        self.compressed_tensors.iter().map(|t| t.data.len()).sum()
    }

    /// Total number of non-zero parameters across all compressed tensors.
    pub fn total_nonzero(&self) -> usize {
        self.compressed_tensors
            .iter()
            .map(|t| t.data.iter().filter(|&&x| x != 0.0).count())
            .sum()
    }

    /// Overall sparsity (fraction of zero weights) across all compressed tensors.
    pub fn overall_sparsity(&self) -> f32 {
        let total = self.total_params();
        if total == 0 {
            return 0.0;
        }
        let nonzero = self.total_nonzero();
        let zeros = total.saturating_sub(nonzero);
        zeros as f32 / total as f32
    }

    /// Compression ratio: `memory_before / encoded_bytes_after` using the
    /// first stage's `memory_before_bytes` and the last stage's
    /// `encoded_bytes_after` (the target-encoding estimate, not the real
    /// in-memory footprint — see [`StageStats::compression_ratio`]). Falls
    /// back to `1.0` if there are no stages.
    pub fn total_compression_ratio(&self) -> f32 {
        if self.stage_stats.is_empty() {
            return 1.0;
        }
        let before = self.memory_before_bytes();
        let after = self.encoded_bytes_after();
        if after == 0 {
            return 1.0;
        }
        before as f32 / after as f32
    }

    /// Memory (bytes) before any compression: taken from the first stage's
    /// `memory_before_bytes`. Returns `0` if there are no stage stats.
    pub fn memory_before_bytes(&self) -> usize {
        self.stage_stats
            .first()
            .map(|s| s.memory_before_bytes)
            .unwrap_or(0)
    }

    /// **Real, measured** memory (bytes) resident after all compression
    /// stages: taken from the last stage's `memory_after_bytes`. Returns
    /// `0` if there are no stage stats. See
    /// [`StageStats::memory_after_bytes`] — this pipeline keeps `f32`
    /// storage throughout, so this is typically equal to
    /// `memory_before_bytes`.
    pub fn memory_after_bytes(&self) -> usize {
        self.stage_stats
            .last()
            .map(|s| s.memory_after_bytes)
            .unwrap_or(0)
    }

    /// Target-encoding size estimate (bytes) after all compression stages:
    /// taken from the last stage's `encoded_bytes_after`. Returns `0` if
    /// there are no stage stats. See [`StageStats::encoded_bytes_after`].
    pub fn encoded_bytes_after(&self) -> usize {
        self.stage_stats
            .last()
            .map(|s| s.encoded_bytes_after)
            .unwrap_or(0)
    }

    /// Human-readable multi-line summary of all stages and overall statistics.
    pub fn summary(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!(
            "=== Compression Summary ({} stage(s)) ===",
            self.stage_stats.len()
        ));
        for (i, stats) in self.stage_stats.iter().enumerate() {
            lines.push(format!(
                "  Stage {}: [{}] processed={} skipped={} sparsity={:.4} ratio={:.3}x \
                 memory={}B->{}B (real, unchanged in this simulation) encoded-target={}B",
                i + 1,
                stats.stage_name,
                stats.tensors_processed,
                stats.tensors_skipped,
                stats.sparsity(),
                stats.compression_ratio(),
                stats.memory_before_bytes,
                stats.memory_after_bytes,
                stats.encoded_bytes_after,
            ));
        }
        lines.push(format!(
            "  Overall: tensors={} total_params={} nonzero={} sparsity={:.4} \
             compression_ratio={:.3}x memory={}B->{}B (real) encoded-target={}B",
            self.compressed_tensors.len(),
            self.total_params(),
            self.total_nonzero(),
            self.overall_sparsity(),
            self.total_compression_ratio(),
            self.memory_before_bytes(),
            self.memory_after_bytes(),
            self.encoded_bytes_after(),
        ));
        lines.join("\n")
    }
}

// ──────────────────────────────────────────────────────────────────
// CompressionError
// ──────────────────────────────────────────────────────────────────

/// Errors that can arise during the compression pipeline.
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    /// Wraps an underlying `PruningError` from the pruning stage.
    #[error("pruning error: {0}")]
    Pruning(#[from] PruningError),

    /// The input tensor slice is empty; there is nothing to compress.
    #[error("empty model: no tensors")]
    EmptyModel,

    /// The pipeline has no stages configured; nothing would be done.
    #[error("empty pipeline: no stages")]
    EmptyPipeline,

    /// The clip percentile is outside the valid range `(0.0, 1.0]`.
    #[error("invalid clip percentile {0}: must be in (0, 1]")]
    InvalidPercentile(f32),
}

// ──────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────

/// Return `true` if this tensor should be skipped based on `skip_embedding_layers`.
#[inline]
fn is_embedding_layer(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("embed") || lower.starts_with("token")
}

/// Bytes occupied by a `WeightTensor` in f32 form.
#[inline]
fn tensor_bytes(tensor: &WeightTensor) -> usize {
    tensor.data.len() * core::mem::size_of::<f32>()
}

/// Count non-zero values in a tensor.
#[inline]
fn count_nonzero(tensor: &WeightTensor) -> usize {
    tensor.data.iter().filter(|&&x| x != 0.0).count()
}

/// Apply the INT8 quantization stage to a single tensor in-place.
///
/// Quantises each tensor per-tensor using `scale = max(|w|) / 127`,
/// then dequantises back to f32 to simulate the precision loss. The tensor
/// remains `f32` in memory (see [`StageStats::memory_after_bytes`]); the
/// theoretical INT8-encoded size is tracked separately as
/// `encoded_bytes_after = memory_before * 0.25` via
/// [`quantize_int8_encoded_bytes`].
fn apply_quantize_int8_inplace(tensor: &mut WeightTensor) {
    let data = &mut tensor.data;
    if data.is_empty() {
        return;
    }

    // Compute per-tensor max absolute value
    let max_abs = data.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);
    if max_abs == 0.0 {
        return; // all-zero tensor; nothing to do
    }

    let scale = max_abs / 127.0_f32;

    // Quantize → i8 → dequantize back to f32
    for w in data.iter_mut() {
        let q = (*w / scale).round().clamp(-127.0_f32, 127.0_f32) as i8;
        *w = q as f32 * scale;
    }
}

/// Apply the clip stage to a single tensor in-place.
///
/// Computes the `percentile`-th percentile of absolute values and zeros every
/// element whose absolute value is at or below that threshold.
fn apply_clip_inplace(tensor: &mut WeightTensor, percentile: f32) -> Result<(), CompressionError> {
    if percentile <= 0.0 || percentile > 1.0 {
        return Err(CompressionError::InvalidPercentile(percentile));
    }
    let data = &mut tensor.data;
    if data.is_empty() {
        return Ok(());
    }

    // Collect absolute values and sort ascending
    let mut abs_vals: Vec<f32> = data.iter().map(|w| w.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let n = abs_vals.len();
    // Index of the percentile threshold value (0-based)
    // e.g. percentile=0.1 on 10 elements → idx = ceil(0.1*10) - 1 = 0
    let idx = ((percentile * n as f32).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    let threshold = abs_vals[idx];

    // Zero out elements at or below the threshold
    for w in data.iter_mut() {
        if w.abs() <= threshold {
            *w = 0.0;
        }
    }

    Ok(())
}

/// Compute the theoretical INT8-*encoded* size for a quantize_int8 stage
/// (0.25 × the real `f32` byte count). This is **not** the tensor's real
/// post-stage memory footprint — the data stays `f32` in memory (see
/// [`StageStats::memory_after_bytes`] vs
/// [`StageStats::encoded_bytes_after`]).
#[inline]
fn quantize_int8_encoded_bytes(memory_before: usize) -> usize {
    // INT8 is 1 byte vs 4 bytes for f32 → theoretical 4× compression
    (memory_before as f32 * 0.25).round() as usize
}

// ──────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────

/// Run the compression pipeline on `tensors` according to `config`.
///
/// Returns a [`CompressionResult`] containing the compressed tensors and
/// per-stage statistics.
///
/// # Errors
///
/// - [`CompressionError::EmptyModel`] if `tensors` is empty.
/// - [`CompressionError::EmptyPipeline`] if `config.stages` is empty.
/// - [`CompressionError::Pruning`] if a pruning stage fails.
/// - [`CompressionError::InvalidPercentile`] if a clip stage has an invalid percentile.
pub fn compress_model(
    tensors: &[WeightTensor],
    config: &CompressionConfig,
) -> Result<CompressionResult, CompressionError> {
    if tensors.is_empty() {
        return Err(CompressionError::EmptyModel);
    }
    if config.stages.is_empty() {
        return Err(CompressionError::EmptyPipeline);
    }

    // Validate clip percentiles before doing any work
    for stage in &config.stages {
        if let CompressionStage::Clip { percentile } = stage {
            if *percentile <= 0.0 || *percentile > 1.0 {
                return Err(CompressionError::InvalidPercentile(*percentile));
            }
        }
    }

    // Working copy — we mutate in-place across stages
    let mut working: Vec<WeightTensor> = tensors.to_vec();
    let mut stage_stats: Vec<StageStats> = Vec::with_capacity(config.stages.len());

    for stage in &config.stages {
        let stage_name = stage.name().to_string();

        let mut tensors_processed = 0usize;
        let mut tensors_skipped = 0usize;
        let mut params_before = 0usize;
        let mut nonzero_after = 0usize;
        let mut memory_before = 0usize;
        let mut memory_after = 0usize;
        let mut encoded_after = 0usize;

        for tensor in working.iter_mut() {
            let should_skip = config.skip_embedding_layers && is_embedding_layer(&tensor.name);

            let tb = tensor_bytes(tensor);
            params_before += tensor.data.len();
            memory_before += tb;

            if should_skip {
                tensors_skipped += 1;
                // Skipped tensors are carried through unchanged: real and
                // encoded size are both just their unchanged f32 footprint.
                nonzero_after += count_nonzero(tensor);
                memory_after += tb;
                encoded_after += tb;
                continue;
            }

            tensors_processed += 1;

            match stage {
                CompressionStage::Prune(prune_cfg) => {
                    let (pruned, _mask) = prune_tensor(tensor, prune_cfg)?;
                    *tensor = pruned;
                    nonzero_after += count_nonzero(tensor);
                    // Pruning zeros elements but keeps dense f32 storage:
                    // real and encoded size are the same here.
                    let after_bytes = tensor_bytes(tensor);
                    memory_after += after_bytes;
                    encoded_after += after_bytes;
                }
                CompressionStage::QuantizeInt8 => {
                    apply_quantize_int8_inplace(tensor);
                    nonzero_after += count_nonzero(tensor);
                    // Real: the tensor is still f32 in memory (dequantized
                    // back after the simulated precision loss).
                    memory_after += tensor_bytes(tensor);
                    // Encoded: theoretical INT8-packed size, 0.25x original.
                    encoded_after += quantize_int8_encoded_bytes(tb);
                }
                CompressionStage::Clip { percentile } => {
                    // percentile already validated above
                    apply_clip_inplace(tensor, *percentile)?;
                    nonzero_after += count_nonzero(tensor);
                    // Clipping zeros elements but keeps dense f32 storage.
                    let after_bytes = tensor_bytes(tensor);
                    memory_after += after_bytes;
                    encoded_after += after_bytes;
                }
            }
        }

        stage_stats.push(StageStats {
            stage_name,
            tensors_processed,
            tensors_skipped,
            params_before,
            nonzero_params_after: nonzero_after,
            memory_before_bytes: memory_before,
            memory_after_bytes: memory_after,
            encoded_bytes_after: encoded_after,
        });
    }

    Ok(CompressionResult {
        compressed_tensors: working,
        stage_stats,
    })
}

/// Estimate the compressed size in bytes without actually compressing.
///
/// For `Prune` stages the estimate assumes a fraction of weights will be
/// zeroed (sparsity), but f32 storage is retained (same byte count).
/// For `QuantizeInt8` stages memory is reduced by 4x (theoretical INT8).
/// For `Clip` stages the estimate is the same as `Prune` (storage unchanged).
///
/// Returns `0` if `tensors` is empty or `config.stages` is empty.
pub fn estimate_compressed_size(tensors: &[WeightTensor], config: &CompressionConfig) -> usize {
    if tensors.is_empty() || config.stages.is_empty() {
        return 0;
    }

    // Total f32 bytes of all tensors
    let total_f32_bytes: usize = tensors.iter().map(tensor_bytes).sum();

    // Compute how many bytes are attributable to embedding layers (which are skipped)
    let embedding_bytes: usize = if config.skip_embedding_layers {
        tensors
            .iter()
            .filter(|t| is_embedding_layer(&t.name))
            .map(tensor_bytes)
            .sum()
    } else {
        0
    };
    let compressible_bytes = total_f32_bytes.saturating_sub(embedding_bytes);

    // Apply each stage's size multiplier to the compressible portion
    let mut size = compressible_bytes as f64;
    for stage in &config.stages {
        match stage {
            CompressionStage::Prune(_) => {
                // Pruning zeros weights but doesn't change storage in dense format
                // No change to byte count for dense storage
            }
            CompressionStage::QuantizeInt8 => {
                // Theoretical 4x compression
                size *= 0.25;
            }
            CompressionStage::Clip { .. } => {
                // Same as pruning — storage unchanged
            }
        }
    }

    // Add back embedding bytes unchanged
    embedding_bytes + size.round() as usize
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

    fn linear_data(n: usize) -> Vec<f32> {
        (1..=n).map(|i| i as f32).collect()
    }

    #[test]
    fn is_embedding_layer_matches_embed_prefix() {
        assert!(is_embedding_layer("embed.weight"));
        assert!(is_embedding_layer("Embed.weight"));
        assert!(is_embedding_layer("embedding_layer"));
        assert!(is_embedding_layer("token_embedding"));
        assert!(!is_embedding_layer("linear.weight"));
        assert!(!is_embedding_layer("layer_norm"));
    }

    #[test]
    fn apply_quantize_int8_preserves_sign() {
        let mut t = make_tensor("w", vec![1.0, -2.0, 0.5, -0.25], vec![4]);
        apply_quantize_int8_inplace(&mut t);
        assert!(t.data[0] > 0.0);
        assert!(t.data[1] < 0.0);
        assert!(t.data[2] > 0.0);
        assert!(t.data[3] < 0.0);
    }

    #[test]
    fn apply_clip_zeros_small_values() {
        let mut t = make_tensor("w", linear_data(10), vec![10]);
        // Clip bottom 30%: values 1, 2, 3 should be zeroed
        apply_clip_inplace(&mut t, 0.3).expect("clip ok");
        assert_eq!(t.data[0], 0.0);
        assert_eq!(t.data[1], 0.0);
        assert_eq!(t.data[2], 0.0);
        assert!(t.data[9] != 0.0);
    }

    #[test]
    fn apply_clip_invalid_percentile_returns_error() {
        let mut t = make_tensor("w", vec![1.0; 4], vec![4]);
        assert!(apply_clip_inplace(&mut t, 0.0).is_err());
        assert!(apply_clip_inplace(&mut t, 1.1).is_err());
        assert!(apply_clip_inplace(&mut t, -0.5).is_err());
        assert!(apply_clip_inplace(&mut t, 1.0).is_ok()); // 1.0 is valid
    }

    #[test]
    fn stage_stats_compression_ratio_equals_before_over_encoded_after() {
        let stats = StageStats {
            stage_name: "prune".to_string(),
            tensors_processed: 1,
            tensors_skipped: 0,
            params_before: 100,
            nonzero_params_after: 50,
            memory_before_bytes: 400,
            memory_after_bytes: 400,
            encoded_bytes_after: 400,
        };
        let ratio = stats.compression_ratio();
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    /// CQ-15 regression: `memory_after_bytes` must be the *real* post-stage
    /// footprint, not the fabricated `before * 0.25`. A `QuantizeInt8`
    /// stage keeps the data as `f32` in memory, so `memory_after_bytes`
    /// must equal `memory_before_bytes` even though `encoded_bytes_after`
    /// (the theoretical INT8 size) is smaller.
    #[test]
    fn stage_stats_quantize_int8_memory_after_is_real_not_fabricated() {
        let stats = StageStats {
            stage_name: "quantize_int8".to_string(),
            tensors_processed: 1,
            tensors_skipped: 0,
            params_before: 10,
            nonzero_params_after: 10,
            memory_before_bytes: 400,
            memory_after_bytes: 400,
            encoded_bytes_after: 100,
        };
        assert_eq!(
            stats.memory_after_bytes, stats.memory_before_bytes,
            "memory_after_bytes must report the real (unchanged) f32 footprint"
        );
        assert!(
            (stats.compression_ratio() - 4.0).abs() < 1e-4,
            "compression_ratio must still reflect the 4x theoretical INT8 saving \
             via encoded_bytes_after, got {}",
            stats.compression_ratio()
        );
    }

    #[test]
    fn stage_stats_sparsity_half() {
        let stats = StageStats {
            stage_name: "prune".to_string(),
            tensors_processed: 2,
            tensors_skipped: 0,
            params_before: 100,
            nonzero_params_after: 50,
            memory_before_bytes: 400,
            memory_after_bytes: 400,
            encoded_bytes_after: 400,
        };
        assert!((stats.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compression_result_memory_helpers() {
        let result = CompressionResult {
            compressed_tensors: vec![],
            stage_stats: vec![
                StageStats {
                    stage_name: "prune".to_string(),
                    tensors_processed: 1,
                    tensors_skipped: 0,
                    params_before: 10,
                    nonzero_params_after: 5,
                    memory_before_bytes: 40,
                    memory_after_bytes: 40,
                    encoded_bytes_after: 40,
                },
                StageStats {
                    stage_name: "quantize_int8".to_string(),
                    tensors_processed: 1,
                    tensors_skipped: 0,
                    params_before: 10,
                    nonzero_params_after: 5,
                    // Real memory stays f32 (unchanged from before) —
                    // only the encoded/target size shrinks.
                    memory_before_bytes: 40,
                    memory_after_bytes: 40,
                    encoded_bytes_after: 10,
                },
            ],
        };
        assert_eq!(result.memory_before_bytes(), 40);
        assert_eq!(
            result.memory_after_bytes(),
            40,
            "real memory footprint is unchanged by this simulation"
        );
        assert_eq!(result.encoded_bytes_after(), 10);
        assert!((result.total_compression_ratio() - 4.0).abs() < 1e-4);
    }

    #[test]
    fn compress_model_returns_same_tensor_count() {
        let tensors = vec![
            make_tensor("layer1.weight", linear_data(8), vec![2, 4]),
            make_tensor("layer2.weight", linear_data(4), vec![2, 2]),
        ];
        let config = CompressionConfig::quantize_only();
        let result = compress_model(&tensors, &config).expect("compress ok");
        assert_eq!(result.compressed_tensors.len(), 2);
    }

    #[test]
    fn compress_model_prune_reduces_nonzero() {
        let tensors = vec![make_tensor("layer.weight", linear_data(10), vec![10])];
        let config = CompressionConfig::prune_only(0.5);
        let result = compress_model(&tensors, &config).expect("compress ok");
        assert!(result.total_nonzero() < 10);
    }
}
