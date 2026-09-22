//! YaRN (Yet Another RoPE extensioN) — extended context via frequency interpolation.
//!
//! References: Peng et al. 2023 — "YaRN: Efficient Context Window Extension of Large Language Models"
//!
//! Key ideas:
//! 1. RoPE frequencies are split into three zones: high (unmodified), medium (linear interp), low (NTK)
//! 2. An attention scaling factor `sqrt(1/log(s))` compensates for distribution shift
//! 3. The effective context is extended by scale factor `s = target_len / training_len`
//!
//! # Relationship to `RopeTable::new_with_scaling` (M-08)
//!
//! [`RopeTable::new_with_scaling`](crate::layers::rope::RopeTable::new_with_scaling)
//! — the path GGUF `<arch>.rope.scaling.type = "yarn"` metadata is routed
//! through — uses [`crate::layers::rope_scaling::RopeScalingStrategy::Yarn`]
//! instead of the types in this module. That variant implements the
//! `ggml`/HF `transformers`-compatible NTK-by-parts formula (blend between
//! the unscaled and `freq / factor` frequency over a correction-dimension
//! ramp derived from `beta_fast`/`beta_slow`, with the standard
//! `0.1 * ln(factor) + 1.0` mscale), which real GGUF files' `beta_fast` /
//! `beta_slow` / `attn_factor` keys are defined against. This module's
//! [`YarnConfig::scaled_frequencies`] implements a related but distinct
//! blend (interpolating toward an NTK-*rescaled base*, per the original
//! bloc97/bloc97-style formulation, and a plain `sqrt(1/ln(s))` attention
//! scale rather than `ggml`'s `0.1*ln(s)+1`), so the two are **not**
//! numerically interchangeable — do not swap one in for the other without
//! re-deriving which formula a given caller actually needs to match. Both
//! remain independently useful: this module offers on-the-fly
//! rotate-without-precomputing-a-table application
//! ([`apply_yarn_rope`]/[`YarnFreqTable::apply`]/[`apply_batch`](YarnFreqTable::apply_batch)),
//! while `RopeTable` precomputes a full cos/sin table up front.

/// YaRN configuration.
#[derive(Debug, Clone)]
pub struct YarnConfig {
    /// Original training context length (e.g. 4096 for Qwen3-8B).
    pub original_max_position: usize,
    /// Target extended context length (e.g. 32768).
    pub extended_max_position: usize,
    /// Base frequency (default 10000.0 for most models, 1000000.0 for Qwen3).
    pub rope_base: f32,
    /// Head dimension.
    pub head_dim: usize,
    /// NTK alpha factor for low-frequency dimensions (default 1.0).
    pub alpha: f32,
    /// Beta factor for high-frequency dimensions (default 32.0).
    pub beta: f32,
}

impl YarnConfig {
    /// Create a new YarnConfig with default alpha=1.0 and beta=32.0.
    pub fn new(
        original_max_position: usize,
        extended_max_position: usize,
        rope_base: f32,
        head_dim: usize,
    ) -> Self {
        Self {
            original_max_position,
            extended_max_position,
            rope_base,
            head_dim,
            alpha: 1.0,
            beta: 32.0,
        }
    }

    /// Scale factor: s = extended / original.
    pub fn scale(&self) -> f32 {
        self.extended_max_position as f32 / self.original_max_position as f32
    }

    /// Attention temperature scaling factor: sqrt(1 / log(s)).
    ///
    /// Compensates for the distribution shift introduced by context extension.
    /// For s > e (~2.718), this value is < 1.0.
    pub fn attention_scale(&self) -> f32 {
        let s = self.scale();
        // Guard against s <= 1 (no extension or negative extension)
        let log_s = s.max(1.0 + f32::EPSILON).ln();
        (1.0_f32 / log_s).sqrt()
    }

    /// Compute per-dimension interpolation factors (one per frequency pair).
    ///
    /// Returns a Vec of length `head_dim/2`, each in [0.0, 1.0]:
    ///   - 0.0 = NTK scaling (low frequency dimensions)
    ///   - 1.0 = no scaling (high frequency dimensions)
    ///   - intermediate = linear blend between the two
    ///
    /// The three zones are defined by alpha and beta thresholds on the
    /// wavelength ratio compared to the original context length.
    pub fn interpolation_factors(&self) -> Vec<f32> {
        let half_dim = self.head_dim / 2;
        let orig = self.original_max_position as f32;

        (0..half_dim)
            .map(|i| {
                // Standard RoPE frequency for dimension i
                let freq = 1.0_f32 / self.rope_base.powf(2.0 * i as f32 / self.head_dim as f32);
                // Wavelength: how many tokens for one full rotation
                let wavelength = if freq > 0.0 {
                    (2.0 * std::f32::consts::PI) / freq
                } else {
                    f32::MAX
                };

                // alpha = threshold below which we use NTK (low-freq zone)
                // beta  = threshold above which we do not interpolate (high-freq zone)
                let low_threshold = orig / self.beta;
                let high_threshold = orig / self.alpha;

                if wavelength < low_threshold {
                    // High-frequency: no modification needed
                    1.0_f32
                } else if wavelength > high_threshold {
                    // Low-frequency: NTK scaling only
                    0.0_f32
                } else {
                    // Medium: linear blend
                    // Ramp from 0 at high_threshold to 1 at low_threshold
                    let range = high_threshold - low_threshold;
                    if range <= 0.0 {
                        0.5_f32
                    } else {
                        1.0 - (wavelength - low_threshold) / range
                    }
                }
            })
            .collect()
    }

    /// Compute YaRN-scaled frequencies for all `head_dim/2` frequency pairs.
    ///
    /// Each frequency is a blend of:
    ///   - The original unscaled frequency (high-freq dims)
    ///   - An NTK-scaled frequency (low-freq dims, scaled by `s^(head_dim/(head_dim-2))`)
    ///   - A linear interpolation between the two (medium-freq dims)
    pub fn scaled_frequencies(&self) -> Vec<f32> {
        let half_dim = self.head_dim / 2;
        let factors = self.interpolation_factors();
        let s = self.scale();

        // NTK scaling exponent: alpha_ntk = head_dim / (head_dim - 2)
        let ntk_exp = if self.head_dim > 2 {
            self.head_dim as f32 / (self.head_dim as f32 - 2.0)
        } else {
            1.0
        };
        let ntk_base = self.rope_base * s.powf(ntk_exp);

        (0..half_dim)
            .map(|i| {
                let dim_ratio = 2.0 * i as f32 / self.head_dim as f32;
                // Original frequency
                let freq_orig = 1.0_f32 / self.rope_base.powf(dim_ratio);
                // NTK-scaled frequency
                let freq_ntk = 1.0_f32 / ntk_base.powf(dim_ratio);
                // Blend: factor=1 → use original (high-freq), factor=0 → use NTK (low-freq)
                let t = factors[i];
                t * freq_orig + (1.0 - t) * freq_ntk
            })
            .collect()
    }
}

// ─── apply_rope ───────────────────────────────────────────────────────────────

/// Apply standard RoPE to a query/key vector at position `pos`.
///
/// `freqs`: precomputed frequencies (head_dim/2 values).
/// Rotates pairs `(q[i], q[i + half])` in-place.
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, freqs: &[f32]) {
    let half = freqs.len();
    debug_assert_eq!(q.len(), half * 2);
    debug_assert_eq!(k.len(), half * 2);

    for i in 0..half {
        let angle = pos as f32 * freqs[i];
        let (sin_a, cos_a) = angle.sin_cos();

        // Query rotation
        let q0 = q[i];
        let q1 = q[half + i];
        q[i] = q0 * cos_a - q1 * sin_a;
        q[half + i] = q0 * sin_a + q1 * cos_a;

        // Key rotation
        let k0 = k[i];
        let k1 = k[half + i];
        k[i] = k0 * cos_a - k1 * sin_a;
        k[half + i] = k0 * sin_a + k1 * cos_a;
    }
}

// ─── apply_yarn_rope ─────────────────────────────────────────────────────────

/// Apply YaRN-scaled RoPE in-place to query and key vectors at position `pos`.
pub fn apply_yarn_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    config: &YarnConfig,
) -> Result<(), YarnError> {
    let head_dim = config.head_dim;

    if !head_dim.is_multiple_of(2) {
        return Err(YarnError::OddHeadDim(head_dim));
    }
    if q.len() != head_dim {
        return Err(YarnError::DimMismatch {
            expected: head_dim,
            got: q.len(),
        });
    }
    if k.len() != head_dim {
        return Err(YarnError::DimMismatch {
            expected: head_dim,
            got: k.len(),
        });
    }
    if pos >= config.extended_max_position {
        return Err(YarnError::PositionExceedsContext {
            pos,
            max_pos: config.extended_max_position,
        });
    }

    let freqs = config.scaled_frequencies();
    apply_rope(q, k, pos, &freqs);
    Ok(())
}

// ─── YarnFreqTable ────────────────────────────────────────────────────────────

/// Precomputed YaRN frequency table for efficient batch application.
pub struct YarnFreqTable {
    config: YarnConfig,
    /// Precomputed scaled frequencies (head_dim/2 values).
    scaled_freqs: Vec<f32>,
    /// Attention temperature scaling factor.
    pub attention_scale: f32,
}

impl YarnFreqTable {
    /// Build the frequency table from the given config.
    pub fn new(config: YarnConfig) -> Self {
        let scaled_freqs = config.scaled_frequencies();
        let attention_scale = config.attention_scale();
        Self {
            config,
            scaled_freqs,
            attention_scale,
        }
    }

    /// Apply YaRN-scaled RoPE to a single (query, key) pair at `pos`.
    pub fn apply(&self, q: &mut [f32], k: &mut [f32], pos: usize) -> Result<(), YarnError> {
        let head_dim = self.config.head_dim;
        if !head_dim.is_multiple_of(2) {
            return Err(YarnError::OddHeadDim(head_dim));
        }
        if q.len() != head_dim {
            return Err(YarnError::DimMismatch {
                expected: head_dim,
                got: q.len(),
            });
        }
        if k.len() != head_dim {
            return Err(YarnError::DimMismatch {
                expected: head_dim,
                got: k.len(),
            });
        }
        if pos >= self.config.extended_max_position {
            return Err(YarnError::PositionExceedsContext {
                pos,
                max_pos: self.config.extended_max_position,
            });
        }

        apply_rope(q, k, pos, &self.scaled_freqs);
        Ok(())
    }

    /// Apply YaRN-scaled RoPE to a batch of query/key vectors.
    ///
    /// `queries` and `keys` are laid out as `[num_tokens * head_dim]`.
    /// `positions` gives the absolute position for each token.
    pub fn apply_batch(
        &self,
        queries: &mut [f32],
        keys: &mut [f32],
        positions: &[usize],
        head_dim: usize,
    ) -> Result<(), YarnError> {
        if !head_dim.is_multiple_of(2) {
            return Err(YarnError::OddHeadDim(head_dim));
        }
        let num_tokens = positions.len();
        if queries.len() != num_tokens * head_dim {
            return Err(YarnError::DimMismatch {
                expected: num_tokens * head_dim,
                got: queries.len(),
            });
        }
        if keys.len() != num_tokens * head_dim {
            return Err(YarnError::DimMismatch {
                expected: num_tokens * head_dim,
                got: keys.len(),
            });
        }

        for (tok_idx, &pos) in positions.iter().enumerate() {
            if pos >= self.config.extended_max_position {
                return Err(YarnError::PositionExceedsContext {
                    pos,
                    max_pos: self.config.extended_max_position,
                });
            }
            let start = tok_idx * head_dim;
            let end = start + head_dim;
            let q_slice = &mut queries[start..end];
            let k_slice = &mut keys[start..end];
            apply_rope(q_slice, k_slice, pos, &self.scaled_freqs);
        }

        Ok(())
    }

    /// Number of frequency pairs (= head_dim / 2).
    #[inline]
    pub fn num_frequencies(&self) -> usize {
        self.scaled_freqs.len()
    }

    /// Effective (extended) context length.
    #[inline]
    pub fn effective_context(&self) -> usize {
        self.config.extended_max_position
    }
}

// ─── LongRopeConfig ──────────────────────────────────────────────────────────

/// LongRoPE-style position remapping (simple linear remapping).
///
/// Maps positions in the extended context back into the range of the
/// original training context length, enabling the model to generalise
/// beyond its training window.
pub struct LongRopeConfig {
    /// Original training context length.
    pub original_max_pos: usize,
    /// Extended (target) context length.
    pub extended_max_pos: usize,
}

impl LongRopeConfig {
    /// Create a new LongRoPE config.
    pub fn new(original: usize, extended: usize) -> Self {
        Self {
            original_max_pos: original,
            extended_max_pos: extended,
        }
    }

    /// Map `pos` ∈ [0, extended_max_pos] → [0, original_max_pos] linearly.
    ///
    /// The formula is:
    /// ```text
    /// remapped = pos * (original_max_pos / extended_max_pos)
    /// ```
    pub fn remap_position(&self, pos: usize) -> f32 {
        if self.extended_max_pos == 0 {
            return 0.0;
        }
        pos as f32 * (self.original_max_pos as f32 / self.extended_max_pos as f32)
    }

    /// Integer-rounded remapped position, clamped to `[0, original_max_pos - 1]`.
    pub fn effective_pos(&self, pos: usize) -> usize {
        let remapped = self.remap_position(pos).round() as usize;
        remapped.min(self.original_max_pos.saturating_sub(1))
    }
}

// ─── YarnError ───────────────────────────────────────────────────────────────

/// Errors from YaRN operations.
#[derive(Debug, thiserror::Error)]
pub enum YarnError {
    #[error("head_dim must be even, got {0}")]
    OddHeadDim(usize),

    #[error("query/key length {got} doesn't match head_dim {expected}")]
    DimMismatch { expected: usize, got: usize },

    #[error("position {pos} exceeds extended context {max_pos}")]
    PositionExceedsContext { pos: usize, max_pos: usize },
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> YarnConfig {
        YarnConfig::new(4096, 32768, 10000.0, 64)
    }

    #[test]
    fn yarn_config_scale() {
        let cfg = make_config();
        let s = cfg.scale();
        assert!(
            (s - 8.0).abs() < 1e-5,
            "scale should be 32768/4096 = 8, got {s}"
        );
    }

    #[test]
    fn yarn_config_attention_scale() {
        let cfg = make_config();
        let a = cfg.attention_scale();
        // s=8 > e, so attention_scale < 1
        assert!(a < 1.0, "attention_scale should be < 1.0 for s=8, got {a}");
        // sqrt(1 / ln(8)) = sqrt(1 / 2.079) ≈ 0.693
        let expected = (1.0_f32 / 8.0_f32.ln()).sqrt();
        assert!(
            (a - expected).abs() < 1e-5,
            "attention_scale = {a}, expected {expected}"
        );
    }

    #[test]
    fn yarn_config_interpolation_factors_bounds() {
        let cfg = make_config();
        let factors = cfg.interpolation_factors();
        for (i, &f) in factors.iter().enumerate() {
            assert!((0.0..=1.0).contains(&f), "factor[{i}] = {f} out of [0, 1]");
        }
    }

    #[test]
    fn yarn_config_interpolation_factors_length() {
        let cfg = make_config();
        let factors = cfg.interpolation_factors();
        assert_eq!(factors.len(), cfg.head_dim / 2);
    }

    #[test]
    fn yarn_scaled_frequencies_positive() {
        let cfg = make_config();
        let freqs = cfg.scaled_frequencies();
        for (i, &f) in freqs.iter().enumerate() {
            assert!(f > 0.0, "freq[{i}] = {f} is not positive");
        }
    }

    #[test]
    fn yarn_scaled_frequencies_monotone_decreasing() {
        let cfg = make_config();
        let freqs = cfg.scaled_frequencies();
        // Higher dimensional indices should have lower frequencies
        // (may not be strictly monotone everywhere due to blending, so check overall trend)
        let first = freqs[0];
        let last = *freqs.last().expect("non-empty");
        assert!(
            first >= last,
            "frequencies should be non-increasing overall, first={first}, last={last}"
        );
    }

    #[test]
    fn apply_rope_identity_zero_pos() {
        let freqs = vec![0.01f32, 0.001f32];
        let mut q = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut k = vec![5.0f32, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();
        apply_rope(&mut q, &mut k, 0, &freqs);
        // At pos=0, angle=0: cos=1, sin=0 → identity
        for i in 0..q.len() {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-5,
                "q[{i}] changed: {} → {}",
                q_orig[i],
                q[i]
            );
            assert!(
                (k[i] - k_orig[i]).abs() < 1e-5,
                "k[{i}] changed: {} → {}",
                k_orig[i],
                k[i]
            );
        }
    }

    #[test]
    fn apply_rope_changes_values() {
        let freqs = vec![0.5f32, 0.1f32];
        let mut q = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut k = vec![1.0f32, 1.0, 1.0, 1.0];
        let q_orig = q.clone();
        apply_rope(&mut q, &mut k, 5, &freqs);
        // At pos=5 with non-trivial freqs, values should differ
        let changed = q
            .iter()
            .zip(q_orig.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(changed, "apply_rope at pos>0 should change values");
    }

    #[test]
    fn apply_yarn_rope_basic() {
        let cfg = make_config();
        let mut q = vec![0.1f32; cfg.head_dim];
        let mut k = vec![0.2f32; cfg.head_dim];
        let result = apply_yarn_rope(&mut q, &mut k, 100, &cfg);
        assert!(result.is_ok(), "apply_yarn_rope failed: {result:?}");
    }

    #[test]
    fn yarn_freq_table_new() {
        let cfg = make_config();
        let table = YarnFreqTable::new(cfg);
        assert!(table.num_frequencies() > 0);
    }

    #[test]
    fn yarn_freq_table_apply_basic() {
        let cfg = make_config();
        let head_dim = cfg.head_dim;
        let table = YarnFreqTable::new(cfg);

        // At pos=0, rotation is identity
        let mut q = vec![1.0f32; head_dim];
        let mut k = vec![2.0f32; head_dim];
        let q_orig = q.clone();
        let k_orig = k.clone();
        table
            .apply(&mut q, &mut k, 0)
            .expect("apply at pos=0 failed");
        for i in 0..head_dim {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-5,
                "q[{i}] should be unchanged at pos=0"
            );
            assert!(
                (k[i] - k_orig[i]).abs() < 1e-5,
                "k[{i}] should be unchanged at pos=0"
            );
        }
    }

    #[test]
    fn yarn_freq_table_num_frequencies() {
        let cfg = make_config();
        let head_dim = cfg.head_dim;
        let table = YarnFreqTable::new(cfg);
        assert_eq!(table.num_frequencies(), head_dim / 2);
    }

    #[test]
    fn yarn_freq_table_effective_context() {
        let cfg = make_config();
        let extended = cfg.extended_max_position;
        let table = YarnFreqTable::new(cfg);
        assert_eq!(table.effective_context(), extended);
    }

    #[test]
    fn yarn_freq_table_apply_batch() {
        let cfg = make_config();
        let head_dim = cfg.head_dim;
        let table = YarnFreqTable::new(cfg);

        let num_tokens = 4;
        let mut queries = vec![0.1f32; num_tokens * head_dim];
        let mut keys = vec![0.2f32; num_tokens * head_dim];
        let positions = vec![0usize, 10, 100, 1000];

        let result = table.apply_batch(&mut queries, &mut keys, &positions, head_dim);
        assert!(result.is_ok(), "apply_batch failed: {result:?}");
    }

    #[test]
    fn longrope_remap_start() {
        let cfg = LongRopeConfig::new(4096, 32768);
        let remapped = cfg.remap_position(0);
        assert!(
            remapped.abs() < 1e-5,
            "pos=0 should remap to 0.0, got {remapped}"
        );
    }

    #[test]
    fn longrope_remap_end() {
        let cfg = LongRopeConfig::new(4096, 32768);
        // pos = extended_max_pos → should map to original_max_pos
        let remapped = cfg.remap_position(32768);
        assert!(
            (remapped - 4096.0).abs() < 1.0,
            "pos=extended should remap to ~original_max_pos, got {remapped}"
        );
    }

    #[test]
    fn longrope_effective_pos_bounded() {
        let cfg = LongRopeConfig::new(4096, 32768);
        // All effective positions should be < original_max_pos
        for pos in [0usize, 1000, 10000, 32768, 40000] {
            let ep = cfg.effective_pos(pos);
            assert!(
                ep < cfg.original_max_pos,
                "effective_pos({pos}) = {ep} should be < {}",
                cfg.original_max_pos
            );
        }
    }
}
