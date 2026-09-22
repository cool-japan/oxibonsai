//! Rotary Position Embeddings (RoPE).
//!
//! Applies rotation to query and key vectors to encode positional
//! information. Uses precomputed cos/sin tables.

use crate::error::{ModelError, ModelResult};
use crate::layers::rope_scaling::{
    compute_rope_frequencies, yarn_inv_freq_f64, yarn_mscale, RopeScalingError, RopeScalingStrategy,
};

/// Precomputed RoPE sin/cos table.
#[derive(Debug)]
pub struct RopeTable {
    /// Cosine values: [max_seq_len × head_dim/2].
    cos: Vec<f32>,
    /// Sine values: [max_seq_len × head_dim/2].
    sin: Vec<f32>,
    /// Half of head dimension (rotation pairs).
    half_dim: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Uniform multiplier baked into every `cos`/`sin` entry above (`1.0`
    /// for standard RoPE; YaRN's "mscale" when built via
    /// [`Self::new_with_scaling`] with [`RopeScalingStrategy::Yarn`] — M-08).
    attention_scale: f32,
}

impl RopeTable {
    /// Precompute RoPE rotation table.
    ///
    /// - `head_dim`: Dimension of each attention head.
    /// - `max_seq_len`: Maximum sequence length to precompute.
    /// - `freq_base`: RoPE frequency base (default: 1000000.0 for Qwen3).
    pub fn new(head_dim: usize, max_seq_len: usize, freq_base: f32) -> Self {
        Self::new_with_freqs(head_dim, max_seq_len, freq_base, &[])
    }

    /// Precompute RoPE rotation table with optional frequency scaling factors.
    ///
    /// Implements the `rope_freqs` / `freq_factors` pattern used by some models
    /// (e.g. Gemma 4) to implement partial / NoPE (No Position Embedding) RoPE.
    ///
    /// The effective inverse frequency for dimension `i` is:
    /// ```text
    /// inv_freq[i] = (1 / freq_base^(2*i/head_dim)) / freq_factors[i]
    /// ```
    ///
    /// When `freq_factors[i] = 1.0` the dimension behaves like standard RoPE.
    /// When `freq_factors[i]` is very large (e.g. `1e30`) the inverse frequency
    /// approaches zero, so the angle ≈ 0 for all positions → `cos ≈ 1, sin ≈ 0`
    /// (identity rotation = no positional encoding for that dimension).
    ///
    /// - `freq_factors`: Per-dimension scaling factors of length `≥ half_dim`.
    ///   Pass an empty slice to use standard RoPE (all factors = 1.0).
    pub fn new_with_freqs(
        head_dim: usize,
        max_seq_len: usize,
        freq_base: f32,
        freq_factors: &[f32],
    ) -> Self {
        let half_dim = head_dim / 2;
        let inv_freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let base_inv_freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
                // Apply freq_factor divisor if provided.
                if i < freq_factors.len() && freq_factors[i] > 1.0 {
                    base_inv_freq / freq_factors[i]
                } else {
                    base_inv_freq
                }
            })
            .collect();

        Self::from_frequencies(head_dim, max_seq_len, &inv_freqs, 1.0)
    }

    /// Precompute a RoPE table honoring an optional
    /// [`RopeScalingStrategy`] — linear / dynamic-NTK / LLaMA-3.1 / LongRoPE
    /// / YaRN (M-08).
    ///
    /// `strategy = None` (or `Some(RopeScalingStrategy::None)`) is identical
    /// to [`Self::new`]. For [`RopeScalingStrategy::Yarn`], the resulting
    /// cos/sin table additionally has the YaRN attention "mscale" multiplier
    /// folded in directly (matching `ggml`'s `rope_yarn`, which scales both
    /// `cos_theta` and `sin_theta`) — so [`Self::apply`] callers get correct
    /// behaviour with no further changes anywhere else in the forward pass.
    ///
    /// For [`RopeScalingStrategy::DynamicNtk`]: this table is a single array
    /// precomputed once for `max_seq_len` (it is not rebuilt as generation
    /// proceeds), so the "current sequence length" fed into the NTK base
    /// formula is `max_seq_len` itself — the table is built assuming
    /// generation may use its full precomputed capacity. A fully dynamic
    /// table whose frequencies for already-emitted positions change as the
    /// live sequence grows would require rebuilding this table from the
    /// forward-pass call site every step; that is a larger, call-site-level
    /// change (see the package's recorded deviations) and is out of scope
    /// for a table that is precomputed once at model-load time.
    ///
    /// # Errors
    ///
    /// Propagates [`RopeScalingError`] from `compute_rope_frequencies` (e.g.
    /// a `scale_factor < 1.0`, or a `LongRope` factor-length mismatch).
    ///
    /// # Precision (M-08 GATE)
    ///
    /// [`RopeScalingStrategy::Yarn`] is built through an `f64`-precision
    /// pipeline (`yarn_inv_freq_f64` → this table's cos/sin, rounding to
    /// `f32` only at the very end) specifically so it can match a reference
    /// table to `1e-6` at large positions (the package gate checks position
    /// 65535). The other four strategies (`Linear`, `DynamicNtk`, `Llama31`,
    /// `LongRope`) still go through the `f32`-returning
    /// `compute_rope_frequencies`, which rounds the frequency to `f32`
    /// *before* the position multiply; at a large position and a
    /// high-frequency dimension this can show angle error on the order of
    /// `1e-3` (position × frequency × the `f32` machine epsilon), same as
    /// plain [`Self::new`]/[`Self::new_with_freqs`] always have. That
    /// asymmetry is deliberate (only `Yarn` is gated at extreme positions
    /// today) but should be kept in mind if a future caller needs the same
    /// precision guarantee from another strategy.
    pub fn new_with_scaling(
        head_dim: usize,
        max_seq_len: usize,
        freq_base: f32,
        strategy: Option<&RopeScalingStrategy>,
    ) -> Result<Self, RopeScalingError> {
        let strategy = match strategy {
            None | Some(RopeScalingStrategy::None) => {
                return Ok(Self::new(head_dim, max_seq_len, freq_base));
            }
            Some(s) => s,
        };

        if let RopeScalingStrategy::Yarn {
            original_max_position,
            factor,
            beta_fast,
            beta_slow,
            attn_factor,
        } = strategy
        {
            // `compute_rope_frequencies` itself performs this exact check
            // (`factor < 1.0` → `InvalidScaleFactor`); it is duplicated here
            // because this branch never calls that function (the `f64`
            // pipeline below computes frequencies directly — see the
            // doc comment above). Kept in sync by
            // `new_with_scaling_yarn_sub_unity_factor_errors_like_compute_rope_frequencies`.
            if *factor < 1.0 {
                return Err(RopeScalingError::InvalidScaleFactor(*factor));
            }
            // Ditto for the `InvalidHeadDim` check every other strategy
            // gets via `compute_rope_frequencies` (verifier finding,
            // minor): this branch bypasses that function entirely, so
            // without this it would hand `head_dim == 0` or an odd
            // `head_dim` straight to `yarn_inv_freq_f64`, which does not
            // itself panic but silently produces a degenerate table
            // instead of reporting the same error every other scaling
            // strategy would. Deliberately placed *inside* the `Yarn`
            // branch rather than before the `strategy` match above: the
            // `None`/no-scaling early return calls `Self::new`, which has
            // never validated `head_dim` (odd values silently truncate via
            // integer division) and must keep not doing so — adding the
            // check earlier would newly reject that previously-accepted
            // input on an unrelated code path.
            if head_dim == 0 || !head_dim.is_multiple_of(2) {
                return Err(RopeScalingError::InvalidHeadDim(head_dim));
            }
            let freqs_f64 = yarn_inv_freq_f64(
                head_dim,
                freq_base as f64,
                *original_max_position,
                *factor as f64,
                *beta_fast as f64,
                *beta_slow as f64,
            );
            let attention_scale = yarn_mscale(*factor, *attn_factor) as f64;
            return Ok(Self::from_frequencies_f64(
                head_dim,
                max_seq_len,
                &freqs_f64,
                attention_scale,
            ));
        }

        let freqs = compute_rope_frequencies(head_dim, freq_base, strategy, max_seq_len)?;
        Ok(Self::from_frequencies(head_dim, max_seq_len, &freqs, 1.0))
    }

    /// Build a table from precomputed per-frequency-pair angular
    /// frequencies (`freqs.len()` should be `head_dim / 2`; a shorter slice
    /// zero-pads the remaining dimensions rather than panicking) and a
    /// uniform multiplier applied to every `cos`/`sin` entry (`1.0` for
    /// standard RoPE).
    fn from_frequencies(
        head_dim: usize,
        max_seq_len: usize,
        freqs: &[f32],
        attention_scale: f32,
    ) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = freqs.get(i).copied().unwrap_or(0.0);
                let angle = pos as f32 * freq;
                cos[pos * half_dim + i] = angle.cos() * attention_scale;
                sin[pos * half_dim + i] = angle.sin() * attention_scale;
            }
        }

        Self {
            cos,
            sin,
            half_dim,
            max_seq_len,
            attention_scale,
        }
    }

    /// `f64`-precision counterpart of [`Self::from_frequencies`], used only
    /// by the [`RopeScalingStrategy::Yarn`] branch of
    /// [`Self::new_with_scaling`] — see that method's "Precision" doc
    /// section for why. `freqs` and `attention_scale` are carried at `f64`
    /// through the `pos * freq` angle computation; only the final
    /// `cos`/`sin` values (always in `[-1, 1]`) are rounded to `f32`, which
    /// costs `~6e-8` absolute regardless of `pos`.
    fn from_frequencies_f64(
        head_dim: usize,
        max_seq_len: usize,
        freqs: &[f64],
        attention_scale: f64,
    ) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = freqs.get(i).copied().unwrap_or(0.0);
                let angle = pos as f64 * freq;
                cos[pos * half_dim + i] = (angle.cos() * attention_scale) as f32;
                sin[pos * half_dim + i] = (angle.sin() * attention_scale) as f32;
            }
        }

        Self {
            cos,
            sin,
            half_dim,
            max_seq_len,
            attention_scale: attention_scale as f32,
        }
    }

    /// Apply RoPE rotation to a query or key vector at the given position.
    ///
    /// - `vec`: Input vector of length `head_dim` (modified in-place via output).
    /// - `output`: Output vector of length `head_dim`.
    /// - `pos`: Token position in the sequence.
    ///
    /// Delegates the inner rotation to SIMD-accelerated `oxibonsai_kernels::rope_apply_simd`.
    ///
    /// # Errors
    ///
    /// - [`ModelError::PositionOutOfRange`] if `pos >= self.max_seq_len`
    ///   (M-27) — a real bounds check instead of a `debug_assert!` that would
    ///   otherwise let a safe-but-uncontrolled slice-index panic through in
    ///   release. It is deliberately not
    ///   [`ModelError::SequenceTooLong`](crate::error::ModelError::SequenceTooLong):
    ///   that is the recoverable "shorten the prompt" condition, whereas a
    ///   RoPE-table bound violation means the table and the KV cache were
    ///   built with inconsistent `max_seq_len` values.
    /// - [`crate::error::ModelError::Kernel`] if `vec`/`output` do not match
    ///   this table's `head_dim` (propagated from `rope_apply_simd`,
    ///   K-02/M-01).
    pub fn apply(&self, vec: &[f32], output: &mut [f32], pos: usize) -> ModelResult<()> {
        if pos >= self.max_seq_len {
            return Err(ModelError::PositionOutOfRange {
                pos,
                max: self.max_seq_len,
            });
        }

        let cos_row = &self.cos[pos * self.half_dim..(pos + 1) * self.half_dim];
        let sin_row = &self.sin[pos * self.half_dim..(pos + 1) * self.half_dim];

        oxibonsai_kernels::rope_apply_simd(vec, output, cos_row, sin_row)?;

        Ok(())
    }

    /// Maximum precomputed sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// The uniform multiplier baked into every `cos`/`sin` entry (`1.0`
    /// unless built via [`Self::new_with_scaling`] with a
    /// [`RopeScalingStrategy::Yarn`] strategy, in which case it is the YaRN
    /// attention "mscale" — M-08).
    pub fn attention_scale(&self) -> f32 {
        self.attention_scale
    }

    /// Checked version of [`Self::cos_at`], returning
    /// [`ModelError::PositionOutOfRange`] instead of panicking (M-27
    /// residue, wave-1.5 addendum).
    ///
    /// [`Self::cos_at`] itself is left panicking on purpose: switching its
    /// return type would require updating every one of its ~40 call sites
    /// (`model/types/forward_metal.rs`, `model/types/forward_cuda/*.rs`,
    /// `forward_cuda_fp8.rs`, `forward_metal_fp8.rs`,
    /// `block/types/helpers.rs`), none of which are in this package's
    /// `owned_files` — see the package's recorded deviations for the exact
    /// list. This checked variant exists so a future migration is possible
    /// without ever needing a breaking signature change to the unchecked
    /// pair.
    pub fn cos_at_checked(&self, pos: usize) -> ModelResult<&[f32]> {
        if pos >= self.max_seq_len {
            return Err(ModelError::PositionOutOfRange {
                pos,
                max: self.max_seq_len,
            });
        }
        Ok(&self.cos[pos * self.half_dim..(pos + 1) * self.half_dim])
    }

    /// Checked version of [`Self::sin_at`] — see [`Self::cos_at_checked`].
    pub fn sin_at_checked(&self, pos: usize) -> ModelResult<&[f32]> {
        if pos >= self.max_seq_len {
            return Err(ModelError::PositionOutOfRange {
                pos,
                max: self.max_seq_len,
            });
        }
        Ok(&self.sin[pos * self.half_dim..(pos + 1) * self.half_dim])
    }

    /// Get cos values for a given position: `&[half_dim]`.
    ///
    /// # Panics
    ///
    /// Panics via out-of-bounds slice indexing if `pos >= max_seq_len()`,
    /// unlike [`Self::apply`] (M-27), which returns
    /// `Err(ModelError::PositionOutOfRange)`-equivalent instead of panicking.
    /// Guarding this the same way by changing this method's signature is out
    /// of scope for this package: its ~40 call sites
    /// (`model/types/forward_metal.rs`, `forward_cuda*.rs`,
    /// `block/types/helpers.rs`) are all outside its `owned_files`. A
    /// non-breaking, fully-guarded alternative, [`Self::cos_at_checked`], is
    /// available now for any caller (existing or new) able to handle a
    /// `Result` — see the package's recorded deviations for the migration
    /// this method's own call sites still need.
    pub fn cos_at(&self, pos: usize) -> &[f32] {
        &self.cos[pos * self.half_dim..(pos + 1) * self.half_dim]
    }

    /// Get sin values for a given position: `&[half_dim]`.
    ///
    /// # Panics
    ///
    /// See [`Self::cos_at`] — same unguarded-`pos` caveat; guarded
    /// alternative: [`Self::sin_at_checked`].
    pub fn sin_at(&self, pos: usize) -> &[f32] {
        &self.sin[pos * self.half_dim..(pos + 1) * self.half_dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_position_zero_is_identity() {
        let table = RopeTable::new(4, 16, 10000.0);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        table
            .apply(&input, &mut output, 0)
            .expect("rope apply should succeed");

        // At position 0, cos=1, sin=0 → identity
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn rope_preserves_norm() {
        let table = RopeTable::new(4, 16, 10000.0);
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0; 4];

        table
            .apply(&input, &mut output, 5)
            .expect("rope apply should succeed");

        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (input_norm - output_norm).abs() < 1e-4,
            "RoPE should preserve vector norm"
        );
    }

    // ── M-27: real position bound check, not a debug_assert ─────────────

    #[test]
    fn apply_rejects_position_at_max_seq_len() {
        // `pos == max_seq_len` is the first out-of-range position (valid
        // positions are `0..max_seq_len`).
        let table = RopeTable::new(4, 8, 10000.0);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        let result = table.apply(&input, &mut output, 8);
        assert!(
            result.is_err(),
            "apply() must reject pos >= max_seq_len instead of panicking on an \
             out-of-bounds slice index (M-27)"
        );
    }

    #[test]
    fn apply_rejects_position_far_out_of_range() {
        let table = RopeTable::new(4, 8, 10000.0);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        let err = table
            .apply(&input, &mut output, 1_000_000)
            .expect_err("apply() must reject a far out-of-range position");
        assert!(
            matches!(
                err,
                ModelError::PositionOutOfRange {
                    pos: 1_000_000,
                    max: 8
                }
            ),
            "error should be the typed PositionOutOfRange variant carrying both \
             bounds (M-27), got: {err:?}"
        );
        assert_eq!(err.error_code(), "POSITION_OUT_OF_RANGE");
        assert_eq!(err.to_string(), "position 1000000 out of range: max 8");
    }

    #[test]
    fn apply_accepts_last_valid_position() {
        // `pos == max_seq_len - 1` is the last valid position and must still
        // succeed after the M-27 bounds check is added.
        let table = RopeTable::new(4, 8, 10000.0);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        table
            .apply(&input, &mut output, 7)
            .expect("the last valid position must still succeed");
    }

    // ── cos_at_checked / sin_at_checked (wave-1.5 addendum, M-27 residue) ────

    #[test]
    fn cos_at_checked_accepts_in_range_position() {
        let table = RopeTable::new(4, 8, 10000.0);
        assert!(table.cos_at_checked(0).is_ok());
        assert!(table.cos_at_checked(7).is_ok());
    }

    #[test]
    fn cos_at_checked_matches_unchecked_cos_at() {
        let table = RopeTable::new(4, 8, 10000.0);
        for pos in 0..8 {
            assert_eq!(
                table.cos_at_checked(pos).expect("in-range"),
                table.cos_at(pos)
            );
        }
    }

    #[test]
    fn cos_at_checked_rejects_out_of_range_position() {
        let table = RopeTable::new(4, 8, 10000.0);
        let err = table
            .cos_at_checked(8)
            .expect_err("pos == max_seq_len must be rejected");
        assert!(matches!(
            err,
            ModelError::PositionOutOfRange { pos: 8, max: 8 }
        ));
    }

    #[test]
    fn sin_at_checked_matches_unchecked_sin_at_and_rejects_out_of_range() {
        let table = RopeTable::new(4, 8, 10000.0);
        for pos in 0..8 {
            assert_eq!(
                table.sin_at_checked(pos).expect("in-range"),
                table.sin_at(pos)
            );
        }
        assert!(table.sin_at_checked(1_000_000).is_err());
    }

    // ── M-08: RopeTable::new_with_scaling ────────────────────────────────────

    fn bonsai_8b_yarn_strategy() -> RopeScalingStrategy {
        // The exact values `models/Bonsai-8B.gguf` ships (M-08): a shipped
        // model whose long-context output was silently wrong because YaRN
        // scaling was parsed nowhere and applied nowhere.
        RopeScalingStrategy::Yarn {
            original_max_position: 16384,
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: None,
        }
    }

    #[test]
    fn new_with_scaling_none_matches_plain_new() {
        let scaled = RopeTable::new_with_scaling(8, 32, 10_000.0, None)
            .expect("None strategy should succeed");
        let plain = RopeTable::new(8, 32, 10_000.0);
        for pos in [0usize, 1, 16, 31] {
            assert_eq!(scaled.cos_at(pos), plain.cos_at(pos));
            assert_eq!(scaled.sin_at(pos), plain.sin_at(pos));
        }
        assert!((scaled.attention_scale() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn new_with_scaling_explicit_none_variant_matches_plain_new() {
        let scaled = RopeTable::new_with_scaling(8, 32, 10_000.0, Some(&RopeScalingStrategy::None))
            .expect("None strategy should succeed");
        let plain = RopeTable::new(8, 32, 10_000.0);
        assert_eq!(scaled.cos_at(10), plain.cos_at(10));
    }

    #[test]
    fn new_with_scaling_linear_halves_effective_position() {
        // Linear scale_factor=2.0 divides every frequency by 2, so the table
        // at position `2*p` under scaling should match the unscaled table at
        // position `p` (same angle: `2p * (f/2) == p * f`).
        let scaled = RopeTable::new_with_scaling(
            8,
            64,
            10_000.0,
            Some(&RopeScalingStrategy::Linear { scale_factor: 2.0 }),
        )
        .expect("Linear strategy should succeed");
        let plain = RopeTable::new(8, 64, 10_000.0);

        for p in [1usize, 5, 10] {
            let scaled_cos = scaled.cos_at(2 * p);
            let plain_cos = plain.cos_at(p);
            for (a, b) in scaled_cos.iter().zip(plain_cos.iter()) {
                assert!((a - b).abs() < 1e-4, "mismatch at p={p}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn new_with_scaling_dynamic_ntk_wires_through() {
        let table = RopeTable::new_with_scaling(
            8,
            64,
            10_000.0,
            Some(&RopeScalingStrategy::DynamicNtk {
                original_max_position: 16,
                base: 10_000.0,
            }),
        )
        .expect("DynamicNtk strategy should succeed");
        // Sanity: table builds to the requested shape and is usable.
        assert_eq!(table.max_seq_len(), 64);
        assert_eq!(table.cos_at(0).len(), 4);
    }

    #[test]
    fn new_with_scaling_propagates_invalid_scale_factor_error() {
        let result = RopeTable::new_with_scaling(
            8,
            64,
            10_000.0,
            Some(&RopeScalingStrategy::Linear { scale_factor: 0.5 }),
        );
        assert!(
            matches!(result, Err(RopeScalingError::InvalidScaleFactor(f)) if (f - 0.5).abs() < 1e-9)
        );
    }

    /// Pins the `factor < 1.0` check `new_with_scaling`'s `Yarn` branch
    /// duplicates from `compute_rope_frequencies` (that branch never calls
    /// `compute_rope_frequencies` itself, since it needs the `f64`-precision
    /// `yarn_inv_freq_f64` instead — see `new_with_scaling`'s doc comment).
    /// If the two checks ever diverge, this test (and
    /// `yarn_rejects_sub_unity_factor` in `rope_scaling.rs`, which pins the
    /// `compute_rope_frequencies` side) is where it would show up.
    #[test]
    fn new_with_scaling_yarn_sub_unity_factor_errors_like_compute_rope_frequencies() {
        let result = RopeTable::new_with_scaling(
            128,
            64,
            1_000_000.0,
            Some(&RopeScalingStrategy::Yarn {
                original_max_position: 16384,
                factor: 0.5,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attn_factor: None,
            }),
        );
        assert!(
            matches!(result, Err(RopeScalingError::InvalidScaleFactor(f)) if (f - 0.5).abs() < 1e-9),
            "Yarn factor < 1.0 must error the same way Linear/Llama31 do, got: {result:?}"
        );
    }

    /// Pins the `InvalidHeadDim` check the `Yarn` branch of
    /// `new_with_scaling` now performs directly (verifier finding, minor):
    /// every other strategy gets this check via `compute_rope_frequencies`,
    /// but `Yarn` bypasses that function (see `new_with_scaling`'s doc
    /// comment), so without a duplicate check here it would silently hand
    /// an invalid `head_dim` to `yarn_inv_freq_f64` instead of reporting the
    /// same error every other strategy would.
    #[test]
    fn new_with_scaling_yarn_rejects_invalid_head_dim() {
        let strategy = bonsai_8b_yarn_strategy();

        let zero = RopeTable::new_with_scaling(0, 64, 1_000_000.0, Some(&strategy));
        assert!(
            matches!(zero, Err(RopeScalingError::InvalidHeadDim(0))),
            "head_dim=0 should be InvalidHeadDim(0), got: {zero:?}"
        );

        let odd = RopeTable::new_with_scaling(65, 64, 1_000_000.0, Some(&strategy));
        assert!(
            matches!(odd, Err(RopeScalingError::InvalidHeadDim(65))),
            "head_dim=65 (odd) should be InvalidHeadDim(65), got: {odd:?}"
        );
    }

    /// Confirms the `Yarn` branch's `head_dim` check does not leak onto the
    /// `None`/no-scaling path, which mirrors `RopeTable::new` and has never
    /// validated `head_dim` (odd values silently truncate via integer
    /// division rather than erroring) — that behaviour must not change as
    /// a side effect of adding the `Yarn`-only check above.
    #[test]
    fn new_with_scaling_none_still_accepts_odd_head_dim_like_plain_new() {
        let scaled = RopeTable::new_with_scaling(5, 16, 10_000.0, None)
            .expect("None strategy must keep accepting odd head_dim, like RopeTable::new");
        let plain = RopeTable::new(5, 16, 10_000.0);
        assert_eq!(scaled.cos_at(3), plain.cos_at(3));
    }

    #[test]
    fn new_with_scaling_yarn_factor_one_matches_plain_new() {
        // The mscale must not "leak in" when factor == 1.0 (no extension
        // requested): yarn_mscale(1.0, None) == 1.0, and the frequency blend
        // degenerates to the standard frequency for every dimension.
        //
        // Not compared with a tight epsilon: `scaled` is built by
        // `new_with_scaling`'s `Yarn` branch, which always uses the `f64`
        // precision pipeline (`from_frequencies_f64`), while `plain` is
        // built by the ordinary `f32`-throughout `new`/`new_with_freqs`.
        // These are two genuinely different (both individually correct)
        // floating-point paths, so their outputs differ by more than a
        // couple of ULPs — and, per `new_with_scaling`'s "Precision" doc
        // section, that gap grows with `position * frequency`, same as any
        // f32-vs-f64 RoPE comparison. The tolerance below
        // (`1e-5 + pos * 2e-7`) is generous enough to absorb that expected
        // growth (≈8e-4 at pos=4095) while remaining ~50x tighter than the
        // ~14% shift an actual mscale leak would cause.
        let strategy = RopeScalingStrategy::Yarn {
            original_max_position: 16384,
            factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: None,
        };
        let scaled = RopeTable::new_with_scaling(128, 4096, 1_000_000.0, Some(&strategy))
            .expect("Yarn factor=1.0 should succeed");
        let plain = RopeTable::new(128, 4096, 1_000_000.0);

        assert!((scaled.attention_scale() - 1.0).abs() < 1e-9);
        for pos in [0usize, 1, 4095] {
            let tolerance = 1e-5 + pos as f32 * 2e-7;
            let scaled_cos = scaled.cos_at(pos);
            let plain_cos = plain.cos_at(pos);
            let scaled_sin = scaled.sin_at(pos);
            let plain_sin = plain.sin_at(pos);
            for i in 0..scaled_cos.len() {
                assert!(
                    (scaled_cos[i] - plain_cos[i]).abs() < tolerance,
                    "cos mismatch at pos={pos} dim={i}: {} vs {} (tolerance {tolerance})",
                    scaled_cos[i],
                    plain_cos[i]
                );
                assert!(
                    (scaled_sin[i] - plain_sin[i]).abs() < tolerance,
                    "sin mismatch at pos={pos} dim={i}: {} vs {} (tolerance {tolerance})",
                    scaled_sin[i],
                    plain_sin[i]
                );
            }
        }
    }

    #[test]
    fn new_with_scaling_yarn_attention_scale_matches_formula() {
        let strategy = bonsai_8b_yarn_strategy();
        let table = RopeTable::new_with_scaling(128, 4096, 1_000_000.0, Some(&strategy))
            .expect("Yarn strategy should succeed");
        let expected = 0.1 * 4.0f32.ln() + 1.0;
        assert!(
            (table.attention_scale() - expected).abs() < 1e-6,
            "attention_scale {} != expected {}",
            table.attention_scale(),
            expected
        );
    }

    /// M-08 GATE: a YaRN table at factor 4.0 (the real Bonsai-8B shape —
    /// `original_context_length=16384`, `head_dim=128`,
    /// `rope_freq_base=1e6`) matches a from-formula reference to 1e-6 at
    /// positions 0, 16383, 16384 and 65535.
    ///
    /// The "reference" here is an independent, `f64`-precision
    /// re-derivation of the same published algorithm (`ggml`'s
    /// `rope_yarn` / HF `transformers`' `_compute_yarn_parameters`) written
    /// directly against the position/cos/sin definitions rather than by
    /// calling any function under test — there is no external golden file
    /// for this model in the scratchpad, so this is the strongest
    /// correctness check available without one.  The ramp-direction
    /// boundary check ([`yarn_ramp_direction_boundary_frequencies`] in
    /// `rope_scaling.rs`) is the fully-independent complement: it validates
    /// the two endpoints from simple closed forms that do not depend on this
    /// blended formula at all.
    #[test]
    fn yarn_table_matches_hand_derived_reference_at_key_positions() {
        const HEAD_DIM: usize = 128;
        const BASE: f32 = 1_000_000.0;
        const ORIGINAL_MAX_POSITION: usize = 16384;
        const FACTOR: f32 = 4.0;
        const MAX_SEQ_LEN: usize = 65536;

        let strategy = RopeScalingStrategy::Yarn {
            original_max_position: ORIGINAL_MAX_POSITION,
            factor: FACTOR,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: None,
        };
        let table = RopeTable::new_with_scaling(HEAD_DIM, MAX_SEQ_LEN, BASE, Some(&strategy))
            .expect("Yarn strategy should succeed");

        for &pos in &[0usize, 16383, 16384, 65535] {
            let (ref_cos, ref_sin) = reference_yarn_cos_sin(
                HEAD_DIM,
                BASE as f64,
                ORIGINAL_MAX_POSITION,
                FACTOR as f64,
                32.0,
                1.0,
                pos,
            );
            let got_cos = table.cos_at(pos);
            let got_sin = table.sin_at(pos);
            for i in 0..got_cos.len() {
                assert!(
                    (got_cos[i] as f64 - ref_cos[i]).abs() < 1e-6,
                    "pos={pos} dim={i}: cos got {} vs reference {}",
                    got_cos[i],
                    ref_cos[i]
                );
                assert!(
                    (got_sin[i] as f64 - ref_sin[i]).abs() < 1e-6,
                    "pos={pos} dim={i}: sin got {} vs reference {}",
                    got_sin[i],
                    ref_sin[i]
                );
            }
        }
    }

    /// In-scope (table-level) half of the M-08 GATE's second bullet:
    /// "greedy output for the real 8B at a 20K-token prompt changes ...
    /// assert it differs from the unscaled table."
    ///
    /// The *model-level* half — an actual greedy decode of
    /// `models/Bonsai-8B.gguf` — needs `RopeTable::new_with_scaling` wired
    /// into `BonsaiModel::from_gguf_with_embd`
    /// (`crate::model::types::mod::from_gguf_with_embd`, currently still
    /// calling plain `RopeTable::new` at that call site) plus a real GGUF
    /// file, neither of which this package can supply from within its own
    /// `owned_files` (`model/types/mod.rs` is MODEL-CORE-FWD's, and this
    /// worktree's `models/` is empty) — see the package's recorded
    /// deviations for the exact required call-site change. This test
    /// proves the part this package *does* own — the table itself — is not
    /// a no-op: built at the real Bonsai-8B shape
    /// (`head_dim=128, rope_freq_base=1e6, original_context_length=16384,
    /// factor=4.0, context_length=65536`), it diverges materially from the
    /// unscaled table it is meant to replace, at exactly the positions the
    /// GATE names (beyond the 16384-token original context).
    #[test]
    fn yarn_table_diverges_materially_from_unscaled_table_at_real_bonsai_8b_shape() {
        const HEAD_DIM: usize = 128;
        const BASE: f32 = 1_000_000.0;
        const MAX_SEQ_LEN: usize = 65536;

        let strategy = bonsai_8b_yarn_strategy();
        let scaled = RopeTable::new_with_scaling(HEAD_DIM, MAX_SEQ_LEN, BASE, Some(&strategy))
            .expect("Yarn strategy should succeed");
        let unscaled = RopeTable::new(HEAD_DIM, MAX_SEQ_LEN, BASE);

        for &pos in &[16384usize, 65535usize] {
            let scaled_cos = scaled.cos_at(pos);
            let scaled_sin = scaled.sin_at(pos);
            let unscaled_cos = unscaled.cos_at(pos);
            let unscaled_sin = unscaled.sin_at(pos);

            let mut sum_abs_diff = 0.0f32;
            let mut n = 0usize;
            for (a, b) in scaled_cos.iter().zip(unscaled_cos.iter()) {
                sum_abs_diff += (a - b).abs();
                n += 1;
            }
            for (a, b) in scaled_sin.iter().zip(unscaled_sin.iter()) {
                sum_abs_diff += (a - b).abs();
                n += 1;
            }
            let mean_abs_diff = sum_abs_diff / n as f32;

            // A uniform mscale (~1.1386 at factor=4.0) alone, plus the
            // NTK-by-parts frequency blend on top of it, produces a mean
            // absolute cos/sin divergence on the order of 0.1-0.2 at these
            // shapes — 0.02 is a conservative floor that a real "the
            // scaling is wired but produces near-identical output" bug
            // could not clear, while still being far below what an actual
            // divergence produces.
            assert!(
                mean_abs_diff > 0.02,
                "pos={pos}: YaRN-scaled table must differ materially from the \
                 unscaled table beyond the original context length (16384) — \
                 this is the exact defect M-08 exists to fix — got mean abs \
                 diff {mean_abs_diff}"
            );
        }
    }

    /// From-formula (`f64`) YaRN reference used only by
    /// `yarn_table_matches_hand_derived_reference_at_key_positions` above —
    /// see that test's doc comment for what independence this does and does
    /// not provide. (Not used by
    /// `yarn_table_diverges_materially_from_unscaled_table_at_real_bonsai_8b_shape`,
    /// which compares the table against plain `RopeTable::new` directly.)
    fn reference_yarn_cos_sin(
        head_dim: usize,
        base: f64,
        original_max_position: usize,
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        pos: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let half_dim = head_dim / 2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let correction_dim = |num_rotations: f64| -> f64 {
            (head_dim as f64 * (original_max_position as f64 / (num_rotations * two_pi)).ln())
                / (2.0 * base.ln())
        };
        // Matches the fix in `rope_scaling.rs::yarn_inv_freq_f64` (verifier
        // finding B3): `high` clamps to `head_dim - 1` (ggml/HF both clamp
        // against `head_dim`, not `half_dim`), and `low` has no upper
        // clamp at all. See that function's doc comment for the exact
        // ggml/HF citations. Keeping this test-only reference in sync with
        // the fix is required — this helper exists specifically so the
        // package's own gate test can catch a divergence in the
        // production formula, which it cannot do while it silently shares
        // the same bug.
        let low = correction_dim(beta_fast).floor().max(0.0);
        let high = correction_dim(beta_slow).ceil().min((head_dim - 1) as f64);
        let denom = (high - low).max(0.001);
        let mscale = if factor <= 1.0 {
            1.0
        } else {
            0.1 * factor.ln() + 1.0
        };

        let mut cos = Vec::with_capacity(half_dim);
        let mut sin = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq_extrap = 1.0 / base.powf(2.0 * i as f64 / head_dim as f64);
            let freq_interp = freq_extrap / factor;
            let ramp = ((i as f64 - low) / denom).clamp(0.0, 1.0);
            let inv_freq = freq_interp * ramp + freq_extrap * (1.0 - ramp);
            let theta = pos as f64 * inv_freq;
            cos.push(theta.cos() * mscale);
            sin.push(theta.sin() * mscale);
        }
        (cos, sin)
    }
}
