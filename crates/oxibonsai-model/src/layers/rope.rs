//! Rotary Position Embeddings (RoPE).
//!
//! Applies rotation to query and key vectors to encode positional
//! information. Uses precomputed cos/sin tables.

use crate::error::ModelResult;

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
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let base_inv_freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
                // Apply freq_factor divisor if provided.
                let inv_freq = if i < freq_factors.len() && freq_factors[i] > 1.0 {
                    base_inv_freq / freq_factors[i]
                } else {
                    base_inv_freq
                };
                let angle = pos as f32 * inv_freq;
                cos[pos * half_dim + i] = angle.cos();
                sin[pos * half_dim + i] = angle.sin();
            }
        }

        Self {
            cos,
            sin,
            half_dim,
            max_seq_len,
        }
    }

    /// Apply RoPE rotation to a query or key vector at the given position.
    ///
    /// - `vec`: Input vector of length `head_dim` (modified in-place via output).
    /// - `output`: Output vector of length `head_dim`.
    /// - `pos`: Token position in the sequence.
    ///
    /// Delegates the inner rotation to SIMD-accelerated `oxibonsai_kernels::rope_apply_simd`.
    pub fn apply(&self, vec: &[f32], output: &mut [f32], pos: usize) -> ModelResult<()> {
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(vec.len(), self.half_dim * 2);
        debug_assert!(output.len() >= self.half_dim * 2);

        let cos_row = &self.cos[pos * self.half_dim..(pos + 1) * self.half_dim];
        let sin_row = &self.sin[pos * self.half_dim..(pos + 1) * self.half_dim];

        oxibonsai_kernels::rope_apply_simd(vec, output, cos_row, sin_row);

        Ok(())
    }

    /// Maximum precomputed sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get cos values for a given position: `&[half_dim]`.
    pub fn cos_at(&self, pos: usize) -> &[f32] {
        &self.cos[pos * self.half_dim..(pos + 1) * self.half_dim]
    }

    /// Get sin values for a given position: `&[half_dim]`.
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
}
