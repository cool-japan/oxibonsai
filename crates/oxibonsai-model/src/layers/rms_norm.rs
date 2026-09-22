//! RMSNorm (Root Mean Square Layer Normalization).
//!
//! Used as pre-norm before attention and FFN in each Transformer block.
//! `output[i] = weight[i] * (input[i] / rms(input))`
//! where `rms(x) = sqrt(mean(x^2) + eps)`.

use crate::error::ModelResult;

/// RMSNorm layer with learnable weight vector.
#[derive(Debug)]
pub struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    /// Create a new RMSNorm layer.
    ///
    /// - `weight`: Per-element scale weights (length = hidden_size).
    /// - `eps`: Small constant for numerical stability.
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Apply RMSNorm to an input vector in-place.
    ///
    /// `output[i] = weight[i] * input[i] / rms(input)`
    ///
    /// Delegates to the SIMD-accelerated implementation in `oxibonsai_kernels`,
    /// propagating its length-contract validation (K-02/M-01) instead of
    /// asserting: a GGUF file is untrusted input, and a norm tensor shorter
    /// than `hidden_size`/`head_dim` must fail cleanly here rather than
    /// reaching the NEON/AVX2 kernels with a too-short `weight` slice.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::Kernel`] if `self.weight.len() !=
    /// input.len()` or `output.len() < input.len()`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        oxibonsai_kernels::rms_norm_simd(input, &self.weight, output, self.eps)?;
        Ok(())
    }

    /// Hidden size (dimension of weight vector).
    pub fn hidden_size(&self) -> usize {
        self.weight.len()
    }

    /// Access the raw weight vector (for batch GPU dispatch).
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Access the epsilon value (for batch GPU dispatch).
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weights() {
        let weight = vec![1.0; 4];
        let norm = RmsNorm::new(weight, 1e-6);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        norm.forward(&input, &mut output)
            .expect("rms norm forward should succeed");

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (30.0f32 / 4.0).sqrt();
        for i in 0..4 {
            let expected = input[i] / rms;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "at {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn rms_norm_with_scale() {
        let weight = vec![2.0; 4];
        let norm = RmsNorm::new(weight, 1e-6);

        let input = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        norm.forward(&input, &mut output)
            .expect("rms norm forward should succeed");

        // RMS = sqrt(1) = 1.0, so output = 2.0 * 1.0 / 1.0 = 2.0
        for &v in &output {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    // ── K-02 / M-01: real length-contract errors, not a debug_assert ────

    #[test]
    fn forward_rejects_mismatched_weight_length() {
        // Simulates a GGUF file whose norm tensor was declared with the
        // wrong shape (e.g. `attn_q_norm.weight` shorter than `head_dim`).
        let weight = vec![1.0; 3]; // hidden_size should be 4
        let norm = RmsNorm::new(weight, 1e-6);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        let result = norm.forward(&input, &mut output);
        assert!(
            result.is_err(),
            "forward() must reject a weight/input length mismatch instead of \
             reading out of bounds in release (K-02/M-01)"
        );
    }

    #[test]
    fn forward_rejects_short_output_buffer() {
        let weight = vec![1.0; 4];
        let norm = RmsNorm::new(weight, 1e-6);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 2]; // too short

        let result = norm.forward(&input, &mut output);
        assert!(
            result.is_err(),
            "forward() must reject a too-short output buffer instead of \
             writing out of bounds in release (K-02/M-01)"
        );
    }
}
