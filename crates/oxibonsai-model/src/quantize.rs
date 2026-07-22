//! FP32 → Q1\_0\_g128 quantization and related utilities.
//!
//! # Q1\_0\_g128 block format
//!
//! Each block covers exactly `GROUP_SIZE` (128) weights:
//!
//! ```text
//! ┌──────────────────┬──────────────────────────────────────────────────────┐
//! │  2 bytes         │  16 bytes                                            │
//! │  FP16 scale      │  128 sign bits (1 bit per weight)                   │
//! │  = max(|w_i|)    │  bit=1 → +scale, bit=0 → −scale                     │
//! └──────────────────┴──────────────────────────────────────────────────────┘
//! ```
//!
//! This is the canonical Q1\_0\_g128 sign convention shared by every reader
//! and kernel in the workspace: [`oxibonsai_core::tensor::BlockQ1_0G128`]
//! (`w[i] = bit[i] ? +d : -d`), the CPU/CUDA/Metal GEMV and GEMM kernels in
//! `oxibonsai-kernels`, and `oxibonsai-model`'s GGUF weight loaders. The
//! encoder and decoder below MUST stay in lock-step with that convention —
//! a mismatch silently negates every 1-bit weight after export→load.
//!
//! Total block size: **18 bytes** per 128 weights → ~1.125 bits/weight.

use half::f16;

/// Number of weights per quantization group.
pub const GROUP_SIZE: usize = 128;

/// Byte size of one encoded block.
///
/// Layout: `[f16 scale: 2 bytes][sign bits: 16 bytes]`
pub const BLOCK_BYTES: usize = 18; // 2 (f16) + 16 (sign bits)

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can arise during Q1\_0\_g128 quantization or dequantization.
#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    /// Input slice length is not a multiple of `GROUP_SIZE`.
    #[error("Input length {got} is not a multiple of GROUP_SIZE ({GROUP_SIZE})")]
    NotAligned { got: usize },

    /// Encoded data length is not a multiple of `BLOCK_BYTES`.
    #[error("Data length {got} is not a multiple of BLOCK_BYTES ({BLOCK_BYTES})")]
    InvalidBlockData { got: usize },

    /// All weights in a group are exactly zero — scale would be zero and
    /// the dequantised values would all be zero regardless of sign bits.
    /// Callers that can tolerate this may catch it and substitute zeros.
    #[error("All weights in group are zero — cannot determine scale")]
    ZeroGroup,
}

// ─── Core group-level primitives ─────────────────────────────────────────────

/// Quantize a single group of exactly `GROUP_SIZE` f32 values into one block.
///
/// The group **must** have at least one non-zero element; if every element is
/// zero, the block is written with a zero scale and all sign bits cleared, but
/// no error is returned (the dequantised values will simply all be zero).
pub fn quantize_group(weights: &[f32]) -> [u8; BLOCK_BYTES] {
    debug_assert_eq!(
        weights.len(),
        GROUP_SIZE,
        "quantize_group: input must be exactly {GROUP_SIZE} elements"
    );

    // ── Find the maximum absolute value (the scale) ──────────────────────
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);

    let mut block = [0u8; BLOCK_BYTES];

    // Store FP16 scale in the first two bytes.
    let scale_f16 = f16::from_f32(max_abs);
    let scale_bits = scale_f16.to_bits();
    block[0] = (scale_bits & 0xFF) as u8;
    block[1] = (scale_bits >> 8) as u8;

    // ── Encode sign bits ─────────────────────────────────────────────────
    // bit = 1 → positive (≥ 0), bit = 0 → negative (< 0).
    //
    // This MUST match the canonical convention used by
    // `oxibonsai_core::tensor::BlockQ1_0G128::weight` and every consumer of
    // GGUF tensor type 41 (kernels, weight loaders) — see module docs.
    for (i, &w) in weights.iter().enumerate() {
        if w >= 0.0 {
            let byte_idx = i / 8 + 2; // +2 to skip the scale bytes
            let bit_idx = i % 8;
            block[byte_idx] |= 1 << bit_idx;
        }
    }

    block
}

/// Dequantize a single 18-byte block back to `GROUP_SIZE` f32 values.
pub fn dequantize_block(block: &[u8; BLOCK_BYTES]) -> [f32; GROUP_SIZE] {
    let scale_bits = u16::from(block[0]) | (u16::from(block[1]) << 8);
    let scale = f16::from_bits(scale_bits).to_f32();

    let mut out = [0.0_f32; GROUP_SIZE];
    for (i, slot) in out.iter_mut().enumerate().take(GROUP_SIZE) {
        let byte_idx = i / 8 + 2;
        let bit_idx = i % 8;
        let sign_bit = (block[byte_idx] >> bit_idx) & 1;
        *slot = if sign_bit != 0 { scale } else { -scale };
    }
    out
}

// ─── Slice-level API ─────────────────────────────────────────────────────────

/// Quantize a slice of f32 weights to Q1\_0\_g128 format.
///
/// The input length must be a multiple of [`GROUP_SIZE`].
///
/// Returns a `Vec<u8>` whose length is
/// `(weights.len() / GROUP_SIZE) * BLOCK_BYTES`.
pub fn quantize_q1_0_g128(weights: &[f32]) -> Result<Vec<u8>, QuantizeError> {
    if weights.len() % GROUP_SIZE != 0 {
        return Err(QuantizeError::NotAligned { got: weights.len() });
    }

    let num_blocks = weights.len() / GROUP_SIZE;
    let mut out = Vec::with_capacity(num_blocks * BLOCK_BYTES);

    for chunk in weights.chunks_exact(GROUP_SIZE) {
        let block = quantize_group(chunk);
        out.extend_from_slice(&block);
    }

    Ok(out)
}

/// Dequantize Q1\_0\_g128 bytes back to f32 weights.
///
/// The input length must be a multiple of [`BLOCK_BYTES`].
pub fn dequantize_q1_0_g128(data: &[u8]) -> Result<Vec<f32>, QuantizeError> {
    if data.len() % BLOCK_BYTES != 0 {
        return Err(QuantizeError::InvalidBlockData { got: data.len() });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(num_blocks * GROUP_SIZE);

    for chunk in data.chunks_exact(BLOCK_BYTES) {
        let block: &[u8; BLOCK_BYTES] = chunk
            .try_into()
            .expect("chunks_exact guarantees correct length");
        let decoded = dequantize_block(block);
        out.extend_from_slice(&decoded);
    }

    Ok(out)
}

// ─── Size estimation ──────────────────────────────────────────────────────────

/// Estimate how many bytes a tensor with `num_weights` elements would occupy
/// in Q1\_0\_g128 format.
///
/// Uses ceiling division so tensors whose weight count is not a multiple of
/// [`GROUP_SIZE`] are still accounted for correctly.
#[inline]
pub fn q1_0_g128_size_bytes(num_weights: usize) -> usize {
    num_weights.div_ceil(GROUP_SIZE) * BLOCK_BYTES
}

// ─── Weight statistics ────────────────────────────────────────────────────────

/// Descriptive statistics about a weight tensor before quantization.
#[derive(Debug, Clone)]
pub struct WeightStats {
    /// Minimum weight value.
    pub min: f32,
    /// Maximum weight value.
    pub max: f32,
    /// Arithmetic mean.
    pub mean: f32,
    /// Population standard deviation.
    pub std: f32,
    /// Fraction of weights considered "near zero" (|w| < 0.01).
    pub sparsity: f32,
    /// Total number of weights.
    pub num_weights: usize,
}

/// Compute descriptive statistics for a weight slice.
///
/// Returns a [`WeightStats`] with all fields set to zero / NaN-free values
/// even for empty slices.
pub fn compute_weight_stats(weights: &[f32]) -> WeightStats {
    let num_weights = weights.len();

    if num_weights == 0 {
        return WeightStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            sparsity: 0.0,
            num_weights: 0,
        };
    }

    let mut min = weights[0];
    let mut max = weights[0];
    let mut sum = 0.0_f64;
    let mut near_zero: usize = 0;

    for &w in weights {
        if w < min {
            min = w;
        }
        if w > max {
            max = w;
        }
        sum += f64::from(w);
        if w.abs() < 0.01 {
            near_zero += 1;
        }
    }

    let mean = (sum / num_weights as f64) as f32;

    // Population standard deviation
    let variance = weights
        .iter()
        .map(|&w| {
            let diff = f64::from(w) - f64::from(mean);
            diff * diff
        })
        .sum::<f64>()
        / num_weights as f64;
    let std = variance.sqrt() as f32;

    let sparsity = near_zero as f32 / num_weights as f32;

    WeightStats {
        min,
        max,
        mean,
        std,
        sparsity,
        num_weights,
    }
}

// ─── Quantization error analysis ─────────────────────────────────────────────

/// Summary statistics comparing original f32 weights to their Q1\_0\_g128
/// encoding.
#[derive(Debug, Clone)]
pub struct QuantizationError {
    /// Mean squared error between original and reconstructed weights.
    pub mse: f32,
    /// Largest absolute difference between any single original and reconstructed weight.
    pub max_abs_error: f32,
    /// Signal-to-noise ratio in dB: `10 · log10(signal_power / noise_power)`.
    /// A higher value indicates less distortion.
    pub snr_db: f32,
    /// Effective bits per weight for this format — should be ~1.125 for Q1\_0\_g128.
    pub bits_per_weight: f32,
}

/// Analyse the quantization error between `original` f32 weights and their
/// Q1\_0\_g128 encoding in `quantized`.
///
/// `quantized` must be the byte slice produced by [`quantize_q1_0_g128`] for
/// the same `original` slice.
pub fn analyze_quantization_error(
    original: &[f32],
    quantized: &[u8],
) -> Result<QuantizationError, QuantizeError> {
    let reconstructed = dequantize_q1_0_g128(quantized)?;

    // The reconstructed slice will have length = num_blocks * GROUP_SIZE, which
    // is ≥ original.len(). We only compare against original.len() elements.
    let n = original.len();
    if n == 0 {
        return Ok(QuantizationError {
            mse: 0.0,
            max_abs_error: 0.0,
            snr_db: f32::INFINITY,
            bits_per_weight: BLOCK_BYTES as f32 * 8.0 / GROUP_SIZE as f32,
        });
    }

    let mut sum_sq_error = 0.0_f64;
    let mut max_abs_error = 0.0_f32;
    let mut signal_power = 0.0_f64;

    for i in 0..n {
        let orig = original[i];
        let recon = reconstructed[i];
        let err = orig - recon;
        sum_sq_error += f64::from(err * err);
        let abs_err = err.abs();
        if abs_err > max_abs_error {
            max_abs_error = abs_err;
        }
        signal_power += f64::from(orig * orig);
    }

    let mse = (sum_sq_error / n as f64) as f32;
    let noise_power = sum_sq_error / n as f64;

    let snr_db = if noise_power == 0.0 {
        f32::INFINITY
    } else {
        let snr_linear = (signal_power / n as f64) / noise_power;
        (10.0 * snr_linear.log10()) as f32
    };

    let bits_per_weight = BLOCK_BYTES as f32 * 8.0 / GROUP_SIZE as f32;

    Ok(QuantizationError {
        mse,
        max_abs_error,
        snr_db,
        bits_per_weight,
    })
}

// ─── Round-trip helper ────────────────────────────────────────────────────────

/// Round each weight to the nearest representable Q1\_0 value for analysis.
///
/// Each weight is replaced by either `+scale` or `−scale` where `scale` is the
/// per-group maximum absolute value.  The group boundaries are determined by
/// [`GROUP_SIZE`]; any trailing weights that do not fill a complete group are
/// handled as a short group using the scale of that partial group.
pub fn round_to_q1_0(weights: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(weights.len());

    for chunk in weights.chunks(GROUP_SIZE) {
        let max_abs = chunk.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);
        let scale = f16::from_f32(max_abs).to_f32(); // apply FP16 rounding to scale

        for &w in chunk {
            out.push(if w >= 0.0 { scale } else { -scale });
        }
    }

    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a group of GROUP_SIZE f32 values.
    fn uniform_group(v: f32) -> Vec<f32> {
        vec![v; GROUP_SIZE]
    }

    // ── quantize_group ────────────────────────────────────────────────────

    #[test]
    fn test_quantize_group_basic() {
        // Mix of positive and negative values with a known max.
        let mut weights = vec![0.0_f32; GROUP_SIZE];
        weights[0] = 2.0;
        weights[1] = -1.0;
        weights[2] = 0.5;

        let block = quantize_group(&weights);

        // Verify scale == f16(2.0)
        let scale_bits = u16::from(block[0]) | (u16::from(block[1]) << 8);
        let scale = f16::from_bits(scale_bits).to_f32();
        assert!(
            (scale - 2.0).abs() < 1e-3,
            "scale should be ~2.0, got {scale}"
        );

        // index 1 is negative → bit 1 of byte 2 should be clear
        assert_eq!(block[2] & (1 << 1), 0, "weight[1] is negative");
        // index 0 is positive → bit 0 of byte 2 should be set
        assert_ne!(block[2] & 1, 0, "weight[0] is positive");
    }

    #[test]
    fn test_quantize_group_all_positive() {
        let weights = uniform_group(3.0);
        let block = quantize_group(&weights);

        // All sign bits should be 1 (positive)
        for byte in &block[2..] {
            assert_eq!(
                *byte, 0xFFu8,
                "all sign bits should be 1 for positive weights"
            );
        }
    }

    #[test]
    fn test_quantize_group_all_negative() {
        let weights = uniform_group(-1.5);
        let block = quantize_group(&weights);

        // All sign bits should be 0 (negative)
        for byte in &block[2..] {
            assert_eq!(
                *byte, 0x00,
                "all sign bits should be 0 for negative weights"
            );
        }
    }

    // ── dequantize_block ──────────────────────────────────────────────────

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Use a simple pattern: alternating +1 and -1.
        let weights: Vec<f32> = (0..GROUP_SIZE)
            .map(|i| if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 })
            .collect();

        let block = quantize_group(&weights);
        let decoded = dequantize_block(&block);

        let scale = f16::from_f32(1.0).to_f32();
        for (i, &d) in decoded.iter().enumerate() {
            let expected = if i % 2 == 0 { scale } else { -scale };
            assert!(
                (d - expected).abs() < 1e-3,
                "decoded[{i}] = {d}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_error_analysis() {
        // Weights already at ±1 — MSE should be very small after round-tripping
        // through FP16 scale.
        let weights: Vec<f32> = (0..GROUP_SIZE * 4)
            .map(|i| if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 })
            .collect();

        let quantized = quantize_q1_0_g128(&weights).expect("quantize");
        let err = analyze_quantization_error(&weights, &quantized).expect("analyze");

        // Scale = f16(1.0) = 1.0 exactly → no error expected.
        assert!(
            err.mse < 1e-6,
            "MSE should be near zero for ±1.0 weights, got {}",
            err.mse
        );
        assert!(
            (err.bits_per_weight - 1.125).abs() < 1e-6,
            "bits_per_weight should be 1.125"
        );
    }

    // ── q1_0_g128_size_bytes ──────────────────────────────────────────────

    #[test]
    fn test_q1_0_g128_size_bytes() {
        assert_eq!(q1_0_g128_size_bytes(0), 0);
        assert_eq!(q1_0_g128_size_bytes(128), BLOCK_BYTES);
        assert_eq!(q1_0_g128_size_bytes(256), 2 * BLOCK_BYTES);
        // Partial group: 129 elements → 2 blocks
        assert_eq!(q1_0_g128_size_bytes(129), 2 * BLOCK_BYTES);
    }

    // ── compute_weight_stats ──────────────────────────────────────────────

    #[test]
    fn test_weight_stats_basic() {
        // [-1, 0, 1] → mean=0, min=-1, max=1
        let weights = vec![-1.0_f32, 0.0, 1.0];
        let stats = compute_weight_stats(&weights);
        assert_eq!(stats.num_weights, 3);
        assert!((stats.min - (-1.0)).abs() < 1e-6);
        assert!((stats.max - 1.0).abs() < 1e-6);
        assert!(stats.mean.abs() < 1e-6);
    }

    #[test]
    fn test_weight_stats_sparsity() {
        // 50% of weights are near zero
        let weights: Vec<f32> = (0..100)
            .map(|i| if i < 50 { 0.005_f32 } else { 1.0_f32 })
            .collect();
        let stats = compute_weight_stats(&weights);
        assert!(
            (stats.sparsity - 0.5).abs() < 1e-6,
            "sparsity should be 0.5, got {}",
            stats.sparsity
        );
    }

    // ── analyze_quantization_error ────────────────────────────────────────

    #[test]
    fn test_analyze_quantization_error() {
        // Weights with varying magnitude — just verify the API works and
        // the reported bits_per_weight is correct.
        let weights: Vec<f32> = (0..GROUP_SIZE * 2)
            .map(|i| (i as f32) * 0.1 - 6.4)
            .collect();
        let quantized = quantize_q1_0_g128(&weights).expect("quantize");
        let err = analyze_quantization_error(&weights, &quantized).expect("analyze");

        assert!(err.mse >= 0.0, "MSE must be non-negative");
        assert!(err.max_abs_error >= 0.0);
        assert!((err.bits_per_weight - 1.125).abs() < 1e-6);
    }

    // ── round_to_q1_0 ─────────────────────────────────────────────────────

    #[test]
    fn test_round_to_q1_0() {
        let weights: Vec<f32> = vec![2.0, -2.0, 1.0, -1.0];
        // Pad to GROUP_SIZE (round_to_q1_0 uses partial groups)
        let rounded = round_to_q1_0(&weights);
        assert_eq!(rounded.len(), weights.len());

        // Scale = f16(2.0) ≈ 2.0 → positive weights → +scale, negative → -scale
        let scale = f16::from_f32(2.0).to_f32();
        assert!((rounded[0] - scale).abs() < 1e-3, "positive weight");
        assert!((rounded[1] - (-scale)).abs() < 1e-3, "negative weight");
    }

    // ── Error paths ───────────────────────────────────────────────────────

    #[test]
    fn test_quantize_wrong_length_returns_error() {
        let weights = vec![1.0_f32; 100]; // 100 is not a multiple of 128
        let result = quantize_q1_0_g128(&weights);
        assert!(
            matches!(result, Err(QuantizeError::NotAligned { got: 100 })),
            "expected NotAligned error"
        );
    }

    #[test]
    fn test_quantize_zero_group_handled() {
        // A group of all zeros — scale will be 0, all dequantized values will be 0.
        // The function should NOT return an error; ZeroGroup is informational only.
        let weights = vec![0.0_f32; GROUP_SIZE];
        let result = quantize_q1_0_g128(&weights);
        assert!(result.is_ok(), "all-zero group should not error");

        let bytes = result.expect("quantize");
        let decoded = dequantize_q1_0_g128(&bytes).expect("dequantize");
        for v in &decoded {
            assert_eq!(*v, 0.0, "dequantized zero group should all be zero");
        }
    }

    // ── dequantize error path ─────────────────────────────────────────────

    #[test]
    fn test_dequantize_wrong_length_returns_error() {
        let data = vec![0u8; 17]; // 17 is not a multiple of 18
        let result = dequantize_q1_0_g128(&data);
        assert!(
            matches!(result, Err(QuantizeError::InvalidBlockData { got: 17 })),
            "expected InvalidBlockData error"
        );
    }

    // ── empty slice edge case ─────────────────────────────────────────────

    #[test]
    fn test_compute_weight_stats_empty() {
        let stats = compute_weight_stats(&[]);
        assert_eq!(stats.num_weights, 0);
        assert_eq!(stats.sparsity, 0.0);
    }
}
