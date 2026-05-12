//! Scalar Q4_0 GEMV reference kernel.
//!
//! Computes `output[row] = dot(W_row, input)` for each row of the weight matrix
//! `W`, where weights are stored as Q4_0 blocks (32 weights per block, 18 bytes).
//!
//! This is a pure scalar Rust correctness-reference implementation — no SIMD,
//! no unsafe. The inner loop dequantizes one block at a time into a 32-element
//! stack buffer to keep stack pressure predictable.

use oxibonsai_core::{BlockQ4_0, QK_Q4_0};

use crate::error::{KernelError, KernelResult};

/// Scalar GEMV for Q4_0-quantized weight matrix.
///
/// Computes `output[row] = dot(weight_row, input)` for each row, where the
/// weight matrix is stored row-major as Q4_0 blocks.
///
/// - `blocks`: Row-major weight blocks.  Row `r` occupies
///   `blocks[r * blocks_per_row .. (r+1) * blocks_per_row]`
///   where `blocks_per_row = in_features / QK_Q4_0`.
/// - `input`: FP32 input vector of length `in_features`.
/// - `output`: FP32 output vector of length `n_rows`.
/// - `n_rows`: Number of output rows (= out_features).
/// - `in_features`: Inner dimension (must be divisible by `QK_Q4_0 = 32`).
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `in_features % QK_Q4_0 != 0`.
/// - [`KernelError::DimensionMismatch`] if `blocks.len() < n_rows * blocks_per_row`
///   or `input.len() < in_features`.
/// - [`KernelError::BufferTooSmall`] if `output.len() < n_rows`.
pub fn gemv_q4_0(
    blocks: &[BlockQ4_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
) -> KernelResult<()> {
    if in_features % QK_Q4_0 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: in_features,
            block_size: QK_Q4_0,
        });
    }
    let blocks_per_row = in_features / QK_Q4_0;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }
    if input.len() < in_features {
        return Err(KernelError::DimensionMismatch {
            expected: in_features,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let mut tmp = [0.0f32; 32];
    for row in 0..n_rows {
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + b];
            block.dequant_to_buf(&mut tmp);
            let base = b * QK_Q4_0;
            for j in 0..QK_Q4_0 {
                acc += tmp[j] * input[base + j];
            }
        }
        output[row] = acc;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_q4_block(scale: f32, nibbles: [u8; 32]) -> BlockQ4_0 {
        // Pack 32 nibbles into 16 bytes (low nibble = even index).
        let mut qs = [0u8; 16];
        for j in 0..32 {
            let n = nibbles[j] & 0x0F;
            if j % 2 == 0 {
                qs[j / 2] = n;
            } else {
                qs[j / 2] |= n << 4;
            }
        }
        BlockQ4_0 {
            d: f16::from_f32(scale),
            qs,
        }
    }

    /// All nibbles = 8 → weight = scale*(8-8) = 0; dot product is 0.
    #[test]
    fn q4_0_all_zero_weights() {
        let block = make_q4_block(1.0, [8u8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![99.0f32; 1];
        gemv_q4_0(&blocks, &input, &mut output, 1, 32).unwrap();
        assert!(
            output[0].abs() < 1e-5,
            "all-zero weights → output 0, got {}",
            output[0]
        );
    }

    /// Single row, single block of nibble=15 (max positive = scale*(15-8)=7*scale).
    /// With all-1 input, output = 32 * 7 * scale.
    #[test]
    fn q4_0_max_nibbles_single_block() {
        let scale = 1.0f32;
        let block = make_q4_block(scale, [15u8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 1];
        gemv_q4_0(&blocks, &input, &mut output, 1, 32).unwrap();
        let expected = 32.0 * 7.0 * scale;
        assert!(
            (output[0] - expected).abs() < 1.0,
            "max nibbles: expected {expected}, got {}",
            output[0]
        );
    }

    /// 3 rows × 64 features (2 blocks per row).
    #[test]
    fn q4_0_multiple_rows() {
        // All nibbles=8 → 0 weights.
        let zero_block = make_q4_block(1.0, [8u8; 32]);
        // nibble=9 → weight = 1.0*(9-8) = 1.0
        let unit_block = make_q4_block(1.0, [9u8; 32]);

        // Row 0: [unit, zero] → dot = 32*1.0 + 0 = 32.0
        // Row 1: [zero, unit] → dot = 0 + 32*1.0 = 32.0
        // Row 2: [unit, unit] → dot = 64.0
        let blocks = vec![
            unit_block, zero_block, // row 0
            zero_block, unit_block, // row 1
            unit_block, unit_block, // row 2
        ];
        let input = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 3];
        gemv_q4_0(&blocks, &input, &mut output, 3, 64).unwrap();
        assert!((output[0] - 32.0).abs() < 1.0, "row0: {}", output[0]);
        assert!((output[1] - 32.0).abs() < 1.0, "row1: {}", output[1]);
        assert!((output[2] - 64.0).abs() < 1.0, "row2: {}", output[2]);
    }

    /// in_features not a multiple of 32 → NotBlockAligned error.
    #[test]
    fn q4_0_not_block_aligned() {
        let block = make_q4_block(1.0, [8u8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 31];
        let mut output = vec![0.0f32; 1];
        let result = gemv_q4_0(&blocks, &input, &mut output, 1, 31);
        assert!(
            matches!(result, Err(KernelError::NotBlockAligned { .. })),
            "expected NotBlockAligned, got {result:?}"
        );
    }

    /// Too few blocks → DimensionMismatch error.
    #[test]
    fn q4_0_wrong_block_count() {
        let block = make_q4_block(1.0, [8u8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 2];
        let result = gemv_q4_0(&blocks, &input, &mut output, 2, 32);
        assert!(
            matches!(result, Err(KernelError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    /// Output buffer too small → BufferTooSmall error.
    #[test]
    fn q4_0_output_too_small() {
        let block = make_q4_block(1.0, [8u8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![];
        let result = gemv_q4_0(&blocks, &input, &mut output, 1, 32);
        assert!(
            matches!(result, Err(KernelError::BufferTooSmall { .. })),
            "expected BufferTooSmall, got {result:?}"
        );
    }

    /// Quantize→dequant round-trip, then GEMV vs naive FP32 matmul.
    #[test]
    fn q4_0_gemv_matches_quantized_dequant() {
        use oxibonsai_core::BlockQ4_0 as BQ;
        let raw: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 8.0).collect();
        let blocks = BQ::quantize(&raw).unwrap();

        // Build dequantized reference
        let mut deq = vec![0.0f32; 64];
        BQ::dequant(&blocks, &mut deq).unwrap();

        let input = vec![1.0f32; 64];
        // Reference: sum of all dequantized weights (input=1)
        let reference: f32 = deq.iter().sum();

        let mut output = vec![0.0f32; 1];
        gemv_q4_0(&blocks, &input, &mut output, 1, 64).unwrap();
        assert!(
            (output[0] - reference).abs() < 1e-3,
            "GEMV must match dequant+dot: expected {reference}, got {}",
            output[0]
        );
    }
}
