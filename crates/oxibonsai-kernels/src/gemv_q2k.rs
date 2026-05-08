//! Scalar GEMV kernel for Q2_K quantized weight matrices.
//!
//! Implements `y = W × x` where W is stored as Q2_K blocks.
//! Each super-block covers 256 weights (QK_K = 256).

use oxibonsai_core::BlockQ2K;

use crate::error::{KernelError, KernelResult};

/// Scalar Q2_K GEMV: computes `output = weight_matrix × input`.
///
/// The weight matrix `W` is stored in row-major Q2_K format:
/// row `i` starts at block index `i * blocks_per_row` where
/// `blocks_per_row = in_features / 256`.
///
/// # Parameters
///
/// - `blocks`:      Q2_K-quantized weight blocks in row-major order.
/// - `input`:       FP32 input vector of length `in_features`.
/// - `output`:      FP32 output vector of length `n_rows`.
/// - `n_rows`:      Number of output rows (out_features).
/// - `in_features`: Inner dimension, must be a multiple of 256 (QK_K).
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `in_features % 256 != 0`.
/// - [`KernelError::DimensionMismatch`] if `blocks` or `input` are too short.
/// - [`KernelError::BufferTooSmall`] if `output` is too short.
pub fn gemv_q2k(
    blocks: &[BlockQ2K],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
) -> KernelResult<()> {
    const QK_K: usize = 256;

    if in_features == 0 || in_features % QK_K != 0 {
        return Err(KernelError::NotBlockAligned {
            count: in_features,
            block_size: QK_K,
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

    let blocks_per_row = in_features / QK_K;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    let mut row_buf = vec![0.0f32; in_features];
    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];

        // Dequantize the entire row into a temporary FP32 buffer.
        BlockQ2K::dequant(row_blocks, &mut row_buf).map_err(KernelError::Core)?;

        // Dot product with the input vector.
        let acc: f32 = row_buf
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum();
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
    use oxibonsai_core::BlockQ2K;

    fn make_q2k_block(value: f32) -> BlockQ2K {
        let input = vec![value; 256];
        let blocks = BlockQ2K::quantize(&input).expect("quantize ok");
        blocks[0]
    }

    #[test]
    fn gemv_q2k_single_row_uniform() {
        // One row, uniform weight = 1.0, input all 1.0.
        // Expected output ≈ 256.0 (with quantization error < 10%).
        let block = make_q2k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        gemv_q2k(&[block], &input, &mut output, 1, 256).expect("gemv ok");
        assert!(
            (output[0] - 256.0).abs() < 30.0,
            "expected ~256.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemv_q2k_two_rows() {
        let block_pos = make_q2k_block(0.5);
        let block_neg = make_q2k_block(-0.5);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];

        gemv_q2k(&[block_pos, block_neg], &input, &mut output, 2, 256).expect("gemv ok");
        assert!(
            output[0] > 0.0,
            "row 0 should be positive, got {}",
            output[0]
        );
        assert!(
            output[1] < 0.0,
            "row 1 should be negative, got {}",
            output[1]
        );
    }

    #[test]
    fn gemv_q2k_not_block_aligned_errors() {
        let block = make_q2k_block(1.0);
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(
            gemv_q2k(&[block], &input, &mut output, 1, 100).is_err(),
            "should error when in_features not multiple of 256"
        );
    }

    #[test]
    fn gemv_q2k_wrong_block_count_errors() {
        let block = make_q2k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];
        assert!(
            gemv_q2k(&[block], &input, &mut output, 2, 256).is_err(),
            "should error on block count mismatch"
        );
    }

    #[test]
    fn gemv_q2k_output_too_small_errors() {
        let block = make_q2k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 0];
        assert!(
            gemv_q2k(&[block], &input, &mut output, 1, 256).is_err(),
            "should error when output buffer is too small"
        );
    }
}
