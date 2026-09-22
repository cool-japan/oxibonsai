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

    if in_features == 0 || !in_features.is_multiple_of(QK_K) {
        return Err(KernelError::NotBlockAligned {
            count: in_features,
            block_size: QK_K,
        });
    }
    if input.len() < in_features {
        return Err(KernelError::dimension_mismatch(
            "input",
            in_features,
            input.len(),
        ));
    }
    if output.len() < n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            n_rows,
            output.len(),
        ));
    }

    let blocks_per_row = in_features / QK_K;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    // Row-parallel scalar GEMV: each output row is an independent
    // dequantize-then-dot, so the row loop is split across Rayon threads for
    // large `n_rows` (numerically identical to sequential — see the driver).
    crate::parallel::gemv_kquant_row_parallel(input, output, n_rows, in_features, |row, row_buf| {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        BlockQ2K::dequant(row_blocks, row_buf).map_err(KernelError::Core)
    })
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

    /// Named-error migration (K-02): the production entry points now name
    /// which buffer was wrong.
    #[test]
    fn gemv_q2k_errors_name_the_offending_buffer() {
        let block = make_q2k_block(1.0);
        let short_input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        let err = gemv_q2k(&[block], &short_input, &mut output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("input"));

        let input = vec![1.0f32; 256];
        let mut tiny_output = vec![0.0f32; 0];
        let err = gemv_q2k(&[block], &input, &mut tiny_output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("output"));

        let mut output = vec![0.0f32; 2];
        let err = gemv_q2k(&[block], &input, &mut output, 2, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// K-15 (a)/(b) apply to Q2_K too even without a dedicated fused
    /// kernel: pin the shared dequantize-then-dot driver against the
    /// ggml-exact reference decoder on non-uniform, distinct-per-element
    /// data across several shapes.
    #[test]
    fn gemv_q2k_matches_reference_dequant_dot_nonuniform() {
        for (n_rows, blocks_per_row) in [(1usize, 1usize), (3, 2), (5, 1), (2, 3)] {
            let in_features = blocks_per_row * 256;
            let raw: Vec<f32> = (0..n_rows * in_features)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.031).sin() * (1.0 + (i % 71) as f32 * 0.035) - 0.15
                })
                .collect();
            let blocks = BlockQ2K::quantize(&raw).expect("quantize q2k");
            let input: Vec<f32> = (0..in_features)
                .map(|i| ((i as f32 * 0.019).cos()) * 1.6 - 0.35)
                .collect();

            let mut got = vec![0.0f32; n_rows];
            gemv_q2k(&blocks, &input, &mut got, n_rows, in_features).expect("gemv_q2k ok");

            for row in 0..n_rows {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                let mut buf = vec![0.0f32; row_blocks.len() * 256];
                BlockQ2K::dequant(row_blocks, &mut buf).expect("dequant ok");
                let expected: f32 = buf.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                let tol = 1e-4 * expected.abs().max(1.0);
                assert!(
                    (got[row] - expected).abs() <= tol,
                    "n_rows={n_rows} blocks_per_row={blocks_per_row} row={row}: \
                     got={}, expected={}",
                    got[row],
                    expected
                );
            }
        }
    }
}
