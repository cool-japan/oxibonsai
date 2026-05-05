//! Scalar Q8_0 GEMV reference kernel.
//!
//! Computes `output[row] = dot(W_row, input)` for each row of the weight matrix
//! `W`, where weights are stored as Q8_0 blocks (32 weights per block, 34 bytes).
//!
//! This is a pure scalar Rust correctness-reference implementation — no SIMD,
//! no unsafe. The inner loop dequantizes one block at a time into a 32-element
//! stack buffer to keep stack pressure predictable.

use oxibonsai_core::{BlockQ8_0, QK_Q8_0};

use crate::error::{KernelError, KernelResult};

/// Scalar GEMV for Q8_0-quantized weight matrix.
///
/// Computes `output[row] = dot(weight_row, input)` for each row, where the
/// weight matrix is stored row-major as Q8_0 blocks.
///
/// - `blocks`: Row-major weight blocks.  Row `r` occupies
///   `blocks[r * blocks_per_row .. (r+1) * blocks_per_row]`
///   where `blocks_per_row = in_features / QK_Q8_0`.
/// - `input`: FP32 input vector of length `in_features`.
/// - `output`: FP32 output vector of length `n_rows`.
/// - `n_rows`: Number of output rows (= out_features).
/// - `in_features`: Inner dimension (must be divisible by `QK_Q8_0 = 32`).
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `in_features % QK_Q8_0 != 0`.
/// - [`KernelError::DimensionMismatch`] if `blocks.len() < n_rows * blocks_per_row`
///   or `input.len() < in_features`.
/// - [`KernelError::BufferTooSmall`] if `output.len() < n_rows`.
pub fn gemv_q8_0(
    blocks: &[BlockQ8_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
) -> KernelResult<()> {
    if in_features % QK_Q8_0 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: in_features,
            block_size: QK_Q8_0,
        });
    }
    let blocks_per_row = in_features / QK_Q8_0;
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
            let base = b * QK_Q8_0;
            for j in 0..QK_Q8_0 {
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

    fn make_q8_block(scale: f32, qs: [i8; 32]) -> BlockQ8_0 {
        BlockQ8_0 {
            d: f16::from_f32(scale),
            qs,
        }
    }

    /// All qs=0 → weight = 0; output must be 0.
    #[test]
    fn q8_0_all_zero_weights() {
        let block = make_q8_block(1.0, [0i8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![99.0f32; 1];
        gemv_q8_0(&blocks, &input, &mut output, 1, 32).unwrap();
        assert!(
            output[0].abs() < 1e-5,
            "all-zero weights → output 0, got {}",
            output[0]
        );
    }

    /// Single row, all qs=1, scale=1.0, all-1 input → output = 32.0.
    #[test]
    fn q8_0_unit_weights_single_block() {
        let block = make_q8_block(1.0, [1i8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 1];
        gemv_q8_0(&blocks, &input, &mut output, 1, 32).unwrap();
        assert!(
            (output[0] - 32.0).abs() < 1e-4,
            "unit weights: expected 32.0, got {}",
            output[0]
        );
    }

    /// Positive + negative weights mixed: alternating +1 and -1.
    #[test]
    fn q8_0_positive_negative_mix() {
        let mut qs = [0i8; 32];
        for j in 0..32 {
            qs[j] = if j % 2 == 0 { 1 } else { -1 };
        }
        // All-1 input → dot = 16*(+1) + 16*(-1) = 0
        let block = make_q8_block(1.0, qs);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![99.0f32; 1];
        gemv_q8_0(&blocks, &input, &mut output, 1, 32).unwrap();
        assert!(
            output[0].abs() < 1e-4,
            "alternating sign: expected 0, got {}",
            output[0]
        );
    }

    /// Max scale: qs=127, scale=1.0, all-1 input → 32*127 = 4064.
    #[test]
    fn q8_0_max_scale() {
        let block = make_q8_block(1.0, [127i8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 1];
        gemv_q8_0(&blocks, &input, &mut output, 1, 32).unwrap();
        assert!(
            (output[0] - 32.0 * 127.0).abs() < 1.0,
            "max scale: expected {}, got {}",
            32.0 * 127.0,
            output[0]
        );
    }

    /// 3 rows × 64 features (2 blocks per row).
    #[test]
    fn q8_0_multiple_rows() {
        let zero_block = make_q8_block(1.0, [0i8; 32]);
        let unit_block = make_q8_block(1.0, [1i8; 32]);

        // Row 0: [unit, zero] → 32
        // Row 1: [zero, unit] → 32
        // Row 2: [unit, unit] → 64
        let blocks = vec![
            unit_block, zero_block,
            zero_block, unit_block,
            unit_block, unit_block,
        ];
        let input = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 3];
        gemv_q8_0(&blocks, &input, &mut output, 3, 64).unwrap();
        assert!((output[0] - 32.0).abs() < 1e-4, "row0: {}", output[0]);
        assert!((output[1] - 32.0).abs() < 1e-4, "row1: {}", output[1]);
        assert!((output[2] - 64.0).abs() < 1e-4, "row2: {}", output[2]);
    }

    /// in_features not a multiple of 32 → NotBlockAligned error.
    #[test]
    fn q8_0_not_block_aligned() {
        let block = make_q8_block(1.0, [0i8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 31];
        let mut output = vec![0.0f32; 1];
        let result = gemv_q8_0(&blocks, &input, &mut output, 1, 31);
        assert!(
            matches!(result, Err(KernelError::NotBlockAligned { .. })),
            "expected NotBlockAligned, got {result:?}"
        );
    }

    /// Too few blocks → DimensionMismatch error.
    #[test]
    fn q8_0_wrong_block_count() {
        let block = make_q8_block(1.0, [0i8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 2];
        let result = gemv_q8_0(&blocks, &input, &mut output, 2, 32);
        assert!(
            matches!(result, Err(KernelError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    /// Output buffer too small → BufferTooSmall error.
    #[test]
    fn q8_0_output_too_small() {
        let block = make_q8_block(1.0, [0i8; 32]);
        let blocks = vec![block];
        let input = vec![1.0f32; 32];
        let mut output = vec![];
        let result = gemv_q8_0(&blocks, &input, &mut output, 1, 32);
        assert!(
            matches!(result, Err(KernelError::BufferTooSmall { .. })),
            "expected BufferTooSmall, got {result:?}"
        );
    }

    /// Quantize→dequant round-trip, then GEMV vs naive FP32 matmul.
    #[test]
    fn q8_0_gemv_matches_quantized_dequant() {
        use oxibonsai_core::BlockQ8_0 as BQ;
        let raw: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();
        let blocks = BQ::quantize(&raw).unwrap();

        let mut deq = vec![0.0f32; 64];
        BQ::dequant(&blocks, &mut deq).unwrap();

        let input = vec![1.0f32; 64];
        let reference: f32 = deq.iter().sum();

        let mut output = vec![0.0f32; 1];
        gemv_q8_0(&blocks, &input, &mut output, 1, 64).unwrap();
        assert!(
            (output[0] - reference).abs() < 1e-3,
            "GEMV must match dequant+dot: expected {reference}, got {}",
            output[0]
        );
    }
}
