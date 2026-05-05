//! FP8 GEMV reference kernels (E4M3FN and E5M2).
//!
//! Computes: `output[row] = sum_k( dequant(block[row * blocks_per_col + b][j]) × input[b*QK_FP8 + j] )`
//! where k = b*QK_FP8 + j iterates over the full k dimension.
//!
//! The weight matrix is stored in row-major order: row `r` starts at
//! `blocks[r * blocks_per_row]` where `blocks_per_row = k / QK_FP8`.
//!
//! These are pure scalar Rust correctness-reference implementations — no SIMD,
//! no unsafe. SIMD specializations are a follow-on Slice.

use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8, fp8_e4m3_decode, fp8_e5m2_decode};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// E4M3FN GEMV
// ---------------------------------------------------------------------------

/// Scalar GEMV for FP8 E4M3FN-quantized weight matrix.
///
/// Computes `output[row] = dot(weight_row, input)` for each row.
///
/// - `blocks`: Row-major weight blocks. Row `r` starts at `r * (k / QK_FP8)`.
/// - `input`: FP32 input vector of length `k`.
/// - `output`: FP32 output vector of length `n_rows`.
/// - `n_rows`: Number of output rows.
/// - `k`: Inner dimension (must be divisible by `QK_FP8 = 32`).
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
/// - [`KernelError::DimensionMismatch`] if `input.len() < k` or
///   `blocks.len() < n_rows * (k / QK_FP8)`.
/// - [`KernelError::BufferTooSmall`] if `output.len() < n_rows`.
pub fn gemv_fp8_e4m3(
    blocks: &[BlockFP8E4M3],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    for row in 0..n_rows {
        let mut acc = 0.0_f32;
        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let input_base = bi * QK_FP8;

            let mut block_dot = 0.0_f32;
            for i in 0..QK_FP8 {
                block_dot += fp8_e4m3_decode(block.qs[i]) * input[input_base + i];
            }
            acc += d * block_dot;
        }
        output[row] = acc;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// E5M2 GEMV
// ---------------------------------------------------------------------------

/// Scalar GEMV for FP8 E5M2-quantized weight matrix.
///
/// Computes `output[row] = dot(weight_row, input)` for each row.
///
/// - `blocks`: Row-major weight blocks. Row `r` starts at `r * (k / QK_FP8)`.
/// - `input`: FP32 input vector of length `k`.
/// - `output`: FP32 output vector of length `n_rows`.
/// - `n_rows`: Number of output rows.
/// - `k`: Inner dimension (must be divisible by `QK_FP8 = 32`).
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
/// - [`KernelError::DimensionMismatch`] if `input.len() < k` or
///   `blocks.len() < n_rows * (k / QK_FP8)`.
/// - [`KernelError::BufferTooSmall`] if `output.len() < n_rows`.
pub fn gemv_fp8_e5m2(
    blocks: &[BlockFP8E5M2],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    for row in 0..n_rows {
        let mut acc = 0.0_f32;
        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let input_base = bi * QK_FP8;

            let mut block_dot = 0.0_f32;
            for i in 0..QK_FP8 {
                block_dot += fp8_e5m2_decode(block.qs[i]) * input[input_base + i];
            }
            acc += d * block_dot;
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
    use oxibonsai_core::fp8_e4m3_encode;

    fn make_e4m3_block(scale: f32, qs: [u8; 32]) -> BlockFP8E4M3 {
        BlockFP8E4M3 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    fn make_e5m2_block(scale: f32, qs: [u8; 32]) -> BlockFP8E5M2 {
        BlockFP8E5M2 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    // --- E4M3 GEMV tests ---

    /// 1 row, k=32; all weights decode to 1.0 (qs=0x38, d=1.0), input=all-1.0.
    /// Expected output[0] = 32.0.
    #[test]
    fn gemv_e4m3_identity_one_row() {
        // qs=0x38: exp=7, man=0 → 2^(7-7) × 1.0 = 1.0
        let blocks = vec![make_e4m3_block(1.0, [0x38u8; 32])];
        let input = vec![1.0_f32; 32];
        let mut output = vec![0.0_f32; 1];
        gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32).expect("gemv should succeed");
        // Each weight decodes to ~1.0 (some tiny fp16/fp8 rounding)
        assert!(
            (output[0] - 32.0).abs() < 0.5,
            "expected ~32.0, got {}",
            output[0]
        );
    }

    /// 2 rows, k=32; row 0 scale=1.0, row 1 scale=2.0, input=all-1.0.
    #[test]
    fn gemv_e4m3_two_rows_different_scales() {
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]), // row 0: 32 × 1.0 × 1.0
            make_e4m3_block(2.0, [0x38u8; 32]), // row 1: 32 × 1.0 × 2.0
        ];
        let input = vec![1.0_f32; 32];
        let mut output = vec![0.0_f32; 2];
        gemv_fp8_e4m3(&blocks, &input, &mut output, 2, 32).expect("gemv should succeed");
        assert!(
            (output[0] - 32.0).abs() < 0.5,
            "row0: expected ~32.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 64.0).abs() < 1.0,
            "row1: expected ~64.0, got {}",
            output[1]
        );
    }

    /// k=64 (two blocks per row), 1 row, all weights=1.0, input=all-1.0 → output=64.0.
    #[test]
    fn gemv_e4m3_two_blocks_per_row() {
        // k=64 → 2 blocks per row
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]),
            make_e4m3_block(1.0, [0x38u8; 32]),
        ];
        let input = vec![1.0_f32; 64];
        let mut output = vec![0.0_f32; 1];
        gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 64).expect("gemv should succeed");
        assert!(
            (output[0] - 64.0).abs() < 1.0,
            "expected ~64.0, got {}",
            output[0]
        );
    }

    /// All-zero input → output all zero.
    #[test]
    fn gemv_e4m3_all_zeros_input() {
        let blocks = vec![make_e4m3_block(1.0, [0x38u8; 32])];
        let input = vec![0.0_f32; 32];
        let mut output = vec![99.0_f32; 1];
        gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32).expect("gemv should succeed");
        assert!(
            output[0].abs() < 1e-6,
            "all-zero input → output should be 0.0, got {}",
            output[0]
        );
    }

    /// All-zero weights (qs=0x00 = E4M3 +0.0) → output all zero.
    #[test]
    fn gemv_e4m3_all_zeros_weights() {
        let blocks = vec![make_e4m3_block(1.0, [0x00u8; 32])];
        let input = vec![1.0_f32; 32];
        let mut output = vec![99.0_f32; 1];
        gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32).expect("gemv should succeed");
        assert!(
            output[0].abs() < 1e-6,
            "all-zero weights → output should be 0.0, got {}",
            output[0]
        );
    }

    /// k=31 (not a multiple of 32) → NotBlockAligned error.
    #[test]
    fn gemv_e4m3_not_block_aligned() {
        let blocks = vec![make_e4m3_block(1.0, [0x38u8; 32])];
        let input = vec![1.0_f32; 31];
        let mut output = vec![0.0_f32; 1];
        let result = gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 31);
        assert!(
            matches!(result, Err(KernelError::NotBlockAligned { .. })),
            "expected NotBlockAligned, got {result:?}"
        );
    }

    /// n_rows=2, k=32 → needs 2 blocks; supply 1 → DimensionMismatch.
    #[test]
    fn gemv_e4m3_dimension_mismatch_blocks() {
        let blocks = vec![make_e4m3_block(1.0, [0x38u8; 32])];
        let input = vec![1.0_f32; 32];
        let mut output = vec![0.0_f32; 2];
        let result = gemv_fp8_e4m3(&blocks, &input, &mut output, 2, 32);
        assert!(
            matches!(result, Err(KernelError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    /// Output buffer too small → BufferTooSmall.
    #[test]
    fn gemv_e4m3_output_buffer_too_small() {
        let blocks = vec![make_e4m3_block(1.0, [0x38u8; 32])];
        let input = vec![1.0_f32; 32];
        let mut output = vec![];
        let result = gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32);
        assert!(
            matches!(result, Err(KernelError::BufferTooSmall { .. })),
            "expected BufferTooSmall, got {result:?}"
        );
    }

    /// Unit input e_1 (only first element = 1.0) → output = dequant(weight[0,0]).
    #[test]
    fn gemv_e4m3_unit_input() {
        // Encode 5.0 into E4M3, then embed in a block with scale=1.0
        let w = fp8_e4m3_encode(5.0);
        let mut qs = [0x00u8; 32];
        qs[0] = w;
        let blocks = vec![make_e4m3_block(1.0, qs)];
        let mut input = vec![0.0_f32; 32];
        input[0] = 1.0;
        let mut output = vec![0.0_f32; 1];
        gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32).expect("gemv should succeed");
        let expected = fp8_e4m3_decode(w); // exact round-trip value
        assert!(
            (output[0] - expected).abs() < 1e-5,
            "unit input: expected {expected}, got {}",
            output[0]
        );
    }

    // --- E5M2 GEMV tests ---

    /// 1 row, k=32, all weights decode to 1.0 (qs=0x3C, d=1.0), input=all-1.0 → 32.0.
    #[test]
    fn gemv_e5m2_identity_one_row() {
        // 0x3C = 0b00111100: exp=0b01111=15, man=0 → 2^(15-15)=1.0
        let blocks = vec![make_e5m2_block(1.0, [0x3Cu8; 32])];
        let input = vec![1.0_f32; 32];
        let mut output = vec![0.0_f32; 1];
        gemv_fp8_e5m2(&blocks, &input, &mut output, 1, 32).expect("gemv should succeed");
        assert!(
            (output[0] - 32.0).abs() < 0.5,
            "expected ~32.0, got {}",
            output[0]
        );
    }

    /// k=33 → NotBlockAligned error.
    #[test]
    fn gemv_e5m2_not_block_aligned() {
        let blocks = vec![make_e5m2_block(1.0, [0x3Cu8; 32])];
        let input = vec![1.0_f32; 33];
        let mut output = vec![0.0_f32; 1];
        let result = gemv_fp8_e5m2(&blocks, &input, &mut output, 1, 33);
        assert!(
            matches!(result, Err(KernelError::NotBlockAligned { .. })),
            "expected NotBlockAligned, got {result:?}"
        );
    }

    /// n_rows=3, k=32 → needs 3 blocks; supply 2 → DimensionMismatch.
    #[test]
    fn gemv_e5m2_dimension_mismatch() {
        let blocks = vec![
            make_e5m2_block(1.0, [0x3Cu8; 32]),
            make_e5m2_block(1.0, [0x3Cu8; 32]),
        ];
        let input = vec![1.0_f32; 32];
        let mut output = vec![0.0_f32; 3];
        let result = gemv_fp8_e5m2(&blocks, &input, &mut output, 3, 32);
        assert!(
            matches!(result, Err(KernelError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {result:?}"
        );
    }
}
