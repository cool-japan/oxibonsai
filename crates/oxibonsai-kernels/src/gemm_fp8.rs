//! FP8 GEMM reference kernels (E4M3FN and E5M2).
//!
//! Computes `output[batch, row] = dot(weight_row, input[batch])` for all
//! batch rows and weight rows. Each batch column is dispatched as an independent
//! GEMV call, reusing the same weight block array.
//!
//! Layout conventions:
//! - `input`: row-major FP32 matrix \[batch × k\].
//! - `output`: row-major FP32 matrix \[batch × n\_rows\].
//! - `blocks`: weight matrix in row-major order \[n\_rows × blocks\_per\_row\].
//!
//! These are pure scalar Rust correctness-reference implementations — no SIMD,
//! no unsafe. SIMD specializations are a follow-on Slice.

use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};

use crate::error::{KernelError, KernelResult};
use crate::gemv_fp8::{gemv_fp8_e4m3, gemv_fp8_e5m2};

// ---------------------------------------------------------------------------
// E4M3FN GEMM
// ---------------------------------------------------------------------------

/// Scalar GEMM for FP8 E4M3FN-quantized weight matrix.
///
/// Computes `output[b, r] = dot(weight_row[r], input[b])` for all batch
/// rows `b` and weight rows `r`.
///
/// - `blocks`: Weight blocks, row-major \[n\_rows × (k / QK\_FP8)\].
/// - `inputs`: Row-major FP32 input matrix \[batch × k\].
/// - `outputs`: Row-major FP32 output matrix \[batch × n\_rows\].
/// - `n_rows`: Number of weight matrix rows.
/// - `k`: Inner dimension (must be divisible by `QK_FP8 = 32`).
/// - `batch`: Batch/sequence dimension.
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
/// - [`KernelError::DimensionMismatch`] if `inputs.len() < batch * k` or
///   `blocks.len() < n_rows * (k / QK_FP8)`.
/// - [`KernelError::BufferTooSmall`] if `outputs.len() < batch * n_rows`.
pub fn gemm_fp8_e4m3(
    blocks: &[BlockFP8E4M3],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch: usize,
) -> KernelResult<()> {
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if inputs.len() < batch * k {
        return Err(KernelError::DimensionMismatch {
            expected: batch * k,
            got: inputs.len(),
        });
    }
    if outputs.len() < batch * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: batch * n_rows,
            available: outputs.len(),
        });
    }

    // Validate block count upfront so we do not silently get DimensionMismatch
    // inside the inner GEMV calls.
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    for b in 0..batch {
        let input_row = &inputs[b * k..(b + 1) * k];
        let output_row = &mut outputs[b * n_rows..(b + 1) * n_rows];
        gemv_fp8_e4m3(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// E5M2 GEMM
// ---------------------------------------------------------------------------

/// Scalar GEMM for FP8 E5M2-quantized weight matrix.
///
/// Computes `output[b, r] = dot(weight_row[r], input[b])` for all batch
/// rows `b` and weight rows `r`.
///
/// - `blocks`: Weight blocks, row-major \[n\_rows × (k / QK\_FP8)\].
/// - `inputs`: Row-major FP32 input matrix \[batch × k\].
/// - `outputs`: Row-major FP32 output matrix \[batch × n\_rows\].
/// - `n_rows`: Number of weight matrix rows.
/// - `k`: Inner dimension (must be divisible by `QK_FP8 = 32`).
/// - `batch`: Batch/sequence dimension.
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
/// - [`KernelError::DimensionMismatch`] if `inputs.len() < batch * k` or
///   `blocks.len() < n_rows * (k / QK_FP8)`.
/// - [`KernelError::BufferTooSmall`] if `outputs.len() < batch * n_rows`.
pub fn gemm_fp8_e5m2(
    blocks: &[BlockFP8E5M2],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch: usize,
) -> KernelResult<()> {
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if inputs.len() < batch * k {
        return Err(KernelError::DimensionMismatch {
            expected: batch * k,
            got: inputs.len(),
        });
    }
    if outputs.len() < batch * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: batch * n_rows,
            available: outputs.len(),
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

    for b in 0..batch {
        let input_row = &inputs[b * k..(b + 1) * k];
        let output_row = &mut outputs[b * n_rows..(b + 1) * n_rows];
        gemv_fp8_e5m2(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv_fp8::gemv_fp8_e4m3;
    use half::f16;

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

    // --- E4M3 GEMM tests ---

    /// GEMM output matches individual GEMV for each batch row.
    /// batch=3, n_rows=2, k=32.
    #[test]
    fn gemm_e4m3_matches_gemv() {
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]), // row 0
            make_e4m3_block(2.0, [0x38u8; 32]), // row 1
        ];
        let batch = 3;
        let n_rows = 2;
        let k = 32;
        let mut inputs = vec![0.0_f32; batch * k];
        for b in 0..batch {
            for j in 0..k {
                inputs[b * k + j] = (b + 1) as f32 * 0.5;
            }
        }

        let mut gemm_out = vec![0.0_f32; batch * n_rows];
        gemm_fp8_e4m3(&blocks, &inputs, &mut gemm_out, n_rows, k, batch)
            .expect("gemm should succeed");

        // Verify against per-batch GEMV
        for b in 0..batch {
            let input_row = &inputs[b * k..(b + 1) * k];
            let mut gemv_out = vec![0.0_f32; n_rows];
            gemv_fp8_e4m3(&blocks, input_row, &mut gemv_out, n_rows, k)
                .expect("gemv should succeed");
            for r in 0..n_rows {
                let gm = gemm_out[b * n_rows + r];
                let gv = gemv_out[r];
                assert!(
                    (gm - gv).abs() < 1e-4,
                    "batch={b} row={r}: gemm={gm} vs gemv={gv}"
                );
            }
        }
    }

    /// batch=1 GEMM should equal a single GEMV call.
    #[test]
    fn gemm_e4m3_batch_one_equals_gemv() {
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]),
            make_e4m3_block(1.5, [0x38u8; 32]),
            make_e4m3_block(0.5, [0x38u8; 32]),
        ];
        let n_rows = 3;
        let k = 32;
        let inputs: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut gemm_out = vec![0.0_f32; n_rows];
        let mut gemv_out = vec![0.0_f32; n_rows];

        gemm_fp8_e4m3(&blocks, &inputs, &mut gemm_out, n_rows, k, 1)
            .expect("gemm(batch=1) should succeed");
        gemv_fp8_e4m3(&blocks, &inputs, &mut gemv_out, n_rows, k).expect("gemv should succeed");

        for r in 0..n_rows {
            assert!(
                (gemm_out[r] - gemv_out[r]).abs() < 1e-4,
                "row={r}: gemm={} vs gemv={}",
                gemm_out[r],
                gemv_out[r]
            );
        }
    }

    /// All-positive weights, all-ones input: every output element = k * scale.
    #[test]
    fn gemm_e4m3_all_positive_weights() {
        let n_rows = 4;
        let k = 32;
        let batch = 3;
        let blocks = vec![make_e4m3_block(1.0, [0x38u8; 32]); n_rows];
        let inputs = vec![1.0_f32; batch * k];
        let mut outputs = vec![0.0_f32; batch * n_rows];

        gemm_fp8_e4m3(&blocks, &inputs, &mut outputs, n_rows, k, batch)
            .expect("gemm should succeed");

        for b in 0..batch {
            for r in 0..n_rows {
                let v = outputs[b * n_rows + r];
                assert!(
                    (v - 32.0).abs() < 1.0,
                    "batch={b} row={r}: expected ~32.0, got {v}"
                );
            }
        }
    }

    /// k=33 → NotBlockAligned error.
    #[test]
    fn gemm_e4m3_not_block_aligned() {
        let blocks = vec![make_e4m3_block(1.0, [0x38u8; 32])];
        let inputs = vec![1.0_f32; 33];
        let mut outputs = vec![0.0_f32; 1];
        let result = gemm_fp8_e4m3(&blocks, &inputs, &mut outputs, 1, 33, 1);
        assert!(
            matches!(result, Err(KernelError::NotBlockAligned { .. })),
            "expected NotBlockAligned, got {result:?}"
        );
    }

    /// outputs buffer too small → BufferTooSmall error.
    #[test]
    fn gemm_e4m3_output_buffer_too_small() {
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]),
            make_e4m3_block(1.0, [0x38u8; 32]),
        ];
        let inputs = vec![1.0_f32; 64]; // batch=2, k=32
        let mut outputs = vec![0.0_f32; 3]; // need batch*n_rows = 2*2=4, provide only 3
        let result = gemm_fp8_e4m3(&blocks, &inputs, &mut outputs, 2, 32, 2);
        assert!(
            matches!(result, Err(KernelError::BufferTooSmall { .. })),
            "expected BufferTooSmall, got {result:?}"
        );
    }

    /// Wrong block count for given n_rows → DimensionMismatch.
    #[test]
    fn gemm_e4m3_dimension_mismatch_blocks() {
        // n_rows=3, k=32 → need 3 blocks; supply 2
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]),
            make_e4m3_block(1.0, [0x38u8; 32]),
        ];
        let inputs = vec![1.0_f32; 32];
        let mut outputs = vec![0.0_f32; 3];
        let result = gemm_fp8_e4m3(&blocks, &inputs, &mut outputs, 3, 32, 1);
        assert!(
            matches!(result, Err(KernelError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    /// Two blocks per row (k=64), batch=2.
    #[test]
    fn gemm_e4m3_two_blocks_per_row_batched() {
        let n_rows = 2;
        let k = 64;
        let batch = 2;
        // Each row: two blocks of scale=1.0, all-1.0 weights
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]), // row 0 block 0
            make_e4m3_block(1.0, [0x38u8; 32]), // row 0 block 1
            make_e4m3_block(1.0, [0x38u8; 32]), // row 1 block 0
            make_e4m3_block(1.0, [0x38u8; 32]), // row 1 block 1
        ];
        let inputs = vec![1.0_f32; batch * k];
        let mut outputs = vec![0.0_f32; batch * n_rows];

        gemm_fp8_e4m3(&blocks, &inputs, &mut outputs, n_rows, k, batch)
            .expect("gemm should succeed");

        for b in 0..batch {
            for r in 0..n_rows {
                let v = outputs[b * n_rows + r];
                // 64 weights × 1.0 × 1.0 ≈ 64.0
                assert!(
                    (v - 64.0).abs() < 1.5,
                    "batch={b} row={r}: expected ~64.0, got {v}"
                );
            }
        }
    }

    // --- E5M2 GEMM tests ---

    /// E5M2 GEMM batch=2 n_rows=2 k=32 → each element ~32.0.
    #[test]
    fn gemm_e5m2_basic() {
        // 0x3C = E5M2 1.0 (exp=15, man=0)
        let blocks = vec![
            make_e5m2_block(1.0, [0x3Cu8; 32]),
            make_e5m2_block(1.0, [0x3Cu8; 32]),
        ];
        let batch = 2;
        let n_rows = 2;
        let k = 32;
        let inputs = vec![1.0_f32; batch * k];
        let mut outputs = vec![0.0_f32; batch * n_rows];

        gemm_fp8_e5m2(&blocks, &inputs, &mut outputs, n_rows, k, batch)
            .expect("e5m2 gemm should succeed");

        for b in 0..batch {
            for r in 0..n_rows {
                let v = outputs[b * n_rows + r];
                assert!(
                    (v - 32.0).abs() < 1.0,
                    "batch={b} row={r}: expected ~32.0, got {v}"
                );
            }
        }
    }

    /// E5M2 k=33 → NotBlockAligned.
    #[test]
    fn gemm_e5m2_not_block_aligned() {
        let blocks = vec![make_e5m2_block(1.0, [0x3Cu8; 32])];
        let inputs = vec![1.0_f32; 33];
        let mut outputs = vec![0.0_f32; 1];
        let result = gemm_fp8_e5m2(&blocks, &inputs, &mut outputs, 1, 33, 1);
        assert!(
            matches!(result, Err(KernelError::NotBlockAligned { .. })),
            "expected NotBlockAligned, got {result:?}"
        );
    }
}
