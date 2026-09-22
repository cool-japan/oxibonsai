//! Reference (naive) GEMV kernels for ternary TQ2\_0\_g128 and TQ2\_0 formats.
//!
//! Computes `output = weight_matrix @ input_vector` where the weight matrix
//! is stored in ternary-quantized format. Each block contributes a scaled
//! dot product: `block_contribution = scale * dot(ternary_weights, input_slice)`.
//!
//! Decode uses the single shared [`oxibonsai_core::ternary_code_to_i8`]
//! table (K-01): every NEON/AVX2/AVX-512 ternary kernel in this crate, plus
//! `BlockTQ2_0_g128::dequant` / `ternary_decode` and `BlockTQ2_0::dequant`
//! in `oxibonsai-core`, route through that same table, so this format's
//! decode cannot drift across the two crates or across kernel tiers.
//! `tests/ternary_cross_tier.rs` asserts the whole matrix agrees.

use oxibonsai_core::{ternary_code_to_i8, BlockTQ2_0, BlockTQ2_0_g128, QK_TQ2_0, QK_TQ2_0_G128};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Decode a single 2-bit ternary code at `lane` (0..4) in `byte` to f32.
///
/// Code map: `0b00→-1.0`, `0b01→0.0`, `0b10→+1.0`, `0b11→0.0` (reserved),
/// via the shared [`ternary_code_to_i8`] table (K-01).
#[inline]
fn decode_weight_f32(byte: u8, lane: usize) -> f32 {
    ternary_code_to_i8(byte >> (lane * 2)) as f32
}

// ---------------------------------------------------------------------------
// TQ2_0_g128 — 128 weights per block, 32 qs bytes
// ---------------------------------------------------------------------------

/// Scalar GEMV for TQ2\_0\_g128-quantized weight matrix.
///
/// Computes `output[row] = dot(weight_row, input)` for each row.
///
/// - `blocks`: Row-major weight blocks. Row `i` starts at `i * (k / 128)`.
/// - `input`: FP32 input vector of length `k`.
/// - `output`: FP32 output vector of length `n_rows`.
/// - `n_rows`: Number of output rows.
/// - `k`: Inner dimension (must be divisible by 128).
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % 128 != 0`.
/// - [`KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
/// - [`KernelError::BufferTooSmall`] if `output` is too short.
pub fn gemv_tq2_0_g128(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::dimension_mismatch("input", k, input.len()));
    }
    if output.len() < n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            n_rows,
            output.len(),
        ));
    }

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    for row in 0..n_rows {
        let mut sum = 0.0_f32;
        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let input_base = bi * QK_TQ2_0_G128;

            let mut block_sum = 0.0_f32;
            // 32 bytes × 4 lanes = 128 weights
            for byte_idx in 0..32 {
                let byte = block.qs[byte_idx];
                for lane in 0..4_usize {
                    let weight = decode_weight_f32(byte, lane);
                    block_sum += weight * input[input_base + byte_idx * 4 + lane];
                }
            }
            sum += d * block_sum;
        }
        output[row] = sum;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TQ2_0 — 256 weights per block, 64 qs bytes
// ---------------------------------------------------------------------------

/// Scalar GEMV for TQ2\_0-quantized weight matrix.
///
/// Computes `output[row] = dot(weight_row, input)` for each row.
///
/// - `blocks`: Row-major weight blocks. Row `i` starts at `i * (k / 256)`.
/// - `input`: FP32 input vector of length `k`.
/// - `output`: FP32 output vector of length `n_rows`.
/// - `n_rows`: Number of output rows.
/// - `k`: Inner dimension (must be divisible by 256).
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % 256 != 0`.
/// - [`KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
/// - [`KernelError::BufferTooSmall`] if `output` is too short.
pub fn gemv_tq2_0(
    blocks: &[BlockTQ2_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_TQ2_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0,
        });
    }
    if input.len() < k {
        return Err(KernelError::dimension_mismatch("input", k, input.len()));
    }
    if output.len() < n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            n_rows,
            output.len(),
        ));
    }

    let blocks_per_row = k / QK_TQ2_0;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    for row in 0..n_rows {
        let mut sum = 0.0_f32;
        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let input_base = bi * QK_TQ2_0;

            let mut block_sum = 0.0_f32;
            // 64 bytes × 4 lanes = 256 weights
            for byte_idx in 0..64 {
                let byte = block.qs[byte_idx];
                for lane in 0..4_usize {
                    let weight = decode_weight_f32(byte, lane);
                    block_sum += weight * input[input_base + byte_idx * 4 + lane];
                }
            }
            sum += d * block_sum;
        }
        output[row] = sum;
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

    fn make_g128_block(scale: f32, qs: [u8; 32]) -> BlockTQ2_0_g128 {
        BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    fn make_g256_block(scale: f32, qs: [u8; 64]) -> BlockTQ2_0 {
        BlockTQ2_0 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    // --- TQ2_0_g128 GEMV tests ---

    /// 2 rows, k=128; all weights +1 (qs=0xAA, d=1.0), input=[1.0;128]
    /// → output=[128.0, 128.0].
    #[test]
    fn gemv_tq2_0_g128_identity() {
        let blocks = vec![
            make_g128_block(1.0, [0xAA; 32]), // row 0: all +1
            make_g128_block(1.0, [0xAA; 32]), // row 1: all +1
        ];
        let input = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 2];

        gemv_tq2_0_g128(&blocks, &input, &mut output, 2, 128).expect("gemv should succeed");

        assert!(
            (output[0] - 128.0).abs() < 0.5,
            "output[0]: expected 128.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 128.0).abs() < 0.5,
            "output[1]: expected 128.0, got {}",
            output[1]
        );
    }

    /// 2 rows, k=128; all weights -1 (qs=0x00, d=1.0), input=[1.0;128]
    /// → output=[-128.0, -128.0].
    #[test]
    fn gemv_tq2_0_g128_all_negative() {
        let blocks = vec![
            make_g128_block(1.0, [0x00; 32]), // row 0: all -1
            make_g128_block(1.0, [0x00; 32]), // row 1: all -1
        ];
        let input = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 2];

        gemv_tq2_0_g128(&blocks, &input, &mut output, 2, 128).expect("gemv should succeed");

        assert!(
            (output[0] + 128.0).abs() < 0.5,
            "output[0]: expected -128.0, got {}",
            output[0]
        );
        assert!(
            (output[1] + 128.0).abs() < 0.5,
            "output[1]: expected -128.0, got {}",
            output[1]
        );
    }

    /// 1 row, k=128; weights alternate +1, 0, -1, 0 per group of 4 (4-lane byte).
    ///
    /// Encoding per byte LSB-first:
    ///   lane 0 = Pos(0b10), lane 1 = Zero(0b01), lane 2 = Neg(0b00), lane 3 = Zero(0b01)
    ///   byte = 0b01_00_01_10 = 0x46
    ///
    /// Each 4-weight group contributes: (+1×1 + 0×1 + -1×1 + 0×1) = 0.
    /// Total sum for 32 such bytes = 0.
    #[test]
    fn gemv_tq2_0_g128_alternating() {
        // byte: lane0=Pos(10) lane1=Zero(01) lane2=Neg(00) lane3=Zero(01)
        // bits: [7:6]=01 [5:4]=00 [3:2]=01 [1:0]=10 → 0b01000110 = 0x46
        let blocks = vec![make_g128_block(1.0, [0x46; 32])];
        let input = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 1];

        gemv_tq2_0_g128(&blocks, &input, &mut output, 1, 128).expect("gemv should succeed");

        assert!(
            output[0].abs() < 1e-4,
            "alternating: expected 0.0, got {}",
            output[0]
        );
    }

    /// k=100 is not a multiple of 128 → NotBlockAligned error.
    #[test]
    fn gemv_tq2_0_g128_not_block_aligned() {
        let blocks = vec![make_g128_block(1.0, [0xAA; 32])];
        let input = vec![1.0_f32; 100];
        let mut output = vec![0.0_f32; 1];

        let result = gemv_tq2_0_g128(&blocks, &input, &mut output, 1, 100);
        assert!(result.is_err(), "expected NotBlockAligned error");
    }

    /// blocks.len() mismatch → DimensionMismatch error.
    #[test]
    fn gemv_tq2_0_g128_dimension_validation() {
        // n_rows=2, k=128 → needs 2 blocks; supply only 1
        let blocks = vec![make_g128_block(1.0, [0xAA; 32])];
        let input = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 2];

        let result = gemv_tq2_0_g128(&blocks, &input, &mut output, 2, 128);
        assert!(result.is_err(), "expected DimensionMismatch error");
    }

    /// 4 rows, k=256, d=1.0, all weights +1 → output=[256.0;4].
    #[test]
    fn gemv_tq2_0_g128_multiple_rows() {
        // k=256 → blocks_per_row = 2; 4 rows → 8 blocks total
        let blocks = vec![make_g128_block(1.0, [0xAA; 32]); 8];
        let input = vec![1.0_f32; 256];
        let mut output = vec![0.0_f32; 4];

        gemv_tq2_0_g128(&blocks, &input, &mut output, 4, 256).expect("gemv should succeed");

        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 256.0).abs() < 1.0,
                "output[{i}]: expected 256.0, got {v}",
            );
        }
    }

    // --- TQ2_0 GEMV tests ---

    /// 1 row, k=256; all weights +1, input=[1.0;256] → output=[256.0].
    #[test]
    fn gemv_tq2_0_identity() {
        let blocks = vec![make_g256_block(1.0, [0xAA; 64])];
        let input = vec![1.0_f32; 256];
        let mut output = vec![0.0_f32; 1];

        gemv_tq2_0(&blocks, &input, &mut output, 1, 256).expect("gemv should succeed");

        assert!(
            (output[0] - 256.0).abs() < 1.0,
            "expected 256.0, got {}",
            output[0]
        );
    }

    /// k=100 is not a multiple of 256 → NotBlockAligned error.
    #[test]
    fn gemv_tq2_0_not_block_aligned() {
        let blocks = vec![make_g256_block(1.0, [0xAA; 64])];
        let input = vec![1.0_f32; 100];
        let mut output = vec![0.0_f32; 1];

        let result = gemv_tq2_0(&blocks, &input, &mut output, 1, 100);
        assert!(result.is_err(), "expected NotBlockAligned error");
    }
}
