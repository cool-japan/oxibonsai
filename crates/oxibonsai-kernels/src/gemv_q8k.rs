//! Scalar / fused-NEON GEMV kernel for Q8_K quantized weight matrices.
//!
//! Implements `y = W × x` where W is stored as Q8_K blocks.
//! Each super-block covers 256 weights (QK_K = 256).
//!
//! On AArch64 this routes through [`neon_fused::row_dot`] (K-15 item (c));
//! everywhere else it falls back to the generic dequantize-then-dot driver.
//! Q8_K's dequant is a single affine `w[i] = d * qs[i]` per block (one
//! scale, no sub-block scales/mins and no nibble/bit-field packing), so this
//! is the simplest of the three fused kernels — the same widening this
//! crate's Q8_0 NEON kernel already uses
//! (`simd_q_std_neon::decode_q8_0_i8x16`), just over 256 elements instead
//! of 32 and against a single `f32` block scale instead of an `f16` one.

use oxibonsai_core::BlockQ8K;

use crate::error::{KernelError, KernelResult};

/// AArch64 fused Q8_K row-dot kernel (K-15 item (c)).
#[cfg(target_arch = "aarch64")]
mod neon_fused {
    use core::arch::aarch64::{
        float32x4_t, int8x16_t, vaddvq_f32, vcvtq_f32_s32, vdupq_n_f32, vfmaq_f32, vget_high_s16,
        vget_high_s8, vget_low_s16, vget_low_s8, vld1q_f32, vld1q_s8, vmovl_s16, vmovl_s8,
    };
    use oxibonsai_core::BlockQ8K;

    /// Widen 16 packed `i8` weights into 4 `float32x4_t` chunks — identical
    /// in structure to `simd_q_std_neon::decode_q8_0_i8x16`.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn decode_i8x16(bytes: int8x16_t) -> [float32x4_t; 4] {
        let lo8 = vget_low_s8(bytes);
        let hi8 = vget_high_s8(bytes);
        let lo16 = vmovl_s8(lo8);
        let hi16 = vmovl_s8(hi8);
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))),
        ]
    }

    /// Fused Q8_K row dot product: `dot(dequant(row_blocks), input)`
    /// without ever materializing the dequantized row. `d` is a single
    /// `f32` scale per 256-element block (not per-sub-block), so it is
    /// folded in once per block by distributivity:
    /// `sum(d*qs[i]*x[i]) = d*sum(qs[i]*x[i])`.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn row_dot(row_blocks: &[BlockQ8K], input: &[f32]) -> f32 {
        const QK_K: usize = 256;
        let mut total = 0.0f32;

        for (bi, block) in row_blocks.iter().enumerate() {
            let base = bi * QK_K;
            let mut acc = vdupq_n_f32(0.0);
            let qs_ptr = block.qs.as_ptr();

            for chunk in 0..(QK_K / 16) {
                let bytes = vld1q_s8(qs_ptr.add(chunk * 16));
                let f32_chunks = decode_i8x16(bytes);
                for (ci, cf) in f32_chunks.iter().enumerate() {
                    let idx = base + chunk * 16 + ci * 4;
                    let iv = vld1q_f32(input.as_ptr().add(idx));
                    acc = vfmaq_f32(acc, *cf, iv);
                }
            }
            total += block.d * vaddvq_f32(acc);
        }
        total
    }
}

/// Scalar Q8_K GEMV: computes `output = weight_matrix × input`.
///
/// The weight matrix `W` is stored in row-major Q8_K format:
/// row `i` starts at block index `i * blocks_per_row` where
/// `blocks_per_row = in_features / 256`.
///
/// # Parameters
///
/// - `blocks`:      Q8_K-quantized weight blocks in row-major order.
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
pub fn gemv_q8k(
    blocks: &[BlockQ8K],
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

    // AArch64: fused row-dot that never materializes a dequantized row
    // (K-15 item (c)).
    #[cfg(target_arch = "aarch64")]
    {
        crate::parallel::gemv_kquant_row_parallel_fused(
            input,
            output,
            n_rows,
            in_features,
            |row, row_input| {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                // SAFETY: NEON is always available on AArch64.
                Ok(unsafe { neon_fused::row_dot(row_blocks, row_input) })
            },
        )
    }

    // Everywhere else: row-parallel dequantize-then-dot (numerically
    // identical to sequential — see the driver).
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::parallel::gemv_kquant_row_parallel(
            input,
            output,
            n_rows,
            in_features,
            |row, row_buf| {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                BlockQ8K::dequant(row_blocks, row_buf).map_err(KernelError::Core)
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::BlockQ8K;

    fn make_q8k_block(value: f32) -> BlockQ8K {
        let input = vec![value; 256];
        let blocks = BlockQ8K::quantize(&input).expect("quantize ok");
        blocks[0]
    }

    #[test]
    fn gemv_q8k_single_row_uniform() {
        // One row, uniform weight = 1.0, input all 1.0.
        // Expected output ≈ 256.0 (Q8 is very accurate).
        let block = make_q8k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        gemv_q8k(&[block], &input, &mut output, 1, 256).expect("gemv ok");
        assert!(
            (output[0] - 256.0).abs() < 5.0,
            "expected ~256.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemv_q8k_two_rows() {
        let block_pos = make_q8k_block(0.5);
        let block_neg = make_q8k_block(-0.5);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];

        gemv_q8k(&[block_pos, block_neg], &input, &mut output, 2, 256).expect("gemv ok");
        assert!(
            (output[0] - 128.0).abs() < 5.0,
            "row 0: expected ~128, got {}",
            output[0]
        );
        assert!(
            (output[1] + 128.0).abs() < 5.0,
            "row 1: expected ~-128, got {}",
            output[1]
        );
    }

    #[test]
    fn gemv_q8k_not_block_aligned_errors() {
        let block = make_q8k_block(1.0);
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(
            gemv_q8k(&[block], &input, &mut output, 1, 100).is_err(),
            "should error when in_features not multiple of 256"
        );
    }

    #[test]
    fn gemv_q8k_wrong_block_count_errors() {
        let block = make_q8k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];
        assert!(
            gemv_q8k(&[block], &input, &mut output, 2, 256).is_err(),
            "should error on block count mismatch"
        );
    }

    #[test]
    fn gemv_q8k_output_too_small_errors() {
        let block = make_q8k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 0];
        assert!(
            gemv_q8k(&[block], &input, &mut output, 1, 256).is_err(),
            "should error when output buffer is too small"
        );
    }

    /// Named-error migration (K-02): the production entry points now name
    /// which buffer was wrong.
    #[test]
    fn gemv_q8k_errors_name_the_offending_buffer() {
        let block = make_q8k_block(1.0);
        let short_input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        let err = gemv_q8k(&[block], &short_input, &mut output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("input"));

        let input = vec![1.0f32; 256];
        let mut tiny_output = vec![0.0f32; 0];
        let err = gemv_q8k(&[block], &input, &mut tiny_output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("output"));

        let mut output = vec![0.0f32; 2];
        let err = gemv_q8k(&[block], &input, &mut output, 2, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// Reference (unfused) dot: dequantize via `BlockQ8K::dequant`, then a
    /// plain scalar dot — independent of `gemv_q8k`'s dispatch and of
    /// `crate::parallel::dot_f32`'s SIMD reduction.
    fn reference_row_dot(row_blocks: &[BlockQ8K], input: &[f32]) -> f32 {
        let mut buf = vec![0.0f32; row_blocks.len() * 256];
        BlockQ8K::dequant(row_blocks, &mut buf).expect("dequant ok");
        buf.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
    }

    /// K-15 item (c): pin the (possibly fused) production path against the
    /// reference decoder on non-uniform, distinct-per-element data across
    /// several shapes.
    #[test]
    fn gemv_q8k_matches_reference_dequant_dot_nonuniform() {
        // (600, 1) is above every platform-tuned `par_gemv_min_rows()`
        // threshold (max tuned value is 448), so it forces the Rayon
        // parallel branch of `gemv_kquant_row_parallel_fused` — otherwise
        // only the sequential fallback would ever be exercised here.
        for (n_rows, blocks_per_row) in [(1usize, 1usize), (3, 2), (5, 1), (2, 3), (600, 1)] {
            let in_features = blocks_per_row * 256;
            let raw: Vec<f32> = (0..n_rows * in_features)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.029).sin() * (1.0 + (i % 61) as f32 * 0.03) - 0.1
                })
                .collect();
            let blocks = BlockQ8K::quantize(&raw).expect("quantize q8k");
            let input: Vec<f32> = (0..in_features)
                .map(|i| ((i as f32 * 0.013).cos()) * 1.7 - 0.4)
                .collect();

            let mut got = vec![0.0f32; n_rows];
            gemv_q8k(&blocks, &input, &mut got, n_rows, in_features).expect("gemv_q8k ok");

            for row in 0..n_rows {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                let expected = reference_row_dot(row_blocks, &input);
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

    /// Hand-derived golden: `qs[i] = i % 4` (cycling `0,1,2,3`, 64 times
    /// over 256 elements), `d = 2.0`, `bsums` unused by dequant/dot so left
    /// zeroed, `input` all `1.0`. Each 4-cycle sums to `0+1+2+3=6`, repeated
    /// 64 times = `384`; scaled by `d=2.0` gives `768.0` exactly (small
    /// integers, exactly representable in f32).
    #[test]
    fn gemv_q8k_hand_derived_golden() {
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (i % 4) as i8;
        }
        let block = BlockQ8K {
            d: 2.0,
            qs,
            bsums: [0i16; 16],
        };

        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        gemv_q8k(&[block], &input, &mut output, 1, 256).expect("gemv_q8k ok");

        assert!(
            (output[0] - 768.0).abs() < 1e-2,
            "hand-derived golden: expected 768.0, got {}",
            output[0]
        );
    }

    /// Sign-cancellation canary: alternating +3/-3 weights against a
    /// uniform input sum to exactly 0, regardless of scale.
    #[test]
    fn gemv_q8k_alternating_sign_cancels() {
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = if i % 2 == 0 { 3 } else { -3 };
        }
        let block = BlockQ8K {
            d: 1.5,
            qs,
            bsums: [0i16; 16],
        };

        let input = vec![1.0f32; 256];
        let mut output = vec![99.0f32; 1];
        gemv_q8k(&[block], &input, &mut output, 1, 256).expect("gemv_q8k ok");

        assert!(
            output[0].abs() < 1e-4,
            "alternating sign: expected 0, got {}",
            output[0]
        );
    }
}
