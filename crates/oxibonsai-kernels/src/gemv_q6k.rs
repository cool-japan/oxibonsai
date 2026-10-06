//! Scalar / fused-NEON GEMV kernel for Q6_K quantized weight matrices.
//!
//! Implements `y = W × x` where W is stored as Q6_K blocks.
//! Each super-block covers 256 weights (QK_K = 256).
//!
//! On AArch64 this routes through `neon_fused::row_dot` (K-15 item (c)),
//! never materializing a dequantized f32 row; everywhere else it falls back
//! to the generic dequantize-then-dot driver.

use oxibonsai_core::BlockQ6K;

use crate::error::{KernelError, KernelResult};

/// AArch64 fused Q6_K row-dot kernel (K-15 item (c)).
///
/// Q6_K packs each output element from **three** sources — a low nibble
/// from `ql`, a 2-bit field from `qh`, and one of 16 per-16-element `i8`
/// sub-scales selected by `l / 16` — across four interleaved "arms" per
/// 32-element group (`ggml-quants.c`'s `dequantize_row_q6_K`, mirrored by
/// `BlockQ6K::dequant`). The audit singled this format out by
/// name (its stale-layout bug hid at rel=0.0119 under a cos tolerance), so
/// this kernel is restructured — not re-derived — directly from
/// `BlockQ6K::dequant`'s loop: same `is = l/16` split, same four bit
/// extractions, same sub-scale indices (`sc_off+is+{0,2,4,6}`), just fusing
/// the multiply-by-input-and-accumulate step instead of writing to a
/// dequantized buffer.
#[cfg(target_arch = "aarch64")]
mod neon_fused {
    use core::arch::aarch64::{
        float32x4_t, uint8x16_t, vaddvq_f32, vandq_u8, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s32,
        vdupq_n_u8, vfmaq_f32, vget_high_u16, vget_high_u8, vget_low_u16, vget_low_u8, vld1q_f32,
        vld1q_u8, vmovl_u16, vmovl_u8, vorrq_u8, vreinterpretq_s32_u32, vshlq_n_u8, vshrq_n_u8,
        vsubq_s32,
    };
    use oxibonsai_core::BlockQ6K;

    /// Widen 16 raw 6-bit codes (`0..=63`, stored one per byte) into 4
    /// `float32x4_t` chunks of the **signed** value `code - 32` (ggml's
    /// `q6 - 32` recentering), matching `BlockQ6K::dequant`'s
    /// `(... as i32) - 32` exactly.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn widen_code_minus32(raw: uint8x16_t) -> [float32x4_t; 4] {
        let lo16 = vmovl_u8(vget_low_u8(raw));
        let hi16 = vmovl_u8(vget_high_u8(raw));
        let thirty_two = vdupq_n_s32(32);
        [
            vcvtq_f32_s32(vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16))),
                thirty_two,
            )),
            vcvtq_f32_s32(vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo16))),
                thirty_two,
            )),
            vcvtq_f32_s32(vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16))),
                thirty_two,
            )),
            vcvtq_f32_s32(vsubq_s32(
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi16))),
                thirty_two,
            )),
        ]
    }

    /// Dot 4 `float32x4_t` chunks (16 lanes total) against 16 consecutive
    /// f32 input elements, horizontally summed to one scalar.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dot16(chunks: [float32x4_t; 4], input16: &[f32]) -> f32 {
        let mut acc = vdupq_n_f32(0.0);
        for (c, chunk) in chunks.iter().enumerate() {
            let iv = vld1q_f32(input16.as_ptr().add(c * 4));
            acc = vfmaq_f32(acc, *chunk, iv);
        }
        vaddvq_f32(acc)
    }

    /// For one 16-element `is`-half of a 32-element group, decode all four
    /// arms (`q1..q4`, ggml's naming) and dot each against its matching
    /// 16-wide input slice. The caller applies the per-arm, per-half
    /// sub-scale and the block's `d` afterward.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn half16_dots(
        ql_lo16: &[u8],
        ql_hi16: &[u8],
        qh16: &[u8],
        in_q1: &[f32],
        in_q2: &[f32],
        in_q3: &[f32],
        in_q4: &[f32],
    ) -> (f32, f32, f32, f32) {
        let mask2 = vdupq_n_u8(0x03);
        let mask4 = vdupq_n_u8(0x0F);

        let ql_lo = vld1q_u8(ql_lo16.as_ptr());
        let ql_hi = vld1q_u8(ql_hi16.as_ptr());
        let qh = vld1q_u8(qh16.as_ptr());

        // q1 = (ql_lo & 0xF) | ((qh & 3) << 4)
        let q1_raw = vorrq_u8(vandq_u8(ql_lo, mask4), vshlq_n_u8::<4>(vandq_u8(qh, mask2)));
        // q2 = (ql_hi & 0xF) | (((qh >> 2) & 3) << 4)
        let q2_raw = vorrq_u8(
            vandq_u8(ql_hi, mask4),
            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh), mask2)),
        );
        // q3 = (ql_lo >> 4) | (((qh >> 4) & 3) << 4)
        let q3_raw = vorrq_u8(
            vshrq_n_u8::<4>(ql_lo),
            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh), mask2)),
        );
        // q4 = (ql_hi >> 4) | (((qh >> 6) & 3) << 4)
        let q4_raw = vorrq_u8(
            vshrq_n_u8::<4>(ql_hi),
            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<6>(qh), mask2)),
        );

        (
            dot16(widen_code_minus32(q1_raw), in_q1),
            dot16(widen_code_minus32(q2_raw), in_q2),
            dot16(widen_code_minus32(q3_raw), in_q3),
            dot16(widen_code_minus32(q4_raw), in_q4),
        )
    }

    /// Fused Q6_K row dot product: `dot(dequant(row_blocks), input)`
    /// without ever materializing the dequantized row.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn row_dot(row_blocks: &[BlockQ6K], input: &[f32]) -> f32 {
        const QK_K: usize = 256;
        let mut total = 0.0f32;

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = bi * QK_K;
            let mut y = 0usize;
            let mut ql_off = 0usize;
            let mut qh_off = 0usize;
            let mut sc_off = 0usize;
            let mut block_total = 0.0f32;

            for _n in 0..(QK_K / 128) {
                for (is, l_start) in [(0usize, 0usize), (1usize, 16usize)] {
                    let ql_lo16 = &block.ql[ql_off + l_start..ql_off + l_start + 16];
                    let ql_hi16 = &block.ql[ql_off + 32 + l_start..ql_off + 32 + l_start + 16];
                    let qh16 = &block.qh[qh_off + l_start..qh_off + l_start + 16];
                    let in_q1 = &input[base + y + l_start..base + y + l_start + 16];
                    let in_q2 = &input[base + y + 32 + l_start..base + y + 32 + l_start + 16];
                    let in_q3 = &input[base + y + 64 + l_start..base + y + 64 + l_start + 16];
                    let in_q4 = &input[base + y + 96 + l_start..base + y + 96 + l_start + 16];

                    let (dot1, dot2, dot3, dot4) =
                        half16_dots(ql_lo16, ql_hi16, qh16, in_q1, in_q2, in_q3, in_q4);

                    block_total += f32::from(block.scales[sc_off + is]) * dot1
                        + f32::from(block.scales[sc_off + is + 2]) * dot2
                        + f32::from(block.scales[sc_off + is + 4]) * dot3
                        + f32::from(block.scales[sc_off + is + 6]) * dot4;
                }
                y += 128;
                ql_off += 64;
                qh_off += 32;
                sc_off += 8;
            }
            total += d * block_total;
        }
        total
    }
}

/// Scalar Q6_K GEMV: computes `output = weight_matrix × input`.
///
/// The weight matrix `W` is stored in row-major Q6_K format:
/// row `i` starts at block index `i * blocks_per_row` where
/// `blocks_per_row = in_features / 256`.
///
/// # Parameters
///
/// - `blocks`:      Q6_K-quantized weight blocks in row-major order.
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
pub fn gemv_q6k(
    blocks: &[BlockQ6K],
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
                BlockQ6K::dequant(row_blocks, row_buf).map_err(KernelError::Core)
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
    use oxibonsai_core::BlockQ6K;

    /// Helper: build a single Q6_K block by quantizing a uniform-value slice.
    fn make_q6k_block(value: f32) -> BlockQ6K {
        let input = vec![value; 256];
        let blocks = BlockQ6K::quantize(&input).expect("quantize ok");
        blocks[0]
    }

    #[test]
    fn gemv_q6k_single_row_uniform() {
        // One row, uniform weight = 1.0, input all 1.0.
        // Expected output ≈ 256.0 (with quantization error < 5%).
        let block = make_q6k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        gemv_q6k(&[block], &input, &mut output, 1, 256).expect("gemv ok");
        assert!(
            (output[0] - 256.0).abs() < 15.0,
            "expected ~256.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemv_q6k_two_rows() {
        // Two rows: row 0 all +0.5, row 1 all -0.5.
        // Input all 1.0 → row 0 ≈ 128.0, row 1 ≈ -128.0.
        let block_pos = make_q6k_block(0.5);
        let block_neg = make_q6k_block(-0.5);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];

        gemv_q6k(&[block_pos, block_neg], &input, &mut output, 2, 256).expect("gemv ok");
        assert!(
            (output[0] - 128.0).abs() < 10.0,
            "row 0: expected ~128, got {}",
            output[0]
        );
        assert!(
            (output[1] + 128.0).abs() < 10.0,
            "row 1: expected ~-128, got {}",
            output[1]
        );
    }

    #[test]
    fn gemv_q6k_not_block_aligned_errors() {
        let block = make_q6k_block(1.0);
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(
            gemv_q6k(&[block], &input, &mut output, 1, 100).is_err(),
            "should error when in_features not multiple of 256"
        );
    }

    #[test]
    fn gemv_q6k_wrong_block_count_errors() {
        // n_rows=2 needs 2 blocks, but only 1 provided.
        let block = make_q6k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];
        assert!(
            gemv_q6k(&[block], &input, &mut output, 2, 256).is_err(),
            "should error on block count mismatch"
        );
    }

    #[test]
    fn gemv_q6k_output_too_small_errors() {
        let block = make_q6k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 0]; // empty output
        assert!(
            gemv_q6k(&[block], &input, &mut output, 1, 256).is_err(),
            "should error when output buffer is too small"
        );
    }

    /// Named-error migration (K-02): the production entry points now name
    /// which buffer was wrong.
    #[test]
    fn gemv_q6k_errors_name_the_offending_buffer() {
        let block = make_q6k_block(1.0);
        let short_input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        let err = gemv_q6k(&[block], &short_input, &mut output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("input"));

        let input = vec![1.0f32; 256];
        let mut tiny_output = vec![0.0f32; 0];
        let err = gemv_q6k(&[block], &input, &mut tiny_output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("output"));

        let mut output = vec![0.0f32; 2];
        let err = gemv_q6k(&[block], &input, &mut output, 2, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// Reference (unfused) dot: dequantize via the ggml-exact `BlockQ6K`
    /// decoder, then a plain scalar dot — independent of both `gemv_q6k`'s
    /// dispatch and of `crate::parallel::dot_f32`'s SIMD reduction.
    fn reference_row_dot(row_blocks: &[BlockQ6K], input: &[f32]) -> f32 {
        let mut buf = vec![0.0f32; row_blocks.len() * 256];
        BlockQ6K::dequant(row_blocks, &mut buf).expect("dequant ok");
        buf.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
    }

    /// K-15 item (c): pin the (possibly fused) production
    /// path against the ggml-exact reference decoder on non-uniform data —
    /// distinct `ql`/`qh`/per-sub-block `scales` — across several shapes.
    /// Q6_K is the format the addenda name explicitly: its stale-layout bug
    /// (before `core-gguf-K0`) hid at rel=0.0119 under a looser tolerance,
    /// so uniform "all weights equal" fixtures are specifically inadequate
    /// here.
    #[test]
    fn gemv_q6k_matches_reference_dequant_dot_nonuniform() {
        // (600, 1) is above every platform-tuned `par_gemv_min_rows()`
        // threshold (max tuned value is 448), so it forces the Rayon
        // parallel branch of `gemv_kquant_row_parallel_fused` — otherwise
        // only the sequential fallback would ever be exercised here.
        for (n_rows, blocks_per_row) in [(1usize, 1usize), (3, 2), (5, 1), (2, 3), (600, 1)] {
            let in_features = blocks_per_row * 256;
            let raw: Vec<f32> = (0..n_rows * in_features)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.043).sin() * (1.0 + (i % 83) as f32 * 0.04) - 0.2
                })
                .collect();
            let blocks = BlockQ6K::quantize(&raw).expect("quantize q6k");
            let input: Vec<f32> = (0..in_features)
                .map(|i| ((i as f32 * 0.017).cos()) * 1.5 - 0.3)
                .collect();

            let mut got = vec![0.0f32; n_rows];
            gemv_q6k(&blocks, &input, &mut got, n_rows, in_features).expect("gemv_q6k ok");

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

    /// Hand-derived golden:
    /// a byte layout built directly, with the expected value traced by hand
    /// from ggml's four-arm bit extraction rather than compared only
    /// against this crate's own dequantizer.
    ///
    /// Construction: `ql = [0xAB; 128]` (every byte's low nibble `0xB=11`,
    /// high nibble `0xA=10`), `qh = [0xC7; 64]` (`0xC7 = 0b1100_0111`, so
    /// `qh&3=3`, `(qh>>2)&3=1`, `(qh>>4)&3=0`, `(qh>>6)&3=3`), `scales =
    /// [1; 16]`, `d = 1.0`. Per `BlockQ6K::dequant`'s four arms:
    /// `q1 = (0xB | (3<<4)) - 32 = 59 - 32 = 27`,
    /// `q2 = (0xB | (1<<4)) - 32 = 27 - 32 = -5`,
    /// `q3 = (0xA | (0<<4)) - 32 = 10 - 32 = -22`,
    /// `q4 = (0xA | (3<<4)) - 32 = 58 - 32 = 26`.
    /// Each arm covers 32 elements per 128-wide group (two groups per
    /// 256-wide block), all with `d=1`, `scale=1`, so one block totals
    /// `2 * 32 * (27 + (-5) + (-22) + 26) = 2 * 32 * 26 = 1664.0` exactly
    /// (small integers, exactly representable in f32).
    #[test]
    fn gemv_q6k_hand_derived_golden() {
        use half::f16;

        let block = BlockQ6K {
            ql: [0xABu8; 128],
            qh: [0xC7u8; 64],
            scales: [1i8; 16],
            d: f16::from_f32(1.0),
        };

        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        gemv_q6k(&[block], &input, &mut output, 1, 256).expect("gemv_q6k ok");

        assert!(
            (output[0] - 1664.0).abs() < 1e-2,
            "hand-derived golden: expected 1664.0, got {}",
            output[0]
        );
    }
}
