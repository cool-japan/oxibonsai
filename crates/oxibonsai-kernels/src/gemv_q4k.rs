//! Scalar / fused-NEON GEMV kernel for Q4_K quantized weight matrices.
//!
//! Implements `y = W × x` where W is stored as Q4_K blocks.
//! Each super-block covers 256 weights (QK_K = 256).
//!
//! On AArch64 this routes through [`neon_fused::row_dot`], which computes
//! each output row's dot product directly from the quantized bytes without
//! ever materializing a dequantized f32 row (K-15 item (c)) — decoding the
//! super-block's scales/mins once per 64-element group and
//! multiply-accumulating the packed nibbles against the input slice in
//! NEON registers, mirroring `simd_q_std_neon.rs`'s Q4_0 kernel structure.
//! Everywhere else, it falls back to the generic dequantize-then-dot driver
//! ([`crate::parallel::gemv_kquant_row_parallel`]), which K-15 items (a)/(b)
//! already made SIMD-reduced and allocation-free in steady state.

use oxibonsai_core::BlockQ4K;

use crate::error::{KernelError, KernelResult};

/// AArch64 fused Q4_K row-dot kernel (K-15 item (c)).
#[cfg(target_arch = "aarch64")]
mod neon_fused {
    use core::arch::aarch64::{
        float32x4_t, uint8x16_t, vaddq_f32, vaddvq_f32, vandq_u8, vcvtq_f32_u32, vdupq_n_f32,
        vdupq_n_u8, vfmaq_f32, vget_high_u16, vget_high_u8, vget_low_u16, vget_low_u8, vld1q_f32,
        vld1q_u8, vmovl_u16, vmovl_u8, vshrq_n_u8,
    };
    use oxibonsai_core::BlockQ4K;

    /// ggml's `get_scale_min_k4` (`ggml-quants.c:935`), reproduced verbatim.
    ///
    /// Duplicated here (rather than called) because
    /// `oxibonsai_core::quant_k::get_scale_min_k4` is `pub(crate)` to that
    /// crate; `crates/oxibonsai-core/src/quant_k.rs` — pinned against
    /// `tests/kquant_ggml_golden.rs` — is the authoritative version this
    /// must stay byte-for-byte identical to. Unpacks sub-block `j`'s 6-bit
    /// scale and 6-bit min from the 12-byte packed `scales` array.
    #[inline]
    fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    /// Widen 16 packed nibbles (unsigned, 0..=15, **not** centered) into 4
    /// `float32x4_t` chunks of 4 lanes each — the Q4_K analogue of
    /// `simd_q_std_neon::decode_q4_0_nibbles`, minus that function's `-8`
    /// recentering (a Q4_K nibble is scaled and offset by the block's
    /// per-sub-block `d`/`min` rather than being a pre-centered signed
    /// weight, so the raw 0..=15 magnitude is what the caller needs).
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn widen_nibbles_f32x4x4(nibbles: uint8x16_t) -> [float32x4_t; 4] {
        let lo16 = vmovl_u8(vget_low_u8(nibbles));
        let hi16 = vmovl_u8(vget_high_u8(nibbles));
        [
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))),
        ]
    }

    /// For one 64-element group's 32 packed bytes (`qs32`, low nibble of
    /// byte `l` feeds element `l` of the "lo" half, high nibble feeds
    /// element `l` of the "hi" half — ggml's Q4_K nibble order, matching
    /// `BlockQ4K::dequant`'s `(qs[q_off+l] & 0xF)` / `(qs[q_off+l] >> 4)`
    /// exactly), accumulate `sum(nibble * input)` and `sum(input)`
    /// separately for each half. Returns `(dot_lo, sum_lo, dot_hi, sum_hi)`;
    /// the caller applies the per-half scale/min afterward
    /// (`d1*dot_lo - m1*sum_lo`), since `sum(w*x) = d*sum(nibble*x) -
    /// m*sum(x)` for the affine dequant `w = d*nibble - m`.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn group32_dot_sum(
        qs32: &[u8],
        input_lo: &[f32],
        input_hi: &[f32],
    ) -> (f32, f32, f32, f32) {
        let mask_lo = vdupq_n_u8(0x0F);
        let mut dot_lo = vdupq_n_f32(0.0);
        let mut sum_lo = vdupq_n_f32(0.0);
        let mut dot_hi = vdupq_n_f32(0.0);
        let mut sum_hi = vdupq_n_f32(0.0);

        for half in 0..2usize {
            let v = vld1q_u8(qs32.as_ptr().add(half * 16));
            let lo = vandq_u8(v, mask_lo);
            let hi = vshrq_n_u8::<4>(v);
            let lo_chunks = widen_nibbles_f32x4x4(lo);
            let hi_chunks = widen_nibbles_f32x4x4(hi);
            for c in 0..4usize {
                let idx = half * 16 + c * 4;
                let iv_lo = vld1q_f32(input_lo.as_ptr().add(idx));
                dot_lo = vfmaq_f32(dot_lo, lo_chunks[c], iv_lo);
                sum_lo = vaddq_f32(sum_lo, iv_lo);
                let iv_hi = vld1q_f32(input_hi.as_ptr().add(idx));
                dot_hi = vfmaq_f32(dot_hi, hi_chunks[c], iv_hi);
                sum_hi = vaddq_f32(sum_hi, iv_hi);
            }
        }
        (
            vaddvq_f32(dot_lo),
            vaddvq_f32(sum_lo),
            vaddvq_f32(dot_hi),
            vaddvq_f32(sum_hi),
        )
    }

    /// Fused Q4_K row dot product: `dot(dequant(row_blocks), input)`
    /// without ever materializing the dequantized row. Mirrors
    /// `BlockQ4K::dequant`'s loop order and arithmetic exactly (same 4
    /// groups-of-64 per 256-element block, same `get_scale_min_k4` calls,
    /// same nibble split), just fusing the multiply-by-input-and-accumulate
    /// step instead of writing dequantized floats to a buffer.
    ///
    /// # Safety
    /// Requires NEON CPU support (always available on AArch64).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn row_dot(row_blocks: &[BlockQ4K], input: &[f32]) -> f32 {
        const QK_K: usize = 256;
        let mut total = 0.0f32;

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let base = bi * QK_K;

            let mut y = 0usize;
            let mut q_off = 0usize;
            let mut is = 0usize;
            for _ in 0..(QK_K / 64) {
                let (sc1, m1) = get_scale_min_k4(is, &block.scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
                let d1 = d * f32::from(sc1);
                let mn1 = min * f32::from(m1);
                let d2 = d * f32::from(sc2);
                let mn2 = min * f32::from(m2);

                let input_lo = &input[base + y..base + y + 32];
                let input_hi = &input[base + y + 32..base + y + 64];
                let (dot_lo, sum_lo, dot_hi, sum_hi) =
                    group32_dot_sum(&block.qs[q_off..q_off + 32], input_lo, input_hi);

                total += d1.mul_add(dot_lo, -mn1 * sum_lo) + d2.mul_add(dot_hi, -mn2 * sum_hi);

                y += 64;
                q_off += 32;
                is += 2;
            }
        }
        total
    }
}

/// Scalar Q4_K GEMV: computes `output = weight_matrix × input`.
///
/// The weight matrix `W` is stored in row-major Q4_K format:
/// row `i` starts at block index `i * blocks_per_row` where
/// `blocks_per_row = in_features / 256`.
///
/// # Parameters
///
/// - `blocks`:      Q4_K-quantized weight blocks in row-major order.
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
pub fn gemv_q4k(
    blocks: &[BlockQ4K],
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

    // Everywhere else: row-parallel dequantize-then-dot. Each output row is
    // an independent dequantize-then-dot, so the row loop is split across
    // Rayon threads for large `n_rows` (numerically identical to sequential
    // — see the driver).
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::parallel::gemv_kquant_row_parallel(
            input,
            output,
            n_rows,
            in_features,
            |row, row_buf| {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                BlockQ4K::dequant(row_blocks, row_buf).map_err(KernelError::Core)
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
    use oxibonsai_core::BlockQ4K;

    fn make_q4k_block(value: f32) -> BlockQ4K {
        let input = vec![value; 256];
        let blocks = BlockQ4K::quantize(&input).expect("quantize ok");
        blocks[0]
    }

    #[test]
    fn gemv_q4k_single_row_uniform() {
        // One row, uniform weight = 1.0, input all 1.0.
        // Expected output ≈ 256.0 (with quantization error < 10%).
        let block = make_q4k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        gemv_q4k(&[block], &input, &mut output, 1, 256).expect("gemv ok");
        assert!(
            (output[0] - 256.0).abs() < 30.0,
            "expected ~256.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemv_q4k_two_rows() {
        let block_pos = make_q4k_block(0.5);
        let block_neg = make_q4k_block(-0.5);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];

        gemv_q4k(&[block_pos, block_neg], &input, &mut output, 2, 256).expect("gemv ok");
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
    fn gemv_q4k_not_block_aligned_errors() {
        let block = make_q4k_block(1.0);
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(
            gemv_q4k(&[block], &input, &mut output, 1, 100).is_err(),
            "should error when in_features not multiple of 256"
        );
    }

    #[test]
    fn gemv_q4k_wrong_block_count_errors() {
        let block = make_q4k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 2];
        assert!(
            gemv_q4k(&[block], &input, &mut output, 2, 256).is_err(),
            "should error on block count mismatch"
        );
    }

    #[test]
    fn gemv_q4k_output_too_small_errors() {
        let block = make_q4k_block(1.0);
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 0];
        assert!(
            gemv_q4k(&[block], &input, &mut output, 1, 256).is_err(),
            "should error when output buffer is too small"
        );
    }

    /// Named-error migration (K-02): the production entry points now name
    /// which buffer was wrong.
    #[test]
    fn gemv_q4k_errors_name_the_offending_buffer() {
        let block = make_q4k_block(1.0);
        let short_input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        let err = gemv_q4k(&[block], &short_input, &mut output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("input"));

        let input = vec![1.0f32; 256];
        let mut tiny_output = vec![0.0f32; 0];
        let err = gemv_q4k(&[block], &input, &mut tiny_output, 1, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("output"));

        let mut output = vec![0.0f32; 2];
        let err = gemv_q4k(&[block], &input, &mut output, 2, 256).unwrap_err();
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// Reference (unfused) dot: dequantize via the ggml-exact `BlockQ4K`
    /// decoder, then a plain scalar dot — independent of both `gemv_q4k`'s
    /// dispatch and of `crate::parallel::dot_f32`'s SIMD reduction, so this
    /// cannot share a bug with either.
    fn reference_row_dot(row_blocks: &[BlockQ4K], input: &[f32]) -> f32 {
        let mut buf = vec![0.0f32; row_blocks.len() * 256];
        BlockQ4K::dequant(row_blocks, &mut buf).expect("dequant ok");
        buf.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
    }

    /// K-15 item (c) / wave-1 addendum: pin the (possibly fused) production
    /// path against the ggml-exact reference decoder on **non-uniform**
    /// data — every sub-block scale/min and every nibble distinct — across
    /// several shapes (multiple blocks per row, multiple rows), since a
    /// uniform fixture cannot detect a sub-block scale permutation or a
    /// nibble-order bug (the exact bug class `core-gguf-K0` found and this
    /// task's addenda warn against reintroducing on the SIMD path).
    #[test]
    fn gemv_q4k_matches_reference_dequant_dot_nonuniform() {
        // (600, 1) is above every platform-tuned `par_gemv_min_rows()`
        // threshold (max tuned value is 448), so it forces the Rayon
        // parallel branch of `gemv_kquant_row_parallel_fused` — otherwise
        // only the sequential fallback would ever be exercised here.
        for (n_rows, blocks_per_row) in [(1usize, 1usize), (3, 2), (5, 1), (2, 3), (600, 1)] {
            let in_features = blocks_per_row * 256;
            // Distinctive per-element signal so quantize() assigns visibly
            // different sub-block scales/mins across the 8 sub-blocks and
            // across rows/blocks.
            let raw: Vec<f32> = (0..n_rows * in_features)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.037).sin() * (1.0 + (i % 97) as f32 * 0.05) - 0.3
                })
                .collect();
            let blocks = BlockQ4K::quantize(&raw).expect("quantize q4k");
            let input: Vec<f32> = (0..in_features)
                .map(|i| ((i as f32 * 0.021).cos()) * 2.0 - 0.5)
                .collect();

            let mut got = vec![0.0f32; n_rows];
            gemv_q4k(&blocks, &input, &mut got, n_rows, in_features).expect("gemv_q4k ok");

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

    /// Hand-derived golden (wave-1/wave-1.5 addenda): a byte layout built
    /// directly (not round-tripped through `BlockQ4K::quantize`), with the
    /// expected value traced by hand from the ggml formula rather than
    /// compared only against this crate's own dequantizer — one
    /// defense-in-depth layer the addenda ask for, since Q6_K's stale-layout
    /// bug slipped through a cos/rel tolerance for months. See **Limits**
    /// below for what this particular fixture does and does not cover.
    ///
    /// Construction: `scales = [1,1,1,1, 0,0,0,0, 1,1,1,1]` makes
    /// `get_scale_min_k4(j)` return `(sc=1, m=0)` for every `j` in `0..8`
    /// (verified by hand against `ggml-quants.c:935`'s formula: for `j<4`,
    /// `(q[j]&63, q[j+4]&63)` = `(1, 0)`; for `j>=4`, since `q[j]` and
    /// `q[j-4]` are all `1` (top two bits `0`), the formula collapses to
    /// `(q[j+4]&0xF, q[j+4]>>4)` = `(1, 0)` for `q[8..12] = [1,1,1,1]`).
    /// With every sub-block scale 1 and min 0, `d = 1.0`, every output
    /// element is exactly its nibble value. `qs[i] = (i%16) | ((15-i%16)<<4)`
    /// gives low nibble `i%16` (elements `0..32`, `64..96`, ...) and high
    /// nibble `15-i%16` (elements `32..64`, `96..128`, ...) within each
    /// 64-wide group. With `input` all `1.0`, each 32-wide half sums its
    /// nibbles: `sum(0..16) + sum(0..16) = 2*120 = 240` for both the lo and
    /// hi halves (by symmetry, `15-j` over `j=0..16` also sums to `120`), so
    /// each of the four 64-wide groups contributes `240 + 240 = 480`, and
    /// the one-block row totals `4 * 480 = 1920.0` exactly (every operand is
    /// a small integer exactly representable in f32, so this is a bit-exact
    /// expectation, not a tolerance-bounded one).
    ///
    /// **Limits (what this golden does NOT catch).** The fixture is uniform by
    /// construction: `input` is all `1.0` and every sub-block scale is `1`
    /// with min `0`, so the expected `1920.0` is invariant under (a) a lo/hi
    /// nibble swap and (b) any permutation of the 8 sub-block scales — the
    /// two bug classes the wave-1/1.5 addenda single out. What it does pin is
    /// the `get_scale_min_k4` derivation and the total magnitude; nothing
    /// positional. Positional coverage lives in
    /// `tests/verifier_kquant_fused_random.rs`
    /// (`fused_q4k_matches_independent_ggml_reference_on_random_bytes`), which
    /// drives random `qs`/`scales` bytes against an independent
    /// transliteration of `ggml-quants.c` and is green. PREFERRED FOLLOW-UP:
    /// to make this golden earn the stronger claim, give it a non-uniform
    /// `input` vector and distinct per-sub-block scales, then re-derive the
    /// expected value by hand from the ggml formula.
    #[test]
    fn gemv_q4k_hand_derived_golden() {
        use half::f16;

        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            let lo = (i % 16) as u8;
            let hi = 15 - lo;
            *q = lo | (hi << 4);
        }
        let block = BlockQ4K {
            d: f16::from_f32(1.0),
            dmin: f16::from_f32(0.0),
            scales: [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            qs,
        };

        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];
        gemv_q4k(&[block], &input, &mut output, 1, 256).expect("gemv_q4k ok");

        assert!(
            (output[0] - 1920.0).abs() < 1e-2,
            "hand-derived golden: expected 1920.0, got {}",
            output[0]
        );
    }
}
