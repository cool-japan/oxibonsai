//! AVX2-accelerated dequant / GEMV / GEMM for the PrismML Bonsai 2 formats:
//! `PQ2_0`, mainline `Q2_0_g64`, and `PTQ1_0`.
//!
//! Structural mirror of [`crate::simd_prism_neon`] — every helper here has a
//! same-named-in-spirit NEON counterpart performing the identical arithmetic
//! (same widen -> multiply -> mask -> multiply -> shift -> narrow order for
//! the `PTQ1_0` trit decode), so the two files can be reviewed line-for-line
//! against each other. This crate has no x86-64 hardware in its validation
//! environment (development machine is Apple Silicon; this module is gated
//! `#[cfg(target_arch = "x86_64")]` and is therefore never even parsed, let
//! alone executed, on that machine — see `simd_avx2.rs`'s pre-existing
//! ternary tier for the same, already-accepted situation in this crate).
//! The tests at the bottom of this file are written to run wherever this
//! module IS compiled (real x86-64 CI or hardware).
//!
//! See [`crate::dequant_prism`]'s module doc for the format backgrounds and
//! the shared `code -> value` map.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::{
    BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, POW3, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64,
};

#[cfg(target_arch = "x86_64")]
use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Decode 2 bytes' worth of 2-bit codes (8 codes) to `__m256` via the
/// arithmetic map `value = (code as i32) - 1`. No reserved-code masking —
/// every 2-bit code is valid for this family (unlike the legacy ternary
/// LUT's AVX2 decode, `decode_2bytes_avx2_to_f32x8` in `simd_avx2.rs`, which
/// this deliberately does not call or share a helper with — see
/// [`crate::dequant_prism`]'s module doc on why `PQ2_0` must never reuse the
/// `TQ2_0_g128` decode).
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn decode_2bytes_avx2_arith_to_f32x8(b0: u8, b1: u8) -> __m256 {
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let mask3 = _mm256_set1_epi32(3);
    let one = _mm256_set1_epi32(1);
    let packed = _mm256_set1_epi32(((b0 as u32) | ((b1 as u32) << 8)) as i32);
    let shifted = _mm256_srlv_epi32(packed, shifts);
    let idx = _mm256_and_si256(shifted, mask3);
    let val_i = _mm256_sub_epi32(idx, one);
    _mm256_cvtepi32_ps(val_i)
}

/// Horizontal sum of 8 f32 lanes.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn hsum8_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps::<1>(v);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

// ---------------------------------------------------------------------------
// PQ2_0 (ggml id 142)
// ---------------------------------------------------------------------------

/// AVX2-accelerated dequantization of `PQ2_0` blocks to FP32.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_pq2_0_avx2(blocks: &[BlockPQ2_0], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_PQ2_0;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let scale = _mm256_set1_ps(block.d.to_f32());
        let base = bi * QK_PQ2_0;
        // 32 bytes / 2 per iteration = 16 iterations of 8 weights each.
        for chunk in 0..16 {
            let val =
                decode_2bytes_avx2_arith_to_f32x8(block.qs[chunk * 2], block.qs[chunk * 2 + 1]);
            let result = _mm256_mul_ps(scale, val);
            _mm256_storeu_ps(output.as_mut_ptr().add(base + chunk * 8), result);
        }
    }
    Ok(())
}

/// AVX2-accelerated GEMV for a `PQ2_0`-quantized weight matrix.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_pq2_0_avx2(
    blocks: &[BlockPQ2_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PQ2_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PQ2_0,
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
    let blocks_per_row = k / QK_PQ2_0;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;
        for (bi, block) in row_blocks.iter().enumerate() {
            let inp_base = bi * QK_PQ2_0;
            let mut acc = _mm256_setzero_ps();
            for chunk in 0..16 {
                let val =
                    decode_2bytes_avx2_arith_to_f32x8(block.qs[chunk * 2], block.qs[chunk * 2 + 1]);
                let inp_vec = _mm256_loadu_ps(input.as_ptr().add(inp_base + chunk * 8));
                acc = _mm256_fmadd_ps(val, inp_vec, acc);
            }
            row_sum += block.d.to_f32() * hsum8_avx2(acc);
        }
        output[row] = row_sum;
    }
    Ok(())
}

/// AVX2-accelerated GEMM for a `PQ2_0`-quantized weight matrix.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_pq2_0_avx2(
    blocks: &[BlockPQ2_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PQ2_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PQ2_0,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::dimension_mismatch("input", m * k, input.len()));
    }
    if output.len() < m * n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            m * n_rows,
            output.len(),
        ));
    }
    for mi in 0..m {
        let input_row = &input[mi * k..];
        let output_row = &mut output[mi * n_rows..(mi + 1) * n_rows];
        gemv_pq2_0_avx2(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Q2_0_g64 (mainline ggml id 42, group 64)
// ---------------------------------------------------------------------------

/// AVX2-accelerated dequantization of `Q2_0_g64` blocks to FP32.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_q2_0_g64_avx2(
    blocks: &[BlockQ2_0G64],
    output: &mut [f32],
) -> KernelResult<()> {
    let needed = blocks.len() * QK_Q2_0_G64;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let scale = _mm256_set1_ps(block.d.to_f32());
        let base = bi * QK_Q2_0_G64;
        // 16 bytes / 2 per iteration = 8 iterations of 8 weights each.
        for chunk in 0..8 {
            let val =
                decode_2bytes_avx2_arith_to_f32x8(block.qs[chunk * 2], block.qs[chunk * 2 + 1]);
            let result = _mm256_mul_ps(scale, val);
            _mm256_storeu_ps(output.as_mut_ptr().add(base + chunk * 8), result);
        }
    }
    Ok(())
}

/// AVX2-accelerated GEMV for a `Q2_0_g64`-quantized weight matrix.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_q2_0_g64_avx2(
    blocks: &[BlockQ2_0G64],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_Q2_0_G64) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_Q2_0_G64,
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
    let blocks_per_row = k / QK_Q2_0_G64;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;
        for (bi, block) in row_blocks.iter().enumerate() {
            let inp_base = bi * QK_Q2_0_G64;
            let mut acc = _mm256_setzero_ps();
            for chunk in 0..8 {
                let val =
                    decode_2bytes_avx2_arith_to_f32x8(block.qs[chunk * 2], block.qs[chunk * 2 + 1]);
                let inp_vec = _mm256_loadu_ps(input.as_ptr().add(inp_base + chunk * 8));
                acc = _mm256_fmadd_ps(val, inp_vec, acc);
            }
            row_sum += block.d.to_f32() * hsum8_avx2(acc);
        }
        output[row] = row_sum;
    }
    Ok(())
}

/// AVX2-accelerated GEMM for a `Q2_0_g64`-quantized weight matrix.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_q2_0_g64_avx2(
    blocks: &[BlockQ2_0G64],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_Q2_0_G64) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_Q2_0_G64,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::dimension_mismatch("input", m * k, input.len()));
    }
    if output.len() < m * n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            m * n_rows,
            output.len(),
        ));
    }
    for mi in 0..m {
        let input_row = &input[mi * k..];
        let output_row = &mut output[mi * n_rows..(mi + 1) * n_rows];
        gemv_q2_0_g64_avx2(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PTQ1_0 (ggml id 143) — vectorized base-3 trit decode
// ---------------------------------------------------------------------------

/// Widen 16 `u8` lanes (held in the low 128 bits of a register) to 16 `u16`
/// lanes, multiply by `pow`, mask to the low byte (== `u8::wrapping_mul`,
/// exactly as in [`crate::simd_prism_neon::trit_of_u8x16`] — the widened
/// product never overflows 16 bits: `255 * 243 = 61965 < 65536`), multiply
/// by 3, shift right 8, narrow back to 16 `u8` lanes.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn trit_of_16_avx2(q_bytes: __m128i, pow: u8) -> __m128i {
    let widened = _mm256_cvtepu8_epi16(q_bytes); // 16 x u16, zero-extended
    let pow16 = _mm256_set1_epi16(pow as i16);
    let mask_ff = _mm256_set1_epi16(0x00FF);
    let three = _mm256_set1_epi16(3);

    let prod = _mm256_mullo_epi16(widened, pow16);
    let masked = _mm256_and_si256(prod, mask_ff);
    let times3 = _mm256_mullo_epi16(masked, three);
    let shifted = _mm256_srli_epi16::<8>(times3); // values 0..2

    let lo = _mm256_extracti128_si256::<0>(shifted);
    let hi = _mm256_extracti128_si256::<1>(shifted);
    _mm_packus_epi16(lo, hi)
}

/// NEON re-implementation counterpart: AVX2 decode of one `PTQ1_0` block's
/// 128 trit codes. Vectorizes both effective stages (`c = 16` over
/// `qs[0..16]`, `c = 8` over `qs[16..24]`, zero-padded to a full 16-byte
/// register so [`trit_of_16_avx2`] can be reused unchanged); the 8-value
/// `qh` tail stays scalar, matching [`crate::simd_prism_neon`].
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn decode_ptq1_0_codes_avx2(block: &BlockPTQ1_0) -> [u8; QK_PTQ1_0] {
    let mut codes = [0u8; QK_PTQ1_0];

    let stage1_src = _mm_loadu_si128(block.qs.as_ptr() as *const __m128i); // bytes 0..16
    for (n, &pow) in POW3.iter().take(5).enumerate() {
        let out16 = trit_of_16_avx2(stage1_src, pow);
        _mm_storeu_si128(codes.as_mut_ptr().add(n * 16) as *mut __m128i, out16);
    }

    // Zero-pad bytes 16..24 into a full 16-byte register (never read past
    // `qs`'s own 24 bytes — the high 8 lanes of `trit_of_16_avx2`'s result
    // are computed but simply not stored).
    let mut stage2_buf = [0u8; 16];
    stage2_buf[..8].copy_from_slice(&block.qs[16..24]);
    let stage2_src = _mm_loadu_si128(stage2_buf.as_ptr() as *const __m128i);
    for (n, &pow) in POW3.iter().take(5).enumerate() {
        let out16 = trit_of_16_avx2(stage2_src, pow);
        let mut tmp = [0u8; 16];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, out16);
        codes[80 + n * 8..80 + n * 8 + 8].copy_from_slice(&tmp[..8]);
    }

    let mut w = 120usize;
    for &pow in POW3.iter().take(4) {
        for &byte in block.qh.iter() {
            let q = byte.wrapping_mul(pow);
            codes[w] = (((q as u16) * 3) >> 8) as u8;
            w += 1;
        }
    }
    codes
}

/// Convert 16 trit codes (`u8`, values `0..=2`, held in the low 128 bits of
/// `codes16`) into two `__m256` vectors of `code - 1`, in original order.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn codes16_to_f32x8x2(codes16: __m128i) -> [__m256; 2] {
    let lo = _mm256_cvtepu8_epi32(codes16); // low 8 lanes -> 8 x i32
    let hi_src = _mm_srli_si128::<8>(codes16); // bring high 8 lanes down
    let hi = _mm256_cvtepu8_epi32(hi_src);
    let one = _mm256_set1_epi32(1);
    let a = _mm256_cvtepi32_ps(_mm256_sub_epi32(lo, one));
    let b = _mm256_cvtepi32_ps(_mm256_sub_epi32(hi, one));
    [a, b]
}

/// AVX2-accelerated dequantization of `PTQ1_0` blocks to FP32.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_ptq1_0_avx2(blocks: &[BlockPTQ1_0], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_PTQ1_0;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let codes = decode_ptq1_0_codes_avx2(block);
        let scale = _mm256_set1_ps(block.d.to_f32());
        let base = bi * QK_PTQ1_0;
        for chunk in 0..8 {
            let codes16 = _mm_loadu_si128(codes.as_ptr().add(chunk * 16) as *const __m128i);
            let [a, b] = codes16_to_f32x8x2(codes16);
            _mm256_storeu_ps(
                output.as_mut_ptr().add(base + chunk * 16),
                _mm256_mul_ps(scale, a),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(base + chunk * 16 + 8),
                _mm256_mul_ps(scale, b),
            );
        }
    }
    Ok(())
}

/// AVX2-accelerated native GEMV for a `PTQ1_0`-quantized weight matrix.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_ptq1_0_avx2(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PTQ1_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PTQ1_0,
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
    let blocks_per_row = k / QK_PTQ1_0;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut row_sum = 0.0f32;
        for (bi, block) in row_blocks.iter().enumerate() {
            let codes = decode_ptq1_0_codes_avx2(block);
            let inp_base = bi * QK_PTQ1_0;
            let mut acc = _mm256_setzero_ps();
            for chunk in 0..8 {
                let codes16 = _mm_loadu_si128(codes.as_ptr().add(chunk * 16) as *const __m128i);
                let [a, b] = codes16_to_f32x8x2(codes16);
                let inp_a = _mm256_loadu_ps(input.as_ptr().add(inp_base + chunk * 16));
                let inp_b = _mm256_loadu_ps(input.as_ptr().add(inp_base + chunk * 16 + 8));
                acc = _mm256_fmadd_ps(a, inp_a, acc);
                acc = _mm256_fmadd_ps(b, inp_b, acc);
            }
            row_sum += block.d.to_f32() * hsum8_avx2(acc);
        }
        output[row] = row_sum;
    }
    Ok(())
}

/// AVX2-accelerated GEMM for a `PTQ1_0`-quantized weight matrix.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_ptq1_0_avx2(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PTQ1_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PTQ1_0,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::dimension_mismatch("input", m * k, input.len()));
    }
    if output.len() < m * n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            m * n_rows,
            output.len(),
        ));
    }
    for mi in 0..m {
        let input_row = &input[mi * k..];
        let output_row = &mut output[mi * n_rows..(mi + 1) * n_rows];
        gemv_ptq1_0_avx2(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Register-blocked (MR-tiled) AVX2 GEMM — K-INT8 / gatekeeper REQUIRED #9
//
// The x86-64 twin of `simd_prism_neon.rs`'s blocked kernels, with the same
// bit-identity contract: one decoded block is consumed by `PRISM_GEMM_MR`
// batch rows, and each (batch row, weight row) pair keeps the exact
// `_mm256_fmadd_ps` sequence, `hsum8_avx2` and `row_sum += d * hsum` order
// the per-row GEMV above uses.
//
// `KernelTier::Avx512` routes here too (see `dispatch_prism.rs`'s module
// doc: the Prism formats have no dedicated AVX-512 kernel, and every
// AVX-512F CPU also has AVX2).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use crate::dequant_prism::{
    for_each_prism_register_block, validate_prism_gemm, PrismTileSpan, TwoBitBlockView,
};

/// # Safety
/// Requires AVX2 + FMA; all indices are pre-validated by
/// [`validate_prism_gemm`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn micro_two_bit_avx2<const MR: usize, B: TwoBitBlockView>(
    row_blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    qk: usize,
    span: PrismTileSpan,
) {
    let mut sums = [0.0f32; MR];
    for (bi, block) in row_blocks.iter().enumerate() {
        let inp_base = bi * qk;
        let qs = block.qs_bytes();
        let mut acc = [_mm256_setzero_ps(); MR];
        for chunk in 0..qs.len() / 2 {
            let val = decode_2bytes_avx2_arith_to_f32x8(qs[chunk * 2], qs[chunk * 2 + 1]);
            let col = inp_base + chunk * 8;
            for (r, a) in acc.iter_mut().enumerate() {
                let x = _mm256_loadu_ps(input.as_ptr().add((span.m0 + r) * span.k + col));
                *a = _mm256_fmadd_ps(val, x, *a);
            }
        }
        let d = block.scale();
        for (a, sum) in acc.iter().zip(sums.iter_mut()) {
            *sum += d * hsum8_avx2(*a);
        }
    }
    for (r, sum) in sums.iter().enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = *sum;
    }
}

/// # Safety
/// See [`micro_two_bit_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_two_bit_avx2<const MR: usize, B: TwoBitBlockView>(
    blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    qk: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    for ni in 0..n_rows {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let span = PrismTileSpan { k, n_rows, ni, m0 };
        micro_two_bit_avx2::<MR, B>(row_blocks, input, output, qk, span);
    }
}

/// Register-blocked AVX2 GEMM for either 2-bit Prism format, bit-identical
/// to the matching per-row AVX2 GEMV sweep.
///
/// # Errors
///
/// See [`validate_prism_gemm`].
///
/// # Safety
/// Requires AVX2 + FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_two_bit_avx2_blocked<B: TwoBitBlockView>(
    blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
    qk: usize,
) -> KernelResult<()> {
    let blocks_per_row =
        validate_prism_gemm(blocks.len(), input.len(), output.len(), m, n_rows, k, qk)?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    for_each_prism_register_block!(
        m,
        tile_two_bit_avx2,
        [B],
        blocks,
        input,
        output,
        n_rows,
        k,
        qk,
        blocks_per_row
    );
    Ok(())
}

/// Register-blocked AVX2 GEMM for `PQ2_0`.
///
/// # Errors
///
/// See [`validate_prism_gemm`].
///
/// # Safety
/// Requires AVX2 + FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_pq2_0_avx2_blocked(
    blocks: &[BlockPQ2_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    gemm_two_bit_avx2_blocked(blocks, input, output, m, n_rows, k, QK_PQ2_0)
}

/// Register-blocked AVX2 GEMM for `Q2_0_g64`.
///
/// # Errors
///
/// See [`validate_prism_gemm`].
///
/// # Safety
/// Requires AVX2 + FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_q2_0_g64_avx2_blocked(
    blocks: &[BlockQ2_0G64],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    gemm_two_bit_avx2_blocked(blocks, input, output, m, n_rows, k, QK_Q2_0_G64)
}

/// # Safety
/// Requires AVX2 + FMA; indices pre-validated by [`validate_prism_gemm`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn micro_ptq1_0_avx2<const MR: usize>(
    row_blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    span: PrismTileSpan,
) {
    let mut sums = [0.0f32; MR];
    for (bi, block) in row_blocks.iter().enumerate() {
        let codes = decode_ptq1_0_codes_avx2(block);
        let inp_base = bi * QK_PTQ1_0;
        let mut acc = [_mm256_setzero_ps(); MR];
        for chunk in 0..8 {
            let codes16 = _mm_loadu_si128(codes.as_ptr().add(chunk * 16) as *const __m128i);
            let [a_vec, b_vec] = codes16_to_f32x8x2(codes16);
            let col = inp_base + chunk * 16;
            for (r, a) in acc.iter_mut().enumerate() {
                let row_base = (span.m0 + r) * span.k + col;
                let inp_a = _mm256_loadu_ps(input.as_ptr().add(row_base));
                let inp_b = _mm256_loadu_ps(input.as_ptr().add(row_base + 8));
                *a = _mm256_fmadd_ps(a_vec, inp_a, *a);
                *a = _mm256_fmadd_ps(b_vec, inp_b, *a);
            }
        }
        let d = block.d.to_f32();
        for (a, sum) in acc.iter().zip(sums.iter_mut()) {
            *sum += d * hsum8_avx2(*a);
        }
    }
    for (r, sum) in sums.iter().enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = *sum;
    }
}

/// # Safety
/// See [`micro_ptq1_0_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tile_ptq1_0_avx2<const MR: usize>(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    for ni in 0..n_rows {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let span = PrismTileSpan { k, n_rows, ni, m0 };
        micro_ptq1_0_avx2::<MR>(row_blocks, input, output, span);
    }
}

/// Register-blocked AVX2 GEMM for `PTQ1_0`, bit-identical to
/// [`gemm_ptq1_0_avx2`] but trit-decoding each block once per register
/// block instead of once per batch row.
///
/// # Errors
///
/// See [`validate_prism_gemm`].
///
/// # Safety
/// Requires AVX2 + FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_ptq1_0_avx2_blocked(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_prism_gemm(
        blocks.len(),
        input.len(),
        output.len(),
        m,
        n_rows,
        k,
        QK_PTQ1_0,
    )?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    for_each_prism_register_block!(
        m,
        tile_ptq1_0_avx2,
        [],
        blocks,
        input,
        output,
        n_rows,
        k,
        blocks_per_row
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — gated to x86_64 like the rest of the file; also runtime-checks
// for AVX2 (the way `simd_avx2.rs`'s own ternary tests already do), since a
// binary built for a generic x86_64 target may run on a CPU without AVX2.
// Module name already contains "prism", matching the package gate's filter.
// ---------------------------------------------------------------------------

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use half::f16;

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    fn make_ptq1_block(scale: f32, qs: [u8; 24], qh: [u8; 2]) -> BlockPTQ1_0 {
        BlockPTQ1_0 {
            qs,
            qh,
            d: f16::from_f32(scale),
        }
    }

    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    #[test]
    fn avx2_pq2_0_dequant_matches_reference_bitwise_over_random_raw_bytes() {
        if !has_avx2() {
            return;
        }
        let mut state = 0xF00D_BEEFu32;
        let mut blocks = Vec::new();
        for _ in 0..80 {
            let qs: [u8; 32] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let scale = f16::from_f32(((lcg(&mut state) % 1000) as f32) / 100.0 - 5.0);
            blocks.push(BlockPQ2_0 { d: scale, qs });
        }
        let mut reference = vec![0.0f32; blocks.len() * QK_PQ2_0];
        crate::dequant_prism::dequant_pq2_0(&blocks, &mut reference).expect("reference");
        let mut avx2 = vec![0.0f32; blocks.len() * QK_PQ2_0];
        unsafe {
            dequant_pq2_0_avx2(&blocks, &mut avx2).expect("avx2");
        }
        assert_eq!(reference, avx2, "AVX2 PQ2_0 dequant must be bitwise-exact");
    }

    #[test]
    fn avx2_q2_0_g64_dequant_matches_reference_bitwise_over_random_raw_bytes() {
        if !has_avx2() {
            return;
        }
        let mut state = 0x1357_2468u32;
        let mut blocks = Vec::new();
        for _ in 0..80 {
            let qs: [u8; 16] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let scale = f16::from_f32(((lcg(&mut state) % 1000) as f32) / 200.0 - 2.5);
            blocks.push(BlockQ2_0G64 { d: scale, qs });
        }
        let mut reference = vec![0.0f32; blocks.len() * QK_Q2_0_G64];
        crate::dequant_prism::dequant_q2_0_g64(&blocks, &mut reference).expect("reference");
        let mut avx2 = vec![0.0f32; blocks.len() * QK_Q2_0_G64];
        unsafe {
            dequant_q2_0_g64_avx2(&blocks, &mut avx2).expect("avx2");
        }
        assert_eq!(
            reference, avx2,
            "AVX2 Q2_0_g64 dequant must be bitwise-exact"
        );
    }

    #[test]
    fn avx2_pq2_0_gemv_matches_reference_gemv() {
        if !has_avx2() {
            return;
        }
        let blocks = vec![
            BlockPQ2_0 {
                d: f16::from_f32(1.25),
                qs: [0xFF; 32],
            },
            BlockPQ2_0 {
                d: f16::from_f32(0.5),
                qs: [0x1B; 32],
            },
        ];
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let mut reference = vec![0.0f32; 1];
        crate::dequant_prism::gemv_pq2_0(&blocks, &input, &mut reference, 1, 256).expect("ref");
        let mut avx2 = vec![0.0f32; 1];
        unsafe {
            gemv_pq2_0_avx2(&blocks, &input, &mut avx2, 1, 256).expect("avx2");
        }
        assert!((reference[0] - avx2[0]).abs() < 1e-3);
    }

    #[test]
    fn avx2_pq2_0_gemm_matches_reference_gemm() {
        if !has_avx2() {
            return;
        }
        let blocks = vec![
            BlockPQ2_0 {
                d: f16::from_f32(1.0),
                qs: [0xAA; 32],
            };
            2
        ];
        let m = 3;
        let n_rows = 2;
        let k = 128;
        let input = vec![1.0f32; m * k];
        let mut reference = vec![0.0f32; m * n_rows];
        crate::dequant_prism::gemm_pq2_0(&blocks, &input, &mut reference, m, n_rows, k)
            .expect("ref");
        let mut avx2 = vec![0.0f32; m * n_rows];
        unsafe {
            gemm_pq2_0_avx2(&blocks, &input, &mut avx2, m, n_rows, k).expect("avx2");
        }
        for i in 0..m * n_rows {
            assert!((reference[i] - avx2[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn avx2_q2_0_g64_gemv_matches_reference_gemv() {
        if !has_avx2() {
            return;
        }
        let blocks = vec![BlockQ2_0G64 {
            d: f16::from_f32(0.75),
            qs: [0xE4; 16],
        }];
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 16.0).collect();
        let mut reference = vec![0.0f32; 1];
        crate::dequant_prism::gemv_q2_0_g64(&blocks, &input, &mut reference, 1, 64).expect("ref");
        let mut avx2 = vec![0.0f32; 1];
        unsafe {
            gemv_q2_0_g64_avx2(&blocks, &input, &mut avx2, 1, 64).expect("avx2");
        }
        assert!((reference[0] - avx2[0]).abs() < 1e-3);
    }

    #[test]
    fn avx2_dequant_reports_buffer_too_small() {
        if !has_avx2() {
            return;
        }
        let blocks = vec![BlockPQ2_0 {
            d: f16::from_f32(1.0),
            qs: [0xAA; 32],
        }];
        let mut output = vec![0.0f32; 0];
        let err = unsafe { dequant_pq2_0_avx2(&blocks, &mut output) }.expect_err("must fail");
        assert_eq!(err.buffer_name(), Some("output"));
    }

    /// Direct comparison of the vectorized trit-code extraction against the
    /// scalar reference, over raw (non-quantized) byte values.
    #[test]
    fn avx2_ptq1_0_code_extraction_matches_scalar_bitwise_over_full_byte_range() {
        if !has_avx2() {
            return;
        }
        let mut state = 0x0BAD_F00Du32;
        for _ in 0..200 {
            let qs: [u8; 24] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let qh: [u8; 2] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let block = make_ptq1_block(1.0, qs, qh);
            let scalar = crate::dequant_prism::decode_ptq1_0_codes(&block);
            let avx2 = unsafe { decode_ptq1_0_codes_avx2(&block) };
            assert_eq!(scalar, avx2, "trit codes diverge for qs={qs:?} qh={qh:?}");
        }
        for &b in &[0x00u8, 0xFFu8, 0x01, 0x80, 0x55, 0xAA] {
            let block = make_ptq1_block(1.0, [b; 24], [b; 2]);
            let scalar = crate::dequant_prism::decode_ptq1_0_codes(&block);
            let avx2 = unsafe { decode_ptq1_0_codes_avx2(&block) };
            assert_eq!(scalar, avx2, "diverge at uniform byte {b:#04x}");
        }
    }

    #[test]
    fn avx2_ptq1_0_dequant_matches_reference_bitwise() {
        if !has_avx2() {
            return;
        }
        let mut state = 0x4242_4242u32;
        let mut blocks = Vec::new();
        for _ in 0..64 {
            let qs: [u8; 24] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let qh: [u8; 2] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let scale = f16::from_f32(((lcg(&mut state) % 1000) as f32) / 300.0);
            blocks.push(BlockPTQ1_0 { qs, qh, d: scale });
        }
        let mut reference = vec![0.0f32; blocks.len() * QK_PTQ1_0];
        crate::dequant_prism::dequant_ptq1_0(&blocks, &mut reference).expect("reference");
        let mut avx2 = vec![0.0f32; blocks.len() * QK_PTQ1_0];
        unsafe {
            dequant_ptq1_0_avx2(&blocks, &mut avx2).expect("avx2");
        }
        assert_eq!(reference, avx2, "AVX2 PTQ1_0 dequant must be bitwise-exact");
    }

    #[test]
    fn avx2_ptq1_0_gemv_matches_reference_gemv() {
        if !has_avx2() {
            return;
        }
        let block = make_ptq1_block(1.0, [0x9C; 24], [0x33; 2]);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
        let mut reference = vec![0.0f32; 1];
        crate::gemv_ptq1::gemv_ptq1_0(&[block], &input, &mut reference, 1, QK_PTQ1_0).expect("ref");
        let mut avx2 = vec![0.0f32; 1];
        unsafe {
            gemv_ptq1_0_avx2(&[block], &input, &mut avx2, 1, QK_PTQ1_0).expect("avx2");
        }
        assert!((reference[0] - avx2[0]).abs() < 1e-3);
    }

    #[test]
    fn avx2_ptq1_0_gemm_matches_reference_gemm() {
        if !has_avx2() {
            return;
        }
        let blocks = vec![make_ptq1_block(1.0, [0x71; 24], [0x22; 2]); 2];
        let m = 2;
        let n_rows = 2;
        let k = QK_PTQ1_0;
        let input = vec![0.5f32; m * k];
        let mut reference = vec![0.0f32; m * n_rows];
        crate::gemv_ptq1::gemm_ptq1_0(&blocks, &input, &mut reference, m, n_rows, k).expect("ref");
        let mut avx2 = vec![0.0f32; m * n_rows];
        unsafe {
            gemm_ptq1_0_avx2(&blocks, &input, &mut avx2, m, n_rows, k).expect("avx2");
        }
        for i in 0..m * n_rows {
            assert!((reference[i] - avx2[i]).abs() < 1e-3);
        }
    }
}
