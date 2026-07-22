//! AVX2+FMA GEMV kernels for the standard GGUF quant formats Q4_0 and Q8_0.
//!
//! These mirror the structure of [`crate::simd_fp8_avx2`]: each row accumulates
//! its dot product in 8-wide f32 registers using `_mm256_fmadd_ps`, and the
//! per-block weight decode is done with SIMD integer instructions instead of a
//! scalar loop.
//!
//! - **Q8_0** (32 × i8 + f16 scale): each byte is sign-extended (`i8 → i32`),
//!   converted to f32, scaled by the block scale and fused-multiply-added with
//!   the matching input slice — 4 chunks of 8 lanes per block.
//! - **Q4_0** (32 × 4-bit + f16 scale, llama.cpp lo-hi split): the 16 packed
//!   bytes are split into lower nibbles (elements 0..16) and upper nibbles
//!   (elements 16..32) with a single `and` / `srli16`+`and`; each nibble group
//!   is zero-extended (`u8 → i32`), centred (`− 8`), converted to f32, scaled
//!   and fused-multiply-added.
//!
//! Both kernels produce results numerically equivalent (within f32 rounding)
//! to the scalar reference kernels in `crate::gemv_q4_0` / `crate::gemv_q8_0`;
//! the `parity` property tests assert this.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};

#[cfg(target_arch = "x86_64")]
use crate::error::{KernelError, KernelResult};

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Horizontal sum of the 8 f32 lanes of an AVX2 register.
///
/// # Safety
/// Requires AVX2 CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x01);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// Validate GEMV arguments shared by Q4_0/Q8_0 and return `blocks_per_row`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn validate_gemv(
    n_blocks: usize,
    input_len: usize,
    output_len: usize,
    n_rows: usize,
    in_features: usize,
    block_len: usize,
) -> KernelResult<usize> {
    if in_features % block_len != 0 {
        return Err(KernelError::NotBlockAligned {
            count: in_features,
            block_size: block_len,
        });
    }
    let blocks_per_row = in_features / block_len;
    let expected_blocks = n_rows * blocks_per_row;
    if n_blocks < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: n_blocks,
        });
    }
    if input_len < in_features {
        return Err(KernelError::DimensionMismatch {
            expected: in_features,
            got: input_len,
        });
    }
    if output_len < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output_len,
        });
    }
    Ok(blocks_per_row)
}

// ─── Q8_0 GEMV ───────────────────────────────────────────────────────────────

/// AVX2+FMA GEMV for a Q8_0-quantized weight matrix.
///
/// See [`crate::gemv_q8_0::gemv_q8_0`] for the parameter contract; this produces
/// the same result within f32 rounding.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_q8_0_avx2(
    blocks: &[BlockQ8_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_gemv(
        blocks.len(),
        input.len(),
        output.len(),
        n_rows,
        in_features,
        QK_Q8_0,
    )?;

    for row in 0..n_rows {
        let mut acc = _mm256_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let scale = _mm256_set1_ps(block.d.to_f32());
            let inp_base = bi * QK_Q8_0;

            // 4 chunks of 8 int8 weights.
            for chunk in 0_usize..4 {
                let off = chunk * 8;
                // Load 8 int8 into the low 64 bits, then sign-extend to 8 × i32.
                let bytes = _mm_loadl_epi64(block.qs.as_ptr().add(off).cast::<__m128i>());
                let wi = _mm256_cvtepi8_epi32(bytes);
                let wf = _mm256_cvtepi32_ps(wi);
                let ws = _mm256_mul_ps(scale, wf);
                let iv = _mm256_loadu_ps(input.as_ptr().add(inp_base + off));
                acc = _mm256_fmadd_ps(ws, iv, acc);
            }
        }

        output[row] = hsum_avx2(acc);
    }

    Ok(())
}

// ─── Q4_0 GEMV ───────────────────────────────────────────────────────────────

/// Decode 8 nibble bytes (already masked to `0..=15`) held in the low lanes of
/// `nibbles`, centre them (`− 8`), convert to f32, scale by `scale`, load the
/// matching 8 input values at `input[base..base + 8]` and fuse into `acc`.
///
/// # Safety
/// Requires AVX2+FMA CPU support and `input[base..base + 8]` in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn fma_q4_chunk(
    acc: __m256,
    nibbles: __m128i,
    eight: __m256i,
    scale: __m256,
    input: &[f32],
    base: usize,
) -> __m256 {
    // Zero-extend the low 8 bytes (u8) to 8 × i32, subtract 8, convert to f32.
    let wi = _mm256_sub_epi32(_mm256_cvtepu8_epi32(nibbles), eight);
    let wf = _mm256_cvtepi32_ps(wi);
    let ws = _mm256_mul_ps(scale, wf);
    let iv = _mm256_loadu_ps(input.as_ptr().add(base));
    _mm256_fmadd_ps(ws, iv, acc)
}

/// AVX2+FMA GEMV for a Q4_0-quantized weight matrix.
///
/// See [`crate::gemv_q4_0::gemv_q4_0`] for the parameter contract; this produces
/// the same result within f32 rounding. Uses the llama.cpp lo-hi split: byte `j`
/// holds element `j` in its lower nibble and element `j + 16` in its upper nibble.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_q4_0_avx2(
    blocks: &[BlockQ4_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_gemv(
        blocks.len(),
        input.len(),
        output.len(),
        n_rows,
        in_features,
        QK_Q4_0,
    )?;

    let mask_lo = _mm_set1_epi8(0x0F);
    let eight = _mm256_set1_epi32(8);

    for row in 0..n_rows {
        let mut acc = _mm256_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let scale = _mm256_set1_ps(block.d.to_f32());
            let inp_base = bi * QK_Q4_0;

            // Load the 16 packed bytes.
            let v = _mm_loadu_si128(block.qs.as_ptr().cast::<__m128i>());
            // Lower nibbles → elements 0..16; upper nibbles → elements 16..32.
            let lo = _mm_and_si128(v, mask_lo);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask_lo);

            // lo bytes 0..8  → elements 0..8   ↔ input[inp_base + 0..8]
            acc = fma_q4_chunk(acc, lo, eight, scale, input, inp_base);
            // lo bytes 8..16 → elements 8..16  ↔ input[inp_base + 8..16]
            acc = fma_q4_chunk(
                acc,
                _mm_srli_si128(lo, 8),
                eight,
                scale,
                input,
                inp_base + 8,
            );
            // hi bytes 0..8  → elements 16..24 ↔ input[inp_base + 16..24]
            acc = fma_q4_chunk(acc, hi, eight, scale, input, inp_base + 16);
            // hi bytes 8..16 → elements 24..32 ↔ input[inp_base + 24..32]
            acc = fma_q4_chunk(
                acc,
                _mm_srli_si128(hi, 8),
                eight,
                scale,
                input,
                inp_base + 24,
            );
        }

        output[row] = hsum_avx2(acc);
    }

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::{gemv_q4_0::gemv_q4_0_scalar, gemv_q8_0::gemv_q8_0_scalar};
    use oxibonsai_core::{BlockQ4_0, BlockQ8_0};

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    fn make_input(k: usize, seed: u32) -> Vec<f32> {
        (0..k)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((x >> 8) as f32 / u32::MAX as f32) * 4.0 - 2.0
            })
            .collect()
    }

    fn q8_blocks(n_rows: usize, blocks_per_row: usize, seed: u32) -> Vec<BlockQ8_0> {
        let mut raw = Vec::with_capacity(n_rows * blocks_per_row * 32);
        for i in 0..n_rows * blocks_per_row * 32 {
            let x = (i as u32).wrapping_mul(40503).wrapping_add(seed);
            raw.push(((x >> 7) as f32 / u32::MAX as f32) * 6.0 - 3.0);
        }
        BlockQ8_0::quantize(&raw).expect("quantize q8_0")
    }

    fn q4_blocks(n_rows: usize, blocks_per_row: usize, seed: u32) -> Vec<BlockQ4_0> {
        let mut raw = Vec::with_capacity(n_rows * blocks_per_row * 32);
        for i in 0..n_rows * blocks_per_row * 32 {
            let x = (i as u32).wrapping_mul(2246822519).wrapping_add(seed);
            raw.push(((x >> 9) as f32 / u32::MAX as f32) * 8.0 - 4.0);
        }
        BlockQ4_0::quantize(&raw).expect("quantize q4_0")
    }

    #[test]
    fn q8_0_avx2_matches_scalar() {
        if !has_avx2() {
            return;
        }
        for (n_rows, in_features) in [(1, 32), (3, 64), (5, 96), (17, 128)] {
            let blocks_per_row = in_features / 32;
            let blocks = q8_blocks(n_rows, blocks_per_row, 11);
            let input = make_input(in_features, 7);

            let mut out_ref = vec![0.0f32; n_rows];
            let mut out_simd = vec![0.0f32; n_rows];
            gemv_q8_0_scalar(&blocks, &input, &mut out_ref, n_rows, in_features).unwrap();
            unsafe {
                gemv_q8_0_avx2(&blocks, &input, &mut out_simd, n_rows, in_features).unwrap();
            }
            for r in 0..n_rows {
                let tol = 1e-3 * out_ref[r].abs().max(1.0);
                assert!(
                    (out_ref[r] - out_simd[r]).abs() <= tol,
                    "q8_0 row {r} (n_rows={n_rows}, k={in_features}): ref={}, simd={}",
                    out_ref[r],
                    out_simd[r]
                );
            }
        }
    }

    #[test]
    fn q4_0_avx2_matches_scalar() {
        if !has_avx2() {
            return;
        }
        for (n_rows, in_features) in [(1, 32), (2, 64), (7, 96), (17, 160)] {
            let blocks_per_row = in_features / 32;
            let blocks = q4_blocks(n_rows, blocks_per_row, 23);
            let input = make_input(in_features, 5);

            let mut out_ref = vec![0.0f32; n_rows];
            let mut out_simd = vec![0.0f32; n_rows];
            gemv_q4_0_scalar(&blocks, &input, &mut out_ref, n_rows, in_features).unwrap();
            unsafe {
                gemv_q4_0_avx2(&blocks, &input, &mut out_simd, n_rows, in_features).unwrap();
            }
            for r in 0..n_rows {
                let tol = 1e-3 * out_ref[r].abs().max(1.0);
                assert!(
                    (out_ref[r] - out_simd[r]).abs() <= tol,
                    "q4_0 row {r} (n_rows={n_rows}, k={in_features}): ref={}, simd={}",
                    out_ref[r],
                    out_simd[r]
                );
            }
        }
    }

    #[test]
    fn q4_0_avx2_rejects_bad_dims() {
        if !has_avx2() {
            return;
        }
        let blocks = q4_blocks(1, 1, 1);
        let input = make_input(32, 1);
        let mut output = vec![0.0f32; 1];
        // Not block aligned.
        unsafe {
            assert!(gemv_q4_0_avx2(&blocks, &input, &mut output, 1, 31).is_err());
        }
    }
}
