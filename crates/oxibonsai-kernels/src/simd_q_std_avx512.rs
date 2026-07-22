//! AVX-512 GEMV kernels for the standard GGUF quant formats Q4_0 and Q8_0.
//!
//! 512-bit counterpart of [`crate::simd_q_std_avx2`]: 16 f32 lanes per register
//! and `_mm512_reduce_add_ps` for the horizontal sum, so a Q8_0 block is two
//! 16-wide chunks and a Q4_0 block is two 16-wide chunks (all 16 lower nibbles,
//! then all 16 upper nibbles). Results match the scalar reference within f32
//! rounding (asserted by the `parity` tests).

// AVX-512 intrinsics were stabilised in Rust 1.89.0, but our workspace MSRV is
// 1.86.0. Every function here is guarded by `#[target_feature(enable =
// "avx512f", …)]` and is therefore only reachable when the CPU actually
// supports AVX-512, making the MSRV lint a false positive for this file.
#![allow(clippy::incompatible_msrv)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};

#[cfg(target_arch = "x86_64")]
use crate::error::{KernelError, KernelResult};

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

/// AVX-512 GEMV for a Q8_0-quantized weight matrix.
///
/// See [`crate::gemv_q8_0::gemv_q8_0`] for the parameter contract.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_q8_0_avx512(
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
        let mut acc = _mm512_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let scale = _mm512_set1_ps(block.d.to_f32());
            let inp_base = bi * QK_Q8_0;

            // 2 chunks of 16 int8 weights.
            for chunk in 0_usize..2 {
                let off = chunk * 16;
                let bytes = _mm_loadu_si128(block.qs.as_ptr().add(off).cast::<__m128i>());
                let wi = _mm512_cvtepi8_epi32(bytes);
                let wf = _mm512_cvtepi32_ps(wi);
                let ws = _mm512_mul_ps(scale, wf);
                let iv = _mm512_loadu_ps(input.as_ptr().add(inp_base + off));
                acc = _mm512_fmadd_ps(ws, iv, acc);
            }
        }

        output[row] = _mm512_reduce_add_ps(acc);
    }

    Ok(())
}

// ─── Q4_0 GEMV ───────────────────────────────────────────────────────────────

/// AVX-512 GEMV for a Q4_0-quantized weight matrix.
///
/// See [`crate::gemv_q4_0::gemv_q4_0`] for the parameter contract. Uses the
/// llama.cpp lo-hi split: byte `j` holds element `j` in its lower nibble and
/// element `j + 16` in its upper nibble.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_q4_0_avx512(
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
    let eight = _mm512_set1_epi32(8);

    for row in 0..n_rows {
        let mut acc = _mm512_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let scale = _mm512_set1_ps(block.d.to_f32());
            let inp_base = bi * QK_Q4_0;

            let v = _mm_loadu_si128(block.qs.as_ptr().cast::<__m128i>());
            // 16 lower nibbles → elements 0..16; 16 upper nibbles → 16..32.
            let lo = _mm_and_si128(v, mask_lo);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask_lo);

            // Lower nibbles ↔ input[inp_base + 0..16].
            let wlo = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(lo), eight));
            let ilo = _mm512_loadu_ps(input.as_ptr().add(inp_base));
            acc = _mm512_fmadd_ps(_mm512_mul_ps(scale, wlo), ilo, acc);

            // Upper nibbles ↔ input[inp_base + 16..32].
            let whi = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(hi), eight));
            let ihi = _mm512_loadu_ps(input.as_ptr().add(inp_base + 16));
            acc = _mm512_fmadd_ps(_mm512_mul_ps(scale, whi), ihi, acc);
        }

        output[row] = _mm512_reduce_add_ps(acc);
    }

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::{gemv_q4_0::gemv_q4_0_scalar, gemv_q8_0::gemv_q8_0_scalar};
    use oxibonsai_core::{BlockQ4_0, BlockQ8_0};

    fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
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
    fn q8_0_avx512_matches_scalar() {
        if !has_avx512() {
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
                gemv_q8_0_avx512(&blocks, &input, &mut out_simd, n_rows, in_features).unwrap();
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
    fn q4_0_avx512_matches_scalar() {
        if !has_avx512() {
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
                gemv_q4_0_avx512(&blocks, &input, &mut out_simd, n_rows, in_features).unwrap();
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
}
