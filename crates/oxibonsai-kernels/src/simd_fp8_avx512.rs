//! AVX-512 FP8 dequantization and GEMV/GEMM kernels.
//!
//! Uses `_mm512_i32gather_ps` for 16-wide LUT lookups per iteration.
//! Each FP8 block holds 32 weights; the block is processed in 2 iterations
//! of 16 lanes each (vs 4 × 8 for AVX2).
//!
//! **Gather pattern:**
//! 1. Load 16 bytes with `_mm_loadu_si128`.
//! 2. Zero-extend u8 → i32 with `_mm512_cvtepu8_epi32`.
//! 3. Gather f32 from the LUT: `_mm512_i32gather_ps(indices, lut_ptr, 4)`.
//! 4. Multiply by the block scale broadcast with `_mm512_set1_ps`.
//! 5. Accumulate with `_mm512_fmadd_ps`.

// AVX-512 intrinsics were stabilised in Rust 1.89.0, but our workspace
// MSRV is 1.86.0.  All functions in this module are guarded by
// `#[target_feature(enable = "avx512f", …)]` — only reachable when the CPU
// actually supports AVX-512 — so the MSRV lint is a false positive here.
#![allow(clippy::incompatible_msrv)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};

#[cfg(target_arch = "x86_64")]
use crate::error::{KernelError, KernelResult};

#[cfg(target_arch = "x86_64")]
use crate::fp8_lut::{fp8_e4m3_lut, fp8_e5m2_lut};

// ─── Helper: horizontal sum ───────────────────────────────────────────────

/// Horizontal sum of all 16 f32 lanes using the AVX-512 built-in reducer.
///
/// # Safety
/// Requires AVX-512F CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_avx512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

// ─── Helper: load 16 FP8 bytes, gather from LUT, return __m512 ──────────

/// Load 16 bytes from `qs_ptr`, look each up in `lut`, and return a `__m512`
/// holding 16 decoded f32 values.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
/// `qs_ptr` must point to at least 16 valid bytes.
/// `lut` must have at least 256 entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn gather16_from_lut(qs_ptr: *const u8, lut: &[f32; 256]) -> __m512 {
    // Load 16 bytes into a 128-bit register (unaligned)
    let bytes16 = _mm_loadu_si128(qs_ptr.cast());
    // Zero-extend 16 × u8 → 16 × i32 in a 512-bit register
    let indices = _mm512_cvtepu8_epi32(bytes16);
    // Gather: indices[i] selects lut[indices[i]], stride = 4 bytes = 1 f32
    _mm512_i32gather_ps(indices, lut.as_ptr().cast(), 4)
}

// ─── Dequantization ──────────────────────────────────────────────────────

/// AVX-512-accelerated dequantization for FP8 E4M3FN blocks.
///
/// Processes 32 weights per block in 2 iterations of 16 lanes.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn dequant_fp8_e4m3_avx512(
    blocks: &[BlockFP8E4M3],
    output: &mut [f32],
) -> KernelResult<()> {
    let needed = blocks.len() * QK_FP8;
    if output.len() < needed {
        return Err(KernelError::BufferTooSmall {
            needed,
            available: output.len(),
        });
    }

    let lut = fp8_e4m3_lut();

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let scale = _mm512_set1_ps(d);
        let base = bi * QK_FP8;

        // 2 chunks × 16 lanes = 32 elements
        for chunk in 0_usize..2 {
            let qs_ptr = block.qs.as_ptr().add(chunk * 16);
            let decoded = gather16_from_lut(qs_ptr, lut);
            let result = _mm512_mul_ps(scale, decoded);
            _mm512_storeu_ps(output.as_mut_ptr().add(base + chunk * 16), result);
        }
    }

    Ok(())
}

/// AVX-512-accelerated dequantization for FP8 E5M2 blocks.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn dequant_fp8_e5m2_avx512(
    blocks: &[BlockFP8E5M2],
    output: &mut [f32],
) -> KernelResult<()> {
    let needed = blocks.len() * QK_FP8;
    if output.len() < needed {
        return Err(KernelError::BufferTooSmall {
            needed,
            available: output.len(),
        });
    }

    let lut = fp8_e5m2_lut();

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let scale = _mm512_set1_ps(d);
        let base = bi * QK_FP8;

        for chunk in 0_usize..2 {
            let qs_ptr = block.qs.as_ptr().add(chunk * 16);
            let decoded = gather16_from_lut(qs_ptr, lut);
            let result = _mm512_mul_ps(scale, decoded);
            _mm512_storeu_ps(output.as_mut_ptr().add(base + chunk * 16), result);
        }
    }

    Ok(())
}

// ─── GEMV ────────────────────────────────────────────────────────────────

/// AVX-512-accelerated FP8 E4M3FN GEMV.
///
/// Computes `output[row] = dot(weight_row, input)` for each row using
/// 16-wide FMA accumulation with LUT gather.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_fp8_e4m3_avx512(
    blocks: &[BlockFP8E4M3],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    validate_gemv_args(blocks.len(), input.len(), output.len(), n_rows, k, QK_FP8)?;

    let lut = fp8_e4m3_lut();
    let blocks_per_row = k / QK_FP8;

    for row in 0..n_rows {
        let mut row_acc = _mm512_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let scale = _mm512_set1_ps(d);
            let inp_base = bi * QK_FP8;

            // 2 chunks of 16
            for chunk in 0_usize..2 {
                let off = chunk * 16;
                let qs_ptr = block.qs.as_ptr().add(off);
                let wv = gather16_from_lut(qs_ptr, lut);
                let ws = _mm512_mul_ps(scale, wv);
                let iv = _mm512_loadu_ps(input.as_ptr().add(inp_base + off));
                row_acc = _mm512_fmadd_ps(ws, iv, row_acc);
            }
        }

        output[row] = hsum_avx512(row_acc);
    }

    Ok(())
}

/// AVX-512-accelerated FP8 E5M2 GEMV.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_fp8_e5m2_avx512(
    blocks: &[BlockFP8E5M2],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    validate_gemv_args(blocks.len(), input.len(), output.len(), n_rows, k, QK_FP8)?;

    let lut = fp8_e5m2_lut();
    let blocks_per_row = k / QK_FP8;

    for row in 0..n_rows {
        let mut row_acc = _mm512_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let scale = _mm512_set1_ps(d);
            let inp_base = bi * QK_FP8;

            for chunk in 0_usize..2 {
                let off = chunk * 16;
                let qs_ptr = block.qs.as_ptr().add(off);
                let wv = gather16_from_lut(qs_ptr, lut);
                let ws = _mm512_mul_ps(scale, wv);
                let iv = _mm512_loadu_ps(input.as_ptr().add(inp_base + off));
                row_acc = _mm512_fmadd_ps(ws, iv, row_acc);
            }
        }

        output[row] = hsum_avx512(row_acc);
    }

    Ok(())
}

// ─── GEMM ────────────────────────────────────────────────────────────────

/// AVX-512-accelerated FP8 E4M3FN GEMM.
///
/// Dispatches one GEMV call per batch element.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemm_fp8_e4m3_avx512(
    blocks: &[BlockFP8E4M3],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch: usize,
) -> KernelResult<()> {
    validate_gemm_args(
        blocks.len(),
        inputs.len(),
        outputs.len(),
        n_rows,
        k,
        batch,
        QK_FP8,
    )?;

    for b in 0..batch {
        let input_row = &inputs[b * k..(b + 1) * k];
        let output_row = &mut outputs[b * n_rows..(b + 1) * n_rows];
        gemv_fp8_e4m3_avx512(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

/// AVX-512-accelerated FP8 E5M2 GEMM.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemm_fp8_e5m2_avx512(
    blocks: &[BlockFP8E5M2],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch: usize,
) -> KernelResult<()> {
    validate_gemm_args(
        blocks.len(),
        inputs.len(),
        outputs.len(),
        n_rows,
        k,
        batch,
        QK_FP8,
    )?;

    for b in 0..batch {
        let input_row = &inputs[b * k..(b + 1) * k];
        let output_row = &mut outputs[b * n_rows..(b + 1) * n_rows];
        gemv_fp8_e5m2_avx512(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ─── Shared validation helpers ────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn validate_gemv_args(
    n_blocks: usize,
    input_len: usize,
    output_len: usize,
    n_rows: usize,
    k: usize,
    qk: usize,
) -> KernelResult<()> {
    if k % qk != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: qk,
        });
    }
    if input_len < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input_len,
        });
    }
    if output_len < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output_len,
        });
    }
    let blocks_per_row = k / qk;
    let expected_blocks = n_rows * blocks_per_row;
    if n_blocks < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: n_blocks,
        });
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn validate_gemm_args(
    n_blocks: usize,
    inputs_len: usize,
    outputs_len: usize,
    n_rows: usize,
    k: usize,
    batch: usize,
    qk: usize,
) -> KernelResult<()> {
    if k % qk != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: qk,
        });
    }
    if inputs_len < batch * k {
        return Err(KernelError::DimensionMismatch {
            expected: batch * k,
            got: inputs_len,
        });
    }
    if outputs_len < batch * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: batch * n_rows,
            available: outputs_len,
        });
    }
    let blocks_per_row = k / qk;
    let expected_blocks = n_rows * blocks_per_row;
    if n_blocks < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: n_blocks,
        });
    }
    Ok(())
}
