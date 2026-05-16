//! AVX2+FMA FP8 dequantization and GEMV/GEMM kernels.
//!
//! Uses the `_mm256_i32gather_ps` instruction to perform 8-wide LUT lookups
//! per iteration.  Each FP8 block holds 32 weights; the block is processed
//! in 4 iterations of 8 lanes each.
//!
//! **Gather pattern:**
//! 1. Load 8 bytes (`_mm_loadl_epi64` + zero-extension).
//! 2. Zero-extend u8 → i32 with `_mm256_cvtepu8_epi32`.
//! 3. Gather f32 from the LUT: `_mm256_i32gather_ps(lut_ptr, indices, 4)`.
//! 4. Multiply by the block scale (broadcast via `_mm256_set1_ps`).
//! 5. Accumulate dot products with `_mm256_fmadd_ps`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};

#[cfg(target_arch = "x86_64")]
use crate::error::{KernelError, KernelResult};

#[cfg(target_arch = "x86_64")]
use crate::fp8_lut::{fp8_e4m3_lut, fp8_e5m2_lut};

// ─── Helper: horizontal sum ───────────────────────────────────────────────

/// Horizontal sum of all 8 f32 lanes in an AVX2 `__m256` register.
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

// ─── Helper: load 8 FP8 bytes, gather from LUT, return __m256 ──────────

/// Load 8 bytes from `qs_ptr` (unaligned), look up each in `lut`, and
/// return an `__m256` holding the 8 decoded f32 values.
///
/// # Safety
/// Requires AVX2 CPU support.  `qs_ptr` must point to at least 8 valid bytes.
/// `lut` must have at least 256 entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn gather8_from_lut(qs_ptr: *const u8, lut: &[f32; 256]) -> __m256 {
    // Load 8 bytes into the low 64 bits of a 128-bit register
    let bytes8 = _mm_loadl_epi64(qs_ptr.cast());
    // Zero-extend each u8 to i32 → 8 × i32 in a 256-bit register
    let indices = _mm256_cvtepu8_epi32(bytes8);
    // Gather: each lane i → lut_ptr[indices[i]], stride = 4 bytes = 1 f32
    _mm256_i32gather_ps(lut.as_ptr(), indices, 4)
}

// ─── Dequantization ──────────────────────────────────────────────────────

/// AVX2-accelerated dequantization for FP8 E4M3FN blocks.
///
/// Each block produces `QK_FP8 = 32` output values.  The 32 weights are
/// processed in 4 chunks of 8 via `_mm256_i32gather_ps`.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_fp8_e4m3_avx2(
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
        let scale = _mm256_set1_ps(d);
        let base = bi * QK_FP8;

        // 4 chunks × 8 lanes = 32 elements per block
        for chunk in 0_usize..4 {
            let qs_ptr = block.qs.as_ptr().add(chunk * 8);
            let decoded = gather8_from_lut(qs_ptr, lut);
            let result = _mm256_mul_ps(scale, decoded);
            _mm256_storeu_ps(output.as_mut_ptr().add(base + chunk * 8), result);
        }
    }

    Ok(())
}

/// AVX2-accelerated dequantization for FP8 E5M2 blocks.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_fp8_e5m2_avx2(
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
        let scale = _mm256_set1_ps(d);
        let base = bi * QK_FP8;

        for chunk in 0_usize..4 {
            let qs_ptr = block.qs.as_ptr().add(chunk * 8);
            let decoded = gather8_from_lut(qs_ptr, lut);
            let result = _mm256_mul_ps(scale, decoded);
            _mm256_storeu_ps(output.as_mut_ptr().add(base + chunk * 8), result);
        }
    }

    Ok(())
}

// ─── GEMV ────────────────────────────────────────────────────────────────

/// AVX2-accelerated FP8 E4M3FN GEMV.
///
/// For each row: `output[row] = Σ_block( d_block × dot(decoded_weights, input_slice) )`
/// using 8-wide FMA accumulation with LUT gather.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_fp8_e4m3_avx2(
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
        let mut row_acc = _mm256_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let scale = _mm256_set1_ps(d);
            let inp_base = bi * QK_FP8;

            // 4 chunks of 8
            for chunk in 0_usize..4 {
                let off = chunk * 8;
                let qs_ptr = block.qs.as_ptr().add(off);
                let wv = gather8_from_lut(qs_ptr, lut);
                // scale the weights: w_scaled = scale × decoded
                let ws = _mm256_mul_ps(scale, wv);
                // load input slice
                let iv = _mm256_loadu_ps(input.as_ptr().add(inp_base + off));
                // FMA: row_acc += ws * iv
                row_acc = _mm256_fmadd_ps(ws, iv, row_acc);
            }
        }

        output[row] = hsum_avx2(row_acc);
    }

    Ok(())
}

/// AVX2-accelerated FP8 E5M2 GEMV.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_fp8_e5m2_avx2(
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
        let mut row_acc = _mm256_setzero_ps();

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let scale = _mm256_set1_ps(d);
            let inp_base = bi * QK_FP8;

            for chunk in 0_usize..4 {
                let off = chunk * 8;
                let qs_ptr = block.qs.as_ptr().add(off);
                let wv = gather8_from_lut(qs_ptr, lut);
                let ws = _mm256_mul_ps(scale, wv);
                let iv = _mm256_loadu_ps(input.as_ptr().add(inp_base + off));
                row_acc = _mm256_fmadd_ps(ws, iv, row_acc);
            }
        }

        output[row] = hsum_avx2(row_acc);
    }

    Ok(())
}

// ─── GEMM ────────────────────────────────────────────────────────────────

/// AVX2-accelerated FP8 E4M3FN GEMM.
///
/// Dispatches one GEMV call per batch element.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_fp8_e4m3_avx2(
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
        // Safety: the validation above guarantees outputs has ≥ batch * n_rows elements.
        let output_row = &mut outputs[b * n_rows..(b + 1) * n_rows];
        gemv_fp8_e4m3_avx2(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

/// AVX2-accelerated FP8 E5M2 GEMM.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_fp8_e5m2_avx2(
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
        gemv_fp8_e5m2_avx2(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ─── Shared validation helpers ────────────────────────────────────────────

/// Validate GEMV preconditions (shared between E4M3 and E5M2 paths).
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

/// Validate GEMM preconditions (shared between E4M3 and E5M2 paths).
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
