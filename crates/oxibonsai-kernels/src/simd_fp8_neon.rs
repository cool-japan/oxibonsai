//! NEON FP8 dequantization and GEMV/GEMM kernels.
//!
//! NEON does not have a scatter/gather instruction, so we perform 4 scalar
//! LUT lookups to fill a stack `[f32; 4]` array, then load it with
//! `vld1q_f32`.  Each FP8 block holds 32 weights; the block is processed in
//! 8 iterations of 4 lanes each.
//!
//! **Compute pattern per chunk:**
//! ```text
//! w = [lut[qs[off+0]], lut[qs[off+1]], lut[qs[off+2]], lut[qs[off+3]]]
//! wv = vld1q_f32(w)
//! iv = vld1q_f32(input + inp_base + off)
//! row_acc = vfmaq_f32(row_acc, vmulq_f32(scale, wv), iv)
//! ```
//! where `scale = vdupq_n_f32(block.d.to_f32())`.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};

#[cfg(target_arch = "aarch64")]
use crate::error::{KernelError, KernelResult};

#[cfg(target_arch = "aarch64")]
use crate::fp8_lut::{fp8_e4m3_lut, fp8_e5m2_lut};

// ─── Helper: horizontal sum ───────────────────────────────────────────────

/// Sum all 4 f32 lanes of a NEON `float32x4_t` register.
///
/// `vaddvq_f32` is a single-instruction horizontal add available on AArch64
/// (NEON v8 and later).
///
/// # Safety
/// Requires NEON CPU support. Always available on AArch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

// ─── Dequantization ──────────────────────────────────────────────────────

/// NEON-accelerated dequantization for FP8 E4M3FN blocks.
///
/// Each block produces `QK_FP8 = 32` output values.  The 32 weights are
/// decoded 4 at a time via scalar LUT lookups then stored with `vst1q_f32`.
///
/// # Safety
/// Requires NEON CPU support. Always available on AArch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_fp8_e4m3_neon(
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
        let scale = vdupq_n_f32(d);
        let base = bi * QK_FP8;

        // 8 chunks × 4 lanes = 32 elements per block
        for chunk in 0_usize..8 {
            let off = chunk * 4;
            let w = [
                lut[block.qs[off] as usize],
                lut[block.qs[off + 1] as usize],
                lut[block.qs[off + 2] as usize],
                lut[block.qs[off + 3] as usize],
            ];
            let wv = vld1q_f32(w.as_ptr());
            let result = vmulq_f32(scale, wv);
            vst1q_f32(output.as_mut_ptr().add(base + off), result);
        }
    }

    Ok(())
}

/// NEON-accelerated dequantization for FP8 E5M2 blocks.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_fp8_e5m2_neon(
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
        let scale = vdupq_n_f32(d);
        let base = bi * QK_FP8;

        for chunk in 0_usize..8 {
            let off = chunk * 4;
            let w = [
                lut[block.qs[off] as usize],
                lut[block.qs[off + 1] as usize],
                lut[block.qs[off + 2] as usize],
                lut[block.qs[off + 3] as usize],
            ];
            let wv = vld1q_f32(w.as_ptr());
            let result = vmulq_f32(scale, wv);
            vst1q_f32(output.as_mut_ptr().add(base + off), result);
        }
    }

    Ok(())
}

// ─── GEMV ────────────────────────────────────────────────────────────────

/// NEON-accelerated FP8 E4M3FN GEMV.
///
/// `output[row] = dot(weight_row, input)` using 4-wide FMA accumulation
/// with scalar LUT decode per lane.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_fp8_e4m3_neon(
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
        let mut row_acc = vdupq_n_f32(0.0_f32);

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let scale = vdupq_n_f32(d);
            let inp_base = bi * QK_FP8;

            // 8 chunks × 4 = 32 elements
            for chunk in 0_usize..8 {
                let off = chunk * 4;
                let w = [
                    lut[block.qs[off] as usize],
                    lut[block.qs[off + 1] as usize],
                    lut[block.qs[off + 2] as usize],
                    lut[block.qs[off + 3] as usize],
                ];
                let wv = vld1q_f32(w.as_ptr());
                let iv = vld1q_f32(input.as_ptr().add(inp_base + off));
                // row_acc += scale × wv × iv
                row_acc = vfmaq_f32(row_acc, vmulq_f32(scale, wv), iv);
            }
        }

        output[row] = hsum_neon(row_acc);
    }

    Ok(())
}

/// NEON-accelerated FP8 E5M2 GEMV.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_fp8_e5m2_neon(
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
        let mut row_acc = vdupq_n_f32(0.0_f32);

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let scale = vdupq_n_f32(d);
            let inp_base = bi * QK_FP8;

            for chunk in 0_usize..8 {
                let off = chunk * 4;
                let w = [
                    lut[block.qs[off] as usize],
                    lut[block.qs[off + 1] as usize],
                    lut[block.qs[off + 2] as usize],
                    lut[block.qs[off + 3] as usize],
                ];
                let wv = vld1q_f32(w.as_ptr());
                let iv = vld1q_f32(input.as_ptr().add(inp_base + off));
                row_acc = vfmaq_f32(row_acc, vmulq_f32(scale, wv), iv);
            }
        }

        output[row] = hsum_neon(row_acc);
    }

    Ok(())
}

// ─── GEMM ────────────────────────────────────────────────────────────────

/// NEON-accelerated FP8 E4M3FN GEMM.
///
/// Dispatches one GEMV call per batch element.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_fp8_e4m3_neon(
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
        gemv_fp8_e4m3_neon(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

/// NEON-accelerated FP8 E5M2 GEMM.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_fp8_e5m2_neon(
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
        gemv_fp8_e5m2_neon(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ─── Shared validation helpers ────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
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

#[cfg(target_arch = "aarch64")]
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
