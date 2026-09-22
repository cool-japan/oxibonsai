//! NEON-accelerated dequant / GEMV / GEMM for the PrismML Bonsai 2 formats:
//! `PQ2_0`, mainline `Q2_0_g64`, and `PTQ1_0`.
//!
//! See [`crate::dequant_prism`]'s module doc for the format backgrounds and
//! the shared `code -> value` map. This file adds two things beyond that
//! scalar reference:
//!
//! - [`decode_byte_arith_to_f32x4`]: one byte -> 4 lanes, the arithmetic
//!   `code - 1` map (no reserved-code masking — every 2-bit code is valid for
//!   this family, unlike the legacy ternary LUT), used by the `PQ2_0` /
//!   `Q2_0_g64` tiers.
//! - [`decode_ptq1_0_codes_neon`]: a genuinely vectorized re-implementation
//!   of [`crate::dequant_prism::decode_ptq1_0_codes`]'s base-3 trit unpack
//!   (widen -> multiply -> mask -> multiply -> shift -> narrow, entirely in
//!   16-bit lanes so the `u8` wrap trap is reproduced exactly: the low byte
//!   of an exact, non-overflowing 16-bit product is bit-identical to a
//!   `u8::wrapping_mul`). Tested bitwise against the scalar decoder below.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use oxibonsai_core::{
    BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, POW3, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64,
};

#[cfg(target_arch = "aarch64")]
use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Decode one byte's four 2-bit codes to `[f32; 4]` (as a NEON vector) via
/// the arithmetic map `value = (code as i32) - 1`. Unlike the legacy ternary
/// LUT's decode helper, no lane is masked to a reserved value — `PQ2_0` /
/// `Q2_0_g64` give every one of the four codes a distinct, valid meaning
/// (`00→-1 01→0 10→+1 11→+2`).
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_byte_arith_to_f32x4(b: u8) -> float32x4_t {
    let idx_arr: [u32; 4] = [
        (b & 0b11) as u32,
        ((b >> 2) & 0b11) as u32,
        ((b >> 4) & 0b11) as u32,
        ((b >> 6) & 0b11) as u32,
    ];
    let idx_v = vreinterpretq_s32_u32(vld1q_u32(idx_arr.as_ptr()));
    let val_i32 = vsubq_s32(idx_v, vdupq_n_s32(1));
    vcvtq_f32_s32(val_i32)
}

/// Horizontal sum of 4 f32 lanes.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum4_neon(v: float32x4_t) -> f32 {
    let pair = vpaddq_f32(v, v);
    let sum = vpaddq_f32(pair, pair);
    vgetq_lane_f32::<0>(sum)
}

// ---------------------------------------------------------------------------
// PQ2_0 (ggml id 142)
// ---------------------------------------------------------------------------

/// NEON-accelerated dequantization of `PQ2_0` blocks to FP32.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_pq2_0_neon(blocks: &[BlockPQ2_0], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_PQ2_0;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let scale = vdupq_n_f32(block.d.to_f32());
        let base = bi * QK_PQ2_0;
        for byte_idx in 0..32 {
            let val_f = decode_byte_arith_to_f32x4(block.qs[byte_idx]);
            let result = vmulq_f32(scale, val_f);
            vst1q_f32(output.as_mut_ptr().add(base + byte_idx * 4), result);
        }
    }
    Ok(())
}

/// NEON-accelerated GEMV for a `PQ2_0`-quantized weight matrix.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_pq2_0_neon(
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
            let mut acc = vdupq_n_f32(0.0);
            for byte_idx in 0..32 {
                let val_f = decode_byte_arith_to_f32x4(block.qs[byte_idx]);
                let inp_vec = vld1q_f32(input.as_ptr().add(inp_base + byte_idx * 4));
                acc = vfmaq_f32(acc, val_f, inp_vec);
            }
            row_sum += block.d.to_f32() * hsum4_neon(acc);
        }
        output[row] = row_sum;
    }
    Ok(())
}

/// NEON-accelerated GEMM for a `PQ2_0`-quantized weight matrix.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_pq2_0_neon(
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
        gemv_pq2_0_neon(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Q2_0_g64 (mainline ggml id 42, group 64)
// ---------------------------------------------------------------------------

/// NEON-accelerated dequantization of `Q2_0_g64` blocks to FP32.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_q2_0_g64_neon(
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
        let scale = vdupq_n_f32(block.d.to_f32());
        let base = bi * QK_Q2_0_G64;
        for byte_idx in 0..16 {
            let val_f = decode_byte_arith_to_f32x4(block.qs[byte_idx]);
            let result = vmulq_f32(scale, val_f);
            vst1q_f32(output.as_mut_ptr().add(base + byte_idx * 4), result);
        }
    }
    Ok(())
}

/// NEON-accelerated GEMV for a `Q2_0_g64`-quantized weight matrix.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q2_0_g64_neon(
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
            let mut acc = vdupq_n_f32(0.0);
            for byte_idx in 0..16 {
                let val_f = decode_byte_arith_to_f32x4(block.qs[byte_idx]);
                let inp_vec = vld1q_f32(input.as_ptr().add(inp_base + byte_idx * 4));
                acc = vfmaq_f32(acc, val_f, inp_vec);
            }
            row_sum += block.d.to_f32() * hsum4_neon(acc);
        }
        output[row] = row_sum;
    }
    Ok(())
}

/// NEON-accelerated GEMM for a `Q2_0_g64`-quantized weight matrix.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_q2_0_g64_neon(
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
        gemv_q2_0_g64_neon(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PTQ1_0 (ggml id 143) — vectorized base-3 trit decode
// ---------------------------------------------------------------------------

/// Widen 16 `u8` lanes, multiply by `pow` (as `u16`), mask to the low byte
/// (== the `u8::wrapping_mul` result, since the widened product never
/// overflows 16 bits: `255 * 243 = 61965 < 65536`), multiply by 3, shift
/// right 8, narrow back to `u8`. This is [`crate::dequant_prism::trit_of`]'s
/// scalar `((byte.wrapping_mul(pow) as u16) * 3) >> 8` applied to 16 lanes
/// at once — see the module doc for why the widen-first order preserves the
/// wrap exactly.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn trit_of_u8x16(q: uint8x16_t, pow: u8) -> uint8x16_t {
    let pow16 = vdupq_n_u16(pow as u16);
    let mask_ff = vdupq_n_u16(0x00FF);
    let three = vdupq_n_u16(3);

    let lo16 = vmovl_u8(vget_low_u8(q));
    let hi16 = vmovl_u8(vget_high_u8(q));

    let lo_t = vandq_u16(vmulq_u16(lo16, pow16), mask_ff);
    let hi_t = vandq_u16(vmulq_u16(hi16, pow16), mask_ff);

    let lo_code16 = vshrq_n_u16::<8>(vmulq_u16(lo_t, three));
    let hi_code16 = vshrq_n_u16::<8>(vmulq_u16(hi_t, three));

    vcombine_u8(vmovn_u16(lo_code16), vmovn_u16(hi_code16))
}

/// The 8-lane twin of [`trit_of_u8x16`], for the second (`c = 8`) stage.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn trit_of_u8x8(q: uint8x8_t, pow: u8) -> uint8x8_t {
    let pow16 = vdupq_n_u16(pow as u16);
    let mask_ff = vdupq_n_u16(0x00FF);
    let three = vdupq_n_u16(3);

    let q16 = vmovl_u8(q);
    let t = vandq_u16(vmulq_u16(q16, pow16), mask_ff);
    let code16 = vshrq_n_u16::<8>(vmulq_u16(t, three));
    vmovn_u16(code16)
}

/// NEON re-implementation of
/// [`crate::dequant_prism::decode_ptq1_0_codes`], vectorizing both effective
/// stages (`c = 16` over `qs[0..16]`, `c = 8` over `qs[16..24]`); the 8-value
/// `qh` tail is cheap enough to stay scalar. Must — and, per the bitwise
/// tests below, does — match the scalar decoder exactly.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_ptq1_0_codes_neon(block: &BlockPTQ1_0) -> [u8; QK_PTQ1_0] {
    let mut codes = [0u8; QK_PTQ1_0];

    let stage1 = vld1q_u8(block.qs.as_ptr()); // bytes 0..16
    for (n, &pow) in POW3.iter().take(5).enumerate() {
        let out16 = trit_of_u8x16(stage1, pow);
        vst1q_u8(codes.as_mut_ptr().add(n * 16), out16);
    }

    let stage2 = vld1_u8(block.qs.as_ptr().add(16)); // bytes 16..24
    for (n, &pow) in POW3.iter().take(5).enumerate() {
        let out8 = trit_of_u8x8(stage2, pow);
        vst1_u8(codes.as_mut_ptr().add(80 + n * 8), out8);
    }

    let mut w = 120usize;
    for &pow in POW3.iter().take(4) {
        for &byte in block.qh.iter() {
            let q = byte.wrapping_mul(pow);
            codes[w] = (((q as u16) * 3) >> 8) as u8;
            w += 1;
        }
    }
    debug_assert_eq!(w, QK_PTQ1_0);
    codes
}

/// Convert 16 trit codes (`u8`, values `0..=2`) into four `[f32; 4]` NEON
/// vectors of `code - 1`, in original order (`out[0]` = codes `0..4`, ...,
/// `out[3]` = codes `12..16`).
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn codes16_to_f32x4x4(codes16: uint8x16_t) -> [float32x4_t; 4] {
    let one = vdupq_n_s32(1);
    let lo16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(codes16)));
    let hi16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(codes16)));

    let a = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(lo16)), one));
    let b = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(lo16)), one));
    let c = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_low_s16(hi16)), one));
    let d = vcvtq_f32_s32(vsubq_s32(vmovl_s16(vget_high_s16(hi16)), one));
    [a, b, c, d]
}

/// NEON-accelerated dequantization of `PTQ1_0` blocks to FP32.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_ptq1_0_neon(blocks: &[BlockPTQ1_0], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_PTQ1_0;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let codes = decode_ptq1_0_codes_neon(block);
        let scale = vdupq_n_f32(block.d.to_f32());
        let base = bi * QK_PTQ1_0;
        for chunk in 0..8 {
            let codes16 = vld1q_u8(codes.as_ptr().add(chunk * 16));
            let vals = codes16_to_f32x4x4(codes16);
            for (i, v) in vals.iter().enumerate() {
                let result = vmulq_f32(scale, *v);
                vst1q_f32(output.as_mut_ptr().add(base + chunk * 16 + i * 4), result);
            }
        }
    }
    Ok(())
}

/// NEON-accelerated native GEMV for a `PTQ1_0`-quantized weight matrix.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_ptq1_0_neon(
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
            let codes = decode_ptq1_0_codes_neon(block);
            let inp_base = bi * QK_PTQ1_0;
            let mut acc = vdupq_n_f32(0.0);
            for chunk in 0..8 {
                let codes16 = vld1q_u8(codes.as_ptr().add(chunk * 16));
                let vals = codes16_to_f32x4x4(codes16);
                for (i, v) in vals.iter().enumerate() {
                    let inp_vec = vld1q_f32(input.as_ptr().add(inp_base + chunk * 16 + i * 4));
                    acc = vfmaq_f32(acc, *v, inp_vec);
                }
            }
            row_sum += block.d.to_f32() * hsum4_neon(acc);
        }
        output[row] = row_sum;
    }
    Ok(())
}

/// NEON-accelerated GEMM for a `PTQ1_0`-quantized weight matrix.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_ptq1_0_neon(
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
        gemv_ptq1_0_neon(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — module name already contains "prism" (matches the package gate's
// `prism` test-name filter without any extra renaming).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn has_neon() -> bool {
        cfg!(target_arch = "aarch64")
    }

    fn make_pq2_block(scale: f32, qs: [u8; 32]) -> BlockPQ2_0 {
        BlockPQ2_0 {
            d: f16::from_f32(scale),
            qs,
        }
    }

    fn make_g64_block(scale: f32, qs: [u8; 16]) -> BlockQ2_0G64 {
        BlockQ2_0G64 {
            d: f16::from_f32(scale),
            qs,
        }
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

    // ── PQ2_0 / Q2_0_g64: NEON must be BITWISE equal to reference on codes ──

    #[test]
    fn neon_pq2_0_dequant_matches_reference_bitwise_over_random_raw_bytes() {
        if !has_neon() {
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
        let mut neon = vec![0.0f32; blocks.len() * QK_PQ2_0];
        unsafe {
            dequant_pq2_0_neon(&blocks, &mut neon).expect("neon");
        }
        assert_eq!(reference, neon, "NEON PQ2_0 dequant must be bitwise-exact");
    }

    #[test]
    fn neon_q2_0_g64_dequant_matches_reference_bitwise_over_random_raw_bytes() {
        if !has_neon() {
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
        let mut neon = vec![0.0f32; blocks.len() * QK_Q2_0_G64];
        unsafe {
            dequant_q2_0_g64_neon(&blocks, &mut neon).expect("neon");
        }
        assert_eq!(
            reference, neon,
            "NEON Q2_0_g64 dequant must be bitwise-exact"
        );
    }

    #[test]
    fn neon_pq2_0_gemv_matches_reference_gemv() {
        if !has_neon() {
            return;
        }
        // n_rows=1, k=256 -> blocks_per_row=2, so exactly 2 blocks are needed.
        let blocks = vec![
            make_pq2_block(1.25, [0xFF; 32]),
            make_pq2_block(0.5, [0x1B; 32]),
        ];
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let mut reference = vec![0.0f32; 1];
        crate::dequant_prism::gemv_pq2_0(&blocks, &input, &mut reference, 1, 256).expect("ref");
        let mut neon = vec![0.0f32; 1];
        unsafe {
            gemv_pq2_0_neon(&blocks, &input, &mut neon, 1, 256).expect("neon");
        }
        assert!((reference[0] - neon[0]).abs() < 1e-3);
    }

    #[test]
    fn neon_pq2_0_gemm_matches_reference_gemm() {
        if !has_neon() {
            return;
        }
        let blocks = vec![make_pq2_block(1.0, [0xAA; 32]); 2];
        let m = 3;
        let n_rows = 2;
        let k = 128;
        let input = vec![1.0f32; m * k];
        let mut reference = vec![0.0f32; m * n_rows];
        crate::dequant_prism::gemm_pq2_0(&blocks, &input, &mut reference, m, n_rows, k)
            .expect("ref");
        let mut neon = vec![0.0f32; m * n_rows];
        unsafe {
            gemm_pq2_0_neon(&blocks, &input, &mut neon, m, n_rows, k).expect("neon");
        }
        for i in 0..m * n_rows {
            assert!((reference[i] - neon[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn neon_q2_0_g64_gemv_matches_reference_gemv() {
        if !has_neon() {
            return;
        }
        let blocks = vec![make_g64_block(0.75, [0xE4; 16])];
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 16.0).collect();
        let mut reference = vec![0.0f32; 1];
        crate::dequant_prism::gemv_q2_0_g64(&blocks, &input, &mut reference, 1, 64).expect("ref");
        let mut neon = vec![0.0f32; 1];
        unsafe {
            gemv_q2_0_g64_neon(&blocks, &input, &mut neon, 1, 64).expect("neon");
        }
        assert!((reference[0] - neon[0]).abs() < 1e-3);
    }

    #[test]
    fn neon_dequant_reports_buffer_too_small() {
        if !has_neon() {
            return;
        }
        let blocks = vec![make_pq2_block(1.0, [0xAA; 32])];
        let mut output = vec![0.0f32; 0];
        let err = unsafe { dequant_pq2_0_neon(&blocks, &mut output) }.expect_err("must fail");
        assert_eq!(err.buffer_name(), Some("output"));
    }

    // ── PTQ1_0: the vectorized trit decode is the correctness-critical part ─

    /// Direct, in-crate comparison of the vectorized trit-code extraction
    /// against the scalar reference, over raw (non-quantized) byte values —
    /// exactly what "NEON == reference BITWISE on codes" means.
    #[test]
    fn neon_ptq1_0_code_extraction_matches_scalar_bitwise_over_full_byte_range() {
        if !has_neon() {
            return;
        }
        let mut state = 0x0BAD_F00Du32;
        for _ in 0..200 {
            let qs: [u8; 24] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let qh: [u8; 2] = core::array::from_fn(|_| lcg(&mut state) as u8);
            let block = make_ptq1_block(1.0, qs, qh);
            let scalar = crate::dequant_prism::decode_ptq1_0_codes(&block);
            let neon = unsafe { decode_ptq1_0_codes_neon(&block) };
            assert_eq!(scalar, neon, "trit codes diverge for qs={qs:?} qh={qh:?}");
        }
        // Edge bytes: 0x00 and 0xFF exercise both wrap extremes.
        for &b in &[0x00u8, 0xFFu8, 0x01, 0x80, 0x55, 0xAA] {
            let block = make_ptq1_block(1.0, [b; 24], [b; 2]);
            let scalar = crate::dequant_prism::decode_ptq1_0_codes(&block);
            let neon = unsafe { decode_ptq1_0_codes_neon(&block) };
            assert_eq!(scalar, neon, "diverge at uniform byte {b:#04x}");
        }
    }

    #[test]
    fn neon_ptq1_0_dequant_matches_reference_bitwise() {
        if !has_neon() {
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
        let mut neon = vec![0.0f32; blocks.len() * QK_PTQ1_0];
        unsafe {
            dequant_ptq1_0_neon(&blocks, &mut neon).expect("neon");
        }
        assert_eq!(reference, neon, "NEON PTQ1_0 dequant must be bitwise-exact");
    }

    #[test]
    fn neon_ptq1_0_gemv_matches_reference_gemv() {
        if !has_neon() {
            return;
        }
        let block = make_ptq1_block(1.0, [0x9C; 24], [0x33; 2]);
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
        let mut reference = vec![0.0f32; 1];
        crate::gemv_ptq1::gemv_ptq1_0(&[block], &input, &mut reference, 1, QK_PTQ1_0).expect("ref");
        let mut neon = vec![0.0f32; 1];
        unsafe {
            gemv_ptq1_0_neon(&[block], &input, &mut neon, 1, QK_PTQ1_0).expect("neon");
        }
        assert!((reference[0] - neon[0]).abs() < 1e-3);
    }

    #[test]
    fn neon_ptq1_0_gemm_matches_reference_gemm() {
        if !has_neon() {
            return;
        }
        let blocks = vec![make_ptq1_block(1.0, [0x71; 24], [0x22; 2]); 2];
        let m = 2;
        let n_rows = 2;
        let k = QK_PTQ1_0;
        let input = vec![0.5f32; m * k];
        let mut reference = vec![0.0f32; m * n_rows];
        crate::gemv_ptq1::gemm_ptq1_0(&blocks, &input, &mut reference, m, n_rows, k).expect("ref");
        let mut neon = vec![0.0f32; m * n_rows];
        unsafe {
            gemm_ptq1_0_neon(&blocks, &input, &mut neon, m, n_rows, k).expect("neon");
        }
        for i in 0..m * n_rows {
            assert!((reference[i] - neon[i]).abs() < 1e-3);
        }
    }
}
