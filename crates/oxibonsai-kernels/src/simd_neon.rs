//! NEON-optimized 1-bit compute kernels for Q1\_0\_g128.
//!
//! Uses 128-bit NEON intrinsics to accelerate dequantization and
//! fused matrix-vector/matrix-matrix products for Q1\_0\_g128 format.
//!
//! **Core insight:** 1-bit weights mean each weight is `+scale` or `-scale`.
//! We load 4 f32 values at a time, broadcast the scale, then use bitwise
//! operations on the weight bits to create sign masks, converting the
//! inner loop to SIMD add/subtract operations.
//!
//! **NEON vs AVX2:** NEON uses 128-bit registers (4 f32 lanes) compared to
//! AVX2's 256-bit registers (8 f32 lanes). Each byte yields 2 NEON iterations
//! (bits 0-3, then bits 4-7), giving 32 iterations per 128-element block.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

#[cfg(target_arch = "aarch64")]
use crate::error::{KernelError, KernelResult};

// --- NEON Dequantization ---

/// NEON-accelerated dequantization of Q1\_0\_g128 blocks to FP32.
///
/// Processes 4 elements per SIMD iteration (128-bit / 32-bit = 4 lanes).
/// Each byte of weight bits produces 2 iterations (low 4 bits, high 4 bits),
/// giving 32 iterations per 128-element block.
///
/// # Safety
/// Requires NEON CPU support. This is always available on AArch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_1bit_g128_neon(
    blocks: &[BlockQ1_0G128],
    output: &mut [f32],
) -> KernelResult<()> {
    let expected_len = blocks.len() * QK1_0_G128;
    if output.len() < expected_len {
        return Err(KernelError::BufferTooSmall {
            needed: expected_len,
            available: output.len(),
        });
    }

    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let scale = vdupq_n_f32(d);
        let base = i * QK1_0_G128;

        // Process 128 elements: 16 bytes, 2 NEON iterations per byte = 32 iterations
        for byte_idx in 0..16 {
            let bits = block.qs[byte_idx];
            let out_base = base + byte_idx * 8;

            // Low 4 bits (lanes 0-3)
            let signs_lo = bits_to_signs_neon(bits, 0);
            let result_lo = vmulq_f32(scale, signs_lo);
            vst1q_f32(output.as_mut_ptr().add(out_base), result_lo);

            // High 4 bits (lanes 4-7)
            let signs_hi = bits_to_signs_neon(bits, 4);
            let result_hi = vmulq_f32(scale, signs_hi);
            vst1q_f32(output.as_mut_ptr().add(out_base + 4), result_hi);
        }
    }

    Ok(())
}

// --- NEON GEMV ---

/// NEON-accelerated 1-bit GEMV: `output[row] = dot(weight_row, input)`.
///
/// Uses FMA to accumulate dot products in 4-wide f32 registers.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_1bit_g128_neon(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut row_acc = vdupq_n_f32(0.0);

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let scale = vdupq_n_f32(d);
            let input_base = bi * QK1_0_G128;

            // Process 128 elements: 16 bytes, 2 NEON iterations per byte
            for byte_idx in 0..16 {
                let bits = block.qs[byte_idx];
                let inp_base = input_base + byte_idx * 8;

                // Low 4 bits
                let inp_lo = vld1q_f32(input.as_ptr().add(inp_base));
                let signs_lo = bits_to_signs_neon(bits, 0);
                let signed_lo = vmulq_f32(signs_lo, inp_lo);
                row_acc = vfmaq_f32(row_acc, scale, signed_lo);

                // High 4 bits
                let inp_hi = vld1q_f32(input.as_ptr().add(inp_base + 4));
                let signs_hi = bits_to_signs_neon(bits, 4);
                let signed_hi = vmulq_f32(signs_hi, inp_hi);
                row_acc = vfmaq_f32(row_acc, scale, signed_hi);
            }
        }

        output[row] = hsum_neon(row_acc);
    }

    Ok(())
}

/// NEON-accelerated 1-bit GEMM.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_1bit_g128_neon(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: input.len(),
        });
    }
    if output.len() < m * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: m * n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    for mi in 0..m {
        let input_row = &input[mi * k..];

        for ni in 0..n_rows {
            let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
            let mut acc = vdupq_n_f32(0.0);

            for (bi, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let scale = vdupq_n_f32(d);
                let input_base = bi * QK1_0_G128;

                for byte_idx in 0..16 {
                    let bits = block.qs[byte_idx];
                    let inp_base = input_base + byte_idx * 8;

                    // Low 4 bits
                    let inp_lo = vld1q_f32(input_row.as_ptr().add(inp_base));
                    let signs_lo = bits_to_signs_neon(bits, 0);
                    let signed_lo = vmulq_f32(signs_lo, inp_lo);
                    acc = vfmaq_f32(acc, scale, signed_lo);

                    // High 4 bits
                    let inp_hi = vld1q_f32(input_row.as_ptr().add(inp_base + 4));
                    let signs_hi = bits_to_signs_neon(bits, 4);
                    let signed_hi = vmulq_f32(signs_hi, inp_hi);
                    acc = vfmaq_f32(acc, scale, signed_hi);
                }
            }

            output[mi * n_rows + ni] = hsum_neon(acc);
        }
    }

    Ok(())
}

// --- Helpers ---

/// Convert 4 weight bits to a 128-bit sign vector: bit=1 -> +1.0, bit=0 -> -1.0.
///
/// Extracts 4 consecutive bits starting at `lane_offset` (0 or 4) from the byte.
/// Uses integer SIMD: expand bits to 32-bit masks, then blend +1/-1.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn bits_to_signs_neon(bits: u8, lane_offset: usize) -> float32x4_t {
    // Extract each bit to a u32 (0 or 1)
    let b0 = ((bits >> lane_offset) & 1) as u32;
    let b1 = ((bits >> (lane_offset + 1)) & 1) as u32;
    let b2 = ((bits >> (lane_offset + 2)) & 1) as u32;
    let b3 = ((bits >> (lane_offset + 3)) & 1) as u32;

    // Build uint32x4_t with extracted bits
    let mut bit_vec = vdupq_n_u32(0);
    bit_vec = vsetq_lane_u32::<0>(b0, bit_vec);
    bit_vec = vsetq_lane_u32::<1>(b1, bit_vec);
    bit_vec = vsetq_lane_u32::<2>(b2, bit_vec);
    bit_vec = vsetq_lane_u32::<3>(b3, bit_vec);

    // Compare with 1: bit=1 -> 0xFFFFFFFF, bit=0 -> 0x00000000
    let ones = vdupq_n_u32(1);
    let mask = vceqq_u32(bit_vec, ones);

    // Use vbslq_f32: where mask=0xFFFF pick +1.0, else pick -1.0
    let pos_one = vdupq_n_f32(1.0);
    let neg_one = vdupq_n_f32(-1.0);
    vbslq_f32(mask, pos_one, neg_one)
}

/// Horizontal sum of 4 f32 lanes in a NEON register.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    // vpaddq_f32 does pairwise addition: [a0+a1, a2+a3, a0+a1, a2+a3]
    let pair = vpaddq_f32(v, v);
    // Another pairwise add: [(a0+a1)+(a2+a3), ...]
    let sum = vpaddq_f32(pair, pair);
    vgetq_lane_f32::<0>(sum)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn decode_byte_neon_to_f32x4(b: u8, one: uint32x4_t) -> float32x4_t {
    let idx_arr: [u32; 4] = [
        (b & 3) as u32,
        ((b >> 2) & 3) as u32,
        ((b >> 4) & 3) as u32,
        ((b >> 6) & 3) as u32,
    ];
    let idx_v = vld1q_u32(idx_arr.as_ptr());
    let pos_part = vshrq_n_u32::<1>(idx_v);
    let min_part = vminq_u32(idx_v, one);
    let neg_part = vsubq_u32(one, min_part);
    let val_i32 = vsubq_s32(
        vreinterpretq_s32_u32(pos_part),
        vreinterpretq_s32_u32(neg_part),
    );
    let reserved = vceqq_u32(idx_v, vdupq_n_u32(3));
    vcvtq_f32_s32(vbslq_s32(reserved, vdupq_n_s32(0), val_i32))
}

// ─── Prefetch-optimized GEMV ────────────────────────────────────────────

/// NEON-accelerated 1-bit GEMV with software prefetch hints.
///
/// While processing the current row's blocks, prefetches the next row's
/// block data into L1 cache. This overlaps memory latency with computation,
/// improving throughput for large matrices where rows don't fit in cache.
///
/// Additionally unrolls the inner byte loop to process 2 bytes (16 elements)
/// per iteration where possible, improving instruction-level parallelism.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_1bit_g128_neon_prefetch(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];

        // Prefetch next row's first block into L1 cache
        if row + 1 < n_rows {
            let next_row_ptr = blocks.as_ptr().add((row + 1) * blocks_per_row) as *const i8;
            // SAFETY: prefetch is a hint; macro degrades to a no-op off-nightly.
            crate::aarch64_prefetch!(next_row_ptr, 0, 3);
        }

        // Use two accumulators for better ILP
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let scale = vdupq_n_f32(d);
            let input_base = bi * QK1_0_G128;

            // Prefetch next block's data
            if bi + 1 < blocks_per_row {
                let next_block_ptr = row_blocks.as_ptr().add(bi + 1) as *const i8;
                // SAFETY: prefetch is a hint; macro degrades to a no-op off-nightly.
                crate::aarch64_prefetch!(next_block_ptr, 0, 3);
            }

            // Unrolled: process 2 bytes (16 elements) per iteration
            let mut byte_idx = 0;
            while byte_idx + 1 < 16 {
                let bits0 = block.qs[byte_idx];
                let bits1 = block.qs[byte_idx + 1];
                let inp_base0 = input_base + byte_idx * 8;
                let inp_base1 = input_base + (byte_idx + 1) * 8;

                // Byte 0, low 4 bits
                let inp0_lo = vld1q_f32(input.as_ptr().add(inp_base0));
                let signs0_lo = bits_to_signs_neon(bits0, 0);
                let signed0_lo = vmulq_f32(signs0_lo, inp0_lo);
                acc0 = vfmaq_f32(acc0, scale, signed0_lo);

                // Byte 0, high 4 bits
                let inp0_hi = vld1q_f32(input.as_ptr().add(inp_base0 + 4));
                let signs0_hi = bits_to_signs_neon(bits0, 4);
                let signed0_hi = vmulq_f32(signs0_hi, inp0_hi);
                acc1 = vfmaq_f32(acc1, scale, signed0_hi);

                // Byte 1, low 4 bits
                let inp1_lo = vld1q_f32(input.as_ptr().add(inp_base1));
                let signs1_lo = bits_to_signs_neon(bits1, 0);
                let signed1_lo = vmulq_f32(signs1_lo, inp1_lo);
                acc0 = vfmaq_f32(acc0, scale, signed1_lo);

                // Byte 1, high 4 bits
                let inp1_hi = vld1q_f32(input.as_ptr().add(inp_base1 + 4));
                let signs1_hi = bits_to_signs_neon(bits1, 4);
                let signed1_hi = vmulq_f32(signs1_hi, inp1_hi);
                acc1 = vfmaq_f32(acc1, scale, signed1_hi);

                byte_idx += 2;
            }

            // Handle remaining byte if 16 is odd (it isn't, but be safe)
            while byte_idx < 16 {
                let bits = block.qs[byte_idx];
                let inp_base = input_base + byte_idx * 8;

                let inp_lo = vld1q_f32(input.as_ptr().add(inp_base));
                let signs_lo = bits_to_signs_neon(bits, 0);
                let signed_lo = vmulq_f32(signs_lo, inp_lo);
                acc0 = vfmaq_f32(acc0, scale, signed_lo);

                let inp_hi = vld1q_f32(input.as_ptr().add(inp_base + 4));
                let signs_hi = bits_to_signs_neon(bits, 4);
                let signed_hi = vmulq_f32(signs_hi, inp_hi);
                acc1 = vfmaq_f32(acc1, scale, signed_hi);

                byte_idx += 1;
            }
        }

        // Merge the two accumulators
        let combined = vaddq_f32(acc0, acc1);
        output[row] = hsum_neon(combined);
    }

    Ok(())
}

/// NEON-accelerated 1-bit GEMM with prefetch and double-buffered accumulators.
///
/// Processes multiple rows with prefetch hints for the next row's weight data.
/// Uses two separate accumulator registers to maximize NEON FMA throughput
/// by reducing data dependencies.
///
/// # Safety
/// Requires NEON CPU support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_1bit_g128_neon_prefetch(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: input.len(),
        });
    }
    if output.len() < m * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: m * n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    for mi in 0..m {
        let input_row = &input[mi * k..];

        for ni in 0..n_rows {
            let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];

            // Prefetch next weight row
            if ni + 1 < n_rows {
                let next_ptr = blocks.as_ptr().add((ni + 1) * blocks_per_row) as *const i8;
                // SAFETY: prefetch is a hint; macro degrades to a no-op off-nightly.
                crate::aarch64_prefetch!(next_ptr, 0, 3);
            }

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);

            for (bi, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let scale = vdupq_n_f32(d);
                let input_base = bi * QK1_0_G128;

                // Prefetch next block
                if bi + 1 < blocks_per_row {
                    let next_block = row_blocks.as_ptr().add(bi + 1) as *const i8;
                    // SAFETY: prefetch is a hint; macro degrades to a no-op off-nightly.
                    crate::aarch64_prefetch!(next_block, 0, 3);
                }

                // Unrolled: 2 bytes per iteration
                let mut byte_idx = 0;
                while byte_idx + 1 < 16 {
                    let bits0 = block.qs[byte_idx];
                    let bits1 = block.qs[byte_idx + 1];
                    let ib0 = input_base + byte_idx * 8;
                    let ib1 = input_base + (byte_idx + 1) * 8;

                    // Byte 0
                    let i0_lo = vld1q_f32(input_row.as_ptr().add(ib0));
                    let s0_lo = bits_to_signs_neon(bits0, 0);
                    acc0 = vfmaq_f32(acc0, scale, vmulq_f32(s0_lo, i0_lo));

                    let i0_hi = vld1q_f32(input_row.as_ptr().add(ib0 + 4));
                    let s0_hi = bits_to_signs_neon(bits0, 4);
                    acc1 = vfmaq_f32(acc1, scale, vmulq_f32(s0_hi, i0_hi));

                    // Byte 1
                    let i1_lo = vld1q_f32(input_row.as_ptr().add(ib1));
                    let s1_lo = bits_to_signs_neon(bits1, 0);
                    acc0 = vfmaq_f32(acc0, scale, vmulq_f32(s1_lo, i1_lo));

                    let i1_hi = vld1q_f32(input_row.as_ptr().add(ib1 + 4));
                    let s1_hi = bits_to_signs_neon(bits1, 4);
                    acc1 = vfmaq_f32(acc1, scale, vmulq_f32(s1_hi, i1_hi));

                    byte_idx += 2;
                }

                // Handle remainder (if 16 is odd — it isn't)
                while byte_idx < 16 {
                    let bits = block.qs[byte_idx];
                    let ib = input_base + byte_idx * 8;

                    let i_lo = vld1q_f32(input_row.as_ptr().add(ib));
                    let s_lo = bits_to_signs_neon(bits, 0);
                    acc0 = vfmaq_f32(acc0, scale, vmulq_f32(s_lo, i_lo));

                    let i_hi = vld1q_f32(input_row.as_ptr().add(ib + 4));
                    let s_hi = bits_to_signs_neon(bits, 4);
                    acc1 = vfmaq_f32(acc1, scale, vmulq_f32(s_hi, i_hi));

                    byte_idx += 1;
                }
            }

            let combined = vaddq_f32(acc0, acc1);
            output[mi * n_rows + ni] = hsum_neon(combined);
        }
    }

    Ok(())
}

// ─── NEON Ternary TQ2_0_g128 Kernels ─────────────────────────────────────

/// NEON-accelerated dequantization of TQ2\_0\_g128 blocks to FP32.
///
/// Decodes 2-bit ternary codes (4 per byte, 32 bytes per block = 128 weights)
/// and scales by the block's FP16 scale factor. Processes 4 weights per
/// NEON iteration (one byte decoded to `[f32; 4]` then scaled).
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dequant_tq2_0_g128_neon(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    output: &mut [f32],
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;
    let needed = blocks.len() * QK_TQ2_0_G128;
    if output.len() < needed {
        return Err(KernelError::BufferTooSmall {
            needed,
            available: output.len(),
        });
    }

    let one = vdupq_n_u32(1);

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let scale = vdupq_n_f32(d);
        let base = bi * QK_TQ2_0_G128;

        // 32 bytes × 4 weights = 128 weights; process 1 byte (4 weights) per NEON iteration
        // 32 iterations per block
        for byte_idx in 0..32 {
            let b = block.qs[byte_idx];
            let val_f = decode_byte_neon_to_f32x4(b, one);

            let result = vmulq_f32(scale, val_f);
            vst1q_f32(output.as_mut_ptr().add(base + byte_idx * 4), result);
        }
    }

    Ok(())
}

/// NEON-accelerated GEMV for TQ2\_0\_g128-quantized weight matrices.
///
/// Computes `output[row] = dot(weight_row, input)` for each row.
/// Decodes 4 weights per iteration (one byte → `[f32; 4]`), loads with
/// `vld1q_f32`, and accumulates with `vfmaq_f32`.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_tq2_0_g128_neon(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;

    if k % QK_TQ2_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    let one = vdupq_n_u32(1);

    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut row_sum = 0.0_f32;

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let inp_base = bi * QK_TQ2_0_G128;
            let mut block_acc = vdupq_n_f32(0.0_f32);

            // 32 iterations × 4 weights = 128 weights per block
            // Each iteration consumes 1 byte of qs
            for byte_idx in 0..32 {
                let b = block.qs[byte_idx];
                let val_f = decode_byte_neon_to_f32x4(b, one);
                let inp_vec = vld1q_f32(input.as_ptr().add(inp_base + byte_idx * 4));
                block_acc = vfmaq_f32(block_acc, val_f, inp_vec);
            }

            row_sum += d * hsum_neon(block_acc);
        }

        output[row] = row_sum;
    }

    Ok(())
}

/// NEON-accelerated GEMV for TQ2\_0\_g128 with prefetch and 2-byte unroll.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_tq2_0_g128_neon_prefetch(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;

    if k % QK_TQ2_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    let one = vdupq_n_u32(1);

    for (row, out) in output.iter_mut().enumerate().take(n_rows) {
        let row_offset = row * blocks_per_row;

        if row + 1 < n_rows {
            let next_row_ptr = blocks.as_ptr().add((row + 1) * blocks_per_row) as *const i8;
            // SAFETY: prefetch is a hint; macro degrades to a no-op off-nightly.
            crate::aarch64_prefetch!(next_row_ptr, 0, 3);
        }

        let mut row_sum = 0.0f32;
        for block_idx in 0..blocks_per_row {
            if block_idx + 1 < blocks_per_row {
                let next_block_ptr = blocks.as_ptr().add(row_offset + block_idx + 1) as *const i8;
                // SAFETY: prefetch is a hint; macro degrades to a no-op off-nightly.
                crate::aarch64_prefetch!(next_block_ptr, 0, 3);
            }

            let block = &blocks[row_offset + block_idx];
            let d = block.d.to_f32();
            let inp_base = block_idx * QK_TQ2_0_G128;
            let mut acc0 = vdupq_n_f32(0.0f32);
            let mut acc1 = vdupq_n_f32(0.0f32);

            for pair in 0..16 {
                let b0 = block.qs[pair * 2];
                let b1 = block.qs[pair * 2 + 1];

                let val0 = decode_byte_neon_to_f32x4(b0, one);
                let inp0 = vld1q_f32(input.as_ptr().add(inp_base + pair * 8));
                acc0 = vfmaq_f32(acc0, val0, inp0);

                let val1 = decode_byte_neon_to_f32x4(b1, one);
                let inp1 = vld1q_f32(input.as_ptr().add(inp_base + pair * 8 + 4));
                acc1 = vfmaq_f32(acc1, val1, inp1);
            }

            row_sum += d * hsum_neon(vaddq_f32(acc0, acc1));
        }

        *out = row_sum;
    }

    Ok(())
}

/// NEON-accelerated GEMM for TQ2\_0\_g128-quantized weight matrices.
///
/// Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])` for each (m, n) pair.
/// Iterates over batch dimension `m`, using per-row GEMV logic with NEON FMA.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemm_tq2_0_g128_neon(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;

    if k % QK_TQ2_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: input.len(),
        });
    }
    if output.len() < m * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: m * n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    let one = vdupq_n_u32(1);

    for mi in 0..m {
        let input_row = &input[mi * k..];

        for ni in 0..n_rows {
            let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
            let mut row_sum = 0.0_f32;

            for (bi, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let inp_base = bi * QK_TQ2_0_G128;
                let mut block_acc = vdupq_n_f32(0.0_f32);

                for byte_idx in 0..32 {
                    let b = block.qs[byte_idx];

                    let idx_arr: [u32; 4] = [
                        (b & 3) as u32,
                        ((b >> 2) & 3) as u32,
                        ((b >> 4) & 3) as u32,
                        ((b >> 6) & 3) as u32,
                    ];
                    let idx_v = vld1q_u32(idx_arr.as_ptr());
                    let pos_part = vshrq_n_u32::<1>(idx_v);
                    let min_part = vminq_u32(idx_v, one);
                    let neg_part = vsubq_u32(one, min_part);
                    let val_i32 = vsubq_s32(
                        vreinterpretq_s32_u32(pos_part),
                        vreinterpretq_s32_u32(neg_part),
                    );
                    let val_f = vcvtq_f32_s32(val_i32);

                    let inp_vec = vld1q_f32(input_row.as_ptr().add(inp_base + byte_idx * 4));
                    block_acc = vfmaq_f32(block_acc, val_f, inp_vec);
                }

                row_sum += d * hsum_neon(block_acc);
            }

            output[mi * n_rows + ni] = row_sum;
        }
    }

    Ok(())
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use half::f16;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn test_dequant_neon_all_positive() {
        let block = make_block(2.0, [0xFF; 16]);
        let mut output = vec![0.0f32; 128];
        unsafe {
            dequant_1bit_g128_neon(&[block], &mut output).expect("dequant should succeed");
        }
        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_neon_all_negative() {
        let block = make_block(3.0, [0x00; 16]);
        let mut output = vec![0.0f32; 128];
        unsafe {
            dequant_1bit_g128_neon(&[block], &mut output).expect("dequant should succeed");
        }
        for &v in &output {
            assert!((v + 3.0).abs() < 0.01, "expected -3.0, got {v}");
        }
    }

    #[test]
    fn test_dequant_neon_matches_reference() {
        let block = make_block(1.5, [0xAA; 16]); // alternating bits
        let mut out_ref = vec![0.0f32; 128];
        let mut out_neon = vec![0.0f32; 128];

        crate::dequant::dequant_1bit_g128(&[block], &mut out_ref)
            .expect("reference dequant should succeed");
        unsafe {
            dequant_1bit_g128_neon(&[block], &mut out_neon).expect("neon dequant should succeed");
        }
        for i in 0..128 {
            assert!(
                (out_ref[i] - out_neon[i]).abs() < 0.01,
                "mismatch at {i}: ref={}, neon={}",
                out_ref[i],
                out_neon[i]
            );
        }
    }

    #[test]
    fn test_gemv_neon_matches_reference() {
        // 4 rows, k=256 (2 blocks per row)
        let n_rows = 4;
        let k = 256;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::new();
        for row in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [row as u8 * 37 + bi as u8 * 13; 16];
                blocks.push(make_block(0.5 + row as f32 * 0.1, bits));
            }
        }
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_neon = vec![0.0f32; n_rows];

        crate::gemv::gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_1bit_g128_neon(&blocks, &input, &mut out_neon, n_rows, k)
                .expect("neon gemv should succeed");
        }

        for i in 0..n_rows {
            assert!(
                (out_ref[i] - out_neon[i]).abs() < 0.1,
                "row {i}: ref={}, neon={}",
                out_ref[i],
                out_neon[i]
            );
        }
    }

    #[test]
    fn test_gemm_neon_matches_reference() {
        let m = 2;
        let n_rows = 3;
        let k = 128;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::new();
        for ni in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [(ni as u8 * 17 + bi as u8 * 7) | 0x55; 16];
                blocks.push(make_block(1.0 + ni as f32 * 0.2, bits));
            }
        }
        let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.005) - 0.32).collect();

        let mut out_ref = vec![0.0f32; m * n_rows];
        let mut out_neon = vec![0.0f32; m * n_rows];

        crate::gemm::gemm_1bit_g128(&blocks, &input, &mut out_ref, m, n_rows, k)
            .expect("reference gemm should succeed");
        unsafe {
            gemm_1bit_g128_neon(&blocks, &input, &mut out_neon, m, n_rows, k)
                .expect("neon gemm should succeed");
        }

        for i in 0..(m * n_rows) {
            assert!(
                (out_ref[i] - out_neon[i]).abs() < 0.5,
                "idx {i}: ref={}, neon={}",
                out_ref[i],
                out_neon[i]
            );
        }
    }

    #[test]
    fn test_gemv_neon_prefetch_matches_reference() {
        let n_rows = 4;
        let k = 256;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::new();
        for row in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [row as u8 * 37 + bi as u8 * 13; 16];
                blocks.push(make_block(0.5 + row as f32 * 0.1, bits));
            }
        }
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_pf = vec![0.0f32; n_rows];

        crate::gemv::gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_1bit_g128_neon_prefetch(&blocks, &input, &mut out_pf, n_rows, k)
                .expect("neon prefetch gemv should succeed");
        }

        for i in 0..n_rows {
            assert!(
                (out_ref[i] - out_pf[i]).abs() < 0.1,
                "row {i}: ref={}, prefetch={}",
                out_ref[i],
                out_pf[i]
            );
        }
    }

    #[test]
    fn test_gemv_neon_prefetch_large() {
        let n_rows = 64;
        let k = 512;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::new();
        for row in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [((row * 23 + bi * 11) & 0xFF) as u8; 16];
                blocks.push(make_block(0.3 + row as f32 * 0.005, bits));
            }
        }
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.005) - 1.0).collect();

        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_pf = vec![0.0f32; n_rows];

        crate::gemv::gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_1bit_g128_neon_prefetch(&blocks, &input, &mut out_pf, n_rows, k)
                .expect("neon prefetch gemv should succeed");
        }

        for i in 0..n_rows {
            assert!(
                (out_ref[i] - out_pf[i]).abs() < 0.5,
                "row {i}: ref={}, prefetch={}",
                out_ref[i],
                out_pf[i]
            );
        }
    }

    #[test]
    fn test_gemm_neon_prefetch_matches_reference() {
        let m = 2;
        let n_rows = 3;
        let k = 128;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::new();
        for ni in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [(ni as u8 * 17 + bi as u8 * 7) | 0x55; 16];
                blocks.push(make_block(1.0 + ni as f32 * 0.2, bits));
            }
        }
        let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.005) - 0.32).collect();

        let mut out_ref = vec![0.0f32; m * n_rows];
        let mut out_pf = vec![0.0f32; m * n_rows];

        crate::gemm::gemm_1bit_g128(&blocks, &input, &mut out_ref, m, n_rows, k)
            .expect("reference gemm should succeed");
        unsafe {
            gemm_1bit_g128_neon_prefetch(&blocks, &input, &mut out_pf, m, n_rows, k)
                .expect("neon prefetch gemm should succeed");
        }

        for i in 0..(m * n_rows) {
            assert!(
                (out_ref[i] - out_pf[i]).abs() < 0.5,
                "idx {i}: ref={}, prefetch={}",
                out_ref[i],
                out_pf[i]
            );
        }
    }

    // ─── Ternary TQ2_0_g128 NEON tests ────────────────────────────────────

    fn make_ternary_block(scale: f32, qs: [u8; 32]) -> oxibonsai_core::BlockTQ2_0_g128 {
        oxibonsai_core::BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    /// qs=0xAA → all codes=0b10 (Pos=+1), d=1.0, input=[1.0;128] → output=[128.0;2]
    #[test]
    fn ternary_gemv_neon_matches_scalar() {
        // 0xAA = 0b10_10_10_10: all 4 codes per byte = 0b10 (Pos = +1)
        let blocks = vec![
            make_ternary_block(1.0, [0xAA; 32]),
            make_ternary_block(1.0, [0xAA; 32]),
        ];
        let input = vec![1.0_f32; 128];
        let mut out_neon = vec![0.0_f32; 2];
        let mut out_ref = vec![0.0_f32; 2];

        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, 2, 128)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_tq2_0_g128_neon(&blocks, &input, &mut out_neon, 2, 128)
                .expect("neon ternary gemv should succeed");
        }

        for i in 0..2 {
            assert!(
                (out_neon[i] - out_ref[i]).abs() < 1e-4,
                "row {i}: neon={}, ref={}",
                out_neon[i],
                out_ref[i]
            );
        }
    }

    /// Alternating +1/0/-1/0 per group of 4 weights; with input=[1.0;128], output should be 0.
    /// byte=0x46 (0b01_00_01_10): lane0=Pos(+1), lane1=Zero, lane2=Neg(-1), lane3=Zero → sum=0
    #[test]
    fn ternary_gemv_neon_sign_canary() {
        let blocks = vec![make_ternary_block(1.0, [0x46; 32])];
        let input = vec![1.0_f32; 128];
        let mut output = vec![99.0_f32; 1];

        unsafe {
            gemv_tq2_0_g128_neon(&blocks, &input, &mut output, 1, 128)
                .expect("neon ternary gemv sign canary should succeed");
        }

        assert!(
            output[0].abs() < 1e-4,
            "sign canary: expected 0.0, got {}",
            output[0]
        );
    }

    /// qs=0x00 → all codes=0b00 (Neg=-1), d=1.0, input=[1.0;128] → output=[-128.0;2]
    #[test]
    fn ternary_gemv_neon_all_negative() {
        let blocks = vec![
            make_ternary_block(1.0, [0x00; 32]),
            make_ternary_block(1.0, [0x00; 32]),
        ];
        let input = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 2];

        unsafe {
            gemv_tq2_0_g128_neon(&blocks, &input, &mut output, 2, 128)
                .expect("neon ternary all-negative gemv should succeed");
        }

        for (i, val) in output.iter().enumerate() {
            assert!(
                (val + 128.0).abs() < 1e-4,
                "row {i}: expected -128.0, got {}",
                val
            );
        }
    }

    #[test]
    fn gemv_tq2_0_g128_neon_prefetch_matches_reference() {
        let n_rows = 64;
        let k = 128;
        let blocks_per_row = k / oxibonsai_core::QK_TQ2_0_G128;
        let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);

        for row in 0..n_rows {
            for bi in 0..blocks_per_row {
                let mut qs = [0u8; 32];
                for (i, byte) in qs.iter_mut().enumerate() {
                    *byte = ((row * 17 + bi * 29 + i * 11) & 0xFF) as u8;
                }
                blocks.push(make_ternary_block(0.25 + row as f32 * 0.01, qs));
            }
        }

        let input: Vec<f32> = (0..k)
            .map(|i| ((i * 7 % 23) as f32 * 0.125) - 1.5)
            .collect();
        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_neon = vec![0.0f32; n_rows];

        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference ternary gemv should succeed");
        unsafe {
            gemv_tq2_0_g128_neon_prefetch(&blocks, &input, &mut out_neon, n_rows, k)
                .expect("neon prefetch ternary gemv should succeed");
        }

        let mse = out_ref
            .iter()
            .zip(&out_neon)
            .map(|(lhs, rhs)| {
                let diff = lhs - rhs;
                diff * diff
            })
            .sum::<f32>()
            / n_rows as f32;
        assert!(mse < 1e-6, "expected MSE < 1e-6, got {mse}");
    }
}
