//! AVX2-optimized 1-bit compute kernels for Q1\_0\_g128.
//!
//! Uses 256-bit AVX2/FMA intrinsics to accelerate dequantization and
//! fused matrix-vector/matrix-matrix products for Q1\_0\_g128 format.
//!
//! **Core insight:** 1-bit weights mean each weight is `+scale` or `-scale`.
//! We load 8 f32 values at a time, broadcast the scale, then use bitwise
//! operations on the weight bits to create sign masks, converting the
//! inner loop to SIMD add/subtract operations.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

#[cfg(target_arch = "x86_64")]
use crate::error::{KernelError, KernelResult};

// ─── AVX2 Dequantization ─────────────────────────────────────────────────

/// AVX2-accelerated dequantization of Q1\_0\_g128 blocks to FP32.
///
/// Processes 8 elements per SIMD iteration (256-bit / 32-bit = 8 lanes).
/// # Safety
/// Requires AVX2 CPU support. Caller must ensure the CPU has AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_1bit_g128_avx2(
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
        let scale = _mm256_set1_ps(d);
        let base = i * QK1_0_G128;

        // Process 128 elements in chunks of 8 (16 iterations per block)
        for chunk in 0..16 {
            let bits = block.qs[chunk];
            let out_offset = base + chunk * 8;

            // Create sign vector from bits: bit=1 → +1.0, bit=0 → -1.0
            let signs = bits_to_signs_avx2(bits);

            let result = _mm256_mul_ps(scale, signs);
            _mm256_storeu_ps(output.as_mut_ptr().add(out_offset), result);
        }
    }

    Ok(())
}

// ─── AVX2 GEMV ───────────────────────────────────────────────────────────

/// AVX2-accelerated 1-bit GEMV: `output[row] = dot(weight_row, input)`.
///
/// Uses FMA to accumulate dot products in 8-wide f32 registers.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_1bit_g128_avx2(
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
        let mut row_acc = _mm256_setzero_ps();

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let scale = _mm256_set1_ps(d);
            let input_base = bi * QK1_0_G128;

            // Process 128 elements in chunks of 8 (16 chunks per block)
            for chunk in 0..16 {
                let bits = block.qs[chunk];
                let inp_offset = input_base + chunk * 8;

                // Load 8 input values
                let inp = _mm256_loadu_ps(input.as_ptr().add(inp_offset));

                // Create sign mask: bit=1 → +1.0, bit=0 → -1.0
                // _mm256_set_ps is high-to-low order
                let signs = bits_to_signs_avx2(bits);

                // signed_input = signs * input
                let signed_input = _mm256_mul_ps(signs, inp);

                // row_acc += scale * signed_input (FMA)
                row_acc = _mm256_fmadd_ps(scale, signed_input, row_acc);
            }
        }

        // Horizontal sum of row_acc
        output[row] = hsum_avx2(row_acc);
    }

    Ok(())
}

/// AVX2-accelerated 1-bit GEMM.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_1bit_g128_avx2(
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
            let mut acc = _mm256_setzero_ps();

            for (bi, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let scale = _mm256_set1_ps(d);
                let input_base = bi * QK1_0_G128;

                for chunk in 0..16 {
                    let bits = block.qs[chunk];
                    let inp_offset = input_base + chunk * 8;

                    let inp = _mm256_loadu_ps(input_row.as_ptr().add(inp_offset));

                    let signs = bits_to_signs_avx2(bits);

                    let signed_input = _mm256_mul_ps(signs, inp);
                    acc = _mm256_fmadd_ps(scale, signed_input, acc);
                }
            }

            output[mi * n_rows + ni] = hsum_avx2(acc);
        }
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Convert 8 weight bits to a 256-bit sign vector: bit=1 → +1.0, bit=0 → -1.0.
///
/// Uses integer SIMD: expand bits to 32-bit masks, then blend +1/-1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bits_to_signs_avx2(bits: u8) -> __m256 {
    // Each lane gets the bit value (0 or 1) for its position
    // Bit 0 → lane 0, bit 1 → lane 1, ..., bit 7 → lane 7
    let bit_masks = _mm256_set_epi32(
        ((bits >> 7) & 1) as i32,
        ((bits >> 6) & 1) as i32,
        ((bits >> 5) & 1) as i32,
        ((bits >> 4) & 1) as i32,
        ((bits >> 3) & 1) as i32,
        ((bits >> 2) & 1) as i32,
        ((bits >> 1) & 1) as i32,
        (bits & 1) as i32,
    );

    // Compare each lane with zero: 0 → 0xFFFFFFFF, 1 → 0x00000000
    let zero = _mm256_setzero_si256();
    let is_zero = _mm256_cmpeq_epi32(bit_masks, zero);

    // Use blend: where bit=0 (is_zero=0xFFFF), pick -1.0; otherwise pick +1.0
    let pos_one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    _mm256_blendv_ps(pos_one, neg_one, _mm256_castsi256_ps(is_zero))
}

/// Horizontal sum of 8 f32 lanes in an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    // [a0 a1 a2 a3 a4 a5 a6 a7]
    let hi128 = _mm256_extractf128_ps(v, 1); // [a4 a5 a6 a7]
    let lo128 = _mm256_castps256_ps128(v); // [a0 a1 a2 a3]
    let sum128 = _mm_add_ps(lo128, hi128); // [a0+a4 a1+a5 a2+a6 a3+a7]
    let shuf = _mm_movehdup_ps(sum128); // [a1+a5 a1+a5 a3+a7 a3+a7]
    let sums = _mm_add_ps(sum128, shuf); // [a0+a1+a4+a5 _ a2+a3+a6+a7 _]
    let shuf2 = _mm_movehl_ps(sums, sums); // [a2+a3+a6+a7 _ _ _]
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn decode_2bytes_avx2_to_f32x8(b0: u8, b1: u8) -> __m256 {
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let one = _mm256_set1_epi32(1);
    let mask3 = _mm256_set1_epi32(3);
    let packed = _mm256_set1_epi32(((b0 as u32) | ((b1 as u32) << 8)) as i32);
    let shifted = _mm256_srlv_epi32(packed, shifts);
    let idx = _mm256_and_si256(shifted, mask3);
    let pos_part = _mm256_srli_epi32(idx, 1);
    let min_part = _mm256_min_epu32(idx, one);
    let neg_part = _mm256_sub_epi32(one, min_part);
    let val_i = _mm256_sub_epi32(pos_part, neg_part);
    let reserved = _mm256_cmpeq_epi32(idx, mask3);
    let val_i = _mm256_andnot_si256(reserved, val_i);
    _mm256_cvtepi32_ps(val_i)
}

// ─── Prefetch-optimized AVX2 GEMV ───────────────────────────────────────

/// AVX2-accelerated 1-bit GEMV with software prefetch hints.
///
/// Prefetches the next row's block data while processing the current row,
/// and unrolls the inner loop to process 2 blocks at a time for better ILP.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_1bit_g128_avx2_prefetch(
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

        // Prefetch next row's first block
        if row + 1 < n_rows {
            let next_ptr = blocks.as_ptr().add((row + 1) * blocks_per_row) as *const i8;
            _mm_prefetch(next_ptr, _MM_HINT_T0);
        }

        // Use two accumulators for ILP
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // Unrolled: process 2 blocks per iteration
        let pairs = blocks_per_row / 2;
        let remainder = blocks_per_row % 2;

        for pair_idx in 0..pairs {
            let bi0 = pair_idx * 2;
            let bi1 = bi0 + 1;
            let block0 = &row_blocks[bi0];
            let block1 = &row_blocks[bi1];

            // Prefetch next pair
            if bi1 + 1 < blocks_per_row {
                let next_ptr = row_blocks.as_ptr().add(bi1 + 1) as *const i8;
                _mm_prefetch(next_ptr, _MM_HINT_T0);
            }

            let d0 = block0.d.to_f32();
            let scale0 = _mm256_set1_ps(d0);
            let base0 = bi0 * QK1_0_G128;

            let d1 = block1.d.to_f32();
            let scale1 = _mm256_set1_ps(d1);
            let base1 = bi1 * QK1_0_G128;

            // Process both blocks' chunks interleaved
            for chunk in 0..16 {
                let bits0 = block0.qs[chunk];
                let offset0 = base0 + chunk * 8;
                let inp0 = _mm256_loadu_ps(input.as_ptr().add(offset0));
                let signs0 = bits_to_signs_avx2(bits0);
                let signed0 = _mm256_mul_ps(signs0, inp0);
                acc0 = _mm256_fmadd_ps(scale0, signed0, acc0);

                let bits1 = block1.qs[chunk];
                let offset1 = base1 + chunk * 8;
                let inp1 = _mm256_loadu_ps(input.as_ptr().add(offset1));
                let signs1 = bits_to_signs_avx2(bits1);
                let signed1 = _mm256_mul_ps(signs1, inp1);
                acc1 = _mm256_fmadd_ps(scale1, signed1, acc1);
            }
        }

        // Handle remaining block if odd count
        for (bi, block) in row_blocks
            .iter()
            .enumerate()
            .skip(pairs * 2)
            .take(remainder)
        {
            let d = block.d.to_f32();
            let scale = _mm256_set1_ps(d);
            let base = bi * QK1_0_G128;

            for chunk in 0..16 {
                let bits = block.qs[chunk];
                let offset = base + chunk * 8;
                let inp = _mm256_loadu_ps(input.as_ptr().add(offset));
                let signs = bits_to_signs_avx2(bits);
                let signed = _mm256_mul_ps(signs, inp);
                acc0 = _mm256_fmadd_ps(scale, signed, acc0);
            }
        }

        // Merge accumulators
        let combined = _mm256_add_ps(acc0, acc1);
        output[row] = hsum_avx2(combined);
    }

    Ok(())
}

/// AVX2-accelerated GEMV for TQ2\_0\_g128 with prefetch and 2-byte unroll.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_tq2_0_g128_avx2_prefetch(
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

    for (row, out) in output.iter_mut().enumerate().take(n_rows) {
        let row_offset = row * blocks_per_row;

        if row + 1 < n_rows {
            let next_ptr = blocks.as_ptr().add((row + 1) * blocks_per_row) as *const i8;
            _mm_prefetch(next_ptr, _MM_HINT_T0);
        }

        let mut row_sum = 0.0f32;
        for block_idx in 0..blocks_per_row {
            if block_idx + 1 < blocks_per_row {
                let next_ptr = blocks.as_ptr().add(row_offset + block_idx + 1) as *const i8;
                _mm_prefetch(next_ptr, _MM_HINT_T0);
            }

            let block = &blocks[row_offset + block_idx];
            let d = block.d.to_f32();
            let inp_base = block_idx * QK_TQ2_0_G128;
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();

            for pair in 0..8 {
                let base = pair * 4;

                let val0 = decode_2bytes_avx2_to_f32x8(block.qs[base], block.qs[base + 1]);
                let inp0 = _mm256_loadu_ps(input.as_ptr().add(inp_base + pair * 16));
                acc0 = _mm256_fmadd_ps(val0, inp0, acc0);

                let val1 = decode_2bytes_avx2_to_f32x8(block.qs[base + 2], block.qs[base + 3]);
                let inp1 = _mm256_loadu_ps(input.as_ptr().add(inp_base + pair * 16 + 8));
                acc1 = _mm256_fmadd_ps(val1, inp1, acc1);
            }

            row_sum += d * hsum_avx2(_mm256_add_ps(acc0, acc1));
        }

        *out = row_sum;
    }

    Ok(())
}

/// AVX2-accelerated 1-bit GEMM with prefetch hints.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_1bit_g128_avx2_prefetch(
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
                _mm_prefetch(next_ptr, _MM_HINT_T0);
            }

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();

            let pairs = blocks_per_row / 2;
            let remainder = blocks_per_row % 2;

            for pair_idx in 0..pairs {
                let bi0 = pair_idx * 2;
                let bi1 = bi0 + 1;
                let block0 = &row_blocks[bi0];
                let block1 = &row_blocks[bi1];

                let d0 = block0.d.to_f32();
                let scale0 = _mm256_set1_ps(d0);
                let base0 = bi0 * QK1_0_G128;

                let d1 = block1.d.to_f32();
                let scale1 = _mm256_set1_ps(d1);
                let base1 = bi1 * QK1_0_G128;

                for chunk in 0..16 {
                    let bits0 = block0.qs[chunk];
                    let inp0 = _mm256_loadu_ps(input_row.as_ptr().add(base0 + chunk * 8));
                    let signs0 = bits_to_signs_avx2(bits0);
                    acc0 = _mm256_fmadd_ps(scale0, _mm256_mul_ps(signs0, inp0), acc0);

                    let bits1 = block1.qs[chunk];
                    let inp1 = _mm256_loadu_ps(input_row.as_ptr().add(base1 + chunk * 8));
                    let signs1 = bits_to_signs_avx2(bits1);
                    acc1 = _mm256_fmadd_ps(scale1, _mm256_mul_ps(signs1, inp1), acc1);
                }
            }

            for (bi, block) in row_blocks
                .iter()
                .enumerate()
                .skip(pairs * 2)
                .take(remainder)
            {
                let d = block.d.to_f32();
                let scale = _mm256_set1_ps(d);
                let base = bi * QK1_0_G128;

                for chunk in 0..16 {
                    let bits = block.qs[chunk];
                    let inp = _mm256_loadu_ps(input_row.as_ptr().add(base + chunk * 8));
                    let signs = bits_to_signs_avx2(bits);
                    acc0 = _mm256_fmadd_ps(scale, _mm256_mul_ps(signs, inp), acc0);
                }
            }

            let combined = _mm256_add_ps(acc0, acc1);
            output[mi * n_rows + ni] = hsum_avx2(combined);
        }
    }

    Ok(())
}

// ─── AVX2 Ternary TQ2_0_g128 Kernels ────────────────────────────────────

/// AVX2-accelerated dequantization of TQ2\_0\_g128 blocks to FP32.
///
/// Decodes 2-bit ternary codes (4 per byte, 32 bytes per block = 128 weights)
/// and scales by the block's FP16 scale factor.
///
/// Uses pure SIMD arithmetic decode: packs 2 bytes into a broadcast i32,
/// extracts each 2-bit code via variable shifts, then computes
/// `val = (idx >> 1) - (1 - min(idx, 1))` giving -1/0/+1 without branches.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_tq2_0_g128_avx2(
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

    // Hoist loop-invariant SIMD constants outside all loops
    let one = _mm256_set1_epi32(1);
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let mask3 = _mm256_set1_epi32(3);

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let scale = _mm256_set1_ps(d);
        let base = bi * QK_TQ2_0_G128;

        // 32 bytes × 4 weights = 128 weights; process 2 bytes (8 weights) per AVX2 iteration
        // 16 iterations per block
        for chunk in 0..16 {
            let b0 = block.qs[chunk * 2];
            let b1 = block.qs[chunk * 2 + 1];

            // Pack both bytes into low 16 bits of u32, then broadcast to all lanes
            let bb = (b0 as u32) | ((b1 as u32) << 8);
            let packed = _mm256_set1_epi32(bb as i32);

            // Per-lane variable shifts to extract each 2-bit code
            let shifted = _mm256_srlv_epi32(packed, shifts);
            let idx = _mm256_and_si256(shifted, mask3);

            // Arithmetic decode: val = (idx >> 1) - (1 - min(idx, 1))
            // idx=0 → -1;  idx=1 → 0;  idx=2 → +1
            let pos_part = _mm256_srli_epi32(idx, 1);
            let min_part = _mm256_min_epu32(idx, one);
            let neg_part = _mm256_sub_epi32(one, min_part);
            let val_i = _mm256_sub_epi32(pos_part, neg_part);
            let val_f = _mm256_cvtepi32_ps(val_i);

            let result = _mm256_mul_ps(scale, val_f);
            _mm256_storeu_ps(output.as_mut_ptr().add(base + chunk * 8), result);
        }
    }

    Ok(())
}

/// AVX2-accelerated GEMV for TQ2\_0\_g128-quantized weight matrices.
///
/// Computes `output[row] = dot(weight_row, input)` for each row.
/// Uses pure SIMD arithmetic decode: packs 2 bytes into a broadcast i32,
/// extracts each 2-bit code via variable shifts, then computes
/// `val = (idx >> 1) - (1 - min(idx, 1))` giving -1/0/+1 without branches.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_tq2_0_g128_avx2(
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

    // Hoist loop-invariant SIMD constants outside all loops
    let one = _mm256_set1_epi32(1);
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let mask3 = _mm256_set1_epi32(3);

    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut row_sum = 0.0_f32;

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let inp_base = bi * QK_TQ2_0_G128;
            let mut block_acc = _mm256_setzero_ps();

            // 16 iterations × 8 weights = 128 weights per block
            // Each iteration consumes 2 bytes of qs
            for chunk in 0..16 {
                let b0 = block.qs[chunk * 2];
                let b1 = block.qs[chunk * 2 + 1];

                // Pack both bytes into low 16 bits of u32, then broadcast to all lanes
                let bb = (b0 as u32) | ((b1 as u32) << 8);
                let packed = _mm256_set1_epi32(bb as i32);

                // Per-lane variable shifts to extract each 2-bit code
                let shifted = _mm256_srlv_epi32(packed, shifts);
                let idx = _mm256_and_si256(shifted, mask3);

                // Arithmetic decode: val = (idx >> 1) - (1 - min(idx, 1))
                let pos_part = _mm256_srli_epi32(idx, 1);
                let min_part = _mm256_min_epu32(idx, one);
                let neg_part = _mm256_sub_epi32(one, min_part);
                let val_i = _mm256_sub_epi32(pos_part, neg_part);
                let val_f = _mm256_cvtepi32_ps(val_i);

                let inp_vec = _mm256_loadu_ps(input.as_ptr().add(inp_base + chunk * 8));
                block_acc = _mm256_fmadd_ps(val_f, inp_vec, block_acc);
            }

            row_sum += d * hsum_avx2(block_acc);
        }

        output[row] = row_sum;
    }

    Ok(())
}

/// AVX2-accelerated GEMM for TQ2\_0\_g128-quantized weight matrices.
///
/// Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])` for each (m, n) pair.
/// Iterates over batch dimension `m`, using per-row GEMV logic with pure SIMD decode.
///
/// Uses pure SIMD arithmetic decode: packs 2 bytes into a broadcast i32,
/// extracts each 2-bit code via variable shifts, then computes
/// `val = (idx >> 1) - (1 - min(idx, 1))` giving -1/0/+1 without branches.
///
/// # Safety
/// Requires AVX2+FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemm_tq2_0_g128_avx2(
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

    // Hoist loop-invariant SIMD constants outside all loops
    let one = _mm256_set1_epi32(1);
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    let mask3 = _mm256_set1_epi32(3);

    for mi in 0..m {
        let input_row = &input[mi * k..];

        for ni in 0..n_rows {
            let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
            let mut row_sum = 0.0_f32;

            for (bi, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let inp_base = bi * QK_TQ2_0_G128;
                let mut block_acc = _mm256_setzero_ps();

                for chunk in 0..16 {
                    let b0 = block.qs[chunk * 2];
                    let b1 = block.qs[chunk * 2 + 1];

                    // Pack both bytes into low 16 bits of u32, then broadcast to all lanes
                    let bb = (b0 as u32) | ((b1 as u32) << 8);
                    let packed = _mm256_set1_epi32(bb as i32);

                    // Per-lane variable shifts to extract each 2-bit code
                    let shifted = _mm256_srlv_epi32(packed, shifts);
                    let idx = _mm256_and_si256(shifted, mask3);

                    // Arithmetic decode: val = (idx >> 1) - (1 - min(idx, 1))
                    let pos_part = _mm256_srli_epi32(idx, 1);
                    let min_part = _mm256_min_epu32(idx, one);
                    let neg_part = _mm256_sub_epi32(one, min_part);
                    let val_i = _mm256_sub_epi32(pos_part, neg_part);
                    let val_f = _mm256_cvtepi32_ps(val_i);

                    let inp_vec = _mm256_loadu_ps(input_row.as_ptr().add(inp_base + chunk * 8));
                    block_acc = _mm256_fmadd_ps(val_f, inp_vec, block_acc);
                }

                row_sum += d * hsum_avx2(block_acc);
            }

            output[mi * n_rows + ni] = row_sum;
        }
    }

    Ok(())
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use half::f16;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[test]
    fn avx2_dequant_all_positive() {
        if !has_avx2() {
            return;
        }
        let block = make_block(2.0, [0xFF; 16]);
        let mut output = vec![0.0f32; 128];
        unsafe {
            dequant_1bit_g128_avx2(&[block], &mut output).expect("avx2 dequant should succeed");
        }
        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn avx2_dequant_all_negative() {
        if !has_avx2() {
            return;
        }
        let block = make_block(3.0, [0x00; 16]);
        let mut output = vec![0.0f32; 128];
        unsafe {
            dequant_1bit_g128_avx2(&[block], &mut output).expect("avx2 dequant should succeed");
        }
        for &v in &output {
            assert!((v + 3.0).abs() < 0.01, "expected -3.0, got {v}");
        }
    }

    #[test]
    fn avx2_dequant_matches_reference() {
        if !has_avx2() {
            return;
        }
        let block = make_block(1.5, [0xAA; 16]); // alternating bits
        let mut out_ref = vec![0.0f32; 128];
        let mut out_avx = vec![0.0f32; 128];

        crate::dequant::dequant_1bit_g128(&[block], &mut out_ref)
            .expect("reference dequant should succeed");
        unsafe {
            dequant_1bit_g128_avx2(&[block], &mut out_avx).expect("avx2 dequant should succeed");
        }
        for i in 0..128 {
            assert!(
                (out_ref[i] - out_avx[i]).abs() < 0.01,
                "mismatch at {i}: ref={}, avx2={}",
                out_ref[i],
                out_avx[i]
            );
        }
    }

    #[test]
    fn avx2_gemv_identity_like() {
        if !has_avx2() {
            return;
        }
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 1];

        unsafe {
            gemv_1bit_g128_avx2(&blocks, &input, &mut output, 1, 128)
                .expect("avx2 gemv should succeed");
        }
        let expected: f32 = (0..128).map(|i| i as f32).sum();
        assert!(
            (output[0] - expected).abs() < 1.0,
            "expected ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn avx2_gemv_matches_reference() {
        if !has_avx2() {
            return;
        }
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
        let mut out_avx = vec![0.0f32; n_rows];

        crate::gemv::gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_1bit_g128_avx2(&blocks, &input, &mut out_avx, n_rows, k)
                .expect("avx2 gemv should succeed");
        }

        for i in 0..n_rows {
            assert!(
                (out_ref[i] - out_avx[i]).abs() < 0.1,
                "row {i}: ref={}, avx2={}",
                out_ref[i],
                out_avx[i]
            );
        }
    }

    #[test]
    fn avx2_gemm_matches_reference() {
        if !has_avx2() {
            return;
        }
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
        let mut out_avx = vec![0.0f32; m * n_rows];

        crate::gemm::gemm_1bit_g128(&blocks, &input, &mut out_ref, m, n_rows, k)
            .expect("reference gemm should succeed");
        unsafe {
            gemm_1bit_g128_avx2(&blocks, &input, &mut out_avx, m, n_rows, k)
                .expect("avx2 gemm should succeed");
        }

        for i in 0..(m * n_rows) {
            assert!(
                (out_ref[i] - out_avx[i]).abs() < 0.5,
                "idx {i}: ref={}, avx2={}",
                out_ref[i],
                out_avx[i]
            );
        }
    }

    // ─── Ternary TQ2_0_g128 AVX2 tests ───────────────────────────────────

    fn make_ternary_block(scale: f32, qs: [u8; 32]) -> oxibonsai_core::BlockTQ2_0_g128 {
        oxibonsai_core::BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    /// qs=0xAA → all codes=0b10 (Pos=+1), d=1.0, input=[1.0;128] → output=[128.0;2]
    #[test]
    fn ternary_gemv_avx2_matches_scalar() {
        if !has_avx2() {
            return;
        }
        // 0xAA = 0b10_10_10_10: all 4 codes per byte = 0b10 (Pos = +1)
        let blocks = vec![
            make_ternary_block(1.0, [0xAA; 32]),
            make_ternary_block(1.0, [0xAA; 32]),
        ];
        let input = vec![1.0_f32; 128];
        let mut out_avx = vec![0.0_f32; 2];
        let mut out_ref = vec![0.0_f32; 2];

        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, 2, 128)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_tq2_0_g128_avx2(&blocks, &input, &mut out_avx, 2, 128)
                .expect("avx2 ternary gemv should succeed");
        }

        for i in 0..2 {
            assert!(
                (out_avx[i] - out_ref[i]).abs() < 1e-4,
                "row {i}: avx2={}, ref={}",
                out_avx[i],
                out_ref[i]
            );
        }
    }

    /// Alternating +1/0/-1/0 per group of 4 weights; with input=[1.0;128], output should be 0.
    /// byte=0x46 (0b01_00_01_10): lane0=Pos(+1), lane1=Neg(-1), lane2=Zero, lane3=Zero → sum=0
    #[test]
    fn ternary_gemv_avx2_sign_canary() {
        if !has_avx2() {
            return;
        }
        // 0x46 = 0b01_00_01_10: lane0=0b10(+1), lane1=0b01(0), lane2=0b00(-1), lane3=0b01(0)
        // per-byte contribution with input [1,1,1,1]: (+1 + 0 - 1 + 0) = 0
        let blocks = vec![make_ternary_block(1.0, [0x46; 32])];
        let input = vec![1.0_f32; 128];
        let mut output = vec![99.0_f32; 1];

        unsafe {
            gemv_tq2_0_g128_avx2(&blocks, &input, &mut output, 1, 128)
                .expect("avx2 ternary gemv sign canary should succeed");
        }

        assert!(
            output[0].abs() < 1e-4,
            "sign canary: expected 0.0, got {}",
            output[0]
        );
    }

    /// qs=0x00 → all codes=0b00 (Neg=-1), d=1.0, input=[1.0;128] → output=[-128.0;2]
    #[test]
    fn ternary_gemv_avx2_all_negative() {
        if !has_avx2() {
            return;
        }
        let blocks = vec![
            make_ternary_block(1.0, [0x00; 32]),
            make_ternary_block(1.0, [0x00; 32]),
        ];
        let input = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 2];

        unsafe {
            gemv_tq2_0_g128_avx2(&blocks, &input, &mut output, 2, 128)
                .expect("avx2 ternary all-negative gemv should succeed");
        }

        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val + 128.0).abs() < 1e-4,
                "row {i}: expected -128.0, got {val}",
            );
        }
    }

    #[test]
    fn gemv_tq2_0_g128_avx2_prefetch_matches_reference() {
        if !has_avx2() {
            return;
        }

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
        let mut out_avx2 = vec![0.0f32; n_rows];

        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference ternary gemv should succeed");
        unsafe {
            gemv_tq2_0_g128_avx2_prefetch(&blocks, &input, &mut out_avx2, n_rows, k)
                .expect("avx2 prefetch ternary gemv should succeed");
        }

        let mse = out_ref
            .iter()
            .zip(&out_avx2)
            .map(|(lhs, rhs)| {
                let diff = lhs - rhs;
                diff * diff
            })
            .sum::<f32>()
            / n_rows as f32;
        assert!(mse < 1e-6, "expected MSE < 1e-6, got {mse}");
    }
}
