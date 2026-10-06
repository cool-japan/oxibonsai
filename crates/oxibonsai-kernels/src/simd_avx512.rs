//! AVX-512 optimized 1-bit compute kernels for Q1\_0\_g128.
//!
//! Uses 512-bit AVX-512 intrinsics to accelerate dequantization and
//! fused matrix-vector/matrix-matrix products for Q1\_0\_g128 format.
//!
//! **Key advantages over AVX2:**
//! - 16 f32 lanes per register (vs 8 for AVX2)
//! - `_mm512_mask_blend_ps` takes a `__mmask16` (u16), mapping directly to packed bits
//! - `_mm512_reduce_add_ps` provides built-in horizontal sum (no manual shuffle)
//! - Only 8 iterations per 128-element block (vs 16 for AVX2)

// AVX-512 intrinsics were stabilised in Rust 1.89.0, but our workspace
// MSRV is 1.86.0.  The functions in this module are unconditionally guarded
// by `#[target_feature(enable = "avx512f", …)]` and therefore are only
// reachable when the CPU actually supports AVX-512, making the MSRV concern
// a false positive for this specific file.
#![allow(clippy::incompatible_msrv)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

#[cfg(target_arch = "x86_64")]
use oxibonsai_core::ternary_code_to_i8;

#[cfg(target_arch = "x86_64")]
use crate::dequant_prism::for_each_prism_register_block;
use crate::error::{KernelError, KernelResult};

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Convert 16 weight bits (2 bytes) to a 512-bit sign vector.
///
/// `mask bit = 1` → `+1.0`, `mask bit = 0` → `-1.0`.
///
/// AVX-512 mask blend maps directly to our packed bit representation,
/// making this beautifully simple compared to AVX2.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[inline]
unsafe fn bits_to_signs_avx512(bits_lo: u8, bits_hi: u8) -> __m512 {
    let mask: __mmask16 = (bits_hi as u16) << 8 | bits_lo as u16;
    let neg_one = _mm512_set1_ps(-1.0);
    let pos_one = _mm512_set1_ps(1.0);
    // mask bit=0 selects first operand (neg_one), bit=1 selects second (pos_one)
    _mm512_mask_blend_ps(mask, neg_one, pos_one)
}

/// Horizontal sum of 16 f32 lanes in an AVX-512 register.
///
/// AVX-512 provides a built-in reduce intrinsic, so no manual shuffle is needed.
///
/// # Safety
/// Requires AVX-512F CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_avx512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// The ONE AVX-512 ternary decode helper (K-01): every dequant/gemv/gemm
/// AVX-512 kernel below routes through this function instead of inlining
/// its own copy, so the ISA tier physically cannot drift from the single
/// shared [`oxibonsai_core::ternary_code_to_i8`] table (which maps the
/// reserved `0b11` code to `0`, matching every GPU decoder and the CPU
/// reference).
///
/// Decodes 16 lanes (four packed bytes, 4 codes each, in `b0..b3` order) via
/// 16 scalar table lookups, then a single vector load — deliberately not a
/// hand-rolled SIMD arithmetic re-implementation of the table, which is
/// exactly the kind of "shadow copy" that caused K-01.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn decode_4bytes_avx512_to_f32x16(b0: u8, b1: u8, b2: u8, b3: u8) -> __m512 {
    let vals: [f32; 16] = [
        ternary_code_to_i8(b0) as f32,
        ternary_code_to_i8(b0 >> 2) as f32,
        ternary_code_to_i8(b0 >> 4) as f32,
        ternary_code_to_i8(b0 >> 6) as f32,
        ternary_code_to_i8(b1) as f32,
        ternary_code_to_i8(b1 >> 2) as f32,
        ternary_code_to_i8(b1 >> 4) as f32,
        ternary_code_to_i8(b1 >> 6) as f32,
        ternary_code_to_i8(b2) as f32,
        ternary_code_to_i8(b2 >> 2) as f32,
        ternary_code_to_i8(b2 >> 4) as f32,
        ternary_code_to_i8(b2 >> 6) as f32,
        ternary_code_to_i8(b3) as f32,
        ternary_code_to_i8(b3 >> 2) as f32,
        ternary_code_to_i8(b3 >> 4) as f32,
        ternary_code_to_i8(b3 >> 6) as f32,
    ];
    _mm512_loadu_ps(vals.as_ptr())
}

// ─── AVX-512 Dequantization ─────────────────────────────────────────────

/// AVX-512 accelerated dequantization of Q1\_0\_g128 blocks to FP32.
///
/// Processes 16 elements per SIMD iteration (512-bit / 32-bit = 16 lanes).
/// Each 128-element block requires only 8 iterations (vs 16 for AVX2).
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn dequant_1bit_g128_avx512(
    blocks: &[BlockQ1_0G128],
    output: &mut [f32],
) -> KernelResult<()> {
    let expected_len = blocks.len() * QK1_0_G128;
    if output.len() < expected_len {
        return Err(KernelError::buffer_too_small(
            "output",
            expected_len,
            output.len(),
        ));
    }

    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let scale = _mm512_set1_ps(d);
        let base = i * QK1_0_G128;

        // Process 128 elements in chunks of 16 (8 iterations per block)
        for chunk in 0..8 {
            let bits_lo = block.qs[chunk * 2];
            let bits_hi = block.qs[chunk * 2 + 1];
            let out_offset = base + chunk * 16;

            // Create sign vector from bits: bit=1 → +1.0, bit=0 → -1.0
            let signs = bits_to_signs_avx512(bits_lo, bits_hi);

            let result = _mm512_mul_ps(scale, signs);
            _mm512_storeu_ps(output.as_mut_ptr().add(out_offset), result);
        }
    }

    Ok(())
}

/// AVX-512 accelerated GEMV for TQ2\_0\_g128 with prefetch and 4-byte unroll.
///
/// # Safety
/// Requires AVX-512F CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_tq2_0_g128_avx512_prefetch(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;

    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
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
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();

            for group in 0..4 {
                let base = group * 8;

                let val0 = decode_4bytes_avx512_to_f32x16(
                    block.qs[base],
                    block.qs[base + 1],
                    block.qs[base + 2],
                    block.qs[base + 3],
                );
                let inp0 = _mm512_loadu_ps(input.as_ptr().add(inp_base + group * 32));
                acc0 = _mm512_fmadd_ps(val0, inp0, acc0);

                let val1 = decode_4bytes_avx512_to_f32x16(
                    block.qs[base + 4],
                    block.qs[base + 5],
                    block.qs[base + 6],
                    block.qs[base + 7],
                );
                let inp1 = _mm512_loadu_ps(input.as_ptr().add(inp_base + group * 32 + 16));
                acc1 = _mm512_fmadd_ps(val1, inp1, acc1);
            }

            row_sum += d * hsum_avx512(_mm512_add_ps(acc0, acc1));
        }

        *out = row_sum;
    }

    Ok(())
}

// ─── AVX-512 GEMV ───────────────────────────────────────────────────────

/// AVX-512 accelerated 1-bit GEMV: `output[row] = dot(weight_row, input)`.
///
/// Uses FMA to accumulate dot products in 16-wide f32 registers.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_1bit_g128_avx512(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
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

    let blocks_per_row = k / QK1_0_G128;
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
        let mut row_acc = _mm512_setzero_ps();

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let scale = _mm512_set1_ps(d);
            let input_base = bi * QK1_0_G128;

            // Process 128 elements in chunks of 16 (8 chunks per block)
            for chunk in 0..8 {
                let bits_lo = block.qs[chunk * 2];
                let bits_hi = block.qs[chunk * 2 + 1];
                let inp_offset = input_base + chunk * 16;

                // Load 16 input values
                let inp = _mm512_loadu_ps(input.as_ptr().add(inp_offset));

                // Create sign mask: bit=1 → +1.0, bit=0 → -1.0
                let signs = bits_to_signs_avx512(bits_lo, bits_hi);

                // signed_input = signs * input
                let signed_input = _mm512_mul_ps(signs, inp);

                // row_acc += scale * signed_input (FMA)
                row_acc = _mm512_fmadd_ps(scale, signed_input, row_acc);
            }
        }

        // Horizontal sum of row_acc (built-in AVX-512 reduce)
        output[row] = hsum_avx512(row_acc);
    }

    Ok(())
}

// ─── Register-blocked micro-kernels (K-18's AVX-512 tier) ───────────────
//
// `BlockedTier::Delegate` (gemm_ternary.rs) routes AVX-512 hosts straight
// into this file's `gemm_*_avx512`, which were once loops of GEMVs:
// correct, and never a GPU escape, but with none of K-18's register
// blocking, so an AVX-512 host re-streamed and re-decoded the whole weight
// matrix once per batch row. Rather than add a second entry point that
// `gemm_ternary.rs` would have to be edited to
// call, the blocking is applied **inside** the existing functions: same
// names, same signatures, same numbers.
//
// Bit-identity is by construction, and is the reason the loops are nested
// this exact way: only the *order in which (batch row, weight row) pairs
// are visited* changes, plus the hoisting of the decode out of the batch
// loop. Each pair still performs the same `_mm512_fmadd_ps` sequence
// against the same accumulator and ends with the same `hsum_avx512`, so
// every output element is the identical `f32`.
//
// Compile-blind: this file cannot be executed on the AArch64 machine the
// package was developed on. It is cross-checked against two real x86-64
// targets, not just one: `cargo check -p oxibonsai-kernels --target
// x86_64-apple-darwin --all-features --lib --profile test` (which also
// compiles the `#[cfg(all(test, target_arch = "x86_64"))]` modules) *and*
// `cargo check -p oxibonsai-kernels --target x86_64-unknown-linux-gnu
// --all-features` — `x86_64-unknown-linux-gnu` is installed on this host
// (`rustup target list --installed`), so this is a real Linux x86-64
// compile check, not merely Darwin's x86-64 target. The one part that
// could be silently *wrong* rather than non-compiling — the decode table —
// is proven exhaustively on any host by
// `simd_dot_int8::int8_dot_tests::ternary_f32_lut_matches_the_shared_decode_exhaustively`.
// No runtime validation is claimed on either target.

/// Batch rows a decoded weight block is consumed by before the next block
/// is touched — the AVX-512 twin of `gemm_ternary::TERNARY_GEMM_MR`.
///
/// 8 `__m512` accumulators plus the decoded weight vector and one input
/// vector sit comfortably inside x86-64's 32 `zmm` registers.
pub const AVX512_GEMM_MR: usize = 8;

/// Assemble one `__m512` of 16 decoded ternary weights from four packed
/// `qs` bytes, via four 4-wide table rows and `_mm512_insertf32x4`.
///
/// Bit-identical to [`decode_4bytes_avx512_to_f32x16`] — both read the same
/// generated [`crate::simd_dot_int8::TERNARY_BYTE_LUT_F32`] values in the
/// same lane order — but it replaces sixteen scalar table lookups plus a
/// stack round trip (perf-08) with four vector loads.
///
/// # Safety
/// Requires AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn decode_4bytes_avx512_lut(b0: u8, b1: u8, b2: u8, b3: u8) -> __m512 {
    let lut = crate::simd_dot_int8::TERNARY_BYTE_LUT_F32.as_ptr() as *const f32;
    let r0 = _mm_loadu_ps(lut.add(b0 as usize * 4));
    let r1 = _mm_loadu_ps(lut.add(b1 as usize * 4));
    let r2 = _mm_loadu_ps(lut.add(b2 as usize * 4));
    let r3 = _mm_loadu_ps(lut.add(b3 as usize * 4));
    let v = _mm512_castps128_ps512(r0);
    let v = _mm512_insertf32x4::<1>(v, r1);
    let v = _mm512_insertf32x4::<2>(v, r2);
    _mm512_insertf32x4::<3>(v, r3)
}

/// One register block of the ternary GEMM: `MR` batch rows against every
/// weight row.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL; every index is bounded by
/// the caller's validation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_tq2_avx512<const MR: usize>(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    use oxibonsai_core::QK_TQ2_0_G128;

    for ni in 0..n_rows {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let mut sums = [0.0f32; MR];

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let inp_base = bi * QK_TQ2_0_G128;
            let mut acc = [_mm512_setzero_ps(); MR];

            for chunk in 0..8 {
                let val_f = decode_4bytes_avx512_lut(
                    block.qs[chunk * 4],
                    block.qs[chunk * 4 + 1],
                    block.qs[chunk * 4 + 2],
                    block.qs[chunk * 4 + 3],
                );
                let col = inp_base + chunk * 16;
                for (r, a) in acc.iter_mut().enumerate() {
                    let inp_vec = _mm512_loadu_ps(input.as_ptr().add((m0 + r) * k + col));
                    *a = _mm512_fmadd_ps(val_f, inp_vec, *a);
                }
            }

            for (a, sum) in acc.iter().zip(sums.iter_mut()) {
                *sum += d * hsum_avx512(*a);
            }
        }

        for (r, sum) in sums.iter().enumerate() {
            output[(m0 + r) * n_rows + ni] = *sum;
        }
    }
}

/// One register block of the 1-bit GEMM.
///
/// # Safety
/// See [`tile_tq2_avx512`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_1bit_avx512<const MR: usize>(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    for ni in 0..n_rows {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let mut acc = [_mm512_setzero_ps(); MR];

        for (bi, block) in row_blocks.iter().enumerate() {
            let scale = _mm512_set1_ps(block.d.to_f32());
            let input_base = bi * QK1_0_G128;

            for chunk in 0..8 {
                let signs = bits_to_signs_avx512(block.qs[chunk * 2], block.qs[chunk * 2 + 1]);
                let col = input_base + chunk * 16;
                for (r, a) in acc.iter_mut().enumerate() {
                    let inp = _mm512_loadu_ps(input.as_ptr().add((m0 + r) * k + col));
                    let signed_input = _mm512_mul_ps(signs, inp);
                    *a = _mm512_fmadd_ps(scale, signed_input, *a);
                }
            }
        }

        for (r, a) in acc.iter().enumerate() {
            output[(m0 + r) * n_rows + ni] = hsum_avx512(*a);
        }
    }
}

// ─── AVX-512 GEMM ───────────────────────────────────────────────────────

/// AVX-512 accelerated 1-bit GEMM.
///
/// Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])`.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemm_1bit_g128_avx512(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
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

    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    if m == 0 || n_rows == 0 {
        return Ok(());
    }

    // K-18's AVX-512 tier: register-blocked over the batch dimension, so a
    // decoded sign vector is consumed by `AVX512_GEMM_MR` batch rows before
    // the next `qs` pair is touched. Bit-identical to the loop of GEMVs
    // this replaces — see this file's "Register-blocked micro-kernels"
    // section.
    for_each_prism_register_block!(
        m,
        tile_1bit_avx512,
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

// ─── AVX-512 Streaming GEMV ─────────────────────────────────────────────

/// Minimum output-row count at which the AVX-512 GEMV tier switches from the
/// prefetch kernel to the non-temporal *streaming-store* kernel.
///
/// Below this, the output vector still fits comfortably in cache and the
/// regular (cache-allocating) stores of the prefetch kernel win. Above it —
/// e.g. an LM-head projection over a large vocabulary — the output is big
/// enough that writing it through the cache would evict live weight/input
/// data, so `_mm512_stream_ps` non-temporal stores are the better choice.
#[cfg(target_arch = "x86_64")]
pub const AVX512_STREAMING_MIN_ROWS: usize = 4096;

/// AVX-512 1-bit GEMV tier entry point: picks the prefetch or streaming-store
/// kernel based on `n_rows`.
///
/// This is the function the [`crate::dispatch::KernelDispatcher`] AVX-512 tier
/// routes through. For `n_rows < AVX512_STREAMING_MIN_ROWS` it uses the
/// double-buffered [`gemv_1bit_g128_avx512_prefetch`]; for larger outputs it
/// uses [`gemv_1bit_g128_avx512_streaming`] (non-temporal stores). Both compute
/// the same dot products; only the output store strategy differs.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_1bit_g128_avx512_auto(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if n_rows >= AVX512_STREAMING_MIN_ROWS {
        gemv_1bit_g128_avx512_streaming(blocks, input, output, n_rows, k)
    } else {
        gemv_1bit_g128_avx512_prefetch(blocks, input, output, n_rows, k)
    }
}

/// AVX-512 1-bit GEMV with streaming stores for large output vectors.
///
/// Uses `_mm512_stream_ps` for non-temporal stores to bypass the cache
/// hierarchy on output writes. This is beneficial when the output buffer
/// is large and won't be read again soon, freeing cache lines for
/// weight data and input.
///
/// Also includes prefetch hints for the next row's weight blocks.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
/// Output pointer must be 64-byte aligned for streaming stores.
/// Falls back to regular stores if alignment is not met.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_1bit_g128_avx512_streaming(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
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

    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    // Process rows in groups of 16 for streaming stores
    // (16 f32 values = 64 bytes = one cache line = one streaming store)
    let full_groups = n_rows / 16;
    let remainder = n_rows % 16;

    // Temporary buffer for group of 16 row results
    let mut group_results = [0.0f32; 16];

    for group in 0..full_groups {
        let base_row = group * 16;

        for (local_row, group_result) in group_results.iter_mut().enumerate() {
            let row = base_row + local_row;
            let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];

            // Prefetch next row's blocks
            if local_row + 1 < 16 {
                let next_ptr = blocks.as_ptr().add((row + 1) * blocks_per_row) as *const i8;
                core::arch::x86_64::_mm_prefetch(next_ptr, core::arch::x86_64::_MM_HINT_T0);
            }

            let mut row_acc = _mm512_setzero_ps();

            for (bi, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let scale = _mm512_set1_ps(d);
                let input_base = bi * QK1_0_G128;

                for chunk in 0..8 {
                    let bits_lo = block.qs[chunk * 2];
                    let bits_hi = block.qs[chunk * 2 + 1];
                    let inp_offset = input_base + chunk * 16;

                    let inp = _mm512_loadu_ps(input.as_ptr().add(inp_offset));
                    let signs = bits_to_signs_avx512(bits_lo, bits_hi);
                    let signed_input = _mm512_mul_ps(signs, inp);
                    row_acc = _mm512_fmadd_ps(scale, signed_input, row_acc);
                }
            }

            *group_result = hsum_avx512(row_acc);
        }

        // Use streaming store for the group of 16 results
        let group_vec = _mm512_loadu_ps(group_results.as_ptr());
        let out_ptr = output.as_mut_ptr().add(base_row);
        // Check alignment for streaming store
        if (out_ptr as usize).is_multiple_of(64) {
            _mm512_stream_ps(out_ptr, group_vec);
        } else {
            _mm512_storeu_ps(out_ptr, group_vec);
        }
    }

    // Handle remaining rows with regular stores
    for row in (full_groups * 16)..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut row_acc = _mm512_setzero_ps();

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let scale = _mm512_set1_ps(d);
            let input_base = bi * QK1_0_G128;

            for chunk in 0..8 {
                let bits_lo = block.qs[chunk * 2];
                let bits_hi = block.qs[chunk * 2 + 1];
                let inp_offset = input_base + chunk * 16;

                let inp = _mm512_loadu_ps(input.as_ptr().add(inp_offset));
                let signs = bits_to_signs_avx512(bits_lo, bits_hi);
                let signed_input = _mm512_mul_ps(signs, inp);
                row_acc = _mm512_fmadd_ps(scale, signed_input, row_acc);
            }
        }

        output[row] = hsum_avx512(row_acc);
    }

    // Memory fence to ensure all streaming stores are visible
    if full_groups > 0 {
        _mm_sfence();
    }

    let _ = remainder; // used implicitly in the remainder loop

    Ok(())
}

/// AVX-512 GEMV with prefetch and double-buffered accumulation.
///
/// Uses two AVX-512 accumulators alternating across blocks for
/// maximum FMA throughput, combined with prefetch hints.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_1bit_g128_avx512_prefetch(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
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

    let blocks_per_row = k / QK1_0_G128;
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

        // Prefetch next row
        if row + 1 < n_rows {
            let next_ptr = blocks.as_ptr().add((row + 1) * blocks_per_row) as *const i8;
            core::arch::x86_64::_mm_prefetch(next_ptr, core::arch::x86_64::_MM_HINT_T0);
        }

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let scale = _mm512_set1_ps(d);
            let input_base = bi * QK1_0_G128;

            // Prefetch next block
            if bi + 1 < blocks_per_row {
                let next_block = row_blocks.as_ptr().add(bi + 1) as *const i8;
                core::arch::x86_64::_mm_prefetch(next_block, core::arch::x86_64::_MM_HINT_T0);
            }

            // Process 8 chunks, alternating accumulators
            for chunk in 0..8 {
                let bits_lo = block.qs[chunk * 2];
                let bits_hi = block.qs[chunk * 2 + 1];
                let inp_offset = input_base + chunk * 16;

                let inp = _mm512_loadu_ps(input.as_ptr().add(inp_offset));
                let signs = bits_to_signs_avx512(bits_lo, bits_hi);
                let signed = _mm512_mul_ps(signs, inp);

                if chunk & 1 == 0 {
                    acc0 = _mm512_fmadd_ps(scale, signed, acc0);
                } else {
                    acc1 = _mm512_fmadd_ps(scale, signed, acc1);
                }
            }
        }

        let combined = _mm512_add_ps(acc0, acc1);
        output[row] = hsum_avx512(combined);
    }

    Ok(())
}

// ─── AVX-512 Ternary TQ2_0_g128 Kernels ─────────────────────────────────

/// AVX-512 accelerated dequantization of TQ2\_0\_g128 blocks to FP32.
///
/// Decodes 2-bit ternary codes (4 per byte, 32 bytes per block = 128 weights)
/// and scales by the block's FP16 scale factor. Processes 16 weights per
/// AVX-512 iteration (4 bytes → 16 lanes).
///
/// Decode is routed through the single shared
/// [`decode_4bytes_avx512_to_f32x16`] helper (K-01), which calls
/// [`oxibonsai_core::ternary_code_to_i8`] per lane rather than re-deriving
/// the map with SIMD arithmetic — this is what makes it impossible for this
/// tier to drift from the reference decode.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn dequant_tq2_0_g128_avx512(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    output: &mut [f32],
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;
    let needed = blocks.len() * QK_TQ2_0_G128;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let scale = _mm512_set1_ps(d);
        let base = bi * QK_TQ2_0_G128;

        // 32 bytes × 4 weights = 128 weights; process 4 bytes (16 weights) per iteration
        // 8 iterations per block
        for chunk in 0..8 {
            let b0 = block.qs[chunk * 4];
            let b1 = block.qs[chunk * 4 + 1];
            let b2 = block.qs[chunk * 4 + 2];
            let b3 = block.qs[chunk * 4 + 3];

            // K-01: this used to be an unmasked inline copy of the decode
            // (pos_part/min_part/neg_part with no `0b11` guard), so a
            // reserved code decoded to +1 here while every other ternary
            // kernel decoded it to 0. Routing through the single shared
            // helper makes that divergence structurally impossible.
            let val_f = decode_4bytes_avx512_to_f32x16(b0, b1, b2, b3);

            let result = _mm512_mul_ps(scale, val_f);
            _mm512_storeu_ps(output.as_mut_ptr().add(base + chunk * 16), result);
        }
    }

    Ok(())
}

/// AVX-512 accelerated GEMV for TQ2\_0\_g128-quantized weight matrices.
///
/// Computes `output[row] = dot(weight_row, input)` for each row.
/// Processes 16 weights per iteration (4 bytes decoded via the shared
/// helper) and accumulates with `_mm512_fmadd_ps`.
///
/// Decode is routed through the single shared
/// [`decode_4bytes_avx512_to_f32x16`] helper (K-01) — see the dequant
/// function above for why.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemv_tq2_0_g128_avx512(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;

    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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

    let blocks_per_row = k / QK_TQ2_0_G128;
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
        let mut row_sum = 0.0_f32;

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let inp_base = bi * QK_TQ2_0_G128;
            let mut block_acc = _mm512_setzero_ps();

            // 8 iterations × 16 weights = 128 weights per block
            // Each iteration consumes 4 bytes of qs
            for chunk in 0..8 {
                let b0 = block.qs[chunk * 4];
                let b1 = block.qs[chunk * 4 + 1];
                let b2 = block.qs[chunk * 4 + 2];
                let b3 = block.qs[chunk * 4 + 3];

                // K-01: routed through the single shared decode helper
                // (previously an unmasked inline copy; see dequant above).
                let val_f = decode_4bytes_avx512_to_f32x16(b0, b1, b2, b3);

                let inp_vec = _mm512_loadu_ps(input.as_ptr().add(inp_base + chunk * 16));
                block_acc = _mm512_fmadd_ps(val_f, inp_vec, block_acc);
            }

            row_sum += d * hsum_avx512(block_acc);
        }

        output[row] = row_sum;
    }

    Ok(())
}

/// AVX-512 accelerated GEMM for TQ2\_0\_g128-quantized weight matrices.
///
/// Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])` for each (m, n) pair.
/// Iterates over batch dimension `m`, using per-row GEMV logic.
///
/// Decode is routed through the single shared
/// [`decode_4bytes_avx512_to_f32x16`] helper (K-01) — see the dequant
/// function above for why.
///
/// # Safety
/// Requires AVX-512F + AVX-512BW + AVX-512VL CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl")]
pub unsafe fn gemm_tq2_0_g128_avx512(
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    use oxibonsai_core::QK_TQ2_0_G128;

    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    if m == 0 || n_rows == 0 {
        return Ok(());
    }

    // K-18's AVX-512 tier: register-blocked over the batch dimension, so
    // each 128-weight block is decoded once per `AVX512_GEMM_MR` batch rows
    // instead of once per batch row. Bit-identical to the loop of GEMVs it
    // replaces — see this file's "Register-blocked micro-kernels" section.
    // K-01 still holds: `decode_4bytes_avx512_lut` reads the same generated
    // `ternary_code_to_i8` table (reserved `0b11` -> 0) the scalar helper
    // does, proved entry by entry by
    // `simd_dot_int8::int8_dot_tests::ternary_f32_lut_matches_the_shared_decode_exhaustively`.
    for_each_prism_register_block!(
        m,
        tile_tq2_avx512,
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

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use half::f16;

    fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
    }

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn avx512_dequant_all_positive() {
        if !has_avx512() {
            return;
        }
        let block = make_block(2.0, [0xFF; 16]);
        let mut output = vec![0.0f32; 128];
        unsafe {
            dequant_1bit_g128_avx512(&[block], &mut output).expect("dequant should succeed");
        }
        for &v in &output {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn avx512_dequant_all_negative() {
        if !has_avx512() {
            return;
        }
        let block = make_block(3.0, [0x00; 16]);
        let mut output = vec![0.0f32; 128];
        unsafe {
            dequant_1bit_g128_avx512(&[block], &mut output).expect("dequant should succeed");
        }
        for &v in &output {
            assert!((v + 3.0).abs() < 0.01, "expected -3.0, got {v}");
        }
    }

    #[test]
    fn avx512_dequant_matches_reference() {
        if !has_avx512() {
            return;
        }
        let block = make_block(1.5, [0xAA; 16]); // alternating bits
        let mut out_ref = vec![0.0f32; 128];
        let mut out_avx512 = vec![0.0f32; 128];

        crate::dequant::dequant_1bit_g128(&[block], &mut out_ref)
            .expect("reference dequant should succeed");
        unsafe {
            dequant_1bit_g128_avx512(&[block], &mut out_avx512)
                .expect("avx512 dequant should succeed");
        }
        for i in 0..128 {
            assert!(
                (out_ref[i] - out_avx512[i]).abs() < 0.01,
                "mismatch at {i}: ref={}, avx512={}",
                out_ref[i],
                out_avx512[i]
            );
        }
    }

    #[test]
    fn avx512_gemv_identity_like() {
        if !has_avx512() {
            return;
        }
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 1];

        unsafe {
            gemv_1bit_g128_avx512(&blocks, &input, &mut output, 1, 128)
                .expect("gemv should succeed");
        }
        let expected: f32 = (0..128).map(|i| i as f32).sum();
        assert!(
            (output[0] - expected).abs() < 1.0,
            "expected ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn avx512_gemv_matches_reference() {
        if !has_avx512() {
            return;
        }
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
        let mut out_avx512 = vec![0.0f32; n_rows];

        crate::gemv::gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_1bit_g128_avx512(&blocks, &input, &mut out_avx512, n_rows, k)
                .expect("avx512 gemv should succeed");
        }

        for i in 0..n_rows {
            assert!(
                (out_ref[i] - out_avx512[i]).abs() < 0.1,
                "row {i}: ref={}, avx512={}",
                out_ref[i],
                out_avx512[i]
            );
        }
    }

    #[test]
    fn avx512_gemv_streaming_matches_reference() {
        if !has_avx512() {
            return;
        }
        // 37 rows = 2 full streaming groups of 16 + a 5-row remainder tail,
        // exercising both the `_mm512_stream_ps` group path and the scalar
        // remainder-store path of the streaming kernel.
        let n_rows = 37;
        let k = 256;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [(row as u8).wrapping_mul(37).wrapping_add(bi as u8 * 13); 16];
                blocks.push(make_block(0.5 + row as f32 * 0.05, bits));
            }
        }
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();

        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_stream = vec![0.0f32; n_rows];

        crate::gemv::gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_1bit_g128_avx512_streaming(&blocks, &input, &mut out_stream, n_rows, k)
                .expect("avx512 streaming gemv should succeed");
        }

        for i in 0..n_rows {
            let tol = 1e-3 * out_ref[i].abs().max(1.0);
            assert!(
                (out_ref[i] - out_stream[i]).abs() <= tol,
                "row {i}: ref={}, stream={}",
                out_ref[i],
                out_stream[i]
            );
        }
    }

    #[test]
    fn avx512_gemv_auto_routes_streaming_above_threshold() {
        if !has_avx512() {
            return;
        }
        // Above AVX512_STREAMING_MIN_ROWS the auto entry point routes to the
        // streaming kernel; its result must still match the scalar reference.
        let n_rows = AVX512_STREAMING_MIN_ROWS + 5;
        let k = 128;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [(row as u8).wrapping_mul(11).wrapping_add(bi as u8 * 7); 16];
                blocks.push(make_block(0.25 + (row % 8) as f32 * 0.1, bits));
            }
        }
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.02) - 1.28).collect();

        let mut out_ref = vec![0.0f32; n_rows];
        let mut out_auto = vec![0.0f32; n_rows];

        crate::gemv::gemv_1bit_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_1bit_g128_avx512_auto(&blocks, &input, &mut out_auto, n_rows, k)
                .expect("auto gemv should succeed");
        }

        for i in 0..n_rows {
            let tol = 1e-3 * out_ref[i].abs().max(1.0);
            assert!(
                (out_ref[i] - out_auto[i]).abs() <= tol,
                "row {i}: ref={}, auto={}",
                out_ref[i],
                out_auto[i]
            );
        }
    }

    #[test]
    fn avx512_gemm_matches_reference() {
        if !has_avx512() {
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
        let mut out_avx512 = vec![0.0f32; m * n_rows];

        crate::gemm::gemm_1bit_g128(&blocks, &input, &mut out_ref, m, n_rows, k)
            .expect("reference gemm should succeed");
        unsafe {
            gemm_1bit_g128_avx512(&blocks, &input, &mut out_avx512, m, n_rows, k)
                .expect("avx512 gemm should succeed");
        }

        for i in 0..(m * n_rows) {
            assert!(
                (out_ref[i] - out_avx512[i]).abs() < 0.5,
                "idx {i}: ref={}, avx512={}",
                out_ref[i],
                out_avx512[i]
            );
        }
    }

    // ─── Ternary TQ2_0_g128 AVX-512 tests ────────────────────────────────

    fn make_ternary_block(scale: f32, qs: [u8; 32]) -> oxibonsai_core::BlockTQ2_0_g128 {
        oxibonsai_core::BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    /// K-01 anti-drift guard: exhaustively checks every one of the 256
    /// possible values of each of `b0..b3` (each mapping to its own group
    /// of 4 lanes) against [`oxibonsai_core::ternary_code_to_i8`]. The four
    /// bytes decode independently (no cross-term), so pinning the other
    /// three bytes at a fixed value while sweeping the full `0..=255` range
    /// of the byte under test is exhaustive, not merely a sample. A future
    /// re-vectorization of `decode_4bytes_avx512_to_f32x16` that
    /// reintroduces hand-rolled SIMD arithmetic (instead of calling the
    /// shared table) fails this the moment it disagrees, including on the
    /// reserved `0b11` code.
    #[test]
    fn decode_4bytes_avx512_to_f32x16_matches_ternary_code_to_i8_exhaustively() {
        if !has_avx512() {
            return;
        }
        let assert_lanes = |b0: u8, b1: u8, b2: u8, b3: u8| {
            let decoded = unsafe { decode_4bytes_avx512_to_f32x16(b0, b1, b2, b3) };
            let mut lanes = [0.0f32; 16];
            unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), decoded) };
            for (group, byte) in [b0, b1, b2, b3].into_iter().enumerate() {
                for (lane, &got) in lanes[group * 4..group * 4 + 4].iter().enumerate() {
                    let expected = oxibonsai_core::ternary_code_to_i8(byte >> (lane * 2)) as f32;
                    assert_eq!(
                        got,
                        expected,
                        "decode_4bytes_avx512_to_f32x16({b0:#04x},{b1:#04x},{b2:#04x},{b3:#04x}) \
                         lane {} (byte group {group}={byte:#04x}): got {got}, expected {expected}",
                        group * 4 + lane
                    );
                }
            }
        };
        for b0 in 0..=255u8 {
            assert_lanes(b0, 0x00, 0x00, 0x00);
        }
        for b1 in 0..=255u8 {
            assert_lanes(0x00, b1, 0x00, 0x00);
        }
        for b2 in 0..=255u8 {
            assert_lanes(0x00, 0x00, b2, 0x00);
        }
        for b3 in 0..=255u8 {
            assert_lanes(0x00, 0x00, 0x00, b3);
        }
    }

    /// qs=0xAA → all codes=0b10 (Pos=+1), d=1.0, input=[1.0;128] → output=[128.0;2]
    #[test]
    fn ternary_gemv_avx512_matches_scalar() {
        if !has_avx512() {
            return;
        }
        let blocks = vec![
            make_ternary_block(1.0, [0xAA; 32]),
            make_ternary_block(1.0, [0xAA; 32]),
        ];
        let input = vec![1.0_f32; 128];
        let mut out_avx512 = vec![0.0_f32; 2];
        let mut out_ref = vec![0.0_f32; 2];

        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, 2, 128)
            .expect("reference gemv should succeed");
        unsafe {
            gemv_tq2_0_g128_avx512(&blocks, &input, &mut out_avx512, 2, 128)
                .expect("avx512 ternary gemv should succeed");
        }

        for i in 0..2 {
            assert!(
                (out_avx512[i] - out_ref[i]).abs() < 1e-4,
                "row {i}: avx512={}, ref={}",
                out_avx512[i],
                out_ref[i]
            );
        }
    }

    /// Alternating +1/0/-1/0 per group of 4 weights; with input=[1.0;128], output should be 0.
    /// byte=0x46 (0b01_00_01_10): lane0=Pos(+1), lane1=Zero, lane2=Neg(-1), lane3=Zero → sum=0
    #[test]
    fn ternary_gemv_avx512_sign_canary() {
        if !has_avx512() {
            return;
        }
        let blocks = vec![make_ternary_block(1.0, [0x46; 32])];
        let input = vec![1.0_f32; 128];
        let mut output = vec![99.0_f32; 1];

        unsafe {
            gemv_tq2_0_g128_avx512(&blocks, &input, &mut output, 1, 128)
                .expect("avx512 ternary gemv sign canary should succeed");
        }

        assert!(
            output[0].abs() < 1e-4,
            "sign canary: expected 0.0, got {}",
            output[0]
        );
    }

    /// qs=0x00 → all codes=0b00 (Neg=-1), d=1.0, input=[1.0;128] → output=[-128.0;2]
    #[test]
    fn ternary_gemv_avx512_all_negative() {
        if !has_avx512() {
            return;
        }
        let blocks = vec![
            make_ternary_block(1.0, [0x00; 32]),
            make_ternary_block(1.0, [0x00; 32]),
        ];
        let input = vec![1.0_f32; 128];
        let mut output = vec![0.0_f32; 2];

        unsafe {
            gemv_tq2_0_g128_avx512(&blocks, &input, &mut output, 2, 128)
                .expect("avx512 ternary all-negative gemv should succeed");
        }

        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val + 128.0).abs() < 1e-4,
                "row {i}: expected -128.0, got {val}",
            );
        }
    }

    #[test]
    fn gemv_tq2_0_g128_avx512_prefetch_matches_reference() {
        if !has_avx512() {
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
        let mut out_avx512 = vec![0.0f32; n_rows];

        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference ternary gemv should succeed");
        unsafe {
            gemv_tq2_0_g128_avx512_prefetch(&blocks, &input, &mut out_avx512, n_rows, k)
                .expect("avx512 prefetch ternary gemv should succeed");
        }

        let mse = out_ref
            .iter()
            .zip(&out_avx512)
            .map(|(lhs, rhs)| {
                let diff = lhs - rhs;
                diff * diff
            })
            .sum::<f32>()
            / n_rows as f32;
        assert!(mse < 1e-6, "expected MSE < 1e-6, got {mse}");
    }
}
