//! NEON GEMV kernels for the standard GGUF quant formats Q4_0 and Q8_0.
//!
//! Mirrors `crate::simd_q_std_avx2` / `crate::simd_q_std_avx512` but targets
//! AArch64 NEON (128-bit registers, 4 × f32 lanes) instead of x86 AVX. Before
//! this module existed, AArch64 CPUs (e.g. Apple Silicon) had no SIMD path for
//! Q4_0/Q8_0 GEMV: [`crate::dispatch::KernelDispatcher`] only wires AVX-512/AVX2
//! tiers for these two formats, so every AArch64 call silently fell through to
//! the scalar reference in [`crate::gemv_q4_0()`] / [`crate::gemv_q8_0()`] — correct,
//! but without the SIMD speedup already available for the Q1_0 / ternary / FP8
//! formats on the same hardware.
//!
//! - **Q8_0** (32 × i8 + f16 scale): each block's 32 packed `i8` weights are
//!   widened `i8 → i16 → i32 → f32` via `vmovl_s8`/`vmovl_s16`/`vcvtq_f32_s32`,
//!   scaled by the block's f16 scale, and fused-multiply-added with the
//!   matching input slice — 8 chunks of 4 lanes per block.
//! - **Q4_0** (32 × 4-bit + f16 scale, llama.cpp lo-hi split): the 16 packed
//!   bytes are split into lower nibbles (elements 0..16) and upper nibbles
//!   (elements 16..32) via `vandq_u8` / `vshrq_n_u8`, each nibble group is
//!   zero-extended `u8 → u16 → u32`, centred (`− 8`), converted to f32, scaled
//!   and fused-multiply-added — 8 chunks of 4 lanes total per block.
//!
//! Both kernels produce results numerically equivalent (within f32 rounding)
//! to the scalar reference kernels in `crate::gemv_q4_0::gemv_q4_0_scalar` /
//! `crate::gemv_q8_0::gemv_q8_0_scalar`; the in-module `tests` below and the
//! `tests/neon_q_std_parity.rs` integration tests assert this.
//!
//! These functions are intentionally **not** wired into
//! [`crate::dispatch::KernelDispatcher`] yet — dispatch wiring for the new
//! `KernelTier::Neon` arm is a deliberately separate change so the two
//! concerns (kernel correctness vs. routing) can be reviewed independently.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use oxibonsai_core::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};

#[cfg(target_arch = "aarch64")]
use crate::error::{KernelError, KernelResult};

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Horizontal sum of the 4 f32 lanes of a NEON `float32x4_t` register.
///
/// # Safety
/// Requires NEON CPU support. Always available on AArch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

/// Validate GEMV arguments shared by Q4_0/Q8_0 and return `blocks_per_row`.
#[cfg(target_arch = "aarch64")]
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

// ─── Q8_0 GEMV ──────────────────────────────────────────────────────────────

/// Decode 16 packed Q8_0 `i8` weights into four `float32x4_t` chunks
/// (elements `0..4`, `4..8`, `8..12`, `12..16`).
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_q8_0_i8x16(bytes: int8x16_t) -> [float32x4_t; 4] {
    let lo8 = vget_low_s8(bytes);
    let hi8 = vget_high_s8(bytes);
    let lo16 = vmovl_s8(lo8);
    let hi16 = vmovl_s8(hi8);
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))),
    ]
}

/// NEON GEMV for a Q8_0-quantized weight matrix.
///
/// See [`crate::gemv_q8_0::gemv_q8_0`] for the parameter contract; this
/// produces the same result within f32 rounding.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q8_0_neon(
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
        let mut acc = vdupq_n_f32(0.0);

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let scale = vdupq_n_f32(block.d.to_f32());
            let inp_base = bi * QK_Q8_0;

            // Load the 32 packed i8 weights as two 16-lane vectors.
            let v0 = vld1q_s8(block.qs.as_ptr());
            let v1 = vld1q_s8(block.qs.as_ptr().add(16));
            let chunks0 = decode_q8_0_i8x16(v0);
            let chunks1 = decode_q8_0_i8x16(v1);

            for (ci, chunk) in chunks0.iter().chain(chunks1.iter()).enumerate() {
                let iv = vld1q_f32(input.as_ptr().add(inp_base + ci * 4));
                acc = vfmaq_f32(acc, vmulq_f32(scale, *chunk), iv);
            }
        }

        output[row] = hsum_neon(acc);
    }

    Ok(())
}

// ─── Q4_0 GEMV ──────────────────────────────────────────────────────────────

/// Decode 16 packed Q4_0 nibble bytes (already masked to `0..=15`, one nibble
/// group per lane in `bytes`) into four `float32x4_t` chunks, centred by
/// subtracting 8 (`w = nibble − 8`).
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decode_q4_0_nibbles(bytes: uint8x16_t, eight: int32x4_t) -> [float32x4_t; 4] {
    let lo8 = vget_low_u8(bytes);
    let hi8 = vget_high_u8(bytes);
    let lo16 = vmovl_u8(lo8);
    let hi16 = vmovl_u8(hi8);
    let a = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16)));
    let b = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo16)));
    let c = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16)));
    let d = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi16)));
    [
        vcvtq_f32_s32(vsubq_s32(a, eight)),
        vcvtq_f32_s32(vsubq_s32(b, eight)),
        vcvtq_f32_s32(vsubq_s32(c, eight)),
        vcvtq_f32_s32(vsubq_s32(d, eight)),
    ]
}

/// NEON GEMV for a Q4_0-quantized weight matrix.
///
/// See [`crate::gemv_q4_0::gemv_q4_0`] for the parameter contract; this
/// produces the same result within f32 rounding. Uses the llama.cpp lo-hi
/// split: byte `j` holds element `j` in its lower nibble and element `j + 16`
/// in its upper nibble.
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q4_0_neon(
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

    let mask_lo = vdupq_n_u8(0x0F);
    let eight = vdupq_n_s32(8);

    for row in 0..n_rows {
        let mut acc = vdupq_n_f32(0.0);

        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let scale = vdupq_n_f32(block.d.to_f32());
            let inp_base = bi * QK_Q4_0;

            // Load the 16 packed bytes; split into lower/upper nibbles.
            let v = vld1q_u8(block.qs.as_ptr());
            let lo = vandq_u8(v, mask_lo);
            let hi = vandq_u8(vshrq_n_u8::<4>(v), mask_lo);

            // lo → elements 0..16 ↔ input[inp_base + 0..16]
            let lo_chunks = decode_q4_0_nibbles(lo, eight);
            for (ci, chunk) in lo_chunks.iter().enumerate() {
                let iv = vld1q_f32(input.as_ptr().add(inp_base + ci * 4));
                acc = vfmaq_f32(acc, vmulq_f32(scale, *chunk), iv);
            }

            // hi → elements 16..32 ↔ input[inp_base + 16..32]
            let hi_chunks = decode_q4_0_nibbles(hi, eight);
            for (ci, chunk) in hi_chunks.iter().enumerate() {
                let iv = vld1q_f32(input.as_ptr().add(inp_base + 16 + ci * 4));
                acc = vfmaq_f32(acc, vmulq_f32(scale, *chunk), iv);
            }
        }

        output[row] = hsum_neon(acc);
    }

    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::{gemv_q4_0::gemv_q4_0_scalar, gemv_q8_0::gemv_q8_0_scalar};

    /// Deterministic LCG-based f32 generator in `[-scale, scale)`.
    fn make_input(k: usize, seed: u32, scale: f32) -> Vec<f32> {
        (0..k)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x >> 8) as f32 / u32::MAX as f32) * 2.0 * scale - scale
            })
            .collect()
    }

    fn q8_blocks(n_rows: usize, blocks_per_row: usize, seed: u32) -> Vec<BlockQ8_0> {
        let mut raw = Vec::with_capacity(n_rows * blocks_per_row * QK_Q8_0);
        for i in 0..n_rows * blocks_per_row * QK_Q8_0 {
            let x = (i as u32).wrapping_mul(40_503).wrapping_add(seed);
            raw.push(((x >> 7) as f32 / u32::MAX as f32) * 6.0 - 3.0);
        }
        BlockQ8_0::quantize(&raw).expect("quantize q8_0 test fixture")
    }

    fn q4_blocks(n_rows: usize, blocks_per_row: usize, seed: u32) -> Vec<BlockQ4_0> {
        let mut raw = Vec::with_capacity(n_rows * blocks_per_row * QK_Q4_0);
        for i in 0..n_rows * blocks_per_row * QK_Q4_0 {
            let x = (i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed);
            raw.push(((x >> 9) as f32 / u32::MAX as f32) * 8.0 - 4.0);
        }
        BlockQ4_0::quantize(&raw).expect("quantize q4_0 test fixture")
    }

    /// Boundary matrix: 1 block, several blocks, odd row counts, larger k.
    const SHAPES: [(usize, usize); 6] = [(1, 32), (2, 64), (3, 96), (5, 128), (17, 160), (1, 512)];

    #[test]
    fn q8_0_neon_matches_scalar_boundary_matrix() {
        for (n_rows, in_features) in SHAPES {
            let blocks_per_row = in_features / QK_Q8_0;
            let blocks = q8_blocks(n_rows, blocks_per_row, 11);
            let input = make_input(in_features, 7, 2.0);

            let mut out_ref = vec![0.0f32; n_rows];
            let mut out_neon = vec![0.0f32; n_rows];
            gemv_q8_0_scalar(&blocks, &input, &mut out_ref, n_rows, in_features)
                .expect("scalar q8_0 gemv should succeed");
            unsafe {
                gemv_q8_0_neon(&blocks, &input, &mut out_neon, n_rows, in_features)
                    .expect("neon q8_0 gemv should succeed");
            }

            for r in 0..n_rows {
                let tol = 1e-4 * out_ref[r].abs().max(1.0);
                assert!(
                    (out_ref[r] - out_neon[r]).abs() <= tol,
                    "q8_0 row {r} (n_rows={n_rows}, k={in_features}): ref={}, neon={}",
                    out_ref[r],
                    out_neon[r]
                );
            }
        }
    }

    #[test]
    fn q4_0_neon_matches_scalar_boundary_matrix() {
        for (n_rows, in_features) in SHAPES {
            let blocks_per_row = in_features / QK_Q4_0;
            let blocks = q4_blocks(n_rows, blocks_per_row, 23);
            let input = make_input(in_features, 5, 3.0);

            let mut out_ref = vec![0.0f32; n_rows];
            let mut out_neon = vec![0.0f32; n_rows];
            gemv_q4_0_scalar(&blocks, &input, &mut out_ref, n_rows, in_features)
                .expect("scalar q4_0 gemv should succeed");
            unsafe {
                gemv_q4_0_neon(&blocks, &input, &mut out_neon, n_rows, in_features)
                    .expect("neon q4_0 gemv should succeed");
            }

            for r in 0..n_rows {
                let tol = 1e-4 * out_ref[r].abs().max(1.0);
                assert!(
                    (out_ref[r] - out_neon[r]).abs() <= tol,
                    "q4_0 row {r} (n_rows={n_rows}, k={in_features}): ref={}, neon={}",
                    out_ref[r],
                    out_neon[r]
                );
            }
        }
    }

    #[test]
    fn q4_0_neon_rejects_bad_dims() {
        let blocks = q4_blocks(1, 1, 1);
        let input = make_input(32, 1, 4.0);
        let mut output = vec![0.0f32; 1];
        unsafe {
            let result = gemv_q4_0_neon(&blocks, &input, &mut output, 1, 31);
            assert!(
                matches!(result, Err(KernelError::NotBlockAligned { .. })),
                "expected NotBlockAligned, got {result:?}"
            );
        }
    }

    #[test]
    fn q8_0_neon_rejects_bad_dims() {
        let blocks = q8_blocks(1, 1, 1);
        let input = make_input(32, 1, 4.0);
        let mut output = vec![0.0f32; 1];
        unsafe {
            let result = gemv_q8_0_neon(&blocks, &input, &mut output, 1, 31);
            assert!(
                matches!(result, Err(KernelError::NotBlockAligned { .. })),
                "expected NotBlockAligned, got {result:?}"
            );
        }
    }

    #[test]
    fn q4_0_neon_output_too_small() {
        let blocks = q4_blocks(1, 1, 2);
        let input = make_input(32, 2, 4.0);
        let mut output: Vec<f32> = vec![];
        unsafe {
            let result = gemv_q4_0_neon(&blocks, &input, &mut output, 1, 32);
            assert!(
                matches!(result, Err(KernelError::BufferTooSmall { .. })),
                "expected BufferTooSmall, got {result:?}"
            );
        }
    }

    #[test]
    fn q8_0_neon_output_too_small() {
        let blocks = q8_blocks(1, 1, 3);
        let input = make_input(32, 3, 4.0);
        let mut output: Vec<f32> = vec![];
        unsafe {
            let result = gemv_q8_0_neon(&blocks, &input, &mut output, 1, 32);
            assert!(
                matches!(result, Err(KernelError::BufferTooSmall { .. })),
                "expected BufferTooSmall, got {result:?}"
            );
        }
    }

    /// Sign canary: alternating +1/-1 int8 weights with all-1 input sums to 0.
    #[test]
    fn q8_0_neon_alternating_sign_cancels() {
        use half::f16;
        let mut qs = [0i8; 32];
        for (j, q) in qs.iter_mut().enumerate() {
            *q = if j % 2 == 0 { 1 } else { -1 };
        }
        let blocks = vec![BlockQ8_0 {
            d: f16::from_f32(1.0),
            qs,
        }];
        let input = vec![1.0f32; 32];
        let mut output = vec![99.0f32; 1];
        unsafe {
            gemv_q8_0_neon(&blocks, &input, &mut output, 1, 32)
                .expect("neon q8_0 gemv should succeed");
        }
        assert!(
            output[0].abs() < 1e-4,
            "alternating sign: expected 0, got {}",
            output[0]
        );
    }

    /// All-zero-weight Q4_0 block (nibble=8) → dot product is 0.
    #[test]
    fn q4_0_neon_all_zero_weights() {
        use half::f16;
        let blocks = vec![BlockQ4_0 {
            d: f16::from_f32(1.0),
            qs: [0x88u8; 16],
        }];
        let input = vec![1.0f32; 32];
        let mut output = vec![99.0f32; 1];
        unsafe {
            gemv_q4_0_neon(&blocks, &input, &mut output, 1, 32)
                .expect("neon q4_0 gemv should succeed");
        }
        assert!(
            output[0].abs() < 1e-5,
            "all-zero weights → output 0, got {}",
            output[0]
        );
    }
}
