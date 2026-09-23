//! INT8 dot-product kernels for the ternary / 2-bit / 1-bit weight formats
//! (K-14, step 2) — the arithmetic core of the `NeonDot` / `NeonI8mm` /
//! `Avx512Vnni` tiers that [`crate::dispatch_int8`] selects.
//!
//! Every existing CPU kernel decodes a packed weight code to `f32` and
//! accumulates with an FMA: four MACs per instruction, plus an int->float
//! conversion per four weights (`simd_neon::decode_byte_neon_to_f32x4`,
//! `simd_avx2::decode_2bytes_avx2_to_f32x8`). This module decodes the same
//! codes **straight to `i8`** and accumulates with `SDOT` (16 MACs per
//! instruction), `SMMLA` (32) or `VPDPBUSD` (64).
//!
//! ## One algebra for every tier: the biased table
//!
//! AVX-512 VNNI's `VPDPBUSD` is **u8 x i8**. Rather than give x86 a
//! different derivation from NEON (and so an untestable one on an Apple
//! machine), every tier here uses the same identity, with the *weight* as
//! the unsigned operand:
//!
//! ```text
//! biased_j = value_j + 1                     in 0..=3 for every format
//! Σ_j value_j * x_j = Σ_j biased_j * x_j − Σ_j x_j
//! ```
//!
//! `Σ_j x_j` is a per-activation-block constant
//! ([`crate::quant_activation::Int8Activation::sums_row`]), so the
//! correction costs one `i32` subtraction per block and the scalar tests
//! below exercise exactly the algebra the blind x86 path relies on.
//!
//! ## Two tables, not one (K-01 / VERIFIED.md)
//!
//! The legacy ternary `TQ2_0_g128` map sends the reserved code `0b11` to
//! `0`, while `PQ2_0` / `Q2_0_g64` send it to `+2`. Both tables are
//! *generated* at compile time from the crate's single shared decode
//! functions ([`oxibonsai_core::ternary_code_to_i8`],
//! [`oxibonsai_core::q2_0_code_to_i32`]) — never transcribed — so K-01's
//! "one table" invariant holds by construction, and
//! `int8_dot_tests::biased_luts_match_the_shared_decode_tables` re-checks
//! all eight entries at test time anyway.
//!
//! ## Why inline `asm!`, not `vdotq_s32`
//!
//! `core::arch::aarch64::vdotq_s32` and `vmmlaq_s32` are still unstable
//! (`stdarch_neon_dotprod` / `stdarch_neon_i8mm`, rustc 1.95), and enabling
//! a library feature needs a crate-level `#![feature(..)]` this package
//! cannot add. `asm!` with `.arch_extension` is stable, emits the identical
//! single instruction, and is guarded at runtime by
//! `is_aarch64_feature_detected!` in [`crate::dispatch_int8`].
//! [`dot_i8x16_widening`] is the ARMv8.0 fallback (`SMULL` + `SADALP`); it
//! is exact integer arithmetic, so it agrees with `SDOT` **bit for bit**,
//! which `int8_dot_tests::neon_sdot_matches_the_widening_fallback` asserts
//! exhaustively over random inputs.
//!
//! ## Range
//!
//! `biased <= 3`, `|x| <= 127`, at most 128 weights per block:
//! `|Σ| <= 3 * 127 * 128 = 48 768`, far inside `i32`. No accumulator here
//! can overflow, on any tier.

use oxibonsai_core::{q2_0_code_to_i32, ternary_code_to_i8};

/// Biased (`value + 1`) weight table for the legacy ternary `TQ2_0_g128`
/// map — `{-1, 0, +1, 0}` biased to `{0, 1, 2, 1}` (code `0b11` is
/// reserved and decodes to `0`).
pub const TQ2_BIASED_LUT: [u8; 4] = build_ternary_biased_lut();

/// Biased (`value + 1`) weight table for the `PQ2_0` / `Q2_0_g64`
/// arithmetic map — `{-1, 0, +1, +2}` biased to `{0, 1, 2, 3}`.
pub const PQ2_BIASED_LUT: [u8; 4] = build_two_bit_biased_lut();

/// Biased weight values for the 1-bit `Q1_0_g128` format: a clear bit is
/// `-1` (biased `0`), a set bit `+1` (biased `2`).
pub const ONE_BIT_BIASED_SET: u8 = 2;

const fn build_ternary_biased_lut() -> [u8; 4] {
    let mut lut = [0u8; 4];
    let mut code = 0usize;
    while code < 4 {
        lut[code] = (ternary_code_to_i8(code as u8) + 1) as u8;
        code += 1;
    }
    lut
}

const fn build_two_bit_biased_lut() -> [u8; 4] {
    let mut lut = [0u8; 4];
    let mut code = 0usize;
    while code < 4 {
        lut[code] = (q2_0_code_to_i32(code as u8) + 1) as u8;
        code += 1;
    }
    lut
}

/// Expand a 4-entry biased table into the 16-byte form a NEON `TBL` /
/// AVX-512 `PSHUFB` lookup indexes (codes only ever reach `3`, so the
/// upper twelve entries are never selected).
#[must_use]
pub const fn biased_lut16(lut: &[u8; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0] = lut[0];
    out[1] = lut[1];
    out[2] = lut[2];
    out[3] = lut[3];
    out
}

// ═════════════════════════════════════════════════════════════════════════
//  Scalar reference — the parity oracle, and the portable tier
// ═════════════════════════════════════════════════════════════════════════

/// `Σ_j biased(code_j) * act[position(j)]` for one 2-bit block.
///
/// `qs` is the block's packed codes (4 per byte, LSB-first); `act` is the
/// matching `qs.len() * 4` int8 activations in
/// [`crate::quant_activation::Int8Layout::Stride4`] order. The caller
/// subtracts the block's activation sum to remove the bias.
///
/// This walks the stride-4 layout explicitly — position `g*64 + s*16 + i`
/// holds element `g*64 + 4*i + s`, which is byte `g*16 + i`, lane `s` —
/// so it is also the executable specification the SIMD tiers are tested
/// against.
#[must_use]
pub fn block_dot_two_bit_scalar(qs: &[u8], act: &[i8], lut: &[u8; 4]) -> i32 {
    let groups = qs.len() / 16;
    let mut acc = 0i32;
    for g in 0..groups {
        for s in 0..4usize {
            for i in 0..16usize {
                let code = ((qs[g * 16 + i] >> (2 * s)) & 0b11) as usize;
                acc += lut[code] as i32 * act[g * 64 + s * 16 + i] as i32;
            }
        }
    }
    acc
}

/// `Σ_j biased(bit_j) * act[j]` for one 1-bit `Q1_0_g128` block.
///
/// `qs` is the block's sign bits (8 weights per byte, LSB-first: weight `w`
/// is bit `w % 8` of byte `w / 8`) and `act` the matching `qs.len() * 8`
/// int8 activations in
/// [`crate::quant_activation::Int8Layout::Sequential`] order.
#[must_use]
pub fn block_dot_one_bit_scalar(qs: &[u8], act: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (byte_idx, &byte) in qs.iter().enumerate() {
        for bit in 0..8usize {
            let set = (byte >> bit) & 1 == 1;
            if set {
                acc += ONE_BIT_BIASED_SET as i32 * act[byte_idx * 8 + bit] as i32;
            }
        }
    }
    acc
}

// ═════════════════════════════════════════════════════════════════════════
//  AArch64 NEON
// ═════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;
    use core::arch::asm;

    /// `acc += dot4(a, b)` via ARMv8.2 `SDOT`: four independent 4-way
    /// int8 dot products, one per `i32` lane.
    ///
    /// # Safety
    /// The host must support the `dotprod` feature — guaranteed by
    /// [`crate::dispatch_int8::Int8Tier`]'s runtime gate.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn dot_i8x16_sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut out = acc;
        asm!(
            ".arch_extension dotprod",
            "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
        out
    }

    /// ARMv8.0 equivalent of [`dot_i8x16_sdot`]: widening multiply plus
    /// pairwise accumulate. Exact integer arithmetic, so bit-for-bit equal
    /// to `SDOT` — `vpaddlq_s16` sums lanes in pairs and `vpaddq_s32` pairs
    /// those, reproducing `SDOT`'s four-bytes-per-lane grouping exactly.
    ///
    /// # Safety
    /// NEON is the AArch64 baseline.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn dot_i8x16_widening(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let p_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let p_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        let s_lo = vpaddlq_s16(p_lo);
        let s_hi = vpaddlq_s16(p_hi);
        vaddq_s32(acc, vpaddq_s32(s_lo, s_hi))
    }

    /// `SDOT` when `USE_SDOT`, else the widening fallback. Both produce the
    /// identical `int32x4_t`.
    ///
    /// # Safety
    /// See the two implementations.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn dot_i8x16<const USE_SDOT: bool>(
        acc: int32x4_t,
        a: int8x16_t,
        b: int8x16_t,
    ) -> int32x4_t {
        if USE_SDOT {
            dot_i8x16_sdot(acc, a, b)
        } else {
            dot_i8x16_widening(acc, a, b)
        }
    }

    /// `acc[2x2] += a[2x8] * b[2x8]^T` via ARMv8.6 `SMMLA`: 32 int8 MACs in
    /// one instruction. Lane order of the result is
    /// `[a0.b0, a0.b1, a1.b0, a1.b1]`.
    ///
    /// # Safety
    /// The host must support the `i8mm` feature — guaranteed by
    /// [`crate::dispatch_int8::Int8Tier`]'s runtime gate.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn mmla_i8(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut out = acc;
        asm!(
            ".arch_extension i8mm",
            "smmla {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
        out
    }

    /// Horizontal sum of the four `i32` lanes.
    ///
    /// # Safety
    /// NEON is the AArch64 baseline.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn hsum_s32(v: int32x4_t) -> i32 {
        vaddvq_s32(v)
    }

    /// Decode one 16-byte `qs` chunk (64 weights) into four biased-weight
    /// vectors, lane `i` of vector `s` holding weight `4*i + s` — the
    /// stride-4 order [`crate::quant_activation::Int8Layout::Stride4`]
    /// pre-permutes the activation into.
    ///
    /// # Safety
    /// NEON is the AArch64 baseline; `qs` must hold at least 16 bytes from
    /// `ptr`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn decode_two_bit_group(ptr: *const u8, lut: uint8x16_t) -> [int8x16_t; 4] {
        let q = vld1q_u8(ptr);
        let mask = vdupq_n_u8(0b11);
        let c0 = vandq_u8(q, mask);
        let c1 = vandq_u8(vshrq_n_u8::<2>(q), mask);
        let c2 = vandq_u8(vshrq_n_u8::<4>(q), mask);
        let c3 = vandq_u8(vshrq_n_u8::<6>(q), mask);
        [
            vreinterpretq_s8_u8(vqtbl1q_u8(lut, c0)),
            vreinterpretq_s8_u8(vqtbl1q_u8(lut, c1)),
            vreinterpretq_s8_u8(vqtbl1q_u8(lut, c2)),
            vreinterpretq_s8_u8(vqtbl1q_u8(lut, c3)),
        ]
    }

    /// Decode two `qs` bytes (16 weights, sequential order) of a 1-bit
    /// block into biased weights (`0` or `2`).
    ///
    /// # Safety
    /// NEON is the AArch64 baseline.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn decode_one_bit_pair(b0: u8, b1: u8) -> int8x16_t {
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bytes = vcombine_u8(vdup_n_u8(b0), vdup_n_u8(b1));
        let bits = vld1q_u8(BITS.as_ptr());
        let set = vtstq_u8(bytes, bits);
        vreinterpretq_s8_u8(vandq_u8(set, vdupq_n_u8(ONE_BIT_BIASED_SET)))
    }

    /// `Σ_j biased(code_j) * act[position(j)]` for one 2-bit block.
    ///
    /// # Safety
    /// `qs` must be a whole number of 16-byte groups and `act` must hold
    /// `qs.len() * 4` int8 codes; both are guaranteed by the block types.
    #[target_feature(enable = "neon")]
    pub unsafe fn block_dot_two_bit_neon<const USE_SDOT: bool>(
        qs: &[u8],
        act: &[i8],
        lut: uint8x16_t,
    ) -> i32 {
        let groups = qs.len() / 16;
        let mut acc = vdupq_n_s32(0);
        for g in 0..groups {
            let w = decode_two_bit_group(qs.as_ptr().add(g * 16), lut);
            for (s, wv) in w.iter().enumerate() {
                let x = vld1q_s8(act.as_ptr().add(g * 64 + s * 16));
                acc = dot_i8x16::<USE_SDOT>(acc, *wv, x);
            }
        }
        hsum_s32(acc)
    }

    /// `Σ_j biased(bit_j) * act[j]` for one 1-bit block.
    ///
    /// # Safety
    /// `act` must hold `qs.len() * 8` int8 codes.
    #[target_feature(enable = "neon")]
    pub unsafe fn block_dot_one_bit_neon<const USE_SDOT: bool>(qs: &[u8], act: &[i8]) -> i32 {
        let mut acc = vdupq_n_s32(0);
        for pair in 0..qs.len() / 2 {
            let w = decode_one_bit_pair(qs[pair * 2], qs[pair * 2 + 1]);
            let x = vld1q_s8(act.as_ptr().add(pair * 16));
            acc = dot_i8x16::<USE_SDOT>(acc, w, x);
        }
        hsum_s32(acc)
    }

    /// One `SMMLA` 2x2 tile for a 2-bit block: two weight rows against two
    /// activation rows, 32 int8 MACs per instruction.
    ///
    /// `SMMLA` computes `A x B^T` with `A` in `Vn` and `B` in `Vm`, storing
    /// `A[i].B[j]` in lane `2*i + j`. Passing the weights as `A` and the
    /// activations as `B` therefore returns
    /// `[w0.x0, w0.x1, w1.x0, w1.x1]` — lane 1 is *weight row 0 against
    /// activation row 1*.
    ///
    /// # Safety
    /// The host must support `i8mm`; `qs0`/`qs1` must be whole 16-byte
    /// groups and each `act` must hold `qs0.len() * 4` codes.
    #[target_feature(enable = "neon")]
    pub unsafe fn block_dot_two_bit_2x2_i8mm(
        qs0: &[u8],
        qs1: &[u8],
        act0: &[i8],
        act1: &[i8],
        lut: uint8x16_t,
    ) -> [i32; 4] {
        let groups = qs0.len() / 16;
        let mut acc = vdupq_n_s32(0);
        for g in 0..groups {
            let w0 = decode_two_bit_group(qs0.as_ptr().add(g * 16), lut);
            let w1 = decode_two_bit_group(qs1.as_ptr().add(g * 16), lut);
            for s in 0..4usize {
                let x0 = vld1q_s8(act0.as_ptr().add(g * 64 + s * 16));
                let x1 = vld1q_s8(act1.as_ptr().add(g * 64 + s * 16));
                // Low 8 lanes, then high 8: SMMLA contracts over 8 lanes.
                let a_lo = vcombine_s8(vget_low_s8(w0[s]), vget_low_s8(w1[s]));
                let b_lo = vcombine_s8(vget_low_s8(x0), vget_low_s8(x1));
                acc = mmla_i8(acc, a_lo, b_lo);
                let a_hi = vcombine_s8(vget_high_s8(w0[s]), vget_high_s8(w1[s]));
                let b_hi = vcombine_s8(vget_high_s8(x0), vget_high_s8(x1));
                acc = mmla_i8(acc, a_hi, b_hi);
            }
        }
        let mut out = [0i32; 4];
        vst1q_s32(out.as_mut_ptr(), acc);
        out
    }

    /// Load a 16-byte biased lookup table into a NEON register.
    ///
    /// # Safety
    /// NEON is the AArch64 baseline.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn load_lut(lut16: &[u8; 16]) -> uint8x16_t {
        vld1q_u8(lut16.as_ptr())
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon::{
    block_dot_one_bit_neon, block_dot_two_bit_2x2_i8mm, block_dot_two_bit_neon, dot_i8x16_sdot,
    dot_i8x16_widening, load_lut, mmla_i8,
};

// ═════════════════════════════════════════════════════════════════════════
//  x86-64 AVX-512 VNNI
// ═════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use core::arch::x86_64::*;

    /// Decode one 16-byte `qs` chunk (64 weights) into biased weights,
    /// 128-bit lane `s` holding the 16 weights `{4*i + s}` — the same
    /// stride-4 order the NEON decode produces, so both ISAs consume the
    /// identically permuted activation.
    ///
    /// Shifting a 32-bit word right by `2s` and masking each byte with
    /// `0b11` yields bits `[2s, 2s+2)` of that byte: the bits borrowed from
    /// the neighbouring byte land above bit 1 and the mask discards them.
    ///
    /// # Safety
    /// Requires AVX-512F + AVX-512BW; `ptr` must be readable for 16 bytes.
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    #[inline]
    unsafe fn decode_two_bit_group(ptr: *const u8, lut: __m512i) -> __m512i {
        let q = _mm_loadu_si128(ptr as *const __m128i);
        let bcast = _mm512_broadcast_i32x4(q);
        let shifts = _mm512_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6);
        let codes = _mm512_and_si512(_mm512_srlv_epi32(bcast, shifts), _mm512_set1_epi8(0b11));
        _mm512_shuffle_epi8(lut, codes)
    }

    /// `Σ_j biased(code_j) * act[position(j)]` for one 2-bit block, via
    /// `VPDPBUSD` (64 int8 MACs per instruction).
    ///
    /// # Safety
    /// Requires AVX-512F + AVX-512BW + AVX-512VNNI; `act` must hold
    /// `qs.len() * 4` int8 codes.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
    pub unsafe fn block_dot_two_bit_avx512vnni(qs: &[u8], act: &[i8], lut16: &[u8; 16]) -> i32 {
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(lut16.as_ptr() as *const __m128i));
        let mut acc = _mm512_setzero_si512();
        for g in 0..qs.len() / 16 {
            let biased = decode_two_bit_group(qs.as_ptr().add(g * 16), lut);
            let x = _mm512_loadu_si512(act.as_ptr().add(g * 64) as *const __m512i);
            acc = _mm512_dpbusd_epi32(acc, biased, x);
        }
        _mm512_reduce_add_epi32(acc)
    }

    /// `Σ_j biased(bit_j) * act[j]` for one 1-bit block, via `VPDPBUSD`.
    ///
    /// # Safety
    /// Requires AVX-512F + AVX-512BW + AVX-512VNNI; `qs.len()` must be a
    /// multiple of 8 and `act` must hold `qs.len() * 8` int8 codes.
    #[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
    pub unsafe fn block_dot_one_bit_avx512vnni(qs: &[u8], act: &[i8]) -> i32 {
        let two = _mm512_set1_epi8(ONE_BIT_BIASED_SET as i8);
        let mut acc = _mm512_setzero_si512();
        for g in 0..qs.len() / 8 {
            let mut mask_bytes = [0u8; 8];
            mask_bytes.copy_from_slice(&qs[g * 8..g * 8 + 8]);
            let mask = u64::from_le_bytes(mask_bytes);
            let biased = _mm512_and_si512(_mm512_movm_epi8(mask), two);
            let x = _mm512_loadu_si512(act.as_ptr().add(g * 64) as *const __m512i);
            acc = _mm512_dpbusd_epi32(acc, biased, x);
        }
        _mm512_reduce_add_epi32(acc)
    }
}

#[cfg(target_arch = "x86_64")]
pub use avx512::{block_dot_one_bit_avx512vnni, block_dot_two_bit_avx512vnni};

/// One packed ternary `qs` byte to its four decoded `f32` weights,
/// LSB-first — the `f32` companion of [`TQ2_BIASED_LUT`], used by
/// `simd_avx512`'s register-blocked GEMM (K-18's missing AVX-512 tier).
///
/// It lives **here**, in an architecture-independent module, rather than
/// beside its only caller, for one reason: `simd_avx512.rs` is
/// `#[cfg(target_arch = "x86_64")]`, so a test of this table placed there
/// could never run on the AArch64 machine this package is developed on.
/// `int8_dot_tests::ternary_f32_lut_matches_the_shared_decode_exhaustively`
/// checks all 1024 entries on every host instead, which is the only part of
/// the blind AVX-512 decode that can actually be wrong.
///
/// Generated by a `const fn` from [`oxibonsai_core::ternary_code_to_i8`],
/// never transcribed, so K-01's "one table" invariant holds by
/// construction. (`gemm_ternary.rs` holds a private, identically-generated
/// copy for its own NEON/AVX2 kernels; that file is not this package's to
/// edit, and two tables generated from the same `const fn` cannot drift.)
pub static TERNARY_BYTE_LUT_F32: [[f32; 4]; 256] = build_ternary_byte_lut_f32();

const fn build_ternary_byte_lut_f32() -> [[f32; 4]; 256] {
    let mut table = [[0.0f32; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let b = byte as u8;
        table[byte] = [
            ternary_code_to_i8(b) as f32,
            ternary_code_to_i8(b >> 2) as f32,
            ternary_code_to_i8(b >> 4) as f32,
            ternary_code_to_i8(b >> 6) as f32,
        ];
        byte += 1;
    }
    table
}

// =========================================================================
//  Weight-block view
// =========================================================================

/// The three 2-bit weight formats this tier serves, behind one view:
/// `TQ2_0_g128` (legacy ternary, reserved `0b11 -> 0`), `PQ2_0` and
/// `Q2_0_g64` (arithmetic, `0b11 -> +2`).
///
/// A *local* trait rather than a reuse of `dequant_prism::TwoBitBlockView`:
/// that module documents, at length, that it must never name
/// `BlockTQ2_0_g128` (finding K-06), and each format's biased table travels
/// with the format here, so a kernel physically cannot pick up the wrong
/// one.
pub trait Int8TwoBitBlock {
    /// Weights per block.
    const QK: usize;
    /// The biased (`value + 1`) code table this format decodes with.
    const BIASED_LUT: [u8; 4];
    /// The block's packed 2-bit codes, 4 per byte, LSB-first.
    fn packed_codes(&self) -> &[u8];
    /// The block's FP16 scale, widened to `f32`.
    fn block_scale(&self) -> f32;
}

impl Int8TwoBitBlock for oxibonsai_core::BlockTQ2_0_g128 {
    const QK: usize = oxibonsai_core::QK_TQ2_0_G128;
    const BIASED_LUT: [u8; 4] = TQ2_BIASED_LUT;
    #[inline(always)]
    fn packed_codes(&self) -> &[u8] {
        &self.qs
    }
    #[inline(always)]
    fn block_scale(&self) -> f32 {
        self.d.to_f32()
    }
}

impl Int8TwoBitBlock for oxibonsai_core::BlockPQ2_0 {
    const QK: usize = oxibonsai_core::QK_PQ2_0;
    const BIASED_LUT: [u8; 4] = PQ2_BIASED_LUT;
    #[inline(always)]
    fn packed_codes(&self) -> &[u8] {
        &self.qs
    }
    #[inline(always)]
    fn block_scale(&self) -> f32 {
        self.d.to_f32()
    }
}

impl Int8TwoBitBlock for oxibonsai_core::BlockQ2_0G64 {
    const QK: usize = oxibonsai_core::QK_Q2_0_G64;
    const BIASED_LUT: [u8; 4] = PQ2_BIASED_LUT;
    #[inline(always)]
    fn packed_codes(&self) -> &[u8] {
        &self.qs
    }
    #[inline(always)]
    fn block_scale(&self) -> f32 {
        self.d.to_f32()
    }
}

#[cfg(test)]
mod int8_dot_tests {
    use super::*;
    use crate::quant_activation::Int8Layout;

    /// Seed for the `SMMLA` tile test (AArch64-only, like that test).
    #[cfg(target_arch = "aarch64")]
    const MM_SEED: u32 = 0x0081_0081;

    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed | 1)
        }
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 19) as u8
        }
        fn next_i8(&mut self) -> i8 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((self.0 >> 17) % 255) as i32 - 127) as i8
        }
    }

    #[test]
    fn biased_luts_match_the_shared_decode_tables() {
        for code in 0u8..4 {
            assert_eq!(
                TQ2_BIASED_LUT[code as usize] as i32,
                ternary_code_to_i8(code) as i32 + 1,
                "ternary biased table drifted at code {code}"
            );
            assert_eq!(
                PQ2_BIASED_LUT[code as usize] as i32,
                q2_0_code_to_i32(code) + 1,
                "two-bit biased table drifted at code {code}"
            );
        }
        // K-01: the two tables genuinely differ, and only at 0b11.
        assert_eq!(TQ2_BIASED_LUT[..3], PQ2_BIASED_LUT[..3]);
        assert_ne!(TQ2_BIASED_LUT[3], PQ2_BIASED_LUT[3]);
        assert_eq!(TQ2_BIASED_LUT, [0, 1, 2, 1]);
        assert_eq!(PQ2_BIASED_LUT, [0, 1, 2, 3]);
    }

    /// Exhaustive, architecture-independent check of the `f32` table the
    /// blind AVX-512 register-blocked GEMM decodes through.
    #[test]
    fn ternary_f32_lut_matches_the_shared_decode_exhaustively() {
        for byte in 0..=255u8 {
            let row = TERNARY_BYTE_LUT_F32[byte as usize];
            for (lane, got) in row.iter().enumerate() {
                let expect = ternary_code_to_i8(byte >> (lane * 2)) as f32;
                assert_eq!(
                    got.to_bits(),
                    expect.to_bits(),
                    "byte {byte} lane {lane}: {got} != {expect}"
                );
            }
        }
        // The reserved code really is mapped to 0 here too (K-01).
        assert_eq!(TERNARY_BYTE_LUT_F32[0b11][0], 0.0);
        assert_eq!(TERNARY_BYTE_LUT_F32[0b10][0], 1.0);
        assert_eq!(TERNARY_BYTE_LUT_F32[0b00][0], -1.0);
    }

    #[test]
    fn biased_lut16_zero_fills_the_unused_entries() {
        let l = biased_lut16(&PQ2_BIASED_LUT);
        assert_eq!(&l[..4], &PQ2_BIASED_LUT[..]);
        assert!(l[4..].iter().all(|&v| v == 0));
    }

    /// The bias identity, stated directly: the scalar biased dot minus the
    /// activation sum must equal the plain signed dot.
    #[test]
    fn biased_identity_reproduces_the_signed_dot() {
        let mut rng = Lcg::new(0xA11CE);
        let qs: Vec<u8> = (0..32).map(|_| rng.next_u8()).collect();
        let act: Vec<i8> = (0..128).map(|_| rng.next_i8()).collect();
        for lut in [&TQ2_BIASED_LUT, &PQ2_BIASED_LUT] {
            let biased = block_dot_two_bit_scalar(&qs, &act, lut);
            let sum_x: i32 = act.iter().map(|&c| c as i32).sum();
            let mut signed = 0i32;
            for j in 0..128usize {
                let code = (qs[j / 4] >> (2 * (j % 4))) & 0b11;
                let value = lut[code as usize] as i32 - 1;
                signed += value * act[Int8Layout::Stride4.position_of(j)] as i32;
            }
            assert_eq!(biased - sum_x, signed);
        }
    }

    /// The scalar 2-bit dot must agree with a straight `f32` dot of the
    /// dequantized weights against the int8 codes — integers, so exactly.
    #[test]
    fn scalar_two_bit_dot_matches_an_elementwise_sum() {
        let mut rng = Lcg::new(0x00BE_EF01);
        let qs: Vec<u8> = (0..32).map(|_| rng.next_u8()).collect();
        let act: Vec<i8> = (0..128).map(|_| rng.next_i8()).collect();
        let mut expect = 0i32;
        for j in 0..128usize {
            let code = (qs[j / 4] >> (2 * (j % 4))) & 0b11;
            expect += PQ2_BIASED_LUT[code as usize] as i32
                * act[Int8Layout::Stride4.position_of(j)] as i32;
        }
        assert_eq!(block_dot_two_bit_scalar(&qs, &act, &PQ2_BIASED_LUT), expect);
    }

    #[test]
    fn scalar_one_bit_dot_matches_an_elementwise_sum() {
        let mut rng = Lcg::new(0x00BE_EF02);
        let qs: Vec<u8> = (0..16).map(|_| rng.next_u8()).collect();
        let act: Vec<i8> = (0..128).map(|_| rng.next_i8()).collect();
        let mut expect = 0i32;
        for j in 0..128usize {
            if (qs[j / 8] >> (j % 8)) & 1 == 1 {
                expect += ONE_BIT_BIASED_SET as i32 * act[j] as i32;
            }
        }
        assert_eq!(block_dot_one_bit_scalar(&qs, &act), expect);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_sdot_matches_the_widening_fallback() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            eprintln!("skipping: host has no dotprod");
            return;
        }
        use core::arch::aarch64::*;
        let mut rng = Lcg::new(0xD07);
        for _ in 0..256 {
            let a: [i8; 16] = std::array::from_fn(|_| (rng.next_u8() % 4) as i8);
            let b: [i8; 16] = std::array::from_fn(|_| rng.next_i8());
            // SAFETY: NEON is baseline; dotprod was just detected.
            unsafe {
                let va = vld1q_s8(a.as_ptr());
                let vb = vld1q_s8(b.as_ptr());
                let zero = vdupq_n_s32(0);
                let mut s = [0i32; 4];
                let mut w = [0i32; 4];
                vst1q_s32(s.as_mut_ptr(), dot_i8x16_sdot(zero, va, vb));
                vst1q_s32(w.as_mut_ptr(), dot_i8x16_widening(zero, va, vb));
                assert_eq!(s, w, "SDOT and the widening fallback must agree");
                for (g, lane) in s.iter().enumerate() {
                    let expect: i32 = (0..4)
                        .map(|j| a[g * 4 + j] as i32 * b[g * 4 + j] as i32)
                        .sum();
                    assert_eq!(*lane, expect, "lane {g}");
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_two_bit_dot_matches_the_scalar_reference() {
        let mut rng = Lcg::new(0xC0FFEE);
        let has_dot = std::arch::is_aarch64_feature_detected!("dotprod");
        for qs_len in [16usize, 32] {
            for _ in 0..64 {
                let qs: Vec<u8> = (0..qs_len).map(|_| rng.next_u8()).collect();
                let act: Vec<i8> = (0..qs_len * 4).map(|_| rng.next_i8()).collect();
                for lut in [&TQ2_BIASED_LUT, &PQ2_BIASED_LUT] {
                    let expect = block_dot_two_bit_scalar(&qs, &act, lut);
                    let l16 = biased_lut16(lut);
                    // SAFETY: NEON is baseline on AArch64; the `sdot`
                    // variant is only reached when `dotprod` is detected.
                    unsafe {
                        let lv = load_lut(&l16);
                        assert_eq!(
                            block_dot_two_bit_neon::<false>(&qs, &act, lv),
                            expect,
                            "widening NEON dot diverged"
                        );
                        if has_dot {
                            assert_eq!(
                                block_dot_two_bit_neon::<true>(&qs, &act, lv),
                                expect,
                                "SDOT NEON dot diverged"
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_one_bit_dot_matches_the_scalar_reference() {
        let mut rng = Lcg::new(0xF00D);
        let has_dot = std::arch::is_aarch64_feature_detected!("dotprod");
        for _ in 0..128 {
            let qs: Vec<u8> = (0..16).map(|_| rng.next_u8()).collect();
            let act: Vec<i8> = (0..128).map(|_| rng.next_i8()).collect();
            let expect = block_dot_one_bit_scalar(&qs, &act);
            // SAFETY: NEON is baseline; `sdot` gated on detection.
            unsafe {
                assert_eq!(block_dot_one_bit_neon::<false>(&qs, &act), expect);
                if has_dot {
                    assert_eq!(block_dot_one_bit_neon::<true>(&qs, &act), expect);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i8mm_2x2_tile_matches_four_scalar_dots() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            eprintln!("skipping: host has no i8mm");
            return;
        }
        let mut rng = Lcg::new(MM_SEED);
        for _ in 0..64 {
            let qs0: Vec<u8> = (0..32).map(|_| rng.next_u8()).collect();
            let qs1: Vec<u8> = (0..32).map(|_| rng.next_u8()).collect();
            let act0: Vec<i8> = (0..128).map(|_| rng.next_i8()).collect();
            let act1: Vec<i8> = (0..128).map(|_| rng.next_i8()).collect();
            let l16 = biased_lut16(&PQ2_BIASED_LUT);
            // SAFETY: i8mm was just detected; shapes are whole groups.
            let tile = unsafe {
                let lv = load_lut(&l16);
                block_dot_two_bit_2x2_i8mm(&qs0, &qs1, &act0, &act1, lv)
            };
            let w0x0 = block_dot_two_bit_scalar(&qs0, &act0, &PQ2_BIASED_LUT);
            let w0x1 = block_dot_two_bit_scalar(&qs0, &act1, &PQ2_BIASED_LUT);
            let w1x0 = block_dot_two_bit_scalar(&qs1, &act0, &PQ2_BIASED_LUT);
            let w1x1 = block_dot_two_bit_scalar(&qs1, &act1, &PQ2_BIASED_LUT);
            assert_eq!(tile, [w0x0, w0x1, w1x0, w1x1], "SMMLA tile lane order");
        }
    }
}
