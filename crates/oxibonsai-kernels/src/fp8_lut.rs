//! Pre-computed FP8 decode lookup tables.
//!
//! Building the decode LUT once (lazily on first access) via `OnceLock`
//! costs 256 scalar decode calls per format — negligible.  All subsequent
//! SIMD kernels can then replace the per-weight float decode with a simple
//! indexed load from a 1-KiB table that fits in L1 cache.
//!
//! Both E4M3FN and E5M2 tables cover the full 0–255 byte range, including
//! the NaN / Inf codes (which decode to the IEEE754 bit patterns produced by
//! the scalar `fp8_*_decode` functions).

use std::sync::OnceLock;

use oxibonsai_core::{fp8_e4m3_decode, fp8_e5m2_decode};

// ─── E4M3FN ──────────────────────────────────────────────────────────────

static FP8_E4M3_LUT: OnceLock<[f32; 256]> = OnceLock::new();

/// Return the static 256-entry E4M3FN decode LUT.
///
/// On the first call, the table is populated by calling
/// [`oxibonsai_core::fp8_e4m3_decode`] for all 256 byte values.
/// Subsequent calls return the already-initialised slice without any
/// synchronisation overhead (just an atomic load).
#[inline]
pub fn fp8_e4m3_lut() -> &'static [f32; 256] {
    FP8_E4M3_LUT.get_or_init(|| {
        let mut lut = [0.0_f32; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = fp8_e4m3_decode(i as u8);
        }
        lut
    })
}

// ─── E5M2 ────────────────────────────────────────────────────────────────

static FP8_E5M2_LUT: OnceLock<[f32; 256]> = OnceLock::new();

/// Return the static 256-entry E5M2 decode LUT.
///
/// On the first call, the table is populated by calling
/// [`oxibonsai_core::fp8_e5m2_decode`] for all 256 byte values.
/// Subsequent calls return the already-initialised slice without any
/// synchronisation overhead.
#[inline]
pub fn fp8_e5m2_lut() -> &'static [f32; 256] {
    FP8_E5M2_LUT.get_or_init(|| {
        let mut lut = [0.0_f32; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = fp8_e5m2_decode(i as u8);
        }
        lut
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// LUT entry for byte 0x00 (positive zero) should be 0.0 for both formats.
    #[test]
    fn lut_zero_byte_is_zero() {
        assert_eq!(fp8_e4m3_lut()[0], 0.0_f32);
        assert_eq!(fp8_e5m2_lut()[0], 0.0_f32);
    }

    /// Every LUT entry must match the scalar decode function exactly.
    #[test]
    fn lut_matches_scalar_decode_e4m3() {
        let lut = fp8_e4m3_lut();
        for i in 0_usize..256 {
            let scalar = fp8_e4m3_decode(i as u8);
            let lut_val = lut[i];
            // NaN != NaN, handle separately
            if scalar.is_nan() {
                assert!(
                    lut_val.is_nan(),
                    "byte {i:#04x}: expected NaN, got {lut_val}"
                );
            } else {
                assert_eq!(
                    lut_val, scalar,
                    "byte {i:#04x}: lut={lut_val} vs scalar={scalar}"
                );
            }
        }
    }

    /// Every LUT entry must match the scalar decode function exactly.
    #[test]
    fn lut_matches_scalar_decode_e5m2() {
        let lut = fp8_e5m2_lut();
        for i in 0_usize..256 {
            let scalar = fp8_e5m2_decode(i as u8);
            let lut_val = lut[i];
            if scalar.is_nan() {
                assert!(
                    lut_val.is_nan(),
                    "byte {i:#04x}: expected NaN, got {lut_val}"
                );
            } else if scalar.is_infinite() {
                assert_eq!(
                    lut_val.is_infinite(),
                    true,
                    "byte {i:#04x}: expected Inf, got {lut_val}"
                );
                assert_eq!(
                    lut_val.is_sign_positive(),
                    scalar.is_sign_positive(),
                    "byte {i:#04x}: sign mismatch"
                );
            } else {
                assert_eq!(
                    lut_val, scalar,
                    "byte {i:#04x}: lut={lut_val} vs scalar={scalar}"
                );
            }
        }
    }

    /// Two successive calls return the same address (singleton).
    #[test]
    fn lut_is_singleton() {
        let a = fp8_e4m3_lut() as *const _;
        let b = fp8_e4m3_lut() as *const _;
        assert_eq!(a, b, "multiple calls should return same static address");
    }
}
