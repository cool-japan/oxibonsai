//! FP8 quantization block types: E4M3FN and E5M2.
//!
//! Two FP8 formats:
//! - `BlockFP8E4M3`: 32 weights × 1 byte (E4M3FN) + FP16 block scale (34 bytes total).
//! - `BlockFP8E5M2`: 32 weights × 1 byte (E5M2) + FP16 block scale (34 bytes total).
//!
//! E4M3FN: sign(1b) + exp(4b) + mantissa(3b), bias=7. No infinity; NaN = 0x7f/0xff.
//! E5M2:   sign(1b) + exp(5b) + mantissa(2b), bias=15. Has infinity; NaN = 0x7e.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of weights per FP8 block.
pub const QK_FP8: usize = 32;

/// Number of bytes per FP8 block (32 bytes qs + 2 bytes f16 scale).
pub const BLOCK_FP8_BYTES: usize = 34;

/// Maximum representable value in E4M3FN: exp=1111, man=110 = 2^8 × 1.75 = 448.0
pub const FP8_E4M3_MAX: f32 = 448.0;

/// Maximum representable value in E5M2: exp=11110, man=11 = 2^15 × 1.75 = 57344.0
pub const FP8_E5M2_MAX: f32 = 57344.0;

// ---------------------------------------------------------------------------
// E4M3FN scalar encode/decode
//
// Format: s[7] exp[6:3] man[2:0], bias=7
// Normal value:  (-1)^s × 2^(exp−7) × (1 + man/8)
// Denormal (exp=0): (-1)^s × 2^(-6) × (man/8)
// NaN: exp=0b1111 AND man=0b111 (only those two patterns: 0x7f, 0xff)
// No infinity (saturation to ±448.0 instead)
// ---------------------------------------------------------------------------

/// Encode a f32 value as an E4M3FN byte.
///
/// - NaN → 0x7f (canonical positive NaN)
/// - ±Infinity → saturate to ±448.0 → 0x7e / 0xfe
/// - Overflow (|x| > 448.0) → saturate to ±448.0
/// - Underflow (|x| < 2^(−10)) → 0x00 (or 0x80 for -0)
/// - Uses RNE (round-to-nearest-even) on truncated mantissa bits
pub fn fp8_e4m3_encode(x: f32) -> u8 {
    // Handle NaN → canonical NaN 0x7f
    if x.is_nan() {
        return 0x7f;
    }

    // Extract sign, handling negative zero
    let bits = x.to_bits();
    let sign_bit: u8 = if (bits >> 31) != 0 { 0x80 } else { 0x00 };
    let abs = f32::from_bits(bits & 0x7FFF_FFFF);

    // Handle infinity: saturate to ±448.0 = 0x7e with sign
    if x.is_infinite() {
        return sign_bit | 0x7e;
    }

    // Handle +0.0 and -0.0
    if abs == 0.0 {
        return sign_bit;
    }

    // Saturate overflow to ±448.0 = 0x7e with sign
    if abs >= FP8_E4M3_MAX {
        return sign_bit | 0x7e;
    }

    // Decode f32 exponent and mantissa
    // f32 format: s[31] exp[30:23] man[22:0], bias=127
    let f32_bits = abs.to_bits();
    let f32_exp = ((f32_bits >> 23) & 0xFF) as i32; // biased exponent
    let f32_man = f32_bits & 0x007F_FFFF; // 23-bit mantissa

    // E4M3FN bias=7, so actual exponent = f32_exp - 127
    // E4M3FN exponent (biased) = (f32_exp - 127) + 7 = f32_exp - 120
    let e4m3_exp_biased = f32_exp - 120; // may be negative for small numbers

    let encoded: u8 = if e4m3_exp_biased >= 1 {
        // Normal number range
        // exp ranges: 1..=15 (bias=7 so actual -6..=8)
        // But exp=15 + man=111 is NaN, so max normal is exp=15, man=110 (0x7e = 448.0)
        let exp_clamped = e4m3_exp_biased.min(15) as u8;

        // Extract top 3 bits of f32 mantissa as E4M3 mantissa (with RNE)
        // f32 has 23 mantissa bits, E4M3 has 3, so we need bits [22:20] with rounding
        // Round bit is bit 19, sticky bits are [18:0]
        let man3 = (f32_man >> 20) as u8; // top 3 bits
        let round_bit = (f32_man >> 19) & 1;
        let sticky_bits = f32_man & ((1 << 19) - 1);

        // RNE: round up if round_bit=1 and (sticky≠0 or man3 is odd)
        let round_up = round_bit == 1 && (sticky_bits != 0 || (man3 & 1) == 1);
        let man3_rounded = if round_up { man3 + 1 } else { man3 };

        // Handle mantissa overflow (man3 was 0b111 and rounded up → carry to exponent)
        if man3_rounded > 7 {
            // Mantissa overflowed, increment exponent
            let new_exp = exp_clamped + 1;
            if new_exp > 15 {
                // Would overflow beyond max — saturate to 0x7e (448.0)
                0x7e
            } else if new_exp == 15 {
                // exp=15 + man=000 is 256.0 (valid), but exp=15+man=111 is NaN
                // man3_rounded overflowed to 0, which gives man=000 — valid
                new_exp << 3
            } else {
                new_exp << 3
            }
        } else {
            (exp_clamped << 3) | man3_rounded
        }
    } else {
        // e4m3_exp_biased < 1: potential denormal or underflow.
        //
        // Denormal: 2^(-6) × (man/8), man ∈ {1..7}.
        // The smallest representable denormal is man=1 → 2^(-9) ≈ 0.001953125.
        // Underflow threshold (halfway to 0) = 2^(-10) ≈ 0.000977.
        // Values below 2^(-10) round to zero; values >= 2^(-10) round to a denormal.
        //
        // To find man: man = round(abs / 2^(-9)) = round(abs × 512).
        // If man rounds to 0, we underflow to ±0.
        // If man rounds to > 7, it would carry into the normal range
        // (abs is close to the normal boundary 2^(-6)).

        // denormal = 2^(-6) × (man/8), so man = abs × 8 × 2^6 = abs × 512
        let scaled = abs * 512.0;
        let man_int = scaled as u32;

        // RNE rounding
        let frac = scaled - (man_int as f32);
        let man_rounded = if frac > 0.5 {
            man_int + 1
        } else if (frac - 0.5).abs() < 1e-9 {
            // Exactly half: round to even
            if man_int & 1 == 1 {
                man_int + 1
            } else {
                man_int
            }
        } else {
            man_int
        };

        if man_rounded >= 8 {
            // Rounding carried us into the normal range: exp=1, man=0
            // 2^(-6) × (1 + 0/8) = 2^(-6)
            0x08 // exp=1 (0b0001), man=000
        } else {
            // Clamp: if man_rounded == 0, this is underflow → returns 0x00
            man_rounded as u8
        }
    };

    sign_bit | encoded
}

/// Decode an E4M3FN byte to f32.
///
/// - 0x7f / 0xff → f32::NAN
/// - exp=0 (denormal): (−1)^s × 2^(−6) × (man/8)
/// - Normal: (−1)^s × 2^(exp−7) × (1 + man/8)
pub fn fp8_e4m3_decode(byte: u8) -> f32 {
    let sign: f32 = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0F; // 4-bit exponent
    let man = byte & 0x07; // 3-bit mantissa

    // NaN: exp=0b1111 (15) AND man=0b111 (7)
    if exp == 15 && man == 7 {
        return f32::NAN;
    }

    if exp == 0 {
        // Denormal: value = (-1)^s × 2^(−6) × (man/8)
        if man == 0 {
            return sign * 0.0; // ±0
        }
        let denorm_val = (man as f32) / 8.0 * (2.0_f32).powi(-6);
        return sign * denorm_val;
    }

    // Normal: (-1)^s × 2^(exp−7) × (1 + man/8)
    let actual_exp = (exp as i32) - 7;
    let mantissa_factor = 1.0 + (man as f32) / 8.0;
    sign * (2.0_f32).powi(actual_exp) * mantissa_factor
}

// ---------------------------------------------------------------------------
// E5M2 scalar encode/decode
//
// Format: s[7] exp[6:2] man[1:0], bias=15
// Normal value:  (-1)^s × 2^(exp−15) × (1 + man/4)
// Denormal (exp=0): (-1)^s × 2^(-14) × (man/4)
// Infinity: exp=31, man=00 (0x7c positive, 0xfc negative)
// NaN: exp=31, man≠00 (canonical 0x7e = exp=11111, man=10)
// ---------------------------------------------------------------------------

/// Encode a f32 value as an E5M2 byte.
///
/// - NaN → 0x7e (canonical NaN: exp=11111, man=10)
/// - ±Infinity → 0x7c / 0xfc
/// - Overflow → ±Infinity
/// - Uses RNE on truncated mantissa bits
pub fn fp8_e5m2_encode(x: f32) -> u8 {
    // Handle NaN
    if x.is_nan() {
        return 0x7e;
    }

    let bits = x.to_bits();
    let sign_bit: u8 = if (bits >> 31) != 0 { 0x80 } else { 0x00 };
    let abs = f32::from_bits(bits & 0x7FFF_FFFF);

    // Handle infinity
    if x.is_infinite() {
        return sign_bit | 0x7c;
    }

    // Handle ±0
    if abs == 0.0 {
        return sign_bit;
    }

    // Overflow: |x| > max E5M2 normal → return ±infinity
    if abs > FP8_E5M2_MAX {
        return sign_bit | 0x7c;
    }

    // f32 biased exp and 23-bit mantissa
    let f32_bits = abs.to_bits();
    let f32_exp = ((f32_bits >> 23) & 0xFF) as i32;
    let f32_man = f32_bits & 0x007F_FFFF;

    // E5M2 bias=15, so e5m2_exp_biased = (f32_exp - 127) + 15 = f32_exp - 112
    let e5m2_exp_biased = f32_exp - 112;

    let encoded: u8 = if e5m2_exp_biased >= 1 {
        // Normal number
        let exp_clamped = e5m2_exp_biased.min(30) as u8; // max normal exp = 30

        // Extract top 2 bits of f32 mantissa as E5M2 mantissa
        // Round bit is bit 20, sticky is [19:0]
        let man2 = (f32_man >> 21) as u8; // top 2 bits: values 0..3
        let round_bit = (f32_man >> 20) & 1;
        let sticky_bits = f32_man & ((1 << 20) - 1);

        let round_up = round_bit == 1 && (sticky_bits != 0 || (man2 & 1) == 1);
        let man2_rounded = if round_up { man2 + 1 } else { man2 };

        if man2_rounded > 3 {
            // Mantissa overflow → increment exponent
            let new_exp = exp_clamped + 1;
            if new_exp >= 31 {
                // Overflow to infinity
                0x7c
            } else {
                new_exp << 2
            }
        } else {
            (exp_clamped << 2) | man2_rounded
        }
    } else {
        // e5m2_exp_biased < 1: potential denormal or underflow.
        //
        // Denormal: 2^(-14) × (man/4), man ∈ {1..3}.
        // Smallest denormal: man=1 → 2^(-16).
        // Underflow threshold: 2^(-17) (halfway to 0).
        //
        // man = round(abs / 2^(-16)) = round(abs × 65536).
        // If man rounds to 0 → underflow to 0.
        // If man rounds to >= 4 → carry into normal range (exp=1, man=0).
        let scaled = abs * 65536.0; // abs × 2^16
        let man_int = scaled as u32;
        let frac = scaled - (man_int as f32);

        let man_rounded = if frac > 0.5 {
            man_int + 1
        } else if (frac - 0.5).abs() < 1e-9 {
            if man_int & 1 == 1 {
                man_int + 1
            } else {
                man_int
            }
        } else {
            man_int
        };

        if man_rounded >= 4 {
            // Carry into normal range: exp=1, man=0
            0x04 // exp=1 (0b00001), man=00
        } else {
            man_rounded as u8
        }
    };

    sign_bit | encoded
}

/// Decode an E5M2 byte to f32.
///
/// - exp=31, man=0 → ±infinity
/// - exp=31, man≠0 → NaN
/// - exp=0 (denormal): (−1)^s × 2^(−14) × (man/4)
/// - Normal: (−1)^s × 2^(exp−15) × (1 + man/4)
pub fn fp8_e5m2_decode(byte: u8) -> f32 {
    let sign: f32 = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 2) & 0x1F; // 5-bit exponent
    let man = byte & 0x03; // 2-bit mantissa

    if exp == 31 {
        if man == 0 {
            // ±infinity
            return if sign < 0.0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
        // NaN
        return f32::NAN;
    }

    if exp == 0 {
        if man == 0 {
            return sign * 0.0; // ±0
        }
        // Denormal: (-1)^s × 2^(-14) × (man/4)
        let val = (man as f32) / 4.0 * (2.0_f32).powi(-14);
        return sign * val;
    }

    // Normal: (-1)^s × 2^(exp−15) × (1 + man/4)
    let actual_exp = (exp as i32) - 15;
    let mantissa_factor = 1.0 + (man as f32) / 4.0;
    sign * (2.0_f32).powi(actual_exp) * mantissa_factor
}

// ---------------------------------------------------------------------------
// BlockFP8E4M3
// ---------------------------------------------------------------------------

/// FP8 E4M3FN block: 32 weights × 1 byte + FP16 block scale.
///
/// Layout (34 bytes): `qs[32]` E4M3FN-encoded weights + `d` FP16 block scale.
/// Actual value of weight i = d × fp8_e4m3_decode(qs\[i\]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockFP8E4M3 {
    /// 32 × E4M3FN-encoded weights.
    pub qs: [u8; 32],
    /// Block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockFP8E4M3>() == BLOCK_FP8_BYTES);

impl BlockFP8E4M3 {
    /// Dequantize a slice of FP8 E4M3FN blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_FP8`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_FP8;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E4M3 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_FP8;
            for (j, &q) in block.qs.iter().enumerate() {
                output[base + j] = d * fp8_e4m3_decode(q);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into FP8 E4M3FN blocks.
    ///
    /// Input length must be a multiple of `QK_FP8` (32).
    pub fn quantize(values: &[f32]) -> BonsaiResult<Vec<Self>> {
        if values.len() % QK_FP8 != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E4M3 quantize: input len {} not a multiple of {}",
                    values.len(),
                    QK_FP8
                ),
            });
        }
        let num_blocks = values.len() / QK_FP8;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_FP8;
            let chunk = &values[base..base + QK_FP8];

            let max_abs = chunk
                .iter()
                .filter(|v| !v.is_nan())
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);

            if max_abs == 0.0 {
                blocks.push(BlockFP8E4M3 {
                    qs: [0u8; 32],
                    d: f16::ZERO,
                });
                continue;
            }

            let d_f32 = max_abs / FP8_E4M3_MAX;
            let d = f16::from_f32(d_f32);
            // Use the f16-rounded scale for encoding to maintain dequant consistency
            let d_f32_actual = d.to_f32();

            let mut qs = [0u8; 32];
            for (j, &val) in chunk.iter().enumerate() {
                let scaled = if d_f32_actual == 0.0 {
                    0.0
                } else {
                    val / d_f32_actual
                };
                qs[j] = fp8_e4m3_encode(scaled);
            }

            blocks.push(BlockFP8E4M3 { qs, d });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockFP8E4M3`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_FP8_BYTES` (34)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_FP8_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E4M3 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_FP8_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E4M3 slice_from_bytes: pointer not {}-byte aligned",
                    align
                ),
            });
        }
        let count = data.len() / BLOCK_FP8_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and alignment
        // checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

// ---------------------------------------------------------------------------
// BlockFP8E5M2
// ---------------------------------------------------------------------------

/// FP8 E5M2 block: 32 weights × 1 byte + FP16 block scale.
///
/// Layout (34 bytes): `qs[32]` E5M2-encoded weights + `d` FP16 block scale.
/// Actual value of weight i = d × fp8_e5m2_decode(qs\[i\]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockFP8E5M2 {
    /// 32 × E5M2-encoded weights.
    pub qs: [u8; 32],
    /// Block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockFP8E5M2>() == BLOCK_FP8_BYTES);

impl BlockFP8E5M2 {
    /// Dequantize a slice of FP8 E5M2 blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_FP8`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_FP8;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E5M2 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_FP8;
            for (j, &q) in block.qs.iter().enumerate() {
                output[base + j] = d * fp8_e5m2_decode(q);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into FP8 E5M2 blocks.
    ///
    /// Input length must be a multiple of `QK_FP8` (32).
    pub fn quantize(values: &[f32]) -> BonsaiResult<Vec<Self>> {
        if values.len() % QK_FP8 != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E5M2 quantize: input len {} not a multiple of {}",
                    values.len(),
                    QK_FP8
                ),
            });
        }
        let num_blocks = values.len() / QK_FP8;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_FP8;
            let chunk = &values[base..base + QK_FP8];

            let max_abs = chunk
                .iter()
                .filter(|v| !v.is_nan())
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);

            if max_abs == 0.0 {
                blocks.push(BlockFP8E5M2 {
                    qs: [0u8; 32],
                    d: f16::ZERO,
                });
                continue;
            }

            let d_f32 = max_abs / FP8_E5M2_MAX;
            let d = f16::from_f32(d_f32);
            let d_f32_actual = d.to_f32();

            let mut qs = [0u8; 32];
            for (j, &val) in chunk.iter().enumerate() {
                let scaled = if d_f32_actual == 0.0 {
                    0.0
                } else {
                    val / d_f32_actual
                };
                qs[j] = fp8_e5m2_encode(scaled);
            }

            blocks.push(BlockFP8E5M2 { qs, d });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockFP8E5M2`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_FP8_BYTES` (34)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_FP8_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E5M2 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_FP8_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "FP8 E5M2 slice_from_bytes: pointer not {}-byte aligned",
                    align
                ),
            });
        }
        let count = data.len() / BLOCK_FP8_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and alignment
        // checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Block size assertions ──────────────────────────────────────────────

    #[test]
    fn block_fp8_e4m3_size() {
        assert_eq!(std::mem::size_of::<BlockFP8E4M3>(), 34);
    }

    #[test]
    fn block_fp8_e5m2_size() {
        assert_eq!(std::mem::size_of::<BlockFP8E5M2>(), 34);
    }

    #[test]
    fn block_fp8_bytes_constant() {
        assert_eq!(BLOCK_FP8_BYTES, 34);
    }

    #[test]
    fn qk_fp8_constant() {
        assert_eq!(QK_FP8, 32);
    }

    // ── E4M3 encode/decode ─────────────────────────────────────────────────

    #[test]
    fn e4m3_encode_zero() {
        assert_eq!(fp8_e4m3_encode(0.0), 0x00);
    }

    #[test]
    fn e4m3_encode_neg_zero() {
        assert_eq!(fp8_e4m3_encode(-0.0), 0x80);
    }

    #[test]
    fn e4m3_encode_nan() {
        assert_eq!(fp8_e4m3_encode(f32::NAN), 0x7f);
    }

    #[test]
    fn e4m3_encode_inf_saturates() {
        // +infinity saturates to 0x7e (448.0)
        assert_eq!(fp8_e4m3_encode(f32::INFINITY), 0x7e);
    }

    #[test]
    fn e4m3_encode_neg_inf_saturates() {
        // -infinity saturates to 0xfe (-448.0)
        assert_eq!(fp8_e4m3_encode(f32::NEG_INFINITY), 0xfe);
    }

    #[test]
    fn e4m3_encode_max() {
        // 448.0 is the maximum representable E4M3FN value = 0x7e
        assert_eq!(fp8_e4m3_encode(448.0), 0x7e);
    }

    #[test]
    fn e4m3_encode_neg_max() {
        assert_eq!(fp8_e4m3_encode(-448.0), 0xfe);
    }

    #[test]
    fn e4m3_encode_one() {
        let enc = fp8_e4m3_encode(1.0);
        let dec = fp8_e4m3_decode(enc);
        assert!((dec - 1.0).abs() < 0.01, "1.0 round-trip: got {dec}");
    }

    #[test]
    fn e4m3_decode_nan() {
        assert!(fp8_e4m3_decode(0x7f).is_nan());
    }

    #[test]
    fn e4m3_decode_neg_nan() {
        assert!(fp8_e4m3_decode(0xff).is_nan());
    }

    #[test]
    fn e4m3_decode_zero() {
        assert_eq!(fp8_e4m3_decode(0x00), 0.0);
    }

    #[test]
    fn e4m3_decode_neg_zero() {
        let v = fp8_e4m3_decode(0x80);
        assert_eq!(v, 0.0); // value is 0 (sign doesn't affect equality check for ±0)
        assert!(v.to_bits() == f32::from_bits(0x8000_0000_u32).to_bits()); // check it's -0
    }

    #[test]
    fn e4m3_decode_max() {
        // 0x7e: exp=0b1111=15, man=0b110=6 → 2^8 × (1 + 6/8) = 256 × 1.75 = 448
        let v = fp8_e4m3_decode(0x7e);
        assert!(
            (v - 448.0).abs() < 0.01,
            "0x7e should decode to 448.0, got {v}"
        );
    }

    #[test]
    fn e4m3_round_trip_values() {
        for &v in &[
            0.0f32, 0.5, 1.0, -1.0, 2.0, -2.0, 100.0, 448.0, -448.0, 0.125,
        ] {
            let enc = fp8_e4m3_encode(v);
            let dec = fp8_e4m3_decode(enc);
            let err = (dec - v).abs();
            let eps = 1.0_f32.max(v.abs()) * 0.25; // E4M3 has limited precision
            assert!(
                err <= eps,
                "e4m3 round-trip for {v}: enc={enc:#04x}, dec={dec}, err={err}"
            );
        }
    }

    #[test]
    fn e4m3_encode_decode_denormal() {
        // The smallest representable E4M3FN denormal is exp=0, man=1 → 2^(-6) × (1/8) = 2^(-9)
        let min_denorm = (2.0_f32).powi(-9); // ≈ 0.001953
        let enc = fp8_e4m3_encode(min_denorm);
        let dec = fp8_e4m3_decode(enc);
        // Should round-trip to the denormal value (man=1, exp=0 → 0x01)
        assert_eq!(enc, 0x01, "smallest denormal should be 0x01");
        assert!(
            (dec - min_denorm).abs() < min_denorm * 0.01,
            "denormal round-trip: got {dec}"
        );
    }

    #[test]
    fn e4m3_encode_overflow_above_max() {
        // Values above 448.0 should saturate to 0x7e
        let enc = fp8_e4m3_encode(1000.0);
        assert_eq!(enc, 0x7e, "1000.0 should saturate to 0x7e");
    }

    // ── E5M2 encode/decode ─────────────────────────────────────────────────

    #[test]
    fn e5m2_encode_zero() {
        assert_eq!(fp8_e5m2_encode(0.0), 0x00);
    }

    #[test]
    fn e5m2_encode_neg_zero() {
        assert_eq!(fp8_e5m2_encode(-0.0), 0x80);
    }

    #[test]
    fn e5m2_encode_inf() {
        assert_eq!(fp8_e5m2_encode(f32::INFINITY), 0x7c);
    }

    #[test]
    fn e5m2_encode_neg_inf() {
        assert_eq!(fp8_e5m2_encode(f32::NEG_INFINITY), 0xfc);
    }

    #[test]
    fn e5m2_encode_nan() {
        // NaN → 0x7e (canonical NaN: exp=11111, man=10)
        assert_eq!(fp8_e5m2_encode(f32::NAN), 0x7e);
    }

    #[test]
    fn e5m2_decode_inf() {
        assert_eq!(fp8_e5m2_decode(0x7c), f32::INFINITY);
    }

    #[test]
    fn e5m2_decode_neg_inf() {
        assert_eq!(fp8_e5m2_decode(0xfc), f32::NEG_INFINITY);
    }

    #[test]
    fn e5m2_decode_nan() {
        // 0x7e = exp=11111, man=10 → NaN
        assert!(fp8_e5m2_decode(0x7e).is_nan());
    }

    #[test]
    fn e5m2_decode_zero() {
        assert_eq!(fp8_e5m2_decode(0x00), 0.0);
    }

    #[test]
    fn e5m2_encode_max_normal() {
        // max E5M2 normal: exp=30, man=11 → 2^15 × 1.75 = 57344.0
        let enc = fp8_e5m2_encode(57344.0);
        let dec = fp8_e5m2_decode(enc);
        assert!(
            (dec - 57344.0).abs() < 1.0,
            "57344.0 should encode/decode correctly: got {dec}"
        );
    }

    #[test]
    fn e5m2_round_trip_values() {
        for &v in &[0.0f32, 1.0, -1.0, 2.0, 100.0, 1000.0] {
            let enc = fp8_e5m2_encode(v);
            let dec = fp8_e5m2_decode(enc);
            let err = (dec - v).abs();
            let eps = 1.0_f32.max(v.abs()) * 0.5;
            assert!(
                err <= eps,
                "e5m2 round-trip for {v}: enc={enc:#04x}, dec={dec}, err={err}"
            );
        }
    }

    #[test]
    fn e5m2_encode_overflow_to_infinity() {
        // Values above FP8_E5M2_MAX should map to infinity
        let enc = fp8_e5m2_encode(1_000_000.0);
        assert_eq!(enc, 0x7c, "overflow should become +infinity 0x7c");
    }

    // ── Block quantize/dequant tests ───────────────────────────────────────

    #[test]
    fn e4m3_block_quantize_dequant_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let blocks = BlockFP8E4M3::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockFP8E4M3::dequant(&blocks, &mut output).unwrap();
        let max_err: f32 = values
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 0.1, "E4M3 block round-trip max error: {max_err}");
    }

    #[test]
    fn e4m3_block_all_zeros() {
        let values = vec![0.0f32; 32];
        let blocks = BlockFP8E4M3::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockFP8E4M3::dequant(&blocks, &mut output).unwrap();
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn e4m3_quantize_wrong_len() {
        assert!(BlockFP8E4M3::quantize(&[1.0f32; 15]).is_err());
    }

    #[test]
    fn e4m3_dequant_too_small_buffer() {
        let blocks = BlockFP8E4M3::quantize(&[0.0f32; 32]).unwrap();
        let mut out = vec![0.0f32; 10];
        assert!(BlockFP8E4M3::dequant(&blocks, &mut out).is_err());
    }

    #[test]
    fn e4m3_slice_from_bytes_bad_len() {
        let data = vec![0u8; 35]; // 35 is not a multiple of 34
        assert!(BlockFP8E4M3::slice_from_bytes(&data).is_err());
    }

    #[test]
    fn e4m3_slice_from_bytes_aligned() {
        let block = BlockFP8E4M3 {
            qs: [0u8; 32],
            d: f16::from_f32(1.0),
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&block as *const BlockFP8E4M3).cast::<u8>(),
                BLOCK_FP8_BYTES,
            )
        };
        let result = BlockFP8E4M3::slice_from_bytes(bytes).expect("aligned slice should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].d, f16::from_f32(1.0));
    }

    #[test]
    fn e5m2_block_quantize_dequant_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 10.0 - 150.0).collect();
        let blocks = BlockFP8E5M2::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockFP8E5M2::dequant(&blocks, &mut output).unwrap();
        let max_err: f32 = values
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 20.0, "E5M2 block round-trip max error: {max_err}");
    }

    #[test]
    fn e5m2_block_all_zeros() {
        let values = vec![0.0f32; 32];
        let blocks = BlockFP8E5M2::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockFP8E5M2::dequant(&blocks, &mut output).unwrap();
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn e5m2_quantize_wrong_len() {
        assert!(BlockFP8E5M2::quantize(&[1.0f32; 17]).is_err());
    }

    #[test]
    fn e5m2_dequant_too_small_buffer() {
        let blocks = BlockFP8E5M2::quantize(&[0.0f32; 32]).unwrap();
        let mut out = vec![0.0f32; 5];
        assert!(BlockFP8E5M2::dequant(&blocks, &mut out).is_err());
    }

    #[test]
    fn e5m2_slice_from_bytes_bad_len() {
        let data = vec![0u8; 35]; // not a multiple of 34
        assert!(BlockFP8E5M2::slice_from_bytes(&data).is_err());
    }

    #[test]
    fn e5m2_slice_from_bytes_aligned() {
        let block = BlockFP8E5M2 {
            qs: [0u8; 32],
            d: f16::from_f32(2.0),
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&block as *const BlockFP8E5M2).cast::<u8>(),
                BLOCK_FP8_BYTES,
            )
        };
        let result = BlockFP8E5M2::slice_from_bytes(bytes).expect("aligned slice should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].d, f16::from_f32(2.0));
    }

    // ── Multiple blocks ────────────────────────────────────────────────────

    #[test]
    fn e4m3_multi_block() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5).collect();
        let blocks = BlockFP8E4M3::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 2);
        let mut output = vec![0.0f32; 64];
        BlockFP8E4M3::dequant(&blocks, &mut output).unwrap();
        // Check no NaN
        assert!(output.iter().all(|x| !x.is_nan()));
    }

    #[test]
    fn e5m2_multi_block() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32) * 100.0 - 3200.0).collect();
        let blocks = BlockFP8E5M2::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 2);
        let mut output = vec![0.0f32; 64];
        BlockFP8E5M2::dequant(&blocks, &mut output).unwrap();
        assert!(output.iter().all(|x| !x.is_nan()));
    }

    #[test]
    fn e4m3_block_count_correct() {
        let values = vec![1.0f32; 96]; // 3 blocks
        let blocks = BlockFP8E4M3::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn e5m2_block_count_correct() {
        let values = vec![1.0f32; 96]; // 3 blocks
        let blocks = BlockFP8E5M2::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 3);
    }

    // ── Constant value assertions ──────────────────────────────────────────

    #[test]
    fn fp8_e4m3_max_constant() {
        assert_eq!(FP8_E4M3_MAX, 448.0);
    }

    #[test]
    fn fp8_e5m2_max_constant() {
        assert_eq!(FP8_E5M2_MAX, 57344.0);
    }

    // ── Edge cases ─────────────────────────────────────────────────────────

    #[test]
    fn e4m3_all_valid_bytes_decode_no_panic() {
        // All 256 byte values should decode without panic
        for byte in 0u8..=255 {
            let _ = fp8_e4m3_decode(byte);
        }
    }

    #[test]
    fn e5m2_all_valid_bytes_decode_no_panic() {
        for byte in 0u8..=255 {
            let _ = fp8_e5m2_decode(byte);
        }
    }

    #[test]
    fn e4m3_encode_negative_one() {
        let enc = fp8_e4m3_encode(-1.0);
        assert!(enc & 0x80 != 0, "sign bit should be set for -1.0");
        let dec = fp8_e4m3_decode(enc);
        assert!((dec + 1.0).abs() < 0.01, "-1.0 round-trip: got {dec}");
    }

    #[test]
    fn e5m2_encode_negative_one() {
        let enc = fp8_e5m2_encode(-1.0);
        assert!(enc & 0x80 != 0, "sign bit should be set for -1.0");
        let dec = fp8_e5m2_decode(enc);
        assert!((dec + 1.0).abs() < 0.1, "-1.0 round-trip: got {dec}");
    }

    #[test]
    fn e4m3_block_scale_is_nonzero_for_nonzero_input() {
        let values = vec![1.0f32; 32];
        let blocks = BlockFP8E4M3::quantize(&values).unwrap();
        assert_ne!(blocks[0].d, f16::ZERO);
    }

    #[test]
    fn e5m2_block_scale_is_nonzero_for_nonzero_input() {
        let values = vec![100.0f32; 32];
        let blocks = BlockFP8E5M2::quantize(&values).unwrap();
        assert_ne!(blocks[0].d, f16::ZERO);
    }
}
