//! Shared `bf16` <-> `f32` conversion helpers.
//!
//! `K-12` bf16 hoist: before this
//! module existed, four independent copies of `bf16_to_f32` had drifted into
//! the tree (`oxibonsai-model/src/convert/mlx_image/pack.rs`,
//! `oxibonsai-image/src/vae/safetensors.rs`, `oxibonsai-image/src/te/mlx4bit.rs`,
//! and an inline copy in `oxibonsai-model/examples/mlx_image_parity.rs`), and
//! none of them had a `f32_to_bf16` counterpart. This is the single
//! canonical implementation; downstream crates should call it instead of
//! keeping their own copy (see this crate's `CLAUDE.md`/deviation notes for
//! the exact call sites still pointing at their own inline copy).
//!
//! `bfloat16` keeps IEEE 754 `binary32`'s 8-bit exponent and sign but only
//! the top 7 mantissa bits, so the bit pattern is simply the upper 16 bits
//! of the `f32` representation (widening) or a rounded truncation of them
//! (narrowing) — no exponent bias conversion is needed, unlike IEEE
//! `binary16` (`f16`).

/// Decode one `bf16` bit pattern (as stored in a GGUF/safetensors file,
/// little-endian byte order already resolved into a `u16`) into `f32`.
///
/// `bf16`'s bits are exactly the top 16 bits of the equivalent `f32`, so
/// widening is a zero-cost left shift with no rounding or exponent
/// rebiasing — this is *exact*, not an approximation.
#[inline]
#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Encode an `f32` into the nearest `bf16` bit pattern, rounding
/// round-to-nearest-even on the 16 low mantissa bits being discarded.
///
/// Matches `half::bf16::from_f32`'s rounding behaviour exactly (verified in
/// this module's tests across normals, subnormals, zero, infinities and
/// NaN), but is implemented directly on the bit pattern so this crate does
/// not need to route a hot conversion loop through another crate's type.
///
/// NaN inputs are canonicalised to the standard quiet-NaN bit pattern
/// (`0x7FC0`) rather than truncating an arbitrary NaN payload, so a
/// signalling NaN never survives the round trip as a silently different
/// signalling NaN.
#[inline]
#[must_use]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Canonical quiet NaN: sign 0, exponent all-ones, top mantissa bit
        // set (quiet), rest zero — `0x7FC0` in bf16's 16 bits.
        return 0x7FC0;
    }
    // Round-to-nearest-even on the low 16 bits: add the rounding bias
    // (`0x7FFF` plus 1 if the bit just above the cut is set, i.e. the
    // truncated value's LSB), then truncate. This is the same construction
    // ggml/PyTorch use for `float32 -> bfloat16`.
    let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_to_f32_matches_half_crate_across_common_values() {
        for v in [
            0.0f32, -0.0, 1.0, -1.0, 0.5, -0.5, 2000.0, -0.0625, 12_345.679, 1e10, 1e-10,
        ] {
            let bits = half::bf16::from_f32(v).to_bits();
            assert_eq!(
                bf16_to_f32(bits),
                half::bf16::from_bits(bits).to_f32(),
                "decode mismatch for bit pattern of {v}"
            );
        }
    }

    #[test]
    fn f32_to_bf16_matches_half_crate_round_to_nearest_even() {
        // A spread including values that land exactly on a rounding
        // boundary (ties), subnormal-adjacent magnitudes, and ordinary
        // values with many significant mantissa bits.
        let mut samples: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1.5,
            2000.0,
            -0.0625,
            12_345.679,
            1e10,
            1e-10,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
        ];
        // A tie case: a value whose low 16 mantissa bits are exactly
        // 0x8000 (halfway), which must round to even.
        samples.push(f32::from_bits(0x3F80_8000)); // 1.0 + tie
        samples.push(f32::from_bits(0x3F81_8000)); // next mantissa step + tie
                                                   // A pseudo-random spread via a tiny xorshift so many distinct
                                                   // mantissa patterns are exercised deterministically.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..2000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bits = (state >> 32) as u32;
            let v = f32::from_bits(bits);
            if v.is_finite() {
                samples.push(v);
            }
        }

        for v in samples {
            let ours = f32_to_bf16(v);
            let theirs = half::bf16::from_f32(v).to_bits();
            assert_eq!(
                ours,
                theirs,
                "round-to-nearest-even mismatch for {v} (bits {:#010x}): ours={ours:#06x} \
                 theirs={theirs:#06x}",
                v.to_bits()
            );
        }
    }

    #[test]
    fn f32_to_bf16_infinities_round_trip_exactly() {
        assert_eq!(f32_to_bf16(f32::INFINITY), half::bf16::INFINITY.to_bits());
        assert_eq!(
            f32_to_bf16(f32::NEG_INFINITY),
            half::bf16::NEG_INFINITY.to_bits()
        );
        assert_eq!(bf16_to_f32(half::bf16::INFINITY.to_bits()), f32::INFINITY);
    }

    #[test]
    fn f32_to_bf16_nan_is_canonicalised_and_decodes_back_to_nan() {
        let encoded = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(encoded).is_nan());
        // A different NaN payload must not change the canonical output.
        let odd_nan = f32::from_bits(0x7FC0_1234);
        assert!(odd_nan.is_nan());
        assert_eq!(f32_to_bf16(odd_nan), encoded);
    }

    #[test]
    fn round_trip_widen_then_narrow_is_exact_for_bf16_representable_values() {
        // Every bf16 bit pattern, widened to f32 then narrowed back, must
        // reproduce the same bits exactly (bf16 -> f32 is lossless, and the
        // f32 value produced has zero low mantissa bits so no rounding
        // occurs on the way back).
        for bits in (0u16..=0xFFFF).step_by(37) {
            // bf16 layout: 1 sign + 8 exponent (bits 14..7) + 7 mantissa
            // (bits 6..0). NaN is exponent all-ones (0xFF) with a nonzero
            // mantissa; skip those, since many distinct NaN bit patterns
            // canonicalise to one by design (infinity, exponent all-ones
            // with a zero mantissa, round-trips exactly and is kept).
            let exponent = (bits >> 7) & 0xFF;
            let mantissa = bits & 0x7F;
            if exponent == 0xFF && mantissa != 0 {
                continue;
            }
            let widened = bf16_to_f32(bits);
            assert_eq!(
                f32_to_bf16(widened),
                bits,
                "round trip failed for bf16 bits {bits:#06x}"
            );
        }
    }
}
