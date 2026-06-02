//! Pure (I/O-free) packing of an MLX-quantized linear module into
//! `BlockTQ2_0_g128` blocks.
//!
//! This is the parity-critical core of the FLUX.2 DiT converter. It is kept
//! free of file/safetensors handling so the synthetic unit tests can drive it
//! directly. See `super` for the byte-level rationale; the short version:
//!
//! * MLX stores a quantized linear as three sub-tensors with logical shapes
//!   `weight: u32[out, in/16]`, `scales: bf16[out, in/128]`,
//!   `biases: bf16[out, in/128]`.
//! * Each `u32` packs 16 two-bit codes, LSB-first, little-endian. For ternary
//!   solver output the codes are `q ∈ {0, 1, 2}` with `bias == -scale`, so the
//!   dequantized weight is `w = scale·(q-1) ∈ {-s, 0, +s}`.
//! * MLX code `q` maps bit-identically to OxiBonsai's `BlockTQ2_0_g128` codes
//!   (`q=0 → 0b00 → -1`, `q=1 → 0b01 → 0`, `q=2 → 0b10 → +1`), and the
//!   little-endian byte layout of 8 consecutive `u32` words equals OxiBonsai's
//!   `qs[32]` (4 codes/byte, LSB-first). So per 128-element group the block's
//!   `qs` is just the 32 LE bytes of those 8 words — after we validate every
//!   code is `≤ 2` and that `bias == -scale`.
//!
//! Block ordering is out-major: for output row `r` and group `g`, the block
//! lives at index `r * (in/128) + g`, matching the GEMM kernel's
//! `[out × in/128]` row-major expectation.

use half::f16;

use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;

use super::error::PackError;

/// Number of input features represented by one TQ2_0_g128 block (group size).
const GROUP_SIZE: usize = 128;

/// Number of 2-bit codes packed into one `u32` weight word.
const CODES_PER_U32: usize = 16;

/// Number of `u32` words spanning one 128-element group (`128 / 16`).
const U32_WORDS_PER_GROUP: usize = GROUP_SIZE / CODES_PER_U32; // = 8

/// Reinterpret a bfloat16 bit pattern as `f32`.
///
/// bfloat16 is the top 16 bits of an IEEE-754 `f32`, so the conversion is an
/// exact left-shift by 16 with no rounding.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Pack one MLX-quantized linear module into `BlockTQ2_0_g128` blocks.
///
/// # Arguments
///
/// * `module` — diffusers module name, used only for error reporting.
/// * `weight` — row-major `u32[out, in/16]`, 16 two-bit codes per word.
/// * `scales` — row-major bf16 bit patterns `[out, in/128]`.
/// * `biases` — row-major bf16 bit patterns `[out, in/128]`.
/// * `out_features` — output dimension (`out`).
/// * `in_features` — input dimension (`in`), must be a positive multiple of 128.
///
/// # Returns
///
/// A `Vec<BlockTQ2_0_g128>` of length `out * (in / 128)`, ordered out-major.
///
/// # Errors
///
/// Returns a [`PackError`] if any shape is inconsistent, if any 2-bit code
/// exceeds 2 (a reserved `q=3`), or if any group's `bias != -scale` (the
/// affine→symmetric-ternary parity guards).
pub fn pack_quantized_module(
    module: &str,
    weight: &[u32],
    scales: &[u16],
    biases: &[u16],
    out_features: usize,
    in_features: usize,
) -> Result<Vec<BlockTQ2_0_g128>, PackError> {
    // ── Shape validation ─────────────────────────────────────────────────────
    if in_features == 0 || in_features % GROUP_SIZE != 0 {
        return Err(PackError::InFeaturesNotAligned {
            module: module.to_string(),
            in_features,
        });
    }

    let weight_cols = in_features / CODES_PER_U32; // in/16
    let group_cols = in_features / GROUP_SIZE; // in/128

    let expected_weight = out_features * weight_cols;
    if weight.len() != expected_weight {
        return Err(PackError::BufferLengthMismatch {
            module: module.to_string(),
            which: "weight",
            got: weight.len(),
            expected: expected_weight,
        });
    }
    let expected_groups = out_features * group_cols;
    if scales.len() != expected_groups {
        return Err(PackError::BufferLengthMismatch {
            module: module.to_string(),
            which: "scales",
            got: scales.len(),
            expected: expected_groups,
        });
    }
    if biases.len() != expected_groups {
        return Err(PackError::BufferLengthMismatch {
            module: module.to_string(),
            which: "biases",
            got: biases.len(),
            expected: expected_groups,
        });
    }

    // ── Pack, out-major ──────────────────────────────────────────────────────
    let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(expected_groups);

    for row in 0..out_features {
        let weight_row_base = row * weight_cols;
        let group_row_base = row * group_cols;

        for group in 0..group_cols {
            // The 8 u32 words covering input features [group*128 .. group*128+128]
            // = weight columns [group*8 .. group*8+8].
            let word_base = weight_row_base + group * U32_WORDS_PER_GROUP;
            let words = &weight[word_base..word_base + U32_WORDS_PER_GROUP];

            // Parity guard #1: every 2-bit code must be ≤ 2 (no reserved q=3).
            // We must inspect every code anyway, so validate-then-copy.
            for &word in words {
                for lane in 0..CODES_PER_U32 {
                    let code = ((word >> (lane * 2)) & 0x3) as u8;
                    if code > 2 {
                        return Err(PackError::CodeOutOfRange {
                            module: module.to_string(),
                            row,
                            group,
                            value: code,
                        });
                    }
                }
            }

            // Parity guard #2: bias == -scale exactly (float compare, so that
            // scale == 0 with bias bits 0x0000 vs -0.0 = 0x8000 still matches).
            let scale_bits = scales[group_row_base + group];
            let bias_bits = biases[group_row_base + group];
            let scale = bf16_to_f32(scale_bits);
            let bias = bf16_to_f32(bias_bits);
            if bias != -scale {
                return Err(PackError::AsymmetricBias {
                    module: module.to_string(),
                    row,
                    group,
                    bias,
                    scale,
                });
            }

            // Codes validated → the 32 little-endian bytes of the 8 words are
            // exactly OxiBonsai's qs[32] (4 codes/byte, LSB-first).
            let mut qs = [0u8; 32];
            for (w_idx, &word) in words.iter().enumerate() {
                let le = word.to_le_bytes();
                let dst = w_idx * 4;
                qs[dst..dst + 4].copy_from_slice(&le);
            }

            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(scale),
            });
        }
    }

    Ok(blocks)
}

/// Convert an `f32` value to a bfloat16 bit pattern (round-to-nearest-even).
///
/// Used by tests to build synthetic MLX scale/bias buffers from chosen f32
/// values. Mirrors `half::bf16::from_f32`'s rounding.
#[cfg(test)]
pub fn f32_to_bf16(value: f32) -> u16 {
    half::bf16::from_f32(value).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;

    /// Pack a single group (8 u32 words) from a slice of 128 MLX codes.
    ///
    /// `codes[j]` is the MLX `q` value for input feature `j` (LSB-first within
    /// each word). Returns the 8 little-endian `u32` words.
    fn words_from_codes(codes: &[u8; 128]) -> [u32; 8] {
        let mut words = [0u32; 8];
        for (j, &q) in codes.iter().enumerate() {
            let word = j / 16;
            let lane = j % 16;
            words[word] |= (q as u32) << (lane * 2);
        }
        words
    }

    /// Build a single-row, single-group module with a uniform code and scale.
    fn single_group_uniform(q: u8, scale: f32) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
        let codes = [q; 128];
        let words = words_from_codes(&codes);
        let weight = words.to_vec();
        let scales = vec![f32_to_bf16(scale)];
        let biases = vec![f32_to_bf16(-scale)];
        (weight, scales, biases)
    }

    #[test]
    fn all_q0_gives_negative_scale() {
        let scale = 0.125_f32; // exactly representable in bf16
        let (w, s, b) = single_group_uniform(0, scale);
        let blocks =
            pack_quantized_module("test.q0", &w, &s, &b, 1, 128).expect("pack should succeed");
        assert_eq!(blocks.len(), 1);

        let mut out = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut out).expect("dequant");
        for &v in &out {
            assert_eq!(v, -scale, "q=0 → -scale");
        }
    }

    #[test]
    fn all_q2_gives_positive_scale() {
        let scale = 0.0625_f32;
        let (w, s, b) = single_group_uniform(2, scale);
        let blocks =
            pack_quantized_module("test.q2", &w, &s, &b, 1, 128).expect("pack should succeed");
        let mut out = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut out).expect("dequant");
        for &v in &out {
            assert_eq!(v, scale, "q=2 → +scale");
        }
    }

    #[test]
    fn all_q1_gives_zero() {
        let scale = 0.5_f32;
        let (w, s, b) = single_group_uniform(1, scale);
        let blocks =
            pack_quantized_module("test.q1", &w, &s, &b, 1, 128).expect("pack should succeed");
        let mut out = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut out).expect("dequant");
        for &v in &out {
            assert_eq!(v, 0.0, "q=1 → 0");
        }
    }

    #[test]
    fn mixed_pattern_within_group() {
        let scale = 0.25_f32;
        // Pattern of q codes repeated to fill 128: 0,1,2 → -s,0,+s.
        let mut codes = [0u8; 128];
        for (j, c) in codes.iter_mut().enumerate() {
            *c = (j % 3) as u8;
        }
        let words = words_from_codes(&codes);
        let weight = words.to_vec();
        let scales = vec![f32_to_bf16(scale)];
        let biases = vec![f32_to_bf16(-scale)];

        let blocks = pack_quantized_module("test.mixed", &weight, &scales, &biases, 1, 128)
            .expect("pack should succeed");
        let mut out = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut out).expect("dequant");

        for (j, &v) in out.iter().enumerate() {
            let expected = match j % 3 {
                0 => -scale,
                1 => 0.0,
                _ => scale,
            };
            assert_eq!(v, expected, "index {j}: q={} mismatch", j % 3);
        }
    }

    #[test]
    fn multi_row_multi_group_roundtrip_exact() {
        // out=4, in=256 → 2 groups per row, 8 blocks total.
        let out = 4usize;
        let in_features = 256usize;
        let group_cols = in_features / 128; // 2
        let weight_cols = in_features / 16; // 16

        // Choose a distinct, bf16-exact scale per (row, group) and a per-element
        // code derived deterministically.
        let mut weight = vec![0u32; out * weight_cols];
        let mut scales = vec![0u16; out * group_cols];
        let mut biases = vec![0u16; out * group_cols];

        // Reference: expected dequantized weight w[row, col] = scale·(q-1).
        let mut expected = vec![0.0f32; out * in_features];

        for row in 0..out {
            for g in 0..group_cols {
                // bf16-exact scale: powers-of-two fractions.
                let scale = 1.0_f32 / ((1 << (row + g + 1)) as f32);
                scales[row * group_cols + g] = f32_to_bf16(scale);
                biases[row * group_cols + g] = f32_to_bf16(-scale);

                // Build 128 codes for this group.
                let mut codes = [0u8; 128];
                for (j, c) in codes.iter_mut().enumerate() {
                    let q = ((row + g + j) % 3) as u8;
                    *c = q;
                    let col = g * 128 + j;
                    expected[row * in_features + col] = scale * (q as f32 - 1.0);
                }
                let words = words_from_codes(&codes);
                let word_base = row * weight_cols + g * 8;
                weight[word_base..word_base + 8].copy_from_slice(&words);
            }
        }

        let blocks =
            pack_quantized_module("test.multi", &weight, &scales, &biases, out, in_features)
                .expect("pack should succeed");
        assert_eq!(blocks.len(), out * group_cols);

        // Dequant all blocks (out-major: row r group g → block r*group_cols+g,
        // each block covers 128 contiguous input features of that row).
        let mut deq = vec![0.0f32; out * in_features];
        for row in 0..out {
            for g in 0..group_cols {
                let blk = &blocks[row * group_cols + g..row * group_cols + g + 1];
                let mut tmp = vec![0.0f32; 128];
                BlockTQ2_0_g128::dequant(blk, &mut tmp).expect("dequant");
                let base = row * in_features + g * 128;
                deq[base..base + 128].copy_from_slice(&tmp);
            }
        }

        for (idx, (&a, &e)) in deq.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "element {idx}: dequant {a} != expected {e}");
        }
    }

    #[test]
    fn errors_on_code_value_3() {
        let scale = 0.25_f32;
        // One code set to 3 (reserved q=3).
        let mut codes = [1u8; 128];
        codes[5] = 3;
        let words = words_from_codes(&codes);
        let weight = words.to_vec();
        let scales = vec![f32_to_bf16(scale)];
        let biases = vec![f32_to_bf16(-scale)];

        let err = pack_quantized_module("test.bad_code", &weight, &scales, &biases, 1, 128)
            .expect_err("q=3 must error");
        match err {
            PackError::CodeOutOfRange { value, .. } => assert_eq!(value, 3),
            other => panic!("expected CodeOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn errors_on_asymmetric_bias() {
        let scale = 0.25_f32;
        let (w, s, _b) = single_group_uniform(0, scale);
        // Bias deliberately != -scale.
        let biases = vec![f32_to_bf16(0.5_f32)];

        let err = pack_quantized_module("test.bad_bias", &w, &s, &biases, 1, 128)
            .expect_err("asymmetric bias must error");
        match err {
            PackError::AsymmetricBias {
                bias, scale: s_out, ..
            } => {
                assert_eq!(bias, 0.5);
                assert_eq!(s_out, scale);
            }
            other => panic!("expected AsymmetricBias, got {other:?}"),
        }
    }

    #[test]
    fn scale_zero_bias_negzero_is_accepted() {
        // scale = +0.0 (bits 0x0000), bias = -0.0 (bits 0x8000): bit-unequal but
        // float-equal, so the float compare must accept it.
        let codes = [1u8; 128];
        let words = words_from_codes(&codes);
        let weight = words.to_vec();
        let scales = vec![0x0000u16]; // +0.0
        let biases = vec![0x8000u16]; // -0.0
        let blocks = pack_quantized_module("test.zero", &weight, &scales, &biases, 1, 128)
            .expect("scale=0 / bias=-0 must be accepted");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].d, f16::from_f32(0.0));
    }

    #[test]
    fn errors_on_wrong_weight_columns() {
        // in=256 needs 16 weight cols/row; give 8.
        let weight = vec![0u32; 8];
        let scales = vec![f32_to_bf16(0.1); 2];
        let biases = vec![f32_to_bf16(-0.1); 2];
        let err = pack_quantized_module("test.shape", &weight, &scales, &biases, 1, 256)
            .expect_err("wrong weight length must error");
        assert!(matches!(
            err,
            PackError::BufferLengthMismatch {
                which: "weight",
                ..
            }
        ));
    }

    #[test]
    fn errors_on_unaligned_in_features() {
        let weight = vec![0u32; 8];
        let scales = vec![0u16; 1];
        let biases = vec![0u16; 1];
        let err = pack_quantized_module("test.align", &weight, &scales, &biases, 1, 100)
            .expect_err("in not multiple of 128 must error");
        assert!(matches!(err, PackError::InFeaturesNotAligned { .. }));
    }

    #[test]
    fn bf16_to_f32_roundtrip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -0.0625, 2000.0] {
            let bits = f32_to_bf16(v);
            // bf16 representable exactly for these → exact round-trip.
            assert_eq!(bf16_to_f32(bits), v, "value {v}");
        }
    }
}
