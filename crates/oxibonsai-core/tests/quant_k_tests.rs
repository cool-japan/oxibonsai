//! Integration tests for Q2_K and Q4_K quantization formats.

use oxibonsai_core::quant_k::{BlockQ2K, BlockQ4K, BLOCK_Q2_K_BYTES, BLOCK_Q4_K_BYTES, QK_K};

/// Simple LCG PRNG for reproducible test data (no rand dependency).
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

fn generate_test_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| lcg(&mut state)).collect()
}

// -----------------------------------------------------------------------
// Q2_K tests
// -----------------------------------------------------------------------

#[test]
fn q2k_block_byte_size() {
    assert_eq!(std::mem::size_of::<BlockQ2K>(), BLOCK_Q2_K_BYTES);
    assert_eq!(BLOCK_Q2_K_BYTES, 84);
}

#[test]
fn q2k_zero_input_roundtrip() {
    let input = vec![0.0f32; QK_K];
    let blocks = BlockQ2K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 1);

    let mut output = vec![0.0f32; QK_K];
    BlockQ2K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output.iter().enumerate() {
        assert!(v.abs() < 0.01, "Q2_K zero roundtrip: index {i}, got {v}");
    }
}

#[test]
fn q2k_constant_positive_roundtrip() {
    let input = vec![1.0f32; QK_K];
    let blocks = BlockQ2K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ2K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 0.2,
            "Q2_K constant roundtrip: index {i}, expected ~1.0, got {v}"
        );
    }
}

#[test]
fn q2k_constant_negative_roundtrip() {
    let input = vec![-0.5f32; QK_K];
    let blocks = BlockQ2K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ2K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - (-0.5)).abs() < 0.2,
            "Q2_K negative roundtrip: index {i}, expected ~-0.5, got {v}"
        );
    }
}

#[test]
fn q2k_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_K, 42);
    let blocks = BlockQ2K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ2K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Q2_K is very coarse (2-bit), allow up to ~0.6 error on [-1,1] range
    assert!(
        max_err < 0.7,
        "Q2_K random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q2k_multiple_blocks() {
    let num_blocks = 4;
    let input = generate_test_data(QK_K * num_blocks, 123);
    let blocks = BlockQ2K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_K * num_blocks];
    BlockQ2K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 0.7,
        "Q2_K multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q2k_dequant_known_values() {
    // Construct a known block manually and verify dequant output
    use half::f16;

    let block = BlockQ2K {
        scales: {
            let mut s = [0u8; 16];
            // sub-block 0: scale=2, min=1 -> byte = 0x12
            s[0] = 0x12;
            // all others zero
            s
        },
        qs: {
            let mut q = [0u8; 64];
            // ggml's Q2_K layout is bit-plane, not element-sequential: at
            // shift=0, sub-block 0 (elements 0..16) reads one crumb from each
            // of bytes qs[0..16]. Setting qs[0..4] = 0xFF gives crumb 3 (q=3)
            // for elements 0..4 only; qs[4..16] stay 0, so elements 4..16
            // decode with crumb 0.
            for item in q[..4].iter_mut() {
                *item = 0xFF;
            }
            q
        },
        d: f16::from_f32(0.5),
        dmin: f16::from_f32(0.25),
    };

    let mut output = vec![0.0f32; QK_K];
    BlockQ2K::dequant(&[block], &mut output).expect("dequant should succeed");

    // Sub-block 0: w = d * scale * q - dmin * min = 0.5 * 2 * 3 - 0.25 * 1 = 2.75
    let expected = 0.5 * 2.0 * 3.0 - 0.25 * 1.0;
    for (i, &val) in output[..4].iter().enumerate() {
        assert!(
            (val - expected).abs() < 0.01,
            "Q2_K known dequant: index {i}, expected {expected}, got {val}"
        );
    }
    for (i, &val) in output[4..16].iter().enumerate() {
        assert!(
            (val + 0.25).abs() < 0.01,
            "Q2_K known dequant: index {} expected -0.25 (d*sc*0 - dmin*mn = 1.0*0 - 0.25), got {val}",
            i + 4
        );
    }

    // Remaining sub-blocks have scale=0, min=0, so all weights should be 0
    for (i, &val) in output.iter().enumerate().take(QK_K).skip(16) {
        assert!(
            val.abs() < 0.01,
            "Q2_K known dequant: index {i}, expected 0.0, got {}",
            val
        );
    }
}

#[test]
fn q2k_invalid_input_length() {
    let input = vec![0.0f32; 100]; // not a multiple of 256
    let result = BlockQ2K::quantize(&input);
    assert!(result.is_err());
}

#[test]
fn q2k_dequant_output_too_small() {
    let input = vec![0.0f32; QK_K];
    let blocks = BlockQ2K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; 10]; // too small
    let result = BlockQ2K::dequant(&blocks, &mut output);
    assert!(result.is_err());
}

// -----------------------------------------------------------------------
// Q4_K tests
// -----------------------------------------------------------------------

#[test]
fn q4k_block_byte_size() {
    assert_eq!(std::mem::size_of::<BlockQ4K>(), BLOCK_Q4_K_BYTES);
    assert_eq!(BLOCK_Q4_K_BYTES, 144);
}

#[test]
fn q4k_zero_input_roundtrip() {
    let input = vec![0.0f32; QK_K];
    let blocks = BlockQ4K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 1);

    let mut output = vec![0.0f32; QK_K];
    BlockQ4K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output.iter().enumerate() {
        assert!(v.abs() < 0.01, "Q4_K zero roundtrip: index {i}, got {v}");
    }
}

#[test]
fn q4k_constant_positive_roundtrip() {
    let input = vec![1.0f32; QK_K];
    let blocks = BlockQ4K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ4K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 0.1,
            "Q4_K constant roundtrip: index {i}, expected ~1.0, got {v}"
        );
    }
}

#[test]
fn q4k_constant_negative_roundtrip() {
    let input = vec![-0.5f32; QK_K];
    let blocks = BlockQ4K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ4K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - (-0.5)).abs() < 0.1,
            "Q4_K negative roundtrip: index {i}, expected ~-0.5, got {v}"
        );
    }
}

#[test]
fn q4k_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_K, 42);
    let blocks = BlockQ4K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ4K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Q4_K is 4-bit, should be much more precise than Q2_K
    assert!(
        max_err < 0.15,
        "Q4_K random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q4k_multiple_blocks() {
    let num_blocks = 4;
    let input = generate_test_data(QK_K * num_blocks, 999);
    let blocks = BlockQ4K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_K * num_blocks];
    BlockQ4K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 0.15,
        "Q4_K multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q4k_dequant_known_values() {
    use half::f16;
    use oxibonsai_core::quant_k::BlockQ4K;

    // Build a block with known scale/min and a single non-zero sub-block
    let sc = [4u8, 0, 0, 0, 0, 0, 0, 0];
    let mn = [2u8, 0, 0, 0, 0, 0, 0, 0];

    // Use the internal encode (we test encode/decode roundtrip separately)
    // Inline the encode logic since it's not pub
    let mut scales_raw = [0u8; 12];
    // Low 4 bits of scales
    scales_raw[0] = sc[0] & 0x0F | ((sc[1] & 0x0F) << 4);
    scales_raw[1] = sc[2] & 0x0F | ((sc[3] & 0x0F) << 4);
    scales_raw[2] = sc[4] & 0x0F | ((sc[5] & 0x0F) << 4);
    scales_raw[3] = sc[6] & 0x0F | ((sc[7] & 0x0F) << 4);
    // Low 4 bits of mins
    scales_raw[4] = mn[0] & 0x0F | ((mn[1] & 0x0F) << 4);
    scales_raw[5] = mn[2] & 0x0F | ((mn[3] & 0x0F) << 4);
    scales_raw[6] = mn[4] & 0x0F | ((mn[5] & 0x0F) << 4);
    scales_raw[7] = mn[6] & 0x0F | ((mn[7] & 0x0F) << 4);
    // Upper bits are all 0 since sc/mn values fit in 4 bits

    let mut qs = [0u8; 128];
    // ggml's Q4_K layout emits 32 LOW nibbles then 32 HIGH nibbles per
    // 64-element group: `qs[l] & 0xF` feeds elements 0..32, `qs[l] >> 4` feeds
    // elements 32..64. Setting qs[0..16] = 0x77 gives nibble 7 (low AND high)
    // for elements 0..16 (low-nibble pass) and 32..48 (high-nibble pass, but
    // scale/min pair 1 is zero so those decode as 0 regardless); qs[16..32]
    // stay 0, so elements 16..32 decode with nibble 0.
    for item in qs[..16].iter_mut() {
        *item = 0x77;
    }

    let block = BlockQ4K {
        d: f16::from_f32(0.5),
        dmin: f16::from_f32(0.25),
        scales: scales_raw,
        qs,
    };

    let mut output = vec![0.0f32; QK_K];
    BlockQ4K::dequant(&[block], &mut output).expect("dequant should succeed");

    // Sub-block 0: w = d * sc * q - dmin * mn = 0.5 * 4 * 7 - 0.25 * 2 = 13.5
    let expected = 0.5 * 4.0 * 7.0 - 0.25 * 2.0;
    for (i, &val) in output[..16].iter().enumerate() {
        assert!(
            (val - expected).abs() < 0.01,
            "Q4_K known dequant: index {i}, expected {expected}, got {val}"
        );
    }
    for (i, &val) in output[16..32].iter().enumerate() {
        assert!(
            (val + 0.5).abs() < 0.01,
            "Q4_K known dequant: index {} expected -0.5 (d1*0 - m1 = 2.0*0 - 0.5), got {val}",
            i + 16
        );
    }

    // Remaining sub-blocks: scale=0, min=0 -> w = 0
    for (i, &val) in output.iter().enumerate().take(QK_K).skip(32) {
        assert!(
            val.abs() < 0.01,
            "Q4_K known dequant: index {i}, expected 0.0, got {}",
            val
        );
    }
}

#[test]
fn q4k_invalid_input_length() {
    let input = vec![0.0f32; 100];
    let result = BlockQ4K::quantize(&input);
    assert!(result.is_err());
}

#[test]
fn q4k_dequant_output_too_small() {
    let input = vec![0.0f32; QK_K];
    let blocks = BlockQ4K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; 10];
    let result = BlockQ4K::dequant(&blocks, &mut output);
    assert!(result.is_err());
}

#[test]
fn q2k_large_values_roundtrip() {
    // Test with values in a wider range
    let mut input = vec![0.0f32; QK_K];
    let mut state = 7777u64;
    for v in input.iter_mut() {
        *v = lcg(&mut state) * 10.0; // range [-10, 10]
    }

    let blocks = BlockQ2K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ2K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Wider range means larger absolute error, but relative should be similar
    assert!(
        max_err < 7.0,
        "Q2_K large values max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q4k_large_values_roundtrip() {
    let mut input = vec![0.0f32; QK_K];
    let mut state = 8888u64;
    for v in input.iter_mut() {
        *v = lcg(&mut state) * 10.0;
    }

    let blocks = BlockQ4K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ4K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1.5,
        "Q4_K large values max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q2k_q4k_precision_ordering() {
    // Q4_K should be more precise than Q2_K on the same data
    let input = generate_test_data(QK_K, 54321);

    let q2_blocks = BlockQ2K::quantize(&input).expect("Q2_K quantize");
    let mut q2_output = vec![0.0f32; QK_K];
    BlockQ2K::dequant(&q2_blocks, &mut q2_output).expect("Q2_K dequant");

    let q4_blocks = BlockQ4K::quantize(&input).expect("Q4_K quantize");
    let mut q4_output = vec![0.0f32; QK_K];
    BlockQ4K::dequant(&q4_blocks, &mut q4_output).expect("Q4_K dequant");

    let q2_mse: f32 = input
        .iter()
        .zip(q2_output.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / QK_K as f32;

    let q4_mse: f32 = input
        .iter()
        .zip(q4_output.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / QK_K as f32;

    assert!(
        q4_mse < q2_mse,
        "Q4_K MSE ({q4_mse}) should be less than Q2_K MSE ({q2_mse})"
    );
}

// -----------------------------------------------------------------------
// Ternary type importability from crate root
// -----------------------------------------------------------------------

#[test]
fn ternary_types_importable_from_crate_root() {
    let _: oxibonsai_core::BlockTQ2_0_g128;
    let _: oxibonsai_core::BlockTQ2_0;
    let _: oxibonsai_core::TernaryCode;
    assert_eq!(oxibonsai_core::QK_TQ2_0_G128, 128);
    assert_eq!(oxibonsai_core::QK_TQ2_0, 256);
    assert_eq!(oxibonsai_core::BLOCK_TQ2_0_G128_BYTES, 34);
    assert_eq!(oxibonsai_core::BLOCK_TQ2_0_BYTES, 66);
}
