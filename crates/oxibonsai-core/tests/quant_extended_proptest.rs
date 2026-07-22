//! Randomized roundtrip / error-bound / sign-preservation tests for the
//! quantization formats that `quant_k_tests.rs` does not cover:
//! Q3_K, Q5_K, Q6_K, Q8_K, TQ2_0, TQ2_0_g128, FP8 E4M3, FP8 E5M2, Q4_0, Q8_0.
//!
//! Mirrors the LCG-seeded `generate_test_data` + bounded-max-error style used
//! by `quant_k_tests.rs` for the deterministic/statistical checks, and adds
//! true `proptest!`-driven property tests (already a dev-dependency of this
//! crate, see `tensor_property.rs`) for sign-preservation invariants — the
//! ternary formats (TQ2_0 / TQ2_0_g128) have a documented historical
//! sign-bug class around LSB-first 2-bit lane packing, so those get the
//! most exhaustive per-lane coverage.

use proptest::prelude::*;

use oxibonsai_core::quant_fp8::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};
use oxibonsai_core::quant_k::{BlockQ3K, BlockQ8K, QK_K};
use oxibonsai_core::quant_k_ext::{BlockQ5K, BlockQ6K};
use oxibonsai_core::quant_std::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};
use oxibonsai_core::quant_ternary::{BlockTQ2_0, BlockTQ2_0_g128, QK_TQ2_0, QK_TQ2_0_G128};

/// Simple LCG PRNG for reproducible test data (no rand dependency), matching
/// the helper in `quant_k_tests.rs`.
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

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ===========================================================================
// Q3_K
// ===========================================================================

#[test]
fn q3k_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_K, 42);
    let blocks = BlockQ3K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ3K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    // Q3_K is a 3-bit symmetric per-sub-block format; on [-1,1] data allow
    // generous headroom above Q2_K's own 0.7 bound proportionally reduced
    // for the extra bit (empirically well under 0.6).
    assert!(
        max_err < 0.6,
        "Q3_K random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q3k_multiple_blocks_random_roundtrip() {
    let num_blocks = 4;
    let input = generate_test_data(QK_K * num_blocks, 123);
    let blocks = BlockQ3K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_K * num_blocks];
    BlockQ3K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.6,
        "Q3_K multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q3k_sign_preservation_canary() {
    // Values well above the quantization noise floor must roundtrip with
    // the same sign as the input.
    let mut input = vec![0.0f32; QK_K];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockQ3K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ3K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "Q3_K sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn q3k_invalid_input_length_errors() {
    assert!(BlockQ3K::quantize(&vec![0.0f32; 100]).is_err());
}

#[test]
fn q3k_block_boundary_independent_scale() {
    // Two super-blocks with very different magnitudes must not leak scale
    // across the boundary.
    let mut input = vec![0.5f32; QK_K * 2];
    for v in input[QK_K..].iter_mut() {
        *v = 50.0;
    }
    let blocks = BlockQ3K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_K * 2];
    BlockQ3K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_K].iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 0.6,
            "Q3_K block 0 leaked block-1 scale at index {i}, got {v}"
        );
    }
}

// ===========================================================================
// Q5_K
// ===========================================================================

#[test]
fn q5k_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_K, 42);
    let blocks = BlockQ5K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ5K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    // Matches the tolerance already established for Q5_K's own deterministic
    // ramp test (quant_k_ext.rs: max_err < 0.2).
    assert!(
        max_err < 0.2,
        "Q5_K random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q5k_multiple_blocks_random_roundtrip() {
    let num_blocks = 4;
    let input = generate_test_data(QK_K * num_blocks, 999);
    let blocks = BlockQ5K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_K * num_blocks];
    BlockQ5K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.2,
        "Q5_K multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q5k_sign_preservation_canary() {
    let mut input = vec![0.0f32; QK_K];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockQ5K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ5K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "Q5_K sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn q5k_invalid_input_length_errors() {
    assert!(BlockQ5K::quantize(&vec![0.0f32; 100]).is_err());
}

#[test]
fn q5k_block_boundary_independent_scale() {
    let mut input = vec![0.5f32; QK_K * 2];
    for v in input[QK_K..].iter_mut() {
        *v = 50.0;
    }
    let blocks = BlockQ5K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_K * 2];
    BlockQ5K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_K].iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 0.2,
            "Q5_K block 0 leaked block-1 scale at index {i}, got {v}"
        );
    }
}

// ===========================================================================
// Q6_K
// ===========================================================================

#[test]
fn q6k_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_K, 42);
    let blocks = BlockQ6K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ6K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    // Matches the tolerance already established for Q6_K's own deterministic
    // ramp test (quant_k_ext.rs: max_err < 0.15).
    assert!(
        max_err < 0.15,
        "Q6_K random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q6k_multiple_blocks_random_roundtrip() {
    let num_blocks = 4;
    let input = generate_test_data(QK_K * num_blocks, 999);
    let blocks = BlockQ6K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_K * num_blocks];
    BlockQ6K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.15,
        "Q6_K multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q6k_sign_preservation_canary() {
    let mut input = vec![0.0f32; QK_K];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockQ6K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ6K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "Q6_K sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn q6k_invalid_input_length_errors() {
    assert!(BlockQ6K::quantize(&vec![0.0f32; 100]).is_err());
}

#[test]
fn q6k_block_boundary_independent_scale() {
    let mut input = vec![0.5f32; QK_K * 2];
    for v in input[QK_K..].iter_mut() {
        *v = 50.0;
    }
    let blocks = BlockQ6K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_K * 2];
    BlockQ6K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_K].iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 0.15,
            "Q6_K block 0 leaked block-1 scale at index {i}, got {v}"
        );
    }
}

// ===========================================================================
// Q8_K
// ===========================================================================

#[test]
fn q8k_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_K, 42);
    let blocks = BlockQ8K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ8K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    // 8-bit format: matches the tolerance already established for Q8_K's own
    // uniform-input test (src/quant_k.rs: max_err < 0.02).
    assert!(
        max_err < 0.02,
        "Q8_K random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q8k_multiple_blocks_random_roundtrip() {
    let num_blocks = 4;
    let input = generate_test_data(QK_K * num_blocks, 999);
    let blocks = BlockQ8K::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_K * num_blocks];
    BlockQ8K::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.02,
        "Q8_K multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q8k_sign_preservation_canary() {
    let mut input = vec![0.0f32; QK_K];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockQ8K::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_K];
    BlockQ8K::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "Q8_K sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn q8k_invalid_input_length_errors() {
    assert!(BlockQ8K::quantize(&vec![0.0f32; 100]).is_err());
}

// ===========================================================================
// TQ2_0_g128 (128-weight ternary, PrismML/Bonsai-shipped format)
// ===========================================================================

#[test]
fn tq2_0_g128_random_roundtrip_sign_and_magnitude() {
    let input = generate_test_data(QK_TQ2_0_G128, 42);
    let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_TQ2_0_G128];
    BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");

    let absmax = input.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
    let threshold = 0.5 * absmax;
    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        if x >= threshold {
            assert!(
                y > 0.0,
                "TQ2_0_g128 idx {i}: input {x} >= threshold should decode positive, got {y}"
            );
        } else if x <= -threshold {
            assert!(
                y < 0.0,
                "TQ2_0_g128 idx {i}: input {x} <= -threshold should decode negative, got {y}"
            );
        } else {
            assert_eq!(
                y, 0.0,
                "TQ2_0_g128 idx {i}: input {x} inside dead-zone should decode to 0, got {y}"
            );
        }
    }
}

/// Sign-preservation canary across every 2-bit lane position (0..4) within a
/// packing byte and every byte index (0..32) in the block: guards against
/// the historical sign-bug class where LSB-first bit-lane extraction could
/// be off-by-shift for some lanes.
#[test]
fn tq2_0_g128_sign_preservation_all_lanes_canary() {
    // Alternating +1/-1 (no zeros) exercises every lane/byte combination
    // with a known, checkable sign.
    let input: Vec<f32> = (0..QK_TQ2_0_G128)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_TQ2_0_G128];
    BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        let byte_idx = i / 4;
        let lane = i % 4;
        assert_eq!(
            x.signum(),
            y.signum(),
            "TQ2_0_g128 sign flipped at index {i} (byte {byte_idx}, lane {lane}): input {x}, output {y}"
        );
    }
}

#[test]
fn tq2_0_g128_all_zero_input_roundtrip() {
    let input = vec![0.0f32; QK_TQ2_0_G128];
    let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_TQ2_0_G128];
    BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");
    for (i, &v) in output.iter().enumerate() {
        assert_eq!(v, 0.0, "TQ2_0_g128 all-zero input: index {i}, got {v}");
    }
}

#[test]
fn tq2_0_g128_invalid_input_length_errors() {
    assert!(BlockTQ2_0_g128::quantize(&vec![0.0f32; 100]).is_err());
}

#[test]
fn tq2_0_g128_block_boundary_independent_sign() {
    // Two blocks: block 0 all-positive, block 1 all-negative. Verify no
    // cross-block bleed at the boundary.
    let mut input = vec![1.0f32; QK_TQ2_0_G128 * 2];
    for v in input[QK_TQ2_0_G128..].iter_mut() {
        *v = -1.0;
    }
    let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_TQ2_0_G128 * 2];
    BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_TQ2_0_G128].iter().enumerate() {
        assert!(
            v > 0.0,
            "TQ2_0_g128 block 0 index {i} should be positive, got {v}"
        );
    }
    for (i, &v) in output[QK_TQ2_0_G128..].iter().enumerate() {
        assert!(
            v < 0.0,
            "TQ2_0_g128 block 1 index {i} should be negative, got {v}"
        );
    }
}

proptest! {
    /// Randomized sign-preservation property: for every generated block, any
    /// weight at or above the block's positive threshold must decode
    /// strictly positive, and any weight at or below the negative threshold
    /// must decode strictly negative. Runs many proptest-generated cases,
    /// unlike the fixed-seed LCG tests above.
    #[test]
    fn tq2_0_g128_sign_property(
        values in prop::collection::vec(-10.0f32..10.0f32, QK_TQ2_0_G128),
    ) {
        let blocks = BlockTQ2_0_g128::quantize(&values).expect("quantize should succeed");
        let mut output = vec![0.0f32; QK_TQ2_0_G128];
        BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");

        let absmax = values.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
        if absmax > 0.0 {
            let threshold = 0.5 * absmax;
            for (i, (&x, &y)) in values.iter().zip(output.iter()).enumerate() {
                if x >= threshold {
                    prop_assert!(y > 0.0, "idx {i}: {x} >= threshold decoded non-positive {y}");
                } else if x <= -threshold {
                    prop_assert!(y < 0.0, "idx {i}: {x} <= -threshold decoded non-negative {y}");
                }
            }
        }
    }
}

// ===========================================================================
// TQ2_0 (256-weight ternary, llama.cpp-compat)
// ===========================================================================

#[test]
fn tq2_0_random_roundtrip_sign_and_magnitude() {
    let input = generate_test_data(QK_TQ2_0, 7);
    let blocks = BlockTQ2_0::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_TQ2_0];
    BlockTQ2_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    let absmax = input.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
    let threshold = 0.5 * absmax;
    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        if x >= threshold {
            assert!(
                y > 0.0,
                "TQ2_0 idx {i}: input {x} >= threshold should decode positive, got {y}"
            );
        } else if x <= -threshold {
            assert!(
                y < 0.0,
                "TQ2_0 idx {i}: input {x} <= -threshold should decode negative, got {y}"
            );
        } else {
            assert_eq!(
                y, 0.0,
                "TQ2_0 idx {i}: input {x} inside dead-zone should decode to 0, got {y}"
            );
        }
    }
}

#[test]
fn tq2_0_sign_preservation_all_lanes_canary() {
    let input: Vec<f32> = (0..QK_TQ2_0)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let blocks = BlockTQ2_0::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_TQ2_0];
    BlockTQ2_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        let byte_idx = i / 4;
        let lane = i % 4;
        assert_eq!(
            x.signum(),
            y.signum(),
            "TQ2_0 sign flipped at index {i} (byte {byte_idx}, lane {lane}): input {x}, output {y}"
        );
    }
}

#[test]
fn tq2_0_all_zero_input_roundtrip() {
    let input = vec![0.0f32; QK_TQ2_0];
    let blocks = BlockTQ2_0::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_TQ2_0];
    BlockTQ2_0::dequant(&blocks, &mut output).expect("dequant should succeed");
    for (i, &v) in output.iter().enumerate() {
        assert_eq!(v, 0.0, "TQ2_0 all-zero input: index {i}, got {v}");
    }
}

#[test]
fn tq2_0_invalid_input_length_errors() {
    assert!(BlockTQ2_0::quantize(&vec![0.0f32; 100]).is_err());
}

#[test]
fn tq2_0_block_boundary_independent_sign() {
    let mut input = vec![1.0f32; QK_TQ2_0 * 2];
    for v in input[QK_TQ2_0..].iter_mut() {
        *v = -1.0;
    }
    let blocks = BlockTQ2_0::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_TQ2_0 * 2];
    BlockTQ2_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_TQ2_0].iter().enumerate() {
        assert!(
            v > 0.0,
            "TQ2_0 block 0 index {i} should be positive, got {v}"
        );
    }
    for (i, &v) in output[QK_TQ2_0..].iter().enumerate() {
        assert!(
            v < 0.0,
            "TQ2_0 block 1 index {i} should be negative, got {v}"
        );
    }
}

proptest! {
    #[test]
    fn tq2_0_sign_property(
        values in prop::collection::vec(-10.0f32..10.0f32, QK_TQ2_0),
    ) {
        let blocks = BlockTQ2_0::quantize(&values).expect("quantize should succeed");
        let mut output = vec![0.0f32; QK_TQ2_0];
        BlockTQ2_0::dequant(&blocks, &mut output).expect("dequant should succeed");

        let absmax = values.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
        if absmax > 0.0 {
            let threshold = 0.5 * absmax;
            for (i, (&x, &y)) in values.iter().zip(output.iter()).enumerate() {
                if x >= threshold {
                    prop_assert!(y > 0.0, "idx {i}: {x} >= threshold decoded non-positive {y}");
                } else if x <= -threshold {
                    prop_assert!(y < 0.0, "idx {i}: {x} <= -threshold decoded non-negative {y}");
                }
            }
        }
    }
}

// ===========================================================================
// FP8 E4M3
// ===========================================================================

#[test]
fn fp8_e4m3_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_FP8, 42);
    let blocks = BlockFP8E4M3::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_FP8];
    BlockFP8E4M3::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    // Matches the tolerance already established in quant_fp8.rs's own
    // deterministic-ramp roundtrip test (max_err < 0.1 for range ~[-1.5, 1.5]).
    assert!(
        max_err < 0.1,
        "FP8 E4M3 random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn fp8_e4m3_multiple_blocks_random_roundtrip() {
    let num_blocks = 4;
    let input = generate_test_data(QK_FP8 * num_blocks, 999);
    let blocks = BlockFP8E4M3::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_FP8 * num_blocks];
    BlockFP8E4M3::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.1,
        "FP8 E4M3 multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn fp8_e4m3_sign_preservation_canary() {
    let mut input = vec![0.0f32; QK_FP8];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockFP8E4M3::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_FP8];
    BlockFP8E4M3::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "FP8 E4M3 sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn fp8_e4m3_invalid_input_length_errors() {
    assert!(BlockFP8E4M3::quantize(&[0.0f32; 15]).is_err());
}

#[test]
fn fp8_e4m3_block_boundary_independent_scale() {
    let mut input = vec![0.5f32; QK_FP8 * 2];
    for v in input[QK_FP8..].iter_mut() {
        *v = 200.0;
    }
    let blocks = BlockFP8E4M3::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_FP8 * 2];
    BlockFP8E4M3::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_FP8].iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 0.1,
            "FP8 E4M3 block 0 leaked block-1 scale at index {i}, got {v}"
        );
    }
}

// ===========================================================================
// FP8 E5M2
// ===========================================================================

#[test]
fn fp8_e5m2_random_roundtrip_error_bounded() {
    // Historical note: `BlockFP8E5M2::quantize` computes the block scale as
    // `d = f16::from_f32(max_abs / FP8_E5M2_MAX)` and re-derives
    // `d_f32_actual = d.to_f32()` to encode every element. f16 rounding could
    // round `d` *down* from the true ratio, making the max-magnitude element
    // compute `max_abs / d_f32_actual > FP8_E5M2_MAX` and encode as IEEE
    // Infinity (see `fp8_e5m2_block_scale_rounding_can_overflow_to_infinity`).
    // That defect is now fixed at the source (quantize clamps each finite
    // scaled value to ±FP8_E5M2_MAX), so this test no longer depends on a
    // hand-picked seed to stay finite; the finiteness assertion below is now a
    // genuine invariant check rather than a workaround.
    let input = generate_test_data(QK_FP8, 1);
    let blocks = BlockFP8E5M2::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_FP8];
    BlockFP8E5M2::dequant(&blocks, &mut output).expect("dequant should succeed");

    assert!(
        output.iter().all(|v| v.is_finite()),
        "FP8 E5M2 random roundtrip produced a non-finite value: {output:?}"
    );
    let max_err = max_abs_err(&input, &output);
    // E5M2 has only 2 mantissa bits; the existing deterministic test allows
    // ~13% relative error over a ±150 range, so allow generous headroom on
    // the [-1,1] LCG range used here.
    assert!(
        max_err < 0.25,
        "FP8 E5M2 random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn fp8_e5m2_multiple_blocks_random_roundtrip() {
    // The scale-rounding overflow-to-infinity defect referenced by
    // `fp8_e5m2_random_roundtrip_error_bounded` above is fixed at the source,
    // so any seed now stays finite across all blocks; the finiteness assertion
    // below is a real invariant check.
    let num_blocks = 4;
    let input = generate_test_data(QK_FP8 * num_blocks, 4);
    let blocks = BlockFP8E5M2::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_FP8 * num_blocks];
    BlockFP8E5M2::dequant(&blocks, &mut output).expect("dequant should succeed");

    assert!(
        output.iter().all(|v| v.is_finite()),
        "FP8 E5M2 multi-block roundtrip produced a non-finite value: {output:?}"
    );
    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.25,
        "FP8 E5M2 multi-block max error {max_err} exceeds tolerance"
    );
}

/// Regression test for a real defect (now fixed) discovered by the
/// randomized tests above: `BlockFP8E5M2::quantize` derives its block scale
/// as `d = f16::from_f32(max_abs / FP8_E5M2_MAX)`, then divides every element
/// by `d.to_f32()` (the *f16-rounded* scale) to encode it. When f16 rounding
/// took `d` below the true ratio, the very element that defined `max_abs`
/// computed `scaled = max_abs / d_f32_actual > FP8_E5M2_MAX`, which
/// `fp8_e5m2_encode` maps to `+Infinity` (0x7c) / `-Infinity` (0xfc) via its
/// overflow-to-infinity path — unlike E4M3FN, which has no Infinity encoding
/// and instead saturates to a finite value, so the same class of
/// scale-rounding error was silently absorbed there instead of corrupting a
/// weight to `inf`.
///
/// Empirically this triggered for roughly half of randomly seeded 32-element
/// blocks — not a rare corner case but a coin flip on ordinary data whenever
/// the max-magnitude element happened to round the scale down. The fix lives
/// in `src/quant_fp8.rs::BlockFP8E5M2::quantize`, which now clamps each
/// finite `scaled` value to `±FP8_E5M2_MAX` before calling `fp8_e5m2_encode`
/// (matching the E4M3FN sibling's finite-saturation convention while keeping
/// the round-to-nearest scale that is optimal for the block's bulk).
/// Genuinely infinite *input* still encodes to ±Inf — only the scale-rounding
/// overflow of finite inputs is eliminated.
///
/// Seed 42 reproduced the original failure (the max-magnitude element
/// dequantized to `-inf` instead of ≈ `-0.978`); this test now asserts the
/// fixed, finite behavior.
#[test]
fn fp8_e5m2_block_scale_rounding_can_overflow_to_infinity() {
    // Seed 42 was found (via the LCG generator used throughout this file)
    // to reliably reproduce the pre-fix defect: the element with the largest
    // magnitude in the block dequantized to -inf instead of ~-0.978.
    let input = generate_test_data(QK_FP8, 42);
    let blocks = BlockFP8E5M2::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_FP8];
    BlockFP8E5M2::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert!(
            y.is_finite(),
            "FP8 E5M2 quantize/dequant produced non-finite output at index {i}: \
             input={x}, output={y} (block scale rounded down; the fix must \
             saturate this element at FP8_E5M2_MAX instead of overflowing)"
        );
        // Sign must survive the saturation, and the previously-overflowing
        // max-magnitude element must decode close to its input rather than inf.
        assert_eq!(
            x.signum(),
            y.signum(),
            "FP8 E5M2 sign flipped at index {i}: input={x}, output={y}"
        );
    }
}

#[test]
fn fp8_e5m2_sign_preservation_canary() {
    let mut input = vec![0.0f32; QK_FP8];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockFP8E5M2::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_FP8];
    BlockFP8E5M2::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "FP8 E5M2 sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn fp8_e5m2_invalid_input_length_errors() {
    assert!(BlockFP8E5M2::quantize(&[0.0f32; 17]).is_err());
}

// ===========================================================================
// FP8 finite-input finiteness invariant (regression coverage for the
// block-scale f16-rounding overflow class in src/quant_fp8.rs)
// ===========================================================================

proptest! {
    /// Any finite input block must quantize+dequantize to an entirely finite
    /// output for E5M2 — no element may become ±Inf or NaN even when
    /// f16-rounding the block scale rounds it below the true ratio. Uses a
    /// wide magnitude range (down to tiny and up past FP8_E5M2_MAX) plus
    /// multiple blocks so the max-magnitude element lands in every regime.
    #[test]
    fn fp8_e5m2_finite_input_never_produces_non_finite(
        values in prop::collection::vec(-1.0e6f32..1.0e6f32, QK_FP8 * 3),
    ) {
        let blocks = BlockFP8E5M2::quantize(&values).expect("quantize should succeed");
        let mut output = vec![0.0f32; values.len()];
        BlockFP8E5M2::dequant(&blocks, &mut output).expect("dequant should succeed");
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "E5M2 finite input produced non-finite output at index {i}: \
                 input={}, output={y}",
                values[i]
            );
        }
    }

    /// E4M3FN counterpart: finite input must never dequantize to a non-finite
    /// value (E4M3FN has no Infinity and encodes NaN as 0x7f, so the failure
    /// mode would be an incorrect NaN / non-finite decode).
    #[test]
    fn fp8_e4m3_finite_input_never_produces_non_finite(
        values in prop::collection::vec(-1.0e6f32..1.0e6f32, QK_FP8 * 3),
    ) {
        let blocks = BlockFP8E4M3::quantize(&values).expect("quantize should succeed");
        let mut output = vec![0.0f32; values.len()];
        BlockFP8E4M3::dequant(&blocks, &mut output).expect("dequant should succeed");
        for (i, &y) in output.iter().enumerate() {
            prop_assert!(
                y.is_finite(),
                "E4M3 finite input produced non-finite output at index {i}: \
                 input={}, output={y}",
                values[i]
            );
        }
    }
}

// ===========================================================================
// Q4_0
// ===========================================================================

#[test]
fn q4_0_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_Q4_0, 42);
    let blocks = BlockQ4_0::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_Q4_0];
    BlockQ4_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    // 4-bit symmetric format, scale = max_abs / 7. On [-1,1] data the
    // quantization step is at most ~1/7 ≈ 0.143, so half-step + rounding
    // slack of 0.2 is a safe, tight bound.
    assert!(
        max_err < 0.2,
        "Q4_0 random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q4_0_multiple_blocks_random_roundtrip() {
    let num_blocks = 4;
    let input = generate_test_data(QK_Q4_0 * num_blocks, 999);
    let blocks = BlockQ4_0::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_Q4_0 * num_blocks];
    BlockQ4_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.2,
        "Q4_0 multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q4_0_sign_preservation_canary() {
    let mut input = vec![0.0f32; QK_Q4_0];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockQ4_0::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_Q4_0];
    BlockQ4_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "Q4_0 sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn q4_0_invalid_input_length_errors() {
    assert!(BlockQ4_0::quantize(&[0.0f32; 17]).is_err());
}

#[test]
fn q4_0_block_boundary_independent_scale() {
    let mut input = vec![0.5f32; QK_Q4_0 * 2];
    for v in input[QK_Q4_0..].iter_mut() {
        *v = 50.0;
    }
    let blocks = BlockQ4_0::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_Q4_0 * 2];
    BlockQ4_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_Q4_0].iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 0.2,
            "Q4_0 block 0 leaked block-1 scale at index {i}, got {v}"
        );
    }
}

// ===========================================================================
// Q8_0
// ===========================================================================

#[test]
fn q8_0_random_roundtrip_error_bounded() {
    let input = generate_test_data(QK_Q8_0, 42);
    let blocks = BlockQ8_0::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_Q8_0];
    BlockQ8_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    // Matches the tolerance already established for Q8_0's own deterministic
    // roundtrip test (quant_std.rs: max_err < 0.05).
    assert!(
        max_err < 0.05,
        "Q8_0 random roundtrip max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q8_0_multiple_blocks_random_roundtrip() {
    let num_blocks = 4;
    let input = generate_test_data(QK_Q8_0 * num_blocks, 999);
    let blocks = BlockQ8_0::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), num_blocks);

    let mut output = vec![0.0f32; QK_Q8_0 * num_blocks];
    BlockQ8_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    let max_err = max_abs_err(&input, &output);
    assert!(
        max_err < 0.05,
        "Q8_0 multi-block max error {max_err} exceeds tolerance"
    );
}

#[test]
fn q8_0_sign_preservation_canary() {
    let mut input = vec![0.0f32; QK_Q8_0];
    for (i, v) in input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let blocks = BlockQ8_0::quantize(&input).expect("quantize should succeed");
    let mut output = vec![0.0f32; QK_Q8_0];
    BlockQ8_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            x.signum(),
            y.signum(),
            "Q8_0 sign flipped at index {i}: input {x}, output {y}"
        );
    }
}

#[test]
fn q8_0_invalid_input_length_errors() {
    assert!(BlockQ8_0::quantize(&[0.0f32; 17]).is_err());
}

#[test]
fn q8_0_block_boundary_independent_scale() {
    let mut input = vec![0.5f32; QK_Q8_0 * 2];
    for v in input[QK_Q8_0..].iter_mut() {
        *v = 50.0;
    }
    let blocks = BlockQ8_0::quantize(&input).expect("quantize should succeed");
    assert_eq!(blocks.len(), 2);
    let mut output = vec![0.0f32; QK_Q8_0 * 2];
    BlockQ8_0::dequant(&blocks, &mut output).expect("dequant should succeed");

    for (i, &v) in output[..QK_Q8_0].iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 0.05,
            "Q8_0 block 0 leaked block-1 scale at index {i}, got {v}"
        );
    }
}
