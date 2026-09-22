//! Tests for individual model layers: RMSNorm, RoPE, SwiGLU, Linear1Bit.

use half::f16;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::layers::linear::Linear1Bit;
use oxibonsai_model::layers::rms_norm::RmsNorm;
use oxibonsai_model::layers::rope::RopeTable;
use oxibonsai_model::layers::swiglu::{silu, try_swiglu};

fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
    BlockQ1_0G128 {
        d: f16::from_f32(scale),
        qs: bits,
    }
}

fn ref_kernel() -> std::sync::Arc<KernelDispatcher> {
    std::sync::Arc::new(KernelDispatcher::with_tier(KernelTier::Reference))
}

// ══════════════════════════════════════════════════════════════
// RMSNorm tests
// ══════════════════════════════════════════════════════════════

#[test]
fn rms_norm_unit_weights_output_has_unit_rms() {
    let weight = vec![1.0; 8];
    let norm = RmsNorm::new(weight, 1e-6);

    let input = vec![3.0, -1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let mut output = vec![0.0; 8];
    norm.forward(&input, &mut output)
        .expect("forward should succeed");

    // Compute RMS of output: should be ~1.0 since weights are 1.0
    let sum_sq: f32 = output.iter().map(|x| x * x).sum();
    let rms = (sum_sq / output.len() as f32).sqrt();
    assert!(
        (rms - 1.0).abs() < 1e-4,
        "output RMS should be ~1.0, got {rms}"
    );
}

#[test]
fn rms_norm_scaling_with_non_unit_weights() {
    let weight = vec![2.0; 4];
    let norm = RmsNorm::new(weight, 1e-6);

    let input = vec![1.0, 1.0, 1.0, 1.0];
    let mut output = vec![0.0; 4];
    norm.forward(&input, &mut output).expect("forward");

    // RMS(input) = 1.0, so output[i] = 2.0 * 1.0 / 1.0 = 2.0
    for &v in &output {
        assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {v}");
    }
}

#[test]
fn rms_norm_zero_input_produces_zero_output() {
    let weight = vec![5.0; 4];
    let norm = RmsNorm::new(weight, 1e-6);

    let input = vec![0.0; 4];
    let mut output = vec![999.0; 4];
    norm.forward(&input, &mut output).expect("forward");

    // RMS = sqrt(eps), output = weight * 0 / rms = 0
    for &v in &output {
        assert!(
            v.abs() < 1e-3,
            "zero input should give near-zero output, got {v}"
        );
    }
}

#[test]
fn rms_norm_large_values_no_overflow() {
    let weight = vec![1.0; 4];
    let norm = RmsNorm::new(weight, 1e-6);

    let input = vec![1e10, 1e10, 1e10, 1e10];
    let mut output = vec![0.0; 4];
    norm.forward(&input, &mut output).expect("forward");

    for &v in &output {
        assert!(v.is_finite(), "should not overflow, got {v}");
        // Should be 1.0 since all values are the same
        assert!((v - 1.0).abs() < 1e-3);
    }
}

#[test]
fn rms_norm_mixed_positive_negative() {
    let weight = vec![1.0; 4];
    let norm = RmsNorm::new(weight, 1e-6);

    let input = vec![1.0, -1.0, 1.0, -1.0];
    let mut output = vec![0.0; 4];
    norm.forward(&input, &mut output).expect("forward");

    // RMS = 1.0, so output = input
    for i in 0..4 {
        assert!(
            (output[i] - input[i]).abs() < 1e-4,
            "at {i}: expected {}, got {}",
            input[i],
            output[i]
        );
    }
}

#[test]
fn rms_norm_hidden_size() {
    let norm = RmsNorm::new(vec![1.0; 256], 1e-6);
    assert_eq!(norm.hidden_size(), 256);
}

// ══════════════════════════════════════════════════════════════
// RoPE tests
// ══════════════════════════════════════════════════════════════

#[test]
fn rope_position_zero_is_identity() {
    let table = RopeTable::new(8, 32, 10000.0);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0; 8];

    table.apply(&input, &mut output, 0).expect("apply");

    // At position 0: cos(0)=1, sin(0)=0 -> identity
    for i in 0..8 {
        assert!(
            (output[i] - input[i]).abs() < 1e-5,
            "pos=0 should be identity at dim {i}: expected {}, got {}",
            input[i],
            output[i]
        );
    }
}

#[test]
fn rope_preserves_vector_norm() {
    let table = RopeTable::new(8, 64, 10000.0);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    for pos in [0, 1, 5, 10, 31, 63] {
        let mut output = vec![0.0; 8];
        table.apply(&input, &mut output, pos).expect("apply");

        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (input_norm - output_norm).abs() < 1e-3,
            "pos={pos}: norms differ: input={input_norm}, output={output_norm}"
        );
    }
}

#[test]
fn rope_different_positions_produce_different_outputs() {
    let table = RopeTable::new(8, 64, 10000.0);
    let input = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let mut out_pos0 = vec![0.0; 8];
    let mut out_pos1 = vec![0.0; 8];
    let mut out_pos10 = vec![0.0; 8];
    table.apply(&input, &mut out_pos0, 0).expect("pos 0");
    table.apply(&input, &mut out_pos1, 1).expect("pos 1");
    table.apply(&input, &mut out_pos10, 10).expect("pos 10");

    // Position 1 should differ from position 0
    let diff_01: f32 = out_pos0
        .iter()
        .zip(out_pos1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff_01 > 1e-4, "pos 0 and 1 should differ");

    // Position 10 should differ from position 1
    let diff_1_10: f32 = out_pos1
        .iter()
        .zip(out_pos10.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff_1_10 > 1e-4, "pos 1 and 10 should differ");
}

#[test]
fn rope_frequency_decreases_with_dimension_index() {
    let head_dim = 8;
    let table = RopeTable::new(head_dim, 64, 10000.0);

    // Apply at position 1 with a unit vector in each dimension pair
    // Higher dimensions should rotate less (lower frequency)
    let mut angles = Vec::new();
    for pair_idx in 0..(head_dim / 2) {
        let mut input = vec![0.0f32; head_dim];
        input[pair_idx] = 1.0; // cos component
                               // sin component is at pair_idx + head_dim/2

        let mut output = vec![0.0f32; head_dim];
        table.apply(&input, &mut output, 1).expect("apply");

        // The angle can be recovered from the rotation:
        // output[pair_idx] = cos(angle), output[pair_idx + half] = sin(angle)
        let cos_val = output[pair_idx];
        let sin_val = output[pair_idx + head_dim / 2];
        let angle = sin_val.atan2(cos_val);
        angles.push(angle.abs());
    }

    // Verify: angles should be decreasing (or at least non-increasing)
    for i in 1..angles.len() {
        assert!(
            angles[i] <= angles[i - 1] + 1e-6,
            "frequency should decrease: angle[{}]={} > angle[{}]={}",
            i,
            angles[i],
            i - 1,
            angles[i - 1]
        );
    }
}

#[test]
fn rope_max_seq_len() {
    let table = RopeTable::new(4, 100, 10000.0);
    assert_eq!(table.max_seq_len(), 100);
}

// ══════════════════════════════════════════════════════════════
// SwiGLU tests
// ══════════════════════════════════════════════════════════════

#[test]
fn silu_at_zero_is_zero() {
    assert!(silu(0.0).abs() < 1e-6);
}

#[test]
fn silu_positive_values() {
    // silu(x) = x * sigmoid(x), for large x -> x
    let result = silu(10.0);
    assert!(
        (result - 10.0).abs() < 0.01,
        "silu(10) should be ~10, got {result}"
    );
}

#[test]
fn silu_negative_values() {
    // silu(x) for very negative x -> 0
    let result = silu(-10.0);
    assert!(result.abs() < 0.01, "silu(-10) should be ~0, got {result}");
}

#[test]
fn silu_is_monotonically_increasing_for_positive_range() {
    // silu(x) = x * sigmoid(x) is monotonically increasing for x >= 0
    let mut prev = silu(0.0);
    for i in 1..100 {
        let x = i as f32 * 0.1;
        let current = silu(x);
        assert!(
            current >= prev - 1e-6,
            "silu should be monotonically increasing for x >= 0: silu({x})={current} < prev={prev}"
        );
        prev = current;
    }
}

#[test]
fn swiglu_gate_times_up_pattern() {
    let gate = vec![1.0, 2.0, 0.0];
    let up = vec![3.0, 4.0, 5.0];
    let mut output = vec![0.0; 3];
    try_swiglu(&gate, &up, &mut output).expect("matching-length buffers must succeed");

    assert!((output[0] - silu(1.0) * 3.0).abs() < 1e-5);
    assert!((output[1] - silu(2.0) * 4.0).abs() < 1e-5);
    assert!((output[2] - silu(0.0) * 5.0).abs() < 1e-5); // silu(0)=0, so 0
}

#[test]
fn swiglu_zero_gate_zeroes_output() {
    let gate = vec![0.0; 8];
    let up = vec![100.0; 8];
    let mut output = vec![999.0; 8];
    try_swiglu(&gate, &up, &mut output).expect("matching-length buffers must succeed");

    for &v in &output {
        assert!(v.abs() < 1e-5, "zero gate should zero output, got {v}");
    }
}

#[test]
fn swiglu_zero_up_zeroes_output() {
    let gate = vec![5.0; 4];
    let up = vec![0.0; 4];
    let mut output = vec![999.0; 4];
    try_swiglu(&gate, &up, &mut output).expect("matching-length buffers must succeed");

    for &v in &output {
        assert!(v.abs() < 1e-5, "zero up should zero output, got {v}");
    }
}

// ══════════════════════════════════════════════════════════════
// Linear1Bit tests
// ══════════════════════════════════════════════════════════════

#[test]
fn linear_1bit_all_positive_weights() {
    // 1 output feature, 128 input features, all bits set = +scale
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let kernel = ref_kernel();
    let layer = Linear1Bit::new(&blocks, 1, 128, kernel.clone()).expect("layer");

    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    layer.forward_vec(&input, &mut output).expect("forward");
    // dot(+1*128, [1;128]) = 128
    assert!((output[0] - 128.0).abs() < 1.0);
}

#[test]
fn linear_1bit_all_negative_weights() {
    // All bits clear = -scale
    let blocks = vec![make_block(1.0, [0x00; 16])];
    let kernel = ref_kernel();
    let layer = Linear1Bit::new(&blocks, 1, 128, kernel.clone()).expect("layer");

    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    layer.forward_vec(&input, &mut output).expect("forward");
    assert!((output[0] + 128.0).abs() < 1.0);
}

#[test]
fn linear_1bit_output_dimension_matches_rows() {
    let n_out = 4;
    let blocks: Vec<BlockQ1_0G128> = (0..n_out).map(|_| make_block(0.5, [0xFF; 16])).collect();
    let kernel = ref_kernel();
    let layer = Linear1Bit::new(&blocks, n_out, 128, kernel.clone()).expect("layer");
    assert_eq!(layer.out_features(), n_out);
    assert_eq!(layer.in_features(), 128);

    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; n_out];
    layer.forward_vec(&input, &mut output).expect("forward");

    for &v in &output {
        assert!((v - 64.0).abs() < 1.0, "expected ~64, got {v}");
    }
}

#[test]
fn linear_1bit_forward_mat_batch() {
    let blocks = vec![make_block(1.0, [0xFF; 16])];
    let kernel = ref_kernel();
    let layer = Linear1Bit::new(&blocks, 1, 128, kernel.clone()).expect("layer");

    // 2 batch elements
    let mut input = vec![0.0f32; 256];
    for v in &mut input[..128] {
        *v = 1.0;
    }
    for v in &mut input[128..] {
        *v = 2.0;
    }

    let mut output = vec![0.0f32; 2];
    layer
        .forward_mat(&input, &mut output, 2)
        .expect("forward_mat");
    assert!((output[0] - 128.0).abs() < 1.0);
    assert!((output[1] - 256.0).abs() < 1.0);
}

#[test]
fn linear_1bit_mixed_weights() {
    // Alternating bits: half positive, half negative
    let blocks = vec![make_block(1.0, [0xAA; 16])];
    let kernel = ref_kernel();
    let layer = Linear1Bit::new(&blocks, 1, 128, kernel.clone()).expect("layer");

    // Uniform input: the positive and negative weights should cancel
    let input = vec![1.0f32; 128];
    let mut output = vec![0.0f32; 1];
    layer.forward_vec(&input, &mut output).expect("forward");
    // 0xAA: 64 ones, 64 zeros -> dot = scale * (64 - 64) = 0
    assert!(
        output[0].abs() < 0.01,
        "alternating should cancel, got {}",
        output[0]
    );
}
