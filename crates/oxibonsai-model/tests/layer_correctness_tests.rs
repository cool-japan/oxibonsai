//! Layer-level correctness tests for oxibonsai-model.
//!
//! Verifies that each layer (RMSNorm, SwiGLU, RoPE, Attention, TransformerBlock)
//! produces mathematically correct results against hand-computed references.

use half::f16;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_model::block::TransformerBlock;
use oxibonsai_model::kv_cache::KvCache;
use oxibonsai_model::layers::attention::{
    attention_head, attention_head_with_mask, dot, multi_head_attention, softmax, CausalMask,
};
use oxibonsai_model::layers::linear::Linear1Bit;
use oxibonsai_model::layers::rms_norm::RmsNorm;
use oxibonsai_model::layers::rms_norm::RmsNorm as RmsNormLayer;
use oxibonsai_model::layers::rope::RopeTable;
use oxibonsai_model::layers::swiglu::{silu, try_swiglu};

// ──────────────────────────────────────────────────────────────────
// Helper utilities
// ──────────────────────────────────────────────────────────────────

fn assert_no_nan(data: &[f32], label: &str) {
    for (i, &v) in data.iter().enumerate() {
        assert!(!v.is_nan(), "{label}[{i}] is NaN");
        assert!(!v.is_infinite(), "{label}[{i}] is Inf");
    }
}

fn assert_approx_eq(a: f32, b: f32, tol: f32, label: &str) {
    let diff = (a - b).abs();
    assert!(diff < tol, "{label}: {a} vs {b}, diff={diff}, tol={tol}");
}

/// Simple LCG for deterministic test data.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

fn random_tensor(state: &mut u64, len: usize) -> Vec<f32> {
    (0..len).map(|_| lcg(state)).collect()
}

fn make_blocks(n: usize, scale: f32, pattern: u8) -> Vec<BlockQ1_0G128> {
    (0..n)
        .map(|_| BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: [pattern; 16],
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────
// RMSNorm correctness
// ──────────────────────────────────────────────────────────────────

#[test]
fn rms_norm_hand_computed_reference() {
    // rms_norm(x) = x * rsqrt(mean(x^2) + eps) * gamma
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32, 1.0, 1.0, 1.0];
    let eps = 1e-6f32;

    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + eps).sqrt();

    let norm = RmsNorm::new(gamma.to_vec(), eps);
    let mut output = vec![0.0f32; 4];
    norm.forward(&input, &mut output)
        .expect("forward should succeed");

    for i in 0..4 {
        let expected = gamma[i] * input[i] / rms;
        assert_approx_eq(output[i], expected, 1e-5, &format!("rms_norm[{i}]"));
    }
}

#[test]
fn rms_norm_all_ones_equals_gamma() {
    let dim = 8;
    let gamma: Vec<f32> = (1..=dim).map(|i| i as f32 * 0.5).collect();
    let norm = RmsNorm::new(gamma.clone(), 1e-6);

    let input = vec![1.0f32; dim];
    let mut output = vec![0.0f32; dim];
    norm.forward(&input, &mut output)
        .expect("forward should succeed");

    // rms of all-ones = sqrt(1.0 + eps) ~ 1.0
    // output[i] ~ gamma[i] * 1.0 / 1.0 = gamma[i]
    for (i, &val) in output.iter().enumerate() {
        assert_approx_eq(val, gamma[i], 1e-4, &format!("all_ones[{i}]"));
    }
}

#[test]
fn rms_norm_all_zeros() {
    let dim = 4;
    let gamma = vec![2.0f32; dim];
    let norm = RmsNorm::new(gamma, 1e-6);

    let input = vec![0.0f32; dim];
    let mut output = vec![0.0f32; dim];
    norm.forward(&input, &mut output)
        .expect("forward should succeed");

    // rms = sqrt(0 + eps) = sqrt(eps), output = gamma * 0 / rms = 0
    for (i, &val) in output.iter().enumerate() {
        assert_approx_eq(val, 0.0, 1e-6, &format!("all_zeros[{i}]"));
    }
}

#[test]
fn rms_norm_normalization_property() {
    // After RMSNorm with unit gamma, the output L2 norm should be approximately sqrt(dim).
    let dim = 64;
    let gamma = vec![1.0f32; dim];
    let norm = RmsNorm::new(gamma, 1e-6);

    let mut state = 42u64;
    let input = random_tensor(&mut state, dim);
    let mut output = vec![0.0f32; dim];
    norm.forward(&input, &mut output)
        .expect("forward should succeed");

    let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    let expected_norm = (dim as f32).sqrt();
    assert_approx_eq(output_norm, expected_norm, 0.1, "norm_property");
}

#[test]
fn rms_norm_scale_invariance() {
    // rms_norm(c*x) ~ sign(c) * rms_norm(x) for scalar c != 0
    let dim = 16;
    let gamma = vec![1.0f32; dim];
    let eps = 1e-6;
    let norm = RmsNorm::new(gamma, eps);

    let mut state = 99u64;
    let input = random_tensor(&mut state, dim);

    let mut output_x = vec![0.0f32; dim];
    norm.forward(&input, &mut output_x)
        .expect("forward should succeed");

    let c = 5.0f32;
    let scaled_input: Vec<f32> = input.iter().map(|&x| c * x).collect();
    let mut output_cx = vec![0.0f32; dim];
    norm.forward(&scaled_input, &mut output_cx)
        .expect("forward should succeed");

    // rms_norm(c*x) = c*x / rms(c*x) = c*x / (|c| * rms(x)) = sign(c) * x / rms(x)
    let sign_c = c.signum();
    for i in 0..dim {
        let expected = sign_c * output_x[i];
        assert_approx_eq(output_cx[i], expected, 1e-4, &format!("scale_inv[{i}]"));
    }
}

// ──────────────────────────────────────────────────────────────────
// SwiGLU correctness
// ──────────────────────────────────────────────────────────────────

#[test]
fn swiglu_manual_computation() {
    // swiglu(gate, up) = silu(gate) * up
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let gate = [1.0f32, -1.0, 0.5, 2.0];
    let up = [2.0f32, 3.0, 4.0, 0.5];
    let mut output = [0.0f32; 4];

    try_swiglu(&gate, &up, &mut output).expect("matching-length buffers must succeed");

    for i in 0..4 {
        let expected = silu(gate[i]) * up[i];
        assert_approx_eq(output[i], expected, 1e-6, &format!("swiglu[{i}]"));
    }
}

#[test]
fn silu_known_values() {
    // silu(0) = 0
    assert_approx_eq(silu(0.0), 0.0, 1e-7, "silu(0)");
    // silu(1) = 1/(1+exp(-1)) ~ 0.7311
    assert_approx_eq(silu(1.0), 0.7311, 1e-3, "silu(1)");
    // silu(-1) = -1/(1+exp(1)) ~ -0.2689
    assert_approx_eq(silu(-1.0), -0.2689, 1e-3, "silu(-1)");
    // silu(x) -> x for large positive x (sigmoid -> 1)
    assert_approx_eq(silu(10.0), 10.0, 1e-3, "silu(10)");
    // silu(x) -> 0 for large negative x
    assert_approx_eq(silu(-10.0), 0.0, 1e-3, "silu(-10)");
}

#[test]
fn swiglu_zero_input() {
    let gate = [0.0f32; 8];
    let up = [0.0f32; 8];
    let mut output = [0.0f32; 8];

    try_swiglu(&gate, &up, &mut output).expect("matching-length buffers must succeed");

    for (i, &val) in output.iter().enumerate() {
        assert_approx_eq(val, 0.0, 1e-7, &format!("swiglu_zero[{i}]"));
    }
}

#[test]
fn swiglu_saturation_behavior() {
    // Large positive gate: silu(gate) ~ gate, so output ~ gate * up
    let gate = [50.0f32];
    let up = [2.0f32];
    let mut output = [0.0f32; 1];
    try_swiglu(&gate, &up, &mut output).expect("matching-length buffers must succeed");
    assert_approx_eq(output[0], 100.0, 0.1, "swiglu_large_pos");

    // Large negative gate: silu(gate) ~ 0, so output ~ 0
    let gate_neg = [-50.0f32];
    try_swiglu(&gate_neg, &up, &mut output).expect("matching-length buffers must succeed");
    assert_approx_eq(output[0], 0.0, 1e-5, "swiglu_large_neg");
}

#[test]
fn silu_is_odd_ish_function() {
    // silu is not odd, but silu(-x) = -x * sigmoid(-x) = -x * (1 - sigmoid(x))
    // Verify: silu(x) + silu(-x) = x * sigmoid(x) - x * (1 - sigmoid(x)) = x * (2*sigmoid(x) - 1)
    let mut state = 77u64;
    for _ in 0..20 {
        let x = lcg(&mut state) * 5.0;
        let sig_x = 1.0 / (1.0 + (-x).exp());
        let expected_sum = x * (2.0 * sig_x - 1.0);
        let actual_sum = silu(x) + silu(-x);
        assert_approx_eq(actual_sum, expected_sum, 1e-5, "silu_symmetry");
    }
}

// ──────────────────────────────────────────────────────────────────
// RoPE correctness
// ──────────────────────────────────────────────────────────────────

#[test]
fn rope_position_zero_is_identity() {
    let head_dim = 8;
    let table = RopeTable::new(head_dim, 32, 10000.0);

    let mut state = 42u64;
    let input = random_tensor(&mut state, head_dim);
    let mut output = vec![0.0f32; head_dim];

    table
        .apply(&input, &mut output, 0)
        .expect("apply should succeed");

    for i in 0..head_dim {
        assert_approx_eq(output[i], input[i], 1e-5, &format!("rope_pos0[{i}]"));
    }
}

#[test]
fn rope_preserves_magnitude() {
    let head_dim = 8;
    let table = RopeTable::new(head_dim, 128, 10000.0);

    let mut state = 123u64;
    for pos in [0, 1, 5, 50, 100] {
        let input = random_tensor(&mut state, head_dim);
        let mut output = vec![0.0f32; head_dim];

        table
            .apply(&input, &mut output, pos)
            .expect("apply should succeed");

        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert_approx_eq(output_norm, input_norm, 1e-4, &format!("rope_mag_pos{pos}"));
    }
}

#[test]
fn rope_relative_position_inner_product() {
    // <rope(x, m), rope(y, n)> should depend only on m - n
    let head_dim = 4;
    let table = RopeTable::new(head_dim, 64, 10000.0);

    let mut state = 55u64;
    let x = random_tensor(&mut state, head_dim);
    let y = random_tensor(&mut state, head_dim);

    // Compute <rope(x, 5), rope(y, 3)>
    let mut rx5 = vec![0.0f32; head_dim];
    let mut ry3 = vec![0.0f32; head_dim];
    table.apply(&x, &mut rx5, 5).expect("apply should succeed");
    table.apply(&y, &mut ry3, 3).expect("apply should succeed");
    let dot1: f32 = rx5.iter().zip(ry3.iter()).map(|(a, b)| a * b).sum();

    // Compute <rope(x, 10), rope(y, 8)> — same relative position (diff = 2)
    let mut rx10 = vec![0.0f32; head_dim];
    let mut ry8 = vec![0.0f32; head_dim];
    table
        .apply(&x, &mut rx10, 10)
        .expect("apply should succeed");
    table.apply(&y, &mut ry8, 8).expect("apply should succeed");
    let dot2: f32 = rx10.iter().zip(ry8.iter()).map(|(a, b)| a * b).sum();

    assert_approx_eq(dot1, dot2, 1e-3, "rope_relative_pos");
}

#[test]
fn rope_head_dim_4_hand_computation() {
    // head_dim=4, half_dim=2
    // freq[0] = 1 / 10000^(0/4) = 1.0
    // freq[1] = 1 / 10000^(2/4) = 1/100 = 0.01
    // At pos=1: angle[0] = 1*1.0 = 1.0, angle[1] = 1*0.01 = 0.01
    let table = RopeTable::new(4, 16, 10000.0);
    let input = [1.0f32, 0.0, 0.0, 1.0]; // x0=1,x1=0 | x2=0,x3=1
    let mut output = [0.0f32; 4];

    table
        .apply(&input, &mut output, 1)
        .expect("apply should succeed");

    let angle0 = 1.0f32;
    let angle1 = 0.01f32;

    // output[0] = x0*cos(a0) - x2*sin(a0) = 1*cos(1) - 0*sin(1) = cos(1)
    // output[1] = x1*cos(a1) - x3*sin(a1) = 0*cos(0.01) - 1*sin(0.01) = -sin(0.01)
    // output[2] = x0*sin(a0) + x2*cos(a0) = 1*sin(1) + 0*cos(1) = sin(1)
    // output[3] = x1*sin(a1) + x3*cos(a1) = 0*sin(0.01) + 1*cos(0.01) = cos(0.01)
    assert_approx_eq(output[0], angle0.cos(), 1e-5, "rope_hand[0]");
    assert_approx_eq(output[1], -angle1.sin(), 1e-5, "rope_hand[1]");
    assert_approx_eq(output[2], angle0.sin(), 1e-5, "rope_hand[2]");
    assert_approx_eq(output[3], angle1.cos(), 1e-5, "rope_hand[3]");
}

// ──────────────────────────────────────────────────────────────────
// Attention correctness
// ──────────────────────────────────────────────────────────────────

#[test]
fn attention_single_head_single_token_returns_value() {
    let head_dim = 4;
    let query = [1.0f32, 0.0, 0.0, 0.0];
    let keys = [1.0f32, 0.0, 0.0, 0.0];
    let values = [0.5f32, 1.5, 2.5, 3.5];
    let mut output = [0.0f32; 4];

    attention_head(&query, &keys, &values, &mut output, 1, head_dim)
        .expect("attention should succeed");

    for i in 0..4 {
        assert_approx_eq(output[i], values[i], 1e-4, &format!("attn_single[{i}]"));
    }
}

#[test]
fn attention_causal_masking_blocks_future() {
    let head_dim = 4;
    let mask = CausalMask::new(16);

    // 4 tokens, query at position 1
    let query = [1.0f32, 0.0, 0.0, 0.0];
    let keys = [
        1.0, 0.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, 0.0, // token 1
        1.0, 1.0, 0.0, 0.0, // token 2 (future)
        0.0, 0.0, 1.0, 0.0, // token 3 (future)
    ];
    let values = [
        1.0, 0.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, 0.0, // token 1
        0.0, 0.0, 99.0, 0.0, // token 2 (should be masked)
        0.0, 0.0, 0.0, 99.0, // token 3 (should be masked)
    ];
    let mut output = [0.0f32; 4];

    attention_head_with_mask(&query, &keys, &values, &mut output, 4, head_dim, 1, &mask)
        .expect("attention should succeed");

    // Future tokens have value 99 in dims 2,3 — those should be zero
    assert_approx_eq(output[2], 0.0, 1e-3, "causal_future_dim2");
    assert_approx_eq(output[3], 0.0, 1e-3, "causal_future_dim3");
}

#[test]
fn attention_weights_sum_to_one() {
    let head_dim = 4;
    let seq_len = 5;

    let mut state = 42u64;
    let query = random_tensor(&mut state, head_dim);
    let keys = random_tensor(&mut state, seq_len * head_dim);
    let values = random_tensor(&mut state, seq_len * head_dim);
    let mut output = vec![0.0f32; head_dim];

    // We can verify softmax sums to 1 by checking with known single-value case
    // But let's directly test: compute scores + softmax
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let key = &keys[t * head_dim..(t + 1) * head_dim];
        scores[t] = dot(&query, key) * scale;
    }
    softmax(&mut scores);

    let sum: f32 = scores.iter().sum();
    assert_approx_eq(sum, 1.0, 1e-5, "attn_weights_sum");

    // Also verify all weights are non-negative
    for (i, &s) in scores.iter().enumerate() {
        assert!(s >= 0.0, "attn_weight[{i}] = {s} is negative");
    }

    // Run the actual attention too
    attention_head(&query, &keys, &values, &mut output, seq_len, head_dim)
        .expect("attention should succeed");
    assert_no_nan(&output, "attn_output");
}

#[test]
fn attention_gqa_kv_head_sharing() {
    let head_dim = 4;
    let num_heads = 4;
    let num_kv_heads = 2; // ratio 2:1
    let seq_len = 1;

    // Two KV heads with distinct values
    let kv0_keys = [1.0f32, 0.0, 0.0, 0.0];
    let kv0_values = [1.0f32, 0.0, 0.0, 0.0];
    let kv1_keys = [0.0f32, 1.0, 0.0, 0.0];
    let kv1_values = [0.0f32, 1.0, 0.0, 0.0];

    let keys_refs: Vec<&[f32]> = vec![&kv0_keys, &kv1_keys];
    let values_refs: Vec<&[f32]> = vec![&kv0_values, &kv1_values];

    let query_all = [
        1.0, 0.0, 0.0, 0.0, // Q head 0 -> KV head 0
        0.0, 0.0, 1.0, 0.0, // Q head 1 -> KV head 0
        0.0, 1.0, 0.0, 0.0, // Q head 2 -> KV head 1
        0.0, 0.0, 0.0, 1.0, // Q head 3 -> KV head 1
    ];
    let mut output = vec![0.0f32; num_heads * head_dim];

    multi_head_attention(
        &query_all,
        &keys_refs,
        &values_refs,
        &mut output,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    )
    .expect("multi_head_attention should succeed");

    // Heads 0,1 use KV head 0 -> value = [1,0,0,0]
    // Heads 2,3 use KV head 1 -> value = [0,1,0,0]
    for i in 0..head_dim {
        assert_approx_eq(output[i], kv0_values[i], 1e-4, &format!("gqa_h0[{i}]"));
    }
    for i in 0..head_dim {
        assert_approx_eq(
            output[head_dim + i],
            kv0_values[i],
            1e-4,
            &format!("gqa_h1[{i}]"),
        );
    }
    for i in 0..head_dim {
        assert_approx_eq(
            output[2 * head_dim + i],
            kv1_values[i],
            1e-4,
            &format!("gqa_h2[{i}]"),
        );
    }
    for i in 0..head_dim {
        assert_approx_eq(
            output[3 * head_dim + i],
            kv1_values[i],
            1e-4,
            &format!("gqa_h3[{i}]"),
        );
    }
}

#[test]
fn attention_two_tokens_uniform_query() {
    // With a uniform query (all same values), attention should produce
    // a weighted average leaning toward higher-scoring keys.
    let head_dim = 4;
    let query = [0.5f32; 4];
    let keys = [
        1.0, 1.0, 1.0, 1.0, // token 0: high dot with query
        0.0, 0.0, 0.0, 0.0, // token 1: zero dot with query
    ];
    let values = [
        1.0, 0.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, 0.0, // token 1
    ];
    let mut output = [0.0f32; 4];

    attention_head(&query, &keys, &values, &mut output, 2, head_dim)
        .expect("attention should succeed");

    // Token 0 has higher score, so output should lean toward [1,0,0,0]
    assert!(output[0] > output[1], "should favor token 0's value");
}

// ──────────────────────────────────────────────────────────────────
// TransformerBlock correctness
// ──────────────────────────────────────────────────────────────────

#[test]
fn transformer_block_output_shape_matches_input() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_row = h / 128;

    let q_blocks = make_blocks(nq * hd * blocks_per_row, 0.01, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_row, 0.01, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

    let kernel_arc = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let block = TransformerBlock::new(
        0,
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&q_blocks, nq * hd, h, kernel_arc.clone())
            .expect("q")
            .into(),
        Linear1Bit::new(&k_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("k")
            .into(),
        Linear1Bit::new(&v_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("v")
            .into(),
        Linear1Bit::new(&o_blocks, h, nq * hd, kernel_arc.clone())
            .expect("o")
            .into(),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&gate_blocks, inter, h, kernel_arc.clone())
            .expect("gate")
            .into(),
        Linear1Bit::new(&up_blocks, inter, h, kernel_arc.clone())
            .expect("up")
            .into(),
        Linear1Bit::new(&down_blocks, h, inter, kernel_arc.clone())
            .expect("down")
            .into(),
        nq,
        nkv,
        hd,
        h,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = &*kernel_arc;
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let original_len = hidden.len();

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel)
        .expect("forward should succeed");

    assert_eq!(
        hidden.len(),
        original_len,
        "output shape must match input shape"
    );
}

#[test]
fn transformer_block_output_differs_from_input() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_row = h / 128;

    let q_blocks = make_blocks(nq * hd * blocks_per_row, 0.01, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_row, 0.01, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

    let kernel_arc = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let block = TransformerBlock::new(
        0,
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&q_blocks, nq * hd, h, kernel_arc.clone())
            .expect("q")
            .into(),
        Linear1Bit::new(&k_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("k")
            .into(),
        Linear1Bit::new(&v_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("v")
            .into(),
        Linear1Bit::new(&o_blocks, h, nq * hd, kernel_arc.clone())
            .expect("o")
            .into(),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&gate_blocks, inter, h, kernel_arc.clone())
            .expect("gate")
            .into(),
        Linear1Bit::new(&up_blocks, inter, h, kernel_arc.clone())
            .expect("up")
            .into(),
        Linear1Bit::new(&down_blocks, h, inter, kernel_arc.clone())
            .expect("down")
            .into(),
        nq,
        nkv,
        hd,
        h,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = &*kernel_arc;
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let original = hidden.clone();

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel)
        .expect("forward should succeed");

    let max_diff = hidden
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "forward should modify hidden state, max_diff={max_diff}"
    );
}

#[test]
fn transformer_block_residual_connection() {
    // The block adds sublayer outputs to input (residual). Verify output != 0
    // even when sublayer output is small, because of the residual.
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_row = h / 128;

    // Very small scale weights -> sublayer output is small
    let q_blocks = make_blocks(nq * hd * blocks_per_row, 0.001, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_row, 0.001, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_row, 0.001, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_row, 0.001, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_row, 0.001, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_row, 0.001, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.001, 0xFF);

    let kernel_arc = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let block = TransformerBlock::new(
        0,
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&q_blocks, nq * hd, h, kernel_arc.clone())
            .expect("q")
            .into(),
        Linear1Bit::new(&k_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("k")
            .into(),
        Linear1Bit::new(&v_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("v")
            .into(),
        Linear1Bit::new(&o_blocks, h, nq * hd, kernel_arc.clone())
            .expect("o")
            .into(),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&gate_blocks, inter, h, kernel_arc.clone())
            .expect("gate")
            .into(),
        Linear1Bit::new(&up_blocks, inter, h, kernel_arc.clone())
            .expect("up")
            .into(),
        Linear1Bit::new(&down_blocks, h, inter, kernel_arc.clone())
            .expect("down")
            .into(),
        nq,
        nkv,
        hd,
        h,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = &*kernel_arc;
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let original = hidden.clone();

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel)
        .expect("forward should succeed");

    // Because of residual, output should be close to input (small perturbation)
    let mut close_count = 0;
    for i in 0..h {
        let diff = (hidden[i] - original[i]).abs();
        // With scale 0.001, the sublayer output is small, so residual dominates
        if diff < 1.0 {
            close_count += 1;
        }
    }
    assert!(
        close_count > h / 2,
        "residual connection should keep output close to input when sublayer weights are small"
    );
}

#[test]
fn transformer_block_no_nan_in_output() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_row = h / 128;

    let q_blocks = make_blocks(nq * hd * blocks_per_row, 0.01, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_row, 0.01, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_row, 0.01, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_row, 0.01, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

    let kernel_arc = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
    let block = TransformerBlock::new(
        0,
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&q_blocks, nq * hd, h, kernel_arc.clone())
            .expect("q")
            .into(),
        Linear1Bit::new(&k_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("k")
            .into(),
        Linear1Bit::new(&v_blocks, nkv * hd, h, kernel_arc.clone())
            .expect("v")
            .into(),
        Linear1Bit::new(&o_blocks, h, nq * hd, kernel_arc.clone())
            .expect("o")
            .into(),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; hd], 1e-6),
        RmsNormLayer::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(&gate_blocks, inter, h, kernel_arc.clone())
            .expect("gate")
            .into(),
        Linear1Bit::new(&up_blocks, inter, h, kernel_arc.clone())
            .expect("up")
            .into(),
        Linear1Bit::new(&down_blocks, h, inter, kernel_arc.clone())
            .expect("down")
            .into(),
        nq,
        nkv,
        hd,
        h,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = &*kernel_arc;
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut state = 42u64;
    let mut hidden = random_tensor(&mut state, h);

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel)
        .expect("forward should succeed");

    assert_no_nan(&hidden, "block_output");
}
