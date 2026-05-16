//! Integration tests for Q5_K and Q6_K linear layers and layer enum integration.

use oxibonsai_core::{BlockQ5K, BlockQ6K};
use oxibonsai_model::layers::linear::LinearLayer;
use oxibonsai_model::layers::linear_kquant_ext::{LinearQ5K, LinearQ6K};

// ---------------------------------------------------------------------------
// LinearQ5K tests
// ---------------------------------------------------------------------------

/// Helper: build a single Q5_K block by quantizing a uniform-value slice.
fn make_q5k_blocks(value: f32, out_features: usize, in_features: usize) -> Vec<BlockQ5K> {
    let blocks_per_row = in_features / 256;
    let total = out_features * blocks_per_row;
    let row_input = vec![value; in_features];
    let mut result = Vec::with_capacity(total);
    let row_blocks = BlockQ5K::quantize(&row_input).expect("quantize ok");
    for _ in 0..out_features {
        result.extend_from_slice(&row_blocks);
    }
    result
}

fn make_q6k_blocks(value: f32, out_features: usize, in_features: usize) -> Vec<BlockQ6K> {
    let blocks_per_row = in_features / 256;
    let total = out_features * blocks_per_row;
    let row_input = vec![value; in_features];
    let mut result = Vec::with_capacity(total);
    let row_blocks = BlockQ6K::quantize(&row_input).expect("quantize ok");
    for _ in 0..out_features {
        result.extend_from_slice(&row_blocks);
    }
    result
}

#[test]
fn test_linear_q5k_new_validates_dimensions() {
    let blocks = make_q5k_blocks(1.0, 4, 256);
    // Correct: 4 out, 256 in → 4 blocks
    assert!(LinearQ5K::new(&blocks, 4, 256).is_ok());

    // Wrong block count: 4 out × (512/256)=2 blocks_per_row → 8 blocks needed, only 4
    assert!(
        LinearQ5K::new(&blocks, 4, 512).is_err(),
        "should fail on block count mismatch"
    );

    // in_features not a multiple of 256
    assert!(
        LinearQ5K::new(&blocks, 4, 128).is_err(),
        "should fail when in_features not multiple of 256"
    );

    // in_features == 0
    assert!(
        LinearQ5K::new(&blocks, 4, 0).is_err(),
        "should fail on in_features=0"
    );
}

#[test]
fn test_linear_q5k_forward_correctness() {
    // 2 output features, 256 input features
    // Uniform weight = 0.5, uniform input = 1.0
    // Expected output per row ≈ 256 × 0.5 = 128.0 (with quantization tolerance)
    let blocks = make_q5k_blocks(0.5, 2, 256);
    let layer = LinearQ5K::new(&blocks, 2, 256).expect("new ok");
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2];
    layer.forward(&input, &mut output).expect("forward ok");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 128.0).abs() < 15.0,
            "row {i}: expected ~128.0, got {v}"
        );
    }
}

#[test]
fn test_linear_q5k_wrong_in_features_error() {
    // in_features=100 is not a multiple of 256 → should error at construction
    let dummy_blocks = vec![BlockQ5K {
        d: half::f16::ONE,
        dmin: half::f16::ZERO,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    }];
    assert!(
        LinearQ5K::new(&dummy_blocks, 1, 100).is_err(),
        "should error on non-256-aligned in_features"
    );
}

#[test]
fn test_linear_q5k_batch_forward() {
    // Batch of 3 tokens, 2 output features, 256 input features
    let blocks = make_q5k_blocks(1.0, 2, 256);
    let layer = LinearQ5K::new(&blocks, 2, 256).expect("new ok");

    let input = vec![1.0f32; 3 * 256]; // m=3 tokens
    let mut output = vec![0.0f32; 3 * 2];
    layer
        .forward_batch(&input, &mut output, 3)
        .expect("batch forward ok");

    // Each row should have output[row_batch × 2 + col] ≈ 256.0
    for token in 0..3 {
        for feat in 0..2 {
            let v = output[token * 2 + feat];
            assert!(
                (v - 256.0).abs() < 20.0,
                "token {token} feat {feat}: expected ~256.0, got {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// LinearQ6K tests
// ---------------------------------------------------------------------------

#[test]
fn test_linear_q6k_new_validates_dimensions() {
    let blocks = make_q6k_blocks(1.0, 4, 256);
    // Correct: 4 out, 256 in → 4 blocks
    assert!(LinearQ6K::new(&blocks, 4, 256).is_ok());

    // Wrong block count: 4 out × (512/256)=2 blocks_per_row → 8 blocks needed, only 4
    assert!(
        LinearQ6K::new(&blocks, 4, 512).is_err(),
        "should fail on block count mismatch"
    );

    // in_features not a multiple of 256
    assert!(
        LinearQ6K::new(&blocks, 4, 128).is_err(),
        "should fail when in_features not multiple of 256"
    );
}

#[test]
fn test_linear_q6k_forward_correctness() {
    // 2 output features, 256 input features
    // Uniform weight = 0.5, uniform input = 1.0 → output ≈ 128.0
    let blocks = make_q6k_blocks(0.5, 2, 256);
    let layer = LinearQ6K::new(&blocks, 2, 256).expect("new ok");
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2];
    layer.forward(&input, &mut output).expect("forward ok");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 128.0).abs() < 15.0,
            "row {i}: expected ~128.0, got {v}"
        );
    }
}

#[test]
fn test_linear_q6k_batch_forward() {
    // Batch of 2 tokens, 3 output features, 256 input features
    let blocks = make_q6k_blocks(1.0, 3, 256);
    let layer = LinearQ6K::new(&blocks, 3, 256).expect("new ok");

    let input = vec![1.0f32; 2 * 256];
    let mut output = vec![0.0f32; 2 * 3];
    layer
        .forward_batch(&input, &mut output, 2)
        .expect("batch forward ok");

    for token in 0..2 {
        for feat in 0..3 {
            let v = output[token * 3 + feat];
            assert!(
                (v - 256.0).abs() < 20.0,
                "token {token} feat {feat}: expected ~256.0, got {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// From trait / LinearLayer enum tests
// ---------------------------------------------------------------------------

#[test]
fn test_linear_layer_from_q5k() {
    let blocks = make_q5k_blocks(1.0, 2, 256);
    let q5k_layer = LinearQ5K::new(&blocks, 2, 256).expect("new ok");
    let layer: LinearLayer = q5k_layer.into();

    assert_eq!(layer.out_features(), 2);
    assert_eq!(layer.in_features(), 256);
    assert!(
        layer.gpu_handle().is_none(),
        "Q5K should have no GPU handle"
    );
    assert!(
        layer.blocks_1bit().is_none(),
        "Q5K should not expose 1-bit blocks"
    );
    assert!(
        layer.blocks_ternary().is_none(),
        "Q5K should not expose ternary blocks"
    );
    assert!(
        layer.blocks_fp8_e4m3().is_none(),
        "Q5K should not expose FP8 E4M3 blocks"
    );
    assert!(
        layer.blocks_fp8_e5m2().is_none(),
        "Q5K should not expose FP8 E5M2 blocks"
    );
    // Q5K blocks should be accessible
    assert!(layer.blocks_q5k().is_some(), "Q5K should expose Q5K blocks");
    assert!(
        layer.blocks_q6k().is_none(),
        "Q5K should not expose Q6K blocks"
    );
}

#[test]
fn test_linear_layer_from_q6k() {
    let blocks = make_q6k_blocks(1.0, 2, 256);
    let q6k_layer = LinearQ6K::new(&blocks, 2, 256).expect("new ok");
    let layer: LinearLayer = q6k_layer.into();

    assert_eq!(layer.out_features(), 2);
    assert_eq!(layer.in_features(), 256);
    assert!(
        layer.gpu_handle().is_none(),
        "Q6K should have no GPU handle"
    );
    assert!(layer.blocks_q6k().is_some(), "Q6K should expose Q6K blocks");
    assert!(
        layer.blocks_q5k().is_none(),
        "Q6K should not expose Q5K blocks"
    );
}

#[test]
fn test_linear_layer_q5k_forward_vec() {
    let blocks = make_q5k_blocks(0.5, 2, 256);
    let q5k = LinearQ5K::new(&blocks, 2, 256).expect("new ok");
    let layer: LinearLayer = q5k.into();

    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2];
    layer
        .forward_vec(&input, &mut output)
        .expect("forward_vec ok");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 128.0).abs() < 15.0,
            "row {i}: expected ~128.0 via LinearLayer::forward_vec, got {v}"
        );
    }
}

#[test]
fn test_linear_layer_q6k_forward_vec() {
    let blocks = make_q6k_blocks(0.5, 2, 256);
    let q6k = LinearQ6K::new(&blocks, 2, 256).expect("new ok");
    let layer: LinearLayer = q6k.into();

    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2];
    layer
        .forward_vec(&input, &mut output)
        .expect("forward_vec ok");

    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 128.0).abs() < 15.0,
            "row {i}: expected ~128.0 via LinearLayer::forward_vec, got {v}"
        );
    }
}

#[test]
fn test_linear_layer_q5k_forward_mat() {
    // Batched (forward_mat)
    let blocks = make_q5k_blocks(1.0, 2, 256);
    let q5k = LinearQ5K::new(&blocks, 2, 256).expect("new ok");
    let layer: LinearLayer = q5k.into();

    let input = vec![1.0f32; 2 * 256]; // m=2
    let mut output = vec![0.0f32; 2 * 2];
    layer
        .forward_mat(&input, &mut output, 2)
        .expect("forward_mat ok");

    for token in 0..2 {
        for feat in 0..2 {
            let v = output[token * 2 + feat];
            assert!(
                (v - 256.0).abs() < 20.0,
                "token {token} feat {feat}: expected ~256.0, got {v}"
            );
        }
    }
}
