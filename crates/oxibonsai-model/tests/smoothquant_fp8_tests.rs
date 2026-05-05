//! Integration tests for SmoothQuant FP8 calibrator and channel-aware quantization.
//!
//! Tests cover:
//! - Per-channel max-abs accumulation
//! - Multi-batch accumulation
//! - Multi-layer independence
//! - smooth_factors correctness
//! - Error paths (layer not found, in_features mismatch)
//! - Outlier reduction via E4M3FN and E5M2 smoothed quantization
//! - Output block counts
//! - Layer tracking helpers

use oxibonsai_core::quant_fp8::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};
use oxibonsai_model::{
    quantize_fp8_e4m3_smooth, quantize_fp8_e5m2_smooth, SmoothQuantCalibrator, SmoothQuantError,
    SmoothQuantConfig,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Dequantize E4M3FN blocks back to f32.
fn dequant_e4m3(blocks: &[BlockFP8E4M3], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    BlockFP8E4M3::dequant(blocks, &mut out).expect("dequant E4M3 failed");
    out
}

/// Dequantize E5M2 blocks back to f32.
fn dequant_e5m2(blocks: &[BlockFP8E5M2], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    BlockFP8E5M2::dequant(blocks, &mut out).expect("dequant E5M2 failed");
    out
}

// ─── Test 1: calibrator records per-channel max-abs ──────────────────────────

#[test]
fn calibrator_records_per_channel_max_abs() {
    // in_features = 4, single batch with 3 tokens
    // activations[token, col] known values:
    //   col 0: 0.1, 0.5, 0.3  -> max = 0.5
    //   col 1: 1.2, 0.0, 0.8  -> max = 1.2
    //   col 2: 0.0, 0.0, 0.0  -> max = 0.0
    //   col 3: -3.0, 1.0, 2.0 -> max_abs = 3.0
    let in_features = 4usize;
    #[rustfmt::skip]
    let activations = vec![
        0.1_f32,  1.2,  0.0, -3.0,
        0.5,      0.0,  0.0,  1.0,
        0.3,      0.8,  0.0,  2.0,
    ];

    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    calib.record_activation("layer0", &activations, in_features);

    // Compute smooth_factors with a dummy all-ones weight matrix (4 rows, 4 cols).
    let out_features = 4usize;
    let weights = vec![1.0_f32; out_features * in_features];
    let factors = calib
        .smooth_factors("layer0", &weights, out_features)
        .expect("smooth_factors failed");

    assert_eq!(factors.len(), in_features, "wrong number of smooth factors");

    // For col 2 (all-zero activations), factor must still be finite and > 0
    // (epsilon prevents division by zero in compute_smooth_factors).
    assert!(factors[2].is_finite() && factors[2] > 0.0, "col 2 factor: {}", factors[2]);

    // col 3 has the largest activation max (3.0), so its factor should dominate.
    // With equal weight maxes (all 1.0), alpha=0.5:
    //   s_j = (max_act + eps)^0.5 / (max_w + eps)^0.5
    // col 3 has higher act_max than col 0, so factor[3] > factor[0].
    assert!(
        factors[3] > factors[0],
        "factor[3]={} should be > factor[0]={} (col 3 has larger act_max)",
        factors[3], factors[0]
    );
}

// ─── Test 2: calibrator accumulates across batches ────────────────────────────

#[test]
fn calibrator_accumulates_across_batches() {
    let in_features = 2usize;
    // Batch 1: col0 max_abs = 1.0, col1 max_abs = 0.5
    let batch1 = vec![1.0_f32, 0.5, -0.3, 0.2];
    // Batch 2: col0 max_abs = 0.2, col1 max_abs = 2.0
    let batch2 = vec![0.2_f32, -2.0, 0.1, 0.9];

    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    calib.record_activation("layer_acc", &batch1, in_features);
    calib.record_activation("layer_acc", &batch2, in_features);

    // After two batches:
    //   col0 running_max_abs = max(1.0, 0.2, 0.2, 0.1) = 1.0
    //   col1 running_max_abs = max(0.5, 0.2, 2.0, 0.9) = 2.0
    // So with equal weight maxes, factor[1] > factor[0].
    let out_features = 2usize;
    let weights = vec![1.0_f32; out_features * in_features];
    let factors = calib
        .smooth_factors("layer_acc", &weights, out_features)
        .expect("smooth_factors failed");

    assert_eq!(factors.len(), in_features);
    assert!(
        factors[1] > factors[0],
        "factor[1]={} should be > factor[0]={} after accumulation",
        factors[1], factors[0]
    );
}

// ─── Test 3: different layers tracked independently ───────────────────────────

#[test]
fn calibrator_different_layers() {
    let in_features = 2usize;
    // layer A: large col0
    let acts_a = vec![10.0_f32, 1.0];
    // layer B: large col1
    let acts_b = vec![1.0_f32, 10.0];

    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    calib.record_activation("layer_a", &acts_a, in_features);
    calib.record_activation("layer_b", &acts_b, in_features);

    let out_features = 2usize;
    let weights = vec![1.0_f32; out_features * in_features];

    let factors_a = calib
        .smooth_factors("layer_a", &weights, out_features)
        .expect("layer_a factors failed");
    let factors_b = calib
        .smooth_factors("layer_b", &weights, out_features)
        .expect("layer_b factors failed");

    // layer A: col0 dominates → factor[0] > factor[1]
    assert!(
        factors_a[0] > factors_a[1],
        "layer_a: factor[0]={} should > factor[1]={}",
        factors_a[0], factors_a[1]
    );
    // layer B: col1 dominates → factor[1] > factor[0]
    assert!(
        factors_b[1] > factors_b[0],
        "layer_b: factor[1]={} should > factor[0]={}",
        factors_b[1], factors_b[0]
    );
}

// ─── Test 4: smooth_factors returns finite, positive values ───────────────────

#[test]
fn smooth_factors_returns_finite_nonzero() {
    let in_features = 8usize;
    let out_features = 4usize;

    // Activations with mixed-magnitude columns
    let mut activations = Vec::with_capacity(3 * in_features);
    for t in 0..3usize {
        for j in 0..in_features {
            activations.push(((t + 1) as f32) * (j as f32 + 1.0) * 0.3);
        }
    }

    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::new(0.5));
    calib.record_activation("fc1", &activations, in_features);

    let weights: Vec<f32> = (0..(out_features * in_features))
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let factors = calib
        .smooth_factors("fc1", &weights, out_features)
        .expect("smooth_factors failed");

    assert_eq!(factors.len(), in_features);
    for (j, &f) in factors.iter().enumerate() {
        assert!(f.is_finite(), "factor[{j}] is not finite: {f}");
        assert!(f > 0.0, "factor[{j}] is not positive: {f}");
    }
}

// ─── Test 5: smooth_factors returns error for unrecorded layer ────────────────

#[test]
fn smooth_factors_layer_not_found_error() {
    let calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    let weights = vec![1.0_f32; 16];
    let result = calib.smooth_factors("nonexistent_layer", &weights, 2);
    match result {
        Err(SmoothQuantError::LayerNotFound(name)) => {
            assert!(
                name.contains("nonexistent_layer"),
                "error should mention the layer name, got: {name}"
            );
        }
        other => panic!("expected LayerNotFound, got: {:?}", other),
    }
}

// ─── Test 6: E4M3FN smoothed quantization reduces outlier-column error ────────

#[test]
fn quantize_fp8_e4m3_smooth_reduces_outlier_error() {
    // Design: activations have a massive outlier in col 3 (1000×) while weights
    // are moderate but col 3 is small (0.01). Smooth factors:
    //   s_3 = 1000^0.5 / 0.01^0.5 = 31.6 / 0.1 = 316  (amplifying)
    //   s_j ≈ 1.0^0.5 / 1.0^0.5 = 1.0 for other cols
    //
    // Smoothed weight col 3 = 0.01 × 316 = 3.16 — now on par with others (≈ 1.0).
    // Unsmoothed: col 3 = 0.01, other cols = 1.0 → relative error on 0.01 is HUGE
    // because the block scale is dominated by the 1.0 values and 0.01 underflows.
    //
    // After smoothing, all weights in the block are near 1–3, so the 0.01 column
    // (now amplified to 3.16) is no longer under-quantized.
    //
    // The benefit is measured as max-abs error on the OUTLIER column (col 3).
    let out_features = 4usize;
    let in_features = 8usize;
    let n = out_features * in_features; // = 32

    let mut weights = vec![1.0_f32; n];
    for row in 0..out_features {
        // col 3: tiny weight → under-quantized without smoothing
        weights[row * in_features + 3] = 0.01;
    }

    // Activations: col 3 has HUGE magnitude → large smooth factor → amplifies col 3 weight.
    let num_tokens = 4usize;
    let mut activations = vec![1.0_f32; num_tokens * in_features];
    for t in 0..num_tokens {
        activations[t * in_features + 3] = 1000.0;
    }

    // --- Unsmoothed path ---
    // With block scale dominated by max weight = 1.0 (not 0.01), the 0.01 value
    // maps to a tiny scaled value that loses precision.
    let blocks_raw = BlockFP8E4M3::quantize(&weights).expect("raw E4M3 quantize failed");
    let recon_raw = dequant_e4m3(&blocks_raw, n);
    // Relative error on col 3 (the tiny weight)
    let err_raw_col3: f32 = weights
        .iter()
        .zip(recon_raw.iter())
        .enumerate()
        .filter(|(idx, _)| idx % in_features == 3)
        .map(|(_, (&w, &r))| (w - r).abs() / w.abs().max(1e-6))
        .fold(0.0_f32, f32::max);

    // --- Smoothed path ---
    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::new(0.5));
    calib.record_activation("linear_e4m3", &activations, in_features);

    let smooth_factors = calib
        .smooth_factors("linear_e4m3", &weights, out_features)
        .expect("smooth_factors failed");

    // The smooth factor for col 3 should be large (amplifying the tiny weight).
    assert!(
        smooth_factors[3] > 10.0,
        "smooth_factors[3]={} should be > 10.0 (amplifying for tiny-weight col with large activation)",
        smooth_factors[3]
    );

    let blocks_smooth =
        quantize_fp8_e4m3_smooth(&weights, out_features, in_features, &smooth_factors)
            .expect("smooth E4M3 quantize failed");

    // Dequantize and un-apply smooth factors to recover approximation of original.
    let deq_smooth_raw = dequant_e4m3(&blocks_smooth, n);
    let recon_smooth: Vec<f32> = deq_smooth_raw
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            let col = idx % in_features;
            let s = smooth_factors[col];
            if s == 0.0 { v } else { v / s }
        })
        .collect();

    // Smoothed path should recover col 3 with lower RELATIVE error.
    let err_smooth_col3: f32 = weights
        .iter()
        .zip(recon_smooth.iter())
        .enumerate()
        .filter(|(idx, _)| idx % in_features == 3)
        .map(|(_, (&w, &r))| (w - r).abs() / w.abs().max(1e-6))
        .fold(0.0_f32, f32::max);

    assert!(
        err_smooth_col3 < err_raw_col3,
        "smoothed E4M3 col-3 relative error ({err_smooth_col3:.4}) should be < unsmoothed ({err_raw_col3:.4})"
    );
}

// ─── Test 7: E5M2 smoothed quantization reduces outlier error ─────────────────

#[test]
fn quantize_fp8_e5m2_smooth_reduces_outlier_error() {
    // Same scenario as test 6 but for E5M2.
    // col 5: large weight outlier (80.0), small activation max (0.1).
    // s_5 = 0.1^0.5 / 80^0.5 ≈ 0.316 / 8.94 ≈ 0.035 → compresses smoothed col-5 weight.
    let out_features = 4usize;
    let in_features = 8usize;
    let n = out_features * in_features;

    let mut weights = vec![1.0_f32; n];
    for row in 0..out_features {
        weights[row * in_features + 5] = 80.0;
    }

    let num_tokens = 4usize;
    let mut activations = vec![1.0_f32; num_tokens * in_features];
    for t in 0..num_tokens {
        // col 5: small activation magnitude → s_5 will be compressive
        activations[t * in_features + 5] = 0.1;
    }

    // Unsmoothed: error on non-outlier columns is high because block scale is
    // dominated by col 5 = 80.0, wasting precision on the 1.0-valued columns.
    let blocks_raw = BlockFP8E5M2::quantize(&weights).expect("raw E5M2 quantize failed");
    let recon_raw = dequant_e5m2(&blocks_raw, n);
    let err_raw_other: f32 = weights
        .iter()
        .zip(recon_raw.iter())
        .enumerate()
        .filter(|(idx, _)| idx % in_features != 5)
        .map(|(_, (&w, &r))| (w - r).abs())
        .fold(0.0_f32, f32::max);

    // Smoothed
    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::new(0.5));
    calib.record_activation("linear5m2", &activations, in_features);
    let smooth_factors = calib
        .smooth_factors("linear5m2", &weights, out_features)
        .expect("smooth_factors failed");

    // Smooth factor for col 5 must be compressive (< 1.0).
    assert!(
        smooth_factors[5] < 0.5,
        "smooth_factors[5]={} should be < 0.5",
        smooth_factors[5]
    );

    let blocks_smooth =
        quantize_fp8_e5m2_smooth(&weights, out_features, in_features, &smooth_factors)
            .expect("smooth E5M2 quantize failed");

    let deq_smooth_raw = dequant_e5m2(&blocks_smooth, n);
    let recon_smooth: Vec<f32> = deq_smooth_raw
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            let col = idx % in_features;
            let s = smooth_factors[col];
            if s == 0.0 { v } else { v / s }
        })
        .collect();
    let err_smooth_other: f32 = weights
        .iter()
        .zip(recon_smooth.iter())
        .enumerate()
        .filter(|(idx, _)| idx % in_features != 5)
        .map(|(_, (&w, &r))| (w - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        err_smooth_other < err_raw_other,
        "smoothed E5M2 non-outlier error ({err_smooth_other:.6}) should be < unsmoothed ({err_raw_other:.6})"
    );
}

// ─── Test 8: record_activation panics on in_features mismatch ────────────────

#[test]
#[should_panic(expected = "in_features mismatch")]
fn calibrator_in_features_mismatch() {
    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    // First recording establishes in_features = 4
    calib.record_activation("layer_mismatch", &[1.0_f32; 4], 4);
    // Second recording with different in_features should panic
    calib.record_activation("layer_mismatch", &[1.0_f32; 6], 6);
}

// ─── Test 9: layer_count tracks unique layers ─────────────────────────────────

#[test]
fn layer_count_tracks_unique_layers() {
    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    assert_eq!(calib.layer_count(), 0);

    calib.record_activation("layer_x", &[1.0_f32, 2.0], 2);
    assert_eq!(calib.layer_count(), 1);

    calib.record_activation("layer_y", &[0.5_f32, 0.3], 2);
    assert_eq!(calib.layer_count(), 2);

    calib.record_activation("layer_z", &[0.1_f32, 0.9], 2);
    assert_eq!(calib.layer_count(), 3);

    // Re-recording an existing layer must not increase count.
    calib.record_activation("layer_x", &[0.2_f32, 0.4], 2);
    assert_eq!(calib.layer_count(), 3, "re-recording existing layer should not increase count");
}

// ─── Test 10: has_layer returns true after recording ──────────────────────────

#[test]
fn has_layer_returns_true_after_recording() {
    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    calib.record_activation("my_layer", &[1.0_f32, -1.0], 2);
    assert!(calib.has_layer("my_layer"), "has_layer should be true after recording");
}

// ─── Test 11: has_layer returns false before recording ────────────────────────

#[test]
fn has_layer_returns_false_before_recording() {
    let calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    assert!(!calib.has_layer("not_recorded"), "has_layer should be false for unrecorded layer");
}

// ─── Test 12: quantize_fp8_e4m3_smooth output block count ────────────────────

#[test]
fn quantize_fp8_e4m3_smooth_output_size() {
    // out_features=8, in_features=8 → 64 weights = 2 FP8 blocks (each 32 weights)
    let out_features = 8usize;
    let in_features = 8usize;
    let n = out_features * in_features; // = 64
    let expected_blocks = n / QK_FP8; // = 2

    let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 + 0.1).collect();
    let activations: Vec<f32> = (0..(2 * in_features))
        .map(|i| (i as f32) * 0.05 + 0.1)
        .collect();

    let mut calib = SmoothQuantCalibrator::new(SmoothQuantConfig::default_alpha());
    calib.record_activation("size_test", &activations, in_features);
    let factors = calib
        .smooth_factors("size_test", &weights, out_features)
        .expect("smooth_factors failed");

    let blocks = quantize_fp8_e4m3_smooth(&weights, out_features, in_features, &factors)
        .expect("smooth E4M3 quantize failed");

    assert_eq!(
        blocks.len(),
        expected_blocks,
        "expected {expected_blocks} blocks for {n} weights, got {}",
        blocks.len()
    );
}
