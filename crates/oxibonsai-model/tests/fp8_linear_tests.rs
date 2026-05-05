//! Phase 16A integration tests: FP8 linear layers, dispatch, and model-variant detection.
//!
//! Tests:
//! 1. `test_linear_fp8_e4m3_gemv`        — 2-row, 32-col LinearFP8E4M3, forward, check output shape.
//! 2. `test_linear_fp8_e5m2_gemv`        — same for E5M2.
//! 3. `test_linear_fp8_e4m3_gemm`        — batched forward_batch (GEMM).
//! 4. `test_linear_layer_fp8_dispatch`   — LinearLayer::FP8E4M3 forward_vec routes correctly.
//! 5. `test_model_variant_fp8_detection` — from_config_and_sample_tensor_type returns FP8Bonsai1_7B.
//! 6. `test_linear_fp8_e4m3_shape_mismatch` — wrong block count → Err on construction.

use half::f16;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, GgufTensorType, Qwen3Config, QK_FP8};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::layers::linear::{LinearFP8E4M3, LinearFP8E5M2, LinearLayer};
use oxibonsai_model::ModelVariant;
use std::sync::Arc;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Reference CPU-only kernel (no GPU, no SIMD — deterministic, portable).
fn ref_kernel() -> Arc<KernelDispatcher> {
    Arc::new(KernelDispatcher::with_tier(KernelTier::Reference))
}

/// Build `num_blocks` FP8 E4M3FN blocks with all weights encoding +1.0
/// (byte 0x38 = 0b0_0111_000 → exp=7, bias=7 → 2^0 = 1.0) and scale = 1.0.
fn make_e4m3_blocks_ones(num_blocks: usize) -> Vec<BlockFP8E4M3> {
    (0..num_blocks)
        .map(|_| BlockFP8E4M3 {
            qs: [0x38u8; QK_FP8], // all weights decode to +1.0 in E4M3FN
            d: f16::ONE,           // scale = 1.0
        })
        .collect()
}

/// Build `num_blocks` FP8 E5M2 blocks where each weight decodes to +1.0.
///
/// In E5M2: 0x3C = 0b0_01111_00 → exp_bits=15, mant=0 → 2^(15-15)*(1.0) = 1.0.
fn make_e5m2_blocks_ones(num_blocks: usize) -> Vec<BlockFP8E5M2> {
    (0..num_blocks)
        .map(|_| BlockFP8E5M2 {
            qs: [0x3Cu8; QK_FP8], // all weights decode to +1.0 in E5M2
            d: f16::ONE,           // scale = 1.0
        })
        .collect()
}

// ─── Test 1: LinearFP8E4M3 GEMV ───────────────────────────────────────────────

/// 2-row × 32-col FP8 E4M3FN linear layer:
/// each weight = +1.0 (0x38), scale = 1.0.
/// Input: all-1.0 vector of length 32.
/// Expected output[i] = sum(w_row_i * x) = 32 × 1.0 × 1.0 = 32.0 per row.
#[test]
fn test_linear_fp8_e4m3_gemv() {
    const OUT: usize = 2;
    const IN: usize = QK_FP8; // 32 — exactly one block per row

    let blocks_owned = make_e4m3_blocks_ones(OUT * (IN / QK_FP8)); // 2 blocks
    let layer = LinearFP8E4M3::new(&blocks_owned, OUT, IN, ref_kernel())
        .expect("LinearFP8E4M3::new should succeed for valid block count");

    assert_eq!(layer.out_features(), OUT);
    assert_eq!(layer.in_features(), IN);

    let input = vec![1.0f32; IN];
    let mut output = vec![0.0f32; OUT];
    layer
        .forward(&input, &mut output)
        .expect("LinearFP8E4M3::forward should succeed");

    assert_eq!(output.len(), OUT, "output length must equal out_features");
    for (i, &val) in output.iter().enumerate() {
        // weight * x = 1.0 * 1.0 = 1.0 per element, summed over IN=32 → 32.0
        // Allow ±5% tolerance for FP8 decode precision.
        let expected = 32.0f32;
        let tol = expected * 0.05 + 0.5;
        assert!(
            (val - expected).abs() < tol,
            "row {i}: expected ~{expected}, got {val}"
        );
    }
}

// ─── Test 2: LinearFP8E5M2 GEMV ───────────────────────────────────────────────

/// 2-row × 32-col FP8 E5M2 linear layer.
/// Same geometry as above, but E5M2 encoding.
/// Expected output[i] ≈ 32.0.
#[test]
fn test_linear_fp8_e5m2_gemv() {
    const OUT: usize = 2;
    const IN: usize = QK_FP8; // 32

    let blocks_owned = make_e5m2_blocks_ones(OUT * (IN / QK_FP8));
    let layer = LinearFP8E5M2::new(&blocks_owned, OUT, IN, ref_kernel())
        .expect("LinearFP8E5M2::new should succeed");

    assert_eq!(layer.out_features(), OUT);
    assert_eq!(layer.in_features(), IN);

    let input = vec![1.0f32; IN];
    let mut output = vec![0.0f32; OUT];
    layer
        .forward(&input, &mut output)
        .expect("LinearFP8E5M2::forward should succeed");

    assert_eq!(output.len(), OUT);
    for (i, &val) in output.iter().enumerate() {
        let expected = 32.0f32;
        // E5M2 has 2-bit mantissa → slightly coarser; allow 10% + 1.0
        let tol = expected * 0.10 + 1.0;
        assert!(
            (val - expected).abs() < tol,
            "row {i}: expected ~{expected}, got {val}"
        );
    }
}

// ─── Test 3: LinearFP8E4M3 batched GEMM ───────────────────────────────────────

/// Batched forward pass (GEMM) with batch_size = 3.
///
/// Layer: 4 rows × 32 cols, all weights +1.0.
/// Input: [3 × 32] all-1.0.
/// Expected: each of the 3 × 4 = 12 output values ≈ 32.0.
#[test]
fn test_linear_fp8_e4m3_gemm() {
    const OUT: usize = 4;
    const IN: usize = QK_FP8; // 32
    const BATCH: usize = 3;

    let blocks_owned = make_e4m3_blocks_ones(OUT * (IN / QK_FP8)); // 4 blocks
    let layer = LinearFP8E4M3::new(&blocks_owned, OUT, IN, ref_kernel())
        .expect("LinearFP8E4M3::new should succeed");

    let input = vec![1.0f32; BATCH * IN];
    let mut output = vec![0.0f32; BATCH * OUT];
    layer
        .forward_batch(&input, &mut output, BATCH)
        .expect("LinearFP8E4M3::forward_batch should succeed");

    assert_eq!(output.len(), BATCH * OUT, "output length = batch * out_features");
    for (idx, &val) in output.iter().enumerate() {
        let expected = 32.0f32;
        let tol = expected * 0.05 + 0.5;
        assert!(
            (val - expected).abs() < tol,
            "output[{idx}]: expected ~{expected}, got {val}"
        );
    }
}

// ─── Test 4: LinearLayer enum dispatch ────────────────────────────────────────

/// `LinearLayer::FP8E4M3` wraps `LinearFP8E4M3` and routes through the
/// correct kernel via `forward_vec`.
///
/// Geometry: 2 rows × 32 cols. Input all-1.0. Expected output ≈ [32.0, 32.0].
#[test]
fn test_linear_layer_fp8_dispatch() {
    const OUT: usize = 2;
    const IN: usize = QK_FP8;

    let blocks_owned = make_e4m3_blocks_ones(OUT * (IN / QK_FP8));
    let inner = LinearFP8E4M3::new(&blocks_owned, OUT, IN, ref_kernel())
        .expect("LinearFP8E4M3::new should succeed");

    let layer = LinearLayer::FP8E4M3(inner);
    assert_eq!(layer.out_features(), OUT);
    assert_eq!(layer.in_features(), IN);
    // FP8 variants have no GPU handle.
    assert!(layer.gpu_handle().is_none(), "FP8 layer should have no GPU handle");

    let input = vec![1.0f32; IN];
    let mut output = vec![0.0f32; OUT];
    layer
        .forward_vec(&input, &mut output)
        .expect("LinearLayer::FP8E4M3 forward_vec should succeed");

    assert_eq!(output.len(), OUT);
    for (i, &val) in output.iter().enumerate() {
        let expected = 32.0f32;
        let tol = expected * 0.05 + 0.5;
        assert!(
            (val - expected).abs() < tol,
            "row {i}: expected ~{expected}, got {val}"
        );
    }
}

// ─── Test 5: ModelVariant FP8 detection ───────────────────────────────────────

/// `ModelVariant::from_config_and_sample_tensor_type` should return
/// `FP8Bonsai1_7B` when presented with a 1.7B config and `F8_E4M3` tensor type.
#[test]
fn test_model_variant_fp8_detection() {
    let config_1_7b = Qwen3Config::bonsai_1_7b();
    let variant = ModelVariant::from_config_and_sample_tensor_type(
        &config_1_7b,
        GgufTensorType::F8_E4M3,
    );
    assert_eq!(
        variant,
        ModelVariant::FP8Bonsai1_7B,
        "1.7B config + F8_E4M3 tensor type should yield FP8Bonsai1_7B"
    );

    // Also verify E5M2 variant upgrades correctly.
    let config_8b = Qwen3Config::bonsai_8b();
    let variant_8b = ModelVariant::from_config_and_sample_tensor_type(
        &config_8b,
        GgufTensorType::F8_E5M2,
    );
    assert_eq!(
        variant_8b,
        ModelVariant::FP8Bonsai8B,
        "8B config + F8_E5M2 tensor type should yield FP8Bonsai8B"
    );

    // Sanity: non-FP8 tensor type should NOT yield an FP8 variant.
    let variant_q1 = ModelVariant::from_config_and_sample_tensor_type(
        &config_1_7b,
        GgufTensorType::Q1_0_g128,
    );
    assert_ne!(
        variant_q1,
        ModelVariant::FP8Bonsai1_7B,
        "Q1_0_G128 tensor type should not be detected as FP8"
    );
}

// ─── Test 6: Shape mismatch on construction ───────────────────────────────────

/// Providing the wrong number of blocks must return `Err(ModelError::ShapeMismatch)`.
///
/// For a 4-row × 32-col layer we need exactly 4 blocks (4 rows × 1 block/row).
/// Providing 3 blocks must fail.
#[test]
fn test_linear_fp8_e4m3_shape_mismatch() {
    const OUT: usize = 4;
    const IN: usize = QK_FP8; // 32

    // Deliberately short by 1 block.
    let wrong_blocks = make_e4m3_blocks_ones(OUT - 1); // 3 blocks, need 4
    let result = LinearFP8E4M3::new(&wrong_blocks, OUT, IN, ref_kernel());
    assert!(
        result.is_err(),
        "LinearFP8E4M3::new with wrong block count should return Err"
    );

    // Also verify in_features not a multiple of QK_FP8 fails.
    let any_blocks = make_e4m3_blocks_ones(1);
    let result_bad_in = LinearFP8E4M3::new(&any_blocks, 1, 31, ref_kernel()); // 31 % 32 != 0
    assert!(
        result_bad_in.is_err(),
        "LinearFP8E4M3::new with in_features=31 (not multiple of 32) should return Err"
    );
}
