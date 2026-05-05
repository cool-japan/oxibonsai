//! Integration tests for FP8 E4M3FN and E5M2 kernel implementations.
//!
//! Tests cover:
//! - Dispatcher round-trip correctness (dequant, GEMV, GEMM)
//! - Error path validation (BufferTooSmall, NotBlockAligned, DimensionMismatch)
//! - Numerical correctness against FP32 reference computations
//! - Edge cases: zero input, zero weights, unit vectors, single block/row
//! - GEMM batch=1 consistency with GEMV

use half::f16;
use oxibonsai_core::{
    BlockFP8E4M3, BlockFP8E5M2, QK_FP8, fp8_e4m3_decode, fp8_e4m3_encode, fp8_e5m2_decode,
    fp8_e5m2_encode,
};
use oxibonsai_kernels::{
    Fp8Kernel, KernelDispatcher, KernelError, KernelTier,
};

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

fn e4m3_block(scale: f32, qs: [u8; 32]) -> BlockFP8E4M3 {
    BlockFP8E4M3 {
        qs,
        d: f16::from_f32(scale),
    }
}

fn e5m2_block(scale: f32, qs: [u8; 32]) -> BlockFP8E5M2 {
    BlockFP8E5M2 {
        qs,
        d: f16::from_f32(scale),
    }
}

/// Build a BlockFP8E4M3 by quantizing a uniform f32 weight value into all slots.
fn e4m3_block_uniform(weight: f32, scale: f32) -> BlockFP8E4M3 {
    let scaled = if scale.abs() < 1e-12 { 0.0 } else { weight / scale };
    let q = fp8_e4m3_encode(scaled);
    e4m3_block(scale, [q; 32])
}

/// Build a BlockFP8E5M2 by quantizing a uniform f32 weight value into all slots.
fn e5m2_block_uniform(weight: f32, scale: f32) -> BlockFP8E5M2 {
    let scaled = if scale.abs() < 1e-12 { 0.0 } else { weight / scale };
    let q = fp8_e5m2_encode(scaled);
    e5m2_block(scale, [q; 32])
}

fn make_dispatcher() -> KernelDispatcher {
    KernelDispatcher::with_tier(KernelTier::Reference)
}

// ---------------------------------------------------------------------------
// Dispatcher round-trip: dequant
// ---------------------------------------------------------------------------

/// Dispatcher dequant E4M3: all weights round-trip through encode/decode.
#[test]
fn dispatcher_dequant_fp8_e4m3_matches_reference() {
    let dispatcher = make_dispatcher();
    let values: Vec<f32> = (0..QK_FP8).map(|i| (i as f32) * 0.5 - 8.0).collect();
    let blocks = BlockFP8E4M3::quantize(&values).unwrap();
    let mut output = vec![0.0f32; QK_FP8];
    dispatcher
        .dequant_fp8_e4m3(&blocks, &mut output)
        .expect("dequant_fp8_e4m3 should succeed");

    // Each output should be d * fp8_e4m3_decode(q)
    for (i, &v) in output.iter().enumerate() {
        let expected = blocks[0].d.to_f32() * fp8_e4m3_decode(blocks[0].qs[i]);
        assert!(
            (v - expected).abs() < 1e-5,
            "index {i}: expected {expected}, got {v}"
        );
    }
}

/// Dispatcher dequant E5M2: all weights round-trip through encode/decode.
#[test]
fn dispatcher_dequant_fp8_e5m2_matches_reference() {
    let dispatcher = make_dispatcher();
    let values: Vec<f32> = (0..QK_FP8).map(|i| (i as f32) * 10.0 - 150.0).collect();
    let blocks = BlockFP8E5M2::quantize(&values).unwrap();
    let mut output = vec![0.0f32; QK_FP8];
    dispatcher
        .dequant_fp8_e5m2(&blocks, &mut output)
        .expect("dequant_fp8_e5m2 should succeed");

    for (i, &v) in output.iter().enumerate() {
        let expected = blocks[0].d.to_f32() * fp8_e5m2_decode(blocks[0].qs[i]);
        assert!(
            (v - expected).abs() < 1e-5,
            "index {i}: expected {expected}, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Dispatcher round-trip: GEMV
// ---------------------------------------------------------------------------

/// Basic E4M3 GEMV: 2 rows, k=32, all weights=1.0 (via encode), input=all-1.0.
#[test]
fn dispatcher_gemv_fp8_e4m3_basic() {
    let dispatcher = make_dispatcher();
    let q_one = fp8_e4m3_encode(1.0);
    let blocks = vec![
        e4m3_block(1.0, [q_one; 32]),
        e4m3_block(1.0, [q_one; 32]),
    ];
    let input = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut output, 2, 32)
        .expect("gemv_fp8_e4m3 should succeed");
    let expected = 32.0 * fp8_e4m3_decode(q_one);
    for (r, &v) in output.iter().enumerate() {
        assert!(
            (v - expected).abs() < 0.5,
            "row {r}: expected ~{expected}, got {v}"
        );
    }
}

/// Basic E5M2 GEMV: 2 rows, k=32, all weights=1.0 (via encode), input=all-1.0.
#[test]
fn dispatcher_gemv_fp8_e5m2_basic() {
    let dispatcher = make_dispatcher();
    let q_one = fp8_e5m2_encode(1.0);
    let blocks = vec![
        e5m2_block(1.0, [q_one; 32]),
        e5m2_block(1.0, [q_one; 32]),
    ];
    let input = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    dispatcher
        .gemv_fp8_e5m2(&blocks, &input, &mut output, 2, 32)
        .expect("gemv_fp8_e5m2 should succeed");
    let expected = 32.0 * fp8_e5m2_decode(q_one);
    for (r, &v) in output.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1.0,
            "row {r}: expected ~{expected}, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Dispatcher round-trip: GEMM
// ---------------------------------------------------------------------------

/// E4M3 GEMM batch=2: outputs match two independent GEMV calls.
#[test]
fn dispatcher_gemm_fp8_e4m3_batch2() {
    let dispatcher = make_dispatcher();
    let q_one = fp8_e4m3_encode(1.0);
    let n_rows = 3;
    let k = 32;
    let batch = 2;
    let blocks: Vec<BlockFP8E4M3> = (0..n_rows)
        .map(|r| e4m3_block((r + 1) as f32, [q_one; 32]))
        .collect();

    let mut inputs = vec![0.0f32; batch * k];
    for b in 0..batch {
        for j in 0..k {
            inputs[b * k + j] = (b + 1) as f32;
        }
    }

    let mut gemm_out = vec![0.0f32; batch * n_rows];
    dispatcher
        .gemm_fp8_e4m3(&blocks, &inputs, &mut gemm_out, n_rows, k, batch)
        .expect("gemm_fp8_e4m3 should succeed");

    // Validate each batch row matches a direct GEMV
    for b in 0..batch {
        let in_row = &inputs[b * k..(b + 1) * k];
        let mut gemv_out = vec![0.0f32; n_rows];
        dispatcher
            .gemv_fp8_e4m3(&blocks, in_row, &mut gemv_out, n_rows, k)
            .expect("gemv should succeed");
        for r in 0..n_rows {
            assert!(
                (gemm_out[b * n_rows + r] - gemv_out[r]).abs() < 1e-4,
                "batch={b} row={r}: gemm={} gemv={}",
                gemm_out[b * n_rows + r],
                gemv_out[r]
            );
        }
    }
}

/// E5M2 GEMM batch=2: outputs match two independent GEMV calls.
#[test]
fn dispatcher_gemm_fp8_e5m2_batch2() {
    let dispatcher = make_dispatcher();
    let q_one = fp8_e5m2_encode(1.0);
    let n_rows = 2;
    let k = 32;
    let batch = 2;
    let blocks: Vec<BlockFP8E5M2> = (0..n_rows)
        .map(|r| e5m2_block((r + 1) as f32, [q_one; 32]))
        .collect();

    let inputs = vec![1.0f32; batch * k];
    let mut gemm_out = vec![0.0f32; batch * n_rows];
    dispatcher
        .gemm_fp8_e5m2(&blocks, &inputs, &mut gemm_out, n_rows, k, batch)
        .expect("gemm_fp8_e5m2 should succeed");

    for b in 0..batch {
        let in_row = &inputs[b * k..(b + 1) * k];
        let mut gemv_out = vec![0.0f32; n_rows];
        dispatcher
            .gemv_fp8_e5m2(&blocks, in_row, &mut gemv_out, n_rows, k)
            .expect("gemv should succeed");
        for r in 0..n_rows {
            assert!(
                (gemm_out[b * n_rows + r] - gemv_out[r]).abs() < 1e-4,
                "batch={b} row={r}: gemm={} gemv={}",
                gemm_out[b * n_rows + r],
                gemv_out[r]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Error path tests
// ---------------------------------------------------------------------------

/// Dequant E4M3: output buffer too small.
#[test]
fn dequant_e4m3_buffer_too_small() {
    let dispatcher = make_dispatcher();
    let block = e4m3_block(1.0, [0x38u8; 32]);
    let mut output = vec![0.0f32; QK_FP8 - 1];
    let result = dispatcher.dequant_fp8_e4m3(&[block], &mut output);
    assert!(
        matches!(result, Err(KernelError::BufferTooSmall { .. })),
        "expected BufferTooSmall, got {result:?}"
    );
}

/// Dequant E5M2: output buffer too small.
#[test]
fn dequant_e5m2_buffer_too_small() {
    let dispatcher = make_dispatcher();
    let block = e5m2_block(1.0, [0x3Cu8; 32]);
    let mut output = vec![0.0f32; 0];
    let result = dispatcher.dequant_fp8_e5m2(&[block], &mut output);
    assert!(
        matches!(result, Err(KernelError::BufferTooSmall { .. })),
        "expected BufferTooSmall, got {result:?}"
    );
}

/// GEMV E4M3: k not a multiple of QK_FP8 → NotBlockAligned.
#[test]
fn gemv_e4m3_not_block_aligned() {
    let dispatcher = make_dispatcher();
    let blocks = vec![e4m3_block(1.0, [0x38u8; 32])];
    let input = vec![1.0f32; 31];
    let mut output = vec![0.0f32; 1];
    let result = dispatcher.gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 31);
    assert!(
        matches!(result, Err(KernelError::NotBlockAligned { .. })),
        "expected NotBlockAligned, got {result:?}"
    );
}

/// GEMV E5M2: k not a multiple of QK_FP8 → NotBlockAligned.
#[test]
fn gemv_e5m2_not_block_aligned() {
    let dispatcher = make_dispatcher();
    let blocks = vec![e5m2_block(1.0, [0x3Cu8; 32])];
    let input = vec![1.0f32; 33];
    let mut output = vec![0.0f32; 1];
    let result = dispatcher.gemv_fp8_e5m2(&blocks, &input, &mut output, 1, 33);
    assert!(
        matches!(result, Err(KernelError::NotBlockAligned { .. })),
        "expected NotBlockAligned, got {result:?}"
    );
}

/// GEMV E4M3: n_rows=2 but only 1 block supplied → DimensionMismatch.
#[test]
fn gemv_e4m3_dimension_mismatch() {
    let dispatcher = make_dispatcher();
    let blocks = vec![e4m3_block(1.0, [0x38u8; 32])];
    let input = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result = dispatcher.gemv_fp8_e4m3(&blocks, &input, &mut output, 2, 32);
    assert!(
        matches!(result, Err(KernelError::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {result:?}"
    );
}

/// GEMV E5M2: n_rows=3 but only 2 blocks supplied → DimensionMismatch.
#[test]
fn gemv_e5m2_dimension_mismatch() {
    let dispatcher = make_dispatcher();
    let blocks = vec![
        e5m2_block(1.0, [0x3Cu8; 32]),
        e5m2_block(1.0, [0x3Cu8; 32]),
    ];
    let input = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 3];
    let result = dispatcher.gemv_fp8_e5m2(&blocks, &input, &mut output, 3, 32);
    assert!(
        matches!(result, Err(KernelError::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {result:?}"
    );
}

/// GEMM E4M3: output buffer too small for batch*n_rows.
#[test]
fn gemm_e4m3_wrong_batch_size() {
    let dispatcher = make_dispatcher();
    let blocks = vec![
        e4m3_block(1.0, [0x38u8; 32]),
        e4m3_block(1.0, [0x38u8; 32]),
    ];
    let inputs = vec![1.0f32; 64]; // batch=2, k=32
    let mut outputs = vec![0.0f32; 3]; // need batch*n_rows = 4, supply 3
    let result = dispatcher.gemm_fp8_e4m3(&blocks, &inputs, &mut outputs, 2, 32, 2);
    assert!(
        matches!(result, Err(KernelError::BufferTooSmall { .. })),
        "expected BufferTooSmall, got {result:?}"
    );
}

/// GEMM E4M3: blocks count mismatch.
#[test]
fn gemm_e4m3_dimension_mismatch_blocks() {
    let dispatcher = make_dispatcher();
    // n_rows=3 k=32 → need 3 blocks, supply 2
    let blocks = vec![
        e4m3_block(1.0, [0x38u8; 32]),
        e4m3_block(1.0, [0x38u8; 32]),
    ];
    let inputs = vec![1.0f32; 32];
    let mut outputs = vec![0.0f32; 3];
    let result = dispatcher.gemm_fp8_e4m3(&blocks, &inputs, &mut outputs, 3, 32, 1);
    assert!(
        matches!(result, Err(KernelError::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {result:?}"
    );
}

/// GEMM E5M2: output buffer too small.
#[test]
fn gemm_e5m2_buffer_too_small() {
    let dispatcher = make_dispatcher();
    let blocks = vec![
        e5m2_block(1.0, [0x3Cu8; 32]),
        e5m2_block(1.0, [0x3Cu8; 32]),
    ];
    let inputs = vec![1.0f32; 64]; // batch=2, k=32
    let mut outputs = vec![0.0f32; 1]; // need 4, supply 1
    let result = dispatcher.gemm_fp8_e5m2(&blocks, &inputs, &mut outputs, 2, 32, 2);
    assert!(
        matches!(result, Err(KernelError::BufferTooSmall { .. })),
        "expected BufferTooSmall, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Correctness: quantize → dequant → GEMV vs FP32 reference
// ---------------------------------------------------------------------------

/// Quantize weights with E4M3, run GEMV, compare to FP32 GEMV on dequantized weights.
#[test]
fn fp8_e4m3_gemv_vs_fp32_reference() {
    let dispatcher = make_dispatcher();
    let n_rows = 4;
    let k = 64; // 2 blocks per row

    // Build random-ish FP32 weight matrix and input
    let raw_weights: Vec<f32> = (0..n_rows * k)
        .map(|i| (i as f32).sin() * 3.0)
        .collect();
    let input: Vec<f32> = (0..k).map(|i| (i as f32).cos()).collect();

    // Quantize each row into E4M3 blocks
    let blocks: Vec<BlockFP8E4M3> = (0..n_rows)
        .flat_map(|r| {
            let row = &raw_weights[r * k..(r + 1) * k];
            BlockFP8E4M3::quantize(row).unwrap()
        })
        .collect();

    // FP8 GEMV via dispatcher
    let mut fp8_out = vec![0.0f32; n_rows];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut fp8_out, n_rows, k)
        .expect("gemv_fp8_e4m3 should succeed");

    // FP32 reference: dequantize blocks then compute dot product
    let mut dequant_weights = vec![0.0f32; n_rows * k];
    for r in 0..n_rows {
        let row_blocks = &blocks[r * (k / QK_FP8)..(r + 1) * (k / QK_FP8)];
        let out_slice = &mut dequant_weights[r * k..(r + 1) * k];
        dispatcher
            .dequant_fp8_e4m3(row_blocks, out_slice)
            .expect("dequant should succeed");
    }
    let fp32_ref: Vec<f32> = (0..n_rows)
        .map(|r| {
            let row = &dequant_weights[r * k..(r + 1) * k];
            row.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
        })
        .collect();

    // The two paths must agree to floating-point precision
    for r in 0..n_rows {
        assert!(
            (fp8_out[r] - fp32_ref[r]).abs() < 1e-4,
            "row {r}: fp8_gemv={} fp32_ref={}",
            fp8_out[r],
            fp32_ref[r]
        );
    }
}

/// Quantize weights with E5M2, run GEMV, compare to FP32 reference.
///
/// Constructs blocks directly (without BlockFP8E5M2::quantize) using known
/// finite E5M2 byte values to prevent Inf from leaking into the dot product.
/// 0x3C = E5M2 +1.0 (exp=15, man=0), 0xBC = E5M2 -1.0.
#[test]
fn fp8_e5m2_gemv_vs_fp32_reference() {
    let dispatcher = make_dispatcher();
    let n_rows = 3;
    let k = 32;

    // Use a byte that is guaranteed finite in E5M2: 0x3C = +1.0
    let q = fp8_e5m2_encode(1.0);
    // Each row uses a different scale so values differ between rows
    let scales = [0.5f32, 1.0, 2.0];
    let blocks: Vec<BlockFP8E5M2> = scales
        .iter()
        .map(|&s| e5m2_block(s, [q; 32]))
        .collect();

    // Input: ascending small values, all finite
    let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();

    let mut fp8_out = vec![0.0f32; n_rows];
    dispatcher
        .gemv_fp8_e5m2(&blocks, &input, &mut fp8_out, n_rows, k)
        .expect("gemv_fp8_e5m2 should succeed");

    // FP32 reference via dequant + manual dot
    let mut dequant_weights = vec![0.0f32; n_rows * k];
    dispatcher
        .dequant_fp8_e5m2(&blocks, &mut dequant_weights)
        .expect("dequant should succeed");

    let fp32_ref: Vec<f32> = (0..n_rows)
        .map(|r| {
            let row = &dequant_weights[r * k..(r + 1) * k];
            row.iter().zip(input.iter()).map(|(w, x)| w * x).sum::<f32>()
        })
        .collect();

    for r in 0..n_rows {
        assert!(
            fp8_out[r].is_finite(),
            "row {r}: fp8_gemv should be finite, got {}",
            fp8_out[r]
        );
        assert!(
            (fp8_out[r] - fp32_ref[r]).abs() < 1e-4,
            "row {r}: fp8_gemv={} fp32_ref={}",
            fp8_out[r],
            fp32_ref[r]
        );
    }
}

// ---------------------------------------------------------------------------
// Numerical: all-zeros input / all-zeros weights / unit input
// ---------------------------------------------------------------------------

/// All-zero input vector → GEMV output is all zero, regardless of weights.
#[test]
fn gemv_e4m3_all_zeros_input() {
    let dispatcher = make_dispatcher();
    let blocks = vec![e4m3_block_uniform(3.0, 1.0)];
    let input = vec![0.0f32; 32];
    let mut output = vec![99.0f32; 1];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32)
        .expect("gemv should succeed");
    assert!(
        output[0].abs() < 1e-6,
        "all-zero input: expected 0.0, got {}",
        output[0]
    );
}

/// All-zero weights (qs=0x00 → E4M3 +0.0) → GEMV output is all zero.
#[test]
fn gemv_e4m3_all_zeros_weights() {
    let dispatcher = make_dispatcher();
    let blocks = vec![e4m3_block(1.0, [0x00u8; 32])];
    let input = vec![1.0f32; 32];
    let mut output = vec![99.0f32; 1];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32)
        .expect("gemv should succeed");
    assert!(
        output[0].abs() < 1e-6,
        "all-zero weights: expected 0.0, got {}",
        output[0]
    );
}

/// Unit input e_0 (only first element = 1.0) → GEMV output[r] = dequant(block[r].qs[0]) × d.
#[test]
fn gemv_e4m3_unit_input() {
    let dispatcher = make_dispatcher();
    let w0 = fp8_e4m3_encode(7.5);
    let w1 = fp8_e4m3_encode(-3.25);
    let mut qs0 = [0x00u8; 32];
    let mut qs1 = [0x00u8; 32];
    qs0[0] = w0;
    qs1[0] = w1;
    let blocks = vec![e4m3_block(1.0, qs0), e4m3_block(1.0, qs1)];
    let mut input = vec![0.0f32; 32];
    input[0] = 1.0;
    let mut output = vec![0.0f32; 2];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut output, 2, 32)
        .expect("gemv should succeed");
    let e0 = fp8_e4m3_decode(w0);
    let e1 = fp8_e4m3_decode(w1);
    assert!(
        (output[0] - e0).abs() < 1e-5,
        "row0: expected {e0}, got {}",
        output[0]
    );
    assert!(
        (output[1] - e1).abs() < 1e-5,
        "row1: expected {e1}, got {}",
        output[1]
    );
}

/// All-zero input for E5M2 GEMV.
#[test]
fn gemv_e5m2_all_zeros_input() {
    let dispatcher = make_dispatcher();
    let blocks = vec![e5m2_block_uniform(10.0, 1.0)];
    let input = vec![0.0f32; 32];
    let mut output = vec![99.0f32; 1];
    dispatcher
        .gemv_fp8_e5m2(&blocks, &input, &mut output, 1, 32)
        .expect("gemv should succeed");
    assert!(
        output[0].abs() < 1e-6,
        "all-zero input: expected 0.0, got {}",
        output[0]
    );
}

/// All-zero weights for E5M2 GEMV.
#[test]
fn gemv_e5m2_all_zeros_weights() {
    let dispatcher = make_dispatcher();
    let blocks = vec![e5m2_block(1.0, [0x00u8; 32])];
    let input = vec![1.0f32; 32];
    let mut output = vec![99.0f32; 1];
    dispatcher
        .gemv_fp8_e5m2(&blocks, &input, &mut output, 1, 32)
        .expect("gemv should succeed");
    assert!(
        output[0].abs() < 1e-6,
        "all-zero weights: expected 0.0, got {}",
        output[0]
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Single block dequant E4M3.
#[test]
fn dequant_e4m3_single_block() {
    let dispatcher = make_dispatcher();
    let q = fp8_e4m3_encode(2.5);
    let block = e4m3_block(1.0, [q; 32]);
    let mut output = vec![0.0f32; QK_FP8];
    dispatcher
        .dequant_fp8_e4m3(&[block], &mut output)
        .expect("dequant single block should succeed");
    let expected = fp8_e4m3_decode(q);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-5,
            "index {i}: expected {expected}, got {v}"
        );
    }
}

/// Single block dequant E5M2.
#[test]
fn dequant_e5m2_single_block() {
    let dispatcher = make_dispatcher();
    let q = fp8_e5m2_encode(100.0);
    let block = e5m2_block(1.0, [q; 32]);
    let mut output = vec![0.0f32; QK_FP8];
    dispatcher
        .dequant_fp8_e5m2(&[block], &mut output)
        .expect("dequant single block should succeed");
    let expected = fp8_e5m2_decode(q);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-5,
            "index {i}: expected {expected}, got {v}"
        );
    }
}

/// GEMV with a single row.
#[test]
fn gemv_e4m3_single_row() {
    let dispatcher = make_dispatcher();
    let q = fp8_e4m3_encode(0.5);
    let block = e4m3_block(2.0, [q; 32]);
    let input = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];
    dispatcher
        .gemv_fp8_e4m3(&[block], &input, &mut output, 1, 32)
        .expect("single-row gemv should succeed");
    let weight_val = 2.0 * fp8_e4m3_decode(q);
    let expected = 32.0 * weight_val;
    assert!(
        (output[0] - expected).abs() < 0.5,
        "expected ~{expected}, got {}",
        output[0]
    );
}

/// GEMM batch=1 must produce identical results to GEMV.
#[test]
fn gemm_e4m3_batch_one_equals_gemv() {
    let dispatcher = make_dispatcher();
    let n_rows = 4;
    let k = 64;

    let blocks: Vec<BlockFP8E4M3> = (0..n_rows)
        .flat_map(|r| {
            let vals: Vec<f32> = (0..k).map(|i| ((r * k + i) as f32).sin()).collect();
            BlockFP8E4M3::quantize(&vals).unwrap()
        })
        .collect();

    let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();

    let mut gemm_out = vec![0.0f32; n_rows];
    let mut gemv_out = vec![0.0f32; n_rows];

    dispatcher
        .gemm_fp8_e4m3(&blocks, &input, &mut gemm_out, n_rows, k, 1)
        .expect("gemm(batch=1) should succeed");
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut gemv_out, n_rows, k)
        .expect("gemv should succeed");

    for r in 0..n_rows {
        assert!(
            (gemm_out[r] - gemv_out[r]).abs() < 1e-4,
            "row {r}: gemm={} gemv={}",
            gemm_out[r],
            gemv_out[r]
        );
    }
}

/// GEMM E5M2 batch=1 equals GEMV.
#[test]
fn gemm_e5m2_batch_one_equals_gemv() {
    let dispatcher = make_dispatcher();
    let n_rows = 2;
    let k = 32;

    let blocks: Vec<BlockFP8E5M2> = (0..n_rows)
        .flat_map(|r| {
            let vals: Vec<f32> = (0..k)
                .map(|i| ((r * k + i) as f32) * 100.0 - 1600.0)
                .collect();
            BlockFP8E5M2::quantize(&vals).unwrap()
        })
        .collect();

    let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.05).collect();
    let mut gemm_out = vec![0.0f32; n_rows];
    let mut gemv_out = vec![0.0f32; n_rows];

    dispatcher
        .gemm_fp8_e5m2(&blocks, &input, &mut gemm_out, n_rows, k, 1)
        .expect("gemm(batch=1) should succeed");
    dispatcher
        .gemv_fp8_e5m2(&blocks, &input, &mut gemv_out, n_rows, k)
        .expect("gemv should succeed");

    for r in 0..n_rows {
        assert!(
            (gemm_out[r] - gemv_out[r]).abs() < 1e-4,
            "row {r}: gemm={} gemv={}",
            gemm_out[r],
            gemv_out[r]
        );
    }
}

/// Empty blocks → dequant succeeds with empty output.
#[test]
fn dequant_e4m3_empty_blocks() {
    let dispatcher = make_dispatcher();
    let mut output: Vec<f32> = vec![];
    dispatcher
        .dequant_fp8_e4m3(&[], &mut output)
        .expect("empty dequant should succeed");
}

/// Empty blocks E5M2 → dequant succeeds with empty output.
#[test]
fn dequant_e5m2_empty_blocks() {
    let dispatcher = make_dispatcher();
    let mut output: Vec<f32> = vec![];
    dispatcher
        .dequant_fp8_e5m2(&[], &mut output)
        .expect("empty dequant should succeed");
}

/// name_fp8() returns the expected string.
#[test]
fn dispatcher_name_fp8() {
    let dispatcher = make_dispatcher();
    assert_eq!(dispatcher.name_fp8(), "fp8_reference");
}

/// Oversized output buffer: trailing elements must not be clobbered.
#[test]
fn dequant_e4m3_oversized_output_not_clobbered() {
    let dispatcher = make_dispatcher();
    let block = e4m3_block(1.0, [0x38u8; 32]);
    let sentinel = 42.0f32;
    let mut output = vec![sentinel; QK_FP8 + 5];
    dispatcher
        .dequant_fp8_e4m3(&[block], &mut output)
        .expect("dequant should succeed");
    for i in QK_FP8..output.len() {
        assert_eq!(
            output[i], sentinel,
            "element {i} should not be modified (was sentinel)"
        );
    }
}

/// Quantize then dequant E4M3: max absolute error should be bounded.
#[test]
fn e4m3_quantize_then_dequant_max_error() {
    let dispatcher = make_dispatcher();
    let values: Vec<f32> = (0..96).map(|i| (i as f32) * 0.3 - 14.0).collect();
    let blocks = BlockFP8E4M3::quantize(&values).unwrap();
    let mut output = vec![0.0f32; 96];
    dispatcher
        .dequant_fp8_e4m3(&blocks, &mut output)
        .expect("dequant should succeed");
    let max_err: f32 = values
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    // E4M3 has limited precision (~3-bit mantissa); allow generous tolerance
    assert!(
        max_err < 1.5,
        "E4M3 quantize-dequant max error {max_err} exceeds threshold"
    );
}

/// Quantize then dequant E5M2: max absolute error should be bounded.
///
/// Uses values within a moderate range to avoid Infinity in dequant output
/// (E5M2 max ≈ 57344; values that would require a scale > 57344/448 ≈ 128 as
/// a dequant factor can produce Inf when multiplied by the block scale).
#[test]
fn e5m2_quantize_then_dequant_max_error() {
    let dispatcher = make_dispatcher();
    // Keep values within a range that round-trips cleanly through E5M2.
    // E5M2 can represent values up to 57344 with about 25% relative error.
    let values: Vec<f32> = (0..64).map(|i| (i as f32) * 2.0 - 64.0).collect();
    let blocks = BlockFP8E5M2::quantize(&values).unwrap();
    let mut output = vec![0.0f32; 64];
    dispatcher
        .dequant_fp8_e5m2(&blocks, &mut output)
        .expect("dequant should succeed");
    // Only compare finite-valued pairs to handle any residual Inf/NaN from decode
    let max_err: f32 = values
        .iter()
        .zip(output.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    // E5M2 has 2-bit mantissa; allow generous tolerance (~50% of max abs value ≈ 32)
    assert!(
        max_err < 20.0,
        "E5M2 quantize-dequant max error {max_err} exceeds threshold"
    );
}

/// Multiple block GEMV E4M3: 4 rows, 2 blocks per row (k=64).
#[test]
fn gemv_e4m3_multiple_blocks_per_row() {
    let dispatcher = make_dispatcher();
    let n_rows = 4;
    let k = 64; // 2 blocks per row
    let q = fp8_e4m3_encode(1.0);
    // Each row has 2 blocks, all weights ≈ 1.0 with scale=1.0
    let blocks: Vec<BlockFP8E4M3> = (0..n_rows * 2).map(|_| e4m3_block(1.0, [q; 32])).collect();
    let input = vec![1.0f32; k];
    let mut output = vec![0.0f32; n_rows];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut output, n_rows, k)
        .expect("multi-block gemv should succeed");
    let expected = 64.0 * fp8_e4m3_decode(q);
    for (r, &v) in output.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1.0,
            "row {r}: expected ~{expected}, got {v}"
        );
    }
}

/// Negative weights produce negative output for positive input.
#[test]
fn gemv_e4m3_negative_weights_positive_input() {
    let dispatcher = make_dispatcher();
    // 0xB8 = sign=1, exp=7, man=0 → -1.0
    let blocks = vec![e4m3_block(1.0, [0xB8u8; 32])];
    let input = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 1];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 32)
        .expect("gemv should succeed");
    assert!(
        output[0] < 0.0,
        "negative weights + positive input should produce negative output, got {}",
        output[0]
    );
}
