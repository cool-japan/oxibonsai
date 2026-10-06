//! Cross-crate integration tests for the OxiBonsai kernel pipeline.
//!
//! Tests the full path from block creation through dequantization,
//! GEMV, and GEMM across crate boundaries.

use half::f16;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_core::GgufTensorType;
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_kernels::{KernelDispatcher, OneBitKernel};

fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
    BlockQ1_0G128 {
        d: f16::from_f32(scale),
        qs: bits,
    }
}

#[test]
fn test_dispatch_auto_detect() {
    let dispatcher = KernelDispatcher::auto_detect();
    let tier = dispatcher.tier();
    let name = dispatcher.name();

    // This whole test was unreachable under default
    // features until the E0004 fix just below (the file failed to compile
    // at all), so the literal expectations here were never actually run and
    // had gone stale against cli-16 (already landed in `oxibonsai-kernels`,
    // not owned here): `OneBitKernel::name()` (== `tier_label()`) is now
    // tier-only ("NEON (128-bit)", no quant-family prefix), because the
    // dispatcher is shared across multiple quant formats and hardcoding
    // "Q1_0_g128" into the generic tier label was misleading for the
    // others. The combined "<quant> <tier>" form these literals used to
    // check now lives in `KernelDispatcher::kernel_label`, asserted below.
    match tier {
        KernelTier::Reference => {
            assert_eq!(name, "reference (scalar)");
        }
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx2 => {
            assert_eq!(name, "AVX2+FMA (256-bit)");
        }
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx512 => {
            assert_eq!(name, "AVX-512 (512-bit)");
        }
        #[cfg(target_arch = "aarch64")]
        KernelTier::Neon => {
            assert_eq!(name, "NEON (128-bit)");
        }
        #[cfg(feature = "gpu")]
        KernelTier::Gpu => {
            assert!(name.contains("GPU") || name.contains("scirs2"));
        }
        // cli-09: this crate's `#[cfg(feature =
        // "gpu")]` arm above tracks THIS crate's OWN `gpu` feature, but
        // `KernelTier::Gpu` (`oxibonsai_kernels::tier::KernelTier`) is
        // gated on `oxibonsai-kernels`' OWN `gpu` feature — a dependency's
        // feature cannot be `cfg`'d on from a consumer crate, so the two
        // can disagree (they did: a reverted `[target.'cfg(target_os =
        // "macos")'.dependencies]` feature union once turned on
        // `oxibonsai-kernels/gpu` while leaving this crate's own `gpu`
        // feature off, making `KernelTier::Gpu` exist with no covering
        // arm here — `E0004` non-exhaustive patterns). This wildcard keeps
        // the match exhaustive under every feature combination this crate
        // can select, including any future mismatch of the same shape;
        // `#[allow]` is required because with the platform-gated union
        // gone, the arm above is unreachable under every combination this
        // crate can select today, and `-D warnings` would otherwise reject
        // that as dead code.
        #[allow(unreachable_patterns)]
        _ => {}
    }

    // cli-16: `kernel_label` recombines a resolved quant family with the
    // tier label, e.g. the exact "Q1_0_g128 NEON (128-bit)" shape this test
    // used to check via `name()` before that method became tier-only above.
    assert_eq!(
        dispatcher.kernel_label(GgufTensorType::Q1_0_g128),
        format!("Q1_0_g128 {name}"),
        "kernel_label should combine the quant family with the same tier label `name()` reports"
    );
}

#[test]
fn test_full_kernel_pipeline() {
    let dispatcher = KernelDispatcher::auto_detect();

    // Create blocks with known patterns
    let blocks = vec![
        make_block(2.0, [0xFF; 16]), // all +2.0
        make_block(1.5, [0x00; 16]), // all -1.5
    ];

    // Step 1: Dequantize
    let mut dequant_output = vec![0.0f32; 256];
    dispatcher
        .dequant(&blocks, &mut dequant_output)
        .expect("dequant should succeed");

    // Verify first block: all +2.0
    for &v in &dequant_output[..128] {
        assert!(
            (v - 2.0).abs() < 0.01,
            "expected +2.0 in first block, got {v}"
        );
    }
    // Verify second block: all -1.5
    for &v in &dequant_output[128..256] {
        assert!(
            (v + 1.5).abs() < 0.01,
            "expected -1.5 in second block, got {v}"
        );
    }

    // Step 2: GEMV with single row using the same 2-block weight
    let n_rows = 1;
    let k = 256;
    let input: Vec<f32> = vec![1.0; k];
    let mut gemv_output = vec![0.0f32; n_rows];
    dispatcher
        .gemv(&blocks, &input, &mut gemv_output, n_rows, k)
        .expect("gemv should succeed");

    // Manual: 128 * 2.0 + 128 * (-1.5) = 256 - 192 = 64
    let expected = 128.0 * 2.0 + 128.0 * (-1.5);
    assert!(
        (gemv_output[0] - expected).abs() < 1.0,
        "expected gemv ~{expected}, got {}",
        gemv_output[0]
    );
}

#[test]
fn test_dispatcher_gemv_matches_reference() {
    let n_rows = 16;
    let k = 512;
    let blocks_per_row = k / QK1_0_G128;
    let total_blocks = n_rows * blocks_per_row;

    let blocks: Vec<BlockQ1_0G128> = (0..total_blocks)
        .map(|i| {
            let bits: [u8; 16] = core::array::from_fn(|j| ((i * 31 + j * 11) & 0xFF) as u8);
            make_block(0.4 + (i as f32) * 0.01, bits)
        })
        .collect();

    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.005) - 1.28).collect();

    // Reference scalar
    let ref_dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
    let mut ref_output = vec![0.0f32; n_rows];
    ref_dispatcher
        .gemv(&blocks, &input, &mut ref_output, n_rows, k)
        .expect("reference gemv should succeed");

    // Auto-detect (may use SIMD)
    let auto_dispatcher = KernelDispatcher::auto_detect();
    let mut auto_output = vec![0.0f32; n_rows];
    auto_dispatcher
        .gemv(&blocks, &input, &mut auto_output, n_rows, k)
        .expect("auto gemv should succeed");

    for i in 0..n_rows {
        assert!(
            (ref_output[i] - auto_output[i]).abs() < 0.5,
            "row {i}: ref={}, auto={}",
            ref_output[i],
            auto_output[i]
        );
    }
}

#[test]
fn test_parallel_gemv_matches_sequential() {
    let n_rows = 128; // Above parallel threshold
    let k = 512;
    let blocks_per_row = k / QK1_0_G128;
    let total_blocks = n_rows * blocks_per_row;

    let blocks: Vec<BlockQ1_0G128> = (0..total_blocks)
        .map(|i| {
            let bits: [u8; 16] = core::array::from_fn(|j| ((i * 23 + j * 17) & 0xFF) as u8);
            make_block(0.5 + (i as f32) * 0.005, bits)
        })
        .collect();

    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.003) - 0.768).collect();

    let dispatcher = KernelDispatcher::auto_detect();

    // Sequential via dispatcher
    let mut seq_output = vec![0.0f32; n_rows];
    dispatcher
        .gemv(&blocks, &input, &mut seq_output, n_rows, k)
        .expect("sequential gemv should succeed");

    // Parallel
    let mut par_output = vec![0.0f32; n_rows];
    oxibonsai_kernels::parallel::gemv_1bit_g128_par(
        &dispatcher,
        &blocks,
        &input,
        &mut par_output,
        n_rows,
        k,
    )
    .expect("parallel gemv should succeed");

    for i in 0..n_rows {
        assert!(
            (seq_output[i] - par_output[i]).abs() < 0.01,
            "row {i}: seq={}, par={}",
            seq_output[i],
            par_output[i]
        );
    }
}

#[test]
fn test_model_linear_layer_forward() {
    // Test cross-crate usage: model layer using kernels dispatcher
    let n_out = 4;
    let n_in = 256;
    let blocks_per_row = n_in / QK1_0_G128;

    let blocks: Vec<BlockQ1_0G128> = (0..n_out * blocks_per_row)
        .map(|i| {
            let bits: [u8; 16] = core::array::from_fn(|j| ((i * 41 + j * 19) & 0xFF) as u8);
            make_block(1.0 + (i as f32) * 0.1, bits)
        })
        .collect();

    let dispatcher = KernelDispatcher::auto_detect();
    let input: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.01) - 1.28).collect();
    let mut output = vec![0.0f32; n_out];

    // Use GEMV directly (same as Linear1Bit::forward_vec)
    dispatcher
        .gemv(&blocks, &input, &mut output, n_out, n_in)
        .expect("linear forward should succeed");

    // All outputs should be finite and non-zero for non-trivial inputs
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}
