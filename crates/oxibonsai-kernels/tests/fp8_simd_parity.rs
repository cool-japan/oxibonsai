//! Cross-architecture FP8 parity tests: dispatcher auto-detect, LUT
//! spot-checks, and the parallel entry points — the six tests here run
//! (and assert something real) on every architecture, unlike the
//! arch-specific SIMD kernel parity tests.
//!
//! T-08: this file used to *also* hold 12 more tests, one per (format ×
//! operation × x86 ISA) combination, each wrapping its only real assertion
//! in `#[cfg(target_arch = "x86_64")]` inside an otherwise cross-arch
//! `#[test]` fn — so on aarch64 (this session's own dev machine, and every
//! other non-x86 host) all 12 silently ran just the scalar-reference half
//! and reported PASS with zero SIMD coverage, while `simd_fp8_neon.rs`'s six
//! real NEON kernels had no dedicated tests anywhere. Split into:
//! - `fp8_simd_parity_x86.rs` — the 12 x86-only tests, now behind a
//!   whole-file `#![cfg(target_arch = "x86_64")]` so a non-x86 host reports
//!   *fewer collected tests*, not phantom passes.
//! - `fp8_simd_parity_neon.rs` — six real NEON-vs-scalar parity tests plus a
//!   tier-selection assertion, so aarch64 finally has SIMD coverage.
//! - This file — the six tests that were already genuinely cross-arch
//!   (dispatcher auto-detect, LUT spot-checks, parallel entry points), kept
//!   in place as the verdict correction for T-08 specifies.

use half::f16;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};

// ─── Deterministic LCG RNG ────────────────────────────────────────────────

/// Knuth 64-bit LCG: produces uniformly distributed u64 values.
fn lcg_rand_u8(state: &mut u64) -> u8 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) & 0xFF) as u8
}

/// LCG-based f32 in [0, 1).
fn lcg_rand_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f32) / (1u64 << 53) as f32
}

// ─── Block generators ─────────────────────────────────────────────────────

fn make_e4m3_blocks(n: usize, rng: &mut u64) -> Vec<BlockFP8E4M3> {
    (0..n)
        .map(|_| {
            let mut qs = [0u8; 32];
            for q in qs.iter_mut() {
                // Avoid NaN codes: 0x7F (NaN) and 0xFF (NaN) — use 0x7E mask
                *q = lcg_rand_u8(rng) & 0x7E;
            }
            // Scale in [0.5, 2.5)
            let scale = 0.5 + lcg_rand_f32(rng) * 2.0;
            BlockFP8E4M3 {
                qs,
                d: f16::from_f32(scale),
            }
        })
        .collect()
}

fn make_e5m2_blocks(n: usize, rng: &mut u64) -> Vec<BlockFP8E5M2> {
    (0..n)
        .map(|_| {
            let mut qs = [0u8; 32];
            for q in qs.iter_mut() {
                // Avoid Inf/NaN codes (0x7C, 0xFC, 0x7E, 0xFF etc.)
                // Keep exponent field ≤ 0b11110 = 0x3C range max
                let raw = lcg_rand_u8(rng);
                // Clear top 2 bits of exponent to stay away from Inf/NaN
                *q = raw & 0b0111_1011;
            }
            let scale = 0.5 + lcg_rand_f32(rng) * 2.0;
            BlockFP8E5M2 {
                qs,
                d: f16::from_f32(scale),
            }
        })
        .collect()
}

fn make_input(len: usize, rng: &mut u64) -> Vec<f32> {
    (0..len).map(|_| (lcg_rand_f32(rng) - 0.5) * 4.0).collect()
}

// ─── Parity assertion helpers ─────────────────────────────────────────────

/// Assert pairwise closeness with a relative + absolute tolerance.
///
/// Passes when `|a - b| <= abs_tol + rel_tol * max(|a|, |b|)`.
/// This handles both near-zero values (where absolute tolerance dominates)
/// and large values (where FMA vs scalar rounding order causes proportional error).
fn assert_close(a: &[f32], b: &[f32], abs_tol: f32, label: &str) {
    // relative tolerance: 1 ULP of f32 mantissa accuracy ≈ 2e-7;
    // allow a small multiple for accumulated FMA vs scalar order differences.
    let rel_tol = 1e-4_f32;
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(1.0);
        let tol = abs_tol + rel_tol * scale;
        assert!(
            diff <= tol,
            "{label}[{i}]: |{va} - {vb}| = {diff} > {tol} (abs={abs_tol} + rel={rel_tol}×{scale})"
        );
    }
}

// ─── Dispatcher round-trip parity (uses auto-detect tier) ─────────────────

#[test]
fn dispatcher_fp8_e4m3_gemv_matches_scalar() {
    use oxibonsai_kernels::{traits::Fp8Kernel, KernelDispatcher};

    let mut rng = 0x0102_0304_0506_0708_u64;
    let n_rows = 8;
    let k = 64;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e4m3_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_rows];
    oxibonsai_kernels::gemv_fp8::gemv_fp8_e4m3(&blocks, &input, &mut scalar_out, n_rows, k)
        .expect("scalar gemv should succeed");

    let dispatcher = KernelDispatcher::auto_detect();
    let mut disp_out = vec![0.0_f32; n_rows];
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut disp_out, n_rows, k)
        .expect("dispatcher gemv should succeed");

    assert_close(&scalar_out, &disp_out, 1e-4, "dispatcher e4m3 gemv");
}

#[test]
fn dispatcher_fp8_e5m2_gemm_matches_scalar() {
    use oxibonsai_kernels::{traits::Fp8Kernel, KernelDispatcher};

    let mut rng = 0xF0F0_F0F0_0F0F_0F0F_u64;
    let n_rows = 3;
    let k = 96;
    let batch = 4;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e5m2_blocks(n_rows * blocks_per_row, &mut rng);
    let inputs = make_input(batch * k, &mut rng);

    let mut scalar_out = vec![0.0_f32; batch * n_rows];
    oxibonsai_kernels::gemm_fp8::gemm_fp8_e5m2(&blocks, &inputs, &mut scalar_out, n_rows, k, batch)
        .expect("scalar gemm should succeed");

    let dispatcher = KernelDispatcher::auto_detect();
    let mut disp_out = vec![0.0_f32; batch * n_rows];
    dispatcher
        .gemm_fp8_e5m2(&blocks, &inputs, &mut disp_out, n_rows, k, batch)
        .expect("dispatcher gemm should succeed");

    assert_close(&scalar_out, &disp_out, 1e-4, "dispatcher e5m2 gemm");
}

// ─── LUT correctness spot-checks ─────────────────────────────────────────

#[test]
fn lut_e4m3_spot_check_byte_0x38() {
    // 0x38 = sign=0, exp=7, man=0 → 2^(7-7) × (1 + 0/8) = 1.0
    let lut = oxibonsai_kernels::fp8_lut::fp8_e4m3_lut();
    assert!(
        (lut[0x38] - 1.0).abs() < 1e-5,
        "byte 0x38 should decode to ~1.0, got {}",
        lut[0x38]
    );
}

#[test]
fn lut_e5m2_spot_check_byte_0x3c() {
    // 0x3C = sign=0, exp=15, man=0 → 2^(15-15) = 1.0
    let lut = oxibonsai_kernels::fp8_lut::fp8_e5m2_lut();
    assert!(
        (lut[0x3C] - 1.0).abs() < 1e-5,
        "byte 0x3C should decode to ~1.0, got {}",
        lut[0x3C]
    );
}

// ─── Parallel FP8 entry-point parity ─────────────────────────────────────

#[test]
fn par_fp8_e4m3_gemv_matches_sequential() {
    use oxibonsai_kernels::{gemv_fp8_e4m3_par, KernelDispatcher};

    let mut rng = 0x1357_2468_9BDF_0ACE_u64;
    let n_rows = 128; // above PAR_GEMV_MIN_ROWS
    let k = 64;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e4m3_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let dispatcher = KernelDispatcher::auto_detect();

    let mut seq_out = vec![0.0_f32; n_rows];
    let mut par_out = vec![0.0_f32; n_rows];

    use oxibonsai_kernels::traits::Fp8Kernel;
    dispatcher
        .gemv_fp8_e4m3(&blocks, &input, &mut seq_out, n_rows, k)
        .expect("sequential gemv should succeed");

    gemv_fp8_e4m3_par(&dispatcher, &blocks, &input, &mut par_out, n_rows, k)
        .expect("parallel gemv should succeed");

    assert_close(&seq_out, &par_out, 1e-4, "par e4m3 gemv");
}

#[test]
fn par_fp8_e5m2_gemm_matches_sequential() {
    use oxibonsai_kernels::{gemm_fp8_e5m2_par, KernelDispatcher};

    let mut rng = 0xECEB_EDED_EFEF_FAFA_u64;
    let n_rows = 4;
    let k = 64;
    let batch = 8; // above PAR_GEMM_MIN_BATCH
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e5m2_blocks(n_rows * blocks_per_row, &mut rng);
    let inputs = make_input(batch * k, &mut rng);

    let dispatcher = KernelDispatcher::auto_detect();

    let mut seq_out = vec![0.0_f32; batch * n_rows];
    let mut par_out = vec![0.0_f32; batch * n_rows];

    use oxibonsai_kernels::traits::Fp8Kernel;
    dispatcher
        .gemm_fp8_e5m2(&blocks, &inputs, &mut seq_out, n_rows, k, batch)
        .expect("sequential gemm should succeed");

    gemm_fp8_e5m2_par(
        &dispatcher,
        &blocks,
        &inputs,
        &mut par_out,
        n_rows,
        k,
        batch,
    )
    .expect("parallel gemm should succeed");

    assert_close(&seq_out, &par_out, 1e-4, "par e5m2 gemm");
}
