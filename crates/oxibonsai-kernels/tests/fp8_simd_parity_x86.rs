//! Parity tests: scalar FP8 reference vs x86-64 AVX2/AVX-512 implementations.
//!
//! T-08: split out of `fp8_simd_parity.rs`, which used to wrap each of these
//! 12 tests' *only* real assertion in `#[cfg(target_arch = "x86_64")]`
//! inside an otherwise cross-arch `#[test]` fn — so on aarch64 (this
//! session's own dev machine) every one of them silently ran just the
//! scalar-reference half and reported PASS with zero parity coverage. A
//! whole-file `#![cfg(target_arch = "x86_64")]` makes that honest: on any
//! other architecture these 12 tests do not exist at all (cargo reports
//! fewer collected tests, not 12 phantom passes). See `fp8_simd_parity.rs`
//! for the six tests that stayed cross-arch, and `fp8_simd_parity_neon.rs`
//! for their aarch64 sibling (the actual fix for the "simd_fp8_neon.rs has
//! zero tests" half of T-08).
#![cfg(target_arch = "x86_64")]

use half::f16;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};

// ─── Deterministic LCG RNG (duplicated from fp8_simd_parity.rs: each of
// these helpers is a few lines, so the coupling cost of sharing them across
// an arch boundary outweighs the duplication cost) ───────────────────────

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

// ─── Parity assertion helper ──────────────────────────────────────────────

/// Assert pairwise closeness with a relative + absolute tolerance.
///
/// Passes when `|a - b| <= abs_tol + rel_tol * max(|a|, |b|)`.
fn assert_close(a: &[f32], b: &[f32], abs_tol: f32, label: &str) {
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

// ─── E4M3 dequant parity ─────────────────────────────────────────────────

#[test]
fn e4m3_dequant_avx2_matches_scalar() {
    let mut rng = 0xDEAD_BEEF_1234_5678_u64;
    let n_blocks = 8;
    let blocks = make_e4m3_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e4m3(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut avx2_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx2::dequant_fp8_e4m3_avx2(&blocks, &mut avx2_out)
            .expect("avx2 dequant should succeed");
    }
    assert_close(&scalar_out, &avx2_out, 1e-4, "e4m3 dequant avx2");
}

#[test]
fn e4m3_dequant_avx512_matches_scalar() {
    let mut rng = 0xABCD_EF01_2345_6789_u64;
    let n_blocks = 6;
    let blocks = make_e4m3_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e4m3(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("avx512vl")
    {
        return;
    }
    let mut avx512_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx512::dequant_fp8_e4m3_avx512(&blocks, &mut avx512_out)
            .expect("avx512 dequant should succeed");
    }
    assert_close(&scalar_out, &avx512_out, 1e-4, "e4m3 dequant avx512");
}

// ─── E5M2 dequant parity ─────────────────────────────────────────────────

#[test]
fn e5m2_dequant_avx2_matches_scalar() {
    let mut rng = 0x1111_2222_3333_4444_u64;
    let n_blocks = 10;
    let blocks = make_e5m2_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e5m2(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut avx2_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx2::dequant_fp8_e5m2_avx2(&blocks, &mut avx2_out)
            .expect("avx2 dequant e5m2 should succeed");
    }
    assert_close(&scalar_out, &avx2_out, 1e-4, "e5m2 dequant avx2");
}

#[test]
fn e5m2_dequant_avx512_matches_scalar() {
    let mut rng = 0x5555_6666_7777_8888_u64;
    let n_blocks = 4;
    let blocks = make_e5m2_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e5m2(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("avx512vl")
    {
        return;
    }
    let mut avx512_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx512::dequant_fp8_e5m2_avx512(&blocks, &mut avx512_out)
            .expect("avx512 dequant e5m2 should succeed");
    }
    assert_close(&scalar_out, &avx512_out, 1e-4, "e5m2 dequant avx512");
}

// ─── E4M3 GEMV parity ────────────────────────────────────────────────────

#[test]
fn e4m3_gemv_avx2_matches_scalar() {
    let mut rng = 0xFEDC_BA98_7654_3210_u64;
    let n_rows = 4;
    let k = 64; // 2 blocks per row
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e4m3_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_rows];
    oxibonsai_kernels::gemv_fp8::gemv_fp8_e4m3(&blocks, &input, &mut scalar_out, n_rows, k)
        .expect("scalar gemv should succeed");

    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut avx2_out = vec![0.0_f32; n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx2::gemv_fp8_e4m3_avx2(
            &blocks,
            &input,
            &mut avx2_out,
            n_rows,
            k,
        )
        .expect("avx2 gemv e4m3 should succeed");
    }
    assert_close(&scalar_out, &avx2_out, 1e-4, "e4m3 gemv avx2");
}

#[test]
fn e4m3_gemv_avx512_matches_scalar() {
    let mut rng = 0xCAFE_BABE_DEAD_BEEF_u64;
    let n_rows = 3;
    let k = 96; // 3 blocks per row
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e4m3_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_rows];
    oxibonsai_kernels::gemv_fp8::gemv_fp8_e4m3(&blocks, &input, &mut scalar_out, n_rows, k)
        .expect("scalar gemv should succeed");

    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("avx512vl")
    {
        return;
    }
    let mut avx512_out = vec![0.0_f32; n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx512::gemv_fp8_e4m3_avx512(
            &blocks,
            &input,
            &mut avx512_out,
            n_rows,
            k,
        )
        .expect("avx512 gemv e4m3 should succeed");
    }
    assert_close(&scalar_out, &avx512_out, 1e-4, "e4m3 gemv avx512");
}

// ─── E5M2 GEMV parity ────────────────────────────────────────────────────

#[test]
fn e5m2_gemv_avx2_matches_scalar() {
    let mut rng = 0x1234_5678_9ABC_DEF0_u64;
    let n_rows = 5;
    let k = 64;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e5m2_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_rows];
    oxibonsai_kernels::gemv_fp8::gemv_fp8_e5m2(&blocks, &input, &mut scalar_out, n_rows, k)
        .expect("scalar gemv e5m2 should succeed");

    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut avx2_out = vec![0.0_f32; n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx2::gemv_fp8_e5m2_avx2(
            &blocks,
            &input,
            &mut avx2_out,
            n_rows,
            k,
        )
        .expect("avx2 gemv e5m2 should succeed");
    }
    assert_close(&scalar_out, &avx2_out, 1e-4, "e5m2 gemv avx2");
}

#[test]
fn e5m2_gemv_avx512_matches_scalar() {
    let mut rng = 0x0F0F_0F0F_F0F0_F0F0_u64;
    let n_rows = 2;
    let k = 128;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e5m2_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_rows];
    oxibonsai_kernels::gemv_fp8::gemv_fp8_e5m2(&blocks, &input, &mut scalar_out, n_rows, k)
        .expect("scalar gemv e5m2 should succeed");

    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("avx512vl")
    {
        return;
    }
    let mut avx512_out = vec![0.0_f32; n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx512::gemv_fp8_e5m2_avx512(
            &blocks,
            &input,
            &mut avx512_out,
            n_rows,
            k,
        )
        .expect("avx512 gemv e5m2 should succeed");
    }
    assert_close(&scalar_out, &avx512_out, 1e-4, "e5m2 gemv avx512");
}

// ─── E4M3 GEMM parity ────────────────────────────────────────────────────

#[test]
fn e4m3_gemm_avx2_matches_scalar() {
    let mut rng = 0xAAAA_BBBB_CCCC_DDDD_u64;
    let n_rows = 3;
    let k = 64;
    let batch = 4;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e4m3_blocks(n_rows * blocks_per_row, &mut rng);
    let inputs = make_input(batch * k, &mut rng);

    let mut scalar_out = vec![0.0_f32; batch * n_rows];
    oxibonsai_kernels::gemm_fp8::gemm_fp8_e4m3(&blocks, &inputs, &mut scalar_out, n_rows, k, batch)
        .expect("scalar gemm e4m3 should succeed");

    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut avx2_out = vec![0.0_f32; batch * n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx2::gemm_fp8_e4m3_avx2(
            &blocks,
            &inputs,
            &mut avx2_out,
            n_rows,
            k,
            batch,
        )
        .expect("avx2 gemm e4m3 should succeed");
    }
    assert_close(&scalar_out, &avx2_out, 1e-4, "e4m3 gemm avx2");
}

#[test]
fn e4m3_gemm_avx512_matches_scalar() {
    let mut rng = 0x9999_8888_7777_6666_u64;
    let n_rows = 4;
    let k = 32;
    let batch = 3;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e4m3_blocks(n_rows * blocks_per_row, &mut rng);
    let inputs = make_input(batch * k, &mut rng);

    let mut scalar_out = vec![0.0_f32; batch * n_rows];
    oxibonsai_kernels::gemm_fp8::gemm_fp8_e4m3(&blocks, &inputs, &mut scalar_out, n_rows, k, batch)
        .expect("scalar gemm e4m3 should succeed");

    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("avx512vl")
    {
        return;
    }
    let mut avx512_out = vec![0.0_f32; batch * n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx512::gemm_fp8_e4m3_avx512(
            &blocks,
            &inputs,
            &mut avx512_out,
            n_rows,
            k,
            batch,
        )
        .expect("avx512 gemm e4m3 should succeed");
    }
    assert_close(&scalar_out, &avx512_out, 1e-4, "e4m3 gemm avx512");
}

// ─── E5M2 GEMM parity ────────────────────────────────────────────────────

#[test]
fn e5m2_gemm_avx2_matches_scalar() {
    let mut rng = 0xBEEF_CAFE_1234_5678_u64;
    let n_rows = 2;
    let k = 96;
    let batch = 5;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e5m2_blocks(n_rows * blocks_per_row, &mut rng);
    let inputs = make_input(batch * k, &mut rng);

    let mut scalar_out = vec![0.0_f32; batch * n_rows];
    oxibonsai_kernels::gemm_fp8::gemm_fp8_e5m2(&blocks, &inputs, &mut scalar_out, n_rows, k, batch)
        .expect("scalar gemm e5m2 should succeed");

    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut avx2_out = vec![0.0_f32; batch * n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx2::gemm_fp8_e5m2_avx2(
            &blocks,
            &inputs,
            &mut avx2_out,
            n_rows,
            k,
            batch,
        )
        .expect("avx2 gemm e5m2 should succeed");
    }
    assert_close(&scalar_out, &avx2_out, 1e-4, "e5m2 gemm avx2");
}

#[test]
fn e5m2_gemm_avx512_matches_scalar() {
    let mut rng = 0xDEAD_C0DE_ABCD_EF01_u64;
    let n_rows = 3;
    let k = 64;
    let batch = 2;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e5m2_blocks(n_rows * blocks_per_row, &mut rng);
    let inputs = make_input(batch * k, &mut rng);

    let mut scalar_out = vec![0.0_f32; batch * n_rows];
    oxibonsai_kernels::gemm_fp8::gemm_fp8_e5m2(&blocks, &inputs, &mut scalar_out, n_rows, k, batch)
        .expect("scalar gemm e5m2 should succeed");

    if !is_x86_feature_detected!("avx512f")
        || !is_x86_feature_detected!("avx512bw")
        || !is_x86_feature_detected!("avx512vl")
    {
        return;
    }
    let mut avx512_out = vec![0.0_f32; batch * n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_avx512::gemm_fp8_e5m2_avx512(
            &blocks,
            &inputs,
            &mut avx512_out,
            n_rows,
            k,
            batch,
        )
        .expect("avx512 gemm e5m2 should succeed");
    }
    assert_close(&scalar_out, &avx512_out, 1e-4, "e5m2 gemm avx512");
}
