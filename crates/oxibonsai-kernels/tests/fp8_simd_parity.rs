//! Parity tests: scalar FP8 reference vs SIMD implementations.
//!
//! For each format (E4M3, E5M2) and each operation (dequant, gemv, gemm),
//! we generate random blocks, run both the scalar reference and the SIMD
//! path, then assert the outputs match within a tight tolerance (≤ 1e-4).
//!
//! SIMD test bodies are gated on runtime feature detection so these tests
//! pass correctly on machines that lack AVX2, AVX-512, or NEON.

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
    (0..len)
        .map(|_| (lcg_rand_f32(rng) - 0.5) * 4.0)
        .collect()
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

// ─── E4M3 dequant parity ─────────────────────────────────────────────────

#[test]
fn e4m3_dequant_avx2_matches_scalar() {
    let mut rng = 0xDEAD_BEEF_1234_5678_u64;
    let n_blocks = 8;
    let blocks = make_e4m3_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e4m3(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    #[cfg(target_arch = "x86_64")]
    {
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
}

#[test]
fn e4m3_dequant_avx512_matches_scalar() {
    let mut rng = 0xABCD_EF01_2345_6789_u64;
    let n_blocks = 6;
    let blocks = make_e4m3_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e4m3(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    #[cfg(target_arch = "x86_64")]
    {
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

    #[cfg(target_arch = "x86_64")]
    {
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
}

#[test]
fn e5m2_dequant_avx512_matches_scalar() {
    let mut rng = 0x5555_6666_7777_8888_u64;
    let n_blocks = 4;
    let blocks = make_e5m2_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e5m2(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    #[cfg(target_arch = "x86_64")]
    {
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut avx2_out = vec![0.0_f32; n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx2::gemv_fp8_e4m3_avx2(
                &blocks, &input, &mut avx2_out, n_rows, k,
            )
            .expect("avx2 gemv e4m3 should succeed");
        }
        assert_close(&scalar_out, &avx2_out, 1e-4, "e4m3 gemv avx2");
    }
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return;
        }
        let mut avx512_out = vec![0.0_f32; n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx512::gemv_fp8_e4m3_avx512(
                &blocks, &input, &mut avx512_out, n_rows, k,
            )
            .expect("avx512 gemv e4m3 should succeed");
        }
        assert_close(&scalar_out, &avx512_out, 1e-4, "e4m3 gemv avx512");
    }
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut avx2_out = vec![0.0_f32; n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx2::gemv_fp8_e5m2_avx2(
                &blocks, &input, &mut avx2_out, n_rows, k,
            )
            .expect("avx2 gemv e5m2 should succeed");
        }
        assert_close(&scalar_out, &avx2_out, 1e-4, "e5m2 gemv avx2");
    }
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return;
        }
        let mut avx512_out = vec![0.0_f32; n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx512::gemv_fp8_e5m2_avx512(
                &blocks, &input, &mut avx512_out, n_rows, k,
            )
            .expect("avx512 gemv e5m2 should succeed");
        }
        assert_close(&scalar_out, &avx512_out, 1e-4, "e5m2 gemv avx512");
    }
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut avx2_out = vec![0.0_f32; batch * n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx2::gemm_fp8_e4m3_avx2(
                &blocks, &inputs, &mut avx2_out, n_rows, k, batch,
            )
            .expect("avx2 gemm e4m3 should succeed");
        }
        assert_close(&scalar_out, &avx2_out, 1e-4, "e4m3 gemm avx2");
    }
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return;
        }
        let mut avx512_out = vec![0.0_f32; batch * n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx512::gemm_fp8_e4m3_avx512(
                &blocks, &inputs, &mut avx512_out, n_rows, k, batch,
            )
            .expect("avx512 gemm e4m3 should succeed");
        }
        assert_close(&scalar_out, &avx512_out, 1e-4, "e4m3 gemm avx512");
    }
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut avx2_out = vec![0.0_f32; batch * n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx2::gemm_fp8_e5m2_avx2(
                &blocks, &inputs, &mut avx2_out, n_rows, k, batch,
            )
            .expect("avx2 gemm e5m2 should succeed");
        }
        assert_close(&scalar_out, &avx2_out, 1e-4, "e5m2 gemm avx2");
    }
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

    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            return;
        }
        let mut avx512_out = vec![0.0_f32; batch * n_rows];
        unsafe {
            oxibonsai_kernels::simd_fp8_avx512::gemm_fp8_e5m2_avx512(
                &blocks, &inputs, &mut avx512_out, n_rows, k, batch,
            )
            .expect("avx512 gemm e5m2 should succeed");
        }
        assert_close(&scalar_out, &avx512_out, 1e-4, "e5m2 gemm avx512");
    }
}

// ─── Dispatcher round-trip parity (uses auto-detect tier) ─────────────────

#[test]
fn dispatcher_fp8_e4m3_gemv_matches_scalar() {
    use oxibonsai_kernels::{KernelDispatcher, traits::Fp8Kernel};

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
    use oxibonsai_kernels::{KernelDispatcher, traits::Fp8Kernel};

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
    use oxibonsai_kernels::{KernelDispatcher, gemv_fp8_e4m3_par};

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
    use oxibonsai_kernels::{KernelDispatcher, gemm_fp8_e5m2_par};

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

    gemm_fp8_e5m2_par(&dispatcher, &blocks, &inputs, &mut par_out, n_rows, k, batch)
        .expect("parallel gemm should succeed");

    assert_close(&seq_out, &par_out, 1e-4, "par e5m2 gemm");
}
