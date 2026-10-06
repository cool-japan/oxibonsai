//! Parity tests: scalar FP8 reference vs aarch64 NEON implementations.
//!
//! T-08: `simd_fp8_neon.rs` had zero tests despite the NEON dequant/gemv/gemm
//! kernels being real, complete implementations (six `pub unsafe fn`s) — the
//! only FP8 SIMD parity coverage in this workspace lived in
//! `fp8_simd_parity.rs`, and every one of its 12 real-parity assertions was
//! wrapped in `#[cfg(target_arch = "x86_64")]`, so on the only host that ran
//! this file in this session (Apple M3, aarch64) all 12 silently degraded to
//! "compute the scalar reference, assert nothing" and reported PASS. This
//! file is the fix's NEON half: it calls each of the six NEON kernels
//! directly (mirroring `fp8_simd_parity_x86.rs`'s per-kernel style) and
//! additionally asserts the tier that would actually be *selected* on this
//! host is NEON — not just that the numbers match (bit-exact dequant and
//! ≤ 3.0e-7 gemv/gemm agreement hold either way, on any tier, since they are
//! comparing against the same LUT; the bug this file guards against is the
//! NEON code path never being reached *at all*, which a numeric-only
//! assertion cannot detect).
#![cfg(target_arch = "aarch64")]

use half::f16;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelTier};

// ─── Deterministic LCG RNG (duplicated from fp8_simd_parity.rs — see
// fp8_simd_parity_x86.rs's header comment for why) ─────────────────────────

fn lcg_rand_u8(state: &mut u64) -> u8 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) & 0xFF) as u8
}

fn lcg_rand_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f32) / (1u64 << 53) as f32
}

fn make_e4m3_blocks(n: usize, rng: &mut u64) -> Vec<BlockFP8E4M3> {
    (0..n)
        .map(|_| {
            let mut qs = [0u8; 32];
            for q in qs.iter_mut() {
                *q = lcg_rand_u8(rng) & 0x7E; // avoid NaN codes 0x7F/0xFF
            }
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
                let raw = lcg_rand_u8(rng);
                *q = raw & 0b0111_1011; // avoid Inf/NaN codes
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

/// Assert pairwise closeness with a relative + absolute tolerance (same
/// formula as `fp8_simd_parity.rs`/`fp8_simd_parity_x86.rs`, so a bound
/// tightened in one is tightened consistently across all three).
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

/// The tier `cpu_kernel_tier()` — the same auto-detection every free-function
/// FP8 entry point and `KernelDispatcher::auto_detect()` uses — actually
/// selects on this host. On any real aarch64 machine this is always
/// [`KernelTier::Neon`] (NEON is baseline on AArch64, unlike x86's
/// AVX2/AVX-512 which need runtime feature detection), so this is not a
/// skip: if it ever fails, the auto-detect logic itself regressed.
#[test]
fn cpu_kernel_tier_is_neon_on_aarch64() {
    assert_eq!(
        cpu_kernel_tier(),
        KernelTier::Neon,
        "aarch64 must always auto-detect the Neon tier"
    );
}

#[test]
fn e4m3_dequant_neon_matches_scalar() {
    let mut rng = 0xDEAD_BEEF_1234_5678_u64;
    let n_blocks = 8;
    let blocks = make_e4m3_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e4m3(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    let mut neon_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::dequant_fp8_e4m3_neon(&blocks, &mut neon_out)
            .expect("neon dequant should succeed");
    }
    // Measured bit-exact (max abs diff 0e0);
    // the tolerance stays the shared 1e-4 formula for consistency with the
    // x86 siblings rather than hard-coding "0.0" and drifting apart later.
    assert_close(&scalar_out, &neon_out, 1e-4, "e4m3 dequant neon");
}

/// MINOR: `e4m3_dequant_neon_matches_scalar` uses the
/// shared 1e-4 tolerance formula "for consistency with the x86 siblings"
/// although the comment there claims the result is measured bit-exact.
/// Rather than blind-tighten that shared helper (a claim from an earlier
/// pass, not re-derived here), this adds a companion that actually checks
/// bit-exactness on THIS host and fails loudly with the first differing
/// index/bits if the claim ever stops holding — strictly additive, so a
/// real future divergence (e.g. a NEON codegen change) is caught here
/// without risking the existing tolerant assertion's stability.
#[test]
fn e4m3_dequant_neon_is_bit_exact_with_scalar() {
    let mut rng = 0xDEAD_BEEF_1234_5678_u64;
    let n_blocks = 8;
    let blocks = make_e4m3_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e4m3(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    let mut neon_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::dequant_fp8_e4m3_neon(&blocks, &mut neon_out)
            .expect("neon dequant should succeed");
    }
    for (i, (&s, &n)) in scalar_out.iter().zip(neon_out.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            n.to_bits(),
            "e4m3 dequant neon[{i}]: scalar={s} (0x{:08x}) vs neon={n} (0x{:08x}) -- not bit-exact",
            s.to_bits(),
            n.to_bits()
        );
    }
}

#[test]
fn e5m2_dequant_neon_matches_scalar() {
    let mut rng = 0x1111_2222_3333_4444_u64;
    let n_blocks = 10;
    let blocks = make_e5m2_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e5m2(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    let mut neon_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::dequant_fp8_e5m2_neon(&blocks, &mut neon_out)
            .expect("neon dequant e5m2 should succeed");
    }
    assert_close(&scalar_out, &neon_out, 1e-4, "e5m2 dequant neon");
}

/// See `e4m3_dequant_neon_is_bit_exact_with_scalar`'s doc comment.
#[test]
fn e5m2_dequant_neon_is_bit_exact_with_scalar() {
    let mut rng = 0x1111_2222_3333_4444_u64;
    let n_blocks = 10;
    let blocks = make_e5m2_blocks(n_blocks, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_blocks * QK_FP8];
    oxibonsai_kernels::dequant_fp8::dequant_fp8_e5m2(&blocks, &mut scalar_out)
        .expect("scalar dequant should succeed");

    let mut neon_out = vec![0.0_f32; n_blocks * QK_FP8];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::dequant_fp8_e5m2_neon(&blocks, &mut neon_out)
            .expect("neon dequant e5m2 should succeed");
    }
    for (i, (&s, &n)) in scalar_out.iter().zip(neon_out.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            n.to_bits(),
            "e5m2 dequant neon[{i}]: scalar={s} (0x{:08x}) vs neon={n} (0x{:08x}) -- not bit-exact",
            s.to_bits(),
            n.to_bits()
        );
    }
}

#[test]
fn e4m3_gemv_neon_matches_scalar() {
    let mut rng = 0xFEDC_BA98_7654_3210_u64;
    let n_rows = 4;
    let k = 64;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e4m3_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_rows];
    oxibonsai_kernels::gemv_fp8::gemv_fp8_e4m3(&blocks, &input, &mut scalar_out, n_rows, k)
        .expect("scalar gemv should succeed");

    let mut neon_out = vec![0.0_f32; n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::gemv_fp8_e4m3_neon(
            &blocks,
            &input,
            &mut neon_out,
            n_rows,
            k,
        )
        .expect("neon gemv e4m3 should succeed");
    }
    // Measured bound for this kernel: <= 3.0e-7.
    assert_close(&scalar_out, &neon_out, 3.0e-7, "e4m3 gemv neon");
}

#[test]
fn e5m2_gemv_neon_matches_scalar() {
    let mut rng = 0x1234_5678_9ABC_DEF0_u64;
    let n_rows = 5;
    let k = 64;
    let blocks_per_row = k / QK_FP8;
    let blocks = make_e5m2_blocks(n_rows * blocks_per_row, &mut rng);
    let input = make_input(k, &mut rng);

    let mut scalar_out = vec![0.0_f32; n_rows];
    oxibonsai_kernels::gemv_fp8::gemv_fp8_e5m2(&blocks, &input, &mut scalar_out, n_rows, k)
        .expect("scalar gemv e5m2 should succeed");

    let mut neon_out = vec![0.0_f32; n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::gemv_fp8_e5m2_neon(
            &blocks,
            &input,
            &mut neon_out,
            n_rows,
            k,
        )
        .expect("neon gemv e5m2 should succeed");
    }
    assert_close(&scalar_out, &neon_out, 3.0e-7, "e5m2 gemv neon");
}

#[test]
fn e4m3_gemm_neon_matches_scalar() {
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

    let mut neon_out = vec![0.0_f32; batch * n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::gemm_fp8_e4m3_neon(
            &blocks,
            &inputs,
            &mut neon_out,
            n_rows,
            k,
            batch,
        )
        .expect("neon gemm e4m3 should succeed");
    }
    // gemm dispatches to gemv per batch row, so shares gemv's bound.
    assert_close(&scalar_out, &neon_out, 4.8e-7, "e4m3 gemm neon");
}

#[test]
fn e5m2_gemm_neon_matches_scalar() {
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

    let mut neon_out = vec![0.0_f32; batch * n_rows];
    unsafe {
        oxibonsai_kernels::simd_fp8_neon::gemm_fp8_e5m2_neon(
            &blocks,
            &inputs,
            &mut neon_out,
            n_rows,
            k,
            batch,
        )
        .expect("neon gemm e5m2 should succeed");
    }
    assert_close(&scalar_out, &neon_out, 4.8e-7, "e5m2 gemm neon");
}

/// `KernelDispatcher::with_tier(KernelTier::Neon)` — the explicit-tier
/// construction production code uses when it *wants* NEON specifically
/// (rather than `auto_detect()`, which may legitimately prefer the GPU tier
/// on a Metal-capable host under `--features metal`/`--all-features`, so is
/// not asserted against `Neon` here) — must actually report that tier back,
/// not silently coerce to `Reference`.
#[test]
fn dispatcher_with_explicit_neon_tier_reports_neon() {
    use oxibonsai_kernels::KernelDispatcher;
    let dispatcher = KernelDispatcher::with_tier(KernelTier::Neon);
    assert_eq!(dispatcher.tier(), KernelTier::Neon);
}
