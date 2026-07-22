//! Cross-tier dispatch test for the `StandardQuantKernel` (Q4_0 / Q8_0) NEON
//! arms wired into `KernelDispatcher` in `dispatch.rs`.
//!
//! Prior to this wave, `KernelDispatcher::gemv_q4_0`/`gemv_q8_0` had no
//! `KernelTier::Neon` match arm, so on AArch64 (where `auto_detect()` always
//! selects `Neon`) every Q4_0/Q8_0 GEMV silently fell through to the scalar
//! reference kernel via the tier match's `_` arm — the NEON kernels in
//! `simd_q_std_neon.rs` were unreachable from the public dispatcher API.
//!
//! This test goes through the *public* `KernelDispatcher` entry point (not
//! the `simd_q_std_neon` functions directly, which `neon_q_std_parity.rs`
//! already covers) so it fails if the dispatch wiring regresses even if the
//! underlying NEON kernel itself is untouched.

#![cfg(target_arch = "aarch64")]

use oxibonsai_core::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_kernels::{KernelDispatcher, StandardQuantKernel};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn lcg_rand_f32(state: &mut u64, scale: f32) -> f32 {
    let x = lcg_next(state);
    (((x >> 11) as f32) / (1u64 << 53) as f32) * 2.0 * scale - scale
}

fn make_input(len: usize, rng: &mut u64, scale: f32) -> Vec<f32> {
    (0..len).map(|_| lcg_rand_f32(rng, scale)).collect()
}

fn assert_close(a: &[f32], b: &[f32], label: &str) {
    let rel_tol = 1e-4_f32;
    let abs_tol = 1e-4_f32;
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

const SHAPES: [(usize, usize); 5] = [(1, 32), (3, 64), (5, 96), (17, 128), (2, 160)];

/// `cpu_kernel_tier()` is the CPU-only tier selector the Q4_0/Q8_0 free
/// functions use to build their dispatcher (never the GPU tier, regardless of
/// whether the `gpu`/`metal` feature is enabled and a device is present — see
/// `dispatch.rs::cpu_kernel_tier`'s doc comment). NEON is mandatory on
/// AArch64, so this must always land on `Neon` on this architecture. This is
/// deliberately *not* `KernelDispatcher::auto_detect().tier()`, which prefers
/// the GPU tier when the `gpu` feature is enabled and a device is present —
/// that is expected and not what this invariant is about.
#[test]
fn cpu_kernel_tier_is_neon_on_aarch64() {
    assert_eq!(oxibonsai_kernels::cpu_kernel_tier(), KernelTier::Neon);
}

#[test]
fn dispatcher_gemv_q4_0_neon_matches_reference() {
    let mut rng = 0xAAAA_BBBB_CCCC_DDDD_u64;
    let reference = KernelDispatcher::with_tier(KernelTier::Reference);
    let neon = KernelDispatcher::with_tier(KernelTier::Neon);

    for (n_rows, in_features) in SHAPES {
        let blocks_per_row = in_features / QK_Q4_0;
        let raw = make_input(n_rows * blocks_per_row * QK_Q4_0, &mut rng, 5.0);
        let blocks = BlockQ4_0::quantize(&raw).expect("Q4_0 quantize should succeed");
        let input = make_input(in_features, &mut rng, 3.0);

        let mut ref_out = vec![0.0f32; n_rows];
        reference
            .gemv_q4_0(&blocks, &input, &mut ref_out, n_rows, in_features)
            .expect("reference q4_0 gemv should succeed");

        let mut neon_out = vec![0.0f32; n_rows];
        neon.gemv_q4_0(&blocks, &input, &mut neon_out, n_rows, in_features)
            .expect("neon-tier dispatcher q4_0 gemv should succeed");

        assert_close(
            &ref_out,
            &neon_out,
            &format!("dispatcher q4_0 gemv (n_rows={n_rows}, k={in_features})"),
        );
    }
}

#[test]
fn dispatcher_gemv_q8_0_neon_matches_reference() {
    let mut rng = 0x1234_5678_9ABC_DEF0_u64;
    let reference = KernelDispatcher::with_tier(KernelTier::Reference);
    let neon = KernelDispatcher::with_tier(KernelTier::Neon);

    for (n_rows, in_features) in SHAPES {
        let blocks_per_row = in_features / QK_Q8_0;
        let raw = make_input(n_rows * blocks_per_row * QK_Q8_0, &mut rng, 5.0);
        let blocks = BlockQ8_0::quantize(&raw).expect("Q8_0 quantize should succeed");
        let input = make_input(in_features, &mut rng, 3.0);

        let mut ref_out = vec![0.0f32; n_rows];
        reference
            .gemv_q8_0(&blocks, &input, &mut ref_out, n_rows, in_features)
            .expect("reference q8_0 gemv should succeed");

        let mut neon_out = vec![0.0f32; n_rows];
        neon.gemv_q8_0(&blocks, &input, &mut neon_out, n_rows, in_features)
            .expect("neon-tier dispatcher q8_0 gemv should succeed");

        assert_close(
            &ref_out,
            &neon_out,
            &format!("dispatcher q8_0 gemv (n_rows={n_rows}, k={in_features})"),
        );
    }
}

/// The real production entry point `oxibonsai_kernels::gemv_q4_0` (used by
/// every CPU-tier `Linear` layer forward call) must byte-for-byte match the
/// explicit `Neon` tier — proving the production hot path really is routing
/// through the new NEON dispatch arm, not silently falling through to
/// scalar via the tier match's `_` arm as it did before this wave.
#[test]
fn production_gemv_q4_0_matches_explicit_neon_tier() {
    let mut rng = 0x0F0F_1E1E_2D2D_3C3C_u64;
    let neon = KernelDispatcher::with_tier(KernelTier::Neon);

    let n_rows = 9;
    let in_features = 256;
    let blocks_per_row = in_features / QK_Q4_0;
    let raw = make_input(n_rows * blocks_per_row * QK_Q4_0, &mut rng, 4.0);
    let blocks = BlockQ4_0::quantize(&raw).expect("Q4_0 quantize should succeed");
    let input = make_input(in_features, &mut rng, 2.0);

    let mut prod_out = vec![0.0f32; n_rows];
    oxibonsai_kernels::gemv_q4_0(&blocks, &input, &mut prod_out, n_rows, in_features)
        .expect("production gemv_q4_0 should succeed");

    let mut neon_out = vec![0.0f32; n_rows];
    neon.gemv_q4_0(&blocks, &input, &mut neon_out, n_rows, in_features)
        .expect("explicit neon q4_0 gemv should succeed");

    assert_eq!(
        prod_out, neon_out,
        "production gemv_q4_0() must byte-for-byte match the explicit Neon tier (same code path)"
    );
}
