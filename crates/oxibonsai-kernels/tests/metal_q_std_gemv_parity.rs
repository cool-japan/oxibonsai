//! CPU-scalar vs. Metal-GPU parity for the standard-quant (`Q4_0` / `Q8_0`)
//! GEMV kernels.
//!
//! Runs only when built with `--features metal` on macOS. On a host without a
//! Metal device (CI runners) each test skips silently. The scalar reference
//! (`KernelDispatcher` at `Reference` tier) is the oracle; the dequant
//! arithmetic is bit-exact against it, so only the reduction order differs and
//! the results agree to within f32 rounding.

#![cfg(all(feature = "metal", target_os = "macos"))]

use oxibonsai_core::{BlockQ4_0, BlockQ8_0};
use oxibonsai_kernels::gpu_backend::{metal_gemv_q4_0, metal_gemv_q8_0};
use oxibonsai_kernels::{KernelDispatcher, KernelTier, StandardQuantKernel};

/// Deterministic, mixed-sign, positive-biased weight matrix (row-major).
///
/// The positive bias keeps the dot products comfortably non-zero so the tight
/// relative tolerance stays meaningful, while the sinusoidal variation exercises
/// the full quantization code range.
fn weights(n_rows: usize, in_features: usize) -> Vec<f32> {
    (0..n_rows * in_features)
        .map(|idx| {
            let r = (idx / in_features) as f32;
            let c = (idx % in_features) as f32;
            (r * 0.13 + c * 0.07).sin() * 2.0 + (c * 0.031).cos() * 1.0 + 1.2
        })
        .collect()
}

/// Deterministic all-positive input vector of length `in_features`.
fn input(in_features: usize) -> Vec<f32> {
    (0..in_features)
        .map(|i| 0.5 + 0.4 * ((i as f32) * 0.05).sin())
        .collect()
}

/// Assert per-row parity: bit-exact dequant means only reduction order differs,
/// so an absolute floor (f32 accumulation noise) OR a 1e-4 relative bound holds.
fn assert_parity(label: &str, n_rows: usize, in_features: usize, cpu: &[f32], gpu: &[f32]) {
    for row in 0..n_rows {
        let diff = (cpu[row] - gpu[row]).abs();
        let rel = diff / cpu[row].abs().max(1e-6);
        assert!(
            diff < 5e-3 || rel < 1e-4,
            "{label} parity failed: n_rows={n_rows} in_features={in_features} row={row} \
             cpu={} gpu={} diff={diff} rel={rel}",
            cpu[row],
            gpu[row]
        );
    }
}

/// `true` if a Metal device is present; used to skip gracefully on CI.
fn metal_skips(err: &oxibonsai_kernels::gpu_backend::MetalGraphError) -> bool {
    err.to_string().contains("no Metal-capable GPU device")
}

/// (n_rows, in_features) matrix: rows {1,7,32,33,256}; in_features covering
/// 1, 2, and 33 blocks (the last exercises the 32-lane stride tail).
const CASES: &[(usize, usize)] = &[
    (1, 32),
    (1, 1056),
    (7, 64),
    (7, 1056),
    (32, 96),
    (33, 64),
    (33, 1056),
    (256, 64),
];

#[test]
fn metal_gemv_q4_0_matches_scalar() {
    let disp = KernelDispatcher::with_tier(KernelTier::Reference);
    for &(n_rows, in_features) in CASES {
        let w = weights(n_rows, in_features);
        let blocks = BlockQ4_0::quantize(&w).expect("Q4_0 quantize");
        let inp = input(in_features);

        let mut cpu = vec![0.0f32; n_rows];
        disp.gemv_q4_0(&blocks, &inp, &mut cpu, n_rows, in_features)
            .expect("scalar Q4_0 GEMV");

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        let mut gpu = vec![0.0f32; n_rows];
        match metal_gemv_q4_0(bytes, &inp, &mut gpu, n_rows, in_features) {
            Ok(()) => {}
            Err(e) if metal_skips(&e) => return,
            Err(e) => panic!("metal Q4_0 GEMV failed: {e}"),
        }
        assert_parity("Q4_0", n_rows, in_features, &cpu, &gpu);
    }
}

#[test]
fn metal_gemv_q8_0_matches_scalar() {
    let disp = KernelDispatcher::with_tier(KernelTier::Reference);
    for &(n_rows, in_features) in CASES {
        let w = weights(n_rows, in_features);
        let blocks = BlockQ8_0::quantize(&w).expect("Q8_0 quantize");
        let inp = input(in_features);

        let mut cpu = vec![0.0f32; n_rows];
        disp.gemv_q8_0(&blocks, &inp, &mut cpu, n_rows, in_features)
            .expect("scalar Q8_0 GEMV");

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * std::mem::size_of::<BlockQ8_0>(),
            )
        };
        let mut gpu = vec![0.0f32; n_rows];
        match metal_gemv_q8_0(bytes, &inp, &mut gpu, n_rows, in_features) {
            Ok(()) => {}
            Err(e) if metal_skips(&e) => return,
            Err(e) => panic!("metal Q8_0 GEMV failed: {e}"),
        }
        assert_parity("Q8_0", n_rows, in_features, &cpu, &gpu);
    }
}

/// Shape-guard: `in_features` not a multiple of 32 is rejected by the host.
#[test]
fn metal_gemv_q4_0_rejects_unaligned_k() {
    let blocks = vec![0u8; std::mem::size_of::<BlockQ4_0>()];
    let inp = vec![0.0f32; 31];
    let mut out = vec![0.0f32; 1];
    let res = metal_gemv_q4_0(&blocks, &inp, &mut out, 1, 31);
    match res {
        Err(e) if metal_skips(&e) => {}
        Err(_) => {}
        Ok(()) => panic!("k=31 must be rejected"),
    }
}
