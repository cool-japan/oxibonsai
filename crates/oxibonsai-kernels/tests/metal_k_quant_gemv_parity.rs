//! CPU-scalar vs. Metal-GPU parity for the six K-quant GEMV kernels
//! (`Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_K`).
//!
//! Runs only when built with `--features metal` on macOS. On a host without a
//! Metal device (CI runners) each test skips silently. The scalar free-function
//! kernels (`gemv_q2k` … `gemv_q8k`) are the oracle; the dequant arithmetic is
//! bit-exact against them, so only the reduction order differs and the results
//! agree to within f32 rounding.

#![cfg(all(feature = "metal", target_os = "macos"))]

use oxibonsai_core::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K};
use oxibonsai_kernels::gpu_backend::{
    metal_gemv_q2k, metal_gemv_q3k, metal_gemv_q4k, metal_gemv_q5k, metal_gemv_q6k, metal_gemv_q8k,
    MetalGraphError,
};
use oxibonsai_kernels::{gemv_q2k, gemv_q3k, gemv_q4k, gemv_q5k, gemv_q6k, gemv_q8k};
use oxibonsai_testkit::capability::{record as record_capability, Capability};

/// Deterministic, mixed-sign, positive-biased weight matrix (row-major).
fn weights(n_rows: usize, in_features: usize) -> Vec<f32> {
    (0..n_rows * in_features)
        .map(|idx| {
            let r = (idx / in_features) as f32;
            let c = (idx % in_features) as f32;
            (r * 0.11 + c * 0.017).sin() * 2.0 + (c * 0.0043).cos() * 1.0 + 1.1
        })
        .collect()
}

/// Deterministic all-positive input vector of length `in_features`.
fn input(in_features: usize) -> Vec<f32> {
    (0..in_features)
        .map(|i| 0.5 + 0.4 * ((i as f32) * 0.03).sin())
        .collect()
}

/// Per-row parity check.
///
/// Wave-1.5 addendum (1) / gatekeeper OPTIONAL #O5: this used to be
/// `diff < 5e-3 || rel < 1e-4` — an OR against an absolute floor that let
/// *any* row whose absolute diff happened to be under 5e-3 pass regardless
/// of its relative error, which is exactly what hid a real stale-layout bug
/// (Q6_K at rel=0.0119, Q2_K at rel=0.0013) while Q3_K/Q4_K/Q5_K were
/// visibly failing at rel 0.93/0.65/0.64. Now that the K-quant GPU kernels
/// are ggml-exact (FIX-05-KQUANT-GPU) and bit-exact dequant means only
/// reduction order differs, every row is held to a real relative bound; a
/// row whose CPU reference is genuinely near zero (where "relative" is
/// ill-defined) instead uses a much smaller absolute floor — never an OR
/// across the whole population.
fn assert_parity(label: &str, n_rows: usize, in_features: usize, cpu: &[f32], gpu: &[f32]) {
    const NEAR_ZERO: f32 = 1e-4;
    const REL_BOUND: f32 = 1e-4;
    const ABS_FLOOR_NEAR_ZERO: f32 = 5e-5;
    for row in 0..n_rows {
        let diff = (cpu[row] - gpu[row]).abs();
        let (ok, rel) = if cpu[row].abs() < NEAR_ZERO {
            (diff < ABS_FLOOR_NEAR_ZERO, f32::NAN)
        } else {
            let rel = diff / cpu[row].abs();
            (rel < REL_BOUND, rel)
        };
        assert!(
            ok,
            "{label} parity failed: n_rows={n_rows} in_features={in_features} row={row} \
             cpu={} gpu={} diff={diff} rel={rel}",
            cpu[row], gpu[row]
        );
    }
}

fn metal_skips(err: &MetalGraphError) -> bool {
    err.to_string().contains("no Metal-capable GPU device")
}

// ── T-05: hardware-capability self-skip reporting ───────────────────────────
//
// This test file used to self-skip to green on a host with no Metal device:
// `Err(e) if metal_skips(&e) => return` looks identical to a passing test
// from `cargo nextest`'s point of view. `record_capability` (imported above
// as `oxibonsai_testkit::capability::record`) appends a JSONL line to the
// capability manifest `scripts/release-gate.sh` enforces (see that script's
// header for the full contract), so a skip is now visibly distinct from a
// real, executed parity check.
//
// T-07 FIX (verifier wave 3): this used to be a byte-for-byte inline copy of
// `oxibonsai_testkit::capability::record`; `oxibonsai-kernels` now takes
// `oxibonsai-testkit` as a dev-dependency, so the copy is deleted in favour
// of the shared implementation (aliased to the old local name above so every
// call site below is unchanged).

/// (n_rows, in_features) matrix: rows {1,7,32,33,256}; in_features covering
/// 1, 2, and 3 super-blocks (256/512/768).
const KQ_CASES: &[(usize, usize)] = &[
    (1, 256),
    (1, 768),
    (7, 512),
    (32, 256),
    (33, 768),
    (256, 512),
];

macro_rules! kq_parity_test {
    ($test_name:ident, $label:literal, $blk:ty, $cpu:path, $metal:path) => {
        #[test]
        fn $test_name() {
            for &(n_rows, in_features) in KQ_CASES {
                let w = weights(n_rows, in_features);
                let blocks = <$blk>::quantize(&w).expect("quantize");
                let inp = input(in_features);

                let mut cpu = vec![0.0f32; n_rows];
                $cpu(&blocks, &inp, &mut cpu, n_rows, in_features).expect("scalar K-quant GEMV");

                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr().cast::<u8>(),
                        blocks.len() * std::mem::size_of::<$blk>(),
                    )
                };
                let mut gpu = vec![0.0f32; n_rows];
                match $metal(bytes, &inp, &mut gpu, n_rows, in_features) {
                    Ok(()) => {}
                    Err(e) if metal_skips(&e) => {
                        record_capability(
                            Capability::Metal,
                            false,
                            concat!(
                                "oxibonsai-kernels::metal_k_quant_gemv_parity::",
                                stringify!($test_name)
                            ),
                        );
                        return;
                    }
                    Err(e) => panic!("{} metal GEMV failed: {e}", $label),
                }
                record_capability(
                    Capability::Metal,
                    true,
                    concat!(
                        "oxibonsai-kernels::metal_k_quant_gemv_parity::",
                        stringify!($test_name)
                    ),
                );
                assert_parity($label, n_rows, in_features, &cpu, &gpu);
            }
        }
    };
}

kq_parity_test!(
    metal_gemv_q2k_matches_scalar,
    "Q2_K",
    BlockQ2K,
    gemv_q2k,
    metal_gemv_q2k
);
kq_parity_test!(
    metal_gemv_q3k_matches_scalar,
    "Q3_K",
    BlockQ3K,
    gemv_q3k,
    metal_gemv_q3k
);
kq_parity_test!(
    metal_gemv_q4k_matches_scalar,
    "Q4_K",
    BlockQ4K,
    gemv_q4k,
    metal_gemv_q4k
);
kq_parity_test!(
    metal_gemv_q5k_matches_scalar,
    "Q5_K",
    BlockQ5K,
    gemv_q5k,
    metal_gemv_q5k
);
kq_parity_test!(
    metal_gemv_q6k_matches_scalar,
    "Q6_K",
    BlockQ6K,
    gemv_q6k,
    metal_gemv_q6k
);
kq_parity_test!(
    metal_gemv_q8k_matches_scalar,
    "Q8_K",
    BlockQ8K,
    gemv_q8k,
    metal_gemv_q8k
);

/// Shape-guard: `k` not a multiple of 256 is rejected by the host.
#[test]
fn metal_gemv_q2k_rejects_unaligned_k() {
    let blocks = vec![0u8; std::mem::size_of::<BlockQ2K>()];
    let inp = vec![0.0f32; 255];
    let mut out = vec![0.0f32; 1];
    match metal_gemv_q2k(&blocks, &inp, &mut out, 1, 255) {
        Err(_) => {}
        Ok(()) => panic!("k=255 must be rejected"),
    }
}
