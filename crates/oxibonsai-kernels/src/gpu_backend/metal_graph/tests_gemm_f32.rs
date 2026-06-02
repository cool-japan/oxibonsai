//! f32-exact `simdgroup_matrix` GEMM parity test (`encode_gemm_f32`).
//!
//! Sweeps the TE-shaped grid against a parallel naive CPU reference. Split
//! out of `tests.rs`.

use super::graph::MetalGraph;

/// Build a deterministic, index-derived row-major f32 weight `[n_rows, k]`.
///
/// Values are bounded (`tanh`-shaped) so the GEMM accumulates a non-trivial,
/// well-conditioned sum at every `(M, N)` — used by the f32 GEMM parity sweep.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn build_f32_weight(n_rows: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0f32; n_rows * k];
    let mut lcg: u32 =
        0x517C_C1B7 ^ ((n_rows as u32) << 7) ^ (k as u32).wrapping_mul(2_654_435_761);
    for v in w.iter_mut() {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map to roughly [-0.5, 0.5).
        *v = ((lcg >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    w
}

/// Parity sweep for the **f32-exact `simdgroup_matrix`** GEMM
/// ([`MetalGraph::encode_gemm_f32`], dispatching `gemm_f32_simdgroup`).
///
/// Sweeps the spec's TE-shaped grid — `M ∈ {1,7,32,33,128,512}`,
/// `N ∈ {40,1024,2560,4096,9728}`, `K ∈ {2560,9728}` (the `N=40` and `M ∈
/// {7,33}` cases are deliberately *not* multiples of the `64×64` tile to
/// exercise the boundary clamps) — and compares the GPU result against a CPU
/// reference (parallel naive `out[m,n] = Σ_k A[m,k]·W[n,k]`, the identical math
/// the image crate's `gemm::gemm_abt` computes). Asserts max-abs error `< 1e-3`
/// (pure f32 — expected ≈ 1e-4) for every shape. Skips cleanly with no Metal
/// device.
///
/// The weight is built + uploaded once per `(N,K)` (via the public
/// `get_or_upload_f32_weight` cache, keyed by a distinct id) and reused across
/// all `M`; the CPU reference is parallelized with rayon.
#[test]
#[ignore = "heavy: Metal f32-GEMM parity sweep (N up to 9728, K up to 9728) — ~15 min on debug build; run explicitly with --run-ignored"]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_encode_gemm_f32_matches_reference() {
    use rayon::prelude::*;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping f32 GEMM parity sweep");
            return;
        }
    };

    let ms = [1usize, 7, 32, 33, 128, 512];
    let ns = [40usize, 1024, 2560, 4096, 9728];
    let ks = [2560usize, 9728];

    let mut weight_key: u64 = 7_950_000;
    let mut worst_overall = 0f32;

    for &k in &ks {
        for &n_rows in &ns {
            let weight = build_f32_weight(n_rows, k);
            let handle = graph
                .get_or_upload_f32_weight(weight_key, &weight)
                .expect("get_or_upload_f32_weight failed");
            weight_key += 1;

            for &m in &ms {
                // Deterministic, index-derived input [M, K] (row-major).
                let input: Vec<f32> = (0..m * k)
                    .map(|i| {
                        let row = i / k;
                        let col = i % k;
                        ((col as f32) * 0.009 - 0.41).sin() + (row as f32) * 0.0003
                    })
                    .collect();

                // GPU GEMM (f32-exact simdgroup kernel).
                let mut got = vec![0f32; m * n_rows];
                graph
                    .encode_gemm_f32(&handle, &input, &mut got, m, n_rows, k)
                    .expect("encode_gemm_f32 failed");

                // CPU reference: out[m,n] = Σ_k input[m,k] · W[n,k], laid out
                // column-major == row-major [M,N] (outputs[col*n_rows+row]).
                let mut expected = vec![0f32; m * n_rows];
                expected
                    .par_chunks_mut(n_rows)
                    .enumerate()
                    .for_each(|(mm, out_row)| {
                        let in_row = &input[mm * k..mm * k + k];
                        for (n, slot) in out_row.iter_mut().enumerate() {
                            let w_row = &weight[n * k..n * k + k];
                            let mut acc = 0f32;
                            for kk in 0..k {
                                acc += in_row[kk] * w_row[kk];
                            }
                            *slot = acc;
                        }
                    });

                let mut max_abs_err = 0f32;
                for (idx, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                    let e = (a - b).abs();
                    if e > max_abs_err {
                        max_abs_err = e;
                    }
                    assert!(
                        e < 1e-3,
                        "f32 N={n_rows} K={k} M={m} idx={idx}: expected {a}, got {b} (|Δ|={e})"
                    );
                }
                if m * n_rows > 0 {
                    let any_nonzero = got.iter().any(|&v| v.abs() > 1e-6);
                    assert!(
                        any_nonzero,
                        "f32 N={n_rows} K={k} M={m}: GEMM output is all-zero (suspicious)"
                    );
                }
                if max_abs_err > worst_overall {
                    worst_overall = max_abs_err;
                }
                eprintln!(
                    "encode_gemm_f32: N={n_rows:>4} K={k:>4} M={m:>4} max_abs_err={max_abs_err:e}"
                );
            }
        }
    }
    eprintln!("encode_gemm_f32 parity sweep WORST max_abs_err={worst_overall:e} (bound 1e-3)");
}
