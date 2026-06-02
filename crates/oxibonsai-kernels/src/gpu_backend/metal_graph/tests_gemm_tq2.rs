//! Ternary (TQ2) GEMM parity + buffer-pool tests for the Metal graph engine.
//!
//! Covers `encode_gemm_tq2` (anti-cap-of-8 regression), the v8/v9/v10 ternary
//! GEMM kernel parity sweeps + back-to-back speed benches, and the
//! deterministic DiT-GEMM buffer-pool alloc-count proof. Split out of
//! `tests.rs`.

use metal::MTLResourceOptions;
use std::sync::Arc;

use super::buffers::{alloc_buf, download_f32, upload_f32};
use super::graph::MetalGraph;

/// Correctness + anti-cap-of-8 regression test for the batched ternary GEMM
/// entry point [`MetalGraph::encode_gemm_tq2`].
///
/// Builds a small ternary weight `[N=40, K=256]` (deterministic LCG codes in
/// `{-1,0,+1}` packed into `BlockTQ2_0_g128` AoS, per-128-block f16 scale) and,
/// for several batch sizes `M ∈ {1, 7, 8, 9, 100}` (deliberately spanning the
/// 8-column tiling boundary to prove the kernel has no cap-of-8 bug), compares
/// the GPU GEMM against a CPU reference `out[m,n] = Σ_k input[m,k]·W[n,k]`.
/// Asserts max-abs error `< 1e-3` for every `M`. Skips gracefully if no Metal
/// device is present.
#[test]
fn test_encode_gemm_tq2_matches_reference() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;

    // Skip cleanly on hosts without a Metal device rather than fail.
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => return,
    };

    let n_rows = 40usize; // N
    let k = 256usize; // inner dim, 2 blocks of 128 per row
    let blocks_per_row = k / 128;

    // Deterministic LCG (Numerical Recipes constants) → codes in {0,1,2}
    // (== {-1, 0, +1} after decode), packed LSB-first 4 codes/byte.
    let mut lcg: u32 = 0x1234_5678;
    let mut next_code = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) % 3) as u8
    };

    let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
    for row in 0..n_rows {
        for bk in 0..blocks_per_row {
            let mut qs = [0u8; 32];
            for b in qs.iter_mut() {
                let c0 = next_code();
                let c1 = next_code();
                let c2 = next_code();
                let c3 = next_code();
                *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
            }
            // Distinct, non-trivial per-block scale.
            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.0625 + 0.0078125 * (row as f32 + 0.5 * bk as f32)),
            });
        }
    }

    // CPU reference: dequantize the weight to f32 [N, K] once.
    let mut dequant_w = vec![0f32; n_rows * k];
    BlockTQ2_0_g128::dequant(&blocks, &mut dequant_w).expect("dequant reference weight failed");

    // Upload via the public SoA cache path with a distinct key.
    let aos_bytes = {
        let ptr = blocks.as_ptr() as *const u8;
        let len = std::mem::size_of_val(blocks.as_slice());
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    let handle = graph
        .get_or_upload_tq2_weight_soa(7_700_001u64, aos_bytes)
        .expect("get_or_upload_tq2_weight_soa failed");

    for &m in &[1usize, 7, 8, 9, 100] {
        // Deterministic, index-derived input [M, K] (row-major).
        let input: Vec<f32> = (0..m * k)
            .map(|i| {
                let row = i / k;
                let col = i % k;
                ((col as f32) * 0.013 - 0.4) + (row as f32) * 0.0007
            })
            .collect();

        // GPU GEMM.
        let mut got = vec![0f32; m * n_rows];
        graph
            .encode_gemm_tq2(&handle, &input, &mut got, m, n_rows, k)
            .expect("encode_gemm_tq2 failed");

        // CPU reference: out[m, n] = Σ_k input[m, k] * W[n, k] (column-major
        // == row-major [M, N], matching the kernel's outputs[col*n_rows+row]).
        let mut expected = vec![0f32; m * n_rows];
        for mm in 0..m {
            for n in 0..n_rows {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += input[mm * k + kk] * dequant_w[n * k + kk];
                }
                expected[mm * n_rows + n] = acc;
            }
        }

        let mut max_abs_err = 0f32;
        for (idx, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
            let e = (a - b).abs();
            if e > max_abs_err {
                max_abs_err = e;
            }
            assert!(e < 1e-3, "M={m} idx={idx}: expected {a}, got {b} (|Δ|={e})");
        }
        // Sanity: ensure the GEMM produced non-trivial output (not all zeros),
        // so the comparison above is meaningful.
        let any_nonzero = got.iter().any(|&v| v.abs() > 1e-6);
        assert!(any_nonzero, "M={m}: GEMM output is all-zero (suspicious)");

        // Diagnostic (visible only under `--nocapture`).
        eprintln!("encode_gemm_tq2: M={m:>4} max_abs_err={max_abs_err:e}");
    }
}

/// Parity sweep for the **tiled v8** ternary GEMM (`encode_gemm_tq2` now
/// dispatches `gemm_tq2_g128_v8_tiled`).
///
/// Sweeps `M ∈ {1,7,8,9,32,33,100,1536}`, `N ∈ {8,40,3072}`,
/// `K ∈ {128,256,3072}` — **including** shapes that are *not* multiples of the
/// `TN=8` / `TM=32` tile sizes (e.g. `N=40`, `M=33`, `M=100`) to exercise the
/// boundary clamps — and compares the GPU result against a CPU reference
/// (`BlockTQ2_0_g128::dequant` + naive `out[m,n] = Σ_k A[m,k]·W[n,k]`).
/// Asserts max-abs error `< 1e-3` for every shape. Skips cleanly with no Metal
/// device.
///
/// The weight is built + uploaded once per `(N,K)` and reused across all `M`
/// to keep the sweep affordable; the CPU reference is parallelized with rayon
/// for the large `M=1536` cases.
#[test]
fn test_encode_gemm_tq2_v8_tiled_parity() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    use rayon::prelude::*;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => return,
    };

    let ms = [1usize, 7, 8, 9, 32, 33, 100, 1536];
    let ns = [8usize, 40, 3072];
    let ks = [128usize, 256, 3072];

    let mut weight_key: u64 = 7_700_100;

    for &n_rows in &ns {
        for &k in &ks {
            let blocks_per_row = k / 128;

            // Deterministic LCG codes in {0,1,2} (== {-1,0,+1} after decode),
            // packed LSB-first 4 codes/byte. Seed varies per (N,K).
            let mut lcg: u32 = 0x9E37_79B1 ^ ((n_rows as u32) << 8) ^ (k as u32);
            let mut next_code = || {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((lcg >> 16) % 3) as u8
            };

            let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
            for row in 0..n_rows {
                for bk in 0..blocks_per_row {
                    let mut qs = [0u8; 32];
                    for b in qs.iter_mut() {
                        let c0 = next_code();
                        let c1 = next_code();
                        let c2 = next_code();
                        let c3 = next_code();
                        *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                    }
                    blocks.push(BlockTQ2_0_g128 {
                        qs,
                        d: f16::from_f32(0.05 + 0.003 * (row % 17) as f32 + 0.002 * bk as f32),
                    });
                }
            }

            // CPU reference weight [N, K].
            let mut dequant_w = vec![0f32; n_rows * k];
            BlockTQ2_0_g128::dequant(&blocks, &mut dequant_w)
                .expect("dequant reference weight failed");

            // Upload via the public SoA cache path (distinct key per (N,K)).
            let aos_bytes = {
                let ptr = blocks.as_ptr() as *const u8;
                let len = std::mem::size_of_val(blocks.as_slice());
                unsafe { std::slice::from_raw_parts(ptr, len) }
            };
            let handle = graph
                .get_or_upload_tq2_weight_soa(weight_key, aos_bytes)
                .expect("get_or_upload_tq2_weight_soa failed");
            weight_key += 1;

            for &m in &ms {
                // Deterministic, index-derived input [M, K] (row-major).
                let input: Vec<f32> = (0..m * k)
                    .map(|i| {
                        let row = i / k;
                        let col = i % k;
                        ((col as f32) * 0.011 - 0.37).sin() + (row as f32) * 0.0005
                    })
                    .collect();

                // GPU GEMM (now the tiled v8 kernel).
                let mut got = vec![0f32; m * n_rows];
                graph
                    .encode_gemm_tq2(&handle, &input, &mut got, m, n_rows, k)
                    .expect("encode_gemm_tq2 (v8) failed");

                // CPU reference: out[m,n] = Σ_k input[m,k] · W[n,k], laid out
                // column-major == row-major [M,N] (outputs[col*n_rows+row]).
                let mut expected = vec![0f32; m * n_rows];
                expected
                    .par_chunks_mut(n_rows)
                    .enumerate()
                    .for_each(|(mm, out_row)| {
                        let in_row = &input[mm * k..mm * k + k];
                        for (n, slot) in out_row.iter_mut().enumerate() {
                            let w_row = &dequant_w[n * k..n * k + k];
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
                        "v8 N={n_rows} K={k} M={m} idx={idx}: expected {a}, got {b} (|Δ|={e})"
                    );
                }
                // Sanity: non-trivial output (unless a degenerate empty shape).
                if m * n_rows > 0 {
                    let any_nonzero = got.iter().any(|&v| v.abs() > 1e-6);
                    assert!(
                        any_nonzero,
                        "v8 N={n_rows} K={k} M={m}: GEMM output is all-zero (suspicious)"
                    );
                }

                eprintln!(
                    "encode_gemm_tq2 v8: N={n_rows:>4} K={k:>4} M={m:>4} max_abs_err={max_abs_err:e}"
                );
            }
        }
    }
}

/// Load-robust relative micro-benchmark: `gemm_tq2_g128_v7` vs the tiled
/// `gemm_tq2_g128_v8_tiled` on the DiT-shaped problems, back-to-back on the
/// same machine state.
///
/// Absolute timings are unreliable when the host is heavily loaded, but the
/// **ratio** `v7 / v8` measured back-to-back cancels the shared load, so it is
/// the trustworthy speed signal. Both kernels read the *same* pre-uploaded SoA
/// weight (so we time compute, not upload), the weight cache is warmed before
/// timing, several iterations are run, and the **median** per-kernel time is
/// reported along with the speedup ratio.
///
/// Ignored by default (it is a benchmark, and needs a Metal device + large
/// buffers); run explicitly with:
/// `cargo test -p oxibonsai-kernels --features metal --release -- --ignored --nocapture gemm_tq2_v7_vs_v8`.
#[test]
#[ignore = "benchmark: run explicitly with --ignored --nocapture"]
fn bench_gemm_tq2_v7_vs_v8_ratio() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    use std::time::Instant;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping bench");
            return;
        }
    };

    // (M, N, K) — the two big DiT shapes from the spec.
    let shapes = [(1536usize, 27648usize, 3072usize), (1536, 3072, 3072)];
    const WARMUP: usize = 3;
    const ITERS: usize = 11; // odd → unambiguous median

    for (shape_idx, (m, n_rows, k)) in shapes.into_iter().enumerate() {
        let bench_key: u64 = 7_800_000 + shape_idx as u64;
        let blocks_per_row = k / 128;

        // Build a deterministic ternary weight [N, K].
        let mut lcg: u32 = 0xDEAD_BEEF ^ (n_rows as u32);
        let mut next_code = || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((lcg >> 16) % 3) as u8
        };
        let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
        for _ in 0..n_rows * blocks_per_row {
            let mut qs = [0u8; 32];
            for b in qs.iter_mut() {
                let c0 = next_code();
                let c1 = next_code();
                let c2 = next_code();
                let c3 = next_code();
                *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
            }
            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.0625),
            });
        }
        let aos_bytes = {
            let ptr = blocks.as_ptr() as *const u8;
            let len = std::mem::size_of_val(blocks.as_slice());
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let handle = graph
            .get_or_upload_tq2_weight_soa(bench_key, aos_bytes)
            .expect("upload weight failed");

        // Pre-upload input + output buffers once (shared-storage, sized exactly).
        let opts = MTLResourceOptions::StorageModeShared;
        let input: Vec<f32> = (0..m * k)
            .map(|i| ((i % 251) as f32) * 0.001 - 0.12)
            .collect();
        let input_buf = alloc_buf(
            &graph.device,
            std::mem::size_of_val(&input[..]) as u64,
            opts,
        )
        .expect("alloc input failed");
        unsafe { upload_f32(&input_buf, &input) };
        let out_buf = alloc_buf(
            &graph.device,
            (m * n_rows * std::mem::size_of::<f32>()) as u64,
            opts,
        )
        .expect("alloc output failed");

        // Closure: run one dispatch of the chosen kernel and block until done.
        let run_once = |use_v8: bool| {
            let cmd_buf = graph.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            if use_v8 {
                graph.dispatch_gemm_tq2_v8(
                    encoder,
                    &handle.buffer,
                    &input_buf,
                    &out_buf,
                    n_rows as u32,
                    k as u32,
                    m as u32,
                );
            } else {
                graph.dispatch_gemm_tq2_v7(
                    encoder,
                    &handle.buffer,
                    &input_buf,
                    &out_buf,
                    n_rows as u32,
                    k as u32,
                    m as u32,
                );
            }
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        };

        let measure = |use_v8: bool| -> f64 {
            for _ in 0..WARMUP {
                run_once(use_v8);
            }
            let mut times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = Instant::now();
                run_once(use_v8);
                times.push(t0.elapsed().as_secs_f64() * 1e3); // ms
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            times[ITERS / 2]
        };

        // Interleave v7 / v8 to share machine state as closely as possible.
        let v7_ms = measure(false);
        let v8_ms = measure(true);
        let v7_ms2 = measure(false);
        let v8_ms2 = measure(true);
        let v7_med = (v7_ms + v7_ms2) / 2.0;
        let v8_med = (v8_ms + v8_ms2) / 2.0;
        let ratio = v7_med / v8_med;

        eprintln!(
            "BENCH M={m} N={n_rows} K={k}: v7={v7_med:.3}ms  v8={v8_med:.3}ms  speedup v7/v8={ratio:.2}x  (median of {ITERS}, warmup {WARMUP}, 2 reps each)"
        );
    }
}

/// Run the `gemm_tq2_g128_v9_simdgroup` kernel directly (mirroring
/// `encode_gemm_tq2`'s shared-storage I/O), independent of which kernel
/// `encode_gemm_tq2` currently dispatches.
///
/// Allocates exact-sized shared-storage input/output buffers, uploads `input`
/// (`[M,K]` row-major), dispatches v9 against the pre-uploaded SoA weight, and
/// returns the `[M,N]` (column-major == row-major) result. Used by the v9
/// parity sweep so it validates v9 even while `encode_gemm_tq2` stays on v8
/// pending the speed ratio.
#[cfg(test)]
fn run_gemm_tq2_v9(
    graph: &MetalGraph,
    weight: &super::error::MetalWeightHandle,
    input: &[f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> Vec<f32> {
    let opts = MTLResourceOptions::StorageModeShared;
    let input_buf = alloc_buf(&graph.device, std::mem::size_of_val(input) as u64, opts)
        .expect("alloc input failed");
    unsafe { upload_f32(&input_buf, input) };
    let out_buf = alloc_buf(
        &graph.device,
        (m * n_rows * std::mem::size_of::<f32>()) as u64,
        opts,
    )
    .expect("alloc output failed");

    let cmd_buf = graph.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    graph.dispatch_gemm_tq2_v9(
        encoder,
        &weight.buffer,
        &input_buf,
        &out_buf,
        n_rows as u32,
        k as u32,
        m as u32,
    );
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut got = vec![0f32; m * n_rows];
    unsafe { download_f32(&out_buf, &mut got) };
    got
}

/// Run the staging-optimized `gemm_tq2_g128_v10_simdgroup` kernel directly
/// (mirroring [`run_gemm_tq2_v9`]), independent of which kernel
/// `encode_gemm_tq2` currently dispatches.
///
/// Allocates exact-sized shared-storage input/output buffers, uploads `input`
/// (`[M,K]` row-major), dispatches v10 against the pre-uploaded SoA weight, and
/// returns the `[M,N]` (column-major == row-major) result. Used by the v10
/// parity sweep so it validates v10 even while `encode_gemm_tq2` stays on v9
/// pending the speed ratio.
#[cfg(test)]
fn run_gemm_tq2_v10(
    graph: &MetalGraph,
    weight: &super::error::MetalWeightHandle,
    input: &[f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> Vec<f32> {
    let opts = MTLResourceOptions::StorageModeShared;
    let input_buf = alloc_buf(&graph.device, std::mem::size_of_val(input) as u64, opts)
        .expect("alloc input failed");
    unsafe { upload_f32(&input_buf, input) };
    let out_buf = alloc_buf(
        &graph.device,
        (m * n_rows * std::mem::size_of::<f32>()) as u64,
        opts,
    )
    .expect("alloc output failed");

    let cmd_buf = graph.command_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    graph.dispatch_gemm_tq2_v10(
        encoder,
        &weight.buffer,
        &input_buf,
        &out_buf,
        n_rows as u32,
        k as u32,
        m as u32,
    );
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut got = vec![0f32; m * n_rows];
    unsafe { download_f32(&out_buf, &mut got) };
    got
}

/// Parity sweep for the **`simdgroup_matrix` v9** ternary GEMM
/// (`gemm_tq2_g128_v9_simdgroup`), dispatched directly via
/// [`run_gemm_tq2_v9`].
///
/// Sweeps `M ∈ {1,7,8,9,32,33,100,1536}`, `N ∈ {8,40,3072}`,
/// `K ∈ {128,256,3072}` — **including** shapes that are *not* multiples of the
/// `V9_TM=64` / `V9_TN=64` / `V9_TK=32` tile sizes (e.g. `N=8`, `N=40`, `M=33`,
/// `M=100`) to exercise the boundary clamps — and compares the GPU result
/// against a CPU reference (`BlockTQ2_0_g128::dequant` + naive
/// `out[m,n] = Σ_k A[m,k]·W[n,k]`). Asserts max-abs error `< 1e-3` for every
/// shape. Skips cleanly with no Metal device.
///
/// The weight is built + uploaded once per `(N,K)` and reused across all `M`;
/// the CPU reference is parallelized with rayon for the large `M=1536` cases.
#[test]
fn test_gemm_tq2_v9_simdgroup_parity() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    use rayon::prelude::*;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => return,
    };

    let ms = [1usize, 7, 8, 9, 32, 33, 100, 1536];
    let ns = [8usize, 40, 3072];
    let ks = [128usize, 256, 3072];

    let mut weight_key: u64 = 7_700_900;

    for &n_rows in &ns {
        for &k in &ks {
            let blocks_per_row = k / 128;

            let mut lcg: u32 = 0x2545_F491 ^ ((n_rows as u32) << 8) ^ (k as u32);
            let mut next_code = || {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((lcg >> 16) % 3) as u8
            };

            let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
            for row in 0..n_rows {
                for bk in 0..blocks_per_row {
                    let mut qs = [0u8; 32];
                    for b in qs.iter_mut() {
                        let c0 = next_code();
                        let c1 = next_code();
                        let c2 = next_code();
                        let c3 = next_code();
                        *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                    }
                    blocks.push(BlockTQ2_0_g128 {
                        qs,
                        d: f16::from_f32(0.05 + 0.003 * (row % 17) as f32 + 0.002 * bk as f32),
                    });
                }
            }

            let mut dequant_w = vec![0f32; n_rows * k];
            BlockTQ2_0_g128::dequant(&blocks, &mut dequant_w)
                .expect("dequant reference weight failed");

            let aos_bytes = {
                let ptr = blocks.as_ptr() as *const u8;
                let len = std::mem::size_of_val(blocks.as_slice());
                unsafe { std::slice::from_raw_parts(ptr, len) }
            };
            let handle = graph
                .get_or_upload_tq2_weight_soa(weight_key, aos_bytes)
                .expect("get_or_upload_tq2_weight_soa failed");
            weight_key += 1;

            for &m in &ms {
                let input: Vec<f32> = (0..m * k)
                    .map(|i| {
                        let row = i / k;
                        let col = i % k;
                        ((col as f32) * 0.011 - 0.37).sin() + (row as f32) * 0.0005
                    })
                    .collect();

                let got = run_gemm_tq2_v9(&graph, &handle, &input, m, n_rows, k);

                let mut expected = vec![0f32; m * n_rows];
                expected
                    .par_chunks_mut(n_rows)
                    .enumerate()
                    .for_each(|(mm, out_row)| {
                        let in_row = &input[mm * k..mm * k + k];
                        for (n, slot) in out_row.iter_mut().enumerate() {
                            let w_row = &dequant_w[n * k..n * k + k];
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
                        "v9 N={n_rows} K={k} M={m} idx={idx}: expected {a}, got {b} (|Δ|={e})"
                    );
                }
                if m * n_rows > 0 {
                    let any_nonzero = got.iter().any(|&v| v.abs() > 1e-6);
                    assert!(
                        any_nonzero,
                        "v9 N={n_rows} K={k} M={m}: GEMM output is all-zero (suspicious)"
                    );
                }

                eprintln!(
                    "gemm_tq2 v9: N={n_rows:>4} K={k:>4} M={m:>4} max_abs_err={max_abs_err:e}"
                );
            }
        }
    }
}

/// Parity sweep for the **staging-optimized `simdgroup_matrix` v10** ternary
/// GEMM (`gemm_tq2_g128_v10_simdgroup`), dispatched directly via
/// [`run_gemm_tq2_v10`].
///
/// Sweeps the *same* shapes as the v9 sweep — `M ∈ {1,7,8,9,32,33,100,1536}`,
/// `N ∈ {8,40,3072}`, `K ∈ {128,256,3072}`, including non-tile-multiples to
/// exercise the boundary clamps — and compares the GPU result against the same
/// CPU reference (`BlockTQ2_0_g128::dequant` + naive
/// `out[m,n] = Σ_k A[m,k]·W[n,k]`). Asserts max-abs error `< 1e-3` for every
/// shape (expected ≈1e-5: `D`'s f16 staging is *exact* for ternary `code×scale`
/// and `A` stays f32). Skips cleanly with no Metal device.
#[test]
fn test_gemm_tq2_v10_simdgroup_parity() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    use rayon::prelude::*;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => return,
    };

    let ms = [1usize, 7, 8, 9, 32, 33, 100, 1536];
    let ns = [8usize, 40, 3072];
    let ks = [128usize, 256, 3072];

    let mut weight_key: u64 = 7_710_900;

    for &n_rows in &ns {
        for &k in &ks {
            let blocks_per_row = k / 128;

            let mut lcg: u32 = 0x2545_F491 ^ ((n_rows as u32) << 8) ^ (k as u32);
            let mut next_code = || {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((lcg >> 16) % 3) as u8
            };

            let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
            for row in 0..n_rows {
                for bk in 0..blocks_per_row {
                    let mut qs = [0u8; 32];
                    for b in qs.iter_mut() {
                        let c0 = next_code();
                        let c1 = next_code();
                        let c2 = next_code();
                        let c3 = next_code();
                        *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                    }
                    blocks.push(BlockTQ2_0_g128 {
                        qs,
                        d: f16::from_f32(0.05 + 0.003 * (row % 17) as f32 + 0.002 * bk as f32),
                    });
                }
            }

            let mut dequant_w = vec![0f32; n_rows * k];
            BlockTQ2_0_g128::dequant(&blocks, &mut dequant_w)
                .expect("dequant reference weight failed");

            let aos_bytes = {
                let ptr = blocks.as_ptr() as *const u8;
                let len = std::mem::size_of_val(blocks.as_slice());
                unsafe { std::slice::from_raw_parts(ptr, len) }
            };
            let handle = graph
                .get_or_upload_tq2_weight_soa(weight_key, aos_bytes)
                .expect("get_or_upload_tq2_weight_soa failed");
            weight_key += 1;

            for &m in &ms {
                let input: Vec<f32> = (0..m * k)
                    .map(|i| {
                        let row = i / k;
                        let col = i % k;
                        ((col as f32) * 0.011 - 0.37).sin() + (row as f32) * 0.0005
                    })
                    .collect();

                let got = run_gemm_tq2_v10(&graph, &handle, &input, m, n_rows, k);

                let mut expected = vec![0f32; m * n_rows];
                expected
                    .par_chunks_mut(n_rows)
                    .enumerate()
                    .for_each(|(mm, out_row)| {
                        let in_row = &input[mm * k..mm * k + k];
                        for (n, slot) in out_row.iter_mut().enumerate() {
                            let w_row = &dequant_w[n * k..n * k + k];
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
                        "v10 N={n_rows} K={k} M={m} idx={idx}: expected {a}, got {b} (|Δ|={e})"
                    );
                }
                if m * n_rows > 0 {
                    let any_nonzero = got.iter().any(|&v| v.abs() > 1e-6);
                    assert!(
                        any_nonzero,
                        "v10 N={n_rows} K={k} M={m}: GEMM output is all-zero (suspicious)"
                    );
                }

                eprintln!(
                    "gemm_tq2 v10: N={n_rows:>4} K={k:>4} M={m:>4} max_abs_err={max_abs_err:e}"
                );
            }
        }
    }
}

/// Load-robust relative micro-benchmark: the `simdgroup_matrix`
/// `gemm_tq2_g128_v9_simdgroup` vs the register-blocked
/// `gemm_tq2_g128_v8_tiled` **and** vs `gemm_tq2_g128_v7`, on the DiT-shaped
/// problems, back-to-back on the same machine state.
///
/// Absolute timings are unreliable under heavy host load, but the back-to-back
/// **ratios** `v8 / v9` and `v7 / v9` cancel the shared load and are the
/// trustworthy speed signal (the headline is v9's speedup over v8). All three
/// kernels read the *same* pre-uploaded SoA weight (timing compute, not
/// upload); the weight cache is warmed before timing; the **median** of several
/// iterations is reported per kernel.
///
/// Ignored by default (benchmark; needs a Metal device + large buffers); run:
/// `cargo test -p oxibonsai-kernels --features metal --release -- --ignored --nocapture gemm_tq2_v9_ratio`.
#[test]
#[ignore = "benchmark: run explicitly with --ignored --nocapture"]
fn bench_gemm_tq2_v9_ratio() {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;
    use std::time::Instant;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping bench");
            return;
        }
    };

    // (M, N, K) — the two big DiT shapes from the spec.
    let shapes = [(1536usize, 27648usize, 3072usize), (1536, 3072, 3072)];
    const WARMUP: usize = 3;
    const ITERS: usize = 11; // odd → unambiguous median

    for (shape_idx, (m, n_rows, k)) in shapes.into_iter().enumerate() {
        let bench_key: u64 = 7_900_000 + shape_idx as u64;
        let blocks_per_row = k / 128;

        let mut lcg: u32 = 0xBADC_0FFE ^ (n_rows as u32);
        let mut next_code = || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((lcg >> 16) % 3) as u8
        };
        let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
        for _ in 0..n_rows * blocks_per_row {
            let mut qs = [0u8; 32];
            for b in qs.iter_mut() {
                let c0 = next_code();
                let c1 = next_code();
                let c2 = next_code();
                let c3 = next_code();
                *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
            }
            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.0625),
            });
        }
        let aos_bytes = {
            let ptr = blocks.as_ptr() as *const u8;
            let len = std::mem::size_of_val(blocks.as_slice());
            unsafe { std::slice::from_raw_parts(ptr, len) }
        };
        let handle = graph
            .get_or_upload_tq2_weight_soa(bench_key, aos_bytes)
            .expect("upload weight failed");

        let opts = MTLResourceOptions::StorageModeShared;
        let input: Vec<f32> = (0..m * k)
            .map(|i| ((i % 251) as f32) * 0.001 - 0.12)
            .collect();
        let input_buf = alloc_buf(
            &graph.device,
            std::mem::size_of_val(&input[..]) as u64,
            opts,
        )
        .expect("alloc input failed");
        unsafe { upload_f32(&input_buf, &input) };
        let out_buf = alloc_buf(
            &graph.device,
            (m * n_rows * std::mem::size_of::<f32>()) as u64,
            opts,
        )
        .expect("alloc output failed");

        // kernel: 0 = v7, 1 = v8, 2 = v9, 3 = v10.
        let run_once = |kernel: u8| {
            let cmd_buf = graph.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            match kernel {
                0 => graph.dispatch_gemm_tq2_v7(
                    encoder,
                    &handle.buffer,
                    &input_buf,
                    &out_buf,
                    n_rows as u32,
                    k as u32,
                    m as u32,
                ),
                1 => graph.dispatch_gemm_tq2_v8(
                    encoder,
                    &handle.buffer,
                    &input_buf,
                    &out_buf,
                    n_rows as u32,
                    k as u32,
                    m as u32,
                ),
                2 => graph.dispatch_gemm_tq2_v9(
                    encoder,
                    &handle.buffer,
                    &input_buf,
                    &out_buf,
                    n_rows as u32,
                    k as u32,
                    m as u32,
                ),
                _ => graph.dispatch_gemm_tq2_v10(
                    encoder,
                    &handle.buffer,
                    &input_buf,
                    &out_buf,
                    n_rows as u32,
                    k as u32,
                    m as u32,
                ),
            }
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        };

        let measure = |kernel: u8| -> f64 {
            for _ in 0..WARMUP {
                run_once(kernel);
            }
            let mut times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = Instant::now();
                run_once(kernel);
                times.push(t0.elapsed().as_secs_f64() * 1e3); // ms
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            times[ITERS / 2]
        };

        // Interleave to share machine state, two reps each, average the medians.
        let v7a = measure(0);
        let v8a = measure(1);
        let v9a = measure(2);
        let v10a = measure(3);
        let v7b = measure(0);
        let v8b = measure(1);
        let v9b = measure(2);
        let v10b = measure(3);
        let v7_med = (v7a + v7b) / 2.0;
        let v8_med = (v8a + v8b) / 2.0;
        let v9_med = (v9a + v9b) / 2.0;
        let v10_med = (v10a + v10b) / 2.0;
        let v8_over_v9 = v8_med / v9_med;
        let v7_over_v9 = v7_med / v9_med;
        // Headline: v10's speedup over v9 (the kernel it would replace), plus
        // v10 vs v8 / v7 for context.
        let v9_over_v10 = v9_med / v10_med;
        let v8_over_v10 = v8_med / v10_med;
        let v7_over_v10 = v7_med / v10_med;

        eprintln!(
            "BENCH M={m} N={n_rows} K={k}: v7={v7_med:.3}ms v8={v8_med:.3}ms v9={v9_med:.3}ms v10={v10_med:.3}ms\n  \
             speedup v8/v9={v8_over_v9:.2}x v7/v9={v7_over_v9:.2}x  ||  HEADLINE v9/v10={v9_over_v10:.3}x  v8/v10={v8_over_v10:.2}x  v7/v10={v7_over_v10:.2}x  \
             (median of {ITERS}, warmup {WARMUP}, 2 reps each)"
        );
    }
}

/// Build a deterministic ternary weight `[n_rows, k]` (`{-1,0,+1}` LCG codes,
/// per-128-block f16 scale) and upload it via the public SoA cache. Helper for
/// the buffer-pool alloc-count proof below; mirrors the packing used by the
/// parity sweeps.
#[cfg(test)]
fn upload_ternary_weight_for_pool_test(
    graph: &MetalGraph,
    key: u64,
    n_rows: usize,
    k: usize,
) -> Arc<super::error::MetalWeightHandle> {
    use half::f16;
    use oxibonsai_core::BlockTQ2_0_g128;

    let blocks_per_row = k / 128;
    let mut lcg: u32 = 0x9E37_79B9 ^ ((n_rows as u32) << 7) ^ (k as u32);
    let mut next_code = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) % 3) as u8
    };
    let mut blocks: Vec<BlockTQ2_0_g128> = Vec::with_capacity(n_rows * blocks_per_row);
    for row in 0..n_rows {
        for bk in 0..blocks_per_row {
            let mut qs = [0u8; 32];
            for b in qs.iter_mut() {
                let c0 = next_code();
                let c1 = next_code();
                let c2 = next_code();
                let c3 = next_code();
                *b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
            }
            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.05 + 0.003 * (row % 13) as f32 + 0.002 * bk as f32),
            });
        }
    }
    let aos_bytes = {
        let ptr = blocks.as_ptr() as *const u8;
        let len = std::mem::size_of_val(blocks.as_slice());
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    graph
        .get_or_upload_tq2_weight_soa(key, aos_bytes)
        .expect("get_or_upload_tq2_weight_soa failed")
}

/// Deterministic, **load-independent** proof that the DiT-GEMM buffer pool
/// engages: once warmed to its peak footprint it allocates **zero** buffers for
/// every subsequent matmul — versus the pre-pool path, which allocated two fresh
/// buffers (`input` + `output`, up to ~170 MB) on every one of the ~100
/// matmuls/forward.
///
/// # Why this is robust to parallel test execution
///
/// The alloc counter [`MetalGraph::gemm_pool_alloc_count`] is process-wide and
/// the `MetalGraph` is a shared singleton, so *other* tests calling
/// `encode_gemm_tq2` concurrently can grow the pool inside a naive measurement
/// window. The two sibling tests that share this pool
/// (`test_encode_gemm_tq2_matches_reference`, `test_encode_gemm_tq2_v8_tiled_parity`)
/// top out at `M=1536, N=3072, K=3072` ⇒ **18 MB input / 18 MB output**.
/// (`test_gemm_tq2_v9_simdgroup_parity` allocates its *own* buffers via
/// `run_gemm_tq2_v9`, never touching this pool.)
///
/// So we first **warm the pool to exactly that ceiling** (`M=1536, N=3072,
/// K=3072`). Because pool capacities are *grow-only* (never shrink), after this
/// warm **no** concurrent sibling test can ever trigger a further grow — every
/// shape any test requests now fits. The steady-state phase then replays many
/// strictly-smaller shapes and asserts the alloc-count delta is **exactly 0**,
/// an assertion that holds regardless of host load or test interleaving.
#[test]
fn test_gemm_pool_alloc_count_is_constant_after_warmup() {
    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => return,
    };

    // Deterministic row-major [M, K] input generator (shared by both weights).
    let make_input = |m: usize, k: usize| -> Vec<f32> {
        (0..m * k)
            .map(|i| {
                let row = i / k;
                let col = i % k;
                ((col as f32) * 0.009 - 0.31).cos() + (row as f32) * 0.0004
            })
            .collect()
    };

    // ── Phase 1: warm the pool to the ecosystem-wide ceiling ──────────────
    // (M, N, K) = (1536, 3072, 3072): input = output = 1536·3072·4 = 18 MB,
    // matching the largest footprint any sibling test drives through this pool.
    // Grow-only caps mean that after this single call no concurrent test can
    // enlarge the pool, so the Phase-2 delta is immune to interleaving.
    let warm_n = 3072usize;
    let warm_k = 3072usize;
    let warm_m = 1536usize;
    let warm_handle = upload_ternary_weight_for_pool_test(&graph, 7_701_400u64, warm_n, warm_k);
    {
        let input = make_input(warm_m, warm_k);
        let mut out = vec![0f32; warm_m * warm_n];
        graph
            .encode_gemm_tq2(&warm_handle, &input, &mut out, warm_m, warm_n, warm_k)
            .expect("encode_gemm_tq2 (pool warm) failed");
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "pool-warm GEMM output is all-zero (suspicious)"
        );
    }

    // ── Phase 2: steady-state replay — MUST allocate nothing ──────────────
    // A small weight (N=64, K=256) swept over many M, every footprint far below
    // the 18 MB ceiling warmed above — exactly the regime of the ~100 matmuls a
    // warmed DiT forward runs after the single-block projection has sized the
    // pool to its peak. With grow-only caps already at the ceiling, none of
    // these — nor any concurrent sibling test — can grow the pool.
    let n_rows = 64usize;
    let k = 256usize; // multiple of 128
    let small_handle = upload_ternary_weight_for_pool_test(&graph, 7_701_500u64, n_rows, k);
    let run_small = |m: usize| {
        let input = make_input(m, k);
        let mut out = vec![0f32; m * n_rows];
        graph
            .encode_gemm_tq2(&small_handle, &input, &mut out, m, n_rows, k)
            .expect("encode_gemm_tq2 (pool steady) failed");
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "M={m}: pooled GEMM output is all-zero (suspicious)"
        );
    };

    let before = MetalGraph::gemm_pool_alloc_count();
    let steady = [1usize, 8, 9, 32, 100, 256, 512, 7, 64, 128, 512, 1];
    for &m in &steady {
        run_small(m);
    }
    let steady_allocs = MetalGraph::gemm_pool_alloc_count() - before;

    eprintln!(
        "gemm pool: {} steady-state matmuls (≤18 MB, warmed) → {steady_allocs} new buffer \
         allocs (pre-pool would be {} = 2 per matmul)",
        steady.len(),
        steady.len() * 2,
    );

    // The headline, load-robust assertion: a warmed pool allocates ZERO buffers
    // across the steady-state matmuls (vs. 2 per matmul = 24 pre-pool).
    assert_eq!(
        steady_allocs,
        0,
        "warmed DiT-GEMM pool must trigger 0 new allocations across {} steady-state \
         matmuls, but it grew {steady_allocs} time(s)",
        steady.len()
    );
}

/// Best-effort A/B timing of the buffer-pool win: the **pooled**
/// `encode_gemm_tq2` (resident I/O scratch) vs the **pre-pool** path that
/// allocated a fresh `StorageModeShared` input *and* output buffer on every
/// call. Both dispatch the identical v9 kernel over the two big DiT shapes from
/// the spec — only the buffer *lifetime* differs — so the delta isolates exactly
/// the per-call alloc/free/first-touch overhead the pool removes (the
/// `M=1536,N=27648` output alone is ~170 MB).
///
/// Absolute timings are noisy under host load; the trustworthy signal is the
/// back-to-back **ratio** `nopool / pool` (shared load cancels). Ignored by
/// default (benchmark; needs a Metal device + ~170 MB buffers); run:
/// `cargo test -p oxibonsai-kernels --features metal --release -- --ignored --nocapture gemm_pool_alloc_timing`.
#[test]
#[ignore = "benchmark: run explicitly with --ignored --nocapture"]
fn bench_gemm_pool_alloc_timing() {
    use std::time::Instant;

    let graph = match MetalGraph::global() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no Metal device — skipping pool timing");
            return;
        }
    };

    // The two big DiT shapes (the 170 MB single-block proj + the 18 MB to_out).
    let shapes = [(1536usize, 27648usize, 3072usize), (1536, 3072, 3072)];
    const WARMUP: usize = 2;
    const ITERS: usize = 7;
    let opts = MTLResourceOptions::StorageModeShared;

    for (idx, (m, n_rows, k)) in shapes.into_iter().enumerate() {
        let handle = upload_ternary_weight_for_pool_test(&graph, 7_702_000 + idx as u64, n_rows, k);
        let input: Vec<f32> = (0..m * k)
            .map(|i| ((i % 251) as f32) * 0.001 - 0.12)
            .collect();

        // POOLED path: the shipped `encode_gemm_tq2` (reuses resident scratch).
        // A fresh output Vec per call keeps CPU-side allocation symmetric with
        // the pre-pool path's fresh `got` below, so the delta is purely the GPU
        // buffer alloc/free the pool removes.
        let pooled_once = || {
            let mut out = vec![0f32; m * n_rows];
            graph
                .encode_gemm_tq2(&handle, &input, &mut out, m, n_rows, k)
                .expect("pooled encode_gemm_tq2 failed");
        };

        // PRE-POOL path: replicate the OLD body — fresh shared input+output
        // buffer per call, identical v9 dispatch, identical commit/wait/download.
        let nopool_once = || {
            let in_buf = alloc_buf(
                &graph.device,
                std::mem::size_of_val(&input[..]) as u64,
                opts,
            )
            .expect("alloc input failed");
            let out_buf = alloc_buf(
                &graph.device,
                (m * n_rows * std::mem::size_of::<f32>()) as u64,
                opts,
            )
            .expect("alloc output failed");
            unsafe { upload_f32(&in_buf, &input) };
            let cmd_buf = graph.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            graph.dispatch_gemm_tq2_v9(
                encoder,
                &handle.buffer,
                &in_buf,
                &out_buf,
                n_rows as u32,
                k as u32,
                m as u32,
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            let mut got = vec![0f32; m * n_rows];
            unsafe { download_f32(&out_buf, &mut got) };
            // in_buf / out_buf drop here → the per-call free the pool removes.
        };

        let measure = |mut f: Box<dyn FnMut()>| -> f64 {
            for _ in 0..WARMUP {
                f();
            }
            let mut times = Vec::with_capacity(ITERS);
            for _ in 0..ITERS {
                let t0 = Instant::now();
                f();
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            times[ITERS / 2]
        };

        // Interleave to share machine state.
        let pool_a = measure(Box::new(pooled_once));
        let nopool_a = measure(Box::new(nopool_once));
        let pool_b = measure(Box::new(pooled_once));
        let nopool_b = measure(Box::new(nopool_once));
        let pool_med = (pool_a + pool_b) / 2.0;
        let nopool_med = (nopool_a + nopool_b) / 2.0;

        eprintln!(
            "POOL-TIMING M={m} N={n_rows} K={k}: pooled={pool_med:.3}ms  nopool={nopool_med:.3}ms  \
             nopool/pool={:.3}x  saved/call≈{:.3}ms  (median of {ITERS}, warmup {WARMUP}, 2 reps)",
            nopool_med / pool_med,
            nopool_med - pool_med,
        );
    }
}
