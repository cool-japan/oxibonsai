//! Kernel benchmarks for Q1\_0\_g128, ternary, and FP8 operations.
//!
//! Compares reference (scalar), AVX2 SIMD, and parallel kernel performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::f16;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::traits::{Fp8Kernel, OneBitKernel};
use oxibonsai_kernels::{
    dequant::dequant_1bit_g128, gemv::gemv_1bit_g128, gemv_fp8_e4m3_par, gemv_fp8_e5m2_par,
};
use std::hint::black_box;

fn make_blocks(count: usize) -> Vec<BlockQ1_0G128> {
    (0..count)
        .map(|i| BlockQ1_0G128 {
            d: f16::from_f32(0.5 + (i as f32) * 0.001),
            qs: [0xAA; 16], // alternating bits
        })
        .collect()
}

fn bench_dequant(c: &mut Criterion) {
    let blocks = make_blocks(32); // 4096 elements = 1 row of hidden_size
    let mut output = vec![0.0f32; 32 * 128];

    let mut group = c.benchmark_group("dequant_4096");

    group.bench_function("reference", |b| {
        b.iter(|| {
            dequant_1bit_g128(black_box(&blocks), black_box(&mut output))
                .expect("dequant should succeed");
        });
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_avx2::dequant_1bit_g128_avx2(
                    black_box(&blocks),
                    black_box(&mut output),
                )
                .expect("avx2 dequant should succeed");
            });
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        group.bench_function("neon", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_neon::dequant_1bit_g128_neon(
                    black_box(&blocks),
                    black_box(&mut output),
                )
                .expect("neon dequant should succeed");
            });
        });
    }

    group.finish();
}

fn bench_gemv_single_row(c: &mut Criterion) {
    let blocks = make_blocks(32); // 4096 input features
    let input = vec![1.0f32; 4096];
    let mut output = vec![0.0f32; 1];

    let mut group = c.benchmark_group("gemv_1row_4096in");

    group.bench_function("reference", |b| {
        b.iter(|| {
            gemv_1bit_g128(
                black_box(&blocks),
                black_box(&input),
                black_box(&mut output),
                1,
                4096,
            )
            .expect("gemv should succeed");
        });
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_avx2::gemv_1bit_g128_avx2(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    1,
                    4096,
                )
                .expect("avx2 gemv should succeed");
            });
        });
    }

    #[cfg(target_arch = "aarch64")]
    {
        group.bench_function("neon", |b| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_neon::gemv_1bit_g128_neon(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    1,
                    4096,
                )
                .expect("neon gemv should succeed");
            });
        });
    }

    group.finish();
}

fn bench_gemv_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_dispatch");

    for &n_rows in &[32, 256, 1024, 4096] {
        let k = 4096;
        let blocks = make_blocks(n_rows * (k / 128));
        let input = vec![0.5f32; k];
        let mut output = vec![0.0f32; n_rows];

        let ref_disp = KernelDispatcher::with_tier(KernelTier::Reference);
        group.bench_with_input(BenchmarkId::new("reference", n_rows), &n_rows, |b, &nr| {
            b.iter(|| {
                ref_disp
                    .gemv(
                        black_box(&blocks),
                        black_box(&input),
                        black_box(&mut output),
                        nr,
                        k,
                    )
                    .expect("dispatcher gemv should succeed");
            });
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx_disp = KernelDispatcher::with_tier(KernelTier::Avx2);
            group.bench_with_input(BenchmarkId::new("avx2", n_rows), &n_rows, |b, &nr| {
                b.iter(|| {
                    avx_disp
                        .gemv(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            nr,
                            k,
                        )
                        .expect("dispatcher gemv should succeed");
                });
            });

            // Parallel (only above threshold)
            if n_rows >= 64 {
                group.bench_with_input(
                    BenchmarkId::new("avx2+parallel", n_rows),
                    &n_rows,
                    |b, &nr| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemv_1bit_g128_par(
                                &avx_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                nr,
                                k,
                            )
                            .expect("dispatcher par gemv should succeed");
                        });
                    },
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon_disp = KernelDispatcher::with_tier(KernelTier::Neon);
            group.bench_with_input(BenchmarkId::new("neon", n_rows), &n_rows, |b, &nr| {
                b.iter(|| {
                    neon_disp
                        .gemv(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            nr,
                            k,
                        )
                        .expect("dispatcher gemv should succeed");
                });
            });

            if n_rows >= 64 {
                group.bench_with_input(
                    BenchmarkId::new("neon+parallel", n_rows),
                    &n_rows,
                    |b, &nr| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemv_1bit_g128_par(
                                &neon_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                nr,
                                k,
                            )
                            .expect("dispatcher par gemv should succeed");
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_gemm_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_dispatch");

    for &m in &[1, 4, 16] {
        let n_rows = 4096;
        let k = 4096;
        let blocks = make_blocks(n_rows * (k / 128));
        let input = vec![0.5f32; m * k];
        let mut output = vec![0.0f32; m * n_rows];

        let ref_disp = KernelDispatcher::with_tier(KernelTier::Reference);
        group.bench_with_input(
            BenchmarkId::new("reference", format!("m{m}")),
            &m,
            |b, &batch| {
                b.iter(|| {
                    ref_disp
                        .gemm(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            batch,
                            n_rows,
                            k,
                        )
                        .expect("dispatcher gemm should succeed");
                });
            },
        );

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx_disp = KernelDispatcher::with_tier(KernelTier::Avx2);
            group.bench_with_input(
                BenchmarkId::new("avx2", format!("m{m}")),
                &m,
                |b, &batch| {
                    b.iter(|| {
                        avx_disp
                            .gemm(
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher gemm should succeed");
                    });
                },
            );

            if m >= 4 {
                group.bench_with_input(
                    BenchmarkId::new("avx2+parallel", format!("m{m}")),
                    &m,
                    |b, &batch| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemm_1bit_g128_par(
                                &avx_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher par gemm should succeed");
                        });
                    },
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon_disp = KernelDispatcher::with_tier(KernelTier::Neon);
            group.bench_with_input(
                BenchmarkId::new("neon", format!("m{m}")),
                &m,
                |b, &batch| {
                    b.iter(|| {
                        neon_disp
                            .gemm(
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher gemm should succeed");
                    });
                },
            );

            if m >= 4 {
                group.bench_with_input(
                    BenchmarkId::new("neon+parallel", format!("m{m}")),
                    &m,
                    |b, &batch| {
                        b.iter(|| {
                            oxibonsai_kernels::parallel::gemm_1bit_g128_par(
                                &neon_disp,
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                batch,
                                n_rows,
                                k,
                            )
                            .expect("dispatcher par gemm should succeed");
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

// ─── FP8 benchmark helpers ────────────────────────────────────────────────────

/// Create `count` FP8 E4M3FN blocks filled with a representative pattern.
///
/// Uses `0x38` (≈ 1.0 in E4M3FN: exp=7, man=0 → 2^(7-7)×1.0 = 1.0) as the
/// weight value and scale 1.0, so dequantised values are all exactly 1.0.
fn make_fp8_e4m3_blocks(count: usize) -> Vec<BlockFP8E4M3> {
    (0..count)
        .map(|i| BlockFP8E4M3 {
            qs: [0x38u8; QK_FP8], // ≈ +1.0 in E4M3FN
            d: f16::from_f32(1.0 + (i as f32) * 0.001),
        })
        .collect()
}

/// Create `count` FP8 E5M2 blocks filled with a representative pattern.
///
/// Uses `0x3c` (≈ 1.0 in E5M2: exp=15, man=0 → 2^(15-15)×1.0 = 1.0) as the
/// weight value and scale 1.0.
fn make_fp8_e5m2_blocks(count: usize) -> Vec<BlockFP8E5M2> {
    (0..count)
        .map(|i| BlockFP8E5M2 {
            qs: [0x3cu8; QK_FP8], // ≈ +1.0 in E5M2
            d: f16::from_f32(1.0 + (i as f32) * 0.001),
        })
        .collect()
}

// ─── FP8 benchmarks ───────────────────────────────────────────────────────────

/// Benchmark FP8 dequantization and GEMV kernels.
///
/// Measures:
/// 1. Dequant E4M3: scalar (reference tier) vs best-available SIMD-auto.
/// 2. GEMV E4M3: scalar vs SIMD-auto (128 rows × 256 features).
/// 3. GEMV E4M3: parallel vs sequential (512 rows × 512 features — exercises rayon).
/// 4. GEMV E5M2: SIMD-auto (128 rows × 256 features).
///
/// Throughput is reported as element operations (multiply-accumulate) per second,
/// so criterion can display MOPS/GOPS in the benchmark report.
fn bench_fp8_kernels(c: &mut Criterion) {
    // ── Setup: shared dispatchers ──────────────────────────────────────────────
    let ref_dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
    let auto_dispatcher = KernelDispatcher::auto_detect();

    // ── 1. Dequant E4M3: 256 blocks (256 × 32 = 8192 weights) ────────────────
    {
        let n_blocks = 256usize;
        let blocks_e4m3 = make_fp8_e4m3_blocks(n_blocks);
        let mut output_deq = vec![0.0f32; n_blocks * QK_FP8];

        let mut group = c.benchmark_group("fp8_dequant_8192w");
        group.throughput(Throughput::Elements((n_blocks * QK_FP8) as u64));

        group.bench_function("e4m3/scalar", |b| {
            b.iter(|| {
                ref_dispatcher
                    .dequant_fp8_e4m3(black_box(&blocks_e4m3), black_box(&mut output_deq))
                    .expect("dequant_fp8_e4m3 should succeed");
            })
        });
        group.bench_function("e4m3/simd_auto", |b| {
            b.iter(|| {
                auto_dispatcher
                    .dequant_fp8_e4m3(black_box(&blocks_e4m3), black_box(&mut output_deq))
                    .expect("dequant_fp8_e4m3 should succeed");
            })
        });

        // E5M2 dequant for comparison
        let blocks_e5m2 = make_fp8_e5m2_blocks(n_blocks);
        group.bench_function("e5m2/simd_auto", |b| {
            b.iter(|| {
                auto_dispatcher
                    .dequant_fp8_e5m2(black_box(&blocks_e5m2), black_box(&mut output_deq))
                    .expect("dequant_fp8_e5m2 should succeed");
            })
        });
        group.finish();
    }

    // ── 2. GEMV E4M3: 128 rows × 256 features ────────────────────────────────
    {
        let n_rows = 128usize;
        let k = 256usize;
        let blocks_per_row = k / QK_FP8;
        let blocks_e4m3 = make_fp8_e4m3_blocks(n_rows * blocks_per_row);
        let input = vec![1.0f32; k];
        let mut output = vec![0.0f32; n_rows];

        let mut group = c.benchmark_group("fp8_gemv_128x256");
        // MACs = n_rows × k (each output element requires k multiply-accumulate ops).
        group.throughput(Throughput::Elements((n_rows * k) as u64));

        group.bench_function("e4m3/scalar", |b| {
            b.iter(|| {
                ref_dispatcher
                    .gemv_fp8_e4m3(
                        black_box(&blocks_e4m3),
                        black_box(&input),
                        black_box(&mut output),
                        n_rows,
                        k,
                    )
                    .expect("gemv_fp8_e4m3 should succeed");
            })
        });
        group.bench_function("e4m3/simd_auto", |b| {
            b.iter(|| {
                auto_dispatcher
                    .gemv_fp8_e4m3(
                        black_box(&blocks_e4m3),
                        black_box(&input),
                        black_box(&mut output),
                        n_rows,
                        k,
                    )
                    .expect("gemv_fp8_e4m3 should succeed");
            })
        });

        // E5M2 for comparison under same dimensions.
        let blocks_e5m2 = make_fp8_e5m2_blocks(n_rows * blocks_per_row);
        group.bench_function("e5m2/simd_auto", |b| {
            b.iter(|| {
                auto_dispatcher
                    .gemv_fp8_e5m2(
                        black_box(&blocks_e5m2),
                        black_box(&input),
                        black_box(&mut output),
                        n_rows,
                        k,
                    )
                    .expect("gemv_fp8_e5m2 should succeed");
            })
        });
        group.finish();
    }

    // ── 3. GEMV E4M3/E5M2 parallel vs sequential: 512 rows × 512 features ───
    //
    // 512 rows is above PAR_GEMV_MIN_ROWS=64, so gemv_fp8_e4m3_par will use rayon.
    // This benchmark isolates the parallel speedup for larger matrices.
    {
        let n_rows_large = 512usize;
        let k_large = 512usize;
        let blocks_per_row = k_large / QK_FP8;
        let blocks_large_e4m3 = make_fp8_e4m3_blocks(n_rows_large * blocks_per_row);
        let blocks_large_e5m2 = make_fp8_e5m2_blocks(n_rows_large * blocks_per_row);
        let input_large = vec![1.0f32; k_large];
        let mut output_large = vec![0.0f32; n_rows_large];

        let mut group = c.benchmark_group("fp8_gemv_512x512");
        group.throughput(Throughput::Elements((n_rows_large * k_large) as u64));

        group.bench_function("e4m3/sequential", |b| {
            b.iter(|| {
                auto_dispatcher
                    .gemv_fp8_e4m3(
                        black_box(&blocks_large_e4m3),
                        black_box(&input_large),
                        black_box(&mut output_large),
                        n_rows_large,
                        k_large,
                    )
                    .expect("sequential gemv should succeed");
            })
        });
        group.bench_function("e4m3/parallel", |b| {
            b.iter(|| {
                gemv_fp8_e4m3_par(
                    &auto_dispatcher,
                    black_box(&blocks_large_e4m3),
                    black_box(&input_large),
                    black_box(&mut output_large),
                    n_rows_large,
                    k_large,
                )
                .expect("parallel gemv should succeed");
            })
        });
        group.bench_function("e5m2/sequential", |b| {
            b.iter(|| {
                auto_dispatcher
                    .gemv_fp8_e5m2(
                        black_box(&blocks_large_e5m2),
                        black_box(&input_large),
                        black_box(&mut output_large),
                        n_rows_large,
                        k_large,
                    )
                    .expect("sequential e5m2 gemv should succeed");
            })
        });
        group.bench_function("e5m2/parallel", |b| {
            b.iter(|| {
                gemv_fp8_e5m2_par(
                    &auto_dispatcher,
                    black_box(&blocks_large_e5m2),
                    black_box(&input_large),
                    black_box(&mut output_large),
                    n_rows_large,
                    k_large,
                )
                .expect("parallel e5m2 gemv should succeed");
            })
        });
        group.finish();
    }

    // ── 4. GEMV dispatch sweep: n_rows ∈ {32, 256, 1024} × k=256 ─────────────
    {
        let k = 256usize;
        let mut group = c.benchmark_group("fp8_gemv_dispatch");

        for &n_rows in &[32usize, 256, 1024] {
            let blocks_per_row = k / QK_FP8;
            let blocks = make_fp8_e4m3_blocks(n_rows * blocks_per_row);
            let input = vec![0.5f32; k];
            let mut output = vec![0.0f32; n_rows];

            group.throughput(Throughput::Elements((n_rows * k) as u64));

            group.bench_with_input(
                BenchmarkId::new("e4m3/scalar", n_rows),
                &n_rows,
                |b, &nr| {
                    b.iter(|| {
                        ref_dispatcher
                            .gemv_fp8_e4m3(
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                nr,
                                k,
                            )
                            .expect("gemv should succeed");
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("e4m3/simd_auto", n_rows),
                &n_rows,
                |b, &nr| {
                    b.iter(|| {
                        auto_dispatcher
                            .gemv_fp8_e4m3(
                                black_box(&blocks),
                                black_box(&input),
                                black_box(&mut output),
                                nr,
                                k,
                            )
                            .expect("gemv should succeed");
                    })
                },
            );
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_dequant,
    bench_gemv_single_row,
    bench_gemv_dispatch,
    bench_gemm_dispatch,
    bench_fp8_kernels,
);
criterion_main!(benches);
