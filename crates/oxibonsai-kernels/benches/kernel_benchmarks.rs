use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::f16;
use oxibonsai_core::{
    transcode_ptq1_0_to_tq2, BlockPQ2_0, BlockPTQ1_0, BlockTQ2_0_g128, QK_TQ2_0_G128,
};
use oxibonsai_kernels::dequant_prism::gemv_pq2_0;
use oxibonsai_kernels::dispatch::KernelDispatcher;
use oxibonsai_kernels::gemv_ptq1::gemv_ptq1_0;
use oxibonsai_kernels::traits::TernaryKernel;
use oxibonsai_kernels::{cpu_kernel_tier, KernelTier};
use std::hint::black_box;
use std::time::Duration;

fn make_ternary_blocks(n_rows: usize, k: usize) -> Vec<BlockTQ2_0_g128> {
    let blocks_per_row = k / QK_TQ2_0_G128;
    (0..n_rows * blocks_per_row)
        .map(|i| BlockTQ2_0_g128 {
            qs: std::array::from_fn(|j| ((i * 37 + j * 13) & 0xFF) as u8),
            d: f16::from_f32(0.5 + i as f32 * 0.001),
        })
        .collect()
}

fn bench_dequant_ternary(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let mut group = c.benchmark_group("dequant_ternary_g128");
    for (label, n_blocks) in [("1x128", 1usize), ("2048x128", 2048usize)] {
        let blocks = make_ternary_blocks(n_blocks, QK_TQ2_0_G128);
        let mut output = vec![0.0f32; n_blocks * QK_TQ2_0_G128];
        group.bench_with_input(BenchmarkId::new("kernel", label), &n_blocks, |b, _| {
            b.iter(|| {
                dispatcher
                    .dequant_ternary_g128(black_box(&blocks), black_box(&mut output))
                    .expect(
                        "dequant_ternary_g128 should succeed with valid blocks and output buffer",
                    );
            });
        });
    }
    group.finish();
}

fn bench_gemv_ternary(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 4096usize;
    let mut group = c.benchmark_group("gemv_ternary_g128");
    for n_rows in [2048usize, 6144usize] {
        let blocks = make_ternary_blocks(n_rows, k);
        let input = vec![0.1f32; k];
        let mut output = vec![0.0f32; n_rows];
        group.bench_with_input(BenchmarkId::new("direct", n_rows), &n_rows, |b, _| {
            b.iter(|| {
                dispatcher
                    .gemv_ternary_g128(
                        black_box(&blocks),
                        black_box(&input),
                        black_box(&mut output),
                        n_rows,
                        k,
                    )
                    .expect("gemv_ternary_g128 should succeed with valid ternary blocks and matching dimensions");
            });
        });
    }
    group.finish();
}

fn bench_gemv_ternary_par(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 4096usize;
    let mut group = c.benchmark_group("gemv_ternary_g128_par");
    for n_rows in [2048usize, 6144usize] {
        let blocks = make_ternary_blocks(n_rows, k);
        let input = vec![0.1f32; k];
        let mut output = vec![0.0f32; n_rows];
        group.bench_with_input(BenchmarkId::new("adaptive", n_rows), &n_rows, |b, _| {
            b.iter(|| {
                oxibonsai_kernels::gemv_adaptive_ternary(
                    black_box(&dispatcher),
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    n_rows,
                    k,
                )
                .expect("gemv_adaptive_ternary should succeed with valid ternary blocks and matching dimensions");
            });
        });
    }
    group.finish();
}

/// Batch sizes the ternary GEMM is benchmarked at (PERF-CPU-PREFILL
/// ACCEPTANCE: "a criterion bench for gemm at m in {1, 4, 8, 64, 512}").
///
/// The sweep spans the whole K-18 regression surface: `m = 1` is a decode
/// step (a plain GEMV, no reuse to win), `m = 4`/`8` bracket
/// [`oxibonsai_kernels::gemm_ternary::TERNARY_GEMM_MR`], and `m = 64`/`512`
/// are prefill micro-batches, where the pre-K-18 loop-of-GEMVs streamed the
/// whole quantized weight matrix once per row.
const GEMM_M_SWEEP: &[usize] = &[1, 4, 8, 64, 512];

/// Above this `m`, skip the scalar `Reference` tier: at `n_rows = 2048`,
/// `k = 4096`, `m = 512` is ~4.3 G MACs per call and the unaccelerated tier
/// would spend seconds per sample for no signal the SIMD tiers do not give.
const GEMM_REFERENCE_MAX_M: usize = 16;

fn bench_gemm_ternary_par(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 4096usize;
    let n_rows = 2048usize;
    let mut group = c.benchmark_group("gemm_ternary_g128_par");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    let blocks = make_ternary_blocks(n_rows, k);
    for &batch in GEMM_M_SWEEP {
        let input = vec![0.1f32; batch * k];
        let mut output = vec![0.0f32; batch * n_rows];
        group.throughput(Throughput::Elements((batch * n_rows * k) as u64));
        group.bench_function(
            BenchmarkId::new("adaptive", format!("m{batch}_{n_rows}rows_{k}k")),
            |b| {
                b.iter(|| {
                    oxibonsai_kernels::gemm_adaptive_ternary(
                        black_box(&dispatcher),
                        black_box(&blocks),
                        black_box(&input),
                        black_box(&mut output),
                        batch,
                        n_rows,
                        k,
                    )
                    .expect("gemm_adaptive_ternary should succeed with valid ternary blocks and matching batch/matrix dimensions");
                });
            },
        );
    }
    group.finish();
}

/// The same `m` sweep straight through [`KernelDispatcher::gemm_ternary_g128`],
/// one benchmark per constructible CPU tier.
///
/// `bench_gemm_ternary_par` above measures the adaptive driver (which may
/// parallelise over row slabs); this measures the dispatcher's own GEMM, so
/// the `Reference`-vs-native gap is visible rather than blended with Rayon.
fn bench_gemm_dispatch(c: &mut Criterion) {
    let k = 4096usize;
    let n_rows = 2048usize;
    let mut group = c.benchmark_group("gemm_dispatch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    let blocks = make_ternary_blocks(n_rows, k);

    let native = cpu_kernel_tier();
    let mut tiers = vec![("reference", KernelTier::Reference)];
    if native != KernelTier::Reference {
        tiers.push(("native_cpu", native));
    }

    for &batch in GEMM_M_SWEEP {
        let input = vec![0.1f32; batch * k];
        let mut output = vec![0.0f32; batch * n_rows];
        group.throughput(Throughput::Elements((batch * n_rows * k) as u64));
        for &(label, tier) in &tiers {
            if tier == KernelTier::Reference && batch > GEMM_REFERENCE_MAX_M {
                continue;
            }
            let dispatcher = KernelDispatcher::with_tier(tier);
            group.bench_function(BenchmarkId::new(label, format!("m{batch}")), |b| {
                b.iter(|| {
                    dispatcher
                        .gemm_ternary_g128(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            batch,
                            n_rows,
                            k,
                        )
                        .expect("gemm_ternary_g128 should succeed with valid ternary blocks and matching dimensions");
                });
            });
        }
    }
    group.finish();
}

// ─── PTQ1_0 / PQ2_0 vs TQ2_0_g128 GEMV comparison (B2-03) ───────────────────

/// A deterministic ternary-valued (-1/0/+1) weight vector, fixed-seed LCG
/// so every format quantizes *the same* logical weights: this makes the
/// three formats' GEMV outputs mutually comparable (and, since all three
/// share the ternary domain, numerically equal within FP tolerance), not
/// just timed on unrelated random bit patterns.
fn random_ternary_weights(n_rows: usize, k: usize) -> Vec<f32> {
    let mut state: u32 = 0x1357_9BDF;
    (0..n_rows * k)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match (state >> 11) % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            }
        })
        .collect()
}

/// B2-03's acceptance criterion ("`gemv_ptq1_0` within 15% of
/// `gemv_tq2_0_g128`"), made visible in tracked `cargo bench` output. The
/// pass/fail assertion itself lives in `gemv_ptq1.rs`'s `#[ignore]`d
/// `prism_ptq1_0_gemv_within_15pct_of_tq2` (wall-clock ratio assertions are
/// unreliable on this shared, multi-agent build machine per session
/// CONTEXT.md and must never gate CI); this benchmark is the missing
/// criterion form so the comparison is measured on every `cargo bench` run,
/// not only on demand.
fn bench_gemv_ptq1_0_vs_ternary(c: &mut Criterion) {
    let k = 4096usize;
    let mut group = c.benchmark_group("gemv_ptq1_0_vs_tq2_0_g128");

    for n_rows in [2048usize, 6144usize] {
        let weights = random_ternary_weights(n_rows, k);
        let ptq_blocks = BlockPTQ1_0::quantize(&weights)
            .expect("PTQ1_0 quantize should succeed for a block-aligned ternary input");
        let tq_blocks = transcode_ptq1_0_to_tq2(&ptq_blocks);
        let input = vec![0.1f32; k];
        let mut out_tq2 = vec![0.0f32; n_rows];
        let mut out_ptq = vec![0.0f32; n_rows];
        let dispatcher = KernelDispatcher::auto_detect();

        // Make this module doc's "numerically equal within FP tolerance"
        // claim load-bearing: compute both once (untimed) and assert
        // max-abs-diff, so a `TQ2_0_g128`/`PTQ1_0` byte-layout regression
        // (K-06, d-first vs qs-first) fails the bench itself instead of
        // only showing up as an unremarked wall-clock number. Measured
        // max diff on this fixture (k=4096, constant 0.1 input) is a
        // single-ULP-scale ~3.8e-6 (2^-18) at both swept `n_rows` on the
        // NEON tier (`dispatcher.gemv_ternary_g128` auto-dispatches to
        // whatever tier this machine has); the AVX2/AVX-512 tiers
        // accumulate in a different order and have not been measured, so a
        // future x86 failure here should first be read as "tolerance too
        // tight for that tier's rounding", not assumed to be a layout
        // regression. 1e-3 leaves ~260x margin over the NEON reading.
        dispatcher
            .gemv_ternary_g128(&tq_blocks, &input, &mut out_tq2, n_rows, k)
            .expect("gemv_ternary_g128 parity precheck should succeed");
        gemv_ptq1_0(&ptq_blocks, &input, &mut out_ptq, n_rows, k)
            .expect("gemv_ptq1_0 parity precheck should succeed");
        let max_diff = out_tq2
            .iter()
            .zip(out_ptq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "gemv_ptq1_0_vs_tq2_0_g128 parity precheck (n_rows={n_rows}): tq2_0_g128 and \
             PTQ1_0->TQ2 outputs diverged (max diff {max_diff}) -- both decode the same \
             `random_ternary_weights` (PTQ1_0 transcoded losslessly to TQ2), so a real \
             divergence indicates a byte-layout or decode regression (K-06), not benign FP noise"
        );

        group.bench_with_input(
            BenchmarkId::new("tq2_0_g128/dispatch_auto", n_rows),
            &n_rows,
            |b, _| {
                b.iter(|| {
                    dispatcher
                        .gemv_ternary_g128(
                            black_box(&tq_blocks),
                            black_box(&input),
                            black_box(&mut out_tq2),
                            n_rows,
                            k,
                        )
                        .expect("gemv_ternary_g128 should succeed");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ptq1_0/scalar", n_rows),
            &n_rows,
            |b, _| {
                b.iter(|| {
                    gemv_ptq1_0(
                        black_box(&ptq_blocks),
                        black_box(&input),
                        black_box(&mut out_ptq),
                        n_rows,
                        k,
                    )
                    .expect("gemv_ptq1_0 should succeed");
                });
            },
        );

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("ptq1_0/neon", n_rows), &n_rows, |b, _| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_prism_neon::gemv_ptq1_0_neon(
                    black_box(&ptq_blocks),
                    black_box(&input),
                    black_box(&mut out_ptq),
                    n_rows,
                    k,
                )
                .expect("gemv_ptq1_0_neon should succeed");
            });
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("ptq1_0/avx2", n_rows), &n_rows, |b, _| {
                b.iter(|| unsafe {
                    oxibonsai_kernels::simd_prism_avx2::gemv_ptq1_0_avx2(
                        black_box(&ptq_blocks),
                        black_box(&input),
                        black_box(&mut out_ptq),
                        n_rows,
                        k,
                    )
                    .expect("gemv_ptq1_0_avx2 should succeed");
                });
            });
        }
    }

    group.finish();
}

/// `PQ2_0` (ggml id 142, `d`-first byte layout) alongside the same
/// ternary/`PTQ1_0` comparison: quantizing the *same* [`random_ternary_weights`]
/// through `BlockPQ2_0::quantize` and `BlockTQ2_0_g128::quantize`
/// independently (rather than transcoding, which `PQ2_0` has no dedicated
/// path for) keeps the comparison apples-to-apples without conflating
/// `PQ2_0`'s `d`-first layout with legacy `TQ2_0_g128`'s `qs`-first one
/// (K-06).
fn bench_gemv_pq2_0_vs_ternary(c: &mut Criterion) {
    let k = 4096usize;
    let mut group = c.benchmark_group("gemv_pq2_0_vs_tq2_0_g128");

    for n_rows in [2048usize, 6144usize] {
        let weights = random_ternary_weights(n_rows, k);
        let tq_blocks = BlockTQ2_0_g128::quantize(&weights)
            .expect("TQ2_0_g128 quantize should succeed for a block-aligned input");
        let pq_blocks = BlockPQ2_0::quantize(&weights)
            .expect("PQ2_0 quantize should succeed for a block-aligned input");
        let input = vec![0.1f32; k];
        let mut out_tq2 = vec![0.0f32; n_rows];
        let mut out_pq2 = vec![0.0f32; n_rows];
        let dispatcher = KernelDispatcher::auto_detect();

        // Same load-bearing parity precheck as `bench_gemv_ptq1_0_vs_ternary`
        // above, for the independently-quantized `PQ2_0` comparison: `PQ2_0`
        // and `TQ2_0_g128` quantize the *same* `random_ternary_weights` via
        // different (but, on exactly-ternary input, mathematically
        // equivalent) rounding rules, so their GEMV outputs should still
        // agree; a real divergence would indicate a `d`-first vs `qs`-first
        // byte-layout mismatch (K-06) or a decode regression. Same NEON-only
        // measurement caveat as `bench_gemv_ptq1_0_vs_ternary`'s precheck
        // above: 1e-3 was set from a NEON-tier reading, not measured on
        // AVX2/AVX-512.
        dispatcher
            .gemv_ternary_g128(&tq_blocks, &input, &mut out_tq2, n_rows, k)
            .expect("gemv_ternary_g128 parity precheck should succeed");
        gemv_pq2_0(&pq_blocks, &input, &mut out_pq2, n_rows, k)
            .expect("gemv_pq2_0 parity precheck should succeed");
        let max_diff = out_tq2
            .iter()
            .zip(out_pq2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "gemv_pq2_0_vs_tq2_0_g128 parity precheck (n_rows={n_rows}): tq2_0_g128 and PQ2_0 \
             outputs diverged (max diff {max_diff}) -- both independently quantize the same \
             `random_ternary_weights`, so a real divergence indicates a byte-layout mismatch \
             (K-06) or decode regression, not benign FP noise"
        );

        group.bench_with_input(
            BenchmarkId::new("tq2_0_g128/dispatch_auto", n_rows),
            &n_rows,
            |b, _| {
                b.iter(|| {
                    dispatcher
                        .gemv_ternary_g128(
                            black_box(&tq_blocks),
                            black_box(&input),
                            black_box(&mut out_tq2),
                            n_rows,
                            k,
                        )
                        .expect("gemv_ternary_g128 should succeed");
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("pq2_0/scalar", n_rows), &n_rows, |b, _| {
            b.iter(|| {
                gemv_pq2_0(
                    black_box(&pq_blocks),
                    black_box(&input),
                    black_box(&mut out_pq2),
                    n_rows,
                    k,
                )
                .expect("gemv_pq2_0 should succeed");
            });
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("pq2_0/neon", n_rows), &n_rows, |b, _| {
            b.iter(|| unsafe {
                oxibonsai_kernels::simd_prism_neon::gemv_pq2_0_neon(
                    black_box(&pq_blocks),
                    black_box(&input),
                    black_box(&mut out_pq2),
                    n_rows,
                    k,
                )
                .expect("gemv_pq2_0_neon should succeed");
            });
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            group.bench_with_input(BenchmarkId::new("pq2_0/avx2", n_rows), &n_rows, |b, _| {
                b.iter(|| unsafe {
                    oxibonsai_kernels::simd_prism_avx2::gemv_pq2_0_avx2(
                        black_box(&pq_blocks),
                        black_box(&input),
                        black_box(&mut out_pq2),
                        n_rows,
                        k,
                    )
                    .expect("gemv_pq2_0_avx2 should succeed");
                });
            });
        }
    }

    group.finish();
}

// ─── KERN-PARALLEL: parallel-efficiency at the Bonsai-2 LM-head shape ──────

/// `token_embd.weight` / `output.weight`'s shape for Bonsai 2 27B
/// (`n_rows=248320` (vocab), `k=5120` (embedding_length) — see CONTEXT.md's
/// model facts). The substance of KERN-PARALLEL's finding (a 4.28-4.64x
/// speedup on 8 cores going from direct to the adaptive tiled path) was
/// previously only checked by an `#[ignore]`d in-crate `Instant` timing
/// test; this is the missing criterion form, tracked on every `cargo
/// bench` run.
fn bench_gemv_parallel_efficiency_lm_head(c: &mut Criterion) {
    const N_ROWS: usize = 248_320;
    const K: usize = 5120;

    let blocks = make_ternary_blocks(N_ROWS, K);
    let input = vec![0.1f32; K];
    let mut output = vec![0.0f32; N_ROWS];
    let dispatcher = KernelDispatcher::auto_detect();

    let mut group = c.benchmark_group("gemv_ternary_lm_head_248320x5120");
    group.sample_size(10);
    group.throughput(Throughput::Elements((N_ROWS * K) as u64));

    group.bench_function("direct", |b| {
        b.iter(|| {
            dispatcher
                .gemv_ternary_g128(
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    N_ROWS,
                    K,
                )
                .expect("gemv_ternary_g128 should succeed");
        });
    });

    group.bench_function("row_parallel", |b| {
        b.iter(|| {
            oxibonsai_kernels::gemv_ternary_g128_par(
                &dispatcher,
                black_box(&blocks),
                black_box(&input),
                black_box(&mut output),
                N_ROWS,
                K,
            )
            .expect("gemv_ternary_g128_par should succeed");
        });
    });

    group.bench_function("adaptive_tiled", |b| {
        b.iter(|| {
            oxibonsai_kernels::gemv_adaptive_ternary(
                &dispatcher,
                black_box(&blocks),
                black_box(&input),
                black_box(&mut output),
                N_ROWS,
                K,
            )
            .expect("gemv_adaptive_ternary should succeed");
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dequant_ternary,
    bench_gemv_ternary,
    bench_gemv_ternary_par,
    bench_gemm_ternary_par,
    bench_gemm_dispatch,
    bench_gemv_ptq1_0_vs_ternary,
    bench_gemv_pq2_0_vs_ternary,
    bench_gemv_parallel_efficiency_lm_head,
);
criterion_main!(benches);
