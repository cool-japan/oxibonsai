use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::f16;
use oxibonsai_core::{
    transcode_ptq1_0_to_tq2, BlockPQ2_0, BlockPTQ1_0, BlockTQ2_0_g128, QK_TQ2_0_G128,
};
use oxibonsai_kernels::dequant_prism::gemv_pq2_0;
use oxibonsai_kernels::dispatch::KernelDispatcher;
use oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV;
use oxibonsai_kernels::gemv_ptq1::gemv_ptq1_0;
use oxibonsai_kernels::traits::{PrismKernel, TernaryKernel};
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

/// Batch sizes the ternary GEMM is benchmarked at: `m` in {1, 4, 8, 64, 512}.
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

// ─── PTQ1_0 / PQ2_0 vs TQ2_0_g128 GEMV comparison ───────────────────────────

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

/// The acceptance criterion that `gemv_ptq1_0` stays within 15% of
/// `gemv_tq2_0_g128`, made visible in tracked `cargo bench` output. The
/// pass/fail assertion itself lives in `gemv_ptq1.rs`'s `#[ignore]`d
/// `prism_ptq1_0_gemv_within_15pct_of_tq2` (wall-clock ratio assertions are
/// unreliable on a shared build machine and must never gate CI); this
/// benchmark is the criterion form, so the comparison is measured on every
/// `cargo bench` run, not only on demand.
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

// ─── Parallel efficiency at the Bonsai-2 LM-head shape ─────────────────────

/// `token_embd.weight` / `output.weight`'s shape for Bonsai 2 27B
/// (`n_rows=248320` (vocab), `k=5120` (embedding_length), from the real GGUF
/// header). The 4.28-4.64x speedup on 8 cores going from the direct to the
/// adaptive tiled path is also checked by an `#[ignore]`d in-crate `Instant`
/// timing test; this is its criterion form, tracked on every `cargo bench`
/// run.
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

// ─── INT8 dot-product tier vs the f32 default, on `gemv_pq2_0` (K-14) ──────

/// A `PQ2_0`-quantized weight matrix of `n_rows x k`, built from
/// [`random_ternary_weights`] so its `PQ2_0` encoding is lossless (exactly
/// representable ternary values) and comparable across shapes.
fn make_pq2_0_blocks(n_rows: usize, k: usize) -> Vec<BlockPQ2_0> {
    let weights = random_ternary_weights(n_rows, k);
    BlockPQ2_0::quantize(&weights).expect("PQ2_0 quantize should succeed for a block-aligned input")
}

/// Runs `dispatcher.gemv_pq2_0` under a specific [`KERNEL_TIER_ENV`] value
/// (or unset for the f32 default), restoring whatever the variable held
/// before this call on the way out — this bench binary is the only reader
/// of the variable in its own process, but leaving a stray override behind
/// for a later group in the same run would silently change what "f32" means
/// for it.
///
/// # Safety contract
///
/// `std::env::set_var`/`remove_var` are `unsafe fn` (edition 2024) because a
/// concurrent `std::env::var` on ANY key can observe a torn `environ` while
/// another thread mutates it. This bench binary's `main` (criterion's
/// generated entry point) runs every benchmark function to completion,
/// sequentially, on one thread before the process exits, so no other reader
/// of the environment is ever concurrent with these calls.
fn with_kernel_tier_env<R>(tier: Option<&str>, f: impl FnOnce() -> R) -> R {
    let prior = std::env::var(KERNEL_TIER_ENV).ok();
    // SAFETY: see this function's doc comment — no concurrent env reader
    // exists in this single-threaded benchmark binary.
    unsafe {
        match tier {
            Some(t) => std::env::set_var(KERNEL_TIER_ENV, t),
            None => std::env::remove_var(KERNEL_TIER_ENV),
        }
    }
    let result = f();
    // SAFETY: same contract as above.
    unsafe {
        match &prior {
            Some(p) => std::env::set_var(KERNEL_TIER_ENV, p),
            None => std::env::remove_var(KERNEL_TIER_ENV),
        }
    }
    result
}

/// K-14's criterion form: the INT8 `SDOT` dot-product tier
/// (`OXIBONSAI_KERNEL_TIER=neon-dot`, aarch64-only — a no-op tier-selection
/// on other architectures, so this group degenerates to f32-vs-f32 there
/// rather than failing) against the f32 default, for `gemv_pq2_0`, through
/// the public [`KernelDispatcher`]/[`TernaryKernel`] API only — never a
/// direct call into `oxibonsai_kernels::dispatch_int8`'s own functions, so
/// the bench measures what callers reach and survives internal changes to
/// how the INT8 tier is dispatched.
///
/// Two shapes: the real Bonsai 2 27B `ffn_up` matrix (`[5120, 17408]`,
/// `embedding_length` x `feed_forward_length` from the real GGUF header),
/// and a smaller, 1.7B-scale shape (`[2048, 5504]`) for comparison — not
/// this repository's own `Ternary-Bonsai-1.7B.gguf` (that file predates
/// `PQ2_0`; a `PQ2_0` tensor of its exact shape does not exist on disk), a
/// representative smaller matrix at the same order of magnitude.
fn bench_int8_gemv_pq2_0(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let mut group = c.benchmark_group("int8_gemv");
    group.sample_size(10);

    for (label, n_rows, k) in [
        ("bonsai2_27b_ffn_up", 17408usize, 5120usize),
        ("1_7b_scale", 5504usize, 2048usize),
    ] {
        let blocks = make_pq2_0_blocks(n_rows, k);
        let input = vec![0.1f32; k];
        let mut output = vec![0.0f32; n_rows];
        group.throughput(Throughput::Elements((n_rows * k) as u64));

        group.bench_with_input(BenchmarkId::new("f32", label), &(), |b, ()| {
            b.iter(|| {
                with_kernel_tier_env(None, || {
                    dispatcher
                        .gemv_pq2_0(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            n_rows,
                            k,
                        )
                        .expect("gemv_pq2_0 (f32 default) should succeed");
                });
            });
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("neon-dot", label), &(), |b, ()| {
            b.iter(|| {
                with_kernel_tier_env(Some("neon-dot"), || {
                    dispatcher
                        .gemv_pq2_0(
                            black_box(&blocks),
                            black_box(&input),
                            black_box(&mut output),
                            n_rows,
                            k,
                        )
                        .expect("gemv_pq2_0 (neon-dot INT8 tier) should succeed");
                });
            });
        });
    }

    group.finish();
}

// ─── Dense FP32 GEMM vs the per-row GEMV loop ───────────────────────────────

/// Deterministic pseudo-random `f32` values in `[-0.5, 0.5)`.
fn random_f32_values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 40) as u32 as f32) / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// [`oxibonsai_kernels::gemm_f32`] against the loop it replaces — one
/// [`oxibonsai_kernels::gemv_f32`] per row of the input — at the two shapes a
/// ViT block runs (`k = 1152 -> n = 4304`, the MLP up-projection, and
/// `k = 4304 -> n = 1152`, the down-projection) for `m` = 1 (a single row,
/// which `gemm_f32` runs as the same GEMV, so the two should tie), 16, 64 and
/// 576 (a full image's worth of patches).
///
/// The two are bit-identical (asserted once per shape before timing, so the
/// bench can never time an implementation that computes something else), and
/// the throughput is declared in floating-point operations (two per
/// multiply-accumulate), so criterion's `Gelem/s` reads as GFLOP/s.
fn bench_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f32");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for (k, n) in [(1152usize, 4304usize), (4304, 1152)] {
        let weights = random_f32_values(n * k, 1);
        for m in [1usize, 16, 64, 576] {
            let input = random_f32_values(m * k, 2);
            let mut output = vec![0.0f32; m * n];
            let mut reference = vec![0.0f32; m * n];
            let per_row = |out: &mut [f32]| {
                for i in 0..m {
                    oxibonsai_kernels::gemv_f32(
                        black_box(&weights),
                        black_box(&input[i * k..(i + 1) * k]),
                        black_box(&mut out[i * n..(i + 1) * n]),
                        n,
                        k,
                    )
                    .expect("gemv_f32 should succeed on well-formed shapes");
                }
            };

            per_row(&mut reference);
            oxibonsai_kernels::gemm_f32(&input, &weights, None, m, k, n, &mut output)
                .expect("gemm_f32 should succeed on well-formed shapes");
            assert!(
                output
                    .iter()
                    .zip(&reference)
                    .all(|(g, r)| g.to_bits() == r.to_bits()),
                "gemm_f32 must be bit-identical to the per-row gemv_f32 loop (m={m}, k={k}, n={n})"
            );

            let label = format!("m{m}_k{k}_n{n}");
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));
            group.bench_function(BenchmarkId::new("blocked_rayon", &label), |b| {
                b.iter(|| {
                    oxibonsai_kernels::gemm_f32(
                        black_box(&input),
                        black_box(&weights),
                        None,
                        m,
                        k,
                        n,
                        black_box(&mut output),
                    )
                    .expect("gemm_f32 should succeed on well-formed shapes");
                });
            });
            group.bench_function(BenchmarkId::new("per_row_gemv", &label), |b| {
                b.iter(|| per_row(black_box(&mut reference)));
            });
        }
    }
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
    bench_int8_gemv_pq2_0,
    bench_gemm_f32,
);
criterion_main!(benches);
