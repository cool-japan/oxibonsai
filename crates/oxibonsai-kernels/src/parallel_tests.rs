use super::*;
use half::f16;

fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
    BlockQ1_0G128 {
        d: f16::from_f32(scale),
        qs: bits,
    }
}

fn make_test_data(n_rows: usize, k: usize) -> (Vec<BlockQ1_0G128>, Vec<f32>) {
    let blocks_per_row = k / QK1_0_G128;
    let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);
    for row in 0..n_rows {
        for bi in 0..blocks_per_row {
            let bits = [((row * 37 + bi * 13) & 0xFF) as u8; 16];
            blocks.push(make_block(0.5 + (row as f32) * 0.01, bits));
        }
    }
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
    (blocks, input)
}

fn make_ternary_block(qs: [u8; 32]) -> oxibonsai_core::BlockTQ2_0_g128 {
    oxibonsai_core::BlockTQ2_0_g128 { qs, d: f16::ONE }
}

#[test]
fn parallel_thresholds_are_platform_tuned() {
    // Regression guard for the tuning-wiring fix: the parallel dispatch
    // thresholds must be sourced from the auto-detected `TunedThresholds`,
    // not a hardcoded constant, so they adapt per platform (core count /
    // cache / SIMD tier).
    let tuned = PlatformProfile::global_thresholds();
    assert_eq!(par_gemv_min_rows(), tuned.par_gemv_min_rows);
    assert_eq!(par_gemm_min_batch(), tuned.par_gemm_min_batch);
}

#[test]
fn par_gemv_matches_sequential() {
    // 512 rows is above the platform-tuned `par_gemv_min_rows()` on every
    // supported platform (max tuned threshold is 448), guaranteeing the
    // parallel path — not the sequential fallback — is exercised here.
    let n_rows = 512;
    let k = 256;
    let (blocks, input) = make_test_data(n_rows, k);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_seq = vec![0.0f32; n_rows];
    let mut out_par = vec![0.0f32; n_rows];

    dispatcher
        .gemv(&blocks, &input, &mut out_seq, n_rows, k)
        .expect("sequential gemv should succeed");
    gemv_1bit_g128_par(&dispatcher, &blocks, &input, &mut out_par, n_rows, k)
        .expect("parallel gemv should succeed");

    for i in 0..n_rows {
        assert!(
            (out_seq[i] - out_par[i]).abs() < 0.01,
            "row {i}: seq={}, par={}",
            out_seq[i],
            out_par[i]
        );
    }
}

#[test]
fn par_gemv_small_is_sequential() {
    let n_rows = 4; // Below threshold
    let k = 128;
    let (blocks, input) = make_test_data(n_rows, k);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_seq = vec![0.0f32; n_rows];
    let mut out_par = vec![0.0f32; n_rows];

    dispatcher
        .gemv(&blocks, &input, &mut out_seq, n_rows, k)
        .expect("sequential gemv should succeed");
    gemv_1bit_g128_par(&dispatcher, &blocks, &input, &mut out_par, n_rows, k)
        .expect("parallel gemv should succeed");

    for i in 0..n_rows {
        assert!(
            (out_seq[i] - out_par[i]).abs() < f32::EPSILON,
            "row {i}: seq={}, par={}",
            out_seq[i],
            out_par[i]
        );
    }
}

#[test]
fn par_gemm_matches_sequential() {
    // 16 is above the platform-tuned `par_gemm_min_batch()` on every
    // supported platform (max tuned threshold is 16), exercising the
    // parallel batch path.
    let m = 16;
    let n_rows = 16;
    let k = 128;
    let blocks_per_row = k / QK1_0_G128;
    let mut blocks = Vec::new();
    for ni in 0..n_rows {
        for bi in 0..blocks_per_row {
            let bits = [((ni * 17 + bi * 7) & 0xFF) as u8; 16];
            blocks.push(make_block(1.0 + ni as f32 * 0.2, bits));
        }
    }
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.005) - 0.32).collect();
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_seq = vec![0.0f32; m * n_rows];
    let mut out_par = vec![0.0f32; m * n_rows];

    dispatcher
        .gemm(&blocks, &input, &mut out_seq, m, n_rows, k)
        .expect("sequential gemm should succeed");
    gemm_1bit_g128_par(&dispatcher, &blocks, &input, &mut out_par, m, n_rows, k)
        .expect("parallel gemm should succeed");

    for i in 0..(m * n_rows) {
        assert!(
            (out_seq[i] - out_par[i]).abs() < 0.01,
            "idx {i}: seq={}, par={}",
            out_seq[i],
            out_par[i]
        );
    }
}

#[test]
fn par_ternary_gemv_matches_sequential() -> KernelResult<()> {
    let n_rows = 128;
    let k = 256;
    let blocks_per_row = k / QK_TQ2_0_G128;
    let blocks = vec![make_ternary_block([0xAAu8; 32]); n_rows * blocks_per_row];
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_seq = vec![0.0f32; n_rows];
    let mut out_par = vec![0.0f32; n_rows];

    dispatcher.gemv_ternary_g128(&blocks, &input, &mut out_seq, n_rows, k)?;
    gemv_ternary_g128_par(&dispatcher, &blocks, &input, &mut out_par, n_rows, k)?;

    for i in 0..n_rows {
        assert!(
            (out_seq[i] - out_par[i]).abs() < 1e-4,
            "row {i}: seq={}, par={}",
            out_seq[i],
            out_par[i]
        );
    }

    Ok(())
}

#[test]
fn par_ternary_gemv_small_is_sequential() -> KernelResult<()> {
    let n_rows = 4;
    let k = 128;
    assert!(
        n_rows < par_gemv_min_rows(),
        "test fixture assumption: n_rows={n_rows} must be below par_gemv_min_rows on this host"
    );

    let blocks_per_row = k / QK_TQ2_0_G128;
    let blocks = vec![make_ternary_block([0xAAu8; 32]); n_rows * blocks_per_row];
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
    let dispatcher = KernelDispatcher::auto_detect();

    // Below par_gemv_min_rows(), gemv_ternary_g128_par short-circuits to the
    // exact same dispatcher.gemv_ternary_g128 call used for `out_direct`
    // below (see parallel.rs::gemv_ternary_g128_par's `n_rows < par_gemv_min_rows()`
    // branch), so the two outputs must be bit-for-bit identical, not merely
    // close. This is the ternary/parallel.rs twin of
    // parallel_tiled.rs::adaptive_ternary_gemv_small_is_direct.
    let mut output = vec![0.0f32; n_rows];
    gemv_ternary_g128_par(&dispatcher, &blocks, &input, &mut output, n_rows, k)?;

    let mut out_direct = vec![0.0f32; n_rows];
    dispatcher.gemv_ternary_g128(&blocks, &input, &mut out_direct, n_rows, k)?;

    for i in 0..n_rows {
        assert_eq!(
            out_direct[i].to_bits(),
            output[i].to_bits(),
            "row {i}: direct={}, par={}",
            out_direct[i],
            output[i]
        );
    }

    Ok(())
}

#[test]
fn par_ternary_gemm_matches_sequential() -> KernelResult<()> {
    let m = 8;
    let n_rows = 16;
    let k = 128;
    let blocks_per_row = k / QK_TQ2_0_G128;
    let blocks = vec![make_ternary_block([0xAAu8; 32]); n_rows * blocks_per_row];
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.005) - 0.32).collect();
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_seq = vec![0.0f32; m * n_rows];
    let mut out_par = vec![0.0f32; m * n_rows];

    dispatcher.gemm_ternary_g128(&blocks, &input, &mut out_seq, m, n_rows, k)?;
    gemm_ternary_g128_par(&dispatcher, &blocks, &input, &mut out_par, m, n_rows, k)?;

    for i in 0..(m * n_rows) {
        assert!(
            (out_seq[i] - out_par[i]).abs() < 1e-4,
            "idx {i}: seq={}, par={}",
            out_seq[i],
            out_par[i]
        );
    }

    Ok(())
}

// ── LayerParallelConfig tests ──

#[test]
fn layer_parallel_config_default() {
    let config = LayerParallelConfig::default();
    assert_eq!(config.max_parallel_layers, 1);
    assert_eq!(config.pipeline_depth, 1);
}

#[test]
fn layer_parallel_config_for_model() {
    let config = LayerParallelConfig::for_model(36, 8);
    assert!(config.max_parallel_layers >= 1);
    assert!(config.max_parallel_layers <= 36);
    assert!(config.pipeline_depth >= 1);
}

#[test]
fn layer_parallel_config_single_thread() {
    let config = LayerParallelConfig::for_model(36, 1);
    assert_eq!(config.max_parallel_layers, 1);
    assert_eq!(config.pipeline_depth, 1);
}

// ── PipelineStage tests ──

#[test]
fn pipeline_stage_display() {
    assert_eq!(format!("{}", PipelineStage::Prefill), "prefill");
    assert_eq!(format!("{}", PipelineStage::Decode), "decode");
    assert_eq!(format!("{}", PipelineStage::PostProcess), "post_process");
}

#[test]
fn pipeline_stage_equality() {
    assert_eq!(PipelineStage::Prefill, PipelineStage::Prefill);
    assert_ne!(PipelineStage::Prefill, PipelineStage::Decode);
}

// ── ParallelStats tests ──

#[test]
fn parallel_stats_default() {
    let stats = ParallelStats::default();
    assert_eq!(stats.total_rows_processed, 0);
    assert_eq!(stats.parallel_invocations, 0);
    assert_eq!(stats.sequential_fallbacks, 0);
    assert!((stats.average_tile_size - 0.0).abs() < f64::EPSILON);
}

#[test]
fn parallel_stats_record() {
    let mut stats = ParallelStats::default();
    stats.record_parallel(256, 32);
    assert_eq!(stats.total_rows_processed, 256);
    assert_eq!(stats.parallel_invocations, 1);
    assert!((stats.average_tile_size - 32.0).abs() < 0.01);
    assert!((stats.parallel_fraction() - 1.0).abs() < f64::EPSILON);

    stats.record_sequential(64);
    assert_eq!(stats.total_rows_processed, 320);
    assert_eq!(stats.sequential_fallbacks, 1);
    assert!((stats.parallel_fraction() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn parallel_stats_gemv_gemm_counts() {
    let mut stats = ParallelStats::default();
    stats.record_gemv();
    stats.record_gemv();
    stats.record_gemm();
    assert_eq!(stats.total_gemv_calls, 2);
    assert_eq!(stats.total_gemm_calls, 1);
}

#[test]
fn parallel_stats_fraction_empty() {
    let stats = ParallelStats::default();
    assert!((stats.parallel_fraction() - 0.0).abs() < f64::EPSILON);
}

// ── Parallel dequant tests ──

#[test]
fn par_dequant_matches_sequential() {
    let n_blocks = 128;
    let mut blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let bits = [(i & 0xFF) as u8; 16];
        blocks.push(make_block(0.5 + i as f32 * 0.01, bits));
    }
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_seq = vec![0.0f32; n_blocks * QK1_0_G128];
    let mut out_par = vec![0.0f32; n_blocks * QK1_0_G128];

    dispatcher
        .dequant(&blocks, &mut out_seq)
        .expect("sequential dequant should succeed");
    dequant_1bit_g128_par(&dispatcher, &blocks, &mut out_par)
        .expect("parallel dequant should succeed");

    for i in 0..out_seq.len() {
        assert!(
            (out_seq[i] - out_par[i]).abs() < 1e-6,
            "idx {i}: seq={}, par={}",
            out_seq[i],
            out_par[i]
        );
    }
}

#[test]
fn par_dequant_small_sequential_fallback() {
    let n_blocks = 4;
    let blocks: Vec<_> = (0..n_blocks)
        .map(|i| make_block(1.0, [(i & 0xFF) as u8; 16]))
        .collect();
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_seq = vec![0.0f32; n_blocks * QK1_0_G128];
    let mut out_par = vec![0.0f32; n_blocks * QK1_0_G128];

    dispatcher
        .dequant(&blocks, &mut out_seq)
        .expect("sequential should succeed");
    dequant_1bit_g128_par(&dispatcher, &blocks, &mut out_par)
        .expect("parallel (fallback) should succeed");

    for i in 0..out_seq.len() {
        assert!(
            (out_seq[i] - out_par[i]).abs() < f32::EPSILON,
            "idx {i}: seq={}, par={}",
            out_seq[i],
            out_par[i]
        );
    }
}

#[test]
fn par_dequant_buffer_too_small() {
    let blocks = vec![make_block(1.0, [0xFF; 16]); 4];
    let dispatcher = KernelDispatcher::auto_detect();
    let mut output = vec![0.0f32; 10]; // Too small: need 4 * 128 = 512
    let result = dequant_1bit_g128_par(&dispatcher, &blocks, &mut output);
    assert!(result.is_err());
}

// ── Standard-quant (Q4_0 / Q8_0) parallel wrappers ──

fn std_input(k: usize, seed: u32) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x >> 8) as f32 / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect()
}

#[test]
fn par_q4_0_matches_scalar_large() {
    // 600 rows is above every platform's tuned `par_gemv_min_rows()` (max
    // 448), so the parallel + SIMD path is exercised, not the fallback.
    let n_rows = 600;
    let in_features = 128;
    let raw: Vec<f32> = std_input(n_rows * in_features, 3);
    let blocks = BlockQ4_0::quantize(&raw).expect("quantize q4_0");
    let input = std_input(in_features, 9);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_ref = vec![0.0f32; n_rows];
    let mut out_par = vec![0.0f32; n_rows];
    crate::gemv_q4_0::gemv_q4_0_scalar(&blocks, &input, &mut out_ref, n_rows, in_features)
        .expect("scalar q4_0");
    gemv_q4_0_par(
        &dispatcher,
        &blocks,
        &input,
        &mut out_par,
        n_rows,
        in_features,
    )
    .expect("parallel q4_0");

    for r in 0..n_rows {
        let tol = 1e-3 * out_ref[r].abs().max(1.0);
        assert!(
            (out_ref[r] - out_par[r]).abs() <= tol,
            "row {r}: scalar={}, par={}",
            out_ref[r],
            out_par[r]
        );
    }
}

#[test]
fn par_q8_0_matches_scalar_large() {
    let n_rows = 600;
    let in_features = 128;
    let raw: Vec<f32> = std_input(n_rows * in_features, 4);
    let blocks = BlockQ8_0::quantize(&raw).expect("quantize q8_0");
    let input = std_input(in_features, 8);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_ref = vec![0.0f32; n_rows];
    let mut out_par = vec![0.0f32; n_rows];
    crate::gemv_q8_0::gemv_q8_0_scalar(&blocks, &input, &mut out_ref, n_rows, in_features)
        .expect("scalar q8_0");
    gemv_q8_0_par(
        &dispatcher,
        &blocks,
        &input,
        &mut out_par,
        n_rows,
        in_features,
    )
    .expect("parallel q8_0");

    for r in 0..n_rows {
        let tol = 1e-3 * out_ref[r].abs().max(1.0);
        assert!(
            (out_ref[r] - out_par[r]).abs() <= tol,
            "row {r}: scalar={}, par={}",
            out_ref[r],
            out_par[r]
        );
    }
}

#[test]
fn par_q4_0_propagates_dim_errors() {
    let dispatcher = KernelDispatcher::with_tier(crate::KernelTier::Reference);
    let blocks = BlockQ4_0::quantize(&std_input(32, 1)).unwrap();
    let input = std_input(32, 1);
    let mut output = vec![0.0f32; 1];
    // in_features not a multiple of 32.
    assert!(gemv_q4_0_par(&dispatcher, &blocks, &input, &mut output, 1, 31).is_err());
}

#[test]
fn kquant_driver_parallel_matches_sequential() {
    // The K-quant row-parallel driver must be bit-identical to the plain
    // sequential loop (rows are independent). 600 rows forces the parallel
    // branch on all platforms.
    //
    // The "expected" reference is computed via `dot_f32` itself (K-15
    // vectorized it: 4/2 SIMD accumulator lanes reduced at the end, which
    // is not bit-identical to a naive `.iter().sum()` fold because f32
    // addition does not reassociate). That reordering is `dot_f32`'s own
    // documented, spec-sanctioned behavior — this test's job is narrower:
    // prove that distributing rows across Rayon tasks (or not) never
    // changes a row's result, which it does by calling the exact same
    // `dot_f32` sequentially here and letting the driver call it via
    // whichever path (parallel or scalar fallback) `n_rows` selects.
    let n_rows = 600;
    let in_features = 64;
    let input = std_input(in_features, 2);

    // Deterministic synthetic per-row "dequant".
    let fill = |row: usize, buf: &mut [f32]| -> KernelResult<()> {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((row * 31 + i * 7) % 17) as f32 * 0.125 - 1.0;
        }
        Ok(())
    };

    // Expected = independent sequential computation, same `dot_f32`.
    let mut expected = vec![0.0f32; n_rows];
    let mut buf = vec![0.0f32; in_features];
    for (row, e) in expected.iter_mut().enumerate() {
        fill(row, &mut buf).unwrap();
        *e = dot_f32(&buf, &input);
    }

    let mut got = vec![0.0f32; n_rows];
    gemv_kquant_row_parallel(&input, &mut got, n_rows, in_features, fill).unwrap();

    for r in 0..n_rows {
        assert_eq!(
            expected[r].to_bits(),
            got[r].to_bits(),
            "row {r}: expected={}, got={}",
            expected[r],
            got[r]
        );
    }
}

/// `dot_f32`'s SIMD reduction must still land within a tight tolerance of
/// a plain scalar fold — proving the accumulator-lane restructuring
/// (K-15) is a reordering, not an arithmetic change, across a range of
/// lengths that exercise the 16-wide main loop, the 4-wide remainder
/// loop, and the scalar tail (`dot_f32_neon`/`dot_f32_avx2` process 16
/// then 4 elements at a time; anything not a multiple of 4 falls to the
/// final scalar `while i < n` loop in each).
#[test]
fn dot_f32_matches_naive_scalar_sum_across_lengths() {
    for len in [0usize, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257] {
        let a: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.7 + 1.0).sin()) * 3.0)
            .collect();
        let b: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 1.3 + 2.0).cos()) * 2.0)
            .collect();

        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let simd = dot_f32(&a, &b);

        let tol = 1e-4 * naive.abs().max(1.0);
        assert!(
            (naive - simd).abs() <= tol,
            "len={len}: naive={naive}, simd={simd}"
        );
    }
}

/// Zero heap allocations after warm-up (K-15 / package title: "no
/// per-call allocs"): call the driver many times back-to-back on the
/// same thread with varying `in_features` and confirm the thread-local
/// scratch buffer converges to being reused rather than reallocated.
/// This is a coarse, allocator-free proxy (no `#[global_allocator]` is
/// available inside a library test binary) but it does exercise the exact
/// code path that
/// used to allocate on every single call, at a scale (10k calls) that
/// would be dominated by allocation overhead if the fix had not landed.
#[test]
#[ignore = "timing smoke test; run explicitly with --ignored, not part of the default gate"]
fn kquant_driver_repeated_calls_do_not_regress_to_per_call_alloc_timing() {
    let n_rows = 4; // below par_gemv_min_rows on every tuned profile: exercises the sequential fallback path specifically.
    let in_features = 256;
    let input = std_input(in_features, 42);
    let fill = |row: usize, buf: &mut [f32]| -> KernelResult<()> {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((row * 13 + i * 3) % 11) as f32 * 0.25 - 1.0;
        }
        Ok(())
    };

    // Warm up the thread-local scratch buffer once.
    let mut out = vec![0.0f32; n_rows];
    gemv_kquant_row_parallel(&input, &mut out, n_rows, in_features, fill).unwrap();

    let iterations = 10_000;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        gemv_kquant_row_parallel(&input, &mut out, n_rows, in_features, fill).unwrap();
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() / iterations as u128;
    // A per-call `vec![0.0f32; 256]` allocation dominates this shape at
    // well over 1000ns/call on any modern allocator; the reused-buffer
    // path should be well under that. This is a loose ceiling, not a
    // precise benchmark (see the recorded deviation for the real
    // criterion bench this should become).
    assert!(
        per_call_ns < 1000,
        "per-call time {per_call_ns}ns looks alloc-dominated (expected steady-state reuse)"
    );
}

// ── K-16 byte-identity: chunked (`par_chunks_mut(rows_per_task)`) vs
// per-row (`par_chunks_mut(1)`, the pre-fix shape) dispatch ──
//
// These use `KernelDispatcher::auto_detect()` — the real, production
// dispatcher — rather than a fixed CPU tier, because on this workspace's
// hardware (Apple M3, Metal available) `--all-features` DOES select
// `KernelTier::Gpu` (confirmed: `auto_detect()` here reports
// "gpu tier (backend=scirs2-metal)"), and unlike the CPU SIMD tiers a
// GPU-backed call is not obviously invariant to how many rows are batched
// into one call unless verified. Each test computes the same rows two ways
// — a manual loop calling the dispatcher once per row (`n_rows = 1`, i.e.
// exactly what `par_chunks_mut(1)` produced before this fix) and the
// package's now-chunked `*_par` entry point — and requires bit-for-bit
// (`to_bits`) equality, which is the literal ACCEPTANCE wording
// ("byte-identical GEMV output before/after"). All six pass under the real
// Metal backend on this host: for the 1-bit/ternary formats this is because
// `rows_per_task`'s 512 upper clamp stays under `GPU_MIN_ROWS` (1024) so
// both shapes take the identical CPU-fallback branch (see
// `rows_per_task_stays_below_gpu_min_rows`); for FP8/Q4_0/Q8_0, which have
// no such row-count gate and do reach the real Metal kernel either way,
// this empirically confirms those kernels compute each row independently
// regardless of batch size.

#[test]
fn gemv_1bit_par_byte_identical_to_per_row_dispatch() {
    let n_rows = 600;
    // T-09 fixture precondition: this test only proves the K-16 chunking
    // fix if the chunked entry point actually takes its parallel branch.
    // Without this guard a future profile with a threshold above 600 would
    // silently test the sequential short-circuit and keep passing with the
    // fix deleted.
    assert!(
        n_rows >= par_gemv_min_rows(),
        "test fixture assumption: n_rows={n_rows} must be >= par_gemv_min_rows() ({}) on \
         this host, otherwise the chunked path under test is never entered",
        par_gemv_min_rows()
    );
    let k = 256;
    let (blocks, input) = make_test_data(n_rows, k);
    let blocks_per_row = k / QK1_0_G128;
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_per_row = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        dispatcher
            .gemv(row_blocks, &input, &mut out_per_row[row..row + 1], 1, k)
            .expect("per-row gemv should succeed");
    }

    let mut out_chunked = vec![0.0f32; n_rows];
    gemv_1bit_g128_par(&dispatcher, &blocks, &input, &mut out_chunked, n_rows, k)
        .expect("chunked gemv should succeed");

    for i in 0..n_rows {
        assert_eq!(
            out_per_row[i].to_bits(),
            out_chunked[i].to_bits(),
            "row {i} diverged under tier {:?}",
            dispatcher.tier()
        );
    }
}

#[test]
fn gemv_ternary_par_byte_identical_to_per_row_dispatch() -> KernelResult<()> {
    let n_rows = 600;
    // T-09 fixture precondition: this test only proves the K-16 chunking
    // fix if the chunked entry point actually takes its parallel branch.
    // Without this guard a future profile with a threshold above 600 would
    // silently test the sequential short-circuit and keep passing with the
    // fix deleted.
    assert!(
        n_rows >= par_gemv_min_rows(),
        "test fixture assumption: n_rows={n_rows} must be >= par_gemv_min_rows() ({}) on \
         this host, otherwise the chunked path under test is never entered",
        par_gemv_min_rows()
    );
    let k = 256;
    let blocks_per_row = k / QK_TQ2_0_G128;
    // Non-uniform: distinct byte pattern per row, not a single repeated block.
    let blocks: Vec<_> = (0..n_rows * blocks_per_row)
        .map(|i| make_ternary_block([((i * 41 + 7) & 0xFF) as u8; 32]))
        .collect();
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_per_row = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        dispatcher.gemv_ternary_g128(row_blocks, &input, &mut out_per_row[row..row + 1], 1, k)?;
    }

    let mut out_chunked = vec![0.0f32; n_rows];
    gemv_ternary_g128_par(&dispatcher, &blocks, &input, &mut out_chunked, n_rows, k)?;

    for i in 0..n_rows {
        assert_eq!(
            out_per_row[i].to_bits(),
            out_chunked[i].to_bits(),
            "row {i} diverged under tier {:?}",
            dispatcher.tier()
        );
    }
    Ok(())
}

#[test]
fn gemv_fp8_e4m3_par_byte_identical_to_per_row_dispatch() -> KernelResult<()> {
    let n_rows = 600;
    // T-09 fixture precondition: this test only proves the K-16 chunking
    // fix if the chunked entry point actually takes its parallel branch.
    // Without this guard a future profile with a threshold above 600 would
    // silently test the sequential short-circuit and keep passing with the
    // fix deleted.
    assert!(
        n_rows >= par_gemv_min_rows(),
        "test fixture assumption: n_rows={n_rows} must be >= par_gemv_min_rows() ({}) on \
         this host, otherwise the chunked path under test is never entered",
        par_gemv_min_rows()
    );
    let k = 256;
    let raw: Vec<f32> = std_input(n_rows * k, 11);
    let blocks = BlockFP8E4M3::quantize(&raw).map_err(KernelError::Core)?;
    let blocks_per_row = k / QK_FP8;
    let input = std_input(k, 21);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_per_row = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        dispatcher.gemv_fp8_e4m3(row_blocks, &input, &mut out_per_row[row..row + 1], 1, k)?;
    }

    let mut out_chunked = vec![0.0f32; n_rows];
    gemv_fp8_e4m3_par(&dispatcher, &blocks, &input, &mut out_chunked, n_rows, k)?;

    for i in 0..n_rows {
        assert_eq!(
            out_per_row[i].to_bits(),
            out_chunked[i].to_bits(),
            "row {i} diverged under tier {:?}",
            dispatcher.tier()
        );
    }
    Ok(())
}

#[test]
fn gemv_fp8_e5m2_par_byte_identical_to_per_row_dispatch() -> KernelResult<()> {
    let n_rows = 600;
    // T-09 fixture precondition: this test only proves the K-16 chunking
    // fix if the chunked entry point actually takes its parallel branch.
    // Without this guard a future profile with a threshold above 600 would
    // silently test the sequential short-circuit and keep passing with the
    // fix deleted.
    assert!(
        n_rows >= par_gemv_min_rows(),
        "test fixture assumption: n_rows={n_rows} must be >= par_gemv_min_rows() ({}) on \
         this host, otherwise the chunked path under test is never entered",
        par_gemv_min_rows()
    );
    let k = 256;
    let raw: Vec<f32> = std_input(n_rows * k, 12);
    let blocks = BlockFP8E5M2::quantize(&raw).map_err(KernelError::Core)?;
    let blocks_per_row = k / QK_FP8;
    let input = std_input(k, 22);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_per_row = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        dispatcher.gemv_fp8_e5m2(row_blocks, &input, &mut out_per_row[row..row + 1], 1, k)?;
    }

    let mut out_chunked = vec![0.0f32; n_rows];
    gemv_fp8_e5m2_par(&dispatcher, &blocks, &input, &mut out_chunked, n_rows, k)?;

    for i in 0..n_rows {
        assert_eq!(
            out_per_row[i].to_bits(),
            out_chunked[i].to_bits(),
            "row {i} diverged under tier {:?}",
            dispatcher.tier()
        );
    }
    Ok(())
}

#[test]
fn gemv_q4_0_par_byte_identical_to_per_row_dispatch() -> KernelResult<()> {
    let n_rows = 600;
    // T-09 fixture precondition: this test only proves the K-16 chunking
    // fix if the chunked entry point actually takes its parallel branch.
    // Without this guard a future profile with a threshold above 600 would
    // silently test the sequential short-circuit and keep passing with the
    // fix deleted.
    assert!(
        n_rows >= par_gemv_min_rows(),
        "test fixture assumption: n_rows={n_rows} must be >= par_gemv_min_rows() ({}) on \
         this host, otherwise the chunked path under test is never entered",
        par_gemv_min_rows()
    );
    let in_features = 256;
    let raw: Vec<f32> = std_input(n_rows * in_features, 31);
    let blocks = BlockQ4_0::quantize(&raw).map_err(KernelError::Core)?;
    let blocks_per_row = in_features / QK_Q4_0;
    let input = std_input(in_features, 32);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_per_row = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        dispatcher.gemv_q4_0(
            row_blocks,
            &input,
            &mut out_per_row[row..row + 1],
            1,
            in_features,
        )?;
    }

    let mut out_chunked = vec![0.0f32; n_rows];
    gemv_q4_0_par(
        &dispatcher,
        &blocks,
        &input,
        &mut out_chunked,
        n_rows,
        in_features,
    )?;

    for i in 0..n_rows {
        assert_eq!(
            out_per_row[i].to_bits(),
            out_chunked[i].to_bits(),
            "row {i} diverged under tier {:?}",
            dispatcher.tier()
        );
    }
    Ok(())
}

#[test]
fn gemv_q8_0_par_byte_identical_to_per_row_dispatch() -> KernelResult<()> {
    let n_rows = 600;
    // T-09 fixture precondition: this test only proves the K-16 chunking
    // fix if the chunked entry point actually takes its parallel branch.
    // Without this guard a future profile with a threshold above 600 would
    // silently test the sequential short-circuit and keep passing with the
    // fix deleted.
    assert!(
        n_rows >= par_gemv_min_rows(),
        "test fixture assumption: n_rows={n_rows} must be >= par_gemv_min_rows() ({}) on \
         this host, otherwise the chunked path under test is never entered",
        par_gemv_min_rows()
    );
    let in_features = 256;
    let raw: Vec<f32> = std_input(n_rows * in_features, 41);
    let blocks = BlockQ8_0::quantize(&raw).map_err(KernelError::Core)?;
    let blocks_per_row = in_features / QK_Q8_0;
    let input = std_input(in_features, 42);
    let dispatcher = KernelDispatcher::auto_detect();

    let mut out_per_row = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        dispatcher.gemv_q8_0(
            row_blocks,
            &input,
            &mut out_per_row[row..row + 1],
            1,
            in_features,
        )?;
    }

    let mut out_chunked = vec![0.0f32; n_rows];
    gemv_q8_0_par(
        &dispatcher,
        &blocks,
        &input,
        &mut out_chunked,
        n_rows,
        in_features,
    )?;

    for i in 0..n_rows {
        assert_eq!(
            out_per_row[i].to_bits(),
            out_chunked[i].to_bits(),
            "row {i} diverged under tier {:?}",
            dispatcher.tier()
        );
    }
    Ok(())
}

/// K-02: representative sample of this file's migrated construction
/// sites, confirming each names the buffer it complained about.
#[test]
fn migrated_errors_name_the_offending_buffer() {
    let dispatcher = KernelDispatcher::auto_detect();

    // gemv_1bit_g128_par: short input -> "input".
    let (blocks, _) = make_test_data(4, 128);
    let short_input = vec![0.0f32; 10];
    let mut out = vec![0.0f32; 4];
    let err = gemv_1bit_g128_par(&dispatcher, &blocks, &short_input, &mut out, 4, 128).unwrap_err();
    assert_eq!(err.buffer_name(), Some("input"));

    // gemv_1bit_g128_par: short output -> "output".
    let input = vec![0.0f32; 128];
    let mut short_out = vec![0.0f32; 0];
    let err = gemv_1bit_g128_par(&dispatcher, &blocks, &input, &mut short_out, 4, 128).unwrap_err();
    assert_eq!(err.buffer_name(), Some("output"));

    // gemv_1bit_g128_par: too few blocks -> "blocks".
    let mut out = vec![0.0f32; 8];
    let err = gemv_1bit_g128_par(&dispatcher, &blocks, &input, &mut out, 8, 128).unwrap_err();
    assert_eq!(err.buffer_name(), Some("blocks"));

    // gemv_ternary_g128_par: short input -> "input".
    let ternary_blocks = vec![make_ternary_block([0xAAu8; 32]); 4];
    let short_input = vec![0.0f32; 10];
    let mut out = vec![0.0f32; 4];
    let err = gemv_ternary_g128_par(&dispatcher, &ternary_blocks, &short_input, &mut out, 4, 128)
        .unwrap_err();
    assert_eq!(err.buffer_name(), Some("input"));

    // gemv_fp8_e4m3_par: too few blocks -> "blocks".
    let raw = std_input(256, 5);
    let fp8_blocks = BlockFP8E4M3::quantize(&raw).expect("quantize fp8");
    let input = std_input(256, 6);
    let mut out = vec![0.0f32; 2];
    let err = gemv_fp8_e4m3_par(&dispatcher, &fp8_blocks, &input, &mut out, 2, 256).unwrap_err();
    assert_eq!(err.buffer_name(), Some("blocks"));

    // dequant_1bit_g128_par: short output -> "output".
    let mut tiny_out = vec![0.0f32; 4];
    let err = dequant_1bit_g128_par(&dispatcher, &blocks, &mut tiny_out).unwrap_err();
    assert_eq!(err.buffer_name(), Some("output"));

    // ── K-02 residue: `validate_std_gemv`'s three migrated sites, reached
    // through the public `gemv_q4_0`/`gemv_q8_0` wrappers. These are the
    // last unnamed length errors in this file; `gemv_q4_0.rs`/`gemv_q8_0.rs`
    // assert the (shared) `error_code`, and the operand NAME is pinned here.
    let q4_blocks = BlockQ4_0::quantize(&std_input(256, 51)).expect("quantize q4_0");
    let q4_input = std_input(256, 52);

    // Too few blocks for the requested rows -> "blocks".
    let mut out = vec![0.0f32; 2];
    let err = crate::gemv_q4_0::gemv_q4_0(&q4_blocks, &q4_input, &mut out, 2, 256)
        .expect_err("too few blocks must be rejected");
    assert_eq!(err.error_code(), "DIMENSION_MISMATCH");
    assert_eq!(err.buffer_name(), Some("blocks"));

    // Short input -> "input".
    let short_input = std_input(32, 53);
    let mut out = vec![0.0f32; 1];
    let err = crate::gemv_q4_0::gemv_q4_0(&q4_blocks, &short_input, &mut out, 1, 256)
        .expect_err("a short input must be rejected");
    assert_eq!(err.error_code(), "DIMENSION_MISMATCH");
    assert_eq!(err.buffer_name(), Some("input"));

    // Short output -> "output".
    let mut no_out: Vec<f32> = Vec::new();
    let err = crate::gemv_q4_0::gemv_q4_0(&q4_blocks, &q4_input, &mut no_out, 1, 256)
        .expect_err("an undersized output buffer must be rejected");
    assert_eq!(err.error_code(), "BUFFER_TOO_SMALL");
    assert_eq!(err.buffer_name(), Some("output"));

    // Same three, through the Q8_0 twin (same validator, other block size).
    let q8_blocks = BlockQ8_0::quantize(&std_input(256, 61)).expect("quantize q8_0");
    let q8_input = std_input(256, 62);

    let mut out = vec![0.0f32; 2];
    let err = crate::gemv_q8_0::gemv_q8_0(&q8_blocks, &q8_input, &mut out, 2, 256)
        .expect_err("too few blocks must be rejected");
    assert_eq!(err.error_code(), "DIMENSION_MISMATCH");
    assert_eq!(err.buffer_name(), Some("blocks"));

    let mut no_out: Vec<f32> = Vec::new();
    let err = crate::gemv_q8_0::gemv_q8_0(&q8_blocks, &q8_input, &mut no_out, 1, 256)
        .expect_err("an undersized output buffer must be rejected");
    assert_eq!(err.error_code(), "BUFFER_TOO_SMALL");
    assert_eq!(err.buffer_name(), Some("output"));
}

#[test]
fn rows_per_task_stays_below_gpu_min_rows() {
    // Load-bearing for the byte-identity tests above: as long as
    // `rows_per_task` never reaches 1024 (`GPU_MIN_ROWS` in
    // `dispatch.rs`), every chunk this crate hands to a Gpu-tier
    // dispatcher takes the same CPU-fallback branch a lone row always
    // did, regardless of core count or matrix size.
    for n_rows in [8, 64, 512, 4096, 248_320, 10_000_000] {
        assert!(
            rows_per_task(n_rows) <= 512,
            "rows_per_task({n_rows}) = {} exceeds the GPU_MIN_ROWS safety margin",
            rows_per_task(n_rows)
        );
        assert!(rows_per_task(n_rows) >= 8);
    }
}

/// Measures the actual parallel speed-up of the K-16 chunking fix on
/// this host (not part of the default gate — an in-crate substitute for
/// the criterion bench). Run explicitly with `--ignored`.
/// Measures the K-16/K-M2 parallel speed-up on the **production** path.
///
/// Uses `KernelDispatcher::auto_detect()` (whatever tier this host actually
/// resolves — confirmed `Gpu`/Metal here, though `TernaryKernel`'s `Gpu` arm
/// itself always falls back to the best CPU SIMD tier, since there is no
/// ternary GPU kernel) rather than a forced `Reference` tier: a scalar
/// tier's per-row compute is expensive enough that fixed per-task scheduling
/// overhead barely shows up, which flatters the speedup ratio and would not
/// describe the tier real inference actually runs.
///
/// Also measures all three routing legs `gemv_adaptive_ternary` can choose
/// between (flat `gemv_ternary_g128_par`, tiled `gemv_parallel_tiled_ternary`,
/// and the adaptive entry point itself) rather than only the flat one,
/// because K-M2 changed which leg the flagship (LM-head-sized) shape
/// actually resolves to — asserting on a leg the adaptive dispatcher no
/// longer selects for this shape would not support the ACCEPTANCE claim.
#[test]
#[ignore = "timing measurement, not a correctness assertion; run explicitly with --ignored"]
fn measure_gemv_ternary_par_efficiency() {
    let n_rows = 248_320; // Bonsai 2 LM head row count, per CONTEXT.md.
    let k = 5120;
    let blocks_per_row = k / QK_TQ2_0_G128;
    let blocks = vec![make_ternary_block([0xAAu8; 32]); n_rows * blocks_per_row];
    let input: Vec<f32> = std_input(k, 99);
    let dispatcher = KernelDispatcher::auto_detect();
    let num_threads = rayon::current_num_threads();

    let strategy = crate::parallel_tiled::select_gemv_strategy(n_rows, k);
    assert_eq!(
        strategy,
        crate::parallel_tiled::AdaptiveStrategy::ParallelTiled,
        "flagship-shape assumption: n_rows={n_rows} must resolve to ParallelTiled on this host \
         for this measurement to describe the shipped routing"
    );

    let mut out_seq = vec![0.0f32; n_rows];
    let start_seq = std::time::Instant::now();
    dispatcher
        .gemv_ternary_g128(&blocks, &input, &mut out_seq, n_rows, k)
        .expect("sequential gemv_ternary_g128 should succeed");
    let seq_elapsed = start_seq.elapsed();

    let mut out_flat = vec![0.0f32; n_rows];
    let start_flat = std::time::Instant::now();
    gemv_ternary_g128_par(&dispatcher, &blocks, &input, &mut out_flat, n_rows, k)
        .expect("flat parallel gemv_ternary_g128_par should succeed");
    let flat_elapsed = start_flat.elapsed();

    let mut out_tiled = vec![0.0f32; n_rows];
    let start_tiled = std::time::Instant::now();
    crate::parallel_tiled::gemv_parallel_tiled_ternary(
        &dispatcher,
        &blocks,
        &input,
        &mut out_tiled,
        n_rows,
        k,
    )
    .expect("tiled gemv_parallel_tiled_ternary should succeed");
    let tiled_elapsed = start_tiled.elapsed();

    let mut out_adaptive = vec![0.0f32; n_rows];
    let start_adaptive = std::time::Instant::now();
    crate::parallel_tiled::gemv_adaptive_ternary(
        &dispatcher,
        &blocks,
        &input,
        &mut out_adaptive,
        n_rows,
        k,
    )
    .expect("gemv_adaptive_ternary should succeed");
    let adaptive_elapsed = start_adaptive.elapsed();

    let flat_speedup = seq_elapsed.as_secs_f64() / flat_elapsed.as_secs_f64();
    let tiled_speedup = seq_elapsed.as_secs_f64() / tiled_elapsed.as_secs_f64();
    let adaptive_speedup = seq_elapsed.as_secs_f64() / adaptive_elapsed.as_secs_f64();
    eprintln!(
        "measure_gemv_ternary_par_efficiency: n_rows={n_rows} k={k} tier={:?} strategy={strategy:?} \
         threads={num_threads} sequential={seq_elapsed:?} | \
         flat={flat_elapsed:?} speedup={flat_speedup:.2}x | \
         tiled={tiled_elapsed:?} speedup={tiled_speedup:.2}x | \
         adaptive(production path)={adaptive_elapsed:?} speedup={adaptive_speedup:.2}x \
         efficiency={:.1}%",
        dispatcher.tier(),
        100.0 * adaptive_speedup / num_threads as f64
    );

    // The ACCEPTANCE bar (">= 4x on 8 cores, up from 1.44x") is about the
    // path production code actually takes for this shape, which after K-M2
    // is the adaptive entry point (ParallelTiled for the LM head).
    assert!(
        adaptive_speedup >= 4.0,
        "expected >= 4x parallel speedup on the production adaptive path ({num_threads} \
         threads, tier={:?}), got {adaptive_speedup:.2}x (flat={flat_speedup:.2}x, \
         tiled={tiled_speedup:.2}x)",
        dispatcher.tier()
    );
}

// ─── K-18: register-blocked GEMM drivers ────────────────────────────────
//
// `gemm_1bit_g128_par` / `gemm_ternary_g128_par` used to hand Rayon one
// batch row per task and call the tier kernel with `m = 1`, which is the
// loop-of-GEMVs shape K-18 names. They now split the batch into slabs and
// run a genuine `m > 1` register-blocked GEMM per slab. The kernels are
// bit-identical to the tier kernels they replace (see
// `gemm_ternary::blocked_tests` / `gemm_onebit::blocked_tests`), so these
// drivers must be bit-identical to the dispatcher too — asserted with
// `assert_eq!`, not a tolerance, at batch sizes that straddle the parallel
// threshold and leave a register-block tail.

fn blocked_test_ternary_matrix(n_rows: usize, k: usize) -> Vec<oxibonsai_core::BlockTQ2_0_g128> {
    let bpr = k / QK_TQ2_0_G128;
    (0..n_rows * bpr)
        .map(|i| {
            let mut qs = [0u8; 32];
            for (j, q) in qs.iter_mut().enumerate() {
                *q = ((i * 31 + j * 17) & 0xFF) as u8;
            }
            oxibonsai_core::BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.25 + (i % 7) as f32 * 0.125),
            }
        })
        .collect()
}

fn blocked_test_onebit_matrix(n_rows: usize, k: usize) -> Vec<BlockQ1_0G128> {
    let bpr = k / QK1_0_G128;
    (0..n_rows * bpr)
        .map(|i| BlockQ1_0G128 {
            d: f16::from_f32(0.125 + (i % 5) as f32 * 0.25),
            qs: [((i * 29) & 0xFF) as u8; 16],
        })
        .collect()
}

fn blocked_test_activations(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| ((i % 251) as f32 * 0.0137) - 1.7)
        .collect()
}

#[test]
fn par_gemm_ternary_blocked_is_bit_identical_to_the_dispatcher() {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 256;
    let n_rows = 24;
    let blocks = blocked_test_ternary_matrix(n_rows, k);
    // Straddles `par_gemm_min_batch()` (max tuned value 16) and leaves a
    // 8/4/2/1 register-block tail in every branch.
    for &m in &[1usize, 2, 3, 7, 8, 16, 17, 33, 64] {
        let input = blocked_test_activations(m, k);
        let mut expected = vec![0.0f32; m * n_rows];
        let mut got = vec![0.0f32; m * n_rows];
        dispatcher
            .gemm_ternary_g128(&blocks, &input, &mut expected, m, n_rows, k)
            .expect("dispatcher ternary gemm should succeed");
        gemm_ternary_g128_par(&dispatcher, &blocks, &input, &mut got, m, n_rows, k)
            .expect("parallel ternary gemm should succeed");
        assert_eq!(got, expected, "ternary par gemm diverged at m={m}");
    }
}

#[test]
fn par_gemm_onebit_blocked_is_bit_identical_to_the_dispatcher() {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 256;
    let n_rows = 24;
    let blocks = blocked_test_onebit_matrix(n_rows, k);
    for &m in &[1usize, 2, 3, 5, 8, 16, 17, 33, 64] {
        let input = blocked_test_activations(m, k);
        let mut expected = vec![0.0f32; m * n_rows];
        let mut got = vec![0.0f32; m * n_rows];
        dispatcher
            .gemm(&blocks, &input, &mut expected, m, n_rows, k)
            .expect("dispatcher gemm should succeed");
        gemm_1bit_g128_par(&dispatcher, &blocks, &input, &mut got, m, n_rows, k)
            .expect("parallel gemm should succeed");
        assert_eq!(got, expected, "1-bit par gemm diverged at m={m}");
    }
}

#[test]
fn par_gemm_blocked_handles_a_prefill_sized_batch() {
    // A prefill-shaped call: more batch rows than Rayon workers, so every
    // slab holds several register blocks and the tail slab is short.
    let dispatcher = KernelDispatcher::auto_detect();
    let (m, n_rows, k) = (257usize, 40usize, 384usize);
    let blocks = blocked_test_ternary_matrix(n_rows, k);
    let input = blocked_test_activations(m, k);
    let mut expected = vec![0.0f32; m * n_rows];
    let mut got = vec![0.0f32; m * n_rows];
    dispatcher
        .gemm_ternary_g128(&blocks, &input, &mut expected, m, n_rows, k)
        .expect("dispatcher ternary gemm should succeed");
    gemm_ternary_g128_par(&dispatcher, &blocks, &input, &mut got, m, n_rows, k)
        .expect("parallel ternary gemm should succeed");
    assert_eq!(got, expected, "ternary par gemm diverged at prefill size");
}

#[test]
fn gemm_batch_chunk_rows_is_at_least_the_register_block() {
    // A slab shorter than MR cannot fill the register block, so the helper
    // must never produce one — and must never produce a zero-sized chunk,
    // which `par_chunks_mut` would reject.
    for &m in &[1usize, 2, 3, 7, 8, 9, 64, 512, 4096] {
        for &mr in &[1usize, 4, 8] {
            let chunk = gemm_batch_chunk_rows(m, mr);
            assert!(chunk >= 1, "m={m} mr={mr}: chunk must be positive");
            assert!(chunk <= m, "m={m} mr={mr}: chunk must not exceed the batch");
            assert!(
                chunk >= mr.min(m),
                "m={m} mr={mr}: chunk {chunk} is below the register block"
            );
        }
    }
}

/// K-18 measurement harness: the register-blocked GEMM at
/// `m in {1, 4, 8, 64, 512}` against the loop-of-GEMVs shape it replaces.
///
/// `#[ignore]` by default — it is a timing measurement, and this repository's
/// gate runs while other builds compete for the same cores, so a threshold
/// assertion here would be a flake generator rather than a regression guard.
/// The correctness of every configuration it times is already asserted, bit
/// for bit, by the tests above. Run it with
/// `cargo test -p oxibonsai-kernels --release -- --ignored --nocapture
/// gemm_register_blocking_speedup` and read the printed table.
#[test]
#[ignore = "timing measurement; run explicitly with --ignored --nocapture"]
fn gemm_register_blocking_speedup() {
    use std::time::Instant;

    // Pinned to the best CPU tier: `auto_detect` can report the GPU tier,
    // and `KernelDispatcher::gemm` then routes 1-bit GEMMs with
    // `n_rows >= GPU_MIN_ROWS` to Metal/CUDA, which would make the
    // "loop-of-gemv" column a GPU timing and the comparison meaningless.
    let dispatcher = KernelDispatcher::with_tier(crate::dispatch::cpu_kernel_tier());
    let (n_rows, k) = (1024usize, 1024usize);
    let ternary = blocked_test_ternary_matrix(n_rows, k);
    let onebit = blocked_test_onebit_matrix(n_rows, k);

    println!(
        "\nK-18 register-blocked GEMM, n_rows={n_rows} k={k}, tier={}",
        dispatcher.name()
    );
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>8}  {:>8}",
        "m", "loop-of-gemv", "blocked", "blocked+par", "seq x", "par x"
    );
    for &m in &[1usize, 4, 8, 64, 512] {
        let input = blocked_test_activations(m, k);
        let mut out = vec![0.0f32; m * n_rows];

        let t0 = Instant::now();
        dispatcher
            .gemm_ternary_g128(&ternary, &input, &mut out, m, n_rows, k)
            .expect("dispatcher ternary gemm should succeed");
        let baseline = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        crate::gemm_ternary::gemm_tq2_0_g128_blocked(
            &dispatcher,
            &ternary,
            &input,
            &mut out,
            m,
            n_rows,
            k,
        )
        .expect("blocked ternary gemm should succeed");
        let blocked = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        gemm_ternary_g128_par(&dispatcher, &ternary, &input, &mut out, m, n_rows, k)
            .expect("parallel ternary gemm should succeed");
        let parallel = t2.elapsed().as_secs_f64();

        println!(
            "{m:>6}  {:>10.3}ms  {:>10.3}ms  {:>10.3}ms  {:>7.2}x  {:>7.2}x",
            baseline * 1e3,
            blocked * 1e3,
            parallel * 1e3,
            baseline / blocked.max(f64::MIN_POSITIVE),
            baseline / parallel.max(f64::MIN_POSITIVE),
        );
    }

    println!("\n1-bit Q1_0_g128:");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>8}  {:>8}",
        "m", "loop-of-gemv", "blocked", "blocked+par", "seq x", "par x"
    );
    for &m in &[1usize, 4, 8, 64, 512] {
        let input = blocked_test_activations(m, k);
        let mut out = vec![0.0f32; m * n_rows];

        let t0 = Instant::now();
        dispatcher
            .gemm(&onebit, &input, &mut out, m, n_rows, k)
            .expect("dispatcher gemm should succeed");
        let baseline = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        crate::gemm_onebit::gemm_1bit_g128_blocked(
            &dispatcher,
            &onebit,
            &input,
            &mut out,
            m,
            n_rows,
            k,
        )
        .expect("blocked gemm should succeed");
        let blocked = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        gemm_1bit_g128_par(&dispatcher, &onebit, &input, &mut out, m, n_rows, k)
            .expect("parallel gemm should succeed");
        let parallel = t2.elapsed().as_secs_f64();

        println!(
            "{m:>6}  {:>10.3}ms  {:>10.3}ms  {:>10.3}ms  {:>7.2}x  {:>7.2}x",
            baseline * 1e3,
            blocked * 1e3,
            parallel * 1e3,
            baseline / blocked.max(f64::MIN_POSITIVE),
            baseline / parallel.max(f64::MIN_POSITIVE),
        );
    }
}
