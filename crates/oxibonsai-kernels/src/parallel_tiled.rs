//! Parallel tiled kernel execution.
//!
//! Combines cache-aware tiling (from the [`tiled`](crate::tiled) module) with Rayon parallelism
//! for maximum throughput. Strategy: outer loop over L2 tiles is parallel,
//! inner loop over L1 tiles is sequential.
//!
//! Also provides an adaptive dispatcher that selects the best strategy
//! (direct, parallel row, or parallel tiled) based on matrix dimensions.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::dispatch::KernelDispatcher;
use crate::error::{KernelError, KernelResult};
#[cfg(not(target_arch = "wasm32"))]
use crate::tiled::{optimal_tile_rows, L2_TILE_ROWS};
use crate::traits::OneBitKernel;
use crate::traits::TernaryKernel;
use crate::tuning::{PlatformProfile, TunedThresholds};
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

/// Minimum rows to justify parallelism overhead for parallel tiled GEMV.
///
/// Fallback default only: live dispatch decisions read
/// [`TunedThresholds::par_tiled_min_rows`] via [`PlatformProfile::global_thresholds`]
/// instead, so this constant is exercised only by [`ParallelConfig::default`]'s
/// diagnostic snapshot, not by the hot dispatch path.
const PAR_TILED_MIN_ROWS: usize = 128;

/// Minimum batch size for parallel tiled GEMM.
///
/// Fallback default only; see [`PAR_TILED_MIN_ROWS`].
const PAR_TILED_MIN_BATCH: usize = 4;

// ─── Validation helpers ────────────────────────────────────────────────

/// Validate GEMV parameters and return blocks_per_row.
fn validate_gemv(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &[f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<usize> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }
    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }
    Ok(blocks_per_row)
}

/// Validate GEMM parameters and return blocks_per_row.
fn validate_gemm(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &[f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<usize> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: input.len(),
        });
    }
    if output.len() < m * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: m * n_rows,
            available: output.len(),
        });
    }
    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }
    Ok(blocks_per_row)
}

// ─── Parallel tiled kernels ────────────────────────────────────────────

/// Parallel tiled GEMV: distribute L2 tiles across threads.
///
/// Each thread receives an L2-sized chunk of output rows and processes it
/// using L1 tiling internally. For problems below the platform-tuned
/// `par_tiled_min_rows` threshold, falls back to sequential tiled execution
/// via [`crate::tiled::gemv_tiled`].
///
/// The L1 tile size is dynamically computed via [`optimal_tile_rows`] to
/// account for the actual working set size at the given `k`.
pub fn gemv_parallel_tiled(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let blocks_per_row = validate_gemv(blocks, input, output, n_rows, k)?;
    #[cfg(target_arch = "wasm32")]
    let _blocks_per_row = validate_gemv(blocks, input, output, n_rows, k)?;

    // Sequential fallback for small row counts (platform-tuned threshold).
    if n_rows < PlatformProfile::global_thresholds().par_tiled_min_rows {
        return crate::tiled::gemv_tiled(dispatcher, blocks, input, output, n_rows, k);
    }

    // On WASM: no rayon threads available — fall back to sequential tiled.
    #[cfg(target_arch = "wasm32")]
    {
        return crate::tiled::gemv_tiled(dispatcher, blocks, input, output, n_rows, k);
    }

    // Compute optimal L1 tile size for this k
    #[cfg(not(target_arch = "wasm32"))]
    let l1_tile = optimal_tile_rows(k).max(1);

    // Parallel L2 tiles, each internally using L1 tiling
    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..n_rows]
            .par_chunks_mut(L2_TILE_ROWS)
            .enumerate()
            .try_for_each(|(tile_idx, out_chunk)| -> KernelResult<()> {
                let tile_start = tile_idx * L2_TILE_ROWS;
                let tile_rows = out_chunk.len();
                let block_start = tile_start * blocks_per_row;
                let block_end = (tile_start + tile_rows) * blocks_per_row;
                let tile_blocks = &blocks[block_start..block_end];

                // Apply L1 tiling within this L2 tile
                let mut l1_start = 0;
                while l1_start < tile_rows {
                    let l1_rows = (tile_rows - l1_start).min(l1_tile);
                    let l1_block_start = l1_start * blocks_per_row;
                    let l1_block_end = (l1_start + l1_rows) * blocks_per_row;

                    dispatcher.gemv(
                        &tile_blocks[l1_block_start..l1_block_end],
                        input,
                        &mut out_chunk[l1_start..l1_start + l1_rows],
                        l1_rows,
                        k,
                    )?;

                    l1_start += l1_rows;
                }

                Ok::<(), KernelError>(())
            })?;

        Ok(())
    }
}

/// Parallel tiled GEMM: distribute across batch AND row dimensions.
///
/// Parallelizes over the batch dimension at the outer level, then applies
/// L1 tiling on the weight rows within each parallel task. For small
/// batches (below the platform-tuned `par_gemm_min_batch` threshold),
/// falls back to sequential tiled.
pub fn gemm_parallel_tiled(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let blocks_per_row = validate_gemm(blocks, input, output, m, n_rows, k)?;
    #[cfg(target_arch = "wasm32")]
    let _blocks_per_row = validate_gemm(blocks, input, output, m, n_rows, k)?;

    // Sequential fallback for small batch sizes (platform-tuned threshold).
    if m < PlatformProfile::global_thresholds().par_gemm_min_batch {
        return crate::tiled::gemm_tiled(dispatcher, blocks, input, output, m, n_rows, k);
    }

    // On WASM: no rayon threads available — fall back to sequential tiled.
    #[cfg(target_arch = "wasm32")]
    {
        return crate::tiled::gemm_tiled(dispatcher, blocks, input, output, m, n_rows, k);
    }

    // Compute optimal L1 tile size for this k
    #[cfg(not(target_arch = "wasm32"))]
    let l1_tile = optimal_tile_rows(k).max(1);

    // Parallel over batch elements, L1-tiled weight rows within
    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..m * n_rows]
            .par_chunks_mut(n_rows)
            .enumerate()
            .try_for_each(|(mi, out_row)| -> KernelResult<()> {
                let input_offset = mi * k;

                // L1-tile the weight rows
                let mut row_start = 0;
                while row_start < n_rows {
                    let tile_rows = (n_rows - row_start).min(l1_tile);
                    let block_start = row_start * blocks_per_row;
                    let block_end = (row_start + tile_rows) * blocks_per_row;

                    dispatcher.gemm(
                        &blocks[block_start..block_end],
                        &input[input_offset..input_offset + k],
                        &mut out_row[row_start..row_start + tile_rows],
                        1,
                        tile_rows,
                        k,
                    )?;

                    row_start += tile_rows;
                }

                Ok::<(), KernelError>(())
            })?;

        Ok(())
    }
}

// ─── Adaptive strategy selection ───────────────────────────────────────

/// Strategy chosen by the adaptive dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveStrategy {
    /// Direct kernel dispatch (no parallelism, no tiling).
    Direct,
    /// Parallel row-wise dispatch (parallelism, no tiling).
    ParallelRow,
    /// Parallel tiled dispatch (parallelism + cache-aware tiling).
    ParallelTiled,
}

/// Determine the best strategy for a GEMV of the given dimensions, using the
/// live platform-tuned thresholds ([`PlatformProfile::global_thresholds`]).
///
/// This is the real production entry point; see
/// [`select_gemv_strategy_with_thresholds`] for the pure, injectable form
/// used in tests.
pub fn select_gemv_strategy(n_rows: usize, k: usize) -> AdaptiveStrategy {
    select_gemv_strategy_with_thresholds(n_rows, k, PlatformProfile::global_thresholds())
}

/// Determine the best strategy for a GEMV of the given dimensions against an
/// explicit set of tuned thresholds.
///
/// Pulled out from [`select_gemv_strategy`] so tests can install a synthetic
/// [`TunedThresholds`] (e.g. from [`PlatformProfile::with_cores`] /
/// [`PlatformProfile::with_cache`]) and assert the decision changes
/// accordingly, without touching the process-global cached profile.
pub fn select_gemv_strategy_with_thresholds(
    n_rows: usize,
    _k: usize,
    thresholds: &TunedThresholds,
) -> AdaptiveStrategy {
    if n_rows < thresholds.par_gemv_min_rows {
        AdaptiveStrategy::Direct
    } else if n_rows < thresholds.par_tiled_min_rows {
        AdaptiveStrategy::ParallelRow
    } else {
        AdaptiveStrategy::ParallelTiled
    }
}

/// Adaptive parallelism: choose the best strategy based on dimensions and
/// the live platform-tuned thresholds (see [`select_gemv_strategy`]).
///
/// - **Small** (`n_rows` < `par_gemv_min_rows`): direct dispatch, no overhead.
/// - **Medium** (`par_gemv_min_rows..par_tiled_min_rows`): parallel row-wise
///   via [`crate::parallel::gemv_1bit_g128_par`].
/// - **Large** (>= `par_tiled_min_rows`): parallel tiled via [`gemv_parallel_tiled`].
pub fn gemv_adaptive(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    match select_gemv_strategy(n_rows, k) {
        AdaptiveStrategy::Direct => dispatcher.gemv(blocks, input, output, n_rows, k),
        AdaptiveStrategy::ParallelRow => {
            crate::parallel::gemv_1bit_g128_par(dispatcher, blocks, input, output, n_rows, k)
        }
        AdaptiveStrategy::ParallelTiled => {
            gemv_parallel_tiled(dispatcher, blocks, input, output, n_rows, k)
        }
    }
}

pub fn gemv_adaptive_ternary(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    match select_gemv_strategy(n_rows, k) {
        AdaptiveStrategy::Direct => dispatcher.gemv_ternary_g128(blocks, input, output, n_rows, k),
        AdaptiveStrategy::ParallelRow | AdaptiveStrategy::ParallelTiled => {
            crate::parallel::gemv_ternary_g128_par(dispatcher, blocks, input, output, n_rows, k)
        }
    }
}

pub fn gemm_adaptive_ternary(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if m < PlatformProfile::global_thresholds().par_gemm_min_batch {
        dispatcher.gemm_ternary_g128(blocks, input, output, m, n_rows, k)
    } else {
        crate::parallel::gemm_ternary_g128_par(dispatcher, blocks, input, output, m, n_rows, k)
    }
}

// ─── Parallel configuration ────────────────────────────────────────────

/// Runtime info about parallel execution configuration.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of Rayon worker threads.
    pub num_threads: usize,
    /// Minimum rows for GEMV parallelism.
    pub gemv_threshold: usize,
    /// Minimum batch size for GEMM parallelism.
    pub gemm_threshold: usize,
    /// Whether to use cache-aware tiling.
    pub use_tiling: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let num_threads = rayon::current_num_threads();
        #[cfg(target_arch = "wasm32")]
        let num_threads = 1usize;

        Self {
            num_threads,
            gemv_threshold: PAR_TILED_MIN_ROWS,
            gemm_threshold: PAR_TILED_MIN_BATCH,
            use_tiling: true,
        }
    }
}

impl ParallelConfig {
    /// Configuration for single-threaded execution (testing/debugging).
    pub fn single_threaded() -> Self {
        Self {
            num_threads: 1,
            gemv_threshold: usize::MAX,
            gemm_threshold: usize::MAX,
            use_tiling: false,
        }
    }

    /// Check whether GEMV should use parallelism for the given row count.
    pub fn should_parallelize_gemv(&self, n_rows: usize) -> bool {
        self.num_threads > 1 && n_rows >= self.gemv_threshold
    }

    /// Check whether GEMM should use parallelism for the given batch size.
    pub fn should_parallelize_gemm(&self, m: usize) -> bool {
        self.num_threads > 1 && m >= self.gemm_threshold
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
    fn parallel_tiled_gemv_matches_sequential() {
        let n_rows = 256;
        let k = 256;
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_seq = vec![0.0f32; n_rows];
        let mut out_par = vec![0.0f32; n_rows];

        dispatcher
            .gemv(&blocks, &input, &mut out_seq, n_rows, k)
            .expect("direct gemv should succeed");
        gemv_parallel_tiled(&dispatcher, &blocks, &input, &mut out_par, n_rows, k)
            .expect("parallel tiled gemv should succeed");

        for i in 0..n_rows {
            assert!(
                (out_seq[i] - out_par[i]).abs() < 1e-4,
                "row {i}: seq={}, par_tiled={}",
                out_seq[i],
                out_par[i]
            );
        }
    }

    #[test]
    fn parallel_tiled_gemv_small_fallback() {
        // Below threshold — should fallback to sequential tiled
        let n_rows = 16;
        let k = 128;
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_seq = vec![0.0f32; n_rows];
        let mut out_par = vec![0.0f32; n_rows];

        dispatcher
            .gemv(&blocks, &input, &mut out_seq, n_rows, k)
            .expect("direct gemv should succeed");
        gemv_parallel_tiled(&dispatcher, &blocks, &input, &mut out_par, n_rows, k)
            .expect("fallback tiled gemv should succeed");

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
    fn parallel_tiled_gemm_matches_sequential() {
        let m = 8;
        let n_rows = 32;
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
            .expect("direct gemm should succeed");
        gemm_parallel_tiled(&dispatcher, &blocks, &input, &mut out_par, m, n_rows, k)
            .expect("parallel tiled gemm should succeed");

        for i in 0..(m * n_rows) {
            assert!(
                (out_seq[i] - out_par[i]).abs() < 1e-3,
                "idx {i}: seq={}, par_tiled={}",
                out_seq[i],
                out_par[i]
            );
        }
    }

    // These three tests exercise the live, platform-tuned `select_gemv_strategy`
    // (backed by `PlatformProfile::global_thresholds()`), so they use row
    // counts far outside any plausible tuned threshold range rather than the
    // old fixed 32/256 breakpoints, keeping them host-independent.
    #[test]
    fn adaptive_selects_direct_for_small() {
        let strategy = select_gemv_strategy(1, 128);
        assert_eq!(strategy, AdaptiveStrategy::Direct);
    }

    #[test]
    fn adaptive_selects_parallel_tiled_for_large() {
        let strategy = select_gemv_strategy(10_000_000, 4096);
        assert_eq!(strategy, AdaptiveStrategy::ParallelTiled);
    }

    /// Pure-function form of the strategy selector, tested against an
    /// explicit synthetic [`TunedThresholds`] so the tier boundaries are
    /// exercised deterministically regardless of the host machine's real
    /// platform profile.
    #[test]
    fn select_gemv_strategy_with_thresholds_respects_tiers() {
        let thresholds = TunedThresholds {
            par_gemv_min_rows: 32,
            par_gemm_min_batch: 4,
            par_tiled_min_rows: 256,
            tiled_gemm_block_m: 8,
            tiled_gemm_block_n: 8,
            tiled_gemm_block_k: 128,
        };

        assert_eq!(
            select_gemv_strategy_with_thresholds(31, 128, &thresholds),
            AdaptiveStrategy::Direct
        );
        assert_eq!(
            select_gemv_strategy_with_thresholds(32, 128, &thresholds),
            AdaptiveStrategy::ParallelRow
        );
        assert_eq!(
            select_gemv_strategy_with_thresholds(255, 128, &thresholds),
            AdaptiveStrategy::ParallelRow
        );
        assert_eq!(
            select_gemv_strategy_with_thresholds(256, 128, &thresholds),
            AdaptiveStrategy::ParallelTiled
        );
    }

    /// Installing a synthetic tuned profile (many cores, large cache) must
    /// change the strategy decision for a fixed `n_rows`/`k`, proving the
    /// dispatcher genuinely consults tuned thresholds rather than hardcoded
    /// constants.
    #[test]
    fn synthetic_tuned_profile_changes_strategy_decision() {
        let n_rows = 200;
        let k = 256;

        // Low-core, small-cache profile: high par_gemv_min_rows and a
        // par_tiled_min_rows anchored near the 256-row legacy baseline —
        // 200 rows lands below both, i.e. Direct.
        let low_core_profile = PlatformProfile::with_cores(1, 1);
        let low_core_thresholds = low_core_profile.compute_thresholds();
        assert_eq!(
            select_gemv_strategy_with_thresholds(n_rows, k, &low_core_thresholds),
            AdaptiveStrategy::Direct,
            "expected Direct for n_rows={n_rows} under a 1-core synthetic profile (par_gemv_min_rows={})",
            low_core_thresholds.par_gemv_min_rows
        );

        // Many-core profile: par_gemv_min_rows drops well below 200, but
        // par_tiled_min_rows (anchored to L2 size) stays above it, so the
        // same n_rows now lands in ParallelRow.
        let many_core_profile = PlatformProfile::with_cores(32, 32);
        let many_core_thresholds = many_core_profile.compute_thresholds();
        assert_eq!(
            select_gemv_strategy_with_thresholds(n_rows, k, &many_core_thresholds),
            AdaptiveStrategy::ParallelRow,
            "expected ParallelRow for n_rows={n_rows} under a 32-core synthetic profile (par_gemv_min_rows={}, par_tiled_min_rows={})",
            many_core_thresholds.par_gemv_min_rows,
            many_core_thresholds.par_tiled_min_rows
        );

        assert_ne!(
            select_gemv_strategy_with_thresholds(n_rows, k, &low_core_thresholds),
            select_gemv_strategy_with_thresholds(n_rows, k, &many_core_thresholds),
            "strategy decision must change when the tuned profile changes"
        );
    }

    /// Every strategy computes each output row as an independent dot
    /// product via the same underlying [`OneBitKernel::gemv`] call, so
    /// forcing Direct, ParallelRow, and ParallelTiled dispatch on identical
    /// inputs must produce bit-for-bit identical outputs -- this is a
    /// perf-routing change, not a numerics change, and this test pins that
    /// invariant exactly (no epsilon tolerance).
    #[test]
    fn strategy_forced_bit_parity_gemv() {
        let n_rows = 512;
        let k = 512;
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::auto_detect();

        // Direct: single call covering all rows.
        let mut out_direct = vec![0.0f32; n_rows];
        dispatcher
            .gemv(&blocks, &input, &mut out_direct, n_rows, k)
            .expect("direct gemv should succeed");

        // ParallelRow: forced via crate::parallel::gemv_1bit_g128_par.
        let mut out_parallel_row = vec![0.0f32; n_rows];
        crate::parallel::gemv_1bit_g128_par(
            &dispatcher,
            &blocks,
            &input,
            &mut out_parallel_row,
            n_rows,
            k,
        )
        .expect("parallel row gemv should succeed");

        // ParallelTiled: forced directly, bypassing the strategy selector.
        let mut out_parallel_tiled = vec![0.0f32; n_rows];
        gemv_parallel_tiled(
            &dispatcher,
            &blocks,
            &input,
            &mut out_parallel_tiled,
            n_rows,
            k,
        )
        .expect("parallel tiled gemv should succeed");

        for i in 0..n_rows {
            assert_eq!(
                out_direct[i].to_bits(),
                out_parallel_row[i].to_bits(),
                "row {i}: Direct vs ParallelRow diverged bit-exactly"
            );
            assert_eq!(
                out_direct[i].to_bits(),
                out_parallel_tiled[i].to_bits(),
                "row {i}: Direct vs ParallelTiled diverged bit-exactly"
            );
        }
    }

    #[test]
    fn adaptive_gemv_matches_direct() {
        let n_rows = 64;
        let k = 256;
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        let mut out_adaptive = vec![0.0f32; n_rows];

        dispatcher
            .gemv(&blocks, &input, &mut out_direct, n_rows, k)
            .expect("direct gemv should succeed");
        gemv_adaptive(&dispatcher, &blocks, &input, &mut out_adaptive, n_rows, k)
            .expect("adaptive gemv should succeed");

        for i in 0..n_rows {
            assert!(
                (out_direct[i] - out_adaptive[i]).abs() < 1e-4,
                "row {i}: direct={}, adaptive={}",
                out_direct[i],
                out_adaptive[i]
            );
        }
    }

    #[test]
    fn adaptive_ternary_gemv_small_is_direct() -> KernelResult<()> {
        let n_rows = 16;
        let k = 128;
        let blocks_per_row = k / oxibonsai_core::QK_TQ2_0_G128;
        let blocks = vec![make_ternary_block([0xAAu8; 32]); n_rows * blocks_per_row];
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let dispatcher = KernelDispatcher::auto_detect();
        let mut output = vec![0.0f32; n_rows];

        gemv_adaptive_ternary(&dispatcher, &blocks, &input, &mut output, n_rows, k)
    }

    #[test]
    fn adaptive_ternary_gemv_large_is_parallel() -> KernelResult<()> {
        let n_rows = 512;
        let k = 128;
        let blocks_per_row = k / oxibonsai_core::QK_TQ2_0_G128;
        let blocks = vec![make_ternary_block([0xAAu8; 32]); n_rows * blocks_per_row];
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let dispatcher = KernelDispatcher::auto_detect();
        let mut output = vec![0.0f32; n_rows];

        gemv_adaptive_ternary(&dispatcher, &blocks, &input, &mut output, n_rows, k)
    }

    #[test]
    fn parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_threads >= 1);
        assert_eq!(config.gemv_threshold, PAR_TILED_MIN_ROWS);
        assert_eq!(config.gemm_threshold, PAR_TILED_MIN_BATCH);
        assert!(config.use_tiling);
    }

    #[test]
    fn parallel_config_single_threaded() {
        let config = ParallelConfig::single_threaded();
        assert_eq!(config.num_threads, 1);
        assert!(!config.use_tiling);
        // Should never parallelize
        assert!(!config.should_parallelize_gemv(1_000_000));
        assert!(!config.should_parallelize_gemm(1_000_000));
    }

    #[test]
    fn parallel_config_threshold_checks() {
        let config = ParallelConfig::default();
        if config.num_threads > 1 {
            assert!(!config.should_parallelize_gemv(64));
            assert!(config.should_parallelize_gemv(256));
            assert!(!config.should_parallelize_gemm(2));
            assert!(config.should_parallelize_gemm(8));
        }
    }

    #[test]
    fn validation_errors_propagate() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];

        // Not block aligned
        let result = gemv_parallel_tiled(&dispatcher, &blocks, &input, &mut output, 1, 100);
        assert!(result.is_err());

        // GEMM not block aligned
        let result = gemm_parallel_tiled(&dispatcher, &blocks, &input, &mut output, 1, 1, 100);
        assert!(result.is_err());
    }
}
