//! Parallel tiled kernel execution.
//!
//! Combines cache-aware tiling (from the [`tiled`](crate::tiled) module) with Rayon parallelism
//! for maximum throughput. Strategy: outer loop over L2 tiles is parallel,
//! inner loop over L1 tiles is sequential.
//!
//! Also provides an adaptive dispatcher that selects the best strategy
//! (direct, parallel row, or parallel tiled) based on matrix dimensions.
//!
//! ## The opt-in INT8 tier (K-14)
//!
//! Every public entry point here — the adaptive drivers the model's linear
//! layers call (`gemv_adaptive` for `Q1_0_g128`, `gemv_adaptive_ternary` /
//! `gemm_adaptive_ternary` for `TQ2_0_g128`) and the parallel-tiled
//! strategies themselves — asks [`KernelDispatcher::native_int8_tier`]
//! once, **before** any strategy is chosen, and hands the whole call to the
//! matching [`crate::dispatch_int8`] kernel when `OXIBONSAI_KERNEL_TIER`
//! selects one. Those kernels quantize the activation once and parallelize
//! themselves, so they must not be re-entered per tile. When no INT8 tier
//! is selected — the default — the strategy runs exactly as before, its
//! tiles calling the dispatcher's `*_on_tier` methods (the 1-bit GEMM's
//! small-batch fallback calls the register-blocked CPU GEMM instead), none
//! of which re-read the environment.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::dispatch::KernelDispatcher;
use crate::dispatch_int8;
use crate::error::{KernelError, KernelResult};
#[cfg(not(target_arch = "wasm32"))]
use crate::tiled::{optimal_tile_rows, L2_TILE_ROWS};
use crate::tuning::{PlatformProfile, TunedThresholds};
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_core::{BlockTQ2_0_g128, QK_TQ2_0_G128};

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
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::dimension_mismatch("input", k, input.len()));
    }
    if output.len() < n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            n_rows,
            output.len(),
        ));
    }
    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }
    Ok(blocks_per_row)
}

/// Validate ternary GEMV parameters and return blocks_per_row (K-M2).
fn validate_gemv_ternary(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &[f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<usize> {
    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::dimension_mismatch("input", k, input.len()));
    }
    if output.len() < n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            n_rows,
            output.len(),
        ));
    }
    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
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
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::dimension_mismatch("input", m * k, input.len()));
    }
    if output.len() < m * n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            m * n_rows,
            output.len(),
        ));
    }
    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }
    Ok(blocks_per_row)
}

// ─── Parallel tiled kernels ────────────────────────────────────────────

/// Sequential L1-tiled 1-bit GEMV: the same [`crate::tiled::L1_TILE_ROWS`]
/// tiling as [`crate::tiled::gemv_tiled`], but each tile calls the
/// dispatcher's own tier directly instead of re-entering
/// `OneBitKernel::gemv` (the INT8 check already happened at the entry
/// point, so no tile re-reads the environment). Used as the
/// below-threshold and WASM fallback for [`gemv_parallel_tiled`].
fn gemv_tiled_1bit_seq(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
) -> KernelResult<()> {
    let mut row_start = 0;
    while row_start < n_rows {
        let tile_rows = (n_rows - row_start).min(crate::tiled::L1_TILE_ROWS);
        let block_start = row_start * blocks_per_row;
        let block_end = (row_start + tile_rows) * blocks_per_row;

        dispatcher.gemv_1bit_on_tier(
            &blocks[block_start..block_end],
            input,
            &mut output[row_start..row_start + tile_rows],
            tile_rows,
            k,
        )?;

        row_start += tile_rows;
    }
    Ok(())
}

/// Parallel tiled GEMV: distribute L2 tiles across threads.
///
/// Each thread receives an L2-sized chunk of output rows and processes it
/// using L1 tiling internally. For problems below the platform-tuned
/// `par_tiled_min_rows` threshold, falls back to sequential tiled execution
/// (the same L1 tiling as [`crate::tiled::gemv_tiled`], on the
/// dispatcher's own tier).
///
/// The L1 tile size is dynamically computed via [`optimal_tile_rows`] to
/// account for the actual working set size at the given `k`.
///
/// Runs on the opt-in INT8 tier instead when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc).
pub fn gemv_parallel_tiled(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return dispatch_int8::gemv_1bit_g128_int8(tier, blocks, input, output, n_rows, k);
    }
    gemv_parallel_tiled_on_tier(dispatcher, blocks, input, output, n_rows, k)
}

/// [`gemv_parallel_tiled`] on the dispatcher's own tier, without the INT8
/// entry check.
fn gemv_parallel_tiled_on_tier(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_gemv(blocks, input, output, n_rows, k)?;

    // Sequential fallback for small row counts (platform-tuned threshold).
    if n_rows < PlatformProfile::global_thresholds().par_tiled_min_rows {
        return gemv_tiled_1bit_seq(dispatcher, blocks, input, output, n_rows, k, blocks_per_row);
    }

    // On WASM: no rayon threads available — fall back to sequential tiled.
    #[cfg(target_arch = "wasm32")]
    {
        gemv_tiled_1bit_seq(dispatcher, blocks, input, output, n_rows, k, blocks_per_row)
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

                    dispatcher.gemv_1bit_on_tier(
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
///
/// Runs on the opt-in INT8 tier instead when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc).
pub fn gemm_parallel_tiled(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return dispatch_int8::gemm_1bit_g128_int8(tier, blocks, input, output, m, n_rows, k);
    }
    gemm_parallel_tiled_on_tier(dispatcher, blocks, input, output, m, n_rows, k)
}

/// [`gemm_parallel_tiled`] on the dispatcher's own tier, without the INT8
/// entry check.
fn gemm_parallel_tiled_on_tier(
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
        crate::tiled::gemm_tiled(dispatcher, blocks, input, output, m, n_rows, k)
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

                    dispatcher.gemm_1bit_on_tier(
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

// ─── Parallel tiled ternary kernel (K-M2) ──────────────────────────────

/// A ternary block is `qs: [u8; 32]` + `d: f16` = 34 bytes, roughly double
/// the 1-bit format's 18 bytes for the same 128-weight group.
const TERNARY_BLOCK_BYTES: usize = 34;

/// Minimum L1 tile size for ternary GEMV (K-M2), measured.
///
/// The naive L1-budget formula in [`optimal_tile_rows_ternary`] — mirroring
/// [`crate::tiled::optimal_tile_rows`] — degenerates badly at the widths
/// this crate actually ships: at `k=5120` (Bonsai 2's `embedding_length`)
/// it computes 33 rows/tile, and at `k=17408` (`feed_forward_length`) the
/// shared input vector alone (`k*4` bytes) exceeds the assumed L1 budget,
/// saturating the available space to zero. Measuring
/// `measure_gemv_ternary_par_efficiency` at the 33-row tile size showed
/// *worse* throughput than the untiled K-16 path (3.31x vs 4.03x parallel
/// speedup on 8 cores) — small enough tiles interrupt the NEON kernel's
/// row-to-row software prefetch stream (`gemv_tq2_0_g128_neon_prefetch`)
/// more often than the extra cache locality pays for. Flooring at 128 rows
/// closed the gap (4.11x tiled vs 4.29x flat, adaptive routing hitting
/// 4.29x) without materially changing the tile count at narrower `k` (a
/// 256-wide layer already saturates at `L2_TILE_ROWS` regardless of this
/// floor). This is the "measurement" the K-M2 finding's fallback text asks
/// for if plain L1-budget tiling does not help — it does, once floored.
const TERNARY_TILE_MIN_ROWS: usize = 128;

/// L1-cache-target tile row count for ternary GEMV.
///
/// Mirrors [`crate::tiled::optimal_tile_rows`]'s L1-budget formula but with
/// the ternary block's actual size substituted for the 1-bit format's 18
/// bytes that function hardcodes — reusing it as-is would under-count a
/// ternary row's footprint by roughly 2x, picking tiles nearly twice as
/// large as actually fit L1. The lower clamp is `TERNARY_TILE_MIN_ROWS`,
/// not the 1-bit path's `4` — see its doc comment for the measurement that
/// motivated raising it.
///
/// `pub` (like [`crate::tiled::optimal_tile_rows`]) rather than
/// WASM-cfg-gated: the computation is plain arithmetic with no Rayon/thread
/// dependency, so unlike [`gemv_parallel_tiled_ternary`] (its only current
/// caller, from a `cfg(not(wasm32))` block) there is no platform reason to
/// exclude it, and a public item is exempt from `dead_code` regardless of
/// in-crate call sites.
pub fn optimal_tile_rows_ternary(k: usize) -> usize {
    let blocks_per_row = k / QK_TQ2_0_G128;
    let bytes_per_row = blocks_per_row * TERNARY_BLOCK_BYTES;
    let l1_bytes = PlatformProfile::global().l1_cache_bytes;
    let l1_available = l1_bytes.saturating_sub(k * 4);
    let l1_rows = l1_available
        .checked_div(bytes_per_row)
        .unwrap_or(crate::tiled::L1_TILE_ROWS);
    l1_rows.clamp(TERNARY_TILE_MIN_ROWS, crate::tiled::L2_TILE_ROWS)
}

/// Sequential L1-tiled ternary GEMV: the ternary counterpart of
/// [`crate::tiled::gemv_tiled`], calling the dispatcher's own tier (the
/// INT8 check already happened at the entry point). Used as the
/// below-threshold and WASM fallback for [`gemv_parallel_tiled_ternary`].
fn gemv_tiled_ternary_seq(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
) -> KernelResult<()> {
    let mut row_start = 0;
    while row_start < n_rows {
        let tile_rows = (n_rows - row_start).min(crate::tiled::L1_TILE_ROWS);
        let block_start = row_start * blocks_per_row;
        let block_end = (row_start + tile_rows) * blocks_per_row;

        dispatcher.gemv_ternary_on_tier(
            &blocks[block_start..block_end],
            input,
            &mut output[row_start..row_start + tile_rows],
            tile_rows,
            k,
        )?;

        row_start += tile_rows;
    }
    Ok(())
}

/// Parallel tiled ternary GEMV: distribute L2 tiles across threads, with
/// L1-cache-aware tiling inside each (K-M2).
///
/// Before this function existed, `gemv_adaptive_ternary` collapsed its
/// `ParallelRow` and `ParallelTiled` arms onto the flat, row-parallel
/// [`crate::parallel::gemv_ternary_g128_par`] — so for the ternary format
/// (the flagship quantization this crate is named after), the cache-tiled
/// strategy `select_gemv_strategy` picked for large matrices (including the
/// 248,320-row Bonsai 2 LM head) was computed and then silently discarded.
/// This mirrors [`gemv_parallel_tiled`]'s structure exactly, parameterized
/// for the ternary block type/size instead of the 1-bit format.
///
/// Runs on the opt-in INT8 tier instead when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc).
pub fn gemv_parallel_tiled_ternary(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return dispatch_int8::gemv_two_bit_int8(tier, blocks, input, output, n_rows, k);
    }
    gemv_parallel_tiled_ternary_on_tier(dispatcher, blocks, input, output, n_rows, k)
}

/// [`gemv_parallel_tiled_ternary`] on the dispatcher's own tier, without
/// the INT8 entry check.
fn gemv_parallel_tiled_ternary_on_tier(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_gemv_ternary(blocks, input, output, n_rows, k)?;

    // Sequential fallback for small row counts (platform-tuned threshold).
    if n_rows < PlatformProfile::global_thresholds().par_tiled_min_rows {
        return gemv_tiled_ternary_seq(
            dispatcher,
            blocks,
            input,
            output,
            n_rows,
            k,
            blocks_per_row,
        );
    }

    // On WASM: no rayon threads available — fall back to sequential tiled.
    #[cfg(target_arch = "wasm32")]
    {
        gemv_tiled_ternary_seq(dispatcher, blocks, input, output, n_rows, k, blocks_per_row)
    }

    // Compute optimal L1 tile size for this k.
    #[cfg(not(target_arch = "wasm32"))]
    let l1_tile = optimal_tile_rows_ternary(k).max(1);

    // Parallel L2 tiles, each internally using L1 tiling.
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

                // Apply L1 tiling within this L2 tile.
                let mut l1_start = 0;
                while l1_start < tile_rows {
                    let l1_rows = (tile_rows - l1_start).min(l1_tile);
                    let l1_block_start = l1_start * blocks_per_row;
                    let l1_block_end = (l1_start + l1_rows) * blocks_per_row;

                    dispatcher.gemv_ternary_on_tier(
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
///
/// Before any strategy is chosen, the opt-in INT8 tier takes the whole call
/// when [`KernelDispatcher::native_int8_tier`] selects one (see the module
/// doc). This is `Linear1Bit::forward_vec`'s decode path.
pub fn gemv_adaptive(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return dispatch_int8::gemv_1bit_g128_int8(tier, blocks, input, output, n_rows, k);
    }
    match select_gemv_strategy(n_rows, k) {
        AdaptiveStrategy::Direct => dispatcher.gemv_1bit_on_tier(blocks, input, output, n_rows, k),
        AdaptiveStrategy::ParallelRow => crate::parallel::gemv_1bit_g128_par_on_tier(
            dispatcher, blocks, input, output, n_rows, k,
        ),
        AdaptiveStrategy::ParallelTiled => {
            gemv_parallel_tiled_on_tier(dispatcher, blocks, input, output, n_rows, k)
        }
    }
}

/// Adaptive ternary GEMV dispatch (K-M2): unlike the earlier version of this
/// function, `ParallelTiled` now genuinely reaches the cache-tiled path
/// ([`gemv_parallel_tiled_ternary`]) instead of being collapsed onto the
/// flat row-parallel one — see that function's docs for why this mattered.
/// `ParallelRow` still uses [`crate::parallel::gemv_ternary_g128_par`],
/// which K-16 already changed from one Rayon task per row to a
/// platform-tuned number of rows per task.
///
/// Before any strategy is chosen, the opt-in INT8 tier (the legacy
/// `{-1, 0, +1, 0}` table, K-01) takes the whole call when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc).
/// This is `LinearTernary::forward`'s decode path.
pub fn gemv_adaptive_ternary(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return dispatch_int8::gemv_two_bit_int8(tier, blocks, input, output, n_rows, k);
    }
    match select_gemv_strategy(n_rows, k) {
        AdaptiveStrategy::Direct => {
            dispatcher.gemv_ternary_on_tier(blocks, input, output, n_rows, k)
        }
        AdaptiveStrategy::ParallelRow => crate::parallel::gemv_ternary_g128_par_on_tier(
            dispatcher, blocks, input, output, n_rows, k,
        ),
        AdaptiveStrategy::ParallelTiled => {
            gemv_parallel_tiled_ternary_on_tier(dispatcher, blocks, input, output, n_rows, k)
        }
    }
}

/// Adaptive ternary GEMM: the dispatcher's own GEMM below the platform-tuned
/// `par_gemm_min_batch`, the batch-slab register-blocked
/// [`crate::parallel::gemm_ternary_g128_par`] driver at or above it.
///
/// Before either is chosen, the opt-in INT8 tier takes the whole call when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc).
/// This is `LinearTernary::forward_batch`'s path.
pub fn gemm_adaptive_ternary(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return dispatch_int8::gemm_two_bit_int8(tier, blocks, input, output, m, n_rows, k);
    }
    if m < PlatformProfile::global_thresholds().par_gemm_min_batch {
        dispatcher.gemm_ternary_on_tier(blocks, input, output, m, n_rows, k)
    } else {
        crate::parallel::gemm_ternary_g128_par_on_tier(
            dispatcher, blocks, input, output, m, n_rows, k,
        )
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
    use crate::traits::{OneBitKernel, TernaryKernel};
    use half::f16;

    /// Deterministic regression guard for the measured tile-floor fix
    /// (`TERNARY_TILE_MIN_ROWS`'s doc comment): at both of the widths this
    /// crate actually ships (Bonsai 2's `embedding_length=5120` and
    /// `feed_forward_length=17408`), the tile size must never fall back
    /// below the measured floor, which is what caused the tiled path to
    /// lose to the flat one (3.31x vs 4.03x parallel speedup) before this
    /// fix. A future edit to the L1-budget formula that reintroduces a
    /// tiny-tile regression fails this test immediately, without needing
    /// the timing-based `measure_gemv_ternary_par_efficiency`.
    #[test]
    fn optimal_tile_rows_ternary_never_below_measured_floor() {
        for k in [5120usize, 17408] {
            let rows = optimal_tile_rows_ternary(k);
            assert!(
                rows >= TERNARY_TILE_MIN_ROWS,
                "k={k}: optimal_tile_rows_ternary returned {rows}, below the \
                 measured floor of {TERNARY_TILE_MIN_ROWS} that keeps the tiled \
                 path competitive with the flat one"
            );
            assert!(rows <= crate::tiled::L2_TILE_ROWS);
        }
    }

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

    /// T-09: the original version of this test only checked that
    /// `gemv_adaptive_ternary` returned `Ok`, which passes identically
    /// whether `select_gevm_strategy` (sic) routes correctly, routes
    /// backwards, or is deleted outright. It now asserts (a) the routing
    /// decision `select_gemv_strategy` actually makes for this shape is
    /// `Direct`, and (b) the adaptive call's output matches a direct
    /// dispatcher call bit-for-bit (both paths reduce to the exact same
    /// `dispatcher.gemv_ternary_g128` call for `n_rows` below every tuned
    /// threshold, so there is no floating-point reordering to tolerate).
    #[test]
    fn adaptive_ternary_gemv_small_is_direct() -> KernelResult<()> {
        let n_rows = 16;
        let k = 128;
        assert_eq!(
            select_gemv_strategy(n_rows, k),
            AdaptiveStrategy::Direct,
            "test fixture assumption: n_rows={n_rows} must be below par_gemv_min_rows on this host"
        );

        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks: Vec<_> = (0..n_rows * blocks_per_row)
            .map(|i| make_ternary_block([((i * 29 + 3) & 0xFF) as u8; 32]))
            .collect();
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        dispatcher.gemv_ternary_g128(&blocks, &input, &mut out_direct, n_rows, k)?;

        let mut out_adaptive = vec![0.0f32; n_rows];
        gemv_adaptive_ternary(&dispatcher, &blocks, &input, &mut out_adaptive, n_rows, k)?;

        for i in 0..n_rows {
            assert_eq!(
                out_direct[i].to_bits(),
                out_adaptive[i].to_bits(),
                "row {i}: direct={}, adaptive={}",
                out_direct[i],
                out_adaptive[i]
            );
        }
        Ok(())
    }

    /// T-09 twin of the above for the large-matrix path: asserts the
    /// resolved strategy is `ParallelTiled` (K-M2's new ternary tiled path,
    /// not the flat row-parallel one it used to collapse onto) and that the
    /// adaptive call's output matches a direct, unparallelized dispatcher
    /// call to within the tolerance the ACCEPTANCE criterion documents for
    /// this crate's parallel routing (rows are independent reductions, so
    /// tiling changes no per-row summation order — see
    /// `strategy_forced_bit_parity_gemv_ternary` below for the bit-exact
    /// version of this same claim across all three strategies).
    #[test]
    fn adaptive_ternary_gemv_large_is_parallel() -> KernelResult<()> {
        let n_rows = 10_000;
        let k = 128;
        assert_eq!(
            select_gemv_strategy(n_rows, k),
            AdaptiveStrategy::ParallelTiled,
            "test fixture assumption: n_rows={n_rows} must be above par_tiled_min_rows on this host"
        );

        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks: Vec<_> = (0..n_rows * blocks_per_row)
            .map(|i| make_ternary_block([((i * 29 + 3) & 0xFF) as u8; 32]))
            .collect();
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        dispatcher.gemv_ternary_g128(&blocks, &input, &mut out_direct, n_rows, k)?;

        let mut out_adaptive = vec![0.0f32; n_rows];
        gemv_adaptive_ternary(&dispatcher, &blocks, &input, &mut out_adaptive, n_rows, k)?;

        for i in 0..n_rows {
            assert_eq!(
                out_direct[i].to_bits(),
                out_adaptive[i].to_bits(),
                "row {i}: direct={}, adaptive(ParallelTiled)={}",
                out_direct[i],
                out_adaptive[i]
            );
        }
        Ok(())
    }

    /// K-M2 wiring proof: forcing Direct / ParallelRow (K-16's chunked
    /// `gemv_ternary_g128_par`) / ParallelTiled (the new
    /// `gemv_parallel_tiled_ternary`) on identical input must produce
    /// bit-for-bit identical output, and `select_gemv_strategy` must
    /// actually resolve to `ParallelTiled` for the large shape — pinning
    /// both the numeric invariant and the routing decision the way
    /// `strategy_forced_bit_parity_gemv` already does for the 1-bit format.
    #[test]
    fn strategy_forced_bit_parity_gemv_ternary() -> KernelResult<()> {
        let n_rows = 10_000;
        let k = 256;
        assert_eq!(
            select_gemv_strategy(n_rows, k),
            AdaptiveStrategy::ParallelTiled
        );

        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks: Vec<_> = (0..n_rows * blocks_per_row)
            .map(|i| make_ternary_block([((i * 53 + 11) & 0xFF) as u8; 32]))
            .collect();
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.003) - 0.4).collect();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        dispatcher.gemv_ternary_g128(&blocks, &input, &mut out_direct, n_rows, k)?;

        let mut out_parallel_row = vec![0.0f32; n_rows];
        crate::parallel::gemv_ternary_g128_par(
            &dispatcher,
            &blocks,
            &input,
            &mut out_parallel_row,
            n_rows,
            k,
        )?;

        let mut out_parallel_tiled = vec![0.0f32; n_rows];
        gemv_parallel_tiled_ternary(
            &dispatcher,
            &blocks,
            &input,
            &mut out_parallel_tiled,
            n_rows,
            k,
        )?;

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
        Ok(())
    }

    #[test]
    fn gemv_parallel_tiled_ternary_matches_direct() -> KernelResult<()> {
        let n_rows = 600;
        let k = 256;
        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks: Vec<_> = (0..n_rows * blocks_per_row)
            .map(|i| make_ternary_block([((i * 19 + 5) & 0xFF) as u8; 32]))
            .collect();
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        dispatcher.gemv_ternary_g128(&blocks, &input, &mut out_direct, n_rows, k)?;

        let mut out_tiled = vec![0.0f32; n_rows];
        gemv_parallel_tiled_ternary(&dispatcher, &blocks, &input, &mut out_tiled, n_rows, k)?;

        for i in 0..n_rows {
            assert_eq!(
                out_direct[i].to_bits(),
                out_tiled[i].to_bits(),
                "row {i}: direct={}, tiled={}",
                out_direct[i],
                out_tiled[i]
            );
        }
        Ok(())
    }

    #[test]
    fn gemv_parallel_tiled_ternary_small_falls_back_to_sequential_tiled() -> KernelResult<()> {
        let n_rows = 8;
        let k = 128;
        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks: Vec<_> = (0..n_rows * blocks_per_row)
            .map(|i| make_ternary_block([((i * 7 + 1) & 0xFF) as u8; 32]))
            .collect();
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        dispatcher.gemv_ternary_g128(&blocks, &input, &mut out_direct, n_rows, k)?;

        let mut out_tiled = vec![0.0f32; n_rows];
        gemv_parallel_tiled_ternary(&dispatcher, &blocks, &input, &mut out_tiled, n_rows, k)?;

        for i in 0..n_rows {
            assert_eq!(out_direct[i].to_bits(), out_tiled[i].to_bits(), "row {i}");
        }
        Ok(())
    }

    #[test]
    fn gemv_parallel_tiled_ternary_validation_errors() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = vec![make_ternary_block([0xAAu8; 32])];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];

        // Not block aligned.
        let result = gemv_parallel_tiled_ternary(&dispatcher, &blocks, &input, &mut output, 1, 100);
        assert!(result.is_err());

        // Output too small.
        let mut tiny_output = vec![0.0f32; 0];
        let result =
            gemv_parallel_tiled_ternary(&dispatcher, &blocks, &input, &mut tiny_output, 1, 128);
        assert!(result.is_err());
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

    /// K-02: this file's migrated construction sites (1-bit and ternary
    /// validators) name the buffer they complained about.
    #[test]
    fn migrated_errors_name_the_offending_buffer() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = vec![make_block(1.0, [0xFF; 16]); 4];

        let short_input = vec![1.0f32; 10];
        let mut output = vec![0.0f32; 4];
        let err = gemv_parallel_tiled(&dispatcher, &blocks, &short_input, &mut output, 4, 128)
            .unwrap_err();
        assert_eq!(err.buffer_name(), Some("input"));

        let input = vec![1.0f32; 128];
        let mut short_output = vec![0.0f32; 0];
        let err = gemv_parallel_tiled(&dispatcher, &blocks, &input, &mut short_output, 4, 128)
            .unwrap_err();
        assert_eq!(err.buffer_name(), Some("output"));

        let mut output = vec![0.0f32; 8];
        let err =
            gemv_parallel_tiled(&dispatcher, &blocks, &input, &mut output, 8, 128).unwrap_err();
        assert_eq!(err.buffer_name(), Some("blocks"));

        let ternary_blocks = vec![make_ternary_block([0xAAu8; 32]); 4];
        let short_input = vec![1.0f32; 10];
        let mut output = vec![0.0f32; 4];
        let err = gemv_parallel_tiled_ternary(
            &dispatcher,
            &ternary_blocks,
            &short_input,
            &mut output,
            4,
            128,
        )
        .unwrap_err();
        assert_eq!(err.buffer_name(), Some("input"));
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

    // ─── The opt-in INT8 tier at every native entry point ──────────────

    /// Deterministic ternary blocks with every code (incl. the reserved
    /// `0b11`) and varied scales.
    fn int8_ternary_blocks(n: usize, seed: u32) -> Vec<BlockTQ2_0_g128> {
        let mut state = seed | 1;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 19) as u8
        };
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 32];
                for b in &mut qs {
                    *b = next();
                }
                BlockTQ2_0_g128 {
                    qs,
                    d: f16::from_f32(0.0625 + (next() % 16) as f32 / 256.0),
                }
            })
            .collect()
    }

    fn int8_inputs(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as i32 % 2001 - 1000) as f32 / 512.0
            })
            .collect()
    }

    fn assert_bits(expect: &[f32], got: &[f32], what: &str) {
        assert_eq!(expect.len(), got.len(), "{what}: length");
        for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
            assert_eq!(e.to_bits(), g.to_bits(), "{what}: element {i}: {e} vs {g}");
        }
    }

    /// Every public native-format entry point in this file and in
    /// `parallel.rs` hands the whole call to the INT8 kernel of the selected
    /// tier when `OXIBONSAI_KERNEL_TIER` names one — bit-identical to calling
    /// that kernel directly — and returns to the dispatcher's own f32 tier
    /// once the variable is cleared.
    #[test]
    fn native_entry_points_route_to_the_selected_int8_tier() {
        use crate::dispatch_int8::{
            gemm_1bit_g128_int8, gemm_two_bit_int8, gemv_1bit_g128_int8, gemv_two_bit_int8,
            Int8Tier, TierEnvGuard, KERNEL_TIER_ENV,
        };

        let _guard = TierEnvGuard::acquire();
        let dispatcher = KernelDispatcher::with_tier(crate::dispatch::cpu_kernel_tier());
        let (n_rows, k, m) = (300usize, 2 * QK_TQ2_0_G128, 3usize);
        let tq2 = int8_ternary_blocks(n_rows * (k / QK_TQ2_0_G128), 0x71E1);
        let (q1, _) = make_test_data(n_rows, k);
        let gemv_in = int8_inputs(k, 0x71E2);
        let gemm_in = int8_inputs(m * k, 0x71E3);
        let tier = Int8Tier::best_available();

        let mut ternary_gemv = vec![0.0f32; n_rows];
        gemv_two_bit_int8(tier, &tq2, &gemv_in, &mut ternary_gemv, n_rows, k).expect("direct");
        let mut ternary_gemm = vec![0.0f32; m * n_rows];
        gemm_two_bit_int8(tier, &tq2, &gemm_in, &mut ternary_gemm, m, n_rows, k).expect("direct");
        let mut one_bit_gemv = vec![0.0f32; n_rows];
        gemv_1bit_g128_int8(tier, &q1, &gemv_in, &mut one_bit_gemv, n_rows, k).expect("direct");
        let mut one_bit_gemm = vec![0.0f32; m * n_rows];
        gemm_1bit_g128_int8(tier, &q1, &gemm_in, &mut one_bit_gemm, m, n_rows, k).expect("direct");

        // Every entry point's `(name, output)` on dispatcher `d`, in the
        // order of `expected` below; run once with the tier selected and
        // once with the variable cleared.
        let run_all = |d: &KernelDispatcher| -> Vec<(&'static str, Vec<f32>)> {
            let mut outs = Vec::new();
            let mut record =
                |name: &'static str, f: &dyn Fn(&mut [f32]) -> KernelResult<()>, len: usize| {
                    let mut out = vec![0.0f32; len];
                    f(&mut out).unwrap_or_else(|e| panic!("{name}: {e}"));
                    outs.push((name, out));
                };
            record(
                "gemv_adaptive_ternary",
                &|o| gemv_adaptive_ternary(d, &tq2, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "gemv_parallel_tiled_ternary",
                &|o| gemv_parallel_tiled_ternary(d, &tq2, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "parallel::gemv_ternary_g128_par",
                &|o| crate::parallel::gemv_ternary_g128_par(d, &tq2, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "TernaryKernel::gemv_ternary_g128",
                &|o| d.gemv_ternary_g128(&tq2, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "gemm_adaptive_ternary",
                &|o| gemm_adaptive_ternary(d, &tq2, &gemm_in, o, m, n_rows, k),
                m * n_rows,
            );
            record(
                "parallel::gemm_ternary_g128_par",
                &|o| crate::parallel::gemm_ternary_g128_par(d, &tq2, &gemm_in, o, m, n_rows, k),
                m * n_rows,
            );
            record(
                "TernaryKernel::gemm_ternary_g128",
                &|o| d.gemm_ternary_g128(&tq2, &gemm_in, o, m, n_rows, k),
                m * n_rows,
            );
            record(
                "gemv_adaptive",
                &|o| gemv_adaptive(d, &q1, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "gemv_parallel_tiled",
                &|o| gemv_parallel_tiled(d, &q1, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "parallel::gemv_1bit_g128_par",
                &|o| crate::parallel::gemv_1bit_g128_par(d, &q1, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "OneBitKernel::gemv",
                &|o| d.gemv(&q1, &gemv_in, o, n_rows, k),
                n_rows,
            );
            record(
                "gemm_parallel_tiled",
                &|o| gemm_parallel_tiled(d, &q1, &gemm_in, o, m, n_rows, k),
                m * n_rows,
            );
            record(
                "parallel::gemm_1bit_g128_par",
                &|o| crate::parallel::gemm_1bit_g128_par(d, &q1, &gemm_in, o, m, n_rows, k),
                m * n_rows,
            );
            record(
                "OneBitKernel::gemm",
                &|o| d.gemm(&q1, &gemm_in, o, m, n_rows, k),
                m * n_rows,
            );
            outs
        };
        let expected: [&[f32]; 14] = [
            &ternary_gemv,
            &ternary_gemv,
            &ternary_gemv,
            &ternary_gemv,
            &ternary_gemm,
            &ternary_gemm,
            &ternary_gemm,
            &one_bit_gemv,
            &one_bit_gemv,
            &one_bit_gemv,
            &one_bit_gemv,
            &one_bit_gemm,
            &one_bit_gemm,
            &one_bit_gemm,
        ];

        // SAFETY: `_guard` holds the crate's env lock (see `TierEnvGuard`).
        unsafe {
            std::env::set_var(KERNEL_TIER_ENV, tier.name());
        }
        assert_eq!(dispatcher.native_int8_tier(), Some(tier));
        let selected = run_all(&dispatcher);
        // SAFETY: as above.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        assert_eq!(dispatcher.native_int8_tier(), None);
        let cleared = run_all(&dispatcher);

        assert_eq!(selected.len(), expected.len());
        for ((name, sel), ((_, clr), exp)) in
            selected.iter().zip(cleared.iter().zip(expected.iter()))
        {
            assert_bits(exp, sel, &format!("{name} with {KERNEL_TIER_ENV}={tier}"));
            assert!(
                sel.iter()
                    .zip(clr.iter())
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "{name}: clearing {KERNEL_TIER_ENV} did not return to the f32 path"
            );
        }
    }

    /// A `KernelTier::Gpu` dispatcher is never diverted, whatever the
    /// variable says.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_gpu_tier_dispatcher_is_never_diverted() {
        use crate::dispatch::KernelTier;
        use crate::dispatch_int8::{Int8Tier, TierEnvGuard, KERNEL_TIER_ENV};

        let _guard = TierEnvGuard::acquire();
        let gpu = KernelDispatcher::with_tier(KernelTier::Gpu);
        let (n_rows, k) = (64usize, 2 * QK_TQ2_0_G128);
        let tq2 = int8_ternary_blocks(n_rows * (k / QK_TQ2_0_G128), 0x71E4);
        let input = int8_inputs(k, 0x71E5);
        let mut cleared = vec![0.0f32; n_rows];
        gemv_adaptive_ternary(&gpu, &tq2, &input, &mut cleared, n_rows, k).expect("cleared");
        // SAFETY: `_guard` holds the crate's env lock.
        unsafe {
            std::env::set_var(KERNEL_TIER_ENV, Int8Tier::best_available().name());
        }
        assert_eq!(gpu.native_int8_tier(), None);
        let mut selected = vec![0.0f32; n_rows];
        gemv_adaptive_ternary(&gpu, &tq2, &input, &mut selected, n_rows, k).expect("selected");
        assert_bits(&cleared, &selected, "Gpu-tier gemv_adaptive_ternary");
    }
}
