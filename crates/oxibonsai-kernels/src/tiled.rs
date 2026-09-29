//! Cache-aware tiled GEMV and GEMM computations.
//!
//! Strategy: divide output rows into tiles that fit in L1/L2 cache,
//! then process each tile sequentially to maximize cache locality.
//! For GEMM, also tile the batch dimension.
//!
//! **Cache hierarchy awareness:**
//! - L1 tile: 32 rows x 128 elements x 4 bytes = 16 KB (fits L1d, typically 32-64 KB)
//! - L2 tile: 256 rows for L2 tiling (fits L2, typically 256 KB - 1 MB)
//!
//! The tiling interacts with the kernel dispatcher, calling the
//! tier-appropriate SIMD kernel on each tile. This gives us cache
//! optimization without duplicating SIMD code for each tier.
//!
//! ## The opt-in INT8 tier (K-14)
//!
//! [`gemv_tiled`] and [`gemv_tiled_par`] ask
//! [`KernelDispatcher::native_int8_tier`] once, at entry, exactly like
//! `parallel::gemv_1bit_g128_par` — and, when it selects a tier, hand the
//! *whole* (untiled) call straight to `dispatch_int8::gemv_1bit_g128_int8`
//! instead of tiling at all. Before this, each L1/L2 tile called
//! `OneBitKernel::gemv` independently, which already carries its own INT8
//! check: with the tier selected, every tile would re-quantize the
//! identical, untiled `input` activation on its own — same values, wasted
//! work. With the tier unset, both functions' tiling loops now call the
//! dispatcher's `pub(crate)` `gemv_1bit_on_tier` per tile directly — the
//! tier-only f32 body `OneBitKernel::gemv` itself falls back to once its own
//! entry check finds nothing selected — rather than re-entering `gemv`
//! (and so re-checking the environment) once per tile.
//!
//! [`gemm_tiled`] and [`gemm_tiled_par`] need no equivalent change: they
//! call the free function `gemm_1bit_g128_blocked` directly, never
//! `OneBitKernel::gemm`, so they never reach the INT8 selector at all (see
//! that function's own doc for why).

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::dispatch::KernelDispatcher;
use crate::error::{KernelError, KernelResult};
use crate::gemm_onebit::gemm_1bit_g128_blocked;
// Only the tests call `KernelDispatcher::gemv`/`gemm` directly any more —
// both production entry points now call `gemv_1bit_on_tier` directly or the
// once-at-entry INT8 check, neither of which needs this trait import.
#[cfg(test)]
use crate::traits::OneBitKernel;

/// Number of rows per L1-sized tile.
/// 32 rows x 128 elements x 4 bytes = 16 KB, fitting within most L1d caches.
pub const L1_TILE_ROWS: usize = 32;

/// Number of rows per L2-sized tile.
/// 256 rows for L2-level tiling, balancing parallelism with cache residency.
pub const L2_TILE_ROWS: usize = 256;

/// Minimum number of rows before engaging parallel tiled GEMV.
const PAR_TILED_GEMV_MIN_ROWS: usize = 64;

/// Minimum batch size before engaging parallel tiled GEMM.
const PAR_TILED_GEMM_MIN_BATCH: usize = 4;

/// Batch tile size for GEMM: number of batch elements processed per tile.
const GEMM_BATCH_TILE: usize = 8;

// ─── Validation helpers ────────────────────────────────────────────────

/// Validate GEMV parameters and return blocks_per_row.
fn validate_gemv_params(
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
fn validate_gemm_params(
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

// ─── Sequential tiled kernels ──────────────────────────────────────────

/// Tiled GEMV: divide `n_rows` into L1-friendly tiles.
///
/// Each tile processes `L1_TILE_ROWS` rows at a time, keeping the
/// weight data for those rows hot in L1 cache while scanning the
/// shared input vector.
///
/// For small `n_rows` (< `L1_TILE_ROWS`), this degrades gracefully
/// to a single-tile call.
///
/// Asks [`KernelDispatcher::native_int8_tier`] once, before any tiling —
/// see this module's doc comment.
pub fn gemv_tiled(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return crate::dispatch_int8::gemv_1bit_g128_int8(tier, blocks, input, output, n_rows, k);
    }
    gemv_tiled_f32(dispatcher, blocks, input, output, n_rows, k)
}

/// The L1-tiling loop [`gemv_tiled`] runs once its own entry check finds no
/// INT8 tier selected.
///
/// Split out so [`gemv_tiled_par`]'s small-`n_rows`/WASM fallback can call it
/// directly instead of re-entering [`gemv_tiled`] and so re-reading
/// [`KernelDispatcher::native_int8_tier`] a second time for the same call —
/// harmless (the answer cannot have changed in between), but this way each
/// public entry point still asks exactly once, matching the module doc.
fn gemv_tiled_f32(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_gemv_params(blocks, input, output, n_rows, k)?;

    // Process rows in L1-sized tiles
    let mut row_start = 0;
    while row_start < n_rows {
        let tile_rows = (n_rows - row_start).min(L1_TILE_ROWS);
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

/// Tiled GEMM: tile both `m` (batch) and `n_rows` dimensions.
///
/// **Two-level tiling:**
/// 1. Outer loop tiles the batch dimension by `GEMM_BATCH_TILE`.
/// 2. Inner loop tiles the weight rows by `L1_TILE_ROWS`.
///
/// This ensures that for each batch tile, the weight rows cycle
/// through L1 cache, and the input tile stays resident in L2.
///
/// **This is a CPU driver (K-18, PERF-CPU-PREFILL).** It used to call
/// `dispatcher.gemm(...)`, which for a GPU-tier dispatcher with
/// `n_rows >= GPU_MIN_ROWS` reached Metal/CUDA. It now always calls
/// [`gemm_1bit_g128_blocked`], which maps a GPU-tier dispatcher onto its
/// best CPU tier the same way [`crate::parallel::gemm_1bit_g128_par`] does
/// — so on a build where `KernelDispatcher::tier` would pick `Gpu`, this
/// function no longer touches the GPU at all. Callers that want the
/// Metal/CUDA GEMM call `KernelDispatcher::gemm` (or the model's fused GPU
/// prefill) directly.
pub fn gemm_tiled(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    validate_gemm_params(blocks, input, output, m, n_rows, k)?;

    // Tile the batch dimension only. The weight-row tiling this loop used
    // to do existed to keep a tile's weights hot in L1 across the batch
    // loop — but it did that by calling the kernel with `m = 1` per batch
    // element (K-18), so the decoded weights were thrown away between
    // batch elements and only the *compressed* bytes stayed cached.
    // `gemm_1bit_g128_blocked` keeps the batch loop innermost and the
    // decoded block live across `ONEBIT_GEMM_MR` rows, which subsumes the
    // row tiling: the weight row is re-read from L1, never re-decoded.
    let mut batch_start = 0;
    while batch_start < m {
        let batch_tile = (m - batch_start).min(GEMM_BATCH_TILE);
        gemm_1bit_g128_blocked(
            dispatcher,
            blocks,
            &input[batch_start * k..(batch_start + batch_tile) * k],
            &mut output[batch_start * n_rows..(batch_start + batch_tile) * n_rows],
            batch_tile,
            n_rows,
            k,
        )?;
        batch_start += batch_tile;
    }

    Ok(())
}

// ─── Parallel tiled kernels ────────────────────────────────────────────

/// Parallel tiled GEMV: combine Rayon with L2-level tiling.
///
/// **Strategy:**
/// 1. Divide rows into L2-sized chunks for Rayon parallelism.
/// 2. Within each Rayon task, apply L1 tiling.
///
/// This gives us coarse-grained parallelism (L2 tiles across cores)
/// with fine-grained cache optimization (L1 tiles within each core).
///
/// Falls back to sequential `gemv_tiled` for small row counts.
///
/// Asks [`KernelDispatcher::native_int8_tier`] once, before any tiling —
/// see this module's doc comment.
pub fn gemv_tiled_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return crate::dispatch_int8::gemv_1bit_g128_int8(tier, blocks, input, output, n_rows, k);
    }

    #[cfg(not(target_arch = "wasm32"))]
    let blocks_per_row = validate_gemv_params(blocks, input, output, n_rows, k)?;
    #[cfg(target_arch = "wasm32")]
    let _blocks_per_row = validate_gemv_params(blocks, input, output, n_rows, k)?;

    // Sequential fallback for small row counts. Calls the tier-check-free
    // `gemv_tiled_f32` directly, not `gemv_tiled`: the entry check just
    // above already asked `native_int8_tier()` and found `None`, so
    // re-entering `gemv_tiled` would only re-read the same answer.
    if n_rows < PAR_TILED_GEMV_MIN_ROWS {
        return gemv_tiled_f32(dispatcher, blocks, input, output, n_rows, k);
    }

    // On WASM: no rayon threads — fall back to sequential tiled (same
    // already-known-`None` reasoning as the small-row-count fallback above).
    #[cfg(target_arch = "wasm32")]
    {
        gemv_tiled_f32(dispatcher, blocks, input, output, n_rows, k)
    }

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

                // Apply L1 tiling within this L2 tile
                let tile_blocks = &blocks[block_start..block_end];
                let mut l1_start = 0;
                while l1_start < tile_rows {
                    let l1_rows = (tile_rows - l1_start).min(L1_TILE_ROWS);
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

/// Parallel tiled GEMM: combine Rayon with two-level tiling.
///
/// Parallelizes over the batch dimension at L2 granularity,
/// with L1 tiling on the weight rows within each parallel task.
///
/// Falls back to sequential `gemm_tiled` for small batch sizes. Same
/// CPU-only note as [`gemm_tiled`], which every slab here calls.
pub fn gemm_tiled_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    validate_gemm_params(blocks, input, output, m, n_rows, k)?;

    // Sequential fallback for small batch sizes
    if m < PAR_TILED_GEMM_MIN_BATCH {
        return gemm_tiled(dispatcher, blocks, input, output, m, n_rows, k);
    }

    // On WASM: no rayon threads — fall back to sequential tiled.
    #[cfg(target_arch = "wasm32")]
    {
        gemm_tiled(dispatcher, blocks, input, output, m, n_rows, k)
    }

    // Parallel over *slabs* of batch elements rather than one task per
    // batch element (K-18): a slab is a genuine `m > 1` GEMM, so the
    // register-blocked kernel reuses each decoded weight block across
    // `ONEBIT_GEMM_MR` rows inside the task instead of re-decoding the
    // whole matrix per token.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let batch_chunk = gemm_par_batch_chunk(m);
        output[..m * n_rows]
            .par_chunks_mut(batch_chunk * n_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| -> KernelResult<()> {
                let m0 = chunk_idx * batch_chunk;
                let rows = out_chunk.len() / n_rows;
                gemm_tiled(
                    dispatcher,
                    blocks,
                    &input[m0 * k..(m0 + rows) * k],
                    out_chunk,
                    rows,
                    n_rows,
                    k,
                )
            })?;

        Ok(())
    }
}

/// Batch rows per Rayon task for [`gemm_tiled_par`], floored at
/// [`GEMM_BATCH_TILE`] so every task still fills the register block.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn gemm_par_batch_chunk(m: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    m.div_ceil(threads).max(GEMM_BATCH_TILE).min(m).max(1)
}

/// Choose optimal tile size based on working set characteristics.
///
/// Returns the recommended tile row count for the given dimensions,
/// considering both L1 and L2 cache sizes.
///
/// The L1 budget is read from the platform-tuned
/// [`crate::tuning::PlatformProfile::global`] (e.g. 64 KB on AArch64/Apple
/// Silicon vs 32 KB on typical x86-64) rather than a hardcoded 32 KB
/// assumption, so the tile size actually reflects the detected hardware.
pub fn optimal_tile_rows(k: usize) -> usize {
    // Estimate bytes per row of weight data
    let blocks_per_row = k / QK1_0_G128;
    // Each block is 18 bytes (2 for f16 scale + 16 for qs)
    let bytes_per_row = blocks_per_row * 18;

    // Target: fit tile weight data in the platform's detected L1 cache,
    // minus the shared input vector: k * 4 bytes.
    let l1_bytes = crate::tuning::PlatformProfile::global().l1_cache_bytes;
    let l1_available = l1_bytes.saturating_sub(k * 4);
    let l1_rows = l1_available
        .checked_div(bytes_per_row)
        .unwrap_or(L1_TILE_ROWS);

    // Clamp to reasonable range
    l1_rows.clamp(4, L2_TILE_ROWS)
}

/// Estimate working set size in bytes for a tiled computation.
///
/// Useful for deciding between tiled and non-tiled paths.
pub fn estimate_tile_working_set(tile_rows: usize, k: usize) -> usize {
    let blocks_per_row = k / QK1_0_G128;
    let weight_bytes = tile_rows * blocks_per_row * 18;
    let input_bytes = k * 4;
    let output_bytes = tile_rows * 4;
    weight_bytes + input_bytes + output_bytes
}

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

    #[test]
    fn tiled_gemv_matches_direct_small() {
        let n_rows = 8;
        let k = 256;
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        let mut out_tiled = vec![0.0f32; n_rows];

        dispatcher
            .gemv(&blocks, &input, &mut out_direct, n_rows, k)
            .expect("direct gemv should succeed");
        gemv_tiled(&dispatcher, &blocks, &input, &mut out_tiled, n_rows, k)
            .expect("tiled gemv should succeed");

        for i in 0..n_rows {
            assert!(
                (out_direct[i] - out_tiled[i]).abs() < 1e-4,
                "row {i}: direct={}, tiled={}",
                out_direct[i],
                out_tiled[i]
            );
        }
    }

    #[test]
    fn tiled_gemv_matches_direct_large() {
        // Larger than one L1 tile
        let n_rows = 100;
        let k = 256;
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        let mut out_tiled = vec![0.0f32; n_rows];

        dispatcher
            .gemv(&blocks, &input, &mut out_direct, n_rows, k)
            .expect("direct gemv should succeed");
        gemv_tiled(&dispatcher, &blocks, &input, &mut out_tiled, n_rows, k)
            .expect("tiled gemv should succeed");

        for i in 0..n_rows {
            assert!(
                (out_direct[i] - out_tiled[i]).abs() < 1e-4,
                "row {i}: direct={}, tiled={}",
                out_direct[i],
                out_tiled[i]
            );
        }
    }

    #[test]
    fn tiled_gemv_par_matches_direct() {
        let n_rows = 128;
        let k = 256;
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; n_rows];
        let mut out_tiled = vec![0.0f32; n_rows];

        dispatcher
            .gemv(&blocks, &input, &mut out_direct, n_rows, k)
            .expect("direct gemv should succeed");
        gemv_tiled_par(&dispatcher, &blocks, &input, &mut out_tiled, n_rows, k)
            .expect("par tiled gemv should succeed");

        for i in 0..n_rows {
            assert!(
                (out_direct[i] - out_tiled[i]).abs() < 1e-4,
                "row {i}: direct={}, tiled_par={}",
                out_direct[i],
                out_tiled[i]
            );
        }
    }

    #[test]
    fn tiled_gemm_matches_direct() {
        let m = 4;
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

        let mut out_direct = vec![0.0f32; m * n_rows];
        let mut out_tiled = vec![0.0f32; m * n_rows];

        dispatcher
            .gemm(&blocks, &input, &mut out_direct, m, n_rows, k)
            .expect("direct gemm should succeed");
        gemm_tiled(&dispatcher, &blocks, &input, &mut out_tiled, m, n_rows, k)
            .expect("tiled gemm should succeed");

        for i in 0..(m * n_rows) {
            assert!(
                (out_direct[i] - out_tiled[i]).abs() < 1e-3,
                "idx {i}: direct={}, tiled={}",
                out_direct[i],
                out_tiled[i]
            );
        }
    }

    #[test]
    fn tiled_gemm_large_matches_direct() {
        // More than one L1 tile of rows
        let m = 2;
        let n_rows = 64;
        let k = 256;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::new();
        for ni in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [((ni * 23 + bi * 11) & 0xFF) as u8; 16];
                blocks.push(make_block(0.3 + ni as f32 * 0.05, bits));
            }
        }
        let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.003) - 0.5).collect();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut out_direct = vec![0.0f32; m * n_rows];
        let mut out_tiled = vec![0.0f32; m * n_rows];

        dispatcher
            .gemm(&blocks, &input, &mut out_direct, m, n_rows, k)
            .expect("direct gemm should succeed");
        gemm_tiled(&dispatcher, &blocks, &input, &mut out_tiled, m, n_rows, k)
            .expect("tiled gemm should succeed");

        for i in 0..(m * n_rows) {
            assert!(
                (out_direct[i] - out_tiled[i]).abs() < 1e-3,
                "idx {i}: direct={}, tiled={}",
                out_direct[i],
                out_tiled[i]
            );
        }
    }

    #[test]
    fn tiled_gemm_par_matches_direct() {
        let m = 8;
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

        let mut out_direct = vec![0.0f32; m * n_rows];
        let mut out_tiled = vec![0.0f32; m * n_rows];

        dispatcher
            .gemm(&blocks, &input, &mut out_direct, m, n_rows, k)
            .expect("direct gemm should succeed");
        gemm_tiled_par(&dispatcher, &blocks, &input, &mut out_tiled, m, n_rows, k)
            .expect("par tiled gemm should succeed");

        for i in 0..(m * n_rows) {
            assert!(
                (out_direct[i] - out_tiled[i]).abs() < 1e-3,
                "idx {i}: direct={}, tiled_par={}",
                out_direct[i],
                out_tiled[i]
            );
        }
    }

    #[test]
    fn tiled_gemv_validation_errors() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];

        // Not block aligned
        let result = gemv_tiled(&dispatcher, &blocks, &input, &mut output, 1, 100);
        assert!(result.is_err());

        // Input too small
        let short_input = vec![1.0f32; 64];
        let result = gemv_tiled(&dispatcher, &blocks, &short_input, &mut output, 1, 128);
        assert!(result.is_err());
    }

    #[test]
    fn tiled_gemm_validation_errors() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];

        // Not block aligned
        let result = gemm_tiled(&dispatcher, &blocks, &input, &mut output, 1, 1, 100);
        assert!(result.is_err());
    }

    #[test]
    fn optimal_tile_rows_reasonable() {
        // For k=128, blocks_per_row=1, 18 bytes/row
        let rows = optimal_tile_rows(128);
        assert!(rows >= 4);
        assert!(rows <= L2_TILE_ROWS);

        // For k=4096, blocks_per_row=32, 576 bytes/row
        let rows_large = optimal_tile_rows(4096);
        assert!(rows_large >= 4);
        assert!(rows_large <= L2_TILE_ROWS);
        // Larger k should yield fewer tile rows
        assert!(rows_large <= rows);
    }

    #[test]
    fn estimate_tile_working_set_correct() {
        let ws = estimate_tile_working_set(32, 128);
        // 32 rows * 1 block * 18 bytes + 128 * 4 (input) + 32 * 4 (output)
        let expected = 32 * 18 + 128 * 4 + 32 * 4;
        assert_eq!(ws, expected);
    }

    // ── The once-at-entry INT8 tier check (K-14) ─────────────────────────

    fn to_bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// With the tier unset, both tiled entry points must still be
    /// bit-identical to `OneBitKernel::gemv` — not merely close — because
    /// they now call `gemv_1bit_on_tier` directly instead of re-entering
    /// `gemv`, and that is the exact same per-tier body `gemv` itself falls
    /// back to once its own entry check finds nothing selected.
    #[test]
    fn tiled_gemv_and_par_are_bit_identical_to_direct_gemv_with_the_tier_unset() {
        let _guard = crate::dispatch_int8::TierEnvGuard::acquire();
        let (n_rows, k) = (300usize, 256usize);
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::with_tier(crate::dispatch::KernelTier::Reference);

        let mut out_direct = vec![0.0f32; n_rows];
        dispatcher
            .gemv(&blocks, &input, &mut out_direct, n_rows, k)
            .expect("direct gemv");

        let mut out_tiled = vec![0.0f32; n_rows];
        gemv_tiled(&dispatcher, &blocks, &input, &mut out_tiled, n_rows, k).expect("tiled gemv");
        assert_eq!(
            to_bits(&out_direct),
            to_bits(&out_tiled),
            "gemv_tiled must be bit-identical to OneBitKernel::gemv with the tier unset"
        );

        let mut out_tiled_par = vec![0.0f32; n_rows];
        gemv_tiled_par(&dispatcher, &blocks, &input, &mut out_tiled_par, n_rows, k)
            .expect("tiled par gemv");
        assert_eq!(
            to_bits(&out_direct),
            to_bits(&out_tiled_par),
            "gemv_tiled_par must be bit-identical to OneBitKernel::gemv with the tier unset"
        );
    }

    /// With the tier set, both tiled entry points must be bit-identical to
    /// the **un-tiled** INT8 GEMV: the whole call is now handed to
    /// `dispatch_int8::gemv_1bit_g128_int8` once, ahead of any tiling,
    /// rather than every L1/L2 tile re-quantizing the identical, untiled
    /// activation on its own.
    #[test]
    fn tiled_gemv_and_par_are_bit_identical_to_the_untiled_int8_gemv_with_the_tier_set() {
        let _guard = crate::dispatch_int8::TierEnvGuard::acquire();
        // SAFETY: serialized by `_guard` above.
        unsafe {
            std::env::set_var(
                crate::dispatch_int8::KERNEL_TIER_ENV,
                crate::dispatch_int8::Int8Tier::Scalar.name(),
            );
        }

        let (n_rows, k) = (300usize, 256usize);
        let (blocks, input) = make_test_data(n_rows, k);
        let dispatcher = KernelDispatcher::with_tier(crate::dispatch::KernelTier::Reference);
        assert!(
            dispatcher.native_int8_tier().is_some(),
            "the guard must have actually selected a tier"
        );

        let mut expect = vec![0.0f32; n_rows];
        crate::dispatch_int8::gemv_1bit_g128_int8(
            crate::dispatch_int8::Int8Tier::Scalar,
            &blocks,
            &input,
            &mut expect,
            n_rows,
            k,
        )
        .expect("direct int8 gemv");

        let mut out_tiled = vec![0.0f32; n_rows];
        gemv_tiled(&dispatcher, &blocks, &input, &mut out_tiled, n_rows, k).expect("tiled gemv");
        assert_eq!(
            to_bits(&expect),
            to_bits(&out_tiled),
            "gemv_tiled must be bit-identical to the un-tiled INT8 gemv with the tier set"
        );

        let mut out_tiled_par = vec![0.0f32; n_rows];
        gemv_tiled_par(&dispatcher, &blocks, &input, &mut out_tiled_par, n_rows, k)
            .expect("tiled par gemv");
        assert_eq!(
            to_bits(&expect),
            to_bits(&out_tiled_par),
            "gemv_tiled_par must be bit-identical to the un-tiled INT8 gemv with the tier set"
        );
    }
}
