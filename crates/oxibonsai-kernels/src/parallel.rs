//! Multi-threaded kernel wrappers using Rayon.
//!
//! Provides parallel versions of the 1-bit GEMV and GEMM kernels that
//! split work across CPU cores. Row-parallel GEMV and batch-parallel GEMM.
//!
//! On WASM targets (`wasm32`), rayon is unavailable (no threads).
//! All parallel entry points fall back to sequential execution transparently.
//!
//! ## The opt-in INT8 tier (K-14)
//!
//! The four native-format drivers (`gemv_1bit_g128_par`,
//! `gemm_1bit_g128_par`, `gemv_ternary_g128_par`, `gemm_ternary_g128_par`)
//! ask [`KernelDispatcher::native_int8_tier`] once, at entry, and hand the
//! whole call to the matching [`crate::dispatch_int8`] kernel when
//! `OXIBONSAI_KERNEL_TIER` selects one — the model's batched CPU prefill
//! calls the two GEMM drivers directly, so this is where the tier reaches
//! it. Otherwise the call runs today's f32 body, whose chunks call the
//! dispatcher's `*_on_tier` methods and so never re-read the environment.

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::dispatch::KernelDispatcher;
use crate::error::{KernelError, KernelResult};
use crate::gemm_onebit::gemm_1bit_g128_blocked;
#[cfg(not(target_arch = "wasm32"))]
use crate::gemm_onebit::ONEBIT_GEMM_MR;
use crate::gemm_ternary::gemm_tq2_0_g128_blocked;
#[cfg(not(target_arch = "wasm32"))]
use crate::gemm_ternary::TERNARY_GEMM_MR;
use crate::traits::Fp8Kernel;
use crate::traits::OneBitKernel;
use crate::traits::StandardQuantKernel;
// The ternary drivers below call the dispatcher's `*_on_tier` methods, not
// the trait; the sibling `parallel_tests.rs` (`use super::*`) still calls
// `TernaryKernel` methods directly.
#[cfg(test)]
use crate::traits::TernaryKernel;
use oxibonsai_core::QK_TQ2_0_G128;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};
use oxibonsai_core::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};

use crate::tuning::PlatformProfile;

/// Minimum number of output rows before engaging parallel GEMV.
///
/// Sourced from the platform-tuned [`crate::tuning::TunedThresholds`]
/// (auto-detected once per process from core count, cache sizes and SIMD tier)
/// rather than a fixed constant, so a low-core laptop and a many-core server
/// each pick their own break-even point. Below the threshold, thread-spawn
/// overhead exceeds the per-row compute parallelism saves.
#[inline]
fn par_gemv_min_rows() -> usize {
    PlatformProfile::global_thresholds().par_gemv_min_rows
}

/// Minimum batch size before engaging parallel GEMM (platform-tuned; see
/// [`par_gemv_min_rows`]).
#[inline]
fn par_gemm_min_batch() -> usize {
    PlatformProfile::global_thresholds().par_gemm_min_batch
}

/// Rows per Rayon work-item for the row-parallel GEMV drivers (K-16).
///
/// One Rayon task per **output row** (`par_chunks_mut(1)`) means every task
/// pays the full validated-dispatcher call overhead (the dimension checks
/// plus a `match self.tier`) for a single row's worth of compute; at
/// `n_rows = 248_320` (the Bonsai 2 LM head) that is a quarter of a million
/// redundant validations per token, measured at 1.44x speedup on 8 cores
/// (18% efficiency — `parallel_tiled.rs`'s `gemv_parallel_tiled` already
/// avoids this with `par_chunks_mut(L2_TILE_ROWS)`).
///
/// Targets roughly 4 tasks per worker thread — enough tasks that Rayon's
/// work-stealing still balances an uneven row cost across cores, but few
/// enough that the per-task overhead stops mattering — clamped to `[8,
/// 512]` so a small matrix still gets genuine parallelism (never a
/// zero-sized or 1-row task) and a huge one never creates an oversized
/// single task that serializes the tail of the matrix.
///
/// The `512` upper clamp is also load-bearing for correctness, not just
/// tuning: [`crate::dispatch::KernelDispatcher`]'s 1-bit/ternary
/// `KernelTier::Gpu` arms only attempt GPU acceleration once a call's row
/// count reaches `GPU_MIN_ROWS` (1024), so every chunk produced here stays
/// under that threshold and takes the exact same CPU fallback branch a
/// `par_chunks_mut(1)` row always took — the chunking change can only
/// affect how many rows are validated and computed per Rayon task, never
/// which tier computes them, which is what keeps output byte-identical to
/// the pre-fix code.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn rows_per_task(n_rows: usize) -> usize {
    let num_threads = rayon::current_num_threads().max(1);
    (n_rows / (num_threads * 4)).clamp(8, 512)
}

/// Parallel row-wise 1-bit GEMV.
///
/// Each row's dot product is independent, making this trivially parallelizable.
/// Falls back to sequential for small `n_rows` to avoid overhead. Runs on
/// the opt-in INT8 tier instead when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc).
pub fn gemv_1bit_g128_par(
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
    gemv_1bit_g128_par_on_tier(dispatcher, blocks, input, output, n_rows, k)
}

/// [`gemv_1bit_g128_par`] on the dispatcher's own tier, without the INT8
/// entry check — for callers that already made it (`parallel_tiled`'s
/// adaptive driver).
pub(crate) fn gemv_1bit_g128_par_on_tier(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    // Validation
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

    // Sequential fallback for small row counts
    if n_rows < par_gemv_min_rows() {
        return dispatcher.gemv_1bit_on_tier(blocks, input, output, n_rows, k);
    }

    // On WASM: no rayon threads available — fall back to sequential.
    #[cfg(target_arch = "wasm32")]
    {
        dispatcher.gemv_1bit_on_tier(blocks, input, output, n_rows, k)
    }

    // Parallel: each chunk processes `chunk_rows` rows at once (K-16), not
    // one Rayon task per row.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = out_chunk.len();
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                let chunk_blocks = &blocks[block_start..block_end];
                dispatcher.gemv_1bit_on_tier(chunk_blocks, input, out_chunk, rows, k)
            })?;

        Ok(())
    }
}

/// Batch rows per Rayon work-item for the register-blocked GEMM drivers (K-18).
///
/// The GEMM drivers used to hand Rayon **one batch row per task** and call
/// the tier kernel with `m = 1`, which is the loop-of-GEMVs shape K-18
/// names: every task re-streamed and re-decoded the entire quantized weight
/// matrix for a single token. A slab of batch rows per task lets the
/// register-blocked kernel keep a decoded weight block live across `MR`
/// rows inside the task, so the weight matrix is decoded
/// `ceil(m / MR)` times in total instead of `m` times.
///
/// Targets one slab per worker thread — the slab's input rows
/// (`slab * k * 4` bytes) then stay L2-resident while the whole weight
/// matrix streams past once — floored at the kernel's own register-block
/// factor `mr`, because a slab shorter than `MR` cannot fill the register
/// block and would throw the reuse away.
///
/// WASM has no Rayon worker pool, so the drivers there stay sequential and
/// never call this.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn gemm_batch_chunk_rows(m: usize, mr: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    m.div_ceil(threads).max(mr).min(m).max(1)
}

/// Parallel batch-wise 1-bit GEMM (K-18 register-blocked).
///
/// Batch rows are independent, so the batch dimension is what Rayon splits —
/// but in **slabs** (`gemm_batch_chunk_rows`), not one row per task, so
/// each task runs a genuine `m > 1` register-blocked GEMM and a decoded
/// weight block is reused across `ONEBIT_GEMM_MR` rows.
///
/// This is a **CPU** driver: [`gemm_1bit_g128_blocked`] reproduces the
/// dispatcher's CPU tier bit for bit, and maps a GPU-tier dispatcher onto
/// its best CPU tier — the same fallback `KernelDispatcher::gemm` itself
/// takes below its GPU row threshold. Callers that want the Metal/CUDA 1-bit
/// GEMM call `KernelDispatcher::gemm` (or the model's fused GPU prefill)
/// directly; nothing in the tree reached the GPU *through* this function.
///
/// Runs on the opt-in INT8 tier instead when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc);
/// the model's batched CPU prefill reaches the tier through here.
pub fn gemm_1bit_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return crate::dispatch_int8::gemm_1bit_g128_int8(
            tier, blocks, input, output, m, n_rows, k,
        );
    }

    // Validation
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

    // Sequential fallback for small batches — still register-blocked (K-18),
    // which is bit-identical to `dispatcher.gemm` but decodes each weight
    // block once per `ONEBIT_GEMM_MR` batch rows.
    if m < par_gemm_min_batch() {
        return gemm_1bit_g128_blocked(dispatcher, blocks, input, output, m, n_rows, k);
    }

    // On WASM: no rayon threads available — fall back to sequential.
    #[cfg(target_arch = "wasm32")]
    {
        gemm_1bit_g128_blocked(dispatcher, blocks, input, output, m, n_rows, k)
    }

    // Parallel over slabs of batch rows: each task runs a genuine
    // `m > 1` GEMM over the whole weight matrix, so the register blocking
    // inside the kernel survives the parallel split (K-18).
    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = gemm_batch_chunk_rows(m, ONEBIT_GEMM_MR);
        output[..m * n_rows]
            .par_chunks_mut(chunk_rows * n_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let m0 = chunk_idx * chunk_rows;
                let rows = out_chunk.len() / n_rows;
                let input_chunk = &input[m0 * k..(m0 + rows) * k];
                gemm_1bit_g128_blocked(dispatcher, blocks, input_chunk, out_chunk, rows, n_rows, k)
            })?;

        Ok(())
    }
}

/// Parallel row-wise ternary GEMV.
///
/// Each row's dot product is independent, making this trivially parallelizable.
/// Falls back to sequential for small `n_rows` to avoid overhead. Runs on
/// the opt-in INT8 tier instead when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc).
pub fn gemv_ternary_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return crate::dispatch_int8::gemv_two_bit_int8(tier, blocks, input, output, n_rows, k);
    }
    gemv_ternary_g128_par_on_tier(dispatcher, blocks, input, output, n_rows, k)
}

/// [`gemv_ternary_g128_par`] on the dispatcher's own tier, without the INT8
/// entry check — for callers that already made it (`parallel_tiled`'s
/// adaptive driver).
pub(crate) fn gemv_ternary_g128_par_on_tier(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
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

    if n_rows < par_gemv_min_rows() {
        return dispatcher.gemv_ternary_on_tier(blocks, input, output, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        dispatcher.gemv_ternary_on_tier(blocks, input, output, n_rows, k)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = out_chunk.len();
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                let chunk_blocks = &blocks[block_start..block_end];
                dispatcher.gemv_ternary_on_tier(chunk_blocks, input, out_chunk, rows, k)
            })?;

        Ok(())
    }
}

/// Parallel batch-wise ternary GEMM (K-18 register-blocked).
///
/// Batch rows are independent, so the batch dimension is what Rayon splits —
/// but in **slabs** (`gemm_batch_chunk_rows`), not one row per task, so
/// each task runs a genuine `m > 1` register-blocked GEMM and a decoded
/// weight block is reused across `TERNARY_GEMM_MR` rows. There is no ternary
/// GPU GEMM kernel at all, so a GPU-tier dispatcher ran this on the CPU
/// before this change too.
///
/// Runs on the opt-in INT8 tier instead when
/// [`KernelDispatcher::native_int8_tier`] selects one (see the module doc);
/// the model's batched CPU prefill reaches the tier through here.
pub fn gemm_ternary_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if let Some(tier) = dispatcher.native_int8_tier() {
        return crate::dispatch_int8::gemm_two_bit_int8(tier, blocks, input, output, m, n_rows, k);
    }
    gemm_ternary_g128_par_on_tier(dispatcher, blocks, input, output, m, n_rows, k)
}

/// [`gemm_ternary_g128_par`] on the dispatcher's own tier, without the INT8
/// entry check — for callers that already made it (`parallel_tiled`'s
/// adaptive driver).
pub(crate) fn gemm_ternary_g128_par_on_tier(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    // Sequential fallback for small batches — still register-blocked (K-18),
    // which is bit-identical to `dispatcher.gemm_ternary_g128` but decodes
    // each weight block once per `TERNARY_GEMM_MR` batch rows.
    if m < par_gemm_min_batch() {
        return gemm_tq2_0_g128_blocked(dispatcher, blocks, input, output, m, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        gemm_tq2_0_g128_blocked(dispatcher, blocks, input, output, m, n_rows, k)
    }

    // Parallel over slabs of batch rows rather than one task per batch row:
    // each task runs a genuine `m > 1` GEMM, so the decoded weight block is
    // reused across `TERNARY_GEMM_MR` rows inside the task (K-18).
    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = gemm_batch_chunk_rows(m, TERNARY_GEMM_MR);
        output[..m * n_rows]
            .par_chunks_mut(chunk_rows * n_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let m0 = chunk_idx * chunk_rows;
                let rows = out_chunk.len() / n_rows;
                let input_chunk = &input[m0 * k..(m0 + rows) * k];
                gemm_tq2_0_g128_blocked(dispatcher, blocks, input_chunk, out_chunk, rows, n_rows, k)
            })?;

        Ok(())
    }
}

// ─── FP8 parallel entry points ────────────────────────────────────────────

/// Parallel row-wise FP8 E4M3FN GEMV.
///
/// Row-parallel split: each output row is an independent dot product.
/// Falls back to sequential for small `n_rows` to avoid thread-spawn overhead.
/// On WASM, always sequential.
pub fn gemv_fp8_e4m3_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockFP8E4M3],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_FP8) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
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
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    if n_rows < par_gemv_min_rows() {
        return dispatcher.gemv_fp8_e4m3(blocks, input, output, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        dispatcher.gemv_fp8_e4m3(blocks, input, output, n_rows, k)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = out_chunk.len();
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                let chunk_blocks = &blocks[block_start..block_end];
                dispatcher.gemv_fp8_e4m3(chunk_blocks, input, out_chunk, rows, k)
            })?;

        Ok(())
    }
}

/// Parallel row-wise FP8 E5M2 GEMV.
///
/// Falls back to sequential for small `n_rows`.  On WASM, always sequential.
pub fn gemv_fp8_e5m2_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockFP8E5M2],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_FP8) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
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
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    if n_rows < par_gemv_min_rows() {
        return dispatcher.gemv_fp8_e5m2(blocks, input, output, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        dispatcher.gemv_fp8_e5m2(blocks, input, output, n_rows, k)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = out_chunk.len();
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                let chunk_blocks = &blocks[block_start..block_end];
                dispatcher.gemv_fp8_e5m2(chunk_blocks, input, out_chunk, rows, k)
            })?;

        Ok(())
    }
}

/// Parallel batch-wise FP8 E4M3FN GEMM.
///
/// Each batch element is an independent GEMV call.
/// Falls back to sequential for small `batch`.  On WASM, always sequential.
pub fn gemm_fp8_e4m3_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockFP8E4M3],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_FP8) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if inputs.len() < batch * k {
        return Err(KernelError::dimension_mismatch(
            "inputs",
            batch * k,
            inputs.len(),
        ));
    }
    if outputs.len() < batch * n_rows {
        return Err(KernelError::buffer_too_small(
            "outputs",
            batch * n_rows,
            outputs.len(),
        ));
    }
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    if batch < par_gemm_min_batch() {
        return dispatcher.gemm_fp8_e4m3(blocks, inputs, outputs, n_rows, k, batch);
    }

    #[cfg(target_arch = "wasm32")]
    {
        dispatcher.gemm_fp8_e4m3(blocks, inputs, outputs, n_rows, k, batch)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        outputs[..batch * n_rows]
            .par_chunks_mut(n_rows)
            .enumerate()
            .try_for_each(|(bi, out_row)| {
                let input_row = &inputs[bi * k..(bi + 1) * k];
                dispatcher.gemm_fp8_e4m3(blocks, input_row, out_row, n_rows, k, 1)
            })?;

        Ok(())
    }
}

/// Parallel batch-wise FP8 E5M2 GEMM.
///
/// Falls back to sequential for small `batch`.  On WASM, always sequential.
pub fn gemm_fp8_e5m2_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockFP8E5M2],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_FP8) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if inputs.len() < batch * k {
        return Err(KernelError::dimension_mismatch(
            "inputs",
            batch * k,
            inputs.len(),
        ));
    }
    if outputs.len() < batch * n_rows {
        return Err(KernelError::buffer_too_small(
            "outputs",
            batch * n_rows,
            outputs.len(),
        ));
    }
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    if batch < par_gemm_min_batch() {
        return dispatcher.gemm_fp8_e5m2(blocks, inputs, outputs, n_rows, k, batch);
    }

    #[cfg(target_arch = "wasm32")]
    {
        dispatcher.gemm_fp8_e5m2(blocks, inputs, outputs, n_rows, k, batch)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        outputs[..batch * n_rows]
            .par_chunks_mut(n_rows)
            .enumerate()
            .try_for_each(|(bi, out_row)| {
                let input_row = &inputs[bi * k..(bi + 1) * k];
                dispatcher.gemm_fp8_e5m2(blocks, input_row, out_row, n_rows, k, 1)
            })?;

        Ok(())
    }
}

// ─── Standard GGUF quant (Q4_0 / Q8_0) parallel entry points ────────────────

/// Validate a standard-quant GEMV and return `blocks_per_row`.
///
/// Mirrors the error semantics of the scalar reference kernels so that routing
/// through the parallel/SIMD path does not change which error a caller sees.
///
/// **K-02 residue, now closed:** this validator is the last one in the file to
/// reach the **named** [`KernelError::dimension_mismatch`] /
/// [`KernelError::buffer_too_small`] constructors, because it backs
/// `gemv_q4_0.rs`'s and `gemv_q8_0.rs`'s **public** `gemv_q4_0`/`gemv_q8_0`
/// entry points (both are thin wrappers around `gemv_q4_0_par`/`gemv_q8_0_par`,
/// which call this function), and those files' four length-contract tests used
/// to `matches!` on the exact unnamed variant shape. Those assertions were
/// first rewritten onto [`KernelError::error_code`] — the migration-proof form
/// documented in `error.rs`, shared by the named and unnamed variants of one
/// condition — and only then were the constructors here migrated, so no test
/// was ever weakened or left red. `blocks` / `input` / `output` are the same
/// operand names this file's other validators use, and
/// `parallel_tests::migrated_errors_name_the_offending_buffer` pins them.
fn validate_std_gemv(
    n_blocks: usize,
    input_len: usize,
    output_len: usize,
    n_rows: usize,
    in_features: usize,
    block_len: usize,
) -> KernelResult<usize> {
    if !in_features.is_multiple_of(block_len) {
        return Err(KernelError::NotBlockAligned {
            count: in_features,
            block_size: block_len,
        });
    }
    let blocks_per_row = in_features / block_len;
    let expected_blocks = n_rows * blocks_per_row;
    if n_blocks < expected_blocks {
        return Err(KernelError::dimension_mismatch(
            "blocks",
            expected_blocks,
            n_blocks,
        ));
    }
    if input_len < in_features {
        return Err(KernelError::dimension_mismatch(
            "input",
            in_features,
            input_len,
        ));
    }
    if output_len < n_rows {
        return Err(KernelError::buffer_too_small("output", n_rows, output_len));
    }
    Ok(blocks_per_row)
}

/// Parallel row-wise Q4_0 GEMV.
///
/// Each output row is an independent dot product. Below the platform-tuned
/// `par_gemv_min_rows` threshold (and on WASM) it falls back to a single
/// tier-dispatched [`StandardQuantKernel::gemv_q4_0`] call; above it, rows are
/// split across Rayon threads with each row processed by the best SIMD tier.
pub fn gemv_q4_0_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ4_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_std_gemv(
        blocks.len(),
        input.len(),
        output.len(),
        n_rows,
        in_features,
        QK_Q4_0,
    )?;

    if n_rows < par_gemv_min_rows() {
        return dispatcher.gemv_q4_0(blocks, input, output, n_rows, in_features);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = blocks_per_row;
        dispatcher.gemv_q4_0(blocks, input, output, n_rows, in_features)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = out_chunk.len();
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                let chunk_blocks = &blocks[block_start..block_end];
                dispatcher.gemv_q4_0(chunk_blocks, input, out_chunk, rows, in_features)
            })?;
        Ok(())
    }
}

/// Parallel row-wise Q8_0 GEMV. See [`gemv_q4_0_par`] for the strategy.
pub fn gemv_q8_0_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ8_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_std_gemv(
        blocks.len(),
        input.len(),
        output.len(),
        n_rows,
        in_features,
        QK_Q8_0,
    )?;

    if n_rows < par_gemv_min_rows() {
        return dispatcher.gemv_q8_0(blocks, input, output, n_rows, in_features);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = blocks_per_row;
        dispatcher.gemv_q8_0(blocks, input, output, n_rows, in_features)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_rows = rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = out_chunk.len();
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                let chunk_blocks = &blocks[block_start..block_end];
                dispatcher.gemv_q8_0(chunk_blocks, input, out_chunk, rows, in_features)
            })?;
        Ok(())
    }
}

// ─── K-quant (Q2_K..Q8_K) row-parallel driver ──────────────────────────────

/// Dot product of two equal-length f32 slices (matches the scalar K-quant loop).
///
/// Plain `a.iter().zip(b).map(|(x, y)| x * y).sum()` is pure scalar: f32
/// addition is not reassociative, so LLVM cannot auto-vectorize a linear
/// `.sum()` fold, and every K-quant GEMV routes through this function once
/// per output row (K-15). Reduced with four (NEON) or two (AVX2+FMA) 128/256
/// -bit accumulator lanes, horizontally summed once at the end, with a
/// scalar tail for the remainder — the standard "wide accumulator" dot
/// product. This **changes the summation order** relative to the plain
/// sequential fold (lanes accumulate independently, then combine), so
/// results differ from the pre-SIMD scalar path at the ~1e-6 relative
/// (f32 rounding) level; callers that need bit-for-bit output stability
/// must not rely on `dot_f32`'s exact reduction order — see the K-quant
/// parity tests, which use a 1e-4 relative tolerance for this reason (the
/// same tolerance the package's ACCEPTANCE criterion documents).
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is the AArch64 ISA baseline (always available), matching the
        // convention used throughout this crate's other NEON kernels.
        return unsafe { dot_f32_neon(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    #[cfg_attr(target_arch = "aarch64", allow(unreachable_code))]
    dot_f32_scalar(a, b)
}

/// Pure-scalar dot product: the tail path shared by every ISA, and the only
/// path on platforms with neither NEON nor AVX2+FMA (e.g. wasm32, or x86-64
/// without AVX2 at runtime).
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// NEON dot product: 4 independent `vfmaq_f32` accumulator lanes (16 f32
/// elements per loop iteration), horizontally summed with `vaddvq_f32`, with
/// a scalar tail for `a.len() % 16 != 0` (`dot_f32_scalar`-equivalent, one
/// element at a time so the tail never re-enters a SIMD path for < 4
/// elements).
///
/// # Safety
/// Requires NEON CPU support (always available on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let n = a.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut i = 0usize;
    while i + 16 <= n {
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(a.as_ptr().add(i)),
            vld1q_f32(b.as_ptr().add(i)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(a.as_ptr().add(i + 4)),
            vld1q_f32(b.as_ptr().add(i + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(a.as_ptr().add(i + 8)),
            vld1q_f32(b.as_ptr().add(i + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(a.as_ptr().add(i + 12)),
            vld1q_f32(b.as_ptr().add(i + 12)),
        );
        i += 16;
    }
    while i + 4 <= n {
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(a.as_ptr().add(i)),
            vld1q_f32(b.as_ptr().add(i)),
        );
        i += 4;
    }

    let mut sum = vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// AVX2+FMA dot product: 2 independent `_mm256_fmadd_ps` accumulator lanes
/// (16 f32 elements per loop iteration), horizontally summed, with 8-wide
/// and scalar tails.
///
/// Compile-checked (`cargo check --target x86_64-apple-darwin`) but not
/// runtime-validated on this workspace's development hardware (Apple
/// Silicon); mirrors the already-shipping AVX2 kernels in
/// `crate::simd_avx2` in structure and intrinsic usage.
///
/// # Safety
/// Caller must have already confirmed `is_x86_feature_detected!("avx2")` and
/// `is_x86_feature_detected!("fma")` (see [`dot_f32`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps,
        _mm_movehl_ps,
    };

    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 16 <= n {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            acc1,
        );
        i += 16;
    }
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum4 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(final_sum);

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

thread_local! {
    /// Reusable per-thread scratch buffer for the K-quant row-parallel
    /// drivers (K-15). Grows to the largest `in_features` seen on this
    /// thread and is never shrunk, so the steady-state cost per GEMV call
    /// is zero heap allocations — both the sequential fallback (previously
    /// `vec![0.0f32; in_features]` on **every** call) and the Rayon-parallel
    /// path (previously one `Vec` per work-split via `try_for_each_init`)
    /// now share this one growable buffer per OS thread.
    static KQUANT_ROW_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Borrow this thread's reusable K-quant scratch row, growing it to
/// `in_features` first if it is not already at least that long.
///
/// Invariant: `f` must not call back into `with_kquant_scratch` on the same
/// thread — the `RefCell` borrow is held for `f`'s whole duration, so a
/// re-entrant call would panic on the second borrow rather than silently
/// corrupting the buffer. Currently unreachable: every caller's `f` is a
/// `BlockQ*K::dequant` call, which touches only its own arguments.
#[inline]
fn with_kquant_scratch<R>(in_features: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    KQUANT_ROW_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < in_features {
            buf.resize(in_features, 0.0);
        }
        f(&mut buf[..in_features])
    })
}

/// Row-parallel scalar GEMV driver shared by the K-quant formats (Q2_K..Q8_K).
///
/// `dequant_row(row, row_buf)` must fill `row_buf[..in_features]` with the
/// dequantized weights of output row `row`; the driver then dots that against
/// `input[..in_features]`. Results are numerically **identical** to a plain
/// sequential loop modulo `dot_f32`'s SIMD reduction order (K-15) — every
/// output row is an independent reduction, so only the row iteration is
/// distributed across Rayon threads (above the platform-tuned
/// [`par_gemv_min_rows`] threshold). Both the parallel and sequential paths
/// share one reusable per-thread scratch row ([`with_kquant_scratch`]), so
/// steady-state this driver performs zero heap allocations per call.
///
/// The caller must validate dimensions (so `input.len() >= in_features` and
/// `output.len() >= n_rows`) before invoking this driver.
pub(crate) fn gemv_kquant_row_parallel<F>(
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
    dequant_row: F,
) -> KernelResult<()>
where
    F: Fn(usize, &mut [f32]) -> KernelResult<()> + Sync,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        if n_rows >= par_gemv_min_rows() {
            return output[..n_rows].par_iter_mut().enumerate().try_for_each(
                |(row, out)| -> KernelResult<()> {
                    with_kquant_scratch(in_features, |row_buf| {
                        dequant_row(row, row_buf)?;
                        *out = dot_f32(row_buf, &input[..in_features]);
                        Ok(())
                    })
                },
            );
        }
    }

    // Sequential fallback (also the WASM path): the reused per-thread
    // scratch row, same as the parallel path above.
    for (row, out) in output[..n_rows].iter_mut().enumerate() {
        with_kquant_scratch(in_features, |row_buf| -> KernelResult<()> {
            dequant_row(row, row_buf)?;
            *out = dot_f32(row_buf, &input[..in_features]);
            Ok(())
        })?;
    }
    Ok(())
}

/// Row-parallel **fused** GEMV driver: like [`gemv_kquant_row_parallel`], but
/// for kernels that compute a row's dot product directly from the quantized
/// bytes without ever materializing a dequantized f32 row (K-15 item (c)).
///
/// `row_dot(row, input)` must return `dot(weight_row, input[..in_features])`
/// for output row `row`. Because there is no intermediate row buffer at all,
/// this driver — unlike [`gemv_kquant_row_parallel`] — needs no scratch
/// buffer of any kind, parallel or sequential: it is zero-allocation by
/// construction, not just in the steady state.
///
/// Gated to AArch64 because that is the only tier with a fused K-quant
/// kernel to call it (Q4_K/Q6_K/Q8_K's `neon_fused::row_dot`, K-15 item
/// (c)); on every other target `gemv_q4k`/`gemv_q6k`/`gemv_q8k` use the
/// generic [`gemv_kquant_row_parallel`] instead, which would otherwise
/// leave this function uncalled (`dead_code`) there.
#[cfg(target_arch = "aarch64")]
pub(crate) fn gemv_kquant_row_parallel_fused<F>(
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    in_features: usize,
    row_dot: F,
) -> KernelResult<()>
where
    F: Fn(usize, &[f32]) -> KernelResult<f32> + Sync,
{
    let input = &input[..in_features];

    #[cfg(not(target_arch = "wasm32"))]
    {
        if n_rows >= par_gemv_min_rows() {
            return output[..n_rows].par_iter_mut().enumerate().try_for_each(
                |(row, out)| -> KernelResult<()> {
                    *out = row_dot(row, input)?;
                    Ok(())
                },
            );
        }
    }

    for (row, out) in output[..n_rows].iter_mut().enumerate() {
        *out = row_dot(row, input)?;
    }
    Ok(())
}

// ─── Layer-parallel utilities ──────────────────────────────────────────

/// Configuration for layer-parallel forward passes.
///
/// Controls how transformer layers are distributed across threads
/// and the depth of the execution pipeline.
#[derive(Debug, Clone)]
pub struct LayerParallelConfig {
    /// Maximum number of transformer layers to process in parallel.
    /// Limited by available memory for intermediate activations.
    pub max_parallel_layers: usize,
    /// Pipeline depth: how many stages of computation overlap.
    /// 1 = no pipelining, 2 = double-buffered, etc.
    pub pipeline_depth: usize,
}

impl Default for LayerParallelConfig {
    fn default() -> Self {
        Self {
            max_parallel_layers: 1,
            pipeline_depth: 1,
        }
    }
}

impl LayerParallelConfig {
    /// Create a config for the given model and hardware.
    pub fn for_model(num_layers: usize, num_threads: usize) -> Self {
        // Conservative: at most half the threads for layer parallelism
        let max_par = (num_threads / 2).max(1).min(num_layers);
        Self {
            max_parallel_layers: max_par,
            pipeline_depth: if max_par > 1 { 2 } else { 1 },
        }
    }
}

/// Pipeline stage for inference pipeline parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Processing the input prompt (compute-heavy, batched).
    Prefill,
    /// Auto-regressive token generation (memory-bound, single token).
    Decode,
    /// Post-processing: detokenization, sampling, etc.
    PostProcess,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prefill => write!(f, "prefill"),
            Self::Decode => write!(f, "decode"),
            Self::PostProcess => write!(f, "post_process"),
        }
    }
}

/// Statistics about parallel execution, accumulated over time.
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    /// Total number of output rows processed.
    pub total_rows_processed: usize,
    /// Number of times parallel execution was used.
    pub parallel_invocations: usize,
    /// Number of times we fell back to sequential execution.
    pub sequential_fallbacks: usize,
    /// Running average tile size (rows per tile).
    pub average_tile_size: f64,
    /// Total number of GEMV calls dispatched.
    pub total_gemv_calls: usize,
    /// Total number of GEMM calls dispatched.
    pub total_gemm_calls: usize,
}

impl ParallelStats {
    /// Record a parallel invocation.
    pub fn record_parallel(&mut self, rows: usize, tile_size: usize) {
        self.total_rows_processed += rows;
        self.parallel_invocations += 1;
        // Incremental average
        let n = self.parallel_invocations as f64;
        self.average_tile_size = self.average_tile_size * ((n - 1.0) / n) + (tile_size as f64 / n);
    }

    /// Record a sequential fallback.
    pub fn record_sequential(&mut self, rows: usize) {
        self.total_rows_processed += rows;
        self.sequential_fallbacks += 1;
    }

    /// Record a GEMV call.
    pub fn record_gemv(&mut self) {
        self.total_gemv_calls += 1;
    }

    /// Record a GEMM call.
    pub fn record_gemm(&mut self) {
        self.total_gemm_calls += 1;
    }

    /// Fraction of invocations that used parallelism (0.0..=1.0).
    pub fn parallel_fraction(&self) -> f64 {
        let total = self.parallel_invocations + self.sequential_fallbacks;
        if total == 0 {
            return 0.0;
        }
        self.parallel_invocations as f64 / total as f64
    }
}

/// Parallel dequantize: unpack many 1-bit blocks in parallel.
///
/// Each block produces `QK1_0_G128` (128) f32 values. The blocks are
/// split across Rayon threads, with each thread dequantizing a contiguous
/// chunk.
pub fn dequant_1bit_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    output: &mut [f32],
) -> KernelResult<()> {
    let elements_per_block = QK1_0_G128;
    let total_elements = blocks.len() * elements_per_block;

    if output.len() < total_elements {
        return Err(KernelError::buffer_too_small(
            "output",
            total_elements,
            output.len(),
        ));
    }

    // For small block counts, sequential is faster
    if blocks.len() < 64 {
        return dispatcher.dequant(blocks, output);
    }

    // On WASM: no rayon threads available — fall back to sequential.
    #[cfg(target_arch = "wasm32")]
    {
        dispatcher.dequant(blocks, output)
    }

    // Parallel: each chunk is a contiguous set of blocks
    #[cfg(not(target_arch = "wasm32"))]
    {
        let chunk_size = 32; // blocks per chunk
        output[..total_elements]
            .par_chunks_mut(chunk_size * elements_per_block)
            .enumerate()
            .try_for_each(|(ci, out_chunk)| {
                let block_start = ci * chunk_size;
                let block_end = (block_start + chunk_size).min(blocks.len());
                let chunk_blocks = &blocks[block_start..block_end];
                dispatcher.dequant(chunk_blocks, out_chunk)
            })?;

        Ok(())
    }
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod tests;
