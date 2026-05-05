//! Multi-threaded kernel wrappers using Rayon.
//!
//! Provides parallel versions of the 1-bit GEMV and GEMM kernels that
//! split work across CPU cores. Row-parallel GEMV and batch-parallel GEMM.
//!
//! On WASM targets (`wasm32`), rayon is unavailable (no threads).
//! All parallel entry points fall back to sequential execution transparently.

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::dispatch::KernelDispatcher;
use crate::error::{KernelError, KernelResult};
use crate::traits::Fp8Kernel;
use crate::traits::OneBitKernel;
use crate::traits::TernaryKernel;
use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8};
use oxibonsai_core::QK_TQ2_0_G128;

/// Minimum number of rows before engaging parallel GEMV.
/// Below this threshold, the overhead of thread spawning exceeds the benefit.
const PAR_GEMV_MIN_ROWS: usize = 64;

/// Minimum batch size before engaging parallel GEMM.
const PAR_GEMM_MIN_BATCH: usize = 4;

/// Parallel row-wise 1-bit GEMV.
///
/// Each row's dot product is independent, making this trivially parallelizable.
/// Falls back to sequential for small `n_rows` to avoid overhead.
pub fn gemv_1bit_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    // Validation
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

    // Sequential fallback for small row counts
    if n_rows < PAR_GEMV_MIN_ROWS {
        return dispatcher.gemv(blocks, input, output, n_rows, k);
    }

    // On WASM: no rayon threads available — fall back to sequential.
    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemv(blocks, input, output, n_rows, k);
    }

    // Parallel: each chunk processes a subset of rows
    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..n_rows]
            .par_chunks_mut(1)
            .enumerate()
            .try_for_each(|(row, out_chunk)| {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                // Use single-row GEMV via the dispatcher
                dispatcher.gemv(row_blocks, input, out_chunk, 1, k)
            })?;

        Ok(())
    }
}

/// Parallel batch-wise 1-bit GEMM.
///
/// Each batch element's row is independent, making this parallelizable
/// along the M (batch/sequence) dimension.
pub fn gemm_1bit_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    // Validation
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

    // Sequential fallback for small batches
    if m < PAR_GEMM_MIN_BATCH {
        return dispatcher.gemm(blocks, input, output, m, n_rows, k);
    }

    // On WASM: no rayon threads available — fall back to sequential.
    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemm(blocks, input, output, m, n_rows, k);
    }

    // Parallel: each batch element processes independently
    // Split output into m chunks of n_rows, each paired with its input row
    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..m * n_rows]
            .par_chunks_mut(n_rows)
            .enumerate()
            .try_for_each(|(mi, out_row)| {
                let input_row = &input[mi * k..(mi + 1) * k];
                // Each batch element is a single-row GEMM (effectively GEMV across all weight rows)
                dispatcher.gemm(blocks, input_row, out_row, 1, n_rows, k)
            })?;

        Ok(())
    }
}

/// Parallel row-wise ternary GEMV.
///
/// Each row's dot product is independent, making this trivially parallelizable.
/// Falls back to sequential for small `n_rows` to avoid overhead.
pub fn gemv_ternary_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK_TQ2_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    if n_rows < PAR_GEMV_MIN_ROWS {
        return dispatcher.gemv_ternary_g128(blocks, input, output, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemv_ternary_g128(blocks, input, output, n_rows, k);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..n_rows]
            .par_chunks_mut(1)
            .enumerate()
            .try_for_each(|(row, out_chunk)| {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                dispatcher.gemv_ternary_g128(row_blocks, input, out_chunk, 1, k)
            })?;

        Ok(())
    }
}

/// Parallel batch-wise ternary GEMM.
///
/// Each batch element's row is independent, making this parallelizable
/// along the M (batch/sequence) dimension.
pub fn gemm_ternary_g128_par(
    dispatcher: &KernelDispatcher,
    blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if k % QK_TQ2_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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

    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    if m < PAR_GEMM_MIN_BATCH {
        return dispatcher.gemm_ternary_g128(blocks, input, output, m, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemm_ternary_g128(blocks, input, output, m, n_rows, k);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..m * n_rows]
            .par_chunks_mut(n_rows)
            .enumerate()
            .try_for_each(|(mi, out_row)| {
                let input_row = &input[mi * k..(mi + 1) * k];
                dispatcher.gemm_ternary_g128(blocks, input_row, out_row, 1, n_rows, k)
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
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
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
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    if n_rows < PAR_GEMV_MIN_ROWS {
        return dispatcher.gemv_fp8_e4m3(blocks, input, output, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemv_fp8_e4m3(blocks, input, output, n_rows, k);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..n_rows]
            .par_chunks_mut(1)
            .enumerate()
            .try_for_each(|(row, out_chunk)| {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                dispatcher.gemv_fp8_e4m3(row_blocks, input, out_chunk, 1, k)
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
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
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
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    if n_rows < PAR_GEMV_MIN_ROWS {
        return dispatcher.gemv_fp8_e5m2(blocks, input, output, n_rows, k);
    }

    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemv_fp8_e5m2(blocks, input, output, n_rows, k);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        output[..n_rows]
            .par_chunks_mut(1)
            .enumerate()
            .try_for_each(|(row, out_chunk)| {
                let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
                dispatcher.gemv_fp8_e5m2(row_blocks, input, out_chunk, 1, k)
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
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if inputs.len() < batch * k {
        return Err(KernelError::DimensionMismatch {
            expected: batch * k,
            got: inputs.len(),
        });
    }
    if outputs.len() < batch * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: batch * n_rows,
            available: outputs.len(),
        });
    }
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    if batch < PAR_GEMM_MIN_BATCH {
        return dispatcher.gemm_fp8_e4m3(blocks, inputs, outputs, n_rows, k, batch);
    }

    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemm_fp8_e4m3(blocks, inputs, outputs, n_rows, k, batch);
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
    if k % QK_FP8 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_FP8,
        });
    }
    if inputs.len() < batch * k {
        return Err(KernelError::DimensionMismatch {
            expected: batch * k,
            got: inputs.len(),
        });
    }
    if outputs.len() < batch * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: batch * n_rows,
            available: outputs.len(),
        });
    }
    let blocks_per_row = k / QK_FP8;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::DimensionMismatch {
            expected: expected_blocks,
            got: blocks.len(),
        });
    }

    if batch < PAR_GEMM_MIN_BATCH {
        return dispatcher.gemm_fp8_e5m2(blocks, inputs, outputs, n_rows, k, batch);
    }

    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.gemm_fp8_e5m2(blocks, inputs, outputs, n_rows, k, batch);
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
        return Err(KernelError::BufferTooSmall {
            needed: total_elements,
            available: output.len(),
        });
    }

    // For small block counts, sequential is faster
    if blocks.len() < 64 {
        return dispatcher.dequant(blocks, output);
    }

    // On WASM: no rayon threads available — fall back to sequential.
    #[cfg(target_arch = "wasm32")]
    {
        return dispatcher.dequant(blocks, output);
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
    fn par_gemv_matches_sequential() {
        let n_rows = 128; // Above PAR_GEMV_MIN_ROWS threshold
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
        let m = 8; // Above PAR_GEMM_MIN_BATCH threshold
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
        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks = vec![make_ternary_block([0xAAu8; 32]); n_rows * blocks_per_row];
        let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let dispatcher = KernelDispatcher::auto_detect();

        let mut output = vec![0.0f32; n_rows];
        gemv_ternary_g128_par(&dispatcher, &blocks, &input, &mut output, n_rows, k)?;
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
}
