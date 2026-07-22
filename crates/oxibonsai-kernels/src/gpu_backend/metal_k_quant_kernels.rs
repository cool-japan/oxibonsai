//! Direct Metal dispatch engine for OxiBonsai K-quant GEMV
//! (`Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_K`).
//!
//! Metal counterpart of `cuda_k_quant_kernels.rs`, mirroring the Phase 27
//! `metal_fp8_kernels.rs` architecture:
//!
//! - Independent singleton (own [`metal::Device`] + [`metal::CommandQueue`]).
//! - Six compute pipelines, compiled lazily from MSL source at first use (no
//!   offline Metal Toolchain required).
//! - All buffers use shared storage (`MTLResourceOptions::StorageModeShared`).
//!
//! All K-quant formats use `QK_K = 256` weights per super-block, so `k` must be
//! a positive multiple of 256.
//!
//! # Public API
//!
//! - [`metal_gemv_q2k`] / [`metal_gemv_q3k`] / [`metal_gemv_q4k`]
//! - [`metal_gemv_q5k`] / [`metal_gemv_q6k`] / [`metal_gemv_q8k`]

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::OnceLock;

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

use super::kernel_sources::{
    MSL_GEMV_Q2K_V1, MSL_GEMV_Q3K_V1, MSL_GEMV_Q4K_V1, MSL_GEMV_Q5K_V1, MSL_GEMV_Q6K_V1,
    MSL_GEMV_Q8K_V1,
};
use super::metal_graph::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Weights per K-quant super-block (`QK_K`).
const QK_K: usize = 256;
/// Bytes per `Q2_K` super-block.
const Q2K_BLOCK_BYTES: usize = 84;
/// Bytes per `Q3_K` super-block.
const Q3K_BLOCK_BYTES: usize = 110;
/// Bytes per `Q4_K` super-block.
const Q4K_BLOCK_BYTES: usize = 144;
/// Bytes per `Q5_K` super-block.
const Q5K_BLOCK_BYTES: usize = 176;
/// Bytes per `Q6_K` super-block.
const Q6K_BLOCK_BYTES: usize = 210;
/// Bytes per `Q8_K` super-block.
const Q8K_BLOCK_BYTES: usize = 292;
/// Simdgroups per threadgroup (one output row per simdgroup).
const SIMDS_PER_TG: usize = 8;
/// Threads per threadgroup (8 simdgroups × 32 lanes).
const THREADS_PER_TG: u64 = 256;

// ═══════════════════════════════════════════════════════════════════════════
// Singleton state
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide Metal K-quant dispatch state (six compiled pipelines).
struct MetalKQuantState {
    device: Device,
    queue: CommandQueue,
    pipeline_q2k: ComputePipelineState,
    pipeline_q3k: ComputePipelineState,
    pipeline_q4k: ComputePipelineState,
    pipeline_q5k: ComputePipelineState,
    pipeline_q6k: ComputePipelineState,
    pipeline_q8k: ComputePipelineState,
}

// SAFETY: `metal::Device` / `metal::CommandQueue` are reference-counted
// Objective-C objects documented by Apple as safe to share across threads once
// initialised. This mirrors `metal_fp8_kernels::MetalFp8State`.
unsafe impl Send for MetalKQuantState {}
unsafe impl Sync for MetalKQuantState {}

impl MetalKQuantState {
    fn new() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();

        Ok(Self {
            pipeline_q2k: compile_pipeline(&device, &options, MSL_GEMV_Q2K_V1, "gemv_q2k")?,
            pipeline_q3k: compile_pipeline(&device, &options, MSL_GEMV_Q3K_V1, "gemv_q3k")?,
            pipeline_q4k: compile_pipeline(&device, &options, MSL_GEMV_Q4K_V1, "gemv_q4k")?,
            pipeline_q5k: compile_pipeline(&device, &options, MSL_GEMV_Q5K_V1, "gemv_q5k")?,
            pipeline_q6k: compile_pipeline(&device, &options, MSL_GEMV_Q6K_V1, "gemv_q6k")?,
            pipeline_q8k: compile_pipeline(&device, &options, MSL_GEMV_Q8K_V1, "gemv_q8k")?,
            device,
            queue,
        })
    }
}

/// Compile one MSL source string into a named compute pipeline.
fn compile_pipeline(
    device: &Device,
    options: &CompileOptions,
    source: &str,
    entry: &str,
) -> Result<ComputePipelineState, MetalGraphError> {
    let library = device
        .new_library_with_source(source, options)
        .map_err(|e| MetalGraphError::CompilationFailed(format!("{entry} library: {e}")))?;
    let function = library
        .get_function(entry, None)
        .map_err(|e| MetalGraphError::CompilationFailed(format!("{entry} function: {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalGraphError::CompilationFailed(format!("{entry} pipeline: {e}")))
}

/// Lazy process-wide singleton.
fn state() -> Result<&'static MetalKQuantState, MetalGraphError> {
    static STATE: OnceLock<Result<MetalKQuantState, MetalGraphError>> = OnceLock::new();
    match STATE.get_or_init(MetalKQuantState::new) {
        Ok(s) => Ok(s),
        Err(e) => Err(clone_err(e)),
    }
}

fn clone_err(e: &MetalGraphError) -> MetalGraphError {
    match e {
        MetalGraphError::DeviceNotFound => MetalGraphError::DeviceNotFound,
        MetalGraphError::CompilationFailed(s) => MetalGraphError::CompilationFailed(s.clone()),
        MetalGraphError::BufferCreationFailed => MetalGraphError::BufferCreationFailed,
        MetalGraphError::EncodingFailed(s) => MetalGraphError::EncodingFailed(s.clone()),
        MetalGraphError::ExecutionFailed(s) => MetalGraphError::ExecutionFailed(s.clone()),
        MetalGraphError::InvalidDimensions(s) => MetalGraphError::InvalidDimensions(s.clone()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public dispatch functions
// ═══════════════════════════════════════════════════════════════════════════

/// `Q2_K` GEMV on the Metal GPU.
///
/// # Arguments
/// - `blocks`: raw AoS super-block bytes, length `n_rows * (k / 256) * 84`.
/// - `input`: dense FP32 input vector, length `k`.
/// - `output`: dense FP32 output vector, length `n_rows`.
/// - `n_rows`: number of output rows.
/// - `k`: input dimension (must be a positive multiple of 256).
///
/// # Errors
/// Returns [`MetalGraphError::DeviceNotFound`] on systems without a Metal device,
/// [`MetalGraphError::CompilationFailed`] on pipeline-build failure, or
/// [`MetalGraphError::EncodingFailed`] for shape/buffer mismatches.
pub fn metal_gemv_q2k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q2k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q2K_BLOCK_BYTES,
        "Q2_K",
    )
}

/// `Q3_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q3k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q3k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q3K_BLOCK_BYTES,
        "Q3_K",
    )
}

/// `Q4_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q4k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q4k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q4K_BLOCK_BYTES,
        "Q4_K",
    )
}

/// `Q5_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q5k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q5k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q5K_BLOCK_BYTES,
        "Q5_K",
    )
}

/// `Q6_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q6k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q6k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q6K_BLOCK_BYTES,
        "Q6_K",
    )
}

/// `Q8_K` GEMV on the Metal GPU. See [`metal_gemv_q2k`].
pub fn metal_gemv_q8k(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_k_quant_gemv(
        s,
        &s.pipeline_q8k,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q8K_BLOCK_BYTES,
        "Q8_K",
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_k_quant_gemv(
    s: &MetalKQuantState,
    pipeline: &ComputePipelineState,
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    block_bytes: usize,
    format: &str,
) -> Result<(), MetalGraphError> {
    // ── Validate dimensions ─────────────────────────────────────────────────
    if k == 0 || k % QK_K != 0 {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: k = {k} must be a non-zero multiple of {QK_K}"
        )));
    }
    let blocks_per_row = k / QK_K;
    let expected_block_bytes = n_rows.saturating_mul(blocks_per_row) * block_bytes;
    if blocks.len() != expected_block_bytes {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: blocks.len() = {} expected {expected_block_bytes} (n_rows = {n_rows}, k = {k})",
            blocks.len()
        )));
    }
    if input.len() != k {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: input.len() = {} expected {k}",
            input.len()
        )));
    }
    if output.len() != n_rows {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: output.len() = {} expected {n_rows}",
            output.len()
        )));
    }
    if n_rows == 0 {
        return Ok(());
    }

    // ── Allocate buffers (shared storage) ───────────────────────────────────
    let block_buf = s.device.new_buffer_with_data(
        blocks.as_ptr() as *const std::ffi::c_void,
        blocks.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = s.device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(input) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = s.device.new_buffer(
        (n_rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    // Zero-initialise output (some drivers leave new buffers uninitialised).
    unsafe {
        std::ptr::write_bytes(output_buf.contents() as *mut f32, 0u8, n_rows);
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("n_rows = {n_rows} exceeds u32::MAX"))
    })?;
    let k_u32 = u32::try_from(k)
        .map_err(|_| MetalGraphError::EncodingFailed(format!("k = {k} exceeds u32::MAX")))?;

    // ── Encode + commit ─────────────────────────────────────────────────────
    let cmd = s.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&block_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &n_rows_u32 as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const std::ffi::c_void,
    );

    let n_tgs = n_rows.div_ceil(SIMDS_PER_TG) as u64;
    let grid = metal::MTLSize::new(n_tgs, 1, 1);
    let tg_size = metal::MTLSize::new(THREADS_PER_TG, 1, 1);
    encoder.dispatch_thread_groups(grid, tg_size);
    encoder.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // ── Read output back ────────────────────────────────────────────────────
    unsafe {
        let src = output_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, output.as_mut_ptr(), n_rows);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — CI-GPU-gated parity, host-only constant checks elsewhere
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_constants_match_core() {
        assert_eq!(QK_K, oxibonsai_core::quant_k::QK_K);
        assert_eq!(Q2K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q2_K_BYTES);
        assert_eq!(Q3K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q3K_BYTES);
        assert_eq!(Q4K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q4_K_BYTES);
        assert_eq!(Q5K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q5K_BYTES);
        assert_eq!(Q6K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q6K_BYTES);
        assert_eq!(Q8K_BLOCK_BYTES, oxibonsai_core::BLOCK_Q8K_BYTES);
    }

    /// `k` not a multiple of 256 is rejected before any GPU work.
    #[test]
    fn q4k_bad_k_rejected() {
        if state().is_err() {
            return;
        }
        let blocks = vec![0u8; Q4K_BLOCK_BYTES];
        let input = vec![0.0f32; 255];
        let mut output = vec![0.0f32; 1];
        assert!(metal_gemv_q4k(&blocks, &input, &mut output, 1, 255).is_err());
    }
}
