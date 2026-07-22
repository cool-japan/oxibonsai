//! Direct Metal dispatch engine for OxiBonsai standard-quant (`Q4_0` / `Q8_0`) GEMV.
//!
//! Metal counterpart of `cuda_q_std_kernels.rs`, mirroring the Phase 27
//! `metal_fp8_kernels.rs` architecture:
//!
//! - Independent singleton (own [`metal::Device`] + [`metal::CommandQueue`]).
//! - Two compute pipelines: `gemv_q4_0` and `gemv_q8_0`, compiled lazily from
//!   MSL source at first use (no offline Metal Toolchain required).
//! - All buffers use shared storage (`MTLResourceOptions::StorageModeShared`)
//!   so CPU-side reads/writes need no explicit blit.
//!
//! Kept in its own file (not merged into `metal_graph`) to honor the 2000-line
//! refactoring policy.
//!
//! # Public API
//!
//! - [`metal_gemv_q4_0`] — `Q4_0` GEMV (18 bytes/block, 32 weights).
//! - [`metal_gemv_q8_0`] — `Q8_0` GEMV (34 bytes/block, 32 weights).

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::OnceLock;

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

use super::kernel_sources::{MSL_GEMV_Q4_0_V1, MSL_GEMV_Q8_0_V1};
use super::metal_graph::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Weights per standard-quant block (`Q4_0` / `Q8_0`).
const Q_STD_BLOCK_K: usize = 32;
/// Bytes per `Q4_0` block (FP16 scale + 16 nibble bytes).
const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes per `Q8_0` block (FP16 scale + 32 int8 weights).
const Q8_0_BLOCK_BYTES: usize = 34;
/// Simdgroups per threadgroup (one output row per simdgroup).
const SIMDS_PER_TG: usize = 8;
/// Threads per threadgroup (8 simdgroups × 32 lanes).
const THREADS_PER_TG: u64 = 256;

// ═══════════════════════════════════════════════════════════════════════════
// Singleton state
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide Metal standard-quant dispatch state.
struct MetalQStdState {
    device: Device,
    queue: CommandQueue,
    pipeline_q4_0: ComputePipelineState,
    pipeline_q8_0: ComputePipelineState,
}

// SAFETY: `metal::Device` / `metal::CommandQueue` are reference-counted
// Objective-C objects documented by Apple as safe to share across threads once
// initialised. This mirrors `metal_fp8_kernels::MetalFp8State`.
unsafe impl Send for MetalQStdState {}
unsafe impl Sync for MetalQStdState {}

impl MetalQStdState {
    fn new() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();

        let pipeline_q4_0 = compile_pipeline(&device, &options, MSL_GEMV_Q4_0_V1, "gemv_q4_0")?;
        let pipeline_q8_0 = compile_pipeline(&device, &options, MSL_GEMV_Q8_0_V1, "gemv_q8_0")?;

        Ok(Self {
            device,
            queue,
            pipeline_q4_0,
            pipeline_q8_0,
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
fn state() -> Result<&'static MetalQStdState, MetalGraphError> {
    static STATE: OnceLock<Result<MetalQStdState, MetalGraphError>> = OnceLock::new();
    match STATE.get_or_init(MetalQStdState::new) {
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

/// `Q4_0` GEMV on the Metal GPU.
///
/// # Arguments
/// - `blocks`: raw AoS block bytes, length `n_rows * (k / 32) * 18`.
/// - `input`: dense FP32 input vector, length `k`.
/// - `output`: dense FP32 output vector, length `n_rows`.
/// - `n_rows`: number of output rows.
/// - `k`: input dimension (must be a positive multiple of 32).
///
/// # Errors
/// Returns [`MetalGraphError::DeviceNotFound`] on systems without a Metal device,
/// [`MetalGraphError::CompilationFailed`] if pipeline creation failed, or
/// [`MetalGraphError::EncodingFailed`] for shape/buffer mismatches.
pub fn metal_gemv_q4_0(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_q_std_gemv(
        s,
        &s.pipeline_q4_0,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q4_0_BLOCK_BYTES,
        "Q4_0",
    )
}

/// `Q8_0` GEMV on the Metal GPU. See [`metal_gemv_q4_0`].
pub fn metal_gemv_q8_0(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    let s = state()?;
    dispatch_q_std_gemv(
        s,
        &s.pipeline_q8_0,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q8_0_BLOCK_BYTES,
        "Q8_0",
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_q_std_gemv(
    s: &MetalQStdState,
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
    if k == 0 || k % Q_STD_BLOCK_K != 0 {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: k = {k} must be a non-zero multiple of {Q_STD_BLOCK_K}"
        )));
    }
    let blocks_per_row = k / Q_STD_BLOCK_K;
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
        assert_eq!(Q4_0_BLOCK_BYTES, oxibonsai_core::BLOCK_Q4_0_BYTES);
        assert_eq!(Q8_0_BLOCK_BYTES, oxibonsai_core::BLOCK_Q8_0_BYTES);
        assert_eq!(Q_STD_BLOCK_K, oxibonsai_core::QK_Q4_0);
        assert_eq!(Q_STD_BLOCK_K, oxibonsai_core::QK_Q8_0);
    }

    /// `k` not a multiple of 32 is rejected before any GPU work.
    #[test]
    fn q4_0_bad_k_rejected() {
        if state().is_err() {
            return;
        }
        let blocks = vec![0u8; Q4_0_BLOCK_BYTES];
        let input = vec![0.0f32; 31];
        let mut output = vec![0.0f32; 1];
        assert!(metal_gemv_q4_0(&blocks, &input, &mut output, 1, 31).is_err());
    }
}
