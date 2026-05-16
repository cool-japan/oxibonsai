//! Direct Metal dispatch engine for OxiBonsai FP8 (E4M3 + E5M2) **batch prefill** kernels.
//!
//! Phase 28 — Metal counterpart of `cuda_fp8_prefill.rs` batch GEMMs.
//!
//! # Architecture
//!
//! - Independent singleton (own [`metal::Device`] + [`metal::CommandQueue`]) — kept
//!   separate from [`metal_fp8_kernels`](super::metal_fp8_kernels)'s singleton so
//!   the batch-prefill pipelines compile lazily without paying single-token GEMV's
//!   init cost on processes that never touch prefill.
//! - 6 compute pipelines, all `[[buffer(N)]]`-annotated:
//!   - `gemm_fp8_e4m3`                          / `gemm_fp8_e5m2`
//!   - `gemm_fp8_e4m3_residual`                 / `gemm_fp8_e5m2_residual`
//!   - `fused_gate_up_swiglu_gemm_fp8_e4m3`     / `fused_gate_up_swiglu_gemm_fp8_e5m2`
//! - All buffers use shared storage (`MTLResourceOptions::StorageModeShared`).
//!
//! Kept in its own file rather than merged into `metal_graph.rs` to honor the
//! 2000-line refactoring policy.
//!
//! # Block layout (AoS, 34 bytes/block — matches `BlockFP8E4M3` / `BlockFP8E5M2`)
//!
//! ```text
//! Block[i] = [q0, q1, ..., q31, scale_lo, scale_hi]
//! ```
//!
//! # Batch tensor layout
//!
//! Inputs and outputs use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch / token index. Matches Q1 / TQ2 V7 batch GEMM.
//!
//! # Public API
//!
//! - [`metal_gemm_fp8_e4m3`] — batch FP8 E4M3 GEMM (accumulate with +=)
//! - [`metal_gemm_fp8_e5m2`] — batch FP8 E5M2 GEMM (accumulate with +=)
//! - [`metal_gemm_fp8_e4m3_residual`] — batch FP8 E4M3 GEMM with fused residual add
//! - [`metal_gemm_fp8_e5m2_residual`] — batch FP8 E5M2 GEMM with fused residual add
//! - [`metal_fused_gate_up_swiglu_fp8_e4m3`] — fused gate + up + SwiGLU FP8 E4M3
//! - [`metal_fused_gate_up_swiglu_fp8_e5m2`] — fused gate + up + SwiGLU FP8 E5M2

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::OnceLock;

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

use super::kernel_sources::{
    MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1, MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1,
    MSL_GEMM_FP8_E4M3_RESIDUAL_V1, MSL_GEMM_FP8_E4M3_V1, MSL_GEMM_FP8_E5M2_RESIDUAL_V1,
    MSL_GEMM_FP8_E5M2_V1,
};
use super::metal_graph::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Singleton state
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide Metal FP8 prefill dispatch state.
struct MetalFp8PrefillState {
    device: Device,
    queue: CommandQueue,
    gemm_e4m3: ComputePipelineState,
    gemm_e4m3_residual: ComputePipelineState,
    fused_gate_up_swiglu_e4m3: ComputePipelineState,
    gemm_e5m2: ComputePipelineState,
    gemm_e5m2_residual: ComputePipelineState,
    fused_gate_up_swiglu_e5m2: ComputePipelineState,
}

// SAFETY: `metal::Device` / `metal::CommandQueue` / `metal::ComputePipelineState`
// are reference-counted ObjC objects documented as thread-safe by Apple's Metal SDK.
unsafe impl Send for MetalFp8PrefillState {}
unsafe impl Sync for MetalFp8PrefillState {}

impl MetalFp8PrefillState {
    fn new() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();

        let gemm_e4m3 = compile_pipeline(&device, &opts, MSL_GEMM_FP8_E4M3_V1, "gemm_fp8_e4m3")?;
        let gemm_e4m3_residual = compile_pipeline(
            &device,
            &opts,
            MSL_GEMM_FP8_E4M3_RESIDUAL_V1,
            "gemm_fp8_e4m3_residual",
        )?;
        let fused_gate_up_swiglu_e4m3 = compile_pipeline(
            &device,
            &opts,
            MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E4M3_V1,
            "fused_gate_up_swiglu_gemm_fp8_e4m3",
        )?;
        let gemm_e5m2 = compile_pipeline(&device, &opts, MSL_GEMM_FP8_E5M2_V1, "gemm_fp8_e5m2")?;
        let gemm_e5m2_residual = compile_pipeline(
            &device,
            &opts,
            MSL_GEMM_FP8_E5M2_RESIDUAL_V1,
            "gemm_fp8_e5m2_residual",
        )?;
        let fused_gate_up_swiglu_e5m2 = compile_pipeline(
            &device,
            &opts,
            MSL_FUSED_GATE_UP_SWIGLU_GEMM_FP8_E5M2_V1,
            "fused_gate_up_swiglu_gemm_fp8_e5m2",
        )?;

        Ok(Self {
            device,
            queue,
            gemm_e4m3,
            gemm_e4m3_residual,
            fused_gate_up_swiglu_e4m3,
            gemm_e5m2,
            gemm_e5m2_residual,
            fused_gate_up_swiglu_e5m2,
        })
    }
}

fn compile_pipeline(
    device: &Device,
    opts: &CompileOptions,
    src: &str,
    entry: &str,
) -> Result<ComputePipelineState, MetalGraphError> {
    let lib = device.new_library_with_source(src, opts).map_err(|e| {
        MetalGraphError::CompilationFailed(format!("FP8 prefill library `{entry}`: {e}"))
    })?;
    let func = lib.get_function(entry, None).map_err(|e| {
        MetalGraphError::CompilationFailed(format!("FP8 prefill function `{entry}`: {e}"))
    })?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            MetalGraphError::CompilationFailed(format!("FP8 prefill pipeline `{entry}`: {e}"))
        })
}

/// Lazy process-wide singleton (separate from the Phase 27 GEMV singleton).
fn state() -> Result<&'static MetalFp8PrefillState, MetalGraphError> {
    static STATE: OnceLock<Result<MetalFp8PrefillState, MetalGraphError>> = OnceLock::new();
    match STATE.get_or_init(MetalFp8PrefillState::new) {
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
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Block size in bytes for FP8 E4M3 / E5M2 (32 quantised weights + FP16 scale).
const FP8_BLOCK_BYTES: usize = 34;
/// Quantisation group size (weights per block).
const FP8_BLOCK_K: usize = 32;
/// Simdgroups per threadgroup (= rows handled per CTA).
const SIMDS_PER_TG: usize = 8;
/// Threads per threadgroup (8 simdgroups × 32 lanes).
const THREADS_PER_TG: u64 = 256;

// ═══════════════════════════════════════════════════════════════════════════
// Variant selectors
// ═══════════════════════════════════════════════════════════════════════════

/// Which FP8 numeric format to use.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Fp8Variant {
    E4M3,
    E5M2,
}

// ═══════════════════════════════════════════════════════════════════════════
// Public batch GEMM API
// ═══════════════════════════════════════════════════════════════════════════

/// Batch FP8 E4M3 GEMM.
///
/// Accumulates into `outputs` with `+=`. Inputs and outputs use column-major
/// layout: `inputs[col * k + elem]`, `outputs[col * n_rows + row]`.
///
/// # Arguments
/// - `blocks`: raw block bytes (AoS, 34 bytes/block), length = `n_rows * (k/32) * 34`.
/// - `inputs`: `batch_size * k` floats, column-major.
/// - `outputs`: `batch_size * n_rows` floats, column-major (must already be sized;
///   the kernel accumulates with `+=` so initialise to zero for a fresh GEMM).
/// - `n_rows`: output rows.
/// - `k`: input dimension (multiple of 32).
/// - `batch_size`: number of batch columns.
pub fn metal_gemm_fp8_e4m3(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    dispatch_gemm(
        blocks,
        inputs,
        outputs,
        n_rows,
        k,
        batch_size,
        None,
        Fp8Variant::E4M3,
    )
}

/// Batch FP8 E5M2 GEMM. See [`metal_gemm_fp8_e4m3`].
pub fn metal_gemm_fp8_e5m2(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    dispatch_gemm(
        blocks,
        inputs,
        outputs,
        n_rows,
        k,
        batch_size,
        None,
        Fp8Variant::E5M2,
    )
}

/// Batch FP8 E4M3 GEMM with fused residual add.
///
/// Writes `outputs[idx] = residual[idx] + dot(blocks_row, inputs_col)` rather
/// than accumulating. Residual layout matches the output: column-major
/// `batch_size * n_rows` floats.
pub fn metal_gemm_fp8_e4m3_residual(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    residual: &[f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    dispatch_gemm(
        blocks,
        inputs,
        outputs,
        n_rows,
        k,
        batch_size,
        Some(residual),
        Fp8Variant::E4M3,
    )
}

/// Batch FP8 E5M2 GEMM with fused residual add. See [`metal_gemm_fp8_e4m3_residual`].
pub fn metal_gemm_fp8_e5m2_residual(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    residual: &[f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    dispatch_gemm(
        blocks,
        inputs,
        outputs,
        n_rows,
        k,
        batch_size,
        Some(residual),
        Fp8Variant::E5M2,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Public fused gate+up+SwiGLU API
// ═══════════════════════════════════════════════════════════════════════════

/// Fused gate + up FP8 E4M3 GEMM with SwiGLU epilogue.
///
/// The concatenated weight matrix at `blocks` covers `2 * n_ffn_rows` rows:
///   gate rows  `0..n_ffn_rows-1`
///   up   rows  `n_ffn_rows..2*n_ffn_rows-1`
///
/// For each `(row r, col c)`:
///   `outputs[c * n_ffn_rows + r] = SiLU(gate_dot(r, c)) * up_dot(r, c)`
///
/// `outputs` is overwritten (not accumulated). Both inputs and outputs are
/// column-major: `inputs[col * k + elem]`, `outputs[col * n_ffn_rows + row]`.
pub fn metal_fused_gate_up_swiglu_fp8_e4m3(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_ffn_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    dispatch_fused_gate_up_swiglu(
        blocks,
        inputs,
        outputs,
        n_ffn_rows,
        k,
        batch_size,
        Fp8Variant::E4M3,
    )
}

/// Fused gate + up FP8 E5M2 GEMM with SwiGLU epilogue. See
/// [`metal_fused_gate_up_swiglu_fp8_e4m3`].
pub fn metal_fused_gate_up_swiglu_fp8_e5m2(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_ffn_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    dispatch_fused_gate_up_swiglu(
        blocks,
        inputs,
        outputs,
        n_ffn_rows,
        k,
        batch_size,
        Fp8Variant::E5M2,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Private dispatch helpers
// ═══════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn dispatch_gemm(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
    residual: Option<&[f32]>,
    variant: Fp8Variant,
) -> Result<(), MetalGraphError> {
    validate_batch_dims(blocks, inputs, outputs, n_rows, k, batch_size)?;
    if let Some(r) = residual {
        if r.len() != batch_size * n_rows {
            return Err(MetalGraphError::EncodingFailed(format!(
                "residual.len() = {} expected {} (batch_size {batch_size} × n_rows {n_rows})",
                r.len(),
                batch_size * n_rows
            )));
        }
    }

    let s = state()?;

    let block_buf = s.device.new_buffer_with_data(
        blocks.as_ptr() as *const std::ffi::c_void,
        blocks.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = s.device.new_buffer_with_data(
        inputs.as_ptr() as *const std::ffi::c_void,
        (inputs.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = s.device.new_buffer_with_data(
        outputs.as_ptr() as *const std::ffi::c_void,
        (outputs.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let residual_buf = residual.map(|r| {
        s.device.new_buffer_with_data(
            r.as_ptr() as *const std::ffi::c_void,
            (r.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("n_rows = {n_rows} exceeds u32::MAX"))
    })?;
    let k_u32 = u32::try_from(k)
        .map_err(|_| MetalGraphError::EncodingFailed(format!("k = {k} exceeds u32::MAX")))?;
    let batch_u32 = u32::try_from(batch_size).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("batch_size = {batch_size} exceeds u32::MAX"))
    })?;

    let cmd = s.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let pipeline = match (variant, residual.is_some()) {
        (Fp8Variant::E4M3, false) => &s.gemm_e4m3,
        (Fp8Variant::E5M2, false) => &s.gemm_e5m2,
        (Fp8Variant::E4M3, true) => &s.gemm_e4m3_residual,
        (Fp8Variant::E5M2, true) => &s.gemm_e5m2_residual,
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&block_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    set_u32(encoder, 3, n_rows_u32);
    set_u32(encoder, 4, batch_u32);
    set_u32(encoder, 5, k_u32);
    if let Some(rbuf) = residual_buf.as_ref() {
        encoder.set_buffer(6, Some(rbuf), 0);
    }

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
        std::ptr::copy_nonoverlapping(src, outputs.as_mut_ptr(), outputs.len());
    }

    Ok(())
}

fn dispatch_fused_gate_up_swiglu(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_ffn_rows: usize,
    k: usize,
    batch_size: usize,
    variant: Fp8Variant,
) -> Result<(), MetalGraphError> {
    // The fused kernel reads gate + up rows from the same buffer, so the buffer
    // covers 2 * n_ffn_rows rows worth of FP8 blocks.
    if k == 0 || k % FP8_BLOCK_K != 0 {
        return Err(MetalGraphError::EncodingFailed(format!(
            "k = {k} must be a non-zero multiple of {FP8_BLOCK_K}"
        )));
    }
    let blocks_per_row = k / FP8_BLOCK_K;
    let expected_block_bytes = 2usize
        .saturating_mul(n_ffn_rows)
        .saturating_mul(blocks_per_row)
        .saturating_mul(FP8_BLOCK_BYTES);
    if blocks.len() != expected_block_bytes {
        return Err(MetalGraphError::EncodingFailed(format!(
            "blocks.len() = {} expected {} (2 × n_ffn_rows {n_ffn_rows} × blocks_per_row {blocks_per_row} × {FP8_BLOCK_BYTES})",
            blocks.len(),
            expected_block_bytes
        )));
    }
    if inputs.len() != batch_size * k {
        return Err(MetalGraphError::EncodingFailed(format!(
            "inputs.len() = {} expected {} (batch_size {batch_size} × k {k})",
            inputs.len(),
            batch_size * k
        )));
    }
    if outputs.len() != batch_size * n_ffn_rows {
        return Err(MetalGraphError::EncodingFailed(format!(
            "outputs.len() = {} expected {} (batch_size {batch_size} × n_ffn_rows {n_ffn_rows})",
            outputs.len(),
            batch_size * n_ffn_rows
        )));
    }

    let s = state()?;

    let block_buf = s.device.new_buffer_with_data(
        blocks.as_ptr() as *const std::ffi::c_void,
        blocks.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = s.device.new_buffer_with_data(
        inputs.as_ptr() as *const std::ffi::c_void,
        (inputs.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = s.device.new_buffer(
        (outputs.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let n_rows_u32 = u32::try_from(n_ffn_rows).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("n_ffn_rows = {n_ffn_rows} exceeds u32::MAX"))
    })?;
    let k_u32 = u32::try_from(k)
        .map_err(|_| MetalGraphError::EncodingFailed(format!("k = {k} exceeds u32::MAX")))?;
    let batch_u32 = u32::try_from(batch_size).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("batch_size = {batch_size} exceeds u32::MAX"))
    })?;

    let cmd = s.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let pipeline = match variant {
        Fp8Variant::E4M3 => &s.fused_gate_up_swiglu_e4m3,
        Fp8Variant::E5M2 => &s.fused_gate_up_swiglu_e5m2,
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&block_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    set_u32(encoder, 3, n_rows_u32);
    set_u32(encoder, 4, batch_u32);
    set_u32(encoder, 5, k_u32);

    let n_tgs = n_ffn_rows.div_ceil(SIMDS_PER_TG) as u64;
    let grid = metal::MTLSize::new(n_tgs, 1, 1);
    let tg_size = metal::MTLSize::new(THREADS_PER_TG, 1, 1);
    encoder.dispatch_thread_groups(grid, tg_size);
    encoder.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    unsafe {
        let src = output_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, outputs.as_mut_ptr(), outputs.len());
    }

    Ok(())
}

fn validate_batch_dims(
    blocks: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), MetalGraphError> {
    if k == 0 || k % FP8_BLOCK_K != 0 {
        return Err(MetalGraphError::EncodingFailed(format!(
            "k = {k} must be a non-zero multiple of {FP8_BLOCK_K}"
        )));
    }
    let blocks_per_row = k / FP8_BLOCK_K;
    let expected_block_bytes = n_rows.saturating_mul(blocks_per_row) * FP8_BLOCK_BYTES;
    if blocks.len() != expected_block_bytes {
        return Err(MetalGraphError::EncodingFailed(format!(
            "blocks.len() = {} expected {} (n_rows {n_rows} × blocks_per_row {blocks_per_row} × {FP8_BLOCK_BYTES})",
            blocks.len(),
            expected_block_bytes
        )));
    }
    if inputs.len() != batch_size * k {
        return Err(MetalGraphError::EncodingFailed(format!(
            "inputs.len() = {} expected {} (batch_size {batch_size} × k {k})",
            inputs.len(),
            batch_size * k
        )));
    }
    if outputs.len() != batch_size * n_rows {
        return Err(MetalGraphError::EncodingFailed(format!(
            "outputs.len() = {} expected {} (batch_size {batch_size} × n_rows {n_rows})",
            outputs.len(),
            batch_size * n_rows
        )));
    }
    Ok(())
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &value as *const u32 as *const std::ffi::c_void,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — CI-GPU-gated parity tests on Metal hardware; auto-skip elsewhere
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_core() {
        assert_eq!(FP8_BLOCK_BYTES, oxibonsai_core::BLOCK_FP8_BYTES);
        assert_eq!(FP8_BLOCK_K, oxibonsai_core::QK_FP8);
    }

    #[test]
    fn variant_compiles() {
        // Static check that both variants are routable.
        let _ = Fp8Variant::E4M3;
        let _ = Fp8Variant::E5M2;
    }

    // ─── CI-GPU-gated correctness tests ────────────────────────────────────
    //
    // Each test calls `state()` first and returns silently on hosts without a
    // Metal device. On Apple Silicon CI this exercises the real GPU dispatch.

    fn make_fp8_e4m3_blocks(
        n_rows: usize,
        k: usize,
        seed: u64,
    ) -> Vec<oxibonsai_core::BlockFP8E4M3> {
        let blocks_per_row = k / oxibonsai_core::QK_FP8;
        let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut qs = [0u8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    let mix = (row as u64)
                        .wrapping_mul(31)
                        .wrapping_add(b as u64 * 17)
                        .wrapping_add(i as u64)
                        .wrapping_add(seed);
                    *q = (mix as u8).wrapping_mul(13).wrapping_add(7);
                    if *q == 0x7F || *q == 0xFF {
                        *q ^= 0x01;
                    }
                }
                let scale_bits = (((row as u16).wrapping_mul(19) ^ (b as u16).wrapping_mul(23))
                    & 0x03FF)
                    | 0x3800;
                blocks.push(oxibonsai_core::BlockFP8E4M3 {
                    qs,
                    d: half::f16::from_bits(scale_bits),
                });
            }
        }
        blocks
    }

    fn make_fp8_e5m2_blocks(
        n_rows: usize,
        k: usize,
        seed: u64,
    ) -> Vec<oxibonsai_core::BlockFP8E5M2> {
        let blocks_per_row = k / oxibonsai_core::QK_FP8;
        let mut blocks = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut qs = [0u8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    let mix = (row as u64)
                        .wrapping_mul(29)
                        .wrapping_add(b as u64 * 11)
                        .wrapping_add(i as u64 * 3)
                        .wrapping_add(seed);
                    *q = (mix as u8).wrapping_mul(7).wrapping_add(3);
                    if (*q & 0x7C) == 0x7C {
                        *q ^= 0x04;
                    }
                }
                let scale_bits = (((row as u16).wrapping_mul(13) ^ (b as u16).wrapping_mul(7))
                    & 0x03FF)
                    | 0x3800;
                blocks.push(oxibonsai_core::BlockFP8E5M2 {
                    qs,
                    d: half::f16::from_bits(scale_bits),
                });
            }
        }
        blocks
    }

    /// Reference CPU batch GEMM for E4M3: outputs[col * n_rows + row] += dot(...)
    fn cpu_batch_gemm_e4m3(
        blocks: &[oxibonsai_core::BlockFP8E4M3],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch_size: usize,
        accumulate: bool,
    ) {
        let blocks_per_row = k / oxibonsai_core::QK_FP8;
        for col in 0..batch_size {
            let mut row_out = vec![0.0f32; n_rows];
            let in_off = col * k;
            crate::gemv_fp8::gemv_fp8_e4m3(
                blocks,
                &inputs[in_off..in_off + k],
                &mut row_out,
                n_rows,
                k,
            );
            let _ = blocks_per_row;
            for row in 0..n_rows {
                let idx = col * n_rows + row;
                if accumulate {
                    outputs[idx] += row_out[row];
                } else {
                    outputs[idx] = row_out[row];
                }
            }
        }
    }

    fn cpu_batch_gemm_e5m2(
        blocks: &[oxibonsai_core::BlockFP8E5M2],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch_size: usize,
        accumulate: bool,
    ) {
        for col in 0..batch_size {
            let mut row_out = vec![0.0f32; n_rows];
            let in_off = col * k;
            crate::gemv_fp8::gemv_fp8_e5m2(
                blocks,
                &inputs[in_off..in_off + k],
                &mut row_out,
                n_rows,
                k,
            );
            for row in 0..n_rows {
                let idx = col * n_rows + row;
                if accumulate {
                    outputs[idx] += row_out[row];
                } else {
                    outputs[idx] = row_out[row];
                }
            }
        }
    }

    fn assert_close(cpu: &[f32], gpu: &[f32], tol: f32, tag: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{tag}: length mismatch");
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let diff = (c - g).abs();
            let rel = diff / c.abs().max(1e-6);
            assert!(
                diff < tol || rel < tol,
                "{tag} idx {i}: cpu={c} gpu={g} diff={diff} rel={rel}"
            );
        }
    }

    #[test]
    fn metal_gemm_fp8_e4m3_matches_cpu_reference() {
        if state().is_err() {
            return;
        }
        let n_rows = 16usize;
        let k = 128usize;
        let batch_size = 4usize;
        let blocks = make_fp8_e4m3_blocks(n_rows, k, 1234);
        let inputs: Vec<f32> = (0..batch_size * k)
            .map(|i| (i as f32 * 0.013).sin() * 0.5)
            .collect();

        let mut cpu_out = vec![0.0f32; batch_size * n_rows];
        cpu_batch_gemm_e4m3(&blocks, &inputs, &mut cpu_out, n_rows, k, batch_size, true);

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; batch_size * n_rows];
        metal_gemm_fp8_e4m3(bytes, &inputs, &mut gpu_out, n_rows, k, batch_size)
            .expect("metal FP8 E4M3 batch GEMM should succeed");
        assert_close(&cpu_out, &gpu_out, 1e-3, "gemm_fp8_e4m3");
    }

    /// Cap-of-8 discriminator: batch_size = 12 (> 8) ensures the outer
    /// `col_base += 8u` chunk loop processes the trailing cols correctly.
    #[test]
    fn metal_gemm_fp8_e4m3_capof8_batch12() {
        if state().is_err() {
            return;
        }
        let n_rows = 24usize;
        let k = 64usize;
        let batch_size = 12usize;
        let blocks = make_fp8_e4m3_blocks(n_rows, k, 9001);
        let inputs: Vec<f32> = (0..batch_size * k)
            .map(|i| ((i as f32 * 0.017).cos() + 0.3) * 0.4)
            .collect();

        let mut cpu_out = vec![0.0f32; batch_size * n_rows];
        cpu_batch_gemm_e4m3(&blocks, &inputs, &mut cpu_out, n_rows, k, batch_size, true);

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; batch_size * n_rows];
        metal_gemm_fp8_e4m3(bytes, &inputs, &mut gpu_out, n_rows, k, batch_size)
            .expect("metal FP8 E4M3 batch GEMM should succeed");
        assert_close(&cpu_out, &gpu_out, 1e-3, "gemm_fp8_e4m3 batch12");
    }

    #[test]
    fn metal_gemm_fp8_e4m3_residual_matches_cpu_reference() {
        if state().is_err() {
            return;
        }
        let n_rows = 16usize;
        let k = 96usize;
        let batch_size = 3usize;
        let blocks = make_fp8_e4m3_blocks(n_rows, k, 42);
        let inputs: Vec<f32> = (0..batch_size * k)
            .map(|i| ((i as f32 * 0.011) % 1.0) - 0.5)
            .collect();
        let residual: Vec<f32> = (0..batch_size * n_rows)
            .map(|i| (i as f32 * 0.05).sin() * 0.25)
            .collect();

        let mut cpu_out = vec![0.0f32; batch_size * n_rows];
        cpu_batch_gemm_e4m3(&blocks, &inputs, &mut cpu_out, n_rows, k, batch_size, false);
        for i in 0..cpu_out.len() {
            cpu_out[i] += residual[i];
        }

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; batch_size * n_rows];
        metal_gemm_fp8_e4m3_residual(
            bytes,
            &inputs,
            &mut gpu_out,
            &residual,
            n_rows,
            k,
            batch_size,
        )
        .expect("metal FP8 E4M3 residual GEMM should succeed");
        assert_close(&cpu_out, &gpu_out, 1e-3, "gemm_fp8_e4m3_residual");
    }

    #[test]
    fn metal_gemm_fp8_e5m2_matches_cpu_reference() {
        if state().is_err() {
            return;
        }
        let n_rows = 17usize; // boundary: row count not a multiple of 8
        let k = 96usize;
        let batch_size = 5usize;
        let blocks = make_fp8_e5m2_blocks(n_rows, k, 2024);
        let inputs: Vec<f32> = (0..batch_size * k)
            .map(|i| ((i as f32 * 0.019).cos()) * 0.3)
            .collect();

        let mut cpu_out = vec![0.0f32; batch_size * n_rows];
        cpu_batch_gemm_e5m2(&blocks, &inputs, &mut cpu_out, n_rows, k, batch_size, true);

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; batch_size * n_rows];
        metal_gemm_fp8_e5m2(bytes, &inputs, &mut gpu_out, n_rows, k, batch_size)
            .expect("metal FP8 E5M2 batch GEMM should succeed");
        assert_close(&cpu_out, &gpu_out, 1e-3, "gemm_fp8_e5m2");
    }

    #[test]
    fn metal_gemm_fp8_e5m2_residual_matches_cpu_reference() {
        if state().is_err() {
            return;
        }
        let n_rows = 16usize;
        let k = 64usize;
        let batch_size = 7usize;
        let blocks = make_fp8_e5m2_blocks(n_rows, k, 7);
        let inputs: Vec<f32> = (0..batch_size * k)
            .map(|i| (i as f32 * 0.007).tan().clamp(-1.0, 1.0))
            .collect();
        let residual: Vec<f32> = (0..batch_size * n_rows)
            .map(|i| (i as f32 * 0.03).cos() * 0.1)
            .collect();

        let mut cpu_out = vec![0.0f32; batch_size * n_rows];
        cpu_batch_gemm_e5m2(&blocks, &inputs, &mut cpu_out, n_rows, k, batch_size, false);
        for i in 0..cpu_out.len() {
            cpu_out[i] += residual[i];
        }

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; batch_size * n_rows];
        metal_gemm_fp8_e5m2_residual(
            bytes,
            &inputs,
            &mut gpu_out,
            &residual,
            n_rows,
            k,
            batch_size,
        )
        .expect("metal FP8 E5M2 residual GEMM should succeed");
        assert_close(&cpu_out, &gpu_out, 1e-3, "gemm_fp8_e5m2_residual");
    }

    #[test]
    fn metal_fused_gate_up_swiglu_fp8_e4m3_matches_cpu_reference() {
        if state().is_err() {
            return;
        }
        let n_ffn_rows = 16usize;
        let k = 64usize;
        let batch_size = 3usize;
        // Concatenated gate + up: 2 * n_ffn_rows total rows.
        let blocks = make_fp8_e4m3_blocks(2 * n_ffn_rows, k, 555);
        let inputs: Vec<f32> = (0..batch_size * k)
            .map(|i| ((i as f32 * 0.021).sin()) * 0.4)
            .collect();

        // Compute CPU reference: gate_dot = first n_ffn_rows rows; up_dot = next n_ffn_rows rows.
        let mut cpu_out = vec![0.0f32; batch_size * n_ffn_rows];
        let gate_blocks = &blocks[0..n_ffn_rows * (k / oxibonsai_core::QK_FP8)];
        let up_blocks = &blocks[n_ffn_rows * (k / oxibonsai_core::QK_FP8)..];
        for col in 0..batch_size {
            let mut gate_out = vec![0.0f32; n_ffn_rows];
            let mut up_out = vec![0.0f32; n_ffn_rows];
            let in_off = col * k;
            crate::gemv_fp8::gemv_fp8_e4m3(
                gate_blocks,
                &inputs[in_off..in_off + k],
                &mut gate_out,
                n_ffn_rows,
                k,
            );
            crate::gemv_fp8::gemv_fp8_e4m3(
                up_blocks,
                &inputs[in_off..in_off + k],
                &mut up_out,
                n_ffn_rows,
                k,
            );
            for row in 0..n_ffn_rows {
                let g = gate_out[row];
                let u = up_out[row];
                let silu_g = g / (1.0 + (-g).exp());
                cpu_out[col * n_ffn_rows + row] = silu_g * u;
            }
        }

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; batch_size * n_ffn_rows];
        metal_fused_gate_up_swiglu_fp8_e4m3(
            bytes,
            &inputs,
            &mut gpu_out,
            n_ffn_rows,
            k,
            batch_size,
        )
        .expect("metal FP8 E4M3 fused gate+up should succeed");
        assert_close(&cpu_out, &gpu_out, 5e-3, "fused_gate_up_swiglu_fp8_e4m3");
    }

    #[test]
    fn metal_fused_gate_up_swiglu_fp8_e5m2_matches_cpu_reference() {
        if state().is_err() {
            return;
        }
        let n_ffn_rows = 16usize;
        let k = 64usize;
        let batch_size = 3usize;
        let blocks = make_fp8_e5m2_blocks(2 * n_ffn_rows, k, 4444);
        let inputs: Vec<f32> = (0..batch_size * k)
            .map(|i| ((i as f32 * 0.023).cos()) * 0.3)
            .collect();

        let mut cpu_out = vec![0.0f32; batch_size * n_ffn_rows];
        let gate_blocks = &blocks[0..n_ffn_rows * (k / oxibonsai_core::QK_FP8)];
        let up_blocks = &blocks[n_ffn_rows * (k / oxibonsai_core::QK_FP8)..];
        for col in 0..batch_size {
            let mut gate_out = vec![0.0f32; n_ffn_rows];
            let mut up_out = vec![0.0f32; n_ffn_rows];
            let in_off = col * k;
            crate::gemv_fp8::gemv_fp8_e5m2(
                gate_blocks,
                &inputs[in_off..in_off + k],
                &mut gate_out,
                n_ffn_rows,
                k,
            );
            crate::gemv_fp8::gemv_fp8_e5m2(
                up_blocks,
                &inputs[in_off..in_off + k],
                &mut up_out,
                n_ffn_rows,
                k,
            );
            for row in 0..n_ffn_rows {
                let g = gate_out[row];
                let u = up_out[row];
                let silu_g = g / (1.0 + (-g).exp());
                cpu_out[col * n_ffn_rows + row] = silu_g * u;
            }
        }

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr().cast::<u8>(),
                blocks.len() * oxibonsai_core::BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; batch_size * n_ffn_rows];
        metal_fused_gate_up_swiglu_fp8_e5m2(
            bytes,
            &inputs,
            &mut gpu_out,
            n_ffn_rows,
            k,
            batch_size,
        )
        .expect("metal FP8 E5M2 fused gate+up should succeed");
        assert_close(&cpu_out, &gpu_out, 5e-2, "fused_gate_up_swiglu_fp8_e5m2");
    }

    /// Shape-validation test (runs on every host, no GPU needed).
    #[test]
    fn rejects_k_not_multiple_of_32() {
        let blocks = vec![0u8; 34];
        let inputs = vec![0.0f32; 33];
        let mut outputs = vec![0.0f32; 1];
        let err = dispatch_gemm(
            &blocks,
            &inputs,
            &mut outputs,
            1,
            33,
            1,
            None,
            Fp8Variant::E4M3,
        );
        match err {
            Err(MetalGraphError::EncodingFailed(msg)) => {
                assert!(
                    msg.contains("must be a non-zero multiple of 32"),
                    "msg = {msg}"
                );
            }
            other => panic!("expected EncodingFailed, got {other:?}"),
        }
    }
}
