//! Low-level kernel launchers for the CUDA prefill path.
//!
//! Every fn in this module is a thin wrapper around `cudarc`'s
//! `launch_builder()` that:
//!  1. Builds a `LaunchConfig` with the standard 256-thread block and
//!     `ceil(n_rows / 8)` grid sweep (matches the Q1/TQ2 V7 GEMM kernel
//!     warp-per-row layout).
//!  2. Binds the kernel arguments in the order the C kernel signature expects.
//!  3. Maps any driver error to [`CudaGraphError::DriverError`].
//!
//! These are deliberately `pub(super)` — only the prefill layer encoders
//! (`encode_q1`, `encode_ternary`) and `try_apis` call them.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::gpu_backend::cuda_graph::{CudaGraph, CudaGraphError};

use super::state::CudaPrefillModules;

/// Launch `gemm_q1_g128_v7` (batch GEMM, accumulate into outputs with `+=`).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_v7(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_v7)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_v7 launch: {e}")))
}

/// Launch `gemm_q1_g128_v7_residual` (batch GEMM + fused residual overwrite).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_v7_residual(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_v7_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_v7_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q1` (batch fused gate+up+SwiGLU GEMM).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_gemm(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm launch: {e}")))
}

/// Launch `batched_rmsnorm_v2` (one block per batch token).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_batched_rmsnorm(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_input: &CudaSlice<f32>,
    d_weight: &CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    n: u32,
    batch_size: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (batch_size, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_rmsnorm)
        .arg(d_input)
        .arg(d_weight)
        .arg(d_output)
        .arg(&n)
        .arg(&batch_size)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm launch: {e}")))
}

// =============================================================================
// TQ2 prefill kernel launchers
// =============================================================================

/// Launch `gemm_tq2_g128_v7` (batch TQ2 GEMM, accumulate into outputs with `+=`).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_tq2_v7(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_soa_raw: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_tq2_v7)
        .arg(d_soa_raw)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_tq2_v7 launch: {e}")))
}

/// Launch `gemm_tq2_g128_v7_residual` (batch TQ2 GEMM + fused residual overwrite).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_tq2_v7_residual(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_soa_raw: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_tq2_v7_residual)
        .arg(d_soa_raw)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_tq2_v7_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_tq2` (batch fused TQ2 gate+up+SwiGLU GEMM).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_gemm_tq2(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_soa_raw: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_tq2)
        .arg(d_soa_raw)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_tq2 launch: {e}"))
        })
}
