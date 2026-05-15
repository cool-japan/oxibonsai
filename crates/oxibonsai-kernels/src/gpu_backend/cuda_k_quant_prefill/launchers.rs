//! Low-level kernel launchers for the K-quant CUDA prefill path.
//!
//! Three launchers per K-quant format × 6 formats = 18 launchers total:
//!   - `launch_gemm_{q2k,q3k,q4k,q5k,q6k,q8k}` — batch GEMM with `+=` accumulate.
//!   - `launch_gemm_{...}_residual` — batch GEMM with fused residual add.
//!   - `launch_fused_gate_up_swiglu_{...}` — fused gate+up+SwiGLU GEMM.
//!
//! All launchers use the standard 256-thread block + `ceil(n_rows / 8)` grid
//! sweep (warp-per-row) and map any driver error to
//! [`CudaGraphError::DriverError`].

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::gpu_backend::cuda_graph::{CudaGraph, CudaGraphError};

use super::state::CudaKQuantPrefillModules;

/// Launch `gemm_q2k` — batch Q2_K GEMM, accumulates with `+=`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_q2k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q2k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q2k launch: {e}")))
}

/// Launch `gemm_q2k_residual` — Q2_K GEMM with fused residual add.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_q2k_residual(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q2k_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q2k_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q2k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_q2k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_ffn_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q2k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q2k launch: {e}"))
        })
}

/// Launch `gemm_q3k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_q3k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q3k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q3k launch: {e}")))
}

/// Launch `gemm_q3k_residual`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_q3k_residual(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q3k_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q3k_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q3k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_q3k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_ffn_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q3k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q3k launch: {e}"))
        })
}

/// Launch `gemm_q4k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_q4k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q4k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q4k launch: {e}")))
}

/// Launch `gemm_q4k_residual`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_q4k_residual(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q4k_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q4k_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q4k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_q4k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_ffn_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q4k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q4k launch: {e}"))
        })
}

/// Launch `gemm_q5k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_q5k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q5k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q5k launch: {e}")))
}

/// Launch `gemm_q5k_residual`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_q5k_residual(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q5k_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q5k_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q5k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_q5k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_ffn_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q5k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q5k launch: {e}"))
        })
}

/// Launch `gemm_q6k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_q6k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q6k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q6k launch: {e}")))
}

/// Launch `gemm_q6k_residual`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_q6k_residual(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q6k_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q6k_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q6k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_q6k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_ffn_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q6k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q6k launch: {e}"))
        })
}

/// Launch `gemm_q8k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_q8k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q8k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q8k launch: {e}")))
}

/// Launch `gemm_q8k_residual`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_gemm_q8k_residual(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_q8k_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q8k_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q8k`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_gate_up_swiglu_q8k(
    graph: &CudaGraph,
    mods: &CudaKQuantPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_ffn_rows.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q8k)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q8k launch: {e}"))
        })
}
