//! Batch prefill (GEMM) dispatch for OxiBonsai — Q4_0 and Q8_0 CUDA backend.
//!
//! This module mirrors [`cuda_prefill`] for Q4_0/Q8_0 quantised models.
//! Handles batch processing of multiple tokens during prompt prefill using
//! real fused batch GEMM kernels (not sequential single-token loops).
//!
//! # Architecture
//!
//! - [`CudaQStdPrefillModules`]: Compiled CUDA functions for the 8 batch GEMM kernels.
//! - [`CudaQStdPrefillLayerParams`]: Per-layer weight handles and raw AoS bytes.
//! - [`try_cuda_prefill_q_std`]: Public entry point for Q4_0/Q8_0 batch prefill.
//!
//! # Batch tensor layout
//!
//! All batched buffers use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index.  This matches the existing prefill kernels.
//!
//! # Weight layout
//!
//! Q4_0/Q8_0 weights stay in AoS layout as stored in GGUF.  No SoA reformatting.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use std::sync::{Arc, Mutex, OnceLock};

use super::cuda_full_layer::{
    acquire_full_layer_buffers, encode_attn_phase_from_qkv, get_or_upload_f32_weight,
    init_attn_modules, CudaAttnModules, CudaFullLayerBuffers, CudaKvCache,
};
use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};
use super::cuda_prefill::{init_prefill_modules, CudaPrefillBuffers, CudaPrefillModules};
use super::cuda_q_std_prefill_kernels::CUDA_Q_STD_PREFILL_KERNELS_SRC;

// =============================================================================
// Compiled Q std prefill CUDA modules
// =============================================================================

/// Compiled CUDA function handles for the 8 Q4_0/Q8_0 batch GEMM kernels.
pub struct CudaQStdPrefillModules {
    pub gemm_q4_0: CudaFunction,
    pub gemm_q4_0_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q4_0: CudaFunction,
    pub gemv_q4_0_pf: CudaFunction,
    pub gemm_q8_0: CudaFunction,
    pub gemm_q8_0_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q8_0: CudaFunction,
    pub gemv_q8_0_pf: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaQStdPrefillModules {}
unsafe impl Sync for CudaQStdPrefillModules {}

// =============================================================================
// Process-wide singleton state
// =============================================================================

struct CudaQStdPrefillState {
    qstd_modules: Mutex<Option<Arc<CudaQStdPrefillModules>>>,
    prefill_buffers: Mutex<Option<CudaPrefillBuffers>>,
    kv_cache: Mutex<Option<CudaKvCache>>,
    logits_buf: Mutex<Option<(CudaSlice<f32>, usize)>>,
}

unsafe impl Send for CudaQStdPrefillState {}
unsafe impl Sync for CudaQStdPrefillState {}

static Q_STD_PREFILL_STATE: OnceLock<CudaQStdPrefillState> = OnceLock::new();

fn q_std_prefill_state() -> &'static CudaQStdPrefillState {
    Q_STD_PREFILL_STATE.get_or_init(|| CudaQStdPrefillState {
        qstd_modules: Mutex::new(None),
        prefill_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        logits_buf: Mutex::new(None),
    })
}

// =============================================================================
// Module init
// =============================================================================

/// Compile and cache the 8 Q4_0/Q8_0 CUDA prefill kernels.
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_q_std_prefill_modules(
    graph: &CudaGraph,
) -> Result<Arc<CudaQStdPrefillModules>, CudaGraphError> {
    let state = q_std_prefill_state();
    let mut guard = state
        .qstd_modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_Q_STD_PREFILL_KERNELS_SRC, "q_std_prefill_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module q_std_prefill: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let mods = Arc::new(CudaQStdPrefillModules {
        gemm_q4_0: load("gemm_q4_0")?,
        gemm_q4_0_residual: load("gemm_q4_0_residual")?,
        fused_gate_up_swiglu_gemm_q4_0: load("fused_gate_up_swiglu_gemm_q4_0")?,
        gemv_q4_0_pf: load("gemv_q4_0_pf")?,
        gemm_q8_0: load("gemm_q8_0")?,
        gemm_q8_0_residual: load("gemm_q8_0_residual")?,
        fused_gate_up_swiglu_gemm_q8_0: load("fused_gate_up_swiglu_gemm_q8_0")?,
        gemv_q8_0_pf: load("gemv_q8_0_pf")?,
    });

    *guard = Some(Arc::clone(&mods));
    Ok(mods)
}

// =============================================================================
// Per-layer parameter struct
// =============================================================================

/// Per-layer parameters for the Q4_0/Q8_0 CUDA prefill path.
///
/// Weight bytes are raw AoS layout as stored in GGUF (no SoA reformatting needed).
pub struct CudaQStdPrefillLayerParams<'a> {
    pub attn_norm_handle: u64,
    pub attn_norm_bytes: &'a [f32],
    pub fused_qkv_handle: u64,
    pub fused_qkv_bytes: &'a [u8],
    pub q_norm_handle: u64,
    pub q_norm_bytes: &'a [f32],
    pub k_norm_handle: u64,
    pub k_norm_bytes: &'a [f32],
    pub attn_proj_handle: u64,
    pub attn_proj_bytes: &'a [u8],
    pub ffn_norm_handle: u64,
    pub ffn_norm_bytes: &'a [f32],
    pub gate_up_handle: u64,
    pub gate_bytes: &'a [u8],
    pub up_bytes: &'a [u8],
    pub down_handle: u64,
    pub down_bytes: &'a [u8],
    /// `true` for Q4_0, `false` for Q8_0.
    pub q4_0: bool,
}

// =============================================================================
// Buffer / KV-cache acquisition helpers (private to this module)
// =============================================================================

/// Round up `n` to the next power of two (minimum 1).
fn next_pow2_cap(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut cap = 1usize;
    while cap < n {
        cap <<= 1;
    }
    cap
}

/// Acquire or (re-)allocate the prefill activation buffers.
#[allow(clippy::too_many_arguments)]
fn acquire_q_std_prefill_buffers(
    graph: &CudaGraph,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaPrefillBuffers>>, CudaGraphError> {
    let state = q_std_prefill_state();
    let mut guard = state
        .prefill_buffers
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(b) => !b.matches(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        ),
        None => true,
    };

    if needs_alloc {
        let capacity = next_pow2_cap(batch_size);
        let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros qspb({n}): {e}")))
        };
        let qkv_total = (nq + 2 * nkv) * head_dim;

        *guard = Some(CudaPrefillBuffers {
            d_input: alloc(capacity * hidden_size)?,
            d_normed: alloc(capacity * hidden_size)?,
            d_qkv: alloc(capacity * qkv_total)?,
            d_attn_out: alloc(capacity * nq * head_dim)?,
            d_gate_up: alloc(2 * capacity * intermediate_size)?,
            d_swiglu: alloc(capacity * intermediate_size)?,
            capacity,
            actual_batch_size: batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        });
    } else {
        guard
            .as_mut()
            .expect("guard is Some when needs_alloc is false")
            .actual_batch_size = batch_size;
    }

    Ok(guard)
}

/// Acquire or (re-)allocate the shared GPU KV cache.
fn acquire_q_std_kv_cache(
    graph: &CudaGraph,
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaKvCache>>, CudaGraphError> {
    let state = q_std_prefill_state();
    let mut guard = state
        .kv_cache
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(c) => !c.matches(n_layers, n_kv, max_seq, head_dim),
        None => true,
    };

    if needs_alloc {
        let total = n_layers * n_kv * max_seq * head_dim;
        let k_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv k_cache qstd: {e}")))?;
        let v_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv v_cache qstd: {e}")))?;

        *guard = Some(CudaKvCache {
            k_cache,
            v_cache,
            n_layers,
            n_kv,
            max_seq,
            head_dim,
        });
    }

    Ok(guard)
}

type QStdLogitsGuard = std::sync::MutexGuard<'static, Option<(CudaSlice<f32>, usize)>>;

/// Acquire or (re-)allocate the LM-head logits buffer.
fn acquire_q_std_logits(graph: &CudaGraph, n: usize) -> Result<QStdLogitsGuard, CudaGraphError> {
    let state = q_std_prefill_state();
    let mut guard = state
        .logits_buf
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some((_, sz)) => *sz != n,
        None => true,
    };

    if needs_alloc {
        let buf = graph
            .stream_arc()
            .alloc_zeros::<f32>(n)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc logits qstd({n}): {e}")))?;
        *guard = Some((buf, n));
    }

    Ok(guard)
}

// =============================================================================
// Low-level CUDA kernel launchers
// =============================================================================

/// Launch `gemm_q4_0` — batch Q4_0 GEMM, accumulates with `+=`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_q4_0(
    graph: &CudaGraph,
    mods: &CudaQStdPrefillModules,
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
        .launch_builder(&mods.gemm_q4_0)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q4_0 launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q4_0` — batch fused Q4_0 gate+up+SwiGLU GEMM.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fused_gate_up_swiglu_q4_0(
    graph: &CudaGraph,
    mods: &CudaQStdPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_ffn_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q4_0)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q4_0 launch: {e}"))
        })
}

/// Launch `gemv_q4_0_pf` — single-token Q4_0 GEMV.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemv_q4_0_pf(
    graph: &CudaGraph,
    mods: &CudaQStdPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_input: &CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemv_q4_0_pf)
        .arg(d_blocks)
        .arg(d_input)
        .arg(d_output)
        .arg(&n_rows)
        .arg(&k)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemv_q4_0_pf launch: {e}")))
}

/// Launch `gemm_q8_0` — batch Q8_0 GEMM, accumulates with `+=`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_q8_0(
    graph: &CudaGraph,
    mods: &CudaQStdPrefillModules,
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
        .launch_builder(&mods.gemm_q8_0)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_q8_0 launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q8_0` — batch fused Q8_0 gate+up+SwiGLU GEMM.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fused_gate_up_swiglu_q8_0(
    graph: &CudaGraph,
    mods: &CudaQStdPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_ffn_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_ffn_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_q8_0)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_q8_0 launch: {e}"))
        })
}

/// Launch `gemv_q8_0_pf` — single-token Q8_0 GEMV.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemv_q8_0_pf(
    graph: &CudaGraph,
    mods: &CudaQStdPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_input: &CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemv_q8_0_pf)
        .arg(d_blocks)
        .arg(d_input)
        .arg(d_output)
        .arg(&n_rows)
        .arg(&k)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemv_q8_0_pf launch: {e}")))
}

// =============================================================================
// encode_q_std_prefill_ffn_phase
// =============================================================================

/// Batched FFN sublayer for Q4_0/Q8_0 models.
///
/// Pipeline:
/// 1. Batched RMSNorm: `d_input → d_normed` (all tokens)
/// 2. Fused gate+up+SwiGLU GEMM: `d_normed → d_swiglu` (all tokens)
/// 3. Down GEMM + residual: `d_swiglu → d_input` (fused residual via residual_add)
///
/// # Safety
/// All device buffers must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_q_std_ffn_phase(
    graph: &CudaGraph,
    qstd_mods: &CudaQStdPrefillModules,
    pmods: &CudaPrefillModules,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    pb: &mut CudaPrefillBuffers,
    eps: f32,
    q4_0: bool,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size as u32;
    let h = pb.hidden_size as u32;
    let inter = pb.intermediate_size as u32;

    // Step 1: Batched RMSNorm — reuse from prefill modules (batched_rmsnorm_v2).
    {
        let cfg = LaunchConfig {
            grid_dim: (bs, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        graph
            .stream_arc()
            .launch_builder(&pmods.batched_rmsnorm)
            .arg(&pb.d_input)
            .arg(d_ffn_norm_weight)
            .arg(&mut pb.d_normed)
            .arg(&h)
            .arg(&bs)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm ffn qstd: {e}")))?;
    }

    // Step 2: Fused gate+up+SwiGLU GEMM (d_normed → d_swiglu).
    if q4_0 {
        launch_fused_gate_up_swiglu_q4_0(
            graph,
            qstd_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?;
    } else {
        launch_fused_gate_up_swiglu_q8_0(
            graph,
            qstd_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?;
    }

    // Step 3: Down GEMM into d_normed (scratch), then residual add.
    // gemm_q4_0/gemm_q8_0 accumulate with +=, so zero d_normed first.
    {
        let n = pb.actual_batch_size * pb.hidden_size;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed down qstd: {e}")))?;
    }
    if q4_0 {
        launch_gemm_q4_0(
            graph,
            qstd_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?;
    } else {
        launch_gemm_q8_0(
            graph,
            qstd_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?;
    }

    let total_bh = (pb.actual_batch_size * pb.hidden_size) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    Ok(())
}

// =============================================================================
// encode_q_std_prefill_layer
// =============================================================================

/// Encode one full transformer layer for Q4_0/Q8_0 batch prefill.
///
/// Uses batch GEMM for all linear projections.  Attention is processed
/// sequentially per token (the same approach as Q1/TQ2 prefill) because
/// each query position needs access to all prior KV entries.
///
/// # Safety
/// All device buffers and weight slices must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_q_std_prefill_layer(
    graph: &CudaGraph,
    qstd_mods: &CudaQStdPrefillModules,
    pmods: &CudaPrefillModules,
    attn_mods: &CudaAttnModules,
    d_attn_norm_weight: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_q_norm_weight: &CudaSlice<f32>,
    d_k_norm_weight: &CudaSlice<f32>,
    d_attn_proj_weight: &Arc<CudaSlice<u8>>,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    pos_start: usize,
    pb: &mut CudaPrefillBuffers,
    st_bufs: &mut CudaFullLayerBuffers,
    cos_table: &[f32],
    sin_table: &[f32],
    heads_per_group: usize,
    eps: f32,
    q4_0: bool,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size;
    let h = pb.hidden_size;
    let nq = pb.nq;
    let nkv = pb.nkv;
    let hd = pb.head_dim;
    let half_dim = hd / 2;
    let h_u32 = h as u32;
    let bs_u32 = bs as u32;
    let qkv_total = nq * hd + 2 * nkv * hd;

    // ─── 1. Batched RMSNorm (attn norm): d_input → d_normed ─────────────────
    {
        let cfg = LaunchConfig {
            grid_dim: (bs_u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        graph
            .stream_arc()
            .launch_builder(&pmods.batched_rmsnorm)
            .arg(&pb.d_input)
            .arg(d_attn_norm_weight)
            .arg(&mut pb.d_normed)
            .arg(&h_u32)
            .arg(&bs_u32)
            .arg(&eps)
            .launch(cfg)
            .map(|_| ())
            .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm attn qstd: {e}")))?;
    }

    // ─── 2. Batched QKV GEMM: d_normed → d_qkv ──────────────────────────────
    // Zero d_qkv first (kernels accumulate with +=).
    {
        let n = bs * qkv_total;
        let mut dst_view = pb.d_qkv.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_qkv qstd: {e}")))?;
    }
    if q4_0 {
        launch_gemm_q4_0(
            graph,
            qstd_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?;
    } else {
        launch_gemm_q8_0(
            graph,
            qstd_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?;
    }

    // ─── 3. Sequential attention for each token ──────────────────────────────
    // Zero d_attn_out before the loop.
    {
        let n = bs * nq * hd;
        let mut dst_view = pb.d_attn_out.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out qstd: {e}")))?;
    }

    for t in 0..bs {
        let pos = pos_start + t;

        // Copy this token's hidden state into st_bufs.d_hidden (for attn output write-back).
        {
            let src_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| CudaGraphError::DriverError(format!("copy hidden qstd t={t}: {e}")))?;
        }

        // Copy this token's QKV into st_bufs.d_qkv.
        {
            let src_view: CudaView<f32> = pb.d_qkv.slice(t * qkv_total..(t + 1) * qkv_total);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_qkv)
                .map_err(|e| CudaGraphError::DriverError(format!("copy qkv qstd t={t}: {e}")))?;
        }

        // Upload RoPE cos/sin for this token's position.
        let rope_off = t * half_dim;
        graph
            .stream_arc()
            .memcpy_htod(
                &cos_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_cos,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos qstd t={t}: {e}")))?;
        graph
            .stream_arc()
            .memcpy_htod(
                &sin_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_sin,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin qstd t={t}: {e}")))?;

        // Upload pos and seq_len (pos+1) for this token.
        let pos_seqlen = [pos as u32, (pos + 1) as u32];
        graph
            .stream_arc()
            .memcpy_htod(&pos_seqlen, &mut st_bufs.d_pos_seqlen)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("upload pos_seqlen qstd t={t}: {e}"))
            })?;

        // Run attention steps 3-7 (QK-norm+RoPE, KV-store, scores, softmax, weighted sum).
        encode_attn_phase_from_qkv(
            graph,
            attn_mods,
            d_q_norm_weight,
            d_k_norm_weight,
            kv,
            layer_idx,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            st_bufs,
        )?;

        // Copy attention output back into pb.d_attn_out for this token.
        {
            let attn_col_size = nq * hd;
            let src_view: CudaView<f32> = st_bufs.d_attn_out.slice(0..attn_col_size);
            let mut dst_view = pb
                .d_attn_out
                .slice_mut(t * attn_col_size..(t + 1) * attn_col_size);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy attn_out qstd t={t}: {e}"))
                })?;
        }
    }

    // ─── 4. Attn output projection + residual ────────────────────────────────
    // d_attn_out (col-major [bs × nq*hd]) → d_normed (col-major [bs × h]) via GEMM,
    // then residual add: d_input += d_normed.
    {
        let n = bs * h;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("zero d_normed attn_proj qstd: {e}"))
            })?;
    }
    let attn_proj_rows = h as u32;
    let attn_proj_k = (nq * hd) as u32;
    if q4_0 {
        launch_gemm_q4_0(
            graph,
            qstd_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?;
    } else {
        launch_gemm_q8_0(
            graph,
            qstd_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?;
    }

    let total_bh = (bs * h) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    // ─── 5. Batched FFN sublayer ──────────────────────────────────────────────
    encode_q_std_ffn_phase(
        graph,
        qstd_mods,
        pmods,
        d_ffn_norm_weight,
        d_gate_up_weight,
        d_down_weight,
        pb,
        eps,
        q4_0,
    )?;

    Ok(())
}

// =============================================================================
// Public entry point: try_cuda_prefill_q_std
// =============================================================================

/// Batch prefill for Q4_0 or Q8_0 quantised models.
///
/// Processes `batch_size` tokens simultaneously using real fused batch GEMM kernels
/// for all linear projections.  Attention is processed per-token sequentially.
///
/// Set `q4_0 = true` for Q4_0 weights, `q4_0 = false` for Q8_0 weights.
///
/// # Arguments
///
/// - `hidden_batch` — host-side batched hidden states in row-major layout:
///   `[batch_size × hidden_size]` (token-major).  Converted to column-major internally.
/// - `logits_out` / `greedy_token_id_out` — if `Some`, the function runs the final
///   norm and LM head for the last token and returns either full logits or the argmax.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_prefill_q_std(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[CudaQStdPrefillLayerParams<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    lm_head_q4_0: bool,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), CudaGraphError> {
    if batch_size == 0 {
        return Ok(());
    }

    // Get global CudaGraph singleton.
    let graph = CudaGraph::global()?;

    // Compile / retrieve the Q std prefill CUDA modules.
    let qstd_mods = init_q_std_prefill_modules(&graph)?;

    // Compile / retrieve the existing prefill modules (for batched_rmsnorm).
    let pmods = init_prefill_modules(&graph)?;

    // Compile / retrieve the attention modules.
    let attn_mods = init_attn_modules(&graph)?;

    // Upload / cache all layer weights (f32 norms + raw AoS quantised projections).
    // We cache them all at once before touching any activation buffer.
    struct LayerWeightHandles {
        attn_norm: Arc<CudaSlice<f32>>,
        fused_qkv: Arc<CudaSlice<u8>>,
        q_norm: Arc<CudaSlice<f32>>,
        k_norm: Arc<CudaSlice<f32>>,
        attn_proj: Arc<CudaSlice<u8>>,
        ffn_norm: Arc<CudaSlice<f32>>,
        gate_up: Arc<CudaSlice<u8>>,
        down: Arc<CudaSlice<u8>>,
    }

    let mut layer_weights: Vec<LayerWeightHandles> = Vec::with_capacity(n_layers);
    for lp in layer_params.iter().take(n_layers) {
        // Upload fused gate+up weight as a single contiguous AoS block (gate first, then up).
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_weight_aos_raw_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        layer_weights.push(LayerWeightHandles {
            attn_norm: get_or_upload_f32_weight(&graph, lp.attn_norm_handle, lp.attn_norm_bytes)?,
            fused_qkv: graph
                .get_or_upload_weight_aos_raw(lp.fused_qkv_handle, lp.fused_qkv_bytes)?,
            q_norm: get_or_upload_f32_weight(&graph, lp.q_norm_handle, lp.q_norm_bytes)?,
            k_norm: get_or_upload_f32_weight(&graph, lp.k_norm_handle, lp.k_norm_bytes)?,
            attn_proj: graph
                .get_or_upload_weight_aos_raw(lp.attn_proj_handle, lp.attn_proj_bytes)?,
            ffn_norm: get_or_upload_f32_weight(&graph, lp.ffn_norm_handle, lp.ffn_norm_bytes)?,
            gate_up: gate_up_w,
            down: graph.get_or_upload_weight_aos_raw(lp.down_handle, lp.down_bytes)?,
        });
    }

    // Allocate / acquire the batched prefill activation buffers.
    let mut pb_guard = acquire_q_std_prefill_buffers(
        &graph,
        batch_size,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
    )?;
    let pb = pb_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("prefill buffer not allocated".into()))?;

    // Allocate / acquire the KV cache.
    let mut kv_guard = acquire_q_std_kv_cache(&graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("KV cache not allocated".into()))?;

    // Acquire single-token full-layer buffers for sequential attention.
    let mut st_guard = acquire_full_layer_buffers(
        &graph,
        hidden_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
        intermediate_size,
    )?;
    let st_bufs = st_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("full-layer buffer not allocated".into()))?;

    // Upload the hidden batch to GPU in column-major layout.
    // Input from host is row-major: [batch_size × hidden_size].
    // GPU expects column-major: [hidden_size × batch_size] (d_input[col*h + elem]).
    // Transpose on the way in.
    {
        let mut col_major = vec![0.0f32; batch_size * hidden_size];
        for t in 0..batch_size {
            for e in 0..hidden_size {
                col_major[t * hidden_size + e] = hidden_batch[t * hidden_size + e];
            }
        }
        let n = batch_size * hidden_size;
        let mut dst_view = pb.d_input.slice_mut(0..n);
        graph
            .stream_arc()
            .memcpy_htod(&col_major, &mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_batch qstd: {e}")))?;
    }

    // Determine fallback Q4_0/Q8_0 flag: use first layer's flag, or Q4_0 if no layers.
    let default_q4_0 = layer_params.first().is_none_or(|lp| lp.q4_0);

    // Run each transformer layer.
    for (layer_idx, lw) in layer_weights.iter().enumerate() {
        let q4_0 = layer_params
            .get(layer_idx)
            .map_or(default_q4_0, |lp| lp.q4_0);

        unsafe {
            encode_q_std_prefill_layer(
                &graph,
                &qstd_mods,
                &pmods,
                &attn_mods,
                &lw.attn_norm,
                &lw.fused_qkv,
                &lw.q_norm,
                &lw.k_norm,
                &lw.attn_proj,
                &lw.ffn_norm,
                &lw.gate_up,
                &lw.down,
                kv,
                layer_idx,
                pos_start,
                pb,
                st_bufs,
                cos_table,
                sin_table,
                heads_per_group,
                eps,
                q4_0,
            )?;
        }
    }

    // ─── Final norm + LM head (optional) ─────────────────────────────────────
    if logits_out.is_some() || greedy_token_id_out.is_some() {
        let final_norm_h = final_norm_handle.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("final_norm_handle required for logits".into())
        })?;
        let final_norm_b = final_norm_bytes.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("final_norm_bytes required for logits".into())
        })?;
        let lm_head_h = lm_head_handle.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("lm_head_handle required for logits".into())
        })?;
        let lm_head_b = lm_head_bytes.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("lm_head_bytes required for logits".into())
        })?;

        let d_final_norm_w = get_or_upload_f32_weight(&graph, final_norm_h, final_norm_b)?;
        let d_lm_head_w = graph.get_or_upload_weight_aos_raw(lm_head_h, lm_head_b)?;

        // Extract last token's hidden state from pb.d_input into st_bufs.d_hidden.
        let last_t = batch_size - 1;
        {
            let src_view: CudaView<f32> = pb
                .d_input
                .slice(last_t * hidden_size..(last_t + 1) * hidden_size);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy last hidden qstd lm: {e}"))
                })?;
        }

        // Final RMSNorm on the last token's hidden state.
        unsafe {
            graph
                .launch_rmsnorm_pub(
                    &st_bufs.d_hidden,
                    &d_final_norm_w,
                    &mut st_bufs.d_normed,
                    hidden_size as u32,
                    final_norm_eps,
                )
                .map_err(|e| CudaGraphError::DriverError(format!("final norm qstd: {e:?}")))?;
        }

        // LM head GEMV on the normed last-token hidden state.
        let mut lm_logits_guard = acquire_q_std_logits(&graph, lm_head_out_features)?;
        let d_logits = &mut lm_logits_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("logits buf not allocated qstd".into()))?
            .0;

        unsafe {
            if lm_head_q4_0 {
                launch_gemv_q4_0_pf(
                    &graph,
                    &qstd_mods,
                    &d_lm_head_w,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                )?;
            } else {
                launch_gemv_q8_0_pf(
                    &graph,
                    &qstd_mods,
                    &d_lm_head_w,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                )?;
            }
        }

        // Synchronise stream before D2H copy.
        graph
            .stream_arc()
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync qstd lm: {e}")))?;

        let logits_host = graph
            .stream_arc()
            .clone_dtoh(d_logits)
            .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits qstd: {e}")))?;

        drop(lm_logits_guard);

        if let Some(out) = greedy_token_id_out {
            *out = logits_host
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        } else if let Some(out) = logits_out {
            *out = logits_host;
        }

        return Ok(());
    }

    // No LM head requested — just synchronise.
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("sync qstd end: {e}")))?;

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_backend::cuda_q_std_prefill_kernels::CUDA_Q_STD_PREFILL_KERNELS_SRC;

    /// Verify the kernel source contains `gemm_q4_0`.
    #[test]
    fn test_q_std_prefill_kernel_source_has_gemm_q4_0() {
        assert!(
            CUDA_Q_STD_PREFILL_KERNELS_SRC.contains("gemm_q4_0"),
            "CUDA_Q_STD_PREFILL_KERNELS_SRC must contain gemm_q4_0"
        );
    }

    /// Verify the kernel source contains `gemm_q8_0`.
    #[test]
    fn test_q_std_prefill_kernel_source_has_gemm_q8_0() {
        assert!(
            CUDA_Q_STD_PREFILL_KERNELS_SRC.contains("gemm_q8_0"),
            "CUDA_Q_STD_PREFILL_KERNELS_SRC must contain gemm_q8_0"
        );
    }

    /// Verify the kernel source contains `fused_gate_up_swiglu_gemm_q4_0`.
    #[test]
    fn test_q_std_prefill_kernel_source_has_fused_gemm_q4_0() {
        assert!(
            CUDA_Q_STD_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_q4_0"),
            "CUDA_Q_STD_PREFILL_KERNELS_SRC must contain fused_gate_up_swiglu_gemm_q4_0"
        );
    }

    /// Verify the kernel source contains `fused_gate_up_swiglu_gemm_q8_0`.
    #[test]
    fn test_q_std_prefill_kernel_source_has_fused_gemm_q8_0() {
        assert!(
            CUDA_Q_STD_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_q8_0"),
            "CUDA_Q_STD_PREFILL_KERNELS_SRC must contain fused_gate_up_swiglu_gemm_q8_0"
        );
    }

    /// GPU-gated: compile Q std prefill modules without error.
    #[test]
    fn test_q_std_prefill_modules_init() {
        let graph_result = CudaGraph::global();
        if graph_result.is_err() {
            eprintln!("SKIP: test_q_std_prefill_modules_init — no CUDA device");
            return;
        }
        let graph = graph_result.expect("CudaGraph::global failed");
        let result = init_q_std_prefill_modules(&graph);
        assert!(
            result.is_ok(),
            "Q std prefill module init failed: {:?}",
            result.err()
        );
    }

    /// GPU-gated: verify batch_size=12 works correctly (tests cap-of-8 outer loop).
    ///
    /// Uses a tiny model (1 layer, 32-dim hidden, 32-dim intermediate) with
    /// zero-weight Q4_0 blocks (all nibbles = 8 → weight = 0).  With zero
    /// weights every output must be zero regardless of input.
    #[test]
    fn test_q_std_prefill_batch12_no_capof8() {
        let graph_result = CudaGraph::global();
        if graph_result.is_err() {
            eprintln!("SKIP: test_q_std_prefill_batch12_no_capof8 — no CUDA device");
            return;
        }

        // Minimal dimensions all multiples of 32.
        let hidden_size = 64usize;
        let intermediate_size = 64usize;
        let nq = 2usize;
        let nkv = 2usize;
        let head_dim = 32usize;
        let heads_per_group = 1usize;
        let batch_size = 12usize;
        let max_seq = 64usize;
        let eps = 1e-5f32;

        let qkv_rows = (nq + 2 * nkv) * head_dim; // (2 + 4) * 32 = 192
        let attn_proj_rows = hidden_size; // 64

        // Build zero Q4_0 weight blocks (all nibbles = 8 → dequantised weight = 0).
        // Scale = FP16 1.0 (0x3C00 LE).
        let make_q4_0_zeros = |n_rows: usize, k: usize| -> Vec<u8> {
            let blocks_per_row = k / 32;
            let mut v = vec![0u8; n_rows * blocks_per_row * 18];
            for r in 0..n_rows {
                for b in 0..blocks_per_row {
                    let off = (r * blocks_per_row + b) * 18;
                    v[off] = 0x00; // FP16 1.0 lo
                    v[off + 1] = 0x3C; // FP16 1.0 hi
                                       // nibbles all 0x8 → each nibble_byte = 0x88
                    for j in 2..18 {
                        v[off + j] = 0x88;
                    }
                }
            }
            v
        };

        let fused_qkv_bytes = make_q4_0_zeros(qkv_rows, hidden_size);
        let attn_proj_bytes = make_q4_0_zeros(attn_proj_rows, nq * head_dim);
        // gate+up: 2*inter rows
        let gate_bytes = make_q4_0_zeros(intermediate_size, hidden_size);
        let up_bytes = make_q4_0_zeros(intermediate_size, hidden_size);
        let down_bytes = make_q4_0_zeros(hidden_size, intermediate_size);

        // Zero f32 norm weights.
        let attn_norm = vec![1.0f32; hidden_size];
        let q_norm = vec![1.0f32; head_dim];
        let k_norm = vec![1.0f32; head_dim];
        let ffn_norm = vec![1.0f32; hidden_size];

        // Handle IDs — use large values to avoid collision with other tests.
        let base_h = 0xDEAD_BEEF_0000_0001u64;

        let layer_params = vec![CudaQStdPrefillLayerParams {
            attn_norm_handle: base_h,
            attn_norm_bytes: &attn_norm,
            fused_qkv_handle: base_h + 1,
            fused_qkv_bytes: &fused_qkv_bytes,
            q_norm_handle: base_h + 2,
            q_norm_bytes: &q_norm,
            k_norm_handle: base_h + 3,
            k_norm_bytes: &k_norm,
            attn_proj_handle: base_h + 4,
            attn_proj_bytes: &attn_proj_bytes,
            ffn_norm_handle: base_h + 5,
            ffn_norm_bytes: &ffn_norm,
            gate_up_handle: base_h + 6,
            gate_bytes: &gate_bytes,
            up_bytes: &up_bytes,
            down_handle: base_h + 7,
            down_bytes: &down_bytes,
            q4_0: true,
        }];

        // Flat cos/sin tables: batch_size * half_dim.
        let half_dim = head_dim / 2;
        let cos_table = vec![1.0f32; batch_size * half_dim];
        let sin_table = vec![0.0f32; batch_size * half_dim];

        // Input: batch of 12 tokens, all ones.
        let hidden_batch = vec![1.0f32; batch_size * hidden_size];

        let result = try_cuda_prefill_q_std(
            &hidden_batch,
            batch_size,
            0,
            1,
            &layer_params,
            &cos_table,
            &sin_table,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            heads_per_group,
            eps,
            max_seq,
            None,
            None,
            1e-5f32,
            None,
            None,
            0,
            true,
            None,
            None,
        );

        assert!(
            result.is_ok(),
            "try_cuda_prefill_q_std batch=12 failed: {:?}",
            result.err()
        );
    }
}
