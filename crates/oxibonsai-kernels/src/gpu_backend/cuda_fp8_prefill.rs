//! Batch prefill (GEMM) dispatch for OxiBonsai — FP8 E4M3/E5M2 CUDA backend.
//!
//! This module mirrors [`cuda_q_std_prefill`] for FP8 E4M3/E5M2 quantised models.
//! Handles batch processing of multiple tokens during prompt prefill using
//! real fused batch GEMM kernels (not sequential single-token loops).
//!
//! # Architecture
//!
//! - [`CudaFP8PrefillModules`]: Compiled CUDA functions for the 8 batch GEMM kernels.
//! - [`CudaFP8PrefillLayerParams`]: Per-layer weight handles and raw AoS bytes.
//! - [`try_cuda_prefill_fp8`]: Public entry point for FP8 E4M3/E5M2 batch prefill.
//!
//! # Block layout (FP8 AoS, 34 bytes/block)
//!
//! ```text
//! bytes  0-31: 32 FP8 quantized weights (E4M3 or E5M2)
//! bytes 32-33: FP16 LE block scale
//! ```
//!
//! This differs from Q8_0 (scale at bytes 0-1, weights at 2-33).
//!
//! # Batch tensor layout
//!
//! All batched buffers use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index.  This matches the existing prefill kernels.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use std::sync::{Arc, Mutex, OnceLock};

use super::cuda_fp8_prefill_kernels::CUDA_FP8_PREFILL_KERNELS_SRC;
use super::cuda_full_layer::{
    acquire_full_layer_buffers, encode_attn_phase_from_qkv, get_or_upload_f32_weight,
    init_attn_modules, CudaAttnModules, CudaFullLayerBuffers, CudaKvCache,
};
use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};
use super::cuda_prefill::{init_prefill_modules, CudaPrefillBuffers, CudaPrefillModules};

// =============================================================================
// Compiled FP8 prefill CUDA modules
// =============================================================================

/// Compiled CUDA function handles for the 8 FP8 E4M3/E5M2 batch GEMM kernels.
pub struct CudaFP8PrefillModules {
    pub gemm_fp8_e4m3: CudaFunction,
    pub gemm_fp8_e4m3_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_fp8_e4m3: CudaFunction,
    pub gemv_fp8_e4m3_pf: CudaFunction,
    pub gemm_fp8_e5m2: CudaFunction,
    pub gemm_fp8_e5m2_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_fp8_e5m2: CudaFunction,
    pub gemv_fp8_e5m2_pf: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaFP8PrefillModules {}
unsafe impl Sync for CudaFP8PrefillModules {}

// =============================================================================
// Process-wide singleton state
// =============================================================================

struct CudaFP8PrefillState {
    fp8_modules: Mutex<Option<Arc<CudaFP8PrefillModules>>>,
    prefill_buffers: Mutex<Option<CudaPrefillBuffers>>,
    kv_cache: Mutex<Option<CudaKvCache>>,
    logits_buf: Mutex<Option<(CudaSlice<f32>, usize)>>,
}

unsafe impl Send for CudaFP8PrefillState {}
unsafe impl Sync for CudaFP8PrefillState {}

static FP8_PREFILL_STATE: OnceLock<CudaFP8PrefillState> = OnceLock::new();

fn fp8_prefill_state() -> &'static CudaFP8PrefillState {
    FP8_PREFILL_STATE.get_or_init(|| CudaFP8PrefillState {
        fp8_modules: Mutex::new(None),
        prefill_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        logits_buf: Mutex::new(None),
    })
}

// =============================================================================
// Module init
// =============================================================================

/// Compile and cache the 8 FP8 E4M3/E5M2 CUDA prefill kernels.
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_fp8_prefill_modules(
    graph: &CudaGraph,
) -> Result<Arc<CudaFP8PrefillModules>, CudaGraphError> {
    let state = fp8_prefill_state();
    let mut guard = state
        .fp8_modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_FP8_PREFILL_KERNELS_SRC, "fp8_prefill_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module fp8_prefill: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let mods = Arc::new(CudaFP8PrefillModules {
        gemm_fp8_e4m3: load("gemm_fp8_e4m3")?,
        gemm_fp8_e4m3_residual: load("gemm_fp8_e4m3_residual")?,
        fused_gate_up_swiglu_gemm_fp8_e4m3: load("fused_gate_up_swiglu_gemm_fp8_e4m3")?,
        gemv_fp8_e4m3_pf: load("gemv_fp8_e4m3_pf")?,
        gemm_fp8_e5m2: load("gemm_fp8_e5m2")?,
        gemm_fp8_e5m2_residual: load("gemm_fp8_e5m2_residual")?,
        fused_gate_up_swiglu_gemm_fp8_e5m2: load("fused_gate_up_swiglu_gemm_fp8_e5m2")?,
        gemv_fp8_e5m2_pf: load("gemv_fp8_e5m2_pf")?,
    });

    *guard = Some(Arc::clone(&mods));
    Ok(mods)
}

// =============================================================================
// Per-layer parameter struct
// =============================================================================

/// Per-layer parameters for the FP8 E4M3/E5M2 CUDA prefill path.
///
/// Weight bytes are raw AoS layout as stored in GGUF (no SoA reformatting needed).
/// FP8 block layout: `[q0..q31, scale_lo, scale_hi]` (34 bytes/block).
pub struct CudaFP8PrefillLayerParams<'a> {
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
    pub gate_up_bytes: &'a [u8],
    pub ffn_down_handle: u64,
    pub ffn_down_bytes: &'a [u8],
}

// =============================================================================
// Buffer / KV-cache acquisition helpers (private to this module)
// =============================================================================

/// Round up `n` to the next power of two (minimum 1).
fn next_pow2_cap_fp8(n: usize) -> usize {
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
fn acquire_fp8_prefill_buffers(
    graph: &CudaGraph,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaPrefillBuffers>>, CudaGraphError> {
    let state = fp8_prefill_state();
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
        let capacity = next_pow2_cap_fp8(batch_size);
        let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros fp8pf({n}): {e}")))
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
fn acquire_fp8_kv_cache(
    graph: &CudaGraph,
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaKvCache>>, CudaGraphError> {
    let state = fp8_prefill_state();
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
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv k_cache fp8: {e}")))?;
        let v_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv v_cache fp8: {e}")))?;

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

type Fp8LogitsGuard = std::sync::MutexGuard<'static, Option<(CudaSlice<f32>, usize)>>;

/// Acquire or (re-)allocate the LM-head logits buffer.
fn acquire_fp8_logits(graph: &CudaGraph, n: usize) -> Result<Fp8LogitsGuard, CudaGraphError> {
    let state = fp8_prefill_state();
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
            .map_err(|e| CudaGraphError::DriverError(format!("alloc logits fp8({n}): {e}")))?;
        *guard = Some((buf, n));
    }

    Ok(guard)
}

// =============================================================================
// Low-level CUDA kernel launchers
// =============================================================================

/// Launch `gemm_fp8_e4m3` — batch FP8 E4M3 GEMM, accumulates with `+=`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_fp8_e4m3(
    graph: &CudaGraph,
    mods: &CudaFP8PrefillModules,
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
        .launch_builder(&mods.gemm_fp8_e4m3)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_fp8_e4m3 launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_fp8_e4m3` — batch fused FP8 E4M3 gate+up+SwiGLU GEMM.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fused_gate_up_swiglu_fp8_e4m3(
    graph: &CudaGraph,
    mods: &CudaFP8PrefillModules,
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
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_fp8_e4m3)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_fp8_e4m3 launch: {e}"))
        })
}

/// Launch `gemv_fp8_e4m3_pf` — single-token FP8 E4M3 GEMV.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemv_fp8_e4m3_pf(
    graph: &CudaGraph,
    mods: &CudaFP8PrefillModules,
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
        .launch_builder(&mods.gemv_fp8_e4m3_pf)
        .arg(d_blocks)
        .arg(d_input)
        .arg(d_output)
        .arg(&n_rows)
        .arg(&k)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemv_fp8_e4m3_pf launch: {e}")))
}

/// Launch `gemm_fp8_e5m2` — batch FP8 E5M2 GEMM, accumulates with `+=`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_fp8_e5m2(
    graph: &CudaGraph,
    mods: &CudaFP8PrefillModules,
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
        .launch_builder(&mods.gemm_fp8_e5m2)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_fp8_e5m2 launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_fp8_e5m2` — batch fused FP8 E5M2 gate+up+SwiGLU GEMM.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fused_gate_up_swiglu_fp8_e5m2(
    graph: &CudaGraph,
    mods: &CudaFP8PrefillModules,
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
        .launch_builder(&mods.fused_gate_up_swiglu_gemm_fp8_e5m2)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_ffn_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm_fp8_e5m2 launch: {e}"))
        })
}

/// Launch `gemv_fp8_e5m2_pf` — single-token FP8 E5M2 GEMV.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemv_fp8_e5m2_pf(
    graph: &CudaGraph,
    mods: &CudaFP8PrefillModules,
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
        .launch_builder(&mods.gemv_fp8_e5m2_pf)
        .arg(d_blocks)
        .arg(d_input)
        .arg(d_output)
        .arg(&n_rows)
        .arg(&k)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemv_fp8_e5m2_pf launch: {e}")))
}

// =============================================================================
// encode_fp8_ffn_phase
// =============================================================================

/// Batched FFN sublayer for FP8 E4M3/E5M2 models.
///
/// Pipeline:
/// 1. Batched RMSNorm: `d_input → d_normed` (all tokens)
/// 2. Fused gate+up+SwiGLU GEMM: `d_normed → d_swiglu` (all tokens)
/// 3. Down GEMM + residual: `d_swiglu → d_input` (fused residual via residual_add)
///
/// # Safety
/// All device buffers must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_fp8_ffn_phase(
    graph: &CudaGraph,
    fp8_mods: &CudaFP8PrefillModules,
    pmods: &CudaPrefillModules,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    pb: &mut CudaPrefillBuffers,
    eps: f32,
    is_e4m3: bool,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size as u32;
    let h = pb.hidden_size as u32;
    let inter = pb.intermediate_size as u32;

    // Step 1: Batched RMSNorm — reuse from prefill modules (batched_rmsnorm).
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
            .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm ffn fp8: {e}")))?;
    }

    // Step 2: Fused gate+up+SwiGLU GEMM (d_normed → d_swiglu).
    if is_e4m3 {
        launch_fused_gate_up_swiglu_fp8_e4m3(
            graph,
            fp8_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?;
    } else {
        launch_fused_gate_up_swiglu_fp8_e5m2(
            graph,
            fp8_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?;
    }

    // Step 3: Down GEMM into d_normed (scratch), then residual add.
    // gemm_fp8_e4m3/e5m2 accumulate with +=, so zero d_normed first.
    {
        let n = pb.actual_batch_size * pb.hidden_size;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed down fp8: {e}")))?;
    }
    if is_e4m3 {
        launch_gemm_fp8_e4m3(
            graph,
            fp8_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?;
    } else {
        launch_gemm_fp8_e5m2(
            graph,
            fp8_mods,
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
// encode_fp8_prefill_layer
// =============================================================================

/// Encode one full transformer layer for FP8 E4M3/E5M2 batch prefill.
///
/// Uses batch GEMM for all linear projections.  Attention is processed
/// sequentially per token (same approach as Q-std prefill) because each query
/// position needs access to all prior KV entries.
///
/// # Safety
/// All device buffers and weight slices must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_fp8_prefill_layer(
    graph: &CudaGraph,
    fp8_mods: &CudaFP8PrefillModules,
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
    is_e4m3: bool,
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
            .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm attn fp8: {e}")))?;
    }

    // ─── 2. Batched QKV GEMM: d_normed → d_qkv ──────────────────────────────
    // Zero d_qkv first (kernels accumulate with +=).
    {
        let n = bs * qkv_total;
        let mut dst_view = pb.d_qkv.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_qkv fp8: {e}")))?;
    }
    if is_e4m3 {
        launch_gemm_fp8_e4m3(
            graph,
            fp8_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?;
    } else {
        launch_gemm_fp8_e5m2(
            graph,
            fp8_mods,
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
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out fp8: {e}")))?;
    }

    for t in 0..bs {
        let pos = pos_start + t;

        // Copy this token's hidden state into st_bufs.d_hidden (for attn output write-back).
        {
            let src_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| CudaGraphError::DriverError(format!("copy hidden fp8 t={t}: {e}")))?;
        }

        // Copy this token's QKV into st_bufs.d_qkv.
        {
            let src_view: CudaView<f32> = pb.d_qkv.slice(t * qkv_total..(t + 1) * qkv_total);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_qkv)
                .map_err(|e| CudaGraphError::DriverError(format!("copy qkv fp8 t={t}: {e}")))?;
        }

        // Upload RoPE cos/sin for this token's position.
        let rope_off = t * half_dim;
        graph
            .stream_arc()
            .memcpy_htod(
                &cos_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_cos,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos fp8 t={t}: {e}")))?;
        graph
            .stream_arc()
            .memcpy_htod(
                &sin_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_sin,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin fp8 t={t}: {e}")))?;

        // Upload pos and seq_len (pos+1) for this token.
        let pos_seqlen = [pos as u32, (pos + 1) as u32];
        graph
            .stream_arc()
            .memcpy_htod(&pos_seqlen, &mut st_bufs.d_pos_seqlen)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("upload pos_seqlen fp8 t={t}: {e}"))
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
                    CudaGraphError::DriverError(format!("copy attn_out fp8 t={t}: {e}"))
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
                CudaGraphError::DriverError(format!("zero d_normed attn_proj fp8: {e}"))
            })?;
    }
    let attn_proj_rows = h as u32;
    let attn_proj_k = (nq * hd) as u32;
    if is_e4m3 {
        launch_gemm_fp8_e4m3(
            graph,
            fp8_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?;
    } else {
        launch_gemm_fp8_e5m2(
            graph,
            fp8_mods,
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
    encode_fp8_ffn_phase(
        graph,
        fp8_mods,
        pmods,
        d_ffn_norm_weight,
        d_gate_up_weight,
        d_down_weight,
        pb,
        eps,
        is_e4m3,
    )?;

    Ok(())
}

// =============================================================================
// Public entry point: try_cuda_prefill_fp8
// =============================================================================

/// Batch prefill for FP8 E4M3 or E5M2 quantised models.
///
/// Processes `batch_size` tokens simultaneously using real fused batch GEMM kernels
/// for all linear projections.  Attention is processed per-token sequentially.
///
/// Set `is_e4m3 = true` for E4M3 weights, `is_e4m3 = false` for E5M2 weights.
///
/// # Arguments
///
/// - `hidden_batch` — host-side batched hidden states in row-major layout:
///   `[batch_size × hidden_size]` (token-major).  Converted to column-major internally.
/// - `logits_out` / `greedy_token_id_out` — if `Some`, the function runs the final
///   norm and LM head for the last token and returns either full logits or the argmax.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_prefill_fp8(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[CudaFP8PrefillLayerParams<'_>],
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
    is_e4m3: bool,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), CudaGraphError> {
    if batch_size == 0 {
        return Ok(());
    }

    // Get global CudaGraph singleton.
    let graph = CudaGraph::global()?;

    // Compile / retrieve the FP8 prefill CUDA modules.
    let fp8_mods = init_fp8_prefill_modules(&graph)?;

    // Compile / retrieve the existing prefill modules (for batched_rmsnorm).
    let pmods = init_prefill_modules(&graph)?;

    // Compile / retrieve the attention modules.
    let attn_mods = init_attn_modules(&graph)?;

    // Upload / cache all layer weights (f32 norms + raw AoS FP8 quantised projections).
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
        layer_weights.push(LayerWeightHandles {
            attn_norm: get_or_upload_f32_weight(&graph, lp.attn_norm_handle, lp.attn_norm_bytes)?,
            fused_qkv: graph
                .get_or_upload_weight_aos_raw(lp.fused_qkv_handle, lp.fused_qkv_bytes)?,
            q_norm: get_or_upload_f32_weight(&graph, lp.q_norm_handle, lp.q_norm_bytes)?,
            k_norm: get_or_upload_f32_weight(&graph, lp.k_norm_handle, lp.k_norm_bytes)?,
            attn_proj: graph
                .get_or_upload_weight_aos_raw(lp.attn_proj_handle, lp.attn_proj_bytes)?,
            ffn_norm: get_or_upload_f32_weight(&graph, lp.ffn_norm_handle, lp.ffn_norm_bytes)?,
            gate_up: graph.get_or_upload_weight_aos_raw(lp.gate_up_handle, lp.gate_up_bytes)?,
            down: graph.get_or_upload_weight_aos_raw(lp.ffn_down_handle, lp.ffn_down_bytes)?,
        });
    }

    // Allocate / acquire the batched prefill activation buffers.
    let mut pb_guard = acquire_fp8_prefill_buffers(
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
        .ok_or_else(|| CudaGraphError::DriverError("fp8 prefill buffer not allocated".into()))?;

    // Allocate / acquire the KV cache.
    let mut kv_guard = acquire_fp8_kv_cache(&graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("KV cache not allocated fp8".into()))?;

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
        .ok_or_else(|| CudaGraphError::DriverError("full-layer buffer not allocated fp8".into()))?;

    // Upload the hidden batch to GPU in column-major layout.
    // Input from host is row-major: [batch_size × hidden_size].
    // GPU expects column-major: [hidden_size × batch_size] (d_input[col*h + elem]).
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
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_batch fp8: {e}")))?;
    }

    // Run each transformer layer.
    for (layer_idx, lw) in layer_weights.iter().enumerate() {
        unsafe {
            encode_fp8_prefill_layer(
                &graph,
                &fp8_mods,
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
                is_e4m3,
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
                    CudaGraphError::DriverError(format!("copy last hidden fp8 lm: {e}"))
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
                .map_err(|e| CudaGraphError::DriverError(format!("final norm fp8: {e:?}")))?;
        }

        // LM head GEMV on the normed last-token hidden state.
        let mut lm_logits_guard = acquire_fp8_logits(&graph, lm_head_out_features)?;
        let d_logits = &mut lm_logits_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("logits buf not allocated fp8".into()))?
            .0;

        unsafe {
            if is_e4m3 {
                launch_gemv_fp8_e4m3_pf(
                    &graph,
                    &fp8_mods,
                    &d_lm_head_w,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                )?;
            } else {
                launch_gemv_fp8_e5m2_pf(
                    &graph,
                    &fp8_mods,
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
            .map_err(|e| CudaGraphError::DriverError(format!("sync fp8 lm: {e}")))?;

        let logits_host = graph
            .stream_arc()
            .clone_dtoh(d_logits)
            .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits fp8: {e}")))?;

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
        .map_err(|e| CudaGraphError::DriverError(format!("sync fp8 end: {e}")))?;

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::gpu_backend::cuda_fp8_prefill_kernels::CUDA_FP8_PREFILL_KERNELS_SRC;

    /// Verify the kernel source contains `gemm_fp8_e4m3`.
    #[test]
    fn test_fp8_prefill_kernel_source_has_gemm_e4m3() {
        assert!(
            CUDA_FP8_PREFILL_KERNELS_SRC.contains("gemm_fp8_e4m3"),
            "CUDA_FP8_PREFILL_KERNELS_SRC must contain gemm_fp8_e4m3"
        );
    }

    /// Verify the kernel source contains `gemm_fp8_e4m3_residual`.
    #[test]
    fn test_fp8_prefill_kernel_source_has_gemm_residual_e4m3() {
        assert!(
            CUDA_FP8_PREFILL_KERNELS_SRC.contains("gemm_fp8_e4m3_residual"),
            "CUDA_FP8_PREFILL_KERNELS_SRC must contain gemm_fp8_e4m3_residual"
        );
    }

    /// Verify the kernel source contains `fused_gate_up_swiglu_gemm_fp8_e4m3`.
    #[test]
    fn test_fp8_prefill_kernel_source_has_fused_gate_up_e4m3() {
        assert!(
            CUDA_FP8_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_fp8_e4m3"),
            "CUDA_FP8_PREFILL_KERNELS_SRC must contain fused_gate_up_swiglu_gemm_fp8_e4m3"
        );
    }

    /// Verify the kernel source contains `gemv_fp8_e5m2_pf`.
    #[test]
    fn test_fp8_prefill_kernel_source_has_gemv_e5m2_pf() {
        assert!(
            CUDA_FP8_PREFILL_KERNELS_SRC.contains("gemv_fp8_e5m2_pf"),
            "CUDA_FP8_PREFILL_KERNELS_SRC must contain gemv_fp8_e5m2_pf"
        );
    }
}
