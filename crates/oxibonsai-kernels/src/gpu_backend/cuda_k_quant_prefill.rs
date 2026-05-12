//! Batch prefill (GEMM) dispatch for OxiBonsai — K-quant CUDA backend.
//!
//! This module provides the batch prefill path for K-quant quantised models
//! (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K). It mirrors the architecture of
//! [`cuda_q_std_prefill`] for Q4_0/Q8_0, but dispatches across 6 formats
//! via the [`KQuantFormat`] enum.
//!
//! # Architecture
//!
//! - [`CudaKQuantPrefillModules`]: Compiled CUDA functions for the 18 batch GEMM kernels.
//! - [`CudaKQuantPrefillLayerParams`]: Per-layer weight handles and raw AoS bytes.
//! - [`try_cuda_prefill_k_quant`]: Public entry point for K-quant batch prefill.
//!
//! # Batch tensor layout
//!
//! All batched buffers use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index.
//!
//! # Weight layout
//!
//! K-quant weights stay in AoS layout as stored in GGUF.  QK_K = 256 weights
//! per super-block; `hidden_size` must be a multiple of 256.

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
use super::cuda_k_quant_prefill_kernels::CUDA_K_QUANT_PREFILL_KERNELS_SRC;
use super::cuda_prefill::{init_prefill_modules, CudaPrefillBuffers, CudaPrefillModules};

// =============================================================================
// KQuantFormat — format selector enum
// =============================================================================

/// K-quant format selector for kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantFormat {
    /// Q2_K: 84 bytes/super-block, 256 weights, 2 bits/weight.
    Q2K,
    /// Q3_K: 110 bytes/super-block, 256 weights, 3 bits/weight.
    Q3K,
    /// Q4_K: 144 bytes/super-block, 256 weights, 4 bits/weight.
    Q4K,
    /// Q5_K: 176 bytes/super-block, 256 weights, 5 bits/weight.
    Q5K,
    /// Q6_K: 210 bytes/super-block, 256 weights, 6 bits/weight.
    Q6K,
    /// Q8_K: 292 bytes/super-block, 256 weights, 8 bits/weight (f32 scale).
    Q8K,
}

// =============================================================================
// Compiled K-quant prefill CUDA modules
// =============================================================================

/// Compiled CUDA function handles for the 18 K-quant batch GEMM kernels.
pub struct CudaKQuantPrefillModules {
    pub gemm_q2k: CudaFunction,
    pub gemm_q2k_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q2k: CudaFunction,
    pub gemm_q3k: CudaFunction,
    pub gemm_q3k_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q3k: CudaFunction,
    pub gemm_q4k: CudaFunction,
    pub gemm_q4k_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q4k: CudaFunction,
    pub gemm_q5k: CudaFunction,
    pub gemm_q5k_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q5k: CudaFunction,
    pub gemm_q6k: CudaFunction,
    pub gemm_q6k_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q6k: CudaFunction,
    pub gemm_q8k: CudaFunction,
    pub gemm_q8k_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm_q8k: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaKQuantPrefillModules {}
unsafe impl Sync for CudaKQuantPrefillModules {}

// =============================================================================
// Process-wide singleton state
// =============================================================================

struct CudaKQuantPrefillState {
    kquant_modules: Mutex<Option<Arc<CudaKQuantPrefillModules>>>,
    prefill_buffers: Mutex<Option<CudaPrefillBuffers>>,
    kv_cache: Mutex<Option<CudaKvCache>>,
    logits_buf: Mutex<Option<(CudaSlice<f32>, usize)>>,
}

unsafe impl Send for CudaKQuantPrefillState {}
unsafe impl Sync for CudaKQuantPrefillState {}

static K_QUANT_PREFILL_STATE: OnceLock<CudaKQuantPrefillState> = OnceLock::new();

fn k_quant_prefill_state() -> &'static CudaKQuantPrefillState {
    K_QUANT_PREFILL_STATE.get_or_init(|| CudaKQuantPrefillState {
        kquant_modules: Mutex::new(None),
        prefill_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        logits_buf: Mutex::new(None),
    })
}

// =============================================================================
// Module init
// =============================================================================

/// Compile and cache the 18 K-quant CUDA prefill kernels.
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_k_quant_prefill_modules(
    graph: &CudaGraph,
) -> Result<Arc<CudaKQuantPrefillModules>, CudaGraphError> {
    let state = k_quant_prefill_state();
    let mut guard = state
        .kquant_modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_K_QUANT_PREFILL_KERNELS_SRC, "k_quant_prefill_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module k_quant_prefill: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let mods = Arc::new(CudaKQuantPrefillModules {
        gemm_q2k: load("gemm_q2k")?,
        gemm_q2k_residual: load("gemm_q2k_residual")?,
        fused_gate_up_swiglu_gemm_q2k: load("fused_gate_up_swiglu_gemm_q2k")?,
        gemm_q3k: load("gemm_q3k")?,
        gemm_q3k_residual: load("gemm_q3k_residual")?,
        fused_gate_up_swiglu_gemm_q3k: load("fused_gate_up_swiglu_gemm_q3k")?,
        gemm_q4k: load("gemm_q4k")?,
        gemm_q4k_residual: load("gemm_q4k_residual")?,
        fused_gate_up_swiglu_gemm_q4k: load("fused_gate_up_swiglu_gemm_q4k")?,
        gemm_q5k: load("gemm_q5k")?,
        gemm_q5k_residual: load("gemm_q5k_residual")?,
        fused_gate_up_swiglu_gemm_q5k: load("fused_gate_up_swiglu_gemm_q5k")?,
        gemm_q6k: load("gemm_q6k")?,
        gemm_q6k_residual: load("gemm_q6k_residual")?,
        fused_gate_up_swiglu_gemm_q6k: load("fused_gate_up_swiglu_gemm_q6k")?,
        gemm_q8k: load("gemm_q8k")?,
        gemm_q8k_residual: load("gemm_q8k_residual")?,
        fused_gate_up_swiglu_gemm_q8k: load("fused_gate_up_swiglu_gemm_q8k")?,
    });

    *guard = Some(Arc::clone(&mods));
    Ok(mods)
}

// =============================================================================
// Per-layer parameter struct
// =============================================================================

/// Per-layer parameters for the K-quant CUDA prefill path.
///
/// Weight bytes are raw AoS layout as stored in GGUF (QK_K = 256 weights/super-block).
pub struct CudaKQuantPrefillLayerParams<'a> {
    /// Which K-quant format this layer uses.
    pub format: KQuantFormat,
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
fn acquire_k_quant_prefill_buffers(
    graph: &CudaGraph,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaPrefillBuffers>>, CudaGraphError> {
    let state = k_quant_prefill_state();
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
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros kqpb({n}): {e}")))
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
fn acquire_k_quant_kv_cache(
    graph: &CudaGraph,
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaKvCache>>, CudaGraphError> {
    let state = k_quant_prefill_state();
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
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv k_cache kquant: {e}")))?;
        let v_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv v_cache kquant: {e}")))?;

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

type KQuantLogitsGuard = std::sync::MutexGuard<'static, Option<(CudaSlice<f32>, usize)>>;

/// Acquire or (re-)allocate the LM-head logits buffer.
fn acquire_k_quant_logits(
    graph: &CudaGraph,
    n: usize,
) -> Result<KQuantLogitsGuard, CudaGraphError> {
    let state = k_quant_prefill_state();
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
            .map_err(|e| CudaGraphError::DriverError(format!("alloc logits kquant({n}): {e}")))?;
        *guard = Some((buf, n));
    }

    Ok(guard)
}

// =============================================================================
// Low-level CUDA kernel launchers (one per kernel, 18 total)
// =============================================================================

/// Launch `gemm_q2k` — batch Q2_K GEMM, accumulates with `+=`.
///
/// # Safety
/// All slices must be valid device pointers on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_q2k(
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
unsafe fn launch_gemm_q2k_residual(
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
unsafe fn launch_fused_gate_up_swiglu_q2k(
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
unsafe fn launch_gemm_q3k(
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
unsafe fn launch_gemm_q3k_residual(
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
unsafe fn launch_fused_gate_up_swiglu_q3k(
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
unsafe fn launch_gemm_q4k(
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
unsafe fn launch_gemm_q4k_residual(
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
unsafe fn launch_fused_gate_up_swiglu_q4k(
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
unsafe fn launch_gemm_q5k(
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
unsafe fn launch_gemm_q5k_residual(
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
unsafe fn launch_fused_gate_up_swiglu_q5k(
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
unsafe fn launch_gemm_q6k(
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
unsafe fn launch_gemm_q6k_residual(
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
unsafe fn launch_fused_gate_up_swiglu_q6k(
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
unsafe fn launch_gemm_q8k(
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
unsafe fn launch_gemm_q8k_residual(
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
unsafe fn launch_fused_gate_up_swiglu_q8k(
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

// =============================================================================
// encode_k_quant_ffn_phase
// =============================================================================

/// Batched FFN sublayer for K-quant models.
///
/// Pipeline:
/// 1. Batched RMSNorm: `d_input → d_normed` (all tokens)
/// 2. Fused gate+up+SwiGLU GEMM: `d_normed → d_swiglu` (format-specific)
/// 3. Down GEMM + residual: `d_swiglu → d_input` (format-specific)
///
/// # Safety
/// All device buffers must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_k_quant_ffn_phase(
    graph: &CudaGraph,
    kq_mods: &CudaKQuantPrefillModules,
    pmods: &CudaPrefillModules,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    pb: &mut CudaPrefillBuffers,
    eps: f32,
    fmt: KQuantFormat,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size as u32;
    let h = pb.hidden_size as u32;
    let inter = pb.intermediate_size as u32;

    // Step 1: Batched RMSNorm → d_normed.
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
            .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm ffn kquant: {e}")))?;
    }

    // Step 2: Fused gate+up+SwiGLU GEMM (d_normed → d_swiglu).
    // Zero d_gate_up buffer first (fused kernels write directly, not +=).
    {
        let n = 2 * pb.actual_batch_size * pb.intermediate_size;
        let mut dst_view = pb.d_gate_up.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_gate_up kquant: {e}")))?;
    }
    match fmt {
        KQuantFormat::Q2K => launch_fused_gate_up_swiglu_q2k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q3K => launch_fused_gate_up_swiglu_q3k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q4K => launch_fused_gate_up_swiglu_q4k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q5K => launch_fused_gate_up_swiglu_q5k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q6K => launch_fused_gate_up_swiglu_q6k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
        KQuantFormat::Q8K => launch_fused_gate_up_swiglu_q8k(
            graph,
            kq_mods,
            d_gate_up_weight,
            &pb.d_normed,
            &mut pb.d_swiglu,
            inter,
            h,
            bs,
        )?,
    }

    // Step 3: Down GEMM into d_normed (scratch), then residual add.
    // Zero d_normed first (kernels accumulate with +=).
    {
        let n = pb.actual_batch_size * pb.hidden_size;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed down kquant: {e}")))?;
    }
    match fmt {
        KQuantFormat::Q2K => launch_gemm_q2k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q3K => launch_gemm_q3k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q4K => launch_gemm_q4k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q5K => launch_gemm_q5k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q6K => launch_gemm_q6k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
        KQuantFormat::Q8K => launch_gemm_q8k(
            graph,
            kq_mods,
            d_down_weight,
            &pb.d_swiglu,
            &mut pb.d_normed,
            h,
            inter,
            bs,
        )?,
    }

    let total_bh = (pb.actual_batch_size * pb.hidden_size) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    Ok(())
}

// =============================================================================
// encode_k_quant_prefill_layer
// =============================================================================

/// Encode one full transformer layer for K-quant batch prefill.
///
/// Same 5-step structure as Phase 24 Q4_0/Q8_0 path, with format dispatch via
/// [`KQuantFormat`] for all linear projections.
///
/// # Safety
/// All device buffers and weight slices must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn encode_k_quant_prefill_layer(
    graph: &CudaGraph,
    kq_mods: &CudaKQuantPrefillModules,
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
    fmt: KQuantFormat,
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
            .map_err(|e| {
                CudaGraphError::DriverError(format!("batched_rmsnorm attn kquant: {e}"))
            })?;
    }

    // ─── 2. Batched QKV GEMM: d_normed → d_qkv ──────────────────────────────
    // Zero d_qkv first (kernels accumulate with +=).
    {
        let n = bs * qkv_total;
        let mut dst_view = pb.d_qkv.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_qkv kquant: {e}")))?;
    }
    match fmt {
        KQuantFormat::Q2K => launch_gemm_q2k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q3K => launch_gemm_q3k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q4K => launch_gemm_q4k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q5K => launch_gemm_q5k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q6K => launch_gemm_q6k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
        KQuantFormat::Q8K => launch_gemm_q8k(
            graph,
            kq_mods,
            d_fused_qkv_weight,
            &pb.d_normed,
            &mut pb.d_qkv,
            qkv_total as u32,
            h_u32,
            bs_u32,
        )?,
    }

    // ─── 3. Sequential attention for each token ──────────────────────────────
    {
        let n = bs * nq * hd;
        let mut dst_view = pb.d_attn_out.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out kquant: {e}")))?;
    }

    for t in 0..bs {
        let pos = pos_start + t;

        // Copy this token's hidden state into st_bufs.d_hidden.
        {
            let src_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy hidden kquant t={t}: {e}"))
                })?;
        }

        // Copy this token's QKV into st_bufs.d_qkv.
        {
            let src_view: CudaView<f32> = pb.d_qkv.slice(t * qkv_total..(t + 1) * qkv_total);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_qkv)
                .map_err(|e| CudaGraphError::DriverError(format!("copy qkv kquant t={t}: {e}")))?;
        }

        // Upload RoPE cos/sin for this token's position.
        let rope_off = t * half_dim;
        graph
            .stream_arc()
            .memcpy_htod(
                &cos_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_cos,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos kquant t={t}: {e}")))?;
        graph
            .stream_arc()
            .memcpy_htod(
                &sin_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_sin,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin kquant t={t}: {e}")))?;

        // Upload pos and seq_len (pos+1) for this token.
        let pos_seqlen = [pos as u32, (pos + 1) as u32];
        graph
            .stream_arc()
            .memcpy_htod(&pos_seqlen, &mut st_bufs.d_pos_seqlen)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("upload pos_seqlen kquant t={t}: {e}"))
            })?;

        // Run attention steps 3-7.
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
                    CudaGraphError::DriverError(format!("copy attn_out kquant t={t}: {e}"))
                })?;
        }
    }

    // ─── 4. Attn output projection + residual ────────────────────────────────
    {
        let n = bs * h;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| {
                CudaGraphError::DriverError(format!("zero d_normed attn_proj kquant: {e}"))
            })?;
    }
    let attn_proj_rows = h_u32;
    let attn_proj_k = (nq * hd) as u32;
    match fmt {
        KQuantFormat::Q2K => launch_gemm_q2k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q3K => launch_gemm_q3k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q4K => launch_gemm_q4k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q5K => launch_gemm_q5k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q6K => launch_gemm_q6k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
        KQuantFormat::Q8K => launch_gemm_q8k(
            graph,
            kq_mods,
            d_attn_proj_weight,
            &pb.d_attn_out,
            &mut pb.d_normed,
            attn_proj_rows,
            attn_proj_k,
            bs_u32,
        )?,
    }

    let total_bh = (bs * h) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    // ─── 5. Batched FFN sublayer ──────────────────────────────────────────────
    encode_k_quant_ffn_phase(
        graph,
        kq_mods,
        pmods,
        d_ffn_norm_weight,
        d_gate_up_weight,
        d_down_weight,
        pb,
        eps,
        fmt,
    )?;

    Ok(())
}

// =============================================================================
// Public entry point: try_cuda_prefill_k_quant
// =============================================================================

/// Batch prefill for K-quant quantised models (Q2_K through Q8_K).
///
/// Processes `batch_size` tokens simultaneously using real fused batch GEMM
/// kernels for all linear projections.  Attention is processed per-token
/// sequentially using the shared single-token attention kernels.
///
/// # Constraints
///
/// - `hidden_size` must be a multiple of 256 (QK_K requirement).
/// - All layer weight byte slices must be valid AoS K-quant super-block data.
///
/// # Arguments
///
/// - `hidden_batch` — host-side batched hidden states `[batch_size × hidden_size]`
///   in row-major (token-major) layout. Uploaded to GPU in column-major format.
/// - `logits_out` / `greedy_token_id_out` — if `Some`, runs final norm and LM
///   head for the last token and returns full logits or the argmax token id.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_prefill_k_quant(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[CudaKQuantPrefillLayerParams<'_>],
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
    lm_head_fmt: KQuantFormat,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), CudaGraphError> {
    if batch_size == 0 {
        return Ok(());
    }

    // K-quant requires hidden_size to be a multiple of 256 (= QK_K).
    if hidden_size % 256 != 0 {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "K-quant prefill: hidden_size={hidden_size} must be a multiple of 256"
        )));
    }

    let graph = CudaGraph::global()?;

    let kq_mods = init_k_quant_prefill_modules(&graph)?;
    let pmods = init_prefill_modules(&graph)?;
    let attn_mods = init_attn_modules(&graph)?;

    // Upload / cache all layer weights.
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
    let mut pb_guard = acquire_k_quant_prefill_buffers(
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
        .ok_or_else(|| CudaGraphError::DriverError("prefill buffer not allocated kquant".into()))?;

    // Allocate / acquire the KV cache.
    let mut kv_guard = acquire_k_quant_kv_cache(&graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("KV cache not allocated kquant".into()))?;

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
    let st_bufs = st_guard.as_mut().ok_or_else(|| {
        CudaGraphError::DriverError("full-layer buffer not allocated kquant".into())
    })?;

    // Upload the hidden batch to GPU in column-major layout.
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
            .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_batch kquant: {e}")))?;
    }

    // Determine fallback format from first layer (or Q4K if no layers).
    let default_fmt = layer_params
        .first()
        .map_or(KQuantFormat::Q4K, |lp| lp.format);

    // Run each transformer layer.
    for (layer_idx, lw) in layer_weights.iter().enumerate() {
        let fmt = layer_params
            .get(layer_idx)
            .map_or(default_fmt, |lp| lp.format);

        unsafe {
            encode_k_quant_prefill_layer(
                &graph,
                &kq_mods,
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
                fmt,
            )?;
        }
    }

    // ─── Final norm + LM head (optional) ─────────────────────────────────────
    if logits_out.is_some() || greedy_token_id_out.is_some() {
        let final_norm_h = final_norm_handle.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("final_norm_handle required for logits kquant".into())
        })?;
        let final_norm_b = final_norm_bytes.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("final_norm_bytes required for logits kquant".into())
        })?;
        let lm_head_h = lm_head_handle.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("lm_head_handle required for logits kquant".into())
        })?;
        let lm_head_b = lm_head_bytes.ok_or_else(|| {
            CudaGraphError::WeightLayoutError("lm_head_bytes required for logits kquant".into())
        })?;

        let d_final_norm_w = get_or_upload_f32_weight(&graph, final_norm_h, final_norm_b)?;
        let d_lm_head = graph.get_or_upload_weight_aos_raw(lm_head_h, lm_head_b)?;

        // Extract last token's hidden state into st_bufs.d_hidden.
        let last_t = batch_size - 1;
        {
            let src_view: CudaView<f32> = pb
                .d_input
                .slice(last_t * hidden_size..(last_t + 1) * hidden_size);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| {
                    CudaGraphError::DriverError(format!("copy last hidden kquant lm: {e}"))
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
                .map_err(|e| CudaGraphError::DriverError(format!("final norm kquant: {e:?}")))?;
        }

        // LM head: batch GEMM with batch_size=1.
        // d_normed holds the normed last-token hidden state [hidden_size].
        // GEMM output d_logits[row] = sum over k (treating d_normed as 1-column input).
        let mut lm_logits_guard = acquire_k_quant_logits(&graph, lm_head_out_features)?;
        let d_logits = &mut lm_logits_guard
            .as_mut()
            .ok_or_else(|| CudaGraphError::DriverError("logits buf not allocated kquant".into()))?
            .0;

        // Zero logits buffer before GEMM (kernels accumulate with +=).
        {
            let mut d_logits_view = d_logits.slice_mut(0..lm_head_out_features);
            graph
                .stream_arc()
                .memset_zeros(&mut d_logits_view)
                .map_err(|e| CudaGraphError::DriverError(format!("zero logits kquant: {e}")))?;
        }

        // Run batch GEMM with batch_size=1.
        let bs_one = 1u32;
        unsafe {
            match lm_head_fmt {
                KQuantFormat::Q2K => launch_gemm_q2k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q3K => launch_gemm_q3k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q4K => launch_gemm_q4k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q5K => launch_gemm_q5k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q6K => launch_gemm_q6k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
                KQuantFormat::Q8K => launch_gemm_q8k(
                    &graph,
                    &kq_mods,
                    &d_lm_head,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                    bs_one,
                )?,
            }
        }

        // Synchronise stream before D2H copy.
        graph
            .stream_arc()
            .synchronize()
            .map_err(|e| CudaGraphError::DriverError(format!("sync kquant lm: {e}")))?;

        let logits_host = graph
            .stream_arc()
            .clone_dtoh(d_logits)
            .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits kquant: {e}")))?;

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
        .map_err(|e| CudaGraphError::DriverError(format!("sync kquant end: {e}")))?;

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::CUDA_K_QUANT_PREFILL_KERNELS_SRC;

    /// Verify the kernel source contains `gemm_q2k`.
    #[test]
    fn test_k_quant_prefill_kernels_src_has_gemm_q2k() {
        assert!(
            CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("gemm_q2k"),
            "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain gemm_q2k"
        );
    }

    /// Verify the kernel source contains `gemm_q4k`.
    #[test]
    fn test_k_quant_prefill_kernels_src_has_gemm_q4k() {
        assert!(
            CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("gemm_q4k"),
            "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain gemm_q4k"
        );
    }

    /// Verify the kernel source contains `fused_gate_up_swiglu_gemm_q6k`.
    #[test]
    fn test_k_quant_prefill_kernels_src_has_fused_gate_up_q6k() {
        assert!(
            CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_q6k"),
            "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain fused_gate_up_swiglu_gemm_q6k"
        );
    }

    /// Verify the kernel source contains `gemm_q8k`.
    #[test]
    fn test_k_quant_prefill_kernels_src_has_gemm_q8k() {
        assert!(
            CUDA_K_QUANT_PREFILL_KERNELS_SRC.contains("gemm_q8k"),
            "CUDA_K_QUANT_PREFILL_KERNELS_SRC must contain gemm_q8k"
        );
    }

    /// Verify all 6 format kernels are present in the source.
    #[test]
    fn test_k_quant_format_variants_all_present() {
        let src = CUDA_K_QUANT_PREFILL_KERNELS_SRC;
        assert!(src.contains("gemm_q2k"), "missing gemm_q2k");
        assert!(src.contains("gemm_q3k"), "missing gemm_q3k");
        assert!(src.contains("gemm_q4k"), "missing gemm_q4k");
        assert!(src.contains("gemm_q5k"), "missing gemm_q5k");
        assert!(src.contains("gemm_q6k"), "missing gemm_q6k");
        assert!(src.contains("gemm_q8k"), "missing gemm_q8k");
    }
}
