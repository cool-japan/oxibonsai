//! Types, init, singleton state, and resource-acquisition helpers for the
//! K-quant CUDA batch prefill path.
//!
//! Holds [`KQuantFormat`], the compiled-modules container
//! [`CudaKQuantPrefillModules`], [`CudaKQuantPrefillLayerParams`], the
//! singleton `CudaKQuantPrefillState`, and the buffer/KV-cache/logits
//! acquisition helpers.

use cudarc::driver::{CudaFunction, CudaSlice};
use std::sync::{Arc, Mutex, OnceLock};

use crate::gpu_backend::cuda_full_layer::CudaKvCache;
use crate::gpu_backend::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};
use crate::gpu_backend::cuda_k_quant_prefill_kernels::CUDA_K_QUANT_PREFILL_KERNELS_SRC;
use crate::gpu_backend::cuda_prefill::CudaPrefillBuffers;

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
pub(super) fn acquire_k_quant_prefill_buffers(
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
pub(super) fn acquire_k_quant_kv_cache(
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

pub(super) type KQuantLogitsGuard = std::sync::MutexGuard<'static, Option<(CudaSlice<f32>, usize)>>;

/// Acquire or (re-)allocate the LM-head logits buffer.
pub(super) fn acquire_k_quant_logits(
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
