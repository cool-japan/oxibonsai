//! Process-wide state, types, init, and resource-acquisition helpers for the
//! CUDA prefill path.
//!
//! Holds the singleton `CudaPrefillState` (compiled modules, cached prefill
//! buffers, KV cache, single-token scratch, cached logits buffer) plus the
//! typed containers [`CudaPrefillBuffers`] and [`CudaPrefillModules`].
//!
//! All public types in this file are re-exported by [`super`] so external
//! callers continue to see the same `super::cuda_prefill::*` API surface.

use cudarc::driver::{CudaFunction, CudaSlice};
use std::sync::{Arc, Mutex, OnceLock};

use crate::gpu_backend::cuda_full_layer::{CudaFullLayerBuffers, CudaKvCache};
use crate::gpu_backend::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};
use crate::gpu_backend::cuda_prefill_kernels::CUDA_PREFILL_KERNELS_SRC;

/// Type alias for the per-layer weight tuple used during prefill.
/// Fields (in order): attn_norm, fused_qkv, q_norm, k_norm, attn_proj, ffn_norm, gate_up, down.
pub(super) type LayerWeightArcs = (
    Arc<CudaSlice<f32>>, // attn_norm
    Arc<CudaSlice<u8>>,  // fused_qkv
    Arc<CudaSlice<f32>>, // q_norm
    Arc<CudaSlice<f32>>, // k_norm
    Arc<CudaSlice<u8>>,  // attn_proj
    Arc<CudaSlice<f32>>, // ffn_norm
    Arc<CudaSlice<u8>>,  // gate_up (fused)
    Arc<CudaSlice<u8>>,  // down
);

// =============================================================================
// Pre-allocated prefill GPU buffers
// =============================================================================

/// Pre-allocated GPU activation buffers for prefill (batch) processing.
///
/// All batched buffers use column-major layout: `buf[col * dim + element]`.
pub struct CudaPrefillBuffers {
    /// Batched hidden states: `[capacity * hidden_size]` f32 (column-major).
    pub d_input: CudaSlice<f32>,
    /// Batched RMSNorm output: `[capacity * hidden_size]` f32 (column-major).
    pub d_normed: CudaSlice<f32>,
    /// Batched QKV GEMM output: `[capacity * (nq+2*nkv)*head_dim]` f32.
    pub d_qkv: CudaSlice<f32>,
    /// Batched attention output: `[capacity * nq*head_dim]` f32 (column-major).
    pub d_attn_out: CudaSlice<f32>,
    /// Batched gate+up GEMM output: `[capacity * intermediate_size]` f32.
    /// Layout: `[gate: bs*inter | up: bs*inter]` for `batched_swiglu`.
    pub d_gate_up: CudaSlice<f32>,
    /// Batched SwiGLU output: `[capacity * intermediate_size]` f32 (column-major).
    pub d_swiglu: CudaSlice<f32>,
    /// Allocated capacity (max batch_size for which buffers are valid).
    pub capacity: usize,
    /// Currently-active batch size (≤ capacity), set before each encode call.
    pub actual_batch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub nq: usize,
    pub nkv: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

// SAFETY: CudaSlice<f32> is Send in cudarc.
unsafe impl Send for CudaPrefillBuffers {}
unsafe impl Sync for CudaPrefillBuffers {}

impl CudaPrefillBuffers {
    /// Check whether these buffers can serve the requested dimensions.
    ///
    /// `batch_size` uses capacity comparison (`<=`) so buffers allocated for a
    /// larger batch can be reused for smaller batches without reallocation.
    /// All other dimensions must match exactly (they determine pointer layouts).
    #[allow(clippy::too_many_arguments)]
    pub fn matches(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> bool {
        batch_size <= self.capacity   // capacity-based: smaller batches reuse larger allocations
            && self.hidden_size == hidden_size
            && self.intermediate_size == intermediate_size
            && self.nq == nq
            && self.nkv == nkv
            && self.head_dim == head_dim
            && self.max_seq == max_seq
    }
}

// =============================================================================
// Compiled prefill CUDA modules
// =============================================================================

/// Compiled CUDA function handles for the 8 prefill kernels (5 Q1 + 3 TQ2).
pub struct CudaPrefillModules {
    pub gemm_v7: CudaFunction,
    pub gemm_v7_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm: CudaFunction,
    pub batched_swiglu: CudaFunction,
    pub batched_rmsnorm: CudaFunction,
    /// TQ2 batch GEMM — accumulates into output with `+=`.
    pub gemm_tq2_v7: CudaFunction,
    /// TQ2 batch GEMM + fused in-place residual add.
    pub gemm_tq2_v7_residual: CudaFunction,
    /// TQ2 fused gate+up+SwiGLU batch GEMM.
    pub fused_gate_up_swiglu_gemm_tq2: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaPrefillModules {}
unsafe impl Sync for CudaPrefillModules {}

// =============================================================================
// Process-wide singleton state for the prefill path
// =============================================================================

struct CudaPrefillState {
    prefill_modules: Mutex<Option<Arc<CudaPrefillModules>>>,
    prefill_buffers: Mutex<Option<CudaPrefillBuffers>>,
    /// Shared KV cache (same singleton as the decode path).
    kv_cache: Mutex<Option<CudaKvCache>>,
    /// Reuse the single-token full-layer buffers for per-token attention.
    full_layer_buffers: Mutex<Option<CudaFullLayerBuffers>>,
    /// Cached logits buffer: (buffer, out_features_count).
    prefill_logits: Mutex<Option<(CudaSlice<f32>, usize)>>,
}

unsafe impl Send for CudaPrefillState {}
unsafe impl Sync for CudaPrefillState {}

static PREFILL_STATE: OnceLock<CudaPrefillState> = OnceLock::new();

fn prefill_state() -> &'static CudaPrefillState {
    PREFILL_STATE.get_or_init(|| CudaPrefillState {
        prefill_modules: Mutex::new(None),
        prefill_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        full_layer_buffers: Mutex::new(None),
        prefill_logits: Mutex::new(None),
    })
}

// =============================================================================
// Module init
// =============================================================================

/// Compile and cache the 5 CUDA prefill kernels.
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_prefill_modules(graph: &CudaGraph) -> Result<Arc<CudaPrefillModules>, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .prefill_modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_PREFILL_KERNELS_SRC, "prefill_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module prefill: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let modules = Arc::new(CudaPrefillModules {
        gemm_v7: load("gemm_q1_g128_v7")?,
        gemm_v7_residual: load("gemm_q1_g128_v7_residual")?,
        fused_gate_up_swiglu_gemm: load("fused_gate_up_swiglu_gemm_q1")?,
        batched_swiglu: load("batched_swiglu")?,
        batched_rmsnorm: load("batched_rmsnorm_v2")?,
        gemm_tq2_v7: load("gemm_tq2_g128_v7")?,
        gemm_tq2_v7_residual: load("gemm_tq2_g128_v7_residual")?,
        fused_gate_up_swiglu_gemm_tq2: load("fused_gate_up_swiglu_gemm_tq2")?,
    });

    *guard = Some(Arc::clone(&modules));
    Ok(modules)
}

// =============================================================================
// Buffer / cache acquisition helpers
// =============================================================================

/// Round up `n` to the next power of two (minimum 1).
fn next_pow2_capacity(n: usize) -> usize {
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
pub(super) fn acquire_prefill_buffers(
    graph: &CudaGraph,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaPrefillBuffers>>, CudaGraphError> {
    let state = prefill_state();
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
        let capacity = next_pow2_capacity(batch_size);
        let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros pb({n}): {e}")))
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
        // Reusing existing allocation — just update the active batch size.
        // SAFETY: needs_alloc is false only when guard is Some(b), so this is infallible.
        guard
            .as_mut()
            .expect("guard is Some when needs_alloc is false")
            .actual_batch_size = batch_size;
    }

    Ok(guard)
}

/// Acquire or (re-)allocate the shared GPU KV cache.
pub(super) fn acquire_prefill_kv_cache(
    graph: &CudaGraph,
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaKvCache>>, CudaGraphError> {
    let state = prefill_state();
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
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv k: {e}")))?;
        let v_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv v: {e}")))?;

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

/// Acquire or (re-)allocate single-token full-layer buffers for per-token attention.
pub(super) fn acquire_single_token_buffers(
    graph: &CudaGraph,
    hidden_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
    intermediate_size: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaFullLayerBuffers>>, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .full_layer_buffers
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(b) => !b.matches(hidden_size, nq, nkv, head_dim, max_seq, intermediate_size),
        None => true,
    };

    if needs_alloc {
        let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc st({n}): {e}")))
        };

        let qkv_total = nq * head_dim + 2 * nkv * head_dim;
        let half_dim = head_dim / 2;

        let alloc_u32 = |n: usize| -> Result<CudaSlice<u32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<u32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc u32({n}): {e}")))
        };
        *guard = Some(CudaFullLayerBuffers {
            d_hidden: alloc(hidden_size)?,
            d_normed: alloc(hidden_size)?,
            d_qkv: alloc(qkv_total)?,
            d_q_rope: alloc(nq * head_dim)?,
            d_k_rope: alloc(nkv * head_dim)?,
            d_cos: alloc(half_dim)?,
            d_sin: alloc(half_dim)?,
            d_scores: alloc(nq * max_seq)?,
            d_attn_out: alloc(nq * head_dim)?,
            d_gate_up: alloc(2 * intermediate_size)?,
            d_swiglu: alloc(intermediate_size)?,
            d_pos_seqlen: alloc_u32(2)?,
            hidden_size,
            nq,
            nkv,
            head_dim,
            max_seq,
            intermediate_size,
        });
    }

    Ok(guard)
}

pub(super) type PrefillLogitsGuard =
    std::sync::MutexGuard<'static, Option<(CudaSlice<f32>, usize)>>;

/// Acquire or (re-)allocate the LM-head logits buffer.
pub(super) fn acquire_prefill_logits(
    graph: &CudaGraph,
    n: usize,
) -> Result<PrefillLogitsGuard, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .prefill_logits
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
            .map_err(|e| CudaGraphError::DriverError(format!("alloc logits buf({n}): {e}")))?;
        *guard = Some((buf, n));
    }
    Ok(guard)
}
