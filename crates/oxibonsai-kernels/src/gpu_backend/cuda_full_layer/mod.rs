//! Full-layer GPU dispatch for OxiBonsai — CUDA backend.
//!
//! Mirrors [`metal_full_layer`] for Linux/Windows, encoding the complete
//! attention + FFN pipeline for one transformer layer on a single CUDA stream.
//! This eliminates CPU–GPU round-trips between the attention and FFN sublayers.
//!
//! # Pipeline (per token, decode path)
//!
//! **Attention sublayer:**
//! 1. Pre-attention RMSNorm (existing `rmsnorm_weighted_v2`)
//! 2. Fused QKV projection (V7/V8 GEMV)
//! 3. Fused QK-Norm + QK-RoPE (`fused_qk_norm_rope`)
//! 4. Fused KV-store (`fused_kv_store`) — writes FP16 into KV cache
//! 5. Batched attention scores V2 (`batched_attn_scores_v2`)
//! 6. Batched softmax (`batched_softmax`)
//! 7. Batched weighted sum (`batched_attn_weighted_sum`)
//!
//! **FFN sublayer:**
//! 8. Output projection + residual add (V7/V8 GEMV)
//! 9. FFN RMSNorm
//! 10. Gate+Up GEMV + SwiGLU
//! 11. Down GEMV + residual add
//!
//! # KV cache layout
//!
//! `[n_layers * nkv * max_seq * head_dim]` stored as FP16 (`u16` on Rust side).
//! Each layer's slice begins at `layer_idx * nkv * max_seq * head_dim` elements.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::result as cudarc_result;
use cudarc::driver::sys;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// ─── CUDA Driver Graph wrapper ────────────────────────────────────────────────
// We bypass cudarc's end_capture() because its CUgraphInstantiate_flags enum
// has no 0 variant, so passing "no flags" via transmute trips Rust's debug-mode
// enum-validity check. Instead we hold the raw CUgraph + CUgraphExec handles.
struct CuGraphHolder {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}
impl CuGraphHolder {
    unsafe fn launch(&self) -> Result<(), cudarc::driver::DriverError> {
        cudarc_result::graph::launch(self.cu_graph_exec, self.stream.cu_stream())
    }
    unsafe fn upload(&self) -> Result<(), cudarc::driver::DriverError> {
        cudarc_result::graph::upload(self.cu_graph_exec, self.stream.cu_stream())
    }
}
impl Drop for CuGraphHolder {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc_result::graph::exec_destroy(self.cu_graph_exec);
            let _ = cudarc_result::graph::destroy(self.cu_graph);
        }
    }
}
// SAFETY: CUgraphExec is safe to send across threads when protected by a Mutex.
// We never call into the graph from multiple threads concurrently.
unsafe impl Send for CuGraphHolder {}

use super::cuda_attn_kernels::CUDA_ATTENTION_KERNELS_SRC;
use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};

// Attention kernel launchers (extracted to keep this file under 2000 lines).
mod launchers;
use launchers::{
    launch_batched_attn_scores_v2, launch_batched_attn_weighted_sum, launch_batched_softmax,
    launch_fused_kv_store, launch_fused_qk_norm_rope,
};

// Q1 full-layer encode functions.
pub mod encode_q1;

// Milestone 2: CUDA ternary full-forward path (placeholder).
pub mod encode_ternary;

// =============================================================================
// Compiled CUDA attention modules
// =============================================================================

/// Compiled CUDA function handles for the 7 attention kernels.
pub struct CudaAttnModules {
    pub fused_qk_norm: CudaFunction,
    pub fused_qk_rope: CudaFunction,
    pub fused_qk_norm_rope: CudaFunction,
    pub fused_kv_store: CudaFunction,
    pub batched_attn_scores_v2: CudaFunction,
    pub batched_softmax: CudaFunction,
    pub batched_attn_weighted_sum: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc (wraps a raw function handle).
unsafe impl Send for CudaAttnModules {}
unsafe impl Sync for CudaAttnModules {}

// =============================================================================
// GPU KV cache
// =============================================================================

/// GPU-resident KV cache stored in FP16 to save VRAM.
///
/// Layout: `[n_layers * nkv * max_seq * head_dim]` as `u16` (FP16 bit pattern).
/// Element offset for layer `l`: `l * nkv * max_seq * head_dim`.
pub struct CudaKvCache {
    pub k_cache: CudaSlice<u16>,
    pub v_cache: CudaSlice<u16>,
    pub n_layers: usize,
    pub n_kv: usize,
    pub max_seq: usize,
    pub head_dim: usize,
}

// SAFETY: CudaSlice<u16> is Send in cudarc.
unsafe impl Send for CudaKvCache {}
unsafe impl Sync for CudaKvCache {}

impl CudaKvCache {
    /// Compute the element offset for a given layer index.
    #[inline]
    pub fn layer_offset_elements(&self, layer_idx: usize) -> u32 {
        (layer_idx * self.n_kv * self.max_seq * self.head_dim) as u32
    }

    /// Check whether this cache's dimensions match the given parameters.
    pub fn matches(&self, n_layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) -> bool {
        self.n_layers == n_layers
            && self.n_kv == n_kv
            && self.max_seq == max_seq
            && self.head_dim == head_dim
    }
}

// =============================================================================
// Full-layer intermediate buffers
// =============================================================================

/// Pre-allocated GPU activation buffers for full-layer (attention + FFN) execution.
///
/// All buffers are allocated once and reused across forward passes.
/// Lazily resized when model dimensions change.
pub struct CudaFullLayerBuffers {
    /// [hidden_size] residual stream
    pub d_hidden: CudaSlice<f32>,
    /// [hidden_size] RMSNorm output / O-proj scratch
    pub d_normed: CudaSlice<f32>,
    /// [nq*hd + 2*nkv*hd] fused QKV GEMV output
    pub d_qkv: CudaSlice<f32>,
    /// [nq * head_dim] Q after norm+RoPE
    pub d_q_rope: CudaSlice<f32>,
    /// [nkv * head_dim] K after norm+RoPE
    pub d_k_rope: CudaSlice<f32>,
    /// [half_dim] RoPE cosines
    pub d_cos: CudaSlice<f32>,
    /// [half_dim] RoPE sines
    pub d_sin: CudaSlice<f32>,
    /// [nq * max_seq] attention scores
    pub d_scores: CudaSlice<f32>,
    /// [nq * head_dim] attention output
    pub d_attn_out: CudaSlice<f32>,
    /// [2 * intermediate_size] gate+up GEMV
    pub d_gate_up: CudaSlice<f32>,
    /// [intermediate_size] SwiGLU output
    pub d_swiglu: CudaSlice<f32>,
    /// [2] pos/seq_len for CUDA-graph-captured attention kernels: [pos, seq_len]
    pub d_pos_seqlen: CudaSlice<u32>,
    /// Dimension tracking.
    pub hidden_size: usize,
    pub nq: usize,
    pub nkv: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub intermediate_size: usize,
}

// SAFETY: CudaSlice<f32> is Send in cudarc.
unsafe impl Send for CudaFullLayerBuffers {}
unsafe impl Sync for CudaFullLayerBuffers {}

impl CudaFullLayerBuffers {
    /// Returns `true` when the buffer set matches all given dimensions.
    pub fn matches(
        &self,
        hidden_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
        intermediate_size: usize,
    ) -> bool {
        self.hidden_size == hidden_size
            && self.nq == nq
            && self.nkv == nkv
            && self.head_dim == head_dim
            && self.max_seq == max_seq
            && self.intermediate_size == intermediate_size
    }
}

// =============================================================================
// Pre-cached GPU weight handles for one transformer layer
// =============================================================================

/// Weights for one transformer layer, already uploaded to GPU device memory.
///
/// Q1_0_G128 projection weights are stored in SoA layout (`Arc<CudaSlice<u8>>`).
/// Norm weights are stored as plain FP32 (`Arc<CudaSlice<f32>>`).
pub struct CudaCachedLayerWeights {
    /// Q projection (Q1 SoA)
    pub q_weight: Arc<CudaSlice<u8>>,
    /// K projection (Q1 SoA)
    pub k_weight: Arc<CudaSlice<u8>>,
    /// V projection (Q1 SoA)
    pub v_weight: Arc<CudaSlice<u8>>,
    /// O projection (Q1 SoA)
    pub o_weight: Arc<CudaSlice<u8>>,
    /// Gate+Up concatenated (Q1 SoA)
    pub gate_up_weight: Arc<CudaSlice<u8>>,
    /// Down projection (Q1 SoA)
    pub down_weight: Arc<CudaSlice<u8>>,
    /// Pre-attention RMSNorm weights
    pub pre_attn_norm: Arc<CudaSlice<f32>>,
    /// Post-attention (FFN) RMSNorm weights
    pub post_attn_norm: Arc<CudaSlice<f32>>,
    /// QK-norm for Q heads
    pub q_norm: Arc<CudaSlice<f32>>,
    /// QK-norm for K heads
    pub k_norm: Arc<CudaSlice<f32>>,
}

// SAFETY: CudaSlice is Send in cudarc; Arc provides Sync.
unsafe impl Send for CudaCachedLayerWeights {}
unsafe impl Sync for CudaCachedLayerWeights {}

// =============================================================================
// Per-layer parameter struct (mirrors metal_full_layer::FullForwardLayerParams)
// =============================================================================

/// Per-layer parameters for the CUDA full-forward path.
///
/// Mirrors [`FullForwardLayerParams`] in `metal_full_layer` so callers can
/// build params in a backend-agnostic fashion.
pub struct CudaFullForwardLayerParams<'a> {
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
// Per-process cached model weights  (avoids 288+ HashMap lookups per token)
// =============================================================================

/// All GPU weight handles for the whole model, built once and reused across tokens.
///
/// On the first decode token, `get_or_build_model_weights` uploads weights and
/// caches them here.  Subsequent tokens clone three `Arc`s (O(1)) instead of
/// running 288+ `HashMap::get` calls under a Mutex.
pub struct CudaCachedModelWeights {
    pub graph: Arc<CudaGraph>,
    /// Dummy 1-byte device slice shared across layers (k / v weight fields are
    /// unused — the fused QKV weight is stored in `q_weight`).
    pub dummy_weight: Arc<CudaSlice<u8>>,
    /// Per-layer weight handles, wrapped in Arc so cloning is O(1).
    pub layers: Arc<Vec<CudaCachedLayerWeights>>,
    /// Number of layers — used as a cheap validity token.
    pub n_layers: usize,
}

unsafe impl Send for CudaCachedModelWeights {}
unsafe impl Sync for CudaCachedModelWeights {}

// =============================================================================
// Per-process singleton for attention-layer extended state
// =============================================================================

/// Process-wide singleton holding state for the full-layer path.
struct CudaFullLayerState {
    attn_modules: Mutex<Option<Arc<CudaAttnModules>>>,
    full_layer_buffers: Mutex<Option<CudaFullLayerBuffers>>,
    kv_cache: Mutex<Option<CudaKvCache>>,
    /// Cache for FP32 norm weights (separate from the Q1 u8 weight cache).
    f32_weight_cache: Mutex<HashMap<u64, Arc<CudaSlice<f32>>>>,
    /// Cached GPU model weights — rebuilt only when the model changes.
    cached_model_weights: Mutex<Option<CudaCachedModelWeights>>,
    /// Captured CUDA driver graph for replaying the 36-layer pipeline.
    ///
    /// Three-state:
    /// - `None`             → capture not yet attempted
    /// - `Some(None)`       → capture was attempted but failed (no retry)
    /// - `Some(Some(h))`    → capture succeeded; `h` is the exec graph
    ///
    /// Reset to `None` when activation-buffer dimensions change (model switch),
    /// so a new capture is attempted with the fresh buffers.
    cuda_driver_graph: Mutex<Option<Option<CuGraphHolder>>>,
}

unsafe impl Send for CudaFullLayerState {}
unsafe impl Sync for CudaFullLayerState {}

static FULL_LAYER_STATE: OnceLock<CudaFullLayerState> = OnceLock::new();

fn full_layer_state() -> &'static CudaFullLayerState {
    FULL_LAYER_STATE.get_or_init(|| CudaFullLayerState {
        attn_modules: Mutex::new(None),
        full_layer_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        f32_weight_cache: Mutex::new(HashMap::new()),
        cached_model_weights: Mutex::new(None),
        // None = not yet attempted; Some(None) = tried & failed; Some(Some(h)) = active
        cuda_driver_graph: Mutex::new(None),
    })
}

// =============================================================================
// Profiling helper  (gated by CUDA_PROFILE env var)
// =============================================================================

static PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

#[inline(always)]
pub(super) fn profiling() -> bool {
    *PROFILE_ENABLED.get_or_init(|| std::env::var("CUDA_PROFILE").is_ok())
}

// =============================================================================
// F32 weight upload and caching
// =============================================================================

/// Upload f32 weights and cache them for reuse.
///
/// On the first call for `key`, the slice is uploaded to GPU device memory
/// and cached.  Subsequent calls return the cached `Arc<CudaSlice<f32>>`.
pub fn get_or_upload_f32_weight(
    graph: &CudaGraph,
    key: u64,
    data: &[f32],
) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
    let state = full_layer_state();
    {
        let cache = state
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(existing) = cache.get(&key) {
            return Ok(Arc::clone(existing));
        }
    }

    // Upload outside the lock to avoid holding it during H2D copy.
    let d_slice = graph
        .stream_arc()
        .clone_htod(data)
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod f32: {e}")))?;
    let arc = Arc::new(d_slice);

    let mut cache = state
        .f32_weight_cache
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;
    cache.insert(key, Arc::clone(&arc));
    Ok(arc)
}

// =============================================================================
// Per-process model weight cache
// =============================================================================

/// Build (or return the already-cached) GPU weight handles for all transformer layers.
///
/// On the **first call** this uploads all Q1/FP32 weights to GPU memory, wraps them in
/// `Arc<Vec<CudaCachedLayerWeights>>`, and stores the result in `FULL_LAYER_STATE`.
///
/// On **subsequent calls** — i.e., every decode token after the first — only three
/// `Arc::clone()` operations are performed (O(1)).  This replaces the previous
/// `try_cuda_full_forward` behaviour of doing 288+ `HashMap` lookups + mutex
/// acquisitions every token.
pub(super) fn get_or_build_model_weights(
    layer_params: &[CudaFullForwardLayerParams<'_>],
) -> Option<(Arc<CudaGraph>, Arc<Vec<CudaCachedLayerWeights>>)> {
    let n_layers = layer_params.len();
    let state = full_layer_state();

    // Fast path: cache hit — three Arc::clones, no HashMap access.
    {
        let guard = state.cached_model_weights.lock().ok()?;
        if let Some(ref cmw) = *guard {
            if cmw.n_layers == n_layers {
                return Some((Arc::clone(&cmw.graph), Arc::clone(&cmw.layers)));
            }
        }
    }

    // Slow path: first call (or model changed).  Build and cache.
    let graph = CudaGraph::global().ok()?;
    let dummy_weight = Arc::new(graph.stream_arc().alloc_zeros::<u8>(1).ok()?);

    let mut cached: Vec<CudaCachedLayerWeights> = Vec::with_capacity(n_layers);
    for lp in layer_params {
        let q_weight = graph
            .get_or_upload_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)
            .ok()?;
        let o_weight = graph
            .get_or_upload_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)
            .ok()?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_weight = graph
            .get_or_upload_weight_soa_lazy(lp.gate_up_handle, || {
                let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
                fused.extend_from_slice(gate_bytes);
                fused.extend_from_slice(up_bytes);
                fused
            })
            .ok()?;
        let down_weight = graph
            .get_or_upload_weight_soa(lp.down_handle, lp.down_bytes)
            .ok()?;
        let pre_attn_norm =
            get_or_upload_f32_weight(&graph, lp.attn_norm_handle, lp.attn_norm_bytes).ok()?;
        let post_attn_norm =
            get_or_upload_f32_weight(&graph, lp.ffn_norm_handle, lp.ffn_norm_bytes).ok()?;
        let q_norm = get_or_upload_f32_weight(&graph, lp.q_norm_handle, lp.q_norm_bytes).ok()?;
        let k_norm = get_or_upload_f32_weight(&graph, lp.k_norm_handle, lp.k_norm_bytes).ok()?;

        cached.push(CudaCachedLayerWeights {
            q_weight,
            k_weight: Arc::clone(&dummy_weight),
            v_weight: Arc::clone(&dummy_weight),
            o_weight,
            gate_up_weight,
            down_weight,
            pre_attn_norm,
            post_attn_norm,
            q_norm,
            k_norm,
        });
    }

    let layers = Arc::new(cached);
    let cmw = CudaCachedModelWeights {
        graph: Arc::clone(&graph),
        dummy_weight,
        layers: Arc::clone(&layers),
        n_layers,
    };

    if let Ok(mut guard) = state.cached_model_weights.lock() {
        *guard = Some(cmw);
    }

    Some((graph, layers))
}

// =============================================================================
// Attention module lazy init
// =============================================================================

/// Compile and cache the 7 CUDA attention kernels.
///
/// Idempotent: on the second call the already-compiled modules are returned
/// immediately from the `Mutex<Option<...>>` cache.
pub fn init_attn_modules(graph: &CudaGraph) -> Result<Arc<CudaAttnModules>, CudaGraphError> {
    let state = full_layer_state();
    let mut guard = state
        .attn_modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    // Compile all 7 attention kernels in a single NVRTC call (disk-cached after first run).
    let ptx = compile_or_load_ptx(CUDA_ATTENTION_KERNELS_SRC, "attn_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module attn: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let modules = Arc::new(CudaAttnModules {
        fused_qk_norm: load("fused_qk_norm")?,
        fused_qk_rope: load("fused_qk_rope")?,
        fused_qk_norm_rope: load("fused_qk_norm_rope")?,
        fused_kv_store: load("fused_kv_store")?,
        batched_attn_scores_v2: load("batched_attn_scores_v2")?,
        batched_softmax: load("batched_softmax")?,
        batched_attn_weighted_sum: load("batched_attn_weighted_sum")?,
    });

    *guard = Some(Arc::clone(&modules));
    Ok(modules)
}

// =============================================================================
// Buffer / cache acquisition helpers
// =============================================================================

/// Acquire or (re-)allocate the full-layer activation buffers.
pub(super) fn acquire_full_layer_buffers(
    graph: &CudaGraph,
    hidden_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
    intermediate_size: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaFullLayerBuffers>>, CudaGraphError> {
    let state = full_layer_state();
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
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros fl({n}): {e}")))
        };

        let qkv_total = nq * head_dim + 2 * nkv * head_dim;
        let half_dim = head_dim / 2;

        let alloc_u32 = |n: usize| -> Result<CudaSlice<u32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<u32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros u32({n}): {e}")))
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
        // Buffer dimensions changed → invalidate any captured CUDA driver graph.
        if let Ok(mut g) = full_layer_state().cuda_driver_graph.lock() {
            *g = None;
        }
    }

    Ok(guard)
}

/// Acquire or (re-)allocate the GPU KV cache.
pub(super) fn acquire_kv_cache(
    graph: &CudaGraph,
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaKvCache>>, CudaGraphError> {
    let state = full_layer_state();
    let mut guard = state
        .kv_cache
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(c) => !c.matches(n_layers, n_kv, max_seq, head_dim),
        None => true,
    };

    if needs_alloc {
        let total_elements = n_layers * n_kv * max_seq * head_dim;
        let k_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total_elements)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv k_cache: {e}")))?;
        let v_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total_elements)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv v_cache: {e}")))?;

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

// =============================================================================
// encode_attn_phase
// =============================================================================

/// Encode the full attention sublayer on the CUDA stream (steps 1-7).
///
/// On return `bufs.d_attn_out` holds `[nq * head_dim]` attention output values.
///
/// # Safety
/// The function launches CUDA kernels.  The caller must ensure all GPU state
/// is valid and the stream is not concurrently used.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_attn_phase(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_pre_norm_weight: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_q_norm_weight: &CudaSlice<f32>,
    d_k_norm_weight: &CudaSlice<f32>,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    _pos: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    hidden_size: usize,
    bufs: &mut CudaFullLayerBuffers,
) -> Result<(), CudaGraphError> {
    let h_u32 = hidden_size as u32;
    let nq_u32 = nq as u32;
    let nkv_u32 = nkv as u32;
    let hd_u32 = head_dim as u32;
    let qkv_total_rows = (nq * head_dim + 2 * nkv * head_dim) as u32;
    let heads_per_group_u32 = heads_per_group as u32;
    let max_seq_u32 = bufs.max_seq as u32;
    let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
    let layer_offset = kv.layer_offset_elements(layer_idx);

    // Step 1: RMSNorm(d_hidden, norm_weight -> d_normed)
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        d_pre_norm_weight,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;

    // Step 2: Fused QKV GEMV (normed -> d_qkv)
    graph.launch_gemv_pub(
        d_fused_qkv_weight,
        &bufs.d_normed,
        &mut bufs.d_qkv,
        qkv_total_rows,
        h_u32,
    )?;

    // Step 3: Fused QK-Norm + RoPE
    // Q occupies elements [0 .. nq*head_dim] in d_qkv.
    // K occupies elements [nq*head_dim .. (nq+nkv)*head_dim] in d_qkv.
    let k_offset = nq * head_dim;
    let k_in_view = bufs.d_qkv.slice(k_offset..);
    launch_fused_qk_norm_rope(
        graph,
        mods,
        &bufs.d_qkv, // q_in: first nq*head_dim elements
        &k_in_view,  // k_in: starts at nq*head_dim
        &mut bufs.d_q_rope,
        &mut bufs.d_k_rope,
        d_q_norm_weight,
        d_k_norm_weight,
        &bufs.d_cos,
        &bufs.d_sin,
        nq_u32,
        nkv_u32,
        hd_u32,
        norm_eps,
    )?;

    // Step 4: Fused KV-Store — pos read from d_pos_seqlen[0] by the kernel
    let v_offset = (nq + nkv) * head_dim;
    let v_view = bufs.d_qkv.slice(v_offset..);
    launch_fused_kv_store(
        graph,
        mods,
        &bufs.d_k_rope,
        &v_view,
        &mut kv.k_cache,
        &mut kv.v_cache,
        hd_u32,
        nkv_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )?;

    // Step 5: Batched attention scores V2 — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_scores_v2(
        graph,
        mods,
        &bufs.d_q_rope,
        &kv.k_cache,
        &mut bufs.d_scores,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        inv_sqrt_hd,
        layer_offset,
    )?;

    // Step 6: Softmax — seq_len read from d_pos_seqlen[1]
    launch_batched_softmax(
        graph,
        mods,
        &mut bufs.d_scores,
        nq_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
    )?;

    // Step 7: Weighted sum — seq_len read from d_pos_seqlen[1]
    launch_batched_attn_weighted_sum(
        graph,
        mods,
        &bufs.d_scores,
        &kv.v_cache,
        &mut bufs.d_attn_out,
        hd_u32,
        nq_u32,
        nkv_u32,
        heads_per_group_u32,
        max_seq_u32,
        &bufs.d_pos_seqlen,
        layer_offset,
    )
}

// =============================================================================
// Re-exports from encode_q1
// =============================================================================

pub use encode_q1::{
    encode_full_forward, encode_full_layer, try_cuda_full_forward,
    try_cuda_full_forward_with_gpu_lm_head, try_cuda_full_layer,
};

// =============================================================================
// Re-exports from encode_ternary
// =============================================================================

pub use encode_ternary::{
    encode_full_forward_ternary, encode_layer_into_ternary, encode_lm_head_gemv_ternary,
    try_cuda_full_forward_ternary, try_cuda_full_forward_ternary_with_gpu_lm_head,
    CudaFullForwardLayerParamsTernary,
};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
