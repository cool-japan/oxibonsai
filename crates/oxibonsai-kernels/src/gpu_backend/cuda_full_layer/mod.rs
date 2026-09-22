//! Full-layer GPU dispatch for OxiBonsai — CUDA backend.
//!
//! Mirrors `metal_full_layer` for Linux/Windows, encoding the complete
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
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView};
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
use super::cuda_device_negotiation::{
    check_cuda_kv_cache_geometry, cuda_kv_layer_offset_elements, prefill_chunk_token_ranges,
};
use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};

use super::cuda_graph_slot::{
    cuda_graph_slot_action, next_cuda_model_epoch, CudaGraphSlotAction, CudaGraphSlotKey,
    CudaQuantKind,
};

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
    /// Element offset of `layer_idx`'s slab (finding **F4**).
    ///
    /// Returns `u64`. This used to be `(layer_idx * n_kv * max_seq * head_dim)
    /// as u32`, and `fused_kv_store` / `batched_attn_scores_v2` /
    /// `batched_attn_weighted_sum` all took the value as `unsigned int` and only
    /// widened it to 64 bits *after* the truncation had already happened. Above
    /// `2^32` elements — roughly 17 GB of KV cache, so 80 GB-class hardware at
    /// 128 K context and above — two layers aliased onto one address range with no
    /// error anywhere. The kernels now take `unsigned long long`, so there is no
    /// 32-bit ceiling left on this path (unlike the Metal twin, whose MSL
    /// bindings are still `constant uint&`).
    ///
    /// The arithmetic itself lives in
    /// [`cuda_kv_layer_offset_elements`](super::cuda_device_negotiation::cuda_kv_layer_offset_elements)
    /// so it is unit-tested on hosts that cannot compile this module.
    #[inline]
    pub fn layer_offset_elements(&self, layer_idx: usize) -> u64 {
        cuda_kv_layer_offset_elements(layer_idx, self.n_kv, self.max_seq, self.head_dim)
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
    /// `[hidden_size]` residual stream
    pub d_hidden: CudaSlice<f32>,
    /// `[hidden_size]` RMSNorm output / O-proj scratch
    pub d_normed: CudaSlice<f32>,
    /// [nq*hd + 2*nkv*hd] fused QKV GEMV output
    pub d_qkv: CudaSlice<f32>,
    /// [nq * head_dim] Q after norm+RoPE
    pub d_q_rope: CudaSlice<f32>,
    /// [nkv * head_dim] K after norm+RoPE
    pub d_k_rope: CudaSlice<f32>,
    /// `[half_dim]` RoPE cosines
    pub d_cos: CudaSlice<f32>,
    /// `[half_dim]` RoPE sines
    pub d_sin: CudaSlice<f32>,
    /// [nq * max_seq] attention scores
    pub d_scores: CudaSlice<f32>,
    /// [nq * head_dim] attention output
    pub d_attn_out: CudaSlice<f32>,
    /// [2 * intermediate_size] gate+up GEMV
    pub d_gate_up: CudaSlice<f32>,
    /// `[intermediate_size]` SwiGLU output
    pub d_swiglu: CudaSlice<f32>,
    /// `[2]` pos/seq_len for CUDA-graph-captured attention kernels: `[pos, seq_len]`
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
/// Mirrors `FullForwardLayerParams` in `metal_full_layer` so callers can
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
    /// Epoch minted by [`next_cuda_model_epoch`] when this weight set was
    /// uploaded (finding **F-M1**/**F-M3**). It both keys the captured CUDA
    /// graph and attributes every uploaded weight, so the model's `Drop` can
    /// release them.
    pub model_epoch: u64,
    /// [`CudaGraphSlotKey::fingerprint_handles`] over this model's per-layer
    /// weight handle ids, in upload order — the identity of the device pointers
    /// a captured graph baked in. Precomputed here so the per-token key
    /// comparison costs nothing.
    pub weight_fingerprint: u64,
}

/// Everything the encode paths need back from the weight-cache builders
/// (finding **F-M1**): the uploaded weights plus the two identity fields the
/// captured-graph slot key is built from.
pub(super) struct CudaResolvedModelWeights {
    /// The process-wide CUDA singleton the weights live on.
    pub graph: Arc<CudaGraph>,
    /// Per-layer uploaded weights.
    pub layers: Arc<Vec<CudaCachedLayerWeights>>,
    /// See [`CudaCachedModelWeights::model_epoch`].
    pub model_epoch: u64,
    /// See [`CudaCachedModelWeights::weight_fingerprint`].
    pub weight_fingerprint: u64,
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
    /// Device-resident per-token RoPE/position inputs for one prefill chunk
    /// (finding **F9**). Held in its own slot rather than inside
    /// [`CudaFullLayerBuffers`] because the per-token `CudaView`s built from it
    /// are alive at the same time as the `&mut CudaFullLayerBuffers` that
    /// `encode_attn_phase` takes.
    prefill_rope_chunk: Mutex<Option<CudaPrefillRopeChunk>>,

    /// Cache for FP32 norm weights (separate from the Q1 u8 weight cache).
    f32_weight_cache: Mutex<HashMap<u64, Arc<CudaSlice<f32>>>>,
    /// Cached GPU model weights for the ternary (TQ2) decode path, paired with a
    /// content fingerprint of the source weight bytes — rebuilt only when the
    /// model changes.  Validated by BOTH the fingerprint AND the layer count so a
    /// same-depth ternary model swap rebuilds rather than silently reusing another
    /// model's uploaded GPU buffers (see
    /// `encode_ternary::get_or_build_ternary_model_weights`).
    cached_model_weights: Mutex<Option<(u64, CudaCachedModelWeights)>>,
    /// Cached GPU model weights for the Q1 decode path, paired with a content
    /// fingerprint of the source weight bytes.  Distinct slot from
    /// `cached_model_weights` so Q1 and TQ2 can never alias, and the fingerprint
    /// guards against returning a *different* same-depth model's uploaded buffers
    /// (e.g. two same-`n_layers` finetunes) — see `get_or_build_model_weights`.
    cached_q1_model_weights: Mutex<Option<(u64, CudaCachedModelWeights)>>,
    /// Captured CUDA driver graph for replaying the N-layer pipeline, **keyed**
    /// on the model it was captured for (finding **F-M1**).
    ///
    /// Three-state, each state carrying the [`CudaGraphSlotKey`] it belongs to:
    /// - `None`                    → capture not yet attempted
    /// - `Some((key, None))`       → capture was attempted for `key` and failed
    ///   (no retry for that key)
    /// - `Some((key, Some(h)))`    → capture for `key` succeeded; `h` is the
    ///   exec graph
    ///
    /// The key is what makes the slot safe to share between the Q1 and ternary
    /// decode paths. A captured `CUgraphExec` bakes in the per-layer weight
    /// device pointers, and the only invalidation used to be "activation-buffer
    /// dimensions changed" — but `Bonsai-8B` (Q1) and `Ternary-Bonsai-8B` (TQ2)
    /// have identical dimensions, so loading one after the other reused the
    /// buffers, skipped invalidation and replayed the **first** model's weights.
    /// Replay now requires an exact key match ([`cuda_graph_slot_action`]) and a
    /// mismatch drops the holder, freeing the exec and forcing a re-capture.
    ///
    /// Still reset to `None` when activation-buffer dimensions
    /// (`acquire_full_layer_buffers`) or the KV-cache shape (`acquire_kv_cache`)
    /// change, since the graph embeds pointers into both.
    cuda_driver_graph: Mutex<Option<(CudaGraphSlotKey, Option<CuGraphHolder>)>>,
}

unsafe impl Send for CudaFullLayerState {}
unsafe impl Sync for CudaFullLayerState {}

static FULL_LAYER_STATE: OnceLock<CudaFullLayerState> = OnceLock::new();

fn full_layer_state() -> &'static CudaFullLayerState {
    FULL_LAYER_STATE.get_or_init(|| CudaFullLayerState {
        attn_modules: Mutex::new(None),
        full_layer_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        prefill_rope_chunk: Mutex::new(None),

        f32_weight_cache: Mutex::new(HashMap::new()),
        cached_model_weights: Mutex::new(None),
        cached_q1_model_weights: Mutex::new(None),
        // None = not yet attempted; Some((key, None)) = tried & failed for that
        // key; Some((key, Some(h))) = active capture for that key.
        cuda_driver_graph: Mutex::new(None),
    })
}

/// Free the GPU weights of the model a cache slot was holding, before a
/// different model's weights are uploaded into the same slot (finding **F-M3**).
///
/// This is what gives `release_model_epoch` a consumer on the full-forward path,
/// and it is also a **correctness** fix, not only a VRAM one. The CUDA decode
/// paths derive their handle ids from the layer index alone
/// (`oxibonsai-model`'s `forward_cuda/ternary.rs`: `6_000_000 + layer * 10`), so
/// two ternary models of equal depth produce **identical** handle ids: after a
/// swap, `get_or_upload_weight_tq2_soa` hit the previous model's cache entry and
/// handed back its device buffer, and the rebuilt `CudaCachedLayerWeights` was
/// a fresh struct pointing at stale weights. Evicting the previous epoch first
/// makes the miss real, so the new model's bytes are actually uploaded. It also
/// keeps peak VRAM at one model rather than two.
///
/// Best-effort: a failure here is logged, never propagated, because the caller's
/// job (building this model's weights) can still succeed.
fn evict_cached_model_weights(previous: Option<(u64, CudaCachedModelWeights)>) {
    let Some((_, cmw)) = previous else {
        return;
    };
    let graph = Arc::clone(&cmw.graph);
    let model_epoch = cmw.model_epoch;
    // Drop this slot's own `Arc`s before the caches drop theirs, so the device
    // memory is actually released rather than kept alive by the cache entry.
    drop(cmw);
    // F-M1: the captured graph embeds this model's weight device pointers, which
    // are about to be freed. The slot key would force a re-capture anyway (the
    // next weight set mints a new `model_epoch`), but that makes the safety of a
    // freed pointer depend on epoch-minting order. Invalidate explicitly, for
    // the same reason `acquire_kv_cache` and `acquire_full_layer_buffers` do:
    // if the rebuild below fails partway (no device, a failed upload) the slot
    // would otherwise keep an exec pointing into released memory.
    if let Ok(mut g) = full_layer_state().cuda_driver_graph.lock() {
        *g = None;
    }
    match graph.release_model_epoch(model_epoch) {
        Ok(released) => {
            tracing::debug!("evicted {released} CUDA weights of model epoch {model_epoch}");
        }
        Err(e) => {
            tracing::warn!("could not evict CUDA weights of model epoch {model_epoch}: {e}");
        }
    }
}

/// Build the captured-graph slot key for one full-forward call (finding
/// **F-M1**).
///
/// `weight_fingerprint` is the per-layer handle fingerprint the weight-cache
/// builder precomputed ([`CudaCachedModelWeights::weight_fingerprint`]); it is
/// combined here with `final_norm_handle`, whose device pointer the capture also
/// bakes in but which is passed separately to the encode paths.
///
/// Dimensions are saturated rather than truncated into the key's `u32` fields:
/// a silent wraparound could make two different models compare equal, which is
/// exactly the class of bug this key exists to prevent.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_slot_key(
    quant_kind: CudaQuantKind,
    model_epoch: u64,
    weight_fingerprint: u64,
    final_norm_handle: u64,
    n_layers: usize,
    hidden_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
    intermediate_size: usize,
) -> CudaGraphSlotKey {
    fn dim(v: usize) -> u32 {
        u32::try_from(v).unwrap_or(u32::MAX)
    }
    CudaGraphSlotKey {
        model_epoch,
        quant_kind,
        n_layers: dim(n_layers),
        hidden_size: dim(hidden_size),
        n_q_heads: dim(nq),
        n_kv_heads: dim(nkv),
        head_dim: dim(head_dim),
        max_seq: dim(max_seq),
        intermediate_size: dim(intermediate_size),
        weight_fingerprint: CudaGraphSlotKey::fingerprint_handles(&[
            weight_fingerprint,
            final_norm_handle,
        ]),
    }
}

/// Decide what to do with the captured-graph slot for `requested`, dropping a
/// stale capture (freeing its `CUgraphExec`) on the spot (finding **F-M1**).
///
/// The decision itself is [`cuda_graph_slot_action`], which lives in the ungated
/// `gpu_backend::cuda_graph_slot` so its branch table is unit-tested on hosts
/// where this module is not compiled at all.
fn slot_action_dropping_stale(
    guard: &mut Option<(CudaGraphSlotKey, Option<CuGraphHolder>)>,
    requested: &CudaGraphSlotKey,
) -> CudaGraphSlotAction {
    // Copy the key out first: `CudaGraphSlotKey` is `Copy`, so the borrow ends
    // before the slot may be cleared below.
    let stored = guard.as_ref().map(|(key, holder)| (*key, holder.is_some()));
    let action = cuda_graph_slot_action(stored, requested);
    if action == CudaGraphSlotAction::Capture {
        // Drops `CuGraphHolder`, which destroys the exec and the graph.
        *guard = None;
    }
    action
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

/// Upload f32 weights and cache them for reuse, **unattributed**.
///
/// On the first call for `key`, the slice is uploaded to GPU device memory
/// and cached.  Subsequent calls return the cached `Arc<CudaSlice<f32>>`.
///
/// Not attributed to a model epoch, so never released by
/// `CudaGraph::release_model_epoch` (finding F-M3) — use
/// [`get_or_upload_f32_weight_for_epoch`] when the owning model is known.
pub fn get_or_upload_f32_weight(
    graph: &CudaGraph,
    key: u64,
    data: &[f32],
) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
    get_or_upload_f32_weight_for_epoch(
        graph,
        key,
        data,
        super::cuda_graph_slot::UNATTRIBUTED_CUDA_MODEL_EPOCH,
    )
}

/// [`get_or_upload_f32_weight`], attributing the upload to `model_epoch` so the
/// model's `Drop` can free it (finding **F-M3**).
///
/// Note this is a **different** cache from `CudaGraph::f32_weight_cache`; both
/// are walked by `CudaGraph::release_weights`.
pub fn get_or_upload_f32_weight_for_epoch(
    graph: &CudaGraph,
    key: u64,
    data: &[f32],
    model_epoch: u64,
) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
    let state = full_layer_state();
    let cached = {
        let cache = state
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.get(&key).map(Arc::clone)
    };
    if let Some(existing) = cached {
        // Register on a hit too: the first upload may have been unattributed,
        // and registration is idempotent.
        graph.register_model_weight(model_epoch, key)?;
        return Ok(existing);
    }

    // Upload outside the lock to avoid holding it during H2D copy.
    let d_slice = graph
        .stream_arc()
        .clone_htod(data)
        .map_err(|e| CudaGraphError::DriverError(format!("clone_htod f32: {e}")))?;
    let arc = Arc::new(d_slice);

    {
        let mut cache = state
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.insert(key, Arc::clone(&arc));
    }
    // Outside the cache lock: the lock order is always "weight cache, then
    // epoch registry".
    graph.register_model_weight(model_epoch, key)?;
    Ok(arc)
}

/// Drop the given handles from the full-layer FP32 norm cache and report
/// `(entries removed, bytes freed)` (finding **F-M3**).
///
/// Called by `CudaGraph::release_weights`, which owns the other two caches; this
/// one is a process-local map here, so the release path must reach into it
/// explicitly or every norm weight of a dropped model stays resident.
pub(super) fn release_f32_weights(handles: &[u64]) -> Result<(usize, usize), CudaGraphError> {
    let mut released = 0usize;
    let mut freed_bytes = 0usize;
    let mut cache = full_layer_state()
        .f32_weight_cache
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;
    for handle in handles {
        if let Some(slice) = cache.remove(handle) {
            freed_bytes += slice.len() * std::mem::size_of::<f32>();
            released += 1;
        }
    }
    Ok((released, freed_bytes))
}

// =============================================================================
// Per-process model weight cache
// =============================================================================

/// Cheap content-and-identity fingerprint of a Q1 model's whole weight set.
///
/// Combines each layer's weight/norm source-slice base pointer + length into an
/// FNV-1a hash.  The pointers are stable for the lifetime of a loaded model
/// (they reference the model's owned / mmap'd bytes or its cached QKV concats)
/// and differ across concurrently-loaded models, so a same-`n_layers` model swap
/// (e.g. two same-depth finetunes) produces a different fingerprint.  Cost is
/// O(n_layers) with a tiny constant, cheap enough to run on every decode token.
///
/// Residual edge case (accepted for this defense-in-depth check): if one model is
/// dropped and a different one is loaded that happens to reuse the *exact* same
/// base addresses and lengths for every layer, the fingerprints collide.  This is
/// effectively impossible in practice and only matters in a (currently
/// unsupported) multi-model-per-process configuration.
fn model_weights_fingerprint(layer_params: &[CudaFullForwardLayerParams<'_>]) -> u64 {
    let mut h = 0xcbf29ce484222325u64; // FNV-1a offset basis
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    };
    mix(layer_params.len() as u64);
    for lp in layer_params {
        let parts: [(u64, u64); 7] = [
            (
                lp.attn_norm_bytes.as_ptr() as usize as u64,
                lp.attn_norm_bytes.len() as u64,
            ),
            (
                lp.fused_qkv_bytes.as_ptr() as usize as u64,
                lp.fused_qkv_bytes.len() as u64,
            ),
            (
                lp.attn_proj_bytes.as_ptr() as usize as u64,
                lp.attn_proj_bytes.len() as u64,
            ),
            (
                lp.gate_bytes.as_ptr() as usize as u64,
                lp.gate_bytes.len() as u64,
            ),
            (
                lp.up_bytes.as_ptr() as usize as u64,
                lp.up_bytes.len() as u64,
            ),
            (
                lp.down_bytes.as_ptr() as usize as u64,
                lp.down_bytes.len() as u64,
            ),
            (
                lp.ffn_norm_bytes.as_ptr() as usize as u64,
                lp.ffn_norm_bytes.len() as u64,
            ),
        ];
        for (ptr, len) in parts {
            mix(ptr);
            mix(len);
        }
    }
    h
}

/// Build (or return the already-cached) GPU weight handles for all transformer layers.
///
/// On the **first call** this uploads all Q1/FP32 weights to GPU memory, wraps them in
/// `Arc<Vec<CudaCachedLayerWeights>>`, and stores the result in `FULL_LAYER_STATE`.
///
/// On **subsequent calls** — i.e., every decode token after the first — only three
/// `Arc::clone()` operations are performed (O(1)).  This replaces the previous
/// `try_cuda_full_forward` behaviour of doing 288+ `HashMap` lookups + mutex
/// acquisitions every token.
///
/// The cache is validated by BOTH the layer count and a content fingerprint
/// ([`model_weights_fingerprint`]) so a same-depth model swap rebuilds rather than
/// silently reusing another model's uploaded GPU buffers.  It uses a Q1-only slot
/// (`cached_q1_model_weights`), disjoint from the ternary slot, so Q1 and TQ2
/// weight sets can never alias even at equal `n_layers`.
pub(super) fn get_or_build_model_weights(
    layer_params: &[CudaFullForwardLayerParams<'_>],
) -> Option<CudaResolvedModelWeights> {
    let n_layers = layer_params.len();
    let fingerprint = model_weights_fingerprint(layer_params);
    let state = full_layer_state();

    // Fast path: cache hit — three Arc::clones, no HashMap access.  Both the
    // fingerprint AND the layer count must match to reuse the cached buffers.
    {
        let guard = state.cached_q1_model_weights.lock().ok()?;
        if let Some((fp, cmw)) = guard.as_ref() {
            if *fp == fingerprint && cmw.n_layers == n_layers {
                return Some(CudaResolvedModelWeights {
                    graph: Arc::clone(&cmw.graph),
                    layers: Arc::clone(&cmw.layers),
                    model_epoch: cmw.model_epoch,
                    weight_fingerprint: cmw.weight_fingerprint,
                });
            }
        }
    }

    // Slow path: first call (or model changed).  Build and cache.
    // F-M3: free the previous model's GPU weights *before* uploading this one —
    // see `evict_cached_model_weights` for why the order matters.
    let previous = state
        .cached_q1_model_weights
        .lock()
        .ok()
        .and_then(|mut guard| guard.take());
    evict_cached_model_weights(previous);

    let graph = CudaGraph::global().ok()?;
    let dummy_weight = Arc::new(graph.stream_arc().alloc_zeros::<u8>(1).ok()?);
    // A fresh epoch per uploaded weight set (findings F-M1 + F-M3): it keys the
    // captured graph and attributes every upload below, so a later
    // `release_model_epoch` frees exactly this model's device memory.
    let model_epoch = next_cuda_model_epoch();
    let mut handle_ids: Vec<u64> = Vec::with_capacity(n_layers * 8);

    let mut cached: Vec<CudaCachedLayerWeights> = Vec::with_capacity(n_layers);
    for lp in layer_params {
        handle_ids.extend_from_slice(&[
            lp.fused_qkv_handle,
            lp.attn_proj_handle,
            lp.gate_up_handle,
            lp.down_handle,
            lp.attn_norm_handle,
            lp.ffn_norm_handle,
            lp.q_norm_handle,
            lp.k_norm_handle,
        ]);
        let q_weight = graph
            .get_or_upload_weight_soa_for_epoch(
                lp.fused_qkv_handle,
                lp.fused_qkv_bytes,
                model_epoch,
            )
            .ok()?;
        let o_weight = graph
            .get_or_upload_weight_soa_for_epoch(
                lp.attn_proj_handle,
                lp.attn_proj_bytes,
                model_epoch,
            )
            .ok()?;
        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_weight = graph
            .get_or_upload_weight_soa_lazy_for_epoch(
                lp.gate_up_handle,
                || {
                    let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
                    fused.extend_from_slice(gate_bytes);
                    fused.extend_from_slice(up_bytes);
                    fused
                },
                model_epoch,
            )
            .ok()?;
        let down_weight = graph
            .get_or_upload_weight_soa_for_epoch(lp.down_handle, lp.down_bytes, model_epoch)
            .ok()?;
        let pre_attn_norm = get_or_upload_f32_weight_for_epoch(
            &graph,
            lp.attn_norm_handle,
            lp.attn_norm_bytes,
            model_epoch,
        )
        .ok()?;
        let post_attn_norm = get_or_upload_f32_weight_for_epoch(
            &graph,
            lp.ffn_norm_handle,
            lp.ffn_norm_bytes,
            model_epoch,
        )
        .ok()?;
        let q_norm = get_or_upload_f32_weight_for_epoch(
            &graph,
            lp.q_norm_handle,
            lp.q_norm_bytes,
            model_epoch,
        )
        .ok()?;
        let k_norm = get_or_upload_f32_weight_for_epoch(
            &graph,
            lp.k_norm_handle,
            lp.k_norm_bytes,
            model_epoch,
        )
        .ok()?;

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
    let weight_fingerprint = CudaGraphSlotKey::fingerprint_handles(&handle_ids);
    let cmw = CudaCachedModelWeights {
        graph: Arc::clone(&graph),
        dummy_weight,
        layers: Arc::clone(&layers),
        n_layers,
        model_epoch,
        weight_fingerprint,
    };

    // Store fingerprint + weights together (atomic under one lock) so a stale
    // fingerprint can never be paired with the wrong cached buffers.
    if let Ok(mut guard) = state.cached_q1_model_weights.lock() {
        *guard = Some((fingerprint, cmw));
    }

    Some(CudaResolvedModelWeights {
        graph,
        layers,
        model_epoch,
        weight_fingerprint,
    })
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
        // F4: this was `n_layers * n_kv * max_seq * head_dim` in `usize` with
        // no overflow check at all, so an absurd `--max-seq-len` could wrap
        // before `alloc_zeros` ever saw it and silently allocate a cache far
        // smaller than every later index assumes.
        // `CudaGraphError` has no `InvalidDimensions` variant and
        // `cuda_graph/types.rs` is outside this package's file grant, so the
        // rejection travels as a prefixed `DriverError` (see deviations).
        // CUDA is unvalidated: this host has no CUDA hardware, so the check
        // below has only ever been exercised as plain host-side arithmetic.
        let total_elements = check_cuda_kv_cache_geometry(n_layers, n_kv, max_seq, head_dim)
            .map_err(|e| CudaGraphError::DriverError(format!("invalid KV geometry: {e}")))?
            as usize;

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
        // F-M1: the captured graph embeds raw `k_cache`/`v_cache` device
        // pointers, so a reallocation here leaves it reading freed memory.
        // `acquire_full_layer_buffers` has always invalidated on its own
        // reallocation; this branch did not, which is the second variant of the
        // finding (two models with equal per-layer dimensions but different
        // `n_layers` reallocate the KV cache while the activation buffers match).
        if let Ok(mut g) = full_layer_state().cuda_driver_graph.lock() {
            *g = None;
        }
    }

    Ok(guard)
}

// =============================================================================
// Prefill chunk inputs (finding F9)
// =============================================================================

/// Device-resident per-token RoPE tables and `[pos, pos+1]` pairs for one
/// prefill chunk (finding **F9**).
///
/// `encode_prefill_layer` used to issue **five** sub-kilobyte copies per token
/// per layer: a device-to-device copy of the hidden column, three
/// host-to-device uploads (`[pos, pos+1]`, the RoPE cosines, the RoPE sines),
/// and a device-to-device copy of the attention output. The three uploads
/// carry *layer-invariant* data, so they were also repeated `n_layers` times
/// over. Staging them here turns `3 x n_tokens` uploads per layer into exactly
/// three, and the per-token attention step then reads
/// [`CudaView`]s into this buffer instead of a freshly uploaded scratch slot.
///
/// The `[pos, pos+1]` pairs stay strictly per token: `encode_prefill_layer`
/// documents that a stale position corrupts the *shared* decode KV cache, so
/// the hoist must not collapse them into one value.
pub struct CudaPrefillRopeChunk {
    /// `[2 * n_tokens]` — `[pos, pos + 1]` for each token, in order.
    d_pos_seqlen: CudaSlice<u32>,
    /// `[n_tokens * half_dim]` RoPE cosines, token-major.
    d_cos: CudaSlice<f32>,
    /// `[n_tokens * half_dim]` RoPE sines, token-major.
    d_sin: CudaSlice<f32>,
    /// Token capacity currently allocated.
    capacity_tokens: usize,
    /// `half_dim` the current allocation was sized for.
    capacity_half_dim: usize,
    /// Tokens actually uploaded by the last [`Self::upload`].
    n_tokens: usize,
    /// `half_dim` of the last [`Self::upload`].
    half_dim: usize,
}

impl CudaPrefillRopeChunk {
    /// Upload one chunk's positions and RoPE tables in exactly three copies.
    ///
    /// `cos_table` / `sin_table` are `[n_tokens * half_dim]`, token-major, as
    /// built by the model side.
    ///
    /// # Errors
    /// Mismatched table lengths, or a device allocation / copy failure.
    fn upload(
        &mut self,
        graph: &CudaGraph,
        pos_start: usize,
        n_tokens: usize,
        half_dim: usize,
        cos_table: &[f32],
        sin_table: &[f32],
    ) -> Result<(), CudaGraphError> {
        let need = n_tokens.checked_mul(half_dim).ok_or_else(|| {
            CudaGraphError::DriverError(format!(
                "prefill rope chunk: n_tokens {n_tokens} x half_dim {half_dim} overflows"
            ))
        })?;
        if cos_table.len() < need || sin_table.len() < need {
            return Err(CudaGraphError::DriverError(format!(
                "prefill rope chunk: need {need} RoPE elements for {n_tokens} tokens at \
                 half_dim {half_dim}, got cos={} sin={}",
                cos_table.len(),
                sin_table.len()
            )));
        }
        if n_tokens > self.capacity_tokens || half_dim > self.capacity_half_dim {
            let tokens = n_tokens.max(self.capacity_tokens);
            let hd = half_dim.max(self.capacity_half_dim);
            let rope_len = tokens.checked_mul(hd).ok_or_else(|| {
                CudaGraphError::DriverError(format!(
                    "prefill rope chunk: capacity {tokens} x {hd} overflows"
                ))
            })?;
            self.d_pos_seqlen = graph
                .stream_arc()
                .alloc_zeros::<u32>(tokens * 2)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc chunk pos: {e}")))?;
            self.d_cos = graph
                .stream_arc()
                .alloc_zeros::<f32>(rope_len)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc chunk cos: {e}")))?;
            self.d_sin = graph
                .stream_arc()
                .alloc_zeros::<f32>(rope_len)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc chunk sin: {e}")))?;
            self.capacity_tokens = tokens;
            self.capacity_half_dim = hd;
        }

        // One upload each, for the whole chunk.
        let mut pos_pairs = Vec::with_capacity(n_tokens * 2);
        for t in 0..n_tokens {
            let pos = pos_start + t;
            pos_pairs.push(pos as u32);
            pos_pairs.push((pos + 1) as u32);
        }
        let mut pos_dst = self.d_pos_seqlen.slice_mut(0..n_tokens * 2);
        graph
            .stream_arc()
            .memcpy_htod(&pos_pairs, &mut pos_dst)
            .map_err(|e| CudaGraphError::DriverError(format!("upload chunk pos: {e}")))?;
        let mut cos_dst = self.d_cos.slice_mut(0..need);
        graph
            .stream_arc()
            .memcpy_htod(&cos_table[..need], &mut cos_dst)
            .map_err(|e| CudaGraphError::DriverError(format!("upload chunk cos: {e}")))?;
        let mut sin_dst = self.d_sin.slice_mut(0..need);
        graph
            .stream_arc()
            .memcpy_htod(&sin_table[..need], &mut sin_dst)
            .map_err(|e| CudaGraphError::DriverError(format!("upload chunk sin: {e}")))?;

        self.n_tokens = n_tokens;
        self.half_dim = half_dim;
        Ok(())
    }

    /// Per-token device views into the uploaded chunk.
    ///
    /// # Errors
    /// `t` outside the last uploaded chunk.
    pub fn token(&self, t: usize) -> Result<AttnTokenInputs<'_>, CudaGraphError> {
        // The index arithmetic lives in the ungated
        // `gpu_backend::cuda_device_negotiation` so it is unit-tested on hosts
        // that cannot compile this module.
        let (pos, rope) = prefill_chunk_token_ranges(t, self.n_tokens, self.half_dim)
            .map_err(CudaGraphError::DriverError)?;
        Ok(AttnTokenInputs {
            d_pos_seqlen: self.d_pos_seqlen.slice(pos),
            d_cos: self.d_cos.slice(rope.clone()),
            d_sin: self.d_sin.slice(rope),
        })
    }
}

/// One token's attention inputs, as device views (finding **F9**).
///
/// `d_pos_seqlen` is the two-element `[pos, seq_len]` pair the KV-store and
/// attention kernels read their position and span from; `d_cos` / `d_sin` are
/// that token's `half_dim` RoPE values.
pub struct AttnTokenInputs<'a> {
    /// `[pos, pos + 1]` for this token.
    pub d_pos_seqlen: CudaView<'a, u32>,
    /// `[half_dim]` RoPE cosines.
    pub d_cos: CudaView<'a, f32>,
    /// `[half_dim]` RoPE sines.
    pub d_sin: CudaView<'a, f32>,
}

/// Acquire (and lazily create) the prefill chunk staging buffers, then upload
/// one chunk's positions and RoPE tables into them — finding **F9**.
///
/// The upload happens on every call rather than being memoised across layers:
/// the only key available here is `(pos_start, n_tokens, half_dim)`, and two
/// models with the same geometry but different `rope.freq_base` would share it.
/// Three uploads per layer is already the whole of the finding — it replaces
/// `3 x n_tokens` of them — and it cannot go stale.
///
/// Spec-accurate accounting: `encode_prefill_layer` (the only caller, in the
/// unowned `cuda_prefill` module) calls this once per layer, so one chunk's
/// full prefill issues `3 x n_layers` uploads here, not the `3` a true
/// per-chunk memoisation would give. Closing that gap needs a chunk epoch
/// threaded down from the prefill loop (see the package's recorded
/// deviations); this function's contract does not change either way.
///
/// # Errors
/// Lock poisoning, or any allocation / copy failure.
pub(super) fn acquire_prefill_rope_chunk(
    graph: &CudaGraph,
    pos_start: usize,
    n_tokens: usize,
    half_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) -> Result<std::sync::MutexGuard<'static, Option<CudaPrefillRopeChunk>>, CudaGraphError> {
    let state = full_layer_state();
    let mut guard = state
        .prefill_rope_chunk
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;
    if guard.is_none() {
        *guard = Some(CudaPrefillRopeChunk {
            d_pos_seqlen: graph
                .stream_arc()
                .alloc_zeros::<u32>(2)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc chunk pos: {e}")))?,
            d_cos: graph
                .stream_arc()
                .alloc_zeros::<f32>(1)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc chunk cos: {e}")))?,
            d_sin: graph
                .stream_arc()
                .alloc_zeros::<f32>(1)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc chunk sin: {e}")))?,
            capacity_tokens: 1,
            capacity_half_dim: 1,
            n_tokens: 0,
            half_dim: 0,
        });
    }
    let chunk = guard.as_mut().ok_or_else(|| {
        CudaGraphError::DriverError("prefill rope chunk missing after init".to_string())
    })?;
    chunk.upload(graph, pos_start, n_tokens, half_dim, cos_table, sin_table)?;
    Ok(guard)
}

// =============================================================================
// encode_attn_phase
// =============================================================================

/// Encode the full attention sublayer on the CUDA stream (steps 1-7).
///
/// On return `bufs.d_attn_out` holds `[nq * head_dim]` attention output values.
///
/// `token` selects where the position pair and the RoPE tables come from
/// (finding **F9**):
/// - `None` — the decode path: `bufs.d_pos_seqlen` / `d_cos` / `d_sin`, which
///   the caller has already filled for the single token in flight;
/// - `Some(..)` — the batch-prefill path: views into the chunk-resident
///   buffers of [`CudaPrefillRopeChunk`], uploaded once per layer for all
///   tokens instead of once per token.
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
    token: Option<&AttnTokenInputs<'_>>,
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
    // F9: `None` (decode) uses the single-token scratch slots; `Some`
    // (batch prefill) points at this token's slice of the chunk-resident
    // upload, so nothing has to be re-uploaded per token.
    let fallback_pos = bufs.d_pos_seqlen.slice(0..);
    let fallback_cos = bufs.d_cos.slice(0..);
    let fallback_sin = bufs.d_sin.slice(0..);
    let (d_pos_seqlen, d_cos, d_sin) = match token {
        Some(t) => (&t.d_pos_seqlen, &t.d_cos, &t.d_sin),
        None => (&fallback_pos, &fallback_cos, &fallback_sin),
    };

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
        d_cos,
        d_sin,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
        layer_offset,
    )
}

// =============================================================================
// encode_attn_phase_tq2
// =============================================================================

/// Encode the full attention sublayer using TQ2 (ternary) QKV GEMV on the CUDA stream (steps 1-7).
///
/// Identical to `encode_attn_phase` but uses `graph.launch_gemv_tq2_v1_pub` for step 2
/// instead of `graph.launch_gemv_pub` (Q1). Required by the ternary prefill path
/// (`encode_prefill_layer_ternary`) which runs sequential per-token attention with TQ2 weights.
///
/// On return `bufs.d_attn_out` holds `[nq * head_dim]` attention output values.
///
/// # Safety
/// The function launches CUDA kernels.  The caller must ensure all GPU state
/// is valid and the stream is not concurrently used.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_attn_phase_tq2(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_pre_norm_weight: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_q_norm_weight: &CudaSlice<f32>,
    d_k_norm_weight: &CudaSlice<f32>,
    kv: &mut CudaKvCache,
    layer_idx: usize,
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
    // F9: the launchers take device views so the prefill path can point them
    // at chunk-resident buffers; this entry point always uses the single-token
    // scratch slots.
    let d_pos_seqlen = &bufs.d_pos_seqlen.slice(0..);
    let d_cos = &bufs.d_cos.slice(0..);
    let d_sin = &bufs.d_sin.slice(0..);

    // Step 1: RMSNorm(d_hidden, norm_weight -> d_normed)
    graph.launch_rmsnorm_pub(
        &bufs.d_hidden,
        d_pre_norm_weight,
        &mut bufs.d_normed,
        h_u32,
        norm_eps,
    )?;

    // Step 2: Fused QKV TQ2 GEMV (normed -> d_qkv)
    graph.launch_gemv_tq2_v1_pub(
        d_fused_qkv_weight,
        &bufs.d_normed,
        &mut bufs.d_qkv,
        qkv_total_rows,
        h_u32,
    )?;

    // Step 3: Fused QK-Norm + RoPE
    let k_offset = nq * head_dim;
    let k_in_view = bufs.d_qkv.slice(k_offset..);
    launch_fused_qk_norm_rope(
        graph,
        mods,
        &bufs.d_qkv,
        &k_in_view,
        &mut bufs.d_q_rope,
        &mut bufs.d_k_rope,
        d_q_norm_weight,
        d_k_norm_weight,
        d_cos,
        d_sin,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
        layer_offset,
    )
}

// =============================================================================
// encode_attn_phase_from_qkv
// =============================================================================

/// Encode attention steps 3-7, assuming QKV is already computed in `bufs.d_qkv`.
///
/// Used by the Q4_0/Q8_0 prefill path where batch GEMM pre-computes QKV for all
/// tokens.  Steps 1 (RMSNorm) and 2 (QKV GEMV) are skipped — the caller must
/// populate `bufs.d_qkv` before calling this function.
///
/// Steps performed:
/// - Step 3: Fused QK-norm + RoPE
/// - Step 4: Fused KV-store (writes FP16 into KV cache)
/// - Step 5: Batched attention scores V2
/// - Step 6: Batched softmax
/// - Step 7: Batched weighted sum
///
/// On return `bufs.d_attn_out` holds `[nq * head_dim]` attention output values.
///
/// # Safety
/// The function launches CUDA kernels.  The caller must ensure all GPU state
/// is valid and `bufs.d_qkv` has been filled with valid QKV data for this token.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_attn_phase_from_qkv(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_norm_weight: &CudaSlice<f32>,
    d_k_norm_weight: &CudaSlice<f32>,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    norm_eps: f32,
    bufs: &mut CudaFullLayerBuffers,
) -> Result<(), CudaGraphError> {
    let nq_u32 = nq as u32;
    let nkv_u32 = nkv as u32;
    let hd_u32 = head_dim as u32;
    let heads_per_group_u32 = heads_per_group as u32;
    let max_seq_u32 = bufs.max_seq as u32;
    let inv_sqrt_hd = 1.0f32 / (head_dim as f32).sqrt();
    let layer_offset = kv.layer_offset_elements(layer_idx);
    // F9: the launchers take device views so the prefill path can point them
    // at chunk-resident buffers; this entry point always uses the single-token
    // scratch slots.
    let d_pos_seqlen = &bufs.d_pos_seqlen.slice(0..);
    let d_cos = &bufs.d_cos.slice(0..);
    let d_sin = &bufs.d_sin.slice(0..);

    // Step 3: Fused QK-Norm + RoPE
    // Q occupies elements [0 .. nq*head_dim] in d_qkv.
    // K occupies elements [nq*head_dim .. (nq+nkv)*head_dim] in d_qkv.
    let k_offset = nq * head_dim;
    let k_in_view = bufs.d_qkv.slice(k_offset..);
    launch_fused_qk_norm_rope(
        graph,
        mods,
        &bufs.d_qkv,
        &k_in_view,
        &mut bufs.d_q_rope,
        &mut bufs.d_k_rope,
        d_q_norm_weight,
        d_k_norm_weight,
        d_cos,
        d_sin,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
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
        d_pos_seqlen,
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
