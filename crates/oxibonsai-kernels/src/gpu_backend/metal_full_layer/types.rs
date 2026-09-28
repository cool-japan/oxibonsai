//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::metal_graph::{alloc_buf, MetalGraphError, MetalWeightHandle};
use metal::{Buffer, MTLResourceOptions};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// GPU weight-cache identity (MET-02)
// ═══════════════════════════════════════════════════════════════════════════

/// On-GPU layout of a cached weight buffer.
///
/// `MetalGraph`'s weight cache is process-wide and shared by the raw-f32 path
/// (norms, the image crate's text encoder and VAE), the Q1 SoA path, the
/// ternary SoA path and — from Bonsai 2 — the `PQ2_0` and `PTQ1_0` paths. The
/// buffers are **not** interchangeable: a Q1 SoA buffer handed to the ternary
/// GEMV decodes garbage. The kind is therefore part of the cache key
/// ([`WeightKey`]) and a lookup that finds the same slot under a *different*
/// kind is an error, not a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightKind {
    /// Raw `f32` bytes (RMSNorm weights, `encode_gemm_f32`/`bf16` matrices).
    RawF32,
    /// `Q1_0_g128` reformatted to SoA `[N×2 B scales][N×16 B signs]`.
    Q1Soa,
    /// `TQ2_0_g128` (ggml 42, qs-first AoS) reformatted to SoA.
    Tq2Soa,
    /// `PQ2_0` (ggml 142, d-first AoS) reformatted to the same SoA shape as
    /// [`WeightKind::Tq2Soa`] but decoded with `0b11 → +2`.
    Pq2Soa,
    /// `PTQ1_0` (ggml 143, 28-byte base-3 blocks) in its GPU layout.
    Ptq1Soa,
}

impl WeightKind {
    /// Short, stable name used in cache diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RawF32 => "raw_f32",
            Self::Q1Soa => "q1_soa",
            Self::Tq2Soa => "tq2_soa",
            Self::Pq2Soa => "pq2_soa",
            Self::Ptq1Soa => "ptq1_soa",
        }
    }

    /// Whether this kind holds packed quantized blocks (everything but
    /// [`WeightKind::RawF32`]). Selects which upload counter is incremented.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        !matches!(self, Self::RawF32)
    }
}

impl fmt::Display for WeightKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stable substring of every weight-cache kind-mismatch error message.
///
/// `MetalGraph`'s cache reports a format collision as
/// [`MetalGraphError::WeightKindMismatch`], whose `Display` opens with this
/// tag. Callers that only need to recognise the condition in a string (logs,
/// downstream error text) match on this rather than on the full wording;
/// callers holding the error itself should match the variant.
///
/// [`MetalGraphError::WeightKindMismatch`]: crate::gpu_backend::metal_graph::MetalGraphError::WeightKindMismatch
pub const WEIGHT_KIND_MISMATCH_TAG: &str = "weight cache kind mismatch";

/// Epoch used by callers that have not been given a model epoch yet.
///
/// [`next_model_epoch`] never returns it, so entries uploaded under a real
/// model epoch can never collide with legacy, pointer-keyed uploads (the image
/// crate keys on `as_ptr()`), and `release_model(LEGACY_MODEL_EPOCH)` releases
/// exactly the un-epoched set.
pub const LEGACY_MODEL_EPOCH: u64 = 0;

/// Composite identity of one entry in `MetalGraph`'s weight cache.
///
/// Before MET-02 the key was a bare `u64` chosen per call site
/// (`2_000_000 + layer * 10`, `as_ptr() as u64`, …) with no model or format
/// component, so unloading a model never freed its buffers and the Q1
/// `final_norm` handle (`2_000_000`) aliased the ternary norm base.
///
/// `slot` is a `u64` (not the `u32` of the original sketch) because the image
/// crate's weights are keyed by their mmap address; truncating a pointer to 32
/// bits would reintroduce exactly the silent aliasing this key removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightKey {
    /// Which loaded model owns the buffer; see [`next_model_epoch`].
    pub model_epoch: u64,
    /// On-GPU layout of the buffer.
    pub kind: WeightKind,
    /// Per-model weight identity (layer/tensor slot, or a stable address).
    pub slot: u64,
}

impl WeightKey {
    /// Key a weight belonging to a specific loaded model.
    #[must_use]
    pub const fn new(model_epoch: u64, kind: WeightKind, slot: u64) -> Self {
        Self {
            model_epoch,
            kind,
            slot,
        }
    }

    /// Key a weight for a caller that has no model epoch (see
    /// [`LEGACY_MODEL_EPOCH`]).
    #[must_use]
    pub const fn legacy(kind: WeightKind, slot: u64) -> Self {
        Self::new(LEGACY_MODEL_EPOCH, kind, slot)
    }

    /// The `(model_epoch, slot)` identity, i.e. the key without its format.
    ///
    /// Two entries sharing a `slot_id` but differing in [`WeightKind`] are a
    /// format collision, which the cache reports instead of serving.
    #[must_use]
    pub const fn slot_id(&self) -> (u64, u64) {
        (self.model_epoch, self.slot)
    }
}

impl fmt::Display for WeightKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "epoch {} slot {} kind {}",
            self.model_epoch, self.slot, self.kind
        )
    }
}

/// Source of model epochs; starts at 1 so no model ever gets
/// [`LEGACY_MODEL_EPOCH`].
static NEXT_MODEL_EPOCH: AtomicU64 = AtomicU64::new(LEGACY_MODEL_EPOCH + 1);

/// Allocate a fresh model epoch.
///
/// Called once per loaded model (`BonsaiModel::from_gguf_with_embd`), carried
/// on the model, used in every [`WeightKey`] it uploads, and passed to
/// `MetalGraph::release_model` when the model is dropped. Epochs are never
/// reused, so a reloaded model can never hit a previous load's buffers.
#[must_use]
pub fn next_model_epoch() -> u64 {
    NEXT_MODEL_EPOCH.fetch_add(1, Ordering::Relaxed)
}

/// Per-layer parameters for the full-forward path.
///
/// Contains weight handle IDs and raw byte slices for each layer's
/// weight matrices. These are used to upload/cache weights on the GPU.
pub struct FullForwardLayerParams<'a> {
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
/// Per-layer parameters for the ternary full-forward path.
///
/// Mirrors [`FullForwardLayerParams`] but carries AoS-packed TQ2_0_g128 block
/// bytes (34 bytes/block) for every GEMV weight.
///
/// # Weight-cache identity (`MET-02`)
///
/// Every `*_handle` below is a **slot**, and every lookup the ternary entry
/// points make — the four RMSNorm weights, the four projections and, through
/// the layers' shared epoch, the final-norm / LM-head tail — is keyed
/// `WeightKey::new(model_epoch, kind, slot)`. Before `MET-02` the kernels
/// hard-coded [`WeightKey::legacy`] for all of them, so a caller that owned a
/// real epoch could upload under it and then simply miss its own buffers on
/// the next lookup. All layers passed to one forward must carry the **same**
/// epoch (one forward binds one model's weights); the entry points reject a
/// mixed slice rather than guess.
pub struct FullForwardLayerParamsTernary<'a> {
    /// Weight-cache epoch this layer's buffers are keyed under — see
    /// [`next_model_epoch`] / [`LEGACY_MODEL_EPOCH`] and
    /// `MetalGraph::release_model`.
    pub model_epoch: u64,
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
/// Bytes per KV-cache element (`half`).
pub(crate) const KV_ELEMENT_BYTES: u64 = 2;

/// Largest KV-cache element count the **current** MSL address math can reach.
///
/// `fused_kv_store`, `batched_attention_scores(_v2)` and
/// `batched_attention_weighted_sum` still declare the layer base as
/// `constant uint&` and derive `dst_offset = layer_offset + (head * max_seq +
/// pos) * head_dim + d` in 32-bit (`kernel_sources/attention.rs:256,264,307,
/// 366,521`). The Rust side now carries the offset as `u64`
/// ([`kv_layer_offset_elements`]) so nothing truncates *here*, but until B2-15
/// widens those four bindings to `ulong` the **whole linear index** must still
/// fit in a `uint`, which is what this cap enforces.
///
/// Note this is the *total element* bound, not the weaker "base offset of the
/// last layer" bound quoted in the MET-07 write-up (8B `max_seq 119_837`,
/// Bonsai 2 `66_577`): the kernel adds the intra-layer term on top of the layer
/// base, so the binding constraint is `n_layers * n_kv * max_seq * head_dim ≤
/// u32::MAX` (8B → `116_508`, Bonsai 2 → `65_535`).
pub(crate) const KV_CACHE_MAX_ELEMENTS: u64 = u32::MAX as u64;

/// Element offset of `layer_idx`'s slab inside a flat KV cache.
///
/// Computed in `u64`: the pre-MET-07 code returned `u32` via a silent
/// `as u32`, which wrapped for the 8B above `--max-seq-len 119_837` and aliased
/// two layers onto the same addresses with no error anywhere.
#[inline]
pub(crate) fn kv_layer_offset_elements(
    layer_idx: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> u64 {
    (layer_idx as u64) * (n_kv as u64) * (max_seq as u64) * (head_dim as u64)
}

/// Validate a requested KV-cache geometry against both hard limits and return
/// the total element count per cache buffer.
///
/// Rejects, with a message naming the largest usable `--max-seq-len`:
/// 1. `total_elements * 2 > max_buffer_length` — `newBufferWithLength:` returns
///    nil above the device limit (14.30 GB measured on this M3), and the
///    caller would otherwise dereference a null buffer (MET-06);
/// 2. `total_elements > u32::MAX` — the 32-bit MSL address math described on
///    [`KV_CACHE_MAX_ELEMENTS`].
///
/// `max_buffer_length` is passed in (rather than read from the device) so the
/// policy is unit-testable without a GPU.
pub(crate) fn check_kv_cache_geometry(
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
    max_buffer_length: u64,
) -> Result<u64, MetalGraphError> {
    if n_layers == 0 || n_kv == 0 || max_seq == 0 || head_dim == 0 {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "KV cache geometry must be non-zero, got n_layers={n_layers}, n_kv={n_kv}, \
             max_seq={max_seq}, head_dim={head_dim}"
        )));
    }

    // Elements added to each cache buffer by one extra sequence position.
    let per_position = (n_layers as u64)
        .checked_mul(n_kv as u64)
        .and_then(|v| v.checked_mul(head_dim as u64))
        .ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "KV cache geometry overflows: n_layers={n_layers} × n_kv={n_kv} × \
                 head_dim={head_dim}"
            ))
        })?;
    let total_elements = per_position.checked_mul(max_seq as u64).ok_or_else(|| {
        MetalGraphError::InvalidDimensions(format!(
            "KV cache geometry overflows: {per_position} × max_seq={max_seq}"
        ))
    })?;
    let byte_len = total_elements
        .checked_mul(KV_ELEMENT_BYTES)
        .ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!(
                "KV cache byte length overflows: {total_elements} × {KV_ELEMENT_BYTES}"
            ))
        })?;

    let max_seq_by_buffer = max_buffer_length / (per_position * KV_ELEMENT_BYTES);
    let max_seq_by_addressing = KV_CACHE_MAX_ELEMENTS / per_position;
    let max_usable_seq = max_seq_by_buffer.min(max_seq_by_addressing);

    if byte_len > max_buffer_length {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "KV cache does not fit in one Metal buffer: {n_layers} layers × {n_kv} KV heads × \
             {max_seq} positions × {head_dim} dims = {byte_len} bytes per cache, above this \
             device's maxBufferLength of {max_buffer_length} bytes. Maximum usable \
             --max-seq-len for this model on this device: {max_usable_seq}"
        )));
    }
    if total_elements > KV_CACHE_MAX_ELEMENTS {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "KV cache exceeds the 32-bit GPU address range: {n_layers} layers × {n_kv} KV heads × \
             {max_seq} positions × {head_dim} dims = {total_elements} elements, above the \
             {KV_CACHE_MAX_ELEMENTS}-element limit of the `constant uint&` layer offset in the \
             attention kernels. Maximum usable --max-seq-len for this model: {max_usable_seq}"
        )));
    }
    Ok(total_elements)
}

/// GPU-resident KV cache for all transformer layers.
///
/// Layout: `[n_layers × n_kv × max_seq × head_dim]` f16, contiguous.
/// Each layer occupies `n_kv * max_seq * head_dim` half-precision elements.
pub(crate) struct GpuKvCache {
    pub k_cache: Buffer,
    pub v_cache: Buffer,
    pub n_layers: usize,
    pub n_kv: usize,
    pub max_seq: usize,
    pub head_dim: usize,
}
impl GpuKvCache {
    /// Allocate the KV cache on the GPU.
    ///
    /// The geometry is validated by [`check_kv_cache_geometry`] **before** any
    /// buffer is requested, so an oversized `--max-seq-len` produces a named
    /// error instead of a nil buffer or silently aliased layers.
    pub fn allocate(
        device: &metal::Device,
        n_layers: usize,
        n_kv: usize,
        max_seq: usize,
        head_dim: usize,
    ) -> Result<Self, MetalGraphError> {
        let total_elements = check_kv_cache_geometry(
            n_layers,
            n_kv,
            max_seq,
            head_dim,
            device.max_buffer_length(),
        )?;
        let byte_len = total_elements * KV_ELEMENT_BYTES;
        let opts = MTLResourceOptions::StorageModePrivate;
        Ok(Self {
            k_cache: alloc_buf(device, byte_len, opts)?,
            v_cache: alloc_buf(device, byte_len, opts)?,
            n_layers,
            n_kv,
            max_seq,
            head_dim,
        })
    }
    /// Element offset into the cache for a given layer.
    ///
    /// `u64` since MET-07: the previous `as u32` wrapped silently for real 8B
    /// geometries above `--max-seq-len 119_837`. [`GpuKvCache::allocate`]
    /// additionally refuses any geometry whose *total* element count leaves the
    /// 32-bit range the MSL kernels still index in.
    #[inline]
    pub fn layer_offset_elements(&self, layer_idx: usize) -> u64 {
        kv_layer_offset_elements(layer_idx, self.n_kv, self.max_seq, self.head_dim)
    }
    /// Check whether this cache matches the given dimensions.
    pub fn matches(&self, n_layers: usize, n_kv: usize, max_seq: usize, head_dim: usize) -> bool {
        self.n_layers == n_layers
            && self.n_kv == n_kv
            && self.max_seq == max_seq
            && self.head_dim == head_dim
    }
}
/// Pre-cached GPU weight handles for a single transformer layer.
/// Eliminates per-token weight lookup and upload overhead.
pub struct CachedLayerWeights {
    pub attn_norm: Arc<MetalWeightHandle>,
    pub fused_qkv: Arc<MetalWeightHandle>,
    pub q_norm: Arc<MetalWeightHandle>,
    pub k_norm: Arc<MetalWeightHandle>,
    pub attn_proj: Arc<MetalWeightHandle>,
    pub ffn_norm: Arc<MetalWeightHandle>,
    pub gate_up: Arc<MetalWeightHandle>,
    pub down: Arc<MetalWeightHandle>,
}
/// Lazily allocated intermediate buffers for full-layer dispatch.
///
/// These are allocated once and reused across all forward passes.
/// Each forward pass uploads new data into these buffers.
pub(crate) struct FullLayerBuffers {
    pub hidden_buf: Buffer,
    pub normed_buf: Buffer,
    pub qkv_buf: Buffer,
    pub q_rope_buf: Buffer,
    pub k_rope_buf: Buffer,
    pub cos_buf: Buffer,
    pub sin_buf: Buffer,
    pub scores_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub swiglu_buf: Buffer,
    /// `[2 × intermediate]` private buffer holding the concatenated gate/up
    /// projections produced by the ternary FFN's first GEMV, before SwiGLU
    /// reduces it down to `swiglu_buf`. Unused on the Q1 path (the fused
    /// `fused_gate_up_swiglu_q1` kernel writes `swiglu_buf` directly).
    pub gate_up_buf: Buffer,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
}
impl FullLayerBuffers {
    /// Allocate all intermediate buffers for the given dimensions.
    pub fn allocate(
        device: &metal::Device,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, MetalGraphError> {
        let f32_size = std::mem::size_of::<f32>();
        let shared = MTLResourceOptions::StorageModeShared;
        let private = MTLResourceOptions::StorageModePrivate;
        let h_bytes = (hidden_size * f32_size) as u64;
        let qkv_total = nq * head_dim + 2 * nkv * head_dim;
        let qkv_bytes = (qkv_total * f32_size) as u64;
        let q_bytes = (nq * head_dim * f32_size) as u64;
        let k_bytes = (nkv * head_dim * f32_size) as u64;
        let half_dim = head_dim / 2;
        let rope_bytes = (half_dim * f32_size) as u64;
        let scores_bytes = (nq * max_seq * f32_size) as u64;
        let inter_bytes = (intermediate_size * f32_size) as u64;
        let gate_up_bytes = (2 * intermediate_size * f32_size) as u64;
        Ok(Self {
            hidden_buf: alloc_buf(device, h_bytes, shared)?,
            normed_buf: alloc_buf(device, h_bytes, private)?,
            qkv_buf: alloc_buf(device, qkv_bytes, private)?,
            q_rope_buf: alloc_buf(device, q_bytes, private)?,
            k_rope_buf: alloc_buf(device, k_bytes, private)?,
            cos_buf: alloc_buf(device, rope_bytes, shared)?,
            sin_buf: alloc_buf(device, rope_bytes, shared)?,
            scores_buf: alloc_buf(device, scores_bytes, private)?,
            attn_out_buf: alloc_buf(device, q_bytes, private)?,
            swiglu_buf: alloc_buf(device, inter_bytes, private)?,
            gate_up_buf: alloc_buf(device, gate_up_bytes, private)?,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        })
    }
    /// Check whether existing buffers match the requested dimensions.
    pub fn matches(
        &self,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> bool {
        self.hidden_size == hidden_size
            && self.intermediate_size == intermediate_size
            && self.nq == nq
            && self.nkv == nkv
            && self.head_dim == head_dim
            && self.max_seq == max_seq
    }
}
/// Pre-cached GPU weight handles for a Q1 (1-bit) model.
///
/// After initial creation, no weight data needs to be copied or uploaded —
/// [`try_metal_full_forward_cached`](super::functions_3::try_metal_full_forward_cached)
/// dispatches straight from these handles.
pub struct CachedQ1Weights {
    pub layers: Vec<CachedLayerWeights>,
    pub final_norm: Arc<MetalWeightHandle>,
    pub lm_head: Arc<MetalWeightHandle>,
}
/// Pre-cached GPU weight handles for one **ternary** (TQ2_0_g128) transformer
/// layer (`MET-03`).
///
/// The ternary twin of [`CachedLayerWeights`]: the same eight handles, but
/// every quantized one holds a [`WeightKind::Tq2Soa`] buffer (the norms are
/// [`WeightKind::RawF32`]). A separate type rather than a reuse of the Q1
/// struct so a ternary cache can never be handed to the Q1 encoder by
/// accident — the two layouts are not interchangeable.
pub struct CachedTernaryLayerWeights {
    /// Attention RMSNorm weight (`RawF32`).
    pub attn_norm: Arc<MetalWeightHandle>,
    /// Concatenated Q‖K‖V projection (`Tq2Soa`).
    pub fused_qkv: Arc<MetalWeightHandle>,
    /// Q RMSNorm weight (`RawF32`).
    pub q_norm: Arc<MetalWeightHandle>,
    /// K RMSNorm weight (`RawF32`).
    pub k_norm: Arc<MetalWeightHandle>,
    /// Attention output projection (`Tq2Soa`).
    pub attn_proj: Arc<MetalWeightHandle>,
    /// FFN RMSNorm weight (`RawF32`).
    pub ffn_norm: Arc<MetalWeightHandle>,
    /// Concatenated gate‖up projection (`Tq2Soa`).
    pub gate_up: Arc<MetalWeightHandle>,
    /// FFN down projection (`Tq2Soa`).
    pub down: Arc<MetalWeightHandle>,
}
/// Pre-cached GPU weight handles for a ternary (TQ2_0_g128) model (`MET-03`).
///
/// Mirrors [`CachedQ1Weights`]: built **once** by
/// [`build_cached_weights_ternary_only`](super::functions_3::build_cached_weights_ternary_only),
/// which uploads (or finds resident) every layer's eight buffers under
/// `model_epoch`, after which the `try_metal_*_ternary_cached` entry points
/// bind these handles directly — no per-token cache lookups, no per-token
/// `FullForwardLayerParamsTernary` rebuild, and **no host copy of any weight**:
/// the struct holds only reference-counted GPU buffers.
///
/// This replaces the pre-`MET-03` shape, which held `Vec<Vec<u8>>` host copies
/// of every projection (`qkv_concats`, `attn_proj_bytes`, `gate_bytes`,
/// `up_bytes`, `down_bytes`, `lm_head_bytes`) for the life of the model and
/// uploaded nothing; with that shape the whole process's peak memory
/// footprint on Ternary-Bonsai-8B was measured at 4.4× the GGUF's size.
pub struct CachedTernaryWeights {
    /// Weight-cache epoch every handle below was resolved under (`MET-02`).
    pub model_epoch: u64,
    /// Per-layer handles, in layer order.
    pub layers: Vec<CachedTernaryLayerWeights>,
    /// Final RMSNorm weight; `Some` exactly when `lm_head` is.
    pub final_norm: Option<Arc<MetalWeightHandle>>,
    /// TQ2 LM-head projection; `None` for a model whose LM head is not ternary
    /// (the caller then runs its own tail on the returned hidden state).
    pub lm_head: Option<Arc<MetalWeightHandle>>,
    /// Rows of the LM head (the logits length); `0` when there is no tail.
    pub lm_head_out_features: usize,
}
/// Pre-cached GPU weights for the whole model — one variant per weight format.
///
/// Both variants hold pre-uploaded, reference-counted GPU handles and nothing
/// else; they are separate types so a Q1 cache can never reach the ternary
/// encoder (or vice versa) — the SoA layouts differ.
pub enum CachedModelWeights {
    /// 1-bit (Q1_0_g128) cache: pre-uploaded per-layer + LM-head GPU handles.
    Q1(CachedQ1Weights),
    /// Ternary (TQ2_0_g128) cache: pre-uploaded per-layer + tail GPU handles
    /// (`MET-03`).
    Ternary(CachedTernaryWeights),
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — KV cache addressing and allocation guards (MET-07)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// This M3's measured `maxBufferLength` (14.30 GB), used so the pure tests
    /// exercise the same thresholds as the device path.
    const M3_MAX_BUFFER_LENGTH: u64 = 15_032_385_536;

    /// Ternary-Bonsai-8B decode geometry: 36 layers, 8 KV heads, head_dim 128.
    const B8: (usize, usize, usize) = (36, 8, 128);
    /// Bonsai 2 27B full-attention geometry: 64 layers, 4 KV heads, head_dim 256.
    const B27: (usize, usize, usize) = (64, 4, 256);

    #[test]
    fn layer_offset_elements_does_not_truncate_past_u32() {
        // 8B top layer at --max-seq-len 131072: 35 × 8 × 131072 × 128.
        let offset = kv_layer_offset_elements(35, 8, 131_072, 128);
        assert_eq!(offset, 4_697_620_480);
        assert!(offset > u64::from(u32::MAX));
        // The pre-MET-07 `as u32` produced this instead — layer 35 aliased onto
        // the address range of a lower layer with no error anywhere.
        assert_eq!(offset as u32, 402_653_184);
    }

    #[test]
    fn layer_offset_elements_matches_the_flat_layout() {
        for layer in 0..4usize {
            assert_eq!(
                kv_layer_offset_elements(layer, 8, 4096, 128),
                (layer * 8 * 4096 * 128) as u64
            );
        }
    }

    #[test]
    fn kv_geometry_accepts_real_decode_shapes() {
        let (n_layers, n_kv, head_dim) = B8;
        let total = check_kv_cache_geometry(n_layers, n_kv, 4096, head_dim, M3_MAX_BUFFER_LENGTH)
            .expect("the shipped 8B decode geometry must be accepted");
        assert_eq!(total, (36 * 8 * 4096 * 128) as u64);

        let (n_layers, n_kv, head_dim) = B27;
        check_kv_cache_geometry(n_layers, n_kv, 32_768, head_dim, M3_MAX_BUFFER_LENGTH)
            .expect("Bonsai 2 at a 32K context must be accepted");
    }

    #[test]
    fn kv_geometry_rejects_u32_address_overflow_and_names_max_seq_len() {
        let (n_layers, n_kv, head_dim) = B8;
        // 36 × 8 × head_dim = 36864 elements per position → the last accepted
        // --max-seq-len is u32::MAX / 36864 = 116508.
        let limit = (u32::MAX as usize) / (n_layers * n_kv * head_dim);
        assert_eq!(limit, 116_508);
        check_kv_cache_geometry(n_layers, n_kv, limit, head_dim, u64::MAX)
            .expect("exactly at the limit must still be accepted");

        let err = check_kv_cache_geometry(n_layers, n_kv, limit + 1, head_dim, u64::MAX)
            .expect_err("one position above the 32-bit range must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("32-bit GPU address range"), "{msg}");
        assert!(msg.contains("--max-seq-len"), "{msg}");
        assert!(msg.contains("116508"), "{msg}");
    }

    #[test]
    fn kv_geometry_rejects_the_27b_full_context() {
        let (n_layers, n_kv, head_dim) = B27;
        // 64 × 4 × 262144 × 256 = 2^34 elements = 34.36 GB per cache buffer.
        let err = check_kv_cache_geometry(n_layers, n_kv, 262_144, head_dim, M3_MAX_BUFFER_LENGTH)
            .expect_err("the 27B at its full 262144 context cannot be allocated");
        let msg = err.to_string();
        assert!(msg.contains("maxBufferLength"), "{msg}");
        // 14.30 GB / (65536 elements/pos × 2 B) = 114 688 positions, but the
        // 32-bit cap binds first at 65 535 — the reported maximum is the min.
        assert!(msg.contains("65535"), "{msg}");
    }

    #[test]
    fn kv_geometry_rejects_buffers_above_max_buffer_length() {
        // Geometry whose element count is inside the 32-bit range but whose
        // byte length is not: 1 × 1 × 128 elements/position.
        let small_limit: u64 = 1 << 30; // 1 GiB
        let per_position = 128u64;
        let max_seq = (small_limit / (per_position * KV_ELEMENT_BYTES)) as usize;
        check_kv_cache_geometry(1, 1, max_seq, 128, small_limit)
            .expect("exactly one buffer's worth must be accepted");

        let err = check_kv_cache_geometry(1, 1, max_seq + 1, 128, small_limit)
            .expect_err("one position past maxBufferLength must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("maxBufferLength"), "{msg}");
        assert!(msg.contains(&max_seq.to_string()), "{msg}");
    }

    #[test]
    fn kv_geometry_rejects_zero_dimensions() {
        for dims in [
            (0, 8, 4096, 128),
            (36, 0, 4096, 128),
            (36, 8, 0, 128),
            (36, 8, 4096, 0),
        ] {
            let err = check_kv_cache_geometry(dims.0, dims.1, dims.2, dims.3, u64::MAX)
                .expect_err("zero dimensions must be rejected");
            assert!(err.to_string().contains("non-zero"), "{err}");
        }
    }

    #[test]
    fn kv_geometry_reports_overflowing_products_instead_of_wrapping() {
        let err = check_kv_cache_geometry(usize::MAX, 8, 4096, 128, u64::MAX)
            .expect_err("a product that overflows u64 must be reported, not wrapped");
        assert!(err.to_string().contains("overflows"), "{err}");
    }

    /// Device-backed: `allocate` must reject the oversized geometry *before*
    /// asking Metal for a 34 GB buffer, and must still serve a small one.
    #[test]
    fn allocate_guards_before_touching_the_device() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let (n_layers, n_kv, head_dim) = B27;
        let err = match GpuKvCache::allocate(&device, n_layers, n_kv, 262_144, head_dim) {
            Ok(_) => panic!("27B @ 262144 must not be allocated on a 24 GB Mac"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("--max-seq-len"), "{err}");

        let cache = GpuKvCache::allocate(&device, 4, 2, 128, 64)
            .expect("a small KV cache must still allocate");
        assert!(cache.matches(4, 2, 128, 64));
        assert_eq!(cache.layer_offset_elements(3), (3 * 2 * 128 * 64) as u64);
    }
}
