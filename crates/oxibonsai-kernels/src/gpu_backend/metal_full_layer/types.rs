//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::metal_graph::{alloc_buf, MetalGraphError, MetalWeightHandle};
use metal::{Buffer, MTLResourceOptions};
use std::sync::Arc;

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
pub struct FullForwardLayerParamsTernary<'a> {
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
    pub fn allocate(
        device: &metal::Device,
        n_layers: usize,
        n_kv: usize,
        max_seq: usize,
        head_dim: usize,
    ) -> Result<Self, MetalGraphError> {
        let total_elements = n_layers * n_kv * max_seq * head_dim;
        let byte_len = (total_elements * 2) as u64;
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
    #[inline]
    pub fn layer_offset_elements(&self, layer_idx: usize) -> u32 {
        (layer_idx * self.n_kv * self.max_seq * self.head_dim) as u32
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
/// Pre-cached GPU weight handles for the entire model.
/// After initial creation, no weight data needs to be copied or uploaded.
pub struct CachedModelWeights {
    pub layers: Vec<CachedLayerWeights>,
    pub final_norm: Arc<MetalWeightHandle>,
    pub lm_head: Arc<MetalWeightHandle>,

    // ── Ternary (TQ2_0_g128) cache fields ───────────────────────────────────
    //
    // These are populated only for ternary models. For Q1 models they remain
    // empty/zero.  Layer params are NOT stored here (they borrow from these
    // vecs); callers rebuild `FullForwardLayerParamsTernary` each decode call
    // by referencing the slices below — cheap struct literals, no leaks.
    //
    // Handle ID allocation (distinct from the Q1 namespace):
    //   norm    handles: 5_000_000 + layer * 10 + offset
    //   weight  handles: 6_000_000 + layer * 10 + offset
    //   lm_head handle : 7_000_000
    pub ternary_qkv_concats: Vec<Vec<u8>>,
    pub ternary_attn_proj_bytes: Vec<Vec<u8>>,
    /// Gate projection bytes per layer (separate from up, kernel concatenates lazily).
    pub ternary_gate_bytes: Vec<Vec<u8>>,
    /// Up projection bytes per layer.
    pub ternary_up_bytes: Vec<Vec<u8>>,
    pub ternary_down_bytes: Vec<Vec<u8>>,
    pub ternary_lm_head_bytes: Vec<u8>,
    pub ternary_lm_head_out_features: usize,
}
