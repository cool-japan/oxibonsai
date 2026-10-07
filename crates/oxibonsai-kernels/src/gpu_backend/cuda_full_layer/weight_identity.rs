//! The identity rule of a cached full-forward GPU weight set, shared by the Q1
//! cache ([`super::model_weights_fingerprint`], slot `cached_q1_model_weights`)
//! and the ternary cache
//! ([`super::encode_ternary::ternary_model_weights_fingerprint`], slot
//! `cached_model_weights`).
//!
//! Each slot holds one uploaded weight set and validates it against this
//! fingerprint on every decode token. A mismatch evicts the cached set,
//! uploads the whole model again under a fresh epoch and forces a CUDA-graph
//! re-capture; a false match decodes the new model with the previous model's
//! device weights. The fingerprint must therefore be **stable for one model**
//! and **different across models**. Per layer it mixes:
//!
//! - every **handle id** the layer's weights are cached under. These carry the
//!   model's identity: `oxibonsai-model` composes them over the model's
//!   `cuda_model_epoch` (`SlotNamespace`: `TAG | epoch << 24 | local`), minted
//!   once per load, or — for the Q1 fused weights of a block uploaded at load —
//!   takes them from the process-wide upload-handle counter. Neither source
//!   ever hands out an id twice, so two loaded models (even two loads of one
//!   file, even when the second model's host slices land on the first one's
//!   freed addresses) never share a fingerprint;
//! - base address + length of every **model-owned** source slice. They borrow
//!   the model's owned or mmap'd bytes, so they are stable for the life of the
//!   model and still separate two weight sets whose caller passes fixed rather
//!   than per-load handle ids;
//! - the **length only** of the fused Q‖K‖V buffer, a caller-side
//!   concatenation of three tensors whose address is an allocator artefact,
//!   not an identity.
//!
//! Host addresses alone were never an identity in either direction. A
//! per-call scratch address made one ternary model miss on every token (the
//! ternary decode thrash), and a dropped model's freed addresses can be handed
//! to the next model: a same-depth model that reused all of them would have
//! been served the previous model's device weights by the Q1 cache, which used
//! to hash nothing else.
//!
//! Cost is O(n_layers) with a tiny constant, cheap enough to run on every
//! decode token.

use super::encode_ternary::CudaFullForwardLayerParamsTernary;
use super::CudaFullForwardLayerParams;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// What one transformer layer contributes to its weight set's identity
/// ([`weight_set_fingerprint`]).
///
/// Built from either family's layer parameters through the two `From` impls
/// below, which take the same fields in the same order, so a Q1 and a ternary
/// layer are judged by one rule.
pub(super) struct LayerWeightIdentity {
    /// Every handle id, in a fixed order: attention norm, fused Q‖K‖V, Q norm,
    /// K norm, attention output, FFN norm, gate‖up, down.
    pub(super) handles: [u64; 8],
    /// `(base address, length)` of every model-owned source slice: attention
    /// norm, attention output, gate, up, down, FFN norm.
    pub(super) owned_slices: [(u64, u64); 6],
    /// Length of the caller-side fused Q‖K‖V concatenation; deliberately not
    /// its address (see the module docs).
    pub(super) fused_qkv_len: u64,
}

/// A slice's `(base address, length)`.
fn slice_identity<T>(s: &[T]) -> (u64, u64) {
    (s.as_ptr() as usize as u64, s.len() as u64)
}

impl From<&CudaFullForwardLayerParams<'_>> for LayerWeightIdentity {
    fn from(lp: &CudaFullForwardLayerParams<'_>) -> Self {
        Self {
            handles: [
                lp.attn_norm_handle,
                lp.fused_qkv_handle,
                lp.q_norm_handle,
                lp.k_norm_handle,
                lp.attn_proj_handle,
                lp.ffn_norm_handle,
                lp.gate_up_handle,
                lp.down_handle,
            ],
            owned_slices: [
                slice_identity(lp.attn_norm_bytes),
                slice_identity(lp.attn_proj_bytes),
                slice_identity(lp.gate_bytes),
                slice_identity(lp.up_bytes),
                slice_identity(lp.down_bytes),
                slice_identity(lp.ffn_norm_bytes),
            ],
            fused_qkv_len: lp.fused_qkv_bytes.len() as u64,
        }
    }
}

impl From<&CudaFullForwardLayerParamsTernary<'_>> for LayerWeightIdentity {
    fn from(lp: &CudaFullForwardLayerParamsTernary<'_>) -> Self {
        Self {
            handles: [
                lp.attn_norm_handle,
                lp.fused_qkv_handle,
                lp.q_norm_handle,
                lp.k_norm_handle,
                lp.attn_proj_handle,
                lp.ffn_norm_handle,
                lp.gate_up_handle,
                lp.down_handle,
            ],
            owned_slices: [
                slice_identity(lp.attn_norm_bytes),
                slice_identity(lp.attn_proj_bytes),
                slice_identity(lp.gate_bytes),
                slice_identity(lp.up_bytes),
                slice_identity(lp.down_bytes),
                slice_identity(lp.ffn_norm_bytes),
            ],
            fused_qkv_len: lp.fused_qkv_bytes.len() as u64,
        }
    }
}

/// FNV-1a over the layer count and then, layer by layer, the handle ids, the
/// owned slices' `(address, length)` pairs and the fused Q‖K‖V length (see
/// the module docs for why exactly these).
///
/// Order-sensitive: the same layers in another order, or two handles trading
/// places, give another fingerprint.
pub(super) fn weight_set_fingerprint<I>(layers: I) -> u64
where
    I: ExactSizeIterator<Item = LayerWeightIdentity>,
{
    let mut h = FNV_OFFSET_BASIS;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(FNV_PRIME);
    };
    mix(layers.len() as u64);
    for layer in layers {
        for handle in layer.handles {
            mix(handle);
        }
        for (ptr, len) in layer.owned_slices {
            mix(ptr);
            mix(len);
        }
        mix(layer.fused_qkv_len);
    }
    h
}
