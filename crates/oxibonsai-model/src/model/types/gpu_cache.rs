//! GPU weight cache management for the Metal zero-overhead decode path.
//!
//! # One slot namespace for the ternary path (MET-02)
//!
//! `MetalGraph`'s weight cache is process-wide and keyed by
//! `WeightKey { model_epoch, kind, slot }`. Before this module owned the slot
//! table the ternary decode paths used **two** disjoint hard-coded namespaces —
//! `2_000_000 + layer * 10` (norms) / `3_000_000 + layer * 10` (weights) inside
//! `try_metal_full_forward_ternary_inner`, and `5_000_000` / `6_000_000` /
//! `5_900_000` / `7_000_000` in every other ternary path.
//! `try_metal_full_forward_with_lm_head_ternary` uploads every layer's weights
//! as it encodes and only then runs the final-norm → LM-head tail, so **any**
//! deterministic tail failure left the 5M/6M namespace fully populated and
//! dropped `BonsaiModel::forward()` into the 2M/3M one, which uploaded a second
//! full copy of the model on the very first token (14.4 GB for the 27B `PQ2_0`
//! against a 19.07 GB recommended working set). The `2_000_000` / `3_000_000`
//! bases additionally aliased the old literal Q1 `final_norm` / `lm_head`
//! handles — and `2_000_000` aliased the Q1 `final_norm` under the *same*
//! `WeightKind::RawF32`, i.e. as a silent stale hit rather than an error.
//!
//! Both namespaces are gone. Every ternary slot is derived here, from the
//! **address of the mapped GGUF tensor the buffer holds**
//! ([`TernaryLayerSlots`] / [`TernaryTailSlots`]) — the keying the image crate
//! already uses (`q.blocks.as_ptr() as u64`). That makes the paths agree by
//! construction rather than by two lists of literals staying in sync, and gives
//! three further properties for free:
//!
//! * **No Q1 collision, structurally.** macOS reserves the whole first 4 GiB of
//!   every process's address space (`__PAGEZERO` is `0x1_0000_0000` on arm64 and
//!   x86_64), so a mapped-tensor slot is always ≥ [`MIN_TENSOR_SLOT`], and a
//!   user-space address never sets bit 63 — while every Q1 norm / LM-head slot
//!   is a tagged composition with bit 63 set ([`super::q1_slots`]).
//!   [`tensor_slot`] enforces both bounds instead of assuming them.
//! * **Two models never alias.** Two GGUFs are two mappings, so their tensors
//!   cannot share an address — where the old per-layer literals gave a draft
//!   model and a target model the exact same slots.
//! * **Engine-pool replicas share one upload.** All replicas built from one
//!   `GgufFile` see the same tensor addresses, so they share a single GPU
//!   buffer instead of each uploading its own.
//!
//! # Upload once, keep no host copy (MET-03 / perf-03)
//!
//! `CachedTernaryWeights` used to hold `Vec<Vec<u8>>` host copies of every
//! projection for the life of the model — built with `.to_vec()` from borrowed
//! mmap slices, so copy #1 stayed evictable page cache while copies #2 (the
//! cache) and #3 (the transient SoA buffer) were dirty anonymous pages. Measured
//! on the shipping release binary, the 8B (2.18 GB on disk) peaked at 9.51 GB.
//! The batched prefill paths were worse: they rebuilt all five `Vec<Vec<u8>>`
//! **per call**, with no memoization (perf-03).
//!
//! Now [`BonsaiModel::get_or_create_gpu_cache`] uploads every ternary weight
//! once, and the `CachedModelWeights::Ternary` value it stores holds **only
//! reference-counted GPU handles** — a `CachedTernaryWeights` of eight handles
//! per layer plus the final-norm / LM-head tail, built by
//! `build_cached_weights_ternary_only` (the MET-03 *shape*: it mirrors the Q1
//! `CachedQ1Weights`). No host byte of any weight is retained. The two layouts
//! that need a real allocation (the Q‖K‖V and gate‖up concatenations) are built
//! in one reused staging buffer at load time, or inside
//! `get_or_upload_*_lazy` closures on a later miss, and are dropped as soon as
//! the GPU buffer is written.
//!
//! The cached decode entry points on this type
//! ([`BonsaiModel::forward_greedy_gpu_ternary_cached`],
//! [`BonsaiModel::forward_logits_gpu_ternary_cached`],
//! [`BonsaiModel::prefill_logits_gpu_ternary_cached`],
//! [`BonsaiModel::prefill_verify_gpu_ternary_cached`]) bind those handles
//! directly — no per-token `FullForwardLayerParamsTernary` rebuild and no
//! per-token cache lookup — and are bit-identical to the uncached paths,
//! because they bind the very same buffers. Every ternary path the model's
//! decode and prefill routes take (`forward_metal.rs`) goes through them, with
//! one exception: the hidden-state-only forward of a ternary body under a
//! **non-ternary** LM head keeps the per-call binding, because the cache it
//! would bind is built only alongside a ternary tail.
//!
//! # Weight-cache epoch: one per GGUF mapping (MET-02)
//!
//! Every ternary buffer — eight per layer plus the tail — is keyed under the
//! model's **mapping epoch** ([`BonsaiModel::gpu_mapping_epoch`]): the epoch
//! of the `super::q1_slots` namespace the model joined at construction,
//! shared by every replica of one GGUF mapping and by the Q1 norm / LM-head
//! slots, distinct across mappings. The address-derived slots already made
//! keys unique per *live* mapping; the epoch closes the one hole they left
//! open — a model dropped and re-loaded at a **reused** address now mints a
//! new epoch, so it can never bind a buffer the previous mapping left
//! behind, whatever happened to that mapping's release.
//!
//! # Release on unload
//!
//! The mapping's buffers leave the GPU in one
//! `MetalGraph::release_model(epoch)` when the **last** replica of the
//! mapping drops (the namespace's own release hook; see
//! `super::q1_slots::MappingState`), and [`BonsaiModel::release_metal_weights`]
//! does the same on demand. A replica dropping while a sibling is alive
//! releases nothing: replicas deliberately share these slots, so evicting on
//! every replica's drop would pull buffers out from under siblings that are
//! still binding them, forcing a mid-serve re-upload of the whole model. The
//! registry's replica count replaces the address-keyed refcount table this
//! module used to keep beside the slots.

use super::{BonsaiModel, OutputWeight};
use crate::block::{blocks_as_bytes, blocks_as_bytes_ternary, TransformerBlock};
use crate::layers::linear::LinearTernary;
#[cfg(test)]
use oxibonsai_kernels::gpu_backend::metal_full_layer::types::WeightKind;
use oxibonsai_kernels::{CachedModelWeights, FullForwardLayerParamsTernary, MetalGraph};

pub(super) use super::q1_slots::MIN_TENSOR_SLOT;

/// Convenience alias for the boxed error every Metal entry point here returns.
type GpuResult<T> = Result<T, Box<dyn std::error::Error>>;

// ═══════════════════════════════════════════════════════════════════════════
// Slot table — one namespace for every ternary Metal path (MET-02)
// ═══════════════════════════════════════════════════════════════════════════

/// Offset added to a layer's `attn_q` anchor for its attention RMSNorm slot.
const SLOT_OFF_ATTN_NORM: u64 = 1;
/// Offset added to a layer's `attn_q` anchor for its Q RMSNorm slot.
const SLOT_OFF_Q_NORM: u64 = 2;
/// Offset added to a layer's `attn_q` anchor for its K RMSNorm slot.
const SLOT_OFF_K_NORM: u64 = 3;
/// Offset added to a layer's `attn_q` anchor for its FFN RMSNorm slot.
const SLOT_OFF_FFN_NORM: u64 = 4;
/// Offset added to the LM-head anchor for the final RMSNorm slot.
const SLOT_OFF_FINAL_NORM: u64 = 1;

/// The `fused_qkv_bytes` a caller passes once the buffer is known resident.
///
/// The Q‖K‖V concatenation is the one ternary layout that cannot be borrowed
/// from the mapping — Q, K and V are three separate GGUF tensors and the kernel
/// wants them contiguous. Keeping a host copy for the life of the model is
/// exactly what MET-03 measured as 4.4× the file size, and
/// `FullForwardLayerParamsTernary::fused_qkv_bytes` is a `&[u8]`, not a closure,
/// so there is nothing to borrow from.
///
/// Every uncached ternary dispatch therefore runs
/// [`BonsaiModel::ternary_gpu_binding`] first, which builds the concatenation
/// inside a `get_or_upload_tq2_weight_soa_lazy_for_epoch` closure — on a cache
/// miss only — and hands it straight to the GPU, **under the same mapping
/// epoch** the binding's parameter structs carry (and the kernels look the
/// slot up under). By the time the kernel looks the slot up the buffer is
/// resident and **these bytes are never read**. (The cached entry points take
/// no parameter structs at all: they bind the handle the cache resolved.)
///
/// The slice is deliberately *empty* rather than a plausible-looking stand-in:
/// `upload_bytes` rejects an empty input with
/// `MetalGraphError::BufferCreationFailed`, so if the residency invariant is
/// ever broken (an eviction racing a dispatch, a future caller that skips the
/// prologue, a lookup keyed under a different epoch than the prologue's) the
/// dispatch **fails closed** with a named error and the caller falls back to
/// the CPU path, instead of binding a short or wrong buffer to the GEMV.
pub(super) const FUSED_QKV_ALREADY_RESIDENT: &[u8] = &[];

/// Derive a weight-cache slot from the address of a mapped tensor.
///
/// `what` names the tensor in the error, which is the only thing that
/// distinguishes one floor violation from another.
fn tensor_slot(bytes: &[u8], what: &str) -> GpuResult<u64> {
    super::q1_slots::mapped_tensor_slot(bytes).ok_or_else(|| {
        format!(
            "ternary GPU slot for `{what}` is {:#x}, outside the mapped-tensor range \
             [{MIN_TENSOR_SLOT:#x}, 1 << 63) that keeps mapped-tensor slots clear of the tagged \
             Q1 slots; refusing to key a GPU weight on it",
            bytes.as_ptr() as u64
        )
        .into()
    })
}

/// Borrow a layer's ternary block bytes, naming the tensor if it is absent.
fn ternary_bytes<'b>(
    blocks: Option<&'b [oxibonsai_core::BlockTQ2_0_g128]>,
    what: &str,
) -> GpuResult<&'b [u8]> {
    let blocks = blocks.ok_or_else(|| format!("{what}: not a ternary layer"))?;
    Ok(blocks_as_bytes_ternary(blocks))
}

/// GPU weight-cache slots for one ternary transformer layer.
///
/// All eight are derived from mapped-tensor addresses, so the same layer always
/// resolves to the same slots no matter which forward path asks (MET-02). The
/// four RMSNorm slots hang off the `attn_q` anchor at `+1..=+4`: they are
/// `WeightKind::RawF32` while the anchor itself is `WeightKind::Tq2Soa`,
/// and `WeightKey` carries the kind, so the only slots those four are ever
/// compared against are other RMSNorm slots — which would have to belong to a
/// tensor starting within four bytes of an `attn_q` tensor to collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TernaryLayerSlots {
    /// Attention RMSNorm (`RawF32`).
    pub attn_norm: u64,
    /// Q RMSNorm (`RawF32`).
    pub q_norm: u64,
    /// K RMSNorm (`RawF32`).
    pub k_norm: u64,
    /// FFN RMSNorm (`RawF32`).
    pub ffn_norm: u64,
    /// Concatenated Q‖K‖V projection (`Tq2Soa`), anchored on `attn_q` — the
    /// same slot a GPU-uploaded block's fused-QKV arm keys on
    /// (`TransformerBlock::ternary_fused_qkv_slot`).
    pub fused_qkv: u64,
    /// Attention output projection (`Tq2Soa`).
    pub attn_proj: u64,
    /// Concatenated gate‖up projection (`Tq2Soa`), anchored on `ffn_gate` —
    /// the same slot a block's fused gate‖up arm keys on.
    pub gate_up: u64,
    /// FFN down projection (`Tq2Soa`).
    pub down: u64,
}

impl TernaryLayerSlots {
    /// Derive every slot of `block` from its mapped tensors.
    pub(super) fn for_block(block: &TransformerBlock<'_>) -> GpuResult<Self> {
        let q_anchor = tensor_slot(
            ternary_bytes(block.attn_q_blocks_ternary(), "attn_q")?,
            "attn_q",
        )?;
        let attn_proj = tensor_slot(
            ternary_bytes(block.attn_output_blocks_ternary(), "attn_output")?,
            "attn_output",
        )?;
        let gate_up = tensor_slot(
            ternary_bytes(block.ffn_gate_blocks_ternary(), "ffn_gate")?,
            "ffn_gate",
        )?;
        let down = tensor_slot(
            ternary_bytes(block.ffn_down_blocks_ternary(), "ffn_down")?,
            "ffn_down",
        )?;
        // K, V and up are not slots of their own — K and V ride in the fused
        // Q‖K‖V buffer and up rides in gate‖up — but they must exist before any
        // path can build those concatenations, so reject a half-ternary layer
        // here rather than several frames deeper.
        ternary_bytes(block.attn_k_blocks_ternary(), "attn_k")?;
        ternary_bytes(block.attn_v_blocks_ternary(), "attn_v")?;
        ternary_bytes(block.ffn_up_blocks_ternary(), "ffn_up")?;
        Ok(Self {
            attn_norm: q_anchor + SLOT_OFF_ATTN_NORM,
            q_norm: q_anchor + SLOT_OFF_Q_NORM,
            k_norm: q_anchor + SLOT_OFF_K_NORM,
            ffn_norm: q_anchor + SLOT_OFF_FFN_NORM,
            fused_qkv: q_anchor,
            attn_proj,
            gate_up,
            down,
        })
    }

    /// The `(slot, kind)` pairs this layer owns — the residency set the tests
    /// probe (the release itself is one `release_model(epoch)`).
    #[cfg(test)]
    pub(super) fn cache_keys(&self) -> [(u64, WeightKind); 8] {
        [
            (self.attn_norm, WeightKind::RawF32),
            (self.q_norm, WeightKind::RawF32),
            (self.k_norm, WeightKind::RawF32),
            (self.ffn_norm, WeightKind::RawF32),
            (self.fused_qkv, WeightKind::Tq2Soa),
            (self.attn_proj, WeightKind::Tq2Soa),
            (self.gate_up, WeightKind::Tq2Soa),
            (self.down, WeightKind::Tq2Soa),
        ]
    }
}

/// GPU weight-cache slots for the ternary final-norm → LM-head tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TernaryTailSlots {
    /// Final RMSNorm (`RawF32`), anchored on the LM head.
    pub final_norm: u64,
    /// LM-head projection (`Tq2Soa`).
    pub lm_head: u64,
}

impl TernaryTailSlots {
    /// Derive the tail slots from the LM head's mapped blocks.
    pub(super) fn for_lm_head(lm_head: &LinearTernary<'_>) -> GpuResult<Self> {
        let anchor = tensor_slot(blocks_as_bytes_ternary(lm_head.blocks()), "output.weight")?;
        Ok(Self {
            final_norm: anchor + SLOT_OFF_FINAL_NORM,
            lm_head: anchor,
        })
    }

    /// The `(slot, kind)` pairs the tail owns — see
    /// [`TernaryLayerSlots::cache_keys`].
    #[cfg(test)]
    pub(super) fn cache_keys(&self) -> [(u64, WeightKind); 2] {
        [
            (self.final_norm, WeightKind::RawF32),
            (self.lm_head, WeightKind::Tq2Soa),
        ]
    }
}

/// Everything a ternary Metal dispatch needs in order to bind its weights.
///
/// Produced by [`BonsaiModel::ternary_gpu_binding`], the shared prologue of
/// every **uncached** ternary dispatch: derive the slots, make each layer's
/// fused Q‖K‖V buffer resident, then build the per-layer parameter structs
/// from borrowed mmap slices, keyed under the model's mapping epoch. Nothing
/// in here owns weight bytes.
///
/// `pub` — and re-exported as `oxibonsai_model::model::TernaryGpuBinding` — so
/// the binding is usable outside the crate: the decode-throughput A/B
/// (`BENCHES`) drives the uncached path with it.
pub struct TernaryGpuBinding<'b> {
    /// Per-layer parameters, in layer order.
    pub layer_params: Vec<FullForwardLayerParamsTernary<'b>>,
    /// Final-norm → LM-head parameters; `None` when the model's output weight
    /// is not ternary (the caller then runs the CPU tail).
    pub tail: Option<TernaryTailBinding<'b>>,
}

/// Final-norm → LM-head half of a [`TernaryGpuBinding`].
pub struct TernaryTailBinding<'b> {
    /// Slot of the final RMSNorm weight.
    pub final_norm_handle: u64,
    /// Borrowed final RMSNorm weights.
    pub final_norm_bytes: &'b [f32],
    /// Slot of the LM-head weight.
    pub lm_head_handle: u64,
    /// Borrowed LM-head blocks, as bytes.
    pub lm_head_bytes: &'b [u8],
    /// Rows of the LM-head matrix (the logits length).
    pub lm_head_out_features: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Upload helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Make one layer's fused Q‖K‖V buffer GPU-resident under `epoch`, building
/// the concatenation only when the slot is not already cached.
///
/// This is the single place the uncached paths materialize the Q‖K‖V layout.
/// On a hit it is one hash lookup and an `Arc` clone; on a miss it allocates
/// the concatenation, hands it to `upload_tq2_weight_soa` and drops it before
/// returning — perf-03's "memoize the concatenated QKV layout ONCE per model,
/// not per call", with the memo living in the GPU cache rather than on the
/// host.
fn ensure_fused_qkv(
    graph: &MetalGraph,
    block: &TransformerBlock<'_>,
    epoch: u64,
    slot: u64,
) -> GpuResult<()> {
    let q_bytes = ternary_bytes(block.attn_q_blocks_ternary(), "attn_q")?;
    let k_bytes = ternary_bytes(block.attn_k_blocks_ternary(), "attn_k")?;
    let v_bytes = ternary_bytes(block.attn_v_blocks_ternary(), "attn_v")?;
    graph
        .get_or_upload_tq2_weight_soa_lazy_for_epoch(epoch, slot, || {
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            concat
        })
        .map_err(|e| -> Box<dyn std::error::Error> {
            format!(
                "fused QKV upload for layer {} (slot {slot:#x}) failed: {e}",
                block.layer_index()
            )
            .into()
        })?;
    Ok(())
}

/// Upload every weight of one ternary layer under `epoch`, keeping no host
/// copy.
///
/// The four RMSNorms and the three single-tensor projections are uploaded
/// straight from their borrowed mmap slices. The two layouts that must be
/// contiguous — Q‖K‖V and gate‖up — are assembled in `scratch`, **one shared
/// buffer reused by every layer** rather than a fresh `Vec` per layer per
/// concatenation, so the load-time staging high-water is one layer's worth
/// (6.7 MB for the 8B) instead of the model's whole QKV + gate/up (1.7 GB).
///
/// All eight uploads are cheap no-ops once the slots are resident; only the two
/// `scratch` fills are unconditional, which is why this runs once at model-load
/// time and the decode path uses [`ensure_fused_qkv`]'s lazy closure instead.
fn upload_ternary_layer(
    graph: &MetalGraph,
    block: &TransformerBlock<'_>,
    slots: &TernaryLayerSlots,
    epoch: u64,
    scratch: &mut Vec<u8>,
) -> GpuResult<()> {
    graph.get_or_upload_f32_weight_for_epoch(epoch, slots.attn_norm, block.attn_norm_weight())?;
    graph.get_or_upload_f32_weight_for_epoch(epoch, slots.q_norm, block.q_norm_weight())?;
    graph.get_or_upload_f32_weight_for_epoch(epoch, slots.k_norm, block.k_norm_weight())?;
    graph.get_or_upload_f32_weight_for_epoch(epoch, slots.ffn_norm, block.ffn_norm_weight())?;

    fill_scratch(
        scratch,
        &[
            ternary_bytes(block.attn_q_blocks_ternary(), "attn_q")?,
            ternary_bytes(block.attn_k_blocks_ternary(), "attn_k")?,
            ternary_bytes(block.attn_v_blocks_ternary(), "attn_v")?,
        ],
    );
    graph.get_or_upload_tq2_weight_soa_for_epoch(epoch, slots.fused_qkv, scratch)?;

    graph.get_or_upload_tq2_weight_soa_for_epoch(
        epoch,
        slots.attn_proj,
        ternary_bytes(block.attn_output_blocks_ternary(), "attn_output")?,
    )?;

    fill_scratch(
        scratch,
        &[
            ternary_bytes(block.ffn_gate_blocks_ternary(), "ffn_gate")?,
            ternary_bytes(block.ffn_up_blocks_ternary(), "ffn_up")?,
        ],
    );
    graph.get_or_upload_tq2_weight_soa_for_epoch(epoch, slots.gate_up, scratch)?;

    graph.get_or_upload_tq2_weight_soa_for_epoch(
        epoch,
        slots.down,
        ternary_bytes(block.ffn_down_blocks_ternary(), "ffn_down")?,
    )?;
    Ok(())
}

/// Refill `scratch` with `parts` concatenated, reusing its allocation.
fn fill_scratch(scratch: &mut Vec<u8>, parts: &[&[u8]]) {
    scratch.clear();
    scratch.reserve(parts.iter().map(|p| p.len()).sum::<usize>());
    for part in parts {
        scratch.extend_from_slice(part);
    }
}

/// Build one layer's dispatch parameters from borrowed mmap slices, keyed
/// under `epoch`.
///
/// Every byte field but `fused_qkv_bytes` borrows straight from the mapping, so
/// this allocates nothing beyond the struct itself; see
/// [`FUSED_QKV_ALREADY_RESIDENT`] for why that one field is empty.
fn ternary_layer_params<'b>(
    block: &'b TransformerBlock<'b>,
    slots: &TernaryLayerSlots,
    epoch: u64,
) -> GpuResult<FullForwardLayerParamsTernary<'b>> {
    Ok(FullForwardLayerParamsTernary {
        model_epoch: epoch,
        attn_norm_handle: slots.attn_norm,
        attn_norm_bytes: block.attn_norm_weight(),
        fused_qkv_handle: slots.fused_qkv,
        fused_qkv_bytes: FUSED_QKV_ALREADY_RESIDENT,
        q_norm_handle: slots.q_norm,
        q_norm_bytes: block.q_norm_weight(),
        k_norm_handle: slots.k_norm,
        k_norm_bytes: block.k_norm_weight(),
        attn_proj_handle: slots.attn_proj,
        attn_proj_bytes: ternary_bytes(block.attn_output_blocks_ternary(), "attn_output")?,
        ffn_norm_handle: slots.ffn_norm,
        ffn_norm_bytes: block.ffn_norm_weight(),
        gate_up_handle: slots.gate_up,
        gate_bytes: ternary_bytes(block.ffn_gate_blocks_ternary(), "ffn_gate")?,
        up_bytes: ternary_bytes(block.ffn_up_blocks_ternary(), "ffn_up")?,
        down_handle: slots.down,
        down_bytes: ternary_bytes(block.ffn_down_blocks_ternary(), "ffn_down")?,
    })
}

impl<'a> BonsaiModel<'a> {
    /// The model's ternary LM head, or `None` for any other output weight.
    fn ternary_lm_head(&self) -> Option<&LinearTernary<'a>> {
        match &self.output_weight {
            OutputWeight::Ternary(linear) => Some(linear),
            _ => None,
        }
    }

    /// Derive the whole model's ternary slot table.
    ///
    /// The tail slots are `None` unless the output weight is ternary, which is
    /// the case `try_metal_full_forward_ternary_inner` runs in: ternary blocks
    /// with a CPU-side (or non-ternary) LM head.
    pub(super) fn ternary_gpu_slots(
        &self,
    ) -> GpuResult<(Vec<TernaryLayerSlots>, Option<TernaryTailSlots>)> {
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            layers.push(TernaryLayerSlots::for_block(block)?);
        }
        let tail = match self.ternary_lm_head() {
            Some(lm_head) => Some(TernaryTailSlots::for_lm_head(lm_head)?),
            None => None,
        };
        Ok((layers, tail))
    }

    /// Shared weight-binding prologue of every **uncached** ternary Metal
    /// dispatch.
    ///
    /// Derives the slot table, makes each layer's fused Q‖K‖V buffer resident
    /// under the model's mapping epoch (building the concatenation only on a
    /// cache miss) and returns parameter structs, keyed under that same epoch,
    /// that borrow every other byte straight from the mapping. It allocates one
    /// `Vec` of parameter structs and nothing else — which is why the five
    /// paths that used to deep-copy the whole model per call (perf-03) could
    /// call it on the decode hot path.
    ///
    /// Public (the `BENCHES` enabler): together with the kernels'
    /// `try_metal_*_ternary` entry points this is the **uncached** ternary
    /// dispatch, and [`Self::forward_greedy_gpu_ternary_cached`] /
    /// [`Self::forward_logits_gpu_ternary_cached`] are the cached one — the
    /// two arms of the decode-throughput A/B.
    ///
    /// # Errors
    ///
    /// A model without blocks, a non-ternary layer, an unavailable Metal
    /// device, or a failed fused-QKV upload.
    pub fn ternary_gpu_binding(&self) -> GpuResult<TernaryGpuBinding<'_>> {
        if self.blocks.is_empty() {
            return Err("no blocks".into());
        }
        let graph = MetalGraph::global().map_err(|e| -> Box<dyn std::error::Error> {
            format!("MetalGraph::global: {e}").into()
        })?;
        let (layer_slots, tail_slots) = self.ternary_gpu_slots()?;
        let epoch = self.gpu_mapping_epoch();
        self.metal_q1_slots.mark_used(self.blocks.len());
        let mut layer_params = Vec::with_capacity(self.blocks.len());
        for (block, slots) in self.blocks.iter().zip(layer_slots.iter()) {
            ensure_fused_qkv(&graph, block, epoch, slots.fused_qkv)?;
            layer_params.push(ternary_layer_params(block, slots, epoch)?);
        }
        self.metal_q1_slots.mark_used(self.blocks.len());
        let tail = match (tail_slots, self.ternary_lm_head()) {
            (Some(slots), Some(lm_head)) => Some(TernaryTailBinding {
                final_norm_handle: slots.final_norm,
                final_norm_bytes: self.output_norm.weight(),
                lm_head_handle: slots.lm_head,
                lm_head_bytes: blocks_as_bytes_ternary(lm_head.blocks()),
                lm_head_out_features: lm_head.out_features(),
            }),
            _ => None,
        };
        Ok(TernaryGpuBinding { layer_params, tail })
    }

    /// Drop this model's mapping's Metal weight buffers from the process-wide
    /// cache — everything keyed under [`Self::gpu_mapping_epoch`]: every
    /// ternary weight of a ternary model, the norms / final norm / LM head of
    /// a Q1 one — in one `MetalGraph::release_model(epoch)`.
    ///
    /// `MetalGraph`'s weight cache never evicts on its own, so without this (or
    /// the last replica's drop, which does the same) a model that is unloaded
    /// keeps its whole quantized self resident on the GPU — 7.2 GB per load for
    /// the 27B `PQ2_0`. The buffers themselves go away once the last
    /// outstanding `Arc<MetalWeightHandle>` (an in-flight dispatch, for
    /// instance) is released — which is why the model's own cached handle set
    /// (`gpu_weight_cache`) is dropped here too — and
    /// `MetalGraph::bytes_uploaded` falls by their size immediately.
    ///
    /// Returns the number of cache entries released; `Ok(0)`, without touching
    /// (or initialising) the Metal graph, for a model whose GPU paths never
    /// ran (or were already released).
    ///
    /// Engine-pool replicas built from one `GgufFile` deliberately share this
    /// epoch, so this releases weights that the surviving replicas would then
    /// have to re-upload on their next dispatch; the last replica's drop
    /// releases them anyway, so call this only to free a mapping early.
    ///
    /// The GPU cache is not consulted again until `get_or_create_gpu_cache`
    /// re-runs, so the model's `gpu_weight_cache` marker is cleared too.
    ///
    /// # Errors
    ///
    /// A poisoned `gpu_weight_cache` lock, or an unreachable Metal graph (the
    /// namespace then stays marked in use, so a later call — or the last
    /// replica's drop — releases it).
    pub fn release_metal_weights(&self) -> GpuResult<usize> {
        {
            let mut guard = self
                .gpu_weight_cache
                .lock()
                .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
            *guard = None;
        }
        let released = self.metal_q1_slots.release()?;
        if released > 0 {
            tracing::info!(
                released,
                epoch = self.gpu_mapping_epoch(),
                "released this model's Metal weight buffers from the process-wide cache"
            );
        }
        Ok(released)
    }

    /// Build and cache GPU weight handles on first call; no-op afterwards.
    ///
    /// For Q1 models the cache contains pre-uploaded `CachedLayerWeights`
    /// handles plus a pre-uploaded LM-head handle — used by
    /// `try_metal_full_forward_cached`. They are keyed exactly like the
    /// uncached fused paths (`Self::q1_layer_params`; the norms and the LM
    /// head under the mapping epoch), so a model that already prefilled
    /// uploads nothing here, and every replica of one mapping hits the first
    /// replica's buffers (`MET-02`, Q1 half).
    ///
    /// For ternary (TQ2_0_g128) models the *weights* are uploaded here and the
    /// stored `CachedModelWeights::Ternary` value carries no bytes at all; see
    /// the module docs (MET-03).
    pub fn get_or_create_gpu_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        {
            let guard = self
                .gpu_weight_cache
                .lock()
                .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
            if guard.is_some() {
                return Ok(());
            }
        }
        let n_layers = self.blocks.len();

        // ── Ternary path ──────────────────────────────────────────────────────
        if let OutputWeight::Ternary(ref lm_head_ternary) = self.output_weight {
            return self.build_ternary_gpu_cache(n_layers, lm_head_ternary);
        }

        // ── Q1 path ───────────────────────────────────────────────────────────
        if self.blocks.iter().any(|b| b.attn_q_blocks().is_none()) {
            return Ok(());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(ref linear) => linear,
            OutputWeight::Fp32 { .. } => return Err("FP32 LM head not supported".into()),
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head reached the Q1 GPU cache builder".into())
            }
            OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_) => {
                return Err("FP8 LM head not supported on Metal GPU cache path".into());
            }
            OutputWeight::Q4_0(_)
            | OutputWeight::Q8_0(_)
            | OutputWeight::Q5K(_)
            | OutputWeight::Q6K(_)
            | OutputWeight::Q2K(_)
            | OutputWeight::Q3K(_)
            | OutputWeight::Q4K(_)
            | OutputWeight::Q8K(_) => {
                return Err(
                    "K-quant / Q-type LM head not yet supported on Metal GPU cache path; use CPU path"
                        .into(),
                );
            }
        };
        let qkv_concats = self.q1_qkv_concats()?;
        let layer_params = self.q1_layer_params(&qkv_concats)?;
        let final_norm_handle = self.metal_q1_slots.final_norm();
        let final_norm_bytes = self.output_norm.weight();
        let lm_head_handle = self.metal_q1_slots.lm_head();
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let cached = oxibonsai_kernels::build_cached_weights(
            &layer_params,
            final_norm_handle,
            final_norm_bytes,
            lm_head_handle,
            lm_head_bytes,
        );
        self.metal_q1_slots.mark_used(n_layers);
        let cached = cached.map_err(|e| format!("build_cached_weights: {e}"))?;
        let mut guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        *guard = Some(cached);
        tracing::info!("GPU weight cache populated (Q1; all subsequent tokens use cached handles)");
        Ok(())
    }

    /// Upload the ternary (TQ2_0_g128) weights and cache their GPU handles.
    ///
    /// Every projection goes to the GPU here, once, keyed by the address of the
    /// mapped tensor it came from under the model's mapping epoch, through one
    /// reused staging buffer (see [`upload_ternary_layer`]). The kernels'
    /// `build_cached_weights_ternary_only` then resolves those now-resident
    /// buffers — pure cache hits — into the MET-03 shape: eight
    /// `Arc<MetalWeightHandle>` per layer plus the tail, and **no host
    /// bytes**. That value is both the "weights are up" marker that makes this
    /// function idempotent and what the cached decode entry points bind
    /// directly.
    fn build_ternary_gpu_cache(
        &self,
        n_layers: usize,
        lm_head_ternary: &LinearTernary<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let graph = MetalGraph::global().map_err(|e| -> Box<dyn std::error::Error> {
            format!("MetalGraph::global: {e}").into()
        })?;
        let epoch = self.gpu_mapping_epoch();
        let mut layer_slots = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            layer_slots.push(TernaryLayerSlots::for_block(block)?);
        }
        self.metal_q1_slots.mark_used(n_layers);
        // One staging buffer for the whole model: see `upload_ternary_layer`.
        let mut scratch: Vec<u8> = Vec::new();
        for (block, slots) in self.blocks.iter().zip(layer_slots.iter()) {
            upload_ternary_layer(&graph, block, slots, epoch, &mut scratch)?;
        }
        drop(scratch);
        let tail = TernaryTailSlots::for_lm_head(lm_head_ternary)?;
        let lm_head_bytes = blocks_as_bytes_ternary(lm_head_ternary.blocks());
        let ternary_lm_head_out_features = lm_head_ternary.out_features();

        // Every weight is resident now, so the builder's lookups are all hits:
        // `fused_qkv_bytes` is the empty `FUSED_QKV_ALREADY_RESIDENT` and the
        // gate‖up closure never runs. Its parameter structs borrow from the
        // mapping and are dropped at the end of this function.
        let mut layer_params = Vec::with_capacity(n_layers);
        for (block, slots) in self.blocks.iter().zip(layer_slots.iter()) {
            layer_params.push(ternary_layer_params(block, slots, epoch)?);
        }
        let cached = oxibonsai_kernels::build_cached_weights_ternary_only(
            &layer_params,
            Some((tail.final_norm, self.output_norm.weight())),
            Some((tail.lm_head, lm_head_bytes)),
            ternary_lm_head_out_features,
        );
        self.metal_q1_slots.mark_used(n_layers);
        let cached = cached.map_err(|e| format!("build_cached_weights_ternary_only: {e}"))?;
        drop(layer_params);

        let mut guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        *guard = Some(cached);
        drop(guard);
        tracing::info!(
            "GPU weight cache populated (ternary; {} layers uploaded, {} lm-head output features, \
             no host copy retained, mapping epoch {epoch})",
            n_layers,
            ternary_lm_head_out_features,
        );
        Ok(())
    }

    /// Run `f` against this model's ternary GPU weight cache (`MET-03`),
    /// building it first if needed.
    ///
    /// The seam every cached ternary dispatch goes through: it guarantees the
    /// cache is populated and is the **ternary** variant (a Q1 cache on a
    /// ternary model is an error, not a misdecode), and it holds the cache lock
    /// only for the duration of `f` — one dispatch.
    pub(super) fn with_ternary_gpu_cache<R>(
        &self,
        f: impl FnOnce(&CachedModelWeights) -> R,
    ) -> GpuResult<R> {
        self.get_or_create_gpu_cache()?;
        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        let cached = guard.as_ref().ok_or("GPU weight cache not populated")?;
        if !matches!(cached, CachedModelWeights::Ternary(_)) {
            return Err("ternary GPU path invoked with a Q1 weight cache".into());
        }
        Ok(f(cached))
    }

    /// Shared guard + embedding prologue of the single-token ternary decode
    /// entry points (cached and uncached): the context-length and
    /// sliding-window refusals, then this token's hidden state and RoPE rows.
    pub(super) fn ternary_decode_prologue(
        &self,
        token_id: u32,
        pos: usize,
    ) -> GpuResult<(Vec<f32>, &[f32], &[f32])> {
        if !matches!(self.output_weight, OutputWeight::Ternary(_)) {
            return Err("cached ternary decode called on a non-ternary model".into());
        }
        if self.blocks.is_empty() {
            return Err("no blocks".into());
        }
        if pos >= self.kv_cache.max_seq_len() {
            return Err(format!(
                "ternary cached decode sequence too long: pos {pos} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        // The fused path attends over the full device KV cache with no
        // windowing; a sliding-window model must take the windowed CPU path
        // (M-17, same refusal as `forward_greedy_gpu`).
        if self.config.sliding_window.is_some() {
            return Err(
                "sliding-window model: fused Metal decode is full-causal; falling back to CPU"
                    .into(),
            );
        }
        let mut hidden = vec![0.0f32; self.config.hidden_size];
        self.token_embd.copy_row(token_id, &mut hidden)?;
        let rope_cos = self.rope.cos_at_checked(pos)?;
        let rope_sin = self.rope.sin_at_checked(pos)?;
        Ok((hidden, rope_cos, rope_sin))
    }

    /// Greedy single-token decode through the **cached** ternary GPU weights
    /// (`MET-03`): all layers, the final norm, the TQ2 LM head and the argmax
    /// in one command buffer, binding the handles `get_or_create_gpu_cache`
    /// resolved once instead of looking all eight per layer up again.
    ///
    /// Bit-identical to the uncached binding on a ternary model (same buffers,
    /// same kernels). Maintains the device KV cache of the session this thread
    /// dispatches in.
    ///
    /// # Errors
    ///
    /// A non-ternary model, a position past the context, a sliding-window
    /// model, a Q1 cache, or any Metal failure — never a CPU fallback.
    pub fn forward_greedy_gpu_ternary_cached(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let (mut hidden, rope_cos, rope_sin) = self.ternary_decode_prologue(token_id, pos)?;
        let mut token: u32 = 0;
        self.with_ternary_gpu_cache(|cached| {
            oxibonsai_kernels::try_metal_forward_greedy_ternary_cached(
                &mut hidden,
                pos,
                cached,
                rope_cos,
                rope_sin,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.blocks[0].attn_norm_eps(),
                self.kv_cache.max_seq_len(),
                self.output_norm.eps(),
                &mut token,
            )
        })??;
        // MET-05: the device KV cache now holds this position, the host one
        // does not.
        self.note_device_kv_used();
        Ok(token)
    }

    /// Logits of one token through the **cached** ternary GPU weights
    /// (`MET-03`) — the sampled-decode twin of
    /// [`Self::forward_greedy_gpu_ternary_cached`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward_greedy_gpu_ternary_cached`].
    pub fn forward_logits_gpu_ternary_cached(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (mut hidden, rope_cos, rope_sin) = self.ternary_decode_prologue(token_id, pos)?;
        let mut logits = Vec::new();
        self.with_ternary_gpu_cache(|cached| {
            oxibonsai_kernels::try_metal_prefill_ternary_cached(
                &mut hidden,
                pos,
                cached,
                rope_cos,
                rope_sin,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.blocks[0].attn_norm_eps(),
                self.kv_cache.max_seq_len(),
                self.output_norm.eps(),
                &mut logits,
            )
        })??;
        self.note_device_kv_used();
        Ok(logits)
    }

    /// Embed `token_ids` column-major and build the per-position RoPE tables
    /// for a batched ternary prefill (cached or uncached) starting at
    /// `pos_start`.
    #[allow(clippy::type_complexity)]
    pub(super) fn ternary_prefill_inputs(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> GpuResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if !matches!(self.output_weight, OutputWeight::Ternary(_)) {
            return Err("cached ternary prefill called on a non-ternary model".into());
        }
        if self.blocks.is_empty() {
            return Err("no blocks".into());
        }
        let batch = token_ids.len();
        if batch == 0 {
            return Err("cached ternary prefill needs at least one token".into());
        }
        if pos_start + batch > self.kv_cache.max_seq_len() {
            return Err(format!(
                "ternary cached prefill sequence too long: {batch} tokens at pos {pos_start} \
                 exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        let h = self.config.hidden_size;
        let half_dim = self.config.head_dim / 2;
        let mut hidden_batch = vec![0.0f32; batch * h];
        self.token_embd.copy_rows(token_ids, &mut hidden_batch)?;
        let mut cos_table = vec![0.0f32; batch * half_dim];
        let mut sin_table = vec![0.0f32; batch * half_dim];
        for t in 0..batch {
            let pos = pos_start + t;
            cos_table[t * half_dim..(t + 1) * half_dim]
                .copy_from_slice(self.rope.cos_at_checked(pos)?);
            sin_table[t * half_dim..(t + 1) * half_dim]
                .copy_from_slice(self.rope.sin_at_checked(pos)?);
        }
        Ok((hidden_batch, cos_table, sin_table))
    }

    /// Batched prefill through the **cached** ternary GPU weights (`MET-03`):
    /// the cached twin of [`Self::prefill_logits_gpu_ternary_uncached`],
    /// returning the last position's logits.
    ///
    /// It performs no weight-cache lookup at all.
    ///
    /// # Errors
    ///
    /// A non-ternary model, an empty or over-long batch, a Q1 cache, or any
    /// Metal failure — never a CPU fallback.
    pub fn prefill_logits_gpu_ternary_cached(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (hidden_batch, cos_table, sin_table) =
            self.ternary_prefill_inputs(token_ids, pos_start)?;
        let mut logits = Vec::new();
        self.with_ternary_gpu_cache(|cached| {
            oxibonsai_kernels::try_metal_full_forward_prefill_ternary_cached(
                &hidden_batch,
                token_ids.len(),
                pos_start,
                cached,
                &cos_table,
                &sin_table,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.blocks[0].attn_norm_eps(),
                self.kv_cache.max_seq_len(),
                self.output_norm.eps(),
                Some(&mut logits),
                None,
            )
        })??;
        self.note_device_kv_used();
        Ok(logits)
    }

    /// Batched speculative-verify prefill through the **cached** ternary GPU
    /// weights (`MET-03`): every position's greedy argmax — the cached twin of
    /// [`Self::prefill_verify_gpu_ternary_uncached`].
    ///
    /// # Errors
    ///
    /// As [`Self::prefill_logits_gpu_ternary_cached`].
    pub fn prefill_verify_gpu_ternary_cached(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let (hidden_batch, cos_table, sin_table) =
            self.ternary_prefill_inputs(token_ids, pos_start)?;
        let mut ids = Vec::with_capacity(token_ids.len());
        self.with_ternary_gpu_cache(|cached| {
            oxibonsai_kernels::try_metal_full_forward_prefill_verify_ternary_cached(
                &hidden_batch,
                token_ids.len(),
                pos_start,
                cached,
                &cos_table,
                &sin_table,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.blocks[0].attn_norm_eps(),
                self.kv_cache.max_seq_len(),
                self.output_norm.eps(),
                &mut ids,
            )
        })??;
        self.note_device_kv_used();
        Ok(ids)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — slot identity (MET-02), host-copy freedom (MET-03), no per-call
// model copy (perf-03), the per-mapping epoch, release-on-unload and the
// last-replica release
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[path = "gpu_cache_tests.rs"]
mod tests;
