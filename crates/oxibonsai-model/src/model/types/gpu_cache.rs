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
//! bases additionally aliased the Q1 `final_norm` / `lm_head` handles — and
//! `2_000_000` aliased the Q1 `final_norm` under the *same*
//! [`WeightKind::RawF32`], i.e. as a silent stale hit rather than an error.
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
//!   x86_64), so a mapped-tensor slot is always ≥ [`MIN_TENSOR_SLOT`] while the
//!   Q1 literals are all below 2²². [`tensor_slot`] enforces the floor instead
//!   of assuming it.
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
//! once and the `CachedModelWeights::Ternary` value it stores carries **no bytes
//! at all** — only `lm_head_out_features`. Every forward path rebuilds its
//! `FullForwardLayerParamsTernary` from borrowed mmap slices, and the two
//! layouts that need a real allocation (the Q‖K‖V concatenation and the
//! gate‖up concatenation) are built inside `get_or_upload_*_lazy` closures, so
//! they exist only on a cache miss and are dropped as soon as the GPU buffer is
//! written.
//!
//! # Release on unload, without a struct field on `BonsaiModel` (spec 1(b)/1(c))
//!
//! The wave-1 `METAL-CACHE` package built a composite `WeightKey { model_epoch,
//! kind, slot }` precisely so a model's whole GPU footprint can be dropped by
//! epoch in one `MetalGraph::release_model(epoch)` call. The full design
//! threads a `metal_model_epoch: u64` field through `BonsaiModel`, allocated
//! once per load by `MetalGraph::next_model_epoch()`. `BonsaiModel` is defined
//! in `mod.rs`, owned by `MODEL-CORE-FWD` this wave, so that field cannot land
//! from this package — see the package deviations.
//!
//! What *can* land here, from data this module already computes, is the
//! safety property the epoch exists for: **a model's slots never outlive the
//! mapping they were derived from.** Every ternary slot is already an address
//! derived from a mapped tensor ([`tensor_slot`]), and that address is by
//! construction identical for every engine-pool replica sharing one `GgufFile`
//! and — *only once the old mapping is actually gone* — free to be reused by
//! an unrelated later load. So instead of a second identity scheme,
//! `BonsaiModel`'s [`Drop`] impl below refcounts that same address in a
//! process-wide table ([`TERNARY_MODEL_REFCOUNTS`]): every replica that
//! populates the shared ternary cache registers once (in
//! [`BonsaiModel::build_ternary_gpu_cache`]), and the buffers are released —
//! via the existing [`BonsaiModel::release_metal_weights`] — only when the
//! *last* registered replica drops. A bare, unconditional call there would be
//! wrong: replicas deliberately share these slots (MET-02), so evicting on
//! every replica's `Drop` would pull buffers out from under siblings that are
//! still binding them, forcing a mid-serve re-upload of the whole model.
//!
//! This is reconciliation option (ii) from the verifier's review of the first
//! attempt at this fix — keep [`WeightKind`]-legacy epoch-0 keying and make
//! `Drop` refcount-aware — rather than option (i), a real epoch derived from
//! the mapping identity: the two are equally safe once eviction is correctly
//! refcounted, and (ii) needs no new keying scheme for every existing caller
//! of `WeightKey::legacy` to agree on, and no change to what tests and
//! callers already probe the cache with.

use super::{BonsaiModel, OutputWeight};
use crate::block::{blocks_as_bytes, blocks_as_bytes_ternary, TransformerBlock};
use crate::layers::linear::LinearTernary;
use oxibonsai_kernels::gpu_backend::metal_full_layer::types::{WeightKey, WeightKind};
use oxibonsai_kernels::{FullForwardLayerParamsTernary, MetalGraph};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Convenience alias for the boxed error every Metal entry point here returns.
type GpuResult<T> = Result<T, Box<dyn std::error::Error>>;

// ═══════════════════════════════════════════════════════════════════════════
// Slot table — one namespace for every ternary Metal path (MET-02)
// ═══════════════════════════════════════════════════════════════════════════

/// Lowest value a mapped-tensor weight slot is allowed to take.
///
/// macOS maps `__PAGEZERO` over the first 4 GiB of every process on both arm64
/// and x86_64, so no valid pointer — mmap'd GGUF data included — is below
/// `1 << 32`. The Q1 decode path still keys its RMSNorm / final-norm / LM-head
/// buffers on the literals `1_000_000`, `2_000_000` and `3_000_000` (all below
/// 2²²), so anchoring the ternary namespace on real addresses puts the two sets
/// provably out of each other's reach. [`tensor_slot`] checks the floor, so a
/// future platform breaking the assumption produces a named error rather than a
/// silent stale hit.
pub(super) const MIN_TENSOR_SLOT: u64 = 1 << 32;

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
/// Every ternary path therefore runs [`BonsaiModel::ternary_gpu_binding`]
/// first, which builds the concatenation inside a
/// `get_or_upload_tq2_weight_soa_lazy` closure — on a cache miss only — and
/// hands it straight to the GPU. By the time the kernel looks the slot up the
/// buffer is resident and **these bytes are never read**.
///
/// The slice is deliberately *empty* rather than a plausible-looking stand-in:
/// `upload_bytes` rejects an empty input with
/// `MetalGraphError::BufferCreationFailed`, so if the residency invariant is
/// ever broken (an eviction racing a dispatch, a future caller that skips the
/// prologue) the dispatch **fails closed** with a named error and the caller
/// falls back to the CPU path, instead of binding a short or wrong buffer to the
/// GEMV.
pub(super) const FUSED_QKV_ALREADY_RESIDENT: &[u8] = &[];

/// Derive a weight-cache slot from the address of a mapped tensor.
///
/// `what` names the tensor in the error, which is the only thing that
/// distinguishes one floor violation from another.
fn tensor_slot(bytes: &[u8], what: &str) -> GpuResult<u64> {
    let slot = bytes.as_ptr() as u64;
    if slot < MIN_TENSOR_SLOT {
        return Err(format!(
            "ternary GPU slot for `{what}` is {slot:#x}, below the {MIN_TENSOR_SLOT:#x} floor that \
             keeps mapped-tensor slots clear of the Q1 path's literal handles; refusing to key a \
             GPU weight on it"
        )
        .into());
    }
    Ok(slot)
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
/// [`WeightKind::RawF32`] while the anchor itself is [`WeightKind::Tq2Soa`],
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
    /// Concatenated Q‖K‖V projection (`Tq2Soa`), anchored on `attn_q`.
    pub fused_qkv: u64,
    /// Attention output projection (`Tq2Soa`).
    pub attn_proj: u64,
    /// Concatenated gate‖up projection (`Tq2Soa`), anchored on `ffn_gate`.
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

    /// The `(slot, kind)` pairs this layer owns, for eviction on unload.
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

    /// The `(slot, kind)` pairs the tail owns, for eviction on unload.
    pub(super) fn cache_keys(&self) -> [(u64, WeightKind); 2] {
        [
            (self.final_norm, WeightKind::RawF32),
            (self.lm_head, WeightKind::Tq2Soa),
        ]
    }
}

/// Everything a ternary Metal dispatch needs in order to bind its weights.
///
/// Produced by [`BonsaiModel::ternary_gpu_binding`], the shared prologue of all
/// five ternary paths: derive the slots, make each layer's fused Q‖K‖V buffer
/// resident, then build the per-layer parameter structs from borrowed mmap
/// slices. Nothing in here owns weight bytes.
pub(super) struct TernaryGpuBinding<'b> {
    /// Per-layer parameters, in layer order.
    pub layer_params: Vec<FullForwardLayerParamsTernary<'b>>,
    /// Final-norm → LM-head parameters; `None` when the model's output weight
    /// is not ternary (the caller then runs the CPU tail).
    pub tail: Option<TernaryTailBinding<'b>>,
}

/// Final-norm → LM-head half of a [`TernaryGpuBinding`].
pub(super) struct TernaryTailBinding<'b> {
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

/// Make one layer's fused Q‖K‖V buffer GPU-resident, building the
/// concatenation only when the slot is not already cached.
///
/// This is the single place the Q‖K‖V layout is ever materialized. On a hit it
/// is one hash lookup and an `Arc` clone; on a miss it allocates the
/// concatenation, hands it to `upload_tq2_weight_soa` and drops it before
/// returning — perf-03's "memoize the concatenated QKV layout ONCE per model,
/// not per call", with the memo living in the GPU cache rather than on the host.
fn ensure_fused_qkv(graph: &MetalGraph, block: &TransformerBlock<'_>, slot: u64) -> GpuResult<()> {
    let q_bytes = ternary_bytes(block.attn_q_blocks_ternary(), "attn_q")?;
    let k_bytes = ternary_bytes(block.attn_k_blocks_ternary(), "attn_k")?;
    let v_bytes = ternary_bytes(block.attn_v_blocks_ternary(), "attn_v")?;
    graph
        .get_or_upload_tq2_weight_soa_lazy(slot, || {
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

/// Upload every weight of one ternary layer, keeping no host copy.
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
    scratch: &mut Vec<u8>,
) -> GpuResult<()> {
    graph.get_or_upload_f32_weight(slots.attn_norm, block.attn_norm_weight())?;
    graph.get_or_upload_f32_weight(slots.q_norm, block.q_norm_weight())?;
    graph.get_or_upload_f32_weight(slots.k_norm, block.k_norm_weight())?;
    graph.get_or_upload_f32_weight(slots.ffn_norm, block.ffn_norm_weight())?;

    fill_scratch(
        scratch,
        &[
            ternary_bytes(block.attn_q_blocks_ternary(), "attn_q")?,
            ternary_bytes(block.attn_k_blocks_ternary(), "attn_k")?,
            ternary_bytes(block.attn_v_blocks_ternary(), "attn_v")?,
        ],
    );
    graph.get_or_upload_tq2_weight_soa(slots.fused_qkv, scratch)?;

    graph.get_or_upload_tq2_weight_soa(
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
    graph.get_or_upload_tq2_weight_soa(slots.gate_up, scratch)?;

    graph.get_or_upload_tq2_weight_soa(
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

/// Build one layer's dispatch parameters from borrowed mmap slices.
///
/// Every byte field but `fused_qkv_bytes` borrows straight from the mapping, so
/// this allocates nothing beyond the struct itself; see
/// [`FUSED_QKV_ALREADY_RESIDENT`] for why that one field is empty.
fn ternary_layer_params<'b>(
    block: &'b TransformerBlock<'b>,
    slots: &TernaryLayerSlots,
) -> GpuResult<FullForwardLayerParamsTernary<'b>> {
    Ok(FullForwardLayerParamsTernary {
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

// ═══════════════════════════════════════════════════════════════════════════
// Refcounted release on unload — no epoch field; see the module docs
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide replica counts for loaded ternary models, keyed by the same
/// mapped-tensor address [`tensor_slot`] derives a model's GPU cache slots
/// from (see [`BonsaiModel::ternary_model_id`]).
///
/// Incremented once per replica by [`BonsaiModel::register_ternary_replica`]
/// and decremented by [`BonsaiModel::forget_ternary_replica_registration`],
/// which both manual [`BonsaiModel::release_metal_weights`] and `Drop` call.
/// A model's buffers are released only when its count reaches zero, which is
/// what lets several engine-pool replicas share one upload (MET-02) without
/// one replica's `Drop` evicting buffers its siblings are still binding.
static TERNARY_MODEL_REFCOUNTS: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();

/// The process-wide refcount table, created on first use.
fn ternary_model_refcounts() -> &'static Mutex<HashMap<u64, usize>> {
    TERNARY_MODEL_REFCOUNTS.get_or_init(|| Mutex::new(HashMap::new()))
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

    /// Shared weight-binding prologue for every ternary Metal path.
    ///
    /// Derives the slot table, makes each layer's fused Q‖K‖V buffer resident
    /// (building the concatenation only on a cache miss) and returns parameter
    /// structs that borrow every other byte straight from the mapping. It
    /// allocates one `Vec` of parameter structs and nothing else — which is why
    /// the five paths that used to deep-copy the whole model per call (perf-03)
    /// can now call it on the decode hot path.
    pub(super) fn ternary_gpu_binding(&self) -> GpuResult<TernaryGpuBinding<'_>> {
        if self.blocks.is_empty() {
            return Err("no blocks".into());
        }
        let graph = MetalGraph::global().map_err(|e| -> Box<dyn std::error::Error> {
            format!("MetalGraph::global: {e}").into()
        })?;
        let (layer_slots, tail_slots) = self.ternary_gpu_slots()?;
        let mut layer_params = Vec::with_capacity(self.blocks.len());
        for (block, slots) in self.blocks.iter().zip(layer_slots.iter()) {
            ensure_fused_qkv(&graph, block, slots.fused_qkv)?;
            layer_params.push(ternary_layer_params(block, slots)?);
        }
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

    /// Stable identity for this model's GPU-cache slots, shared by every
    /// engine-pool replica built from the same `GgufFile` (MET-02: all such
    /// replicas resolve to the same tensor addresses). `None` for a model
    /// with no ternary layers — nothing here to refcount.
    fn ternary_model_id(&self) -> Option<u64> {
        if let Some(block) = self.blocks.first() {
            return TernaryLayerSlots::for_block(block)
                .ok()
                .map(|s| s.fused_qkv);
        }
        // No transformer blocks (a bare ternary-LM-head fixture): fall back
        // to the tail slot, so a model like that still has an identity.
        TernaryTailSlots::for_lm_head(self.ternary_lm_head()?)
            .ok()
            .map(|t| t.lm_head)
    }

    /// Record that this replica now shares the process-wide ternary GPU
    /// cache, so `Drop` waits for every sibling replica before releasing the
    /// buffers they all share.
    ///
    /// Called by [`Self::build_ternary_gpu_cache`] only on the call that
    /// actually transitions `gpu_weight_cache` from empty to populated —
    /// `get_or_create_gpu_cache`'s own not-yet-populated check and that
    /// transition are two separate lock acquisitions on `gpu_weight_cache`,
    /// so without that guard two callers racing to warm the same
    /// never-yet-populated replica could both reach here and both register,
    /// over-counting that one replica by one and leaking its buffers past its
    /// actual last `Drop`.
    fn register_ternary_replica(&self) {
        if let Some(id) = self.ternary_model_id() {
            let mut counts = ternary_model_refcounts()
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            *counts.entry(id).or_insert(0) += 1;
        }
    }

    /// Undo this replica's [`Self::register_ternary_replica`], if it ever
    /// registered; otherwise a no-op.
    ///
    /// Called by both `Drop` and manual [`Self::release_metal_weights`] so
    /// the refcount table never drifts from reality: a model released
    /// manually — the documented "call it when the last replica goes"
    /// contract — must not leave a stale count behind for some later,
    /// unrelated model to inherit if it happens to reuse these addresses.
    fn forget_ternary_replica_registration(&self) {
        let Some(id) = self.ternary_model_id() else {
            return;
        };
        let mut counts = ternary_model_refcounts()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        match counts.get_mut(&id) {
            Some(count) if *count > 1 => *count -= 1,
            Some(_) => {
                counts.remove(&id);
            }
            None => {}
        }
    }

    /// Drop this model's Metal weight buffers from the process-wide cache.
    ///
    /// `MetalGraph`'s weight cache never evicts on its own, so without this a
    /// model that is unloaded keeps its whole quantized self resident on the GPU
    /// — 7.2 GB per load for the 27B `PQ2_0`. Call it before dropping a model
    /// whose weights are no longer wanted; the buffers themselves go away once
    /// the last outstanding `Arc<MetalWeightHandle>` (an in-flight dispatch, for
    /// instance) is released, and `MetalGraph::bytes_uploaded` falls by their
    /// size immediately.
    ///
    /// Returns the number of cache slots released. A model with no ternary
    /// layers returns `Ok(0)`: the Q1 path still keys its norms on process-wide
    /// literals shared by every concurrently loaded Q1 model, so evicting those
    /// here would pull buffers out from under a live sibling.
    ///
    /// Engine-pool replicas built from one `GgufFile` deliberately share slots,
    /// so this releases weights that the surviving replicas would then have to
    /// re-upload on their next dispatch; call it when the last replica goes.
    ///
    /// The GPU cache is not consulted again until `get_or_create_gpu_cache`
    /// re-runs, so the model's `gpu_weight_cache` marker is cleared too.
    pub fn release_metal_weights(&self) -> GpuResult<usize> {
        // Clear this replica's own bookkeeping first (a no-op if it never
        // registered) so a manual release never leaves a stale refcount entry
        // for some later, unrelated model to inherit if it happens to reuse
        // these addresses; see `forget_ternary_replica_registration`.
        self.forget_ternary_replica_registration();
        let (layer_slots, tail_slots) = match self.ternary_gpu_slots() {
            Ok(slots) => slots,
            // A non-ternary (or partially ternary) model owns no address-keyed
            // slots; there is nothing of ours to release. Logged at `debug`
            // rather than silently returning 0, so a genuine floor violation
            // (see `tensor_slot`) or a half-ternary layer does not vanish
            // indistinguishably from "this model never touched the GPU".
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "release_metal_weights: not an all-ternary model, nothing of ours to release"
                );
                return Ok(0);
            }
        };
        if layer_slots.is_empty() && tail_slots.is_none() {
            return Ok(0);
        }
        let graph = MetalGraph::global().map_err(|e| -> Box<dyn std::error::Error> {
            format!("MetalGraph::global: {e}").into()
        })?;
        let mut released = 0usize;
        for slots in &layer_slots {
            for (slot, kind) in slots.cache_keys() {
                graph.evict_weight(WeightKey::legacy(kind, slot))?;
                released += 1;
            }
        }
        if let Some(tail) = tail_slots {
            for (slot, kind) in tail.cache_keys() {
                graph.evict_weight(WeightKey::legacy(kind, slot))?;
                released += 1;
            }
        }
        {
            let mut guard = self
                .gpu_weight_cache
                .lock()
                .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
            *guard = None;
        }
        tracing::info!(
            released,
            "released this model's Metal weight buffers from the process-wide cache"
        );
        Ok(released)
    }

    /// Build and cache GPU weight handles on first call; no-op afterwards.
    ///
    /// For Q1 models the cache contains pre-uploaded `CachedLayerWeights`
    /// handles plus a pre-uploaded LM-head handle — used by
    /// `try_metal_full_forward_cached`.
    ///
    /// For ternary (TQ2_0_g128) models the *weights* are uploaded here and the
    /// stored `CachedModelWeights::Ternary` value carries no bytes at all; see
    /// the module docs (MET-03).
    pub fn get_or_create_gpu_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        use oxibonsai_kernels::FullForwardLayerParams;
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
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        let mut layer_params: Vec<FullForwardLayerParams<'_>> = Vec::with_capacity(n_layers);
        for (i, block) in self.blocks.iter().enumerate() {
            let norm_handle_base = 1_000_000u64 + (block.layer_index() as u64) * 10;
            layer_params.push(FullForwardLayerParams {
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: block
                    .fused_qkv_gpu_handle()
                    .map(|hnd| hnd.id())
                    .ok_or_else(|| {
                        format!(
                            "missing GPU handle for layer {} fused_qkv",
                            block.layer_index()
                        )
                    })?,
                fused_qkv_bytes: &qkv_concats[i],
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: block
                    .attn_output_gpu_handle()
                    .map(|hnd| hnd.id())
                    .ok_or_else(|| {
                        format!(
                            "missing GPU handle for layer {} attn_proj",
                            block.layer_index()
                        )
                    })?,
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: block
                    .fused_gate_up_gpu_handle()
                    .map(|hnd| hnd.id())
                    .ok_or_else(|| {
                        format!(
                            "missing GPU handle for layer {} gate_up",
                            block.layer_index()
                        )
                    })?,
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: block
                    .ffn_down_gpu_handle()
                    .map(|hnd| hnd.id())
                    .ok_or_else(|| {
                        format!("missing GPU handle for layer {} down", block.layer_index())
                    })?,
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        let final_norm_handle = 2_000_000u64;
        let final_norm_bytes = self.output_norm.weight();
        let lm_head_handle = 3_000_000u64;
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let cached = oxibonsai_kernels::build_cached_weights(
            &layer_params,
            final_norm_handle,
            final_norm_bytes,
            lm_head_handle,
            lm_head_bytes,
        )
        .map_err(|e| format!("build_cached_weights: {e}"))?;
        let mut guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        *guard = Some(cached);
        tracing::info!("GPU weight cache populated (Q1; all subsequent tokens use cached handles)");
        Ok(())
    }

    /// Upload the ternary (TQ2_0_g128) weights and record that it happened.
    ///
    /// Every projection goes to the GPU here, once, keyed by the address of the
    /// mapped tensor it came from, and **no host bytes are retained**: the
    /// `CachedModelWeights::Ternary` value stored on the model carries empty
    /// vectors and only `lm_head_out_features` is meaningful (MET-03). It exists
    /// purely as the "weights are up" marker that makes this function idempotent
    /// and that lets a decode path notice a Q1 cache on a ternary model.
    fn build_ternary_gpu_cache(
        &self,
        n_layers: usize,
        lm_head_ternary: &LinearTernary<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let graph = MetalGraph::global().map_err(|e| -> Box<dyn std::error::Error> {
            format!("MetalGraph::global: {e}").into()
        })?;
        let mut layer_slots = Vec::with_capacity(n_layers);
        for block in &self.blocks {
            layer_slots.push(TernaryLayerSlots::for_block(block)?);
        }
        // One staging buffer for the whole model: see `upload_ternary_layer`.
        let mut scratch: Vec<u8> = Vec::new();
        for (block, slots) in self.blocks.iter().zip(layer_slots.iter()) {
            upload_ternary_layer(&graph, block, slots, &mut scratch)?;
        }
        drop(scratch);
        let tail = TernaryTailSlots::for_lm_head(lm_head_ternary)?;
        graph.get_or_upload_f32_weight(tail.final_norm, self.output_norm.weight())?;
        graph.get_or_upload_tq2_weight_soa(
            tail.lm_head,
            blocks_as_bytes_ternary(lm_head_ternary.blocks()),
        )?;
        let ternary_lm_head_out_features = lm_head_ternary.out_features();

        // The ternary-only builder takes the byte blobs by value; passing empty
        // vectors is what makes "the cache holds no host copy of the weights"
        // true by construction. `Vec::new()` does not allocate.
        let cached = oxibonsai_kernels::build_cached_weights_ternary_only(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            ternary_lm_head_out_features,
        )
        .map_err(|e| format!("build_cached_weights_ternary_only: {e}"))?;

        let mut guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        // Register this replica only on the call that actually transitions
        // the cache from empty to populated, checked and set under one
        // continuous lock hold — see `register_ternary_replica`'s doc for the
        // double-count this closes.
        let was_empty = guard.is_none();
        *guard = Some(cached);
        drop(guard);
        if was_empty {
            self.register_ternary_replica();
        }
        tracing::info!(
            "GPU weight cache populated (ternary; {} layers uploaded, {} lm-head output features, \
             no host copy retained)",
            n_layers,
            ternary_lm_head_out_features,
        );
        Ok(())
    }

    /// The LM-head row count recorded when the ternary GPU cache was built.
    ///
    /// Also the guard that a ternary decode path is not running against a Q1
    /// cache, which the byte-slice length checks used to provide.
    pub(super) fn ternary_gpu_cache_out_features(&self) -> GpuResult<usize> {
        use oxibonsai_kernels::CachedModelWeights;
        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        match guard.as_ref().ok_or("GPU weight cache not populated")? {
            CachedModelWeights::Ternary(tern) => Ok(tern.lm_head_out_features),
            CachedModelWeights::Q1(_) => {
                Err("ternary GPU path invoked with a Q1 weight cache".into())
            }
        }
    }
}

/// Free a ternary model's Metal weight buffers once its last live replica is
/// dropped (spec 1(c); ACCEPTANCE "`release_model` drops the handles on
/// Drop").
///
/// Engine-pool replicas deliberately SHARE their ternary GPU-cache slots
/// (MET-02: they resolve to the same mapped-tensor addresses), so a bare,
/// unconditional call to [`BonsaiModel::release_metal_weights`] here would be
/// wrong — it would evict buffers a still-live sibling replica is actively
/// binding, forcing a mid-serve re-upload of the whole model (7.2 GB for the
/// 27B `PQ2_0`). This only evicts once the refcount this replica itself
/// registered (in [`BonsaiModel::build_ternary_gpu_cache`]) reaches zero,
/// i.e. when the replica dropping is the last one sharing these slots. See
/// the module docs for why this refcount, rather than a per-model `WeightKey`
/// epoch, is what lands from this package.
///
/// A replica that never populated the ternary GPU cache — CPU-only inference,
/// a Q1/non-ternary model, or a weightless test fixture, overwhelmingly the
/// common case — never registered a count, so this is one `Mutex` lock and a
/// non-match on the cached-weights variant: negligible.
impl<'a> Drop for BonsaiModel<'a> {
    fn drop(&mut self) {
        let is_ternary_cache = self
            .gpu_weight_cache
            .lock()
            .map(|guard| {
                matches!(
                    *guard,
                    Some(oxibonsai_kernels::CachedModelWeights::Ternary(_))
                )
            })
            .unwrap_or(false);
        if !is_ternary_cache {
            return;
        }
        let Some(id) = self.ternary_model_id() else {
            return;
        };
        let last_replica = {
            let mut counts = ternary_model_refcounts()
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            match counts.get_mut(&id) {
                Some(count) if *count > 1 => {
                    *count -= 1;
                    false
                }
                Some(_) => {
                    counts.remove(&id);
                    true
                }
                // Not registered under this id — should not happen given
                // `is_ternary_cache` just confirmed this replica populated
                // the cache, but release rather than leak the buffers
                // forever if it somehow does.
                None => true,
            }
        };
        if last_replica {
            if let Err(e) = self.release_metal_weights() {
                tracing::debug!(
                    error = %e,
                    "BonsaiModel::drop: release_metal_weights failed; its GPU buffers may outlive it"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — slot identity (MET-02), host-copy freedom (MET-03), no per-call
// model copy (perf-03), release-on-unload and refcounted `Drop`
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_alloc::count_allocations;
    use half::f16;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
    use oxibonsai_kernels::{CachedModelWeights, MetalGraphError, MetalWeightHandle};
    use std::sync::{Arc, Mutex, MutexGuard};

    // ── Synthetic fully-ternary fixture ──────────────────────────────────

    /// Hidden size of the fixture (≥ 128: one `TQ2_0_g128` block).
    const FIXTURE_HIDDEN: usize = 128;
    /// FFN intermediate size of the fixture (multiple of 128).
    const FIXTURE_INTER: usize = 256;
    /// Transformer layers in the fixture.
    const FIXTURE_LAYERS: usize = 2;
    /// Attention heads in the fixture.
    const FIXTURE_NQ: usize = 4;
    /// KV heads in the fixture.
    const FIXTURE_NKV: usize = 2;
    /// Head dimension of the fixture (`FIXTURE_HIDDEN / FIXTURE_NQ`).
    const FIXTURE_HD: usize = 32;
    /// Vocabulary size of the fixture.
    const FIXTURE_VOCAB: usize = 32;
    /// Context length the fixture's models are built with.
    const FIXTURE_MAX_SEQ: usize = 64;

    /// Build a `TQ2_0_g128` blob that emits only the three ternary codes.
    ///
    /// Blocks are 34 bytes: 32 bytes of 2-bit codes (four per byte, LSB-first)
    /// then an f16 scale. The reserved code `0b11` is the `PQ2_0` `+2` encoding
    /// and `upload_tq2_weight_soa` rejects any block containing it, so the
    /// generator folds each lane into `{0, 1, 2}`.
    fn tq2_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
        assert_eq!(
            num_weights % 128,
            0,
            "num_weights must be a multiple of 128"
        );
        let mut data = Vec::with_capacity(num_weights / 128 * 34);
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for _ in 0..num_weights / 128 {
            for _ in 0..32 {
                let mut byte = 0u8;
                for lane in 0..4 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    byte |= (((state >> 33) % 3) as u8) << (2 * lane);
                }
                data.push(byte);
            }
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let scale = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
            data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        data
    }

    /// Build an index-varying FP32 tensor, so nothing degenerates to a constant.
    fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            let val = scale * (1.0_f32 + 0.25_f32 * ((i as f32) * 0.013_f32).sin());
            v.extend_from_slice(&val.to_le_bytes());
        }
        v
    }

    /// Assemble a synthetic fully-ternary GGUF in memory.
    ///
    /// `salt` perturbs the weight seeds so two fixtures are distinguishable;
    /// each `Vec` is its own allocation, so two fixtures also mean two distinct
    /// sets of tensor addresses — which is what the slot table keys on.
    fn synthetic_ternary_gguf(salt: u64) -> Vec<u8> {
        let (h, inter, nq, nkv, hd, vocab) = (
            FIXTURE_HIDDEN,
            FIXTURE_INTER,
            FIXTURE_NQ,
            FIXTURE_NKV,
            FIXTURE_HD,
            FIXTURE_VOCAB,
        );
        let mut writer = GgufWriter::new();
        writer.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen3".to_string()),
        );
        writer.add_metadata(
            "general.name",
            MetadataWriteValue::Str("GpuCacheSlotTest".to_string()),
        );
        writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
        writer.add_metadata(
            "qwen3.block_count",
            MetadataWriteValue::U32(FIXTURE_LAYERS as u32),
        );
        writer.add_metadata(
            "qwen3.attention.head_count",
            MetadataWriteValue::U32(nq as u32),
        );
        writer.add_metadata(
            "qwen3.attention.head_count_kv",
            MetadataWriteValue::U32(nkv as u32),
        );
        writer.add_metadata(
            "qwen3.feed_forward_length",
            MetadataWriteValue::U32(inter as u32),
        );
        writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
        writer.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
        writer.add_metadata(
            "qwen3.attention.layer_norm_rms_epsilon",
            MetadataWriteValue::F32(1e-6),
        );
        writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

        writer.add_tensor(TensorEntry {
            name: "token_embd.weight".to_string(),
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(vocab * h, 0.5),
        });
        writer.add_tensor(TensorEntry {
            name: "output_norm.weight".to_string(),
            shape: vec![h as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(h, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: "output.weight".to_string(),
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_pattern(vocab * h, 0xCAFE_BABE ^ salt),
        });

        for layer in 0..FIXTURE_LAYERS {
            let pfx = format!("blk.{layer}");
            for (name, len) in [
                (format!("{pfx}.attn_norm.weight"), h),
                (format!("{pfx}.ffn_norm.weight"), h),
                (format!("{pfx}.attn_q_norm.weight"), hd),
                (format!("{pfx}.attn_k_norm.weight"), hd),
            ] {
                writer.add_tensor(TensorEntry {
                    name,
                    shape: vec![len as u64],
                    tensor_type: TensorType::F32,
                    data: f32_pattern(len, 1.0),
                });
            }
            let seed = 0x1000_0000_u64.wrapping_add((layer as u64) << 16) ^ salt;
            for (name, rows, cols, bump) in [
                (format!("{pfx}.attn_q.weight"), h, nq * hd, 0),
                (format!("{pfx}.attn_k.weight"), h, nkv * hd, 1),
                (format!("{pfx}.attn_v.weight"), h, nkv * hd, 2),
                (format!("{pfx}.attn_output.weight"), nq * hd, h, 3),
                (format!("{pfx}.ffn_gate.weight"), h, inter, 4),
                (format!("{pfx}.ffn_up.weight"), h, inter, 5),
                (format!("{pfx}.ffn_down.weight"), inter, h, 6),
            ] {
                writer.add_tensor(TensorEntry {
                    name,
                    shape: vec![rows as u64, cols as u64],
                    tensor_type: TensorType::TQ2_0_g128,
                    data: tq2_pattern(rows * cols, seed.wrapping_add(bump)),
                });
            }
        }
        writer.to_bytes().expect("GgufWriter::to_bytes")
    }

    /// Serialise every GPU-touching test in this binary.
    ///
    /// `MetalGraph` is a process-global singleton (one device, one weight
    /// cache, one KV cache) and the tail-failure seam these tests flip is
    /// process-global too, so two of them running concurrently would observe
    /// each other. A poisoned lock is recovered rather than propagated: one
    /// panicking test must not turn every sibling into a second failure.
    fn gpu_serial() -> MutexGuard<'static, ()> {
        static GPU_LOCK: Mutex<()> = Mutex::new(());
        GPU_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Look a slot up **without** being willing to upload it.
    ///
    /// `get_or_upload_keyed` only runs the closure on a miss, so a closure that
    /// always fails turns the call into a pure residency probe: `Some` means the
    /// buffer is cached under exactly this `(kind, slot)`, `None` means it is
    /// not (or is cached under a different kind).
    fn probe(graph: &MetalGraph, slot: u64, kind: WeightKind) -> Option<Arc<MetalWeightHandle>> {
        graph
            .get_or_upload_keyed(WeightKey::legacy(kind, slot), || {
                Err(MetalGraphError::ExecutionFailed(
                    "probe: not resident".into(),
                ))
            })
            .ok()
    }

    /// Every `(slot, kind)` pair a ternary model owns, layers then tail.
    fn all_cache_keys(
        layers: &[TernaryLayerSlots],
        tail: Option<TernaryTailSlots>,
    ) -> Vec<(u64, WeightKind)> {
        let mut keys: Vec<(u64, WeightKind)> = layers.iter().flat_map(|s| s.cache_keys()).collect();
        if let Some(tail) = tail {
            keys.extend(tail.cache_keys());
        }
        keys
    }

    // ── Pure tests: the slot table ───────────────────────────────────────

    /// The mapped-tensor namespace cannot reach the Q1 path's literal handles.
    ///
    /// The Q1 decode path keys its norms on `1_000_000 + layer * 10 + off`, its
    /// final norm on `2_000_000` and its LM head on `3_000_000`. Before MET-02
    /// the ternary fallback path used `2_000_000 + layer * 10` for **its**
    /// norms, so a mixed process handed the ternary layer-0 attention norm the
    /// Q1 final-norm buffer — same `WeightKind::RawF32`, so a silent stale hit
    /// rather than an error.
    #[test]
    fn mapped_tensor_slots_cannot_reach_the_q1_literal_handles() {
        let highest_q1_literal = 3_000_000u64 + 100_000 * 10 + 3;
        assert!(
            MIN_TENSOR_SLOT > highest_q1_literal,
            "the mapped-tensor floor {MIN_TENSOR_SLOT} must sit above every Q1 literal handle \
             ({highest_q1_literal})"
        );
    }

    /// A slot below the floor is refused rather than keyed on.
    #[test]
    fn tensor_slot_rejects_an_address_below_the_floor() {
        // SAFETY: `from_raw_parts` with `len == 0` requires only a non-null,
        // aligned pointer — `0x1000` is both for `u8` — and the slice is never
        // dereferenced: `tensor_slot` reads `as_ptr()` and nothing else.
        let below: &[u8] =
            unsafe { std::slice::from_raw_parts(std::ptr::without_provenance(0x1000), 0) };
        let err = tensor_slot(below, "fake_tensor").expect_err("below-floor slot must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("fake_tensor"),
            "error must name the tensor: {msg}"
        );
    }

    /// Every slot of a real ternary model is distinct, above the floor and
    /// stable across repeated derivation — which is what makes the fused path
    /// and the fallback path agree (MET-02).
    #[test]
    fn ternary_slots_are_distinct_stable_and_above_the_floor() {
        let bytes = synthetic_ternary_gguf(0);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");

        let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
        assert_eq!(layers.len(), FIXTURE_LAYERS);
        let tail = tail.expect("synthetic fixture has a ternary LM head");

        let keys = all_cache_keys(&layers, Some(tail));
        assert_eq!(keys.len(), FIXTURE_LAYERS * 8 + 2);
        for (slot, _) in &keys {
            assert!(
                *slot >= MIN_TENSOR_SLOT,
                "slot {slot:#x} is below the mapped-tensor floor"
            );
        }
        let mut sorted: Vec<u64> = keys.iter().map(|(slot, _)| *slot).collect();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            before,
            "two ternary weights resolved to the same slot"
        );

        let (layers_again, tail_again) =
            model.ternary_gpu_slots().expect("re-derive ternary slots");
        assert_eq!(layers, layers_again, "slot derivation must be stable");
        assert_eq!(
            Some(tail),
            tail_again,
            "tail slot derivation must be stable"
        );
    }

    /// Two models loaded from two GGUFs never share a slot.
    ///
    /// The old per-layer literals (`5_000_000 + layer * 10`, …) were identical
    /// for every ternary model in the process, so a draft model and a target
    /// model decoded through each other's weights.
    #[test]
    fn two_models_do_not_share_slots() {
        let bytes_a = synthetic_ternary_gguf(0);
        let bytes_b = synthetic_ternary_gguf(0xABCD);
        let gguf_a = GgufFile::parse(&bytes_a).expect("parse model A");
        let gguf_b = GgufFile::parse(&bytes_b).expect("parse model B");
        let model_a = BonsaiModel::from_gguf(&gguf_a, FIXTURE_MAX_SEQ).expect("load model A");
        let model_b = BonsaiModel::from_gguf(&gguf_b, FIXTURE_MAX_SEQ).expect("load model B");

        let (layers_a, tail_a) = model_a.ternary_gpu_slots().expect("slots A");
        let (layers_b, tail_b) = model_b.ternary_gpu_slots().expect("slots B");
        let keys_a = all_cache_keys(&layers_a, tail_a);
        let keys_b = all_cache_keys(&layers_b, tail_b);
        for key in &keys_a {
            assert!(
                !keys_b.contains(key),
                "slot {:#x} is shared between two independently loaded models",
                key.0
            );
        }
    }

    // ── GPU tests ────────────────────────────────────────────────────────

    /// The ternary GPU cache keeps no host copy of the weights (MET-03).
    ///
    /// `CachedTernaryWeights` used to hold `Vec<Vec<u8>>` copies of every
    /// projection for the life of the model — measured at 9.51 GB peak
    /// footprint for the 2.18 GB 8B. The weights now live only on the GPU (and
    /// in the mapping they were read from, which is clean, file-backed and
    /// reclaimable), and the cached value carries nothing but the LM-head row
    /// count.
    #[test]
    fn ternary_gpu_cache_retains_no_host_bytes() {
        let _gpu = gpu_serial();
        let Ok(graph) = MetalGraph::global() else {
            return; // no Metal device in this environment
        };
        let bytes = synthetic_ternary_gguf(0x11);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
        model
            .get_or_create_gpu_cache()
            .expect("build the ternary GPU weight cache");

        {
            let guard = model
                .gpu_weight_cache
                .lock()
                .expect("gpu_weight_cache lock");
            match guard.as_ref().expect("cache populated") {
                CachedModelWeights::Ternary(tern) => {
                    // Both emptiness *and* capacity: `Vec::new()` never
                    // allocates, so a zero capacity proves no host buffer was
                    // retained rather than merely cleared.
                    for (name, vecs) in [
                        ("qkv_concats", &tern.qkv_concats),
                        ("attn_proj_bytes", &tern.attn_proj_bytes),
                        ("gate_bytes", &tern.gate_bytes),
                        ("up_bytes", &tern.up_bytes),
                        ("down_bytes", &tern.down_bytes),
                    ] {
                        assert!(vecs.is_empty(), "{name} still holds host weight bytes");
                        assert_eq!(vecs.capacity(), 0, "{name} still owns a host allocation");
                    }
                    assert!(tern.lm_head_bytes.is_empty(), "lm_head_bytes retained");
                    assert_eq!(tern.lm_head_bytes.capacity(), 0, "lm_head_bytes allocated");
                    assert_eq!(tern.lm_head_out_features, FIXTURE_VOCAB);
                }
                CachedModelWeights::Q1(_) => panic!("ternary model produced a Q1 cache"),
            }
        }

        // The bytes really did go to the GPU: every slot is resident and the
        // fused QKV buffer is exactly Q‖K‖V long.
        let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
        for (slot, kind) in all_cache_keys(&layers, tail) {
            let handle = probe(&graph, slot, kind)
                .unwrap_or_else(|| panic!("slot {slot:#x} ({kind}) is not GPU-resident"));
            assert!(
                handle.byte_len() > 0,
                "slot {slot:#x} uploaded an empty buffer"
            );
            assert_eq!(
                handle.kind(),
                kind,
                "slot {slot:#x} cached under the wrong kind"
            );
        }
        let block = &model.blocks[0];
        let expected_qkv = [
            block.attn_q_blocks_ternary().expect("attn_q").len(),
            block.attn_k_blocks_ternary().expect("attn_k").len(),
            block.attn_v_blocks_ternary().expect("attn_v").len(),
        ]
        .iter()
        .sum::<usize>()
            * 34;
        let qkv = probe(&graph, layers[0].fused_qkv, WeightKind::Tq2Soa).expect("fused qkv");
        assert_eq!(
            qkv.byte_len(),
            expected_qkv,
            "the fused QKV buffer must hold all three projections"
        );

        let _ = model.release_metal_weights();
    }

    /// The empty [`FUSED_QKV_ALREADY_RESIDENT`] slice fails closed.
    ///
    /// Every ternary path binds `fused_qkv_bytes` to an empty slice because the
    /// buffer is known resident by then (the Q‖K‖V concatenation cannot be
    /// borrowed from the mapping, and keeping a host copy of it is MET-03).
    /// This pins the property that makes that safe: were the residency
    /// invariant ever broken, the kernel-side upload **errors** rather than
    /// binding a zero-length buffer to the GEMV, so the caller falls back to
    /// the CPU instead of reading garbage.
    #[test]
    fn an_empty_fused_qkv_upload_fails_closed() {
        let _gpu = gpu_serial();
        let Ok(graph) = MetalGraph::global() else {
            return;
        };
        let bytes = synthetic_ternary_gguf(0x55);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
        let (layers, _tail) = model.ternary_gpu_slots().expect("derive ternary slots");
        let slot = layers[0].fused_qkv;
        // Start from a genuinely free slot, so the upload cannot be short-cut
        // by a cache hit.
        let _ = model.release_metal_weights();
        assert!(
            probe(&graph, slot, WeightKind::Tq2Soa).is_none(),
            "slot {slot:#x} must be free for this test to mean anything"
        );

        graph
            .get_or_upload_tq2_weight_soa(slot, FUSED_QKV_ALREADY_RESIDENT)
            .expect_err("an empty TQ2 upload must fail, not allocate a zero-length buffer");
        assert!(
            probe(&graph, slot, WeightKind::Tq2Soa).is_none(),
            "a failed upload must leave nothing behind in the cache"
        );
    }

    /// Binding the ternary weights allocates a bounded, weight-size-independent
    /// amount of host memory (perf-03).
    ///
    /// Every batched prefill call used to rebuild five `Vec<Vec<u8>>` holding
    /// the entire quantized model — `5 × n_layers + 1` allocations totalling the
    /// whole model, per call, with no memoization. The binding now borrows from
    /// the mapping, so the only allocations left are the parameter vector and
    /// the slot vector.
    #[test]
    fn ternary_binding_does_not_copy_the_model_per_call() {
        let _gpu = gpu_serial();
        let Ok(_graph) = MetalGraph::global() else {
            return;
        };
        let bytes = synthetic_ternary_gguf(0x22);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
        model.get_or_create_gpu_cache().expect("warm the GPU cache");

        // Warm any lazily-initialised global the first binding would touch, so
        // the measured call sees only its own allocations.
        let warm = model.ternary_gpu_binding().expect("warm-up binding");
        drop(warm);

        let (result, allocations) = count_allocations(|| model.ternary_gpu_binding());
        let binding = result.expect("binding on a warm cache");
        assert_eq!(binding.layer_params.len(), FIXTURE_LAYERS);
        assert!(
            allocations <= 8,
            "binding the ternary weights made {allocations} allocations; it must be a small \
             constant (the parameter vector and the slot vector), not a copy of the model"
        );

        let _ = model.release_metal_weights();
    }

    /// A tail failure followed by the fallback path must not upload a second
    /// copy of the model (MET-02).
    ///
    /// `try_metal_full_forward_with_lm_head_ternary` uploads every layer's
    /// weights and only then runs the final-norm → LM-head tail, so a
    /// deterministic tail failure leaves the model fully resident and sends
    /// `BonsaiModel::forward()` into `try_metal_full_forward_ternary_inner`.
    /// That path used to own a different handle namespace, so the fallback
    /// doubled GPU residency on the very first token.
    ///
    /// **Assertion (b), the `cached_weight_count()` delta, is the load-bearing
    /// one.** A second namespace uploads its duplicate at a *different* slot
    /// and leaves the canonical entries untouched, so (a)'s per-slot
    /// `Arc::ptr_eq` check passes straight through it — verified by
    /// reintroducing a shifted `attn_proj`/`down` namespace, which (a) missed
    /// and (b) caught as `left: 22, right: 18`. The price is that (b) is a
    /// process-global count: it would also trip if some future test elsewhere
    /// in this lib-test binary uploaded a Metal weight concurrently (none does
    /// today — `gpu_serial()` covers every GPU test here). Assertions (a) and
    /// (c) are immune to that, so a red (b) with green (a)/(c) means look for a
    /// concurrent GPU test before suspecting the slot table.
    #[test]
    fn forced_tail_failure_does_not_duplicate_gpu_weights() {
        let _gpu = gpu_serial();
        let Ok(graph) = MetalGraph::global() else {
            return;
        };
        let bytes = synthetic_ternary_gguf(0x33);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");

        let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
        let keys = all_cache_keys(&layers, tail);

        // ── Fused path, with the tail forced to fail after the uploads ──
        let mut hidden = vec![0.05_f32; FIXTURE_HIDDEN];
        let mut logits = Vec::new();
        crate::model::types::forward_metal::set_force_ternary_tail_failure(true);
        let fused = model.try_metal_full_forward_with_lm_head_ternary(&mut hidden, 0, &mut logits);
        crate::model::types::forward_metal::set_force_ternary_tail_failure(false);
        let err = fused.expect_err("the tail failure seam must make the fused path fail");
        assert!(
            err.to_string().contains("OXIBONSAI_FORCE_METAL_TAIL_FAIL"),
            "the failure must come from the seam, not from something else: {err}"
        );

        let before: Vec<Arc<MetalWeightHandle>> = keys
            .iter()
            .map(|(slot, kind)| {
                probe(&graph, *slot, *kind)
                    .unwrap_or_else(|| panic!("slot {slot:#x} not resident after the fused path"))
            })
            .collect();
        let cached_before = graph.cached_weight_count().expect("cached weight count");

        // ── The fallback `forward()` takes on that failure ──────────────
        model
            .try_metal_full_forward_ternary_inner(&mut hidden, 0)
            .expect("the ternary fallback path must run on the already-resident weights");

        // (a) The canonical slots still hold the very same buffers.
        for (i, (slot, kind)) in keys.iter().enumerate() {
            let after = probe(&graph, *slot, *kind)
                .unwrap_or_else(|| panic!("slot {slot:#x} disappeared during the fallback"));
            assert!(
                Arc::ptr_eq(&before[i], &after),
                "slot {slot:#x} ({kind}) was re-uploaded by the fallback path: the two ternary \
                 handle namespaces are back"
            );
        }

        // (b) …and no buffer appeared *anywhere else* either, which is what the
        // old 2M/3M namespace did: it left the 5M/6M entries untouched and
        // uploaded a whole second copy beside them.
        let cached_after = graph.cached_weight_count().expect("cached weight count");
        assert_eq!(
            cached_after,
            cached_before,
            "the ternary fallback path added {} GPU weight buffers; every weight it binds must \
             already be resident under this model's slots",
            cached_after.saturating_sub(cached_before)
        );

        // (c) Structurally: every handle the fallback binds is one of this
        // model's canonical slots, so there is no second namespace to drift
        // into in the first place.
        let canonical: Vec<u64> = keys.iter().map(|(slot, _)| *slot).collect();
        let binding = model.ternary_gpu_binding().expect("rebuild the binding");
        for (i, lp) in binding.layer_params.iter().enumerate() {
            for (name, handle) in [
                ("attn_norm", lp.attn_norm_handle),
                ("q_norm", lp.q_norm_handle),
                ("k_norm", lp.k_norm_handle),
                ("ffn_norm", lp.ffn_norm_handle),
                ("fused_qkv", lp.fused_qkv_handle),
                ("attn_proj", lp.attn_proj_handle),
                ("gate_up", lp.gate_up_handle),
                ("down", lp.down_handle),
            ] {
                assert!(
                    canonical.contains(&handle),
                    "layer {i} binds {name} to {handle:#x}, which is not one of this model's slots"
                );
            }
        }

        drop(before);
        let _ = model.release_metal_weights();
    }

    /// Unloading a model releases its GPU buffers.
    ///
    /// Nothing evicts `MetalGraph`'s weight cache on its own, so without this a
    /// model that is unloaded keeps its whole quantized self on the GPU — 7.2 GB
    /// per load for the 27B `PQ2_0`.
    #[test]
    fn release_metal_weights_drops_this_models_buffers() {
        let _gpu = gpu_serial();
        let Ok(graph) = MetalGraph::global() else {
            return;
        };
        let bytes = synthetic_ternary_gguf(0x44);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
        let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load synthetic model");
        model
            .get_or_create_gpu_cache()
            .expect("build the GPU cache");

        let (layers, tail) = model.ternary_gpu_slots().expect("derive ternary slots");
        let keys = all_cache_keys(&layers, tail);
        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_some(),
                "slot {slot:#x} should be resident before release"
            );
        }

        let resident_before = graph.resident_weight_bytes().expect("resident bytes");
        let released = model.release_metal_weights().expect("release this model");
        assert_eq!(released, keys.len(), "every owned slot must be released");

        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_none(),
                "slot {slot:#x} is still cached after release"
            );
        }
        let resident_after = graph.resident_weight_bytes().expect("resident bytes");
        assert!(
            resident_after < resident_before,
            "releasing {} buffers must lower the resident-byte gauge ({resident_before} → \
             {resident_after})",
            keys.len()
        );

        // The model rebuilds its cache on demand afterwards.
        model
            .get_or_create_gpu_cache()
            .expect("the cache must rebuild after a release");
        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_some(),
                "slot {slot:#x} should be resident again after a rebuild"
            );
        }
        let _ = model.release_metal_weights();
    }

    /// Dropping the *last* (here, only) replica sharing a model's ternary GPU
    /// buffers releases them — spec 1(c) / ACCEPTANCE "`release_model` drops
    /// the handles on Drop" — with no explicit `release_metal_weights()` call
    /// anywhere in this test.
    #[test]
    fn drop_of_the_last_ternary_replica_releases_the_shared_gpu_buffers() {
        let _gpu = gpu_serial();
        let Ok(graph) = MetalGraph::global() else {
            return;
        };
        let bytes = synthetic_ternary_gguf(0x66);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");
        let keys = {
            let model = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load model");
            model.get_or_create_gpu_cache().expect("warm the GPU cache");
            let (layers, tail) = model.ternary_gpu_slots().expect("derive slots");
            let keys = all_cache_keys(&layers, tail);
            for (slot, kind) in &keys {
                assert!(
                    probe(&graph, *slot, *kind).is_some(),
                    "slot {slot:#x} should be resident before drop"
                );
            }
            keys
            // `model` is dropped here, at the end of this block — nothing
            // else calls `release_metal_weights()` in this test.
        };
        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_none(),
                "slot {slot:#x} is still cached after its only model dropped"
            );
        }
    }

    /// Dropping one of several replicas that share a model's ternary GPU
    /// buffers must NOT evict them out from under the replicas still alive —
    /// the failure mode a bare, unconditional `Drop -> release_metal_weights`
    /// would reintroduce (a 27B pool re-uploading 7.2 GB mid-serve). Once the
    /// last replica drops, the buffers must still be released.
    #[test]
    fn drop_of_a_non_last_ternary_replica_does_not_evict_the_shared_buffers() {
        let _gpu = gpu_serial();
        let Ok(graph) = MetalGraph::global() else {
            return;
        };
        let bytes = synthetic_ternary_gguf(0x88);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic ternary GGUF");

        let replica_a = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load replica A");
        let replica_b = BonsaiModel::from_gguf(&gguf, FIXTURE_MAX_SEQ).expect("load replica B");
        replica_a.get_or_create_gpu_cache().expect("warm A");
        replica_b.get_or_create_gpu_cache().expect("warm B");

        // Non-vacuous: both replicas must resolve to the very same slots
        // (MET-02's "engine-pool replicas share one upload"), or dropping one
        // trivially wouldn't touch the other's and the rest of this test
        // would pass for the wrong reason.
        let (layers_a, tail_a) = replica_a.ternary_gpu_slots().expect("slots A");
        let (layers_b, tail_b) = replica_b.ternary_gpu_slots().expect("slots B");
        assert_eq!(
            layers_a, layers_b,
            "two replicas of one GgufFile must resolve to the same layer slots"
        );
        assert_eq!(
            tail_a, tail_b,
            "two replicas of one GgufFile must resolve to the same tail slots"
        );
        let keys = all_cache_keys(&layers_a, tail_a);

        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_some(),
                "slot {slot:#x} should be resident once both replicas are warm"
            );
        }

        drop(replica_a);

        // Replica B is still alive and shares these exact slots: none of them
        // may have been evicted by A's drop.
        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_some(),
                "slot {slot:#x} was evicted while a sibling replica was still alive"
            );
        }

        drop(replica_b);

        // Now the last replica is gone: the shared buffers must be released.
        for (slot, kind) in &keys {
            assert!(
                probe(&graph, *slot, *kind).is_none(),
                "slot {slot:#x} is still cached after the last replica dropped"
            );
        }
    }
}
