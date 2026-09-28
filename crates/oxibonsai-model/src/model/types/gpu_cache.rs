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
//! because they bind the very same buffers.
//!
//! # Weight-cache epoch (MET-02)
//!
//! The kernels key every ternary lookup on
//! `FullForwardLayerParamsTernary::model_epoch`. This module sets it to
//! [`TERNARY_GPU_EPOCH`], which is deliberately the legacy epoch — see that
//! constant for why a per-load epoch cannot be switched on from here yet.
//!
//! # Release on unload, without a struct field on `BonsaiModel` (spec 1(b)/1(c))
//!
//! The wave-1 `METAL-CACHE` package built a composite `WeightKey { model_epoch,
//! kind, slot }` precisely so a model's whole GPU footprint can be dropped by
//! epoch in one `MetalGraph::release_model(epoch)` call. The full design
//! threads a `metal_model_epoch: u64` field through `BonsaiModel`, allocated
//! once per load by `MetalGraph::next_model_epoch()`. `BonsaiModel` is defined
//! in `mod.rs`, which this module's owner does not own, so that field cannot
//! land from here — see the package deviations.
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
use oxibonsai_kernels::gpu_backend::metal_full_layer::types::{
    WeightKey, WeightKind, LEGACY_MODEL_EPOCH,
};
use oxibonsai_kernels::{CachedModelWeights, FullForwardLayerParamsTernary, MetalGraph};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Convenience alias for the boxed error every Metal entry point here returns.
type GpuResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Weight-cache epoch the ternary slot table is keyed under (`MET-02`).
///
/// The kernels key every ternary lookup — eight per layer plus the tail — on
/// `FullForwardLayerParamsTernary::model_epoch`, and this module fills that
/// field (and keys its own uploads, probes and evictions) with this value.
/// It is **the legacy epoch on purpose**, for two reasons:
///
/// 1. It loses nothing today. Ternary slots are derived from the addresses of
///    the mapped tensors ([`tensor_slot`]), which already makes them unique
///    per mapping and shared by every replica of one `GgufFile` — the two
///    properties a per-load epoch exists to provide — and the refcounted
///    `Drop` below releases a model's slots before its mapping can be reused.
/// 2. A different value would break the flagship prefill. Two of the five
///    ternary paths, the batched prefill and its verify twin
///    (`try_metal_full_forward_prefill_ternary` /
///    `…_prefill_verify_ternary`, `metal_prefill/functions_2.rs`), still look
///    their weights up under the legacy key regardless of `model_epoch`. With
///    any other epoch here every prompt prefill would miss its resident
///    buffers, try to upload the deliberately empty
///    [`FUSED_QKV_ALREADY_RESIDENT`], fail closed and fall back to the CPU.
///
/// Moving to a per-load epoch therefore waits on one of: those two lookups
/// keyed on `lp.model_epoch`, or `forward_metal.rs` routing its batched
/// prefill through the cached entry points (which perform no lookup at all).
/// Both are recorded as deviations of this package.
pub(super) const TERNARY_GPU_EPOCH: u64 = LEGACY_MODEL_EPOCH;

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
///
/// `pub` so the binding is usable outside the crate (the decode-throughput
/// A/B in `BENCHES` drives the uncached path with it); the type is reachable
/// by value and field today, and nameable once `model/types/mod.rs`
/// re-exports it (a deviation of the package that made it `pub`).
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
        .get_or_upload_tq2_weight_soa_lazy_for_epoch(TERNARY_GPU_EPOCH, slot, || {
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
    let epoch = TERNARY_GPU_EPOCH;
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
        model_epoch: TERNARY_GPU_EPOCH,
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
                graph.evict_weight(WeightKey::new(TERNARY_GPU_EPOCH, kind, slot))?;
                released += 1;
            }
        }
        if let Some(tail) = tail_slots {
            for (slot, kind) in tail.cache_keys() {
                graph.evict_weight(WeightKey::new(TERNARY_GPU_EPOCH, kind, slot))?;
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

    /// Upload the ternary (TQ2_0_g128) weights and cache their GPU handles.
    ///
    /// Every projection goes to the GPU here, once, keyed by the address of the
    /// mapped tensor it came from, through one reused staging buffer (see
    /// [`upload_ternary_layer`]). The kernels' `build_cached_weights_ternary_only`
    /// then resolves those now-resident buffers — pure cache hits — into the
    /// MET-03 shape: eight `Arc<MetalWeightHandle>` per layer plus the tail,
    /// and **no host bytes**. That value is both the "weights are up" marker
    /// that makes this function idempotent and what the cached decode entry
    /// points bind directly.
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
        let lm_head_bytes = blocks_as_bytes_ternary(lm_head_ternary.blocks());
        let ternary_lm_head_out_features = lm_head_ternary.out_features();

        // Every weight is resident now, so the builder's lookups are all hits:
        // `fused_qkv_bytes` is the empty `FUSED_QKV_ALREADY_RESIDENT` and the
        // gate‖up closure never runs. Its parameter structs borrow from the
        // mapping and are dropped at the end of this function.
        let mut layer_params = Vec::with_capacity(n_layers);
        for (block, slots) in self.blocks.iter().zip(layer_slots.iter()) {
            layer_params.push(ternary_layer_params(block, slots)?);
        }
        let cached = oxibonsai_kernels::build_cached_weights_ternary_only(
            &layer_params,
            Some((tail.final_norm, self.output_norm.weight())),
            Some((tail.lm_head, lm_head_bytes)),
            ternary_lm_head_out_features,
        )
        .map_err(|e| format!("build_cached_weights_ternary_only: {e}"))?;
        drop(layer_params);

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

    /// Shared guard + embedding prologue of the single-token cached decode
    /// entry points: the context-length and sliding-window refusals of the
    /// uncached twins, then this token's hidden state and RoPE rows.
    fn ternary_decode_prologue(
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
    /// Bit-identical to the uncached `forward_greedy_gpu` on a ternary model
    /// (same buffers, same kernels). Maintains the device KV cache of the
    /// session this thread dispatches in, exactly like that path.
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

    /// Logits of one token through the **uncached** ternary GPU path, strictly
    /// (no CPU fallback): the per-call `FullForwardLayerParamsTernary` binding
    /// and eight weight-cache lookups per layer that `forward()` makes first on
    /// a ternary model.
    ///
    /// It exists as the reference the cached shape is proven against — the
    /// MET-03 parity evidence compares this path and
    /// [`Self::forward_logits_gpu_ternary_cached`] bit for bit on the real
    /// model — and as the explicit "no fallback" entry a caller can use to
    /// tell a GPU failure from a CPU answer.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_greedy_gpu_ternary_cached`].
    pub fn forward_logits_gpu_ternary_uncached(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (mut hidden, _, _) = self.ternary_decode_prologue(token_id, pos)?;
        let mut logits = Vec::new();
        self.try_metal_full_forward_with_lm_head_ternary(&mut hidden, pos, &mut logits)?;
        self.note_device_kv_used();
        Ok(logits)
    }

    /// Embed `token_ids` column-major and build the per-position RoPE tables
    /// for a batched prefill starting at `pos_start`.
    #[allow(clippy::type_complexity)]
    fn ternary_prefill_inputs(
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
    /// the cached twin of `try_metal_prefill_with_lm_head_ternary`, returning
    /// the last position's logits.
    ///
    /// It performs no weight-cache lookup at all, which is what takes the
    /// ternary prefill off the legacy-keyed lookups of the uncached
    /// `metal_prefill` entry (see [`TERNARY_GPU_EPOCH`]).
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
    /// `try_metal_prefill_verify_ternary_path`.
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
#[path = "gpu_cache_tests.rs"]
mod tests;
