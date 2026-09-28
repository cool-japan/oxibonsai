//! One GPU weight-cache namespace per GGUF **mapping** (`MET-02`, Q1 half and
//! ternary epoch).
//!
//! # The problem this solves
//!
//! Every fused GPU path hands the kernels `u64` slot ids for the weights that
//! are not keyed by an upload handle: the RMSNorm weights, the final norm and
//! the LM head (Q1), and — for the ternary route — every weight, keyed by the
//! address of its mapped tensor. Those slots used to be process-wide literals
//! (`1_000_000 + layer * 10`, `2_000_000`, `3_000_000`), so two *different*
//! models in one process were served each other's norms and LM head. The
//! first fix gave every `BonsaiModel` its **own** epoch, which cured the
//! collision but broke the property engine-pool replicas depend on: N
//! replicas of one model must hold its weights **once**. Measured on the real
//! `Bonsai-8B` (Q1): every replica re-uploaded its norms and its 88 MB LM
//! head, and within one replica the prefill and the greedy decode keyed the
//! same LM head on two different slot sets (+84.5 MiB).
//!
//! # The shape
//!
//! * [`SlotNamespace`] — the pure slot composition `TAG | epoch << 24 |
//!   local`, with the historical local layout (norms at
//!   `1_000_000 + layer * 10 + k`, final norm `2_000_000`, LM head
//!   `3_000_000`) and a CUDA-only weight-fallback range. Bit 63 is set on
//!   every composed slot, so no composed slot can ever equal a user-space
//!   address (the ternary and image caches key on those) or an upload-handle
//!   counter (the Q1 block weights key on those).
//! * [`MappingState`] — one per mapping: the anchor address, the epoch, and a
//!   "touched the GPU" flag. Shared (`Arc`) by every replica of the mapping
//!   and by every block of those replicas. When the last holder drops it, and
//!   only if some holder marked it used, its release hook runs once for the
//!   whole mapping (on Metal: `MetalGraph::release_model(epoch)`, which drops
//!   every buffer keyed under the epoch — Q1 norms and LM head as well as the
//!   ternary weights).
//! * [`MappingRegistry`] — the process-wide table `anchor → (state, replica
//!   count)`. [`MappingRegistration::join`] (one per `BonsaiModel`, taken at
//!   construction) increments the count and hands back the shared state;
//!   dropping the registration decrements it, and the entry disappears with
//!   the last replica — so a later mapping that happens to land on the same
//!   address mints a **new** epoch instead of inheriting buffers the old one
//!   left behind.
//!
//! # Why register at construction (not at first fused use)
//!
//! A model's slots are then fixed for its whole life: every path — the fused
//! decode, the batched prefill, the cached greedy builder, the per-layer block
//! path — derives the same slots no matter which one runs first, and a cached
//! weight set built under one epoch can never be looked up under another.
//! Registering lazily would make the epoch depend on which replica happened to
//! dispatch first and would need the registry lock on the decode hot path.
//! Registration itself touches no GPU state (it is a hash-map entry), so a
//! CPU-only model pays nothing measurable.
//!
//! # What may anchor a mapping
//!
//! Only data **borrowed from the mapped GGUF**: the first block's `Q1_0_g128`
//! or `TQ2_0_g128` `attn_q` blocks, falling back to the LM head's. The
//! borrow's lifetime guarantees that the mapping outlives every model built
//! over it, hence that an anchor address cannot be recycled while its registry
//! entry is live. Heap-owned weights carry no such guarantee (a freed buffer's
//! address can be reused while the entry still exists), so a model with no
//! mapped `Q1`/`TQ2` weight gets a **private** namespace — a fresh epoch that
//! no other model can ever join. No other format keys GPU buffers on this
//! namespace, so nothing is lost by not sharing it.
//!
//! This module holds no GPU code at all (the release hook is a plain function
//! pointer supplied by the Metal build), so its tests run on every host.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::OutputWeight;
use crate::block::TransformerBlock;

/// Number of K-quant formats the CUDA slot layout below reserves a
/// norm/weight/tail range for (Q2K, Q3K, Q4K, Q5K, Q6K, Q8K).
///
/// The methods below take a plain `format_idx: u64` (`0..K_QUANT_FORMAT_COUNT`)
/// rather than `oxibonsai_kernels::KQuantFormat` directly: that type is
/// exported only on the `native-cuda` + linux/windows CUDA build
/// (`oxibonsai-kernels/src/lib.rs`'s `pub use gpu_backend::{...,
/// KQuantFormat}` is itself `target_os`-gated), while this module's slot
/// arithmetic — like the rest of it — is plain data with no device
/// dependency and stays compiled (and its tests run) on every host,
/// including this one. `forward_cuda/k_quant.rs`, which *is*
/// `native-cuda` + linux/windows-gated and so has the real
/// `KQuantFormat` in scope, converts it to `0..6` right before calling in.
pub const K_QUANT_FORMAT_COUNT: u64 = 6;

// ═══════════════════════════════════════════════════════════════════════════
// Slot composition
// ═══════════════════════════════════════════════════════════════════════════

/// Tag bit of every composed slot.
///
/// User-space addresses (which the ternary and image caches key on) never set
/// bit 63, and upload-handle ids are small monotonic counters, so a tagged
/// slot collides with neither namespace.
pub const SLOT_TAG: u64 = 1 << 63;

/// Bits below the epoch: the per-model local slot (`< 2^24`).
pub const SLOT_LOCAL_BITS: u32 = 24;

/// Mask keeping the epoch clear of the tag bit (`2^39` epochs).
pub const SLOT_EPOCH_MASK: u64 = (1 << (63 - SLOT_LOCAL_BITS)) - 1;

/// Local slot of layer 0's first RMSNorm (`+0` attn, `+1` q, `+2` k,
/// `+3` ffn).
const NORM_LOCAL_BASE: u64 = 1_000_000;
/// Distance between two layers' norm groups.
const NORM_LOCAL_STRIDE: u64 = 10;
/// Local slot of the final `output_norm`.
const FINAL_NORM_LOCAL: u64 = 2_000_000;
/// Local slot of the LM head.
const LM_HEAD_LOCAL: u64 = 3_000_000;
/// Local slot of layer 0's first CUDA weight fallback (`+0` fused QKV,
/// `+1` attention output, `+2` gate‖up, `+3` down) — used only when a CUDA
/// block has no upload handle for that weight.
const WEIGHT_FALLBACK_LOCAL_BASE: u64 = 4_000_000;
/// Distance between two layers' weight-fallback groups.
const WEIGHT_FALLBACK_LOCAL_STRIDE: u64 = 4;
/// Local slot of layer 0's first CUDA **ternary** RMSNorm (`+0` attn, `+1`
/// q, `+2` k, `+3` ffn) — see [`SlotNamespace::cuda_ternary_keys`].
const TERNARY_NORM_LOCAL_BASE: u64 = 5_000_000;
/// Local slot of layer 0's first CUDA ternary weight (`+0` fused QKV, `+1`
/// attention output, `+2` gate‖up, `+3` down).
const TERNARY_WEIGHT_LOCAL_BASE: u64 = 6_000_000;
/// Distance between two layers' CUDA ternary norm / weight groups.
const TERNARY_LOCAL_STRIDE: u64 = 4;
/// Local slot of the CUDA ternary final norm.
const TERNARY_FINAL_NORM_LOCAL: u64 = 7_000_000;
/// Local slot of the CUDA ternary LM head.
const TERNARY_LM_HEAD_LOCAL: u64 = 7_000_001;

// ── CUDA Q4_0 / Q8_0 ("q_std") and K-quant slot composition ────────────────
//
// `forward_cuda/q_std.rs` and `forward_cuda/k_quant.rs` used to key every
// norm / weight / final-norm / LM-head slot on process-wide literals
// (`8_000_000 + layer * 10`, `24_000_000 + format_offset`, ...), exactly the
// M-16-class bug the ternary and Q1 ranges above were already fixed for: two
// *different* models of the same quant family in one process would serve
// each other's norms and LM head. Composed over `cuda_model_epoch` via
// [`SlotNamespace`] below, following the ternary layout's shape (stride 4,
// not the old literals' stride 10 — a `layer * 10` stride across
// [`MAX_SLOT_LAYERS`] layers needs 16 ranges' worth of headroom this
// namespace does not have; stride 4 costs only 4 sequential offsets per
// layer, which every per-layer group here uses in full).
//
// Layout, in local-slot order (each per-layer range spans
// `BASE..=BASE + (MAX_SLOT_LAYERS-1)*4 + 3`, i.e. `BASE..BASE+400_000`):
//
// | range | base | cache |
// |---|---|---|
// | Q4_0 norms | `Q4_0_NORM_LOCAL_BASE` (8_000_000) | `f32` |
// | Q4_0 weights | `Q4_0_WEIGHT_LOCAL_BASE` (8_400_000) | `u8` |
// | Q8_0 norms | `Q8_0_NORM_LOCAL_BASE` (8_800_000) | `f32` |
// | Q8_0 weights | `Q8_0_WEIGHT_LOCAL_BASE` (9_200_000) | `u8` |
// | Q4_0/Q8_0 final norms + LM heads | `Q_STD_TAIL_LOCAL_BASE` (9_600_000) | mixed |
// | Q2K..Q8K norms/weights | `K_QUANT_NORM_LOCAL_BASE` (10_000_000).. | mixed |
// | Q2K..Q8K final norms + LM heads | `K_QUANT_TAIL_LOCAL_BASE` (14_800_000) | mixed |

/// Local slot of layer 0's first CUDA Q4_0 norm (`+0` attn, `+1` q, `+2` k,
/// `+3` ffn; `f32` cache).
const Q4_0_NORM_LOCAL_BASE: u64 = 8_000_000;
/// Local slot of layer 0's first CUDA Q4_0 weight (`+0` fused QKV, `+1`
/// attention output, `+2` gate‖up, `+3` down; `u8` cache).
const Q4_0_WEIGHT_LOCAL_BASE: u64 = 8_400_000;
/// Local slot of layer 0's first CUDA Q8_0 norm.
const Q8_0_NORM_LOCAL_BASE: u64 = 8_800_000;
/// Local slot of layer 0's first CUDA Q8_0 weight.
const Q8_0_WEIGHT_LOCAL_BASE: u64 = 9_200_000;
/// Distance between two layers' CUDA Q4_0/Q8_0 norm / weight groups.
const Q_STD_LOCAL_STRIDE: u64 = 4;
/// Base of the four CUDA Q4_0/Q8_0 tail slots: `+0` Q4_0 final norm (`f32`),
/// `+1` Q4_0 LM head (`u8`), `+2` Q8_0 final norm (`f32`), `+3` Q8_0 LM head
/// (`u8`).
const Q_STD_TAIL_LOCAL_BASE: u64 = 9_600_000;

/// Local slot of layer 0's first CUDA K-quant norm, per format (`+0` Q2K,
/// `+1` Q3K, `+2` Q4K, `+3` Q5K, `+4` Q6K, `+5` Q8K, each multiplied by
/// [`K_QUANT_FORMAT_LOCAL_STRIDE`]; `f32` cache).
const K_QUANT_NORM_LOCAL_BASE: u64 = 10_000_000;
/// Local slot of layer 0's first CUDA K-quant weight, per format (same
/// per-format indexing as [`K_QUANT_NORM_LOCAL_BASE`]; `u8` cache).
const K_QUANT_WEIGHT_LOCAL_BASE: u64 = 10_400_000;
/// Distance between two layers' CUDA K-quant norm / weight groups, for one
/// format.
const K_QUANT_LOCAL_STRIDE: u64 = 4;
/// Distance between two formats' CUDA K-quant norm/weight range pairs
/// (covers one format's norm range, its weight range, and headroom for both
/// to reach [`MAX_SLOT_LAYERS`] without touching the next format's).
const K_QUANT_FORMAT_LOCAL_STRIDE: u64 = 800_000;
/// Base of the CUDA K-quant tail slots: format `f`'s final norm (`f32`) is
/// `K_QUANT_TAIL_LOCAL_BASE + f`, its LM head (`u8`) is
/// `K_QUANT_TAIL_LOCAL_BASE + K_QUANT_LM_HEAD_TAIL_OFFSET + f`, `f` in
/// `0..`[`K_QUANT_FORMAT_COUNT`] (`KQuantFormat`'s declaration order: Q2K,
/// Q3K, Q4K, Q5K, Q6K, Q8K).
const K_QUANT_TAIL_LOCAL_BASE: u64 = 14_800_000;
/// Offset from [`K_QUANT_TAIL_LOCAL_BASE`] to the first LM-head tail slot
/// (past the 6 final-norm slots, with a gap so the two groups are never
/// adjacent by accident).
const K_QUANT_LM_HEAD_TAIL_OFFSET: u64 = 10;

/// Largest layer count whose norm slots stay below [`FINAL_NORM_LOCAL`]
/// (`1_000_000 + 99_999 * 10 + 3 < 2_000_000`) and whose weight-fallback and
/// CUDA ternary groups stay below the next range (`4_399_999`, `5_399_999`,
/// `6_399_999`), all well inside `2^24`. Every fused path checks it through
/// [`SlotNamespace::check_layer_count`] before deriving a slot.
pub const MAX_SLOT_LAYERS: usize = 100_000;

// The local ranges, in order, never overlap and never reach the epoch bits
// for any layer count up to `MAX_SLOT_LAYERS` — checked at compile time, so a
// change to a base or a stride that breaks the layout does not build.
const _: () = {
    let last = (MAX_SLOT_LAYERS - 1) as u64;
    assert!(NORM_LOCAL_BASE + last * NORM_LOCAL_STRIDE + 3 < FINAL_NORM_LOCAL);
    assert!(FINAL_NORM_LOCAL < LM_HEAD_LOCAL);
    assert!(LM_HEAD_LOCAL < WEIGHT_FALLBACK_LOCAL_BASE);
    assert!(
        WEIGHT_FALLBACK_LOCAL_BASE + last * WEIGHT_FALLBACK_LOCAL_STRIDE + 3
            < TERNARY_NORM_LOCAL_BASE
    );
    assert!(TERNARY_NORM_LOCAL_BASE + last * TERNARY_LOCAL_STRIDE + 3 < TERNARY_WEIGHT_LOCAL_BASE);
    assert!(TERNARY_WEIGHT_LOCAL_BASE + last * TERNARY_LOCAL_STRIDE + 3 < TERNARY_FINAL_NORM_LOCAL);
    assert!(TERNARY_FINAL_NORM_LOCAL < TERNARY_LM_HEAD_LOCAL);
    assert!(TERNARY_LM_HEAD_LOCAL < Q4_0_NORM_LOCAL_BASE);
    // Q4_0 / Q8_0.
    assert!(Q4_0_NORM_LOCAL_BASE + last * Q_STD_LOCAL_STRIDE + 3 < Q4_0_WEIGHT_LOCAL_BASE);
    assert!(Q4_0_WEIGHT_LOCAL_BASE + last * Q_STD_LOCAL_STRIDE + 3 < Q8_0_NORM_LOCAL_BASE);
    assert!(Q8_0_NORM_LOCAL_BASE + last * Q_STD_LOCAL_STRIDE + 3 < Q8_0_WEIGHT_LOCAL_BASE);
    assert!(Q8_0_WEIGHT_LOCAL_BASE + last * Q_STD_LOCAL_STRIDE + 3 < Q_STD_TAIL_LOCAL_BASE);
    assert!(Q_STD_TAIL_LOCAL_BASE + 3 < K_QUANT_NORM_LOCAL_BASE);
    // K-quant: format 5 (Q8K, the last of the 6) must still fit below the
    // next format's base, and format 0's (Q2K) full per-layer range must fit
    // below its own weight range.
    assert!(K_QUANT_NORM_LOCAL_BASE + last * K_QUANT_LOCAL_STRIDE + 3 < K_QUANT_WEIGHT_LOCAL_BASE);
    assert!(
        K_QUANT_WEIGHT_LOCAL_BASE + last * K_QUANT_LOCAL_STRIDE + 3
            < K_QUANT_NORM_LOCAL_BASE + K_QUANT_FORMAT_LOCAL_STRIDE
    );
    assert!(
        K_QUANT_NORM_LOCAL_BASE + 5 * K_QUANT_FORMAT_LOCAL_STRIDE + last * K_QUANT_LOCAL_STRIDE + 3
            < K_QUANT_WEIGHT_LOCAL_BASE + 5 * K_QUANT_FORMAT_LOCAL_STRIDE
    );
    assert!(
        K_QUANT_WEIGHT_LOCAL_BASE
            + 5 * K_QUANT_FORMAT_LOCAL_STRIDE
            + last * K_QUANT_LOCAL_STRIDE
            + 3
            < K_QUANT_TAIL_LOCAL_BASE
    );
    assert!(K_QUANT_TAIL_LOCAL_BASE + 5 < K_QUANT_TAIL_LOCAL_BASE + K_QUANT_LM_HEAD_TAIL_OFFSET);
    assert!(K_QUANT_TAIL_LOCAL_BASE + K_QUANT_LM_HEAD_TAIL_OFFSET + 5 < (1 << SLOT_LOCAL_BITS));
};

/// Lowest value a mapped-tensor slot is allowed to take.
///
/// macOS maps `__PAGEZERO` over the first 4 GiB of every process on both
/// arm64 and x86_64, so no valid pointer — mmap'd GGUF data included — lies
/// below `1 << 32`. [`mapped_tensor_slot`] enforces the floor instead of
/// assuming it, so a platform breaking the assumption produces a named error
/// rather than a slot that a small literal could reach.
pub const MIN_TENSOR_SLOT: u64 = 1 << 32;

/// Which GPU cache a composed slot is looked up in.
///
/// The Metal build keeps one kind-tagged cache (`RawF32` for norms, `Q1Soa`
/// for the LM head); the CUDA build keeps separate `f32` and `u8` maps. The
/// local ranges of the two kinds never overlap, so a slot is unique whichever
/// cache it lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotCache {
    /// An `f32` RMSNorm weight.
    Norm,
    /// A quantized weight (the LM head, or a CUDA weight fallback).
    Quant,
}

/// The slot namespace of one epoch: `TAG | (epoch & mask) << 24 | local`.
///
/// Pure arithmetic — two values built from the same epoch compose the same
/// slots, and two different epochs never compose a common one (for fewer than
/// `2^39` epochs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotNamespace {
    epoch: u64,
}

impl SlotNamespace {
    /// The namespace of `epoch`.
    #[must_use]
    pub const fn new(epoch: u64) -> Self {
        Self { epoch }
    }

    /// The epoch this namespace composes.
    #[must_use]
    pub const fn epoch(self) -> u64 {
        self.epoch
    }

    /// Compose `local` into this namespace.
    #[must_use]
    pub const fn slot(self, local: u64) -> u64 {
        SLOT_TAG | ((self.epoch & SLOT_EPOCH_MASK) << SLOT_LOCAL_BITS) | local
    }

    /// Base of layer `layer`'s four norm slots (`+0` attn, `+1` q, `+2` k,
    /// `+3` ffn). `layer` must be below [`MAX_SLOT_LAYERS`].
    #[must_use]
    pub const fn norm_base(self, layer: usize) -> u64 {
        self.slot(NORM_LOCAL_BASE + (layer as u64) * NORM_LOCAL_STRIDE)
    }

    /// Slot of the final `output_norm`.
    #[must_use]
    pub const fn final_norm(self) -> u64 {
        self.slot(FINAL_NORM_LOCAL)
    }

    /// Slot of the LM head.
    #[must_use]
    pub const fn lm_head(self) -> u64 {
        self.slot(LM_HEAD_LOCAL)
    }

    /// Base of layer `layer`'s four CUDA weight-fallback slots (`+0` fused
    /// QKV, `+1` attention output, `+2` gate‖up, `+3` down). `layer` must be
    /// below [`MAX_SLOT_LAYERS`].
    #[must_use]
    pub const fn weight_fallback_base(self, layer: usize) -> u64 {
        self.slot(WEIGHT_FALLBACK_LOCAL_BASE + (layer as u64) * WEIGHT_FALLBACK_LOCAL_STRIDE)
    }

    /// Refuse a layer count whose norm slots would leave their range.
    ///
    /// # Errors
    ///
    /// A message naming the count and the limit when `n_layers >`
    /// [`MAX_SLOT_LAYERS`].
    pub fn check_layer_count(n_layers: usize) -> Result<(), String> {
        if n_layers > MAX_SLOT_LAYERS {
            return Err(format!(
                "{n_layers} layers exceed the {MAX_SLOT_LAYERS}-layer range of the GPU slot \
                 namespace; refusing to derive slots that would alias the final norm"
            ));
        }
        Ok(())
    }

    /// Every norm / final-norm / LM-head slot an `n_layers`-layer model's Q1
    /// fused paths can populate, with the cache each one lives in.
    #[must_use]
    pub fn keys(self, n_layers: usize) -> Vec<(u64, SlotCache)> {
        let mut keys = Vec::with_capacity(n_layers * 4 + 2);
        for layer in 0..n_layers {
            let base = self.norm_base(layer);
            keys.extend((0..4).map(|k| (base + k, SlotCache::Norm)));
        }
        keys.push((self.final_norm(), SlotCache::Norm));
        keys.push((self.lm_head(), SlotCache::Quant));
        keys
    }

    /// Every slot the CUDA Q1 paths can populate for an `n_layers`-layer
    /// model: [`Self::keys`] plus the four weight fallbacks per layer.
    #[must_use]
    pub fn cuda_keys(self, n_layers: usize) -> Vec<u64> {
        let mut keys: Vec<u64> = self.keys(n_layers).into_iter().map(|(s, _)| s).collect();
        for layer in 0..n_layers {
            let base = self.weight_fallback_base(layer);
            keys.extend((0..4).map(|k| base + k));
        }
        keys
    }

    /// Base of layer `layer`'s four CUDA **ternary** norm slots (`+0` attn,
    /// `+1` q, `+2` k, `+3` ffn; `f32` cache). `layer` must be below
    /// [`MAX_SLOT_LAYERS`].
    #[must_use]
    pub const fn ternary_norm_base(self, layer: usize) -> u64 {
        self.slot(TERNARY_NORM_LOCAL_BASE + (layer as u64) * TERNARY_LOCAL_STRIDE)
    }

    /// Base of layer `layer`'s four CUDA ternary weight slots (`+0` fused
    /// QKV, `+1` attention output, `+2` gate‖up, `+3` down; `u8` cache).
    /// `layer` must be below [`MAX_SLOT_LAYERS`].
    #[must_use]
    pub const fn ternary_weight_base(self, layer: usize) -> u64 {
        self.slot(TERNARY_WEIGHT_LOCAL_BASE + (layer as u64) * TERNARY_LOCAL_STRIDE)
    }

    /// Slot of the CUDA ternary final norm (`f32` cache).
    #[must_use]
    pub const fn ternary_final_norm(self) -> u64 {
        self.slot(TERNARY_FINAL_NORM_LOCAL)
    }

    /// Slot of the CUDA ternary LM head (`u8` cache).
    #[must_use]
    pub const fn ternary_lm_head(self) -> u64 {
        self.slot(TERNARY_LM_HEAD_LOCAL)
    }

    /// Every slot the CUDA **ternary** full-forward paths key under this
    /// namespace for an `n_layers`-layer model: four norms and four weights
    /// per layer, the final norm and the LM head.
    ///
    /// `forward_cuda/ternary.rs` and the ternary tail in `forward_cuda/q1.rs`
    /// both compose this layout (`ternary_norm_base`/`ternary_weight_base`
    /// per layer, `ternary_final_norm`/`ternary_lm_head` for the tail) rather
    /// than the process-wide literals `5_000_000 + layer * 10` / `6_000_000 +
    /// layer * 10` / `5_900_000` / `7_000_000` they used to key on, and the
    /// two call sites stay equal by construction — both derive from this same
    /// namespace. The model's `Drop` sweeps every key this method returns.
    #[must_use]
    pub fn cuda_ternary_keys(self, n_layers: usize) -> Vec<u64> {
        let mut keys = Vec::with_capacity(n_layers * 8 + 2);
        for layer in 0..n_layers {
            let norms = self.ternary_norm_base(layer);
            let weights = self.ternary_weight_base(layer);
            keys.extend((0..4).map(|k| norms + k));
            keys.extend((0..4).map(|k| weights + k));
        }
        keys.push(self.ternary_final_norm());
        keys.push(self.ternary_lm_head());
        keys
    }

    // ── CUDA Q4_0 / Q8_0 ("q_std") slots ────────────────────────────────────

    /// Base of layer `layer`'s four CUDA Q4_0 (`q4_0 == true`) or Q8_0
    /// (`q4_0 == false`) norm slots (`+0` attn, `+1` q, `+2` k, `+3` ffn;
    /// `f32` cache). `layer` must be below [`MAX_SLOT_LAYERS`].
    #[must_use]
    pub const fn q_std_norm_base(self, layer: usize, q4_0: bool) -> u64 {
        let base = if q4_0 {
            Q4_0_NORM_LOCAL_BASE
        } else {
            Q8_0_NORM_LOCAL_BASE
        };
        self.slot(base + (layer as u64) * Q_STD_LOCAL_STRIDE)
    }

    /// Base of layer `layer`'s four CUDA Q4_0/Q8_0 weight slots (`+0` fused
    /// QKV, `+1` attention output, `+2` gate‖up, `+3` down; `u8` cache).
    /// `layer` must be below [`MAX_SLOT_LAYERS`].
    #[must_use]
    pub const fn q_std_weight_base(self, layer: usize, q4_0: bool) -> u64 {
        let base = if q4_0 {
            Q4_0_WEIGHT_LOCAL_BASE
        } else {
            Q8_0_WEIGHT_LOCAL_BASE
        };
        self.slot(base + (layer as u64) * Q_STD_LOCAL_STRIDE)
    }

    /// Slot of the CUDA Q4_0/Q8_0 final norm (`f32` cache).
    #[must_use]
    pub const fn q_std_final_norm(self, q4_0: bool) -> u64 {
        self.slot(Q_STD_TAIL_LOCAL_BASE + if q4_0 { 0 } else { 2 })
    }

    /// Slot of the CUDA Q4_0/Q8_0 LM head (`u8` cache).
    #[must_use]
    pub const fn q_std_lm_head(self, q4_0: bool) -> u64 {
        self.slot(Q_STD_TAIL_LOCAL_BASE + if q4_0 { 1 } else { 3 })
    }

    /// Every slot the CUDA Q4_0 **and** Q8_0 full-forward paths key under
    /// this namespace for an `n_layers`-layer model: four norms and four
    /// weights per layer for each of the two formats, plus each format's
    /// final norm and LM head.
    ///
    /// Both formats are always swept together (rather than per-format):
    /// `forward_cuda/mod.rs`'s `Drop` releases a model's whole namespace in
    /// one pass, and a model is exactly one of Q4_0 or Q8_0, so the unused
    /// format's slots are simply never populated — sweeping them anyway
    /// costs one extra `HashMap` miss per release, not a correctness risk.
    #[must_use]
    pub fn cuda_std_quant_keys(self, n_layers: usize) -> Vec<u64> {
        let mut keys = Vec::with_capacity(n_layers * 16 + 4);
        for q4_0 in [true, false] {
            for layer in 0..n_layers {
                let norms = self.q_std_norm_base(layer, q4_0);
                let weights = self.q_std_weight_base(layer, q4_0);
                keys.extend((0..4).map(|k| norms + k));
                keys.extend((0..4).map(|k| weights + k));
            }
            keys.push(self.q_std_final_norm(q4_0));
            keys.push(self.q_std_lm_head(q4_0));
        }
        keys
    }

    // ── CUDA K-quant (Q2K/Q3K/Q4K/Q5K/Q6K/Q8K) slots ────────────────────────

    /// Base of layer `layer`'s four CUDA K-quant norm slots for format
    /// index `format_idx` (`0..`[`K_QUANT_FORMAT_COUNT`], `KQuantFormat`'s
    /// declaration order: Q2K, Q3K, Q4K, Q5K, Q6K, Q8K — `+0` attn, `+1` q,
    /// `+2` k, `+3` ffn; `f32` cache). `layer` must be below
    /// [`MAX_SLOT_LAYERS`], `format_idx` below [`K_QUANT_FORMAT_COUNT`].
    #[must_use]
    pub const fn k_quant_norm_base(self, layer: usize, format_idx: u64) -> u64 {
        let format_off = format_idx * K_QUANT_FORMAT_LOCAL_STRIDE;
        self.slot(K_QUANT_NORM_LOCAL_BASE + format_off + (layer as u64) * K_QUANT_LOCAL_STRIDE)
    }

    /// Base of layer `layer`'s four CUDA K-quant weight slots for format
    /// index `format_idx` (same indexing as [`Self::k_quant_norm_base`];
    /// `+0` fused QKV, `+1` attention output, `+2` gate‖up, `+3` down; `u8`
    /// cache). `layer` must be below [`MAX_SLOT_LAYERS`], `format_idx`
    /// below [`K_QUANT_FORMAT_COUNT`].
    #[must_use]
    pub const fn k_quant_weight_base(self, layer: usize, format_idx: u64) -> u64 {
        let format_off = format_idx * K_QUANT_FORMAT_LOCAL_STRIDE;
        self.slot(K_QUANT_WEIGHT_LOCAL_BASE + format_off + (layer as u64) * K_QUANT_LOCAL_STRIDE)
    }

    /// Slot of format index `format_idx`'s CUDA K-quant final norm (`f32`
    /// cache). `format_idx` must be below [`K_QUANT_FORMAT_COUNT`].
    #[must_use]
    pub const fn k_quant_final_norm(self, format_idx: u64) -> u64 {
        self.slot(K_QUANT_TAIL_LOCAL_BASE + format_idx)
    }

    /// Slot of format index `format_idx`'s CUDA K-quant LM head (`u8`
    /// cache). `format_idx` must be below [`K_QUANT_FORMAT_COUNT`].
    #[must_use]
    pub const fn k_quant_lm_head(self, format_idx: u64) -> u64 {
        self.slot(K_QUANT_TAIL_LOCAL_BASE + K_QUANT_LM_HEAD_TAIL_OFFSET + format_idx)
    }

    /// Every slot the CUDA K-quant full-forward paths key under this
    /// namespace for an `n_layers`-layer model, across **all**
    /// [`K_QUANT_FORMAT_COUNT`] formats (same rationale as
    /// [`Self::cuda_std_quant_keys`]: a model is exactly one format,
    /// sweeping every format's range is still correct and keeps the `Drop`
    /// path format-agnostic).
    #[must_use]
    pub fn cuda_k_quant_keys(self, n_layers: usize) -> Vec<u64> {
        let n_formats = K_QUANT_FORMAT_COUNT as usize;
        let mut keys = Vec::with_capacity(n_layers * 8 * n_formats + 2 * n_formats);
        for format_idx in 0..K_QUANT_FORMAT_COUNT {
            for layer in 0..n_layers {
                let norms = self.k_quant_norm_base(layer, format_idx);
                let weights = self.k_quant_weight_base(layer, format_idx);
                keys.extend((0..4).map(|k| norms + k));
                keys.extend((0..4).map(|k| weights + k));
            }
            keys.push(self.k_quant_final_norm(format_idx));
            keys.push(self.k_quant_lm_head(format_idx));
        }
        keys
    }
}

/// The weight-cache slot of a mapped tensor: its address, checked against
/// [`MIN_TENSOR_SLOT`] and against the tag bit. `None` for an empty slice
/// (which has no stable address) or an address outside that range.
#[must_use]
pub fn mapped_tensor_slot(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }
    let slot = bytes.as_ptr() as u64;
    if !(MIN_TENSOR_SLOT..SLOT_TAG).contains(&slot) {
        return None;
    }
    Some(slot)
}

/// Address of a borrowed weight slice, as an anchor candidate.
///
/// Like [`mapping_anchor`], compiled (and unit-tested) on every host but
/// called only by the Metal build.
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
fn slice_anchor<T>(blocks: &[T]) -> Option<u64> {
    if blocks.is_empty() {
        return None;
    }
    let addr = blocks.as_ptr() as u64;
    (MIN_TENSOR_SLOT..SLOT_TAG).contains(&addr).then_some(addr)
}

/// The anchor of the mapping a model's weights are borrowed from: the first
/// block's `Q1_0_g128` / `TQ2_0_g128` `attn_q` blocks, else the LM head's
/// (`Q1_0_g128` / `TQ2_0_g128`), else `None` (see the module docs for why no
/// other data may anchor a mapping).
///
/// Only the Metal build joins mappings (a CUDA model keys its slots on its own
/// `cuda_model_epoch`, and a CPU build keys nothing), so elsewhere this is
/// reached only by the unit tests — it stays compiled so the anchor rule is
/// tested on every host.
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
pub(super) fn mapping_anchor(
    blocks: &[TransformerBlock<'_>],
    output: &OutputWeight<'_>,
) -> Option<u64> {
    let from_block = blocks.first().and_then(|block| {
        block
            .attn_q_blocks()
            .and_then(slice_anchor)
            .or_else(|| block.attn_q_blocks_ternary().and_then(slice_anchor))
    });
    from_block.or_else(|| match output {
        OutputWeight::OneBit(linear) => slice_anchor(linear.blocks()),
        OutputWeight::Ternary(linear) => slice_anchor(linear.blocks()),
        _ => None,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-mapping state and the process-wide registry
// ═══════════════════════════════════════════════════════════════════════════

/// Called once with a mapping's epoch when its last holder drops, iff some
/// holder marked the mapping as having touched the GPU.
pub type ReleaseHook = fn(u64);

/// The state every replica (and every block of every replica) of one mapping
/// shares: its anchor, its epoch and whether any of them put buffers on the
/// GPU under that epoch.
#[derive(Debug)]
pub struct MappingState {
    anchor: Option<u64>,
    epoch: u64,
    gpu_used: AtomicBool,
    release_hook: Option<ReleaseHook>,
}

impl MappingState {
    fn new(anchor: Option<u64>, epoch: u64, release_hook: Option<ReleaseHook>) -> Self {
        Self {
            anchor,
            epoch,
            gpu_used: AtomicBool::new(false),
            release_hook,
        }
    }

    /// A namespace no other model can join: `epoch` must be fresh (or, for a
    /// deliberate control in a test, an epoch whose buffers the caller wants
    /// to address). Its buffers are released when the last holder drops.
    #[must_use]
    pub fn private(epoch: u64, release_hook: Option<ReleaseHook>) -> Arc<Self> {
        Arc::new(Self::new(None, epoch, release_hook))
    }

    /// The mapping's anchor address; `None` for a private namespace.
    #[must_use]
    pub fn anchor(&self) -> Option<u64> {
        self.anchor
    }

    /// The epoch every buffer of this mapping is keyed under.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The slot namespace of [`Self::epoch`].
    #[must_use]
    pub fn slots(&self) -> SlotNamespace {
        SlotNamespace::new(self.epoch)
    }

    /// Record that a path is putting (or has put) buffers on the GPU under
    /// this mapping's epoch. Paths call it before **and** after their
    /// dispatch, so an explicit release racing a sibling's upload can never
    /// leave a resident buffer with the flag cleared.
    pub fn mark_gpu_used(&self) {
        self.gpu_used.store(true, Ordering::Release);
    }

    /// Whether any holder has marked the mapping used since it was created or
    /// last released.
    #[must_use]
    pub fn is_gpu_used(&self) -> bool {
        self.gpu_used.load(Ordering::Acquire)
    }

    /// Clear the flag, returning whether it was set (an explicit release).
    pub fn take_gpu_used(&self) -> bool {
        self.gpu_used.swap(false, Ordering::AcqRel)
    }
}

impl Drop for MappingState {
    fn drop(&mut self) {
        if *self.gpu_used.get_mut() {
            if let Some(release) = self.release_hook {
                release(self.epoch);
            }
        }
    }
}

/// One registry entry: the shared state and how many live registrations
/// (models) hold it.
#[derive(Debug)]
struct RegistryEntry {
    state: Arc<MappingState>,
    replicas: usize,
}

/// The table `mapping anchor → (state, replica count)`.
///
/// A plain value so its bookkeeping is unit-testable in isolation; the
/// process uses the one behind [`global_registry`].
#[derive(Debug, Default)]
pub struct MappingRegistry {
    entries: HashMap<u64, RegistryEntry>,
}

impl MappingRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Join the mapping anchored at `anchor`, minting its epoch with `mint`
    /// when this is its first live replica.
    pub fn join(
        &mut self,
        anchor: u64,
        mint: impl FnOnce() -> u64,
        release_hook: Option<ReleaseHook>,
    ) -> Arc<MappingState> {
        let entry = self.entries.entry(anchor).or_insert_with(|| RegistryEntry {
            state: Arc::new(MappingState::new(Some(anchor), mint(), release_hook)),
            replicas: 0,
        });
        entry.replicas = entry.replicas.saturating_add(1);
        Arc::clone(&entry.state)
    }

    /// Leave the mapping anchored at `anchor` under `epoch`.
    ///
    /// Returns the registry's own reference to the state when this was the
    /// last replica — the entry is gone, so the next `join` at this anchor
    /// mints a new epoch. The caller drops the returned `Arc` **after**
    /// releasing the registry lock: if it is the last reference, dropping it
    /// runs the release hook, which must not run under the lock.
    #[must_use]
    pub fn leave(&mut self, anchor: u64, epoch: u64) -> Option<Arc<MappingState>> {
        let entry = self.entries.get_mut(&anchor)?;
        if entry.state.epoch != epoch {
            // A registration can only leave the entry it joined, and an entry
            // with a live registration is never replaced — so this is a
            // stale call, and the live mapping must not be touched.
            return None;
        }
        entry.replicas = entry.replicas.saturating_sub(1);
        if entry.replicas > 0 {
            return None;
        }
        self.entries.remove(&anchor).map(|entry| entry.state)
    }

    /// Live replicas of the mapping anchored at `anchor` (0 when absent).
    #[must_use]
    pub fn replicas(&self, anchor: u64) -> usize {
        self.entries.get(&anchor).map_or(0, |entry| entry.replicas)
    }

    /// The epoch of the live mapping anchored at `anchor`.
    #[must_use]
    pub fn epoch_of(&self, anchor: u64) -> Option<u64> {
        self.entries.get(&anchor).map(|entry| entry.state.epoch)
    }

    /// Number of live mappings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no mapping is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The process-wide registry every `BonsaiModel` joins at construction.
fn global_registry() -> &'static Mutex<MappingRegistry> {
    static REGISTRY: OnceLock<Mutex<MappingRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(MappingRegistry::new()))
}

/// Lock the process-wide registry. A poisoned lock is recovered: `join` and
/// `leave` leave the table consistent at every await-free step, so a panic in
/// an unrelated holder cannot have torn an entry.
fn lock_global() -> MutexGuard<'static, MappingRegistry> {
    global_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Live replicas of the mapping anchored at `anchor`, process-wide.
#[must_use]
pub fn global_replicas(anchor: u64) -> usize {
    lock_global().replicas(anchor)
}

/// The epoch of the live mapping anchored at `anchor`, process-wide.
#[must_use]
pub fn global_epoch_of(anchor: u64) -> Option<u64> {
    lock_global().epoch_of(anchor)
}

/// One model's membership in its mapping's namespace (RAII).
///
/// Taken once per `BonsaiModel` at construction; dropping it leaves the
/// mapping. A registration built for a model with no mapped anchor (or with an
/// explicit epoch) is **private**: nobody else can join it.
#[derive(Debug)]
pub struct MappingRegistration {
    state: Arc<MappingState>,
    registered: bool,
}

impl MappingRegistration {
    /// Join the mapping anchored at `anchor` (shared with every live replica
    /// of it), or — for `None` — take a private namespace. `mint` supplies the
    /// epoch when a new one is needed.
    #[must_use]
    pub fn join(
        anchor: Option<u64>,
        mint: impl FnOnce() -> u64,
        release_hook: Option<ReleaseHook>,
    ) -> Self {
        match anchor {
            Some(anchor) => Self {
                state: lock_global().join(anchor, mint, release_hook),
                registered: true,
            },
            None => Self::private(mint(), release_hook),
        }
    }

    /// A private namespace under `epoch`.
    #[must_use]
    pub fn private(epoch: u64, release_hook: Option<ReleaseHook>) -> Self {
        Self {
            state: MappingState::private(epoch, release_hook),
            registered: false,
        }
    }

    /// The shared state (clone it into the model's blocks).
    #[must_use]
    pub fn state(&self) -> &Arc<MappingState> {
        &self.state
    }

    /// The mapping's epoch.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.state.epoch
    }

    /// The mapping's slot namespace.
    #[must_use]
    pub fn slots(&self) -> SlotNamespace {
        self.state.slots()
    }

    /// Whether this registration joined a shared mapping (as opposed to a
    /// private namespace).
    #[must_use]
    pub fn is_shared_mapping(&self) -> bool {
        self.registered
    }

    /// Live replicas of this registration's mapping (1 for a private one).
    #[must_use]
    pub fn replicas(&self) -> usize {
        match (self.registered, self.state.anchor) {
            (true, Some(anchor)) => global_replicas(anchor),
            _ => 1,
        }
    }
}

impl Drop for MappingRegistration {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        let Some(anchor) = self.state.anchor else {
            return;
        };
        // Take the registry's reference out under the lock, drop it outside:
        // if it is the last one, its `Drop` runs the release hook.
        let last = lock_global().leave(anchor, self.state.epoch);
        drop(last);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — pure: slot composition, anchors and the registry
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicU64;

    /// A deterministic epoch source for registry tests (the process-wide
    /// counter would do too, but the tests assert exact values).
    fn counter() -> impl FnMut() -> u64 {
        let mut next = 100u64;
        move || {
            next += 1;
            next
        }
    }

    #[test]
    fn slots_are_tagged_and_keep_the_historical_local_layout() {
        let ns = SlotNamespace::new(7);
        for slot in ns.cuda_keys(3) {
            assert_eq!(
                slot & SLOT_TAG,
                SLOT_TAG,
                "every composed slot carries the tag"
            );
        }
        assert_eq!(ns.norm_base(1) - ns.norm_base(0), NORM_LOCAL_STRIDE);
        assert_eq!(ns.final_norm() - ns.norm_base(0), 1_000_000);
        assert_eq!(ns.lm_head() - ns.final_norm(), 1_000_000);
        assert_eq!(ns.slot(0) >> SLOT_LOCAL_BITS & SLOT_EPOCH_MASK, 7);
        // The epoch never reaches the tag bit.
        let huge = SlotNamespace::new(u64::MAX);
        assert_eq!(huge.lm_head() & SLOT_TAG, SLOT_TAG);
        assert_eq!(huge.lm_head() & ((1 << SLOT_LOCAL_BITS) - 1), LM_HEAD_LOCAL);
    }

    #[test]
    fn one_namespace_never_repeats_a_slot_and_two_never_share_one() {
        let a = SlotNamespace::new(1);
        let b = SlotNamespace::new(2);
        let keys_a: HashSet<u64> = a.cuda_keys(64).into_iter().collect();
        let keys_b: HashSet<u64> = b.cuda_keys(64).into_iter().collect();
        assert_eq!(
            keys_a.len(),
            64 * 8 + 2,
            "every slot of one model is distinct"
        );
        assert!(keys_a.is_disjoint(&keys_b));
        // The historical literals are outside the namespace.
        for literal in [1_000_000u64, 2_000_000, 3_000_000, 4_000_000] {
            assert!(!keys_a.contains(&literal));
        }
    }

    #[test]
    fn the_largest_layer_count_stays_inside_its_ranges() {
        let ns = SlotNamespace::new(3);
        let last = MAX_SLOT_LAYERS - 1;
        assert!(ns.norm_base(last) + 3 < ns.final_norm());
        // The weight fallbacks of the last layer still fit in the local bits,
        // so they can never carry into the epoch field.
        let last_fallback_local =
            WEIGHT_FALLBACK_LOCAL_BASE + (last as u64) * WEIGHT_FALLBACK_LOCAL_STRIDE + 3;
        assert!(last_fallback_local < (1 << SLOT_LOCAL_BITS));
        assert_eq!(
            ns.weight_fallback_base(last) + 3,
            ns.slot(last_fallback_local)
        );
        assert!(SlotNamespace::check_layer_count(MAX_SLOT_LAYERS).is_ok());
        let err = SlotNamespace::check_layer_count(MAX_SLOT_LAYERS + 1)
            .expect_err("one layer past the range must be refused");
        assert!(err.contains("layers"), "{err}");
    }

    /// The CUDA ternary layout: disjoint from every Q1 slot of the same
    /// namespace (a model is swept with both sets), from another namespace's,
    /// and — for the largest legal layer count — inside its own range.
    #[test]
    fn the_cuda_ternary_layout_is_disjoint_and_in_range() {
        let ns = SlotNamespace::new(11);
        let q1: HashSet<u64> = ns.cuda_keys(64).into_iter().collect();
        let ternary: HashSet<u64> = ns.cuda_ternary_keys(64).into_iter().collect();
        assert_eq!(ternary.len(), 64 * 8 + 2, "every ternary slot is distinct");
        assert!(
            q1.is_disjoint(&ternary),
            "Q1 and ternary slots never coincide"
        );
        let other: HashSet<u64> = SlotNamespace::new(12)
            .cuda_ternary_keys(64)
            .into_iter()
            .collect();
        assert!(ternary.is_disjoint(&other));
        for slot in &ternary {
            assert_eq!(slot & SLOT_TAG, SLOT_TAG);
        }
        // The literals the ternary paths key today are outside the namespace.
        for literal in [5_000_000u64, 6_000_000, 5_900_000, 7_000_000] {
            assert!(!ternary.contains(&literal));
        }
        // At the layer limit the last group is still composed in place: its
        // local part never carries into the epoch field (the range bounds
        // themselves are compile-time assertions next to the constants).
        let last = (MAX_SLOT_LAYERS - 1) as u64;
        let norm_end = TERNARY_NORM_LOCAL_BASE + last * TERNARY_LOCAL_STRIDE + 3;
        let weight_end = TERNARY_WEIGHT_LOCAL_BASE + last * TERNARY_LOCAL_STRIDE + 3;
        assert_eq!(
            ns.ternary_norm_base(MAX_SLOT_LAYERS - 1) + 3,
            ns.slot(norm_end)
        );
        assert_eq!(
            ns.ternary_weight_base(MAX_SLOT_LAYERS - 1) + 3,
            ns.slot(weight_end)
        );
    }

    /// The CUDA Q4_0/Q8_0 layout: disjoint from every Q1 and ternary slot of
    /// the same namespace, from another namespace's, from the two formats'
    /// own literal ranges, and inside its range at the layer limit.
    #[test]
    fn the_cuda_std_quant_layout_is_disjoint_and_in_range() {
        let ns = SlotNamespace::new(13);
        let q1: HashSet<u64> = ns.cuda_keys(64).into_iter().collect();
        let ternary: HashSet<u64> = ns.cuda_ternary_keys(64).into_iter().collect();
        let std_quant: HashSet<u64> = ns.cuda_std_quant_keys(64).into_iter().collect();
        assert_eq!(
            std_quant.len(),
            64 * 16 + 4,
            "every Q4_0+Q8_0 slot is distinct"
        );
        assert!(q1.is_disjoint(&std_quant));
        assert!(ternary.is_disjoint(&std_quant));
        let other: HashSet<u64> = SlotNamespace::new(14)
            .cuda_std_quant_keys(64)
            .into_iter()
            .collect();
        assert!(std_quant.is_disjoint(&other));
        for slot in &std_quant {
            assert_eq!(slot & SLOT_TAG, SLOT_TAG);
        }
        // The literals the old q_std paths keyed are outside the namespace.
        for literal in [
            8_000_000u64,
            9_000_000,
            10_000_000,
            11_000_000,
            8_900_000,
            9_900_000,
            10_900_000,
            11_900_000,
        ] {
            assert!(!std_quant.contains(&literal));
        }
        // At the layer limit each format's norm/weight range still composes
        // in place (no carry into the sibling range or the tail slots).
        let last = MAX_SLOT_LAYERS - 1;
        for q4_0 in [true, false] {
            let norm_last = ns.q_std_norm_base(last, q4_0) + 3;
            let weight_last = ns.q_std_weight_base(last, q4_0) + 3;
            assert!(norm_last < ns.q_std_weight_base(0, q4_0));
            assert!(weight_last < ns.slot(Q_STD_TAIL_LOCAL_BASE));
        }
        assert_ne!(
            ns.q_std_final_norm(true),
            ns.q_std_final_norm(false),
            "Q4_0 and Q8_0 final norms never collide"
        );
        assert_ne!(ns.q_std_lm_head(true), ns.q_std_lm_head(false));
    }

    /// The CUDA K-quant layout: disjoint from every other family's slots of
    /// the same namespace, across all six formats, from another namespace's,
    /// and inside its range at the layer limit.
    #[test]
    fn the_cuda_k_quant_layout_is_disjoint_and_in_range() {
        let ns = SlotNamespace::new(15);
        let q1: HashSet<u64> = ns.cuda_keys(64).into_iter().collect();
        let ternary: HashSet<u64> = ns.cuda_ternary_keys(64).into_iter().collect();
        let std_quant: HashSet<u64> = ns.cuda_std_quant_keys(64).into_iter().collect();
        let k_quant: HashSet<u64> = ns.cuda_k_quant_keys(64).into_iter().collect();
        assert_eq!(
            k_quant.len(),
            64 * 8 * 6 + 2 * 6,
            "every K-quant slot, across all six formats, is distinct"
        );
        assert!(q1.is_disjoint(&k_quant));
        assert!(ternary.is_disjoint(&k_quant));
        assert!(std_quant.is_disjoint(&k_quant));
        let other: HashSet<u64> = SlotNamespace::new(16)
            .cuda_k_quant_keys(64)
            .into_iter()
            .collect();
        assert!(k_quant.is_disjoint(&other));
        for slot in &k_quant {
            assert_eq!(slot & SLOT_TAG, SLOT_TAG);
        }
        // The literals the old k_quant paths keyed are outside the namespace.
        for literal in [
            12_000_000u64,
            13_000_000,
            22_000_000,
            23_000_000,
            24_000_000,
            25_000_000,
        ] {
            assert!(!k_quant.contains(&literal));
        }
        // Every format's final norm / LM head is distinct from every other
        // format's — the old literals shared no format index at all, so two
        // K-quant formats in one process could serve each other's norms.
        let final_norms: HashSet<u64> = (0..K_QUANT_FORMAT_COUNT)
            .map(|f| ns.k_quant_final_norm(f))
            .collect();
        let lm_heads: HashSet<u64> = (0..K_QUANT_FORMAT_COUNT)
            .map(|f| ns.k_quant_lm_head(f))
            .collect();
        assert_eq!(final_norms.len(), K_QUANT_FORMAT_COUNT as usize);
        assert_eq!(lm_heads.len(), K_QUANT_FORMAT_COUNT as usize);
        assert!(final_norms.is_disjoint(&lm_heads));
        // At the layer limit each format's own range still composes in
        // place (no carry into the next format's or the epoch bits).
        let last = MAX_SLOT_LAYERS - 1;
        for format_idx in 0..K_QUANT_FORMAT_COUNT {
            let norm_last = ns.k_quant_norm_base(last, format_idx) + 3;
            let weight_last = ns.k_quant_weight_base(last, format_idx) + 3;
            assert!(norm_last < ns.k_quant_weight_base(0, format_idx));
            assert!(weight_last < ns.slot(K_QUANT_TAIL_LOCAL_BASE));
        }
    }

    #[test]
    fn the_norm_and_quant_keys_are_classified() {
        let keys = SlotNamespace::new(9).keys(2);
        let kinds: Vec<SlotCache> = keys.iter().map(|(_, k)| *k).collect();
        assert_eq!(kinds.len(), 10);
        assert!(kinds[..9].iter().all(|k| *k == SlotCache::Norm));
        assert_eq!(kinds[9], SlotCache::Quant);
    }

    #[test]
    fn mapped_tensor_slots_respect_the_floor_and_the_tag() {
        let data = vec![1u8; 64];
        let slot = mapped_tensor_slot(&data).expect("a heap address is above the floor");
        assert!((MIN_TENSOR_SLOT..SLOT_TAG).contains(&slot));
        assert_eq!(
            mapped_tensor_slot(&[]),
            None,
            "an empty slice has no address"
        );
        // SAFETY: a zero-length slice needs only a non-null, aligned pointer
        // and is never dereferenced — only its address is read.
        let below: &[u8] =
            unsafe { std::slice::from_raw_parts(std::ptr::without_provenance(0x1000), 0) };
        assert_eq!(slice_anchor(below), None);
    }

    /// The anchor rule: the first block's mapped `attn_q` (`Q1_0_g128` or
    /// `TQ2_0_g128`), else the LM head's, else nothing — pure data, so it is
    /// checked on every host (a CPU-tier kernel uploads nothing).
    #[test]
    fn the_mapping_anchor_is_the_first_attn_q_then_the_lm_head() {
        use crate::layers::linear::{Linear1Bit, LinearLayer, LinearTernary};
        use crate::layers::rms_norm::RmsNorm;
        use oxibonsai_core::tensor::BlockQ1_0G128;
        use oxibonsai_core::BlockTQ2_0_g128;
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};

        const H: usize = 128;
        let kernel = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));
        let q1 = |n: usize| -> &'static [BlockQ1_0G128] {
            let block = BlockQ1_0G128 {
                d: half::f16::from_f32(0.5),
                qs: [0x5A; 16],
            };
            Box::leak(vec![block; n].into_boxed_slice())
        };
        let tq2 = |n: usize| -> &'static [BlockTQ2_0_g128] {
            let block = BlockTQ2_0_g128 {
                qs: [0x46; 32], // codes 10, 01, 00, 01: +1, 0, -1, 0 — never 0b11
                d: half::f16::from_f32(0.5),
            };
            Box::leak(vec![block; n].into_boxed_slice())
        };
        // Square H x H projections: H * H / 128 = H blocks each.
        let one_bit = |blocks: &'static [BlockQ1_0G128]| -> LinearLayer<'static> {
            Linear1Bit::new(blocks, H, H, Arc::clone(&kernel))
                .expect("Q1 linear")
                .into()
        };
        let ternary = |blocks: &'static [BlockTQ2_0_g128]| -> LinearLayer<'static> {
            LinearTernary::new(blocks, H, H, Arc::clone(&kernel))
                .expect("ternary linear")
                .into()
        };
        let block_with = |q: LinearLayer<'static>| {
            TransformerBlock::new(
                0,
                RmsNorm::new(vec![1.0; H], 1e-6),
                q,
                one_bit(q1(H)),
                one_bit(q1(H)),
                one_bit(q1(H)),
                RmsNorm::new(vec![1.0; H], 1e-6),
                RmsNorm::new(vec![1.0; H], 1e-6),
                RmsNorm::new(vec![1.0; H], 1e-6),
                one_bit(q1(H)),
                one_bit(q1(H)),
                one_bit(q1(H)),
                1,
                1,
                H,
                H,
            )
        };
        let head_blocks = q1(H);
        let head = OutputWeight::OneBit(
            Linear1Bit::new(head_blocks, H, H, Arc::clone(&kernel)).expect("LM head"),
        );
        let head_anchor = Some(head_blocks.as_ptr() as u64);

        let q1_q = q1(H);
        let q1_block = block_with(one_bit(q1_q));
        assert_eq!(
            mapping_anchor(std::slice::from_ref(&q1_block), &head),
            Some(q1_q.as_ptr() as u64),
            "a Q1 model is anchored on its first block's attn_q"
        );
        let tq2_q = tq2(H);
        let tq2_block = block_with(ternary(tq2_q));
        assert_eq!(
            mapping_anchor(std::slice::from_ref(&tq2_block), &head),
            Some(tq2_q.as_ptr() as u64),
            "a ternary model is anchored on its first block's attn_q"
        );
        assert_eq!(
            mapping_anchor(&[], &head),
            head_anchor,
            "without blocks the LM head anchors the mapping"
        );
        assert_eq!(
            mapping_anchor(&[], &OutputWeight::zero_fp32(4, H)),
            None,
            "nothing mapped: the model gets a private namespace"
        );
    }

    #[test]
    fn replicas_share_one_state_until_the_last_leaves() {
        let mut registry = MappingRegistry::new();
        let mut mint = counter();
        let a = registry.join(0xAAAA_0000_0000, &mut mint, None);
        let b = registry.join(0xAAAA_0000_0000, &mut mint, None);
        assert!(
            Arc::ptr_eq(&a, &b),
            "two replicas of one mapping share one state"
        );
        assert_eq!(a.epoch(), 101);
        assert_eq!(registry.replicas(0xAAAA_0000_0000), 2);
        let other = registry.join(0xBBBB_0000_0000, &mut mint, None);
        assert_ne!(
            other.epoch(),
            a.epoch(),
            "a different mapping gets its own epoch"
        );

        assert!(registry.leave(0xAAAA_0000_0000, a.epoch()).is_none());
        assert_eq!(registry.replicas(0xAAAA_0000_0000), 1);
        let last = registry
            .leave(0xAAAA_0000_0000, a.epoch())
            .expect("the last replica takes the entry out");
        assert!(Arc::ptr_eq(&last, &a));
        assert_eq!(
            registry.epoch_of(0xAAAA_0000_0000),
            None,
            "the entry is gone"
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn a_reused_address_mints_a_new_epoch_after_the_count_reaches_zero() {
        let mut registry = MappingRegistry::new();
        let mut mint = counter();
        let first = registry.join(0xCAFE_0000_0000, &mut mint, None);
        let first_epoch = first.epoch();
        drop(registry.leave(0xCAFE_0000_0000, first_epoch));
        drop(first);
        let second = registry.join(0xCAFE_0000_0000, &mut mint, None);
        assert_ne!(
            second.epoch(),
            first_epoch,
            "a later mapping at a reused address must not inherit the old epoch"
        );
    }

    #[test]
    fn a_stale_leave_does_not_touch_the_live_mapping() {
        let mut registry = MappingRegistry::new();
        let mut mint = counter();
        let live = registry.join(0xD00D_0000_0000, &mut mint, None);
        assert!(registry.leave(0xD00D_0000_0000, live.epoch() + 1).is_none());
        assert_eq!(registry.replicas(0xD00D_0000_0000), 1);
        assert!(
            registry.leave(0xFFFF_0000_0000, 1).is_none(),
            "absent anchor"
        );
    }

    /// Records the epochs the release hook was called with.
    static RELEASED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    fn record_release(epoch: u64) {
        RELEASED
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(epoch);
    }
    fn released(epoch: u64) -> usize {
        RELEASED
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|e| **e == epoch)
            .count()
    }
    /// Unique epochs for the hook tests, far from the process counter.
    fn test_epoch() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1 << 38);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    #[test]
    fn the_release_hook_runs_once_with_the_last_holder_and_only_if_used() {
        // Unused: no release, however it is dropped.
        let unused = MappingState::private(test_epoch(), Some(record_release));
        let unused_epoch = unused.epoch();
        drop(unused);
        assert_eq!(released(unused_epoch), 0);

        // Used, held by a "model" and two "blocks": released exactly once,
        // when the last of the three goes.
        let used = MappingState::private(test_epoch(), Some(record_release));
        let epoch = used.epoch();
        let block_a = Arc::clone(&used);
        let block_b = Arc::clone(&used);
        block_a.mark_gpu_used();
        drop(used);
        drop(block_a);
        assert_eq!(released(epoch), 0, "a block still holds the mapping");
        drop(block_b);
        assert_eq!(released(epoch), 1);

        // An explicit release clears the flag, so the drop does not repeat it.
        let explicit = MappingState::private(test_epoch(), Some(record_release));
        let explicit_epoch = explicit.epoch();
        explicit.mark_gpu_used();
        assert!(explicit.take_gpu_used());
        assert!(!explicit.is_gpu_used());
        drop(explicit);
        assert_eq!(released(explicit_epoch), 0);
    }

    #[test]
    fn global_registrations_share_and_release_with_the_last_replica() {
        // A unique fake anchor inside the valid address range, so parallel
        // tests (and real models) can never share it.
        let anchor = MIN_TENSOR_SLOT + (test_epoch() << 8);
        let a = MappingRegistration::join(Some(anchor), test_epoch, Some(record_release));
        let b = MappingRegistration::join(Some(anchor), test_epoch, Some(record_release));
        assert!(a.is_shared_mapping());
        assert_eq!(a.epoch(), b.epoch(), "replicas share the epoch");
        assert_eq!(a.replicas(), 2);
        assert_eq!(global_epoch_of(anchor), Some(a.epoch()));
        let epoch = a.epoch();
        a.state().mark_gpu_used();
        drop(a);
        assert_eq!(global_replicas(anchor), 1);
        assert_eq!(released(epoch), 0, "a replica is still alive");
        drop(b);
        assert_eq!(global_replicas(anchor), 0);
        assert_eq!(global_epoch_of(anchor), None);
        assert_eq!(
            released(epoch),
            1,
            "the last replica released the mapping once"
        );

        let again = MappingRegistration::join(Some(anchor), test_epoch, None);
        assert_ne!(
            again.epoch(),
            epoch,
            "a new mapping at the address gets a new epoch"
        );

        let private = MappingRegistration::join(None, test_epoch, None);
        assert!(!private.is_shared_mapping());
        assert_eq!(private.replicas(), 1);
        assert_eq!(private.state().anchor(), None);
    }
}
