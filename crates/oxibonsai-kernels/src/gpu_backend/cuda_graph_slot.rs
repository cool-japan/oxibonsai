//! Split out of `gpu_backend/mod.rs` to keep it under the workspace's
//! 2000-line-per-file limit; the module path (`crate::gpu_backend::
//! cuda_graph_slot`) is unchanged.
//!
//! Identity key for the process-global captured-CUDA-graph slot (**F-M1**).
//!
//! The captured CUDA driver graph lives in **one un-keyed global slot**
//! (`cuda_full_layer::CudaFullLayerState::cuda_driver_graph`) that both the Q1
//! and the ternary decode paths read and write. A captured `CUgraphExec` bakes
//! in the per-layer **weight device pointers** it was recorded from, and the
//! only invalidation fires when the activation-buffer dimensions change.
//! `Bonsai-8B` (Q1_0_g128) and `Ternary-Bonsai-8B` (TQ2_0_g128) have identical
//! dimensions, so loading one after the other in one process reuses the
//! buffers, skips invalidation and replays the **first** model's weights —
//! silently wrong logits, no error. Second variant: two models with equal
//! per-layer dimensions but different `n_layers` reallocate the KV cache while
//! leaving the graph intact, so the replay reads through freed pointers. A
//! captured graph must therefore be keyed on [`CudaGraphSlotKey`]: replay only
//! when the caller's key equals the stored key, else drop the holder (freeing
//! the exec) and re-capture. The type is platform-independent so its equality
//! contract is tested on every host; the CUDA replay path consuming it is
//! **UNVALIDATED on hardware** (compile-checked only).

use std::sync::atomic::{AtomicU64, Ordering};

/// Which quantisation family a captured graph was recorded from. Q1 and
/// ternary capture into the same slot and their dimensions can be
/// identical, so the family is part of the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaQuantKind {
    /// `Q1_0_g128` — 1-bit, 18-byte blocks.
    Q1G128,
    /// `TQ2_0_g128` — ternary, 34-byte blocks.
    Tq2G128,
}

/// Monotonic model-load counter. `0` is never returned, so it stays usable
/// as an "unset" sentinel, mirroring `metal_full_layer::next_model_epoch`.
static NEXT_MODEL_EPOCH: AtomicU64 = AtomicU64::new(1);

/// Allocate the next model epoch — one per loaded model, so two models (or
/// two loads of one file) can never share cached GPU state.
pub fn next_cuda_model_epoch() -> u64 {
    NEXT_MODEL_EPOCH.fetch_add(1, Ordering::Relaxed)
}

/// Everything a captured CUDA driver graph is only valid for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaGraphSlotKey {
    /// Which loaded model captured the graph ([`next_cuda_model_epoch`]).
    pub model_epoch: u64,
    /// Which quantisation family the captured kernels belong to.
    pub quant_kind: CudaQuantKind,
    /// Layer count — part of the KV-cache shape, which the graph embeds
    /// pointers into and which the buffer key does *not* cover.
    pub n_layers: u32,
    /// Model hidden size.
    pub hidden_size: u32,
    /// Query head count.
    pub n_q_heads: u32,
    /// Key/value head count.
    pub n_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// KV-cache capacity in tokens.
    pub max_seq: u32,
    /// FFN intermediate size.
    pub intermediate_size: u32,
    /// Fingerprint of the weight handles whose device pointers the capture
    /// baked in — see [`CudaGraphSlotKey::fingerprint_handles`].
    pub weight_fingerprint: u64,
}

impl CudaGraphSlotKey {
    /// Fingerprint the weight-handle identities a capture depends on.
    ///
    /// Mixing every handle plus the count distinguishes any two handle sets,
    /// so a capture recorded against one set can never be replayed for another.
    ///
    /// **How much this field is worth depends on where the ids came from**, and
    /// it is not the whole key on its own:
    ///
    /// - Handles from `cuda_graph::functions::alloc_handle_id` are globally
    ///   unique, so the fingerprint alone separates two uploads — including two
    ///   same-shape finetunes.
    /// - The CUDA **decode** paths do not use that counter. They derive ids from
    ///   the layer index (`oxibonsai-model`'s `forward_cuda/ternary.rs`:
    ///   `6_000_000 + layer * 10`, and the Q1 twin), so two ternary models of
    ///   equal depth produce byte-identical handle sets and therefore the same
    ///   fingerprint. There, [`CudaGraphSlotKey::model_epoch`] — freshly minted
    ///   for every uploaded weight set — is what tells the two models apart.
    ///
    /// Both fields are in the key precisely so neither has to be sufficient
    /// alone.
    pub fn fingerprint_handles(handles: &[u64]) -> u64 {
        let mut bytes = Vec::with_capacity(handles.len() * 8 + 8);
        bytes.extend_from_slice(&(handles.len() as u64).to_le_bytes());
        for h in handles {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        super::kernel_artifact_cache::fnv1a_64(&bytes)
    }

    /// Whether a graph captured under `self` may be replayed for
    /// `requested`. Any difference at all forces a re-capture.
    pub fn may_replay(&self, requested: &Self) -> bool {
        self == requested
    }
}

// ── Captured-graph slot decision (finding F-M1) ──────────────────────────────

/// What the caller must do with the process-global captured-graph slot for the
/// key it is about to run (finding **F-M1**).
///
/// The branch logic lives here, and not inline in the two `cfg`-gated
/// `cuda_full_layer` encode paths, so it is executed by the ordinary CPU test
/// suite on every host — including this project's macOS development machine,
/// where nothing under `cuda_full_layer` is compiled at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaGraphSlotAction {
    /// The stored key matches and the slot holds a live `CUgraphExec`: replay.
    Replay,
    /// The slot is empty, or holds a graph captured under a **different** key.
    /// Drop whatever is there (which frees the exec) and capture afresh — this
    /// is the F-M1 fix: without it, `Bonsai-8B` (Q1) and `Ternary-Bonsai-8B`
    /// (TQ2) share one slot and the second model replays the first's weights.
    Capture,
    /// The stored key matches but capture already failed for it. Run the
    /// kernels eagerly and do **not** retry the capture.
    RunEager,
}

/// Decide what to do with the captured-graph slot.
///
/// `stored` describes the slot as it is: `None` when nothing was ever captured,
/// `Some((key, has_exec))` when a capture was attempted under `key` — with
/// `has_exec == false` meaning "tried and failed, never retry for this key".
pub fn cuda_graph_slot_action(
    stored: Option<(CudaGraphSlotKey, bool)>,
    requested: &CudaGraphSlotKey,
) -> CudaGraphSlotAction {
    match stored {
        None => CudaGraphSlotAction::Capture,
        Some((key, _)) if !key.may_replay(requested) => CudaGraphSlotAction::Capture,
        Some((_, true)) => CudaGraphSlotAction::Replay,
        Some((_, false)) => CudaGraphSlotAction::RunEager,
    }
}

// ── Weight-cache epoch bookkeeping (finding F-M3) ────────────────────────────

/// Epoch value meaning "this GPU upload is not attributed to any loaded model".
///
/// [`next_cuda_model_epoch`] never returns it, so it can never collide with a
/// real model. Registering under it is a no-op: an upload whose owning model is
/// unknown must **leak** rather than be released by an unrelated model's `Drop`,
/// because the device pointer may still be live. Every CUDA upload path that
/// does not thread a model epoch (the prefill and image entry points, whose call
/// sites are outside this module) therefore stays unattributed and uncollected
/// — see this crate's `cuda_graph::cudagraph_global_group::CudaGraph::
/// release_model_epoch`.
pub const UNATTRIBUTED_CUDA_MODEL_EPOCH: u64 = 0;

/// `model_epoch → the GPU weight handles uploaded under it` (finding **F-M3**).
///
/// Before F-M3 the two CUDA weight caches were never evicted, so swapping models
/// in one process leaked every uploaded weight (~2 GB for an 8B) for the life of
/// the process. The bookkeeping is a plain map with no CUDA dependency, so it
/// lives here and is unit-tested on every host; the cache eviction it drives is
/// in the `cfg`-gated `cuda_graph::cudagraph_global_group` and is **UNVALIDATED
/// on hardware**.
#[derive(Debug, Default)]
pub struct EpochWeightRegistry {
    by_epoch: std::collections::HashMap<u64, Vec<u64>>,
}

impl EpochWeightRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `handle_id` was uploaded for `model_epoch`.
    ///
    /// Returns `true` when the handle was newly added. Re-registering the same
    /// handle (every decode token re-enters the upload helpers, which hit the
    /// cache) is idempotent, and [`UNATTRIBUTED_CUDA_MODEL_EPOCH`] is ignored
    /// entirely so an unattributed upload can never be released by mistake.
    pub fn register(&mut self, model_epoch: u64, handle_id: u64) -> bool {
        if model_epoch == UNATTRIBUTED_CUDA_MODEL_EPOCH {
            return false;
        }
        let handles = self.by_epoch.entry(model_epoch).or_default();
        if handles.contains(&handle_id) {
            return false;
        }
        handles.push(handle_id);
        true
    }

    /// Remove and return every handle registered under `model_epoch`, in
    /// registration order. Unknown epochs yield an empty vector.
    pub fn take(&mut self, model_epoch: u64) -> Vec<u64> {
        self.by_epoch.remove(&model_epoch).unwrap_or_default()
    }

    /// How many handles are currently registered under `model_epoch`.
    pub fn handle_count(&self, model_epoch: u64) -> usize {
        self.by_epoch.get(&model_epoch).map_or(0, Vec::len)
    }

    /// How many epochs currently hold at least one handle.
    pub fn epoch_count(&self) -> usize {
        self.by_epoch.len()
    }

    /// Total handles registered across every epoch.
    pub fn total_handles(&self) -> usize {
        self.by_epoch.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(
        quant_kind: CudaQuantKind,
        model_epoch: u64,
        n_layers: u32,
        weight_fingerprint: u64,
    ) -> CudaGraphSlotKey {
        CudaGraphSlotKey {
            model_epoch,
            quant_kind,
            n_layers,
            hidden_size: 4096,
            n_q_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            max_seq: 4096,
            intermediate_size: 11008,
            weight_fingerprint,
        }
    }

    /// The exact F-M1 trigger: `Bonsai-8B` (Q1) and `Ternary-Bonsai-8B`
    /// (TQ2) have identical dimensions, so only the quant family tells the
    /// two captured graphs apart. An identical key still replays.
    #[test]
    fn q1_and_ternary_keys_with_identical_dims_are_never_equal() {
        let q1 = key(CudaQuantKind::Q1G128, 1, 36, 0xfeed);
        let tq2 = key(CudaQuantKind::Tq2G128, 1, 36, 0xfeed);
        assert_ne!(q1, tq2);
        assert!(!q1.may_replay(&tq2));
        assert!(!tq2.may_replay(&q1));
        assert!(q1.may_replay(&key(CudaQuantKind::Q1G128, 1, 36, 0xfeed)));
    }

    /// Epoch, layer count, weight upload and every dimension must each be
    /// enough on their own to forbid a replay.
    #[test]
    fn every_field_participates_in_the_key() {
        let base = key(CudaQuantKind::Q1G128, 1, 36, 7);
        let mut mutations = vec![
            key(CudaQuantKind::Q1G128, 2, 36, 7),
            key(CudaQuantKind::Q1G128, 1, 48, 7),
            key(CudaQuantKind::Q1G128, 1, 36, 8),
        ];
        let dims: [fn(&mut CudaGraphSlotKey); 6] = [
            |k| k.hidden_size = 5120,
            |k| k.n_q_heads = 24,
            |k| k.n_kv_heads = 4,
            |k| k.head_dim = 256,
            |k| k.max_seq = 8192,
            |k| k.intermediate_size = 17408,
        ];
        for mutate in dims {
            let mut m = base;
            mutate(&mut m);
            mutations.push(m);
        }
        for m in mutations {
            assert!(!base.may_replay(&m), "mutation {m:?} must forbid replay");
        }
    }

    #[test]
    fn handle_fingerprints_distinguish_uploads() {
        let a = CudaGraphSlotKey::fingerprint_handles(&[1, 2, 3]);
        assert_eq!(a, CudaGraphSlotKey::fingerprint_handles(&[1, 2, 3]));
        assert_ne!(a, CudaGraphSlotKey::fingerprint_handles(&[3, 2, 1]));
        assert_ne!(a, CudaGraphSlotKey::fingerprint_handles(&[1, 2, 3, 4]));
        assert_ne!(a, CudaGraphSlotKey::fingerprint_handles(&[]));
        assert_ne!(a, CudaGraphSlotKey::fingerprint_handles(&[1, 2, 4]));
    }

    /// **F-M1 acceptance (hardware-free half).** The whole replay/drop/recapture
    /// branch the two `cuda_full_layer` encode paths run, exercised here because
    /// those paths are `cfg`-gated out on this host.
    ///
    /// The exact F-M1 trigger is row 3: a graph captured for `Bonsai-8B` (Q1)
    /// with a ternary forward requesting the slot must yield `Capture`, never
    /// `Replay` — before the fix that bare `if let Some(Some(ref holder))`
    /// replayed the Q1 graph for the ternary model.
    #[test]
    fn slot_action_covers_replay_drop_and_recapture() {
        let q1 = key(CudaQuantKind::Q1G128, 1, 36, 0xfeed);
        let tq2 = key(CudaQuantKind::Tq2G128, 1, 36, 0xfeed);

        // Nothing captured yet -> capture.
        assert_eq!(
            cuda_graph_slot_action(None, &q1),
            CudaGraphSlotAction::Capture
        );
        // Same key, live exec -> replay (the fast path must survive the fix).
        assert_eq!(
            cuda_graph_slot_action(Some((q1, true)), &q1),
            CudaGraphSlotAction::Replay
        );
        // THE F-M1 BUG: Q1 graph in the slot, ternary forward requesting it.
        assert_eq!(
            cuda_graph_slot_action(Some((q1, true)), &tq2),
            CudaGraphSlotAction::Capture
        );
        assert_eq!(
            cuda_graph_slot_action(Some((tq2, true)), &q1),
            CudaGraphSlotAction::Capture
        );
        // Same key, capture previously failed -> run eagerly, never retry.
        assert_eq!(
            cuda_graph_slot_action(Some((q1, false)), &q1),
            CudaGraphSlotAction::RunEager
        );
        // A *failed* capture under a different key must not suppress this
        // key's capture.
        assert_eq!(
            cuda_graph_slot_action(Some((tq2, false)), &q1),
            CudaGraphSlotAction::Capture
        );
    }

    /// F-M1: every field of the key must force a re-capture through the slot
    /// decision too, not merely through `may_replay` — including the second
    /// variant the finding names (equal per-layer dims, different `n_layers`,
    /// which reallocates the KV cache and leaves the graph reading freed
    /// pointers).
    #[test]
    fn slot_action_recaptures_on_every_key_difference() {
        let base = key(CudaQuantKind::Tq2G128, 7, 36, 0xabc);
        let mut mutations = vec![
            key(CudaQuantKind::Q1G128, 7, 36, 0xabc),
            key(CudaQuantKind::Tq2G128, 8, 36, 0xabc),
            key(CudaQuantKind::Tq2G128, 7, 48, 0xabc),
            key(CudaQuantKind::Tq2G128, 7, 36, 0xabd),
        ];
        let dims: [fn(&mut CudaGraphSlotKey); 6] = [
            |k| k.hidden_size = 5120,
            |k| k.n_q_heads = 24,
            |k| k.n_kv_heads = 4,
            |k| k.head_dim = 256,
            |k| k.max_seq = 8192,
            |k| k.intermediate_size = 17408,
        ];
        for mutate in dims {
            let mut m = base;
            mutate(&mut m);
            mutations.push(m);
        }
        for requested in mutations {
            for has_exec in [true, false] {
                assert_eq!(
                    cuda_graph_slot_action(Some((base, has_exec)), &requested),
                    CudaGraphSlotAction::Capture,
                    "stored {base:?} vs requested {requested:?} must re-capture"
                );
            }
        }
    }

    /// **F-M3 acceptance (hardware-free half).** Register N handles under two
    /// epochs, release one, and assert the other survives intact.
    ///
    /// The byte-accounting half (`CudaGraph::weight_cache_bytes` dropping by the
    /// released weights' size) needs a CUDA device and is **not** asserted here;
    /// this covers the bookkeeping that decides *which* handles are freed.
    #[test]
    fn epoch_registry_releases_only_the_named_epoch() {
        let (first, second) = (next_cuda_model_epoch(), next_cuda_model_epoch());
        assert_ne!(first, second);
        let mut registry = EpochWeightRegistry::new();

        for handle in 100..110u64 {
            assert!(registry.register(first, handle), "handle {handle} is new");
        }
        for handle in 200..205u64 {
            assert!(registry.register(second, handle), "handle {handle} is new");
        }
        assert_eq!(registry.handle_count(first), 10);
        assert_eq!(registry.handle_count(second), 5);
        assert_eq!(registry.epoch_count(), 2);
        assert_eq!(registry.total_handles(), 15);

        // Re-registering is idempotent — every decode token re-enters the
        // upload helpers and hits the cache.
        assert!(!registry.register(first, 100));
        assert_eq!(registry.handle_count(first), 10);
        assert_eq!(registry.total_handles(), 15);

        // Releasing the first model frees exactly its ten handles, in order.
        let released = registry.take(first);
        assert_eq!(released, (100..110).collect::<Vec<u64>>());
        assert_eq!(registry.handle_count(first), 0);
        assert_eq!(registry.epoch_count(), 1);
        assert_eq!(registry.total_handles(), 5);
        // The surviving model's weights are untouched.
        assert_eq!(registry.handle_count(second), 5);
        assert_eq!(registry.take(second), (200..205).collect::<Vec<u64>>());
        assert_eq!(registry.total_handles(), 0);
        // Releasing an unknown or already-released epoch frees nothing.
        assert!(registry.take(first).is_empty());
        assert!(registry.take(u64::MAX).is_empty());
    }

    /// F-M3: an upload that carries no model epoch must never be registered,
    /// so a later `release_model_epoch` cannot free a pointer that is still in
    /// use by whoever uploaded it.
    #[test]
    fn unattributed_uploads_are_never_registered() {
        assert_eq!(UNATTRIBUTED_CUDA_MODEL_EPOCH, 0);
        assert!(
            (0..64).all(|_| next_cuda_model_epoch() != UNATTRIBUTED_CUDA_MODEL_EPOCH),
            "a real model epoch must never collide with the unattributed sentinel"
        );
        let mut registry = EpochWeightRegistry::new();
        assert!(!registry.register(UNATTRIBUTED_CUDA_MODEL_EPOCH, 42));
        assert_eq!(registry.handle_count(UNATTRIBUTED_CUDA_MODEL_EPOCH), 0);
        assert_eq!(registry.epoch_count(), 0);
        assert!(registry.take(UNATTRIBUTED_CUDA_MODEL_EPOCH).is_empty());
    }

    #[test]
    fn model_epochs_are_unique_and_never_zero() {
        let epochs: Vec<u64> = (0..32).map(|_| next_cuda_model_epoch()).collect();
        for w in epochs.windows(2) {
            assert!(w[1] > w[0], "epochs must strictly increase");
        }
        assert!(epochs.iter().all(|&e| e != 0), "0 is the unset sentinel");
    }
}
