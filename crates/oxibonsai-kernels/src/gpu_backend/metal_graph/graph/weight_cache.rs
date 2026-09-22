//! Kind-tagged, epoch-scoped store behind `MetalGraph`'s weight cache (MET-02).
//!
//! The cache is process-wide and shared by every upload path (raw f32, Q1 SoA,
//! TQ2 SoA, PQ2 SoA, PTQ1 SoA, and the image crate's pointer-keyed weights).
//! Before MET-02 it was a bare `HashMap<u64, Arc<MetalWeightHandle>>`:
//!
//! * nothing identified the **format**, so `2_000_000` could be inserted as a
//!   Q1 `final_norm` buffer and served to the ternary GEMV as if it were TQ2;
//! * nothing identified the **model**, so a second load of the same model
//!   re-used stale buffers and nothing ever freed them.
//!
//! [`WeightCache`] fixes both: entries are keyed by
//! [`WeightKey`] `{ model_epoch, kind, slot }`, a lookup whose
//! `(model_epoch, slot)` is already held under a different
//! [`WeightKind`] returns [`WeightKindMismatch`] instead of a stale hit, and
//! [`WeightCache::release_model`] drops every entry of one epoch while
//! reporting the bytes freed.
//!
//! The store is generic over its value type so the keying, accounting and
//! release logic is unit-tested without a GPU; `MetalGraph` instantiates it
//! with `Arc<MetalWeightHandle>`.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::gpu_backend::metal_full_layer::types::{
    WeightKey, WeightKind, WEIGHT_KIND_MISMATCH_TAG,
};

use super::super::error::MetalWeightHandle;

/// A cache slot already holds a buffer in a different on-GPU format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WeightKindMismatch {
    /// The key that was requested.
    pub requested: WeightKey,
    /// The kind already resident under `requested.slot_id()`.
    pub found: WeightKind,
}

impl fmt::Display for WeightKindMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{WEIGHT_KIND_MISMATCH_TAG}: epoch {} slot {} already holds a {} buffer but {} was \
             requested — the two layouts are not interchangeable; use a distinct slot or release \
             the model first",
            self.requested.model_epoch, self.requested.slot, self.found, self.requested.kind
        )
    }
}

/// Size accounting for a cached value, so the store can maintain a byte gauge.
pub(super) trait CachedWeight {
    /// Bytes of GPU memory this entry keeps resident.
    fn cached_byte_len(&self) -> u64;
}

impl CachedWeight for Arc<MetalWeightHandle> {
    fn cached_byte_len(&self) -> u64 {
        self.byte_len() as u64
    }
}

/// Kind-tagged, epoch-scoped weight store.
pub(super) struct WeightCache<V> {
    /// The cached buffers, keyed by their full composite identity.
    entries: HashMap<WeightKey, V>,
    /// `(model_epoch, slot) → kind` index, so a format collision is detected in
    /// O(1) instead of a scan, and so a slot can hold exactly one format.
    slot_kind: HashMap<(u64, u64), WeightKind>,
    /// Sum of `cached_byte_len()` over `entries`.
    resident_bytes: u64,
}

impl<V: CachedWeight> WeightCache<V> {
    /// An empty cache.
    pub(super) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            slot_kind: HashMap::new(),
            resident_bytes: 0,
        }
    }

    /// Look up `key`.
    ///
    /// * `Ok(Some(_))` — a buffer of the requested kind is resident.
    /// * `Ok(None)` — nothing is resident for this slot; the caller uploads.
    /// * `Err(_)` — the slot holds a *different* format. Serving it would feed
    ///   one quantization's bytes to another's decoder, so it is refused.
    pub(super) fn lookup(&self, key: WeightKey) -> Result<Option<&V>, WeightKindMismatch> {
        match self.slot_kind.get(&key.slot_id()) {
            Some(&found) if found != key.kind => Err(WeightKindMismatch {
                requested: key,
                found,
            }),
            _ => Ok(self.entries.get(&key)),
        }
    }

    /// Insert `value` under `key`, returning the bytes added to the gauge.
    ///
    /// Replacing an existing entry of the same kind (the non-resident
    /// evict-and-reupload pattern) subtracts the old size first.
    pub(super) fn insert(&mut self, key: WeightKey, value: V) -> u64 {
        let added = value.cached_byte_len();
        if let Some(previous) = self.entries.insert(key, value) {
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(previous.cached_byte_len());
        }
        self.slot_kind.insert(key.slot_id(), key.kind);
        self.resident_bytes = self.resident_bytes.saturating_add(added);
        added
    }

    /// Remove one entry, returning the bytes freed.
    pub(super) fn remove(&mut self, key: WeightKey) -> u64 {
        match self.entries.remove(&key) {
            Some(removed) => {
                self.slot_kind.remove(&key.slot_id());
                let freed = removed.cached_byte_len();
                self.resident_bytes = self.resident_bytes.saturating_sub(freed);
                freed
            }
            None => 0,
        }
    }

    /// Drop every entry belonging to `model_epoch`.
    ///
    /// Returns `(entries dropped, bytes freed)`. The GPU buffers are released
    /// as soon as no other handle is outstanding, which is what makes unloading
    /// a model actually return its memory.
    pub(super) fn release_model(&mut self, model_epoch: u64) -> (usize, u64) {
        let mut freed = 0u64;
        let mut dropped = 0usize;
        self.entries.retain(|key, value| {
            if key.model_epoch == model_epoch {
                freed = freed.saturating_add(value.cached_byte_len());
                dropped += 1;
                false
            } else {
                true
            }
        });
        self.slot_kind.retain(|(epoch, _), _| *epoch != model_epoch);
        self.resident_bytes = self.resident_bytes.saturating_sub(freed);
        (dropped, freed)
    }

    /// Bytes of GPU weight memory currently held by the cache.
    pub(super) fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Number of cached buffers.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_backend::metal_full_layer::types::{next_model_epoch, LEGACY_MODEL_EPOCH};

    /// Stand-in for `Arc<MetalWeightHandle>` so the keying logic is testable
    /// without a Metal device.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct FakeWeight(u64);

    impl CachedWeight for FakeWeight {
        fn cached_byte_len(&self) -> u64 {
            self.0
        }
    }

    fn cache() -> WeightCache<FakeWeight> {
        WeightCache::new()
    }

    /// The MET-02 acceptance case: insert K as `Q1Soa`, request K as `Tq2Soa`.
    #[test]
    fn a_slot_inserted_as_q1_refuses_a_tq2_lookup() {
        let mut cache = cache();
        let q1 = WeightKey::legacy(WeightKind::Q1Soa, 2_000_000);
        let tq2 = WeightKey::legacy(WeightKind::Tq2Soa, 2_000_000);
        cache.insert(q1, FakeWeight(1024));

        assert_eq!(cache.lookup(q1), Ok(Some(&FakeWeight(1024))));
        let err = cache
            .lookup(tq2)
            .expect_err("a differently-formatted slot must not be served");
        assert_eq!(err.found, WeightKind::Q1Soa);
        assert_eq!(err.requested.kind, WeightKind::Tq2Soa);
        let msg = err.to_string();
        assert!(msg.contains(WEIGHT_KIND_MISMATCH_TAG), "{msg}");
        assert!(msg.contains("q1_soa") && msg.contains("tq2_soa"), "{msg}");
    }

    #[test]
    fn every_pair_of_distinct_kinds_collides_on_a_shared_slot() {
        let kinds = [
            WeightKind::RawF32,
            WeightKind::Q1Soa,
            WeightKind::Tq2Soa,
            WeightKind::Pq2Soa,
            WeightKind::Ptq1Soa,
        ];
        for inserted in kinds {
            for requested in kinds {
                let mut cache = cache();
                cache.insert(WeightKey::new(7, inserted, 42), FakeWeight(8));
                let got = cache.lookup(WeightKey::new(7, requested, 42));
                if inserted == requested {
                    assert_eq!(got, Ok(Some(&FakeWeight(8))), "{inserted} vs {requested}");
                } else {
                    assert_eq!(
                        got.expect_err("kinds differ").found,
                        inserted,
                        "{inserted} vs {requested}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_same_slot_in_another_epoch_is_a_separate_entry() {
        let mut cache = cache();
        let first = WeightKey::new(1, WeightKind::Tq2Soa, 6_000_000);
        let second = WeightKey::new(2, WeightKind::Q1Soa, 6_000_000);
        cache.insert(first, FakeWeight(100));
        // A different epoch shares neither the entry nor the format constraint.
        assert_eq!(cache.lookup(second), Ok(None));
        cache.insert(second, FakeWeight(200));
        assert_eq!(cache.lookup(first), Ok(Some(&FakeWeight(100))));
        assert_eq!(cache.resident_bytes(), 300);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn release_model_frees_exactly_one_epoch_and_returns_the_gauge_to_zero() {
        let mut cache = cache();
        let epoch_a = next_model_epoch();
        let epoch_b = next_model_epoch();
        assert_ne!(epoch_a, epoch_b);
        assert_ne!(epoch_a, LEGACY_MODEL_EPOCH);

        for slot in 0..4u64 {
            cache.insert(
                WeightKey::new(epoch_a, WeightKind::Tq2Soa, slot),
                FakeWeight(1_000),
            );
            cache.insert(
                WeightKey::new(epoch_b, WeightKind::Q1Soa, slot),
                FakeWeight(10),
            );
        }
        cache.insert(WeightKey::legacy(WeightKind::RawF32, 999), FakeWeight(7));
        assert_eq!(cache.resident_bytes(), 4 * 1_000 + 4 * 10 + 7);

        let (dropped, freed) = cache.release_model(epoch_a);
        assert_eq!((dropped, freed), (4, 4_000));
        assert_eq!(cache.resident_bytes(), 4 * 10 + 7);
        for slot in 0..4u64 {
            assert_eq!(
                cache.lookup(WeightKey::new(epoch_a, WeightKind::Tq2Soa, slot)),
                Ok(None),
                "epoch {epoch_a} slot {slot} must be gone"
            );
            // The released slot no longer constrains its format either.
            assert_eq!(
                cache.lookup(WeightKey::new(epoch_a, WeightKind::Pq2Soa, slot)),
                Ok(None)
            );
        }

        let (dropped_b, _) = cache.release_model(epoch_b);
        assert_eq!(dropped_b, 4);
        assert_eq!(cache.release_model(LEGACY_MODEL_EPOCH), (1, 7));
        assert_eq!(cache.resident_bytes(), 0, "gauge must return to zero");
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn releasing_an_unknown_epoch_is_a_no_op() {
        let mut cache = cache();
        cache.insert(WeightKey::new(3, WeightKind::RawF32, 1), FakeWeight(64));
        assert_eq!(cache.release_model(99), (0, 0));
        assert_eq!(cache.resident_bytes(), 64);
    }

    #[test]
    fn remove_and_reinsert_track_the_gauge_exactly() {
        let mut cache = cache();
        let key = WeightKey::legacy(WeightKind::RawF32, 0xDEAD_BEEF);
        assert_eq!(cache.remove(key), 0, "removing an absent key frees nothing");
        cache.insert(key, FakeWeight(512));
        assert_eq!(cache.resident_bytes(), 512);
        // Re-uploading the same slot (non-resident image weights do this)
        // replaces rather than double-counts.
        cache.insert(key, FakeWeight(256));
        assert_eq!(cache.resident_bytes(), 256);
        assert_eq!(cache.remove(key), 256);
        assert_eq!(cache.resident_bytes(), 0);
        // …and the slot is free to take another format afterwards.
        assert_eq!(
            cache.lookup(WeightKey::legacy(WeightKind::Tq2Soa, 0xDEAD_BEEF)),
            Ok(None)
        );
    }

    #[test]
    fn model_epochs_are_unique_and_never_the_legacy_epoch() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            let epoch = next_model_epoch();
            assert_ne!(epoch, LEGACY_MODEL_EPOCH);
            assert!(seen.insert(epoch), "epoch {epoch} handed out twice");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Device-backed tests — the cache as `MetalGraph` exposes it
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod metal_graph_tests {
    use super::super::MetalGraph;
    use super::*;
    use crate::gpu_backend::metal_full_layer::types::next_model_epoch;
    use metal::Device;

    /// `n_blocks` valid ternary (qs-first) 34-byte blocks: scale `1.0`, codes
    /// cycling through `00/01/10` only.
    fn tq2_blocks(n_blocks: usize) -> Vec<u8> {
        let mut out = vec![0u8; n_blocks * 34];
        for (i, block) in out.chunks_exact_mut(34).enumerate() {
            for (j, byte) in block[..32].iter_mut().enumerate() {
                let c = |s: usize| ((i + j + s) % 3) as u8;
                *byte = c(0) | (c(1) << 2) | (c(2) << 4) | (c(3) << 6);
            }
            block[32..34].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
        out
    }

    /// `n_blocks` valid Q1 (18-byte) blocks.
    fn q1_blocks(n_blocks: usize) -> Vec<u8> {
        (0..n_blocks * 18).map(|i| (i % 251) as u8).collect()
    }

    /// MET-02 acceptance: a slot uploaded as `Q1Soa` must not be served to a
    /// `Tq2Soa` request — the collision that made `2_000_000` (the Q1
    /// `final_norm` handle) alias the ternary norm base.
    #[test]
    fn requesting_a_q1_slot_as_tq2_errors_instead_of_returning_a_stale_hit() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");
        let slot = 2_000_000u64;
        let epoch = next_model_epoch();

        let q1 = graph
            .get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Q1Soa, slot), || {
                graph.upload_q1_weight_soa(&q1_blocks(4))
            })
            .expect("Q1 upload");
        assert_eq!(q1.byte_len(), 4 * 18);

        let err = match graph
            .get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Tq2Soa, slot), || {
                graph.upload_tq2_weight_soa(&tq2_blocks(4))
            }) {
            Ok(_) => panic!("a Q1 buffer must never be served to the ternary path"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains(WEIGHT_KIND_MISMATCH_TAG), "{err}");
        assert!(err.contains("q1_soa") && err.contains("tq2_soa"), "{err}");

        // The legacy (bare-u64) entry points collide the same way.
        let legacy_slot = 3_000_000u64 ^ (epoch << 32);
        graph
            .get_or_upload_q1_weight_soa(legacy_slot, &q1_blocks(2))
            .expect("legacy Q1 upload");
        assert!(graph
            .get_or_upload_tq2_weight_soa(legacy_slot, &tq2_blocks(2))
            .is_err());

        graph.release_model(epoch).expect("release epoch");
        graph
            .evict_weight(WeightKey::legacy(WeightKind::Q1Soa, legacy_slot))
            .expect("evict legacy");
    }

    /// `release_model` frees every buffer of one epoch and the byte gauge
    /// returns to where it started.
    #[test]
    fn release_model_frees_the_epoch_and_restores_the_byte_gauge() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");
        let epoch = next_model_epoch();
        let bytes_before = graph.bytes_uploaded();
        let count_before = graph.cached_weight_count().expect("count");

        let mut expected_bytes = 0u64;
        for slot in 0..3u64 {
            let blocks = tq2_blocks(2 + slot as usize);
            let handle = graph
                .get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Tq2Soa, slot), || {
                    graph.upload_tq2_weight_soa(&blocks)
                })
                .expect("ternary upload");
            expected_bytes += handle.byte_len() as u64;
        }
        assert_eq!(graph.bytes_uploaded(), bytes_before + expected_bytes);
        assert_eq!(
            graph.cached_weight_count().expect("count"),
            count_before + 3
        );
        assert_eq!(
            graph.bytes_uploaded(),
            graph.resident_weight_bytes().expect("resident bytes"),
            "the lock-free gauge must mirror the cache's own accounting"
        );

        assert_eq!(graph.release_model(epoch).expect("release"), 3);
        assert_eq!(
            graph.bytes_uploaded(),
            bytes_before,
            "the gauge must return to its pre-model value"
        );
        assert_eq!(graph.cached_weight_count().expect("count"), count_before);
        // Releasing twice is a no-op, and the slots are free again.
        assert_eq!(graph.release_model(epoch).expect("release again"), 0);
    }

    /// MET-14: raw-f32 and quantized uploads are counted separately, and a
    /// cache hit increments neither.
    #[test]
    fn upload_counters_separate_norms_from_quantized_weights() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");
        let epoch = next_model_epoch();
        let norms_before = graph.norm_upload_count();
        let quant_before = graph.quant_upload_count();
        let total_before = graph.weight_upload_count();

        let norm_data = vec![0.5f32; 64];
        let norm_key = WeightKey::new(epoch, WeightKind::RawF32, 0);
        let norm = graph
            .get_or_upload_keyed(norm_key, || {
                graph.upload_weight(
                    &norm_data
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )
            })
            .expect("norm upload");
        let quant_key = WeightKey::new(epoch, WeightKind::Tq2Soa, 1);
        let blocks = tq2_blocks(3);
        let quant = graph
            .get_or_upload_keyed(quant_key, || graph.upload_tq2_weight_soa(&blocks))
            .expect("quant upload");

        assert_eq!(graph.norm_upload_count(), norms_before + 1);
        assert_eq!(graph.quant_upload_count(), quant_before + 1);
        assert_eq!(graph.weight_upload_count(), total_before + 2);

        // Cache hits: same keys, no new uploads, same buffers.
        let norm_again = graph
            .get_or_upload_keyed(norm_key, || panic!("must not re-upload a cached norm"))
            .expect("norm hit");
        let quant_again = graph
            .get_or_upload_keyed(quant_key, || panic!("must not re-upload a cached weight"))
            .expect("quant hit");
        assert!(Arc::ptr_eq(&norm, &norm_again));
        assert!(Arc::ptr_eq(&quant, &quant_again));
        assert_eq!(graph.weight_upload_count(), total_before + 2);

        graph.release_model(epoch).expect("release");
    }

    /// MET-11: `PQ2_0` bytes reaching the ternary upload path are rejected by
    /// name, while the PQ2 path accepts them.
    #[test]
    fn a_pq2_tensor_is_refused_by_the_ternary_upload_path() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");
        let epoch = next_model_epoch();

        // A PQ2 block: d first, and a `+2` (0b11) code in the payload.
        let mut pq2 = vec![0u8; 34];
        pq2[0..2].copy_from_slice(&0x2247u16.to_le_bytes());
        pq2[2] = 0b11_10_01_00;

        let err = match graph
            .get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Tq2Soa, 10), || {
                graph.upload_tq2_weight_soa(&pq2)
            }) {
            Ok(_) => panic!("a PQ2 tensor must not be uploaded as ternary"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("upload_pq2_weight_soa"), "{err}");
        assert!(err.contains("0b11"), "{err}");

        let handle = graph
            .get_or_upload_keyed(WeightKey::new(epoch, WeightKind::Pq2Soa, 11), || {
                graph.upload_pq2_weight_soa(&pq2)
            })
            .expect("the PQ2 path must accept it");
        assert_eq!(handle.byte_len(), 34);

        // Legacy entry points behave the same way.
        assert!(graph
            .get_or_upload_tq2_weight_soa(u64::MAX - 1, &pq2)
            .is_err());
        graph
            .get_or_upload_pq2_weight_soa_lazy(u64::MAX - 2, || pq2.clone())
            .expect("legacy PQ2 upload");

        graph.release_model(epoch).expect("release");
        graph
            .evict_weight(WeightKey::legacy(WeightKind::Pq2Soa, u64::MAX - 2))
            .expect("evict legacy PQ2");
    }

    /// Eviction removes exactly one entry and its bytes, leaving the slot free
    /// for another format (the non-resident image-weight pattern).
    #[test]
    fn evicting_a_weight_frees_its_bytes_and_its_slot() {
        if Device::system_default().is_none() {
            return;
        }
        let graph = MetalGraph::new().expect("MetalGraph::new");
        let slot = 0x5AFE_0000_0001u64;
        let before = graph.bytes_uploaded();

        let data: Vec<u8> = (0..256u32).map(|v| v as u8).collect();
        graph.get_or_upload_weight(slot, &data).expect("raw upload");
        assert_eq!(graph.bytes_uploaded(), before + 256);

        graph.evict_f32_weight(slot).expect("evict");
        assert_eq!(graph.bytes_uploaded(), before);
        // Evicting again is a no-op, and the slot now accepts another kind.
        graph.evict_f32_weight(slot).expect("evict again");
        graph
            .get_or_upload_q1_weight_soa(slot, &q1_blocks(1))
            .expect("slot must be reusable for another format after eviction");
        graph
            .evict_weight(WeightKey::legacy(WeightKind::Q1Soa, slot))
            .expect("cleanup");
        assert_eq!(graph.bytes_uploaded(), before);
    }
}
