//! Content-deduplicated, epoch-refcounted weight cache of
//! [`super::Scirs2Backend`] (`MET-M1` eviction half + Q1 replica sharing).
//!
//! A `#[path]` child of `scirs2_backend` (split out to keep that file under
//! the 2000-line ceiling). Items are `pub(super)` so the backend's upload,
//! lookup and release paths use them directly.

use std::collections::HashMap;

use scirs2_core::gpu::GpuBuffer;
#[cfg(all(feature = "metal", target_os = "macos"))]
use tracing::{debug, warn};

/// On-device layout of a cached weight buffer.
///
/// Part of a buffer's content identity: the same input bytes uploaded
/// through [`super::Scirs2Backend::upload_weights`] (stored verbatim, read by
/// the Q1 kernels) and through [`super::Scirs2Backend::upload_weights_ternary`]
/// (reformatted to SoA) are *different* device buffers and must never alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum UploadLayout {
    /// Bytes stored exactly as uploaded.
    RawBlocks,
    /// `TQ2_0_g128` blocks reformatted to `[N×2 B scales][N×32 B qs]`.
    TernarySoa,
}

/// Content identity of a resident buffer: layout, length and a 128-bit
/// fingerprint of the stored bytes.
///
/// The fingerprint only *nominates* a candidate — a hit is confirmed by a
/// byte-for-byte comparison against the resident buffer before any sharing
/// happens (see `Scirs2Backend::retain_upload`), so a collision can cost an
/// extra upload but can never serve the wrong weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ContentKey {
    /// On-device layout.
    pub(super) layout: UploadLayout,
    /// Stored length in bytes.
    pub(super) len: usize,
    /// Fingerprint, low lane.
    pub(super) lo: u64,
    /// Fingerprint, high lane.
    pub(super) hi: u64,
}

/// 128-bit content fingerprint over little-endian 8-byte words.
///
/// Two independent multiply-rotate lanes, finalised with a SplitMix64
/// avalanche each. Not cryptographic and not meant to be: it exists to find
/// the (single) previously-uploaded buffer a byte comparison then confirms.
pub(super) fn content_fingerprint(bytes: &[u8]) -> (u64, u64) {
    const K0: u64 = 0x9E37_79B9_7F4A_7C15;
    const K1: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const K2: u64 = 0x1656_67B1_9E37_79F9;
    const K3: u64 = 0xD6E8_FEB8_6659_FD93;

    fn avalanche(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    let len = bytes.len() as u64;
    let mut lo = K0 ^ len;
    let mut hi = K1.wrapping_add(len.rotate_left(17));
    let mut words = bytes.chunks_exact(8);
    for chunk in &mut words {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        let w = u64::from_le_bytes(word);
        lo = (lo ^ w).wrapping_mul(K2).rotate_left(31);
        hi = hi
            .wrapping_add(w.rotate_left(23))
            .wrapping_mul(K3)
            .rotate_left(29);
    }
    let rest = words.remainder();
    if !rest.is_empty() {
        let mut word = [0u8; 8];
        word[..rest.len()].copy_from_slice(rest);
        let w = u64::from_le_bytes(word) ^ ((rest.len() as u64) << 56);
        lo = (lo ^ w).wrapping_mul(K2).rotate_left(31);
        hi = hi
            .wrapping_add(w.rotate_left(23))
            .wrapping_mul(K3)
            .rotate_left(29);
    }
    (avalanche(lo ^ hi.rotate_left(7)), avalanche(hi ^ K0))
}

/// Whether `buf` holds exactly `bytes` (length and every byte).
///
/// The confirmation step behind every shared upload. On Apple silicon the
/// scirs2 Metal buffers live in unified memory, so this is a host memcpy +
/// compare; it runs at model-load time only.
pub(super) fn resident_bytes_equal(buf: &GpuBuffer<u8>, bytes: &[u8]) -> bool {
    if buf.len() != bytes.len() {
        return false;
    }
    let mut host = vec![0u8; bytes.len()];
    if buf.copy_to_host(&mut host).is_err() {
        return false;
    }
    host == bytes
}

/// One resident weight buffer.
pub(super) struct CachedWeight {
    /// The device buffer itself.
    pub(super) buf: GpuBuffer<u8>,
    /// Its size in bytes.
    pub(super) bytes: usize,
    /// Its content identity (so a release can drop the dedupe index entry).
    pub(super) content: ContentKey,
    /// Live registrations across every epoch; the buffer is freed when this
    /// reaches zero through [`WeightCacheState::release_epoch`].
    pub(super) refs: usize,
}

/// The whole weight cache, behind one mutex so the entry table, the
/// content index and the per-epoch registrations can never disagree.
///
/// **Sharing (verify:METAL-CONCURRENCY blocking #1).** Every replica of a
/// Q1 model used to run `upload_weights_to_gpu`, and every upload minted a
/// fresh handle from `NEXT_HANDLE_ID` — N replicas, N resident copies, in
/// this cache *and* in the `MetalGraph` weight cache the Q1 fused path keys
/// on these very handle ids. Now an upload whose bytes are already resident
/// (same layout, byte-for-byte equal) returns the **existing** handle, so
/// the fused path's `WeightKey::slot`s coincide too and both caches hold the
/// model once.
///
/// **Eviction (`MET-M1`).** Each retained upload registers one reference
/// under the uploading thread's
/// [`crate::gpu_backend::current_upload_epoch`]; a [`Self::release_epoch`]
/// drops exactly that epoch's references and frees only the buffers nobody
/// else still references.
#[derive(Default)]
pub(super) struct WeightCacheState {
    /// Resident buffers by handle id.
    pub(super) entries: HashMap<u64, CachedWeight>,
    /// Content → the handle currently serving it.
    pub(super) by_content: HashMap<ContentKey, u64>,
    /// Epoch → (handle → registrations under that epoch).
    pub(super) by_epoch: HashMap<u64, HashMap<u64, usize>>,
    /// Bytes currently resident.
    pub(super) resident_bytes: u64,
    /// Cumulative bytes freshly placed on the device (monotonic).
    pub(super) uploaded_total: u64,
}

impl WeightCacheState {
    /// Clone the buffer behind `handle`, if resident.
    pub(super) fn get(&self, handle: u64) -> Option<GpuBuffer<u8>> {
        self.entries.get(&handle).map(|entry| entry.buf.clone())
    }

    /// Record one more reference to `handle` under `epoch`.
    pub(super) fn register(&mut self, epoch: u64, handle: u64) {
        if let Some(entry) = self.entries.get_mut(&handle) {
            entry.refs = entry.refs.saturating_add(1);
            let count = self
                .by_epoch
                .entry(epoch)
                .or_default()
                .entry(handle)
                .or_insert(0);
            *count = count.saturating_add(1);
        }
    }

    /// Insert a freshly uploaded buffer and register its first reference.
    pub(super) fn insert_fresh(
        &mut self,
        epoch: u64,
        handle: u64,
        buf: GpuBuffer<u8>,
        content: ContentKey,
        bytes: usize,
    ) {
        self.entries.insert(
            handle,
            CachedWeight {
                buf,
                bytes,
                content,
                refs: 0,
            },
        );
        // A concurrent identical upload may have indexed this content first;
        // keep that mapping (both buffers are correct, this one is merely not
        // discoverable for future sharing).
        self.by_content.entry(content).or_insert(handle);
        self.resident_bytes = self.resident_bytes.saturating_add(bytes as u64);
        self.uploaded_total = self.uploaded_total.saturating_add(bytes as u64);
        self.register(epoch, handle);
    }

    /// Drop every registration `epoch` holds; free and return the
    /// `(handle, bytes)` of each buffer that no longer has any reference.
    pub(super) fn release_epoch(&mut self, epoch: u64) -> Vec<(u64, usize)> {
        let Some(registrations) = self.by_epoch.remove(&epoch) else {
            return Vec::new();
        };
        let mut freed = Vec::new();
        for (handle, count) in registrations {
            let now_unreferenced = match self.entries.get_mut(&handle) {
                Some(entry) => {
                    entry.refs = entry.refs.saturating_sub(count);
                    entry.refs == 0
                }
                None => false,
            };
            if now_unreferenced {
                if let Some(entry) = self.entries.remove(&handle) {
                    if self.by_content.get(&entry.content) == Some(&handle) {
                        self.by_content.remove(&entry.content);
                    }
                    self.resident_bytes = self.resident_bytes.saturating_sub(entry.bytes as u64);
                    freed.push((handle, entry.bytes));
                }
            }
        }
        freed
    }

    /// Evict everything (process-wide), returning how many buffers went.
    pub(super) fn clear(&mut self) -> usize {
        let evicted = self.entries.len();
        self.entries.clear();
        self.by_content.clear();
        self.by_epoch.clear();
        self.resident_bytes = 0;
        evicted
    }

    /// Registrations `epoch` currently holds.
    pub(super) fn registrations(&self, epoch: u64) -> usize {
        self.by_epoch
            .get(&epoch)
            .map_or(0, |handles| handles.values().sum())
    }
}

/// Drop the `MetalGraph` weight-cache mirrors of freed handles (`MET-M1`).
///
/// The Q1 fused path keys `MetalGraph`'s cache on these handle ids
/// (`WeightKey::legacy(WeightKind::Q1Soa, handle)`, via
/// `build_cached_weights`), and a handle id is only ever reused by the
/// deduplication above for byte-identical content, so once its last
/// registration is gone no live model can still need the mirror. The
/// ternary/Prism kinds are evicted too: a block-dispatch fallback may have
/// keyed a mirror on the same id.
///
/// Also forwards the release to `MetalGraph::release_model`, which drops any
/// entry keyed under this epoch proper (the engine mints its epochs from the
/// same counter as `WeightKey::model_epoch`).
///
/// Never *opens* a Metal device: when no session is live nothing can hold a
/// mirror, and a release path must not pay a device-open + MSL compile.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) fn release_metal_mirrors(freed: &[(u64, usize)], model_epoch: u64) {
    use crate::gpu_backend::metal_full_layer::types::{WeightKey, WeightKind};
    use crate::gpu_backend::MetalGraph;

    let session = match MetalGraph::current_session() {
        Some(session) => Some(session),
        None if MetalGraph::live_session_count() > 0 => MetalGraph::global().ok(),
        None => None,
    };
    let Some(graph) = session else {
        return;
    };
    for &(handle, _) in freed {
        for kind in [
            WeightKind::Q1Soa,
            WeightKind::Tq2Soa,
            WeightKind::Pq2Soa,
            WeightKind::Ptq1Soa,
        ] {
            if let Err(e) = graph.evict_weight(WeightKey::legacy(kind, handle)) {
                warn!(error = %e, handle, "failed to evict a MetalGraph weight mirror");
            }
        }
    }
    match graph.release_model(model_epoch) {
        Ok(0) => {}
        Ok(dropped) => debug!(
            dropped,
            model_epoch, "released MetalGraph entries of the epoch"
        ),
        Err(e) => warn!(error = %e, model_epoch, "MetalGraph release_model failed"),
    }
}
