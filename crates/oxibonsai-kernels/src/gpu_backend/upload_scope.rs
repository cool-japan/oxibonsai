//! GPU weight-upload attribution and the CPU-only backend-selection scope.
//!
//! Split out of `gpu_backend/mod.rs` so that file keeps headroom under the
//! workspace 2000-line ceiling; everything here is re-exported from
//! [`crate::gpu_backend`], so callers name it exactly as before.
//!
//! * **Attribution (`MET-M1`).** [`GpuUploadScope`] tags every weight upload a
//!   backend performs on the current thread with a model epoch, so
//!   [`crate::gpu_backend::GpuBackendTrait::release_model`] can later free
//!   exactly that model's buffers -- and, with a deduplicating backend, only
//!   the ones no other replica still references.
//! * **CPU-only selection (engine `Backend::Cpu`).** [`CpuOnlyBackendScope`]
//!   makes [`crate::gpu_backend::select_backend`] choose the CPU tier on the
//!   current thread, which is how an explicit CPU request reaches the
//!   dispatchers `oxibonsai-model` creates internally.

// ═══════════════════════════════════════════════════════════════════════════
// Weight-upload attribution (MET-M1 eviction + replica sharing)
// ═══════════════════════════════════════════════════════════════════════════

/// The epoch an upload is attributed to when no [`GpuUploadScope`] is active
/// on the uploading thread.
///
/// Such uploads have no known owner, so
/// [`GpuBackendTrait::release_model`](crate::gpu_backend::GpuBackendTrait::release_model)
/// never frees them: they live for the process, exactly as every upload did
/// before `MET-M1`. It is numerically the same value as the Metal weight
/// cache's `LEGACY_MODEL_EPOCH`, and [`next_gpu_model_epoch`] never returns
/// it.
pub const UNATTRIBUTED_MODEL_EPOCH: u64 = 0;

/// Epoch counter for builds without the Metal backend (the Metal build
/// shares the `MetalGraph` weight-key namespace instead; see
/// [`next_gpu_model_epoch`]).
#[cfg(not(all(feature = "metal", target_os = "macos")))]
static NEXT_UPLOAD_EPOCH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(UNATTRIBUTED_MODEL_EPOCH + 1);

/// Mint a fresh, process-unique model epoch for GPU weight attribution.
///
/// On a Metal build this is the very same counter `MetalGraph`'s composite
/// `WeightKey { model_epoch, .. }` uses (`MetalGraph::next_model_epoch`, a
/// bare atomic that never opens the device), so a backend that forwards a
/// release to `MetalGraph::release_model(epoch)` can only ever drop entries
/// that belong to this epoch. Never returns [`UNATTRIBUTED_MODEL_EPOCH`].
#[must_use]
pub fn next_gpu_model_epoch() -> u64 {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        super::metal_full_layer::types::next_model_epoch()
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        NEXT_UPLOAD_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

/// What the uploads made inside one [`GpuUploadScope`] cost.
///
/// A "fresh" upload allocated a new device buffer; a "shared" one found an
/// identical, already-resident buffer (typically uploaded by a sibling
/// replica of the same model) and registered a reference to it instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UploadStats {
    /// Uploads that allocated a new device buffer.
    pub fresh_buffers: usize,
    /// Bytes those fresh uploads placed on the device.
    pub fresh_bytes: u64,
    /// Uploads satisfied by an existing, byte-identical resident buffer.
    pub shared_buffers: usize,
    /// Bytes those shared uploads did **not** have to place on the device.
    pub shared_bytes: u64,
}

impl UploadStats {
    /// Total retained uploads, fresh plus shared.
    #[must_use]
    pub fn total_buffers(&self) -> usize {
        self.fresh_buffers + self.shared_buffers
    }

    /// `true` when nothing was uploaded inside the scope (a CPU tier, or a
    /// route that skips the upload).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_buffers() == 0
    }
}

/// One active attribution frame on the thread-local stack.
#[derive(Debug, Clone, Copy)]
struct UploadFrame {
    epoch: u64,
    stats: UploadStats,
}

thread_local! {
    /// Stack of active [`GpuUploadScope`]s on this thread (innermost last).
    static UPLOAD_FRAMES: std::cell::RefCell<Vec<UploadFrame>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// RAII attribution of GPU weight uploads to a model epoch (`MET-M1`).
///
/// While the scope lives, every weight upload a backend performs **on this
/// thread** is registered under [`Self::epoch`], so a later
/// [`GpuBackendTrait::release_model`](crate::gpu_backend::GpuBackendTrait::release_model)
/// of that epoch frees exactly what this
/// scope uploaded (minus whatever another epoch still shares). Uploads happen
/// synchronously on the thread that loads the model — `KernelDispatcher`'s
/// upload entry points call straight into the backend — which is why a
/// thread-local frame, rather than a parameter threaded through every
/// `LinearLayer::upload_to_gpu` in `oxibonsai-model`, is enough.
///
/// Scopes nest: the innermost one wins, and dropping it restores the outer
/// one. The type is `!Send` so it cannot be dropped on a different thread
/// from the one that entered it.
#[derive(Debug)]
pub struct GpuUploadScope {
    epoch: u64,
    /// Stack depth *after* this scope's frame was pushed.
    depth: usize,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl GpuUploadScope {
    /// Attribute this thread's uploads to `model_epoch` until the returned
    /// scope is dropped.
    #[must_use]
    pub fn enter(model_epoch: u64) -> Self {
        let depth = UPLOAD_FRAMES
            .try_with(|frames| {
                let mut frames = frames.borrow_mut();
                frames.push(UploadFrame {
                    epoch: model_epoch,
                    stats: UploadStats::default(),
                });
                frames.len()
            })
            .unwrap_or(0);
        Self {
            epoch: model_epoch,
            depth,
            _not_send: std::marker::PhantomData,
        }
    }

    /// The epoch uploads are currently attributed to by this scope.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// What this scope has uploaded so far.
    #[must_use]
    pub fn stats(&self) -> UploadStats {
        if self.depth == 0 {
            return UploadStats::default();
        }
        UPLOAD_FRAMES
            .try_with(|frames| {
                frames
                    .borrow()
                    .get(self.depth - 1)
                    .map(|frame| frame.stats)
                    .unwrap_or_default()
            })
            .unwrap_or_default()
    }

    /// End the scope and report what it uploaded.
    #[must_use]
    pub fn finish(self) -> UploadStats {
        self.stats()
    }
}

impl Drop for GpuUploadScope {
    fn drop(&mut self) {
        if self.depth == 0 {
            return;
        }
        let keep = self.depth - 1;
        let _ = UPLOAD_FRAMES.try_with(|frames| {
            frames.borrow_mut().truncate(keep);
        });
    }
}

/// The epoch a weight upload made on this thread right now is attributed
/// to: the innermost active [`GpuUploadScope`]'s, else
/// [`UNATTRIBUTED_MODEL_EPOCH`].
#[must_use]
pub fn current_upload_epoch() -> u64 {
    UPLOAD_FRAMES
        .try_with(|frames| frames.borrow().last().map(|frame| frame.epoch))
        .ok()
        .flatten()
        .unwrap_or(UNATTRIBUTED_MODEL_EPOCH)
}

/// Record one retained upload against the innermost active scope (no-op
/// outside any scope). Called by backends that keep a weight cache.
pub fn note_weight_upload(bytes: u64, shared: bool) {
    let _ = UPLOAD_FRAMES.try_with(|frames| {
        if let Some(frame) = frames.borrow_mut().last_mut() {
            if shared {
                frame.stats.shared_buffers += 1;
                frame.stats.shared_bytes = frame.stats.shared_bytes.saturating_add(bytes);
            } else {
                frame.stats.fresh_buffers += 1;
                frame.stats.fresh_bytes = frame.stats.fresh_bytes.saturating_add(bytes);
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU-only backend selection scope (engine `Backend::Cpu`)
// ═══════════════════════════════════════════════════════════════════════════

thread_local! {
    /// Nesting depth of active [`CpuOnlyBackendScope`]s on this thread.
    static CPU_ONLY_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// RAII scope that makes
/// [`select_backend`](crate::gpu_backend::select_backend) — and therefore every
/// `KernelDispatcher::auto_detect()` made on this thread — choose the CPU
/// tier.
///
/// `oxibonsai-model` builds each layer's own `Arc<KernelDispatcher>` with
/// `auto_detect()` internally, so an engine asked to run on the CPU cannot
/// honour that request merely by pinning *its* dispatcher: the layers would
/// still dispatch their GEMVs to the GPU. Constructing the model inside this
/// scope is what makes a CPU request mean CPU all the way down, with no
/// change to the model's constructors. The scope affects only the thread
/// that entered it, and only while it lives; it is `!Send`.
#[derive(Debug)]
pub struct CpuOnlyBackendScope {
    _not_send: std::marker::PhantomData<*const ()>,
}

impl CpuOnlyBackendScope {
    /// Force CPU backend selection on this thread until the scope drops.
    #[must_use]
    pub fn enter() -> Self {
        let _ = CPU_ONLY_DEPTH.try_with(|depth| depth.set(depth.get().saturating_add(1)));
        Self {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for CpuOnlyBackendScope {
    fn drop(&mut self) {
        let _ = CPU_ONLY_DEPTH.try_with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Whether a [`CpuOnlyBackendScope`] is active on this thread.
#[must_use]
pub fn cpu_only_backend_active() -> bool {
    CPU_ONLY_DEPTH
        .try_with(|depth| depth.get() > 0)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Upload attribution scopes (MET-M1) ──────────────────────────────

    #[test]
    fn upload_scope_attributes_and_restores_the_outer_epoch() {
        assert_eq!(current_upload_epoch(), UNATTRIBUTED_MODEL_EPOCH);
        let outer_epoch = next_gpu_model_epoch();
        let inner_epoch = next_gpu_model_epoch();
        assert_ne!(outer_epoch, UNATTRIBUTED_MODEL_EPOCH);
        assert_ne!(outer_epoch, inner_epoch, "epochs are never reused");
        {
            let outer = GpuUploadScope::enter(outer_epoch);
            assert_eq!(current_upload_epoch(), outer_epoch);
            note_weight_upload(100, false);
            {
                let inner = GpuUploadScope::enter(inner_epoch);
                assert_eq!(current_upload_epoch(), inner_epoch);
                note_weight_upload(40, true);
                note_weight_upload(60, false);
                let stats = inner.finish();
                assert_eq!(stats.fresh_buffers, 1);
                assert_eq!(stats.fresh_bytes, 60);
                assert_eq!(stats.shared_buffers, 1);
                assert_eq!(stats.shared_bytes, 40);
                assert_eq!(stats.total_buffers(), 2);
            }
            // The inner scope's uploads never leak into the outer tally.
            assert_eq!(current_upload_epoch(), outer_epoch);
            let stats = outer.stats();
            assert_eq!(stats.fresh_buffers, 1);
            assert_eq!(stats.fresh_bytes, 100);
            assert_eq!(stats.shared_buffers, 0);
        }
        assert_eq!(current_upload_epoch(), UNATTRIBUTED_MODEL_EPOCH);
        // Outside any scope a noted upload goes nowhere and panics nothing.
        note_weight_upload(7, false);
    }

    #[test]
    fn upload_scope_is_thread_local() {
        let epoch = next_gpu_model_epoch();
        let _scope = GpuUploadScope::enter(epoch);
        let seen = std::thread::spawn(current_upload_epoch)
            .join()
            .expect("probe thread");
        assert_eq!(
            seen, UNATTRIBUTED_MODEL_EPOCH,
            "another thread must not inherit this thread's attribution"
        );
        assert_eq!(current_upload_epoch(), epoch);
    }

    #[test]
    fn empty_upload_stats_report_empty() {
        assert!(UploadStats::default().is_empty());
        let scope = GpuUploadScope::enter(next_gpu_model_epoch());
        assert!(scope.finish().is_empty());
    }

    // ── CPU-only backend scope (engine `Backend::Cpu`) ───────────────────

    #[test]
    fn cpu_only_scope_forces_the_cpu_backend_and_nests() {
        assert!(!cpu_only_backend_active());
        {
            let _outer = CpuOnlyBackendScope::enter();
            assert!(cpu_only_backend_active());
            let backend = super::super::select_backend();
            assert_eq!(backend.name(), "cpu");
            assert!(!backend.is_accelerated());
            {
                let _inner = CpuOnlyBackendScope::enter();
                assert!(cpu_only_backend_active());
            }
            assert!(cpu_only_backend_active(), "the outer scope still holds");
        }
        assert!(!cpu_only_backend_active());
    }
}
