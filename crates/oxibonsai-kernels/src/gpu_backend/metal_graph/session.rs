//! Shared Metal device vs. per-session graph state (`MET-08`).
//!
//! # Why this split exists
//!
//! Before `MET-08` a single process-wide [`MetalGraph`] owned *everything*:
//! the `Device`, the one `CommandQueue`, the compiled pipelines, the weight
//! cache **and** every piece of mutable per-sequence workspace (the device KV
//! cache, the full-layer intermediates, the prefill buffers, the logits and
//! argmax output buffers). The serialisation was never the `OnceLock` holding
//! that singleton — it was the `MutexGuard`s on that shared workspace, taken
//! at the top of a fused forward (`acquire_full_layer_buffers` then
//! `acquire_kv_cache`) and held through every layer's encode plus `commit()`
//! and `wait_until_completed()`. Two inference replicas therefore could not
//! overlap at all, and — because `GpuKvCache::matches` compares only
//! `(n_layers, n_kv, max_seq, head_dim)` — two replicas of the *same* model
//! shape would silently trample each other's attention state. That is why
//! `engine_pool::resolve_pool_size` clamped GPU pools to one replica.
//!
//! The type is now split along the mutable/immutable line:
//!
//! - [`MetalDevice`] — process-global, `Arc`-shared, effectively immutable:
//!   the `Device`, the compiled [`MetalPipelines`], the lazily compiled
//!   prefill-attention library, the correctly-keyed weight cache (`MET-02`)
//!   and the two DiT I/O pools. **Sharing the weight cache is the point**: it
//!   keeps an N-session process at 1x weights instead of Nx.
//! - [`MetalGraph`] — one per session (one per engine replica): its own
//!   `CommandQueue`, its own device KV cache, full-layer buffers, prefill
//!   buffers, logits buffer and argmax token buffer. No cross-session mutex is
//!   held across `commit()`, so submissions from two sessions genuinely
//!   overlap on the GPU.
//!
//! # Sizing reality
//!
//! A session is cheap to *create* (one `CommandQueue`; every workspace buffer
//! is allocated lazily on first use) but not cheap to *use*: the device KV
//! cache for the 8B at `ctx = 4096` is **604 MB per session**. On a 24 GB box
//! the practical GPU pool is therefore 2–3 sessions even with weights shared
//! 1x, which is why [`MetalGraph::max_sessions`] defaults to a small number
//! and is the bound `engine_pool::resolve_pool_sizing` uses on the GPU tier.
//!
//! # How a session is selected
//!
//! [`MetalGraph::global`] — the accessor every kernel entry point already
//! calls — returns the session **bound to the current thread**, or the
//! process-default session when the thread has no binding. A process that
//! never binds therefore behaves exactly as it did before this split: one
//! session, one queue, one KV cache, byte-identical output. Callers that want
//! real concurrency bind a session for the duration of their work with
//! [`MetalGraph::with_session`] (scoped, re-entrant) or with the
//! [`MetalGraph::bind_current`] / [`MetalGraph::unbind_current_if`] primitives
//! (for RAII wrappers such as `engine_pool::EngineLease`, where the bind and
//! the release live in different functions).

use metal::{CommandQueue, Device};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::super::error::{MetalGraphError, MetalWeightHandle};
use super::super::pipelines::MetalPipelines;
use super::weight_cache::WeightCache;
use super::{GemmIoPool, JointAttnIoPool, MetalGraph};
use crate::gpu_backend::metal_full_layer::types::{WeightKey, WeightKind};
use crate::gpu_backend::metal_prefill;

/// Environment variable overriding [`MetalGraph::max_sessions`].
///
/// A value of `0`, a non-numeric value or an absent variable all leave the
/// built-in default in place.
pub(crate) const MAX_SESSIONS_ENV: &str = "OXIBONSAI_METAL_MAX_SESSIONS";

/// Default ceiling on concurrently live Metal sessions.
///
/// Deliberately small: the bound that matters is not the queue count but the
/// per-session device KV cache (604 MB for the 8B at `ctx = 4096`), so a 24 GB
/// box saturates at 2–3 sessions. Raise it with [`MAX_SESSIONS_ENV`] on a
/// machine with more unified memory.
pub(crate) const DEFAULT_MAX_SESSIONS: usize = 4;

/// The process-global shared device, created on first use.
static GLOBAL_METAL_DEVICE: OnceLock<Mutex<Option<Arc<MetalDevice>>>> = OnceLock::new();

/// The process-default session: what [`MetalGraph::global`] returns on a
/// thread with no explicit binding.
static DEFAULT_SESSION: OnceLock<Mutex<Option<Arc<MetalGraph>>>> = OnceLock::new();

/// Monotonic session-id source, so every session is distinguishable in logs
/// and in the thread-local binding check.
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// The session bound to this thread, if any.
    ///
    /// The `u64` is the bound session's [`MetalGraph::session_id`], cached
    /// beside the `Arc` so the hot path (`EngineLease::deref`) can decide
    /// "already bound to this session" with a load and an integer compare
    /// instead of an `Arc` clone.
    static CURRENT_SESSION: RefCell<Option<(u64, Arc<MetalGraph>)>> =
        const { RefCell::new(None) };
}

// ═══════════════════════════════════════════════════════════════════════════
// MetalDevice — process-global, immutable, shared by every session
// ═══════════════════════════════════════════════════════════════════════════

/// Device-scoped state shared by every [`MetalGraph`] session.
///
/// Everything here is either immutable after construction (`device`,
/// `pipelines`) or explicitly designed to be shared (`weight_cache`, the DiT
/// I/O pools, the upload counters). Nothing here is held across a `commit()`
/// on a session's command queue except the weight-cache lock, which is taken
/// only around a cache lookup or an upload — never around a dispatch.
pub struct MetalDevice {
    /// The Metal device. Cloning a `metal::Device` bumps an Objective-C
    /// refcount; every session's `device` field is the same underlying
    /// `MTLDevice`.
    pub(crate) device: Device,
    /// Compiled pipeline states, shared by every session so a new session
    /// never re-runs MSL compilation (the expensive part of the old
    /// `MetalGraph::new`).
    pub(crate) pipelines: Arc<MetalPipelines>,
    /// Lazily compiled batched-prefill attention pipelines (`perf-01`).
    ///
    /// Shared rather than per-session: it is a compiled Metal *library*, not
    /// workspace state, so making it per-session would make every new session
    /// pay its own `new_library_with_source` on first prefill.
    pub(crate) prefill_attn: Arc<OnceLock<Option<metal_prefill::attention::PrefillAttnPipelines>>>,
    /// Lazy cache of GPU-resident weight buffers, keyed by
    /// [`WeightKey`] `{ model_epoch, kind, slot }` (`MET-02`).
    ///
    /// Shared across sessions **by design**: N replicas of one model hold one
    /// copy of its weights, so per-session memory is the KV cache plus
    /// scratch, not another 1.85 GB of matrices.
    weight_cache: Mutex<WeightCache<Arc<MetalWeightHandle>>>,
    /// Resizable shared-storage I/O scratch for the DiT `encode_gemm_tq2`
    /// path (image crate only — see [`GemmIoPool`]).
    pub(super) gemm_io_pool: Mutex<Option<GemmIoPool>>,
    /// Resizable shared-storage q/k/v/out scratch for the DiT joint-attention
    /// path (image crate only — see [`JointAttnIoPool`]).
    pub(super) joint_attn_pool: Mutex<Option<JointAttnIoPool>>,
    /// Running count of **raw f32** weight uploads that allocated a buffer.
    pub(super) norm_upload_count: AtomicUsize,
    /// Running count of **quantized** weight uploads (Q1/TQ2/PQ2/PTQ1 SoA).
    pub(super) quant_upload_count: AtomicUsize,
    /// GPU weight bytes **currently resident** in `weight_cache` (a gauge).
    pub(super) bytes_uploaded: AtomicU64,
    /// Sessions currently alive on this device (see
    /// [`MetalGraph::live_session_count`]).
    live_sessions: AtomicUsize,
}

// Metal objects (Device, CommandQueue, pipeline states) are Send+Sync in the
// metal crate; every mutable field above is behind a `Mutex` or an atomic.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    /// Open the system default device and compile every pipeline.
    fn open() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let pipelines = Arc::new(MetalPipelines::compile(&device)?);
        Ok(Self::with_pipelines(device, pipelines))
    }

    /// Assemble a device around an already-compiled pipeline set.
    fn with_pipelines(device: Device, pipelines: Arc<MetalPipelines>) -> Self {
        Self {
            device,
            pipelines,
            prefill_attn: Arc::new(OnceLock::new()),
            weight_cache: Mutex::new(WeightCache::new()),
            gemm_io_pool: Mutex::new(None),
            joint_attn_pool: Mutex::new(None),
            norm_upload_count: AtomicUsize::new(0),
            quant_upload_count: AtomicUsize::new(0),
            bytes_uploaded: AtomicU64::new(0),
            live_sessions: AtomicUsize::new(0),
        }
    }

    /// The process-global shared device, opened and compiled on first use.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::DeviceNotFound`] when no Metal device is
    /// available, or a compilation error when the MSL library cannot be built.
    pub fn global() -> Result<Arc<Self>, MetalGraphError> {
        let mutex = GLOBAL_METAL_DEVICE.get_or_init(|| Mutex::new(None));
        let mut guard = mutex
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("MetalDevice lock poisoned".into()))?;
        if let Some(shared) = guard.as_ref() {
            return Ok(Arc::clone(shared));
        }
        let shared = Arc::new(Self::open()?);
        *guard = Some(Arc::clone(&shared));
        Ok(shared)
    }

    /// An **isolated** device: the same `MTLDevice` and the same compiled
    /// pipelines as [`Self::global`], but a private weight cache, private DiT
    /// pools and private upload counters.
    ///
    /// This is what [`MetalGraph::new`] builds on, so a caller that
    /// constructs its own graph (the kernel unit tests, the DiT benchmarks)
    /// keeps the *fully independent* accounting it had before `MET-08` — its
    /// upload counters and cache contents cannot be perturbed by a sibling
    /// test running in parallel — while still skipping the MSL compile.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::global`]'s device/compilation errors.
    pub fn isolated() -> Result<Arc<Self>, MetalGraphError> {
        let shared = Self::global()?;
        Ok(Arc::new(Self::with_pipelines(
            shared.device.clone(),
            Arc::clone(&shared.pipelines),
        )))
    }

    /// Create a fresh command queue on this device (one per session).
    fn new_command_queue(&self) -> CommandQueue {
        self.device.new_command_queue()
    }

    /// Lock the weight cache, mapping a poisoned mutex to an error.
    fn lock_weight_cache(
        &self,
    ) -> Result<MutexGuard<'_, WeightCache<Arc<MetalWeightHandle>>>, MetalGraphError> {
        self.weight_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("weight cache lock poisoned".into()))
    }

    /// Cache-aware upload: return the buffer already resident under `key`, or
    /// run `upload` once and cache its result.
    ///
    /// This is the single place where the cache is consulted, so the kind
    /// check, the upload counters and the resident-byte gauge cannot drift
    /// between upload paths. A slot already holding a **different**
    /// [`WeightKind`] is an error rather than a hit — serving it would feed
    /// one quantization's bytes to another's decoder (`MET-02`).
    ///
    /// `upload` runs with the cache lock held, so two threads racing on the
    /// same key upload exactly once.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::WeightKindMismatch`] when the slot is occupied by a
    /// different [`WeightKind`]; otherwise whatever `upload` returns.
    fn get_or_upload_keyed(
        &self,
        key: WeightKey,
        upload: impl FnOnce() -> Result<MetalWeightHandle, MetalGraphError>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        let mut cache = self.lock_weight_cache()?;
        let hit = match cache.lookup(key) {
            Ok(found) => found.map(Arc::clone),
            Err(mismatch) => {
                // The typed variant carries the two kinds plus the slot and
                // epoch (`MET-02`), but the log line is still the only place
                // the *requested* kind and the found kind appear side by side
                // with the full cache context.
                tracing::error!("{mismatch}");
                return Err(MetalGraphError::WeightKindMismatch {
                    expected: mismatch.requested.kind,
                    found: mismatch.found,
                    slot: mismatch.requested.slot,
                    model_epoch: mismatch.requested.model_epoch,
                });
            }
        };
        if let Some(handle) = hit {
            return Ok(handle);
        }
        let handle = Arc::new(upload()?);
        let added = cache.insert(key, Arc::clone(&handle));
        self.bytes_uploaded.fetch_add(added, Ordering::Relaxed);
        if key.kind.is_quantized() {
            self.quant_upload_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.norm_upload_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(handle)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Session lifecycle on MetalGraph
// ═══════════════════════════════════════════════════════════════════════════

impl MetalGraph {
    /// Assemble a session on `shared`.
    ///
    /// Allocates one `CommandQueue`; every workspace buffer stays `None`
    /// until the first dispatch that needs it.
    pub(super) fn from_shared(shared: Arc<MetalDevice>) -> Self {
        let command_queue = shared.new_command_queue();
        let device = shared.device.clone();
        let pipelines = Arc::clone(&shared.pipelines);
        let prefill_attn = Arc::clone(&shared.prefill_attn);
        shared.live_sessions.fetch_add(1, Ordering::Relaxed);
        Self {
            device,
            command_queue,
            pipelines,
            prefill_attn,
            session_id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
            shared,
            buffers: Mutex::new(None),
            kv_cache: Mutex::new(None),
            full_layer_buffers: Mutex::new(None),
            logits_buf: Mutex::new(None),
            token_id_buf: Mutex::new(None),
            prefill_buffers: Mutex::new(None),
        }
    }

    /// Create a new **isolated** `MetalGraph`.
    ///
    /// The returned graph shares the system `MTLDevice` and the compiled MSL
    /// pipelines with every other graph in the process (so this no longer
    /// costs a shader compile), but owns a private weight cache, private DiT
    /// I/O pools, private upload counters and a private command queue.
    ///
    /// Use [`Self::new_session`] instead when the caller wants to *share*
    /// model weights with the rest of the process — that is what an inference
    /// replica wants. Use this one when independent accounting is the point.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::DeviceNotFound`] when no Metal device is
    /// available, or a compilation error when the MSL library cannot be built.
    pub fn new() -> Result<Self, MetalGraphError> {
        Ok(Self::from_shared(MetalDevice::isolated()?))
    }

    /// Create a new session on the process-shared device.
    ///
    /// The session has its own command queue and its own device KV cache,
    /// full-layer buffers, prefill buffers, logits buffer and argmax token
    /// buffer — the state whose sharing made concurrent GPU inference
    /// impossible — while the device, the compiled pipelines and the weight
    /// cache stay shared, so N sessions hold **1x** the model weights.
    ///
    /// Bind it with [`Self::with_session`] (or [`Self::bind_current`]) for the
    /// duration of the work that should run in it; unbound threads keep using
    /// the process-default session.
    ///
    /// # Errors
    ///
    /// Propagates [`MetalDevice::global`]'s device/compilation errors.
    pub fn new_session() -> Result<Arc<Self>, MetalGraphError> {
        Ok(Arc::new(Self::from_shared(MetalDevice::global()?)))
    }

    /// The process-default session, created on first use.
    fn default_session() -> Result<Arc<Self>, MetalGraphError> {
        let mutex = DEFAULT_SESSION.get_or_init(|| Mutex::new(None));
        let mut guard = mutex
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("MetalGraph lock poisoned".into()))?;
        if let Some(session) = guard.as_ref() {
            return Ok(Arc::clone(session));
        }
        let session = Self::new_session()?;
        *guard = Some(Arc::clone(&session));
        Ok(session)
    }

    /// The session this thread should dispatch into.
    ///
    /// Returns the session bound to the current thread by
    /// [`Self::with_session`] / [`Self::bind_current`], or the process-default
    /// session when this thread has no binding. A process that never binds
    /// sees exactly the pre-`MET-08` behaviour: one session, one queue, one
    /// device KV cache, byte-identical output.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::DeviceNotFound`] when no Metal device is
    /// available, or a compilation error when the MSL library cannot be built.
    pub fn global() -> Result<Arc<Self>, MetalGraphError> {
        if let Some(bound) = Self::current_session() {
            return Ok(bound);
        }
        Self::default_session()
    }

    /// This session's process-unique id.
    #[must_use]
    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    /// The session bound to the current thread, if any.
    #[must_use]
    pub fn current_session() -> Option<Arc<Self>> {
        CURRENT_SESSION
            .try_with(|cell| cell.borrow().as_ref().map(|(_, s)| Arc::clone(s)))
            .ok()
            .flatten()
    }

    /// The id of the session bound to the current thread, if any.
    ///
    /// Cheaper than [`Self::current_session`]: no `Arc` clone.
    #[must_use]
    pub fn current_session_id() -> Option<u64> {
        CURRENT_SESSION
            .try_with(|cell| cell.borrow().as_ref().map(|(id, _)| *id))
            .ok()
            .flatten()
    }

    /// Bind `session` to the current thread, returning the previous binding.
    ///
    /// Prefer [`Self::with_session`], which restores the previous binding for
    /// you. This primitive exists for RAII wrappers whose bind and release
    /// live in different functions (`engine_pool::EngineLease` binds in
    /// `Deref` and releases in `Drop`).
    ///
    /// Binding is a no-op when `session` is already the current binding, so
    /// calling it on every `Deref` costs an integer compare.
    pub fn bind_current(session: &Arc<Self>) -> Option<Arc<Self>> {
        let id = session.session_id;
        CURRENT_SESSION
            .try_with(|cell| {
                let mut slot = cell.borrow_mut();
                if slot.as_ref().is_some_and(|(bound, _)| *bound == id) {
                    return None;
                }
                slot.replace((id, Arc::clone(session)))
                    .map(|(_, prev)| prev)
            })
            .ok()
            .flatten()
    }

    /// Restore a binding previously returned by [`Self::bind_current`].
    pub fn restore_current(previous: Option<Arc<Self>>) {
        let _ = CURRENT_SESSION.try_with(|cell| {
            let mut slot = cell.borrow_mut();
            match previous {
                Some(session) => {
                    let id = session.session_id;
                    slot.replace((id, session));
                }
                None => {
                    slot.take();
                }
            }
        });
    }

    /// Clear this thread's binding **iff** it is the session with
    /// `session_id`. Returns `true` when a binding was actually cleared.
    ///
    /// The id check is what makes a nested or re-entrant binding safe: a lease
    /// that was superseded by an inner one does not clear the inner binding.
    pub fn unbind_current_if(session_id: u64) -> bool {
        CURRENT_SESSION
            .try_with(|cell| {
                let mut slot = cell.borrow_mut();
                if slot.as_ref().is_some_and(|(id, _)| *id == session_id) {
                    slot.take();
                    true
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    /// Run `f` with `session` bound to the current thread, restoring the
    /// previous binding (if any) afterwards — including on unwind.
    ///
    /// This is the safe, re-entrant way to say "this work happens in that
    /// session": every [`Self::global`] call made by `f`, however deep in the
    /// model/kernel stack, resolves to `session`.
    pub fn with_session<R>(session: &Arc<Self>, f: impl FnOnce() -> R) -> R {
        /// Restores the previous binding when the scope ends, however it ends.
        struct Restore {
            previous: Option<Arc<MetalGraph>>,
            bound: u64,
            /// `true` when this scope actually changed the binding; a re-bind
            /// of the already-bound session must not clear it on the way out.
            changed: bool,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                if !self.changed {
                    return;
                }
                if MetalGraph::unbind_current_if(self.bound) || self.previous.is_some() {
                    MetalGraph::restore_current(self.previous.take());
                }
            }
        }

        let already = Self::current_session_id() == Some(session.session_id);
        let previous = Self::bind_current(session);
        let _restore = Restore {
            previous,
            bound: session.session_id,
            changed: !already,
        };
        f()
    }

    /// Bind a **fresh** session to this thread for as long as the returned
    /// guard lives.
    ///
    /// The RAII form of [`Self::with_session`], for callers whose work is not
    /// shaped like a closure — an integration test whose whole body should run
    /// in its own session, for instance:
    ///
    /// ```no_run
    /// # use oxibonsai_kernels::MetalGraph;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let _session = MetalGraph::bind_new_session()?;
    /// // every MetalGraph::global() below this line — including the ones made
    /// // deep inside the model and kernel stack — resolves to that session
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Dropping the guard restores whatever binding was in place before,
    /// exactly as leaving a [`Self::with_session`] scope does.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::new_session`]'s device/compilation errors.
    pub fn bind_new_session() -> Result<SessionScope, MetalGraphError> {
        let session = Self::new_session()?;
        Ok(Self::bind_scope(session))
    }

    /// Bind an existing session to this thread until the guard drops.
    ///
    /// See [`Self::bind_new_session`]; this is the same thing for a session
    /// the caller already owns (an engine replica's, say).
    pub fn bind_scope(session: Arc<Self>) -> SessionScope {
        let already = Self::current_session_id() == Some(session.session_id);
        let previous = Self::bind_current(&session);
        SessionScope {
            bound: session.session_id,
            session,
            previous,
            changed: !already,
        }
    }

    /// Number of sessions currently alive on the process-shared device.
    ///
    /// Zero before the device is opened; isolated graphs built by
    /// [`Self::new`] are counted on their own device, not here.
    #[must_use]
    pub fn live_session_count() -> usize {
        let Some(mutex) = GLOBAL_METAL_DEVICE.get() else {
            return 0;
        };
        let Ok(guard) = mutex.lock() else {
            return 0;
        };
        guard
            .as_ref()
            .map_or(0, |d| d.live_sessions.load(Ordering::Relaxed))
    }

    /// Ceiling on concurrently useful Metal sessions on this host.
    ///
    /// This is the number `engine_pool::resolve_pool_sizing` clamps a GPU
    /// pool to. It is a *memory* bound, not a queue bound: each session's
    /// device KV cache is 604 MB for the 8B at `ctx = 4096`, so the default is
    /// deliberately small ([`DEFAULT_MAX_SESSIONS`]). Override with the
    /// `OXIBONSAI_METAL_MAX_SESSIONS` environment variable.
    #[must_use]
    pub fn max_sessions() -> usize {
        max_sessions_from(std::env::var(MAX_SESSIONS_ENV).ok().as_deref())
    }

    /// The device shared by every session in this process.
    ///
    /// # Errors
    ///
    /// Propagates [`MetalDevice::global`]'s device/compilation errors.
    pub fn shared_device() -> Result<Arc<MetalDevice>, MetalGraphError> {
        MetalDevice::global()
    }
}

/// RAII binding of a [`MetalGraph`] session to the current thread.
///
/// Created by [`MetalGraph::bind_new_session`] / [`MetalGraph::bind_scope`].
/// While it lives, [`MetalGraph::global`] on this thread returns the bound
/// session; when it drops, the previous binding (if any) is restored.
pub struct SessionScope {
    /// The session kept alive for the duration of the scope.
    session: Arc<MetalGraph>,
    /// Its id, so the release only clears *this* binding.
    bound: u64,
    /// The binding to restore on the way out.
    previous: Option<Arc<MetalGraph>>,
    /// `false` when this scope re-bound the already-bound session, in which
    /// case it must not unbind on exit.
    changed: bool,
}

impl SessionScope {
    /// The session this scope bound.
    #[must_use]
    pub fn session(&self) -> &Arc<MetalGraph> {
        &self.session
    }

    /// The bound session's id.
    #[must_use]
    pub fn session_id(&self) -> u64 {
        self.bound
    }
}

impl Drop for SessionScope {
    fn drop(&mut self) {
        if !self.changed {
            return;
        }
        if MetalGraph::unbind_current_if(self.bound) || self.previous.is_some() {
            MetalGraph::restore_current(self.previous.take());
        }
    }
}

/// Pure half of [`MetalGraph::max_sessions`] (unit-testable without a GPU).
///
/// An absent, empty, non-numeric or zero value all mean "use the default".
pub(crate) fn max_sessions_from(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_MAX_SESSIONS)
}

impl Drop for MetalGraph {
    fn drop(&mut self) {
        self.shared.live_sessions.fetch_sub(1, Ordering::Relaxed);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Weight cache — delegated to the shared device
// ═══════════════════════════════════════════════════════════════════════════

impl MetalGraph {
    /// Cache-aware upload keyed by the composite [`WeightKey`].
    ///
    /// Delegates to the shared device, so every session of a model sees the
    /// same GPU buffer for the same key: N replicas, 1x weights.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::WeightKindMismatch`] when the slot is occupied by a
    /// different [`WeightKind`]; otherwise whatever `upload` returns.
    pub fn get_or_upload_keyed(
        &self,
        key: WeightKey,
        upload: impl FnOnce() -> Result<MetalWeightHandle, MetalGraphError>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.shared.get_or_upload_keyed(key, upload)
    }

    /// Get a cached `MetalWeightHandle` or upload raw bytes and cache it.
    ///
    /// `key` is a legacy bare slot id (typically the `GpuWeightHandle`'s `u64`
    /// ID or an mmap address); it is keyed under `LEGACY_MODEL_EPOCH` and
    /// [`WeightKind::RawF32`]. Model code that owns an epoch should build a
    /// [`WeightKey`] and call [`Self::get_or_upload_keyed`] instead, so
    /// [`Self::release_model`] can free the buffer.
    ///
    /// # Errors
    ///
    /// Propagates the upload's allocation failure, or a kind mismatch on the
    /// slot.
    pub fn get_or_upload_weight(
        &self,
        key: u64,
        raw_bytes: &[u8],
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(WeightKey::legacy(WeightKind::RawF32, key), || {
            self.upload_weight(raw_bytes)
        })
    }

    /// Like [`Self::get_or_upload_weight`], but takes a closure producing the
    /// bytes, so a cache hit never materialises them.
    ///
    /// # Errors
    ///
    /// Propagates the upload's allocation failure, or a kind mismatch on the
    /// slot.
    pub fn get_or_upload_weight_lazy(
        &self,
        key: u64,
        data_fn: impl FnOnce() -> Vec<u8>,
    ) -> Result<Arc<MetalWeightHandle>, MetalGraphError> {
        self.get_or_upload_keyed(WeightKey::legacy(WeightKind::RawF32, key), || {
            self.upload_weight(&data_fn())
        })
    }

    /// Evict one cached weight by its composite key.
    ///
    /// Dropping the cached [`Arc`] frees the GPU buffer once no other handle
    /// is outstanding, and the freed bytes leave [`Self::bytes_uploaded`]. A
    /// key that is not present is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::ExecutionFailed`] if the cache lock is
    /// poisoned.
    pub fn evict_weight(&self, key: WeightKey) -> Result<(), MetalGraphError> {
        let freed = self.shared.lock_weight_cache()?.remove(key);
        self.shared
            .bytes_uploaded
            .fetch_sub(freed, Ordering::Relaxed);
        Ok(())
    }

    /// Evict a previously-uploaded raw-f32 weight from the cache by legacy key.
    ///
    /// Called after each GEMM when weights are **non-resident** (the host
    /// dequant buffer is freed after use so its `as_ptr()` key is unstable —
    /// the allocator will recycle the address, causing a stale cache hit on
    /// the next call). Mirrors the CUDA `evict_f32_weight` on `CudaGraph`.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::ExecutionFailed`] if the cache lock is
    /// poisoned.
    pub fn evict_f32_weight(&self, key: u64) -> Result<(), MetalGraphError> {
        self.evict_weight(WeightKey::legacy(WeightKind::RawF32, key))
    }

    /// Drop every cached weight belonging to `model_epoch` and return how many
    /// buffers were released.
    ///
    /// Call this when a model is unloaded (`BonsaiModel::Drop`): without it
    /// the shared cache keeps every buffer of every model ever loaded, which
    /// for the 27B is ~7.2 GB per load. Buffers still referenced elsewhere —
    /// by a *sibling session still decoding that model* — stay alive until
    /// that last `Arc` drops; the cache stops holding them either way and
    /// [`Self::bytes_uploaded`] falls by their size.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::ExecutionFailed`] if the cache lock is
    /// poisoned.
    pub fn release_model(&self, model_epoch: u64) -> Result<usize, MetalGraphError> {
        let (dropped, freed) = self.shared.lock_weight_cache()?.release_model(model_epoch);
        self.shared
            .bytes_uploaded
            .fetch_sub(freed, Ordering::Relaxed);
        Ok(dropped)
    }

    /// Total number of weight uploads (raw f32 **plus** quantized) on this
    /// session's device.
    ///
    /// Counts each distinct cache miss where a fresh GPU buffer was actually
    /// allocated. Useful for asserting that multi-image resident forwards
    /// amortize uploads (counter stays flat after the first forward) and that
    /// non-resident evict-after-GEMM forwards stay correct (no stale-cache
    /// collision, counter increments on every call as expected).
    #[must_use]
    pub fn weight_upload_count(&self) -> usize {
        self.norm_upload_count() + self.quant_upload_count()
    }

    /// Number of raw-f32 weight uploads (norms, f32/bf16 GEMM matrices).
    #[must_use]
    pub fn norm_upload_count(&self) -> usize {
        self.shared.norm_upload_count.load(Ordering::Relaxed)
    }

    /// Number of quantized weight uploads (Q1/TQ2/PQ2/PTQ1 SoA).
    #[must_use]
    pub fn quant_upload_count(&self) -> usize {
        self.shared.quant_upload_count.load(Ordering::Relaxed)
    }

    /// GPU weight bytes currently resident in the cache.
    ///
    /// Rises on every upload and falls on [`Self::evict_weight`] /
    /// [`Self::release_model`]; it is the observable that turns a duplicate
    /// model upload into a number.
    #[must_use]
    pub fn bytes_uploaded(&self) -> u64 {
        self.shared.bytes_uploaded.load(Ordering::Relaxed)
    }

    /// Number of weight buffers currently cached.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::ExecutionFailed`] if the cache lock is
    /// poisoned.
    pub fn cached_weight_count(&self) -> Result<usize, MetalGraphError> {
        Ok(self.shared.lock_weight_cache()?.len())
    }

    /// Resident weight bytes as accounted by the cache itself, under its lock.
    ///
    /// The authoritative figure; [`Self::bytes_uploaded`] is its lock-free
    /// mirror and the two must always agree (asserted in the cache tests).
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::ExecutionFailed`] if the cache lock is
    /// poisoned.
    pub fn resident_weight_bytes(&self) -> Result<u64, MetalGraphError> {
        Ok(self.shared.lock_weight_cache()?.resident_bytes())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Device KV cache lifecycle
// ═══════════════════════════════════════════════════════════════════════════

impl MetalGraph {
    /// Release this session's device-resident KV cache buffers (`M-Missed-3`).
    ///
    /// The fused full-layer decode/prefill paths lazily allocate their KV
    /// cache (`acquire_kv_cache` in `metal_full_layer`) and never free it on
    /// their own — it lives for the life of the session, sized for the largest
    /// sequence that session has decoded. `BonsaiModel::reset()` calls this so
    /// a model reset actually releases that memory instead of merely clearing
    /// the host-side `KvCache` and leaving the device copy resident.
    ///
    /// Idempotent: clearing an already-empty cache is a no-op. The next
    /// fused-GPU forward call re-allocates on demand, exactly as if this were
    /// the first call ever made.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::ExecutionFailed`] if the KV-cache lock is
    /// poisoned.
    pub fn clear_kv_cache(&self) -> Result<(), MetalGraphError> {
        *self
            .kv_cache
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("kv_cache lock poisoned".into()))? = None;
        Ok(())
    }

    /// Clear the device-resident KV cache of the session this thread would
    /// dispatch into, **without** constructing one if none exists yet
    /// (`M-Missed-3`).
    ///
    /// [`MetalGraph::global`] lazily opens the default Metal device and
    /// compiles every MSL pipeline on first use — expensive, and pointless for
    /// a model whose sequence never touched the fused GPU decode path. A plain
    /// host-side (or not-yet-run) model must not pay that cost just to
    /// discover it has nothing to clear, so this peeks at the thread binding
    /// and then at the process-default session, and only calls through to
    /// [`Self::clear_kv_cache`] when one is already resident.
    ///
    /// Sibling sessions are deliberately **not** touched: another replica may
    /// be mid-sequence in its own KV cache, and freeing it from under that
    /// replica is exactly the cross-session trampling `MET-08` removes.
    ///
    /// # Errors
    ///
    /// Returns [`MetalGraphError::ExecutionFailed`] if a lock is poisoned.
    pub fn clear_global_kv_cache_if_present() -> Result<(), MetalGraphError> {
        if let Some(bound) = Self::current_session() {
            return bound.clear_kv_cache();
        }
        let Some(mutex) = DEFAULT_SESSION.get() else {
            return Ok(());
        };
        // Clone the `Arc` and drop the `DEFAULT_SESSION` lock before calling
        // into the session: `clear_kv_cache` takes its own `kv_cache` lock,
        // and nothing here needs to prove that no path ever acquires that lock
        // and then reaches back into `MetalGraph::global()`.
        let session = {
            let guard = mutex
                .lock()
                .map_err(|_| MetalGraphError::ExecutionFailed("MetalGraph lock poisoned".into()))?;
            guard.as_ref().map(Arc::clone)
        };
        match session {
            Some(session) => session.clear_kv_cache(),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_sessions_falls_back_to_the_default() {
        assert_eq!(max_sessions_from(None), DEFAULT_MAX_SESSIONS);
        assert_eq!(max_sessions_from(Some("")), DEFAULT_MAX_SESSIONS);
        assert_eq!(
            max_sessions_from(Some("not-a-number")),
            DEFAULT_MAX_SESSIONS
        );
        // Zero sessions would mean "no GPU pool at all"; treat it as unset.
        assert_eq!(max_sessions_from(Some("0")), DEFAULT_MAX_SESSIONS);
    }

    #[test]
    fn max_sessions_honours_an_explicit_override() {
        assert_eq!(max_sessions_from(Some("1")), 1);
        assert_eq!(max_sessions_from(Some("7")), 7);
        assert_eq!(max_sessions_from(Some("  3 ")), 3);
    }
}
