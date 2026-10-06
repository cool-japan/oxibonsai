//! Engine-replica pool for concurrent serving.
//!
//! The HTTP server historically wrapped a single [`InferenceEngine`] in a
//! `tokio::sync::Mutex`, which serialized every request: a long generation
//! blocked all others for its full duration. This module replaces that single
//! mutex with a *pool* of `N` engine replicas guarded by a semaphore, so up to
//! `N` requests can generate concurrently.
//!
//! ## Why a pool works (and is safe)
//!
//! - Weights are process-global and immutable: [`InferenceEngine::from_gguf_path`]
//!   leaks the mmap and parsed `GgufFile` to `'static`, so every replica borrows
//!   the *same* `&'static GgufFile` zero-copy. The immutable, dequantized
//!   `token_embd` table is likewise shared across replicas via one `Arc<[f32]>`
//!   (see [`build_pool_from_gguf`]). Only the per-replica `KvCache` and light
//!   wrappers are duplicated.
//! - On CPU tiers (Reference / AVX / NEON), `BonsaiModel::forward` mutates only
//!   `self.kv_cache` over a shared `&dyn OneBitKernel` on immutable weights, so
//!   distinct engine instances run fully parallel.
//! - On the Metal tier, each replica now owns a `MetalGraph` **session**
//!   (`MET-08`): its own command queue and its own device KV cache, prefill
//!   and full-layer buffers, over a process-shared device that still holds one
//!   copy of the weights. The lease binds its replica's session for as long as
//!   the replica is in use, so two replicas submit concurrently instead of
//!   serialising on one another's `MutexGuard`s. The pool size is therefore
//!   `min(requested, MetalGraph::max_sessions())` — a *memory* bound (604 MB
//!   of device KV per session for the 8B at `ctx = 4096`), not a correctness
//!   one. The weights really are held once, on both routes. Every replica of
//!   one `GgufFile` joins that mapping's GPU namespace — one model epoch,
//!   minted by the first replica and shared by all of them — so the ternary
//!   fused route, which keys all of its device buffers on the mapped tensors'
//!   addresses under that epoch, and the 1-bit route, which keys its norms,
//!   final norm and LM head on the namespace's slots, both bind the first
//!   replica's buffers. The 1-bit route's block weights come from the
//!   per-replica `upload_weights_to_gpu`, which `Scirs2Backend` deduplicates
//!   by content — a replica uploading byte-identical weights gets the
//!   resident handle back, so the `MetalGraph` slots keyed on those handles
//!   coincide too. Each replica's scirs2 registrations are released when it
//!   drops (`MET-M1`) and a buffer is freed only with its last replica; the
//!   namespace's buffers likewise go with the mapping's last replica.
//! - On the CUDA tier the process-global `CudaGraph` singleton is unchanged,
//!   so `N > 1` replicas would still corrupt each other's KV: the clamp to `1`
//!   stays there (see [`resolve_pool_sizing`]).
//! - A **hybrid** (`qwen35`) replica on the Metal hybrid runner is not the
//!   `MET-05` singleton either: every runner owns a `MetalGraph::new_session()`
//!   with its own command queue, sparse `f16` KV cache, recurrent state and
//!   activation scratch, and binds the weights as one no-copy buffer over the
//!   same file mapping (the pages are resident once, whichever runner reads
//!   them). Two runners in one process are therefore independent by
//!   construction — `two_metal_hybrid_replicas_interleave_bit_identically_to_solo_runs`
//!   decodes interleaved requests on two replicas bit-identically to solo
//!   runs — and the pool sizes a Metal-backed hybrid like any GPU tier:
//!   `min(requested, MetalGraph::max_sessions())`, one replica when nothing
//!   was requested. The bound is memory: each runner allocates its KV cache
//!   for the whole window at load (64 KiB per position for the 27B — 512 MiB
//!   at the default 8192), about 150 MiB of recurrent state, the widened
//!   gates and its scratch, and each replica's CPU model keeps its own
//!   recurrent state; the engine's per-replica window budget
//!   ([`crate::engine_hybrid_gpu::plan_hybrid_metal_window`]) counts one CPU
//!   model and one runner, so an operator asking for more replicas is asking
//!   for that much more. The pool binds no session of its own for a hybrid
//!   replica (the runner never dispatches through `MetalGraph::global()`), and
//!   replicas `2..N` are built on the executor replica `#1` resolved to, so a
//!   pool never mixes Metal and CPU hybrids.
//!
//! ## What replicas buy
//!
//! Replicas buy isolation and latency fairness, not GPU throughput. Measured
//! on an M3 with the real Ternary-Bonsai-1.7B and 8 concurrent greedy
//! requests, pools of 1 / 2 / 3 replicas served the batch at 52.2 / 54.2 /
//! 54.5 tok/s aggregate (a median +4.5 % / +5.5 %; on a loaded host a paired
//! per-round median of 1.04×), with 457 MB of resident weights for all three
//! (the weights are shared) and byte-identical outputs; a Bonsai-8B Q1 pool
//! adds no weight memory per extra replica either. A single stream already
//! keeps the GPU busy for the whole token, so more sessions overlap
//! host-side work rather than multiplying throughput — see
//! `crates/oxibonsai-kernels/src/gpu_backend/metal_graph/session.rs` for the
//! derivation.
//!
//! ## Back-compatibility
//!
//! The default path wraps exactly one engine in a 1-element pool whose
//! [`EngineLease`] calls the identical `generate*` methods on the identical
//! engine. Single-request behavior — including RNG progression — is therefore
//! byte-identical to the previous single-mutex design.
//!
//! ## Lock discipline
//!
//! `idle` is a *synchronous* [`std::sync::Mutex`] held only for the duration of
//! a `pop`/`push` — never across an `.await`. Async waiting happens purely on
//! the [`tokio::sync::Semaphore`], whose permit count equals the pool size.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::engine::InferenceEngine;

/// Per-replica GPU session binding (`MET-08`).
///
/// On the Metal tier a replica owns a `MetalGraph` session — its own command
/// queue and its own device KV cache / prefill / full-layer buffers — and the
/// lease binds it to whichever thread is running that replica, so every
/// `MetalGraph::global()` call made deep inside the model and kernel stack
/// resolves to *this replica's* session.
///
/// Off the Metal tier (CPU tiers, CUDA, non-macOS, `metal` feature off) the
/// type is an empty placeholder whose `bind`/`release` compile away, and
/// `MetalGraph::global()` keeps returning the process-default session exactly
/// as it did before the split.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Clone, Default)]
struct GpuSession(Option<Arc<oxibonsai_kernels::MetalGraph>>);

#[cfg(all(feature = "metal", target_os = "macos"))]
impl GpuSession {
    /// No session: a replica that issues no work through
    /// `MetalGraph::global()` (a CPU tier, or a hybrid replica whose Metal
    /// runner owns a session of its own).
    fn none() -> Self {
        Self(None)
    }

    /// Create a session for a replica running on `tier`.
    ///
    /// Only the GPU tier gets one, and that is sufficient rather than merely
    /// cheap: every Metal path a replica can take is itself tier-gated —
    /// `InferenceEngine::uses_fused_gpu_decode` ANDs `fused_gpu_decode` with
    /// `kernel_is_gpu_tier`, and the per-layer/GEMV Metal entry points are
    /// only reached through GPU weight handles a non-GPU tier never uploads.
    /// A CPU-tier replica therefore issues no GPU work to share, while
    /// creating a session for it would open the Metal device and compile the
    /// MSL library for a pool that never dispatches.
    ///
    /// Failure to create one is not fatal — the replica falls back to the
    /// process-default session, i.e. the pre-`MET-08` behaviour.
    fn for_tier(tier: oxibonsai_kernels::KernelTier) -> Self {
        if tier != oxibonsai_kernels::KernelTier::Gpu {
            return Self(None);
        }
        match oxibonsai_kernels::MetalGraph::new_session() {
            Ok(session) => Self(Some(session)),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "could not create a per-replica Metal session; using the shared one"
                );
                Self(None)
            }
        }
    }

    /// Bind this replica's session to the calling thread.
    ///
    /// Called from `Deref`/`DerefMut`, i.e. immediately before every use of
    /// the engine, so the binding lands on the thread that actually runs the
    /// generation even when the lease was created on another one (the server
    /// acquires on a tokio worker and then runs inside `spawn_blocking`).
    /// Re-binding the already-bound session costs one `u64` compare.
    fn bind(&self) {
        if let Some(session) = &self.0 {
            oxibonsai_kernels::MetalGraph::bind_current(session);
        }
    }

    /// Release this replica's binding, if this thread still holds it.
    fn release(&self) {
        if let Some(session) = &self.0 {
            oxibonsai_kernels::MetalGraph::unbind_current_if(session.session_id());
        }
    }

    /// This replica's session id, for diagnostics and tests.
    fn id(&self) -> Option<u64> {
        self.0.as_ref().map(|s| s.session_id())
    }
}

/// Placeholder [`GpuSession`] for builds without the Metal backend.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[derive(Clone, Default)]
struct GpuSession;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
impl GpuSession {
    /// No GPU session exists off the Metal tier.
    fn none() -> Self {
        Self
    }

    /// No GPU session exists off the Metal tier.
    fn for_tier(_tier: oxibonsai_kernels::KernelTier) -> Self {
        Self
    }

    /// No-op: nothing to bind.
    fn bind(&self) {}

    /// No-op: nothing to release.
    fn release(&self) {}

    /// No session, no id.
    fn id(&self) -> Option<u64> {
        None
    }
}

/// One pool slot: an engine replica plus the GPU session it dispatches in.
struct Replica {
    /// The replica itself.
    engine: InferenceEngine<'static>,
    /// Its GPU session (`MET-08`); empty off the Metal tier.
    session: GpuSession,
}

impl Replica {
    /// Wrap an engine, giving it a session sized for its own kernel tier.
    ///
    /// A hybrid engine gets none: a Metal-backed one decodes on its runner,
    /// which already owns a Metal session of its own (and never dispatches
    /// through `MetalGraph::global()`), and a CPU one issues no GPU work.
    fn new(engine: InferenceEngine<'static>) -> Self {
        let session = if engine.is_hybrid() {
            GpuSession::none()
        } else {
            GpuSession::for_tier(engine.kernel_tier())
        };
        Self { engine, session }
    }
}

/// Errors that can occur while acquiring an engine from the pool.
///
/// These map to HTTP `503 Service Unavailable` at the call site — they all
/// represent a transient inability to serve, never a client error.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// The pool's semaphore was closed (the pool is shutting down).
    #[error("engine pool is closed")]
    Closed,
    /// A permit was acquired but no idle engine was available.
    ///
    /// This cannot happen under correct operation — permits and idle engines
    /// are kept in lock-step by construction — but is surfaced as an error
    /// rather than a panic to uphold the crate's no-panic policy.
    #[error("engine pool is unexpectedly empty")]
    Empty,
    /// The `idle` mutex was poisoned by a panic in another thread.
    #[error("engine pool mutex was poisoned")]
    Poisoned,
}

/// A pool of [`InferenceEngine`] replicas guarded by a semaphore.
///
/// Construct with [`EnginePool::new`]; acquire an engine with
/// [`EnginePool::acquire`]. The returned [`EngineLease`] derefs to the engine
/// and returns it to the pool on drop.
pub struct EnginePool {
    /// Idle (available) engines. Guarded by a *synchronous* mutex held only for
    /// `pop`/`push`. The number of engines ever simultaneously checked out plus
    /// the length of this vector always equals [`EnginePool::size`].
    idle: Mutex<Vec<Replica>>,
    /// Async gate: exactly `size` permits. A permit is held for the lifetime of
    /// each outstanding [`EngineLease`] and released only after the engine has
    /// been returned to `idle`.
    sem: Arc<Semaphore>,
    /// Number of replicas in the pool (immutable after construction).
    size: usize,
}

impl EnginePool {
    /// Build a pool from a vector of engine replicas.
    ///
    /// The pool size is `engines.len()`, guaranteed to be at least `1` (an
    /// empty input yields a 1-permit pool with no engines, which would only
    /// ever return [`PoolError::Empty`]; callers must pass at least one
    /// engine). The semaphore is seeded with `size` permits.
    pub fn new(engines: Vec<InferenceEngine<'static>>) -> Arc<Self> {
        let size = engines.len().max(1);
        // `MET-08`: each replica gets its own Metal session on the GPU tier, so
        // the KV cache and scratch it locks across `commit()` are its own.
        let replicas: Vec<Replica> = engines.into_iter().map(Replica::new).collect();
        Arc::new(Self {
            idle: Mutex::new(replicas),
            sem: Arc::new(Semaphore::new(size)),
            size,
        })
    }

    /// Number of replicas in the pool.
    ///
    /// This is the *effective* capacity after any tier clamp — on the GPU
    /// tier it is always `1`, whatever was requested (see
    /// [`resolve_pool_size`]).
    pub fn size(&self) -> usize {
        self.size
    }

    /// Number of replicas currently idle (not leased out).
    ///
    /// A cheap gauge for `/admin/status` and the queue-depth metric; returns
    /// `0` if the idle mutex is poisoned rather than propagating an error,
    /// because a gauge must never fail a request.
    pub fn idle_count(&self) -> usize {
        self.idle.lock().map(|idle| idle.len()).unwrap_or(0)
    }

    /// The number of requests that may usefully be admitted at once.
    ///
    /// `perf-M1`: the server admits `--max-concurrent-requests` (default 32)
    /// while the pool may hold a *single* replica — on the GPU tier it always
    /// does, because decode funnels through a process-global singleton. The
    /// surplus requests do not fail fast; they queue invisibly behind one
    /// engine on `acquire()` and then time out, which measures as flat
    /// throughput plus a timing-out tail (1 request 1.40 s; 4 concurrent
    /// 1.44/2.76/4.12/5.39 s).
    ///
    /// An admission controller should derive its limit from this rather than
    /// from a standalone configuration default, and shed beyond it with
    /// `503` + `Retry-After` instead of accepting work it cannot start.
    /// `queue_depth_per_replica` is how many *waiting* requests per replica
    /// the operator is willing to hold: `0` admits only what can run
    /// immediately, `1` (a reasonable default) keeps one warm request behind
    /// each replica so a replica never idles between requests.
    ///
    /// Saturating arithmetic: a hostile `queue_depth_per_replica` cannot
    /// overflow the limit into a small number.
    pub fn admission_limit(&self, queue_depth_per_replica: usize) -> usize {
        self.size
            .saturating_mul(queue_depth_per_replica.saturating_add(1))
            .max(1)
    }

    /// Attach a shared [`crate::metrics::InferenceMetrics`] to every replica in the pool.
    ///
    /// This wires the per-engine telemetry (prefill / decode-token /
    /// tokens-per-second histograms recorded inside `generate*`) onto each
    /// replica, mirroring the single-engine `engine.set_metrics(..)` call the
    /// CLI `serve` handler performed before this pool existed. It must be called
    /// while the pool is *idle* (right after construction, before any lease is
    /// handed out), so all replicas are present in `idle`.
    ///
    /// Returns [`PoolError::Poisoned`] if the idle mutex was poisoned. The
    /// metrics `Arc` is cloned once per replica so they all share one instance.
    pub fn set_metrics_all(
        &self,
        metrics: &Arc<crate::metrics::InferenceMetrics>,
    ) -> Result<(), PoolError> {
        let mut idle = self.idle.lock().map_err(|_| PoolError::Poisoned)?;
        for replica in idle.iter_mut() {
            replica.engine.set_metrics(Arc::clone(metrics));
        }
        Ok(())
    }

    /// Set every replica's baseline sampler `min_p` (a GGUF-declared
    /// `general.sampling.min_p`): a post-build visitor rather than a
    /// builder parameter, so every existing pool-builder signature stays
    /// unchanged. Mirrors [`Self::set_metrics_all`] — same "call while the
    /// pool is idle" contract.
    ///
    /// A per-request `min_p` in an HTTP body overrides this baseline for
    /// that request only and restores it afterwards
    /// (`server/sampling_scope.rs`); this call sets what an *unspecified*
    /// request's `min_p` samples with.
    ///
    /// Returns [`PoolError::Poisoned`] if the idle mutex was poisoned.
    pub fn set_min_p_all(&self, min_p: f32) -> Result<(), PoolError> {
        let mut idle = self.idle.lock().map_err(|_| PoolError::Poisoned)?;
        for replica in idle.iter_mut() {
            replica.engine.set_min_p(min_p);
        }
        Ok(())
    }

    /// Acquire an engine from the pool, waiting asynchronously if all replicas
    /// are currently in use.
    ///
    /// Returns an [`EngineLease`] that derefs to the engine and returns it to
    /// the pool on drop. The acquired permit is held for the lease's lifetime.
    pub async fn acquire(self: &Arc<Self>) -> Result<EngineLease, PoolError> {
        // Wait for a free slot. The permit count mirrors the idle count, so a
        // granted permit guarantees an idle engine is (about to be) available.
        let permit = Arc::clone(&self.sem)
            .acquire_owned()
            .await
            .map_err(|_| PoolError::Closed)?;

        // Pop an idle replica. The lock is held only for this `pop`.
        let replica = {
            let mut idle = self.idle.lock().map_err(|_| PoolError::Poisoned)?;
            idle.pop().ok_or(PoolError::Empty)?
        };

        Ok(EngineLease {
            engine: ManuallyDrop::new(replica.engine),
            session: replica.session,
            pool: Arc::clone(self),
            _permit: permit,
        })
    }
}

impl std::fmt::Debug for EnginePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let idle_len = self.idle.lock().map(|g| g.len()).ok();
        f.debug_struct("EnginePool")
            .field("size", &self.size)
            .field("available_permits", &self.sem.available_permits())
            .field("idle_len", &idle_len)
            .finish()
    }
}

/// An exclusive lease on one engine from an [`EnginePool`].
///
/// Derefs (mutably) to the borrowed [`InferenceEngine`], so callers invoke the
/// usual `generate*` methods directly. On drop, the engine is returned to the
/// pool and the semaphore permit is released — in that order, so a waiter is
/// guaranteed to find an idle engine the instant its permit is granted.
///
/// The engine is stored in [`ManuallyDrop`] so [`Drop`] can move it back into
/// the pool without an `Option`/`unwrap` dance: this is the whole reason the
/// guard is panic-free.
pub struct EngineLease {
    engine: ManuallyDrop<InferenceEngine<'static>>,
    /// The replica's GPU session (`MET-08`), bound to the calling thread on
    /// every `Deref`/`DerefMut` and released in `Drop`.
    session: GpuSession,
    pool: Arc<EnginePool>,
    // Field order matters: `_permit` is declared last so it is dropped *after*
    // the explicit `Drop::drop` body below has returned the engine to `idle`.
    // (Rust drops a struct's fields in declaration order after running its
    // `Drop::drop`.) This guarantees the slot is only freed once the engine is
    // back in the pool.
    _permit: OwnedSemaphorePermit,
}

impl EngineLease {
    /// The id of the Metal session this replica dispatches in (`MET-08`).
    ///
    /// `None` off the Metal tier, where there is no per-replica session and
    /// GPU work (if any) goes to the process-default one. Two leases held at
    /// the same time from a pool sized `> 1` on the GPU tier always report
    /// *different* ids — that is what makes their submissions overlap instead
    /// of serialising on a shared KV cache — and the id is stable across
    /// leases of the same replica.
    pub fn gpu_session_id(&self) -> Option<u64> {
        self.session.id()
    }
}

impl Deref for EngineLease {
    type Target = InferenceEngine<'static>;

    fn deref(&self) -> &Self::Target {
        // `MET-08`: bind here rather than in `acquire`, because the thread
        // that acquires a lease is not always the thread that uses it — the
        // server acquires on a tokio worker and then moves the lease into
        // `spawn_blocking`. Every use of the engine goes through
        // `Deref`/`DerefMut`, so the binding is always established on the
        // thread that is about to dispatch, and re-binding the session that
        // is already bound costs one `u64` compare.
        self.session.bind();
        // Total: `engine` is always populated until `Drop` takes it exactly
        // once, and the lease is never accessed after drop.
        &self.engine
    }
}

impl DerefMut for EngineLease {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.bind();
        // Total: see `deref`.
        &mut self.engine
    }
}

impl Drop for EngineLease {
    fn drop(&mut self) {
        // SAFETY: `ManuallyDrop::take` is called exactly once, here in `Drop`;
        // `self.engine` is never accessed afterwards (the struct is being
        // destroyed), so no double-take or use-after-take can occur.
        let mut engine = unsafe { ManuallyDrop::take(&mut self.engine) };

        // `MET-08`: drop this replica's GPU-session binding from the current
        // thread, so a later, unleased dispatch on this thread falls back to
        // the process-default session instead of quietly sharing a session
        // with whichever replica ran here last.
        //
        // The binding is established in `Deref`, on the thread that *uses* the
        // lease, and every caller in this tree uses and drops a lease on one
        // thread (`server::blocking::run_blocking_generation` moves it into the
        // blocking closure and drops it there). A lease used on one thread and
        // dropped on another would leave the using thread bound until its next
        // lease rebinds it: still correct — a session is only ever used by the
        // replica that owns it — but no longer replica-affine.
        self.session.release();

        // `SV-09`: a cancellation token is scoped to the request that armed
        // it. Drop it here, on the replica's way back into the pool, so a
        // request that was cancelled (or timed out) cannot leave a latched
        // flag that instantly cancels whichever request is served next by
        // this replica. This is also why `InferenceEngine::reset` does not
        // clear it: `run_blocking_generation` resets *before* the closure
        // runs, so clearing there would disarm the request's own token.
        engine.clear_cancellation_token();

        // Return the engine to the pool. We deliberately do NOT run a heavy
        // reset here: every `generate*` entry point resets the model KV cache
        // at the start of the call (today's behavior), and the sampler's RNG
        // state is intentionally preserved, so resetting here would be both
        // redundant and a risk to byte-identical single-request behavior.
        match self.pool.idle.lock() {
            Ok(mut idle) => idle.push(Replica {
                engine,
                // A cheap `Arc` refcount bump: the replica keeps the same
                // session for its whole life, so the next lease of it binds
                // the same queue and the same device KV cache.
                session: self.session.clone(),
            }),
            Err(_poisoned) => {
                // The pool mutex is poisoned (a thread panicked while holding
                // it). Dropping `engine` here is the safe choice: we must not
                // panic in `Drop`, and pushing into a poisoned pool is not
                // meaningfully recoverable. The permit still releases below.
                tracing::error!("engine pool mutex poisoned on lease return; dropping replica");
                drop(engine);
            }
        }

        // `_permit` is dropped *after* this body returns (it is the last field
        // in declaration order), so the semaphore slot frees only once the
        // engine is back in `idle`.
    }
}

/// Default pool size on CPU tiers: the host's available parallelism, capped at
/// `4`, with a floor of `1`.
///
/// The cap keeps memory bounded (each replica adds its own KV cache; the
/// `token_embd` table is shared across replicas via one `Arc<[f32]>`) while
/// still allowing a handful of concurrent generations on typical multi-core
/// hosts.
pub fn default_cpu_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(1)
}

/// Resolve the effective pool size for a given kernel tier.
///
/// - On the GPU tier, the size is `min(requested, `[`gpu_max_replicas`]`)`,
///   defaulting to `1` when nothing was requested. Since `MET-08` a Metal
///   replica owns its own session (command queue + device KV cache), so
///   `N > 1` is *correct*; what bounds it is memory — 604 MB of device KV per
///   session for the 8B at `ctx = 4096`. A CUDA-only build keeps the hard
///   clamp to `1`, because `CudaGraph` is still a process-global singleton
///   with one shared KV cache.
/// - On CPU tiers, the size is `requested` (clamped to `>= 1`) or, if `None`,
///   [`default_cpu_pool_size`].
///
/// The GPU comparison is gated behind the GPU-enabling features (`metal` /
/// `native-cuda`) because `oxibonsai_kernels::KernelTier::Gpu` only exists
/// when one of them is compiled in; non-GPU builds always take the CPU branch.
pub fn resolve_pool_size(requested: Option<usize>, tier: oxibonsai_kernels::KernelTier) -> usize {
    resolve_pool_sizing(requested, tier).effective
}

/// How a pool was sized, including whether a tier clamp overrode the request.
///
/// `perf-M1`: the clamp is invisible in `resolve_pool_size`'s `usize`, so a
/// server cannot tell "the operator asked for 8 and got 8" from "the operator
/// asked for 8 and the GPU singleton forced 1" — and it is exactly that
/// difference an admission limit and `/admin/status` must report. Returned by
/// [`resolve_pool_sizing`] and carried alongside the pool by
/// [`build_pool_from_gguf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSizing {
    /// What the caller asked for (`None` = "use the default").
    pub requested: Option<usize>,
    /// What the pool will actually hold.
    pub effective: usize,
    /// Whether `effective` is below `requested` because of the GPU-tier
    /// clamp.
    pub clamped_by_gpu_tier: bool,
    /// The GPU-tier ceiling that produced `effective`, when the GPU arm was
    /// taken (`1` on a CUDA-only build, `MetalGraph::max_sessions()` with the
    /// Metal backend). `None` on the CPU arm.
    pub gpu_max: Option<usize>,
}

impl PoolSizing {
    /// One-line, operator-facing explanation of the effective size.
    pub fn reason(&self) -> String {
        if self.clamped_by_gpu_tier {
            format!(
                "pool size {} (requested {}, clamped to {} on the GPU tier: one Metal session per \
                 replica, each with its own device KV cache)",
                self.effective,
                self.requested.unwrap_or(self.effective),
                self.gpu_max.unwrap_or(self.effective)
            )
        } else {
            match self.requested {
                Some(r) => format!("pool size {} (requested {r})", self.effective),
                None => format!("pool size {} (host default)", self.effective),
            }
        }
    }
}

/// Resolve the effective pool size *and* why it came out that way.
///
/// The data behind [`resolve_pool_size`]; see [`PoolSizing`]. Pure and
/// unit-testable, with the same clamp semantics.
pub fn resolve_pool_sizing(
    requested: Option<usize>,
    tier: oxibonsai_kernels::KernelTier,
) -> PoolSizing {
    resolve_pool_sizing_with_gpu_max(requested, tier, gpu_max_replicas())
}

/// How many replicas may usefully share the GPU on this build and host.
///
/// Metal: `MetalGraph::max_sessions()` — the per-session device KV cache is
/// the bound, so it is small by default and tunable with
/// `OXIBONSAI_METAL_MAX_SESSIONS`. CUDA-only: `1`, because `CudaGraph` remains
/// a process-global singleton with one shared KV cache (`MET-08` split the
/// Metal graph only).
pub fn gpu_max_replicas() -> usize {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        oxibonsai_kernels::MetalGraph::max_sessions().max(1)
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        1
    }
}

/// The pure half of [`resolve_pool_sizing`]: the same clamp semantics with the
/// GPU ceiling supplied by the caller, so it is unit-testable without a GPU.
///
/// An unspecified `requested` stays at `1` on the GPU tier: sessions are
/// correct but not free (604 MB of device KV each for the 8B at `ctx = 4096`),
/// so growing the pool is an explicit operator choice and the ceiling only
/// ever caps what was actually asked for.
pub fn resolve_pool_sizing_with_gpu_max(
    requested: Option<usize>,
    tier: oxibonsai_kernels::KernelTier,
    gpu_max: usize,
) -> PoolSizing {
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    {
        if tier == oxibonsai_kernels::KernelTier::Gpu {
            let ceiling = gpu_max.max(1);
            let effective = requested.unwrap_or(1).max(1).min(ceiling);
            return PoolSizing {
                requested,
                effective,
                clamped_by_gpu_tier: requested.is_some_and(|r| r > effective),
                gpu_max: Some(ceiling),
            };
        }
    }
    // Silence the unused-variable lints on non-GPU builds where neither is
    // inspected.
    let _ = tier;
    let _ = gpu_max;
    PoolSizing {
        requested,
        effective: requested.unwrap_or_else(default_cpu_pool_size).max(1),
        clamped_by_gpu_tier: false,
        gpu_max: None,
    }
}

/// Build an [`EnginePool`] from a GGUF file, sizing it for the detected tier.
///
/// Replica `#1` is loaded via [`InferenceEngine::from_gguf_path`], which
/// memory-maps and parses the GGUF and leaks both to `'static`. Its kernel tier
/// is read to size the pool via [`resolve_pool_size`]; replicas `2..size` are
/// then built off the *same* leaked `&'static GgufFile` via
/// [`InferenceEngine::from_gguf_static_with_embd`], so no additional mmap or
/// weight copy occurs. Every replica is seeded identically, so the served
/// output is deterministic regardless of which replica handles a request.
///
/// ## Shared token embedding
///
/// The dequantized `token_embd` table (FP32 `vocab × hidden` — ~1.16 GiB for
/// the 1.7B) is immutable and load-once. Replica `#1` loads it into an
/// `Arc<[f32]>`; that single `Arc` is then *cloned* (a refcount bump, not a
/// data copy) into every replica `2..size`. The whole pool therefore holds
/// **one** embedding allocation regardless of `size`, rather than N duplicates,
/// and replicas `2..size` skip re-dequantizing it. Per-replica `KvCache`s stay
/// fully independent.
///
/// Returns the pool, the detected [`oxibonsai_kernels::KernelTier`], and the
/// effective size.
pub fn build_pool_from_gguf(
    path: impl AsRef<std::path::Path>,
    sampling_params: crate::sampling::SamplingParams,
    seed: u64,
    max_seq_len: usize,
    requested_size: Option<usize>,
) -> crate::error::RuntimeResult<(Arc<EnginePool>, oxibonsai_kernels::KernelTier, usize)> {
    let built = build_pool_from_gguf_parts(
        path,
        sampling_params,
        seed,
        max_seq_len,
        requested_size,
        crate::engine_seam::Backend::Auto,
    )?;
    Ok((built.pool, built.tier, built.size))
}

/// Everything [`build_pool_from_gguf_parts`] built, including the pieces a
/// server needs to construct *further* engines off the same weights without
/// a second mapping — e.g. the dedicated embedding engine of
/// `ModelEmbedder::from_static_gguf`.
pub struct PoolBuild {
    /// The replica pool.
    pub pool: Arc<EnginePool>,
    /// The kernel tier replica `#1` resolved to.
    pub tier: oxibonsai_kernels::KernelTier,
    /// The effective replica count.
    pub size: usize,
    /// The leaked, process-lifetime GGUF every replica borrows.
    pub gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static>,
    /// The shared token-embedding handle every replica was built with (empty
    /// for a quantized or hybrid embedding, which is read from the mapping).
    pub shared_token_embd: Arc<[f32]>,
    /// Whether the replicas hold a hybrid (`qwen35`) model.
    pub hybrid: bool,
}

impl std::fmt::Debug for PoolBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolBuild")
            .field("pool", &self.pool)
            .field("tier", &self.tier)
            .field("size", &self.size)
            .field("hybrid", &self.hybrid)
            .finish_non_exhaustive()
    }
}

/// [`build_pool_from_gguf`] on an explicit [`crate::engine_seam::Backend`],
/// returning every part of the build ([`PoolBuild`]).
///
/// A **hybrid** (`qwen35`) model decodes on the Metal hybrid runner or on a
/// CPU tier. On a CPU tier the generic sizing would default to `min(4,
/// cores)` replicas; each 27B replica carries its own KV cache, a ~157 MB
/// recurrent state and its chunk scratch, and a single replica already
/// saturates memory bandwidth, so an *unspecified* size is 1 for a hybrid
/// model on either executor (logged). An explicit `requested_size` is
/// honoured on the CPU and capped at `MetalGraph::max_sessions()` on the
/// Metal runner (see the module docs).
///
/// # Errors
///
/// Anything replica construction returns (see
/// [`InferenceEngine::from_gguf_path_leaked_with_backend`]).
pub fn build_pool_from_gguf_parts(
    path: impl AsRef<std::path::Path>,
    sampling_params: crate::sampling::SamplingParams,
    seed: u64,
    max_seq_len: usize,
    requested_size: Option<usize>,
    backend: crate::engine_seam::Backend,
) -> crate::error::RuntimeResult<PoolBuild> {
    // Replica #1 — this leaks the mmap + parsed GGUF to `'static`.
    let (first, gguf) = InferenceEngine::from_gguf_path_leaked_with_backend(
        path,
        sampling_params.clone(),
        seed,
        max_seq_len,
        backend,
    )?;
    finish_pool_from_first_replica(
        first,
        gguf,
        sampling_params,
        seed,
        max_seq_len,
        requested_size,
        backend,
    )
}

/// Shared by every builder in this module: given replica
/// `#1` and the already-`'static` `gguf` it was built from, resolve the
/// pool's effective size (logged), build replicas `2..size` off the same
/// GGUF and replica `#1`'s shared token-embedding table, and assemble the
/// [`PoolBuild`]. `build_pool_from_gguf_parts` and
/// `build_pool_from_static_gguf_with_rope` differ only in how replica `#1`
/// itself is obtained (from a path vs. an already-`'static` GGUF, with or
/// without a RoPE-scaling override in scope); everything after that is
/// byte-for-byte identical sizing/logging/replica-loop/assembly, so it lives
/// here once instead of twice.
fn finish_pool_from_first_replica(
    first: InferenceEngine<'static>,
    gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static>,
    sampling_params: crate::sampling::SamplingParams,
    seed: u64,
    max_seq_len: usize,
    requested_size: Option<usize>,
    backend: crate::engine_seam::Backend,
) -> crate::error::RuntimeResult<PoolBuild> {
    let tier = first.kernel_tier();
    let hybrid = first.is_hybrid();
    let sizing = if hybrid && requested_size.is_none() {
        tracing::info!(
            architecture = %first.architecture(),
            executor = ?first.hybrid_backend(),
            "hybrid model: defaulting the engine pool to 1 replica (each replica holds its own \
             KV cache and recurrent state — a Metal-backed one also its own Metal session and \
             whole-window device KV — and one replica already saturates memory bandwidth); pass \
             an explicit pool size to run more"
        );
        PoolSizing {
            requested: None,
            effective: 1,
            clamped_by_gpu_tier: false,
            gpu_max: None,
        }
    } else {
        resolve_pool_sizing(requested_size, tier)
    };
    let size = sizing.effective;

    // The effective size is what an admission controller must budget against
    // (`perf-M1`): admitting 32 requests into a 1-replica pool queues them
    // invisibly on `acquire()` until they time out. `EnginePool::size` and
    // `EnginePool::admission_limit` expose it after construction; this line
    // explains it once at startup.
    tracing::info!(
        tier = %tier,
        effective = size,
        clamped = sizing.clamped_by_gpu_tier,
        "{}",
        sizing.reason()
    );

    // Extract replica #1's shared token-embedding table (a cheap refcount-bumped
    // `Arc<[f32]>` handle, not a copy). Replicas 2..size clone this same `Arc`
    // instead of re-dequantizing their own ~1.16 GiB copy, so the whole pool
    // holds one embedding allocation total.
    let shared_token_embd = first.model_token_embd();

    // A hybrid pool is built on the executor replica #1 resolved to: under
    // `auto` a later replica must not quietly land on the CPU (or on Metal)
    // when #1 did not, or two requests could decode the same prompt on two
    // different executors.
    let replica_backend = match first.hybrid_backend() {
        Some(crate::engine_hybrid_gpu::HybridBackend::Metal) => crate::engine_seam::Backend::Metal,
        Some(crate::engine_hybrid_gpu::HybridBackend::Cpu) => crate::engine_seam::Backend::Cpu,
        None => backend,
    };

    let mut engines = Vec::with_capacity(size);
    engines.push(first);
    // Replicas 2..size reuse the already-`'static` GGUF (zero extra mmap/copy)
    // and the shared `Arc<[f32]>` token-embedding table (zero extra dequant/copy).
    for _ in 1..size {
        let mut replica = InferenceEngine::from_gguf_static_with_embd_and_backend(
            gguf,
            sampling_params.clone(),
            seed,
            max_seq_len,
            Arc::clone(&shared_token_embd),
            replica_backend,
        )?;
        // Every replica reports the knob the pool was asked for, whatever
        // executor it was pinned to.
        replica.backend = backend;
        engines.push(replica);
    }

    Ok(PoolBuild {
        pool: EnginePool::new(engines),
        tier,
        size,
        gguf,
        shared_token_embd,
        hybrid,
    })
}

/// [`build_pool_from_gguf_parts`] with a `--rope-scaling auto|on|off`
/// override (additive — the existing builders are
/// unchanged). Memory-maps and leaks the GGUF exactly as
/// [`InferenceEngine::from_gguf_path_leaked_with_backend`] does, then
/// defers to [`build_pool_from_static_gguf_with_rope`].
///
/// # Errors
///
/// [`crate::error::RuntimeError::FileNotFound`] for a missing file, the
/// refusals of [`crate::engine_seam::resolve_rope_scaling_at_load`], and
/// anything replica construction returns.
pub fn build_pool_from_gguf_parts_with_rope(
    path: impl AsRef<std::path::Path>,
    sampling_params: crate::sampling::SamplingParams,
    seed: u64,
    max_seq_len: usize,
    requested_size: Option<usize>,
    backend: crate::engine_seam::Backend,
    rope: oxibonsai_core::config::RopeScalingOverride,
) -> crate::error::RuntimeResult<PoolBuild> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        return Err(crate::error::RuntimeError::FileNotFound {
            path: path_ref.display().to_string(),
        });
    }
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(path_ref)?;
    let mmap: &'static memmap2::Mmap = Box::leak(Box::new(mmap));
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(mmap)?;
    let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> = Box::leak(Box::new(gguf));
    build_pool_from_static_gguf_with_rope(
        gguf,
        sampling_params,
        seed,
        max_seq_len,
        requested_size,
        backend,
        rope,
    )
}

/// Build an [`EnginePool`] off an already-`'static` GGUF — a leaked memory
/// map, or an in-memory image such as the CLI's `--ptq1-transcode` output —
/// with a `--rope-scaling` override (additive).
///
/// Same sizing policy and replica sharing as [`build_pool_from_gguf_parts`]
/// (hybrid → 1 replica unless asked; replicas `2..size` share replica
/// `#1`'s token-embedding handle). Every replica is constructed on the
/// calling thread inside one
/// [`RopeScalingOverrideScope`](oxibonsai_core::config::RopeScalingOverrideScope),
/// so all of them get the same effective RoPE table.
///
/// # Errors
///
/// The refusals of [`crate::engine_seam::resolve_rope_scaling_at_load`], and
/// anything replica construction returns.
pub fn build_pool_from_static_gguf_with_rope(
    gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static>,
    sampling_params: crate::sampling::SamplingParams,
    seed: u64,
    max_seq_len: usize,
    requested_size: Option<usize>,
    backend: crate::engine_seam::Backend,
    rope: oxibonsai_core::config::RopeScalingOverride,
) -> crate::error::RuntimeResult<PoolBuild> {
    crate::engine_seam::resolve_rope_scaling_at_load(gguf, rope)?;
    let _rope_scope = oxibonsai_core::config::RopeScalingOverrideScope::enter(rope);

    let first = InferenceEngine::from_gguf_static_with_embd_and_backend(
        gguf,
        sampling_params.clone(),
        seed,
        max_seq_len,
        Arc::from(Vec::new()),
        backend,
    )?;
    finish_pool_from_first_replica(
        first,
        gguf,
        sampling_params,
        seed,
        max_seq_len,
        requested_size,
        backend,
    )
}

#[cfg(test)]
#[path = "engine_pool_tests.rs"]
pub(crate) mod tests;
