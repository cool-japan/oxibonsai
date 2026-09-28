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
//!   one. The weights really are held once: the ternary fused route keys its
//!   device buffers on the mapped tensors' addresses (shared by every replica
//!   of one `GgufFile`), and the 1-bit route's per-replica
//!   `upload_weights_to_gpu` is deduplicated by content in `Scirs2Backend` —
//!   a replica uploading byte-identical weights gets the resident handle back,
//!   so the `MetalGraph` slots keyed on those handles coincide too. Each
//!   replica's registrations are released when it drops (`MET-M1`), and a
//!   buffer is freed only with its last replica.
//! - On the CUDA tier the process-global `CudaGraph` singleton is unchanged,
//!   so `N > 1` replicas would still corrupt each other's KV: the clamp to `1`
//!   stays there (see [`resolve_pool_sizing`]).
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
    fn new(engine: InferenceEngine<'static>) -> Self {
        let session = GpuSession::for_tier(engine.kernel_tier());
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
/// A **hybrid** (`qwen35`) model runs on a CPU tier, where the generic sizing
/// would default to `min(4, cores)` replicas. Each 27B replica carries its own
/// KV cache, a ~157 MB recurrent state and its chunk scratch, and a single
/// CPU replica already saturates memory bandwidth, so an *unspecified* size
/// is 1 for a hybrid model (logged); an explicit `requested_size` is honoured.
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

    let tier = first.kernel_tier();
    let hybrid = first.is_hybrid();
    let sizing = if hybrid && requested_size.is_none() {
        tracing::info!(
            architecture = %first.architecture(),
            "hybrid model: defaulting the engine pool to 1 replica (each replica holds its own \
             KV cache and recurrent state, and one CPU replica already saturates memory \
             bandwidth); pass an explicit pool size to run more"
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

    let mut engines = Vec::with_capacity(size);
    engines.push(first);
    // Replicas 2..size reuse the already-`'static` GGUF (zero extra mmap/copy)
    // and the shared `Arc<[f32]>` token-embedding table (zero extra dequant/copy).
    for _ in 1..size {
        let replica = InferenceEngine::from_gguf_static_with_embd_and_backend(
            gguf,
            sampling_params.clone(),
            seed,
            max_seq_len,
            Arc::clone(&shared_token_embd),
            backend,
        )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingParams;
    use oxibonsai_core::config::Qwen3Config;
    use std::time::Duration;

    fn tiny_engine() -> InferenceEngine<'static> {
        InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
    }

    // ── synthetic GGUF fixture (for the shared-embd pool test) ───────────────
    //
    // A minimal 2-layer, fully-quantized GGUF (h=128, inter=256, vocab=32) that
    // `BonsaiModel::from_gguf` can load on any CPU tier. Attention/FFN are
    // Q1_0_g128 and the LM head is Q1_0_g128; the token embedding is F32. This
    // is the same shape family used by the model crate's ternary integration
    // fixture, reproduced compactly here so the runtime pool builder can be
    // exercised end-to-end (it needs a real on-disk GGUF path).

    fn q1_0_g128_data(num_weights: usize) -> Vec<u8> {
        let num_blocks = num_weights / 128;
        let scale = half::f16::ONE.to_le_bytes();
        let mut data = Vec::with_capacity(num_blocks * 18);
        for _ in 0..num_blocks {
            data.extend_from_slice(&scale);
            data.extend_from_slice(&[0xFFu8; 16]);
        }
        data
    }

    fn build_tiny_gguf_bytes() -> Vec<u8> {
        use oxibonsai_core::gguf::writer::{
            GgufWriter, MetadataWriteValue, TensorEntry, TensorType,
        };

        let h: usize = 128;
        let inter: usize = 256;
        let num_layers: usize = 2;
        let nq: usize = 4;
        let nkv: usize = 2;
        let hd: usize = 32;
        let vocab: usize = 32;

        let mut w = GgufWriter::new();
        w.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen3".into()),
        );
        w.add_metadata("general.name", MetadataWriteValue::Str("TinyPool".into()));
        w.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
        w.add_metadata(
            "qwen3.block_count",
            MetadataWriteValue::U32(num_layers as u32),
        );
        w.add_metadata(
            "qwen3.attention.head_count",
            MetadataWriteValue::U32(nq as u32),
        );
        w.add_metadata(
            "qwen3.attention.head_count_kv",
            MetadataWriteValue::U32(nkv as u32),
        );
        w.add_metadata(
            "qwen3.feed_forward_length",
            MetadataWriteValue::U32(inter as u32),
        );
        w.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
        w.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
        w.add_metadata(
            "qwen3.attention.layer_norm_rms_epsilon",
            MetadataWriteValue::F32(1e-6),
        );
        w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

        let f32_ones = |n: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(n * 4);
            for _ in 0..n {
                v.extend_from_slice(&1.0_f32.to_le_bytes());
            }
            v
        };

        w.add_tensor(TensorEntry {
            name: "token_embd.weight".into(),
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::F32,
            data: f32_ones(vocab * h),
        });
        w.add_tensor(TensorEntry {
            name: "output_norm.weight".into(),
            shape: vec![h as u64],
            tensor_type: TensorType::F32,
            data: f32_ones(h),
        });
        w.add_tensor(TensorEntry {
            name: "output.weight".into(),
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_data(vocab * h),
        });

        for layer in 0..num_layers {
            let pfx = format!("blk.{layer}");
            for suffix in ["attn_norm.weight", "ffn_norm.weight"] {
                w.add_tensor(TensorEntry {
                    name: format!("{pfx}.{suffix}"),
                    shape: vec![h as u64],
                    tensor_type: TensorType::F32,
                    data: f32_ones(h),
                });
            }
            for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                w.add_tensor(TensorEntry {
                    name: format!("{pfx}.{suffix}"),
                    shape: vec![hd as u64],
                    tensor_type: TensorType::F32,
                    data: f32_ones(hd),
                });
            }
            let q1 = |name: &str, shape: Vec<u64>, n: usize| TensorEntry {
                name: name.to_string(),
                shape,
                tensor_type: TensorType::Q1_0G128,
                data: q1_0_g128_data(n),
            };
            w.add_tensor(q1(
                &format!("{pfx}.attn_q.weight"),
                vec![h as u64, (nq * hd) as u64],
                nq * hd * h,
            ));
            w.add_tensor(q1(
                &format!("{pfx}.attn_k.weight"),
                vec![h as u64, (nkv * hd) as u64],
                nkv * hd * h,
            ));
            w.add_tensor(q1(
                &format!("{pfx}.attn_v.weight"),
                vec![h as u64, (nkv * hd) as u64],
                nkv * hd * h,
            ));
            w.add_tensor(q1(
                &format!("{pfx}.attn_output.weight"),
                vec![(nq * hd) as u64, h as u64],
                h * nq * hd,
            ));
            w.add_tensor(q1(
                &format!("{pfx}.ffn_gate.weight"),
                vec![h as u64, inter as u64],
                inter * h,
            ));
            w.add_tensor(q1(
                &format!("{pfx}.ffn_up.weight"),
                vec![h as u64, inter as u64],
                inter * h,
            ));
            w.add_tensor(q1(
                &format!("{pfx}.ffn_down.weight"),
                vec![inter as u64, h as u64],
                h * inter,
            ));
        }

        w.to_bytes().expect("GgufWriter::to_bytes")
    }

    // ── resolve_pool_size ────────────────────────────────────────────────

    #[test]
    fn resolve_pool_size_explicit_cpu() {
        // On a CPU tier, an explicit request is honored (clamped to >= 1).
        let tier = oxibonsai_kernels::KernelTier::Reference;
        assert_eq!(resolve_pool_size(Some(8), tier), 8);
        assert_eq!(resolve_pool_size(Some(1), tier), 1);
        // Zero is clamped up to the floor of 1.
        assert_eq!(resolve_pool_size(Some(0), tier), 1);
    }

    #[test]
    fn resolve_pool_size_default_cpu() {
        let tier = oxibonsai_kernels::KernelTier::Reference;
        assert_eq!(resolve_pool_size(None, tier), default_cpu_pool_size());
    }

    #[test]
    fn default_cpu_pool_size_in_range() {
        let n = default_cpu_pool_size();
        assert!((1..=4).contains(&n), "expected 1..=4, got {n}");
    }

    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    #[test]
    fn resolve_pool_size_gpu_is_capped_by_the_session_ceiling() {
        // `MET-08` changed this rule deliberately: the GPU tier used to be a
        // hard clamp to 1 because decode funnelled through a process-global
        // graph with one KV cache. A Metal replica now owns its own session,
        // so the tier admits up to `gpu_max` replicas — the bound is device
        // memory (one KV cache per session), not correctness.
        let tier = oxibonsai_kernels::KernelTier::Gpu;
        assert_eq!(
            resolve_pool_sizing_with_gpu_max(Some(8), tier, 3).effective,
            3
        );
        assert_eq!(
            resolve_pool_sizing_with_gpu_max(Some(2), tier, 3).effective,
            2
        );
        assert_eq!(
            resolve_pool_sizing_with_gpu_max(Some(1), tier, 3).effective,
            1
        );
        // An unspecified request stays conservative: one replica.
        assert_eq!(resolve_pool_sizing_with_gpu_max(None, tier, 3).effective, 1);
        // A ceiling of 1 (a CUDA-only build) reproduces the old hard clamp.
        assert_eq!(
            resolve_pool_sizing_with_gpu_max(Some(8), tier, 1).effective,
            1
        );
        // A zero/absurd ceiling can never yield an empty pool.
        assert_eq!(
            resolve_pool_sizing_with_gpu_max(Some(8), tier, 0).effective,
            1
        );
        // The live wiring agrees with the pure function.
        assert_eq!(
            resolve_pool_size(Some(8), tier),
            resolve_pool_sizing_with_gpu_max(Some(8), tier, gpu_max_replicas()).effective
        );
    }

    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    #[test]
    fn gpu_max_replicas_is_at_least_one() {
        // A pool of zero replicas would deadlock every request on `acquire`.
        assert!(gpu_max_replicas() >= 1);
    }

    // ── lease / pool mechanics ───────────────────────────────────────────

    #[tokio::test]
    async fn pool_size_reflects_input() {
        let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
        assert_eq!(pool.size(), 2);
    }

    #[tokio::test]
    async fn acquire_blocks_when_exhausted_then_resumes_on_drop() {
        let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);

        // Take both engines.
        let lease_a = pool.acquire().await.expect("acquire a");
        let lease_b = pool.acquire().await.expect("acquire b");

        // Idle is now empty and no permits remain.
        assert_eq!(pool.sem.available_permits(), 0);
        {
            let idle = pool.idle.lock().expect("lock idle");
            assert!(idle.is_empty(), "idle should be empty with 2/2 checked out");
        }

        // A third acquire must NOT resolve while both leases are held.
        let pending = pool.acquire();
        let timed_out = tokio::time::timeout(Duration::from_millis(150), pending).await;
        assert!(
            timed_out.is_err(),
            "third acquire resolved while pool was exhausted"
        );

        // Returning one engine must let a waiting acquire proceed.
        drop(lease_a);
        let lease_c = tokio::time::timeout(Duration::from_millis(500), pool.acquire())
            .await
            .expect("acquire should resolve after a lease is dropped")
            .expect("acquire c");

        // Drop the rest; the pool returns to full availability.
        drop(lease_b);
        drop(lease_c);
        assert_eq!(pool.sem.available_permits(), 2);
        {
            let idle = pool.idle.lock().expect("lock idle");
            assert_eq!(idle.len(), 2, "all engines should be back in the pool");
        }
    }

    // ── single-request golden: byte-identical behavior ───────────────────

    #[tokio::test]
    async fn single_element_pool_is_byte_identical_to_direct_engine() {
        // The hard invariant: a 1-element pool that acquires and calls
        // `generate_with_params` must produce the EXACT same token vector as a
        // fresh engine with the same config/seed/params calling the same
        // method directly.
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let seed = 42u64;
        let prompt: Vec<u32> = vec![151644, 872, 9707, 11];
        let max_tokens = 8usize;

        // Direct engine baseline.
        let mut direct = InferenceEngine::new(config.clone(), params.clone(), seed);
        let direct_out = direct
            .generate_with_params(&prompt, max_tokens, &params)
            .expect("direct generate");

        // 1-element pool.
        let pool = EnginePool::new(vec![InferenceEngine::new(
            config.clone(),
            params.clone(),
            seed,
        )]);
        let mut lease = pool.acquire().await.expect("acquire");
        let pool_out = lease
            .generate_with_params(&prompt, max_tokens, &params)
            .expect("pool generate");

        assert_eq!(
            direct_out, pool_out,
            "1-element pool output diverged from direct engine — byte-identity broken"
        );
    }

    // ── concurrent isolation: no KV / RNG cross-talk ─────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_leases_match_isolated_baselines() {
        // NOTE on test strength: `Qwen3Config::tiny_test()` builds a model with
        // zero-initialized weights and no transformer blocks, so its forward
        // pass is prompt-INDEPENDENT (every prompt yields the same logits ->
        // same greedy token). That makes "distinct prompts => distinct outputs"
        // impossible to assert here. What we CAN — and do — assert is the
        // structural isolation guarantee: each concurrently-leased engine
        // produces output bit-identical to the SAME prompt run alone on a fresh
        // single engine. If concurrent leases shared/corrupted KV or RNG state,
        // the concurrent outputs would diverge from their isolated baselines.
        //
        // A richer cross-talk test (distinct prompts => distinct outputs)
        // requires non-degenerate weights; that variant is best built behind
        // `#[cfg(all(feature = "metal", target_os = "macos"))]` using the
        // synthetic ternary GGUF fixture from
        // `crates/oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs`
        // with a couple of `KernelTier::Reference` replicas. See task notes.
        use std::sync::Arc as StdArc;

        let config = Qwen3Config::tiny_test();
        // GREEDY params (temperature 0) make outputs deterministic and remove
        // any dependence on RNG ordering, isolating the KV-cache question.
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let seed = 42u64;
        let max_tokens = 6usize;

        let prompts: Vec<Vec<u32>> = vec![
            vec![151644, 872],
            vec![151644, 9707, 11, 1879],
            vec![151644, 1986, 374, 264, 1273],
            vec![151644, 264],
        ];

        // Isolated baselines: each prompt alone on a fresh single engine.
        let mut baselines = Vec::with_capacity(prompts.len());
        for p in &prompts {
            let mut e = InferenceEngine::new(config.clone(), params.clone(), seed);
            let out = e
                .generate_with_params(p, max_tokens, &params)
                .expect("baseline generate");
            baselines.push(out);
        }

        // Pool with one replica per prompt so all run truly concurrently.
        let engines: Vec<InferenceEngine<'static>> = (0..prompts.len())
            .map(|_| InferenceEngine::new(config.clone(), params.clone(), seed))
            .collect();
        let pool = EnginePool::new(engines);

        let params = StdArc::new(params);
        let mut handles = Vec::with_capacity(prompts.len());
        for p in prompts.clone() {
            let pool = StdArc::clone(&pool);
            let params = StdArc::clone(&params);
            handles.push(tokio::spawn(async move {
                let mut lease = pool.acquire().await.expect("acquire");
                lease
                    .generate_with_params(&p, max_tokens, &params)
                    .expect("concurrent generate")
            }));
        }

        for (i, h) in handles.into_iter().enumerate() {
            let got = h.await.expect("task join");
            assert_eq!(
                got, baselines[i],
                "concurrent task {i} diverged from its isolated baseline — KV/RNG cross-talk"
            );
        }
    }

    // ── shared token-embedding Arc across pool replicas ──────────────────────

    #[tokio::test]
    async fn pool_replicas_share_one_token_embd_allocation() {
        // The end-to-end Part-B proof: `build_pool_from_gguf` must build all
        // replicas sharing ONE `Arc<[f32]>` token-embedding table (collapsing N
        // duplicate ~1.16 GiB allocations into one for the real 1.7B), while
        // each replica keeps its own KV cache.
        //
        // The synthetic fixture loads on any CPU tier; `from_gguf` auto-detects
        // the kernel. On a GPU tier the pool clamps to size 1 (a process-global
        // singleton), in which case the multi-replica ptr-equality assertion is
        // vacuous — so we skip it and only sanity-check the single replica. On
        // this Mac `auto_detect` returns NEON (a CPU tier), so the multi-replica
        // path is the one normally exercised here.
        let bytes = build_tiny_gguf_bytes();
        let path = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "oxibonsai_pool_shared_embd_{}.gguf",
                std::process::id()
            ));
            p
        };
        std::fs::write(&path, &bytes).expect("write temp GGUF");

        let (pool, _tier, size) =
            build_pool_from_gguf(&path, SamplingParams::default(), 42, 512, Some(3))
                .expect("build_pool_from_gguf");

        // Clean up the temp file now that the GGUF is mmapped + leaked into the
        // pool (the leaked mmap keeps the bytes alive regardless of the file).
        let _ = std::fs::remove_file(&path);

        if size <= 1 {
            // GPU tier (or single-core host): only one replica exists, so there
            // is nothing to share. Just confirm the lone replica is usable.
            let lease = pool.acquire().await.expect("acquire sole replica");
            let embd = lease.model_token_embd();
            assert!(!embd.is_empty(), "token_embd must be populated");
            return;
        }

        // Acquire ALL replicas at once so we can compare every replica's
        // `token_embd` handle simultaneously. With `size` permits this never
        // blocks.
        let mut leases = Vec::with_capacity(size);
        for _ in 0..size {
            leases.push(pool.acquire().await.expect("acquire replica"));
        }

        // Every replica's token_embd must be the SAME allocation.
        let first_embd = leases[0].model_token_embd();
        for (i, lease) in leases.iter().enumerate().skip(1) {
            let other = lease.model_token_embd();
            assert!(
                Arc::ptr_eq(&first_embd, &other),
                "replica #{i} token_embd is a different allocation — sharing broken"
            );
        }

        // KV caches must be DISTINCT per replica (per-request mutable state).
        let kv_ptrs: Vec<*const _> = leases
            .iter()
            .map(|l| {
                l.dense_model()
                    .expect("the Q1 fixture is a dense model")
                    .kv_cache() as *const _
            })
            .collect();
        for i in 0..kv_ptrs.len() {
            for j in (i + 1)..kv_ptrs.len() {
                assert_ne!(
                    kv_ptrs[i], kv_ptrs[j],
                    "replicas #{i} and #{j} share a KV cache — isolation broken"
                );
            }
        }

        // Strong count: all `size` replicas alias the one allocation. We hold
        // `first_embd` plus `size` replica-held clones; the per-replica handle
        // pulled inside the loop above has been dropped. So the count is
        // `size + 1`.
        assert_eq!(
            Arc::strong_count(&first_embd),
            size + 1,
            "expected {size} replicas + the local handle to alias one allocation"
        );
    }

    // ── perf-M1: the real capacity must be visible to admission control ──

    #[test]
    fn admission_limit_is_derived_from_the_effective_pool_size() {
        let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
        assert_eq!(pool.size(), 2);
        // No queueing: admit only what can start immediately.
        assert_eq!(pool.admission_limit(0), 2);
        // One warm request per replica.
        assert_eq!(pool.admission_limit(1), 4);
        // A hostile depth must saturate, not wrap to a small limit.
        assert!(pool.admission_limit(usize::MAX) >= 2);
    }

    #[test]
    fn admission_limit_is_never_zero() {
        let pool = EnginePool::new(vec![tiny_engine()]);
        assert_eq!(pool.admission_limit(0), 1);
    }

    /// `SV-09` lifecycle: a request's cancellation token must not survive
    /// the lease that armed it. Without this, a replica that served one
    /// cancelled (or timed-out) request would instantly cancel the next
    /// request it is handed — a one-request outage per timeout.
    #[tokio::test]
    async fn a_cancelled_request_does_not_poison_the_next_lease() {
        let pool = EnginePool::new(vec![tiny_engine()]);

        let token = {
            let mut lease = pool.acquire().await.expect("acquire");
            // The server's shape: reset (as `run_blocking_generation` does),
            // then arm, then generate.
            lease.reset();
            let token = lease.arm_cancellation();
            token.cancel();
            assert!(lease.is_cancelled());
            token
        };
        // The token itself stays cancelled for whoever still holds it ...
        assert!(token.is_cancelled());

        // ... but the replica handed to the next request is not armed at all.
        let next = pool.acquire().await.expect("re-acquire");
        assert!(
            next.cancellation_token().is_none(),
            "the returned replica must carry no token from the previous request"
        );
        assert!(!next.is_cancelled());
    }

    #[tokio::test]
    async fn idle_count_tracks_outstanding_leases() {
        let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
        assert_eq!(pool.idle_count(), 2);
        let lease = pool.acquire().await.expect("acquire");
        assert_eq!(pool.idle_count(), 1);
        drop(lease);
        assert_eq!(pool.idle_count(), 2);
    }

    #[test]
    fn pool_sizing_reports_the_host_default_when_nothing_was_requested() {
        let sizing = resolve_pool_sizing(None, oxibonsai_kernels::KernelTier::Reference);
        assert_eq!(sizing.effective, default_cpu_pool_size());
        assert_eq!(sizing.requested, None);
        assert!(!sizing.clamped_by_gpu_tier);
        assert!(
            sizing.reason().contains("host default"),
            "{}",
            sizing.reason()
        );
    }

    #[test]
    fn pool_sizing_reports_an_honoured_request_on_a_cpu_tier() {
        let sizing = resolve_pool_sizing(Some(3), oxibonsai_kernels::KernelTier::Reference);
        assert_eq!(sizing.effective, 3);
        assert!(!sizing.clamped_by_gpu_tier);
        assert_eq!(
            sizing.effective,
            resolve_pool_size(Some(3), oxibonsai_kernels::KernelTier::Reference)
        );
    }

    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    #[test]
    fn pool_sizing_records_the_gpu_clamp_instead_of_hiding_it() {
        // `perf-M1`: a request above the ceiling must still be *reported* as a
        // clamp, so an admission controller can shed instead of queueing 8
        // requests behind 2 replicas.
        let tier = oxibonsai_kernels::KernelTier::Gpu;
        let sizing = resolve_pool_sizing_with_gpu_max(Some(8), tier, 2);
        assert_eq!(sizing.effective, 2);
        assert_eq!(sizing.gpu_max, Some(2));
        assert!(
            sizing.clamped_by_gpu_tier,
            "a server must be able to tell a clamp from an honoured request"
        );
        let reason = sizing.reason();
        assert!(reason.contains("requested 8"), "{reason}");
        assert!(reason.contains("GPU tier"), "{reason}");
        assert!(reason.contains("clamped to 2"), "{reason}");

        // A request at or below the ceiling is honoured, not clamped.
        let honoured = resolve_pool_sizing_with_gpu_max(Some(2), tier, 2);
        assert_eq!(honoured.effective, 2);
        assert!(!honoured.clamped_by_gpu_tier);

        // An unspecified request that lands on 1 anyway is not a clamp.
        let default_sizing = resolve_pool_sizing_with_gpu_max(None, tier, 4);
        assert_eq!(default_sizing.effective, 1);
        assert!(!default_sizing.clamped_by_gpu_tier);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_session_binds_and_releases_on_the_using_thread() {
        // Direct, non-vacuous cover for the lease's `Deref`-binds /
        // `Drop`-releases contract (`MET-08`). The pool fixtures in this crate
        // all resolve to a CPU tier, where `GpuSession` is empty, so the
        // binding itself is exercised here against a real session instead.
        let Ok(session) = oxibonsai_kernels::MetalGraph::new_session() else {
            return; // no Metal device on this host
        };
        let id = session.session_id();
        let gpu = GpuSession(Some(session));

        assert_eq!(oxibonsai_kernels::MetalGraph::current_session_id(), None);
        gpu.bind();
        assert_eq!(
            oxibonsai_kernels::MetalGraph::current_session_id(),
            Some(id),
            "the replica's session must be bound to the thread using it"
        );
        // Re-binding the same session is idempotent (this runs on every deref).
        gpu.bind();
        assert_eq!(
            oxibonsai_kernels::MetalGraph::current_session_id(),
            Some(id)
        );
        assert_eq!(
            oxibonsai_kernels::MetalGraph::global()
                .expect("global")
                .session_id(),
            id,
            "global() must resolve to the leased replica's session"
        );

        gpu.release();
        assert_eq!(
            oxibonsai_kernels::MetalGraph::current_session_id(),
            None,
            "returning a replica must release its binding"
        );
        // Releasing a binding this thread no longer holds is a no-op.
        gpu.release();
        assert_eq!(oxibonsai_kernels::MetalGraph::current_session_id(), None);
    }

    #[tokio::test]
    async fn every_replica_gets_its_own_gpu_session() {
        // `MET-08`: two replicas leased at once must never share a session —
        // a shared session means a shared device KV cache, which is exactly
        // the state two concurrent decodes would trample. Off the Metal tier
        // (this fixture is a CPU-tier `tiny_test` engine, and CI hosts have no
        // GPU at all) there is no session and both report `None`, which the
        // assertion below tolerates.
        let pool = EnginePool::new(vec![tiny_engine(), tiny_engine()]);
        let a = pool.acquire().await.expect("acquire a");
        let b = pool.acquire().await.expect("acquire b");
        match (a.gpu_session_id(), b.gpu_session_id()) {
            (Some(x), Some(y)) => assert_ne!(x, y, "two live replicas share one Metal session"),
            (None, None) => {}
            other => panic!("replicas disagree about having a GPU session: {other:?}"),
        }

        // A replica keeps its session across leases: the id it reports after
        // being returned and re-acquired is the one it had before.
        let a_id = a.gpu_session_id();
        drop(a);
        let a_again = pool.acquire().await.expect("re-acquire a");
        assert_eq!(a_again.gpu_session_id(), a_id);
    }
}
