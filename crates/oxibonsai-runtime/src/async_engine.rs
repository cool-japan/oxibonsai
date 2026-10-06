//! Async, pool-backed front end for the synchronous [`InferenceEngine`].
//!
//! CPU-bound generation runs on [`tokio::task::spawn_blocking`] so the tokio
//! runtime is never blocked, and a [`Semaphore`] bounds admission so a burst
//! cannot exhaust memory.
//!
//! # `SV-32`: what changed
//!
//! This wrapper used to hold a single `Arc<Mutex<InferenceEngine>>`. Its
//! `max_concurrent` semaphore therefore bounded *admission* while every
//! admitted request still queued on one engine — "bounded concurrency" that
//! delivered none — and each `spawn_blocking` closure called
//! `Handle::block_on(engine.lock())`, blocking a blocking-pool thread on an
//! async mutex. It was also unreferenced outside its own tests, so the defect
//! never surfaced in the served path.
//!
//! It is now backed by an [`EnginePool`]: `acquire()` hands out one replica per
//! in-flight request, the lease is moved into the blocking task and dropped
//! there (returning the replica to the pool), and on the Metal tier each
//! replica carries its own GPU session (`MET-08`), so requests overlap on the
//! GPU instead of serialising on a shared graph. `Self::new` still accepts a
//! lone engine — it becomes a one-replica pool, which is honestly reported by
//! `Self::replicas` — and `Self::from_pool` is the multi-replica path.
//!
//! This module is not available on WASM targets (`wasm32`) because tokio's
//! full feature set (including threads and network I/O) is not supported there.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::engine::InferenceEngine;
use crate::engine_pool::EnginePool;
use crate::error::{RuntimeError, RuntimeResult};
use crate::metrics::InferenceMetrics;

/// Async inference front end with bounded admission over a replica pool.
///
/// Every request takes one semaphore permit (admission) and one pool replica
/// (execution). With a one-replica pool the two coincide and requests run one
/// at a time; with an N-replica pool up to `min(max_concurrent, N)` run
/// genuinely in parallel.
pub struct AsyncInferenceEngine {
    /// The replica pool requests are served from.
    pool: Arc<EnginePool>,
    /// Admission gate: how many requests may be in flight at once.
    concurrency_limit: Arc<Semaphore>,
    /// The permit count, kept for [`Self::active_requests`].
    max_concurrent: usize,
    /// Optional shared telemetry.
    metrics: Option<Arc<InferenceMetrics>>,
}

impl AsyncInferenceEngine {
    /// Wrap a single engine, admitting at most `max_concurrent` requests.
    ///
    /// The engine becomes a **one-replica** pool, so admitted requests still
    /// execute one at a time — [`Self::replicas`] reports `1` rather than
    /// implying `max_concurrent`-way parallelism. Use [`Self::from_pool`] when
    /// real concurrency is wanted.
    ///
    /// A `max_concurrent` of `0` is raised to `1`.
    pub fn new(engine: InferenceEngine<'static>, max_concurrent: usize) -> Self {
        Self::from_pool_with_limit(EnginePool::new(vec![engine]), max_concurrent)
    }

    /// Wrap an existing pool, admitting exactly as many requests as it has
    /// replicas.
    ///
    /// This is the sizing an admission controller wants (`perf-M1`): admitting
    /// more than the pool can run does not fail fast, it queues invisibly on
    /// `acquire()` until requests time out.
    pub fn from_pool(pool: Arc<EnginePool>) -> Self {
        let size = pool.size();
        Self::from_pool_with_limit(pool, size)
    }

    /// Wrap a pool with an explicit admission limit.
    ///
    /// A limit above the pool size lets requests queue on `acquire()`; a limit
    /// below it leaves replicas idle. Both are occasionally what an operator
    /// wants, so neither is clamped — but see [`EnginePool::admission_limit`]
    /// for the recommended value.
    pub fn from_pool_with_limit(pool: Arc<EnginePool>, max_concurrent: usize) -> Self {
        let effective_max = max_concurrent.max(1);
        Self {
            pool,
            concurrency_limit: Arc::new(Semaphore::new(effective_max)),
            max_concurrent: effective_max,
            metrics: None,
        }
    }

    /// Attach shared metrics for recording inference telemetry.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<InferenceMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Generate tokens asynchronously.
    ///
    /// Waits for an admission permit, leases a replica, and runs the
    /// generation on a blocking thread. The lease is dropped inside that
    /// thread, so the replica is back in the pool the instant the work ends —
    /// and, on the Metal tier, its GPU session is released from that thread
    /// too (`MET-08`).
    ///
    /// # Errors
    ///
    /// [`RuntimeError::Server`] if the semaphore is closed, the pool cannot
    /// hand out a replica, or the blocking task panics; otherwise the
    /// engine's own error.
    pub async fn generate(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> RuntimeResult<Vec<u32>> {
        let _permit = self
            .concurrency_limit
            .acquire()
            .await
            .map_err(|_| RuntimeError::Server("semaphore closed".to_string()))?;

        if let Some(m) = &self.metrics {
            m.active_requests.inc();
        }

        let lease = self.pool.acquire().await;
        let metrics = self.metrics.clone();
        let result = match lease {
            Ok(lease) => tokio::task::spawn_blocking(move || {
                let mut lease = lease;
                // `RT-03`: never inherit the previous request's KV state — the
                // same reset the server's blocking seam performs.
                lease.reset();
                lease.generate(&prompt_tokens, max_tokens)
            })
            .await
            .map_err(|e| RuntimeError::Server(format!("task join error: {e}")))?,
            Err(e) => Err(RuntimeError::Server(format!("engine pool: {e}"))),
        };

        if let Some(m) = &metrics {
            m.active_requests.dec();
        }

        result
    }

    /// Generate tokens with streaming via an unbounded channel.
    ///
    /// Returns a receiver that yields tokens as they are generated. The
    /// generation happens on a blocking thread; the receiver can be consumed
    /// asynchronously. The admission permit and the replica lease are both
    /// held until the stream ends.
    ///
    /// # Errors
    ///
    /// [`RuntimeError::Server`] if the semaphore is closed or the pool cannot
    /// hand out a replica.
    pub async fn generate_streaming(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> RuntimeResult<tokio::sync::mpsc::UnboundedReceiver<u32>> {
        let permit = self
            .concurrency_limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RuntimeError::Server("semaphore closed".to_string()))?;

        if let Some(m) = &self.metrics {
            m.active_requests.inc();
        }

        let lease = match self.pool.acquire().await {
            Ok(lease) => lease,
            Err(e) => {
                if let Some(m) = &self.metrics {
                    m.active_requests.dec();
                }
                return Err(RuntimeError::Server(format!("engine pool: {e}")));
            }
        };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let metrics = self.metrics.clone();

        tokio::task::spawn_blocking(move || {
            let mut lease = lease;
            // `RT-03`, as in `generate`.
            lease.reset();
            let _result = lease.generate_streaming(&prompt_tokens, max_tokens, &tx);
            // `tx` is dropped here, closing the channel; `lease` is dropped
            // right after, returning the replica (and releasing its GPU
            // session binding) before the permit below.
            drop(lease);

            if let Some(m) = &metrics {
                m.active_requests.dec();
            }

            // Permit is dropped here, releasing the admission slot.
            drop(permit);
        });

        Ok(rx)
    }

    /// Current number of active (in-flight) requests.
    ///
    /// Computed as `max_concurrent - available_permits`.
    #[must_use]
    pub fn active_requests(&self) -> usize {
        self.max_concurrent
            .saturating_sub(self.concurrency_limit.available_permits())
    }

    /// Maximum concurrent requests this front end admits.
    #[must_use]
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// How many requests can actually *execute* at once.
    ///
    /// `SV-32`: this is the pool size, and it is not the same number as
    /// [`Self::max_concurrent`] — admitting more than this queues the surplus
    /// on `acquire()`.
    #[must_use]
    pub fn replicas(&self) -> usize {
        self.pool.size()
    }

    /// Check if the front end has capacity for at least one more request.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.concurrency_limit.available_permits() > 0
    }

    /// The replica pool behind this front end.
    #[must_use]
    pub fn pool(&self) -> &Arc<EnginePool> {
        &self.pool
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingParams;
    use oxibonsai_core::config::Qwen3Config;

    // Test memory bound: `bonsai_8b()` would make `BonsaiModel::new` allocate
    // ~5 GB of token_embd + output_weight tables (plus a ~1.2 GB KV cache) for
    // every test using this helper (7 call sites below); none exercise
    // anything dimension-dependent, so the helper uses `tiny_test()`, which
    // exercises the identical async-wrapper/concurrency-limiting wiring for a
    // few tens of MB.
    fn make_engine() -> InferenceEngine<'static> {
        let config = Qwen3Config::tiny_test();
        InferenceEngine::new(config, SamplingParams::default(), 42)
    }

    #[test]
    fn async_engine_creation() {
        let engine = make_engine();
        let async_engine = AsyncInferenceEngine::new(engine, 4);
        assert_eq!(async_engine.max_concurrent(), 4);
        assert_eq!(async_engine.active_requests(), 0);
        assert!(async_engine.has_capacity());
    }

    #[test]
    fn async_engine_min_concurrency_is_one() {
        let engine = make_engine();
        let async_engine = AsyncInferenceEngine::new(engine, 0);
        assert_eq!(async_engine.max_concurrent(), 1);
    }

    #[test]
    fn async_engine_with_metrics() {
        let engine = make_engine();
        let metrics = Arc::new(InferenceMetrics::new());
        let async_engine = AsyncInferenceEngine::new(engine, 2).with_metrics(Arc::clone(&metrics));
        assert_eq!(async_engine.max_concurrent(), 2);
        assert!(async_engine.has_capacity());
    }

    #[test]
    fn async_engine_capacity_tracking() {
        let engine = make_engine();
        let async_engine = AsyncInferenceEngine::new(engine, 3);
        // Initially at full capacity
        assert_eq!(async_engine.active_requests(), 0);
        assert!(async_engine.has_capacity());
        assert_eq!(async_engine.max_concurrent(), 3);
    }

    /// `SV-32`: a lone engine is a one-replica pool, and the wrapper says so
    /// instead of implying `max_concurrent`-way parallelism.
    #[test]
    fn a_single_engine_is_reported_as_one_replica() {
        let async_engine = AsyncInferenceEngine::new(make_engine(), 4);
        assert_eq!(async_engine.replicas(), 1);
        assert_eq!(async_engine.max_concurrent(), 4);
        assert_eq!(async_engine.pool().size(), 1);
    }

    /// `SV-32`: built from a pool, admission and execution agree.
    #[test]
    fn from_pool_admits_exactly_the_replica_count() {
        let pool = EnginePool::new(vec![make_engine(), make_engine(), make_engine()]);
        let async_engine = AsyncInferenceEngine::from_pool(Arc::clone(&pool));
        assert_eq!(async_engine.replicas(), 3);
        assert_eq!(async_engine.max_concurrent(), 3);
        assert!(Arc::ptr_eq(async_engine.pool(), &pool));
    }

    #[tokio::test]
    async fn async_engine_generate_empty_prompt() {
        let engine = make_engine();
        let async_engine = AsyncInferenceEngine::new(engine, 1);
        let result = async_engine.generate(vec![], 10).await;
        assert!(result.is_ok());
        let tokens = result.expect("should succeed");
        assert!(tokens.is_empty());
    }

    #[tokio::test]
    async fn async_engine_streaming_empty_prompt() {
        let engine = make_engine();
        let async_engine = AsyncInferenceEngine::new(engine, 1);
        let result = async_engine.generate_streaming(vec![], 10).await;
        assert!(result.is_ok());
        let mut rx = result.expect("should succeed");
        // Channel should be closed immediately (empty prompt produces no tokens)
        let token = rx.recv().await;
        assert!(token.is_none());
    }

    #[tokio::test]
    async fn async_engine_concurrency_respected() {
        let engine = make_engine();
        let async_engine = Arc::new(AsyncInferenceEngine::new(engine, 2));

        // We can check capacity before any requests
        assert!(async_engine.has_capacity());
        assert_eq!(async_engine.active_requests(), 0);

        // Generate with empty prompt should not exhaust permits
        let r1 = async_engine.generate(vec![], 1).await;
        assert!(r1.is_ok());
        // After completion, permits should be returned
        assert!(async_engine.has_capacity());
        assert_eq!(async_engine.active_requests(), 0);
    }

    /// The replica really is returned to the pool after each request, so a
    /// one-replica front end can serve request after request.
    #[tokio::test]
    async fn replicas_return_to_the_pool_between_requests() {
        let async_engine = AsyncInferenceEngine::new(make_engine(), 1);
        for _ in 0..3 {
            let out = async_engine
                .generate(vec![1, 2, 3], 2)
                .await
                .expect("generate");
            assert_eq!(out.len(), 2);
        }
        assert_eq!(async_engine.pool().idle_count(), 1);
    }

    /// Three replicas serve three requests at once, and every request still
    /// reproduces what it produces alone.
    #[tokio::test]
    async fn pool_backed_requests_run_concurrently_and_agree_with_baselines() {
        let prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let baselines: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| {
                let mut engine = make_engine();
                engine.generate(p, 4).expect("baseline")
            })
            .collect();

        let pool = EnginePool::new(vec![make_engine(), make_engine(), make_engine()]);
        let async_engine = Arc::new(AsyncInferenceEngine::from_pool(pool));

        let mut handles = Vec::new();
        for p in prompts {
            let ae = Arc::clone(&async_engine);
            handles.push(tokio::spawn(async move { ae.generate(p, 4).await }));
        }
        for (i, h) in handles.into_iter().enumerate() {
            let got = h.await.expect("join").expect("generate");
            assert_eq!(got, baselines[i], "concurrent request {i} diverged");
        }
    }
}
