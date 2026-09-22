//! Async wrapper around the synchronous [`InferenceEngine`].
//!
//! Uses [`tokio::task::spawn_blocking`] for CPU-bound inference work,
//! ensuring the Tokio runtime is not blocked. Bounded concurrency is
//! enforced via a [`Semaphore`] to prevent resource exhaustion.
//!
//! This module is not available on WASM targets (`wasm32`) because tokio's
//! full feature set (including threads and network I/O) is not supported there.

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::Semaphore;

use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};
use crate::metrics::InferenceMetrics;

/// Async inference engine with bounded concurrency.
///
/// Wraps a synchronous [`InferenceEngine`] and provides async methods
/// that use `spawn_blocking` under the hood. The semaphore limits how
/// many concurrent inference requests can be in flight, protecting
/// both memory and CPU utilization.
pub struct AsyncInferenceEngine {
    engine: Arc<Mutex<InferenceEngine<'static>>>,
    concurrency_limit: Arc<Semaphore>,
    max_concurrent: usize,
    metrics: Option<Arc<InferenceMetrics>>,
}

impl AsyncInferenceEngine {
    /// Create a new async inference engine wrapping the given engine.
    ///
    /// `max_concurrent` controls how many inference requests may execute
    /// concurrently. A value of 1 serializes all requests.
    pub fn new(engine: InferenceEngine<'static>, max_concurrent: usize) -> Self {
        let effective_max = max_concurrent.max(1);
        Self {
            engine: Arc::new(Mutex::new(engine)),
            concurrency_limit: Arc::new(Semaphore::new(effective_max)),
            max_concurrent: effective_max,
            metrics: None,
        }
    }

    /// Attach shared metrics for recording inference telemetry.
    pub fn with_metrics(mut self, metrics: Arc<InferenceMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Generate tokens asynchronously.
    ///
    /// Blocks the caller until a semaphore permit is acquired, then
    /// dispatches the CPU-bound generation to a blocking thread.
    pub async fn generate(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> RuntimeResult<Vec<u32>> {
        // Acquire concurrency permit
        let _permit = self
            .concurrency_limit
            .acquire()
            .await
            .map_err(|_| RuntimeError::Server("semaphore closed".to_string()))?;

        if let Some(m) = &self.metrics {
            m.active_requests.inc();
        }

        let engine = Arc::clone(&self.engine);
        let metrics = self.metrics.clone();

        // Move CPU-bound work to a blocking thread
        let result = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            let mut engine_guard = rt.block_on(engine.lock());
            engine_guard.generate(&prompt_tokens, max_tokens)
        })
        .await
        .map_err(|e| RuntimeError::Server(format!("task join error: {e}")))?;

        if let Some(m) = &metrics {
            m.active_requests.dec();
        }

        result
    }

    /// Generate tokens with streaming via an unbounded channel.
    ///
    /// Returns a receiver that yields tokens as they are generated.
    /// The generation happens on a blocking thread; the receiver can
    /// be consumed asynchronously.
    pub async fn generate_streaming(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> RuntimeResult<tokio::sync::mpsc::UnboundedReceiver<u32>> {
        // Acquire concurrency permit
        let permit = self
            .concurrency_limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RuntimeError::Server("semaphore closed".to_string()))?;

        if let Some(m) = &self.metrics {
            m.active_requests.inc();
        }

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let engine = Arc::clone(&self.engine);
        let metrics = self.metrics.clone();

        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            let mut engine_guard = rt.block_on(engine.lock());
            let _result = engine_guard.generate_streaming(&prompt_tokens, max_tokens, &tx);
            // tx is dropped here, closing the channel

            if let Some(m) = &metrics {
                m.active_requests.dec();
            }

            // Permit is dropped here, releasing the semaphore slot
            drop(permit);
        });

        Ok(rx)
    }

    /// Current number of active (in-flight) requests.
    ///
    /// Computed as `max_concurrent - available_permits`.
    pub fn active_requests(&self) -> usize {
        self.max_concurrent - self.concurrency_limit.available_permits()
    }

    /// Maximum concurrent requests this engine allows.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Check if the engine has capacity for at least one more request.
    pub fn has_capacity(&self) -> bool {
        self.concurrency_limit.available_permits() > 0
    }

    /// Get a reference to the underlying engine (behind a mutex).
    pub fn engine(&self) -> &Arc<Mutex<InferenceEngine<'static>>> {
        &self.engine
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingParams;
    use oxibonsai_core::config::Qwen3Config;

    // HOTFIX-TESTMEM: `bonsai_8b()` made `BonsaiModel::new` allocate ~5 GB of
    // token_embd + output_weight tables (plus a ~1.2 GB KV cache) for every
    // test using this helper (7 call sites below); none exercise anything
    // dimension-dependent. `tiny_test()` exercises the identical
    // async-wrapper/concurrency-limiting wiring for a few tens of MB.
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
}
