//! Off-runtime execution of the (synchronous, CPU-bound) generation calls.
//!
//! Finding `sec-03` (with `SV-01`): the non-streaming chat path called
//! `generate_with_params_and_penalties` / `generate_with_logprobs` **directly
//! inside the async handler**, so a single request occupied a tokio worker
//! thread for its entire duration. On the default runtime that starves every
//! other connection on that worker, and combined with `sec-01` one 400 KB
//! prompt froze the server outright. The streaming path already used
//! `spawn_blocking`; the non-streaming ones did not.
//!
//! [`run_blocking_generation`] is the single seam every non-streaming
//! generation goes through — `/v1/chat/completions`, the `n`-loop of
//! `/v1/chat/completions/extended`, the prompt loop of `/v1/completions` and
//! `/rag/query` (the `rag` feature, whose generation used to run inline on a
//! worker too). It also closes finding `RT-03`: the engine is
//! [`InferenceEngine::reset`](crate::engine::InferenceEngine::reset)-ed on
//! acquisition, so KV state from the previous request served by that pool
//! replica cannot leak into this one.
//!
//! # Abandoned requests
//!
//! Moving the lease onto the blocking pool also moves it out of the handler's
//! reach: when the handler future is dropped — the client disconnected, or a
//! layer outside the handler gave up on it — the blocking task runs on. Every
//! non-streamed path (the three generation routes and `/rag/query`)
//! therefore holds a [`CancelOnAbandon`] built from the generation's
//! cancellation token across the `.await` and disarms it once the result is
//! in hand: a handler future dropped mid-generation cancels the token, the
//! generation stops at its next step, and the replica returns to the pool
//! instead of decoding to `max_tokens` for nobody.
//!
//! A late cancel cannot reach the next request: the token is scoped to the
//! lease that armed it (`EngineLease` detaches it on its way back into the
//! pool) and every request arms a fresh one, so cancelling a finished
//! request's token touches no engine.

use crate::engine_control::CancellationToken;
use crate::engine_pool::EngineLease;
use crate::server::api_error::ApiError;

/// Cancels a generation when dropped before [`CancelOnAbandon::disarm`]: the
/// handler future that started it was dropped (a client disconnect, an outer
/// timeout), so nobody will read the answer. See the module docs.
#[must_use = "a guard dropped at once cancels the generation it guards"]
#[derive(Debug)]
pub(crate) struct CancelOnAbandon(Option<CancellationToken>);

impl CancelOnAbandon {
    /// Guard the generation `token` was armed for.
    pub(crate) fn new(token: CancellationToken) -> Self {
        Self(Some(token))
    }

    /// The generation's result is in hand: dropping the guard now cancels
    /// nothing.
    pub(crate) fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for CancelOnAbandon {
    fn drop(&mut self) {
        if let Some(token) = self.0.take() {
            token.cancel();
        }
    }
}

/// Run a blocking generation closure on tokio's blocking pool.
///
/// The [`EngineLease`] is moved into the blocking task and dropped there, so
/// the engine returns to the pool (a synchronous mutex push — no async in
/// `Drop`) as soon as the work finishes, and the async handler never blocks.
///
/// The engine is reset before `work` runs, so every request starts from a clean
/// KV cache (`RT-03`).
///
/// # Errors
///
/// Returns `500` when the blocking task itself fails (panicked or was
/// cancelled); the closure's own `Result` is passed through untouched as `T`.
pub(crate) async fn run_blocking_generation<T, F>(
    lease: EngineLease,
    work: F,
) -> Result<T, ApiError>
where
    F: FnOnce(&mut EngineLease) -> T + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let mut lease = lease;
        // RT-03: never inherit the previous request's KV state.
        lease.reset();
        work(&mut lease)
    })
    .await
    .map_err(|e| {
        tracing::error!(error = %e, "blocking generation task failed");
        ApiError::internal("generation task failed")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use crate::engine_pool::EnginePool;
    use crate::sampling::SamplingParams;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn pool() -> Arc<EnginePool> {
        let engine = InferenceEngine::new(
            oxibonsai_core::config::Qwen3Config::tiny_test(),
            SamplingParams::default(),
            42,
        );
        EnginePool::new(vec![engine])
    }

    #[tokio::test]
    async fn returns_the_closure_value_and_releases_the_lease() {
        let pool = pool();
        let lease = pool.acquire().await.expect("acquire");
        let value = run_blocking_generation(lease, |lease| lease.vocab_size())
            .await
            .expect("blocking task");
        assert_eq!(
            value,
            oxibonsai_core::config::Qwen3Config::tiny_test().vocab_size
        );

        // The lease was dropped inside the blocking task, so the pool can hand
        // the replica out again immediately.
        let again = tokio::time::timeout(std::time::Duration::from_secs(5), pool.acquire())
            .await
            .expect("pool must not be starved");
        assert!(again.is_ok());
    }

    /// The `sec-03` property at unit level: on a *single-threaded* runtime, a
    /// concurrently spawned task must still make progress while a long
    /// synchronous generation is in flight. Running the work inline (the old
    /// behaviour) makes this impossible — the executor cannot poll anything
    /// until it returns.
    #[tokio::test(flavor = "current_thread")]
    async fn other_tasks_progress_while_blocking_work_runs() {
        let pool = pool();
        let lease = pool.acquire().await.expect("acquire");
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticks_for_task = Arc::clone(&ticks);

        let generation = tokio::spawn(run_blocking_generation(lease, move |_lease| {
            // Stand in for a long synchronous forward pass.
            std::thread::sleep(std::time::Duration::from_millis(300));
            7u32
        }));

        let ticker = tokio::spawn(async move {
            for _ in 0..5 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                ticks_for_task.fetch_add(1, Ordering::SeqCst);
            }
        });

        ticker.await.expect("ticker task");
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            5,
            "the async runtime must stay responsive while generation runs"
        );

        let value = generation.await.expect("join").expect("blocking task");
        assert_eq!(value, 7);
    }

    #[test]
    fn a_guard_dropped_while_armed_cancels_its_token() {
        let token = CancellationToken::new();
        drop(CancelOnAbandon::new(token.clone()));
        assert!(token.is_cancelled());
    }

    #[test]
    fn a_disarmed_guard_cancels_nothing() {
        let token = CancellationToken::new();
        CancelOnAbandon::new(token.clone()).disarm();
        assert!(!token.is_cancelled());
    }

    /// A guard that fires late — after its generation finished, its lease
    /// went back to the pool and the next request armed the same replica —
    /// cancels only its own (finished) request's token: the next request's
    /// generation runs to its end.
    #[tokio::test]
    async fn a_late_guard_does_not_reach_the_next_request_on_the_replica() {
        let pool = pool();
        let mut first = pool.acquire().await.expect("acquire");
        let first_token = first.arm_cancellation();
        let late = CancelOnAbandon::new(first_token.clone());
        let _ = run_blocking_generation(first, |lease| lease.generate(&[1, 2, 3], 2))
            .await
            .expect("blocking task");

        // The same (only) replica, armed for the next request.
        let mut next = pool.acquire().await.expect("re-acquire");
        let next_token = next.arm_cancellation();
        drop(late);
        assert!(first_token.is_cancelled(), "the late guard did fire");
        assert!(!next_token.is_cancelled());
        assert!(!next.is_cancelled(), "the replica carries the next token");
        let tokens = run_blocking_generation(next, |lease| lease.generate(&[1, 2, 3], 4))
            .await
            .expect("blocking task")
            .expect("generation");
        assert_eq!(tokens.len(), 4, "the next request runs to its budget");
    }
}

#[cfg(test)]
#[path = "abandon_tests.rs"]
mod abandon_tests;
