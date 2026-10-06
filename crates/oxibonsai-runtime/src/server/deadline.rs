//! The per-request deadline every generation endpoint enforces, and the
//! cancellation it reaches.
//!
//! `/v1/chat/completions`, `/v1/chat/completions/extended` and
//! `/v1/completions` share one deadline: the server's per-request timeout
//! (`RequestLimits::per_request_timeout`, `--request-timeout-ms`), enforced
//! inside the handler by [`run_with_deadline`]. `/rag/query` (the `rag`
//! feature) enforces the same deadline with [`enforce_deadline`], handing it
//! the timeout its router was built with: the RAG router has no `AppState`
//! for [`run_with_deadline`] to read it from. When it expires:
//!
//! * the generation the request started is cancelled through the request's
//!   [`CancelSlot`] (and no later generation of the request may start), so
//!   the replica is free for the next request instead of decoding to
//!   `max_tokens` for nobody;
//! * a request whose response has not started gets `504` with `error.code:
//!   request_timeout`, a message naming the stage, `error.phase` set to the
//!   stage's stable name ([`super::phase::Phase::name`]) and, in decode,
//!   `error.generated_tokens`;
//! * a stream that is already open carries the same error object in its
//!   terminal SSE `error` event, followed by `[DONE]`
//!   ([`super::sse::sse_response_tracked`], which also cancels the slot).
//!
//! A request whose handler future is dropped for any other reason — the
//! client went away, or a layer outside the handler gave up on it — is
//! cancelled by the guard each non-streamed path (and `/rag/query`) holds
//! across its blocking generation ([`super::blocking::CancelOnAbandon`]), and
//! an open stream cancels its slot when the client stops reading.

use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use axum::response::Response;

use crate::engine_control::CancellationToken;
use crate::engine_pool::EngineLease;
use crate::metrics::InferenceMetrics;
use crate::server::api_error::ApiError;
use crate::server::phase::RequestPhase;
use crate::server::AppState;

/// Prefill chunk size armed alongside a request's cancellation token, so a
/// long prompt's ingest observes a cancellation between chunks instead of
/// running the whole prefill as one uninterruptible call. Small enough to
/// keep cancellation latency low, large enough not to meaningfully slow down
/// prefill throughput.
pub(crate) const CANCELLATION_PREFILL_CHUNK_TOKENS: usize = 512;

/// What a [`CancelSlot`] holds.
#[derive(Debug, Default)]
struct SlotState {
    /// The token of the generation running for the request, once one is.
    current: Option<CancellationToken>,
    /// The request is over for its client (its deadline expired, or its
    /// stream was abandoned): no further generation may start for it.
    abandoned: bool,
}

/// One request's handle on the generations it runs, shared between the
/// handler (which arms each generation it starts), the deadline branch of
/// [`run_with_deadline`] and an open stream (which cancel it), plus the
/// request's stage record ([`RequestPhase`]), which the deadline reads to say
/// where it caught the request.
///
/// Cloning yields another handle to the same slot.
#[derive(Clone, Default)]
pub(crate) struct CancelSlot {
    state: Arc<Mutex<SlotState>>,
    phase: RequestPhase,
}

impl std::fmt::Debug for CancelSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelSlot")
            .field("abandoned", &self.is_abandoned())
            .field("phase", &self.phase.current())
            .finish()
    }
}

impl CancelSlot {
    /// The slot's state. A poisoned lock still holds plain data (a token and
    /// a flag), so it is recovered rather than propagated.
    fn lock(&self) -> MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Arm `lease` for the request's next generation: a fresh cancellation
    /// token on the engine, recorded here so the deadline (or an abandoned
    /// stream) can cancel it, and the prefill chunked
    /// ([`CANCELLATION_PREFILL_CHUNK_TOKENS`]) so a long prompt's ingest
    /// observes it. Returns the token. Once the request has been abandoned
    /// the token comes back already cancelled, so the generation stops at
    /// its first step.
    pub(crate) fn arm_lease(&self, lease: &mut EngineLease) -> CancellationToken {
        let token = lease.arm_cancellation();
        lease.set_prefill_chunk_tokens(Some(CANCELLATION_PREFILL_CHUNK_TOKENS));
        self.arm(&token);
        token
    }

    /// Record `token` as the token of the generation the request runs now.
    /// `false` — with `token` cancelled — once the request was abandoned: no
    /// further generation may start for it.
    pub(crate) fn arm(&self, token: &CancellationToken) -> bool {
        let mut state = self.lock();
        if state.abandoned {
            token.cancel();
            return false;
        }
        state.current = Some(token.clone());
        true
    }

    /// The request is over for its client: cancel the generation in flight
    /// (if one was armed) and every one armed after this. Idempotent, and
    /// safe to call before anything was armed.
    pub(crate) fn request_cancel(&self) {
        let mut state = self.lock();
        state.abandoned = true;
        if let Some(token) = state.current.as_ref() {
            token.cancel();
        }
    }

    /// Whether [`Self::request_cancel`] was called.
    pub(crate) fn is_abandoned(&self) -> bool {
        self.lock().abandoned
    }

    /// A handle to the request's stage record.
    pub(crate) fn phase(&self) -> RequestPhase {
        self.phase.clone()
    }
}

/// Run `handler` — the body of one generation request — under the server's
/// per-request deadline (see the module docs). Without a configured deadline
/// it simply runs `handler`.
///
/// On expiry `handler` is dropped (with everything it holds: an abandoned
/// non-streamed generation's guard cancels its token as it goes), the slot is
/// cancelled, `errors_total` counts the failure and the answer is the
/// stage-naming `504`.
pub(crate) async fn run_with_deadline<F>(
    state: &AppState,
    slot: &CancelSlot,
    handler: F,
) -> Result<Response, ApiError>
where
    F: Future<Output = Result<Response, ApiError>>,
{
    enforce_deadline(
        state.limits.per_request_timeout,
        &state.metrics,
        slot,
        handler,
    )
    .await
}

/// [`run_with_deadline`] with the two things it reads from the server state —
/// the deadline (`None`: no deadline) and the metrics that count a request
/// it ends — passed in. A route whose state is not an [`AppState`]
/// (`/rag/query`) calls it directly.
///
/// The slot is cancelled on expiry whatever `handler` held: the guards a
/// handler holds go with it when it is dropped, but a generation the request
/// armed is the slot's to stop, guarded or not — and the slot then refuses
/// every generation the request would arm later.
pub(crate) async fn enforce_deadline<F>(
    limit: Option<Duration>,
    metrics: &InferenceMetrics,
    slot: &CancelSlot,
    handler: F,
) -> Result<Response, ApiError>
where
    F: Future<Output = Result<Response, ApiError>>,
{
    let Some(limit) = limit else {
        return handler.await;
    };
    match tokio::time::timeout(limit, handler).await {
        Ok(result) => result,
        Err(_) => {
            // Stop the in-flight generation (if one had started) rather than
            // letting it run to completion on a replica whose answer the
            // client will never see.
            slot.request_cancel();
            metrics.errors_total.inc();
            // Which stage the deadline caught the request in: the operator's
            // next move differs (raise the timeout for an image prefill, add
            // replicas for a queued request).
            let caught = slot.phase().snapshot();
            tracing::warn!(
                timeout_ms = u64::try_from(limit.as_millis()).unwrap_or(u64::MAX),
                phase = caught.phase.name(),
                prompt_rows = caught.prompt_rows,
                generated = caught.generated,
                "request exceeded the per-request timeout"
            );
            Err(caught.timeout_error(limit.as_millis()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::phase::Phase;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    #[test]
    fn a_slot_cancels_what_it_armed_and_everything_armed_later() {
        let slot = CancelSlot::default();
        let first = CancellationToken::new();
        assert!(slot.arm(&first));
        assert!(!first.is_cancelled());
        assert!(!slot.is_abandoned());

        slot.request_cancel();
        assert!(
            first.is_cancelled(),
            "the generation in flight is cancelled"
        );
        assert!(slot.is_abandoned());

        let later = CancellationToken::new();
        assert!(
            !slot.arm(&later),
            "an abandoned request starts no further generation"
        );
        assert!(later.is_cancelled());
    }

    #[test]
    fn cancelling_before_anything_was_armed_is_harmless_and_sticks() {
        let slot = CancelSlot::default();
        slot.request_cancel();
        slot.request_cancel();
        let token = CancellationToken::new();
        assert!(!slot.arm(&token));
        assert!(token.is_cancelled());
    }

    #[test]
    fn clones_share_the_slot_and_the_stage_record() {
        let slot = CancelSlot::default();
        let seen_by_the_deadline = slot.clone();
        let token = CancellationToken::new();
        assert!(slot.arm(&token));
        slot.phase()
            .enter(crate::server::phase::Phase::WaitingForEngine);
        assert_eq!(
            seen_by_the_deadline.phase().current(),
            crate::server::phase::Phase::WaitingForEngine
        );
        seen_by_the_deadline.request_cancel();
        assert!(token.is_cancelled());
        assert!(slot.is_abandoned());
    }

    // ── The deadline's own cancel ────────────────────────────────────────────
    //
    // Every non-streamed generation path also holds a guard that cancels its
    // generation when the handler is dropped (`blocking::CancelOnAbandon`),
    // so through the router a deadline cancels twice over. These tests take
    // the guard out of the picture — the handler armed a generation it does
    // not guard — so only the deadline's own cancel of the slot can stop it.

    /// A deadline short enough to keep the tests quick. The handlers below
    /// never answer (or answer at once), so nothing races it.
    const SHORT_DEADLINE: Duration = Duration::from_millis(50);

    /// How long a replica may take to come back once the deadline cancelled
    /// its generation: far below the scripted engine's five-second hold, so
    /// only a cancelled generation can make it.
    const REPLICA_BACK_WITHIN: Duration = Duration::from_millis(2_000);

    /// A handler that arms `token` in `slot` for a generation it does not
    /// guard, records one generated token, and never answers.
    async fn arm_unguarded_and_never_answer(
        slot: CancelSlot,
        token: CancellationToken,
    ) -> Result<Response, ApiError> {
        assert!(slot.arm(&token), "the request was not abandoned yet");
        let phase = slot.phase();
        phase.enter(Phase::Prefill);
        phase.token_generated();
        std::future::pending().await
    }

    /// Fails if the deadline stops cancelling the slot: the token of an
    /// unguarded generation would stay live, and the request could still
    /// arm another one.
    #[tokio::test]
    async fn an_expired_deadline_cancels_the_generation_the_request_armed() {
        let slot = CancelSlot::default();
        let metrics = InferenceMetrics::new();
        let token = CancellationToken::new();
        let outcome = enforce_deadline(
            Some(SHORT_DEADLINE),
            &metrics,
            &slot,
            arm_unguarded_and_never_answer(slot.clone(), token.clone()),
        )
        .await;
        let Err(error) = outcome else {
            panic!("the handler never answers: the deadline must");
        };
        assert_eq!(error.status(), StatusCode::GATEWAY_TIMEOUT);
        let json = error.to_json();
        assert_eq!(json["error"]["code"], "request_timeout", "{json}");
        assert_eq!(json["error"]["phase"], "decode", "{json}");
        assert_eq!(json["error"]["generated_tokens"], 1, "{json}");
        assert_eq!(metrics.errors_total.get(), 1);

        assert!(
            token.is_cancelled(),
            "the deadline cancels the generation in flight, guarded or not"
        );
        assert!(slot.is_abandoned());
        let later = CancellationToken::new();
        assert!(
            !slot.arm(&later),
            "no further generation of the request may start"
        );
        assert!(later.is_cancelled());
    }

    /// A handler that answers — with no deadline, or inside a generous one —
    /// is passed through untouched: nothing is cancelled or counted.
    #[tokio::test]
    async fn a_handler_that_answers_in_time_is_left_alone() {
        for limit in [None, Some(Duration::from_secs(600))] {
            let slot = CancelSlot::default();
            let metrics = InferenceMetrics::new();
            let token = CancellationToken::new();
            let handler = {
                let slot = slot.clone();
                let token = token.clone();
                async move {
                    assert!(slot.arm(&token));
                    Ok(StatusCode::NO_CONTENT.into_response())
                }
            };
            let outcome = enforce_deadline(limit, &metrics, &slot, handler).await;
            let Ok(response) = outcome else {
                panic!("{limit:?}: the handler's own answer is passed through");
            };
            assert_eq!(response.status(), StatusCode::NO_CONTENT, "{limit:?}");
            assert!(!token.is_cancelled(), "{limit:?}");
            assert!(!slot.is_abandoned(), "{limit:?}");
            assert_eq!(metrics.errors_total.get(), 0, "{limit:?}");
        }
    }

    /// The same, end to end on a replica: a generation armed through the
    /// slot runs on the blocking pool outside the handler, held by the
    /// scripted engine after its first token until it is cancelled (or for
    /// five seconds). Once the deadline expires the generation stops short of
    /// its script and the pool hands the replica out again promptly. Fails if
    /// the deadline stops cancelling the slot: the replica stays held for the
    /// whole hold.
    #[tokio::test]
    async fn an_expired_deadline_frees_a_replica_whose_generation_it_does_not_hold() {
        use crate::engine_pool::EnginePool;
        use crate::server::blocking::run_blocking_generation;
        use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

        const SCRIPT: &str = "abcdefghijklmnopqrstuvwxyz";
        let pool = EnginePool::new(vec![fx::scripted_byte_engine_held(SCRIPT)]);
        let slot = CancelSlot::default();
        let mut lease = pool.acquire().await.expect("the only replica");
        let token = slot.arm_lease(&mut lease);
        let generation = tokio::spawn(run_blocking_generation(lease, |lease| {
            lease.generate(&fx::byte_ids("hi"), 64)
        }));

        let metrics = InferenceMetrics::new();
        let outcome = enforce_deadline(
            Some(SHORT_DEADLINE),
            &metrics,
            &slot,
            std::future::pending::<Result<Response, ApiError>>(),
        )
        .await;
        assert!(outcome.is_err(), "the deadline answers");

        let back = tokio::time::timeout(REPLICA_BACK_WITHIN, pool.acquire()).await;
        assert!(
            matches!(back, Ok(Ok(_))),
            "the deadline must cancel the generation, freeing its replica"
        );
        assert!(token.is_cancelled());
        let finished = tokio::time::timeout(REPLICA_BACK_WITHIN, generation).await;
        let Ok(Ok(Ok(Ok(tokens)))) = finished else {
            panic!("the cancelled generation ends with the tokens it produced: {finished:?}");
        };
        assert!(
            tokens.len() < SCRIPT.len(),
            "the generation stopped short of its script: {tokens:?}"
        );
    }
}
