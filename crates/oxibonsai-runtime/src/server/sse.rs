//! SSE response assembly: keep-alive, the whole-body deadline, `[DONE]`.
//!
//! Finding `sec-08` (with `SV-17` / `SV-23`): `per_request_timeout_ms` bounded
//! neither branch of the chat endpoint. The tower timeout in the serve
//! binaries races only the service *future*, which for an SSE handler resolves
//! as soon as the response head and the body object exist — the body stream is
//! entirely outside the budget, and the concurrency permit is released at the
//! head rather than at end-of-body. The `Sse` response also had no keep-alive,
//! so a stalled reader held the connection (and, through the unbounded token
//! channel, an engine replica's turn) indefinitely, and nothing told the client
//! the server was still alive.
//!
//! [`sse_response_tracked`] fixes all of that in one place:
//!
//! * the body is pumped by a task that is wrapped in
//!   [`tokio::time::timeout`], so the deadline covers the *stream*, not just
//!   the handler;
//! * on expiry it emits a terminal SSE `error` event carrying the canonical
//!   [`ApiError`] envelope, then `[DONE]`, then closes;
//! * a bounded channel gives the pump backpressure against a slow reader while
//!   the terminal events use a short send timeout so a stalled reader cannot
//!   pin the pump task forever;
//! * `KeepAlive` comments go out on the configured interval (15 s by default).
//!
//! Given the request's [`CancelSlot`], the pump also stops the generation
//! behind the stream the moment the stream is over for its client — the
//! deadline expired, or the client stopped reading (noticed while the pump
//! waits for the next payload too, so a client that leaves during a long
//! prefill is not discovered only at the first token) — and the terminal
//! `error` event names the stage the request was in.

use std::convert::Infallible;
use std::time::Duration;

use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};

use crate::server::api_error::ApiError;
use crate::server::budget::RequestLimits;
use crate::server::deadline::CancelSlot;

/// Terminal SSE payload every OpenAI-compatible stream ends with.
pub(crate) const DONE_PAYLOAD: &str = "[DONE]";

/// Buffered SSE events between the pump task and the connection.
const CHANNEL_CAPACITY: usize = 32;

/// How long the pump waits to hand a *terminal* event to a stalled reader
/// before giving up and closing the stream.
const TERMINAL_SEND_TIMEOUT: Duration = Duration::from_secs(1);

/// How the pump's copy loop ended.
enum PumpEnd {
    /// The payload stream ended.
    Finished,
    /// The client stopped reading.
    ClientGone,
}

/// Build the SSE response for a stream of already-serialized JSON payloads.
///
/// `payloads` yields chunk bodies (without the `data: ` prefix); this function
/// appends the terminal `[DONE]`, applies the keep-alive interval and enforces
/// `limits.per_request_timeout` over the whole body.
///
/// With the request's `slot`, an expired deadline names the stage the
/// request was in (its [`crate::server::phase::RequestPhase`]) instead of
/// saying only that time ran out, and both an expired deadline and a client
/// that stops reading cancel the generation the slot armed (see the module
/// docs).
pub(crate) fn sse_response_tracked<S>(
    payloads: S,
    limits: &RequestLimits,
    headers: HeaderMap,
    slot: Option<&CancelSlot>,
) -> Response
where
    S: Stream<Item = String> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(CHANNEL_CAPACITY);
    let deadline = limits.per_request_timeout;
    let slot = slot.cloned();

    tokio::spawn(async move {
        tokio::pin!(payloads);

        let timed_out = {
            let pump = async {
                loop {
                    // Waiting for the next payload also watches the client:
                    // a stream that goes quiet (a long prefill, a held-back
                    // tool call) must not hide a client that left.
                    let next = tokio::select! {
                        biased;
                        next = payloads.next() => next,
                        () = tx.closed() => return PumpEnd::ClientGone,
                    };
                    let Some(json) = next else {
                        return PumpEnd::Finished;
                    };
                    if tx.send(Ok(Event::default().data(json))).await.is_err() {
                        return PumpEnd::ClientGone;
                    }
                }
            };
            let ended = match deadline {
                Some(limit) => tokio::time::timeout(limit, pump).await.ok(),
                None => Some(pump.await),
            };
            match ended {
                Some(PumpEnd::Finished) => false,
                Some(PumpEnd::ClientGone) => {
                    // Nobody reads the rest: stop the generation behind it.
                    if let Some(slot) = slot.as_ref() {
                        slot.request_cancel();
                    }
                    return;
                }
                None => {
                    // The deadline expired: stop the generation first, then
                    // tell the client why the stream ends.
                    if let Some(slot) = slot.as_ref() {
                        slot.request_cancel();
                    }
                    true
                }
            }
        };

        if timed_out {
            let limit_ms = deadline.map(|d| d.as_millis()).unwrap_or_default();
            let error = match slot.as_ref().map(|slot| slot.phase().snapshot()) {
                Some(caught) => {
                    tracing::warn!(
                        timeout_ms = limit_ms as u64,
                        phase = caught.phase.name(),
                        prompt_rows = caught.prompt_rows,
                        generated = caught.generated,
                        "streaming request exceeded its deadline; closing the SSE connection"
                    );
                    caught.timeout_error(limit_ms)
                }
                None => {
                    tracing::warn!(
                        timeout_ms = limit_ms as u64,
                        "streaming request exceeded its deadline; closing the SSE connection"
                    );
                    ApiError::timeout(format!(
                        "request exceeded the server's per-request timeout of {limit_ms} ms"
                    ))
                }
            }
            .to_json()
            .to_string();
            if tx
                .send_timeout(
                    Ok(Event::default().event("error").data(error)),
                    TERMINAL_SEND_TIMEOUT,
                )
                .await
                .is_err()
            {
                return;
            }
        }

        let _ = tx
            .send_timeout(
                Ok(Event::default().data(DONE_PAYLOAD)),
                TERMINAL_SEND_TIMEOUT,
            )
            .await;
    });

    // `Sse::keep_alive` changes the body type, so the two arms build the
    // response separately rather than reassigning one binding.
    let body = ReceiverStream::new(rx);
    match limits.sse_keep_alive {
        Some(interval) => (
            headers,
            Sse::new(body).keep_alive(KeepAlive::new().interval(interval)),
        )
            .into_response(),
        None => (headers, Sse::new(body)).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_control::CancellationToken;
    use crate::server::phase::RequestPhase;
    use axum::body::to_bytes;

    async fn collect_body(response: Response) -> String {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read sse body");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// A stream that keeps no stage record and cancels nothing.
    fn sse_response<S>(payloads: S, limits: &RequestLimits, headers: HeaderMap) -> Response
    where
        S: Stream<Item = String> + Send + 'static,
    {
        sse_response_tracked(payloads, limits, headers, None)
    }

    /// A fresh slot, armed with a token the test keeps.
    fn armed_slot() -> (CancelSlot, CancellationToken) {
        let slot = CancelSlot::default();
        let token = CancellationToken::new();
        assert!(slot.arm(&token));
        (slot, token)
    }

    /// Wait (bounded) until `token` is cancelled; whether it was.
    async fn cancelled_within(token: &CancellationToken, limit: Duration) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed() < limit {
            if token.is_cancelled() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        token.is_cancelled()
    }

    #[tokio::test]
    async fn normal_stream_is_forwarded_and_terminated_with_done() {
        let payloads = tokio_stream::iter(vec!["{\"a\":1}".to_string(), "{\"a\":2}".to_string()]);
        let response = sse_response(payloads, &RequestLimits::default(), HeaderMap::new());
        let body = collect_body(response).await;
        assert!(body.contains("data: {\"a\":1}"), "body: {body}");
        assert!(body.contains("data: {\"a\":2}"), "body: {body}");
        assert!(body.trim_end().ends_with("data: [DONE]"), "body: {body}");
        assert!(
            !body.contains("event: error"),
            "a clean stream must not emit an error event: {body}"
        );
    }

    #[tokio::test]
    async fn deadline_emits_an_error_event_and_closes() {
        // A stream that yields one chunk and then never completes: without the
        // deadline this connection would stay open forever (sec-08).
        let payloads = tokio_stream::iter(vec!["{\"a\":1}".to_string()])
            .chain(tokio_stream::pending::<String>());
        let limits = RequestLimits::default().with_timeout(Some(Duration::from_millis(60)));
        let response = sse_response(payloads, &limits, HeaderMap::new());

        let body = tokio::time::timeout(Duration::from_secs(10), collect_body(response))
            .await
            .expect("the deadline must close the stream");
        assert!(body.contains("data: {\"a\":1}"), "body: {body}");
        assert!(body.contains("event: error"), "body: {body}");
        assert!(
            body.contains("request_timeout"),
            "the error event must carry the canonical envelope: {body}"
        );
        assert!(body.trim_end().ends_with("data: [DONE]"), "body: {body}");
    }

    /// The terminal `error` event of an expired deadline names the stage the
    /// request was in when it expired — read at expiry, not when the response
    /// was built.
    #[tokio::test]
    async fn a_tracked_deadline_names_the_stage_the_request_was_in() {
        use crate::server::phase::Phase;

        // A stream that stays open (its sender is alive) and never ends.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let payloads = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        let (slot, token) = armed_slot();
        let phase: RequestPhase = slot.phase();
        phase.set_workload(67, 1);
        phase.enter(Phase::Prefill);
        let limits = RequestLimits::default().with_timeout(Some(Duration::from_millis(400)));
        let response = sse_response_tracked(payloads, &limits, HeaderMap::new(), Some(&slot));
        // The request produces its first token before the deadline expires.
        assert!(tx.send("{\"a\":1}".to_string()).is_ok());
        phase.token_generated();
        phase.token_generated();

        let body = tokio::time::timeout(Duration::from_secs(10), collect_body(response))
            .await
            .expect("the deadline must close the stream");
        drop(tx);
        assert!(body.contains("event: error"), "body: {body}");
        assert!(body.contains("request_timeout"), "body: {body}");
        assert!(
            body.contains("during decode, after 2 generated tokens of the 67-position prompt"),
            "body: {body}"
        );
        assert!(body.contains("\"phase\":\"decode\""), "body: {body}");
        assert!(body.trim_end().ends_with("data: [DONE]"), "body: {body}");
        assert!(
            token.is_cancelled(),
            "the expired deadline cancels the generation"
        );
    }

    #[tokio::test]
    async fn a_tracked_deadline_in_prefill_says_no_token_had_been_generated() {
        use crate::server::phase::Phase;

        let payloads = tokio_stream::pending::<String>();
        let (slot, token) = armed_slot();
        let phase = slot.phase();
        phase.set_workload(67, 1);
        phase.enter(Phase::Prefill);
        let limits = RequestLimits::default().with_timeout(Some(Duration::from_millis(60)));
        let response = sse_response_tracked(payloads, &limits, HeaderMap::new(), Some(&slot));
        let body = tokio::time::timeout(Duration::from_secs(10), collect_body(response))
            .await
            .expect("the deadline must close the stream");
        assert!(
            body.contains(
                "during prefill of the 67-position prompt (no token had been generated yet)"
            ),
            "body: {body}"
        );
        assert!(
            body.contains("raise the server's per-request timeout"),
            "body: {body}"
        );
        assert!(body.contains("\"phase\":\"prefill\""), "body: {body}");
        assert!(
            token.is_cancelled(),
            "the generation in prefill is cancelled"
        );
    }

    /// A client that leaves while the stream is quiet — no payload has come
    /// for a while, as during a long prefill — is noticed without waiting for
    /// the next payload: the generation behind the stream is cancelled.
    #[tokio::test]
    async fn a_client_that_leaves_a_quiet_stream_cancels_its_generation() {
        let (slot, token) = armed_slot();
        let response = sse_response_tracked(
            tokio_stream::pending::<String>(),
            &RequestLimits::default(),
            HeaderMap::new(),
            Some(&slot),
        );
        // The client reads the head and goes away.
        drop(response);
        assert!(
            cancelled_within(&token, Duration::from_secs(5)).await,
            "the generation behind an abandoned stream must be cancelled"
        );
        assert!(slot.is_abandoned());
    }

    /// A stream that ends on its own cancels nothing.
    #[tokio::test]
    async fn a_finished_stream_cancels_nothing() {
        let (slot, token) = armed_slot();
        let response = sse_response_tracked(
            tokio_stream::iter(vec!["{\"a\":1}".to_string()]),
            &RequestLimits::default().with_timeout(Some(Duration::from_secs(60))),
            HeaderMap::new(),
            Some(&slot),
        );
        let body = collect_body(response).await;
        assert!(body.trim_end().ends_with("data: [DONE]"), "body: {body}");
        assert!(!token.is_cancelled());
        assert!(!slot.is_abandoned());
    }

    #[tokio::test]
    async fn an_untracked_deadline_keeps_the_plain_message() {
        let payloads = tokio_stream::pending::<String>();
        let limits = RequestLimits::default().with_timeout(Some(Duration::from_millis(60)));
        let response = sse_response(payloads, &limits, HeaderMap::new());
        let body = tokio::time::timeout(Duration::from_secs(10), collect_body(response))
            .await
            .expect("the deadline must close the stream");
        assert!(
            body.contains("request exceeded the server's per-request timeout of 60 ms"),
            "body: {body}"
        );
        assert!(
            !body.contains("\"phase\""),
            "no stage record, no stage: {body}"
        );
    }

    #[tokio::test]
    async fn keep_alive_can_be_disabled() {
        // Configuration plumbing only; the interval itself is axum's.
        let limits = RequestLimits::default().with_sse_keep_alive(None);
        assert!(limits.sse_keep_alive.is_none());
        let response = sse_response(
            tokio_stream::iter(vec!["x".to_string()]),
            &limits,
            HeaderMap::new(),
        );
        let body = collect_body(response).await;
        assert!(body.contains("data: x"), "body: {body}");
    }

    #[tokio::test]
    async fn response_headers_are_preserved() {
        let mut headers = HeaderMap::new();
        headers.insert(
            crate::server::REQUEST_ID_HEADER,
            axum::http::HeaderValue::from_static("11111111-1111-1111-1111-111111111111"),
        );
        let response = sse_response(
            tokio_stream::iter(vec!["x".to_string()]),
            &RequestLimits::default(),
            headers,
        );
        assert_eq!(
            response
                .headers()
                .get(crate::server::REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("11111111-1111-1111-1111-111111111111")
        );
    }
}
