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
//! [`sse_response`] fixes all of that in one place:
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

use std::convert::Infallible;
use std::time::Duration;

use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};

use crate::server::api_error::ApiError;
use crate::server::budget::RequestLimits;

/// Terminal SSE payload every OpenAI-compatible stream ends with.
pub(crate) const DONE_PAYLOAD: &str = "[DONE]";

/// Buffered SSE events between the pump task and the connection.
const CHANNEL_CAPACITY: usize = 32;

/// How long the pump waits to hand a *terminal* event to a stalled reader
/// before giving up and closing the stream.
const TERMINAL_SEND_TIMEOUT: Duration = Duration::from_secs(1);

/// Build the SSE response for a stream of already-serialized JSON payloads.
///
/// `payloads` yields chunk bodies (without the `data: ` prefix); this function
/// appends the terminal `[DONE]`, applies the keep-alive interval and enforces
/// `limits.per_request_timeout` over the whole body.
pub(crate) fn sse_response<S>(payloads: S, limits: &RequestLimits, headers: HeaderMap) -> Response
where
    S: Stream<Item = String> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(CHANNEL_CAPACITY);
    let deadline = limits.per_request_timeout;

    tokio::spawn(async move {
        tokio::pin!(payloads);

        let timed_out = {
            let pump = async {
                while let Some(json) = payloads.next().await {
                    if tx.send(Ok(Event::default().data(json))).await.is_err() {
                        // The client went away; nothing left to do.
                        return true;
                    }
                }
                false
            };
            match deadline {
                Some(limit) => match tokio::time::timeout(limit, pump).await {
                    Ok(client_gone) => {
                        if client_gone {
                            return;
                        }
                        false
                    }
                    Err(_) => true,
                },
                None => {
                    if pump.await {
                        return;
                    }
                    false
                }
            }
        };

        if timed_out {
            let limit_ms = deadline.map(|d| d.as_millis()).unwrap_or_default();
            tracing::warn!(
                timeout_ms = limit_ms as u64,
                "streaming request exceeded its deadline; closing the SSE connection"
            );
            let error = ApiError::timeout(format!(
                "request exceeded the server's per-request timeout of {limit_ms} ms"
            ))
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
    use axum::body::to_bytes;

    async fn collect_body(response: Response) -> String {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read sse body");
        String::from_utf8_lossy(&bytes).into_owned()
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
