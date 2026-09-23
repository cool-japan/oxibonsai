//! `POST /v1/completions` SSE streaming (`stream: true`).
//!
//! ORCHESTRATOR RULING D-3 (2026-09-22, final): the legacy endpoint must
//! support `stream: true` with real SSE — `text_completion` chunks,
//! `finish_reason`, optional `logprobs`, a `usage` chunk when
//! `stream_options.include_usage` is set, then `data: [DONE]` — sharing the
//! chat endpoint's SSE machinery ([`crate::server::sse::sse_response`])
//! rather than the wave-2 interim `400` this supersedes.
//!
//! Split out of `completions.rs` (`mod stream;`, `use super::*;` — the same
//! pattern `server/chat.rs` uses to stay under the workspace's 2000-line
//! ceiling relative to `server.rs`) both for that reason and because the
//! streaming and non-streaming paths share almost none of their generation
//! logic beyond the request fields `create_completion` already validated
//! and destructured.
//!
//! Deliberately scoped to the single-prompt, non-`logprobs`, non-`seed`
//! case — `validate_completion_request` rejects every other combination
//! with `stream: true` before this module ever runs, naming the field that
//! has no streaming-capable engine seam
//! ([`crate::engine::InferenceEngine::generate_with_logprobs`] and
//! [`crate::engine::InferenceEngine::generate_with_seed`] have no streaming
//! counterpart).
//!
//! A configured stop sequence is honored with a byte-accurate hold-back
//! window ([`crate::pipeline::StopSequenceMatcher`], the same primitive
//! `api_extensions.rs`'s chat streaming path uses) so a match split across
//! two decoded chunks can never leak a prefix of it to the client — but,
//! unlike the chat streaming path, a match does not cancel the underlying
//! generation early: the engine keeps decoding up to `max_tokens` in the
//! background exactly as the *non-streaming* `/v1/completions` path already
//! does (it also generates the full completion and only truncates the text
//! afterward), so this is not a regression relative to today's behavior —
//! it simply stops **emitting** SSE chunks once the match is confirmed.

use super::*;

use crate::pipeline::{StopMatch, StopSequenceMatcher};
use crate::server::sse::sse_response;
use tokio_stream::StreamExt;

/// Per-stream stop-sequence hold-back state, mirroring
/// `server/chat.rs::StopTracker`'s two-stage `push_and_release` +
/// `take_remaining` contract (that type is private to its module; this is
/// an independent implementation over the same shared
/// [`StopSequenceMatcher`] primitives, not a copy of its source).
#[derive(Default)]
struct CompletionStopTracker {
    accumulated: String,
    released_len: usize,
    stopped: bool,
}

impl CompletionStopTracker {
    /// Append `piece` and return whatever portion is now safe to release:
    /// everything up to (but not including) a stop-sequence match, or —
    /// when no match is found — everything except a trailing hold-back
    /// window that could still grow into one.
    fn push_and_release(&mut self, piece: &str, matcher: &StopSequenceMatcher) -> String {
        if self.stopped {
            return String::new();
        }
        self.accumulated.push_str(piece);
        if matcher.is_empty() {
            let safe_len = self.accumulated.len();
            if safe_len > self.released_len {
                let out = self.accumulated[self.released_len..safe_len].to_string();
                self.released_len = safe_len;
                return out;
            }
            return String::new();
        }
        match matcher.check(&self.accumulated) {
            StopMatch::Found { start, .. } => {
                self.stopped = true;
                if start > self.released_len {
                    let out = self.accumulated[self.released_len..start].to_string();
                    self.released_len = self.accumulated.len();
                    out
                } else {
                    self.released_len = self.accumulated.len();
                    String::new()
                }
            }
            StopMatch::None => {
                let hold = matcher.hold_back_len(&self.accumulated);
                let safe_len = self.accumulated.len().saturating_sub(hold);
                if safe_len > self.released_len {
                    let out = self.accumulated[self.released_len..safe_len].to_string();
                    self.released_len = safe_len;
                    out
                } else {
                    String::new()
                }
            }
        }
    }

    /// Release everything still held back once generation has ended
    /// naturally (EOS / `max_tokens`) without ever matching a stop
    /// sequence. A no-op once a match already occurred or nothing remains.
    fn take_remaining(&mut self) -> String {
        if self.stopped || self.released_len >= self.accumulated.len() {
            return String::new();
        }
        let out = self.accumulated[self.released_len..].to_string();
        self.released_len = self.accumulated.len();
        out
    }
}

/// Inputs [`create_completion`] has already validated, resolved and
/// tokenized-ready for a single-prompt streaming request.
pub(super) struct StreamRequest {
    pub(super) prompt_text: String,
    pub(super) max_tokens: usize,
    pub(super) penalties: PenaltyParams,
    pub(super) req_temperature: Option<f32>,
    pub(super) req_top_p: Option<f32>,
    pub(super) req_repetition_penalty: Option<f32>,
    pub(super) echo: bool,
    pub(super) stop_checker: StopChecker,
    pub(super) include_usage: bool,
    /// Moved here (not held by [`create_completion`] itself for this
    /// branch) so the `active_requests` gauge decrements when the
    /// background generation task actually finishes, not the moment this
    /// function returns the initial SSE response object.
    pub(super) active_guard: ActiveRequestGuard,
}

/// A `text_completion` SSE chunk (OpenAI-compatible).
#[derive(Debug, Serialize)]
struct CompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
    /// `None` → omitted entirely (a request that did not ask for usage);
    /// `Some(Value::Null)` → `"usage": null` on every non-final chunk once
    /// `stream_options.include_usage` is set; `Some(object)` → the final
    /// usage-carrying chunk. Mirrors `server/chat.rs`'s identical contract
    /// for `ChatCompletionChunk`.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<serde_json::Value>,
}

/// One choice in a [`CompletionChunk`] — always index `0`, since streaming
/// is single-prompt-only.
#[derive(Debug, Serialize)]
struct CompletionChunkChoice {
    text: String,
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<serde_json::Value>,
    finish_reason: Option<String>,
}

fn usage_placeholder(include_usage: bool) -> Option<serde_json::Value> {
    include_usage.then_some(serde_json::Value::Null)
}

/// A content delta chunk carrying `text`, `finish_reason: null`.
fn text_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    include_usage: bool,
) -> String {
    let chunk = CompletionChunk {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![CompletionChunkChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: None,
        }],
        usage: usage_placeholder(include_usage),
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// The terminal chunk: empty `text`, a real `finish_reason`.
fn finish_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &str,
    include_usage: bool,
) -> String {
    let chunk = CompletionChunk {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![CompletionChunkChoice {
            text: String::new(),
            index: 0,
            logprobs: None,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage: usage_placeholder(include_usage),
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// The final usage-carrying chunk, emitted just before `[DONE]` only when
/// `stream_options.include_usage` was set.
fn usage_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    let chunk = CompletionChunk {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: Vec::new(),
        usage: Some(serde_json::json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })),
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// Handle a validated, single-prompt `stream: true` request end to end.
pub(super) async fn stream_completion(
    state: Arc<AppState>,
    req: StreamRequest,
) -> Result<Response, ApiError> {
    let StreamRequest {
        prompt_text,
        max_tokens,
        penalties,
        req_temperature,
        req_top_p,
        req_repetition_penalty,
        echo,
        stop_checker,
        include_usage,
        active_guard,
    } = req;

    let prompt_tokens = if let Some(tok) = state.tokenizer() {
        tok.encode(&prompt_text).map_err(|e| {
            tracing::error!(error = %e, "tokenisation failed");
            state.metrics().errors_total.inc();
            ApiError::internal(format!("tokenisation failed: {e}"))
        })?
    } else {
        vec![151644u32]
    };
    let prompt_token_count = prompt_tokens.len();

    // Resolved BEFORE acquiring the engine lease below, deliberately: on a
    // pool with no engine replica free besides the one this request is
    // about to hold, `ServedModelInfo::descriptor` (`server.rs`) acquires
    // its own, *separate* lease from the very same pool to read the
    // model's config on its first (cache-filling) call — calling it while
    // already holding this request's own lease self-deadlocks a
    // single-replica pool (confirmed by reproduction: every streaming test
    // hung indefinitely until this call was moved ahead of
    // `acquire_engine` below, matching the ordering
    // `server/chat.rs::chat_completions_stream` already uses at its own
    // `model_id = state.model_info().descriptor().await.id` call site).
    let model_id = state.model_info().descriptor().await.id;

    let lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        state.metrics().errors_total.inc();
        ApiError::service_unavailable(format!("no inference engine replica is available: {e}"))
            .with_code("engine_unavailable")
    })?;

    // Gatekeeper `REQUIRED #1`/`REQUIRED #3`: seeded from the engine's own
    // ambient configuration, never `SamplingParams::default()` — see
    // `completions.rs`'s module doc.
    let params = resolve_sampling_params(
        lease.sampling_params(),
        req_temperature,
        req_top_p,
        req_repetition_penalty,
    );

    let completion_id = format!("cmpl-{}", completion_id_from_nanos());
    let created = unix_timestamp_secs();

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    let (terminal_tx, terminal_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let metrics_for_task = Arc::clone(state.metrics());
    let id_for_task = completion_id.clone();
    let model_for_task = model_id.clone();
    tokio::task::spawn_blocking(move || {
        // Keeps `active_requests` accurate for the lifetime of the actual
        // generation, not just until this function returns the initial SSE
        // response (see `StreamRequest::active_guard`'s doc).
        let _active_guard = active_guard;

        let mut lease = lease;
        lease.reset();
        let prev_penalties = lease.penalties();
        lease.set_penalties(penalties);
        let result =
            lease.generate_streaming_with_params(&prompt_tokens, max_tokens, &params, &token_tx);
        lease.set_penalties(prev_penalties);

        match &result {
            Ok(count) => metrics_for_task
                .tokens_generated_total
                .inc_by(*count as u64),
            Err(_) => metrics_for_task.errors_total.inc(),
        }

        let payload = match result {
            Ok(generated) => {
                let finish_reason = if generated >= max_tokens {
                    "length"
                } else {
                    "stop"
                };
                let mut payloads = vec![finish_chunk_json(
                    &id_for_task,
                    created,
                    &model_for_task,
                    finish_reason,
                    include_usage,
                )];
                if include_usage {
                    payloads.push(usage_chunk_json(
                        &id_for_task,
                        created,
                        &model_for_task,
                        prompt_token_count,
                        generated,
                    ));
                }
                payloads
            }
            Err(e) => {
                tracing::error!(error = %e, "streaming generation failed mid-stream");
                vec![ApiError::internal(e.to_string()).to_json().to_string()]
            }
        };
        for chunk in payload {
            let _ = terminal_tx.send(chunk);
        }
        // `lease` (and `token_tx`) drop here: the engine returns to the
        // pool and the token channel closes, ending `content_stream` below.
    });

    // Streaming decode: BPE tokens may straddle UTF-8 codepoint boundaries
    // (CJK, emoji), so buffer through the tokenizer's own decode-stream
    // state exactly like the chat endpoint does, rather than decoding each
    // token id in isolation.
    let mut decode_state = state.tokenizer().map(|t| t.new_decode_stream(true));
    let state_for_stream = Arc::clone(&state);
    // Shared with the flush stage below (built after `content_stream` is
    // exhausted): the hold-back window means up to `max_stop_len - 1`
    // already-safe bytes can still be sitting unreleased when generation
    // ends naturally (EOS / `max_tokens`) without ever matching a stop
    // sequence, and they must still reach the client rather than being
    // silently dropped — the same two-stage `push_and_release` +
    // `take_remaining` shape `server/chat.rs::StopTracker` uses (that type
    // is private to its module, so this is a from-scratch but
    // behaviourally identical implementation over the shared
    // [`StopSequenceMatcher`] primitives).
    let stop_state = Arc::new(std::sync::Mutex::new(CompletionStopTracker::default()));
    let stop_state_for_stream = Arc::clone(&stop_state);
    let stop_matcher = Arc::new(StopSequenceMatcher::new(stop_checker.sequences()));
    let stop_matcher_for_stream = Arc::clone(&stop_matcher);
    let id_for_stream = completion_id.clone();
    let model_for_stream = model_id.clone();

    let content_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(token_rx).filter_map(
        move |token_id| {
            let already_stopped = stop_state_for_stream
                .lock()
                .map(|s| s.stopped)
                .unwrap_or(false);
            if already_stopped {
                return None;
            }
            let piece = match (state_for_stream.tokenizer(), decode_state.as_mut()) {
                (Some(tok), Some(ds)) => match tok.step_decode(ds, token_id) {
                    Ok(Some(text)) => text,
                    Ok(None) => return None,
                    Err(_) => format!("[{token_id}]"),
                },
                _ => format!("[{token_id}]"),
            };

            let text = {
                let Ok(mut state) = stop_state_for_stream.lock() else {
                    return None;
                };
                state.push_and_release(&piece, &stop_matcher_for_stream)
            };
            if text.is_empty() {
                return None;
            }
            Some(text_chunk_json(
                &id_for_stream,
                created,
                &model_for_stream,
                text,
                include_usage,
            ))
        },
    );

    // Release whatever the holdback window above is still sitting on, once
    // `content_stream` is exhausted (`.chain` only polls this after that),
    // *provided* no stop sequence ever matched (a match already released
    // everything it safely could and marked nothing more should follow).
    let id_for_flush = completion_id.clone();
    let model_for_flush = model_id.clone();
    let flush_stream = tokio_stream::iter(std::iter::once_with(move || {
        let mut state = stop_state.lock().ok()?;
        let leftover = state.take_remaining();
        if leftover.is_empty() {
            return None;
        }
        Some(text_chunk_json(
            &id_for_flush,
            created,
            &model_for_flush,
            leftover,
            include_usage,
        ))
    }))
    .filter_map(|item| item);

    let finish_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(terminal_rx);

    let echo_stream = if echo && !prompt_text.is_empty() {
        tokio_stream::iter(vec![text_chunk_json(
            &completion_id,
            created,
            &model_id,
            prompt_text,
            include_usage,
        )])
    } else {
        tokio_stream::iter(Vec::new())
    };

    let full_stream = echo_stream
        .chain(content_stream)
        .chain(flush_stream)
        .chain(finish_stream);

    Ok(sse_response(
        full_stream,
        state.limits(),
        axum::http::HeaderMap::new(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    fn test_router() -> axum::Router {
        let config = oxibonsai_core::config::Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = crate::engine::InferenceEngine::new(config, params, 42);
        crate::server::create_router(engine, None)
    }

    async fn post_sse(
        app: axum::Router,
        body: serde_json::Value,
    ) -> (axum::http::StatusCode, String) {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&body).expect("body serialisation"),
            ))
            .expect("request build");
        let resp = app.oneshot(req).await.expect("response");
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read sse body");
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn stream_true_returns_ok_not_400() {
        let app = test_router();
        let (status, _) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn stream_emits_text_completion_chunks_and_done() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            body.contains("\"object\":\"text_completion\""),
            "body: {body}"
        );
        assert!(body.contains("\"finish_reason\":"), "body: {body}");
        assert!(body.trim_end().ends_with("data: [DONE]"), "body: {body}");
    }

    #[tokio::test]
    async fn stream_with_include_usage_emits_a_usage_chunk() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({
                "prompt": "hello", "max_tokens": 3, "stream": true,
                "stream_options": {"include_usage": true},
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(body.contains("\"prompt_tokens\":"), "body: {body}");
        assert!(body.contains("\"total_tokens\":"), "body: {body}");
    }

    #[tokio::test]
    async fn stream_without_include_usage_never_emits_a_real_usage_object() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(!body.contains("\"prompt_tokens\":"), "body: {body}");
    }

    #[tokio::test]
    async fn stream_batch_prompt_is_rejected_with_400() {
        let app = test_router();
        let (status, json_or_text) = post_sse(
            app,
            serde_json::json!({ "prompt": ["a", "b"], "max_tokens": 3, "stream": true }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(json_or_text.contains("stream"), "body: {json_or_text}");
    }

    #[tokio::test]
    async fn stream_with_logprobs_is_rejected_with_400() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true, "logprobs": 2 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body.contains("stream"), "body: {body}");
    }

    #[tokio::test]
    async fn stream_with_seed_is_rejected_with_400() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true, "seed": 7 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body.contains("stream"), "body: {body}");
    }

    #[tokio::test]
    async fn stream_with_echo_includes_the_prompt_text() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true, "echo": true }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(body.contains("hello"), "body: {body}");
    }
}
