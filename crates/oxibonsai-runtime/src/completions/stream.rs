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
//! `logprobs` is a real, honored combination with `stream: true` (`B2-13`
//! wave-4b `B8`): [`stream_completion_with_logprobs`] builds a dedicated
//! per-token streaming decode + logit-capture loop
//! ([`generate_streaming_logprobs_chunks`]) directly over
//! [`crate::engine::InferenceEngine`]'s own `prefill_for_generate` /
//! `sampler` / `decode_step` seams — `generate_with_logprobs` itself has no
//! streaming counterpart, but nothing about it needed one once those seams
//! were available to this crate. `seed` remains rejected together with
//! `stream: true` (`validate_completion_request`, `completions.rs`) — no
//! engine path streams deterministically-seeded generation.
//!
//! A configured stop sequence is honored with a byte-accurate hold-back
//! window ([`crate::pipeline::StopSequenceMatcher`], the same primitive
//! `api_extensions.rs`'s chat streaming path uses) so a match split across
//! two decoded chunks can never leak a prefix of it to the client. `B7`
//! (fixed, wave-4b): a match now also actually CANCELS the underlying
//! generation (`lease.arm_cancellation()` + a stop-tracker-triggered
//! `cancel_token.cancel()`, mirroring `server/chat.rs::chat_completions_stream`)
//! instead of only suppressing further SSE emission while the engine kept
//! decoding to `max_tokens` in the background — which also means
//! `finish_reason` now honestly reports `"stop"` rather than deriving it
//! from a token count the match never actually shortened.
//! [`StreamCancelGuard`] is the backstop for every other way the SSE
//! stream can end without a match (an early client disconnect, or
//! `sse_response`'s own per-request deadline) — cancellation on `Drop` is
//! unconditional and a no-op once generation already finished normally.

use super::*;

use crate::pipeline::{StopMatch, StopSequenceMatcher};
use crate::server::sse::sse_response;
use tokio_stream::StreamExt;

/// Cancels generation on drop (B7): an early client disconnect, or
/// `sse::sse_response`'s own per-request deadline, drops this stream
/// without ever reaching a stop-sequence match — before this fix neither
/// case told the background `generate_streaming_with_params` loop to stop,
/// so it kept decoding all the way to `max_tokens` on a lease the client
/// could no longer see, wasting engine-pool capacity ("the SSE deadline
/// does not cancel generation either"). Idempotent: cancelling a
/// generation that already finished normally (the common case) is a
/// documented no-op ([`crate::engine_control::CancellationToken::cancel`]),
/// so this is safe to run unconditionally on every drop, not just an
/// early one.
struct CancelOnDrop(crate::engine_control::CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Wraps [`stream_completion`]'s assembled SSE stream so cancellation fires
/// the instant the stream itself is dropped — completion, early client
/// disconnect, or the SSE layer's own deadline — rather than only on an
/// explicit stop-sequence match ([`CompletionStopTracker::push_and_release`]
/// already cancels inline; this is the backstop for every other way the
/// stream can end).
struct StreamCancelGuard<S> {
    inner: S,
    _cancel: CancelOnDrop,
}

impl<S: tokio_stream::Stream + Unpin> tokio_stream::Stream for StreamCancelGuard<S> {
    type Item = S::Item;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_next(cx)
    }
}

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
    /// `logprobs` top-k (B8): `Some` routes through
    /// [`stream_completion_with_logprobs`] instead of the plain content
    /// path. ORCHESTRATOR RULING D-3 required real SSE for `/v1/completions`
    /// without narrowing `logprobs` back out of streaming — an earlier
    /// revision of this endpoint rejected `stream + logprobs` with `400`;
    /// `InferenceEngine::sampler`/`model`/`kernel` are `pub(crate)`
    /// (`engine.rs:184-243`), so a streaming decode loop with logit capture
    /// can be, and now is, built entirely in this crate's owned files.
    pub(super) logprobs_top_k: Option<usize>,
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
        logprobs_top_k,
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

    let mut lease = state.acquire_engine().await.map_err(|e| {
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

    // B8: `logprobs` + `stream` together route through a dedicated
    // pipeline — the chunk shape (each `text_completion` delta carries its
    // own `logprobs` object) and the stop-sequence hold-back (token-
    // aligned, so a released chunk's parallel `tokens`/`token_logprobs`/
    // `top_logprobs`/`text_offset` arrays are never split mid-token) are
    // different enough from the plain content path below that sharing one
    // function would obscure both. `seed` is not threaded through here:
    // `validate_completion_request` already rejects `stream + seed`
    // unconditionally (no streaming engine seam accepts a per-call seed at
    // all), so it can never reach this branch.
    if let Some(top_k) = logprobs_top_k {
        return stream_completion_with_logprobs(
            state,
            lease,
            prompt_tokens,
            prompt_text,
            max_tokens,
            params,
            penalties,
            top_k,
            stop_checker,
            echo,
            include_usage,
            active_guard,
            model_id,
        )
        .await;
    }

    // B7: armed BEFORE the lease moves into the blocking task, exactly like
    // `server/chat.rs::chat_completions_stream` — a clone lets the
    // content-decoding closure below cancel generation the instant a stop
    // sequence matches (previously nothing did: the text was correctly
    // truncated for the client, but the underlying `generate_streaming_with_params`
    // loop kept running all the way to `max_tokens`, and `finish_reason`
    // — derived from the raw generated-token count — reported the wrong
    // `"length"` instead of `"stop"`). `StreamCancelGuard` (wrapped around
    // the final assembled stream, near the end of this function) is the
    // backstop for every OTHER way the stream can end without a match —
    // an early client disconnect or `sse_response`'s own per-request
    // deadline.
    let cancel_token = lease.arm_cancellation();
    let cancel_token_for_content = cancel_token.clone();

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
                let released = state.push_and_release(&piece, &stop_matcher_for_stream);
                // B7: a stop-sequence match must actually stop generation,
                // not just the text this stream chooses to emit — without
                // this, `generate_streaming_with_params` kept decoding all
                // the way to `max_tokens` on a lease the client could no
                // longer see any output from, and the blocking task's own
                // `finish_reason` (derived from the raw generated-token
                // count) reported the wrong `"length"` instead of `"stop"`.
                if state.stopped {
                    cancel_token_for_content.cancel();
                }
                released
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

    // B7: cancels generation the instant this stream is dropped for ANY
    // reason (natural completion is a harmless no-op cancel; an early
    // client disconnect or `sse_response`'s own per-request deadline are
    // the cases that previously left the blocking task running unobserved).
    let guarded_stream = StreamCancelGuard {
        inner: full_stream,
        _cancel: CancelOnDrop(cancel_token),
    };

    Ok(sse_response(
        guarded_stream,
        state.limits(),
        axum::http::HeaderMap::new(),
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// B8: `logprobs` + `stream` for `/v1/completions`
// ═══════════════════════════════════════════════════════════════════════

/// Per-stream stop-sequence hold-back state for the `logprobs` + `stream`
/// combination: the same byte-level "never emit a byte that could still be
/// the unfinished start of a match" guarantee [`CompletionStopTracker`]
/// gives the plain streaming path, but holding back whole pending
/// (piece, logprobs-entries) pairs rather than raw bytes, so a released
/// chunk's `tokens`/`token_logprobs`/`top_logprobs`/`text_offset` arrays
/// stay genuinely parallel to the text it carries — a token's own entry is
/// only ever released together with that token's own complete text, never
/// split.
#[derive(Default)]
struct LogprobsStopTracker {
    /// Exactly the concatenation of every `(piece, _)` in `pending`,
    /// maintained as an invariant by [`Self::release_up_to`]'s trim.
    accumulated: String,
    /// Not-yet-released `(decoded piece, logprobs entries)` pairs, oldest
    /// first. More than one entry per piece happens when a piece only
    /// became decodable after several UTF-8-incomplete tokens were
    /// buffered via [`Self::buffer_pending_entry`].
    pending: Vec<(String, Vec<crate::api_types::LogprobsContent>)>,
    /// Logprobs entries for tokens that have been sampled (and thus have a
    /// real entry) but whose own text is not yet decodable on its own
    /// (`step_decode` returned `Ok(None)`) — attached to the next piece
    /// that DOES complete.
    utf8_pending_entries: Vec<crate::api_types::LogprobsContent>,
    stopped: bool,
}

impl LogprobsStopTracker {
    /// Remember a logprobs entry whose token contributed to a still-
    /// incomplete UTF-8 sequence. See the field doc.
    fn buffer_pending_entry(&mut self, entry: crate::api_types::LogprobsContent) {
        self.utf8_pending_entries.push(entry);
    }

    /// Feed one COMPLETE decoded piece and the logprobs entry for the
    /// token that just completed it (plus every entry buffered by
    /// [`Self::buffer_pending_entry`] since the last completed piece).
    /// Returns the `(text, entries)` now safe to release as one SSE
    /// chunk, if any.
    fn push(
        &mut self,
        piece: String,
        entry: crate::api_types::LogprobsContent,
        matcher: &StopSequenceMatcher,
    ) -> Option<(String, Vec<crate::api_types::LogprobsContent>)> {
        if self.stopped || piece.is_empty() {
            return None;
        }
        let mut entries = std::mem::take(&mut self.utf8_pending_entries);
        entries.push(entry);

        self.accumulated.push_str(&piece);
        self.pending.push((piece, entries));

        if matcher.is_empty() {
            return self.release_up_to(self.accumulated.len());
        }
        let safe_len = match matcher.check(&self.accumulated) {
            StopMatch::Found { start, .. } => {
                self.stopped = true;
                start
            }
            StopMatch::None => {
                let hold = matcher.hold_back_len(&self.accumulated);
                self.accumulated.len().saturating_sub(hold)
            }
        };
        self.release_up_to(safe_len)
    }

    /// Release every whole pending piece whose contribution ends at or
    /// before the byte offset `safe_len` (measured into `self.accumulated`
    /// as it stood at the start of the call that computed `safe_len`).
    fn release_up_to(
        &mut self,
        safe_len: usize,
    ) -> Option<(String, Vec<crate::api_types::LogprobsContent>)> {
        let mut cursor = 0usize;
        let mut split_at = 0usize;
        for (piece, _) in &self.pending {
            let end = cursor + piece.len();
            if end > safe_len {
                break;
            }
            cursor = end;
            split_at += 1;
        }
        if split_at == 0 {
            return None;
        }
        let released: Vec<(String, Vec<crate::api_types::LogprobsContent>)> =
            self.pending.drain(..split_at).collect();
        let mut text = String::with_capacity(cursor);
        let mut entries = Vec::new();
        for (piece, piece_entries) in released {
            text.push_str(&piece);
            entries.extend(piece_entries);
        }
        self.accumulated = self.accumulated[cursor..].to_string();
        Some((text, entries))
    }

    /// Release everything still pending once generation ends naturally
    /// (EOS / `max_tokens` / cancellation) without ever matching a stop
    /// sequence. A no-op once a match already occurred or nothing remains.
    fn take_remaining(&mut self) -> Option<(String, Vec<crate::api_types::LogprobsContent>)> {
        if self.stopped || self.pending.is_empty() {
            return None;
        }
        self.release_up_to(self.accumulated.len())
    }
}

/// Runs entirely on the blocking pool: prefill, then per step — cancel
/// check, sample, EOS check, capture logprobs, UTF-8-safe piece decode,
/// stop-sequence hold-back — sending each safe-to-release `(text, entries)`
/// pair through `tx` as it becomes available.
///
/// Reimplements `InferenceEngine::generate_with_logprobs`'s exact per-step
/// loop (prefill_for_generate → sample_with_history → EOS check →
/// compute_logprobs → decode_step) using the same engine seams its
/// non-streaming counterpart uses — `prefill_for_generate` is `pub(crate)`,
/// `sampler` is a `pub(crate)` field, and the per-token forward pass is the
/// existing `pub fn decode_step` (already exactly `self.model.forward(token,
/// pos, &self.kernel)`, `engine.rs:632-635`, so this file never needs to
/// split `model`/`kernel` itself) — adding the streaming decode +
/// stop-sequence pairing that has no equivalent in the collect-everything
/// non-streaming method: this is the "streaming decode loop with logit
/// capture built in owned code" B8 calls for.
///
/// Returns the number of tokens actually generated (mirrors
/// `InferenceEngine::generate_streaming`'s return convention; the caller
/// uses it to derive `finish_reason`).
fn generate_streaming_logprobs_chunks(
    lease: &mut crate::engine_pool::EngineLease,
    tokenizer: Option<&crate::tokenizer_bridge::TokenizerBridge>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    top_k: usize,
    stop_matcher: &StopSequenceMatcher,
    tx: &tokio::sync::mpsc::UnboundedSender<(String, Vec<crate::api_types::LogprobsContent>)>,
) -> crate::error::RuntimeResult<usize> {
    if prompt_tokens.is_empty() {
        return Ok(0);
    }
    let top_k = top_k.min(20);
    let Some(mut last_logits) = lease.prefill_for_generate(prompt_tokens)? else {
        return Ok(0);
    };
    let id_to_token = |id: u32| -> String {
        match tokenizer {
            Some(tok) => tok.decode(&[id]).unwrap_or_else(|_| format!("<{id}>")),
            None => format!("<{id}>"),
        }
    };
    let mut decode_state = tokenizer.map(|t| t.new_decode_stream(true));
    let mut output_tokens: Vec<u32> = Vec::new();
    let mut tracker = LogprobsStopTracker::default();

    for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
        if lease.is_cancelled() {
            tracing::debug!(pos, "streaming logprobs generation cancelled");
            break;
        }
        let next_token = lease
            .sampler
            .sample_with_history(&last_logits, &output_tokens)?;
        if lease.is_eos(next_token) {
            tracing::debug!(pos, "EOS token generated (streaming logprobs)");
            break;
        }
        let entry =
            crate::api_types::compute_logprobs(&last_logits, next_token, top_k, &id_to_token);
        output_tokens.push(next_token);

        let piece = match (tokenizer, decode_state.as_mut()) {
            (Some(tok), Some(state)) => match tok.step_decode(state, next_token) {
                Ok(Some(text)) => Some(text),
                Ok(None) => None,
                Err(_) => Some(format!("[{next_token}]")),
            },
            _ => Some(format!("[{next_token}]")),
        };

        match piece {
            Some(piece) => {
                if let Some((text, entries)) = tracker.push(piece, entry, stop_matcher) {
                    if tx.send((text, entries)).is_err() {
                        tracing::debug!(pos, "receiver dropped, stopping generation");
                        break;
                    }
                }
            }
            None => tracker.buffer_pending_entry(entry),
        }

        if tracker.stopped {
            break;
        }

        last_logits = lease.decode_step(next_token, pos)?;
    }
    if let Some((text, entries)) = tracker.take_remaining() {
        let _ = tx.send((text, entries));
    }
    Ok(output_tokens.len())
}

/// Build one `text_completion` SSE chunk carrying its own `logprobs`
/// object — `entries`' parallel arrays describe exactly `text` (never more,
/// never less; see [`LogprobsStopTracker`]).
fn logprobs_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    entries: &[crate::api_types::LogprobsContent],
    base_offset: usize,
    include_usage: bool,
) -> String {
    // Every entry's own text is fully included (`LogprobsStopTracker` only
    // ever releases whole pieces), so passing a cursor of `usize::MAX`
    // guarantees `build_completion_logprobs` (this crate's shared,
    // already-tested per-token logprobs builder) never drops one.
    let logprobs = build_completion_logprobs(entries, usize::MAX, base_offset);
    let chunk = CompletionChunk {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![CompletionChunkChoice {
            text,
            index: 0,
            logprobs: serde_json::to_value(logprobs).ok(),
            finish_reason: None,
        }],
        usage: usage_placeholder(include_usage),
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// `logprobs` + `stream: true` end to end (B8). Reuses the caller's
/// already-resolved `prompt_tokens`/`lease`/`params`/`model_id` (all
/// resolved once by [`stream_completion`] before branching here).
#[allow(clippy::too_many_arguments)]
async fn stream_completion_with_logprobs(
    state: Arc<AppState>,
    mut lease: crate::engine_pool::EngineLease,
    prompt_tokens: Vec<u32>,
    prompt_text: String,
    max_tokens: usize,
    params: crate::sampling::SamplingParams,
    penalties: PenaltyParams,
    top_k: usize,
    stop_checker: StopChecker,
    echo: bool,
    include_usage: bool,
    active_guard: ActiveRequestGuard,
    model_id: String,
) -> Result<Response, ApiError> {
    let prompt_token_count = prompt_tokens.len();

    // B7, same as the plain content path: armed before the lease moves
    // into the blocking task, and the whole assembled stream is guarded so
    // an early disconnect / the SSE deadline still cancels generation.
    let cancel_token = lease.arm_cancellation();

    let completion_id = format!("cmpl-{}", completion_id_from_nanos());
    let created = unix_timestamp_secs();

    let (chunk_tx, chunk_rx) =
        tokio::sync::mpsc::unbounded_channel::<(String, Vec<crate::api_types::LogprobsContent>)>();
    let (terminal_tx, terminal_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let stop_matcher = StopSequenceMatcher::new(stop_checker.sequences());
    let state_for_task = Arc::clone(&state);
    let id_for_task = completion_id.clone();
    let model_for_task = model_id.clone();
    tokio::task::spawn_blocking(move || {
        // Keeps `active_requests` accurate for the whole generation, not
        // just until this function returns the initial SSE response.
        let _active_guard = active_guard;

        let mut lease = lease;
        lease.reset();
        let prev_penalties = lease.penalties();
        lease.set_penalties(penalties);
        // Params-only swap (RNG state preserved), matching
        // `generate_streaming_with_params`'s own contract — `seed` never
        // reaches this function (see `stream_completion`'s call site doc).
        let prev_params = lease.sampler.params().clone();
        lease.sampler.set_params(params);

        let tokenizer = state_for_task.tokenizer();
        let result = generate_streaming_logprobs_chunks(
            &mut lease,
            tokenizer,
            &prompt_tokens,
            max_tokens,
            top_k,
            &stop_matcher,
            &chunk_tx,
        );

        lease.sampler.set_params(prev_params);
        lease.set_penalties(prev_penalties);

        match &result {
            Ok(count) => state_for_task
                .metrics()
                .tokens_generated_total
                .inc_by(*count as u64),
            Err(_) => state_for_task.metrics().errors_total.inc(),
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
                tracing::error!(error = %e, "streaming logprobs generation failed mid-stream");
                vec![ApiError::internal(e.to_string()).to_json().to_string()]
            }
        };
        for chunk in payload {
            let _ = terminal_tx.send(chunk);
        }
        // `lease` (and thus `chunk_tx`) drops here.
    });

    // Running char offset for `text_offset` (legacy-completions
    // convention: starts past the echoed prompt's own length when `echo`
    // is set, matching `completions.rs::build_completion_logprobs`'s
    // non-streaming `base_offset`).
    let mut running_offset = if echo { prompt_text.chars().count() } else { 0 };
    let id_for_content = completion_id.clone();
    let model_for_content = model_id.clone();
    let content_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(chunk_rx).map(
        move |(text, entries)| {
            let chars = text.chars().count();
            let json = logprobs_chunk_json(
                &id_for_content,
                created,
                &model_for_content,
                text,
                &entries,
                running_offset,
                include_usage,
            );
            running_offset += chars;
            json
        },
    );

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

    let finish_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(terminal_rx);
    let full_stream = echo_stream.chain(content_stream).chain(finish_stream);
    let guarded_stream = StreamCancelGuard {
        inner: full_stream,
        _cancel: CancelOnDrop(cancel_token),
    };

    Ok(sse_response(
        guarded_stream,
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

    /// POST `body` to `/v1/completions` on `app` and return `(status,
    /// parsed JSON)`, for a plain (non-`stream`) request — the counterpart
    /// to [`post_sse`] above, needed by the cancellation-poisoning tests
    /// further down which must send a *non-streaming* request as the
    /// second half of the pair. Mirrors
    /// `completions_tests.rs::post_completion` exactly; duplicated locally
    /// rather than shared since the two `mod tests` are unrelated siblings.
    async fn post_completion(
        app: axum::Router,
        body: serde_json::Value,
    ) -> (axum::http::StatusCode, serde_json::Value) {
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
            .expect("body bytes");
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
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

    // ── B8: `logprobs` + `stream` is a real, honoured combination ────────

    /// `stream: true` + `logprobs` must be accepted (`200`) and every
    /// `text_completion` chunk carrying non-empty text must carry its own
    /// `logprobs` object with the legacy-completions parallel-array shape
    /// (`tokens`/`token_logprobs`/`top_logprobs`/`text_offset`, all the
    /// same length) — B8's "stream per-token logprobs in the
    /// text_completion chunks" — ending in a real `finish_reason` and
    /// `[DONE]`.
    #[tokio::test]
    async fn stream_with_logprobs_emits_a_logprobs_object_per_chunk() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 4, "stream": true, "logprobs": 2 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "body: {body}");
        assert!(
            body.trim_end().ends_with("data: [DONE]"),
            "must still terminate with [DONE]: {body}"
        );
        assert!(
            body.contains("\"finish_reason\":\"stop\"")
                || body.contains("\"finish_reason\":\"length\""),
            "must carry a real finish_reason: {body}"
        );

        let mut saw_a_logprobs_chunk = false;
        for line in body.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(data).expect("chunk must be JSON");
            let Some(choice) = v["choices"].get(0) else {
                continue;
            };
            let Some(text) = choice["text"].as_str() else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            let logprobs = &choice["logprobs"];
            assert!(
                !logprobs.is_null(),
                "a non-empty text_completion delta must carry logprobs: {data}"
            );
            let tokens = logprobs["tokens"].as_array().expect("tokens array");
            let token_logprobs = logprobs["token_logprobs"]
                .as_array()
                .expect("token_logprobs");
            let top_logprobs = logprobs["top_logprobs"].as_array().expect("top_logprobs");
            let text_offset = logprobs["text_offset"].as_array().expect("text_offset");
            assert_eq!(tokens.len(), token_logprobs.len());
            assert_eq!(tokens.len(), top_logprobs.len());
            assert_eq!(tokens.len(), text_offset.len());
            assert!(!tokens.is_empty(), "tokens must not be empty: {data}");
            saw_a_logprobs_chunk = true;
        }
        assert!(
            saw_a_logprobs_chunk,
            "no chunk in the stream carried a logprobs object: {body}"
        );
    }

    /// `text_offset` must be a genuine running cumulative offset across
    /// chunks, not reset to `0` on every one — the same convention
    /// `completions.rs::build_completion_logprobs` uses non-streaming.
    #[tokio::test]
    async fn stream_with_logprobs_text_offset_accumulates_across_chunks() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 6, "stream": true, "logprobs": 1 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);

        let mut offsets: Vec<u64> = Vec::new();
        for line in body.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(data).expect("chunk json");
            if let Some(offs) = v["choices"][0]["logprobs"]["text_offset"].as_array() {
                for o in offs {
                    offsets.push(o.as_u64().expect("offset is a number"));
                }
            }
        }
        assert!(
            offsets.len() >= 2,
            "need at least 2 offsets to prove accumulation, got {offsets:?}"
        );
        for pair in offsets.windows(2) {
            assert!(
                pair[1] > pair[0],
                "text_offset must strictly increase across tokens, got {offsets:?}"
            );
        }
    }

    /// `logprobs` + `stream` + `stop` together: the stream must still stop
    /// emitting once the stop sequence is found — proving B8's dedicated
    /// pipeline honours `stop` too, not just the plain content path.
    #[tokio::test]
    async fn stream_with_logprobs_and_stop_sequence_truncates() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({
                "prompt": "hello", "max_tokens": 32, "stream": true, "logprobs": 1, "stop": "["
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            body.contains("\"finish_reason\":\"stop\""),
            "a stop-sequence match must report \"stop\": {body}"
        );
        let mut total_text = String::new();
        for line in body.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(t) = v["choices"][0]["text"].as_str() {
                        total_text.push_str(t);
                    }
                }
            }
        }
        assert_eq!(
            total_text, "",
            "text must be truncated at the stop sequence found at position 0, got {total_text:?}"
        );
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

    // ── B7: a stop-sequence match actually cancels generation ────────────

    /// With no tokenizer attached, every streamed piece decodes to
    /// `"[<id>]"` (see `content_stream`'s `_ => format!("[{token_id}]")`
    /// fallback), so a stop sequence of `"["` matches on the very first
    /// token — deterministic, model-behavior-independent, mirroring
    /// `completions_tests.rs::handler_stop_sequence_truncates_the_completion_end_to_end`'s
    /// identical trick for the non-streaming path. Before the B7 fix this
    /// reported `finish_reason: "length"` (`generate_streaming_with_params`
    /// was never told to stop, so it ran all the way to `max_tokens`); now
    /// it must report `"stop"` AND generation must actually have been cut
    /// short (fewer than `max_tokens` tokens produced).
    #[tokio::test]
    async fn stream_stop_sequence_match_reports_stop_not_length_and_actually_cancels() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({
                "prompt": "hello", "max_tokens": 32, "stream": true, "stop": "["
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            body.contains("\"finish_reason\":\"stop\""),
            "a stop-sequence match must report finish_reason \"stop\", not \"length\": {body}"
        );
        assert!(
            !body.contains("\"finish_reason\":\"length\""),
            "must not ALSO carry a length finish event: {body}"
        );

        // The content itself must be empty (truncated at position 0, the
        // very first token) -- proving the match was caught immediately,
        // not merely mislabeled after running to completion.
        let mut total_text = String::new();
        for line in body.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(t) = v["choices"][0]["text"].as_str() {
                        total_text.push_str(t);
                    }
                }
            }
        }
        assert_eq!(
            total_text, "",
            "text must be truncated at the stop sequence found at position 0, got {total_text:?}"
        );
    }

    /// Companion to the above: WITHOUT a stop sequence, the same request
    /// still runs to `max_tokens` and reports `"length"` — proving the fix
    /// didn't just always report `"stop"`.
    #[tokio::test]
    async fn stream_without_stop_sequence_still_reports_length_at_the_token_limit() {
        let app = test_router();
        let (status, body) = post_sse(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            body.contains("\"finish_reason\":\"length\""),
            "no stop sequence configured: must exhaust max_tokens and report \"length\": {body}"
        );
    }

    // ── cross-request cancellation isolation, exercised over real HTTP ────
    //
    // `EngineLease::Drop` (`engine_pool.rs`, `SV-09`) already guarantees
    // `engine_pool::tests::a_cancelled_request_does_not_poison_the_next_lease`:
    // a request's cancellation token is cleared (`clear_cancellation_token`)
    // the moment its lease returns to the pool, whether or not that token
    // was ever actually cancelled. That test proves the guarantee at the
    // pool API; `test_router()` here builds a pool with exactly one
    // replica, so these two confirm the same guarantee still holds when the
    // token really was cancelled mid-stream (a live stop-sequence match,
    // `B7`) and the request is driven through the real `/v1/completions`
    // handlers (both the plain-content and the logprobs streaming closures)
    // rather than the pool directly.
    #[tokio::test]
    async fn a_stream_stopped_by_a_real_cancel_does_not_affect_the_next_request() {
        let app = test_router();
        let (stream_status, body) = post_sse(
            app.clone(),
            serde_json::json!({
                "prompt": "hello", "max_tokens": 32, "stream": true, "stop": "["
            }),
        )
        .await;
        assert_eq!(stream_status, axum::http::StatusCode::OK);
        assert!(
            body.contains("\"finish_reason\":\"stop\""),
            "sanity: the stop sequence must actually have matched: {body}"
        );

        let (status, json) = post_completion(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            json["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
            "a request after a real cancellation on this replica must still \
             generate tokens, got {json}"
        );
    }

    // ── B7, on the real model ──────────────────────────────────────────
    //
    // Every test above uses `Qwen3Config::tiny_test()` — a real
    // `InferenceEngine`, but not real generation, so a byte-accurate
    // stop-sequence match landing inside real, unpredictable model output
    // is untested. `server.rs`'s own `gpu_argmax_routing` module
    // establishes the convention this follows: real GGUF + tokenizer,
    // `#[ignore]`d by default, `OXI_MODEL`/`OXI_TOKENIZER`-overridable,
    // skip (not fail) when the fixture is absent so a fresh clone still
    // passes `cargo test`.

    /// Real-GGUF `InferenceEngine` + tokenizer as a router, or `None` (with
    /// an `eprintln!` explaining why) when the fixture is not available in
    /// this environment.
    fn real_model_router() -> Option<axum::Router> {
        let model_path = std::env::var("OXI_MODEL")
            .unwrap_or_else(|_| "models/Ternary-Bonsai-1.7B.gguf".to_string());
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("skip: real model not found at {model_path} (set OXI_MODEL)");
            return None;
        }
        let tokenizer_path =
            std::env::var("OXI_TOKENIZER").unwrap_or_else(|_| "models/tokenizer.json".to_string());
        let Ok(tokenizer) = crate::tokenizer_bridge::TokenizerBridge::from_file(&tokenizer_path)
        else {
            eprintln!("skip: could not load tokenizer at {tokenizer_path} (set OXI_TOKENIZER)");
            return None;
        };
        let params = SamplingParams::default();
        let engine = crate::engine::InferenceEngine::from_gguf_path(&model_path, params, 42, 4096)
            .expect("load the real GGUF");
        Some(crate::server::create_router(engine, Some(tokenizer)))
    }

    /// The same prompt/settings as `server::tests::gpu_argmax_routing`'s
    /// golden (temperature 0, so this is deterministic): with `stop:
    /// ["sea"]`, generation must actually be cut short at that match — not
    /// merely have its SSE emission suppressed while decoding ran on to
    /// `max_tokens` in the background (B7's original defect) — and the
    /// streamed text must equal the non-streaming path's text byte for
    /// byte, since both go through the same `StopSequenceMatcher` over the
    /// same deterministic generation.
    #[tokio::test]
    #[ignore = "requires the real Ternary-Bonsai-1.7B.gguf + tokenizer.json; run with --ignored"]
    async fn real_model_stream_stop_sequence_matches_the_non_stream_text_and_reports_stop() {
        let Some(app) = real_model_router() else {
            return;
        };
        let stream_body = serde_json::json!({
            "prompt": "Once upon a time, in a small village by the sea,",
            "max_tokens": 32,
            "temperature": 0.0,
            "stop": ["sea"],
            "stream": true
        });

        let (stream_status, sse_body) = post_sse(app.clone(), stream_body).await;
        assert_eq!(stream_status, axum::http::StatusCode::OK);
        assert!(
            sse_body.contains("\"finish_reason\":\"stop\""),
            "a real stop-sequence match must report finish_reason \"stop\": {sse_body}"
        );
        assert!(
            !sse_body.contains("\"finish_reason\":\"length\""),
            "must not ALSO carry a length finish event: {sse_body}"
        );
        let mut stream_text = String::new();
        for line in sse_body.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(t) = v["choices"][0]["text"].as_str() {
                        stream_text.push_str(t);
                    }
                }
            }
        }
        assert!(
            !stream_text.contains("sea"),
            "the stop sequence itself must never reach the client: {stream_text:?}"
        );

        let non_stream_body = serde_json::json!({
            "prompt": "Once upon a time, in a small village by the sea,",
            "max_tokens": 32,
            "temperature": 0.0,
            "stop": ["sea"]
        });
        let (status, json) = post_completion(app, non_stream_body).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        let non_stream_text = json["choices"][0]["text"].as_str().unwrap_or_default();

        assert_eq!(
            stream_text, non_stream_text,
            "streamed text must equal the non-streaming path's text byte for byte"
        );
    }

    /// Same confirmation through `stream_completion_with_logprobs`'s own,
    /// separate blocking closure and `arm_cancellation()` call.
    #[tokio::test]
    async fn a_logprobs_stream_stopped_by_a_real_cancel_does_not_affect_the_next_request() {
        let app = test_router();
        let (stream_status, body) = post_sse(
            app.clone(),
            serde_json::json!({
                "prompt": "hello", "max_tokens": 32, "stream": true, "logprobs": 2, "stop": "["
            }),
        )
        .await;
        assert_eq!(stream_status, axum::http::StatusCode::OK);
        assert!(
            body.contains("\"finish_reason\":\"stop\""),
            "sanity: the stop sequence must actually have matched: {body}"
        );

        let (status, json) = post_completion(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            json["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
            "a request after a real cancellation on this replica must still \
             generate tokens, got {json}"
        );
    }
}
