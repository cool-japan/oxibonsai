//! The SSE-streaming half of `POST /v1/chat/completions/extended`.
//!
//! A child module of [`super`] (`api_extensions.rs`), in its own file so
//! both stay under the workspace's 2000-line ceiling. `use super::*;`
//! reaches every item `api_extensions.rs` sees (its imports and its private
//! helpers); the items the parent or its tests need back are `pub(super)`.

use super::*;

// ── Streaming (SSE) ───────────────────────────────────────────────────────────

/// One chunk of an extended-endpoint SSE stream (OpenAI `chat.completion.chunk`
/// shape). Structurally identical to `server.rs`'s private `ChatCompletionChunk`
/// but kept as its own type here since that one isn't `pub`.
#[derive(Debug, serde::Serialize)]
struct ExtendedChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ExtendedChunkChoice>,
    /// `None` → omitted (the client did not ask for usage); `Some(null)` on
    /// every chunk but the last once `stream_options.include_usage` is set;
    /// the usage object on the final usage chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, serde::Serialize)]
struct ExtendedChunkChoice {
    index: usize,
    delta: ExtendedChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, serde::Serialize)]
struct ExtendedChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// A native `reasoning_content` delta (the model's `<think>` span).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    /// A `tool_calls` delta: one completed call, whole, at its index.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<StreamToolCallDelta>>,
}

// ── Stop-sequence-safe streaming decode state (RT-06) ────────────────────────

/// Per-token stop-sequence hold-back state machine: the `/extended`
/// endpoint's [`ContentStop`] stage, streamed or not.
///
/// Factored out of the task itself so the hold-back / token-id-fast-path
/// logic — the actual fix for the chunk-boundary leak — is unit-testable
/// without a tokio runtime, a channel, or an HTTP router: feed it decoded
/// text (or a raw token id, for the id fast path) one step at a time and
/// assert on exactly what it says is safe to emit.
///
/// Never emits a byte that could still be swallowed by a stop sequence: text
/// is only returned once it is provably outside any window that could still
/// grow into a configured match (`hold_back_len`), and a token whose whole
/// decoded text equals a stop sequence in its own right is caught by id,
/// before any of its text is even considered.
pub(super) struct StreamDecodeState {
    matcher: StopSequenceMatcher,
    stop_token_ids: HashSet<u32>,
    accumulated: String,
    emitted_len: usize,
    hit_stop: bool,
}

impl StreamDecodeState {
    pub(super) fn new(stop_sequences: &[String], stop_token_ids: HashSet<u32>) -> Self {
        Self {
            matcher: StopSequenceMatcher::new(stop_sequences),
            stop_token_ids,
            accumulated: String::new(),
            emitted_len: 0,
            hit_stop: false,
        }
    }

    /// `true` once a stop sequence (by text or by id) has been matched; once
    /// set, [`feed`](Self::feed) and [`finish`](Self::finish) are inert.
    pub(super) fn is_stopped(&self) -> bool {
        self.hit_stop
    }

    /// Check `token_id` against the id fast path (see the struct docs) and,
    /// if it matches, mark the stream stopped. The whole point is to never
    /// EMIT so much as a byte of a token that is itself a configured stop
    /// marker — `flush_before_stop`, not `feed`, is what the caller must
    /// use once this returns `true`. Only the content channel is checked: a
    /// stop id occurring inside the reasoning span must never truncate the
    /// real answer.
    #[cfg(test)]
    pub(super) fn hit_stop_by_id(&mut self, token_id: u32) -> bool {
        if !self.hit_stop && self.stop_token_ids.contains(&token_id) {
            self.hit_stop = true;
        }
        self.hit_stop
    }

    /// Feed one token's already-decoded text. Returns the text, if any, that
    /// is now provably safe to emit to the client.
    pub(super) fn feed(&mut self, decoded_text: &str) -> Option<String> {
        if self.hit_stop || decoded_text.is_empty() {
            return None;
        }
        self.accumulated.push_str(decoded_text);

        match self.matcher.check(&self.accumulated) {
            StopMatch::Found { start, .. } => {
                self.hit_stop = true;
                (start > self.emitted_len)
                    .then(|| self.accumulated[self.emitted_len..start].to_string())
            }
            StopMatch::None => {
                // `hold_back_len` guarantees the cut lands on a UTF-8 char
                // boundary, so this never panics mid-codepoint on
                // multi-byte (e.g. CJK) output.
                let safe_len =
                    self.accumulated.len() - self.matcher.hold_back_len(&self.accumulated);
                if safe_len > self.emitted_len {
                    let visible = self.accumulated[self.emitted_len..safe_len].to_string();
                    self.emitted_len = safe_len;
                    Some(visible)
                } else {
                    None
                }
            }
        }
    }

    /// Flush whatever text is still held back once the token stream ends.
    /// A no-op once a stop sequence has already been matched (that text was
    /// never meant to reach the client) or when nothing is held back.
    pub(super) fn finish(&mut self) -> Option<String> {
        if self.hit_stop || self.accumulated.len() <= self.emitted_len {
            return None;
        }
        let visible = self.accumulated[self.emitted_len..].to_string();
        self.emitted_len = self.accumulated.len();
        Some(visible)
    }

    /// Flush whatever text is currently held back, bypassing the `hit_stop`
    /// guard [`finish`](Self::finish) applies — the companion of
    /// [`hit_stop_by_id`](Self::hit_stop_by_id) (RT-06): the token that
    /// trips the id fast path contributes no text of its own, so anything
    /// still in the hold-back window is real output from *earlier* tokens
    /// and must reach the client. The realistic trigger: `stop:
    /// ["</think>", "\nUser:"]` — the newline before `</think>` is held back
    /// as an unfinished prefix of `"\nUser:"`, then `</think>` arrives as its
    /// own token; that `"\n"` belongs to neither stop sequence.
    ///
    /// Deliberately unconditional (no attempt to detect whether the
    /// held-back tail is *also* an unfinished prefix of the very sequence
    /// that just tripped by id): that would need the stop text of each id,
    /// which `HashSet<u32>` does not keep, and only a model that spells part
    /// of its own one-token marker out of ordinary tokens right before
    /// emitting the marker itself could reach it.
    ///
    /// The response pipeline reaches the same result through
    /// [`ContentStop`]: on a stop id it releases the held text
    /// ([`ContentStop::release_held`]) before stopping the stage.
    #[cfg(test)]
    pub(super) fn flush_before_stop(&mut self) -> Option<String> {
        if self.accumulated.len() <= self.emitted_len {
            return None;
        }
        let visible = self.accumulated[self.emitted_len..].to_string();
        self.emitted_len = self.accumulated.len();
        Some(visible)
    }
}

/// The `/extended` endpoint's [`ContentStop`] stage: text matching with a
/// byte-accurate hold-back window, plus the id fast path for a stop
/// sequence that is one token's whole text.
impl ContentStop for StreamDecodeState {
    fn push_text(&mut self, text: &str) -> String {
        self.feed(text).unwrap_or_default()
    }

    fn is_stop_id(&self, id: u32) -> bool {
        !self.hit_stop && self.stop_token_ids.contains(&id)
    }

    fn release_held(&mut self) -> String {
        self.finish().unwrap_or_default()
    }

    fn mark_stopped(&mut self) {
        self.hit_stop = true;
    }

    fn stopped(&self) -> bool {
        self.is_stopped()
    }
}

/// The stop stage for one `/extended` response over `stop_sequences`: text
/// matching, plus the id fast path for every sequence that is exactly one
/// token of the loaded vocabulary (`RT-06`).
pub(super) fn stop_stage(state: &AppState, stop_sequences: &[String]) -> StreamDecodeState {
    let stop_token_ids: HashSet<u32> = stop_sequences
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|seq| {
            state
                .tokenizer()
                .and_then(|tok| tok.inner().token_to_id(seq))
        })
        .collect();
    StreamDecodeState::new(stop_sequences, stop_token_ids)
}

/// The `/extended` endpoint's SSE chunk shapes ([`StreamChunks`]).
struct ExtendedChunks {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    prompt_tokens: usize,
}

impl ExtendedChunks {
    fn chunk(&self, delta: ExtendedChunkDelta, finish_reason: Option<String>) -> String {
        let chunk = ExtendedChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ExtendedChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: self.include_usage.then_some(serde_json::Value::Null),
        };
        serde_json::to_string(&chunk).unwrap_or_default()
    }
}

impl StreamChunks for ExtendedChunks {
    fn role(&self) -> String {
        self.chunk(
            ExtendedChunkDelta {
                role: Some("assistant".to_string()),
                ..ExtendedChunkDelta::default()
            },
            None,
        )
    }

    fn reasoning(&self, text: String) -> String {
        self.chunk(
            ExtendedChunkDelta {
                reasoning_content: Some(text),
                ..ExtendedChunkDelta::default()
            },
            None,
        )
    }

    fn content(&self, text: String) -> String {
        self.chunk(
            ExtendedChunkDelta {
                content: Some(text),
                ..ExtendedChunkDelta::default()
            },
            None,
        )
    }

    fn tool_call(&self, index: usize, call: &crate::api_types::ToolCall) -> String {
        self.chunk(
            ExtendedChunkDelta {
                tool_calls: Some(vec![StreamToolCallDelta {
                    index,
                    call: call.clone(),
                }]),
                ..ExtendedChunkDelta::default()
            },
            None,
        )
    }

    fn terminal(&self, outcome: Result<(&'static str, usize), String>) -> Vec<String> {
        match outcome {
            Ok((finish_reason, generated)) => {
                let mut payloads = vec![self.chunk(
                    ExtendedChunkDelta::default(),
                    Some(finish_reason.to_string()),
                )];
                if self.include_usage {
                    let usage = ExtendedChunk {
                        id: self.id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: self.created,
                        model: self.model.clone(),
                        choices: Vec::new(),
                        usage: Some(serde_json::json!({
                            "prompt_tokens": self.prompt_tokens,
                            "completion_tokens": generated,
                            "total_tokens": self.prompt_tokens + generated,
                        })),
                    };
                    payloads.push(serde_json::to_string(&usage).unwrap_or_default());
                }
                payloads
            }
            Err(message) => {
                vec![
                    crate::server::ApiError::internal(format!("generation failed: {message}"))
                        .to_json()
                        .to_string(),
                ]
            }
        }
    }
}

/// Inputs of one `/extended` stream, resolved by
/// [`extended_chat_completions`] before it hands the lease over.
pub(super) struct ExtendedStream {
    pub(super) lease: EngineLease,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) max_tokens: usize,
    /// The request's sampling configuration ([`RequestSampling`]).
    pub(super) sampling: RequestSampling,
    pub(super) stop_sequences: Vec<String>,
    pub(super) model_id: String,
    /// How the response's tokens split into reasoning, content and tool
    /// calls.
    pub(super) shape: ResponseShape,
    /// `stream_options.include_usage`: every chunk carries `"usage": null`
    /// and a final usage chunk follows the finish chunk.
    pub(super) include_usage: bool,
    pub(super) metrics_guard: ActiveRequestGuard,
    pub(super) request_start: Instant,
}

/// Real SSE streaming for `POST /v1/chat/completions/extended`.
///
/// Reachable for a single choice without JSON mode (`n > 1` and a
/// `json_object`/`json_schema` `response_format` are refused with `400`
/// before this runs); `tools` stream. The generation runs on the blocking
/// pool under the request's [`RequestSampling`] (a seeded request on a
/// freshly seeded sampler, so two identical seeded requests produce
/// byte-identical streams), and a [`StreamDriver`] task feeds each token
/// through the response's [`ResponsePipeline`] — reasoning split, tool-call
/// extraction, then the stop stage ([`stop_stage`], `RT-06`: text is only
/// sent once it is provably outside any window that could still grow into a
/// stop sequence, and a one-token stop sequence matches by id) — sending
/// reasoning deltas, content deltas and one `tool_calls` delta per
/// completed call, then the final chunk (and, with
/// `stream_options.include_usage`, the usage chunk).
pub(super) async fn extended_chat_completions_stream(
    state: Arc<AppState>,
    request: ExtendedStream,
) -> axum::response::Response {
    let ExtendedStream {
        mut lease,
        prompt_tokens,
        max_tokens,
        sampling,
        stop_sequences,
        model_id,
        shape,
        include_usage,
        metrics_guard,
        request_start,
    } = request;
    let completion_id = format!("chatcmpl-ext-{}", rand_ext_id());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let prompt_len = prompt_tokens.len();

    // A stop-sequence match (or a vanished client) cancels the generation.
    let cancel_token = lease.arm_cancellation();

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel::<GenerationOutcome>();
    let metrics_for_task = Arc::clone(state.metrics());

    // Run generation on a blocking thread. The lease (and thus `token_tx`)
    // drops at the end of the closure, which both returns the engine to the
    // pool and closes the token channel so the driver below terminates.
    tokio::task::spawn_blocking(move || {
        lease.reset();
        let result = sampling.run(&mut lease, |engine| {
            engine.generate_streaming(&prompt_tokens, max_tokens, &token_tx)
        });
        // A failed generation ends the stream with an error object, never
        // with a clean `"stop"` that would pass the failure off as an empty
        // answer.
        if result.is_err() {
            metrics_for_task.errors_total.inc();
        }
        let _ = outcome_tx.send(result.map_err(|e| e.to_string()));
    });

    let pipeline = ResponsePipeline::new(
        &shape,
        state.tokenizer(),
        stop_stage(&state, &stop_sequences),
    );
    let (payload_tx, payload_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let driver = StreamDriver {
        state: Arc::clone(&state),
        pipeline,
        chunks: ExtendedChunks {
            id: completion_id,
            created,
            model: model_id,
            include_usage,
            prompt_tokens: prompt_len,
        },
        cancel: cancel_token,
        max_tokens,
        rate_tracker: None,
    };
    let state_for_driver = Arc::clone(&state);
    tokio::spawn(async move {
        // `SV-25`: `active_requests` / `request_duration_seconds` cover the
        // whole stream, not just the instant this function hands back the
        // initial `Sse` response.
        let _metrics_guard = metrics_guard;
        driver.run(token_rx, outcome_rx, payload_tx).await;
        state_for_driver
            .metrics()
            .request_duration_seconds
            .observe(request_start.elapsed().as_secs_f64());
    });

    let full_stream = UnboundedReceiverStream::new(payload_rx)
        .map(|json_str| -> Result<Event, Infallible> { Ok(Event::default().data(json_str)) })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));

    Sse::new(full_stream).into_response()
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
