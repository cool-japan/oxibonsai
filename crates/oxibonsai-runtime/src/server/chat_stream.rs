//! The SSE-streaming half of the `/v1/chat/completions` pipeline.
//!
//! A child module of [`super`] (`server/chat.rs`), declared there through
//! `#[path]` so both files stay under the workspace's 2000-line ceiling.
//! `use super::*;` reaches every item `chat.rs` itself sees (its own
//! imports, `server.rs`'s types, and `chat.rs`'s private helpers); the items
//! `chat.rs` or its tests need back are `pub(super)`.

use super::*;

/// Owns everything that must live exactly as long as a streaming chat
/// completion's SSE body (`SV-08`): the in-flight gauge and the workload
/// rate-tracker sample fed to the `/admin/workload-stats` aggregator
/// (`SV-19`).
///
/// Wrapping the underlying stream in this type — rather than a bare local
/// variable in the handler — is what makes the gauge decrement (and the
/// rate sample get recorded) exactly once, whether the stream runs to
/// completion or the client disconnects early: both paths drop the wrapper
/// (a client disconnect drops the whole response body, which drops this
/// wrapper along with it), and `Drop` runs exactly once either way. The
/// previous design created the gauge guard as a plain local in the async
/// handler, which dropped as soon as the *initial* SSE `Response` object was
/// constructed — i.e. before the client had received a single byte of the
/// actual generation — permanently under-counting genuinely in-flight
/// streaming requests and never recording a rate sample for them at all.
pub(super) struct StreamLifecycleGuard<S> {
    pub(super) inner: S,
    pub(super) rate_aggregator: Arc<crate::request_metrics::RequestRateAggregator>,
    pub(super) tracker: Arc<std::sync::Mutex<crate::request_metrics::RequestRateTracker>>,
    /// SV-08: the instant the *whole request* started (not just the SSE
    /// head). `chat_completions` skips its own post-`.await` observation on
    /// the streaming branch, so `Drop` here is the only place a streaming
    /// request's latency is recorded, at true stream end.
    pub(super) request_start: std::time::Instant,
    /// Shared metrics registry the histogram observation above is recorded
    /// into.
    pub(super) metrics: Arc<InferenceMetrics>,
    /// Decrements `active_requests` when this guard drops. Declared last so
    /// it is dropped last among this struct's own fields (Rust drops fields
    /// in declaration order) — immaterial for correctness here since the
    /// two effects are independent, but keeps the gauge decrement visibly
    /// "closest to the end" for a reader.
    pub(super) _active_guard: ActiveRequestGuard,
}

impl<S: tokio_stream::Stream + Unpin> tokio_stream::Stream for StreamLifecycleGuard<S> {
    type Item = S::Item;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_next(cx)
    }
}

impl<S> Drop for StreamLifecycleGuard<S> {
    fn drop(&mut self) {
        if let Ok(tracker) = self.tracker.lock() {
            // Only record a sample when at least one token was actually
            // emitted. An unconditional record would feed a
            // `tokens_per_second: 0.0` / zero-elapsed sample into the
            // aggregator for every request that never got that far (a
            // prefill error, or a client that disconnects before the first
            // token) — dragging down the very averages `/admin/workload-stats`
            // (`SV-19`) exists to report *truthfully*.
            if tracker.tokens_emitted() > 0 {
                self.rate_aggregator.record(tracker.snapshot());
            }
        }
        // SV-08: observe the latency histogram at the stream's *true* end
        // (completion or disconnect, either of which drops this guard
        // exactly once), not at SSE-head construction time --
        // `chat_completions` skips its own observation on this branch so it
        // is not double-counted.
        self.metrics
            .request_duration_seconds
            .observe(self.request_start.elapsed().as_secs_f64());
        // `_active_guard`'s own `Drop` decrements `active_requests`
        // automatically once this function returns.
    }
}

/// Incremental stop-sequence state for a single streaming chat completion
/// (`SV-12`/`RT-04`), shared between the per-token decode closure and the
/// end-of-stream flush step in [`chat_completions_stream`].
///
/// Holds back the last `max_stop_len - 1` bytes of accumulated decoded text
/// at all times, so a stop sequence split across two decode chunks is still
/// caught before any of it reaches the client (the same class of bug
/// `RT-06` names for `api_extensions.rs`'s independent implementation).
#[derive(Default)]
pub(super) struct StopTracker {
    pub(super) accumulated: String,
    pub(super) emitted_len: usize,
    pub(super) stopped: bool,
}

impl StopTracker {
    /// Append `text` to the accumulated decoded output and return whatever
    /// portion is now safe to release to the client: everything up to (but
    /// not including) a stop-sequence match, or — when no match is found —
    /// everything except a trailing `max_stop_len - 1`-byte holdback
    /// window, snapped back to a valid UTF-8 char boundary (`accumulated`
    /// is always valid UTF-8 as a whole, since it is built only from
    /// `step_decode`'s already-complete pieces, but an arbitrary byte
    /// offset within it need not land on a char boundary).
    ///
    /// Sets `self.stopped` on a match. A no-op once `self.stopped` is
    /// already set.
    pub(super) fn push_and_release(
        &mut self,
        text: &str,
        stop_sequences: &[String],
        max_stop_len: usize,
    ) -> String {
        if self.stopped {
            return String::new();
        }
        self.accumulated.push_str(text);
        // A match can only ever start at or after `emitted_len`: were one
        // to start earlier, the holdback below would already have caught
        // it (an at-most-`max_stop_len - 1`-byte tail can never contain a
        // *complete* match on its own, by construction) — searching only
        // the unemitted suffix is therefore both correct and cheaper than
        // rescanning from the start every time.
        let unemitted = &self.accumulated[self.emitted_len..];
        match find_first_stop_match(unemitted, stop_sequences) {
            Some((rel_pos, _matched)) => {
                self.stopped = true;
                let abs_pos = self.emitted_len + rel_pos;
                let out = self.accumulated[self.emitted_len..abs_pos].to_string();
                self.emitted_len = self.accumulated.len(); // nothing more will ever be released.
                out
            }
            None => {
                let target = self
                    .accumulated
                    .len()
                    .saturating_sub(max_stop_len.saturating_sub(1));
                let mut boundary = target.min(self.accumulated.len());
                while boundary > self.emitted_len && !self.accumulated.is_char_boundary(boundary) {
                    boundary -= 1;
                }
                if boundary > self.emitted_len {
                    let out = self.accumulated[self.emitted_len..boundary].to_string();
                    self.emitted_len = boundary;
                    out
                } else {
                    String::new()
                }
            }
        }
    }

    /// Release everything still held back by the holdback window. Called
    /// once generation has ended (EOS / `max_tokens`) without ever matching
    /// a stop sequence, since "might still be the unfinished start of a
    /// match" no longer applies once there is nothing left to arrive.
    ///
    /// Returns an empty string (never re-releasing anything) once a stop
    /// sequence already matched, or once nothing remains to release.
    pub(super) fn take_remaining(&mut self) -> String {
        if self.stopped || self.emitted_len >= self.accumulated.len() {
            return String::new();
        }
        let out = self.accumulated[self.emitted_len..].to_string();
        self.emitted_len = self.accumulated.len();
        out
    }
}

/// The base endpoint's [`ContentStop`] stage: a [`StopTracker`] over the
/// request's stop sequences (text matching with a hold-back window of the
/// longest sequence's length minus one byte).
pub(super) struct TextStop {
    tracker: StopTracker,
    sequences: Vec<String>,
    max_len: usize,
}

impl TextStop {
    /// A stage over `sequences` (empty strings never match).
    pub(super) fn new(sequences: Vec<String>) -> Self {
        let max_len = sequences
            .iter()
            .filter(|s| !s.is_empty())
            .map(String::len)
            .max()
            .unwrap_or(0);
        Self {
            tracker: StopTracker::default(),
            sequences,
            max_len,
        }
    }
}

impl ContentStop for TextStop {
    fn push_text(&mut self, text: &str) -> String {
        self.tracker
            .push_and_release(text, &self.sequences, self.max_len)
    }

    fn release_held(&mut self) -> String {
        self.tracker.take_remaining()
    }

    fn mark_stopped(&mut self) {
        self.tracker.stopped = true;
    }

    fn stopped(&self) -> bool {
        self.tracker.stopped
    }
}

/// Bundled inputs for one streaming chat completion.
pub(super) struct StreamRequest {
    /// The prompt: token ids, or text with images spliced in (SV-11).
    pub(super) prompt: crate::vision_prefill::ChatPrompt,
    pub(super) max_tokens: usize,
    pub(super) overrides: SamplingOverrides,
    pub(super) penalties: PenaltyParams,
    /// The request's `min_p` (`None` keeps the replica's baseline).
    pub(super) min_p: Option<f32>,
    /// Stop sequences (`SV-12`/`RT-04`), checked incrementally against the
    /// content channel; a match truncates the emitted content and cancels
    /// the in-flight generation early.
    pub(super) stop_sequences: Vec<String>,
    /// Deterministic sampling seed (`RT-12`): when `Some`, the stream runs
    /// on a freshly seeded sampler for this one generation
    /// ([`RequestSampling`]); `None` leaves the replica's ambient PRNG
    /// advancing.
    pub(super) seed: Option<u64>,
    /// SV-09 server wiring: armed with a fresh [`crate::engine_control::CancellationToken`]
    /// once the engine lease is acquired, so the outer per-request timeout
    /// can cancel this generation early.
    pub(super) cancel_slot: CancelSlot,
    /// Whether the client set `stream_options.include_usage`, which adds the
    /// final usage chunk before `[DONE]` (finding `sec-08`).
    pub(super) include_usage: bool,
    /// How the response's tokens split into reasoning, content and tool
    /// calls.
    pub(super) shape: ResponseShape,
    pub(super) request_id: RequestId,
    /// SV-08: the whole request's start instant, carried into
    /// [`StreamLifecycleGuard`] so the latency histogram is observed at
    /// true stream end (or early disconnect) rather than at SSE-head
    /// construction time.
    pub(super) request_start: std::time::Instant,
    /// SV-08: kept alive for the whole SSE body (moved into
    /// [`StreamLifecycleGuard`]), not just until the initial response head
    /// is constructed.
    pub(super) active_guard: ActiveRequestGuard,
}

/// Build the JSON payload for the terminal SSE event of a streaming chat
/// completion.
///
/// * `Ok(finish_reason)` → an ordinary finish chunk carrying it (honest
///   truncation and tool-call signalling — findings `serve-api-01`, `RT-11`;
///   see [`crate::server::response_pipeline::ResponseEnd::finish_reason`]).
/// * `Err(message)` → an OpenAI-style error object, so a mid-stream failure
///   (including a prefill error that emitted zero tokens) is surfaced to the
///   client instead of being masked as a clean, complete response (finding
///   `serve-api-02`).
pub(super) fn stream_terminal_json(
    outcome: Result<&str, String>,
    id: &str,
    created: u64,
    model: &str,
    include_usage: bool,
) -> String {
    match outcome {
        Ok(finish_reason) => {
            let finish_chunk = ChatCompletionChunk {
                id: id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
                usage: usage_placeholder(include_usage),
            };
            serde_json::to_string(&finish_chunk).unwrap_or_default()
        }
        Err(message) => ApiError::internal(message).to_json().to_string(),
    }
}

/// `Some(null)` when the client asked for usage (OpenAI puts an explicit
/// `"usage": null` on every non-final chunk in that mode), `None` — i.e. the
/// member is omitted entirely — otherwise.
pub(super) fn usage_placeholder(include_usage: bool) -> Option<serde_json::Value> {
    include_usage.then_some(serde_json::Value::Null)
}

/// One non-final `chat.completion.chunk` carrying `delta` — the role delta,
/// a `content` delta, a native `reasoning_content` delta, or a `tool_calls`
/// delta.
fn delta_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    delta: ChunkDelta,
    include_usage: bool,
) -> String {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
        usage: usage_placeholder(include_usage),
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// The final usage-carrying chunk emitted just before `[DONE]` when
/// `stream_options.include_usage` is set (finding `sec-08`). It has an empty
/// `choices` array and real token counts, exactly like the OpenAI stream.
fn stream_usage_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
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

/// Everything the SSE tail emits for one generation outcome — `Ok((
/// finish_reason, generated))` or the error's message: the finish (or
/// error) chunk, followed by the usage chunk when the client asked for it and
/// generation succeeded.
pub(super) fn stream_terminal_payloads(
    outcome: Result<(&str, usize), String>,
    id: &str,
    created: u64,
    model: &str,
    include_usage: bool,
    prompt_tokens: usize,
) -> Vec<String> {
    let generated = outcome.as_ref().ok().map(|(_, generated)| *generated);
    let mut payloads = vec![stream_terminal_json(
        outcome.map(|(finish_reason, _)| finish_reason),
        id,
        created,
        model,
        include_usage,
    )];
    if include_usage {
        if let Some(completion_tokens) = generated {
            payloads.push(stream_usage_chunk_json(
                id,
                created,
                model,
                prompt_tokens,
                completion_tokens,
            ));
        }
    }
    payloads
}

/// The base endpoint's SSE chunk shapes ([`StreamChunks`]).
struct ChatChunks {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    prompt_tokens: usize,
}

impl ChatChunks {
    fn delta(&self, delta: ChunkDelta) -> String {
        delta_chunk_json(
            &self.id,
            self.created,
            &self.model,
            delta,
            self.include_usage,
        )
    }
}

impl StreamChunks for ChatChunks {
    fn role(&self) -> String {
        self.delta(ChunkDelta {
            role: Some("assistant".to_string()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        })
    }

    fn reasoning(&self, text: String) -> String {
        self.delta(ChunkDelta {
            role: None,
            content: None,
            reasoning_content: Some(text),
            tool_calls: None,
        })
    }

    fn content(&self, text: String) -> String {
        self.delta(ChunkDelta {
            role: None,
            content: Some(text),
            reasoning_content: None,
            tool_calls: None,
        })
    }

    fn tool_call(&self, index: usize, call: &crate::api_types::ToolCall) -> String {
        self.delta(ChunkDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![StreamToolCallDelta {
                index,
                call: call.clone(),
            }]),
        })
    }

    fn terminal(&self, outcome: Result<(&'static str, usize), String>) -> Vec<String> {
        stream_terminal_payloads(
            outcome,
            &self.id,
            self.created,
            &self.model,
            self.include_usage,
            self.prompt_tokens,
        )
    }
}

/// SSE streaming chat completion handler.
///
/// The generation runs on the blocking pool under the request's
/// [`RequestSampling`] and sends each token id through a channel; a
/// [`StreamDriver`] task feeds those ids through the response's
/// [`ResponsePipeline`] (reasoning split, tool-call extraction, stop
/// sequences) and sends each released event as its SSE chunk — reasoning
/// deltas, content deltas, and one `tool_calls` delta per completed call —
/// then the final chunk once the generation reports its outcome.
pub(super) async fn chat_completions_stream(
    state: Arc<AppState>,
    req: StreamRequest,
) -> Result<Response, ApiError> {
    let StreamRequest {
        prompt,
        max_tokens,
        overrides,
        penalties,
        min_p,
        stop_sequences,
        seed,
        cancel_slot,
        include_usage,
        shape,
        request_id,
        request_start,
        active_guard,
    } = req;
    let prompt_len = prompt.len();
    let completion_id = format!("chatcmpl-{}", rand_id());
    let created = unix_now_secs();

    // Report the real loaded model id in the streaming chunks (resolved once
    // and cached) instead of a hard-coded literal. Resolved before the lease
    // below: a cache-cold descriptor lookup acquires its own lease.
    let model_id = state.model_info().descriptor().await.id;

    // Acquire an engine lease in async context, then move it into the blocking
    // generation task. The lease's Drop (a synchronous std-mutex push) runs at
    // the closure's end — no async in Drop, so this is safe off the runtime.
    let mut lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        ApiError::service_unavailable(format!("no inference engine replica is available: {e}"))
            .with_code("engine_unavailable")
    })?;

    // Seed from the engine's actual running
    // configuration, never `SamplingParams::default()`.
    let sampling = RequestSampling {
        params: overrides.apply(lease.sampling_params()),
        penalties: Some(penalties),
        min_p,
        seed,
    };

    // SV-09 server wiring: arm cancellation, and keep a clone for the
    // stream driver so a stop-sequence match (or a vanished client) can cut
    // generation short, not just the outer per-request timeout.
    let cancel_token = lease.arm_cancellation();
    cancel_slot.arm(cancel_token.clone());
    lease.set_prefill_chunk_tokens(Some(CANCELLATION_PREFILL_CHUNK_TOKENS));

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel::<GenerationOutcome>();
    let metrics_for_task = Arc::clone(&state.metrics);
    tokio::task::spawn_blocking(move || {
        // RT-03: start from a clean KV cache so this request cannot inherit
        // state from the previous one served by this pool replica.
        lease.reset();
        // The request's params, penalties, min_p and seed for this one
        // generation; the replica's own sampler comes back afterwards.
        let result = sampling.run(&mut lease, |engine| {
            prompt.generate_streaming(engine, max_tokens, &token_tx)
        });
        // SV-08: streaming requests record `tokens_generated_total` /
        // `errors_total` from the real outcome the engine reported, like the
        // non-streaming path.
        match &result {
            Ok(count) => metrics_for_task
                .tokens_generated_total
                .inc_by(*count as u64),
            Err(_) => metrics_for_task.errors_total.inc(),
        }
        // A dropped receiver is harmless (the client went away).
        let _ = outcome_tx.send(result.map_err(|e| e.to_string()));
        // `lease` (and `token_tx`) drop here: the engine returns to the pool
        // and the token channel closes.
    });

    // SV-19: per-request rate tracker, shared with `StreamLifecycleGuard`
    // (below), which records the final snapshot into the admin aggregator
    // when the stream ends; the driver records each token as it arrives.
    let tracker = Arc::new(std::sync::Mutex::new(
        crate::request_metrics::RequestRateTracker::new(),
    ));
    if let Ok(mut t) = tracker.lock() {
        t.record_admission();
    }

    let pipeline = ResponsePipeline::new(&shape, state.tokenizer(), TextStop::new(stop_sequences));
    let (payload_tx, payload_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let driver = StreamDriver {
        state: Arc::clone(&state),
        pipeline,
        chunks: ChatChunks {
            id: completion_id,
            created,
            model: model_id,
            include_usage,
            prompt_tokens: prompt_len,
        },
        cancel: cancel_token,
        max_tokens,
        rate_tracker: Some(Arc::clone(&tracker)),
    };
    tokio::spawn(driver.run(token_rx, outcome_rx, payload_tx));

    // SV-08/SV-19: keep the in-flight gauge decremented (and the rate
    // sample + latency histogram recorded) at the true end of this stream's
    // life — completion *or* early client disconnect — rather than at the
    // moment this function returns the initial `Response` object, which
    // happens long before the client has seen any generated text.
    let guarded_stream = StreamLifecycleGuard {
        inner: tokio_stream::wrappers::UnboundedReceiverStream::new(payload_rx),
        rate_aggregator: Arc::clone(&state.rate_aggregator),
        tracker,
        request_start,
        metrics: Arc::clone(&state.metrics),
        _active_guard: active_guard,
    };

    // `sse::sse_response` appends `[DONE]`, applies the keep-alive interval
    // and enforces the per-request deadline over the whole body (finding
    // `sec-08`).
    Ok(sse::sse_response(
        guarded_stream,
        &state.limits,
        request_id_header_map(request_id),
    ))
}

#[cfg(test)]
#[path = "chat_stream_tests.rs"]
mod tests;
