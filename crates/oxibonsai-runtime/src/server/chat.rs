//! The `/v1/chat/completions` request pipeline: validation, sampling-params
//! resolution, non-streaming and SSE-streaming generation.
//!
//! Split out of `server.rs` (which stayed the single assembly/route/type
//! definition point) purely to keep both files under the workspace's
//! 2000-line ceiling — every item here is `server`-private (or
//! `pub(super)` for the one route handler `create_router_full` registers),
//! so this split changes no public API and no behavior.
//!
//! `use super::*;` pulls in `server.rs`'s own imports (axum types,
//! `AppState`, the request/response/chunk types, `RequestId`, `SamplingParams`
//! / `PenaltyParams`, `TokenizerBridge`, the `budget`/`sanitize`/`sse`/
//! `blocking` sibling submodules, …) — the same pattern `server.rs`'s own
//! `#[cfg(test)] mod tests` already uses to reach its parent's private items.
//!
//! # B2-13 (wave 4b): real chat-template rendering + `<think>` split
//!
//! Prompt construction now goes through [`crate::tokenizer_bridge::chat_render::render_chat_prompt`]
//! (B1) — the model's own resolved chat template, rendered through the
//! real Jinja engine, in place of the hardcoded ChatML segment builder
//! (`sanitize::encode_chat_prompt`, still used only for the no-tokenizer
//! fallback shape and unaffected otherwise) — and the raw `<think>`/
//! `</think>` token-id split (`B3`,
//! [`crate::reasoning::ReasoningSplitter`]) is wired into both the
//! non-streaming and streaming paths, surfacing `reasoning_content`
//! alongside `content`.
//!
//! `reasoning_content` and `chat_template_kwargs`/`enable_thinking`/
//! `reasoning_effort` have no field on `ChatCompletionRequest`/`ChatMessage`/
//! `ChunkDelta` (`server.rs`, owned by ENGINE-SEAM this wave — see
//! `deviations` for the exact diff those types need once this package owns
//! `server.rs` again): the request side is read from the RAW request body
//! via [`ChatRequestExtras`] (captured alongside the typed
//! `ChatCompletionRequest`, from the same bytes — this is also B4's raw-JSON
//! seam for `tools`, preserving the client's own key order instead of
//! re-sorting it through a `serde_json::Value` round trip); the response
//! side is patched into the JSON `Value` after serializing the typed
//! response/chunk structs (see [`with_extra_delta_field`] /
//! [`chat_completions_non_stream`]'s own response-building tail).

use super::*;
use crate::tokenizer_bridge::chat_render::{self, to_render_messages, ChatRequestExtras};

/// Sampling-parameter overrides extracted from a request.
///
/// Overlaid onto whatever the acquired engine replica is actually running
/// with ([`crate::engine::InferenceEngine::sampling_params`]) — **never**
/// onto a bare [`SamplingParams::default()`] (gatekeeper REQUIRED#1(b)): the
/// latter silently reintroduced a `repetition_penalty` of `1.1` for every
/// request regardless of the engine's configured startup defaults, which
/// also permanently disqualified a plain `temperature: 0` request from the
/// fused GPU-argmax fast path (`InferenceEngine::greedy_gpu_eligible`
/// requires `repetition_penalty == 1.0`).
#[derive(Debug, Clone)]
struct SamplingOverrides {
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
}

impl SamplingOverrides {
    fn from_request(body: &ChatCompletionRequest) -> Self {
        Self {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            repetition_penalty: body.repetition_penalty,
        }
    }

    /// Overlay these overrides onto `seeded` (the engine's ambient
    /// [`SamplingParams`], read via
    /// [`crate::engine::InferenceEngine::sampling_params`] on the lease that
    /// will actually run this request).
    fn apply(&self, seeded: &SamplingParams) -> SamplingParams {
        let mut params = seeded.clone();
        params.temperature = self.temperature;
        if let Some(top_p) = self.top_p {
            params.top_p = top_p;
        }
        if let Some(top_k) = self.top_k {
            params.top_k = top_k;
        }
        if let Some(rp) = self.repetition_penalty {
            params.repetition_penalty = rp;
        }
        params
    }
}

/// A slot the outer per-request-timeout branch in [`chat_completions`] uses
/// to cancel whichever generation is in flight for this request, once one
/// has actually started (`SV-09` server wiring).
///
/// The token itself is created deep inside [`chat_completions_non_stream`]
/// / [`chat_completions_stream`] (where the lease is acquired), so this slot
/// is how it becomes visible to the outer timeout branch, which runs
/// concurrently via `tokio::time::timeout`. `None` until a lease is
/// acquired and armed — the timeout branch is a no-op if it fires before
/// that (the request is still queued on the pool, not yet running, so there
/// is nothing to cancel; the pool acquire future is what gets dropped, which
/// is enough on its own).
#[derive(Clone, Default)]
struct CancelSlot(Arc<std::sync::Mutex<Option<crate::engine_control::CancellationToken>>>);

impl CancelSlot {
    /// Record the token this request's generation was armed with.
    fn arm(&self, token: crate::engine_control::CancellationToken) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some(token);
        }
    }

    /// Cancel whichever generation this slot currently holds a token for, if
    /// any. Idempotent and safe to call even when nothing has armed yet.
    fn request_cancel(&self) {
        if let Ok(slot) = self.0.lock() {
            if let Some(token) = slot.as_ref() {
                token.cancel();
            }
        }
    }
}

/// Prefill chunk size armed alongside a [`CancelSlot`] cancellation token
/// (`SV-09`), so a long prompt's ingest observes the deadline instead of
/// running the whole prefill as one uninterruptible call. Small enough to
/// keep cancellation latency low, large enough not to meaningfully slow
/// down prefill throughput.
const CANCELLATION_PREFILL_CHUNK_TOKENS: usize = 512;

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
struct StreamLifecycleGuard<S> {
    inner: S,
    rate_aggregator: Arc<crate::request_metrics::RequestRateAggregator>,
    tracker: Arc<std::sync::Mutex<crate::request_metrics::RequestRateTracker>>,
    /// SV-08: the instant the *whole request* started (not just the SSE
    /// head). `chat_completions` skips its own post-`.await` observation on
    /// the streaming branch, so `Drop` here is the only place a streaming
    /// request's latency is recorded, at true stream end.
    request_start: std::time::Instant,
    /// Shared metrics registry the histogram observation above is recorded
    /// into.
    metrics: Arc<InferenceMetrics>,
    /// Decrements `active_requests` when this guard drops. Declared last so
    /// it is dropped last among this struct's own fields (Rust drops fields
    /// in declaration order) — immaterial for correctness here since the
    /// two effects are independent, but keeps the gauge decrement visibly
    /// "closest to the end" for a reader.
    _active_guard: ActiveRequestGuard,
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

/// Find the earliest occurrence of any of `stop_sequences` in `text`.
///
/// Returns `(byte_offset, matched_sequence)` for the match that starts at
/// the smallest byte offset (ties broken by the order `stop_sequences` lists
/// them, matching a linear left-to-right scan). `None` when no sequence
/// appears, or when `stop_sequences` is empty, or when a sequence is empty
/// (an empty stop string can never delimit anything and must not spuriously
/// "match" at offset 0).
fn find_first_stop_match<'a>(text: &str, stop_sequences: &'a [String]) -> Option<(usize, &'a str)> {
    stop_sequences
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()).map(|pos| (pos, s.as_str())))
        .min_by_key(|&(pos, _)| pos)
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
struct StopTracker {
    accumulated: String,
    emitted_len: usize,
    stopped: bool,
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
    fn push_and_release(
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
    fn take_remaining(&mut self) -> String {
        if self.stopped || self.emitted_len >= self.accumulated.len() {
            return String::new();
        }
        let out = self.accumulated[self.emitted_len..].to_string();
        self.emitted_len = self.accumulated.len();
        out
    }
}

#[tracing::instrument(skip(state, headers, raw), fields(request_id))]
pub(super) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    // B4: `Box<RawValue>` in place of `OpenAiJson<ChatCompletionRequest>`
    // directly — still goes through `OpenAiJson`, so a genuinely malformed
    // body (wrong content-type, unparseable JSON syntax) still hits
    // axum's `JsonRejection` -> `ApiError::from_json_rejection` exactly as
    // before (`RawValue` only requires syntactically valid JSON, not any
    // particular shape). The typed `ChatCompletionRequest` is then parsed
    // from these SAME bytes below, alongside `ChatRequestExtras` — the
    // seam that lets this package read `tools`' raw text and
    // `chat_template_kwargs` without `ChatCompletionRequest` declaring
    // them (see the module doc).
    OpenAiJson(raw): OpenAiJson<Box<serde_json::value::RawValue>>,
) -> Result<Response, ApiError> {
    let request_id = resolve_request_id(&headers);
    tracing::Span::current().record("request_id", tracing::field::display(&request_id));

    // B11/SV-11: recover what `ChatMessage.content: Option<String>` cannot
    // represent (a vision-shaped content array — flattened here or
    // honestly rejected, never silently schema-error'd) and what it has no
    // field for at all (`reasoning_content` on a replayed assistant turn)
    // from the raw body, BEFORE the typed parse below. `extras` still
    // parses from the ORIGINAL `raw.get()` text (not `rewritten`) — see
    // `preprocess_message_content_and_reasoning`'s own doc for why that
    // matters for `tools`' key order.
    let (rewritten, reasoning_contents) =
        chat_render::preprocess_message_content_and_reasoning(raw.get()).map_err(|e| {
            state.metrics.errors_total.inc();
            e.with_request_id(request_id)
        })?;
    let body: ChatCompletionRequest = match serde_json::from_str(&rewritten) {
        Ok(b) => b,
        Err(e) => {
            state.metrics.errors_total.inc();
            // Mirrors `ApiError::from_json_rejection`'s `JsonDataError`
            // shape exactly (`400`, `invalid_request_error`,
            // `invalid_request_body`) — this is "valid JSON, wrong shape
            // for `ChatCompletionRequest`", the same case axum's own
            // `Json<T>` extractor would have reported had it been asked to
            // deserialize the typed struct directly.
            return Err(ApiError::new(StatusCode::BAD_REQUEST, e.to_string())
                .with_type(crate::server::api_error::ERROR_TYPE_INVALID_REQUEST)
                .with_code("invalid_request_body")
                .with_request_id(request_id));
        }
    };
    // Best-effort: a client that sends none of these extra fields still
    // gets `ChatRequestExtras::default()` here, never a hard failure — the
    // fields are all optional and `ChatCompletionRequest`'s own
    // deserialization above already validated the body is well-formed
    // JSON matching its own shape.
    let extras: ChatRequestExtras = serde_json::from_str(raw.get()).unwrap_or_default();

    // SV-09 server wiring: this slot lets the timeout branch below cancel
    // whichever generation `chat_completions_inner` starts, once it starts
    // one — see [`CancelSlot`].
    let cancel_slot = CancelSlot::default();

    // sec-08: bound the WHOLE handler, not just the generation call. For the
    // streaming branch this covers the response head; the SSE body carries the
    // same deadline into `sse::sse_response`, so a stalled client cannot hold
    // the connection (and the engine replica's turn) open indefinitely.
    let result = match state.limits.per_request_timeout {
        Some(limit) => {
            match tokio::time::timeout(
                limit,
                chat_completions_inner(
                    Arc::clone(&state),
                    body,
                    extras,
                    reasoning_contents,
                    request_id,
                    cancel_slot.clone(),
                ),
            )
            .await
            {
                Ok(inner) => inner,
                Err(_) => {
                    // SV-09: stop the in-flight generation (if one had
                    // started) rather than letting it run to completion on a
                    // pool replica the client will never see the answer
                    // from.
                    cancel_slot.request_cancel();
                    state.metrics.errors_total.inc();
                    tracing::warn!(
                        timeout_ms = limit.as_millis() as u64,
                        "request exceeded the per-request timeout"
                    );
                    Err(ApiError::timeout(format!(
                        "request exceeded the server's per-request timeout of {} ms",
                        limit.as_millis()
                    )))
                }
            }
        }
        None => {
            chat_completions_inner(
                Arc::clone(&state),
                body,
                extras,
                reasoning_contents,
                request_id,
                cancel_slot,
            )
            .await
        }
    };

    result.map_err(|err| err.with_request_id(request_id))
}

/// The chat handler proper. Split out so [`chat_completions`] can wrap it in
/// the per-request deadline without duplicating the body.
async fn chat_completions_inner(
    state: Arc<AppState>,
    body: ChatCompletionRequest,
    extras: ChatRequestExtras,
    reasoning_contents: Vec<Option<String>>,
    request_id: RequestId,
    cancel_slot: CancelSlot,
) -> Result<Response, ApiError> {
    // SV-15(c) / SV-12 (`max_completion_tokens`): resolve the effective
    // completion budget from the request and the server's configured
    // default *before* validating, so `validate_chat_request` checks the
    // number that will actually be used.
    let effective_max_tokens = resolve_effective_max_tokens(&body, state.default_max_tokens);

    // Validate client-supplied limits/sampling params *before* touching the
    // engine or the in-flight counter, so a rejected request neither drives an
    // unbounded allocation nor unbalances `active_requests`.
    if let Err((message, param)) =
        validate_chat_request(&body, effective_max_tokens, state.max_output_tokens_ceiling)
    {
        state.metrics.requests_total.inc();
        state.metrics.errors_total.inc();
        return Err(ApiError::bad_request(message, param));
    }

    // SV-12 (`model`): when a multi-model router is attached, reject a
    // request naming a model this deployment does not serve rather than
    // silently running it on whatever the pool happens to hold. With no
    // router attached (the common single-model deployment) the field is
    // accepted and ignored, matching every real OpenAI client, which always
    // sends it.
    if let (Some(requested), Some(router)) = (body.model.as_deref(), state.model_router()) {
        let known = router.models_list().into_iter().any(|m| m.id == requested);
        if !known {
            state.metrics.requests_total.inc();
            state.metrics.errors_total.inc();
            return Err(ApiError::bad_request(
                format!("model \"{requested}\" is not served by this deployment"),
                "model",
            ));
        }
    }

    // `tools`/`tool_choice` and `logprobs` need either the complete generated
    // text (tool-call parsing) or the per-step logits (logprobs), neither of
    // which the token-by-token SSE path exposes, so combining them with
    // `stream: true` is rejected honestly with `400` — the same decision the
    // extended endpoint makes for stream + tools — instead of silently
    // dropping the incompatible field. `tool_choice: "none"` is the client
    // explicitly opting out of tool calling, so streaming stays allowed there.
    let tool_choice_none = matches!(
        body.tool_choice
            .as_ref()
            .and_then(serde_json::Value::as_str),
        Some("none")
    );
    let tools_active =
        body.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false) && !tool_choice_none;
    let want_logprobs = body.logprobs.unwrap_or(false);
    if body.stream && tools_active {
        state.metrics.requests_total.inc();
        state.metrics.errors_total.inc();
        return Err(ApiError::bad_request(
            "stream: true is not supported together with tools: tool-call parsing needs the \
             complete generated text, which isn't available until streaming finishes; omit \
             tools, set tool_choice to \"none\", or set stream to false",
            "stream",
        ));
    }
    if body.stream && want_logprobs {
        state.metrics.requests_total.inc();
        state.metrics.errors_total.inc();
        return Err(ApiError::bad_request(
            "stream: true is not supported together with logprobs: per-token logprobs are \
             captured from the non-streaming decode path only; omit logprobs or set stream to \
             false",
            "logprobs",
        ));
    }

    // Sampling overrides are resolved into a full `SamplingParams` once each
    // generation path has acquired its engine lease (gatekeeper
    // REQUIRED#1(b)): seeding from `engine.sampling_params()` needs a lease,
    // and acquiring one *here* — before the model-descriptor lookup just
    // below, which itself acquires a lease on its first (cache-cold) call —
    // would self-deadlock a single-replica pool. See [`SamplingOverrides`].
    let overrides = SamplingOverrides::from_request(&body);

    // OpenAI frequency/presence penalties are now applied for real over the
    // generated-token history (previously this endpoint silently ignored
    // them). An all-zero `PenaltyParams` is a no-op, so a request that omits
    // both penalties stays bit-identical to the previous behavior.
    let penalties = PenaltyParams::new(
        body.frequency_penalty.unwrap_or(0.0),
        body.presence_penalty.unwrap_or(0.0),
    );
    let top_logprobs = body.top_logprobs.unwrap_or(0);

    // SV-12/RT-04: stop sequences, applied by both generation paths (see
    // `find_first_stop_match` and its two call sites below).
    let stop_sequences = body
        .stop
        .map(crate::api_types::StopSequences::into_vec)
        .unwrap_or_default();
    // SV-12: deterministic seed, validated against stream/logprobs
    // incompatibility above.
    let seed = body.seed;

    let request_start = std::time::Instant::now();
    state.metrics.requests_total.inc();
    state.metrics.active_requests.inc();
    let active_guard = ActiveRequestGuard(Arc::clone(&state.metrics));

    // sec-05 (byte half): bound the raw input BEFORE sanitizing or tokenizing,
    // both of which are linear in the prompt length. This is also what keeps
    // sec-01's scan and the tokenizer off an arbitrarily large body.
    let prompt_bytes: usize = body
        .messages
        .iter()
        .filter_map(|m| m.content.as_ref())
        .map(|c| c.len())
        .sum();
    budget::validate_prompt_bytes(prompt_bytes, state.limits.max_prompt_bytes).inspect_err(
        |_| {
            state.metrics.errors_total.inc();
        },
    )?;

    // Build the prompt (B1): the model's own resolved chat template,
    // rendered through the real Jinja engine and encoded in one
    // whole-prompt call — see `chat_render`'s module doc for why one call,
    // and for how TOK-M2 (finding `sec-01`'s id-level half) is preserved
    // despite it. The no-tokenizer fallback (`None` arm) is unchanged.
    let prompt_tokens = match &state.tokenizer {
        Some(tok) => {
            let render_messages = to_render_messages(&body.messages, &reasoning_contents);
            let opts = oxibonsai_tokenizer::chat_templates::RenderOptions {
                add_generation_prompt: true,
                enable_thinking: extras.effective_enable_thinking(),
                reasoning_effort: extras.effective_reasoning_effort(),
                preserve_thinking: extras.effective_preserve_thinking(),
                add_vision_id: false,
                tools: extras.tools_raw_json(),
            };
            let (_rendered, tokens) = chat_render::render_chat_prompt(
                tok,
                &state.special_tokens,
                &render_messages,
                &opts,
                state.sanitize_prompt(),
            )
            .inspect_err(|_| {
                state.metrics.errors_total.inc();
            })?;
            tokens
        }
        // Fallback: single start token
        None => vec![151644],
    };

    // B3: resolve once from the loaded vocabulary + the actual rendered
    // prompt — never assumed from `enable_thinking` alone, since the
    // real template's OWN generation-prompt tail already closes the think
    // span again when thinking is disabled (see `started_in_think`'s doc).
    let think_close_id = state.tokenizer.as_ref().and_then(|t| t.think_close_id());
    let think_open_id = state.tokenizer.as_ref().and_then(|t| t.think_open_id());
    let started_in_think =
        chat_render::started_in_think(&prompt_tokens, think_open_id, think_close_id);

    // sec-05 (token half): reject an over-long prompt with a 400 naming the
    // real numbers instead of letting it become an opaque 500 in the engine.
    let descriptor = state.model_info().descriptor().await;
    let ctx_len = descriptor.max_context_length;
    budget::validate_request_budget(
        prompt_tokens.len(),
        effective_max_tokens,
        ctx_len,
        state.limits.max_input_tokens,
    )
    .inspect_err(|_| {
        state.metrics.errors_total.inc();
    })?;

    state
        .metrics
        .prompt_tokens_total
        .inc_by(prompt_tokens.len() as u64);

    // SV-19: feed a real per-request context-utilization pressure sample
    // into the KV cache policy so `/admin/cache-stats` reflects genuine
    // load instead of a permanent null. This deployment does not act on the
    // policy's tier decisions (RT-14 is out of this package's scope), only
    // observes and reports them — the demote-to-telemetry option that
    // finding's own fix explicitly allows.
    if ctx_len > 0 {
        let pressure = prompt_tokens.len() as f64 / ctx_len as f64;
        state.kv_cache_policy.observe(pressure);
    }

    let created = unix_now_secs();
    let model_id = descriptor.id.clone();

    // SV-08: captured before branching so the post-`.await` skip-logic
    // below can tell whether a `StreamLifecycleGuard` was actually
    // constructed (and will observe `request_duration_seconds` itself, in
    // `Drop`) without re-reading `body`.
    let is_stream = body.stream;
    let result = if is_stream {
        // ── SSE streaming mode ──
        chat_completions_stream(
            Arc::clone(&state),
            StreamRequest {
                prompt_tokens,
                max_tokens: effective_max_tokens,
                overrides,
                penalties,
                stop_sequences,
                seed,
                cancel_slot,
                include_usage: body
                    .stream_options
                    .as_ref()
                    .map(|o| o.include_usage)
                    .unwrap_or(false),
                started_in_think,
                think_close_id,
                request_id,
                request_start,
                active_guard,
            },
        )
        .await
    } else {
        // ── Non-streaming mode ──
        // `active_guard` is not moved in this branch, so it stays alive
        // (per ordinary Rust scoping) until `chat_completions_inner`
        // returns — i.e. for the whole non-streaming generation, exactly
        // as the original single-guard design did.
        chat_completions_non_stream(
            Arc::clone(&state),
            NonStreamRequest {
                prompt_tokens,
                max_tokens: effective_max_tokens,
                overrides,
                penalties,
                stop_sequences,
                seed,
                cancel_slot,
                tools_active,
                want_logprobs,
                top_logprobs,
                started_in_think,
                think_close_id,
                request_id,
                created,
                model_id,
            },
        )
        .await
    };

    // SV-08: a *successful* stream already carries `request_start` into its
    // `StreamLifecycleGuard`, which observes the real latency in `Drop` at
    // true stream end -- observing it here too would under-report it
    // (near-zero) and double-count the sample. A stream that fails *before*
    // the guard exists (e.g. no engine replica available) has no such
    // guard, so it is still observed here, like a non-streaming request.
    let already_covered_by_stream_guard = is_stream && result.is_ok();
    if !already_covered_by_stream_guard {
        let elapsed = request_start.elapsed().as_secs_f64();
        state.metrics.request_duration_seconds.observe(elapsed);
    }

    if result.is_err() {
        state.metrics.errors_total.inc();
    }

    result
}

/// Bundled inputs for a single non-streaming chat completion.
///
/// Grouped into one struct so [`chat_completions_non_stream`] stays within
/// clippy's argument-count budget now that the base endpoint honors penalties,
/// tool calling, and logprobs in addition to the original sampling params.
/// Bundled inputs for one streaming chat completion.
struct StreamRequest {
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    overrides: SamplingOverrides,
    penalties: PenaltyParams,
    /// Stop sequences (`SV-12`/`RT-04`), checked incrementally against the
    /// accumulated decoded text; a match truncates the emitted content and
    /// cancels the in-flight generation early.
    stop_sequences: Vec<String>,
    /// Deterministic sampling seed. Streaming + seed is rejected with `400`
    /// by `validate_chat_request` (no streaming generation seam accepts a
    /// per-call seed), so this is always `None` by the time it reaches
    /// [`chat_completions_stream`] — carried through anyway so the type
    /// documents the (currently unreachable) combination rather than
    /// silently dropping the field earlier in the pipeline.
    seed: Option<u64>,
    /// SV-09 server wiring: armed with a fresh [`crate::engine_control::CancellationToken`]
    /// once the engine lease is acquired, so the outer per-request timeout
    /// can cancel this generation early.
    cancel_slot: CancelSlot,
    /// Whether the client set `stream_options.include_usage`, which adds the
    /// final usage chunk before `[DONE]` (finding `sec-08`).
    include_usage: bool,
    /// B3: whether the rendered+encoded prompt ends inside an open
    /// `<think>` span — see [`chat_render::started_in_think`].
    started_in_think: bool,
    /// B3: this vocabulary's `</think>` token id, or `None` for a model
    /// whose vocabulary defines no `<think>` at all (RT-10).
    think_close_id: Option<u32>,
    request_id: RequestId,
    /// SV-08: the whole request's start instant, carried into
    /// [`StreamLifecycleGuard`] so the latency histogram is observed at
    /// true stream end (or early disconnect) rather than at SSE-head
    /// construction time.
    request_start: std::time::Instant,
    /// SV-08: kept alive for the whole SSE body (moved into
    /// [`StreamLifecycleGuard`]), not just until the initial response head
    /// is constructed.
    active_guard: ActiveRequestGuard,
}

struct NonStreamRequest {
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    overrides: SamplingOverrides,
    penalties: PenaltyParams,
    /// Stop sequences (`SV-12`/`RT-04`), applied by truncating the decoded
    /// text at the first match (see [`chat_completions_non_stream`]'s doc
    /// comment for why this is post-hoc rather than incremental on this
    /// path).
    stop_sequences: Vec<String>,
    /// Deterministic sampling seed (`SV-12`), honored via
    /// [`crate::engine::InferenceEngine::generate_with_seed`] when present
    /// and `logprobs` was not requested (the two are mutually exclusive —
    /// enforced by `validate_chat_request`).
    seed: Option<u64>,
    /// SV-09 server wiring: see [`StreamRequest::cancel_slot`].
    cancel_slot: CancelSlot,
    /// Whether tool-call parsing should run on the generated text (the client
    /// supplied a non-empty `tools` list and did not set `tool_choice: "none"`).
    tools_active: bool,
    /// Whether to capture and return per-token logprobs.
    want_logprobs: bool,
    /// Number of top alternatives to report per token (`top_logprobs`, 0..=20).
    top_logprobs: usize,
    /// B3: whether the rendered+encoded prompt ends inside an open
    /// `<think>` span — see [`chat_render::started_in_think`].
    started_in_think: bool,
    /// B3: this vocabulary's `</think>` token id, or `None` for a model
    /// whose vocabulary defines no `<think>` at all (RT-10).
    think_close_id: Option<u32>,
    request_id: RequestId,
    /// Unix timestamp for the response's `created` field (`SV-03`).
    created: u64,
    /// The served model's id for the response's `model` field (`SV-03`).
    model_id: String,
}

/// The outcome of [`parse_base_tool_calls`] — a base-endpoint-shaped mirror
/// of [`crate::tool_calling::ToolCallParseOutcome`] carrying OpenAI
/// [`crate::api_types::ToolCallResult`]s instead of the lower-level
/// [`crate::api_types::ToolCall`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BaseToolCallOutcome {
    /// No tool-call markup found; the whole text is ordinary content.
    None,
    /// One or more complete tool calls were found, plus whatever
    /// natural-language text preceded the first one.
    Found {
        leading_text: String,
        calls: Vec<crate::api_types::ToolCallResult>,
    },
    /// A `<tool_call>` was opened but never closed — the model was cut off.
    /// Carries the (non-tool-call) text that preceded it.
    Truncated { leading_text: String },
}

/// Parse a tool call out of generated assistant text, reusing the same
/// `<tool_call>...</tool_call>` parser the extended endpoint uses.
///
/// Returns [`BaseToolCallOutcome::None`] when tool calling is inactive (no
/// `tools` supplied, or `tool_choice: "none"`) or when the text contains no
/// parseable tool-call block. This is the seam that stops the base
/// `/v1/chat/completions` endpoint from silently discarding an advertised
/// `tools` field (finding `serve-api-03`).
///
/// B2-13/RT-11: was JSON-only (`crate::api_types::parse_tool_call`), so
/// Bonsai 2's `<tool_call><function=NAME>…</function></tool_call>` XML
/// shape (design §5.4) came back as `None` and its raw XML rendered
/// verbatim as ordinary assistant content. Now tries
/// [`crate::tool_calling::parse_tool_calls`] (XML-first, legacy-JSON
/// fallback) and collects every call found, not just the first — matching
/// `finish_reason: "tool_calls"` semantics for a model that emits several
/// calls in one turn.
///
/// B6 (RT-11 / design §5.4 "text before the first `<tool_call>` is kept"):
/// [`BaseToolCallOutcome::Found`] carries `leading_text` — the caller must
/// use its trimmed, non-empty value as `message.content` instead of
/// discarding it, matching the extended endpoint's own behavior (the two
/// endpoints previously disagreed).
pub(crate) fn parse_base_tool_calls(content: &str, tools_active: bool) -> BaseToolCallOutcome {
    if !tools_active {
        return BaseToolCallOutcome::None;
    }
    match crate::tool_calling::parse_tool_calls(content) {
        crate::tool_calling::ToolCallParseOutcome::Found {
            leading_text,
            calls,
        } => BaseToolCallOutcome::Found {
            leading_text,
            calls: calls
                .into_iter()
                .map(|tc| {
                    crate::api_types::ToolCallResult::new_function(
                        tc.id,
                        tc.function.name,
                        tc.function.arguments,
                    )
                })
                .collect(),
        },
        crate::tool_calling::ToolCallParseOutcome::Truncated { leading_text } => {
            BaseToolCallOutcome::Truncated { leading_text }
        }
        crate::tool_calling::ToolCallParseOutcome::None => BaseToolCallOutcome::None,
    }
}

/// Decode a single token id into a display string for OpenAI logprobs
/// (`TOK-M1`).
///
/// Uses [`TokenizerBridge::piece`] — the raw byte-level vocabulary entry,
/// unmapped through the byte-level alphabet — rather than
/// [`TokenizerBridge::decode`] on a one-element slice. `decode` corrupts a
/// token that is only part of a multi-byte UTF-8 sequence (verified against
/// the HF oracle: `"日本語処理"`'s middle byte-fragment tokens came back as
/// the literal placeholder string `"<id>"` via the old per-id `decode`
/// call). `piece` + `from_utf8_lossy` cannot fully solve the general case
/// either — a lone byte fragment is not valid UTF-8 on its own, so it still
/// renders as `U+FFFD` — but it never fabricates a wrong *value* the way
/// `decode`'s failure path did, and every token whose piece *is* complete,
/// valid UTF-8 (the overwhelming majority, including every whole-CJK-word
/// and whole-emoji token) decodes correctly. See this package's recorded
/// deviations for the `bytes`-field caveat this implies.
fn logprob_id_to_token(tokenizer: Option<&TokenizerBridge>, id: u32) -> String {
    match tokenizer {
        Some(tok) => String::from_utf8_lossy(&tok.piece(id)).into_owned(),
        None => format!("<{id}>"),
    }
}

/// TOK-M1, now closed in full (B2-13): correct **every** `bytes` field --
/// the chosen token's on each [`crate::api_types::LogprobsContent`] *and*
/// every `top_logprobs` alternative's -- using the real raw vocabulary
/// bytes instead of `crate::api_types::compute_logprobs`'s necessarily
/// lossy display-string derivation.
///
/// `compute_logprobs` derives `bytes` from the *display string* -- correct
/// only when that string is a complete, valid UTF-8 token. A byte-fragment
/// token (part of a multi-byte character split across several vocabulary
/// entries) renders as `U+FFFD`, so `bytes` came back as `U+FFFD`'s own
/// 3-byte encoding instead of the token's real raw byte(s) -- verified
/// against the HF oracle on `"日本語処理"`'s middle byte-fragment tokens.
///
/// An earlier revision of this function only fixed the *chosen* token,
/// zipping a separately-tracked `output_tokens: &[u32]` against the content
/// Vec because neither `LogprobsContent` nor `TopLogprob` carried their own
/// id -- so an alternative's id was unrecoverable by the time it reached
/// this function. `id` now lives on both types directly (`api_types.rs`,
/// this package's file), so this delegates to the shared
/// [`crate::api_types::fix_logprob_bytes`], which fixes every entry from
/// its own embedded id and cannot desync the way a parallel-array zip
/// could.
fn correct_logprob_bytes(
    mut content: Vec<crate::api_types::LogprobsContent>,
    tokenizer: Option<&TokenizerBridge>,
) -> Vec<crate::api_types::LogprobsContent> {
    let Some(tok) = tokenizer else {
        return content;
    };
    crate::api_types::fix_logprob_bytes(&mut content, &|id| tok.piece(id));
    content
}

/// Non-streaming chat completion handler.
async fn chat_completions_non_stream(
    state: Arc<AppState>,
    req: NonStreamRequest,
) -> Result<Response, ApiError> {
    let NonStreamRequest {
        prompt_tokens,
        max_tokens,
        overrides,
        penalties,
        stop_sequences,
        seed,
        cancel_slot,
        tools_active,
        want_logprobs,
        top_logprobs,
        started_in_think,
        think_close_id,
        request_id,
        created,
        model_id,
    } = req;
    let prompt_len = prompt_tokens.len();

    let mut lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        ApiError::service_unavailable(format!("no inference engine replica is available: {e}"))
            .with_code("engine_unavailable")
    })?;

    // gatekeeper REQUIRED#1(b): seed from the engine's actual running
    // configuration, never `SamplingParams::default()`.
    let seeded = lease.sampling_params().clone();
    let params = overrides.apply(&seeded);

    // SV-09 server wiring: arm cancellation now that generation is actually
    // about to start, and expose the token to the outer per-request timeout
    // via `cancel_slot`. Chunking prefill lets a long prompt's ingest also
    // observe the deadline instead of running as one uninterruptible call.
    let cancel_token = lease.arm_cancellation();
    cancel_slot.arm(cancel_token);
    lease.set_prefill_chunk_tokens(Some(CANCELLATION_PREFILL_CHUNK_TOKENS));

    // sec-03: generation is synchronous, CPU-bound work and must not run on a
    // tokio worker thread. `run_blocking_generation` moves the lease onto the
    // blocking pool, resets the engine first (RT-03: no KV contamination
    // across requests) and returns the replica to the pool when it finishes.
    //
    // `logprobs` generation has no per-call params argument of its own; the
    // closure below swaps `params` onto the engine's sampler for that one
    // call and restores it afterward (`RT-26`).
    let state_for_generation = Arc::clone(&state);
    let generated = run_blocking_generation(lease, move |lease| {
        let prev_penalties = lease.penalties();
        lease.set_penalties(penalties);
        let result = if want_logprobs {
            // RT-26: `lease.sampler` (`pub(crate)`, same pattern
            // `prefix_cache_engine.rs::decode_loop` uses) is swapped and
            // restored unconditionally, never via `?` — an early return
            // here would leak this request's params onto the next one
            // served by this pool replica.
            let prev_params = lease.sampler.params().clone();
            lease.sampler.set_params(params.clone());
            let id_to_token = |id: u32| -> String {
                logprob_id_to_token(state_for_generation.tokenizer.as_ref(), id)
            };
            let r = lease
                .generate_with_logprobs(&prompt_tokens, max_tokens, top_logprobs, &id_to_token)
                .map(|(tokens, lp)| (tokens, Some(lp)));
            lease.sampler.set_params(prev_params);
            r
        } else if let Some(seed) = seed {
            // SV-12: deterministic seed. `generate_with_seed` swaps in a
            // fresh, seeded sampler carrying over whatever penalties are
            // *currently* set on the engine (just set above), runs
            // generation, then restores the previous sampler.
            lease
                .generate_with_seed(&prompt_tokens, max_tokens, seed, &params)
                .map(|tokens| (tokens, None))
        } else {
            lease
                .generate_with_params_and_penalties(&prompt_tokens, max_tokens, &params, &penalties)
                .map(|tokens| (tokens, None))
        };
        lease.set_penalties(prev_penalties);
        result
    })
    .await?;

    let (output_tokens, logprobs_content) = generated.map_err(|e| {
        tracing::error!(error = %e, "generation failed");
        ApiError::internal(format!("generation failed: {e}"))
    })?;

    let completion_len = output_tokens.len();

    // Record token metrics
    state
        .metrics
        .tokens_generated_total
        .inc_by(completion_len as u64);

    // Decode, per-token (not a single batch `decode` call): B3 needs each
    // token's own id alongside its decoded piece to classify it into the
    // `reasoning_content`/`content` channels
    // ([`crate::reasoning::split_reasoning`]) — `step_decode`'s UTF-8-safe
    // windowing means a piece can still legitimately represent more than
    // one id (a multi-byte character split across tokens); which of that
    // piece's own ids gets carried alongside it does not affect
    // classification, since only the SINGLE-TOKEN `</think>` id itself is
    // ever compared against.
    let (reasoning_content, content) = if let Some(tok) = &state.tokenizer {
        let mut decode_state = tok.new_decode_stream(true);
        let mut pieces: Vec<(u32, String)> = Vec::with_capacity(output_tokens.len());
        for &id in &output_tokens {
            let piece = tok.step_decode(&mut decode_state, id).map_err(|e| {
                tracing::error!(error = %e, "decoding the generated tokens failed");
                ApiError::internal(format!("failed to decode the generated tokens: {e}"))
            })?;
            // Post-verifier-review fix: a token flagged `special` in the
            // vocabulary (real chat-model `<|...|>`-family markers commonly
            // are, and `<think>`/`</think>` themselves could be) decodes to
            // `Ok(None)` — no bytes of its own — but it must still reach
            // `split_reasoning` below by id, or a special-flagged
            // `</think>` would never be seen at all and every token after
            // it would stay misclassified as reasoning for the rest of the
            // response. `unwrap_or_default()` keeps every id in `pieces`
            // (an empty piece classifies to an empty, harmless chunk).
            pieces.push((id, piece.unwrap_or_default()));
        }
        crate::reasoning::split_reasoning(
            pieces.iter().map(|(id, p)| (*id, p.as_str())),
            started_in_think,
            think_close_id,
        )
    } else {
        (None, format!("{output_tokens:?}"))
    };

    // SV-12/RT-04: stop sequences, applied to the CONTENT channel only —
    // B3/RT-10 correction (`reasoning.rs`'s own doc): evaluating this
    // BEFORE the reasoning split would let a stop string occurring inside
    // the reasoning span truncate the real answer before it is even
    // reached. Applied post-hoc — the engine returns the
    // complete text in one shot on this path, so there is no incremental
    // hook to stop the *model* early (that would need an
    // `InferenceEngine`-level per-token text callback, outside this
    // package's owned files; see recorded deviations) — but the *returned*
    // text is truncated at the first match exactly like a real incremental
    // stop would produce, so the client-visible contract is honored even
    // though `completion_tokens` still reflects every token the engine
    // actually computed (including the ones whose text is discarded here).
    let (content, stopped) = match find_first_stop_match(&content, &stop_sequences) {
        Some((pos, _matched)) => (content[..pos].to_string(), true),
        None => (content, false),
    };

    // Tool calling: when the client supplied tools (and did not opt out via
    // `tool_choice: "none"`), parse the generated text for a `<tool_call>` block
    // using the same machinery as `/v1/chat/completions/extended` instead of
    // silently dropping the advertised `tools` field (finding `serve-api-03`).
    //
    // B6: a `Found` result's `leading_text` — the model's natural-language
    // preamble before the call — becomes `message.content` (trimmed,
    // `None` when empty) instead of being discarded; a `Truncated` result
    // (opened but never closed) similarly keeps its own leading text as
    // content and forces `finish_reason: "length"` rather than reporting
    // the partial `<tool_call>` XML as if it were prose (the model was cut
    // off, it did not finish normally).
    let (message_content, tool_calls, forced_length) =
        match parse_base_tool_calls(&content, tools_active) {
            BaseToolCallOutcome::Found {
                leading_text,
                calls,
            } => {
                let trimmed = leading_text.trim();
                let content = (!trimmed.is_empty()).then(|| trimmed.to_string());
                (content, Some(calls), false)
            }
            BaseToolCallOutcome::Truncated { leading_text } => {
                let trimmed = leading_text.trim();
                let content = (!trimmed.is_empty()).then(|| trimmed.to_string());
                (content, None, true)
            }
            BaseToolCallOutcome::None => (Some(content), None, false),
        };
    let has_tool_calls = tool_calls.is_some();

    // Honest finish_reason: a parsed tool call wins; a truncated tool call
    // is an honest "length" (the model was cut off mid-call); a
    // stop-sequence match wins next; otherwise report "length" when
    // generation was truncated at max_tokens and "stop" when it ended
    // naturally on EOS (finding `serve-api-01`).
    let finish_reason = if has_tool_calls {
        "tool_calls".to_string()
    } else if forced_length {
        "length".to_string()
    } else if stopped {
        "stop".to_string()
    } else if completion_len >= max_tokens {
        "length".to_string()
    } else {
        "stop".to_string()
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", rand_id()),
        object: "chat.completion".to_string(),
        created,
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: message_content,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
            // TOK-M1 (closed in full): fix up every entry's -- and every
            // alternative's -- `bytes` using each one's own embedded raw
            // vocabulary id, rather than shipping api_types.rs's
            // lossy-string-derived bytes for a byte-fragment token.
            logprobs: logprobs_content.map(|content| crate::api_types::ChoiceLogprobs {
                content: Some(correct_logprob_bytes(content, state.tokenizer.as_ref())),
            }),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: completion_len,
            total_tokens: prompt_len + completion_len,
        },
    };

    // B3: `reasoning_content` has no field on `ChatMessage` (`server.rs`,
    // owned by ENGINE-SEAM this wave) — patched into the serialized
    // response's `choices[0].message` object instead (see the module doc).
    let mut response_json = serde_json::to_value(&response)
        .map_err(|e| ApiError::internal(format!("failed to serialize the response: {e}")))?;
    if let Some(reasoning) = reasoning_content {
        if let Some(obj) = response_json
            .pointer_mut("/choices/0/message")
            .and_then(|m| m.as_object_mut())
        {
            obj.insert(
                "reasoning_content".to_string(),
                serde_json::Value::String(reasoning),
            );
        }
    }

    let headers = request_id_header_map(request_id);
    Ok((headers, Json(response_json)).into_response())
}

/// Build the JSON payload for the terminal SSE event of a streaming chat
/// completion, from the real generation `outcome`.
///
/// * `Ok(generated)` → an ordinary finish chunk whose `finish_reason` is
///   `"length"` when generation was truncated at `max_tokens` and `"stop"`
///   otherwise (honest truncation signaling — finding `serve-api-01`).
/// * `Err(message)` → an OpenAI-style error object, so a mid-stream failure
///   (including a prefill error that emitted zero tokens) is surfaced to the
///   client instead of being masked as a clean, complete response (finding
///   `serve-api-02`).
fn stream_terminal_json(
    outcome: Result<usize, String>,
    max_tokens: usize,
    id: &str,
    created: u64,
    model: &str,
    include_usage: bool,
) -> String {
    match outcome {
        Ok(generated) => {
            let finish_reason = if generated >= max_tokens {
                "length"
            } else {
                "stop"
            };
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
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
                usage: usage_placeholder(include_usage),
            };
            serde_json::to_string(&finish_chunk).unwrap_or_default()
        }
        Err(message) => {
            tracing::error!(error = %message, "streaming generation failed mid-stream");
            ApiError::internal(message).to_json().to_string()
        }
    }
}

/// `Some(null)` when the client asked for usage (OpenAI puts an explicit
/// `"usage": null` on every non-final chunk in that mode), `None` — i.e. the
/// member is omitted entirely — otherwise.
fn usage_placeholder(include_usage: bool) -> Option<serde_json::Value> {
    include_usage.then_some(serde_json::Value::Null)
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

/// Everything the SSE tail emits for one generation outcome: the finish (or
/// error) chunk, followed by the usage chunk when the client asked for it and
/// generation succeeded.
fn stream_terminal_payloads(
    outcome: Result<usize, String>,
    max_tokens: usize,
    id: &str,
    created: u64,
    model: &str,
    include_usage: bool,
    prompt_tokens: usize,
) -> Vec<String> {
    let generated = outcome.as_ref().ok().copied();
    let mut payloads = vec![stream_terminal_json(
        outcome,
        max_tokens,
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

/// SSE streaming chat completion handler.
async fn chat_completions_stream(
    state: Arc<AppState>,
    req: StreamRequest,
) -> Result<Response, ApiError> {
    let StreamRequest {
        prompt_tokens,
        max_tokens,
        overrides,
        penalties,
        stop_sequences,
        seed: _seed, // always None here: stream + seed is rejected by validate_chat_request.
        cancel_slot,
        include_usage,
        started_in_think,
        think_close_id,
        request_id,
        request_start,
        active_guard,
    } = req;
    let prompt_len = prompt_tokens.len();
    let completion_id = format!("chatcmpl-{}", rand_id());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Report the real loaded model id in the streaming chunks (resolved once and
    // cached) instead of a hard-coded literal.
    let model_id = state.model_info().descriptor().await.id;

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    // Carries the terminal payloads out of the blocking task: the finish chunk
    // (so "stop" vs "length" is honest — finding `serve-api-01`), an error
    // object when generation failed mid-stream (finding `serve-api-02`), and
    // the usage chunk when the client set `stream_options.include_usage`
    // (finding `sec-08`).
    let (terminal_tx, terminal_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Acquire an engine lease in async context, then move it into the blocking
    // generation task. The lease's Drop (a synchronous std-mutex push) runs at
    // the closure's end — no async in Drop, so this is safe off the runtime.
    let mut lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        ApiError::service_unavailable(format!("no inference engine replica is available: {e}"))
            .with_code("engine_unavailable")
    })?;

    // gatekeeper REQUIRED#1(b): seed from the engine's actual running
    // configuration, never `SamplingParams::default()`.
    let params = overrides.apply(lease.sampling_params());

    // SV-09 server wiring: arm cancellation, and keep a second clone in
    // async scope so a stop-sequence match detected on the receiving side
    // (below) can also cut generation short, not just the outer
    // per-request timeout.
    let cancel_token = lease.arm_cancellation();
    cancel_slot.arm(cancel_token.clone());
    lease.set_prefill_chunk_tokens(Some(CANCELLATION_PREFILL_CHUNK_TOKENS));

    let metrics_for_task = Arc::clone(&state.metrics);
    let id_for_task = completion_id.clone();
    let model_for_task = model_id.clone();
    tokio::task::spawn_blocking(move || {
        // RT-03: start from a clean KV cache so this request cannot inherit
        // state from the previous one served by this pool replica.
        lease.reset();
        // Apply OpenAI frequency/presence penalties for the duration of this
        // run, restoring the engine's previous penalties before the lease drops
        // so they don't leak to the next request served by this pool replica.
        let prev_penalties = lease.penalties();
        lease.set_penalties(penalties);
        let result =
            lease.generate_streaming_with_params(&prompt_tokens, max_tokens, &params, &token_tx);
        lease.set_penalties(prev_penalties);

        // SV-08: streaming requests were previously invisible to
        // `tokens_generated_total`/`errors_total` — recorded here (from the
        // real outcome the engine reported), mirroring the non-streaming
        // path's bookkeeping instead of leaving both counters untouched.
        match &result {
            Ok(count) => metrics_for_task
                .tokens_generated_total
                .inc_by(*count as u64),
            Err(_) => metrics_for_task.errors_total.inc(),
        }

        // Report the outcome to the SSE tail. A dropped receiver is harmless.
        for payload in stream_terminal_payloads(
            result.map_err(|e| e.to_string()),
            max_tokens,
            &id_for_task,
            created,
            &model_for_task,
            include_usage,
            prompt_len,
        ) {
            let _ = terminal_tx.send(payload);
        }
        // lease (and thus token_tx) is dropped here: the engine returns to the
        // pool and the channel closes.
    });

    // Build SSE stream from the token receiver
    let id_for_stream = completion_id;
    let state_for_stream = Arc::clone(&state);

    // First, send a role delta
    let role_chunk = ChatCompletionChunk {
        id: id_for_stream.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_id.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
        usage: usage_placeholder(include_usage),
    };

    let role_event = match serde_json::to_string(&role_chunk) {
        Ok(json) => json,
        Err(e) => {
            tracing::error!(error = %e, "failed to serialize the SSE role chunk");
            return Err(ApiError::internal(
                "failed to serialize the streaming response",
            ));
        }
    };

    let id_clone = id_for_stream.clone();
    let model_for_stream = model_id.clone();

    // Convert token receiver into a stream of SSE events
    let token_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(token_rx);

    // Per-request streaming-decode state.  BPE tokens may straddle UTF-8
    // codepoint boundaries (CJK, emoji), so we buffer through HF's
    // step_decode_stream and only emit a chunk when a complete UTF-8 piece is
    // ready.  Mid-codepoint tokens yield `Ok(None)` and are filtered out.
    let mut stream_state = state_for_stream
        .tokenizer
        .as_ref()
        .map(|t| t.new_decode_stream(true));

    // SV-19: per-request rate tracker, shared with `StreamLifecycleGuard`
    // (below) which records the final snapshot into the admin aggregator
    // when the stream ends. Tracked here (the async receiving side) rather
    // than inside the blocking generation task because that is where each
    // token becomes observable to *this* handler at all — the engine call
    // that produces them offers no per-token callback across the
    // process/thread boundary.
    let tracker = Arc::new(std::sync::Mutex::new(
        crate::request_metrics::RequestRateTracker::new(),
    ));
    if let Ok(mut t) = tracker.lock() {
        t.record_admission();
    }
    let tracker_for_stream = Arc::clone(&tracker);

    // SV-12/RT-04: stop sequences, checked against the accumulated decoded
    // text as it arrives.
    //
    // A naive "check `accumulated`, emit whatever this token decoded to"
    // scheme leaks a stop sequence that happens to split across two decode
    // chunks: chunk N-1 emits "ST", chunk N would emit "OP", and only once
    // "OP" arrives does "STOP" become visible in `accumulated` — but "ST"
    // was already sent to the client, and SSE has no take-backs (this is
    // finding `RT-06`'s bug class, for the chunk boundary rather than the
    // UTF-8-codepoint one `step_decode` already handles). The fix is the
    // same one RT-06 prescribes: hold back the last `max_stop_len - 1`
    // bytes of `accumulated` — never emit a suffix that could still be the
    // unfinished start of a stop sequence — and only release bytes once a
    // longer window has passed without them completing a match.
    let max_stop_len = stop_sequences
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.len())
        .max()
        .unwrap_or(0);
    // Shared with the flush step below (built after `content_stream` ends
    // without ever matching): the holdback window means up to
    // `max_stop_len - 1` already-safe bytes can still be sitting unreleased
    // when generation ends naturally (EOS / `max_tokens`), and they must
    // still reach the client rather than being silently dropped.
    let stop_state = Arc::new(std::sync::Mutex::new(StopTracker::default()));
    let stop_state_for_stream = Arc::clone(&stop_state);

    // B3: classifies each token's decoded piece into `reasoning_content`
    // vs `content` — fed BEFORE stop-sequence tracking (see that struct's
    // module doc: a stop string occurring inside the reasoning span must
    // never truncate the real answer before it is even reached, and
    // `stop` describes the visible answer, not the reasoning trace).
    let mut reasoning_splitter =
        crate::reasoning::ReasoningSplitter::new(started_in_think, think_close_id);

    let content_stream = token_stream.filter_map(move |token_id| {
        if let Ok(mut t) = tracker_for_stream.lock() {
            if t.tokens_emitted() == 0 {
                t.record_first_token();
            } else {
                t.record_token();
            }
        }

        let already_stopped = stop_state_for_stream
            .lock()
            .map(|s| s.stopped)
            .unwrap_or(false);
        if already_stopped {
            return None;
        }

        let text = match (&state_for_stream.tokenizer, stream_state.as_mut()) {
            (Some(tok), Some(state)) => match tok.step_decode(state, token_id) {
                Ok(Some(txt)) => txt,
                // Post-verifier-review fix: must NOT short-circuit before
                // `reasoning_splitter.push` sees this id — a special-flagged
                // `</think>` decodes to exactly this, and skipping the push
                // would leave the splitter stuck in reasoning for the rest
                // of the stream. An empty piece classifies to an empty,
                // harmless chunk in every `Phase` (see `reasoning.rs::push`).
                Ok(None) => String::new(),
                Err(_) => format!("[{token_id}]"),
            },
            _ => format!("[{token_id}]"),
        };

        let (is_reasoning, text) = match reasoning_splitter.push(token_id, &text) {
            crate::reasoning::ReasoningChunk::Reasoning(s) => (true, s),
            crate::reasoning::ReasoningChunk::Content(s) => (false, s),
            // The `</think>` token itself, or a swallowed post-boundary
            // newline: neither channel shows it (see `reasoning.rs`'s
            // module doc's "post-boundary gap").
            crate::reasoning::ReasoningChunk::Boundary => return None,
        };

        if is_reasoning {
            if text.is_empty() {
                return None;
            }
            let chunk = ChatCompletionChunk {
                id: id_clone.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_for_stream.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: None,
                }],
                usage: usage_placeholder(include_usage),
            };
            return Some(chat_render::with_extra_delta_field(
                &chunk,
                "/choices/0/delta",
                "reasoning_content",
                text,
            ));
        }

        let text = if stop_sequences.is_empty() {
            text
        } else {
            let Ok(mut state) = stop_state_for_stream.lock() else {
                return None;
            };
            let released = state.push_and_release(&text, &stop_sequences, max_stop_len);
            if state.stopped {
                cancel_token.cancel();
            }
            released
        };

        if text.is_empty() {
            // Either nothing is safe to release yet (still within the
            // holdback window), this chunk's text was entirely consumed by
            // a just-detected stop boundary, or the decoder legitimately
            // produced an empty piece — either way, an empty content delta
            // would serve no purpose.
            return None;
        }

        let chunk = ChatCompletionChunk {
            id: id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_for_stream.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(text),
                },
                finish_reason: None,
            }],
            usage: usage_placeholder(include_usage),
        };

        Some(serde_json::to_string(&chunk).unwrap_or_default())
    });

    // Release whatever the holdback window above is still sitting on, once
    // `content_stream` is exhausted (`.chain` only polls this after that),
    // *provided* no stop sequence ever matched (a match already released
    // everything it safely could and marked nothing more should follow).
    // `std::iter::once_with` defers evaluation to the first (only) poll, so
    // this genuinely runs "at content_stream's end", not eagerly when the
    // chain is assembled.
    let id_for_flush = id_for_stream.clone();
    let model_for_flush = model_id.clone();
    let flush_stream = tokio_stream::iter(std::iter::once_with(move || {
        let mut state = stop_state.lock().ok()?;
        let leftover = state.take_remaining();
        if leftover.is_empty() {
            return None;
        }
        let chunk = ChatCompletionChunk {
            id: id_for_flush,
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_for_flush,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(leftover),
                },
                finish_reason: None,
            }],
            usage: usage_placeholder(include_usage),
        };
        Some(serde_json::to_string(&chunk).unwrap_or_default())
    }))
    .filter_map(|item| item);

    // Terminal events, derived from the real generation outcome and produced by
    // the blocking task itself:
    //   * `Ok(count)` → an ordinary finish chunk whose `finish_reason` is
    //     "length" when generation was truncated at `max_tokens` and "stop"
    //     otherwise (honest truncation signaling — finding `serve-api-01`;
    //     a stop-sequence match also reports "stop" here, since cancelling
    //     generation means `count < max_tokens`),
    //     optionally followed by the usage chunk (finding `sec-08`).
    //   * `Err(message)` → an OpenAI-style error object, so a mid-stream failure
    //     (including a prefill error that emitted zero tokens) is no longer
    //     masked as a clean, complete response (finding `serve-api-02`).
    let finish_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(terminal_rx);

    // Prepend role event; `sse::sse_response` appends `[DONE]`, applies the
    // keep-alive interval and enforces the per-request deadline over the whole
    // body (finding `sec-08`).
    let role_stream = tokio_stream::once(role_event);
    let full_stream = role_stream
        .chain(content_stream)
        .chain(flush_stream)
        .chain(finish_stream);

    // SV-08/SV-19: keep the in-flight gauge decremented (and the rate
    // sample + latency histogram recorded) at the true end of this stream's
    // life — completion *or* early client disconnect — rather than at the
    // moment this function returns the initial `Response` object, which
    // happens long before the client has seen any generated text.
    let guarded_stream = StreamLifecycleGuard {
        inner: full_stream,
        rate_aggregator: Arc::clone(&state.rate_aggregator),
        tracker,
        request_start,
        metrics: Arc::clone(&state.metrics),
        _active_guard: active_guard,
    };

    Ok(sse::sse_response(
        guarded_stream,
        &state.limits,
        request_id_header_map(request_id),
    ))
}

/// Generate a short random-ish ID for completion responses.
fn rand_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{ts:x}")
}

#[cfg(test)]
#[path = "chat_tests.rs"]
mod tests;
