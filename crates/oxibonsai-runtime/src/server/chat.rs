//! The `/v1/chat/completions` request pipeline: validation, sampling-params
//! resolution, non-streaming and SSE-streaming generation.
//!
//! Split out of `server.rs` (which stays the single assembly/route/type
//! definition point) to keep both files under the workspace's 2000-line
//! ceiling — every item here is `server`-private (or `pub(super)` for the one
//! route handler `create_router_full` registers).
//!
//! `use super::*;` pulls in `server.rs`'s own imports (axum types,
//! `AppState`, the request/response/chunk types, `RequestId`, `SamplingParams`
//! / `PenaltyParams`, `TokenizerBridge`, the `budget`/`sanitize`/`sse`/
//! `blocking` sibling submodules, …).
//!
//! # Prompt, sampling, response
//!
//! * **Prompt** — the model's own resolved chat template, rendered through
//!   the real Jinja engine ([`chat_render::render_chat_prompt`]);
//!   `chat_template_kwargs` / `enable_thinking` / `reasoning_effort` and the
//!   raw `tools` text are read from the raw request body
//!   ([`ChatRequestExtras`], parsed from the same bytes as the typed
//!   `ChatCompletionRequest`, which keeps the client's own key order).
//!   Without a tokenizer there is nothing to render into: the configured
//!   prompt start token, or `400 tokenizer_required`.
//! * **Sampling** — the request's resolved params, penalties, `min_p` and
//!   `seed` are installed on the leased replica for the one generation
//!   ([`RequestSampling`]), streamed or not, `logprobs` included.
//! * **Response** — every generated token goes through one
//!   [`ResponsePipeline`] (decode → `<think>` split → tool-call extraction
//!   → stop sequences): the non-streaming path collects it, the streaming
//!   path sends each released event as an SSE chunk, so both report the
//!   same `reasoning_content`, `content`, `tool_calls` and `finish_reason`.
//!   Tool-call openers are keyed off the vocabulary's own `<tool_call>` id
//!   (design App. A.1), so the literal text `<tool_call>` spelled out of
//!   ordinary tokens is never a call. A token the decoder cannot render
//!   never becomes id syntax, and without a tokenizer the content is empty.

use super::response_pipeline::{
    CollectedResponse, ContentStop, GenerationOutcome, ResponsePipeline, ResponseShape,
    StreamChunks, StreamDriver, StreamToolCallDelta,
};
use super::sampling_scope::RequestSampling;
use super::*;
use crate::tokenizer_bridge::chat_render::{self, to_render_messages, ChatRequestExtras};

/// The SSE-streaming half of this pipeline, in its own file to keep both
/// under the workspace's 2000-line ceiling.
#[path = "chat_stream.rs"]
mod stream;

use stream::{chat_completions_stream, StreamRequest, TextStop};
#[cfg(test)]
use stream::{
    stream_terminal_json, stream_terminal_payloads, usage_placeholder, StopTracker,
    StreamLifecycleGuard,
};

/// Sampling-parameter overrides extracted from a request.
///
/// Overlaid onto whatever the acquired engine replica is actually running
/// with ([`crate::engine::InferenceEngine::sampling_params`]) — **never**
/// onto a bare [`SamplingParams::default()`]: the
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

#[tracing::instrument(skip(state, headers, raw), fields(request_id))]
pub(super) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    // `Box<RawValue>` in place of `OpenAiJson<ChatCompletionRequest>`
    // directly — still goes through `OpenAiJson`, so a genuinely malformed
    // body (wrong content-type, unparseable JSON syntax) still hits
    // axum's `JsonRejection` -> `ApiError::from_json_rejection` exactly as
    // before (`RawValue` only requires syntactically valid JSON, not any
    // particular shape). The typed `ChatCompletionRequest` is then parsed
    // from these SAME bytes below, alongside `ChatRequestExtras` — the
    // seam that lets this handler read `tools`' raw text and
    // `chat_template_kwargs` without `ChatCompletionRequest` declaring
    // them (see the module doc).
    OpenAiJson(raw): OpenAiJson<Box<serde_json::value::RawValue>>,
) -> Result<Response, ApiError> {
    let request_id = resolve_request_id(&headers);
    tracing::Span::current().record("request_id", tracing::field::display(&request_id));

    // SV-11: recover what `ChatMessage.content: Option<String>` cannot
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

    // `tool_choice: "none"` is the client explicitly opting out of tool
    // calling: the tools may still be rendered into the prompt, but nothing
    // is extracted from the answer. Tools combine with `stream: true` (each
    // completed call is streamed as one `tool_calls` delta); `n > 1` is
    // refused for every request by `validate_chat_request`.
    let tool_choice_none = matches!(
        body.tool_choice
            .as_ref()
            .and_then(serde_json::Value::as_str),
        Some("none")
    );
    let tools_active =
        body.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false) && !tool_choice_none;
    let want_logprobs = body.logprobs.unwrap_or(false);
    // Per-token logprobs are captured by the non-streaming decode only.
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
    // generation path has acquired its engine lease: seeding from
    // `engine.sampling_params()` needs a lease,
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
    // SV-12 / RT-12: deterministic seed, honoured by both paths, `logprobs`
    // included.
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

    // Build the prompt: the model's own resolved chat template, rendered
    // through the real Jinja engine and encoded in one whole-prompt call —
    // see `chat_render`'s module doc for why one call, and for how TOK-M2
    // (finding `sec-01`'s id-level half) is preserved despite it. Without a
    // tokenizer there is no vocabulary to render into: the configured
    // prompt start token, or `400 tokenizer_required`.
    let (prompt_tokens, rendered) = match &state.tokenizer {
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
            let (rendered, tokens) = chat_render::render_chat_prompt(
                tok,
                &state.special_tokens,
                &render_messages,
                &opts,
                state.sanitize_prompt(),
            )
            .inspect_err(|_| {
                state.metrics.errors_total.inc();
            })?;
            (tokens, Some(rendered))
        }
        None => {
            let tokens = state.tokenizerless_prompt("messages").inspect_err(|_| {
                state.metrics.errors_total.inc();
            })?;
            warn_generating_without_tokenizer("/v1/chat/completions");
            (tokens, None)
        }
    };

    // How this response's tokens split into reasoning, content and tool
    // calls — resolved once from the loaded vocabulary and the actual
    // rendered prompt, never assumed from `enable_thinking` alone (the
    // template's own generation-prompt tail decides whether generation
    // starts inside a think span, and a tail that already closed one stops a
    // model-emitted `<think>` from opening another).
    let shape = ResponseShape::resolve(
        state.tokenizer.as_ref(),
        &prompt_tokens,
        rendered.as_deref(),
        tools_active,
    );

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
    // policy's tier decisions (RT-14), only
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
                min_p: body.min_p,
                stop_sequences,
                seed,
                cancel_slot,
                include_usage: body
                    .stream_options
                    .as_ref()
                    .map(|o| o.include_usage)
                    .unwrap_or(false),
                shape,
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
        // returns — i.e. for the whole non-streaming generation.
        chat_completions_non_stream(
            Arc::clone(&state),
            NonStreamRequest {
                prompt_tokens,
                max_tokens: effective_max_tokens,
                overrides,
                penalties,
                min_p: body.min_p,
                stop_sequences,
                seed,
                cancel_slot,
                want_logprobs,
                top_logprobs,
                shape,
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
/// clippy's argument-count budget.
struct NonStreamRequest {
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    overrides: SamplingOverrides,
    penalties: PenaltyParams,
    /// The request's `min_p` (`None` keeps the replica's baseline).
    min_p: Option<f32>,
    /// Stop sequences (`SV-12`/`RT-04`), applied to the content channel by
    /// the response pipeline.
    stop_sequences: Vec<String>,
    /// Deterministic sampling seed (`SV-12`): the generation — `logprobs`
    /// included — runs on a freshly seeded sampler ([`RequestSampling`]).
    seed: Option<u64>,
    /// SV-09 server wiring: see [`StreamRequest::cancel_slot`].
    cancel_slot: CancelSlot,
    /// Whether to capture and return per-token logprobs.
    want_logprobs: bool,
    /// Number of top alternatives to report per token (`top_logprobs`, 0..=20).
    top_logprobs: usize,
    /// How the response's tokens split into reasoning, content and tool
    /// calls.
    shape: ResponseShape,
    request_id: RequestId,
    /// Unix timestamp for the response's `created` field (`SV-03`).
    created: u64,
    /// The served model's id for the response's `model` field (`SV-03`).
    model_id: String,
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
/// and whole-emoji token) decodes correctly. The `bytes` field is corrected
/// separately from the raw vocabulary bytes ([`correct_logprob_bytes`]).
fn logprob_id_to_token(tokenizer: Option<&TokenizerBridge>, id: u32) -> String {
    match tokenizer {
        Some(tok) => String::from_utf8_lossy(&tok.piece(id)).into_owned(),
        None => format!("<{id}>"),
    }
}

/// TOK-M1: correct **every** `bytes` field --
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
/// `id` lives on both `LogprobsContent` and `TopLogprob`, so this delegates
/// to the shared [`crate::api_types::fix_logprob_bytes`], which fixes every
/// entry — the chosen token's and every alternative's — from its own
/// embedded id and cannot desync the way zipping a separately-tracked id
/// list against the entries could.
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
        min_p,
        stop_sequences,
        seed,
        cancel_slot,
        want_logprobs,
        top_logprobs,
        shape,
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

    // Seed from the engine's actual running
    // configuration, never `SamplingParams::default()`.
    let sampling = RequestSampling {
        params: overrides.apply(lease.sampling_params()),
        penalties: Some(penalties),
        min_p,
        seed,
    };

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
    // The request's sampling configuration is installed for this one
    // generation and the replica's own restored afterwards, whichever
    // generate variant runs (`RT-26`; seeded `logprobs` included).
    let state_for_generation = Arc::clone(&state);
    let generated = run_blocking_generation(lease, move |lease| {
        sampling.run(lease, |engine| {
            if want_logprobs {
                let id_to_token = |id: u32| -> String {
                    logprob_id_to_token(state_for_generation.tokenizer.as_ref(), id)
                };
                engine
                    .generate_with_logprobs(&prompt_tokens, max_tokens, top_logprobs, &id_to_token)
                    .map(|(tokens, lp)| (tokens, Some(lp)))
            } else {
                engine
                    .generate(&prompt_tokens, max_tokens)
                    .map(|tokens| (tokens, None))
            }
        })
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

    // The same pipeline the streaming path runs, over the finished ids:
    // decode → `<think>` split → tool-call extraction → stop sequences
    // (applied to the content channel only — a stop string inside the
    // reasoning or inside a call's arguments stops nothing). Stop sequences
    // are applied after the fact here: the returned text is truncated at
    // the first match exactly as an incremental stop would produce, while
    // `completion_tokens` still counts every token the engine computed.
    let CollectedResponse {
        reasoning_content,
        content,
        tool_calls,
        end,
    } = ResponsePipeline::new(
        &shape,
        state.tokenizer.as_ref(),
        TextStop::new(stop_sequences),
    )
    .collect(state.tokenizer.as_ref(), &output_tokens);

    // A response with tool calls carries `content: null` unless the model
    // also wrote text before its first call.
    let (message_content, tool_calls) = if tool_calls.is_empty() {
        (Some(content), None)
    } else {
        let calls = tool_calls
            .into_iter()
            .map(|call| {
                crate::api_types::ToolCallResult::new_function(
                    call.id,
                    call.function.name,
                    call.function.arguments,
                )
            })
            .collect();
        ((!content.is_empty()).then_some(content), Some(calls))
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
                reasoning_content,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason: end.finish_reason(completion_len, max_tokens).to_string(),
            // TOK-M1: every entry's -- and every alternative's -- `bytes`
            // come from each one's own raw vocabulary id, not from the
            // lossy display string of a byte-fragment token.
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

    let headers = request_id_header_map(request_id);
    Ok((headers, Json(response)).into_response())
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
