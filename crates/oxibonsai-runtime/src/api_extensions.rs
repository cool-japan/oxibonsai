//! Extended `/v1/chat/completions` handler.
//!
//! Adds support for tools (function calling), `n > 1` completions, response
//! format constraints (JSON mode / JSON Schema), stop sequences, and real SSE
//! streaming on top of the base server implementation.
//!
//! # Sampling
//!
//! `temperature` / `top_p` / `repetition_penalty` are resolved against the
//! leased replica's own ambient configuration (`resolve_sampling_params` —
//! never `SamplingParams::default()`), and together with
//! `frequency_penalty` / `presence_penalty` (validated to `[-2.0, 2.0]` and
//! applied over the generated-token history), `min_p` and `seed` they are
//! installed on the replica for each generation and restored afterwards
//! (`crate::server::sampling_scope::RequestSampling`) — on every path,
//! `logprobs` and streaming included. Choice `i` of a request is seeded with
//! `seed + i`; without a `seed`, a plain request still seeds choice `i` with
//! `42 + i` (so `n > 1` choices differ reproducibly), while a `logprobs`
//! request samples from the replica's ambient PRNG.
//!
//! # Response
//!
//! Every choice's tokens go through the shared response pipeline
//! (`crate::server::response_pipeline`): decode → `<think>` split →
//! tool-call extraction (when `tools` are given, keyed off the vocabulary's
//! `<tool_call>` id) → stop sequences (a byte-accurate hold-back window plus
//! an id fast path for a one-token stop sequence, `RT-06`). The
//! non-streaming path collects it, `stream: true` sends each released event
//! as an SSE chunk — so both agree on `reasoning_content`, `content`,
//! `tool_calls` and `finish_reason`. JSON mode (`response_format`
//! `json_object` / `json_schema`) is enforced on a choice's content after
//! extraction, when the choice carries no tool call.
//!
//! `stream: true` supports a single choice: `n > 1` (interleaving several
//! streamed choices) and JSON mode (extraction/wrapping needs the finished
//! text) are refused with `400`. `stream_options.include_usage` adds the
//! usage chunk, as on the base endpoint.
//!
//! # Images
//!
//! `SV-11` — with a vision projector loaded at startup, `image_url` content
//! parts are honoured exactly as on the base endpoint, streamed or not: the
//! template renders each image as its vision placeholder, the image is
//! resolved (a base64 `data:` URI, or a `file://` reference inside the
//! server's media directory), decoded, preprocessed and encoded, and its
//! rows replace the placeholder (`crate::vision_prefill`). Each image counts
//! as the rows it expands to — in `usage.prompt_tokens` and in the context
//! budget, which is checked before any vision-tower work is spent.
//!
//! # Other behaviour
//!
//! - `RT-05` — `usage.completion_tokens` is the real number of tokens the
//!   engine emitted, not an estimate from the (possibly truncated or
//!   JSON-rewritten) final text.
//! - `sec-03` — the non-streaming `n`-loop runs as one unit on the blocking
//!   pool (`crate::server::blocking::run_blocking_generation`).
//! - **Deadline and abandonment** — the request runs under the server's
//!   per-request deadline exactly as on the base endpoint
//!   (`crate::server::deadline`): `504 request_timeout` naming the stage in
//!   `error.phase` (`preparing`, `image_fetch`, `vision_encode`,
//!   `waiting_for_engine`, `prefill`, `decode` with `generated_tokens`), or
//!   the same error in an SSE `error` event followed by `[DONE]` once a
//!   stream is open, and the generation cancelled either way. A non-streamed
//!   request whose handler is dropped (the client went away) cancels its
//!   generation too, and runs none of its remaining choices.
//! - `SV-25` — both routes record the same [`crate::metrics::InferenceMetrics`]
//!   counters/gauges/histogram the base endpoint does.
//! - `SV-32` — the non-streaming path honours an `Idempotency-Key` request
//!   header through [`crate::middleware::IdempotencyCache`]; a replayed
//!   (cache-hit) request observes its own handler-local duration.
//! - `TOK-M2` — the prompt goes through the same template render and
//!   vocabulary-driven special-token guard as the base endpoint
//!   (`chat_render::render_chat_prompt`).
//! - `sec-05` (token half) — a prompt that cannot fit is refused with `400`
//!   naming the real numbers (`context_length_exceeded` /
//!   `max_input_tokens_exceeded`), exactly as the base endpoint refuses it,
//!   instead of failing inside the engine.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::api_types::{
    ChoiceLogprobs, ExtendedChatRequest, ExtendedChatResponse, ExtendedChoice, UsageInfo,
};
use crate::engine_pool::EngineLease;
// Only consumed by `api_extensions_tests.rs`'s `use super::*;`; gating the
// import the same way keeps a non-test build warning-free.
#[cfg(test)]
use crate::metrics::InferenceMetrics;
use crate::middleware::IdempotencyCache;
use crate::pipeline::{StopMatch, StopSequenceMatcher};
use crate::sampling::{PenaltyParams, SamplingParams};
use crate::server::blocking::{run_blocking_generation, CancelOnAbandon};
use crate::server::deadline::{run_with_deadline, CancelSlot};
use crate::server::phase::{self, Phase};
use crate::server::response_pipeline::{
    CollectedResponse, ContentStop, GenerationOutcome, ResponseEnd, ResponsePipeline,
    ResponseShape, StreamChunks, StreamDriver, StreamToolCallDelta,
};
use crate::server::sampling_scope::RequestSampling;
use crate::server::{ActiveRequestGuard, AppState, ChatMessage, MAX_OUTPUT_TOKENS};
use crate::tokenizer_bridge::chat_render::{self, ChatRequestExtras};

/// The SSE-streaming half of this endpoint, in its own file to keep both
/// under the workspace's 2000-line ceiling.
mod stream;

#[cfg(test)]
use stream::StreamDecodeState;
use stream::{extended_chat_completions_stream, stop_stage, ExtendedStream};

// ── Extended handler ──────────────────────────────────────────────────────────

/// Maximum number of independent completions (`n`) the extended endpoint can
/// produce in a single request. Requests asking for more are rejected with
/// `400 Bad Request` rather than silently clamped.
pub const MAX_EXTENDED_N_CHOICES: usize = 4;

/// Build an OpenAI-compatible `400 Bad Request` JSON error response.
///
/// Delegates to the shared [`crate::http_error`] envelope so every route on the
/// server emits the identical `{"error": {message, type, param, code}}` shape.
fn bad_request(message: String, param: &str) -> axum::response::Response {
    crate::http_error::bad_request(message, param)
}

/// Build the per-request [`SamplingParams`] from the engine's own
/// ambient/startup configuration, honoring whichever fields the client
/// request explicitly overrides.
///
/// Every field without an explicit `Some` override comes from
/// `engine_defaults` — never from [`SamplingParams::default`]: a server
/// started with non-default ambient `top_k`/`top_p`/`repetition_penalty`
/// must not have those silently reset to the library values just because a
/// request customized only `temperature`. Shared with `completions.rs`.
pub(crate) fn resolve_sampling_params(
    engine_defaults: &SamplingParams,
    req_temperature: Option<f32>,
    req_top_p: Option<f32>,
    req_repetition_penalty: Option<f32>,
) -> SamplingParams {
    SamplingParams {
        temperature: req_temperature.unwrap_or(engine_defaults.temperature),
        top_p: req_top_p.unwrap_or(engine_defaults.top_p),
        repetition_penalty: req_repetition_penalty.unwrap_or(engine_defaults.repetition_penalty),
        ..engine_defaults.clone()
    }
}

/// One finished choice of the non-streaming `n`-loop, before its tokens go
/// through the response pipeline.
struct RawCompletion {
    /// The generated ids.
    output_tokens: Vec<u32>,
    logprobs: Option<Vec<crate::api_types::LogprobsContent>>,
}

/// The seed choice `index` of a request runs with — see the module doc.
fn choice_seed(seed: Option<u64>, want_logprobs: bool, index: usize) -> Option<u64> {
    match (seed, want_logprobs) {
        (Some(seed), _) => Some(seed.wrapping_add(index as u64)),
        (None, false) => Some(42u64.wrapping_add(index as u64)),
        (None, true) => None,
    }
}

/// Module-local idempotency cache for the extended **non-streaming**
/// endpoint (`SV-32`).
///
/// A client that supplies the same `Idempotency-Key` header twice within the
/// TTL gets back the exact cached response instead of re-running generation.
/// Deliberately **non-streaming only**: an SSE stream cannot be replayed
/// from a byte cache without buffering the whole thing first, at which point
/// it has stopped being a stream, so `extended_chat_completions_stream`
/// never consults this cache and a `stream: true` request ignores the
/// header entirely.
fn idempotency_cache() -> &'static IdempotencyCache {
    static CACHE: std::sync::OnceLock<IdempotencyCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| IdempotencyCache::new(256, std::time::Duration::from_secs(300)))
}

/// Combine a client-supplied `Idempotency-Key` header with a fingerprint of
/// the request body into the actual cache key.
///
/// The header value **alone** is not a safe cache key: two different
/// requests that happen to reuse the same key (a client bug, a colliding
/// value, or simply a different client on an unauthenticated default-open
/// server) would otherwise receive each other's cached completion — the
/// wrong response served with a `200`, not a cache miss. Folding a hash of
/// the semantically-relevant fields into the key (same technique as
/// [`crate::api_types::fingerprint_from_config`]) makes a same-key-
/// different-body request a clean cache **miss** that runs generation
/// normally, rather than serving a stranger's answer.
///
/// Every semantically relevant field is folded in — full tool definitions,
/// `tool_choice`, the `response_format` schema, `logprobs`,
/// `top_logprobs`, `min_p`, `user` — so two requests replayed under the
/// same key that differ in any of them are a cache **miss**. `Tool`,
/// `ToolChoice` and `JsonSchemaFormat` don't implement [`Hash`] (they are
/// wire types built from `serde_json::Value`), so those fields are folded in
/// via their serialized JSON text instead of a structural hash; this is
/// intentionally conservative rather than a canonical fingerprint — two
/// requests with object keys in a different order (but otherwise identical)
/// hash differently and simply produce an extra cache **miss**, never a
/// wrong hit. There is deliberately no `model` field here:
/// `ExtendedChatRequest` has none (this server has exactly one loaded
/// model; an OpenAI-shaped `"model"` key in the request body is ignored).
fn idempotency_cache_key(header_value: &str, req: &ExtendedChatRequest) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for msg in &req.messages {
        msg.role.hash(&mut hasher);
        msg.content.hash(&mut hasher);
    }
    req.max_tokens.hash(&mut hasher);
    req.temperature.map(f32::to_bits).hash(&mut hasher);
    req.top_p.map(f32::to_bits).hash(&mut hasher);
    req.seed.hash(&mut hasher);
    req.n.hash(&mut hasher);
    req.presence_penalty.map(f32::to_bits).hash(&mut hasher);
    req.frequency_penalty.map(f32::to_bits).hash(&mut hasher);
    req.repetition_penalty.map(f32::to_bits).hash(&mut hasher);
    req.min_p.map(f32::to_bits).hash(&mut hasher);
    if let Some(stop) = &req.stop {
        stop.as_slice().hash(&mut hasher);
    }
    req.logprobs.hash(&mut hasher);
    req.top_logprobs.hash(&mut hasher);
    req.user.hash(&mut hasher);
    // Full tool definitions (not just presence) and the tool-choice
    // constraint: different tools/choice must not replay each other's
    // cached completion.
    if let Ok(s) = serde_json::to_string(&req.tools) {
        s.hash(&mut hasher);
    }
    if let Ok(s) = serde_json::to_string(&req.tool_choice) {
        s.hash(&mut hasher);
    }
    if let Some(rf) = &req.response_format {
        rf.format_type.hash(&mut hasher);
        if let Ok(s) = serde_json::to_string(&rf.json_schema) {
            s.hash(&mut hasher);
        }
    }
    format!("{header_value}:{:x}", hasher.finish())
}

/// [`idempotency_cache_key`] for a request that may carry images (SV-11):
/// the typed request only holds each message's flattened text, so the
/// image references are folded in too — two requests that differ only in
/// their pictures must never replay each other's completion. A request
/// without images keeps exactly the text-only key.
fn idempotency_cache_key_with_images(
    header_value: &str,
    req: &ExtendedChatRequest,
    image_references: &[String],
) -> String {
    use std::hash::{Hash, Hasher};
    let key = idempotency_cache_key(header_value, req);
    if image_references.is_empty() {
        return key;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    image_references.hash(&mut hasher);
    format!("{key}:img{:x}", hasher.finish())
}

/// Handler for `POST /v1/chat/completions/extended`.
///
/// Supports all standard fields plus `tools`, `tool_choice`, `logprobs`,
/// `top_logprobs`, `response_format`, `n`, `stop`, `min_p` and
/// `stream_options` (see the module docs); `n` is capped at
/// [`MAX_EXTENDED_N_CHOICES`] and any larger value is rejected with `400`
/// rather than silently clamped.
///
/// The request runs under the server's per-request deadline, exactly as on
/// the base endpoint (`crate::server::deadline`): an expired deadline
/// cancels the request's generation and answers `504 request_timeout` naming
/// the stage (`error.phase`) — or, once a stream is open, ends it with that
/// error in an SSE `error` event and `[DONE]`.
pub async fn extended_chat_completions(
    State(state): State<Arc<AppState>>,
    // SV-11: the vision projector loaded at startup (`--mmproj`), attached
    // to the router as an extension; `None` for a text-only server.
    vision: Option<axum::Extension<Arc<crate::vision_prefill::VisionService>>>,
    headers: HeaderMap,
    // Raw JSON in place of `Json<ExtendedChatRequest>` directly — the typed
    // request is still built from these same bytes immediately below
    // (preserving this route's malformed-JSON behavior exactly), but
    // `tools`' raw text is ALSO captured before it goes through
    // `ExtendedChatRequest.tools: Option<Vec<Tool>>` ->
    // `Tool::function::parameters: serde_json::Value`, whose `Value::Object`
    // is a `BTreeMap` (this workspace's `serde_json` has no
    // `preserve_order`): the schema's key order is destroyed at THAT
    // deserialization step, so nothing downstream of the typed field could
    // render it in the client's order (G7 case 5 diverges at byte 393 when
    // built from the typed path).
    Json(raw): Json<Box<serde_json::value::RawValue>>,
) -> impl IntoResponse {
    let vision = vision.map(|axum::Extension(service)| service);
    // The request's handle on its generation and its stage record: the
    // deadline cancels whatever generation the handler starts and names the
    // stage it caught the request in.
    let slot = CancelSlot::default();
    let handler =
        extended_chat_completions_inner(Arc::clone(&state), vision, headers, raw, slot.clone());
    match run_with_deadline(&state, &slot, async { Ok(handler.await) }).await {
        Ok(response) => response,
        Err(timeout) => timeout.into_response(),
    }
}

/// The `/extended` handler proper, split out so
/// [`extended_chat_completions`] can run it under the per-request deadline.
async fn extended_chat_completions_inner(
    state: Arc<AppState>,
    vision: Option<Arc<crate::vision_prefill::VisionService>>,
    headers: HeaderMap,
    raw: Box<serde_json::value::RawValue>,
    slot: CancelSlot,
) -> axum::response::Response {
    let phase = slot.phase();
    // SV-11: recover what `ChatMessage.content: Option<String>` cannot
    // represent (a content-parts array — its text flattened into the typed
    // field, its image parts kept for the renderer, never silently
    // schema-error'd or dropped) and what it has no field for at all
    // (`reasoning_content` on a replayed assistant turn) from the raw body,
    // BEFORE the typed parse below — mirrors
    // `server/chat.rs::chat_completions`'s identical wiring. `extras` still
    // parses from the ORIGINAL `raw.get()` text (not the rewrite) — see
    // `preprocess_message_content_and_reasoning`'s own doc for why that
    // matters for `tools`' key order.
    let preprocessed = match chat_render::preprocess_message_content_and_reasoning(raw.get()) {
        Ok(pre) => pre,
        Err(e) => {
            state.metrics().errors_total.inc();
            return crate::http_error::error_response(e.status(), e.message(), None);
        }
    };
    let image_references = preprocessed.image_references();
    let req: ExtendedChatRequest = match serde_json::from_str(&preprocessed.rewritten) {
        Ok(r) => r,
        Err(e) => {
            state.metrics().errors_total.inc();
            return crate::http_error::error_response(
                StatusCode::BAD_REQUEST,
                format!("invalid request body: {e}"),
                None,
            );
        }
    };
    let extras: ChatRequestExtras = serde_json::from_str(raw.get()).unwrap_or_default();

    // Separate from `request_start` below (which starts only once the engine
    // has been acquired, so it measures the generation-inclusive tail the
    // way the sibling endpoints do): this one covers the whole handler,
    // purely so the idempotency-cache-hit early return a little further down
    // has *something* to observe into `request_duration_seconds` without
    // redefining what that histogram means for every
    // other request on this route.
    let handler_start = Instant::now();
    // `SV-25`: this route previously recorded no metrics at all. Incremented
    // unconditionally, once, regardless of how the request is ultimately
    // resolved (mirrors the base endpoint's `requests_total` semantics
    // without needing a `requests_total.inc()` call at every one of the
    // early-return validation branches below).
    state.metrics().requests_total.inc();

    if req.max_tokens < 1 {
        state.metrics().errors_total.inc();
        return bad_request("max_tokens must be at least 1".to_string(), "max_tokens");
    }
    if req.max_tokens > MAX_OUTPUT_TOKENS {
        state.metrics().errors_total.inc();
        return bad_request(
            format!(
                "max_tokens {} exceeds the maximum of {MAX_OUTPUT_TOKENS}",
                req.max_tokens
            ),
            "max_tokens",
        );
    }
    let requested_n = req.n.unwrap_or(1);
    if !(1..=MAX_EXTENDED_N_CHOICES).contains(&requested_n) {
        state.metrics().errors_total.inc();
        return bad_request(
            format!("n must be between 1 and {MAX_EXTENDED_N_CHOICES}, got {requested_n}"),
            "n",
        );
    }
    let n = requested_n;
    // OpenAI frequency/presence penalties are now applied for real over the
    // generated-token history via the sampler's penalty seam (previously they
    // were rejected with `400` because no seam existed). They are validated to
    // the OpenAI `[-2.0, 2.0]` range and forwarded to the engine; an all-zero
    // `PenaltyParams` is a no-op.
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
        state.metrics().errors_total.inc();
        return bad_request(
            "frequency_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
            "frequency_penalty",
        );
    }
    if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
        state.metrics().errors_total.inc();
        return bad_request(
            "presence_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
            "presence_penalty",
        );
    }
    // Validated identically to the base `/v1/chat/completions` endpoint
    // (`>= 1.0`): a value below `1.0` would REWARD repeated tokens.
    if let Some(rp) = req.repetition_penalty {
        if !rp.is_finite() || rp < 1.0 {
            state.metrics().errors_total.inc();
            return bad_request(
                "repetition_penalty must be a finite number >= 1.0".to_string(),
                "repetition_penalty",
            );
        }
    }
    // `min_p`: this request only; omitted = the replica's baseline, `0.0`
    // disables it for the request.
    if let Some(min_p) = req.min_p {
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            state.metrics().errors_total.inc();
            return bad_request(
                "min_p must be a finite number in the range [0.0, 1.0]".to_string(),
                "min_p",
            );
        }
    }
    let penalties = PenaltyParams::new(frequency_penalty, presence_penalty);
    let max_tokens = req.max_tokens;
    let seed = req.seed;
    let min_p = req.min_p;
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs_k = req.top_logprobs.unwrap_or(0).clamp(0, 20);
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let response_format = req.response_format.clone();
    let tools = req.tools.clone();
    let is_json_mode = response_format
        .as_ref()
        .map(|rf| rf.format_type == "json_object" || rf.format_type == "json_schema")
        .unwrap_or(false);

    // `stream: true` streams a single choice without JSON mode (see the
    // module docs): interleaving several streamed choices and JSON-mode
    // extraction/wrapping (which needs the finished text) are refused with
    // `400` rather than silently ignored. `tools` stream.
    let stream = req.stream.unwrap_or(false);
    if stream {
        if n > 1 {
            state.metrics().errors_total.inc();
            return bad_request(
                format!(
                    "stream: true only supports n = 1: interleaving {n} streamed choices \
                     is not implemented; omit n (or set it to 1) or set stream to false"
                ),
                "n",
            );
        }
        if is_json_mode {
            state.metrics().errors_total.inc();
            return bad_request(
                "stream: true is not supported together with a json_object/json_schema \
                 response_format: JSON-mode extraction/wrapping needs the complete \
                 generated text, which isn't available until streaming finishes; omit \
                 response_format or set stream to false"
                    .to_string(),
                "response_format",
            );
        }
    }

    // `SV-32`: only the non-streaming path is idempotency-cached — see
    // `idempotency_cache`'s docs for why streaming is excluded. The cache
    // key folds in a fingerprint of the request body
    // (`idempotency_cache_key`, plus the image references of a multimodal
    // request) so a repeated header value with a different body cannot
    // return a different client's cached completion.
    let idempotency_key = if stream {
        None
    } else {
        headers
            .get("idempotency-key")
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
            .map(|header_value| {
                idempotency_cache_key_with_images(header_value, &req, &image_references)
            })
    };
    if let Some(key) = idempotency_key.as_deref() {
        if let Some((status, body)) = idempotency_cache().get(key) {
            // A replayed request is counted in `requests_total` (above)
            // and must not be invisible in
            // `request_duration_seconds` — the histogram's `_count` and the
            // counter could silently disagree. A cache hit does no
            // tokenization/generation, so `prompt_tokens_total` is
            // deliberately left alone (the cached body's own `usage` field
            // still reports the real prompt-token count to the client;
            // re-tokenizing here solely to re-observe a metric would defeat
            // the point of caching).
            state
                .metrics()
                .request_duration_seconds
                .observe(handler_start.elapsed().as_secs_f64());
            return (
                StatusCode::from_u16(status).unwrap_or(StatusCode::OK),
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response();
        }
    }

    // Stop sequences (empty strings never match).
    let stop_sequences: Vec<String> = req
        .stop
        .as_ref()
        .map(|seqs| seqs.as_slice().to_vec())
        .unwrap_or_default()
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();

    // The real loaded-model descriptor, resolved once: an image request is
    // refused here when the engine cannot prefill image rows (a typed `400`,
    // streamed or not — ahead of the template render, the image decode and
    // the vision tower), its context window bounds the request below, and its
    // id is the response `model` field. MUST run before the engine is
    // acquired further down: on an uncached first call,
    // `ServedModelInfo::descriptor` acquires its own (briefly held) lease from
    // this same pool, so calling it while `lease` below is already held would
    // self-deadlock a single-replica pool waiting on a permit only this
    // request holds.
    let descriptor = state.model_info().descriptor().await;
    if !image_references.is_empty() {
        if let Some(refusal) = descriptor.image_support.refusal() {
            state.metrics().errors_total.inc();
            return refusal.into_response();
        }
        // The engine could serve the image; a server started without a
        // vision projector cannot encode it. Checked after the engine's own
        // refusal, which a projector would not lift.
        if vision.is_none() {
            state.metrics().errors_total.inc();
            return chat_render::api_error_from_multimodal(
                &crate::vision_prefill::MultimodalError::VisionUnavailable,
            )
            .into_response();
        }
    }

    // Build the prompt: the model's own resolved chat template, rendered
    // through the real Jinja engine and encoded in one whole-prompt call —
    // see `chat_render`'s module doc for the rendering contract and for how
    // TOK-M2 (`<think>`/`<tool_call>`/`<tool_response>` are real added
    // tokens) is preserved despite the single combined encode.
    let (prompt_tokens, rendered) = match state.tokenizer() {
        Some(tok) => {
            // A message carrying images renders from its content parts,
            // each image as the template's placeholder (SV-11).
            let render_messages = chat_render::to_render_messages_with_parts(
                &req.messages,
                &preprocessed.reasoning_contents,
                &preprocessed.content_parts,
            );
            let opts = oxibonsai_tokenizer::chat_templates::RenderOptions {
                add_generation_prompt: true,
                enable_thinking: extras.effective_enable_thinking(),
                reasoning_effort: extras.effective_reasoning_effort(),
                preserve_thinking: extras.effective_preserve_thinking(),
                add_vision_id: extras.effective_add_vision_id(),
                tools: extras.tools_raw_json(),
            };
            match chat_render::render_chat_prompt(
                tok,
                state.special_tokens(),
                &render_messages,
                &opts,
                state.sanitize_prompt(),
            ) {
                Ok((rendered, tokens)) => (tokens, Some(rendered)),
                Err(err) => {
                    state.metrics().errors_total.inc();
                    tracing::error!(error = %err, "prompt rendering failed");
                    return crate::http_error::error_response(err.status(), err.message(), None);
                }
            }
        }
        // No vocabulary to render into: the configured prompt start token,
        // or `400 tokenizer_required` — always for image content, whose
        // placeholders only exist in a rendered template.
        None if !image_references.is_empty() => {
            state.metrics().errors_total.inc();
            return crate::server::api_error::ApiError::bad_request(
                "image content needs the model's tokenizer to render its placeholders, and this \
                 server has none",
                "messages",
            )
            .with_code("tokenizer_required")
            .into_response();
        }
        None => match state.tokenizerless_prompt("messages") {
            Ok(tokens) => {
                crate::server::warn_generating_without_tokenizer("/v1/chat/completions/extended");
                (tokens, None)
            }
            Err(err) => {
                state.metrics().errors_total.inc();
                return err.into_response();
            }
        },
    };

    // SV-11: images resolved, decoded, preprocessed and spliced against the
    // rendered ids, then encoded; a text-only request is its token ids. A
    // remote image (when the operator enabled them) is fetched while they
    // are resolved, on the blocking pool: the request's own view of the
    // vision service reports each fetch to the stage record (`image_fetch`),
    // and dropping this handler (its deadline, or a client that went away)
    // stops the fetch in flight and starts no further one. A fetch refused
    // because the fetcher is at capacity is the typed, retryable `503`.
    let image_fetches = crate::server::image_fetch::RequestImageFetches::new(Some(slot.phase()));
    let pending = match chat_render::prepare_chat_prompt(
        prompt_tokens,
        image_references,
        image_fetches.watch(vision),
    )
    .await
    {
        Ok(pending) => pending,
        Err(err) => {
            state.metrics().errors_total.inc();
            return image_fetches.classify_failure(err).into_response();
        }
    };
    drop(image_fetches);

    // How each choice's tokens split into reasoning, content and tool
    // calls, from the loaded vocabulary and the actual rendered prompt.
    let shape = ResponseShape::resolve(
        state.tokenizer(),
        pending.tokens(),
        rendered.as_deref(),
        tools.is_some(),
    );

    // sec-05 (token half), as on the base endpoint: an over-long prompt is a
    // `400` naming the real numbers rather than an engine failure midway
    // through generation, and — an image counting as the rows it expands
    // to — it is refused before any vision-tower work is spent on it. Each
    // of the `n` choices generates from the same prompt independently, so
    // the per-choice budget is the whole budget. The bound is the engine's KV
    // window (`min(declared context, window)`), as on the base endpoint.
    if let Err(err) = crate::server::validate_request_budget_in_window(
        pending.len(),
        max_tokens,
        descriptor.max_context_length,
        descriptor.declared_context_length,
        state.limits().max_input_tokens,
    ) {
        state.metrics().errors_total.inc();
        return err.into_response();
    }

    // An image counts as the rows it expands to. The stage record learns the
    // prompt's size (image rows included) and, for an image request, that
    // the deadline may now catch the vision encode.
    let prompt_len = pending.len();
    let image_count = match &pending {
        chat_render::PendingChatPrompt::Multimodal { prepared, .. } => prepared.len(),
        chat_render::PendingChatPrompt::Text(_) => 0,
    };
    phase.set_workload(prompt_len, image_count);
    if image_count > 0 {
        phase.enter(Phase::VisionEncode);
    }
    let prompt = match pending.encode().await {
        Ok(prompt) => prompt,
        Err(err) => {
            state.metrics().errors_total.inc();
            return err.into_response();
        }
    };
    state
        .metrics()
        .prompt_tokens_total
        .inc_by(prompt_len as u64);

    // The response `model` field and the fingerprint input.
    let model_id = descriptor.served_id;

    // Acquire the engine once, both to serve the request and to seed the
    // per-request `SamplingParams` from the engine's own ambient/startup
    // configuration (see `resolve_sampling_params`). Queued until a replica
    // is free; once one is held the engine is ingesting the prompt until it
    // produces its first token.
    phase.enter(Phase::WaitingForEngine);
    let mut lease = match state.acquire_engine().await {
        Ok(lease) => lease,
        Err(e) => {
            state.metrics().errors_total.inc();
            tracing::error!(error = %e, "engine pool acquire failed");
            return crate::http_error::error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "engine pool unavailable",
                None,
            );
        }
    };
    phase.enter(Phase::Prefill);
    state.metrics().active_requests.inc();
    let active_guard = ActiveRequestGuard(Arc::clone(state.metrics()));
    let request_start = Instant::now();

    let sampling_params = resolve_sampling_params(
        lease.sampling_params(),
        req.temperature,
        req.top_p,
        req.repetition_penalty,
    );

    if stream {
        // A text prompt streams from its token ids, a multimodal one through
        // the engine's multimodal prefill (SV-11).
        return extended_chat_completions_stream(
            Arc::clone(&state),
            ExtendedStream {
                lease,
                prompt,
                max_tokens,
                sampling: RequestSampling {
                    params: sampling_params,
                    penalties: Some(penalties),
                    min_p,
                    seed,
                },
                stop_sequences,
                model_id,
                shape,
                include_usage,
                metrics_guard: active_guard,
                request_start,
                slot,
            },
        )
        .await;
    }

    // Arm cancellation now that generation is about to start: the token is
    // recorded in the slot the deadline cancels (with the prefill chunked, so
    // a long prompt's ingest observes it), and the guard holds another handle
    // across the blocking generation, so a handler future dropped before the
    // answer is in hand — the client went away, or a layer outside the
    // handler gave up on it — cancels the generation instead of leaving the
    // replica decoding every choice to `max_tokens` for nobody.
    let cancel_token = slot.arm_lease(&mut lease);
    let abandon = CancelOnAbandon::new(cancel_token);

    // `sec-03`: the whole `n`-loop — up to `MAX_EXTENDED_N_CHOICES` full
    // generations of up to `max_tokens` tokens each — runs as ONE unit on
    // `run_blocking_generation`'s blocking pool: the lease is moved into the
    // closure and reset once up front (`RT-03`), and each of the `n`
    // independent runs additionally resets *inside* the loop so run `i > 0`
    // never inherits run `i - 1`'s generated tokens. Each run installs the
    // request's sampling configuration (choice `i` seeded per
    // [`choice_seed`]) and restores the replica's own afterwards. The stage
    // record counts the tokens of every choice: the first one moves the
    // request from prefill to decode.
    let state_for_generation = Arc::clone(&state);
    let generation_phase = phase.clone();
    let generation = run_blocking_generation(lease, move |lease| {
        let mut results: Vec<RawCompletion> = Vec::with_capacity(n);
        for i in 0..n {
            // An abandoned request (its client gone, its deadline expired)
            // runs no further choice: the token is cancelled, and nobody
            // will read the answer.
            if lease.is_cancelled() {
                tracing::debug!(
                    completed = i,
                    requested = n,
                    "extended completion abandoned; the remaining choices are not run"
                );
                break;
            }
            lease.reset();
            let sampling = RequestSampling {
                params: sampling_params.clone(),
                penalties: Some(penalties),
                min_p,
                seed: choice_seed(seed, want_logprobs, i),
            };
            let outcome = sampling.run(lease, |engine| {
                if want_logprobs {
                    // The callback runs for the chosen token of every step
                    // (and for its alternatives): the first call is the
                    // first token.
                    let id_to_token = |id: u32| -> String {
                        generation_phase.decode_started();
                        match state_for_generation.tokenizer() {
                            Some(tok) => tok.decode(&[id]).unwrap_or_else(|_| format!("<{id}>")),
                            None => format!("<{id}>"),
                        }
                    };
                    prompt
                        .generate_with_logprobs(engine, max_tokens, top_logprobs_k, &id_to_token)
                        .map(|(toks, lp)| (toks, Some(lp)))
                } else {
                    phase::generate_observed(&prompt, engine, max_tokens, &generation_phase)
                        .map(|toks| (toks, None))
                }
            });
            match outcome {
                Ok((output_tokens, logprobs)) => {
                    // TOK-M1: every `bytes` field — the chosen token's and
                    // every `top_logprobs` alternative's — from the token's
                    // own raw vocabulary bytes, not the lossy display
                    // string of a byte-fragment token.
                    let logprobs = logprobs.map(|mut lp| {
                        if let Some(tok) = state_for_generation.tokenizer() {
                            crate::api_types::fix_logprob_bytes(&mut lp, &|id| tok.piece(id));
                        }
                        lp
                    });
                    results.push(RawCompletion {
                        output_tokens,
                        logprobs,
                    });
                }
                Err(e) => {
                    tracing::error!(error = %e, "generation failed for extended completion {i}");
                    return Err(e);
                }
            }
        }
        Ok(results)
    })
    .await;
    // Every choice (or the task's failure) is in hand: nothing is abandoned.
    abandon.disarm();

    let raw_completions: Vec<RawCompletion> = match generation {
        Ok(Ok(results)) => results,
        Ok(Err(e)) => {
            state.metrics().errors_total.inc();
            state
                .metrics()
                .request_duration_seconds
                .observe(request_start.elapsed().as_secs_f64());
            tracing::error!(error = %e, "generation failed");
            // Names the failure like the base chat and legacy completions
            // endpoints do: an engine refusal carries its stable `[CODE]` in
            // `e`'s display.
            return crate::http_error::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation failed: {e}"),
                None,
            );
        }
        Err(api_err) => {
            state.metrics().errors_total.inc();
            state
                .metrics()
                .request_duration_seconds
                .observe(request_start.elapsed().as_secs_f64());
            return api_err.into_response();
        }
    };

    let json_enforcer = JsonModeEnforcer::new();

    let mut total_completion_tokens = 0usize;
    let choices: Vec<ExtendedChoice> = raw_completions
        .into_iter()
        .enumerate()
        .map(
            |(
                idx,
                RawCompletion {
                    output_tokens,
                    logprobs: run_logprobs,
                },
            )| {
                let output_len = output_tokens.len();
                // The same pipeline the streaming path runs, over the
                // finished ids: decode → `<think>` split → tool-call
                // extraction → stop sequences on the content channel.
                let CollectedResponse {
                    reasoning_content,
                    content,
                    tool_calls,
                    end,
                } = ResponsePipeline::new(
                    &shape,
                    state.tokenizer(),
                    stop_stage(&state, &stop_sequences),
                )
                .collect(state.tokenizer(), &output_tokens);

                // A choice with tool calls carries `content: null` unless
                // the model also wrote text before its first call; JSON mode
                // shapes the content of a choice without calls.
                let (content_text, tool_calls) = if tool_calls.is_empty() {
                    let content = if is_json_mode {
                        json_enforcer.enforce(&content)
                    } else {
                        content
                    };
                    (Some(content), None)
                } else {
                    ((!content.is_empty()).then_some(content), Some(tool_calls))
                };

                let finish_reason = determine_extended_finish_reason(
                    end.calls > 0,
                    end.stopped,
                    output_len,
                    max_tokens,
                );

                // Real per-token logprobs, captured during generation by the
                // engine's logits-capturing variant when the client requested
                // them: one entry per generated token.
                let logprobs: Option<ChoiceLogprobs> = run_logprobs.map(|content| ChoiceLogprobs {
                    content: Some(content),
                });

                // `RT-05`: the real number of tokens the engine emitted for
                // this completion (before stop-sequence truncation /
                // JSON-mode rewriting, which is what OpenAI's own
                // `completion_tokens` counts).
                total_completion_tokens += output_len;

                ExtendedChoice {
                    index: idx,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: content_text,
                        reasoning_content,
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason,
                    logprobs,
                    tool_calls,
                }
            },
        )
        .collect();

    state
        .metrics()
        .tokens_generated_total
        .inc_by(total_completion_tokens as u64);

    // Build system fingerprint from the real loaded-model id (resolved above).
    let system_fingerprint = Some(crate::api_types::fingerprint_from_config(&model_id));

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = ExtendedChatResponse {
        id: format!("chatcmpl-ext-{}", rand_ext_id()),
        object: "chat.completion".to_string(),
        created,
        model: model_id,
        choices,
        usage: UsageInfo {
            prompt_tokens: prompt_len,
            completion_tokens: total_completion_tokens,
            total_tokens: prompt_len + total_completion_tokens,
        },
        system_fingerprint,
    };

    // `reasoning_content` is a native `ChatMessage` field, so the
    // idempotency cache (below) stores it with the rest of the response and
    // a cache-hit replay carries it too.
    if let Some(key) = idempotency_key.as_deref() {
        if let Ok(body_bytes) = serde_json::to_vec(&response) {
            idempotency_cache().insert(key, 200, body_bytes);
        }
    }

    drop(active_guard);
    state
        .metrics()
        .request_duration_seconds
        .observe(request_start.elapsed().as_secs_f64());
    Json(response).into_response()
}

fn rand_ext_id() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{ts:x}")
}

// ── Finish reason ─────────────────────────────────────────────────────────────

/// Determine the OpenAI-compatible `finish_reason` for one extended-chat
/// choice — [`ResponseEnd::finish_reason`], the one rule every chat path
/// shares.
///
/// Priority order (matches OpenAI semantics):
/// 1. `"tool_calls"` — the model's output carried at least one tool call.
/// 2. `"stop"` — an explicit client-supplied stop sequence was matched.
/// 3. `"length"` — generation exhausted `max_tokens` without hitting EOS or a
///    stop sequence (the decode loop only returns exactly `max_tokens`
///    tokens when it never broke early on EOS).
/// 4. `"stop"` — otherwise the run ended naturally on EOS.
fn determine_extended_finish_reason(
    has_tool_calls: bool,
    hit_stop: bool,
    output_len: usize,
    max_tokens: usize,
) -> String {
    ResponseEnd {
        calls: usize::from(has_tool_calls),
        stopped: hit_stop,
        unclosed: false,
    }
    .finish_reason(output_len, max_tokens)
    .to_string()
}

// ── JSON mode enforcer ────────────────────────────────────────────────────────

/// Wraps generation to produce valid JSON output.
///
/// Strategy (applied in order):
/// 1. If the text already parses as JSON — return it as-is.
/// 2. Try to extract the first `{…}` or `[…]` substring and parse that.
/// 3. If still not valid JSON — wrap the text in `{"response": "<text>"}`.
pub struct JsonModeEnforcer {
    /// Maximum extraction/wrap attempts (unused here; reserved for future streaming use).
    pub max_retries: usize,
}

impl JsonModeEnforcer {
    /// Create a new enforcer with default settings.
    pub fn new() -> Self {
        Self { max_retries: 3 }
    }

    /// Return a string guaranteed to be valid JSON, applying extraction or
    /// wrapping if needed.
    pub fn enforce(&self, text: &str) -> String {
        // Fast path: already valid JSON
        if crate::api_types::is_valid_json(text) {
            return text.to_string();
        }

        // Try to extract a JSON object substring
        if let Some(extracted) = extract_json_substring(text) {
            if crate::api_types::is_valid_json(&extracted) {
                return extracted;
            }
        }

        // Fallback: wrap in a JSON object
        let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
        format!(r#"{{"response": "{escaped}"}}"#)
    }
}

impl Default for JsonModeEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Try to find and return the first valid-looking JSON object or array in `text`.
fn extract_json_substring(text: &str) -> Option<String> {
    // Look for first `{` and last matching `}` (greedy — works for well-nested JSON)
    if let Some(obj) = extract_balanced(text, '{', '}') {
        return Some(obj);
    }
    // Try array
    if let Some(arr) = extract_balanced(text, '[', ']') {
        return Some(arr);
    }
    None
}

/// Extract the outermost balanced delimited substring starting from the first
/// occurrence of `open` in `text`.
fn extract_balanced(text: &str, open: char, close: char) -> Option<String> {
    let start = text.find(open)?;
    let substr = &text[start..];
    let mut depth = 0i32;
    let mut end_idx = None;

    for (i, ch) in substr.char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                end_idx = Some(i + ch.len_utf8());
                break;
            }
        }
    }

    end_idx.map(|e| substr[..e].to_string())
}

// ── Stop sequence checker ─────────────────────────────────────────────────────

/// Detects and truncates text at stop sequences.
pub struct StopChecker {
    sequences: Vec<String>,
}

impl StopChecker {
    /// Create a new checker with the given stop sequences.
    pub fn new(sequences: Vec<String>) -> Self {
        Self { sequences }
    }

    /// The configured stop sequences (already filtered of empty strings by
    /// every constructor's caller).
    ///
    /// `completions::stream` builds a [`crate::pipeline::StopSequenceMatcher`]
    /// from these for its SSE hold-back window — this checker's own
    /// `check`/`truncate_at_stop` only ever run against a complete,
    /// already-finished string, so they have no notion of "safe to flush so
    /// far".
    pub(crate) fn sequences(&self) -> &[String] {
        &self.sequences
    }

    /// Returns `Some(&str)` with the first matched stop sequence, or `None`.
    pub fn check<'a>(&'a self, text: &str) -> Option<&'a str> {
        for seq in &self.sequences {
            if text.contains(seq.as_str()) {
                return Some(seq.as_str());
            }
        }
        None
    }

    /// Return `(truncated_text, hit_stop)`.
    ///
    /// If any stop sequence is found, the text is truncated at that point.
    pub fn truncate_at_stop(&self, text: &str) -> (String, bool) {
        let mut earliest: Option<(usize, &str)> = None;
        for seq in &self.sequences {
            if let Some(pos) = text.find(seq.as_str()) {
                match earliest {
                    None => earliest = Some((pos, seq.as_str())),
                    Some((prev_pos, _)) if pos < prev_pos => {
                        earliest = Some((pos, seq.as_str()));
                    }
                    _ => {}
                }
            }
        }

        match earliest {
            Some((pos, _)) => (text[..pos].to_string(), true),
            None => (text.to_string(), false),
        }
    }

    /// Returns `true` if no stop sequences are configured.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

// ── Frequency / presence penalty ─────────────────────────────────────────────
//
// `apply_frequency_penalty` below is a standalone, raw-logit-space utility
// kept for API stability (an external integration test,
// `tests/api_extensions_tests.rs`, imports and exercises it directly) rather
// than because it sits on the live decode path: real frequency/presence
// penalty application goes through `crate::sampling::PenaltyParams` /
// `crate::sampling::Sampler::sample_with_history` (see the module docs),
// which both handlers in this file use. `generate_n_completions`, the other
// half of finding `SV-32`'s "two dead primitives" that lived in *this* file
// (a whitespace-tokenizing, debug-formatting duplicate of the real `n`-loop
// now implemented in `extended_chat_completions`), had no such caller and
// has been removed rather than wired in — wiring it would have meant
// routing production traffic through fake tokenization.

/// Apply frequency and presence penalties in-place to a logit vector.
///
/// For each token that has been seen:
/// - **frequency penalty** reduces the logit proportionally to its count.
/// - **presence penalty** reduces the logit by a fixed amount for any seen token.
pub fn apply_frequency_penalty(
    logits: &mut [f32],
    token_counts: &HashMap<u32, usize>,
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    for (&token_id, &count) in token_counts {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            *logit -= frequency_penalty * count as f32;
            *logit -= presence_penalty;
        }
    }
}

#[cfg(test)]
#[path = "api_extensions_tests.rs"]
mod tests;
