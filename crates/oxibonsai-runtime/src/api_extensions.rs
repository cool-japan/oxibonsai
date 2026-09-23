//! Extended `/v1/chat/completions` handler.
//!
//! Adds support for tools (function calling), `n > 1` completions, response
//! format constraints (JSON mode / JSON Schema), stop sequences, and real SSE
//! streaming on top of the base server implementation.
//!
//! OpenAI `frequency_penalty` / `presence_penalty` and `logprobs` are now
//! honored for real (they were previously rejected with `400` because no
//! sampler/logits seam existed):
//!
//! - `frequency_penalty` / `presence_penalty` — validated to `[-2.0, 2.0]` and
//!   applied over the generated-token history via
//!   [`crate::sampling::PenaltyParams`] /
//!   [`crate::sampling::Sampler::sample_with_history`]. An all-zero pair is a
//!   no-op.
//! - `logprobs` — when requested, generation runs through
//!   [`crate::engine::InferenceEngine::generate_with_logprobs`], which captures the per-step
//!   logits and returns real per-token log probabilities plus `top_logprobs`
//!   alternatives. Because that variant has no per-call seed or params seam, a
//!   `logprobs` request honors the frequency/presence penalties but samples
//!   with the engine's ambient `temperature` / `top_p` and is not
//!   seed-reproducible (the seeded, params-honoring, non-logprobs path via
//!   [`crate::engine::InferenceEngine::generate_with_seed`] still is).
//!
//! `stream: true` is implemented as real token-by-token SSE (see
//! `extended_chat_completions_stream`), reusing the same
//! `generate_streaming_with_params` machinery the base `/v1/chat/completions`
//! endpoint uses in `server.rs`. It is only supported for the plain-text,
//! single-choice case: combining it with `tools`, `n > 1`, or a JSON-mode
//! `response_format` is rejected with `400 Bad Request` rather than silently
//! ignored, because each of those features needs the complete generated text
//! before it can be applied (tool-call parsing, multi-choice interleaving,
//! and JSON extraction/wrapping all operate on a finished string, not a
//! partial one).
//!
//! ## Wave-3 fixes (RT-API-EXT)
//!
//! - `RT-05` — `usage.completion_tokens` reported a whitespace-split estimate
//!   of the (possibly stop-truncated / JSON-rewritten) final text instead of
//!   the real emitted token count. Both handlers now report the real
//!   `output_len` the engine returned.
//! - `RT-06` — the streaming path's stop-sequence check only ever suppressed
//!   the *current* SSE chunk's trailing fragment; a stop sequence split
//!   across two chunks leaked its prefix, because every earlier chunk had
//!   already been sent. `extended_chat_completions_stream` now reuses
//!   [`crate::pipeline::StopSequenceMatcher`]'s hold-back window (nothing is
//!   emitted until it is provably outside any window that could still grow
//!   into a match) and, for stop sequences that are themselves a single
//!   token (a model's own `<|im_end|>`/`<think>`/`<tool_call>`-style
//!   markers), an id-based fast path that removes the chunk-boundary-leak
//!   class entirely for that case. Post-verifier-review fix: the id fast
//!   path was itself silently dropping any *other* text still sitting in
//!   the hold-back window at the moment it fired (`StreamDecodeState`'s
//!   `finish()` refuses to flush once `hit_stop` is set, which the text-match
//!   path relies on — it already flushed its own safe prefix inline — but
//!   the id path set `hit_stop` without ever flushing anything); the decode
//!   task now calls `StreamDecodeState::flush_before_stop` on that path
//!   before breaking, so real, already-decoded output never goes missing.
//! - `RT-12` — a client-supplied `seed` now reaches the streaming path too
//!   (previously only the non-streaming path honored it); omitting `seed`
//!   leaves the engine's ambient PRNG state untouched, so the default
//!   (unseeded) case stays bit-identical to previous behavior.
//! - `sec-03` — the non-streaming `n`-loop ran directly on the tokio worker
//!   thread (up to [`MAX_EXTENDED_N_CHOICES`] full generations of up to
//!   [`crate::server::MAX_OUTPUT_TOKENS`] tokens each); it now runs as one
//!   unit on [`crate::server::blocking::run_blocking_generation`]'s blocking
//!   pool, exactly like the streaming path already did.
//! - `SV-25` — both routes mounted from this file recorded no metrics at
//!   all; they now increment the same [`crate::metrics::InferenceMetrics`]
//!   counters/gauges/histogram the base endpoint does.
//! - `SV-32` — [`crate::middleware::IdempotencyCache`] was a complete,
//!   tested, but wholly unreferenced primitive; the non-streaming path now
//!   honors an `Idempotency-Key` request header through it.
//! - `SV-11` (prepare only) — see [`crate::api_types::MessageContent`] /
//!   [`crate::api_types::ContentPart`].
//! - gatekeeper `REQUIRED #1` — `SamplingParams` is now seeded from the
//!   engine's own ambient/startup parameters, never `SamplingParams::default()`
//!   (see `resolve_sampling_params`'s doc, `REQUIRED #18`, for why this
//!   still matters now that `SamplingParams::default`'s `repetition_penalty`
//!   is `1.0`, not the `1.1` an earlier revision of this doc named), with a
//!   client-supplied `repetition_penalty` honored as an override.
//!
//! ## Post-wave-3-verifier-review fixes
//!
//! - `TOK-M2` (blocking) — the prompt was assembled as raw text
//!   (`build_extended_prompt`) and handed to `TokenizerBridge::encode` in one
//!   call, so only the raw-text `<|...|>` guard ever ran. `<think>`,
//!   `</think>`, `<tool_call>`, and `</tool_call>` contain no `<|`, so a
//!   client message consisting of exactly one of those real, atomic control
//!   tokens was tokenized here as the model's genuine control-token id, while
//!   the base `/v1/chat/completions` endpoint's vocabulary-driven
//!   [`crate::server::SpecialTokenGuard`] dropped that same id. The tokenize
//!   step now goes through [`crate::server::sanitize::encode_chat_prompt`],
//!   reusing the base endpoint's per-segment encode + id-level carve-out
//!   exactly (`build_extended_prompt` is gone; `encode_chat_prompt` already
//!   covers the `sanitize == false` escape hatch internally).
//! - logprobs / sampling-params conflict — `generate_with_logprobs` has no
//!   `&SamplingParams` seam (see above), so a `logprobs: true` request used
//!   to silently drop a validated `repetition_penalty` / `temperature` /
//!   `top_p` instead of ever honoring it. Now rejected with `400`
//!   (`param: "logprobs"`), mirroring `completions.rs`'s identical guard.
//! - idempotency cache-hit metrics — a replayed (cache-hit) request was
//!   counted in `requests_total` but never observed into
//!   `request_duration_seconds`, leaving the histogram and the counter
//!   silently inconsistent. The cache-hit return now observes its own
//!   (near-zero) handler-local duration.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};

use crate::api_types::{
    ChoiceLogprobs, ExtendedChatRequest, ExtendedChatResponse, ExtendedChoice, UsageInfo,
};
use crate::engine_pool::EngineLease;
// Only consumed by `api_extensions_tests.rs`'s `use super::*;` (that file is
// declared `#[cfg(test)]` below and is not in this package's `owned_files`,
// so it cannot be edited to import this itself); gating the import the same
// way keeps a non-test build warning-free instead of flagging it unused.
#[cfg(test)]
use crate::metrics::InferenceMetrics;
use crate::middleware::IdempotencyCache;
use crate::pipeline::{StopMatch, StopSequenceMatcher};
use crate::sampling::{PenaltyParams, Sampler, SamplingParams};
use crate::server::{ActiveRequestGuard, AppState, ChatMessage, MAX_OUTPUT_TOKENS};

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
/// Gatekeeper `REQUIRED #1`: every field this function does not receive an
/// explicit `Some` override for comes from `engine_defaults` — never from
/// [`SamplingParams::default`]. `SamplingParams::default`'s
/// `repetition_penalty` is `1.0` today (the RT-24/`REQUIRED #1(a)` fix
/// landed in `sampling.rs`, not `1.1` as an earlier revision of this
/// comment said — corrected here, `REQUIRED #18`), so the eligibility
/// break that number used to cause is gone; seeding from `engine_defaults`
/// remains necessary for a *different* reason that number's fix didn't
/// touch: a server started with non-default ambient `top_k`/`top_p`/
/// `repetition_penalty` must not have those silently reset to
/// `SamplingParams::default`'s library values just because a request
/// customized only `temperature`.
///
/// Shared with `completions.rs` (`pub(crate)`, gatekeeper `REQUIRED #3`) —
/// `/v1/completions` had the identical gap and needs the identical fix; one
/// implementation, not two independently-maintained copies.
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

// `ActiveRequestGuard` (`SV-25`: "always decrements, even on an early
// return or a panicking join") now lives once, `pub(crate)`, in
// `server.rs` — wave-3.5 gatekeeper triage item (5). Imported above via
// `crate::server::{ActiveRequestGuard, ...}` rather than carried here as a
// second, independently-maintained copy.

/// Module-local idempotency cache for the extended **non-streaming**
/// endpoint (`SV-32`).
///
/// [`IdempotencyCache`] was a complete, fully unit-tested primitive with no
/// caller anywhere in the crate (`grep -rn IdempotencyCache` outside
/// `middleware.rs` returned nothing but its own tests) — dead in the request
/// path it was built for. Wired in here, scoped to this one endpoint, via a
/// module-private [`std::sync::OnceLock`] rather than a new `AppState`
/// field, since `AppState` and `middleware.rs` both live in `server.rs` /
/// `middleware.rs`, neither of which this package owns.
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
/// Post-verifier-review fix: this used to fold in only `tools.is_some()`
/// (not the tool *definitions*) and `response_format.format_type` (not its
/// `json_schema`), and omitted `logprobs`, `top_logprobs`, `tool_choice`,
/// and `user` entirely — so two requests replayed under the same key with,
/// say, `logprobs: true` added on the second, would silently get back the
/// first's cached (non-logprobs-shaped) `200` instead of a cache miss.
/// `Tool`/`ToolChoice`/`JsonSchemaFormat` don't implement [`Hash`] (they are
/// wire types built from `serde_json::Value`), so those fields are folded
/// in via their serialized JSON text instead of a structural hash; this is
/// intentionally conservative rather than a canonical fingerprint — two
/// requests with e.g. object keys in a different order (but otherwise
/// identical) hash differently and simply produce an extra cache **miss**,
/// never a wrong hit, which is the same direction of error this function is
/// already allowed to make (a fresh key always re-runs generation safely).
/// There is deliberately no `model` field here: `ExtendedChatRequest` has
/// none (this server has exactly one loaded model; an OpenAI-shaped
/// `"model"` key in the request body is simply an unknown field that serde
/// drops), so there is nothing named `model` to fold in.
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

/// Handler for `POST /v1/chat/completions/extended`.
///
/// Supports all standard fields plus `tools`, `tool_choice`, `logprobs`,
/// `top_logprobs`, `response_format`, `n`, and `stop`. `frequency_penalty` /
/// `presence_penalty` are validated but rejected with `400` when non-zero
/// (see module docs); `n` is capped at [`MAX_EXTENDED_N_CHOICES`] and any
/// larger value is rejected with `400` rather than silently clamped.
pub async fn extended_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ExtendedChatRequest>,
) -> impl IntoResponse {
    // Separate from `request_start` below (which starts only once the engine
    // has been acquired, so it measures the generation-inclusive tail the
    // way the sibling endpoints do): this one covers the whole handler,
    // purely so the idempotency-cache-hit early return a little further down
    // has *something* to observe into `request_duration_seconds` (verifier
    // wave-3 review) without redefining what that histogram means for every
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
    // gatekeeper `REQUIRED #1`: validated the same way the two penalties
    // above are (a client error is rejected honestly, not silently coerced),
    // additionally requiring a strictly positive value — `0.0` or negative
    // would zero out or invert every logit's repetition adjustment, which is
    // never a real sampling strategy.
    if let Some(rp) = req.repetition_penalty {
        if !rp.is_finite() || rp <= 0.0 {
            state.metrics().errors_total.inc();
            return bad_request(
                "repetition_penalty must be a finite number greater than 0.0".to_string(),
                "repetition_penalty",
            );
        }
    }
    let penalties = PenaltyParams::new(frequency_penalty, presence_penalty);
    let max_tokens = req.max_tokens;
    let seed = req.seed;
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs_k = req.top_logprobs.unwrap_or(0).clamp(0, 20);
    // Verifier wave-3 review: `generate_with_logprobs` (used below whenever
    // `want_logprobs` is set) takes no `&SamplingParams` at all — it samples
    // with the engine's ambient sampler (see the module docs) — so a
    // `repetition_penalty` / `temperature` / `top_p` override validated just
    // above (or defaulted from the request struct) would otherwise be
    // silently dropped for this request instead of ever reaching the
    // sampler, with no signal to the client beyond a `200`. Reject the
    // combination honestly, mirroring `completions.rs`'s identical guard for
    // `temperature`/`top_p` (RT-32 / SV-22). `frequency_penalty` /
    // `presence_penalty` are deliberately excluded: they ARE honored on the
    // logprobs path (`lease.set_penalties(penalties)` runs before either
    // branch below), so there is no dropped field to guard against there.
    if want_logprobs
        && (req.repetition_penalty.is_some() || req.temperature.is_some() || req.top_p.is_some())
    {
        state.metrics().errors_total.inc();
        return bad_request(
            "logprobs cannot be combined with repetition_penalty, temperature, or top_p \
             in the same /v1/chat/completions/extended request (the logprobs code path \
             samples with the engine's ambient sampler and has no per-call params seam \
             yet); omit logprobs, or omit repetition_penalty/temperature/top_p"
                .to_string(),
            "logprobs",
        );
    }
    let response_format = req.response_format.clone();
    let tools = req.tools.clone();
    let is_json_mode = response_format
        .as_ref()
        .map(|rf| rf.format_type == "json_object" || rf.format_type == "json_schema")
        .unwrap_or(false);

    // `stream: true` is only implemented for the plain-text, single-choice
    // case (see module docs): tool-call parsing, multi-choice interleaving,
    // and JSON-mode extraction/wrapping all need the complete generated text,
    // which isn't available mid-stream. Reject those combinations honestly
    // with `400` rather than silently ignoring `stream` (the previous
    // behavior) or silently ignoring the incompatible field.
    let stream = req.stream.unwrap_or(false);
    if stream {
        if tools.is_some() {
            state.metrics().errors_total.inc();
            return bad_request(
                "stream: true is not supported together with tools: tool-call parsing \
                 looks for a complete <tool_call>...</tool_call> block, which isn't \
                 available until streaming finishes; omit tools or set stream to false"
                    .to_string(),
                "stream",
            );
        }
        if n > 1 {
            state.metrics().errors_total.inc();
            return bad_request(
                format!(
                    "stream: true only supports n = 1: interleaving {n} streamed choices \
                     is not implemented; omit n (or set it to 1) or set stream to false"
                ),
                "stream",
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
    // (`idempotency_cache_key`) so a repeated header value with a different
    // body cannot return a different client's cached completion.
    let idempotency_key = if stream {
        None
    } else {
        headers
            .get("idempotency-key")
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
            .map(|header_value| idempotency_cache_key(header_value, &req))
    };
    if let Some(key) = idempotency_key.as_deref() {
        if let Some((status, body)) = idempotency_cache().get(key) {
            // Verifier wave-3 review: a replayed request was previously
            // counted in `requests_total` (above) but invisible in
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

    // Build stop checker
    let stop_checker = match req.stop {
        Some(ref seqs) => StopChecker::new(seqs.as_slice().to_vec()),
        None => StopChecker::new(vec![]),
    };

    // Tokenize the prompt.
    //
    // `TOK-M2` (blocking, wave-3 verifier re-review): this used to assemble
    // the whole prompt as one string (`build_extended_prompt`, applying only
    // the raw-text `<|...|>` guard, [`crate::server::neutralize_special_markers`])
    // and hand it to `TokenizerBridge::encode` in a single call. That raw-text
    // guard never matches `<think>`, `</think>`, `<tool_call>`, or
    // `</tool_call>` — none of them contain `<|` — even though every one of
    // them is a real, atomic control token in the shipped vocabularies
    // (Bonsai 2's `token_type = 4` added tokens), so a client message whose
    // content was exactly one of those strings was tokenized here as the
    // model's real control-token id, while the base `/v1/chat/completions`
    // endpoint's vocabulary-driven [`crate::server::SpecialTokenGuard`]
    // silently drops that same id — the two mounted endpoints disagreed
    // about prompt-injection safety. Routing through
    // [`crate::server::sanitize::encode_chat_prompt`] closes that gap by
    // reusing the exact same per-segment encode + id-level carve-out the
    // base endpoint uses (mirrors server.rs's `chat_completions`,
    // server.rs:1004-1011, including the no-tokenizer fallback below).
    let prompt_tokens = match state.tokenizer() {
        Some(tok) => match crate::server::sanitize::encode_chat_prompt(
            tok,
            &req.messages,
            state.special_tokens(),
            state.sanitize_prompt(),
        ) {
            Ok(tokens) => tokens,
            Err(e) => {
                state.metrics().errors_total.inc();
                tracing::error!(error = %e, "tokenization failed");
                return crate::http_error::error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "tokenization failed",
                    None,
                );
            }
        },
        None => vec![151644u32],
    };

    let prompt_len = prompt_tokens.len();
    state
        .metrics()
        .prompt_tokens_total
        .inc_by(prompt_len as u64);

    // Resolve the real loaded-model id once, for both the response `model`
    // field and the fingerprint input, instead of the previous hardcoded
    // "bonsai-8b" literal (mirrors the same fix already applied to the base
    // /v1/chat/completions and /v1/models handlers in server.rs). MUST run
    // before the engine is acquired below: on an uncached first call,
    // `ServedModelInfo::descriptor` acquires its own (briefly held) lease
    // from this same pool, so calling it while `lease` below is already held
    // would self-deadlock a single-replica pool waiting on a permit only
    // this request holds.
    let model_id = state.model_info().descriptor().await.id;

    // Acquire the engine once, both to serve the request and — gatekeeper
    // `REQUIRED #1` — to seed the per-request `SamplingParams` from the
    // engine's own *ambient/startup* configuration rather than a hardcoded
    // literal or `SamplingParams::default()` (see `resolve_sampling_params`'s
    // doc, `REQUIRED #18`, for why this still matters now that
    // `SamplingParams::default`'s `repetition_penalty` is `1.0`, not the
    // `1.1` an earlier revision of this comment named). A client-supplied
    // `repetition_penalty` / `temperature` / `top_p` still overrides the
    // engine default.
    let lease = match state.acquire_engine().await {
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
        let stop_sequences = stop_checker.sequences.clone();
        let resp = extended_chat_completions_stream(
            Arc::clone(&state),
            lease,
            prompt_tokens,
            max_tokens,
            sampling_params,
            penalties,
            stop_sequences,
            model_id,
            seed,
            active_guard,
            request_start,
        )
        .await;
        return resp;
    }

    // `sec-03`: the whole `n`-loop — up to `MAX_EXTENDED_N_CHOICES` full
    // generations of up to `max_tokens` tokens each — previously ran
    // directly inside this async handler, pinning a tokio worker (and the
    // engine replica) for the entire batch. It now runs as ONE unit on
    // `run_blocking_generation`'s blocking pool: the lease is moved into the
    // closure and reset once up front (`RT-03`), and each of the `n`
    // independent runs additionally resets *inside* the loop so run `i > 0`
    // never inherits run `i - 1`'s generated tokens.
    type RawCompletion = (
        String,
        usize,
        Option<Vec<crate::api_types::LogprobsContent>>,
    );
    let state_for_generation = Arc::clone(&state);
    let generation = crate::server::blocking::run_blocking_generation(lease, move |lease| {
        let prev_penalties = lease.penalties();
        lease.set_penalties(penalties);

        let mut results: Vec<RawCompletion> = Vec::with_capacity(n);
        let mut failure: Option<crate::error::RuntimeError> = None;
        for i in 0..n {
            lease.reset();

            // Logprobs and seeded generation are mutually exclusive at this
            // layer: the logits-capturing variant honors the engine's penalties
            // (set above) but has no per-call seed seam, so a `logprobs`
            // request is not seed-reproducible (documented on the function).
            // The far more common non-logprobs path stays fully seeded
            // (`seed` defaults to `42` here only when the client omitted it
            // *and* asked for `n > 1`, purely so the `n` completions differ
            // from one another; a single-choice, unseeded request keeps
            // consuming the engine's ambient PRNG exactly as before, via
            // `generate_with_seed(..., seed.unwrap_or(42) + i, ...)`, matching
            // this handler's pre-existing default of `42` for that case).
            let outcome = if want_logprobs {
                let id_to_token = |id: u32| -> String {
                    match state_for_generation.tokenizer() {
                        Some(tok) => tok.decode(&[id]).unwrap_or_else(|_| format!("<{id}>")),
                        None => format!("<{id}>"),
                    }
                };
                lease
                    .generate_with_logprobs(
                        &prompt_tokens,
                        max_tokens,
                        top_logprobs_k,
                        &id_to_token,
                    )
                    .map(|(toks, lp)| (toks, Some(lp)))
            } else {
                let run_seed = seed.unwrap_or(42).wrapping_add(i as u64);
                lease
                    .generate_with_seed(&prompt_tokens, max_tokens, run_seed, &sampling_params)
                    .map(|toks| (toks, None))
            };

            match outcome {
                Ok((output_tokens, logprobs)) => {
                    let output_len = output_tokens.len();
                    let text = match state_for_generation.tokenizer() {
                        Some(tok) => tok
                            .decode(&output_tokens)
                            .unwrap_or_else(|_| format!("{output_tokens:?}")),
                        None => format!("{output_tokens:?}"),
                    };
                    // TOK-M1: this endpoint applied no `bytes` correction at
                    // all -- `id_to_token`'s single-id `decode` above (not
                    // the raw-vocabulary `piece` lookup) makes a
                    // byte-fragment token's `bytes` the display string's
                    // (lossy, `U+FFFD`-derived) encoding, for the chosen
                    // token AND every `top_logprobs` alternative alike.
                    // Mirrors `server/chat.rs::correct_logprob_bytes`, now
                    // possible here too because `id` lives on
                    // `LogprobsContent`/`TopLogprob` directly
                    // (`api_types.rs`, this package's file).
                    let logprobs = logprobs.map(|mut lp| {
                        if let Some(tok) = state_for_generation.tokenizer() {
                            crate::api_types::fix_logprob_bytes(&mut lp, &|id| tok.piece(id));
                        }
                        lp
                    });
                    results.push((text, output_len, logprobs));
                }
                Err(e) => {
                    tracing::error!(error = %e, "generation failed for extended completion {i}");
                    failure = Some(e);
                    break;
                }
            }
        }

        lease.set_penalties(prev_penalties);
        match failure {
            Some(e) => Err(e),
            None => Ok(results),
        }
    })
    .await;

    let raw_completions: Vec<RawCompletion> = match generation {
        Ok(Ok(results)) => results,
        Ok(Err(e)) => {
            state.metrics().errors_total.inc();
            state
                .metrics()
                .request_duration_seconds
                .observe(request_start.elapsed().as_secs_f64());
            tracing::error!(error = %e, "generation failed");
            return crate::http_error::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation failed",
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

    // Apply stop sequences and response format enforcement (`is_json_mode`
    // was already computed above, before the stream/tools/n compatibility
    // checks).
    let json_enforcer = JsonModeEnforcer::new();

    let mut total_completion_tokens = 0usize;
    let choices: Vec<ExtendedChoice> = raw_completions
        .into_iter()
        .enumerate()
        .map(|(idx, (raw_text, output_len, run_logprobs))| {
            let (truncated, hit_stop) = stop_checker.truncate_at_stop(&raw_text);

            // Apply JSON mode enforcement if requested
            let final_text = if is_json_mode {
                json_enforcer.enforce(&truncated)
            } else {
                truncated
            };

            // B2-13/RT-11: was JSON-only (`crate::api_types::parse_tool_call`),
            // so Bonsai 2's `<tool_call><function=NAME>…</function></tool_call>`
            // XML shape (design §5.4) came back as `None` and its raw XML
            // rendered verbatim as `message.content` with `finish_reason:
            // "stop"` — exactly RT-11's complaint. `parse_tool_calls` tries
            // the XML shape first, falling back to the legacy JSON payload,
            // and — when it finds a call — also reports the natural-language
            // text that preceded it, which becomes this choice's `content`
            // instead of the whole raw text (including the tool-call
            // markup). A `Truncated` result (opened but unclosed) is treated
            // as "no tool call" here, matching the base endpoint: the caller
            // still gets the complete raw text as ordinary content rather
            // than an error for a block the model never finished.
            let (content_text, tool_calls) = if tools.is_some() {
                match crate::tool_calling::parse_tool_calls(&final_text) {
                    crate::tool_calling::ToolCallParseOutcome::Found {
                        leading_text,
                        calls,
                    } => {
                        let trimmed = leading_text.trim();
                        let content = if trimmed.is_empty() {
                            None
                        } else {
                            Some(trimmed.to_string())
                        };
                        (content, Some(calls))
                    }
                    crate::tool_calling::ToolCallParseOutcome::None
                    | crate::tool_calling::ToolCallParseOutcome::Truncated => {
                        (Some(final_text), None)
                    }
                }
            } else {
                (Some(final_text), None)
            };

            let finish_reason = determine_extended_finish_reason(
                tool_calls.is_some(),
                hit_stop,
                output_len,
                max_tokens,
            );

            // Real per-token logprobs, captured during generation by the
            // engine's logits-capturing variant when the client requested
            // them (`logprobs: true`). `content: Some([...])` carries one
            // entry per generated token, each with the chosen token's log
            // probability and its `top_logprobs` alternatives.
            let logprobs: Option<ChoiceLogprobs> = run_logprobs.map(|content| ChoiceLogprobs {
                content: Some(content),
            });

            // `RT-05`: report the real number of tokens the engine emitted
            // for this completion (bound before stop-sequence truncation /
            // JSON-mode rewriting, which is what OpenAI's own
            // `completion_tokens` counts), not a whitespace-split estimate
            // of the possibly-truncated, possibly-rewritten final text.
            total_completion_tokens += output_len;

            ExtendedChoice {
                index: idx,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: content_text,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason,
                logprobs,
                tool_calls,
            }
        })
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
}

#[derive(Debug, serde::Serialize)]
struct ExtendedChunkChoice {
    index: usize,
    delta: ExtendedChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct ExtendedChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn extended_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    delta: ExtendedChunkDelta,
    finish_reason: Option<String>,
) -> String {
    let chunk = ExtendedChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ExtendedChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

// ── Stop-sequence-safe streaming decode state (RT-06) ────────────────────────

/// Pure per-token decode + stop-sequence hold-back state machine driving
/// [`extended_chat_completions_stream`]'s decode task.
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
struct StreamDecodeState {
    matcher: StopSequenceMatcher,
    stop_token_ids: HashSet<u32>,
    accumulated: String,
    emitted_len: usize,
    hit_stop: bool,
}

impl StreamDecodeState {
    fn new(stop_sequences: &[String], stop_token_ids: HashSet<u32>) -> Self {
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
    fn is_stopped(&self) -> bool {
        self.hit_stop
    }

    /// Check `token_id` against the id fast path (see the struct docs) and,
    /// if it matches, mark the stream stopped. Must be called *before*
    /// decoding the token's text — the whole point is to never decode (and
    /// thus never risk emitting so much as a byte of) a token that is itself
    /// a configured stop marker.
    fn hit_stop_by_id(&mut self, token_id: u32) -> bool {
        if !self.hit_stop && self.stop_token_ids.contains(&token_id) {
            self.hit_stop = true;
        }
        self.hit_stop
    }

    /// Feed one token's already-decoded text. Returns the text, if any, that
    /// is now provably safe to emit to the client.
    fn feed(&mut self, decoded_text: &str) -> Option<String> {
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
    fn finish(&mut self) -> Option<String> {
        if self.hit_stop || self.accumulated.len() <= self.emitted_len {
            return None;
        }
        let visible = self.accumulated[self.emitted_len..].to_string();
        self.emitted_len = self.accumulated.len();
        Some(visible)
    }

    /// Flush whatever text is currently held back, bypassing the `hit_stop`
    /// guard [`finish`](Self::finish) applies. Post-verifier-review
    /// companion to [`hit_stop_by_id`](Self::hit_stop_by_id) (RT-06
    /// regression fix): that method sets `hit_stop` *before* the caller
    /// `break`s out of the decode loop, and the token that trips it never
    /// contributes any text of its own to `accumulated` (it is checked by
    /// id precisely so its text is never decoded at all) — so anything
    /// still sitting in the hold-back window at that point is real,
    /// already-decoded model output from *earlier* tokens that `finish()`
    /// would otherwise silently discard (its `hit_stop` guard is correct
    /// for the *text*-match path, where [`feed`](Self::feed) already
    /// flushed the safe prefix itself via `StopMatch::Found`, but wrong for
    /// this one). The realistic trigger: `stop: ["</think>", "\nUser:"]`
    /// with a tokenizer attached — the newline that normally precedes
    /// `</think>` is held back as an unfinished prefix of `"\nUser:"`, and
    /// then the model emits `</think>` as its own single token, tripping
    /// the id fast path; that held-back `"\n"` belongs to neither stop
    /// sequence and must reach the client.
    ///
    /// Deliberately unconditional (no attempt to detect whether the
    /// held-back tail happens to *also* be an unfinished prefix of the very
    /// sequence that just tripped by id): that would require tracking which
    /// specific stop text each id in `stop_token_ids` corresponds to, which
    /// `HashSet<u32>` throws away. That narrower case can only arise if the
    /// model spells part of its own single-token marker out via *other*
    /// ordinary tokens and then, as its very next token, also emits the
    /// literal special token for the same marker — a pathological
    /// double-rendering of one marker, not the chunk-boundary leak this fix
    /// targets (see the `..._realistic_held_back_prefix_before_id_stop`
    /// test in `api_extensions_tests.rs` for the realistic, non-overlapping
    /// case this method is for).
    /// Called on the `break` path in `extended_chat_completions_stream`'s
    /// decode task, before `delta_tx` closes, so the flushed text is always
    /// emitted as its own SSE delta chunk strictly before the `finish_reason`
    /// chunk (`full_stream` `.chain()`s `content_stream` ahead of
    /// `finish_stream`, so the latter is never even polled until the former
    /// is fully drained).
    fn flush_before_stop(&mut self) -> Option<String> {
        if self.accumulated.len() <= self.emitted_len {
            return None;
        }
        let visible = self.accumulated[self.emitted_len..].to_string();
        self.emitted_len = self.accumulated.len();
        Some(visible)
    }
}

/// Real SSE streaming for `POST /v1/chat/completions/extended`.
///
/// Only reachable for the plain-text, single-choice, non-JSON-mode case (see
/// [`extended_chat_completions`]'s compatibility checks); `tools`, `n > 1`,
/// and JSON-mode `response_format` are all rejected with `400` before this
/// function is ever called. `lease` and `metrics_guard` are already-acquired
/// resources handed off by the caller (which also seeded `sampling_params`
/// from the engine's own defaults, gatekeeper `REQUIRED #1`), so this
/// function never touches the engine pool or `active_requests` itself.
///
/// Reuses [`crate::engine::InferenceEngine::generate_streaming_with_params`] / the seeded
/// [`crate::engine::InferenceEngine::generate_streaming`] variant below — the same
/// primitive the base `/v1/chat/completions` endpoint uses for streaming in
/// `server.rs` — and applies client-supplied `stop` sequences via
/// [`StopSequenceMatcher`] (`RT-06`): text is only ever handed to the client
/// once it is provably outside any window that could still grow into a
/// configured stop sequence, which is what actually closes the
/// chunk-boundary leak (the previous per-chunk `accumulated.find` scan only
/// ever suppressed the *current* chunk's trailing fragment — every earlier
/// chunk carrying a leaked prefix had already been sent). A stop sequence
/// that is itself exactly one token (a model's own
/// `<|im_end|>`/`<think>`/`<tool_call>`-style markers) is additionally
/// matched by id the instant it arrives, which removes the chunk-boundary-
/// leak class entirely for that case, since there is no partial byte
/// sequence to have split across a chunk boundary in the first place.
///
/// `seed` (`RT-12`): when `Some`, a fresh, request-seeded [`Sampler`] runs
/// the streaming generation instead of the engine's ambient one (mirroring
/// [`crate::engine::InferenceEngine::generate_with_seed`]'s swap/restore pattern via the
/// engine's `pub(crate)` sampler field, since `crate::engine::InferenceEngine` has no
/// built-in seeded-streaming method); two identical requests with the same
/// seed then produce byte-identical streams. When `None` (the common,
/// unseeded case) this is not touched at all — `generate_streaming_with_params`
/// runs exactly as before, so the engine's ambient PRNG state, and therefore
/// the previous default behavior, stays bit-for-bit unchanged.
#[allow(clippy::too_many_arguments)]
async fn extended_chat_completions_stream(
    state: Arc<AppState>,
    mut lease: EngineLease,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling_params: SamplingParams,
    penalties: PenaltyParams,
    stop_sequences: Vec<String>,
    model_id: String,
    seed: Option<u64>,
    metrics_guard: ActiveRequestGuard,
    request_start: Instant,
) -> axum::response::Response {
    let completion_id = format!("chatcmpl-ext-{}", rand_ext_id());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
    let (finish_tx, finish_rx) = tokio::sync::mpsc::unbounded_channel::<usize>();

    // Run generation on a blocking thread, exactly like the base endpoint's
    // `chat_completions_stream` in server.rs. The lease (and thus `token_tx`)
    // drops at the end of the closure, which both returns the engine to the
    // pool and closes the token channel so the decode task below terminates.
    tokio::task::spawn_blocking(move || {
        lease.reset();
        // Apply frequency/presence penalties for this run, restoring the
        // engine's previous penalties before the lease drops so they don't leak
        // to the next request served by this pool replica.
        let prev_penalties = lease.penalties();
        lease.set_penalties(penalties);
        let result = match seed {
            // `RT-12`: `InferenceEngine::sampler` is `pub(crate)`, so this
            // crate (this file included) can swap in a freshly seeded
            // `Sampler` for the duration of the call exactly the way
            // `generate_with_seed` does internally, without engine.rs
            // needing a new seeded-streaming method.
            Some(seed) => {
                let mut fresh = Sampler::new(sampling_params.clone(), seed);
                fresh.set_penalties(*lease.sampler.penalties());
                let old_sampler = std::mem::replace(&mut lease.sampler, fresh);
                let r = lease.generate_streaming(&prompt_tokens, max_tokens, &token_tx);
                lease.sampler = old_sampler;
                r
            }
            None => lease.generate_streaming_with_params(
                &prompt_tokens,
                max_tokens,
                &sampling_params,
                &token_tx,
            ),
        };
        lease.set_penalties(prev_penalties);
        // Send the real generated-token count for finish_reason, even on
        // error (0 generated is an honest "stop" rather than "length").
        let _ = finish_tx.send(result.unwrap_or(0));
    });

    let hit_stop = Arc::new(AtomicBool::new(false));
    let hit_stop_for_finish = Arc::clone(&hit_stop);
    let hit_stop_for_content = Arc::clone(&hit_stop);

    // `RT-06`: a preferential token-id fast path for any configured stop
    // sequence that is exactly one token's own decoded text (see the
    // function docs). Computed once, up front.
    let stop_token_ids: HashSet<u32> = if stop_sequences.iter().all(|s| s.is_empty()) {
        HashSet::new()
    } else {
        stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|seq| {
                state
                    .tokenizer()
                    .and_then(|tok| tok.inner().token_to_id(seq))
            })
            .collect()
    };
    let mut decode_loop = StreamDecodeState::new(&stop_sequences, stop_token_ids);

    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let mut decode_state = state.tokenizer().map(|t| t.new_decode_stream(true));
    let state_for_content = Arc::clone(&state);

    // Owns the decode loop's state (`accumulated`/`emitted_len`, inside
    // `decode_loop`) and runs to completion regardless of how fast the
    // client reads the SSE response, which is what makes the trailing flush
    // below possible: a `Stream` combinator built directly on `token_rx` has
    // no hook that fires once after the source stream ends, but an explicit
    // task does.
    tokio::spawn(async move {
        // Keeps `active_requests`/`request_duration_seconds` (`SV-25`) live
        // for the whole decode loop, not just the instant this function
        // hands back the initial `Sse` response — dropped when this task
        // ends, i.e. once decoding (and any trailing flush) is fully done.
        let _metrics_guard = metrics_guard;

        let mut token_stream = UnboundedReceiverStream::new(token_rx);

        while let Some(token_id) = token_stream.next().await {
            // `SV-25` deliberately does *not* increment `tokens_generated_total`
            // per token here, matching `server.rs`'s base streaming handler
            // (`chat_completions_stream`, which also does not): both rely on
            // `InferenceEngine::generate_streaming`'s own internal
            // `self.metrics.tokens_generated_total.inc_by(..)` when a metrics
            // handle is wired onto the engine (`EnginePool::set_metrics_all`,
            // done by `cmd_serve.rs`'s real CLI path). Incrementing again
            // here would double-count in exactly that configuration; this
            // stays consistent with the base endpoint's existing choice
            // rather than fixing that pre-existing gap unscoped.
            if decode_loop.hit_stop_by_id(token_id) {
                hit_stop_for_content.store(true, Ordering::Relaxed);
                // Post-verifier-review regression fix: the token that just
                // tripped the id fast path never contributed any text of
                // its own to `accumulated` (its text is never decoded at
                // all), so anything still sitting in the hold-back window
                // is real, already-decoded output from *earlier* tokens.
                // `finish()` below is a no-op once `hit_stop` is set, so
                // without this the hold-back window's contents would be
                // silently dropped instead of reaching the client. See
                // `flush_before_stop`'s doc comment for the full rationale.
                if let Some(visible) = decode_loop.flush_before_stop() {
                    let _ = delta_tx.send(visible);
                }
                break;
            }

            let text = match (state_for_content.tokenizer(), decode_state.as_mut()) {
                (Some(tok), Some(dec_state)) => match tok.step_decode(dec_state, token_id) {
                    Ok(Some(txt)) => txt,
                    Ok(None) => continue,
                    Err(_) => format!("[{token_id}]"),
                },
                _ => format!("[{token_id}]"),
            };

            if let Some(visible) = decode_loop.feed(&text) {
                let _ = delta_tx.send(visible);
            }
            if decode_loop.is_stopped() {
                hit_stop_for_content.store(true, Ordering::Relaxed);
                break;
            }
        }

        // Generation ended without ever matching a stop sequence (EOS or
        // max_tokens): flush whatever text was still held back as a
        // possible stop-sequence prefix — it never grew into one, so it is
        // real, final output that must not be silently dropped.
        if let Some(visible) = decode_loop.finish() {
            let _ = delta_tx.send(visible);
        }

        state_for_content
            .metrics()
            .request_duration_seconds
            .observe(request_start.elapsed().as_secs_f64());
        // `delta_tx` (closing `content_stream` below) and `_metrics_guard`
        // (decrementing `active_requests`) both drop here.
    });

    let id_for_content = completion_id.clone();
    let model_for_content = model_id.clone();
    let content_stream = UnboundedReceiverStream::new(delta_rx).map(move |visible_text| {
        extended_chunk_json(
            &id_for_content,
            created,
            &model_for_content,
            ExtendedChunkDelta {
                role: None,
                content: Some(visible_text),
            },
            None,
        )
    });

    let id_for_finish = completion_id.clone();
    let model_for_finish = model_id.clone();
    let finish_stream = UnboundedReceiverStream::new(finish_rx).map(move |generated| {
        // Mirrors `determine_extended_finish_reason`'s precedence for the
        // (tool-call-free) streaming case: an explicit stop-sequence match
        // wins over a length-based truncation.
        let finish_reason = if hit_stop_for_finish.load(Ordering::Relaxed) || generated < max_tokens
        {
            "stop"
        } else {
            "length"
        };
        extended_chunk_json(
            &id_for_finish,
            created,
            &model_for_finish,
            ExtendedChunkDelta {
                role: None,
                content: None,
            },
            Some(finish_reason.to_string()),
        )
    });

    let role_event = extended_chunk_json(
        &completion_id,
        created,
        &model_id,
        ExtendedChunkDelta {
            role: Some("assistant".to_string()),
            content: None,
        },
        None,
    );

    let full_stream = tokio_stream::once(role_event)
        .chain(content_stream)
        .chain(finish_stream)
        .map(|json_str| -> Result<Event, Infallible> { Ok(Event::default().data(json_str)) })
        .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));

    Sse::new(full_stream).into_response()
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
/// choice.
///
/// Priority order (matches OpenAI semantics):
/// 1. `"tool_calls"` — the model's output was parsed as a tool call.
/// 2. `"stop"` — an explicit client-supplied stop sequence was matched.
/// 3. `"length"` — generation exhausted `max_tokens` without hitting EOS or a
///    stop sequence. This relies on `engine::generate`'s decode loop, which
///    only ever returns exactly `max_tokens` output tokens when it never
///    broke early on `EOS_TOKEN_ID` (see `engine.rs::generate`), so
///    `output_len >= max_tokens` reliably signals truncation. Mirrors
///    `completions.rs::determine_finish_reason`.
/// 4. `"stop"` — otherwise the run ended naturally on EOS.
fn determine_extended_finish_reason(
    has_tool_calls: bool,
    hit_stop: bool,
    output_len: usize,
    max_tokens: usize,
) -> String {
    if has_tool_calls {
        "tool_calls".to_string()
    } else if hit_stop {
        "stop".to_string()
    } else if output_len >= max_tokens {
        "length".to_string()
    } else {
        "stop".to_string()
    }
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
    /// B2-13: `completions::stream` needs these to build a
    /// [`crate::pipeline::StopSequenceMatcher`] for its SSE hold-back
    /// window — this checker's own `check`/`truncate_at_stop` only ever run
    /// against a complete, already-finished string (the non-streaming
    /// path), so they have no notion of "safe to flush so far".
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
