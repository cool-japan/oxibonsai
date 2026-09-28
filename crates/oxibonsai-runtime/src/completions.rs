//! OpenAI v1 completions endpoint (legacy, non-chat).
//!
//! Implements `POST /v1/completions` — the original text completion API that is
//! still widely used by clients that pre-date the chat-completions interface.
//!
//! ## B2-13 (2026-09-23)
//!
//! - **`stream: true` is real SSE**, not a `400`. ORCHESTRATOR RULING D-3
//!   (final): the wave-2 interim that rejected `stream: true` is superseded
//!   — this endpoint now emits `text_completion` chunks sharing the chat
//!   endpoint's SSE machinery ([`crate::server::sse::sse_response`]); see
//!   [`stream`] for the chunk shapes and the (documented, deliberate)
//!   combinations still rejected — a batch of more than one prompt,
//!   `logprobs`, and `seed` — because none of those has a streaming-capable
//!   engine seam.
//! - **`repetition_penalty`** is now a real request field (gatekeeper
//!   `REQUIRED #3`), validated `> 0.0` exactly like
//!   `ExtendedChatRequest::repetition_penalty`.
//! - **Sampling params are seeded from the engine's own ambient/startup
//!   configuration** (`lease.sampling_params()`), never
//!   `SamplingParams::default()` (gatekeeper `REQUIRED #1`/`#3`): the old
//!   `let mut sampling_params = SamplingParams::default();` meant a server
//!   started with non-default `top_k`/`top_p`/`repetition_penalty` had every
//!   field the request did not itself override silently reset to the
//!   library defaults on every request. Shared with `api_extensions.rs` via
//!   [`crate::api_extensions::resolve_sampling_params`] (`pub(crate)`) —
//!   one implementation, not two.
//!
//! # Behaviour
//!
//! - Accepts a single prompt string **or** a batch of prompt strings.
//! - `echo` — when `true`, prepends the original prompt text to each completion.
//! - `stop` — one or more stop sequences, honoured via the same
//!   [`crate::api_extensions::StopChecker`] the extended chat endpoint uses:
//!   each completion is truncated at the first stop sequence found, and
//!   `finish_reason` reports `"stop"` for that choice regardless of how many
//!   tokens were generated (`RT-32`).
//! - `seed` — honoured via [`crate::engine::InferenceEngine::generate_with_seed`]
//!   for deterministic generation. Each prompt in a batch gets a distinct
//!   seed (`seed + its position`), matching the `base_seed + i` convention
//!   `api_extensions.rs`'s `generate_n_completions` already uses, rather than
//!   silently having no effect (`RT-32`).
//! - `logprobs` — when set, per-token log probabilities are captured via
//!   [`crate::engine::InferenceEngine::generate_with_logprobs`] and returned
//!   in the legacy `{tokens, token_logprobs, top_logprobs, text_offset}`
//!   shape (`RT-32`; previously always `null`). **`B9`**: `logprobs` now
//!   combines with `seed`, `temperature`, `top_p` and `repetition_penalty`
//!   — `create_completion`'s logprobs branch swaps the resolved
//!   [`crate::sampling::SamplingParams`] (and, when `seed` is set, a
//!   freshly-seeded whole [`crate::sampling::Sampler`]) onto the engine
//!   lease for the duration of that one call and restores it afterward,
//!   the same seam [`crate::server::chat`]'s non-streaming path and
//!   `api_extensions.rs`'s seeded streaming path already use
//!   (`InferenceEngine::sampler` is `pub(crate)`). An earlier revision of
//!   this endpoint rejected every one of those combinations with `400`,
//!   citing "no public seam" — that claim was stale.
//!   `frequency_penalty`/`presence_penalty` are honoured the same way they
//!   always were (applied via `set_penalties` before the logprobs-capturing
//!   decode loop runs).
//! - `stream: true` streams real SSE (see the `B2-13` section above and
//!   [`stream`]) for a single prompt, `logprobs` INCLUDED (`B8`:
//!   [`stream::stream_completion_with_logprobs`] streams each token's own
//!   logprobs in its `text_completion` chunk). Only a batched prompt
//!   (more than one entry) or `seed` are still rejected with `400 Bad
//!   Request` naming the field, since neither has a streaming-capable
//!   engine seam. A non-empty `suffix` is rejected the same way regardless
//!   of `stream` (`RT-32` / `SV-22`): the engine has no fill-in-the-middle
//!   generation mode. `stream: false` (or the field omitted, the common
//!   case) is unaffected by any of this.
//! - Every prompt in a batch is generated: the engine lease is held for the
//!   whole request and every prompt's generation runs inside a single
//!   [`crate::server::blocking::run_blocking_generation`] call (`sec-03`) —
//!   tokenisation (not CPU-bound in the way generation is) stays on the async
//!   task, mirroring `server.rs`'s own non-streaming path — and the response
//!   has one [`CompletionChoice`] per prompt with `index` matching its
//!   position in the batch. Batches larger than
//!   [`MAX_COMPLETION_BATCH_SIZE`] are rejected with `400 Bad Request`
//!   rather than silently truncated.
//! - `n` — only `n = 1` (the default) is supported; any other value is
//!   rejected with `400 Bad Request` rather than silently generating a single
//!   completion.
//! - `frequency_penalty` / `presence_penalty` — validated to the OpenAI
//!   `[-2.0, 2.0]` range and applied for real over the generated-token history
//!   via [`crate::sampling::PenaltyParams`] /
//!   [`crate::sampling::Sampler::sample_with_history`]. `temperature` / `top_p`
//!   are likewise honored when supplied, **except** together with `logprobs`
//!   (see above: that combination is rejected with `400` instead of silently
//!   dropping `temperature`/`top_p` and generating at the engine's ambient
//!   sampler). A request that customizes none of these takes the vanilla
//!   decode path and is bit-identical to before.
//! - `max_tokens` — bounded by [`crate::server::MAX_OUTPUT_TOKENS`], the same
//!   ceiling the base `/v1/chat/completions` handler enforces; requests above
//!   it are rejected with `400 Bad Request`.
//! - `user` — logged at `debug` level (an opaque end-user tag; still not
//!   otherwise processed, matching the OpenAI contract for this field).
//! - Every failure path — including this fix's new validation rules — renders
//!   the shared [`crate::server::api_error::ApiError`] envelope, so a `500`
//!   or `503` always carries a parseable JSON body instead of an empty one.

use axum::{
    extract::State,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_extensions::{resolve_sampling_params, StopChecker};
use crate::api_types::{LogprobsContent, StopSequences, UsageInfo};
use crate::error::RuntimeResult;
use crate::sampling::PenaltyParams;
// Only used by this file's and `stream.rs`'s `#[cfg(test)]` `test_router()`
// helpers (both reach it via `use super::*;`); a non-test build resolves
// `SamplingParams` entirely through `resolve_sampling_params`'s return type,
// never by name, so gating the import the same way `InferenceMetrics` is
// gated in `api_extensions.rs` keeps a non-test build warning-free.
#[cfg(test)]
use crate::sampling::SamplingParams;
use crate::server::api_error::{ApiError, OpenAiJson};
use crate::server::blocking::run_blocking_generation;
use crate::server::{ActiveRequestGuard, AppState, StreamOptions, MAX_OUTPUT_TOKENS};

mod stream;

/// Maximum number of prompts a single `POST /v1/completions` batch request
/// (`prompt: [...]`) may contain. Requests with a larger batch are rejected
/// with `400 Bad Request` rather than silently generating only a prefix of
/// the batch. Matches [`crate::batch_engine::BatchConfig`]'s default
/// `max_batch_size` for consistency across the two batch-shaped endpoints.
pub const MAX_COMPLETION_BATCH_SIZE: usize = 8;

// ─── Request ─────────────────────────────────────────────────────────────────

/// Input prompt: a single string or a batch.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    /// A single prompt string.
    Single(String),
    /// A batch of prompt strings.
    Batch(Vec<String>),
}

impl PromptInput {
    /// Return all prompt strings as a `Vec<&str>`.
    pub fn as_strings(&self) -> Vec<&str> {
        match self {
            PromptInput::Single(s) => vec![s.as_str()],
            PromptInput::Batch(v) => v.iter().map(String::as_str).collect(),
        }
    }

    /// Return the first prompt string, or an empty string if the batch is empty.
    pub fn first(&self) -> &str {
        match self {
            PromptInput::Single(s) => s.as_str(),
            PromptInput::Batch(v) => v.first().map(String::as_str).unwrap_or(""),
        }
    }
}

/// `POST /v1/completions` request body.
///
/// Follows the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create).
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    /// The model to use (ignored for generation — OxiBonsai always uses the
    /// loaded engine, and the response `model` reports that engine's real id).
    pub model: Option<String>,
    /// The prompt to complete.
    pub prompt: PromptInput,
    /// Maximum number of tokens to generate per completion.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus (top-p) sampling threshold.
    pub top_p: Option<f32>,
    /// Number of completions to generate (only 1 is currently supported).
    pub n: Option<usize>,
    /// Whether to stream the response as SSE (B2-13 / ORCHESTRATOR RULING
    /// D-3). Real SSE for a single prompt, `logprobs` included (`B8`); a
    /// batched prompt or `seed` are rejected with `400` naming the
    /// offending field rather than `stream` itself (see the module docs and
    /// [`stream`]).
    pub stream: Option<bool>,
    /// OpenAI `stream_options`; only meaningful with `stream: true`. Only
    /// `include_usage` is honored, exactly like the base chat endpoint: a
    /// final usage-only chunk is emitted just before `[DONE]` when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Sequences that terminate generation. Honoured via
    /// [`crate::api_extensions::StopChecker`].
    pub stop: Option<StopSequences>,
    /// Penalise tokens that appear at least once in the context.
    pub presence_penalty: Option<f32>,
    /// Penalise tokens proportional to their frequency.
    pub frequency_penalty: Option<f32>,
    /// Repetition penalty (`1.0` = disabled), matching
    /// `ChatCompletionRequest`/`ExtendedChatRequest`'s field of the same
    /// name (gatekeeper `REQUIRED #3`; not a standard OpenAI Completions
    /// field, but accepted the same way vLLM does — and this endpoint
    /// already accepted the analogous non-standard `top_p`/`temperature`
    /// overrides). Validated `>= 1.0` (gatekeeper `REQUIRED #3`'s
    /// `B2-13` wave-4b follow-up — a value below `1.0` would REWARD
    /// repeated tokens instead of suppressing them). When omitted, the
    /// engine's own startup value is used (never `SamplingParams::default`'s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Return the log probabilities for the top-N tokens at each step. A
    /// real, honored combination with both `seed` and `stream: true` (`B9`
    /// / `B8`, `B2-13` wave-4b): `create_completion`'s logprobs branch
    /// swaps a resolved `SamplingParams` — a freshly-seeded whole `Sampler`
    /// when `seed` is also set — onto `lease.sampler` for the call and
    /// restores it unconditionally afterward, and
    /// [`stream::stream_completion_with_logprobs`] builds a dedicated
    /// per-token streaming decode + logit-capture loop for the `stream`
    /// case. `seed` together with `stream: true` remains rejected (no
    /// engine path streams deterministically-seeded generation yet).
    pub logprobs: Option<usize>,
    /// If `true`, the prompt is echoed back at the start of the completion text.
    pub echo: Option<bool>,
    /// Random seed for deterministic generation. A real, honored
    /// combination with `logprobs` (`B9`, `B2-13` wave-4b — see that
    /// field's own doc); still mutually exclusive with `stream: true` (see
    /// the module docs).
    ///
    /// Seeded from the engine's own ambient `SamplingParams`
    /// ([`crate::engine::InferenceEngine::sampling_params`], via
    /// [`resolve_sampling_params`]) before being passed to
    /// [`crate::engine::InferenceEngine::generate_with_seed`] — gatekeeper
    /// `REQUIRED #3` closed the gap an earlier revision of this doc
    /// recorded (`generate_with_seed` building its temporary sampler from
    /// library defaults instead of the server's actual configuration): that
    /// accessor now exists and this handler uses it, so setting only `seed`
    /// no longer silently resets `temperature`/`top_p`/`repetition_penalty`
    /// to their library defaults for that one request.
    pub seed: Option<u64>,
    /// Text to append after the completion. Not supported — the engine has
    /// no fill-in-the-middle generation mode; a non-empty value is rejected
    /// with `400` naming this field (see the module docs). An empty string
    /// (or the field omitted) is accepted as a no-op.
    pub suffix: Option<String>,
    /// Opaque end-user identifier, logged at `debug` level but not otherwise
    /// processed.
    pub user: Option<String>,
}

fn default_max_tokens() -> usize {
    16
}

// ─── Response types ───────────────────────────────────────────────────────────

/// Log-probability information attached to a completion choice.
///
/// The `tokens`, `token_logprobs`, `top_logprobs`, and `text_offset` arrays
/// are parallel and have one entry per generated token.
#[derive(Debug, Serialize)]
pub struct CompletionLogprobs {
    /// The string form of each generated token.
    pub tokens: Vec<String>,
    /// The log probability of each generated token.
    pub token_logprobs: Vec<f32>,
    /// Top-N alternative tokens at each position (as JSON objects mapping
    /// token text to its log probability).
    pub top_logprobs: Vec<serde_json::Value>,
    /// Character offset of each token within the choice's `text` field
    /// (including the echoed prompt's length when `echo` is set).
    pub text_offset: Vec<usize>,
}

/// A single completion choice.
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    /// The generated (and optionally echoed) text.
    pub text: String,
    /// Zero-based index among all returned choices.
    pub index: usize,
    /// Log-probability information (present only when `logprobs` was requested).
    pub logprobs: Option<CompletionLogprobs>,
    /// Why generation stopped (`"stop"` or `"length"`).
    pub finish_reason: String,
}

/// `POST /v1/completions` response body.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    /// Unique completion identifier (prefix `cmpl-`).
    pub id: String,
    /// Object type: always `"text_completion"`.
    pub object: String,
    /// Unix timestamp at which the completion was created.
    pub created: u64,
    /// The model that generated the completion.
    pub model: String,
    /// One or more completion choices.
    pub choices: Vec<CompletionChoice>,
    /// Token usage statistics.
    pub usage: UsageInfo,
}

// ─── Validation ───────────────────────────────────────────────────────────────

/// One prompt's raw generation result, before decoding to text.
struct PromptOutcome {
    tokens: Vec<u32>,
    logprobs: Option<Vec<LogprobsContent>>,
}

/// Parameters derived from a [`CompletionRequest`] after full validation,
/// ready to drive generation. Kept separate from the wire request so
/// validation is a pure function of the request body — testable without an
/// Axum extractor, an engine, or a tokio runtime.
///
/// Carries the raw `temperature`/`top_p`/`repetition_penalty` *overrides*
/// rather than a resolved [`SamplingParams`] (gatekeeper `REQUIRED #3`):
/// resolving them needs the acquired engine lease's own ambient
/// configuration ([`resolve_sampling_params`]), which only exists in
/// [`create_completion`]/[`stream::stream_completion`], not here.
struct ValidatedRequest {
    prompts: Vec<String>,
    max_tokens: usize,
    penalties: PenaltyParams,
    req_temperature: Option<f32>,
    req_top_p: Option<f32>,
    req_repetition_penalty: Option<f32>,
    custom_sampling: bool,
    echo: bool,
    seed: Option<u64>,
    logprobs_top_k: Option<usize>,
    stop_checker: StopChecker,
    stream: bool,
    include_usage: bool,
}

/// Validate a [`CompletionRequest`], or return the first [`ApiError`] naming
/// the offending field.
fn validate_completion_request(req: CompletionRequest) -> Result<ValidatedRequest, ApiError> {
    if req.max_tokens < 1 {
        return Err(ApiError::bad_request(
            "max_tokens must be at least 1",
            "max_tokens",
        ));
    }
    if req.max_tokens > MAX_OUTPUT_TOKENS {
        return Err(ApiError::bad_request(
            format!(
                "max_tokens {} exceeds the maximum of {MAX_OUTPUT_TOKENS}",
                req.max_tokens
            ),
            "max_tokens",
        ));
    }
    if let Some(n) = req.n {
        if n != 1 {
            return Err(ApiError::bad_request(
                "n must be 1; multiple completions per request are not supported",
                "n",
            ));
        }
    }

    let prompts: Vec<String> = req
        .prompt
        .as_strings()
        .into_iter()
        .map(str::to_owned)
        .collect();
    if prompts.is_empty() {
        return Err(ApiError::bad_request(
            "prompt must contain at least one entry",
            "prompt",
        ));
    }
    if prompts.len() > MAX_COMPLETION_BATCH_SIZE {
        return Err(ApiError::bad_request(
            format!(
                "batch of {} prompts exceeds the maximum of {MAX_COMPLETION_BATCH_SIZE}; \
                 split the request into smaller batches",
                prompts.len()
            ),
            "prompt",
        ));
    }

    // RT-32 / SV-22: a non-empty `suffix` is a declared-but-dropped field —
    // reject it by name instead of pretending to support it. `stream: true`
    // itself is no longer rejected here (ORCHESTRATOR RULING D-3 supersedes
    // the wave-2 interim); see the streaming-specific guards below instead.
    if req.suffix.as_deref().is_some_and(|s| !s.is_empty()) {
        return Err(ApiError::bad_request(
            "suffix is not supported: the engine has no fill-in-the-middle generation mode, \
             so text after the completion cannot be honoured; omit suffix",
            "suffix",
        ));
    }

    // OpenAI frequency/presence penalties are applied for real over the
    // generated-token history. Validate them to the OpenAI `[-2.0, 2.0]`
    // range.
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(ApiError::bad_request(
            "frequency_penalty must be a finite number in the range [-2.0, 2.0]",
            "frequency_penalty",
        ));
    }
    if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(ApiError::bad_request(
            "presence_penalty must be a finite number in the range [-2.0, 2.0]",
            "presence_penalty",
        ));
    }
    let penalties = PenaltyParams::new(frequency_penalty, presence_penalty);

    // Optional temperature / top_p / repetition_penalty sampling overrides.
    // Validated here; *resolved* against the engine's own ambient
    // `SamplingParams` later, once a lease is available (gatekeeper
    // `REQUIRED #3` — see `ValidatedRequest`'s doc).
    if let Some(temperature) = req.temperature {
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::bad_request(
                "temperature must be a finite number in the range [0.0, 2.0]",
                "temperature",
            ));
        }
    }
    if let Some(top_p) = req.top_p {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(ApiError::bad_request(
                "top_p must be a finite number in the range (0.0, 1.0]",
                "top_p",
            ));
        }
    }
    // Gatekeeper `REQUIRED #3`: `repetition_penalty` was entirely absent
    // from this request type, so a client sending it got no validation and
    // no effect at all (silently dropped as an unknown field) — validated
    // identically to `ChatCompletionRequest`'s field of the same name
    // (`server.rs:624-630`, `>= 1.0`), not the earlier `> 0.0` this file and
    // `ExtendedChatRequest` used: the later gatekeeper text ("validated
    // identically to chat") supersedes the wave-3.5 `> 0.0` text, and a
    // value in `(0.0, 1.0)` would otherwise silently REWARD repeated
    // tokens instead of just failing to penalise them, on this endpoint
    // only — a real cross-endpoint discrepancy, not a cosmetic one.
    if let Some(rp) = req.repetition_penalty {
        if !rp.is_finite() || rp < 1.0 {
            return Err(ApiError::bad_request(
                "repetition_penalty must be a finite number >= 1.0",
                "repetition_penalty",
            ));
        }
    }
    // Whether the request customizes sampling at all. When it does not, the
    // vanilla `generate` path is used so default requests stay bit-identical
    // to the previous behavior; only customized requests take the
    // params+penalties path.
    let custom_sampling = penalties.is_active()
        || req.temperature.is_some()
        || req.top_p.is_some()
        || req.repetition_penalty.is_some();

    // B2-13 / ORCHESTRATOR RULING D-3: `stream: true` is real SSE, but only
    // for the shapes that have a streaming-capable engine seam — a single
    // prompt and no `seed` (no engine path streams deterministically-seeded
    // generation). `logprobs` is no longer in this list (`B8`):
    // `completions::stream::stream_completion_with_logprobs` builds a
    // dedicated per-token streaming decode + logit-capture loop directly
    // over `InferenceEngine`'s `pub(crate)` `model`/`kernel`/`sampler`
    // fields, so `stream + logprobs` is a real, honoured combination now,
    // not a rejected one. Each REMAINING unsupported combination is still
    // rejected by name rather than silently falling back to a
    // non-streaming response the client's `Accept: text/event-stream`
    // never expected.
    let stream = req.stream.unwrap_or(false);
    if stream && prompts.len() > 1 {
        return Err(ApiError::bad_request(
            "stream: true does not support a batched prompt (more than one entry); \
             send one prompt per streaming request",
            "stream",
        ));
    }
    if stream && req.seed.is_some() {
        return Err(ApiError::bad_request(
            "stream: true cannot be combined with seed (no engine path streams \
             deterministically-seeded generation yet); omit one",
            "stream",
        ));
    }
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage);

    // B9 correction (this file's own module doc claimed "no public seam" —
    // stale: `engine.rs:187`'s `pub(crate) sampler` field exists, and
    // `server/chat.rs`'s RT-26 fix / `api_extensions.rs`'s seeded-streaming
    // swap already use it the same way). `seed`, `temperature` and `top_p`
    // now ALL combine with `logprobs`: `create_completion`'s logprobs
    // branch swaps a resolved `SamplingParams` (and, when `seed` is set, a
    // freshly-seeded whole `Sampler`) onto `lease.sampler` for the
    // duration of the `generate_with_logprobs` call and restores it
    // unconditionally afterward — see that closure. No rejection needed
    // here any more; every one of `stop`/`suffix`/`logprobs`/`seed`/`user`
    // (and now `temperature`/`top_p`/`repetition_penalty` alongside
    // `logprobs`) is honoured for real.

    // An empty stop sequence would make `StopChecker::truncate_at_stop`'s
    // `text.find("")` match at position 0 of every completion — truncating
    // every response to the empty string. `StopChecker` lives in
    // `api_extensions.rs` (outside this file's ownership) and cannot be
    // changed here, so the filter belongs at this construction site. Before
    // this fix `stop` was never read at all, so `stop: [""]` was harmless by
    // omission; now that it is honoured, it must stay harmless by filtering.
    // `pipeline.rs`'s own `StopMatcher` already drops empty strings for the
    // identical reason (`stop_matcher_drops_empty_strings`).
    let stop_sequences: Vec<String> = req
        .stop
        .map(StopSequences::into_vec)
        .unwrap_or_default()
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();

    Ok(ValidatedRequest {
        prompts,
        max_tokens: req.max_tokens,
        penalties,
        req_temperature: req.temperature,
        req_top_p: req.top_p,
        req_repetition_penalty: req.repetition_penalty,
        custom_sampling,
        echo: req.echo.unwrap_or(false),
        seed: req.seed,
        logprobs_top_k: req.logprobs,
        stop_checker: StopChecker::new(stop_sequences),
        stream,
        include_usage,
    })
}

// ─── Handler ──────────────────────────────────────────────────────────────────

// `ActiveRequestGuard` — decrements `active_requests` on every exit from
// `create_completion`, including the validation early-return paths — now
// lives once, `pub(crate)`, in `server.rs` (wave-3.5 gatekeeper triage item
// (5)) rather than as a third independently-maintained copy here.

/// Handler for `POST /v1/completions`.
///
/// Runs the inference engine over the supplied prompt and returns an
/// OpenAI-compatible completion response.
#[tracing::instrument(skip(state))]
pub async fn create_completion(
    State(state): State<Arc<AppState>>,
    OpenAiJson(req): OpenAiJson<CompletionRequest>,
) -> Result<Response, ApiError> {
    let request_start = std::time::Instant::now();
    state.metrics().requests_total.inc();
    state.metrics().active_requests.inc();
    // Constructed once, here, so a validation error (the `?` below) is
    // still covered by its `Drop` — but *moved* into `StreamRequest` for
    // the streaming branch below rather than living out this function's
    // own scope, so the gauge decrements when the background generation
    // task actually finishes, not the moment this function returns the
    // initial SSE response object (which happens long before any content
    // has been streamed). The non-streaming branch keeps it as an ordinary
    // `_`-prefixed drop guard, unchanged from before.
    let _active_guard = ActiveRequestGuard(Arc::clone(state.metrics()));

    if let Some(user) = req.user.as_deref() {
        tracing::debug!(user, "completion request tagged with an end-user id");
    }

    let ValidatedRequest {
        prompts,
        max_tokens,
        penalties,
        req_temperature,
        req_top_p,
        req_repetition_penalty,
        custom_sampling,
        echo,
        seed,
        logprobs_top_k,
        stop_checker,
        stream,
        include_usage,
    } = validate_completion_request(req)?;

    // B2-13 / ORCHESTRATOR RULING D-3: real SSE for `stream: true`, split
    // into its own module both to keep this file under the workspace's
    // 2000-line ceiling and because the streaming and non-streaming paths
    // share almost no code beyond the request shape already destructured
    // above (validation already guarantees exactly one prompt here).
    if stream {
        let prompt_text = prompts.into_iter().next().unwrap_or_default();
        return stream::stream_completion(
            Arc::clone(&state),
            stream::StreamRequest {
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
                active_guard: _active_guard,
            },
        )
        .await;
    }

    // Tokenise every prompt up front, in the async context: BPE tokenisation
    // is not the CPU-bound step `sec-03` targets (generation is), so it stays
    // here rather than moving into `run_blocking_generation`'s closure —
    // mirroring `server.rs`'s own non-streaming path.
    let mut prompt_token_batches: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
    for (index, prompt_text) in prompts.iter().enumerate() {
        let prompt_tokens = if let Some(tok) = state.tokenizer() {
            tok.encode(prompt_text).map_err(|e| {
                tracing::error!(error = %e, index, "tokenisation failed");
                state.metrics().errors_total.inc();
                ApiError::internal(format!("tokenisation failed for prompt {index}: {e}"))
            })?
        } else {
            // Fallback: a single start token
            vec![151644u32]
        };
        prompt_token_batches.push(prompt_tokens);
    }
    let prompt_token_counts: Vec<usize> = prompt_token_batches.iter().map(Vec::len).collect();

    // One engine lease serves every prompt in the batch (reset between runs,
    // mirroring `api_extensions.rs`'s per-`n` reset), so the replica is held
    // for the whole request rather than re-acquired per prompt.
    let lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        state.metrics().errors_total.inc();
        ApiError::service_unavailable(format!("no inference engine replica is available: {e}"))
            .with_code("engine_unavailable")
    })?;

    // Gatekeeper `REQUIRED #1`/`REQUIRED #3`: seeded from the engine's own
    // ambient/startup `SamplingParams`, never `SamplingParams::default()` —
    // see the module doc and `resolve_sampling_params`'s doc for why this
    // still matters even though `SamplingParams::default`'s
    // `repetition_penalty` is `1.0` today.
    let sampling_params = resolve_sampling_params(
        lease.sampling_params(),
        req_temperature,
        req_top_p,
        req_repetition_penalty,
    );

    // sec-03: generation is synchronous, CPU-bound work and must not run on a
    // tokio worker thread. The *whole* batch runs inside ONE
    // `run_blocking_generation` call, not one call per prompt: the lease is
    // shared across every prompt in the batch, so wrapping each iteration
    // individually would still leave it (and the blocking-pool thread, for
    // everything between spawns) pinned to synchronous work for the whole
    // request. See `crate::server::blocking`.
    let state_for_generation = Arc::clone(&state);
    let generated: RuntimeResult<Vec<PromptOutcome>> =
        run_blocking_generation(lease, move |lease| {
            let mut outcomes: Vec<PromptOutcome> = Vec::with_capacity(prompt_token_batches.len());
            for prompt_tokens in &prompt_token_batches {
                lease.reset();
                let outcome = if let Some(top_k) = logprobs_top_k {
                    let id_to_token = |id: u32| -> String {
                        match state_for_generation.tokenizer() {
                            Some(tok) => tok.decode(&[id]).unwrap_or_else(|_| format!("<{id}>")),
                            None => format!("<{id}>"),
                        }
                    };
                    // B9: `generate_with_logprobs` has no per-call
                    // params/seed argument of its own — it always samples
                    // with whatever `lease.sampler` currently holds live.
                    // `sampler` is `pub(crate)` (`engine.rs`), so this
                    // crate swaps in the resolved `sampling_params` (and,
                    // when the client also set `seed`, a freshly-seeded
                    // whole `Sampler` carrying those same params — the
                    // identical swap `InferenceEngine::generate_with_seed`
                    // performs internally) for the duration of this one
                    // call and restores the previous sampler
                    // unconditionally afterward, exactly like
                    // `server/chat.rs`'s RT-26 fix and
                    // `api_extensions.rs`'s seeded-streaming swap already
                    // do. This is what lets `temperature`/`top_p`/`seed`
                    // combine with `logprobs` instead of the 400 an
                    // earlier revision of this endpoint required.
                    let prev_penalties = lease.penalties();
                    lease.set_penalties(penalties);
                    let result = if let Some(seed) = seed {
                        let mut fresh =
                            crate::sampling::Sampler::new(sampling_params.clone(), seed);
                        fresh.set_penalties(penalties);
                        let old_sampler = std::mem::replace(&mut lease.sampler, fresh);
                        let r = lease.generate_with_logprobs(
                            prompt_tokens,
                            max_tokens,
                            top_k,
                            &id_to_token,
                        );
                        lease.sampler = old_sampler;
                        r
                    } else {
                        let prev_params = lease.sampler.params().clone();
                        lease.sampler.set_params(sampling_params.clone());
                        let r = lease.generate_with_logprobs(
                            prompt_tokens,
                            max_tokens,
                            top_k,
                            &id_to_token,
                        );
                        lease.sampler.set_params(prev_params);
                        r
                    };
                    lease.set_penalties(prev_penalties);
                    result.map(|(tokens, logprobs)| PromptOutcome {
                        tokens,
                        logprobs: Some(logprobs),
                    })
                } else if let Some(seed) = seed {
                    // Each prompt in the batch gets a distinct seed (the base
                    // seed offset by its position), matching the `base_seed +
                    // i` convention `api_extensions.rs`'s
                    // `generate_n_completions` already uses for multiple
                    // draws in one request — otherwise two different prompts
                    // sharing the literal seed would be a surprising (if
                    // harmless) coupling, and two identical prompts in one
                    // batch would be indistinguishable from a single request
                    // repeated.
                    let per_prompt_seed = seed.wrapping_add(outcomes.len() as u64);
                    let prev_penalties = lease.penalties();
                    lease.set_penalties(penalties);
                    let result = lease
                        .generate_with_seed(
                            prompt_tokens,
                            max_tokens,
                            per_prompt_seed,
                            &sampling_params,
                        )
                        .map(|tokens| PromptOutcome {
                            tokens,
                            logprobs: None,
                        });
                    lease.set_penalties(prev_penalties);
                    result
                } else if custom_sampling {
                    lease
                        .generate_with_params_and_penalties(
                            prompt_tokens,
                            max_tokens,
                            &sampling_params,
                            &penalties,
                        )
                        .map(|tokens| PromptOutcome {
                            tokens,
                            logprobs: None,
                        })
                } else {
                    lease
                        .generate(prompt_tokens, max_tokens)
                        .map(|tokens| PromptOutcome {
                            tokens,
                            logprobs: None,
                        })
                };
                match outcome {
                    Ok(o) => outcomes.push(o),
                    Err(e) => return Err(e),
                }
            }
            Ok(outcomes)
        })
        .await?;

    let outcomes = generated.map_err(|e| {
        tracing::error!(error = %e, "generation failed");
        state.metrics().errors_total.inc();
        ApiError::internal(format!("generation failed: {e}"))
    })?;

    let mut choices: Vec<CompletionChoice> = Vec::with_capacity(prompts.len());
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    for (index, ((prompt_text, prompt_token_count), outcome)) in prompts
        .iter()
        .zip(prompt_token_counts.iter().copied())
        .zip(outcomes.into_iter())
        .enumerate()
    {
        total_prompt_tokens += prompt_token_count;
        let completion_token_count = outcome.tokens.len();
        total_completion_tokens += completion_token_count;

        // Decode output tokens to text
        let completion_text = if let Some(tok) = state.tokenizer() {
            tok.decode(&outcome.tokens).map_err(|e| {
                tracing::error!(error = %e, index, "decoding failed");
                state.metrics().errors_total.inc();
                ApiError::internal(format!(
                    "failed to decode the generated tokens for prompt {index}: {e}"
                ))
            })?
        } else {
            format!("{:?}", outcome.tokens)
        };

        let (truncated_completion, hit_stop) = stop_checker.truncate_at_stop(&completion_text);
        let truncated_chars = truncated_completion.chars().count();

        let logprobs = outcome.logprobs.map(|content| {
            let base_offset = if echo { prompt_text.chars().count() } else { 0 };
            build_completion_logprobs(&content, truncated_chars, base_offset)
        });

        choices.push(build_completion_choice(ChoiceInputs {
            index,
            prompt: prompt_text,
            completion: &truncated_completion,
            echo,
            completion_tokens: completion_token_count,
            max_tokens,
            hit_stop,
            logprobs,
        }));
    }
    // The engine lease was already dropped inside `run_blocking_generation`'s
    // closure once the last prompt's generation returned, so the pool can
    // hand the replica out again before the (comparatively cheap) response
    // assembly above even runs.

    state
        .metrics()
        .prompt_tokens_total
        .inc_by(total_prompt_tokens as u64);
    state
        .metrics()
        .tokens_generated_total
        .inc_by(total_completion_tokens as u64);

    let completion_id = format!("cmpl-{}", completion_id_from_nanos());
    let created = unix_timestamp_secs();
    // Report the real loaded-model id (resolved once from the engine via the
    // shared descriptor cache), not a hard-coded "bonsai-8b" literal — the same
    // mechanism the base `/v1/chat/completions` and `/v1/models` handlers use.
    let model_name = state.model_info().descriptor().await.id;

    let response = build_completion_response(
        &completion_id,
        &model_name,
        created,
        choices,
        total_prompt_tokens,
        total_completion_tokens,
    );

    let elapsed = request_start.elapsed().as_secs_f64();
    state.metrics().request_duration_seconds.observe(elapsed);

    Ok(Json(response).into_response())
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Inputs to [`build_completion_choice`], bundled into a struct so the
/// function stays under clippy's argument-count lint as this fix adds more
/// per-choice information (`hit_stop`, `logprobs`) alongside the original
/// fields.
struct ChoiceInputs<'a> {
    index: usize,
    prompt: &'a str,
    completion: &'a str,
    echo: bool,
    completion_tokens: usize,
    max_tokens: usize,
    /// Whether [`StopChecker::truncate_at_stop`] found a stop sequence in
    /// this choice's completion text — forces `finish_reason` to `"stop"`
    /// regardless of the token count.
    hit_stop: bool,
    logprobs: Option<CompletionLogprobs>,
}

/// Build a single [`CompletionChoice`] for one prompt in a (possibly
/// batched) request.
///
/// When `echo` is `true` the prompt text is prepended to `completion` in the
/// choice text so that the full context is visible to the caller.
fn build_completion_choice(inputs: ChoiceInputs<'_>) -> CompletionChoice {
    let ChoiceInputs {
        index,
        prompt,
        completion,
        echo,
        completion_tokens,
        max_tokens,
        hit_stop,
        logprobs,
    } = inputs;

    let text = if echo {
        format!("{prompt}{completion}")
    } else {
        completion.to_owned()
    };

    CompletionChoice {
        text,
        index,
        logprobs,
        finish_reason: determine_finish_reason(completion_tokens, max_tokens, hit_stop),
    }
}

/// Build the OpenAI-shaped `logprobs` object for one completion choice from
/// the engine's raw per-token capture, applying the same stop-sequence
/// truncation as the completion text itself so a client zipping
/// `tokens`/`token_logprobs`/`text_offset` never sees an entry describing
/// text beyond what `choice.text` actually contains.
///
/// `text_offset[i]` is the cumulative character offset of token `i` within
/// the *choice's* `text` field (i.e. including the echoed prompt's length
/// when `echo` is set), computed by summing each preceding token's own
/// decoded length — the standard legacy-completions convention. This is
/// exact whenever detokenizing one token at a time reproduces the same text
/// as batch-decoding the whole sequence; a tokenizer whose joint decode
/// inserts different inter-token whitespace than per-token decoding would
/// see it drift slightly, a known limitation of the legacy `text_offset`
/// contract in general, not specific to this endpoint.
fn build_completion_logprobs(
    content: &[LogprobsContent],
    truncated_completion_chars: usize,
    base_offset: usize,
) -> CompletionLogprobs {
    let mut tokens = Vec::with_capacity(content.len());
    let mut token_logprobs = Vec::with_capacity(content.len());
    let mut top_logprobs = Vec::with_capacity(content.len());
    let mut text_offset = Vec::with_capacity(content.len());

    let mut cursor = 0usize;
    for entry in content {
        if cursor >= truncated_completion_chars {
            // This and every later token starts at or after the stop-sequence
            // cut, so it describes text the client's `choice.text` no longer
            // contains.
            break;
        }
        text_offset.push(base_offset + cursor);
        tokens.push(entry.token.clone());
        token_logprobs.push(entry.logprob);

        let mut alt = serde_json::Map::with_capacity(entry.top_logprobs.len());
        for top in &entry.top_logprobs {
            alt.insert(top.token.clone(), serde_json::json!(top.logprob));
        }
        top_logprobs.push(serde_json::Value::Object(alt));

        cursor += entry.token.chars().count();
    }

    CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

/// Build a [`CompletionResponse`] from already-built per-prompt choices and
/// the batch's aggregated token usage.
fn build_completion_response(
    id: &str,
    model: &str,
    created: u64,
    choices: Vec<CompletionChoice>,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> CompletionResponse {
    CompletionResponse {
        id: id.to_owned(),
        object: "text_completion".to_owned(),
        created,
        model: model.to_owned(),
        choices,
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

/// Determine the finish reason.
///
/// A stop-sequence hit always reports `"stop"` regardless of token count;
/// otherwise `"length"` when `completion_tokens >= max_tokens` and `"stop"`
/// when the model produced an EOS token first.
fn determine_finish_reason(completion_tokens: usize, max_tokens: usize, hit_stop: bool) -> String {
    if !hit_stop && completion_tokens >= max_tokens {
        "length".to_owned()
    } else {
        "stop".to_owned()
    }
}

/// Return the current Unix timestamp in whole seconds.
fn unix_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Derive a short hex string from the current nanosecond timestamp for use as
/// a completion ID suffix.
fn completion_id_from_nanos() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{ts:x}")
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "completions_tests.rs"]
mod tests;
