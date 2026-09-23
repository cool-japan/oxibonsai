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
//!   shape (`RT-32`; previously always `null`). `logprobs` is mutually
//!   exclusive with **both** `seed` and `temperature`/`top_p`: the engine has
//!   no seam that combines logit capture with either a seeded sampler or
//!   caller-supplied sampling parameters
//!   (`generate_with_logprobs` always samples with the engine's ambient,
//!   unmodified sampler — there is no public accessor to install custom
//!   `SamplingParams` on it first), so each combination is rejected with
//!   `400` naming the offending field rather than silently favouring one
//!   over the other. `frequency_penalty`/`presence_penalty` remain honoured
//!   alongside `logprobs` (applied via `set_penalties` before the
//!   logprobs-capturing decode loop runs), since that combination has no
//!   such gap.
//! - `stream: true` streams real SSE (see the `B2-13` section above and
//!   [`stream`]) for the single-prompt, non-`logprobs`, non-`seed` case;
//!   those three combinations are rejected with `400 Bad Request` naming
//!   the field, since none has a streaming-capable engine seam. A
//!   non-empty `suffix` is rejected the same way regardless of `stream`
//!   (`RT-32` / `SV-22`): the engine has no fill-in-the-middle generation
//!   mode. `stream: false` (or the field omitted, the common case) is
//!   unaffected by any of this.
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
    /// D-3). Real SSE for the single-prompt, non-`logprobs`, non-`seed`
    /// case; those three combinations are rejected with `400` naming the
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
    /// overrides). Validated `> 0.0`. When omitted, the engine's own
    /// startup value is used (never `SamplingParams::default`'s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Return the log probabilities for the top-N tokens at each step.
    /// Mutually exclusive with `seed` and with `stream: true` (see the
    /// module docs).
    pub logprobs: Option<usize>,
    /// If `true`, the prompt is echoed back at the start of the completion text.
    pub echo: Option<bool>,
    /// Random seed for deterministic generation. Mutually exclusive with
    /// `logprobs` and with `stream: true` (see the module docs).
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
    // identically to `ChatCompletionRequest`/`ExtendedChatRequest`'s field
    // of the same name.
    if let Some(rp) = req.repetition_penalty {
        if !rp.is_finite() || rp <= 0.0 {
            return Err(ApiError::bad_request(
                "repetition_penalty must be a finite number greater than 0.0",
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
    // prompt, no `logprobs` (`generate_streaming_with_params` has no
    // logit-capturing counterpart) and no `seed`
    // (`generate_streaming_with_params` has no seeded counterpart either;
    // see `crate::engine::InferenceEngine`). Each unsupported combination is
    // rejected by name rather than silently falling back to a non-streaming
    // response the client's `Accept: text/event-stream` never expected.
    let stream = req.stream.unwrap_or(false);
    if stream && prompts.len() > 1 {
        return Err(ApiError::bad_request(
            "stream: true does not support a batched prompt (more than one entry); \
             send one prompt per streaming request",
            "stream",
        ));
    }
    if stream && req.logprobs.is_some() {
        return Err(ApiError::bad_request(
            "stream: true cannot be combined with logprobs (no engine path streams \
             per-token log probabilities yet); omit one",
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

    // RT-32: `seed` and `logprobs` each have an existing engine seam
    // (`generate_with_seed`, `generate_with_logprobs`), but no *combined*
    // one: `generate_with_logprobs` samples with the engine's ambient
    // sampler rather than a freshly-seeded one, and this crate has no public
    // way to install a seeded sampler and then still reach the
    // logits-capturing decode loop (`InferenceEngine`'s sampler field is
    // private). Reject the combination honestly instead of silently
    // honouring only one of the two documented fields.
    if req.seed.is_some() && req.logprobs.is_some() {
        return Err(ApiError::bad_request(
            "seed and logprobs cannot both be honoured in the same /v1/completions request \
             (no engine path combines a seeded sampler with logit capture yet); omit one",
            "seed",
        ));
    }

    // RT-32 / SV-22: the same gap as the seed+logprobs guard immediately
    // above, for `temperature`/`top_p` instead of `seed`.
    // `generate_with_logprobs` always samples with the engine's ambient
    // sampler — there is no public seam to install caller-supplied
    // `SamplingParams` on it first (`InferenceEngine` exposes no
    // sampling-params setter; only `Sampler::params`/`set_params`, behind a
    // private field) — so without this guard a request combining `logprobs`
    // with `temperature`/`top_p` would silently take the logprobs branch and
    // generate at the engine's default sampler, dropping the client's
    // sampling customization without any error (the exact accept-and-drop
    // defect class this endpoint exists to remove). Frequency/presence
    // penalties are deliberately excluded from this guard: they ARE honoured
    // on the logprobs path (`set_penalties` is called before
    // `generate_with_logprobs` runs), so there is no dropped field to guard
    // against there.
    if req.logprobs.is_some() && (req.temperature.is_some() || req.top_p.is_some()) {
        return Err(ApiError::bad_request(
            "logprobs cannot be combined with temperature or top_p in the same \
             /v1/completions request (no engine path combines logit capture with a \
             customized sampler yet); omit logprobs, or omit temperature and top_p",
            "logprobs",
        ));
    }

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
                    let prev_penalties = lease.penalties();
                    lease.set_penalties(penalties);
                    let result = lease
                        .generate_with_logprobs(prompt_tokens, max_tokens, top_k, &id_to_token)
                        .map(|(tokens, logprobs)| PromptOutcome {
                            tokens,
                            logprobs: Some(logprobs),
                        });
                    lease.set_penalties(prev_penalties);
                    result
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
mod tests {
    use super::*;

    /// `Result::expect_err` requires `T: Debug` (to render the `Ok` value in
    /// its own panic message); `ValidatedRequest` intentionally does not
    /// derive `Debug` (it embeds `StopChecker`, which does not either since
    /// it lives in `api_extensions.rs`, outside this file's ownership), so
    /// this local helper extracts the error without that bound.
    fn expect_err(result: Result<ValidatedRequest, ApiError>) -> ApiError {
        match result {
            Ok(_) => panic!("expected validation to reject the request, but it succeeded"),
            Err(e) => e,
        }
    }

    fn base_request(prompt: PromptInput) -> CompletionRequest {
        CompletionRequest {
            model: None,
            prompt,
            max_tokens: 16,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stream_options: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            logprobs: None,
            echo: None,
            seed: None,
            suffix: None,
            user: None,
        }
    }

    #[test]
    fn prompt_input_single_as_strings() {
        let p = PromptInput::Single("hello world".to_string());
        assert_eq!(p.as_strings(), vec!["hello world"]);
    }

    #[test]
    fn prompt_input_batch_as_strings() {
        let p = PromptInput::Batch(vec!["foo".to_string(), "bar".to_string()]);
        assert_eq!(p.as_strings(), vec!["foo", "bar"]);
    }

    #[test]
    fn prompt_input_single_first() {
        let p = PromptInput::Single("hello".to_string());
        assert_eq!(p.first(), "hello");
    }

    #[test]
    fn prompt_input_batch_first() {
        let p = PromptInput::Batch(vec!["alpha".to_string(), "beta".to_string()]);
        assert_eq!(p.first(), "alpha");
    }

    #[test]
    fn prompt_input_empty_batch_first() {
        let p = PromptInput::Batch(vec![]);
        assert_eq!(p.first(), "");
    }

    #[test]
    fn build_completion_response_no_echo() {
        let choice = build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "Say hello",
            completion: " world",
            echo: false,
            completion_tokens: 2,
            max_tokens: 16,
            hit_stop: false,
            logprobs: None,
        });
        let resp =
            build_completion_response("cmpl-abc", "bonsai-8b", 1_000_000, vec![choice], 4, 2);
        assert_eq!(resp.object, "text_completion");
        assert_eq!(resp.choices[0].text, " world");
        assert_eq!(resp.usage.prompt_tokens, 4);
        assert_eq!(resp.usage.completion_tokens, 2);
        assert_eq!(resp.usage.total_tokens, 6);
    }

    #[test]
    fn build_completion_response_with_echo() {
        let choice = build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "Say hello",
            completion: " world",
            echo: true,
            completion_tokens: 2,
            max_tokens: 16,
            hit_stop: false,
            logprobs: None,
        });
        let resp =
            build_completion_response("cmpl-abc", "bonsai-8b", 1_000_000, vec![choice], 4, 2);
        assert_eq!(resp.choices[0].text, "Say hello world");
    }

    #[test]
    fn build_completion_response_id_preserved() {
        let choice = build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "prompt",
            completion: "completion",
            echo: false,
            completion_tokens: 1,
            max_tokens: 16,
            hit_stop: false,
            logprobs: None,
        });
        let resp = build_completion_response("cmpl-xyz", "bonsai-8b", 42, vec![choice], 1, 1);
        assert_eq!(resp.id, "cmpl-xyz");
        assert_eq!(resp.created, 42);
    }

    /// Regression test for finding 31: `build_completion_choice` must derive
    /// `finish_reason` from the *caller-supplied* `max_tokens`, not the
    /// hardcoded literal `16` the field defaults to. A request that
    /// overrides `max_tokens` away from `16` and is truncated exactly at
    /// that limit must report `"length"`, not `"stop"`.
    #[test]
    fn build_completion_response_uses_real_max_tokens_for_finish_reason() {
        // max_tokens = 5, completion_tokens = 5 (exhausted the limit) ->
        // "length". Under the old hardcoded-16 bug this would incorrectly
        // report "stop" because 5 < 16.
        let truncated = build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "prompt",
            completion: "completion",
            echo: false,
            completion_tokens: 5,
            max_tokens: 5,
            hit_stop: false,
            logprobs: None,
        });
        assert_eq!(truncated.finish_reason, "length");

        // max_tokens = 100, completion_tokens = 30 (stopped early on EOS,
        // well under the limit) -> "stop". Under the old hardcoded-16 bug
        // this would incorrectly report "length" because 30 >= 16.
        let natural_stop = build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "prompt",
            completion: "completion",
            echo: false,
            completion_tokens: 30,
            max_tokens: 100,
            hit_stop: false,
            logprobs: None,
        });
        assert_eq!(natural_stop.finish_reason, "stop");
    }

    /// A stop-sequence hit always reports `"stop"`, even if (by construction
    /// of the caller) `completion_tokens >= max_tokens` — the point of
    /// `hit_stop` is that it wins over the length-based determination.
    #[test]
    fn hit_stop_forces_stop_finish_reason_even_at_the_token_limit() {
        let choice = build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "prompt",
            completion: "trunc",
            echo: false,
            completion_tokens: 16,
            max_tokens: 16,
            hit_stop: true,
            logprobs: None,
        });
        assert_eq!(choice.finish_reason, "stop");
    }

    /// Regression test for finding serve-api-09: every prompt in a batch
    /// must produce its own [`CompletionChoice`] with a matching `index`,
    /// not just the first one.
    #[test]
    fn build_completion_response_batch_has_one_choice_per_prompt() {
        let choices = vec![
            build_completion_choice(ChoiceInputs {
                index: 0,
                prompt: "first",
                completion: "alpha",
                echo: false,
                completion_tokens: 1,
                max_tokens: 16,
                hit_stop: false,
                logprobs: None,
            }),
            build_completion_choice(ChoiceInputs {
                index: 1,
                prompt: "second",
                completion: "beta",
                echo: false,
                completion_tokens: 1,
                max_tokens: 16,
                hit_stop: false,
                logprobs: None,
            }),
            build_completion_choice(ChoiceInputs {
                index: 2,
                prompt: "third",
                completion: "gamma",
                echo: false,
                completion_tokens: 1,
                max_tokens: 16,
                hit_stop: false,
                logprobs: None,
            }),
        ];
        let resp = build_completion_response("cmpl-batch", "bonsai-8b", 1, choices, 3, 3);
        assert_eq!(resp.choices.len(), 3, "one choice per batch prompt");
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[0].text, "alpha");
        assert_eq!(resp.choices[1].index, 1);
        assert_eq!(resp.choices[1].text, "beta");
        assert_eq!(resp.choices[2].index, 2);
        assert_eq!(resp.choices[2].text, "gamma");
    }

    #[test]
    fn determine_finish_reason_stop() {
        assert_eq!(determine_finish_reason(8, 16, false), "stop");
    }

    #[test]
    fn determine_finish_reason_length() {
        assert_eq!(determine_finish_reason(16, 16, false), "length");
    }

    #[test]
    fn determine_finish_reason_hit_stop_overrides_length() {
        assert_eq!(determine_finish_reason(16, 16, true), "stop");
    }

    #[test]
    fn completion_id_from_nanos_nonempty() {
        let id = completion_id_from_nanos();
        assert!(!id.is_empty());
    }

    #[test]
    fn unix_timestamp_secs_nonzero() {
        let ts = unix_timestamp_secs();
        // Any reasonable Unix timestamp will be well above 0
        assert!(ts > 1_000_000_000);
    }

    #[test]
    fn serialise_completion_response() {
        let choice = build_completion_choice(ChoiceInputs {
            index: 0,
            prompt: "prompt",
            completion: "result",
            echo: false,
            completion_tokens: 5,
            max_tokens: 16,
            hit_stop: false,
            logprobs: None,
        });
        let resp = build_completion_response("cmpl-test", "bonsai-8b", 99, vec![choice], 3, 5);
        let json = serde_json::to_string(&resp).expect("serialisation must succeed");
        assert!(json.contains("\"object\":\"text_completion\""));
        assert!(json.contains("\"finish_reason\""));
    }

    // ── build_completion_logprobs ────────────────────────────────────────────

    fn logprobs_content(token: &str, logprob: f32) -> LogprobsContent {
        LogprobsContent {
            id: 0,
            token: token.to_string(),
            logprob,
            bytes: None,
            top_logprobs: vec![],
        }
    }

    #[test]
    fn build_completion_logprobs_untruncated_keeps_every_token() {
        let content = vec![logprobs_content("ab", -0.1), logprobs_content("cd", -0.2)];
        // "abcd" is 4 chars, nothing truncated.
        let logprobs = build_completion_logprobs(&content, 4, 0);
        assert_eq!(logprobs.tokens, vec!["ab", "cd"]);
        assert_eq!(logprobs.token_logprobs, vec![-0.1, -0.2]);
        assert_eq!(logprobs.text_offset, vec![0, 2]);
    }

    #[test]
    fn build_completion_logprobs_applies_base_offset_for_echo() {
        let content = vec![logprobs_content("hi", -0.1)];
        let logprobs = build_completion_logprobs(&content, 2, 10);
        assert_eq!(logprobs.text_offset, vec![10]);
    }

    #[test]
    fn build_completion_logprobs_drops_tokens_past_the_stop_truncation() {
        let content = vec![
            logprobs_content("ab", -0.1),
            logprobs_content("cd", -0.2),
            logprobs_content("ef", -0.3),
        ];
        // Only the first 3 characters ("abc") survived stop-truncation, so
        // the second token (starting at char 2, "cd") is still partially
        // visible and kept, but the third ("ef", starting at char 4) must be
        // dropped entirely.
        let logprobs = build_completion_logprobs(&content, 3, 0);
        assert_eq!(logprobs.tokens, vec!["ab", "cd"]);
        assert_eq!(logprobs.token_logprobs, vec![-0.1, -0.2]);
    }

    #[test]
    fn build_completion_logprobs_zero_length_truncation_drops_everything() {
        let content = vec![logprobs_content("ab", -0.1)];
        let logprobs = build_completion_logprobs(&content, 0, 0);
        assert!(logprobs.tokens.is_empty());
        assert!(logprobs.token_logprobs.is_empty());
        assert!(logprobs.text_offset.is_empty());
    }

    #[test]
    fn build_completion_logprobs_top_logprobs_are_json_objects() {
        let mut content = logprobs_content("a", -0.05);
        content.top_logprobs = vec![
            crate::api_types::TopLogprob {
                id: 0,
                token: "a".to_string(),
                logprob: -0.05,
                bytes: None,
            },
            crate::api_types::TopLogprob {
                id: 1,
                token: "b".to_string(),
                logprob: -1.2,
                bytes: None,
            },
        ];
        let logprobs = build_completion_logprobs(&[content], 1, 0);
        let obj = logprobs.top_logprobs[0]
            .as_object()
            .expect("top_logprobs entry must be a JSON object");
        // `top.logprob` is `f32`; round-tripping it through
        // `serde_json::json!` widens it to `f64`, so compare after narrowing
        // back to `f32` rather than against an `f64` literal (which is not
        // bit-identical to the widened `f32` value).
        assert_eq!(
            obj.get("a")
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32),
            Some(-0.05_f32)
        );
        assert_eq!(
            obj.get("b")
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32),
            Some(-1.2_f32)
        );
    }

    // ── validate_completion_request ──────────────────────────────────────────

    #[test]
    fn validate_accepts_a_plain_request() {
        let req = base_request(PromptInput::Single("hello".to_string()));
        let validated = validate_completion_request(req).expect("must validate");
        assert_eq!(validated.prompts, vec!["hello".to_string()]);
        assert!(!validated.custom_sampling);
        assert!(validated.seed.is_none());
        assert!(validated.logprobs_top_k.is_none());
    }

    #[test]
    fn validate_rejects_max_tokens_zero() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.max_tokens = 0;
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_rejects_max_tokens_over_the_ceiling() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.max_tokens = MAX_OUTPUT_TOKENS + 1;
        assert!(validate_completion_request(req).is_err());
    }

    #[test]
    fn validate_rejects_n_other_than_one() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.n = Some(2);
        assert!(validate_completion_request(req).is_err());
    }

    #[test]
    fn validate_accepts_n_equal_to_one() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.n = Some(1);
        assert!(validate_completion_request(req).is_ok());
    }

    #[test]
    fn validate_rejects_empty_batch() {
        let req = base_request(PromptInput::Batch(vec![]));
        assert!(validate_completion_request(req).is_err());
    }

    #[test]
    fn validate_rejects_batch_over_the_cap() {
        let prompts: Vec<String> = (0..(MAX_COMPLETION_BATCH_SIZE + 1))
            .map(|i| format!("p{i}"))
            .collect();
        let req = base_request(PromptInput::Batch(prompts));
        assert!(validate_completion_request(req).is_err());
    }

    /// `stream: true` is rejected naming the field (RT-32 / SV-22).
    #[test]
    fn validate_accepts_stream_true_alone() {
        // ORCHESTRATOR RULING D-3 (final) supersedes the wave-2 interim this
        // test used to assert (`stream: true` -> unconditional 400): a bare
        // `stream: true` with a single prompt and no `logprobs`/`seed` is now
        // valid and must resolve to the streaming branch.
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.stream = Some(true);
        let validated = validate_completion_request(req).expect("stream: true alone must validate");
        assert!(validated.stream);
    }

    /// The three combinations that remain rejected under D-3: a streaming
    /// request has no engine seam for a batch, `logprobs`, or `seed`.
    #[test]
    fn validate_rejects_stream_with_batch() {
        let mut req = base_request(PromptInput::Batch(vec!["a".to_string(), "b".to_string()]));
        req.stream = Some(true);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_rejects_stream_with_logprobs() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.stream = Some(true);
        req.logprobs = Some(2);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_rejects_stream_with_seed() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.stream = Some(true);
        req.seed = Some(7);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// `stream: false` is the common, explicit-default case and must not be
    /// rejected.
    #[test]
    fn validate_accepts_stream_false() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.stream = Some(false);
        assert!(validate_completion_request(req).is_ok());
    }

    /// `stream` omitted entirely must not be rejected.
    #[test]
    fn validate_accepts_stream_omitted() {
        let req = base_request(PromptInput::Single("hi".to_string()));
        assert!(validate_completion_request(req).is_ok());
    }

    /// A non-empty `suffix` is rejected naming the field (RT-32).
    #[test]
    fn validate_rejects_nonempty_suffix() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.suffix = Some("the end".to_string());
        assert!(validate_completion_request(req).is_err());
    }

    /// An empty-string `suffix` is a no-op, not a rejection — a client that
    /// always sends `suffix: ""` must not be broken by this fix.
    #[test]
    fn validate_accepts_empty_suffix() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.suffix = Some(String::new());
        assert!(validate_completion_request(req).is_ok());
    }

    #[test]
    fn validate_rejects_frequency_penalty_out_of_range() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.frequency_penalty = Some(3.0);
        assert!(validate_completion_request(req).is_err());
    }

    #[test]
    fn validate_rejects_presence_penalty_out_of_range() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.presence_penalty = Some(-3.0);
        assert!(validate_completion_request(req).is_err());
    }

    #[test]
    fn validate_rejects_non_finite_temperature() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.temperature = Some(f32::NAN);
        assert!(validate_completion_request(req).is_err());
    }

    #[test]
    fn validate_rejects_top_p_out_of_range() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.top_p = Some(0.0);
        assert!(validate_completion_request(req).is_err());
    }

    #[test]
    fn validate_marks_custom_sampling_when_temperature_set() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.temperature = Some(0.5);
        let validated = validate_completion_request(req).expect("must validate");
        assert!(validated.custom_sampling);
    }

    /// `seed` and `logprobs` together are rejected — there is no engine seam
    /// combining a seeded sampler with logit capture (RT-32).
    #[test]
    fn validate_rejects_seed_and_logprobs_together() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.seed = Some(42);
        req.logprobs = Some(3);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// `logprobs` and `temperature` together are rejected — `generate_with_logprobs`
    /// always samples with the engine's ambient sampler, so a caller-supplied
    /// `temperature` would otherwise be silently dropped without ever
    /// erroring (RT-32 / SV-22: the same defect class this file's other
    /// field-honouring fixes exist to remove).
    #[test]
    fn validate_rejects_logprobs_and_temperature_together() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.logprobs = Some(3);
        req.temperature = Some(0.1);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// Same guard, for `top_p` instead of `temperature`.
    #[test]
    fn validate_rejects_logprobs_and_top_p_together() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.logprobs = Some(3);
        req.top_p = Some(0.5);
        let err = expect_err(validate_completion_request(req));
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// `logprobs` combined with only frequency/presence penalties (no
    /// `temperature`/`top_p`) must still be ACCEPTED: penalties are honoured
    /// on the logprobs path via `set_penalties` before the logits-capturing
    /// decode loop runs, so there is no dropped field to guard against here
    /// — the new guard must not over-reject this working combination.
    #[test]
    fn validate_accepts_logprobs_and_penalties_together() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.logprobs = Some(3);
        req.frequency_penalty = Some(0.5);
        let validated = validate_completion_request(req).expect("must validate");
        assert_eq!(validated.logprobs_top_k, Some(3));
        assert!(validated.penalties.is_active());
    }

    #[test]
    fn validate_accepts_seed_alone() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.seed = Some(42);
        let validated = validate_completion_request(req).expect("must validate");
        assert_eq!(validated.seed, Some(42));
    }

    #[test]
    fn validate_accepts_logprobs_alone() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.logprobs = Some(3);
        let validated = validate_completion_request(req).expect("must validate");
        assert_eq!(validated.logprobs_top_k, Some(3));
    }

    #[test]
    fn validate_stop_sequences_reach_the_stop_checker() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.stop = Some(StopSequences::Single("STOP".to_string()));
        let validated = validate_completion_request(req).expect("must validate");
        assert!(!validated.stop_checker.is_empty());
        let (truncated, hit) = validated.stop_checker.truncate_at_stop("hello STOP world");
        assert!(hit);
        assert_eq!(truncated, "hello ");
    }

    #[test]
    fn validate_no_stop_sequences_yields_an_empty_stop_checker() {
        let req = base_request(PromptInput::Single("hi".to_string()));
        let validated = validate_completion_request(req).expect("must validate");
        assert!(validated.stop_checker.is_empty());
    }

    /// Regression: `StopChecker::truncate_at_stop` does `text.find("")`,
    /// which matches at position 0 of *any* string — an unfiltered empty
    /// stop sequence would truncate every completion to `""`. Before this
    /// fix `stop` was never read at all, so `stop: [""]` was harmless by
    /// omission; now that `stop` is honoured, an empty entry must be
    /// filtered out rather than newly breaking every response that sets it
    /// (mirroring `pipeline.rs`'s own `StopMatcher`, which drops empty
    /// strings for the identical reason).
    #[test]
    fn validate_filters_out_an_empty_stop_sequence() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.stop = Some(StopSequences::Single(String::new()));
        let validated = validate_completion_request(req).expect("must validate");
        assert!(
            validated.stop_checker.is_empty(),
            "an empty stop sequence must be dropped, not installed"
        );
        let (truncated, hit) = validated.stop_checker.truncate_at_stop("hello world");
        assert!(!hit, "an empty stop sequence must never match");
        assert_eq!(truncated, "hello world");
    }

    /// Same regression, in a batch that mixes an empty entry with a real one
    /// — only the empty entry is dropped, the real one still works.
    #[test]
    fn validate_filters_empty_stop_sequence_but_keeps_real_ones() {
        let mut req = base_request(PromptInput::Single("hi".to_string()));
        req.stop = Some(StopSequences::Multiple(vec![
            String::new(),
            "STOP".to_string(),
        ]));
        let validated = validate_completion_request(req).expect("must validate");
        assert!(!validated.stop_checker.is_empty());
        let (truncated, hit) = validated.stop_checker.truncate_at_stop("hello STOP world");
        assert!(hit);
        assert_eq!(truncated, "hello ");
    }

    // ── Handler-level (end-to-end) tests ─────────────────────────────────────
    //
    // Everything above exercises `validate_completion_request` and the pure
    // helper functions directly. The three headline `RT-32` / `SV-22`
    // behaviors — logprobs actually populated (not always `null`), `stream:
    // true` actually rejected, and a stop sequence actually truncating the
    // *decoded* text — only really live in the handler itself (the
    // `run_blocking_generation` closure, the post-generation decode +
    // `StopChecker::truncate_at_stop` pass), which the pure-function tests
    // cannot reach. These build a real router with the workspace's tiny test
    // model, the same pattern the sibling (unowned)
    // `tests/completions_tests.rs` integration suite and
    // `server/blocking.rs`'s own tests both already use.

    fn test_router() -> axum::Router {
        let config = oxibonsai_core::config::Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = crate::engine::InferenceEngine::new(config, params, 42);
        crate::server::create_router(engine, None)
    }

    /// POST `body` to `/v1/completions` on `app` and return (status, JSON).
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
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    /// `logprobs` must actually be populated, not the `null` SV-22 named.
    /// With no tokenizer configured (`test_router`'s `None`), per-token
    /// decoding falls back to `<id>` strings, so the exact token count
    /// varies with the tiny test model's own (deterministic, but
    /// implementation-detail) generation length — assert the field is
    /// *present* and internally consistent (equal-length parallel arrays),
    /// not an exact count.
    #[tokio::test]
    async fn handler_logprobs_field_is_populated_not_null() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "logprobs": 2 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);

        let logprobs = &json["choices"][0]["logprobs"];
        assert!(
            !logprobs.is_null(),
            "logprobs must be populated when requested, not always null (SV-22); got {json}"
        );
        let tokens = logprobs["tokens"].as_array().expect("tokens array");
        let token_logprobs = logprobs["token_logprobs"]
            .as_array()
            .expect("token_logprobs array");
        let top_logprobs = logprobs["top_logprobs"]
            .as_array()
            .expect("top_logprobs array");
        let text_offset = logprobs["text_offset"]
            .as_array()
            .expect("text_offset array");
        assert_eq!(tokens.len(), token_logprobs.len());
        assert_eq!(tokens.len(), top_logprobs.len());
        assert_eq!(tokens.len(), text_offset.len());
    }

    /// A request without `logprobs` must still report `null` — the fix adds
    /// the field, it does not turn it on unconditionally.
    #[tokio::test]
    async fn handler_logprobs_is_still_null_when_not_requested() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3 }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(json["choices"][0]["logprobs"].is_null());
    }

    /// `stream: true` alone now resolves to real SSE (ORCHESTRATOR RULING
    /// D-3, final — supersedes this test's former wave-2-interim
    /// assertion that it was unconditionally rejected with `400`): status
    /// is `200 OK`, and the body is SSE text (not the non-streaming JSON
    /// object shape), so `post_completion`'s `serde_json::from_slice` on it
    /// falls back to `Value::Null`. The detailed chunk-shape assertions
    /// (`text_completion` chunks, `[DONE]`, `stream_options.include_usage`)
    /// live in `stream::tests`, which reads the raw body as text rather
    /// than attempting a JSON parse.
    #[tokio::test]
    async fn handler_stream_true_alone_is_ok_not_400() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stream": true }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(
            json,
            serde_json::Value::Null,
            "an SSE body must not parse as a single JSON document"
        );
    }

    /// The three combinations that remain rejected under D-3 (batch,
    /// `logprobs`, `seed`) still return `400` naming `stream`, end to end.
    #[tokio::test]
    async fn handler_stream_with_logprobs_is_rejected_with_400() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({
                "prompt": "hello", "max_tokens": 3, "stream": true, "logprobs": 2
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["param"], "stream");
    }

    /// `logprobs` combined with `temperature` is rejected end-to-end with
    /// `400`, naming `logprobs` — not silently generating at the engine's
    /// ambient sampler while pretending the client's `temperature` was
    /// honoured (RT-32 / SV-22).
    #[tokio::test]
    async fn handler_logprobs_and_temperature_together_is_rejected_with_400() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({
                "prompt": "hello", "max_tokens": 3, "logprobs": 2, "temperature": 0.1
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["param"], "logprobs");
    }

    /// Same guard, for `top_p` instead of `temperature`.
    #[tokio::test]
    async fn handler_logprobs_and_top_p_together_is_rejected_with_400() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({
                "prompt": "hello", "max_tokens": 3, "logprobs": 2, "top_p": 0.5
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["param"], "logprobs");
    }

    /// `logprobs` combined with only a frequency/presence penalty must still
    /// succeed end-to-end — the new guard must not over-reject a combination
    /// that is actually honoured correctly (penalties are applied on the
    /// logprobs path via `set_penalties`).
    #[tokio::test]
    async fn handler_logprobs_and_frequency_penalty_together_still_succeeds() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({
                "prompt": "hello", "max_tokens": 3, "logprobs": 2, "frequency_penalty": 0.5
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(!json["choices"][0]["logprobs"].is_null());
    }

    /// A stop sequence actually truncates the *decoded* completion text
    /// end-to-end through the real handler, not just through
    /// `StopChecker` in isolation.
    ///
    /// With no tokenizer configured, the completion text is
    /// `format!("{output_tokens:?}")`, which for a `Vec<u32>` always starts
    /// with `'['` — even `"[]"` for zero generated tokens — regardless of
    /// the tiny test model's own generation length. Using `"["` as the stop
    /// sequence therefore gives a deterministic, model-behaviour-independent
    /// stop-hit: before this fix (`stop` never reaching the engine at all),
    /// the choice text would always be the full, non-empty `"[...]"` string.
    #[tokio::test]
    async fn handler_stop_sequence_truncates_the_completion_end_to_end() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 4, "stop": "[" }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let text = json["choices"][0]["text"].as_str().expect("text field");
        assert_eq!(
            text, "",
            "text must be truncated at the stop sequence found at position 0, got {text:?}"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    /// Regression guard for the empty-stop-sequence bug, end-to-end: an
    /// empty `stop` entry must not truncate every completion to `""`.
    #[tokio::test]
    async fn handler_empty_stop_sequence_does_not_truncate_everything() {
        let app = test_router();
        let (status, json) = post_completion(
            app,
            serde_json::json!({ "prompt": "hello", "max_tokens": 3, "stop": "" }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        let text = json["choices"][0]["text"].as_str().expect("text field");
        assert!(
            text.starts_with('['),
            "an empty stop sequence must not truncate the completion; got {text:?}"
        );
    }

    /// `sec-03` concurrency guard, pinned on this route specifically: a long
    /// `/v1/completions` generation must not block a concurrent `/health`
    /// request on the same runtime. The underlying seam
    /// (`run_blocking_generation`) already has its own unit tests in
    /// `server/blocking.rs`, but nothing previously exercised the property
    /// through the real `/v1/completions` handler; this mirrors
    /// `server_hardening_round4.rs`'s
    /// `a_second_request_is_served_while_a_long_generation_runs`, which
    /// pins the identical property for the base `/v1/chat/completions`
    /// route.
    #[tokio::test]
    async fn handler_long_completion_does_not_block_a_concurrent_health_check() {
        use std::time::{Duration, Instant};

        let app = test_router();

        // Calibrate the tiny engine's speed so the "long" completion is long
        // enough to be unambiguous on any machine without being needlessly
        // slow on a fast one.
        let probe_start = Instant::now();
        let (probe_status, _) = post_completion(
            app.clone(),
            serde_json::json!({ "prompt": "hello", "max_tokens": 4 }),
        )
        .await;
        assert_eq!(probe_status, axum::http::StatusCode::OK);
        let per_token = probe_start.elapsed() / 4;
        let target = Duration::from_millis(1_500);
        let long_tokens = (target.as_nanos() / per_token.as_nanos().max(1)).clamp(16, 400) as usize;

        let app_for_completion = app.clone();
        let completion = tokio::spawn(async move {
            let started = Instant::now();
            let (status, _) = post_completion(
                app_for_completion,
                serde_json::json!({ "prompt": "hello", "max_tokens": long_tokens }),
            )
            .await;
            (status, started.elapsed())
        });

        // Let the generation task actually start before timing the
        // concurrent request.
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(20)).await;

        let health_start = Instant::now();
        let health = {
            use axum::body::Body;
            use axum::http::Request;
            use tower::ServiceExt;
            app.oneshot(
                Request::get("/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await
            .expect("health response")
        };
        let health_elapsed = health_start.elapsed();
        assert_eq!(health.status(), axum::http::StatusCode::OK);

        let (completion_status, completion_elapsed) = completion.await.expect("completion task");
        assert_eq!(completion_status, axum::http::StatusCode::OK);
        assert!(
            health_elapsed < completion_elapsed,
            "the concurrent health check ({health_elapsed:?}) must finish before the long \
             completion ({completion_elapsed:?}); generation is blocking the runtime"
        );
    }
}
