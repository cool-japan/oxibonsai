//! OpenAI v1 completions endpoint (legacy, non-chat).
//!
//! Implements `POST /v1/completions` — the original text completion API that is
//! still widely used by clients that pre-date the chat-completions interface.
//!
//! # Request
//!
//! [`CompletionRequest`] declares every documented OpenAI completions field
//! plus this server's sampling extensions (`top_k`, `min_p`,
//! `repetition_penalty`), each honoured or refused by name (`SV-12`), and
//! denies every other field: a misspelt parameter is a `400` whose `param`
//! names it (`code: "unknown_parameter"`), never a silently ignored one.
//!
//! # Behaviour
//!
//! - Accepts a single prompt string **or** a batch of prompt strings (at
//!   most [`MAX_COMPLETION_BATCH_SIZE`]); every prompt is generated, one
//!   choice per prompt with `index` matching its position, sequentially on
//!   one leased engine replica (reset between prompts).
//! - **Sampling** — `temperature` / `top_p` / `top_k` /
//!   `repetition_penalty` are resolved against the replica's own ambient
//!   configuration (never `SamplingParams::default()`); with
//!   `frequency_penalty` / `presence_penalty` (`[-2.0, 2.0]`), `min_p`
//!   (omitted = the replica's baseline, `0.0` = disabled) and `seed` they
//!   are installed on the replica for each generation and restored
//!   afterwards (`crate::server::sampling_scope::RequestSampling`), on
//!   every path — `logprobs` and streaming included. A request that
//!   customizes nothing samples exactly as the replica would on its own.
//! - `seed` — prompt `i` of a batch runs on a fresh sampler seeded with
//!   `seed + i`, so a batch never couples two prompts' draws and a
//!   single-prompt request uses `seed` itself.
//! - `echo` — prepends the prompt text to its choice's text.
//! - `stop` — each completion is truncated at the first stop sequence and
//!   reports `finish_reason: "stop"` (`RT-32`); on a stream the match also
//!   cancels the generation.
//! - `logprobs` — per-token log probabilities in the legacy `{tokens,
//!   token_logprobs, top_logprobs, text_offset}` shape (`RT-32`), streamed
//!   or not.
//! - `stream: true` — real SSE (`text_completion` chunks through the chat
//!   endpoint's SSE machinery, see the `stream` submodule) for a single
//!   prompt or a batch: one stream carries every prompt's chunks, each with
//!   `choices[0].index` = its prompt's index, prompts generated in order,
//!   one `finish_reason` per index, then (with
//!   `stream_options.include_usage`) one usage chunk aggregated over the
//!   batch, then `[DONE]`.
//! - Declared but not supported, so refused with `400` naming the field: a
//!   non-empty `suffix` (no fill-in-the-middle mode), `n` other than `1`,
//!   `best_of` other than `1`, a non-empty `logit_bias`.
//! - `max_tokens` — bounded by [`crate::server::MAX_OUTPUT_TOKENS`].
//! - `user` — logged at `debug` level (an opaque end-user tag).
//! - Without a tokenizer a text prompt is tokenized to the configured prompt
//!   start token (or refused with `400 tokenizer_required`), and the
//!   completion text is empty — the same on both paths — while `usage` and
//!   `logprobs` still count every token.
//! - Every path answers with an `x-request-id` header, observes
//!   `request_duration_seconds` (a stream at its true end), and renders
//!   failures in the shared [`crate::server::api_error::ApiError`]
//!   envelope.

use axum::{
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_extensions::{resolve_sampling_params, StopChecker};
use crate::api_types::{LogprobsContent, StopSequences, UsageInfo};
use crate::error::RuntimeResult;
use crate::request_id::RequestId;
use crate::sampling::{PenaltyParams, SamplingParams};
use crate::server::api_error::{ApiError, OpenAiJson};
use crate::server::blocking::run_blocking_generation;
use crate::server::sampling_scope::RequestSampling;
use crate::server::{
    request_id_header_map, resolve_request_id, ActiveRequestGuard, AppState, StreamOptions,
    MAX_OUTPUT_TOKENS,
};

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
/// Follows the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions/create)
/// field for field — every documented member is declared and either
/// honoured or refused with a `400` naming it (`SV-12`) — plus this server's
/// sampling extensions (`top_k`, `min_p`, `repetition_penalty`). Any other
/// member is refused with `400` (`param` = the member, `code:
/// "unknown_parameter"`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    /// The model to use (ignored for generation — OxiBonsai always uses the
    /// loaded engine, and the response `model` reports that engine's real id).
    pub model: Option<String>,
    /// The prompt to complete: one string or a batch of strings.
    pub prompt: PromptInput,
    /// Text to append after the completion. Not supported — the engine has
    /// no fill-in-the-middle generation mode; a non-empty value is rejected
    /// with `400` naming this field. An empty string (or the field omitted)
    /// is accepted as a no-op.
    pub suffix: Option<String>,
    /// Maximum number of tokens to generate per completion.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature in `[0.0, 2.0]`.
    pub temperature: Option<f32>,
    /// Nucleus (top-p) sampling threshold in `(0.0, 1.0]`.
    pub top_p: Option<f32>,
    /// Number of completions per prompt; only `1` is supported.
    pub n: Option<usize>,
    /// Whether to stream the response as SSE (a batch included: one stream
    /// carries every prompt's chunks, indexed by prompt).
    pub stream: Option<bool>,
    /// OpenAI `stream_options`; only meaningful with `stream: true`. Only
    /// `include_usage` is honored: a final usage-only chunk, aggregated over
    /// every prompt, is emitted just before `[DONE]` when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Return the log probabilities of the top-N tokens at each step
    /// (streamed or not, `seed` included).
    pub logprobs: Option<usize>,
    /// If `true`, the prompt is echoed back at the start of the completion text.
    pub echo: Option<bool>,
    /// Sequences that terminate generation. Honoured via
    /// [`crate::api_extensions::StopChecker`] (a stream also cancels its
    /// generation on a match).
    pub stop: Option<StopSequences>,
    /// Penalise tokens that appear at least once in the generated text
    /// (`[-2.0, 2.0]`).
    pub presence_penalty: Option<f32>,
    /// Penalise tokens proportionally to their frequency in the generated
    /// text (`[-2.0, 2.0]`).
    pub frequency_penalty: Option<f32>,
    /// Server-side best-of-`n` selection. Only `1` (or omitting the field) is
    /// supported; any other value is rejected with `400` naming this field.
    pub best_of: Option<usize>,
    /// Per-token logit bias map, `{token_id_as_string: bias}`. A non-empty
    /// map is rejected with `400`: the sampling pipeline has no per-token
    /// bias stage. An empty map (or the field omitted) is a no-op.
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Opaque end-user identifier, logged at `debug` level but not otherwise
    /// processed.
    pub user: Option<String>,
    /// Random seed for deterministic generation, honoured on every path;
    /// prompt `i` of a batch is seeded with `seed + i`.
    pub seed: Option<u64>,
    /// Top-k filtering threshold (not a standard OpenAI Completions field,
    /// but accepted the same way vLLM does; `0` disables it). When omitted
    /// the engine's own startup value is used.
    pub top_k: Option<usize>,
    /// Min-p threshold in `[0.0, 1.0]` for this request only (not a standard
    /// OpenAI Completions field, but accepted the same way vLLM and
    /// llama.cpp do): omitted = the serving replica's baseline, `0.0` =
    /// disabled.
    pub min_p: Option<f32>,
    /// Repetition penalty (`1.0` = disabled; not a standard OpenAI
    /// Completions field, but accepted the same way vLLM does), validated
    /// `>= 1.0` exactly like `ChatCompletionRequest`'s field of the same
    /// name — a value below `1.0` would REWARD repeated tokens. When
    /// omitted, the engine's own startup value is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
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
/// Carries the raw `temperature`/`top_p`/`top_k`/`repetition_penalty`
/// *overrides* rather than a resolved [`SamplingParams`]: resolving them
/// needs the acquired engine lease's own ambient configuration
/// ([`resolve_sampling_params`]), which only exists once generation is
/// about to run.
struct ValidatedRequest {
    prompts: Vec<String>,
    max_tokens: usize,
    penalties: PenaltyParams,
    req_temperature: Option<f32>,
    req_top_p: Option<f32>,
    req_top_k: Option<usize>,
    req_repetition_penalty: Option<f32>,
    /// The request's `min_p` (`None` keeps the replica's baseline).
    min_p: Option<f32>,
    custom_sampling: bool,
    echo: bool,
    seed: Option<u64>,
    logprobs_top_k: Option<usize>,
    stop_checker: StopChecker,
    stream: bool,
    include_usage: bool,
}

impl ValidatedRequest {
    /// The sampling configuration of prompt `index` of this request on a
    /// replica whose ambient parameters are `ambient` — see the module doc.
    /// A request that customizes nothing keeps the replica's own penalties.
    fn sampling(&self, ambient: &SamplingParams, index: usize) -> RequestSampling {
        let mut params = resolve_sampling_params(
            ambient,
            self.req_temperature,
            self.req_top_p,
            self.req_repetition_penalty,
        );
        if let Some(top_k) = self.req_top_k {
            params.top_k = top_k;
        }
        let seed = self.seed.map(|seed| seed.wrapping_add(index as u64));
        let override_penalties =
            self.custom_sampling || seed.is_some() || self.logprobs_top_k.is_some() || self.stream;
        RequestSampling {
            params,
            penalties: override_penalties.then_some(self.penalties),
            min_p: self.min_p,
            seed,
        }
    }
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
    if let Some(best_of) = req.best_of {
        if best_of != 1 {
            return Err(ApiError::bad_request(
                "best_of must be 1: server-side best-of-n selection is not supported",
                "best_of",
            ));
        }
    }
    if req.logit_bias.as_ref().is_some_and(|bias| !bias.is_empty()) {
        return Err(ApiError::bad_request(
            "logit_bias is not supported: the sampling pipeline has no per-token bias stage; \
             omit logit_bias or send an empty object",
            "logit_bias",
        ));
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

    // RT-32 / SV-22: a non-empty `suffix` is a declared field the engine
    // cannot honour — reject it by name instead of pretending to support it.
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

    // Optional temperature / top_p / top_k / repetition_penalty / min_p
    // sampling overrides. Validated here; *resolved* against the engine's
    // own ambient `SamplingParams` once a lease is available (see
    // `ValidatedRequest`'s doc).
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
    if let Some(top_k) = req.top_k {
        // An absurdly large value is almost certainly a client error rather
        // than an intentional "disable filtering" request (that is `0`) —
        // the same bound `/v1/chat/completions` applies.
        if top_k > 1_000_000 {
            return Err(ApiError::bad_request(
                "top_k must be at most 1,000,000",
                "top_k",
            ));
        }
    }
    if let Some(min_p) = req.min_p {
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(ApiError::bad_request(
                "min_p must be a finite number in the range [0.0, 1.0]",
                "min_p",
            ));
        }
    }
    // Validated identically to `ChatCompletionRequest`'s field of the same
    // name (`>= 1.0`): a value in `(0.0, 1.0)` would REWARD repeated tokens
    // instead of just failing to penalise them.
    if let Some(rp) = req.repetition_penalty {
        if !rp.is_finite() || rp < 1.0 {
            return Err(ApiError::bad_request(
                "repetition_penalty must be a finite number >= 1.0",
                "repetition_penalty",
            ));
        }
    }
    // Whether the request customizes sampling at all. When it does not (and
    // sets no seed / logprobs / stream), the replica's own penalties stay in
    // force, so a default request samples exactly as the replica would on
    // its own.
    let custom_sampling = penalties.is_active()
        || req.temperature.is_some()
        || req.top_p.is_some()
        || req.top_k.is_some()
        || req.repetition_penalty.is_some();

    // `stream: true` is real SSE for a single prompt and for a batch (one
    // stream, chunks indexed by prompt).
    let stream = req.stream.unwrap_or(false);
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage);

    // An empty stop sequence would make `StopChecker::truncate_at_stop`'s
    // `text.find("")` match at position 0 of every completion — truncating
    // every response to the empty string — so it is dropped here, the way
    // `pipeline.rs`'s own `StopMatcher` drops it
    // (`stop_matcher_drops_empty_strings`).
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
        req_top_k: req.top_k,
        req_repetition_penalty: req.repetition_penalty,
        min_p: req.min_p,
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

/// Handler for `POST /v1/completions`.
///
/// Runs the inference engine over the supplied prompt(s) and returns an
/// OpenAI-compatible completion response (or SSE stream), tagged with an
/// `x-request-id` header (the client's own, when it sent a well-formed one —
/// see [`crate::server::resolve_request_id`]) on success and on error alike.
#[tracing::instrument(skip(state, headers), fields(request_id))]
pub async fn create_completion(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    OpenAiJson(req): OpenAiJson<CompletionRequest>,
) -> Result<Response, ApiError> {
    let request_id = resolve_request_id(&headers);
    tracing::Span::current().record("request_id", tracing::field::display(&request_id));
    create_completion_inner(state, req, request_id)
        .await
        .map_err(|err| err.with_request_id(request_id))
}

/// Tokenize every prompt of a validated request, in the async context
/// (tokenization is not the CPU-bound step `sec-03` targets — generation
/// is). Without a tokenizer every prompt becomes the configured prompt start
/// token, or the request is refused with `400 tokenizer_required`; the one
/// warning such a request gets is logged here.
fn tokenize_prompts(state: &AppState, prompts: &[String]) -> Result<Vec<Vec<u32>>, ApiError> {
    let mut batches: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
    for (index, prompt_text) in prompts.iter().enumerate() {
        let prompt_tokens = match state.tokenizer() {
            Some(tok) => tok.encode(prompt_text).map_err(|e| {
                tracing::error!(error = %e, index, "tokenisation failed");
                state.metrics().errors_total.inc();
                ApiError::internal(format!("tokenisation failed for prompt {index}: {e}"))
            })?,
            None => state.tokenizerless_prompt("prompt").inspect_err(|_| {
                state.metrics().errors_total.inc();
            })?,
        };
        batches.push(prompt_tokens);
    }
    if state.tokenizer().is_none() {
        crate::server::warn_generating_without_tokenizer("/v1/completions");
    }
    Ok(batches)
}

/// The handler proper, split out of [`create_completion`] so every error it
/// returns is tagged with the request id in one place.
async fn create_completion_inner(
    state: Arc<AppState>,
    req: CompletionRequest,
    request_id: RequestId,
) -> Result<Response, ApiError> {
    let request_start = std::time::Instant::now();
    state.metrics().requests_total.inc();
    state.metrics().active_requests.inc();
    // Constructed once, here, so a validation error (the `?` below) is
    // still covered by its `Drop` — but *moved* into the stream for the
    // streaming branch, so the gauge decrements when the background
    // generation actually finishes, not the moment this function returns
    // the initial SSE response object.
    let active_guard = ActiveRequestGuard(Arc::clone(state.metrics()));

    if let Some(user) = req.user.as_deref() {
        tracing::debug!(user, "completion request tagged with an end-user id");
    }

    let validated = validate_completion_request(req)?;
    let prompt_token_batches = tokenize_prompts(&state, &validated.prompts)?;

    if validated.stream {
        return stream::stream_completion(
            Arc::clone(&state),
            stream::StreamRequest {
                validated,
                prompt_token_batches,
                request_id,
                request_start,
                active_guard,
            },
        )
        .await;
    }
    let _active_guard = active_guard;
    let prompt_token_counts: Vec<usize> = prompt_token_batches.iter().map(Vec::len).collect();

    // One engine lease serves every prompt in the batch (reset between runs),
    // so the replica is held for the whole request rather than re-acquired
    // per prompt.
    let lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        state.metrics().errors_total.inc();
        ApiError::service_unavailable(format!("no inference engine replica is available: {e}"))
            .with_code("engine_unavailable")
    })?;

    // Prompt `i`'s sampling configuration, resolved against the replica's
    // own ambient parameters (never `SamplingParams::default()`).
    let samplings: Vec<RequestSampling> = (0..prompt_token_batches.len())
        .map(|index| validated.sampling(lease.sampling_params(), index))
        .collect();
    let max_tokens = validated.max_tokens;
    let logprobs_top_k = validated.logprobs_top_k;

    // sec-03: generation is synchronous, CPU-bound work and must not run on a
    // tokio worker thread. The *whole* batch runs inside ONE
    // `run_blocking_generation` call, not one call per prompt: the lease is
    // shared across every prompt in the batch. Each prompt installs its own
    // sampling configuration and restores the replica's afterwards.
    let state_for_generation = Arc::clone(&state);
    let generated: RuntimeResult<Vec<PromptOutcome>> =
        run_blocking_generation(lease, move |lease| {
            let mut outcomes: Vec<PromptOutcome> = Vec::with_capacity(prompt_token_batches.len());
            for (prompt_tokens, sampling) in prompt_token_batches.iter().zip(&samplings) {
                lease.reset();
                let outcome = sampling.run(lease, |engine| match logprobs_top_k {
                    Some(top_k) => {
                        let id_to_token = |id: u32| -> String {
                            match state_for_generation.tokenizer() {
                                Some(tok) => {
                                    tok.decode(&[id]).unwrap_or_else(|_| format!("<{id}>"))
                                }
                                None => format!("<{id}>"),
                            }
                        };
                        engine
                            .generate_with_logprobs(prompt_tokens, max_tokens, top_k, &id_to_token)
                            .map(|(tokens, logprobs)| PromptOutcome {
                                tokens,
                                logprobs: Some(logprobs),
                            })
                    }
                    None => {
                        engine
                            .generate(prompt_tokens, max_tokens)
                            .map(|tokens| PromptOutcome {
                                tokens,
                                logprobs: None,
                            })
                    }
                });
                outcomes.push(outcome?);
            }
            Ok(outcomes)
        })
        .await?;

    let outcomes = generated.map_err(|e| {
        tracing::error!(error = %e, "generation failed");
        state.metrics().errors_total.inc();
        ApiError::internal(format!("generation failed: {e}"))
    })?;

    let mut choices: Vec<CompletionChoice> = Vec::with_capacity(validated.prompts.len());
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    for (index, ((prompt_text, prompt_token_count), outcome)) in validated
        .prompts
        .iter()
        .zip(prompt_token_counts.iter().copied())
        .zip(outcomes)
        .enumerate()
    {
        total_prompt_tokens += prompt_token_count;
        let completion_token_count = outcome.tokens.len();
        total_completion_tokens += completion_token_count;

        // Decode output tokens to text; without a tokenizer there is no text
        // to render (the streaming path shows nothing either).
        let completion_text = match state.tokenizer() {
            Some(tok) => tok.decode(&outcome.tokens).map_err(|e| {
                tracing::error!(error = %e, index, "decoding failed");
                state.metrics().errors_total.inc();
                ApiError::internal(format!(
                    "failed to decode the generated tokens for prompt {index}: {e}"
                ))
            })?,
            None => String::new(),
        };

        let (truncated_completion, hit_stop) =
            validated.stop_checker.truncate_at_stop(&completion_text);
        // Only a stop-sequence cut drops the entries of tokens past the
        // visible text; otherwise every generated token keeps its entry —
        // also when there is no text at all (no tokenizer), exactly as the
        // streaming path releases them.
        let truncated_chars = if hit_stop {
            truncated_completion.chars().count()
        } else {
            usize::MAX
        };

        let logprobs = outcome.logprobs.map(|content| {
            let base_offset = if validated.echo {
                prompt_text.chars().count()
            } else {
                0
            };
            build_completion_logprobs(&content, truncated_chars, base_offset)
        });

        choices.push(build_completion_choice(ChoiceInputs {
            index,
            prompt: prompt_text,
            completion: &truncated_completion,
            echo: validated.echo,
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
    // shared descriptor cache), not a hard-coded literal — the same
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

    Ok((request_id_header_map(request_id), Json(response)).into_response())
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
