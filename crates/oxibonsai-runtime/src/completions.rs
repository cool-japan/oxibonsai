//! OpenAI v1 completions endpoint (legacy, non-chat).
//!
//! Implements `POST /v1/completions` — the original text completion API that is
//! still widely used by clients that pre-date the chat-completions interface.
//!
//! # Behaviour
//!
//! - Accepts a single prompt string **or** a batch of prompt strings.
//! - `echo` — when `true`, prepends the original prompt text to each completion.
//! - `logprobs` — field is accepted and reflected back as `null` (logit values
//!   are not exposed at this layer; extend once the engine surfaces them).
//! - `stream` — accepted in the request but always runs non-streaming for now
//!   (the field is part of the OpenAI schema; full SSE streaming can be added
//!   by composing the same token-stream machinery used in `server.rs`).
//! - Every prompt in a batch is generated: the engine lease is held for the
//!   whole request and each prompt runs its own tokenize → generate → decode
//!   pass sequentially (`lease.reset()` between prompts, mirroring the
//!   extended chat endpoint's per-`n` reset in `api_extensions.rs`), and the
//!   response has one [`CompletionChoice`] per prompt with `index` matching
//!   its position in the batch. Batches larger than
//!   [`MAX_COMPLETION_BATCH_SIZE`] are rejected with `400 Bad Request`
//!   rather than silently truncated.
//! - `n` — only `n = 1` (the default) is supported; any other value is
//!   rejected with `400 Bad Request` rather than silently generating a single
//!   completion.
//! - `frequency_penalty` / `presence_penalty` — validated to the OpenAI
//!   `[-2.0, 2.0]` range and applied for real over the generated-token history
//!   via [`crate::sampling::PenaltyParams`] /
//!   [`crate::sampling::Sampler::sample_with_history`] (they were previously
//!   rejected with `400` because no such seam existed). `temperature` / `top_p`
//!   are likewise honored when supplied. A request that customizes none of
//!   these takes the vanilla decode path and is bit-identical to before.
//! - `max_tokens` — bounded by [`crate::server::MAX_OUTPUT_TOKENS`], the same
//!   ceiling the base `/v1/chat/completions` handler enforces; requests above
//!   it are rejected with `400 Bad Request`.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_types::{StopSequences, UsageInfo};
use crate::sampling::{PenaltyParams, SamplingParams};
use crate::server::{AppState, MAX_OUTPUT_TOKENS};

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
    /// Whether to stream the response as SSE (accepted but not yet used).
    pub stream: Option<bool>,
    /// Sequences that terminate generation.
    pub stop: Option<StopSequences>,
    /// Penalise tokens that appear at least once in the context.
    pub presence_penalty: Option<f32>,
    /// Penalise tokens proportional to their frequency.
    pub frequency_penalty: Option<f32>,
    /// Return the log probabilities for the top-N tokens at each step.
    pub logprobs: Option<usize>,
    /// If `true`, the prompt is echoed back at the start of the completion text.
    pub echo: Option<bool>,
    /// Random seed for deterministic generation.
    pub seed: Option<u64>,
    /// Text to append after the completion (not yet used in generation).
    pub suffix: Option<String>,
    /// Opaque end-user identifier (logged but not processed).
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
    /// Top-N alternative tokens at each position (as JSON objects).
    pub top_logprobs: Vec<serde_json::Value>,
    /// Character offset of each token within the completion text.
    pub text_offset: Vec<usize>,
}

/// A single completion choice.
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    /// The generated (and optionally echoed) text.
    pub text: String,
    /// Zero-based index among all returned choices.
    pub index: usize,
    /// Log-probability information (currently always `null`).
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

// ─── Handler ──────────────────────────────────────────────────────────────────

/// Build an OpenAI-compatible `400 Bad Request` JSON error response.
///
/// Delegates to the shared [`crate::http_error`] envelope so every route on the
/// server emits the identical `{"error": {message, type, param, code}}` shape.
fn bad_request(message: String, param: &str) -> Response {
    crate::http_error::bad_request(message, param)
}

/// Handler for `POST /v1/completions`.
///
/// Runs the inference engine over the supplied prompt and returns an
/// OpenAI-compatible completion response.
#[tracing::instrument(skip(state))]
pub async fn create_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, StatusCode> {
    let request_start = std::time::Instant::now();
    state.metrics().requests_total.inc();
    state.metrics().active_requests.inc();

    if req.max_tokens < 1 {
        state.metrics().active_requests.dec();
        return Ok(bad_request(
            "max_tokens must be at least 1".to_string(),
            "max_tokens",
        ));
    }
    if req.max_tokens > MAX_OUTPUT_TOKENS {
        state.metrics().active_requests.dec();
        return Ok(bad_request(
            format!(
                "max_tokens {} exceeds the maximum of {MAX_OUTPUT_TOKENS}",
                req.max_tokens
            ),
            "max_tokens",
        ));
    }
    if let Some(n) = req.n {
        if n != 1 {
            state.metrics().active_requests.dec();
            return Ok(bad_request(
                "n must be 1; multiple completions per request are not supported".to_string(),
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
        state.metrics().active_requests.dec();
        return Ok(bad_request(
            "prompt must contain at least one entry".to_string(),
            "prompt",
        ));
    }
    if prompts.len() > MAX_COMPLETION_BATCH_SIZE {
        state.metrics().active_requests.dec();
        return Ok(bad_request(
            format!(
                "batch of {} prompts exceeds the maximum of {MAX_COMPLETION_BATCH_SIZE}; \
                 split the request into smaller batches",
                prompts.len()
            ),
            "prompt",
        ));
    }
    // OpenAI frequency/presence penalties are now applied for real over the
    // generated-token history (previously they were rejected with `400` because
    // no sampler seam existed). Validate them to the OpenAI `[-2.0, 2.0]` range.
    let frequency_penalty = req.frequency_penalty.unwrap_or(0.0);
    let presence_penalty = req.presence_penalty.unwrap_or(0.0);
    if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
        state.metrics().active_requests.dec();
        return Ok(bad_request(
            "frequency_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
            "frequency_penalty",
        ));
    }
    if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
        state.metrics().active_requests.dec();
        return Ok(bad_request(
            "presence_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
            "presence_penalty",
        ));
    }
    let penalties = PenaltyParams::new(frequency_penalty, presence_penalty);

    // Optional temperature / top_p sampling overrides. Built on top of the
    // engine's defaults so an omitted field keeps the previous behavior.
    let mut sampling_params = SamplingParams::default();
    if let Some(temperature) = req.temperature {
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            state.metrics().active_requests.dec();
            return Ok(bad_request(
                "temperature must be a finite number in the range [0.0, 2.0]".to_string(),
                "temperature",
            ));
        }
        sampling_params.temperature = temperature;
    }
    if let Some(top_p) = req.top_p {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            state.metrics().active_requests.dec();
            return Ok(bad_request(
                "top_p must be a finite number in the range (0.0, 1.0]".to_string(),
                "top_p",
            ));
        }
        sampling_params.top_p = top_p;
    }
    // Whether the request customizes sampling at all. When it does not, the
    // vanilla `generate` path is used so default requests stay bit-identical to
    // the previous behavior; only customized requests take the params+penalties
    // path.
    let custom_sampling = penalties.is_active() || req.temperature.is_some() || req.top_p.is_some();

    let echo = req.echo.unwrap_or(false);
    let max_tokens = req.max_tokens;

    // One engine lease serves every prompt in the batch (reset between runs,
    // mirroring `api_extensions.rs`'s per-`n` reset), so the replica is held
    // for the whole request rather than re-acquired per prompt.
    let mut lease = state.acquire_engine().await.map_err(|e| {
        tracing::error!(error = %e, "engine pool acquire failed");
        state.metrics().errors_total.inc();
        state.metrics().active_requests.dec();
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let mut choices: Vec<CompletionChoice> = Vec::with_capacity(prompts.len());
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    for (index, prompt_text) in prompts.iter().enumerate() {
        // Tokenise this prompt
        let prompt_tokens = if let Some(tok) = state.tokenizer() {
            tok.encode(prompt_text).map_err(|e| {
                tracing::error!(error = %e, index, "tokenisation failed");
                state.metrics().errors_total.inc();
                state.metrics().active_requests.dec();
                StatusCode::INTERNAL_SERVER_ERROR
            })?
        } else {
            // Fallback: a single start token
            vec![151644u32]
        };
        let prompt_token_count = prompt_tokens.len();
        total_prompt_tokens += prompt_token_count;

        // Generate this prompt's completion. A vanilla request (no penalties,
        // no temperature/top_p override) uses the plain `generate` path so it
        // stays bit-identical to the previous behavior; a customized request
        // applies the sampling params and frequency/presence penalties for real
        // (both are restored on the engine after the call).
        lease.reset();
        let generate_result = if custom_sampling {
            lease.generate_with_params_and_penalties(
                &prompt_tokens,
                max_tokens,
                &sampling_params,
                &penalties,
            )
        } else {
            lease.generate(&prompt_tokens, max_tokens)
        };
        let output_tokens = generate_result.map_err(|e| {
            tracing::error!(error = %e, index, "generation failed");
            state.metrics().errors_total.inc();
            state.metrics().active_requests.dec();
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let completion_token_count = output_tokens.len();
        total_completion_tokens += completion_token_count;

        // Decode output tokens to text
        let completion_text = if let Some(tok) = state.tokenizer() {
            tok.decode(&output_tokens).map_err(|e| {
                tracing::error!(error = %e, index, "decoding failed");
                StatusCode::INTERNAL_SERVER_ERROR
            })?
        } else {
            format!("{output_tokens:?}")
        };

        choices.push(build_completion_choice(
            index,
            prompt_text,
            &completion_text,
            echo,
            completion_token_count,
            max_tokens,
        ));
    }
    // Return the engine to the pool before the (comparatively cheap) response
    // assembly below.
    drop(lease);

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
    state.metrics().active_requests.dec();

    Ok(Json(response).into_response())
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build a single [`CompletionChoice`] for one prompt in a (possibly
/// batched) request.
///
/// When `echo` is `true` the prompt text is prepended to `completion` in the
/// choice text so that the full context is visible to the caller.
/// `finish_reason` is derived from *this prompt's own* `completion_tokens`
/// against the shared `max_tokens` ceiling, so a batch where one prompt hits
/// the limit and another stops early reports each choice's `finish_reason`
/// independently.
fn build_completion_choice(
    index: usize,
    prompt: &str,
    completion: &str,
    echo: bool,
    completion_tokens: usize,
    max_tokens: usize,
) -> CompletionChoice {
    let text = if echo {
        format!("{prompt}{completion}")
    } else {
        completion.to_owned()
    };

    CompletionChoice {
        text,
        index,
        logprobs: None,
        finish_reason: determine_finish_reason(completion_tokens, max_tokens),
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

/// Determine the finish reason based on whether generation hit the limit.
///
/// Returns `"length"` when `completion_tokens >= max_tokens` and `"stop"`
/// otherwise (i.e. the model produced an EOS token).
fn determine_finish_reason(completion_tokens: usize, max_tokens: usize) -> String {
    if completion_tokens >= max_tokens {
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
        let choice = build_completion_choice(0, "Say hello", " world", false, 2, 16);
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
        let choice = build_completion_choice(0, "Say hello", " world", true, 2, 16);
        let resp =
            build_completion_response("cmpl-abc", "bonsai-8b", 1_000_000, vec![choice], 4, 2);
        assert_eq!(resp.choices[0].text, "Say hello world");
    }

    #[test]
    fn build_completion_response_id_preserved() {
        let choice = build_completion_choice(0, "prompt", "completion", false, 1, 16);
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
        let truncated = build_completion_choice(0, "prompt", "completion", false, 5, 5);
        assert_eq!(truncated.finish_reason, "length");

        // max_tokens = 100, completion_tokens = 30 (stopped early on EOS,
        // well under the limit) -> "stop". Under the old hardcoded-16 bug
        // this would incorrectly report "length" because 30 >= 16.
        let natural_stop = build_completion_choice(0, "prompt", "completion", false, 30, 100);
        assert_eq!(natural_stop.finish_reason, "stop");
    }

    /// Regression test for finding serve-api-09: every prompt in a batch
    /// must produce its own [`CompletionChoice`] with a matching `index`,
    /// not just the first one.
    #[test]
    fn build_completion_response_batch_has_one_choice_per_prompt() {
        let choices = vec![
            build_completion_choice(0, "first", "alpha", false, 1, 16),
            build_completion_choice(1, "second", "beta", false, 1, 16),
            build_completion_choice(2, "third", "gamma", false, 1, 16),
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
        assert_eq!(determine_finish_reason(8, 16), "stop");
    }

    #[test]
    fn determine_finish_reason_length() {
        assert_eq!(determine_finish_reason(16, 16), "length");
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
        let choice = build_completion_choice(0, "prompt", "result", false, 5, 16);
        let resp = build_completion_response("cmpl-test", "bonsai-8b", 99, vec![choice], 3, 5);
        let json = serde_json::to_string(&resp).expect("serialisation must succeed");
        assert!(json.contains("\"object\":\"text_completion\""));
        assert!(json.contains("\"finish_reason\""));
    }
}
