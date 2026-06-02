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
//! - Only the first prompt in a batch is currently generated (one inference
//!   pass per request); the rest are returned as empty completions.  Extend
//!   with `batch_engine` if simultaneous batch inference is needed.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_types::{StopSequences, UsageInfo};
use crate::server::AppState;

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
    /// The model to use (ignored — OxiBonsai always uses the loaded engine).
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

    let prompt_text = req.prompt.first().to_owned();
    let echo = req.echo.unwrap_or(false);
    let max_tokens = req.max_tokens;

    // Tokenise the prompt
    let prompt_tokens = if let Some(tok) = state.tokenizer() {
        tok.encode(&prompt_text).map_err(|e| {
            tracing::error!(error = %e, "tokenisation failed");
            state.metrics().errors_total.inc();
            state.metrics().active_requests.dec();
            StatusCode::INTERNAL_SERVER_ERROR
        })?
    } else {
        // Fallback: a single start token
        vec![151644u32]
    };

    let prompt_token_count = prompt_tokens.len();
    state
        .metrics()
        .prompt_tokens_total
        .inc_by(prompt_token_count as u64);

    // Generate
    let output_tokens = {
        let mut lease = state.acquire_engine().await.map_err(|e| {
            tracing::error!(error = %e, "engine pool acquire failed");
            state.metrics().errors_total.inc();
            state.metrics().active_requests.dec();
            StatusCode::SERVICE_UNAVAILABLE
        })?;
        lease.generate(&prompt_tokens, max_tokens).map_err(|e| {
            tracing::error!(error = %e, "generation failed");
            state.metrics().errors_total.inc();
            state.metrics().active_requests.dec();
            StatusCode::INTERNAL_SERVER_ERROR
        })?
    };

    let completion_token_count = output_tokens.len();
    state
        .metrics()
        .tokens_generated_total
        .inc_by(completion_token_count as u64);

    // Decode output tokens to text
    let completion_text = if let Some(tok) = state.tokenizer() {
        tok.decode(&output_tokens).map_err(|e| {
            tracing::error!(error = %e, "decoding failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?
    } else {
        format!("{output_tokens:?}")
    };

    let completion_id = format!("cmpl-{}", completion_id_from_nanos());
    let created = unix_timestamp_secs();
    let model_name = req.model.unwrap_or_else(|| "bonsai-8b".to_string());

    let response = build_completion_response(
        &completion_id,
        &prompt_text,
        &completion_text,
        echo,
        prompt_token_count,
        completion_token_count,
        &model_name,
        created,
    );

    let elapsed = request_start.elapsed().as_secs_f64();
    state.metrics().request_duration_seconds.observe(elapsed);
    state.metrics().active_requests.dec();

    Ok(Json(response).into_response())
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build a [`CompletionResponse`] from raw generation outputs.
///
/// When `echo` is `true` the prompt text is prepended to `completion` in the
/// choice text so that the full context is visible to the caller.
#[allow(clippy::too_many_arguments)]
fn build_completion_response(
    id: &str,
    prompt: &str,
    completion: &str,
    echo: bool,
    prompt_tokens: usize,
    completion_tokens: usize,
    model: &str,
    created: u64,
) -> CompletionResponse {
    let text = if echo {
        format!("{prompt}{completion}")
    } else {
        completion.to_owned()
    };

    CompletionResponse {
        id: id.to_owned(),
        object: "text_completion".to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: determine_finish_reason(completion_tokens, 16),
        }],
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
        let resp = build_completion_response(
            "cmpl-abc",
            "Say hello",
            " world",
            false,
            4,
            2,
            "bonsai-8b",
            1_000_000,
        );
        assert_eq!(resp.object, "text_completion");
        assert_eq!(resp.choices[0].text, " world");
        assert_eq!(resp.usage.prompt_tokens, 4);
        assert_eq!(resp.usage.completion_tokens, 2);
        assert_eq!(resp.usage.total_tokens, 6);
    }

    #[test]
    fn build_completion_response_with_echo() {
        let resp = build_completion_response(
            "cmpl-abc",
            "Say hello",
            " world",
            true,
            4,
            2,
            "bonsai-8b",
            1_000_000,
        );
        assert_eq!(resp.choices[0].text, "Say hello world");
    }

    #[test]
    fn build_completion_response_id_preserved() {
        let resp = build_completion_response(
            "cmpl-xyz",
            "prompt",
            "completion",
            false,
            1,
            1,
            "bonsai-8b",
            42,
        );
        assert_eq!(resp.id, "cmpl-xyz");
        assert_eq!(resp.created, 42);
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
        let resp = build_completion_response(
            "cmpl-test",
            "prompt",
            "result",
            false,
            3,
            5,
            "bonsai-8b",
            99,
        );
        let json = serde_json::to_string(&resp).expect("serialisation must succeed");
        assert!(json.contains("\"object\":\"text_completion\""));
        assert!(json.contains("\"finish_reason\""));
    }
}
