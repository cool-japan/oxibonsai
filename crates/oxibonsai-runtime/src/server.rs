//! OpenAI-compatible chat completions server.
//!
//! Provides an Axum-based HTTP server with the following endpoints:
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
//! | GET | `/v1/models` | List available models |
//! | GET | `/health` | Liveness probe |
//! | GET | `/metrics` | Prometheus text exposition |
//!
//! Use [`create_router`] or [`create_router_with_metrics`] to build
//! the Axum router, then serve it with `axum::serve`.
//!
//! ## `logprobs` sampling
//!
//! When a request sets `logprobs: true`, generation runs through the engine's
//! logits-capturing variant ([`InferenceEngine::generate_with_logprobs`]),
//! which — unlike the plain `generate*` methods — has no per-call
//! [`SamplingParams`] argument of its own. Per-request `temperature`/`top_p`
//! overrides are honored on this path too (`RT-26`): the handler temporarily
//! swaps the engine's sampler parameters onto the request's resolved overlay
//! for the duration of the call (the same swap-then-restore pattern
//! [`InferenceEngine::generate_with_params_and_penalties`] uses for the
//! non-`logprobs` path) and restores the previous value unconditionally
//! afterward, so a later request served by the same pool replica is
//! unaffected. Frequency/presence penalties are applied the same way, through
//! the separate `set_penalties` accessor. The returned logprobs are always
//! the model's real per-token log probabilities for the tokens that were
//! actually generated, under whichever configuration — ambient or
//! per-request override — actually produced them.

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};

use crate::engine::InferenceEngine;
use crate::engine_pool::{EngineLease, EnginePool, PoolError};
use crate::metrics::InferenceMetrics;
use crate::middleware::MiddlewareConfig;
use crate::multi_model::ModelRouter;
use crate::request_id::RequestId;
use crate::sampling::{PenaltyParams, SamplingParams};
use crate::tokenizer_bridge::TokenizerBridge;

pub mod api_error;
pub mod auth;
pub(crate) mod blocking;
pub mod budget;
pub mod lifecycle;
pub(crate) mod response_pipeline;
pub(crate) mod sampling_scope;
pub mod sanitize;
pub(crate) mod sse;

pub use api_error::{ApiError, OpenAiJson};
pub use auth::{AdminAuth, AuthConfig};
pub use budget::{validate_prompt_bytes, validate_request_budget, RequestLimits};
pub use lifecycle::{
    create_server, install_shutdown_signals, serve_with_shutdown, serve_with_shutdown_deadline,
    shutdown_signal, QueueDepthTracker, ServerConfig, DEFAULT_DRAIN_DEADLINE,
};
pub use sanitize::{build_prompt, neutralize_special_markers, SpecialTokenGuard};

use blocking::run_blocking_generation;

/// Hard upper bound on the client-requested output-token count for a single
/// chat request. Requests asking for more than this are rejected with
/// `400 Bad Request` instead of being allowed to drive an unbounded
/// `Vec::with_capacity(max_tokens)` allocation inside the engine (a single
/// oversized request can otherwise trip `handle_alloc_error` → `abort()`,
/// killing every concurrent request).
pub const MAX_OUTPUT_TOKENS: usize = 8192;

/// Maximum number of completion choices (`n`) the chat handler can produce.
/// Only single-choice generation is implemented, so any other value is
/// rejected rather than silently collapsed to one choice.
const MAX_N_CHOICES: usize = 1;

/// Decrements `active_requests` however a handler leaves — success, an
/// early-return validation error, or a mid-generation failure — by tying the
/// decrement to this guard's `Drop` rather than one `.dec()` call per exit
/// path (`SV-25`; a handler with several early returns reliably leaks the
/// gauge otherwise, since it is easy to add a new `return` and forget the
/// matching `.dec()`).
///
/// One implementation, `pub(crate)` here, used by `server/chat.rs`,
/// `api_extensions.rs`, `completions.rs` and `embeddings.rs` rather than an
/// independently-maintained copy in each.
pub(crate) struct ActiveRequestGuard(pub(crate) Arc<InferenceMetrics>);

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.0.active_requests.dec();
    }
}

/// Header name used for end-to-end request correlation. Request handlers
/// echo whatever the client supplied in the response, or generate a fresh
/// UUIDv4-style id when the header is absent.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Resolve a [`RequestId`] from an incoming request header, falling back to
/// a freshly generated id when none is supplied or when the supplied value
/// is malformed (in either case we still want a usable id to thread through
/// tracing spans and the response).
///
/// Accepts both the 32-hex form (no dashes) and the 36-char UUID form
/// (`8-4-4-4-12`).
pub fn resolve_request_id(headers: &HeaderMap) -> RequestId {
    if let Some(v) = headers.get(REQUEST_ID_HEADER) {
        if let Ok(s) = v.to_str() {
            if let Some(id) = RequestId::from_uuid(s).or_else(|| RequestId::from_hex(s)) {
                return id;
            }
        }
    }
    RequestId::new()
}

/// Build response headers for a [`RequestId`]. Returns a `HeaderMap` with the
/// `X-Request-ID` set to the canonical 36-char UUID form.
pub fn request_id_header_map(id: RequestId) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Ok(value) = HeaderValue::from_str(&id.as_uuid()) {
        headers.insert(REQUEST_ID_HEADER, value);
    }
    headers
}

/// Seconds since the Unix epoch, saturating to `0` on the (impossible) case of
/// a pre-epoch system clock.
fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// A real description of the model actually served by an [`EnginePool`].
///
/// The model's identity (name / architecture / context length) lives inside the
/// engine replicas held by the pool; reading it requires borrowing an engine,
/// which is only possible from async code. [`ServedModelInfo`] therefore
/// resolves this lazily on first request and caches it. `created` is a real
/// Unix timestamp captured once when the router is built.
#[derive(Debug, Clone, Serialize)]
pub struct ModelDescriptor {
    /// Model identifier — the loaded model's name (from GGUF metadata / config).
    pub id: String,
    /// Architecture tag (e.g. `"qwen3"`).
    pub architecture: String,
    /// Maximum context length in tokens.
    pub max_context_length: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Unix timestamp (seconds) captured when the server router was built.
    pub created: u64,
}

/// Resolves and caches the [`ModelDescriptor`] of the model served by a pool.
///
/// Shared (via `Arc`) between the chat/models handlers and the admin API so the
/// resolution happens at most once for the life of the server.
pub struct ServedModelInfo {
    engines: Arc<EnginePool>,
    created: u64,
    cache: OnceLock<ModelDescriptor>,
}

impl ServedModelInfo {
    fn new(engines: Arc<EnginePool>) -> Self {
        Self {
            engines,
            created: unix_now_secs(),
            cache: OnceLock::new(),
        }
    }

    /// Return the served model's descriptor, resolving it from a live engine
    /// replica on first call and caching it for every subsequent call.
    ///
    /// If the pool cannot hand out an engine (e.g. it is shutting down) a
    /// minimal `"unknown"` descriptor is returned without being cached, so a
    /// later call can still resolve the real value.
    pub async fn descriptor(&self) -> ModelDescriptor {
        if let Some(cached) = self.cache.get() {
            return cached.clone();
        }
        match self.engines.acquire().await {
            Ok(lease) => {
                // Read through the engine, which answers for a
                // dense and a hybrid (`qwen35`) model alike.
                let descriptor = ModelDescriptor {
                    id: lease.model_name().to_string(),
                    architecture: lease.architecture().to_string(),
                    max_context_length: lease.context_length(),
                    vocab_size: lease.vocab_size(),
                    created: self.created,
                };
                drop(lease);
                // First writer wins; concurrent callers converge on one value.
                let _ = self.cache.set(descriptor.clone());
                self.cache.get().cloned().unwrap_or(descriptor)
            }
            Err(_) => ModelDescriptor {
                id: "unknown".to_string(),
                architecture: "unknown".to_string(),
                max_context_length: 0,
                vocab_size: 0,
                created: self.created,
            },
        }
    }
}

/// Server state.
///
/// Holds a *pool* of inference-engine replicas behind a semaphore rather than a
/// single mutex, so up to `pool.size()` requests can generate concurrently. The
/// default path (a 1-element pool) is byte-identical to the previous
/// single-mutex design.
pub struct AppState {
    engines: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
    model_info: Arc<ServedModelInfo>,
    model_router: Option<Arc<ModelRouter>>,
    /// When `true` (the default), special-token markers (the `<|...|>` ChatML
    /// family, e.g. `<|im_start|>` / `<|im_end|>`) are stripped from user- and
    /// system-supplied message content before the chat template is assembled,
    /// so client text cannot forge fake role/turn boundaries once the merged
    /// prompt passes through the special-token-aware tokenizer
    /// (finding `security-03`). Disabled by setting the
    /// `OXI_DISABLE_PROMPT_SANITIZATION` environment variable.
    sanitize_prompt: bool,
    /// Control/special token ids that must never survive in client-supplied
    /// message content (finding `TOK-M2`). Resolved once from the loaded
    /// tokenizer's added vocabulary; empty when no tokenizer is attached.
    special_tokens: SpecialTokenGuard,
    /// Per-request admission limits: prompt size, context budget, deadlines
    /// (findings `sec-05` / `sec-08`).
    limits: RequestLimits,
    /// Authentication policy for the operator `/admin/*` surface
    /// (finding `sec-15`).
    auth: Arc<AuthConfig>,
    /// The `max_tokens` a request gets when it omits both `max_tokens` and
    /// `max_completion_tokens` (`SV-15(c)`). Sourced from
    /// `oxibonsai-serve`'s `sampling.default_max_tokens` config, which used
    /// to be validated at startup and then silently discarded in favor of a
    /// hardcoded literal.
    default_max_tokens: usize,
    /// The hard ceiling a request's effective `max_tokens` is rejected
    /// above (`SV-28`) — a pure allocation-safety backstop, independent of
    /// [`budget::validate_request_budget`]'s context-aware check (which
    /// already derives the *real* completion budget from the model's
    /// context length minus the prompt). Configurable; defaults to
    /// [`MAX_OUTPUT_TOKENS`].
    max_output_tokens_ceiling: usize,
    /// Workload rate aggregator surfaced via `/admin/workload-stats`
    /// (`SV-19`). Fed a sample per streaming request (the path with genuine
    /// per-token timing available to this handler); see
    /// [`chat_completions_stream`].
    rate_aggregator: Arc<crate::request_metrics::RequestRateAggregator>,
    /// KV-cache pressure policy surfaced via `/admin/cache-stats` (`SV-19`).
    /// Observed (never acted on: tier decisions are reported, not applied)
    /// once per request from the prompt's context utilization.
    kv_cache_policy: Arc<crate::kv_cache_policy::KvCachePolicy>,
    /// The single token id a tokenizer-less server feeds as the prompt of a
    /// text request ([`RouterOptions::with_prompt_start_token`]). `None`
    /// (the default) makes a tokenizer-less server refuse text prompts with
    /// `400 tokenizer_required` instead of guessing a vocabulary-specific
    /// id; ignored whenever a tokenizer is attached.
    prompt_start_token: Option<u32>,
}

impl AppState {
    /// Acquire an exclusive lease on one engine replica from the pool, waiting
    /// asynchronously if every replica is currently busy.
    ///
    /// The returned [`EngineLease`] derefs to the engine (so callers invoke the
    /// usual `generate*` methods) and returns it to the pool on drop.
    pub async fn acquire_engine(&self) -> Result<EngineLease, PoolError> {
        self.engines.acquire().await
    }

    /// Access the underlying engine pool.
    pub fn engines(&self) -> &Arc<EnginePool> {
        &self.engines
    }

    /// Access the optional tokenizer.
    pub fn tokenizer(&self) -> Option<&TokenizerBridge> {
        self.tokenizer.as_ref()
    }

    /// Access the shared metrics instance.
    pub fn metrics(&self) -> &Arc<InferenceMetrics> {
        &self.metrics
    }

    /// Access the served-model descriptor resolver.
    pub fn model_info(&self) -> &Arc<ServedModelInfo> {
        &self.model_info
    }

    /// Access the optional multi-model router used to answer `/v1/models`.
    pub fn model_router(&self) -> Option<&Arc<ModelRouter>> {
        self.model_router.as_ref()
    }

    /// The per-request admission limits in force.
    pub fn limits(&self) -> &RequestLimits {
        &self.limits
    }

    /// The authentication policy in force.
    pub fn auth(&self) -> &Arc<AuthConfig> {
        &self.auth
    }

    /// The control-token guard applied to client message content.
    pub fn special_tokens(&self) -> &SpecialTokenGuard {
        &self.special_tokens
    }

    /// Whether message-content special-token sanitization is enabled (see the
    /// [`AppState::sanitize_prompt`] field docs).
    pub fn sanitize_prompt(&self) -> bool {
        self.sanitize_prompt
    }

    /// The configured `max_tokens` default (`SV-15(c)`); see the field docs.
    pub fn default_max_tokens(&self) -> usize {
        self.default_max_tokens
    }

    /// The configured hard `max_tokens` ceiling (`SV-28`); see the field docs.
    pub fn max_output_tokens_ceiling(&self) -> usize {
        self.max_output_tokens_ceiling
    }

    /// The workload rate aggregator (`SV-19`).
    pub fn rate_aggregator(&self) -> &Arc<crate::request_metrics::RequestRateAggregator> {
        &self.rate_aggregator
    }

    /// The KV-cache pressure policy (`SV-19`).
    pub fn kv_cache_policy(&self) -> &Arc<crate::kv_cache_policy::KvCachePolicy> {
        &self.kv_cache_policy
    }

    /// The prompt token a tokenizer-less server feeds for a text prompt
    /// ([`RouterOptions::with_prompt_start_token`]), if one was configured.
    pub fn prompt_start_token(&self) -> Option<u32> {
        self.prompt_start_token
    }

    /// The prompt ids for a text prompt on a server with **no** tokenizer:
    /// the configured [`Self::prompt_start_token`], or the typed refusal a
    /// tokenizer-less server answers with — `400 invalid_request_error`,
    /// code `tokenizer_required`, naming `param` (`"messages"` for chat,
    /// `"prompt"` for completions). A server with a tokenizer never calls
    /// this: it encodes the text.
    pub(crate) fn tokenizerless_prompt(&self, param: &str) -> Result<Vec<u32>, ApiError> {
        match self.prompt_start_token {
            Some(id) => Ok(vec![id]),
            None => Err(tokenizer_required_error(param)),
        }
    }
}

/// `400 tokenizer_required`: a text prompt reached a server that has no
/// tokenizer to encode it with and no configured prompt start token.
pub(crate) fn tokenizer_required_error(param: &str) -> ApiError {
    ApiError::bad_request(
        "this server has no tokenizer attached, so a text prompt cannot be tokenized; start the \
         server with the model's tokenizer (a GGUF that embeds its vocabulary, or a \
         tokenizer.json)",
        param,
    )
    .with_code(TOKENIZER_REQUIRED_CODE)
}

/// The `error.code` of the `400` a tokenizer-less server answers a text
/// prompt with when no prompt start token is configured
/// ([`RouterOptions::with_prompt_start_token`]).
pub const TOKENIZER_REQUIRED_CODE: &str = "tokenizer_required";

/// Log the one warning a request generated without a tokenizer gets: its
/// tokens cannot be rendered as text, so the response's text is empty —
/// streamed or not — while `usage` (and `logprobs`, where requested) still
/// count every token.
pub(crate) fn warn_generating_without_tokenizer(endpoint: &str) {
    tracing::warn!(
        endpoint,
        "generating without a tokenizer: the generated tokens cannot be rendered as text, so \
         this response's text is empty (attach the model's tokenizer to see its output)"
    );
}

/// Resolve whether chat-prompt sanitization should be enabled.
///
/// Enabled by default; disabled when the `OXI_DISABLE_PROMPT_SANITIZATION`
/// environment variable is present and set to a truthy value (`1`, `true`,
/// `yes`, or `on`, case-insensitive). Any other value — including an empty or
/// unset variable — leaves sanitization on.
fn resolve_prompt_sanitization() -> bool {
    match std::env::var("OXI_DISABLE_PROMPT_SANITIZATION") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => true,
    }
}

/// Chat message (OpenAI-compatible).
///
/// `content` is `Option<String>` so that it can be `null` when `tool_calls`
/// is set (the model produced a tool call instead of a text reply).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender: `"system"`, `"user"`, `"assistant"`, `"tool"`.
    pub role: String,
    /// Text content of the message.  `null` when the assistant returns tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The model's reasoning (`<think>` span) for an assistant message
    /// (RT-10): filled on a response when the model reasoned, and
    /// accepted on a replayed assistant turn of a request (the chat template
    /// re-renders it). Omitted when absent — never an empty string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls produced by the model (assistant role only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::api_types::ToolCallResult>>,
    /// ID of the tool call being responded to (tool role only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Construct a plain text assistant or user message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Chat completion request.
///
/// Fields outside this list are accepted and ignored rather than refused:
/// the endpoint also reads `chat_template_kwargs`, `enable_thinking`,
/// `reasoning_effort` and the raw `tools` text from the same body
/// (`ChatRequestExtras`), and OpenAI clients send further optional members
/// (`parallel_tool_calls`, `metadata`, `store`, …), so this struct's field
/// list is not the request's whole vocabulary and cannot be declared
/// exhaustively.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Conversation history.
    pub messages: Vec<ChatMessage>,
    /// Model identifier (OpenAI compatibility field). This server is
    /// single-model per process, so the value is not used to route the
    /// request — it is validated against [`ModelRouter`] when one is
    /// attached (`SV-12`), and otherwise accepted and ignored, matching
    /// every real OpenAI client that always sends it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Maximum tokens to generate (the deprecated OpenAI name). `None` means
    /// "use the server's configured default"
    /// ([`AppState::default_max_tokens`]) unless [`Self::max_completion_tokens`]
    /// is set, which takes precedence when both are present (`SV-15(c)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// Maximum tokens to generate (the current OpenAI name, `SV-12`). Takes
    /// precedence over [`Self::max_tokens`] when both are present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<usize>,
    /// Sampling temperature. Validated to `[0.0, 2.0]` before use.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Optional nucleus-sampling threshold. Validated to `(0.0, 1.0]` when
    /// present; when omitted the engine's startup `top_p` default is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Optional top-k filtering threshold (`RT-23`; not a standard OpenAI
    /// field, but accepted the same way vLLM/text-generation-inference do).
    /// `0` disables it. When omitted the engine's startup `top_k` default is
    /// used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Optional min-p (probabilistic nucleus) threshold (`RT-23`; not a
    /// standard OpenAI field, but accepted the same way vLLM and llama.cpp
    /// do). Validated to `[0.0, 1.0]`. Applied for this request only, on
    /// the leased replica's sampler; omitting it samples with the replica's
    /// own baseline (the server's configured `min_p`), and `0.0` disables
    /// min-p for the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    /// Optional repetition penalty (`RT-23`; not a standard OpenAI field, but
    /// accepted the same way vLLM does, alongside the standard
    /// `frequency_penalty`/`presence_penalty`). `1.0` disables it; must be
    /// `>= 1.0`. When omitted the engine's startup value is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Optional number of completions to generate. Only `n = 1` is supported;
    /// any other value is rejected with `400 Bad Request`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    /// Whether to stream the response as SSE.
    #[serde(default)]
    pub stream: bool,
    /// OpenAI `stream_options`. Only `include_usage` is honored: when set, the
    /// stream emits a final chunk carrying real token usage just before
    /// `[DONE]` (finding `sec-08`), and every other chunk carries
    /// `"usage": null` as the OpenAI contract requires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// One or more sequences that stop generation when they appear in the
    /// generated text (`SV-12`/`RT-04`). Applied by truncating the decoded
    /// text at the first match; on the streaming path this also cancels the
    /// in-flight generation early (see `server::chat::chat_completions_stream`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<crate::api_types::StopSequences>,
    /// Deterministic sampling seed (`SV-12` / `RT-12`): the generation runs
    /// on a freshly seeded sampler carrying the request's params, penalties
    /// and effective `min_p`, and the replica's own sampler is restored
    /// afterwards — streamed or not, `logprobs` included — so two identical
    /// requests with the same seed produce the same tokens (and the same
    /// logprobs). Omitting it leaves the replica's ambient PRNG advancing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// The format the model's response must follow (`SV-12`). Only
    /// `{"type": "text"}` (or omitting the field, OpenAI's default) is
    /// honored on this endpoint; any other `format_type` is rejected with
    /// `400` rather than silently generating unconstrained text (the
    /// extended endpoint implements JSON mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<crate::api_types::ResponseFormat>,
    /// Per-token logit bias map, `{token_id_as_string: bias}` (`SV-12`). A
    /// non-empty map is rejected with `400`: the sampling pipeline has no
    /// per-token bias stage. An empty map (or the field's absence) is a
    /// no-op and is accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Opaque end-user identifier for abuse monitoring (`SV-12`). Accepted
    /// and ignored — it carries no decoding behavior in the OpenAI spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Tools available to the model, rendered into the prompt by the chat
    /// template. The model's `<tool_call>` blocks come back as
    /// `message.tool_calls` — or, streamed, as one `tool_calls` delta per
    /// completed call — with `finish_reason: "tool_calls"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<crate::api_types::ToolDefinition>>,
    /// Tool choice: `"auto"`, `"none"`, or a specific function selector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI `frequency_penalty` in `[-2.0, 2.0]`. Applied for real over the
    /// generated-token history via the sampler's penalty seam (no longer
    /// silently ignored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// OpenAI `presence_penalty` in `[-2.0, 2.0]`. Applied for real (see
    /// [`ChatCompletionRequest::frequency_penalty`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Whether to return per-token log probabilities for the generated tokens.
    /// Honored on the non-streaming path via the engine's logits-capturing
    /// generate variant; streaming + `logprobs` is rejected with `400`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top alternative tokens to report per position (OpenAI
    /// `top_logprobs`, `0..=20`). Requires `logprobs: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
}

/// OpenAI `stream_options` object.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Emit a final usage-only chunk before `[DONE]`.
    #[serde(default)]
    pub include_usage: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}

/// The default `max_tokens` the chat handler applies when a request omits it.
/// Exposed so the admin `/admin/config` endpoint can report the real running
/// default from a single source of truth instead of a duplicated literal.
pub fn default_max_tokens_value() -> usize {
    default_max_tokens()
}

/// The default sampling temperature the chat handler applies when a request
/// omits it. See [`default_max_tokens_value`].
pub fn default_temperature_value() -> f32 {
    default_temperature()
}

/// Resolve the effective completion-length budget for a request (`SV-15(c)`,
/// `SV-12`'s `max_completion_tokens`).
///
/// Precedence, matching the current OpenAI contract: `max_completion_tokens`
/// (the modern field) wins when present, then the deprecated `max_tokens`,
/// and finally `server_default` — the server's configured default
/// ([`AppState::default_max_tokens`]) — when the client sent neither. This
/// replaces the previous hardcoded `default_max_tokens()` fallback, which
/// ignored `sampling.default_max_tokens` entirely even though
/// `oxibonsai-serve` validated it at startup.
pub fn resolve_effective_max_tokens(req: &ChatCompletionRequest, server_default: usize) -> usize {
    req.max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(server_default)
}

/// Validate the client-supplied sampling / limit parameters of a chat request.
///
/// Returns the offending `(message, param)` on failure so the caller can emit
/// an OpenAI-style `400 Bad Request`. This is the single guard that keeps an
/// unbounded `max_tokens` from reaching `Vec::with_capacity` in the engine and
/// rejects out-of-range `temperature` / `top_p` (rather than silently coercing
/// a negative temperature to greedy decoding).
///
/// `effective_max_tokens` is the already-resolved completion-length budget
/// (`max_completion_tokens` if present, else `max_tokens`, else the server's
/// configured default — see [`resolve_effective_max_tokens`]), and
/// `max_output_tokens_ceiling` is the server's configurable hard ceiling
/// (`SV-28`, [`AppState::max_output_tokens_ceiling`]) that replaces the old
/// hardcoded [`MAX_OUTPUT_TOKENS`] constant (still the compiled-in default).
fn validate_chat_request(
    req: &ChatCompletionRequest,
    effective_max_tokens: usize,
    max_output_tokens_ceiling: usize,
) -> Result<(), (String, &'static str)> {
    if effective_max_tokens < 1 {
        return Err(("max_tokens must be at least 1".to_string(), "max_tokens"));
    }
    if effective_max_tokens > max_output_tokens_ceiling {
        return Err((
            format!(
                "max_tokens {effective_max_tokens} exceeds the server's configured maximum of \
                 {max_output_tokens_ceiling}"
            ),
            "max_tokens",
        ));
    }
    if let Some(top_k) = req.top_k {
        // No finite-range check needed (an unsigned count), but an
        // absurdly large value is almost certainly a client error rather
        // than an intentional "disable filtering" request (that is `0`),
        // and would otherwise silently behave exactly like `top_k: 0` once
        // it exceeds the vocabulary size — reject it explicitly instead.
        if top_k > 1_000_000 {
            return Err(("top_k must be at most 1,000,000".to_string(), "top_k"));
        }
    }
    if let Some(min_p) = req.min_p {
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err((
                "min_p must be a finite number in the range [0.0, 1.0]".to_string(),
                "min_p",
            ));
        }
    }
    if let Some(rp) = req.repetition_penalty {
        if !rp.is_finite() || rp < 1.0 {
            return Err((
                "repetition_penalty must be a finite number >= 1.0".to_string(),
                "repetition_penalty",
            ));
        }
    }
    if let Some(bias) = &req.logit_bias {
        if !bias.is_empty() {
            return Err((
                "logit_bias is not yet supported by this server: no per-token bias seam \
                 exists in the sampling pipeline; omit logit_bias or send an empty object"
                    .to_string(),
                "logit_bias",
            ));
        }
    }
    if let Some(rf) = &req.response_format {
        if rf.format_type != "text" {
            return Err((
                format!(
                    "response_format.type \"{}\" is not yet supported on this endpoint: only \
                     \"text\" (or omitting response_format) is honored; the constrained-decoding \
                     machinery for json_object/json_schema is not wired into this handler",
                    rf.format_type
                ),
                "response_format",
            ));
        }
    }
    if !req.temperature.is_finite() || !(0.0..=2.0).contains(&req.temperature) {
        return Err((
            "temperature must be a finite number in the range [0.0, 2.0]".to_string(),
            "temperature",
        ));
    }
    if let Some(top_p) = req.top_p {
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err((
                "top_p must be a finite number in the range (0.0, 1.0]".to_string(),
                "top_p",
            ));
        }
    }
    if let Some(n) = req.n {
        if n < 1 || n > MAX_N_CHOICES {
            return Err((
                "n must be 1; multiple completions per request are not supported".to_string(),
                "n",
            ));
        }
    }
    if let Some(fp) = req.frequency_penalty {
        if !fp.is_finite() || !(-2.0..=2.0).contains(&fp) {
            return Err((
                "frequency_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
                "frequency_penalty",
            ));
        }
    }
    if let Some(pp) = req.presence_penalty {
        if !pp.is_finite() || !(-2.0..=2.0).contains(&pp) {
            return Err((
                "presence_penalty must be a finite number in the range [-2.0, 2.0]".to_string(),
                "presence_penalty",
            ));
        }
    }
    if let Some(top_logprobs) = req.top_logprobs {
        if top_logprobs > 20 {
            return Err((
                "top_logprobs must be in the range [0, 20]".to_string(),
                "top_logprobs",
            ));
        }
        if req.logprobs != Some(true) {
            return Err((
                "top_logprobs requires logprobs to be set to true".to_string(),
                "top_logprobs",
            ));
        }
    }
    // RT-07 correction: an unrecognized `role` used to fall into
    // `sanitize::chat_prompt_segments`'s `_` arm and be silently spliced
    // into the prompt as bare, unwrapped text — a protocol-correctness gap
    // (an OpenAI-shaped client sending a typo'd or future role should get a
    // clear `400`, never a request that "succeeds" against a prompt the
    // model was never designed to see) rather than the injection risk an
    // earlier draft of this finding named (content reaching that arm is
    // already covered by `neutralize_special_markers` when sanitization is
    // on). Every message's role must be one of the four this server's
    // template vocabulary understands.
    for (index, msg) in req.messages.iter().enumerate() {
        if !matches!(msg.role.as_str(), "system" | "user" | "assistant" | "tool") {
            return Err((
                format!(
                    "messages[{index}].role {:?} is not recognized: expected one of \
                     \"system\", \"user\", \"assistant\", \"tool\"",
                    msg.role
                ),
                "messages",
            ));
        }
    }
    Ok(())
}

/// Chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    /// Unix timestamp (seconds) the completion was created (`SV-03`). Both
    /// `created` and `model` are non-optional in the real OpenAI chat
    /// completion object; the streaming chunk shape
    /// (`ChatCompletionChunk`) already carried them.
    pub created: u64,
    /// The model that generated the completion (`SV-03`), resolved from the
    /// same [`ServedModelInfo`] the streaming path and `/v1/models` use —
    /// never a hard-coded literal.
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// Token usage info.
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// A choice in the completion response.
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
    /// Per-token log probabilities, present only when the request set
    /// `logprobs: true`. OpenAI-compatible `{ "content": [ ... ] }` shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<crate::api_types::ChoiceLogprobs>,
}

/// SSE streaming chunk (OpenAI-compatible).
#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    /// `None` → the member is omitted (a request that did not ask for usage).
    /// `Some(Value::Null)` → `"usage": null`, which is what OpenAI sends on
    /// every non-final chunk once `stream_options.include_usage` is set.
    /// `Some(object)` → the final usage-carrying chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<serde_json::Value>,
}

/// A choice in the SSE streaming chunk.
#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Serialize)]
struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// A `reasoning_content` delta (the model's `<think>` span); omitted on
    /// every chunk that carries none.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    /// A `tool_calls` delta: each call whole, in its own chunk, at its
    /// 0-based index among the response's calls; omitted on every chunk
    /// that carries none.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<response_pipeline::StreamToolCallDelta>>,
}

/// Everything the router assembly needs beyond the engine pool.
///
/// Introduced so new knobs (auth, per-request limits) can be added without
/// churning the positional signatures of [`create_router_with_options`] and
/// friends, which downstream code and tests already call.
pub struct RouterOptions {
    /// Optional multi-model router backing `/v1/models`.
    pub model_router: Option<Arc<ModelRouter>>,
    /// CORS / logging / rate-limit middleware.
    pub middleware: MiddlewareConfig,
    /// Per-request admission limits (`sec-05` / `sec-08`).
    pub limits: RequestLimits,
    /// Admin authentication policy (`sec-15`).
    pub auth: AuthConfig,
    /// The `max_tokens` default applied when a request omits both
    /// `max_tokens` and `max_completion_tokens` (`SV-15(c)`); see
    /// [`AppState::default_max_tokens`].
    pub default_max_tokens: usize,
    /// The hard `max_tokens` ceiling (`SV-28`); see
    /// [`AppState::max_output_tokens_ceiling`].
    pub max_output_tokens_ceiling: usize,
    /// Whether to mount the bundled chat UI at `GET /ui` (`SV-26`).
    ///
    /// Defaults to `false`: the UI is a debugging convenience whose bundled
    /// demo page sends `stream: false` (the synchronous path `sec-03`
    /// documents as unsafe under load) and, being just another route on
    /// this router, is unauthenticated by construction whenever the whole
    /// server is — it must be opted into explicitly rather than shipped on
    /// by default.
    pub enable_ui: bool,
    /// The real, model-backed embedder behind `/v1/embeddings` (`RT-08`).
    ///
    /// `Some` → the endpoint serves mean-pooled hidden states of the loaded
    /// model; `None` (the default) → it is served by
    /// [`Self::embeddings_registry`] when one is configured, and otherwise
    /// answers the honest `501` (no model-backed embedder was configured;
    /// [`Self::embedder_unavailable`] says why in the body).
    pub embedder: Option<Arc<crate::embed_engine::ModelEmbedder>>,
    /// A caller-configured [`crate::embeddings::EmbedderRegistry`] serving
    /// `/v1/embeddings` in place of the default model-only registry (e.g. a
    /// fitted TF-IDF backend). Precedence: [`Self::embedder`] wins — when
    /// both are set, the model embedder is installed into this registry
    /// (keeping its batch and input caps), where a model backend outranks
    /// TF-IDF and the identity fallback; this registry alone is served
    /// as-is, including its own `require_model_backend` setting; with
    /// neither, the router builds a registry that requires a model backend
    /// and answers `501`.
    pub embeddings_registry: Option<crate::embeddings::EmbedderRegistry>,
    /// Why this server has no model-backed embedder, carried into the `501`
    /// body of `/v1/embeddings` (`error.code` and the message suffix) —
    /// see [`Self::with_embedder_unavailable`].
    pub embedder_unavailable: Option<crate::embeddings::EmbedderUnavailable>,
    /// The engine report `/admin/*` shows for the served model
    /// ([`crate::admin::AdminState::with_engine_report`]). `None` (the
    /// default) leaves the admin router's own fallback in place.
    pub engine_report: Option<crate::admin::EngineReport>,
    /// The single token id a tokenizer-less server feeds as the prompt of a
    /// text request — see [`Self::with_prompt_start_token`]. `None` (the
    /// default) makes a tokenizer-less server answer a text prompt with
    /// `400 tokenizer_required`.
    pub prompt_start_token: Option<u32>,
}

impl Default for RouterOptions {
    fn default() -> Self {
        Self {
            model_router: None,
            middleware: MiddlewareConfig::default(),
            // Admin credentials come from the environment for the convenience
            // constructors; absent one, `/admin/*` is refused outright.
            auth: AuthConfig::from_env(),
            limits: RequestLimits::default(),
            default_max_tokens: default_max_tokens_value(),
            max_output_tokens_ceiling: MAX_OUTPUT_TOKENS,
            enable_ui: false,
            embedder: None,
            embeddings_registry: None,
            embedder_unavailable: None,
            engine_report: None,
            prompt_start_token: None,
        }
    }
}

impl RouterOptions {
    /// Attach the model-backed embedder that serves `/v1/embeddings`
    /// (`RT-08`). `None` keeps whatever [`Self::with_embeddings_registry`]
    /// configured, or the honest `501`.
    pub fn with_embedder(
        mut self,
        embedder: Option<Arc<crate::embed_engine::ModelEmbedder>>,
    ) -> Self {
        self.embedder = embedder;
        self
    }

    /// Serve `/v1/embeddings` from `registry` (e.g. a fitted TF-IDF backend)
    /// instead of the default model-only registry. [`Self::with_embedder`]
    /// still wins when both are set — see [`Self::embeddings_registry`] for
    /// the full precedence.
    pub fn with_embeddings_registry(
        mut self,
        registry: crate::embeddings::EmbedderRegistry,
    ) -> Self {
        self.embeddings_registry = Some(registry);
        self
    }

    /// Record why this server has no model-backed embedder, so the
    /// `/v1/embeddings` `501` body names it: `error.message` becomes
    /// `"<the standard text>: <message>"` and `error.code` becomes `code`
    /// (e.g. an engine refusal's `"NOT_A_DENSE_MODEL"`), or
    /// `"embeddings_unavailable"` when `code` is `None`.
    pub fn with_embedder_unavailable(
        mut self,
        code: Option<&'static str>,
        message: impl Into<String>,
    ) -> Self {
        self.embedder_unavailable =
            Some(crate::embeddings::EmbedderUnavailable::new(code, message));
        self
    }

    /// Attach the served engine's [`crate::admin::EngineReport`] to the
    /// `/admin/*` router ([`crate::admin::AdminState::with_engine_report`]).
    pub fn with_engine_report(mut self, report: crate::admin::EngineReport) -> Self {
        self.engine_report = Some(report);
        self
    }

    /// Let a tokenizer-less server answer text prompts by feeding the single
    /// token `id` as the prompt (library and test use; a server with a
    /// tokenizer ignores it). Without it a tokenizer-less server refuses a
    /// text prompt with `400 tokenizer_required`: there is no vocabulary to
    /// derive a correct start token from, and a hardcoded one is wrong for
    /// every model outside the family it was copied from.
    pub fn with_prompt_start_token(mut self, id: u32) -> Self {
        self.prompt_start_token = Some(id);
        self
    }

    /// Attach a multi-model router.
    pub fn with_model_router(mut self, model_router: Option<Arc<ModelRouter>>) -> Self {
        self.model_router = model_router;
        self
    }

    /// Attach a middleware configuration.
    pub fn with_middleware(mut self, middleware: MiddlewareConfig) -> Self {
        self.middleware = middleware;
        self
    }

    /// Attach per-request admission limits.
    pub fn with_limits(mut self, limits: RequestLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Attach an authentication policy.
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.auth = auth;
        self
    }

    /// Set the `max_tokens` default (`SV-15(c)`). `0` is coerced to `1` — a
    /// zero-token completion budget can never make progress.
    pub fn with_default_max_tokens(mut self, default_max_tokens: usize) -> Self {
        self.default_max_tokens = default_max_tokens.max(1);
        self
    }

    /// Set the hard `max_tokens` ceiling (`SV-28`). `0` is coerced to `1`
    /// for the same reason as [`Self::with_default_max_tokens`].
    pub fn with_max_output_tokens_ceiling(mut self, ceiling: usize) -> Self {
        self.max_output_tokens_ceiling = ceiling.max(1);
        self
    }

    /// Gate the bundled chat UI (`SV-26`); see [`Self::enable_ui`]'s field docs.
    pub fn with_enable_ui(mut self, enable_ui: bool) -> Self {
        self.enable_ui = enable_ui;
        self
    }
}

/// Create the Axum router.
///
/// Wraps the single `engine` in a 1-element [`EnginePool`], preserving
/// byte-identical single-request behavior. Use
/// [`create_router_with_pool`] to serve from a multi-replica pool.
///
/// `/admin/*` is authenticated: the credential is taken from `OXI_ADMIN_TOKEN`,
/// and when that is unset every admin request is refused with `403`
/// (finding `sec-15`). Use [`create_router_with_auth`] to pass the token
/// explicitly.
pub fn create_router(
    engine: InferenceEngine<'static>,
    tokenizer: Option<TokenizerBridge>,
) -> Router {
    create_router_with_metrics(engine, tokenizer, Arc::new(InferenceMetrics::new()))
}

/// Create the Axum router with an explicit authentication policy.
///
/// This is the constructor the serve binaries use: it is the only way to make
/// `/admin/*` reachable without going through the `OXI_ADMIN_TOKEN`
/// environment variable.
pub fn create_router_with_auth(
    engine: InferenceEngine<'static>,
    tokenizer: Option<TokenizerBridge>,
    auth: AuthConfig,
) -> Router {
    create_router_full(
        EnginePool::new(vec![engine]),
        tokenizer,
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default().with_auth(auth),
    )
}

/// Create the Axum router with a shared metrics instance.
///
/// Wraps the single `engine` in a 1-element [`EnginePool`] and delegates to
/// [`create_router_with_pool`].
pub fn create_router_with_metrics(
    engine: InferenceEngine<'static>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
) -> Router {
    create_router_with_pool(EnginePool::new(vec![engine]), tokenizer, metrics)
}

/// Create the Axum router from a pre-built [`EnginePool`].
///
/// This is the shared core behind [`create_router`] and
/// [`create_router_with_metrics`]; it lets server entry points serve from a
/// multi-replica pool so independent requests generate concurrently instead of
/// serializing on a single engine mutex.
pub fn create_router_with_pool(
    engines: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
) -> Router {
    create_router_with_options(
        engines,
        tokenizer,
        metrics,
        None,
        MiddlewareConfig::default(),
    )
}

/// Create the fully-featured Axum router from a pre-built [`EnginePool`].
///
/// This is the single place that assembles the served router. In addition to
/// the OpenAI-compatible inference routes it also mounts:
///
/// - the embeddings sub-router,
/// - the chat web UI (`GET /ui`),
/// - the operator admin API (`/admin/*`), wired to the *real* running config
///   and cache sources, and
/// - the request middleware described by `middleware_config` (CORS, request
///   logging, and — when configured — the token-bucket rate limiter).
///
/// When `model_router` is `Some`, `/v1/models` reports every endpoint the
/// router knows about; otherwise it reports the single model actually loaded
/// into the pool (resolved from the engine, not a hard-coded literal).
pub fn create_router_with_options(
    engines: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
    model_router: Option<Arc<ModelRouter>>,
    middleware_config: MiddlewareConfig,
) -> Router {
    create_router_full(
        engines,
        tokenizer,
        metrics,
        RouterOptions::default()
            .with_model_router(model_router)
            .with_middleware(middleware_config),
    )
}

/// Create the fully-featured Axum router with every option set explicitly.
///
/// This is the single assembly point; every other `create_router*` constructor
/// delegates here. It is also where `/admin/*` is wrapped in the admin
/// authentication layer — unconditionally, with no code path that mounts the
/// admin surface unauthenticated (finding `sec-15`).
pub fn create_router_full(
    engines: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
    options: RouterOptions,
) -> Router {
    let RouterOptions {
        model_router,
        middleware: middleware_config,
        limits,
        auth,
        default_max_tokens,
        max_output_tokens_ceiling,
        enable_ui,
        embedder,
        embeddings_registry,
        embedder_unavailable,
        engine_report,
        prompt_start_token,
    } = options;

    let model_info = Arc::new(ServedModelInfo::new(Arc::clone(&engines)));
    let special_tokens = match &tokenizer {
        Some(tok) => SpecialTokenGuard::from_tokenizer(tok),
        None => SpecialTokenGuard::default(),
    };
    let auth = Arc::new(auth);
    if !auth.admin_enabled() {
        tracing::warn!(
            "no admin token configured: the /admin API is refused on this server; \
             set {} or build the router with create_router_with_auth",
            auth::ADMIN_TOKEN_ENV
        );
    }
    // SV-19: shared with the admin router below so `/admin/workload-stats`
    // and `/admin/cache-stats` report real, request-derived data instead of
    // a permanent null.
    let rate_aggregator = Arc::new(crate::request_metrics::RequestRateAggregator::new());
    let kv_cache_policy = Arc::new(crate::kv_cache_policy::KvCachePolicy::default());
    let state = Arc::new(AppState {
        engines,
        tokenizer,
        metrics: Arc::clone(&metrics),
        model_info: Arc::clone(&model_info),
        model_router,
        sanitize_prompt: resolve_prompt_sanitization(),
        special_tokens,
        limits,
        auth: Arc::clone(&auth),
        default_max_tokens: default_max_tokens.max(1),
        max_output_tokens_ceiling: max_output_tokens_ceiling.max(1),
        rate_aggregator: Arc::clone(&rate_aggregator),
        kv_cache_policy: Arc::clone(&kv_cache_policy),
        prompt_start_token,
    });

    // The embeddings router carries its own Arc<EmbeddingAppState>; merge it
    // before attaching the main AppState so the states don't conflict.
    //
    // `RT-08`: serve REAL model-backed embeddings when this server was given
    // an embedder; otherwise serve the caller's configured registry (e.g. a
    // fitted TF-IDF backend) as-is; with neither, a registry that requires a
    // model backend answers the honest `501`, whose body carries the
    // recorded `embedder_unavailable` reason. Every branch records onto the
    // SAME `InferenceMetrics` every other route uses (`SV-25`).
    let embeddings_router = {
        let registry =
            build_embeddings_registry(embedder, embeddings_registry, embedder_unavailable);
        crate::embeddings::create_embeddings_router_from_state(
            crate::embeddings::EmbeddingAppState::from_registry(registry)
                .with_metrics(Arc::clone(&metrics)),
        )
    };

    // Admin API, wired to the real metrics + model descriptor so `/admin/config`
    // reports the running configuration instead of hard-coded defaults.
    // SV-19: `with_rate_aggregator`/`with_kv_cache_policy` attach the same
    // instances the chat handlers feed, so `/admin/workload-stats` and
    // `/admin/cache-stats` stop being permanently null. The served engine's
    // report, when the caller has one, is attached to the same state.
    let mut admin_state = crate::admin::AdminState::new(Arc::clone(&metrics))
        .with_model_info(Arc::clone(&model_info))
        .with_rate_aggregator(rate_aggregator)
        .with_kv_cache_policy(kv_cache_policy);
    if let Some(report) = engine_report {
        admin_state = admin_state.with_engine_report(report);
    }
    let admin_state = Arc::new(admin_state);
    // sec-15: the admin surface is authenticated unconditionally. The layer is
    // attached to the admin sub-router only, so the inference routes are
    // unaffected and a serve binary can still put its own bearer auth in front
    // of everything.
    //
    // This MUST be `route_layer`, not `layer`. `Router::layer` also wraps the
    // router's fallback, and `Router::merge` propagates that wrapped fallback
    // to the whole merged app — so a plain `.layer(...)` here would put every
    // unmatched path on the ENTIRE server (not just under `/admin`) behind the
    // admin-auth check, turning a request to a typo'd or unknown route into a
    // `403` that discloses the server's admin-credential configuration hints
    // instead of an ordinary `404`. `route_layer` only wraps the routes
    // actually registered on this sub-router, so an unmatched path anywhere
    // (including under `/admin/*`) still falls through to the app's own 404.
    let admin_router: Router = crate::admin::create_admin_router(Arc::clone(&admin_state))
        .with_state(admin_state)
        .route_layer(axum::middleware::from_fn_with_state(
            Arc::clone(&auth),
            auth::admin_auth_mw,
        ));

    let mut app = Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat::chat_completions),
        )
        .route(
            "/v1/chat/completions/extended",
            axum::routing::post(crate::api_extensions::extended_chat_completions),
        )
        .route(
            "/v1/completions",
            axum::routing::post(crate::completions::create_completion),
        )
        .route("/v1/models", axum::routing::get(list_models))
        .route("/v1/models/{model}", axum::routing::get(get_model))
        .route("/health", axum::routing::get(health))
        .route("/readyz", axum::routing::get(readyz))
        .route("/metrics", axum::routing::get(prometheus_metrics))
        .with_state(state)
        .merge(embeddings_router);

    // SV-26: the bundled chat UI is opt-in (`enable_ui`, default `false`).
    // When mounted it is merged in exactly like every other route here, so
    // it inherits whatever bearer-auth middleware a serve binary wraps the
    // whole router in — "the same auth as everything else" — rather than
    // being carved out as an exception.
    if enable_ui {
        app = app.merge(crate::web_ui::create_ui_router());
    }

    let app = app.merge(admin_router);

    crate::middleware::apply_middleware(app, middleware_config)
}

/// Default embedding width of the registry a router builds for itself (the
/// `IdentityEmbedder` dimension; never served, since that registry requires
/// a model backend).
const DEFAULT_ROUTER_EMBEDDING_DIM: usize = 512;

/// The registry `/v1/embeddings` serves, by [`RouterOptions::embeddings_registry`]'s
/// precedence: a model `embedder` always wins (installed into the caller's
/// registry when there is one, else into a registry that requires a model
/// backend); a caller's registry alone is served as configured; with
/// neither, a model-only registry answers `501`. The `unavailable` reason
/// is attached to whichever registry is served, so a `501` from any of them
/// names it.
fn build_embeddings_registry(
    embedder: Option<Arc<crate::embed_engine::ModelEmbedder>>,
    configured: Option<crate::embeddings::EmbedderRegistry>,
    unavailable: Option<crate::embeddings::EmbedderUnavailable>,
) -> crate::embeddings::EmbedderRegistry {
    let registry = match (embedder, configured) {
        (Some(embedder), Some(configured)) => configured.with_model(embedder),
        (Some(embedder), None) => {
            crate::embeddings::EmbedderRegistry::new(DEFAULT_ROUTER_EMBEDDING_DIM)
                .with_require_model_backend(true)
                .with_model(embedder)
        }
        (None, Some(configured)) => configured,
        (None, None) => crate::embeddings::EmbedderRegistry::new(DEFAULT_ROUTER_EMBEDDING_DIM)
            .with_require_model_backend(true),
    };
    match unavailable {
        Some(reason) => registry.with_unavailable_reason(reason),
        None => registry,
    }
}

async fn health() -> &'static str {
    "ok"
}

/// Prometheus metrics endpoint.
async fn prometheus_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let body = state.metrics.render_prometheus();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

/// `GET /v1/models` — report the actually-loaded model(s).
///
/// When a multi-model [`ModelRouter`] is attached, every registered endpoint is
/// listed (OpenAI-compatible shape). Otherwise the single loaded model is
/// reported, with its `id` read from the engine's real configuration and a real
/// `created` timestamp — never a hard-coded literal.
async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    if let Some(router) = state.model_router() {
        let data: Vec<serde_json::Value> = router
            .models_list()
            .into_iter()
            .map(|entry| {
                serde_json::json!({
                    "id": entry.id,
                    "object": entry.object,
                    "owned_by": entry.owned_by,
                    "created": entry.created,
                })
            })
            .collect();
        return Json(serde_json::json!({ "object": "list", "data": data }));
    }

    let descriptor = state.model_info().descriptor().await;
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": descriptor.id,
            "object": "model",
            "owned_by": "oxibonsai",
            "created": descriptor.created,
            "max_context_length": descriptor.max_context_length,
        }]
    }))
}

/// `GET /v1/models/{model}` (`SV-21`): the per-model companion to
/// `/v1/models`, resolved from the same sources as it — the attached
/// [`ModelRouter`] when present, otherwise the single loaded model's real
/// descriptor. 404s with the canonical OpenAI error envelope for an unknown
/// id, instead of axum's default bare-status, empty-body 404.
async fn get_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let not_found = || {
        ApiError::new(
            StatusCode::NOT_FOUND,
            format!("model \"{model_id}\" not found"),
        )
        .with_code("model_not_found")
    };

    if let Some(router) = state.model_router() {
        return router
            .models_list()
            .into_iter()
            .find(|m| m.id == model_id)
            .map(|entry| {
                Json(serde_json::json!({
                    "id": entry.id,
                    "object": entry.object,
                    "owned_by": entry.owned_by,
                    "created": entry.created,
                }))
            })
            .ok_or_else(not_found);
    }

    let descriptor = state.model_info().descriptor().await;
    if descriptor.id == model_id {
        Ok(Json(serde_json::json!({
            "id": descriptor.id,
            "object": "model",
            "owned_by": "oxibonsai",
            "created": descriptor.created,
            "max_context_length": descriptor.max_context_length,
        })))
    } else {
        Err(not_found())
    }
}

/// `GET /readyz` (`SV-14`) — readiness probe, distinct from `/health`'s pure
/// liveness check: reports whether this server can actually serve a
/// request *right now* (a model is loaded **and** at least one engine-pool
/// replica is currently idle), not merely that the process is up.
/// `docs/DEPLOYMENT.md` already tells operators to point a Kubernetes
/// readiness probe here; this is what makes that claim true rather than
/// silently falling back to `/health`, which cannot express "overloaded."
async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    let descriptor = state.model_info().descriptor().await;
    let model_loaded = descriptor.id != "unknown";
    let engine_slot_available = state.engines().idle_count() > 0;
    let ready = model_loaded && engine_slot_available;
    let body = serde_json::json!({
        "status": if ready { "ready" } else { "not_ready" },
        "model_loaded": model_loaded,
        "engine_slot_available": engine_slot_available,
    });
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body)).into_response()
}

/// The `/v1/chat/completions` request pipeline (validation, sampling-params
/// resolution, non-streaming and SSE-streaming generation) — split into its
/// own file purely to keep both under the workspace's 2000-line ceiling; see
/// that module's doc comment.
mod chat;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod sampling_tests;

#[cfg(test)]
mod seam_tests;
