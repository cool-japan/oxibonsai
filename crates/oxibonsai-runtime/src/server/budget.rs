//! Per-request admission limits: prompt size, context window, deadlines.
//!
//! Finding `sec-05` (with `RT-13` / `SV-16`): the HTTP layer validated
//! `max_tokens`, `temperature`, `top_p` and friends but **nothing at all about
//! the prompt**. An over-long prompt therefore reached the engine and came back
//! as an opaque `500`, and `limits.max_input_tokens` — a field
//! `oxibonsai-serve` validates at startup — was never consulted per request.
//!
//! The two guards here run in the order that matters:
//!
//! 1. [`validate_prompt_bytes`] — a *byte* guard on the raw message content,
//!    checked **before** sanitizing or tokenizing. It is what bounds the cost
//!    of both (and so also blunts `sec-01`), because both are linear in the
//!    input and the input is otherwise limited only by the body limit.
//! 2. [`validate_request_budget`] — the *token* guard, checked after encoding:
//!    `max_input_tokens` and `prompt_tokens + max_tokens <= context_length`.
//!
//! Both return an [`ApiError`] whose JSON body names the offending numbers, so
//! a client can tell how far over it went instead of guessing.

use std::time::Duration;

use crate::server::api_error::ApiError;

/// Default ceiling on the summed byte length of a request's message content.
///
/// 1 MiB is comfortably above any real chat prompt (a 262 K-token context at
/// ~3.5 bytes/token is ~900 KB) while keeping the pre-tokenizer work bounded.
pub const DEFAULT_MAX_PROMPT_BYTES: usize = 1024 * 1024;

/// Default SSE keep-alive interval (finding `sec-08`).
pub const DEFAULT_SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

/// Per-request limits applied by the chat handlers.
///
/// The defaults preserve the historical behaviour for everything that used to
/// be unbounded *except* the prompt byte ceiling, which is a new backstop:
/// no deadline unless one is configured (the serve binaries pass
/// `limits.per_request_timeout_ms`), and SSE keep-alive on.
#[derive(Debug, Clone)]
pub struct RequestLimits {
    /// Maximum summed byte length of the request's message content.
    pub max_prompt_bytes: usize,
    /// Maximum prompt length in tokens, independent of the context window.
    pub max_input_tokens: Option<usize>,
    /// Wall-clock deadline for a whole request, including the SSE body.
    pub per_request_timeout: Option<Duration>,
    /// Interval between SSE keep-alive comments; `None` disables them.
    pub sse_keep_alive: Option<Duration>,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            max_input_tokens: None,
            per_request_timeout: None,
            sse_keep_alive: Some(DEFAULT_SSE_KEEP_ALIVE),
        }
    }
}

impl RequestLimits {
    /// Set the raw prompt byte ceiling.
    pub fn with_max_prompt_bytes(mut self, bytes: usize) -> Self {
        self.max_prompt_bytes = bytes.max(1);
        self
    }

    /// Set the prompt token ceiling (`limits.max_input_tokens`).
    pub fn with_max_input_tokens(mut self, tokens: Option<usize>) -> Self {
        self.max_input_tokens = tokens.filter(|t| *t > 0);
        self
    }

    /// Set the whole-request deadline (`limits.per_request_timeout_ms`).
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.per_request_timeout = timeout.filter(|d| !d.is_zero());
        self
    }

    /// Set the whole-request deadline from milliseconds; `0` disables it.
    pub fn with_timeout_ms(self, timeout_ms: u64) -> Self {
        self.with_timeout(Some(Duration::from_millis(timeout_ms)))
    }

    /// Set the SSE keep-alive interval; `None` disables keep-alives.
    pub fn with_sse_keep_alive(mut self, interval: Option<Duration>) -> Self {
        self.sse_keep_alive = interval.filter(|d| !d.is_zero());
        self
    }
}

/// Reject a request whose raw message content exceeds `max_prompt_bytes`.
///
/// Checked **before** sanitization and tokenization, so neither runs on an
/// oversized input. Returns `413 Payload Too Large` naming both numbers.
pub fn validate_prompt_bytes(n_bytes: usize, max_prompt_bytes: usize) -> Result<(), ApiError> {
    if n_bytes > max_prompt_bytes {
        return Err(ApiError::new(
            axum::http::StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "prompt is {n_bytes} bytes, which exceeds the server's limit of \
                 {max_prompt_bytes} bytes"
            ),
        )
        .with_param("messages")
        .with_code("prompt_too_large")
        .with_field("prompt_bytes", n_bytes)
        .with_field("max_prompt_bytes", max_prompt_bytes));
    }
    Ok(())
}

/// Validate a request's token budget against the model's context window.
///
/// * `n_prompt_tokens` — encoded prompt length.
/// * `max_tokens` — requested completion length.
/// * `ctx_len` — the model's context length; `0` means "unknown", in which case
///   only `max_input_tokens` is enforced.
/// * `max_input_tokens` — the server's configured prompt ceiling, if any.
///
/// Returns `400 Bad Request` with `code = "context_length_exceeded"` (or
/// `"max_input_tokens_exceeded"`) and a body naming `n_prompt_tokens`,
/// `max_tokens`, `max_input_tokens` and `context_length`.
///
/// Shared by `server.rs`, the completions/extended endpoints and
/// `oxibonsai-serve`, so every entry point reports the same numbers.
pub fn validate_request_budget(
    n_prompt_tokens: usize,
    max_tokens: usize,
    ctx_len: usize,
    max_input_tokens: Option<usize>,
) -> Result<(), ApiError> {
    let annotate = |err: ApiError| {
        err.with_param("messages")
            .with_field("n_prompt_tokens", n_prompt_tokens)
            .with_field("max_tokens", max_tokens)
            .with_field(
                "max_input_tokens",
                match max_input_tokens {
                    Some(m) => serde_json::Value::from(m),
                    None => serde_json::Value::Null,
                },
            )
            .with_field(
                "context_length",
                match ctx_len {
                    0 => serde_json::Value::Null,
                    n => serde_json::Value::from(n),
                },
            )
    };

    if let Some(limit) = max_input_tokens {
        if n_prompt_tokens > limit {
            return Err(annotate(
                ApiError::new(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!(
                        "prompt is {n_prompt_tokens} tokens, which exceeds the server's \
                         max_input_tokens limit of {limit}"
                    ),
                )
                .with_code("max_input_tokens_exceeded"),
            ));
        }
    }

    if ctx_len == 0 {
        return Ok(());
    }

    if n_prompt_tokens >= ctx_len {
        return Err(annotate(
            ApiError::new(
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "prompt is {n_prompt_tokens} tokens, which does not fit in the model's \
                     context length of {ctx_len} tokens"
                ),
            )
            .with_code("context_length_exceeded"),
        ));
    }

    let total = n_prompt_tokens.saturating_add(max_tokens);
    if total > ctx_len {
        return Err(annotate(
            ApiError::new(
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "prompt ({n_prompt_tokens} tokens) plus max_tokens ({max_tokens}) is \
                     {total} tokens, which exceeds the model's context length of {ctx_len} \
                     tokens; reduce max_tokens to at most {} or shorten the prompt",
                    ctx_len - n_prompt_tokens
                ),
            )
            .with_code("context_length_exceeded"),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn fitting_request_is_accepted() {
        assert!(validate_request_budget(100, 100, 512, None).is_ok());
        assert!(validate_request_budget(100, 412, 512, None).is_ok());
    }

    #[test]
    fn prompt_plus_max_tokens_over_context_is_rejected() {
        let err = validate_request_budget(100, 413, 512, None).expect_err("must reject");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        let json = err.to_json();
        assert_eq!(json["error"]["code"], "context_length_exceeded");
        assert_eq!(json["error"]["n_prompt_tokens"], 100);
        assert_eq!(json["error"]["max_tokens"], 413);
        assert_eq!(json["error"]["context_length"], 512);
        assert!(json["error"]["max_input_tokens"].is_null());
        assert!(
            err.message().contains("512"),
            "message must name the context length: {}",
            err.message()
        );
    }

    #[test]
    fn prompt_alone_over_context_is_rejected() {
        let err = validate_request_budget(600, 1, 512, None).expect_err("must reject");
        assert_eq!(err.to_json()["error"]["code"], "context_length_exceeded");
    }

    #[test]
    fn max_input_tokens_is_enforced_independently() {
        // Fits in the context window, but over the configured input ceiling.
        let err = validate_request_budget(300, 10, 4096, Some(256)).expect_err("must reject");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        let json = err.to_json();
        assert_eq!(json["error"]["code"], "max_input_tokens_exceeded");
        assert_eq!(json["error"]["max_input_tokens"], 256);
    }

    #[test]
    fn unknown_context_length_only_checks_the_input_ceiling() {
        assert!(validate_request_budget(10_000, 10_000, 0, None).is_ok());
        assert!(validate_request_budget(10_000, 1, 0, Some(128)).is_err());
    }

    #[test]
    fn max_tokens_overflow_saturates_instead_of_wrapping() {
        let err =
            validate_request_budget(1, usize::MAX, 512, None).expect_err("must reject, not wrap");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn prompt_bytes_guard() {
        assert!(validate_prompt_bytes(1024, 4096).is_ok());
        let err = validate_prompt_bytes(4097, 4096).expect_err("must reject");
        assert_eq!(err.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let json = err.to_json();
        assert_eq!(json["error"]["code"], "prompt_too_large");
        assert_eq!(json["error"]["prompt_bytes"], 4097);
        assert_eq!(json["error"]["max_prompt_bytes"], 4096);
    }

    #[test]
    fn limits_builders_reject_degenerate_values() {
        let limits = RequestLimits::default()
            .with_max_prompt_bytes(0)
            .with_max_input_tokens(Some(0))
            .with_timeout(Some(Duration::ZERO))
            .with_sse_keep_alive(Some(Duration::ZERO));
        assert_eq!(limits.max_prompt_bytes, 1);
        assert_eq!(limits.max_input_tokens, None);
        assert_eq!(limits.per_request_timeout, None);
        assert_eq!(limits.sse_keep_alive, None);

        let limits = RequestLimits::default().with_timeout_ms(1_500);
        assert_eq!(
            limits.per_request_timeout,
            Some(Duration::from_millis(1500))
        );
    }

    #[test]
    fn defaults_are_the_documented_ones() {
        let limits = RequestLimits::default();
        assert_eq!(limits.max_prompt_bytes, DEFAULT_MAX_PROMPT_BYTES);
        assert_eq!(limits.max_input_tokens, None);
        assert_eq!(limits.per_request_timeout, None);
        assert_eq!(limits.sse_keep_alive, Some(DEFAULT_SSE_KEEP_ALIVE));
    }
}
