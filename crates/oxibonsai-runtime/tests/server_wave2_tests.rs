//! Wave-2 regression tests for the OpenAI-compatible surface integration:
//!
//! * `deferred-features-03` — the base `/v1/chat/completions` endpoint accepts
//!   (and no longer silently ignores) `frequency_penalty` / `presence_penalty`
//!   / `logprobs`, and rejects out-of-range values honestly.
//! * `serve-api-01` — the base endpoint reports `finish_reason` honestly
//!   ("length" on max_tokens truncation, "stop" otherwise).
//! * `serve-api-02` — covered by unit tests on `stream_terminal_json`.
//! * `serve-api-03` — the base endpoint no longer silently drops `tools`:
//!   streaming + tools is a `400`, non-streaming + tools is processed.
//! * `serve-api-10` — every route emits the same OpenAI-style nested error
//!   envelope (asserted across four routes on one router).
//!
//! Uses a tiny synthetic engine and no tokenizer, so these exercise the HTTP
//! handler wiring rather than model quality (the engine-level penalty/logprobs
//! behavior is covered by `penalty_seam_tests.rs`).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::create_router;

fn test_router() -> axum::Router {
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    create_router(engine, None)
}

async fn post(
    app: axum::Router,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
        .expect("build request");
    let resp = app.oneshot(req).await.expect("send request");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

// ── deferred-features-03: penalties accepted on the base endpoint ─────────────

#[tokio::test]
async fn base_penalties_in_range_are_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn base_frequency_penalty_out_of_range_is_rejected() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "frequency_penalty": 9.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "frequency_penalty");
}

#[tokio::test]
async fn base_presence_penalty_out_of_range_is_rejected() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "presence_penalty": -9.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "presence_penalty");
}

// ── deferred-features-03 / logprobs shape (OpenAI) ────────────────────────────

#[tokio::test]
async fn base_logprobs_response_has_openai_shape() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "logprobs": true,
            "top_logprobs": 3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let choice = &json["choices"][0];
    // OpenAI chat logprobs shape: choice.logprobs.content is an array; each
    // entry carries a token, a numeric logprob, and its top_logprobs.
    let content = choice["logprobs"]["content"]
        .as_array()
        .expect("logprobs.content must be an array");
    for entry in content {
        assert!(entry["token"].is_string(), "each entry has a token string");
        assert!(
            entry["logprob"].is_number(),
            "each entry has a numeric logprob"
        );
        assert!(
            entry["top_logprobs"].is_array(),
            "each entry has a top_logprobs array"
        );
    }
}

#[tokio::test]
async fn base_top_logprobs_requires_logprobs_true() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "top_logprobs": 5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "top_logprobs");
}

#[tokio::test]
async fn base_top_logprobs_over_20_is_rejected() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "logprobs": true,
            "top_logprobs": 21,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "top_logprobs");
}

// ── serve-api-01: honest finish_reason on the base endpoint ────────────────────

#[tokio::test]
async fn base_finish_reason_matches_token_count() {
    let max_tokens = 1u64;
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": max_tokens,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let completion_tokens = json["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens");
    let finish = json["choices"][0]["finish_reason"]
        .as_str()
        .expect("finish_reason string");
    // The mapping must be honest regardless of what the tiny model produced: a
    // run truncated at max_tokens reports "length", a natural EOS reports "stop".
    if completion_tokens >= max_tokens {
        assert_eq!(finish, "length", "truncated run must report length");
    } else {
        assert_eq!(finish, "stop", "natural stop must report stop");
    }
}

// ── serve-api-03: tools no longer silently ignored on the base endpoint ────────

#[tokio::test]
async fn base_stream_plus_tools_is_rejected() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": true,
            "tools": [{
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}}
            }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "stream");
}

#[tokio::test]
async fn base_stream_plus_logprobs_is_rejected() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": true,
            "logprobs": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "logprobs");
}

#[tokio::test]
async fn base_non_stream_tools_are_processed() {
    // With tools supplied (non-streaming) the request is honored rather than
    // rejected; the response is a well-formed chat completion. (Whether an
    // actual tool_call is produced depends on the generated text, covered by
    // the `parse_base_tool_calls` unit tests.)
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}}
            }],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
    assert!(json["choices"].is_array());
}

// ── serve-api-10: one error envelope across every route ────────────────────────

/// A `400`/`4xx` from four different routes mounted on the same router must all
/// carry the identical OpenAI-style nested envelope (`error.message` string,
/// `error.type` string, `error` object), so a client parsing one shape parses
/// them all.
#[tokio::test]
async fn error_envelope_is_uniform_across_routes() {
    let cases: [(&str, serde_json::Value); 4] = [
        (
            "/v1/chat/completions",
            serde_json::json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}),
        ),
        (
            "/v1/completions",
            serde_json::json!({"prompt": "x", "max_tokens": 0}),
        ),
        (
            "/v1/chat/completions/extended",
            serde_json::json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}),
        ),
        // Empty batch input → 422 with the same envelope (previously a body-less
        // status code).
        ("/v1/embeddings", serde_json::json!({"input": []})),
    ];

    for (path, body) in cases {
        let (status, json) = post(test_router(), path, body).await;
        assert!(
            status.is_client_error(),
            "{path} should be a client error, got {status}"
        );
        assert!(
            json["error"].is_object(),
            "{path} error must be a nested object; got {json}"
        );
        assert!(
            json["error"]["message"].is_string(),
            "{path} must expose error.message; got {json}"
        );
        assert!(
            json["error"]["type"].is_string(),
            "{path} must expose error.type; got {json}"
        );
    }
}
