//! Regression tests for the "server-api" OpenAI-compat honesty audit (findings
//! 28-31 and 55): `/v1/chat/completions/extended` and `/v1/completions` must
//! reject unsupported/out-of-range client input with `400 Bad Request`
//! (`max_tokens` ceiling, `n` cap, non-zero `frequency_penalty` /
//! `presence_penalty`) instead of silently ignoring or clamping it.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::api_extensions::MAX_EXTENDED_N_CHOICES;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router, MAX_OUTPUT_TOKENS};

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

// ── /v1/chat/completions/extended ─────────────────────────────────────────────

/// Finding 55: `n` beyond the supported cap must be rejected, not silently
/// clamped down to the cap.
#[tokio::test]
async fn extended_n_over_cap_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "n": (MAX_EXTENDED_N_CHOICES + 1),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "n");
}

/// `n` at exactly the cap must still succeed (only *over*-cap is rejected).
#[tokio::test]
async fn extended_n_at_cap_is_accepted() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "n": MAX_EXTENDED_N_CHOICES,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let choices = json["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), MAX_EXTENDED_N_CHOICES);
}

/// Wave-2: an in-range non-zero `frequency_penalty` is now applied for real
/// (previously rejected with `400` when no sampler seam existed), so it must be
/// accepted.
#[tokio::test]
async fn extended_frequency_penalty_nonzero_is_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "frequency_penalty": 0.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Wave-2: an in-range non-zero `presence_penalty` is now applied for real and
/// must be accepted.
#[tokio::test]
async fn extended_presence_penalty_nonzero_is_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "presence_penalty": 0.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// A penalty outside the OpenAI `[-2.0, 2.0]` range is still rejected honestly
/// rather than clamped.
#[tokio::test]
async fn extended_frequency_penalty_out_of_range_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "frequency_penalty": 5.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "frequency_penalty");
}

/// A zero (or omitted) penalty is a legitimate no-op request and must not be
/// rejected.
#[tokio::test]
async fn extended_zero_penalties_are_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// `max_tokens` above the shared server-wide ceiling must be rejected on the
/// extended endpoint exactly as it is on the base `/v1/chat/completions`
/// endpoint.
#[tokio::test]
async fn extended_max_tokens_over_cap_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": MAX_OUTPUT_TOKENS + 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "max_tokens");
}

#[tokio::test]
async fn extended_max_tokens_zero_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/chat/completions/extended",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "max_tokens");
}

// ── /v1/completions ────────────────────────────────────────────────────────────

/// `max_tokens` above the shared server-wide ceiling must be rejected on the
/// legacy completions endpoint too (mirrors the extended-endpoint and base
/// chat-endpoint cap).
#[tokio::test]
async fn completions_max_tokens_over_cap_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": MAX_OUTPUT_TOKENS + 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "max_tokens");
}

#[tokio::test]
async fn completions_max_tokens_zero_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "max_tokens");
}

/// `n != 1` is not supported by the legacy completions endpoint (only a
/// single completion is ever generated) and must be rejected rather than
/// silently ignored.
#[tokio::test]
async fn completions_n_other_than_one_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 2,
            "n": 3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "n");
}

#[tokio::test]
async fn completions_n_one_is_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 2,
            "n": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Wave-2 (completions.rs mirror): an in-range non-zero `frequency_penalty` is
/// now applied for real and must be accepted, not rejected.
#[tokio::test]
async fn completions_frequency_penalty_nonzero_is_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 2,
            "frequency_penalty": 0.3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Wave-2 (completions.rs mirror): an in-range non-zero `presence_penalty` is
/// now applied for real and must be accepted.
#[tokio::test]
async fn completions_presence_penalty_nonzero_is_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 2,
            "presence_penalty": 0.3,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// A `presence_penalty` outside the OpenAI `[-2.0, 2.0]` range is still
/// rejected honestly on the completions endpoint.
#[tokio::test]
async fn completions_presence_penalty_out_of_range_is_rejected_with_400() {
    let (status, json) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 2,
            "presence_penalty": -3.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["param"], "presence_penalty");
}

#[tokio::test]
async fn completions_zero_penalties_are_accepted() {
    let (status, _) = post(
        test_router(),
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "max_tokens": 2,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}
