//! Integration tests for the `/v1/completions` endpoint.
//!
//! These tests stand up a real Axum router (with a tiny test engine) and fire
//! HTTP requests against it, verifying the shape and content of the returned
//! JSON against the OpenAI completions schema.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::completions::MAX_COMPLETION_BATCH_SIZE;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::create_router;

// ── Helper ────────────────────────────────────────────────────────────────────

/// Build a minimal router backed by the tiny test engine.
fn test_router() -> axum::Router {
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    create_router(engine, None)
}

/// Send a POST request with a JSON body and return the parsed response.
async fn post_completions(app: axum::Router, body: serde_json::Value) -> serde_json::Value {
    let req = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("body serialisation"),
        ))
        .expect("request build");

    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "expected 200 OK from /v1/completions"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).expect("JSON parse")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// A basic valid request must return 200 with a parseable JSON body.
#[tokio::test]
async fn test_completions_valid_request() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": "Once upon a time",
        "max_tokens": 8
    });
    let json = post_completions(app, body).await;
    // The response should at minimum have `choices` and `usage`.
    assert!(json.get("choices").is_some(), "missing 'choices' field");
    assert!(json.get("usage").is_some(), "missing 'usage' field");
}

/// Every response must carry a non-empty `id` field prefixed with `"cmpl-"`.
#[tokio::test]
async fn test_completions_response_has_id() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 4
    });
    let json = post_completions(app, body).await;
    let id = json["id"].as_str().expect("id must be a string");
    assert!(!id.is_empty(), "id must not be empty");
    assert!(
        id.starts_with("cmpl-"),
        "expected id to start with 'cmpl-', got: {id}"
    );
}

/// The `object` field must be exactly `"text_completion"`.
#[tokio::test]
async fn test_completions_response_object_is_text_completion() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": "Say something",
        "max_tokens": 4
    });
    let json = post_completions(app, body).await;
    assert_eq!(
        json["object"].as_str().expect("object field"),
        "text_completion"
    );
}

/// When `echo: true`, the first choice's `text` must start with the prompt.
#[tokio::test]
async fn test_completions_echo_includes_prompt() {
    let app = test_router();
    let prompt = "Repeat after me:";
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 4,
        "echo": true
    });
    let json = post_completions(app, body).await;
    let text = json["choices"][0]["text"]
        .as_str()
        .expect("choices[0].text must be a string");
    assert!(
        text.starts_with(prompt),
        "echoed text must start with the prompt.\ntext: {text:?}\nprompt: {prompt:?}"
    );
}

/// The `usage` object must have `prompt_tokens`, `completion_tokens`, and
/// `total_tokens`, and the latter must equal the sum of the former two.
#[tokio::test]
async fn test_completions_usage_fields() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": "Count tokens please",
        "max_tokens": 6
    });
    let json = post_completions(app, body).await;
    let usage = &json["usage"];
    let prompt = usage["prompt_tokens"].as_u64().expect("prompt_tokens");
    let completion = usage["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    let total = usage["total_tokens"].as_u64().expect("total_tokens");
    assert_eq!(
        total,
        prompt + completion,
        "total_tokens must equal prompt_tokens + completion_tokens"
    );
    assert!(prompt > 0, "prompt_tokens must be > 0");
}

/// Regression test for finding serve-api-09: every prompt in a batch request
/// must be generated and returned as its own choice, not just the first one.
#[tokio::test]
async fn test_completions_batch_prompt_all_returned() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": ["First prompt", "Second prompt", "Third prompt"],
        "max_tokens": 4
    });
    let json = post_completions(app, body).await;
    let choices = json["choices"].as_array().expect("choices array");
    assert_eq!(
        choices.len(),
        3,
        "batch of 3 prompts must produce 3 choices, got: {json}"
    );
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], i, "choice index must match its position");
        assert!(
            choice["text"].is_string(),
            "choice {i} must have generated text"
        );
        assert!(
            choice["finish_reason"].is_string(),
            "choice {i} must have a finish_reason"
        );
    }
}

/// Batch usage must be the sum across every prompt in the batch, not just
/// the first one.
#[tokio::test]
async fn test_completions_batch_prompt_usage_is_aggregated() {
    let app = test_router();
    let single_body = serde_json::json!({
        "prompt": "First prompt",
        "max_tokens": 4
    });
    let single_json = post_completions(app.clone(), single_body).await;
    let single_prompt_tokens = single_json["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt_tokens");

    let batch_body = serde_json::json!({
        "prompt": ["First prompt", "First prompt"],
        "max_tokens": 4
    });
    let batch_json = post_completions(app, batch_body).await;
    let batch_prompt_tokens = batch_json["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt_tokens");

    assert_eq!(
        batch_prompt_tokens,
        single_prompt_tokens * 2,
        "two identical prompts in a batch must double the aggregated prompt_tokens"
    );
}

/// A batch larger than `MAX_COMPLETION_BATCH_SIZE` must be rejected with an
/// honest `400`, not silently truncated.
#[tokio::test]
async fn test_completions_batch_over_cap_rejected() {
    let app = test_router();
    let prompts: Vec<String> = (0..(MAX_COMPLETION_BATCH_SIZE + 1))
        .map(|i| format!("prompt {i}"))
        .collect();
    let body = serde_json::json!({
        "prompt": prompts,
        "max_tokens": 4
    });
    let req = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("body serialisation"),
        ))
        .expect("request build");
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "batch over the cap must be rejected with 400, not silently truncated"
    );
}

/// An empty prompt batch (`"prompt": []`) must be rejected with `400` rather
/// than silently generating from an empty string.
#[tokio::test]
async fn test_completions_empty_batch_rejected() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": [],
        "max_tokens": 4
    });
    let req = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("body serialisation"),
        ))
        .expect("request build");
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── PromptInput unit tests ─────────────────────────────────────────────────────

/// `PromptInput::Single` must return a vec with one element equal to the string.
#[test]
fn test_prompt_input_as_strings_single() {
    use oxibonsai_runtime::completions::PromptInput;
    let p = PromptInput::Single("hello world".to_string());
    let strings = p.as_strings();
    assert_eq!(strings.len(), 1);
    assert_eq!(strings[0], "hello world");
}

/// `PromptInput::Batch` must return a vec matching the original slice.
#[test]
fn test_prompt_input_as_strings_batch() {
    use oxibonsai_runtime::completions::PromptInput;
    let p = PromptInput::Batch(vec![
        "alpha".to_string(),
        "beta".to_string(),
        "gamma".to_string(),
    ]);
    let strings = p.as_strings();
    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], "alpha");
    assert_eq!(strings[1], "beta");
    assert_eq!(strings[2], "gamma");
}

/// `finish_reason` must be a non-empty string.
#[tokio::test]
async fn test_completions_finish_reason_present() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": "finish?",
        "max_tokens": 4
    });
    let json = post_completions(app, body).await;
    let reason = json["choices"][0]["finish_reason"]
        .as_str()
        .expect("finish_reason must be a string");
    assert!(
        reason == "stop" || reason == "length",
        "unexpected finish_reason: {reason}"
    );
}

/// The `created` field must be a positive integer (Unix timestamp).
#[tokio::test]
async fn test_completions_created_timestamp() {
    let app = test_router();
    let body = serde_json::json!({
        "prompt": "timestamp test",
        "max_tokens": 2
    });
    let json = post_completions(app, body).await;
    let created = json["created"].as_u64().expect("created must be a u64");
    // Any reasonable Unix timestamp is well above 1 billion.
    assert!(
        created > 1_000_000_000,
        "created timestamp looks invalid: {created}"
    );
}
