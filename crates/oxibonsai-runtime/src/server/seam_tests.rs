//! The `RouterOptions` seams a serve binary configures a router through —
//! `with_engine_report`, `with_embeddings_registry`,
//! `with_embedder_unavailable`, `with_prompt_start_token` — and the
//! `400 tokenizer_required` a tokenizer-less server answers text prompts
//! with when no prompt start token is configured.
//!
//! Attached to `server.rs` as a `#[cfg(test)]` child module.

use super::*;
use crate::tokenizer_bridge::chat_render::test_fixtures as fx;

/// A tiny-model engine (no tokenizer attached to its router).
fn tiny_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        SamplingParams::default(),
        42,
    )
}

/// `create_router_full` over a one-replica pool of [`tiny_engine`].
fn router_with(options: RouterOptions) -> Router {
    create_router_full(
        EnginePool::new(vec![tiny_engine()]),
        None,
        Arc::new(InferenceMetrics::new()),
        options,
    )
}

/// GET `path` on `app`, returning the status and the JSON body.
async fn get_json(
    app: Router,
    path: &str,
    bearer: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    let mut builder = axum::http::Request::get(path);
    if let Some(token) = bearer {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    let req = builder
        .body(axum::body::Body::empty())
        .expect("build request");
    let resp = tower::ServiceExt::oneshot(app, req)
        .await
        .expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

/// POST `body` to `path` on `app`, returning the status and the JSON body
/// (`Null` for a body that is not one JSON document, e.g. SSE).
async fn post_json(
    app: Router,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let (status, _, text) = fx::post(app, path, body).await;
    (
        status,
        serde_json::from_str(&text).unwrap_or(serde_json::Value::Null),
    )
}

// ── with_engine_report ────────────────────────────────────────────────────

#[tokio::test]
async fn router_seam_with_engine_report_reaches_the_admin_surface() {
    const TOKEN: &str = "seam-admin-token";
    let report = crate::admin::EngineReport {
        variant: "Seam-Test-Variant".to_string(),
        quant_type: "Q1_0".to_string(),
        kernel_tier: "scalar".to_string(),
        kernel_tier_reason: "a router seam test".to_string(),
        kernel_label: "Q1_0 scalar".to_string(),
        backend: "cpu".to_string(),
        description: "a report attached through RouterOptions".to_string(),
    };
    let app = router_with(
        RouterOptions::default()
            .with_auth(AuthConfig::with_admin_token(TOKEN))
            .with_engine_report(report.clone()),
    );
    let (status, json) = get_json(app.clone(), "/admin/config", Some(TOKEN)).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["engine"]["variant"], "Seam-Test-Variant", "{json}");
    assert_eq!(json["engine"]["kernel_label"], "Q1_0 scalar", "{json}");
    let (status, json) = get_json(app, "/admin/status", Some(TOKEN)).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["engine"]["description"], report.description, "{json}");
}

// ── with_embeddings_registry ──────────────────────────────────────────────

/// A configured non-model registry is served natively (here: the identity
/// backend of a registry that does not require a model), instead of the
/// router's own model-only registry that answers `501`.
#[tokio::test]
async fn router_seam_with_embeddings_registry_serves_the_configured_registry() {
    let request = serde_json::json!({"input": "hello world"});
    let (status, json) = post_json(
        router_with(RouterOptions::default()),
        "/v1/embeddings",
        request.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{json}");

    let app = router_with(
        RouterOptions::default()
            .with_embeddings_registry(crate::embeddings::EmbedderRegistry::new(16)),
    );
    let (status, json) = post_json(app, "/v1/embeddings", request).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["model"], "bonsai-embeddings-identity", "{json}");
    assert_eq!(
        json["data"][0]["embedding"].as_array().map(Vec::len),
        Some(16),
        "{json}"
    );
}

/// The precedence rule's other half: a configured registry that itself
/// requires a model backend (and has none) keeps its `501`, carrying the
/// recorded reason.
#[tokio::test]
async fn router_seam_with_embeddings_registry_keeps_its_own_model_requirement() {
    let app = router_with(
        RouterOptions::default()
            .with_embeddings_registry(
                crate::embeddings::EmbedderRegistry::new(16).with_require_model_backend(true),
            )
            .with_embedder_unavailable(None, "no embedder was configured"),
    );
    let (status, json) = post_json(app, "/v1/embeddings", serde_json::json!({"input": "x"})).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{json}");
    assert_eq!(json["error"]["code"], "embeddings_unavailable", "{json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|m| m.ends_with(": no embedder was configured")),
        "{json}"
    );
}

// ── with_embedder_unavailable ─────────────────────────────────────────────

/// The reason a server has no model-backed embedder — here the typed engine
/// refusal a hybrid model's embedder construction produces — is in the
/// `/v1/embeddings` `501` body itself, not only in the operator's log.
#[tokio::test]
async fn embedder_unavailable_reason_reaches_the_501_body() {
    let refusal = crate::engine_seam::EngineError::NotADenseModel {
        operation: "embeddings",
        architecture: "qwen35".to_string(),
        reason: "the hybrid model has no dense hidden-state embedding path",
    };
    let app = router_with(
        RouterOptions::default()
            .with_embedder_unavailable(Some(refusal.error_code()), refusal.to_string()),
    );
    let (status, json) =
        post_json(app, "/v1/embeddings", serde_json::json!({"input": "hello"})).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{json}");
    assert_eq!(json["error"]["type"], "not_implemented_error", "{json}");
    assert_eq!(json["error"]["code"], "NOT_A_DENSE_MODEL", "{json}");
    assert_eq!(
        json["error"]["message"],
        format!(
            "{}: {refusal}",
            crate::embeddings::MODEL_BACKEND_MISSING_MESSAGE
        ),
        "{json}"
    );
}

#[tokio::test]
async fn embedder_unavailable_without_a_reason_keeps_the_standard_501() {
    let (status, json) = post_json(
        router_with(RouterOptions::default()),
        "/v1/embeddings",
        serde_json::json!({"input": "hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{json}");
    assert_eq!(json["error"]["code"], "embeddings_unavailable", "{json}");
    assert_eq!(
        json["error"]["message"],
        crate::embeddings::MODEL_BACKEND_MISSING_MESSAGE,
        "{json}"
    );
}

// ── with_prompt_start_token / tokenizer_required ─────────────────────────

/// Every entry point that tokenizes a text prompt, with the field its
/// `400 tokenizer_required` names.
fn text_prompt_requests() -> Vec<(&'static str, serde_json::Value, &'static str)> {
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
    vec![
        (
            "/v1/chat/completions",
            serde_json::json!({"messages": messages, "max_tokens": 2}),
            "messages",
        ),
        (
            "/v1/chat/completions",
            serde_json::json!({"messages": messages, "max_tokens": 2, "stream": true}),
            "messages",
        ),
        (
            "/v1/chat/completions/extended",
            serde_json::json!({"messages": messages, "max_tokens": 2}),
            "messages",
        ),
        (
            "/v1/chat/completions/extended",
            serde_json::json!({"messages": messages, "max_tokens": 2, "stream": true}),
            "messages",
        ),
        (
            "/v1/completions",
            serde_json::json!({"prompt": "hi", "max_tokens": 2}),
            "prompt",
        ),
        (
            "/v1/completions",
            serde_json::json!({"prompt": ["a", "b"], "max_tokens": 2, "stream": true}),
            "prompt",
        ),
    ]
}

/// A tokenizer-less server with no configured prompt start token refuses a
/// text prompt on every entry point with `400 invalid_request_error`, code
/// `tokenizer_required`, naming the prompt field — it never guesses a
/// vocabulary-specific start token.
#[tokio::test]
async fn tokenizer_required_is_the_answer_to_a_text_prompt_without_a_tokenizer() {
    for (path, body, param) in text_prompt_requests() {
        let (status, json) =
            post_json(router_with(RouterOptions::default()), path, body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path} {body}: {json}");
        assert_eq!(
            json["error"]["code"], TOKENIZER_REQUIRED_CODE,
            "{path}: {json}"
        );
        assert_eq!(
            json["error"]["type"], "invalid_request_error",
            "{path}: {json}"
        );
        assert_eq!(json["error"]["param"], param, "{path}: {json}");
        assert!(
            json["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("tokenizer")),
            "the message names the missing tokenizer: {json}"
        );
    }
}

/// `RouterOptions::with_prompt_start_token` lifts the refusal: the same
/// requests run, feeding that single token as the prompt.
#[tokio::test]
async fn tokenizer_required_is_lifted_by_a_configured_prompt_start_token() {
    for (path, body, _) in text_prompt_requests() {
        let app = router_with(RouterOptions::default().with_prompt_start_token(fx::QWEN3_IM_START));
        let (status, json) = post_json(app, path, body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{path} {body}: {json}");
    }
}

/// The configured id really is the prompt: `usage.prompt_tokens` is `1`
/// whatever the text, and the accessor reports it.
#[tokio::test]
async fn tokenizer_required_prompt_start_token_is_the_whole_prompt() {
    let options = RouterOptions::default().with_prompt_start_token(fx::QWEN3_IM_START);
    assert_eq!(options.prompt_start_token, Some(fx::QWEN3_IM_START));
    let (status, json) = post_json(
        router_with(options),
        "/v1/completions",
        serde_json::json!({"prompt": "a much longer prompt than one token", "max_tokens": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["usage"]["prompt_tokens"], 1, "{json}");
    assert_eq!(
        json["choices"][0]["text"], "",
        "no tokenizer, no text: {json}"
    );
}

/// Without a tokenizer there is no text to render, so both chat endpoints
/// answer with empty content — non-streamed exactly like streamed, which
/// sends no content delta at all — while `usage` still counts every
/// generated token. (It used to be the debug rendering of the id list.)
#[tokio::test]
async fn a_tokenizerless_chat_answer_is_empty_content_streamed_or_not() {
    for path in ["/v1/chat/completions", "/v1/chat/completions/extended"] {
        let body = |stream: bool| {
            serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 3,
                "stream": stream,
            })
        };
        let app = router_with(RouterOptions::default().with_prompt_start_token(fx::QWEN3_IM_START));
        let (status, json) = post_json(app, path, body(false)).await;
        assert_eq!(status, StatusCode::OK, "{path}: {json}");
        assert_eq!(
            json["choices"][0]["message"]["content"], "",
            "{path}: {json}"
        );
        assert!(
            json["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
            "{path}: {json}"
        );

        let app = router_with(RouterOptions::default().with_prompt_start_token(fx::QWEN3_IM_START));
        let (status, _, sse) = fx::post(app, path, body(true)).await;
        assert_eq!(status, StatusCode::OK, "{path}: {sse}");
        assert!(fx::delta_texts(&sse, "content").is_empty(), "{path}: {sse}");
        assert!(sse.trim_end().ends_with("data: [DONE]"), "{path}: {sse}");
    }
}

/// A server WITH a tokenizer never consults the prompt start token.
#[tokio::test]
async fn tokenizer_required_never_applies_to_a_server_with_a_tokenizer() {
    let app = create_router(fx::scripted_byte_engine("ok"), Some(fx::byte_tokenizer()));
    let (status, json) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "abc", "max_tokens": 4}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["usage"]["prompt_tokens"], 3, "{json}");
    assert_eq!(json["choices"][0]["text"], "ok", "{json}");
}
