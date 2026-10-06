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

// ── with_served_model_id / the served context window ─────────────────────

/// `general.name` of the Bonsai 2 27B GGUFs, and the file they ship in.
const PLACEHOLDER_NAME: &str = "Hf";
const FILE_STEM: &str = "Ternary-Bonsai-2-27B-PQ2_0";

/// A tiny dense engine whose model reports `name` as its `general.name`.
fn engine_named(name: &str) -> InferenceEngine<'static> {
    let mut config = oxibonsai_core::config::Qwen3Config::tiny_test();
    config.model_name = name.to_string();
    InferenceEngine::new(config, SamplingParams::default(), 42)
}

/// A router over one replica named `name`, prompt start token configured so
/// text requests run without a tokenizer.
fn router_named(name: &str, options: RouterOptions) -> Router {
    create_router_full(
        EnginePool::new(vec![engine_named(name)]),
        None,
        Arc::new(InferenceMetrics::new()),
        options.with_prompt_start_token(fx::QWEN3_IM_START),
    )
}

/// The hybrid `qwen35` fixture (declared context 4096) on the CPU model with
/// a KV window of `window` positions.
fn hybrid_engine(window: usize) -> InferenceEngine<'static> {
    let bytes: &'static [u8] =
        Box::leak(oxibonsai_testkit::qwen35_fixture::synthetic_qwen35_gguf().into_boxed_slice());
    let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> = Box::leak(Box::new(
        oxibonsai_core::gguf::reader::GgufFile::parse(bytes).expect("the fixture parses"),
    ));
    InferenceEngine::from_gguf_with_backend(
        gguf,
        SamplingParams::default(),
        42,
        window,
        crate::engine_seam::Backend::Cpu,
    )
    .expect("a CPU hybrid engine")
}

/// A model whose `general.name` is a placeholder is listed under its file's
/// stem, everywhere the id is shown: `GET /v1/models`, `GET
/// /v1/models/{id}` (the placeholder no longer resolves), and the `model`
/// member of a chat, a streamed chat and a completion answer.
#[tokio::test]
async fn a_placeholder_general_name_is_served_under_the_file_stem() {
    let id = crate::multi_model::served_model_id(Some(PLACEHOLDER_NAME), Some(FILE_STEM))
        .expect("the stem stands in for the placeholder");
    assert_eq!(id, FILE_STEM);
    let app = || {
        router_named(
            PLACEHOLDER_NAME,
            RouterOptions::default().with_served_model_id(&id),
        )
    };

    let (status, json) = get_json(app(), "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["data"][0]["id"], FILE_STEM, "{json}");
    assert_eq!(json["data"].as_array().map(Vec::len), Some(1), "{json}");

    let (status, json) = get_json(app(), &format!("/v1/models/{FILE_STEM}"), None).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["id"], FILE_STEM, "{json}");
    let (status, json) = get_json(app(), &format!("/v1/models/{PLACEHOLDER_NAME}"), None).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "the placeholder names no model: {json}"
    );

    let chat = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
    });
    let (status, json) = post_json(app(), "/v1/chat/completions", chat.clone()).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["model"], FILE_STEM, "{json}");

    let mut streamed = chat;
    streamed["stream"] = serde_json::json!(true);
    let (status, _, sse) = fx::post(app(), "/v1/chat/completions", streamed).await;
    assert_eq!(status, StatusCode::OK, "{sse}");
    let chunks = fx::sse_payloads(&sse);
    assert!(!chunks.is_empty(), "{sse}");
    assert!(
        chunks.iter().all(|chunk| chunk["model"] == FILE_STEM),
        "every chunk carries the served id: {sse}"
    );

    let (status, json) = post_json(
        app(),
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "max_tokens": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["model"], FILE_STEM, "{json}");
}

/// A real `general.name` is the served id; the descriptor keeps both the
/// model's own name (`id`, what `/admin/config` reports) and the served id.
#[tokio::test]
async fn a_real_general_name_is_served_as_it_is() {
    let name = "Ternary-Bonsai-1.7B";
    let id = crate::multi_model::served_model_id(Some(name), Some("some-renamed-file"))
        .expect("a usable name");
    assert_eq!(id, name);
    let app = router_named(name, RouterOptions::default().with_served_model_id(id));
    let (status, json) = get_json(app, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["data"][0]["id"], name, "{json}");

    let pool = EnginePool::new(vec![engine_named(PLACEHOLDER_NAME)]);
    let info = ServedModelInfo::new(pool, Some(FILE_STEM.to_string()));
    let descriptor = info.descriptor().await;
    assert_eq!(descriptor.served_id, FILE_STEM);
    assert_eq!(
        descriptor.id, PLACEHOLDER_NAME,
        "the model's own name is what the operator surface keeps reporting"
    );
}

/// A router built without a launcher-chosen id lists the name the loaded
/// model reports, as before; a blank id is no id.
#[tokio::test]
async fn without_a_served_id_the_models_reported_name_is_listed() {
    for options in [
        RouterOptions::default(),
        RouterOptions::default().with_served_model_id("  "),
    ] {
        let (status, json) =
            get_json(router_named(PLACEHOLDER_NAME, options), "/v1/models", None).await;
        assert_eq!(status, StatusCode::OK, "{json}");
        assert_eq!(json["data"][0]["id"], PLACEHOLDER_NAME, "{json}");
    }
}

/// The context a server reports and enforces is the engine's KV window when
/// that is smaller than the context the model declares.
#[tokio::test]
async fn the_served_context_is_the_kv_window_not_the_declared_context() {
    const WINDOW: usize = 64;
    let pool = EnginePool::new(vec![hybrid_engine(WINDOW)]);
    let info = ServedModelInfo::new(Arc::clone(&pool), None);
    let descriptor = info.descriptor().await;
    assert_eq!(
        descriptor.declared_context_length,
        oxibonsai_testkit::qwen35_fixture::CONTEXT_LENGTH,
        "the file declares its own context"
    );
    assert_eq!(
        descriptor.max_context_length, WINDOW,
        "the engine can run WINDOW positions"
    );
    assert_eq!(descriptor.architecture, "qwen35");

    let app = create_router_full(
        pool,
        None,
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default(),
    );
    let (status, json) = get_json(app, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["data"][0]["max_context_length"], WINDOW, "{json}");
}

/// A dense engine's window is its declared context unless it was built with
/// less: the served context never exceeds either.
#[tokio::test]
async fn a_dense_engine_serves_its_declared_context() {
    let pool = EnginePool::new(vec![engine_named("Bonsai-Tiny-Test")]);
    let descriptor = ServedModelInfo::new(pool, None).descriptor().await;
    assert!(descriptor.max_context_length > 0);
    assert_eq!(
        descriptor.max_context_length,
        descriptor.declared_context_length
    );
    assert_eq!(
        descriptor.served_id, descriptor.id,
        "no launcher-chosen id: served under the model's own name"
    );
}

// ── The per-request deadline names the stage it caught the request in ────

/// A request queued behind a busy replica waits for the whole deadline; the
/// `504` says that it was waiting (not generating), streamed or not. The
/// descriptor cache is warmed first, while the replica is free: its first
/// resolution briefly acquires a replica of its own (`GET /v1/models` does it
/// without a generation, so no host speed can make the warm-up outlast the
/// deadline).
#[tokio::test]
async fn a_request_queued_behind_a_busy_replica_times_out_naming_the_wait() {
    // Short enough to keep the test quick; nothing below depends on a
    // generation finishing inside it.
    const TIMEOUT_MS: u64 = 400;
    let pool = EnginePool::new(vec![tiny_engine()]);
    let app = create_router_full(
        Arc::clone(&pool),
        None,
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default()
            .with_prompt_start_token(fx::QWEN3_IM_START)
            .with_limits(RequestLimits::default().with_timeout_ms(TIMEOUT_MS)),
    );
    let body = |stream: bool| {
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": stream,
        })
    };
    let (status, json) = get_json(app.clone(), "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "the warm-up request: {json}");

    let held = pool.acquire().await.expect("the only replica");
    for stream in [false, true] {
        let (status, json) = post_json(app.clone(), "/v1/chat/completions", body(stream)).await;
        assert_eq!(
            status,
            StatusCode::GATEWAY_TIMEOUT,
            "stream={stream}: {json}"
        );
        assert_eq!(json["error"]["code"], "request_timeout", "{json}");
        assert_eq!(json["error"]["phase"], "waiting_for_engine", "{json}");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(&format!("{TIMEOUT_MS} ms")),
            "names the limit: {message}"
        );
        assert!(
            message.contains("waiting for a free engine replica"),
            "names the stage: {message}"
        );
    }
    drop(held);

    // The timed-out requests left nothing holding the replica: it is free
    // again.
    let again = tokio::time::timeout(std::time::Duration::from_secs(30), pool.acquire())
        .await
        .expect("the replica is released, not leaked by the timeouts")
        .expect("the pool hands it out");
    drop(again);
}
