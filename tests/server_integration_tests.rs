//! Server integration tests covering all endpoints.
//!
//! Uses `axum::body::Body` + `tower::ServiceExt::oneshot` to make in-process
//! HTTP requests without binding to a real socket.  All tests are gated on the
//! `server` feature (the default).

#[cfg(feature = "server")]
mod server_tests {
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{header, Method, Request, StatusCode},
    };
    use bytes::Bytes;
    use http_body_util::BodyExt;
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::{
        admin::AdminState, engine::InferenceEngine, metrics::InferenceMetrics,
        sampling::SamplingParams, server::create_router_with_metrics,
    };
    use serde_json::Value;
    use tower::ServiceExt;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_server() -> axum::Router {
        let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
        let metrics = Arc::new(InferenceMetrics::new());
        create_router_with_metrics(engine, None, metrics)
    }

    async fn collect_body(body: Body) -> Bytes {
        body.collect()
            .await
            .expect("body collection should succeed")
            .to_bytes()
    }

    async fn body_json(body: Body) -> Value {
        let bytes = collect_body(body).await;
        serde_json::from_slice(&bytes).expect("response should be valid JSON")
    }

    // ── Health endpoint ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_endpoint_ok() {
        let app = make_server();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(resp.status(), StatusCode::OK, "/health must return 200 OK");
    }

    // ── Models endpoint ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_models_endpoint_reports_loaded_model() {
        // `/v1/models` reports the *actually-loaded* model, reading its `id`
        // from the engine's real configuration — never a hard-coded literal.
        // `make_server` loads `Qwen3Config::tiny_test()`, whose `model_name` is
        // `"Bonsai-Tiny-Test"`, so that is the id the endpoint must return.
        let app = make_server();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/v1/models")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(resp.status(), StatusCode::OK, "/v1/models must return 200");

        let json = body_json(resp.into_body()).await;
        assert_eq!(
            json["object"].as_str(),
            Some("list"),
            "object must be 'list'"
        );
        let models = json["data"].as_array().expect("data must be an array");
        let ids: Vec<&str> = models.iter().filter_map(|m| m["id"].as_str()).collect();

        let expected = Qwen3Config::tiny_test().model_name;
        assert!(
            ids.contains(&expected.as_str()),
            "models list must report the loaded model {expected:?}; got: {ids:?}"
        );
    }

    // ── Chat completions ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_chat_completions_basic() {
        let app = make_server();

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": false
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/chat/completions basic request must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert!(
            json["choices"].as_array().is_some(),
            "response must contain 'choices'"
        );
    }

    // ── Completions endpoint ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_completions_endpoint_basic() {
        let app = make_server();

        let body = serde_json::json!({
            "prompt": "hello",
            "max_tokens": 2,
            "temperature": 0.0
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/completions basic request must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert_eq!(
            json["object"].as_str(),
            Some("text_completion"),
            "completions response must have object=text_completion"
        );
    }

    // ── Embeddings endpoint ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_embeddings_endpoint_refuses_without_a_model_backend() {
        // D-1 (wave 2.5, `RT-EMBEDDINGS` blocking 1; re-confirmed FIX3-BUILD,
        // wave 3.5): `make_server()` builds its router the same way the real
        // `oxibonsai serve` binary does (`server.rs::create_router_full`),
        // which mounts `/v1/embeddings` via
        // `crate::embeddings::create_embeddings_router_requiring_model`. That
        // constructor disables the stateless TF-IDF/`IdentityEmbedder`
        // fallback, so with no model-backed `Embedder` installed (there is no
        // seam to build one from yet — see `oxibonsai_runtime::embeddings`'s
        // module docs), the endpoint must refuse honestly with `501 Not
        // Implemented` naming the missing backend rather than silently
        // answer `200` with a non-semantic byte-hash vector. This replaces
        // the former `test_embeddings_endpoint_basic`, which asserted `200`
        // against this same router and went red the moment D-1 landed; the
        // bare, unaffected `create_embeddings_router` (still `200` by its
        // own documented, unchanged contract) is covered elsewhere, e.g.
        // `tests/cli_surface_tests.rs`'s `embeddings_base64` module and
        // `crates/oxibonsai-runtime/tests/embeddings_tests.rs`.
        let app = make_server();

        let body = serde_json::json!({
            "model": "bonsai-8b",
            "input": "hello world"
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::NOT_IMPLEMENTED,
            "/v1/embeddings must refuse with 501 when no model-backed embedder is installed \
             (D-1: the stateless TF-IDF/identity fallback is disabled on the running server), \
             not silently answer 200 with a byte-hash vector"
        );

        let json = body_json(resp.into_body()).await;
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("model-backed"),
            "the 501 body must name the missing model-backed embedder, not just carry a bare \
             status code; got: {json}"
        );
    }

    // ── Prometheus metrics endpoint ───────────────────────────────────────────

    #[tokio::test]
    async fn test_metrics_endpoint_prometheus_format() {
        let app = make_server();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/metrics")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(resp.status(), StatusCode::OK, "/metrics must return 200");

        let bytes = collect_body(resp.into_body()).await;
        let text = std::str::from_utf8(&bytes).expect("metrics should be valid UTF-8");
        // Prometheus text format uses `#` for comments / TYPE declarations,
        // or may be empty when no metrics have been recorded yet.
        assert!(
            text.is_empty() || text.contains('#') || text.contains('_'),
            "Prometheus output should be valid text format; got: {text}"
        );
    }

    // ── Extended chat completions ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_extended_chat_completions_basic() {
        let app = make_server();

        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "n": 1
        });

        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions/extended")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/chat/completions/extended must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert!(
            json["choices"].as_array().is_some(),
            "extended chat response must contain 'choices'"
        );
    }

    // ── Admin status endpoint ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_status_endpoint() {
        // Build a standalone admin router (without wrapping in the main server).
        let metrics = Arc::new(InferenceMetrics::new());
        let state = Arc::new(AdminState::new(Arc::clone(&metrics)));

        // create_admin_router returns Router<Arc<AdminState>> (already state-seeded).
        // We need to convert it to Router<()> via into_make_service or finalise state.
        let admin_router = axum::Router::new()
            .route(
                "/admin/status",
                axum::routing::get(oxibonsai_runtime::admin::get_status),
            )
            .with_state(state);

        let req = Request::builder()
            .method(Method::GET)
            .uri("/admin/status")
            .body(Body::empty())
            .expect("request should build");

        let resp = admin_router
            .oneshot(req)
            .await
            .expect("oneshot should succeed");

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/admin/status must return 200"
        );

        let json = body_json(resp.into_body()).await;
        assert!(
            json["version"].as_str().is_some(),
            "admin status must include version field; got: {json}"
        );
        assert!(
            json["uptime_secs"].as_u64().is_some(),
            "admin status must include uptime_secs; got: {json}"
        );
    }
}

// When the server feature is not enabled, provide a placeholder so the file
// compiles cleanly.
#[cfg(not(feature = "server"))]
#[test]
fn test_server_skipped_no_feature() {
    eprintln!("server feature not enabled; skipping server integration tests");
}
