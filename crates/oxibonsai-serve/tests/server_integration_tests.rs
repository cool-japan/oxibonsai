//! End-to-end integration tests for the oxibonsai-serve HTTP surface.
//!
//! These boot an Axum router on an ephemeral `127.0.0.1:0` port, issue real
//! HTTP requests via `reqwest`, then verify status codes, headers, and the
//! OpenAI-compatible error envelope.
//!
//! The tests intentionally use `Qwen3Config::tiny_test()` so no GGUF file is
//! required; the focus is on HTTP plumbing (auth, CORS-free defaults, health
//! checks, metrics, JSON error shape), not on generation quality.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::error_handling::HandleErrorLayer;
use axum::extract::Query;
use axum::http::{header, Request, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::BoxError;
use axum::Json;
use axum::Router;
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router, serve_with_shutdown};
use tokio::sync::oneshot;
use tower::ServiceBuilder;

// ─── Shared helpers ───────────────────────────────────────────────────────

fn tiny_engine(seed: u64) -> InferenceEngine<'static> {
    let config = Qwen3Config::tiny_test();
    let sampling = SamplingParams::default();
    InferenceEngine::new(config, sampling, seed)
}

/// Bind to an ephemeral port, spawn the server, return the live socket.
async fn spawn_server(router: Router) -> (SocketAddr, oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr = listener.local_addr().expect("local_addr");

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_future = async move {
        let _ = shutdown_rx.await;
    };

    tokio::spawn(async move {
        // `axum::serve` consumes the TcpListener directly.
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_future)
            .await;
    });

    // Give the server a moment to enter its accept loop.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, shutdown_tx)
}

fn client_with_timeout() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("reqwest client")
}

/// Mirror of the bearer-auth middleware in `main.rs` — kept here so integration
/// tests can exercise it without booting the binary.
mod bearer_auth {
    use super::*;
    use axum::extract::State;
    use axum::http::header;

    #[derive(Debug, Clone)]
    pub struct BearerAuthState {
        pub token: String,
    }

    pub async fn middleware(
        State(state): State<BearerAuthState>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        let path = req.uri().path();
        if path == "/health" || path == "/metrics" {
            return next.run(req).await;
        }
        let header_value = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());
        let presented = match header_value.and_then(|h| h.strip_prefix("Bearer ")) {
            Some(tok) => tok.trim(),
            None => {
                return unauthorized("missing or malformed Authorization header").into_response();
            }
        };
        if presented != state.token {
            return unauthorized("invalid bearer token").into_response();
        }
        next.run(req).await
    }

    fn unauthorized(msg: &str) -> (StatusCode, Json<serde_json::Value>) {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": {
                    "message": msg,
                    "type": "auth_error",
                    "param": null,
                    "code": null,
                }
            })),
        )
    }
}

// ─── Health ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let router = create_router(tiny_engine(1), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("health request");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.text().await.expect("body");
    assert_eq!(body, "ok");

    let _ = shutdown.send(());
}

// ─── Metrics ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_body() {
    let router = create_router(tiny_engine(2), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("metrics request");
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/plain"),
        "expected text/plain content-type, got: {ct}"
    );

    let _ = shutdown.send(());
}

// ─── Models list ──────────────────────────────────────────────────────────

#[tokio::test]
async fn models_endpoint_returns_list() {
    let router = create_router(tiny_engine(3), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .expect("models request");
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array(), "data field should be an array");

    let _ = shutdown.send(());
}

// ─── Chat completions: malformed / bad request ────────────────────────────

#[tokio::test]
async fn chat_completions_rejects_malformed_body() {
    let router = create_router(tiny_engine(4), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header(header::CONTENT_TYPE, "application/json")
        .body("this is not JSON")
        .send()
        .await
        .expect("bad body");
    // Axum rejects malformed JSON with 400 or 415; accept either.
    assert!(
        matches!(
            resp.status(),
            StatusCode::BAD_REQUEST
                | StatusCode::UNPROCESSABLE_ENTITY
                | StatusCode::UNSUPPORTED_MEDIA_TYPE
        ),
        "got status {}",
        resp.status()
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn chat_completions_rejects_missing_fields() {
    let router = create_router(tiny_engine(5), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    // No "messages" field.
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .expect("missing messages");
    assert!(
        matches!(
            resp.status(),
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY
        ),
        "got status {}",
        resp.status()
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn chat_completions_rejects_negative_max_tokens() {
    let router = create_router(tiny_engine(6), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": -5,
        }))
        .send()
        .await
        .expect("negative max_tokens");
    // serde rejects negative values for a usize field with 422.
    assert!(
        matches!(
            resp.status(),
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY
        ),
        "got status {}",
        resp.status()
    );

    let _ = shutdown.send(());
}

// ─── Bearer authentication ────────────────────────────────────────────────

#[tokio::test]
async fn bearer_auth_rejects_missing_header() {
    let state = bearer_auth::BearerAuthState {
        token: "test-token-1234567890".to_string(),
    };
    let router = create_router(tiny_engine(7), None)
        .layer(from_fn_with_state(state, bearer_auth::middleware));
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .expect("no auth header");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["type"], "auth_error");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn bearer_auth_rejects_wrong_token() {
    let state = bearer_auth::BearerAuthState {
        token: "test-token-1234567890".to_string(),
    };
    let router = create_router(tiny_engine(8), None)
        .layer(from_fn_with_state(state, bearer_auth::middleware));
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .bearer_auth("wrong-token")
        .send()
        .await
        .expect("wrong token");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn bearer_auth_accepts_correct_token() {
    let state = bearer_auth::BearerAuthState {
        token: "correct-token-abcdefghij".to_string(),
    };
    let router = create_router(tiny_engine(9), None)
        .layer(from_fn_with_state(state, bearer_auth::middleware));
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .bearer_auth("correct-token-abcdefghij")
        .send()
        .await
        .expect("correct token");
    assert_eq!(resp.status(), StatusCode::OK);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn bearer_auth_health_endpoint_bypassed() {
    let state = bearer_auth::BearerAuthState {
        token: "token-abcdefghijklmnop".to_string(),
    };
    let router = create_router(tiny_engine(10), None)
        .layer(from_fn_with_state(state, bearer_auth::middleware));
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    // No auth header — /health must still return 200.
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("health no auth");
    assert_eq!(resp.status(), StatusCode::OK);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn bearer_auth_metrics_endpoint_bypassed() {
    let state = bearer_auth::BearerAuthState {
        token: "token-abcdefghijklmnop".to_string(),
    };
    let router = create_router(tiny_engine(11), None)
        .layer(from_fn_with_state(state, bearer_auth::middleware));
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("metrics no auth");
    assert_eq!(resp.status(), StatusCode::OK);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn bearer_auth_rejects_malformed_header() {
    let state = bearer_auth::BearerAuthState {
        token: "token-abcdefghijklmnop".to_string(),
    };
    let router = create_router(tiny_engine(12), None)
        .layer(from_fn_with_state(state, bearer_auth::middleware));
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    // "Basic" instead of "Bearer"
    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .header(header::AUTHORIZATION, "Basic dXNlcjpwYXNz")
        .send()
        .await
        .expect("basic auth");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let _ = shutdown.send(());
}

// ─── Admission control (concurrency limit + timeout) ──────────────────────
//
// Mirrors the `ServiceBuilder` admission stack assembled in `src/main.rs`
// ("── 7b. Admission control") so these can be exercised without booting the
// binary — the same pattern already used for `bearer_auth` above. Verifies
// that `limits.max_concurrent_requests` / `limits.per_request_timeout_ms` are
// enforced HTTP-level controls, not just validated-but-inert config fields.

/// Sleeps for `ms` (query param, default 200) then returns `200 OK`. Lets
/// tests control exactly how long a request stays in flight.
async fn slow_handler(Query(params): Query<HashMap<String, String>>) -> StatusCode {
    let ms: u64 = params.get("ms").and_then(|s| s.parse().ok()).unwrap_or(200);
    tokio::time::sleep(Duration::from_millis(ms)).await;
    StatusCode::OK
}

/// Mirror of `main.rs`'s `handle_admission_error`.
async fn handle_admission_error(err: BoxError) -> (StatusCode, Json<serde_json::Value>) {
    let (status, kind) = if err.is::<tower::load_shed::error::Overloaded>() {
        (StatusCode::SERVICE_UNAVAILABLE, "overloaded_error")
    } else if err.is::<tower::timeout::error::Elapsed>() {
        (StatusCode::REQUEST_TIMEOUT, "timeout_error")
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
    };
    (
        status,
        Json(serde_json::json!({
            "error": { "message": err.to_string(), "type": kind, "param": null, "code": null }
        })),
    )
}

/// Mirror of `main.rs`'s "── 7b. Admission control" `ServiceBuilder` stack.
///
/// Must use `GlobalConcurrencyLimitLayer` (pre-built `Arc<Semaphore>`), not
/// `ServiceBuilder::concurrency_limit`/`tower::limit::ConcurrencyLimitLayer` —
/// `axum::Router::layer` applies the given layer once *per route*
/// (`PathRouter::layer` clones and re-applies it per registered route), so a
/// bare `ConcurrencyLimitLayer` would silently allocate one independent
/// semaphore per route instead of a single budget shared across the whole
/// router. `admission_concurrency_limit_sheds_excess_requests` below is the
/// regression test that catches a reintroduction of that bug.
fn with_admission_layer(
    router: Router,
    max_concurrent_requests: usize,
    per_request_timeout_ms: u64,
) -> Router {
    let concurrency_semaphore =
        tower::limit::GlobalConcurrencyLimitLayer::new(max_concurrent_requests);
    let admission = ServiceBuilder::new()
        .layer(HandleErrorLayer::new(handle_admission_error))
        .load_shed()
        .layer(concurrency_semaphore)
        .timeout(Duration::from_millis(per_request_timeout_ms));
    router.layer(admission)
}

#[tokio::test]
async fn admission_timeout_aborts_slow_request_with_408() {
    let router = create_router(tiny_engine(30), None)
        .route("/__test_slow", axum::routing::get(slow_handler));
    let router = with_admission_layer(
        router, /* max_concurrent_requests */ 10, /* per_request_timeout_ms */ 50,
    );
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/__test_slow?ms=400"))
        .send()
        .await
        .expect("slow request");
    assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["type"], "timeout_error");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn admission_fast_request_within_timeout_succeeds() {
    let router = create_router(tiny_engine(31), None)
        .route("/__test_slow", axum::routing::get(slow_handler));
    let router = with_admission_layer(
        router, /* max_concurrent_requests */ 10, /* per_request_timeout_ms */ 500,
    );
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/__test_slow?ms=20"))
        .send()
        .await
        .expect("fast-enough request");
    assert_eq!(resp.status(), StatusCode::OK);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn admission_concurrency_limit_sheds_excess_requests() {
    let router = create_router(tiny_engine(32), None)
        .route("/__test_slow", axum::routing::get(slow_handler));
    let router = with_admission_layer(
        router, /* max_concurrent_requests */ 1, /* per_request_timeout_ms */ 5_000,
    );
    let (addr, shutdown) = spawn_server(router).await;

    // Occupy the single concurrency slot with a long-running request.
    let first = tokio::spawn(async move {
        let c = client_with_timeout();
        c.get(format!("http://{addr}/__test_slow?ms=300"))
            .send()
            .await
    });
    // Give the first request time to actually be admitted (acquire the
    // semaphore permit and start sleeping inside the handler) before firing
    // the second — otherwise both could race for the single slot.
    tokio::time::sleep(Duration::from_millis(80)).await;

    // With max_concurrent_requests == 1, the in-flight first request holds
    // the only slot; the second must be shed immediately (503) rather than
    // queued until the first completes.
    let client = client_with_timeout();
    let second = client
        .get(format!("http://{addr}/__test_slow?ms=10"))
        .send()
        .await
        .expect("second request");
    assert_eq!(second.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = second.json().await.expect("json");
    assert_eq!(body["error"]["type"], "overloaded_error");

    let first_resp = first
        .await
        .expect("join first request task")
        .expect("first request");
    assert_eq!(first_resp.status(), StatusCode::OK);

    let _ = shutdown.send(());
}

// ─── Unknown routes ───────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_route_returns_404() {
    let router = create_router(tiny_engine(13), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = client_with_timeout();

    let resp = client
        .get(format!("http://{addr}/no-such-path"))
        .send()
        .await
        .expect("unknown route");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    let _ = shutdown.send(());
}

// ─── Graceful shutdown ────────────────────────────────────────────────────

#[tokio::test]
async fn graceful_shutdown_stops_server() {
    let router = create_router(tiny_engine(14), None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener); // immediately release so `serve_with_shutdown` can rebind

    let (tx, rx) = oneshot::channel::<()>();
    let shutdown_signal = async move {
        let _ = rx.await;
    };

    let handle = tokio::spawn(async move {
        serve_with_shutdown(router, addr, shutdown_signal)
            .await
            .expect("serve")
    });

    tokio::time::sleep(Duration::from_millis(80)).await;

    // Sanity: server is up.
    let client = client_with_timeout();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("pre-shutdown");
    assert_eq!(resp.status(), StatusCode::OK);

    // Signal shutdown.
    let _ = tx.send(());

    // Await the task with a generous timeout — if graceful shutdown hangs
    // this test will fail with a timeout error rather than hanging the suite.
    let result = tokio::time::timeout(Duration::from_secs(5), handle).await;
    assert!(result.is_ok(), "graceful shutdown did not complete in time");
}

// ─── Shared-state correctness under concurrency ───────────────────────────

#[tokio::test]
async fn multiple_concurrent_health_checks() {
    let router = create_router(tiny_engine(15), None);
    let (addr, shutdown) = spawn_server(router).await;
    let client = Arc::new(client_with_timeout());

    let mut handles = Vec::new();
    for _ in 0..10 {
        let c = Arc::clone(&client);
        let a = addr;
        handles.push(tokio::spawn(async move {
            c.get(format!("http://{a}/health"))
                .send()
                .await
                .map(|r| r.status())
        }));
    }

    for h in handles {
        let status = h.await.expect("join").expect("reqwest");
        assert_eq!(status, StatusCode::OK);
    }

    let _ = shutdown.send(());
}

// ─── CLI → config env pipeline ────────────────────────────────────────────

#[tokio::test]
async fn server_config_load_composes_across_layers() {
    use oxibonsai_serve::config::{PartialServerConfig, ServerConfig};

    let cli_partial = PartialServerConfig {
        port: Some(12345),
        ..Default::default()
    };
    let cfg = ServerConfig::load(None, None, Some(cli_partial)).expect("load");
    assert_eq!(cfg.bind.port, 12345);
}
