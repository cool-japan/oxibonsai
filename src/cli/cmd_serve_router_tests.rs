//! The `harden_router` composition tests of `cmd_serve.rs`: the full
//! hardened router against a tiny in-memory engine pool (embeddings,
//! admission, timeouts, body limits, CORS, rate limiting). A sibling of
//! `cmd_serve_tests.rs`, declared there as `mod router_tests` via `#[path]`,
//! so `super` still names the `tests` module and every test keeps its path.

use super::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use std::time::Duration;
use tower::ServiceExt;

fn default_opts() -> HardeningOptions {
    HardeningOptions {
        bearer_token: None,
        max_concurrent_requests: 32,
        request_timeout_ms: 60_000,
        max_body_bytes: 4 * 1024 * 1024,
        cors_origins: Vec::new(),
        cors_allow_credentials: false,
        rate_limit_rpm: None,
        rate_limit_burst: 20.0,
        chat_defaults: ChatDefaults::default(),
    }
}

/// A fitted TF-IDF [`oxibonsai_runtime::embeddings::EmbedderRegistry`],
/// for `RouterOptions::with_embeddings_registry`.
fn tfidf_registry(corpus: &[String]) -> oxibonsai_runtime::embeddings::EmbedderRegistry {
    let registry = oxibonsai_runtime::embeddings::EmbedderRegistry::new(TFIDF_MAX_FEATURES);
    registry.fit_tfidf(corpus);
    registry
}

// ── --embedding-backend tfidf ──────────────────────────────────────────

async fn post_embeddings(router: &Router, body: &str) -> (StatusCode, serde_json::Value) {
    let resp = router
        .clone()
        .oneshot(
            Request::post("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// Without a registry the runtime router's model-only route answers the
/// honest 501; with `--embedding-backend tfidf`'s registry attached via
/// `RouterOptions::with_embeddings_registry` (no CLI-side interceptor)
/// the same request is answered from the vocabulary fitted
/// on the corpus — deterministic, reported as `"tfidf"`, and still
/// behind every hardening layer (this router build wraps the base
/// router in the FULL stack, so the TF-IDF route is never a special
/// case).
#[tokio::test]
async fn tfidf_embeddings_are_served_from_the_fitted_corpus() {
    let body = r#"{"input":["the cat sat on the mat","a dog ran"],"model":"m"}"#;
    let plain = router_for(default_opts());
    let (status, _) = post_embeddings(&plain, body).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);

    let corpus: Vec<String> = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a bird sang on the tree",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let router = router_for_with_options(
        default_opts(),
        RouterOptions::default().with_embeddings_registry(tfidf_registry(&corpus)),
    );
    let (status, first) = post_embeddings(&router, body).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(
        first["model"],
        serde_json::json!("bonsai-embeddings-tfidf"),
        "the response names the backend that answered: {first}"
    );
    let vectors = first["data"].as_array().expect("data");
    assert_eq!(vectors.len(), 2);
    let dim = vectors[0]["embedding"].as_array().expect("embedding").len();
    assert!(dim > 0 && dim <= TFIDF_MAX_FEATURES, "dim {dim}");
    let (_, second) = post_embeddings(&router, body).await;
    assert_eq!(
        first["data"], second["data"],
        "the fitted space never drifts"
    );

    // Other routes are untouched.
    let health = router
        .clone()
        .oneshot(
            Request::get("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(health.status(), StatusCode::OK);

    // ...and the bearer layer still guards the TF-IDF route.
    let mut guarded = default_opts();
    guarded.bearer_token = Some("y".repeat(20));
    let guarded_router = router_for_with_options(
        guarded,
        RouterOptions::default().with_embeddings_registry(tfidf_registry(&corpus)),
    );
    let (status, _) = post_embeddings(&guarded_router, body).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

/// The `/v1/embeddings` `501` body carries the reason `build_embedder`
/// handed the router via `RouterOptions::with_embedder_unavailable`
/// (`error.code` and a message naming it), not just the server log —
/// the CLI mirror of the serve crate's own
/// `the_serve_path_without_a_tokenizer_keeps_the_honest_501`.
#[tokio::test]
async fn embeddings_501_body_carries_the_unavailable_reason() {
    let router = router_for_with_options(
        default_opts(),
        RouterOptions::default().with_embedder_unavailable(
            Some("NOT_A_DENSE_MODEL"),
            "no dense embedder for this model",
        ),
    );
    let (status, json) = post_embeddings(&router, r#"{"input":"king","model":"m"}"#).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{json}");
    assert_eq!(json["error"]["code"], "NOT_A_DENSE_MODEL", "{json}");
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("no dense embedder for this model"),
        "the 501 body must name why: {json}"
    );
}

/// [`router_for_with_options`] with no extra `RouterOptions` beyond the
/// locked admin auth and the fixed prompt-start token every test in this
/// module relies on.
fn router_for(opts: HardeningOptions) -> Router {
    router_for_with_options(opts, RouterOptions::default())
}

/// Built under the crate-wide env lock: `RouterOptions::default()` reads
/// `OXI_ADMIN_TOKEN`, which sibling tests may be mutating (the lock is
/// released before any `.await`). `extra` lets a test attach further
/// `RouterOptions` (e.g. `with_embeddings_registry`) before the fixed
/// auth and prompt-start-token are applied on top.
///
/// This router carries no tokenizer (`create_router_full(pool, None,
/// ..)`), so every text-prompting request needs
/// `RouterOptions::with_prompt_start_token` or it would fail fast with
/// `400 tokenizer_required` before ever reaching the admission/auth/
/// rate-limit/body-limit layers these tests actually exercise; no test
/// in this module asserts that contract, so it is set unconditionally.
fn router_for_with_options(opts: HardeningOptions, extra: RouterOptions) -> Router {
    let _env = test_env::lock();
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    let pool = EnginePool::new(vec![engine]);
    let pool_size = pool.size();
    let metrics = Arc::new(InferenceMetrics::new());
    let router_options = extra
        .with_auth(AdminAuthConfig::locked())
        .with_prompt_start_token(151_644);
    let base = create_router_full(pool, None, metrics, router_options);
    harden_router(base, pool_size, &opts, "127.0.0.1")
}

/// Both deadlines are `--request-timeout-ms`, and the admission layer's
/// timer starts first (before the body is read): at the same value it
/// would always win the race and answer a bare `408` from a dropped
/// handler — no cancellation of the in-flight generation, no stage named.
/// The handler's own deadline is the real one and answers first on every
/// generation route (chat, extended chat, completions); the admission
/// timeout stays a backstop, later, for routes with no deadline of their
/// own (here a test-only route whose handler never answers).
#[tokio::test]
async fn the_handler_deadline_answers_before_the_admission_backstop() {
    const TIMEOUT_MS: u64 = 500;
    /// What separates the two timers' start times on the real path: the
    /// time a request spends inside the admission layer before its
    /// handler begins (reading a multi-megabyte image body, the chat
    /// defaults rewrite, extraction). Here a middleware that waits.
    const HANDLER_START_DELAY_MS: u64 = 150;
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    let pool = EnginePool::new(vec![engine]);
    let base = {
        // `RouterOptions::default()` reads `OXI_ADMIN_TOKEN`: built under
        // the env lock, released before any `.await`.
        let _env = test_env::lock();
        create_router_full(
            Arc::clone(&pool),
            None,
            Arc::new(InferenceMetrics::new()),
            RouterOptions::default()
                .with_auth(AdminAuthConfig::locked())
                .with_prompt_start_token(151_644)
                .with_limits(RequestLimits::default().with_timeout_ms(TIMEOUT_MS)),
        )
        // A route with no deadline of its own: its handler never answers.
        .route(
            "/__test_no_deadline",
            axum::routing::post(|| async { std::future::pending::<StatusCode>().await }),
        )
    }
    .layer(axum::middleware::from_fn(
        |request: Request<Body>, next: axum::middleware::Next| async move {
            tokio::time::sleep(Duration::from_millis(HANDLER_START_DELAY_MS)).await;
            next.run(request).await
        },
    ));
    let mut opts = default_opts();
    opts.request_timeout_ms = TIMEOUT_MS;
    let router = harden_router(base, pool.size(), &opts, "127.0.0.1");

    let post = |path: &'static str| {
        let router = router.clone();
        // Each endpoint takes its own request shape (the completions body
        // refuses unknown members).
        let body = if path == "/v1/completions" {
            r#"{"prompt":"hi","max_tokens":2}"#
        } else {
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#
        };
        async move {
            let started = std::time::Instant::now();
            let resp = router
                .oneshot(
                    Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .expect("request"),
                )
                .await
                .expect("response");
            let status = resp.status();
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .expect("body");
            let json: serde_json::Value =
                serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
            (status, json, started.elapsed())
        }
    };
    // Warm the model descriptor while the replica is free (`GET
    // /v1/models` resolves it without a generation, so no host speed can
    // make the warm-up outlast the deadline), then keep the only replica
    // busy so every request queues.
    let warm = router
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(warm.status(), StatusCode::OK, "the warm-up request");
    let held = pool.acquire().await.expect("the only replica");

    let (status, json, elapsed) = post("/v1/chat/completions").await;
    assert_eq!(
        status,
        StatusCode::GATEWAY_TIMEOUT,
        "the handler's deadline, not the admission layer's 408: {json}"
    );
    assert_eq!(json["error"]["code"], "request_timeout", "{json}");
    assert_eq!(json["error"]["phase"], "waiting_for_engine", "{json}");
    assert!(
        elapsed >= Duration::from_millis(HANDLER_START_DELAY_MS + TIMEOUT_MS - 20),
        "the handler's own deadline ran its full length after it began: {elapsed:?}"
    );

    // The other two generation routes carry the same deadline of their own.
    for path in ["/v1/chat/completions/extended", "/v1/completions"] {
        let (status, json, _) = post(path).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{path}: {json}");
        assert_eq!(json["error"]["code"], "request_timeout", "{path}: {json}");
        assert_eq!(
            json["error"]["phase"], "waiting_for_engine",
            "{path}: {json}"
        );
    }

    // A route with no deadline of its own is still bounded — later.
    let (status, json, elapsed) = post("/__test_no_deadline").await;
    assert_eq!(status, StatusCode::REQUEST_TIMEOUT, "the backstop: {json}");
    assert_eq!(json["error"]["type"], "timeout_error", "{json}");
    assert!(
        elapsed >= Duration::from_millis(TIMEOUT_MS + ADMISSION_TIMEOUT_GRACE_MS - 50),
        "the backstop waits out the grace: {elapsed:?}"
    );
    drop(held);
}

#[tokio::test]
async fn health_is_reachable_with_no_config() {
    let router = router_for(default_opts());
    let resp = router
        .oneshot(
            Request::get("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn admin_is_refused_by_default() {
    let router = router_for(default_opts());
    let resp = router
        .oneshot(
            Request::get("/admin/status")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

/// `/admin/status` and `/admin/config` report the served engine's
/// resolved variant and effective kernel tier through the CLI's own
/// hardened router (`harden_router`), attached via
/// `RouterOptions::with_engine_report` rather than a process-wide
/// static.
#[tokio::test]
async fn admin_reports_the_resolved_variant_and_kernel_tier_through_harden_router() {
    const TOKEN: &str = "cli-admin-report-token";
    let report_engine =
        InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    let report = oxibonsai_runtime::admin::EngineReport::from_engine(&report_engine);
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    let pool = EnginePool::new(vec![engine]);
    let pool_size = pool.size();
    let metrics = Arc::new(InferenceMetrics::new());
    // The lock guards only `RouterOptions::default()`'s own
    // `OXI_ADMIN_TOKEN` read (immediately overridden below by
    // `with_auth`); it is released here, before any `.await`, matching
    // `router_for_with_options`'s own convention. This test cannot use
    // `router_for_with_options` itself: that helper always overwrites
    // its `RouterOptions` with a locked admin auth, which would keep
    // every request in this test at 403.
    let router = {
        let _env = test_env::lock();
        let router_options = RouterOptions::default()
            .with_auth(AdminAuthConfig::with_admin_token(TOKEN))
            .with_prompt_start_token(151_644)
            .with_engine_report(report.clone());
        let base = create_router_full(pool, None, metrics, router_options);
        harden_router(base, pool_size, &default_opts(), "127.0.0.1")
    };

    for path in ["/admin/status", "/admin/config"] {
        let resp = router
            .clone()
            .oneshot(
                Request::get(path)
                    .header("x-admin-token", TOKEN)
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK, "{path}");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(json["engine"]["variant"], report.variant, "{path}: {json}");
        assert_eq!(
            json["engine"]["kernel_tier"], report.kernel_tier,
            "{path}: {json}"
        );
    }
}

#[tokio::test]
async fn bearer_auth_protects_inference_routes_when_configured() {
    let mut opts = default_opts();
    opts.bearer_token = Some("x".repeat(20));
    let router = router_for(opts);

    let unauth = router
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    let health = router
        .oneshot(
            Request::get("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(health.status(), StatusCode::OK, "/health must stay exempt");
}

/// cli-18: two consecutive unauthenticated requests must be
/// rejected identically. **Not discriminating on its own**: both
/// requests run sequentially through `oneshot`, so the sole permit
/// (if bearer auth were ever mounted inside the admission stack)
/// would already be released before the second request starts, and
/// this assertion would hold under BOTH layer orderings. Kept as a
/// cheap idempotency check; `unauthenticated_request_is_401_while_the_sole_permit_is_held`
/// below is the test that actually proves the layer ordering.
#[tokio::test]
async fn unauthenticated_request_never_reaches_admission() {
    let mut opts = default_opts();
    opts.bearer_token = Some("x".repeat(20));
    opts.max_concurrent_requests = 1;
    let router = router_for(opts);

    for _ in 0..2 {
        let resp = router
            .clone()
            .oneshot(
                Request::get("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}

/// cli-18, mirroring the discriminating version in
/// `crates/oxibonsai-serve/src/hardening.rs`'s
/// `build_router_tests`: the sibling
/// `unauthenticated_request_never_reaches_admission` above drives
/// its two requests sequentially through `oneshot`, so the permit
/// is released before the second request starts and the assertion
/// holds under BOTH layer orderings -- it does not actually
/// discriminate. Here a slow *authenticated* chat completion is
/// kept in flight on a separate task, holding the sole concurrency
/// permit (effective ceiling = `resolve_admission_limit(1, 1)` =
/// `1`); an unauthenticated request issued while it runs must still
/// be `401`. If bearer auth were mounted INSIDE the admission
/// stack, `load_shed` would answer `503` instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unauthenticated_request_is_401_while_the_sole_permit_is_held() {
    let mut opts = default_opts();
    opts.bearer_token = Some("x".repeat(20));
    opts.max_concurrent_requests = 1;
    let router = router_for(opts);

    let slow_router = router.clone();
    let slow = tokio::spawn(async move {
        let payload = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 500,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", "x".repeat(20)))
            .body(Body::from(payload.to_string()))
            .expect("request");
        slow_router.oneshot(req).await.map(|r| r.status())
    });

    tokio::time::sleep(Duration::from_millis(30)).await;
    let resp = router
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let slow_status = slow.await.expect("join").expect("slow response");
    assert_eq!(
        slow_status,
        StatusCode::OK,
        "the authenticated request must have been the one holding the permit"
    );
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "an unauthenticated request must be 401 even while the sole admission permit \
         is held (a 503 means bearer auth sits INSIDE the admission stack)"
    );
}

/// sec-20/perf-M1: the `503` an overloaded admission layer returns
/// must carry `Retry-After` -- previously an omission (only the
/// `429` rate-limit path below set it). Same permit-holding trick
/// as `unauthenticated_request_is_401_while_the_sole_permit_is_held`,
/// but with no bearer token configured, so the second request
/// actually reaches (and is shed by) the admission layer instead of
/// being rejected by auth first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overloaded_request_gets_503_with_retry_after() {
    let mut opts = default_opts();
    opts.max_concurrent_requests = 1;
    let router = router_for(opts);

    let slow_router = router.clone();
    let slow = tokio::spawn(async move {
        let payload = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 500,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .expect("request");
        slow_router.oneshot(req).await.map(|r| r.status())
    });

    tokio::time::sleep(Duration::from_millis(30)).await;
    let resp = router
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let slow_status = slow.await.expect("join").expect("slow response");
    assert_eq!(
        slow_status,
        StatusCode::OK,
        "the in-flight request must have been the one holding the sole permit"
    );
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        resp.headers().get("retry-after").is_some(),
        "a 503 from the admission layer must carry Retry-After"
    );
}

#[tokio::test]
async fn rate_limit_returns_429_with_retry_after() {
    let mut opts = default_opts();
    opts.rate_limit_rpm = Some(60.0);
    opts.rate_limit_burst = 1.0;
    let router = router_for(opts);

    let first = router
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(first.status(), StatusCode::OK);

    let second = router
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(second.headers().get("retry-after").is_some());
}

#[tokio::test]
async fn cors_preflight_bypasses_bearer_auth() {
    let mut opts = default_opts();
    opts.bearer_token = Some("x".repeat(20));
    opts.cors_origins = vec!["https://app.example.com".to_string()];
    let router = router_for(opts);

    let preflight = Request::builder()
        .method("OPTIONS")
        .uri("/v1/chat/completions")
        .header("origin", "https://app.example.com")
        .body(Body::empty())
        .expect("preflight request");
    let resp = router.oneshot(preflight).await.expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "an OPTIONS preflight must bypass bearer auth entirely (SV-06)"
    );
}

#[tokio::test]
async fn no_cors_configured_emits_no_cors_headers() {
    let router = router_for(default_opts());
    let resp = router
        .oneshot(
            Request::get("/health")
                .header("origin", "https://anything.example.com")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert!(resp.headers().get("access-control-allow-origin").is_none());
}

#[tokio::test]
async fn oversized_body_is_rejected_with_413() {
    let mut opts = default_opts();
    opts.max_body_bytes = 16;
    let router = router_for(opts);

    let body = Body::from(vec![b'a'; 4096]);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(body)
        .expect("request");
    let resp = router.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

/// Ported from the sibling test added to
/// `crates/oxibonsai-serve/src/hardening.rs`'s `build_router_tests`:
/// `DefaultBodyLimit` was reordered from inside the admission stack
/// to outside it (still inside rate-limiting). **Not discriminating
/// on its own** (same caveat as
/// `unauthenticated_request_never_reaches_admission` above for a
/// different layer pair): `axum::extract::DefaultBodyLimit` only
/// inserts a request extension consulted later by the handler's own
/// extractor and never rejects anything itself at either position,
/// so this only proves the ceiling still applies once
/// `max_concurrent_requests` is tightened to its minimum (1), and
/// that the response is genuinely `413`, not a `408`/`504` timeout
/// from admission's own `.timeout(..)`.
#[tokio::test]
async fn oversized_body_is_rejected_with_413_even_with_a_single_permit() {
    let mut opts = default_opts();
    opts.max_body_bytes = 16;
    opts.max_concurrent_requests = 1;
    let router = router_for(opts);

    let body = Body::from(vec![b'a'; 4096]);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(body)
        .expect("request");
    let resp = router.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

/// The mirrored `Content-Length` precheck refuses
/// a declared oversize with 413 WHILE the sole admission permit is held
/// by a slow request — i.e. before admission. Without the guard the
/// oversized request would queue behind the permit (load-shed 503).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn declared_oversize_is_413_while_the_sole_permit_is_held() {
    let mut opts = default_opts();
    opts.max_body_bytes = 256;
    opts.max_concurrent_requests = 1;
    let router = router_for(opts);

    let slow_router = router.clone();
    let slow = tokio::spawn(async move {
        let payload = serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 500,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .expect("request");
        slow_router.oneshot(req).await.map(|r| r.status())
    });
    tokio::time::sleep(Duration::from_millis(30)).await;

    let oversized = vec![b'a'; 4096];
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("content-length", oversized.len())
        .body(Body::from(oversized))
        .expect("request");
    let resp = router.oneshot(req).await.expect("response");
    let slow_status = slow.await.expect("join").expect("slow response");
    assert_eq!(slow_status, StatusCode::OK);
    assert_eq!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "a declared oversize must be refused before admission (not shed with 503)"
    );
}
