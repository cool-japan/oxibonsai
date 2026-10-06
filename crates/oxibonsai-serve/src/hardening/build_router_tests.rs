//! Tests for [`super::build_router`], split out of `hardening.rs` to keep
//! that file under the workspace's 2000-line-per-file policy (declared
//! there via `#[path]`, so `super` still names that module).

use std::time::Duration;

use super::*;
use axum::http::Request;
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use tower::ServiceExt;

fn tiny_pool() -> Arc<EnginePool> {
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    EnginePool::new(vec![engine])
}

/// A router over the tokenizer-less tiny pool. Every text-prompting request
/// this module posts feeds token `151_644` (`RouterOptions::with_prompt_start_token`)
/// as its prompt, so it exercises admission/auth/rate-limit/body-limit
/// exactly as a real, tokenizer-carrying server would rather than failing
/// fast with `400 tokenizer_required` before those layers ever matter.
fn router_for(config: &ServerConfig) -> Router {
    let pool = tiny_pool();
    let pool_size = pool.size();
    build_router(
        pool,
        None,
        Arc::new(InferenceMetrics::new()),
        Arc::new(MetricsRegistry::new()),
        config,
        RouterBuildOptions::new(AdminAuthConfig::locked(), pool_size, false, None)
            .with_prompt_start_token(151_644),
    )
}

#[tokio::test]
async fn health_is_reachable_with_no_config() {
    let cfg = ServerConfig::default();
    let router = router_for(&cfg);
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

// ── active Content-Length precheck ──────────────────────────────────

#[tokio::test]
async fn content_length_guard_rejects_a_declared_oversized_body_with_413() {
    let mut cfg = ServerConfig::default();
    cfg.limits.max_body_bytes = 100;
    let router = router_for(&cfg);

    let resp = router
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("content-length", "1000")
                // The actual body bytes are irrelevant -- the guard
                // rejects on the *declared* Content-Length alone,
                // before the body is ever read.
                .body(Body::from(vec![b'x'; 10]))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "a declared Content-Length above the configured limit must be rejected \
         synchronously, before admission"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON envelope");
    assert_eq!(json["error"]["code"], "content_too_large");
}

#[tokio::test]
async fn content_length_guard_allows_a_request_within_the_limit() {
    let mut cfg = ServerConfig::default();
    cfg.limits.max_body_bytes = 1_000_000;
    let router = router_for(&cfg);

    let resp = router
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("content-length", "2")
                .body(Body::from(b"{}".to_vec()))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_ne!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "a request within the configured limit must not be rejected by this guard \
         (whatever else it gets rejected for downstream, e.g. a malformed body, is a \
         different concern)"
    );
}

#[tokio::test]
async fn content_length_guard_lets_through_a_request_with_no_content_length_header() {
    // A request with no declared Content-Length (e.g. genuine
    // Transfer-Encoding: chunked) must not be rejected by *this guard*.
    // `max_body_bytes` is deliberately generous here (unlike the two
    // tests above) so the body's *actual* size cannot itself trip
    // `DefaultBodyLimit`'s own separate, extension-based enforcement
    // once the handler reads it -- this test isolates "no header ->
    // this guard is a no-op", not the unrelated real-size check.
    let mut cfg = ServerConfig::default();
    cfg.limits.max_body_bytes = 1_000_000;
    let router = router_for(&cfg);

    let resp = router
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(b"{}".to_vec()))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_ne!(
        resp.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "no Content-Length header means this guard has nothing to check and must not block"
    );
}

#[tokio::test]
async fn admin_is_refused_by_default() {
    let cfg = ServerConfig::default();
    let router = router_for(&cfg);
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

/// `/admin/status` and `/admin/config` report the served engine's resolved
/// variant and effective kernel tier through the standalone binary's own
/// hardened router (`build_router`), attached via
/// `RouterBuildOptions::with_engine_report` rather than a process-wide
/// static.
#[tokio::test]
async fn admin_reports_the_resolved_variant_and_kernel_tier_through_build_router() {
    const TOKEN: &str = "serve-admin-report-token";
    let report_engine =
        InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    let report = oxibonsai_runtime::admin::EngineReport::from_engine(&report_engine);
    let cfg = ServerConfig::default();
    let pool = tiny_pool();
    let pool_size = pool.size();
    let router = build_router(
        pool,
        None,
        Arc::new(InferenceMetrics::new()),
        Arc::new(MetricsRegistry::new()),
        &cfg,
        RouterBuildOptions::new(
            AdminAuthConfig::with_admin_token(TOKEN),
            pool_size,
            false,
            None,
        )
        .with_prompt_start_token(151_644)
        .with_engine_report(report.clone()),
    );

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
    let mut cfg = ServerConfig::default();
    cfg.auth.bearer_token = Some("x".repeat(20));
    let router = router_for(&cfg);

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

/// cli-18: two unauthenticated requests in a row must be rejected
/// identically. **Not discriminating on its own**: both requests run
/// sequentially through `oneshot`, so the sole permit (if bearer auth
/// were ever mounted inside the admission stack) would already be
/// released before the second request starts, and this assertion would
/// hold under BOTH layer orderings. Kept as a cheap idempotency check;
/// the sibling `unauthenticated_request_is_401_while_the_sole_permit_is_held`
/// below is the test that actually proves the layer ordering.
#[tokio::test]
async fn unauthenticated_request_never_reaches_admission() {
    let mut cfg = ServerConfig::default();
    cfg.auth.bearer_token = Some("x".repeat(20));
    cfg.limits.max_concurrent_requests = 1;
    let router = router_for(&cfg);

    // Two unauthenticated requests in a row: if the first one consumed
    // the sole concurrency permit, a bug would make this observable via
    // a 503 instead of a consistent 401 -- but since bearer-auth sits
    // outside admission, both must be rejected identically.
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

/// cli-18: a genuinely discriminating version of the
/// sibling `unauthenticated_request_never_reaches_admission`, which
/// drives its two requests sequentially and therefore passes under BOTH
/// layer orderings. Here a slow *authenticated* chat completion is kept
/// in flight on a separate task, holding the sole concurrency permit
/// (effective ceiling = min(1, 1*4) = 1); an unauthenticated request
/// issued while it runs must still be `401`. If bearer auth were mounted
/// INSIDE the admission stack, `load_shed` would answer `503` instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unauthenticated_request_is_401_while_the_sole_permit_is_held() {
    let mut cfg = ServerConfig::default();
    cfg.auth.bearer_token = Some("x".repeat(20));
    cfg.limits.max_concurrent_requests = 1;
    let router = router_for(&cfg);

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

/// sec-20/perf-M1: the `503` an overloaded admission layer returns must
/// carry `Retry-After`, matching this module's own doc comment on
/// [`resolve_admission_limit`] ("shedding it immediately with a fast
/// `503` + `Retry-After`") -- previously an omission (only the `429`
/// rate-limit path below set it). Same permit-holding trick as
/// `unauthenticated_request_is_401_while_the_sole_permit_is_held`, but
/// with no bearer token configured, so the second request actually
/// reaches (and is shed by) the admission layer instead of being
/// rejected by auth first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overloaded_request_gets_503_with_retry_after() {
    let mut cfg = ServerConfig::default();
    cfg.limits.max_concurrent_requests = 1;
    let router = router_for(&cfg);

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
    let mut cfg = ServerConfig::default();
    cfg.rate_limit.rpm = Some(60.0);
    cfg.rate_limit.burst = 1.0;
    let router = router_for(&cfg);

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
    let mut cfg = ServerConfig::default();
    cfg.auth.bearer_token = Some("x".repeat(20));
    cfg.cors.allowed_origins = vec!["https://app.example.com".to_string()];
    let router = router_for(&cfg);

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
    let cfg = ServerConfig::default();
    let router = router_for(&cfg);
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
    let mut cfg = ServerConfig::default();
    cfg.limits.max_body_bytes = 16;
    let router = router_for(&cfg);

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

/// `DefaultBodyLimit` sits outside the admission stack
/// (`metrics_gate_mw -> admission -> DefaultBodyLimit`), still inside
/// rate-limiting, matching this module's own doc comment.
/// **Not discriminating on its own** (the same caveat
/// `unauthenticated_request_never_reaches_admission` above carries for a
/// different layer pair): `axum::extract::DefaultBodyLimit`'s `Layer`
/// implementation only inserts a request extension consulted later by
/// the `Bytes`/`Json` extractors deep inside the handler -- it never
/// rejects anything itself, at either position, and the admission layer
/// decides a genuine "is a permit available" race before either
/// position is reached.
/// This test only proves the ceiling still applies once
/// `max_concurrent_requests` is tightened to its minimum (1) -- i.e.
/// the reorder must not regress `oversized_body_is_rejected_with_413`
/// above -- and that the response is genuinely `413`, not a `408`/`504`
/// timeout from the admission layer's own `.timeout(..)`.
#[tokio::test]
async fn oversized_body_is_rejected_with_413_even_with_a_single_permit() {
    let mut cfg = ServerConfig::default();
    cfg.limits.max_body_bytes = 16;
    cfg.limits.max_concurrent_requests = 1;
    let router = router_for(&cfg);

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

#[tokio::test]
async fn serve_metrics_registry_is_mounted_and_counts_requests() {
    let cfg = ServerConfig::default();
    let router = router_for(&cfg);

    let _ = router
        .clone()
        .oneshot(
            Request::get("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    let resp = router
        .oneshot(
            Request::get("/metrics/serve")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let text = String::from_utf8_lossy(&body);
    assert!(
        text.contains("oxibonsai_serve_http_requests_total"),
        "the mounted registry must report the /health request it just counted: {text}"
    );
}

/// THE regression test for the blocking finding: five distinct,
/// never-registered paths must collapse into exactly ONE
/// `route="other"` counter series (aggregating all five hits), never
/// five independent series keyed by the raw client-chosen path.
/// Reproduced (pre-fix) as: `GET /attacker-path-0..4` each produced
/// their own `oxibonsai_serve_http_requests_total{route="/attacker-path-N",...}`
/// line plus a full ~12-line histogram, an unbounded-cardinality
/// remote memory-growth vector.
#[tokio::test]
async fn unmatched_paths_collapse_into_a_single_other_metrics_series() {
    let cfg = ServerConfig::default();
    let router = router_for(&cfg);

    for i in 0..5 {
        let resp = router
            .clone()
            .oneshot(
                Request::get(format!("/attacker-path-{i}"))
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        // Unmatched paths hit axum's default fallback: 404.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    let resp = router
        .oneshot(
            Request::get("/metrics/serve")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    let text = String::from_utf8_lossy(&body);

    for i in 0..5 {
        assert!(
            !text.contains(&format!("attacker-path-{i}")),
            "a raw client-chosen path must never appear as a metric label: {text}"
        );
    }

    let other_counter_lines: Vec<&str> = text
        .lines()
        .filter(|l| {
            l.starts_with("oxibonsai_serve_http_requests_total{") && l.contains("route=\"other\"")
        })
        .collect();
    assert_eq!(
        other_counter_lines.len(),
        1,
        "5 distinct unmatched paths must collapse into exactly one route=\"other\" \
         counter series, got: {other_counter_lines:?}\nfull body:\n{text}"
    );
    assert!(
        other_counter_lines[0].trim_end().ends_with(" 5"),
        "the single other-labelled series must have aggregated all 5 requests: \
         {other_counter_lines:?}"
    );
}

#[tokio::test]
async fn metrics_disabled_returns_404() {
    let mut cfg = ServerConfig::default();
    cfg.observability.metrics_enabled = false;
    let router = router_for(&cfg);
    let resp = router
        .oneshot(
            Request::get("/metrics")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// `/metrics/serve` must obey
/// `observability.metrics_enabled` too -- it previously stayed live (and
/// is separately exempt from `bearer_auth`, alongside `/health` and
/// `/metrics`, so Prometheus scrapers work unauthenticated), so an
/// operator who explicitly disabled metrics still exposed the full
/// serve registry unauthenticated.
#[tokio::test]
async fn metrics_disabled_also_blocks_the_serve_registry() {
    let mut cfg = ServerConfig::default();
    cfg.observability.metrics_enabled = false;
    let router = router_for(&cfg);
    let resp = router
        .oneshot(
            Request::get("/metrics/serve")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn custom_metrics_path_reaches_the_real_handler() {
    let mut cfg = ServerConfig::default();
    cfg.observability.metrics_path = "/custom-metrics".to_string();
    let router = router_for(&cfg);
    let resp = router
        .oneshot(
            Request::get("/custom-metrics")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

// ─── sec-20 / perf-M1: pool-size-derived admission limit ───────────────

#[test]
fn admission_limit_is_clamped_by_a_small_pool() {
    // The exact regression: default 32, single-replica GPU-tier pool.
    assert_eq!(resolve_admission_limit(32, 1), 4);
}

#[test]
fn admission_limit_respects_a_lower_explicit_configuration() {
    // An operator who deliberately tightened it below the heuristic is
    // never loosened back up.
    assert_eq!(resolve_admission_limit(2, 1), 2);
}

#[test]
fn admission_limit_is_unaffected_by_a_large_enough_pool() {
    assert_eq!(resolve_admission_limit(32, 8), 32);
}

#[test]
fn admission_limit_is_never_zero() {
    assert_eq!(resolve_admission_limit(0, 0), 1);
}

/// Direct, deterministic proof that `config.limits.max_input_tokens` /
/// `per_request_timeout_ms` are actually converted into the
/// `RequestLimits` `build_router` passes into `RouterOptions` -- the
/// specific wiring gap findings `SV-15`/`SV-16`/`SV-23` describe
/// (validation existed, the value never reached `AppState`).
#[test]
fn resolve_request_limits_derives_from_config() {
    let mut cfg = ServerConfig::default();
    cfg.limits.max_input_tokens = 777;
    cfg.limits.per_request_timeout_ms = 4321;
    let limits = resolve_request_limits(&cfg);
    assert_eq!(limits.max_input_tokens, Some(777));
    assert_eq!(
        limits.per_request_timeout,
        Some(Duration::from_millis(4321))
    );
}

/// End-to-end proof that `per_request_timeout_ms` actually bounds the
/// handler (not just that the value is threaded through, which the test
/// above already covers deterministically): the tokio-level timeout
/// wraps engine acquisition + tokenization + generation as a whole, so
/// an implausibly short 1ms budget must trip it even on the tiny
/// in-memory test model.
#[tokio::test]
async fn per_request_timeout_returns_408_for_a_slow_request() {
    let mut cfg = ServerConfig::default();
    cfg.limits.per_request_timeout_ms = 1;
    let router = router_for(&cfg);

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
    let resp = router.oneshot(req).await.expect("response");
    // The same `per_request_timeout_ms` value bounds two independent
    // layers that race on a slow request: the outer tower
    // admission-layer `.timeout(..)` (`handle_admission_error` maps its
    // `Elapsed` to `408`) and the inner `RequestLimits.per_request_timeout`
    // this test exists to prove is wired (`chat_completions` maps its
    // `tokio::time::timeout` to `ApiError::timeout()` = `504`). Either
    // one firing proves the 1ms budget took effect instead of the
    // request completing normally with `200`.
    assert!(
        matches!(
            resp.status(),
            StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT
        ),
        "expected a timeout status (408 or 504), got {}",
        resp.status()
    );
}

/// The id a launcher hands `RouterBuildOptions::with_served_model_id` is what
/// `GET /v1/models` lists, in place of the name the loaded model reports.
#[tokio::test]
async fn a_launcher_chosen_served_model_id_is_what_v1_models_lists() {
    let cfg = ServerConfig::default();
    let pool = tiny_pool();
    let pool_size = pool.size();
    let router = build_router(
        pool,
        None,
        Arc::new(InferenceMetrics::new()),
        Arc::new(MetricsRegistry::new()),
        &cfg,
        RouterBuildOptions::new(AdminAuthConfig::locked(), pool_size, false, None)
            .with_served_model_id("Ternary-Bonsai-2-27B-PQ2_0"),
    );
    let resp = router
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON");
    assert_eq!(
        json["data"][0]["id"], "Ternary-Bonsai-2-27B-PQ2_0",
        "{json}"
    );
}

/// Without a launcher-chosen id the model is listed under the name it reports.
#[tokio::test]
async fn without_a_served_model_id_v1_models_lists_the_loaded_models_name() {
    let cfg = ServerConfig::default();
    let router = router_for(&cfg);
    let resp = router
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON");
    assert_eq!(json["data"][0]["id"], "Bonsai-Tiny-Test", "{json}");
}
