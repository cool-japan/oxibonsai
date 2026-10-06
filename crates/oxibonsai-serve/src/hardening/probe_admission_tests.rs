//! The probe and observability routes through the whole hardened router of
//! the standalone server while its concurrency budget is full.
//!
//! Each test saturates the budget with requests parked inside the stack (see
//! `admission::test_support::HeldGate`; the runtime's scripted engine hold is
//! test-only inside the runtime crate, so it is not available here) against
//! the real runtime router, with every optional surface mounted. Waiting is
//! on counters and bounded awaits, never on a sleep.
//!
//! A sibling of `build_router_tests.rs`, declared in `hardening.rs` via
//! `#[path]`, so `super` names that module.

use std::collections::BTreeSet;

use super::admission::test_support::{
    finish_parked, get_path, request, send, HeldGate, Reply, BOUND, HELD_PATH,
};
use super::*;
use axum::http::Request;
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EngineLease;
use oxibonsai_runtime::sampling::SamplingParams;

const TOKEN: &str = "probe-test-bearer-token";

/// A metrics alias the server also serves (`observability.metrics_path`).
const ALIAS: &str = "/ops/metrics";

fn tiny_pool() -> Arc<EnginePool> {
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    EnginePool::new(vec![engine])
}

/// A configuration with `max_concurrent_requests` slots and the metrics alias
/// set.
fn config(max_concurrent_requests: usize) -> ServerConfig {
    let mut config = ServerConfig::default();
    config.limits.max_concurrent_requests = max_concurrent_requests;
    config.observability.metrics_path = ALIAS.to_string();
    config
}

fn build_options(pool_size: usize) -> RouterBuildOptions {
    RouterBuildOptions::new(AdminAuthConfig::locked(), pool_size, true, None)
        .with_prompt_start_token(151_644)
}

/// The hardened router over a one-replica pool, plus the handles a test
/// needs: the pool (to occupy its replica) and the gate (to fill the budget).
struct Served {
    /// What the standalone server serves.
    router: Router,
    pool: Arc<EnginePool>,
    gate: HeldGate,
}

/// Build the real runtime router with the chat UI mounted plus the held
/// route, then harden it exactly as `build_served_router` does. With `warm`,
/// the model descriptor is resolved first, as the served router does before
/// it opens the listener.
async fn served(config: &ServerConfig, warm: bool) -> Served {
    let pool = tiny_pool();
    let gate = HeldGate::new();
    let base = build_base_router(
        Arc::clone(&pool),
        None,
        Arc::new(InferenceMetrics::new()),
        Arc::new(MetricsRegistry::new()),
        config,
        build_options(pool.size()),
    )
    .merge(gate.router());
    if warm {
        admission::warm_model_descriptor(&base).await;
    }
    let router = harden_router(base, config, pool.size());
    Served { router, pool, gate }
}

fn authorized(mut request: Request<Body>) -> Request<Body> {
    request.headers_mut().insert(
        axum::http::header::AUTHORIZATION,
        axum::http::HeaderValue::from_str(&format!("Bearer {TOKEN}")).expect("a header value"),
    );
    request
}

/// A request that declares a body far larger than `max_body_bytes` (only the
/// header is inspected, so none is sent).
fn declared_oversize(method: &str, path: &str) -> Request<Body> {
    let mut oversize = request(method, path);
    oversize.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_static("1000000"),
    );
    oversize
}

/// Occupy the pool's only replica, as a running generation does: the replica
/// goes back to the pool when the returned lease is dropped.
async fn lease_replica(pool: &Arc<EnginePool>) -> EngineLease {
    tokio::time::timeout(BOUND, pool.acquire())
        .await
        .expect("the replica is free")
        .expect("the pool leases it")
}

fn assert_overload_shed(reply: &Reply, what: &str) {
    assert!(
        reply.is_overload_shed(),
        "{what} must be shed with the overload 503 + Retry-After at the limit, got {} {}",
        reply.status,
        reply.text()
    );
}

// ── the exemption, end to end ───────────────────────────────────────────

/// The reported defect: with the concurrency budget full and the only replica
/// busy (the requests that fill the budget in production are generating on
/// it), `/health`, `/readyz`, `/metrics`, `/metrics/serve`, the configured
/// metrics alias and `/ui/health` still answer `200` (`/readyz` as
/// ready-but-saturated); `/v1/models` and every generation, embedding and
/// admin route answer the documented overload; and once the held requests
/// finish everything answers normally.
///
/// Before the exemption existed each of the probe assertions below got the
/// overload `503` instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_routes_answer_while_the_budget_is_full() {
    let served = served(&config(2), true).await;
    let router = &served.router;
    let parked = served.gate.park(router, 2).await;
    let lease = lease_replica(&served.pool).await;

    let health = get_path(router, "/health").await;
    assert_eq!(health.status, StatusCode::OK, "/health: {}", health.text());
    assert_eq!(health.text(), "ok");

    let ready = get_path(router, "/readyz").await;
    assert_eq!(ready.status, StatusCode::OK, "/readyz: {}", ready.text());
    let body = ready.json();
    assert_eq!(body["status"], "ready", "{body}");
    assert_eq!(body["engine_slot_available"], false, "{body}");

    for path in ["/metrics", ALIAS] {
        let metrics = get_path(router, path).await;
        assert_eq!(metrics.status, StatusCode::OK, "{path}: {}", metrics.text());
        assert!(
            metrics
                .headers
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/plain")),
            "{path}: Prometheus text exposition"
        );
    }

    let serve_metrics = get_path(router, "/metrics/serve").await;
    assert_eq!(
        serve_metrics.status,
        StatusCode::OK,
        "{}",
        serve_metrics.text()
    );
    assert!(
        serve_metrics
            .text()
            .contains("oxibonsai_serve_http_requests_total"),
        "the serve registry keeps counting at the limit: {}",
        serve_metrics.text()
    );

    let ui = get_path(router, "/ui/health").await;
    assert_eq!(ui.status, StatusCode::OK, "/ui/health: {}", ui.text());
    assert_eq!(ui.json()["status"], "ok");

    for (method, path) in [
        ("GET", "/v1/models"),
        ("GET", "/v1/models/x"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/completions/extended"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("GET", "/admin/status"),
        ("GET", "/ui"),
    ] {
        let reply = send(router, request(method, path)).await;
        assert_overload_shed(&reply, &format!("{method} {path}"));
    }

    drop(lease);
    served.gate.release();
    finish_parked(parked).await;
    for path in [
        "/v1/models",
        "/health",
        "/readyz",
        "/metrics",
        "/metrics/serve",
        ALIAS,
        "/ui/health",
    ] {
        let reply = get_path(router, path).await;
        assert_eq!(reply.status, StatusCode::OK, "{path} after release");
    }
}

/// A saturated server is ready: with the budget full and a replica still
/// idle, and then, as in production, with the only replica leased as well (the
/// generations that fill the budget run on it), `/readyz` answers `200 ready`
/// promptly, with `engine_slot_available: false` once no replica is idle, and
/// never the admission layer's `overloaded_error`, while `GET /v1/models` is
/// shed. A readiness probe that failed here would pull a merely busy server
/// out of rotation for the length of every generation.
///
/// Before the exemption existed both answers were the overload `503`, with no
/// `status` member; and a readiness verdict that followed the pool's idle
/// replicas answered `503 not_ready` in the second state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn readyz_at_the_limit_is_ready_but_saturated() {
    let served = served(&config(1), true).await;
    let router = &served.router;
    let parked = served.gate.park(router, 1).await;
    assert_overload_shed(&get_path(router, "/v1/models").await, "the budget is full");

    // The budget is full and a replica is idle.
    let free = get_path(router, "/readyz").await;
    assert_eq!(free.status, StatusCode::OK, "{}", free.text());
    let body = free.json();
    assert_eq!(body["status"], "ready", "{body}");
    assert_eq!(body["model_loaded"], true, "{body}");
    assert_eq!(body["engine_slot_available"], true, "{body}");

    // The budget is full and the only replica is busy: ready-but-saturated.
    let lease = lease_replica(&served.pool).await;
    assert_eq!(served.pool.idle_count(), 0, "no replica is idle");
    assert_overload_shed(&get_path(router, "/v1/models").await, "the budget is full");
    let busy = get_path(router, "/readyz").await;
    assert_eq!(busy.status, StatusCode::OK, "{}", busy.text());
    let body = busy.json();
    assert_eq!(body["status"], "ready", "{body}");
    assert_eq!(body["model_loaded"], true, "{body}");
    assert_eq!(body["engine_slot_available"], false, "{body}");
    assert!(
        body.get("error").is_none(),
        "the readiness answer, not an error envelope: {body}"
    );
    assert_eq!(
        get_path(router, "/health").await.status,
        StatusCode::OK,
        "liveness is independent of the pool"
    );
    assert_eq!(get_path(router, "/metrics").await.status, StatusCode::OK);

    drop(lease);
    let again = get_path(router, "/readyz").await;
    assert_eq!(again.status, StatusCode::OK, "{}", again.text());
    assert_eq!(again.json()["engine_slot_available"], true);

    served.gate.release();
    finish_parked(parked).await;
}

/// A probe that arrives before the model descriptor is cached, with every
/// replica busy, waits for a replica (the runtime resolves the descriptor by
/// leasing one). Probes are outside the concurrency budget but not outside
/// the timeout, so such a probe is cut off with the timeout's `408`.
/// `build_served_router`, which the binary serves, resolves the descriptor
/// first: the same probe is answered at once, as ready-but-saturated.
///
/// On the pre-change router the cold half fails (its `408` carried
/// `code: null`); `build_served_router` and its warm-up are new, so the warm
/// half has no pre-change counterpart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_served_router_resolves_the_descriptor_before_serving() {
    let mut cfg = config(4);
    cfg.limits.per_request_timeout_ms = 40;
    let build = |pool: &Arc<EnginePool>| {
        (
            Arc::clone(pool),
            None,
            Arc::new(InferenceMetrics::new()),
            Arc::new(MetricsRegistry::new()),
            build_options(pool.size()),
        )
    };

    // Without the warm-up (the composition tests use): cold, so `/readyz`
    // needs a replica to learn the model, and none is free.
    let cold_pool = tiny_pool();
    let (pool, tokenizer, metrics, serve_metrics, options) = build(&cold_pool);
    let cold_router = build_router(pool, tokenizer, metrics, serve_metrics, &cfg, options);
    let lease = tokio::time::timeout(BOUND, cold_pool.acquire())
        .await
        .expect("the replica is free")
        .expect("the pool leases it");
    let cold = get_path(&cold_router, "/readyz").await;
    assert_eq!(cold.status, StatusCode::REQUEST_TIMEOUT, "{}", cold.text());
    assert_eq!(cold.json()["error"]["code"], "request_timeout");
    drop(lease);

    // The served router: the descriptor is resolved while the replica is
    // free, so the same probe answers although the replica is busy.
    let warm_pool = tiny_pool();
    let (pool, tokenizer, metrics, serve_metrics, options) = build(&warm_pool);
    let warm_router =
        build_served_router(pool, tokenizer, metrics, serve_metrics, &cfg, options).await;
    let lease = tokio::time::timeout(BOUND, warm_pool.acquire())
        .await
        .expect("the replica is free")
        .expect("the pool leases it");
    let warm = get_path(&warm_router, "/readyz").await;
    assert_eq!(warm.status, StatusCode::OK, "{}", warm.text());
    let body = warm.json();
    assert_eq!(body["status"], "ready", "{body}");
    assert_eq!(body["model_loaded"], true, "{body}");
    assert_eq!(body["engine_slot_available"], false, "{body}");
    drop(lease);
}

/// Both timeouts of the served stack carry the stable `request_timeout`
/// code. `/v1/completions` has a deadline of its own: a request waiting for a
/// replica that never frees answers the handler's `504`, which names the
/// stage. A route with no deadline of its own (the held test route stands in
/// for one) is cut off by the admission backstop instead: a `408` in the same
/// error envelope, with the same code and no phase.
///
/// The backstop's `code` used to be `null`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_timeouts_of_the_served_stack_carry_the_stable_code() {
    let mut cfg = config(4);
    cfg.limits.per_request_timeout_ms = 40;
    let served = served(&cfg, true).await;
    let lease = tokio::time::timeout(BOUND, served.pool.acquire())
        .await
        .expect("the replica is free")
        .expect("the pool leases it");

    let completion = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"prompt":"hi","max_tokens":2}"#))
        .expect("a well-formed request");
    let reply = send(&served.router, completion).await;
    assert_eq!(
        reply.status,
        StatusCode::GATEWAY_TIMEOUT,
        "{}",
        reply.text()
    );
    let body = reply.json();
    assert_eq!(body["error"]["code"], "request_timeout", "{body}");
    assert_eq!(body["error"]["phase"], "waiting_for_engine", "{body}");
    drop(lease);

    // The held route never answers by itself, and has no deadline of its own.
    let reply = send(&served.router, request("GET", HELD_PATH)).await;
    assert_eq!(
        reply.status,
        StatusCode::REQUEST_TIMEOUT,
        "{}",
        reply.text()
    );
    let body = reply.json();
    assert_eq!(body["error"]["type"], "timeout_error", "{body}");
    assert_eq!(body["error"]["code"], "request_timeout", "{body}");
    assert!(
        body["error"].get("phase").is_none(),
        "the backstop cannot know the stage: {body}"
    );
    served.gate.release();
}

// ── what the exempt routes keep ─────────────────────────────────────────

/// The exempt routes honour the body limit and the bearer rules exactly as
/// before, with the budget full: `/health`, `/metrics`, `/metrics/serve` and
/// `/ui/health` need no token, `/readyz` and the metrics alias do (the alias
/// is not one of bearer auth's exemptions), an API route needs one and is
/// then shed, and a declared-oversize body is a `413` before it reaches any
/// of them.
///
/// Before the exemption existed every row that expects `200` or `413` after
/// the bearer check got the overload `503` instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exempt_routes_keep_their_bearer_and_body_limit_rules_at_the_limit() {
    let mut cfg = config(1);
    cfg.auth.bearer_token = Some(TOKEN.to_string());
    cfg.limits.max_body_bytes = 64;
    let served = served(&cfg, true).await;
    let router = &served.router;
    let parked = served.gate.park_as(router, 1, Some(TOKEN)).await;

    let ok = StatusCode::OK;
    let unauthorized = StatusCode::UNAUTHORIZED;
    let too_large = StatusCode::PAYLOAD_TOO_LARGE;
    let shed = StatusCode::SERVICE_UNAVAILABLE;
    // (method, path, oversize, status without a token, status with one)
    let table: [(&str, &str, bool, StatusCode, StatusCode); 15] = [
        ("GET", "/health", false, ok, ok),
        ("GET", "/metrics", false, ok, ok),
        ("GET", "/metrics/serve", false, ok, ok),
        ("GET", "/ui/health", false, ok, ok),
        ("GET", "/readyz", false, unauthorized, ok),
        ("GET", ALIAS, false, unauthorized, ok),
        ("GET", "/v1/models", false, unauthorized, shed),
        ("POST", "/health", true, too_large, too_large),
        ("POST", "/metrics", true, too_large, too_large),
        ("POST", "/metrics/serve", true, too_large, too_large),
        ("POST", "/ui/health", true, too_large, too_large),
        ("POST", "/readyz", true, unauthorized, too_large),
        ("POST", ALIAS, true, unauthorized, too_large),
        (
            "POST",
            "/v1/chat/completions",
            true,
            unauthorized,
            too_large,
        ),
        ("POST", "/v1/embeddings", true, unauthorized, too_large),
    ];
    for (method, path, oversize, without_token, with_token) in table {
        let build = || {
            if oversize {
                declared_oversize(method, path)
            } else {
                request(method, path)
            }
        };
        let anonymous = send(router, build()).await;
        assert_eq!(
            anonymous.status,
            without_token,
            "{method} {path} without a token: {}",
            anonymous.text()
        );
        let credentialed = send(router, authorized(build())).await;
        assert_eq!(
            credentialed.status,
            with_token,
            "{method} {path} with a token: {}",
            credentialed.text()
        );
        match with_token {
            StatusCode::PAYLOAD_TOO_LARGE => {
                assert_eq!(credentialed.json()["error"]["code"], "content_too_large");
            }
            StatusCode::SERVICE_UNAVAILABLE => assert_overload_shed(&credentialed, path),
            _ => {}
        }
        if without_token == StatusCode::UNAUTHORIZED {
            assert_eq!(anonymous.json()["error"]["type"], "auth_error");
        }
    }

    served.gate.release();
    finish_parked(parked).await;
}

/// CORS is still the outermost layer for the exempt routes: at the limit a
/// probe from an allowed origin is answered `200` with the CORS headers, and
/// a preflight is answered by the CORS layer without reaching anything.
///
/// Before the exemption existed the probe was shed with the overload `503`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cors_still_wraps_the_exempt_routes_at_the_limit() {
    const ORIGIN: &str = "https://app.example.com";
    let mut cfg = config(1);
    cfg.cors.allowed_origins = vec![ORIGIN.to_string()];
    let served = served(&cfg, true).await;
    let router = &served.router;
    let parked = served.gate.park(router, 1).await;

    let mut probe = request("GET", "/health");
    probe.headers_mut().insert(
        axum::http::header::ORIGIN,
        axum::http::HeaderValue::from_static(ORIGIN),
    );
    let reply = send(router, probe).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(
        reply
            .headers
            .get("access-control-allow-origin")
            .and_then(|value| value.to_str().ok()),
        Some(ORIGIN),
        "the CORS layer still decorates the probe's response"
    );

    let mut preflight = request("OPTIONS", "/health");
    preflight.headers_mut().insert(
        axum::http::header::ORIGIN,
        axum::http::HeaderValue::from_static(ORIGIN),
    );
    let reply = send(router, preflight).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());

    served.gate.release();
    finish_parked(parked).await;
}

/// The rate limiter is unchanged: it exempts `/health` and `/metrics` and
/// nothing else, so they answer at the limit however many probes arrive,
/// while `/readyz`, `/metrics/serve`, `/ui/health` and the metrics alias draw
/// on the same per-client bucket as the API routes (their `429` comes from
/// the limiter, which sits outside admission, not from the budget).
///
/// Before the exemption existed `/health` and `/metrics` were shed with the
/// overload `503` here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_rate_limiter_still_exempts_only_health_and_metrics() {
    let mut cfg = config(1);
    // Two tokens, refilled at one per 100 s: the parked request takes one and
    // the first `/readyz` the other, whatever the machine's speed.
    cfg.rate_limit.rpm = Some(0.6);
    cfg.rate_limit.burst = 2.0;
    let served = served(&cfg, true).await;
    let router = &served.router;
    let parked = served.gate.park(router, 1).await;

    for _ in 0..5 {
        for path in ["/health", "/metrics"] {
            let reply = get_path(router, path).await;
            assert_eq!(reply.status, StatusCode::OK, "{path} is never limited");
        }
    }
    assert_eq!(get_path(router, "/readyz").await.status, StatusCode::OK);
    for path in [
        "/readyz",
        "/metrics/serve",
        "/ui/health",
        ALIAS,
        "/v1/models",
    ] {
        let reply = get_path(router, path).await;
        assert_eq!(reply.status, StatusCode::TOO_MANY_REQUESTS, "{path}");
        assert_eq!(reply.json()["error"]["type"], "rate_limit_error");
    }
    assert_eq!(get_path(router, "/health").await.status, StatusCode::OK);

    served.gate.release();
    finish_parked(parked).await;
}

/// `observability.metrics_enabled = false` still turns the three metrics
/// routes into `404` at the limit: exempt from admission, they are answered
/// by the metrics gate, not by the overload and not by the metrics.
///
/// Before the exemption existed they were shed with the overload `503`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disabled_metrics_stay_disabled_at_the_limit() {
    let mut cfg = config(1);
    cfg.observability.metrics_enabled = false;
    let served = served(&cfg, true).await;
    let router = &served.router;
    let parked = served.gate.park(router, 1).await;

    for path in ["/metrics", "/metrics/serve", ALIAS] {
        let reply = get_path(router, path).await;
        assert_eq!(
            reply.status,
            StatusCode::NOT_FOUND,
            "{path}: {}",
            reply.text()
        );
        assert!(
            reply.text().contains("metrics are disabled"),
            "{path}: {}",
            reply.text()
        );
    }
    assert_eq!(get_path(router, "/health").await.status, StatusCode::OK);

    served.gate.release();
    finish_parked(parked).await;
}

// ── the route table ─────────────────────────────────────────────────────

/// Every path the router registers. axum has no route-listing API, so this
/// reads the route table out of the router's `Debug` output: the paths of its
/// first (non-fallback) table, each rendered as a quoted string.
fn registered_paths(router: &Router) -> BTreeSet<String> {
    let dump = format!("{router:?}");
    let routes = dump.split("fallback_router").next().unwrap_or_default();
    let mut paths = BTreeSet::new();
    let mut rest = routes;
    while let Some(open) = rest.find("\"/") {
        let after = &rest[open + 1..];
        let Some(len) = after.find('"') else { break };
        let path = &after[..len];
        if !path.contains("__private__") {
            paths.insert(path.to_string());
        }
        rest = &after[len + 1..];
    }
    paths
}

/// `registered_paths` depends on the shape of axum's `Debug` output, which is
/// not a stable interface. This pins it on a router whose routes are known (a
/// parameter, two methods on one path, a merged router, a layer on top), so a
/// change to that format fails here, by name, instead of as a confusing
/// failure of the route-table tests below.
#[test]
fn registered_paths_reads_the_routes_of_a_known_router() {
    let merged = Router::new().route("/merged/{id}", get(|| async { "merged" }));
    let router = Router::new()
        .route("/plain", get(|| async { "plain" }))
        .route("/both", get(|| async { "both" }).post(|| async { "both" }))
        .merge(merged)
        .layer(axum::middleware::from_fn(
            |request: Request<Body>, next: axum::middleware::Next| async move {
                next.run(request).await
            },
        ));
    let expected: BTreeSet<String> = ["/both", "/merged/{id}", "/plain"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(registered_paths(&router), expected);
}

/// `/v1/models/{model}` -> `/v1/models/x`.
fn concrete(path: &str) -> String {
    path.split('/')
        .map(|segment| {
            if segment.starts_with('{') && segment.ends_with('}') {
                "x"
            } else {
                segment
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// What is wrong with the route table, with the budget full (the caller has
/// parked a request on the gate), checked in both directions:
///
/// * a registered path that is **not** in `exempt` must be behind admission:
///   every method it serves is shed with the overload response (a method it
///   does not serve may answer `405`, which is not a verdict) and at least
///   one is shed, otherwise the route escaped admission;
/// * a registered path that **is** in `exempt` must not be shed: a `GET` that
///   gets the overload response means the exemption does not work.
async fn route_table_problems(router: &Router, exempt: &[String]) -> Vec<String> {
    let mut problems = Vec::new();
    for path in registered_paths(router) {
        let target = concrete(&path);
        if exempt.contains(&path) {
            let reply = send(router, request("GET", &target)).await;
            if reply.is_overload_shed() {
                problems.push(format!(
                    "{path}: exempt, but shed with the overload response"
                ));
            }
            continue;
        }
        let mut shed = 0;
        let mut answered = Vec::new();
        for method in ["GET", "POST"] {
            let reply = send(router, request(method, &target)).await;
            if reply.is_overload_shed() {
                shed += 1;
            } else if reply.status != StatusCode::METHOD_NOT_ALLOWED {
                answered.push(format!("{method} -> {}", reply.status));
            }
        }
        if shed == 0 || !answered.is_empty() {
            problems.push(format!(
                "{path}: not behind admission ({})",
                answered.join(", ")
            ));
        }
    }
    problems
}

/// A route is inside admission unless it is explicitly exempt. The test
/// enumerates every route of the full router (the chat UI and the metrics
/// alias included), pins that the table is read correctly, that each exempt
/// path really is a route (so the list cannot go stale), and then, with the
/// budget full, that every exempt route answers and every other route is shed
/// with the overload response. A new route that is neither named in
/// `admission::ADMISSION_EXEMPT_PATHS` (or the metrics alias) nor behind
/// admission fails here, and so does an exempt route that admission sheds.
///
/// On the pre-change router the exempt half fails: every probe route is shed
/// there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_route_is_exempt_or_behind_admission() {
    let cfg = config(1);
    let served = served(&cfg, true).await;
    let paths = registered_paths(&served.router);

    for expected in [
        "/v1/chat/completions",
        "/v1/chat/completions/extended",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/models",
        "/v1/models/{model}",
        "/health",
        "/readyz",
        "/metrics",
        "/metrics/serve",
        ALIAS,
        "/ui",
        "/ui/health",
        "/admin/status",
        "/admin/config",
        HELD_PATH,
    ] {
        assert!(
            paths.contains(expected),
            "the route table was not read correctly: {expected} is missing from {paths:?}"
        );
    }
    assert!(paths.len() >= 16, "{paths:?}");
    let exempt = admission::exempt_paths(&cfg.observability.metrics_path);
    for exempt in &exempt {
        assert!(
            paths.contains(exempt),
            "{exempt} is exempt but no such route exists: {paths:?}"
        );
    }

    let parked = served.gate.park(&served.router, 1).await;
    let problems = route_table_problems(&served.router, &exempt).await;
    assert!(
        problems.is_empty(),
        "the route table and admission disagree: {problems:?}"
    );
    served.gate.release();
    finish_parked(parked).await;
}

/// The route-table test has teeth in both directions. A route mounted on the
/// router *after* admission was applied (which is how a route would escape
/// it, since a layer wraps only the routes that exist when it is applied) is
/// reported as not behind admission; and a route named as exempt that
/// admission nevertheless sheds is reported too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_route_table_check_reports_escaped_and_shed_routes() {
    let cfg = config(1);
    let served = served(&cfg, true).await;
    let router = served
        .router
        .clone()
        .route("/__late", get(|| async { "late" }))
        .route("/__late_post", axum::routing::post(|| async { "late" }));
    let parked = served.gate.park(&router, 1).await;

    let exempt = admission::exempt_paths(&cfg.observability.metrics_path);
    let escaped = route_table_problems(&router, &exempt).await;
    assert_eq!(escaped.len(), 2, "{escaped:?}");
    assert!(
        escaped[0].starts_with("/__late: not behind admission"),
        "{escaped:?}"
    );
    assert!(
        escaped[1].starts_with("/__late_post: not behind admission"),
        "{escaped:?}"
    );

    // `/v1/models` is inside admission: naming it exempt is a mistake the
    // check must catch.
    let mut mistaken_exempt = exempt;
    mistaken_exempt.push("/v1/models".to_string());
    let mistaken = route_table_problems(&served.router, &mistaken_exempt).await;
    assert_eq!(mistaken.len(), 1, "{mistaken:?}");
    assert!(
        mistaken[0].starts_with("/v1/models: exempt, but shed"),
        "{mistaken:?}"
    );

    served.gate.release();
    finish_parked(parked).await;
}
