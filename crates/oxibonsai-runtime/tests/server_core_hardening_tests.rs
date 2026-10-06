//! Regression tests for the server-core hardening bundle.
//!
//! Covers, on the real served router built by `create_router*`:
//!   * client-input validation → `400 Bad Request`
//!     (`max_tokens` ceiling, negative/out-of-range `temperature`, `top_p`, `n`),
//!   * the token-bucket rate limiter → `429 Too Many Requests`,
//!   * the chat web UI mounted at `GET /ui`,
//!   * the admin API mounted at `/admin/*` reporting the *real* running config,
//!   * `/admin/cache-stats` reporting an honest "not enabled" structure rather
//!     than fabricated all-zero counters.

#![cfg(feature = "server")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::middleware::MiddlewareConfig;
use oxibonsai_runtime::rate_limiter::RateLimitConfig;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{
    create_router_full, create_router_with_auth, create_router_with_options, AuthConfig,
    RouterOptions, MAX_OUTPUT_TOKENS,
};

/// Admin credential used by the `/admin/*` tests below (finding `sec-15`).
const ADMIN_TOKEN: &str = "test-admin-token";

/// A router whose admin surface is reachable with [`ADMIN_TOKEN`].
fn admin_router() -> axum::Router {
    create_router_with_auth(engine(), None, AuthConfig::with_admin_token(ADMIN_TOKEN))
}

fn engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

/// Qwen3's `<|im_start|>` id: the tokenizer-less routers of this suite serve
/// a `Qwen3Config::tiny_test()` engine (the Qwen3 vocabulary size) and run a
/// text prompt as this single token.
const QWEN3_IM_START: u32 = 151_644;

/// A tokenizer-less router over `engine`. Without a tokenizer a server needs
/// a configured prompt start token to accept a text prompt at all (it
/// answers `400 tokenizer_required` otherwise), and the answer's text is
/// empty — nothing to render it with — while `usage` still counts every
/// generated token.
fn tokenizerless_router(engine: InferenceEngine<'static>) -> axum::Router {
    oxibonsai_runtime::server::create_router_full(
        oxibonsai_runtime::engine_pool::EnginePool::new(vec![engine]),
        None,
        std::sync::Arc::new(oxibonsai_runtime::metrics::InferenceMetrics::new()),
        oxibonsai_runtime::server::RouterOptions::default().with_prompt_start_token(QWEN3_IM_START),
    )
}

fn router() -> axum::Router {
    tokenizerless_router(engine())
}

/// A router with the bundled chat UI explicitly opted into (`SV-26`'s
/// `RouterOptions::enable_ui`, default `false`). [`router()`] above (like
/// every `create_router*` convenience constructor) leaves it at that
/// default, so the two UI tests below need this instead --
/// [`ui_is_not_mounted_by_default`] is what pins the default itself.
fn ui_enabled_router() -> axum::Router {
    create_router_full(
        EnginePool::new(vec![engine()]),
        None,
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default()
            .with_enable_ui(true)
            .with_prompt_start_token(QWEN3_IM_START),
    )
}

fn chat_request(body: serde_json::Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request")
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("parse json")
}

// ── max_tokens ceiling (finding 1) ────────────────────────────────────────────

#[tokio::test]
async fn oversized_max_tokens_is_rejected_with_400() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 9_000_000_000_000u64,
        })))
        .await
        .expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "an unbounded max_tokens must be rejected, not allowed to drive a giant allocation"
    );
    let json = body_json(resp).await;
    assert_eq!(json["error"]["param"], "max_tokens");
}

#[tokio::test]
async fn max_tokens_just_over_ceiling_is_rejected() {
    // The boundary is enforced at exactly MAX_OUTPUT_TOKENS: one past it is a
    // 400 (validated up-front, before any generation runs).
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": MAX_OUTPUT_TOKENS + 1,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["param"], "max_tokens");
}

#[tokio::test]
async fn small_max_tokens_within_ceiling_is_accepted() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn zero_max_tokens_is_rejected_with_400() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── temperature validation (findings 88, 7) ───────────────────────────────────

#[tokio::test]
async fn negative_temperature_is_rejected_with_400() {
    // A negative temperature must be a 400 — NOT silently coerced to greedy
    // decoding by the sampler's `< 1e-6` branch.
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": -5.0,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["param"], "temperature");
}

#[tokio::test]
async fn temperature_above_two_is_rejected_with_400() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 3.5,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── top_p / n validation ──────────────────────────────────────────────────────

#[tokio::test]
async fn out_of_range_top_p_is_rejected_with_400() {
    for bad in [0.0f32, 1.5f32] {
        let resp = router()
            .oneshot(chat_request(serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "top_p": bad,
            })))
            .await
            .expect("response");
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "top_p {bad} should be rejected"
        );
        let json = body_json(resp).await;
        assert_eq!(json["error"]["param"], "top_p");
    }
}

#[tokio::test]
async fn valid_top_p_is_accepted() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "top_p": 0.5,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn multi_choice_n_is_rejected_with_400() {
    let resp = router()
        .oneshot(chat_request(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "n": 3,
        })))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["param"], "n");
}

// ── Rate limiter wiring (finding 54) ──────────────────────────────────────────

#[tokio::test]
async fn rate_limiter_returns_429_after_burst() {
    let mw = MiddlewareConfig::none().with_rate_limit(RateLimitConfig {
        rps: 1.0,
        burst: 2.0,
        ..Default::default()
    });
    let router = create_router_with_options(
        EnginePool::new(vec![engine()]),
        None,
        Arc::new(InferenceMetrics::new()),
        None,
        mw,
    );

    let make = || {
        Request::get("/v1/models")
            .header("x-forwarded-for", "9.9.9.9")
            .body(Body::empty())
            .expect("request")
    };

    // Burst of 2 is allowed…
    for i in 0..2 {
        let resp = router.clone().oneshot(make()).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK, "request {i} should pass");
    }
    // …the third from the same client is throttled.
    let throttled = router.clone().oneshot(make()).await.expect("response");
    assert_eq!(throttled.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        throttled.headers().get("retry-after").is_some(),
        "a 429 should carry a Retry-After header"
    );
}

#[tokio::test]
async fn health_is_exempt_from_rate_limiting() {
    let mw = MiddlewareConfig::none().with_rate_limit(RateLimitConfig {
        rps: 1.0,
        burst: 1.0,
        ..Default::default()
    });
    let router = create_router_with_options(
        EnginePool::new(vec![engine()]),
        None,
        Arc::new(InferenceMetrics::new()),
        None,
        mw,
    );
    // Many health checks in a row never trip the limiter.
    for _ in 0..5 {
        let resp = router
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).expect("req"))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

// ── Web UI mount (finding 68) ─────────────────────────────────────────────────

#[tokio::test]
async fn chat_ui_is_served_at_ui() {
    let resp = ui_enabled_router()
        .oneshot(Request::get("/ui").body(Body::empty()).expect("req"))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.contains("text/html"),
        "UI should be served as HTML; got {content_type}"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body");
    let html = String::from_utf8_lossy(&bytes);
    assert!(
        html.contains("<!DOCTYPE html>"),
        "UI body should be an HTML page"
    );
}

#[tokio::test]
async fn ui_health_is_served() {
    let resp = ui_enabled_router()
        .oneshot(Request::get("/ui/health").body(Body::empty()).expect("req"))
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
}

/// SV-26: the chat UI is opt-in, so the shared
/// [`router()`] helper -- built the same way every convenience constructor
/// builds a router, `RouterOptions::default()` -- must get a `404` at `/ui`,
/// not the `200` the two tests above intentionally opt into via
/// [`ui_enabled_router()`].
#[tokio::test]
async fn ui_is_not_mounted_by_default() {
    let resp = router()
        .oneshot(Request::get("/ui").body(Body::empty()).expect("req"))
        .await
        .expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "the bundled chat UI must not be reachable unless --enable-ui opted in"
    );
}

// ── Admin API mounted + authenticated + real config (findings 52, 90, sec-15) ─

/// sec-15: the admin surface is gated unconditionally. A router built by the
/// convenience constructor with no admin credential configured refuses every
/// `/admin/*` request instead of publishing the running configuration.
///
/// This used to call
/// `router()`, which builds its `AuthConfig` via `AuthConfig::from_env()` —
/// so an operator/CI environment with `OXI_ADMIN_TOKEN` exported flips the
/// expected 403 (`admin_auth_not_configured`) to 401 (a real, but
/// differently-shaped, auth failure) and reds this test for a reason that
/// has nothing to do with what it is meant to guard. Building the router
/// explicitly with `AuthConfig::locked()` makes the "no credential
/// configured" condition hermetic: true regardless of the process
/// environment, exactly as this test's own name promises.
#[tokio::test]
async fn admin_is_refused_without_a_configured_credential() {
    let resp = create_router_with_auth(engine(), None, AuthConfig::locked())
        .oneshot(
            Request::get("/admin/config")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let json = body_json(resp).await;
    assert_eq!(json["error"]["code"], "admin_auth_not_configured");
}

#[tokio::test]
async fn admin_config_is_mounted_and_reports_real_model() {
    let resp = admin_router()
        .oneshot(
            Request::get("/admin/config")
                .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "/admin/config must be mounted on the served router"
    );
    let json = body_json(resp).await;
    assert!(json["server_version"].is_string());
    assert_eq!(
        json["max_tokens_default"], 256,
        "should report the runtime's real default"
    );
    // The real loaded model is surfaced (tiny_test → "Bonsai-Tiny-Test").
    assert_eq!(
        json["model"]["id"], "Bonsai-Tiny-Test",
        "config should reflect the actually-loaded model; got {json}"
    );
    assert_eq!(json["model"]["architecture"], "qwen3");
}

/// `/admin/config` keeps reporting the model's own name as `model.id` and shows
/// the id the model is served under beside it (`model.served_id`).
#[tokio::test]
async fn admin_config_reports_the_served_id_beside_the_models_own_name() {
    let router = create_router_full(
        EnginePool::new(vec![engine()]),
        None,
        Arc::new(InferenceMetrics::new()),
        RouterOptions::default()
            .with_auth(AuthConfig::with_admin_token(ADMIN_TOKEN))
            .with_served_model_id("Ternary-Bonsai-2-27B-PQ2_0"),
    );
    let resp = router
        .oneshot(
            Request::get("/admin/config")
                .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp).await;
    assert_eq!(json["model"]["id"], "Bonsai-Tiny-Test", "{json}");
    assert_eq!(
        json["model"]["served_id"], "Ternary-Bonsai-2-27B-PQ2_0",
        "{json}"
    );
    assert!(
        json["model"]["declared_context_length"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "{json}"
    );
}

#[tokio::test]
async fn admin_cache_stats_is_honest_about_unwired_caches() {
    let resp = admin_router()
        .oneshot(
            Request::get("/admin/cache-stats")
                .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp).await;
    // No prefix cache is wired into the base server, so it must be reported
    // as null / not-enabled — NOT as a fabricated all-zero counter presented
    // as real data.
    assert!(
        json["prefix_cache"].is_null(),
        "prefix_cache must be null when not wired; got {json}"
    );
    assert_eq!(json["prefix_cache_enabled"], false);
    // SV-19: `create_router_full` now always
    // attaches a real `KvCachePolicy` (fed from real per-request context
    // pressure — see `chat.rs`), so the *served* router's `/admin/cache-stats`
    // is genuinely populated, not fabricated. The null case is exercised
    // separately, against a bare `AdminState` with no policy attached at
    // all, by `admin_cache_stats_reports_null_kv_cache_without_a_policy`
    // below.
    assert!(
        json["kv_cache"].is_object(),
        "kv_cache must be a populated object on the served router (a real \
         KvCachePolicy is always attached); got {json}"
    );
    assert_eq!(json["kv_cache_enabled"], true);
    assert!(
        json["kv_cache"]["level"].is_string(),
        "kv_cache must report a real tier level; got {json}"
    );
    // The old fabrication returned a blanket `"status":"ok"` alongside fake
    // zeros; that must still be gone.
    assert!(
        json.get("status").is_none(),
        "cache-stats must not claim a blanket status; got {json}"
    );
    assert!(
        json.get("capacity_blocks").is_none(),
        "no fabricated capacity_blocks should be present; got {json}"
    );
}

/// Companion to the test above (SV-19): a bare
/// [`oxibonsai_runtime::admin::AdminState`] with no `KvCachePolicy` attached
/// -- the shape every `/admin/*` test predating `create_router_full`'s
/// unconditional `with_kv_cache_policy` wiring used -- must still report
/// `kv_cache: null`. The served router's populated object above comes from
/// `create_router_full` always attaching a policy, not from
/// `/admin/cache-stats` fabricating one when none exists.
#[tokio::test]
async fn admin_cache_stats_reports_null_kv_cache_without_a_policy() {
    let metrics = Arc::new(InferenceMetrics::new());
    let state = Arc::new(oxibonsai_runtime::admin::AdminState::new(metrics));
    let bare_admin_router =
        oxibonsai_runtime::admin::create_admin_router(Arc::clone(&state)).with_state(state);

    let resp = bare_admin_router
        .oneshot(
            Request::get("/admin/cache-stats")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp).await;
    assert!(
        json["kv_cache"].is_null(),
        "kv_cache must be null when no policy is attached; got {json}"
    );
    assert_eq!(json["kv_cache_enabled"], false);
}
