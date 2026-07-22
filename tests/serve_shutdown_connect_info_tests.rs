//! Regression tests proving the root `oxibonsai serve` CLI path
//! (`Commands::Serve` in `src/main.rs`) has serving-infrastructure parity
//! with the standalone `oxibonsai-serve` binary (`crates/oxibonsai-serve`).
//!
//! Before this change, the `Commands::Serve` arm called a bare
//! `axum::serve(listener, router).await?` directly:
//!
//! 1. No graceful shutdown — no `with_graceful_shutdown` / SIGTERM+Ctrl-C
//!    handling, unlike `oxibonsai-serve`'s `serve_with_shutdown` helper.
//! 2. No `into_make_service_with_connect_info::<SocketAddr>()` — so the
//!    wave-2 trusted-proxy rate limiter's `MaybePeerAddr` extractor
//!    (`oxibonsai_runtime::middleware`) never saw a real peer address on
//!    this binary and every direct client fell back to a single shared
//!    `"unknown"` rate-limit bucket.
//!
//! `Commands::Serve` now calls the exact same public
//! `oxibonsai_runtime::server::serve_with_shutdown` helper the standalone
//! binary uses, which both binds the graceful-shutdown future and wires
//! connect-info. These tests build the router with the same public building
//! blocks `Commands::Serve` uses (`create_router_with_pool`, optionally
//! `apply_middleware` for the rate-limiter case) and drive it through that
//! same `serve_with_shutdown` entry point, mirroring the pattern already
//! used by `oxibonsai-serve`'s own
//! `tests/server_integration_tests.rs::graceful_shutdown_stops_server`.

#![cfg(feature = "server")]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::middleware::{apply_middleware, MiddlewareConfig};
use oxibonsai_runtime::rate_limiter::RateLimitConfig;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router_with_pool, serve_with_shutdown};
use tokio::sync::oneshot;

/// Build the same kind of router `Commands::Serve` mounts: a one-replica
/// pool around `Qwen3Config::tiny_test()` (no GGUF file needed) plus a
/// fresh `InferenceMetrics`, via the exact public `create_router_with_pool`
/// call the CLI arm uses.
fn tiny_pool_router() -> axum::Router {
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    let pool = EnginePool::new(vec![engine]);
    let metrics = Arc::new(InferenceMetrics::new());
    create_router_with_pool(pool, None, metrics)
}

fn client_with_timeout() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("reqwest client")
}

// ─── Graceful shutdown ─────────────────────────────────────────────────────

/// `serve_with_shutdown` (now called from `Commands::Serve`, mirroring
/// `oxibonsai-serve`'s own `run()`) must serve real requests and then exit
/// cleanly once the shutdown future completes, instead of hanging or being
/// torn down mid-request by a bare `axum::serve(...).await?`.
#[tokio::test]
async fn serve_with_shutdown_serves_then_exits_on_signal() {
    let router = tiny_pool_router();

    // Reserve an ephemeral port, then release it immediately so
    // `serve_with_shutdown` (which owns its own `TcpListener::bind`) can
    // rebind it — same trick `oxibonsai-serve`'s
    // `graceful_shutdown_stops_server` test uses.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_future = async move {
        let _ = shutdown_rx.await;
    };

    let handle = tokio::spawn(async move {
        serve_with_shutdown(router, addr, shutdown_future)
            .await
            .expect("serve_with_shutdown should exit cleanly")
    });

    // Give the accept loop a moment to start.
    tokio::time::sleep(Duration::from_millis(80)).await;

    let client = client_with_timeout();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("pre-shutdown health request");
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "router must serve real requests once bound via serve_with_shutdown"
    );

    // Signal shutdown; the serving task must complete within a generous
    // bound instead of hanging (which a bare `axum::serve` with no
    // `with_graceful_shutdown` cannot even be asked to do).
    let _ = shutdown_tx.send(());
    let result = tokio::time::timeout(Duration::from_secs(5), handle).await;
    assert!(
        result.is_ok(),
        "serve_with_shutdown did not exit within the timeout after the shutdown signal"
    );
}

// ─── Connect-info wiring ────────────────────────────────────────────────────

/// Proves `serve_with_shutdown`'s `into_make_service_with_connect_info::
/// <SocketAddr>()` wiring actually reaches request handlers: with a
/// `trusted_proxies` rate-limit config that trusts the test client's real
/// loopback peer address, `X-Forwarded-For` is honored (per
/// `extract_client_id`'s documented priority order), so two requests
/// carrying *different* forwarded client IPs land in independent
/// rate-limit buckets and both succeed even though `burst == 1`.
///
/// Without connect-info wiring, `MaybePeerAddr` degrades to `None`
/// (`oxibonsai_runtime::middleware`'s documented fallback), the peer is
/// never recognized as a trusted proxy, `X-Forwarded-For` is ignored, and
/// every request collapses onto the single shared `"unknown"` bucket — see
/// `connect_info_missing_collapses_clients_into_one_bucket` below, which
/// exercises the pre-fix `axum::serve(listener, router)` shape directly to
/// document exactly the regression this test guards against.
#[tokio::test]
async fn connect_info_lets_trusted_proxy_forwarded_headers_distinguish_clients() {
    let loopback: std::net::IpAddr = "127.0.0.1".parse().expect("valid loopback address");
    let rate_limit = RateLimitConfig {
        rps: 0.0,
        burst: 1.0,
        trusted_proxies: vec![loopback],
        ..RateLimitConfig::default()
    };
    let router = apply_middleware(
        tiny_pool_router(),
        MiddlewareConfig::none().with_rate_limit(rate_limit),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_future = async move {
        let _ = shutdown_rx.await;
    };
    let handle = tokio::spawn(async move {
        serve_with_shutdown(router, addr, shutdown_future)
            .await
            .expect("serve_with_shutdown should exit cleanly")
    });
    tokio::time::sleep(Duration::from_millis(80)).await;

    let client = client_with_timeout();
    let models_url = format!("http://{addr}/v1/models");

    // First forwarded client: burst of 1 allows exactly one request.
    let resp_a1 = client
        .get(&models_url)
        .header("X-Forwarded-For", "10.1.1.1")
        .send()
        .await
        .expect("client-a first request");
    assert_eq!(
        resp_a1.status(),
        reqwest::StatusCode::OK,
        "first request from forwarded client A must be allowed"
    );

    // Second, *different* forwarded client: must get its own bucket (i.e.
    // must NOT be denied because of client A's consumed token), which is
    // only possible if the real (trusted) peer address was resolved so
    // `X-Forwarded-For` could be honored at all.
    let resp_b1 = client
        .get(&models_url)
        .header("X-Forwarded-For", "10.1.1.2")
        .send()
        .await
        .expect("client-b first request");
    assert_eq!(
        resp_b1.status(),
        reqwest::StatusCode::OK,
        "a different forwarded client must get an independent rate-limit bucket, \
         proving connect-info wiring resolved the real (trusted) peer address"
    );

    // Client A again: bucket already exhausted (burst=1, rps=0 ⇒ no
    // refill), so this must now be denied.
    let resp_a2 = client
        .get(&models_url)
        .header("X-Forwarded-For", "10.1.1.1")
        .send()
        .await
        .expect("client-a second request");
    assert_eq!(
        resp_a2.status(),
        reqwest::StatusCode::TOO_MANY_REQUESTS,
        "client A's second request must be denied once its own bucket is exhausted"
    );

    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
}

/// Documents the exact pre-fix regression: serving the same rate-limited
/// router via a bare `axum::serve(listener, router)` (no
/// `into_make_service_with_connect_info`) — the shape `Commands::Serve`
/// used before this change — makes `MaybePeerAddr` resolve to `None` for
/// every request, so trusted-proxy `X-Forwarded-For` handling never
/// triggers and *every* client (regardless of forwarded IP) collapses onto
/// the single shared `"unknown"` bucket. This is the opposite outcome of
/// `connect_info_lets_trusted_proxy_forwarded_headers_distinguish_clients`
/// above, confirming that test would have failed against the pre-fix code
/// path.
#[tokio::test]
async fn connect_info_missing_collapses_clients_into_one_bucket() {
    let loopback: std::net::IpAddr = "127.0.0.1".parse().expect("valid loopback address");
    let rate_limit = RateLimitConfig {
        rps: 0.0,
        burst: 1.0,
        trusted_proxies: vec![loopback],
        ..RateLimitConfig::default()
    };
    let router = apply_middleware(
        tiny_pool_router(),
        MiddlewareConfig::none().with_rate_limit(rate_limit),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr: SocketAddr = listener.local_addr().expect("local_addr");

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shutdown_future = async move {
        let _ = shutdown_rx.await;
    };
    let handle = tokio::spawn(async move {
        // Deliberately the pre-fix shape: no
        // `.into_make_service_with_connect_info::<SocketAddr>()`.
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_future)
            .await;
    });
    tokio::time::sleep(Duration::from_millis(80)).await;

    let client = client_with_timeout();
    let models_url = format!("http://{addr}/v1/models");

    let resp_a1 = client
        .get(&models_url)
        .header("X-Forwarded-For", "10.1.1.1")
        .send()
        .await
        .expect("client-a first request");
    assert_eq!(resp_a1.status(), reqwest::StatusCode::OK);

    // Without connect-info, this "different" forwarded client is
    // indistinguishable from client A — both fall back to `"unknown"` — so
    // it is denied even though it presented a different X-Forwarded-For.
    let resp_b1 = client
        .get(&models_url)
        .header("X-Forwarded-For", "10.1.1.2")
        .send()
        .await
        .expect("client-b first request");
    assert_eq!(
        resp_b1.status(),
        reqwest::StatusCode::TOO_MANY_REQUESTS,
        "without connect-info wiring, every client collapses onto the shared \
         \"unknown\" bucket regardless of X-Forwarded-For"
    );

    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
}
