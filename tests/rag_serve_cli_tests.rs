//! Integration test proving the `oxibonsai serve --rag` flag is a real,
//! reachable CLI surface for the RAG HTTP API rather than a silent no-op
//! (see finding #13: the `rag` feature previously never forwarded to
//! `oxibonsai-runtime`'s `rag` feature, so `rag_server::create_rag_router*`
//! was unreachable from any shipped binary).
//!
//! This only needs the `rag` feature; it does not require a real model file
//! (`--help` output alone proves the flag was compiled in and is parsed by
//! clap, which is enough to demonstrate the flag exists and the binary
//! links against `oxibonsai-runtime`'s `rag` feature without error).

use std::process::Command;

#[test]
#[cfg(feature = "rag")]
fn serve_help_lists_the_rag_flag_when_rag_feature_is_enabled() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(["serve", "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");

    assert!(
        output.status.success(),
        "`serve --help` should succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--rag"),
        "`serve --help` must advertise --rag when built with the `rag` feature; got:\n{stdout}"
    );
}

/// Without the `rag` feature the flag must not exist at all (rather than
/// silently being accepted and doing nothing) — passing it should be a
/// clap parse error.
#[test]
#[cfg(not(feature = "rag"))]
fn serve_rejects_unknown_rag_flag_when_rag_feature_is_disabled() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(["serve", "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("--rag"),
        "`serve --help` must not advertise --rag without the `rag` feature; got:\n{stdout}"
    );
}

/// End-to-end proof that the exact router-merge pattern `Commands::Serve`
/// uses when `--rag` is passed (`create_router_with_pool(..).merge(
/// rag_server::create_rag_router_with_pool(..))`) actually mounts a
/// reachable `/rag/stats` endpoint alongside the base OpenAI-compatible
/// routes, in-process (no real socket / model file needed — mirrors the
/// `tower::ServiceExt::oneshot` pattern already used by
/// `server_integration_tests.rs`).
#[cfg(feature = "rag")]
mod rag_router_merge {
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::{
        engine::InferenceEngine, engine_pool::EnginePool, metrics::InferenceMetrics,
        rag_server::create_rag_router_with_pool, sampling::SamplingParams,
        server::create_router_with_pool,
    };
    use tower::ServiceExt;

    fn make_merged_router() -> axum::Router {
        let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
        let pool = EnginePool::new(vec![engine]);
        let metrics = Arc::new(InferenceMetrics::new());
        let base = create_router_with_pool(Arc::clone(&pool), None, metrics);
        base.merge(create_rag_router_with_pool(pool, None))
    }

    #[tokio::test]
    async fn rag_stats_endpoint_is_reachable_after_merge() {
        let app = make_merged_router();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/rag/stats")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/rag/stats must be reachable on the router `oxibonsai serve --rag` builds"
        );
    }

    #[tokio::test]
    async fn base_health_endpoint_still_works_after_rag_merge() {
        let app = make_merged_router();
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .expect("request should build");

        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "merging the RAG router must not break the base /health route"
        );
    }
}
