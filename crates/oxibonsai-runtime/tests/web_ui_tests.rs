//! Integration tests for the web UI module.
//!
//! Tests cover:
//! - Static properties of the embedded HTML
//! - Router construction
//! - HTTP endpoint behaviour via `axum::test` helpers

// The `web_ui` module is only compiled with the `server` feature, so this
// integration test must be gated the same way to keep `--no-default-features`
// green.
#![cfg(feature = "server")]

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use oxibonsai_runtime::web_ui::{create_ui_router, CHAT_UI_HTML};
use tower::ServiceExt; // for `oneshot`

// ─── Static HTML tests ────────────────────────────────────────────────────────

#[test]
fn ui_html_is_nonempty() {
    assert!(
        !CHAT_UI_HTML.is_empty(),
        "CHAT_UI_HTML constant must not be empty"
    );
}

#[test]
fn ui_html_contains_doctype() {
    assert!(
        CHAT_UI_HTML.contains("<!DOCTYPE html>"),
        "HTML must contain a DOCTYPE declaration"
    );
}

#[test]
fn ui_html_contains_fetch() {
    assert!(
        CHAT_UI_HTML.contains("fetch"),
        "HTML must use fetch() to call the API"
    );
}

// T-09: `ui_router_creates_ok` used to build a router and assert nothing
// beyond "did not panic". Deleted rather than given a contrived new
// assertion: `ui_health_endpoint_returns_200` and `ui_serve_html_returns_200`
// just below build this exact same router (`create_ui_router()`) and drive
// real requests through it, which is strictly stronger evidence that
// construction works.

// ─── Async HTTP endpoint tests ────────────────────────────────────────────────

#[tokio::test]
async fn ui_health_endpoint_returns_200() {
    let app = create_ui_router();
    let req = Request::builder()
        .method("GET")
        .uri("/ui/health")
        .body(Body::empty())
        .expect("failed to build request");

    let response = app
        .oneshot(req)
        .await
        .expect("handler should not return a transport error");

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "/ui/health must return 200"
    );
}

#[tokio::test]
async fn ui_serve_html_returns_200() {
    let app = create_ui_router();
    let req = Request::builder()
        .method("GET")
        .uri("/ui")
        .body(Body::empty())
        .expect("failed to build request");

    let response = app
        .oneshot(req)
        .await
        .expect("handler should not return a transport error");

    assert_eq!(response.status(), StatusCode::OK, "GET /ui must return 200");

    let content_type = response
        .headers()
        .get("content-type")
        .expect("/ui should set Content-Type")
        .to_str()
        .expect("Content-Type should be valid UTF-8");

    assert!(
        content_type.starts_with("text/html"),
        "Content-Type should be text/html, got: {}",
        content_type
    );
}
