//! Tests for the X-Request-ID propagation added in 0.1.4.
//!
//! Verifies that:
//! 1. A client-supplied `X-Request-ID` is echoed verbatim in the response.
//! 2. When no header is supplied, the server generates a fresh UUIDv4-style
//!    id and emits it in the response.
//! 3. Malformed ids fall back to a freshly generated id (server doesn't 4xx).
//! 4. The streaming path also emits the header on the SSE response.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::request_id::RequestId;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::server::{create_router, REQUEST_ID_HEADER};

fn test_router() -> axum::Router {
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    let engine = InferenceEngine::new(config, params, 42);
    create_router(engine, None)
}

fn chat_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 2,
        "temperature": 0.0,
        "stream": stream,
    })
}

#[tokio::test]
async fn supplied_request_id_is_echoed_in_non_streaming_response() {
    let app = test_router();
    let supplied = "11111111-2222-4333-8444-555555555555";
    let body = chat_body(false);
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(REQUEST_ID_HEADER, supplied)
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request");
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let echoed = resp
        .headers()
        .get(REQUEST_ID_HEADER)
        .expect("X-Request-ID header present")
        .to_str()
        .expect("ascii header");
    assert_eq!(echoed, supplied, "header must round-trip");
    // Drain the body so the test is well-formed.
    let _ = axum::body::to_bytes(resp.into_body(), usize::MAX).await;
}

#[tokio::test]
async fn auto_generated_request_id_when_header_absent() {
    let app = test_router();
    let body = chat_body(false);
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request");
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let header = resp
        .headers()
        .get(REQUEST_ID_HEADER)
        .expect("auto-generated X-Request-ID")
        .to_str()
        .expect("ascii header");
    // Must parse as a valid UUIDv4-style id.
    let parsed = RequestId::from_uuid(header);
    assert!(parsed.is_some(), "auto-generated id must be valid UUID");
}

#[tokio::test]
async fn malformed_request_id_falls_back_to_generated() {
    let app = test_router();
    let body = chat_body(false);
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(REQUEST_ID_HEADER, "not-a-valid-uuid")
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request");
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "malformed header should not 4xx"
    );
    let header = resp
        .headers()
        .get(REQUEST_ID_HEADER)
        .expect("response carries an id")
        .to_str()
        .expect("ascii header");
    // The server falls back to a generated id, so it's NOT the bogus input.
    assert_ne!(header, "not-a-valid-uuid");
    assert!(RequestId::from_uuid(header).is_some());
}

#[tokio::test]
async fn supplied_request_id_is_echoed_on_streaming() {
    let app = test_router();
    let supplied = "aaaaaaaa-bbbb-4ccc-9ddd-eeeeeeeeeeee";
    let body = chat_body(true); // SSE streaming mode
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(REQUEST_ID_HEADER, supplied)
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request");
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let echoed = resp
        .headers()
        .get(REQUEST_ID_HEADER)
        .expect("X-Request-ID echoed on SSE response")
        .to_str()
        .expect("ascii header");
    assert_eq!(echoed, supplied);
}

#[tokio::test]
async fn supplied_hex_form_request_id_is_accepted() {
    let app = test_router();
    let hex = "11111111222243338444555555555555"; // 32 hex chars (no dashes)
    let body = chat_body(false);
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(REQUEST_ID_HEADER, hex)
        .body(Body::from(serde_json::to_string(&body).expect("serialize")))
        .expect("request");
    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::OK);
    let header = resp
        .headers()
        .get(REQUEST_ID_HEADER)
        .expect("response has request id")
        .to_str()
        .expect("ascii");
    // Parsed-then-rendered always returns the canonical UUID form.
    assert_eq!(header.len(), 36);
    let parsed = RequestId::from_uuid(header).expect("valid uuid");
    let original = RequestId::from_hex(hex).expect("valid hex");
    assert_eq!(parsed, original);
}

#[test]
fn resolve_request_id_prefers_uuid_form() {
    use axum::http::HeaderMap;
    use oxibonsai_runtime::server::resolve_request_id;
    let mut h = HeaderMap::new();
    h.insert(
        REQUEST_ID_HEADER,
        "11111111-2222-4333-8444-555555555555".parse().unwrap(),
    );
    let id = resolve_request_id(&h);
    assert_eq!(id.as_uuid(), "11111111-2222-4333-8444-555555555555");
}

#[test]
fn resolve_request_id_generates_on_missing_header() {
    use axum::http::HeaderMap;
    use oxibonsai_runtime::server::resolve_request_id;
    let h = HeaderMap::new();
    let a = resolve_request_id(&h);
    let b = resolve_request_id(&h);
    // Both calls produce fresh ids — they should differ.
    assert_ne!(a, b);
}

#[test]
fn resolve_request_id_generates_on_malformed() {
    use axum::http::HeaderMap;
    use oxibonsai_runtime::server::resolve_request_id;
    let mut h = HeaderMap::new();
    h.insert(REQUEST_ID_HEADER, "garbage".parse().unwrap());
    let id = resolve_request_id(&h);
    // Successfully generated a valid v4-style id.
    let s = id.as_uuid();
    assert!(RequestId::from_uuid(&s).is_some());
}
