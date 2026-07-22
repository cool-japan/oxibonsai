//! Optional bearer-auth + admission-control hardening for `oxibonsai serve`.
//!
//! The `oxibonsai serve` facade command (this binary) and the standalone
//! `oxibonsai-serve` binary both mount the same unauthenticated
//! `oxibonsai_runtime::server` router. `oxibonsai-serve` wraps it in a
//! constant-time bearer-auth layer plus a bounded-concurrency /
//! per-request-timeout admission stack; this module replicates that exact
//! building-block shape (same middleware behavior, same tower layer
//! ordering) so `oxibonsai serve` offers equivalent protection instead of
//! shipping the bare router directly to `axum::serve`.

use std::time::Duration;

use axum::body::Body;
use axum::error_handling::HandleErrorLayer;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{BoxError, Json, Router};
use tower::ServiceBuilder;

/// State shared by the bearer-auth middleware.
#[derive(Debug, Clone)]
pub struct BearerAuthState {
    /// The expected token. Any request that does not present exactly this
    /// token in `Authorization: Bearer <token>` is rejected with 401.
    pub token: String,
}

/// `axum::middleware::from_fn_with_state` handler enforcing bearer auth.
///
/// `/health` and `/metrics` are exempted so load balancers and Prometheus
/// scrapers keep working without a token.
pub async fn bearer_auth(
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

    if !constant_time_eq(presented.as_bytes(), state.token.as_bytes()) {
        return unauthorized("invalid bearer token").into_response();
    }

    next.run(req).await
}

/// Constant-time byte-string comparison (see `oxibonsai-serve`'s
/// `middleware::constant_time_eq` for the full rationale: plain `!=` leaks a
/// timing signal proportional to the correctly-guessed token prefix length).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
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

/// Wrap `router` in the same bounded-concurrency + per-request-timeout
/// admission stack `oxibonsai-serve` applies: a shared
/// `GlobalConcurrencyLimitLayer` (one semaphore across every route, not one
/// per route — see `oxibonsai-serve`'s `main.rs` comment on why a bare
/// `ConcurrencyLimitLayer` would be wrong under `Router::layer`) plus
/// `.timeout(..)`, both bridged back into proper HTTP responses via
/// `HandleErrorLayer` (axum requires an `Infallible` error type on the
/// outermost service).
pub fn apply_admission(router: Router, max_concurrent_requests: usize, timeout_ms: u64) -> Router {
    let concurrency_semaphore =
        tower::limit::GlobalConcurrencyLimitLayer::new(max_concurrent_requests);
    let admission = ServiceBuilder::new()
        .layer(HandleErrorLayer::new(handle_admission_error))
        .load_shed()
        .layer(concurrency_semaphore)
        .timeout(Duration::from_millis(timeout_ms));
    router.layer(admission)
}

/// Turn an admission-layer failure (overloaded `load_shed`, elapsed
/// `timeout`) into a proper HTTP response.
async fn handle_admission_error(err: BoxError) -> (StatusCode, Json<serde_json::Value>) {
    let (status, kind, message) = if err.is::<tower::load_shed::error::Overloaded>() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "server is at its configured --max-concurrent-requests capacity; retry after a \
             short backoff"
                .to_string(),
        )
    } else if err.is::<tower::timeout::error::Elapsed>() {
        (
            StatusCode::REQUEST_TIMEOUT,
            "timeout_error",
            "request exceeded the configured --request-timeout-ms budget".to_string(),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("unhandled admission-layer error: {err}"),
        )
    };
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": kind,
                "param": null,
                "code": null,
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Method;
    use axum::routing::get;
    use tower::ServiceExt;

    fn protected_router() -> Router {
        Router::new().route("/protected", get(|| async { "ok" }))
    }

    fn auth_router(token: &str) -> Router {
        let state = BearerAuthState {
            token: token.to_string(),
        };
        protected_router().layer(axum::middleware::from_fn_with_state(state, bearer_auth))
    }

    // ── constant_time_eq ─────────────────────────────────────────────────

    #[test]
    fn equal_bytes_are_equal() {
        assert!(constant_time_eq(b"my-secret-token", b"my-secret-token"));
    }

    #[test]
    fn different_bytes_are_unequal() {
        assert!(!constant_time_eq(b"my-secret-token", b"not-the-token!!"));
    }

    #[test]
    fn different_lengths_are_unequal() {
        assert!(!constant_time_eq(b"short", b"a-much-longer-token"));
    }

    // ── bearer_auth middleware ───────────────────────────────────────────

    #[tokio::test]
    async fn bearer_auth_rejects_missing_header() {
        let app = auth_router("secret-abc");
        let req = Request::builder()
            .method(Method::GET)
            .uri("/protected")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("oneshot succeeds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn bearer_auth_rejects_wrong_token() {
        let app = auth_router("secret-abc");
        let req = Request::builder()
            .method(Method::GET)
            .uri("/protected")
            .header(header::AUTHORIZATION, "Bearer wrong-token")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("oneshot succeeds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn bearer_auth_accepts_correct_token() {
        let app = auth_router("secret-abc");
        let req = Request::builder()
            .method(Method::GET)
            .uri("/protected")
            .header(header::AUTHORIZATION, "Bearer secret-abc")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("oneshot succeeds");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn bearer_auth_exempts_health_and_metrics() {
        for path in ["/health", "/metrics"] {
            let app = Router::new().route(path, get(|| async { "ok" })).layer(
                axum::middleware::from_fn_with_state(
                    BearerAuthState {
                        token: "secret-abc".to_string(),
                    },
                    bearer_auth,
                ),
            );
            let req = Request::builder()
                .method(Method::GET)
                .uri(path)
                .body(Body::empty())
                .expect("request builds");
            let resp = app.oneshot(req).await.expect("oneshot succeeds");
            assert_eq!(resp.status(), StatusCode::OK, "{path} must bypass auth");
        }
    }

    // ── apply_admission ──────────────────────────────────────────────────

    #[tokio::test]
    async fn apply_admission_lets_ordinary_requests_through() {
        let app = apply_admission(protected_router(), 8, 60_000);
        let req = Request::builder()
            .method(Method::GET)
            .uri("/protected")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("oneshot succeeds");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn apply_admission_times_out_slow_handlers() {
        let slow = Router::new().route(
            "/slow",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                "too slow"
            }),
        );
        let app = apply_admission(slow, 8, 5);
        let req = Request::builder()
            .method(Method::GET)
            .uri("/slow")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("oneshot succeeds");
        assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
    }
}
