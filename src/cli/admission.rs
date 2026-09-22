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
use axum::http::{header, HeaderName, HeaderValue, Method, Request, StatusCode};
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
/// `/health`, `/metrics` and `/ui/health` are exempted so load balancers,
/// Prometheus scrapers, and the chat web UI's own liveness probe keep
/// working without a token (finding `SV-06`'s correction (b)).
pub async fn bearer_auth(
    State(state): State<BearerAuthState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // SV-06 (correction): an `OPTIONS` preflight carries no credentials by
    // specification, so it must never be rejected on that basis — checked
    // *before* the `Authorization` lookup. When the caller also configures
    // CORS via `oxibonsai_runtime::middleware::apply_middleware` mounted
    // outside this layer, this is already unreachable (the outer `cors_mw`
    // short-circuits preflight first), but this keeps the same guarantee
    // even when CORS is not configured at all.
    if req.method() == Method::OPTIONS {
        return next.run(req).await;
    }

    let path = req.uri().path();
    if matches!(path, "/health" | "/metrics" | "/ui/health") {
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

/// Derive the real admission ceiling from the engine pool's actual size
/// (findings `sec-20` / `perf-M1`): measured serve throughput is strictly
/// serialized on the GPU tier (pool size clamped to 1), so admitting the
/// configured `--max-concurrent-requests` (default 32) queues every excess
/// request behind that one engine until the `--request-timeout-ms` budget
/// fires, instead of shedding it immediately with a fast `503` +
/// `Retry-After`.
///
/// The single canonical implementation now lives in
/// `oxibonsai_runtime::serve_shared` (findings `SV-30` / `sec-M3`); this
/// crate already depends on `oxibonsai-runtime` (see `cmd_serve.rs`'s own
/// imports), so re-exporting it here needs no new dependency. `pub` since
/// `cmd_serve.rs::harden_router` calls it as `admission::resolve_admission_limit`.
pub use oxibonsai_runtime::serve_shared::resolve_admission_limit;

/// Validate the three "serve hardening" parameters `oxibonsai serve` accepts
/// (findings `SV-30` / `sec-M3`): an unset-or-short bearer token is fine
/// (auth is optional), but a *configured* token must meet the minimum
/// length, and the two admission knobs must be positive —
/// `max_concurrent_requests == 0` builds a zero-permit semaphore that
/// `load_shed`s every request forever, and `request_timeout_ms == 0` fires
/// the timeout before any handler can complete.
///
/// Re-exports the single canonical `oxibonsai_runtime::serve_shared::
/// validate_serve_params` under this module's historical name (also used by
/// `oxibonsai_serve::validation::validate_serve_params`'s own re-export),
/// closing the "byte-for-byte duplicate implementation across two binaries"
/// half of findings `SV-30` / `sec-M3`. `pub` since `cmd_serve.rs::run`
/// calls it as `admission::validate_serve_args`.
pub use oxibonsai_runtime::serve_shared::validate_serve_params as validate_serve_args;

/// Turn an admission-layer failure (overloaded `load_shed`, elapsed
/// `timeout`) into a proper HTTP response.
///
/// sec-20/perf-M1: the `Overloaded` (`503`) branch carries a `Retry-After`
/// header, mirroring `crates/oxibonsai-serve/src/hardening.rs`'s
/// `handle_admission_error` -- previously an omission (only the `429`
/// rate-limit path, `oxibonsai_runtime::rate_limiter`, set it).
async fn handle_admission_error(err: BoxError) -> Response {
    if err.is::<tower::load_shed::error::Overloaded>() {
        let body = Json(serde_json::json!({
            "error": {
                "message": "server is at its configured --max-concurrent-requests capacity; \
                             retry after a short backoff",
                "type": "overloaded_error",
                "param": null,
                "code": null,
            }
        }));
        let mut response = (StatusCode::SERVICE_UNAVAILABLE, body).into_response();
        response.headers_mut().insert(
            HeaderName::from_static("retry-after"),
            HeaderValue::from_static("1"),
        );
        return response;
    }
    if err.is::<tower::timeout::error::Elapsed>() {
        return (
            StatusCode::REQUEST_TIMEOUT,
            Json(serde_json::json!({
                "error": {
                    "message": "request exceeded the configured --request-timeout-ms budget",
                    "type": "timeout_error",
                    "param": null,
                    "code": null,
                }
            })),
        )
            .into_response();
    }
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": {
                "message": format!("unhandled admission-layer error: {err}"),
                "type": "internal_error",
                "param": null,
                "code": null,
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Method;
    use axum::routing::get;
    // Test-only: no non-test call site references the constant directly by
    // name in this crate (`validate_serve_args` uses it internally, inside
    // `oxibonsai_runtime::serve_shared`), so it is imported here rather
    // than as a module-level re-export.
    use oxibonsai_runtime::serve_shared::MIN_BEARER_TOKEN_LEN;
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
        for path in ["/health", "/metrics", "/ui/health"] {
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

    /// SV-06 (correction): an `OPTIONS` preflight must never be rejected on
    /// the basis of a missing/invalid `Authorization` header, even for a
    /// path that would otherwise require it -- checked before the
    /// `Authorization` lookup so this holds even when no outer CORS layer
    /// is mounted at all. In production, CORS (when configured) is the
    /// outermost layer and short-circuits preflight with `200` before this
    /// code ever runs (see `crates/oxibonsai-serve/src/main.rs`'s
    /// `build_router_tests::cors_preflight_bypasses_bearer_auth`); here,
    /// with *no* CORS layer at all, axum's own method routing naturally
    /// answers `405` for a path with no registered `OPTIONS` handler --
    /// still not `401`, which is the actual invariant this test proves:
    /// bearer_auth's own OPTIONS check never lets a stale/missing
    /// `Authorization` header turn into an auth rejection.
    #[tokio::test]
    async fn bearer_auth_exempts_options_preflight_on_any_path() {
        let app = auth_router("secret-abc");
        let req = Request::builder()
            .method(Method::OPTIONS)
            .uri("/protected")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("oneshot succeeds");
        assert_ne!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "an OPTIONS preflight must never be rejected for missing/invalid auth"
        );
    }

    /// A normal (non-`OPTIONS`) request to the same path must still be
    /// rejected -- proving the exemption above is preflight-specific, not a
    /// blanket bypass.
    #[tokio::test]
    async fn bearer_auth_still_protects_get_on_the_same_path() {
        let app = auth_router("secret-abc");
        let req = Request::builder()
            .method(Method::GET)
            .uri("/protected")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("oneshot succeeds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── resolve_admission_limit (sec-20 / perf-M1) ──────────────────────────

    #[test]
    fn admission_limit_is_clamped_by_a_small_pool() {
        assert_eq!(resolve_admission_limit(32, 1), 4);
    }

    #[test]
    fn admission_limit_respects_a_lower_explicit_configuration() {
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

    // ── validate_serve_args (SV-30 / sec-M3) ────────────────────────────────
    //
    // `validate_serve_args` is now a re-export of the single canonical
    // `oxibonsai_runtime::serve_shared::validate_serve_params` (see this
    // module's doc comment on it), so its error text is the shared,
    // field-name-style wording rather than a CLI-flag-styled (`--foo`)
    // byte-for-byte copy -- these assertions match that shared wording
    // (the same substrings `oxibonsai_serve::validation`'s own
    // `validate_serve_params_rejects_*` tests check).

    #[test]
    fn validate_serve_args_accepts_sane_defaults() {
        assert!(validate_serve_args(None, 32, 60_000).is_ok());
        assert!(validate_serve_args(Some(&"x".repeat(MIN_BEARER_TOKEN_LEN)), 1, 1).is_ok());
    }

    #[test]
    fn validate_serve_args_rejects_short_token() {
        let err = validate_serve_args(Some("short"), 32, 60_000).expect_err("must reject");
        assert!(err.contains("bearer token"));
    }

    #[test]
    fn validate_serve_args_rejects_zero_concurrency() {
        let err = validate_serve_args(None, 0, 60_000).expect_err("must reject");
        assert!(err.contains("max_concurrent_requests"));
    }

    #[test]
    fn validate_serve_args_rejects_zero_timeout() {
        let err = validate_serve_args(None, 32, 0).expect_err("must reject");
        assert!(err.contains("request_timeout_ms"));
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

    /// sec-20/perf-M1: the `503` an overloaded admission layer returns must
    /// carry `Retry-After` -- previously an omission (only the `429`
    /// rate-limit path set it). A concurrency ceiling of 1 plus a slow
    /// in-flight request means a second concurrent request must be shed
    /// immediately by `load_shed` rather than queued.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_admission_returns_503_with_retry_after_when_overloaded() {
        let slow = Router::new().route(
            "/slow",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                "done"
            }),
        );
        let app = apply_admission(slow, 1, 60_000);

        let first_app = app.clone();
        let first = tokio::spawn(async move {
            let req = Request::builder()
                .method(Method::GET)
                .uri("/slow")
                .body(Body::empty())
                .expect("request builds");
            first_app.oneshot(req).await.map(|r| r.status())
        });

        tokio::time::sleep(Duration::from_millis(20)).await;
        let second_req = Request::builder()
            .method(Method::GET)
            .uri("/slow")
            .body(Body::empty())
            .expect("request builds");
        let second_resp = app.oneshot(second_req).await.expect("oneshot succeeds");

        let first_status = first.await.expect("join").expect("first response");
        assert_eq!(
            first_status,
            StatusCode::OK,
            "the in-flight request must have been the one holding the sole permit"
        );
        assert_eq!(second_resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            second_resp.headers().get("retry-after").is_some(),
            "a 503 from the admission layer must carry Retry-After"
        );
    }
}
