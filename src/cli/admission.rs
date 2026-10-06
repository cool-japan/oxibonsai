//! Optional bearer-auth + admission-control hardening for `oxibonsai serve`.
//!
//! The `oxibonsai serve` facade command (this binary) and the standalone
//! `oxibonsai-serve` binary both mount the same unauthenticated
//! `oxibonsai_runtime::server` router. `oxibonsai-serve` wraps it in a
//! constant-time bearer-auth layer plus a bounded-concurrency /
//! per-request-timeout admission stack; this module replicates that exact
//! building-block shape (same middleware behavior, same layer ordering) so
//! `oxibonsai serve` offers equivalent protection instead of shipping the
//! bare router directly to `axum::serve`.
//!
//! # Admission and the probe routes
//!
//! Admission is one shared concurrency budget plus a per-request timeout for
//! the whole HTTP surface: a request that finds the budget full is shed
//! immediately with `503` + `Retry-After`, and one that outlasts the timeout
//! is answered `408`. The routes in [`ADMISSION_EXEMPT_PATHS`] are the only
//! exception: they neither take nor wait for a budget slot, so a server that
//! is busy generating still answers its liveness probe and its metrics
//! scrape. Every other route, present or added later, is inside the budget by
//! default.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, Method, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use oxibonsai_runtime::server::ApiError;
use tokio::sync::Semaphore;
use tower::ServiceExt;

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

/// The routes answered **outside** the admission budget.
///
/// A route listed here neither takes a concurrency slot nor is shed when the
/// budget is full, so it keeps answering while every slot is busy. It is still
/// bound by the request timeout, and everything mounted outside admission
/// still applies to it exactly as to any other route: the body limit, the
/// rate limiter, bearer auth (with its own, separate exemptions in
/// [`bearer_auth`]) and CORS.
///
/// The list is deliberately short and explicit. A route that is not named
/// here is inside admission, so a route added later sheds with `503` +
/// `Retry-After` at the limit unless someone decides, here and in
/// `docs/DEPLOYMENT.md`, that it is a probe or observability route. Matching
/// is on the exact request path: no prefix, suffix, case or trailing-slash
/// tolerance.
///
/// `GET /v1/models` and every generation, tokenisation, embedding and
/// `/admin/*` route stay inside admission: answering `503` there is the
/// intended overload behaviour, not an outage.
pub const ADMISSION_EXEMPT_PATHS: &[&str] = &[
    // Liveness. An orchestrator restarts a process that fails it, so a server
    // that is merely busy must never fail it.
    "/health",
    // Readiness. A saturated server is ready-but-saturated: `/readyz` answers
    // `200` (with `engine_slot_available: false` once no replica is idle) and
    // `503 not_ready` only for a server that cannot serve at all, never the
    // admission layer's overload error.
    "/readyz",
    // The Prometheus scrape, which has to keep working exactly when the server
    // is under load.
    "/metrics",
    // The bundled chat UI's own liveness ping (mounted only with
    // `--enable-ui`); the UI page itself, `GET /ui`, is inside admission.
    "/ui/health",
];

/// The shared admission state: one concurrency budget and one timeout for
/// every route of a router, plus the paths that skip the budget.
///
/// Mounted once with [`Admission::apply`]; the single [`Semaphore`] behind an
/// [`Arc`] is what makes the budget shared across routes rather than one
/// budget per route (`Router::layer` instantiates a layer once per route, so
/// per-layer state would silently multiply the limit).
#[derive(Debug)]
pub struct Admission {
    permits: Arc<Semaphore>,
    timeout: Duration,
    exempt: &'static [&'static str],
}

impl Admission {
    /// A budget of `capacity` concurrent requests (clamped to what a
    /// semaphore can hold) with a per-request `timeout`; the routes named in
    /// `exempt` skip the budget.
    pub fn new(capacity: usize, timeout: Duration, exempt: &'static [&'static str]) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(capacity.min(Semaphore::MAX_PERMITS))),
            timeout,
            exempt,
        }
    }

    /// Whether `path` is answered outside the budget. Exact match only.
    pub fn is_exempt(&self, path: &str) -> bool {
        self.exempt.contains(&path)
    }

    /// Budget slots not currently held by a request.
    #[cfg(test)]
    pub fn free_slots(&self) -> usize {
        self.permits.available_permits()
    }

    /// Wrap every route of `router` (and its fallback) in this admission
    /// state.
    pub fn apply(self: Arc<Self>, router: Router) -> Router {
        router.layer(axum::middleware::from_fn_with_state(self, admission_mw))
    }
}

/// The admission middleware.
///
/// A non-exempt request takes a slot without waiting: with none free it is
/// shed on the spot (`503` + `Retry-After`, [`overloaded_response`]) rather
/// than queued behind the engine until its timeout fires. The slot is held
/// until the response is ready (for a streamed response, until its headers
/// are), which is also when the timeout stops counting. Every request,
/// exempt or not, is bounded by the timeout: one that outlasts it is dropped
/// and answered `408` ([`timeout_response`]).
async fn admission_mw(
    State(admission): State<Arc<Admission>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let _slot = if admission.is_exempt(req.uri().path()) {
        None
    } else {
        match Arc::clone(&admission.permits).try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => return overloaded_response(),
        }
    };
    match tokio::time::timeout(admission.timeout, next.run(req)).await {
        Ok(response) => response,
        Err(_elapsed) => timeout_response(),
    }
}

/// Wrap `router` in the bounded-concurrency + per-request-timeout admission
/// stack `oxibonsai-serve` applies: one shared budget across every route
/// (not one per route, see [`Admission`]) plus a timeout, with the routes in
/// [`ADMISSION_EXEMPT_PATHS`] answered outside the budget.
pub fn apply_admission(router: Router, max_concurrent_requests: usize, timeout_ms: u64) -> Router {
    Arc::new(Admission::new(
        max_concurrent_requests,
        Duration::from_millis(timeout_ms),
        ADMISSION_EXEMPT_PATHS,
    ))
    .apply(router)
}

/// Resolve the served model's descriptor now, while every engine replica is
/// idle.
///
/// The runtime resolves the descriptor lazily, on the first request that
/// needs it, by leasing a replica; `/readyz` and `GET /v1/models` read it. A
/// probe that arrived before the descriptor was cached, with every replica
/// busy, would wait for a replica. Sending one `GET /v1/models` through the
/// still-unhardened `router` before the listener opens caches it, so the
/// probe routes never need a lease to answer. A failure is logged and
/// otherwise ignored: the server still starts.
pub(crate) async fn warm_model_descriptor(router: &Router) {
    /// Generous: at start-up the replicas are idle, so this answers at once.
    const WARMUP_TIMEOUT: Duration = Duration::from_secs(30);
    let request = match Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .body(Body::empty())
    {
        Ok(request) => request,
        Err(error) => {
            tracing::warn!(%error, "could not build the model-descriptor warm-up request");
            return;
        }
    };
    match tokio::time::timeout(WARMUP_TIMEOUT, router.clone().oneshot(request)).await {
        Ok(Ok(response)) if response.status().is_success() => {}
        Ok(Ok(response)) => tracing::warn!(
            status = %response.status(),
            "the model-descriptor warm-up request was not answered with a success status"
        ),
        Ok(Err(never)) => match never {},
        Err(_elapsed) => tracing::warn!(
            timeout_secs = WARMUP_TIMEOUT.as_secs(),
            "the model-descriptor warm-up request timed out"
        ),
    }
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
/// `max_concurrent_requests == 0` builds a zero-slot budget that sheds
/// every request forever, and `request_timeout_ms == 0` fires the timeout
/// before any handler can complete.
///
/// Re-exports the single canonical `oxibonsai_runtime::serve_shared::
/// validate_serve_params` under this module's historical name (also used by
/// `oxibonsai_serve::validation::validate_serve_params`'s own re-export),
/// closing the "byte-for-byte duplicate implementation across two binaries"
/// half of findings `SV-30` / `sec-M3`. `pub` since `cmd_serve.rs::run`
/// calls it as `admission::validate_serve_args`.
pub use oxibonsai_runtime::serve_shared::validate_serve_params as validate_serve_args;

/// The `503` for a request that finds the budget full: an
/// `overloaded_error` with `Retry-After: 1`, so a client backs off instead of
/// queueing behind the engine.
fn overloaded_response() -> Response {
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
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    response
}

/// `error.code` of the admission layer's timeout response: the same stable
/// code the handler's own `504` deadline carries, so a client treats both
/// as "the request outlasted the server's budget".
pub const ADMISSION_TIMEOUT_CODE: &str = "request_timeout";

/// The `408` for a request that outlasted `--request-timeout-ms`.
///
/// This is the backstop for routes with no deadline of their own (the three
/// generation handlers answer their own, stage-naming `504` first). It uses
/// the API's one error envelope, `{message, type, param, code}`, with the
/// code [`ADMISSION_TIMEOUT_CODE`]; unlike the handler's `504` it names no
/// `phase`, because this layer cannot know where the request was caught.
fn timeout_response() -> Response {
    ApiError::new(
        StatusCode::REQUEST_TIMEOUT,
        "request exceeded the configured --request-timeout-ms budget",
    )
    .with_type("timeout_error")
    .with_code(ADMISSION_TIMEOUT_CODE)
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

    /// Deterministic by construction, not by a race between a fixed sleep
    /// and a fixed timeout. The handler awaits [`std::future::pending`],
    /// which by definition never completes, so ANY finite timeout must
    /// fire — there is no wall-clock value the handler could "win" a race
    /// against, on however loaded a machine.
    #[tokio::test]
    async fn apply_admission_times_out_slow_handlers() {
        let slow = Router::new().route(
            "/slow",
            get(|| async {
                std::future::pending::<()>().await;
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
    /// immediately by the admission layer rather than queued.
    ///
    /// Deterministic by construction, not by a fixed
    /// 20ms sleep racing a fixed 100ms handler (a loaded machine can starve
    /// the "wait 20ms" task past the handler's own 100ms, making the second
    /// request arrive AFTER the first already released its permit — a
    /// spurious 200 instead of the intended 503). A barrier instead: the
    /// in-flight request signals (`Notify`) the instant it is actually
    /// running INSIDE the handler (i.e. it already holds the sole permit,
    /// admission having already accepted it), then parks on a second
    /// `Notify` until released; the second request is sent ONLY after that
    /// signal, so it is structurally guaranteed to observe the permit held,
    /// on any machine at any load.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_admission_returns_503_with_retry_after_when_overloaded() {
        let entered = std::sync::Arc::new(tokio::sync::Notify::new());
        let release = std::sync::Arc::new(tokio::sync::Notify::new());
        let entered_for_handler = std::sync::Arc::clone(&entered);
        let release_for_handler = std::sync::Arc::clone(&release);

        let slow = Router::new().route(
            "/slow",
            get(move || {
                let entered = std::sync::Arc::clone(&entered_for_handler);
                let release = std::sync::Arc::clone(&release_for_handler);
                async move {
                    entered.notify_one();
                    release.notified().await;
                    "done"
                }
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

        // Waits until the in-flight request is genuinely INSIDE the handler
        // (so admission has already granted it the sole permit) before the
        // second request is even built — no sleep, no race window.
        entered.notified().await;
        let second_req = Request::builder()
            .method(Method::GET)
            .uri("/slow")
            .body(Body::empty())
            .expect("request builds");
        let second_resp = app.oneshot(second_req).await.expect("oneshot succeeds");

        // Now let the first request finish.
        release.notify_one();
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

/// Request-holding helpers shared with `cmd_serve`'s router tests.
#[cfg(test)]
pub(crate) mod test_support;

/// The probe-route exemption, the timeout response and the start-up warm-up,
/// tested against a toy router.
#[cfg(test)]
#[path = "admission_tests.rs"]
mod exemption_tests;
