//! Admission control for the standalone server: one shared concurrency
//! budget plus a per-request timeout over the whole HTTP surface, with the
//! probe and observability routes answered outside the budget.
//!
//! A request that finds the budget full is shed immediately with `503` +
//! `Retry-After`, and one that outlasts the timeout is answered `408`. The
//! routes in [`ADMISSION_EXEMPT_PATHS`] (and the configured metrics alias,
//! see [`exempt_paths`]) are the only exception: they neither take nor wait
//! for a budget slot, so a server that is busy generating still answers its
//! liveness probe and its metrics scrape. Every other route, present or added
//! later, is inside the budget by default.
//!
//! `oxibonsai serve` (`src/cli/admission.rs` in the root package) mounts the
//! same layer; the two copies are kept in step by hand, since the root
//! package cannot depend on this crate.

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

/// The routes answered **outside** the admission budget.
///
/// A route listed here neither takes a concurrency slot nor is shed when the
/// budget is full, so it keeps answering while every slot is busy. It is still
/// bound by the request timeout, and everything mounted outside admission
/// still applies to it exactly as to any other route: the body limit, the
/// rate limiter, bearer auth (with its own, separate exemptions in
/// `main.rs`) and CORS.
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
    // This server's own request counters: the same scrape, from this crate's
    // registry.
    "/metrics/serve",
    // The bundled chat UI's own liveness ping (mounted only with
    // `--enable-ui`); the UI page itself, `GET /ui`, is inside admission.
    "/ui/health",
];

/// The paths to exempt for a server whose Prometheus output is also served at
/// `metrics_path` (`observability.metrics_path`): [`ADMISSION_EXEMPT_PATHS`]
/// plus that alias when it is not already one of them. The alias renders the
/// same scrape as `/metrics`, so it is exempt for the same reason.
pub fn exempt_paths(metrics_path: &str) -> Vec<String> {
    let mut paths: Vec<String> = ADMISSION_EXEMPT_PATHS
        .iter()
        .map(|path| (*path).to_string())
        .collect();
    if !paths.iter().any(|path| path == metrics_path) {
        paths.push(metrics_path.to_string());
    }
    paths
}

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
    exempt: Vec<String>,
}

impl Admission {
    /// A budget of `capacity` concurrent requests (clamped to what a
    /// semaphore can hold) with a per-request `timeout`; the routes named in
    /// `exempt` skip the budget.
    pub fn new(capacity: usize, timeout: Duration, exempt: Vec<String>) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(capacity.min(Semaphore::MAX_PERMITS))),
            timeout,
            exempt,
        }
    }

    /// Whether `path` is answered outside the budget. Exact match only.
    pub fn is_exempt(&self, path: &str) -> bool {
        self.exempt.iter().any(|exempt| exempt == path)
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
/// stack: one shared budget of `max_concurrent_requests` across every route
/// (not one per route, see [`Admission`]) and a timeout of `timeout_ms`, with
/// the routes in `exempt` answered outside the budget.
pub fn apply_admission(
    router: Router,
    max_concurrent_requests: usize,
    timeout_ms: u64,
    exempt: Vec<String>,
) -> Router {
    Arc::new(Admission::new(
        max_concurrent_requests,
        Duration::from_millis(timeout_ms),
        exempt,
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
///
/// The request is an ordinary one for the router it is sent to, so this
/// crate's own request counters see it: a fresh server's `/metrics/serve`
/// already shows `oxibonsai_serve_http_requests_total{route="/v1/models",status="200"} 1`
/// before any client has called it.
pub async fn warm_model_descriptor(router: &Router) {
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

/// The `503` for a request that finds the budget full: an
/// `overloaded_error` with `Retry-After: 1`, so a client backs off instead of
/// queueing behind the engine.
fn overloaded_response() -> Response {
    let body = Json(serde_json::json!({
        "error": {
            "message": "server is at its configured limits.max_concurrent_requests \
                         capacity; retry after a short backoff",
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

/// The `408` for a request that outlasted `limits.per_request_timeout_ms`.
///
/// This is the backstop for routes with no deadline of their own (the three
/// generation handlers answer their own, stage-naming `504` first). It uses
/// the API's one error envelope, `{message, type, param, code}`, with the
/// code [`ADMISSION_TIMEOUT_CODE`]; unlike the handler's `504` it names no
/// `phase`, because this layer cannot know where the request was caught.
fn timeout_response() -> Response {
    ApiError::new(
        StatusCode::REQUEST_TIMEOUT,
        "request exceeded the configured limits.per_request_timeout_ms budget",
    )
    .with_type("timeout_error")
    .with_code(ADMISSION_TIMEOUT_CODE)
    .into_response()
}

/// Request-holding helpers shared with `build_router`'s tests.
#[cfg(test)]
pub(crate) mod test_support;

/// The probe-route exemption, the timeout response and the start-up
/// warm-up, tested against a toy router.
#[cfg(test)]
#[path = "admission_tests.rs"]
mod exemption_tests;
