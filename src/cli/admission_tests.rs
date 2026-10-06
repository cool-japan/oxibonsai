//! The admission layer against a toy router: which routes skip the budget,
//! what a full budget answers, the timeout response and the start-up
//! warm-up. A sibling of `admission.rs`, declared there via `#[path]`, so
//! `super` names that module.
//!
//! Every test that needs a full budget parks real requests inside the stack
//! with a [`HeldGate`] and waits on its counter, so none of them depends on
//! a sleep or on how fast the machine is.
//!
//! Most of these tests specify the exemption, which the router did not have
//! before. The few that also pass on the pre-change layer are regression
//! guards for what the exemption must not change, and their doc comments say
//! so.

use std::sync::atomic::{AtomicUsize, Ordering};

use axum::routing::get;

use super::test_support::{finish_parked, get_path, request, send, HeldGate, BOUND, HELD_PATH};
use super::*;

/// The probe routes plus `/v1/models` (an API route) and the parked route.
fn toy_router(gate: &HeldGate) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/readyz", get(|| async { "ready" }))
        .route("/metrics", get(|| async { "metrics" }))
        .route("/ui/health", get(|| async { "ui" }))
        .route("/v1/models", get(|| async { "models" }))
        .merge(gate.router())
}

const LONG: Duration = Duration::from_secs(60);

#[test]
fn the_exempt_list_is_exactly_the_documented_probe_routes() {
    assert_eq!(
        ADMISSION_EXEMPT_PATHS,
        ["/health", "/readyz", "/metrics", "/ui/health"],
        "adding a route here is a decision to document in docs/DEPLOYMENT.md and docs/CLI.md"
    );
}

#[test]
fn exemption_is_an_exact_path_match() {
    let admission = Admission::new(1, LONG, ADMISSION_EXEMPT_PATHS);
    for exempt in ADMISSION_EXEMPT_PATHS {
        assert!(admission.is_exempt(exempt), "{exempt}");
    }
    for inside in [
        "/health/",
        "/healthz",
        "/HEALTH",
        "//health",
        "/readyz/x",
        "/metrics/serve",
        "/ui",
        "/ui/",
        "/v1/models",
        "/v1/models/health",
        "/admin/status",
        "/",
        "",
    ] {
        assert!(
            !admission.is_exempt(inside),
            "{inside:?} must stay inside admission"
        );
    }
}

/// The scenario the exemption exists for: every slot is held, the probe
/// routes still answer, an API route is shed with the documented overload,
/// and once the held requests finish everything answers normally.
///
/// Before the exemption existed every probe route here answered the
/// overload `503`, so the four probe assertions failed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_routes_answer_while_the_budget_is_full() {
    let gate = HeldGate::new();
    let admission = Arc::new(Admission::new(2, LONG, ADMISSION_EXEMPT_PATHS));
    let router = Arc::clone(&admission).apply(toy_router(&gate));

    let parked = gate.park(&router, 2).await;
    assert_eq!(admission.free_slots(), 0, "both slots are held");

    for path in ADMISSION_EXEMPT_PATHS {
        let reply = get_path(&router, path).await;
        assert_eq!(reply.status, StatusCode::OK, "{path} at the limit");
    }
    for shed in ["/v1/models", HELD_PATH] {
        let reply = get_path(&router, shed).await;
        assert!(
            reply.is_overload_shed(),
            "{shed} at the limit: {}",
            reply.text()
        );
    }
    assert_eq!(admission.free_slots(), 0, "shedding takes no slot");

    gate.release();
    finish_parked(parked).await;
    assert_eq!(admission.free_slots(), 2, "every slot comes back");
    for path in ["/v1/models", "/health", "/readyz", "/metrics", "/ui/health"] {
        let reply = get_path(&router, path).await;
        assert_eq!(reply.status, StatusCode::OK, "{path} after release");
    }
}

/// The paths exempted in [`exempt_requests_never_hold_a_slot`]: the parked
/// route stands in for a probe whose request stays in flight.
const PARKED_IS_EXEMPT: &[&str] = &["/health", HELD_PATH];

/// An exempt request never holds a slot: three requests parked inside an
/// exempt route, against a budget of one, leave the whole budget available to
/// the API routes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exempt_requests_never_hold_a_slot() {
    let gate = HeldGate::new();
    let admission = Arc::new(Admission::new(1, LONG, PARKED_IS_EXEMPT));
    let router = Arc::clone(&admission).apply(toy_router(&gate));

    let parked = gate.park(&router, 3).await;
    assert_eq!(
        admission.free_slots(),
        1,
        "requests in flight on an exempt route hold no slot"
    );
    assert_eq!(
        get_path(&router, "/v1/models").await.status,
        StatusCode::OK,
        "the whole budget is still there for an API route"
    );

    gate.release();
    finish_parked(parked).await;
    assert_eq!(admission.free_slots(), 1);
}

/// The budget is one shared pool, not one per route: a slot held on one
/// route sheds a request to another.
///
/// A regression guard: the budget was already shared before the probe routes
/// were exempted, so this also passes on the pre-change layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_budget_is_shared_by_every_route() {
    let gate = HeldGate::new();
    let router = apply_admission(toy_router(&gate), 1, 60_000);

    let parked = gate.park(&router, 1).await;
    let reply = get_path(&router, "/v1/models").await;
    assert!(reply.is_overload_shed(), "{}", reply.text());
    gate.release();
    finish_parked(parked).await;
}

/// An exempt route skips the budget but not the timeout: a probe whose
/// handler never answers is cut off and answered `408`, not left hanging,
/// and it never held a slot while it waited.
///
/// Mostly a regression guard: every request was already bound by the timeout.
/// On the pre-change layer only the `code` assertion fails (that `408` body
/// carried `code: null`).
#[tokio::test]
async fn an_exempt_route_is_still_bounded_by_the_timeout() {
    let admission = Arc::new(Admission::new(
        1,
        Duration::from_millis(5),
        ADMISSION_EXEMPT_PATHS,
    ));
    let stuck = Router::new().route(
        "/readyz",
        get(|| async {
            std::future::pending::<()>().await;
            "never"
        }),
    );
    let router = Arc::clone(&admission).apply(stuck);
    let reply = get_path(&router, "/readyz").await;
    assert_eq!(reply.status, StatusCode::REQUEST_TIMEOUT);
    assert_eq!(reply.json()["error"]["code"], ADMISSION_TIMEOUT_CODE);
    assert_eq!(admission.free_slots(), 1, "the probe never held a slot");
}

/// The admission timeout answers with the API's error envelope and a stable
/// `code`: the same `request_timeout` the handler's own deadline reports, in
/// the same `{message, type, param, code}` shape. The status stays `408` and
/// the `type` stays `timeout_error`.
///
/// The `code` used to be `null`.
#[tokio::test]
async fn the_timeout_response_carries_a_stable_error_code() {
    let slow = Router::new().route(
        "/slow",
        get(|| async {
            std::future::pending::<()>().await;
            "never"
        }),
    );
    let router = apply_admission(slow, 4, 5);
    let reply = get_path(&router, "/slow").await;
    assert_eq!(reply.status, StatusCode::REQUEST_TIMEOUT);
    let body = reply.json();
    let error = &body["error"];
    assert_eq!(error["code"], "request_timeout", "{body}");
    assert_eq!(error["type"], "timeout_error", "{body}");
    assert!(error["param"].is_null(), "{body}");
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|message| message.contains("--request-timeout-ms")),
        "the message names the knob: {body}"
    );

    // The same members, in the same envelope, as the handler's `504`.
    let handler_deadline = ApiError::timeout("deadline").to_json();
    let keys = |value: &serde_json::Value| -> Vec<String> {
        value["error"]
            .as_object()
            .map(|members| members.keys().cloned().collect())
            .unwrap_or_default()
    };
    assert_eq!(keys(&body), keys(&handler_deadline));
    assert_eq!(error["code"], handler_deadline["error"]["code"]);
}

/// The shed response keeps its documented shape: `503`, `Retry-After: 1`
/// and an `overloaded_error` in the API's error envelope.
///
/// A regression guard: the response had this shape before the probe routes
/// were exempted, so this also passes on the pre-change layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_overload_response_keeps_its_documented_shape() {
    let gate = HeldGate::new();
    let router = apply_admission(toy_router(&gate), 1, 60_000);
    let parked = gate.park(&router, 1).await;

    let reply = get_path(&router, "/v1/models").await;
    assert_eq!(reply.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        reply
            .headers
            .get(header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let body = reply.json();
    assert_eq!(body["error"]["type"], "overloaded_error", "{body}");
    assert!(body["error"]["param"].is_null(), "{body}");
    assert!(body["error"]["message"].is_string(), "{body}");

    gate.release();
    finish_parked(parked).await;
}

/// A POST (which would carry a body) and a HEAD are shed or exempt by path
/// alone, the same as a GET.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exemption_does_not_depend_on_the_method() {
    let gate = HeldGate::new();
    let router = apply_admission(toy_router(&gate), 1, 60_000);
    let parked = gate.park(&router, 1).await;

    let head = send(&router, request("HEAD", "/health")).await;
    assert_eq!(head.status, StatusCode::OK, "HEAD /health at the limit");
    let post_probe = send(&router, request("POST", "/health")).await;
    assert_eq!(
        post_probe.status,
        StatusCode::METHOD_NOT_ALLOWED,
        "a probe route's own 405 is answered, not the overload"
    );
    let post_api = send(&router, request("POST", "/v1/models")).await;
    assert!(post_api.is_overload_shed(), "{}", post_api.text());

    gate.release();
    finish_parked(parked).await;
}

/// The warm-up sends exactly one `GET /v1/models` through the router it is
/// given, and a router that cannot answer it (no such route, or an error)
/// neither panics nor hangs: the server still starts.
///
/// The warm-up is new, so there is nothing for this to fail on before the
/// change: it pins the helper's contract, and the probe tests of `cmd_serve`
/// show its effect on the whole router.
#[tokio::test]
async fn the_descriptor_warm_up_asks_v1_models_once_and_tolerates_failure() {
    let asked = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&asked);
    let answering = Router::new().route(
        "/v1/models",
        get(move || {
            let counted = Arc::clone(&counted);
            async move {
                counted.fetch_add(1, Ordering::SeqCst);
                "models"
            }
        }),
    );
    tokio::time::timeout(BOUND, warm_model_descriptor(&answering))
        .await
        .expect("the warm-up returns");
    assert_eq!(asked.load(Ordering::SeqCst), 1);

    let missing = Router::new().route("/health", get(|| async { "ok" }));
    tokio::time::timeout(BOUND, warm_model_descriptor(&missing))
        .await
        .expect("a missing route does not hang the warm-up");

    let failing = Router::new().route(
        "/v1/models",
        get(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
    );
    tokio::time::timeout(BOUND, warm_model_descriptor(&failing))
        .await
        .expect("a failing route does not hang the warm-up");
}
