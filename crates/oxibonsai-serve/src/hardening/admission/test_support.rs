//! Deterministic request holding for the admission tests.
//!
//! A [`HeldGate`] serves `GET /__held`, a route whose handler parks until the
//! test releases it. A request parked there holds an admission slot for as
//! long as it is parked, so a test fills the budget with exactly as many
//! parked requests as it has slots. It learns that every one of them is
//! inside the stack from a counter the handler bumps *after* admission, never
//! from a sleep, so the setup cannot race on a slow or loaded machine.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, Request, StatusCode};
use axum::routing::get;
use axum::Router;
use tokio::sync::{watch, Semaphore};
use tokio::task::JoinHandle;
use tower::ServiceExt;

/// The path of the parked route.
pub(crate) const HELD_PATH: &str = "/__held";

/// The longest any single wait below may take. Every wait it guards returns
/// at once when the code under test is right, so reaching it means a hang and
/// fails the test instead of stalling the run; it is generous so a heavily
/// loaded machine does not turn a slow scheduler into a failure.
pub(crate) const BOUND: Duration = Duration::from_secs(30);

/// A route whose requests park inside the admission stack until released.
#[derive(Clone)]
pub(crate) struct HeldGate {
    entered: Arc<Semaphore>,
    release: Arc<watch::Sender<bool>>,
}

impl HeldGate {
    pub(crate) fn new() -> Self {
        let (release, _receiver) = watch::channel(false);
        Self {
            entered: Arc::new(Semaphore::new(0)),
            release: Arc::new(release),
        }
    }

    /// A router serving [`HELD_PATH`]: merge it into the router under test
    /// *before* the admission layer is applied, so the route is inside
    /// admission like any production route.
    pub(crate) fn router(&self) -> Router {
        Router::new()
            .route(HELD_PATH, get(parked))
            .with_state(self.clone())
    }

    /// Send `count` requests to [`HELD_PATH`] and return once every one of
    /// them is parked inside the stack (so holds one admission slot).
    ///
    /// The returned handles resolve with each request's status once
    /// [`Self::release`] is called.
    pub(crate) async fn park(&self, router: &Router, count: usize) -> Vec<JoinHandle<StatusCode>> {
        self.park_as(router, count, None).await
    }

    /// [`Self::park`] with an `Authorization: Bearer` header, for a router
    /// that requires one.
    pub(crate) async fn park_as(
        &self,
        router: &Router,
        count: usize,
        bearer: Option<&str>,
    ) -> Vec<JoinHandle<StatusCode>> {
        let handles: Vec<JoinHandle<StatusCode>> = (0..count)
            .map(|_| {
                let router = router.clone();
                let mut parked = request("GET", HELD_PATH);
                if let Some(token) = bearer {
                    parked.headers_mut().insert(
                        header::AUTHORIZATION,
                        HeaderValue::from_str(&format!("Bearer {token}"))
                            .expect("a header-safe token"),
                    );
                }
                tokio::spawn(async move {
                    match router.oneshot(parked).await {
                        Ok(response) => response.status(),
                        Err(never) => match never {},
                    }
                })
            })
            .collect();
        let wanted = u32::try_from(count).expect("a small number of parked requests");
        let entered = tokio::time::timeout(BOUND, self.entered.acquire_many(wanted))
            .await
            .expect("every parked request reaches the handler");
        entered
            .expect("the entered counter is never closed")
            .forget();
        handles
    }

    /// Let every parked request (and any later one) finish.
    pub(crate) fn release(&self) {
        self.release.send_replace(true);
    }
}

async fn parked(State(gate): State<HeldGate>) -> &'static str {
    gate.entered.add_permits(1);
    let mut released = gate.release.subscribe();
    // A dropped sender also ends the wait, so a test that failed before it
    // released the gate does not leave a handler parked forever.
    let _ = released.wait_for(|released| *released).await;
    "released"
}

/// A request with no body.
pub(crate) fn request(method: &str, path: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .expect("a well-formed request")
}

/// A response read to the end.
pub(crate) struct Reply {
    pub(crate) status: StatusCode,
    pub(crate) headers: HeaderMap,
    pub(crate) body: Vec<u8>,
}

impl Reply {
    /// The body as JSON, or `Null` when it is not JSON.
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }

    /// The body as text.
    pub(crate) fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// Whether this is the admission layer's overload response: `503`, an
    /// `overloaded_error` body and `Retry-After`.
    pub(crate) fn is_overload_shed(&self) -> bool {
        self.status == StatusCode::SERVICE_UNAVAILABLE
            && self.json()["error"]["type"] == "overloaded_error"
            && self.headers.contains_key("retry-after")
    }
}

/// Send `request` through `router` and read the whole response, failing the
/// test if the answer takes longer than [`BOUND`].
pub(crate) async fn send(router: &Router, request: Request<Body>) -> Reply {
    let answered = tokio::time::timeout(BOUND, router.clone().oneshot(request))
        .await
        .expect("the request is answered promptly");
    let response = match answered {
        Ok(response) => response,
        Err(never) => match never {},
    };
    let (parts, body) = response.into_parts();
    let body = axum::body::to_bytes(body, usize::MAX)
        .await
        .expect("a readable body");
    Reply {
        status: parts.status,
        headers: parts.headers,
        body: body.to_vec(),
    }
}

/// `GET path` through `router`.
pub(crate) async fn get_path(router: &Router, path: &str) -> Reply {
    send(router, request("GET", path)).await
}

/// Await every handle of [`HeldGate::park`] and check that each request was
/// answered `200` once released.
pub(crate) async fn finish_parked(handles: Vec<JoinHandle<StatusCode>>) {
    for handle in handles {
        let status = tokio::time::timeout(BOUND, handle)
            .await
            .expect("a released request finishes")
            .expect("the request task does not panic");
        assert_eq!(status, StatusCode::OK, "a released request is answered");
    }
}
