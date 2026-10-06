//! The remote-image fetches of one chat request.
//!
//! With remote images enabled (the operator's opt-in and a fetcher with an
//! address policy, see [`oxibonsai_model::vision::remote`]), a chat
//! request's `image_url` that names an `http(s)` URL is fetched while the
//! request prepares its images, on the blocking pool, one image after
//! another. [`RequestImageFetches`] is that request's record of them, on
//! both chat endpoints:
//!
//! * while a fetch is in flight the request's stage is `image_fetch`
//!   ([`super::phase::Phase::ImageFetch`]), so a per-request deadline that
//!   expires then answers `504` with `error.phase = "image_fetch"` instead
//!   of folding the fetch into `preparing` or `vision_encode`;
//! * the record lives in the handler's future: when that future is dropped —
//!   the per-request deadline fired, or the client disconnected — the record
//!   marks the request abandoned. The request starts no further fetch, and
//!   the fetch in flight is dropped (its connection closed) by a fetcher that
//!   watches the request ([`crate::vision_prefill::CurrentFetch`], as the
//!   `oxibonsai` command's fetcher does) instead of running on to its own
//!   per-image deadline;
//! * a fetch the fetcher refused because it is at capacity (it bounds how
//!   many fetches run and wait at once, process-wide) is answered as a load
//!   shed, not as a fault in the request: `503` with `Retry-After`,
//!   `error.type = "overloaded_error"` and `error.code =
//!   "image_fetch_overloaded"` ([`RequestImageFetches::classify_failure`]).
//!
//! Fetch time therefore counts toward the per-request timeout: the fetch
//! runs inside the handler that deadline bounds.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::server::api_error::{ApiError, ERROR_TYPE_OVERLOADED};
use crate::server::phase::RequestPhase;
use crate::vision_prefill::{ImageFetchObserver, VisionService, IMAGE_FETCH_OVERLOADED_CODE};

/// The `Retry-After` (seconds) of a request refused because the remote-image
/// fetcher was at capacity: the same short backoff the admission layer's own
/// overload asks for.
pub(crate) const IMAGE_FETCH_RETRY_AFTER_SECS: u64 = 1;

/// What a request's fetches report to.
#[derive(Debug, Default)]
struct FetchState {
    /// The request's stage record; `None` for an endpoint that keeps none.
    phase: Option<RequestPhase>,
    /// Set once the request's handler is gone.
    abandoned: AtomicBool,
    /// Set when the fetcher refused one of the request's fetches because it
    /// is at capacity.
    overloaded: AtomicBool,
}

impl ImageFetchObserver for FetchState {
    fn fetch_started(&self) {
        if let Some(phase) = &self.phase {
            phase.image_fetch_started();
        }
    }

    fn fetch_finished(&self) {
        if let Some(phase) = &self.phase {
            phase.image_fetch_finished();
        }
    }

    fn abandoned(&self) -> bool {
        self.abandoned.load(Ordering::Acquire)
    }

    fn fetch_overloaded(&self) {
        self.overloaded.store(true, Ordering::Release);
    }
}

/// One request's remote-image fetches (see the module docs). Hold it for as
/// long as the request may fetch; dropping it abandons the request.
#[derive(Debug)]
pub(crate) struct RequestImageFetches {
    state: Arc<FetchState>,
}

impl RequestImageFetches {
    /// A record reporting to `phase` (the request's stage record, when the
    /// endpoint keeps one).
    pub(crate) fn new(phase: Option<RequestPhase>) -> Self {
        Self {
            state: Arc::new(FetchState {
                phase,
                abandoned: AtomicBool::new(false),
                overloaded: AtomicBool::new(false),
            }),
        }
    }

    /// The request's view of the server's vision service: its remote-image
    /// fetcher, when one is installed, reports to this record.
    pub(crate) fn watch(&self, vision: Option<Arc<VisionService>>) -> Option<Arc<VisionService>> {
        vision.map(|service| {
            let observer: Arc<dyn ImageFetchObserver> = self.state.clone();
            VisionService::with_fetch_observer(&service, observer)
        })
    }

    /// The answer to a request whose image preparation failed with `error`:
    /// the retryable overload when the fetcher refused one of the request's
    /// fetches because it is at capacity (see [`overload_error`]), else
    /// `error` itself.
    pub(crate) fn classify_failure(&self, error: ApiError) -> ApiError {
        if self.state.overloaded.load(Ordering::Acquire) {
            overload_error(error.message())
        } else {
            error
        }
    }

    /// Whether the request has been abandoned.
    #[cfg(test)]
    pub(crate) fn is_abandoned(&self) -> bool {
        self.state.abandoned()
    }
}

impl Drop for RequestImageFetches {
    fn drop(&mut self) {
        self.state.abandoned.store(true, Ordering::Release);
    }
}

/// The load-shed answer of a request whose remote image the fetcher refused
/// because it is at capacity — the admission layer's overload shape (`503`,
/// `Retry-After`, `type: overloaded_error`) with a typed `code`
/// ([`IMAGE_FETCH_OVERLOADED_CODE`]) so a client can tell the two apart.
/// `message` says which image (the request's own message for it).
pub(crate) fn overload_error(message: &str) -> ApiError {
    ApiError::service_unavailable(message)
        .with_type(ERROR_TYPE_OVERLOADED)
        .with_code(IMAGE_FETCH_OVERLOADED_CODE)
        .with_retry_after(IMAGE_FETCH_RETRY_AFTER_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::phase::Phase;
    use axum::response::IntoResponse;

    #[test]
    fn a_fetch_is_reported_to_the_requests_stage_record() {
        let phase = RequestPhase::default();
        let fetches = RequestImageFetches::new(Some(phase.clone()));
        fetches.state.fetch_started();
        assert_eq!(phase.current(), Phase::ImageFetch);
        fetches.state.fetch_finished();
        assert_eq!(phase.current(), Phase::Preparing);
        assert!(!fetches.is_abandoned());
    }

    #[test]
    fn dropping_the_record_abandons_the_request() {
        let fetches = RequestImageFetches::new(None);
        let observer: Arc<dyn ImageFetchObserver> = fetches.state.clone();
        assert!(!observer.abandoned());
        // An endpoint without a stage record still hears nothing wrong.
        observer.fetch_started();
        observer.fetch_finished();
        drop(fetches);
        assert!(observer.abandoned(), "the handler is gone");
    }

    #[test]
    fn a_failure_is_an_overload_only_when_the_fetcher_said_so() {
        let fetches = RequestImageFetches::new(None);
        let refused = ApiError::bad_request("image 0: refused", "messages").with_code("x");
        let kept = fetches.classify_failure(refused);
        assert_eq!(kept.status(), axum::http::StatusCode::BAD_REQUEST);

        fetches.state.fetch_overloaded();
        let shed = fetches.classify_failure(ApiError::internal("image 0: at capacity"));
        assert_eq!(shed.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        let json = shed.to_json();
        assert_eq!(json["error"]["type"], "overloaded_error");
        assert_eq!(json["error"]["code"], "image_fetch_overloaded");
        assert_eq!(json["error"]["message"], "image 0: at capacity");
        assert!(json["error"]["param"].is_null());
    }

    #[test]
    fn the_overload_carries_retry_after() {
        let response = overload_error("image 0: at capacity").into_response();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }
}
