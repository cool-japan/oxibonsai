//! Watching the remote-image fetches one request makes.
//!
//! The fetcher an [`ImageSourcePolicy`](oxibonsai_model::vision::ImageSourcePolicy)
//! carries is shared by every request a server answers, so it cannot know
//! which request a fetch belongs to. [`super::VisionService::with_fetch_observer`]
//! gives one request its own view of the shared service: the same tower and
//! policy, with the fetcher wrapped in an [`ObservedFetcher`] that tells the
//! request's [`ImageFetchObserver`] when each fetch starts and ends, and asks
//! it before every fetch whether the request is still wanted.
//!
//! While the wrapped fetcher's `fetch` runs, the request is also the
//! **current fetch** of the calling thread ([`CurrentFetch`]): the fetcher
//! reads it there — the model crate's fetcher seam stays a plain synchronous
//! call with no network-aware parameter — and can
//!
//! * keep asking whether the request was abandoned *during* the transfer
//!   ([`CurrentFetch::is_abandoned`]), and drop the transfer (closing its
//!   connection) as soon as it was, instead of running on to its own
//!   per-image deadline; and
//! * report that it refused the fetch because it is at capacity
//!   ([`CurrentFetch::report_overloaded`]): a server answers that request
//!   with a retryable overload, not with a fault in the request, and the
//!   request's view of the service reports the image as
//!   [`super::MultimodalError::FetchOverloaded`].
//!
//! A server uses this for its per-request deadline (which names the
//! `image_fetch` stage while a fetch is in flight) and for clients that go
//! away: a request whose client went away, or whose deadline expired, starts
//! no further fetch, and the fetch in flight is dropped by a fetcher that
//! watches the current fetch.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use oxibonsai_model::vision::remote::RemoteFetchFailure;
use oxibonsai_model::vision::{ImageInputError, RemoteImageFetcher, SharedRemoteImageFetcher};

/// Told about the remote-image fetches of one request.
///
/// Every method has a do-nothing default, so an observer implements only
/// what it needs. The methods are called on the thread that resolves the
/// request's images (a blocking-pool thread on a server), and
/// [`Self::abandoned`] also from the fetcher while a fetch waits or
/// transfers, so it must be cheap.
pub trait ImageFetchObserver: Send + Sync {
    /// A fetch is about to start.
    fn fetch_started(&self) {}

    /// The fetch that started last has ended, whatever its outcome.
    fn fetch_finished(&self) {}

    /// Whether the request was abandoned: `true` refuses every later fetch
    /// with [`RemoteFetchFailure::Abandoned`] before it starts, and tells a
    /// fetcher that watches the [`CurrentFetch`] to drop the fetch in flight.
    fn abandoned(&self) -> bool {
        false
    }

    /// The fetcher refused the request's fetch because it is at capacity
    /// (the fetch failed with that refusal; nothing was opened).
    fn fetch_overloaded(&self) {}
}

/// The request a remote-image fetch on this thread is made for, as the
/// fetcher sees it (see the module docs).
///
/// A request's view of the vision service makes its request the current
/// fetch of the calling thread for exactly as long as the wrapped fetcher's
/// `fetch` runs; [`CurrentFetch::scope`] does the same for any other caller.
/// Elsewhere — `run` / `chat`, a library caller without an observer — there
/// is none, and a fetcher runs as it always did.
#[derive(Clone)]
pub struct CurrentFetch {
    observer: Arc<dyn ImageFetchObserver>,
    /// Shared with the request's view of the vision service, which reports an
    /// image refused for capacity as [`super::MultimodalError::FetchOverloaded`].
    overloaded: Arc<AtomicBool>,
}

thread_local! {
    /// The calling thread's current fetch, while one runs.
    static CURRENT: Cell<Option<CurrentFetch>> = const { Cell::new(None) };
}

impl std::fmt::Debug for CurrentFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurrentFetch")
            .field("abandoned", &self.is_abandoned())
            .field("overloaded", &self.overloaded.load(Ordering::Acquire))
            .finish()
    }
}

impl CurrentFetch {
    /// The request whose fetch runs on the calling thread, if one does.
    #[must_use]
    pub fn on_this_thread() -> Option<Self> {
        CURRENT
            .try_with(|current| {
                let value = current.take();
                let copy = value.clone();
                current.set(value);
                copy
            })
            .ok()
            .flatten()
    }

    /// Whether the request was abandoned (its client went away, or its
    /// deadline expired): a fetcher drops the fetch in flight when this
    /// becomes `true`.
    #[must_use]
    pub fn is_abandoned(&self) -> bool {
        self.observer.abandoned()
    }

    /// The fetcher refused this request's fetch because it is at capacity:
    /// call it right before returning that refusal.
    pub fn report_overloaded(&self) {
        self.overloaded.store(true, Ordering::Release);
        self.observer.fetch_overloaded();
    }

    /// Run `fetch` with `observer`'s request as the calling thread's current
    /// fetch — what a request's view of the vision service does around every
    /// fetch, for a caller that drives a fetcher itself. The previous current
    /// fetch (if any) is restored afterwards, also when `fetch` unwinds.
    pub fn scope<R>(observer: Arc<dyn ImageFetchObserver>, fetch: impl FnOnce() -> R) -> R {
        Self {
            observer,
            overloaded: Arc::new(AtomicBool::new(false)),
        }
        .enter(fetch)
    }

    /// Make this the calling thread's current fetch while `fetch` runs.
    fn enter<R>(self, fetch: impl FnOnce() -> R) -> R {
        let previous = CURRENT.try_with(|current| current.replace(Some(self)));
        let _restore = RestoreCurrent(previous.ok().flatten());
        fetch()
    }
}

/// Puts the previous current fetch back when dropped.
struct RestoreCurrent(Option<CurrentFetch>);

impl Drop for RestoreCurrent {
    fn drop(&mut self) {
        let previous = self.0.take();
        // A thread being torn down has no current fetch to restore.
        let _ = CURRENT.try_with(|current| current.set(previous));
    }
}

/// A fetcher that reports each fetch of the fetcher it wraps to one
/// request's observer.
pub(crate) struct ObservedFetcher {
    inner: SharedRemoteImageFetcher,
    observer: Arc<dyn ImageFetchObserver>,
    /// Set when the wrapped fetcher refused one of the request's fetches for
    /// capacity; shared with the request's view of the vision service.
    overloaded: Arc<AtomicBool>,
}

impl ObservedFetcher {
    /// Wrap `inner`, reporting to `observer`; an overload refusal is also
    /// recorded in `overloaded`.
    pub(crate) fn new(
        inner: SharedRemoteImageFetcher,
        observer: Arc<dyn ImageFetchObserver>,
        overloaded: Arc<AtomicBool>,
    ) -> Self {
        Self {
            inner,
            observer,
            overloaded,
        }
    }
}

/// Reports the end of a fetch when dropped, so the observer hears of it
/// even if the wrapped fetcher unwinds.
struct FetchInFlight<'a>(&'a dyn ImageFetchObserver);

impl Drop for FetchInFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_finished();
    }
}

impl RemoteImageFetcher for ObservedFetcher {
    fn fetch(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        if self.observer.abandoned() {
            return Err(RemoteFetchFailure::Abandoned.into_error(url));
        }
        self.observer.fetch_started();
        let _in_flight = FetchInFlight(self.observer.as_ref());
        CurrentFetch {
            observer: Arc::clone(&self.observer),
            overloaded: Arc::clone(&self.overloaded),
        }
        .enter(|| self.inner.fetcher().fetch(url, max_bytes))
    }

    fn preflight(&self, url: &str) -> Result<(), ImageInputError> {
        self.inner.fetcher().preflight(url)
    }

    fn describe(&self) -> String {
        self.inner.fetcher().describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Records what it is told; `abandoned()` answers from a flag.
    #[derive(Default)]
    struct Recorder {
        abandoned: AtomicBool,
        overloads: AtomicUsize,
        started: AtomicUsize,
        finished: AtomicUsize,
    }

    impl ImageFetchObserver for Recorder {
        fn fetch_started(&self) {
            self.started.fetch_add(1, Ordering::SeqCst);
        }
        fn fetch_finished(&self) {
            self.finished.fetch_add(1, Ordering::SeqCst);
        }
        fn abandoned(&self) -> bool {
            self.abandoned.load(Ordering::SeqCst)
        }
        fn fetch_overloaded(&self) {
            self.overloads.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A fetcher that reads the current fetch the way a real one does, and
    /// does what the test scripted with it.
    struct Probe {
        report_overload: bool,
        saw_current: AtomicBool,
        saw_abandoned: AtomicBool,
    }

    impl Probe {
        fn new(report_overload: bool) -> Arc<Self> {
            Arc::new(Self {
                report_overload,
                saw_current: AtomicBool::new(false),
                saw_abandoned: AtomicBool::new(false),
            })
        }
    }

    impl RemoteImageFetcher for Probe {
        fn fetch(&self, url: &str, _max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
            let current = CurrentFetch::on_this_thread();
            self.saw_current.store(current.is_some(), Ordering::SeqCst);
            if let Some(current) = current {
                self.saw_abandoned
                    .store(current.is_abandoned(), Ordering::SeqCst);
                if self.report_overload {
                    current.report_overloaded();
                    return Err(RemoteFetchFailure::Unavailable {
                        reason: "at capacity".to_string(),
                    }
                    .into_error(url));
                }
            }
            Ok(vec![1, 2, 3])
        }
    }

    fn observed(
        probe: &Arc<Probe>,
        recorder: &Arc<Recorder>,
    ) -> (ObservedFetcher, Arc<AtomicBool>) {
        let overloaded = Arc::new(AtomicBool::new(false));
        let inner: Arc<dyn RemoteImageFetcher> = probe.clone();
        let observer: Arc<dyn ImageFetchObserver> = recorder.clone();
        (
            ObservedFetcher::new(
                SharedRemoteImageFetcher::new(inner),
                observer,
                Arc::clone(&overloaded),
            ),
            overloaded,
        )
    }

    #[test]
    fn the_wrapped_fetcher_sees_its_request_and_only_while_it_runs() {
        let probe = Probe::new(false);
        let recorder = Arc::new(Recorder::default());
        let (fetcher, overloaded) = observed(&probe, &recorder);
        assert!(CurrentFetch::on_this_thread().is_none());
        let bytes = fetcher.fetch("https://images.example/a.png", 16);
        assert_eq!(bytes.ok(), Some(vec![1, 2, 3]));
        assert!(probe.saw_current.load(Ordering::SeqCst));
        assert!(!probe.saw_abandoned.load(Ordering::SeqCst));
        assert!(
            CurrentFetch::on_this_thread().is_none(),
            "the scope ends with the fetch"
        );
        assert_eq!(recorder.started.load(Ordering::SeqCst), 1);
        assert_eq!(recorder.finished.load(Ordering::SeqCst), 1);
        assert!(!overloaded.load(Ordering::SeqCst));
    }

    #[test]
    fn an_overload_reported_by_the_fetcher_reaches_the_request_and_its_record() {
        let probe = Probe::new(true);
        let recorder = Arc::new(Recorder::default());
        let (fetcher, overloaded) = observed(&probe, &recorder);
        let error = fetcher
            .fetch("https://images.example/a.png", 16)
            .expect_err("refused for capacity");
        assert_eq!(error.code(), "image_url_fetch_failed");
        assert_eq!(recorder.overloads.load(Ordering::SeqCst), 1);
        assert!(overloaded.load(Ordering::SeqCst));
        assert_eq!(recorder.finished.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_abandoned_request_starts_no_fetch_and_a_fetcher_can_see_it_mid_fetch() {
        let probe = Probe::new(false);
        let recorder = Arc::new(Recorder::default());
        recorder.abandoned.store(true, Ordering::SeqCst);
        let (fetcher, _) = observed(&probe, &recorder);
        let error = fetcher
            .fetch("https://images.example/a.png", 16)
            .expect_err("abandoned");
        assert!(error.to_string().contains("abandoned"), "{error}");
        assert!(!probe.saw_current.load(Ordering::SeqCst), "never called");

        // A caller that drives a fetcher itself gets the same view.
        let observer: Arc<dyn ImageFetchObserver> = recorder.clone();
        let seen = CurrentFetch::scope(observer, || {
            CurrentFetch::on_this_thread().map(|current| current.is_abandoned())
        });
        assert_eq!(seen, Some(true));
        assert!(CurrentFetch::on_this_thread().is_none());
    }

    #[test]
    fn scopes_nest_and_unwind_back_to_the_outer_request() {
        let outer = Arc::new(Recorder::default());
        let inner = Arc::new(Recorder::default());
        inner.abandoned.store(true, Ordering::SeqCst);
        let outer_observer: Arc<dyn ImageFetchObserver> = outer.clone();
        let inner_observer: Arc<dyn ImageFetchObserver> = inner.clone();
        CurrentFetch::scope(outer_observer, || {
            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                CurrentFetch::scope(Arc::clone(&inner_observer), || {
                    assert_eq!(
                        CurrentFetch::on_this_thread().map(|c| c.is_abandoned()),
                        Some(true)
                    );
                    panic!("a fetcher that unwinds");
                })
            }));
            assert!(unwound.is_err());
            assert_eq!(
                CurrentFetch::on_this_thread().map(|c| c.is_abandoned()),
                Some(false),
                "the outer request is current again"
            );
        });
        assert!(CurrentFetch::on_this_thread().is_none());
    }
}
