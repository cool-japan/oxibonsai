//! A fetch whose request is abandoned, and the process-wide bound on
//! fetches, against loopback listeners that count the connections they
//! accept and notice the ones the fetcher closes.
//!
//! The request a fetch is made for is what a server hands the fetcher:
//! [`CurrentFetch`], set around the fetch by the runtime
//! ([`CurrentFetch::scope`] here, a request's view of the vision service on
//! a server). Every test fetches under transfer slots of its own
//! ([`ImageUrlFetcher::with_capacity`]) sized like the process-wide bound, so
//! tests running side by side in one process never take each other's slots.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use oxibonsai_model::vision::{ImageInputError, ImageSourcePolicy, RemoteImageFetcher};
use oxibonsai_runtime::vision_prefill::{CurrentFetch, ImageFetchObserver};

use super::capacity::{FetchCapacity, FetchCounts, MAX_FETCHES_IN_FLIGHT, MAX_FETCHES_WAITING};
use super::resolver::{AddressLookup, LookupFuture};
use super::test_server::{Gate, Reply, TestServer};
use super::{classify_address, ImageFetchSettings, ImageUrlFetcher};
use crate::cli::bonsai2::tests::PATTERN_PNG_DATA_URI;

/// How long an abandoned fetch may take to close its connection: far below
/// the 10 s per-image deadline these tests fetch under.
const CLOSED_WITHIN: Duration = Duration::from_secs(2);

/// The per-image deadline of the tests that must not hit it.
const LONG_DEADLINE_MS: u64 = 10_000;

fn png() -> Vec<u8> {
    oxibonsai_model::vision::load_image_bytes(
        PATTERN_PNG_DATA_URI,
        &ImageSourcePolicy::local_user(),
    )
    .expect("the embedded PNG decodes from base64")
}

/// Transfer slots of a test's own, sized like the process-wide bound.
fn own_capacity() -> Arc<FetchCapacity> {
    FetchCapacity::new(MAX_FETCHES_IN_FLIGHT, MAX_FETCHES_WAITING)
}

/// A fetcher with `timeout_ms`, the server's address allowlisted, under
/// `capacity`.
fn fetcher(timeout_ms: u64, server: &TestServer, capacity: &Arc<FetchCapacity>) -> ImageUrlFetcher {
    let allow = vec![format!("127.0.0.1:{}", server.port())];
    let settings =
        ImageFetchSettings::resolve(Some(timeout_ms), None, &allow, None).expect("valid settings");
    ImageUrlFetcher::new(settings).with_capacity(Arc::clone(capacity))
}

/// A request as the fetcher sees it: abandoned when the test says so; it
/// counts the overloads reported to it.
#[derive(Default)]
struct Request {
    abandoned: AtomicBool,
    overloads: AtomicUsize,
}

impl Request {
    fn abandon(&self) {
        self.abandoned.store(true, Ordering::SeqCst);
    }
}

impl ImageFetchObserver for Request {
    fn abandoned(&self) -> bool {
        self.abandoned.load(Ordering::SeqCst)
    }

    fn fetch_overloaded(&self) {
        self.overloads.fetch_add(1, Ordering::SeqCst);
    }
}

type Fetch = JoinHandle<Result<Vec<u8>, ImageInputError>>;

/// Fetch `url` on a thread of its own, on behalf of `request`.
fn fetch_for(fetcher: &Arc<ImageUrlFetcher>, request: &Arc<Request>, url: String) -> Fetch {
    let fetcher = Arc::clone(fetcher);
    let observer: Arc<dyn ImageFetchObserver> = request.clone();
    std::thread::spawn(move || CurrentFetch::scope(observer, || fetcher.fetch(&url, 1 << 22)))
}

/// Poll `ready` (every few milliseconds, at most `limit`); whether it held.
fn wait_until(limit: Duration, ready: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if ready() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    ready()
}

/// Wait (bounded) until every slot of `capacity` is back.
fn all_slots_back(capacity: &FetchCapacity) -> bool {
    wait_until(Duration::from_secs(5), || {
        capacity.counts() == FetchCounts::default()
    })
}

// ── A fetch whose request is abandoned ──────────────────────────────────────

/// A fetch stalled by its server is in flight when its request is abandoned
/// (the client went away, or the request's deadline expired): the fetch ends
/// at once as abandoned, our side closes the connection (the listener sees
/// it closed) far below the per-image deadline, and its slot comes back.
#[test]
fn an_abandoned_fetch_closes_its_connection_long_before_its_deadline() {
    let server = TestServer::start("127.0.0.1", |_| Reply::Stall(Duration::from_secs(60)));
    let capacity = own_capacity();
    let fetcher = Arc::new(fetcher(LONG_DEADLINE_MS, &server, &capacity));
    let request = Arc::new(Request::default());
    let fetch = fetch_for(&fetcher, &request, server.url("/stalled.png"));
    assert!(
        wait_until(Duration::from_secs(10), || server.accepts() == 1),
        "the fetch never connected"
    );
    assert_eq!(capacity.counts().in_flight, 1);

    let abandoned_at = Instant::now();
    request.abandon();
    let outcome = fetch.join().expect("the fetching thread finished");
    let returned_after = abandoned_at.elapsed();
    let error = outcome.expect_err("an abandoned fetch returns no image");
    assert_eq!(error.code(), "image_url_fetch_failed", "{error}");
    assert!(error.to_string().contains("abandoned"), "{error}");
    assert!(
        returned_after < CLOSED_WITHIN,
        "the fetch returned {returned_after:?} after its request was abandoned"
    );
    assert!(
        server.wait_finished(1, CLOSED_WITHIN),
        "our side must close the connection of an abandoned fetch long before its deadline"
    );
    assert!(all_slots_back(&capacity), "{:?}", capacity.counts());
    assert_eq!(server.accepts(), 1, "nothing was retried");
}

/// A request abandoned before its fetch starts opens nothing.
#[test]
fn a_fetch_for_an_abandoned_request_opens_nothing() {
    let server = TestServer::start("127.0.0.1", |_| Reply::Ok(png()));
    let capacity = own_capacity();
    let fetcher = Arc::new(fetcher(LONG_DEADLINE_MS, &server, &capacity));
    let request = Arc::new(Request::default());
    request.abandon();
    let error = fetch_for(&fetcher, &request, server.url("/img.png"))
        .join()
        .expect("the fetching thread finished")
        .expect_err("abandoned");
    assert!(error.to_string().contains("abandoned"), "{error}");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(server.accepts(), 0);
    assert_eq!(capacity.counts(), FetchCounts::default());
}

// ── The bound ───────────────────────────────────────────────────────────────

/// With every transfer slot taken by a stalled fetch and every waiting place
/// taken, the next fetch is refused at once — the typed overload, reported to
/// its request — and opens no connection. Once the stall clears, the waiting
/// fetches run, every fetch answers, the counts return to zero, and fetching
/// works again.
#[test]
fn the_bound_refuses_the_next_fetch_at_once_and_opens_no_connection() {
    let gate = Arc::new(Gate::default());
    let body = png();
    let server = {
        let gate = Arc::clone(&gate);
        let body = body.clone();
        TestServer::start("127.0.0.1", move |head| {
            if head.contains("/stall") {
                Reply::Gated {
                    gate: Arc::clone(&gate),
                    body: body.clone(),
                }
            } else {
                Reply::Ok(body.clone())
            }
        })
    };
    let capacity = own_capacity();
    let fetcher = Arc::new(fetcher(60_000, &server, &capacity));

    let mut fetches = Vec::new();
    for i in 0..MAX_FETCHES_IN_FLIGHT {
        let request = Arc::new(Request::default());
        fetches.push(fetch_for(
            &fetcher,
            &request,
            server.url(&format!("/stall/{i}.png")),
        ));
    }
    assert!(
        wait_until(Duration::from_secs(10), || {
            server.accepts() == MAX_FETCHES_IN_FLIGHT
                && capacity.counts().in_flight == MAX_FETCHES_IN_FLIGHT
        }),
        "every slot holds a stalled transfer: {:?}, {} accepted",
        capacity.counts(),
        server.accepts()
    );
    for i in 0..MAX_FETCHES_WAITING {
        let request = Arc::new(Request::default());
        fetches.push(fetch_for(
            &fetcher,
            &request,
            server.url(&format!("/stall/waiting-{i}.png")),
        ));
    }
    assert!(
        wait_until(Duration::from_secs(10), || {
            capacity.counts().waiting == MAX_FETCHES_WAITING
        }),
        "every waiting place is taken: {:?}",
        capacity.counts()
    );
    assert_eq!(
        server.accepts(),
        MAX_FETCHES_IN_FLIGHT,
        "a waiting fetch opens no connection"
    );

    // The next one is refused at once, typed, and opens nothing.
    let refused_request = Arc::new(Request::default());
    let started = Instant::now();
    let error = fetch_for(
        &fetcher,
        &refused_request,
        server.url("/stall/one-too-many.png"),
    )
    .join()
    .expect("the fetching thread finished")
    .expect_err("over the bound");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "refused at once, not after waiting: {:?}",
        started.elapsed()
    );
    assert_eq!(error.code(), "image_url_fetch_failed", "{error}");
    assert!(error.to_string().contains("at capacity"), "{error}");
    assert_eq!(
        refused_request.overloads.load(Ordering::SeqCst),
        1,
        "the overload is reported to the request"
    );
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        server.accepts(),
        MAX_FETCHES_IN_FLIGHT,
        "the refused fetch opened no connection"
    );
    assert_eq!(
        capacity.counts(),
        FetchCounts {
            in_flight: MAX_FETCHES_IN_FLIGHT,
            waiting: MAX_FETCHES_WAITING
        },
        "a refusal takes no slot"
    );

    // The stall clears: every fetch, waiting ones included, gets its image.
    gate.open();
    for fetch in fetches {
        let bytes = fetch
            .join()
            .expect("the fetching thread finished")
            .expect("fetched once the stall cleared");
        assert_eq!(bytes, body);
    }
    assert_eq!(
        server.accepts(),
        MAX_FETCHES_IN_FLIGHT + MAX_FETCHES_WAITING,
        "the waiting fetches connected once they got a slot"
    );
    assert!(all_slots_back(&capacity), "{:?}", capacity.counts());
    let again = fetcher
        .fetch(&server.url("/img.png"), 1 << 22)
        .expect("fetching works again");
    assert_eq!(again, body);
    assert!(all_slots_back(&capacity), "{:?}", capacity.counts());
}

/// A fetch waiting for a slot gives its place up when its request is
/// abandoned, and opens nothing.
#[test]
fn a_waiting_fetch_gives_up_its_place_when_abandoned() {
    let gate = Arc::new(Gate::default());
    let server = {
        let gate = Arc::clone(&gate);
        TestServer::start("127.0.0.1", move |_| Reply::Gated {
            gate: Arc::clone(&gate),
            body: Vec::new(),
        })
    };
    let capacity = FetchCapacity::new(1, 1);
    let fetcher = Arc::new(fetcher(LONG_DEADLINE_MS, &server, &capacity));
    let holder = fetch_for(
        &fetcher,
        &Arc::new(Request::default()),
        server.url("/held.png"),
    );
    assert!(wait_until(Duration::from_secs(10), || server.accepts() == 1));
    let request = Arc::new(Request::default());
    let waiter = fetch_for(&fetcher, &request, server.url("/waiting.png"));
    assert!(wait_until(Duration::from_secs(10), || capacity
        .counts()
        .waiting
        == 1));
    request.abandon();
    let error = waiter
        .join()
        .expect("the waiting thread finished")
        .expect_err("abandoned while waiting");
    assert!(error.to_string().contains("abandoned"), "{error}");
    assert_eq!(capacity.counts().waiting, 0);
    assert_eq!(server.accepts(), 1, "the waiting fetch never connected");
    gate.open();
    let _ = holder.join();
    assert!(all_slots_back(&capacity), "{:?}", capacity.counts());
}

// ── Every exit returns its slot ─────────────────────────────────────────────

/// A resolver whose lookup panics: the worker's task unwinds mid-transfer.
struct PanickingLookup;

impl AddressLookup for PanickingLookup {
    fn lookup(&self, _host: &str) -> LookupFuture {
        Box::pin(async { panic!("a resolver that fails hard") })
    }
}

/// A request whose `abandoned()` fails hard once the fetch is waiting for a
/// slot (its first answer, before the fetch asks for a slot, is `false`).
#[derive(Default)]
struct FailingProbe {
    asked: AtomicUsize,
}

impl ImageFetchObserver for FailingProbe {
    fn abandoned(&self) -> bool {
        if self.asked.fetch_add(1, Ordering::SeqCst) > 0 {
            panic!("a probe that fails hard");
        }
        false
    }
}

/// No exit leaks a slot or a waiting place: an image, a refusal by policy, a
/// failure status, the deadline, an abandoned request, a transfer that
/// panics on the worker, and a waiter whose probe panics.
#[test]
fn every_exit_returns_its_slot() {
    let body = png();
    let server = {
        let body = body.clone();
        TestServer::start("127.0.0.1", move |head| {
            if head.contains("/missing") {
                Reply::Status(404, b"no".to_vec())
            } else if head.contains("/stall") {
                Reply::Stall(Duration::from_secs(60))
            } else {
                Reply::Ok(body.clone())
            }
        })
    };
    let capacity = own_capacity();
    let quick = fetcher(LONG_DEADLINE_MS, &server, &capacity);

    // An image.
    assert_eq!(
        quick.fetch(&server.url("/img.png"), 1 << 22).ok(),
        Some(body.clone())
    );
    assert!(all_slots_back(&capacity), "image: {:?}", capacity.counts());

    // A refusal by the address policy (no slot is ever taken).
    let not_allowlisted = server.url("/img.png").replace("127.0.0.1", "127.0.0.2");
    let refused = quick
        .fetch(&not_allowlisted, 1 << 22)
        .expect_err("not allowlisted");
    assert_eq!(refused.code(), "image_url_refused", "{refused}");
    assert_eq!(capacity.counts(), FetchCounts::default());

    // A failure status.
    let missing = quick
        .fetch(&server.url("/missing.png"), 1 << 22)
        .expect_err("404");
    assert!(missing.to_string().contains("status 404"), "{missing}");
    assert!(all_slots_back(&capacity), "status: {:?}", capacity.counts());

    // The deadline.
    let timed_out = fetcher(300, &server, &capacity)
        .fetch(&server.url("/stall.png"), 1 << 22)
        .expect_err("too slow");
    assert!(
        timed_out.to_string().contains("timed out after 300 ms"),
        "{timed_out}"
    );
    assert!(
        all_slots_back(&capacity),
        "deadline: {:?}",
        capacity.counts()
    );

    // An abandoned request.
    let request = Arc::new(Request::default());
    let shared = Arc::new(fetcher(LONG_DEADLINE_MS, &server, &capacity));
    let stalled = fetch_for(&shared, &request, server.url("/stall.png"));
    assert!(wait_until(Duration::from_secs(10), || capacity
        .counts()
        .in_flight
        == 1));
    request.abandon();
    assert!(stalled
        .join()
        .expect("the fetching thread finished")
        .is_err());
    assert!(
        all_slots_back(&capacity),
        "abandoned: {:?}",
        capacity.counts()
    );

    // A transfer that panics on the worker (a name goes through the
    // resolver hook, which panics there).
    let lookup: Arc<dyn AddressLookup> = Arc::new(PanickingLookup);
    let settings = ImageFetchSettings::resolve(Some(LONG_DEADLINE_MS), None, &[], None)
        .expect("valid settings");
    let panicking = ImageUrlFetcher::with_parts(settings, lookup, classify_address)
        .with_capacity(Arc::clone(&capacity));
    let failed = panicking
        .fetch("http://images.example/img.png", 1 << 22)
        .expect_err("the transfer unwound");
    assert_eq!(failed.code(), "image_url_fetch_failed", "{failed}");
    assert!(all_slots_back(&capacity), "panic: {:?}", capacity.counts());

    // A waiter whose probe panics: it gives its waiting place back.
    let tight = FetchCapacity::new(1, 1);
    let tight_fetcher = Arc::new(fetcher(LONG_DEADLINE_MS, &server, &tight));
    let holder_request = Arc::new(Request::default());
    let holder = fetch_for(&tight_fetcher, &holder_request, server.url("/stall.png"));
    assert!(wait_until(Duration::from_secs(10), || tight
        .counts()
        .in_flight
        == 1));
    let probe: Arc<dyn ImageFetchObserver> = Arc::new(FailingProbe::default());
    let waiter = {
        let tight_fetcher = Arc::clone(&tight_fetcher);
        let url = server.url("/waiting.png");
        std::thread::spawn(move || {
            CurrentFetch::scope(probe, || tight_fetcher.fetch(&url, 1 << 22))
        })
    };
    assert!(waiter.join().is_err(), "the probe unwound the waiter");
    assert_eq!(tight.counts().waiting, 0, "the waiting place came back");
    holder_request.abandon();
    let _ = holder.join();
    assert!(all_slots_back(&tight), "{:?}", tight.counts());
}

/// Every fetcher a command builds shares the one process-wide bound.
#[test]
fn fetchers_built_by_a_command_share_the_process_bound() {
    let settings = || {
        ImageFetchSettings::resolve(Some(LONG_DEADLINE_MS), None, &[], None)
            .expect("valid settings")
    };
    let one = ImageUrlFetcher::new(settings());
    let other = ImageUrlFetcher::new(settings());
    assert!(Arc::ptr_eq(one.capacity(), other.capacity()));
    assert_eq!(one.capacity().in_flight_limit(), MAX_FETCHES_IN_FLIGHT);
    assert_eq!(one.capacity().waiting_limit(), MAX_FETCHES_WAITING);
    assert!(
        one.describe().contains(&format!(
            "at most {MAX_FETCHES_IN_FLIGHT} fetches in flight and {MAX_FETCHES_WAITING} waiting"
        )),
        "{}",
        one.describe()
    );
}
