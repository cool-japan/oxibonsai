//! How many remote-image fetches the process runs and queues at once.
//!
//! A fetch is driven by request input — a server answers any client's
//! `image_url` — so what it holds must not be: each transfer holds a socket
//! and up to the encoded-image cap (32 MiB) of body, and each fetch waiting
//! to start holds the thread that asked for it. The server's concurrency
//! limit does not count them (a request whose client went away no longer
//! holds its admission permit), so the fetcher bounds them itself, for the
//! whole process: at most [`MAX_FETCHES_IN_FLIGHT`] transfers run at once and
//! at most [`MAX_FETCHES_WAITING`] more wait for one of those slots; a fetch
//! beyond both is refused at once, before any connection is opened, and a
//! server answers its request with a retryable overload (`503`,
//! `Retry-After`).
//!
//! The bound is a compile-time constant on purpose: nothing a request sends
//! can raise it, and the operator's lever is the server's own admission and
//! rate limits (with remote fetching on, set the rate limit).
//!
//! A slot is held by the fetch's transfer itself (the worker's task), not by
//! the thread that waits for the answer: it is released when the transfer
//! ends however it ends — an image, a refusal, the deadline, the request
//! being abandoned, a panic — so the counts always describe the sockets that
//! really are open.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// Transfers the process runs at once. Eight covers the default admission
/// limit of a two-replica server (four requests per replica, and a request
/// fetches its images one after another, so it holds at most one transfer),
/// while bounding the body buffers of transfers in flight to 8 x 32 MiB.
pub(crate) const MAX_FETCHES_IN_FLIGHT: usize = 8;

/// Fetches that wait for a transfer slot. As many again as run: a burst of
/// requests queues briefly instead of failing, but no more than this many
/// callers are ever parked on the fetcher. A waiting fetch holds no socket
/// and no body, and gives up at its per-image deadline or when its request
/// is abandoned.
pub(crate) const MAX_FETCHES_WAITING: usize = 8;

/// How often a fetch that waits — for a slot, or for its transfer's answer —
/// checks whether its request was abandoned: the bound on how long a fetch
/// whose client has gone keeps its slot (and its connection).
pub(crate) const ABANDON_POLL: Duration = Duration::from_millis(20);

/// What a [`FetchCapacity`] counts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FetchCounts {
    /// Transfers holding a slot.
    pub(crate) in_flight: usize,
    /// Fetches waiting for one.
    pub(crate) waiting: usize,
}

/// Why a fetch got no transfer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Every slot is taken and the waiting room is full: refused at once.
    Overloaded,
    /// The fetch's request was abandoned while it waited.
    Abandoned,
    /// The per-image deadline passed while it waited.
    TimedOut,
}

/// The transfer slots of the process (see the module docs). Production
/// fetchers share [`FetchCapacity::process`]; a test gives a fetcher its own.
#[derive(Debug)]
pub(crate) struct FetchCapacity {
    in_flight_limit: usize,
    waiting_limit: usize,
    counts: Mutex<FetchCounts>,
    /// Signalled whenever a slot is released.
    freed: Condvar,
}

impl FetchCapacity {
    /// A bound of `in_flight_limit` transfers plus `waiting_limit` waiting
    /// fetches.
    pub(crate) fn new(in_flight_limit: usize, waiting_limit: usize) -> Arc<Self> {
        Arc::new(Self {
            in_flight_limit,
            waiting_limit,
            counts: Mutex::new(FetchCounts::default()),
            freed: Condvar::new(),
        })
    }

    /// The process-wide bound every fetcher a command builds shares:
    /// [`MAX_FETCHES_IN_FLIGHT`] and [`MAX_FETCHES_WAITING`].
    pub(crate) fn process() -> Arc<Self> {
        static PROCESS: OnceLock<Arc<FetchCapacity>> = OnceLock::new();
        Arc::clone(PROCESS.get_or_init(|| Self::new(MAX_FETCHES_IN_FLIGHT, MAX_FETCHES_WAITING)))
    }

    /// Transfers allowed at once.
    pub(crate) fn in_flight_limit(&self) -> usize {
        self.in_flight_limit
    }

    /// Fetches allowed to wait for a slot.
    pub(crate) fn waiting_limit(&self) -> usize {
        self.waiting_limit
    }

    /// What is held right now.
    pub(crate) fn counts(&self) -> FetchCounts {
        *self.lock()
    }

    /// The counts. They are plain numbers, so a poisoned lock is recovered
    /// rather than propagated.
    fn lock(&self) -> MutexGuard<'_, FetchCounts> {
        self.counts.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A transfer slot for one fetch: at once when one is free; else, while
    /// the waiting room has space, as soon as one frees up — giving up when
    /// `abandoned` says the fetch's request is gone (checked every
    /// [`ABANDON_POLL`], never with the lock held) or at `deadline`; else
    /// [`Refusal::Overloaded`] at once.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] that kept the fetch from a slot.
    pub(crate) fn admit(
        self: &Arc<Self>,
        abandoned: &dyn Fn() -> bool,
        deadline: Instant,
    ) -> Result<InFlightPermit, Refusal> {
        {
            let mut counts = self.lock();
            if counts.in_flight < self.in_flight_limit {
                counts.in_flight += 1;
                return Ok(InFlightPermit {
                    capacity: Arc::clone(self),
                });
            }
            if counts.waiting >= self.waiting_limit {
                return Err(Refusal::Overloaded);
            }
            counts.waiting += 1;
        }
        let waiting = WaitingSlot {
            capacity: self,
            held: true,
        };
        loop {
            if abandoned() {
                return Err(Refusal::Abandoned);
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(Refusal::TimedOut);
            }
            let mut counts = self.lock();
            if counts.in_flight < self.in_flight_limit {
                counts.in_flight += 1;
                counts.waiting = counts.waiting.saturating_sub(1);
                drop(counts);
                waiting.converted();
                return Ok(InFlightPermit {
                    capacity: Arc::clone(self),
                });
            }
            let pause = ABANDON_POLL.min(deadline.saturating_duration_since(now));
            let (counts, _) = self
                .freed
                .wait_timeout(counts, pause)
                .unwrap_or_else(PoisonError::into_inner);
            drop(counts);
        }
    }

    /// Release one transfer slot.
    fn release_in_flight(&self) {
        let mut counts = self.lock();
        counts.in_flight = counts.in_flight.saturating_sub(1);
        drop(counts);
        self.freed.notify_all();
    }

    /// Release one waiting place.
    fn release_waiting(&self) {
        let mut counts = self.lock();
        counts.waiting = counts.waiting.saturating_sub(1);
    }
}

/// One fetch's transfer slot: released when dropped (see the module docs for
/// who holds it).
#[derive(Debug)]
pub(crate) struct InFlightPermit {
    capacity: Arc<FetchCapacity>,
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.capacity.release_in_flight();
    }
}

/// A place in the waiting room, released when dropped unless the fetch moved
/// on to a transfer slot — so a waiter that unwinds (a probe that panics)
/// gives its place back too.
struct WaitingSlot<'a> {
    capacity: &'a FetchCapacity,
    held: bool,
}

impl WaitingSlot<'_> {
    /// The fetch got a transfer slot; the waiting count was already moved
    /// over under the lock.
    fn converted(mut self) {
        self.held = false;
    }
}

impl Drop for WaitingSlot<'_> {
    fn drop(&mut self) {
        if self.held {
            self.capacity.release_waiting();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn never() -> bool {
        false
    }

    fn far() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    #[test]
    fn slots_are_granted_up_to_the_limit_and_released_on_drop() {
        let capacity = FetchCapacity::new(2, 0);
        let first = capacity.admit(&never, far()).expect("a free slot");
        let second = capacity.admit(&never, far()).expect("a free slot");
        assert_eq!(
            capacity.counts(),
            FetchCounts {
                in_flight: 2,
                waiting: 0
            }
        );
        assert_eq!(
            capacity.admit(&never, far()).err(),
            Some(Refusal::Overloaded),
            "no slot and no waiting room: refused at once"
        );
        drop(first);
        drop(second);
        assert_eq!(capacity.counts(), FetchCounts::default());
    }

    #[test]
    fn a_waiter_gets_the_slot_a_transfer_releases() {
        let capacity = FetchCapacity::new(1, 1);
        let held = capacity.admit(&never, far()).expect("a free slot");
        let waiter = {
            let capacity = Arc::clone(&capacity);
            std::thread::spawn(move || capacity.admit(&never, far()).map(drop))
        };
        let start = Instant::now();
        while capacity.counts().waiting == 0 && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(capacity.counts().waiting, 1);
        assert_eq!(
            capacity.admit(&never, far()).err(),
            Some(Refusal::Overloaded),
            "the waiting room is full"
        );
        drop(held);
        assert_eq!(waiter.join().ok(), Some(Ok(())));
        assert_eq!(capacity.counts(), FetchCounts::default());
    }

    #[test]
    fn a_waiter_gives_up_when_abandoned_or_at_its_deadline() {
        let capacity = FetchCapacity::new(1, 2);
        let _held = capacity.admit(&never, far()).expect("a free slot");

        let gone = AtomicBool::new(false);
        let abandoned = || gone.load(Ordering::SeqCst);
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| capacity.admit(&abandoned, far()).err());
            let start = Instant::now();
            while capacity.counts().waiting == 0 && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            gone.store(true, Ordering::SeqCst);
            assert_eq!(waiter.join().ok().flatten(), Some(Refusal::Abandoned));
        });
        assert_eq!(capacity.counts().waiting, 0);

        let soon = Instant::now() + Duration::from_millis(50);
        assert_eq!(capacity.admit(&never, soon).err(), Some(Refusal::TimedOut));
        assert_eq!(
            capacity.counts(),
            FetchCounts {
                in_flight: 1,
                waiting: 0
            }
        );
    }

    #[test]
    fn a_waiter_that_unwinds_gives_its_place_back() {
        let capacity = FetchCapacity::new(1, 1);
        let _held = capacity.admit(&never, far()).expect("a free slot");
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let probe = || -> bool { panic!("a probe that fails") };
            capacity.admit(&probe, far()).map(drop)
        }));
        assert!(unwound.is_err());
        assert_eq!(
            capacity.counts(),
            FetchCounts {
                in_flight: 1,
                waiting: 0
            }
        );
    }

    #[test]
    fn every_fetcher_of_the_process_shares_one_bound() {
        let one = FetchCapacity::process();
        let other = FetchCapacity::process();
        assert!(Arc::ptr_eq(&one, &other));
        assert_eq!(one.in_flight_limit(), MAX_FETCHES_IN_FLIGHT);
        assert_eq!(one.waiting_limit(), MAX_FETCHES_WAITING);
    }
}
