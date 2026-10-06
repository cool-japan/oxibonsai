//! The fetcher's own thread and runtime (see the parent module, "Threads").
//!
//! One thread per fetcher runs a current-thread tokio runtime that is never
//! the caller's: a fetch is a job sent over a channel (a non-blocking send,
//! safe from any thread or runtime), the worker runs each job as a task under
//! what is left of the per-image deadline, and the caller waits for the
//! answer on a plain `std` channel with a bounded `recv_timeout`. Nothing in
//! the caller's path is a tokio primitive that needs a runtime, so no runtime
//! the caller is in (a multi-thread worker, a current-thread runtime, none)
//! can make it panic or deadlock, and a busy caller runtime cannot stretch
//! the deadline.
//!
//! **Abandonment.** The caller waits in short steps
//! ([`super::capacity::ABANDON_POLL`]) and asks, between them, whether the
//! fetch's request was abandoned (its client went away, or its deadline
//! expired). When it was — or when the caller stops waiting for any other
//! reason — it closes the job's abort channel, and the worker drops the
//! transfer: the HTTP client's future goes, and with it the connection. A
//! fetch whose client has gone therefore ends within one poll interval
//! instead of running on to its own per-image deadline.
//!
//! **Bound.** Every job carries the transfer slot its fetch was admitted with
//! ([`super::capacity::InFlightPermit`]); the task holds it until the
//! transfer future is dropped, so the queue never holds more jobs than there
//! are slots, and a slot is released exactly when its socket is.

use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use oxibonsai_model::vision::remote::{RemoteFetchFailure, RemoteImageUrl};

use super::capacity::{InFlightPermit, ABANDON_POLL};
use super::transfer::{self, FetchReport};
use super::FetchContext;

/// How much longer than the per-image deadline a caller waits for the
/// worker's answer before it gives up on its own. The worker answers at the
/// deadline; the margin only covers the hand-over between the threads.
const ANSWER_GRACE: Duration = Duration::from_secs(2);

/// One fetch's per-image deadline: the instant it runs out, and its length
/// (what a timeout names).
#[derive(Debug, Clone, Copy)]
pub(super) struct FetchDeadline {
    /// When the fetch must be over.
    pub(super) at: Instant,
    /// The per-image deadline it was started with.
    pub(super) timeout: Duration,
}

/// One fetch handed to the worker.
struct Job {
    url: RemoteImageUrl,
    max_bytes: usize,
    /// What is left of the per-image deadline for the transfer.
    budget: Duration,
    reply: SyncSender<FetchReport>,
    /// Closed by the caller once nobody waits for the answer: the worker then
    /// drops the transfer, closing its connection.
    abort: tokio::sync::oneshot::Receiver<()>,
    /// The fetch's transfer slot, held until the transfer future is dropped.
    permit: InFlightPermit,
}

/// The handle a fetcher keeps: dropping it (with the fetcher) closes the
/// queue, which ends the worker thread and its runtime.
pub(super) struct Worker {
    jobs: tokio::sync::mpsc::UnboundedSender<Job>,
}

impl Worker {
    /// Start the worker thread.
    ///
    /// # Errors
    ///
    /// The runtime or the thread could not be created.
    pub(super) fn spawn(context: Arc<FetchContext>) -> std::io::Result<Self> {
        let (jobs, mut queue) = tokio::sync::mpsc::unbounded_channel::<Job>();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        std::thread::Builder::new()
            .name("oxibonsai-image-fetch".to_string())
            .spawn(move || {
                // The HTTP client stack below logs connection details (the
                // socket address of every attempt) at debug level. Nothing on
                // this thread is recorded: the fetcher logs each outcome on
                // the calling thread, with the scheme, host and port only.
                let _silent = tracing::dispatcher::set_default(&tracing::Dispatch::none());
                runtime.block_on(async move {
                    while let Some(job) = queue.recv().await {
                        let context = Arc::clone(&context);
                        tokio::spawn(run_job(context, job));
                    }
                });
            })?;
        Ok(Self { jobs })
    }

    /// Fetch `url` on the worker under the slot `permit`, within `deadline`
    /// (the transfer gets what is left of it), and wait for the answer: at
    /// most until the deadline plus a short grace, and only while
    /// `abandoned` says the fetch's request is still wanted (see the module
    /// docs).
    pub(super) fn run(
        &self,
        url: RemoteImageUrl,
        max_bytes: usize,
        deadline: FetchDeadline,
        permit: InFlightPermit,
        abandoned: &dyn Fn() -> bool,
    ) -> FetchReport {
        let FetchDeadline {
            at: deadline,
            timeout,
        } = deadline;
        let origin = url.origin();
        let (reply, answer) = sync_channel(1);
        // Dropped on every way out of this function: a transfer still running
        // then is dropped by the worker.
        let (abort, abort_rx) = tokio::sync::oneshot::channel::<()>();
        let job = Job {
            url,
            max_bytes,
            budget: deadline.saturating_duration_since(Instant::now()),
            reply,
            abort: abort_rx,
            permit,
        };
        if let Err(refused) = self.jobs.send(job) {
            // The job — and its slot — comes back inside the error and is
            // released here.
            drop(refused);
            return FetchReport::failed(
                origin,
                RemoteFetchFailure::Unavailable {
                    reason: "its fetch thread has stopped".to_string(),
                },
            );
        }
        let give_up_at = deadline + ANSWER_GRACE;
        let outcome = loop {
            let now = Instant::now();
            if now >= give_up_at {
                break Err(RemoteFetchFailure::TimedOut {
                    ms: transfer::millis(timeout),
                });
            }
            match answer.recv_timeout(ABANDON_POLL.min(give_up_at - now)) {
                Ok(report) => break Ok(report),
                Err(RecvTimeoutError::Timeout) => {
                    if abandoned() {
                        break Err(RemoteFetchFailure::Abandoned);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    break Err(RemoteFetchFailure::Unavailable {
                        reason: "its fetch thread stopped during the fetch".to_string(),
                    });
                }
            }
        };
        // Nobody waits any more: a transfer still running is dropped by the
        // worker (its connection closes).
        drop(abort);
        match outcome {
            Ok(report) => report,
            Err(failure) => FetchReport::failed(origin, failure),
        }
    }
}

/// One job on the worker: the transfer, unless the caller stops waiting
/// first. The slot is released when this task ends — after the transfer
/// future is gone, however it went.
async fn run_job(context: Arc<FetchContext>, job: Job) {
    let Job {
        url,
        max_bytes,
        budget,
        reply,
        abort,
        permit,
    } = job;
    let _slot = permit;
    tokio::select! {
        report = transfer::fetch(&context, url, max_bytes, budget) => {
            // The caller stops waiting a little after the deadline; an
            // answer nobody waits for is dropped.
            let _ = reply.send(report);
        }
        _ = abort => {
            // The caller is gone (its request was abandoned, or it gave up):
            // the transfer future is dropped here, which closes its
            // connection.
        }
    }
}
