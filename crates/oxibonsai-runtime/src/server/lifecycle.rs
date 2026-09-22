//! Server lifecycle: bind, serve, graceful shutdown, drain deadline.
//!
//! Findings `SV-29` / `deps-11`: `shutdown_signal()` called
//! `.expect("failed to install Ctrl+C handler")` and
//! `.expect("failed to install SIGTERM handler")` in production code. Signal
//! registration fails on environment conditions (a restricted sandbox, a
//! masked signal, fd exhaustion) — not a proven invariant — so the process
//! whose *graceful shutdown* this is panicked instead. And
//! `serve_with_shutdown` handed the signal to
//! `axum::serve(..).with_graceful_shutdown(..)` with **no deadline**, so one
//! un-drained SSE connection (see `sec-08`) kept the drain open forever.
//!
//! The fixes here:
//!
//! * [`install_shutdown_signals`] registers SIGTERM eagerly and **propagates**
//!   the error, so a caller can refuse to start;
//! * [`shutdown_signal`] keeps its signature (both serve binaries pass it
//!   straight to `serve_with_shutdown`) and never panics: a handler that cannot
//!   be installed becomes `std::future::pending()` *inside its own select arm*,
//!   so a broken Ctrl+C still leaves SIGTERM working and vice versa;
//! * [`serve_with_shutdown_deadline`] bounds the drain, and
//!   [`serve_with_shutdown`] delegates to it with
//!   [`DEFAULT_DRAIN_DEADLINE`].

use std::future::Future;
use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;

use crate::engine::InferenceEngine;
use crate::metrics::InferenceMetrics;
use crate::tokenizer_bridge::TokenizerBridge;
use std::sync::Arc;

/// How long [`serve_with_shutdown`] waits for in-flight connections to drain
/// after the shutdown signal before closing what is left.
pub const DEFAULT_DRAIN_DEADLINE: Duration = Duration::from_secs(30);

/// Start the server with graceful shutdown and the default drain deadline.
///
/// Binds `addr`, serves `router` with connect-info wiring (so the rate
/// limiter's `MaybePeerAddr` extractor sees the real client address), and shuts
/// down when `shutdown_signal` completes, giving in-flight requests
/// [`DEFAULT_DRAIN_DEADLINE`] to finish.
pub async fn serve_with_shutdown(
    router: Router,
    addr: SocketAddr,
    shutdown_signal: impl Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    serve_with_shutdown_deadline(router, addr, shutdown_signal, Some(DEFAULT_DRAIN_DEADLINE)).await
}

/// Start the server with an explicit drain deadline.
///
/// `drain_deadline` of `None` restores the unbounded behaviour (wait for every
/// connection to close, however long that takes).
pub async fn serve_with_shutdown_deadline(
    router: Router,
    addr: SocketAddr,
    shutdown_signal: impl Future<Output = ()> + Send + 'static,
    drain_deadline: Option<Duration>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    tracing::info!(addr = %local_addr, "server listening");

    // Fires when the shutdown signal is observed, so the drain timer starts at
    // the signal rather than at startup.
    let (drain_tx, drain_rx) = tokio::sync::oneshot::channel::<()>();
    let signal = async move {
        shutdown_signal.await;
        let _ = drain_tx.send(());
    };

    // Serve with connect-info so the `MaybePeerAddr` extractor in `middleware`
    // can see the real client socket address. Without this, the rate limiter's
    // `extract_client_id` falls back to a single shared "unknown" bucket for
    // every direct (non-proxied) client (findings `serve-api-07` /
    // `security-05`).
    let server = axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(signal);

    match drain_deadline {
        None => {
            server.await?;
            tracing::info!("server shut down gracefully");
        }
        Some(deadline) => {
            let drain_guard = async move {
                // No signal observed (sender dropped) means no drain to bound.
                if drain_rx.await.is_err() {
                    std::future::pending::<()>().await;
                }
                tokio::time::sleep(deadline).await;
            };
            tokio::select! {
                result = server => {
                    result?;
                    tracing::info!("server shut down gracefully");
                }
                () = drain_guard => {
                    tracing::warn!(
                        drain_deadline_secs = deadline.as_secs_f64(),
                        "graceful-shutdown drain deadline exceeded; closing remaining connections"
                    );
                }
            }
        }
    }

    Ok(())
}

/// Install the SIGTERM/Ctrl+C handlers, returning the shutdown future.
///
/// SIGTERM registration happens eagerly so its failure is reported to the
/// caller (which should refuse to start) instead of panicking later. Ctrl+C
/// cannot be pre-registered by tokio; if awaiting it fails, that select arm
/// becomes pending and SIGTERM still works.
///
/// # Errors
///
/// Returns the OS error when the SIGTERM handler cannot be installed (Unix
/// only; on other platforms this never fails).
pub fn install_shutdown_signals() -> std::io::Result<impl Future<Output = ()> + Send + 'static> {
    #[cfg(unix)]
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    Ok(async move {
        let ctrl_c = async {
            match tokio::signal::ctrl_c().await {
                Ok(()) => tracing::info!("received Ctrl+C, initiating shutdown"),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Ctrl+C handler unavailable; relying on SIGTERM for shutdown"
                    );
                    // Never resolve: a broken Ctrl+C must not look like a
                    // shutdown request, and must not disable the other arm.
                    std::future::pending::<()>().await;
                }
            }
        };

        #[cfg(unix)]
        let terminate = async {
            terminate.recv().await;
            tracing::info!("received SIGTERM, initiating shutdown");
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            () = ctrl_c => {}
            () = terminate => {}
        }
    })
}

/// Create a shutdown signal that responds to SIGTERM and SIGINT (Ctrl+C).
///
/// Completes when either signal is received. Never panics: when the SIGTERM
/// handler cannot be installed the future falls back to Ctrl+C only, and when
/// neither handler is available it never completes (the honest outcome — there
/// is no signal to shut down on) while logging the reason.
///
/// Prefer [`install_shutdown_signals`] in new code: it reports the failure
/// instead of degrading silently.
pub async fn shutdown_signal() {
    match install_shutdown_signals() {
        Ok(signals) => signals.await,
        Err(e) => {
            tracing::error!(
                error = %e,
                "failed to install the SIGTERM handler; falling back to Ctrl+C only"
            );
            match tokio::signal::ctrl_c().await {
                Ok(()) => tracing::info!("received Ctrl+C, initiating shutdown"),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "no shutdown signal handler could be installed; \
                         the server will run until the process is killed"
                    );
                    std::future::pending::<()>().await;
                }
            }
        }
    }
}

/// Create the full server setup: router + graceful shutdown future.
///
/// Returns a future that runs the server until a shutdown signal is received.
pub async fn create_server(
    engine: InferenceEngine<'static>,
    tokenizer: Option<TokenizerBridge>,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let metrics = Arc::new(InferenceMetrics::new());
    let router = crate::server::create_router_with_metrics(engine, tokenizer, metrics);
    serve_with_shutdown(router, addr, shutdown_signal()).await
}

// ─── Request queue depth tracking ──────────────────────────────────────

/// Server configuration with request management.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Maximum number of queued requests before rejecting new ones.
    pub max_queue_depth: usize,
    /// Request timeout in seconds.
    pub request_timeout_seconds: u64,
    /// Address to bind to.
    pub bind_addr: SocketAddr,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 128,
            request_timeout_seconds: 60,
            bind_addr: SocketAddr::from(([127, 0, 0, 1], 8080)),
        }
    }
}

/// Request queue depth tracker.
///
/// Thread-safe counter for tracking how many requests are currently
/// queued or in-flight. Used to implement backpressure.
pub struct QueueDepthTracker {
    current: std::sync::atomic::AtomicUsize,
    max_depth: usize,
}

impl QueueDepthTracker {
    /// Create a new tracker with the given maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            current: std::sync::atomic::AtomicUsize::new(0),
            max_depth: max_depth.max(1),
        }
    }

    /// Try to acquire a slot. Returns `true` if successful, `false` if queue is full.
    pub fn try_acquire(&self) -> bool {
        let current = self.current.load(std::sync::atomic::Ordering::Relaxed);
        if current >= self.max_depth {
            return false;
        }
        // CAS loop for correctness under contention
        self.current
            .compare_exchange(
                current,
                current + 1,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Release a slot.
    pub fn release(&self) {
        self.current
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }

    /// Current queue depth.
    pub fn depth(&self) -> usize {
        self.current.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Maximum allowed depth.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Whether the queue has capacity for more requests.
    pub fn has_capacity(&self) -> bool {
        self.depth() < self.max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ServerConfig ──

    #[test]
    fn server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_queue_depth, 128);
        assert_eq!(config.request_timeout_seconds, 60);
        assert_eq!(config.bind_addr, SocketAddr::from(([127, 0, 0, 1], 8080)));
    }

    // ── QueueDepthTracker ──

    #[test]
    fn queue_depth_tracker_basic() {
        let tracker = QueueDepthTracker::new(3);
        assert_eq!(tracker.depth(), 0);
        assert_eq!(tracker.max_depth(), 3);
        assert!(tracker.has_capacity());

        assert!(tracker.try_acquire());
        assert_eq!(tracker.depth(), 1);
        assert!(tracker.try_acquire());
        assert_eq!(tracker.depth(), 2);
        assert!(tracker.try_acquire());
        assert_eq!(tracker.depth(), 3);
        assert!(!tracker.has_capacity());

        // Should fail when full
        assert!(!tracker.try_acquire());

        tracker.release();
        assert_eq!(tracker.depth(), 2);
        assert!(tracker.has_capacity());
        assert!(tracker.try_acquire());
    }

    #[test]
    fn queue_depth_tracker_min_capacity() {
        let tracker = QueueDepthTracker::new(0);
        assert_eq!(tracker.max_depth(), 1);
        assert!(tracker.try_acquire());
        assert!(!tracker.try_acquire());
    }

    // ── SV-29 / deps-11: shutdown ──

    #[test]
    fn shutdown_signal_handlers_install_without_panicking() {
        // The point of SV-29: installation is fallible and must be reported,
        // not `expect()`ed. On a normal host it succeeds.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let _guard = rt.enter();
        assert!(
            install_shutdown_signals().is_ok(),
            "signal installation must succeed on this platform"
        );
    }

    #[tokio::test]
    async fn serve_returns_when_the_shutdown_signal_fires() {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let router = Router::new().route("/health", axum::routing::get(|| async { "ok" }));
        let addr = SocketAddr::from(([127, 0, 0, 1], 0));
        let server = tokio::spawn(serve_with_shutdown_deadline(
            router,
            addr,
            async move {
                let _ = rx.await;
            },
            Some(Duration::from_secs(2)),
        ));

        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = tx.send(());

        let result = tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("server must stop after the shutdown signal")
            .expect("server task");
        assert!(result.is_ok(), "serve returned an error: {result:?}");
    }

    #[tokio::test]
    async fn drain_deadline_closes_a_stuck_connection() {
        // A handler that never returns stands in for an un-drained SSE body.
        // Without a drain deadline this serve future would hang forever.
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let router = Router::new().route(
            "/hang",
            axum::routing::get(|| async {
                std::future::pending::<()>().await;
                "unreachable"
            }),
        );
        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        drop(listener);

        let server = tokio::spawn(serve_with_shutdown_deadline(
            router,
            addr,
            async move {
                let _ = rx.await;
            },
            Some(Duration::from_millis(200)),
        ));

        // Give the server a moment to bind, then open a request that hangs.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let hang = tokio::spawn(async move {
            if let Ok(mut stream) = tokio::net::TcpStream::connect(addr).await {
                use tokio::io::AsyncWriteExt;
                let _ = stream
                    .write_all(b"GET /hang HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    .await;
                let mut buf = [0u8; 16];
                use tokio::io::AsyncReadExt;
                let _ = stream.read(&mut buf).await;
            }
        });

        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = tx.send(());

        let result = tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("the drain deadline must end the wait")
            .expect("server task");
        assert!(result.is_ok(), "serve returned an error: {result:?}");
        hang.abort();
    }
}
