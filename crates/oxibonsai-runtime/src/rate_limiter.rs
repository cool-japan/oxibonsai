//! Token bucket rate limiter with per-client and optional global limits.
//!
//! Provides a thread-safe [`RateLimiter`] that enforces request-per-second
//! limits per client (identified by IP or key) using the token bucket algorithm.
//! An optional global bucket caps aggregate throughput across all clients.
//!
//! # Example
//!
//! ```
//! use oxibonsai_runtime::rate_limiter::{RateLimiter, RateLimitConfig, RateLimitDecision};
//! use std::sync::Arc;
//!
//! let config = RateLimitConfig { rps: 5.0, burst: 10.0, ..Default::default() };
//! let limiter = Arc::new(RateLimiter::new(config));
//!
//! match limiter.check_and_consume("127.0.0.1") {
//!     RateLimitDecision::Allow => println!("request allowed"),
//!     RateLimitDecision::Deny { retry_after_ms } => {
//!         println!("rate limited, retry after {retry_after_ms}ms");
//!     }
//! }
//! ```
//!
//! # Ready-to-mount layer (`server` feature)
//!
//! [`rate_limit_layer`] builds a [`RateLimitLayer`] — a plain [`tower::Layer`]
//! — from a [`RateLimitConfig`], for a binary to mount directly with
//! [`axum::Router::layer`] without going through [`crate::middleware`]'s
//! bundled [`crate::middleware::apply_middleware`] convenience path. This is
//! the seam a binary needs to interleave its own bearer-auth layer between
//! CORS and rate limiting (findings sec-06/sec-07/SV-06/SV-10/cli-18):
//!
//! ```text
//! # required layer order (outermost first, i.e. each `.layer()` call below
//! # is issued in the OPPOSITE order, since axum/tower's later `.layer()`
//! # call becomes the more-outer layer):
//! cors -> auth -> rate-limit -> admission -> routes
//! ```
//!
//! so that a request rejected by auth never consumes a rate-limit token, and
//! a request rejected by the rate limiter never consumes an
//! admission/concurrency permit. Concretely, from a binary (every call below
//! is real, existing public API — `build_routes`, `max_concurrent_requests`
//! / `timeout_ms`, `auth_state` and `bearer_auth` are the *binary's own*
//! routes/admission-config/auth, not part of this crate; `cors_mw` itself is
//! a private implementation detail of [`crate::middleware`], so CORS is
//! mounted through the public [`crate::middleware::apply_middleware`]
//! instead of by name):
//!
//! ```ignore
//! use oxibonsai_runtime::middleware::{apply_middleware, MiddlewareConfig};
//! use oxibonsai_runtime::rate_limiter::rate_limit_layer;
//!
//! let mut router = build_routes(...);
//! router = apply_admission(router, max_concurrent_requests, timeout_ms); // innermost
//! router = router.layer(rate_limit_layer(rate_limit_cfg));               // wraps admission
//! router = router.layer(axum::middleware::from_fn_with_state(auth_state, bearer_auth));
//! // CORS must be outermost. `apply_middleware` with only `cors` set (via
//! // `MiddlewareConfig::none().with_cors(..)`, which leaves logging off and
//! // rate limiting `None`) mounts *exactly* one layer -- CORS -- so this
//! // call is equivalent to a standalone `cors_layer(cors_cfg)`.
//! router = apply_middleware(router, MiddlewareConfig::none().with_cors(cors_cfg));
//! ```
//!
//! [`crate::middleware::apply_middleware`] mounts the *same* [`RateLimitLayer`]
//! internally when [`crate::middleware::MiddlewareConfig::rate_limit`] is
//! set, so both call paths share one implementation and one observable
//! 429/`Retry-After` contract.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ─── TokenBucket ────────────────────────────────────────────────────────────

/// Token bucket for a single client.
///
/// Starts full at `capacity` tokens. Tokens refill at `refill_rate` per second
/// up to `capacity`. Consuming `n` tokens fails if fewer than `n` are available.
struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new full token bucket.
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed_secs = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + self.refill_rate * elapsed_secs).min(self.capacity);
        self.last_refill = now;
    }

    /// Attempt to consume `n` tokens.
    ///
    /// Returns `true` if `n` tokens were available and consumed; `false` if insufficient.
    fn try_consume(&mut self, n: f64) -> bool {
        self.refill();
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Return currently available tokens (after a refill).
    #[allow(dead_code)]
    fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    /// Estimate milliseconds until `n` tokens are available (without consuming).
    ///
    /// Returns 0 if tokens are already available.
    fn ms_until_available(&self, n: f64) -> u64 {
        if self.tokens >= n {
            return 0;
        }
        let deficit = n - self.tokens;
        let secs = deficit / self.refill_rate;
        (secs * 1000.0).ceil() as u64
    }
}

// ─── RateLimitConfig ────────────────────────────────────────────────────────

/// Configuration for the rate limiter.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Steady-state requests per second per client (default: 10.0).
    pub rps: f64,
    /// Burst capacity: maximum tokens a client can accumulate (default: 20.0).
    pub burst: f64,
    /// Maximum number of tracked clients before LRU eviction (default: 10_000).
    pub max_clients: usize,
    /// Evict clients that have been inactive for longer than this duration (default: 300 s).
    pub client_ttl: Duration,
    /// Optional global rate limit across all clients combined.
    pub global_rps: Option<f64>,
    /// IP addresses of upstream reverse proxies trusted to set
    /// `X-Forwarded-For` / `X-Real-IP` on requests they pass through.
    ///
    /// Defaults to empty, meaning **no** proxy is trusted: those headers are
    /// never honored, and the client identity used for rate limiting is
    /// derived from the actual TCP peer address instead (or the literal
    /// `"unknown"` string if the peer address is unavailable to the caller).
    /// This closes the trivial bypass where a client talking directly to the
    /// server sets an arbitrary/rotating `X-Forwarded-For` value to obtain a
    /// fresh token bucket on every request (findings `serve-api-07` /
    /// `security-05`).
    ///
    /// Only add an address here when `oxibonsai-serve` genuinely sits behind
    /// a reverse proxy or load balancer at that address which you control
    /// and which overwrites (rather than blindly appends to) these headers.
    pub trusted_proxies: Vec<std::net::IpAddr>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            rps: 10.0,
            burst: 20.0,
            max_clients: 10_000,
            client_ttl: Duration::from_secs(300),
            global_rps: None,
            trusted_proxies: Vec::new(),
        }
    }
}

impl RateLimitConfig {
    /// Build a config from a requests-per-minute figure — the unit both
    /// `oxibonsai-serve`'s `--rate-limit-rpm` / `rate_limit.rpm` /
    /// `OXIBONSAI_RATE_LIMIT_RPM` and the `oxibonsai serve` CLI's
    /// `OXIBONSAI_RATE_LIMIT_RPM` accept (findings `sec-07` / `RT-34` /
    /// `SV-10`), converted here to the internal per-second rate this
    /// limiter actually enforces so the conversion has exactly one
    /// implementation instead of being duplicated at each call site.
    ///
    /// `burst` is a token count (not a rate) and passes through unchanged —
    /// see [`RateLimitConfig::burst`].
    ///
    /// # Examples
    ///
    /// ```
    /// use oxibonsai_runtime::rate_limiter::RateLimitConfig;
    ///
    /// let cfg = RateLimitConfig::from_rpm(600.0, 50.0);
    /// assert!((cfg.rps - 10.0).abs() < 1e-9);
    /// assert!((cfg.burst - 50.0).abs() < 1e-9);
    /// ```
    pub fn from_rpm(rpm: f64, burst: f64) -> Self {
        Self {
            rps: rpm / 60.0,
            burst,
            ..Self::default()
        }
    }
}

// ─── RateLimitDecision ──────────────────────────────────────────────────────

/// Decision returned by the rate limiter.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitDecision {
    /// The request is within the allowed rate — proceed.
    Allow,
    /// The request exceeds the allowed rate.
    Deny {
        /// Suggested delay in milliseconds before retrying.
        retry_after_ms: u64,
    },
}

impl RateLimitDecision {
    /// Returns `true` if the request is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, RateLimitDecision::Allow)
    }

    /// Returns the retry-after hint in milliseconds, or `None` if the request is allowed.
    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            RateLimitDecision::Deny { retry_after_ms } => Some(*retry_after_ms),
            RateLimitDecision::Allow => None,
        }
    }
}

// ─── RateLimiter ────────────────────────────────────────────────────────────

/// Per-client rate limiter with optional global aggregate limit.
///
/// Thread-safe; intended to be shared via `Arc<RateLimiter>`.
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Map from client_id → (bucket, last_seen).
    clients: Mutex<HashMap<String, (TokenBucket, Instant)>>,
    /// Optional global token bucket shared across all clients.
    global: Option<Mutex<TokenBucket>>,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        let global = config.global_rps.map(|rps| {
            // Global burst is 2× the per-second limit.
            Mutex::new(TokenBucket::new(rps * 2.0, rps))
        });
        Self {
            config,
            clients: Mutex::new(HashMap::new()),
            global,
        }
    }

    /// Check whether a request from `client_id` is within rate limits.
    ///
    /// This is a read-only peek — no token is consumed.
    pub fn check(&self, client_id: &str) -> RateLimitDecision {
        // Check global limit first (read-only: just inspect available tokens).
        if let Some(ref global_mutex) = self.global {
            let global = global_mutex
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if global.tokens < 1.0 {
                let retry_ms = global.ms_until_available(1.0);
                return RateLimitDecision::Deny {
                    retry_after_ms: retry_ms.max(1),
                };
            }
        }

        // Check per-client limit (read-only).
        let mut clients = self
            .clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some((bucket, _last_seen)) = clients.get_mut(client_id) {
            // Peek: refill without consuming.
            bucket.refill();
            if bucket.tokens < 1.0 {
                let retry_ms = bucket.ms_until_available(1.0);
                return RateLimitDecision::Deny {
                    retry_after_ms: retry_ms.max(1),
                };
            }
        }
        // New client or sufficient tokens — allow.
        RateLimitDecision::Allow
    }

    /// Check rate limit and consume one token if allowed.
    ///
    /// Returns [`RateLimitDecision::Allow`] and deducts a token, or
    /// [`RateLimitDecision::Deny`] without modifying any state.
    pub fn check_and_consume(&self, client_id: &str) -> RateLimitDecision {
        // Check and consume from global bucket first.
        if let Some(ref global_mutex) = self.global {
            let mut global = global_mutex
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !global.try_consume(1.0) {
                let retry_ms = global.ms_until_available(1.0);
                return RateLimitDecision::Deny {
                    retry_after_ms: retry_ms.max(1),
                };
            }
        }

        let mut clients = self
            .clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Evict stale entries if at capacity.
        if clients.len() >= self.config.max_clients {
            let ttl = self.config.client_ttl;
            let now = Instant::now();
            clients.retain(|_, (_, last_seen)| now.duration_since(*last_seen) < ttl);
        }

        let bucket = clients.entry(client_id.to_owned()).or_insert_with(|| {
            (
                TokenBucket::new(self.config.burst, self.config.rps),
                Instant::now(),
            )
        });

        let (token_bucket, last_seen) = bucket;
        *last_seen = Instant::now();

        if token_bucket.try_consume(1.0) {
            RateLimitDecision::Allow
        } else {
            let retry_ms = token_bucket.ms_until_available(1.0);
            RateLimitDecision::Deny {
                retry_after_ms: retry_ms.max(1),
            }
        }
    }

    /// Evict clients that have been inactive longer than `client_ttl`.
    pub fn evict_stale(&self) {
        let ttl = self.config.client_ttl;
        let now = Instant::now();
        let mut clients = self
            .clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clients.retain(|_, (_, last_seen)| now.duration_since(*last_seen) < ttl);
    }

    /// Number of currently tracked (active) clients.
    pub fn active_clients(&self) -> usize {
        self.clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Remove a specific client from the tracking map (resets their bucket).
    pub fn reset_client(&self, client_id: &str) {
        self.clients
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(client_id);
    }

    /// The configured trusted-proxy allowlist (see
    /// [`RateLimitConfig::trusted_proxies`]).
    pub fn trusted_proxies(&self) -> &[std::net::IpAddr] {
        &self.config.trusted_proxies
    }

    /// Returns `true` if the global rate limit is currently saturated.
    pub fn is_global_limited(&self) -> bool {
        match &self.global {
            None => false,
            Some(global_mutex) => {
                let global = global_mutex
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                global.tokens < 1.0
            }
        }
    }
}

// ─── Axum middleware helper ──────────────────────────────────────────────────

/// Apply rate limiting in an Axum middleware context.
///
/// Extracts the client ID and delegates to [`RateLimiter::check_and_consume`].
/// Intended to be called from a middleware layer before routing.
pub fn rate_limit_middleware(
    limiter: std::sync::Arc<RateLimiter>,
    client_id: &str,
) -> RateLimitDecision {
    limiter.check_and_consume(client_id)
}

/// Extract a client identifier from HTTP headers, the real TCP peer address,
/// and a trusted-proxy allowlist.
///
/// `X-Forwarded-For` / `X-Real-IP` are client-suppliable and therefore
/// trivially spoofable: a direct (non-reverse-proxied) client can rotate
/// them on every request to obtain a fresh rate-limit bucket each time, or
/// omit them to pile onto a shared fallback bucket with every other
/// header-less client (findings `serve-api-07` / `security-05`). To close
/// that bypass, those headers are honored **only** when `peer_addr` is
/// present *and* is a member of `trusted_proxies` -- i.e. only when the
/// immediate connection really did come from an operator-configured reverse
/// proxy that is expected to set them.
///
/// Priority order:
/// 1. If `peer_addr` is a trusted proxy: `X-Forwarded-For` (first hop), then
///    `X-Real-IP`.
/// 2. The real peer address (`peer_addr`), when known.
/// 3. Fallback string `"unknown"`.
#[cfg(feature = "server")]
pub fn extract_client_id(
    headers: &axum::http::HeaderMap,
    peer_addr: Option<std::net::IpAddr>,
    trusted_proxies: &[std::net::IpAddr],
) -> String {
    let peer_is_trusted_proxy = peer_addr
        .map(|ip| trusted_proxies.contains(&ip))
        .unwrap_or(false);

    if peer_is_trusted_proxy {
        if let Some(id) = forwarded_client_id(headers) {
            return id;
        }
    }

    match peer_addr {
        Some(ip) => ip.to_string(),
        None => "unknown".to_owned(),
    }
}

/// Read a client-supplied forwarding header (`X-Forwarded-For`, then
/// `X-Real-IP`). Only meaningful when the immediate peer is a configured
/// trusted proxy -- see [`extract_client_id`].
#[cfg(feature = "server")]
fn forwarded_client_id(headers: &axum::http::HeaderMap) -> Option<String> {
    // X-Forwarded-For: client, proxy1, proxy2
    if let Some(xff) = headers.get("x-forwarded-for") {
        if let Ok(val) = xff.to_str() {
            let first = val.split(',').next().unwrap_or("").trim();
            if !first.is_empty() {
                return Some(first.to_owned());
            }
        }
    }

    // X-Real-IP
    if let Some(real_ip) = headers.get("x-real-ip") {
        if let Ok(val) = real_ip.to_str() {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }

    None
}

// ─── Ready-to-mount Tower layer (server feature) ─────────────────────────────

/// Build the `429 Too Many Requests` response: a JSON error envelope plus a
/// `Retry-After` header in whole seconds (rounded up, minimum 1 -- a `0`
/// would tell the client to retry immediately, which defeats the point).
#[cfg(feature = "server")]
fn too_many_requests_response(retry_after_ms: u64) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue, StatusCode};
    use axum::response::IntoResponse;

    let retry_secs = (retry_after_ms.saturating_add(999) / 1000).max(1);
    let body = axum::Json(serde_json::json!({
        "error": {
            "message": "rate limit exceeded",
            "type": "rate_limit_error",
            "retry_after_ms": retry_after_ms,
        }
    }));
    let mut response = (StatusCode::TOO_MANY_REQUESTS, body).into_response();
    if let Ok(header_value) = HeaderValue::from_str(&retry_secs.to_string()) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("retry-after"), header_value);
    }
    response
}

/// Resolve the client identifier for `req`: the real TCP peer address, if
/// the router was served via
/// [`axum::routing::Router::into_make_service_with_connect_info`], is
/// honored as `X-Forwarded-For` / `X-Real-IP` only when it is a configured
/// trusted proxy; otherwise the real peer (or the `"unknown"` fallback) is
/// used directly. See [`extract_client_id`].
///
/// Reads the [`axum::extract::connect_info::ConnectInfo`] extension
/// directly -- equivalent to, but without requiring, the
/// [`axum::extract::FromRequestParts`] extractor machinery, since a raw
/// [`tower::Service`] only sees the whole [`axum::extract::Request`] -- and,
/// when that extension is absent, falls back to
/// [`axum::extract::connect_info::MockConnectInfo`] exactly as
/// `ConnectInfo::from_request_parts` itself does internally. That fallback
/// is not merely cosmetic: `MockConnectInfo`'s `Layer` impl inserts a
/// `MockConnectInfo<T>` extension, a *different type* from `ConnectInfo<T>`,
/// so a plain `ConnectInfo<T>` extension lookup alone would silently never
/// see a test's `.layer(MockConnectInfo(addr))` and every mocked-peer test
/// would collapse onto the `"unknown"` bucket.
#[cfg(feature = "server")]
fn client_id_for_request(
    req: &axum::extract::Request,
    trusted_proxies: &[std::net::IpAddr],
) -> String {
    use axum::extract::connect_info::{ConnectInfo, MockConnectInfo};

    let peer_ip = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.ip())
        .or_else(|| {
            req.extensions()
                .get::<MockConnectInfo<std::net::SocketAddr>>()
                .map(|MockConnectInfo(addr)| addr.ip())
        });
    extract_client_id(req.headers(), peer_ip, trusted_proxies)
}

/// Ready-to-mount [`tower::Layer`] enforcing a [`RateLimitConfig`].
///
/// Construct with [`rate_limit_layer`] and mount with
/// [`axum::Router::layer`]. `/health` and `/metrics` are always exempt
/// (liveness and metrics probes must never be throttled). See the module
/// docs ("Ready-to-mount layer") for the required position of this layer
/// relative to CORS, auth, and the admission/concurrency layer.
///
/// This is the exact same enforcement [`crate::middleware::apply_middleware`]
/// installs when [`crate::middleware::MiddlewareConfig::rate_limit`] is
/// `Some` -- both paths construct a [`RateLimitLayer`], so there is one
/// implementation and one observable behavior.
#[cfg(feature = "server")]
#[derive(Clone)]
pub struct RateLimitLayer {
    limiter: std::sync::Arc<RateLimiter>,
}

#[cfg(feature = "server")]
impl<S> tower::Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            limiter: std::sync::Arc::clone(&self.limiter),
        }
    }
}

/// The [`tower::Service`] produced by [`RateLimitLayer`]. Not constructed
/// directly -- see [`rate_limit_layer`].
#[cfg(feature = "server")]
#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    limiter: std::sync::Arc<RateLimiter>,
}

#[cfg(feature = "server")]
impl<S> tower::Service<axum::extract::Request> for RateLimitService<S>
where
    S: tower::Service<axum::extract::Request, Response = axum::response::Response>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = axum::response::Response;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: axum::extract::Request) -> Self::Future {
        let path = req.uri().path();
        if path == "/health" || path == "/metrics" {
            return Box::pin(self.inner.call(req));
        }

        let client_id = client_id_for_request(&req, self.limiter.trusted_proxies());
        match self.limiter.check_and_consume(&client_id) {
            RateLimitDecision::Allow => Box::pin(self.inner.call(req)),
            RateLimitDecision::Deny { retry_after_ms } => Box::pin(std::future::ready(Ok(
                too_many_requests_response(retry_after_ms),
            ))),
        }
    }
}

/// Build a ready-to-mount rate-limiting [`tower::Layer`] from `cfg`.
///
/// Enforces [`RateLimiter::check_and_consume`] per client (identity from
/// [`extract_client_id`]), responding `429 Too Many Requests` with a
/// `Retry-After` header (whole seconds, rounded up, minimum 1) when the
/// client's budget is exhausted; `/health` and `/metrics` are always exempt.
///
/// See the module docs ("Ready-to-mount layer") for the mandatory layer
/// order: mount this **between** the bearer-auth layer and the
/// admission/concurrency-limit layer (`cors -> auth -> rate-limit ->
/// admission`), so an unauthenticated request is rejected before it can
/// consume a rate-limit token, and a rate-limited request is rejected
/// before it can consume an admission permit (cli-18).
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::rate_limiter::{rate_limit_layer, RateLimitConfig};
/// use axum::{routing::get, Router};
///
/// async fn handler() -> &'static str { "ok" }
///
/// let cfg = RateLimitConfig { rps: 100.0, burst: 200.0, ..Default::default() };
/// let router: Router = Router::new()
///     .route("/", get(handler))
///     .layer(rate_limit_layer(cfg));
/// ```
#[cfg(feature = "server")]
pub fn rate_limit_layer(cfg: RateLimitConfig) -> RateLimitLayer {
    RateLimitLayer {
        limiter: std::sync::Arc::new(RateLimiter::new(cfg)),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_token_bucket_initial_full() {
        let mut bucket = TokenBucket::new(10.0, 1.0);
        assert!((bucket.available() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_token_bucket_consume_success() {
        let mut bucket = TokenBucket::new(10.0, 1.0);
        assert!(bucket.try_consume(5.0));
        let remaining = bucket.available();
        assert!((4.9..=5.1).contains(&remaining), "remaining={remaining}");
    }

    #[test]
    fn test_token_bucket_consume_fail_insufficient() {
        let mut bucket = TokenBucket::new(3.0, 0.01); // very slow refill
        assert!(bucket.try_consume(3.0)); // drain
        assert!(!bucket.try_consume(1.0)); // nothing left
    }

    #[test]
    fn test_token_bucket_refills_over_time() {
        let mut bucket = TokenBucket::new(10.0, 1000.0); // 1000 tok/s = refills quickly
        assert!(bucket.try_consume(10.0)); // drain completely
                                           // Sleep briefly and check that tokens have refilled
        thread::sleep(Duration::from_millis(20));
        let available = bucket.available();
        // At 1000 tok/s, 20ms should yield ~20 tokens (capped at 10)
        assert!(
            available > 1.0,
            "bucket should have refilled; got {available}"
        );
    }

    #[test]
    fn test_rate_limiter_allows_first_request() {
        let config = RateLimitConfig {
            rps: 10.0,
            burst: 10.0,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);
        let decision = limiter.check_and_consume("client-1");
        assert_eq!(decision, RateLimitDecision::Allow);
    }

    #[test]
    fn test_rate_limiter_denies_after_burst() {
        let config = RateLimitConfig {
            rps: 1.0,
            burst: 3.0, // only 3 burst tokens
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // First 3 requests should be allowed
        for i in 0..3 {
            let d = limiter.check_and_consume("client-burst");
            assert_eq!(d, RateLimitDecision::Allow, "request {i} should be allowed");
        }

        // 4th request should be denied
        let denied = limiter.check_and_consume("client-burst");
        assert!(
            denied.retry_after_ms().is_some(),
            "4th request should be denied"
        );
    }

    #[test]
    fn test_rate_limiter_different_clients_independent() {
        let config = RateLimitConfig {
            rps: 1.0,
            burst: 1.0,
            ..Default::default()
        };
        let limiter = RateLimiter::new(config);

        // Exhaust client-a
        assert_eq!(
            limiter.check_and_consume("client-a"),
            RateLimitDecision::Allow
        );
        let denied = limiter.check_and_consume("client-a");
        assert!(!denied.is_allowed());

        // client-b should still have its own full bucket
        assert_eq!(
            limiter.check_and_consume("client-b"),
            RateLimitDecision::Allow
        );
    }

    // ─── RateLimitConfig::from_rpm (sec-07 / RT-34 / SV-10) ───────────────

    #[test]
    fn from_rpm_converts_to_the_internal_per_second_rate() {
        let cfg = RateLimitConfig::from_rpm(600.0, 50.0);
        assert!((cfg.rps - 10.0).abs() < 1e-9);
        assert!((cfg.burst - 50.0).abs() < 1e-9);
    }

    #[test]
    fn from_rpm_leaves_the_other_defaults_untouched() {
        let cfg = RateLimitConfig::from_rpm(60.0, 5.0);
        let defaults = RateLimitConfig::default();
        assert_eq!(cfg.max_clients, defaults.max_clients);
        assert_eq!(cfg.client_ttl, defaults.client_ttl);
        assert!(cfg.global_rps.is_none());
        assert!(cfg.trusted_proxies.is_empty());
    }

    #[test]
    fn from_rpm_enforces_the_converted_rate_end_to_end() {
        // 60 rpm == 1 rps; with a burst of 1, a second immediate request
        // (well under a second later) must be denied.
        let limiter = RateLimiter::new(RateLimitConfig::from_rpm(60.0, 1.0));
        assert_eq!(
            limiter.check_and_consume("client"),
            RateLimitDecision::Allow
        );
        assert!(!limiter.check_and_consume("client").is_allowed());
    }

    #[test]
    fn test_rate_limit_decision_is_allowed() {
        assert!(RateLimitDecision::Allow.is_allowed());
        assert_eq!(RateLimitDecision::Allow.retry_after_ms(), None);

        let denied = RateLimitDecision::Deny {
            retry_after_ms: 500,
        };
        assert!(!denied.is_allowed());
        assert_eq!(denied.retry_after_ms(), Some(500));
    }

    /// Regression test for findings `serve-api-07` / `security-05`: when the
    /// immediate peer is *not* a configured trusted proxy (the default), a
    /// client-supplied `X-Forwarded-For` header must be completely ignored
    /// -- otherwise any direct client could rotate the header on every
    /// request to obtain a fresh rate-limit bucket each time.
    // `extract_client_id` is `#[cfg(feature = "server")]`; without it this
    // test cannot compile (it would break `--no-default-features --lib`).
    #[cfg(feature = "server")]
    #[test]
    fn test_extract_client_id_ignores_untrusted_forwarded_header() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;
        use std::net::IpAddr;

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            HeaderValue::from_static("203.0.113.42, 10.0.0.1"),
        );
        let peer: IpAddr = "198.51.100.7".parse().expect("valid ip");
        // No trusted proxies configured -- the spoofed header must be
        // ignored and the real peer address used instead.
        let id = extract_client_id(&headers, Some(peer), &[]);
        assert_eq!(
            id, "198.51.100.7",
            "an untrusted peer's X-Forwarded-For must not override the real peer address"
        );
    }

    /// When the immediate peer *is* a configured trusted proxy, the
    /// forwarded header is honored (the legitimate reverse-proxy case).
    // `extract_client_id` is `#[cfg(feature = "server")]`; without it this
    // test cannot compile (it would break `--no-default-features --lib`).
    #[cfg(feature = "server")]
    #[test]
    fn test_extract_client_id_honors_forwarded_header_from_trusted_proxy() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;
        use std::net::IpAddr;

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            HeaderValue::from_static("203.0.113.42, 10.0.0.1"),
        );
        let peer: IpAddr = "10.0.0.1".parse().expect("valid ip");
        let trusted = [peer];
        let id = extract_client_id(&headers, Some(peer), &trusted);
        assert_eq!(id, "203.0.113.42");
    }

    /// With no trusted proxies and no forwarded headers, a known peer
    /// address is used directly as the client identifier.
    // `extract_client_id` is `#[cfg(feature = "server")]`; without it this
    // test cannot compile (it would break `--no-default-features --lib`).
    #[cfg(feature = "server")]
    #[test]
    fn test_extract_client_id_uses_real_peer_when_no_headers() {
        use axum::http::HeaderMap;
        use std::net::IpAddr;

        let headers = HeaderMap::new();
        let peer: IpAddr = "203.0.113.9".parse().expect("valid ip");
        let id = extract_client_id(&headers, Some(peer), &[]);
        assert_eq!(id, "203.0.113.9");
    }

    /// With neither a known peer address nor headers, fall back to the
    /// literal `"unknown"` string (unchanged legacy behavior for callers
    /// that cannot supply connection info).
    // `extract_client_id` is `#[cfg(feature = "server")]`; without it this
    // test cannot compile (it would break `--no-default-features --lib`).
    #[cfg(feature = "server")]
    #[test]
    fn test_extract_client_id_fallback() {
        use axum::http::HeaderMap;
        let headers = HeaderMap::new();
        let id = extract_client_id(&headers, None, &[]);
        assert_eq!(id, "unknown");
    }

    /// A trusted-proxy allowlist entry that does not match the actual peer
    /// must not grant header trust (exact-match only, no accidental prefix
    /// or subnet matching).
    // `extract_client_id` is `#[cfg(feature = "server")]`; without it this
    // test cannot compile (it would break `--no-default-features --lib`).
    #[cfg(feature = "server")]
    #[test]
    fn test_extract_client_id_trusted_proxies_list_is_exact_match() {
        use axum::http::HeaderMap;
        use axum::http::HeaderValue;
        use std::net::IpAddr;

        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", HeaderValue::from_static("203.0.113.42"));
        let peer: IpAddr = "10.0.0.2".parse().expect("valid ip");
        let other_trusted: IpAddr = "10.0.0.1".parse().expect("valid ip");
        let id = extract_client_id(&headers, Some(peer), &[other_trusted]);
        assert_eq!(
            id, "10.0.0.2",
            "a peer not exactly in the trusted_proxies list must not have its \
             forwarded header honored"
        );
    }

    #[test]
    fn test_rate_limiter_active_clients_tracked() {
        let limiter = RateLimiter::new(RateLimitConfig::default());
        limiter.check_and_consume("alpha");
        limiter.check_and_consume("beta");
        assert_eq!(limiter.active_clients(), 2);
        limiter.reset_client("alpha");
        assert_eq!(limiter.active_clients(), 1);
    }

    #[test]
    fn test_rate_limiter_no_global_limit_by_default() {
        let limiter = RateLimiter::new(RateLimitConfig::default());
        assert!(!limiter.is_global_limited());
    }
}

// ─── `rate_limit_layer` integration tests (server feature) ───────────────────

#[cfg(all(test, feature = "server"))]
mod rate_limit_layer_tests {
    use super::{rate_limit_layer, RateLimitConfig};
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Request, StatusCode};
    use axum::middleware::{self, Next};
    use axum::response::Response;
    use axum::routing::get;
    use axum::Router;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tower::ServiceExt;

    async fn handler() -> &'static str {
        "ok"
    }

    /// Stands in for the admission/concurrency-limit layer's in-flight
    /// permit count: a monotonic counter incremented once per request that
    /// actually reaches it. Unlike an increment-then-decrement gauge, a
    /// bug that let a denied request slip through cannot "self-heal" back
    /// to the pre-request reading before the test observes it -- the
    /// counter would simply read one higher.
    #[derive(Clone, Default)]
    struct AdmissionEntryCounter(Arc<AtomicUsize>);

    async fn admission_stand_in(
        State(counter): State<AdmissionEntryCounter>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        counter.0.fetch_add(1, Ordering::SeqCst);
        next.run(req).await
    }

    /// `rate_limit_layer` returns `429` with a `Retry-After` header once a
    /// client's burst is exhausted, and -- because it never calls the inner
    /// service for a denied request -- a layer nested inside it (standing
    /// in for the admission/concurrency-permit layer, per the mandatory
    /// `rate-limit -> admission` order) never runs for that request
    /// (acceptance criterion: "the concurrency gauge did not move";
    /// cli-18).
    #[tokio::test]
    async fn denied_request_gets_429_and_never_reaches_admission() {
        let counter = AdmissionEntryCounter::default();
        let router = Router::new()
            .route("/", get(handler))
            // innermost: stands in for the admission/concurrency layer.
            .layer(middleware::from_fn_with_state(
                counter.clone(),
                admission_stand_in,
            ))
            // outer: the layer under test.
            .layer(rate_limit_layer(RateLimitConfig {
                rps: 1.0,
                burst: 1.0,
                ..Default::default()
            }));

        let make_req = || {
            Request::builder()
                .uri("/")
                .header("x-forwarded-for", "203.0.113.5")
                .body(Body::empty())
                .expect("request")
        };

        // First request: within burst -> allowed, reaches the admission
        // stand-in exactly once.
        let resp1 = router.clone().oneshot(make_req()).await.expect("resp1");
        assert_eq!(resp1.status(), StatusCode::OK);
        assert_eq!(
            counter.0.load(Ordering::SeqCst),
            1,
            "the allowed request must reach the admission stand-in"
        );

        // Second request from the same client: burst exhausted -> 429, and
        // the gauge must read the exact same value as before this call --
        // it must not move at all.
        let resp2 = router.clone().oneshot(make_req()).await.expect("resp2");
        assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(
            resp2.headers().get("retry-after").is_some(),
            "a 429 response must carry a Retry-After header"
        );
        assert_eq!(
            counter.0.load(Ordering::SeqCst),
            1,
            "a rate-limited request must never reach the admission/concurrency \
             layer -- the gauge must not have moved"
        );

        let retry_after = resp2
            .headers()
            .get("retry-after")
            .expect("retry-after present")
            .to_str()
            .expect("retry-after is ASCII")
            .parse::<u64>()
            .expect("retry-after is an integer number of seconds");
        assert!(retry_after >= 1, "Retry-After must be at least 1 second");

        let body = axum::body::to_bytes(resp2.into_body(), usize::MAX)
            .await
            .expect("read body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("parse json");
        assert_eq!(json["error"]["type"], "rate_limit_error");
    }

    /// `/health` and `/metrics` are always exempt, matching
    /// [`crate::middleware::apply_middleware`]'s historical rate-limit
    /// behavior (both now share this same [`super::RateLimitLayer`]).
    #[tokio::test]
    async fn health_and_metrics_are_exempt_from_rate_limit_layer() {
        let router = Router::new()
            .route("/health", get(handler))
            .route("/metrics", get(handler))
            .layer(rate_limit_layer(RateLimitConfig {
                rps: 1.0,
                burst: 1.0,
                ..Default::default()
            }));

        for _ in 0..5 {
            for path in ["/health", "/metrics"] {
                let resp = router
                    .clone()
                    .oneshot(
                        Request::builder()
                            .uri(path)
                            .body(Body::empty())
                            .expect("request"),
                    )
                    .await
                    .expect("response");
                assert_eq!(
                    resp.status(),
                    StatusCode::OK,
                    "{path} must never be throttled"
                );
            }
        }
    }

    /// [`rate_limit_layer`] is directly mountable with a single `.layer()`
    /// call given only a [`RateLimitConfig`] -- the "ready to mount"
    /// contract it promises (findings sec-07 / SV-10), independent of
    /// [`crate::middleware::MiddlewareConfig`] / `apply_middleware`.
    #[tokio::test]
    async fn mounts_directly_from_just_a_config() {
        let router: Router = Router::new()
            .route("/", get(handler))
            .layer(rate_limit_layer(RateLimitConfig::default()));
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Two requests carrying the same (untrusted, so ignored)
    /// `X-Forwarded-For` collapse onto the shared `"unknown"` bucket and
    /// exhaust it together -- the standalone layer's default client
    /// identity is the real peer, not a client-suppliable header (see
    /// [`extract_client_id`]).
    #[tokio::test]
    async fn same_untrusted_forwarded_header_shares_one_bucket() {
        let router = Router::new()
            .route("/", get(handler))
            .layer(rate_limit_layer(RateLimitConfig {
                rps: 1.0,
                burst: 1.0,
                ..Default::default()
            }));

        let req = |peer: &str| {
            Request::builder()
                .uri("/")
                .header("x-forwarded-for", peer)
                .body(Body::empty())
                .expect("request")
        };

        let resp = router.clone().oneshot(req("9.9.9.9")).await.expect("r1");
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = router.clone().oneshot(req("9.9.9.9")).await.expect("r2");
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the shared \"unknown\" bucket (no trusted ConnectInfo) must now be exhausted"
        );
    }

    /// Distinct *real* peers (via [`MockConnectInfo`], standing in for a
    /// real TCP connection's address) get independent buckets through the
    /// standalone layer, exactly as they do through
    /// [`crate::middleware::apply_middleware`] -- one client's exhausted
    /// budget must never throttle another.
    #[tokio::test]
    async fn distinct_real_peers_get_independent_buckets() {
        use axum::extract::connect_info::MockConnectInfo;
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let layer = rate_limit_layer(RateLimitConfig {
            rps: 1.0,
            burst: 1.0,
            ..Default::default()
        });
        let router_for_peer = |ip: IpAddr| {
            Router::new()
                .route("/", get(handler))
                .layer(layer.clone())
                .layer(MockConnectInfo(SocketAddr::new(ip, 4000)))
        };

        let client_a = router_for_peer(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        let client_b = router_for_peer(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)));

        let plain_req = || {
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("request")
        };

        // Client A exhausts its own (burst = 1) bucket.
        let resp = client_a.clone().oneshot(plain_req()).await.expect("a1");
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = client_a.oneshot(plain_req()).await.expect("a2");
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        // Client B, a distinct real peer sharing the same `RateLimitLayer`
        // (and thus the same underlying `RateLimiter`), is unaffected.
        let resp = client_b.oneshot(plain_req()).await.expect("b1");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a distinct real peer must get its own bucket, not share client A's"
        );
    }
}
