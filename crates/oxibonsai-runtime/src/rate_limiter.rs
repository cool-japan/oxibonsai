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
                .expect("global rate limiter mutex poisoned");
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
            .expect("client rate limiter mutex poisoned");

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
                .expect("global rate limiter mutex poisoned");
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
            .expect("client rate limiter mutex poisoned");

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
            .expect("client rate limiter mutex poisoned");
        clients.retain(|_, (_, last_seen)| now.duration_since(*last_seen) < ttl);
    }

    /// Number of currently tracked (active) clients.
    pub fn active_clients(&self) -> usize {
        self.clients
            .lock()
            .expect("client rate limiter mutex poisoned")
            .len()
    }

    /// Remove a specific client from the tracking map (resets their bucket).
    pub fn reset_client(&self, client_id: &str) {
        self.clients
            .lock()
            .expect("client rate limiter mutex poisoned")
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
                    .expect("global rate limiter mutex poisoned");
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
