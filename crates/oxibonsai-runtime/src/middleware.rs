//! Request middleware: context injection, logging, CORS, and idempotency caching.
//!
//! This module provides building blocks for production-grade HTTP middleware:
//!
//! - [`RequestContext`] — per-request metadata injected at the entry point
//! - [`RequestIdGen`] — atomic, monotonically increasing request ID generator
//! - [`RequestLogger`] — structured request/response logging with optional body capture
//! - [`CorsConfig`] — configurable CORS policy with header generation helpers
//! - [`IdempotencyCache`] — idempotency-key cache for safe request deduplication
//!
//! # Example
//!
//! ```
//! use oxibonsai_runtime::middleware::{RequestContext, RequestLogger, CorsConfig};
//!
//! let ctx = RequestContext::new("/v1/chat/completions", "POST", "10.0.0.1");
//! let logger = RequestLogger::new();
//! logger.log_request(&ctx);
//! logger.log_response(&ctx, 200, 512);
//!
//! let cors = CorsConfig::default();
//! assert!(cors.is_origin_allowed("*"));
//! ```

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};
use std::time::{Duration, Instant};

// ─── RequestContext ──────────────────────────────────────────────────────────

/// Per-request context injected by middleware at the entry point.
///
/// Carries metadata needed for logging, tracing, and metrics throughout
/// the request lifetime.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique request identifier (e.g. `"oxibonsai-1714000000000-1"`).
    pub request_id: String,
    /// Caller identity — typically an IP address or API key prefix.
    pub client_id: String,
    /// Wall-clock instant when the request was received.
    pub started_at: Instant,
    /// Request path (e.g. `"/v1/chat/completions"`).
    pub path: String,
    /// HTTP method in upper-case (e.g. `"POST"`).
    pub method: String,
}

impl RequestContext {
    /// Create a new context with an auto-generated request ID.
    pub fn new(path: &str, method: &str, client_id: &str) -> Self {
        // Generate a lightweight ID without an external generator so the type
        // is self-contained; callers can supply a [`RequestIdGen`] for prod use.
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let request_id = format!("req-{ts_ms}");
        Self {
            request_id,
            client_id: client_id.to_owned(),
            started_at: Instant::now(),
            path: path.to_owned(),
            method: method.to_uppercase(),
        }
    }

    /// Create a context with an explicit request ID (used with [`RequestIdGen`]).
    pub fn with_id(request_id: String, path: &str, method: &str, client_id: &str) -> Self {
        Self {
            request_id,
            client_id: client_id.to_owned(),
            started_at: Instant::now(),
            path: path.to_owned(),
            method: method.to_uppercase(),
        }
    }

    /// Elapsed time since the request was received, in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }

    /// Elapsed time since the request was received as a [`Duration`].
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

// ─── RequestIdGen ────────────────────────────────────────────────────────────

/// Atomic, monotonically increasing request ID generator.
///
/// IDs have the form `"{prefix}-{timestamp_ms}-{counter}"`, e.g.
/// `"oxibonsai-1714000000000-42"`. The combination of a millisecond
/// timestamp and a per-process counter makes collisions practically
/// impossible across restarts.
pub struct RequestIdGen {
    counter: AtomicU64,
    prefix: String,
}

impl RequestIdGen {
    /// Create a new generator with the given prefix string.
    pub fn new(prefix: &str) -> Self {
        Self {
            counter: AtomicU64::new(0),
            prefix: prefix.to_owned(),
        }
    }

    /// Generate the next unique request ID.
    ///
    /// Format: `"{prefix}-{timestamp_ms}-{counter}"`
    pub fn next(&self) -> String {
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("{}-{ts_ms}-{counter}", self.prefix)
    }
}

// ─── RequestLogger ───────────────────────────────────────────────────────────

/// Middleware that logs each request and its corresponding response.
///
/// Emits structured log lines via [`tracing`]. Body logging is opt-in and
/// truncated at `max_body_log_bytes` to avoid flooding logs with large payloads.
pub struct RequestLogger {
    /// Whether to include body content in log output.
    pub log_bodies: bool,
    /// Maximum number of body bytes to include in a log line.
    pub max_body_log_bytes: usize,
}

impl RequestLogger {
    /// Create a logger that does not log bodies.
    pub fn new() -> Self {
        Self {
            log_bodies: false,
            max_body_log_bytes: 0,
        }
    }

    /// Create a logger that includes up to `max_bytes` of body content.
    pub fn with_body_logging(max_bytes: usize) -> Self {
        Self {
            log_bodies: true,
            max_body_log_bytes: max_bytes,
        }
    }

    /// Log an incoming request.
    pub fn log_request(&self, ctx: &RequestContext) {
        let line = Self::format_request_line(ctx);
        tracing::info!(target: "oxibonsai::middleware", "{line}");
    }

    /// Log an outgoing response.
    pub fn log_response(&self, ctx: &RequestContext, status: u16, body_bytes: usize) {
        let elapsed_ms = ctx.elapsed_ms();
        let line = Self::format_response_line(ctx, status, elapsed_ms);
        if self.log_bodies && body_bytes > 0 {
            tracing::info!(
                target: "oxibonsai::middleware",
                "{line} body_bytes={body_bytes}"
            );
        } else {
            tracing::info!(target: "oxibonsai::middleware", "{line}");
        }
    }

    /// Format an incoming-request log line.
    ///
    /// Output: `"[{request_id}] {method} {path} from {client_id}"`
    pub fn format_request_line(ctx: &RequestContext) -> String {
        format!(
            "[{}] {} {} from {}",
            ctx.request_id, ctx.method, ctx.path, ctx.client_id
        )
    }

    /// Format an outgoing-response log line.
    ///
    /// Output: `"[{request_id}] {status} in {elapsed_ms}ms"`
    pub fn format_response_line(ctx: &RequestContext, status: u16, elapsed_ms: u64) -> String {
        format!("[{}] {} in {}ms", ctx.request_id, status, elapsed_ms)
    }
}

impl Default for RequestLogger {
    fn default() -> Self {
        Self::new()
    }
}

// ─── CorsConfig ──────────────────────────────────────────────────────────────

/// Cross-Origin Resource Sharing (CORS) policy configuration.
///
/// Used to generate `Access-Control-*` headers for preflight and main requests.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins. Use `["*"]` to permit all origins.
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods.
    pub allowed_methods: Vec<String>,
    /// Allowed request headers.
    pub allowed_headers: Vec<String>,
    /// `Access-Control-Max-Age` in seconds (how long browsers may cache the preflight).
    pub max_age_secs: u64,
    /// Whether to allow credentials (cookies, auth headers).
    pub allow_credentials: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
            allowed_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
            max_age_secs: 3600,
            allow_credentials: false,
        }
    }
}

impl CorsConfig {
    /// Returns `true` if the given `origin` is permitted by this policy.
    ///
    /// An entry of `"*"` in `allowed_origins` permits all origins.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Generate `Access-Control-*` response headers as `(name, value)` pairs.
    ///
    /// Returns headers suitable for both preflight (`OPTIONS`) and actual responses.
    pub fn access_control_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::with_capacity(5);

        let origin_value = if self.allowed_origins.iter().any(|o| o == "*") {
            "*".to_owned()
        } else {
            self.allowed_origins.join(", ")
        };
        headers.push(("Access-Control-Allow-Origin".to_owned(), origin_value));

        headers.push((
            "Access-Control-Allow-Methods".to_owned(),
            self.allowed_methods.join(", "),
        ));

        headers.push((
            "Access-Control-Allow-Headers".to_owned(),
            self.allowed_headers.join(", "),
        ));

        headers.push((
            "Access-Control-Max-Age".to_owned(),
            self.max_age_secs.to_string(),
        ));

        if self.allow_credentials {
            headers.push((
                "Access-Control-Allow-Credentials".to_owned(),
                "true".to_owned(),
            ));
        }

        headers
    }
}

// ─── IdempotencyCache ────────────────────────────────────────────────────────

/// Cached entry for a previously processed idempotent request.
struct CachedResponse {
    status: u16,
    body: Vec<u8>,
    created_at: Instant,
}

/// Request deduplication cache keyed on client-supplied idempotency keys.
///
/// When a client sends the same idempotency key twice, the second request
/// receives the cached response without re-executing the operation.
/// Entries expire after `ttl` and are lazily evicted.
pub struct IdempotencyCache {
    cache: Mutex<HashMap<String, CachedResponse>>,
    max_entries: usize,
    ttl: Duration,
}

impl IdempotencyCache {
    /// Create a new cache with the given capacity and TTL.
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_entries,
            ttl,
        }
    }

    /// Look up a previously cached response by idempotency key.
    ///
    /// Returns `(status_code, body)` if a fresh entry exists; `None` otherwise.
    pub fn get(&self, key: &str) -> Option<(u16, Vec<u8>)> {
        let cache = self.cache.lock().expect("idempotency cache mutex poisoned");
        if let Some(entry) = cache.get(key) {
            if entry.created_at.elapsed() < self.ttl {
                return Some((entry.status, entry.body.clone()));
            }
        }
        None
    }

    /// Store a response under the given idempotency key.
    ///
    /// If the cache is full, expired entries are evicted first. If still
    /// full after eviction, the insert is silently dropped to prevent
    /// unbounded memory growth.
    pub fn insert(&self, key: &str, status: u16, body: Vec<u8>) {
        let mut cache = self.cache.lock().expect("idempotency cache mutex poisoned");

        // Evict expired entries when approaching capacity.
        if cache.len() >= self.max_entries {
            let ttl = self.ttl;
            cache.retain(|_, v| v.created_at.elapsed() < ttl);
        }

        // After eviction, only insert if we still have room.
        if cache.len() < self.max_entries {
            cache.insert(
                key.to_owned(),
                CachedResponse {
                    status,
                    body,
                    created_at: Instant::now(),
                },
            );
        }
    }

    /// Remove all expired entries from the cache.
    pub fn evict_expired(&self) {
        let ttl = self.ttl;
        let mut cache = self.cache.lock().expect("idempotency cache mutex poisoned");
        cache.retain(|_, v| v.created_at.elapsed() < ttl);
    }

    /// Return the number of entries currently in the cache (including stale ones).
    pub fn len(&self) -> usize {
        self.cache
            .lock()
            .expect("idempotency cache mutex poisoned")
            .len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─── MiddlewareConfig ────────────────────────────────────────────────────────

/// Declarative configuration for the request middleware applied to the served
/// router by [`apply_middleware`].
///
/// This is the type the server assembles from its own configuration to decide
/// which of the building blocks in this module (CORS, request logging) and the
/// [`crate::rate_limiter::RateLimiter`] to attach. It is inert data — applying
/// it (which requires `axum`) lives behind the `server` feature.
#[derive(Debug, Clone)]
pub struct MiddlewareConfig {
    /// CORS policy to apply. `None` disables CORS header injection entirely.
    pub cors: Option<CorsConfig>,
    /// Whether to emit a structured request/response log line per request.
    pub enable_request_logging: bool,
    /// Optional per-client token-bucket rate limit. `None` disables rate
    /// limiting (the default — a rate cap is opt-in so it never silently
    /// throttles a deployment that did not ask for it).
    pub rate_limit: Option<crate::rate_limiter::RateLimitConfig>,
}

impl Default for MiddlewareConfig {
    fn default() -> Self {
        Self {
            cors: Some(CorsConfig::default()),
            enable_request_logging: true,
            rate_limit: None,
        }
    }
}

impl MiddlewareConfig {
    /// A configuration that applies no middleware at all.
    pub fn none() -> Self {
        Self {
            cors: None,
            enable_request_logging: false,
            rate_limit: None,
        }
    }

    /// Enable per-client rate limiting with the given config (builder style).
    pub fn with_rate_limit(mut self, config: crate::rate_limiter::RateLimitConfig) -> Self {
        self.rate_limit = Some(config);
        self
    }
}

// ─── Axum layer wiring (server feature) ───────────────────────────────────────

#[cfg(feature = "server")]
mod layer {
    use super::{CorsConfig, MiddlewareConfig, RequestContext, RequestLogger};
    use crate::rate_limiter::{extract_client_id, RateLimitDecision, RateLimiter};
    use axum::body::Body;
    use axum::extract::{ConnectInfo, FromRequestParts, State};
    use axum::http::{request::Parts, HeaderName, HeaderValue, Method, Request, StatusCode};
    use axum::middleware::Next;
    use axum::response::{IntoResponse, Response};
    use axum::{Json, Router};
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use std::sync::Arc;

    /// Best-effort [`ConnectInfo`] extractor that degrades to `None` instead
    /// of rejecting the request when the router was not served via
    /// [`axum::routing::Router::into_make_service_with_connect_info`] (the
    /// case for the plain `axum::serve(listener, router)` this crate's
    /// server currently uses).
    ///
    /// `axum`'s own `Option<ConnectInfo<T>>` cannot be used here: as of
    /// axum-core 0.5 an extractor must opt in to `OptionalFromRequestParts`
    /// to be wrapped in `Option<..>`, and [`ConnectInfo`] does not. This
    /// thin wrapper implements [`FromRequestParts`] directly instead, so
    /// [`rate_limit_mw`] can consult the real peer address *when available*
    /// without hard-requiring connect-info wiring everywhere
    /// [`apply_middleware`] with rate limiting enabled is used.
    struct MaybePeerAddr(Option<SocketAddr>);

    impl<S> FromRequestParts<S> for MaybePeerAddr
    where
        S: Send + Sync,
    {
        type Rejection = Infallible;

        async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
            match ConnectInfo::<SocketAddr>::from_request_parts(parts, state).await {
                Ok(ConnectInfo(addr)) => Ok(Self(Some(addr))),
                Err(_) => Ok(Self(None)),
            }
        }
    }

    /// Attach the configured middleware to `router`.
    ///
    /// Layers are applied outermost-first at request time: CORS wraps the rate
    /// limiter (so even a `429` carries CORS headers), which wraps request
    /// logging, which wraps the routes.
    pub fn apply_middleware(mut router: Router, config: MiddlewareConfig) -> Router {
        if config.enable_request_logging {
            let logger = Arc::new(RequestLogger::new());
            router = router.layer(axum::middleware::from_fn_with_state(logger, logging_mw));
        }
        if let Some(rate_config) = config.rate_limit {
            let limiter = Arc::new(RateLimiter::new(rate_config));
            router = router.layer(axum::middleware::from_fn_with_state(limiter, rate_limit_mw));
        }
        if let Some(cors) = config.cors {
            let cors = Arc::new(cors);
            router = router.layer(axum::middleware::from_fn_with_state(cors, cors_mw));
        }
        router
    }

    /// Inject the configured `Access-Control-*` headers on every response and
    /// short-circuit `OPTIONS` preflight requests with `204 No Content`.
    async fn cors_mw(
        State(cors): State<Arc<CorsConfig>>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        let is_preflight = req.method() == Method::OPTIONS;
        let mut response = if is_preflight {
            StatusCode::NO_CONTENT.into_response()
        } else {
            next.run(req).await
        };
        let headers = response.headers_mut();
        for (name, value) in cors.access_control_headers() {
            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(&value),
            ) {
                headers.insert(header_name, header_value);
            }
        }
        response
    }

    /// Emit a structured request/response log line via the shared logger.
    async fn logging_mw(
        State(logger): State<Arc<RequestLogger>>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        let ctx = RequestContext::new(req.uri().path(), req.method().as_str(), "");
        logger.log_request(&ctx);
        let response = next.run(req).await;
        logger.log_response(&ctx, response.status().as_u16(), 0);
        response
    }

    /// Enforce the per-client rate limit, returning `429 Too Many Requests`
    /// (with a `Retry-After` header) when a client exceeds its budget. The
    /// liveness (`/health`) and metrics (`/metrics`) probes are always exempt.
    ///
    /// Client identity is derived via [`extract_client_id`]: `X-Forwarded-For`
    /// / `X-Real-IP` are honored only when the real TCP peer (via
    /// [`MaybePeerAddr`]) is present in
    /// [`RateLimitConfig::trusted_proxies`]; otherwise the real peer address
    /// is used directly. [`MaybePeerAddr`] degrades gracefully (falls back to
    /// the peer-less `"unknown"` bucket, matching the historical behavior)
    /// rather than rejecting every request when the router is served without
    /// `into_make_service_with_connect_info`.
    ///
    /// [`RateLimitConfig::trusted_proxies`]: crate::rate_limiter::RateLimitConfig::trusted_proxies
    async fn rate_limit_mw(
        State(limiter): State<Arc<RateLimiter>>,
        MaybePeerAddr(peer): MaybePeerAddr,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        let path = req.uri().path();
        if path == "/health" || path == "/metrics" {
            return next.run(req).await;
        }
        let peer_ip = peer.map(|addr| addr.ip());
        let client_id = extract_client_id(req.headers(), peer_ip, limiter.trusted_proxies());
        match limiter.check_and_consume(&client_id) {
            RateLimitDecision::Allow => next.run(req).await,
            RateLimitDecision::Deny { retry_after_ms } => {
                let retry_secs = (retry_after_ms.saturating_add(999) / 1000).max(1);
                let body = Json(serde_json::json!({
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
        }
    }
}

#[cfg(feature = "server")]
pub use layer::apply_middleware;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_request_context_elapsed() {
        let ctx = RequestContext::new("/health", "GET", "10.0.0.1");
        // Elapsed should be very small immediately after creation.
        assert!(
            ctx.elapsed_ms() < 500,
            "elapsed should be <500ms at creation"
        );
        assert!(ctx.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn test_request_id_gen_unique() {
        let gen = RequestIdGen::new("test");
        let ids: Vec<String> = (0..100).map(|_| gen.next()).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "all generated IDs must be unique");
    }

    #[test]
    fn test_request_id_gen_prefix() {
        let gen = RequestIdGen::new("oxibonsai");
        let id = gen.next();
        assert!(
            id.starts_with("oxibonsai-"),
            "ID should start with prefix; got {id}"
        );
    }

    #[test]
    fn test_request_logger_format_request_line() {
        let mut ctx = RequestContext::new("/v1/chat/completions", "post", "1.2.3.4");
        ctx.request_id = "req-42".to_owned();
        let line = RequestLogger::format_request_line(&ctx);
        assert_eq!(line, "[req-42] POST /v1/chat/completions from 1.2.3.4");
    }

    #[test]
    fn test_request_logger_format_response_line() {
        let mut ctx = RequestContext::new("/health", "GET", "127.0.0.1");
        ctx.request_id = "req-99".to_owned();
        let line = RequestLogger::format_response_line(&ctx, 200, 15);
        assert_eq!(line, "[req-99] 200 in 15ms");
    }

    #[test]
    fn test_cors_config_default_allows_all() {
        let cors = CorsConfig::default();
        assert!(cors.is_origin_allowed("https://example.com"));
        assert!(cors.is_origin_allowed("null"));
        assert!(cors.is_origin_allowed("*"));
    }

    #[test]
    fn test_cors_config_specific_origin() {
        let cors = CorsConfig {
            allowed_origins: vec!["https://app.example.com".to_string()],
            ..Default::default()
        };
        assert!(cors.is_origin_allowed("https://app.example.com"));
        assert!(!cors.is_origin_allowed("https://evil.example.com"));
    }

    #[test]
    fn test_cors_access_control_headers() {
        let cors = CorsConfig::default();
        let headers = cors.access_control_headers();

        // Should contain Access-Control-Allow-Origin
        let has_origin = headers
            .iter()
            .any(|(k, v)| k == "Access-Control-Allow-Origin" && v == "*");
        assert!(has_origin, "should have wildcard Allow-Origin header");

        // Should contain methods
        let has_methods = headers
            .iter()
            .any(|(k, _)| k == "Access-Control-Allow-Methods");
        assert!(has_methods);

        // allow_credentials is false by default, so no credentials header
        let has_creds = headers
            .iter()
            .any(|(k, _)| k == "Access-Control-Allow-Credentials");
        assert!(
            !has_creds,
            "should not include credentials header by default"
        );
    }

    #[test]
    fn test_idempotency_cache_insert_and_get() {
        let cache = IdempotencyCache::new(100, Duration::from_secs(60));
        cache.insert("key-1", 200, b"hello".to_vec());
        let result = cache.get("key-1");
        assert_eq!(result, Some((200, b"hello".to_vec())));
    }

    #[test]
    fn test_idempotency_cache_miss() {
        let cache = IdempotencyCache::new(100, Duration::from_secs(60));
        assert!(cache.get("nonexistent-key").is_none());
    }

    #[test]
    fn test_idempotency_cache_evicts_expired() {
        // TTL of 10ms so entries expire quickly in tests.
        let cache = IdempotencyCache::new(100, Duration::from_millis(10));
        cache.insert("exp-key", 200, vec![]);
        assert_eq!(cache.len(), 1);

        thread::sleep(Duration::from_millis(20));
        cache.evict_expired();
        assert_eq!(cache.len(), 0, "expired entry should have been evicted");
    }

    #[test]
    fn test_idempotency_cache_expired_returns_none() {
        let cache = IdempotencyCache::new(100, Duration::from_millis(10));
        cache.insert("ttl-key", 201, b"data".to_vec());
        thread::sleep(Duration::from_millis(20));
        // get() should not return stale entries.
        assert!(
            cache.get("ttl-key").is_none(),
            "stale cache entry must not be returned"
        );
    }
}

// ─── Rate-limit trusted-proxy integration tests (findings serve-api-07 /
//     security-05) ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "server"))]
mod rate_limit_trusted_proxy_tests {
    use super::apply_middleware;
    use crate::rate_limiter::RateLimitConfig;
    use axum::body::Body;
    use axum::extract::connect_info::MockConnectInfo;
    use axum::http::{Request, StatusCode};
    use axum::routing::get;
    use axum::Router;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use tower::ServiceExt;

    async fn handler() -> &'static str {
        "ok"
    }

    /// Regression test for findings `serve-api-07` / `security-05`: with the
    /// default (empty) `trusted_proxies` allowlist and no real connect-info
    /// available (mirroring how `oxibonsai-serve` currently serves its
    /// router via plain `axum::serve`, without connect-info wiring), a
    /// spoofed `X-Forwarded-For` must NOT grant a fresh rate-limit bucket:
    /// two requests carrying different claimed origins must land in the
    /// same ("unknown") bucket, so the second one is rate-limited exactly as
    /// it would be if the header were absent entirely.
    #[tokio::test]
    async fn spoofed_forwarded_header_is_ignored_by_default() {
        let config = super::MiddlewareConfig::none().with_rate_limit(RateLimitConfig {
            rps: 1.0,
            burst: 1.0,
            ..Default::default()
        });
        let router = apply_middleware(Router::new().route("/", get(handler)), config);

        let req1 = Request::builder()
            .uri("/")
            .header("x-forwarded-for", "1.1.1.1")
            .body(Body::empty())
            .expect("request 1");
        let resp1 = router.clone().oneshot(req1).await.expect("response 1");
        assert_eq!(
            resp1.status(),
            StatusCode::OK,
            "first request should be allowed"
        );

        let req2 = Request::builder()
            .uri("/")
            .header("x-forwarded-for", "2.2.2.2")
            .body(Body::empty())
            .expect("request 2");
        let resp2 = router.clone().oneshot(req2).await.expect("response 2");
        assert_eq!(
            resp2.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "a spoofed X-Forwarded-For claiming a different origin must not \
             bypass the rate limit by default -- got {:?}",
            resp2.status()
        );
    }

    /// When the real TCP peer *is* present (via `ConnectInfo`, mocked here
    /// with `MockConnectInfo` since the test harness has no real socket) and
    /// explicitly listed in `trusted_proxies`, the forwarded header is
    /// honored -- distinct claimed clients behind that proxy get
    /// independent buckets (the legitimate reverse-proxy deployment case).
    #[tokio::test]
    async fn forwarded_header_from_trusted_proxy_grants_independent_buckets() {
        let proxy_addr: IpAddr = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let config = super::MiddlewareConfig::none().with_rate_limit(RateLimitConfig {
            rps: 1.0,
            burst: 1.0,
            trusted_proxies: vec![proxy_addr],
            ..Default::default()
        });
        let router = apply_middleware(Router::new().route("/", get(handler)), config)
            .layer(MockConnectInfo(SocketAddr::new(proxy_addr, 4000)));

        let req1 = Request::builder()
            .uri("/")
            .header("x-forwarded-for", "1.1.1.1")
            .body(Body::empty())
            .expect("request 1");
        let resp1 = router.clone().oneshot(req1).await.expect("response 1");
        assert_eq!(resp1.status(), StatusCode::OK);

        let req2 = Request::builder()
            .uri("/")
            .header("x-forwarded-for", "2.2.2.2")
            .body(Body::empty())
            .expect("request 2");
        let resp2 = router.clone().oneshot(req2).await.expect("response 2");
        assert_eq!(
            resp2.status(),
            StatusCode::OK,
            "a distinct forwarded client behind a trusted proxy should get \
             its own bucket, not share the first client's"
        );
    }

    /// An untrusted peer (not in `trusted_proxies`) must not have its
    /// forwarded header honored even when `ConnectInfo` is available --
    /// the allowlist match must be exact.
    #[tokio::test]
    async fn forwarded_header_from_untrusted_peer_is_still_ignored() {
        let untrusted_peer: IpAddr = IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7));
        let trusted: IpAddr = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let config = super::MiddlewareConfig::none().with_rate_limit(RateLimitConfig {
            rps: 1.0,
            burst: 1.0,
            trusted_proxies: vec![trusted],
            ..Default::default()
        });
        let router = apply_middleware(Router::new().route("/", get(handler)), config)
            .layer(MockConnectInfo(SocketAddr::new(untrusted_peer, 4000)));

        let req1 = Request::builder()
            .uri("/")
            .header("x-forwarded-for", "1.1.1.1")
            .body(Body::empty())
            .expect("request 1");
        let resp1 = router.clone().oneshot(req1).await.expect("response 1");
        assert_eq!(resp1.status(), StatusCode::OK);

        let req2 = Request::builder()
            .uri("/")
            .header("x-forwarded-for", "2.2.2.2")
            .body(Body::empty())
            .expect("request 2");
        let resp2 = router.clone().oneshot(req2).await.expect("response 2");
        assert_eq!(
            resp2.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "an untrusted peer's forwarded header must not grant a fresh bucket"
        );
    }
}
