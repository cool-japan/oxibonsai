//! Request middleware: context injection, logging, CORS, and idempotency caching.
//!
//! This module provides building blocks for production-grade HTTP middleware:
//!
//! - [`RequestContext`] — per-request metadata injected at the entry point
//! - [`RequestIdGen`] — atomic, monotonically increasing request ID generator
//! - [`RequestLogger`] — structured request/response logging with optional body capture
//! - [`CorsConfig`] — configurable CORS policy with header generation helpers
//! - [`IdempotencyCache`] — idempotency-key cache for safe request deduplication
//! - [`MiddlewareConfig`] + [`apply_middleware`] — assemble and attach the
//!   above (plus, optionally, [`crate::rate_limiter::rate_limit_layer`]) to a
//!   served router in one call
//!
//! # CORS default (findings sec-06 / sec-09 / SV-20)
//!
//! [`MiddlewareConfig::default()`] emits **no** `Access-Control-*` header on
//! any route. CORS is strictly opt-in: build a policy with
//! [`CorsConfig::from_origins`] and attach it via
//! [`MiddlewareConfig::with_cors`]. When a policy is attached, the exact
//! response header set is documented on [`CorsConfig::response_headers`];
//! `/admin/*` never receives CORS headers regardless of configuration.
//!
//! # Layer order (findings sec-06/sec-07/SV-06/SV-10/cli-18)
//!
//! A server that also enforces bearer auth and an admission/concurrency
//! limit (both live outside this crate — see `oxibonsai-serve` and the CLI's
//! `admission` module) must compose its layers in this order, outermost
//! first:
//!
//! ```text
//! cors -> auth -> rate-limit -> admission -> routes
//! ```
//!
//! - **cors outside auth**: a CORS preflight (`OPTIONS`) carries no
//!   credentials by specification, so if auth were outside CORS every
//!   preflight would be rejected and a browser could never use an
//!   authenticated server (finding SV-06). [`apply_middleware`]'s `cors_mw`
//!   short-circuits `OPTIONS` with `200 OK` before any nested layer runs.
//! - **auth outside rate-limit**: an unauthenticated request must never
//!   consume a rate-limit token.
//! - **rate-limit outside admission**: a rate-limited request must never
//!   consume an admission/concurrency permit (cli-18).
//!
//! [`apply_middleware`] bundles CORS + rate limiting + logging for the
//! common case with no auth layer to interpose; a caller that needs auth
//! *between* CORS and rate limiting mounts
//! [`crate::rate_limiter::rate_limit_layer`] directly instead (see that
//! function's docs for the exact `.layer()` call sequence).
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
//! // `CorsConfig::default()` is intentionally permissive (see its docs);
//! // production configuration should use `CorsConfig::from_origins` and
//! // `MiddlewareConfig::with_cors` instead.
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
/// Used to generate `Access-Control-*` headers for preflight and main
/// requests via [`CorsConfig::response_headers`]. A `CorsConfig` only takes
/// effect when installed via [`MiddlewareConfig::cors`]
/// (`Some(CorsConfig{..})`) — the crate-wide default,
/// [`MiddlewareConfig::default()`], sets `cors: None`, i.e. **no**
/// `Access-Control-*` header is emitted on any route unless a caller
/// explicitly opts in (findings sec-06 / SV-20).
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins. Use `["*"]` to permit all origins.
    ///
    /// # Warning
    /// [`CorsConfig::default()`] sets this to `["*"]` so that existing code
    /// which explicitly asks for the [`Default`] impl keeps a permissive
    /// policy. That is almost never the right starting point for a policy
    /// built from *operator-supplied* configuration (a `--cors-origin` CLI
    /// flag, a `[cors]` TOML section, ...) — use
    /// [`CorsConfig::from_origins`] there instead, so an empty/unset
    /// configuration cannot silently widen into a wildcard.
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods, sent as a comma-joined
    /// `Access-Control-Allow-Methods` value on a matched request.
    pub allowed_methods: Vec<String>,
    /// Allowed request headers, sent as a comma-joined
    /// `Access-Control-Allow-Headers` value on a matched request.
    pub allowed_headers: Vec<String>,
    /// `Access-Control-Max-Age` in seconds (how long browsers may cache the preflight).
    pub max_age_secs: u64,
    /// Whether to allow credentials (cookies, auth headers). When `true`,
    /// [`CorsConfig::response_headers`] never emits the literal wildcard
    /// `Access-Control-Allow-Origin: *` (the Fetch spec forbids combining
    /// it with `Access-Control-Allow-Credentials: true`, and every browser
    /// rejects the response) — it echoes the exact matched origin instead,
    /// even when `"*"` is present in `allowed_origins`.
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
    /// Build a policy that allows exactly `origins` (never a wildcard),
    /// keeping the [`Default`] methods/headers/max-age and credentials
    /// disabled.
    ///
    /// Prefer this over `CorsConfig::default()` when building a policy from
    /// operator-supplied configuration — see the `allowed_origins` field
    /// docs for why.
    pub fn from_origins(origins: impl IntoIterator<Item = String>) -> Self {
        Self {
            allowed_origins: origins.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Returns `true` if the given `origin` is permitted by this policy.
    ///
    /// An entry of `"*"` in `allowed_origins` permits all origins.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Returns `true` when this policy's `allowed_origins` contains the
    /// literal wildcard `"*"` *and* credentials are disabled — the only
    /// case in which `Access-Control-Allow-Origin: *` may legally be sent
    /// (see the `allow_credentials` field docs).
    fn is_unrestricted_wildcard(&self) -> bool {
        !self.allow_credentials && self.allowed_origins.iter().any(|o| o == "*")
    }

    /// Compute the exact `Access-Control-*` / `Vary` response headers for a
    /// request whose `Origin` header was `request_origin` (`None` when the
    /// request carried no `Origin` header at all — a same-origin or
    /// non-browser request).
    ///
    /// Contract (findings sec-06 / sec-09 / SV-20):
    /// - `Vary: Origin` is **always** present, regardless of whether
    ///   `request_origin` is present or matches, because whether the other
    ///   headers below appear depends on it — a cache that ignored this
    ///   could replay one origin's grant (or denial) to a different origin.
    /// - `Access-Control-Allow-Origin` is present **only** when
    ///   `request_origin` is `Some` and [`CorsConfig::is_origin_allowed`]
    ///   returns `true` for it. Its value is the *exact* origin, echoed back
    ///   verbatim, **except** when the policy is the unrestricted wildcard
    ///   (`allowed_origins` contains `"*"` and `allow_credentials` is
    ///   `false`), in which case it is the literal `"*"`. Multiple
    ///   configured origins are **never** comma-joined into this header —
    ///   that value is invalid per the Fetch spec and every browser rejects
    ///   it.
    /// - `Access-Control-Allow-Methods` / `-Headers` / `-Max-Age` /
    ///   `-Credentials` accompany the allow decision only on a match.
    pub fn response_headers(&self, request_origin: Option<&str>) -> Vec<(String, String)> {
        let mut headers = Vec::with_capacity(6);
        headers.push(("Vary".to_owned(), "Origin".to_owned()));

        let Some(origin) = request_origin else {
            return headers;
        };
        if !self.is_origin_allowed(origin) {
            return headers;
        }

        let allow_origin_value = if self.is_unrestricted_wildcard() {
            "*".to_owned()
        } else {
            origin.to_owned()
        };
        headers.push(("Access-Control-Allow-Origin".to_owned(), allow_origin_value));
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
    /// CORS policy to apply. `None` (the default — see below) disables CORS
    /// header injection entirely: no route, including `/admin/*`, emits any
    /// `Access-Control-*` header.
    pub cors: Option<CorsConfig>,
    /// Whether to emit a structured request/response log line per request.
    pub enable_request_logging: bool,
    /// Optional per-client token-bucket rate limit. `None` disables rate
    /// limiting (the default — a rate cap is opt-in so it never silently
    /// throttles a deployment that did not ask for it).
    pub rate_limit: Option<crate::rate_limiter::RateLimitConfig>,
}

impl Default for MiddlewareConfig {
    /// The production default: **no** CORS header on any route (`cors:
    /// None`), request logging on, and no rate limit. Earlier versions
    /// defaulted `cors` to `Some(CorsConfig::default())` — an unconditional
    /// `Access-Control-Allow-Origin: *` on every route including
    /// `/admin/*` — which let any web page a browser visited read
    /// inference results from a server reachable from that browser
    /// (findings sec-06 / SV-20). CORS is now strictly opt-in: build a
    /// policy with [`CorsConfig::from_origins`] and attach it with
    /// [`MiddlewareConfig::with_cors`].
    fn default() -> Self {
        Self {
            cors: None,
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

    /// Enable CORS with the given policy (builder style). See
    /// [`CorsConfig::from_origins`] for building `config` from an
    /// operator-supplied allow-list.
    pub fn with_cors(mut self, config: CorsConfig) -> Self {
        self.cors = Some(config);
        self
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
    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{header, HeaderName, HeaderValue, Method, Request, StatusCode};
    use axum::middleware::Next;
    use axum::response::{IntoResponse, Response};
    use axum::Router;
    use std::sync::Arc;

    /// Attach the configured middleware to `router`.
    ///
    /// Layers are applied outermost-first at request time: CORS wraps the rate
    /// limiter (so even a `429` carries CORS headers), which wraps request
    /// logging, which wraps the routes. This mirrors (and, for rate limiting,
    /// shares an implementation with — see
    /// [`crate::rate_limiter::rate_limit_layer`]) the standalone
    /// `cors -> auth -> rate-limit -> admission` order documented on
    /// [`crate::rate_limiter`] for callers that need to interleave their own
    /// auth layer between CORS and rate limiting instead of using this
    /// all-in-one convenience function.
    pub fn apply_middleware(mut router: Router, config: MiddlewareConfig) -> Router {
        if config.enable_request_logging {
            let logger = Arc::new(RequestLogger::new());
            router = router.layer(axum::middleware::from_fn_with_state(logger, logging_mw));
        }
        if let Some(rate_config) = config.rate_limit {
            router = router.layer(crate::rate_limiter::rate_limit_layer(rate_config));
        }
        if let Some(cors) = config.cors {
            let cors = Arc::new(cors);
            router = router.layer(axum::middleware::from_fn_with_state(cors, cors_mw));
        }
        router
    }

    /// Inject the configured `Access-Control-*` / `Vary` headers per
    /// [`CorsConfig::response_headers`] and short-circuit `OPTIONS` preflight
    /// requests with `200 OK` *before* they reach any downstream layer —
    /// including a caller-supplied auth layer nested inside this one — since
    /// a browser sends preflight without credentials and would otherwise
    /// always be rejected by auth (finding SV-06). `/admin/*` never receives
    /// CORS headers regardless of configuration (finding sec-06): mount this
    /// as the outermost layer (see [`crate::rate_limiter`]'s "Ready-to-mount
    /// layer" docs) so it also wraps the admin router.
    async fn cors_mw(
        State(cors): State<Arc<CorsConfig>>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        let path = req.uri().path().to_owned();
        let is_admin_path = path == "/admin" || path.starts_with("/admin/");
        let is_preflight = req.method() == Method::OPTIONS;
        let request_origin = req
            .headers()
            .get(header::ORIGIN)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);

        let mut response = if is_preflight {
            StatusCode::OK.into_response()
        } else {
            next.run(req).await
        };

        if is_admin_path {
            return response;
        }

        for (name, value) in cors.response_headers(request_origin.as_deref()) {
            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(&value),
            ) {
                if header_name == header::VARY {
                    // `Vary` is a genuinely multi-valued header (e.g. a
                    // handler or a compression layer may already have set
                    // `Vary: Accept-Encoding`); `insert` would silently
                    // replace it with just `Origin` and destroy that
                    // cache-variance declaration. `append` adds this as an
                    // additional value instead.
                    response.headers_mut().append(header_name, header_value);
                } else {
                    response.headers_mut().insert(header_name, header_value);
                }
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
        let headers = cors.response_headers(Some("https://example.com"));

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

    /// Helper: look up a header's value in a `response_headers()` result.
    fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Table-driven coverage of the exact `CorsConfig::response_headers`
    /// contract (findings sec-06 / sec-09 / SV-20): echo-on-match, always
    /// `Vary: Origin`, never a comma-joined `Access-Control-Allow-Origin`,
    /// and no grant at all on a non-match or a missing `Origin` header.
    #[test]
    fn test_cors_response_headers_table() {
        struct Case {
            name: &'static str,
            config: CorsConfig,
            request_origin: Option<&'static str>,
            expect_acao: Option<&'static str>,
        }

        let two_origins = CorsConfig::from_origins([
            "https://a.example".to_string(),
            "https://b.example".to_string(),
        ]);

        let cases = vec![
            Case {
                name: "no Origin header at all -> no ACAO",
                config: CorsConfig::from_origins(["https://a.example".to_string()]),
                request_origin: None,
                expect_acao: None,
            },
            Case {
                name: "single configured origin, matching request -> echoed",
                config: CorsConfig::from_origins(["https://a.example".to_string()]),
                request_origin: Some("https://a.example"),
                expect_acao: Some("https://a.example"),
            },
            Case {
                name: "single configured origin, non-matching request -> no ACAO",
                config: CorsConfig::from_origins(["https://a.example".to_string()]),
                request_origin: Some("https://evil.example"),
                expect_acao: None,
            },
            Case {
                name: "two configured origins, first matches -> echoed (not joined)",
                config: two_origins.clone(),
                request_origin: Some("https://a.example"),
                expect_acao: Some("https://a.example"),
            },
            Case {
                name: "two configured origins, second matches -> echoed (not joined)",
                config: two_origins.clone(),
                request_origin: Some("https://b.example"),
                expect_acao: Some("https://b.example"),
            },
            Case {
                name: "two configured origins, non-match -> no ACAO",
                config: two_origins,
                request_origin: Some("https://evil.example"),
                expect_acao: None,
            },
            Case {
                name: "literal wildcard policy, credentials off -> literal *",
                config: CorsConfig::default(),
                request_origin: Some("https://anything.example"),
                expect_acao: Some("*"),
            },
            Case {
                name: "wildcard + credentials on -> exact origin, never literal *",
                config: CorsConfig {
                    allow_credentials: true,
                    ..CorsConfig::default()
                },
                request_origin: Some("https://anything.example"),
                expect_acao: Some("https://anything.example"),
            },
        ];

        for case in cases {
            let headers = case.config.response_headers(case.request_origin);

            // The comma-joined value the pre-fix code emitted for a
            // multi-origin config is never valid per the Fetch spec.
            if let Some(acao) = header_value(&headers, "Access-Control-Allow-Origin") {
                assert!(
                    !acao.contains(','),
                    "case '{}': Access-Control-Allow-Origin must never be comma-joined; got {acao:?}",
                    case.name
                );
            }

            assert_eq!(
                header_value(&headers, "Access-Control-Allow-Origin"),
                case.expect_acao,
                "case '{}': unexpected Access-Control-Allow-Origin",
                case.name
            );

            // "always emit Vary: Origin" (sec-06 fix text) whenever a
            // policy is installed at all, matched or not.
            assert_eq!(
                header_value(&headers, "Vary"),
                Some("Origin"),
                "case '{}': Vary: Origin must always be present when a CORS policy runs",
                case.name
            );

            // Allow-Methods/-Headers/-Max-Age travel with a grant only.
            let has_methods = header_value(&headers, "Access-Control-Allow-Methods").is_some();
            assert_eq!(
                has_methods,
                case.expect_acao.is_some(),
                "case '{}': Allow-Methods presence must match the ACAO grant",
                case.name
            );
        }
    }

    #[test]
    fn test_cors_credentials_and_wildcard_never_combined() {
        // allow_credentials=true must never coincide with a literal "*"
        // Access-Control-Allow-Origin, even if "*" is (mis)configured into
        // allowed_origins -- browsers reject that exact combination.
        let cors = CorsConfig {
            allowed_origins: vec!["*".to_string()],
            allow_credentials: true,
            ..Default::default()
        };
        let headers = cors.response_headers(Some("https://client.example"));
        assert_eq!(
            header_value(&headers, "Access-Control-Allow-Origin"),
            Some("https://client.example")
        );
        assert_eq!(
            header_value(&headers, "Access-Control-Allow-Credentials"),
            Some("true")
        );
    }

    #[test]
    fn test_cors_from_origins_builder() {
        let cors = CorsConfig::from_origins(["https://a.example".to_string()]);
        assert!(cors.is_origin_allowed("https://a.example"));
        assert!(!cors.is_origin_allowed("*"));
        assert!(!cors.allow_credentials);
        assert_eq!(cors.max_age_secs, CorsConfig::default().max_age_secs);
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

// ─── CORS header-contract integration tests (findings sec-06 / sec-09 /
//     SV-06 / SV-20) ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "server"))]
mod cors_header_contract_tests {
    use super::{apply_middleware, CorsConfig, MiddlewareConfig};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::middleware::{self, Next};
    use axum::response::{IntoResponse, Response};
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    async fn handler() -> &'static str {
        "ok"
    }

    fn get_request(path: &str, origin: Option<&str>) -> Request<Body> {
        let mut builder = Request::get(path);
        if let Some(origin) = origin {
            builder = builder.header("origin", origin);
        }
        builder.body(Body::empty()).expect("request")
    }

    fn options_request(path: &str) -> Request<Body> {
        Request::builder()
            .method("OPTIONS")
            .uri(path)
            .body(Body::empty())
            .expect("preflight request")
    }

    fn header<'a>(resp: &'a Response, name: &str) -> Option<&'a str> {
        resp.headers().get(name).and_then(|v| v.to_str().ok())
    }

    fn base_router() -> Router {
        Router::new().route("/", get(handler)).route(
            "/admin/status",
            get(|| async { "admin: super-secret config" }),
        )
    }

    /// The no-config case: [`MiddlewareConfig::none()`] (and, equivalently,
    /// [`MiddlewareConfig::default()`] as of the sec-06 fix) installs no
    /// CORS layer at all, so **no** `Access-Control-*` or `Vary` header
    /// appears on any response, matched origin or not.
    #[tokio::test]
    async fn no_cors_config_emits_no_header_at_all() {
        let router = apply_middleware(base_router(), MiddlewareConfig::none());
        let resp = router
            .oneshot(get_request("/", Some("https://example.com")))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(header(&resp, "access-control-allow-origin").is_none());
        assert!(
            header(&resp, "vary").is_none(),
            "no CORS layer at all means no Vary either"
        );
    }

    /// `MiddlewareConfig::default()` is the production default and must
    /// behave identically to `none()` for CORS: no header on any route.
    #[tokio::test]
    async fn default_middleware_config_emits_no_cors_header() {
        let router = apply_middleware(base_router(), MiddlewareConfig::default());
        let resp = router
            .oneshot(get_request("/", Some("https://example.com")))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            header(&resp, "access-control-allow-origin").is_none(),
            "MiddlewareConfig::default() must not restore the old wildcard-by-default CORS"
        );
    }

    /// The single-match case: a request whose `Origin` matches the
    /// configured allow-list gets the origin echoed back plus `Vary:
    /// Origin`.
    #[tokio::test]
    async fn matching_origin_is_echoed_with_vary() {
        let cors = CorsConfig::from_origins(["https://app.example.com".to_string()]);
        let router = apply_middleware(base_router(), MiddlewareConfig::none().with_cors(cors));

        let resp = router
            .oneshot(get_request("/", Some("https://app.example.com")))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            header(&resp, "access-control-allow-origin"),
            Some("https://app.example.com")
        );
        assert_eq!(header(&resp, "vary"), Some("Origin"));
    }

    /// The non-match case: a request whose `Origin` does **not** match the
    /// configured allow-list gets no grant header, but `Vary: Origin` is
    /// still present because the decision to withhold the grant itself
    /// depended on the `Origin` header (sec-06 fix text: "always emit
    /// Vary: Origin").
    #[tokio::test]
    async fn non_matching_origin_gets_no_acao_but_still_vary() {
        let cors = CorsConfig::from_origins(["https://app.example.com".to_string()]);
        let router = apply_middleware(base_router(), MiddlewareConfig::none().with_cors(cors));

        let resp = router
            .oneshot(get_request("/", Some("https://evil.example.com")))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            header(&resp, "access-control-allow-origin").is_none(),
            "a non-matching origin must never be granted access"
        );
        assert_eq!(header(&resp, "vary"), Some("Origin"));
        assert!(
            header(&resp, "access-control-allow-methods").is_none(),
            "no grant headers should accompany a non-match"
        );
    }

    /// A config listing multiple origins must never comma-join them into a
    /// single (invalid) `Access-Control-Allow-Origin` value -- each request
    /// gets exactly its own matched origin echoed back.
    #[tokio::test]
    async fn multi_origin_config_never_comma_joins() {
        let cors = CorsConfig::from_origins([
            "https://a.example.com".to_string(),
            "https://b.example.com".to_string(),
        ]);
        let router = apply_middleware(base_router(), MiddlewareConfig::none().with_cors(cors));

        let resp_a = router
            .clone()
            .oneshot(get_request("/", Some("https://a.example.com")))
            .await
            .expect("response a");
        assert_eq!(
            header(&resp_a, "access-control-allow-origin"),
            Some("https://a.example.com")
        );

        let resp_b = router
            .oneshot(get_request("/", Some("https://b.example.com")))
            .await
            .expect("response b");
        assert_eq!(
            header(&resp_b, "access-control-allow-origin"),
            Some("https://b.example.com")
        );
    }

    /// `cors_mw` must not clobber a `Vary` header a handler (or another
    /// layer, e.g. compression) already set: `Vary: Origin` must be added
    /// *alongside* an existing `Vary: Accept-Encoding`, not replace it --
    /// `HeaderMap::insert` on a genuinely multi-valued header would destroy
    /// the handler's own cache-variance declaration.
    #[tokio::test]
    async fn vary_from_a_handler_is_preserved_alongside_origin() {
        async fn handler_with_vary() -> Response {
            (
                StatusCode::OK,
                [(axum::http::header::VARY, "Accept-Encoding")],
                "ok",
            )
                .into_response()
        }

        let cors = CorsConfig::from_origins(["https://app.example.com".to_string()]);
        let inner = Router::new().route("/", get(handler_with_vary));
        let router = apply_middleware(inner, MiddlewareConfig::none().with_cors(cors));

        let resp = router
            .oneshot(get_request("/", Some("https://app.example.com")))
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);

        let vary_values: Vec<&str> = resp
            .headers()
            .get_all("vary")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .collect();
        assert!(
            vary_values.contains(&"Accept-Encoding"),
            "the handler's own Vary: Accept-Encoding must survive; got {vary_values:?}"
        );
        assert!(
            vary_values.contains(&"Origin"),
            "cors_mw must still add Vary: Origin; got {vary_values:?}"
        );
    }

    /// `/admin/*` must never receive a CORS grant, even when the configured
    /// policy would otherwise allow the request's origin on a normal route
    /// (finding sec-06: "Never apply permissive CORS to /admin/*").
    #[tokio::test]
    async fn admin_routes_never_get_cors_headers() {
        let cors = CorsConfig::from_origins(["https://app.example.com".to_string()]);
        let router = apply_middleware(base_router(), MiddlewareConfig::none().with_cors(cors));

        // Sanity: the same origin IS granted on a non-admin route.
        let resp = router
            .clone()
            .oneshot(get_request("/", Some("https://app.example.com")))
            .await
            .expect("non-admin response");
        assert!(header(&resp, "access-control-allow-origin").is_some());

        let admin_resp = router
            .oneshot(get_request(
                "/admin/status",
                Some("https://app.example.com"),
            ))
            .await
            .expect("admin response");
        assert_eq!(admin_resp.status(), StatusCode::OK);
        assert!(
            header(&admin_resp, "access-control-allow-origin").is_none(),
            "/admin/* must never receive a CORS grant regardless of configuration"
        );
    }

    /// A stand-in "deny-all" auth layer, mirroring the shape of the real
    /// bearer-auth middleware this crate does not own (`oxibonsai-serve`'s
    /// `middleware::bearer_auth` / the CLI's `admission::bearer_auth`):
    /// rejects every request with `401` regardless of path or method.
    async fn deny_all_auth(_req: Request<Body>, _next: Next) -> Response {
        StatusCode::UNAUTHORIZED.into_response()
    }

    /// The layer-order contract (module docs, "Layer order"): CORS must be
    /// the outermost layer relative to auth, and its `OPTIONS` preflight
    /// short-circuit must return success *before* reaching a nested auth
    /// layer (finding SV-06) -- while a normal `GET` still reaches (and is
    /// rejected by) that same auth layer, proving the short-circuit is
    /// preflight-specific, not a blanket auth bypass.
    #[tokio::test]
    async fn preflight_bypasses_a_nested_auth_layer_but_get_does_not() {
        let cors = CorsConfig::from_origins(["https://app.example.com".to_string()]);
        // `apply_middleware` mounts CORS as the outermost of its own
        // bundle; layering the deny-all auth stand-in on the *base* router
        // first places it *inside* CORS once `apply_middleware` wraps it,
        // exactly matching the required `cors -> auth -> ...` order.
        let inner = Router::new()
            .route("/v1/chat/completions", get(handler))
            .layer(middleware::from_fn(deny_all_auth));
        let router = apply_middleware(inner, MiddlewareConfig::none().with_cors(cors));

        let get_resp = router
            .clone()
            .oneshot(get_request(
                "/v1/chat/completions",
                Some("https://app.example.com"),
            ))
            .await
            .expect("get response");
        assert_eq!(
            get_resp.status(),
            StatusCode::UNAUTHORIZED,
            "a normal GET must still reach and be rejected by the nested auth layer"
        );

        let preflight_resp = router
            .oneshot(options_request("/v1/chat/completions"))
            .await
            .expect("preflight response");
        assert_eq!(
            preflight_resp.status(),
            StatusCode::OK,
            "an OPTIONS preflight must bypass the nested auth layer entirely \
             (finding SV-06) -- a browser could otherwise never use an \
             authenticated server"
        );
    }
}
