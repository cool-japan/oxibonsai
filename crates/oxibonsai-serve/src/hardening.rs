//! Router hardening: bind-safety, bearer/admin auth resolution, checksum
//! verification, extra `OXIBONSAI_*` env overrides, and the fully-composed
//! router (findings `sec-15`/`SV-07`/`sec-M2`, `sec-07`/`RT-34`/`SV-10`,
//! `cli-18`, `SV-06`, `SV-15`/`SV-16`/`SV-23`, `sec-20`/`perf-M1`,
//! `SV-27`/`sec-16`, `SV-24`, `SV-33`, `sec-12`).
//!
//! Split out of `main.rs` to keep that file under the 2000-line policy
//! limit; `main.rs::run()` is the only caller.
//!
//! [`build_router`] composes the served router in the order mandated by
//! `oxibonsai_runtime::middleware`'s and `oxibonsai_runtime::rate_limiter`'s
//! module docs, innermost first:
//!
//! ```text
//! routes -> metrics_gate_mw -> admission -> DefaultBodyLimit
//!        -> rate-limit -> bearer-auth -> CORS (outermost)
//! ```
//!
//! (`DefaultBodyLimit` moved from inside `admission` to
//! outside it, still inside rate-limit -- see the note at its call site in
//! [`build_router`] for what this does and does not achieve.)

use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::error_handling::HandleErrorLayer;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{BoxError, Json, Router};
use oxibonsai_runtime::engine_pool::EnginePool;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::middleware::{apply_middleware, CorsConfig, MiddlewareConfig};
use oxibonsai_runtime::rate_limiter::{rate_limit_layer, RateLimitConfig};
use oxibonsai_runtime::server::{
    create_router_full, AuthConfig as AdminAuthConfig, RequestLimits, RouterOptions,
};
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_serve::config::{PartialServerConfig, SamplingConfig, ServerConfig};
use oxibonsai_serve::metrics::MetricsRegistry;
use tower::ServiceBuilder;

use crate::middleware as bearer_middleware;

// ─── sec-15 / SV-07 / sec-M2: bind-safety guard ─────────────────────────────

/// Returns `true` when `host` resolves to a loopback address (`127.0.0.0/8`,
/// `::1`, or the literal `"localhost"`).
pub fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Resolve the `/admin/*` authentication policy: an explicit
/// `auth.admin_token` (`--admin-token` / TOML / `OXIBONSAI_ADMIN_TOKEN`)
/// takes precedence; otherwise fall back to the `OXI_ADMIN_TOKEN`
/// environment variable (`AdminAuthConfig::from_env`, the pre-existing
/// convention every router constructor already honors), and finally to
/// locked (`403` on every `/admin/*` request — finding `sec-15`).
pub fn resolve_admin_auth(config: &ServerConfig) -> AdminAuthConfig {
    match config.auth.admin_token.as_deref() {
        Some(token) => AdminAuthConfig::with_admin_token(token.to_string()),
        None => AdminAuthConfig::from_env(),
    }
}

/// Resolve the effective inference bearer token (finding `SV-33`):
/// `auth.bearer_token_file`, when set, is read and trimmed and takes
/// precedence — it is the `ps`-safe option this finding recommends over a
/// raw command-line value; otherwise `auth.bearer_token` (from
/// `--bearer-token` / TOML / `OXIBONSAI_BEARER_TOKEN`) is used as-is.
/// `ServerConfig::validate` (called inside `ServerConfig::load`) already
/// confirmed the file exists; a read failure here (permissions, a TOCTOU
/// removal) is still surfaced as a startup error rather than silently
/// falling back to an unauthenticated server.
pub fn resolve_bearer_token(config: &ServerConfig) -> Result<Option<String>, String> {
    if let Some(ref path) = config.auth.bearer_token_file {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            format!(
                "failed to read auth.bearer_token_file {}: {e}",
                path.display()
            )
        })?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(format!(
                "auth.bearer_token_file {} is empty",
                path.display()
            ));
        }
        return Ok(Some(trimmed.to_string()));
    }
    Ok(config.auth.bearer_token.clone())
}

/// Refuse to bind a non-loopback host with no auth configured, unless the
/// operator explicitly acknowledges the risk (findings `sec-15` / `SV-07` /
/// `sec-M2`).
///
/// Deliberately checks the **inference** bearer token specifically (not
/// `admin_enabled`, which is passed only for the diagnostic message): an
/// admin token alone protects `/admin/*` but leaves `/v1/chat/completions`
/// and friends open to the network, which is the actual danger this guard
/// exists to prevent.
pub fn bind_safety_check(config: &ServerConfig, admin_enabled: bool) -> Result<(), String> {
    if is_loopback_host(&config.bind.host) {
        return Ok(());
    }
    if config.auth.bearer_token.is_some() || config.auth.insecure_no_auth {
        return Ok(());
    }
    Err(format!(
        "refusing to bind to non-loopback host '{host}' with no bearer token configured \
         (admin_token {admin_state}): set auth.bearer_token / --bearer-token / \
         OXIBONSAI_BEARER_TOKEN, bind to a loopback address instead (127.0.0.1, ::1, or \
         localhost), or explicitly acknowledge the risk with --insecure-no-auth / \
         auth.insecure_no_auth = true / OXIBONSAI_INSECURE_NO_AUTH=1",
        host = config.bind.host,
        admin_state = if admin_enabled {
            "configured"
        } else {
            "also not configured"
        },
    ))
}

/// SV-15(c): the per-request fallback `max_tokens` (`ChatCompletionRequest`'s
/// `#[serde(default = "default_max_tokens")]`, in
/// `crates/oxibonsai-runtime/src/server.rs`) is a
/// hardcoded literal with no config seam. Rather than silently discarding a
/// configured `sampling.default_max_tokens` that disagrees with it, name the
/// discrepancy loudly so an operator who set it is not left guessing why
/// requests that omit `max_tokens` behave differently than expected.
/// Historical name kept for API stability (`main.rs` still calls it) — the
/// mismatch this used to warn about is now fixed, not merely explained.
///
/// SV-15(c) is resolved for real: [`build_router`] threads
/// `sampling.default_max_tokens` through
/// `RouterOptions::with_default_max_tokens`, so
/// `oxibonsai_runtime::server::resolve_effective_max_tokens` (via
/// `AppState::default_max_tokens`) reads the *same* configured value this
/// function is given — the two can never diverge by construction, not just
/// by convention, so there is nothing left to warn about on the steady-state
/// path. This now only logs (at `debug`, not `warn`) as a defense-in-depth
/// trip-wire: if a future refactor ever reintroduces two independent
/// sources of truth for the default, the two values could diverge again,
/// and this would be the one place positioned to notice.
pub fn warn_if_default_max_tokens_diverges(sampling: &SamplingConfig) {
    tracing::debug!(
        configured_default_max_tokens = sampling.default_max_tokens,
        "sampling.default_max_tokens is threaded through RouterOptions::with_default_max_tokens \
         into AppState::default_max_tokens (SV-15(c)); this log line is a defense-in-depth \
         trip-wire, not a mismatch report"
    );
}

// ─── sec-12 / SV-30 / sec-M3: model checksum verification ──────────────────
//
// `lookup_expected_checksum` / `verify_model_checksum` (which computes a
// real streaming SHA-256 via `sha256_file` -- no stub, no `Ok(None)`
// shortcut) now live once, canonically, in
// `oxibonsai_runtime::serve_shared` (also used by `src/cli/admission.rs`),
// closing the "duplicated verbatim across two binaries" finding at the same
// time as the checksum stub. `verify_model_checksum` needs `pub` here since
// `main.rs` calls it as `hardening::verify_model_checksum`.
pub use oxibonsai_runtime::serve_shared::verify_model_checksum;

// ─── OXIBONSAI_* env vars this crate's `env.rs` does not (yet) map ─────────

/// Fold the `OXIBONSAI_*` environment variables introduced in this crate
/// (admin token, insecure-no-auth, CORS, rate limit, body size) into an
/// already-parsed [`PartialServerConfig`], preserving the documented
/// `defaults < TOML < env < CLI` precedence (this runs at the "env" layer,
/// before the CLI partial is merged on top in `main.rs::run()`).
///
/// Kept here rather than in `oxibonsai_serve::env` — a future consolidation
/// should fold this back into `env::parse_env_map`.
pub fn apply_extra_env_overrides<I>(partial: &mut PartialServerConfig, vars: I)
where
    I: IntoIterator<Item = (String, String)>,
{
    for (key, value) in vars {
        match key.as_str() {
            "OXIBONSAI_ADMIN_TOKEN" if !value.is_empty() => {
                partial.admin_token = Some(value);
            }
            "OXIBONSAI_INSECURE_NO_AUTH" => {
                if let Some(b) = parse_bool_env(&value) {
                    partial.insecure_no_auth = Some(b);
                }
            }
            "OXIBONSAI_CORS_ORIGIN" => {
                let origins: Vec<String> = value
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                if !origins.is_empty() {
                    partial.cors_allowed_origins = Some(origins);
                }
            }
            "OXIBONSAI_CORS_ALLOW_CREDENTIALS" => {
                if let Some(b) = parse_bool_env(&value) {
                    partial.cors_allow_credentials = Some(b);
                }
            }
            "OXIBONSAI_RATE_LIMIT_RPM" => {
                if let Ok(f) = value.parse::<f64>() {
                    partial.rate_limit_rpm = Some(f);
                }
            }
            "OXIBONSAI_RATE_LIMIT_BURST" => {
                if let Ok(f) = value.parse::<f64>() {
                    partial.rate_limit_burst = Some(f);
                }
            }
            "OXIBONSAI_MAX_BODY_BYTES" => {
                if let Ok(n) = value.parse::<usize>() {
                    partial.max_body_bytes = Some(n);
                }
            }
            _ => {}
        }
    }
}

/// Parse a boolean env var accepting the same spellings as
/// `oxibonsai_serve::env`'s `parse_bool` (kept independent since that
/// function is private to an unowned file).
fn parse_bool_env(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

// ─── Router composition (SV-06 / sec-06 / sec-07 / SV-10 / cli-18) ─────────

/// Convert this crate's `LimitsConfig` into the `RequestLimits` the handlers
/// in `oxibonsai_runtime::server` already consult (finding `SV-15`/`SV-16`/
/// `SV-23`: the fields were validated at startup and then never reached
/// `AppState` at all, because every router constructor was called with
/// `RouterOptions::default()`, whose `RequestLimits::default()` leaves both
/// `max_input_tokens` and `per_request_timeout` at `None`).
///
/// Split out from [`build_router`] so this specific conversion is directly
/// unit-testable without standing up a router or making an HTTP request.
pub fn resolve_request_limits(config: &ServerConfig) -> RequestLimits {
    RequestLimits::default()
        .with_max_input_tokens(Some(config.limits.max_input_tokens))
        .with_timeout_ms(config.limits.per_request_timeout_ms)
}

/// Derive the real admission ceiling from the engine pool's actual size
/// (findings `sec-20` / `perf-M1`): measured serve throughput is strictly
/// serialized on the GPU tier (pool size clamped to 1), so admitting the
/// configured `limits.max_concurrent_requests` (default 32) queues every
/// excess request behind that one engine until the 60s timeout fires,
/// instead of shedding it immediately with a fast `503` + `Retry-After`.
///
/// The single canonical implementation now lives in
/// `oxibonsai_runtime::serve_shared` (findings `SV-30` / `sec-M3`);
/// `src/cli/admission.rs::resolve_admission_limit` re-exports the same
/// function rather than keeping its own byte-for-byte copy. `pub` here
/// since [`build_router`] below calls it unqualified via this re-export.
pub use oxibonsai_runtime::serve_shared::resolve_admission_limit;

/// Bundles [`build_router`]'s knobs that aren't the pool/tokenizer/metrics
/// triple or `config` itself.
///
/// `enable_ui` and `max_output_tokens_ceiling` join `admin_auth` and
/// `pool_size` here rather than as positional parameters: nine positional
/// arguments would trip clippy's `too_many_arguments` (limit 7). Grouping
/// them here (rather than adding `#[allow(...)]`) also gives the two call
/// sites (`main.rs`'s real server and this module's own
/// `build_router_tests`) one named place to construct the bundle instead of
/// a positional list that silently tolerates argument-order mistakes.
pub struct RouterBuildOptions {
    /// Admin authentication policy (`sec-15`).
    pub admin_auth: AdminAuthConfig,
    /// The engine pool's real size, used to derive the admission ceiling
    /// (findings `sec-20`/`perf-M1` — see [`resolve_admission_limit`]).
    pub pool_size: usize,
    /// Whether to mount the bundled chat UI (`SV-26`) — the merged config's
    /// `ui.enabled` (CLI flag > env > TOML > `false`).
    pub enable_ui: bool,
    /// The hard `max_tokens` ceiling override (`SV-28`) — the merged
    /// config's `limits.max_output_tokens` (CLI flag > env > TOML), `None`
    /// to keep the router's own default ceiling.
    pub max_output_tokens_ceiling: Option<usize>,
    /// The model-backed embedder `/v1/embeddings` answers from
    /// (built by `oxibonsai_serve::embedder`), or
    /// `None` for the route's honest `501`.
    pub embedder: Option<Arc<oxibonsai_runtime::embed_engine::ModelEmbedder>>,
    /// Why there is no model-backed embedder, carried into the `/v1/embeddings`
    /// `501` body (`error.code` and the message suffix) when [`Self::embedder`]
    /// is `None` — see [`Self::with_embedder_unavailable`].
    pub embedder_unavailable: Option<(Option<&'static str>, String)>,
    /// The served engine's resolved variant and effective kernel tier, shown
    /// by `/admin/status` and `/admin/config` — see
    /// [`Self::with_engine_report`].
    pub engine_report: Option<oxibonsai_runtime::admin::EngineReport>,
    /// The single token id a tokenizer-less server feeds as the prompt of a
    /// text request — see [`Self::with_prompt_start_token`]. Library/test
    /// use only: a served model always resolves a real tokenizer or a real
    /// GGUF-embedded one.
    pub prompt_start_token: Option<u32>,
}

impl RouterBuildOptions {
    /// Name every field positionally -- the shape both call sites already
    /// had before this bundle existed. No embedder, unavailability reason,
    /// engine report or prompt-start-token override: see the `with_*`
    /// builders.
    pub fn new(
        admin_auth: AdminAuthConfig,
        pool_size: usize,
        enable_ui: bool,
        max_output_tokens_ceiling: Option<usize>,
    ) -> Self {
        Self {
            admin_auth,
            pool_size,
            enable_ui,
            max_output_tokens_ceiling,
            embedder: None,
            embedder_unavailable: None,
            engine_report: None,
            prompt_start_token: None,
        }
    }

    /// Serve `/v1/embeddings` from `embedder` (builder).
    #[must_use]
    pub fn with_embedder(
        mut self,
        embedder: Option<Arc<oxibonsai_runtime::embed_engine::ModelEmbedder>>,
    ) -> Self {
        self.embedder = embedder;
        self
    }

    /// Record why this server has no model-backed embedder, so the
    /// `/v1/embeddings` `501` body names it (builder).
    #[must_use]
    pub fn with_embedder_unavailable(
        mut self,
        code: Option<&'static str>,
        message: impl Into<String>,
    ) -> Self {
        self.embedder_unavailable = Some((code, message.into()));
        self
    }

    /// Attach the served engine's [`oxibonsai_runtime::admin::EngineReport`]
    /// to the `/admin/*` router (builder).
    #[must_use]
    pub fn with_engine_report(mut self, report: oxibonsai_runtime::admin::EngineReport) -> Self {
        self.engine_report = Some(report);
        self
    }

    /// Let a tokenizer-less server answer text prompts by feeding the single
    /// token `id` as the prompt (builder). Test-only: the standalone binary
    /// always resolves a real tokenizer or refuses to start; a production
    /// build of this crate has no caller.
    #[cfg(test)]
    #[must_use]
    pub fn with_prompt_start_token(mut self, id: u32) -> Self {
        self.prompt_start_token = Some(id);
        self
    }
}

/// Compose the fully-hardened router.
///
/// Layer order, innermost first (see the module docs of
/// `oxibonsai_runtime::middleware` and `oxibonsai_runtime::rate_limiter` for
/// why this exact order is required):
///
/// ```text
/// routes -> metrics_gate_mw -> admission -> DefaultBodyLimit
///        -> rate-limit -> bearer-auth -> CORS (outermost)
/// ```
///
/// `/admin/*` authentication (`build_options.admin_auth`) is unconditional
/// and independent of this stack (`create_router_full` wraps it internally
/// — finding `sec-15`); `RequestLimits` carries `limits.max_input_tokens`
/// and `limits.per_request_timeout_ms` into the handlers that already
/// consult them (finding `SV-15`/`SV-16`/`SV-23` — the validation existed,
/// the wiring did not); `build_options.pool_size` derives the real
/// admission ceiling (findings `sec-20`/`perf-M1` — see
/// [`resolve_admission_limit`]).
///
/// `build_options.enable_ui` (`SV-26`) and
/// `build_options.max_output_tokens_ceiling` (`SV-28`) come from the merged
/// `config`'s own `ui.enabled` / `limits.max_output_tokens`
/// (`oxibonsai-serve/src/config.rs`, CLI flag > env > TOML > default) —
/// `main.rs` reads them off `config` (not off `ServerArgs` directly) and
/// passes them here via [`RouterBuildOptions`], the same way every other
/// `build_router` knob arrives.
pub fn build_router(
    pool: Arc<EnginePool>,
    tokenizer: Option<TokenizerBridge>,
    metrics: Arc<InferenceMetrics>,
    serve_metrics: Arc<MetricsRegistry>,
    config: &ServerConfig,
    build_options: RouterBuildOptions,
) -> Router {
    let RouterBuildOptions {
        admin_auth,
        pool_size,
        enable_ui,
        max_output_tokens_ceiling,
        embedder,
        embedder_unavailable,
        engine_report,
        prompt_start_token,
    } = build_options;

    let mut router_options = RouterOptions::default()
        .with_limits(resolve_request_limits(config))
        .with_auth(admin_auth)
        .with_default_max_tokens(config.sampling.default_max_tokens)
        .with_enable_ui(enable_ui)
        .with_embedder(embedder);
    if let Some(ceiling) = max_output_tokens_ceiling {
        router_options = router_options.with_max_output_tokens_ceiling(ceiling);
    }
    if let Some((code, message)) = embedder_unavailable {
        router_options = router_options.with_embedder_unavailable(code, message);
    }
    if let Some(report) = engine_report {
        router_options = router_options.with_engine_report(report);
    }
    if let Some(id) = prompt_start_token {
        router_options = router_options.with_prompt_start_token(id);
    }

    let mut base_router = create_router_full(pool, tokenizer, Arc::clone(&metrics), router_options);

    // SV-24: mount this crate's own Prometheus registry (previously built
    // and never served) at a distinct path -- `/metrics` continues to serve
    // `oxibonsai_runtime::metrics::InferenceMetrics` unchanged. The handler
    // is mounted via a capturing closure (not an axum `State<S>` extractor)
    // because `base_router` is already state-erased (`Router<()>`, returned
    // fully-stated by `create_router_full`); `from_fn_with_state`, unlike
    // `.route()`, bakes its state into the middleware independently of the
    // router's own state type, so it needs no such wrapper.
    let serve_metrics_for_route = Arc::clone(&serve_metrics);
    base_router = base_router
        .route(
            "/metrics/serve",
            get(move || {
                let registry = Arc::clone(&serve_metrics_for_route);
                async move { serve_metrics_handler(registry).await }
            }),
        )
        .layer(axum::middleware::from_fn_with_state(
            serve_metrics,
            record_http_metrics,
        ));

    // SV-15(b): `observability.metrics_path`, when it differs from the
    // hardcoded `/metrics`, is honored by mounting an EXTRA route that
    // renders the exact same `InferenceMetrics::render_prometheus()` output
    // (captured via `metrics`, the same handle already passed into
    // `create_router_full` above) -- *not* by rewriting requests toward
    // `/metrics` inside a layer. `axum::Router::layer()` applies to each
    // already-*resolved* endpoint (confirmed empirically: routing happens
    // before any `.layer()`-added code runs), so a layer cannot redirect an
    // unmatched path to a different registered route by mutating the
    // request URI -- there is no "re-route" hook to call into.
    if config.observability.metrics_path != "/metrics" {
        let metrics_for_alias = Arc::clone(&metrics);
        base_router = base_router.route(
            config.observability.metrics_path.as_str(),
            get(move || {
                let m = Arc::clone(&metrics_for_alias);
                async move { render_metrics_text(&m) }
            }),
        );
    }

    // `observability.metrics_enabled` gates both `/metrics` and the alias
    // route above (a no-op on every other route -- the check is path-scoped
    // inside the middleware, since `.layer()` cannot be targeted at a
    // specific path).
    let metrics_gate_cfg = Arc::new(MetricsGateConfig {
        enabled: config.observability.metrics_enabled,
        custom_path: config.observability.metrics_path.clone(),
    });
    base_router = base_router.layer(axum::middleware::from_fn_with_state(
        metrics_gate_cfg,
        metrics_gate_mw,
    ));

    // Innermost of the "admission" group: bounded concurrency + per-request
    // timeout, both bridged back into JSON HTTP responses via
    // `HandleErrorLayer` (axum requires an `Infallible` error type on the
    // outermost service). `GlobalConcurrencyLimitLayer` (not
    // `tower::limit::ConcurrencyLimitLayer` via `ServiceBuilder`) is
    // required: `Router::layer` applies a given `Layer` independently to
    // *every registered route*, so a bare `ConcurrencyLimitLayer` -- which
    // allocates its own `Arc<Semaphore>` inside `Layer::layer()` -- would
    // silently create one independent semaphore *per route* instead of one
    // shared budget across the whole HTTP surface.
    //
    // Applied here, BEFORE `DefaultBodyLimit` below, so
    // `DefaultBodyLimit` ends up mounted OUTSIDE (more outer than) this
    // admission stack -- see that call site's note for what this reorder
    // does and does not achieve.
    let effective_max_concurrent_requests =
        resolve_admission_limit(config.limits.max_concurrent_requests, pool_size);
    let concurrency_semaphore =
        tower::limit::GlobalConcurrencyLimitLayer::new(effective_max_concurrent_requests);
    let admission = ServiceBuilder::new()
        .layer(HandleErrorLayer::new(handle_admission_error))
        .load_shed()
        .layer(concurrency_semaphore)
        .timeout(Duration::from_millis(config.limits.per_request_timeout_ms));
    let mut router = base_router.layer(admission);

    // SV-27/sec-16: explicit, configurable request body ceiling instead of
    // axum's implicit 2 MiB default.
    //
    // Moved from *inside* `admission` (above) to
    // *outside* it (still inside rate-limit, below) so the mandated
    // ordering documented at the top of this file holds. This reorder is a
    // correctness-neutral, forward-looking position fix, not a fix for the
    // permit-holding concern the finding named: `axum`'s `DefaultBodyLimit`
    // (`axum_core::extract::default_body_limit::DefaultBodyLimitService::call`)
    // only inserts a request *extension* consulted later by the
    // `Bytes`/`Json` extractors deep inside the eventual handler -- it
    // unconditionally calls its inner service with no check of its own, at
    // either position. So a request whose declared size already exceeds
    // the limit still has to reach the handler's body-collecting extractor
    // (past `admission`, wherever `DefaultBodyLimit` sits) before anything
    // rejects it, and `tower::load_shed` decides the "is a concurrency
    // permit available" question before either layer is reached. Sitting
    // here is still the right relative position for when a real
    // synchronous body-size guard (e.g. a `Content-Length` precheck, or
    // `tower_http::limit::RequestBodyLimitLayer`) is added outside
    // `admission`.
    router = router.layer(DefaultBodyLimit::max(config.limits.max_body_bytes));

    // An ACTIVE synchronous `Content-Length` precheck, mounted at this same
    // "outside admission" position, so a request whose
    // *declared* size already exceeds the limit is rejected with a real
    // `413` before it can hold (or wait for) a concurrency permit at all --
    // closing the gap `DefaultBodyLimit` alone does not (see the note on
    // that line above). See [`content_length_guard_mw`]'s own doc comment
    // for why this does not regress a legitimate streaming-body request.
    router = router.layer(axum::middleware::from_fn_with_state(
        Arc::new(config.limits.max_body_bytes),
        content_length_guard_mw,
    ));

    // sec-07/RT-34/SV-10/cli-10: rate limiting, mounted *outside* admission
    // so a rate-limited request never consumes a concurrency permit
    // (cli-18) -- `rate_limit_layer` itself exempts `/health`/`/metrics`.
    if let Some(rpm) = config.rate_limit.rpm {
        let rate_cfg = RateLimitConfig::from_rpm(rpm, config.rate_limit.burst);
        router = router.layer(rate_limit_layer(rate_cfg));
    }

    // sec-15/SV-07/sec-07: bearer auth, mounted *outside* rate limiting so
    // an unauthenticated request never consumes a rate-limit token either.
    if let Some(ref token) = config.auth.bearer_token {
        let state = bearer_middleware::BearerAuthState {
            token: token.clone(),
        };
        router = router.layer(axum::middleware::from_fn_with_state(
            state,
            bearer_middleware::bearer_auth,
        ));
    }

    // SV-06/sec-06/cli-10: CORS is the outermost layer, so its `OPTIONS`
    // preflight short-circuit (`200 OK`, no body) runs *before* the nested
    // bearer-auth layer ever sees the request -- a browser can otherwise
    // never use an authenticated server at all.
    if !config.cors.allowed_origins.is_empty() {
        let cors = CorsConfig {
            allow_credentials: config.cors.allow_credentials,
            ..CorsConfig::from_origins(config.cors.allowed_origins.clone())
        };
        router = apply_middleware(router, MiddlewareConfig::none().with_cors(cors));
    }

    router
}

/// Convert an admission-layer error (an overloaded `load_shed` or an elapsed
/// `timeout`) into an OpenAI-style JSON error response.
///
/// Required because axum's `Router` demands an `Infallible` error type on the
/// outermost service; `HandleErrorLayer` is the documented bridge from the
/// `tower::BoxError` the admission stack produces back into a `Response`. See
/// <https://docs.rs/axum/latest/axum/error_handling/index.html>.
///
/// sec-20/perf-M1: the `Overloaded` (`503`) branch carries a `Retry-After`
/// header -- this module's own doc comment on [`resolve_admission_limit`]
/// promises "a fast `503` + `Retry-After`", which only the `429` rate-limit
/// path (`oxibonsai_runtime::rate_limiter`) actually delivered until now.
async fn handle_admission_error(err: BoxError) -> Response {
    if err.is::<tower::load_shed::error::Overloaded>() {
        let body = Json(serde_json::json!({
            "error": {
                "message": "server is at its configured limits.max_concurrent_requests \
                             capacity; retry after a short backoff",
                "type": "overloaded_error",
                "param": null,
                "code": null,
            }
        }));
        let mut response = (StatusCode::SERVICE_UNAVAILABLE, body).into_response();
        response.headers_mut().insert(
            HeaderName::from_static("retry-after"),
            HeaderValue::from_static("1"),
        );
        return response;
    }
    if err.is::<tower::timeout::error::Elapsed>() {
        return (
            StatusCode::REQUEST_TIMEOUT,
            Json(serde_json::json!({
                "error": {
                    "message": "request exceeded the configured limits.per_request_timeout_ms \
                                 budget",
                    "type": "timeout_error",
                    "param": null,
                    "code": null,
                }
            })),
        )
            .into_response();
    }
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": {
                "message": format!("unhandled admission-layer error: {err}"),
                "type": "internal_error",
                "param": null,
                "code": null,
            }
        })),
    )
        .into_response()
}

// ─── SV-24: this crate's own Prometheus registry, actually mounted ─────────

/// Serve this crate's [`MetricsRegistry`] as Prometheus text exposition.
/// Distinct from `/metrics` (the runtime's `InferenceMetrics`, unchanged).
///
/// Takes the registry directly (not via an axum `State<S>` extractor) so it
/// can be mounted on an already state-erased `Router<()>` via a capturing
/// closure — see the call site in `build_router`.
async fn serve_metrics_handler(registry: Arc<MetricsRegistry>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        registry.render(),
    )
}

/// Record one HTTP request's route/status counter and duration histogram
/// into this crate's [`MetricsRegistry`] (finding `SV-24`'s "mount it"
/// half — the per-route/TTFT depth the corrected finding tracks separately
/// at P2 is out of scope here). Uses the raw request path (not axum's
/// `MatchedPath`, which is only visible to middleware registered *inside*
/// the router's own routing via `route_layer`, not to a layer applied with
/// `Router::layer`), mapped through [`route_label`] to a bounded label
/// before being recorded.
///
/// **Correcting a false claim a previous version of this comment made**:
/// "every route on this server is a static string with no path parameters,
/// so cardinality is bounded by construction" is WRONG — `axum::Router::layer`
/// also wraps the router's *fallback* service, so an unmatched (404) request
/// reaches this middleware carrying its own raw, client-chosen
/// `req.uri().path()`. Labelling directly with that path (the prior
/// behavior) let any client grow this [`MetricsRegistry`]'s `BTreeMap`s by
/// one permanent counter series plus one ~12-line histogram series per
/// distinct URI it chose to request — reachable by any client on an
/// `--insecure-no-auth` non-loopback listener, by any local process against
/// the loopback default, and by any credentialed client, since this layer
/// is innermost (see the module docs). Verified in-tree before this fix:
/// `GET /attacker-path-0..4` produced five independent
/// `oxibonsai_serve_http_requests_total{route="/attacker-path-N",...}`
/// series. [`route_label`] closes this by collapsing every path outside the
/// fixed allow-list to `"other"`.
async fn record_http_metrics(
    State(registry): State<Arc<MetricsRegistry>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let route = route_label(req.uri().path());
    let start = Instant::now();
    let response = next.run(req).await;
    let status = response.status().as_u16().to_string();
    registry.inc_counter(
        "oxibonsai_serve_http_requests_total",
        &[("route", route), ("status", status.as_str())],
    );
    registry.observe_histogram(
        "oxibonsai_serve_http_request_duration_seconds",
        &[("route", route)],
        start.elapsed().as_secs_f64(),
    );
    response
}

/// Map a raw request path to a bounded metrics-cardinality label.
///
/// Anything not literally present in
/// [`oxibonsai_serve::validation::RESERVED_ROUTE_PATHS`] — every unmatched
/// (404) path, and (within the letter of this fix) every `/admin/*`
/// sub-path and a non-default `observability.metrics_path` alias, none of
/// which are individually listed in that allow-list — collapses to the
/// single `"other"` bucket. Cardinality is therefore bounded by the
/// allow-list's fixed size, not by client-controlled input; a future owner
/// wanting per-admin-route breakdown should extend the allow-list rather
/// than reading this as an oversight.
fn route_label(path: &str) -> &'static str {
    oxibonsai_serve::validation::RESERVED_ROUTE_PATHS
        .iter()
        .find(|&&reserved| reserved == path)
        .copied()
        .unwrap_or("other")
}

// ─── SV-15(b): metrics_enabled / metrics_path gate ─────────────────────────

/// Render an `InferenceMetrics` snapshot as Prometheus text exposition —
/// byte-identical to `oxibonsai_runtime::server`'s own (private)
/// `prometheus_metrics` handler, used here to back a non-default
/// `observability.metrics_path` alias (see `build_router`).
fn render_metrics_text(metrics: &InferenceMetrics) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        metrics.render_prometheus(),
    )
}

/// Runtime configuration for [`metrics_gate_mw`].
struct MetricsGateConfig {
    enabled: bool,
    custom_path: String,
}

/// Gate `/metrics`, this crate's own `/metrics/serve` (finding `SV-24`), and
/// the `observability.metrics_path` alias route `build_router` mounts when
/// it differs from `/metrics` — all three behind `observability.metrics_enabled`
/// (finding `SV-15`(b)). A no-op for every other path.
///
/// `/metrics/serve` was previously missing from this
/// check, so it stayed reachable even when an operator explicitly set
/// `observability.metrics_enabled = false` — and `/metrics/serve` is also in
/// `bearer_auth`'s exempt list (`main.rs`'s `middleware::bearer_auth`,
/// alongside `/health`/`/metrics`, so Prometheus scrapers work without a
/// token), so a disabled-metrics server still served the full serve
/// registry unauthenticated. This middleware is innermost (see the module
/// docs), so the 404 below fires regardless of `bearer_auth`'s exemption.
async fn metrics_gate_mw(
    State(cfg): State<Arc<MetricsGateConfig>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if cfg.enabled {
        return next.run(req).await;
    }
    let path = req.uri().path();
    if path == "/metrics" || path == "/metrics/serve" || path == cfg.custom_path {
        return (
            StatusCode::NOT_FOUND,
            "metrics are disabled on this server (observability.metrics_enabled = false)",
        )
            .into_response();
    }
    next.run(req).await
}

/// Active `Content-Length` precheck.
///
/// `DefaultBodyLimit` (see the note where [`build_router`] applies it)
/// enforces nothing by itself at either layer position: it only stamps a
/// request extension that the *handler's* `Bytes`/`Json` extractor consults
/// once it actually buffers the body, which happens *inside* `admission` --
/// so a flood of oversized bodies can still each briefly hold (or queue for)
/// the sole concurrency permit before being rejected. This middleware
/// rejects a request whose *declared* `Content-Length` already exceeds the
/// limit with a real, synchronous `413` before it is ever routed to
/// `admission` at all, by construction (`Router::layer` wraps every request
/// at the point it is applied, before anything further down the stack sees
/// it), closing that gap.
///
/// Only the header is inspected -- the body is never read, buffered or
/// polled here -- so this cannot regress axum's own documented case where
/// `DefaultBodyLimit` "does not apply to a handler that consumes the
/// request body via `poll_frame`" (a legitimate streaming-body request):
/// nothing about a streaming body's *later* consumption is touched. A
/// request with no `Content-Length` at all (e.g. genuine
/// `Transfer-Encoding: chunked`) is let through unchanged; it is exactly
/// the case `DefaultBodyLimit`'s own extension-based enforcement still
/// covers once the handler starts buffering it.
///
/// `src/cli/cmd_serve.rs::harden_router` mounts the
/// identical guard for the CLI `serve` subcommand's own admission stack,
/// per the SV-30/sec-M3 "one shared module" intent; that file lives in a
/// different crate, so the two copies are kept in sync by hand.
async fn content_length_guard_mw(
    State(max_body_bytes): State<Arc<usize>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let declared_len = req
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok());

    if let Some(len) = declared_len {
        if len > *max_body_bytes {
            let body = serde_json::json!({
                "error": {
                    "message": format!(
                        "request body is {len} bytes (Content-Length), which exceeds the \
                         server's limit of {max_body_bytes} bytes"
                    ),
                    "type": "invalid_request_error",
                    "param": serde_json::Value::Null,
                    "code": "content_too_large",
                }
            });
            return (StatusCode::PAYLOAD_TOO_LARGE, Json(body)).into_response();
        }
    }

    next.run(req).await
}

#[cfg(test)]
mod bind_safety_tests {
    use super::*;

    #[test]
    fn loopback_hosts_are_recognized() {
        for host in ["127.0.0.1", "127.5.5.5", "::1", "localhost", "LOCALHOST"] {
            assert!(is_loopback_host(host), "{host} should be loopback");
        }
    }

    #[test]
    fn non_loopback_hosts_are_not_loopback() {
        for host in ["0.0.0.0", "192.168.1.1", "10.0.0.1", "example.com", ""] {
            assert!(!is_loopback_host(host), "{host} should not be loopback");
        }
    }

    fn base_config() -> ServerConfig {
        ServerConfig::default()
    }

    #[test]
    fn loopback_bind_never_needs_auth() {
        let cfg = base_config();
        assert_eq!(cfg.bind.host, "127.0.0.1");
        bind_safety_check(&cfg, false).expect("loopback bind is always safe");
    }

    #[test]
    fn non_loopback_bind_without_auth_is_refused() {
        let mut cfg = base_config();
        cfg.bind.host = "0.0.0.0".to_string();
        let err = bind_safety_check(&cfg, false).expect_err("must refuse");
        assert!(err.contains("0.0.0.0"));
        assert!(err.contains("insecure-no-auth"));
    }

    #[test]
    fn non_loopback_bind_with_bearer_token_is_allowed() {
        let mut cfg = base_config();
        cfg.bind.host = "0.0.0.0".to_string();
        cfg.auth.bearer_token = Some("x".repeat(20));
        bind_safety_check(&cfg, false).expect("a configured bearer token makes this safe");
    }

    #[test]
    fn non_loopback_bind_with_insecure_no_auth_is_allowed() {
        let mut cfg = base_config();
        cfg.bind.host = "0.0.0.0".to_string();
        cfg.auth.insecure_no_auth = true;
        bind_safety_check(&cfg, false).expect("explicit opt-out is honored");
    }

    #[test]
    fn admin_token_alone_does_not_satisfy_the_guard() {
        // An admin token only protects /admin/*; the inference routes would
        // still be open to the network, so `admin_enabled=true` alone must
        // not bypass this guard.
        let mut cfg = base_config();
        cfg.bind.host = "0.0.0.0".to_string();
        cfg.auth.admin_token = Some("x".repeat(20));
        let err = bind_safety_check(&cfg, true).expect_err("admin token alone is not enough");
        assert!(
            err.contains("configured"),
            "message should reflect admin_enabled=true: {err}"
        );
    }

    #[test]
    fn resolve_admin_auth_prefers_configured_token_over_env() {
        let mut cfg = base_config();
        cfg.auth.admin_token = Some("explicit-admin-token-1234".to_string());
        let auth = resolve_admin_auth(&cfg);
        assert!(auth.admin_enabled());
    }

    #[test]
    fn resolve_admin_auth_locked_when_nothing_configured() {
        // Assumes OXI_ADMIN_TOKEN is not set in the test environment; if a
        // developer's shell happens to export it, this documents the
        // fallback behavior rather than asserting a false negative.
        let cfg = base_config();
        if std::env::var("OXI_ADMIN_TOKEN").is_err() {
            assert!(!resolve_admin_auth(&cfg).admin_enabled());
        }
    }

    // ─── SV-33: resolve_bearer_token ────────────────────────────────────────

    #[test]
    fn resolve_bearer_token_uses_the_plain_field_when_no_file_is_set() {
        let mut cfg = base_config();
        cfg.auth.bearer_token = Some("plain-value".to_string());
        assert_eq!(
            resolve_bearer_token(&cfg).expect("resolve"),
            Some("plain-value".to_string())
        );
    }

    #[test]
    fn resolve_bearer_token_prefers_the_file_over_the_plain_field() {
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_bearer_token_file_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("token");
        std::fs::write(&path, "  from-file-value  \n").expect("write token file");

        let mut cfg = base_config();
        cfg.auth.bearer_token = Some("plain-value".to_string());
        cfg.auth.bearer_token_file = Some(path);

        assert_eq!(
            resolve_bearer_token(&cfg).expect("resolve"),
            Some("from-file-value".to_string()),
            "the file must win, and its contents must be trimmed"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_bearer_token_errors_on_an_empty_file() {
        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_bearer_token_file_test_empty_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("token");
        std::fs::write(&path, "   \n").expect("write empty token file");

        let mut cfg = base_config();
        cfg.auth.bearer_token_file = Some(path);
        let err = resolve_bearer_token(&cfg).expect_err("an empty file must error");
        assert!(err.contains("empty"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_bearer_token_errors_when_file_is_unreadable() {
        let mut cfg = base_config();
        cfg.auth.bearer_token_file =
            Some(std::env::temp_dir().join("oxibonsai_serve_definitely_missing_bearer_token_file"));
        assert!(resolve_bearer_token(&cfg).is_err());
    }

    #[test]
    fn resolve_bearer_token_is_none_when_nothing_configured() {
        let cfg = base_config();
        assert_eq!(resolve_bearer_token(&cfg).expect("resolve"), None);
    }
}

#[cfg(test)]
mod checksum_tests {
    use super::*;
    // Neither `Path` nor `lookup_expected_checksum` has a non-test call
    // site left in this *binary* target now that `verify_model_checksum`'s
    // implementation lives in `oxibonsai_runtime::serve_shared` -- imported
    // here, inside `#[cfg(test)]`, rather than at module level, so a
    // non-test build of this bin target does not see (and cannot warn
    // about) an otherwise-unused import/re-export.
    use oxibonsai_runtime::serve_shared::lookup_expected_checksum;
    use std::path::Path;

    #[test]
    fn lookup_matches_by_bare_filename() {
        let text = "\
# comment line
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo-Q2_0.gguf
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  models/Bar.gguf
";
        let got = lookup_expected_checksum(text, Path::new("/somewhere/else/Bar.gguf"));
        assert_eq!(
            got,
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string())
        );
    }

    #[test]
    fn lookup_is_none_for_unknown_file() {
        let text =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo.gguf\n";
        assert!(lookup_expected_checksum(text, Path::new("models/Unknown.gguf")).is_none());
    }

    #[test]
    fn lookup_ignores_malformed_lines() {
        let text = "not-a-valid-hex-line models/Foo.gguf\ntoo-short  models/Foo.gguf\n";
        assert!(lookup_expected_checksum(text, Path::new("models/Foo.gguf")).is_none());
    }

    #[test]
    fn verify_is_a_noop_when_checksums_file_is_absent() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_checksum_test_{}",
            std::process::id()
        ));
        let missing_checksums = tmp_dir.join("does-not-exist.sha256");
        let model = tmp_dir.join("Model.gguf");
        verify_model_checksum(&model, &missing_checksums)
            .expect("missing checksums file must not block loading");
    }

    #[test]
    fn verify_is_a_noop_when_no_entry_matches() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_checksum_test2_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let checksums_path = tmp_dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Other.gguf\n",
        )
        .expect("write checksums file");
        let model = tmp_dir.join("Model.gguf");
        verify_model_checksum(&model, &checksums_path)
            .expect("no matching entry must not block loading");
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // NOTE (sec-12): an earlier version of this test
    // ("verify_warns_but_does_not_fail_when_hasher_is_unavailable") never
    // actually wrote a model file, only the checksums entry -- so it
    // exercised `verify_model_checksum`'s `Err(_)` (unreadable file)
    // branch, not an "unavailable hasher" branch, regardless of what
    // `compute_sha256_hex` returned. It is replaced below by tests that
    // exercise the real equality/mismatch branches end to end through this
    // binary's own re-exported `verify_model_checksum`, plus one correctly
    // named/scoped test for the unreadable-file case it was accidentally
    // covering.

    #[test]
    fn verify_succeeds_when_the_checksum_really_matches() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_checksum_test_matches_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let model = tmp_dir.join("Model.gguf");
        std::fs::write(&model, b"the real bytes of a model file").expect("write model");
        let hex = oxibonsai_runtime::serve_shared::compute_sha256_hex(&model)
            .expect("compute hex")
            .expect("digest is always Some for a readable file");
        let checksums_path = tmp_dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            format!(
                "{hex}  {}\n",
                model.file_name().expect("filename").to_string_lossy()
            ),
        )
        .expect("write checksums file");

        verify_model_checksum(&model, &checksums_path)
            .expect("a genuinely matching checksum must not block loading");
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    /// THE regression test finding `sec-12` was missing: a known-good
    /// checksum entry that no longer matches the file on disk must be
    /// **fatal** -- this is `verify_model_checksum`'s one fatal branch, and
    /// it was unreachable dead code while `compute_sha256_hex` always
    /// returned `Ok(None)`.
    #[test]
    fn verify_fails_when_a_known_good_entry_no_longer_matches_a_corrupted_file() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_checksum_test_corrupted_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let model = tmp_dir.join("Model.gguf");

        std::fs::write(&model, b"the original, known-good model bytes").expect("write model");
        let good_hex = oxibonsai_runtime::serve_shared::compute_sha256_hex(&model)
            .expect("compute hex")
            .expect("digest is always Some for a readable file");
        let checksums_path = tmp_dir.join("checksums.sha256");
        std::fs::write(
            &checksums_path,
            format!(
                "{good_hex}  {}\n",
                model.file_name().expect("filename").to_string_lossy()
            ),
        )
        .expect("write checksums file");

        // Corrupt the file in place; the checksums file still lists the
        // ORIGINAL (now stale) digest as "known-good".
        std::fs::write(
            &model,
            b"corrupted! these are not the original bytes at all",
        )
        .expect("corrupt model");

        let err = verify_model_checksum(&model, &checksums_path)
            .expect_err("a stale known-good entry against corrupted bytes must be fatal");
        assert!(
            err.contains("checksum mismatch"),
            "error should explain the mismatch: {err}"
        );
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn verify_degrades_gracefully_when_the_model_file_cannot_be_opened() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "oxibonsai_serve_checksum_test_missing_model_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let model = tmp_dir.join("Model.gguf"); // Deliberately never created.
        let checksums_path = tmp_dir.join("checksums.sha256");
        std::fs::write(&checksums_path, format!("{}  Model.gguf\n", "c".repeat(64)))
            .expect("write checksums file");
        verify_model_checksum(&model, &checksums_path)
            .expect("an unreadable model file must not block loading (degrades to a warning)");
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}

#[cfg(test)]
mod env_override_tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn admin_token_and_insecure_no_auth_are_read() {
        let mut partial = PartialServerConfig::default();
        apply_extra_env_overrides(
            &mut partial,
            vars(&[
                ("OXIBONSAI_ADMIN_TOKEN", "admin-secret-value"),
                ("OXIBONSAI_INSECURE_NO_AUTH", "true"),
            ]),
        );
        assert_eq!(partial.admin_token.as_deref(), Some("admin-secret-value"));
        assert_eq!(partial.insecure_no_auth, Some(true));
    }

    #[test]
    fn cors_origin_list_is_comma_split_and_trimmed() {
        let mut partial = PartialServerConfig::default();
        apply_extra_env_overrides(
            &mut partial,
            vars(&[(
                "OXIBONSAI_CORS_ORIGIN",
                "https://a.example.com, https://b.example.com ,",
            )]),
        );
        assert_eq!(
            partial.cors_allowed_origins,
            Some(vec![
                "https://a.example.com".to_string(),
                "https://b.example.com".to_string()
            ])
        );
    }

    #[test]
    fn rate_limit_and_body_bytes_are_parsed() {
        let mut partial = PartialServerConfig::default();
        apply_extra_env_overrides(
            &mut partial,
            vars(&[
                ("OXIBONSAI_RATE_LIMIT_RPM", "600"),
                ("OXIBONSAI_RATE_LIMIT_BURST", "50"),
                ("OXIBONSAI_MAX_BODY_BYTES", "2097152"),
            ]),
        );
        assert_eq!(partial.rate_limit_rpm, Some(600.0));
        assert_eq!(partial.rate_limit_burst, Some(50.0));
        assert_eq!(partial.max_body_bytes, Some(2_097_152));
    }

    #[test]
    fn unrecognized_and_malformed_values_are_ignored() {
        let mut partial = PartialServerConfig::default();
        apply_extra_env_overrides(
            &mut partial,
            vars(&[
                ("OXIBONSAI_UNRELATED", "whatever"),
                ("OXIBONSAI_RATE_LIMIT_RPM", "not-a-number"),
                ("OXIBONSAI_INSECURE_NO_AUTH", "maybe"),
            ]),
        );
        assert!(partial.rate_limit_rpm.is_none());
        assert!(partial.insecure_no_auth.is_none());
    }

    #[test]
    fn empty_admin_token_is_ignored() {
        let mut partial = PartialServerConfig::default();
        apply_extra_env_overrides(&mut partial, vars(&[("OXIBONSAI_ADMIN_TOKEN", "")]));
        assert!(partial.admin_token.is_none());
    }
}

#[cfg(test)]
#[path = "hardening/build_router_tests.rs"]
mod build_router_tests;
