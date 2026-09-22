//! Admin API endpoints for operational management.
//!
//! Provides non-OpenAI routes used by operators to inspect and control a
//! running OxiBonsai server instance.
//!
//! | Method | Path                    | Description                        |
//! |--------|-------------------------|------------------------------------|
//! | GET    | `/admin/status`         | Server status and live metrics     |
//! | GET    | `/admin/config`         | Current configuration snapshot     |
//! | POST   | `/admin/reset-metrics`  | Rebase the admin view of every counter to ~zero (see [`reset_metrics`]) |
//! | GET    | `/admin/cache-stats`    | KV/inference cache statistics      |
//! | GET    | `/admin/workload-stats` | Workload aggregator + KV policy    |
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use oxibonsai_runtime::admin::{AdminState, create_admin_router};
//! use oxibonsai_runtime::metrics::InferenceMetrics;
//!
//! let metrics = Arc::new(InferenceMetrics::new());
//! let state = Arc::new(AdminState::new(metrics));
//! let router = create_admin_router(Arc::clone(&state));
//! ```

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::kv_cache_policy::KvCachePolicy;
use crate::metrics::InferenceMetrics;
use crate::request_metrics::RequestRateAggregator;

// ─── Server status response ─────────────────────────────────────────────────

/// Live server status snapshot.
#[derive(Debug, Serialize)]
pub struct ServerStatus {
    /// Crate version string (from `CARGO_PKG_VERSION`).
    pub version: &'static str,
    /// Seconds elapsed since the server started.
    pub uptime_secs: u64,
    /// Whether the inference model has been loaded.
    pub model_loaded: bool,
    /// Cumulative requests received since last reset.
    pub requests_total: u64,
    /// Cumulative tokens generated since last reset.
    pub tokens_generated: u64,
    /// Number of requests currently in flight.
    pub active_connections: u64,
    /// Process resident-set-size in bytes, if available.
    pub memory_rss_bytes: Option<u64>,
}

// ─── Config snapshot response ────────────────────────────────────────────────

/// Snapshot of key server configuration values.
#[derive(Debug, Serialize)]
pub struct ConfigSnapshot {
    /// Default maximum generation tokens per request.
    pub max_tokens_default: usize,
    /// Default sampling temperature.
    pub temperature_default: f32,
    /// Default nucleus sampling probability threshold.
    pub top_p_default: f32,
    /// Crate version string.
    pub server_version: &'static str,
    /// List of compiled-in feature flags.
    pub features: Vec<String>,
}

// ─── AdminState ──────────────────────────────────────────────────────────────

/// Shared state passed to all admin route handlers.
pub struct AdminState {
    /// Time at which the server was started (used to compute uptime).
    pub started_at: Instant,
    /// Shared metrics instance.
    pub metrics: Arc<InferenceMetrics>,
    /// Optional workload aggregator surfaced via `/admin/workload-stats`.
    pub rate_aggregator: Option<Arc<RequestRateAggregator>>,
    /// Optional KV-cache compression policy surfaced via `/admin/workload-stats`
    /// and `/admin/cache-stats`.
    pub kv_cache_policy: Option<Arc<KvCachePolicy>>,
    /// Optional handle to the served model's descriptor, so `/admin/config` can
    /// report the real loaded model rather than only the sampling defaults.
    pub model_info: Option<Arc<crate::server::ServedModelInfo>>,
    /// Baseline snapshot of `metrics.requests_total` recorded at the last
    /// `/admin/reset-metrics` call (`0` if never reset). See
    /// [`AdminState::requests_total`] (RT-19/SV-18).
    requests_baseline: AtomicU64,
    /// Baseline snapshot of `metrics.tokens_generated_total`.
    tokens_baseline: AtomicU64,
    /// Baseline snapshot of `metrics.errors_total`.
    errors_baseline: AtomicU64,
    /// Baseline snapshot of `metrics.prompt_tokens_total`.
    prompt_tokens_baseline: AtomicU64,
    /// Unix-epoch seconds of the last `/admin/reset-metrics` call, or `0` if
    /// it has never been called (mirrors the `rss_raw == 0 => None`
    /// zero-as-absent convention already used by [`get_status`] below).
    last_reset_unix_secs: AtomicU64,
}

impl AdminState {
    /// Create a new `AdminState` with the given metrics. Workload sources
    /// (rate aggregator and KV-cache policy) start unset; attach them with
    /// [`AdminState::with_rate_aggregator`] and
    /// [`AdminState::with_kv_cache_policy`].
    pub fn new(metrics: Arc<InferenceMetrics>) -> Self {
        Self {
            started_at: Instant::now(),
            metrics,
            rate_aggregator: None,
            kv_cache_policy: None,
            model_info: None,
            requests_baseline: AtomicU64::new(0),
            tokens_baseline: AtomicU64::new(0),
            errors_baseline: AtomicU64::new(0),
            prompt_tokens_baseline: AtomicU64::new(0),
            last_reset_unix_secs: AtomicU64::new(0),
        }
    }

    /// Attach a workload [`RequestRateAggregator`] to surface via
    /// `/admin/workload-stats`. Builder-style consuming setter.
    pub fn with_rate_aggregator(mut self, aggregator: Arc<RequestRateAggregator>) -> Self {
        self.rate_aggregator = Some(aggregator);
        self
    }

    /// Attach a [`KvCachePolicy`] to surface via `/admin/workload-stats`.
    /// Builder-style consuming setter.
    pub fn with_kv_cache_policy(mut self, policy: Arc<KvCachePolicy>) -> Self {
        self.kv_cache_policy = Some(policy);
        self
    }

    /// Attach the served-model descriptor so `/admin/config` reports the real
    /// loaded model. Builder-style consuming setter.
    pub fn with_model_info(mut self, info: Arc<crate::server::ServedModelInfo>) -> Self {
        self.model_info = Some(info);
        self
    }

    /// Return the number of whole seconds the server has been running.
    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Requests received since the server started, or since the last
    /// `/admin/reset-metrics` call if one has happened (RT-19/SV-18).
    ///
    /// This never mutates — or even wraps-and-corrects — the underlying
    /// [`crate::metrics::Counter`]; a Prometheus counter is monotonic *by
    /// contract*, and any consumer scraping `metrics` directly (e.g. a raw
    /// `/metrics` text endpoint, if one is mounted alongside this admin
    /// router) must keep seeing that true, ever-increasing value. Instead a
    /// baseline snapshot is recorded at reset time and subtracted here, so
    /// only this admin view's *reported* number resets to (approximately)
    /// zero — the source counter is untouched and cannot wrap.
    pub fn requests_total(&self) -> u64 {
        self.metrics
            .requests_total
            .get()
            .saturating_sub(self.requests_baseline.load(Ordering::Relaxed))
    }

    /// Tokens generated since the server started or the last reset. See
    /// [`AdminState::requests_total`] for the baseline-subtraction contract.
    pub fn tokens_generated(&self) -> u64 {
        self.metrics
            .tokens_generated_total
            .get()
            .saturating_sub(self.tokens_baseline.load(Ordering::Relaxed))
    }

    /// Errors recorded since the server started or the last reset. See
    /// [`AdminState::requests_total`] for the baseline-subtraction contract.
    pub fn errors_total(&self) -> u64 {
        self.metrics
            .errors_total
            .get()
            .saturating_sub(self.errors_baseline.load(Ordering::Relaxed))
    }

    /// Prompt tokens processed since the server started or the last reset.
    /// See [`AdminState::requests_total`] for the baseline-subtraction
    /// contract.
    pub fn prompt_tokens_total(&self) -> u64 {
        self.metrics
            .prompt_tokens_total
            .get()
            .saturating_sub(self.prompt_tokens_baseline.load(Ordering::Relaxed))
    }

    /// Unix-epoch seconds of the last `/admin/reset-metrics` call, or `None`
    /// if it has never been called.
    pub fn last_reset_unix_secs(&self) -> Option<u64> {
        match self.last_reset_unix_secs.load(Ordering::Relaxed) {
            0 => None,
            secs => Some(secs),
        }
    }

    /// Record a fresh baseline snapshot of every cumulative counter and the
    /// current wall-clock time. This is the entire "reset" — it never calls
    /// into `metrics` with a mutating operation, so the counters it snapshots
    /// keep counting exactly as Prometheus expects (RT-19/SV-18).
    fn record_metrics_reset(&self) {
        self.requests_baseline
            .store(self.metrics.requests_total.get(), Ordering::Relaxed);
        self.tokens_baseline
            .store(self.metrics.tokens_generated_total.get(), Ordering::Relaxed);
        self.errors_baseline
            .store(self.metrics.errors_total.get(), Ordering::Relaxed);
        self.prompt_tokens_baseline
            .store(self.metrics.prompt_tokens_total.get(), Ordering::Relaxed);
        // `.max(1)` keeps the stored value distinct from the `0` "never
        // reset" sentinel even in the (impossible in practice) case of a
        // pre-epoch system clock.
        self.last_reset_unix_secs
            .store(unix_now_secs().max(1), Ordering::Relaxed);
    }
}

/// Seconds since the Unix epoch, saturating to `0` on the (impossible) case
/// of a pre-epoch system clock.
fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Convert Unix seconds (UTC) into an RFC 3339 / ISO 8601 timestamp string
/// with second precision, e.g. `"2026-09-21T12:34:56Z"`.
///
/// Hand-rolled (no additional date/time dependency — `admin.rs` is not the
/// owner of this crate's `Cargo.toml`) using a plain proleptic-Gregorian
/// year/month/day decomposition, the same technique already used by
/// `oxibonsai-eval::report::iso8601_now`. Correct for any Unix timestamp on
/// or after the epoch, which every real wall-clock reading here satisfies.
fn unix_secs_to_iso8601(total_secs: u64) -> String {
    let second = total_secs % 60;
    let minutes = total_secs / 60;
    let minute = minutes % 60;
    let hours = minutes / 60;
    let hour = hours % 24;
    let mut days = hours / 24;

    let is_leap = |year: u64| {
        (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
    };

    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let month_days: [u64; 12] = [
        31,
        if is_leap(year) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for &dim in &month_days {
        if days < dim {
            break;
        }
        days -= dim;
        month += 1;
    }
    let day = days + 1;

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// ─── Route handlers ──────────────────────────────────────────────────────────

/// `GET /admin/status` — return live server status and metrics.
pub async fn get_status(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    let rss = {
        let rss_raw = crate::memory::get_rss_bytes();
        if rss_raw == 0 {
            None
        } else {
            Some(rss_raw)
        }
    };

    // RT-30: report the real loaded-model identity from the model registry
    // rather than a request-count heuristic (a 400 rejection on an
    // unauthenticated port would previously flip `model_loaded` to `true`
    // with zero requests ever reaching the model). `ServedModelInfo`
    // resolves and caches its descriptor from a live engine-pool lease
    // (`get_config` below already relies on the same cached resolution), and
    // its documented not-ready sentinel is `id == "unknown"` — every real
    // config has a non-empty model name. Without an attached
    // `ServedModelInfo` there is nothing in `AdminState` that can answer the
    // question honestly, so `false` is reported instead of guessing from
    // unrelated counters.
    let model_loaded = match &state.model_info {
        Some(info) => info.descriptor().await.id != "unknown",
        None => false,
    };

    let status = ServerStatus {
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: state.uptime_secs(),
        model_loaded,
        requests_total: state.requests_total(),
        tokens_generated: state.tokens_generated(),
        active_connections: state.metrics.active_requests.get() as u64,
        memory_rss_bytes: rss,
    };

    (StatusCode::OK, Json(status))
}

/// `GET /admin/config` — return the server's real running configuration.
///
/// The sampling defaults are read from the runtime chat handler's single source
/// of truth (the same functions that supply the request `serde` defaults), not
/// duplicated literals. When a served-model descriptor is attached, the loaded
/// model's real identity/context length is included under `model`.
pub async fn get_config(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    let snapshot = ConfigSnapshot {
        max_tokens_default: crate::server::default_max_tokens_value(),
        temperature_default: crate::server::default_temperature_value(),
        top_p_default: crate::sampling::SamplingParams::default().top_p,
        server_version: env!("CARGO_PKG_VERSION"),
        features: features_enabled(),
    };

    let mut body = serde_json::to_value(&snapshot).unwrap_or_else(|_| serde_json::json!({}));
    if let (Some(info), serde_json::Value::Object(map)) = (&state.model_info, &mut body) {
        let descriptor = info.descriptor().await;
        map.insert(
            "model".to_string(),
            serde_json::json!({
                "id": descriptor.id,
                "architecture": descriptor.architecture,
                "max_context_length": descriptor.max_context_length,
                "vocab_size": descriptor.vocab_size,
                "created": descriptor.created,
            }),
        );
    }

    (StatusCode::OK, Json(body))
}

/// `POST /admin/reset-metrics` — reset the admin view of every cumulative
/// counter back to (approximately) zero.
///
/// Returns a JSON object: `{"reset": true, "timestamp": "<ISO-8601>"}`.
///
/// # Honesty contract (RT-19 / SV-18)
///
/// A Prometheus [`Counter`](crate::metrics::Counter) is monotonic by
/// contract — real dashboards compute `rate()`/`increase()` over it and
/// silently assume it never goes backward. The previous implementation
/// "reset" a counter by reading its value and then adding
/// `u64::MAX - value + 1` back onto it (relying on wrapping arithmetic to
/// land on zero); a concurrent increment landing between the read and the
/// compensating add made the counter jump to `u64::MAX - k` instead of `0`
/// — silent, permanent, and reported to every future scrape.
///
/// This handler never touches the underlying counters at all. It records a
/// baseline snapshot (see [`AdminState::requests_total`] and friends), and
/// every admin-surfaced count from that point on is `raw − baseline`. Since
/// `raw` only ever increases, the subtraction can never underflow, there is
/// no read-then-write race window, and a `/metrics` scrape mounted
/// elsewhere against the same [`InferenceMetrics`] keeps seeing the true,
/// ever-increasing values Prometheus requires — calling this endpoint
/// invalidates that scrape's in-flight `rate()` window (a real counter
/// reset, even a well-formed one, always does), which is expected, not a
/// defect.
///
/// This also stops resetting the `active_requests` / `kv_cache_utilization`
/// gauges to `0.0`: those track *current* state (in-flight requests, live
/// cache pressure), not a cumulative total, so forcing them to zero while a
/// request is genuinely in flight would itself have been a fabricated
/// reading — the same class of dishonesty this endpoint exists to remove.
pub async fn reset_metrics(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    state.record_metrics_reset();

    let body = serde_json::json!({
        "reset": true,
        "timestamp": unix_secs_to_iso8601(
            state.last_reset_unix_secs().unwrap_or_else(unix_now_secs)
        ),
    });

    (StatusCode::OK, Json(body))
}

/// `GET /admin/workload-stats` — return runtime workload telemetry.
///
/// Combines the [`RequestRateAggregator`]'s sliding-window snapshot
/// (TBT p50/p95, EWMA tokens/sec, queue-wait, completed requests) with the
/// [`KvCachePolicy`] state (current tier, smoothed pressure, transition
/// counters) into one operator-friendly JSON document.
///
/// Either source may be `null` if it wasn't attached to the [`AdminState`].
pub async fn get_workload_stats(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    let request_rate = state.rate_aggregator.as_ref().map(|agg| {
        let snap = agg.snapshot();
        serde_json::json!({
            "completed_requests": snap.completed_requests,
            "mean_tokens_per_second": snap.mean_tokens_per_second,
            "tbt_p50_seconds": snap.tbt_p50_seconds,
            "tbt_p95_seconds": snap.tbt_p95_seconds,
            "mean_queue_wait_seconds": snap.mean_queue_wait_seconds,
        })
    });

    let kv_cache = state.kv_cache_policy.as_ref().map(|policy| {
        let level = policy.current_level();
        serde_json::json!({
            "level": level.tag(),
            "memory_factor": level.memory_factor(),
            "pressure_ewma": policy.pressure(),
            "samples": policy.samples(),
            "upgrades": policy.upgrades(),
            "downgrades": policy.downgrades(),
        })
    });

    let body = serde_json::json!({
        "request_rate": request_rate,
        "kv_cache": kv_cache,
        "status": "ok",
    });
    (StatusCode::OK, Json(body))
}

/// `GET /admin/cache-stats` — report cache statistics for the caches actually
/// wired into this server.
///
/// Only sources that are attached to the [`AdminState`] are reported with real
/// numbers. A cache that is not wired in is reported as `null` (and flagged via
/// an `*_enabled: false` field) rather than as a fabricated all-zero object an
/// operator could mistake for a genuinely empty-but-healthy cache. The default
/// server engine pool runs neither an adaptive KV-cache policy nor a prefix
/// cache, so both are reported as `not enabled` there.
pub async fn get_cache_stats(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    let kv_cache_enabled = state.kv_cache_policy.is_some();
    let kv_cache = state.kv_cache_policy.as_ref().map(|policy| {
        let level = policy.current_level();
        serde_json::json!({
            "level": level.tag(),
            "memory_factor": level.memory_factor(),
            "pressure_ewma": policy.pressure(),
            "samples": policy.samples(),
            "upgrades": policy.upgrades(),
            "downgrades": policy.downgrades(),
        })
    });

    let body = serde_json::json!({
        "kv_cache": kv_cache,
        "kv_cache_enabled": kv_cache_enabled,
        // The base chat server serves from a plain engine pool with no prefix
        // cache attached; report that honestly instead of faking zeros.
        "prefix_cache": serde_json::Value::Null,
        "prefix_cache_enabled": false,
    });

    (StatusCode::OK, Json(body))
}

// ─── Router builder ──────────────────────────────────────────────────────────

/// Build the Axum router for all admin endpoints.
///
/// Mount at a path prefix such as `/admin` in your main router, or use
/// directly on its own in tests.
pub fn create_admin_router(state: Arc<AdminState>) -> Router<Arc<AdminState>> {
    Router::new()
        .route("/admin/status", get(get_status))
        .route("/admin/config", get(get_config))
        .route("/admin/reset-metrics", post(reset_metrics))
        .route("/admin/cache-stats", get(get_cache_stats))
        .route("/admin/workload-stats", get(get_workload_stats))
        .with_state(state)
}

// ─── Feature detection ───────────────────────────────────────────────────────

/// Return the list of Cargo features that were enabled at compile time.
#[allow(clippy::vec_init_then_push)]
pub fn features_enabled() -> Vec<String> {
    let mut features = Vec::new();

    #[cfg(feature = "server")]
    features.push("server".to_owned());

    #[cfg(feature = "rag")]
    features.push("rag".to_owned());

    #[cfg(feature = "wasm")]
    features.push("wasm".to_owned());

    #[cfg(target_arch = "wasm32")]
    features.push("wasm32".to_owned());

    #[cfg(target_arch = "x86_64")]
    features.push("x86_64".to_owned());

    #[cfg(target_arch = "aarch64")]
    features.push("aarch64".to_owned());

    // Always include the runtime itself.
    features.push("runtime".to_owned());

    features
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admin_state_uptime() {
        let metrics = Arc::new(InferenceMetrics::new());
        let state = AdminState::new(metrics);
        // Uptime should be 0 right after construction (well under 1 second).
        let uptime = state.uptime_secs();
        assert!(
            uptime < 5,
            "uptime should be nearly 0 at creation; got {uptime}"
        );
    }

    #[test]
    fn test_admin_state_with_rate_aggregator() {
        let metrics = Arc::new(InferenceMetrics::new());
        let agg = Arc::new(RequestRateAggregator::new());
        let state = AdminState::new(metrics).with_rate_aggregator(Arc::clone(&agg));
        assert!(state.rate_aggregator.is_some());
        assert!(state.kv_cache_policy.is_none());
    }

    #[test]
    fn test_admin_state_with_kv_cache_policy() {
        let metrics = Arc::new(InferenceMetrics::new());
        let policy = Arc::new(KvCachePolicy::default());
        let state = AdminState::new(metrics).with_kv_cache_policy(Arc::clone(&policy));
        assert!(state.kv_cache_policy.is_some());
        assert!(state.rate_aggregator.is_none());
    }

    #[tokio::test]
    async fn test_get_workload_stats_empty() {
        let metrics = Arc::new(InferenceMetrics::new());
        let state = Arc::new(AdminState::new(metrics));
        // Without aggregator or policy, both fields should serialize as null.
        let response = get_workload_stats(State(Arc::clone(&state))).await;
        let response = response.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_workload_stats_with_sources() {
        let metrics = Arc::new(InferenceMetrics::new());
        let agg = Arc::new(RequestRateAggregator::new());
        let policy = Arc::new(KvCachePolicy::default());
        let state = Arc::new(
            AdminState::new(metrics)
                .with_rate_aggregator(Arc::clone(&agg))
                .with_kv_cache_policy(Arc::clone(&policy)),
        );
        let response = get_workload_stats(State(Arc::clone(&state))).await;
        let response = response.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_features_enabled_non_empty() {
        let features = features_enabled();
        assert!(!features.is_empty(), "features list should not be empty");
        assert!(
            features.contains(&"runtime".to_owned()),
            "should always include 'runtime'"
        );
    }

    #[test]
    fn test_server_version_non_empty() {
        let version: &'static str = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty(), "CARGO_PKG_VERSION should not be empty");
    }

    // ── RT-19 / SV-18: honest metric reset ─────────────────────────────────

    #[test]
    fn reset_metrics_never_mutates_the_underlying_counter() {
        let metrics = Arc::new(InferenceMetrics::new());
        metrics.requests_total.inc_by(7);
        let state = AdminState::new(Arc::clone(&metrics));

        state.record_metrics_reset();

        // The raw, shared counter must be untouched — any other consumer of
        // `metrics` (e.g. a `/metrics` Prometheus scrape) still sees the true
        // monotonic value.
        assert_eq!(
            metrics.requests_total.get(),
            7,
            "the shared Counter must never be mutated by a reset"
        );
        // But this admin view now reports (approximately) zero.
        assert_eq!(state.requests_total(), 0);
    }

    #[test]
    fn requests_total_reports_delta_since_last_reset() {
        let metrics = Arc::new(InferenceMetrics::new());
        let state = AdminState::new(Arc::clone(&metrics));

        metrics.requests_total.inc_by(3);
        assert_eq!(state.requests_total(), 3);

        state.record_metrics_reset();
        assert_eq!(state.requests_total(), 0);

        metrics.requests_total.inc_by(5);
        assert_eq!(
            state.requests_total(),
            5,
            "post-reset requests must count from the new baseline, not from zero absolute"
        );
    }

    #[test]
    fn reset_cannot_underflow_even_if_baseline_races_ahead() {
        // Pathological but reachable: a baseline recorded from a read that
        // raced past a concurrent increment must never make the subsequent
        // `saturating_sub` panic or wrap — it must clamp to 0.
        let metrics = Arc::new(InferenceMetrics::new());
        let state = AdminState::new(Arc::clone(&metrics));
        state
            .requests_baseline
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(state.requests_total(), 0, "must saturate, never underflow");
    }

    #[test]
    fn last_reset_unix_secs_starts_absent_and_becomes_present() {
        let metrics = Arc::new(InferenceMetrics::new());
        let state = AdminState::new(metrics);
        assert!(state.last_reset_unix_secs().is_none());
        state.record_metrics_reset();
        assert!(state.last_reset_unix_secs().is_some());
    }

    #[tokio::test]
    async fn reset_metrics_handler_never_produces_a_near_u64_max_counter() {
        let metrics = Arc::new(InferenceMetrics::new());
        metrics.requests_total.inc_by(42);
        metrics.tokens_generated_total.inc_by(1000);
        let state = Arc::new(AdminState::new(Arc::clone(&metrics)));

        let response = reset_metrics(State(Arc::clone(&state)))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("should read body");
        let json: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("body should be valid JSON");
        assert_eq!(json["reset"], serde_json::json!(true));
        let ts = json["timestamp"]
            .as_str()
            .expect("timestamp should be a string");
        // ISO 8601 / RFC 3339, e.g. "2026-09-21T12:34:56Z" — never the racy
        // wrap-around's ~u64::MAX, and never a bare unix-epoch integer.
        assert!(
            ts.ends_with('Z') && ts.contains('T') && ts.len() == 20,
            "timestamp should look like an ISO-8601 UTC instant, got {ts:?}"
        );

        // The raw counters are exactly what they were — never anywhere near
        // u64::MAX, which is the signature of the old wrap-around bug.
        assert_eq!(metrics.requests_total.get(), 42);
        assert_eq!(metrics.tokens_generated_total.get(), 1000);
    }

    #[test]
    fn concurrent_increments_during_reset_never_wrap_the_counter() {
        use std::thread;

        let metrics = Arc::new(InferenceMetrics::new());
        let state = Arc::new(AdminState::new(Arc::clone(&metrics)));

        let writer_metrics = Arc::clone(&metrics);
        let writer = thread::spawn(move || {
            for _ in 0..10_000 {
                writer_metrics.requests_total.inc();
            }
        });
        for _ in 0..50 {
            state.record_metrics_reset();
        }
        writer.join().expect("writer thread panicked");

        let raw = metrics.requests_total.get();
        assert!(
            raw <= 10_000,
            "the true counter must never exceed the number of real increments \
             issued; got {raw} (a value near u64::MAX would indicate the old \
             wrap-around race)"
        );
        // The admin view must also never underflow/wrap regardless of
        // exactly when the last reset raced the last increment.
        assert!(state.requests_total() <= 10_000);
    }

    // ── RT-30: real model_loaded ────────────────────────────────────────────

    #[tokio::test]
    async fn model_loaded_is_honestly_false_without_a_served_model_info() {
        let metrics = Arc::new(InferenceMetrics::new());
        let state = Arc::new(AdminState::new(metrics));
        assert!(
            state.model_info.is_none(),
            "precondition: no ServedModelInfo attached"
        );

        let response = get_status(State(Arc::clone(&state))).await.into_response();
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("should read body");
        let json: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("body should be valid JSON");
        assert_eq!(
            json["model_loaded"],
            serde_json::json!(false),
            "without a real model-registry handle, model_loaded must not be \
             guessed from unrelated request/token counters (the old \
             placeholder heuristic)"
        );
    }

    #[tokio::test]
    async fn model_loaded_placeholder_heuristic_is_gone() {
        // Regression guard for the exact old bug: a request being *handled*
        // (even a rejected one, which still increments requests_total on the
        // validation-failure path elsewhere in the server) must not flip
        // model_loaded to true on its own.
        let metrics = Arc::new(InferenceMetrics::new());
        metrics.requests_total.inc_by(5);
        metrics.tokens_generated_total.inc_by(100);
        let state = Arc::new(AdminState::new(metrics));

        let response = get_status(State(Arc::clone(&state))).await.into_response();
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("should read body");
        let json: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("body should be valid JSON");
        assert_eq!(
            json["model_loaded"],
            serde_json::json!(false),
            "nonzero request/token counters alone must never imply a model is loaded"
        );
    }

    // ── unix_secs_to_iso8601 ─────────────────────────────────────────────────

    #[test]
    fn iso8601_epoch_zero() {
        assert_eq!(unix_secs_to_iso8601(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn iso8601_known_reference_instants() {
        // Cross-checked against `date -u -r <secs>`.
        assert_eq!(unix_secs_to_iso8601(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(unix_secs_to_iso8601(1_789_948_800), "2026-09-21T00:00:00Z");
    }

    #[test]
    fn iso8601_handles_leap_day() {
        // 2000-02-29 is a real leap day (divisible by 400).
        assert_eq!(unix_secs_to_iso8601(951_782_400), "2000-02-29T00:00:00Z");
    }
}
