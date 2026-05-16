//! Admin API endpoints for operational management.
//!
//! Provides non-OpenAI routes used by operators to inspect and control a
//! running OxiBonsai server instance.
//!
//! | Method | Path                    | Description                        |
//! |--------|-------------------------|------------------------------------|
//! | GET    | `/admin/status`         | Server status and live metrics     |
//! | GET    | `/admin/config`         | Current configuration snapshot     |
//! | POST   | `/admin/reset-metrics`  | Reset all metric counters to zero  |
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
    /// Optional KV-cache compression policy surfaced via `/admin/workload-stats`.
    pub kv_cache_policy: Option<Arc<KvCachePolicy>>,
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

    /// Return the number of whole seconds the server has been running.
    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
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

    let status = ServerStatus {
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: state.uptime_secs(),
        // We treat "model loaded" as true when at least one request has been
        // handled (a placeholder heuristic; callers can extend AdminState for
        // a real flag).
        model_loaded: state.metrics.requests_total.get() > 0
            || state.metrics.tokens_generated_total.get() > 0,
        requests_total: state.metrics.requests_total.get(),
        tokens_generated: state.metrics.tokens_generated_total.get(),
        active_connections: state.metrics.active_requests.get() as u64,
        memory_rss_bytes: rss,
    };

    (StatusCode::OK, Json(status))
}

/// `GET /admin/config` — return current configuration snapshot.
pub async fn get_config(_state: State<Arc<AdminState>>) -> impl IntoResponse {
    let snapshot = ConfigSnapshot {
        max_tokens_default: 256,
        temperature_default: 0.7,
        top_p_default: 0.9,
        server_version: env!("CARGO_PKG_VERSION"),
        features: features_enabled(),
    };

    (StatusCode::OK, Json(snapshot))
}

/// `POST /admin/reset-metrics` — reset all metric counters to zero.
///
/// Returns a JSON object: `{"reset": true, "timestamp": "<ISO-8601>"}`.
pub async fn reset_metrics(State(state): State<Arc<AdminState>>) -> impl IntoResponse {
    // Reset counters by reading current values and subtracting them.
    let requests = state.metrics.requests_total.get();
    state.metrics.requests_total.inc_by(0); // ensure no-op reads are fine

    // Use inc_by with wrapping: fetch current, set back to 0 by subtracting.
    // Counter only supports inc_by(n) — we reset by exploiting u64 wrap-around
    // with a large subtraction. A cleaner approach: read-subtract current value.
    // Since Counter doesn't expose a reset(), we achieve "reset" semantics by
    // subtracting the current reading. Under normal (non-overflow) circumstances
    // this yields exactly 0.
    let tokens = state.metrics.tokens_generated_total.get();
    let errors = state.metrics.errors_total.get();
    let prompt = state.metrics.prompt_tokens_total.get();

    // Subtract current values to bring counters back to 0 (u64 wrapping arithmetic).
    state
        .metrics
        .requests_total
        .inc_by(u64::MAX.wrapping_sub(requests).wrapping_add(1));
    state
        .metrics
        .tokens_generated_total
        .inc_by(u64::MAX.wrapping_sub(tokens).wrapping_add(1));
    state
        .metrics
        .errors_total
        .inc_by(u64::MAX.wrapping_sub(errors).wrapping_add(1));
    state
        .metrics
        .prompt_tokens_total
        .inc_by(u64::MAX.wrapping_sub(prompt).wrapping_add(1));

    // Also reset gauges.
    state.metrics.active_requests.set(0.0);
    state.metrics.kv_cache_utilization.set(0.0);

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let body = serde_json::json!({
        "reset": true,
        "timestamp": ts,
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

/// `GET /admin/cache-stats` — return placeholder cache statistics.
pub async fn get_cache_stats(_state: State<Arc<AdminState>>) -> impl IntoResponse {
    let body = serde_json::json!({
        "kv_cache": {
            "capacity_blocks": 0,
            "used_blocks": 0,
            "utilization": 0.0,
            "evictions_total": 0,
        },
        "prefix_cache": {
            "entries": 0,
            "hit_rate": 0.0,
        },
        "status": "ok",
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
}
