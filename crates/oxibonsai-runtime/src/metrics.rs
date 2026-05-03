//! Lightweight, thread-safe Prometheus-compatible metrics system.
//!
//! Self-contained with zero external dependencies beyond `std`.
//! Provides counters, gauges, and histograms with Prometheus text
//! exposition format rendering.
//!
//! ## Metric Types
//!
//! | Type | Behaviour | Use case |
//! |------|-----------|----------|
//! | [`Counter`] | Monotonically increasing u64 | Request counts, token counts |
//! | [`Gauge`] | Arbitrary f64, inc/dec/set | Active connections, cache utilisation |
//! | [`Histogram`] | Cumulative bucket counts + sum | Latency distributions |
//!
//! ## Usage
//!
//! [`InferenceMetrics`] bundles all OxiBonsai-specific counters, gauges,
//! and histograms into a single struct that can be shared via
//! `Arc<InferenceMetrics>` across the engine and HTTP handlers.

use std::fmt::Write as FmtWrite;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ─── Counter ────────────────────────────────────────────────────────

/// Thread-safe monotonically increasing counter.
pub struct Counter {
    value: AtomicU64,
    name: &'static str,
    help: &'static str,
}

impl Counter {
    /// Create a new counter with the given name and help text.
    pub fn new(name: &'static str, help: &'static str) -> Self {
        Self {
            value: AtomicU64::new(0),
            name,
            help,
        }
    }

    /// Increment by 1.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment by `n`.
    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Name of this counter.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Help text for this counter.
    pub fn help(&self) -> &'static str {
        self.help
    }
}

// ─── Gauge ──────────────────────────────────────────────────────────

/// Thread-safe gauge that can go up and down.
///
/// Stores f64 bits as u64 for atomic operations.
pub struct Gauge {
    value: AtomicU64,
    name: &'static str,
    help: &'static str,
}

impl Gauge {
    /// Create a new gauge with the given name and help text.
    pub fn new(name: &'static str, help: &'static str) -> Self {
        Self {
            value: AtomicU64::new(f64::to_bits(0.0)),
            name,
            help,
        }
    }

    /// Set the gauge to an absolute value.
    pub fn set(&self, val: f64) {
        self.value.store(f64::to_bits(val), Ordering::Relaxed);
    }

    /// Increment the gauge by 1.0.
    pub fn inc(&self) {
        self.add(1.0);
    }

    /// Decrement the gauge by 1.0.
    pub fn dec(&self) {
        self.add(-1.0);
    }

    /// Add a delta to the gauge (can be negative).
    fn add(&self, delta: f64) {
        loop {
            let current_bits = self.value.load(Ordering::Relaxed);
            let current = f64::from_bits(current_bits);
            let new_val = current + delta;
            let new_bits = f64::to_bits(new_val);
            if self
                .value
                .compare_exchange_weak(current_bits, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Get the current value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    /// Name of this gauge.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Help text for this gauge.
    pub fn help(&self) -> &'static str {
        self.help
    }
}

// ─── Histogram ──────────────────────────────────────────────────────

/// Thread-safe histogram with configurable buckets.
///
/// Each observation is placed into the appropriate bucket(s) and
/// contributes to the running sum and count.
pub struct Histogram {
    buckets: Vec<f64>,
    /// One counter per bucket upper-bound, plus one for +Inf.
    counts: Vec<AtomicU64>,
    /// Running sum of observed values (f64 bits stored as u64).
    sum: AtomicU64,
    /// Total number of observations.
    count: AtomicU64,
    name: &'static str,
    help: &'static str,
}

impl Histogram {
    /// Create a new histogram with the given bucket boundaries.
    ///
    /// Buckets are sorted automatically. A `+Inf` bucket is always
    /// appended internally.
    pub fn new(name: &'static str, help: &'static str, mut buckets: Vec<f64>) -> Self {
        buckets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        buckets.dedup();

        // +1 for the implicit +Inf bucket
        let counts: Vec<AtomicU64> = (0..=buckets.len()).map(|_| AtomicU64::new(0)).collect();

        Self {
            buckets,
            counts,
            sum: AtomicU64::new(f64::to_bits(0.0)),
            count: AtomicU64::new(0),
            name,
            help,
        }
    }

    /// Record an observation.
    pub fn observe(&self, value: f64) {
        // Increment all bucket counters where value <= boundary (cumulative)
        for (i, &boundary) in self.buckets.iter().enumerate() {
            if value <= boundary {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        // Always increment the +Inf bucket
        if let Some(inf_bucket) = self.counts.last() {
            inf_bucket.fetch_add(1, Ordering::Relaxed);
        }

        // Add to sum (CAS loop for f64 atomics)
        loop {
            let current_bits = self.sum.load(Ordering::Relaxed);
            let current = f64::from_bits(current_bits);
            let new_val = current + value;
            let new_bits = f64::to_bits(new_val);
            if self
                .sum
                .compare_exchange_weak(current_bits, new_bits, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Time a closure and record its duration in seconds.
    pub fn time<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_secs_f64();
        self.observe(elapsed);
        result
    }

    /// Name of this histogram.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Help text for this histogram.
    pub fn help(&self) -> &'static str {
        self.help
    }

    /// Get the total observation count.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the running sum.
    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    /// Get cumulative count for a specific bucket index.
    pub fn bucket_count(&self, index: usize) -> u64 {
        self.counts
            .get(index)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get the bucket boundaries (excluding +Inf).
    pub fn bucket_boundaries(&self) -> &[f64] {
        &self.buckets
    }
}

// ─── Default bucket helpers ─────────────────────────────────────────

/// Default latency buckets in seconds.
///
/// 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
pub fn default_latency_buckets() -> Vec<f64> {
    vec![
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ]
}

/// Default token rate buckets (tokens per second).
///
/// 1, 5, 10, 20, 50, 100, 200 tok/s
pub fn default_rate_buckets() -> Vec<f64> {
    vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
}

// ─── InferenceMetrics ───────────────────────────────────────────────

/// All OxiBonsai inference metrics collected in one place.
///
/// Thread-safe — can be shared via `Arc<InferenceMetrics>` across
/// handlers and the inference engine.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::metrics::InferenceMetrics;
///
/// let metrics = InferenceMetrics::new();
/// metrics.requests_total.inc_by(5);
/// metrics.tokens_generated_total.inc_by(100);
/// metrics.active_requests.set(2.0);
/// metrics.request_duration_seconds.observe(0.42);
///
/// assert_eq!(metrics.requests_total.get(), 5);
/// assert_eq!(metrics.tokens_generated_total.get(), 100);
///
/// let prom = metrics.render_prometheus();
/// assert!(prom.contains("oxibonsai_requests_total 5"));
/// ```
pub struct InferenceMetrics {
    // ── Counters ──
    /// Total number of tokens generated across all requests.
    pub tokens_generated_total: Counter,
    /// Total number of requests received.
    pub requests_total: Counter,
    /// Total number of errors.
    pub errors_total: Counter,
    /// Total number of prompt tokens processed.
    pub prompt_tokens_total: Counter,

    // ── Histograms ──
    /// Duration of the prefill (prompt processing) phase.
    pub prefill_duration_seconds: Histogram,
    /// Duration of each individual decode step.
    pub decode_token_duration_seconds: Histogram,
    /// Total end-to-end request duration.
    pub request_duration_seconds: Histogram,
    /// Observed tokens-per-second rate.
    pub tokens_per_second: Histogram,

    // ── Gauges ──
    /// Number of currently active (in-flight) requests.
    pub active_requests: Gauge,
    /// KV cache utilization ratio (0.0 – 1.0).
    pub kv_cache_utilization: Gauge,
    /// Total model memory usage in bytes.
    pub model_memory_bytes: Gauge,
    /// Smoothed average tokens-per-second across recent requests.
    pub request_tokens_per_second: Gauge,
    /// p50 inter-token latency across recent requests, in seconds.
    pub inter_token_latency_p50_seconds: Gauge,
    /// p95 inter-token latency across recent requests, in seconds.
    pub inter_token_latency_p95_seconds: Gauge,
    /// Mean queue-wait time (admission → first token) in seconds.
    pub queue_wait_seconds: Gauge,
    /// Effective tier of the runtime KV-cache compression policy:
    /// 0 = FP16, 1 = Q8, 2 = Q4.
    pub kv_cache_compression_level: Gauge,
}

impl InferenceMetrics {
    /// Update the `model_memory_bytes` gauge from the current process RSS.
    ///
    /// This is a best-effort call: on platforms where RSS is unavailable it
    /// records `0.0`, which is still a valid (though unhelpful) gauge value.
    pub fn update_memory_from_rss(&self) {
        let rss = crate::memory::get_rss_bytes();
        self.model_memory_bytes.set(rss as f64);
    }

    /// Create a new set of inference metrics with default buckets.
    pub fn new() -> Self {
        Self {
            tokens_generated_total: Counter::new(
                "oxibonsai_tokens_generated_total",
                "Total tokens generated",
            ),
            requests_total: Counter::new("oxibonsai_requests_total", "Total inference requests"),
            errors_total: Counter::new("oxibonsai_errors_total", "Total inference errors"),
            prompt_tokens_total: Counter::new(
                "oxibonsai_prompt_tokens_total",
                "Total prompt tokens processed",
            ),

            prefill_duration_seconds: Histogram::new(
                "oxibonsai_prefill_duration_seconds",
                "Prefill (prompt processing) duration in seconds",
                default_latency_buckets(),
            ),
            decode_token_duration_seconds: Histogram::new(
                "oxibonsai_decode_token_duration_seconds",
                "Per-token decode step duration in seconds",
                default_latency_buckets(),
            ),
            request_duration_seconds: Histogram::new(
                "oxibonsai_request_duration_seconds",
                "End-to-end request duration in seconds",
                default_latency_buckets(),
            ),
            tokens_per_second: Histogram::new(
                "oxibonsai_tokens_per_second",
                "Observed tokens per second rate",
                default_rate_buckets(),
            ),

            active_requests: Gauge::new(
                "oxibonsai_active_requests",
                "Number of currently active requests",
            ),
            kv_cache_utilization: Gauge::new(
                "oxibonsai_kv_cache_utilization",
                "KV cache utilization ratio (0.0 to 1.0)",
            ),
            model_memory_bytes: Gauge::new(
                "oxibonsai_model_memory_bytes",
                "Model memory usage in bytes",
            ),
            request_tokens_per_second: Gauge::new(
                "oxibonsai_request_tokens_per_second",
                "EWMA tokens-per-second across recent requests",
            ),
            inter_token_latency_p50_seconds: Gauge::new(
                "oxibonsai_inter_token_latency_p50_seconds",
                "Median inter-token latency across recent requests (seconds)",
            ),
            inter_token_latency_p95_seconds: Gauge::new(
                "oxibonsai_inter_token_latency_p95_seconds",
                "p95 inter-token latency across recent requests (seconds)",
            ),
            queue_wait_seconds: Gauge::new(
                "oxibonsai_queue_wait_seconds",
                "Mean queue-wait (admission to first token) across recent requests (seconds)",
            ),
            kv_cache_compression_level: Gauge::new(
                "oxibonsai_kv_cache_compression_level",
                "KV cache compression tier: 0=FP16, 1=Q8, 2=Q4",
            ),
        }
    }

    /// Update the per-request rate gauges from a [`crate::request_metrics::AggregateRateSnapshot`].
    pub fn update_request_rate(&self, snap: &crate::request_metrics::AggregateRateSnapshot) {
        self.request_tokens_per_second
            .set(snap.mean_tokens_per_second);
        self.inter_token_latency_p50_seconds
            .set(snap.tbt_p50_seconds);
        self.inter_token_latency_p95_seconds
            .set(snap.tbt_p95_seconds);
        self.queue_wait_seconds.set(snap.mean_queue_wait_seconds);
    }

    /// Update the KV-cache compression-level gauge from a [`crate::kv_cache_policy::KvCacheLevel`].
    pub fn update_kv_cache_level(&self, level: crate::kv_cache_policy::KvCacheLevel) {
        self.kv_cache_compression_level.set(level.ordinal() as f64);
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Counters
        render_counter(&mut out, &self.tokens_generated_total);
        render_counter(&mut out, &self.requests_total);
        render_counter(&mut out, &self.errors_total);
        render_counter(&mut out, &self.prompt_tokens_total);

        // Histograms
        render_histogram(&mut out, &self.prefill_duration_seconds);
        render_histogram(&mut out, &self.decode_token_duration_seconds);
        render_histogram(&mut out, &self.request_duration_seconds);
        render_histogram(&mut out, &self.tokens_per_second);

        // Gauges
        render_gauge(&mut out, &self.active_requests);
        render_gauge(&mut out, &self.kv_cache_utilization);
        render_gauge(&mut out, &self.model_memory_bytes);
        render_gauge(&mut out, &self.request_tokens_per_second);
        render_gauge(&mut out, &self.inter_token_latency_p50_seconds);
        render_gauge(&mut out, &self.inter_token_latency_p95_seconds);
        render_gauge(&mut out, &self.queue_wait_seconds);
        render_gauge(&mut out, &self.kv_cache_compression_level);

        out
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Prometheus rendering helpers ───────────────────────────────────

fn render_counter(out: &mut String, counter: &Counter) {
    let _ = writeln!(out, "# HELP {} {}", counter.name(), counter.help());
    let _ = writeln!(out, "# TYPE {} counter", counter.name());
    let _ = writeln!(out, "{} {}", counter.name(), counter.get());
    let _ = writeln!(out);
}

fn render_gauge(out: &mut String, gauge: &Gauge) {
    let _ = writeln!(out, "# HELP {} {}", gauge.name(), gauge.help());
    let _ = writeln!(out, "# TYPE {} gauge", gauge.name());
    let value = gauge.get();
    // Render integers without decimal point for cleanliness
    if value.fract() == 0.0 && value.is_finite() {
        let _ = writeln!(out, "{} {}", gauge.name(), value as i64);
    } else {
        let _ = writeln!(out, "{} {value}", gauge.name());
    }
    let _ = writeln!(out);
}

fn render_histogram(out: &mut String, hist: &Histogram) {
    let _ = writeln!(out, "# HELP {} {}", hist.name(), hist.help());
    let _ = writeln!(out, "# TYPE {} histogram", hist.name());

    for (i, &boundary) in hist.bucket_boundaries().iter().enumerate() {
        let count = hist.bucket_count(i);
        // Format bucket boundary: strip trailing zeros but keep at least one decimal
        let le = format_f64_prometheus(boundary);
        let _ = writeln!(out, "{}_bucket{{le=\"{le}\"}} {count}", hist.name());
    }

    // +Inf bucket
    let inf_count = hist.bucket_count(hist.bucket_boundaries().len());
    let _ = writeln!(out, "{}_bucket{{le=\"+Inf\"}} {inf_count}", hist.name());

    let sum = hist.sum();
    let _ = writeln!(out, "{}_sum {}", hist.name(), format_f64_prometheus(sum));
    let _ = writeln!(out, "{}_count {}", hist.name(), hist.count());
    let _ = writeln!(out);
}

/// Format f64 for Prometheus output.
///
/// Integers are rendered without unnecessary decimals; floats use
/// enough precision to be accurate.
fn format_f64_prometheus(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        // Use enough precision, trim trailing zeros
        let s = format!("{value:.6}");
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_basic() {
        let c = Counter::new("test_counter", "A test counter");
        assert_eq!(c.get(), 0);
        c.inc();
        assert_eq!(c.get(), 1);
        c.inc_by(5);
        assert_eq!(c.get(), 6);
        c.inc_by(0);
        assert_eq!(c.get(), 6);
    }

    #[test]
    fn counter_concurrent() {
        use std::sync::Arc;
        let c = Arc::new(Counter::new("concurrent_counter", "concurrent test"));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let c = Arc::clone(&c);
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    c.inc();
                }
            }));
        }
        for h in handles {
            h.join().expect("thread should not panic");
        }
        assert_eq!(c.get(), 10_000);
    }

    #[test]
    fn gauge_set_and_get() {
        let g = Gauge::new("test_gauge", "A test gauge");
        assert!((g.get() - 0.0).abs() < f64::EPSILON);
        g.set(42.5);
        assert!((g.get() - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_inc_dec() {
        let g = Gauge::new("test_gauge_incdec", "inc dec test");
        g.inc();
        assert!((g.get() - 1.0).abs() < f64::EPSILON);
        g.inc();
        assert!((g.get() - 2.0).abs() < f64::EPSILON);
        g.dec();
        assert!((g.get() - 1.0).abs() < f64::EPSILON);
        g.dec();
        assert!(g.get().abs() < f64::EPSILON);
    }

    #[test]
    fn gauge_concurrent() {
        use std::sync::Arc;
        let g = Arc::new(Gauge::new("concurrent_gauge", "concurrent gauge"));
        let mut handles = Vec::new();
        // 5 threads inc, 5 threads dec — should net to 0
        for i in 0..10 {
            let g = Arc::clone(&g);
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    if i < 5 {
                        g.inc();
                    } else {
                        g.dec();
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("thread should not panic");
        }
        assert!(g.get().abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_observe() {
        let h = Histogram::new("test_hist", "A test histogram", vec![1.0, 5.0, 10.0]);
        h.observe(0.5);
        h.observe(3.0);
        h.observe(7.0);
        h.observe(15.0);

        // Cumulative counts:
        // le=1.0: 1 (0.5)
        // le=5.0: 2 (0.5, 3.0)
        // le=10.0: 3 (0.5, 3.0, 7.0)
        // le=+Inf: 4 (all)
        assert_eq!(h.bucket_count(0), 1);
        assert_eq!(h.bucket_count(1), 2);
        assert_eq!(h.bucket_count(2), 3);
        assert_eq!(h.bucket_count(3), 4); // +Inf

        assert_eq!(h.count(), 4);
        let expected_sum = 0.5 + 3.0 + 7.0 + 15.0;
        assert!((h.sum() - expected_sum).abs() < 1e-9);
    }

    #[test]
    fn histogram_empty() {
        let h = Histogram::new("empty_hist", "empty", vec![1.0, 5.0]);
        assert_eq!(h.count(), 0);
        assert!(h.sum().abs() < f64::EPSILON);
        assert_eq!(h.bucket_count(0), 0);
        assert_eq!(h.bucket_count(1), 0);
        assert_eq!(h.bucket_count(2), 0); // +Inf
    }

    #[test]
    fn histogram_time_closure() {
        let h = Histogram::new("timed_hist", "timed", vec![0.001, 0.01, 0.1, 1.0]);
        let result = h.time(|| {
            // Quick operation
            42
        });
        assert_eq!(result, 42);
        assert_eq!(h.count(), 1);
        // Duration should be very small (< 1s)
        assert!(h.sum() < 1.0);
    }

    #[test]
    fn histogram_boundary_values() {
        let h = Histogram::new("boundary_hist", "boundary", vec![1.0, 5.0, 10.0]);
        // Observe exactly on a boundary
        h.observe(5.0);
        // le=1.0: 0
        // le=5.0: 1 (5.0 <= 5.0)
        // le=10.0: 1
        // le=+Inf: 1
        assert_eq!(h.bucket_count(0), 0);
        assert_eq!(h.bucket_count(1), 1);
        assert_eq!(h.bucket_count(2), 1);
        assert_eq!(h.bucket_count(3), 1);
    }

    #[test]
    fn default_buckets_sorted() {
        let latency = default_latency_buckets();
        for pair in latency.windows(2) {
            assert!(pair[0] < pair[1], "latency buckets must be sorted");
        }

        let rate = default_rate_buckets();
        for pair in rate.windows(2) {
            assert!(pair[0] < pair[1], "rate buckets must be sorted");
        }
    }

    #[test]
    fn inference_metrics_default() {
        let m = InferenceMetrics::default();
        assert_eq!(m.tokens_generated_total.get(), 0);
        assert_eq!(m.requests_total.get(), 0);
        assert_eq!(m.errors_total.get(), 0);
        assert!(m.active_requests.get().abs() < f64::EPSILON);
    }

    #[test]
    fn render_prometheus_counter_format() {
        let m = InferenceMetrics::new();
        m.requests_total.inc_by(42);
        let output = m.render_prometheus();

        assert!(output.contains("# HELP oxibonsai_requests_total Total inference requests"));
        assert!(output.contains("# TYPE oxibonsai_requests_total counter"));
        assert!(output.contains("oxibonsai_requests_total 42"));
    }

    #[test]
    fn render_prometheus_gauge_format() {
        let m = InferenceMetrics::new();
        m.active_requests.set(3.0);
        let output = m.render_prometheus();

        assert!(output.contains("# HELP oxibonsai_active_requests"));
        assert!(output.contains("# TYPE oxibonsai_active_requests gauge"));
        assert!(output.contains("oxibonsai_active_requests 3"));
    }

    #[test]
    fn render_prometheus_histogram_format() {
        let m = InferenceMetrics::new();
        m.request_duration_seconds.observe(0.002);
        m.request_duration_seconds.observe(0.05);
        let output = m.render_prometheus();

        assert!(output.contains("# HELP oxibonsai_request_duration_seconds"));
        assert!(output.contains("# TYPE oxibonsai_request_duration_seconds histogram"));
        assert!(output.contains("oxibonsai_request_duration_seconds_bucket{le=\"0.001\"} 0"));
        assert!(output.contains("oxibonsai_request_duration_seconds_bucket{le=\"+Inf\"} 2"));
        assert!(output.contains("oxibonsai_request_duration_seconds_count 2"));
    }

    #[test]
    fn render_prometheus_full_output_parseable() {
        let m = InferenceMetrics::new();
        m.tokens_generated_total.inc_by(100);
        m.requests_total.inc_by(5);
        m.errors_total.inc();
        m.prompt_tokens_total.inc_by(50);
        m.active_requests.set(2.0);
        m.kv_cache_utilization.set(0.75);
        m.model_memory_bytes.set(1_073_741_824.0);
        m.request_duration_seconds.observe(0.1);
        m.prefill_duration_seconds.observe(0.01);
        m.decode_token_duration_seconds.observe(0.001);
        m.tokens_per_second.observe(42.0);

        let output = m.render_prometheus();

        // Every HELP line should have a matching TYPE line
        let help_count = output.lines().filter(|l| l.starts_with("# HELP")).count();
        let type_count = output.lines().filter(|l| l.starts_with("# TYPE")).count();
        assert_eq!(help_count, type_count);

        // 4 counters + 4 histograms + 8 gauges = 16 metric families.
        // Gauges added in 0.1.4: request_tokens_per_second,
        // inter_token_latency_p50/p95_seconds, queue_wait_seconds,
        // kv_cache_compression_level (5 new) plus the original 3.
        assert_eq!(help_count, 16);
    }

    #[test]
    fn update_request_rate_writes_gauges() {
        use crate::request_metrics::AggregateRateSnapshot;
        let m = InferenceMetrics::new();
        let snap = AggregateRateSnapshot {
            completed_requests: 10,
            mean_tokens_per_second: 42.5,
            tbt_p50_seconds: 0.020,
            tbt_p95_seconds: 0.080,
            mean_queue_wait_seconds: 0.005,
        };
        m.update_request_rate(&snap);
        assert!((m.request_tokens_per_second.get() - 42.5).abs() < 1e-6);
        assert!((m.inter_token_latency_p50_seconds.get() - 0.020).abs() < 1e-6);
        assert!((m.inter_token_latency_p95_seconds.get() - 0.080).abs() < 1e-6);
        assert!((m.queue_wait_seconds.get() - 0.005).abs() < 1e-6);
    }

    #[test]
    fn update_kv_cache_level_writes_gauge() {
        use crate::kv_cache_policy::KvCacheLevel;
        let m = InferenceMetrics::new();
        m.update_kv_cache_level(KvCacheLevel::Fp16);
        assert!(m.kv_cache_compression_level.get().abs() < 1e-6);
        m.update_kv_cache_level(KvCacheLevel::Q8);
        assert!((m.kv_cache_compression_level.get() - 1.0).abs() < 1e-6);
        m.update_kv_cache_level(KvCacheLevel::Q4);
        assert!((m.kv_cache_compression_level.get() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn render_prometheus_includes_new_gauges() {
        let m = InferenceMetrics::new();
        let output = m.render_prometheus();
        assert!(output.contains("oxibonsai_request_tokens_per_second"));
        assert!(output.contains("oxibonsai_inter_token_latency_p50_seconds"));
        assert!(output.contains("oxibonsai_inter_token_latency_p95_seconds"));
        assert!(output.contains("oxibonsai_queue_wait_seconds"));
        assert!(output.contains("oxibonsai_kv_cache_compression_level"));
    }

    #[test]
    fn format_f64_prometheus_integers() {
        assert_eq!(format_f64_prometheus(0.0), "0");
        assert_eq!(format_f64_prometheus(42.0), "42");
        assert_eq!(format_f64_prometheus(1000.0), "1000");
    }

    #[test]
    fn format_f64_prometheus_fractions() {
        assert_eq!(format_f64_prometheus(0.001), "0.001");
        assert_eq!(format_f64_prometheus(0.5), "0.5");
        assert_eq!(format_f64_prometheus(2.5), "2.5");
    }

    #[test]
    fn histogram_deduplicates_and_sorts_buckets() {
        let h = Histogram::new("dedup", "test", vec![5.0, 1.0, 5.0, 3.0, 1.0]);
        assert_eq!(h.bucket_boundaries(), &[1.0, 3.0, 5.0]);
    }
}
