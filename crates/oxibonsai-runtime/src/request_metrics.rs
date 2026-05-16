//! Per-request token-rate and latency metrics.
//!
//! [`RequestRateTracker`] records per-token timing for a single in-flight
//! request and produces:
//! - EMA-smoothed tokens per second
//! - p50 / p95 inter-token latency (TBT — time-between-tokens)
//! - queue-wait time (admission → first token)
//!
//! These per-request rollups are aggregated by [`RequestRateAggregator`]
//! across a window of recent requests, exposing global p50/p95 inter-token
//! latency and a smoothed average tokens-per-second.
//!
//! ## Design
//!
//! - **TBT samples** are stored in a small ring buffer (default 128 slots).
//!   Quantiles are computed by sorting on demand: O(n log n) but n is
//!   bounded so the cost is constant.
//! - **EMA tokens/sec** uses an exponentially-weighted moving average with
//!   alpha = 0.20 (20% new sample, 80% prior smoothed value).
//! - **Aggregator** keeps the last `window` request rollups in a fixed-size
//!   ring; the workload-level p50/p95 is computed from the union of the
//!   most recent rollups.
//!
//! ## Usage
//!
//! ```
//! use oxibonsai_runtime::request_metrics::RequestRateTracker;
//!
//! let mut t = RequestRateTracker::new();
//! t.record_admission();              // request received at queue
//! std::thread::sleep(std::time::Duration::from_micros(100));
//! t.record_first_token();            // first token emitted
//! for _ in 0..10 {
//!     std::thread::sleep(std::time::Duration::from_micros(50));
//!     t.record_token();              // subsequent tokens
//! }
//! let snap = t.snapshot();
//! assert!(snap.tokens_emitted >= 11);
//! assert!(snap.tokens_per_second > 0.0);
//! ```

use std::sync::Mutex;
use std::time::Instant;

const DEFAULT_WINDOW_TBT_SAMPLES: usize = 128;
const DEFAULT_AGGREGATOR_WINDOW: usize = 256;
const DEFAULT_TPS_ALPHA: f64 = 0.20;

// ─── Per-request tracker ───────────────────────────────────────────────────

/// Snapshot of a single request's rate metrics at one point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestRateSnapshot {
    /// Number of tokens emitted so far.
    pub tokens_emitted: u64,
    /// EMA-smoothed tokens per second.
    pub tokens_per_second: f64,
    /// Median (p50) inter-token latency in seconds.
    pub tbt_p50_seconds: f64,
    /// 95th-percentile inter-token latency in seconds.
    pub tbt_p95_seconds: f64,
    /// Queue wait time (admission → first token) in seconds, or `None`
    /// if no token has been emitted yet.
    pub queue_wait_seconds: Option<f64>,
    /// Total time elapsed since admission, in seconds.
    pub elapsed_seconds: f64,
}

/// Per-request rate tracker.
///
/// Not thread-safe — intended to be owned by a single request handler.
/// Callers needing concurrent access should wrap in a `Mutex`.
#[derive(Debug, Clone)]
pub struct RequestRateTracker {
    admission: Option<Instant>,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
    tokens_emitted: u64,
    tps_ema: f64,
    tps_alpha: f64,
    /// Ring buffer of recent inter-token deltas in seconds.
    tbt_samples: Vec<f64>,
    tbt_capacity: usize,
    tbt_next_idx: usize,
    tbt_filled: usize,
}

impl RequestRateTracker {
    /// Create a tracker with default settings (window = 128 TBT samples,
    /// alpha = 0.20).
    pub fn new() -> Self {
        Self::with_params(DEFAULT_WINDOW_TBT_SAMPLES, DEFAULT_TPS_ALPHA)
    }

    /// Create a tracker with custom parameters.
    ///
    /// `tbt_capacity` is clamped to at least 1; `alpha` is clamped to
    /// `[0.0, 1.0]`.
    pub fn with_params(tbt_capacity: usize, alpha: f64) -> Self {
        let cap = tbt_capacity.max(1);
        Self {
            admission: None,
            first_token: None,
            last_token: None,
            tokens_emitted: 0,
            tps_ema: 0.0,
            tps_alpha: alpha.clamp(0.0, 1.0),
            tbt_samples: vec![0.0; cap],
            tbt_capacity: cap,
            tbt_next_idx: 0,
            tbt_filled: 0,
        }
    }

    /// Mark the request as admitted (e.g. dequeued from the request queue).
    pub fn record_admission(&mut self) {
        self.admission = Some(Instant::now());
    }

    /// Mark the first token as emitted. This implicitly counts the token,
    /// so callers should not also call `record_token` for the first token.
    pub fn record_first_token(&mut self) {
        let now = Instant::now();
        self.first_token = Some(now);
        self.last_token = Some(now);
        self.tokens_emitted = self.tokens_emitted.saturating_add(1);
    }

    /// Mark a subsequent (non-first) token as emitted.
    ///
    /// If `record_first_token` was never called, this also doubles as
    /// the first-token marker.
    pub fn record_token(&mut self) {
        let now = Instant::now();
        if self.first_token.is_none() {
            self.first_token = Some(now);
        }
        if let Some(prev) = self.last_token {
            let delta = (now - prev).as_secs_f64();
            self.push_tbt_sample(delta);
            // Update tokens/sec EMA from instantaneous rate.
            if delta > 0.0 {
                let inst = 1.0 / delta;
                if self.tokens_emitted < 2 {
                    // Seed the EMA with the first observation.
                    self.tps_ema = inst;
                } else {
                    self.tps_ema = self.tps_alpha * inst + (1.0 - self.tps_alpha) * self.tps_ema;
                }
            }
        }
        self.last_token = Some(now);
        self.tokens_emitted = self.tokens_emitted.saturating_add(1);
    }

    /// Take a snapshot of the current state without disturbing the tracker.
    pub fn snapshot(&self) -> RequestRateSnapshot {
        let now = Instant::now();
        let elapsed = self
            .admission
            .map(|t| (now - t).as_secs_f64())
            .unwrap_or(0.0);
        let queue_wait = self.queue_wait_seconds();
        let (p50, p95) = self.tbt_quantiles();

        RequestRateSnapshot {
            tokens_emitted: self.tokens_emitted,
            tokens_per_second: self.tps_ema,
            tbt_p50_seconds: p50,
            tbt_p95_seconds: p95,
            queue_wait_seconds: queue_wait,
            elapsed_seconds: elapsed,
        }
    }

    /// Queue wait time (admission → first token), if both are recorded.
    pub fn queue_wait_seconds(&self) -> Option<f64> {
        match (self.admission, self.first_token) {
            (Some(a), Some(f)) => Some((f - a).as_secs_f64()),
            _ => None,
        }
    }

    /// Number of tokens emitted so far.
    pub fn tokens_emitted(&self) -> u64 {
        self.tokens_emitted
    }

    /// EMA-smoothed tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        self.tps_ema
    }

    /// Compute (p50, p95) inter-token latency in seconds from the current
    /// ring-buffer contents.
    fn tbt_quantiles(&self) -> (f64, f64) {
        if self.tbt_filled == 0 {
            return (0.0, 0.0);
        }
        let n = self.tbt_filled.min(self.tbt_capacity);
        let mut buf: Vec<f64> = self.tbt_samples[..n].to_vec();
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = quantile_sorted(&buf, 0.50);
        let p95 = quantile_sorted(&buf, 0.95);
        (p50, p95)
    }

    fn push_tbt_sample(&mut self, delta: f64) {
        self.tbt_samples[self.tbt_next_idx] = delta;
        self.tbt_next_idx = (self.tbt_next_idx + 1) % self.tbt_capacity;
        if self.tbt_filled < self.tbt_capacity {
            self.tbt_filled += 1;
        }
    }
}

impl Default for RequestRateTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Aggregator ────────────────────────────────────────────────────────────

/// Workload-level rollup of recent request rate snapshots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AggregateRateSnapshot {
    /// Number of completed requests in the window.
    pub completed_requests: u64,
    /// Mean tokens-per-second across the window.
    pub mean_tokens_per_second: f64,
    /// p50 inter-token latency across the window.
    pub tbt_p50_seconds: f64,
    /// p95 inter-token latency across the window.
    pub tbt_p95_seconds: f64,
    /// Mean queue wait time across the window.
    pub mean_queue_wait_seconds: f64,
}

/// Aggregator over the most recent `window` per-request snapshots.
///
/// Thread-safe via an internal mutex.
#[derive(Debug)]
pub struct RequestRateAggregator {
    inner: Mutex<RingState>,
}

#[derive(Debug)]
struct RingState {
    samples: Vec<RequestRateSnapshot>,
    capacity: usize,
    next_idx: usize,
    filled: usize,
    completed: u64,
}

impl RequestRateAggregator {
    /// Create an aggregator with the default window of 256 requests.
    pub fn new() -> Self {
        Self::with_window(DEFAULT_AGGREGATOR_WINDOW)
    }

    /// Create an aggregator with a custom window size (clamped to >= 1).
    pub fn with_window(window: usize) -> Self {
        let cap = window.max(1);
        Self {
            inner: Mutex::new(RingState {
                samples: Vec::with_capacity(cap),
                capacity: cap,
                next_idx: 0,
                filled: 0,
                completed: 0,
            }),
        }
    }

    /// Record a completed request's snapshot.
    pub fn record(&self, snap: RequestRateSnapshot) {
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if g.samples.len() < g.capacity {
            g.samples.push(snap);
        } else {
            let idx = g.next_idx;
            g.samples[idx] = snap;
        }
        g.next_idx = (g.next_idx + 1) % g.capacity;
        if g.filled < g.capacity {
            g.filled += 1;
        }
        g.completed = g.completed.saturating_add(1);
    }

    /// Compute the workload-level snapshot from the current window.
    pub fn snapshot(&self) -> AggregateRateSnapshot {
        let g = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let n = g.filled;
        if n == 0 {
            return AggregateRateSnapshot {
                completed_requests: 0,
                mean_tokens_per_second: 0.0,
                tbt_p50_seconds: 0.0,
                tbt_p95_seconds: 0.0,
                mean_queue_wait_seconds: 0.0,
            };
        }

        let mut tps_sum = 0.0;
        let mut wait_sum = 0.0;
        let mut wait_n = 0;
        let mut tbt_p50: Vec<f64> = Vec::with_capacity(n);
        let mut tbt_p95: Vec<f64> = Vec::with_capacity(n);

        for s in &g.samples[..n] {
            tps_sum += s.tokens_per_second;
            if let Some(w) = s.queue_wait_seconds {
                wait_sum += w;
                wait_n += 1;
            }
            tbt_p50.push(s.tbt_p50_seconds);
            tbt_p95.push(s.tbt_p95_seconds);
        }

        tbt_p50.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        tbt_p95.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_tps = tps_sum / n as f64;
        let mean_wait = if wait_n == 0 {
            0.0
        } else {
            wait_sum / wait_n as f64
        };

        // Window-level p50/p95: take the median and 95th-percentile of
        // per-request medians/95s — a reasonable proxy for global
        // latency without needing all per-token samples.
        let p50_window = quantile_sorted(&tbt_p50, 0.50);
        let p95_window = quantile_sorted(&tbt_p95, 0.95);

        AggregateRateSnapshot {
            completed_requests: g.completed,
            mean_tokens_per_second: mean_tps,
            tbt_p50_seconds: p50_window,
            tbt_p95_seconds: p95_window,
            mean_queue_wait_seconds: mean_wait,
        }
    }

    /// Number of requests recorded since construction (not capped by window).
    pub fn completed(&self) -> u64 {
        match self.inner.lock() {
            Ok(g) => g.completed,
            Err(poisoned) => poisoned.into_inner().completed,
        }
    }

    /// Drop all recorded samples. Counters are not reset.
    pub fn clear(&self) {
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        g.samples.clear();
        g.next_idx = 0;
        g.filled = 0;
    }
}

impl Default for RequestRateAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Quantile helper ───────────────────────────────────────────────────────

/// Compute the `q` quantile of a *sorted* slice in `[0.0, 1.0]`.
///
/// Returns 0.0 for an empty slice. Uses linear interpolation between
/// neighbouring samples.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn fresh_tracker_has_zero_tokens() {
        let t = RequestRateTracker::new();
        let s = t.snapshot();
        assert_eq!(s.tokens_emitted, 0);
        assert!(s.tokens_per_second.abs() < f64::EPSILON);
        assert!(s.queue_wait_seconds.is_none());
    }

    #[test]
    fn first_token_records_count() {
        let mut t = RequestRateTracker::new();
        t.record_admission();
        t.record_first_token();
        assert_eq!(t.tokens_emitted(), 1);
        assert!(t.queue_wait_seconds().is_some());
    }

    #[test]
    fn queue_wait_measured() {
        let mut t = RequestRateTracker::new();
        t.record_admission();
        sleep(ms(2));
        t.record_first_token();
        let wait = t.queue_wait_seconds().expect("wait recorded");
        assert!(wait >= 0.001, "queue wait should be >= 1ms, got {wait}");
    }

    #[test]
    fn token_rate_increases_with_decoding() {
        let mut t = RequestRateTracker::new();
        t.record_admission();
        t.record_first_token();
        for _ in 0..5 {
            sleep(ms(2));
            t.record_token();
        }
        let s = t.snapshot();
        assert_eq!(s.tokens_emitted, 6);
        assert!(s.tokens_per_second > 0.0);
        assert!(s.tbt_p50_seconds > 0.0);
        assert!(s.tbt_p95_seconds >= s.tbt_p50_seconds);
    }

    #[test]
    fn tbt_quantiles_match_expectations() {
        let mut t = RequestRateTracker::with_params(64, 0.20);
        t.record_admission();
        t.record_first_token();
        // 20 fast (~1ms) tokens followed by a tail of 5 slow (~10ms) tokens.
        // With only 25 total samples, p95 (~position 22.8) lands inside the
        // slow tail — exercising the quantile interpolation path.
        for _ in 0..20 {
            sleep(ms(1));
            t.record_token();
        }
        for _ in 0..5 {
            sleep(ms(10));
            t.record_token();
        }
        let s = t.snapshot();
        assert!(s.tbt_p95_seconds >= s.tbt_p50_seconds);
        // The slow tail dominates p95 (timing is OS-scheduler dependent so we
        // use a relaxed threshold).
        assert!(
            s.tbt_p95_seconds >= 0.003,
            "p95 should reflect slow tail; got {}",
            s.tbt_p95_seconds
        );
    }

    #[test]
    fn tbt_ring_buffer_overwrites_oldest() {
        let mut t = RequestRateTracker::with_params(4, 0.20);
        t.record_admission();
        t.record_first_token();
        for _ in 0..10 {
            sleep(ms(1));
            t.record_token();
        }
        let s = t.snapshot();
        // Capacity 4; most recent 4 deltas are kept.
        assert!(s.tbt_p50_seconds > 0.0);
    }

    #[test]
    fn quantile_sorted_basic() {
        assert!((quantile_sorted(&[], 0.5) - 0.0).abs() < f64::EPSILON);
        assert!((quantile_sorted(&[5.0], 0.5) - 5.0).abs() < f64::EPSILON);
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile_sorted(&v, 0.0) - 1.0).abs() < f64::EPSILON);
        assert!((quantile_sorted(&v, 1.0) - 5.0).abs() < f64::EPSILON);
        assert!((quantile_sorted(&v, 0.5) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregator_records_and_aggregates() {
        let agg = RequestRateAggregator::with_window(8);
        for i in 0..5 {
            let snap = RequestRateSnapshot {
                tokens_emitted: 100,
                tokens_per_second: 50.0 + i as f64,
                tbt_p50_seconds: 0.020,
                tbt_p95_seconds: 0.050,
                queue_wait_seconds: Some(0.010),
                elapsed_seconds: 2.0,
            };
            agg.record(snap);
        }
        let agg_snap = agg.snapshot();
        assert_eq!(agg_snap.completed_requests, 5);
        assert!(agg_snap.mean_tokens_per_second >= 50.0);
        assert!(agg_snap.tbt_p50_seconds > 0.0);
        assert!(agg_snap.tbt_p95_seconds >= agg_snap.tbt_p50_seconds);
        assert!(agg_snap.mean_queue_wait_seconds > 0.0);
    }

    #[test]
    fn aggregator_handles_empty() {
        let agg = RequestRateAggregator::new();
        let s = agg.snapshot();
        assert_eq!(s.completed_requests, 0);
        assert!(s.mean_tokens_per_second.abs() < f64::EPSILON);
        assert!(s.tbt_p50_seconds.abs() < f64::EPSILON);
        assert!(s.tbt_p95_seconds.abs() < f64::EPSILON);
    }

    #[test]
    fn aggregator_window_overwrites() {
        let agg = RequestRateAggregator::with_window(4);
        for i in 0..10 {
            let snap = RequestRateSnapshot {
                tokens_emitted: 1,
                tokens_per_second: i as f64,
                tbt_p50_seconds: 0.01,
                tbt_p95_seconds: 0.02,
                queue_wait_seconds: None,
                elapsed_seconds: 0.0,
            };
            agg.record(snap);
        }
        let s = agg.snapshot();
        assert_eq!(s.completed_requests, 10);
        // Only last 4 (6,7,8,9) contribute to mean
        assert!((s.mean_tokens_per_second - 7.5).abs() < 1e-6);
    }

    #[test]
    fn aggregator_clear() {
        let agg = RequestRateAggregator::new();
        agg.record(RequestRateSnapshot {
            tokens_emitted: 1,
            tokens_per_second: 100.0,
            tbt_p50_seconds: 0.01,
            tbt_p95_seconds: 0.02,
            queue_wait_seconds: None,
            elapsed_seconds: 0.0,
        });
        assert_eq!(agg.completed(), 1);
        agg.clear();
        let s = agg.snapshot();
        assert_eq!(s.mean_tokens_per_second, 0.0);
        // `completed` is a lifetime counter, not affected by clear.
        assert_eq!(agg.completed(), 1);
    }

    #[test]
    fn record_token_without_first_token_works() {
        let mut t = RequestRateTracker::new();
        t.record_admission();
        // Skip explicit first_token; record_token should adopt the role.
        t.record_token();
        sleep(ms(2));
        t.record_token();
        assert_eq!(t.tokens_emitted(), 2);
        assert!(t.queue_wait_seconds().is_some());
    }

    #[test]
    fn aggregator_is_thread_safe() {
        use std::sync::Arc;
        use std::thread;

        let agg = Arc::new(RequestRateAggregator::with_window(64));
        let mut handles = Vec::new();
        for tid in 0..4 {
            let agg = Arc::clone(&agg);
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    agg.record(RequestRateSnapshot {
                        tokens_emitted: 1,
                        tokens_per_second: (tid * 100 + i) as f64,
                        tbt_p50_seconds: 0.01,
                        tbt_p95_seconds: 0.02,
                        queue_wait_seconds: None,
                        elapsed_seconds: 0.0,
                    });
                }
            }));
        }
        for h in handles {
            h.join().expect("worker panicked");
        }
        assert_eq!(agg.completed(), 4 * 50);
    }
}
