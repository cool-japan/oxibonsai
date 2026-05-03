//! Microbenchmarks for the runtime controllers added in 0.1.4.
//!
//! These verify that the per-call overhead of the controllers is negligible
//! relative to a token's compute cost (typically tens of microseconds for
//! the smallest models, milliseconds for production sizes). Targets are:
//!
//! - `KvCachePolicy::observe`: < 100 ns
//! - `AdaptiveLookahead::observe_step`: < 100 ns
//! - `RequestRateTracker::record_token`: < 1 µs (depends on Instant cost)
//! - `RequestRateAggregator::record`: < 200 ns
//! - `RequestRateAggregator::snapshot` (window=256): < 50 µs
//!
//! Run with:
//! ```text
//! cargo bench --bench controllers_bench
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use oxibonsai_runtime::adaptive_lookahead::{AdaptiveLookahead, AdaptiveLookaheadConfig};
use oxibonsai_runtime::kv_cache_policy::{KvCachePolicy, KvCachePolicyConfig};
use oxibonsai_runtime::request_id::RequestId;
use oxibonsai_runtime::request_metrics::{
    RequestRateAggregator, RequestRateSnapshot, RequestRateTracker,
};

// ─── KvCachePolicy ──────────────────────────────────────────────────────────

fn bench_kv_cache_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("KvCachePolicy");

    group.bench_function("observe_steady", |b| {
        let p = KvCachePolicy::default();
        // Steady-state pressure observation — no level transitions.
        b.iter(|| {
            let lvl = p.observe(black_box(0.50));
            black_box(lvl);
        });
    });

    group.bench_function("observe_oscillating", |b| {
        let p = KvCachePolicy::default();
        // Pressure swinging across the q8/q4 thresholds — exercises the
        // hysteresis branches.
        let mut tick = 0u64;
        b.iter(|| {
            let v = if tick % 2 == 0 { 0.99 } else { 0.40 };
            tick = tick.wrapping_add(1);
            let lvl = p.observe(black_box(v));
            black_box(lvl);
        });
    });

    group.bench_function("aggressive_profile", |b| {
        let p =
            KvCachePolicy::new(KvCachePolicyConfig::aggressive()).expect("aggressive cfg valid");
        b.iter(|| {
            let lvl = p.observe(black_box(0.85));
            black_box(lvl);
        });
    });

    group.finish();
}

// ─── AdaptiveLookahead ──────────────────────────────────────────────────────

fn bench_adaptive_lookahead(c: &mut Criterion) {
    let mut group = c.benchmark_group("AdaptiveLookahead");

    group.bench_function("observe_step_warm", |b| {
        let mut adj = AdaptiveLookahead::new(AdaptiveLookaheadConfig::default());
        // Pre-warm so we measure steady-state cost.
        for _ in 0..16 {
            adj.observe_step(5, 3);
        }
        b.iter(|| {
            adj.observe_step(black_box(5), black_box(3));
        });
    });

    group.bench_function("observe_step_with_update", |b| {
        let cfg = AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 0, // every observation may trigger an update
        };
        let mut adj = AdaptiveLookahead::new(cfg);
        let mut t = 0u64;
        b.iter(|| {
            // Alternate between high and low acceptance to force updates.
            let acc = if t % 2 == 0 { 5 } else { 0 };
            t = t.wrapping_add(1);
            adj.observe_step(black_box(5), black_box(acc));
        });
    });

    group.finish();
}

// ─── RequestRateTracker ─────────────────────────────────────────────────────

fn bench_request_rate_tracker(c: &mut Criterion) {
    let mut group = c.benchmark_group("RequestRateTracker");

    group.bench_function("record_token_warm", |b| {
        let mut t = RequestRateTracker::new();
        t.record_admission();
        t.record_first_token();
        b.iter(|| {
            t.record_token();
        });
    });

    group.bench_function("snapshot_full_window", |b| {
        let mut t = RequestRateTracker::with_params(128, 0.20);
        t.record_admission();
        t.record_first_token();
        for _ in 0..128 {
            t.record_token();
        }
        b.iter(|| {
            let s = t.snapshot();
            black_box(s);
        });
    });

    group.finish();
}

// ─── RequestRateAggregator ──────────────────────────────────────────────────

fn bench_request_rate_aggregator(c: &mut Criterion) {
    let mut group = c.benchmark_group("RequestRateAggregator");

    let snap = RequestRateSnapshot {
        tokens_emitted: 100,
        tokens_per_second: 42.0,
        tbt_p50_seconds: 0.02,
        tbt_p95_seconds: 0.08,
        queue_wait_seconds: Some(0.01),
        elapsed_seconds: 2.5,
    };

    group.bench_function("record", |b| {
        let agg = RequestRateAggregator::with_window(256);
        b.iter(|| {
            agg.record(black_box(snap));
        });
    });

    group.bench_function("snapshot_full_window_256", |b| {
        let agg = RequestRateAggregator::with_window(256);
        for _ in 0..256 {
            agg.record(snap);
        }
        b.iter(|| {
            let s = agg.snapshot();
            black_box(s);
        });
    });

    group.finish();
}

// ─── RequestId ──────────────────────────────────────────────────────────────

fn bench_request_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("RequestId");

    group.bench_function("new", |b| {
        b.iter(|| {
            let id = RequestId::new();
            black_box(id);
        });
    });

    group.bench_function("as_uuid", |b| {
        let id = RequestId::new();
        b.iter(|| {
            let s = id.as_uuid();
            black_box(s);
        });
    });

    group.bench_function("from_uuid", |b| {
        let id = RequestId::new();
        let s = id.as_uuid();
        b.iter(|| {
            let parsed = RequestId::from_uuid(black_box(&s));
            black_box(parsed);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_kv_cache_policy,
    bench_adaptive_lookahead,
    bench_request_rate_tracker,
    bench_request_rate_aggregator,
    bench_request_id
);
criterion_main!(benches);
