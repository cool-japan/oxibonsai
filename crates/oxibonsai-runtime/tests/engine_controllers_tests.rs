//! Integration tests for the engine ⇄ runtime-controller plumbing added in
//! 0.1.4 (`generate_tracked`, `generate_with_request_id`,
//! `set_rate_aggregator`).
//!
//! These tests exercise the public API of [`InferenceEngine`] without going
//! through GGUF: they use [`Qwen3Config::tiny_test`] to spin up a minimal
//! engine and exercise the tracking + aggregation paths end-to-end.

use std::sync::Arc;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::request_id::RequestId;
use oxibonsai_runtime::request_metrics::{RequestRateAggregator, RequestRateTracker};
use oxibonsai_runtime::sampling::SamplingParams;

fn engine() -> InferenceEngine<'static> {
    let config = Qwen3Config::tiny_test();
    let params = SamplingParams::default();
    InferenceEngine::new(config, params, 42)
}

#[test]
fn generate_tracked_records_admission_and_first_token() {
    let mut engine = engine();
    let mut tracker = RequestRateTracker::new();
    let prompt = vec![1u32, 2, 3];
    let _ = engine
        .generate_tracked(&prompt, 4, &mut tracker)
        .expect("tracked generate ok");
    let snap = tracker.snapshot();
    // Whatever number of tokens were emitted, queue-wait should be set
    // (admission and first-token both fired) when at least one token came out.
    if snap.tokens_emitted > 0 {
        assert!(
            snap.queue_wait_seconds.is_some(),
            "queue_wait should be set after at least one emitted token"
        );
        assert!(snap.elapsed_seconds >= 0.0);
    }
}

#[test]
fn generate_tracked_pushes_to_aggregator() {
    let mut engine = engine();
    let agg = Arc::new(RequestRateAggregator::with_window(8));
    engine.set_rate_aggregator(Arc::clone(&agg));
    assert_eq!(agg.completed(), 0);

    let prompt = vec![1u32, 2, 3];
    let mut tracker = RequestRateTracker::new();
    let _ = engine
        .generate_tracked(&prompt, 3, &mut tracker)
        .expect("tracked generate ok");
    assert_eq!(agg.completed(), 1, "aggregator should record the request");

    // A second tracked call increments the counter again.
    let mut tracker2 = RequestRateTracker::new();
    let _ = engine
        .generate_tracked(&prompt, 3, &mut tracker2)
        .expect("tracked generate ok 2");
    assert_eq!(agg.completed(), 2);
}

#[test]
fn generate_tracked_does_not_push_without_aggregator() {
    let mut engine = engine();
    let prompt = vec![1u32, 2, 3];
    let mut tracker = RequestRateTracker::new();
    // No aggregator attached — should not panic.
    let _ = engine
        .generate_tracked(&prompt, 3, &mut tracker)
        .expect("tracked generate ok");
    assert!(engine.rate_aggregator().is_none());
}

#[test]
fn generate_with_request_id_returns_tracker() {
    let mut engine = engine();
    let id = RequestId::new();
    let prompt = vec![1u32, 2, 3];
    let (_tokens, tracker) = engine
        .generate_with_request_id(id, &prompt, 3)
        .expect("generate_with_request_id ok");
    let snap = tracker.snapshot();
    assert!(snap.elapsed_seconds >= 0.0);
}

#[test]
fn generate_with_request_id_pushes_to_aggregator() {
    let mut engine = engine();
    let agg = Arc::new(RequestRateAggregator::with_window(4));
    engine.set_rate_aggregator(Arc::clone(&agg));

    for _ in 0..3 {
        let id = RequestId::new();
        let _ = engine
            .generate_with_request_id(id, &[1u32, 2, 3], 2)
            .expect("ok");
    }
    assert_eq!(agg.completed(), 3);
}

#[test]
fn empty_prompt_does_not_record_admission() {
    let mut engine = engine();
    let agg = Arc::new(RequestRateAggregator::with_window(4));
    engine.set_rate_aggregator(Arc::clone(&agg));

    let mut tracker = RequestRateTracker::new();
    let out = engine
        .generate_tracked(&[], 5, &mut tracker)
        .expect("empty ok");
    assert!(out.is_empty());
    // Empty prompts return early before the tracker is touched.
    assert_eq!(tracker.tokens_emitted(), 0);
    // And no snapshot should be pushed to the aggregator.
    assert_eq!(agg.completed(), 0);
}

#[test]
fn unique_request_ids_per_call() {
    let mut engine = engine();
    let prompt = vec![1u32, 2, 3];
    let id_a = RequestId::new();
    let id_b = RequestId::new();
    assert_ne!(id_a, id_b);
    let _ = engine
        .generate_with_request_id(id_a, &prompt, 1)
        .expect("ok");
    let _ = engine
        .generate_with_request_id(id_b, &prompt, 1)
        .expect("ok");
}

#[test]
fn aggregator_snapshot_after_many_requests() {
    let mut engine = engine();
    let agg = Arc::new(RequestRateAggregator::with_window(16));
    engine.set_rate_aggregator(Arc::clone(&agg));

    for _ in 0..5 {
        let _ = engine
            .generate_with_request_id(RequestId::new(), &[1u32, 2, 3], 2)
            .expect("ok");
    }
    let snap = agg.snapshot();
    assert_eq!(snap.completed_requests, 5);
    // p50/p95 may be 0 if no inter-token deltas were captured (very fast
    // tiny model), but the gauges must be non-negative.
    assert!(snap.tbt_p50_seconds >= 0.0);
    assert!(snap.tbt_p95_seconds >= 0.0);
    assert!(snap.mean_tokens_per_second >= 0.0);
}
