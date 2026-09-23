//! Concurrency tests for async engine, batch engine, and request queue.

use std::sync::Arc;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::async_engine::AsyncInferenceEngine;
use oxibonsai_runtime::batch_engine::{
    batch_generate, batch_generate_with_timeout, BatchConfig, BatchRequest, BatchResult,
    FinishReason, RequestQueue,
};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;

// ── Helper ───────────────────────────────────────────────────────────────

fn make_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

// ── 1. AsyncInferenceEngine creation and capacity ────────────────────────

#[test]
fn async_engine_creation_and_capacity() {
    let engine = make_engine();
    let async_engine = AsyncInferenceEngine::new(engine, 4);
    assert_eq!(async_engine.max_concurrent(), 4);
    assert_eq!(async_engine.active_requests(), 0);
    assert!(async_engine.has_capacity());
}

#[test]
fn async_engine_min_concurrency() {
    let engine = make_engine();
    let async_engine = AsyncInferenceEngine::new(engine, 0);
    assert_eq!(
        async_engine.max_concurrent(),
        1,
        "min concurrency should be 1"
    );
}

#[test]
fn async_engine_with_metrics() {
    let engine = make_engine();
    let metrics = Arc::new(InferenceMetrics::new());
    let async_engine = AsyncInferenceEngine::new(engine, 2).with_metrics(Arc::clone(&metrics));
    assert_eq!(async_engine.max_concurrent(), 2);
    assert!(async_engine.has_capacity());
}

// ── 2. Batch generation with multiple prompts ────────────────────────────

#[test]
fn batch_generate_four_prompts() {
    let mut engine = make_engine();
    let prompts = vec![vec![1u32], vec![2, 3], vec![4, 5, 6], vec![7, 8, 9, 10]];
    let results = batch_generate(&mut engine, &prompts, 3);
    assert_eq!(results.len(), 4, "should return 4 results for 4 prompts");
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_ok(), "prompt {i} should succeed");
        let br = r.as_ref().expect("should be ok");
        assert_eq!(br.prompt_tokens, prompts[i].len());
    }
}

#[test]
fn batch_generate_empty_prompts_in_batch() {
    let mut engine = make_engine();
    let prompts = vec![vec![], vec![], vec![]];
    let results = batch_generate(&mut engine, &prompts, 5);
    assert_eq!(results.len(), 3);
    for r in &results {
        let br = r.as_ref().expect("empty prompt should succeed");
        assert_eq!(br.prompt_tokens, 0);
        assert!(br.generated_tokens.is_empty());
        assert_eq!(br.finish_reason, FinishReason::Eos);
    }
}

// ── 3. RequestQueue push/drain cycle ─────────────────────────────────────

#[test]
fn request_queue_push_drain_cycle() {
    let mut queue = RequestQueue::new(10);
    assert!(queue.is_empty());
    assert!(!queue.is_full());

    // Push 5 requests
    for i in 0..5 {
        let req = BatchRequest {
            prompt_tokens: vec![i as u32],
            max_tokens: 10,
            params: SamplingParams::default(),
        };
        queue.push(req).expect("push should succeed");
    }
    assert_eq!(queue.len(), 5);
    assert!(!queue.is_full());

    // Drain 3
    let batch = queue.drain_batch(3);
    assert_eq!(batch.len(), 3);
    assert_eq!(queue.len(), 2);

    // Verify FIFO order
    assert_eq!(batch[0].prompt_tokens, vec![0]);
    assert_eq!(batch[1].prompt_tokens, vec![1]);
    assert_eq!(batch[2].prompt_tokens, vec![2]);

    // Drain remaining
    let rest = queue.drain_batch(100);
    assert_eq!(rest.len(), 2);
    assert!(queue.is_empty());
}

// ── 4. RequestQueue full rejection ───────────────────────────────────────

#[test]
fn request_queue_full_rejects() {
    let mut queue = RequestQueue::new(2);

    let make_req = |id: u32| BatchRequest {
        prompt_tokens: vec![id],
        max_tokens: 5,
        params: SamplingParams::default(),
    };

    queue.push(make_req(1)).expect("first push should succeed");
    queue.push(make_req(2)).expect("second push should succeed");
    assert!(queue.is_full());

    let result = queue.push(make_req(3));
    assert!(result.is_err(), "push to full queue should fail");
    assert_eq!(
        queue.len(),
        2,
        "queue length should not change on rejection"
    );
}

// ── 5. BatchConfig default values ────────────────────────────────────────

#[test]
fn batch_config_defaults() {
    let config = BatchConfig::default();
    assert_eq!(config.max_batch_size, 8);
    assert_eq!(config.max_tokens_per_request, 512);
    assert_eq!(config.timeout_per_request_ms, Some(30_000));
}

// ── 6. FinishReason variants ─────────────────────────────────────────────

#[test]
fn finish_reason_display_all_variants() {
    assert_eq!(format!("{}", FinishReason::MaxTokens), "max_tokens");
    assert_eq!(format!("{}", FinishReason::Eos), "eos");
    assert_eq!(format!("{}", FinishReason::Error), "error");
    assert_eq!(format!("{}", FinishReason::Timeout), "timeout");
}

#[test]
fn finish_reason_eq() {
    assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
    assert_eq!(FinishReason::Eos, FinishReason::Eos);
    assert_ne!(FinishReason::MaxTokens, FinishReason::Eos);
    assert_ne!(FinishReason::Error, FinishReason::Timeout);
}

// ── 7. BatchResult fields populated correctly ────────────────────────────

#[test]
fn batch_result_fields() {
    let mut engine = make_engine();
    let prompts = vec![vec![1u32, 2, 3]];
    let results = batch_generate(&mut engine, &prompts, 3);
    let br = results[0].as_ref().expect("should succeed");

    assert_eq!(br.prompt_tokens, 3);
    assert!(br.elapsed_seconds >= 0.0);
    assert!(br.generated_tokens.len() <= 3);
    assert!(br.finish_reason == FinishReason::Eos || br.finish_reason == FinishReason::MaxTokens);
}

#[test]
fn batch_result_clone() {
    let br = BatchResult {
        prompt_tokens: 5,
        generated_tokens: vec![10, 20, 30],
        finish_reason: FinishReason::MaxTokens,
        elapsed_seconds: 1.5,
    };
    let cloned = br.clone();
    assert_eq!(cloned.prompt_tokens, 5);
    assert_eq!(cloned.generated_tokens, vec![10, 20, 30]);
    assert_eq!(cloned.finish_reason, FinishReason::MaxTokens);
    assert!((cloned.elapsed_seconds - 1.5).abs() < f64::EPSILON);
}

// ── 8. Async engine generate empty prompt ────────────────────────────────

#[tokio::test]
async fn async_generate_empty_prompt() {
    let engine = make_engine();
    let async_engine = AsyncInferenceEngine::new(engine, 1);
    let result = async_engine.generate(vec![], 10).await;
    assert!(result.is_ok());
    let tokens = result.expect("should succeed");
    assert!(tokens.is_empty());
}

// ── 9. Async engine streaming empty prompt ───────────────────────────────

#[tokio::test]
async fn async_streaming_empty_prompt() {
    let engine = make_engine();
    let async_engine = AsyncInferenceEngine::new(engine, 1);
    let result = async_engine.generate_streaming(vec![], 10).await;
    assert!(result.is_ok());
    let mut rx = result.expect("should succeed");
    let token = rx.recv().await;
    assert!(token.is_none(), "no tokens for empty prompt");
}

// ── 10. Async engine concurrency respected ───────────────────────────────

#[tokio::test]
async fn async_concurrency_permits_returned() {
    let engine = make_engine();
    let async_engine = Arc::new(AsyncInferenceEngine::new(engine, 2));

    assert!(async_engine.has_capacity());
    assert_eq!(async_engine.active_requests(), 0);

    // Generate with empty prompt (fast)
    let r1 = async_engine.generate(vec![], 1).await;
    assert!(r1.is_ok());

    // Permits should be returned after completion
    assert!(async_engine.has_capacity());
    assert_eq!(async_engine.active_requests(), 0);
}

// ── 11. Batch with timeout config ────────────────────────────────────────

#[test]
fn batch_with_timeout_respects_max_batch_size() {
    let mut engine = make_engine();
    let config = BatchConfig {
        max_batch_size: 2,
        max_tokens_per_request: 5,
        timeout_per_request_ms: Some(5_000),
    };
    let prompts = vec![vec![1u32]; 5]; // 5 prompts, but max batch = 2
    let results = batch_generate_with_timeout(&mut engine, &prompts, &config);
    assert_eq!(
        results.len(),
        2,
        "should only process max_batch_size prompts"
    );
}

#[test]
fn batch_with_timeout_no_timeout_config() {
    let mut engine = make_engine();
    let config = BatchConfig {
        max_batch_size: 4,
        max_tokens_per_request: 3,
        timeout_per_request_ms: None,
    };
    let prompts = vec![vec![1u32], vec![2u32]];
    let results = batch_generate_with_timeout(&mut engine, &prompts, &config);
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.is_ok());
    }
}

// ── 12. RequestQueue edge cases ──────────────────────────────────────────

#[test]
fn request_queue_drain_empty() {
    let mut queue = RequestQueue::new(10);
    let batch = queue.drain_batch(5);
    assert!(batch.is_empty());
}

#[test]
fn request_queue_drain_more_than_available() {
    let mut queue = RequestQueue::new(10);
    queue
        .push(BatchRequest {
            prompt_tokens: vec![1],
            max_tokens: 5,
            params: SamplingParams::default(),
        })
        .expect("push should succeed");

    let batch = queue.drain_batch(100);
    assert_eq!(batch.len(), 1);
    assert!(queue.is_empty());
}

#[test]
fn request_queue_capacity_clamped() {
    let queue = RequestQueue::new(0);
    assert_eq!(queue.capacity(), 1, "min capacity should be 1");
}

// ── 13. Multiple sequential batch generations ────────────────────────────

#[test]
fn sequential_batch_generations() {
    let mut engine = make_engine();

    let results1 = batch_generate(&mut engine, &[vec![1u32]], 2);
    assert_eq!(results1.len(), 1);
    assert!(results1[0].is_ok());

    let results2 = batch_generate(&mut engine, &[vec![2u32], vec![3u32]], 2);
    assert_eq!(results2.len(), 2);
    for r in &results2 {
        assert!(r.is_ok());
    }
}

// ── 14. Engine batch_generate tracks stats ───────────────────────────────

#[test]
fn engine_batch_generate_tracks_stats() {
    let mut engine = make_engine();
    let prompts = vec![vec![1u32], vec![2u32], vec![3u32]];
    let _results = engine.batch_generate(&prompts, 2);

    let stats = engine.stats();
    assert!(
        stats.requests_completed() >= 3,
        "should record at least 3 requests"
    );
}

// ── SV-32: the async front end is backed by a replica pool ──────────────────

/// `SV-32`: `AsyncInferenceEngine` used to hold one `Mutex<InferenceEngine>`,
/// so `max_concurrent` bounded admission while every admitted request still
/// queued on a single engine. It now wraps an `EnginePool`, and the two
/// numbers are reported separately: `max_concurrent` (admission) and
/// `replicas` (execution).
#[test]
fn async_engine_reports_admission_and_execution_separately() {
    let async_engine = AsyncInferenceEngine::new(make_engine(), 4);
    assert_eq!(async_engine.max_concurrent(), 4);
    assert_eq!(
        async_engine.replicas(),
        1,
        "a lone engine is one replica, whatever the admission limit says"
    );
}

/// Built from a pool, the front end admits exactly what it can execute, and
/// concurrent requests reproduce their isolated baselines.
#[tokio::test]
async fn async_engine_from_pool_runs_requests_concurrently() {
    use oxibonsai_runtime::engine_pool::EnginePool;

    let prompts: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let baselines: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| make_engine().generate(p, 3).expect("baseline"))
        .collect();

    let pool = EnginePool::new(vec![make_engine(), make_engine()]);
    let async_engine = Arc::new(AsyncInferenceEngine::from_pool(pool));
    assert_eq!(async_engine.max_concurrent(), 2);
    assert_eq!(async_engine.replicas(), 2);

    let mut handles = Vec::new();
    for p in prompts {
        let ae = Arc::clone(&async_engine);
        handles.push(tokio::spawn(async move { ae.generate(p, 3).await }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        let got = h.await.expect("join").expect("generate");
        assert_eq!(got, baselines[i], "concurrent request {i} diverged");
    }
}
