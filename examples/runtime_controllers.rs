//! Demonstration of the 0.1.4 runtime controllers.
//!
//! Wires together [`KvCachePolicy`], [`AdaptiveLookahead`], [`RequestId`],
//! and [`RequestRateTracker`] / [`RequestRateAggregator`] in a single end-to-end
//! flow against a tiny test model. Prints periodic status so you can see the
//! controllers responding to synthesized workload pressure.
//!
//! Run with:
//! ```text
//! cargo run --example runtime_controllers
//! ```

use std::sync::Arc;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_runtime::adaptive_lookahead::{AdaptiveLookahead, AdaptiveLookaheadConfig};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::kv_cache_policy::{KvCacheLevel, KvCachePolicy, KvCachePolicyConfig};
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::request_id::RequestId;
use oxibonsai_runtime::request_metrics::RequestRateAggregator;
use oxibonsai_runtime::sampling::SamplingParams;

fn main() {
    let metrics = Arc::new(InferenceMetrics::new());

    // ─── 1. KV cache policy: starts FP16, escalates with pressure ───────────
    let kv_policy = KvCachePolicy::new(KvCachePolicyConfig {
        q8_threshold: 0.70,
        q4_threshold: 0.90,
        hysteresis: 0.05,
        ewma_alpha: 0.30,
        min_level: KvCacheLevel::Fp16,
        max_level: KvCacheLevel::Q4,
    })
    .expect("valid policy config");
    println!(
        "[kv] initial level = {} (memory factor = {})",
        kv_policy.current_level().tag(),
        kv_policy.current_level().memory_factor()
    );

    // ─── 2. Adaptive lookahead: tunes the speculative draft `k` ─────────────
    let mut lookahead = AdaptiveLookahead::new(AdaptiveLookaheadConfig {
        initial: 4,
        min: 2,
        max: 10,
        alpha: 0.30,
        cooldown_steps: 2,
    });
    println!(
        "[ah] initial lookahead = {} (window=[{}, {}])",
        lookahead.lookahead(),
        lookahead.config().min,
        lookahead.config().max
    );

    // ─── 3. Build engine with attached aggregator + metrics ─────────────────
    let aggregator = Arc::new(RequestRateAggregator::with_window(64));
    let mut engine = InferenceEngine::new(
        Qwen3Config::tiny_test(),
        SamplingParams::default(),
        0xc0_de_dc_af,
    );
    engine.set_metrics(Arc::clone(&metrics));
    engine.set_rate_aggregator(Arc::clone(&aggregator));
    println!("[engine] kernel = {}", engine.kernel().name());

    // ─── 4. Drive a workload, simulating cache pressure + acceptance ────────
    let prompt: Vec<u32> = (1..=8).collect();
    for round in 0..16 {
        // Generate one tracked request.
        let id = RequestId::new();
        let (tokens, tracker) = engine
            .generate_with_request_id(id, &prompt, 4)
            .expect("tracked generate");
        let snap = tracker.snapshot();

        // Simulate a synthetic "cache pressure" curve: early rounds are
        // light, later rounds saturate the cache.
        let pressure = (round as f64 / 16.0).min(1.0);
        let new_level = kv_policy.observe(pressure);

        // Simulate a synthetic "acceptance" pattern: high in the middle,
        // low at the edges (a typical realistic trajectory).
        let proposed = lookahead.lookahead();
        let accepted = if (4..=11).contains(&round) {
            proposed
        } else if round == 0 {
            0
        } else {
            (proposed / 2).max(1)
        };
        lookahead.observe_step(proposed, accepted);

        // Push to Prometheus gauges.
        metrics.update_kv_cache_level(new_level);
        metrics.update_request_rate(&aggregator.snapshot());

        println!(
            "[round {round:>2}] id={} tokens={} tps={:.1} k={} level={} pressure={:.2}",
            id.as_hex().chars().take(8).collect::<String>(),
            tokens.len(),
            snap.tokens_per_second,
            lookahead.lookahead(),
            new_level.tag(),
            pressure
        );
    }

    // ─── 5. Final state ─────────────────────────────────────────────────────
    let final_snapshot = aggregator.snapshot();
    println!();
    println!("=== Workload summary ===");
    println!(
        "  completed requests : {}",
        final_snapshot.completed_requests
    );
    println!(
        "  EMA tokens/sec     : {:.2}",
        final_snapshot.mean_tokens_per_second
    );
    println!(
        "  TBT p50/p95 (s)    : {:.6} / {:.6}",
        final_snapshot.tbt_p50_seconds, final_snapshot.tbt_p95_seconds
    );
    println!(
        "  mean queue wait (s): {:.6}",
        final_snapshot.mean_queue_wait_seconds
    );
    println!();
    println!("=== KV cache policy ===");
    println!(
        "  final level    : {} (factor {:.2})",
        kv_policy.current_level().tag(),
        kv_policy.current_level().memory_factor()
    );
    println!("  pressure (EWMA): {:.3}", kv_policy.pressure());
    println!("  upgrades       : {}", kv_policy.upgrades());
    println!("  downgrades     : {}", kv_policy.downgrades());
    println!();
    println!("=== Adaptive lookahead ===");
    println!("  final k   : {}", lookahead.lookahead());
    println!("  EWMA acc. : {:.2}", lookahead.mean_accepted());
    println!("  updates   : {}", lookahead.updates());
    println!();
    let prom = metrics.render_prometheus();
    println!("=== Prometheus surface (truncated) ===");
    for line in prom.lines().take(20) {
        println!("  {line}");
    }
    println!("  …");
}
