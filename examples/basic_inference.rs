//! Basic inference pipeline demonstration for OxiBonsai.
//!
//! Shows the minimal steps to set up an inference engine using a
//! tiny test configuration (no real GGUF model file needed):
//!
//! 1. Create a `Qwen3Config` via `tiny_test()`
//! 2. Choose sampling parameters from a preset
//! 3. Build an `InferenceEngine`
//! 4. Generate tokens from a dummy prompt
//! 5. Print formatted results

use std::sync::Arc;
use std::time::Instant;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_kernels::OneBitKernel;
use oxibonsai_runtime::convenience::{format_bytes, format_token_count, format_tokens_per_second};
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::presets::SamplingPreset;
use oxibonsai_runtime::sampling::SamplingParams;

fn main() -> anyhow::Result<()> {
    // ─── 1. Configuration ────────────────────────────────────────
    // Use a tiny configuration so the example runs instantly.
    // In production you would use Qwen3Config::bonsai_8b() or load
    // from a GGUF file via InferenceEngine::from_gguf().
    let config = Qwen3Config::tiny_test();

    println!("=== OxiBonsai Basic Inference Demo ===\n");
    println!(
        "Model: {} (hidden={}, layers={}, heads={})",
        config.model_name, config.hidden_size, config.num_layers, config.num_attention_heads,
    );
    println!("Vocab size: {}", format_token_count(config.vocab_size));

    // ─── 2. Sampling parameters ──────────────────────────────────
    // Start with a named preset and display its parameters.
    let preset = SamplingPreset::Balanced;
    let params: SamplingParams = preset.params();

    println!(
        "\nSampling preset: {} — {}",
        preset.name(),
        preset.description()
    );
    println!(
        "  temperature={:.1}  top_k={}  top_p={:.2}  rep_pen={:.1}",
        params.temperature, params.top_k, params.top_p, params.repetition_penalty,
    );

    // ─── 3. Create the inference engine ──────────────────────────
    let seed: u64 = 42;
    let mut engine = InferenceEngine::new(config.clone(), params, seed);

    // Optionally attach metrics for telemetry
    let metrics = Arc::new(InferenceMetrics::new());
    engine.set_metrics(Arc::clone(&metrics));

    println!("\nEngine initialized:");
    println!("  Kernel: {}", engine.kernel().name());

    // ─── 4. Generate tokens ──────────────────────────────────────
    // A dummy prompt: in a real scenario these would come from a tokenizer.
    let prompt_tokens: Vec<u32> = vec![151644, 872, 1059, 315, 2324]; // "<|im_start|> The capital of France"
    let max_tokens = 20;

    println!(
        "\nPrompt ({} tokens): {:?}",
        prompt_tokens.len(),
        prompt_tokens
    );
    println!("Generating up to {} tokens...\n", max_tokens);

    let start = Instant::now();
    let generated = engine.generate(&prompt_tokens, max_tokens)?;
    let elapsed = start.elapsed();

    // ─── 5. Display results ──────────────────────────────────────
    let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
        generated.len() as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("Generated {} tokens in {:.2?}", generated.len(), elapsed);
    println!("Speed: {}", format_tokens_per_second(tokens_per_sec));
    println!("Token IDs: {:?}", generated);

    // ─── 6. Engine statistics ────────────────────────────────────
    let stats = engine.stats();
    println!("\n--- Engine Statistics ---");
    println!("Total tokens generated: {}", stats.tokens_generated());
    println!("Total requests: {}", stats.requests_completed());
    println!("Avg tokens/request: {:.1}", stats.avg_tokens_per_request());
    println!("Uptime: {:.2}s", stats.uptime_seconds());

    // ─── 7. Metrics summary ──────────────────────────────────────
    println!("\n--- Metrics Summary ---");
    println!(
        "Tokens generated (counter): {}",
        metrics.tokens_generated_total.get()
    );
    println!("Requests (counter): {}", metrics.requests_total.get());

    // ─── 8. Memory estimation ────────────────────────────────────
    let est = oxibonsai_runtime::convenience::estimate_memory_requirements(
        2_200_000_000, // ~2.2 GB model file
        4096,
        config.num_kv_heads,
        config.head_dim,
        config.num_layers,
    );
    println!("\n--- Memory Estimate (Bonsai-8B) ---");
    println!("Model weights: {}", format_bytes(est.model_weights_bytes));
    println!("KV cache:      {}", format_bytes(est.kv_cache_bytes));
    println!(
        "Runtime:       {}",
        format_bytes(est.runtime_overhead_bytes)
    );
    println!("Total:         {}", format_bytes(est.total_bytes));
    println!(
        "Fits in memory: {}",
        if est.fits_in_memory { "yes" } else { "no" }
    );

    // ─── 9. Show all available presets ───────────────────────────
    println!("\n--- Available Sampling Presets ---");
    for p in SamplingPreset::all() {
        let sp = p.params();
        println!(
            "  {:15} temp={:.1}  top_k={:3}  top_p={:.2}  rep={:.1} -- {}",
            p.name(),
            sp.temperature,
            sp.top_k,
            sp.top_p,
            sp.repetition_penalty,
            p.description(),
        );
    }

    println!("\nDone.");
    Ok(())
}
