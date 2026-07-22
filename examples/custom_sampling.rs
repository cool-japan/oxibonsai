//! Custom sampling strategies demonstration.
//!
//! Compares different sampling presets and shows how to build
//! a custom sampler using `SamplerBuilder` with validation.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::builders::SamplerBuilder;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::presets::SamplingPreset;
use oxibonsai_runtime::sampling::SamplingParams;

fn main() -> anyhow::Result<()> {
    println!("=== OxiBonsai Custom Sampling Demo ===\n");

    let config = Qwen3Config::tiny_test();
    let prompt_tokens: Vec<u32> = vec![151644, 872, 1059];
    let max_tokens: usize = 10;

    // ─── 1. Compare presets ──────────────────────────────────────
    println!("--- Comparing Sampling Presets ---\n");

    let presets = [
        SamplingPreset::Greedy,
        SamplingPreset::Precise,
        SamplingPreset::Balanced,
        SamplingPreset::Conversational,
        SamplingPreset::Creative,
    ];

    for preset in &presets {
        let params = preset.params();
        let mut engine = InferenceEngine::new(config.clone(), params, 42);
        let tokens = engine.generate(&prompt_tokens, max_tokens)?;

        println!(
            "{:15} => {} tokens: {:?}",
            preset.name(),
            tokens.len(),
            tokens,
        );
    }

    // ─── 2. Custom sampler via builder ───────────────────────────
    println!("\n--- Custom Sampler via SamplerBuilder ---\n");

    // Build a sampler with very specific parameters
    let sampler = SamplerBuilder::new()
        .temperature(0.3)
        .top_k(20)
        .top_p(0.85)
        .repetition_penalty(1.3)
        .seed(999)
        .build()?;

    let p = sampler.params();
    println!(
        "Custom sampler: temp={:.1} top_k={} top_p={:.2} rep_pen={:.1}",
        p.temperature, p.top_k, p.top_p, p.repetition_penalty,
    );

    // Use the custom params with the engine
    let custom_params = SamplingParams {
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        repetition_penalty: p.repetition_penalty,
        ..SamplingParams::default()
    };
    let mut engine = InferenceEngine::new(config.clone(), custom_params, 999);
    let tokens = engine.generate(&prompt_tokens, max_tokens)?;
    println!("Generated: {:?}\n", tokens);

    // ─── 3. Validation demo ──────────────────────────────────────
    println!("--- Builder Validation ---\n");

    let bad_temp = SamplerBuilder::new().temperature(-1.0).build();
    match bad_temp {
        Ok(_) => println!("Unexpected: negative temperature accepted"),
        Err(e) => println!("Rejected negative temperature: {}", e),
    }

    let bad_top_p = SamplerBuilder::new().top_p(1.5).build();
    match bad_top_p {
        Ok(_) => println!("Unexpected: top_p > 1.0 accepted"),
        Err(e) => println!("Rejected invalid top_p: {}", e),
    }

    let bad_rep = SamplerBuilder::new().repetition_penalty(0.5).build();
    match bad_rep {
        Ok(_) => println!("Unexpected: rep_pen < 1.0 accepted"),
        Err(e) => println!("Rejected invalid repetition_penalty: {}", e),
    }

    // ─── 4. Effect of repetition penalty ─────────────────────────
    println!("\n--- Repetition Penalty Effect ---\n");

    let penalties = [1.0, 1.1, 1.3, 1.5];
    for &rp in &penalties {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: rp,
            ..SamplingParams::default()
        };
        let mut engine = InferenceEngine::new(config.clone(), params, 42);
        let tokens = engine.generate(&prompt_tokens, max_tokens)?;

        // Count unique tokens as a rough diversity measure
        let mut unique = tokens.clone();
        unique.sort();
        unique.dedup();
        println!(
            "rep_pen={:.1} => {} tokens, {} unique: {:?}",
            rp,
            tokens.len(),
            unique.len(),
            tokens,
        );
    }

    println!("\nDone.");
    Ok(())
}
