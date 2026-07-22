//! `oxibonsai benchmark` — quick throughput benchmark (no real model weights required).

pub(crate) fn run(tokens: usize, warmup: usize, temperature: f32, seed: u64) -> anyhow::Result<()> {
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::model_cache::ModelWarmup;
    use oxibonsai_runtime::sampling::SamplingParams;

    let config = Qwen3Config::tiny_test();
    let params = SamplingParams {
        temperature,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    };

    let mut engine = oxibonsai_runtime::InferenceEngine::new(config, params.clone(), seed);

    // ── Warmup pass ──────────────────────────────────────────────
    let warmup_helper = ModelWarmup::new().with_tokens(warmup);
    let warmup_ms = warmup_helper.run(&mut engine, &params);
    eprintln!("Warmup: {warmup} tokens in {warmup_ms} ms");

    // ── Benchmark pass ───────────────────────────────────────────
    let prompt_tokens: Vec<u32> = vec![151644u32]; // <|im_start|>
    let bench_start = std::time::Instant::now();
    let output_tokens = engine.generate(&prompt_tokens, tokens)?;
    let bench_elapsed = bench_start.elapsed();

    let generated = output_tokens.len();
    let tok_per_sec = if bench_elapsed.as_secs_f64() > 0.0 {
        generated as f64 / bench_elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "Warmup: {warmup} tokens, Benchmark: {tok_per_sec:.1} tokens/sec \
         ({generated} tokens in {:.2}s)",
        bench_elapsed.as_secs_f64()
    );

    Ok(())
}
