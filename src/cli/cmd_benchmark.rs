//! `oxibonsai benchmark` — throughput benchmark.
//!
//! cli-13: with `--model`, benchmarks the real loaded model end to end;
//! without it, `--synthetic` must be passed explicitly to run the
//! untrained 2-layer toy benchmark, which is no longer the silent default.

use super::model_desc;
use super::tokenizer_backend::TokenizerBackendChoice;
use super::util::{
    build_sampling_params, missing_tokenizer_warning, model_vocab_size,
    resolve_tokenizer_vocab_aware,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: Option<String>,
    synthetic: bool,
    tokenizer: Option<String>,
    tokenizer_backend: TokenizerBackendChoice,
    tokens: usize,
    warmup: usize,
    temperature: f32,
    seed: u64,
) -> anyhow::Result<()> {
    let model = model.or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()));

    match model {
        Some(model) => run_real_model(
            &model,
            tokenizer,
            tokenizer_backend,
            tokens,
            warmup,
            temperature,
            seed,
        ),
        None => {
            if !synthetic {
                anyhow::bail!(
                    "benchmark requires either --model <gguf> (benchmark a real model) or \
                     --synthetic (run the untrained 2-layer toy benchmark, whose numbers are \
                     not representative of any real model); pass one explicitly"
                );
            }
            run_synthetic(tokens, warmup, temperature, seed)
        }
    }
}

/// Benchmark a real, loaded GGUF model end to end (tokenizer, prefill,
/// decode) — the honest replacement for the toy-only default (cli-13).
fn run_real_model(
    model: &str,
    tokenizer: Option<String>,
    tokenizer_backend: TokenizerBackendChoice,
    tokens: usize,
    warmup: usize,
    temperature: f32,
    seed: u64,
) -> anyhow::Result<()> {
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;
    let arch = gguf
        .metadata
        .get_string(oxibonsai_core::gguf::tensor_info::keys::GENERAL_ARCHITECTURE)
        .unwrap_or("")
        .to_string();

    let params = build_sampling_params(temperature, 40, 0.9, 1.0);
    let mut engine = oxibonsai_runtime::InferenceEngine::from_gguf(
        &gguf,
        params,
        seed,
        super::bonsai2::default_max_seq_len(&arch),
    )?;

    // cli-16: the engine's own resolved variant, quant type, kernel label
    // and tier — never the raw parse-time tensor-type guess.
    eprintln!("{}", model_desc::engine_summary(&engine));

    let expected_vocab = model_vocab_size(&gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(tokenizer.as_deref(), model, expected_vocab);
    // Shared with `run` (GGUF-embedded tokenizer fallback, the
    // GGUF's own chat template, the TOK-08 compatibility check), loading the
    // on-disk candidate through the chosen `--tokenizer-backend`.
    let resolved = super::cmd_run::resolve_model_tokenizer(
        tokenizer.as_deref(),
        &lookup,
        &gguf,
        expected_vocab,
        tokenizer_backend,
        false,
    )?;
    let prompt_tokens: Vec<u32> = if let Some(tok) = resolved {
        // A short, fixed benchmark prompt: real tokenization, not a
        // synthetic byte-derived sequence.
        tok.encode("The quick brown fox jumps over the lazy dog.")?
    } else {
        tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
        let bos = gguf
            .metadata
            .get_u32(oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_BOS_TOKEN_ID)
            .ok();
        bos.map(|id| vec![id]).unwrap_or_default()
    };
    if prompt_tokens.is_empty() {
        anyhow::bail!(
            "cannot benchmark: no tokenizer was found and the model declares no BOS token id. \
             Pass --tokenizer <path/to/tokenizer.json>."
        );
    }

    // Warmup pass: real forward passes on the real model, propagating any
    // failure instead of `ModelWarmup::run`'s "non-fatal" swallow (which
    // would otherwise let a broken warmup still print a plausible-looking
    // elapsed time for zero real work).
    let warmup_start = std::time::Instant::now();
    if warmup > 0 {
        engine.generate(&prompt_tokens, warmup)?;
        engine.reset();
    }
    let warmup_ms = warmup_start.elapsed().as_millis();
    eprintln!("Warmup: {warmup} tokens in {warmup_ms} ms");

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
        "Model: {model}\nWarmup: {warmup} tokens, Benchmark: {tok_per_sec:.1} tokens/sec \
         ({generated} tokens in {:.2}s)",
        bench_elapsed.as_secs_f64()
    );

    Ok(())
}

/// The original synthetic, untrained 2-layer toy benchmark. Only reachable
/// via explicit `--synthetic` (cli-13): its numbers are roughly two orders
/// of magnitude off any real model and must never be the silent default.
fn run_synthetic(tokens: usize, warmup: usize, temperature: f32, seed: u64) -> anyhow::Result<()> {
    use oxibonsai_core::config::Qwen3Config;

    println!(
        "WARNING: --synthetic benchmarks an untrained 2-layer toy model with no real weights; \
         these numbers are NOT representative of any real model's throughput."
    );

    let config = Qwen3Config::tiny_test();
    let params = build_sampling_params(temperature, 40, 0.9, 1.0);

    let mut engine = oxibonsai_runtime::InferenceEngine::new(config, params, seed);

    // No hardcoded token id (cli-07): the synthetic config's vocabulary
    // includes id 0, which is always in range regardless of vocab_size.
    let prompt_tokens: Vec<u32> = vec![0u32];

    // Warmup pass: a real forward pass on the synthetic engine,
    // propagating any failure instead of `ModelWarmup::run`'s "non-fatal"
    // swallow (model_cache.rs), which returns a plausible-looking elapsed
    // time even when the warmup produced zero tokens — mirrors
    // `run_real_model`'s warmup honesty fix exactly, so `--warmup N` on
    // *either* path only ever reports a time for `N` tokens that were
    // actually generated.
    let warmup_start = std::time::Instant::now();
    if warmup > 0 {
        engine.generate(&prompt_tokens, warmup)?;
        engine.reset();
    }
    let warmup_ms = warmup_start.elapsed().as_millis();
    eprintln!("Warmup: {warmup} tokens in {warmup_ms} ms");

    // ── Benchmark pass ───────────────────────────────────────────
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
