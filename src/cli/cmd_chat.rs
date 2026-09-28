//! `oxibonsai chat` — interactive multi-turn conversation.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::model_desc;
use super::tokenizer_backend::TokenizerBackendChoice;
use super::util::{
    build_sampling_params, missing_tokenizer_warning, model_vocab_size,
    reject_penalties_with_constrained_decode, resolve_tokenizer_vocab_aware, StopChecker,
};

/// Resolved arguments for `oxibonsai chat`, merged from CLI flags and
/// `--config` in `mod.rs` (cli-04). See [`super::cmd_run::RunArgs`] for why
/// this is a struct rather than positional parameters.
pub(crate) struct ChatArgs {
    pub(crate) model: Option<String>,
    pub(crate) max_tokens: usize,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) repetition_penalty: f32,
    pub(crate) frequency_penalty: f32,
    pub(crate) presence_penalty: f32,
    pub(crate) seed: u64,
    pub(crate) max_seq_len: usize,
    pub(crate) tokenizer: Option<String>,
    pub(crate) tokenizer_backend: TokenizerBackendChoice,
    pub(crate) grammar: Option<String>,
    pub(crate) stop: Vec<String>,
    pub(crate) allow_vocab_mismatch: bool,
}

pub(crate) fn run(args: ChatArgs) -> anyhow::Result<()> {
    let ChatArgs {
        model,
        max_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        seed,
        max_seq_len,
        tokenizer,
        tokenizer_backend,
        grammar,
        stop,
        allow_vocab_mismatch,
    } = args;

    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;

    // Memory-map the GGUF file
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

    // cli-12 / orchestrator P0 addendum: same shared constructor as `run`,
    // with the same 1.0/0.0/0.0 no-hidden-penalty defaults.
    let params = build_sampling_params(temperature, top_k, top_p, repetition_penalty);

    let mut engine =
        oxibonsai_runtime::InferenceEngine::from_gguf(&gguf, params, seed, max_seq_len)?;
    engine.set_penalties(oxibonsai_runtime::PenaltyParams::new(
        frequency_penalty,
        presence_penalty,
    ));

    // cli-16: report the RESOLVED quant variant + effective kernel tier.
    if let Ok(config) = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata) {
        let dominant_type = gguf
            .tensors
            .count_by_type()
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(ty, _)| *ty)
            .unwrap_or(oxibonsai_core::GgufTensorType::Q1_0_g128);
        let variant = oxibonsai_model::ModelVariant::from_config_and_sample_tensor_type(
            &config,
            dominant_type,
        );
        eprintln!(
            "{}",
            model_desc::resolved_engine_summary(
                variant.name(),
                dominant_type,
                engine.kernel_tier(),
                &engine.kernel().effective_tier_reason(),
            )
        );
    }

    // TOK-08: vocab-aware resolution + a hard compatibility check,
    // regardless of whether the tokenizer path came from auto-detection
    // or an explicit --tokenizer.
    let expected_vocab = model_vocab_size(&gguf).ok();
    // ENGINE-SEAM: shared with `run` (GGUF-embedded tokenizer fallback).
    let tok = {
        let lookup = resolve_tokenizer_vocab_aware(tokenizer.as_deref(), &model, expected_vocab);
        let resolved = super::cmd_run::resolve_model_tokenizer(
            tokenizer.as_deref(),
            &lookup,
            &gguf,
            expected_vocab,
            tokenizer_backend,
            allow_vocab_mismatch,
        )?;
        if resolved.is_none() {
            tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
        }
        resolved
    };

    if grammar.is_some() && tok.is_none() {
        anyhow::bail!("--grammar requires a tokenizer; none was found (see the warning above)");
    }
    // Loaded once for the whole session; a fresh `ConstrainedSampler` (and
    // therefore fresh recognizer state) is built from it every turn — see
    // `cmd_run::build_constrained_sampler_from_grammar`'s doc.
    let cached_grammar = match &grammar {
        Some(path) => Some(super::cmd_run::load_grammar(path)?),
        None => None,
    };

    let use_constrained_or_stop = cached_grammar.is_some() || !stop.is_empty();
    reject_penalties_with_constrained_decode(
        use_constrained_or_stop,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
    )?;

    println!("OxiBonsai Interactive Chat (type 'quit' or Ctrl-D to exit)");
    println!("Tip: press Ctrl-C during generation to interrupt output without exiting.");
    println!("---");

    // Shared cancellation flag.  The ctrlc handler sets this to true;
    // the generation receive loop checks it and drops the receiver,
    // which causes tx.send() in generate_streaming_sync to fail and
    // stop the generation thread naturally.
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&interrupted);
        ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        })
        .map_err(|e| anyhow::anyhow!("failed to install Ctrl-C handler: {e}"))?;
    }

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                // EOF (Ctrl-D)
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                // Ctrl-C while waiting for input (not during generation).
                interrupted.store(false, Ordering::SeqCst);
                println!();
                eprintln!("[Ctrl-C: type 'quit' or press Ctrl-D to exit]");
                continue;
            }
            Err(e) => return Err(e.into()),
        }
        let input = input.trim();
        if input.is_empty() {
            // Reset stale interrupt flag that fired just before the prompt
            interrupted.store(false, Ordering::SeqCst);
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }
        if input == "/reset" {
            engine.reset();
            println!("[context cleared]");
            continue;
        }

        let prompt_tokens = if let Some(tok) = &tok {
            tok.encode(input)?
        } else {
            // cli-07: no hardcoded token id. Without a tokenizer there is
            // no honest way to turn `input` into token ids at all.
            anyhow::bail!(
                "cannot encode input: no tokenizer was found. Pass --tokenizer \
                 <path/to/tokenizer.json>."
            );
        };

        // Clear any stale interrupt before starting generation
        interrupted.store(false, Ordering::SeqCst);

        let start = std::time::Instant::now();

        let output_count = if use_constrained_or_stop {
            run_constrained_or_stopped_turn(
                &mut engine,
                &prompt_tokens,
                max_tokens,
                cached_grammar.as_ref(),
                &stop,
                tok.as_ref(),
                seed,
                temperature,
                top_k,
                top_p,
                &interrupted,
            )?
        } else {
            run_streaming_turn(
                &mut engine,
                &prompt_tokens,
                max_tokens,
                tok.as_ref(),
                &interrupted,
            )?
        };

        let elapsed = start.elapsed();
        println!(); // newline after streamed output

        if interrupted.swap(false, Ordering::SeqCst) {
            eprintln!("[interrupted after {output_count} tokens]");
        } else {
            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                output_count as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            eprintln!(
                "[{output_count} tokens in {:.2}s, {tok_per_sec:.1} tok/s]",
                elapsed.as_secs_f64()
            );
        }
    }

    Ok(())
}

/// One turn of the original streaming decode loop (no `--grammar`/`--stop`).
fn run_streaming_turn(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    tok: Option<&oxibonsai_runtime::TokenizerBridge>,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<usize> {
    let (tx, rx) = std::sync::mpsc::channel::<u32>();

    // Fresh decode-stream state per chat turn: each `> ` cycle is an
    // independent generation request, so multi-byte UTF-8 buffering
    // must NOT carry across turns.
    let mut stream_state = tok.map(|t| t.new_decode_stream(true));
    let output_count = std::thread::scope(|s| -> anyhow::Result<usize> {
        let thread_tx = tx.clone();
        let gen_handle =
            s.spawn(move || engine.generate_streaming_sync(prompt_tokens, max_tokens, &thread_tx));
        drop(tx);

        let mut count = 0usize;
        for token_id in rx {
            // Check cancellation before printing each token. Breaking
            // here drops the Receiver, which makes the next tx.send() in
            // the generation thread return Err, stopping generation after
            // at most one more forward pass.
            if interrupted.load(Ordering::SeqCst) {
                break;
            }
            count += 1;
            match (tok, stream_state.as_mut()) {
                (Some(tok), Some(state)) => {
                    if let Some(text) = tok.step_decode(state, token_id)? {
                        print!("{text}");
                    }
                }
                _ => {
                    if count == 1 {
                        print!("Tokens:");
                    }
                    print!(" {token_id}");
                }
            }
            let _ = io::stdout().flush();
        }
        // rx is dropped here; gen_handle will stop within one token step

        match gen_handle.join() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => return Err(anyhow::anyhow!("generation thread panicked")),
        }
        Ok(count)
    })?;
    Ok(output_count)
}

/// One turn of the grammar-constrained and/or stop-sequence-aware decode
/// loop (cli-17). See [`super::cmd_run`]'s equivalent for the design
/// rationale (single portable loop, buffered-then-printed output, no
/// penalties applied — with or without a grammar; `run`'s caller rejects a
/// non-default penalty before this is ever reached, see
/// [`super::util::reject_penalties_with_constrained_decode`]).
#[allow(clippy::too_many_arguments)]
fn run_constrained_or_stopped_turn(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    cached_grammar: Option<&oxibonsai_runtime::Grammar>,
    stop: &[String],
    tok: Option<&oxibonsai_runtime::TokenizerBridge>,
    seed: u64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<usize> {
    engine.reset();
    let prompt_len = prompt_tokens.len();
    let mut logits = engine.prefill_from_pos(prompt_tokens, 0)?;

    let mut constrained = cached_grammar.map(|g| {
        super::cmd_run::build_constrained_sampler_from_grammar(
            g.clone(),
            tok,
            tok.map(|t| t.vocab_size()).unwrap_or(logits.len()),
            seed,
            temperature,
            top_k,
            top_p,
        )
    });

    let stop_checker = StopChecker::new(stop.to_vec());
    let mut accumulated = String::new();
    let mut stream_state = tok.map(|t| t.new_decode_stream(true));
    let mut generated = 0usize;
    let mut pos = prompt_len;

    while generated < max_tokens {
        if interrupted.load(Ordering::SeqCst) || logits.is_empty() {
            break;
        }
        let token = match constrained.as_mut() {
            Some(cs) => cs.sample(&mut logits),
            None => engine.sample(&logits)?,
        };
        if token == engine.eos_token_id() {
            break;
        }
        generated += 1;
        match (tok, stream_state.as_mut()) {
            (Some(tok), Some(state)) => {
                if let Some(text) = tok.step_decode(state, token)? {
                    accumulated.push_str(&text);
                }
            }
            _ => {
                accumulated.push(' ');
                accumulated.push_str(&token.to_string());
            }
        }
        if let Some(cs) = constrained.as_ref() {
            if cs.is_complete() {
                break;
            }
        }
        if !stop_checker.is_empty() && stop_checker.check(&accumulated) {
            break;
        }
        logits = engine.decode_step(token, pos)?;
        pos += 1;
    }

    let truncated = stop_checker.truncate_at_stop(&accumulated);
    print!("{truncated}");
    io::stdout().flush()?;

    Ok(generated)
}
