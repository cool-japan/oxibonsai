//! `oxibonsai run` — single-shot inference on a GGUF model.

use std::io::{self, Write};

use super::util::{missing_tokenizer_warning, read_prompt_stdin, resolve_tokenizer};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: Option<String>,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    max_seq_len: usize,
    tokenizer: Option<String>,
) -> anyhow::Result<()> {
    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;
    let tokenizer = tokenizer.or_else(|| {
        std::env::var("OXI_TOKENIZER")
            .ok()
            .filter(|s| !s.is_empty())
    });

    let prompt_text = if prompt == "-" {
        read_prompt_stdin()
    } else {
        prompt
    };

    tracing::info!(
        model = %model,
        max_tokens,
        temperature,
        "starting inference"
    );

    // Memory-map the GGUF file
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

    let params = oxibonsai_runtime::sampling::SamplingParams {
        temperature,
        top_k,
        top_p,
        repetition_penalty: 1.1,
        ..oxibonsai_runtime::sampling::SamplingParams::default()
    };

    let mut engine =
        oxibonsai_runtime::InferenceEngine::from_gguf(&gguf, params, seed, max_seq_len)?;

    // Tokenize prompt and retain the bridge for streaming decode
    let lookup = resolve_tokenizer(tokenizer.as_deref(), &model);
    let (prompt_tokens, tok_bridge) = if let Some(tok_path) = &lookup.found {
        let tok = oxibonsai_runtime::TokenizerBridge::from_file(tok_path)?;
        let tokens = tok.encode(&prompt_text)?;
        (tokens, Some(tok))
    } else {
        tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
        (vec![151644], None) // <|im_start|>
    };

    tracing::info!(prompt_tokens = prompt_tokens.len(), "prefilling");

    let start = std::time::Instant::now();

    // Greedy GPU path: when temperature=0 and Metal is available,
    // run argmax on GPU and download only 4-byte token IDs instead
    // of the full ~607KB logits vector per token.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let use_greedy_gpu = temperature == 0.0;
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let use_greedy_gpu = false;

    let (prompt_len, output_count) = if use_greedy_gpu {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            tracing::info!("using greedy GPU path (argmax on Metal, 4-byte download)");
            let p_len = prompt_tokens.len();
            let tokens = engine.generate_greedy_gpu(&prompt_tokens, max_tokens)?;
            // Print tokens via the streaming decoder so multi-byte
            // UTF-8 characters that span more than one token (CJK,
            // emoji, etc.) round-trip correctly instead of
            // surfacing as U+FFFD replacement chars.
            let mut stream_state = tok_bridge.as_ref().map(|t| t.new_decode_stream(true));
            for &token_id in &tokens {
                match (&tok_bridge, stream_state.as_mut()) {
                    (Some(tok), Some(state)) => {
                        if let Some(text) = tok.step_decode(state, token_id)? {
                            print!("{text}");
                        }
                    }
                    _ => {
                        print!(" {token_id}");
                    }
                }
                let _ = io::stdout().flush();
            }
            (p_len, tokens.len())
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        unreachable!("use_greedy_gpu was set true outside the Metal+macOS feature path; check cfg-gate consistency in the use_greedy_gpu computation above")
    } else {
        // CUDA direct path (Linux / Windows): call engine.generate() directly on the
        // main thread, bypassing std::thread::scope and its ~107ms OS scheduling
        // overhead.  Tokens are printed synchronously after the full generation
        // completes, which is fine for non-interactive benchmarking.
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            let p_len = prompt_tokens.len();
            let tokens = engine.generate(&prompt_tokens, max_tokens)?;
            let mut stream_state = tok_bridge.as_ref().map(|t| t.new_decode_stream(true));
            for &token_id in &tokens {
                match (&tok_bridge, stream_state.as_mut()) {
                    (Some(tok), Some(state)) => {
                        if let Some(text) = tok.step_decode(state, token_id)? {
                            print!("{text}");
                        }
                    }
                    _ => {
                        print!(" {token_id}");
                    }
                }
                let _ = io::stdout().flush();
            }
            (p_len, tokens.len())
        }

        // All other platforms (no CUDA): streaming path via a worker thread so that
        // the main thread can decode and print tokens as they arrive.
        #[cfg(not(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        )))]
        {
            let (tx, rx) = std::sync::mpsc::channel::<u32>();

            let p_len = prompt_tokens.len();
            let count = std::thread::scope(|s| -> anyhow::Result<usize> {
                let thread_tx = tx.clone();
                let gen_handle = s.spawn(move || {
                    engine.generate_streaming_sync(&prompt_tokens, max_tokens, &thread_tx)
                });
                drop(tx);

                let mut stream_state = tok_bridge.as_ref().map(|t| t.new_decode_stream(true));
                let mut count = 0usize;
                for token_id in rx {
                    count += 1;
                    match (&tok_bridge, stream_state.as_mut()) {
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

                match gen_handle.join() {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => return Err(e.into()),
                    Err(_) => return Err(anyhow::anyhow!("generation thread panicked")),
                }
                Ok(count)
            })?;
            (p_len, count)
        }
    };

    let elapsed = start.elapsed();

    let total_tokens = prompt_len + output_count;
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        output_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!();
    eprintln!(
        "---\n{} prompt + {} generated = {} total tokens in {:.2}s ({:.1} tok/s)",
        prompt_len,
        output_count,
        total_tokens,
        elapsed.as_secs_f64(),
        tok_per_sec
    );

    // Print GPU profiling summary if OXIBONSAI_PROFILE_GPU=1 was set
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let model_size = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
        oxibonsai_kernels::print_gpu_profile_summary(model_size);
    }

    Ok(())
}
