//! `oxibonsai chat` — interactive multi-turn conversation.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::util::{missing_tokenizer_warning, resolve_tokenizer};

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    model: Option<String>,
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

    let tok = {
        let lookup = resolve_tokenizer(tokenizer.as_deref(), &model);
        if let Some(tok_path) = lookup.found {
            Some(oxibonsai_runtime::TokenizerBridge::from_file(&tok_path)?)
        } else {
            tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
            None
        }
    };

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
            vec![151644]
        };

        // Clear any stale interrupt before starting generation
        interrupted.store(false, Ordering::SeqCst);

        let start = std::time::Instant::now();
        let (tx, rx) = std::sync::mpsc::channel::<u32>();

        // Use std::thread::scope so engine's borrow stays valid.
        // The receive loop breaks (dropping rx) when the interrupted flag
        // is set; generate_streaming_sync detects tx.send() failure and
        // returns cleanly, so gen_handle.join() never blocks indefinitely.
        let interrupted_ref = Arc::clone(&interrupted);
        // Fresh decode-stream state per chat turn: each `> ` cycle is an
        // independent generation request, so multi-byte UTF-8 buffering
        // must NOT carry across turns.
        let mut stream_state = tok.as_ref().map(|t| t.new_decode_stream(true));
        let output_count = std::thread::scope(|s| -> anyhow::Result<usize> {
            let thread_tx = tx.clone();
            let engine_ref = &mut engine;
            let tokens_ref = &prompt_tokens;
            let gen_handle = s.spawn(move || {
                engine_ref.generate_streaming_sync(tokens_ref, max_tokens, &thread_tx)
            });
            drop(tx);

            let mut count = 0usize;
            for token_id in rx {
                // Check cancellation before printing each token.
                // Breaking here drops the Receiver, which makes the
                // next tx.send() in the generation thread return Err,
                // stopping generation after at most one more forward pass.
                if interrupted_ref.load(Ordering::SeqCst) {
                    break;
                }
                count += 1;
                match (&tok, stream_state.as_mut()) {
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

        let elapsed = start.elapsed();
        println!(); // newline after streamed output

        if interrupted.swap(false, Ordering::SeqCst) {
            eprintln!("[interrupted after {} tokens]", output_count);
        } else {
            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                output_count as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            eprintln!(
                "[{} tokens in {:.2}s, {:.1} tok/s]",
                output_count,
                elapsed.as_secs_f64(),
                tok_per_sec
            );
        }
    }

    Ok(())
}
