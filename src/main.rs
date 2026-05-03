//! OxiBonsai — 1-bit LLM inference engine for Bonsai-8B.
//!
//! This binary is not functional on WASM targets; the WASM entry points
//! are exposed via [`oxibonsai_runtime`] library APIs instead.

/// WASM stub: this binary is a no-op on wasm32 targets.
/// Consumers should use the `oxibonsai_runtime` library crate APIs directly.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod cli {
    use clap::{Parser, Subcommand};
    use std::io::{self, BufRead, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use oxibonsai_runtime::OxiBonsaiConfig;

    #[derive(Parser)]
    #[command(
        name = "oxibonsai",
        version,
        about = "1-bit LLM inference engine for Bonsai-8B"
    )]
    pub struct Cli {
        /// Path to an OxiBonsai TOML configuration file.
        #[arg(long, global = true)]
        config: Option<String>,

        #[command(subcommand)]
        pub command: Commands,
    }

    #[derive(Subcommand)]
    pub enum Commands {
        /// Run inference on a GGUF model.
        Run {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Prompt text. Use "-" for stdin.
            #[arg(short, long)]
            prompt: String,

            /// Maximum number of tokens to generate.
            #[arg(long, default_value_t = 256)]
            max_tokens: usize,

            /// Sampling temperature (0.0 = greedy).
            #[arg(long, default_value_t = 0.7)]
            temperature: f32,

            /// Top-k sampling (0 = disabled).
            #[arg(long, default_value_t = 40)]
            top_k: usize,

            /// Top-p (nucleus) sampling.
            #[arg(long, default_value_t = 0.9)]
            top_p: f32,

            /// Random seed.
            #[arg(long, default_value_t = 42)]
            seed: u64,

            /// Maximum sequence length (prompt + generated).
            #[arg(long, default_value_t = 4096)]
            max_seq_len: usize,

            /// Path to tokenizer.json file.
            #[arg(long)]
            tokenizer: Option<String>,
        },

        /// Interactive multi-turn conversation.
        Chat {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Maximum number of tokens to generate per turn.
            #[arg(long, default_value_t = 512)]
            max_tokens: usize,

            /// Sampling temperature (0.0 = greedy).
            #[arg(long, default_value_t = 0.7)]
            temperature: f32,

            /// Top-k sampling (0 = disabled).
            #[arg(long, default_value_t = 40)]
            top_k: usize,

            /// Top-p (nucleus) sampling.
            #[arg(long, default_value_t = 0.9)]
            top_p: f32,

            /// Random seed.
            #[arg(long, default_value_t = 42)]
            seed: u64,

            /// Maximum sequence length.
            #[arg(long, default_value_t = 4096)]
            max_seq_len: usize,

            /// Path to tokenizer.json file.
            #[arg(long)]
            tokenizer: Option<String>,
        },

        /// Start an OpenAI-compatible API server.
        #[cfg(feature = "server")]
        Serve {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Host to bind to.
            #[arg(long, default_value = "127.0.0.1")]
            host: String,

            /// Port to listen on.
            #[arg(long, default_value_t = 8080)]
            port: u16,

            /// Maximum sequence length.
            #[arg(long, default_value_t = 4096)]
            max_seq_len: usize,

            /// Path to tokenizer.json file.
            #[arg(long)]
            tokenizer: Option<String>,
        },

        /// Display model info from a GGUF file.
        Info {
            /// Path to the GGUF model file.
            #[arg(short, long)]
            model: String,

            /// Emit info as JSON instead of human-readable text.
            #[arg(long, default_value_t = false)]
            json: bool,
        },

        /// Run a quick throughput benchmark (no real model weights required).
        Benchmark {
            /// Total tokens to generate during the benchmark pass.
            #[arg(long, default_value_t = 100)]
            tokens: usize,

            /// Number of warmup tokens generated before timing begins.
            #[arg(long, default_value_t = 10)]
            warmup: usize,

            /// Sampling temperature.
            #[arg(long, default_value_t = 0.7)]
            temperature: f32,

            /// Random seed.
            #[arg(long, default_value_t = 42)]
            seed: u64,
        },

        /// Quantize a GGUF model to a lower-precision format (simulation).
        Quantize {
            /// Path to the input GGUF model file.
            #[arg(long)]
            input: String,

            /// Destination path for the quantized file.
            #[arg(long)]
            output: String,

            /// Target quantization format (e.g. q1_0, q4_0, q8_0).
            #[arg(long, default_value = "q1_0")]
            format: String,
        },

        /// Validate that a GGUF file is well-formed and display a metadata summary.
        Validate {
            /// Path to the GGUF model file to validate.
            #[arg(short, long)]
            model: String,
        },

        /// Convert a HuggingFace safetensors model to GGUF format.
        Convert {
            /// Input directory containing model.safetensors (or shards) and config.json.
            #[arg(long)]
            from: String,

            /// Output GGUF file path.
            #[arg(long)]
            to: String,

            /// Quantization format: tq2_0_g128 (default), q1_0_g128.
            #[arg(long, default_value = "tq2_0_g128")]
            quant: String,

            /// Treat --from as an ONNX model file (MatMulNBits, bits=2) and use the ONNX→GGUF converter.
            #[arg(long, default_value_t = false)]
            onnx: bool,
        },

        /// Manage the Qwen3 tokenizer (download / inspect).
        Tokenizer {
            #[command(subcommand)]
            cmd: TokenizerCmd,
        },
    }

    #[derive(Subcommand)]
    pub enum TokenizerCmd {
        /// Download tokenizer.json from HuggingFace and save it next to the model.
        ///
        /// Example:
        ///   oxibonsai tokenizer download --output models/tokenizer.json
        Download {
            /// Destination path (default: models/tokenizer.json in the current directory).
            #[arg(long, default_value = "models/tokenizer.json")]
            output: String,

            /// HuggingFace repo to download from (must contain tokenizer.json).
            #[arg(long, default_value = "Qwen/Qwen3-8B")]
            repo: String,

            /// Overwrite an existing tokenizer.json without prompting.
            #[arg(long, default_value_t = false)]
            force: bool,
        },

        /// Show the vocabulary size and model type stored in tokenizer.json.
        Info {
            /// Path to tokenizer.json.
            #[arg(long, default_value = "models/tokenizer.json")]
            path: String,
        },
    }

    pub fn read_prompt_stdin() -> String {
        let stdin = io::stdin();
        let mut lines = Vec::new();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => lines.push(l),
                Err(_) => break,
            }
        }
        lines.join("\n")
    }

    /// Result of attempting to locate a `tokenizer.json` for a given model.
    ///
    /// `found` is `Some(path)` when either an explicit override was supplied
    /// or auto-detection succeeded.  `searched` lists every candidate path
    /// inspected during auto-detection so the user can see exactly where we
    /// looked when nothing turned up.
    pub(crate) struct TokenizerLookup {
        pub(crate) found: Option<String>,
        pub(crate) searched: Vec<PathBuf>,
    }

    /// Strip a trailing GGUF quantization suffix (e.g. `-Q2_0`, `-Q4_K_M`,
    /// `-F16`, `-BF16`, `-F32`) from a model basename without pulling in the
    /// `regex` crate.  Returns the basename unchanged when no recognized
    /// suffix is present.
    pub(crate) fn strip_quant_suffix(basename: &str) -> &str {
        // Locate the last '-' segment; only that segment is a quant suffix
        // candidate.
        let Some(dash_pos) = basename.rfind('-') else {
            return basename;
        };
        let suffix = &basename[dash_pos + 1..];
        if suffix.is_empty() {
            return basename;
        }

        let is_float = matches!(suffix, "F16" | "BF16" | "F32");
        let is_quant = {
            let mut chars = suffix.chars();
            match chars.next() {
                Some('Q') => {
                    // Accept Q<digits>(_<alnum>+)*  e.g. Q1_0, Q2_K, Q4_K_M, Q8_0.
                    let rest: String = chars.collect();
                    if rest.is_empty() {
                        false
                    } else {
                        let mut parts = rest.split('_');
                        let first = parts.next().unwrap_or("");
                        if first.is_empty() || !first.chars().all(|c| c.is_ascii_digit()) {
                            false
                        } else {
                            parts.all(|p| {
                                !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric())
                            })
                        }
                    }
                }
                _ => false,
            }
        };

        if is_float || is_quant {
            &basename[..dash_pos]
        } else {
            basename
        }
    }

    /// Build the ordered list of candidate `tokenizer.json` paths to probe
    /// for a given model file.  Duplicates (after canonical lexical form)
    /// are removed so the warning message stays compact.
    pub(crate) fn tokenizer_candidates(model_path: &Path) -> Vec<PathBuf> {
        let parent = model_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let mut out: Vec<PathBuf> = Vec::new();
        let push_unique = |p: PathBuf, out: &mut Vec<PathBuf>| {
            if !out.iter().any(|existing| existing == &p) {
                out.push(p);
            }
        };

        // 1. Same directory as the model.
        push_unique(parent.join("tokenizer.json"), &mut out);

        // 2. Parent of the model directory.
        push_unique(parent.join("..").join("tokenizer.json"), &mut out);

        // 3. Sibling directories derived from the model basename.
        if let Some(stem) = model_path.file_stem().and_then(|s| s.to_str()) {
            let base = strip_quant_suffix(stem);
            for variant in [
                base.to_string(),
                format!("{base}-unpacked"),
                format!("{base}-ONNX"),
            ] {
                push_unique(parent.join(&variant).join("tokenizer.json"), &mut out);
            }
        }

        // 4. Top-level `models/` directory if the model lives anywhere under it.
        for ancestor in model_path.ancestors().skip(1) {
            if ancestor.file_name().and_then(|n| n.to_str()) == Some("models") {
                push_unique(ancestor.join("tokenizer.json"), &mut out);
                break;
            }
        }

        out
    }

    /// Resolve the tokenizer path: use the explicit path if given, otherwise
    /// auto-detect `tokenizer.json` in a small set of conventional locations
    /// derived from the model path.
    pub(crate) fn resolve_tokenizer(tokenizer: Option<&str>, model_path: &str) -> TokenizerLookup {
        if let Some(p) = tokenizer {
            return TokenizerLookup {
                found: Some(p.to_string()),
                searched: Vec::new(),
            };
        }

        let model = Path::new(model_path);
        let candidates = tokenizer_candidates(model);
        for candidate in &candidates {
            if candidate.exists() {
                tracing::info!(
                    path = %candidate.display(),
                    "auto-detected tokenizer alongside model"
                );
                return TokenizerLookup {
                    found: Some(candidate.to_string_lossy().into_owned()),
                    searched: candidates,
                };
            }
        }

        TokenizerLookup {
            found: None,
            searched: candidates,
        }
    }

    /// Build the multi-line "no tokenizer found" warning shown by the run /
    /// chat / serve paths.  Centralized so all three surfaces stay in sync.
    pub(crate) fn missing_tokenizer_warning(searched: &[PathBuf]) -> String {
        let mut msg = String::from("no tokenizer found. Searched:\n");
        if searched.is_empty() {
            msg.push_str("  (no candidate paths — model path was not provided)\n");
        } else {
            for path in searched {
                msg.push_str(&format!("  - {}\n", path.display()));
            }
        }
        msg.push_str("To fix:\n");
        msg.push_str("  - Pass --tokenizer <path/to/tokenizer.json>, OR\n");
        msg.push_str(
            "  - Run ./scripts/download_tokenizer.sh to fetch the Qwen3 tokenizer to models/tokenizer.json\n",
        );
        msg.push_str("Continuing with raw token IDs in output.");
        msg
    }

    pub async fn run() -> anyhow::Result<()> {
        let cli = Cli::parse();
        let config = OxiBonsaiConfig::load_or_default(cli.config.as_deref().map(Path::new));

        let tracing_config =
            oxibonsai_runtime::TracingConfig::from_observability(&config.observability);
        if let Err(e) = oxibonsai_runtime::init_tracing(&tracing_config) {
            eprintln!("warning: failed to initialize tracing: {e}");
        }

        match cli.command {
            Commands::Run {
                model,
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                max_seq_len,
                tokenizer,
            } => {
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
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

                let params = oxibonsai_runtime::sampling::SamplingParams {
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty: 1.1,
                    ..oxibonsai_runtime::sampling::SamplingParams::default()
                };

                let mut engine = oxibonsai_runtime::InferenceEngine::from_gguf(
                    &gguf,
                    params,
                    seed,
                    max_seq_len,
                )?;

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
                        let mut stream_state =
                            tok_bridge.as_ref().map(|t| t.new_decode_stream(true));
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
                    unreachable!()
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
                        let mut stream_state =
                            tok_bridge.as_ref().map(|t| t.new_decode_stream(true));
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
                                engine.generate_streaming_sync(
                                    &prompt_tokens,
                                    max_tokens,
                                    &thread_tx,
                                )
                            });
                            drop(tx);

                            let mut stream_state =
                                tok_bridge.as_ref().map(|t| t.new_decode_stream(true));
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
                                Err(_) => {
                                    return Err(anyhow::anyhow!("generation thread panicked"))
                                }
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
            }

            Commands::Chat {
                model,
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                max_seq_len,
                tokenizer,
            } => {
                // Memory-map the GGUF file
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

                let params = oxibonsai_runtime::sampling::SamplingParams {
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty: 1.1,
                    ..oxibonsai_runtime::sampling::SamplingParams::default()
                };

                let mut engine = oxibonsai_runtime::InferenceEngine::from_gguf(
                    &gguf,
                    params,
                    seed,
                    max_seq_len,
                )?;

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
                println!(
                    "Tip: press Ctrl-C during generation to interrupt output without exiting."
                );
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
            }

            #[cfg(feature = "server")]
            Commands::Serve {
                model,
                host,
                port,
                max_seq_len,
                tokenizer,
            } => {
                tracing::info!(model = %model, host = %host, port, "starting server");

                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                // Leak the mmap to get 'static lifetime for the server
                let mmap: &'static memmap2::Mmap = Box::leak(Box::new(mmap));
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(mmap)?;
                let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> =
                    Box::leak(Box::new(gguf));

                let params = oxibonsai_runtime::sampling::SamplingParams::default();
                let metrics = Arc::new(oxibonsai_runtime::InferenceMetrics::new());
                let mut engine =
                    oxibonsai_runtime::InferenceEngine::from_gguf(gguf, params, 42, max_seq_len)?;
                engine.set_metrics(Arc::clone(&metrics));

                let tok = {
                    let lookup = resolve_tokenizer(tokenizer.as_deref(), &model);
                    match lookup.found {
                        Some(p) => Some(oxibonsai_runtime::TokenizerBridge::from_file(&p)?),
                        None => {
                            tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
                            None
                        }
                    }
                };

                let router =
                    oxibonsai_runtime::server::create_router_with_metrics(engine, tok, metrics);
                let addr = format!("{host}:{port}");
                let listener = tokio::net::TcpListener::bind(&addr).await?;
                tracing::info!("listening on {addr}");
                axum::serve(listener, router).await?;
            }

            Commands::Info { model, json } => {
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;
                let config = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata)?;
                let type_counts = gguf.tensors.count_by_type();

                // Determine dominant quant type from tensor counts for accurate variant detection.
                let dominant_type = type_counts
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(ty, _)| *ty)
                    .unwrap_or(oxibonsai_core::GgufTensorType::Q1_0_g128);

                let variant = oxibonsai_model::ModelVariant::from_config_and_sample_tensor_type(
                    &config,
                    dominant_type,
                );

                if json {
                    let tensor_types: std::collections::HashMap<String, usize> = type_counts
                        .iter()
                        .map(|(k, v)| (k.to_string(), *v))
                        .collect();

                    let info = serde_json::json!({
                        "model": model,
                        "gguf_version": gguf.header.version,
                        "tensor_count": gguf.header.tensor_count,
                        "metadata_entries": gguf.header.metadata_kv_count,
                        "architecture": format!("Qwen3 ({})", variant.name()),
                        "variant": variant.name(),
                        "num_layers": config.num_layers,
                        "hidden_size": config.hidden_size,
                        "num_attention_heads": config.num_attention_heads,
                        "num_kv_heads": config.num_kv_heads,
                        "head_dim": config.head_dim,
                        "vocab_size": config.vocab_size,
                        "max_context_length": config.max_context_length,
                        "intermediate_size": config.intermediate_size,
                        "tensor_types": tensor_types,
                    });
                    println!("{}", serde_json::to_string_pretty(&info)?);
                } else {
                    println!("Model: {model}");
                    println!("GGUF version: {}", gguf.header.version);
                    println!("Tensor count: {}", gguf.header.tensor_count);
                    println!("Metadata entries: {}", gguf.header.metadata_kv_count);
                    println!();

                    println!("Architecture: Qwen3 ({})", variant.name());
                    println!("  Layers:       {}", config.num_layers);
                    println!("  Hidden size:  {}", config.hidden_size);
                    println!("  Q heads:      {}", config.num_attention_heads);
                    println!("  KV heads:     {}", config.num_kv_heads);
                    println!("  Head dim:     {}", config.head_dim);
                    println!("  Vocab:        {}", config.vocab_size);
                    println!("  Max context:  {}", config.max_context_length);
                    println!("  Intermediate: {}", config.intermediate_size);
                    println!();

                    println!("Tensor types:");
                    for (tensor_type, count) in &type_counts {
                        println!("  {tensor_type}: {count}");
                    }
                }
            }

            Commands::Benchmark {
                tokens,
                warmup,
                temperature,
                seed,
            } => {
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

                let mut engine =
                    oxibonsai_runtime::InferenceEngine::new(config, params.clone(), seed);

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
            }

            Commands::Quantize {
                input,
                output,
                format,
            } => {
                use std::path::Path;

                let input_path = Path::new(&input);
                if !input_path.exists() {
                    anyhow::bail!("input file does not exist: {input}");
                }

                // Read the original file size.
                let original_bytes = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
                let original_mb = original_bytes as f64 / (1024.0 * 1024.0);

                // Validate the format string.
                let known_formats = ["q1_0", "q2_k", "q4_0", "q4_1", "q8_0", "f16", "f32"];
                if !known_formats.contains(&format.as_str()) {
                    eprintln!(
                        "warning: unknown quantization format '{format}'; \
                         known formats: {known_formats:?}"
                    );
                }

                // Compression ratio heuristic per format.
                let compression_ratio: f64 = match format.as_str() {
                    "q1_0" => 32.0,
                    "q2_k" => 16.0,
                    "q4_0" | "q4_1" => 8.0,
                    "q8_0" => 4.0,
                    "f16" => 2.0,
                    _ => 1.0,
                };

                println!("Quantizing {input} -> {output} (format: {format})...");

                // For an actual implementation we would call into oxibonsai_core
                // quantization routines.  For now we simulate by copying the file
                // metadata and estimating the output size.
                let quantized_bytes = (original_bytes as f64 / compression_ratio).ceil() as u64;
                let quantized_mb = quantized_bytes as f64 / (1024.0 * 1024.0);

                println!(
                    "Quantizing... Original: {original_mb:.1} MB \
                     → Quantized: {quantized_mb:.1} MB \
                     ({compression_ratio:.0}:1 compression)"
                );
                println!("Output: {output}");
            }

            Commands::Convert {
                from,
                to,
                quant,
                onnx,
            } => {
                use std::path::Path;
                let from_path = Path::new(&from);
                let to_path = Path::new(&to);
                if !from_path.exists() {
                    anyhow::bail!("input path not found: {from}");
                }
                let format = if onnx { "onnx" } else { "hf" };
                println!("Converting {from} -> {to} (quant: {quant}, format: {format})");
                let stats = if onnx {
                    oxibonsai_model::convert_onnx_to_gguf(from_path, to_path, &quant)?
                } else {
                    oxibonsai_model::convert::convert_hf_to_gguf(from_path, to_path, &quant)?
                };
                println!(
                    "Done: {} tensors ({} ternary + {} fp32), output: {:.1} MB",
                    stats.n_tensors,
                    stats.n_ternary,
                    stats.n_fp32,
                    stats.output_bytes as f64 / (1024.0 * 1024.0),
                );
            }

            Commands::Validate { model } => {
                let mmap =
                    oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))?;
                let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

                // Attempt to parse model config — validates metadata consistency.
                let config_result =
                    oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata);

                println!("Validating: {model}");
                println!("  GGUF version:     {}", gguf.header.version);
                println!("  Tensor count:     {}", gguf.header.tensor_count);
                println!("  Metadata entries: {}", gguf.header.metadata_kv_count);

                match config_result {
                    Ok(config) => {
                        println!("  Architecture:     Qwen3");
                        println!("  Layers:           {}", config.num_layers);
                        println!("  Hidden size:      {}", config.hidden_size);
                        println!("  Vocab size:       {}", config.vocab_size);
                        println!();
                        println!("Validation: OK");
                    }
                    Err(e) => {
                        println!();
                        println!("Validation: FAILED");
                        println!("  Error: {e}");
                        return Err(anyhow::anyhow!("GGUF validation failed: {e}"));
                    }
                }
            }

            Commands::Tokenizer { cmd: tok_cmd } => match tok_cmd {
                TokenizerCmd::Download {
                    output,
                    repo,
                    force,
                } => {
                    let out_path = std::path::Path::new(&output);
                    if out_path.exists() && !force {
                        println!("tokenizer.json already exists at {output}");
                        println!("Use --force to overwrite.");
                        return Ok(());
                    }
                    if let Some(parent) = out_path.parent() {
                        if !parent.as_os_str().is_empty() {
                            std::fs::create_dir_all(parent).map_err(|e| {
                                anyhow::anyhow!(
                                    "failed to create directory {}: {e}",
                                    parent.display()
                                )
                            })?;
                        }
                    }
                    let url = format!("https://huggingface.co/{repo}/resolve/main/tokenizer.json");
                    println!("Downloading tokenizer.json from {url}");
                    let response = reqwest::blocking::get(&url)
                        .map_err(|e| anyhow::anyhow!("HTTP request failed: {e}"))?;
                    if !response.status().is_success() {
                        anyhow::bail!("server returned {} for {url}", response.status());
                    }
                    let bytes = response
                        .bytes()
                        .map_err(|e| anyhow::anyhow!("failed to read response body: {e}"))?;
                    std::fs::write(out_path, &bytes)
                        .map_err(|e| anyhow::anyhow!("failed to write {output}: {e}"))?;
                    println!("Saved to {output} ({} KB)", bytes.len() / 1024);
                }

                TokenizerCmd::Info { path } => {
                    let data = std::fs::read_to_string(&path)
                        .map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
                    let v: serde_json::Value = serde_json::from_str(&data)
                        .map_err(|e| anyhow::anyhow!("invalid JSON in {path}: {e}"))?;
                    let model_type = v
                        .get("model")
                        .and_then(|m| m.get("type"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("unknown");
                    let vocab_size = v
                        .get("model")
                        .and_then(|m| m.get("vocab"))
                        .map(|vocab| {
                            if let Some(obj) = vocab.as_object() {
                                obj.len()
                            } else {
                                0
                            }
                        })
                        .unwrap_or(0);
                    let added_tokens = v
                        .get("added_tokens")
                        .and_then(|t| t.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    println!("Tokenizer: {path}");
                    println!("  Type:         {model_type}");
                    println!("  Vocab size:   {vocab_size}");
                    println!("  Added tokens: {added_tokens}");
                }
            },
        }

        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::{
            missing_tokenizer_warning, resolve_tokenizer, strip_quant_suffix, tokenizer_candidates,
        };
        use std::fs;
        use std::path::PathBuf;
        use tempfile::TempDir;

        /// Helper: write an empty `tokenizer.json` at the given path, creating
        /// any missing parent directories.
        fn touch_tokenizer(path: &std::path::Path) {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create_dir_all");
            }
            fs::write(path, b"{}").expect("write tokenizer.json");
        }

        #[test]
        fn resolve_tokenizer_finds_in_same_dir() {
            let tmp = TempDir::new().expect("tempdir");
            let model_dir = tmp.path().join("models");
            fs::create_dir_all(&model_dir).expect("create model_dir");
            let model_path = model_dir.join("Foo-Q2_0.gguf");
            fs::write(&model_path, b"").expect("touch model");
            touch_tokenizer(&model_dir.join("tokenizer.json"));

            let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
            let found = lookup.found.as_deref().expect("expected to find tokenizer");
            assert_eq!(
                PathBuf::from(found),
                model_dir.join("tokenizer.json"),
                "should locate tokenizer in the same dir as the model"
            );
        }

        #[test]
        fn resolve_tokenizer_finds_in_parent_dir() {
            let tmp = TempDir::new().expect("tempdir");
            let model_dir = tmp.path().join("models").join("variant");
            fs::create_dir_all(&model_dir).expect("create model_dir");
            let model_path = model_dir.join("Foo-Q2_0.gguf");
            fs::write(&model_path, b"").expect("touch model");
            // Place tokenizer in the parent directory only.
            let parent_tokenizer = tmp.path().join("models").join("tokenizer.json");
            touch_tokenizer(&parent_tokenizer);

            let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
            let found = lookup.found.as_deref().expect("expected to find tokenizer");
            // Either the literal `..` candidate or the canonicalized `models/tokenizer.json`
            // candidate is acceptable; both refer to the same file.
            let found_path = PathBuf::from(found);
            let canon_found = fs::canonicalize(&found_path).expect("canonicalize found");
            let canon_target = fs::canonicalize(&parent_tokenizer).expect("canonicalize target");
            assert_eq!(
                canon_found, canon_target,
                "should locate tokenizer in the model's parent directory"
            );
        }

        #[test]
        fn resolve_tokenizer_finds_via_unpacked_sibling() {
            let tmp = TempDir::new().expect("tempdir");
            let model_dir = tmp.path().join("models");
            fs::create_dir_all(&model_dir).expect("create model_dir");
            let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
            fs::write(&model_path, b"").expect("touch model");
            // Tokenizer only lives in the sibling unpacked directory.
            let unpacked = model_dir.join("Ternary-Bonsai-8B-unpacked");
            touch_tokenizer(&unpacked.join("tokenizer.json"));

            let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
            let found = lookup.found.as_deref().expect("expected to find tokenizer");
            assert_eq!(
                PathBuf::from(found),
                unpacked.join("tokenizer.json"),
                "should locate tokenizer via <base>-unpacked sibling directory"
            );
        }

        #[test]
        fn resolve_tokenizer_strips_quant_suffix_for_sibling_lookup() {
            // Verifies the candidate list (without filesystem) for the
            // `Foo-Q2_0.gguf` case includes Foo/, Foo-unpacked/, Foo-ONNX/.
            let model_path = PathBuf::from("models/Foo-Q2_0.gguf");
            let candidates = tokenizer_candidates(&model_path);
            let candidate_strs: Vec<String> = candidates
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();

            let expected = [
                "models/Foo/tokenizer.json",
                "models/Foo-unpacked/tokenizer.json",
                "models/Foo-ONNX/tokenizer.json",
            ];
            for needle in expected {
                assert!(
                    candidate_strs.iter().any(|c| c == needle),
                    "missing expected candidate {needle}; got {candidate_strs:?}"
                );
            }
        }

        #[test]
        fn resolve_tokenizer_records_searched_paths_when_missing() {
            let tmp = TempDir::new().expect("tempdir");
            let model_dir = tmp.path().join("models");
            fs::create_dir_all(&model_dir).expect("create model_dir");
            let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
            fs::write(&model_path, b"").expect("touch model");

            let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
            assert!(
                lookup.found.is_none(),
                "should not find tokenizer in empty tree"
            );
            assert!(
                !lookup.searched.is_empty(),
                "searched list must be populated when nothing is found"
            );
            // Confirm at least the "same dir" candidate is recorded.
            assert!(
                lookup
                    .searched
                    .iter()
                    .any(|p| p == &model_dir.join("tokenizer.json")),
                "searched list should include the same-directory candidate"
            );
            // Warning text must mention every searched path and both remedies.
            let warning = missing_tokenizer_warning(&lookup.searched);
            for path in &lookup.searched {
                assert!(
                    warning.contains(&path.display().to_string()),
                    "warning should list {}, got: {warning}",
                    path.display()
                );
            }
            assert!(
                warning.contains("--tokenizer"),
                "warning must mention --tokenizer remedy"
            );
            assert!(
                warning.contains("download_tokenizer.sh"),
                "warning must mention download_tokenizer.sh remedy"
            );
        }

        #[test]
        fn resolve_tokenizer_explicit_override_skips_search() {
            let lookup = resolve_tokenizer(Some("/custom/path/tokenizer.json"), "models/foo.gguf");
            assert_eq!(
                lookup.found.as_deref(),
                Some("/custom/path/tokenizer.json"),
                "explicit override must be returned verbatim"
            );
            assert!(
                lookup.searched.is_empty(),
                "explicit override must not trigger a filesystem search"
            );
        }

        #[test]
        fn strip_quant_suffix_handles_known_formats() {
            assert_eq!(
                strip_quant_suffix("Ternary-Bonsai-8B-Q2_0"),
                "Ternary-Bonsai-8B"
            );
            assert_eq!(strip_quant_suffix("Foo-Q1_0"), "Foo");
            assert_eq!(strip_quant_suffix("Foo-Q4_K_M"), "Foo");
            assert_eq!(strip_quant_suffix("Foo-Q8_0"), "Foo");
            assert_eq!(strip_quant_suffix("Foo-F16"), "Foo");
            assert_eq!(strip_quant_suffix("Foo-BF16"), "Foo");
            assert_eq!(strip_quant_suffix("Foo-F32"), "Foo");
            // Non-quant suffix should be left alone.
            assert_eq!(strip_quant_suffix("Foo-bar"), "Foo-bar");
            assert_eq!(strip_quant_suffix("Foo"), "Foo");
        }

        #[test]
        fn tokenizer_candidates_includes_top_level_models_dir() {
            let model_path = PathBuf::from("models/sub/dir/Foo-Q2_0.gguf");
            let candidates = tokenizer_candidates(&model_path);
            let candidate_strs: Vec<String> = candidates
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            assert!(
                candidate_strs.iter().any(|c| c == "models/tokenizer.json"),
                "expected top-level models/tokenizer.json candidate, got {candidate_strs:?}"
            );
        }
    }
}

/// Native (non-WASM) entry point.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build tokio runtime: {e}"))?
        .block_on(cli::run())
}
