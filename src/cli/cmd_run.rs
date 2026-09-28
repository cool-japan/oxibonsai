//! `oxibonsai run` — single-shot inference on a GGUF model.

use std::io::{self, Write};

use super::model_desc;
use super::tokenizer_backend::{self, TokenizerBackendChoice};
use super::util::{
    build_sampling_params, check_tokenizer_model_compatibility, missing_tokenizer_warning,
    model_vocab_size, read_prompt_stdin, reject_penalties_with_constrained_decode,
    resolve_tokenizer_vocab_aware, StopChecker, TokenizerLookup,
};

/// Resolved arguments for `oxibonsai run`, merged from CLI flags and
/// `--config` in `mod.rs` (cli-04).
///
/// A struct rather than 15+ positional parameters: past a certain field
/// count, position-based argument passing between `args.rs`'s
/// destructuring, `mod.rs`'s forwarding call, and this function's
/// signature becomes error-prone to keep in sync by hand — a struct with
/// named fields lets the compiler catch a mismatch instead of silently
/// swapping two `f32` parameters.
pub(crate) struct RunArgs {
    pub(crate) model: Option<String>,
    pub(crate) prompt: String,
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
    pub(crate) no_stream: bool,
}

pub(crate) fn run(args: RunArgs) -> anyhow::Result<()> {
    let RunArgs {
        model,
        prompt,
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
        no_stream,
    } = args;

    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;

    let prompt_text = if prompt == "-" {
        read_prompt_stdin()?
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

    // cli-12: `temperature`/`top_p`/`repetition_penalty`/`max_tokens` are
    // already range-checked by clap's `value_parser`s in args.rs; this is
    // the shared constructor (orchestrator P0 addendum) so `--temperature
    // 0` means exactly argmax on every backend — `repetition_penalty`
    // defaults to 1.0 here (mod.rs resolves it that way), never the
    // `SamplingParams::default()` struct's own 1.1.
    let params = build_sampling_params(temperature, top_k, top_p, repetition_penalty);

    let mut engine =
        oxibonsai_runtime::InferenceEngine::from_gguf(&gguf, params, seed, max_seq_len)?;
    engine.set_penalties(oxibonsai_runtime::PenaltyParams::new(
        frequency_penalty,
        presence_penalty,
    ));

    // cli-16: report the RESOLVED quant variant + effective kernel tier,
    // never a hardcoded kernel-family string.
    if let Some(hybrid) = engine.hybrid_model() {
        // ENGINE-SEAM: a hybrid (`qwen35`) engine resolved its own variant
        // and weight quantization at load; the dense tensor-count heuristic
        // below would misname it.
        let variant = hybrid.variant().map_or_else(
            || engine.architecture().to_string(),
            |v| v.name().to_string(),
        );
        eprintln!(
            "{}",
            model_desc::resolved_engine_summary(
                &variant,
                engine.dominant_quant_type(),
                engine.kernel_tier(),
                &engine.effective_tier_reason(),
            )
        );
    } else if let Ok(config) = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata) {
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

    // Tokenize prompt and retain the bridge for streaming decode.
    // TOK-08: resolution prefers a vocab-matching auto-detected candidate,
    // and (below) the tokenizer is hard-checked against the model
    // regardless of whether it was auto-detected or passed explicitly.
    // ENGINE-SEAM: a GGUF that embeds its own tokenizer (every Bonsai 2
    // `qwen35` file) uses it when no on-disk candidate fits the model.
    let expected_vocab = model_vocab_size(&gguf).ok();
    let lookup = resolve_tokenizer_vocab_aware(tokenizer.as_deref(), &model, expected_vocab);
    let resolved = resolve_model_tokenizer(
        tokenizer.as_deref(),
        &lookup,
        &gguf,
        expected_vocab,
        tokenizer_backend,
        allow_vocab_mismatch,
    )?;
    let (prompt_tokens, tok_bridge) = if let Some(tok) = resolved {
        let tokens = tok.encode(&prompt_text)?;
        (tokens, Some(tok))
    } else {
        tracing::warn!("{}", missing_tokenizer_warning(&lookup.searched));
        // cli-07: no hardcoded token id. Fall back to the model's own
        // declared BOS id when present; otherwise there is no honest
        // prompt token to emit at all (checked below, before any forward
        // pass runs).
        let bos = gguf
            .metadata
            .get_u32(oxibonsai_core::gguf::tensor_info::keys::TOKENIZER_BOS_TOKEN_ID)
            .ok();
        (bos.map(|id| vec![id]).unwrap_or_default(), None)
    };
    if prompt_tokens.is_empty() {
        anyhow::bail!(
            "cannot start generation: no tokenizer was found and the model declares no BOS \
             token id, so there is no way to honestly tokenize the prompt. Pass --tokenizer \
             <path/to/tokenizer.json>."
        );
    }

    if grammar.is_some() && tok_bridge.is_none() {
        anyhow::bail!("--grammar requires a tokenizer; none was found (see the warning above)");
    }

    let use_constrained_or_stop = grammar.is_some() || !stop.is_empty();
    reject_penalties_with_constrained_decode(
        use_constrained_or_stop,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
    )?;

    tracing::info!(prompt_tokens = prompt_tokens.len(), "prefilling");

    let start = std::time::Instant::now();

    let (prompt_len, output_count) = if use_constrained_or_stop {
        run_constrained_or_stopped(
            &mut engine,
            &prompt_tokens,
            max_tokens,
            grammar.as_deref(),
            &stop,
            tok_bridge.as_ref(),
            seed,
            temperature,
            top_k,
            top_p,
        )?
    } else {
        run_fast_path(
            &mut engine,
            &prompt_tokens,
            max_tokens,
            temperature,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            tok_bridge.as_ref(),
            no_stream,
        )?
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

/// The tokenizer a model-loading subcommand uses (TOK-08 + ENGINE-SEAM).
///
/// An explicit `--tokenizer`, or an auto-detected `tokenizer.json` that
/// passes [`check_tokenizer_model_compatibility`], is used exactly as before.
/// When no on-disk candidate exists, or an *auto-detected* one does not fit
/// the model, a tokenizer embedded in the GGUF itself (`tokenizer.ggml.*`)
/// whose vocabulary matches the model is used instead: every Bonsai 2
/// `qwen35` file carries its `pre = qwen35` tokenizer, while the repository's
/// `models/tokenizer.json` is the legacy Qwen3 one, so without this fallback a
/// Bonsai 2 run either failed the compatibility check or mis-tokenized. An
/// explicit `--tokenizer` that does not fit still fails loudly.
///
/// Returns `Ok(None)` when there is no usable tokenizer at all (the caller's
/// existing "no tokenizer" handling applies).
pub(crate) fn resolve_model_tokenizer(
    explicit: Option<&str>,
    lookup: &TokenizerLookup,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
    backend: TokenizerBackendChoice,
    allow_vocab_mismatch: bool,
) -> anyhow::Result<Option<oxibonsai_runtime::TokenizerBridge>> {
    resolve_model_tokenizer_with(
        explicit,
        lookup,
        gguf,
        expected_vocab,
        allow_vocab_mismatch,
        |path| tokenizer_backend::load_tokenizer_bridge(path, backend),
    )
}

/// [`resolve_model_tokenizer`] with the caller's own loader for the on-disk
/// candidate (`benchmark` loads through `TokenizerBridge::from_file`).
pub(crate) fn resolve_model_tokenizer_with(
    explicit: Option<&str>,
    lookup: &TokenizerLookup,
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
    allow_vocab_mismatch: bool,
    load: impl FnOnce(&str) -> anyhow::Result<oxibonsai_runtime::TokenizerBridge>,
) -> anyhow::Result<Option<oxibonsai_runtime::TokenizerBridge>> {
    let Some(path) = &lookup.found else {
        return Ok(gguf_embedded_tokenizer(gguf, expected_vocab));
    };
    let tok = load(path)?;
    match check_tokenizer_model_compatibility(&tok, path, gguf, allow_vocab_mismatch) {
        Ok(()) => Ok(Some(tok)),
        Err(mismatch) if explicit.is_none() => {
            match gguf_embedded_tokenizer(gguf, expected_vocab) {
                Some(embedded) => {
                    tracing::info!(
                        skipped = %path,
                        reason = %mismatch,
                        "the auto-detected tokenizer does not fit this model; using the tokenizer \
                         embedded in the GGUF instead"
                    );
                    Ok(Some(embedded))
                }
                None => Err(mismatch),
            }
        }
        Err(mismatch) => Err(mismatch),
    }
}

/// The tokenizer embedded in `gguf`'s `tokenizer.ggml.*` metadata, when the
/// file carries one whose vocabulary equals the model's (`expected_vocab`).
pub(crate) fn gguf_embedded_tokenizer(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    expected_vocab: Option<usize>,
) -> Option<oxibonsai_runtime::TokenizerBridge> {
    let expected = expected_vocab?;
    match oxibonsai_runtime::engine::tokenizer_from_gguf(gguf) {
        Ok(tok) if tok.vocab_size() == expected => {
            tracing::info!(vocab = expected, "using the tokenizer embedded in the GGUF");
            Some(tok)
        }
        Ok(tok) => {
            tracing::debug!(
                embedded_vocab = tok.vocab_size(),
                model_vocab = expected,
                "the GGUF-embedded tokenizer does not match the model's vocabulary; not used"
            );
            None
        }
        Err(e) => {
            tracing::debug!(error = %e, "no usable GGUF-embedded tokenizer");
            None
        }
    }
}

/// The original three-way fast path (greedy-GPU / native-CUDA /
/// worker-thread streaming), used whenever neither `--grammar` nor
/// `--stop` is requested.
///
/// F14: the native-CUDA arm used to call `engine.generate()` to
/// completion and print only afterward. The whole point of collapsing it
/// into the same `cfg(not(...))` branch as every other non-greedy-GPU
/// platform is that the cfg ladder must be re-derived as a single unit —
/// the Metal greedy-GPU branch above it ends in an `unreachable!()`
/// guarded by the *complement* of this branch's old condition, so editing
/// one arm in isolation would silently change which cfg combinations
/// reach that `unreachable!()`. `--no-stream` restores the old
/// "wait for completion, print once" behavior as an explicit opt-in
/// (useful for non-interactive benchmarking) instead of a silent,
/// platform-dependent default.
#[allow(clippy::too_many_arguments)]
fn run_fast_path(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    tok_bridge: Option<&oxibonsai_runtime::TokenizerBridge>,
    no_stream: bool,
) -> anyhow::Result<(usize, usize)> {
    // Greedy GPU path: when temperature=0 and Metal is available,
    // run argmax on GPU and download only 4-byte token IDs instead
    // of the full ~607KB logits vector per token.
    //
    // `generate_greedy_gpu` is pure argmax: it never consults the
    // engine's sampler, so it cannot apply a repetition/frequency/presence
    // penalty even though `engine.set_penalties(...)` was called above.
    // Gating on `temperature == 0.0` alone (the original condition, from
    // back when `repetition_penalty` was hardcoded 1.1 and "temperature 0"
    // was never truly argmax anyway) would silently drop any penalty this
    // orchestrator P0 addendum's own new flags request the moment they are
    // combined with `--temperature 0` on a Metal build — exactly the class
    // of divergence that addendum exists to close. Only the fully
    // penalty-free case takes this path; any explicit penalty falls
    // through to the streaming path below, whose sampler does apply them.
    //
    // NOTE on cli-09 (wave-1 addendum item 4; STILL BLOCKED, verifier
    // re-confirmed this on the previous pass): genuinely not closable
    // from this file alone, not merely deferred. `InferenceEngine::
    // generate_greedy_gpu` is itself defined only under
    // `#[cfg(all(feature = "metal", target_os = "macos"))]` in
    // `oxibonsai-runtime` (verified in `engine.rs`), and this crate's own
    // `metal` feature (root `Cargo.toml`, NOT this crate's `owned_files`
    // this wave — verified: `B2-08`'s `owned_files` list includes it,
    // per its `scratchpad/pkg/B2-08.json`) is off by default
    // (`default = ["server", "hf-tokenizer"]`). Dropping `feature =
    // "metal"` from the 5 remaining `#[cfg(all(feature = "metal",
    // target_os = "macos"))]` sites in this file (this one, and the ones
    // at the `use_greedy_gpu` binding and its two `#[cfg]` arms just
    // below) WITHOUT the manifest change landing FIRST breaks a plain
    // `cargo install oxibonsai-cli` / default `cargo build` on macOS
    // outright (this package's own gate uses `--all-features` and would
    // not catch that regression).
    //
    // The exact two-sided patch, for whoever next holds root `Cargo.toml`
    // edit rights this wave (B2-08) plus this file, to land IN THE SAME
    // integration step:
    //   1. Root `Cargo.toml`, add:
    //        [target.'cfg(target_os = "macos")'.dependencies]
    //        oxibonsai-kernels = { workspace = true, features = ["metal", "gpu"] }
    //        oxibonsai-model = { workspace = true, features = ["metal"] }
    //        oxibonsai-runtime = { workspace = true, features = ["metal"] }
    //        oxibonsai-image = { workspace = true, features = ["metal"] }
    //      (Cargo unifies features per-target when the same crate also
    //      appears in the unconditional `[dependencies]` table, so this
    //      does not need to touch that table — see the Cargo reference on
    //      platform-specific dependencies. This is the only way to get a
    //      macOS-only default-on capability: Cargo cannot make a
    //      package's OWN named feature default-on per target. UNTESTED
    //      HERE: root `Cargo.toml` is not this package's to edit or build
    //      against this wave, so this feature-union behavior has not been
    //      verified against THIS workspace's actual dependency graph.
    //      Whoever applies step 1 must confirm with a real `cargo build`
    //      — no `--features`, macOS target — that `oxibonsai-kernels`'s
    //      `metal` feature is actually active before flipping step 2; if
    //      it is not, step 2 must wait.)
    //   2. This file: replace `#[cfg(all(feature = "metal", target_os =
    //      "macos"))]` with `#[cfg(target_os = "macos")]` (and the
    //      matching `#[cfg(not(...))]` arms) at every site below.
    // Not applied here because (1) is outside this package's owned_files
    // this wave; applying (2) alone, ahead of (1), is the exact
    // known-to-break-the-default-build half-fix the addendum warns
    // against. Recorded as an unresolved, cross-package-blocked item in
    // this package's `deviations`, not silently assumed closed.
    let penalty_free =
        repetition_penalty == 1.0 && frequency_penalty == 0.0 && presence_penalty == 0.0;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let use_greedy_gpu = temperature == 0.0 && penalty_free;
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let use_greedy_gpu = {
        // `temperature`/`penalty_free` only feed the argmax-fast-path
        // decision on the metal+macOS cfg arm above; read them
        // unconditionally here too so neither is ever an unused binding
        // on other platforms.
        let _ = (temperature, penalty_free);
        false
    };

    if use_greedy_gpu {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            tracing::info!("using greedy GPU path (argmax on Metal, 4-byte download)");
            let p_len = prompt_tokens.len();
            let tokens = engine.generate_greedy_gpu(prompt_tokens, max_tokens)?;
            let mut stream_state = tok_bridge.map(|t| t.new_decode_stream(true));
            for &token_id in &tokens {
                match (tok_bridge, stream_state.as_mut()) {
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
            Ok((p_len, tokens.len()))
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        unreachable!(
            "use_greedy_gpu is only ever true on the metal+macOS cfg arm; see the let-binding above"
        )
    } else if no_stream {
        // F14: `--no-stream` is now the ONLY way to get "wait for the
        // full completion, print once" behavior, on every platform
        // (including native CUDA, which used to do this unconditionally
        // and silently with no flag to opt out of it).
        let p_len = prompt_tokens.len();
        let tokens = engine.generate(prompt_tokens, max_tokens)?;
        let mut stream_state = tok_bridge.map(|t| t.new_decode_stream(true));
        for &token_id in &tokens {
            match (tok_bridge, stream_state.as_mut()) {
                (Some(tok), Some(state)) => {
                    if let Some(text) = tok.step_decode(state, token_id)? {
                        print!("{text}");
                    }
                }
                _ => {
                    print!(" {token_id}");
                }
            }
        }
        let _ = io::stdout().flush();
        Ok((p_len, tokens.len()))
    } else {
        // Every platform (CPU, Metal without the greedy-argmax fast path,
        // and native CUDA alike): stream via a worker thread so the main
        // thread can decode and print tokens as they arrive (F14).
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        let p_len = prompt_tokens.len();
        let count = std::thread::scope(|s| -> anyhow::Result<usize> {
            let thread_tx = tx.clone();
            let gen_handle = s.spawn(move || {
                engine.generate_streaming_sync(prompt_tokens, max_tokens, &thread_tx)
            });
            drop(tx);

            let mut stream_state = tok_bridge.map(|t| t.new_decode_stream(true));
            let mut count = 0usize;
            for token_id in rx {
                count += 1;
                match (tok_bridge, stream_state.as_mut()) {
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
        Ok((p_len, count))
    }
}

/// Grammar-constrained and/or stop-sequence-aware decode loop (cli-17).
///
/// Deliberately a single, portable, token-by-token loop rather than
/// threading `--grammar`/`--stop` through the fast path's
/// Metal/CUDA/worker-thread specializations: both features are opt-in and
/// already change the sampling procedure itself (masked logits for a
/// grammar; early stop on a matched sequence), so reusing one
/// straightforward implementation is far less risky than retrofitting
/// three specialized fast paths to support both. Text is buffered and
/// printed once generation ends (matched stop sequence, EOS, grammar
/// completion, or `max_tokens`) rather than streamed live: this sidesteps
/// the project's known "stop-sequence chunk-boundary leak" limitation
/// entirely (the accumulated buffer covers the whole output, so
/// `StopChecker` — the server's own hardened matcher, reused here rather
/// than a second naive implementation — never sees a sequence split
/// across print calls) at the cost of not streaming to the terminal.
///
/// Penalties (`--frequency-penalty`/`--presence-penalty`/
/// `--repetition-penalty`) are never applied in this loop, with or without
/// `--grammar`: every token is drawn via [`oxibonsai_runtime::InferenceEngine::sample`]
/// (or, under `--grammar`, the constrained sampler's own minimal
/// `SamplerChain` — temperature/top-k/top-p only), neither of which
/// consults generated-token history the way `sample_with_history` does.
/// `run`'s caller (`run::run`) refuses to reach this function at all when
/// any penalty is non-default (see
/// [`super::util::reject_penalties_with_constrained_decode`]), rather than
/// silently accepting a flag it cannot honor.
#[allow(clippy::too_many_arguments)]
fn run_constrained_or_stopped(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    grammar_path: Option<&str>,
    stop: &[String],
    tok_bridge: Option<&oxibonsai_runtime::TokenizerBridge>,
    seed: u64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> anyhow::Result<(usize, usize)> {
    engine.reset();
    let prompt_len = prompt_tokens.len();
    let mut logits = engine.prefill_from_pos(prompt_tokens, 0)?;

    let mut constrained = match grammar_path {
        Some(path) => Some(build_constrained_sampler(
            path,
            tok_bridge,
            tok_bridge.map(|t| t.vocab_size()).unwrap_or(logits.len()),
            seed,
            temperature,
            top_k,
            top_p,
        )?),
        None => None,
    };

    let stop_checker = StopChecker::new(stop.to_vec());
    let mut accumulated = String::new();
    let mut stream_state = tok_bridge.map(|t| t.new_decode_stream(true));
    let mut generated = 0usize;
    let mut pos = prompt_len;

    while generated < max_tokens {
        if logits.is_empty() {
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
        match (tok_bridge, stream_state.as_mut()) {
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
    println!();

    Ok((prompt_len, generated))
}

/// Load a grammar file: `.gbnf` is parsed as GBNF, any other extension is
/// compiled as a JSON Schema.
///
/// `pub(crate)` (not `fn`-private): [`super::cmd_chat`] shares this and
/// [`build_constrained_sampler_from_grammar`] rather than duplicating
/// grammar loading, since a chat session needs to load the grammar file
/// once but build a fresh [`oxibonsai_runtime::ConstrainedSampler`] (fresh
/// recognizer state) for every turn.
pub(crate) fn load_grammar(path: &str) -> anyhow::Result<oxibonsai_runtime::Grammar> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read grammar file '{path}': {e}"))?;
    if path.ends_with(".gbnf") {
        oxibonsai_runtime::parse_gbnf(&content)
            .map_err(|e| anyhow::anyhow!("failed to parse GBNF grammar '{path}': {e}"))
    } else {
        oxibonsai_runtime::compile_json_schema_str(&content)
            .map_err(|e| anyhow::anyhow!("failed to compile JSON Schema grammar '{path}': {e}"))
    }
}

/// Build a [`oxibonsai_runtime::ConstrainedSampler`] enforcing `grammar_path`
/// (single-shot convenience: loads the file and builds the sampler in one
/// call, for `run`'s one-generation-per-process use).
#[allow(clippy::too_many_arguments)]
fn build_constrained_sampler(
    grammar_path: &str,
    tok: Option<&oxibonsai_runtime::TokenizerBridge>,
    vocab_size: usize,
    seed: u64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> anyhow::Result<oxibonsai_runtime::ConstrainedSampler> {
    let grammar = load_grammar(grammar_path)?;
    Ok(build_constrained_sampler_from_grammar(
        grammar,
        tok,
        vocab_size,
        seed,
        temperature,
        top_k,
        top_p,
    ))
}

/// Build a [`oxibonsai_runtime::ConstrainedSampler`] from an already-loaded
/// [`oxibonsai_runtime::Grammar`] — the per-turn half `chat` uses, so a
/// multi-turn session parses the grammar file only once.
///
/// Note this still pays `GrammarConstraint::new`'s eager "decode every id
/// in `0..vocab_size`" cost on every call (fresh recognizer state is
/// required per turn regardless): a real per-session cache of that decode
/// table would need `GrammarConstraint` to expose a way to reset its
/// recognizer without rebuilding the whole constraint, which it does not
/// today. Correct, not maximally optimized — acceptable for an
/// already-opt-in advanced feature.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_constrained_sampler_from_grammar(
    grammar: oxibonsai_runtime::Grammar,
    tok: Option<&oxibonsai_runtime::TokenizerBridge>,
    vocab_size: usize,
    seed: u64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> oxibonsai_runtime::ConstrainedSampler {
    // `GrammarConstraint::new` eagerly decodes every id in `0..vocab_size`
    // exactly once at construction, so precomputing an owned lookup table
    // up front (rather than capturing `tok` by reference in the closure)
    // both satisfies the `Send + Sync + 'static` bound the constraint
    // requires and avoids re-decoding on every `allowed_tokens` call.
    let id_to_bytes: Vec<Vec<u8>> = (0..vocab_size as u32)
        .map(|id| match tok {
            Some(t) => t
                .inner()
                .id_to_token(id)
                .map(String::into_bytes)
                .unwrap_or_default(),
            None => Vec::new(),
        })
        .collect();
    let decode_fn =
        move |id: u32| -> Vec<u8> { id_to_bytes.get(id as usize).cloned().unwrap_or_default() };
    let constraint = oxibonsai_runtime::GrammarConstraint::new(grammar, decode_fn, vocab_size);

    let mut chain = oxibonsai_runtime::SamplerChain::new(seed);
    chain = if temperature <= 0.0 {
        chain.add(oxibonsai_runtime::SamplerStep::Greedy)
    } else {
        chain = chain.add(oxibonsai_runtime::SamplerStep::Temperature(temperature));
        if top_k > 0 {
            chain = chain.add(oxibonsai_runtime::SamplerStep::TopK(top_k));
        }
        if top_p < 1.0 {
            chain = chain.add(oxibonsai_runtime::SamplerStep::TopP(top_p));
        }
        chain
    };

    oxibonsai_runtime::ConstrainedSampler::new(chain, Box::new(constraint), vocab_size)
}
