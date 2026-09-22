//! `oxibonsai` CLI: argument parsing and subcommand dispatch.
//!
//! [`args`] holds the `clap` definitions; [`util`] holds helpers shared
//! across subcommands (tokenizer auto-detection, GGUF tensor
//! dequantization, `--config` merging); each `cmd_*` module implements one
//! subcommand's body. `main.rs` parses argv (before building the tokio
//! runtime — see [`apply_pre_runtime_env`]) and calls [`run_with`], which
//! stays a thin dispatch point: config loading and the flag/config-file
//! merge for each subcommand's arguments happen here, in one place, so
//! every subcommand applies `--config` the same way (cli-04).

use std::path::Path;

use oxibonsai_runtime::OxiBonsaiConfig;

#[cfg(feature = "server")]
mod admission;
mod model_desc;
mod repl;
mod term;
mod tokenizer_backend;

mod args;
mod util;

mod cmd_benchmark;
mod cmd_chat;
mod cmd_convert;
#[cfg(feature = "eval")]
mod cmd_eval;
mod cmd_image;
mod cmd_info;
mod cmd_quantize;
mod cmd_run;
#[cfg(feature = "server")]
mod cmd_serve;
mod cmd_tokenizer;
mod cmd_validate;

pub(crate) use args::Cli;
use args::Commands;
use util::RawTomlSections;

/// cli-24: `oxibonsai repl --cpu-te`-less invocations set `OXI_TE_GPU=1`
/// so the text encoder's GPU dispatch `OnceCell` latch reads it on the
/// first forward pass. `std::env::set_var` is only actually safe to call
/// when no other thread can be concurrently reading the environment (it
/// becomes an `unsafe fn` outright under edition 2024) — which is true
/// here, in `main()`, before the tokio runtime (and therefore any worker
/// thread) has been built, but was NOT reliably true when this used to
/// run from inside `repl::run`, called from async code already running on
/// a multi-threaded tokio runtime. Call this before building that
/// runtime, on the single main thread, with the same [`Cli`] then passed
/// to [`run_with`].
pub fn apply_pre_runtime_env(cli: &Cli) {
    if let Commands::Repl { cpu_te, .. } = &cli.command {
        if !cpu_te && std::env::var_os("OXI_TE_GPU").is_none() {
            // SAFETY: called from `main()` before the tokio runtime (and
            // therefore any other thread) is built, so no concurrent
            // reader of the environment can exist yet.
            std::env::set_var("OXI_TE_GPU", "1");
        }
    }
}

/// Convert one of `args.rs`'s `validate_*`/`parse_*` `Result<T, String>`
/// outcomes into `anyhow::Result<T>`.
///
/// cli-12's range checks live in `args.rs` as clap `value_parser`s, which
/// only ever see a value that arrived as a `--flag`. A value resolved from
/// `[sampling]`/`[model]` in `--config` (cli-04) is parsed by this
/// module's own `toml_f32`/`toml_usize` — plain `str::parse`, with none of
/// those range checks — so every sampling/length value resolved below is
/// re-validated here regardless of which side (flag or config) it came
/// from, closing the gap a config file would otherwise have through a
/// different door than the one cli-12 closed on `--flag`.
fn validated<T>(result: Result<T, String>) -> anyhow::Result<T> {
    result.map_err(|e| anyhow::anyhow!(e))
}

/// Dispatch an already-parsed [`Cli`]. See [`apply_pre_runtime_env`] for
/// why `main.rs` parses argv itself, before building the tokio runtime,
/// instead of this function parsing argv internally.
pub async fn run_with(cli: Cli) -> anyhow::Result<()> {
    // ── --config (cli-04): load every section, error on a bad path,
    // reject unknown keys. `sections` is empty (not missing) when no
    // --config was given, so every per-field resolver below reads `None`
    // uniformly instead of needing a separate "was --config passed" branch.
    let (config, sections): (OxiBonsaiConfig, RawTomlSections) = match cli.config.as_deref() {
        Some(path) => {
            let path = Path::new(path);
            let content = std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("failed to read config file {}: {e}", path.display())
            })?;
            let sections = util::parse_flat_toml_sections(&content, path)?;
            let config = util::load_config_strict(&content, path)?;
            (config, sections)
        }
        None => (OxiBonsaiConfig::default(), RawTomlSections::new()),
    };

    // `oxibonsai info --json` must be machine-parseable: `init_tracing`
    // installs `fmt::layer()` on the default STDOUT writer, so any WARN
    // emitted while reading the model (e.g. the GGUF reader's
    // forward-compat notice) would precede the JSON document on the same
    // stream and break every downstream `| jq`/`json.load`. `info --json`
    // never itself calls `tracing::*`, so skipping init entirely for this
    // one command is a plain, complete fix rather than a partial
    // stderr-routing workaround.
    let skip_tracing_init = matches!(&cli.command, Commands::Info { json: true, .. });
    if !skip_tracing_init {
        let tracing_config =
            oxibonsai_runtime::TracingConfig::from_observability(&config.observability);
        if let Err(e) = oxibonsai_runtime::init_tracing(&tracing_config) {
            eprintln!("warning: failed to initialize tracing: {e}");
        }
    }

    match cli.command {
        Commands::Run {
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
        } => {
            let run_args = cmd_run::RunArgs {
                model: util::resolve_str(model, &sections, "model", "model_path"),
                prompt,
                max_tokens: validated(args::validate_max_tokens(util::resolve_usize(
                    max_tokens,
                    &sections,
                    "sampling",
                    "max_tokens",
                    256,
                )))?,
                temperature: validated(args::validate_temperature(util::resolve_f32(
                    temperature,
                    &sections,
                    "sampling",
                    "temperature",
                    0.7,
                )))?,
                top_k: util::resolve_usize(top_k, &sections, "sampling", "top_k", 40),
                top_p: validated(args::validate_top_p(util::resolve_f32(
                    top_p, &sections, "sampling", "top_p", 0.9,
                )))?,
                repetition_penalty: validated(args::validate_repetition_penalty(
                    util::resolve_f32(
                        repetition_penalty,
                        &sections,
                        "sampling",
                        "repetition_penalty",
                        1.0,
                    ),
                ))?,
                frequency_penalty: validated(args::validate_openai_penalty(util::resolve_f32(
                    frequency_penalty,
                    &sections,
                    "sampling",
                    "frequency_penalty",
                    0.0,
                )))?,
                presence_penalty: validated(args::validate_openai_penalty(util::resolve_f32(
                    presence_penalty,
                    &sections,
                    "sampling",
                    "presence_penalty",
                    0.0,
                )))?,
                seed,
                max_seq_len: validated(args::validate_max_seq_len(util::resolve_usize(
                    max_seq_len,
                    &sections,
                    "model",
                    "max_seq_len",
                    4096,
                )))?,
                tokenizer: util::resolve_str(tokenizer, &sections, "model", "tokenizer_path"),
                tokenizer_backend,
                grammar,
                stop,
                allow_vocab_mismatch,
                no_stream,
            };
            cmd_run::run(run_args)?
        }

        Commands::Image {
            prompt,
            out,
            seed,
            steps,
            width,
            height,
            dit,
            vae,
            te,
            tokenizer,
        } => cmd_image::run_image(
            prompt, out, seed, steps, width, height, dit, vae, te, tokenizer,
        )?,

        Commands::Repl {
            seed,
            steps,
            width,
            height,
            cpu_te,
            dit,
            vae,
            te,
            tokenizer,
        } => cmd_image::run_repl(seed, steps, width, height, cpu_te, dit, vae, te, tokenizer)?,

        Commands::Chat {
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
        } => {
            let chat_args = cmd_chat::ChatArgs {
                model: util::resolve_str(model, &sections, "model", "model_path"),
                max_tokens: validated(args::validate_max_tokens(util::resolve_usize(
                    max_tokens,
                    &sections,
                    "sampling",
                    "max_tokens",
                    512,
                )))?,
                temperature: validated(args::validate_temperature(util::resolve_f32(
                    temperature,
                    &sections,
                    "sampling",
                    "temperature",
                    0.7,
                )))?,
                top_k: util::resolve_usize(top_k, &sections, "sampling", "top_k", 40),
                top_p: validated(args::validate_top_p(util::resolve_f32(
                    top_p, &sections, "sampling", "top_p", 0.9,
                )))?,
                repetition_penalty: validated(args::validate_repetition_penalty(
                    util::resolve_f32(
                        repetition_penalty,
                        &sections,
                        "sampling",
                        "repetition_penalty",
                        1.0,
                    ),
                ))?,
                frequency_penalty: validated(args::validate_openai_penalty(util::resolve_f32(
                    frequency_penalty,
                    &sections,
                    "sampling",
                    "frequency_penalty",
                    0.0,
                )))?,
                presence_penalty: validated(args::validate_openai_penalty(util::resolve_f32(
                    presence_penalty,
                    &sections,
                    "sampling",
                    "presence_penalty",
                    0.0,
                )))?,
                seed,
                max_seq_len: validated(args::validate_max_seq_len(util::resolve_usize(
                    max_seq_len,
                    &sections,
                    "model",
                    "max_seq_len",
                    4096,
                )))?,
                tokenizer: util::resolve_str(tokenizer, &sections, "model", "tokenizer_path"),
                tokenizer_backend,
                grammar,
                stop,
                allow_vocab_mismatch,
            };
            cmd_chat::run(chat_args)?
        }

        #[cfg(feature = "server")]
        Commands::Serve {
            model,
            host,
            port,
            max_seq_len,
            tokenizer,
            pool_size,
            bearer_token,
            max_concurrent_requests,
            request_timeout_ms,
            #[cfg(feature = "rag")]
            rag,
        } => {
            let model = util::resolve_str(model, &sections, "model", "model_path");
            let tokenizer = util::resolve_str(tokenizer, &sections, "model", "tokenizer_path");
            let max_seq_len = validated(args::validate_max_seq_len(util::resolve_usize(
                max_seq_len,
                &sections,
                "model",
                "max_seq_len",
                4096,
            )))?;
            // `--host`/`--port` only ever fall back to `[server]` when the
            // flag itself was absent from argv (both are `Option<T>` with
            // no clap default), so a TOML value can never silently widen
            // the bind address a user explicitly chose on the command
            // line (wave-1 addendum correction on cli-04).
            let host = util::resolve_str(host, &sections, "server", "host")
                .unwrap_or_else(|| "127.0.0.1".to_string());
            let port = util::resolve_u16(port, &sections, "server", "port", 8080);
            tracing::info!(host = %host, port, "resolved server bind address");

            #[cfg(feature = "rag")]
            cmd_serve::run(
                model,
                host,
                port,
                max_seq_len,
                tokenizer,
                pool_size,
                bearer_token,
                max_concurrent_requests,
                request_timeout_ms,
                rag,
            )
            .await?;
            #[cfg(not(feature = "rag"))]
            cmd_serve::run(
                model,
                host,
                port,
                max_seq_len,
                tokenizer,
                pool_size,
                bearer_token,
                max_concurrent_requests,
                request_timeout_ms,
            )
            .await?;
        }

        Commands::Info { model, json } => cmd_info::run(model, json)?,

        Commands::BuildInfo => model_desc::print_build_info(),

        Commands::Benchmark {
            model,
            synthetic,
            tokenizer,
            tokens,
            warmup,
            temperature,
            seed,
        } => cmd_benchmark::run(
            model,
            synthetic,
            tokenizer,
            tokens,
            warmup,
            temperature,
            seed,
        )?,

        Commands::Quantize {
            input,
            output,
            format,
            force,
        } => cmd_quantize::run(input, output, format, force)?,

        Commands::Convert {
            from,
            to,
            quant,
            onnx,
        } => cmd_convert::run(from, to, quant, onnx)?,

        #[cfg(feature = "eval")]
        Commands::Eval {
            model,
            task,
            dataset,
            limit,
            max_tokens,
            max_seq_len,
            tokenizer,
            report_json,
            report_markdown,
            allow_vocab_mismatch,
        } => {
            let model = util::resolve_str(model, &sections, "model", "model_path");
            let tokenizer = util::resolve_str(tokenizer, &sections, "model", "tokenizer_path");
            let max_seq_len = validated(args::validate_max_seq_len(util::resolve_usize(
                max_seq_len,
                &sections,
                "model",
                "max_seq_len",
                4096,
            )))?;
            cmd_eval::run(cmd_eval::EvalArgs {
                model,
                task,
                dataset,
                limit,
                max_tokens,
                max_seq_len,
                tokenizer,
                report_json,
                report_markdown,
                allow_vocab_mismatch,
            })?
        }

        Commands::Validate { model } => cmd_validate::run(model)?,

        Commands::Tokenizer { cmd: tok_cmd } => cmd_tokenizer::run(tok_cmd)?,
    }

    Ok(())
}
