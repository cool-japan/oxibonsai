//! `oxibonsai` CLI: argument parsing and subcommand dispatch.
//!
//! [`args`] holds the `clap` definitions; [`util`] holds helpers shared
//! across subcommands (tokenizer auto-detection, GGUF tensor
//! dequantization); each `cmd_*` module implements one subcommand's body.
//! [`run`] itself stays a thin parse-then-dispatch entry point.

use clap::Parser;
use std::path::Path;

use oxibonsai_runtime::OxiBonsaiConfig;

#[cfg(feature = "server")]
mod admission;
mod repl;
mod term;

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

use args::{Cli, Commands};

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
        } => cmd_run::run(
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            max_seq_len,
            tokenizer,
        )?,

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
            seed,
            max_seq_len,
            tokenizer,
        } => cmd_chat::run(
            model,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            max_seq_len,
            tokenizer,
        )?,

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

        Commands::Benchmark {
            tokens,
            warmup,
            temperature,
            seed,
        } => cmd_benchmark::run(tokens, warmup, temperature, seed)?,

        Commands::Quantize {
            input,
            output,
            format,
        } => cmd_quantize::run(input, output, format)?,

        Commands::Convert {
            from,
            to,
            quant,
            onnx,
        } => cmd_convert::run(from, to, quant, onnx)?,

        #[cfg(feature = "eval")]
        Commands::Eval {
            model,
            dataset,
            limit,
            max_tokens,
            max_seq_len,
            tokenizer,
            report_json,
            report_markdown,
        } => cmd_eval::run(
            model,
            dataset,
            limit,
            max_tokens,
            max_seq_len,
            tokenizer,
            report_json,
            report_markdown,
        )?,

        Commands::Validate { model } => cmd_validate::run(model)?,

        Commands::Tokenizer { cmd: tok_cmd } => cmd_tokenizer::run(tok_cmd)?,
    }

    Ok(())
}
