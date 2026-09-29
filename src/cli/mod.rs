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
//!
//! Every value that can be resolved before the model is loaded is
//! validated here, whichever source it came from (a flag already passed its
//! clap `value_parser`; a `--config` value only passes through
//! [`util::validated_opt`]) — so a bad config value is refused, naming the
//! field, before any model is resolved or loaded.

use std::path::Path;

use oxibonsai_runtime::OxiBonsaiConfig;

#[cfg(feature = "server")]
mod admission;
mod bonsai2;
mod generate;
mod model_desc;
mod model_source;
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
mod pull;

pub(crate) use args::Cli;
use args::Commands;
use util::{validated, validated_opt, RawTomlSections};

/// The `OXIBONSAI_CUDA_DEVICE` variable `oxibonsai-kernels` reads when it
/// creates its CUDA context (F-M4).
#[cfg(feature = "server")]
const CUDA_DEVICE_ENV: &str = "OXIBONSAI_CUDA_DEVICE";

/// Apply every environment-variable latch a subcommand needs, on the single
/// main thread, BEFORE the tokio runtime (and therefore any worker thread)
/// exists — the only point where `std::env::set_var` is sound (it becomes
/// an `unsafe fn` outright under edition 2024).
///
/// * cli-24: `oxibonsai repl` without `--cpu-te` sets `OXI_TE_GPU=1` so the
///   text encoder's GPU dispatch `OnceCell` latch reads it on the first
///   forward pass.
/// * F-M4: `oxibonsai serve --cuda-device N` (or `[server].cuda_device` in
///   `--config`) sets `OXIBONSAI_CUDA_DEVICE=N`, which the CUDA backend
///   reads when it creates its context. The `--config` file is read here
///   too (a plain synchronous read); an unreadable or malformed file is
///   ignored at this point because [`run_with`] reports it as the hard error
///   it is a moment later.
pub fn apply_pre_runtime_env(cli: &Cli) {
    match &cli.command {
        Commands::Repl { cpu_te, .. } if !cpu_te && std::env::var_os("OXI_TE_GPU").is_none() => {
            // SAFETY: called from `main()` before the tokio runtime (and
            // therefore any other thread) is built, so no concurrent reader
            // of the environment can exist yet.
            unsafe { std::env::set_var("OXI_TE_GPU", "1") };
        }
        #[cfg(feature = "server")]
        Commands::Serve { cuda_device, .. } => {
            let from_config = || -> Option<u32> {
                let path = cli.config.as_deref()?;
                let content = std::fs::read_to_string(path).ok()?;
                let sections = util::parse_flat_toml_sections(&content, Path::new(path)).ok()?;
                util::toml_u32(&sections, "server", "cuda_device")
            };
            if let Some(device) = cuda_device.or_else(from_config) {
                // SAFETY: see the `Repl` arm — still single-threaded.
                unsafe { std::env::set_var(CUDA_DEVICE_ENV, device.to_string()) };
            }
        }
        _ => {}
    }
}

/// Resolve `--think`/`--no-think` (cli-11) into the template's
/// `enable_thinking: Option<bool>`: `Some(true)`/`Some(false)` for an
/// explicit flag, else `[model].enable_thinking` from `--config`, else
/// `None` — the template's own undefined branch (see
/// [`bonsai2::default_enable_thinking`] for what that means per model).
/// `args.rs`'s `conflicts_with` already makes both flags at once a parse
/// error.
fn resolve_enable_thinking(
    think: bool,
    no_think: bool,
    sections: &RawTomlSections,
) -> Option<bool> {
    if think {
        Some(true)
    } else if no_think {
        Some(false)
    } else {
        util::toml_bool(sections, "model", "enable_thinking")
    }
}

/// `--reasoning-effort` or `[model].reasoning_effort`, validated.
fn resolve_reasoning_effort(
    cli: Option<String>,
    sections: &RawTomlSections,
) -> anyhow::Result<Option<String>> {
    util::resolve_str(cli, sections, "model", "reasoning_effort")
        .map(|s| validated(args::validate_reasoning_effort(&s)))
        .transpose()
}

/// `--prefill-chunk` or `[model].prefill_chunk`, validated.
fn resolve_prefill_chunk(
    cli: Option<usize>,
    sections: &RawTomlSections,
) -> anyhow::Result<Option<usize>> {
    validated_opt(
        cli.or_else(|| util::toml_usize(sections, "model", "prefill_chunk")),
        args::validate_prefill_chunk,
    )
}

/// `--seed`/`[sampling].seed`, validated: precedence flag > TOML > `None`
/// (`run`/`chat`/`benchmark` each apply their own final default, 42, after
/// this; `serve` has no `--seed` flag, so it calls this with `cli: None`
/// and applies its own final default, `OXIBONSAI_SEED`-or-pseudo-random,
/// after this). Unlike [`resolve_sampling_overrides`]'s
/// `f32`/`usize` fields (whose `toml_*` lookups already treat an unparsable
/// value as merely absent, deferring entirely to
/// [`args::validate_temperature`] et al. for range checks), a malformed
/// `[sampling].seed` — present in the file but not a valid non-negative
/// 64-bit integer, e.g. `seed = -1` or `seed = "x"` — must be REJECTED by
/// name rather than silently treated as unset and defaulted: a typo'd seed
/// silently losing reproducibility is exactly the class of bug cli-04
/// exists to catch.
///
/// Underscore digit separators (`seed = 1_000`, valid TOML integer syntax)
/// are stripped before parsing: [`util::parse_flat_toml_sections`] hands
/// this function the raw source text of the value, and `u64::from_str`
/// alone does not accept `_` the way a TOML (or Rust literal) parser does.
fn resolve_seed_override(
    cli: Option<u64>,
    sections: &RawTomlSections,
) -> anyhow::Result<Option<u64>> {
    if let Some(seed) = cli {
        return Ok(Some(seed));
    }
    match sections.get("sampling").and_then(|s| s.get("seed")) {
        Some(raw) => raw.replace('_', "").parse::<u64>().map(Some).map_err(|_| {
            anyhow::anyhow!(
                "config [sampling].seed = {raw}: not a valid seed (must be a non-negative \
                 64-bit integer)"
            )
        }),
        None => Ok(None),
    }
}

/// The RT-17 sampling values a flag or `--config` supplied, validated; the
/// model's own `general.sampling.*` defaults (and the final literals) are
/// applied after the GGUF is loaded, so `None` is kept as "not given".
struct SamplingOverrides {
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
}

fn resolve_sampling_overrides(
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    sections: &RawTomlSections,
) -> anyhow::Result<SamplingOverrides> {
    Ok(SamplingOverrides {
        temperature: validated_opt(
            temperature.or_else(|| util::toml_f32(sections, "sampling", "temperature")),
            args::validate_temperature,
        )?,
        top_k: top_k.or_else(|| util::toml_usize(sections, "sampling", "top_k")),
        top_p: validated_opt(
            top_p.or_else(|| util::toml_f32(sections, "sampling", "top_p")),
            args::validate_top_p,
        )?,
        min_p: validated_opt(
            min_p.or_else(|| util::toml_f32(sections, "sampling", "min_p")),
            args::validate_min_p,
        )?,
    })
}

/// `--max-seq-len`/`--ctx` or `[model].max_seq_len`, validated; the
/// per-architecture default is applied once the GGUF is parsed.
fn resolve_max_seq_len(
    cli: Option<usize>,
    sections: &RawTomlSections,
) -> anyhow::Result<Option<usize>> {
    validated_opt(
        cli.or_else(|| util::toml_usize(sections, "model", "max_seq_len")),
        args::validate_max_seq_len,
    )
}

/// `oxibonsai image`/`repl`: resolve the `[imagen]` section of `--config`
/// into an [`oxibonsai_runtime::config::ImagenConfig`]: an explicit flag
/// wins, then `[imagen]`, then the CLI's own
/// literal default (seed 42, 4 steps, 512 x 512). The result is validated
/// through `OxiBonsaiConfig::validate` (non-zero size and step count).
/// `[imagen].guidance_scale` is refused rather than ignored: the FLUX.2
/// Klein DiT is guidance-distilled and this pipeline has no CFG stage.
fn resolve_imagen_config(
    seed: Option<u64>,
    steps: Option<usize>,
    width: Option<usize>,
    height: Option<usize>,
    sections: &RawTomlSections,
) -> anyhow::Result<oxibonsai_runtime::config::ImagenConfig> {
    if util::toml_f32(sections, "imagen", "guidance_scale").is_some() {
        anyhow::bail!(
            "[imagen].guidance_scale is not supported: the Bonsai-Image (FLUX.2 Klein) pipeline \
             is guidance-distilled and has no classifier-free-guidance stage; remove the key"
        );
    }
    let to_u32 = |name: &str, value: usize| {
        u32::try_from(value).map_err(|_| anyhow::anyhow!("--{name} {value} is out of range"))
    };
    let width = match width {
        Some(w) => to_u32("width", w)?,
        None => util::toml_u32(sections, "imagen", "width").unwrap_or(512),
    };
    let height = match height {
        Some(h) => to_u32("height", h)?,
        None => util::toml_u32(sections, "imagen", "height").unwrap_or(512),
    };
    let steps = match steps {
        Some(s) => to_u32("steps", s)?,
        None => util::toml_u32(sections, "imagen", "steps").unwrap_or(4),
    };
    let imagen = oxibonsai_runtime::config::ImagenConfig {
        model_path: util::toml_str(sections, "imagen", "model_path"),
        width,
        height,
        steps,
        seed: Some(
            seed.or_else(|| util::toml_u64(sections, "imagen", "seed"))
                .unwrap_or(42),
        ),
        output_dir: util::toml_str(sections, "imagen", "output_dir"),
        ..oxibonsai_runtime::config::ImagenConfig::default()
    };
    OxiBonsaiConfig {
        imagen: imagen.clone(),
        ..OxiBonsaiConfig::default()
    }
    .validate()
    .map_err(|e| anyhow::anyhow!("invalid image settings: {e}"))?;
    Ok(imagen)
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
            chat,
            grammar,
            stop,
            min_p,
            backend,
            rope_scaling,
            think,
            no_think,
            reasoning_effort,
            tools,
            show_reasoning,
            hide_reasoning,
            ptq1_transcode,
            prefill_chunk,
            mmproj,
            image,
            image_max_tokens,
            allow_vocab_mismatch,
            no_stream,
        } => {
            let sampling = resolve_sampling_overrides(temperature, top_k, top_p, min_p, &sections)?;
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
                temperature: sampling.temperature,
                top_k: sampling.top_k,
                top_p: sampling.top_p,
                min_p: sampling.min_p,
                repetition_penalty: validated(args::validate_repetition_penalty(
                    util::resolve_f32(
                        repetition_penalty,
                        &sections,
                        "sampling",
                        "repetition_penalty",
                        util::DEFAULT_REPETITION_PENALTY,
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
                seed: resolve_seed_override(seed, &sections)?.unwrap_or(util::DEFAULT_SEED),
                max_seq_len: resolve_max_seq_len(max_seq_len, &sections)?,
                tokenizer: util::resolve_str(tokenizer, &sections, "model", "tokenizer_path"),
                tokenizer_backend,
                chat,
                grammar,
                stop,
                backend: util::resolve_backend(backend, &sections, "model", "backend")?,
                rope_scaling: util::resolve_rope_scaling(
                    rope_scaling,
                    &sections,
                    "model",
                    "rope_scaling",
                )?,
                enable_thinking: resolve_enable_thinking(think, no_think, &sections),
                reasoning_effort: resolve_reasoning_effort(reasoning_effort, &sections)?,
                tools,
                show_reasoning,
                hide_reasoning,
                ptq1_transcode: ptq1_transcode
                    || util::toml_bool(&sections, "model", "ptq1_transcode").unwrap_or(false),
                prefill_chunk: resolve_prefill_chunk(prefill_chunk, &sections)?,
                vision: bonsai2::VisionRequest {
                    mmproj,
                    images: image,
                    image_max_tokens,
                },
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
        } => {
            let imagen = resolve_imagen_config(seed, steps, width, height, &sections)?;
            let out = match &imagen.output_dir {
                Some(dir) if Path::new(&out).is_relative() => {
                    Path::new(dir).join(&out).to_string_lossy().into_owned()
                }
                _ => out,
            };
            cmd_image::run_image(
                prompt,
                out,
                imagen.seed.unwrap_or(42),
                imagen.steps as usize,
                imagen.width as usize,
                imagen.height as usize,
                dit.or(imagen.model_path),
                vae,
                te,
                tokenizer,
            )?
        }

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
        } => {
            let imagen = resolve_imagen_config(seed, steps, width, height, &sections)?;
            cmd_image::run_repl(
                imagen.seed.unwrap_or(42),
                imagen.steps as usize,
                imagen.width as usize,
                imagen.height as usize,
                cpu_te,
                dit.or(imagen.model_path),
                vae,
                te,
                tokenizer,
            )?
        }

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
            min_p,
            backend,
            rope_scaling,
            think,
            no_think,
            reasoning_effort,
            tools,
            show_reasoning,
            hide_reasoning,
            ptq1_transcode,
            prefill_chunk,
            mmproj,
            image,
            image_max_tokens,
            allow_vocab_mismatch,
        } => {
            let sampling = resolve_sampling_overrides(temperature, top_k, top_p, min_p, &sections)?;
            let chat_args = cmd_chat::ChatArgs {
                model: util::resolve_str(model, &sections, "model", "model_path"),
                max_tokens: validated(args::validate_max_tokens(util::resolve_usize(
                    max_tokens,
                    &sections,
                    "sampling",
                    "max_tokens",
                    512,
                )))?,
                temperature: sampling.temperature,
                top_k: sampling.top_k,
                top_p: sampling.top_p,
                min_p: sampling.min_p,
                repetition_penalty: validated(args::validate_repetition_penalty(
                    util::resolve_f32(
                        repetition_penalty,
                        &sections,
                        "sampling",
                        "repetition_penalty",
                        util::DEFAULT_REPETITION_PENALTY,
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
                seed: resolve_seed_override(seed, &sections)?.unwrap_or(util::DEFAULT_SEED),
                max_seq_len: resolve_max_seq_len(max_seq_len, &sections)?,
                tokenizer: util::resolve_str(tokenizer, &sections, "model", "tokenizer_path"),
                tokenizer_backend,
                grammar,
                stop,
                backend: util::resolve_backend(backend, &sections, "model", "backend")?,
                rope_scaling: util::resolve_rope_scaling(
                    rope_scaling,
                    &sections,
                    "model",
                    "rope_scaling",
                )?,
                enable_thinking: resolve_enable_thinking(think, no_think, &sections),
                reasoning_effort: resolve_reasoning_effort(reasoning_effort, &sections)?,
                tools,
                show_reasoning,
                hide_reasoning,
                ptq1_transcode: ptq1_transcode
                    || util::toml_bool(&sections, "model", "ptq1_transcode").unwrap_or(false),
                prefill_chunk: resolve_prefill_chunk(prefill_chunk, &sections)?,
                vision: bonsai2::VisionRequest {
                    mmproj,
                    images: image,
                    image_max_tokens,
                },
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
            backend,
            rope_scaling,
            think,
            no_think,
            reasoning_effort,
            tools,
            cuda_device,
            bearer_token_file,
            rate_limit_rpm,
            rate_limit_burst,
            cors_origin,
            cors_allow_credentials,
            max_body_bytes,
            enable_ui,
            max_output_tokens,
            ptq1_transcode,
            prefill_chunk,
            mmproj,
            image,
            image_max_tokens,
            embedding_backend,
            embedding_corpus,
        } => {
            let model = util::resolve_str(model, &sections, "model", "model_path");
            let tokenizer = util::resolve_str(tokenizer, &sections, "model", "tokenizer_path");
            // `--host`/`--port` only ever fall back to `[server]` when the
            // flag itself was absent from argv (both are `Option<T>` with
            // no clap default), so a TOML value can never silently widen
            // the bind address a user explicitly chose on the command
            // line (cli-04).
            let host = util::resolve_str(host, &sections, "server", "host")
                .unwrap_or_else(|| "127.0.0.1".to_string());
            let port = util::resolve_u16(port, &sections, "server", "port", 8080);
            tracing::info!(host = %host, port, "resolved server bind address");

            let serve_args = cmd_serve::ServeArgs {
                model,
                host,
                port,
                max_seq_len: resolve_max_seq_len(max_seq_len, &sections)?,
                // `serve` has no `--seed` flag of its own (unlike
                // run/chat/benchmark), so `cli` is always `None` here: this
                // validates and surfaces only `[sampling].seed` from
                // --config, naming a malformed value by field (cli-04).
                toml_seed: resolve_seed_override(None, &sections)?,
                tokenizer,
                pool_size,
                bearer_token,
                max_concurrent_requests,
                request_timeout_ms,
                #[cfg(feature = "rag")]
                rag,
                backend: util::resolve_backend(backend, &sections, "model", "backend")?,
                rope_scaling: util::resolve_rope_scaling(
                    rope_scaling,
                    &sections,
                    "model",
                    "rope_scaling",
                )?,
                enable_thinking: resolve_enable_thinking(think, no_think, &sections),
                reasoning_effort: resolve_reasoning_effort(reasoning_effort, &sections)?,
                tools,
                cuda_device: cuda_device
                    .or_else(|| util::toml_u32(&sections, "server", "cuda_device")),
                bearer_token_file: util::resolve_str(
                    bearer_token_file,
                    &sections,
                    "server",
                    "bearer_token_file",
                ),
                rate_limit_rpm: rate_limit_rpm
                    .or_else(|| util::toml_u32(&sections, "server", "rate_limit_rpm")),
                rate_limit_burst: rate_limit_burst
                    .or_else(|| util::toml_u32(&sections, "server", "rate_limit_burst")),
                cors_origin: util::resolve_str(cors_origin, &sections, "server", "cors_origin"),
                cors_allow_credentials,
                max_body_bytes: max_body_bytes
                    .or_else(|| util::toml_u64(&sections, "server", "max_body_bytes")),
                enable_ui: enable_ui
                    || util::toml_bool(&sections, "server", "enable_ui").unwrap_or(false),
                max_output_tokens: validated_opt(
                    max_output_tokens
                        .or_else(|| util::toml_usize(&sections, "server", "max_output_tokens")),
                    args::validate_max_output_tokens,
                )?,
                ptq1_transcode: ptq1_transcode
                    || util::toml_bool(&sections, "model", "ptq1_transcode").unwrap_or(false),
                prefill_chunk: resolve_prefill_chunk(prefill_chunk, &sections)?,
                vision: bonsai2::VisionRequest {
                    mmproj,
                    images: image,
                    image_max_tokens,
                },
                embedding_backend,
                embedding_corpus,
            };
            cmd_serve::run(serve_args).await?;
        }

        Commands::Info { model, json } => cmd_info::run(model, json)?,

        Commands::BuildInfo => model_desc::print_build_info(),

        Commands::Benchmark {
            model,
            synthetic,
            tokenizer,
            tokenizer_backend,
            tokens,
            warmup,
            temperature,
            seed,
        } => cmd_benchmark::run(
            model,
            synthetic,
            tokenizer,
            tokenizer_backend,
            tokens,
            warmup,
            temperature,
            resolve_seed_override(seed, &sections)?.unwrap_or(util::DEFAULT_SEED),
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
            allow_unmapped,
        } => cmd_convert::run(from, to, quant, onnx, allow_unmapped)?,

        #[cfg(feature = "eval")]
        Commands::Eval {
            model,
            task,
            dataset,
            limit,
            max_tokens,
            max_seq_len,
            tokenizer,
            tokenizer_backend,
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
                tokenizer_backend,
                report_json,
                report_markdown,
                allow_vocab_mismatch,
            })?
        }

        Commands::Validate { model } => cmd_validate::run(model)?,

        Commands::Tokenizer { cmd: tok_cmd } => cmd_tokenizer::run(tok_cmd)?,

        Commands::Pull {
            model_or_url,
            out,
            band,
            vision,
            force,
        } => {
            pull::run(pull::PullArgs {
                model_or_url,
                out_dir: out,
                band,
                vision,
                force,
            })
            .await?
        }
    }

    Ok(())
}

#[cfg(test)]
mod test_fixtures;

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod dispatch_tests;
