//! Unit tests for `args.rs`, split into a sibling file to keep
//! `args.rs` under the 2000-line policy limit (declared there via
//! `#[path]`, so `super` still names that module).

use super::*;

#[test]
fn parse_temperature_rejects_negative() {
    assert!(parse_temperature("-1").is_err());
    assert!(parse_temperature("-0.0001").is_err());
}

#[test]
fn parse_temperature_accepts_zero_and_positive() {
    assert_eq!(parse_temperature("0").unwrap(), 0.0);
    assert_eq!(parse_temperature("0.7").unwrap(), 0.7);
}

#[test]
fn parse_temperature_rejects_nan_and_infinity() {
    assert!(parse_temperature("nan").is_err());
    assert!(parse_temperature("inf").is_err());
}

#[test]
fn parse_top_p_rejects_out_of_range() {
    assert!(
        parse_top_p("0").is_err(),
        "0.0 excluded (empty distribution)"
    );
    assert!(parse_top_p("1.5").is_err());
    assert!(parse_top_p("-0.1").is_err());
}

#[test]
fn parse_top_p_accepts_valid_range() {
    assert_eq!(parse_top_p("1.0").unwrap(), 1.0);
    assert_eq!(parse_top_p("0.9").unwrap(), 0.9);
    assert!(parse_top_p("0.0001").is_ok());
}

#[test]
fn parse_repetition_penalty_rejects_non_positive() {
    assert!(parse_repetition_penalty("0").is_err());
    assert!(parse_repetition_penalty("-1.1").is_err());
}

#[test]
fn parse_repetition_penalty_accepts_positive() {
    assert_eq!(parse_repetition_penalty("1.0").unwrap(), 1.0);
    assert_eq!(parse_repetition_penalty("1.1").unwrap(), 1.1);
}

#[test]
fn parse_openai_penalty_accepts_negative_within_range() {
    assert_eq!(parse_openai_penalty("-2.0").unwrap(), -2.0);
    assert_eq!(parse_openai_penalty("0.0").unwrap(), 0.0);
    assert_eq!(parse_openai_penalty("2.0").unwrap(), 2.0);
}

#[test]
fn parse_openai_penalty_rejects_out_of_range() {
    assert!(parse_openai_penalty("-2.1").is_err());
    assert!(parse_openai_penalty("2.1").is_err());
}

#[test]
fn parse_max_tokens_rejects_zero() {
    assert!(parse_max_tokens("0").is_err());
}

#[test]
fn parse_max_tokens_accepts_positive() {
    assert_eq!(parse_max_tokens("1").unwrap(), 1);
    assert_eq!(parse_max_tokens("256").unwrap(), 256);
}

#[test]
fn parse_max_seq_len_rejects_zero() {
    assert!(parse_max_seq_len("0").is_err());
}

#[test]
fn cli_parses_run_with_new_penalty_flags() {
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "run",
        "--prompt",
        "hi",
        "--repetition-penalty",
        "1.2",
        "--frequency-penalty",
        "-0.5",
        "--presence-penalty",
        "0.5",
    ])
    .expect("should parse");
    match cli.command {
        Commands::Run {
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            ..
        } => {
            assert_eq!(repetition_penalty, Some(1.2));
            assert_eq!(frequency_penalty, Some(-0.5));
            assert_eq!(presence_penalty, Some(0.5));
        }
        _ => panic!("expected Run"),
    }
}

#[test]
fn cli_rejects_negative_temperature() {
    let result = Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi", "--temperature", "-1"]);
    assert!(result.is_err(), "negative temperature must be rejected");
}

#[test]
fn cli_rejects_top_p_above_one() {
    let result = Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi", "--top-p", "5.0"]);
    assert!(result.is_err(), "top-p > 1.0 must be rejected");
}

#[test]
fn cli_rejects_zero_max_tokens() {
    let result = Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi", "--max-tokens", "0"]);
    assert!(result.is_err(), "max-tokens == 0 must be rejected");
}

#[test]
fn cli_defaults_are_none_for_config_layering() {
    let cli = Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi"]).expect("parse");
    match cli.command {
        Commands::Run {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            max_tokens,
            max_seq_len,
            ..
        } => {
            assert_eq!(temperature, None);
            assert_eq!(top_k, None);
            assert_eq!(top_p, None);
            assert_eq!(repetition_penalty, None);
            assert_eq!(max_tokens, None);
            assert_eq!(max_seq_len, None);
        }
        _ => panic!("expected Run"),
    }
}

#[test]
fn benchmark_requires_explicit_synthetic_or_model() {
    let cli = Cli::try_parse_from(["oxibonsai", "benchmark"]).expect("parse");
    match cli.command {
        Commands::Benchmark {
            model, synthetic, ..
        } => {
            assert!(model.is_none());
            assert!(
                !synthetic,
                "synthetic must default to false, not silently on"
            );
        }
        _ => panic!("expected Benchmark"),
    }
}

#[test]
fn validate_temperature_rejects_a_value_that_never_went_through_clap() {
    // The scenario a config-file-sourced value hits: parsed by
    // `mod.rs`'s own `toml_f32` (plain `str::parse`, no clap
    // `value_parser`), then re-validated by calling this directly.
    assert!(validate_temperature(-5.0).is_err());
    assert!(validate_temperature(0.0).is_ok());
}

#[test]
fn validate_top_p_rejects_a_value_that_never_went_through_clap() {
    assert!(validate_top_p(5.0).is_err());
    assert!(validate_top_p(0.5).is_ok());
}

#[test]
fn validate_repetition_penalty_rejects_a_value_that_never_went_through_clap() {
    assert!(validate_repetition_penalty(-1.0).is_err());
    assert!(validate_repetition_penalty(1.0).is_ok());
}

#[test]
fn validate_max_tokens_rejects_a_value_that_never_went_through_clap() {
    assert!(validate_max_tokens(0).is_err());
    assert!(validate_max_tokens(1).is_ok());
}

#[test]
fn quantize_force_defaults_to_false() {
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "quantize",
        "--input",
        "a.gguf",
        "--output",
        "b.gguf",
    ])
    .expect("parse");
    match cli.command {
        Commands::Quantize { force, .. } => assert!(!force),
        _ => panic!("expected Quantize"),
    }
}

// ── every subcommand's flag surface parses ──────────────────────────────────

fn run_cli(extra: &[&str]) -> Commands {
    let mut argv = vec!["oxibonsai", "run", "--prompt", "hi"];
    argv.extend_from_slice(extra);
    Cli::try_parse_from(argv).expect("run flags parse").command
}

fn chat_cli(extra: &[&str]) -> Commands {
    let mut argv = vec!["oxibonsai", "chat"];
    argv.extend_from_slice(extra);
    Cli::try_parse_from(argv).expect("chat flags parse").command
}

#[test]
fn run_parses_min_p_backend_and_every_rope_scaling_value() {
    use oxibonsai_runtime::config::RopeScalingMode;
    use oxibonsai_runtime::engine_seam::Backend;
    for (value, mode) in [
        ("auto", RopeScalingMode::Auto),
        ("on", RopeScalingMode::On),
        ("off", RopeScalingMode::Off),
    ] {
        match run_cli(&["--rope-scaling", value]) {
            Commands::Run { rope_scaling, .. } => assert_eq!(rope_scaling, Some(mode)),
            _ => panic!("expected Run"),
        }
    }
    for (value, backend_value) in [
        ("auto", Backend::Auto),
        ("cpu", Backend::Cpu),
        ("metal", Backend::Metal),
    ] {
        match run_cli(&["--backend", value]) {
            Commands::Run { backend, .. } => assert_eq!(backend, Some(backend_value)),
            _ => panic!("expected Run"),
        }
    }
    match run_cli(&["--min-p", "0.05"]) {
        Commands::Run { min_p, .. } => assert_eq!(min_p, Some(0.05)),
        _ => panic!("expected Run"),
    }
    let bad = [
        vec!["--rope-scaling", "yarn"],
        vec!["--backend", "tpu"],
        vec!["--min-p", "1.5"],
    ];
    for flags in bad {
        let mut argv = vec!["oxibonsai", "run", "--prompt", "hi"];
        argv.extend(flags.iter().copied());
        assert!(
            Cli::try_parse_from(argv).is_err(),
            "{flags:?} must be rejected"
        );
    }
}

#[test]
fn run_parses_the_chat_contract_flags() {
    match run_cli(&[
        "--chat",
        "--think",
        "--reasoning-effort",
        "low",
        "--tools",
        "tools.json",
        "--show-reasoning",
    ]) {
        Commands::Run {
            chat,
            think,
            no_think,
            reasoning_effort,
            tools,
            show_reasoning,
            hide_reasoning,
            ..
        } => {
            assert!(chat && think && !no_think && show_reasoning && !hide_reasoning);
            assert_eq!(reasoning_effort.as_deref(), Some("low"));
            assert_eq!(tools.as_deref(), Some("tools.json"));
        }
        _ => panic!("expected Run"),
    }
    match run_cli(&[
        "--no-think",
        "--hide-reasoning",
        "--reasoning-effort",
        "xhigh",
    ]) {
        Commands::Run {
            think,
            no_think,
            hide_reasoning,
            reasoning_effort,
            ..
        } => {
            assert!(!think && no_think && hide_reasoning);
            assert_eq!(reasoning_effort.as_deref(), Some("xhigh"));
        }
        _ => panic!("expected Run"),
    }
    for conflicting in [
        vec!["--think", "--no-think"],
        vec!["--show-reasoning", "--hide-reasoning"],
        vec!["--reasoning-effort", "maximum"],
    ] {
        let mut argv = vec!["oxibonsai", "run", "--prompt", "hi"];
        argv.extend(conflicting.iter().copied());
        assert!(
            Cli::try_parse_from(argv).is_err(),
            "{conflicting:?} must be rejected"
        );
    }
}

#[test]
fn run_parses_ctx_prefill_chunk_transcode_and_the_vision_flags() {
    match run_cli(&[
        "--ctx",
        "300000",
        "--prefill-chunk",
        "256",
        "--ptq1-transcode",
        "--mmproj",
        "mmproj.gguf",
        "--image",
        "a.png",
        "--image",
        "https://example.com/b.png",
        "--image-max-tokens",
        "512",
    ]) {
        Commands::Run {
            max_seq_len,
            prefill_chunk,
            ptq1_transcode,
            mmproj,
            image,
            image_max_tokens,
            ..
        } => {
            assert_eq!(
                max_seq_len,
                Some(300_000),
                "--ctx is an alias of --max-seq-len"
            );
            assert_eq!(prefill_chunk, Some(256));
            assert!(ptq1_transcode);
            assert_eq!(mmproj.as_deref(), Some("mmproj.gguf"));
            assert_eq!(
                image,
                vec!["a.png".to_string(), "https://example.com/b.png".to_string()]
            );
            assert_eq!(image_max_tokens, Some(512));
        }
        _ => panic!("expected Run"),
    }
    match run_cli(&[]) {
        Commands::Run {
            prefill_chunk,
            image_max_tokens,
            chat,
            ptq1_transcode,
            ..
        } => {
            assert_eq!(
                prefill_chunk, None,
                "no clap default: the model's own applies"
            );
            assert_eq!(
                image_max_tokens, None,
                "1024 is applied by the vision request"
            );
            assert!(!chat && !ptq1_transcode);
        }
        _ => panic!("expected Run"),
    }
    for bad in [
        vec!["--prefill-chunk", "0"],
        vec!["--image-max-tokens", "0"],
        vec!["--ctx", "0"],
    ] {
        let mut argv = vec!["oxibonsai", "run", "--prompt", "hi"];
        argv.extend(bad.iter().copied());
        assert!(
            Cli::try_parse_from(argv).is_err(),
            "{bad:?} must be rejected"
        );
    }
}

#[test]
fn chat_parses_every_new_flag() {
    match chat_cli(&[
        "--min-p",
        "0.1",
        "--backend",
        "cpu",
        "--rope-scaling",
        "off",
        "--no-think",
        "--reasoning-effort",
        "medium",
        "--tools",
        "t.json",
        "--hide-reasoning",
        "--ptq1-transcode",
        "--prefill-chunk",
        "64",
        "--ctx",
        "8192",
        "--mmproj",
        "m.gguf",
        "--image",
        "x.png",
        "--image-max-tokens",
        "256",
        "--seed",
        "7",
    ]) {
        Commands::Chat {
            min_p,
            backend,
            rope_scaling,
            no_think,
            reasoning_effort,
            tools,
            hide_reasoning,
            ptq1_transcode,
            prefill_chunk,
            max_seq_len,
            mmproj,
            image,
            image_max_tokens,
            seed,
            ..
        } => {
            assert_eq!(min_p, Some(0.1));
            assert_eq!(backend, Some(oxibonsai_runtime::engine_seam::Backend::Cpu));
            assert_eq!(
                rope_scaling,
                Some(oxibonsai_runtime::config::RopeScalingMode::Off)
            );
            assert!(no_think && hide_reasoning && ptq1_transcode);
            assert_eq!(reasoning_effort.as_deref(), Some("medium"));
            assert_eq!(tools.as_deref(), Some("t.json"));
            assert_eq!(prefill_chunk, Some(64));
            assert_eq!(max_seq_len, Some(8192));
            assert_eq!(mmproj.as_deref(), Some("m.gguf"));
            assert_eq!(image, vec!["x.png".to_string()]);
            assert_eq!(image_max_tokens, Some(256));
            assert_eq!(seed, Some(7));
        }
        _ => panic!("expected Chat"),
    }
}

#[cfg(feature = "server")]
#[test]
fn serve_parses_every_hardening_contract_and_vision_flag() {
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "serve",
        "--ctx",
        "16384",
        "--backend",
        "cpu",
        "--rope-scaling",
        "on",
        "--think",
        "--reasoning-effort",
        "xhigh",
        "--tools",
        "tools.json",
        "--cuda-device",
        "1",
        "--bearer-token-file",
        "token.txt",
        "--rate-limit-rpm",
        "120",
        "--rate-limit-burst",
        "10",
        "--cors-origin",
        "https://app.example.com",
        "--cors-allow-credentials",
        "--max-body-bytes",
        "1048576",
        "--enable-ui",
        "--max-output-tokens",
        "2048",
        "--ptq1-transcode",
        "--prefill-chunk",
        "128",
        "--mmproj",
        "m.gguf",
        "--image",
        "i.png",
        "--image-max-tokens",
        "768",
        "--embedding-backend",
        "none",
    ])
    .expect("serve flags parse");
    match cli.command {
        Commands::Serve {
            max_seq_len,
            backend,
            rope_scaling,
            think,
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
            ..
        } => {
            assert_eq!(max_seq_len, Some(16_384));
            assert_eq!(backend, Some(oxibonsai_runtime::engine_seam::Backend::Cpu));
            assert_eq!(
                rope_scaling,
                Some(oxibonsai_runtime::config::RopeScalingMode::On)
            );
            assert!(think);
            assert_eq!(reasoning_effort.as_deref(), Some("xhigh"));
            assert_eq!(tools.as_deref(), Some("tools.json"));
            assert_eq!(cuda_device, Some(1));
            assert_eq!(bearer_token_file.as_deref(), Some("token.txt"));
            assert_eq!(rate_limit_rpm, Some(120));
            assert_eq!(rate_limit_burst, Some(10));
            assert_eq!(cors_origin.as_deref(), Some("https://app.example.com"));
            assert!(cors_allow_credentials && enable_ui && ptq1_transcode);
            assert_eq!(max_body_bytes, Some(1_048_576));
            assert_eq!(max_output_tokens, Some(2048));
            assert_eq!(prefill_chunk, Some(128));
            assert_eq!(mmproj.as_deref(), Some("m.gguf"));
            assert_eq!(image, vec!["i.png".to_string()]);
            assert_eq!(image_max_tokens, Some(768));
            assert_eq!(embedding_backend, EmbeddingBackendChoice::None);
        }
        _ => panic!("expected Serve"),
    }
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "serve",
        "--embedding-backend",
        "tfidf",
        "--embedding-corpus",
        "corpus.txt",
    ])
    .expect("tfidf flags parse");
    match cli.command {
        Commands::Serve {
            embedding_backend,
            embedding_corpus,
            ..
        } => {
            assert_eq!(embedding_backend, EmbeddingBackendChoice::Tfidf);
            assert_eq!(embedding_corpus.as_deref(), Some("corpus.txt"));
        }
        _ => panic!("expected Serve"),
    }
    // Defaults and refusals.
    let cli = Cli::try_parse_from(["oxibonsai", "serve"]).expect("bare serve");
    match cli.command {
        Commands::Serve {
            embedding_backend,
            enable_ui,
            max_output_tokens,
            ..
        } => {
            assert_eq!(embedding_backend, EmbeddingBackendChoice::Model);
            assert!(!enable_ui);
            assert_eq!(max_output_tokens, None);
        }
        _ => panic!("expected Serve"),
    }
    for bad in [
        vec!["--max-output-tokens", "0"],
        vec!["--embedding-backend", "bm25"],
        vec!["--embedding-corpus"],
        vec!["--think", "--no-think"],
        vec!["--bearer-token", "x", "--bearer-token-file", "f"],
    ] {
        let mut argv = vec!["oxibonsai", "serve"];
        argv.extend(bad.iter().copied());
        assert!(
            Cli::try_parse_from(argv).is_err(),
            "{bad:?} must be rejected"
        );
    }
}

#[cfg(feature = "server")]
#[test]
fn serve_help_documents_the_checksums_environment() {
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    let serve = cmd.find_subcommand_mut("serve").expect("serve subcommand");
    let help = serve.render_long_help().to_string();
    assert!(help.contains("OXIBONSAI_CHECKSUMS_FILE"), "{help}");
    assert!(help.contains("scripts/checksums.sha256"), "{help}");
    assert!(help.contains("RELATIVE TO THE CURRENT WORKING"), "{help}");
}

#[test]
fn pull_parses_band_vision_and_force() {
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "pull",
        "bonsai2-27b",
        "--band",
        "ptq1",
        "--vision",
        "--force",
        "--out",
        "weights",
    ])
    .expect("pull flags parse");
    match cli.command {
        Commands::Pull {
            model_or_url,
            band,
            vision,
            force,
            out,
        } => {
            assert_eq!(model_or_url, "bonsai2-27b");
            assert_eq!(band, "ptq1");
            assert!(vision && force);
            assert_eq!(out, "weights");
        }
        _ => panic!("expected Pull"),
    }
    let cli = Cli::try_parse_from(["oxibonsai", "pull", "bonsai-8b"]).expect("defaults");
    match cli.command {
        Commands::Pull {
            band,
            vision,
            force,
            out,
            ..
        } => {
            assert_eq!(band, "pq2");
            assert!(!vision && !force);
            assert_eq!(out, "models");
        }
        _ => panic!("expected Pull"),
    }
}

#[test]
fn convert_parses_allow_unmapped_and_documents_the_prism_formats() {
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "convert",
        "--from",
        "hf",
        "--to",
        "out.gguf",
        "--quant",
        "pq2_0",
        "--allow-unmapped",
    ])
    .expect("convert flags parse");
    match cli.command {
        Commands::Convert {
            allow_unmapped,
            quant,
            ..
        } => {
            assert!(allow_unmapped);
            assert_eq!(quant, "pq2_0");
        }
        _ => panic!("expected Convert"),
    }
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    let help = cmd
        .find_subcommand_mut("convert")
        .expect("convert")
        .render_long_help()
        .to_string();
    for format in ["pq2_0", "ptq1_0", "q2_0_g64"] {
        assert!(
            help.contains(format),
            "convert --help must list {format}: {help}"
        );
    }
    let mut cmd = Cli::command();
    let help = cmd
        .find_subcommand_mut("quantize")
        .expect("quantize")
        .render_long_help()
        .to_string();
    for format in ["q2_k", "q3_k", "q8_k"] {
        assert!(
            help.contains(format),
            "quantize --help must list {format}: {help}"
        );
    }
}

#[test]
fn benchmark_parses_tokenizer_backend_and_image_flags_are_optional() {
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "benchmark",
        "--model",
        "m.gguf",
        "--tokenizer-backend",
        "native",
    ])
    .expect("benchmark flags parse");
    match cli.command {
        Commands::Benchmark {
            tokenizer_backend, ..
        } => assert_eq!(tokenizer_backend, TokenizerBackendChoice::Native),
        _ => panic!("expected Benchmark"),
    }
    let cli = Cli::try_parse_from(["oxibonsai", "image", "--prompt", "p", "--out", "o.png"])
        .expect("image parses");
    match cli.command {
        Commands::Image {
            seed,
            steps,
            width,
            height,
            ..
        } => {
            assert_eq!((seed, steps, width, height), (None, None, None, None));
        }
        _ => panic!("expected Image"),
    }
}

/// `--tokenizer-backend` reaches `eval` the same
/// way it already does `run`/`chat`/`benchmark`, and defaults to `Auto`
/// when not given.
#[cfg(feature = "eval")]
#[test]
fn eval_parses_tokenizer_backend_and_defaults_to_auto() {
    let cli = Cli::try_parse_from([
        "oxibonsai",
        "eval",
        "--dataset",
        "d.jsonl",
        "--tokenizer-backend",
        "native",
    ])
    .expect("eval flags parse");
    match cli.command {
        Commands::Eval {
            tokenizer_backend, ..
        } => assert_eq!(tokenizer_backend, TokenizerBackendChoice::Native),
        _ => panic!("expected Eval"),
    }

    let cli = Cli::try_parse_from(["oxibonsai", "eval", "--dataset", "d.jsonl"])
        .expect("eval without --tokenizer-backend still parses");
    match cli.command {
        Commands::Eval {
            tokenizer_backend, ..
        } => assert_eq!(tokenizer_backend, TokenizerBackendChoice::Auto),
        _ => panic!("expected Eval"),
    }
}
