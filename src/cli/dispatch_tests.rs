//! Dispatch-level tests for `mod.rs` (`run_with` / `apply_pre_runtime_env`
//! and their resolvers), declared there via `#[path]`.

use super::*;
use crate::cli::util::test_env::{self, EnvVarGuard};
use clap::Parser;

fn config_file(tag: &str, toml: &str) -> (std::path::PathBuf, String) {
    let dir = crate::cli::test_fixtures::scratch_dir(&format!("dispatch_{tag}"));
    let path = dir.join("oxibonsai.toml");
    std::fs::write(&path, toml).expect("write config");
    let path_str = path.to_string_lossy().into_owned();
    (dir, path_str)
}

/// `oxibonsai --config <file> <subcommand...>` with NO model anywhere: a
/// config value must be refused (naming the field) BEFORE model resolution,
/// so the error is about the value — never "no model: pass --model".
fn run_with_config(tag: &str, toml: &str, subcommand: &[&str]) -> String {
    let (dir, path) = config_file(tag, toml);
    let mut argv = vec!["oxibonsai", "--config", path.as_str()];
    argv.extend_from_slice(subcommand);
    let cli = Cli::try_parse_from(argv).expect("argv parses");
    let result = {
        // `run` would fall back to OXI_MODEL; make sure none is set.
        let _env = test_env::lock();
        let _model = EnvVarGuard::remove("OXI_MODEL");
        // The resolvers under test run before any await point that could
        // observe the environment again.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(run_with(cli))
    };
    let _ = std::fs::remove_dir_all(&dir);
    match result {
        Ok(()) => "OK".to_string(),
        Err(e) => e.to_string(),
    }
}

#[test]
fn config_negative_temperature_is_refused_naming_the_field_before_model_resolution() {
    let msg = run_with_config(
        "temp",
        "[sampling]\ntemperature = -5.0\n",
        &["run", "--prompt", "hi"],
    );
    assert!(msg.contains("temperature"), "{msg}");
    assert!(
        !msg.contains("no model"),
        "validated before model resolution: {msg}"
    );
}

#[test]
fn config_top_p_above_one_is_refused_naming_the_field() {
    let msg = run_with_config(
        "top_p",
        "[sampling]\ntop_p = 5.0\n",
        &["run", "--prompt", "hi"],
    );
    assert!(
        msg.contains("top-p must be in the range (0.0, 1.0]"),
        "{msg}"
    );
    assert!(!msg.contains("no model"), "{msg}");
}

#[test]
fn config_min_p_and_max_seq_len_are_validated_before_model_resolution() {
    let msg = run_with_config("min_p", "[sampling]\nmin_p = 1.5\n", &["chat"]);
    assert!(msg.contains("min-p must be in the range"), "{msg}");

    let msg = run_with_config(
        "max_seq_len",
        "[model]\nmax_seq_len = 0\n",
        &["run", "--prompt", "hi"],
    );
    assert!(msg.contains("max-seq-len must be >= 1"), "{msg}");

    let msg = run_with_config(
        "effort",
        "[model]\nreasoning_effort = \"maximum\"\n",
        &["run", "--prompt", "hi"],
    );
    assert!(msg.contains("reasoning-effort"), "{msg}");

    let msg = run_with_config(
        "prefill",
        "[model]\nprefill_chunk = 0\n",
        &["run", "--prompt", "hi"],
    );
    assert!(msg.contains("prefill-chunk must be >= 1"), "{msg}");
}

#[cfg(feature = "server")]
#[test]
fn config_serve_values_are_validated_before_model_resolution() {
    let msg = run_with_config("serve_ctx", "[model]\nmax_seq_len = 0\n", &["serve"]);
    assert!(msg.contains("max-seq-len must be >= 1"), "{msg}");
    let msg = run_with_config("serve_cap", "[server]\nmax_output_tokens = 0\n", &["serve"]);
    assert!(
        msg.contains("max-output-tokens must be at least 1"),
        "{msg}"
    );
    // `--embedding-backend tfidf` without a corpus is refused before any
    // model is resolved (the config is otherwise valid).
    let msg = run_with_config(
        "serve_tfidf",
        "[server]\nport = 8080\n",
        &["serve", "--embedding-backend", "tfidf"],
    );
    assert!(msg.contains("--embedding-corpus"), "{msg}");
    assert!(!msg.contains("no model"), "{msg}");
}

#[test]
fn a_valid_config_value_reaches_model_resolution() {
    // Sanity: with valid values the command proceeds to the model step.
    let msg = run_with_config(
        "valid",
        "[sampling]\ntemperature = 0.5\ntop_p = 0.9\n",
        &["run", "--prompt", "hi"],
    );
    assert!(msg.contains("no model"), "{msg}");
}

// ── resolvers ───────────────────────────────────────────────────────────────

#[test]
fn enable_thinking_resolves_flag_then_config_then_undefined() {
    let mut sections = RawTomlSections::new();
    assert_eq!(resolve_enable_thinking(true, false, &sections), Some(true));
    assert_eq!(resolve_enable_thinking(false, true, &sections), Some(false));
    assert_eq!(resolve_enable_thinking(false, false, &sections), None);
    sections
        .entry("model".to_string())
        .or_default()
        .insert("enable_thinking".to_string(), "false".to_string());
    assert_eq!(
        resolve_enable_thinking(false, false, &sections),
        Some(false)
    );
    assert_eq!(resolve_enable_thinking(true, false, &sections), Some(true));
}

#[test]
fn imagen_config_layers_flag_over_the_imagen_section_over_the_literal() {
    let toml = "[imagen]\nwidth = 768\nsteps = 8\nseed = 7\noutput_dir = \"renders\"\nmodel_path = \"dit.gguf\"\n";
    let sections = util::parse_flat_toml_sections(toml, Path::new("test.toml"))
        .expect("[imagen] is a known section");
    let resolved = resolve_imagen_config(None, None, Some(1024), None, &sections).expect("valid");
    assert_eq!(resolved.width, 1024, "the flag wins");
    assert_eq!(resolved.height, 512, "no flag, no TOML: the CLI literal");
    assert_eq!(resolved.steps, 8, "the TOML value");
    assert_eq!(resolved.seed, Some(7));
    assert_eq!(resolved.output_dir.as_deref(), Some("renders"));
    assert_eq!(resolved.model_path.as_deref(), Some("dit.gguf"));

    let defaults =
        resolve_imagen_config(None, None, None, None, &RawTomlSections::new()).expect("defaults");
    assert_eq!(
        (defaults.width, defaults.height, defaults.steps),
        (512, 512, 4)
    );
    assert_eq!(defaults.seed, Some(42));
}

#[test]
fn imagen_config_refuses_zero_sizes_and_a_guidance_scale() {
    let err = resolve_imagen_config(None, Some(0), None, None, &RawTomlSections::new())
        .expect_err("zero steps");
    assert!(err.to_string().contains("steps"), "{err}");
    let sections =
        util::parse_flat_toml_sections("[imagen]\nguidance_scale = 3.5\n", Path::new("t.toml"))
            .expect("parse");
    let err = resolve_imagen_config(None, None, None, None, &sections).expect_err("guidance");
    assert!(err.to_string().contains("guidance_scale"), "{err}");
}

/// F-M4: `--cuda-device` (or `[server].cuda_device`) is applied to the
/// process environment by `apply_pre_runtime_env` — before any runtime
/// thread exists — never inside the async server.
#[cfg(feature = "server")]
#[test]
fn cuda_device_is_applied_before_the_runtime_from_the_flag_or_the_config() {
    let _env = test_env::lock();
    let _device = EnvVarGuard::remove(CUDA_DEVICE_ENV);
    let cli = Cli::try_parse_from(["oxibonsai", "serve", "--cuda-device", "3"]).expect("parse");
    apply_pre_runtime_env(&cli);
    assert_eq!(std::env::var(CUDA_DEVICE_ENV).ok().as_deref(), Some("3"));

    let (dir, path) = config_file("cuda", "[server]\ncuda_device = 2\n");
    let cli =
        Cli::try_parse_from(["oxibonsai", "--config", path.as_str(), "serve"]).expect("parse");
    apply_pre_runtime_env(&cli);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(std::env::var(CUDA_DEVICE_ENV).ok().as_deref(), Some("2"));
}
