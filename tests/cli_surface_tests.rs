//! End-to-end CLI surface tests (T-14) and regression tests for several
//! honesty/safety fixes made alongside them.
//!
//! T-14: `tokenizer`, `benchmark` and `image` had no invocation-level
//! coverage at all, and the base64 embeddings encoding path (`encoding_format:
//! "base64"`) had no test proving `data[0].embedding` actually serialises as
//! a string. These tests spawn the real compiled binary
//! (`CARGO_BIN_EXE_oxibonsai`) for the CLI surface, mirroring the existing
//! `tests/image_guidance_removed_tests.rs` / `tests/quantize_cli_tests.rs`
//! style, and use the in-process `tower::ServiceExt::oneshot` style from
//! `tests/server_integration_tests.rs` for the embeddings handler (no real
//! model or network socket needed for either).

use std::path::{Path, PathBuf};
use std::process::Command;

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

// ── Shared helpers ──────────────────────────────────────────────────────────

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai_cli_surface_test_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn run_bin(args: &[&str]) -> std::process::Output {
    // `main.rs` auto-loads a `.env` by walking UP from the process's
    // current directory (documented, intentional behavior); this
    // worktree lives *under* the main checkout, which has its own
    // developer `.env` setting `OXI_DIT_GGUF`/`OXI_VAE_WEIGHTS`/etc, so a
    // child spawned with this crate's own directory as its cwd would
    // silently inherit those values regardless of `env_remove` (that only
    // clears the *process* environment, not a `.env` file `dotenvy`
    // would still find on disk). Running from a bare temp directory keeps
    // these tests hermetic to the flags/env they actually set.
    Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(args)
        .current_dir(std::env::temp_dir())
        .env_remove("OXI_MODEL")
        .env_remove("OXI_TOKENIZER")
        .env_remove("OXI_DIT_GGUF")
        .env_remove("OXI_VAE_WEIGHTS")
        .env_remove("OXI_TE_4BIT")
        .env_remove("OXI_TE_WEIGHTS")
        .env_remove("OXI_TE_TOKENIZER_DIR")
        .env_remove("XDG_DATA_HOME")
        .output()
        .expect("failed to spawn oxibonsai binary")
}

fn stdout_of(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr_of(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Build a minimal, well-formed GGUF byte buffer with the given
/// `architecture` and a `token_embd.weight` tensor of shape
/// `[hidden, vocab]` (small enough to construct by hand), for testing
/// `info`/`validate`'s honesty logic without a multi-GB real model.
fn build_minimal_gguf(architecture: &str, extra: &[(&str, MetadataWriteValue)]) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str(architecture.to_string()),
    );
    for (key, value) in extra {
        writer.add_metadata(key, value.clone());
    }
    let hidden = 8u64;
    let vocab = 4u64;
    let embd_data: Vec<u8> = (0..(hidden * vocab))
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![hidden, vocab],
        tensor_type: TensorType::F32,
        data: embd_data,
    });
    writer.to_bytes().expect("build minimal GGUF")
}

fn write_gguf(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write gguf fixture");
    path
}

// ── tokenizer (T-14) ─────────────────────────────────────────────────────────

#[test]
fn tokenizer_help_advertises_download_and_info() {
    let output = run_bin(&["tokenizer", "--help"]);
    assert!(
        output.status.success(),
        "`tokenizer --help` should succeed; stderr={}",
        stderr_of(&output)
    );
    let stdout = stdout_of(&output);
    assert!(
        stdout.contains("download"),
        "missing 'download' subcommand: {stdout}"
    );
    assert!(
        stdout.contains("info"),
        "missing 'info' subcommand: {stdout}"
    );
}

#[test]
fn tokenizer_info_reports_real_vocab_and_added_token_counts() {
    let dir = scratch_dir("tokenizer_info");
    let path = dir.join("tokenizer.json");
    // 3 vocab entries, 2 added tokens, model type "BPE" — every field
    // `tokenizer info` reads (see src/cli/cmd_tokenizer.rs).
    std::fs::write(
        &path,
        r#"{
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}},
            "added_tokens": [{"id": 3, "content": "<x>"}, {"id": 4, "content": "<y>"}]
        }"#,
    )
    .expect("write tokenizer.json fixture");

    let output = run_bin(&["tokenizer", "info", "--path", path.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "`tokenizer info` should succeed; stderr={}",
        stderr_of(&output)
    );
    let stdout = stdout_of(&output);
    assert!(
        stdout.contains("BPE"),
        "should report the real model type: {stdout}"
    );
    assert!(stdout.contains('3'), "should report vocab size 3: {stdout}");
    assert!(
        stdout.contains('2'),
        "should report 2 added tokens: {stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tokenizer_info_errors_on_missing_file() {
    let dir = scratch_dir("tokenizer_info_missing");
    let path = dir.join("does-not-exist.json");
    let output = run_bin(&["tokenizer", "info", "--path", path.to_str().unwrap()]);
    assert!(
        !output.status.success(),
        "must fail for a missing tokenizer.json"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── benchmark (T-14 / cli-13) ────────────────────────────────────────────────

#[test]
fn benchmark_requires_model_or_synthetic() {
    // cli-13: the untrained synthetic toy model must never be the silent
    // default; without --model, --synthetic must be explicit.
    let output = run_bin(&["benchmark", "--tokens", "1", "--warmup", "0"]);
    assert!(
        !output.status.success(),
        "benchmark with neither --model nor --synthetic must fail"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("--synthetic") && stderr.contains("--model"),
        "error must name both flags; got: {stderr}"
    );
}

#[test]
fn benchmark_synthetic_runs_and_prints_a_warning() {
    let output = run_bin(&["benchmark", "--synthetic", "--tokens", "3", "--warmup", "0"]);
    assert!(
        output.status.success(),
        "`benchmark --synthetic` should succeed; stdout={} stderr={}",
        stdout_of(&output),
        stderr_of(&output)
    );
    let stdout = stdout_of(&output);
    assert!(
        stdout.contains("WARNING") && stdout.to_lowercase().contains("not representative"),
        "--synthetic must warn its numbers are not representative of a real model: {stdout}"
    );
    assert!(
        stdout.contains("Benchmark:"),
        "should still report throughput: {stdout}"
    );
}

#[test]
fn benchmark_model_flag_with_missing_file_fails_honestly() {
    let dir = scratch_dir("benchmark_missing_model");
    let path = dir.join("nope.gguf");
    let output = run_bin(&[
        "benchmark",
        "--model",
        path.to_str().unwrap(),
        "--tokens",
        "1",
    ]);
    assert!(!output.status.success(), "a missing --model file must fail");
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains(path.to_str().unwrap()),
        "error should name the missing path; got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── image (T-14 / deps-08 / RAG-EVAL-IMG-02) ─────────────────────────────────

#[test]
fn image_help_never_advertises_a_tmp_default() {
    let output = run_bin(&["image", "--help"]);
    assert!(output.status.success());
    let stdout = stdout_of(&output);
    assert!(
        !stdout.contains("/tmp"),
        "`image --help` must not document a /tmp default (deps-08): {stdout}"
    );
}

#[test]
fn image_without_dit_or_env_errors_naming_the_flag_never_defaulting_to_tmp() {
    let dir = scratch_dir("image_no_dit");
    let out_path = dir.join("out.png");
    let output = run_bin(&[
        "image",
        "--prompt",
        "a cat",
        "--out",
        out_path.to_str().unwrap(),
    ]);
    assert!(
        !output.status.success(),
        "image with no --dit/env/data-dir default must fail rather than silently using /tmp"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("--dit"),
        "error must name --dit; got: {stderr}"
    );
    assert!(
        !stderr.contains("/tmp/parity.gguf") && !stderr.contains("/tmp/bonsai_golden"),
        "error must never mention the removed /tmp defaults; got: {stderr}"
    );
    assert!(
        !out_path.exists(),
        "no output file should be written when path resolution fails"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── base64 embeddings (T-14) ─────────────────────────────────────────────────
//
// In-process, mirroring `tests/server_integration_tests.rs`'s style: no real
// model or network socket is needed to exercise the embeddings handler.

#[cfg(feature = "server")]
mod embeddings_base64 {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use oxibonsai_runtime::embeddings::create_embeddings_router;
    use serde_json::Value;
    use tower::ServiceExt;

    async fn post_embeddings(body: Value) -> Value {
        let app = create_embeddings_router(16);
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_vec(&body).expect("serialize request"),
            ))
            .expect("build request");
        let resp = app.oneshot(req).await.expect("oneshot should succeed");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/embeddings must return 200"
        );
        let bytes = resp
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("response must be valid JSON")
    }

    #[tokio::test]
    async fn encoding_format_base64_serialises_embedding_as_a_string() {
        let json = post_embeddings(serde_json::json!({
            "input": "hello world",
            "encoding_format": "base64",
        }))
        .await;
        let embedding = &json["data"][0]["embedding"];
        assert!(
            embedding.is_string(),
            "encoding_format: \"base64\" must serialise data[0].embedding as a JSON string, \
             not a float array; got: {embedding}"
        );
        // Every character of a real base64 string is from the RFC 4648
        // alphabet — this would fail immediately if the field were ever a
        // JSON array (which does not even have a meaningful `.as_str()`).
        let s = embedding.as_str().expect("checked is_string above");
        assert!(
            !s.is_empty(),
            "a non-empty input must not produce an empty encoded embedding"
        );
        assert!(s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'='));
    }

    #[tokio::test]
    async fn default_encoding_format_is_still_a_float_array() {
        // Regression guard for the base64 fix above: the default (no
        // `encoding_format`, or `"float"`) must remain a plain array,
        // never accidentally switched to base64.
        let json = post_embeddings(serde_json::json!({ "input": "hello world" })).await;
        let embedding = &json["data"][0]["embedding"];
        assert!(
            embedding.is_array(),
            "default encoding_format must serialise as a JSON array; got: {embedding}"
        );
    }
}

// ── cli-02 / gatekeeper REQUIRED #1: honest info/validate ────────────────────

#[test]
fn validate_rejects_a_non_language_model_architecture() {
    let dir = scratch_dir("validate_clip");
    let bytes = build_minimal_gguf("clip", &[]);
    let path = write_gguf(&dir, "clip.gguf", &bytes);

    let output = run_bin(&["validate", "--model", path.to_str().unwrap()]);
    assert!(
        !output.status.success(),
        "a non-language-model architecture must not validate as OK"
    );
    let stdout = stdout_of(&output);
    assert!(
        !stdout.contains("Validation: OK"),
        "must never print 'Validation: OK' for a 'clip' architecture file: {stdout}"
    );
    assert!(
        stdout.contains("FAILED"),
        "should report FAILED with a reason: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn info_never_fabricates_config_numbers_for_a_non_language_model_file() {
    let dir = scratch_dir("info_clip");
    let bytes = build_minimal_gguf("clip", &[]);
    let path = write_gguf(&dir, "clip.gguf", &bytes);

    let output = run_bin(&["info", "--model", path.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "`info` should still succeed (it describes, it does not gate); stderr={}",
        stderr_of(&output)
    );
    let stdout = stdout_of(&output);
    // The old bug fabricated the 8B defaults (36 layers, hidden 4096, 32
    // heads) for any file, regardless of architecture.
    assert!(
        !stdout.contains("Layers:       36") && !stdout.contains("Hidden size:  4096"),
        "must not print the fabricated 8B defaults for a 'clip' file: {stdout}"
    );
    assert!(
        stdout.contains("Layers:       -") || stdout.contains("Layers:       -\n"),
        "a genuinely absent field must display as '-': {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn validate_accepts_a_well_formed_known_architecture_file() {
    let dir = scratch_dir("validate_ok");
    // `build_minimal_gguf` only adds `token_embd.weight`; validate also
    // requires an output norm/head tensor for its tensor-coverage check
    // (core-gguf-03), so this builds both tensors directly rather than
    // using that helper.
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata("qwen3.block_count", MetadataWriteValue::U32(2));
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(8));
    writer.add_metadata("qwen3.attention.head_count", MetadataWriteValue::U32(2));
    let embd_data: Vec<u8> = (0..32)
        .flat_map(|i: i32| (i as f32 * 0.01).to_le_bytes())
        .collect();
    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![8, 4],
        tensor_type: TensorType::F32,
        data: embd_data,
    });
    let norm_data: Vec<u8> = (0..8)
        .flat_map(|i: i32| (i as f32 * 0.01).to_le_bytes())
        .collect();
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![8],
        tensor_type: TensorType::F32,
        data: norm_data,
    });
    let writer_bytes = writer.to_bytes().expect("build gguf");
    let path = write_gguf(&dir, "qwen3.gguf", &writer_bytes);

    let output = run_bin(&["validate", "--model", path.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "a well-formed, known-architecture, fully-covered file should validate OK; \
         stdout={} stderr={}",
        stdout_of(&output),
        stderr_of(&output)
    );
    assert!(stdout_of(&output).contains("Validation: OK"));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── cli-19: build-info ───────────────────────────────────────────────────────

#[test]
fn build_info_runs_with_no_model_and_reports_features() {
    let output = run_bin(&["build-info"]);
    assert!(
        output.status.success(),
        "build-info must never require a model; stderr={}",
        stderr_of(&output)
    );
    let stdout = stdout_of(&output);
    assert!(stdout.contains("Build features"), "{stdout}");
    assert!(stdout.contains("Detected runtime tier"), "{stdout}");
    assert!(stdout.contains("Git commit"), "{stdout}");
}

// ── cli-23: stale "1-bit ... Bonsai-8B" about string ─────────────────────────

#[test]
fn top_level_help_no_longer_says_1_bit_bonsai_8b() {
    let output = run_bin(&["--help"]);
    assert!(output.status.success());
    let stdout = stdout_of(&output);
    assert!(
        !stdout.contains("1-bit LLM inference engine for Bonsai-8B"),
        "top-level --help must not carry the stale 1-bit/Bonsai-8B description: {stdout}"
    );
}

// ── cli-04: --config applies every section, errors honestly ─────────────────

#[test]
fn config_unknown_key_is_a_hard_error() {
    let dir = scratch_dir("config_unknown_key");
    let path = dir.join("bad.toml");
    std::fs::write(&path, "[sampling]\ntemperture = 0.5\n").expect("write config");

    let output = run_bin(&["--config", path.to_str().unwrap(), "run", "--prompt", "hi"]);
    assert!(
        !output.status.success(),
        "a typo'd config key must be rejected"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("temperture"),
        "error should name the unknown key; got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn config_missing_file_is_a_hard_error_naming_the_path() {
    let dir = scratch_dir("config_missing");
    let path = dir.join("does-not-exist.toml");

    let output = run_bin(&["--config", path.to_str().unwrap(), "run", "--prompt", "hi"]);
    assert!(
        !output.status.success(),
        "a missing --config file must fail"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains(path.to_str().unwrap()),
        "error must name the missing config path; got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn config_unrecognized_section_is_a_hard_error() {
    let dir = scratch_dir("config_unknown_section");
    let path = dir.join("bad.toml");
    std::fs::write(&path, "[imagen]\ndit_path = \"x\"\n").expect("write config");

    let output = run_bin(&["--config", path.to_str().unwrap(), "run", "--prompt", "hi"]);
    assert!(
        !output.status.success(),
        "an unknown [section] must be rejected"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("imagen"),
        "error should name the unknown section; got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── cli-12: sampling argument validation ─────────────────────────────────────

#[test]
fn run_rejects_negative_temperature_at_the_cli_boundary() {
    let output = run_bin(&["run", "--prompt", "hi", "--temperature", "-1"]);
    assert!(
        !output.status.success(),
        "negative temperature must be rejected"
    );
}

#[test]
fn run_rejects_top_p_above_one_at_the_cli_boundary() {
    let output = run_bin(&["run", "--prompt", "hi", "--top-p", "5.0"]);
    assert!(!output.status.success(), "top-p > 1.0 must be rejected");
}

#[test]
fn run_rejects_zero_max_tokens_at_the_cli_boundary() {
    let output = run_bin(&["run", "--prompt", "hi", "--max-tokens", "0"]);
    assert!(!output.status.success(), "max-tokens == 0 must be rejected");
}

#[test]
fn config_sourced_negative_temperature_is_rejected_same_as_the_cli_flag() {
    // cli-12's range checks live in clap's `value_parser`, which only ever
    // sees a `--flag` value; a `[sampling]` value from `--config` is
    // parsed by this crate's own TOML scanner (plain `str::parse`, no
    // clap involved) and must be re-validated on that path too, or a
    // config file reopens exactly the "negative temperature inverts the
    // distribution" bug on a different door than the CLI-flag one it was
    // closed on.
    let dir = scratch_dir("config_bad_temperature");
    let path = dir.join("bad.toml");
    std::fs::write(&path, "[sampling]\ntemperature = -5.0\n").expect("write config");

    let output = run_bin(&["--config", path.to_str().unwrap(), "run", "--prompt", "hi"]);
    assert!(
        !output.status.success(),
        "a config-sourced out-of-range temperature must be rejected, not silently applied"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("temperature"),
        "error should mention temperature; got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn config_sourced_top_p_above_one_is_rejected() {
    let dir = scratch_dir("config_bad_top_p");
    let path = dir.join("bad.toml");
    std::fs::write(&path, "[sampling]\ntop_p = 5.0\n").expect("write config");

    let output = run_bin(&["--config", path.to_str().unwrap(), "run", "--prompt", "hi"]);
    assert!(
        !output.status.success(),
        "a config-sourced top_p > 1.0 must be rejected"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── cli-04: [model] section actually threads through to the command ─────────

#[test]
fn config_model_path_section_is_actually_applied() {
    // Proves `[model].model_path` reaches `run` (not just parses): the
    // command must fail on the *configured* (nonexistent) path, not on
    // "no model" — that would mean the section was silently ignored, the
    // original cli-04 bug.
    let dir = scratch_dir("config_model_path");
    let configured_model = dir.join("configured-model.gguf");
    let config_path = dir.join("cfg.toml");
    std::fs::write(
        &config_path,
        format!(
            "[model]\nmodel_path = \"{}\"\n",
            configured_model.to_str().unwrap().replace('\\', "\\\\")
        ),
    )
    .expect("write config");

    let output = run_bin(&[
        "--config",
        config_path.to_str().unwrap(),
        "run",
        "--prompt",
        "hi",
    ]);
    assert!(
        !output.status.success(),
        "the configured model path does not exist"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains(configured_model.to_str().unwrap()),
        "error must name the [model].model_path-configured path (proving the section was \
         actually applied, not silently discarded); got: {stderr}"
    );
    assert!(
        !stderr.contains("no model: pass --model"),
        "must not fall through to the 'no model at all' error when [model].model_path was set; \
         got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── cli-04: [server] / [observability] sections are actually applied ────────
//
// `[model]` and `[sampling]` are covered above via `run`; `serve` fails
// fast on "no model" (before ever touching a socket, see
// `cmd_serve::run`'s first check), and `mod.rs` resolves and logs
// `[server].host`/`[server].port` — via `tracing::info!(host, port,
// "resolved server bind address")` — *before* calling into `cmd_serve::run`
// at all, so that log line is a real, observable signal that the section
// reached `serve` rather than being silently discarded. `[observability]`
// is proven the same way: it controls whether that very log line renders
// as JSON.

#[test]
fn config_server_section_is_actually_applied() {
    let dir = scratch_dir("config_server_section");
    let config_path = dir.join("cfg.toml");
    std::fs::write(&config_path, "[server]\nport = 19246\n").expect("write config");

    let output = run_bin(&["--config", config_path.to_str().unwrap(), "serve"]);
    assert!(
        !output.status.success(),
        "serve must still fail: no --model/OXI_MODEL was given"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("no model"),
        "must fail on the missing model, proving [server] parsing itself did not error; \
         got: {stderr}"
    );
    let combined = format!("{}{}", stdout_of(&output), stderr);
    assert!(
        combined.contains("19246"),
        "the [server].port-configured value must reach `serve` (logged before the \
         'no model' failure), proving the section was actually applied rather than \
         silently discarded; got stdout+stderr: {combined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn config_observability_section_is_actually_applied() {
    let dir = scratch_dir("config_observability_section");
    let config_path = dir.join("cfg.toml");
    std::fs::write(&config_path, "[observability]\njson_logs = true\n").expect("write config");

    let output = run_bin(&["--config", config_path.to_str().unwrap(), "serve"]);
    assert!(
        !output.status.success(),
        "serve must still fail: no --model/OXI_MODEL was given"
    );
    let combined = format!("{}{}", stdout_of(&output), stderr_of(&output));
    assert!(
        combined.contains("\"level\":\"INFO\"") || combined.contains("\"level\": \"INFO\""),
        "[observability].json_logs = true must make the pre-dispatch 'resolved server bind \
         address' log line render as JSON instead of the human-readable default, proving the \
         section was actually applied; got stdout+stderr: {combined}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
