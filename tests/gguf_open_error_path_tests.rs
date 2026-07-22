//! Regression test for finding cli-facade-02: GGUF-open errors on
//! `run`/`chat`/`info`/`quantize`/`validate` used to lose the file path
//! entirely — `mmap_gguf_file(..)?` propagated a bare `std::io::Error`
//! ("memory mapping failed: No such file or directory (os error 2)") with no
//! indication of *which* file was missing.
//!
//! Each subcommand below now wraps the mmap call with
//! `.map_err(|e| anyhow!("failed to open model '{model}': {e}"))` (mirroring
//! the pre-existing `quantize`/`convert` "input file does not exist" pattern),
//! so the offending path always appears in the error text.

use std::process::Command;

/// A path that is guaranteed not to exist, unique per test process so
/// parallel runs never collide (project policy: use `std::env::temp_dir()`,
/// never a hard-coded `/tmp` literal).
fn missing_model_path(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "oxibonsai_missing_model_{tag}_{}.gguf",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

fn assert_error_mentions_path(args: &[&str], model_path: &str) {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(args)
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        !output.status.success(),
        "command should fail against a nonexistent model path; args={args:?}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(model_path),
        "error output must mention the offending model path {model_path:?}; got:\n{stderr}"
    );
}

#[test]
fn info_error_includes_model_path() {
    let model = missing_model_path("info");
    assert_error_mentions_path(&["info", "--model", &model], &model);
}

#[test]
fn validate_error_includes_model_path() {
    let model = missing_model_path("validate");
    assert_error_mentions_path(&["validate", "--model", &model], &model);
}

#[test]
fn run_error_includes_model_path() {
    let model = missing_model_path("run");
    assert_error_mentions_path(
        &[
            "run",
            "--model",
            &model,
            "--prompt",
            "hi",
            "--max-tokens",
            "1",
        ],
        &model,
    );
}

#[test]
fn chat_error_includes_model_path() {
    // The mmap-open failure happens before chat reads anything from stdin,
    // so this terminates immediately without requiring interactive input.
    let model = missing_model_path("chat");
    assert_error_mentions_path(&["chat", "--model", &model], &model);
}

#[test]
fn quantize_error_includes_input_path() {
    let input = missing_model_path("quantize_input");
    let output = std::env::temp_dir()
        .join(format!(
            "oxibonsai_quantize_out_{}.gguf",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned();
    assert_error_mentions_path(
        &["quantize", "--input", &input, "--output", &output],
        &input,
    );
}
