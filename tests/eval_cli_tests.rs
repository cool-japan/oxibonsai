//! Regression test for finding cli-facade-03: the `eval` Cargo feature
//! pulled in the full `oxibonsai-eval` crate (perplexity, MMLU, BLEU, chrF,
//! calibration, bootstrap CIs, ...) with zero CLI surface — `--features
//! eval` compiled real evaluation code that was unreachable from the
//! command line.
//!
//! `oxibonsai eval` is now a real subcommand: it loads a JSONL dataset of
//! `{"input": <prompt>, "expected_output": <reference>}` examples, runs the
//! model's greedy generation on each prompt, and scores the corpus with
//! ROUGE-1/2/L (`oxibonsai_eval::CorpusRouge`). These tests exercise the
//! parts reachable without a real GGUF model/tokenizer (dataset loading
//! happens *before* the model is opened, by design, so bad `--dataset`
//! input fails fast) plus the feature-gating contract itself.

#[cfg(feature = "eval")]
use std::path::PathBuf;
use std::process::Command;

#[cfg(feature = "eval")]
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai_eval_cli_test_{tag}_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[cfg(feature = "eval")]
#[test]
fn eval_help_advertises_the_real_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(["eval", "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        output.status.success(),
        "`eval --help` should succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in ["--dataset", "--model", "--max-tokens", "--report-json"] {
        assert!(
            stdout.contains(flag),
            "`eval --help` must advertise {flag}; got:\n{stdout}"
        );
    }
}

#[cfg(feature = "eval")]
#[test]
fn eval_rejects_missing_dataset_before_touching_the_model() {
    // The dataset is read before the (possibly multi-GB) model file is
    // opened, so a bad --dataset path must fail fast, mentioning the path,
    // without requiring a real --model at all reaching the mmap stage.
    let dataset = scratch_dir("missing_dataset").join("nope.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "eval",
            "--model",
            "/dev/null",
            "--dataset",
            dataset.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        !output.status.success(),
        "eval must fail when --dataset does not exist"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(dataset.to_str().unwrap()),
        "error must mention the missing dataset path; got:\n{stderr}"
    );
}

#[cfg(feature = "eval")]
#[test]
fn eval_rejects_empty_dataset() {
    let dir = scratch_dir("empty_dataset");
    let dataset = dir.join("empty.jsonl");
    std::fs::write(&dataset, "").expect("write empty dataset");

    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "eval",
            "--model",
            "/dev/null",
            "--dataset",
            dataset.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        !output.status.success(),
        "eval must reject a dataset with zero examples"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no examples") || stderr.contains("empty"),
        "error must explain the dataset is empty; got:\n{stderr}"
    );
}

#[cfg(feature = "eval")]
#[test]
fn eval_rejects_malformed_dataset_line() {
    let dir = scratch_dir("malformed_dataset");
    let dataset = dir.join("bad.jsonl");
    std::fs::write(&dataset, "not json at all\n").expect("write malformed dataset");

    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "eval",
            "--model",
            "/dev/null",
            "--dataset",
            dataset.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        !output.status.success(),
        "eval must reject a dataset line that is not valid JSON"
    );
}

/// Without the `eval` feature, `eval` must not exist as a subcommand at all
/// (rather than silently being accepted and doing nothing) — invoking it
/// must be a clap "unrecognized subcommand" error.
#[cfg(not(feature = "eval"))]
#[test]
fn eval_subcommand_absent_without_eval_feature() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(["eval", "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        !output.status.success(),
        "`eval` must not be a recognized subcommand without the `eval` feature"
    );
}
