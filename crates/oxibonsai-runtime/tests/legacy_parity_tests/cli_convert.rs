//! The ONNX-export and CLI gates on the real 1.7B model: an ONNX export
//! converted to GGUF and read back through the real tokenizer, and the
//! shipped `oxibonsai eval --task mmlu` subcommand run as a subprocess.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::cli_bin::resolve_cli_binary;
use oxibonsai_testkit::workspace::{find_model, models_dir};

use super::TierEnvGuard;

// ═════════════════════════════════════════════════════════════════════════
//  A real ONNX -> GGUF conversion, then the real tokenizer round-trip
// ═════════════════════════════════════════════════════════════════════════

/// The real `oxibonsai_model::convert_onnx_to_gguf` on the real
/// `Ternary-Bonsai-1.7B-ONNX` export, then the real
/// `oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata` on the converted
/// file's own embedded tokenizer metadata: the whole ONNX-import ->
/// GGUF-write -> tokenizer-load chain, exercised end to end rather than only
/// asserting the converted metadata is present and well-typed.
#[test]
fn onnx_converted_gguf_loads_through_the_real_tokenizer_round_trip() {
    const TEST: &str = "oxibonsai-runtime::legacy_parity_tests::onnx_converted_gguf_loads_through_the_real_tokenizer_round_trip";
    let _tier_env = TierEnvGuard::cleared();
    let require_real_files = std::env::var("OXI_REQUIRE_MODEL_FILES")
        .map(|v| v == "1")
        .unwrap_or(false);
    let onnx_dir = models_dir().join("Ternary-Bonsai-1.7B-ONNX");
    let onnx_path = onnx_dir.join("onnx").join("model_q2.onnx");
    if !onnx_path.is_file() {
        assert!(
            !require_real_files,
            "OXI_REQUIRE_MODEL_FILES=1: {TEST} needs {onnx_path:?}"
        );
        eprintln!("skip {TEST}: {onnx_path:?} not found");
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }
    let gate_start = std::time::Instant::now();

    let out_path = oxibonsai_testkit::temp_path::unique_path("onnx-convert", ".gguf");
    let stats = oxibonsai_model::convert_onnx_to_gguf(&onnx_path, &out_path, "tq2_0_g128")
        .unwrap_or_else(|e| panic!("convert_onnx_to_gguf({onnx_path:?}): {e}"));
    eprintln!("{TEST}: converted {stats:?}");

    let bytes = std::fs::read(&out_path).expect("read the converted GGUF back");
    let _ = std::fs::remove_file(&out_path);
    let gguf = GgufFile::parse(&bytes).expect("the converted GGUF parses");

    // The real tokenizer constructor, on the converted file's own embedded
    // `tokenizer.ggml.*` metadata -- not a key-presence/well-typedness
    // stand-in.
    let tokenizer = oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata(&gguf.metadata)
        .expect("OxiTokenizer::from_gguf_metadata on the converted file");

    let text = "The capital of Japan is Tokyo.";
    let ids = tokenizer
        .encode(text)
        .expect("encode through the real tokenizer");
    assert!(!ids.is_empty(), "encoding real text must produce real ids");
    let decoded = tokenizer
        .decode(&ids)
        .expect("decode through the real tokenizer");
    assert_eq!(
        decoded, text,
        "encode -> decode through the converted file's own tokenizer must round-trip exactly"
    );

    record_executed_timed(Capability::LegacyModels, TEST, gate_start.elapsed());
}

// ─── `oxibonsai eval --task mmlu` against a real model ─────────────────────
//
// The logit-based eval tasks (mmlu, arc-easy/challenge, hellaswag,
// winogrande, boolq, truthfulqa-mc1/mc2, calibration) and teacher-forced
// perplexity compile and pass clippy, but none of `oxibonsai-eval`'s own
// tests exercises `score_choices_logprob` at runtime against a real model
// (that crate has no GGUF of its own). `score_choices_logprob` lives in
// `src/cli/cmd_eval.rs`, a bin-only crate with no `[lib]` target — the same
// shape as `bonsai2_runtime_tests.rs`'s `oxibonsai info` gate — so this
// drives the real, shipped `oxibonsai eval --task mmlu` subcommand as a
// subprocess against a real legacy GGUF and a tiny in-memory MMLU-shaped
// fixture, and asserts the JSON report it writes is well-formed and reports
// a real, in-range score — proof the logit-based path actually executes
// rather than only compiling.
//
// The binary comes from `oxibonsai_testkit::cli_bin::resolve_cli_binary`
// (a pre-built `OXIBONSAI_CLI_BIN`, else one `--all-features` release
// build), resolved before the test takes the environment guard or starts
// its timer: the compile is never billed to the gate, never holds up the
// other real-model gates in this binary, and never overwrites a wider
// binary with a narrower one.

/// A 2-question, MMLU-shaped fixture (`{"id","question","choices",
/// "correct_answer","subject"}` JSONL, per `EvalTask::Mmlu`'s own doc
/// comment; `correct_answer` is the 0-based index into `choices`, per
/// `oxibonsai_eval::dataset::MultipleChoiceQuestion`), written under
/// `std::env::temp_dir()` so this test needs no checked-in fixture asset
/// and no absolute path baked into the repository.
fn write_mmlu_fixture() -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "oxibonsai-cli-mmlu-fixture-{}-{unique}.jsonl",
        std::process::id()
    ));
    let jsonl = concat!(
        r#"{"id":"q1","question":"What is the capital of France?","choices":["#,
        r#""London","Paris","Berlin","Madrid"],"correct_answer":1,"subject":"geography"}"#,
        "\n",
        r#"{"id":"q2","question":"Which planet is known as the Red Planet?","choices":["#,
        r#""Venus","Mars","Jupiter","Saturn"],"correct_answer":1,"subject":"astronomy"}"#,
        "\n",
    );
    std::fs::write(&path, jsonl).expect("write mmlu fixture jsonl to std::env::temp_dir()");
    path
}

#[test]
fn eval_cli_scores_a_real_mmlu_style_dataset_through_score_choices_logprob() {
    const TEST: &str = "oxibonsai-runtime::legacy_parity_tests::\
                        eval_cli_scores_a_real_mmlu_style_dataset_through_score_choices_logprob";

    let Some(model_path) = find_model("Ternary-Bonsai-1.7B.gguf") else {
        eprintln!(
            "skip: Ternary-Bonsai-1.7B.gguf not found under {:?}",
            models_dir()
        );
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };

    // Resolve (and, without OXIBONSAI_CLI_BIN, build) the binary before the
    // environment guard is taken and before the timer starts; the model file
    // is only located above, never opened.
    let bin = resolve_cli_binary()
        .unwrap_or_else(|e| panic!("{TEST}: cannot resolve the oxibonsai CLI binary: {e}"));
    let _tier_env = TierEnvGuard::cleared();
    let gate_start = std::time::Instant::now();

    let dataset_path = write_mmlu_fixture();
    let report_path = std::env::temp_dir().join(format!(
        "oxibonsai-cli-mmlu-report-{}.json",
        std::process::id()
    ));
    // Clean up whatever a previous, possibly-panicked run of this same test
    // process id/host left behind; the write below always replaces it, so
    // this is a courtesy, not a correctness requirement.
    let _ = std::fs::remove_file(&report_path);

    let output = std::process::Command::new(&bin)
        .arg("eval")
        .arg("--task")
        .arg("mmlu")
        .arg("--model")
        .arg(&model_path)
        .arg("--tokenizer")
        .arg(tokenizer_path)
        .arg("--dataset")
        .arg(&dataset_path)
        .arg("--report-json")
        .arg(&report_path)
        .output()
        .unwrap_or_else(|e| panic!("spawning {} failed: {e}", bin.display()));

    let cleanup = || {
        let _ = std::fs::remove_file(&dataset_path);
        let _ = std::fs::remove_file(&report_path);
    };

    if !output.status.success() {
        cleanup();
        panic!(
            "{} eval --task mmlu --model {} exited with {:?}\nstdout:\n{}\nstderr:\n{}",
            bin.display(),
            model_path.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let report_bytes = std::fs::read(&report_path).unwrap_or_else(|e| {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        cleanup();
        panic!(
            "reading eval report at {}: {e}\nstderr:\n{stderr}",
            report_path.display()
        )
    });
    let report: serde_json::Value = serde_json::from_slice(&report_bytes).unwrap_or_else(|e| {
        cleanup();
        panic!(
            "eval --report-json produced non-JSON output: {e}\n{}",
            String::from_utf8_lossy(&report_bytes)
        )
    });
    cleanup();

    let results = report["results"]
        .as_array()
        .unwrap_or_else(|| panic!("eval report has no \"results\" array: {report}"));
    assert!(
        !results.is_empty(),
        "eval --task mmlu produced an empty \"results\" array: {report}"
    );
    // `cmd_eval.rs`'s MMLU arm reports `metric: "mmlu_accuracy"`,
    // `value: result.accuracy_pct` (a PERCENT, 0..=100, `unit: "%"`) and
    // `notes: Some(format!("n={}", result.total))` — NOT the generic
    // `"accuracy"` fraction `EvalReport::add_accuracy` writes for other
    // tasks. Match the real field this task actually writes.
    let accuracy_entry = results
        .iter()
        .find(|r| r["metric"].as_str() == Some("mmlu_accuracy"))
        .unwrap_or_else(|| panic!("no \"mmlu_accuracy\" metric entry in eval report: {report}"));
    let accuracy_pct = accuracy_entry["value"].as_f64().unwrap_or_else(|| {
        panic!("mmlu_accuracy entry has no numeric \"value\": {accuracy_entry}")
    });
    assert!(
        accuracy_pct.is_finite() && (0.0..=100.0).contains(&accuracy_pct),
        "score_choices_logprob produced an out-of-range mmlu_accuracy {accuracy_pct} \
         (report: {report})"
    );
    let notes = accuracy_entry["notes"].as_str().unwrap_or("");
    assert!(
        notes.contains("n=2"),
        "expected the 2-question fixture to score as n=2, got notes={notes:?} (report: {report})"
    );
    eprintln!(
        "measured: oxibonsai eval --task mmlu on Ternary-Bonsai-1.7B.gguf, 2 real \
         questions through score_choices_logprob -> mmlu_accuracy={accuracy_pct}% ({notes})"
    );

    record_executed_timed(Capability::LegacyModels, TEST, gate_start.elapsed());
}
