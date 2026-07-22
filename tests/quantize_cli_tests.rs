//! Integration tests for the `oxibonsai quantize` CLI subcommand.
//!
//! These tests build a tiny synthetic GGUF model on disk, invoke the real
//! `oxibonsai` binary (`quantize --input ... --output ... --format ...`),
//! and verify the command actually writes a loadable, correctly re-quantized
//! GGUF file at `--output` — rather than merely printing plausible-looking
//! numbers (the behavior before this test was added, see finding #35).

use std::path::PathBuf;
use std::process::Command;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::GgufTensorType;
use oxibonsai_model::export::{export_to_gguf, ExportConfig, ExportFormat, WeightTensor};

/// Build a minimal but valid synthetic GGUF file at `path`, containing one
/// "embedding" tensor (which the default fp32-exception list keeps in f32)
/// and one "weight" tensor eligible for quantization.
fn write_synthetic_gguf(path: &std::path::Path) {
    let tensors = vec![
        WeightTensor::new(
            "token_embd.weight",
            (0..32).map(|i| i as f32 * 0.01).collect(),
            vec![8, 4],
        ),
        WeightTensor::new(
            "blk.0.attn_q.weight",
            (0..64).map(|i| (i as f32 * 0.037).sin()).collect(),
            vec![8, 8],
        ),
    ];
    let config = ExportConfig::new(ExportFormat::Float32, "quantize-cli-test-source");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("build synthetic source GGUF");
    std::fs::write(path, bytes).expect("write synthetic source GGUF to disk");
}

/// Unique scratch directory for this test binary's run, under the shared
/// temp dir per project policy (`std::env::temp_dir()`), so parallel test
/// processes never collide.
fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai_quantize_cli_test_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn quantize_cli_writes_a_real_requantized_gguf_file() {
    let dir = scratch_dir("happy_path");
    let input_path = dir.join("source.gguf");
    let output_path = dir.join("quantized.gguf");
    write_synthetic_gguf(&input_path);

    assert!(
        !output_path.exists(),
        "sanity: output must not exist before running the CLI"
    );

    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "quantize",
            "--input",
            input_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
            "--format",
            "q4_0",
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");

    assert!(
        output.status.success(),
        "quantize should succeed; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // The core assertion this test exists for: a real file was written.
    assert!(
        output_path.exists(),
        "quantize must actually write the output file, not just print stats"
    );
    let written_len = std::fs::metadata(&output_path)
        .expect("stat output file")
        .len();
    assert!(written_len > 0, "output GGUF file must be non-empty");

    // The written file must itself be a loadable, well-formed GGUF, with
    // the requested format actually applied.
    let mmap = mmap_gguf_file(&output_path).expect("mmap quantized output");
    let gguf = GgufFile::parse(&mmap).expect("parse quantized output as GGUF");

    let embd = gguf
        .tensors
        .require("token_embd.weight")
        .expect("token_embd.weight must survive re-export");
    assert_eq!(
        embd.tensor_type,
        GgufTensorType::F32,
        "token_embd.weight is a default fp32-exception layer and must stay F32"
    );

    let attn_q = gguf
        .tensors
        .require("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight must survive re-export");
    assert_eq!(
        attn_q.tensor_type,
        GgufTensorType::Q4_0,
        "blk.0.attn_q.weight must actually be re-quantized to the requested Q4_0 format"
    );

    // The reported stats in stdout should reflect real work, not a
    // hardcoded/simulated ratio.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("2 tensors"),
        "stdout should report the real tensor count; got: {stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn quantize_cli_rejects_unsupported_format_without_writing_a_file() {
    let dir = scratch_dir("unsupported_format");
    let input_path = dir.join("source.gguf");
    let output_path = dir.join("quantized.gguf");
    write_synthetic_gguf(&input_path);

    // "f16" is accepted syntactically by nothing in `ExportFormat` — the
    // export pipeline has no F16 quantization arm — so this must fail
    // honestly instead of silently falling back to some other format.
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "quantize",
            "--input",
            input_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
            "--format",
            "f16",
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");

    assert!(
        !output.status.success(),
        "quantize must fail for an unsupported target format instead of fabricating output"
    );
    assert!(
        !output_path.exists(),
        "no output file should be written when the requested format is rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported quantization format"),
        "stderr should explain why the format was rejected; got: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn quantize_cli_errors_on_missing_input_file() {
    let dir = scratch_dir("missing_input");
    let input_path = dir.join("does-not-exist.gguf");
    let output_path = dir.join("quantized.gguf");

    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "quantize",
            "--input",
            input_path.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
            "--format",
            "q4_0",
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");

    assert!(
        !output.status.success(),
        "quantize must fail when the input file does not exist"
    );
    assert!(!output_path.exists());

    let _ = std::fs::remove_dir_all(&dir);
}
