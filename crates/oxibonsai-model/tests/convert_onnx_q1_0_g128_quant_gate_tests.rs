//! Regression tests for the `--quant` validation gate in
//! `convert_onnx_to_gguf` (ONNX → GGUF path of cli-facade-01).
//!
//! A full end-to-end ONNX conversion test would require constructing a
//! synthetic `MatMulNBits`-quantized ONNX protobuf graph, which is out of
//! proportion for this fix (the ONNX pipeline shares its Q1_0_g128 tensor
//! encoding with the already end-to-end-tested safetensors path in
//! `convert_q1_0_g128_tests.rs` — see `TensorKind::Weight if quant ==
//! "q1_0_g128"` in `convert::onnx::convert_onnx_to_gguf`). These tests
//! instead lock down the validation gate itself: `quant` is checked *before*
//! the `.onnx` file is even opened, so a bogus path deterministically proves
//! which branch ran from the error variant alone.

use std::path::PathBuf;

use oxibonsai_model::{convert_onnx_to_gguf, OnnxImportError};

fn nonexistent_onnx_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "oxibonsai_convert_onnx_{tag}_{}_{}.onnx",
        std::process::id(),
        nanos
    ))
}

#[test]
fn convert_onnx_to_gguf_rejects_unsupported_quant_before_touching_the_file() {
    // The input path does not exist. If the quant guard did not run first,
    // this would surface as `OnnxImportError::Io`, not `Other`.
    let bogus_path = nonexistent_onnx_path("bad_quant");
    let out_path = std::env::temp_dir().join("oxibonsai_convert_onnx_bad_quant_out.gguf");

    let err = convert_onnx_to_gguf(&bogus_path, &out_path, "q4_0")
        .expect_err("q4_0 is not a supported convert --quant format");

    match err {
        OnnxImportError::Other(msg) => {
            assert!(
                msg.contains("q4_0") && msg.contains("tq2_0_g128") && msg.contains("q1_0_g128"),
                "error message should name the rejected value and both supported formats, got: {msg}"
            );
        }
        other => panic!(
            "expected OnnxImportError::Other from the quant validation gate \
             (proving it ran before file I/O), got: {other:?}"
        ),
    }
}

#[test]
fn convert_onnx_to_gguf_accepts_q1_0_g128_past_the_quant_gate() {
    // `q1_0_g128` must now pass the validation gate and proceed to opening
    // the (nonexistent) file, surfacing as `OnnxImportError::Io` rather than
    // being rejected as an unsupported format — this is the exact
    // regression for cli-facade-01 on the ONNX path.
    let bogus_path = nonexistent_onnx_path("good_quant");
    let out_path = std::env::temp_dir().join("oxibonsai_convert_onnx_good_quant_out.gguf");

    let err = convert_onnx_to_gguf(&bogus_path, &out_path, "q1_0_g128")
        .expect_err("nonexistent .onnx path must still fail, but past the quant gate");

    match err {
        OnnxImportError::Io { path, .. } => {
            assert_eq!(path, bogus_path, "I/O error must reference the input path");
        }
        other => panic!(
            "expected OnnxImportError::Io (quant gate passed, file I/O failed), got: {other:?}"
        ),
    }
}
