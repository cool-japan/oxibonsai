//! Regression test for finding cli-facade-01: `convert --quant q1_0_g128`
//! was advertised by `--help` and `docs/CLI.md` but always failed at
//! runtime — `crates/oxibonsai-model/src/convert/{mod,onnx/mod}.rs` only
//! implemented `tq2_0_g128`.
//!
//! Wave 1 closed the honesty gap by removing `q1_0_g128` from `--help`.
//! Wave 2 implements Q1\_0\_g128 for real in the converter (reusing the
//! shipping `crate::quantize::quantize_q1_0_g128` encoder and its canonical
//! sign convention), so this test now pins the *positive* contract restored
//! alongside it:
//!   1. `--help` advertises `q1_0_g128` again (it is genuinely supported now).
//!   2. `convert --quant q1_0_g128` against a real HF-shaped safetensors
//!      directory succeeds end to end through the compiled `oxibonsai`
//!      binary, producing a GGUF whose weight tensors are Q1_0_g128 and
//!      whose norm tensors stay FP32.
//!   3. A genuinely unsupported format is still rejected clearly, with no
//!      output file written.
//!
//! `crates/oxibonsai-model/src/convert` tests validate the encoding itself
//! (sign convention, round-trip, FP32 exceptions) at the library level in
//! `crates/oxibonsai-model/tests/convert_q1_0_g128_tests.rs`; this file
//! covers the CLI surface (`--help` text + subprocess exit code / file
//! output) that only the compiled binary can exercise.

use std::path::{Path, PathBuf};
use std::process::Command;

// ─── Scratch directory + synthetic model helpers ───────────────────────────

fn scratch_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "oxibonsai_convert_quant_test_{tag}_{}_{}",
        std::process::id(),
        nanos
    ));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

const HIDDEN: usize = 128;

/// Deterministic, mixed-sign f32 pattern (avoids an all-zero group, which
/// would make the Q1_0_g128 sign-bit check trivially pass).
fn pattern(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (seed as f32) * 0.017 + (i as f32) * 0.031;
            x.sin() * (1.0 + (i % 7) as f32 * 0.1)
        })
        .collect()
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Hand-build a minimal `model.safetensors` file (no `safetensors` crate
/// dependency needed at this scope): an 8-byte little-endian header length,
/// a JSON header describing each tensor's dtype/shape/byte-offsets, and the
/// raw tensor bytes concatenated in the same order the offsets describe.
/// Tensor ordering in the JSON text is irrelevant — the reader sorts by
/// `data_offsets` before validating contiguity (see `safetensors` crate
/// `Metadata::try_from`), so any deterministic order works as long as the
/// offsets and the byte layout agree, which this helper guarantees by
/// construction.
fn write_synthetic_safetensors(dir: &Path, tensors: &[(&str, Vec<f32>, Vec<usize>)]) {
    let mut offset = 0usize;
    let mut entries = Vec::with_capacity(tensors.len());
    let mut payload = Vec::new();
    for (name, data, shape) in tensors {
        let bytes = f32_bytes(data);
        let start = offset;
        let end = offset + bytes.len();
        offset = end;
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_str}],\"data_offsets\":[{start},{end}]}}"
        ));
        payload.extend_from_slice(&bytes);
    }
    let header = format!("{{{}}}", entries.join(","));
    let header_bytes = header.into_bytes();

    let mut file = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&payload);

    std::fs::write(dir.join("model.safetensors"), file).expect("write model.safetensors");
}

/// Write a minimal single-layer Qwen3-shaped HF model directory
/// (`config.json` + `model.safetensors`) that the real converter accepts.
fn write_synthetic_hf_model(dir: &Path) {
    let config = serde_json::json!({
        "num_hidden_layers": 1,
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "vocab_size": HIDDEN,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).expect("serialize config.json"),
    )
    .expect("write config.json");

    let tensors: Vec<(&str, Vec<f32>, Vec<usize>)> = vec![
        (
            "model.embed_tokens.weight",
            pattern(1, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
        ("model.norm.weight", pattern(2, HIDDEN), vec![HIDDEN]),
        (
            "model.layers.0.input_layernorm.weight",
            pattern(3, HIDDEN),
            vec![HIDDEN],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            pattern(4, HIDDEN),
            vec![HIDDEN],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            pattern(5, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            pattern(6, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            pattern(7, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            pattern(8, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            pattern(9, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            pattern(10, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            pattern(11, HIDDEN * HIDDEN),
            vec![HIDDEN, HIDDEN],
        ),
    ];
    write_synthetic_safetensors(dir, &tensors);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn convert_help_advertises_q1_0_g128_again() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(["convert", "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        output.status.success(),
        "`convert --help` should succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("q1_0_g128"),
        "`convert --help` must advertise q1_0_g128 as a supported --quant value \
         now that the converter implements it; got:\n{stdout}"
    );
    assert!(
        stdout.contains("tq2_0_g128"),
        "`convert --help` must still advertise the default tq2_0_g128 format; got:\n{stdout}"
    );
}

#[test]
fn convert_quant_q1_0_g128_succeeds_end_to_end_via_cli() {
    let from_dir = scratch_dir("from_ok");
    write_synthetic_hf_model(&from_dir);
    let to_path = scratch_dir("to_ok").join("out.gguf");

    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "convert",
            "--from",
            from_dir.to_str().expect("utf8 path"),
            "--to",
            to_path.to_str().expect("utf8 path"),
            "--quant",
            "q1_0_g128",
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");

    assert!(
        output.status.success(),
        "convert --quant q1_0_g128 must now succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        to_path.exists(),
        "a GGUF file must be written when q1_0_g128 conversion succeeds"
    );

    // Re-open with the real GGUF reader and verify tensor types.
    let bytes = std::fs::read(&to_path).expect("read converted GGUF file");
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes)
        .expect("re-parse the GGUF file the CLI just wrote");

    let norm = gguf
        .tensors
        .require("output_norm.weight")
        .expect("output_norm.weight present");
    assert_eq!(
        norm.tensor_type,
        oxibonsai_core::GgufTensorType::F32,
        "norm tensors must remain FP32"
    );

    let weight = gguf
        .tensors
        .require("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight present");
    assert_eq!(
        weight.tensor_type,
        oxibonsai_core::GgufTensorType::Q1_0_g128,
        "weight tensors must be quantized to Q1_0_g128 when --quant q1_0_g128 is passed"
    );
}

#[test]
fn convert_quant_unsupported_format_still_fails_clearly() {
    let from_dir = scratch_dir("from_bad");
    write_synthetic_hf_model(&from_dir);
    let to_path = scratch_dir("to_bad").join("out.gguf");

    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "convert",
            "--from",
            from_dir.to_str().expect("utf8 path"),
            "--to",
            to_path.to_str().expect("utf8 path"),
            "--quant",
            "not_a_real_format",
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");

    assert!(
        !output.status.success(),
        "convert --quant not_a_real_format must fail (genuinely unsupported)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not_a_real_format") && stderr.contains("unsupported"),
        "error must clearly explain the format is unsupported; got:\n{stderr}"
    );
    assert!(
        !to_path.exists(),
        "no output GGUF should be written when the format is rejected"
    );
}
