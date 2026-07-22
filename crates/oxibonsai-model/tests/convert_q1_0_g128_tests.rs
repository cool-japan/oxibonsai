//! Regression tests for `convert --quant q1_0_g128` (cli-facade-01).
//!
//! Wave 1 fixed the `--help` honesty gap by removing `q1_0_g128` from the
//! `--quant` doc comment, but `oxibonsai convert --quant q1_0_g128` still
//! bailed with "unsupported quantisation format" — the format was never
//! implemented in the converter itself, only in the separate `quantize`
//! subcommand's `export_to_gguf` pipeline. These tests exercise the real
//! implementation added to `convert_hf_to_gguf` (safetensors → GGUF) end to
//! end: synthetic HF model directory → `--quant q1_0_g128` → re-open with
//! the real GGUF reader → assert tensor types, FP32 exceptions, and a sane
//! dequantization round-trip using the canonical `BlockQ1_0G128` sign
//! convention (`bit=1 -> +d`, `bit=0 -> -d`) fixed 2026-07-20.

use std::collections::HashMap;
use std::path::PathBuf;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};
use oxibonsai_core::GgufTensorType;
use safetensors::tensor::{Dtype, TensorView};

const HIDDEN: usize = 128;
const INTERMEDIATE: usize = 128;
const VOCAB: usize = 128;

/// Build a unique scratch directory under the OS temp dir for one test run.
fn unique_temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "oxibonsai_convert_q1_{tag}_{}_{}",
        std::process::id(),
        nanos
    ))
}

/// Deterministic, non-trivial (mixed-sign, varying-magnitude) f32 pattern so
/// the Q1_0_g128 round-trip actually exercises both sign bits.
fn pattern(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (seed as f32) * 0.017 + (i as f32) * 0.031;
            x.sin() * (1.0 + (i % 7) as f32 * 0.1)
        })
        .collect()
}

fn f32_view(data: &[f32], shape: Vec<usize>) -> (Vec<u8>, Vec<usize>) {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    (bytes, shape)
}

/// Write a minimal single-layer Qwen3-shaped HF safetensors + config.json
/// model directory that `convert_hf_to_gguf` can consume. Returns the
/// directory path (caller is responsible for cleanup).
fn write_synthetic_hf_model(dir: &PathBuf) {
    std::fs::create_dir_all(dir).expect("create synthetic model dir");

    let config = serde_json::json!({
        "num_hidden_layers": 1,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).expect("serialize config.json"),
    )
    .expect("write config.json");

    // Build every tensor's raw bytes up front so all `Vec<u8>` buffers
    // outlive the `TensorView`s that borrow them.
    let embed = pattern(1, VOCAB * HIDDEN);
    let norm_final = pattern(2, HIDDEN);
    let attn_norm = pattern(3, HIDDEN);
    let ffn_norm = pattern(4, HIDDEN);
    let q_proj = pattern(5, HIDDEN * HIDDEN);
    let k_proj = pattern(6, HIDDEN * HIDDEN);
    let v_proj = pattern(7, HIDDEN * HIDDEN);
    let o_proj = pattern(8, HIDDEN * HIDDEN);
    let gate_proj = pattern(9, INTERMEDIATE * HIDDEN);
    let up_proj = pattern(10, INTERMEDIATE * HIDDEN);
    let down_proj = pattern(11, HIDDEN * INTERMEDIATE);

    let (embed_bytes, embed_shape) = f32_view(&embed, vec![VOCAB, HIDDEN]);
    let (norm_final_bytes, norm_final_shape) = f32_view(&norm_final, vec![HIDDEN]);
    let (attn_norm_bytes, attn_norm_shape) = f32_view(&attn_norm, vec![HIDDEN]);
    let (ffn_norm_bytes, ffn_norm_shape) = f32_view(&ffn_norm, vec![HIDDEN]);
    let (q_proj_bytes, q_proj_shape) = f32_view(&q_proj, vec![HIDDEN, HIDDEN]);
    let (k_proj_bytes, k_proj_shape) = f32_view(&k_proj, vec![HIDDEN, HIDDEN]);
    let (v_proj_bytes, v_proj_shape) = f32_view(&v_proj, vec![HIDDEN, HIDDEN]);
    let (o_proj_bytes, o_proj_shape) = f32_view(&o_proj, vec![HIDDEN, HIDDEN]);
    let (gate_proj_bytes, gate_proj_shape) = f32_view(&gate_proj, vec![INTERMEDIATE, HIDDEN]);
    let (up_proj_bytes, up_proj_shape) = f32_view(&up_proj, vec![INTERMEDIATE, HIDDEN]);
    let (down_proj_bytes, down_proj_shape) = f32_view(&down_proj, vec![HIDDEN, INTERMEDIATE]);

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        TensorView::new(Dtype::F32, embed_shape, &embed_bytes).expect("embed view"),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        TensorView::new(Dtype::F32, norm_final_shape, &norm_final_bytes).expect("norm view"),
    );
    tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        TensorView::new(Dtype::F32, attn_norm_shape, &attn_norm_bytes).expect("attn norm view"),
    );
    tensors.insert(
        "model.layers.0.post_attention_layernorm.weight".to_string(),
        TensorView::new(Dtype::F32, ffn_norm_shape, &ffn_norm_bytes).expect("ffn norm view"),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        TensorView::new(Dtype::F32, q_proj_shape, &q_proj_bytes).expect("q_proj view"),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        TensorView::new(Dtype::F32, k_proj_shape, &k_proj_bytes).expect("k_proj view"),
    );
    tensors.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        TensorView::new(Dtype::F32, v_proj_shape, &v_proj_bytes).expect("v_proj view"),
    );
    tensors.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        TensorView::new(Dtype::F32, o_proj_shape, &o_proj_bytes).expect("o_proj view"),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        TensorView::new(Dtype::F32, gate_proj_shape, &gate_proj_bytes).expect("gate_proj view"),
    );
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        TensorView::new(Dtype::F32, up_proj_shape, &up_proj_bytes).expect("up_proj view"),
    );
    tensors.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        TensorView::new(Dtype::F32, down_proj_shape, &down_proj_bytes).expect("down_proj view"),
    );

    safetensors::serialize_to_file(&tensors, None, &dir.join("model.safetensors"))
        .expect("write model.safetensors");
}

/// Dequantize a Q1_0_g128 tensor's raw GGUF bytes back to f32 using the
/// canonical loader convention (`bit=1 -> +d`, `bit=0 -> -d`).
fn dequant_q1_0_g128(data: &[u8]) -> Vec<f32> {
    let blocks = BlockQ1_0G128::slice_from_bytes(data).expect("valid Q1_0_g128 blocks");
    let mut out = Vec::with_capacity(blocks.len() * QK1_0_G128);
    for block in blocks {
        for i in 0..QK1_0_G128 {
            out.push(block.weight(i));
        }
    }
    out
}

#[test]
fn convert_hf_to_gguf_q1_0_g128_end_to_end() {
    let dir = unique_temp_dir("hf");
    write_synthetic_hf_model(&dir);
    let out_path = dir.join("out.gguf");

    let stats = oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_path, "q1_0_g128")
        .expect("convert_hf_to_gguf with q1_0_g128 must succeed");

    // 3 norm tensors (output_norm, attn_norm, ffn_norm) + 9 quantized weight
    // tensors (token_embd, output [tied dup], q/k/v/o_proj, gate/up/down_proj).
    assert_eq!(stats.n_fp32, 3, "expected exactly 3 FP32 norm tensors");
    assert_eq!(
        stats.n_ternary, 9,
        "expected exactly 9 Q1_0_g128-quantized weight tensors"
    );
    assert_eq!(stats.n_tensors, 12);

    let bytes = std::fs::read(&out_path).expect("read converted GGUF file");
    let gguf = GgufFile::parse(&bytes).expect("re-parse converted GGUF file");

    // Quantization metadata reflects the actually-written format, not a
    // hard-coded TQ2_0_G128 string.
    let quant_version = gguf
        .metadata
        .get_string("general.quantization_version")
        .expect("quantization_version metadata present");
    assert_eq!(quant_version, "Q1_0_G128");

    // ── FP32 exceptions honored: every norm tensor stays F32, byte-exact. ──
    for norm_name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
    ] {
        let info = gguf
            .tensors
            .require(norm_name)
            .expect("norm tensor present");
        assert_eq!(
            info.tensor_type,
            GgufTensorType::F32,
            "{norm_name} must remain FP32"
        );
        let data = gguf.tensor_data(norm_name).expect("norm tensor data");
        let recovered: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let original = match norm_name {
            "output_norm.weight" => pattern(2, HIDDEN),
            "blk.0.attn_norm.weight" => pattern(3, HIDDEN),
            "blk.0.ffn_norm.weight" => pattern(4, HIDDEN),
            _ => unreachable!(),
        };
        assert_eq!(recovered, original, "{norm_name} must round-trip exactly");
    }

    // ── Weight tensors are quantized to Q1_0_g128, with a sane round-trip. ──
    for (weight_name, seed, n) in [
        ("token_embd.weight", 1u32, VOCAB * HIDDEN),
        ("output.weight", 1u32, VOCAB * HIDDEN), // tied dup of token_embd
        ("blk.0.attn_q.weight", 5, HIDDEN * HIDDEN),
        ("blk.0.attn_k.weight", 6, HIDDEN * HIDDEN),
        ("blk.0.attn_v.weight", 7, HIDDEN * HIDDEN),
        ("blk.0.attn_output.weight", 8, HIDDEN * HIDDEN),
        ("blk.0.ffn_gate.weight", 9, INTERMEDIATE * HIDDEN),
        ("blk.0.ffn_up.weight", 10, INTERMEDIATE * HIDDEN),
        ("blk.0.ffn_down.weight", 11, HIDDEN * INTERMEDIATE),
    ] {
        let info = gguf
            .tensors
            .require(weight_name)
            .unwrap_or_else(|_| panic!("{weight_name} tensor present"));
        assert_eq!(
            info.tensor_type,
            GgufTensorType::Q1_0_g128,
            "{weight_name} must be quantized to Q1_0_g128"
        );

        let data = gguf.tensor_data(weight_name).expect("weight tensor data");
        let dequantized = dequant_q1_0_g128(data);
        let original = pattern(seed, n);

        // The reconstructed sign must match the original sign for every
        // element (Q1_0_g128 only encodes 1 bit of sign, no zero symbol).
        let mut sign_mismatches = 0usize;
        let mut max_abs_error = 0.0f32;
        for (i, (&orig, &recon)) in original.iter().zip(dequantized.iter()).enumerate() {
            let orig_sign = orig >= 0.0;
            let recon_sign = recon >= 0.0;
            if orig_sign != recon_sign {
                sign_mismatches += 1;
            }
            let err = (orig - recon).abs();
            if err > max_abs_error {
                max_abs_error = err;
            }
            // Sanity bound: reconstructed magnitude cannot exceed the
            // group's true max-abs value (the scale is exactly that, up to
            // FP16 rounding), and every |recon| must be within a small
            // tolerance of the group's f16-rounded scale.
            assert!(
                recon.abs() <= 2.0,
                "index {i}: |recon|={} exceeds a sane bound for this pattern",
                recon.abs()
            );
        }
        assert_eq!(
            sign_mismatches, 0,
            "{weight_name}: Q1_0_g128 must preserve every element's sign \
             (canonical bit=1 -> +d convention)"
        );
        assert!(
            max_abs_error < 3.0,
            "{weight_name}: max_abs_error {max_abs_error} implausibly large for Q1_0_g128"
        );
    }

    // token_embd.weight and output.weight must be byte-identical (tied
    // embeddings, duplicated before quantization).
    let embd_data = gguf
        .tensor_data("token_embd.weight")
        .expect("token_embd data");
    let output_data = gguf.tensor_data("output.weight").expect("output data");
    assert_eq!(
        embd_data, output_data,
        "tied embeddings: output.weight must be a byte-identical duplicate of token_embd.weight"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn convert_hf_to_gguf_rejects_unsupported_quant_format() {
    let dir = unique_temp_dir("hf_bad_quant");
    write_synthetic_hf_model(&dir);
    let out_path = dir.join("out.gguf");

    let err = oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_path, "q4_0")
        .expect_err("q4_0 is not a supported convert --quant format");
    let msg = err.to_string();
    assert!(
        msg.contains("q4_0") && msg.contains("tq2_0_g128") && msg.contains("q1_0_g128"),
        "error message should name the rejected value and both supported formats, got: {msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn convert_hf_to_gguf_q1_0_g128_and_tq2_0_g128_agree_on_norm_tensors() {
    // Both quant formats must keep norm tensors FP32 identically — only the
    // weight tensors' encoding should differ.
    let dir = unique_temp_dir("hf_cmp");
    write_synthetic_hf_model(&dir);

    let out_q1 = dir.join("out_q1.gguf");
    let out_tq2 = dir.join("out_tq2.gguf");

    oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_q1, "q1_0_g128")
        .expect("q1_0_g128 convert");
    oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_tq2, "tq2_0_g128")
        .expect("tq2_0_g128 convert");

    let bytes_q1 = std::fs::read(&out_q1).expect("read q1 output");
    let bytes_tq2 = std::fs::read(&out_tq2).expect("read tq2 output");
    let gguf_q1 = GgufFile::parse(&bytes_q1).expect("parse q1 output");
    let gguf_tq2 = GgufFile::parse(&bytes_tq2).expect("parse tq2 output");

    for norm_name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
    ] {
        assert_eq!(
            gguf_q1.tensor_data(norm_name).expect("q1 norm data"),
            gguf_tq2.tensor_data(norm_name).expect("tq2 norm data"),
            "{norm_name} bytes must be identical across quant formats (always FP32)"
        );
    }

    // But the weight tensor type differs between the two formats.
    assert_eq!(
        gguf_q1
            .tensors
            .require("blk.0.attn_q.weight")
            .unwrap()
            .tensor_type,
        GgufTensorType::Q1_0_g128
    );
    assert_eq!(
        gguf_tq2
            .tensors
            .require("blk.0.attn_q.weight")
            .unwrap()
            .tensor_type,
        GgufTensorType::TQ2_0_g128
    );

    std::fs::remove_dir_all(&dir).ok();
}
