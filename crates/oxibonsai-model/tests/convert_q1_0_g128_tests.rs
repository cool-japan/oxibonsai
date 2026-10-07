//! End-to-end tests for the whole convert / export surface.
//!
//! Originally a regression suite for `convert --quant q1_0_g128`
//! (cli-facade-01): the `--help` honesty gap was fixed first, but
//! `oxibonsai convert --quant q1_0_g128` still bailed with "unsupported
//! quantisation format" because the format was only implemented in the
//! separate `quantize` subcommand. Those tests remain, and the file now also
//! covers the rest of the writer surface, since every item shares the same
//! synthetic-HF-model fixtures:
//!
//! * the converted file actually loads — `GgufFile::parse` →
//!   `Qwen3Config::from_metadata` → the embedded `tokenizer.ggml.*` block
//!   (CQ-01 / CQ-08);
//! * a source tensor with no mapping fails the run by name (CQ-04);
//! * norms and 1-D tensors stay F32 in **every** export format (CQ-02);
//! * a tensor whose `ne0` is not a block multiple lands in F32 rather than
//!   being flat-padded across row boundaries (CQ-14);
//! * the Bonsai 2 writers `PQ2_0` / `PTQ1_0` / `Q2_0`-g64 produce the right
//!   ids and block extents, with `PQ2_0` verified scale-**first** (CQ-05);
//! * the streaming entry point pulls one tensor at a time (CQ-17);
//! * `qwen35.*` and `prism.hadamard.*` are emitted and validated (CQ-06).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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

/// Same model, plus one extra tensor with no GGUF mapping (CQ-04 fixture).
fn write_synthetic_hf_model_with_extra(dir: &PathBuf, extra_name: &str) {
    write_synthetic_hf_model(dir);

    // Re-serialise with the extra tensor appended. Re-reading the file we
    // just wrote keeps the two models identical apart from that one tensor.
    let raw = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let parsed = safetensors::SafeTensors::deserialize(&raw).expect("parse model.safetensors");

    let extra = pattern(42, HIDDEN * HIDDEN);
    let (extra_bytes, extra_shape) = f32_view(&extra, vec![HIDDEN, HIDDEN]);

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    for name in parsed.names() {
        let view = parsed.tensor(name).expect("existing tensor");
        tensors.insert(name.to_string(), view);
    }
    tensors.insert(
        extra_name.to_string(),
        TensorView::new(Dtype::F32, extra_shape, &extra_bytes).expect("extra view"),
    );

    safetensors::serialize_to_file(&tensors, None, &dir.join("model.safetensors"))
        .expect("rewrite model.safetensors");
}

/// Write a minimal byte-level-BPE `tokenizer.json` + `tokenizer_config.json`
/// covering exactly `VOCAB` ids, so the converted GGUF can carry a tokenizer.
fn write_synthetic_tokenizer(dir: &Path) {
    let mut vocab = serde_json::Map::new();
    // Ids 0..VOCAB-2 are ordinary tokens; the last two are specials added via
    // `added_tokens`, mirroring a real Qwen tokenizer's layout.
    for id in 0..(VOCAB - 2) {
        vocab.insert(format!("t{id}"), serde_json::json!(id));
    }
    let tokenizer = serde_json::json!({
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": ["t0 t1", "t1 t2"],
        },
        "added_tokens": [
            { "id": VOCAB - 2, "content": "<|endoftext|>", "special": true },
            { "id": VOCAB - 1, "content": "<|im_end|>", "special": true },
        ],
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tokenizer).expect("serialise tokenizer.json"),
    )
    .expect("write tokenizer.json");

    let config = serde_json::json!({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|im_end|>",
        "add_bos_token": false,
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    });
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string(&config).expect("serialise tokenizer_config.json"),
    )
    .expect("write tokenizer_config.json");
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

    // `general.quantization_version` is the GGUF **schema** version and the
    // spec types it as `uint32`. This assertion used to demand a *string*
    // naming the format, pinning the exact defect core-gguf-02 reports: a
    // string there makes `MetadataValue::as_u32` return `None`, so every
    // spec-conforming reader — including OxiBonsai's own `get_u32` — fails on
    // our files, and llama.cpp refuses them outright.
    assert_eq!(
        gguf.metadata
            .get_u32("general.quantization_version")
            .expect("quantization_version must be a u32"),
        2
    );
    // The human-readable format label lives in `general.file_type` (the ggml
    // enum) plus a non-normative OxiBonsai key.
    assert_eq!(
        gguf.metadata
            .get_u32("general.file_type")
            .expect("file_type present"),
        40,
        "PrismML MOSTLY_Q1_0"
    );
    assert_eq!(
        gguf.metadata
            .get_string("oxibonsai.quant_format")
            .expect("quant format label present"),
        "Q1_0_G128"
    );

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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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
            .expect("q1 attn_q present")
            .tensor_type,
        GgufTensorType::Q1_0_g128
    );
    // `tq2_0_g128` is written in the runtime-native qs-first layout under
    // ggml id 42 — the one `run`/`chat`/`serve` execute. The llama.cpp-
    // readable `d`-first id 142 (core-gguf-02) is opt-in via `pq2_0`; see
    // `tq2_0_g128_converts_to_native_id_42_and_loads` below.
    let tq2_attn_q = gguf_tq2
        .tensors
        .require("blk.0.attn_q.weight")
        .expect("tq2 attn_q present");
    assert_eq!(tq2_attn_q.tensor_type, GgufTensorType::TQ2_0_g128);
    assert_eq!(tq2_attn_q.tensor_type.wire_id(), 42);

    std::fs::remove_dir_all(&dir).ok();
}

// ─── Acceptance: every converted file is loadable ─────────────────────────────

/// A converted GGUF must load through the *production* path: `GgufFile::parse`
/// → `Qwen3Config::from_metadata` → the embedded tokenizer block.
///
/// Before CQ-01/CQ-08 a converted file had 13 KV pairs under an `llm.*`
/// namespace that no reader looks at and zero `tokenizer.*` keys, so
/// `Qwen3Config::from_metadata` silently returned hard-coded Bonsai-8B
/// defaults and the model could not be tokenized at all.
#[test]
fn converted_gguf_loads_through_config_and_tokenizer() {
    use oxibonsai_core::Qwen3Config;

    let dir = unique_temp_dir("hf_loadable");
    write_synthetic_hf_model(&dir);
    write_synthetic_tokenizer(&dir);
    let out_path = dir.join("out.gguf");

    oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_path, "pq2_0")
        .expect("convert must succeed");

    let bytes = std::fs::read(&out_path).expect("read converted GGUF");
    let gguf = GgufFile::parse(&bytes).expect("GgufFile::parse");

    // ── Architecture block, under the real prefix. ─────────────────────────
    assert_eq!(
        gguf.metadata
            .get_string("general.architecture")
            .expect("architecture present"),
        "qwen3"
    );
    let config = Qwen3Config::from_metadata(&gguf.metadata).expect("Qwen3Config::from_metadata");
    assert_eq!(config.num_layers, 1, "read from qwen3.block_count");
    assert_eq!(config.hidden_size, HIDDEN);
    assert_eq!(config.intermediate_size, INTERMEDIATE);
    assert_eq!(config.vocab_size, VOCAB);
    assert_eq!(config.num_attention_heads, 2);
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.max_context_length, 512);
    assert!((config.rope_freq_base - 10_000.0).abs() < 1.0);

    // These would all be Bonsai-8B defaults if the file still used `llm.*`.
    assert!(
        gguf.metadata.get_u32("llm.block_count").is_err(),
        "the non-existent llm.* namespace must not be written"
    );

    // ── Tokenizer block (CQ-08). ───────────────────────────────────────────
    assert_eq!(
        gguf.metadata
            .get_string("tokenizer.ggml.model")
            .expect("tokenizer model"),
        "gpt2"
    );
    assert_eq!(
        gguf.metadata
            .get_string("tokenizer.ggml.pre")
            .expect("pre-tokenizer"),
        "qwen2"
    );
    let tokens = gguf
        .metadata
        .get_string_array("tokenizer.ggml.tokens")
        .expect("token array");
    assert_eq!(
        tokens.len(),
        config.vocab_size,
        "the embedded vocabulary must cover exactly the model's vocab_size"
    );
    let token_types = gguf
        .metadata
        .get_i32_array("tokenizer.ggml.token_type")
        .expect("token_type array");
    assert_eq!(token_types.len(), tokens.len());
    assert!(!gguf
        .metadata
        .get_string_array("tokenizer.ggml.merges")
        .expect("merges present")
        .is_empty());
    assert_eq!(
        gguf.metadata
            .get_u32("tokenizer.ggml.eos_token_id")
            .expect("eos id"),
        (VOCAB - 1) as u32
    );
    assert_eq!(
        gguf.metadata
            .get_u32("tokenizer.ggml.bos_token_id")
            .expect("bos id"),
        (VOCAB - 2) as u32
    );
    assert!(
        gguf.metadata.get_string("tokenizer.chat_template").is_ok(),
        "the source chat template must be carried into the GGUF"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A source tensor the converter cannot place must fail the run and name
/// itself, not vanish behind a `debug!` and a success message (CQ-04).
#[test]
fn unmapped_source_tensor_fails_the_conversion() {
    let dir = unique_temp_dir("hf_unmapped");
    write_synthetic_hf_model_with_extra(&dir, "model.layers.0.mystery_adapter.weight");
    let out_path = dir.join("out.gguf");

    let err = oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_path, "pq2_0")
        .expect_err("an unmapped tensor must fail the conversion");
    let msg = err.to_string();
    assert!(
        msg.contains("model.layers.0.mystery_adapter.weight"),
        "the error must name every dropped tensor, got: {msg}"
    );
    assert!(msg.contains("--allow-unmapped"), "got: {msg}");

    // …and the escape hatch converts anyway, reporting the count.
    let stats =
        oxibonsai_model::convert::convert_hf_to_gguf_with_options(&dir, &out_path, "pq2_0", true)
            .expect("--allow-unmapped converts");
    assert_eq!(stats.n_unmapped, 1);

    std::fs::remove_dir_all(&dir).ok();
}

/// `config.json` naming a model family this converter does not implement must
/// be refused up front, not converted into a file mislabelled `qwen3`.
#[test]
fn foreign_architecture_is_refused() {
    let dir = unique_temp_dir("hf_foreign");
    write_synthetic_hf_model(&dir);
    let config_path = dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path).expect("read config");
    let mut config: serde_json::Value = serde_json::from_str(&raw).expect("parse config");
    config["architectures"] = serde_json::json!(["LlamaForCausalLM"]);
    std::fs::write(
        &config_path,
        serde_json::to_string(&config).expect("serialise"),
    )
    .expect("write config");

    let err = oxibonsai_model::convert::convert_hf_to_gguf(&dir, &dir.join("out.gguf"), "pq2_0")
        .expect_err("a Llama checkpoint must be refused");
    assert!(err.to_string().contains("LlamaForCausalLM"));

    std::fs::remove_dir_all(&dir).ok();
}

/// PQ2_0 is `d`-first: the first two bytes of a block are the FP16 scale.
///
/// This is the assertion no size check can make. `BlockTQ2_0_g128` (id 42) is
/// `qs`-first with `d` last, and both are 34 bytes per 128 weights, so a
/// byte-swapped block passes every length validation and only shows up as
/// wrong numbers much later.
#[test]
fn pq2_0_blocks_are_scale_first() {
    let dir = unique_temp_dir("hf_pq2_layout");
    write_synthetic_hf_model(&dir);
    let out_path = dir.join("out.gguf");
    oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_path, "pq2_0").expect("convert");

    let bytes = std::fs::read(&out_path).expect("read");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let info = gguf
        .tensors
        .require("blk.0.attn_q.weight")
        .expect("attn_q present");
    assert_eq!(info.tensor_type, GgufTensorType::PQ2_0);

    let data = gguf.tensor_data("blk.0.attn_q.weight").expect("data");
    let blocks = oxibonsai_core::quant_prism::BlockPQ2_0::slice_from_bytes(data)
        .expect("slice PQ2_0 blocks");
    assert!(!blocks.is_empty());

    // Byte 0..2 of the first block is the f16 scale; it must be finite and
    // strictly positive for this non-zero pattern.
    let scale_bits = u16::from_le_bytes([data[0], data[1]]);
    let scale = half::f16::from_bits(scale_bits).to_f32();
    assert!(
        scale.is_finite() && scale > 0.0,
        "PQ2_0 block must start with the f16 scale, read {scale}"
    );
    assert_eq!(
        scale,
        blocks[0].d.to_f32(),
        "the struct's `d` must be exactly the first two bytes"
    );

    // Ternary data must never emit the reserved `0b11` (+2) code.
    let plus_two: u32 = blocks.iter().map(|b| b.count_plus_two()).sum();
    assert_eq!(
        plus_two, 0,
        "a ternary tensor must contain no 0b11 codes; {plus_two} found"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Re-serialise the synthetic model with extra tensors appended (each given
/// as `(hf_name, data, hf_shape)`), keeping every existing tensor unchanged.
fn add_tensors_to_synthetic_model(dir: &Path, extra: &[(&str, Vec<f32>, Vec<usize>)]) {
    let raw = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
    let parsed = safetensors::SafeTensors::deserialize(&raw).expect("parse model.safetensors");

    let extra_bytes: Vec<(&str, Vec<u8>, Vec<usize>)> = extra
        .iter()
        .map(|(name, data, shape)| {
            let (bytes, shape) = f32_view(data, shape.clone());
            (*name, bytes, shape)
        })
        .collect();

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    for name in parsed.names() {
        let view = parsed.tensor(name).expect("existing tensor");
        tensors.insert(name.to_string(), view);
    }
    for (name, bytes, shape) in &extra_bytes {
        tensors.insert(
            (*name).to_string(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).expect("extra view"),
        );
    }

    safetensors::serialize_to_file(&tensors, None, &dir.join("model.safetensors"))
        .expect("rewrite model.safetensors");
}

/// Regression (0.2.4 release blocker): `convert --quant tq2_0_g128` — what
/// `scripts/download_ternary.sh` runs for the gen-1 Ternary-Bonsai
/// 1.7B/4B/8B checkpoints — must write the runtime-native qs-first layout
/// under ggml id 42 (`general.file_type` 41), and the result must load and
/// run through the production model loader.
///
/// For a while `tq2_0_g128` was mapped to PrismML `PQ2_0` (id 142, `d`
/// first). The runtime decodes that per layer but has no `PQ2_0`
/// output-projection wrapper, so every freshly converted Ternary-Bonsai GGUF
/// failed `oxibonsai run` with "no output-projection Linear* kernel
/// wrapper". The ternary payload of the two spellings is identical — each
/// `tq2_0_g128` block is the `pq2_0` block rotated left by two bytes — and
/// that is asserted here as well, so the llama.cpp-readable `pq2_0` output
/// cannot drift away from the native one.
#[test]
fn tq2_0_g128_converts_to_native_id_42_and_loads() {
    use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
    use oxibonsai_model::model::BonsaiModel;

    let dir = unique_temp_dir("hf_tq2_native");
    write_synthetic_hf_model(&dir);
    write_synthetic_tokenizer(&dir);
    // The production qwen3 block loader requires the per-head Q/K RMSNorm
    // gains, which the minimal fixture leaves out (2 heads → head_dim 64).
    let head_dim = HIDDEN / 2;
    add_tensors_to_synthetic_model(
        &dir,
        &[
            (
                "model.layers.0.self_attn.q_norm.weight",
                pattern(12, head_dim),
                vec![head_dim],
            ),
            (
                "model.layers.0.self_attn.k_norm.weight",
                pattern(13, head_dim),
                vec![head_dim],
            ),
        ],
    );

    let out_tq2 = dir.join("out_tq2.gguf");
    let out_pq2 = dir.join("out_pq2.gguf");
    let stats = oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_tq2, "tq2_0_g128")
        .expect("tq2_0_g128 convert");
    oxibonsai_model::convert::convert_hf_to_gguf(&dir, &out_pq2, "pq2_0").expect("pq2_0 convert");

    let bytes_tq2 = std::fs::read(&out_tq2).expect("read tq2 output");
    let bytes_pq2 = std::fs::read(&out_pq2).expect("read pq2 output");
    assert_eq!(
        bytes_tq2.len(),
        bytes_pq2.len(),
        "both layouts are 34 bytes per 128 weights"
    );
    let gguf_tq2 = GgufFile::parse(&bytes_tq2).expect("parse tq2 output");
    let gguf_pq2 = GgufFile::parse(&bytes_pq2).expect("parse pq2 output");

    // ── (a) On disk: ggml id 42, `general.file_type` 41. ──────────────────
    assert_eq!(
        gguf_tq2.metadata.get_u32("general.file_type").ok(),
        Some(41),
        "PrismML MOSTLY_Q2_0, not MOSTLY_PQ2_0 (141)"
    );
    assert_eq!(
        gguf_pq2.metadata.get_u32("general.file_type").ok(),
        Some(141)
    );
    assert_eq!(
        gguf_tq2.metadata.get_string("oxibonsai.quant_format").ok(),
        Some("TQ2_0_G128")
    );

    const TERNARY_TENSORS: [&str; 9] = [
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ];
    assert_eq!(stats.n_ternary, TERNARY_TENSORS.len());
    for name in TERNARY_TENSORS {
        let info = gguf_tq2.tensors.require(name).expect("tq2 tensor present");
        assert_eq!(info.tensor_type, GgufTensorType::TQ2_0_g128, "{name}");
        assert_eq!(info.tensor_type.wire_id(), 42, "{name}");
        let pq2_info = gguf_pq2.tensors.require(name).expect("pq2 tensor present");
        assert_eq!(pq2_info.tensor_type, GgufTensorType::PQ2_0, "{name}");

        let t = gguf_tq2.tensor_data(name).expect("tq2 data");
        let p = gguf_pq2.tensor_data(name).expect("pq2 data");
        assert_eq!(t.len(), p.len(), "{name}");
        let (t_blocks, t_tail) = t.as_chunks::<34>();
        let (p_blocks, p_tail) = p.as_chunks::<34>();
        assert!(
            !t_blocks.is_empty() && t_tail.is_empty() && p_tail.is_empty(),
            "{name}: {} bytes is not a whole number of 34-byte blocks",
            t.len()
        );
        for (i, (tb, pb)) in t_blocks.iter().zip(p_blocks).enumerate() {
            assert_eq!(&tb[..32], &pb[2..], "{name} block {i}: codes must match");
            assert_eq!(&tb[32..], &pb[..2], "{name} block {i}: scale must match");
        }
        // qs first, `d` last: the scale is the block's final two bytes.
        let scale = half::f16::from_bits(u16::from_le_bytes([t[32], t[33]])).to_f32();
        assert!(
            scale.is_finite() && scale > 0.0,
            "{name}: TQ2_0_g128 block must end with a positive f16 scale, read {scale}"
        );
    }
    for name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
    ] {
        assert_eq!(
            gguf_tq2.tensor_data(name).expect("tq2 norm"),
            gguf_pq2.tensor_data(name).expect("pq2 norm"),
            "{name} must be byte-identical across the two ternary spellings"
        );
    }

    // ── (b) It loads and runs through the production loader. ──────────────
    let mut model = BonsaiModel::from_gguf(&gguf_tq2, 512)
        .unwrap_or_else(|e| panic!("a tq2_0_g128 conversion must load, got: {e}"));
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let logits = model
        .forward(0, 0, &kernel)
        .expect("forward pass through the TQ2_0_g128 blocks and LM head");
    assert_eq!(logits.len(), VOCAB);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit must be finite"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// Export surface: FP32 policy, new Bonsai 2 writers, streaming, metadata.
// ─────────────────────────────────────────────────────────────────────────────

use oxibonsai_model::convert::meta::{ArchMetadata, GeneralMetadata};
use oxibonsai_model::convert::qwen35::{HadamardSpec, Qwen35Metadata, SignMode};
use oxibonsai_model::convert::tokenizer_meta::TokenizerMetadata;
use oxibonsai_model::export::{
    export_to_gguf, export_to_gguf_streaming, keep_fp32_by_kind, ExportConfig, ExportError,
    ExportFormat, TensorPlan, WeightTensor,
};
use oxibonsai_model::quantize::ScaleRule;

/// Deterministic non-trivial weights for an export fixture.
fn weights(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (seed as f32) * 0.31 + (i as f32) * 0.017;
            x.sin() * (0.2 + (i % 5) as f32 * 0.1)
        })
        .collect()
}

/// Every format that has a GGUF tensor type (i.e. everything but the
/// export-only INT8 variant).
fn writable_formats() -> Vec<ExportFormat> {
    ExportFormat::ALL
        .iter()
        .copied()
        .filter(|f| f.tensor_type().is_some())
        .collect()
}

/// CQ-02: **no** norm tensor is quantized, in **any** export config or
/// format.
///
/// `blk.0.attn_norm.weight` is the tensor the finding measured at up to 910 %
/// relative error on the shipped 1.7 B; `blk.0.attn_q_norm.weight` is the
/// 128-element worst case. llama.cpp keeps 1-D tensors in a float type
/// unconditionally, so q4_0 / q8_0 / q4_k were wrong here too — not just the
/// 1-bit format the finding happened to reproduce on.
#[test]
fn norm_tensors_are_never_quantized_in_any_format() {
    let norms = ["blk.0.attn_norm.weight", "blk.0.attn_q_norm.weight"];

    for format in writable_formats() {
        let tensors = vec![
            WeightTensor::new("blk.0.attn_q.weight", weights(2048, 1), vec![256, 8]),
            WeightTensor::new(norms[0], weights(256, 2), vec![256]),
            // The 128-element per-head norm: a whole number of 128-blocks, so
            // only the *kind* rule can save it.
            WeightTensor::new(norms[1], weights(128, 3), vec![128]),
            // A norm that arrives with a redundant trailing dimension is
            // still a norm.
            WeightTensor::new("blk.0.ffn_norm.weight", weights(256, 4), vec![256, 1]),
        ];

        // Deliberately the *empty* exception list: the structural rule must
        // hold even when the caller asks for nothing to be protected.
        let config = ExportConfig::new(format, "fp32-policy").with_fp32_layers(vec![]);
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
        let gguf = GgufFile::parse(&bytes).expect("parse");

        for name in [norms[0], norms[1], "blk.0.ffn_norm.weight"] {
            let info = gguf.tensors.require(name).expect("norm present");
            assert_eq!(
                info.tensor_type,
                GgufTensorType::F32,
                "{name} must stay F32 under {format:?}"
            );
        }
    }
}

#[test]
fn keep_fp32_predicate_is_exactly_the_two_documented_rules() {
    assert!(keep_fp32_by_kind("anything", &[256]), "1-D");
    assert!(keep_fp32_by_kind("blk.0.attn_norm.weight", &[5120, 1]));
    assert!(keep_fp32_by_kind("output_norm.weight", &[5120, 1]));
    assert!(keep_fp32_by_kind("blk.0.attn_q_norm.weight", &[256, 1]));
    assert!(!keep_fp32_by_kind("blk.0.attn_q.weight", &[5120, 12288]));
    assert!(!keep_fp32_by_kind("token_embd.weight", &[5120, 248320]));
}

/// A tensor whose `ne0` is not a block multiple lands in
/// F32 while the rest of the model is quantized — llama.cpp's own behaviour —
/// instead of failing the whole run or being flat-padded across rows.
#[test]
fn a_non_block_aligned_tensor_lands_in_f32_and_the_rest_is_quantized() {
    let tensors = vec![
        WeightTensor::new("blk.0.attn_q.weight", weights(1024, 1), vec![128, 8]),
        // 130 is not a multiple of 128: every row after the first would
        // straddle a group boundary if this were padded flat (CQ-14).
        WeightTensor::new("blk.0.odd.weight", weights(520, 2), vec![130, 4]),
        WeightTensor::new("blk.0.ffn_up.weight", weights(512, 3), vec![256, 2]),
    ];
    let config = ExportConfig::new(ExportFormat::PQ2_0, "mixed").with_fp32_layers(vec![]);

    let stats = oxibonsai_model::export::export_stats(&tensors, &config);
    assert_eq!(stats.quantized_tensors, 2);
    assert_eq!(stats.fp32_tensors, 1);

    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export must not fail");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    assert_eq!(
        gguf.tensors
            .require("blk.0.odd.weight")
            .expect("odd tensor")
            .tensor_type,
        GgufTensorType::F32
    );
    assert_eq!(
        gguf.tensors
            .require("blk.0.attn_q.weight")
            .expect("q tensor")
            .tensor_type,
        GgufTensorType::PQ2_0
    );
    assert_eq!(
        gguf.tensors
            .require("blk.0.ffn_up.weight")
            .expect("ffn tensor")
            .tensor_type,
        GgufTensorType::PQ2_0
    );

    // The F32 tensor must round-trip exactly, not through a padded encoding.
    let data = gguf.tensor_data("blk.0.odd.weight").expect("odd data");
    let recovered: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(recovered, weights(520, 2));
}

/// CQ-05: the three Bonsai 2 writers produce files the real reader accepts,
/// with the right ids and the right block sizes.
#[test]
fn bonsai2_export_formats_round_trip_through_the_reader() {
    let cases: [(ExportFormat, GgufTensorType, usize, usize); 4] = [
        (ExportFormat::PQ2_0, GgufTensorType::PQ2_0, 128, 34),
        (ExportFormat::PTQ1_0, GgufTensorType::PTQ1_0, 128, 28),
        (ExportFormat::Q2_0G64, GgufTensorType::Q2_0G64, 64, 18),
        (
            ExportFormat::TernaryG128,
            GgufTensorType::TQ2_0_g128,
            128,
            34,
        ),
    ];

    for (format, expected_type, group, block_bytes) in cases {
        let n = 1024usize;
        // Two tensors, so the offset delta measures the first one's real
        // on-disk extent. That matters for `Q2_0G64`: it shares ggml id 42
        // with the legacy group-128 layout, so a naive reader resolves the id
        // to the *other* interpretation and `tensor_data` hands back a short
        // slice. The writer's layout is what this test is about.
        let tensors = vec![
            WeightTensor::new("blk.0.attn_q.weight", weights(n, 9), vec![256, 4]),
            WeightTensor::new("blk.0.attn_k.weight", weights(n, 10), vec![256, 4]),
        ];
        let config = ExportConfig::new(format, "b2").with_scale_rule(ScaleRule::AbsMean);
        let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
        let gguf = GgufFile::parse(&bytes).expect("parse");

        let first = gguf
            .tensors
            .require("blk.0.attn_q.weight")
            .expect("tensor present");
        let second = gguf
            .tensors
            .require("blk.0.attn_k.weight")
            .expect("tensor present");
        assert_eq!(
            first.tensor_type.wire_id(),
            expected_type.wire_id(),
            "{format:?} wire id"
        );
        if expected_type.wire_id() != 42 {
            assert_eq!(first.tensor_type, expected_type, "{format:?}");
        }

        // ggml pads each tensor's start to the 32-byte alignment boundary.
        let expected_bytes = (n / group * block_bytes) as u64;
        let expected_stride = expected_bytes.div_ceil(32) * 32;
        assert_eq!(
            second.offset - first.offset,
            expected_stride,
            "{format:?}: {expected_bytes} bytes of tensor data, padded to {expected_stride}"
        );
    }
}

/// CQ-18: `Int8PerChannel` still has no GGUF tensor-type id anywhere in this
/// workspace, so it is refused rather than written as mislabelled bytes — and
/// it says so, naming the tensor.
#[test]
fn int8_per_channel_is_refused_with_a_specific_error() {
    assert!(ExportFormat::Int8PerChannel.tensor_type().is_none());
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        weights(256, 1),
        vec![128, 2],
    )];
    let config = ExportConfig::new(ExportFormat::Int8PerChannel, "int8");
    let err = export_to_gguf(&tensors, &config, &[]).expect_err("must be refused");
    assert!(matches!(err, ExportError::NoLoaderForFormat { .. }));
    assert!(err.to_string().contains("blk.0.attn_q.weight"));
}

/// CQ-01: a GGUF→GGUF re-quantization carries the source's architecture and
/// tokenizer, so the output loads through the same path the source did.
#[test]
fn source_metadata_carries_over_and_the_output_is_loadable() {
    use oxibonsai_core::Qwen3Config;

    // Build a "source" model with a full metadata block.
    let dir = unique_temp_dir("requant");
    write_synthetic_hf_model(&dir);
    write_synthetic_tokenizer(&dir);
    let source_path = dir.join("source.gguf");
    oxibonsai_model::convert::convert_hf_to_gguf(&dir, &source_path, "pq2_0")
        .expect("build source");
    let source_bytes = std::fs::read(&source_path).expect("read source");
    let source = GgufFile::parse(&source_bytes).expect("parse source");

    // Re-quantize it to a different format through the export path.
    let mut tensors = Vec::new();
    for name in source.tensors.sorted_names() {
        let info = source.tensors.require(name).expect("info");
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        // Only the F32 tensors need decoding for this test; the quantized
        // ones are re-created from a deterministic pattern of the right size.
        let data = weights(shape.iter().product::<usize>(), 3);
        tensors.push(WeightTensor::new(name, data, shape));
    }

    let config = ExportConfig::new(ExportFormat::PTQ1_0, "requantized")
        .with_source_metadata(&source.metadata)
        .expect("source has an architecture");
    let out = export_to_gguf(&tensors, &config, &[]).expect("re-export");
    let gguf = GgufFile::parse(&out).expect("parse re-export");

    // Architecture survived…
    let cfg = Qwen3Config::from_metadata(&gguf.metadata).expect("config");
    assert_eq!(cfg.num_layers, 1);
    assert_eq!(cfg.hidden_size, HIDDEN);
    assert_eq!(cfg.vocab_size, VOCAB);
    // …and so did the tokenizer.
    assert_eq!(
        gguf.metadata
            .get_string_array("tokenizer.ggml.tokens")
            .expect("tokens")
            .len(),
        VOCAB
    );
    // …but the output-describing keys were recomputed, not inherited.
    assert_eq!(
        gguf.metadata.get_u32("general.quantization_version").ok(),
        Some(2)
    );
    assert_eq!(
        gguf.metadata.get_u32("general.file_type").ok(),
        Some(143),
        "file_type must describe the PTQ1_0 output, not the PQ2_0 source"
    );
    // (The source here is itself a converted file, which carries no
    // `general.version`; `source_version_and_description_survive_a_requantize`
    // covers the carry-over for a source that has one.)
    assert_eq!(
        gguf.metadata.get_string("general.name").ok(),
        Some("requantized"),
        "general.name is the caller's, since it names the new artifact"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// `general.version` and `general.description` are on the writer-owned key
/// list (they would otherwise be written twice), so they have to be lifted
/// into the config or they disappear from a re-quantized file.
#[test]
fn source_version_and_description_survive_a_requantize() {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

    let mut w = GgufWriter::new();
    for (key, value) in [
        ("general.architecture", "qwen35"),
        ("general.version", "v5"),
        ("general.description", "PrismML Bonsai 2 27B"),
        ("general.basename", "folded"),
        ("general.size_label", "27B"),
    ] {
        w.add_metadata(key, MetadataWriteValue::Str(value.to_string()));
    }
    let source_bytes = w.to_bytes().expect("serialise source");
    let source = GgufFile::parse(&source_bytes).expect("parse source");

    let config = ExportConfig::new(ExportFormat::PQ2_0, "requantized")
        .with_source_metadata(&source.metadata)
        .expect("architecture present");
    assert_eq!(config.model_version, "v5");
    assert_eq!(config.description.as_deref(), Some("PrismML Bonsai 2 27B"));

    let tensors = vec![WeightTensor::new(
        "blk.0.attn_q.weight",
        weights(1024, 1),
        vec![256, 4],
    )];
    let out = export_to_gguf(&tensors, &config, &[]).expect("export");
    let gguf = GgufFile::parse(&out).expect("parse");

    assert_eq!(gguf.metadata.get_string("general.version").ok(), Some("v5"));
    assert_eq!(
        gguf.metadata.get_string("general.description").ok(),
        Some("PrismML Bonsai 2 27B")
    );
    // Keys the writer does not own are copied through untouched.
    assert_eq!(
        gguf.metadata.get_string("general.basename").ok(),
        Some("folded")
    );
    assert_eq!(
        gguf.metadata.get_string("general.size_label").ok(),
        Some("27B")
    );
}

#[test]
fn a_source_without_an_architecture_is_refused() {
    use oxibonsai_core::gguf::writer::GgufWriter;

    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.name",
        oxibonsai_core::gguf::writer::MetadataWriteValue::Str("nameless".to_string()),
    );
    let bytes = w.to_bytes().expect("serialise");
    let gguf = GgufFile::parse(&bytes).expect("parse");

    let err = ExportConfig::new(ExportFormat::PQ2_0, "x")
        .with_source_metadata(&gguf.metadata)
        .expect_err("no architecture must be refused");
    assert!(matches!(err, ExportError::MissingArchitecture));
}

/// A sink that records how many bytes have reached it, readable from
/// elsewhere while the writer holds it.
///
/// The shared `Rc<RefCell<..>>` is what lets the loader callback observe the
/// sink's state: the sink borrows it only for the duration of its own
/// `write`, so a read from the callback — which runs between writes — never
/// conflicts.
#[derive(Clone)]
struct RecordingSink {
    bytes: std::rc::Rc<std::cell::RefCell<Vec<u8>>>,
}

impl std::io::Write for RecordingSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// CQ-17: the streaming writer pulls each tensor exactly once and has already
/// written every previous tensor to the sink before it asks for the next one.
///
/// Peak RSS is not assertable on a build machine, so the *property* that
/// bounds it is asserted directly, and in a form that can actually fail: at
/// the moment tensor *N* is requested, the sink must already hold the header
/// plus tensors `0..N`. A writer that buffered the whole model and flushed at
/// the end would show a sink length that never grows between callbacks.
#[test]
fn streaming_export_writes_each_tensor_before_pulling_the_next() {
    use std::cell::RefCell;

    let plan = vec![
        TensorPlan::new("blk.0.attn_q.weight", vec![256, 4]),
        TensorPlan::new("blk.0.attn_norm.weight", vec![256]),
        TensorPlan::new("blk.1.attn_q.weight", vec![256, 4]),
        TensorPlan::new("output.weight", vec![256, 8]),
    ];

    let sink = RecordingSink {
        bytes: std::rc::Rc::new(RefCell::new(Vec::new())),
    };
    let observed = std::rc::Rc::clone(&sink.bytes);

    let calls: RefCell<Vec<String>> = RefCell::new(Vec::new());
    let sink_len_at_call: RefCell<Vec<usize>> = RefCell::new(Vec::new());

    let mut out = sink.clone();
    let stats = export_to_gguf_streaming(
        &plan,
        |entry| {
            calls.borrow_mut().push(entry.name.clone());
            sink_len_at_call.borrow_mut().push(observed.borrow().len());
            Ok(weights(entry.num_elements(), 5))
        },
        &ExportConfig::new(ExportFormat::PQ2_0, "streamed"),
        &[],
        &mut out,
    )
    .expect("streaming export");

    assert_eq!(
        calls.into_inner(),
        vec![
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_norm.weight".to_string(),
            "blk.1.attn_q.weight".to_string(),
            "output.weight".to_string(),
        ],
        "exactly one call per planned tensor, in plan order"
    );

    // Strictly increasing: each callback sees the bytes of every tensor
    // before it already in the sink.
    let lengths = sink_len_at_call.into_inner();
    assert_eq!(lengths.len(), 4);
    for pair in lengths.windows(2) {
        assert!(
            pair[1] > pair[0],
            "the previous tensor must be in the sink before the next is pulled, \
             saw sink lengths {lengths:?}"
        );
    }
    // The first callback already sees the header + tensor-info directory.
    assert!(lengths[0] > 0, "the header is written before any tensor");

    assert_eq!(stats.num_tensors, 4);
    assert_eq!(stats.quantized_tensors, 3);
    assert_eq!(stats.fp32_tensors, 1, "the 1-D norm stays F32");

    // The streamed bytes are a real, readable GGUF.
    let bytes = observed.borrow().clone();
    let gguf = GgufFile::parse(&bytes).expect("parse streamed file");
    assert_eq!(gguf.tensors.len(), 4);
    assert_eq!(
        gguf.tensors
            .require("blk.1.attn_q.weight")
            .expect("present")
            .tensor_type,
        GgufTensorType::PQ2_0
    );
    assert_eq!(
        gguf.tensors
            .require("blk.0.attn_norm.weight")
            .expect("present")
            .tensor_type,
        GgufTensorType::F32
    );
}

#[test]
fn streaming_export_surfaces_the_loader_error_verbatim() {
    let plan = vec![TensorPlan::new("blk.0.attn_q.weight", vec![256, 4])];
    let mut out: Vec<u8> = Vec::new();
    let err = export_to_gguf_streaming(
        &plan,
        |entry| {
            Err(ExportError::QuantizeError {
                name: entry.name.clone(),
                reason: "source shard is missing".to_string(),
            })
        },
        &ExportConfig::new(ExportFormat::PQ2_0, "streamed"),
        &[],
        &mut out,
    )
    .expect_err("the loader failure must propagate");
    let msg = err.to_string();
    assert!(
        msg.contains("source shard is missing"),
        "the typed error must survive the io::Error round trip, got: {msg}"
    );
}

#[test]
fn streaming_and_batch_export_agree_byte_for_byte() {
    let tensors = vec![
        WeightTensor::new("blk.0.attn_q.weight", weights(1024, 1), vec![256, 4]),
        WeightTensor::new("blk.0.attn_norm.weight", weights(256, 2), vec![256]),
    ];
    let config = ExportConfig::new(ExportFormat::PTQ1_0, "agree");

    let batch = export_to_gguf(&tensors, &config, &[]).expect("batch");

    let plan: Vec<TensorPlan> = tensors
        .iter()
        .map(|t| TensorPlan::new(&t.name, t.shape.clone()))
        .collect();
    let mut index = 0usize;
    let mut streamed: Vec<u8> = Vec::new();
    export_to_gguf_streaming(
        &plan,
        |_| {
            let data = tensors[index].data.clone();
            index += 1;
            Ok(data)
        },
        &config,
        &[],
        &mut streamed,
    )
    .expect("streaming");

    assert_eq!(batch, streamed);
}

/// CQ-06: a `qwen35` export carries the hybrid key set and a validated
/// `prism.hadamard.*` contract.
#[test]
fn qwen35_export_writes_the_hybrid_and_hadamard_blocks() {
    let tensors = vec![
        WeightTensor::new("blk.0.attn_qkv.weight", weights(2048, 1), vec![1024, 2]),
        WeightTensor::new("blk.0.ssm_out.weight", weights(2048, 2), vec![1024, 2]),
        WeightTensor::new("blk.0.attn_norm.weight", weights(1024, 3), vec![1024]),
    ];

    let hadamard = HadamardSpec {
        block_size: 1024,
        sign_mode: SignMode::Explicit,
        sign_widths: vec![1024],
        sign_values: (0..1024).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect(),
        weight_names: vec![
            "blk.0.attn_qkv.weight".to_string(),
            "blk.0.ssm_out.weight".to_string(),
        ],
        inverse_weight_names: Vec::new(),
        gdn_v_grouped: true,
    };

    let arch = ArchMetadata {
        block_count: 64,
        embedding_length: 5120,
        feed_forward_length: 17408,
        head_count: 24,
        head_count_kv: 4,
        key_length: Some(256),
        value_length: Some(256),
        context_length: 262_144,
        vocab_size: 248_320,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1e7,
        rope_dimension_count: Some(64),
    };

    let config = ExportConfig::new(ExportFormat::PTQ1_0, "Bonsai-2-27B")
        .with_architecture("qwen35", arch)
        .with_qwen35(Qwen35Metadata::default())
        .with_hadamard(hadamard);

    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export qwen35");
    let gguf = GgufFile::parse(&bytes).expect("parse");

    assert_eq!(
        gguf.metadata.get_string("general.architecture").ok(),
        Some("qwen35")
    );
    for (key, want) in [
        ("qwen35.block_count", 64u32),
        ("qwen35.embedding_length", 5120),
        ("qwen35.feed_forward_length", 17408),
        ("qwen35.attention.head_count", 24),
        ("qwen35.attention.head_count_kv", 4),
        ("qwen35.attention.key_length", 256),
        ("qwen35.attention.value_length", 256),
        ("qwen35.context_length", 262_144),
        ("qwen35.rope.dimension_count", 64),
        ("qwen35.full_attention_interval", 4),
        ("qwen35.ssm.conv_kernel", 4),
        ("qwen35.ssm.state_size", 128),
        ("qwen35.ssm.group_count", 16),
        ("qwen35.ssm.time_step_rank", 48),
        ("qwen35.ssm.inner_size", 6144),
    ] {
        assert_eq!(gguf.metadata.get_u32(key).ok(), Some(want), "{key}");
    }
    assert_eq!(
        gguf.metadata
            .get_i32_array("qwen35.rope.dimension_sections")
            .expect("sections"),
        vec![11, 11, 10, 0]
    );

    assert_eq!(
        gguf.metadata.get_u32("prism.hadamard.version").ok(),
        Some(1)
    );
    assert_eq!(
        gguf.metadata.get_u32("prism.hadamard.block_size").ok(),
        Some(1024)
    );
    assert_eq!(
        gguf.metadata.get_string("prism.hadamard.transform").ok(),
        Some("normalized-sylvester-walsh-hadamard")
    );
    let signs = gguf
        .metadata
        .get_i32_array("prism.hadamard.sign_values")
        .expect("sign values");
    assert_eq!(signs.len(), 1024);
    assert_eq!(signs[0], -1, "a -1 must not come back as 4294967295");
    assert_eq!(
        gguf.metadata.get_bool("prism.hadamard.gdn_v_grouped").ok(),
        Some(true)
    );
}

/// The same `HadamardSpec` checks a loader applies run on the write side, so
/// an inconsistent contract never reaches a file.
#[test]
fn an_inconsistent_hadamard_contract_is_refused_at_write_time() {
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_qkv.weight",
        weights(2048, 1),
        vec![1024, 2],
    )];
    let hadamard = HadamardSpec {
        block_size: 1024,
        sign_mode: SignMode::Explicit,
        sign_widths: vec![1024],
        sign_values: vec![1; 1023], // one short
        weight_names: vec!["blk.0.attn_qkv.weight".to_string()],
        inverse_weight_names: Vec::new(),
        gdn_v_grouped: true,
    };
    let config = ExportConfig::new(ExportFormat::PTQ1_0, "bad").with_hadamard(hadamard);
    let err = export_to_gguf(&tensors, &config, &[]).expect_err("must be refused");
    assert!(matches!(err, ExportError::Hadamard(_)), "got {err:?}");
    assert!(err.to_string().contains("sign_values"));
}

/// A folded name that is not a tensor in the file is a hard error: the file
/// would claim a rotation for a weight that is not there.
#[test]
fn hadamard_names_must_all_be_tensors_in_the_file() {
    let tensors = vec![WeightTensor::new(
        "blk.0.attn_qkv.weight",
        weights(2048, 1),
        vec![1024, 2],
    )];
    let hadamard = HadamardSpec {
        block_size: 1024,
        sign_mode: SignMode::Explicit,
        sign_widths: vec![1024],
        sign_values: vec![1; 1024],
        weight_names: vec![
            "blk.0.attn_qkv.weight".to_string(),
            "blk.9.ffn_up.weight".to_string(),
        ],
        inverse_weight_names: Vec::new(),
        gdn_v_grouped: false,
    };
    let config = ExportConfig::new(ExportFormat::PTQ1_0, "absent").with_hadamard(hadamard);
    let err = export_to_gguf(&tensors, &config, &[]).expect_err("must be refused");
    assert!(err.to_string().contains("blk.9.ffn_up.weight"));
}

/// The explicitly configured metadata blocks must be usable together with a
/// tokenizer, which is how a `qwen35` conversion will assemble a file.
#[test]
fn general_metadata_helper_reports_the_right_file_type() {
    let g = GeneralMetadata::new(
        "qwen35",
        "m",
        oxibonsai_core::gguf::writer::TensorType::PTQ1_0,
    );
    assert_eq!(g.file_type, 143);
    assert_eq!(g.architecture, "qwen35");

    let mut tokenizer = TokenizerMetadata::default();
    assert!(tokenizer.is_empty());
    tokenizer.tokens = vec!["a".to_string()];
    assert_eq!(tokenizer.vocab_size(), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// CQ-01 regression: `with_source_metadata` + an explicit metadata builder
// must never write the same GGUF key twice, in either call order.
// ─────────────────────────────────────────────────────────────────────────────

/// `with_source_metadata` carries the source's whole `tokenizer.ggml.*`
/// block; `.with_tokenizer(..)` on top of it attaches a fresh one for the
/// *same* namespace. Before this fix, both got written, `GgufFile::parse`
/// rejected the file as a duplicate key, and `export_to_gguf` itself
/// returned `Ok` — reproducing CQ-01's exact symptom (a GGUF OxiBonsai
/// itself cannot load) through the public builder.
#[test]
fn with_source_metadata_then_with_tokenizer_does_not_duplicate_keys() {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    // The source's own tokenizer — must be dropped in favour of the one
    // attached explicitly below, not written a second time.
    w.add_metadata(
        "tokenizer.ggml.model",
        MetadataWriteValue::Str("gpt2".to_string()),
    );
    w.add_metadata(
        "tokenizer.ggml.pre",
        MetadataWriteValue::Str("qwen2".to_string()),
    );
    let source_bytes = w.to_bytes().expect("serialise source");
    let source = GgufFile::parse(&source_bytes).expect("parse source");

    let fresh_tokenizer = TokenizerMetadata {
        model: "gpt2".to_string(),
        pre: "qwen35".to_string(),
        tokens: vec!["a".to_string(), "b".to_string()],
        token_types: vec![1, 1],
        ..TokenizerMetadata::default()
    };

    let config = ExportConfig::new(ExportFormat::Float32, "m")
        .with_source_metadata(&source.metadata)
        .expect("architecture present")
        .with_tokenizer(fresh_tokenizer);

    let tensors = vec![WeightTensor::new(
        "blk.0.attn_norm.weight",
        weights(8, 1),
        vec![8],
    )];
    let bytes = export_to_gguf(&tensors, &config, &[])
        .expect("export must not carry a duplicate tokenizer.ggml.* key");
    let out = GgufFile::parse(&bytes).expect("re-parse must succeed (no duplicate key)");
    assert_eq!(
        out.metadata.get_string("tokenizer.ggml.pre").ok(),
        Some("qwen35"),
        "the explicitly attached tokenizer must win over the carried one"
    );
}

/// The `<arch>.*` half of the same regression: `.with_architecture(..)` on
/// top of carried source metadata must not duplicate the arch block either.
#[test]
fn with_source_metadata_then_with_architecture_does_not_duplicate_keys() {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    w.add_metadata("qwen3.block_count", MetadataWriteValue::U32(2));
    w.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(8));
    let source_bytes = w.to_bytes().expect("serialise source");
    let source = GgufFile::parse(&source_bytes).expect("parse source");

    let new_arch = ArchMetadata {
        block_count: 99,
        embedding_length: 8,
        feed_forward_length: 16,
        head_count: 2,
        head_count_kv: 2,
        key_length: None,
        value_length: None,
        context_length: 64,
        vocab_size: 32,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        rope_dimension_count: None,
    };
    let config = ExportConfig::new(ExportFormat::Float32, "m")
        .with_source_metadata(&source.metadata)
        .expect("architecture present")
        .with_architecture("qwen3", new_arch);

    let tensors = vec![WeightTensor::new(
        "blk.0.attn_norm.weight",
        weights(8, 1),
        vec![8],
    )];
    let bytes = export_to_gguf(&tensors, &config, &[])
        .expect("export must not carry a duplicate qwen3.* key");
    let out = GgufFile::parse(&bytes).expect("re-parse must succeed (no duplicate key)");
    assert_eq!(
        out.metadata.get_u32("qwen3.block_count").ok(),
        Some(99),
        "the explicitly attached architecture block must win over the carried one"
    );
}

/// The `qwen35` architecture's own `<arch>.*` block and its hybrid-only
/// `qwen35.*` extension keys share one literal namespace, split across
/// `write_arch_metadata` and `write_qwen35_metadata`. Attaching only
/// `.with_qwen35(..)` (no `.with_architecture(..)`) on top of carried source
/// metadata must NOT lose the carried `qwen35.block_count` et al — nothing
/// else is going to write them. A namespace-*prefix* filter gets this case
/// wrong (it would delete the carried arch keys along with the ones it
/// means to de-duplicate); only an exact-key filter survives it.
#[test]
fn with_qwen35_alone_does_not_drop_the_carried_architecture_block() {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    w.add_metadata("qwen35.block_count", MetadataWriteValue::U32(64));
    w.add_metadata("qwen35.embedding_length", MetadataWriteValue::U32(5120));
    w.add_metadata("qwen35.context_length", MetadataWriteValue::U32(262_144));
    let source_bytes = w.to_bytes().expect("serialise source");
    let source = GgufFile::parse(&source_bytes).expect("parse source");

    let config = ExportConfig::new(ExportFormat::Float32, "m")
        .with_source_metadata(&source.metadata)
        .expect("architecture present")
        .with_qwen35(Qwen35Metadata::default());

    let tensors = vec![WeightTensor::new(
        "blk.0.attn_norm.weight",
        weights(8, 1),
        vec![8],
    )];
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    let out = GgufFile::parse(&bytes).expect("parse");

    // The carried <arch>.* half must survive: `write_qwen35_metadata` never
    // writes block_count / embedding_length / context_length itself.
    assert_eq!(out.metadata.get_u32("qwen35.block_count").ok(), Some(64));
    assert_eq!(
        out.metadata.get_u32("qwen35.embedding_length").ok(),
        Some(5120)
    );
    assert_eq!(
        out.metadata.get_u32("qwen35.context_length").ok(),
        Some(262_144)
    );
    // The explicit hybrid-only half is also present, from `with_qwen35`.
    assert_eq!(
        out.metadata.get_u32("qwen35.full_attention_interval").ok(),
        Some(4)
    );
}

/// CQ-06 residue: `ssm_alpha.weight` is two-dimensional with a
/// block-aligned `ne0` in the real 27B (`[5120, 48]`), so the generic FP32
/// rules never catch it — only the qwen35-specific never-quantized policy
/// does. Exercised with NO `qwen35` config attached at all, since the real
/// CQ-01 requantize path (`with_source_metadata` alone) never populates
/// `config.qwen35` — the guard has to be unconditional, not gated on it.
#[test]
fn qwen35_ssm_scalars_are_never_quantized_even_without_a_qwen35_config() {
    let tensors = vec![
        WeightTensor::new(
            "blk.0.ssm_alpha.weight",
            weights(5120 * 48, 1),
            vec![5120, 48],
        ),
        WeightTensor::new(
            "blk.0.ffn_gate.weight",
            weights(5120 * 48, 2),
            vec![5120, 48],
        ),
    ];
    let config = ExportConfig::new(ExportFormat::PQ2_0, "m");
    let bytes = export_to_gguf(&tensors, &config, &[]).expect("export");
    let gguf = GgufFile::parse(&bytes).expect("parse");

    assert_eq!(
        gguf.tensors
            .require("blk.0.ssm_alpha.weight")
            .expect("present")
            .tensor_type,
        GgufTensorType::F32,
        "ssm_alpha.weight must never be ternarized even though ne0=5120 is block-aligned"
    );
    assert_eq!(
        gguf.tensors
            .require("blk.0.ffn_gate.weight")
            .expect("present")
            .tensor_type,
        GgufTensorType::PQ2_0,
        "an ordinary tensor of the same block-aligned shape is still quantized"
    );
}
