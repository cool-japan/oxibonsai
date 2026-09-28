//! Unit tests for [`crate::engine_seam`]: the `InferenceEngine` ↔
//! `LoadedModel` seam, driven end to end on a complete synthetic `qwen35`
//! (Bonsai 2 hybrid) GGUF built in-process, plus the dense-side seam
//! behaviour (snapshots, rewinds, the backend knob).
//!
//! The hybrid fixture carries every tensor name, dtype and shape
//! relationship of the real 27B file (full-attention `q|gate` layers,
//! Gated-DeltaNet layers, a `prism.hadamard.*` fold), at a width small enough
//! to run in a debug build.

use super::*;

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::gpu_backend::UNATTRIBUTED_MODEL_EPOCH;
use oxibonsai_kernels::KernelTier;
use oxibonsai_model::quantize::{encode_quantized_tensor, ScaleRule};

use crate::sampling::SamplingParams;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic qwen35 fixture
// ─────────────────────────────────────────────────────────────────────────────

const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 512;
const N_LAYERS: usize = 8;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const VOCAB: usize = 512;
const STATE: usize = 64;
const N_K_HEADS: usize = 2;
const N_V_HEADS: usize = 6;
const CONV_KERNEL: usize = 4;
const HADAMARD_BLOCK: usize = 128;
const MAX_SEQ: usize = 64;

fn inner() -> usize {
    N_V_HEADS * STATE
}

fn conv_dim() -> usize {
    2 * STATE * N_K_HEADS + inner()
}

fn heads_width() -> usize {
    N_HEADS * HEAD_DIM
}

fn is_full(layer: usize) -> bool {
    (layer + 1).is_multiple_of(4)
}

/// Deterministic pseudo-random values in `[-1, 1)`.
fn ramp(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn quant_tensor(name: &str, ne0: usize, ne1: usize, seed: u64) -> TensorEntry {
    let values = ramp(ne0 * ne1, seed);
    let data = encode_quantized_tensor(&values, ne0, TensorType::PQ2_0, ScaleRule::AbsMax)
        .unwrap_or_else(|e| panic!("fixture {name}: {e}"));
    TensorEntry {
        name: name.to_string(),
        shape: vec![ne0 as u64, ne1 as u64],
        tensor_type: TensorType::PQ2_0,
        data,
    }
}

fn f32_tensor(name: &str, shape: &[usize], values: Vec<f32>) -> TensorEntry {
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in &values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    TensorEntry {
        name: name.to_string(),
        shape: shape.iter().map(|d| *d as u64).collect(),
        tensor_type: TensorType::F32,
        data,
    }
}

fn bf16_tensor(name: &str, ne0: usize, ne1: usize, seed: u64) -> TensorEntry {
    let values = ramp(ne0 * ne1, seed);
    let mut data = Vec::with_capacity(values.len() * 2);
    for v in &values {
        let bits = v.to_bits();
        let rounded = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
        data.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
    TensorEntry {
        name: name.to_string(),
        shape: vec![ne0 as u64, ne1 as u64],
        tensor_type: TensorType::BF16,
        data,
    }
}

/// A complete synthetic `qwen35` GGUF: PQ2_0 weights, Hadamard-folded,
/// grouped v-heads, EOS id 3.
pub(crate) fn synthetic_qwen35_gguf() -> Vec<u8> {
    let u32v = |v: usize| MetadataWriteValue::U32(v as u32);
    let mut writer = GgufWriter::new();
    writer
        .add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen35".into()),
        )
        .add_metadata(
            "general.name",
            MetadataWriteValue::Str("synthetic-qwen35-engine".into()),
        )
        .add_metadata("qwen35.embedding_length", u32v(HIDDEN))
        .add_metadata("qwen35.feed_forward_length", u32v(INTERMEDIATE))
        .add_metadata("qwen35.block_count", u32v(N_LAYERS))
        .add_metadata("qwen35.attention.head_count", u32v(N_HEADS))
        .add_metadata("qwen35.attention.head_count_kv", u32v(N_KV_HEADS))
        .add_metadata("qwen35.attention.key_length", u32v(HEAD_DIM))
        .add_metadata("qwen35.attention.value_length", u32v(HEAD_DIM))
        .add_metadata("qwen35.vocab_size", u32v(VOCAB))
        .add_metadata("qwen35.context_length", u32v(4096))
        .add_metadata(
            "qwen35.attention.layer_norm_rms_epsilon",
            MetadataWriteValue::F32(1e-6),
        )
        .add_metadata("qwen35.rope.freq_base", MetadataWriteValue::F32(1e7))
        .add_metadata("qwen35.rope.dimension_count", u32v(16))
        .add_metadata(
            "qwen35.rope.dimension_sections",
            MetadataWriteValue::ArrayU32(vec![3, 3, 2, 0]),
        )
        .add_metadata("qwen35.full_attention_interval", u32v(4))
        .add_metadata("qwen35.ssm.conv_kernel", u32v(CONV_KERNEL))
        .add_metadata("qwen35.ssm.state_size", u32v(STATE))
        .add_metadata("qwen35.ssm.group_count", u32v(N_K_HEADS))
        .add_metadata("qwen35.ssm.time_step_rank", u32v(N_V_HEADS))
        .add_metadata("qwen35.ssm.inner_size", u32v(inner()))
        .add_metadata("tokenizer.ggml.eos_token_id", u32v(3));

    let mut widths = vec![HIDDEN, heads_width(), inner(), INTERMEDIATE];
    widths.sort_unstable();
    widths.dedup();
    let mut sign_values: Vec<i32> = Vec::new();
    for width in &widths {
        for i in 0..*width {
            sign_values.push(if i % 3 == 0 { -1 } else { 1 });
        }
    }
    let mut weight_names = vec!["output.weight".to_string()];
    for layer in 0..N_LAYERS {
        let folded: &[&str] = if is_full(layer) {
            &[
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ]
        } else {
            &[
                "attn_qkv",
                "attn_gate",
                "ssm_out",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ]
        };
        for suffix in folded {
            weight_names.push(format!("blk.{layer}.{suffix}.weight"));
        }
    }
    writer
        .add_metadata("prism.hadamard.version", MetadataWriteValue::U32(1))
        .add_metadata("prism.hadamard.block_size", u32v(HADAMARD_BLOCK))
        .add_metadata(
            "prism.hadamard.transform",
            MetadataWriteValue::Str("normalized-sylvester-walsh-hadamard".into()),
        )
        .add_metadata(
            "prism.hadamard.axis",
            MetadataWriteValue::Str("input-last-dimension".into()),
        )
        .add_metadata(
            "prism.hadamard.sign_mode",
            MetadataWriteValue::Str("explicit".into()),
        )
        .add_metadata(
            "prism.hadamard.sign_widths",
            MetadataWriteValue::ArrayU32(widths.iter().map(|w| *w as u32).collect()),
        )
        .add_metadata(
            "prism.hadamard.sign_values",
            MetadataWriteValue::ArrayI32(sign_values),
        )
        .add_metadata(
            "prism.hadamard.weight_names",
            MetadataWriteValue::ArrayStr(weight_names),
        )
        .add_metadata(
            "prism.hadamard.inverse_weight_names",
            MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
        )
        .add_metadata(
            "prism.hadamard.gdn_v_grouped",
            MetadataWriteValue::Bool(true),
        );

    let mut seed = 1u64;
    let mut next_seed = || {
        seed = seed.wrapping_add(7919);
        seed
    };
    writer.add_tensor(quant_tensor(
        "token_embd.weight",
        HIDDEN,
        VOCAB,
        next_seed(),
    ));
    writer.add_tensor(quant_tensor("output.weight", HIDDEN, VOCAB, next_seed()));
    writer.add_tensor(f32_tensor(
        "output_norm.weight",
        &[HIDDEN],
        vec![1.0; HIDDEN],
    ));
    for layer in 0..N_LAYERS {
        let blk = |suffix: &str| format!("blk.{layer}.{suffix}");
        writer.add_tensor(f32_tensor(
            &blk("attn_norm.weight"),
            &[HIDDEN],
            vec![1.0; HIDDEN],
        ));
        writer.add_tensor(f32_tensor(
            &blk("post_attention_norm.weight"),
            &[HIDDEN],
            vec![1.0; HIDDEN],
        ));
        writer.add_tensor(quant_tensor(
            &blk("ffn_gate.weight"),
            HIDDEN,
            INTERMEDIATE,
            next_seed(),
        ));
        writer.add_tensor(quant_tensor(
            &blk("ffn_up.weight"),
            HIDDEN,
            INTERMEDIATE,
            next_seed(),
        ));
        writer.add_tensor(quant_tensor(
            &blk("ffn_down.weight"),
            INTERMEDIATE,
            HIDDEN,
            next_seed(),
        ));
        if is_full(layer) {
            writer.add_tensor(quant_tensor(
                &blk("attn_q.weight"),
                HIDDEN,
                heads_width() * 2,
                next_seed(),
            ));
            writer.add_tensor(quant_tensor(
                &blk("attn_k.weight"),
                HIDDEN,
                N_KV_HEADS * HEAD_DIM,
                next_seed(),
            ));
            writer.add_tensor(quant_tensor(
                &blk("attn_v.weight"),
                HIDDEN,
                N_KV_HEADS * HEAD_DIM,
                next_seed(),
            ));
            writer.add_tensor(quant_tensor(
                &blk("attn_output.weight"),
                heads_width(),
                HIDDEN,
                next_seed(),
            ));
            writer.add_tensor(f32_tensor(
                &blk("attn_q_norm.weight"),
                &[HEAD_DIM],
                vec![1.0; HEAD_DIM],
            ));
            writer.add_tensor(f32_tensor(
                &blk("attn_k_norm.weight"),
                &[HEAD_DIM],
                vec![1.0; HEAD_DIM],
            ));
        } else {
            writer.add_tensor(quant_tensor(
                &blk("attn_qkv.weight"),
                HIDDEN,
                conv_dim(),
                next_seed(),
            ));
            writer.add_tensor(quant_tensor(
                &blk("attn_gate.weight"),
                HIDDEN,
                inner(),
                next_seed(),
            ));
            writer.add_tensor(quant_tensor(
                &blk("ssm_out.weight"),
                inner(),
                HIDDEN,
                next_seed(),
            ));
            writer.add_tensor(bf16_tensor(
                &blk("ssm_alpha.weight"),
                HIDDEN,
                N_V_HEADS,
                next_seed(),
            ));
            writer.add_tensor(bf16_tensor(
                &blk("ssm_beta.weight"),
                HIDDEN,
                N_V_HEADS,
                next_seed(),
            ));
            writer.add_tensor(f32_tensor(
                &blk("ssm_conv1d.weight"),
                &[CONV_KERNEL, conv_dim()],
                ramp(CONV_KERNEL * conv_dim(), next_seed()),
            ));
            writer.add_tensor(f32_tensor(
                &blk("ssm_a"),
                &[N_V_HEADS],
                (0..N_V_HEADS).map(|h| -0.25 - (h as f32) * 0.5).collect(),
            ));
            writer.add_tensor(f32_tensor(
                &blk("ssm_dt.bias"),
                &[N_V_HEADS],
                (0..N_V_HEADS).map(|h| -2.0 + (h as f32) * 0.125).collect(),
            ));
            writer.add_tensor(f32_tensor(
                &blk("ssm_norm.weight"),
                &[STATE],
                vec![1.0; STATE],
            ));
        }
    }
    writer
        .to_bytes()
        .expect("synthetic qwen35 fixture serialises")
}

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 8,
    }
}

fn argmax(row: &[f32]) -> u32 {
    crate::engine_greedy::argmax_first(row)
}

/// Bit patterns of a logit row, for exact (not approximate) comparisons.
fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

const PROMPT: [u32; 5] = [7, 11, 13, 17, 19];

// ─────────────────────────────────────────────────────────────────────────────
// Backend knob and typed errors
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn backend_parses_displays_and_defaults_to_auto() {
    assert_eq!(Backend::default(), Backend::Auto);
    for backend in [Backend::Auto, Backend::Cpu, Backend::Metal] {
        assert_eq!(Backend::parse(backend.as_str()), Some(backend));
        assert_eq!(backend.to_string().parse::<Backend>(), Ok(backend));
    }
    assert_eq!(Backend::parse(" CPU "), Some(Backend::Cpu));
    assert_eq!(Backend::parse("gpu"), Some(Backend::Metal));
    assert_eq!(Backend::parse("cuda"), None);
    assert!("vulkan".parse::<Backend>().is_err());
}

#[test]
fn every_engine_error_round_trips_its_code_through_runtime_error() {
    let errors = [
        EngineError::NotADenseModel {
            operation: "op",
            architecture: "qwen35".into(),
            reason: "why",
        },
        EngineError::RecurrentRollbackRequired {
            operation: "op",
            architecture: "qwen35".into(),
        },
        EngineError::HybridGpuBackendUnsupported {
            requested: Backend::Metal,
            architecture: "qwen35".into(),
        },
        EngineError::BackendUnavailable {
            requested: Backend::Metal,
            reason: "none".into(),
        },
        EngineError::NonContiguousPosition {
            expected: 3,
            got: 5,
        },
        EngineError::SharedEmbeddingUnsupported {
            architecture: "qwen35".into(),
            len: 7,
        },
        EngineError::RecurrentStateNotSnapshotable { name: "x".into() },
        EngineError::SnapshotMismatch { detail: "d".into() },
    ];
    assert_eq!(errors.len(), EngineError::ALL_CODES.len());
    for error in errors {
        let code = error.error_code();
        assert!(EngineError::ALL_CODES.contains(&code));
        let runtime: RuntimeError = error.into();
        assert_eq!(engine_error_code(&runtime), Some(code), "{runtime}");
    }
    // A non-engine error carries no engine code, even one that looks close.
    assert_eq!(
        engine_error_code(&RuntimeError::Config("[NOT_A_CODE] x".into())),
        None
    );
    assert_eq!(engine_error_code(&RuntimeError::CircuitOpen), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid engine: construction and introspection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_qwen35_gguf_loads_as_a_cpu_hybrid_engine() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");

    assert!(engine.is_hybrid());
    assert!(engine.dense_model().is_none());
    assert!(engine.hybrid_model().is_some());
    assert!(engine.loaded_model().is_hybrid());
    assert_eq!(engine.architecture(), "qwen35");
    assert_eq!(engine.model_name(), "synthetic-qwen35-engine");
    assert_eq!(engine.vocab_size(), VOCAB);
    assert_eq!(engine.hidden_size(), HIDDEN);
    assert_eq!(engine.num_layers(), N_LAYERS);
    assert_eq!(engine.max_seq_len(), MAX_SEQ);
    assert_eq!(engine.max_context(), MAX_SEQ);
    assert_eq!(engine.context_length(), 4096);
    assert_eq!(engine.sequence_position(), 0);
    // KV only for the two full-attention layers.
    assert_eq!(engine.kv_cache_geometry(), (2, N_KV_HEADS, HEAD_DIM));
    assert!(engine.recurrent_memory_bytes() > 0);
    assert!(engine.sequence_state_bytes() > engine.recurrent_memory_bytes());
    // CPU only: no hybrid GPU encoder exists yet.
    assert!(!engine.uses_fused_gpu_decode());
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    assert_ne!(engine.kernel_tier(), KernelTier::Gpu);
    assert_eq!(engine.model_epoch(), UNATTRIBUTED_MODEL_EPOCH);
    assert_eq!(engine.backend(), Backend::Auto);
    // cli-16: the label names the resolved quant type, not a guessed family.
    assert_eq!(
        engine.dominant_quant_type(),
        oxibonsai_core::GgufTensorType::PQ2_0
    );
    assert!(
        engine.kernel_label().starts_with("PQ2_0 "),
        "{}",
        engine.kernel_label()
    );
    assert!(engine.model_description().contains("qwen35"));
    // EOS resolved from the file, not the Qwen3 fallback.
    assert!(engine.is_eos(3));
    assert!(!engine.recurrent_rollback_supported());
    // The shared-embedding handle a pool hands replicas 2..N is empty.
    assert!(engine.model_token_embd().is_empty());
    assert!(matches!(
        engine.require_dense("test"),
        Err(EngineError::NotADenseModel { .. })
    ));
    // The tier explanation says why a hybrid runs on the CPU, rather than the
    // pinned CPU dispatcher's own "explicitly requested".
    let reason = engine.effective_tier_reason();
    assert!(
        reason.contains("hybrid `qwen35` model") && reason.contains("backend=auto"),
        "{reason}"
    );
}

/// What `oxibonsai serve` builds for a hybrid file: a pool that defaults to
/// one CPU replica (an explicit size is honoured) and no embedder — the
/// embedding engine is refused with the typed code before a second model
/// instance is ever built.
#[test]
fn a_hybrid_pool_defaults_to_one_cpu_replica_and_refuses_an_embedder() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path = std::env::temp_dir().join(format!(
        "oxibonsai_engine_seam_pool_{}_{stamp}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, synthetic_qwen35_gguf()).expect("write the fixture");

    let built = crate::engine_pool::build_pool_from_gguf_parts(
        &path,
        greedy_params(),
        42,
        MAX_SEQ,
        None,
        Backend::Auto,
    )
    .expect("hybrid pool");
    assert!(built.hybrid);
    assert_eq!(built.size, 1, "an unspecified size is one hybrid replica");
    assert_eq!(built.pool.size(), 1);
    assert!(built.shared_token_embd.is_empty());
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    assert_ne!(built.tier, KernelTier::Gpu);

    let tokenizer = Arc::new(TokenizerBridge::from_native_tokenizer(
        oxibonsai_tokenizer::OxiTokenizer::char_level_stub(VOCAB),
    ));
    let refusal = crate::embed_engine::ModelEmbedder::from_static_gguf(
        built.gguf,
        Arc::clone(&built.shared_token_embd),
        tokenizer,
        greedy_params(),
        42,
        MAX_SEQ,
    )
    .expect_err("a hybrid model has no embedder yet");
    assert_eq!(engine_error_code(&refusal), Some("NOT_A_DENSE_MODEL"));

    let two = crate::engine_pool::build_pool_from_gguf_parts(
        &path,
        greedy_params(),
        42,
        MAX_SEQ,
        Some(2),
        Backend::Auto,
    )
    .expect("explicitly sized hybrid pool");
    assert_eq!(two.size, 2, "an explicit size is honoured");
    assert_eq!(two.pool.size(), 2);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn backend_cpu_loads_the_hybrid_and_backend_metal_is_a_typed_refusal() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu =
        InferenceEngine::from_gguf_with_backend(&gguf, greedy_params(), 42, MAX_SEQ, Backend::Cpu)
            .expect("cpu hybrid engine");
    assert_eq!(cpu.backend(), Backend::Cpu);
    assert!(cpu.is_hybrid());

    let err = InferenceEngine::from_gguf_with_backend(
        &gguf,
        greedy_params(),
        42,
        MAX_SEQ,
        Backend::Metal,
    )
    .err()
    .expect("an explicit Metal request for a hybrid model must be refused");
    assert_eq!(
        engine_error_code(&err),
        Some("HYBRID_GPU_BACKEND_UNSUPPORTED"),
        "{err}"
    );
}

#[test]
fn a_shared_dense_embedding_table_is_refused_for_a_hybrid_file() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let table: std::sync::Arc<[f32]> = std::sync::Arc::from(vec![0.5f32; VOCAB * HIDDEN]);
    let err = InferenceEngine::from_gguf_with_embd(&gguf, greedy_params(), 42, MAX_SEQ, table)
        .err()
        .expect("a dense table cannot stand in for the rotated hybrid embedding");
    assert_eq!(
        engine_error_code(&err),
        Some("SHARED_EMBEDDING_UNSUPPORTED"),
        "{err}"
    );
    // The empty handle a hybrid engine itself hands out is accepted.
    let ok = InferenceEngine::from_gguf_with_embd(
        &gguf,
        greedy_params(),
        42,
        MAX_SEQ,
        std::sync::Arc::from(Vec::new()),
    );
    assert!(ok.is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid engine: generation paths
// ─────────────────────────────────────────────────────────────────────────────

/// The engine's greedy `generate` must be exactly the hybrid model's own
/// prefill + decode + first-index argmax: the seam adds dispatch, never
/// arithmetic.
#[test]
fn hybrid_generate_equals_the_model_driven_by_hand() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let via_engine = engine.generate(&PROMPT, 6).expect("engine generate");
    assert_eq!(via_engine.len(), 6.min(via_engine.len()));
    assert!(!via_engine.is_empty());

    let config = oxibonsai_model::hybrid::HybridModel::config_from_gguf(&gguf).expect("config");
    let kernel = std::sync::Arc::new(crate::engine_seam::cpu_dispatcher());
    let mut model =
        oxibonsai_model::hybrid::HybridModel::from_gguf_with(&gguf, config, MAX_SEQ, &kernel)
            .expect("hybrid model");
    let mut logits = vec![0.0f32; VOCAB];
    model
        .forward_prefill(&PROMPT, 0, &mut logits)
        .expect("prefill");
    let mut by_hand = Vec::new();
    for pos in PROMPT.len()..PROMPT.len() + via_engine.len() {
        let next = argmax(&logits);
        by_hand.push(next);
        model.forward(next, pos, &mut logits).expect("decode");
    }
    assert_eq!(via_engine, by_hand);
}

/// Every public generation entry point decodes the same greedy sequence on a
/// hybrid engine, and a second `generate` without an explicit reset starts a
/// fresh sequence (an implicit reset at position 0) instead of erroring or
/// carrying the recurrent state over.
#[test]
fn every_generation_entry_point_agrees_on_a_hybrid_engine() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");

    let first = engine.generate(&PROMPT, 5).expect("generate");
    let again = engine
        .generate(&PROMPT, 5)
        .expect("generate again, no reset");
    assert_eq!(
        first, again,
        "an implicit restart must not leak recurrent state"
    );

    let mut tracker = crate::request_metrics::RequestRateTracker::new();
    let tracked = engine
        .generate_tracked(&PROMPT, 5, &mut tracker)
        .expect("tracked");
    assert_eq!(first, tracked);

    let (tx, rx) = std::sync::mpsc::channel();
    let count = engine
        .generate_streaming_sync(&PROMPT, 5, &tx)
        .expect("streaming sync");
    drop(tx);
    let streamed: Vec<u32> = rx.into_iter().collect();
    assert_eq!(count, streamed.len());
    assert_eq!(first, streamed);

    #[cfg(feature = "server")]
    {
        let (tokens, logprobs) = engine
            .generate_with_logprobs(&PROMPT, 5, 3, &|id| id.to_string())
            .expect("logprobs");
        assert_eq!(first, tokens);
        assert_eq!(logprobs.len(), tokens.len());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let greedy = engine
            .generate_greedy_gpu(&PROMPT, 5)
            .expect("greedy entry");
        assert_eq!(
            first, greedy,
            "the explicit greedy-GPU entry point must fall back to the CPU for a hybrid"
        );
    }

    let batch = engine.batch_generate(&[PROMPT.to_vec(), PROMPT.to_vec()], 5);
    for result in batch {
        assert_eq!(result.expect("batch item").generated_tokens, first);
    }
}

/// Chunked prefill through the engine (the cancellation-aware split) and the
/// model's own internal chunking give the same continuation as one shot.
#[test]
fn hybrid_engine_prefill_chunking_is_invisible() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut single =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let mut chunked =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    chunked.set_prefill_chunk_tokens(Some(2));
    let a = single.generate(&PROMPT, 4).expect("single-shot");
    let b = chunked.generate(&PROMPT, 4).expect("chunked");
    assert_eq!(a, b);
}

#[test]
fn hybrid_engine_honours_cancellation_before_prefill() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let token = engine.arm_cancellation();
    token.cancel();
    assert!(engine.generate(&PROMPT, 4).expect("cancelled").is_empty());
}

#[test]
fn reset_clears_both_the_kv_cursor_and_the_recurrent_state() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
    assert_eq!(engine.sequence_position(), PROMPT.len());
    let model = engine.hybrid_model().expect("hybrid");
    assert_eq!(model.kv_cache().seq_len(), PROMPT.len());
    assert!(
        model
            .recurrent()
            .ssm(0)
            .expect("slot 0")
            .iter()
            .any(|v| *v != 0.0),
        "a prefill must have written the recurrent state"
    );

    engine.reset();
    assert_eq!(engine.sequence_position(), 0);
    let model = engine.hybrid_model().expect("hybrid");
    assert_eq!(model.kv_cache().seq_len(), 0);
    assert!(model
        .recurrent()
        .ssm(0)
        .expect("slot 0")
        .iter()
        .all(|v| *v == 0.0));
}

/// An implicit restart — a forward at position 0 in the middle of a
/// sequence — clears an attached opaque `RecurrentState` exactly as an
/// explicit `reset` does: that state belongs to the abandoned sequence and,
/// unlike a KV cache, is not masked by position. A forward at position 0 on
/// a fresh sequence abandons nothing and leaves it alone.
#[test]
fn an_implicit_restart_clears_an_attached_recurrent_state() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingState(Arc<AtomicUsize>);
    impl crate::engine_control::RecurrentState for CountingState {
        fn reset_recurrent(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn recurrent_name(&self) -> &str {
            "counting"
        }
    }

    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let resets = Arc::new(AtomicUsize::new(0));
    engine.set_recurrent_state(Box::new(CountingState(Arc::clone(&resets))));

    // A fresh sequence: nothing to abandon.
    engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
    assert_eq!(resets.load(Ordering::SeqCst), 0);
    assert_eq!(engine.sequence_position(), PROMPT.len());

    // Prefill again from position 0 without a reset: implicit restart.
    engine
        .prefill_from_pos(&PROMPT, 0)
        .expect("implicit restart via prefill");
    assert_eq!(
        resets.load(Ordering::SeqCst),
        1,
        "the implicit restart must clear the attached recurrent state"
    );
    assert_eq!(engine.sequence_position(), PROMPT.len());

    // The explicit reset clears it too (unchanged).
    engine.reset();
    assert_eq!(resets.load(Ordering::SeqCst), 2);

    // Position 0 on the freshly reset sequence abandons nothing ...
    engine
        .decode_step(PROMPT[0], 0)
        .expect("decode at position 0 on a fresh sequence");
    assert_eq!(resets.load(Ordering::SeqCst), 2);
    // ... while position 0 one token in is an implicit restart again.
    engine
        .decode_step(PROMPT[1], 0)
        .expect("implicit restart via decode");
    assert_eq!(resets.load(Ordering::SeqCst), 3);
    assert_eq!(engine.sequence_position(), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid engine: typed refusals
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hybrid_engine_refuses_rollback_and_position_gaps_with_typed_errors() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    engine.prefill_from_pos(&PROMPT, 0).expect("prefill");

    // Rolling back to the current position is the identity.
    engine
        .try_rewind_cache(PROMPT.len())
        .expect("identity rewind");
    // Anything earlier is the typed model error, with the real token count.
    match engine.try_rewind_cache(2) {
        Err(RuntimeError::Model(ModelError::RecurrentRollbackUnsupported { pos, tokens })) => {
            assert_eq!(pos, 2);
            assert_eq!(tokens, PROMPT.len());
        }
        other => panic!("expected RecurrentRollbackUnsupported, got {other:?}"),
    }
    // The infallible spelling must not have moved anything.
    engine.rewind_cache(1);
    assert_eq!(engine.sequence_position(), PROMPT.len());

    // A decode at an earlier position is a rollback too.
    assert!(matches!(
        engine.decode_step(1, 2),
        Err(RuntimeError::Model(
            ModelError::RecurrentRollbackUnsupported { .. }
        ))
    ));
    // A gap is its own typed refusal.
    let gap = engine
        .decode_step(1, PROMPT.len() + 3)
        .expect_err("a position gap must be refused");
    assert_eq!(engine_error_code(&gap), Some("NON_CONTIGUOUS_POSITION"));
    // The contiguous position still works.
    engine
        .decode_step(1, PROMPT.len())
        .expect("contiguous decode");
    assert_eq!(engine.sequence_position(), PROMPT.len() + 1);
}

#[test]
fn hybrid_engine_refuses_dense_only_features_with_typed_errors() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");

    let verify = engine
        .verify_batch(&PROMPT, 0)
        .expect_err("verification needs a rollback");
    assert_eq!(
        engine_error_code(&verify),
        Some("RECURRENT_ROLLBACK_REQUIRED")
    );

    let embed = engine
        .embed(&PROMPT)
        .expect_err("the hybrid has no forward_hidden yet");
    assert_eq!(engine_error_code(&embed), Some("NOT_A_DENSE_MODEL"));

    let snapshot_ok = engine.snapshot_sequence();
    assert!(snapshot_ok.is_ok(), "a hybrid sequence IS snapshot-able");
}

#[test]
fn speculative_decoding_and_prefix_caching_refuse_a_hybrid_engine() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let hybrid =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let err = crate::speculative::SpeculativeDecoder::try_new(
        hybrid,
        crate::speculative::SpeculativeConfig::default(),
    )
    .err()
    .expect("a hybrid draft engine must be refused");
    assert_eq!(engine_error_code(&err), Some("RECURRENT_ROLLBACK_REQUIRED"));

    // A dense draft against a hybrid target: refused at generate time.
    let draft = InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        greedy_params(),
        42,
    );
    let mut decoder = crate::speculative::SpeculativeDecoder::new(
        draft,
        crate::speculative::SpeculativeConfig::default(),
    );
    let mut target =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let err = decoder
        .generate_verified(&mut target, &PROMPT, 4, &greedy_params())
        .expect_err("a hybrid target must be refused");
    assert_eq!(engine_error_code(&err), Some("RECURRENT_ROLLBACK_REQUIRED"));

    let hybrid =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let err = crate::prefix_cache_engine::PrefixCachedEngine::try_new(hybrid, 8, 42)
        .err()
        .expect("prefix caching a recurrence must be refused");
    assert_eq!(engine_error_code(&err), Some("RECURRENT_ROLLBACK_REQUIRED"));

    // The infallible constructor serves it uncached -- correct output, empty trie.
    let hybrid =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let mut reference =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let expected = reference.generate(&PROMPT, 3).expect("reference");
    let mut wrapped = crate::prefix_cache_engine::PrefixCachedEngine::new(hybrid, 8, 42);
    let params = SamplingParams {
        max_tokens: 3,
        ..greedy_params()
    };
    assert_eq!(wrapped.generate(&PROMPT, &params), expected);
    assert_eq!(wrapped.cache_stats().cached_blocks, 0);
}

/// Beam search on a hybrid engine replays instead of rolling back, and so
/// agrees exactly with the naive re-prefill-every-beam reference.
#[test]
fn beam_search_on_a_hybrid_engine_replays_exactly() {
    use crate::beam_search::{BeamSearchConfig, BeamSearchEngine};
    use crate::pipeline::PipelineBuilder;

    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cfg = BeamSearchConfig {
        beam_width: 2,
        max_tokens: 3,
        eos_token_id: 3,
        early_stopping: false,
        ..Default::default()
    };

    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let mut pipeline = PipelineBuilder::new().with_beam_search(cfg.clone()).build();
    let piped = pipeline.run(PROMPT.to_vec(), &mut engine);
    assert!(
        !matches!(piped.stop_reason, crate::pipeline::StopReason::Error(_)),
        "beam search on a hybrid engine must not error: {:?}",
        piped.stop_reason
    );

    let mut naive_engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let naive = BeamSearchEngine::new(cfg).search(PROMPT.to_vec(), VOCAB, |beam, _| {
        naive_engine.reset();
        naive_engine
            .prefill_from_pos(beam, 0)
            .expect("naive prefill")
    });
    let best = naive.best();
    let naive_generated = if best.len() > PROMPT.len() {
        best[PROMPT.len()..].to_vec()
    } else {
        Vec::new()
    };
    assert!(!piped.token_ids.is_empty());
    assert_eq!(piped.token_ids, naive_generated);
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshots
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hybrid_snapshot_restore_is_bit_exact() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let prefill = engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
    let first = argmax(&prefill);

    let snapshot = engine.snapshot_sequence().expect("snapshot");
    assert!(snapshot.has_recurrent_state());
    assert_eq!(snapshot.position(), PROMPT.len());

    let reference = engine.decode_step(first, PROMPT.len()).expect("decode");
    // Wander off: two more tokens.
    engine
        .decode_step(argmax(&reference), PROMPT.len() + 1)
        .expect("decode");
    engine.decode_step(5, PROMPT.len() + 2).expect("decode");

    engine.restore_sequence(&snapshot).expect("restore");
    assert_eq!(engine.sequence_position(), PROMPT.len());
    let replayed = engine.decode_step(first, PROMPT.len()).expect("decode");
    assert_eq!(
        reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        replayed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a restored snapshot must reproduce the next step bit for bit"
    );

    // Any reset invalidates the snapshot.
    engine.reset();
    let stale = engine
        .restore_sequence(&snapshot)
        .expect_err("a snapshot from a reset sequence must be refused");
    assert_eq!(engine_error_code(&stale), Some("SNAPSHOT_MISMATCH"));
}

#[test]
fn dense_snapshot_restore_moves_the_cursor_and_rejects_the_other_kind() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let hybrid_engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let hybrid_snapshot = hybrid_engine.snapshot_sequence().expect("snapshot");

    let config = oxibonsai_core::config::Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        vocab_size: 64,
        max_context_length: 64,
        ..oxibonsai_core::config::Qwen3Config::tiny_test()
    };
    let model = BonsaiModel::new_for_testing_with_blocks(config);
    let mut dense =
        InferenceEngine::from_model_with_tier(model, KernelTier::Reference, greedy_params(), 42);
    let prefill = dense.prefill_from_pos(&[1, 2, 3], 0).expect("prefill");
    let snapshot = dense.snapshot_sequence().expect("snapshot");
    assert!(!snapshot.has_recurrent_state());
    let reference = dense.decode_step(argmax(&prefill), 3).expect("decode");
    dense.decode_step(4, 4).expect("decode");
    dense.restore_sequence(&snapshot).expect("restore");
    assert_eq!(dense.sequence_position(), 3);
    let replayed = dense.decode_step(argmax(&prefill), 3).expect("decode");
    assert_eq!(reference, replayed);

    // A hybrid snapshot on a dense engine: wrong kind (or wrong sequence).
    let wrong = dense
        .restore_sequence(&hybrid_snapshot)
        .expect_err("a foreign snapshot must be refused");
    assert_eq!(engine_error_code(&wrong), Some("SNAPSHOT_MISMATCH"));
}

#[test]
fn an_attached_opaque_recurrent_state_is_not_snapshotable() {
    struct Opaque;
    impl crate::engine_control::RecurrentState for Opaque {
        fn reset_recurrent(&mut self) {}
        fn recurrent_name(&self) -> &str {
            "opaque"
        }
    }
    let mut engine = InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        greedy_params(),
        42,
    );
    engine.set_recurrent_state(Box::new(Opaque));
    let err = engine
        .snapshot_sequence()
        .expect_err("no snapshot hook on the trait");
    assert_eq!(
        engine_error_code(&err),
        Some("RECURRENT_STATE_NOT_SNAPSHOTABLE")
    );
    assert!(!engine.recurrent_rollback_supported());
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense engine: the seam is transparent
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn a_dense_engine_answers_the_seam_accessors_from_its_model() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config.clone(), greedy_params(), 42);
    assert!(!engine.is_hybrid());
    assert!(engine.hybrid_model().is_none());
    assert!(engine.require_dense("test").is_ok());
    assert_eq!(engine.architecture(), config.architecture);
    assert_eq!(engine.vocab_size(), config.vocab_size);
    assert_eq!(engine.hidden_size(), config.hidden_size);
    assert_eq!(engine.num_layers(), config.num_layers);
    assert_eq!(engine.context_length(), config.max_context_length);
    assert_eq!(
        engine.kv_cache_geometry(),
        (config.num_layers, config.num_kv_heads, config.head_dim)
    );
    assert_eq!(engine.recurrent_memory_bytes(), 0);
    assert!(engine.recurrent_rollback_supported());
    assert_eq!(engine.model_epoch(), UNATTRIBUTED_MODEL_EPOCH);
}

#[test]
fn dense_try_rewind_cache_still_moves_the_kv_cursor() {
    // Real (tiny) blocks: `BonsaiModel::new` has none, so its KV cursor would
    // never move and the rewind would be vacuous.
    let config = oxibonsai_core::config::Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        vocab_size: 64,
        max_context_length: 64,
        ..oxibonsai_core::config::Qwen3Config::tiny_test()
    };
    let mut engine = InferenceEngine::from_model_with_tier(
        BonsaiModel::new_for_testing_with_blocks(config),
        KernelTier::Reference,
        greedy_params(),
        42,
    );
    engine.prefill_from_pos(&[1, 2, 3, 4], 0).expect("prefill");
    assert_eq!(engine.sequence_position(), 4);
    engine.try_rewind_cache(4).expect("identity rewind");
    assert_eq!(engine.sequence_position(), 4);
    engine.try_rewind_cache(2).expect("dense rewind");
    assert_eq!(engine.sequence_position(), 2);
    engine.rewind_cache(1);
    assert_eq!(engine.sequence_position(), 1);
    // The rewound cursor is a real rollback point: re-decoding position 1
    // reproduces what a fresh prefill of the same prefix computes.
    let rewound = engine.decode_step(2, 1).expect("decode after rewind");
    engine.reset();
    let fresh = engine.prefill_from_pos(&[1, 2], 0).expect("fresh prefill");
    assert_eq!(bits(&rewound), bits(&fresh));
}

#[test]
fn backend_metal_on_a_dense_model_is_gpu_or_a_typed_refusal() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test();
    let _ = config;
    match crate::engine_seam::dense_dispatcher(Backend::Metal) {
        Ok(kernel) => {
            // Only reachable on a Metal build with an accelerated device.
            #[cfg(any(feature = "metal", feature = "native-cuda"))]
            assert_eq!(kernel.tier(), KernelTier::Gpu);
            #[cfg(not(any(feature = "metal", feature = "native-cuda")))]
            let _ = kernel;
        }
        Err(e) => assert_eq!(engine_error_code(&e), Some("BACKEND_UNAVAILABLE"), "{e}"),
    }
    let cpu = crate::engine_seam::dense_dispatcher(Backend::Cpu).expect("cpu always works");
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    assert_ne!(cpu.tier(), KernelTier::Gpu);
    #[cfg(not(any(feature = "metal", feature = "native-cuda")))]
    let _ = cpu;
}
