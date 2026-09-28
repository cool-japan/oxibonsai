//! Unit tests for [`crate::engine_seam`]: the `InferenceEngine` ↔
//! `LoadedModel` seam, driven end to end on the workspace's synthetic `qwen35`
//! (Bonsai 2 hybrid) GGUF — `oxibonsai_testkit::qwen35_fixture`, the same
//! file this crate's external tests load — plus the dense-side seam
//! behaviour (snapshots, rewinds, the backend knob).
//!
//! The hybrid fixture carries every tensor name, dtype and shape
//! relationship of the real 27B file (full-attention `q|gate` layers,
//! Gated-DeltaNet layers, a `prism.hadamard.*` fold), at a width small enough
//! to run in a debug build.

use super::*;

use oxibonsai_kernels::gpu_backend::UNATTRIBUTED_MODEL_EPOCH;
use oxibonsai_kernels::KernelTier;
use oxibonsai_testkit::qwen35_fixture::{
    full_layer_count, synthetic_qwen35_gguf, CONTEXT_LENGTH, EOS_TOKEN_ID, HEAD_DIM, HIDDEN,
    MODEL_NAME, N_KV_HEADS, N_LAYERS, VOCAB,
};

use crate::sampling::SamplingParams;

/// KV window every hybrid engine in this file is built with.
const MAX_SEQ: usize = 64;

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

/// `engine_error_code` is one function reachable at
/// two paths — here and the `crate::engine` re-export the CLI calls — with
/// one signature (both coerce to the same `fn` pointer type), answering
/// identically on both: the engine's code for every refusal kind (the code
/// `RuntimeError::error_code` now reports too), `None` for anything else —
/// including a `Config` error whose text merely spells a code, which the
/// old string-encoded refusal would have matched.
#[test]
fn engine_error_code_answers_identically_on_both_paths() {
    type CodeOf = fn(&RuntimeError) -> Option<&'static str>;
    let via_seam: CodeOf = crate::engine_seam::engine_error_code;
    let via_engine: CodeOf = crate::engine::engine_error_code;
    let refusals = EngineError::one_of_each_kind();
    assert_eq!(refusals.len(), EngineError::ALL_CODES.len());
    for (refusal, code) in refusals.into_iter().zip(EngineError::ALL_CODES) {
        let runtime = RuntimeError::from(refusal);
        assert_eq!(via_seam(&runtime), Some(code), "{runtime}");
        assert_eq!(via_engine(&runtime), Some(code), "{runtime}");
        assert_eq!(runtime.error_code(), code, "{runtime}");
        assert!(
            runtime.to_string().contains(&format!("[{code}]")),
            "{runtime}"
        );
    }
    for other in [
        RuntimeError::Config("[NOT_A_DENSE_MODEL] spelled, not typed".into()),
        RuntimeError::Server("down".into()),
        RuntimeError::CircuitOpen,
    ] {
        assert_eq!(via_seam(&other), None, "{other}");
        assert_eq!(via_engine(&other), None, "{other}");
    }
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
    assert_eq!(engine.model_name(), MODEL_NAME);
    assert_eq!(engine.vocab_size(), VOCAB);
    assert_eq!(engine.hidden_size(), HIDDEN);
    assert_eq!(engine.num_layers(), N_LAYERS);
    assert_eq!(engine.max_seq_len(), MAX_SEQ);
    assert_eq!(engine.max_context(), MAX_SEQ);
    assert_eq!(engine.context_length(), CONTEXT_LENGTH);
    assert_eq!(engine.sequence_position(), 0);
    // KV only for the two full-attention layers.
    assert_eq!(full_layer_count(), 2);
    assert_eq!(
        engine.kv_cache_geometry(),
        (full_layer_count(), N_KV_HEADS, HEAD_DIM)
    );
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
    assert!(engine.is_eos(EOS_TOKEN_ID));
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
/// one CPU replica (an explicit size is honoured) and a model-backed
/// embedder over a dedicated engine that shares the pool's mapping — the
/// hybrid serves `/v1/embeddings` like a dense model does.
#[test]
fn a_hybrid_pool_defaults_to_one_cpu_replica_and_builds_an_embedder() {
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
    let embedder = crate::embed_engine::ModelEmbedder::from_static_gguf(
        built.gguf,
        Arc::clone(&built.shared_token_embd),
        tokenizer,
        greedy_params(),
        42,
        MAX_SEQ,
    )
    .expect("a hybrid model builds an embedder");
    assert_eq!(embedder.dimension(), HIDDEN);
    let vector = embedder
        .embed_tokens(&PROMPT)
        .expect("the hybrid embedder embeds");
    assert_eq!(vector.len(), HIDDEN);
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "unit norm, got {norm}");

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

    let snapshot_ok = engine.snapshot_sequence();
    assert!(snapshot_ok.is_ok(), "a hybrid sequence IS snapshot-able");
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid engine: embeddings
// ─────────────────────────────────────────────────────────────────────────────

/// A hybrid engine embeds: a finite, unit-length, `hidden_size`-wide vector,
/// deterministic across two fresh engines, input-dependent, and — since the
/// pass clears the sequence on both sides — leaving no position behind.
#[test]
fn hybrid_engine_embeds_to_a_deterministic_unit_vector() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut first_engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let mut second_engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 7, MAX_SEQ).expect("hybrid engine");
    assert_eq!(first_engine.embedding_dim(), HIDDEN);
    assert_eq!(first_engine.embedding_max_tokens(), MAX_SEQ);

    let first = first_engine.embed(&PROMPT).expect("the hybrid embeds");
    assert_eq!(first.len(), HIDDEN);
    assert!(first.iter().all(|v| v.is_finite()));
    let norm = first.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "unit norm, got {norm}");
    let second = second_engine.embed(&PROMPT).expect("the hybrid embeds");
    assert_eq!(
        bits(&first),
        bits(&second),
        "two fresh engines over the same file must embed identically"
    );
    let other = first_engine.embed(&PROMPT[..3]).expect("a shorter input");
    assert_ne!(
        bits(&first),
        bits(&other),
        "a different input embeds differently"
    );
    assert_eq!(first_engine.sequence_position(), 0, "no position survives");

    // Exactly the model-level recipe, pooled from the model's own rows.
    let direct = first_engine
        .hybrid_model_mut()
        .expect("hybrid")
        .embed_mean_pooled(&PROMPT)
        .expect("model-level embedding");
    assert_eq!(bits(&direct), bits(&first));
}

/// An embedding in the middle of a conversation leaves the engine usable:
/// the next generation starts a fresh sequence and matches a clean engine,
/// and a snapshot of the abandoned sequence can no longer be restored.
#[test]
fn a_hybrid_embedding_ends_the_current_sequence_cleanly() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let mut clean =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let expected = clean.generate(&PROMPT, 4).expect("clean generate");

    engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
    let snapshot = engine.snapshot_sequence().expect("snapshot");
    engine.embed(&PROMPT[..2]).expect("embed mid-conversation");
    let stale = engine
        .restore_sequence(&snapshot)
        .expect_err("the embedding ended the snapshotted sequence");
    assert_eq!(engine_error_code(&stale), Some("SNAPSHOT_MISMATCH"));
    assert_eq!(engine.generate(&PROMPT, 4).expect("generate"), expected);
}

/// The dense twin of the snapshot rule: an embedding on a dense engine ends
/// its sequence too.
#[test]
fn a_dense_embedding_ends_the_current_sequence() {
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
    let mut dense = InferenceEngine::from_model_with_tier(
        BonsaiModel::new_for_testing_with_blocks(config),
        KernelTier::Reference,
        greedy_params(),
        42,
    );
    dense.prefill_from_pos(&[1, 2, 3], 0).expect("prefill");
    let snapshot = dense.snapshot_sequence().expect("snapshot");
    let vector = dense.embed(&[4, 5, 6]).expect("dense embed");
    assert_eq!(vector.len(), 128);
    let stale = dense
        .restore_sequence(&snapshot)
        .expect_err("the embedding ended the snapshotted sequence");
    assert_eq!(engine_error_code(&stale), Some("SNAPSHOT_MISMATCH"));
    assert_eq!(dense.sequence_position(), 0);
}

/// `ModelEmbedder::from_engine` wraps a hybrid engine, and the embedder
/// serves exactly the engine's own vector.
#[test]
fn model_embedder_wraps_a_hybrid_engine() {
    let bytes = synthetic_qwen35_gguf();
    let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
        GgufFile::parse(Box::leak(bytes.into_boxed_slice())).expect("fixture parses"),
    ));
    let mut reference =
        InferenceEngine::from_gguf(gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let expected = reference.embed(&PROMPT).expect("reference embedding");
    let engine =
        InferenceEngine::from_gguf(gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    let tokenizer = Arc::new(TokenizerBridge::from_native_tokenizer(
        oxibonsai_tokenizer::OxiTokenizer::char_level_stub(VOCAB),
    ));
    let embedder = crate::embed_engine::ModelEmbedder::from_engine(engine, tokenizer)
        .expect("a hybrid engine builds an embedder");
    assert_eq!(embedder.dimension(), HIDDEN);
    assert_eq!(embedder.max_tokens(), MAX_SEQ);
    let served = embedder.embed_tokens(&PROMPT).expect("served");
    assert_eq!(bits(&served), bits(&expected));
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
        eos_token_id: EOS_TOKEN_ID,
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
