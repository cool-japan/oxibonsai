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
//!
//! Engine-level hybrid behaviour runs on **both** executors where this build
//! and host have them ([`hybrid_backends`]): the CPU model always, the Metal
//! hybrid runner when a Metal device exists. Tests that read the CPU model's
//! own internals pin [`Backend::Cpu`].

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

/// Whether this build and host give a hybrid engine the Metal runner. A
/// missing device is the only "no"; any other failure to open the device on
/// a Metal build is a test failure, never a skip.
fn metal_hybrid_available() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

/// The backends every engine-level hybrid test runs on.
fn hybrid_backends() -> Vec<Backend> {
    let mut backends = vec![Backend::Cpu];
    if metal_hybrid_available() {
        backends.push(Backend::Metal);
    }
    backends
}

/// A hybrid engine over `gguf` on `backend`, asserting the executor it got.
fn hybrid_engine<'a>(gguf: &'a GgufFile<'a>, backend: Backend) -> InferenceEngine<'a> {
    let engine =
        InferenceEngine::from_gguf_with_backend(gguf, greedy_params(), 42, MAX_SEQ, backend)
            .unwrap_or_else(|e| panic!("hybrid engine on {backend}: {e}"));
    let expected = match backend {
        Backend::Cpu => HybridBackend::Cpu,
        Backend::Metal => HybridBackend::Metal,
        Backend::Auto if metal_hybrid_available() => HybridBackend::Metal,
        Backend::Auto => HybridBackend::Cpu,
    };
    assert_eq!(engine.hybrid_backend(), Some(expected), "{backend}");
    engine
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
            reason: "no Metal device was found on this host".into(),
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
    let engine = hybrid_engine(&gguf, Backend::Cpu);

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
    // `--backend cpu`: the CPU model decodes; no runner, no GPU tier.
    assert!(!engine.uses_fused_gpu_decode());
    assert!(engine.hybrid_metal_window().is_none());
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    assert_ne!(engine.kernel_tier(), KernelTier::Gpu);
    assert_eq!(engine.model_epoch(), UNATTRIBUTED_MODEL_EPOCH);
    assert_eq!(engine.backend(), Backend::Cpu);
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
    // The tier explanation names the executor, rather than the pinned CPU
    // dispatcher's own "explicitly requested", and how the INT8 selector
    // applies.
    let reason = engine.effective_tier_reason();
    assert!(
        reason.contains("hybrid `qwen35` model")
            && reason.contains("backend=cpu")
            && reason.contains("OXIBONSAI_KERNEL_TIER"),
        "{reason}"
    );
}

/// `--backend auto` decodes a hybrid on the Metal runner when one serves it
/// (a Metal build with a device) and reports it everywhere a tier is
/// reported; without one it runs on the CPU tier and says so.
#[test]
fn backend_auto_picks_the_metal_runner_when_one_serves_the_model() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("hybrid engine");
    assert_eq!(engine.backend(), Backend::Auto);
    let reason = engine.effective_tier_reason();
    assert!(
        reason.contains("hybrid `qwen35` model") && reason.contains("backend=auto"),
        "{reason}"
    );
    // The fused dense route never applies to a hybrid, on either executor.
    assert!(!engine.uses_fused_gpu_decode());
    assert!(!engine.greedy_gpu_eligible(false));
    if !metal_hybrid_available() {
        assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Cpu));
        assert!(engine.hybrid_metal_window().is_none());
        return;
    }
    assert_eq!(engine.hybrid_backend(), Some(HybridBackend::Metal));
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    assert_eq!(engine.kernel_tier(), KernelTier::Gpu);
    assert!(
        engine.kernel_label().ends_with("Metal (hybrid runner)"),
        "{}",
        engine.kernel_label()
    );
    assert!(reason.contains("Metal (hybrid runner)"), "{reason}");
    let window = engine.hybrid_metal_window().expect("a Metal window");
    assert_eq!(window.window, MAX_SEQ);
    assert_eq!(engine.max_seq_len(), MAX_SEQ);
    assert!(!window.clamped());
    assert!(window.device_ceiling >= MAX_SEQ);
    assert!(
        reason.contains(&format!("KV window {MAX_SEQ} positions")),
        "{reason}"
    );
    // The runner reads the image in place exactly when it starts on a page
    // boundary (a large heap buffer often does), and copies it otherwise.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    assert_eq!(
        engine.hybrid_metal_weights_mapped(),
        Some(
            oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::Qwen35MappedRegion::page_aligned(
                gguf.data
            )
            .is_some()
        )
    );
    // Both residents' state is accounted for.
    let cpu = hybrid_engine(&gguf, Backend::Cpu);
    assert!(engine.recurrent_memory_bytes() > cpu.recurrent_memory_bytes());
    assert!(engine.kv_cache_memory_bytes() > cpu.kv_cache_memory_bytes());
    assert!(engine.hybrid_metal_device_allocated_bytes().unwrap_or(0) > 0);
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
        Backend::Cpu,
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
        Backend::Cpu,
    )
    .expect("explicitly sized hybrid pool");
    assert_eq!(two.size, 2, "an explicit size is honoured");
    assert_eq!(two.pool.size(), 2);

    let _ = std::fs::remove_file(&path);
}

/// Two Metal hybrid runners coexist in one process: each owns its own
/// Metal session, KV cache, recurrent state and scratch over the same
/// mapped weights. A pool of two Metal hybrid replicas (sized by
/// `MetalGraph::max_sessions()`, not clamped to one) decodes interleaved
/// requests, each bit-identical to a solo engine's run, and every replica
/// runs on the backend replica #1 resolved to.
#[test]
fn two_metal_hybrid_replicas_interleave_bit_identically_to_solo_runs() {
    if !metal_hybrid_available() {
        return;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path = std::env::temp_dir().join(format!(
        "oxibonsai_engine_seam_metal_pool_{}_{stamp}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, synthetic_qwen35_gguf()).expect("write the fixture");

    // The solo reference: one Metal engine, one prompt at a time.
    let prompts: [&[u32]; 2] = [&PROMPT, &[5, 3, 2, 9, 11, 17, 23]];
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut solo = hybrid_engine(&gguf, Backend::Metal);
    let reference: Vec<Vec<Vec<u32>>> = prompts
        .iter()
        .map(|prompt| {
            let prefill = solo.prefill_from_pos(prompt, 0).expect("solo prefill");
            let mut rows = vec![bits(&prefill)];
            let mut next = argmax(&prefill);
            for step in 0..6 {
                let row = solo
                    .decode_step(next, prompt.len() + step)
                    .expect("solo decode");
                next = argmax(&row);
                rows.push(bits(&row));
            }
            rows
        })
        .collect();

    let built = crate::engine_pool::build_pool_from_gguf_parts(
        &path,
        greedy_params(),
        42,
        MAX_SEQ,
        Some(2),
        Backend::Auto,
    )
    .expect("a two-replica Metal hybrid pool");
    assert!(built.hybrid);
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    assert_eq!(built.tier, KernelTier::Gpu);
    assert_eq!(built.size, 2.min(crate::engine_pool::gpu_max_replicas()));
    if built.size < 2 {
        // `OXIBONSAI_METAL_MAX_SESSIONS=1` caps the pool at one replica;
        // there is nothing to interleave then.
        let _ = std::fs::remove_file(&path);
        return;
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    let (mut a, mut b) = rt.block_on(async {
        let a = built.pool.acquire().await.expect("replica a");
        let b = built.pool.acquire().await.expect("replica b");
        (a, b)
    });
    assert_eq!(a.hybrid_backend(), Some(HybridBackend::Metal));
    assert_eq!(b.hybrid_backend(), Some(HybridBackend::Metal));
    // The runner owns its session; the pool binds none for a hybrid replica.
    assert_eq!(a.gpu_session_id(), None);

    // Interleave: prefill both, then alternate decode steps.
    let mut state = Vec::new();
    for (lease, prompt) in [(&mut a, prompts[0]), (&mut b, prompts[1])] {
        let prefill = lease.prefill_from_pos(prompt, 0).expect("prefill");
        state.push((vec![bits(&prefill)], argmax(&prefill)));
    }
    for step in 0..6 {
        for (i, lease) in [&mut a, &mut b].into_iter().enumerate() {
            let (rows, next) = &mut state[i];
            let row = lease
                .decode_step(*next, prompts[i].len() + step)
                .expect("interleaved decode");
            *next = argmax(&row);
            rows.push(bits(&row));
        }
    }
    for (i, (rows, _)) in state.iter().enumerate() {
        assert_eq!(
            rows, &reference[i],
            "replica {i} diverged from its solo run"
        );
    }
    drop((a, b));
    let _ = std::fs::remove_file(&path);
}

/// `--backend cpu` loads the hybrid on the CPU and never touches Metal.
/// `--backend metal` builds the Metal hybrid runner when a device exists —
/// and says so everywhere a tier is reported — or, with no device (or no
/// Metal build), is the typed `HYBRID_GPU_BACKEND_UNSUPPORTED` refusal
/// naming the constraint. `--backend auto` picks Metal when available.
#[test]
fn backend_cpu_stays_on_the_cpu_and_backend_metal_builds_the_runner_or_names_the_constraint() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let cpu = hybrid_engine(&gguf, Backend::Cpu);
    assert_eq!(cpu.backend(), Backend::Cpu);
    assert!(cpu.is_hybrid());
    assert_eq!(cpu.hybrid_backend(), Some(HybridBackend::Cpu));

    let metal = InferenceEngine::from_gguf_with_backend(
        &gguf,
        greedy_params(),
        42,
        MAX_SEQ,
        Backend::Metal,
    );
    let auto =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("auto engine");
    if metal_hybrid_available() {
        let metal = metal.expect("a Metal device serves the fixture");
        assert_eq!(metal.backend(), Backend::Metal);
        assert_eq!(metal.hybrid_backend(), Some(HybridBackend::Metal));
        assert!(metal.effective_tier_reason().contains("backend=metal"));
        assert_eq!(auto.hybrid_backend(), Some(HybridBackend::Metal));
    } else {
        let err = metal
            .err()
            .expect("without a Metal device an explicit Metal request must be refused");
        assert_eq!(
            engine_error_code(&err),
            Some("HYBRID_GPU_BACKEND_UNSUPPORTED"),
            "{err}"
        );
        let text = err.to_string();
        assert!(
            text.contains("no Metal device") || text.contains("no Metal backend compiled in"),
            "the refusal must name the violated constraint: {text}"
        );
        assert_eq!(auto.hybrid_backend(), Some(HybridBackend::Cpu));
    }
}

/// When no runner serves the model, `--backend metal` is the typed refusal
/// carrying the planner's reason verbatim, `--backend auto` falls back to a
/// CPU engine that decodes exactly as `--backend cpu` does, and `--backend
/// cpu` never consults the planner at all.
#[test]
fn no_runner_is_a_typed_refusal_for_metal_and_a_cpu_fallback_for_auto() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let reason = "qwen35 Metal runner: layer 0 attn_qkv is Q4_0, which has no Metal GEMV here";
    let _forced = crate::engine_hybrid_gpu::ForcedCpuPlan::enter(reason);
    let err = InferenceEngine::from_gguf_with_backend(
        &gguf,
        greedy_params(),
        42,
        MAX_SEQ,
        Backend::Metal,
    )
    .err()
    .expect("no runner: an explicit Metal request is refused");
    assert_eq!(
        engine_error_code(&err),
        Some("HYBRID_GPU_BACKEND_UNSUPPORTED")
    );
    assert!(err.to_string().contains(reason), "{err}");
    let mut auto =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ).expect("auto falls back");
    assert_eq!(auto.hybrid_backend(), Some(HybridBackend::Cpu));
    assert_eq!(auto.backend(), Backend::Auto);
    let mut cpu = hybrid_engine(&gguf, Backend::Cpu);
    assert_eq!(
        auto.generate(&PROMPT, 5).expect("auto generate"),
        cpu.generate(&PROMPT, 5).expect("cpu generate")
    );
}

/// A Metal window request past the declared context is clamped to it (one
/// warning naming every limit), and the CPU model beside the runner is
/// bound at the same window.
#[test]
fn a_metal_window_request_past_the_declared_context_is_clamped_once() {
    if !metal_hybrid_available() {
        return;
    }
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let engine = InferenceEngine::from_gguf_with_backend(
        &gguf,
        greedy_params(),
        42,
        CONTEXT_LENGTH + 1000,
        Backend::Metal,
    )
    .expect("a clamped Metal engine");
    let window = engine.hybrid_metal_window().expect("a Metal window");
    assert_eq!(window.window, CONTEXT_LENGTH);
    assert_eq!(window.requested, CONTEXT_LENGTH + 1000);
    assert!(window.clamped());
    assert!(window
        .limits_applied
        .contains(&crate::engine_hybrid_gpu::HybridWindowLimit::Declared));
    // The CPU model is bound at the same window as the runner.
    assert_eq!(engine.max_seq_len(), CONTEXT_LENGTH);
    assert_eq!(engine.max_context(), CONTEXT_LENGTH);
    assert!(window.describe_limits().contains("declared context 4096"));
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
    let mut engine = hybrid_engine(&gguf, Backend::Cpu);
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

/// The Metal twin: a Metal-backed engine's greedy `generate` is exactly a
/// Metal hybrid runner driven by hand (prefill, then `forward_into` with a
/// first-index argmax), logit row for logit row.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_hybrid_generate_equals_the_runner_driven_by_hand() {
    if !metal_hybrid_available() {
        return;
    }
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine = hybrid_engine(&gguf, Backend::Metal);
    let via_engine = engine.generate(&PROMPT, 6).expect("engine generate");
    assert!(!via_engine.is_empty());

    let config = oxibonsai_model::hybrid::HybridModel::config_from_gguf(&gguf).expect("config");
    let kernel = std::sync::Arc::new(crate::engine_seam::cpu_dispatcher());
    let model =
        oxibonsai_model::hybrid::HybridModel::from_gguf_with(&gguf, config, MAX_SEQ, &kernel)
            .expect("hybrid model");
    let mut runner =
        oxibonsai_model::hybrid::metal::HybridMetalRunner::new_in_place(&model, gguf.data)
            .expect("runner");
    let mut logits = vec![0.0f32; VOCAB];
    runner
        .forward_prefill(&PROMPT, 0, &mut logits)
        .expect("prefill");
    let mut by_hand = Vec::new();
    for pos in PROMPT.len()..PROMPT.len() + via_engine.len() {
        let next = argmax(&logits);
        by_hand.push(next);
        runner.forward_into(next, pos, &mut logits).expect("decode");
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
    for backend in hybrid_backends() {
        every_entry_point_agrees(hybrid_engine(&gguf, backend));
    }
}

fn every_entry_point_agrees(mut engine: InferenceEngine<'_>) {
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
            "the explicit greedy-GPU entry point must decode a hybrid on its own executor, \
             full row"
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
    for backend in hybrid_backends() {
        let mut single = hybrid_engine(&gguf, backend);
        let mut chunked = hybrid_engine(&gguf, backend);
        chunked.set_prefill_chunk_tokens(Some(2));
        let a = single.generate(&PROMPT, 4).expect("single-shot");
        let b = chunked.generate(&PROMPT, 4).expect("chunked");
        assert_eq!(a, b, "{backend}");
        // The model's own chunk (what `--prefill-chunk` sets) is invisible
        // too, on either executor.
        let mut small_model_chunk = hybrid_engine(&gguf, backend);
        small_model_chunk
            .hybrid_model_mut()
            .expect("hybrid")
            .set_prefill_chunk(2)
            .expect("chunk");
        let prefill = small_model_chunk
            .prefill_from_pos(&PROMPT, 0)
            .expect("small-chunk prefill");
        single.reset();
        let reference = single.prefill_from_pos(&PROMPT, 0).expect("prefill");
        assert_eq!(bits(&prefill), bits(&reference), "{backend}");
    }
}

#[test]
fn hybrid_engine_honours_cancellation_before_prefill() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    for backend in hybrid_backends() {
        let mut engine = hybrid_engine(&gguf, backend);
        let token = engine.arm_cancellation();
        token.cancel();
        assert!(engine.generate(&PROMPT, 4).expect("cancelled").is_empty());
    }
}

/// On a Metal-backed engine a reset clears the runner's state: the position
/// returns to zero, a continuation is refused as a gap, and the next
/// sequence decodes exactly as a fresh engine's.
#[test]
fn reset_clears_the_metal_runner_state() {
    if !metal_hybrid_available() {
        return;
    }
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine = hybrid_engine(&gguf, Backend::Metal);
    let mut fresh = hybrid_engine(&gguf, Backend::Metal);
    let reference = fresh.prefill_from_pos(&PROMPT, 0).expect("fresh prefill");
    engine.prefill_from_pos(&[9, 8, 7], 0).expect("prefill");
    assert_eq!(engine.sequence_position(), 3);
    engine.reset();
    assert_eq!(engine.sequence_position(), 0);
    let gap = engine
        .decode_step(1, 3)
        .expect_err("the reset sequence cannot continue at 3");
    assert_eq!(engine_error_code(&gap), Some("NON_CONTIGUOUS_POSITION"));
    let replay = engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
    assert_eq!(bits(&replay), bits(&reference));
    // `reset_recurrent` alone clears it too.
    engine.reset_recurrent();
    assert_eq!(engine.sequence_position(), 0);
}

#[test]
fn reset_clears_both_the_kv_cursor_and_the_recurrent_state() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut engine = hybrid_engine(&gguf, Backend::Cpu);
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
    for backend in hybrid_backends() {
        let mut engine = hybrid_engine(&gguf, backend);
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
            "{backend}: the implicit restart must clear the attached recurrent state"
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid engine: typed refusals
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn hybrid_engine_refuses_rollback_and_position_gaps_with_typed_errors() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    for backend in hybrid_backends() {
        rollback_and_gaps_are_typed_refusals(hybrid_engine(&gguf, backend));
    }
}

fn rollback_and_gaps_are_typed_refusals(mut engine: InferenceEngine<'_>) {
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
    for backend in hybrid_backends() {
        let mut engine = hybrid_engine(&gguf, backend);
        let verify = engine
            .verify_batch(&PROMPT, 0)
            .expect_err("verification needs a rollback");
        assert_eq!(
            engine_error_code(&verify),
            Some("RECURRENT_ROLLBACK_REQUIRED")
        );

        let snapshot = engine
            .snapshot_sequence()
            .expect("a hybrid sequence IS snapshot-able");
        assert!(snapshot.has_recurrent_state());
        assert_eq!(
            snapshot.hybrid_backend(),
            engine.hybrid_backend(),
            "{backend}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybrid engine: embeddings
// ─────────────────────────────────────────────────────────────────────────────

/// A hybrid engine embeds: a finite, unit-length, `hidden_size`-wide vector,
/// deterministic across two fresh engines, input-dependent, and — since the
/// pass clears the sequence on both sides — leaving no position behind. The
/// embedding pass runs on the CPU model on either executor, so a
/// Metal-backed engine embeds bit-identically to a CPU one.
#[test]
fn hybrid_engine_embeds_to_a_deterministic_unit_vector() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let vectors: Vec<Vec<u32>> = hybrid_backends()
        .into_iter()
        .map(|backend| embeds_deterministically(&gguf, backend))
        .collect();
    for pair in vectors.windows(2) {
        assert_eq!(pair[0], pair[1], "CPU and Metal engines embed identically");
    }
}

fn embeds_deterministically(gguf: &GgufFile<'_>, backend: Backend) -> Vec<u32> {
    let mut first_engine = hybrid_engine(gguf, backend);
    let mut second_engine =
        InferenceEngine::from_gguf_with_backend(gguf, greedy_params(), 7, MAX_SEQ, backend)
            .expect("hybrid engine");
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
    bits(&first)
}

/// An embedding in the middle of a conversation leaves the engine usable:
/// the next generation starts a fresh sequence and matches a clean engine,
/// and a snapshot of the abandoned sequence can no longer be restored — on
/// either executor.
#[test]
fn a_hybrid_embedding_ends_the_current_sequence_cleanly() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    for backend in hybrid_backends() {
        let mut engine = hybrid_engine(&gguf, backend);
        let mut clean = hybrid_engine(&gguf, backend);
        let expected = clean.generate(&PROMPT, 4).expect("clean generate");

        engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
        let snapshot = engine.snapshot_sequence().expect("snapshot");
        engine.embed(&PROMPT[..2]).expect("embed mid-conversation");
        assert_eq!(engine.sequence_position(), 0, "{backend}");
        let stale = engine
            .restore_sequence(&snapshot)
            .expect_err("the embedding ended the snapshotted sequence");
        assert_eq!(engine_error_code(&stale), Some("SNAPSHOT_MISMATCH"));
        assert_eq!(
            engine.generate(&PROMPT, 4).expect("generate"),
            expected,
            "{backend}"
        );
    }
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
    for backend in hybrid_backends() {
        let mut engine = hybrid_engine(&gguf, backend);
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
            bits(&reference),
            bits(&replayed),
            "{backend}: a restored snapshot must reproduce the next step bit for bit"
        );

        // Any reset invalidates the snapshot.
        engine.reset();
        let stale = engine
            .restore_sequence(&snapshot)
            .expect_err("a snapshot from a reset sequence must be refused");
        assert_eq!(engine_error_code(&stale), Some("SNAPSHOT_MISMATCH"));
    }
}

/// Snapshot at position `p`, decode eight more tokens, restore, decode the
/// same eight: every logit row is bit-identical — on the Metal runner (its
/// device recurrent state is copied out and back; the device KV below `p`
/// is untouched by the replay) exactly as on the CPU.
#[test]
fn a_restored_snapshot_replays_eight_decoded_tokens_bit_for_bit() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    for backend in hybrid_backends() {
        let mut engine = hybrid_engine(&gguf, backend);
        let prefill = engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
        let snapshot = engine.snapshot_sequence().expect("snapshot");
        assert_eq!(snapshot.hybrid_backend(), engine.hybrid_backend());
        let decode_eight = |engine: &mut InferenceEngine<'_>| -> Vec<Vec<u32>> {
            let mut next = argmax(&prefill);
            (0..8)
                .map(|step| {
                    let row = engine
                        .decode_step(next, PROMPT.len() + step)
                        .expect("decode");
                    next = argmax(&row);
                    bits(&row)
                })
                .collect()
        };
        let reference = decode_eight(&mut engine);
        assert_eq!(engine.sequence_position(), PROMPT.len() + 8);
        engine.restore_sequence(&snapshot).expect("restore");
        assert_eq!(engine.sequence_position(), PROMPT.len());
        let replayed = decode_eight(&mut engine);
        assert_eq!(reference, replayed, "{backend}");
    }
}

/// Both executors refuse the same snapshot misuse with the same error
/// codes: a snapshot past the current position, a snapshot after a reset,
/// after an implicit restart and after an embedding pass, a snapshot of the
/// other hybrid executor, a snapshot with an opaque attached state, a
/// rewind, a position gap and a speculative verification.
#[test]
fn snapshot_misuse_answers_the_same_codes_on_both_executors() {
    let bytes = synthetic_qwen35_gguf();
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let backends = hybrid_backends();
    // The engine refusal's code, or for a model error its variant name.
    let code = |r: RuntimeResult<()>| -> Option<String> {
        r.err().map(|e| match &e {
            RuntimeError::Model(m) => {
                let debug = format!("{m:?}");
                let variant = debug
                    .split([' ', '(', '{'])
                    .next()
                    .unwrap_or_default()
                    .to_string();
                format!("MODEL_ERROR:{variant}")
            }
            other => other.error_code().to_string(),
        })
    };
    let codes: Vec<Vec<Option<String>>> = backends
        .iter()
        .map(|&backend| {
            let mut engine = hybrid_engine(&gguf, backend);
            let mut out = Vec::new();
            engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
            let early = engine.snapshot_sequence().expect("snapshot");
            engine.decode_step(1, PROMPT.len()).expect("decode");
            let late = engine.snapshot_sequence().expect("snapshot");
            engine.restore_sequence(&early).expect("restore");
            // Past the current position.
            out.push(code(engine.restore_sequence(&late)));
            // A rewind and a gap.
            out.push(code(engine.try_rewind_cache(1)));
            out.push(code(engine.decode_step(1, PROMPT.len() + 5).map(|_| ())));
            // Speculative verification.
            out.push(code(engine.verify_batch(&PROMPT, 0).map(|_| ())));
            // After an implicit restart at position 0.
            let before_restart = engine.snapshot_sequence().expect("snapshot");
            engine.prefill_from_pos(&PROMPT, 0).expect("restart");
            out.push(code(engine.restore_sequence(&before_restart)));
            // After an explicit reset.
            let before_reset = engine.snapshot_sequence().expect("snapshot");
            engine.reset();
            out.push(code(engine.restore_sequence(&before_reset)));
            // After an embedding pass.
            engine.prefill_from_pos(&PROMPT, 0).expect("prefill");
            let before_embed = engine.snapshot_sequence().expect("snapshot");
            engine.embed(&PROMPT).expect("embed");
            out.push(code(engine.restore_sequence(&before_embed)));
            // An opaque attached state has no snapshot hook.
            struct Opaque;
            impl crate::engine_control::RecurrentState for Opaque {
                fn reset_recurrent(&mut self) {}
            }
            engine.set_recurrent_state(Box::new(Opaque));
            out.push(code(engine.snapshot_sequence().map(|_| ())));
            out
        })
        .collect();
    let expected: Vec<Option<String>> = [
        "SNAPSHOT_MISMATCH",
        "MODEL_ERROR:RecurrentRollbackUnsupported",
        "NON_CONTIGUOUS_POSITION",
        "RECURRENT_ROLLBACK_REQUIRED",
        "SNAPSHOT_MISMATCH",
        "SNAPSHOT_MISMATCH",
        "SNAPSHOT_MISMATCH",
        "RECURRENT_STATE_NOT_SNAPSHOTABLE",
    ]
    .iter()
    .map(|c| Some((*c).to_string()))
    .collect();
    for (backend, got) in backends.iter().zip(&codes) {
        assert_eq!(got, &expected, "{backend}");
    }

    // A snapshot of the other executor is a mismatch on both sides, even on
    // the same sequence number.
    if backends.contains(&Backend::Metal) {
        let mut cpu = hybrid_engine(&gguf, Backend::Cpu);
        let mut metal = hybrid_engine(&gguf, Backend::Metal);
        let cpu_snapshot = cpu.snapshot_sequence().expect("snapshot");
        let metal_snapshot = metal.snapshot_sequence().expect("snapshot");
        for (engine, snapshot) in [(&mut cpu, &metal_snapshot), (&mut metal, &cpu_snapshot)] {
            let err = engine
                .restore_sequence(snapshot)
                .expect_err("the other executor's snapshot");
            assert_eq!(engine_error_code(&err), Some("SNAPSHOT_MISMATCH"));
            assert!(err.to_string().contains("the Metal runner"), "{err}");
        }
    }
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
