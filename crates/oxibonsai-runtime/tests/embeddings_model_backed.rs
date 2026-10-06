//! Integration gate for the **real, model-backed** `/v1/embeddings` path
//! (findings `RT-08` / `SV-02`).
//!
//! The endpoint once had only lexical backends: a TF-IDF bag of words whose
//! vector space mutated with every request, and a byte-hash
//! `IdentityEmbedder`. It now serves a genuine embedding computed from the
//! loaded model's mean-pooled final hidden state. This file is the
//! outside-the-crate gate on that: the engine seam, the [`ModelEmbedder`], the
//! HTTP surface it serves, a real-model semantic smoke test, and the
//! benchmark of the batched path against the per-token loop.
//!
//! # Why most tests here use a synthetic *weighted* GGUF, not `tiny_test()`
//!
//! `Qwen3Config::tiny_test()` through `InferenceEngine::new` builds a
//! **weight-less** model: `BonsaiModel::new` leaves `blocks` empty and
//! synthesizes an all-zero `token_embd`, so every hidden state is exactly zero
//! and the pooled vector is the zero vector — which has norm `0`, not `1`. A
//! "unit norm" assertion against that config would be asserting the opposite
//! of what the code does. So the shape/contract assertions that only need a
//! config run on `tiny_test()` (and assert the *honest* zero-vector
//! behaviour), while every semantic assertion — unit norm, two texts differ,
//! `dimensions` truncation stays normalised — runs on a small synthetic
//! **fully-ternary** GGUF with real, deterministic weights, built by
//! [`fixture::build_synthetic_ternary_gguf`].
//!
//! # The real-model smoke test
//!
//! [`real_model_places_queen_nearer_to_king_than_banana`] needs
//! `models/Ternary-Bonsai-1.7B.gguf` + `models/tokenizer.json`. When they are
//! absent it self-skips **through the capability report**
//! (`Capability::LegacyModels`, `record_skipped`) rather than returning green,
//! so a run that never exercised a real model is distinguishable in the
//! manifest from one that did.
//!
//! # `tiny_test()` is asserted for shape, never for unit norm
//!
//! `weightless_tiny_test_engine_embeds_to_the_right_shape` asserts SHAPE only
//! — never unit norm — because `Qwen3Config::tiny_test()` through
//! `BonsaiModel::new` leaves `blocks` empty and synthesizes an all-zero
//! `token_embd`, so every hidden state (and therefore the pooled vector) is
//! exactly zero, which has norm `0`, not `1`. The unit-norm assertion runs on
//! the WEIGHTED synthetic ternary GGUF instead
//! (`engine_embed_returns_hidden_size_floats_with_unit_norm`,
//! `model_backed_router_answers_200_with_unit_norm_vectors`), never on
//! `tiny_test()`: asserting unit norm against a model that produces the zero
//! vector would be asserting something false.
//!
//! # The batched engine lock and the truncation split
//!
//! Section 2b proves `EmbedderRegistry::embed_texts` /
//! `embed_token_batches` — the methods the HTTP handler actually calls —
//! reach [`ModelEmbedder`]'s batched engine lock (one acquisition per
//! request, not per item). Section 3b proves the same end to end over HTTP,
//! plus the `context_length_exceeded` / truncation split: TEXT that
//! tokenizes past the backend's ceiling is refused rather than silently
//! truncated and billed in full; raw token ids keep truncating, billed for
//! exactly what gets embedded.
//!
//! # The benchmark
//!
//! `embed_bench_short_and_long` (section 6) times the production
//! `InferenceEngine::embed` (the head-free Metal prefill on a GPU engine)
//! against the batched CPU pass it superseded and the per-token loop both
//! replaced, on a real model, and asserts parity, the short-input speed-up
//! and the long-input wall-time target. It runs only when asked: both
//! `OXIBONSAI_EMBED_BENCH=1` and `OXI_MODEL=<gguf>` must be set (release
//! build). Neither variable starts it alone — `OXI_MODEL` is exported for the
//! other real-model gates, and the benchmark maps a multi-GB model and runs
//! for minutes — and `scripts/release-gate.sh`'s embedding leg sets both
//! explicitly, so that leg never self-skips.
//!
//! The per-token reference loop costs about five to six minutes at 2000 tokens
//! on an M3 at a load average of about 10 (309-353 s measured, longer under a
//! heavier load), so on a long input it is a manual measurement
//! behind a second opt-in, `OXIBONSAI_EMBED_BENCH_PER_TOKEN=1`, which the
//! release gate never sets (it removes an inherited one). Without it the
//! 2000-token input compares the production embedding with the batched CPU
//! pass (pooled cosine of at least 0.9999) and asserts the 20 s Metal target;
//! the 10- and 200-token inputs always compare every leg with the per-token
//! reference, and the 10-token speed-up assertion stands.

// `embeddings` is only compiled with the `server` feature; gate the whole file
// the same way so `--no-default-features` stays green.
#![cfg(feature = "server")]

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_rag::embedding::Embedder;
use oxibonsai_rag::error::RagError;
use oxibonsai_runtime::embed_engine::ModelEmbedder;
use oxibonsai_runtime::embeddings::{
    create_embeddings_router_from_state, EmbedderRegistry, EmbeddingAppState,
};
use oxibonsai_runtime::error::RuntimeError;
use oxibonsai_runtime::metrics::InferenceMetrics;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::workspace::{find_model, models_dir};
use oxibonsai_tokenizer::OxiTokenizer;

/// A small, fully-ternary synthetic GGUF with real (deterministic) weights.
///
/// Adapted from the fixture `crates/oxibonsai-runtime/tests/
/// generate_pipeline_tests.rs` and `cross_backend_determinism_tests.rs`
/// already use — same shapes, same packing rules — with the vocabulary made a
/// parameter so it can be matched to the char-level tokenizer's id range.
mod fixture {
    use half::f16;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
    use oxibonsai_testkit::gguf_fixture::Lcg;

    /// Hidden size. Must be ≥ 128 (one `TQ2_0_g128` block) and a multiple of it.
    const HIDDEN: usize = 128;
    const INTER: usize = 256;
    const LAYERS: usize = 2;
    const N_Q: usize = 4;
    const N_KV: usize = 2;
    const HEAD_DIM: usize = 32;

    /// `TQ2_0_g128` blob: 32 bytes of 2-bit codes (`00→-1, 01→0, 10→+1`;
    /// `11` is reserved and never emitted — `screen_ternary_codes` rejects it)
    /// followed by a two-byte FP16 scale, per 128 weights.
    fn tq2_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
        assert_eq!(
            num_weights % 128,
            0,
            "num_weights must be a multiple of 128"
        );
        let num_blocks = num_weights / 128;
        let mut data = Vec::with_capacity(num_blocks * 34);
        let mut lcg = Lcg::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        for _ in 0..num_blocks {
            for _ in 0..32 {
                data.push(lcg.next_valid_tq2_byte());
            }
            let scale =
                0.25_f32 + ((lcg.next_u64() >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
            data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        data
    }

    /// An FP32 tensor whose values vary with the index, so no two embedding
    /// rows are identical (a constant table would make every text embed to the
    /// same vector and quietly vacate half this file's assertions).
    fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            let phase = (i as f32) * 0.013_f32;
            let value = scale * (1.0_f32 + 0.25_f32 * phase.sin());
            v.extend_from_slice(&value.to_le_bytes());
        }
        v
    }

    /// Build the fixture. `vocab` must be a multiple of 128 for the ternary LM
    /// head, and no token id fed to the model may reach it.
    pub fn build_synthetic_ternary_gguf(vocab: usize) -> Vec<u8> {
        assert_eq!(
            (vocab * HIDDEN) % 128,
            0,
            "the ternary LM head needs vocab * hidden to be a multiple of 128"
        );
        let mut writer = GgufWriter::new();

        writer.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen3".to_string()),
        );
        writer.add_metadata(
            "general.name",
            MetadataWriteValue::Str("EmbedModelSyntheticFixture".to_string()),
        );
        writer.add_metadata(
            "qwen3.embedding_length",
            MetadataWriteValue::U32(HIDDEN as u32),
        );
        writer.add_metadata("qwen3.block_count", MetadataWriteValue::U32(LAYERS as u32));
        writer.add_metadata(
            "qwen3.attention.head_count",
            MetadataWriteValue::U32(N_Q as u32),
        );
        writer.add_metadata(
            "qwen3.attention.head_count_kv",
            MetadataWriteValue::U32(N_KV as u32),
        );
        writer.add_metadata(
            "qwen3.feed_forward_length",
            MetadataWriteValue::U32(INTER as u32),
        );
        writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
        writer.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
        writer.add_metadata(
            "qwen3.attention.layer_norm_rms_epsilon",
            MetadataWriteValue::F32(1e-6),
        );
        writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

        writer.add_tensor(TensorEntry {
            name: "token_embd.weight".to_string(),
            shape: vec![HIDDEN as u64, vocab as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(vocab * HIDDEN, 0.5),
        });
        writer.add_tensor(TensorEntry {
            name: "output_norm.weight".to_string(),
            shape: vec![HIDDEN as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(HIDDEN, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: "output.weight".to_string(),
            shape: vec![HIDDEN as u64, vocab as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(vocab * HIDDEN, 0xCAFE_BABE),
        });

        for layer in 0..LAYERS {
            let pfx = format!("blk.{layer}");
            for (name, dim) in [
                ("attn_norm.weight", HIDDEN),
                ("ffn_norm.weight", HIDDEN),
                ("attn_q_norm.weight", HEAD_DIM),
                ("attn_k_norm.weight", HEAD_DIM),
            ] {
                writer.add_tensor(TensorEntry {
                    name: format!("{pfx}.{name}"),
                    shape: vec![dim as u64],
                    tensor_type: TensorType::F32,
                    data: f32_pattern(dim, 1.0),
                });
            }

            let seed = 0x2000_0000_u64.wrapping_add((layer as u64) << 16);
            for (name, in_dim, out_dim, bump) in [
                ("attn_q.weight", HIDDEN, N_Q * HEAD_DIM, 0u64),
                ("attn_k.weight", HIDDEN, N_KV * HEAD_DIM, 1),
                ("attn_v.weight", HIDDEN, N_KV * HEAD_DIM, 2),
                ("attn_output.weight", N_Q * HEAD_DIM, HIDDEN, 3),
                ("ffn_gate.weight", HIDDEN, INTER, 4),
                ("ffn_up.weight", HIDDEN, INTER, 5),
                ("ffn_down.weight", INTER, HIDDEN, 6),
            ] {
                writer.add_tensor(TensorEntry {
                    name: format!("{pfx}.{name}"),
                    shape: vec![in_dim as u64, out_dim as u64],
                    tensor_type: TensorType::TQ2_0_g128,
                    data: tq2_0_g128_pattern(in_dim * out_dim, seed.wrapping_add(bump)),
                });
            }
        }

        writer.to_bytes().expect("GgufWriter::to_bytes")
    }
}

/// KV-cache / context budget for every engine built here.
const MAX_SEQ: usize = 256;

/// Vocabulary of the synthetic fixture. Matches the char-level tokenizer's
/// vocabulary below so no encoded id can land past `token_embd`'s last row.
const FIXTURE_VOCAB: usize = 256;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 8,
    }
}

/// A `'static` engine over `gguf_bytes`.
///
/// `ModelEmbedder` holds an `InferenceEngine<'static>`, and a GGUF-backed
/// model borrows its weights from the file bytes, so both the bytes and the
/// parsed header are leaked. Acceptable in a test process, and the same
/// technique `BonsaiModel::new_for_testing_with_blocks` already uses for its
/// own fixture weights.
fn static_engine(
    gguf_bytes: Vec<u8>,
    tier: KernelTier,
) -> oxibonsai_runtime::engine::InferenceEngine<'static> {
    let bytes: &'static [u8] = Box::leak(gguf_bytes.into_boxed_slice());
    let gguf: &'static GgufFile<'static> = Box::leak(Box::new(
        GgufFile::parse(bytes).expect("parse GGUF fixture"),
    ));
    let model = BonsaiModel::from_gguf(gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    oxibonsai_runtime::engine::InferenceEngine::from_model_with_tier(
        model,
        tier,
        greedy_params(),
        42,
    )
}

/// Like [`static_engine`], but with the kernel tier auto-detected (the fastest
/// correct CPU path on this host) instead of pinned.
fn static_engine_auto_tier(
    gguf_bytes: Vec<u8>,
) -> oxibonsai_runtime::engine::InferenceEngine<'static> {
    let bytes: &'static [u8] = Box::leak(gguf_bytes.into_boxed_slice());
    let gguf: &'static GgufFile<'static> =
        Box::leak(Box::new(GgufFile::parse(bytes).expect("parse GGUF")));
    let model = BonsaiModel::from_gguf(gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    oxibonsai_runtime::engine::InferenceEngine::from_model(model, greedy_params(), 42)
}

/// An engine over the synthetic fully-ternary fixture (real weights).
fn weighted_engine() -> oxibonsai_runtime::engine::InferenceEngine<'static> {
    static_engine(
        fixture::build_synthetic_ternary_gguf(FIXTURE_VOCAB),
        KernelTier::Reference,
    )
}

/// A weight-less engine from the shared `tiny_test()` config.
fn tiny_test_engine() -> oxibonsai_runtime::engine::InferenceEngine<'static> {
    oxibonsai_runtime::engine::InferenceEngine::from_model_with_tier(
        BonsaiModel::new(Qwen3Config::tiny_test()),
        KernelTier::Reference,
        greedy_params(),
        42,
    )
}

/// A char-level tokenizer whose ids are all `< FIXTURE_VOCAB`.
fn fixture_tokenizer() -> Arc<TokenizerBridge> {
    Arc::new(TokenizerBridge::from_native_tokenizer(
        OxiTokenizer::char_level_stub(FIXTURE_VOCAB),
    ))
}

fn weighted_embedder() -> Arc<ModelEmbedder> {
    Arc::new(ModelEmbedder::new(
        Arc::new(Mutex::new(weighted_engine())),
        fixture_tokenizer(),
    ))
}

/// The model-backed router the server mounts: `require_model_backend`, a real
/// [`ModelEmbedder`], and a shared metrics registry.
fn model_backed_router(metrics: &Arc<InferenceMetrics>) -> axum::Router {
    create_embeddings_router_from_state(
        EmbeddingAppState::from_registry(
            EmbedderRegistry::new(32)
                .with_require_model_backend(true)
                .with_model(weighted_embedder()),
        )
        .with_metrics(Arc::clone(metrics)),
    )
}

/// The same router with no model installed — the honest-`501` deployment.
fn refusing_router(metrics: &Arc<InferenceMetrics>) -> axum::Router {
    create_embeddings_router_from_state(
        EmbeddingAppState::from_registry(
            EmbedderRegistry::new(32).with_require_model_backend(true),
        )
        .with_metrics(Arc::clone(metrics)),
    )
}

async fn post_embeddings(
    app: axum::Router,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::post("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&body).expect("body serialisation"),
        ))
        .expect("request build");
    let resp = app.oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

fn float_vector(item: &serde_json::Value) -> Vec<f32> {
    item["embedding"]
        .as_array()
        .expect("embedding must be a JSON array under encoding_format=float")
        .iter()
        .map(|v| v.as_f64().expect("embedding component must be a number") as f32)
        .collect()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine needs equal-length vectors");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    assert!(na > 0.0 && nb > 0.0, "cosine of a zero vector is undefined");
    dot / (na * nb)
}

// ─── 1. Engine seam: InferenceEngine::embed ──────────────────────────────────

#[test]
fn engine_embed_returns_hidden_size_floats_with_unit_norm() {
    let mut engine = weighted_engine();
    let dim = engine.embedding_dim();
    let vector = engine.embed(&[3, 14, 15, 9]).expect("embed should succeed");
    assert_eq!(
        vector.len(),
        dim,
        "an embedding must have exactly hidden_size components"
    );
    let norm = l2_norm(&vector);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "a model-backed embedding must be L2-normalised; got norm {norm}"
    );
    assert!(
        vector.iter().all(|x| x.is_finite()),
        "no component may be NaN or infinite"
    );
}

#[test]
fn engine_embed_rejects_empty_input_with_a_typed_error() {
    let mut engine = weighted_engine();
    let err = engine
        .embed(&[])
        .expect_err("an empty token slice has no hidden state to pool");
    assert!(
        matches!(
            err,
            RuntimeError::Model(oxibonsai_model::error::ModelError::ShapeInvariant { .. })
        ),
        "empty input must surface as a typed model error, not a panic or a zero vector; got {err:?}"
    );
}

#[test]
fn engine_embed_is_deterministic_across_calls_and_engines() {
    let mut first = weighted_engine();
    let mut second = weighted_engine();
    let a = first.embed(&[1, 2, 3]).expect("first embed");
    let b = first.embed(&[1, 2, 3]).expect("second embed, same engine");
    let c = second.embed(&[1, 2, 3]).expect("embed on a fresh engine");
    assert_eq!(a, b, "the same engine must embed the same ids identically");
    assert_eq!(
        a, c,
        "two engines over the same weights must agree bit-for-bit — an embedding may not \
         depend on what the process embedded before"
    );
}

#[test]
fn engine_embed_does_not_disturb_the_kv_cache() {
    let mut engine = weighted_engine();
    let _ = engine.embed(&[5, 6, 7]).expect("embed");
    assert_eq!(
        engine
            .dense_model()
            .expect("the weighted fixture is a dense model")
            .kv_cache()
            .seq_len(),
        0,
        "embedding must leave no per-sequence state behind for the next request"
    );
}

#[test]
fn engine_embed_refuses_a_prompt_past_the_context() {
    let mut engine = weighted_engine();
    let too_long: Vec<u32> = (0..(MAX_SEQ as u32 + 1)).map(|i| i % 32).collect();
    let err = engine
        .embed(&too_long)
        .expect_err("a prompt past the context must be refused, not truncated silently");
    assert!(
        matches!(
            err,
            RuntimeError::Model(oxibonsai_model::error::ModelError::SequenceTooLong { .. })
        ),
        "got {err:?}"
    );
}

#[test]
fn weightless_tiny_test_engine_embeds_to_the_right_shape() {
    // The spec's `tiny_test()` case. A weight-less model pools to the zero
    // vector, which is reported honestly rather than divided into NaNs — see
    // this file's module docs.
    let mut engine = tiny_test_engine();
    let dim = engine.embedding_dim();
    assert_eq!(dim, Qwen3Config::tiny_test().hidden_size);
    let vector = engine.embed(&[1, 2]).expect("embed should still run");
    assert_eq!(vector.len(), dim);
    assert!(
        vector.iter().all(|x| x.is_finite()),
        "an all-zero hidden state must not normalise into NaNs"
    );
}

// ─── 2. ModelEmbedder / the `Embedder` trait ─────────────────────────────────

#[test]
fn two_different_texts_embed_to_different_vectors() {
    let embedder = weighted_embedder();
    let a = embedder.embed("alpha").expect("embed alpha");
    let b = embedder.embed("omega").expect("embed omega");
    assert_eq!(a.len(), b.len());
    assert_ne!(
        a, b,
        "two different texts must not collapse to the same vector"
    );
    let similarity = cosine(&a, &b);
    assert!(
        similarity < 0.999_9,
        "two different texts must be distinguishable by cosine similarity; got {similarity}"
    );
}

#[test]
fn identical_texts_embed_identically() {
    let embedder = weighted_embedder();
    let a = embedder.embed("the same text").expect("first");
    let b = embedder.embed("the same text").expect("second");
    assert_eq!(
        a, b,
        "the same text must embed bit-identically — this is exactly what the TF-IDF backend \
         SV-02 condemned could not promise"
    );
}

#[test]
fn model_embedder_reports_the_models_hidden_size() {
    let embedder = weighted_embedder();
    let engine_dim = weighted_engine().embedding_dim();
    assert_eq!(embedder.embedding_dim(), engine_dim);
    assert_eq!(embedder.dimension(), engine_dim);
}

#[test]
fn model_embedder_refuses_text_that_tokenizes_to_nothing() {
    let embedder = weighted_embedder();
    let err = embedder
        .embed("")
        .expect_err("the empty string has no tokens");
    assert!(matches!(err, RagError::EmptyDocument), "got {err:?}");
}

#[test]
fn embed_batch_agrees_with_per_item_embed() {
    let embedder = weighted_embedder();
    let texts = vec!["one".to_string(), "two".to_string(), "three".to_string()];
    let batched = embedder.embed_owned_batch(&texts);
    assert_eq!(batched.len(), 3);
    for (i, text) in texts.iter().enumerate() {
        let single = embedder.embed(text).expect("single embed");
        let from_batch = batched[i].as_ref().expect("batched embed");
        assert_eq!(
            &single, from_batch,
            "batching must not change the result for item {i}"
        );
    }
}

#[test]
fn embed_tokens_bypasses_the_tokenizer() {
    let embedder = weighted_embedder();
    let ids = embedder.tokenize("abc").expect("tokenize");
    let from_text = embedder.embed("abc").expect("embed text");
    let from_ids = embedder.embed_tokens(&ids).expect("embed ids");
    assert_eq!(
        from_text, from_ids,
        "embedding the ids of a text must equal embedding the text"
    );
}

// ─── 2b. The registry reaches the batched lock ───────────────────────────────
//
// `ModelEmbedder`'s batch paths (section 2 above) take its lock once for a
// whole batch, but that is only worth anything if
// `EmbedderRegistry::embed_texts` — the ONE method the HTTP handler actually
// calls — reaches them: otherwise an N-input request takes the engine mutex
// N times regardless. These tests exercise the registry entry point, not the
// embedder directly, to prove that wiring holds.

#[test]
fn embed_texts_through_the_registry_takes_the_engine_lock_once_for_the_whole_batch() {
    let embedder = weighted_embedder();
    let registry = EmbedderRegistry::new(32).with_model(Arc::clone(&embedder));
    let before = embedder.lock_acquisitions();
    let texts = vec![
        "one".to_string(),
        "two".to_string(),
        "three".to_string(),
        "four".to_string(),
    ];
    let batched = registry.embed_texts(&texts);
    assert_eq!(
        embedder.lock_acquisitions() - before,
        1,
        "EmbedderRegistry::embed_texts must take the engine lock once for the whole batch, \
         not once per item"
    );
    for (i, text) in texts.iter().enumerate() {
        let single = embedder.embed(text).expect("single embed");
        assert_eq!(
            batched[i], single,
            "the batched result must equal the per-item path for item {i}"
        );
    }
}

#[test]
fn token_id_batches_through_the_registry_take_the_engine_lock_once_for_the_whole_batch() {
    let embedder = weighted_embedder();
    let registry = EmbedderRegistry::new(32).with_model(Arc::clone(&embedder));
    let before = embedder.lock_acquisitions();
    let batches = vec![vec![1u32, 2, 3], vec![4u32, 5], vec![6u32]];
    let batched = registry
        .embed_token_batches(&batches)
        .expect("a model backend installs a token embedder");
    assert_eq!(
        embedder.lock_acquisitions() - before,
        1,
        "a 3-batch `\"input\": [[..], [..], [..]]` request must take the engine lock once, \
         not once per sequence"
    );
    for (i, ids) in batches.iter().enumerate() {
        let single = embedder.embed_tokens(ids).expect("single embed_tokens");
        assert_eq!(
            batched[i], single,
            "the batched result must equal the per-item path for batch {i}"
        );
    }
}

#[test]
fn embedding_dim_agrees_with_the_vectors_the_model_path_actually_returns() {
    let embedder = weighted_embedder();
    // Deliberately mismatched `default_dim` (999): if `embedding_dim()` ever
    // fell back to it instead of asking the installed model backend, this
    // assertion would catch it immediately.
    let registry = EmbedderRegistry::new(999).with_model(Arc::clone(&embedder));
    let vectors = registry.embed_texts(&["hello".to_string(), "world".to_string()]);
    for (i, v) in vectors.iter().enumerate() {
        assert_eq!(
            v.len(),
            registry.embedding_dim(),
            "item {i}: EmbedderRegistry::embedding_dim() must agree with the vectors the \
             installed model backend actually returns"
        );
    }
}

// ─── 3. HTTP surface ─────────────────────────────────────────────────────────

#[tokio::test]
async fn model_backed_router_answers_200_with_unit_norm_vectors() {
    let metrics = Arc::new(InferenceMetrics::new());
    let app = model_backed_router(&metrics);
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": "hello world" })).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a router carrying a real model embedder must answer 200, not the no-backend 501: \
         {json}"
    );
    assert_eq!(json["object"].as_str(), Some("list"));
    assert_eq!(
        json["model"].as_str(),
        Some("bonsai-embeddings-model"),
        "the response must name the backend that actually answered"
    );
    assert_eq!(json["normalized"].as_bool(), Some(true));

    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    let vector = float_vector(&data[0]);
    let expected_dim = json["dimension"].as_u64().expect("dimension") as usize;
    assert_eq!(vector.len(), expected_dim);
    assert_eq!(
        expected_dim,
        weighted_engine().embedding_dim(),
        "the reported dimension must be the model's hidden size"
    );
    let norm = l2_norm(&vector);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "an untruncated embedding must be unit-length; got {norm}"
    );
}

#[tokio::test]
async fn dimensions_truncation_keeps_unit_norm() {
    let metrics = Arc::new(InferenceMetrics::new());
    let app = model_backed_router(&metrics);
    let (status, json) = post_embeddings(
        app,
        serde_json::json!({ "input": "matryoshka", "dimensions": 8 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["dimension"].as_u64(), Some(8));
    assert_eq!(json["normalized"].as_bool(), Some(true));

    let data = json["data"].as_array().expect("data array");
    let vector = float_vector(&data[0]);
    assert_eq!(vector.len(), 8, "the vector must actually be truncated");
    let norm = l2_norm(&vector);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "a truncated embedding must be RE-normalised (SV-02 correction (a)); got {norm}"
    );
}

#[tokio::test]
async fn two_texts_in_one_batch_get_distinct_unit_vectors() {
    let metrics = Arc::new(InferenceMetrics::new());
    let app = model_backed_router(&metrics);
    let (status, json) = post_embeddings(
        app,
        serde_json::json!({ "input": ["first text", "second text", "first text"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 3);
    let a = float_vector(&data[0]);
    let b = float_vector(&data[1]);
    let a_again = float_vector(&data[2]);
    assert_ne!(a, b, "different inputs must give different vectors");
    assert_eq!(
        a, a_again,
        "the same input twice in one batch must give the same vector"
    );
    for (i, item) in data.iter().enumerate() {
        let norm = l2_norm(&float_vector(item));
        assert!((norm - 1.0).abs() < 1e-4, "item {i} norm {norm}");
    }
}

#[tokio::test]
async fn token_id_input_is_embedded_as_ids_not_as_decimal_text() {
    let metrics = Arc::new(InferenceMetrics::new());
    let embedder = weighted_embedder();
    let ids: Vec<u32> = vec![10, 20, 30];
    let expected = embedder.embed_tokens(&ids).expect("direct id embed");

    let app = create_embeddings_router_from_state(
        EmbeddingAppState::from_registry(
            EmbedderRegistry::new(32)
                .with_require_model_backend(true)
                .with_model(Arc::clone(&embedder)),
        )
        .with_metrics(Arc::clone(&metrics)),
    );
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": ids })).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(
        json["usage"]["prompt_tokens"].as_u64(),
        Some(3),
        "usage must charge the THREE ids actually embedded, not the tokens of the decimal \
         string \"10 20 30\" that this request never embeds: {json}"
    );
    let data = json["data"].as_array().expect("data array");
    let served = float_vector(&data[0]);
    for (i, (got, want)) in served.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "component {i}: a `\"input\": [10, 20, 30]` request must embed those TOKEN IDS, \
             not the string \"10 20 30\"; got {got} want {want}"
        );
    }
}

#[tokio::test]
async fn usage_prompt_tokens_comes_from_the_real_tokenizer() {
    let metrics = Arc::new(InferenceMetrics::new());
    let app = model_backed_router(&metrics);
    // Four characters, one whitespace-delimited "word": the old word-count
    // approximation would report 1.
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": "abcd" })).await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(
        json["usage"]["prompt_tokens"].as_u64(),
        Some(4),
        "with a model backend installed, usage must be the tokenizer's count, not a \
         whitespace word count: {json}"
    );
    assert_eq!(json["usage"]["total_tokens"].as_u64(), Some(4));
    assert_eq!(
        metrics.prompt_tokens_total.get(),
        4,
        "the same count must reach the shared metrics registry"
    );
}

#[tokio::test]
async fn empty_input_is_refused_before_any_model_work() {
    let metrics = Arc::new(InferenceMetrics::new());
    let app = model_backed_router(&metrics);
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": [] })).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("input"),
        "the error body must name the offending field: {json}"
    );
    assert_eq!(
        metrics.errors_total.get(),
        1,
        "a refused request is still an error in /metrics"
    );
}

// ─── 3b. Batched lock and truncation, end to end ─────────────────────────────

#[tokio::test]
async fn http_batch_request_takes_the_engine_lock_once_for_the_whole_batch() {
    let embedder = weighted_embedder();
    let metrics = Arc::new(InferenceMetrics::new());
    let app = create_embeddings_router_from_state(
        EmbeddingAppState::from_registry(
            EmbedderRegistry::new(32)
                .with_require_model_backend(true)
                .with_model(Arc::clone(&embedder)),
        )
        .with_metrics(Arc::clone(&metrics)),
    );
    let before = embedder.lock_acquisitions();
    let (status, json) = post_embeddings(
        app,
        serde_json::json!({ "input": ["one", "two", "three", "four"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{json}");
    assert_eq!(json["data"].as_array().expect("data").len(), 4);
    assert_eq!(
        embedder.lock_acquisitions() - before,
        1,
        "a 4-input POST /v1/embeddings request must take the engine lock exactly once end to \
         end, not once per item"
    );
}

/// A [`ModelEmbedder`] with a deliberately tiny token ceiling, for the
/// truncation tests below.
fn embedder_with_max_tokens(max_tokens: usize) -> Arc<ModelEmbedder> {
    Arc::new(
        ModelEmbedder::new(Arc::new(Mutex::new(weighted_engine())), fixture_tokenizer())
            .with_max_tokens(max_tokens),
    )
}

#[tokio::test]
async fn text_longer_than_max_tokens_is_refused_with_context_length_exceeded() {
    let metrics = Arc::new(InferenceMetrics::new());
    let embedder = embedder_with_max_tokens(4);
    let app = create_embeddings_router_from_state(
        EmbeddingAppState::from_registry(
            EmbedderRegistry::new(32)
                .with_require_model_backend(true)
                .with_model(Arc::clone(&embedder)),
        )
        .with_metrics(Arc::clone(&metrics)),
    );
    // The char-level fixture tokenizer maps one id per character, so 10
    // characters is unambiguously 10 tokens against a ceiling of 4.
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": "0123456789" })).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an over-length TEXT input must be refused, not silently truncated and billed in \
         full: {json}"
    );
    assert_eq!(
        json["error"]["code"].as_str(),
        Some("context_length_exceeded"),
        "{json}"
    );
    assert_eq!(json["error"]["n_tokens"].as_u64(), Some(10), "{json}");
    assert_eq!(json["error"]["max_tokens"].as_u64(), Some(4), "{json}");
    assert_eq!(
        metrics.errors_total.get(),
        1,
        "a refused over-long request is still an error in /metrics"
    );
    assert_eq!(
        metrics.prompt_tokens_total.get(),
        0,
        "a request refused before any embedding work must not be billed at all"
    );
}

#[tokio::test]
async fn token_id_input_longer_than_max_tokens_still_truncates_and_charges_only_what_was_embedded()
{
    let metrics = Arc::new(InferenceMetrics::new());
    let embedder = embedder_with_max_tokens(4);
    let app = create_embeddings_router_from_state(
        EmbeddingAppState::from_registry(
            EmbedderRegistry::new(32)
                .with_require_model_backend(true)
                .with_model(Arc::clone(&embedder)),
        )
        .with_metrics(Arc::clone(&metrics)),
    );
    // 6 ids against a ceiling of 4: the raw token-id path must keep
    // truncating (its deliberate carve-out from the text refusal), not error.
    let ids: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": ids })).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the raw token-id path must keep truncating rather than erroring: {json}"
    );
    assert_eq!(
        json["usage"]["prompt_tokens"].as_u64(),
        Some(4),
        "usage must charge exactly the 4 ids actually embedded after truncation, not the 6 \
         supplied: {json}"
    );
    assert_eq!(
        metrics.prompt_tokens_total.get(),
        4,
        "the shared metrics must see the same corrected (truncated) count: {json}"
    );

    // And the vector actually served must equal embedding only the first 4
    // ids directly — proving this is really truncation, not a coincidence
    // in the token count alone.
    let expected = embedder
        .embed_tokens(&ids[..4])
        .expect("direct truncated embed");
    let served = float_vector(&json["data"][0]);
    for (i, (got, want)) in served.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "component {i}: got {got} want {want}"
        );
    }
}

// ─── 4. SV-25: both branches are instrumented ────────────────────────────────

#[tokio::test]
async fn the_success_branch_records_onto_the_shared_metrics() {
    let metrics = Arc::new(InferenceMetrics::new());
    let app = model_backed_router(&metrics);
    let (status, _json) =
        post_embeddings(app, serde_json::json!({ "input": "instrumented" })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(metrics.requests_total.get(), 1, "requests_total");
    assert_eq!(metrics.errors_total.get(), 0, "errors_total");
    assert_eq!(
        metrics.request_duration_seconds.count(),
        1,
        "request_duration_seconds"
    );
    assert!(
        metrics.prompt_tokens_total.get() > 0,
        "prompt_tokens_total must be charged"
    );
    assert!(
        (metrics.active_requests.get() - 0.0).abs() < f64::EPSILON,
        "the in-flight gauge must return to zero once the request is done; got {}",
        metrics.active_requests.get()
    );
}

#[tokio::test]
async fn the_501_refusal_branch_records_onto_the_shared_metrics() {
    let metrics = Arc::new(InferenceMetrics::new());
    let app = refusing_router(&metrics);
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": "refused" })).await;
    assert_eq!(
        status,
        StatusCode::NOT_IMPLEMENTED,
        "with no model installed the endpoint must still refuse honestly: {json}"
    );
    assert!(json["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("model-backed"));
    assert_eq!(
        metrics.requests_total.get(),
        1,
        "SV-25: the refusal branch is a request too"
    );
    assert_eq!(
        metrics.errors_total.get(),
        1,
        "SV-25: a 501 must be visible as an error, not invisible"
    );
    assert_eq!(metrics.request_duration_seconds.count(), 1);
    assert!(
        (metrics.active_requests.get() - 0.0).abs() < f64::EPSILON,
        "the in-flight gauge must return to zero on the refusal path too"
    );
}

#[tokio::test]
async fn a_router_without_metrics_still_serves() {
    // The bare library constructors attach no metrics registry; the handler
    // must skip instrumentation rather than fabricate a private one.
    let app = create_embeddings_router_from_state(EmbeddingAppState::from_registry(
        EmbedderRegistry::new(32)
            .with_require_model_backend(true)
            .with_model(weighted_embedder()),
    ));
    let (status, json) = post_embeddings(app, serde_json::json!({ "input": "no metrics" })).await;
    assert_eq!(status, StatusCode::OK, "{json}");
}

// ─── 5. Real-model semantic smoke test ───────────────────────────────────────

/// `cosine("king", "queen") > cosine("king", "banana")` on the real 1.7B
/// ternary model.
///
/// This is the assertion that a synthetic fixture *cannot* make: random
/// ternary weights carry no semantics, so only a trained model can be expected
/// to place two royal titles nearer each other than either is to a fruit. Runs
/// only when both `models/Ternary-Bonsai-1.7B.gguf` and `models/tokenizer.json`
/// are present; otherwise it self-skips through the capability report
/// (`executed: false`) rather than passing green.
///
/// # Anisotropy: why the second, sentence-level assertion is the real gate
///
/// Raw mean-pooled hidden states from a causal LM are strongly anisotropic —
/// they occupy a narrow cone, so *every* pair of short inputs has a cosine
/// near `1`. Measured on this model (Apple M3, `KernelTier` auto-detected
/// CPU): every one-token input — `king`, `queen`, `banana`, `apple`, `the` —
/// sits within `0.9989 … 0.9996` of every other, and the dominant axis of
/// variation is the input's *token count*, not its meaning (a two-token word
/// like `monarch` lands at ~0.81 from `king`). The ordering
/// `cos(king, queen) > cos(king, banana)` therefore does hold, but by only
/// ~2.5e-4 — a true ordering with a margin too narrow to defend on its own.
///
/// Averaging more than one or two hidden states dissolves most of that: two
/// sentences differing in a single word measure ~0.979, while a sentence about
/// breakfast measures ~0.619 from either — a margin of ~0.36. Both assertions
/// are kept: the first because it is the classic sanity check, the second
/// because it is the one that would actually catch a regression.
///
/// This is a property of the mean-pooling recipe, not a defect in it; a
/// production deployment that needs isotropic vectors
/// whitens or centres them downstream, which needs a corpus and is out of this
/// endpoint's scope.
#[test]
fn real_model_places_queen_nearer_to_king_than_banana() {
    const TEST_NAME: &str = "oxibonsai-runtime::embeddings_model_backed::\
                             real_model_places_queen_nearer_to_king_than_banana";

    let Some(model_path) = find_model("Ternary-Bonsai-1.7B.gguf") else {
        eprintln!(
            "skip: Ternary-Bonsai-1.7B.gguf not found under {:?}",
            models_dir()
        );
        record_skipped(Capability::LegacyModels, TEST_NAME);
        return;
    };
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, TEST_NAME);
        return;
    };
    let gate_start = std::time::Instant::now();
    let tokenizer_path_str = tokenizer_path
        .to_str()
        .expect("models/ path is valid UTF-8");
    let tokenizer =
        Arc::new(TokenizerBridge::from_file(tokenizer_path_str).expect("load real tokenizer.json"));
    let gguf_bytes = std::fs::read(&model_path).expect("read the real 1.7B GGUF");

    // Auto-detected CPU tier (NEON on this host), not `Reference`: this test
    // runs six full forward passes over a 1.7B model and the scalar reference
    // kernel makes that needlessly slow. Nothing here compares tiers, so the
    // fastest correct CPU path is the right one.
    let engine = static_engine_auto_tier(gguf_bytes);
    let embedder = ModelEmbedder::new(Arc::new(Mutex::new(engine)), tokenizer);

    let king = embedder.embed("king").expect("embed king");
    let queen = embedder.embed("queen").expect("embed queen");
    let banana = embedder.embed("banana").expect("embed banana");

    for (name, v) in [("king", &king), ("queen", &queen), ("banana", &banana)] {
        let norm = l2_norm(v);
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "{name}: a real-model embedding must be unit-length; got {norm}"
        );
    }

    let royal = cosine(&king, &queen);
    let fruit = cosine(&king, &banana);
    // Printed (visible under `--nocapture`) so the *margin* is inspectable: a
    // pass with royal ≈ fruit ≈ 1.0 would mean the model collapsed every input
    // to one vector, which the strict inequality alone would not reveal.
    eprintln!(
        "real-model single-word cosines: cos(king, queen) = {royal:.6}, \
         cos(king, banana) = {fruit:.6}, margin = {:.6}",
        royal - fruit
    );
    assert!(
        royal > fruit,
        "a real model-backed embedding must place \"queen\" nearer to \"king\" than \
         \"banana\" is: cos(king, queen) = {royal}, cos(king, banana) = {fruit}"
    );

    // The single-word ordering above is correct but its margin is ~2e-4 (see
    // this test's doc comment on anisotropy), which is too narrow to be much
    // of a gate on its own. The same comparison over full sentences — where
    // pooling has more than one or two hidden states to average — is where
    // the geometry is actually informative, so that is asserted too, with a
    // margin wide enough that a regression cannot slip through as noise.
    let king_sentence = embedder
        .embed("The king ruled the kingdom.")
        .expect("embed the king sentence");
    let queen_sentence = embedder
        .embed("The queen ruled the kingdom.")
        .expect("embed the queen sentence");
    let banana_sentence = embedder
        .embed("I ate a banana for breakfast.")
        .expect("embed the banana sentence");
    let royal_sentences = cosine(&king_sentence, &queen_sentence);
    let fruit_sentences = cosine(&king_sentence, &banana_sentence);
    eprintln!(
        "real-model sentence cosines: royal/royal = {royal_sentences:.6}, \
         royal/fruit = {fruit_sentences:.6}, margin = {:.6}",
        royal_sentences - fruit_sentences
    );
    assert!(
        royal_sentences - fruit_sentences > 0.1,
        "two sentences differing only in \"king\"/\"queen\" must be far nearer to each other \
         than either is to a sentence about breakfast: {royal_sentences} vs {fruit_sentences}"
    );

    record_executed_timed(Capability::LegacyModels, TEST_NAME, gate_start.elapsed());
}

// ─── 6. Benchmark: the batched path against the per-token loop ───────────────

/// The request budget the HTTP layer gives one embeddings request; the
/// benchmark reports whether the longest input fits inside it.
const EMBED_REQUEST_BUDGET_SECS: f64 = 60.0;

/// Timed runs per leg; every figure the benchmark reports is the minimum —
/// the sample closest to the machine's own floor under contention.
const BENCH_RUNS: usize = 3;

/// Input lengths the benchmark measures: a short query, a paragraph, and the
/// embedder's default token ceiling (`DEFAULT_MAX_EMBEDDING_TOKENS`).
const BENCH_LENGTHS: [usize; 3] = [10, 200, 2000];

/// From this input length the batched CPU pass and the per-token loop are
/// slow enough (tens of seconds to minutes per run) to be timed once instead
/// of [`BENCH_RUNS`] times, and the per-token loop needs its own opt-in (see
/// [`per_token_reference_runs`]).
const SLOW_LEG_FROM_TOKENS: usize = 1000;

/// The acceptance ratio for the shortest input.
const SHORT_INPUT_MIN_SPEEDUP: f64 = 10.0;

/// A few paragraphs of ordinary English; repeated, it tokenizes to well over
/// the longest benchmark length.
const BENCH_TEXT: &str = "Rivers carve valleys over millions of years. Water collects in the \
    highlands, gathers speed as it descends, and carries sand and stone that grind the riverbed \
    deeper with every flood. Where the land flattens, the current slows and drops its load, \
    building fertile plains and deltas that have fed cities since the first harvests. \
    Engineers now build dams and levees to tame that energy, trading the old cycle of flood and \
    renewal for steady water and electricity, and the rivers answer by moving their sediment \
    somewhere else. A mixture of experts routes every token to a few specialised feed-forward \
    networks, while a dense model sends every token through all of its weights. Both are \
    trained end to end, and both are evaluated on the same benchmarks, but their memory and \
    compute profiles differ sharply at inference time. The quick brown fox jumps over the lazy \
    dog, and the dog, unimpressed, goes back to sleep in the afternoon sun. ";

/// Best-effort 1/5/15-minute load average for the benchmark record (read
/// through `uptime`: a printed diagnostic is no reason for a dependency).
fn load_average() -> String {
    match std::process::Command::new("uptime").output() {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            match text.split_once("load average") {
                Some((_, tail)) => tail.trim_start_matches([':', 's', ' ']).trim().to_string(),
                None => text.trim().to_string(),
            }
        }
        _ => "unavailable".to_string(),
    }
}

/// Mean-pool `rows` (`[n x hidden]`) and scale to unit length — the recipe
/// `embed_mean_pooled` applies, for turning the per-token loop's rows into a
/// comparable vector.
fn pool_rows(rows: &[f32], hidden: usize) -> Vec<f32> {
    let n = rows.len() / hidden;
    let mut pooled = vec![0.0f32; hidden];
    for row in rows.chunks_exact(hidden) {
        for (acc, v) in pooled.iter_mut().zip(row) {
            *acc += *v;
        }
    }
    let inv = 1.0 / n as f32;
    pooled.iter_mut().for_each(|v| *v *= inv);
    let norm = l2_norm(&pooled);
    if norm > 1e-10 {
        pooled.iter_mut().for_each(|v| *v /= norm);
    }
    pooled
}

/// Run `f` `runs` times and return every wall time plus the last result.
fn time_runs<T>(runs: usize, mut f: impl FnMut() -> T) -> (Vec<f64>, T) {
    let mut times = Vec::with_capacity(runs);
    let mut last = None;
    for _ in 0..runs.max(1) {
        let started = std::time::Instant::now();
        let value = f();
        times.push(started.elapsed().as_secs_f64());
        last = Some(value);
    }
    match last {
        Some(value) => (times, value),
        None => unreachable!("runs.max(1) >= 1 iterations always ran"),
    }
}

fn min_of(times: &[f64]) -> f64 {
    times.iter().copied().fold(f64::INFINITY, f64::min)
}

fn fmt_runs(times: &[f64]) -> String {
    times
        .iter()
        .map(|t| format!("{t:.3}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The 2000-token embedding's wall-time target on the production (Metal)
/// route, seconds: a third of the request budget, so a long input still fits
/// it on a loaded host.
const LONG_INPUT_TARGET_SECS: f64 = 20.0;

/// Environment variable that opts into the benchmark (together with
/// `OXI_MODEL`, see [`embed_bench_requested`]).
const EMBED_BENCH_ENV: &str = "OXIBONSAI_EMBED_BENCH";

/// Environment variable that opts a long input into the per-token reference
/// loop (see [`per_token_reference_requested`]); the release gate never sets
/// it.
const EMBED_BENCH_PER_TOKEN_ENV: &str = "OXIBONSAI_EMBED_BENCH_PER_TOKEN";

/// Whether the benchmark was explicitly asked for: `OXIBONSAI_EMBED_BENCH`
/// is exactly `1` **and** `OXI_MODEL` names a model.
///
/// Neither alone is enough. `OXI_MODEL` is exported for the other real-model
/// gates (a workspace `cargo nextest run` with it set reaches this test), and
/// the benchmark maps a multi-GB model and runs for minutes, so the model
/// variable alone must not start it; the opt-in alone has no model to run on,
/// and the benchmark never guesses one.
fn embed_bench_requested(embed_bench: Option<&str>, oxi_model: Option<&std::ffi::OsStr>) -> bool {
    embed_bench == Some("1") && oxi_model.is_some_and(|model| !model.is_empty())
}

/// Whether the per-token reference loop was explicitly asked for on a long
/// input: `OXIBONSAI_EMBED_BENCH_PER_TOKEN` is exactly `1`. A second opt-in on
/// top of [`embed_bench_requested`]: the loop costs about five to six minutes
/// at 2000 tokens on an M3 at a load average of about 10 (309-353 s measured,
/// longer under a heavier load), which is a manual
/// measurement, not something a gate leg or a casual benchmark run pays.
fn per_token_reference_requested(per_token: Option<&str>) -> bool {
    per_token == Some("1")
}

/// Whether the per-token reference loop runs on an `n`-token input: always
/// below [`SLOW_LEG_FROM_TOKENS`] (the 10- and 200-token inputs, seconds
/// each), and from there only when `per_token_opt_in` is set.
fn per_token_reference_runs(n: usize, per_token_opt_in: bool) -> bool {
    n < SLOW_LEG_FROM_TOKENS || per_token_opt_in
}

/// What the per-token reference loop measured on one input.
struct PerTokenLeg {
    /// Wall time of every run, seconds.
    times: Vec<f64>,
    /// The loop's rows, mean-pooled and normalised.
    pooled: Vec<f32>,
}

/// The tokenizer the benchmark runs with: the testkit resolver's
/// `tokenizer.json`, else the one next to the model file. `None` when neither
/// exists.
fn bench_tokenizer(model: &std::path::Path) -> Option<std::path::PathBuf> {
    find_model("tokenizer.json").or_else(|| {
        let beside = model.parent()?.join("tokenizer.json");
        beside.is_file().then_some(beside)
    })
}

/// The benchmark needs both variables, and neither alone starts it.
#[test]
fn the_embed_bench_needs_both_the_opt_in_and_a_model() {
    use std::ffi::OsStr;
    let model = Some(OsStr::new("model.gguf"));
    assert!(embed_bench_requested(Some("1"), model));
    assert!(
        !embed_bench_requested(None, model),
        "OXI_MODEL alone must not start the benchmark"
    );
    assert!(!embed_bench_requested(Some("0"), model));
    assert!(
        !embed_bench_requested(Some("true"), model),
        "only the exact value 1 opts in"
    );
    assert!(
        !embed_bench_requested(Some("1"), None),
        "the opt-in without a model has nothing to run on"
    );
    assert!(!embed_bench_requested(Some("1"), Some(OsStr::new(""))));
    assert!(!embed_bench_requested(None, None));
}

/// The per-token reference loop on a long input has its own opt-in: the
/// benchmark's two variables never start it, and only the exact value `1`
/// does.
#[test]
fn the_per_token_reference_at_a_long_input_needs_its_own_opt_in() {
    assert!(per_token_reference_requested(Some("1")));
    assert!(
        !per_token_reference_requested(None),
        "the benchmark's own opt-in must not start the per-token reference"
    );
    for not_exactly_one in ["0", "true", "yes", "", " 1", "1 ", "11"] {
        assert!(
            !per_token_reference_requested(Some(not_exactly_one)),
            "{not_exactly_one:?} is not the explicit opt-in"
        );
    }

    // The benchmark's own lengths: the short and medium inputs always measure
    // the per-token reference, the 2000-token one only on the opt-in.
    assert_eq!(BENCH_LENGTHS, [10, 200, 2000]);
    for n in [BENCH_LENGTHS[0], BENCH_LENGTHS[1]] {
        assert!(per_token_reference_runs(n, false), "{n} tokens, no opt-in");
        assert!(per_token_reference_runs(n, true), "{n} tokens, opt-in");
    }
    assert!(
        !per_token_reference_runs(BENCH_LENGTHS[2], false),
        "2000 tokens must skip the per-token reference unless explicitly asked"
    );
    assert!(per_token_reference_runs(BENCH_LENGTHS[2], true));

    // The boundary itself, and nothing above it that always runs.
    assert!(per_token_reference_runs(SLOW_LEG_FROM_TOKENS - 1, false));
    assert!(!per_token_reference_runs(SLOW_LEG_FROM_TOKENS, false));
    assert!(!per_token_reference_runs(usize::MAX, false));
    // The short-input speed-up assertion needs the shortest input to be
    // measured without any opt-in.
    assert!(BENCH_LENGTHS[0] < SLOW_LEG_FROM_TOKENS);
}

/// 10-, 200- and 2000-token embeddings on a real model (`OXI_MODEL`):
/// the production `InferenceEngine::embed` — the head-free Metal prefill on a
/// GPU engine — against the batched CPU pass the same engine ran before it
/// (`forward_hidden` on a CPU-tier dispatcher) and the per-token host-KV loop
/// both replaced (`BonsaiModel::forward_hidden_sequential` on a dispatcher
/// built exactly as the engine's own), minimum of three runs per leg (one
/// run for the batched CPU pass and the per-token loop at 1000+ tokens), with
/// the load average printed beside every figure.
///
/// Asserts, on every input, that the production embedding agrees with the
/// batched CPU pass (a pooled cosine of at least 0.9999); on every input that
/// measures the per-token reference — the 10- and 200-token ones always, the
/// 2000-token one only with `OXIBONSAI_EMBED_BENCH_PER_TOKEN=1` (see
/// [`per_token_reference_runs`]; the release gate never sets it) — that both
/// legs agree with it too; the acceptance ratio on the 10-token input; and,
/// on a GPU engine, that the 2000-token embedding takes at most
/// [`LONG_INPUT_TARGET_SECS`]. The verdict and every assertion message
/// carry the load average the figures were measured under. Runs only when
/// both `OXIBONSAI_EMBED_BENCH=1` and `OXI_MODEL` are set (release build; see
/// [`embed_bench_requested`]), and self-skips — recording the skip —
/// otherwise or when the tokenizer is missing.
#[test]
fn embed_bench_short_and_long() {
    use oxibonsai_kernels::{KernelDispatcher, OneBitKernel};

    const TEST_NAME: &str =
        "oxibonsai-runtime::embeddings_model_backed::embed_bench_short_and_long";
    let embed_bench = std::env::var(EMBED_BENCH_ENV).ok();
    let oxi_model = std::env::var_os("OXI_MODEL");
    let model_path = match oxi_model.as_deref() {
        Some(model) if embed_bench_requested(embed_bench.as_deref(), Some(model)) => {
            std::path::PathBuf::from(model)
        }
        _ => {
            eprintln!(
                "{TEST_NAME}: set {EMBED_BENCH_ENV}=1 and OXI_MODEL=<gguf> in a release build to \
                 run (the benchmark maps a multi-GB model and runs for minutes, so neither \
                 variable starts it alone) -- skipping"
            );
            record_skipped(Capability::LegacyModels, TEST_NAME);
            return;
        }
    };
    let Some(tokenizer_path) = bench_tokenizer(&model_path) else {
        eprintln!(
            "{TEST_NAME}: tokenizer.json is missing ({:?} and next to {model_path:?}) -- skipping",
            models_dir()
        );
        record_skipped(Capability::LegacyModels, TEST_NAME);
        return;
    };
    let gate_start = std::time::Instant::now();
    let tokenizer = TokenizerBridge::from_file(
        tokenizer_path
            .to_str()
            .expect("the tokenizer path is valid UTF-8"),
    )
    .expect("load the real tokenizer.json");
    let longest = BENCH_LENGTHS.iter().copied().max().unwrap_or(0);
    let mut text = String::new();
    let ids = loop {
        text.push_str(BENCH_TEXT);
        let ids = tokenizer
            .encode(&text)
            .expect("tokenize the benchmark text");
        if ids.len() >= longest {
            break ids;
        }
    };

    // The production constructor: the engine `oxibonsai serve`'s embedder
    // builds, on the auto-detected backend.
    let (mut engine, _gguf) = oxibonsai_runtime::engine::InferenceEngine::from_gguf_path_leaked(
        &model_path,
        greedy_params(),
        42,
        4096,
    )
    .expect("load the real model through the production constructor");
    // The dispatcher the engine was built with (`Backend::Auto`), rebuilt so
    // the per-token leg can borrow the model mutably beside it.
    let per_token_kernel = KernelDispatcher::auto_detect();
    assert_eq!(per_token_kernel.tier(), engine.kernel_tier());
    let cpu_kernel = KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier());
    // `KernelTier::Gpu` exists only with the kernels' `gpu` feature; the
    // dispatcher answers on every build.
    let gpu_engine = per_token_kernel.is_gpu_accelerated();
    let hidden = engine.hidden_size();
    let per_token_opt_in =
        per_token_reference_requested(std::env::var(EMBED_BENCH_PER_TOKEN_ENV).ok().as_deref());
    eprintln!(
        "embed_bench: model {model_path:?}, engine tier {:?}; per-token leg on {:?}; batched \
         CPU leg on {:?}; per-token reference at {SLOW_LEG_FROM_TOKENS}+ tokens: {}; load \
         average before: {}",
        engine.kernel_tier(),
        per_token_kernel.tier(),
        cpu_kernel.tier(),
        if per_token_opt_in {
            format!("on ({EMBED_BENCH_PER_TOKEN_ENV}=1)")
        } else {
            format!("off (set {EMBED_BENCH_PER_TOKEN_ENV}=1 to measure it)")
        },
        load_average()
    );

    let mut short_speedup = None;
    for n in BENCH_LENGTHS {
        let tokens = &ids[..n];
        let slow_runs = if n >= SLOW_LEG_FROM_TOKENS {
            1
        } else {
            BENCH_RUNS
        };
        let (production_times, production) = time_runs(BENCH_RUNS, || {
            engine.embed(tokens).expect("production embed")
        });
        let (cpu_times, cpu_rows) = time_runs(slow_runs, || {
            engine
                .dense_model_mut()
                .expect("a dense model")
                .forward_hidden(tokens, &cpu_kernel)
                .expect("batched CPU forward_hidden")
        });
        let per_token_leg = per_token_reference_runs(n, per_token_opt_in).then(|| {
            let (times, rows) = time_runs(slow_runs, || {
                engine
                    .dense_model_mut()
                    .expect("a dense model")
                    .forward_hidden_sequential(tokens, &per_token_kernel)
                    .expect("per-token forward_hidden")
            });
            PerTokenLeg {
                times,
                pooled: pool_rows(&rows, hidden),
            }
        });
        let cpu = pool_rows(&cpu_rows, hidden);
        let cos_production_cpu = cosine(&production, &cpu);
        let production_min = min_of(&production_times);
        let cpu_min = min_of(&cpu_times);
        // `(per-token min, production speed-up, cos(production), cos(batched
        // CPU))` against the per-token reference, when it ran.
        let per_token = per_token_leg.as_ref().map(|leg| {
            let per_token_min = min_of(&leg.times);
            (
                per_token_min,
                per_token_min / production_min.max(f64::MIN_POSITIVE),
                cosine(&production, &leg.pooled),
                cosine(&cpu, &leg.pooled),
            )
        });
        let per_token_report = match per_token_leg.as_ref().zip(per_token) {
            Some((leg, (per_token_min, speedup, cos_production, cos_cpu))) => format!(
                "per-token loop min {per_token_min:.3} s [{}]: production {speedup:.1}x faster; \
                 pooled cos vs per-token: production {cos_production:.6}, batched CPU \
                 {cos_cpu:.6}",
                fmt_runs(&leg.times)
            ),
            None => format!(
                "per-token loop skipped at {n} tokens (set {EMBED_BENCH_PER_TOKEN_ENV}=1 to \
                 measure it)"
            ),
        };
        eprintln!(
            "embed_bench: {n} tokens: production ({}) min {production_min:.3} s [{}]; batched \
             CPU min {cpu_min:.3} s [{}] ({:.1}x slower than production); pooled cos production \
             vs batched CPU {cos_production_cpu:.6}; {per_token_report}; load {}",
            if gpu_engine {
                "Metal hidden prefill"
            } else {
                "batched CPU prefill"
            },
            fmt_runs(&production_times),
            fmt_runs(&cpu_times),
            cpu_min / production_min.max(f64::MIN_POSITIVE),
            load_average()
        );
        assert!(
            f64::from(cos_production_cpu) >= 0.9999,
            "{n} tokens: the production embedding diverged from the batched CPU pass: \
             cos {cos_production_cpu}"
        );
        if let Some((_, speedup, cos_production, cos_cpu)) = per_token {
            assert!(
                f64::from(cos_production) >= 0.9999,
                "{n} tokens: the production embedding diverged from the per-token loop: \
                 cos {cos_production}"
            );
            assert!(
                f64::from(cos_cpu) >= 0.9999,
                "{n} tokens: the batched CPU embedding diverged from the per-token loop: \
                 cos {cos_cpu}"
            );
            if n == BENCH_LENGTHS[0] {
                short_speedup = Some(speedup);
            }
        }
        if n == longest {
            eprintln!(
                "embed_bench: {n} tokens: production wall time {production_min:.3} s (target \
                 {LONG_INPUT_TARGET_SECS:.0} s, request budget {EMBED_REQUEST_BUDGET_SECS:.0} s: \
                 {}); load {}",
                if production_min <= LONG_INPUT_TARGET_SECS {
                    "met"
                } else {
                    "MISSED"
                },
                load_average()
            );
            if gpu_engine {
                assert!(
                    production_min <= LONG_INPUT_TARGET_SECS,
                    "{n} tokens: the Metal embedding took {production_min:.3} s (best of \
                     {BENCH_RUNS}), over the {LONG_INPUT_TARGET_SECS} s target; load average {}",
                    load_average()
                );
            }
        }
    }
    eprintln!("embed_bench: load average after: {}", load_average());
    let short_speedup =
        short_speedup.expect("the shortest input always measures the per-token reference");
    assert!(
        short_speedup >= SHORT_INPUT_MIN_SPEEDUP,
        "{} tokens: the production path must be at least {SHORT_INPUT_MIN_SPEEDUP}x faster than \
         the per-token loop, measured {short_speedup:.2}x; load average {}",
        BENCH_LENGTHS[0],
        load_average()
    );
    record_executed_timed(Capability::LegacyModels, TEST_NAME, gate_start.elapsed());
}
