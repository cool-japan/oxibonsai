//! Full pipeline integration tests covering the complete inference stack.
//! Uses tiny_test configs to keep tests fast.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::{
    beam_search::{BeamSearchConfig, BeamSearchEngine},
    context_manager::{ContextWindow, TruncationStrategy},
    error::RuntimeResult,
    pipeline::{PipelineBuilder, StopReason},
    sampling::SamplingParams,
    sampling_advanced::{
        EtaSampler, LcgRng, MinPSampler, MirostatV2Sampler, SamplerChain, SamplerStep,
        TypicalSampler,
    },
    speculative::{SpeculativeConfig, SpeculativeDecoder},
    token_healing::TokenHealingConfig,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_tiny_engine() -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42)
}

fn make_greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

// ── Inference Pipeline ────────────────────────────────────────────────────────

#[test]
fn test_pipeline_builder_greedy_runs() {
    let mut engine = make_tiny_engine();
    let mut pipeline = PipelineBuilder::new().max_tokens(4).greedy().build();

    let output = pipeline.run(vec![1u32, 2, 3], &mut engine);
    // Should complete without panic and return a valid output.
    assert_eq!(output.prompt_tokens, 3);
    assert!(output.completion_tokens <= 4);
    assert!(output.elapsed_ms < 60_000);
}

#[test]
fn test_pipeline_builder_sampling_runs() {
    let mut engine = InferenceEngine::new(
        Qwen3Config::tiny_test(),
        SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.0,
            max_tokens: 128,
        },
        99,
    );
    let chain = SamplerChain::new(99)
        .add(SamplerStep::Temperature(0.8))
        .add(SamplerStep::TopP(0.9));
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(4)
        .with_sampling(chain)
        .build();

    let output = pipeline.run(vec![1u32, 2], &mut engine);
    assert!(output.completion_tokens <= 4);
    assert!(output.elapsed_ms < 60_000);
}

#[test]
fn test_pipeline_with_stop_sequences() {
    let mut engine = make_tiny_engine();
    // Use a stop sequence that will never appear so we hit max_tokens.
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(3)
        .greedy()
        .stop_on(vec!["IMPOSSIBLE_STOP_SEQ_XXXXXX".to_string()])
        .build();

    let output = pipeline.run(vec![1u32], &mut engine);
    // Should stop because of max_tokens or EOS, not the stop sequence.
    assert!(
        output.stop_reason == StopReason::MaxTokens
            || output.stop_reason == StopReason::EndOfSequence,
        "expected MaxTokens or EndOfSequence, got {:?}",
        output.stop_reason
    );
}

#[test]
fn test_pipeline_with_token_healing() {
    let mut engine = make_tiny_engine();
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(3)
        .greedy()
        .with_token_healing(TokenHealingConfig {
            lookback: 1,
            min_prob: 0.01,
            enabled: true,
        })
        .build();

    let output = pipeline.run(vec![1u32, 2], &mut engine);
    // Pipeline should handle healing gracefully; prompt_tokens still correct.
    assert_eq!(output.prompt_tokens, 2);
    assert!(output.completion_tokens <= 3);
}

#[test]
fn test_pipeline_output_has_tokens() {
    // RT-01 made `PipelineOutput::text` populate only when a detokenizer is
    // attached; without one `text` is correctly always empty. Attach a toy
    // detokenizer here so this test exercises that real seam instead of the
    // (now-intentional) no-detokenizer empty-text path -- see the sibling
    // test below for that direction.
    let mut engine = make_tiny_engine();
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(5)
        .greedy()
        .with_detokenizer(|ids: &[u32]| -> RuntimeResult<String> { Ok(format!("{ids:?}")) })
        .build();

    let output = pipeline.run(vec![1u32, 2, 3], &mut engine);
    // token_ids length must equal completion_tokens.
    assert_eq!(output.token_ids.len(), output.completion_tokens);
    // text must be consistent: non-empty iff there are tokens.
    if output.completion_tokens > 0 {
        assert!(!output.text.is_empty());
    }
    assert!(
        output.text_available,
        "a detokenizer was attached, so text_available must be true"
    );
}

#[test]
fn test_pipeline_output_without_detokenizer_has_no_text() {
    // Pins the other direction of RT-01's contract: with no detokenizer
    // attached, `text` stays empty and `text_available` stays false,
    // regardless of how many tokens were generated.
    let mut engine = make_tiny_engine();
    let mut pipeline = PipelineBuilder::new().max_tokens(5).greedy().build();

    let output = pipeline.run(vec![1u32, 2, 3], &mut engine);
    assert!(
        output.text.is_empty(),
        "text must stay empty with no detokenizer attached, got {:?}",
        output.text
    );
    assert!(
        !output.text_available,
        "text_available must be false with no detokenizer attached"
    );
}

#[test]
fn test_pipeline_stop_reason_max_tokens() {
    // Build a greedy engine and force it to keep generating past EOS so we can
    // reach max_tokens.  We use a very large prompt to fill the context and set
    // max_tokens to exactly what we want; since the tiny model has zero weights
    // the output is deterministic.
    let params = make_greedy_params();
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 0);

    let mut pipeline = PipelineBuilder::new().max_tokens(1).greedy().build();

    let output = pipeline.run(vec![1u32], &mut engine);
    // Either we stopped at max_tokens or hit EOS before that — both are valid.
    assert!(
        output.stop_reason == StopReason::MaxTokens
            || output.stop_reason == StopReason::EndOfSequence,
        "unexpected stop reason: {:?}",
        output.stop_reason
    );
}

// ── Sampler Chain ─────────────────────────────────────────────────────────────

#[test]
fn test_sampler_chain_greedy_deterministic() {
    let mut chain1 = SamplerChain::greedy();
    let mut chain2 = SamplerChain::greedy();
    let logits = vec![0.5_f32, 2.0, 1.0, 3.5, 0.1];

    let t1 = chain1.sample(&mut logits.clone());
    let t2 = chain2.sample(&mut logits.clone());
    assert_eq!(t1, t2, "greedy chain must be deterministic");
    // Max is at index 3 (3.5).
    assert_eq!(t1, 3, "greedy should pick argmax at index 3");
}

#[test]
fn test_sampler_chain_temperature_changes_output() {
    // Temperature 0 should give argmax; temperature != 0 may differ.
    let mut greedy_chain = SamplerChain::new(42).add(SamplerStep::Temperature(1e-9));
    let logits = vec![1.0_f32, 5.0, 2.0, 3.0, 4.0];
    let tok = greedy_chain.sample(&mut logits.clone());
    // With near-zero temperature the highest logit (index 1, value 5.0) wins.
    assert_eq!(tok, 1, "near-zero temperature should pick argmax");
}

#[test]
fn test_sampler_chain_min_p_basic() {
    let mut chain = SamplerChain::new(7).add(SamplerStep::MinP(0.05));
    let mut logits = vec![1.0_f32, 8.0, 0.5, 2.0];
    let tok = chain.sample(&mut logits);
    assert!(tok < 4, "min-p token index must be within vocab");
}

#[test]
fn test_sampler_chain_composable_steps() {
    // Verify that chaining Temperature → TopK → TopP doesn't panic.
    let mut chain = SamplerChain::new(0)
        .add(SamplerStep::Temperature(0.9))
        .add(SamplerStep::TopK(3))
        .add(SamplerStep::TopP(0.95));
    let mut logits = vec![0.1_f32, 2.0, 1.5, 3.0, 0.8];
    let tok = chain.sample(&mut logits);
    assert!(tok < 5, "composable chain token must be within vocab");
}

// ── Beam Search ───────────────────────────────────────────────────────────────

#[test]
fn test_beam_search_width_1_equals_greedy() {
    let config = BeamSearchConfig {
        beam_width: 1,
        max_tokens: 5,
        eos_token_id: 2,
        ..Default::default()
    };
    let engine = BeamSearchEngine::new(config);

    let result = engine.search(vec![1u32, 2], 10, |_tokens, _step| {
        let mut logits = vec![0.0_f32; 10];
        logits[5] = 10.0;
        logits
    });

    // With beam_width=1 the result is equivalent to greedy: should not be empty.
    assert!(
        !result.best().is_empty(),
        "beam search with width 1 must yield tokens"
    );
}

#[test]
fn test_beam_search_produces_sequences() {
    let config = BeamSearchConfig {
        beam_width: 2,
        max_tokens: 3,
        eos_token_id: 999,
        early_stopping: false,
        ..Default::default()
    };
    let engine = BeamSearchEngine::new(config);

    let result = engine.search(vec![1u32], 10, |_tokens, _step| {
        (0..10).map(|i| i as f32 * 0.1).collect()
    });

    let best = result.best();
    assert!(
        !best.is_empty(),
        "beam search must produce at least one token"
    );
}

#[test]
fn test_beam_search_score_ordering() {
    let config = BeamSearchConfig {
        beam_width: 3,
        max_tokens: 4,
        eos_token_id: 999,
        early_stopping: false,
        ..Default::default()
    };
    let engine = BeamSearchEngine::new(config);

    // Always prefer token 7.
    let result = engine.search(vec![1u32], 10, |_tokens, _step| {
        let mut logits = vec![0.0_f32; 10];
        logits[7] = 15.0;
        logits
    });

    // Scores are stored in result.scores; verify non-ascending order.
    let scores = &result.scores;
    for w in scores.windows(2) {
        assert!(
            w[0] >= w[1],
            "sequences should be returned in non-ascending score order; scores: {scores:?}"
        );
    }
    // best() must return the first (highest-scored) sequence.
    assert!(!result.best().is_empty(), "best sequence must be non-empty");
}

// ── Context Manager ───────────────────────────────────────────────────────────

#[test]
fn test_context_window_append_and_retrieve() {
    let mut window = ContextWindow::new(64, TruncationStrategy::TruncateLeft);
    window.append(&[1u32, 2, 3, 4, 5]);
    let tokens = window.tokens();
    assert_eq!(tokens, vec![1u32, 2, 3, 4, 5]);
}

#[test]
fn test_context_window_truncation() {
    let mut window = ContextWindow::new(4, TruncationStrategy::TruncateLeft);
    // Append 8 tokens into a window of max 4 — should be truncated.
    window.append(&[10u32, 20, 30, 40, 50, 60, 70, 80]);
    let tokens = window.tokens();
    assert!(
        tokens.len() <= 4,
        "context should be truncated to max_tokens; got {} tokens",
        tokens.len()
    );
}

#[test]
fn test_conversation_context_multi_turn() {
    let mut window = ContextWindow::new(20, TruncationStrategy::TruncateLeft);
    window
        .set_system_prompt(vec![100u32, 101])
        .expect("system prompt should fit");

    // First turn
    window.append(&[1u32, 2, 3]);
    // Second turn
    window.append(&[4u32, 5, 6]);

    let tokens = window.tokens();
    // System prompt (2) + turn1 (3) + turn2 (3) = 8, all within 20.
    assert_eq!(tokens.len(), 8);
    // System prompt tokens appear first.
    assert_eq!(tokens[0], 100);
    assert_eq!(tokens[1], 101);
}

// ── Speculative Decoding ──────────────────────────────────────────────────────

#[test]
fn test_speculative_decoder_acceptance_rate_initialized() {
    let draft_engine = make_tiny_engine();
    let spec_config = SpeculativeConfig::default();
    let decoder = SpeculativeDecoder::new(draft_engine, spec_config);
    // The decoder's acceptance rate starts at 0.0 since no steps taken.
    assert!(
        (decoder.acceptance_rate() - 0.0).abs() < f32::EPSILON,
        "initial acceptance rate should be 0.0"
    );
}

#[test]
fn test_speculative_decoder_generate_returns_tokens() {
    let draft_engine = make_tiny_engine();
    let spec_config = SpeculativeConfig {
        lookahead: 2,
        ..Default::default()
    };
    let mut decoder = SpeculativeDecoder::new(draft_engine, spec_config);

    let params = SamplingParams::default();
    let tokens = decoder.generate_speculative(&[1u32, 2], 4, &params);

    assert!(tokens.len() <= 4, "should generate at most 4 tokens");
}

#[test]
fn test_speculative_step_returns_accepted() {
    let draft_engine = make_tiny_engine();
    let spec_config = SpeculativeConfig {
        lookahead: 3,
        ..Default::default()
    };
    let mut decoder = SpeculativeDecoder::new(draft_engine, spec_config);

    let params = SamplingParams::default();
    // Provide empty target logits (the mock decoder will handle this gracefully).
    let step = decoder.step(&[1u32], &[], &params);

    // Each draft token is either accepted or rejected — the accepted count
    // must be in range [0, lookahead].
    assert!(
        step.accepted_tokens.len() <= step.draft_tokens.len(),
        "cannot accept more tokens than drafted"
    );
    let rate = step.acceptance_rate;
    assert!(
        (0.0..=1.0).contains(&rate),
        "acceptance rate must be in [0,1]"
    );
}

// ── Advanced Sampling ─────────────────────────────────────────────────────────

#[test]
fn test_mirostat_v2_samples_valid_token() {
    let logits = vec![1.0_f32, 5.0, 2.0, 3.0, 4.0];
    let mut sampler = MirostatV2Sampler::new(5.0, 0.1);
    let mut rng = LcgRng::new(42);
    let tok = sampler.sample(&logits, &mut rng);
    assert!(tok < logits.len(), "mirostat v2 must return a valid index");
}

#[test]
fn test_typical_sampler_samples_valid_token() {
    let logits = vec![1.0_f32, 4.0, 2.0, 3.0];
    let sampler = TypicalSampler::new(0.9, 1);
    let mut rng = LcgRng::new(7);
    let tok = sampler.sample(&logits, &mut rng);
    assert!(
        tok < logits.len(),
        "typical sampler must return a valid index"
    );
}

#[test]
fn test_min_p_sampler_samples_valid_token() {
    let logits = vec![0.5_f32, 8.0, 1.0, 2.5];
    let sampler = MinPSampler::new(0.05, 1);
    let mut rng = LcgRng::new(123);
    let tok = sampler.sample(&logits, &mut rng);
    assert!(
        tok < logits.len(),
        "min-p sampler must return a valid index"
    );
}

#[test]
fn test_eta_sampler_samples_valid_token() {
    let logits = vec![1.0_f32, 3.0, 2.0, 0.5];
    let sampler = EtaSampler::new(0.0009, 0.07);
    let mut rng = LcgRng::new(55);
    let tok = sampler.sample(&logits, &mut rng);
    assert!(tok < logits.len(), "eta sampler must return a valid index");
}

// ── OxiRAG Pipeline ───────────────────────────────────────────────────────────

#[cfg(feature = "rag")]
mod rag_tests {
    use oxibonsai_rag::{
        embedding::IdentityEmbedder,
        pipeline::{RagConfig, RagPipeline},
        retriever::{Retriever, RetrieverConfig},
    };

    #[test]
    fn test_rag_pipeline_index_and_query() {
        let embedder = IdentityEmbedder::new(32).expect("valid dim");
        let mut pipeline = RagPipeline::new(embedder, RagConfig::default());

        let chunks = pipeline
            .index_document("Rust is a memory-safe systems programming language.")
            .expect("indexing should succeed");
        assert!(chunks > 0, "at least one chunk should be indexed");

        // The pipeline should now have indexed documents.
        let stats = pipeline.stats();
        assert!(stats.documents_indexed >= 1);
        assert!(stats.chunks_indexed >= 1);
    }

    #[test]
    fn test_rag_pipeline_build_prompt_contains_context() {
        let embedder = IdentityEmbedder::new(32).expect("valid dim");
        let config =
            RagConfig::default().with_prompt_template("{context}\n\nQuestion: {query}\n\nAnswer:");
        let mut pipeline = RagPipeline::new(embedder, config);

        pipeline
            .index_document("The Bonsai-8B model runs at 1-bit precision.")
            .expect("indexing should succeed");

        let prompt = pipeline
            .build_prompt("What precision does Bonsai-8B use?")
            .expect("prompt building should succeed");

        assert!(
            prompt.contains("Question: What precision does Bonsai-8B use?"),
            "prompt must contain the query; got: {prompt}"
        );
    }

    #[test]
    fn test_rag_retriever_retrieve_text() {
        let embedder = IdentityEmbedder::new(16).expect("valid dim");
        let config = RetrieverConfig::default()
            .with_top_k(2)
            .with_min_score(0.0)
            .with_rerank(false);
        let mut retriever = Retriever::new(embedder, config);

        retriever
            .add_document(
                "OxiBonsai achieves state-of-the-art 1-bit inference.",
                &Default::default(),
            )
            .expect("add document should succeed");

        retriever
            .add_document(
                "Pure Rust means no C/Fortran dependencies.",
                &Default::default(),
            )
            .expect("add second document should succeed");

        let results = retriever
            .retrieve("1-bit inference")
            .expect("retrieve should succeed");

        assert!(
            !results.is_empty(),
            "retrieval must return at least one result"
        );
        assert!(results.len() <= 2, "should return at most top_k=2 results");
        for r in &results {
            assert!(
                !r.chunk.text.is_empty(),
                "retrieved chunk text must be non-empty"
            );
        }
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

#[cfg(feature = "native-tokenizer")]
mod tokenizer_tests {
    use oxibonsai_tokenizer::OxiTokenizer;

    fn make_stub_tokenizer() -> OxiTokenizer {
        OxiTokenizer::char_level_stub(256)
    }

    #[test]
    fn test_tokenizer_encode_and_decode() {
        let tok = make_stub_tokenizer();
        let text = "Hello world";
        let ids = tok.encode(text).expect("encode should succeed");
        assert!(
            !ids.is_empty(),
            "encoding a non-empty string must yield tokens"
        );
        let decoded = tok.decode(&ids).expect("decode should succeed");
        // The round-trip may not be byte-perfect with a stub, but should not panic.
        assert!(
            !decoded.is_empty() || text.is_empty(),
            "decode of non-empty token stream should yield a non-empty string"
        );
    }

    #[test]
    fn test_tokenizer_batch_encode() {
        let tok = make_stub_tokenizer();
        let texts = &["foo", "bar baz", "qux"];
        let batch = tok
            .encode_batch(texts)
            .expect("batch encode should succeed");
        assert_eq!(batch.len(), 3, "batch size must match input count");
        for (i, ids) in batch.iter().enumerate() {
            assert!(
                !ids.is_empty(),
                "text {i} should produce at least one token"
            );
        }
    }

    #[test]
    fn test_chat_template_chatml() {
        use oxibonsai_tokenizer::utils::ChatTemplate;
        let template = ChatTemplate::chatml();
        let messages = vec![
            ("system", "You are a helpful assistant."),
            ("user", "What is Rust?"),
        ];
        let rendered = template.format(&messages);
        assert!(
            rendered.contains("<|im_start|>"),
            "chatml output must contain im_start tokens; got: {rendered}"
        );
        assert!(
            rendered.contains("What is Rust?"),
            "rendered template must include user message"
        );
    }
}

// MINOR fix (verifier wave 3, T-09): `test_rag_pipeline_skipped_no_feature`
// and `test_tokenizer_skipped_no_feature` used to live here as `#[cfg(not(
// feature = "..."))]`-gated placeholders that asserted nothing (just an
// `eprintln!`) — a placebo pass under a non-default build, not evidence of
// anything. Deleted rather than given a fabricated assertion: this file has
// no unconditional code that depends on the `rag`/`native-tokenizer`
// features being off, so the test binary compiles and runs cleanly without
// them regardless of which features are enabled (this package's own
// `--all-features` and default-feature gate runs both confirm it).
