//! Integration tests for the full generate() pipeline.
//!
//! Uses `Qwen3Config::tiny_test()` to construct lightweight engines and
//! exercises the end-to-end generation path including prefill, decode,
//! sampling parameter effects, edge cases, and engine state management.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::{Sampler, SamplingParams};

// ══════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════

fn tiny_engine(params: SamplingParams, seed: u64) -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), params, seed)
}

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

fn default_prompt() -> Vec<u32> {
    vec![151644, 872, 1234]
}

// ══════════════════════════════════════════════════════════════════════════
// Basic generation
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn generate_single_token() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&default_prompt(), 1)
        .expect("generate with max_tokens=1 should succeed");
    assert!(
        tokens.len() <= 1,
        "max_tokens=1 should produce at most 1 token, got {}",
        tokens.len()
    );
}

#[test]
fn generate_multiple_tokens_respects_max() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&default_prompt(), 5)
        .expect("generate with max_tokens=5 should succeed");
    assert!(
        tokens.len() <= 5,
        "max_tokens=5 should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

#[test]
fn generate_greedy_is_deterministic() {
    let prompt = default_prompt();

    let mut engine1 = tiny_engine(greedy_params(), 42);
    let out1 = engine1
        .generate(&prompt, 10)
        .expect("first greedy generate should succeed");

    let mut engine2 = tiny_engine(greedy_params(), 42);
    let out2 = engine2
        .generate(&prompt, 10)
        .expect("second greedy generate should succeed");

    assert_eq!(
        out1, out2,
        "greedy decoding with same seed must produce identical output"
    );
}

#[test]
fn generate_greedy_deterministic_three_runs() {
    let prompt = default_prompt();
    let mut results = Vec::new();

    for _ in 0..3 {
        let mut engine = tiny_engine(greedy_params(), 99);
        let tokens = engine
            .generate(&prompt, 8)
            .expect("greedy generate should succeed");
        results.push(tokens);
    }

    assert_eq!(
        results[0], results[1],
        "run 0 and run 1 must match for greedy"
    );
    assert_eq!(
        results[1], results[2],
        "run 1 and run 2 must match for greedy"
    );
}

#[test]
fn generate_different_seeds_can_differ() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine1 = tiny_engine(params.clone(), 1);
    let out1 = engine1
        .generate(&prompt, 20)
        .expect("generate with seed 1 should succeed");

    let mut engine2 = tiny_engine(params, 9999);
    let out2 = engine2
        .generate(&prompt, 20)
        .expect("generate with seed 9999 should succeed");

    // With high temperature and different seeds, outputs should differ
    // (not guaranteed but overwhelmingly likely with 20 tokens)
    let differ = out1 != out2;
    assert!(
        differ,
        "different seeds with temp=1.0 should produce different outputs \
         (this can fail with astronomically low probability)"
    );
}

#[test]
fn generate_empty_prompt_returns_empty() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&[], 10)
        .expect("empty prompt generate should succeed");
    assert!(
        tokens.is_empty(),
        "empty prompt should produce no output, got {} tokens",
        tokens.len()
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Sampling parameter effects
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn temperature_zero_is_greedy() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut results = Vec::new();
    for seed in [1, 42, 12345] {
        let mut engine = tiny_engine(params.clone(), seed);
        let tokens = engine
            .generate(&prompt, 5)
            .expect("greedy generate should succeed");
        results.push(tokens);
    }

    // Temperature 0 is pure argmax -- seed should not matter
    assert_eq!(
        results[0], results[1],
        "temperature=0 should be deterministic regardless of seed"
    );
    assert_eq!(
        results[1], results[2],
        "temperature=0 should be deterministic regardless of seed"
    );
}

#[test]
fn high_temperature_produces_valid_tokens() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 2.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 10)
        .expect("high temperature generate should succeed");

    // Should still produce valid output
    assert!(
        !tokens.is_empty(),
        "high temperature should still generate tokens"
    );
    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

#[test]
fn top_k_1_is_deterministic() {
    let prompt = default_prompt();

    let top_k_1 = SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    // top_k=1 always picks the highest-logit token, so it should be
    // deterministic regardless of seed (only one candidate remains).
    let mut engine1 = tiny_engine(top_k_1.clone(), 42);
    let out1 = engine1
        .generate(&prompt, 8)
        .expect("top_k=1 generate should succeed");

    let mut engine2 = tiny_engine(top_k_1, 9999);
    let out2 = engine2
        .generate(&prompt, 8)
        .expect("top_k=1 generate with different seed should succeed");

    assert_eq!(
        out1, out2,
        "top_k=1 should produce the same result regardless of seed"
    );
}

#[test]
fn top_k_full_vocab_no_filtering() {
    let prompt = default_prompt();
    let vocab_size = Qwen3Config::tiny_test().vocab_size;

    let params = SamplingParams {
        temperature: 0.7,
        top_k: vocab_size,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 5)
        .expect("top_k=vocab_size generate should succeed");

    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

#[test]
fn top_p_near_zero_produces_valid_output() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.01,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    // Very small top_p restricts the candidate set heavily.
    // With the tiny test model (random weights), the logit distribution
    // may still have multiple tokens in the nucleus, so we just verify
    // that valid tokens are produced.
    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 5)
        .expect("top_p=0.01 should succeed");

    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

#[test]
fn top_p_1_allows_all_tokens() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 10)
        .expect("top_p=1.0 generate should succeed");

    assert!(!tokens.is_empty(), "top_p=1.0 should generate tokens");
}

#[test]
fn repetition_penalty_above_one_produces_valid_output() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.5,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&prompt, 10)
        .expect("repetition penalty generate should succeed");

    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

// ── T-09 / FIX3-PARITY item 4: `repetition_penalty_vs_no_penalty_can_differ`
//
// That test asserted nothing. It ended in `let _ = (out_no, out_with);` under
// the comment "We just verify both succeed", while its NAME claimed a
// behaviour it never checked — verbatim the defect T-09 is about.
//
// Its comment was also wrong about the mechanism: "repetition penalty in the
// basic Sampler does NOT track previous tokens, it's only applied if the
// engine feeds them back. With the basic generate() pipeline, the effect may
// be zero." The history IS fed — `InferenceEngine::generate` samples through
// `sampler.sample_with_history(&last_logits, &output_tokens)` (`engine.rs`),
// and `Sampler::sample_with_history` calls `apply_repetition_penalty` whenever
// `repetition_penalty != 1.0` (`sampling.rs`).
//
// What is actually true here was MEASURED while writing these two tests:
// `Qwen3Config::tiny_test()`'s forward produces an ALL-ZERO logit vector. At
// `temperature = 0.7` that is a uniform multinomial over 151 936 tokens; at
// `temperature = 0.0` the argmax is token 0 forever. And
// `apply_repetition_penalty` is `logit >= 0.0 ? logit / penalty : logit *
// penalty` (`sampling_advanced.rs:241`), which maps `0.0 -> 0.0`: on an
// all-zero logit vector the penalty is the identity, so NO setting of
// `repetition_penalty` can change what `generate()` returns on this fixture.
//
// So the old test's name could not be made true through `generate()` at all.
// It is replaced by two tests that each assert what their name says: one
// pinning the real, documented penalty behaviour on a non-degenerate logit
// vector, and one pinning the fixture property that makes the pipeline-level
// comparison inert.

/// The documented behaviour, asserted deterministically at the exact API
/// `InferenceEngine::generate` decodes through: a repeated token's logit is
/// penalized, and at `temperature = 0` that moves the argmax.
///
/// This is the control for
/// [`repetition_penalty_cannot_change_generate_output_on_the_tiny_test_fixture`]
/// below: it is what proves that test pins a property of the FIXTURE and not
/// a broken penalty path. A regression that dropped `sample_with_history`'s
/// history argument, or reverted the penalty to a no-op, fails here.
#[test]
fn repetition_penalty_moves_the_greedy_pick_on_non_degenerate_logits() {
    let penalized = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 2.0,
        max_tokens: 128,
    };
    // Deliberately not all-equal, unlike `tiny_test()`'s logits.
    let logits = [4.0f32, 3.0, 2.0, 1.0];

    let mut sampler = Sampler::new(penalized.clone(), 42);
    assert_eq!(
        sampler
            .sample_with_history(&logits, &[])
            .expect("sample with empty history"),
        0,
        "with an empty history the penalty cannot apply: argmax of {logits:?} is 0"
    );
    assert_eq!(
        sampler
            .sample_with_history(&logits, &[0])
            .expect("sample with token 0 in history"),
        1,
        "token 0 is in the history, so its logit becomes 4.0 / 2.0 = 2.0 and \
         loses to token 1's untouched 3.0"
    );

    let mut unpenalized = Sampler::new(
        SamplingParams {
            repetition_penalty: 1.0,
            ..penalized
        },
        42,
    );
    assert_eq!(
        unpenalized
            .sample_with_history(&logits, &[0])
            .expect("sample with penalty 1.0"),
        0,
        "repetition_penalty = 1.0 must leave the same history without effect"
    );
}

/// The pipeline-level fact, with the reason its name now states: on
/// `Qwen3Config::tiny_test()` the repetition penalty CANNOT change what
/// `generate()` returns, because the fixture's logits are all zero and the
/// penalty maps `0.0 -> 0.0`.
///
/// This is a tripwire, not a wish: if `tiny_test()` ever grows weights that
/// produce a non-degenerate logit vector, this test fails, and the right
/// response is to turn it back into the "the outputs differ" assertion the
/// old test's name promised — with
/// [`repetition_penalty_moves_the_greedy_pick_on_non_degenerate_logits`]
/// above as the proof that the penalty path itself works.
#[test]
fn repetition_penalty_cannot_change_generate_output_on_the_tiny_test_fixture() {
    let prompt = default_prompt();

    let no_penalty = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let with_penalty = SamplingParams {
        repetition_penalty: 2.0,
        ..no_penalty.clone()
    };

    let mut engine_no = tiny_engine(no_penalty, 42);
    let out_no = engine_no
        .generate(&prompt, 20)
        .expect("no penalty generate should succeed");

    let mut engine_with = tiny_engine(with_penalty, 42);
    let out_with = engine_with
        .generate(&prompt, 20)
        .expect("with penalty generate should succeed");

    assert!(
        !out_no.is_empty(),
        "the unpenalized run must generate something to compare"
    );
    assert!(
        out_no.iter().all(|&t| t == out_no[0]),
        "premise: tiny_test()'s logits are all equal, so greedy decode repeats \
         one token forever — if this fails the fixture has changed and this \
         test must be re-derived (see its doc comment); got {out_no:?}"
    );
    assert_eq!(
        out_no, out_with,
        "on an all-zero logit vector apply_repetition_penalty is the identity \
         (0.0 / penalty == 0.0), so repetition_penalty = 2.0 must be \
         indistinguishable from 1.0 through generate() on this fixture"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn generate_max_tokens_zero_returns_empty() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&default_prompt(), 0)
        .expect("max_tokens=0 should succeed");
    assert!(
        tokens.is_empty(),
        "max_tokens=0 should produce no output, got {} tokens",
        tokens.len()
    );
}

#[test]
fn generate_single_token_prompt() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let tokens = engine
        .generate(&[151644], 5)
        .expect("single token prompt should succeed");
    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

#[test]
fn generate_with_various_prompt_lengths() {
    for len in [1, 2, 5, 10, 50] {
        let prompt: Vec<u32> = (0..len).map(|i| (i % 1000) as u32).collect();
        let mut engine = tiny_engine(greedy_params(), 42);
        let tokens = engine
            .generate(&prompt, 3)
            .unwrap_or_else(|e| panic!("generate with prompt length {len} should succeed: {e}"));
        assert!(
            tokens.len() <= 3,
            "prompt_len={len}: should produce at most 3 tokens, got {}",
            tokens.len()
        );
    }
}

#[test]
fn generate_long_prompt_near_context_limit() {
    let config = Qwen3Config::tiny_test();
    // Use a 64-token prompt to verify that long prompts are handled without
    // panicking and respect the max_tokens limit.
    let prompt: Vec<u32> = (0..64).map(|i| (i % 1000) as u32).collect();
    let mut engine = InferenceEngine::new(config, greedy_params(), 42);
    let tokens = engine
        .generate(&prompt, 5)
        .expect("long prompt should succeed");
    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Engine state
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn sequential_generates_independent() {
    let prompt = default_prompt();
    let mut engine = tiny_engine(greedy_params(), 42);

    let out1 = engine
        .generate(&prompt, 5)
        .expect("first generate should succeed");

    // Reset and generate again with same params
    engine.reset();
    let mut engine2 = tiny_engine(greedy_params(), 42);
    let out2 = engine2
        .generate(&prompt, 5)
        .expect("second generate on fresh engine should succeed");

    assert_eq!(
        out1, out2,
        "greedy generation after reset should match a fresh engine"
    );
}

#[test]
fn multiple_generates_without_reset_succeed() {
    let mut engine = tiny_engine(greedy_params(), 42);

    for i in 0..5 {
        let prompt = vec![151644, (i * 100 + 1) as u32];
        let tokens = engine
            .generate(&prompt, 3)
            .unwrap_or_else(|e| panic!("generate iteration {i} should succeed: {e}"));
        assert!(
            tokens.len() <= 3,
            "iteration {i}: should produce at most 3 tokens"
        );
    }
}

#[test]
fn engine_stats_update_after_generate() {
    let mut engine = tiny_engine(greedy_params(), 42);
    assert_eq!(engine.stats().requests_completed(), 0);

    let _ = engine
        .generate(&default_prompt(), 5)
        .expect("generate should succeed");
    assert!(
        engine.stats().requests_completed() >= 1,
        "stats should reflect completed request"
    );
}

#[test]
fn engine_stats_accumulate_tokens() {
    let mut engine = tiny_engine(greedy_params(), 42);

    let out1 = engine
        .generate(&default_prompt(), 3)
        .expect("first generate should succeed");
    let count1 = engine.stats().tokens_generated();

    engine.reset();
    let out2 = engine
        .generate(&default_prompt(), 3)
        .expect("second generate should succeed");
    let count2 = engine.stats().tokens_generated();

    assert_eq!(
        count2,
        count1 + out2.len() as u64,
        "token count should accumulate: first={}, second output={}, total={}",
        out1.len(),
        out2.len(),
        count2,
    );
}

#[test]
fn generate_with_seed_method() {
    let prompt = default_prompt();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };

    let mut engine = tiny_engine(params.clone(), 42);
    let out1 = engine
        .generate_with_seed(&prompt, 5, 100, &params)
        .expect("generate_with_seed should succeed");

    engine.reset();
    let out2 = engine
        .generate_with_seed(&prompt, 5, 100, &params)
        .expect("generate_with_seed with same seed should succeed");

    assert_eq!(
        out1, out2,
        "generate_with_seed with identical seeds must produce identical output"
    );
}

#[test]
fn generate_tokens_within_vocab_range() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 50,
        top_p: 0.95,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut engine = tiny_engine(params, 42);
    let tokens = engine
        .generate(&default_prompt(), 20)
        .expect("generate should succeed");

    let vocab_size = Qwen3Config::tiny_test().vocab_size;
    for &t in &tokens {
        assert!(
            (t as usize) < vocab_size,
            "token {} exceeds vocab size {}",
            t,
            vocab_size
        );
    }
}

#[test]
fn batch_generate_returns_correct_count() {
    let mut engine = tiny_engine(greedy_params(), 42);
    let prompts = vec![vec![151644, 100], vec![151644, 200], vec![151644, 300]];
    let results = engine.batch_generate(&prompts, 3);
    assert_eq!(
        results.len(),
        3,
        "batch_generate should return one result per prompt"
    );
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.is_ok(),
            "batch_generate prompt {i} should succeed: {:?}",
            r.as_ref().err()
        );
    }
}

#[test]
fn generate_with_default_params() {
    let mut engine = tiny_engine(SamplingParams::default(), 42);
    let tokens = engine
        .generate(&default_prompt(), 5)
        .expect("generate with default params should succeed");
    assert!(
        tokens.len() <= 5,
        "should produce at most 5 tokens, got {}",
        tokens.len()
    );
}

// ══════════════════════════════════════════════════════════════════════════
// InferencePipeline regression tests (RT-01 / RT-02 / RT-20 / RT-33)
//
// Unlike the `InferenceEngine::generate` tests above, these drive the
// higher-level `oxibonsai_runtime::pipeline::InferencePipeline` -- the
// public builder API that composes sampling, stop sequences, constraints
// and beam search. Each block reproduces one previously-broken behaviour
// end to end through that public API.
// ══════════════════════════════════════════════════════════════════════════

mod inference_pipeline_regressions {
    use super::{default_prompt, greedy_params, tiny_engine};
    use oxibonsai_runtime::constrained_decoding::AllowListConstraint;
    use oxibonsai_runtime::pipeline::{PipelineBuilder, StopReason};
    use oxibonsai_runtime::sampling_advanced::{SamplerChain, SamplerStep};

    // ── RT-01: real decoded text, not token-id decimals ─────────────────

    /// Deterministic, easily-inverted stand-in for a real tokenizer: token
    /// id `n` decodes to `"w<n>"`, pieces joined with `-`.
    fn word_detokenizer(ids: &[u32]) -> oxibonsai_runtime::error::RuntimeResult<String> {
        Ok(ids
            .iter()
            .map(|id| format!("w{id}"))
            .collect::<Vec<_>>()
            .join("-"))
    }

    #[test]
    fn pipeline_text_round_trips_through_detokenizer() {
        let mut engine = tiny_engine(greedy_params(), 42);
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(5)
            .greedy()
            .with_detokenizer(word_detokenizer)
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert!(
            !output.token_ids.is_empty(),
            "expected at least one generated token"
        );
        assert!(
            output.text_available,
            "a detokenizer was attached; text_available must be true"
        );
        let expected = word_detokenizer(&output.token_ids).expect("must decode");
        assert_eq!(
            output.text, expected,
            "text must decode the exact token ids the pipeline reports"
        );
        // The historic bug: `text` was a space-separated dump of the ids'
        // own decimal digits (e.g. "4521 892 17"). Confirm the output is
        // nowhere near that shape.
        assert!(
            !output.text.chars().all(|c| c.is_ascii_digit() || c == ' '),
            "text must not be the old digits-and-spaces token-id dump: {:?}",
            output.text
        );
    }

    #[test]
    fn pipeline_text_empty_without_detokenizer() {
        let mut engine = tiny_engine(greedy_params(), 42);
        let mut pipeline = PipelineBuilder::new().max_tokens(5).greedy().build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert!(
            !output.token_ids.is_empty(),
            "tokens should still be generated without a detokenizer"
        );
        assert!(
            !output.text_available,
            "no detokenizer attached: text_available must be false"
        );
        assert!(
            output.text.is_empty(),
            "no detokenizer attached: text must be empty, never a token-id dump; got {:?}",
            output.text
        );
    }

    // ── RT-02: stop sequences match real decoded text ───────────────────

    /// Decodes token 99 -> "hello-", 10 -> "<|en", 20 -> "d|>", 40 -> "-tail",
    /// anything else -> "X". Neither token 10 nor token 20 alone contains
    /// the target stop sequence "<|end|>" -- only their concatenation does,
    /// which is the whole point of matching on decoded text instead of
    /// per-token (or worse, per-token-id-decimal) substrings.
    fn split_stop_detokenizer(ids: &[u32]) -> oxibonsai_runtime::error::RuntimeResult<String> {
        Ok(ids
            .iter()
            .map(|&id| match id {
                99 => "hello-",
                10 => "<|en",
                20 => "d|>",
                40 => "-tail",
                _ => "X",
            })
            .collect())
    }

    #[test]
    fn pipeline_stop_sequence_detected_when_split_across_tokens() {
        // Force the exact sequence [99, 10, 20, 40]; the stop sequence only
        // becomes complete once both 10 and 20 have been decoded.
        let candidate = vec![99u32, 10, 20, 40];
        let mut engine = tiny_engine(greedy_params(), 42);
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(4)
            .greedy()
            .with_constraint(Box::new(AllowListConstraint::new(vec![candidate])))
            .stop_on(vec!["<|end|>".to_string()])
            .with_detokenizer(split_stop_detokenizer)
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert_eq!(
            output.stop_reason,
            StopReason::StopSequence("<|end|>".to_string()),
            "a stop sequence split across two tokens must still be detected; got {:?} \
             (token_ids: {:?})",
            output.stop_reason,
            output.token_ids
        );
        // Token 99 ("hello-") precedes the match and must survive; tokens 10
        // and 20 (which together form the match) must not.
        assert_eq!(
            output.token_ids,
            vec![99u32],
            "only the pre-match token should remain"
        );
        assert_eq!(output.text, "hello-");
        assert!(
            !output.text.contains("<|end|>"),
            "the stop sequence itself must never appear in the final text"
        );
    }

    #[test]
    fn pipeline_stop_sequence_does_not_misfire_on_digit_coincidence() {
        // Historic bug: a stop sequence of "42" fired on ANY token whose
        // decimal id happened to contain "42" (e.g. id 42, 420, 1425...).
        // With a real detokenizer attached, token id 42 decoding to a word
        // that contains no "42" must never trigger `stop_on(["42"])`.
        fn detok(ids: &[u32]) -> oxibonsai_runtime::error::RuntimeResult<String> {
            Ok(ids.iter().map(|_| "safe-word-").collect())
        }

        let candidate = vec![42u32, 42, 42, 42];
        let mut engine = tiny_engine(greedy_params(), 42);
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(4)
            .greedy()
            .with_constraint(Box::new(AllowListConstraint::new(vec![candidate])))
            .stop_on(vec!["42".to_string()])
            .with_detokenizer(detok)
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert!(
            !matches!(output.stop_reason, StopReason::StopSequence(_)),
            "token id 42 must not trigger stop_on([\"42\"]) when its decoded \
             text contains no \"42\"; got {:?}",
            output.stop_reason
        );
        assert_eq!(output.token_ids, vec![42u32, 42, 42, 42]);
        assert!(!output.text.contains("42"));
    }

    #[test]
    fn pipeline_stop_sequences_skipped_without_detokenizer() {
        // Without a detokenizer, stop sequences cannot be matched against
        // real text; generation must proceed as if none were configured
        // rather than resurrecting the old (never-correct) token-id-decimal
        // match.
        let mut engine = tiny_engine(greedy_params(), 42);
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(4)
            .greedy()
            .stop_on(vec!["anything".to_string()])
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert_eq!(output.token_ids.len(), 4, "must run to max_tokens");
        assert_eq!(output.stop_reason, StopReason::MaxTokens);
    }

    // ── RT-20: constraint masking never produces NaN / a disallowed token ──

    #[test]
    fn pipeline_fully_constrained_vocab_samples_single_token_greedy() {
        let only_token = 7u32;
        let mut engine = tiny_engine(greedy_params(), 42);
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(6)
            .greedy()
            .with_constraint(Box::new(AllowListConstraint::new(vec![vec![
                only_token;
                6
            ]])))
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert!(
            !output.token_ids.is_empty(),
            "must generate at least one token"
        );
        assert!(
            output.token_ids.iter().all(|&t| t == only_token),
            "every emitted token must be the sole constraint-allowed one, got {:?}",
            output.token_ids
        );
        assert!(
            !matches!(output.stop_reason, StopReason::Error(_)),
            "a fully-constrained run must terminate cleanly, not error out \
             (e.g. from NaN-driven sampling); got {:?}",
            output.stop_reason
        );
    }

    #[test]
    fn pipeline_fully_constrained_vocab_samples_single_token_high_temperature() {
        // High temperature plus a real sampler chain (not greedy argmax) is
        // exactly the regime RT-20's old `-1e9` sentinel was flagged for;
        // `f32::NEG_INFINITY` must hold exactly regardless of scaling.
        let only_token = 11u32;
        let mut engine = tiny_engine(greedy_params(), 42);
        let chain = SamplerChain::new(123).add(SamplerStep::Temperature(2.0));
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(6)
            .with_sampling(chain)
            .with_constraint(Box::new(AllowListConstraint::new(vec![vec![
                only_token;
                6
            ]])))
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert!(
            !output.token_ids.is_empty(),
            "must generate at least one token"
        );
        assert!(
            output.token_ids.iter().all(|&t| t == only_token),
            "sampling at temperature=2.0 must never emit a disallowed token, got {:?}",
            output.token_ids
        );
        assert!(
            !matches!(output.stop_reason, StopReason::Error(_)),
            "a fully-constrained sampling run must not error out, got {:?}",
            output.stop_reason
        );
    }

    /// A constraint that allows **no** token at all -- the degenerate case
    /// `f32::NEG_INFINITY` masking (RT-20) must handle without ever handing
    /// the sampler an all-`-inf` vector, which would otherwise hit the
    /// exact NaN-driven arbitrary-token path RT-21 describes (in a
    /// different, unowned file: `sampling.rs`, and equivalently in
    /// `sampling_advanced.rs`'s `softmax_inplace`/`categorical_sample`).
    struct NoLegalToken;
    impl oxibonsai_runtime::constrained_decoding::TokenConstraint for NoLegalToken {
        fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
            Some(vec![false; vocab_size])
        }
        fn advance(&mut self, _token: u32) -> bool {
            true
        }
        fn is_complete(&self) -> bool {
            false
        }
        fn reset(&mut self) {}
        fn name(&self) -> &str {
            "NoLegalToken"
        }
    }

    #[test]
    fn pipeline_all_disallowed_mask_stops_cleanly_without_nan() {
        let mut engine = tiny_engine(greedy_params(), 42);
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(4)
            .greedy()
            .with_constraint(Box::new(NoLegalToken))
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert!(
            output.token_ids.is_empty(),
            "no token can legally be emitted, so none should be, got {:?}",
            output.token_ids
        );
        assert_eq!(
            output.stop_reason,
            StopReason::ConstraintUnsatisfiable,
            "an all-disallowed mask must stop the run honestly (and be reported \
             distinctly from a constraint that completed normally), not error or hang"
        );
    }

    #[test]
    fn pipeline_all_disallowed_mask_stops_cleanly_with_sampling_chain() {
        // Same scenario, but through the `SamplerChain` path (softmax +
        // categorical sampling) rather than greedy argmax -- the path that
        // would actually divide by a NaN sum if an all-`-inf` vector ever
        // reached it.
        let mut engine = tiny_engine(greedy_params(), 42);
        let chain = SamplerChain::new(7).add(SamplerStep::Temperature(1.0));
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(4)
            .with_sampling(chain)
            .with_constraint(Box::new(NoLegalToken))
            .build();

        let output = pipeline.run(default_prompt(), &mut engine);

        assert!(output.token_ids.is_empty());
        assert_eq!(output.stop_reason, StopReason::ConstraintUnsatisfiable);
    }

    // ── RT-33: beam search reuses the shared KV cache across beams ──────

    #[test]
    fn pipeline_beam_search_avoids_quadratic_reprefill() {
        use oxibonsai_runtime::beam_search::BeamSearchConfig;

        let mut engine = tiny_engine(greedy_params(), 42);
        let prompt = vec![1u32; 40]; // a nontrivial prompt length L
        let beam_width = 4u32;
        let max_tokens = 6u32;

        let cfg = BeamSearchConfig {
            beam_width: beam_width as usize,
            max_tokens: max_tokens as usize,
            eos_token_id: 999_999, // unreachable, so the full budget runs
            early_stopping: false,
            ..Default::default()
        };
        let mut pipeline = PipelineBuilder::new().with_beam_search(cfg).build();

        let output = pipeline.run(prompt.clone(), &mut engine);
        assert!(
            !output.token_ids.is_empty(),
            "beam search must still produce tokens"
        );

        let prompt_len = prompt.len() as u64;
        let naive_bound = u64::from(beam_width) * u64::from(max_tokens) * prompt_len;
        let actual = engine.prefill_token_count();
        assert!(
            actual < naive_bound,
            "expected far fewer than the naive O(beams*steps*prompt_len) \
             token-forwards ({naive_bound}), got {actual} -- beam search must \
             reuse the shared KV cache across beams (RT-33)"
        );
        // A generous but still meaningful bound: with KV reuse, each beam
        // call should cost roughly O(1) extra tokens once diverged from the
        // previously evaluated beam, not O(prompt_len) every time.
        let generous_bound = prompt_len + u64::from(beam_width) * u64::from(max_tokens) * 4;
        assert!(
            actual <= generous_bound,
            "prefill token count {actual} exceeds the generous KV-reuse bound {generous_bound}"
        );
        // NOTE: `Qwen3Config::tiny_test()` has zero transformer blocks and
        // zero weights, so every forward pass here yields an all-zero logit
        // vector -- this test can only ever assert on *how many* forward
        // passes ran (`prefill_token_count`), never on which tokens came
        // out, so it cannot distinguish correct KV-cache reuse from a
        // corrupted one. See `rt33_beam_search_kv_reuse_correctness` below
        // for the companion correctness guard on a real (non-zero-weight)
        // forward pass.
    }
}

// ══════════════════════════════════════════════════════════════════════════
// RT-33 correctness guard: beam-search KV-reuse vs. naive full reprefill
//
// `pipeline_beam_search_avoids_quadratic_reprefill` above only asserts on
// `engine.prefill_token_count()`, which is vacuous for correctness: on
// `Qwen3Config::tiny_test()` every forward pass yields an all-zero logit
// vector (zero transformer blocks, zero weights), so *any* token sequence
// at all would satisfy it -- it cannot tell a correct KV-cache rewind from
// a corrupted one.
//
// This module drives a small but non-degenerate synthetic ternary model
// (real, varied weights -- the same known-good fixture already validated
// end to end by `cross_backend_determinism_tests.rs` /
// `metal_greedy_cpu_fallback_tests.rs` against the README's CPU/Metal
// determinism promise) through both `InferencePipeline::run`'s optimized
// beam search (which rewinds and reuses the shared KV cache across beams --
// see `InferencePipeline::run_beam_search`'s module docs for the RT-33
// cost argument) and a naive baseline that resets and re-prefills each
// beam's *entire* history from scratch on every single `get_logits` call --
// exactly the quadratic behaviour RT-33 optimized away. Both engines run on
// `KernelTier::Reference` (plain scalar CPU), so the comparison is never
// confused by GPU-specific numerics, and the fixture is reproduced verbatim
// here (rather than shared) so this module has no dependency on the
// `metal` feature or `target_os = "macos"` those other two files gate on.
// ══════════════════════════════════════════════════════════════════════════

mod rt33_beam_search_kv_reuse_correctness {
    use super::greedy_params;
    use half::f16;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
    use oxibonsai_kernels::dispatch::KernelTier;
    use oxibonsai_model::model::BonsaiModel;
    use oxibonsai_runtime::beam_search::{BeamSearchConfig, BeamSearchEngine};
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::pipeline::PipelineBuilder;
    use oxibonsai_testkit::gguf_fixture::Lcg;

    /// KV-cache / context budget for the synthetic model.
    const MAX_SEQ: usize = 512;

    // ─────────────────────────────────────────────────────────────────────
    // Synthetic ternary fixture (h=128, inter=256, 2 layers, vocab=32, all
    // projections + LM head `TQ2_0_g128`) -- copied verbatim from
    // `cross_backend_determinism_tests.rs`, which is itself copied verbatim
    // from `oxibonsai-model/tests/metal_prefill_ternary_parity_tests.rs`,
    // so this stays the one known-good fixture rather than growing a fourth
    // slightly-different copy.
    // ─────────────────────────────────────────────────────────────────────

    /// Build a TQ2_0_g128 weight blob with deterministic per-block patterns.
    ///
    /// Each block is 34 bytes: 32 bytes of 2-bit codes (4 weights/byte,
    /// LSB-first) followed by a 2-byte FP16 scale
    /// (`00→-1, 01→0, 10→+1`).
    ///
    /// CQ-14 (wave-2.5 deviation routing #7): `11` (`0b11`) is a *reserved*
    /// code, not a fourth value — `screen_ternary_codes` in
    /// `oxibonsai-model/src/weight_loaders.rs` now rejects it outright, so
    /// this fixture must never emit it. A raw `(state >> 33) as u8` byte
    /// (the previous body) lands on `0b11` in about a quarter of *lanes*;
    /// each lane is folded into `{0, 1, 2}` before packing instead, exactly
    /// as `crates/oxibonsai-model/src/model/types/gpu_cache.rs::tq2_pattern`
    /// already does.
    ///
    /// T-07 FIX (verifier wave 3): re-pointed at
    /// `oxibonsai_testkit::gguf_fixture::Lcg::next_valid_tq2_byte`,
    /// byte-for-byte identical to the previous hand-rolled state machine
    /// (`Lcg::new(s)` stores `s` as its state directly, so pre-adding the
    /// same golden-ratio constant this file always added before its first
    /// `next_u64()` reproduces the exact sequence) — confirmed by re-running
    /// this file's tests unchanged.
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
            let scale_f32 =
                0.25_f32 + ((lcg.next_u64() >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
            data.extend_from_slice(&f16::from_f32(scale_f32).to_le_bytes());
        }
        data
    }

    /// Build an FP32 tensor whose values vary with the index.
    fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            let phase = (i as f32) * 0.013_f32;
            let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
            v.extend_from_slice(&val.to_le_bytes());
        }
        v
    }

    /// Build a synthetic fully-ternary GGUF for `BonsaiModel::from_gguf`.
    fn build_synthetic_ternary_gguf() -> Vec<u8> {
        let h: usize = 128;
        let inter: usize = 256;
        let num_layers: usize = 2;
        let nq: usize = 4;
        let nkv: usize = 2;
        let hd: usize = 32;
        let vocab: usize = 32;

        let mut writer = GgufWriter::new();

        writer.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen3".to_string()),
        );
        writer.add_metadata(
            "general.name",
            MetadataWriteValue::Str("Rt33BeamSearchCorrectnessTest".to_string()),
        );
        writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
        writer.add_metadata(
            "qwen3.block_count",
            MetadataWriteValue::U32(num_layers as u32),
        );
        writer.add_metadata(
            "qwen3.attention.head_count",
            MetadataWriteValue::U32(nq as u32),
        );
        writer.add_metadata(
            "qwen3.attention.head_count_kv",
            MetadataWriteValue::U32(nkv as u32),
        );
        writer.add_metadata(
            "qwen3.feed_forward_length",
            MetadataWriteValue::U32(inter as u32),
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
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(vocab * h, 0.5),
        });
        writer.add_tensor(TensorEntry {
            name: "output_norm.weight".to_string(),
            shape: vec![h as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(h, 1.0),
        });
        writer.add_tensor(TensorEntry {
            name: "output.weight".to_string(),
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(vocab * h, 0xCAFE_BABE),
        });

        for layer in 0..num_layers {
            let pfx = format!("blk.{layer}");

            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.attn_norm.weight"),
                shape: vec![h as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(h, 1.0),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.ffn_norm.weight"),
                shape: vec![h as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(h, 1.0),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.attn_q_norm.weight"),
                shape: vec![hd as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(hd, 1.0),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.attn_k_norm.weight"),
                shape: vec![hd as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(hd, 1.0),
            });

            let layer_seed = 0x2000_0000_u64.wrapping_add((layer as u64) << 16);
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.attn_q.weight"),
                shape: vec![h as u64, (nq * hd) as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_0_g128_pattern(nq * hd * h, layer_seed),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.attn_k.weight"),
                shape: vec![h as u64, (nkv * hd) as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_0_g128_pattern(nkv * hd * h, layer_seed.wrapping_add(1)),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.attn_v.weight"),
                shape: vec![h as u64, (nkv * hd) as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_0_g128_pattern(nkv * hd * h, layer_seed.wrapping_add(2)),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.attn_output.weight"),
                shape: vec![(nq * hd) as u64, h as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_0_g128_pattern(h * nq * hd, layer_seed.wrapping_add(3)),
            });

            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.ffn_gate.weight"),
                shape: vec![h as u64, inter as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_0_g128_pattern(inter * h, layer_seed.wrapping_add(4)),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.ffn_up.weight"),
                shape: vec![h as u64, inter as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_0_g128_pattern(inter * h, layer_seed.wrapping_add(5)),
            });
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.ffn_down.weight"),
                shape: vec![inter as u64, h as u64],
                tensor_type: TensorType::TQ2_0_g128,
                data: tq2_0_g128_pattern(h * inter, layer_seed.wrapping_add(6)),
            });
        }

        writer.to_bytes().expect("GgufWriter::to_bytes")
    }

    /// The RT-33 correctness guard: `InferencePipeline`'s beam search (which
    /// shares one KV cache across beams, rewinding to the longest common
    /// prefix before each forward pass) must produce byte-identical tokens
    /// to a naive baseline that resets and fully re-prefills each beam's
    /// entire history from position 0 on *every* `get_logits` call. Both
    /// sides use the same synthetic model, config, and prompt; the only
    /// difference is how much of the KV cache each call reuses -- if the
    /// pipeline's cache-rewind logic were silently corrupting a beam's
    /// context (rather than merely avoiding redundant work), this test
    /// would catch it as a token divergence, not just a slower run.
    ///
    /// Each side parses its own fresh `GgufFile`/`BonsaiModel`/
    /// `InferenceEngine` from the same `gguf_bytes` inside its own block
    /// scope (mirroring `cross_backend_determinism_tests.rs`'s `run()`
    /// helper) so nothing borrowed from `gguf_bytes` needs to be named
    /// across a function boundary; only the resulting owned `Vec<u32>`
    /// escapes each block.
    ///
    /// Not vacuous: this fixture/prompt/`beam_width` combination was checked
    /// (by temporarily logging every distinct `beam_tokens` history passed
    /// to the naive side's `get_logits`) to explore ~35 distinct, genuinely
    /// diverging beam histories per run (e.g. sibling branches like
    /// `[19, 25, 19, 25]` vs. `[19, 25, 25]`, or independent single-token
    /// branches `[22]` / `[25]` / `[31]`) -- so `common_prefix_len` is
    /// exercised well below "always the full previous length" on both
    /// sides, not just re-confirming a single deterministic path. The final
    /// *winning* sequence happening to converge on one repeated token
    /// (a real, length-penalty-driven property of this fixture's weights)
    /// does not make the search itself trivial.
    #[test]
    fn pipeline_beam_search_matches_naive_full_reprefill_baseline() {
        let gguf_bytes = build_synthetic_ternary_gguf();
        let prompt: Vec<u32> = vec![5, 12, 2, 27, 8, 19, 0, 31, 14, 22, 6, 29];
        let beam_width = 6;
        let max_tokens = 8;
        // Unreachable within vocab=32 so the full token budget always runs
        // on both sides (mirrors the existing RT-33 cost test above).
        let eos_token_id = 999_999;

        let pipeline_generated = {
            let gguf =
                GgufFile::parse(&gguf_bytes).expect("GgufFile::parse synthetic (pipeline side)");
            let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ)
                .expect("BonsaiModel::from_gguf (pipeline side)");
            let mut engine = InferenceEngine::from_model_with_tier(
                model,
                KernelTier::Reference,
                greedy_params(),
                42,
            );
            let cfg = BeamSearchConfig {
                beam_width,
                max_tokens,
                eos_token_id,
                early_stopping: false,
                ..Default::default()
            };
            let mut pipeline = PipelineBuilder::new().with_beam_search(cfg).build();
            pipeline.run(prompt.clone(), &mut engine).token_ids
        };

        let naive_generated = {
            let gguf =
                GgufFile::parse(&gguf_bytes).expect("GgufFile::parse synthetic (naive side)");
            let model = BonsaiModel::from_gguf(&gguf, MAX_SEQ)
                .expect("BonsaiModel::from_gguf (naive side)");
            let mut engine = InferenceEngine::from_model_with_tier(
                model,
                KernelTier::Reference,
                greedy_params(),
                42,
            );
            let vocab_size = engine.vocab_size();
            let cfg = BeamSearchConfig {
                beam_width,
                max_tokens,
                eos_token_id,
                early_stopping: false,
                ..Default::default()
            };
            let beam_engine = BeamSearchEngine::new(cfg);
            let result = beam_engine.search(prompt.clone(), vocab_size, |beam_tokens, _step| {
                // The exact O(beams*steps*len) baseline `run_beam_search`
                // optimizes away: forget the cache and re-prefill this
                // beam's *entire* history from scratch on every call,
                // independent of what any previous call already computed.
                engine.reset();
                engine
                    .prefill_from_pos(beam_tokens, 0)
                    .expect("naive baseline prefill_from_pos failed")
            });
            let best = result.best();
            if best.len() > prompt.len() {
                best[prompt.len()..].to_vec()
            } else {
                Vec::new()
            }
        };

        assert!(
            !pipeline_generated.is_empty(),
            "beam search must produce tokens on a real (non-zero-weight) forward pass"
        );
        assert_eq!(
            pipeline_generated, naive_generated,
            "InferencePipeline's KV-reuse beam search (rewind_cache + partial \
             reprefill) diverged from a naive full-reprefill-every-call baseline \
             on a real forward pass -- the RT-33 cache-sharing optimization must \
             never change *which* tokens are produced, only how many redundant \
             forward passes it takes to produce them.\n  pipeline = {pipeline_generated:?}\n  \
             naive    = {naive_generated:?}"
        );
    }
}
