//! Unit tests for [`crate::pipeline`].
//!
//! Attached to `pipeline.rs` as its `#[cfg(test)] mod tests` via `#[path]`,
//! so the tests keep full access to the module's private items while
//! `pipeline.rs` itself stays comfortably under the workspace 2000-line
//! ceiling (mirrors `api_extensions.rs` / `api_extensions_tests.rs`,
//! `engine.rs` / `engine_tests.rs`). Mechanical split, no behavior change
//! (gatekeeper OPTIONAL O1).

use super::*;
use crate::sampling::SamplingParams;

// ── Builder tests ────────────────────────────────────────────────────────

#[test]
fn test_pipeline_builder_default() {
    let pipeline = PipelineBuilder::new().build();
    assert_eq!(pipeline.max_tokens(), 256);
    assert!(!pipeline.has_healing());
    assert!(!pipeline.has_constraint());
    assert!(pipeline.stop_sequences().is_empty());
}

#[test]
fn test_pipeline_builder_max_tokens() {
    let pipeline = PipelineBuilder::new().max_tokens(512).build();
    assert_eq!(pipeline.max_tokens(), 512);
}

#[test]
fn test_pipeline_builder_greedy() {
    let pipeline = PipelineBuilder::new().greedy().build();
    assert!(matches!(
        pipeline.config.strategy,
        GenerationStrategy::Greedy
    ));
}

#[test]
fn test_pipeline_builder_stop_sequences() {
    let stops = vec!["<|end|>".to_string(), "STOP".to_string()];
    let pipeline = PipelineBuilder::new().stop_on(stops.clone()).build();
    assert_eq!(pipeline.stop_sequences(), stops.as_slice());
}

#[test]
fn test_pipeline_builder_with_healing() {
    let cfg = TokenHealingConfig {
        lookback: 2,
        min_prob: 0.1,
        enabled: true,
    };
    let pipeline = PipelineBuilder::new().with_token_healing(cfg).build();
    assert!(pipeline.has_healing());
}

// ── Output / StopReason tests ────────────────────────────────────────────

#[test]
fn test_pipeline_output_stop_reason() {
    let output = PipelineOutput {
        text: "hello".to_string(),
        text_available: true,
        token_ids: vec![1, 2, 3],
        prompt_tokens: 5,
        completion_tokens: 3,
        stop_reason: StopReason::StopSequence("STOP".to_string()),
        healing_applied: false,
        elapsed_ms: 10,
    };
    assert_eq!(
        output.stop_reason,
        StopReason::StopSequence("STOP".to_string())
    );
    assert_eq!(output.completion_tokens, 3);
    assert_eq!(output.prompt_tokens, 5);
    assert!(output.text_available);
}

// ── Preset tests ─────────────────────────────────────────────────────────

#[test]
fn test_chat_pipeline_preset() {
    let pipeline = chat_pipeline(42, 256, None);
    assert_eq!(pipeline.max_tokens(), 256);
    assert!(!pipeline.has_healing());
    assert!(pipeline.stop_sequences().is_empty());
    assert!(!pipeline.has_detokenizer());
    // Context window should be 4096
    assert_eq!(pipeline.config.context_max_tokens, 4096);
}

#[test]
fn test_chat_pipeline_preset_with_detokenizer() {
    let pipeline = chat_pipeline(
        42,
        256,
        Some(Arc::new(|ids: &[u32]| Ok(format!("{ids:?}")))),
    );
    assert!(pipeline.has_detokenizer());
}

#[test]
fn test_code_pipeline_preset() {
    let pipeline = code_pipeline(0, 128, None);
    assert_eq!(pipeline.max_tokens(), 128);
    assert!(pipeline.has_healing());
    assert_eq!(pipeline.stop_sequences(), &["\n\n"]);
}

#[test]
fn test_greedy_pipeline_preset() {
    let pipeline = greedy_pipeline(64, None);
    assert_eq!(pipeline.max_tokens(), 64);
    assert!(!pipeline.has_healing());
    assert!(!pipeline.has_constraint());
    assert!(matches!(
        pipeline.config.strategy,
        GenerationStrategy::Greedy
    ));
}

// ── Full run test ────────────────────────────────────────────────────────

#[test]
fn test_pipeline_run_basic() {
    use oxibonsai_core::config::Qwen3Config;

    let config = Qwen3Config::tiny_test();
    let mut engine = InferenceEngine::new(
        config,
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        42,
    );

    let mut pipeline = PipelineBuilder::new().max_tokens(5).greedy().build();

    let output = pipeline.run(vec![151644u32, 872], &mut engine);
    // We care that the pipeline runs without panic and produces a result.
    assert_eq!(output.prompt_tokens, 2);
    assert!(output.elapsed_ms < 60_000, "should finish in under 60s");
}

// ── Strategy / constraint / healing wiring tests ─────────────────────────
//
// The `tiny_test` model has no transformer blocks and zero weights, so every
// forward pass yields an all-zero logit vector on any backend. That makes
// greedy argmax pick the last vocab index deterministically and lets these
// tests assert exact behaviour without a real checkpoint.

/// Build a deterministic greedy engine over the tiny test config.
fn greedy_engine() -> InferenceEngine<'static> {
    use oxibonsai_core::config::Qwen3Config;
    InferenceEngine::new(
        Qwen3Config::tiny_test(),
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        42,
    )
}

/// Test constraint that allows exactly one token id at every step.
struct OnlyToken(u32);
impl TokenConstraint for OnlyToken {
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        let mut mask = vec![false; vocab_size];
        let idx = self.0 as usize;
        if idx < vocab_size {
            mask[idx] = true;
        }
        Some(mask)
    }
    fn advance(&mut self, _token: u32) -> bool {
        true
    }
    fn is_complete(&self) -> bool {
        false
    }
    fn reset(&mut self) {}
    fn name(&self) -> &str {
        "OnlyToken"
    }
}

/// Test constraint that reports completion after `n` committed tokens.
struct CompleteAfterN {
    n: usize,
    count: usize,
}
impl TokenConstraint for CompleteAfterN {
    fn allowed_tokens(&self, _generated: &[u32], _vocab_size: usize) -> Option<Vec<bool>> {
        None
    }
    fn advance(&mut self, _token: u32) -> bool {
        self.count += 1;
        true
    }
    fn is_complete(&self) -> bool {
        self.count >= self.n
    }
    fn reset(&mut self) {
        self.count = 0;
    }
    fn name(&self) -> &str {
        "CompleteAfterN"
    }
}

/// Test constraint whose mask is one element *longer* than
/// `vocab_size`, allowing only that unreachable trailing index --
/// reproduces RT-PIPELINE's guard needing to bound its "is anything
/// allowed" scan to `logits.len()` rather than the whole mask (a mask
/// longer than `logits` whose only allowed index sits past the end
/// must still count as fully disallowed, since no in-range logit could
/// ever reflect it).
struct AllowOnlyPastVocabEnd;
impl TokenConstraint for AllowOnlyPastVocabEnd {
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        let mut mask = vec![false; vocab_size + 1];
        mask[vocab_size] = true;
        Some(mask)
    }
    fn advance(&mut self, _token: u32) -> bool {
        true
    }
    fn is_complete(&self) -> bool {
        false
    }
    fn reset(&mut self) {}
    fn name(&self) -> &str {
        "AllowOnlyPastVocabEnd"
    }
}

#[test]
fn test_pipeline_run_greedy_deterministic() {
    let prompt = vec![151644u32, 872];

    let mut e1 = greedy_engine();
    let out1 = PipelineBuilder::new()
        .max_tokens(5)
        .greedy()
        .build()
        .run(prompt.clone(), &mut e1);

    let mut e2 = greedy_engine();
    let out2 = PipelineBuilder::new()
        .max_tokens(5)
        .greedy()
        .build()
        .run(prompt.clone(), &mut e2);

    assert_eq!(out1.token_ids.len(), 5);
    assert_eq!(out1.stop_reason, StopReason::MaxTokens);
    assert_eq!(
        out1.token_ids, out2.token_ids,
        "greedy decoding must be deterministic"
    );
    // Identical logits every step → greedy repeats the same token.
    assert!(out1.token_ids.iter().all(|&t| t == out1.token_ids[0]));
}

/// `SV-09` (pipeline half): `run_autoregressive` drives the engine's
/// prefill/decode primitives directly rather than going through
/// `InferenceEngine::generate`/`generate_streaming`, so it did not
/// inherit their per-step cancellation check -- an armed token was
/// silently ignored for the whole duration of a pipeline run. Arming and
/// cancelling *before* the run must now stop generation before any token
/// is committed, where an otherwise-identical uncancelled run reaches
/// `max_tokens` (the baseline `test_pipeline_run_greedy_deterministic`
/// establishes just above).
#[test]
fn test_pipeline_run_stops_immediately_when_cancelled_before_the_run() {
    let prompt = vec![151644u32, 872];
    let mut engine = greedy_engine();
    let token = engine.arm_cancellation();
    token.cancel();

    let output = PipelineBuilder::new()
        .max_tokens(5)
        .greedy()
        .build()
        .run(prompt, &mut engine);

    assert!(
        output.token_ids.is_empty(),
        "a token cancelled before the run started must stop generation \
         before any token is committed, got {:?}",
        output.token_ids
    );
}

#[test]
fn test_pipeline_run_sampling_honors_chain() {
    use oxibonsai_core::config::Qwen3Config;
    let vocab = Qwen3Config::tiny_test().vocab_size;
    let prompt = vec![151644u32, 872];

    let mut ge = greedy_engine();
    let out_greedy = PipelineBuilder::new()
        .max_tokens(5)
        .greedy()
        .build()
        .run(prompt.clone(), &mut ge);

    // A stochastic chain must drive selection itself. If the pipeline
    // ignored it (the pre-fix bug), the output would match pure argmax.
    let chain = SamplerChain::default_chat(7);
    let mut se = greedy_engine();
    let out_sampling = PipelineBuilder::new()
        .max_tokens(5)
        .with_sampling(chain)
        .build()
        .run(prompt.clone(), &mut se);

    assert_ne!(
        out_sampling.token_ids, out_greedy.token_ids,
        "the configured SamplerChain must be honored, not argmax/the engine sampler"
    );
    assert!(out_sampling.token_ids.iter().all(|&t| (t as usize) < vocab));
}

#[test]
fn test_pipeline_run_beam_search_generates() {
    let cfg = BeamSearchConfig {
        beam_width: 2,
        max_tokens: 3,
        eos_token_id: 999_999,
        early_stopping: false,
        ..Default::default()
    };
    let mut engine = greedy_engine();
    let output = PipelineBuilder::new()
        .with_beam_search(cfg)
        .build()
        .run(vec![151644u32, 872], &mut engine);

    // Regression: the beam path used to stall on empty logits and return
    // nothing; it must now actually explore and produce tokens.
    assert!(
        !output.token_ids.is_empty(),
        "beam search must produce tokens"
    );
    assert_eq!(output.token_ids.len(), 3);
}

#[test]
fn test_pipeline_run_beam_search_honors_constraint() {
    // Regression (runtime-engine-01): InferencePipeline::run_beam_search
    // used to silently ignore any attached TokenConstraint -- combining
    // `.with_constraint(...)` with `.with_beam_search(...)` produced
    // fully unconstrained output. The constraint must now mask every
    // beam's candidates before top-k expansion, exactly like the
    // autoregressive path already does.
    let cfg = BeamSearchConfig {
        beam_width: 2,
        max_tokens: 4,
        eos_token_id: 999_999,
        early_stopping: false,
        ..Default::default()
    };
    let mut engine = greedy_engine();
    let output = PipelineBuilder::new()
        .with_beam_search(cfg)
        .with_constraint(Box::new(OnlyToken(5)))
        .build()
        .run(vec![151644u32, 872], &mut engine);

    assert!(
        !output.token_ids.is_empty(),
        "constrained beam search must still produce tokens"
    );
    assert!(
        output.token_ids.iter().all(|&t| t == 5),
        "the attached constraint must be honored on the beam-search path too, got {:?}",
        output.token_ids
    );
}

#[test]
fn test_pipeline_run_constraint_masks_tokens() {
    let mut engine = greedy_engine();
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(4)
        .greedy()
        .with_constraint(Box::new(OnlyToken(5)))
        .build();

    let output = pipeline.run(vec![151644u32, 872], &mut engine);
    assert_eq!(
        output.token_ids,
        vec![5u32, 5, 5, 5],
        "only the constraint-allowed token may be emitted"
    );
    assert_eq!(output.stop_reason, StopReason::MaxTokens);
}

#[test]
fn test_pipeline_run_constraint_completes() {
    let mut engine = greedy_engine();
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(5)
        .greedy()
        .with_constraint(Box::new(CompleteAfterN { n: 2, count: 0 }))
        .build();

    let output = pipeline.run(vec![151644u32, 872], &mut engine);
    assert_eq!(output.stop_reason, StopReason::ConstraintComplete);
    assert_eq!(
        output.token_ids.len(),
        2,
        "generation must stop when the constraint reports completion"
    );
}

#[test]
fn test_pipeline_run_constraint_over_long_mask_treated_as_unsatisfiable() {
    // Reproduce first: pre-fix, `Some(mask) if
    // !mask.iter().any(|&allowed| allowed)` scans the WHOLE (over-long)
    // mask, finds the one `true` entry past `logits.len()`, and reports
    // "satisfiable". `apply_constraint_mask` is correctly bounded by
    // `logits.get_mut(i)`, so it then drives every REACHABLE logit to
    // `-inf` -- an all-`-inf` vector reaches the sampler anyway, and a
    // token is emitted regardless.
    let mut engine = greedy_engine();
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(4)
        .greedy()
        .with_constraint(Box::new(AllowOnlyPastVocabEnd))
        .build();

    let output = pipeline.run(vec![151644u32, 872], &mut engine);
    assert_eq!(
        output.token_ids,
        Vec::<u32>::new(),
        "no token can legally be emitted when the only allowed index is \
         unreachable past logits.len()"
    );
    assert_eq!(output.stop_reason, StopReason::ConstraintUnsatisfiable);
}

#[test]
fn test_pipeline_run_healing_applies() {
    // lookback=1, min_prob=0.0: the model's argmax over the prefix (the last
    // vocab index) differs from the final prompt token, so healing must fire
    // — proving it is actually driven against the model rather than a no-op.
    let mut engine = greedy_engine();
    let mut pipeline = PipelineBuilder::new()
        .max_tokens(3)
        .greedy()
        .with_token_healing(TokenHealingConfig::default())
        .build();

    let output = pipeline.run(vec![151644u32, 872], &mut engine);
    assert!(
        output.healing_applied,
        "token healing must actually run against the model"
    );
    assert_eq!(output.prompt_tokens, 2);
}

#[test]
fn test_pipeline_try_run_ok() {
    let mut engine = greedy_engine();
    let mut pipeline = PipelineBuilder::new().max_tokens(4).greedy().build();

    let output = pipeline
        .try_run(vec![151644u32, 872], &mut engine)
        .expect("happy-path try_run must return Ok");
    assert_eq!(output.token_ids.len(), 4);
    assert!(matches!(output.stop_reason, StopReason::MaxTokens));
}

// ── StopSequenceMatcher tests (RT-02 / RT-06) ────────────────────────────

#[test]
fn stop_matcher_finds_sequence_split_across_tokens() {
    // "<|end" + "|>" decoded from two separate tokens -- the whole point
    // of RT-02 is that this must be found even though no single token
    // carries the full sequence; matching against the fully decoded text
    // (rather than per-token decimals) makes the split irrelevant.
    let matcher = StopSequenceMatcher::new(&["<|end|>".to_string()]);
    let text = "hello <|end|>";
    match matcher.check(text) {
        StopMatch::Found { sequence, start } => {
            assert_eq!(sequence, "<|end|>");
            assert_eq!(start, text.find("<|end|>").expect("must be found"));
        }
        StopMatch::None => panic!("expected a match"),
    }
}

#[test]
fn stop_matcher_no_match_returns_none() {
    let matcher = StopSequenceMatcher::new(&["STOP".to_string()]);
    assert!(matches!(
        matcher.check("nothing to see here"),
        StopMatch::None
    ));
}

#[test]
fn stop_matcher_empty_sequences_never_match() {
    let matcher = StopSequenceMatcher::new(&[]);
    assert!(matcher.is_empty());
    assert!(matches!(matcher.check("anything at all"), StopMatch::None));
}

#[test]
fn stop_matcher_drops_empty_strings() {
    // An empty stop string would trivially "match" everywhere; treat it
    // as absent rather than terminating generation immediately.
    let matcher = StopSequenceMatcher::new(&["".to_string(), "STOP".to_string()]);
    assert_eq!(matcher.max_stop_len(), 4); // "STOP".len()
    assert!(matches!(matcher.check(""), StopMatch::None));
}

#[test]
fn stop_matcher_picks_earliest_match_among_several_sequences() {
    let matcher = StopSequenceMatcher::new(&["world".to_string(), "hello".to_string()]);
    match matcher.check("hello world") {
        StopMatch::Found { sequence, start } => {
            assert_eq!(sequence, "hello");
            assert_eq!(start, 0);
        }
        StopMatch::None => panic!("expected a match"),
    }
}

#[test]
fn stop_matcher_keep_len_drops_only_triggering_token() {
    // Three tokens decoding to "AB", "CD", "EF" (2 bytes each); a stop
    // match starting inside the last token's span keeps the first two.
    let detok = |ids: &[u32]| -> RuntimeResult<String> {
        let pieces = ["AB", "CD", "EF"];
        Ok(ids.iter().map(|&i| pieces[i as usize]).collect())
    };
    let tokens = [0u32, 1, 2];
    let full = detok(&tokens).expect("decode must succeed");
    let match_start = full.find("EF").expect("must contain EF");
    let keep =
        StopSequenceMatcher::keep_len(&tokens, match_start, &detok).expect("keep_len must succeed");
    assert_eq!(keep, 2, "only the first two tokens should survive");
}

#[test]
fn stop_matcher_keep_len_drops_multiple_tokens_when_match_starts_earlier() {
    let detok = |ids: &[u32]| -> RuntimeResult<String> {
        let pieces = ["AB", "CD", "EF"];
        Ok(ids.iter().map(|&i| pieces[i as usize]).collect())
    };
    let tokens = [0u32, 1, 2];
    // Match starts at byte 2, exactly at the start of token 1's span --
    // token 0 survives, tokens 1 and 2 do not.
    let keep = StopSequenceMatcher::keep_len(&tokens, 2, &detok).expect("keep_len must succeed");
    assert_eq!(keep, 1);
}

#[test]
fn stop_matcher_keep_len_match_at_zero_drops_everything() {
    let detok = |ids: &[u32]| -> RuntimeResult<String> {
        let pieces = ["AB", "CD"];
        Ok(ids.iter().map(|&i| pieces[i as usize]).collect())
    };
    let tokens = [0u32, 1];
    let keep = StopSequenceMatcher::keep_len(&tokens, 0, &detok).expect("keep_len must succeed");
    assert_eq!(keep, 0, "a match at byte 0 drops every token");
}

#[test]
fn stop_matcher_keep_len_propagates_detokenizer_errors() {
    let detok =
        |_ids: &[u32]| -> RuntimeResult<String> { Err(RuntimeError::Tokenizer("boom".into())) };
    let tokens = [0u32, 1];
    assert!(StopSequenceMatcher::keep_len(&tokens, 0, &detok).is_err());
}

#[test]
fn stop_matcher_hold_back_len_for_partial_suffix() {
    let matcher = StopSequenceMatcher::new(&["<|end|>".to_string()]);
    // "<|en" is a genuine prefix of "<|end|>" -- must be held back.
    let hold = matcher.hold_back_len("hello <|en");
    assert_eq!(
        hold, 4,
        "\"<|en\" must be held back as a possible partial match"
    );
}

#[test]
fn stop_matcher_hold_back_len_zero_when_no_partial_match() {
    let matcher = StopSequenceMatcher::new(&["<|end|>".to_string()]);
    assert_eq!(matcher.hold_back_len("nothing relevant here"), 0);
}

#[test]
fn stop_matcher_hold_back_len_zero_for_complete_text() {
    let matcher = StopSequenceMatcher::new(&["STOP".to_string()]);
    assert_eq!(matcher.hold_back_len(""), 0);
    assert_eq!(matcher.hold_back_len("totally unrelated text."), 0);
}

#[test]
fn stop_matcher_hold_back_len_never_panics_on_multibyte_boundary() {
    // A CJK character straddling the hold-back window must not cause a
    // char-boundary panic when a streaming caller later slices at
    // `text.len() - hold_back_len(text)` (RT-06's corrected fix).
    let matcher = StopSequenceMatcher::new(&["<|end|>".to_string()]);
    let text = "日本語のテキスト<|en";
    let hold = matcher.hold_back_len(text);
    let cut = text.len() - hold;
    assert!(
        text.is_char_boundary(cut),
        "cut point must be a char boundary"
    );
    // And it must still catch the genuine partial match.
    assert!(hold >= "<|en".len());
}

// ── common_prefix_len tests (RT-33) ───────────────────────────────────────

#[test]
fn common_prefix_len_basic() {
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
    assert_eq!(common_prefix_len(&[], &[1, 2]), 0);
    assert_eq!(common_prefix_len(&[1, 2], &[1, 2]), 2);
    assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2]), 2);
}

// ── argmax_logits tests (RT-22) ───────────────────────────────────────────
//
// `sampling.rs::argmax` (RT-SAMPLING) and `engine.rs`'s
// `generate_greedy_gpu` already break ties first-wins; `argmax_logits`
// was the odd one out, still using `.max_by(...)` (last-wins, and
// treats a NaN comparison as `Equal`).

#[test]
fn argmax_logits_breaks_ties_toward_the_first_index() {
    // A plateau of equal maxima: `.max_by(...)` (pre-fix) returns the
    // LAST tied index; llama.cpp and this crate's other two greedy
    // paths return the FIRST.
    let logits = vec![1.0_f32, 5.0, 5.0, 5.0, 2.0];
    assert_eq!(
        argmax_logits(&logits),
        1,
        "must return the first of several equal maxima, not the last"
    );
}

#[test]
fn argmax_logits_ignores_a_leading_nan() {
    // `.max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))` (pre-fix)
    // treats a NaN comparison as `Equal`, which can let a leading NaN
    // masquerade as the maximum depending on fold direction. The
    // first-wins fold (`if v > best_val`) never lets a NaN win, since
    // every comparison against NaN is false.
    let logits = vec![f32::NAN, 1.0, 9.0, 3.0];
    assert_eq!(
        argmax_logits(&logits),
        2,
        "a NaN entry must never be selected over a real maximum"
    );
}

#[test]
fn argmax_logits_single_element_and_all_equal() {
    assert_eq!(argmax_logits(&[42.0]), 0);
    assert_eq!(argmax_logits(&[3.0, 3.0, 3.0]), 0);
}
