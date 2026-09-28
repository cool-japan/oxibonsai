//! Unit tests for [`crate::engine`].
//!
//! Attached to `engine.rs` as its `#[cfg(test)] mod tests` via `#[path]`, so
//! the tests keep full access to the module's private items while `engine.rs`
//! itself stays under the workspace 2000-line ceiling.

use super::*;
use crate::request_metrics::RequestRateTracker;

// The tests below only need *some* config to prove
// `InferenceEngine::new`/`batch_generate`/session-tracking behavior —
// none of them depend on production-sized dimensions. They use
// `Qwen3Config::tiny_test()`, not `Qwen3Config::bonsai_8b()`, whose
// `BonsaiModel::new` would allocate ~5 GB of token_embd + output_weight
// tables (plus a ~1.2 GB KV cache) per test; `tiny_test()` exercises the
// identical code path for a few tens of MB.

#[test]
fn engine_creation() {
    let config = Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    // tiny_test()'s num_layers (2), not bonsai_8b()'s (36): the assertion
    // only needs to prove the config passed to `new` is the config
    // the dense model reports back, which holds for any config.
    let dense = engine
        .dense_model()
        .expect("a config-built engine holds a dense model");
    assert_eq!(dense.config().num_layers, 2);
    // The seam's own accessor answers the same for a dense engine.
    assert_eq!(engine.num_layers(), 2);
    assert!(!engine.is_hybrid());
    assert!(engine.hybrid_model().is_none());
}

#[test]
fn engine_stats_initial() {
    let config = Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    let stats = engine.stats();
    assert_eq!(stats.tokens_generated(), 0);
    assert_eq!(stats.requests_completed(), 0);
    assert_eq!(stats.active_session_count(), 0);
    assert!(stats.uptime_seconds() >= 0.0);
    assert!((stats.avg_tokens_per_request() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn engine_stats_record() {
    let stats = EngineStats::new();
    stats.record_request(10);
    stats.record_request(20);
    assert_eq!(stats.tokens_generated(), 30);
    assert_eq!(stats.requests_completed(), 2);
    assert!((stats.avg_tokens_per_request() - 15.0).abs() < f64::EPSILON);
}

#[test]
fn engine_session_tracking() {
    let config = Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    assert_eq!(engine.active_sessions(), 0);
    assert_eq!(engine.session_count(), 0);
}

#[test]
fn engine_batch_generate_empty() {
    let config = Qwen3Config::tiny_test();
    let mut engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    let results = engine.batch_generate(&[], 10);
    assert!(results.is_empty());
    assert_eq!(engine.session_count(), 0);
}

#[test]
fn engine_batch_generate_empty_prompts() {
    let config = Qwen3Config::tiny_test();
    let mut engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    let prompts = vec![vec![], vec![]];
    let results = engine.batch_generate(&prompts, 5);
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.is_ok());
    }
    // Stats should reflect the completed requests
    assert_eq!(engine.stats().requests_completed(), 2);
}

#[test]
fn engine_stats_default() {
    let stats = EngineStats::default();
    assert_eq!(stats.tokens_generated(), 0);
    assert_eq!(stats.requests_completed(), 0);
}

// ── EOS resolution ────────────────────────────────────────────────────

#[test]
fn resolve_eos_from_gguf_metadata() {
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

    // A base Qwen3 tokenizer, for instance, assigns 151643 rather than the
    // instruct EOS 151645 — the engine must honour the model's own key.
    let mut w = GgufWriter::new();
    w.add_metadata(
        "tokenizer.ggml.eos_token_id",
        MetadataWriteValue::U32(151643),
    );
    let bytes = w.to_bytes().expect("write synthetic gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
    assert_eq!(resolve_eos_token_id(&gguf), 151643);
    assert_ne!(resolve_eos_token_id(&gguf), EOS_TOKEN_ID);
}

#[test]
fn resolve_eos_falls_back_when_absent() {
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::writer::GgufWriter;

    // No eos metadata key → fall back to the hardcoded Qwen3 default.
    let w = GgufWriter::new();
    let bytes = w.to_bytes().expect("write synthetic gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
    assert_eq!(resolve_eos_token_id(&gguf), EOS_TOKEN_ID);
}

#[test]
fn synthetic_engine_uses_default_eos() {
    let config = Qwen3Config::tiny_test();
    let engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    assert_eq!(engine.eos_token_id(), EOS_TOKEN_ID);
    // A synthetic engine has no GGUF vocabulary to resolve names against,
    // so its set is exactly the one fallback id.
    assert_eq!(engine.eos_token_ids(), &[EOS_TOKEN_ID]);
    assert!(engine.is_eos(EOS_TOKEN_ID));
    assert!(!engine.is_eos(EOS_TOKEN_ID + 1));
}

// ── RT-18: the EOS *set* ─────────────────────────────────────────────

#[test]
fn set_eos_token_ids_replaces_the_set_and_keeps_the_primary_first() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    engine.set_eos_token_ids([248_046, 248_044]);
    assert_eq!(engine.eos_token_id(), 248_046, "first id is the primary");
    assert_eq!(engine.eos_token_ids(), &[248_046, 248_044]);
    assert!(engine.is_eos(248_044), "a secondary id also terminates");
    assert!(!engine.is_eos(EOS_TOKEN_ID), "the old id is gone");
}

#[test]
fn set_eos_token_ids_ignores_an_empty_set() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    engine.set_eos_token_ids(std::iter::empty());
    assert_eq!(
        engine.eos_token_ids(),
        &[EOS_TOKEN_ID],
        "an engine with no terminator would never stop"
    );
}

/// The behavioural half of `RT-18`: generation must stop on a
/// **non-primary** member of the set, which a single-id comparison
/// cannot express.
#[test]
fn generation_stops_on_a_secondary_eos_id() {
    let params = greedy_params();
    let prompt = [1u32, 2, 3];

    let mut baseline = InferenceEngine::new(Qwen3Config::tiny_test(), params.clone(), 42);
    let produced = baseline.generate(&prompt, 6).expect("baseline generate");
    assert!(
        !produced.is_empty(),
        "the fixture must generate something to terminate on"
    );
    let first = produced[0];
    assert_ne!(first, EOS_TOKEN_ID);

    // Keep the Qwen3 fallback as the primary and add the token this
    // model actually emits first as a secondary terminator.
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 42);
    engine.set_eos_token_ids([EOS_TOKEN_ID, first]);
    let stopped = engine.generate(&prompt, 6).expect("generate");
    assert!(
        stopped.is_empty(),
        "generation must stop on the secondary EOS id, got {stopped:?}"
    );
}

// ── SV-09: cooperative cancellation ──────────────────────────────────

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 32,
    }
}

#[test]
fn cancellation_token_is_attachable_and_detachable() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    assert!(!engine.is_cancelled(), "no token armed");
    assert!(engine.cancellation_token().is_none());

    let token = CancellationToken::new();
    engine.set_cancellation_token(token.clone());
    assert!(engine.cancellation_token().is_some());
    assert!(!engine.is_cancelled());
    token.cancel();
    assert!(engine.is_cancelled());

    engine.clear_cancellation_token();
    assert!(
        !engine.is_cancelled(),
        "a detached token must not keep cancelling"
    );
}

#[test]
fn a_cancelled_token_stops_generation_before_any_token() {
    let prompt = [1u32, 2, 3];
    let token = CancellationToken::new();
    token.cancel();

    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    engine.set_cancellation_token(token.clone());
    assert!(engine.generate(&prompt, 8).expect("generate").is_empty());

    let (tx, rx) = std::sync::mpsc::channel();
    let sent = engine
        .generate_streaming_sync(&prompt, 8, &tx)
        .expect("streaming");
    assert_eq!(sent, 0, "the streaming path must be cancellable too");
    drop(tx);
    assert!(rx.recv().is_err(), "nothing may be emitted after a cancel");

    let mut tracker = RequestRateTracker::new();
    let tracked = engine
        .generate_tracked(&prompt, 8, &mut tracker)
        .expect("tracked");
    assert!(tracked.is_empty());
}

/// Cancellation lands *between decode steps*, deterministically: the
/// emit callback cancels after the third token, so the fourth step must
/// not run and the tokens already produced are returned (not discarded,
/// and not an error).
#[test]
fn cancellation_mid_generation_returns_the_tokens_produced_so_far() {
    let prompt = [1u32, 2, 3];
    let token = CancellationToken::new();
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    engine.set_cancellation_token(token.clone());

    let mut emitted = 0usize;
    let out = engine
        .generate_greedy_penalised(&prompt, 32, |_tok| {
            emitted += 1;
            if emitted == 3 {
                token.cancel();
            }
            true
        })
        .expect("penalised greedy");

    assert_eq!(
        out.len(),
        3,
        "exactly the tokens emitted before the cancel, got {out:?}"
    );
}

#[test]
fn generation_without_a_token_is_unaffected() {
    let prompt = [1u32, 2, 3];
    let mut with_token = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    with_token.set_cancellation_token(CancellationToken::new());
    let mut without = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);

    assert_eq!(
        with_token.generate(&prompt, 6).expect("armed"),
        without.generate(&prompt, 6).expect("unarmed"),
        "an un-cancelled token must not change a single token of output"
    );
}

// ── RT-28: the recurrent-state reset seam ────────────────────────────

#[derive(Default)]
struct CountingRecurrent {
    resets: Arc<AtomicUsize>,
}

impl crate::engine_control::RecurrentState for CountingRecurrent {
    fn reset_recurrent(&mut self) {
        self.resets.fetch_add(1, Ordering::Relaxed);
    }

    fn recurrent_memory_bytes(&self) -> usize {
        163_184_640
    }

    fn recurrent_name(&self) -> &str {
        "counting-fake"
    }
}

#[test]
fn reset_clears_the_attached_recurrent_state() {
    let resets = Arc::new(AtomicUsize::new(0));
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);

    // No state attached: reset must still work.
    engine.reset();
    assert_eq!(engine.recurrent_memory_bytes(), 0);

    engine.set_recurrent_state(Box::new(CountingRecurrent {
        resets: Arc::clone(&resets),
    }));
    assert_eq!(engine.recurrent_memory_bytes(), 163_184_640);

    engine.reset();
    assert_eq!(
        resets.load(Ordering::Relaxed),
        1,
        "the per-request reset (RT-03) must clear the recurrence too"
    );
    engine.reset_recurrent();
    assert_eq!(resets.load(Ordering::Relaxed), 2);

    let taken = engine.take_recurrent_state();
    assert!(taken.is_some());
    engine.reset();
    assert_eq!(
        resets.load(Ordering::Relaxed),
        2,
        "a detached state is no longer reset"
    );
}

/// `reset()` must NOT disarm the request's own token: the server's
/// `run_blocking_generation` resets the engine *before* running the closure
/// that generates, so a token armed for this request would be thrown away
/// before it could ever be observed. (The pool's `EngineLease` clears it on
/// return instead — see `engine_pool`'s lease-lifecycle test.)
#[test]
fn reset_keeps_the_token_armed_for_the_running_request() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    let token = engine.arm_cancellation();
    engine.reset();
    assert!(
        engine.cancellation_token().is_some(),
        "the per-request reset must not disarm the request's own token"
    );
    token.cancel();
    assert!(engine.is_cancelled());
    assert!(
        engine
            .generate(&[1u32, 2, 3], 8)
            .expect("generate")
            .is_empty(),
        "the token armed before the reset must still stop generation"
    );

    engine.clear_cancellation_token();
    assert!(!engine.is_cancelled());
}

// ── Chunked prefill (SV-09 for long prompts) ─────────────────────────

#[test]
fn prefill_chunking_is_off_by_default_and_configurable() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    assert_eq!(engine.prefill_chunk_tokens(), None, "one-shot by default");
    engine.set_prefill_chunk_tokens(Some(64));
    assert_eq!(engine.prefill_chunk_tokens(), Some(64));
    engine.set_prefill_chunk_tokens(Some(0));
    assert_eq!(
        engine.prefill_chunk_tokens(),
        None,
        "a zero chunk size would never advance"
    );
    engine.set_prefill_chunk_tokens(None);
    assert_eq!(engine.prefill_chunk_tokens(), None);
}

#[test]
fn chunked_prefill_generates_the_same_tokens_as_one_shot() {
    let prompt = [1u32, 2, 3, 4, 5, 6, 7];
    let mut one_shot = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    let mut chunked = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    chunked.set_prefill_chunk_tokens(Some(2));

    assert_eq!(
        one_shot.generate(&prompt, 6).expect("one-shot"),
        chunked.generate(&prompt, 6).expect("chunked"),
        "splitting prefill must not change a single generated token"
    );
}

#[test]
fn chunked_prefill_observes_a_cancel_during_the_prompt() {
    let prompt: Vec<u32> = (1..=32u32).collect();
    let token = CancellationToken::new();
    token.cancel();
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    engine.set_prefill_chunk_tokens(Some(4));
    engine.set_cancellation_token(token);

    assert!(
        engine.generate(&prompt, 8).expect("generate").is_empty(),
        "a cancel must abandon the prompt ingest, not only the decode loop"
    );
}

// ── RT-27: speculative decoding is configuration, not an env var ─────

#[test]
fn speculative_config_defaults_off_and_is_settable() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    // The process may carry OXIBONSAI_SPEC from an outer shell; assert on
    // the settable seam rather than on the inherited default.
    engine.set_speculative(crate::engine_control::SpeculativeConfig::default());
    assert!(!engine.speculative().is_enabled());

    engine.set_speculative(crate::engine_control::SpeculativeConfig::ngram());
    assert!(engine.speculative().is_enabled());
    assert_eq!(engine.speculative().draft_len, 4);
}

// ── K-03: tier selection and its surface ─────────────────────────────

#[test]
fn try_from_model_with_tier_accepts_the_universal_reference_tier() {
    let model = BonsaiModel::new(Qwen3Config::tiny_test());
    let engine = InferenceEngine::try_from_model_with_tier(
        model,
        KernelTier::Reference,
        greedy_params(),
        42,
    )
    .expect("Reference is executable on every host");
    assert_eq!(engine.kernel_tier(), KernelTier::Reference);
}

#[test]
fn effective_tier_reason_names_the_tier_that_will_run() {
    let model = BonsaiModel::new(Qwen3Config::tiny_test());
    let engine =
        InferenceEngine::from_model_with_tier(model, KernelTier::Reference, greedy_params(), 42);
    let reason = engine.effective_tier_reason();
    assert!(
        reason.contains("reference"),
        "the operator-facing reason must name the effective tier: {reason}"
    );
    assert!(!engine.uses_fused_gpu_decode());
}

// ── RT-24: one decoding contract ─────────────────────────────────────

#[test]
fn greedy_gpu_eligibility_requires_a_penalty_free_greedy_sampler() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    // A synthetic engine is never on the fused route, so eligibility is
    // false regardless; the penalty terms are asserted directly.
    assert!(!engine.greedy_gpu_eligible(false));
    assert!(!engine.greedy_penalties_active());

    engine.set_penalties(PenaltyParams::new(0.7, 0.0));
    assert!(
        engine.greedy_penalties_active(),
        "a frequency penalty must be visible to the routing predicate"
    );
    assert!(!engine.greedy_gpu_eligible(false));
}

/// `RT-24`: `SamplingParams::default()`
/// used to carry a hidden `repetition_penalty: 1.1`, which silently
/// disqualified every plain `temperature: 0` request from the fused GPU
/// argmax path (and every other `..SamplingParams::default()` seed site in
/// the codebase inherited the same bug). The fixed default (`1.0`) is no
/// longer penalised — `greedy_penalties_active()` must be `false` — though
/// a synthetic/`tiny_test()` engine still never takes the fused GPU route,
/// same as [`greedy_gpu_eligibility_requires_a_penalty_free_greedy_sampler`]
/// above: it was never built as a fusion-capable engine in the first place,
/// which `greedy_gpu_eligible` checks *before* consulting the sampler.
#[test]
fn default_sampling_params_are_not_penalised_and_the_argmax_route_only_needs_gpu_fusion() {
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    assert_eq!(
        engine.sampling_params().repetition_penalty,
        1.0,
        "the fixed Default must not carry a hidden penalty"
    );
    assert!(
        !engine.greedy_penalties_active(),
        "an unpenalised default must not read as 'penalties active'"
    );
    // Still `false` here -- but now purely because a synthetic engine is
    // never `uses_fused_gpu_decode()`, not because of a hidden penalty.
    assert!(!engine.greedy_gpu_eligible(false));
}

#[test]
fn penalised_greedy_honours_the_penalty_and_restores_the_sampler() {
    let prompt = [1u32, 2, 3];
    let params = SamplingParams {
        temperature: 0.9,
        repetition_penalty: 1.3,
        ..greedy_params()
    };
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params.clone(), 42);
    let out = engine
        .generate_greedy_penalised(&prompt, 4, |_| true)
        .expect("penalised greedy");
    assert!(out.len() <= 4);
    assert_eq!(
        engine.sampling_params().repetition_penalty,
        params.repetition_penalty,
        "the temporary greedy override must be restored"
    );
    assert_eq!(
        engine.sampling_params().temperature,
        params.temperature,
        "temperature must be restored after the greedy override"
    );
}

#[test]
fn penalised_greedy_is_deterministic_regardless_of_seed() {
    let prompt = [1u32, 2, 3];
    let params = SamplingParams {
        temperature: 0.9,
        repetition_penalty: 1.3,
        ..greedy_params()
    };
    let mut a = InferenceEngine::new(Qwen3Config::tiny_test(), params.clone(), 1);
    let mut b = InferenceEngine::new(Qwen3Config::tiny_test(), params, 9_999);
    assert_eq!(
        a.generate_greedy_penalised(&prompt, 5, |_| true)
            .expect("a"),
        b.generate_greedy_penalised(&prompt, 5, |_| true)
            .expect("b"),
        "greedy decoding must not consume RNG, whatever the seed"
    );
}

// ── Penalty seam wiring ───────────────────────────────────────────────

#[test]
fn penalties_default_off_and_settable() {
    let config = Qwen3Config::tiny_test();
    let mut engine = InferenceEngine::new(config, SamplingParams::default(), 42);
    assert!(!engine.penalties().is_active(), "penalties default off");
    engine.set_penalties(PenaltyParams::new(0.5, 0.25));
    assert!(engine.penalties().is_active());
    assert_eq!(engine.penalties().frequency_penalty, 0.5);
    assert_eq!(engine.penalties().presence_penalty, 0.25);
}

#[test]
fn generate_is_deterministic_with_penalties() {
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        repetition_penalty: 1.3,
        max_tokens: 128,
    };
    let prompt = vec![151644u32, 872, 1234];

    let mut e1 = InferenceEngine::new(Qwen3Config::tiny_test(), params.clone(), 7);
    e1.set_penalties(PenaltyParams::new(0.5, 0.5));
    let o1 = e1.generate(&prompt, 16).expect("gen1");

    let mut e2 = InferenceEngine::new(Qwen3Config::tiny_test(), params, 7);
    e2.set_penalties(PenaltyParams::new(0.5, 0.5));
    let o2 = e2.generate(&prompt, 16).expect("gen2");

    assert_eq!(
        o1, o2,
        "same seed + params + penalties must be deterministic"
    );
}

#[test]
fn generate_with_params_and_penalties_restores_state() {
    let prompt = vec![151644u32, 872, 1234];
    let base = SamplingParams::default();
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), base.clone(), 11);

    let temp_params = SamplingParams {
        temperature: 0.5,
        repetition_penalty: 1.5,
        ..base.clone()
    };
    let penalties = PenaltyParams::new(0.8, 0.2);
    let _ = engine
        .generate_with_params_and_penalties(&prompt, 8, &temp_params, &penalties)
        .expect("scoped generate");

    // Both params and penalties are restored to their pre-call values.
    assert_eq!(engine.penalties(), PenaltyParams::default());
    assert!((engine.sampler.params().temperature - base.temperature).abs() < f32::EPSILON);
    assert!(
        (engine.sampler.params().repetition_penalty - base.repetition_penalty).abs() < f32::EPSILON
    );
}

#[cfg(feature = "server")]
#[test]
fn generate_with_logprobs_returns_sane_values() {
    let params = SamplingParams {
        temperature: 0.0, // greedy for a stable emitted token
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let prompt = vec![151644u32, 872, 1234];
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, 3);

    let (tokens, logprobs) = engine
        .generate_with_logprobs(&prompt, 6, 5, &|id| format!("tok{id}"))
        .expect("generate_with_logprobs");

    assert_eq!(
        tokens.len(),
        logprobs.len(),
        "one logprob entry per emitted token"
    );
    assert!(!tokens.is_empty(), "greedy tiny model should emit tokens");

    for lp in &logprobs {
        // A log-probability is always <= 0.
        assert!(
            lp.logprob <= 1e-4,
            "chosen-token logprob must be <= 0, got {}",
            lp.logprob
        );
        // top_logprobs is clamped to k (<= 20) and sorted descending.
        assert!(lp.top_logprobs.len() <= 5);
        assert!(!lp.top_logprobs.is_empty());
        for pair in lp.top_logprobs.windows(2) {
            assert!(
                pair[0].logprob >= pair[1].logprob,
                "top_logprobs must be sorted descending"
            );
        }
    }
}

#[cfg(feature = "server")]
#[test]
fn generate_with_logprobs_empty_prompt() {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 1);
    let (tokens, logprobs) = engine
        .generate_with_logprobs(&[], 4, 3, &|id| format!("t{id}"))
        .expect("empty prompt ok");
    assert!(tokens.is_empty());
    assert!(logprobs.is_empty());
}

// ── GPU-argmax tie-break dependency gate ──────────────────────────────

/// Proves the gate actually gates: an engine presenting every *other*
/// condition `greedy_gpu_eligible` checks (fused route, GPU kernel tier,
/// greedy sampler) is eligible exactly when
/// `GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX` is `true` -- the gate is what
/// flips the predicate here, not any of the other conditions, which all
/// already hold.
///
/// Built through `assemble` directly (reachable here only because
/// `engine_tests` is a descendant module of `engine`, so it shares
/// private-item visibility) because every *public* constructor either
/// forces `fused_gpu_decode: false` (`from_model_with_kernel`/`new`) or
/// requires a real GGUF file that itself classifies as fused
/// (`from_gguf`). `KernelTier::Gpu` here is a nominal tag, never dispatched
/// through — this test only reads it back via `greedy_gpu_eligible`'s
/// predicate — which is exactly the "benchmarks only" use
/// `with_tier_unchecked` documents, extended to a pure-predicate unit test.
#[cfg(any(feature = "metal", feature = "native-cuda"))]
#[test]
fn greedy_gpu_eligible_reflects_the_tiebreak_gate_state() {
    let model = BonsaiModel::new(Qwen3Config::tiny_test());
    // Safety: never dispatched through a real kernel call in this test.
    let kernel = unsafe { KernelDispatcher::with_tier_unchecked(KernelTier::Gpu) };
    let sampler = Sampler::new(greedy_params(), 42);

    let engine = InferenceEngine::assemble(
        LoadedModel::Dense(Box::new(model)),
        kernel,
        sampler,
        EosTokenSet::single(EOS_TOKEN_ID),
        true, // fused_gpu_decode
    );

    assert!(
        engine.uses_fused_gpu_decode(),
        "test setup: the engine must present as the fused-GPU route for \
         this assertion to mean anything"
    );
    // Compile-time, not runtime (clippy: `assert!` on a `const` is
    // pointless as a test): if this ever flips back to `false`, the crate
    // stops building here until the assertion below is also revisited.
    const {
        assert!(
            GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX,
            "read GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX's doc comment before \
             changing this -- the assertion right below assumes it is true"
        );
    }
    assert!(
        engine.greedy_gpu_eligible(false),
        "with the tie-break gate open, an engine presenting the fused \
         route, GPU tier and greedy sampler conditions must now be \
         eligible"
    );
}

// ── EngineStats symmetry across decode routes ─────────────────────────

#[test]
fn generate_streaming_sync_records_engine_stats() {
    let prompt = [1u32, 2, 3];
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    assert_eq!(engine.session_count(), 0);

    let (tx, rx) = std::sync::mpsc::channel();
    let sent = engine
        .generate_streaming_sync(&prompt, 4, &tx)
        .expect("streaming_sync");
    drop(tx);
    assert_eq!(rx.try_iter().count(), sent, "every sent token must arrive");

    assert_eq!(
        engine.session_count(),
        1,
        "generate_streaming_sync's CPU tail must record engine stats just \
         like generate/generate_tracked's do, so \
         EngineStats::requests_completed does not depend on which decode \
         route a request happened to take"
    );
    assert_eq!(engine.stats().tokens_generated(), sent as u64);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn generate_streaming_records_engine_stats() {
    let prompt = [1u32, 2, 3];
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    assert_eq!(engine.session_count(), 0);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let sent = engine
        .generate_streaming(&prompt, 4, &tx)
        .expect("streaming");
    drop(tx);
    let mut received = 0usize;
    while rx.try_recv().is_ok() {
        received += 1;
    }
    assert_eq!(received, sent, "every sent token must arrive");

    assert_eq!(
        engine.session_count(),
        1,
        "generate_streaming's CPU tail must record engine stats \
         symmetrically with generate_streaming_sync's"
    );
}

// ── MET-M1: GPU weight lifecycle (eviction half + replica sharing) ─────

/// A 2-layer, all-`Q1_0_g128` GGUF whose every quantized tensor carries
/// distinct pseudo-random bits drawn from `seed`.
///
/// Distinct per tensor *and* per seed on purpose: the GPU backend shares
/// byte-identical uploads, so a fixture of uniform blocks would share
/// buffers between its own tensors and with any other test's fixture,
/// and the sharing assertions below would measure the wrong thing.
fn distinct_q1_gguf(seed: u64) -> Vec<u8> {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

    let (h, inter, layers, nq, nkv, hd, vocab) = (128usize, 256usize, 2usize, 4usize, 2, 32, 32);
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut q1 = |num_weights: usize| -> Vec<u8> {
        let mut data = Vec::with_capacity(num_weights / 128 * 18);
        for _ in 0..num_weights / 128 {
            let scale = 0.02 + ((next() >> 40) % 1000) as f32 * 1e-5;
            data.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            for _ in 0..2 {
                data.extend_from_slice(&next().to_le_bytes());
            }
        }
        data
    };
    let f32_bytes = |n: usize, value: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| (value * (1.0 + 0.01 * (i % 7) as f32)).to_le_bytes())
            .collect()
    };

    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".into()),
    );
    w.add_metadata(
        "general.name",
        MetadataWriteValue::Str("GpuLifecycle".into()),
    );
    w.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
    w.add_metadata("qwen3.block_count", MetadataWriteValue::U32(layers as u32));
    w.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(nq as u32),
    );
    w.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(nkv as u32),
    );
    w.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(inter as u32),
    );
    w.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
    w.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
    w.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
    w.add_tensor(TensorEntry {
        name: "token_embd.weight".into(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::F32,
        data: f32_bytes(vocab * h, 0.5),
    });
    w.add_tensor(TensorEntry {
        name: "output_norm.weight".into(),
        shape: vec![h as u64],
        tensor_type: TensorType::F32,
        data: f32_bytes(h, 1.0),
    });
    w.add_tensor(TensorEntry {
        name: "output.weight".into(),
        shape: vec![h as u64, vocab as u64],
        tensor_type: TensorType::Q1_0G128,
        data: q1(vocab * h),
    });
    for layer in 0..layers {
        let p = format!("blk.{layer}");
        for (name, n) in [
            ("attn_norm", h),
            ("ffn_norm", h),
            ("attn_q_norm", hd),
            ("attn_k_norm", hd),
        ] {
            w.add_tensor(TensorEntry {
                name: format!("{p}.{name}.weight"),
                shape: vec![n as u64],
                tensor_type: TensorType::F32,
                data: f32_bytes(n, 1.0),
            });
        }
        for (name, ne0, ne1) in [
            ("attn_q", h, nq * hd),
            ("attn_k", h, nkv * hd),
            ("attn_v", h, nkv * hd),
            ("attn_output", nq * hd, h),
            ("ffn_gate", h, inter),
            ("ffn_up", h, inter),
            ("ffn_down", inter, h),
        ] {
            w.add_tensor(TensorEntry {
                name: format!("{p}.{name}.weight"),
                shape: vec![ne0 as u64, ne1 as u64],
                tensor_type: TensorType::Q1_0G128,
                data: q1(ne0 * ne1),
            });
        }
    }
    w.to_bytes().expect("fixture serialises")
}

/// An engine that never uploaded under an epoch of its own reports nothing
/// and releases nothing when it drops (the `Drop` impl must be inert for
/// every synthetic-config engine, i.e. most of this suite).
#[test]
fn an_engine_without_its_own_epoch_reports_and_releases_nothing() {
    let engine = InferenceEngine::new(Qwen3Config::tiny_test(), greedy_params(), 42);
    assert_eq!(engine.model_epoch(), UNATTRIBUTED_MODEL_EPOCH);
    assert!(engine.gpu_upload_stats().is_empty());
    assert_eq!(engine.gpu_weight_registrations(), 0);
    drop(engine);
}

/// `Backend::Cpu` places nothing on the GPU even on a Metal build: the
/// engine gets an epoch (the constructor mints one before it knows) but
/// registers no upload under it.
#[test]
fn a_cpu_backend_engine_uploads_nothing() {
    let bytes = distinct_q1_gguf(0x0C0F_FEE0_0000_0001);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let engine =
        InferenceEngine::from_gguf_with_backend(&gguf, greedy_params(), 42, 64, Backend::Cpu)
            .expect("cpu engine");
    assert_eq!(engine.backend(), Backend::Cpu);
    #[cfg(any(feature = "metal", feature = "native-cuda"))]
    assert_ne!(engine.kernel_tier(), KernelTier::Gpu);
    assert!(engine.gpu_upload_stats().is_empty());
    assert_eq!(engine.gpu_weight_registrations(), 0);
}

/// `MET-M1` end to end at the engine level: two replicas of one Q1 model
/// share every resident buffer (the second uploads nothing), dropping one
/// releases exactly its own epoch while its sibling keeps decoding the same
/// tokens through the still-resident buffers, and the last release frees
/// them (a later load uploads afresh).
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn replicas_share_resident_weights_and_drop_releases_only_their_own_epoch() {
    let bytes = distinct_q1_gguf(0x0005_EED0_A11C_E001);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let prompt = [1u32, 5, 9];

    let mut first = InferenceEngine::from_gguf(&gguf, greedy_params(), 42, 64).expect("replica 1");
    if first.kernel_tier() != KernelTier::Gpu {
        eprintln!(
            "capability report: replicas_share_resident_weights_and_drop_releases_only_their_own_epoch \
             SKIPPED -- no accelerated Metal device on this host"
        );
        return;
    }
    let first_uploads = first.gpu_upload_stats();
    assert!(
        first_uploads.fresh_buffers > 0,
        "a Q1 engine on the GPU tier uploads its weights: {first_uploads:?}"
    );
    assert_eq!(
        first_uploads.shared_buffers, 0,
        "every tensor of the fixture is distinct: nothing to share within one model"
    );
    assert_eq!(
        first.gpu_weight_registrations(),
        first_uploads.total_buffers()
    );

    let mut second = InferenceEngine::from_gguf(&gguf, greedy_params(), 42, 64).expect("replica 2");
    let second_uploads = second.gpu_upload_stats();
    assert_ne!(second.model_epoch(), first.model_epoch());
    assert_eq!(
        second_uploads.fresh_buffers, 0,
        "replica 2 must reuse replica 1's resident buffers, not upload a second copy"
    );
    assert_eq!(second_uploads.shared_buffers, first_uploads.fresh_buffers);
    assert_eq!(second_uploads.shared_bytes, first_uploads.fresh_bytes);
    assert_eq!(
        second.gpu_weight_registrations(),
        second_uploads.total_buffers()
    );

    let from_first = first.generate(&prompt, 6).expect("replica 1 decodes");
    let from_second = second.generate(&prompt, 6).expect("replica 2 decodes");
    assert_eq!(from_first, from_second, "shared buffers, identical output");

    let probe = KernelDispatcher::auto_detect();
    let first_epoch = first.model_epoch();
    drop(first);
    assert_eq!(
        oxibonsai_kernels::gpu_backend::model_registration_count(&probe, first_epoch),
        0,
        "dropping replica 1 releases its epoch"
    );
    assert_eq!(
        second.gpu_weight_registrations(),
        second_uploads.total_buffers(),
        "and only its epoch: the sibling's registrations are untouched"
    );
    second.reset();
    let survivor = second.generate(&prompt, 6).expect("the survivor decodes");
    assert_eq!(
        survivor, from_first,
        "the survivor's buffers stayed resident through its sibling's release"
    );

    let second_epoch = second.model_epoch();
    drop(second);
    assert_eq!(
        oxibonsai_kernels::gpu_backend::model_registration_count(&probe, second_epoch),
        0
    );

    // The last release freed the buffers: a new load uploads afresh.
    let third = InferenceEngine::from_gguf(&gguf, greedy_params(), 42, 64).expect("replica 3");
    let third_uploads = third.gpu_upload_stats();
    assert_eq!(third_uploads.fresh_buffers, first_uploads.fresh_buffers);
    assert_eq!(third_uploads.fresh_bytes, first_uploads.fresh_bytes);
    assert_eq!(third_uploads.shared_buffers, 0);
}

// ── RT-23: the engine's min-p seam (`set_min_p` / `min_p`) ─────────────

/// A known logit row for the min-p tests: five live ids whose probabilities
/// at temperature 1.0 are ≈ 0.479, 0.392, 0.065, 0.039 and 0.024, so min-p
/// 0.5 (threshold ≈ 0.240) keeps exactly [`MIN_P_SURVIVORS`].
const MIN_P_ROW: [(u32, f32); 5] = [(100, 3.0), (101, 2.8), (102, 1.0), (103, 0.5), (104, 0.0)];

/// The ids of [`MIN_P_ROW`] at or above half the top probability.
const MIN_P_SURVIVORS: [u32; 2] = [100, 101];

fn min_p_params(top_k: usize) -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_k,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 24,
    }
}

/// A synthetic engine whose every logit row is [`MIN_P_ROW`] (through the
/// test-only scripted-row seam), with its min-p set through the seam under
/// test.
fn known_row_engine(params: SamplingParams, seed: u64, min_p: f32) -> InferenceEngine<'static> {
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), params, seed);
    engine.script_known_row(MIN_P_ROW.to_vec());
    engine.set_min_p(min_p);
    engine
}

/// `n` seeded draws over [`MIN_P_ROW`] (padded with the scripted floor to
/// `width`) by a standalone sampler with the given min-p: what an engine
/// honouring that min-p must produce.
fn known_row_reference(
    params: &SamplingParams,
    seed: u64,
    min_p: f32,
    width: usize,
    n: usize,
) -> Vec<u32> {
    let mut row = vec![SCRIPTED_LOGIT_FLOOR; width];
    for &(id, logit) in &MIN_P_ROW {
        row[id as usize] = logit;
    }
    let mut sampler = Sampler::new(params.clone(), seed);
    sampler.set_min_p(min_p);
    (0..n)
        .map(|_| sampler.sample(&row).expect("reference draw"))
        .collect()
}

/// The seam's contract: a pure forwarder to the engine's sampler, disabled
/// by default, stored as given, and sampler configuration — it survives
/// `reset` and every generation entry point until changed.
#[test]
fn engine_set_min_p_round_trips_and_persists_across_requests() {
    let prompt = [1u32, 2, 3];
    let mut engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
    assert_eq!(engine.min_p(), 0.0, "disabled by default");

    engine.set_min_p(0.1);
    assert_eq!(engine.min_p(), 0.1);
    assert_eq!(
        engine.sampler.min_p(),
        0.1,
        "a pure forwarder to the sampler every entry point decodes with"
    );

    engine.reset();
    assert_eq!(
        engine.min_p(),
        0.1,
        "reset clears sequence state, not sampler configuration"
    );

    let other = SamplingParams {
        temperature: 0.5,
        top_k: 5,
        ..SamplingParams::default()
    };
    let _ = engine.generate(&prompt, 2).expect("generate");
    assert_eq!(engine.min_p(), 0.1, "generate");
    let _ = engine
        .generate_with_params(&prompt, 2, &other)
        .expect("generate_with_params");
    assert_eq!(engine.min_p(), 0.1, "generate_with_params");
    let _ = engine
        .generate_with_params_and_penalties(&prompt, 2, &other, &PenaltyParams::new(0.2, 0.1))
        .expect("generate_with_params_and_penalties");
    assert_eq!(engine.min_p(), 0.1, "generate_with_params_and_penalties");
    let _ = engine
        .generate_with_seed(&prompt, 2, 7, &other)
        .expect("generate_with_seed");
    assert_eq!(engine.min_p(), 0.1, "generate_with_seed");
    let (tx, _rx) = std::sync::mpsc::channel();
    engine
        .generate_streaming_sync(&prompt, 2, &tx)
        .expect("generate_streaming_sync");
    assert_eq!(engine.min_p(), 0.1, "generate_streaming_sync");

    // Stored exactly as given: clamping happens when a draw applies it.
    for value in [1.5f32, -0.25, 0.0] {
        engine.set_min_p(value);
        assert_eq!(engine.min_p(), value);
    }
    // A server's per-request override sets the same sampler field.
    engine.sampler.set_min_p(0.3);
    assert_eq!(engine.min_p(), 0.3);
}

/// On a known logit row, min-p 0.5 removes every low-probability survivor
/// from the seeded draw that min-p 0.0 keeps, through the ranked (`top_k`
/// 20, the Bonsai 2 default) and the unranked (`top_k` 0) sampler paths;
/// and the engine's realisation is draw for draw a standalone sampler's
/// with the same min-p.
#[test]
fn engine_set_min_p_removes_low_probability_survivors_from_the_seeded_draw() {
    const TOKENS: usize = 24;
    let prompt = [1u32, 2, 3];
    let mut low_probability_draws = 0usize;
    for top_k in [0usize, 20] {
        let params = min_p_params(top_k);
        for seed in 0..8u64 {
            let mut unfiltered = known_row_engine(params.clone(), seed, 0.0);
            let without = unfiltered.generate(&prompt, TOKENS).expect("min-p 0.0");
            let width = unfiltered.vocab_size();
            assert_eq!(without.len(), TOKENS, "the known row never scripts EOS");
            assert_eq!(
                without,
                known_row_reference(&params, seed, 0.0, width, TOKENS),
                "top_k {top_k} seed {seed}: min-p 0.0 is the plain sampler"
            );
            assert!(
                without
                    .iter()
                    .all(|token| MIN_P_ROW.iter().any(|(id, _)| id == token)),
                "top_k {top_k} seed {seed}: a draw left the known support: {without:?}"
            );
            low_probability_draws += without
                .iter()
                .filter(|token| !MIN_P_SURVIVORS.contains(token))
                .count();

            let mut filtered = known_row_engine(params.clone(), seed, 0.5);
            let with = filtered.generate(&prompt, TOKENS).expect("min-p 0.5");
            assert_eq!(
                with,
                known_row_reference(&params, seed, 0.5, width, TOKENS),
                "top_k {top_k} seed {seed}: the engine applies exactly the sampler's min-p"
            );
            assert!(
                with.iter().all(|token| MIN_P_SURVIVORS.contains(token)),
                "top_k {top_k} seed {seed}: min-p 0.5 must remove every candidate below half \
                 the top probability, got {with:?}"
            );
        }
    }
    assert!(
        low_probability_draws > 0,
        "without min-p the low-probability ids must be drawn somewhere, or their removal \
         proves nothing"
    );
}

/// Min-p is clamped when a draw applies it, exactly as the sampler always
/// has: above 1.0 behaves as 1.0 (only the most likely candidate survives,
/// never none), and zero, negative or `NaN` disable the filter draw for
/// draw.
#[test]
fn engine_set_min_p_is_clamped_by_the_sampler() {
    const TOKENS: usize = 24;
    let prompt = [1u32, 2, 3];
    let params = min_p_params(20);

    let mut above_one = known_row_engine(params.clone(), 9, 5.0);
    assert_eq!(above_one.min_p(), 5.0, "stored as given");
    let strict = above_one.generate(&prompt, TOKENS).expect("min-p 5.0");
    assert_eq!(strict, vec![MIN_P_ROW[0].0; TOKENS]);
    let mut one = known_row_engine(params.clone(), 9, 1.0);
    assert_eq!(
        one.generate(&prompt, TOKENS).expect("min-p 1.0"),
        strict,
        "5.0 behaves exactly as 1.0"
    );

    let mut disabled = known_row_engine(params.clone(), 9, 0.0);
    let unfiltered = disabled.generate(&prompt, TOKENS).expect("min-p 0.0");
    for min_p in [-0.25f32, f32::NAN] {
        let mut engine = known_row_engine(params.clone(), 9, min_p);
        assert_eq!(
            engine.generate(&prompt, TOKENS).expect("disabled min-p"),
            unfiltered,
            "min-p {min_p} must disable the filter"
        );
    }
}

/// `generate_with_seed` runs a fresh per-call sampler; it carries the
/// engine's min-p into it (as it does the penalties), so a seeded request
/// is filtered exactly like an unseeded one, and the engine's own sampler —
/// min-p included — is restored afterwards.
#[test]
fn engine_set_min_p_is_carried_into_generate_with_seed() {
    const TOKENS: usize = 24;
    let prompt = [1u32, 2, 3];
    let params = min_p_params(20);
    let mut engine = known_row_engine(params.clone(), 1, 0.5);
    for seed in 0..4u64 {
        let seeded = engine
            .generate_with_seed(&prompt, TOKENS, seed, &params)
            .expect("seeded");
        let mut fresh = known_row_engine(params.clone(), seed, 0.5);
        assert_eq!(
            seeded,
            fresh.generate(&prompt, TOKENS).expect("fresh"),
            "seed {seed}: the per-call sampler must carry the engine's min-p"
        );
        assert!(seeded.iter().all(|token| MIN_P_SURVIVORS.contains(token)));
        assert_eq!(engine.min_p(), 0.5, "the engine's sampler is restored");
    }
}

/// The fused GPU route honours the sampler's min-p: on the 1-bit fused
/// fixture, a seeded sampled request with min-p 0.1 set through
/// [`InferenceEngine::set_min_p`] is token-for-token the same with the
/// sampled top-k route on (the default) and off, over 8 seeds, and equal to
/// an independently spelled-out classic loop with a min-p 0.1 sampler. Not
/// vacuous: the route serves every decode step from GPU candidates, and
/// min-p 0.1 changes the route-off realisation of at least one seed. (This
/// fixture's logits span only ~1.2, so the request samples at temperature
/// 0.2: at 1.0 every top-20 candidate would sit above a tenth of the most
/// likely one's probability and min-p 0.1 would filter nothing.)
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn engine_set_min_p_route_on_equals_route_off() {
    use crate::engine_greedy::{SampledTopKConfig, SampledTopKMode};
    use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};

    const TEST: &str = "oxibonsai-runtime::lib::engine_set_min_p_route_on_equals_route_off";
    const TOKENS: usize = 16;
    const MIN_P: f32 = 0.1;
    let _session = oxibonsai_kernels::MetalGraph::bind_new_session().expect("metal session");
    let bytes = distinct_q1_gguf(0x0005_EED0_0111_0B10);
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let params = SamplingParams {
        temperature: 0.2,
        top_k: 20,
        top_p: 0.95,
        repetition_penalty: 1.0,
        max_tokens: TOKENS,
    };
    let prompt = [1u32, 5, 9];

    let mut on = InferenceEngine::from_gguf(&gguf, params.clone(), 0, 64).expect("route on");
    if !on.uses_fused_gpu_decode() {
        eprintln!(
            "capability report: {TEST} SKIPPED -- no accelerated Metal device, so the fused \
             route (and its sampled top-k route) does not run on this host"
        );
        record_skipped(Capability::Metal, TEST);
        return;
    }
    assert_eq!(on.sampled_topk().mode, SampledTopKMode::GpuCandidates);
    assert!(on.sampled_topk_eligible(false));
    let mut off = InferenceEngine::from_gguf(&gguf, params.clone(), 0, 64).expect("route off");
    off.set_sampled_topk(SampledTopKConfig {
        mode: SampledTopKMode::Off,
        ..SampledTopKConfig::default()
    });

    let mut min_p_changed = 0usize;
    for seed in 0..8u64 {
        on.set_min_p(MIN_P);
        off.set_min_p(MIN_P);
        let served = on.stats().sampled_topk_steps();
        on.reset();
        let via_on = on
            .generate_with_seed(&prompt, TOKENS, seed, &params)
            .expect("route on");
        off.reset();
        let via_off = off
            .generate_with_seed(&prompt, TOKENS, seed, &params)
            .expect("route off");
        assert_eq!(via_on.len(), TOKENS, "seed {seed}: no EOS in the fixture");
        assert_eq!(
            via_on, via_off,
            "seed {seed}: route on != route off at min-p {MIN_P}"
        );
        assert_eq!(
            on.stats().sampled_topk_steps() - served,
            (TOKENS - 1) as u64,
            "seed {seed}: every decode step served from GPU candidates"
        );

        off.set_min_p(0.0);
        off.reset();
        let unfiltered = off
            .generate_with_seed(&prompt, TOKENS, seed, &params)
            .expect("route off, min-p 0.0");
        if unfiltered != via_off {
            min_p_changed += 1;
        }

        if seed == 0 {
            let mut reference =
                InferenceEngine::from_gguf(&gguf, params.clone(), 0, 64).expect("reference");
            let mut sampler = Sampler::new(params.clone(), seed);
            sampler.set_min_p(MIN_P);
            let mut row = reference.prefill_from_pos(&prompt, 0).expect("prefill");
            let mut classic = Vec::new();
            for pos in prompt.len()..prompt.len() + TOKENS {
                let token = sampler.sample(&row).expect("classic draw");
                if reference.is_eos(token) {
                    break;
                }
                classic.push(token);
                row = reference.decode_step(token, pos).expect("decode step");
            }
            assert_eq!(
                via_on, classic,
                "the route's min-p draw is the classic sampler's"
            );
        }
    }
    println!(
        "engine_set_min_p_route_on_equals_route_off: 8 seeds identical at min-p {MIN_P}; \
         min-p changed {min_p_changed} of 8 route-off realisations"
    );
    assert!(
        min_p_changed > 0,
        "min-p {MIN_P} never changed a realisation: the comparison proves nothing"
    );
    record_executed(Capability::Metal, TEST);
}
