//! Integration tests for the sampler/engine penalty seam and logprobs variant.
//!
//! These exercise the *public* API surface that the OpenAI-compatible server
//! layer (wave 2) consumes: the exported [`PenaltyParams`] type, the
//! [`apply_frequency_presence_penalty`] primitive, the engine-level
//! `set_penalties` / `penalties` / `eos_token_id` accessors, the
//! `generate_with_params_and_penalties` convenience, and (under the `server`
//! feature) the `generate_with_logprobs` capture variant.

use oxibonsai_core::config::Qwen3Config;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::{apply_frequency_presence_penalty, SamplingParams};
use oxibonsai_runtime::PenaltyParams;

fn tiny_engine(params: SamplingParams, seed: u64) -> InferenceEngine<'static> {
    InferenceEngine::new(Qwen3Config::tiny_test(), params, seed)
}

fn prompt() -> Vec<u32> {
    vec![151644u32, 872, 1234]
}

// ── Exported penalty primitive ────────────────────────────────────────────

#[test]
fn exported_frequency_presence_penalty_matches_openai_formula() {
    // logit -= presence * I(count > 0) + frequency * count
    let mut logits = vec![0.0f32; 6];
    let history = [2u32, 2, 2, 5];
    apply_frequency_presence_penalty(&mut logits, &history, 1.0, 0.5);
    // token 2 seen 3 times: -(0.5 + 1.0*3) = -3.5
    assert!((logits[2] - (-3.5)).abs() < 1e-6, "logits[2]={}", logits[2]);
    // token 5 seen once: -(0.5 + 1.0*1) = -1.5
    assert!((logits[5] - (-1.5)).abs() < 1e-6, "logits[5]={}", logits[5]);
    assert_eq!(logits[0], 0.0);
}

#[test]
fn penalty_params_is_active_semantics() {
    assert!(!PenaltyParams::default().is_active());
    assert!(PenaltyParams::new(0.1, 0.0).is_active());
    assert!(PenaltyParams::new(0.0, 0.1).is_active());
    assert!(!PenaltyParams::new(0.0, 0.0).is_active());
}

// ── Engine-level penalty seam ─────────────────────────────────────────────

#[test]
fn engine_penalties_roundtrip() {
    let mut engine = tiny_engine(SamplingParams::default(), 5);
    assert_eq!(engine.penalties(), PenaltyParams::default());
    let p = PenaltyParams::new(0.7, 0.3);
    engine.set_penalties(p);
    assert_eq!(engine.penalties(), p);
}

#[test]
fn engine_default_eos_token_id() {
    let engine = tiny_engine(SamplingParams::default(), 5);
    // Synthetic config has no GGUF metadata → the Qwen3 default.
    assert_eq!(engine.eos_token_id(), 151645);
}

#[test]
fn generate_repeatable_with_penalties_active() {
    let params = SamplingParams {
        temperature: 0.9,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.4,
        max_tokens: 128,
    };
    let mut a = tiny_engine(params.clone(), 99);
    a.set_penalties(PenaltyParams::new(0.6, 0.4));
    let oa = a.generate(&prompt(), 20).expect("gen a");

    let mut b = tiny_engine(params, 99);
    b.set_penalties(PenaltyParams::new(0.6, 0.4));
    let ob = b.generate(&prompt(), 20).expect("gen b");

    assert_eq!(oa, ob, "penalised generation must be reproducible");
}

#[test]
fn generate_with_params_and_penalties_is_scoped() {
    // The engine keeps its own (default-off) penalties; the scoped call uses
    // per-call ones and restores afterwards.
    let mut engine = tiny_engine(SamplingParams::default(), 21);
    let scoped_params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.2,
        max_tokens: 128,
    };
    let out = engine
        .generate_with_params_and_penalties(
            &prompt(),
            10,
            &scoped_params,
            &PenaltyParams::new(0.5, 0.5),
        )
        .expect("scoped generate");
    assert!(out.len() <= 10);
    // Engine penalties restored to default-off.
    assert!(!engine.penalties().is_active());
}

// ── Logprobs capture variant (server feature) ─────────────────────────────

#[cfg(feature = "server")]
#[test]
fn generate_with_logprobs_public_api() {
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut engine = tiny_engine(params, 4);
    let (tokens, logprobs) = engine
        .generate_with_logprobs(&prompt(), 5, 20, &|id| format!("id{id}"))
        .expect("logprobs generate");

    assert_eq!(tokens.len(), logprobs.len());
    for lp in &logprobs {
        assert!(
            lp.logprob <= 1e-4,
            "chosen logprob <= 0, got {}",
            lp.logprob
        );
        // top_k=20 is honoured (clamped to vocab), sorted descending.
        assert!(lp.top_logprobs.len() <= 20);
        for w in lp.top_logprobs.windows(2) {
            assert!(w[0].logprob >= w[1].logprob, "descending order");
        }
    }
}
