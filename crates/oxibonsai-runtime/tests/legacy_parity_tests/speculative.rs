//! Engine-level speculative-decoding gates on the real Ternary-Bonsai-1.7B:
//! the production speculative paths against plain greedy decoding, token for
//! token, on the Metal route.
//!
//! The kernel-family contract itself — a verify of 8 rows or more runs the
//! tiled ternary GEMM while the decode it must equal runs the single-token
//! GEMV — is pinned on the model API by
//! `oxibonsai-model/tests/speculative_ternary_metal_gates.rs` (draft lengths
//! 4, 7 and 12 and the adaptive range 2..=12, on the 1.7B and the 8B). These
//! two gates drive the product code that reaches it:
//!
//! * **n-gram speculation** — `InferenceEngine::generate_greedy_gpu` with
//!   `SpeculativeConfig { mode: Ngram, draft_len }` for `draft_len` 4, 7 and
//!   12 (verify batches of 5, 8 and 13 rows), on two prompts whose greedy
//!   continuation repeats, so the n-gram cache really drafts. Every token the
//!   speculative run emits must equal the engine's non-speculative chain at
//!   the same position, and the process-wide fused-prefill counter must show
//!   the verifies ran as batched Metal runs. A speculative run emits exactly
//!   `max_tokens` tokens, like the plain run of the same budget: the engine's
//!   verify step caps the accepted drafts and the bonus token at the
//!   remaining budget, and the gate prints the overshoot (always 0) beside
//!   each verdict.
//! * **two-engine speculation with an adaptive lookahead** —
//!   `SpeculativeDecoder::with_adaptive` over a CPU draft engine and a Metal
//!   target through `generate_verified`. The controller starts at 5 and may
//!   grow to 12, i.e. verify batches of up to 13 rows on the tiled kernel;
//!   its EWMA is configured to ramp quickly (`alpha` 0.5, one-step cooldown)
//!   so a 200-token generation crosses the 8-row boundary well within it.
//!
//! Both record `legacy-models` (timed, only after every assertion) and
//! self-skip with a record when the model or tokenizer is absent.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelTier};
use oxibonsai_kernels::MetalGraph;
use oxibonsai_runtime::adaptive_lookahead::AdaptiveLookaheadConfig;
use oxibonsai_runtime::engine_control::{SpeculativeConfig as NgramConfig, SpeculativeMode};
use oxibonsai_runtime::speculative::{SpeculativeConfig as DraftConfig, SpeculativeDecoder};
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::parity::gpu_serial;
use oxibonsai_testkit::workspace::{find_model, models_dir};

use super::cross_tier::{engine_for_tier, greedy_sampling_params};
use super::TierEnvGuard;

const MODEL_FILE: &str = "Ternary-Bonsai-1.7B.gguf";

/// Prompts whose greedy continuation repeats (so the n-gram cache has
/// something to draft from): the golden capture's repetitive prompt, and a
/// counting loop.
const NGRAM_PROMPTS: [(&str, &str); 2] = [
    ("capital", "The capital of Japan is"),
    (
        "counting",
        "one two three four five six seven eight nine ten, one two three four five six seven \
         eight nine ten, one two three",
    ),
];

/// Draft lengths: 5 verify rows (row-wise kernel), then 8 and 13 (tiled).
const NGRAM_DRAFT_LENS: [usize; 3] = [4, 7, 12];

/// The `max_tokens` budget of each n-gram run (and of the plain chain it is
/// compared with).
const NGRAM_TOKENS: usize = 64;

/// Tokens the adaptive two-engine run generates: with the fast-ramp
/// controller below, enough for the lookahead to reach its maximum of 12.
const ADAPTIVE_TOKENS: usize = 200;

/// The least tokens a run must produce for the comparison to mean anything.
const MIN_TOKENS: usize = 48;

/// The loaded model, tokenizer and bytes, or `None` after recording the skip.
fn locate(test_name: &str) -> Option<(Vec<u8>, TokenizerBridge)> {
    let Some(model_path) = find_model(MODEL_FILE) else {
        eprintln!("skip: {MODEL_FILE} not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, test_name);
        return None;
    };
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, test_name);
        return None;
    };
    let tokenizer = TokenizerBridge::from_file(
        tokenizer_path
            .to_str()
            .expect("models/ path is valid UTF-8"),
    )
    .expect("load real tokenizer.json");
    let bytes = std::fs::read(&model_path).expect("read real legacy GGUF");
    Some((bytes, tokenizer))
}

/// The engine's n-gram speculation, with every attempt allowed (no
/// accept-rate gating) and no warm-up, so drafting starts at the first decode
/// step.
fn ngram_config(draft_len: usize) -> NgramConfig {
    NgramConfig {
        mode: SpeculativeMode::Ngram,
        draft_len,
        warmup_tokens: 0,
        min_accept_rate: 0.0,
        min_attempts_before_gating: u64::MAX,
        retry_interval: 1,
        force_cpu_decode_after: None,
    }
}

/// `engine.generate_greedy_gpu` on a freshly reset engine, with the number of
/// batched Metal runs it completed (the prompt prefill plus every verify).
fn greedy_with_batched_runs(
    engine: &mut oxibonsai_runtime::engine::InferenceEngine<'_>,
    prompt: &[u32],
    tokens: usize,
    context: &str,
) -> (Vec<u32>, u64) {
    engine.reset();
    let before = MetalGraph::prefill_fused_call_count();
    let out = engine
        .generate_greedy_gpu(prompt, tokens)
        .unwrap_or_else(|e| panic!("{context}: generate_greedy_gpu: {e}"));
    (out, MetalGraph::prefill_fused_call_count() - before)
}

#[test]
fn ternary_1_7b_ngram_speculation_equals_plain_greedy_on_metal() {
    const TEST_NAME: &str =
        "oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_ngram_speculation_equals_plain_greedy_on_metal";
    let _tier_env = TierEnvGuard::cleared();
    let _gpu = gpu_serial();
    let Some((bytes, tokenizer)) = locate(TEST_NAME) else {
        return;
    };
    let started = std::time::Instant::now();
    let gguf = GgufFile::parse(&bytes).expect("parse real legacy GGUF");
    let mut engine = engine_for_tier(&gguf, KernelTier::Gpu);
    engine.set_speculative(NgramConfig::default());
    assert_eq!(
        engine.speculative().mode,
        SpeculativeMode::Off,
        "the reference runs are non-speculative"
    );

    let mut verifies_by_len = [0u64; NGRAM_DRAFT_LENS.len()];
    for (name, text) in NGRAM_PROMPTS {
        let prompt = tokenizer.encode(text).expect("encode the prompt");
        let context = format!("{MODEL_FILE} prompt={name}");
        engine.set_speculative(NgramConfig::default());
        let (plain, plain_runs) =
            greedy_with_batched_runs(&mut engine, &prompt, NGRAM_TOKENS, &context);
        assert_eq!(
            plain.len(),
            NGRAM_TOKENS,
            "{context}: the plain greedy run produced {} tokens (wanted all {NGRAM_TOKENS}, no \
             EOS): the comparison would be vacuous",
            plain.len()
        );
        assert_eq!(
            plain_runs, 1,
            "{context}: the non-speculative decode must run one batched prefill and no batched \
             verify"
        );
        for (slot, &draft_len) in NGRAM_DRAFT_LENS.iter().enumerate() {
            engine.set_speculative(ngram_config(draft_len));
            let label = format!("{context} draft_len={draft_len}");
            let (speculative, runs) =
                greedy_with_batched_runs(&mut engine, &prompt, NGRAM_TOKENS, &label);
            let verifies = runs.saturating_sub(1);
            verifies_by_len[slot] += verifies;
            let overshoot = speculative.len().saturating_sub(NGRAM_TOKENS);
            assert_eq!(
                overshoot,
                0,
                "{label}: the speculative run emitted {} tokens for max_tokens={NGRAM_TOKENS}, \
                 {overshoot} past the budget: a verify step must cap its commit at the \
                 remaining budget",
                speculative.len()
            );
            assert_eq!(
                speculative.len(),
                NGRAM_TOKENS,
                "{label}: the speculative run stopped after {} tokens, short of the budget the \
                 plain chain fills",
                speculative.len()
            );
            let first_diff = speculative
                .iter()
                .zip(&plain)
                .position(|(a, b)| a != b)
                .map_or_else(
                    || "none".to_string(),
                    |i| {
                        format!(
                            "token {i}: plain {} vs speculative {}",
                            plain[i], speculative[i]
                        )
                    },
                );
            assert_eq!(
                speculative[..],
                plain[..speculative.len()],
                "{label}: n-gram speculative greedy diverged from plain greedy; first difference \
                 {first_diff}"
            );
            eprintln!(
                "spec-gate {label}: {} tokens identical to plain greedy (max_tokens \
                 {NGRAM_TOKENS}, overshoot {overshoot}); {verifies} batched verifies",
                speculative.len()
            );
        }
    }
    for (slot, &draft_len) in NGRAM_DRAFT_LENS.iter().enumerate() {
        assert!(
            verifies_by_len[slot] > 0,
            "draft_len={draft_len}: no prompt drafted anything, so no verify ran and the \
             equality above proves nothing about that batch size"
        );
    }
    record_executed_timed(Capability::LegacyModels, TEST_NAME, started.elapsed());
}

#[test]
fn ternary_1_7b_adaptive_two_engine_speculation_equals_plain_greedy_on_metal() {
    const TEST_NAME: &str =
        "oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_adaptive_two_engine_speculation_equals_plain_greedy_on_metal";
    let _tier_env = TierEnvGuard::cleared();
    let _gpu = gpu_serial();
    let Some((bytes, tokenizer)) = locate(TEST_NAME) else {
        return;
    };
    let started = std::time::Instant::now();
    let gguf = GgufFile::parse(&bytes).expect("parse real legacy GGUF");
    let mut target = engine_for_tier(&gguf, KernelTier::Gpu);
    target.set_speculative(NgramConfig::default());
    let draft_engine = engine_for_tier(&gguf, cpu_kernel_tier());
    let prompt = tokenizer
        .encode(NGRAM_PROMPTS[0].1)
        .expect("encode the prompt");
    let context = format!("{MODEL_FILE} two-engine adaptive");

    let (plain, _) = greedy_with_batched_runs(&mut target, &prompt, ADAPTIVE_TOKENS, &context);
    assert!(
        plain.len() >= MIN_TOKENS,
        "{context}: the plain greedy run produced {} tokens (wanted >= {MIN_TOKENS})",
        plain.len()
    );

    // The production default (initial 5, range 2..=12) with a quick EWMA so a
    // fully accepting draft ramps 5 -> 12 within ~126 tokens.
    let adaptive = AdaptiveLookaheadConfig {
        initial: 5,
        min: 2,
        max: 12,
        alpha: 0.5,
        cooldown_steps: 1,
    };
    let mut decoder = SpeculativeDecoder::with_adaptive(
        draft_engine,
        DraftConfig {
            lookahead: 5,
            acceptance_threshold: 0.0,
        },
        adaptive,
    )
    .expect("a valid adaptive lookahead configuration");
    let before = MetalGraph::prefill_fused_call_count();
    let speculative = decoder
        .generate_verified(
            &mut target,
            &prompt,
            ADAPTIVE_TOKENS,
            &greedy_sampling_params(),
        )
        .unwrap_or_else(|e| panic!("{context}: generate_verified: {e}"));
    let runs = MetalGraph::prefill_fused_call_count() - before;

    let first_diff = plain
        .iter()
        .zip(&speculative)
        .position(|(a, b)| a != b)
        .map_or_else(
            || "none (a length difference)".to_string(),
            |i| {
                format!(
                    "token {i}: plain {} vs speculative {}",
                    plain[i], speculative[i]
                )
            },
        );
    assert_eq!(
        speculative, plain,
        "{context}: two-engine speculative greedy diverged from plain greedy; first difference \
         {first_diff}"
    );
    // One batched run for the target's prompt prefill, then one per step: every
    // verify completed on the batched Metal route.
    assert_eq!(
        runs,
        1 + decoder.total_steps,
        "{context}: every verify must complete as one batched Metal run ({} steps)",
        decoder.total_steps
    );
    let mean_draft = decoder.total_draft_tokens as f64 / decoder.total_steps.max(1) as f64;
    assert!(
        mean_draft >= 7.5,
        "{context}: the adaptive lookahead averaged {mean_draft:.2} drafted tokens per step \
         ({} steps, {} accepted of {} drafted); the run never reached the tiled kernel's batch \
         sizes, so it proves nothing about them",
        decoder.total_steps,
        decoder.total_accepted_tokens,
        decoder.total_draft_tokens
    );
    eprintln!(
        "spec-gate {context}: {} tokens identical to plain greedy in {} steps, mean draft \
         {mean_draft:.2}, accepted {} of {} drafted, final lookahead {:?}",
        speculative.len(),
        decoder.total_steps,
        decoder.total_accepted_tokens,
        decoder.total_draft_tokens,
        decoder.adaptive().map(|a| a.lookahead()),
    );
    record_executed_timed(Capability::LegacyModels, TEST_NAME, started.elapsed());
}
