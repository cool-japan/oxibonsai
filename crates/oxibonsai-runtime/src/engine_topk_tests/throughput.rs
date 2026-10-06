//! The sampled-throughput guard (real model, Metal).
//!
//! The sampled top-k route replaces a shared-memory read of the logit row
//! with a GPU selection kernel that costs about 25 ms per token at a 151 669
//! token vocabulary: with the route on, default-sampled decode of
//! Ternary-Bonsai-1.7B runs at 21-23 tok/s against 57-59 tok/s for the
//! full-row draw. This test measures sampled throughput, so that a slower
//! route cannot become the default unnoticed: on the real 1.7B, in one
//! process, the default sampling parameters must decode at no less than
//! [`MIN_RATIO`] times the greedy rate.
//!
//! Why a ratio within one process: host load moves both arms together, so the
//! bound does not depend on how busy the machine is. Each arm runs
//! [`ROUNDS`] times, interleaved with the other (greedy, sampled, greedy,
//! sampled), and the best run of each counts, so one scheduling hiccup cannot
//! fail it. The ratio of a healthy build and the regression it guards against
//! are far apart, and [`MIN_RATIO`] sits between them: on an M3 (release
//! build, load average about 12) the test printed 62.2 tok/s greedy, 53.1
//! tok/s default-sampled (ratio 0.85) and 22.2 tok/s with the route opted in
//! (ratio 0.33).
//!
//! The second half measures the same decode with the route opted in
//! (`set_sampled_topk(SampledTopKConfig::gpu_candidates())`) and prints its
//! ratio, asserting only that the route's step counter moved: it is the proof
//! that this measurement would have caught the regression.
//!
//! Run it on a quiet host, one real-model process at a time
//! (`--test-threads=1`, as the release gate's real-model stages do; the nextest
//! `real-model-gate` group serialises only its own members): another real model
//! decoding on the same GPU moves the two arms unevenly, which the
//! best-of-[`ROUNDS`] rule absorbs for a burst but not for a sustained load.

use std::time::{Duration, Instant};

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::MetalGraph;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

use super::*;
use crate::engine_control::SpeculativeConfig;

/// File name of the real model, looked up under the testkit `models/`
/// directory when `OXI_MODEL` is not set.
const MODEL_FILE: &str = "Ternary-Bonsai-1.7B.gguf";
/// Tokens decoded by every measured run (at least 96, so one-off costs are
/// amortised).
const TOKENS: usize = 128;
/// Tokens decoded by each warm-up run.
const WARMUP_TOKENS: usize = 16;
/// Measured runs per arm; the best one counts.
const ROUNDS: usize = 2;
/// Lowest default-sampled rate, as a multiple of the greedy rate.
const MIN_RATIO: f64 = 0.6;
/// The fixed seed of every run.
const SEED: u64 = 5;
/// An id no vocabulary has: as the engine's only end-of-sequence id it makes
/// every run decode exactly [`TOKENS`] tokens, whatever the model would say.
const NEVER_EOS: u32 = u32::MAX;
/// `The capital of Japan is` in the Qwen3 vocabulary.
const PROMPT: [u32; 5] = [785, 6722, 315, 6323, 374];

/// One timed decode.
#[derive(Clone, Copy)]
struct Run {
    tokens: usize,
    elapsed: Duration,
}

impl Run {
    /// Tokens per second over the whole call (the 5-token prefill included,
    /// identically in every arm).
    fn rate(self) -> f64 {
        self.tokens as f64 / self.elapsed.as_secs_f64().max(1e-9)
    }
}

/// The best rate among `runs`.
fn best(runs: &[Run]) -> f64 {
    runs.iter().map(|run| run.rate()).fold(0.0, f64::max)
}

/// `runs` as `[12.3, 45.6]` tok/s, for the printed report.
fn rates(runs: &[Run]) -> String {
    let each: Vec<String> = runs
        .iter()
        .map(|run| format!("{:.1}", run.rate()))
        .collect();
    format!("[{}]", each.join(", "))
}

/// Best-effort 1/5/15-minute load average, printed beside every figure (read
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

/// The real model: `OXI_MODEL`, else [`MODEL_FILE`] under the testkit
/// `models/` directory (`$OXIBONSAI_MODELS_DIR`, else the repository's own).
/// When neither exists the test records its skip — or fails under
/// `OXI_REQUIRE_MODEL_FILES=1`, which says the model must be there.
fn locate_model(test: &str) -> Option<memmap2::Mmap> {
    let explicit = std::env::var_os("OXI_MODEL")
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from);
    let Some(path) = explicit.or_else(|| oxibonsai_testkit::workspace::find_model(MODEL_FILE))
    else {
        assert!(
            !std::env::var("OXI_REQUIRE_MODEL_FILES").is_ok_and(|value| value == "1"),
            "OXI_REQUIRE_MODEL_FILES=1: {test} needs {MODEL_FILE} (set OXI_MODEL, or \
             OXIBONSAI_MODELS_DIR to a directory holding it)"
        );
        eprintln!(
            "capability report: {test} SKIPPED — set OXI_MODEL, or OXIBONSAI_MODELS_DIR to a \
             directory holding {MODEL_FILE} (checked: {:?})",
            oxibonsai_testkit::workspace::models_dir()
        );
        record_skipped(Capability::LegacyModels, test);
        return None;
    };
    Some(oxibonsai_core::gguf::reader::mmap_gguf_file(&path).expect("the real model maps"))
}

/// One timed decode of [`PROMPT`] with `params` and the fixed seed. The
/// engine is reset first (outside the timer); the run must produce exactly
/// `tokens` tokens.
fn decode(engine: &mut InferenceEngine<'_>, params: &SamplingParams, tokens: usize) -> Run {
    engine.reset();
    let start = Instant::now();
    let output = engine
        .generate_with_seed(&PROMPT, tokens, SEED, params)
        .expect("decode");
    let elapsed = start.elapsed();
    assert_eq!(
        output.len(),
        tokens,
        "every run must decode exactly the requested number of tokens"
    );
    Run {
        tokens: output.len(),
        elapsed,
    }
}

/// On the real 1.7B, default-sampled decode keeps at least [`MIN_RATIO`] of
/// the greedy rate, in one process, and the opted-in GPU top-k route — whose
/// ratio is printed — is what the bound exists to catch.
#[test]
fn real_model_default_sampled_decode_keeps_pace_with_greedy() {
    const TEST: &str =
        "oxibonsai-runtime::lib::real_model_default_sampled_decode_keeps_pace_with_greedy";
    let Some(mmap) = locate_model(TEST) else {
        return;
    };
    let gate_start = Instant::now();
    let Ok(_session) = MetalGraph::bind_new_session() else {
        eprintln!("capability report: {TEST} SKIPPED — no Metal device");
        record_skipped(Capability::LegacyModels, TEST);
        return;
    };
    let gguf = GgufFile::parse(&mmap).expect("the real model parses");

    // `SamplingParams::default()` is what the CLI and the server sample with
    // (temperature 0.7, top_k 40, top_p 0.9, no penalty): the arm under test
    // never touches the route's configuration.
    let defaults = SamplingParams::default();
    let greedy = SamplingParams {
        temperature: 0.0,
        ..defaults.clone()
    };
    let mut engine =
        InferenceEngine::from_gguf(&gguf, defaults.clone(), SEED, 512).expect("real engine");
    if !engine.uses_fused_gpu_decode() {
        eprintln!("capability report: {TEST} SKIPPED — the model is not on the fused GPU route");
        record_skipped(Capability::LegacyModels, TEST);
        return;
    }
    // Pin what could move one arm and not the other: the environment's
    // speculation / forced-CPU seams (greedy decode honours them) and the
    // model's own end-of-sequence ids (a run must not stop early).
    engine.set_speculative(SpeculativeConfig::default());
    engine.set_eos_token_ids([NEVER_EOS]);
    assert_eq!(engine.sampled_topk(), SampledTopKConfig::default());
    assert_eq!(engine.sampled_topk().mode, SampledTopKMode::Off);
    assert!(
        !engine.sampled_topk_eligible(false),
        "the default engine must not be eligible for the GPU top-k route"
    );
    assert!(
        defaults.temperature > 0.0
            && defaults.top_k > 0
            && defaults.top_k <= DEFAULT_SAMPLED_TOPK_CANDIDATES,
        "the default sampling parameters must be a request the route could serve"
    );
    let load_before = load_average();

    // Warm up every path (weight pages, command buffers, the sampler).
    decode(&mut engine, &greedy, WARMUP_TOKENS);
    decode(&mut engine, &defaults, WARMUP_TOKENS);

    // The measurement: greedy and default-sampled, interleaved.
    let full_row_before = engine.stats().sampled_full_row_requests();
    let mut greedy_runs = Vec::with_capacity(ROUNDS);
    let mut default_runs = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        greedy_runs.push(decode(&mut engine, &greedy, TOKENS));
        default_runs.push(decode(&mut engine, &defaults, TOKENS));
    }
    let greedy_rate = best(&greedy_runs);
    let default_rate = best(&default_runs);
    let ratio = default_rate / greedy_rate.max(1e-9);
    let load_default = load_average();
    assert_eq!(
        engine.stats().sampled_topk_steps(),
        0,
        "default sampling must never be served from GPU top-k candidates"
    );
    assert_eq!(
        engine.stats().sampled_full_row_requests() - full_row_before,
        ROUNDS as u64,
        "every default-sampled run decodes the full row on the fused route"
    );

    // The opted-in route, on the same engine: the regression this guard is
    // for. Its ratio is reported, not asserted; its counter must move.
    engine.set_sampled_topk(SampledTopKConfig::gpu_candidates());
    assert!(
        engine.sampled_topk_eligible(false),
        "once opted in, the default sampling parameters are eligible for the route"
    );
    decode(&mut engine, &defaults, WARMUP_TOKENS);
    let steps_before = engine.stats().sampled_topk_steps();
    let mut opt_in_runs = Vec::with_capacity(ROUNDS);
    let mut late_greedy_runs = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        opt_in_runs.push(decode(&mut engine, &defaults, TOKENS));
        // Greedy never consults the route; a fresh reference beside each run.
        late_greedy_runs.push(decode(&mut engine, &greedy, TOKENS));
    }
    let opt_in_rate = best(&opt_in_runs);
    let reference_rate = greedy_rate.max(best(&late_greedy_runs));
    let opt_in_ratio = opt_in_rate / reference_rate.max(1e-9);
    let load_after = load_average();
    let served = engine.stats().sampled_topk_steps() - steps_before;

    eprintln!(
        "sampled-throughput guard ({MODEL_FILE} or OXI_MODEL): {TOKENS} tokens per run, best of \
         {ROUNDS}, seed {SEED}"
    );
    eprintln!(
        "  greedy (GPU argmax)           {greedy_rate:6.1} tok/s  runs {}",
        rates(&greedy_runs)
    );
    eprintln!(
        "  default sampling (route off)  {default_rate:6.1} tok/s  runs {}  ratio {ratio:.3} \
         (floor {MIN_RATIO})",
        rates(&default_runs)
    );
    eprintln!(
        "  opted-in GPU top-k route      {opt_in_rate:6.1} tok/s  runs {}  ratio {opt_in_ratio:.3} \
         (informational; {served} candidate steps)",
        rates(&opt_in_runs)
    );
    eprintln!(
        "  load average: before {load_before} | after the default arms {load_default} | at the \
         end {load_after}"
    );

    assert!(
        ratio >= MIN_RATIO,
        "default-sampled decode ran at {default_rate:.1} tok/s against {greedy_rate:.1} tok/s \
         greedy: ratio {ratio:.3} is below {MIN_RATIO} (opted-in GPU top-k route: \
         {opt_in_rate:.1} tok/s, ratio {opt_in_ratio:.3}; load average {load_default})"
    );
    assert!(
        served > 0,
        "the opted-in runs must be served from GPU candidates: the comparison above measured \
         the wrong path otherwise"
    );
    record_executed_timed(Capability::LegacyModels, TEST, gate_start.elapsed());
}
