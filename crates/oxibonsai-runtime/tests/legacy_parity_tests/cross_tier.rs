//! The cross-tier greedy parity gates: each legacy model decoded on every
//! kernel tier this host can reach (`Reference`, the auto-detected CPU tier,
//! Metal), the tiers compared step by step, and the decoded text checked
//! against the vendored golden captures. The parent file's module docs say
//! what is asserted and why.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::dispatch::{cpu_kernel_tier, KernelDispatcher, KernelTier};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::engine_control::gguf_fused_metal_route;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_runtime::sampling_advanced::apply_repetition_penalty;
use oxibonsai_runtime::tokenizer_bridge::TokenizerBridge;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::gguf_fixture::tiny_dense_qwen3_gguf;
// The comparison logic (token-chain-first, then bit-exact-CPU /
// relative-bound-GPU per step) lives in `oxibonsai_testkit::parity`, shared
// with `crates/oxibonsai-model/tests/legacy_parity_tests.rs` rather than
// duplicated — see that module's doc comment for the full predicate and the
// measured (model, slot) points `REL_TOL` is calibrated against.
use oxibonsai_testkit::parity::{
    argmax_first, compare_against_reference, gpu_serial, top_k, top_two, TierRun,
};
use oxibonsai_testkit::workspace::{find_model, models_dir};

use super::TierEnvGuard;

/// Generation budget for the LOGIT pass's cross-tier comparison: long
/// enough to reach a repetition-inducing tail on every golden prompt, so a
/// backend divergence has room to show up even when the first several
/// steps agree everywhere.
const MAX_TOKENS: usize = 64;

/// The PrismML llama.cpp fork's own `-n 32` capture budget: every string in
/// [`GOLDEN_CASES`] is the fork's greedy continuation for EXACTLY this many
/// tokens, so the TEXT pass below decodes exactly this many of its own
/// (longer, [`MAX_TOKENS`]-token) generation and checks it for EXACT
/// equality against the golden — never a shared/truncated-prefix
/// comparison, which cannot distinguish "the golden capture legitimately
/// ends here" from "the golden string itself is missing its last
/// character" (the defect one golden string actually had: a dropped
/// trailing newline that a prefix check let pass silently).
const GOLDEN_TOKEN_BUDGET: usize = 32;

/// Context budget: generous headroom over `MAX_TOKENS` plus the longest
/// prompt below, well inside all three legacy models' trained context.
const MAX_SEQ: usize = 512;

/// A deterministic seed. Inert at `temperature == 0.0` (see the parent
/// file's module docs).
const SEED: u64 = 42;

/// Below this many decoded characters, an "equals the golden text"
/// assertion would be vacuous (T-09: the gate must be a real assertion, not
/// one that trivially passes on an empty or near-empty string).
const MIN_DECODED_CHARS: usize = 20;

/// Is this pair's non-reference side a CPU tier? Only an accelerator pair
/// gets a numeric tolerance; every CPU tier must be bit-exact against
/// `KernelTier::Reference`. Kept local (not in `oxibonsai_testkit::parity`,
/// which never depends on `oxibonsai-kernels`) and passed to
/// `compare_against_reference` as a closure.
///
/// Written as an exhaustive match over the named CPU tiers, deliberately
/// NOT as `!matches!(tier, KernelTier::Gpu)`: `KernelTier` is not
/// `#[non_exhaustive]`, so a future accelerator variant added under the
/// negated form would silently inherit the bit-exactness assertion and go
/// red for a non-bug, whereas this form fails to COMPILE here and forces
/// the classification to be decided.
fn is_cpu_tier(tier: KernelTier) -> bool {
    match tier {
        KernelTier::Reference => true,
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx2 | KernelTier::Avx512 => true,
        #[cfg(target_arch = "aarch64")]
        KernelTier::Neon => true,
        KernelTier::Gpu => false,
    }
}

// ── Golden fixtures (embedded from the vendored golden capture, legacy_golden.json) ─

/// One `(model, prompt, golden Metal continuation text)` triple.
struct GoldenCase {
    /// GGUF filename under `models/`.
    model_file: &'static str,
    /// The exact prompt text the golden capture was taken from.
    prompt: &'static str,
    /// The captured Metal-backend greedy continuation (does NOT include the
    /// prompt itself), from `legacy_golden.json`'s `backend == "metal"`
    /// rows — the design's chosen greedy truth.
    metal_text: &'static str,
    /// Whether `metal_text` is independently confirmed byte-identical to
    /// the PrismML llama.cpp fork's own greedy output (temp 0) on the same
    /// file and prompt, not only to an earlier OxiBonsai capture of itself.
    /// True for the three `Bonsai-8B.gguf` cases (see the doc comment on
    /// the last of them): a change that matches this text is checked
    /// against llama.cpp's own semantics, not only against this project's
    /// history.
    fork_oracle: bool,
}

/// The 9 `backend == "metal"` rows of the vendored golden capture,
/// `legacy_golden.json` (3 legacy models x 3 prompts each), embedded
/// verbatim.
const GOLDEN_CASES: [GoldenCase; 9] = [
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital of Japan is Tokyo. The capital",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: "\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-1.7B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        metal_text: " there lived a young girl named Lila. She was known for her kindness and her love for the sea. One day, she discovered a mysterious shell that gl",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: " if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2) print(fibonacci(10)) \n\nThe",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Ternary-Bonsai-8B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        metal_text: " there lived a young girl named Lila. She was known for her curiosity and love for the ocean. One day, while exploring the beach, she discovered a",
        fork_oracle: false,
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "The capital of Japan is",
        metal_text: " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital",
        fork_oracle: true,
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "def fibonacci(n):",
        metal_text: "\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef fibonacci(n):\n",
        fork_oracle: true,
    },
    GoldenCase {
        model_file: "Bonsai-8B.gguf",
        prompt: "Once upon a time, in a small village by the sea,",
        // RE-CAPTURED 2026-09-23: Bonsai-8B.gguf declares `rope.scaling.type = yarn`
        // (factor 4, original_context_length 16384). OxiBonsai <= 0.2.4 ignored it; the
        // engine now honours it (M-08), and this text is byte-identical to the PrismML
        // llama.cpp fork (greedy, temp 0) on the same file.
        metal_text: " there lived a young girl named Lila. She was known for her curious nature and her love of the sea. Lila often spent her days exploring the cliffs",
        fork_oracle: true,
    },
];

/// The PrismML llama.cpp fork's own greedy continuation of each
/// `Bonsai-8B.gguf` golden prompt, captured independently of
/// [`GOLDEN_CASES`] and LONGER than its 32-token budget: `llama-completion
/// -n 48 --temp 0 --top-k 1 -s 42` on Metal (fork commit 9a9394a), with the
/// blank line the tool itself prints after every generation removed. The
/// text pass requires every tier's 32-token decode of these prompts to be a
/// prefix of this capture, so the gate stays pinned to llama.cpp's own
/// output even if [`GOLDEN_CASES`] is ever re-captured.
const BONSAI_8B_FORK_CAPTURES: [(&str, &str); 3] = [
    (
        "The capital of Japan is",
        " Tokyo. The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Brazil is Brasília. The capital of Canada is Ottawa. The capital of",
    ),
    (
        "def fibonacci(n):",
        "\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return",
    ),
    (
        "Once upon a time, in a small village by the sea,",
        " there lived a young girl named Lila. She was known for her curious nature and her love of the sea. Lila often spent her days exploring the cliffs, listening to the waves, and collecting seashells. One day, while",
    ),
];

/// The fork's own capture for a fork-oracle case's prompt, if any.
fn fork_capture_for(prompt: &str) -> Option<&'static str> {
    BONSAI_8B_FORK_CAPTURES
        .iter()
        .find(|(p, _)| *p == prompt)
        .map(|(_, capture)| *capture)
}

/// The distinct model filenames in [`GOLDEN_CASES`], in first-appearance
/// order (drives the three top-level `#[test]` functions below).
const LEGACY_MODEL_FILES: [&str; 3] = [
    "Ternary-Bonsai-1.7B.gguf",
    "Ternary-Bonsai-8B.gguf",
    "Bonsai-8B.gguf",
];

/// Fixture-authoring guard: exactly the three `Bonsai-8B.gguf` cases carry
/// `fork_oracle = true`, matching the doc comment on that field, and each
/// one's golden text is a byte prefix of the fork's own longer capture
/// ([`BONSAI_8B_FORK_CAPTURES`]) — so a golden string edited or re-captured
/// away from llama.cpp's output fails here, with no model file needed.
#[test]
fn fork_oracle_rows_are_exactly_the_three_bonsai_8b_cases() {
    for case in &GOLDEN_CASES {
        assert_eq!(
            case.fork_oracle,
            case.model_file == "Bonsai-8B.gguf",
            "{} {:?}: fork_oracle must be true iff the model is Bonsai-8B.gguf",
            case.model_file,
            case.prompt,
        );
        if case.fork_oracle {
            let capture = fork_capture_for(case.prompt)
                .unwrap_or_else(|| panic!("{:?}: no fork capture for this prompt", case.prompt));
            assert!(
                capture.starts_with(case.metal_text) && capture.len() > case.metal_text.len(),
                "{:?}: the golden text is not a strict prefix of the fork's own capture\n  \
                 golden: {:?}\n  fork:   {capture:?}",
                case.prompt,
                case.metal_text
            );
        }
    }
    assert_eq!(
        GOLDEN_CASES.iter().filter(|c| c.fork_oracle).count(),
        3,
        "expected exactly 3 fork-oracle rows"
    );
    assert_eq!(BONSAI_8B_FORK_CAPTURES.len(), 3, "one fork capture per row");
}

/// Mirrors `src/cli/util.rs::build_sampling_params(0.0, 40, 0.9, 1.0)` (see
/// the parent file's module docs for why it cannot literally be `use`d).
/// `repetition_penalty: 1.0`, not `SamplingParams::default()`'s `1.1`, is
/// exactly the fix for the CPU/Metal divergence this gate guards against.
pub(super) fn greedy_sampling_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 40,
        top_p: 0.9,
        repetition_penalty: 1.0,
        ..SamplingParams::default()
    }
}

/// Asserts `decoded` is a plausible, non-vacuous match for `golden`:
/// long enough to rule out a truncated-but-not-empty generation (at least
/// [`MIN_DECODED_CHARS`], AND at least `golden`'s own length capped at 40
/// chars — a spurious early EOS a few tokens in must still fail this, not
/// just a fully empty decode), and EXACTLY equal to `golden` byte for
/// byte — never a shared-prefix-of-whichever-is-shorter comparison. `decoded`
/// here is already sliced by the caller ([`run_text_pass`]) to exactly
/// [`GOLDEN_TOKEN_BUDGET`] tokens, the same budget the fork's own capture
/// used, so "equal" and "the fork's own greedy continuation for this many
/// steps" are the same claim. A shared/truncated-prefix comparison cannot
/// tell "the golden capture legitimately ends here" apart from "the golden
/// string itself lost a trailing character" — the second is a real defect
/// one embedded golden string had (a dropped trailing newline) that such a
/// comparison let pass silently; exact equality at a fixed, known budget
/// closes that hole.
///
/// T-09: a bug that made `decoded` empty (an
/// immediate EOS, a broken tokenizer round-trip, ...) — or merely short,
/// from an EOS a handful of tokens in — must fail this assertion, not
/// vacuously satisfy "is a prefix of everything".
///
/// `logit_pass_run` is this same case/tier's already-computed [`TierRun`]
/// from [`run_logit_pass`] (now run BEFORE the text pass in
/// [`run_model_gate`], see that function's doc comment), threaded through
/// purely for diagnostics on a mismatch: the logit pass is an independent
/// decode (fresh engine, manual argmax loop — see the module doc's "Two
/// independent passes per tier"), so its token ids are used only to locate
/// the first differing TOKEN and its logits are printed as diagnostic
/// context, never asserted equal to this pass's own tokens.
fn assert_exact_match_golden(
    decoded: &str,
    golden: &str,
    context: &str,
    tokenizer: &TokenizerBridge,
    generated_tokens: &[u32],
    logit_pass_run: Option<&TierRun<KernelTier>>,
) {
    let min_chars = MIN_DECODED_CHARS.max(golden.chars().count().min(40));
    assert!(
        decoded.chars().count() >= min_chars,
        "{context}: decoded continuation is implausibly short ({} chars, wanted >= {min_chars}): {decoded:?}",
        decoded.chars().count()
    );
    if decoded == golden {
        eprintln!("{context}: text OK (exact, {GOLDEN_TOKEN_BUDGET}-token budget): {decoded:?}");
        return;
    }

    // Tokenize the golden continuation and diff it against this tier's own
    // generated ids to find the first differing TOKEN (not just byte), then
    // — when this case/tier's logit pass already ran — print that step's
    // top-2 gap and top-5 so a near-tie is distinguishable from a real
    // disagreement. `top_two`/`top_k` already exist for this; nothing new
    // is invented here.
    let golden_tokens = tokenizer.encode(golden).unwrap_or_default();
    let token_detail = match generated_tokens
        .iter()
        .zip(golden_tokens.iter())
        .position(|(a, b)| a != b)
    {
        Some(i) => {
            let this_id = generated_tokens[i];
            let golden_id = golden_tokens[i];
            let this_str = tokenizer
                .decode(&[this_id])
                .unwrap_or_else(|e| format!("<decode failed: {e}>"));
            let golden_str = tokenizer
                .decode(&[golden_id])
                .unwrap_or_else(|e| format!("<decode failed: {e}>"));
            let logit_note = match logit_pass_run.and_then(|r| r.logits.get(i)) {
                Some(logits) => {
                    let top2 = top_two(logits);
                    format!(
                        " | this tier's LOGIT-PASS step {i} (an independent decode — diagnostic \
                         only, not asserted equal to this pass): top-2 gap={:e} top-5={}",
                        top2.gap(),
                        top_k(logits, 5),
                    )
                }
                None => String::new(),
            };
            format!(
                "\n  first differing TOKEN index {i}: this tier picked id {this_id} \
                 ({this_str:?}), golden picked id {golden_id} ({golden_str:?}){logit_note}"
            )
        }
        None => "\n  every zipped token position matches; the two token sequences differ only \
                 in LENGTH (one is a prefix of the other)"
            .to_string(),
    };

    panic!(
        "{context}: decoded text does not EXACTLY match the golden Metal capture over the \
         {GOLDEN_TOKEN_BUDGET}-token budget both were captured at\n  decoded: {decoded:?}\n  \
         golden:  {golden:?}{token_detail}"
    );
}

/// Builds the engine for one tier of the matrix. A CPU tier is pinned with
/// [`InferenceEngine::from_model_with_tier`]. The GPU tier is built the way
/// [`InferenceEngine::from_gguf`] builds a production Metal engine: the
/// auto-detected dispatcher (the only constructor wired to a live GPU
/// backend), the weight upload `from_gguf` performs — skipped exactly where
/// it skips it, on the ternary fused route, whose Metal paths bind their own
/// cached weight set — and the eager GPU cache build. The GPU arm therefore
/// runs the fused Metal batch prefill and decode production runs; the
/// callers assert it ([`assert_fused_gpu_prefill`]).
pub(super) fn engine_for_tier<'a>(gguf: &'a GgufFile<'a>, tier: KernelTier) -> InferenceEngine<'a> {
    let mut model = BonsaiModel::from_gguf(gguf, MAX_SEQ).expect("BonsaiModel::from_gguf");
    if tier != KernelTier::Gpu {
        return InferenceEngine::from_model_with_tier(model, tier, greedy_sampling_params(), SEED);
    }
    let kernel = KernelDispatcher::auto_detect();
    assert_eq!(
        kernel.tier(),
        KernelTier::Gpu,
        "KernelDispatcher::auto_detect() did not select the GPU tier for the GPU arm"
    );
    if !gguf_fused_metal_route(gguf).gpu_weight_upload_redundant() {
        model.upload_weights_to_gpu(&kernel);
    }
    model
        .get_or_create_gpu_cache()
        .unwrap_or_else(|e| panic!("GPU weight cache build for the GPU arm: {e}"));
    InferenceEngine::from_model_with_kernel(model, kernel, greedy_sampling_params(), SEED)
}

/// Captures the `error` field of the WARN event [`PrefillFallbackDetector`]
/// matches.
struct ErrorFieldVisitor<'a>(&'a mut Option<String>);

impl tracing::field::Visit for ErrorFieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "error" {
            *self.0 = Some(format!("{value:?}"));
        }
    }
}

/// A minimal `tracing` subscriber recording the WARN that
/// `BonsaiModel::forward_prefill` emits (from its `prefill_dispatch` module)
/// when a GPU batch prefill fails and it silently falls back to a CPU path,
/// with that event's `error` field.
struct PrefillFallbackDetector {
    reason: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl tracing::Subscriber for PrefillFallbackDetector {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.level() <= &tracing::Level::WARN
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let metadata = event.metadata();
        if *metadata.level() <= tracing::Level::WARN
            && metadata
                .module_path()
                .is_some_and(|m| m.contains("prefill_dispatch"))
        {
            let mut reason = None;
            event.record(&mut ErrorFieldVisitor(&mut reason));
            *self.reason.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(reason.unwrap_or_else(|| "<no error field>".to_string()));
        }
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Runs `f` with [`PrefillFallbackDetector`] installed on this thread,
/// returning `f`'s result and the fallback reason, if a fallback happened.
fn run_detecting_prefill_fallback<T>(f: impl FnOnce() -> T) -> (T, Option<String>) {
    let reason = std::sync::Arc::new(std::sync::Mutex::new(None));
    let detector = PrefillFallbackDetector {
        reason: reason.clone(),
    };
    let result = tracing::subscriber::with_default(detector, f);
    let fallback = reason.lock().unwrap_or_else(|e| e.into_inner()).clone();
    (result, fallback)
}

/// The GPU arm's prefill ran on the fused Metal batch path: no fallback
/// warning, and the model's device-KV latch (`gpu_path_active`, which the
/// CPU prefill fallback clears) is set.
fn assert_fused_gpu_prefill(engine: &InferenceEngine<'_>, fallback: Option<&str>, context: &str) {
    let device_kv = engine
        .dense_model()
        .is_some_and(BonsaiModel::gpu_path_active);
    assert!(
        fallback.is_none() && device_kv,
        "{context}: the GPU arm's prefill did not run on the fused Metal batch path \
         (fallback reason {fallback:?}; device-KV latch set: {device_kv})"
    );
}

/// Runs the "text pass" for one `(tier, case)` pair: a fresh engine drives
/// `generate()` for [`MAX_TOKENS`] steps, and exactly the first
/// [`GOLDEN_TOKEN_BUDGET`] of those tokens (the fork's own capture budget)
/// are detokenized and checked against the golden text for EXACT equality.
/// `logit_pass_run` is threaded through to [`assert_exact_match_golden`]
/// purely for diagnostics — see its doc comment.
fn run_text_pass(
    gguf_bytes: &[u8],
    tier: KernelTier,
    tokenizer: &TokenizerBridge,
    case: &GoldenCase,
    logit_pass_run: Option<&TierRun<KernelTier>>,
) {
    let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
    let mut engine = engine_for_tier(&gguf, tier);

    let prompt_ids = tokenizer.encode(case.prompt).expect("encode golden prompt");
    let (generated, fallback) =
        run_detecting_prefill_fallback(|| engine.generate(&prompt_ids, MAX_TOKENS));
    let generated = generated.expect("engine.generate");
    if tier == KernelTier::Gpu {
        assert_fused_gpu_prefill(
            &engine,
            fallback.as_deref(),
            &format!("TEXT PASS {} prompt {:?}", case.model_file, case.prompt),
        );
    }
    assert!(
        generated.len() >= GOLDEN_TOKEN_BUDGET,
        "{} prompt {:?} tier {tier}: generate() returned only {} tokens, fewer than the \
         {GOLDEN_TOKEN_BUDGET}-token golden budget -- an early EOS or a generation-loop defect, \
         not a text mismatch to diagnose below",
        case.model_file,
        case.prompt,
        generated.len(),
    );
    let golden_budget_tokens = &generated[..GOLDEN_TOKEN_BUDGET];
    let decoded = tokenizer
        .decode(golden_budget_tokens)
        .expect("decode generated ids");

    assert_exact_match_golden(
        &decoded,
        case.metal_text,
        // The GOLDEN-TEXT half is reported separately from the numeric
        // gate — a token-chain match against the METAL goldens with only a
        // numeric bound tripping is a very different release posture from
        // a text divergence, so every message says which half it came from.
        &format!(
            "TEXT PASS {} prompt {:?} tier {tier} ({tier:?})",
            case.model_file, case.prompt
        ),
        tokenizer,
        &generated,
        logit_pass_run,
    );

    // ORACLE: the fork-oracle cases' decode must also continue llama.cpp's
    // own (longer) capture, independently of `GOLDEN_CASES`.
    if case.fork_oracle {
        let capture = fork_capture_for(case.prompt).unwrap_or_else(|| {
            panic!("{:?}: fork-oracle case without a fork capture", case.prompt)
        });
        assert!(
            capture.starts_with(decoded.as_str()),
            "TEXT PASS {} prompt {:?} tier {tier}: the {GOLDEN_TOKEN_BUDGET}-token decode is not \
             a prefix of the PrismML fork's own greedy capture\n  decoded: {decoded:?}\n  \
             fork:    {capture:?}",
            case.model_file,
            case.prompt
        );
        eprintln!(
            "TEXT PASS {} prompt {:?} tier {tier}: fork oracle OK (prefix of the fork's own \
             capture)",
            case.model_file, case.prompt
        );
    }
}

/// Runs the "logit pass" for one case across every tier in `tiers`: a fresh
/// engine per tier drives a manual, pure-argmax decode loop (bypassing
/// `Sampler`) and returns each tier's `(token ids, per-step logits)`.
fn run_logit_pass(
    gguf_bytes: &[u8],
    tiers: &[KernelTier],
    prompt_ids: &[u32],
) -> Vec<TierRun<KernelTier>> {
    let mut per_tier = Vec::with_capacity(tiers.len());
    for &tier in tiers {
        let gguf = GgufFile::parse(gguf_bytes).expect("parse real legacy GGUF");
        let mut engine = engine_for_tier(&gguf, tier);

        let (prefill, fallback) =
            run_detecting_prefill_fallback(|| engine.prefill_from_pos(prompt_ids, 0));
        let mut logits = prefill.expect("prefill_from_pos");
        if tier == KernelTier::Gpu {
            assert_fused_gpu_prefill(&engine, fallback.as_deref(), "LOGIT PASS");
        }
        let mut generated = Vec::with_capacity(MAX_TOKENS);
        let mut all_logits = Vec::with_capacity(MAX_TOKENS);
        for step in 0..MAX_TOKENS {
            let tok = argmax_first(&logits);
            generated.push(tok);
            all_logits.push(logits.clone());
            let pos = prompt_ids.len() + step;
            logits = engine.decode_step(tok, pos).expect("decode_step");
        }
        per_tier.push(TierRun {
            tier,
            tokens: generated,
            logits: all_logits,
        });
    }
    per_tier
}

/// Runs the full parity gate for one legacy model: for each of its golden
/// cases, the LOGIT pass (D-6's cross-tier comparison against the Reference
/// tier — token chain first, then bit-exact for a CPU tier / the relative
/// bound for the GPU tier) and then the TEXT pass (per tier, vs. the golden
/// Metal capture).
///
/// LOGIT BEFORE TEXT, deliberately: `tiers` always starts with
/// `KernelTier::Reference`, so a text-then-logit order would let a text-pass
/// mismatch on the very first tier panic out of this function before the
/// logit pass — the cross-tier numeric verdict for that case — ever ran.
/// Running the logit pass first means a cross-tier numeric claim is always
/// either earned (the logit pass actually completes and its verdict is
/// real) or superseded by a logit-pass panic that reports the true failure
/// — never asserted without having been checked.
fn run_model_gate(model_file: &'static str, test_name: &str) {
    let _tier_env = TierEnvGuard::cleared();
    let gate_start = std::time::Instant::now();
    // Serialized against this binary's other real-model gates — see
    // [`gpu_serial`]. Taken before the fixture probes so the whole gate,
    // including its capability bookkeeping, is one critical section.
    let _gpu = gpu_serial();

    // D-6(4): these tests are NOT `#[ignore]`d — they
    // run whenever the models are present and self-skip (reported, never
    // green) when they are not. Both probes below are missing-FIXTURE skips,
    // not missing-Metal-hardware skips, hence `Capability::LegacyModels`:
    // recording them as `Metal` let an unrelated Metal-hardware record
    // elsewhere in the manifest (e.g. `metal_k_quant_gemv_parity.rs`) satisfy
    // the release gate's `metal` check regardless of whether this gate ever
    // ran. `find_model` is one `fs::metadata` call, so a models-less host
    // pays nothing for the `#[ignore]` removal.
    let Some(model_path) = find_model(model_file) else {
        eprintln!("skip: {model_file} not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };
    let Some(tokenizer_path) = find_model("tokenizer.json") else {
        eprintln!("skip: tokenizer.json not found under {:?}", models_dir());
        record_skipped(Capability::LegacyModels, test_name);
        return;
    };

    let tokenizer_path_str = tokenizer_path
        .to_str()
        .expect("models/ path is valid UTF-8");
    let tokenizer =
        TokenizerBridge::from_file(tokenizer_path_str).expect("load real tokenizer.json");
    let gguf_bytes = std::fs::read(&model_path).expect("read real legacy GGUF");

    let tiers: Vec<KernelTier> = {
        let mut v = vec![KernelTier::Reference, cpu_kernel_tier(), KernelTier::Gpu];
        v.dedup();
        v
    };

    for case in GOLDEN_CASES.iter().filter(|c| c.model_file == model_file) {
        // ---- Logit pass FIRST: D-6's cross-tier comparison (token chain
        // first, then bit-exact for CPU / relative bound for GPU) over the
        // real tokenized prompt. See this function's doc comment for why
        // this now runs before the text pass. ----
        let prompt_ids = tokenizer.encode(case.prompt).expect("encode golden prompt");
        let per_tier = run_logit_pass(&gguf_bytes, &tiers, &prompt_ids);
        {
            let (reference, others) = per_tier
                .split_first()
                .expect("the tier matrix always starts with KernelTier::Reference");
            let context = format!("LOGIT PASS {model_file} prompt {:?}", case.prompt);
            for other in others {
                compare_against_reference(&context, reference, other, is_cpu_tier);
            }
        }

        // ---- Text pass: real tokenizer + real SamplingParams, per tier.
        // `per_tier` (just computed above) is threaded through so a text
        // mismatch's panic can report the matching tier's own logit-pass
        // step data (top-2 gap / top-5) instead of only the decoded
        // strings — see `assert_exact_match_golden`. ----
        for &tier in &tiers {
            let logit_run = per_tier.iter().find(|t| t.tier == tier);
            run_text_pass(&gguf_bytes, tier, &tokenizer, case, logit_run);
        }
    }

    // T-05 honesty: the `executed: true` records are written HERE, once the
    // whole matrix — every golden case, every tier, both passes — has
    // actually completed. Writing them on fixture PRESENCE instead would
    // claim coverage for work that may still panic, which is the exact
    // false-green class the capability manifest exists to eliminate. Any
    // assertion above panics out of this function and neither record is
    // written.
    let elapsed = gate_start.elapsed();
    record_executed_timed(Capability::Metal, test_name, elapsed);
    record_executed_timed(Capability::LegacyModels, test_name, elapsed);
}

#[test]
fn ternary_1_7b_greedy_text_matches_golden_across_tiers() {
    run_model_gate(
        LEGACY_MODEL_FILES[0],
        "oxibonsai-runtime::legacy_parity_tests::ternary_1_7b_greedy_text_matches_golden_across_tiers",
    );
}

#[test]
fn ternary_8b_greedy_text_matches_golden_across_tiers() {
    run_model_gate(
        LEGACY_MODEL_FILES[1],
        "oxibonsai-runtime::legacy_parity_tests::ternary_8b_greedy_text_matches_golden_across_tiers",
    );
}

#[test]
fn bonsai_8b_greedy_text_matches_golden_across_tiers() {
    run_model_gate(
        LEGACY_MODEL_FILES[2],
        "oxibonsai-runtime::legacy_parity_tests::bonsai_8b_greedy_text_matches_golden_across_tiers",
    );
}

// ═════════════════════════════════════════════════════════════════════════
//  A non-degenerate tiny fixture for logit-shape settings
// ═════════════════════════════════════════════════════════════════════════

/// `Qwen3Config::tiny_test()` combined with an all-zero-weight model
/// produces an all-zero logit vector, under which `apply_repetition_penalty`
/// (`logit >= 0 ? logit / penalty : logit * penalty`) is the identity at
/// every logit regardless of `penalty` — a setting test built on that
/// fixture alone cannot observe the setting do anything. This test instead
/// loads `oxibonsai_testkit::gguf_fixture::tiny_dense_qwen3_gguf` — the
/// same size budget as `tiny_test()`, but with real, non-zero, deterministic
/// weights, through the real `BonsaiModel::from_gguf` — and checks both that
/// its logits are genuinely non-degenerate and that the real repetition
/// penalty actually moves them, needing no model file and no real weights.
#[test]
fn tiny_dense_qwen3_fixture_produces_non_degenerate_logits_the_repetition_penalty_can_move() {
    let _tier_env = TierEnvGuard::cleared();
    let bytes = tiny_dense_qwen3_gguf(0xF00D_BEEF).expect("build tiny dense qwen3 fixture");
    let gguf = GgufFile::parse(&bytes).expect("parse tiny dense qwen3 fixture");
    let mut model = BonsaiModel::from_gguf(&gguf, 64).expect("load tiny dense qwen3 fixture");
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);

    let logits = model.forward(0, 0, &kernel).expect("forward");
    assert!(
        logits.iter().any(|&v| v != 0.0),
        "the tiny dense qwen3 fixture's logits must not be all-zero"
    );
    let first = logits[0];
    assert!(
        logits.iter().any(|&v| v != first),
        "the fixture's logits must not be a degenerate constant vector either"
    );

    // Covering every vocabulary index guarantees the penalty is applied to
    // at least one non-zero logit, which a penalty != 1.0 always moves.
    let history: Vec<u32> = (0..logits.len() as u32).collect();
    let mut penalized = logits.clone();
    apply_repetition_penalty(&mut penalized, &history, 1.3);
    assert_ne!(
        logits, penalized,
        "the repetition penalty must change at least one logit on a non-degenerate vector"
    );
}
