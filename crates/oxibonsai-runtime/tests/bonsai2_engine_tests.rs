//! Real Bonsai 2 27B end to end through the ordinary [`InferenceEngine`]
//! (ENGINE-SEAM acceptance (a)).
//!
//! `B2-11` proved the hybrid forward against the PrismML fork at the
//! *model* level. This file proves the product path: the same
//! `InferenceEngine::from_gguf` the CLI and the server call loads a `qwen35`
//! file as a hybrid engine, the GGUF-embedded tokenizer (`pre = qwen35`)
//! produces exactly the fork's prompt tokens, and `InferenceEngine::generate`
//! at temperature 0 produces exactly the fork's 24 greedy tokens — and hence
//! byte-identical text — for all three golden prompts.
//!
//! # Oracles
//!
//! `$OXI_BONSAI2_GOLDEN_DIR` is the fork's golden directory (`golden2`):
//!
//! * `Ternary-Bonsai-2-27B-<quant>.prompt{i}.prompt_tokens.txt` — the fork's
//!   own tokenisation dump (`llama-completion --verbose-prompt`);
//! * `PQ2_0.prompt{i}.server.json` — `llama-server /completion` at
//!   temperature 0 / top-k 1, 24 tokens: `completion_probabilities[].id` and
//!   `content`;
//! * `Ternary-Bonsai-2-27B-<quant>.prompt{i}.txt` — the fork's 32-token
//!   raw continuation, whose first 24 tokens' text must be a byte prefix.
//!
//! The PTQ1_0 file is checked against the same PQ2_0 server ids: the fork
//! produces byte-identical output for both quantisations on all three
//! prompts (`bonsai2-design.md` §7.0.2), which its own `.txt` dumps confirm.
//!
//! # Skips
//!
//! Model files live outside the repository. Each case reads its paths from
//! the environment (`OXI_BONSAI2_PQ2_GGUF` / `OXI_BONSAI2_PTQ1_GGUF` and
//! `OXI_BONSAI2_GOLDEN_DIR`) and, when a variable is unset, prints a
//! capability report and records a `executed: false` capability entry —
//! never a silent pass, never an `#[ignore]`. A set variable pointing at a
//! missing file is a hard failure.
//!
//! # Executor
//!
//! `InferenceEngine::from_gguf` is `--backend auto`: on a Metal host it
//! decodes the 27B on the Metal hybrid runner, elsewhere on the CPU model —
//! this gate checks whichever the product path picks against the fork
//! (`bonsai2_metal_engine_tests` pins each executor and compares them).
//! `OXIBONSAI_KERNEL_TIER` is cleared for the run: the CPU model's `PQ2_0`
//! GEMV honours it (K-14), which would move a CPU run off the fork's ids.
//!
//! # Memory
//!
//! The GGUF is memory-mapped (never materialised as f32), the KV window is
//! 4096 tokens (256 MiB of f16 KV for the 27B), and the two real-model cases
//! share one process-wide lock so they never hold two 27B mappings at once.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV;
use oxibonsai_runtime::engine::{tokenizer_from_gguf, InferenceEngine};
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};

/// The three raw (un-templated) golden prompts of `make_golden2.sh`.
const PROMPTS: [&str; 3] = [
    "The capital of Japan is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
];

/// Tokens the fork's server dump carries per prompt.
const GOLDEN_STEPS: usize = 24;

/// KV window for the run (well above the longest prompt + 24).
const MAX_SEQ_LEN: usize = 4096;

/// One real 27B mapping at a time (peer-process RSS budget).
static REAL_MODEL_LOCK: Mutex<()> = Mutex::new(());

/// Holds [`REAL_MODEL_LOCK`] and keeps `OXIBONSAI_KERNEL_TIER` cleared for
/// one gate, restoring its prior value on drop (also while unwinding).
struct RealModelRun {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior_tier: Option<String>,
}

impl RealModelRun {
    fn start() -> Self {
        let lock = REAL_MODEL_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior_tier = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: every test in this binary that touches the environment
        // holds `REAL_MODEL_LOCK` first, so no other thread reads or writes
        // it concurrently.
        unsafe { std::env::remove_var(KERNEL_TIER_ENV) };
        Self {
            _lock: lock,
            prior_tier,
        }
    }
}

impl Drop for RealModelRun {
    fn drop(&mut self) {
        // SAFETY: still under `REAL_MODEL_LOCK` (see `start`).
        unsafe {
            match &self.prior_tier {
                Some(v) => std::env::set_var(KERNEL_TIER_ENV, v),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: GOLDEN_STEPS,
    }
}

/// `$var` as a path, or `None` (with a capability report) when unset/empty.
fn env_path(var: &str, test: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => {
            eprintln!(
                "capability report: {test} SKIPPED — ${var} is not set (point it at the real \
                 Bonsai 2 27B file / golden directory to run this acceptance case)"
            );
            record_skipped(Capability::Bonsai2Models, test);
            None
        }
    }
}

/// The prompt token ids of a `llama-cli` tokenisation dump, whose lines look
/// like `0.04.926.888 I    760 -> 'The'`.
fn parse_prompt_tokens(dump: &str) -> Vec<u32> {
    dump.lines()
        .filter_map(|line| {
            let (left, _) = line.split_once("->")?;
            left.split_whitespace().next_back()?.parse().ok()
        })
        .collect()
}

/// `(ids, content)` of a fork `llama-server /completion` dump.
fn parse_server_golden(json: &str) -> (Vec<u32>, String) {
    let root: serde_json::Value = serde_json::from_str(json).expect("golden JSON parses");
    let ids = root
        .get("completion_probabilities")
        .and_then(serde_json::Value::as_array)
        .expect("golden carries completion_probabilities")
        .iter()
        .map(|entry| {
            let id = entry
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .expect("step id");
            u32::try_from(id).expect("token id fits u32")
        })
        .collect();
    let content = root
        .get("content")
        .and_then(serde_json::Value::as_str)
        .expect("golden carries content")
        .to_string();
    (ids, content)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Load `model_path` as an engine and check every prompt against the goldens.
fn check_engine_against_goldens(model_path: &Path, golden_dir: &Path, quant: &str) {
    assert!(
        model_path.is_file(),
        "{} is set but is not a file",
        model_path.display()
    );
    assert!(
        golden_dir.is_dir(),
        "{} is set but is not a directory",
        golden_dir.display()
    );

    let mmap = mmap_gguf_file(model_path).expect("27B GGUF maps");
    let gguf = GgufFile::parse(&mmap).expect("27B GGUF parses");

    // The product constructor -- the one the CLI and the server call.
    let load_start = std::time::Instant::now();
    let mut engine =
        InferenceEngine::from_gguf(&gguf, greedy_params(), 42, MAX_SEQ_LEN).expect("27B loads");
    eprintln!(
        "[{quant}] engine loaded in {:.1}s: {} | kernel {} | {}",
        load_start.elapsed().as_secs_f64(),
        engine.model_description(),
        engine.kernel_label(),
        engine.effective_tier_reason()
    );
    assert!(
        engine.is_hybrid(),
        "a qwen35 file must load as a hybrid engine"
    );
    assert_eq!(engine.architecture(), "qwen35");
    assert!(engine.dense_model().is_none());
    assert!(
        !engine.uses_fused_gpu_decode(),
        "the dense fused Metal route never applies to a hybrid model"
    );
    eprintln!(
        "[{quant}] executor under --backend auto: {:?}",
        engine.hybrid_backend()
    );
    assert_eq!(engine.vocab_size(), 248_320);
    assert!(
        engine.is_eos(248_046),
        "<|im_end|> (248046) must terminate generation, resolved from the GGUF"
    );

    // The engine's tokenizer: the GGUF-embedded one (`pre = qwen35`).
    let tokenizer = tokenizer_from_gguf(&gguf).expect("GGUF-embedded tokenizer builds");

    for (index, prompt) in PROMPTS.iter().enumerate() {
        let n = index + 1;
        let tokens = tokenizer.encode(prompt).expect("prompt encodes");
        let golden_tokens = parse_prompt_tokens(&read(&golden_dir.join(format!(
            "Ternary-Bonsai-2-27B-{quant}.prompt{n}.prompt_tokens.txt"
        ))));
        assert!(!golden_tokens.is_empty(), "prompt {n}: empty golden dump");
        assert_eq!(
            tokens, golden_tokens,
            "prompt {n}: the engine's tokenizer must reproduce the fork's prompt tokens"
        );

        let (golden_ids, golden_content) = parse_server_golden(&read(
            &golden_dir.join(format!("PQ2_0.prompt{n}.server.json")),
        ));
        assert_eq!(golden_ids.len(), GOLDEN_STEPS, "prompt {n}: golden length");

        engine.reset();
        let started = std::time::Instant::now();
        let generated = engine
            .generate(&tokens, GOLDEN_STEPS)
            .expect("greedy generation");
        let elapsed = started.elapsed().as_secs_f64();
        let text = tokenizer.decode(&generated).expect("continuation decodes");
        eprintln!(
            "[{quant}] prompt {n}: {} prompt + {} generated tokens in {elapsed:.1}s -> {text:?}",
            tokens.len(),
            generated.len()
        );

        assert_eq!(
            generated, golden_ids,
            "prompt {n}: greedy token ids must equal the fork's completion_probabilities[].id"
        );
        assert_eq!(
            text, golden_content,
            "prompt {n}: decoded text must equal the fork server's content byte for byte"
        );
        let raw = read(&golden_dir.join(format!("Ternary-Bonsai-2-27B-{quant}.prompt{n}.txt")));
        assert!(
            raw.starts_with(&text),
            "prompt {n}: the 24-token text must be a byte prefix of the fork's raw \
             continuation {raw:?}"
        );
    }
}

#[test]
fn bonsai2_pq2_engine_greedy_matches_the_fork_goldens() {
    const TEST: &str =
        "oxibonsai-runtime::bonsai2_engine_tests::bonsai2_pq2_engine_greedy_matches_the_fork_goldens";
    let Some(model) = env_path("OXI_BONSAI2_PQ2_GGUF", TEST) else {
        return;
    };
    let Some(golden_dir) = env_path("OXI_BONSAI2_GOLDEN_DIR", TEST) else {
        return;
    };
    let gate_start = std::time::Instant::now();
    let _run = RealModelRun::start();
    check_engine_against_goldens(&model, &golden_dir, "PQ2_0");
    record_executed_timed(Capability::Bonsai2Models, TEST, gate_start.elapsed());
}

#[test]
fn bonsai2_ptq1_engine_greedy_matches_the_fork_goldens() {
    const TEST: &str =
        "oxibonsai-runtime::bonsai2_engine_tests::bonsai2_ptq1_engine_greedy_matches_the_fork_goldens";
    let Some(model) = env_path("OXI_BONSAI2_PTQ1_GGUF", TEST) else {
        return;
    };
    let Some(golden_dir) = env_path("OXI_BONSAI2_GOLDEN_DIR", TEST) else {
        return;
    };
    let gate_start = std::time::Instant::now();
    let _run = RealModelRun::start();
    check_engine_against_goldens(&model, &golden_dir, "PTQ1_0");
    record_executed_timed(Capability::Bonsai2Models, TEST, gate_start.elapsed());
}

/// The golden parsers themselves, proven on inline copies of both fork
/// formats so a parsing mistake can never masquerade as a model divergence.
#[test]
fn golden_parsers_read_both_fork_formats() {
    let dump = "0.04.926.888 I    760 -> 'The'\n\
                0.04.926.892 I   6511 -> ' capital'\n\
                not a token line\n\
                0.04.926.900 I    369 -> ' is'\n";
    assert_eq!(parse_prompt_tokens(dump), vec![760, 6511, 369]);
    assert!(parse_prompt_tokens("").is_empty());

    let server = r#"{"content": " Tokyo.", "completion_probabilities": [
        {"id": 25358, "token": " Tokyo", "top_logprobs": []},
        {"id": 13, "token": ".", "top_logprobs": []}]}"#;
    let (ids, content) = parse_server_golden(server);
    assert_eq!(ids, vec![25358, 13]);
    assert_eq!(content, " Tokyo.");
}
