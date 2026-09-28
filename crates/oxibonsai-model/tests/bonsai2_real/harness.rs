//! Shared harness for the real Bonsai 2 27B acceptance gates (B2-11-FIX):
//! locating the model files and the fork's goldens, the capability report,
//! the golden parsers, `log_softmax`/top-N ranking and a GPT-2 byte-level
//! detokeniser built from the GGUF's own vocabulary.
//!
//! # Where the files come from
//!
//! Real-model files are never looked up at a hardcoded path. A gate runs
//! only when its environment variable names the file:
//!
//! | variable | meaning |
//! |---|---|
//! | `OXI_BONSAI2_PQ2_GGUF` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` |
//! | `OXI_BONSAI2_PTQ1_GGUF` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` |
//! | `OXIBONSAI_MODELS_DIR` | a directory holding either file under its release name |
//! | `OXI_BONSAI2_GOLDEN_DIR` | the fork's Metal goldens (`golden2/`) |
//! | `OXI_BONSAI2_GOLDEN_CPU_DIR` | the fork's CPU goldens (`golden_cpu/`) |
//!
//! When a model variable is unset the gate **skips with a capability
//! report** (a `bonsai2-models` / `executed: false` record in the manifest
//! `oxibonsai_testkit::capability::report_path()` names, plus a
//! `CAPABILITY-REPORT` line on stderr) — unless `OXI_REQUIRE_MODEL_FILES=1`,
//! which turns the absence into a hard failure so CI that mounts the weights
//! can demand that the gate really ran.
//!
//! The goldens are vendored under `tests/fixtures/bonsai2_golden/` and
//! `tests/fixtures/bonsai2_golden_cpu/` — the fork's own `llama-server`
//! per-step dumps and `llama-cli` prompt-token / text dumps, byte-for-byte
//! except one field: each server dump's `"model"` value, the capture
//! machine's absolute GGUF path, is reduced to the release file name — so
//! the oracle lives in the repository rather than only in a session
//! scratchpad, and no local path does. The environment variables above take
//! precedence when set; `hybrid_vendored_fork_goldens_are_complete_and_consistent_bonsai2`
//! checks the vendored copy on every run.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use oxibonsai_core::gguf::reader::GgufFile;

/// `OXI_BONSAI2_PQ2_GGUF`.
pub const PQ2_ENV: &str = "OXI_BONSAI2_PQ2_GGUF";
/// `OXI_BONSAI2_PTQ1_GGUF`.
pub const PTQ1_ENV: &str = "OXI_BONSAI2_PTQ1_GGUF";
/// `OXI_BONSAI2_GOLDEN_DIR`.
pub const GOLDEN_ENV: &str = "OXI_BONSAI2_GOLDEN_DIR";
/// `OXI_BONSAI2_GOLDEN_CPU_DIR`.
pub const GOLDEN_CPU_ENV: &str = "OXI_BONSAI2_GOLDEN_CPU_DIR";
/// `OXIBONSAI_MODELS_DIR`.
pub const MODELS_DIR_ENV: &str = "OXIBONSAI_MODELS_DIR";
/// `OXI_REQUIRE_MODEL_FILES` — `1` turns a missing model into a failure.
pub const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
/// Release file name of the `PQ2_0` build.
pub const PQ2_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
/// Release file name of the `PTQ1_0` build.
pub const PTQ1_FILE: &str = "Ternary-Bonsai-2-27B-PTQ1_0.gguf";
/// Capability name these gates record under (see the module docs).
pub const CAPABILITY: &str = "bonsai2-models";

/// The three raw prompts of the golden set (no chat template).
pub const PROMPTS: [&str; 3] = [
    "The capital of Japan is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
];

/// Top-N width the fork's server reported (`n_probs = 10`).
pub const TOP_N: usize = 10;

/// Ruling R2' (`pkg/wave4b.rulings.md`): the worst `|Δlogprob|` over the
/// common top-10 at every step must stay at or below this absolute bound —
/// itself below the fork's own Metal-vs-CPU spread (1e-2..6e-2, 2e-1 at one
/// near-tie).
pub const G4_LOGPROB_BAND: f64 = 2.0e-2;

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Whether `OXI_REQUIRE_MODEL_FILES=1`.
#[must_use]
pub fn require_model_files() -> bool {
    env_nonempty(REQUIRE_ENV).is_some_and(|v| v.trim() == "1")
}

/// Append one `{"capability", "executed", "test"}` record to the capability
/// manifest, in the schema `scripts/release-gate.sh` documents, with a
/// single `write` so concurrent producers never interleave mid-line.
///
/// `oxibonsai_testkit::capability::Capability` has no Bonsai 2 variant yet
/// (that enum belongs to a later package), so the record is written here
/// with the same format and the dedicated capability name [`CAPABILITY`].
pub fn record_capability(executed: bool, test_name: &str) {
    let path = oxibonsai_testkit::capability::report_path();
    let line = serde_json::json!({
        "capability": CAPABILITY,
        "executed": executed,
        "test": test_name,
    })
    .to_string()
        + "\n";
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && std::fs::create_dir_all(parent).is_err() {
            eprintln!("capability report: cannot create {}", parent.display());
            return;
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut file) => {
            if let Err(e) = file.write_all(line.as_bytes()) {
                eprintln!("capability report: write to {} failed: {e}", path.display());
            }
        }
        Err(e) => eprintln!("capability report: open {} failed: {e}", path.display()),
    }
    eprintln!("CAPABILITY-REPORT capability={CAPABILITY} executed={executed} test={test_name}");
}

/// Serialise this binary's real-27B gates.
///
/// Each gate maps a 5.9–7.2 GB GGUF and decodes for minutes. Under libtest's
/// default parallelism the three of them would map ~20 GB at once — the
/// line past which this machine's peer process kills a test (ruling R3: one
/// real-27B process at a time) — and fight over the same cores. The gate
/// command already passes `--test-threads=1`; this lock makes a plain
/// `cargo test` just as safe. Take it only after [`locate_model`] found the
/// file, so a skipping gate never waits. A poisoned lock (a sibling gate
/// panicked) is recovered rather than propagated: one red gate must not turn
/// the others red.
pub fn real_model_serial() -> std::sync::MutexGuard<'static, ()> {
    static REAL_MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    REAL_MODEL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Resolve a real-model file from `env_var` (a path) or from
/// `OXIBONSAI_MODELS_DIR/<file_name>`.
///
/// Returns `None` — after recording the skip — when neither is set, or when
/// the named file does not exist. Panics instead under
/// `OXI_REQUIRE_MODEL_FILES=1`.
#[must_use]
pub fn locate_model(env_var: &str, file_name: &str, test_name: &str) -> Option<PathBuf> {
    let candidate = env_nonempty(env_var)
        .map(PathBuf::from)
        .or_else(|| env_nonempty(MODELS_DIR_ENV).map(|dir| PathBuf::from(dir).join(file_name)));
    let found = candidate.filter(|p| p.is_file());
    if found.is_none() {
        assert!(
            !require_model_files(),
            "{REQUIRE_ENV}=1 but {file_name} was not found: set {env_var} (or \
             {MODELS_DIR_ENV}) to the real file"
        );
        eprintln!("skip {test_name}: {file_name} not located (set {env_var} or {MODELS_DIR_ENV})");
        record_capability(false, test_name);
    }
    found
}

/// The fork's Metal goldens (`golden2/`): `OXI_BONSAI2_GOLDEN_DIR`, else the
/// vendored copy.
#[must_use]
pub fn golden_dir() -> PathBuf {
    env_nonempty(GOLDEN_ENV).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bonsai2_golden"),
        PathBuf::from,
    )
}

/// The fork's CPU goldens (`golden_cpu/`): `OXI_BONSAI2_GOLDEN_CPU_DIR`,
/// else the vendored copy.
#[must_use]
pub fn golden_cpu_dir() -> PathBuf {
    env_nonempty(GOLDEN_CPU_ENV).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bonsai2_golden_cpu"),
        PathBuf::from,
    )
}

/// Read a golden file, panicking with its path on failure.
#[must_use]
pub fn read_golden(dir: &Path, name: &str) -> String {
    let path = dir.join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Read a golden file as raw bytes.
#[must_use]
pub fn read_golden_bytes(dir: &Path, name: &str) -> Vec<u8> {
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// One golden decode step: the token the fork chose and its top-N.
#[derive(Debug, Clone)]
pub struct GoldenStep {
    /// The greedy token the fork emitted.
    pub id: u32,
    /// `(id, logprob)` pairs in the fork's reported order.
    pub top: Vec<(u32, f64)>,
}

/// Parse the fork server's `completion_probabilities` array.
#[must_use]
pub fn parse_golden_steps(json: &str) -> Vec<GoldenStep> {
    let root: serde_json::Value = serde_json::from_str(json).expect("golden JSON parses");
    let entries = root
        .get("completion_probabilities")
        .and_then(serde_json::Value::as_array)
        .expect("golden carries completion_probabilities");
    entries
        .iter()
        .map(|entry| {
            let id = entry
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .expect("step id") as u32;
            let top = entry
                .get("top_logprobs")
                .and_then(serde_json::Value::as_array)
                .expect("top_logprobs")
                .iter()
                .map(|alt| {
                    let alt_id = alt
                        .get("id")
                        .and_then(serde_json::Value::as_u64)
                        .expect("alt id") as u32;
                    let logprob = alt
                        .get("logprob")
                        .and_then(serde_json::Value::as_f64)
                        .expect("alt logprob");
                    (alt_id, logprob)
                })
                .collect();
            GoldenStep { id, top }
        })
        .collect()
}

/// The prompt token ids out of `llama-cli`'s tokenisation dump, whose lines
/// look like `0.04.926.888 I    760 -> 'The'`.
#[must_use]
pub fn parse_prompt_tokens(dump: &str) -> Vec<u32> {
    dump.lines()
        .filter_map(|line| {
            let (left, _) = line.split_once("->")?;
            left.split_whitespace().next_back()?.parse().ok()
        })
        .collect()
}

/// `log_softmax` of a logit row: what the fork server reports as `logprob`.
#[must_use]
pub fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
    let log_sum = sum.ln();
    logits
        .iter()
        .map(|&l| f64::from(l - max) - log_sum)
        .collect()
}

/// The `n` highest-scoring `(id, logprob)` pairs, ties broken by id so the
/// order is deterministic.
#[must_use]
pub fn top_n(logprobs: &[f64], n: usize) -> Vec<(u32, f64)> {
    let mut indexed: Vec<(u32, f64)> = logprobs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (u32::try_from(i).unwrap_or(u32::MAX), lp))
        .collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    indexed.truncate(n);
    indexed
}

/// First-index argmax (the fork's `top_k = 1` tie-break).
#[must_use]
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

// ─────────────────────────────────────────────────────────────────────────
//  Per-step comparison against one golden (ruling R2')
// ─────────────────────────────────────────────────────────────────────────

/// How one decode step compares with one golden step.
#[derive(Debug, Clone)]
pub struct StepComparison {
    /// Our argmax at this step.
    pub ours_id: u32,
    /// The golden's greedy id.
    pub golden_id: u32,
    /// Whether our top-N id *set* equals the golden's.
    pub same_set: bool,
    /// Whether the two top-N lists are in the same order.
    pub same_order: bool,
    /// Worst `|Δlogprob|` over the ids present in both top-N lists.
    pub worst_delta: f64,
    /// Rank (in the golden's list) and id where `worst_delta` occurs.
    pub worst_at: (usize, u32),
    /// Ids in the golden's top-N but not in ours (and vice versa).
    pub missing: Vec<u32>,
    /// Ids in our top-N but not in the golden's.
    pub extra: Vec<u32>,
    /// Golden top-1 minus top-2 logprob gap (the decision margin).
    pub golden_gap: f64,
}

/// Compare our logprob row with one golden step.
#[must_use]
pub fn compare_step(ours_logprobs: &[f64], ours_id: u32, golden: &GoldenStep) -> StepComparison {
    let n = golden.top.len();
    let ours_top = top_n(ours_logprobs, n);
    let ours_ids: Vec<u32> = ours_top.iter().map(|p| p.0).collect();
    let golden_ids: Vec<u32> = golden.top.iter().map(|p| p.0).collect();
    let mut sorted_ours = ours_ids.clone();
    sorted_ours.sort_unstable();
    let mut sorted_golden = golden_ids.clone();
    sorted_golden.sort_unstable();
    let same_set = sorted_ours == sorted_golden;
    let same_order = ours_ids == golden_ids;
    let mut worst_delta = 0.0f64;
    let mut worst_at = (0usize, golden.id);
    for (rank, &(id, golden_lp)) in golden.top.iter().enumerate() {
        if let Some(&ours_lp) = usize::try_from(id)
            .ok()
            .and_then(|i| ours_logprobs.get(i))
            .filter(|_| ours_ids.contains(&id))
        {
            let delta = (ours_lp - golden_lp).abs();
            if delta > worst_delta {
                worst_delta = delta;
                worst_at = (rank, id);
            }
        }
    }
    let missing = golden_ids
        .iter()
        .copied()
        .filter(|id| !ours_ids.contains(id))
        .collect();
    let extra = ours_ids
        .iter()
        .copied()
        .filter(|id| !golden_ids.contains(id))
        .collect();
    let golden_gap = match (golden.top.first(), golden.top.get(1)) {
        (Some(a), Some(b)) => a.1 - b.1,
        _ => f64::INFINITY,
    };
    StepComparison {
        ours_id,
        golden_id: golden.id,
        same_set,
        same_order,
        worst_delta,
        worst_at,
        missing,
        extra,
        golden_gap,
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  GPT-2 byte-level detokeniser from the GGUF vocabulary
// ─────────────────────────────────────────────────────────────────────────

/// GPT-2's `bytes_to_unicode` table, inverted: printable stand-in char →
/// the byte it encodes.
#[must_use]
pub fn gpt2_byte_decoder() -> HashMap<char, u8> {
    let mut direct: Vec<u32> = Vec::with_capacity(256);
    direct.extend(u32::from(b'!')..=u32::from(b'~'));
    direct.extend(0xA1u32..=0xACu32);
    direct.extend(0xAEu32..=0xFFu32);
    let mut map = HashMap::with_capacity(256);
    let mut next_shifted = 0u32;
    for byte in 0u32..256 {
        let code = if direct.contains(&byte) {
            byte
        } else {
            let c = 256 + next_shifted;
            next_shifted += 1;
            c
        };
        if let (Some(ch), Ok(b)) = (char::from_u32(code), u8::try_from(byte)) {
            map.insert(ch, b);
        }
    }
    map
}

/// `tokenizer.ggml.tokens` of a GGUF, as owned strings.
#[must_use]
pub fn gguf_vocab(gguf: &GgufFile<'_>) -> Vec<String> {
    gguf.metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .expect("the GGUF carries tokenizer.ggml.tokens")
        .iter()
        .map(|v| v.as_str().expect("vocab entries are strings").to_string())
        .collect()
}

/// Decode token ids to raw bytes through the byte-level vocabulary: every
/// char of a piece maps back through [`gpt2_byte_decoder`], and a char with
/// no byte mapping (a special token's literal text) contributes its UTF-8.
#[must_use]
pub fn detokenize(vocab: &[String], decoder: &HashMap<char, u8>, ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &id in ids {
        let piece = usize::try_from(id)
            .ok()
            .and_then(|i| vocab.get(i))
            .unwrap_or_else(|| panic!("token id {id} is past the vocabulary"));
        for ch in piece.chars() {
            match decoder.get(&ch) {
                Some(&b) => out.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }
    out
}

/// Render bytes for a failure message, escaping control characters.
#[must_use]
pub fn show_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).escape_debug().to_string()
}
