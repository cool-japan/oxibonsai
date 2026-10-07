//! CUDA-P14 / CUDA-P15 hardware-parity harness: the fused CUDA batch prefill
//! (the "F9 view path") against the sequential per-token CUDA path, on the
//! real 8B models.
//!
//! The two checklist items (quoted from
//! `crates/oxibonsai-kernels/tests/cuda_hybrid_parity_plan.rs`; this file
//! never ticks them):
//!
//! * **CUDA-P14** — `cuda_prefill::encode_prefill_layer` (`encode_attn_phase`
//!   with `hidden_in` / `attn_out` views) on Bonsai-8B (Q1_0_g128): last-token
//!   logits vs the sequential per-token CUDA path (cos >= 0.999) and an
//!   identical greedy decode.
//! * **CUDA-P15** — `cuda_prefill::encode_prefill_layer_ternary`
//!   (`encode_attn_phase_tq2` with `hidden_in` / `attn_out` views) on
//!   Ternary-Bonsai-8B, the same comparison.
//!
//! # The two arms
//!
//! Both arms run on ONE `BonsaiModel` with a GPU-tier dispatcher, separated
//! by `BonsaiModel::reset`, the batch arm first.
//!
//! * **(a) batch** — `BonsaiModel::forward_prefill(prompt, 0, gpu)` with a
//!   prompt of more than 16 tokens. On a native-cuda build
//!   `forward_prefill_unchunked` sends such a prompt to
//!   `try_cuda_prefill_with_lm_head` (`prefill_dispatch.rs`), which routes a
//!   Q1 LM head to the Q1 batch prefill and a TQ2 LM head to
//!   `try_cuda_prefill_with_lm_head_ternary` (`forward_cuda/q1.rs`).
//!   Chunking is switched off (`set_prefill_chunk_tokens(0)`), so the whole
//!   prompt is one batch.
//! * **(b) sequential** — `BonsaiModel::forward(token, pos, gpu)` for every
//!   prompt position, driven from this test. That loop is exactly the body
//!   of the private `forward_sequential`, which `forward_prefill` itself runs
//!   for a GPU prompt of at most 16 tokens; every call takes the fused
//!   per-token CUDA layer path and the shared device KV cache. No product
//!   switch forces the sequential path for a long prompt, and none is added
//!   here. `set_prefill_chunk_tokens(1)` would be equivalent (each 1-token
//!   chunk goes straight to `forward`), but a chunk of 2..=16 tokens at
//!   `pos_start > 0` goes through `forward_sequential`, whose host-KV
//!   coherence guard refuses it once the device-KV latch is set; the
//!   diagnostic
//!   [`cuda_q1_chunked_prefill_short_tail_chunk_characterize_cache_rebuild_refusal`]
//!   pins down that shape after a batch chunk (product finding F-3, not a
//!   P item): the chunk planner now merges such a tail into a batched chunk
//!   before it, and the refusal remains for callers that cut their own
//!   windows and for a chunk size of 2..=16 itself, whose windows all take
//!   the per-token path and are refused from the second one on.
//!
//! # What is compared
//!
//! The batch arm decodes [`DECODE_TOKENS`] greedy tokens free-running. The
//! sequential arm is teacher-forced with the batch arm's tokens, so every
//! decode step of the two arms sees the same input and its logits are
//! comparable even after a near-tie; the sequential arm's own argmax at each
//! step is recorded, and the two greedy decodes are identical exactly when
//! those argmaxes equal the batch arm's tokens. Per prompt:
//!
//! * the last-token prefill logits: cosine >= [`COS_MIN`], max|delta| printed;
//! * every decode step's logits: cosine >= [`COS_MIN`] (this is the
//!   prefill -> decode KV handoff: arm (a)'s decode reads the K/V the batch
//!   prefill wrote);
//! * the [`DECODE_TOKENS`] greedy tokens: identical.
//!
//! # Honesty guards
//!
//! * A real device probe (`CudaGraph::global()`) and an undemoted GPU-tier
//!   dispatcher, or the test self-skips and records `executed: false`.
//! * The LM head kind decides the dispatch branch. Each item reads it from
//!   the GGUF header first, and a file whose LM head is not the one the item
//!   needs self-skips with the reason; the model's own load event must then
//!   name the same kind, or the test fails.
//! * Every `oxibonsai*` tracing event of the batch prefill is captured; any
//!   event from `prefill_dispatch` or any model-side "falling back" event
//!   fails the test, because a silent fallback would make the comparison
//!   trivially pass. The batch arm must also leave the device-KV latch set.
//! * Every per-token stretch (the batch arm's decode, the sequential arm's
//!   prompt and its decode) is captured too. A model-side event that shows
//!   the per-token `forward` leaving the fused CUDA layer path for the
//!   host-KV per-block loop fails the test, and every stretch must end with
//!   the device-KV latch set. On CUDA the device KV cache survives `reset`,
//!   so a sequential arm that ran one position on the host KV could
//!   otherwise attend over the batch arm's K/V for that position.
//! * A fixture-build error (the PQ2_0 -> TQ2_0_g128 re-encode below, which
//!   also refuses any `+2` code), an unreadable model file, or a model whose
//!   load event contradicts its GGUF LM-head type fails the test. Only a
//!   missing device, a missing model file, or a file whose LM head cannot
//!   reach the named batch prefill self-skips.
//! * `Capability::CudaHardware` is recorded `executed: true` only after every
//!   comparison of the test passed.
//!
//! Run (one GPU process at a time; `--nocapture` shows the metrics):
//!
//! ```text
//! cargo test --release -p oxibonsai-runtime --features native-cuda \
//!   --test cuda_p14_p15_batch_prefill_vs_sequential -- --test-threads=1 --nocapture
//! ```
//!
//! The binary also holds one diagnostic,
//! [`cuda_q1_chunked_prefill_short_tail_chunk_characterize_cache_rebuild_refusal`],
//! which asserts the F-3 fix (the planner merges a short tail chunk) and the
//! library-level refusal that remains for caller-cut windows, and writes no
//! capability record. To run only the P items, name
//! `cuda_p14_q1_batch_prefill_matches_sequential` or
//! `cuda_p15_tq2_batch_prefill_matches_sequential` with `--exact`.
//!
//! Models come from `oxibonsai_testkit::workspace::find_model`
//! (`models/` or `$OXIBONSAI_MODELS_DIR`): `Bonsai-8B.gguf` for P14,
//! `Ternary-Bonsai-8B.gguf` for P15, and `tokenizer.json` for the prose
//! prompt (without it only the synthetic prompts run). Each item first reads
//! the file's LM-head tensor type, because that type alone picks the
//! dispatch branch. Since 3c9993a, `oxibonsai convert --quant tq2_0_g128`
//! (the `scripts/download_ternary.sh` path) writes TQ2_0_g128 (ggml id 42),
//! which loads as is. Files converted before 3c9993a carry PQ2_0 (id 142),
//! as the current `models/Ternary-Bonsai-8B.gguf` does; the harness accepts
//! both. `BonsaiModel` refuses a PQ2_0 LM head, so for P15 such a file is
//! re-encoded as TQ2_0_g128 in `std::env::temp_dir()` and deleted
//! afterwards. The re-encode is lossless for ternary blocks, and a tensor
//! with any `+2` code (`0b11`) fails the test instead. An
//! `OXIBONSAI_MODELS_DIR` farm whose `Ternary-Bonsai-8B.gguf` links to
//! `Ternary-Bonsai-8B-tq2_0_g128.gguf` avoids the re-encode. Any other
//! LM-head mismatch self-skips with the reason and records
//! `executed: false`.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::quant_prism::{count_plus_two_codes, BlockPQ2_0};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::CudaGraph;
use oxibonsai_model::chunked_prefill::{prefill_chunk_windows, PREFILL_PER_TOKEN_MAX_TOKENS};
use oxibonsai_model::export::{
    export_to_gguf_streaming, ExportConfig, ExportError, ExportFormat, TensorPlan,
};
use oxibonsai_model::model::{gpu_fallback_cache_rebuild_pos, BonsaiModel};
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::workspace::{find_model, models_dir};
use oxibonsai_tokenizer::OxiTokenizer;

/// Capability-record name of [`cuda_p14_q1_batch_prefill_matches_sequential`].
const P14_TEST: &str = "oxibonsai-runtime::cuda_p14_p15_batch_prefill_vs_sequential::\
                        cuda_p14_q1_batch_prefill_matches_sequential";
/// Capability-record name of [`cuda_p15_tq2_batch_prefill_matches_sequential`].
const P15_TEST: &str = "oxibonsai-runtime::cuda_p14_p15_batch_prefill_vs_sequential::\
                        cuda_p15_tq2_batch_prefill_matches_sequential";
/// Full name of the diagnostic
/// [`cuda_q1_chunked_prefill_short_tail_chunk_characterize_cache_rebuild_refusal`],
/// used in its messages only: a diagnostic writes no capability record.
const CHUNK_TAIL_TEST: &str = "oxibonsai-runtime::cuda_p14_p15_batch_prefill_vs_sequential::\
     cuda_q1_chunked_prefill_short_tail_chunk_characterize_cache_rebuild_refusal";

/// KV capacity of the model under test; every prompt plus its decode fits.
const MAX_SEQ: usize = 512;
/// Greedy tokens compared per prompt (the first comes from the prefill
/// logits, the rest from `DECODE_TOKENS - 1` decode steps).
const DECODE_TOKENS: usize = 8;
/// Cosine floor of the checklist items.
const COS_MIN: f64 = 0.999;
/// `forward_prefill` hands a GPU prompt of at most this many tokens to the
/// sequential path (`prefill_dispatch.rs`); every batch-arm prompt is longer.
const SEQUENTIAL_MAX_TOKENS: usize = 16;

/// English prose for the tokenized prompt (about 60 tokens with the Qwen3
/// tokenizer): a natural prompt gives peaked next-token distributions, so a
/// greedy mismatch is a real signal rather than a near-tie.
const PROSE: &str = "The art of growing miniature trees in shallow containers began in China \
                     more than a thousand years ago and was later refined in Japan, where it \
                     became known as bonsai. A carefully tended bonsai can live for centuries, \
                     and gardeners pass the oldest trees from one generation to the next. The \
                     most important rule for a beginner is";

/// Serialises the GPU tests of this binary: the CUDA graph, its weight cache
/// and the device KV cache are process-global, so two tests must never drive
/// them at once (`cargo test` runs tests on parallel threads by default).
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn gpu_serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

// ── tracing capture ───────────────────────────────────────────────────────

/// One captured `oxibonsai*` tracing event.
#[derive(Clone, Debug)]
struct CapturedEvent {
    level: tracing::Level,
    module: String,
    message: String,
    fields: String,
    lm_head: Option<String>,
}

impl std::fmt::Display for CapturedEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{} {}] {} {}",
            self.level, self.module, self.message, self.fields
        )
    }
}

/// Collects an event's `message`, its other fields, and the `lm_head` field
/// of the model-load event.
#[derive(Default)]
struct FieldCollector {
    message: String,
    fields: String,
    lm_head: Option<String>,
}

impl FieldCollector {
    fn push(&mut self, name: &str, value: &str) {
        if !self.fields.is_empty() {
            self.fields.push(' ');
        }
        self.fields.push_str(name);
        self.fields.push('=');
        self.fields.push_str(value);
    }
}

impl tracing::field::Visit for FieldCollector {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        match field.name() {
            "message" => self.message = value.to_owned(),
            "lm_head" => {
                self.lm_head = Some(value.to_owned());
                self.push("lm_head", value);
            }
            name => self.push(name, value),
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let rendered = format!("{value:?}");
        match field.name() {
            "message" => self.message = rendered,
            name => self.push(name, &rendered),
        }
    }
}

/// A hand-rolled `tracing::Subscriber` that keeps every DEBUG-or-more-severe
/// event emitted from an `oxibonsai*` module (the per-token `fwd_profile`
/// timing events excepted) while it is the thread's default subscriber.
#[derive(Clone, Default)]
struct EventCapture {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl tracing::Subscriber for EventCapture {
    fn register_callsite(
        &self,
        _metadata: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }

    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.is_event()
            && *metadata.level() <= tracing::Level::DEBUG
            && metadata.target() != "fwd_profile"
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let metadata = event.metadata();
        let module = metadata.module_path().unwrap_or_default();
        if !module.starts_with("oxibonsai") {
            return;
        }
        let mut collector = FieldCollector::default();
        event.record(&mut collector);
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(CapturedEvent {
                level: *metadata.level(),
                module: module.to_owned(),
                message: collector.message,
                fields: collector.fields,
                lm_head: collector.lm_head,
            });
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Run `f` with an [`EventCapture`] as this thread's subscriber and return
/// its result together with every event it captured.
fn captured<T>(f: impl FnOnce() -> T) -> (T, Vec<CapturedEvent>) {
    let capture = EventCapture::default();
    let events = Arc::clone(&capture.events);
    let out = tracing::subscriber::with_default(capture, f);
    let log = std::mem::take(&mut *events.lock().unwrap_or_else(|e| e.into_inner()));
    (out, log)
}

/// The captured events that show a batch prefill did NOT run on the fused
/// CUDA path: anything `prefill_dispatch` logged (it logs only when a GPU
/// batch path fails or is skipped) and any model-side fallback message.
fn prefill_fallback_events(events: &[CapturedEvent]) -> Vec<&CapturedEvent> {
    events
        .iter()
        .filter(|e| {
            e.module.contains("prefill_dispatch")
                || (e.module.starts_with("oxibonsai_model")
                    && (e.message.contains("falling back")
                        || e.message.contains("fallback")
                        || e.message.contains("unavailable")))
        })
        .collect()
}

/// Print every WARN/ERROR event of `events` under `label`.
fn print_warnings(label: &str, events: &[CapturedEvent]) {
    for e in events.iter().filter(|e| e.level <= tracing::Level::WARN) {
        println!("  {label}: {e}");
    }
}

/// The captured events that show a per-token `forward` left the fused CUDA
/// layer path (device KV) for the host-KV per-block loop: the model's
/// "fused CUDA full-layer forward unavailable, falling back to per-block
/// CPU/GPU dispatch", a block-level "falling back to CPU" message, or
/// anything `prefill_dispatch` logged. The benign "fused CUDA forward+LM
/// head unavailable, trying the next path" step does not match: the next
/// path tried is the fused CUDA layer path, still on the device KV.
fn per_token_fallback_events(events: &[CapturedEvent]) -> Vec<&CapturedEvent> {
    events
        .iter()
        .filter(|e| {
            e.module.contains("prefill_dispatch")
                || (e.module.starts_with("oxibonsai_model")
                    && (e.message.contains("per-block")
                        || e.message.contains("falling back to CPU")))
        })
        .collect()
}

/// Panic when `events` (one per-token stretch of an arm) shows a fallback
/// off the fused per-token CUDA path (see [`per_token_fallback_events`]).
fn assert_no_per_token_fallback(label: &str, stretch: &str, events: &[CapturedEvent]) {
    let fallbacks = per_token_fallback_events(events);
    assert!(
        fallbacks.is_empty(),
        "{label}: the {stretch} left the fused per-token CUDA path for the host-KV loop (the \
         comparison would mix host-KV and device-KV positions); captured: {:#?}",
        fallbacks.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
}

// ── numeric helpers ───────────────────────────────────────────────────────

/// Cosine similarity in f64; two all-zero vectors count as identical, one
/// all-zero vector against a non-zero one as orthogonal.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Index of the largest logit (the first one on a tie; NaN never wins).
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}

/// `(top1 index, top1 value, top2 value)`, the margin diagnostic for a
/// greedy mismatch: a near-zero margin is FP noise at a decision boundary.
fn top2(logits: &[f32]) -> (u32, f32, f32) {
    let top = argmax(logits);
    let second = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != top as usize)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    (
        top,
        logits.get(top as usize).copied().unwrap_or(f32::NAN),
        second,
    )
}

// ── prompts ───────────────────────────────────────────────────────────────

/// A deterministic prompt of `len` tokens that repeats a `period`-token
/// cycle drawn from a seeded LCG over ids `1000..31000` (inside every Qwen3
/// vocabulary). Repetition makes the continuation depend on attention over
/// the earliest positions, which is where a wrong K/V shows first.
fn cyclic_prompt(len: usize, period: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let cycle: Vec<u32> = (0..period)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            1000 + ((state >> 33) % 30_000) as u32
        })
        .collect();
    (0..len).map(|i| cycle[i % period]).collect()
}

/// The prompts every P-item runs: the tokenized prose (when
/// `tokenizer.json` is available), the shortest prompt that takes the batch
/// path (17 tokens), and a longer odd-length one (133 tokens).
fn prompts() -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    match find_model("tokenizer.json") {
        Some(path) => match OxiTokenizer::from_json_file(&path) {
            Ok(tokenizer) => match tokenizer.encode(PROSE) {
                Ok(ids) if ids.len() > SEQUENTIAL_MAX_TOKENS => out.push(("prose".to_owned(), ids)),
                Ok(ids) => println!(
                    "note: the prose prompt tokenized to only {} tokens; prose prompt omitted",
                    ids.len()
                ),
                Err(e) => {
                    println!("note: tokenizing the prose prompt failed ({e}); prose prompt omitted")
                }
            },
            Err(e) => println!("note: cannot load {path:?} ({e}); prose prompt omitted"),
        },
        None => println!(
            "note: tokenizer.json not found under {:?}; prose prompt omitted (only the \
             synthetic prompts run)",
            models_dir()
        ),
    }
    out.push((
        "cycle5-len17".to_owned(),
        cyclic_prompt(SEQUENTIAL_MAX_TOKENS + 1, 5, 0x5014),
    ));
    out.push(("cycle11-len133".to_owned(), cyclic_prompt(133, 11, 0x5015)));
    out
}

// ── the two arms ──────────────────────────────────────────────────────────

/// What one arm produced for one prompt.
struct ArmOutcome {
    /// Last-token logits after the prompt.
    prefill_logits: Vec<f32>,
    /// Logits of each decode step (`DECODE_TOKENS - 1` of them).
    decode_logits: Vec<Vec<f32>>,
    /// The greedy token of the prefill logits and of every decode step.
    tokens: Vec<u32>,
    /// Wall time of the prompt part alone.
    prefill_wall: Duration,
}

/// Arm (a): reset, one `forward_prefill` over the whole prompt (the fused
/// CUDA batch prefill), then a free-running greedy decode.
///
/// Panics when the batch prefill provably did not run on the fused CUDA
/// path (a captured fallback event, or no device-KV latch afterwards).
fn run_batch_arm(
    model: &mut BonsaiModel<'_>,
    kernel: &KernelDispatcher,
    prompt: &[u32],
    label: &str,
) -> ArmOutcome {
    assert!(
        prompt.len() > SEQUENTIAL_MAX_TOKENS,
        "{label}: a {}-token prompt would take the sequential path, not the batch prefill",
        prompt.len()
    );
    model.reset();
    let start = Instant::now();
    let (prefill, events) = captured(|| model.forward_prefill(prompt, 0, kernel));
    let prefill_wall = start.elapsed();
    print_warnings(&format!("{label} batch prefill"), &events);
    let prefill_logits = prefill.unwrap_or_else(|e| {
        panic!(
            "{label}: batch forward_prefill({} tokens) failed: {e}",
            prompt.len()
        )
    });
    let fallbacks = prefill_fallback_events(&events);
    assert!(
        fallbacks.is_empty(),
        "{label}: the batch prefill did not run on the fused CUDA path (the comparison would \
         be vacuous); captured: {:#?}",
        fallbacks.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
    assert!(
        model.gpu_path_active(),
        "{label}: forward_prefill returned without setting the device-KV latch, so the fused \
         CUDA batch prefill did not answer"
    );

    let mut tokens = vec![argmax(&prefill_logits)];
    let mut decode_logits = Vec::with_capacity(DECODE_TOKENS - 1);
    let (decode, decode_events) = captured(|| -> Result<(), String> {
        for step in 0..DECODE_TOKENS - 1 {
            let input = tokens[step];
            let logits = model
                .forward(input, prompt.len() + step, kernel)
                .map_err(|e| format!("decode step {step}: {e}"))?;
            tokens.push(argmax(&logits));
            decode_logits.push(logits);
        }
        Ok(())
    });
    print_warnings(&format!("{label} batch-arm decode"), &decode_events);
    decode.unwrap_or_else(|e| panic!("{label}: batch-arm {e}"));
    assert_no_per_token_fallback(label, "batch arm's decode", &decode_events);
    assert!(
        model.gpu_path_active(),
        "{label}: the batch arm's decode left the fused CUDA per-token path"
    );
    ArmOutcome {
        prefill_logits,
        decode_logits,
        tokens,
        prefill_wall,
    }
}

/// Arm (b): reset, `forward` per prompt token from position 0 (the
/// sequential per-token CUDA path), then a decode teacher-forced with
/// `forced` (the batch arm's greedy tokens).
fn run_sequential_arm(
    model: &mut BonsaiModel<'_>,
    kernel: &KernelDispatcher,
    prompt: &[u32],
    forced: &[u32],
    label: &str,
) -> ArmOutcome {
    assert_eq!(
        forced.len(),
        DECODE_TOKENS,
        "{label}: teacher-forcing needs the batch arm's {DECODE_TOKENS} tokens"
    );
    model.reset();
    let start = Instant::now();
    let (prefill, events) = captured(|| -> Result<Vec<f32>, String> {
        let mut last = Vec::new();
        for (pos, &token) in prompt.iter().enumerate() {
            last = model
                .forward(token, pos, kernel)
                .map_err(|e| format!("sequential forward at pos {pos}: {e}"))?;
        }
        Ok(last)
    });
    let prefill_wall = start.elapsed();
    print_warnings(&format!("{label} sequential prompt"), &events);
    let prefill_logits = prefill.unwrap_or_else(|e| panic!("{label}: {e}"));
    let unavailable = events
        .iter()
        .filter(|e| e.message.contains("unavailable"))
        .count();
    if unavailable > 0 {
        println!(
            "  {label}: {unavailable} per-token step(s) reported a fused CUDA entry point \
             unavailable during the sequential arm"
        );
    }
    assert_no_per_token_fallback(label, "sequential arm's prompt", &events);
    assert!(
        model.gpu_path_active(),
        "{label}: the sequential arm did not run on the fused per-token CUDA path (device-KV \
         latch not set)"
    );

    let mut tokens = vec![argmax(&prefill_logits)];
    let mut decode_logits = Vec::with_capacity(DECODE_TOKENS - 1);
    let (decode, decode_events) = captured(|| -> Result<(), String> {
        for (step, &input) in forced.iter().take(DECODE_TOKENS - 1).enumerate() {
            let logits = model
                .forward(input, prompt.len() + step, kernel)
                .map_err(|e| format!("sequential-arm decode step {step}: {e}"))?;
            tokens.push(argmax(&logits));
            decode_logits.push(logits);
        }
        Ok(())
    });
    print_warnings(&format!("{label} sequential-arm decode"), &decode_events);
    decode.unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_no_per_token_fallback(label, "sequential arm's decode", &decode_events);
    assert!(
        model.gpu_path_active(),
        "{label}: the sequential arm's decode left the fused CUDA per-token path"
    );
    ArmOutcome {
        prefill_logits,
        decode_logits,
        tokens,
        prefill_wall,
    }
}

/// Compare one prompt's two arms, print the metrics, and return every
/// violated criterion (empty on a pass).
fn compare_arms(label: &str, batch: &ArmOutcome, seq: &ArmOutcome) -> Vec<String> {
    let mut failures = Vec::new();
    if batch.prefill_logits.len() != seq.prefill_logits.len() || batch.prefill_logits.is_empty() {
        failures.push(format!(
            "{label}: logits length differs or is empty (batch {}, sequential {})",
            batch.prefill_logits.len(),
            seq.prefill_logits.len()
        ));
        return failures;
    }
    for (arm, out) in [("batch", batch), ("sequential", seq)] {
        let finite = out.prefill_logits.iter().all(|v| v.is_finite())
            && out
                .decode_logits
                .iter()
                .all(|row| row.iter().all(|v| v.is_finite()));
        if !finite {
            failures.push(format!("{label}: the {arm} arm produced non-finite logits"));
        }
    }

    let prefill_cos = cosine(&batch.prefill_logits, &seq.prefill_logits);
    let prefill_delta = max_abs_delta(&batch.prefill_logits, &seq.prefill_logits);
    let mut min_cos = prefill_cos;
    let mut max_delta = prefill_delta;
    let mut step_report = Vec::with_capacity(batch.decode_logits.len());
    for (step, (a, b)) in batch
        .decode_logits
        .iter()
        .zip(&seq.decode_logits)
        .enumerate()
    {
        let c = cosine(a, b);
        let d = max_abs_delta(a, b);
        min_cos = min_cos.min(c);
        max_delta = max_delta.max(d);
        step_report.push(format!("{c:.6}"));
        if c < COS_MIN {
            failures.push(format!(
                "{label}: decode step {step} logits cos {c:.6} < {COS_MIN} (max|delta| {d:.5})"
            ));
        }
    }
    let first_diff = batch
        .tokens
        .iter()
        .zip(&seq.tokens)
        .position(|(a, b)| a != b);
    let equal_count = batch
        .tokens
        .iter()
        .zip(&seq.tokens)
        .filter(|(a, b)| a == b)
        .count();
    let speedup = seq.prefill_wall.as_secs_f64() / batch.prefill_wall.as_secs_f64().max(1e-9);

    println!(
        "{label}: prefill last-token cos={prefill_cos:.6} max|delta|={prefill_delta:.5}; \
         decode-step cos=[{}]; min cos={min_cos:.6} max|delta|={max_delta:.5}; greedy tokens \
         equal {equal_count}/{DECODE_TOKENS}; batch prefill {:.1} ms vs sequential {:.1} ms \
         ({speedup:.1}x)",
        step_report.join(", "),
        batch.prefill_wall.as_secs_f64() * 1e3,
        seq.prefill_wall.as_secs_f64() * 1e3,
    );
    println!("  {label}: batch tokens      = {:?}", batch.tokens);
    println!("  {label}: sequential argmax = {:?}", seq.tokens);

    if prefill_cos < COS_MIN {
        failures.push(format!(
            "{label}: last-token prefill logits cos {prefill_cos:.6} < {COS_MIN} (max|delta| \
             {prefill_delta:.5})"
        ));
    }
    if let Some(i) = first_diff {
        let a_logits = if i == 0 {
            &batch.prefill_logits
        } else {
            &batch.decode_logits[i - 1]
        };
        let b_logits = if i == 0 {
            &seq.prefill_logits
        } else {
            &seq.decode_logits[i - 1]
        };
        let (a_top, a1, a2) = top2(a_logits);
        let (b_top, b1, b2) = top2(b_logits);
        failures.push(format!(
            "{label}: greedy decode diverges at token {i}: batch {a_top} (top1 {a1:.4}, top2 \
             {a2:.4}, margin {:.4}) vs sequential {b_top} (top1 {b1:.4}, top2 {b2:.4}, margin \
             {:.4})",
            a1 - a2,
            b1 - b2
        ));
    }
    failures
}

// ── the P-item driver ─────────────────────────────────────────────────────

/// The LM head's GGUF type: `output.weight`, or `token_embd.weight` for a
/// tied model (the loader's own rule).
fn lm_head_type(gguf: &GgufFile<'_>) -> Option<GgufTensorType> {
    gguf.tensors
        .get("output.weight")
        .or_else(|| gguf.tensors.get("token_embd.weight"))
        .map(|info| info.tensor_type)
}

/// Removes a harness-built fixture when the test ends, pass or fail.
struct TempFixture(PathBuf);

impl Drop for TempFixture {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_file(&self.0) {
            println!("note: could not remove {:?}: {e}", self.0);
        }
    }
}

/// Re-encode a PQ2_0 (ggml id 142) ternary GGUF as TQ2_0_g128 (id 42).
///
/// Files converted before 3c9993a (`oxibonsai convert --quant tq2_0_g128`)
/// carry PQ2_0; since 3c9993a the converter writes TQ2_0_g128 directly, and
/// the harness accepts both. `BonsaiModel` has no PQ2_0 LM-head wrapper
/// (`load_output_weight` refuses it), so a PQ2_0 file cannot load, let
/// alone reach the TQ2 batch prefill. Every PQ2_0 tensor is decoded and
/// re-encoded through the export pipeline as `TQ2_0_g128`; for genuine
/// ternary blocks (codes -1/0/+1, absmax == d) the TQ2 absmax quantizer
/// reproduces the same values exactly. A block holding the `0b11` code
/// (PQ2_0 decodes it as `+2`, which no ternary block can represent) would
/// make the fixture a different model, so every PQ2_0 tensor is scanned
/// first and any `+2` code is an error (as in the P18 and F-M1 harnesses).
/// FP32 tensors stay FP32 (`output_norm.weight` is listed, the other norms
/// are 1-D and FP32 by rule).
fn build_tq2_fixture(source: &Path, out: &Path) -> Result<(), String> {
    let mmap = mmap_gguf_file(source).map_err(|e| format!("mmap {source:?}: {e}"))?;
    let gguf = GgufFile::parse(&mmap).map_err(|e| format!("parse {source:?}: {e}"))?;
    let mut names: Vec<&str> = gguf.tensors.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    let mut plan = Vec::with_capacity(names.len());
    let mut plus_two_tensors = Vec::new();
    let mut plus_two_total = 0u64;
    for &name in &names {
        let info = gguf
            .tensors
            .get(name)
            .ok_or_else(|| format!("{source:?}: tensor {name} vanished"))?;
        if !matches!(
            info.tensor_type,
            GgufTensorType::F32 | GgufTensorType::PQ2_0
        ) {
            return Err(format!(
                "{source:?}: tensor {name} is {:?}; the TQ2 fixture builder handles F32 and \
                 PQ2_0 only",
                info.tensor_type
            ));
        }
        if info.tensor_type == GgufTensorType::PQ2_0 {
            let bytes = gguf
                .tensor_data(name)
                .map_err(|e| format!("{source:?}: read {name}: {e}"))?;
            let blocks = BlockPQ2_0::slice_from_bytes(bytes)
                .map_err(|e| format!("{source:?}: {name} as PQ2_0 blocks: {e}"))?;
            let plus_two: u64 = blocks
                .iter()
                .map(|block| u64::from(count_plus_two_codes(&block.qs)))
                .sum();
            if plus_two > 0 {
                plus_two_total += plus_two;
                plus_two_tensors.push(format!("{name} ({plus_two})"));
            }
        }
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        plan.push(TensorPlan::new(name, shape));
    }
    if !plus_two_tensors.is_empty() {
        return Err(format!(
            "{source:?}: {plus_two_total} `+2` code(s) (0b11) in {} PQ2_0 tensor(s): {}. A \
             TQ2_0_g128 re-encode cannot represent +2, so the fixture would not be the model \
             this item names; point OXIBONSAI_MODELS_DIR at a genuine TQ2_0_g128 file instead",
            plus_two_tensors.len(),
            plus_two_tensors.join(", ")
        ));
    }
    let config = ExportConfig::new(ExportFormat::TernaryG128, "p15-tq2-fixture")
        .with_fp32_layers(vec!["output_norm.weight".to_owned()])
        .with_source_metadata(&gguf.metadata)
        .map_err(|e| format!("carry metadata of {source:?}: {e}"))?;
    let file = std::fs::File::create(out).map_err(|e| format!("create {out:?}: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    let stats = export_to_gguf_streaming(
        &plan,
        |entry| {
            let fail = |reason: String| ExportError::QuantizeError {
                name: entry.name.clone(),
                reason,
            };
            let info = gguf
                .tensors
                .get(&entry.name)
                .ok_or_else(|| fail("tensor vanished".to_owned()))?;
            let bytes = gguf
                .tensor_data(&entry.name)
                .map_err(|e| fail(e.to_string()))?;
            if info.tensor_type == GgufTensorType::F32 {
                let (words, _) = bytes.as_chunks::<4>();
                return Ok(words.iter().map(|w| f32::from_le_bytes(*w)).collect());
            }
            let blocks = BlockPQ2_0::slice_from_bytes(bytes).map_err(|e| fail(e.to_string()))?;
            let mut values = vec![0.0f32; blocks.len() * 128];
            BlockPQ2_0::dequant(blocks, &mut values).map_err(|e| fail(e.to_string()))?;
            Ok(values)
        },
        &config,
        &[],
        &mut writer,
    )
    .map_err(|e| format!("export {out:?}: {e}"))?;
    std::io::Write::flush(&mut writer).map_err(|e| format!("flush {out:?}: {e}"))?;
    println!(
        "fixture: built {out:?} from {source:?}: {} tensors, {} quantized, {} kept FP32",
        stats.num_tensors, stats.quantized_tensors, stats.fp32_tensors
    );
    Ok(())
}

/// The GGUF to load for an item that needs a `want` LM head, plus the
/// guard of a harness-built fixture. A PQ2_0 file is re-encoded when the
/// item needs TQ2_0_g128 (see [`build_tq2_fixture`]); a missing file or any
/// other LM-head mismatch is a reason to skip (`Err`). A file that exists
/// but cannot be read, and any fixture-build error, panic: those are
/// failures of the file or of the export pipeline, not a missing fixture.
fn resolve_model(
    model_file: &str,
    want: GgufTensorType,
) -> Result<(PathBuf, Option<TempFixture>), String> {
    let path = find_model(model_file).ok_or_else(|| {
        format!(
            "{model_file} not found under {:?} (set OXIBONSAI_MODELS_DIR)",
            models_dir()
        )
    })?;
    let head = {
        let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {path:?}: {e}"));
        let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
        lm_head_type(&gguf)
    };
    match head {
        Some(t) if t == want => Ok((path, None)),
        Some(GgufTensorType::PQ2_0) if want == GgufTensorType::TQ2_0_g128 => {
            println!(
                "fixture: {model_file} is PQ2_0 (ggml id 142), which BonsaiModel cannot load as \
                 an LM head; re-encoding it as TQ2_0_g128 (id 42) in the temp dir"
            );
            let out = std::env::temp_dir().join(format!(
                "oxibonsai-p15-tq2-fixture-{}.gguf",
                std::process::id()
            ));
            let guard = TempFixture(out.clone());
            let start = Instant::now();
            if let Err(e) = build_tq2_fixture(&path, &out) {
                panic!(
                    "{model_file}: building the TQ2_0_g128 fixture {out:?} FAILED (a fixture-build \
                     error fails the item; it is not a skip): {e}"
                );
            }
            println!(
                "fixture: {out:?} built in {:.1} s",
                start.elapsed().as_secs_f64()
            );
            Ok((out, Some(guard)))
        }
        other => Err(format!(
            "{model_file} has a {other:?} LM head; this item needs {want:?}, the only LM head \
             that routes forward_prefill to the batch prefill it names"
        )),
    }
}

/// The LM head kind the model-load event reported (`lm_head` field), if any.
fn load_lm_head(events: &[CapturedEvent]) -> Option<String> {
    events.iter().find_map(|e| e.lm_head.clone())
}

/// Probe the device and build an undemoted GPU-tier dispatcher, or say why
/// not.
fn gpu_kernel() -> Result<KernelDispatcher, String> {
    if let Err(e) = CudaGraph::global() {
        return Err(format!("no CUDA device accessible on this host ({e})"));
    }
    // The engine's GPU dispatcher carries the accelerated backend handle
    // (`try_with_tier`); the fused CUDA paths do not need it, so a host
    // whose backend handle cannot be built still runs them through
    // `with_tier`.
    let kernel = match KernelDispatcher::try_with_tier(KernelTier::Gpu) {
        Ok(k) => k,
        Err(e) => {
            println!(
                "note: KernelDispatcher::try_with_tier(Gpu) failed ({e}); using \
                 with_tier(Gpu) without a backend handle"
            );
            KernelDispatcher::with_tier(KernelTier::Gpu)
        }
    };
    if kernel.tier() != KernelTier::Gpu || !kernel.is_gpu_accelerated() {
        return Err(format!(
            "the GPU-tier dispatcher was demoted to {:?}",
            kernel.tier()
        ));
    }
    Ok(kernel)
}

/// Shared body of the P14 and P15 tests.
fn run_batch_vs_sequential_item(
    item: &str,
    test: &str,
    model_file: &str,
    want: GgufTensorType,
    expected_lm_head: &str,
) {
    let _gpu = gpu_serial();
    let kernel = match gpu_kernel() {
        Ok(k) => k,
        Err(why) => {
            println!("skip: {test} — {why}");
            record_skipped(Capability::CudaHardware, test);
            return;
        }
    };
    let (path, _fixture_guard) = match resolve_model(model_file, want) {
        Ok(found) => found,
        Err(why) => {
            println!("skip: {test} — {why}");
            record_skipped(Capability::CudaHardware, test);
            return;
        }
    };
    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {path:?}: {e}"));
    let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
    let (model, load_events) = captured(|| BonsaiModel::from_gguf(&gguf, MAX_SEQ));
    let mut model = model.unwrap_or_else(|e| panic!("BonsaiModel::from_gguf({path:?}): {e}"));
    print_warnings(&format!("{item} load"), &load_events);
    match load_lm_head(&load_events) {
        Some(kind) if kind == expected_lm_head => {
            println!("{item}: {model_file} LM head = {kind}");
        }
        Some(kind) => panic!(
            "{item}: {path:?} declares a {want:?} LM head in its GGUF header, but the model \
             loaded a {kind} LM head; only a {expected_lm_head} LM head routes forward_prefill \
             to the batch prefill this item names"
        ),
        None => println!(
            "{item}: the model-load event carried no lm_head field; relying on the fallback \
             detector alone"
        ),
    }
    model.set_prefill_chunk_tokens(0);

    let prompts = prompts();
    let start = Instant::now();
    // Warm-up, not compared: the first call of each path builds its CUDA
    // graph and uploads the weights, which may log on its own.
    if let Some((_, warm)) = prompts.last() {
        model.reset();
        let (warm_batch, warm_events) = captured(|| model.forward_prefill(warm, 0, &kernel));
        print_warnings(&format!("{item} warm-up batch"), &warm_events);
        if let Err(e) = warm_batch {
            println!("{item}: warm-up batch prefill returned Err: {e}");
        }
        model.reset();
        if let Err(e) = model.forward(warm[0], 0, &kernel) {
            println!("{item}: warm-up single-token forward returned Err: {e}");
        }
    }

    let mut failures = Vec::new();
    for (name, prompt) in &prompts {
        let label = format!("{item}[{name} n={}]", prompt.len());
        let batch = run_batch_arm(&mut model, &kernel, prompt, &label);
        let seq = run_sequential_arm(&mut model, &kernel, prompt, &batch.tokens, &label);
        failures.extend(compare_arms(&label, &batch, &seq));
    }
    model.reset();
    assert!(
        failures.is_empty(),
        "{item}: batch prefill vs sequential per-token CUDA path FAILED:\n{}",
        failures.join("\n")
    );
    println!(
        "{item}: PASS ({} prompts, cos >= {COS_MIN}, {DECODE_TOKENS} greedy tokens identical)",
        prompts.len()
    );
    record_executed_timed(Capability::CudaHardware, test, start.elapsed());
}

/// CUDA-P14: the Q1 batch prefill (`cuda_prefill::encode_prefill_layer`,
/// F9 view path) on Bonsai-8B against the sequential per-token CUDA path.
#[test]
fn cuda_p14_q1_batch_prefill_matches_sequential() {
    run_batch_vs_sequential_item(
        "P14",
        P14_TEST,
        "Bonsai-8B.gguf",
        GgufTensorType::Q1_0_g128,
        "Q1_0_g128",
    );
}

/// CUDA-P15: the TQ2 batch prefill
/// (`cuda_prefill::encode_prefill_layer_ternary`, F9 view path) on
/// Ternary-Bonsai-8B against the sequential per-token CUDA path.
#[test]
fn cuda_p15_tq2_batch_prefill_matches_sequential() {
    run_batch_vs_sequential_item(
        "P15",
        P15_TEST,
        "Ternary-Bonsai-8B.gguf",
        GgufTensorType::TQ2_0_g128,
        "TQ2_0_g128",
    );
}

/// Diagnostic next to P14 (not a checklist item; it writes no capability
/// record): what a chunked `forward_prefill` does with a 133-token prompt at
/// a 120-token chunk, whose plain split would end on a 13-token tail after a
/// device-KV batch chunk.
///
/// # Product finding F-3 (closed at the planner level)
///
/// `forward_prefill_unchunked` sends any GPU window of at most
/// `PREFILL_PER_TOKEN_MAX_TOKENS` (16) tokens to `forward_sequential`
/// (`prefill_dispatch.rs`), which first calls
/// `require_host_kv_coherent(pos_start)` (`model/types/mod.rs`). The fused
/// CUDA batch prefill of a Q1 model (the F9 view path) keeps the prompt's
/// K/V only in the device KV cache and sets the device-KV latch, so a
/// 2..=16-token window after it finds an empty host KV cache and the call
/// returns `gpu_fallback_requires_cache_rebuild` (MET-05,
/// `ModelError::GpuFallbackRequiresCacheRebuild`), although the per-token
/// `forward` that guard protects would itself run on the fused device-KV
/// path. A TQ2 model, whose batch prefill latches the device KV the same
/// way, has the same shape. Before the fix, `set_prefill_chunk_tokens(120)`
/// cut this prompt into `[0, 120)` and `[120, 133)` and the call failed at
/// position 120. Now `prefill_chunk_windows` (`chunked_prefill.rs`), which
/// the model's chunk executor and the runtime's own prefill windows both
/// use, merges a final window of at most 16 tokens into the one before it
/// whenever the chunk exceeds 16, so the prompt is one 133-token window.
///
/// # What is asserted
///
/// * (a) The model-level API path: the plan for 133 tokens at chunk 120 is
///   the single window `[0, 133)`, and `forward_prefill` with
///   `set_prefill_chunk_tokens(120)` SUCCEEDS on the fused CUDA batch
///   prefill (no fallback event, device-KV latch set) and matches the
///   sequential per-token CUDA path by the P14 criteria (cos >= [`COS_MIN`],
///   [`DECODE_TOKENS`] greedy tokens identical).
/// * (b) The library-level limit that remains for callers that cut their
///   own windows: chunking off, `forward_prefill(&prompt[..120], 0)` runs on
///   the fused CUDA batch prefill (no fallback event, latch set), and the
///   following `forward_prefill(&prompt[120..], 120)` returns exactly
///   `gpu_fallback_requires_cache_rebuild` at position 120 (checked with
///   `gpu_fallback_cache_rebuild_pos`).
///
/// If (b) succeeds instead, that library-level limitation is gone and this
/// test FAILS on purpose: it first runs the parity comparison (the manual
/// two-window prefill vs the sequential per-token CUDA path, the P14
/// criteria) and prints the verdict, so the maintainer can turn (b) into a
/// parity check.
#[test]
fn cuda_q1_chunked_prefill_short_tail_chunk_characterize_cache_rebuild_refusal() {
    const TEST: &str = CHUNK_TAIL_TEST;
    const CHUNK: usize = 120;
    let _gpu = gpu_serial();
    let kernel = match gpu_kernel() {
        Ok(k) => k,
        Err(why) => {
            println!("skip: {TEST} — {why} (diagnostic: writes no capability record)");
            return;
        }
    };
    let Some(path) = find_model("Bonsai-8B.gguf") else {
        println!(
            "skip: {TEST} — Bonsai-8B.gguf not found under {:?} (diagnostic: writes no \
             capability record)",
            models_dir()
        );
        return;
    };
    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {path:?}: {e}"));
    let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
    let head = lm_head_type(&gguf);
    if head != Some(GgufTensorType::Q1_0_g128) {
        println!(
            "skip: {TEST} — {path:?} has a {head:?} LM head; only a Q1_0_g128 LM head routes \
             forward_prefill to the Q1 batch prefill (diagnostic: writes no capability record)"
        );
        return;
    }
    let (model, load_events) = captured(|| BonsaiModel::from_gguf(&gguf, MAX_SEQ));
    let mut model = model.unwrap_or_else(|e| panic!("BonsaiModel::from_gguf({path:?}): {e}"));
    print_warnings("chunked-diagnostic load", &load_events);
    let prompt = cyclic_prompt(133, 11, 0x5015);
    let tail = prompt.len() % CHUNK;
    assert!(
        tail > 1 && tail <= SEQUENTIAL_MAX_TOKENS && prompt.len() / CHUNK == 1,
        "the prompt must split plainly into one {CHUNK}-token chunk and a 2..=16-token tail \
         chunk (got {} tokens, tail {tail})",
        prompt.len()
    );
    // This harness's per-token threshold must be the model's.
    const _: () = assert!(SEQUENTIAL_MAX_TOKENS == PREFILL_PER_TOKEN_MAX_TOKENS);
    let label = format!("chunked[chunk={CHUNK} n={} tail={tail}]", prompt.len());

    // Warm-up, not asserted: the first batch prefill builds its CUDA graph
    // and uploads the weights, which may log on its own.
    model.set_prefill_chunk_tokens(0);
    model.reset();
    let (warm, warm_events) = captured(|| model.forward_prefill(&prompt[..CHUNK], 0, &kernel));
    print_warnings(&format!("{label} warm-up batch"), &warm_events);
    if let Err(e) = warm {
        println!("{label}: warm-up batch prefill returned Err: {e}");
    }

    // (a) The model-level API path: the planner folds the 13-token tail, so
    // the whole prompt is one window on the fused CUDA batch prefill.
    let plan = prefill_chunk_windows(prompt.len(), CHUNK);
    assert_eq!(
        plan,
        vec![0..prompt.len()],
        "{label}: the planner must merge the {tail}-token tail into the previous window (F-3)"
    );
    model.set_prefill_chunk_tokens(CHUNK);
    let planned = run_batch_arm(&mut model, &kernel, &prompt, &format!("{label} planned"));
    model.set_prefill_chunk_tokens(0);
    let seq = run_sequential_arm(
        &mut model,
        &kernel,
        &prompt,
        &planned.tokens,
        &format!("{label} planned"),
    );
    let failures = compare_arms(&format!("{label} planned"), &planned, &seq);
    model.reset();
    assert!(
        failures.is_empty(),
        "{label}: forward_prefill with set_prefill_chunk_tokens({CHUNK}) vs the sequential \
         per-token CUDA path FAILED:\n{}",
        failures.join("\n")
    );
    println!(
        "{label}: PASS (a): set_prefill_chunk_tokens({CHUNK}) ran one {}-token window on the \
         fused CUDA batch prefill (cos >= {COS_MIN}, {DECODE_TOKENS} greedy tokens identical)",
        prompt.len()
    );

    // (b) The library-level limit for callers that cut their own windows:
    // chunking off, a 120-token batch window, then the 13-token tail.
    model.set_prefill_chunk_tokens(0);
    model.reset();
    let (head_window, head_events) =
        captured(|| model.forward_prefill(&prompt[..CHUNK], 0, &kernel));
    print_warnings(&format!("{label} manual [0, {CHUNK})"), &head_events);
    if let Err(e) = head_window {
        panic!("{label}: the manual [0, {CHUNK}) batch window failed: {e}");
    }
    // The first window must have run on the fused CUDA batch prefill.
    let fallbacks = prefill_fallback_events(&head_events);
    assert!(
        fallbacks.is_empty(),
        "{label}: the {CHUNK}-token window did not run on the fused CUDA batch prefill, so this \
         diagnostic characterizes nothing; captured: {:#?}",
        fallbacks.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
    assert!(
        model.gpu_path_active(),
        "{label}: no device-KV latch after the manual {CHUNK}-token window: it did not run on \
         the fused CUDA batch prefill"
    );
    let t0 = Instant::now();
    let (manual_tail, tail_events) =
        captured(|| model.forward_prefill(&prompt[CHUNK..], CHUNK, &kernel));
    let manual_wall = t0.elapsed();
    print_warnings(
        &format!("{label} manual [{CHUNK}, {})", prompt.len()),
        &tail_events,
    );

    match manual_tail {
        Err(e) => {
            assert_eq!(
                gpu_fallback_cache_rebuild_pos(&e),
                Some(CHUNK),
                "{label}: the manual tail window failed, but not with the documented \
                 gpu_fallback_requires_cache_rebuild at position {CHUNK}: {e}"
            );
            model.reset();
            println!(
                "{label}: CONFIRMED (b), the library-level limit: after the device-KV batch \
                 window [0, {CHUNK}) a caller-cut {tail}-token window returned `{e}` (MET-05 \
                 host-KV guard in forward_sequential); diagnostic, no capability record"
            );
        }
        Ok(manual_logits) => {
            let mut tokens = vec![argmax(&manual_logits)];
            let mut decode_logits = Vec::with_capacity(DECODE_TOKENS - 1);
            for step in 0..DECODE_TOKENS - 1 {
                let logits = model
                    .forward(tokens[step], prompt.len() + step, &kernel)
                    .unwrap_or_else(|e| panic!("{label}: decode step {step}: {e}"));
                tokens.push(argmax(&logits));
                decode_logits.push(logits);
            }
            let manual_arm = ArmOutcome {
                prefill_logits: manual_logits,
                decode_logits,
                tokens,
                prefill_wall: manual_wall,
            };
            let seq = run_sequential_arm(&mut model, &kernel, &prompt, &manual_arm.tokens, &label);
            let failures = compare_arms(&label, &manual_arm, &seq);
            model.reset();
            let verdict = if failures.is_empty() {
                format!("parity PASSED (cos >= {COS_MIN}, {DECODE_TOKENS} greedy tokens identical)")
            } else {
                format!("parity FAILED:\n{}", failures.join("\n"))
            };
            panic!(
                "{label}: a caller-cut {tail}-token window after the device-KV batch window \
                 [0, {CHUNK}) now SUCCEEDS. The library-level limitation documented on {TEST} \
                 appears fixed: turn arm (b) into a parity check (manual windows vs the \
                 sequential per-token CUDA path, the P14 criteria). This run's {verdict}"
            );
        }
    }
}
