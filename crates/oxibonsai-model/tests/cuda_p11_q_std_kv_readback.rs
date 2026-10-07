//! CUDA-P11 hardware-parity harness: the Q4_0/Q8_0 CUDA batch prefill with
//! its device -> host K/V read-back, against the sequential per-token
//! host-KV path, on a real re-quantized Ternary-Bonsai-1.7B.
//!
//! The checklist item (quoted from
//! `crates/oxibonsai-kernels/tests/cuda_hybrid_parity_plan.rs`; this file
//! never ticks it):
//!
//! > CUDA-P11 Q4_0/Q8_0 KV read-back: try_cuda_prefill_q_std(kv_readback_out
//! > = Some) at pos_start 0 on a real Q4_0 and a real Q8_0 GGUF, read-back
//! > K/V vs the K/V the sequential host path stores for the same prompt
//! > (cos >= 0.999 per layer and head), then the greedy tokens decoded after
//! > the prefill equal the sequential path's.
//!
//! # The two arms
//!
//! Both arms run on ONE `BonsaiModel` with a GPU-tier dispatcher, separated
//! by `BonsaiModel::reset`, the batch arm first.
//!
//! * **(a) batch** — `forward_prefill(prompt, 0, gpu)` with a prompt of more
//!   than 16 tokens and chunking off. With a Q4_0/Q8_0 LM head
//!   `prefill_dispatch` sends it to `try_cuda_prefill_with_lm_head_q_std`,
//!   which runs `try_cuda_prefill_q_std` with the K/V read-back and stores
//!   the read-back into the host `KvCache` (`store_cuda_kv_readback`).
//! * **(b) sequential** — `BonsaiModel::forward(token, pos, gpu)` per prompt
//!   token, driven from this test: for a Q4_0/Q8_0 LM head every call is the
//!   host-KV path (CUDA GEMV per projection, attention over `self.kv_cache`),
//!   i.e. exactly what the sequential fallback of `forward_prefill` runs.
//!
//! The host K/V of both arms is read through the public
//! `BonsaiModel::kv_cache()` -> `KvCache::{keys_for_owned, values_for_owned}`
//! for every (layer, KV head) over the prompt window, so the per-head clause
//! of the item is covered directly. Then the batch arm decodes
//! [`DECODE_TOKENS`] greedy tokens free-running, the sequential arm is
//! teacher-forced with them (so every step's logits are comparable), and both
//! run a second turn of more than 16 tokens at `pos_start > 0`, which the
//! batch entry point must refuse (F6) and hand to the sequential fallback.
//!
//! # Pass criteria
//!
//! * the batch arm's `forward_prefill` completed exactly one Q4_0/Q8_0 CUDA
//!   batch prefill with its K/V read-back stored (positive evidence, below),
//!   and logged no fallback;
//! * read-back K and V vs sequential K and V: cos >= [`COS_MIN`] for every
//!   (layer, KV head) and for every (layer, KV head, position), and the
//!   sequential K/V is non-zero;
//! * last-token prefill logits and every decode step: cos >= [`COS_MIN`];
//! * the [`DECODE_TOKENS`] greedy tokens are identical;
//! * the second turn's last-token logits: cos >= [`COS_MIN`], the second turn
//!   completed no CUDA batch prefill (the F6 refusal at `pos_start > 0`), and
//!   (when DEBUG events are compiled in) the by-construction refusal event of
//!   the batch entry point was observed;
//! * the sequential arm completed no CUDA batch prefill.
//!
//! # Positive evidence and guards
//!
//! A silent fallback of the batch arm would leave behind the very host K/V
//! the sequential arm writes, so its comparison would pass vacuously. Two
//! guards close that hole:
//!
//! * **before the arms** — the model declares no sliding window (M-17 sends a
//!   windowed model's `forward_prefill` to the sequential path) and the
//!   `OXIBONSAI_FORCE_CPU_DECODE_AFTER` seam, which `BonsaiModel` reads at
//!   load and which forces the CPU path from that position on, is unset;
//!   either one fails the test;
//! * **after the batch prefill** — `BonsaiModel::cuda_q_std_prefill_count`
//!   (the process-wide count of Q4_0/Q8_0 CUDA batch prefills that ran on
//!   the device and stored their K/V read-back into the host cache) advanced
//!   by exactly one. This binary's tests hold [`GPU_SERIAL`], so no other
//!   prefill can interleave. The Q4_0/Q8_0 family's success path latches no
//!   device-KV state (its prefill is a host-KV write, so
//!   `gpu_path_active()` stays false either way), and it logs nothing on
//!   success; this counter is the only public marker that tells it apart
//!   from a fallback.
//!
//! `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL` is not set by this harness. The
//! model logs it once per process (a `std::sync::Once`) at the first F6
//! refusal it sees while the variable is present, and both tests here
//! trigger such refusals, so with the variable set process-wide whichever
//! test refuses first takes the single ERROR line. Setting it from inside a
//! test would need `std::env::set_var` in libtest's multi-threaded process
//! with live CUDA driver threads. The CLI-level check covers the
//! once-per-process line. Running this binary with the variable set still
//! checks the other half in-process: the second turn must complete no CUDA
//! batch prefill and must match the sequential arm, so an override that
//! re-enabled the split prefill would fail here.
//!
//! # Fixtures
//!
//! `models/tb17-q4_0.gguf` / `models/tb17-q8_0.gguf` (or
//! `$OXIBONSAI_MODELS_DIR`), made with `oxibonsai quantize`. That command
//! keeps `output.weight` in FP32 (`ExportConfig::default_fp32_exceptions`),
//! and every CUDA Q4_0/Q8_0 branch dispatches on the LM head's type, so such
//! a file never reaches the path under test. When the named file has a
//! non-Q4_0/Q8_0 LM head this harness therefore builds a variant in
//! `std::env::temp_dir()` from `models/tb17-f32.gguf` (the FP32 source the
//! quantized files were made from) with the same export pipeline, keeping
//! only `token_embd.weight` and `output_norm.weight` in FP32 (1-D tensors
//! stay FP32 by rule), and deletes it afterwards. Without either file the
//! test self-skips and records `executed: false`, as it does without a CUDA
//! device. A fixture that exists but cannot be read, or a fixture build that
//! fails, fails the test: it is a real error, not a missing input.
//!
//! Run (one GPU process at a time; `--nocapture` shows the metrics):
//!
//! ```text
//! cargo test --release -p oxibonsai-model --features native-cuda \
//!   --test cuda_p11_q_std_kv_readback -- --test-threads=1 --nocapture
//! ```

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::CudaGraph;
use oxibonsai_model::export::{
    export_to_gguf_streaming, ExportConfig, ExportError, ExportFormat, TensorPlan,
};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::workspace::{find_model, models_dir};
use oxibonsai_tokenizer::OxiTokenizer;

/// Capability-record name of [`cuda_p11_q4_0_kv_readback_matches_sequential`].
const P11_Q4_0_TEST: &str =
    "oxibonsai-model::cuda_p11_q_std_kv_readback::cuda_p11_q4_0_kv_readback_matches_sequential";
/// Capability-record name of [`cuda_p11_q8_0_kv_readback_matches_sequential`].
const P11_Q8_0_TEST: &str =
    "oxibonsai-model::cuda_p11_q_std_kv_readback::cuda_p11_q8_0_kv_readback_matches_sequential";

/// The FP32 source a Q4_0/Q8_0-LM-head fixture is rebuilt from.
const F32_SOURCE: &str = "tb17-f32.gguf";
/// KV capacity of the model under test; every prompt, decode and second
/// turn fits.
const MAX_SEQ: usize = 512;
/// Greedy tokens compared per prompt (the first from the prefill logits).
const DECODE_TOKENS: usize = 8;
/// Cosine floor of the checklist item.
const COS_MIN: f64 = 0.999;
/// `forward_prefill` hands a GPU prompt of at most this many tokens to the
/// sequential path; every batch prompt (and the second turn) is longer.
const SEQUENTIAL_MAX_TOKENS: usize = 16;
/// Length of the second turn (fed at `pos_start > 0`).
const TURN2_TOKENS: usize = 24;
/// The force-CPU test seam `BonsaiModel` reads once at load (MET-05).
const FORCE_CPU_DECODE_AFTER_ENV: &str = "OXIBONSAI_FORCE_CPU_DECODE_AFTER";

/// English prose for the tokenized prompt.
const PROSE: &str = "The art of growing miniature trees in shallow containers began in China \
                     more than a thousand years ago and was later refined in Japan, where it \
                     became known as bonsai. A carefully tended bonsai can live for centuries, \
                     and gardeners pass the oldest trees from one generation to the next. The \
                     most important rule for a beginner is";

/// Serialises this binary's GPU tests (the CUDA graph, weight cache and
/// device KV caches are process-global).
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

/// A hand-rolled subscriber keeping every DEBUG-or-more-severe event of an
/// `oxibonsai*` module (per-token `fwd_profile` timing events excepted).
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

fn captured<T>(f: impl FnOnce() -> T) -> (T, Vec<CapturedEvent>) {
    let capture = EventCapture::default();
    let events = Arc::clone(&capture.events);
    let out = tracing::subscriber::with_default(capture, f);
    let log = std::mem::take(&mut *events.lock().unwrap_or_else(|e| e.into_inner()));
    (out, log)
}

/// Run `f` and count the Q4_0/Q8_0 CUDA batch prefills it completed with the
/// K/V read-back stored (`BonsaiModel::cuda_q_std_prefill_count`, a
/// process-wide counter; every caller holds [`GPU_SERIAL`], so nothing else
/// in this binary can advance it meanwhile). A counter that went backwards
/// wraps to a huge count, which every check below rejects.
fn counting_q_std_prefills<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = BonsaiModel::cuda_q_std_prefill_count();
    let out = f();
    let after = BonsaiModel::cuda_q_std_prefill_count();
    (out, after.wrapping_sub(before))
}

/// Events showing the Q4_0/Q8_0 batch prefill did not answer: anything
/// `prefill_dispatch` logged, any model-side WARN/ERROR (the q_std dispatch
/// failure is a WARN), and any model-side fallback message.
fn prefill_fallback_events(events: &[CapturedEvent]) -> Vec<&CapturedEvent> {
    events
        .iter()
        .filter(|e| {
            e.module.contains("prefill_dispatch")
                || (e.module.starts_with("oxibonsai_model")
                    && (e.level <= tracing::Level::WARN
                        || e.message.contains("falling back")
                        || e.message.contains("fallback")))
        })
        .collect()
}

fn print_warnings(label: &str, events: &[CapturedEvent]) {
    for e in events.iter().filter(|e| e.level <= tracing::Level::WARN) {
        println!("  {label}: {e}");
    }
}

// ── numeric helpers ───────────────────────────────────────────────────────

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

/// Deterministic `len`-token prompt repeating a `period`-token cycle drawn
/// from a seeded LCG over ids `1000..31000`.
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

fn prompts() -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    match find_model("tokenizer.json") {
        Some(path) => match OxiTokenizer::from_json_file(&path) {
            Ok(tokenizer) => match tokenizer.encode(PROSE) {
                Ok(ids) if ids.len() > SEQUENTIAL_MAX_TOKENS => out.push(("prose".to_owned(), ids)),
                Ok(ids) => println!(
                    "note: the prose prompt tokenized to only {} tokens; prose prompt not run",
                    ids.len()
                ),
                Err(e) => {
                    println!("note: tokenizing the prose prompt failed ({e}); prose prompt not run")
                }
            },
            Err(e) => println!("note: cannot load {path:?} ({e}); prose prompt not run"),
        },
        None => println!(
            "note: tokenizer.json not found under {:?}; prose prompt not run (put tokenizer.json \
             next to the models to include it)",
            models_dir()
        ),
    }
    out.push((
        "cycle5-len17".to_owned(),
        cyclic_prompt(SEQUENTIAL_MAX_TOKENS + 1, 5, 0x5011),
    ));
    out.push(("cycle11-len133".to_owned(), cyclic_prompt(133, 11, 0x5111)));
    out
}

// ── fixture resolution ────────────────────────────────────────────────────

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

/// Build `out` from the all-FP32 GGUF at `source`, quantizing every tensor
/// except `token_embd.weight`, `output_norm.weight` and the 1-D tensors to
/// `format` — the `oxibonsai quantize` pipeline with `output.weight` taken
/// off its FP32 exception list.
fn build_q_std_fixture(source: &Path, out: &Path, format: ExportFormat) -> Result<(), String> {
    let mmap = mmap_gguf_file(source).map_err(|e| format!("mmap {source:?}: {e}"))?;
    let gguf = GgufFile::parse(&mmap).map_err(|e| format!("parse {source:?}: {e}"))?;
    let mut names: Vec<&str> = gguf.tensors.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    let mut plan = Vec::with_capacity(names.len());
    for &name in &names {
        let info = gguf
            .tensors
            .get(name)
            .ok_or_else(|| format!("{source:?}: tensor {name} vanished"))?;
        if info.tensor_type != GgufTensorType::F32 {
            return Err(format!(
                "{source:?}: tensor {name} is {:?}, not F32; the fixture builder needs an \
                 all-FP32 source",
                info.tensor_type
            ));
        }
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        plan.push(TensorPlan::new(name, shape));
    }
    let config = ExportConfig::new(format, "p11-q-std-fixture")
        .with_fp32_layers(vec![
            "token_embd.weight".to_owned(),
            "output_norm.weight".to_owned(),
        ])
        .with_source_metadata(&gguf.metadata)
        .map_err(|e| format!("carry metadata of {source:?}: {e}"))?;
    let file = std::fs::File::create(out).map_err(|e| format!("create {out:?}: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);
    let stats = export_to_gguf_streaming(
        &plan,
        |entry| {
            let bytes = gguf
                .tensor_data(&entry.name)
                .map_err(|e| ExportError::QuantizeError {
                    name: entry.name.clone(),
                    reason: e.to_string(),
                })?;
            let (words, _) = bytes.as_chunks::<4>();
            Ok(words.iter().map(|w| f32::from_le_bytes(*w)).collect())
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

/// The GGUF to test plus, when the harness built it, the guard deleting it.
///
/// `Err` only for missing inputs — no `file` with a `want` LM head and no
/// [`F32_SOURCE`] to build one from — which the caller reports as a skip.
/// A file that exists but cannot be read, and a fixture build that fails,
/// panic: those are real failures, not missing inputs.
fn resolve_fixture(
    file: &str,
    want: GgufTensorType,
    format: ExportFormat,
) -> Result<(PathBuf, Option<TempFixture>), String> {
    if let Some(path) = find_model(file) {
        let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {path:?}: {e}"));
        let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
        match lm_head_type(&gguf) {
            Some(t) if t == want => return Ok((path, None)),
            other => println!(
                "fixture: {file} has a {other:?} LM head; every CUDA {want:?} branch dispatches \
                 on the LM head type, so this file cannot reach the path under test \
                 (`oxibonsai quantize` keeps output.weight FP32)"
            ),
        }
    } else {
        println!("fixture: {file} not found under {:?}", models_dir());
    }
    let Some(source) = find_model(F32_SOURCE) else {
        return Err(format!(
            "no {file} with a {want:?} LM head and no {F32_SOURCE} to build one from under {:?}",
            models_dir()
        ));
    };
    let out = std::env::temp_dir().join(format!(
        "oxibonsai-p11-{want:?}-lmhead-{}.gguf",
        std::process::id()
    ));
    let guard = TempFixture(out.clone());
    let start = Instant::now();
    // `guard` deletes a partial file while this panic unwinds.
    if let Err(e) = build_q_std_fixture(&source, &out, format) {
        panic!("building the {want:?}-LM-head fixture {out:?} from {source:?} failed: {e}");
    }
    println!(
        "fixture: {out:?} built in {:.1} s",
        start.elapsed().as_secs_f64()
    );
    Ok((out, Some(guard)))
}

// ── KV snapshot ───────────────────────────────────────────────────────────

/// Host K/V of every (layer, KV head) over positions `0..n`.
struct KvSnapshot {
    layers: usize,
    heads: usize,
    head_dim: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

fn snapshot_kv(model: &BonsaiModel<'_>, n: usize) -> KvSnapshot {
    let kv = model.kv_cache();
    let (layers, heads) = (kv.num_layers(), kv.num_kv_heads());
    let mut keys = Vec::with_capacity(layers * heads);
    let mut values = Vec::with_capacity(layers * heads);
    for layer in 0..layers {
        for head in 0..heads {
            keys.push(kv.keys_for_owned(layer, head, n));
            values.push(kv.values_for_owned(layer, head, n));
        }
    }
    KvSnapshot {
        layers,
        heads,
        head_dim: kv.head_dim(),
        keys,
        values,
    }
}

/// Worst position of one (layer, head) window: `(position, cos)`.
fn worst_position(a: &[f32], b: &[f32], head_dim: usize) -> (usize, f64) {
    a.chunks_exact(head_dim.max(1))
        .zip(b.chunks_exact(head_dim.max(1)))
        .enumerate()
        .map(|(pos, (x, y))| (pos, cosine(x, y)))
        .fold(
            (0, f64::INFINITY),
            |acc, cur| if cur.1 < acc.1 { cur } else { acc },
        )
}

/// Positions of one (layer, head) window whose cosine is below [`COS_MIN`].
fn positions_below(a: &[f32], b: &[f32], head_dim: usize) -> usize {
    a.chunks_exact(head_dim.max(1))
        .zip(b.chunks_exact(head_dim.max(1)))
        .filter(|(x, y)| cosine(x, y) < COS_MIN)
        .count()
}

/// Compare the batch arm's read-back K/V with the sequential arm's, per
/// (layer, head) window and per (layer, head, position).
fn compare_kv(label: &str, readback: &KvSnapshot, seq: &KvSnapshot, n: usize) -> Vec<String> {
    let mut failures = Vec::new();
    if (readback.layers, readback.heads, readback.head_dim) != (seq.layers, seq.heads, seq.head_dim)
    {
        failures.push(format!(
            "{label}: KV geometry differs: read-back {}x{}x{} vs sequential {}x{}x{}",
            readback.layers, readback.heads, readback.head_dim, seq.layers, seq.heads, seq.head_dim
        ));
        return failures;
    }
    for (what, a_all, b_all) in [
        ("K", &readback.keys, &seq.keys),
        ("V", &readback.values, &seq.values),
    ] {
        let mut min_cos = f64::INFINITY;
        let mut worst = (0usize, 0usize);
        let mut max_delta = 0.0f32;
        let mut below = 0usize;
        // Per-position gate: one bad position among many can keep its
        // window's pooled cosine above the floor.
        let mut worst_triple = (0usize, 0usize, 0usize, f64::INFINITY);
        let mut below_positions = 0usize;
        for (idx, (a, b)) in a_all.iter().zip(b_all.iter()).enumerate() {
            let (layer, head) = (idx / seq.heads, idx % seq.heads);
            if a.len() != n * seq.head_dim || b.len() != n * seq.head_dim {
                failures.push(format!(
                    "{label}: {what}[layer {layer}, head {head}] window length read-back {} / \
                     sequential {} != {n} x {}",
                    a.len(),
                    b.len(),
                    seq.head_dim
                ));
                continue;
            }
            if b.iter().all(|&v| v == 0.0) {
                failures.push(format!(
                    "{label}: sequential {what}[layer {layer}, head {head}] is all zero: the \
                     sequential arm did not write the host KV cache"
                ));
            }
            let c = cosine(a, b);
            max_delta = max_delta.max(max_abs_delta(a, b));
            if c < min_cos {
                min_cos = c;
                worst = (layer, head);
            }
            if c < COS_MIN {
                below += 1;
            }
            let (pos, pos_cos) = worst_position(a, b, seq.head_dim);
            if pos_cos < worst_triple.3 {
                worst_triple = (layer, head, pos, pos_cos);
            }
            below_positions += positions_below(a, b, seq.head_dim);
        }
        let idx = worst.0 * seq.heads + worst.1;
        let (pos, pos_cos) = worst_position(&a_all[idx], &b_all[idx], seq.head_dim);
        println!(
            "{label}: read-back {what} vs sequential {what}: min cos={min_cos:.6} at layer {} \
             head {} (worst position {pos}: cos={pos_cos:.6}), max|delta|={max_delta:.5}, \
             {below}/{} (layer, head) pairs below {COS_MIN}",
            worst.0,
            worst.1,
            a_all.len()
        );
        println!(
            "{label}: read-back {what} per position: worst cos={:.6} at layer {} head {} \
             position {}, {below_positions}/{} (layer, head, position) triples below {COS_MIN}",
            worst_triple.3,
            worst_triple.0,
            worst_triple.1,
            worst_triple.2,
            a_all.len() * n
        );
        if below > 0 {
            failures.push(format!(
                "{label}: {below} (layer, head) pairs of {what} below cos {COS_MIN}; worst \
                 layer {} head {} cos {min_cos:.6}",
                worst.0, worst.1
            ));
        }
        if below_positions > 0 {
            failures.push(format!(
                "{label}: {below_positions} (layer, head, position) triples of {what} below cos \
                 {COS_MIN}; worst layer {} head {} position {} cos {:.6}",
                worst_triple.0, worst_triple.1, worst_triple.2, worst_triple.3
            ));
        }
    }
    failures
}

// ── the two arms ──────────────────────────────────────────────────────────

struct ArmOutcome {
    prefill_logits: Vec<f32>,
    kv: KvSnapshot,
    decode_logits: Vec<Vec<f32>>,
    tokens: Vec<u32>,
    turn2_logits: Vec<f32>,
    prefill_wall: Duration,
}

/// How the batch arm's second turn (at `pos_start > 0`) was dispatched.
struct Turn2Dispatch {
    /// The by-construction refusal event of the batch entry point (DEBUG).
    refusal_seen: bool,
    /// Q4_0/Q8_0 CUDA batch prefills the second turn completed; F6 refuses
    /// every one at `pos_start > 0`, so this must be 0.
    q_std_prefills: u64,
}

/// Arm (a): the Q4_0/Q8_0 CUDA batch prefill with K/V read-back, a
/// free-running greedy decode, then the second turn through
/// `forward_prefill` (refused at `pos_start > 0`, sequential fallback).
/// Returns the outcome and how the second turn was dispatched.
///
/// # Panics
/// When the batch prefill fails, logs a fallback, or did not complete
/// exactly one Q4_0/Q8_0 CUDA batch prefill (the comparison would be
/// vacuous), or when the host KV cache was not filled.
fn run_batch_arm(
    model: &mut BonsaiModel<'_>,
    kernel: &KernelDispatcher,
    prompt: &[u32],
    turn2: &[u32],
    label: &str,
) -> (ArmOutcome, Turn2Dispatch) {
    let n = prompt.len();
    assert!(
        n > SEQUENTIAL_MAX_TOKENS && turn2.len() > SEQUENTIAL_MAX_TOKENS,
        "{label}: prompts of at most {SEQUENTIAL_MAX_TOKENS} tokens never reach the batch path"
    );
    model.reset();
    let start = Instant::now();
    let ((prefill, events), q_std_prefills) =
        counting_q_std_prefills(|| captured(|| model.forward_prefill(prompt, 0, kernel)));
    let prefill_wall = start.elapsed();
    print_warnings(&format!("{label} batch prefill"), &events);
    let prefill_logits =
        prefill.unwrap_or_else(|e| panic!("{label}: batch forward_prefill({n} tokens): {e}"));
    let fallbacks = prefill_fallback_events(&events);
    assert!(
        fallbacks.is_empty(),
        "{label}: the Q4_0/Q8_0 CUDA batch prefill did not answer (the comparison would be \
         vacuous); captured: {:#?}",
        fallbacks.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
    assert_eq!(
        q_std_prefills, 1,
        "{label}: the batch forward_prefill({n} tokens, chunking off) completed {q_std_prefills} \
         Q4_0/Q8_0 CUDA batch prefills with a stored K/V read-back \
         (BonsaiModel::cuda_q_std_prefill_count), not exactly 1: a silent fallback answered, \
         and its host K/V would make the comparison vacuous"
    );
    println!(
        "{label}: positive evidence: the batch forward_prefill completed exactly 1 Q4_0/Q8_0 \
         CUDA batch prefill with a stored K/V read-back"
    );
    let seq_len = model.kv_cache().seq_len();
    assert!(
        seq_len >= n,
        "{label}: after the batch prefill the host KV cache holds {seq_len} positions, not \
         {n}: the read-back was not stored"
    );
    let kv = snapshot_kv(model, n);

    let mut tokens = vec![argmax(&prefill_logits)];
    let mut decode_logits = Vec::with_capacity(DECODE_TOKENS - 1);
    for step in 0..DECODE_TOKENS - 1 {
        let logits = model
            .forward(tokens[step], n + step, kernel)
            .unwrap_or_else(|e| panic!("{label}: batch-arm decode step {step}: {e}"));
        tokens.push(argmax(&logits));
        decode_logits.push(logits);
    }

    let pos2 = n + DECODE_TOKENS - 1;
    let ((turn2_result, turn2_events), turn2_q_std_prefills) =
        counting_q_std_prefills(|| captured(|| model.forward_prefill(turn2, pos2, kernel)));
    print_warnings(&format!("{label} batch-arm second turn"), &turn2_events);
    let turn2_logits =
        turn2_result.unwrap_or_else(|e| panic!("{label}: second turn at pos_start {pos2}: {e}"));
    let refusal_seen = turn2_events.iter().any(|e| {
        e.module.contains("prefill_dispatch")
            && e.message.contains("falling back")
            && e.fields.contains("pos_start")
    });
    (
        ArmOutcome {
            prefill_logits,
            kv,
            decode_logits,
            tokens,
            turn2_logits,
            prefill_wall,
        },
        Turn2Dispatch {
            refusal_seen,
            q_std_prefills: turn2_q_std_prefills,
        },
    )
}

/// Arm (b): `forward` per prompt token (host-KV path), the decode
/// teacher-forced with `forced`, and the second turn token by token.
///
/// # Panics
/// When a step fails, the host KV cache was not filled, or the arm
/// completed a Q4_0/Q8_0 CUDA batch prefill (it must not reach one).
fn run_sequential_arm(
    model: &mut BonsaiModel<'_>,
    kernel: &KernelDispatcher,
    prompt: &[u32],
    turn2: &[u32],
    forced: &[u32],
    label: &str,
) -> ArmOutcome {
    let (outcome, q_std_prefills) = counting_q_std_prefills(|| {
        run_sequential_arm_inner(model, kernel, prompt, turn2, forced, label)
    });
    assert_eq!(
        q_std_prefills, 0,
        "{label}: the sequential arm completed {q_std_prefills} Q4_0/Q8_0 CUDA batch prefills; \
         it must run the per-token host-KV path only"
    );
    outcome
}

fn run_sequential_arm_inner(
    model: &mut BonsaiModel<'_>,
    kernel: &KernelDispatcher,
    prompt: &[u32],
    turn2: &[u32],
    forced: &[u32],
    label: &str,
) -> ArmOutcome {
    let n = prompt.len();
    model.reset();
    let start = Instant::now();
    let mut prefill_logits = Vec::new();
    for (pos, &token) in prompt.iter().enumerate() {
        prefill_logits = model
            .forward(token, pos, kernel)
            .unwrap_or_else(|e| panic!("{label}: sequential forward at pos {pos}: {e}"));
    }
    let prefill_wall = start.elapsed();
    let seq_len = model.kv_cache().seq_len();
    assert!(
        seq_len >= n,
        "{label}: the sequential arm left {seq_len} host KV positions, not {n}"
    );
    let kv = snapshot_kv(model, n);

    let mut tokens = vec![argmax(&prefill_logits)];
    let mut decode_logits = Vec::with_capacity(DECODE_TOKENS - 1);
    for (step, &input) in forced.iter().take(DECODE_TOKENS - 1).enumerate() {
        let logits = model
            .forward(input, n + step, kernel)
            .unwrap_or_else(|e| panic!("{label}: sequential-arm decode step {step}: {e}"));
        tokens.push(argmax(&logits));
        decode_logits.push(logits);
    }

    let pos2 = n + DECODE_TOKENS - 1;
    let mut turn2_logits = Vec::new();
    for (j, &token) in turn2.iter().enumerate() {
        turn2_logits = model
            .forward(token, pos2 + j, kernel)
            .unwrap_or_else(|e| panic!("{label}: sequential second turn at pos {}: {e}", pos2 + j));
    }
    ArmOutcome {
        prefill_logits,
        kv,
        decode_logits,
        tokens,
        turn2_logits,
        prefill_wall,
    }
}

/// Compare the logits, greedy tokens and second turn of the two arms.
fn compare_logits(label: &str, batch: &ArmOutcome, seq: &ArmOutcome) -> Vec<String> {
    let mut failures = Vec::new();
    if batch.prefill_logits.is_empty() || batch.prefill_logits.len() != seq.prefill_logits.len() {
        failures.push(format!(
            "{label}: logits length differs or is empty (batch {}, sequential {})",
            batch.prefill_logits.len(),
            seq.prefill_logits.len()
        ));
        return failures;
    }
    for (arm, out) in [("batch", batch), ("sequential", seq)] {
        let finite = out.prefill_logits.iter().all(|v| v.is_finite())
            && out.turn2_logits.iter().all(|v| v.is_finite())
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
    if prefill_cos < COS_MIN {
        failures.push(format!(
            "{label}: last-token prefill logits cos {prefill_cos:.6} < {COS_MIN} (max|delta| \
             {prefill_delta:.5})"
        ));
    }
    let mut step_cos = Vec::with_capacity(batch.decode_logits.len());
    let mut min_cos = prefill_cos;
    let mut max_delta = prefill_delta;
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
        step_cos.push(format!("{c:.6}"));
        if c < COS_MIN {
            failures.push(format!(
                "{label}: decode step {step} logits cos {c:.6} < {COS_MIN} (max|delta| {d:.5})"
            ));
        }
    }
    let turn2_cos = cosine(&batch.turn2_logits, &seq.turn2_logits);
    let turn2_delta = max_abs_delta(&batch.turn2_logits, &seq.turn2_logits);
    if turn2_cos < COS_MIN {
        failures.push(format!(
            "{label}: second-turn last-token logits cos {turn2_cos:.6} < {COS_MIN} (max|delta| \
             {turn2_delta:.5})"
        ));
    }
    let equal = batch
        .tokens
        .iter()
        .zip(&seq.tokens)
        .filter(|(a, b)| a == b)
        .count();
    let speedup = seq.prefill_wall.as_secs_f64() / batch.prefill_wall.as_secs_f64().max(1e-9);
    println!(
        "{label}: prefill last-token cos={prefill_cos:.6} max|delta|={prefill_delta:.5}; \
         decode-step cos=[{}]; min cos={min_cos:.6} max|delta|={max_delta:.5}; greedy tokens \
         equal {equal}/{DECODE_TOKENS}; second turn cos={turn2_cos:.6} max|delta|=\
         {turn2_delta:.5} (argmax {} vs {}); batch prefill {:.1} ms vs sequential {:.1} ms \
         ({speedup:.1}x)",
        step_cos.join(", "),
        argmax(&batch.turn2_logits),
        argmax(&seq.turn2_logits),
        batch.prefill_wall.as_secs_f64() * 1e3,
        seq.prefill_wall.as_secs_f64() * 1e3,
    );
    println!("  {label}: batch tokens      = {:?}", batch.tokens);
    println!("  {label}: sequential argmax = {:?}", seq.tokens);
    if let Some(i) = batch
        .tokens
        .iter()
        .zip(&seq.tokens)
        .position(|(a, b)| a != b)
    {
        let pick = |out: &ArmOutcome| {
            if i == 0 {
                top2(&out.prefill_logits)
            } else {
                top2(&out.decode_logits[i - 1])
            }
        };
        let (a_top, a1, a2) = pick(batch);
        let (b_top, b1, b2) = pick(seq);
        failures.push(format!(
            "{label}: greedy decode diverges at token {i}: batch {a_top} (margin {:.4}) vs \
             sequential {b_top} (margin {:.4})",
            a1 - a2,
            b1 - b2
        ));
    }
    failures
}

// ── the P11 driver ────────────────────────────────────────────────────────

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

fn run_p11(test: &str, file: &str, want: GgufTensorType, format: ExportFormat, kind: &str) {
    let _gpu = gpu_serial();
    let kernel = match gpu_kernel() {
        Ok(k) => k,
        Err(why) => {
            println!("skip: {test} — {why}");
            record_skipped(Capability::CudaHardware, test);
            return;
        }
    };
    // The only `Err` is a missing input; every other fixture error panics.
    let (path, _fixture_guard) = match resolve_fixture(file, want, format) {
        Ok(found) => found,
        Err(why) => {
            println!("skip: {test} — {why}");
            record_skipped(Capability::CudaHardware, test);
            return;
        }
    };
    // The model reads this seam once at load and, from that position on,
    // sends `forward_prefill` (and every decode step) to the CPU path: the
    // arms would not run the CUDA paths under test. Same parse as the model
    // (`parse_force_cpu_after`): an unparsable value means "never".
    if let Some(after) = std::env::var(FORCE_CPU_DECODE_AFTER_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        panic!(
            "{test}: {FORCE_CPU_DECODE_AFTER_ENV}={after} forces the CPU path from position \
             {after} on, so the CUDA paths under test would not run; unset it to run CUDA-P11"
        );
    }
    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {path:?}: {e}"));
    let gguf = GgufFile::parse(&mmap).unwrap_or_else(|e| panic!("parse {path:?}: {e}"));
    let (model, load_events) = captured(|| BonsaiModel::from_gguf(&gguf, MAX_SEQ));
    let mut model = model.unwrap_or_else(|e| panic!("BonsaiModel::from_gguf({path:?}): {e}"));
    print_warnings("P11 load", &load_events);
    match load_events.iter().find_map(|e| e.lm_head.clone()) {
        Some(head) => assert_eq!(
            head, kind,
            "{test}: the model reports a {head} LM head although the GGUF's is {want:?}"
        ),
        None => println!("P11: the model-load event carried no lm_head field"),
    }
    println!("P11[{kind}]: model {path:?}, LM head {kind}");
    // M-17: a windowed model's `forward_prefill` takes the sequential path,
    // never the CUDA batch prefill under test.
    assert!(
        model.config().sliding_window.is_none(),
        "{test}: {path:?} declares a sliding window ({:?}); forward_prefill would take the \
         sequential path (M-17) and never reach the CUDA batch prefill under test",
        model.config().sliding_window
    );
    model.set_prefill_chunk_tokens(0);
    let debug_compiled =
        tracing::level_filters::STATIC_MAX_LEVEL >= tracing::level_filters::LevelFilter::DEBUG;

    let prompts = prompts();
    let turn2 = cyclic_prompt(TURN2_TOKENS, 7, 0x7011);
    let start = Instant::now();
    // Warm-up, not compared: first-use CUDA initialisation may log.
    if let Some((_, warm)) = prompts.last() {
        model.reset();
        let ((warm_result, warm_events), warm_prefills) =
            counting_q_std_prefills(|| captured(|| model.forward_prefill(warm, 0, &kernel)));
        print_warnings("P11 warm-up", &warm_events);
        if let Err(e) = warm_result {
            println!("P11 warm-up batch prefill returned Err: {e}");
        }
        println!("P11 warm-up: {warm_prefills} Q4_0/Q8_0 CUDA batch prefill(s) completed");
    }
    let mut failures = Vec::new();
    for (name, prompt) in &prompts {
        let label = format!("P11[{kind} {name} n={}]", prompt.len());
        let (batch, dispatch) = run_batch_arm(&mut model, &kernel, prompt, &turn2, &label);
        let seq = run_sequential_arm(&mut model, &kernel, prompt, &turn2, &batch.tokens, &label);
        failures.extend(compare_kv(&label, &batch.kv, &seq.kv, prompt.len()));
        failures.extend(compare_logits(&label, &batch, &seq));
        if dispatch.q_std_prefills != 0 {
            failures.push(format!(
                "{label}: the second turn at pos_start {} completed {} Q4_0/Q8_0 CUDA batch \
                 prefills; the F6 guard must refuse every one at pos_start > 0",
                prompt.len() + DECODE_TOKENS - 1,
                dispatch.q_std_prefills
            ));
        }
        if dispatch.refusal_seen {
            println!(
                "{label}: the second turn (pos_start {}) was refused by the batch entry point and \
                 took the sequential path",
                prompt.len() + DECODE_TOKENS - 1
            );
        } else if debug_compiled {
            failures.push(format!(
                "{label}: no refusal event for the second turn at pos_start {}: the F6 guard \
                 did not fire",
                prompt.len() + DECODE_TOKENS - 1
            ));
        } else {
            println!("{label}: DEBUG events are compiled out; the refusal is not observable");
        }
    }
    model.reset();
    assert!(
        failures.is_empty(),
        "P11[{kind}]: Q4_0/Q8_0 K/V read-back vs sequential host path FAILED:\n{}",
        failures.join("\n")
    );
    println!(
        "P11[{kind}]: PASS ({} prompts; each batch arm completed exactly 1 Q4_0/Q8_0 CUDA batch \
         prefill; K/V cos >= {COS_MIN} per layer and head and per position; logits cos >= \
         {COS_MIN}; {DECODE_TOKENS} greedy tokens identical; second turn refused and equal)",
        prompts.len()
    );
    record_executed_timed(Capability::CudaHardware, test, start.elapsed());
}

/// CUDA-P11 on a Q4_0 GGUF.
#[test]
fn cuda_p11_q4_0_kv_readback_matches_sequential() {
    run_p11(
        P11_Q4_0_TEST,
        "tb17-q4_0.gguf",
        GgufTensorType::Q4_0,
        ExportFormat::Q4_0,
        "Q4_0",
    );
}

/// CUDA-P11 on a Q8_0 GGUF.
#[test]
fn cuda_p11_q8_0_kv_readback_matches_sequential() {
    run_p11(
        P11_Q8_0_TEST,
        "tb17-q8_0.gguf",
        GgufTensorType::Q8_0,
        ExportFormat::Q8_0,
        "Q8_0",
    );
}
