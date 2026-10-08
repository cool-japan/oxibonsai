//! F-M1 on real hardware — the keyed captured-CUDA-graph slot, two real models
//! in **one process**.
//!
//! The captured CUDA driver graph lives in one process-global slot
//! (`cuda_full_layer::CudaFullLayerState::cuda_driver_graph`) shared by the Q1
//! and the ternary full-forward decode paths. A captured `CUgraphExec` bakes in
//! the per-layer weight device pointers. `Bonsai-8B` (Q1_0_g128) and
//! `Ternary-Bonsai-8B` (TQ2_0_g128) have identical dimensions, so before F-M1
//! the second model reused the activation buffers, skipped invalidation and
//! replayed the **first** model's graph: silently wrong logits, no error. The
//! fix keys the slot on `CudaGraphSlotKey` (`gpu_backend/cuda_graph_slot.rs`)
//! and its consumers (`cuda_full_layer/encode_q1.rs::encode_full_forward`,
//! `encode_ternary.rs::encode_full_forward_ternary`) drop a slot whose key
//! differs and re-capture. The host logic is unit-tested
//! (`cuda_full_layer/tests.rs`); this harness is the hardware half.
//!
//! What it does, in one process:
//!
//! 1. CPU references: both GGUFs greedily decoded on `KernelTier::Reference`
//!    (no CUDA involved), exactly like `cuda_cross_backend_determinism_tests`
//!    builds its engines.
//! 2. Phase 1: `Bonsai-8B` on `KernelTier::Gpu` — captures the Q1 graph.
//! 3. Phase 2: `Ternary-Bonsai-8B` on `KernelTier::Gpu`, with the Q1 engine
//!    (and so its weights and its captured graph) still alive. The slot holds
//!    the Q1 key, the request carries the ternary key, so the ternary path must
//!    drop it and capture its own.
//! 4. Phase 3: `Bonsai-8B` again on its still-alive engine. The slot now holds
//!    the ternary key, so the Q1 path must re-capture once more.
//!
//! Oracles (all must hold):
//!
//! - **Tokens (the F-M1 oracle):** phase 2's greedy tokens equal the ternary
//!   model's CPU-reference tokens. Replaying the Q1 graph would run Bonsai-8B's
//!   layers under the ternary LM head and diverge at once.
//! - **A→B→A:** phase 3's tokens equal phase 1's (same model, same kernels;
//!   only the slot moved in between).
//! - **Capture counter:** the slot exposes no capture counter or key accessor
//!   (`cuda_driver_graph` is a private field of a private struct), so this
//!   harness counts the two capture-success `debug!` events the consumers emit
//!   ("CUDA graph captured and uploaded successfully" from `encode_q1.rs`,
//!   "CUDA ternary graph captured and uploaded" from `encode_ternary.rs`) with a
//!   process-global `tracing` layer. Phase 1 must capture ≥ 1 Q1 graph and no
//!   ternary one, phase 2 ≥ 1 ternary graph and no Q1 one, phase 3 ≥ 1 Q1
//!   graph and no ternary one; no phase may log a per-block fallback ("fused
//!   CUDA full-layer forward unavailable", which would mean the graph path was
//!   never used) or a kernel warning about capture or the full forward. The
//!   counter is **mandatory**: right after installing it the harness emits one
//!   `debug!` probe under an `oxibonsai_kernels` target and requires the layer
//!   to have counted it. When the counter cannot be installed (another global
//!   subscriber, or `debug` compiled out via `tracing`'s `max_level_*` /
//!   `release_max_level_*` features) or misses its probe, the test **fails**
//!   rather than deciding on the token oracles alone: an engine that never
//!   reaches the fused graph path (per-block fallback) decodes its CPU tokens
//!   and repeats its own phase-1 tokens without ever touching the slot, so the
//!   token checks alone can pass vacuously.
//! - **Precondition:** both models must share every buffer dimension (layers,
//!   hidden, heads, head_dim, intermediate); otherwise a re-capture could be
//!   dimension-driven and prove nothing about the key.
//!
//! The token oracles are exact greedy equality, so the prompt is natural
//! English: the first 16 tokens of the prose prompt the P14/P15 harness uses,
//! tokenized with `tokenizer.json` (found like the models). A natural prompt
//! gives peaked next-token distributions, so a mismatch is a real signal and
//! not FP noise at a near-tie. 16 is the longest prompt the CUDA build still
//! prefills token by token through the same full-forward (graph) path as
//! decode (`prefill_dispatch.rs`: `_gpu_kernel && token_ids.len() <= 16`).
//! Without a usable `tokenizer.json` the harness falls back to a synthetic
//! 8-token id ramp and says so in its output.
//!
//! # The ternary file's format
//!
//! The slot's ternary consumer, the CUDA ternary full forward, takes only
//! `TQ2_0_g128` blocks: `oxibonsai-model`'s `forward_cuda/ternary.rs` builds
//! its layer parameters from `Linear::blocks_ternary()`, which is `None` for
//! every other format. Since 3c9993a, `oxibonsai convert … --quant
//! tq2_0_g128` (`scripts/download_ternary.sh`) writes `TQ2_0_g128` (ggml id
//! 42) directly. Files converted before 3c9993a — such as the current
//! `models/Ternary-Bonsai-8B.gguf` — carry **`PQ2_0`** (ggml id 142, `d`
//! first) instead, which loads as `LinearPQ2_0` and so never reaches the slot
//! at all. The harness accepts both. It looks for `Ternary-Bonsai-8B.gguf`
//! first and then for `Ternary-Bonsai-8B-tq2_0_g128.gguf` (the id-42 file a
//! lossless re-encode produces), so a symlink farm under
//! `$OXIBONSAI_MODELS_DIR` and the real file names both work. When the file is
//! `PQ2_0`, the harness re-encodes it **in memory** as `TQ2_0_g128`: every
//! `PQ2_0` tensor's type id becomes 42 and each 34-byte block has its FP16
//! scale moved behind its 32 code bytes. Offsets and sizes are unchanged, and
//! the re-encode is refused (the test self-skips) if any block holds a `0b11`
//! (`+2`) code, the only value the two formats decode differently. The model
//! loader then resolves id 42 by its data sniff as `TQ2_0_g128`, so both the
//! CPU reference and the GPU run use the real Ternary-Bonsai-8B weights in the
//! format the slot's consumer takes. A file already stored as `TQ2_0_g128` is
//! used as is. The testkit's `find_model` now returns the
//! `Ternary-Bonsai-8B-tq2_0_g128.gguf` re-encode itself when
//! `Ternary-Bonsai-8B.gguf` is `PQ2_0` and the re-encode sits beside it, so
//! the in-memory re-encode branch only runs on a checkout without that
//! sibling.
//!
//! The first model's GPU-vs-CPU agreement is printed but is not part of the
//! verdict: it is the subject of `cuda_cross_backend_determinism_tests` /
//! Step 4e, and an unrelated near-tie there must not mask the F-M1 result.
//!
//! Records `cuda-hardware` `executed: true` only after every oracle passed,
//! the per-phase capture counts included; `executed: false` when no CUDA
//! device answers, either GGUF is missing, or the ternary file cannot be used
//! (a `PQ2_0` file with `+2` codes, or neither two-bit format). An unavailable
//! capture counter is a test failure (no record), never a token-only pass.
//! Needs both files (about 3.4 GB) in host RAM. Run with:
//!
//! ```text
//! cargo test --release -p oxibonsai-runtime --features native-cuda \
//!     --test cuda_graph_slot_two_models -- --test-threads=1 --nocapture
//! ```
//!
//! This binary holds exactly one test on purpose: it installs a
//! process-global `tracing` subscriber.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::{count_plus_two_codes, GgufTensorType, Qwen3Config};
use oxibonsai_kernels::dispatch::KernelTier;
use oxibonsai_kernels::CudaGraph;
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_runtime::engine::InferenceEngine;
use oxibonsai_runtime::sampling::SamplingParams;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::workspace::{find_model, models_dir};
use oxibonsai_tokenizer::OxiTokenizer;
use tracing::field::{Field, Visit};
use tracing::level_filters::{LevelFilter, STATIC_MAX_LEVEL};
use tracing::subscriber::Interest;
use tracing::{Event, Level, Metadata, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;

/// Capability-record name of [`fm1_second_model_recaptures_cuda_graph_slot`].
const TEST: &str = "oxibonsai-runtime::cuda_graph_slot_two_models::\
                    fm1_second_model_recaptures_cuda_graph_slot";
/// First model: Q1_0_g128.
const Q1_FILE: &str = "Bonsai-8B.gguf";
/// Second model: TQ2_0_g128, same dimensions (`PQ2_0` is re-encoded in memory).
const TQ2_FILE: &str = "Ternary-Bonsai-8B.gguf";
/// Second model's name when only the id-42 `TQ2_0_g128` file is present.
const TQ2_FILE_NATIVE: &str = "Ternary-Bonsai-8B-tq2_0_g128.gguf";
/// KV capacity, identical for every engine so no buffer reallocation (and so
/// no dimension-driven slot invalidation) can happen between phases.
const MAX_SEQ: usize = 256;
/// Prompt length cap: at most 16, so the CUDA build prefills token by token
/// through the same full-forward (graph) path as decode
/// (`prefill_dispatch.rs`: `_gpu_kernel && token_ids.len() <= 16`). The
/// tokenized prose prompt is cut to exactly this many tokens.
const PROMPT_MAX_TOKENS: usize = 16;
/// Length of the synthetic fallback prompt (no usable `tokenizer.json`).
const SYNTHETIC_PROMPT_LEN: u32 = 8;
/// English prose, the same text the P14/P15 harness tokenizes: a natural
/// prompt gives peaked next-token distributions, so a greedy mismatch is a
/// real signal rather than a near-tie. Only its first [`PROMPT_MAX_TOKENS`]
/// tokens are used here.
const PROSE: &str = "The art of growing miniature trees in shallow containers began in China \
                     more than a thousand years ago and was later refined in Japan, where it \
                     became known as bonsai. A carefully tended bonsai can live for centuries, \
                     and gardeners pass the oldest trees from one generation to the next. The \
                     most important rule for a beginner is";
/// Greedy tokens generated per run.
const N_NEW: usize = 8;

/// `encode_q1.rs::encode_full_forward`'s capture-success event.
const Q1_CAPTURED: &str = "CUDA graph captured and uploaded successfully";
/// `encode_ternary.rs::encode_full_forward_ternary`'s capture-success event.
const TQ2_CAPTURED: &str = "CUDA ternary graph captured and uploaded";
/// `decode.rs`: the fused full-layer forward was refused; per-block dispatch.
const LAYER_FALLBACK: &str = "fused CUDA full-layer forward unavailable";
/// `decode.rs`: the fused forward + LM head was refused; next path tried.
const LM_HEAD_FALLBACK: &str = "fused CUDA forward+LM head unavailable";
/// Target of the counter's self-test probe: inside the `oxibonsai_kernels`
/// namespace the capture events use, so it passes the same filter.
const COUNTER_PROBE_TARGET: &str = "oxibonsai_kernels::fm1_capture_counter_probe";
/// Message of the counter's self-test probe.
const COUNTER_PROBE: &str = "F-M1 capture-counter self-test probe";

// ─────────────────────────────────────────────────────────────────────────────
// Capture counter (a `tracing` layer — the slot itself exposes nothing)
// ─────────────────────────────────────────────────────────────────────────────

/// Event counts observed by [`SlotEventLayer`].
#[derive(Default)]
struct SlotEvents {
    /// [`COUNTER_PROBE`] events (the install-time self-test).
    probes: AtomicUsize,
    q1_captures: AtomicUsize,
    tq2_captures: AtomicUsize,
    layer_fallbacks: AtomicUsize,
    lm_head_fallbacks: AtomicUsize,
    /// Every WARN/ERROR event from `oxibonsai_kernels` / `oxibonsai_model`.
    warnings: Mutex<Vec<String>>,
}

/// A point-in-time copy of [`SlotEvents`].
#[derive(Debug, Clone, Copy, Default)]
struct Snapshot {
    q1: usize,
    tq2: usize,
    layer_fb: usize,
    lm_fb: usize,
    warnings: usize,
}

impl Snapshot {
    /// Events between `earlier` and `self`.
    fn since(self, earlier: Self) -> Self {
        Self {
            q1: self.q1 - earlier.q1,
            tq2: self.tq2 - earlier.tq2,
            layer_fb: self.layer_fb - earlier.layer_fb,
            lm_fb: self.lm_fb - earlier.lm_fb,
            warnings: self.warnings - earlier.warnings,
        }
    }
}

impl SlotEvents {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            q1: self.q1_captures.load(Ordering::SeqCst),
            tq2: self.tq2_captures.load(Ordering::SeqCst),
            layer_fb: self.layer_fallbacks.load(Ordering::SeqCst),
            lm_fb: self.lm_head_fallbacks.load(Ordering::SeqCst),
            warnings: self
                .warnings
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .len(),
        }
    }

    /// Warning lines recorded in `[from, to)`.
    fn warnings_between(&self, from: usize, to: usize) -> Vec<String> {
        let all = self.warnings.lock().unwrap_or_else(PoisonError::into_inner);
        all.get(from..to.min(all.len()))
            .map(<[String]>::to_vec)
            .unwrap_or_default()
    }
}

/// Only events (never spans) at DEBUG or above from the two crates whose
/// messages this harness reads, so the per-layer `#[instrument]` spans of the
/// decode path are never even created.
fn wanted(meta: &Metadata<'_>) -> bool {
    meta.is_event()
        && *meta.level() <= Level::DEBUG
        && (meta.target().starts_with("oxibonsai_kernels")
            || meta.target().starts_with("oxibonsai_model"))
}

/// Extracts an event's `message` field.
#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        }
    }
}

/// The counting layer.
struct SlotEventLayer {
    events: Arc<SlotEvents>,
}

impl<S: Subscriber> Layer<S> for SlotEventLayer {
    fn register_callsite(&self, meta: &'static Metadata<'static>) -> Interest {
        if wanted(meta) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, meta: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        wanted(meta)
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let message = visitor.message;
        if meta.target().starts_with("oxibonsai_kernels") {
            if message == Q1_CAPTURED {
                self.events.q1_captures.fetch_add(1, Ordering::SeqCst);
            } else if message == TQ2_CAPTURED {
                self.events.tq2_captures.fetch_add(1, Ordering::SeqCst);
            } else if message == COUNTER_PROBE {
                self.events.probes.fetch_add(1, Ordering::SeqCst);
            }
        } else if message.starts_with(LAYER_FALLBACK) {
            self.events.layer_fallbacks.fetch_add(1, Ordering::SeqCst);
        } else if message.starts_with(LM_HEAD_FALLBACK) {
            self.events.lm_head_fallbacks.fetch_add(1, Ordering::SeqCst);
        }
        if *meta.level() <= Level::WARN {
            self.events
                .warnings
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(format!("{} {}: {message}", meta.level(), meta.target()));
        }
    }
}

/// Install the counter as the process-global subscriber and prove it works:
/// one `debug!` probe under an `oxibonsai_kernels` target must be counted.
/// `Err(reason)` when the debug level is compiled out, another global
/// subscriber is already installed, or the probe is not seen; the test then
/// fails (it never decides on the token oracles alone).
fn install_event_counter() -> Result<Arc<SlotEvents>, String> {
    if STATIC_MAX_LEVEL < LevelFilter::DEBUG {
        return Err(format!(
            "tracing's static max level is {STATIC_MAX_LEVEL}, so the capture-success debug! \
             events are compiled out"
        ));
    }
    let events = Arc::new(SlotEvents::default());
    let subscriber = Registry::default().with(SlotEventLayer {
        events: Arc::clone(&events),
    });
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| format!("set_global_default failed: {e}"))?;
    tracing::debug!(target: COUNTER_PROBE_TARGET, "{}", COUNTER_PROBE);
    match events.probes.load(Ordering::SeqCst) {
        1 => Ok(events),
        seen => Err(format!(
            "the counter is installed but counted {seen} of 1 self-test debug! probe(s) \
             (target {COUNTER_PROBE_TARGET}) — debug events from oxibonsai_kernels do not reach it \
             (max level now {})",
            LevelFilter::current()
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// In-memory PQ2_0 → TQ2_0_g128 re-encode of the ternary GGUF
// ─────────────────────────────────────────────────────────────────────────────

/// ggml wire id of `PQ2_0` (`d` first).
const PQ2_0_WIRE_ID: u32 = 142;
/// ggml wire id of `TQ2_0_g128` (`qs` first), as the legacy writer emitted it.
const TQ2_0_G128_WIRE_ID: u32 = 42;
/// Bytes per two-bit block in both layouts.
const TWO_BIT_BLOCK_BYTES: usize = 34;

/// A bounds-checked little-endian reader over a GGUF header.
struct HeaderCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> HeaderCursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| format!("GGUF header truncated at byte {}", self.pos))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn len(&mut self) -> Result<usize, String> {
        usize::try_from(self.u64()?).map_err(|e| format!("GGUF length overflows usize: {e}"))
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.len()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    /// Skip one metadata value of GGUF value type `ty`.
    fn skip_value(&mut self, ty: u32) -> Result<(), String> {
        let fixed = |ty: u32| match ty {
            0 | 1 | 7 => Some(1usize),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        match ty {
            8 => {
                let n = self.len()?;
                self.take(n)?;
            }
            9 => {
                let elem = self.u32()?;
                let count = self.len()?;
                match fixed(elem) {
                    Some(size) => {
                        let n = count
                            .checked_mul(size)
                            .ok_or_else(|| "GGUF array size overflows".to_string())?;
                        self.take(n)?;
                    }
                    None => {
                        for _ in 0..count {
                            self.skip_value(elem)?;
                        }
                    }
                }
            }
            other => {
                let size = fixed(other)
                    .ok_or_else(|| format!("unknown GGUF value type {other} at {}", self.pos))?;
                self.take(size)?;
            }
        }
        Ok(())
    }
}

/// Every tensor-info record's name with the byte position of its `u32` type
/// field (GGUF v2/v3 header layout).
fn tensor_type_fields(bytes: &[u8]) -> Result<Vec<(String, usize)>, String> {
    let mut c = HeaderCursor { bytes, pos: 0 };
    if c.take(4)? != b"GGUF" {
        return Err("not a GGUF file".to_string());
    }
    let version = c.u32()?;
    if version < 2 {
        return Err(format!(
            "GGUF v{version} header layout is not supported here"
        ));
    }
    let n_tensors = c.len()?;
    let n_kv = c.len()?;
    for _ in 0..n_kv {
        c.string()?;
        let ty = c.u32()?;
        c.skip_value(ty)?;
    }
    let mut out = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()?;
        for _ in 0..n_dims {
            c.u64()?;
        }
        let type_pos = c.pos;
        c.u32()?;
        c.u64()?;
        out.push((name, type_pos));
    }
    Ok(out)
}

/// Re-encode every `PQ2_0` tensor of a GGUF image in place as `TQ2_0_g128`
/// (see the module docs). Returns how many tensors were re-encoded. Nothing is
/// modified unless every tensor passes the checks first.
fn reencode_pq2_as_tq2(bytes: &mut [u8]) -> Result<usize, String> {
    let fields = tensor_type_fields(bytes)?;
    // (type-field position, data start, data length) of every PQ2_0 tensor,
    // with the extents taken from the real parser.
    let jobs: Vec<(String, usize, usize, usize)> = {
        let gguf = GgufFile::parse(bytes).map_err(|e| format!("parse: {e}"))?;
        let mut jobs = Vec::new();
        for (name, type_pos) in fields {
            let info = gguf
                .tensors
                .get(&name)
                .ok_or_else(|| format!("header walk found {name}, the parser did not"))?;
            if info.tensor_type != GgufTensorType::PQ2_0 {
                continue;
            }
            let len = gguf
                .tensor_data(&name)
                .map_err(|e| format!("tensor_data({name}): {e}"))?
                .len();
            let offset = usize::try_from(info.offset).map_err(|e| format!("{name} offset: {e}"))?;
            jobs.push((name, type_pos, gguf.data_offset + offset, len));
        }
        jobs
    };
    for (name, type_pos, start, len) in &jobs {
        let wire = u32::from_le_bytes([
            bytes[*type_pos],
            bytes[type_pos + 1],
            bytes[type_pos + 2],
            bytes[type_pos + 3],
        ]);
        if wire != PQ2_0_WIRE_ID {
            return Err(format!(
                "{name}: header walk read type id {wire} where the parser saw PQ2_0"
            ));
        }
        if len % TWO_BIT_BLOCK_BYTES != 0 {
            return Err(format!(
                "{name}: {len} bytes is not a whole number of blocks"
            ));
        }
        let plus_two: u64 = bytes[*start..start + len]
            .as_chunks::<TWO_BIT_BLOCK_BYTES>()
            .0
            .iter()
            .map(|block| u64::from(count_plus_two_codes(&block[2..])))
            .sum();
        if plus_two > 0 {
            return Err(format!(
                "{name}: {plus_two} `+2` code(s) — not ternary, a TQ2_0_g128 re-encode would \
                 change its values"
            ));
        }
    }
    for (_, type_pos, start, len) in &jobs {
        bytes[*type_pos..type_pos + 4].copy_from_slice(&TQ2_0_G128_WIRE_ID.to_le_bytes());
        for block in bytes[*start..start + len]
            .as_chunks_mut::<TWO_BIT_BLOCK_BYTES>()
            .0
        {
            // `[d0 d1 qs0..qs31]` → `[qs0..qs31 d0 d1]`.
            block.rotate_left(2);
        }
    }
    Ok(jobs.len())
}

// ─────────────────────────────────────────────────────────────────────────────
// Engines
// ─────────────────────────────────────────────────────────────────────────────

/// The dimensions the shared CUDA activation buffers and KV cache are keyed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Geometry {
    n_layers: usize,
    hidden: usize,
    intermediate: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Geometry {
    fn of(config: &Qwen3Config) -> Self {
        Self {
            n_layers: config.num_layers,
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
            n_q_heads: config.num_attention_heads,
            n_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        }
    }
}

/// Greedy sampling: temperature 0 routes the sampler to argmax.
fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

/// Build a model from `gguf` and pin an engine to `tier`, failing loudly when
/// the tier is not executable here (`try_from_model_with_tier`), so a `Gpu`
/// request can never quietly degrade to the CPU fallback.
fn engine_for<'a>(
    gguf: &'a GgufFile<'a>,
    tier: KernelTier,
    label: &str,
) -> (InferenceEngine<'a>, Geometry) {
    let model = BonsaiModel::from_gguf(gguf, MAX_SEQ)
        .unwrap_or_else(|e| panic!("F-M1: {label}: BonsaiModel::from_gguf: {e}"));
    let geometry = Geometry::of(model.config());
    let engine = InferenceEngine::try_from_model_with_tier(model, tier, greedy_params(), 42)
        .unwrap_or_else(|e| panic!("F-M1: {label}: tier {tier:?} unavailable: {e}"));
    (engine, geometry)
}

/// Greedily decode `prompt` on `engine`.
fn decode(engine: &mut InferenceEngine<'_>, prompt: &[u32], label: &str) -> Vec<u32> {
    let t = Instant::now();
    let tokens = engine
        .generate(prompt, N_NEW)
        .unwrap_or_else(|e| panic!("F-M1: {label}: generate: {e}"));
    println!(
        "F-M1 {label}: {tokens:?} ({:.1} s)",
        t.elapsed().as_secs_f64()
    );
    tokens
}

/// The decode prompt and a label saying where it came from: the first
/// [`PROMPT_MAX_TOKENS`] tokens of [`PROSE`] when `tokenizer.json` is
/// available (found like the models), else the synthetic id ramp of the
/// original harness, announced as a fallback.
fn fm1_prompt() -> (Vec<u32>, String) {
    let unusable = match find_model("tokenizer.json") {
        Some(path) => match OxiTokenizer::from_json_file(&path) {
            Ok(tokenizer) => match tokenizer.encode(PROSE) {
                Ok(mut ids) if ids.len() >= PROMPT_MAX_TOKENS => {
                    ids.truncate(PROMPT_MAX_TOKENS);
                    return (ids, format!("prose, first {PROMPT_MAX_TOKENS} tokens"));
                }
                Ok(ids) => format!(
                    "the prose prompt tokenized to only {} tokens (< {PROMPT_MAX_TOKENS})",
                    ids.len()
                ),
                Err(e) => format!("tokenizing the prose prompt failed ({e})"),
            },
            Err(e) => format!("cannot load {path:?} ({e})"),
        },
        None => format!("tokenizer.json not found under {:?}", models_dir()),
    };
    println!(
        "F-M1 note: prose prompt not used — {unusable}; falling back to the synthetic \
         {SYNTHETIC_PROMPT_LEN}-token id ramp, whose flatter next-token distributions make a \
         near-tie token mismatch more likely (put tokenizer.json next to the models)"
    );
    (
        (0..SYNTHETIC_PROMPT_LEN).map(|i| 1000 + i * 37).collect(),
        format!("synthetic fallback, {unusable}"),
    )
}

/// First index at which two token streams differ (length mismatch included).
fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// Check one GPU phase's event counts between `before` and `after`, appending
/// every violated expectation to `failures`. `expect_q1` names which family's
/// graph the phase must capture (the other one must not be captured at all).
fn check_phase(
    failures: &mut Vec<String>,
    events: &SlotEvents,
    phase: &str,
    before: Snapshot,
    after: Snapshot,
    expect_q1: bool,
) {
    let delta = after.since(before);
    println!(
        "F-M1 {phase}: captures q1={} tq2={}, per-block fallbacks={}, LM-head fallbacks={}, \
         warnings={}",
        delta.q1, delta.tq2, delta.layer_fb, delta.lm_fb, delta.warnings
    );
    let (want, got_want, other, got_other) = if expect_q1 {
        ("Q1", delta.q1, "ternary", delta.tq2)
    } else {
        ("ternary", delta.tq2, "Q1", delta.q1)
    };
    if got_want == 0 {
        failures.push(format!(
            "{phase}: no {want} CUDA graph was captured — the slot was replayed for another \
             model's key, or the fused full-forward never ran / its capture failed"
        ));
    }
    if got_other != 0 {
        failures.push(format!(
            "{phase}: {got_other} {other} graph capture(s) while only the {want} model ran"
        ));
    }
    if delta.layer_fb != 0 {
        failures.push(format!(
            "{phase}: {} per-block fallback(s) — the fused CUDA full-layer forward (the graph \
             path) was refused",
            delta.layer_fb
        ));
    }
    for line in events.warnings_between(before.warnings, after.warnings) {
        println!("F-M1 {phase} warning: {line}");
        let lower = line.to_ascii_lowercase();
        if lower.contains("graph") || lower.contains("full-forward") {
            failures.push(format!("{phase}: kernel warning: {line}"));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The test
// ─────────────────────────────────────────────────────────────────────────────

/// F-M1: the second model loaded into one process re-captures its own CUDA
/// graph instead of replaying the first model's, and the first model
/// re-captures again once the slot has moved.
#[test]
fn fm1_second_model_recaptures_cuda_graph_slot() {
    // Installed before anything touches CUDA, so no capture can predate it.
    let events = install_event_counter();

    if let Err(e) = CudaGraph::global() {
        println!("skip: {TEST} — no CUDA device accessible ({e})");
        record_skipped(Capability::CudaHardware, TEST);
        return;
    }
    let (Some(q1_path), Some(tq2_path)) = (
        find_model(Q1_FILE),
        find_model(TQ2_FILE).or_else(|| find_model(TQ2_FILE_NATIVE)),
    ) else {
        println!(
            "skip: {TEST} — needs both {Q1_FILE} and {TQ2_FILE} (or {TQ2_FILE_NATIVE}) under \
             models/ (or $OXIBONSAI_MODELS_DIR)"
        );
        record_skipped(Capability::CudaHardware, TEST);
        return;
    };
    // Past the skip paths every run must be decided with the per-phase
    // capture counts: the token oracles alone can pass vacuously.
    let events = events.unwrap_or_else(|reason| {
        panic!(
            "F-M1: the capture counter is unavailable ({reason}). Without the per-phase capture \
             counts the token oracles alone can pass vacuously (an engine that never reaches the \
             fused graph path decodes its CPU tokens and repeats its own phase-1 tokens without \
             ever touching the slot), so this harness refuses to decide. Run this binary on its \
             own (its single test must install the only global tracing subscriber) and without \
             tracing max_level_* / release_max_level_* features below debug."
        )
    });
    let tq2_name = tq2_path.file_name().map_or_else(
        || TQ2_FILE.to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    let start = Instant::now();
    let q1_bytes =
        std::fs::read(&q1_path).unwrap_or_else(|e| panic!("F-M1: read {q1_path:?}: {e}"));
    let mut tq2_bytes =
        std::fs::read(&tq2_path).unwrap_or_else(|e| panic!("F-M1: read {tq2_path:?}: {e}"));
    let stored_as = GgufFile::parse(&tq2_bytes)
        .unwrap_or_else(|e| panic!("F-M1: parse {tq2_path:?}: {e}"))
        .tensors
        .get("blk.0.attn_q.weight")
        .map(|info| info.tensor_type);
    match stored_as {
        Some(GgufTensorType::TQ2_0_g128) => {
            println!("F-M1: {tq2_name} is stored as TQ2_0_g128 — used as is");
        }
        Some(GgufTensorType::PQ2_0) => {
            println!(
                "F-M1: {tq2_name} is stored as PQ2_0 (ggml id 142), which the CUDA ternary full \
                 forward (the slot's ternary consumer) never takes — re-encoding it in memory \
                 as TQ2_0_g128 (lossless for a ternary checkpoint)"
            );
            match reencode_pq2_as_tq2(&mut tq2_bytes) {
                Ok(n) => println!("F-M1: re-encoded {n} PQ2_0 tensors as TQ2_0_g128"),
                Err(e) => {
                    println!("skip: {TEST} — {tq2_name} cannot be re-encoded as TQ2_0_g128: {e}");
                    record_skipped(Capability::CudaHardware, TEST);
                    return;
                }
            }
        }
        other => {
            println!(
                "skip: {TEST} — {tq2_name}'s blk.0.attn_q.weight is {other:?}; the CUDA ternary \
                 full forward takes TQ2_0_g128 only"
            );
            record_skipped(Capability::CudaHardware, TEST);
            return;
        }
    }
    let q1_gguf =
        GgufFile::parse(&q1_bytes).unwrap_or_else(|e| panic!("F-M1: parse {q1_path:?}: {e}"));
    let tq2_gguf =
        GgufFile::parse(&tq2_bytes).unwrap_or_else(|e| panic!("F-M1: parse {tq2_path:?}: {e}"));
    assert_eq!(
        tq2_gguf
            .tensors
            .get("blk.0.attn_q.weight")
            .map(|info| info.tensor_type),
        Some(GgufTensorType::TQ2_0_g128),
        "F-M1: the ternary model must reach the loader as TQ2_0_g128"
    );
    let (prompt, prompt_kind) = fm1_prompt();
    assert!(
        !prompt.is_empty() && prompt.len() <= PROMPT_MAX_TOKENS,
        "F-M1: the prompt must have 1..={PROMPT_MAX_TOKENS} tokens to be prefilled through the \
         graph path, got {}",
        prompt.len()
    );
    println!(
        "F-M1 prompt ({prompt_kind}; {} tokens): {prompt:?}, {N_NEW} greedy tokens per run, \
         max_seq {MAX_SEQ}",
        prompt.len()
    );

    // ── CPU references (KernelTier::Reference never touches CUDA) ──────────
    let (cpu_q1, q1_geometry) = {
        let (mut engine, geometry) = engine_for(&q1_gguf, KernelTier::Reference, "Bonsai-8B CPU");
        (
            decode(&mut engine, &prompt, "Bonsai-8B CPU-reference"),
            geometry,
        )
    };
    let (cpu_tq2, tq2_geometry) = {
        let (mut engine, geometry) =
            engine_for(&tq2_gguf, KernelTier::Reference, "Ternary-Bonsai-8B CPU");
        (
            decode(&mut engine, &prompt, "Ternary-Bonsai-8B CPU-reference"),
            geometry,
        )
    };
    println!("F-M1 geometry: Bonsai-8B {q1_geometry:?}, Ternary-Bonsai-8B {tq2_geometry:?}");
    assert_eq!(
        q1_geometry, tq2_geometry,
        "F-M1 precondition: both models must share every buffer dimension, otherwise a \
         re-capture is dimension-driven and says nothing about the slot key"
    );
    assert!(
        !cpu_tq2.is_empty(),
        "F-M1: the ternary CPU reference produced no tokens — nothing to compare against"
    );

    // ── Phase 1: Bonsai-8B on the GPU (captures the Q1 graph) ───────────────
    let s0 = events.snapshot();
    let (mut gpu_q1_engine, _) = engine_for(&q1_gguf, KernelTier::Gpu, "Bonsai-8B GPU");
    let gpu_q1 = decode(&mut gpu_q1_engine, &prompt, "phase 1 Bonsai-8B GPU");
    let s1 = events.snapshot();

    // ── Phase 2: Ternary-Bonsai-8B on the GPU, Q1 engine still alive ────────
    let (mut gpu_tq2_engine, _) = engine_for(&tq2_gguf, KernelTier::Gpu, "Ternary-Bonsai-8B GPU");
    let gpu_tq2 = decode(
        &mut gpu_tq2_engine,
        &prompt,
        "phase 2 Ternary-Bonsai-8B GPU",
    );
    let s2 = events.snapshot();

    // ── Phase 3: Bonsai-8B again — the slot now holds the ternary key ──────
    let gpu_q1_again = decode(&mut gpu_q1_engine, &prompt, "phase 3 Bonsai-8B GPU again");
    let s3 = events.snapshot();

    // ── Verdict ─────────────────────────────────────────────────────────────
    let mut failures: Vec<String> = Vec::new();
    match first_diff(&cpu_tq2, &gpu_tq2) {
        None => println!("F-M1 tokens: second model GPU == CPU reference: true"),
        Some(i) => failures.push(format!(
            "second model (Ternary-Bonsai-8B) GPU tokens {gpu_tq2:?} != its CPU-reference \
             tokens {cpu_tq2:?} (first diff at {i}) — the slot replayed the Q1 graph, or the \
             ternary GPU path disagrees on its own (run cuda_ternary_forward_parity alone to \
             separate the two)"
        )),
    }
    match first_diff(&gpu_q1, &gpu_q1_again) {
        None => println!("F-M1 tokens: Bonsai-8B phase 3 == phase 1: true"),
        Some(i) => failures.push(format!(
            "Bonsai-8B decoded again after the ternary model took the slot gave {gpu_q1_again:?}, \
             phase 1 gave {gpu_q1:?} (first diff at {i}) — the re-capture is wrong or the \
             ternary graph was replayed"
        )),
    }
    match first_diff(&cpu_q1, &gpu_q1) {
        None => println!("F-M1 tokens: first model GPU == CPU reference: true (informational)"),
        Some(i) => println!(
            "F-M1 tokens: first model GPU {gpu_q1:?} != CPU reference {cpu_q1:?} (first diff \
             at {i}) — informational only, see cuda_cross_backend_determinism_tests / Step 4e"
        ),
    }
    if gpu_q1 == gpu_tq2 {
        println!("F-M1 note: both models produced identical GPU tokens (possible, not expected)");
    }

    // The capture counts are part of every verdict (the counter is mandatory).
    check_phase(&mut failures, &events, "phase 1 (Bonsai-8B)", s0, s1, true);
    check_phase(
        &mut failures,
        &events,
        "phase 2 (Ternary-Bonsai-8B)",
        s1,
        s2,
        false,
    );
    check_phase(
        &mut failures,
        &events,
        "phase 3 (Bonsai-8B again)",
        s2,
        s3,
        true,
    );

    assert!(
        failures.is_empty(),
        "F-M1 FAIL ({} problem(s)):\n  - {}",
        failures.len(),
        failures.join("\n  - ")
    );
    println!(
        "F-M1 PASS: second model re-captured its own graph (tokens match its CPU reference), \
         first model re-captured after the slot moved, capture counts as required in all three \
         phases (prompt: {prompt_kind}; {:.1} s total)",
        start.elapsed().as_secs_f64()
    );
    drop(gpu_tq2_engine);
    drop(gpu_q1_engine);
    record_executed_timed(Capability::CudaHardware, TEST, start.elapsed());
}
