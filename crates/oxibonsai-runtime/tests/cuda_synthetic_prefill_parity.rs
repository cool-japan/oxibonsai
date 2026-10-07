//! Synthetic-model CPU↔CUDA batch-prefill→decode parity gate (no external GGUF).
//!
//! This is the regression gate for the CUDA prefill→decode KV-cache handoff bug.
//! Historically, CUDA batch prefill wrote the prompt K/V into a *prefill-private*
//! GPU KV cache while per-token decode read a *different* (`FULL_LAYER_STATE`)
//! cache, so any prompt longer than 16 tokens made decode attend over stale
//! (all-zero) KV and silently corrupt generation (decode logit Δ vs CPU ≈ 7.3).
//! The fix unifies the two caches (`cuda_prefill` delegates to
//! `cuda_full_layer::acquire_kv_cache`) and, for Q1, uploads the per-token
//! `d_pos_seqlen` the fused KV-store needs.
//!
//! Unlike `cuda_ternary_forward_parity.rs` (which needs a multi-GB real GGUF),
//! this test assembles tiny fully-Q1 and fully-ternary GGUFs in memory, so it
//! runs unattended on any CUDA box.  It drives a >16-token prompt through
//! `forward_prefill` (which routes multi-token batches to the fused CUDA prefill
//! path) then several decode steps, and asserts the CUDA (`KernelTier::Gpu`)
//! logits and greedy tokens match a CPU reference leg.
//!
//! The CPU leg is CPU all the way down, which passing `KernelTier::Reference`
//! to `forward_prefill` / `forward` does not by itself guarantee:
//! `BonsaiModel::from_gguf` builds every layer's own dispatcher (and the LM
//! head's) with `KernelDispatcher::auto_detect()`, which is GPU-tier on a CUDA
//! host, and the linears GEMV through that stored dispatcher.  The CPU leg
//! therefore constructs its model and runs every forward call inside a
//! `CpuOnlyBackendScope` (see [`leg_backend_scope`]): those internal
//! dispatchers then resolve to the best CPU SIMD tier, while the dispatcher the
//! leg passes to `forward_prefill` / `forward` stays `KernelTier::Reference`.
//! This closes the earlier CUDA-against-CUDA FP8 comparison: without the
//! scope, the FP8 linears ran the CUDA FP8 GEMV in the "CPU" leg too (RTX
//! A4000, 2026-10-07: every FP8 logit delta printed as 0, and nsys counted
//! 1920 `gemv_fp8_e4m3` launches for the E4M3 test, 810 of them in its two
//! `Reference` legs).  The Q1/TQ2 legs were already CPU: a layer without an
//! uploaded weight handle (a CPU leg never uploads one) runs its Q1 GEMV/GEMM
//! on the GPU only at 1024 or more output rows and its TQ2 GEMV/GEMM never, and
//! these synthetic shapes have at most 256 rows.
//!
//! The GPU leg, in turn, has to show that it ran on CUDA: a CUDA path that
//! silently falls back to the CPU turns the comparison into CPU against CPU,
//! which passes trivially.  Besides the comparisons, each test therefore
//! requires:
//!
//! * a GPU-tier `auto_detect()` in the GPU leg, so the layer dispatchers
//!   `from_gguf` builds for it reach CUDA (see [`leg_backend_scope`]);
//! * no fallback event: the GPU leg's forward calls run under a capture of
//!   every `oxibonsai*` tracing event (see [`run_leg`]), and a warning about
//!   a fallback (a failed CUDA FP8 GEMV, a fused CUDA forward that gave up, a
//!   failed CUDA batch prefill) fails the test, as does, for Q1/TQ2, any
//!   `prefill_dispatch` event or a model-side event about leaving the fused
//!   layer path;
//! * for Q1/TQ2, the model's device-KV latch (`BonsaiModel::gpu_path_active`)
//!   set at the end of the GPU leg's prefill and of its decode: only the
//!   fused CUDA batch prefill and the fused per-token CUDA decode set it, and
//!   every CPU (host-KV) path clears it (see [`assert_device_kv_latch`]);
//! * logits that differ between the legs somewhere in the run (see
//!   [`assert_legs_differ`]): CPU SIMD and CUDA kernels accumulate in
//!   different orders, so logits identical at every compared step mean both
//!   legs ran the same kernels.  That is what F-2 looked like (every FP8 delta
//!   exactly 0) and what a GPU leg that fell back to the CPU looks like, and
//!   it is the main evidence for FP8, whose per-token CUDA path runs over the
//!   host KV cache and leaves the latch clear.
//!
//! Capability records (`oxibonsai_testkit::capability`): each test writes
//! `cuda` and `cuda-hardware` with `executed: true` only as its last statement,
//! after every assertion of its run has passed.  That is: a successful
//! `CudaGraph::global()` device probe; the GPU-leg evidence above (GPU-tier
//! layer dispatchers, no fallback event, for Q1/TQ2 the device-KV latch, and
//! logits not bit-identical to the CPU leg's); the logit tolerances and the
//! greedy tokens; and, for FP8, the CPU KV gate.  `cuda-hardware` is the claim
//! this makes, CPU-against-CUDA numeric parity behind a real device probe;
//! `cuda` is the capability `--require-cuda` reads.
//!
//! Gracefully skips (passes, recording both capabilities `executed: false`)
//! when no CUDA device is present.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_kernels::gpu_backend::{cpu_only_backend_active, CpuOnlyBackendScope};
use oxibonsai_model::model::BonsaiModel;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Synthetic model dimensions (CUDA-friendly: k=hidden multiple of 128,
//    head_dim=64, all weight tensors a multiple of 128 weights). ───────────────
const MAX_SEQ: usize = 256;
const H: usize = 128; // hidden_size (= nq * head_dim, multiple of 128 for g128)
const INTER: usize = 256; // intermediate_size (multiple of 128)
const NUM_LAYERS: usize = 2;
const NQ: usize = 2;
const NKV: usize = 1;
const HD: usize = 64; // head_dim
const VOCAB: usize = 128;

/// Returns `true` when a CUDA device is accessible.
fn cuda_available() -> bool {
    oxibonsai_kernels::CudaGraph::global().is_ok()
}

/// `<cargo package>::<test target>::` prefix of this file's capability records.
const RECORD_PREFIX: &str = "oxibonsai-runtime::cuda_synthetic_prefill_parity::";

/// Record the no-device self-skip of `test_fn` under both CUDA capabilities.
fn record_skip(test_fn: &str) {
    let test = format!("{RECORD_PREFIX}{test_fn}");
    record_skipped(Capability::Cuda, &test);
    record_skipped(Capability::CudaHardware, &test);
}

/// Record a completed CPU-against-CUDA parity run of `test_fn` under both CUDA
/// capabilities.  Call only after every assertion of the run has passed,
/// including the GPU-leg evidence (see the module docs): without it a GPU leg
/// that fell back to the CPU would be recorded as CUDA execution.
fn record_pass(test_fn: &str, elapsed: Duration) {
    let test = format!("{RECORD_PREFIX}{test_fn}");
    record_executed_timed(Capability::Cuda, &test, elapsed);
    record_executed_timed(Capability::CudaHardware, &test, elapsed);
}

/// Deterministic FP32 tensor whose values vary with the index.
fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let phase = (i as f32) * 0.013_f32;
        let val = scale * (1.0_f32 + 0.25_f32 * phase.sin());
        v.extend_from_slice(&val.to_le_bytes());
    }
    v
}

/// Build a `TQ2_0_g128` weight blob (34 bytes/block: 32B of 2-bit codes + FP16 scale).
fn tq2_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 34);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_blocks {
        for _ in 0..32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale_f32).to_le_bytes());
    }
    data
}

/// Build a `Q1_0G128` weight blob (18 bytes/block: FP16 scale + 16B of 128 sign bits).
fn q1_0_g128_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        num_weights % 128,
        0,
        "num_weights must be a multiple of 128"
    );
    let num_blocks = num_weights / 128;
    let mut data = Vec::with_capacity(num_blocks * 18);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_blocks {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale_f32 = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale_f32).to_le_bytes());
        for _ in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
    }
    data
}

/// Emit one quantised projection tensor in the requested format.
fn quant_tensor(
    name: String,
    shape: Vec<u64>,
    num_weights: usize,
    seed: u64,
    ternary: bool,
) -> TensorEntry {
    if ternary {
        TensorEntry {
            name,
            shape,
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_0_g128_pattern(num_weights, seed),
        }
    } else {
        TensorEntry {
            name,
            shape,
            tensor_type: TensorType::Q1_0G128,
            data: q1_0_g128_pattern(num_weights, seed),
        }
    }
}

/// Build a fully-quantised synthetic GGUF (`ternary=true` → TQ2, else Q1).
fn build_synthetic_gguf(ternary: bool) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("CudaPrefillParityTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(H as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(NUM_LAYERS as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(NQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(NKV as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(INTER as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
    writer.add_metadata(
        "qwen3.context_length",
        MetadataWriteValue::U32(MAX_SEQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![H as u64, VOCAB as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(VOCAB * H, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![H as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(H, 1.0),
    });
    writer.add_tensor(quant_tensor(
        "output.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        VOCAB * H,
        0xCAFE_BABE,
        ternary,
    ));

    for layer in 0..NUM_LAYERS {
        let pfx = format!("blk.{layer}");
        for (suffix, dim) in [
            ("attn_norm", H),
            ("ffn_norm", H),
            ("attn_q_norm", HD),
            ("attn_k_norm", HD),
        ] {
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}.weight"),
                shape: vec![dim as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(dim, 1.0),
            });
        }
        let s = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_q.weight"),
            vec![H as u64, (NQ * HD) as u64],
            NQ * HD * H,
            s,
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_k.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(1),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_v.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(2),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.attn_output.weight"),
            vec![(NQ * HD) as u64, H as u64],
            H * NQ * HD,
            s.wrapping_add(3),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_gate.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(4),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_up.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(5),
            ternary,
        ));
        writer.add_tensor(quant_tensor(
            format!("{pfx}.ffn_down.weight"),
            vec![INTER as u64, H as u64],
            H * INTER,
            s.wrapping_add(6),
            ternary,
        ));
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Make one test leg run where its `tier` says, for the whole model.
///
/// The tier a leg passes to `forward_prefill` / `forward` is not the only
/// dispatcher involved: `BonsaiModel::from_gguf` builds every layer's own
/// dispatcher with `KernelDispatcher::auto_detect()`, and the linears GEMV
/// through that stored dispatcher.  On a CUDA host `auto_detect()` is
/// GPU-tier, so a model built outside a scope runs the CUDA FP8 GEMV even in a
/// `KernelTier::Reference` leg (the tier gate in `dispatch_fp8.rs` checks the
/// *layer's* dispatcher).
///
/// For a CPU leg this returns an active [`CpuOnlyBackendScope`].  While it
/// lives, every `auto_detect()` made on this thread skips `select_backend()`
/// and resolves to the best CPU SIMD tier, so a model constructed under it has
/// CPU-tier dispatchers throughout.  The scope is a thread-local depth counter:
/// it nests, it is `!Send`, and it does nothing to the process-wide `CudaGraph`
/// that `cuda_available()` has already initialised, which the GPU leg keeps
/// using.  Rayon workers do not see the scope, and need not: the forward paths
/// these models take create no dispatcher of their own, they use the ones
/// built at construction.  The caller binds the guard for the whole leg (model
/// construction, prefill and every decode step); it drops when the leg
/// returns, before the caller starts the GPU leg.
///
/// For the GPU leg this returns `None`, after checking that no scope is still
/// active on this thread and that `auto_detect()` is GPU-tier there, i.e. that
/// the model the leg builds gets GPU-tier layer dispatchers (the FP8 linears
/// reach the CUDA FP8 GEMV through nothing else).
#[must_use]
fn leg_backend_scope(tier: KernelTier) -> Option<CpuOnlyBackendScope> {
    if tier == KernelTier::Gpu {
        assert!(
            !cpu_only_backend_active(),
            "a CpuOnlyBackendScope is still active when the GPU leg starts"
        );
        // What `BonsaiModel::from_gguf` builds for every layer of this leg.
        let layer_tier = KernelDispatcher::auto_detect().tier();
        assert_eq!(
            layer_tier,
            KernelTier::Gpu,
            "GPU leg: auto_detect() is not GPU-tier, so the model's layers would run on the CPU"
        );
        return None;
    }
    let scope = CpuOnlyBackendScope::enter();
    // What `BonsaiModel::from_gguf` builds for every layer under this scope.
    let layer_tier = KernelDispatcher::auto_detect().tier();
    assert_ne!(
        layer_tier,
        KernelTier::Gpu,
        "CpuOnlyBackendScope did not keep auto_detect() off the GPU"
    );
    eprintln!("CPU leg: layer dispatchers resolve to {layer_tier:?} under CpuOnlyBackendScope");
    Some(scope)
}

// ── GPU-leg evidence ─────────────────────────────────────────────────────────

/// How a model family's GPU leg keeps the attention KV cache, which decides
/// the evidence that leg can give of having run on CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuKv {
    /// Q1 / TQ2: the fused CUDA batch prefill and the fused per-token CUDA
    /// decode keep the K/V in the device KV cache, and both set the model's
    /// device-KV latch (`BonsaiModel::gpu_path_active`), which every host-KV
    /// (CPU) path clears.
    Device,
    /// FP8: the FP8 batch prefill is refused by construction, so the GPU leg
    /// runs the per-token path: CUDA FP8 GEMVs through the layers' GPU-tier
    /// dispatchers over the host KV cache, which leaves the latch clear.
    Host,
}

/// One captured `oxibonsai*` tracing event.
#[derive(Debug)]
struct CapturedEvent {
    level: tracing::Level,
    module: String,
    message: String,
    fields: String,
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

/// Collects an event's `message` and its other fields.
#[derive(Default)]
struct FieldCollector {
    message: String,
    fields: String,
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

/// A minimal `tracing::Subscriber` that keeps every DEBUG-or-more-severe event
/// emitted from an `oxibonsai*` module (the per-token `fwd_profile` timing
/// events excepted) while it is the thread's default subscriber.  The same
/// capture `cuda_p14_p15_batch_prefill_vs_sequential.rs` uses.
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
            });
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Run `f` with an [`EventCapture`] as this thread's subscriber and return its
/// result together with every event it captured.
fn captured<T>(f: impl FnOnce() -> T) -> (T, Vec<CapturedEvent>) {
    let capture = EventCapture::default();
    let events = Arc::clone(&capture.events);
    let out = tracing::subscriber::with_default(capture, f);
    let log = std::mem::take(&mut *events.lock().unwrap_or_else(|e| e.into_inner()));
    (out, log)
}

/// The events captured during a GPU leg's forward calls that show part of
/// the leg left CUDA:
///
/// * any WARN-or-worse event about a fallback ("falling back", "fallback").
///   During a forward call each of these is a GPU path giving up: a CUDA FP8
///   GEMV that failed (`dispatch_fp8.rs`), a fused CUDA forward that returned
///   nothing, a CUDA batch prefill that failed for any reason but the FP8
///   split-cache refusal.  That refusal is logged at DEBUG because it is by
///   construction, and the per-token path it leaves the FP8 GPU leg on is the
///   one this test compares;
/// * for a device-KV family, any `prefill_dispatch` event (it logs only when a
///   GPU batch prefill fails or is skipped) and any model-side event about
///   leaving the fused layer path for the per-block loop or the CPU.
fn gpu_fallback_events(events: &[CapturedEvent], kv: GpuKv) -> Vec<&CapturedEvent> {
    events
        .iter()
        .filter(|e| {
            let warned_fallback = e.level <= tracing::Level::WARN
                && (e.message.contains("falling back") || e.message.contains("fallback"));
            let left_device_kv = kv == GpuKv::Device
                && (e.module.contains("prefill_dispatch")
                    || (e.module.starts_with("oxibonsai_model")
                        && (e.message.contains("per-block")
                            || e.message.contains("falling back to CPU"))));
            warned_fallback || left_device_kv
        })
        .collect()
}

/// Run the forward calls of one leg (its prefill and every decode step).
///
/// A CPU leg's calls simply run.  The GPU leg's calls run under an
/// [`EventCapture`]: their WARN-or-worse events are printed, and the test fails
/// when the capture shows that the leg left CUDA (see
/// [`gpu_fallback_events`]).  Model construction stays outside the capture:
/// its warnings are about the weights, not the backend (for example, the TQ2
/// layout resolver warns that it is "falling back to the legacy qs-first
/// TQ2_0_g128 reading" for these small synthetic tensors, in both legs
/// alike), and the tier of the layer dispatchers it builds is what
/// [`leg_backend_scope`] checks.
fn run_leg<T>(tier: KernelTier, kv: GpuKv, label: &str, body: impl FnOnce() -> T) -> T {
    if tier != KernelTier::Gpu {
        return body();
    }
    let (out, events) = captured(body);
    for e in events.iter().filter(|e| e.level <= tracing::Level::WARN) {
        eprintln!("  {label}: {e}");
    }
    let fallbacks = gpu_fallback_events(&events, kv);
    assert!(
        fallbacks.is_empty(),
        "{label}: the GPU leg left CUDA, so the comparison would be against a CPU result; \
         captured: {:#?}",
        fallbacks.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
    out
}

/// Fail a Q1 / TQ2 GPU leg whose `stretch` ended with the device-KV latch clear.
///
/// The fused CUDA batch prefill and the fused per-token CUDA decode set
/// `BonsaiModel::gpu_path_active`; the batched CPU prefill and the per-block
/// host-KV loop clear it.  A clear latch therefore means a CPU path answered:
/// the CUDA batch prefill failed or was skipped, or the CPU path was forced
/// (`OXIBONSAI_FORCE_CPU_DECODE_AFTER`).  A CPU leg, and an FP8 GPU leg (host
/// KV by design), are not checked here.
fn assert_device_kv_latch(
    model: &BonsaiModel<'_>,
    tier: KernelTier,
    kv: GpuKv,
    label: &str,
    stretch: &str,
) {
    if tier == KernelTier::Gpu && kv == GpuKv::Device {
        assert!(
            model.gpu_path_active(),
            "{label}: the GPU leg's {stretch} left the device-KV latch clear, so a CPU (host-KV) \
             path answered it instead of the fused CUDA path"
        );
    }
}

/// Prefill `prompt`, then teacher-force `decode_inputs` at increasing positions.
/// Returns `(prefill_last_logits, per-decode-step logits)`.
fn drive_teacher_forced(
    bytes: &[u8],
    tier: KernelTier,
    kv: GpuKv,
    name: &str,
    prompt: &[u32],
    decode_inputs: &[u32],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    // Held for the whole leg: model construction and every forward call.
    let _backend_scope = leg_backend_scope(tier);
    let label = format!("{name} {tier:?} leg (teacher-forced)");
    let gguf = GgufFile::parse(bytes).expect("parse gguf");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
    let kernel = KernelDispatcher::with_tier(tier);
    run_leg(tier, kv, &label, || {
        let prefill = model
            .forward_prefill(prompt, 0, &kernel)
            .expect("forward_prefill");
        assert_device_kv_latch(&model, tier, kv, &label, "prefill");
        let mut decode_logits = Vec::with_capacity(decode_inputs.len());
        for (i, &tok) in decode_inputs.iter().enumerate() {
            let lg = model
                .forward(tok, prompt.len() + i, &kernel)
                .expect("decode forward");
            decode_logits.push(lg);
        }
        assert_device_kv_latch(&model, tier, kv, &label, "decode");
        (prefill, decode_logits)
    })
}

/// Prefill `prompt`, then greedily decode `n_steps` tokens (argmax feedback).
fn drive_greedy(
    bytes: &[u8],
    tier: KernelTier,
    kv: GpuKv,
    name: &str,
    prompt: &[u32],
    n_steps: usize,
) -> Vec<u32> {
    // Held for the whole leg: model construction and every forward call.
    let _backend_scope = leg_backend_scope(tier);
    let label = format!("{name} {tier:?} leg (greedy)");
    let gguf = GgufFile::parse(bytes).expect("parse gguf");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
    let kernel = KernelDispatcher::with_tier(tier);
    run_leg(tier, kv, &label, || {
        let mut logits = model
            .forward_prefill(prompt, 0, &kernel)
            .expect("forward_prefill");
        assert_device_kv_latch(&model, tier, kv, &label, "prefill");
        let mut out = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let next = argmax(&logits);
            out.push(next);
            logits = model
                .forward(next, prompt.len() + step, &kernel)
                .expect("greedy decode forward");
        }
        assert_device_kv_latch(&model, tier, kv, &label, "decode");
        out
    })
}

/// Cosine similarity of two logit vectors, accumulated in `f64`.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt()).max(f64::MIN_POSITIVE)
}

/// Assert two logit vectors agree within a cross-backend tolerance (the CPU
/// leg vs CUDA fp32 compute with an FP16 KV cache).  The broken KV handoff
/// produced Δ ≈ 7.3; this threshold is far below that yet tolerant of FP16-KV
/// noise.  Returns the largest absolute difference, for
/// [`assert_legs_differ`].
fn assert_logits_close(cpu: &[f32], gpu: &[f32], label: &str) -> f32 {
    assert_eq!(cpu.len(), gpu.len(), "{label}: logit length mismatch");
    let mut max_abs = 0.0_f32;
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}: non-finite logit at {i}"
        );
        let abs = (a - b).abs();
        let rel = abs / a.abs().max(b.abs()).max(1e-3);
        if abs > max_abs {
            max_abs = abs;
        }
        assert!(
            abs < 0.25 || rel < 0.05,
            "{label}: logit[{i}] CPU={a:.5} CUDA={b:.5} abs={abs:.4} rel={rel:.4}"
        );
    }
    let cosine = cosine_similarity(cpu, gpu);
    eprintln!("{label}: max_abs logit delta = {max_abs:.3e}, cosine = {cosine:.9}");
    max_abs
}

/// Fail when the CPU and GPU legs agreed bit for bit at every compared step.
///
/// `run_max_abs` is the largest CPU-against-CUDA logit difference over the
/// prefill and every decode step.  The legs are meant to run different
/// kernels, CPU SIMD against CUDA, whose f32 accumulation orders differ, so
/// their logits differ in the last bits somewhere (RTX A4000, 2026-10-07:
/// max|delta| between about 4e-6 and 3e-5 at every step).  Exactly 0 at
/// every step means both legs ran the same kernels: the CPU leg reached the
/// CUDA kernels (F-2: every FP8 delta was 0), or the GPU leg fell back to the
/// CPU.  A GPU-tier dispatcher's CPU fallback runs the best CPU SIMD tier,
/// which is also the tier `auto_detect()` picks for the CPU leg's layers under
/// `CpuOnlyBackendScope` (both use the same feature detection), so such a
/// fallback reproduces the CPU leg exactly.
fn assert_legs_differ(name: &str, run_max_abs: f32) {
    assert!(
        run_max_abs > 0.0,
        "{name}: the CPU and GPU legs gave bit-identical logits at every compared step, so \
         both ran the same kernels (the CPU leg reached CUDA, or the GPU leg fell back to the \
         CPU); the comparison proves nothing"
    );
    eprintln!(
        "{name}: run max_abs logit delta = {run_max_abs:.3e} (> 0: the legs ran different kernels)"
    );
}

fn run_parity(ternary: bool, test_fn: &str) {
    let name = if ternary { "ternary(TQ2)" } else { "Q1" };
    if !cuda_available() {
        eprintln!("skip {name}: no CUDA device available");
        record_skip(test_fn);
        return;
    }
    let start = Instant::now();
    let bytes = build_synthetic_gguf(ternary);

    // 20-token prompt (> 16) forces the fused CUDA batch-prefill path rather than
    // the <=16 sequential fast path.
    let prompt: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let decode_inputs: Vec<u32> = (0..6u32).map(|i| (i * 5 + 11) % VOCAB as u32).collect();

    let kv = GpuKv::Device;
    let (cpu_prefill, cpu_decode) = drive_teacher_forced(
        &bytes,
        KernelTier::Reference,
        kv,
        name,
        &prompt,
        &decode_inputs,
    );
    let (gpu_prefill, gpu_decode) =
        drive_teacher_forced(&bytes, KernelTier::Gpu, kv, name, &prompt, &decode_inputs);

    // Prefill's last-token logits must agree (validates the batched prefill math).
    let mut run_max_abs =
        assert_logits_close(&cpu_prefill, &gpu_prefill, &format!("{name} prefill-last"));

    // Each decode step attends over the prompt KV — the decisive check for the
    // handoff bug (a stale/zero KV cache would make these diverge wildly).
    for (i, (c, g)) in cpu_decode.iter().zip(gpu_decode.iter()).enumerate() {
        run_max_abs = run_max_abs.max(assert_logits_close(c, g, &format!("{name} decode[{i}]")));
    }
    // ...but not bit for bit: that would mean both legs ran the same kernels.
    assert_legs_differ(name, run_max_abs);

    // Greedy sequences (tolerance-free) must be identical.
    let cpu_greedy = drive_greedy(&bytes, KernelTier::Reference, kv, name, &prompt, 8);
    let gpu_greedy = drive_greedy(&bytes, KernelTier::Gpu, kv, name, &prompt, 8);
    assert_eq!(
        cpu_greedy, gpu_greedy,
        "{name}: greedy decode sequence CPU vs CUDA mismatch"
    );
    eprintln!("{name}: greedy sequence match = {cpu_greedy:?}");
    record_pass(test_fn, start.elapsed());
}

#[test]
fn cuda_synthetic_q1_prefill_decode_parity() {
    run_parity(false, "cuda_synthetic_q1_prefill_decode_parity");
}

#[test]
fn cuda_synthetic_ternary_prefill_decode_parity() {
    run_parity(true, "cuda_synthetic_ternary_prefill_decode_parity");
}

// ── FP8 (E4M3 / E5M2) prefill→decode fallback parity ─────────────────────────
//
// The FP8 CUDA batch-prefill path (`try_cuda_prefill_with_lm_head_fp8`) writes
// the prompt K/V into a *GPU-private* KV cache (`FP8_PREFILL_STATE` in
// `cuda_fp8_prefill.rs`) that per-token FP8 decode — which attends over
// `self.kv_cache` on the CPU — never reads.  Like the Q4_0/Q8_0/K-quant paths,
// the FP8 batch prefill is therefore disabled by default (guarded by
// `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL`) so `forward_prefill` falls back to the
// bit-correct sequential per-token path, which populates `self.kv_cache`.
//
// This test proves the fallback two ways:
//
//  1. A deterministic, weight-independent regression gate: after a >16-token
//     GPU-tier prefill, every prompt position must have a non-zero key in the CPU
//     `self.kv_cache` that decode reads.  The guarded fallback (sequential
//     per-token) populates it; the FP8 batch-prefill path (env override) writes
//     only the GPU-private cache and leaves every CPU KV row at zero — verified
//     directly, so the check fails the instant the guard/fallback stops engaging,
//     regardless of how benign the synthetic weights are.
//
//  2. A >16-token prompt followed by several decode steps must produce
//     logits/greedy tokens matching the CPU reference leg — end-to-end output
//     correctness through the fallback.  That leg runs inside
//     `leg_backend_scope`, so its FP8 linears run the CPU GEMV; before the
//     scope they ran the CUDA FP8 GEMV as well and this compared CUDA against
//     CUDA.  The GPU leg's FP8 GEMVs must in turn run on CUDA: no CUDA FP8
//     GEMV may warn that it fell back, and the two legs' logits may not be
//     bit-identical (`assert_legs_differ`).

/// Build a `F8_E4M3` / `F8_E5M2` weight blob (34 bytes/block: 32 E4M3/E5M2 codes
/// followed by a FP16 scale).  Values are quantised from a deterministic float
/// pattern so the bytes are always valid FP8 codes (never NaN/Inf encodings).
fn fp8_pattern(num_weights: usize, seed: u64, e4m3: bool) -> Vec<u8> {
    assert_eq!(
        num_weights % 32,
        0,
        "num_weights must be a multiple of 32 (QK_FP8)"
    );
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut vals = Vec::with_capacity(num_weights);
    for _ in 0..num_weights {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = ((state >> 33) as u32 as f32) / (u32::MAX as f32); // 0..1
        vals.push((u - 0.5) * 0.5); // small centred weights
    }
    if e4m3 {
        let blocks = oxibonsai_core::BlockFP8E4M3::quantize(&vals).expect("fp8 e4m3 quantize");
        let mut data = Vec::with_capacity(blocks.len() * 34);
        for b in &blocks {
            data.extend_from_slice(&b.qs);
            data.extend_from_slice(&b.d.to_le_bytes());
        }
        data
    } else {
        let blocks = oxibonsai_core::BlockFP8E5M2::quantize(&vals).expect("fp8 e5m2 quantize");
        let mut data = Vec::with_capacity(blocks.len() * 34);
        for b in &blocks {
            data.extend_from_slice(&b.qs);
            data.extend_from_slice(&b.d.to_le_bytes());
        }
        data
    }
}

/// Emit one FP8 projection tensor (E4M3 when `e4m3`, else E5M2).
fn fp8_tensor(
    name: String,
    shape: Vec<u64>,
    num_weights: usize,
    seed: u64,
    e4m3: bool,
) -> TensorEntry {
    TensorEntry {
        name,
        shape,
        tensor_type: if e4m3 {
            TensorType::F8_E4M3
        } else {
            TensorType::F8_E5M2
        },
        data: fp8_pattern(num_weights, seed, e4m3),
    }
}

/// Build a fully-FP8 synthetic GGUF (`e4m3=true` → E4M3, else E5M2).
fn build_synthetic_fp8_gguf(e4m3: bool) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("CudaFp8PrefillParityTest".to_string()),
    );
    writer.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(H as u32));
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(NUM_LAYERS as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(NQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(NKV as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(INTER as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
    writer.add_metadata(
        "qwen3.context_length",
        MetadataWriteValue::U32(MAX_SEQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![H as u64, VOCAB as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(VOCAB * H, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![H as u64],
        tensor_type: TensorType::F32,
        data: f32_pattern(H, 1.0),
    });
    writer.add_tensor(fp8_tensor(
        "output.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        VOCAB * H,
        0xCAFE_BABE,
        e4m3,
    ));

    for layer in 0..NUM_LAYERS {
        let pfx = format!("blk.{layer}");
        for (suffix, dim) in [
            ("attn_norm", H),
            ("ffn_norm", H),
            ("attn_q_norm", HD),
            ("attn_k_norm", HD),
        ] {
            writer.add_tensor(TensorEntry {
                name: format!("{pfx}.{suffix}.weight"),
                shape: vec![dim as u64],
                tensor_type: TensorType::F32,
                data: f32_pattern(dim, 1.0),
            });
        }
        let s = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_q.weight"),
            vec![H as u64, (NQ * HD) as u64],
            NQ * HD * H,
            s,
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_k.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(1),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_v.weight"),
            vec![H as u64, (NKV * HD) as u64],
            NKV * HD * H,
            s.wrapping_add(2),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.attn_output.weight"),
            vec![(NQ * HD) as u64, H as u64],
            H * NQ * HD,
            s.wrapping_add(3),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.ffn_gate.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(4),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.ffn_up.weight"),
            vec![H as u64, INTER as u64],
            INTER * H,
            s.wrapping_add(5),
            e4m3,
        ));
        writer.add_tensor(fp8_tensor(
            format!("{pfx}.ffn_down.weight"),
            vec![INTER as u64, H as u64],
            H * INTER,
            s.wrapping_add(6),
            e4m3,
        ));
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

/// Prefill `prompt` on the GPU tier and return the layer-0 head-0 key-vector norm
/// at each prompt position, read back from the CPU `self.kv_cache`.
///
/// This is the weight-independent regression gate for the FP8 split-cache bug: the
/// guarded fallback runs the sequential per-token path, which populates
/// `self.kv_cache`, so every norm is non-zero.  If the FP8 batch-prefill path is
/// re-enabled (`OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL=1`), it writes only the
/// GPU-private cache and leaves every CPU KV row at zero — the exact state that
/// makes decode attend over stale/zero prompt KV.
fn gpu_prefill_prompt_kv_norms(bytes: &[u8], prompt: &[u32]) -> Vec<f32> {
    let gguf = GgufFile::parse(bytes).expect("parse gguf");
    let mut model = BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("from_gguf");
    let kernel = KernelDispatcher::with_tier(KernelTier::Gpu);
    model
        .forward_prefill(prompt, 0, &kernel)
        .expect("forward_prefill");
    let keys = model.kv_cache().keys_for(0, 0, prompt.len());
    (0..prompt.len())
        .map(|p| {
            let row = &keys[p * HD..(p + 1) * HD];
            row.iter().map(|x| x * x).sum::<f32>().sqrt()
        })
        .collect()
}

fn run_fp8_parity(e4m3: bool, test_fn: &str) {
    let name = if e4m3 { "FP8(E4M3)" } else { "FP8(E5M2)" };
    if !cuda_available() {
        eprintln!("skip {name}: no CUDA device available");
        record_skip(test_fn);
        return;
    }
    let start = Instant::now();
    let bytes = build_synthetic_fp8_gguf(e4m3);

    // 20-token prompt (> 16) forces the FP8 batch-prefill dispatch; the guard
    // then falls back to the sequential per-token path that populates the CPU KV
    // cache decode reads.
    let prompt: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let decode_inputs: Vec<u32> = (0..6u32).map(|i| (i * 5 + 11) % VOCAB as u32).collect();

    // Regression gate (weight-independent): after the >16-token GPU-tier prefill,
    // every prompt position must have a non-zero key in the CPU self.kv_cache the
    // per-token FP8 decode reads.  The guarded fallback populates it; the disabled
    // FP8 batch-prefill path would write only its GPU-private cache and leave every
    // row at zero.  This assertion fails the instant the guard stops engaging.
    let kv_norms = gpu_prefill_prompt_kv_norms(&bytes, &prompt);
    for (p, &n) in kv_norms.iter().enumerate() {
        assert!(
            n > 1e-6,
            "{name}: CPU KV cache key at prompt position {p} is zero (norm={n}) — the \
             FP8 batch prefill wrote a GPU-private cache decode never reads; the \
             split-cache guard/fallback is not engaged"
        );
    }
    eprintln!(
        "{name}: all {} prompt KV positions populated in CPU cache",
        kv_norms.len()
    );

    let kv = GpuKv::Host;
    let (cpu_prefill, cpu_decode) = drive_teacher_forced(
        &bytes,
        KernelTier::Reference,
        kv,
        name,
        &prompt,
        &decode_inputs,
    );
    let (gpu_prefill, gpu_decode) =
        drive_teacher_forced(&bytes, KernelTier::Gpu, kv, name, &prompt, &decode_inputs);

    // Prefill's last-token logits must agree (validates the fallback prefill math).
    let mut run_max_abs =
        assert_logits_close(&cpu_prefill, &gpu_prefill, &format!("{name} prefill-last"));

    // Each decode step attends over the prompt KV — the decisive anti-corruption
    // check.  With the split-cache bug decode would attend over all-zero KV and
    // these would diverge wildly; the fallback keeps them in parity.
    for (i, (c, g)) in cpu_decode.iter().zip(gpu_decode.iter()).enumerate() {
        run_max_abs = run_max_abs.max(assert_logits_close(c, g, &format!("{name} decode[{i}]")));
    }
    // ...but not bit for bit: that would mean both legs ran the same kernels.
    assert_legs_differ(name, run_max_abs);

    // Greedy sequences (tolerance-free) must be identical.
    let cpu_greedy = drive_greedy(&bytes, KernelTier::Reference, kv, name, &prompt, 8);
    let gpu_greedy = drive_greedy(&bytes, KernelTier::Gpu, kv, name, &prompt, 8);
    assert_eq!(
        cpu_greedy, gpu_greedy,
        "{name}: greedy decode sequence CPU vs CUDA mismatch"
    );
    eprintln!("{name}: greedy sequence match = {cpu_greedy:?}");
    record_pass(test_fn, start.elapsed());
}

#[test]
fn cuda_synthetic_fp8_e4m3_prefill_decode_parity() {
    run_fp8_parity(true, "cuda_synthetic_fp8_e4m3_prefill_decode_parity");
}

#[test]
fn cuda_synthetic_fp8_e5m2_prefill_decode_parity() {
    run_fp8_parity(false, "cuda_synthetic_fp8_e5m2_prefill_decode_parity");
}
