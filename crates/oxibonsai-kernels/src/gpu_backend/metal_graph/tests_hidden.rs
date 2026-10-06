//! Metal tests for the batched prefill runner: the per-kernel GPU timeline
//! (M-18 root cause), the head-free hidden prefill and its request-scoped
//! device KV (MET-05), and the bounded command-buffer wait.

use metal::{Buffer, MTLResourceOptions};

use super::buffers::{alloc_buf, upload_f32};
use super::MetalGraph;
use crate::gpu_backend::metal_prefill::attention::PrefillAttnDims;

/// Deterministic xorshift stream for synthetic weights and activations.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// A shared-storage buffer holding `bytes`.
fn shared_bytes(graph: &MetalGraph, bytes: &[u8]) -> Buffer {
    let buf = alloc_buf(
        &graph.device,
        bytes.len().max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
    .expect("alloc shared buffer");
    // SAFETY: shared storage, `bytes.len()` bytes allocated above.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents() as *mut u8, bytes.len());
    }
    buf
}

/// A shared-storage buffer of `n` f32 activations in `[-1, 1)`.
fn shared_f32(graph: &MetalGraph, n: usize, rng: &mut Rng) -> Buffer {
    let data: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    let buf = alloc_buf(
        &graph.device,
        (n.max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
    .expect("alloc f32 buffer");
    // SAFETY: shared storage sized for `n` floats.
    unsafe { upload_f32(&buf, &data) };
    buf
}

/// Synthetic `Q1_0_g128` SoA weights: `[nb × f16 scale][nb × 16 B sign bits]`.
fn q1_soa(graph: &MetalGraph, n_rows: usize, k: usize, rng: &mut Rng) -> Buffer {
    let nb = n_rows * (k / 128);
    let mut bytes = Vec::with_capacity(nb * 18);
    for _ in 0..nb {
        let scale = half::f16::from_f32(0.01 + 0.01 * (rng.next_u32() % 100) as f32 / 100.0);
        bytes.extend_from_slice(&scale.to_le_bytes());
    }
    for _ in 0..nb * 16 {
        bytes.push(rng.next_u32() as u8);
    }
    shared_bytes(graph, &bytes)
}

/// Synthetic `TQ2_0_g128` SoA weights: `[nb × f16 scale][nb × 32 B codes]`,
/// every 2-bit code in `{0, 1, 2}`.
fn tq2_soa(graph: &MetalGraph, n_rows: usize, k: usize, rng: &mut Rng) -> Buffer {
    let nb = n_rows * (k / 128);
    let mut bytes = Vec::with_capacity(nb * 34);
    for _ in 0..nb {
        let scale = half::f16::from_f32(0.01 + 0.01 * (rng.next_u32() % 100) as f32 / 100.0);
        bytes.extend_from_slice(&scale.to_le_bytes());
    }
    for _ in 0..nb * 32 {
        let mut byte = 0u8;
        for lane in 0..4 {
            byte |= ((rng.next_u32() % 3) as u8) << (2 * lane);
        }
        bytes.push(byte);
    }
    shared_bytes(graph, &bytes)
}

/// GPU start/end of a completed command buffer, in seconds.
///
/// # Safety
/// `cmd` must have completed.
unsafe fn gpu_seconds(cmd: &metal::CommandBufferRef) -> f64 {
    use metal::objc::{msg_send, sel, sel_impl};
    let start: f64 = msg_send![cmd, GPUStartTime];
    let end: f64 = msg_send![cmd, GPUEndTime];
    end - start
}

/// Minimum GPU milliseconds of `reps` command buffers, each holding what
/// `encode` puts in one compute encoder.
fn time_gpu(
    graph: &MetalGraph,
    reps: usize,
    encode: impl Fn(&metal::ComputeCommandEncoderRef),
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode(enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        // SAFETY: completed above.
        best = best.min(unsafe { gpu_seconds(cmd) } * 1e3);
    }
    best
}

/// Real-model layer geometry the timeline is measured at.
struct Geometry {
    name: &'static str,
    layers: usize,
    hidden: usize,
    inter: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
}

const GEOMETRIES: [Geometry; 2] = [
    Geometry {
        name: "Bonsai-8B",
        layers: 36,
        hidden: 4096,
        inter: 12288,
        nq: 32,
        nkv: 8,
        head_dim: 128,
    },
    Geometry {
        name: "Ternary-Bonsai-1.7B",
        layers: 28,
        hidden: 2048,
        inter: 6144,
        nq: 16,
        nkv: 8,
        head_dim: 128,
    },
];

/// The per-kernel GPU timeline of one prefill layer at the real 8B / 1.7B
/// geometries, for batches from 64 to 4096 tokens (M-18 root cause).
///
/// Heavy (it allocates real-size synthetic weights and runs every kernel at
/// every batch size), so it runs only with `OXIBONSAI_PREFILL_TIMELINE=1`.
#[test]
fn metal_prefill_kernel_timeline_at_real_geometries() {
    if std::env::var("OXIBONSAI_PREFILL_TIMELINE").as_deref() != Ok("1") {
        eprintln!("set OXIBONSAI_PREFILL_TIMELINE=1 to print the prefill kernel timeline");
        return;
    }
    let graph = MetalGraph::new().expect("Metal device");
    let attn = graph
        .prefill_attn_pipelines()
        .expect("batched prefill attention pipelines");
    let batches: Vec<usize> = std::env::var("OXIBONSAI_PREFILL_TIMELINE_BATCHES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 128, 256, 512, 1024, 4096]);
    let max_batch = batches.iter().copied().max().unwrap_or(64);
    for geo in &GEOMETRIES {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let h = geo.hidden;
        let qkv_rows = (geo.nq + 2 * geo.nkv) * geo.head_dim;
        let attn_dim = geo.nq * geo.head_dim;
        let w_qkv_q1 = q1_soa(&graph, qkv_rows, h, &mut rng);
        let w_o_q1 = q1_soa(&graph, h, attn_dim, &mut rng);
        let w_gu_q1 = q1_soa(&graph, 2 * geo.inter, h, &mut rng);
        let w_down_q1 = q1_soa(&graph, h, geo.inter, &mut rng);
        let w_qkv_t = tq2_soa(&graph, qkv_rows, h, &mut rng);
        let w_gu_t = tq2_soa(&graph, 2 * geo.inter, h, &mut rng);
        let w_down_t = tq2_soa(&graph, h, geo.inter, &mut rng);
        let norm_w = shared_f32(&graph, h, &mut rng);
        let hd_w = shared_f32(&graph, geo.head_dim, &mut rng);
        let x = shared_f32(&graph, max_batch * geo.inter.max(h), &mut rng);
        let y = shared_f32(&graph, max_batch * 2 * geo.inter.max(qkv_rows), &mut rng);
        let qkv = shared_f32(&graph, max_batch * qkv_rows, &mut rng);
        let attn_out = shared_f32(&graph, max_batch * attn_dim, &mut rng);
        let cos = shared_f32(&graph, max_batch * geo.head_dim / 2, &mut rng);
        let sin = shared_f32(&graph, max_batch * geo.head_dim / 2, &mut rng);
        let kv_elems = geo.nkv * max_batch * geo.head_dim;
        let k_cache = alloc_buf(
            &graph.device,
            (kv_elems * 2) as u64,
            MTLResourceOptions::StorageModePrivate,
        )
        .expect("k cache");
        let v_cache = alloc_buf(
            &graph.device,
            (kv_elems * 2) as u64,
            MTLResourceOptions::StorageModePrivate,
        )
        .expect("v cache");
        eprintln!(
            "\nPREFILL_TIMELINE {} (per layer, GPU ms; x{} layers for the model)",
            geo.name, geo.layers
        );
        eprintln!(
            "{:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}",
            "batch",
            "norm",
            "q1_qkv",
            "q1_o_res",
            "q1_gu_sw",
            "q1_down",
            "attn_prep",
            "attn_flash",
            "t_qkv_v10",
            "t_gu_v10",
            "q1_us/tok",
            "t_us/tok"
        );
        for &bs in &batches {
            let b = bs as u32;
            let norm = time_gpu(&graph, 2, |e| {
                graph.dispatch_batched_rmsnorm(e, &x, &norm_w, &y, 1e-6, h as u32, b)
            });
            let q1_qkv = time_gpu(&graph, 2, |e| {
                graph.dispatch_gemm_q1_v7(e, &w_qkv_q1, &x, &y, qkv_rows as u32, h as u32, b)
            });
            let q1_o = time_gpu(&graph, 2, |e| {
                graph.dispatch_gemm_q1_v7_residual(
                    e,
                    &w_o_q1,
                    &x,
                    &y,
                    h as u32,
                    attn_dim as u32,
                    b,
                    &y,
                )
            });
            let q1_gu = time_gpu(&graph, 2, |e| {
                graph.dispatch_fused_gate_up_swiglu_gemm(
                    e,
                    &w_gu_q1,
                    &x,
                    &y,
                    geo.inter as u32,
                    h as u32,
                    b,
                )
            });
            let q1_down = time_gpu(&graph, 2, |e| {
                graph.dispatch_gemm_q1_v7_residual(
                    e,
                    &w_down_q1,
                    &x,
                    &y,
                    h as u32,
                    geo.inter as u32,
                    b,
                    &y,
                )
            });
            let dims = PrefillAttnDims {
                nq: geo.nq as u32,
                nkv: geo.nkv as u32,
                heads_per_group: (geo.nq / geo.nkv) as u32,
                head_dim: geo.head_dim as u32,
                eps: 1e-6,
                max_seq: max_batch as u32,
                pos_start: 0,
                batch_size: b,
                scale: 1.0 / (geo.head_dim as f32).sqrt(),
                layer_offset: 0,
            };
            let prep = time_gpu(&graph, 2, |e| {
                graph.dispatch_prefill_qkv_prepare(
                    attn, e, &qkv, &hd_w, &hd_w, &cos, &sin, &k_cache, &v_cache, &dims,
                )
            });
            let flash = time_gpu(&graph, 2, |e| {
                graph.dispatch_prefill_flash_attention(
                    attn, e, &qkv, &k_cache, &v_cache, &attn_out, &dims,
                )
            });
            let t_qkv = time_gpu(&graph, 2, |e| {
                graph.dispatch_gemm_tq2_v10(e, &w_qkv_t, &x, &y, qkv_rows as u32, h as u32, b)
            });
            let t_gu = time_gpu(&graph, 2, |e| {
                graph.dispatch_gemm_tq2_v10(e, &w_gu_t, &x, &y, (2 * geo.inter) as u32, h as u32, b)
            });
            let t_down = time_gpu(&graph, 2, |e| {
                graph.dispatch_gemm_tq2_v10(e, &w_down_t, &x, &y, h as u32, geo.inter as u32, b)
            });
            // The row-wise ternary kernel on the same three projections, for
            // the tiled/row-wise crossover.
            let r_layer = time_gpu(&graph, 2, |e| {
                graph.dispatch_gemm_tq2_v7(e, &w_qkv_t, &x, &y, qkv_rows as u32, h as u32, b);
                graph.dispatch_gemm_tq2_v7(e, &w_gu_t, &x, &y, (2 * geo.inter) as u32, h as u32, b);
                graph.dispatch_gemm_tq2_v7(e, &w_down_t, &x, &y, h as u32, geo.inter as u32, b);
            });
            let t2s = graph
                .prefill_tq2_tiled_pipeline()
                .expect("gemm_tq2_g128_simdgroup pipeline")
                .clone();
            let t2s_dispatch =
                |e: &metal::ComputeCommandEncoderRef, w: &Buffer, n_rows: usize, k: usize| {
                    e.set_compute_pipeline_state(&t2s);
                    e.set_buffer(0, Some(w), 0);
                    e.set_buffer(1, Some(&x), 0);
                    e.set_buffer(2, Some(&y), 0);
                    // SAFETY: scalar bindings at indices the kernel declares.
                    unsafe {
                        super::buffers::set_scalar(e, 3, &(n_rows as u32));
                        super::buffers::set_scalar(e, 4, &b);
                        super::buffers::set_scalar(e, 5, &(k as u32));
                    }
                    e.dispatch_thread_groups(
                        metal::MTLSize::new(n_rows.div_ceil(64) as u64, bs.div_ceil(64) as u64, 1),
                        metal::MTLSize::new(128, 1, 1),
                    );
                };
            let s_layer = time_gpu(&graph, 2, |e| {
                t2s_dispatch(e, &w_qkv_t, qkv_rows, h);
                t2s_dispatch(e, &w_gu_t, 2 * geo.inter, h);
                t2s_dispatch(e, &w_down_t, h, geo.inter);
            });
            eprintln!(
                "{bs:>6} ternary qkv+gu+down: v10 {:.3} ms, prefill-tiled {s_layer:.3} ms, \
                 row-wise {r_layer:.3} ms",
                t_qkv + t_gu + t_down
            );
            let q1s = graph
                .pipeline_for("gemm_q1_g128_simdgroup")
                .expect("gemm_q1_g128_simdgroup pipeline");
            let q1s_dispatch = |e: &metal::ComputeCommandEncoderRef,
                                w: &Buffer,
                                n_rows: usize,
                                k: usize,
                                mode: u32| {
                e.set_compute_pipeline_state(&q1s);
                e.set_buffer(0, Some(w), 0);
                e.set_buffer(1, Some(&x), 0);
                e.set_buffer(2, Some(&y), 0);
                e.set_buffer(6, Some(&y), 0);
                // SAFETY: scalar bindings at indices the kernel declares.
                unsafe {
                    super::buffers::set_scalar(e, 3, &(n_rows as u32));
                    super::buffers::set_scalar(e, 4, &b);
                    super::buffers::set_scalar(e, 5, &(k as u32));
                    super::buffers::set_scalar(e, 7, &mode);
                }
                e.dispatch_thread_groups(
                    metal::MTLSize::new(n_rows.div_ceil(64) as u64, bs.div_ceil(64) as u64, 1),
                    metal::MTLSize::new(128, 1, 1),
                );
            };
            let s_qkv = time_gpu(&graph, 2, |e| q1s_dispatch(e, &w_qkv_q1, qkv_rows, h, 0));
            let s_o = time_gpu(&graph, 2, |e| q1s_dispatch(e, &w_o_q1, h, attn_dim, 1));
            let s_gu = time_gpu(&graph, 2, |e| {
                q1s_dispatch(e, &w_gu_q1, 2 * geo.inter, h, 0);
                graph.dispatch_batched_swiglu(e, &y, &x, geo.inter as u32, b);
            });
            let s_down = time_gpu(&graph, 2, |e| q1s_dispatch(e, &w_down_q1, h, geo.inter, 1));
            let s_layer = 2.0 * norm + s_qkv + s_o + s_gu + s_down + prep + flash;
            eprintln!(
                "{bs:>6} simdgroup-Q1: qkv {s_qkv:.3} o {s_o:.3} gu+swiglu {s_gu:.3} down \
                 {s_down:.3} -> {:.1} us/token",
                s_layer * geo.layers as f64 * 1e3 / bs as f64
            );
            let q1_layer = 2.0 * norm + q1_qkv + q1_o + q1_gu + q1_down + prep + flash;
            // The O projection is square in `hidden`; time it with the QKV
            // kernel's per-row cost scaled by the row ratio.
            let t_o = t_qkv * h as f64 / qkv_rows as f64;
            let t_layer = 2.0 * norm + t_qkv + t_o + t_gu + t_down + prep + flash;
            eprintln!(
                "{bs:>6} {norm:>9.3} {q1_qkv:>9.3} {q1_o:>9.3} {q1_gu:>9.3} {q1_down:>9.3} \
                 {prep:>9.3} {flash:>9.3} {t_qkv:>9.3} {t_gu:>9.3} {:>10.1} {:>10.1}",
                q1_layer * geo.layers as f64 * 1e3 / bs as f64,
                t_layer * geo.layers as f64 * 1e3 / bs as f64,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Functional tests (run on every Metal host; skip without a device)
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::error::MetalWeightHandle;
use crate::gpu_backend::metal_full_layer::types::WeightKind;
use crate::gpu_backend::metal_full_layer::{
    CachedLayerWeights, CachedModelWeights, CachedQ1Weights, CachedTernaryLayerWeights,
    CachedTernaryWeights,
};
use crate::gpu_backend::metal_prefill::functions::{
    layer_refs, PrefillFormat, PrefillGemm, PrefillLayerHandles, PrefillRun, PrefillTail,
};
use crate::gpu_backend::metal_prefill::hidden::{
    try_metal_full_forward_prefill_hidden_cached,
    try_metal_full_forward_prefill_hidden_cached_micro, HiddenPrefillInput, HiddenPrefillShape,
};
use crate::gpu_backend::metal_prefill::types::LayerConfig;

/// An isolated Metal graph bound to this thread for the test's duration, or
/// `None` (skip) without a Metal device.
fn private_graph() -> Option<(Arc<MetalGraph>, super::SessionScope)> {
    metal::Device::system_default()?;
    let graph = Arc::new(
        MetalGraph::new().unwrap_or_else(|e| panic!("the Metal graph must build here: {e}")),
    );
    let scope = MetalGraph::bind_scope(Arc::clone(&graph));
    Some((graph, scope))
}

fn handle(buffer: Buffer, kind: WeightKind) -> Arc<MetalWeightHandle> {
    let byte_len = buffer.length() as usize;
    Arc::new(MetalWeightHandle {
        buffer,
        byte_len,
        kind,
    })
}

/// Near-1 RMSNorm weights.
fn norm_weights(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| 1.0 + 0.1 * rng.unit()).collect()
}

fn f32_handle(graph: &MetalGraph, data: &[f32]) -> Arc<MetalWeightHandle> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    handle(shared_bytes(graph, &bytes), WeightKind::RawF32)
}

/// A small synthetic model: its cached fused weight set, output norm, LM head
/// (same format) and geometry.
struct Fixture {
    cached: CachedModelWeights,
    final_norm: Vec<f32>,
    final_norm_handle: Arc<MetalWeightHandle>,
    lm_head: Arc<MetalWeightHandle>,
    vocab: usize,
    shape: HiddenPrefillShape,
    format: PrefillFormat,
}

const FIX_HIDDEN: usize = 256;
const FIX_INTER: usize = 512;
const FIX_NQ: usize = 4;
const FIX_NKV: usize = 2;
const FIX_HEAD_DIM: usize = 64;
const FIX_LAYERS: usize = 2;
const FIX_VOCAB: usize = 96;

fn quant_handle(
    graph: &MetalGraph,
    format: PrefillFormat,
    rows: usize,
    k: usize,
    rng: &mut Rng,
) -> Arc<MetalWeightHandle> {
    match format {
        PrefillFormat::OneBit => handle(q1_soa(graph, rows, k, rng), WeightKind::Q1Soa),
        PrefillFormat::Ternary => handle(tq2_soa(graph, rows, k, rng), WeightKind::Tq2Soa),
    }
}

fn fixture(graph: &MetalGraph, format: PrefillFormat, seed: u64) -> Fixture {
    let mut rng = Rng(seed | 1);
    let h = FIX_HIDDEN;
    let qkv_rows = (FIX_NQ + 2 * FIX_NKV) * FIX_HEAD_DIM;
    let attn = FIX_NQ * FIX_HEAD_DIM;
    let mut q1_layers = Vec::new();
    let mut tern_layers = Vec::new();
    for _ in 0..FIX_LAYERS {
        let attn_norm = f32_handle(graph, &norm_weights(h, &mut rng));
        let fused_qkv = quant_handle(graph, format, qkv_rows, h, &mut rng);
        let q_norm = f32_handle(graph, &norm_weights(FIX_HEAD_DIM, &mut rng));
        let k_norm = f32_handle(graph, &norm_weights(FIX_HEAD_DIM, &mut rng));
        let attn_proj = quant_handle(graph, format, h, attn, &mut rng);
        let ffn_norm = f32_handle(graph, &norm_weights(h, &mut rng));
        let gate_up = quant_handle(graph, format, 2 * FIX_INTER, h, &mut rng);
        let down = quant_handle(graph, format, h, FIX_INTER, &mut rng);
        match format {
            PrefillFormat::OneBit => q1_layers.push(CachedLayerWeights {
                attn_norm,
                fused_qkv,
                q_norm,
                k_norm,
                attn_proj,
                ffn_norm,
                gate_up,
                down,
            }),
            PrefillFormat::Ternary => tern_layers.push(CachedTernaryLayerWeights {
                attn_norm,
                fused_qkv,
                q_norm,
                k_norm,
                attn_proj,
                ffn_norm,
                gate_up,
                down,
            }),
        }
    }
    let final_norm = norm_weights(h, &mut rng);
    let final_norm_handle = f32_handle(graph, &final_norm);
    let lm_head = quant_handle(graph, format, FIX_VOCAB, h, &mut rng);
    let cached = match format {
        PrefillFormat::OneBit => CachedModelWeights::Q1(CachedQ1Weights {
            layers: q1_layers,
            final_norm: Arc::clone(&final_norm_handle),
            lm_head: Arc::clone(&lm_head),
        }),
        PrefillFormat::Ternary => CachedModelWeights::Ternary(CachedTernaryWeights {
            model_epoch: 0,
            layers: tern_layers,
            final_norm: Some(Arc::clone(&final_norm_handle)),
            lm_head: Some(Arc::clone(&lm_head)),
            lm_head_out_features: FIX_VOCAB,
        }),
    };
    Fixture {
        cached,
        final_norm,
        final_norm_handle,
        lm_head,
        vocab: FIX_VOCAB,
        shape: HiddenPrefillShape {
            hidden_size: h,
            intermediate_size: FIX_INTER,
            nq: FIX_NQ,
            nkv: FIX_NKV,
            head_dim: FIX_HEAD_DIM,
            eps: 1e-6,
            final_norm_eps: 1e-6,
        },
        format,
    }
}

/// Embeddings and RoPE tables of a `batch`-token input at positions
/// `pos_start..`.
struct Inputs {
    hidden: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn inputs(batch: usize, pos_start: usize, seed: u64) -> Inputs {
    let mut rng = Rng(seed | 1);
    let half = FIX_HEAD_DIM / 2;
    let mut cos = Vec::with_capacity(batch * half);
    let mut sin = Vec::with_capacity(batch * half);
    for t in 0..batch {
        let pos = (pos_start + t) as f64;
        for i in 0..half {
            let theta = pos / 10_000f64.powf(2.0 * i as f64 / FIX_HEAD_DIM as f64);
            cos.push(theta.cos() as f32);
            sin.push(theta.sin() as f32);
        }
    }
    Inputs {
        hidden: (0..batch * FIX_HIDDEN).map(|_| rng.unit()).collect(),
        cos,
        sin,
    }
}

fn layer_handles(cached: &CachedModelWeights) -> Vec<PrefillLayerHandles<'_>> {
    use crate::gpu_backend::metal_full_layer::functions_3::{q1_layer_refs, ternary_layer_refs};
    match cached {
        CachedModelWeights::Q1(q1) => q1_layer_refs(&q1.layers),
        CachedModelWeights::Ternary(t) => ternary_layer_refs(&t.layers),
    }
}

fn fix_config(max_seq: usize) -> LayerConfig {
    LayerConfig {
        hidden_size: FIX_HIDDEN,
        intermediate_size: FIX_INTER,
        n_q_heads: FIX_NQ,
        n_kv_heads: FIX_NKV,
        head_dim: FIX_HEAD_DIM,
        eps: 1e-6,
        max_seq_len: max_seq,
    }
}

/// Last-row logits of a `fix` prefill of `inp` at `pos_start` in `graph`,
/// micro-batched at `micro` rows.
fn logits_prefill(
    graph: &MetalGraph,
    fix: &Fixture,
    inp: &Inputs,
    pos_start: usize,
    micro: usize,
    max_seq: usize,
) -> Result<Vec<f32>, super::MetalGraphError> {
    let handles = layer_handles(&fix.cached);
    let refs = layer_refs(&handles);
    let batch = inp.hidden.len() / FIX_HIDDEN;
    let run = PrefillRun {
        format: fix.format,
        hidden_batch: &inp.hidden,
        cos_table: &inp.cos,
        sin_table: &inp.sin,
        pos_start,
        batch_size: batch,
        layers: &refs,
        config: fix_config(max_seq),
        micro_batch: micro,
        label: "test_prefill_logits",
    };
    let mut logits = Vec::new();
    graph.run_prefill(
        &run,
        PrefillTail::LastRow {
            final_norm: &fix.final_norm_handle.buffer,
            eps: 1e-6,
            lm_head: &fix.lm_head.buffer,
            out_features: fix.vocab,
            logits_out: Some(&mut logits),
            greedy_out: None,
        },
    )?;
    Ok(logits)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Copy a private GPU buffer's first `len` bytes back to the host.
fn read_private(graph: &MetalGraph, buf: &Buffer, len: u64) -> Vec<u8> {
    let staging = alloc_buf(&graph.device, len, MTLResourceOptions::StorageModeShared)
        .expect("staging buffer");
    let cmd = graph.command_queue.new_command_buffer();
    let blit = cmd.new_blit_command_encoder();
    blit.copy_from_buffer(buf, 0, &staging, 0, len);
    blit.end_encoding();
    super::buffers::commit_and_wait(cmd, "test_read_private").expect("blit");
    let mut out = vec![0u8; len as usize];
    // SAFETY: shared staging buffer of `len` bytes, written by the blit.
    unsafe {
        std::ptr::copy_nonoverlapping(staging.contents() as *const u8, out.as_mut_ptr(), out.len())
    };
    out
}

/// The K and V bytes of `graph`'s device KV cache.
fn kv_bytes(graph: &MetalGraph) -> (Vec<u8>, Vec<u8>) {
    let guard = graph.kv_cache.lock().expect("kv lock");
    let kv = guard.as_ref().expect("a KV cache is allocated");
    let len = kv.k_cache.length();
    (
        read_private(graph, &kv.k_cache, len),
        read_private(graph, &kv.v_cache, len),
    )
}

/// Dequantised Q1 row `r` of a `[n_rows x k]` SoA buffer.
fn q1_row(buf: &Buffer, n_rows: usize, k: usize, r: usize) -> Vec<f64> {
    let bpr = k / 128;
    let base = buf.contents() as *const u8;
    let data_offset = n_rows * bpr * 2;
    let mut row = Vec::with_capacity(k);
    for b in 0..bpr {
        let block = r * bpr + b;
        // SAFETY: shared-storage SoA buffer of `n_rows * bpr` blocks.
        let (d, bits) = unsafe {
            let d = half::f16::from_le_bytes([*base.add(block * 2), *base.add(block * 2 + 1)]);
            let bits = std::slice::from_raw_parts(base.add(data_offset + block * 16), 16);
            (f64::from(d.to_f32()), bits)
        };
        for i in 0..128 {
            row.push(if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                d
            } else {
                -d
            });
        }
    }
    row
}

/// Run the tiled Q1 GEMM once: `out = [out +] W·x` for `m` columns.
#[allow(clippy::too_many_arguments)]
fn run_tiled_q1(
    graph: &MetalGraph,
    pso: &metal::ComputePipelineState,
    w: &Buffer,
    x: &Buffer,
    out: &Buffer,
    n_rows: usize,
    k: usize,
    m: usize,
    mode: u32,
) {
    let cmd = graph.command_queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(w), 0);
    enc.set_buffer(1, Some(x), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_buffer(6, Some(out), 0);
    // SAFETY: scalar bindings at the kernel's declared indices.
    unsafe {
        super::buffers::set_scalar(enc, 3, &(n_rows as u32));
        super::buffers::set_scalar(enc, 4, &(m as u32));
        super::buffers::set_scalar(enc, 5, &(k as u32));
        super::buffers::set_scalar(enc, 7, &mode);
    }
    enc.dispatch_thread_groups(
        metal::MTLSize::new(n_rows.div_ceil(64) as u64, m.div_ceil(64) as u64, 1),
        metal::MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    super::buffers::commit_and_wait(cmd, "test_tiled_q1").expect("tiled GEMM");
}

/// M-18: the tiled Q1 GEMM against an f64 CPU reference (plain and residual,
/// tile-multiple and ragged shapes) and against the row-wise kernel it
/// replaces on the prefill.
#[test]
fn metal_tiled_q1_gemm_matches_a_cpu_reference_and_the_rowwise_kernel() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    let pso = graph
        .prefill_q1_tiled_pipeline()
        .expect("the tiled Q1 GEMM must build on a Metal device")
        .clone();
    let mut rng = Rng(0xC0FF_EE11);
    let shapes: [(usize, usize, usize); 4] = [
        (192, 256, 5),
        (64, 128, 64),
        (130, 384, 70),
        (200, 256, 131),
    ];
    for &(n_rows, k, m) in &shapes {
        let w = q1_soa(&graph, n_rows, k, &mut rng);
        let x_host: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
        let r_host: Vec<f32> = (0..m * n_rows).map(|_| rng.unit()).collect();
        let x = shared_f32(&graph, m * k, &mut Rng(1));
        // SAFETY: shared buffer sized for `m * k` floats.
        unsafe { upload_f32(&x, &x_host) };
        for mode in [0u32, 1] {
            let out = shared_f32(&graph, m * n_rows, &mut Rng(2));
            // SAFETY: shared buffer sized for `m * n_rows` floats.
            unsafe { upload_f32(&out, &r_host) };
            run_tiled_q1(&graph, &pso, &w, &x, &out, n_rows, k, m, mode);
            let mut got = vec![0f32; m * n_rows];
            // SAFETY: shared buffer of `m * n_rows` floats.
            unsafe { super::buffers::download_f32(&out, &mut got) };
            for r in 0..n_rows {
                let row = q1_row(&w, n_rows, k, r);
                for c in 0..m {
                    let xs = &x_host[c * k..(c + 1) * k];
                    let dot: f64 = row.iter().zip(xs).map(|(a, b)| a * f64::from(*b)).sum();
                    let mag: f64 = row
                        .iter()
                        .zip(xs)
                        .map(|(a, b)| (a * f64::from(*b)).abs())
                        .sum();
                    let want = if mode == 1 {
                        dot + f64::from(r_host[c * n_rows + r])
                    } else {
                        dot
                    };
                    let g = f64::from(got[c * n_rows + r]);
                    assert!(
                        (g - want).abs() <= 1e-5 * (mag + 1.0),
                        "n={n_rows} k={k} m={m} mode={mode} row {r} col {c}: {g} vs {want}"
                    );
                }
            }
        }
        let out_v7 = shared_f32(&graph, m * n_rows, &mut Rng(3));
        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        graph.dispatch_gemm_q1_v7(enc, &w, &x, &out_v7, n_rows as u32, k as u32, m as u32);
        enc.end_encoding();
        super::buffers::commit_and_wait(cmd, "test_v7_q1").expect("v7 GEMM");
        let mut v7 = vec![0f32; m * n_rows];
        // SAFETY: shared buffer of `m * n_rows` floats.
        unsafe { super::buffers::download_f32(&out_v7, &mut v7) };
        let out_t = shared_f32(&graph, m * n_rows, &mut Rng(4));
        run_tiled_q1(&graph, &pso, &w, &x, &out_t, n_rows, k, m, 0);
        let mut tiled = vec![0f32; m * n_rows];
        // SAFETY: shared buffer of `m * n_rows` floats.
        unsafe { super::buffers::download_f32(&out_t, &mut tiled) };
        for (i, (a, b)) in tiled.iter().zip(&v7).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
                "tiled vs row-wise at {i}: {a} vs {b}"
            );
        }
    }
}

/// The prefill's tiled ternary GEMM computes exactly what
/// `gemm_tq2_g128_v10_simdgroup` computes — same tile, same K slicing, same
/// MAC order, the same exactly-dequantised weights — bit for bit, on
/// tile-multiple and ragged shapes.
#[test]
fn metal_tiled_tq2_gemm_is_bit_identical_to_v10() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    let pso = graph
        .prefill_tq2_tiled_pipeline()
        .expect("the tiled ternary GEMM must build on a Metal device")
        .clone();
    let mut rng = Rng(0x7A2_5EED);
    let shapes: [(usize, usize, usize); 4] = [
        (192, 256, 5),
        (64, 128, 64),
        (130, 384, 70),
        (1024, 512, 131),
    ];
    for &(n_rows, k, m) in &shapes {
        let w = tq2_soa(&graph, n_rows, k, &mut rng);
        let x = shared_f32(&graph, m * k, &mut rng);
        let run = |use_v10: bool| -> Vec<f32> {
            let out = shared_f32(&graph, m * n_rows, &mut Rng(5));
            let cmd = graph.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            if use_v10 {
                graph.dispatch_gemm_tq2_v10(enc, &w, &x, &out, n_rows as u32, k as u32, m as u32);
            } else {
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&w), 0);
                enc.set_buffer(1, Some(&x), 0);
                enc.set_buffer(2, Some(&out), 0);
                // SAFETY: scalar bindings at the kernel's declared indices.
                unsafe {
                    super::buffers::set_scalar(enc, 3, &(n_rows as u32));
                    super::buffers::set_scalar(enc, 4, &(m as u32));
                    super::buffers::set_scalar(enc, 5, &(k as u32));
                }
                enc.dispatch_thread_groups(
                    metal::MTLSize::new(n_rows.div_ceil(64) as u64, m.div_ceil(64) as u64, 1),
                    metal::MTLSize::new(128, 1, 1),
                );
            }
            enc.end_encoding();
            super::buffers::commit_and_wait(cmd, "test_tiled_tq2").expect("ternary GEMM");
            let mut host = vec![0f32; m * n_rows];
            // SAFETY: shared buffer of `m * n_rows` floats.
            unsafe { super::buffers::download_f32(&out, &mut host) };
            host
        };
        let tiled = run(false);
        let v10 = run(true);
        assert!(tiled.iter().all(|v| v.is_finite()));
        assert!(
            tiled.iter().any(|v| *v != 0.0),
            "n={n_rows} k={k} m={m}: written"
        );
        assert_eq!(
            bits(&tiled),
            bits(&v10),
            "n={n_rows} k={k} m={m}: the tiled ternary GEMM differs from v10"
        );
    }
}

#[test]
fn metal_prefill_gemm_family_is_chosen_once_per_request() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    for format in [PrefillFormat::OneBit, PrefillFormat::Ternary] {
        for batch in [1, 4, 7] {
            assert_eq!(
                graph.choose_prefill_gemm(format, batch),
                PrefillGemm::Rowwise,
                "{format:?}: a {batch}-row batch (below the 8-row tiled threshold, like a \
                 speculative verify at the default draft length of 4, which is 5 rows) stays on \
                 the row-wise kernel"
            );
        }
        for batch in [8, 63, 64, 300, 4096] {
            assert_eq!(
                graph.choose_prefill_gemm(format, batch),
                PrefillGemm::Tiled,
                "{format:?}: a {batch}-row batch runs the tiled kernel"
            );
        }
    }
}

/// MET-05 / micro-batching: the 300-token hidden prefill in 128-row
/// micro-batches (128 + 128 + 44) is bit-for-bit the single-batch pass, on
/// both weight formats.
#[test]
fn metal_hidden_prefill_micro_batches_are_bit_identical_to_one_batch() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    for format in [PrefillFormat::OneBit, PrefillFormat::Ternary] {
        let fix = fixture(&graph, format, 0x5EED_0300);
        let inp = inputs(300, 0, 77);
        let input = HiddenPrefillInput {
            hidden_batch: &inp.hidden,
            cos_table: &inp.cos,
            sin_table: &inp.sin,
            batch_size: 300,
        };
        let mut micro = Vec::new();
        try_metal_full_forward_prefill_hidden_cached(
            &input,
            &fix.cached,
            &fix.final_norm,
            &fix.shape,
            &mut micro,
        )
        .expect("micro-batched hidden prefill");
        let mut single = Vec::new();
        try_metal_full_forward_prefill_hidden_cached_micro(
            &input,
            &fix.cached,
            &fix.final_norm,
            &fix.shape,
            300,
            &mut single,
        )
        .expect("single-batch hidden prefill");
        assert_eq!(micro.len(), 300 * FIX_HIDDEN);
        assert!(
            micro.iter().all(|v| v.is_finite()),
            "{format:?}: finite rows"
        );
        assert_eq!(
            bits(&micro),
            bits(&single),
            "{format:?}: 128-row micro-batches must be bit-identical to one batch"
        );
        let last = &micro[299 * FIX_HIDDEN..];
        assert!(
            last.iter().any(|v| *v != 0.0),
            "{format:?}: last row written"
        );
    }
}

/// MET-05: a hidden prefill between two steps of a decode session leaves that
/// session's device KV cache bit-identical, and the session continues to the
/// same logits a session that never saw the embedding computes.
#[test]
fn metal_hidden_prefill_leaves_a_decode_sessions_kv_bit_identical() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    for format in [PrefillFormat::OneBit, PrefillFormat::Ternary] {
        let fix = fixture(&graph, format, 0x0D0C_5E55);
        let max_seq = 96;
        let prompt = inputs(40, 0, 5);
        let reference = logits_prefill(&graph, &fix, &prompt, 0, 512, max_seq)
            .expect("the decode session's prefill");
        let before = kv_bytes(&graph);
        let live_before = graph.sessions_on_this_device();

        let other = inputs(150, 0, 9);
        let mut rows = Vec::new();
        try_metal_full_forward_prefill_hidden_cached(
            &HiddenPrefillInput {
                hidden_batch: &other.hidden,
                cos_table: &other.cos,
                sin_table: &other.sin,
                batch_size: 150,
            },
            &fix.cached,
            &fix.final_norm,
            &fix.shape,
            &mut rows,
        )
        .expect("interleaved hidden prefill");
        assert_eq!(rows.len(), 150 * FIX_HIDDEN);

        let after = kv_bytes(&graph);
        assert!(before.0 == after.0, "{format:?}: K cache changed");
        assert!(before.1 == after.1, "{format:?}: V cache changed");
        assert_eq!(
            graph.sessions_on_this_device(),
            live_before,
            "{format:?}: the request-scoped session is gone after the call"
        );
        let next = inputs(1, 40, 13);
        let cont_a = logits_prefill(&graph, &fix, &next, 40, 512, max_seq).expect("continue");
        let fresh = Arc::new(MetalGraph::new().expect("second graph"));
        let _fresh_scope = MetalGraph::bind_scope(Arc::clone(&fresh));
        let replay = logits_prefill(&fresh, &fix, &prompt, 0, 512, max_seq).expect("replay");
        assert_eq!(bits(&replay), bits(&reference));
        let cont_b = logits_prefill(&fresh, &fix, &next, 40, 512, max_seq).expect("continue");
        assert_eq!(
            bits(&cont_a),
            bits(&cont_b),
            "{format:?}: the decode session continues as if no embedding ran"
        );
    }
}

/// The logits prefill is micro-batch invariant too: 200 rows in 64-row
/// micro-batches give the one-batch logits bit for bit.
#[test]
fn metal_logits_prefill_micro_batches_match_one_batch() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    for format in [PrefillFormat::OneBit, PrefillFormat::Ternary] {
        let fix = fixture(&graph, format, 0x0010_6175);
        let inp = inputs(200, 0, 21);
        let one = logits_prefill(&graph, &fix, &inp, 0, 512, 256).expect("one batch");
        let micro = logits_prefill(&graph, &fix, &inp, 0, 64, 256).expect("micro-batched");
        assert_eq!(bits(&one), bits(&micro), "{format:?}");
        assert!(one.iter().all(|v| v.is_finite()));
    }
}

/// M-18: a bounded wait on a command buffer that never completes returns the
/// typed timeout at its deadline instead of blocking.
#[test]
fn metal_bounded_wait_gives_up_at_the_deadline_with_a_typed_timeout() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    // Never committed: it can only ever report `NotEnqueued`.
    let cmd = graph.command_queue.new_command_buffer();
    let started = Instant::now();
    let err = super::buffers::wait_bounded(
        cmd,
        "test_never_committed",
        started + Duration::from_millis(30),
        false,
    )
    .expect_err("an uncommitted command buffer never completes");
    assert!(err.is_command_buffer_timeout(), "{err}");
    assert!(started.elapsed() >= Duration::from_millis(30));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the wait gave up at its deadline"
    );
    let done = graph.command_queue.new_command_buffer();
    super::buffers::commit_and_wait_bounded(
        done,
        "test_empty",
        Instant::now() + Duration::from_secs(30),
    )
    .expect("an empty command buffer completes");
    let failed = super::MetalGraphError::CommandBufferFailed {
        what: "t",
        status: metal::MTLCommandBufferStatus::Error,
        error: None,
    };
    assert!(
        !failed.is_command_buffer_timeout(),
        "a GPU fault is not a timeout"
    );
}

/// M-18: an injected timeout on a prefill parks its command buffer; the next
/// prefill in the same session drains it and computes exactly what a fresh
/// session computes.
#[test]
fn metal_timed_out_prefill_is_parked_and_drained_by_the_next_one() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    let fix = fixture(&graph, PrefillFormat::OneBit, 0x0071_31E0);
    let inp = inputs(90, 0, 31);
    MetalGraph::force_prefill_timeouts(1);
    let err = logits_prefill(&graph, &fix, &inp, 0, 512, 128)
        .expect_err("the injected timeout surfaces as an error");
    MetalGraph::force_prefill_timeouts(0);
    assert!(err.is_command_buffer_timeout(), "{err}");
    assert!(
        graph.prefill_inflight.lock().expect("slot").is_some(),
        "the timed-out command buffer is parked in the session"
    );
    let calls = MetalGraph::prefill_fused_call_count();
    let again = logits_prefill(&graph, &fix, &inp, 0, 512, 128).expect("the next prefill");
    assert!(MetalGraph::prefill_fused_call_count() > calls);
    assert!(
        graph.prefill_inflight.lock().expect("slot").is_none(),
        "the next prefill drained the parked command buffer"
    );
    let fresh = Arc::new(MetalGraph::new().expect("second graph"));
    let _fresh_scope = MetalGraph::bind_scope(Arc::clone(&fresh));
    let reference = logits_prefill(&fresh, &fix, &inp, 0, 512, 128).expect("fresh session");
    assert_eq!(bits(&again), bits(&reference));
}

/// `prefill_flash_attention`'s online softmax as first written: one thread
/// walks each query row.
const ORIGINAL_SOFTMAX: &str = r#"        // Causal online softmax (one query row per strided thread).
        for (uint r = lid; r < PFA_BQ; r += PFA_THREADS) {
            float c = 1.0f;
            // Keys of THIS tile visible to row r: j < row_valid.
            uint row_valid = 0u;
            if (r < q_valid) {
                const uint qpos = pos_start + q0 + r;
                if (k0 <= qpos) {
                    row_valid = min(k_valid, qpos - k0 + 1u);
                }
            }
            if (row_valid == 0u) {
                // Padded query row, or a key-tile entirely in this row's
                // future: contribute nothing and leave (m, l, O) untouched.
                for (uint j = 0u; j < PFA_BK; j++) {
                    Ssh[r * PFA_BK + j] = 0.0f;
                }
            } else {
                float tile_max = -INFINITY;
                for (uint j = 0u; j < row_valid; j++) {
                    tile_max = max(tile_max, Ssh[r * PFA_BK + j] * scale);
                }
                const float m_old = mrow[r];
                const float m_new = max(m_old, tile_max);
                float tile_sum = 0.0f;
                for (uint j = 0u; j < PFA_BK; j++) {
                    float p = 0.0f;
                    if (j < row_valid) {
                        p = exp(Ssh[r * PFA_BK + j] * scale - m_new);
                    }
                    Ssh[r * PFA_BK + j] = p;
                    tile_sum += p;
                }
                c = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
                lrow[r] = lrow[r] * c + tile_sum;
                mrow[r] = m_new;
            }
            corr[r] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"#;

/// The original kernel's O rescale: `diag(corr) * O` on every key tile ...
const ORIGINAL_RESCALE: &str = r#"        // Rescale the O accumulator: O = diag(corr) * O.
        {
            const uint dbase = (sgid * PFA_MFRAGS) * (PFA_FRAG * PFA_FRAG);
            const uint lane = lid % 32u;
            for (uint idx = lane; idx < PFA_MFRAGS * PFA_FRAG * PFA_FRAG; idx += 32u) {
                const uint mi = idx / (PFA_FRAG * PFA_FRAG);
                const uint e  = idx % (PFA_FRAG * PFA_FRAG);
                const uint rr = e / PFA_FRAG;
                const uint cc = e % PFA_FRAG;
                float val = 0.0f;
                if (rr == cc) {
                    val = corr[sg_m0 + mi * PFA_FRAG + rr];
                }
                diagsh[dbase + idx] = val;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint dbase = (sgid * PFA_MFRAGS) * (PFA_FRAG * PFA_FRAG);
            for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                simdgroup_float8x8 dfrag;
                simdgroup_load(dfrag, diagsh + dbase + mi * (PFA_FRAG * PFA_FRAG), PFA_FRAG);
                for (uint di = 0u; di < d_frags; di++) {
                    simdgroup_float8x8 tmp;
                    simdgroup_multiply(tmp, dfrag, oacc[mi][di]);
                    oacc[mi][di] = tmp;
                }
            }
        }
"#;

/// ... which the shipped kernel skips in a simdgroup whose rows all kept
/// their running max.
const SKIPPING_RESCALE: &str = r#"        // Rescale the O accumulator: O = diag(corr) * O. A simdgroup whose
        // rows all kept their running max (corr == 1 on every row) skips it:
        // diag(1) * O is O itself, bit for bit, since O never holds a
        // negative zero (it starts at +0 and only ever gains P * V sums).
        const uint sm_lane = lid % 32u;
        bool row_rescaled = false;
        if (sm_lane < PFA_SG_M) {
            row_rescaled = corr[sg_m0 + sm_lane] != 1.0f;
        }
        if (simd_any(row_rescaled)) {
            {
                const uint dbase = (sgid * PFA_MFRAGS) * (PFA_FRAG * PFA_FRAG);
                for (uint idx = sm_lane; idx < PFA_MFRAGS * PFA_FRAG * PFA_FRAG; idx += 32u) {
                    const uint mi = idx / (PFA_FRAG * PFA_FRAG);
                    const uint e  = idx % (PFA_FRAG * PFA_FRAG);
                    const uint rr = e / PFA_FRAG;
                    const uint cc = e % PFA_FRAG;
                    float val = 0.0f;
                    if (rr == cc) {
                        val = corr[sg_m0 + mi * PFA_FRAG + rr];
                    }
                    diagsh[dbase + idx] = val;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            {
                const uint dbase = (sgid * PFA_MFRAGS) * (PFA_FRAG * PFA_FRAG);
                for (uint mi = 0u; mi < PFA_MFRAGS; mi++) {
                    simdgroup_float8x8 dfrag;
                    simdgroup_load(dfrag, diagsh + dbase + mi * (PFA_FRAG * PFA_FRAG), PFA_FRAG);
                    for (uint di = 0u; di < d_frags; di++) {
                        simdgroup_float8x8 tmp;
                        simdgroup_multiply(tmp, dfrag, oacc[mi][di]);
                        oacc[mi][di] = tmp;
                    }
                }
            }
        }
"#;

/// The original kernel's V staging: one value per thread and step, behind a
/// barrier of its own ...
const ORIGINAL_V_STAGING: &str = r#"        // Stage V over the (now dead) K tile, then O += P * V.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < PFA_BK * head_dim; i += PFA_THREADS) {
            const uint j = i / head_dim;
            const uint d = i % head_dim;
            float val = 0.0f;
            if (j < k_valid) {
                val = float(v_cache[head_off + (k0 + j) * head_dim + d]);
            }
            KVsh[j * head_dim + d] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"#;

/// ... and the shipped one: four values per thread and step, no extra
/// barrier.
const QUAD_V_STAGING: &str = r#"        // Stage V over the K tile, then O += P * V. Every read of the K
        // tile happened before the barrier that published S, so the tile is
        // already dead here. Four values per thread and step: head_dim is a
        // multiple of 8, so a quad never straddles two keys.
        for (uint i = lid * 4u; i < PFA_BK * head_dim; i += PFA_THREADS * 4u) {
            const uint j = i / head_dim;
            const uint d = i % head_dim;
            float4 val = float4(0.0f);
            if (j < k_valid) {
                val = float4(*((device const half4*)(v_cache + head_off + (k0 + j) * head_dim + d)));
            }
            *((threadgroup float4*)(KVsh + j * head_dim + d)) = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"#;

/// The original kernel's K staging: one value per thread and step ...
const ORIGINAL_K_STAGING: &str = r#"        // Stage K-transposed: KVsh[d*PFA_BK + j] = k_cache[kv_head, k0+j, d].
        for (uint i = lid; i < head_dim * PFA_BK; i += PFA_THREADS) {
            const uint d = i / PFA_BK;
            const uint j = i % PFA_BK;
            float val = 0.0f;
            if (j < k_valid) {
                val = float(k_cache[head_off + (k0 + j) * head_dim + d]);
            }
            KVsh[d * PFA_BK + j] = val;
        }
"#;

/// ... and the shipped one: four head dimensions per thread and step.
const QUAD_K_STAGING: &str = r#"        // Stage K-transposed: KVsh[d*PFA_BK + j] = k_cache[kv_head, k0+j, d],
        // four head dimensions per thread and step; the lanes walk the keys,
        // so each of the four transposed stores is conflict-free.
        for (uint i = lid; i < (head_dim / 4u) * PFA_BK; i += PFA_THREADS) {
            const uint d = (i / PFA_BK) * 4u;
            const uint j = i % PFA_BK;
            float4 val = float4(0.0f);
            if (j < k_valid) {
                val = float4(*((device const half4*)(k_cache + head_off + (k0 + j) * head_dim + d)));
            }
            KVsh[(d + 0u) * PFA_BK + j] = val.x;
            KVsh[(d + 1u) * PFA_BK + j] = val.y;
            KVsh[(d + 2u) * PFA_BK + j] = val.z;
            KVsh[(d + 3u) * PFA_BK + j] = val.w;
        }
"#;

/// `MSL_PREFILL_FLASH_ATTENTION` rebuilt as first written: the scalar K and
/// V staging, the row-serial online softmax and the unconditional O rescale.
fn original_flash_source(src: &str) -> String {
    let start = src
        .find("        // Causal online softmax.")
        .expect("the flash kernel's softmax block");
    let end = src
        .find("        // Rescale the O accumulator")
        .expect("the flash kernel's rescale block");
    let rebuilt = format!("{}{ORIGINAL_SOFTMAX}\n{}", &src[..start], &src[end..]);
    for block in [QUAD_K_STAGING, SKIPPING_RESCALE, QUAD_V_STAGING] {
        assert!(
            rebuilt.contains(block),
            "the flash kernel changed; update this test's copy of it"
        );
    }
    rebuilt
        .replace(QUAD_K_STAGING, ORIGINAL_K_STAGING)
        .replace(SKIPPING_RESCALE, ORIGINAL_RESCALE)
        .replace(QUAD_V_STAGING, ORIGINAL_V_STAGING)
}

/// Encode one `prefill_flash_attention`-shaped dispatch of `pso` (layer 0).
#[allow(clippy::too_many_arguments)]
fn dispatch_flash(
    enc: &metal::ComputeCommandEncoderRef,
    pso: &metal::ComputePipelineState,
    q: &Buffer,
    k: &Buffer,
    v: &Buffer,
    out: &Buffer,
    dims: &[u32; 9],
    scale: f32,
) {
    use crate::gpu_backend::kernel_sources::{PREFILL_FLASH_BQ, PREFILL_FLASH_THREADS};
    let [nq, _, _, _, _, _, _, _, batch] = *dims;
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(q), 0);
    enc.set_buffer(1, Some(k), 0);
    enc.set_buffer(2, Some(v), 0);
    enc.set_buffer(3, Some(out), 0);
    // SAFETY: scalar bindings at the kernel's declared indices (4..=14).
    unsafe {
        for (index, value) in (4u64..).zip(dims.iter()) {
            super::buffers::set_scalar(enc, index, value);
        }
        super::buffers::set_scalar(enc, 13, &scale);
        super::buffers::set_scalar(enc, 14, &0u32);
    }
    enc.dispatch_thread_groups(
        metal::MTLSize::new(
            (batch as usize).div_ceil(PREFILL_FLASH_BQ) as u64,
            u64::from(nq),
            1,
        ),
        metal::MTLSize::new(PREFILL_FLASH_THREADS as u64, 1, 1),
    );
}

/// Compile `src`'s flash kernel under `name` with the runtime compiler.
fn flash_pipeline(graph: &MetalGraph, src: &str, name: &str) -> metal::ComputePipelineState {
    let renamed = src.replace(
        "kernel void prefill_flash_attention(",
        &format!("kernel void {name}("),
    );
    let lib = graph
        .device
        .new_library_with_source(&renamed, &metal::CompileOptions::new())
        .unwrap_or_else(|e| panic!("{name} compiles: {e}"));
    let func = lib
        .get_function(name, None)
        .unwrap_or_else(|e| panic!("{name} entry point: {e}"));
    graph
        .device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap_or_else(|e| panic!("{name} pipeline: {e}"))
}

/// `prefill_flash_attention` — its online softmax split over four lanes per
/// query row, its O rescale skipped where no row's running max moved —
/// computes exactly the bits of the kernel as first written (one thread per
/// row, a rescale on every key tile), both compiled from source by the same
/// compiler: on partial query and key tiles, a single row, `pos_start > 0`,
/// MHA and GQA, and rows whose key tile lies entirely in their future. The
/// shipped pipeline must agree too. With `OXIBONSAI_PREFILL_TIMELINE=1` it
/// also times both at the real 1.7B / 8B head geometries over a 4096-token
/// prompt (M-18).
#[test]
fn metal_prefill_flash_attention_is_bit_identical_to_its_original_form() {
    let Some((graph, _scope)) = private_graph() else {
        return;
    };
    let src = crate::gpu_backend::kernel_sources::MSL_PREFILL_FLASH_ATTENTION;
    let current = flash_pipeline(&graph, src, "prefill_flash_attention_current");
    let original = flash_pipeline(
        &graph,
        &original_flash_source(src),
        "prefill_flash_attention_original",
    );
    let shipped = graph
        .pipeline_for("prefill_flash_attention")
        .expect("shipped pipeline");
    let timeline = std::env::var("OXIBONSAI_PREFILL_TIMELINE").as_deref() == Ok("1");
    // (nq, nkv, head_dim, batch, pos_start, timed)
    let mut cases = vec![
        (4u32, 2u32, 64u32, 1u32, 0u32, false),
        (4, 4, 64, 37, 0, false),
        (8, 2, 128, 100, 13, false),
        (16, 8, 128, 300, 0, false),
        (32, 8, 128, 200, 1000, false),
    ];
    if timeline {
        cases.push((16, 8, 128, 4096, 0, true));
        cases.push((32, 8, 128, 4096, 0, true));
    }
    let mut rng = Rng(0xF1A5_50F7);
    for (nq, nkv, hd, batch, pos_start, timed) in cases {
        let max_seq = pos_start + batch;
        let q_stride = (nq + 2 * nkv) * hd;
        let out_stride = nq * hd;
        let q = shared_f32(&graph, (batch * q_stride) as usize, &mut rng);
        let kv_len = (nkv * max_seq * hd) as usize;
        let halves = |rng: &mut Rng| -> Vec<u8> {
            (0..kv_len)
                .flat_map(|_| half::f16::from_f32(rng.unit()).to_le_bytes())
                .collect()
        };
        let k = shared_bytes(&graph, &halves(&mut rng));
        let v = shared_bytes(&graph, &halves(&mut rng));
        let dims = [
            nq,
            nkv,
            nq / nkv,
            hd,
            q_stride,
            out_stride,
            max_seq,
            pos_start,
            batch,
        ];
        let scale = 1.0 / (hd as f32).sqrt();
        let reps = if timed { 3 } else { 1 };
        let mut outs = Vec::new();
        let mut times = Vec::new();
        for pso in [&current, &original, &shipped] {
            let out = shared_f32(&graph, (batch * out_stride) as usize, &mut Rng(9));
            let ms = time_gpu(&graph, reps, |e| {
                dispatch_flash(e, pso, &q, &k, &v, &out, &dims, scale)
            });
            let mut host = vec![0f32; (batch * out_stride) as usize];
            // SAFETY: shared buffer of `batch * out_stride` floats.
            unsafe { super::buffers::download_f32(&out, &mut host) };
            outs.push(host);
            times.push(ms);
        }
        let what = format!("nq={nq} nkv={nkv} head_dim={hd} batch={batch} pos_start={pos_start}");
        assert!(outs[0].iter().all(|x| x.is_finite()), "{what}: finite");
        assert!(outs[0].iter().any(|x| *x != 0.0), "{what}: written");
        assert_eq!(
            bits(&outs[0]),
            bits(&outs[1]),
            "{what}: the reworked kernel changed the attention bits"
        );
        assert_eq!(
            bits(&outs[2]),
            bits(&outs[0]),
            "{what}: the shipped pipeline differs from the runtime-compiled source"
        );
        if timed {
            eprintln!(
                "prefill_flash_attention {what}: current {:.3} ms, original {:.3} ms ({:.2}x)",
                times[0],
                times[1],
                times[1] / times[0].max(f64::MIN_POSITIVE)
            );
        }
    }
}
