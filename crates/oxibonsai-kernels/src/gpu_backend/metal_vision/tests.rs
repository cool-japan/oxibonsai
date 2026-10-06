//! The vision-tower kernels against this crate's CPU primitives, the flash
//! attention at the tower's `head_dim` of 72, a whole tiny tower end to end,
//! and the micro-benchmarks behind the weight-format decision.

use std::sync::Arc;

use metal::objc::rc::autoreleasepool;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, ComputeCommandEncoderRef, MTLResourceOptions, MTLSize};

use super::*;
use crate::gemm_f32::gemm_f32;
use crate::gpu_backend::metal_graph::commit_and_wait;
use crate::rope_mrope::{mrope_vision_build_tables, rope_partial_splithalf_simd};

/// A reproducible stream (an LCG), uniform in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as u32 as f32 / 2_147_483_648.0) - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
}

fn session() -> Option<Arc<MetalGraph>> {
    match MetalGraph::new_session() {
        Ok(graph) => Some(graph),
        Err(MetalGraphError::DeviceNotFound) => None,
        Err(e) => panic!("the combined Metal library must build on this device: {e}"),
    }
}

fn buf(graph: &MetalGraph, values: &[f32]) -> Buffer {
    let b = alloc_buf(
        &graph.device,
        (values.len().max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
    .expect("buffer");
    // SAFETY: a fresh shared buffer of at least `values.len()` floats.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), b.contents().cast::<f32>(), values.len())
    };
    b
}

fn half_buf(graph: &MetalGraph, values: &[f32]) -> Buffer {
    let b = alloc_buf(
        &graph.device,
        (values.len().max(2) * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
    .expect("buffer");
    // SAFETY: a fresh shared buffer of at least `values.len()` halves.
    let dst = unsafe { std::slice::from_raw_parts_mut(b.contents().cast::<u16>(), values.len()) };
    for (d, &v) in dst.iter_mut().zip(values) {
        *d = half::f16::from_f32(v).to_bits();
    }
    b
}

fn read(b: &Buffer, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    // SAFETY: the caller sized `b` for `n` floats and waited for the GPU.
    unsafe { std::ptr::copy_nonoverlapping(b.contents().cast::<f32>(), out.as_mut_ptr(), n) };
    out
}

/// Encode with `f`, commit, wait; returns the GPU seconds.
fn run(graph: &MetalGraph, f: impl FnOnce(&ComputeCommandEncoderRef)) -> f64 {
    autoreleasepool(|| {
        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        f(enc);
        enc.end_encoding();
        commit_and_wait(cmd, "vision test").expect("command buffer");
        // SAFETY: completed command buffer: its timestamps are readable.
        let (s, e): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        (e - s).max(0.0)
    })
}

fn set_u32(enc: &ComputeCommandEncoderRef, index: u64, v: u32) {
    enc.set_bytes(index, 4, (&v as *const u32).cast());
}

fn set_f32(enc: &ComputeCommandEncoderRef, index: u64, v: f32) {
    enc.set_bytes(index, 4, (&v as *const f32).cast());
}

/// `n × k` weights as `Q8_0` blocks (`f16` scale, then 32 int8): the raw
/// bytes the device reads, and the `f32` values they dequantise to
/// (`d · q`, exact in `f32`). The scales vary by a factor of about 100
/// across blocks, as a real projector's do.
fn q8_0_weights(rng: &mut Lcg, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(k.is_multiple_of(32));
    let mut raw = Vec::with_capacity(n * k / 32 * 34);
    let mut values = Vec::with_capacity(n * k);
    for _ in 0..n * k / 32 {
        let d = half::f16::from_f32(1e-4 + 1e-2 * rng.unit().abs());
        raw.extend_from_slice(&d.to_bits().to_le_bytes());
        for _ in 0..32 {
            let q = (rng.unit() * 127.0) as i8;
            raw.push(q as u8);
            values.push(d.to_f32() * f32::from(q));
        }
    }
    (raw, values)
}

fn byte_buf(graph: &MetalGraph, bytes: &[u8]) -> Buffer {
    let b = alloc_buf(
        &graph.device,
        bytes.len().max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
    .expect("buffer");
    // SAFETY: a fresh shared buffer of at least `bytes.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().cast::<u8>(), bytes.len())
    };
    b
}

/// `max |a − b| / max |b|`.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

/// The CPU tower's tanh GELU (`oxibonsai-model::vision::tower::gelu_tanh`),
/// transliterated: `0.5 x (1 + tanh(√(2/π) x (1 + 0.044715 x²)))`.
fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044_715 * x * x)).tanh())
}

/// Dispatch one of the tower's GEMMs.
#[allow(clippy::too_many_arguments)]
fn gemm(
    graph: &MetalGraph,
    kernel: &str,
    w: &Buffer,
    a: &Buffer,
    c: &Buffer,
    bias: &Buffer,
    n: usize,
    k: usize,
    m: usize,
    flags: u32,
) -> f64 {
    let pso = graph.pipeline_for(kernel).expect("pipeline");
    run(graph, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(a), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_buffer(3, Some(bias), 0);
        set_u32(enc, 4, n as u32);
        set_u32(enc, 5, k as u32);
        set_u32(enc, 6, m as u32);
        set_u32(enc, 7, flags);
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(64) as u64, m.div_ceil(64) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    })
}

/// LayerNorm of every row against an `f64` evaluation of the same formula
/// (the CPU tower's `layer_norm_rows` accumulates in `f64`): worst relative
/// error at most 1e-6, at the tower's width.
#[test]
fn layer_norm_matches_an_f64_evaluation() {
    let Some(graph) = session() else {
        return;
    };
    let (rows, dim, eps) = (64usize, 1152usize, 1e-6f32);
    let mut rng = Lcg(3);
    let x: Vec<f32> = (0..rows * dim)
        .map(|i| rng.unit() * 3.0 + (i % 7) as f32)
        .collect();
    let w = rng.vec(dim, 1.0);
    let b = rng.vec(dim, 0.5);
    let (xb, wb, bb) = (buf(&graph, &x), buf(&graph, &w), buf(&graph, &b));
    let ob = buf(&graph, &vec![0.0; rows * dim]);
    let pso = graph.pipeline_for("vit_layer_norm").expect("pipeline");
    run(&graph, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&xb), 0);
        enc.set_buffer(1, Some(&wb), 0);
        enc.set_buffer(2, Some(&bb), 0);
        enc.set_buffer(3, Some(&ob), 0);
        set_u32(enc, 4, dim as u32);
        set_f32(enc, 5, eps);
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    });
    let got = read(&ob, rows * dim);
    let mut want = vec![0.0f32; rows * dim];
    for (o, xi) in want.chunks_mut(dim).zip(x.chunks(dim)) {
        let n = dim as f64;
        let mean = xi.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = xi
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for (((d, &v), &wi), &bi) in o.iter_mut().zip(xi).zip(&w).zip(&b) {
            *d = ((f64::from(v) - mean) * inv * f64::from(wi) + f64::from(bi)) as f32;
        }
    }
    let err = rel(&got, &want);
    assert!(err <= 1e-6, "layer norm worst relative error {err:e}");
}

/// The rotation and head-major pack against the CPU rotation
/// (`rope_partial_splithalf_simd` with `n_rot = head_dim`) on angle rows
/// from `mrope_vision_build_tables`: bitwise (each product rounded on its
/// own, as the NEON path computes it), and `v` copied verbatim.
#[test]
fn qkv_rope_pack_is_bitwise_the_cpu_rotation() {
    let Some(graph) = session() else {
        return;
    };
    let (n, heads, hd) = (24usize, 4usize, 72usize);
    let hidden = heads * hd;
    let half = hd / 2;
    let mut rng = Lcg(5);
    let qkv = rng.vec(n * 3 * hidden, 2.0);
    let mut cos = vec![0.0f32; n * half];
    let mut sin = vec![0.0f32; n * half];
    for t in 0..n {
        let (y, x) = ((t / 6) as i32, (t % 6) as i32);
        mrope_vision_build_tables(
            [y, x, y, x],
            [18; 4],
            hd,
            1e4,
            &mut cos[t * half..(t + 1) * half],
            &mut sin[t * half..(t + 1) * half],
        )
        .expect("angles");
    }
    let (qkvb, cb, sb) = (buf(&graph, &qkv), buf(&graph, &cos), buf(&graph, &sin));
    let zeros = vec![0.0f32; n * hidden];
    let (qb, kb, vb) = (
        buf(&graph, &zeros),
        buf(&graph, &zeros),
        buf(&graph, &zeros),
    );
    let pso = graph.pipeline_for("vit_qkv_rope_pack").expect("pipeline");
    run(&graph, |enc| {
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in [&qkvb, &cb, &sb, &qb, &kb, &vb].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        set_u32(enc, 6, n as u32);
        set_u32(enc, 7, heads as u32);
        set_u32(enc, 8, hd as u32);
        enc.dispatch_thread_groups(
            MTLSize::new(half.div_ceil(32) as u64, heads as u64, n as u64),
            MTLSize::new(32, 1, 1),
        );
    });
    let (q, k, v) = (
        read(&qb, n * hidden),
        read(&kb, n * hidden),
        read(&vb, n * hidden),
    );
    let mut expect = vec![0.0f32; hd];
    for t in 0..n {
        let (c, s) = (
            &cos[t * half..(t + 1) * half],
            &sin[t * half..(t + 1) * half],
        );
        for h in 0..heads {
            let dst = (h * n + t) * hd;
            for (part, got) in [(0usize, &q), (hidden, &k)] {
                let src = &qkv[t * 3 * hidden + part + h * hd..][..hd];
                rope_partial_splithalf_simd(src, &mut expect, hd, hd, c, s).expect("cpu rope");
                for (j, (a, b)) in got[dst..dst + hd].iter().zip(&expect).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "t {t} h {h} part {part} j {j}");
                }
            }
            let src = &qkv[t * 3 * hidden + 2 * hidden + h * hd..][..hd];
            assert_eq!(&v[dst..dst + hd], src, "v t {t} h {h}");
        }
    }
}

/// Weights rounded to `f16` first, so the device's `f16` copy is exact: the
/// GEMM against the CPU `f32` GEMM (plus bias, the residual add) to a
/// relative 1e-5, with a `K` (4304) that is not a multiple of the 32-wide
/// slice and an `N` (4304) that leaves a partial tile.
#[test]
fn f16_weight_gemm_matches_the_cpu_gemm_with_clamps() {
    let Some(graph) = session() else {
        return;
    };
    let mut rng = Lcg(7);
    for (m, n, k) in [
        (100usize, 1152usize, 4304usize),
        (37, 4304, 1152),
        (5, 96, 64),
    ] {
        let w: Vec<f32> = rng
            .vec(n * k, 0.05)
            .into_iter()
            .map(|v| half::f16::from_f32(v).to_f32())
            .collect();
        let a = rng.vec(m * k, 1.0);
        let bias = rng.vec(n, 0.1);
        let resid = rng.vec(m * n, 1.0);
        let (wb, ab, bb) = (half_buf(&graph, &w), buf(&graph, &a), buf(&graph, &bias));
        let mut want = vec![0.0f32; m * n];
        gemm_f32(&a, &w, Some(&bias), m, k, n, &mut want).expect("cpu gemm");
        let cb = buf(&graph, &vec![0.0; m * n]);
        gemm(
            &graph,
            "vit_gemm_f16w",
            &wb,
            &ab,
            &cb,
            &bb,
            n,
            k,
            m,
            FLAG_BIAS,
        );
        let err = rel(&read(&cb, m * n), &want);
        assert!(err <= 1e-5, "m {m} n {n} k {k}: relative error {err:e}");
        // Residual add: out = resid + (a w^T + b).
        let cb = buf(&graph, &resid);
        gemm(
            &graph,
            "vit_gemm_f16w",
            &wb,
            &ab,
            &cb,
            &bb,
            n,
            k,
            m,
            FLAG_BIAS | FLAG_ACCUMULATE,
        );
        let summed: Vec<f32> = resid.iter().zip(&want).map(|(r, v)| r + v).collect();
        let err = rel(&read(&cb, m * n), &summed);
        assert!(
            err <= 1e-5,
            "accumulate m {m} n {n} k {k}: relative error {err:e}"
        );
        // The f32-weight twin (the patch kernel's) on the same values.
        let wf = buf(&graph, &w);
        let cb = buf(&graph, &vec![0.0; m * n]);
        gemm(
            &graph,
            "vit_gemm_f32w",
            &wf,
            &ab,
            &cb,
            &bb,
            n,
            k,
            m,
            FLAG_BIAS,
        );
        let err = rel(&read(&cb, m * n), &want);
        assert!(
            err <= 1e-5,
            "f32 weights m {m} n {n} k {k}: relative error {err:e}"
        );
    }
}

/// The exact `Q8_0` GEMM against the CPU `f32` GEMM on the dequantised
/// weights (`d · q`): relative 1e-5 — the weights are not rounded to
/// `half` — with partial tiles in `M` and `N`, every epilogue (bias, GELU,
/// the residual add), and the tower's `K` of 1152 and 4608.
#[test]
fn q8_0_gemm_is_the_dequantised_f32_gemm() {
    let Some(graph) = session() else {
        return;
    };
    let mut rng = Lcg(11);
    for (m, n, k) in [
        (100usize, 1152usize, 1152usize),
        (37, 200, 4608),
        (5, 96, 64),
    ] {
        let (raw, w) = q8_0_weights(&mut rng, n, k);
        let a = rng.vec(m * k, 1.0);
        let bias = rng.vec(n, 0.1);
        let resid = rng.vec(m * n, 1.0);
        let (wb, ab, bb) = (byte_buf(&graph, &raw), buf(&graph, &a), buf(&graph, &bias));
        let mut want = vec![0.0f32; m * n];
        gemm_f32(&a, &w, Some(&bias), m, k, n, &mut want).expect("cpu gemm");
        let cb = buf(&graph, &vec![0.0; m * n]);
        gemm(
            &graph,
            "vit_gemm_q80",
            &wb,
            &ab,
            &cb,
            &bb,
            n,
            k,
            m,
            FLAG_BIAS,
        );
        let err = rel(&read(&cb, m * n), &want);
        assert!(err <= 1e-5, "m {m} n {n} k {k}: relative error {err:e}");
        // The residual add.
        let cb = buf(&graph, &resid);
        gemm(
            &graph,
            "vit_gemm_q80",
            &wb,
            &ab,
            &cb,
            &bb,
            n,
            k,
            m,
            FLAG_BIAS | FLAG_ACCUMULATE,
        );
        let summed: Vec<f32> = resid.iter().zip(&want).map(|(r, v)| r + v).collect();
        let err = rel(&read(&cb, m * n), &summed);
        assert!(
            err <= 1e-5,
            "accumulate m {m} n {n} k {k}: relative error {err:e}"
        );
        // The GELU epilogue.
        let cb = buf(&graph, &vec![0.0; m * n]);
        gemm(
            &graph,
            "vit_gemm_q80",
            &wb,
            &ab,
            &cb,
            &bb,
            n,
            k,
            m,
            FLAG_BIAS | FLAG_GELU,
        );
        let gelu: Vec<f32> = want.iter().map(|&v| gelu_tanh(v)).collect();
        let err = rel(&read(&cb, m * n), &gelu);
        assert!(
            err <= 1e-5,
            "gelu m {m} n {n} k {k}: relative error {err:e}"
        );
        // The same weights rounded to f16 are measurably worse: the exact
        // path is not the f16 one.
        let wh = half_buf(&graph, &w);
        let cb = buf(&graph, &vec![0.0; m * n]);
        gemm(
            &graph,
            "vit_gemm_f16w",
            &wh,
            &ab,
            &cb,
            &bb,
            n,
            k,
            m,
            FLAG_BIAS,
        );
        let rounded = rel(&read(&cb, m * n), &want);
        eprintln!("q8_0 m {m} n {n} k {k}: exact {err:.2e}, f16-rounded weights {rounded:.2e}");
    }
}

/// The GELU epilogue (`precise::tanh` under fast math) against the CPU
/// tower's formula over `[-20, 20]`, through an identity GEMM (exact: one
/// weight of 1 per row) so the epilogue alone is measured: relative error
/// at most 1e-6 of `1 + |gelu(x)|`.
#[test]
fn gelu_epilogue_matches_the_cpu_tanh_gelu() {
    let Some(graph) = session() else {
        return;
    };
    let n = 256usize;
    let xs: Vec<f32> = (0..16 * n)
        .map(|i| -20.0 + 40.0 * i as f32 / (16 * n - 1) as f32)
        .collect();
    let m = xs.len() / n;
    let mut eye = vec![0.0f32; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    let (wb, ab, zb) = (
        half_buf(&graph, &eye),
        buf(&graph, &xs),
        buf(&graph, &vec![0.0; n]),
    );
    let cb = buf(&graph, &vec![0.0; m * n]);
    gemm(
        &graph,
        "vit_gemm_f16w",
        &wb,
        &ab,
        &cb,
        &zb,
        n,
        n,
        m,
        FLAG_GELU,
    );
    let got = read(&cb, m * n);
    let mut worst = 0.0f32;
    for (&g, &x) in got.iter().zip(&xs) {
        let want = gelu_tanh(x);
        let err = (g - want).abs() / (1.0 + want.abs());
        worst = worst.max(err);
    }
    eprintln!("vit GELU vs the CPU tanh GELU over [-20, 20]: worst relative error {worst:.3e}");
    assert!(worst <= 1e-6, "GELU worst relative error {worst:e}");
}

/// The flash attention at the tower's head width (72, not the DiT's 128)
/// against a direct `f64` softmax attention, non-causal, at 50 (a partial
/// tile), 192 (the 256 × 192 golden image) and 2304 (768 × 768) patches:
/// worst absolute error at most 5e-6. Also its throughput at 2304.
#[test]
fn flash_attention_at_head_dim_72_matches_the_cpu() {
    let Some(graph) = session() else {
        return;
    };
    let (heads, hd) = (16usize, 72usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut rng = Lcg(11);
    for seq in [50usize, 192, 2304] {
        validate_vit_attention(seq, hd).expect("in bounds");
        let q = rng.vec(heads * seq * hd, 1.0);
        let k = rng.vec(heads * seq * hd, 1.0);
        let v = rng.vec(heads * seq * hd, 1.0);
        let (qb, kb, vb) = (buf(&graph, &q), buf(&graph, &k), buf(&graph, &v));
        let ob = buf(&graph, &vec![0.0; seq * heads * hd]);
        let seconds = run(&graph, |enc| {
            graph.dispatch_joint_attention_flash(
                enc,
                &qb,
                &kb,
                &vb,
                &ob,
                heads as u32,
                seq as u32,
                hd as u32,
                scale,
            );
        });
        let got = read(&ob, seq * heads * hd);
        // Check a spread of query rows of every head (all of them for the
        // small sequences).
        let stride = if seq > 256 { 37 } else { 1 };
        let mut worst = 0.0f64;
        for h in 0..heads {
            let base = h * seq * hd;
            for qi in (0..seq).step_by(stride) {
                let qrow = &q[base + qi * hd..][..hd];
                let scores: Vec<f64> = (0..seq)
                    .map(|ki| {
                        let krow = &k[base + ki * hd..][..hd];
                        qrow.iter()
                            .zip(krow)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = probs.iter().sum();
                for d in 0..hd {
                    let want = probs
                        .iter()
                        .enumerate()
                        .map(|(ki, p)| p * f64::from(v[base + ki * hd + d]))
                        .sum::<f64>()
                        / total;
                    let g = f64::from(got[qi * heads * hd + h * hd + d]);
                    worst = worst.max((g - want).abs());
                }
            }
        }
        let flops = 4.0 * (heads * seq * seq * hd) as f64;
        eprintln!(
            "flash attention d72 seq {seq}: worst |err| {worst:.3e}; GPU {:.3} ms = {:.0} GFLOP/s",
            seconds * 1e3,
            flops / seconds / 1e9
        );
        assert!(worst <= 5e-6, "seq {seq}: worst absolute error {worst:e}");
    }
    assert!(validate_vit_attention(VIT_ATTN_MAX_SEQ + 1, hd).is_err());
    assert!(validate_vit_attention(64, 76).is_err());
    assert!(validate_vit_attention(0, hd).is_err());
}

/// The tiny tower's geometry, with `formats`.
fn tiny_config(formats: VisionMatrixFormats) -> VisionGpuConfig {
    VisionGpuConfig {
        hidden: 64,
        heads: 4,
        head_dim: 16,
        ffn: 96,
        blocks: 2,
        eps: 1e-6,
        patch_len: 768,
        merger_hidden: 256,
        projection_dim: 80,
        max_patches: 64,
        formats,
    }
}

/// Build a tiny tower in `formats` from one weight stream: a `Q8_0` matrix
/// gets random blocks, and every other tensor the values those blocks (or
/// the stream) give — so towers built from the same seed in different
/// formats hold the same weights.
fn tiny_tower(formats: VisionMatrixFormats, seed: u64) -> (VisionGpuConfig, VisionGpuModel) {
    let cfg = tiny_config(formats);
    let mut builder = VisionGpuBuilder::new(cfg.clone()).expect("builder");
    let mut rng = Lcg(seed);
    for tensor in VisionTensor::all(cfg.blocks) {
        let (len, format) = tensor.shape(&cfg);
        let inner = match tensor {
            VisionTensor::QkvWeight(_) | VisionTensor::OutWeight(_) | VisionTensor::UpWeight(_) => {
                Some(cfg.hidden)
            }
            VisionTensor::DownWeight(_) => Some(cfg.ffn),
            VisionTensor::Mm0Weight => Some(cfg.merged_width()),
            VisionTensor::Mm2Weight => Some(cfg.merger_hidden),
            _ => None,
        };
        match inner {
            Some(k) => {
                let (raw, values) = q8_0_weights(&mut rng, len / k, k);
                if format == VisionWeightFormat::Q8_0 {
                    builder.set_q8_0(tensor, &raw).expect("set blocks");
                } else {
                    builder.set(tensor, &values).expect("set values");
                }
            }
            None => builder.set(tensor, &rng.vec(len, 0.05)).expect("set"),
        }
    }
    (cfg, builder.build().expect("model"))
}

/// Every matrix kind held as `Q8_0` blocks except `ffn_down` (`F16`, as in
/// the Bonsai 2 file).
const BONSAI_FORMATS: VisionMatrixFormats = VisionMatrixFormats {
    down: VisionWeightFormat::F16,
    ..VisionMatrixFormats::uniform(VisionWeightFormat::Q8_0)
};

/// A tiny tower end to end: every tensor set, one encode per geometry, the
/// merged rows finite, the footprint what the model allocated, and every
/// malformed call refused before any GPU work.
#[test]
fn a_tiny_tower_encodes_and_refuses_malformed_calls() {
    if session().is_none() {
        return;
    }
    let (cfg, mut model) = tiny_tower(BONSAI_FORMATS, 13);
    let mut rng = Lcg(14);
    let mut builder = VisionGpuBuilder::new(cfg.clone()).expect("builder");
    assert!(builder
        .set(
            VisionTensor::Ln1Weight(cfg.blocks),
            &rng.vec(cfg.hidden, 1.0)
        )
        .is_err());
    assert!(builder.set(VisionTensor::PatchBias, &[0.0; 3]).is_err());
    // A Q8_0 matrix takes blocks, nothing else does, and the byte count is
    // the matrix's.
    let qkv_len = 3 * cfg.hidden * cfg.hidden;
    assert!(builder
        .set(VisionTensor::QkvWeight(0), &rng.vec(qkv_len, 0.05))
        .is_err());
    assert!(builder
        .set_q8_0(VisionTensor::DownWeight(0), &[0u8; 34])
        .is_err());
    assert!(builder
        .set_q8_0(
            VisionTensor::QkvWeight(0),
            &vec![0u8; qkv_len / 32 * 34 - 34]
        )
        .is_err());
    assert!(builder
        .set_q8_0(VisionTensor::QkvWeight(0), &vec![0u8; qkv_len / 32 * 34])
        .is_ok());
    let fp = VisionGpuModel::footprint(&cfg);
    assert_eq!(model.weight_bytes(), fp.weight_bytes);
    assert_eq!(model.resident_bytes(), fp.total_bytes());
    let all_f32 = VisionGpuModel::footprint(&tiny_config(VisionMatrixFormats::uniform(
        VisionWeightFormat::F32,
    )));
    assert!(fp.weight_bytes < all_f32.weight_bytes);
    let n = 16usize;
    let half = cfg.head_dim / 2;
    let patches = rng.vec(n * cfg.patch_len, 1.0);
    let pos = rng.vec(n * cfg.hidden, 0.1);
    let (cos, sin) = (vec![1.0f32; n * half], vec![0.0f32; n * half]);
    let mut out = vec![0.0f32; n / 4 * cfg.projection_dim];
    model
        .encode(&patches, &pos, &cos, &sin, &mut out)
        .expect("encode");
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|&v| v != 0.0));
    assert!(model.last_gpu_seconds() > 0.0);
    let short = vec![0.0f32; 3];
    assert!(model
        .encode(&patches[..cfg.patch_len * 6], &pos, &cos, &sin, &mut out)
        .is_err());
    assert!(model
        .encode(&patches, &short, &cos, &sin, &mut out)
        .is_err());
    assert!(model
        .encode(&patches, &pos, &cos, &sin, &mut out[..4])
        .is_err());
    let unbuilt = VisionGpuBuilder::new(cfg.clone()).expect("builder");
    assert!(unbuilt.build().is_err(), "a tower with unset tensors");
    let wide = VisionGpuConfig {
        head_dim: 20,
        heads: 3,
        hidden: 60,
        ..cfg.clone()
    };
    assert!(
        wide.validate().is_err(),
        "head_dim 20 is not a multiple of 8"
    );
    let many = VisionGpuConfig {
        max_patches: VIT_ATTN_MAX_SEQ + 4,
        ..cfg.clone()
    };
    assert!(many.validate().is_err());
    // Q8_0 blocks need an inner dimension that is a multiple of 32.
    let uneven = VisionGpuConfig {
        ffn: 100,
        formats: VisionMatrixFormats::uniform(VisionWeightFormat::Q8_0),
        ..cfg
    };
    let err = uneven.validate().expect_err("ffn_down's K 100");
    assert!(err.to_string().contains("ffn_down"), "{err}");
}

/// The same weights held as `Q8_0` blocks, as `f16` where exact (`ffn_down`
/// of the Bonsai layout) and as `f32` everywhere encode the same rows: the
/// formats change how the device reads a weight, never its value.
#[test]
fn every_weight_format_encodes_the_same_rows() {
    if session().is_none() {
        return;
    }
    let (cfg, mut native) = tiny_tower(BONSAI_FORMATS, 21);
    let (_, mut wide) = tiny_tower(VisionMatrixFormats::uniform(VisionWeightFormat::F32), 21);
    let mut rng = Lcg(22);
    let n = 32usize;
    let half = cfg.head_dim / 2;
    let patches = rng.vec(n * cfg.patch_len, 1.0);
    let pos = rng.vec(n * cfg.hidden, 0.1);
    let cos: Vec<f32> = (0..n * half).map(|i| (i as f32 * 0.37).cos()).collect();
    let sin: Vec<f32> = (0..n * half).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut a = vec![0.0f32; n / 4 * cfg.projection_dim];
    let mut b = vec![0.0f32; n / 4 * cfg.projection_dim];
    native
        .encode(&patches, &pos, &cos, &sin, &mut a)
        .expect("native");
    wide.encode(&patches, &pos, &cos, &sin, &mut b)
        .expect("f32");
    let err = rel(&a, &b);
    eprintln!("tiny tower, native formats vs f32 weights: relative {err:.2e}");
    assert!(err <= 1e-5, "native formats vs f32 weights: {err:e}");
}

/// One GEMM shape of the tower and its GFLOP count at `m` rows.
struct VitShape {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

/// The tower's GEMMs at a 768 × 768 image (2304 patches, 576 merged rows).
const VIT_SHAPES: [VitShape; 6] = [
    VitShape {
        name: "attn_qkv",
        m: 2304,
        n: 3456,
        k: 1152,
    },
    VitShape {
        name: "attn_out",
        m: 2304,
        n: 1152,
        k: 1152,
    },
    VitShape {
        name: "ffn_up",
        m: 2304,
        n: 4304,
        k: 1152,
    },
    VitShape {
        name: "ffn_down",
        m: 2304,
        n: 1152,
        k: 4304,
    },
    VitShape {
        name: "mm.0",
        m: 576,
        n: 4608,
        k: 4608,
    },
    VitShape {
        name: "mm.2",
        m: 576,
        n: 5120,
        k: 4608,
    },
];

/// The weight-format decision: every tower GEMM shape with the file's
/// `Q8_0` blocks read exactly (`vit_gemm_q80`), with the weights rounded to
/// `f16` (`vit_gemm_f16w`) and with `f32` weights (`vit_gemm_f32w`), best
/// of three GPU times, summed over one 768 × 768 image (27 blocks, the
/// merger once). Asserts the recorded decision holds: holding the blocks
/// exactly costs at most half as much again as rounding them to `f16` (the
/// three run at about the same rate; the blocks are the smallest in
/// memory).
#[test]
fn exact_q8_0_weights_stay_close_to_f16_speed_at_the_vit_shapes() {
    let Some(graph) = session() else {
        return;
    };
    let mut rng = Lcg(17);
    let (mut f16_time, mut q80_time, mut flops_total) = (0.0f64, 0.0f64, 0.0f64);
    let mut f32_time = 0.0f64;
    for s in &VIT_SHAPES {
        let per_image = if s.name.starts_with("mm") { 1.0 } else { 27.0 };
        let a = buf(&graph, &rng.vec(s.m * s.k, 1.0));
        let c = buf(&graph, &vec![0.0; s.m * s.n]);
        let bias = buf(&graph, &rng.vec(s.n, 0.1));
        let w_values = rng.vec(s.n * s.k, 0.05);
        let w16 = half_buf(&graph, &w_values);
        let w32 = buf(&graph, &w_values);
        // `ffn_down`'s K (4304) is no Q8_0 width (the file stores it as
        // F16); its Q8_0 timing uses the nearest multiple of 32.
        let q8_k = s.k / 32 * 32;
        let (q8, _) = q8_0_weights(&mut rng, s.n, q8_k);
        let w8 = byte_buf(&graph, &q8);
        let best = |kernel: &str, w: &Buffer, k: usize| {
            gemm(&graph, kernel, w, &a, &c, &bias, s.n, k, s.m, FLAG_BIAS);
            (0..3)
                .map(|_| gemm(&graph, kernel, w, &a, &c, &bias, s.n, k, s.m, FLAG_BIAS))
                .fold(f64::INFINITY, f64::min)
        };
        let (t16, t8) = (
            best("vit_gemm_f16w", &w16, s.k),
            best("vit_gemm_q80", &w8, q8_k),
        );
        let t32 = best("vit_gemm_f32w", &w32, s.k);
        let flops = 2.0 * (s.m * s.n * s.k) as f64;
        f16_time += per_image * t16;
        q80_time += per_image * t8;
        f32_time += per_image * t32;
        flops_total += per_image * flops;
        eprintln!(
            "vit GEMM {:<9} {}x{}x{}: f16 {:.2} ms {:.2} TFLOP/s | q8_0 {:.2} ms {:.2} TFLOP/s | \
             f32 {:.2} ms {:.2} TFLOP/s",
            s.name,
            s.m,
            s.n,
            s.k,
            t16 * 1e3,
            flops / t16 / 1e12,
            t8 * 1e3,
            flops / t8 / 1e12,
            t32 * 1e3,
            flops / t32 / 1e12
        );
    }
    eprintln!(
        "vit GEMMs per 768x768 image: f16 weights {:.1} ms ({:.2} TFLOP/s), exact q8_0 {:.1} ms \
         ({:.2} TFLOP/s), ratio {:.2}; f32 weights {:.1} ms ({:.2} TFLOP/s)",
        f16_time * 1e3,
        flops_total / f16_time / 1e12,
        q80_time * 1e3,
        flops_total / q80_time / 1e12,
        q80_time / f16_time,
        f32_time * 1e3,
        flops_total / f32_time / 1e12,
    );
    assert!(
        q80_time <= 1.5 * f16_time,
        "the exact Q8_0 path is the chosen format because it keeps near-f16 speed: q8_0 \
         {q80_time:.4}s, f16 {f16_time:.4}s, f32 {f32_time:.4}s"
    );
}

/// The tower's session is one of the process's live Metal sessions while
/// the tower lives, and opening one past the pool ceiling is reported with
/// both numbers (the ceiling is not enforced when a session opens).
#[test]
fn the_tower_session_is_counted_and_a_session_past_the_ceiling_is_reported() {
    assert_eq!(session_ceiling_warning(1, 4), None);
    assert_eq!(session_ceiling_warning(4, 4), None);
    let warning = session_ceiling_warning(5, 4).expect("past the ceiling");
    assert!(warning.contains("5 live Metal sessions"), "{warning}");
    assert!(warning.contains("ceiling of 4"), "{warning}");
    assert!(
        warning.contains("OXIBONSAI_METAL_MAX_SESSIONS"),
        "{warning}"
    );
    assert!(warning.contains("not enforced"), "{warning}");
    if session().is_none() {
        return;
    }
    let (_, model) = tiny_tower(BONSAI_FORMATS, 21);
    // Other tests of this binary open and drop sessions concurrently, so
    // only the tower's own share of the count is certain.
    assert!(
        MetalGraph::live_session_count() >= 1,
        "the tower's session is live while the tower is"
    );
    drop(model);
}

/// The command-buffer sites of the hybrid runner (`metal_full_layer`'s
/// `qwen35*.rs`) and of this tower, scanned for their autorelease pools.
///
/// `-[MTLCommandQueue commandBuffer]` and
/// `-[MTLCommandBuffer computeCommandEncoder]` return autoreleased objects,
/// which a thread without a pool keeps until it exits (~1.8 KiB per
/// dispatch: a server's decode thread would grow with every batched
/// prefill chunk, rows prefill and image encode). Each site here is either
/// inside a function that opens an `autoreleasepool(` before it, or inside
/// an `*_unpooled` function every call of which sits inside an
/// `autoreleasepool(|| …)` closure — the shape every runner and tower entry
/// point uses. The model crate's footprint tests measure the routes;
/// this pins the structure, including sites no route test reaches. The
/// `*_tests.rs` / `tests.rs` files are not scanned: they build throwaway
/// command buffers of their own and dispatch nothing a caller reaches.
#[test]
fn qwen35_and_vision_command_buffers_are_created_inside_autorelease_pools() {
    let sources: [(&str, &str); 7] = [
        (
            "metal_full_layer/qwen35.rs",
            include_str!("../metal_full_layer/qwen35.rs"),
        ),
        (
            "metal_full_layer/qwen35_encode.rs",
            include_str!("../metal_full_layer/qwen35_encode.rs"),
        ),
        (
            "metal_full_layer/qwen35_prefill.rs",
            include_str!("../metal_full_layer/qwen35_prefill.rs"),
        ),
        (
            "metal_full_layer/qwen35_rows.rs",
            include_str!("../metal_full_layer/qwen35_rows.rs"),
        ),
        (
            "metal_full_layer/qwen35_state.rs",
            include_str!("../metal_full_layer/qwen35_state.rs"),
        ),
        ("metal_vision/mod.rs", include_str!("mod.rs")),
        ("metal_vision/encode.rs", include_str!("encode.rs")),
    ];

    // Every non-test `qwen35*.rs` and tower source on disk is scanned: a
    // dispatch file added later must join the list.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu_backend");
    let mut on_disk = Vec::new();
    for (dir, keep) in [("metal_full_layer", "qwen35"), ("metal_vision", "")] {
        let entries = std::fs::read_dir(root.join(dir)).expect("the source directory lists");
        for entry in entries {
            let name = entry.expect("a directory entry").file_name();
            let name = name.to_string_lossy().to_string();
            let is_test = name == "tests.rs" || name.ends_with("_tests.rs");
            if name.starts_with(keep) && name.ends_with(".rs") && !is_test {
                on_disk.push(format!("{dir}/{name}"));
            }
        }
    }
    on_disk.sort();
    let mut scanned: Vec<String> = sources.iter().map(|(f, _)| (*f).to_string()).collect();
    scanned.sort();
    assert_eq!(scanned, on_disk, "scan every non-test dispatch source");

    let fn_name = |line: &str| -> Option<String> {
        let code = line.trim_start();
        let rest = [
            "pub(crate) unsafe fn ",
            "pub(super) unsafe fn ",
            "pub unsafe fn ",
            "unsafe fn ",
            "pub(crate) fn ",
            "pub(super) fn ",
            "pub fn ",
            "fn ",
        ]
        .iter()
        .find_map(|prefix| code.strip_prefix(prefix))?;
        let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
        Some(rest[..end].to_string())
    };
    let all: Vec<(&str, Vec<&str>)> = sources
        .iter()
        .map(|(file, src)| (*file, src.lines().collect()))
        .collect();
    let mut sites = 0usize;
    for (file, lines) in &all {
        for (i, line) in lines.iter().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") || !code.contains(".new_command_buffer()") {
                continue;
            }
            sites += 1;
            let (start, name) = lines[..i]
                .iter()
                .enumerate()
                .rev()
                .find_map(|(at, l)| fn_name(l).map(|n| (at, n)))
                .unwrap_or_else(|| panic!("{file}:{}: no enclosing function", i + 1));
            if lines[start..i]
                .iter()
                .any(|l| l.contains("autoreleasepool("))
            {
                continue;
            }
            assert!(
                name.ends_with("_unpooled"),
                "{file}:{}: command buffer created in `{name}`, which neither opens an \
                 autoreleasepool before it nor is an `_unpooled` body called inside one",
                i + 1
            );
            let call = format!("{name}(");
            let mut calls = 0usize;
            for (caller, caller_lines) in &all {
                for (j, l) in caller_lines.iter().enumerate() {
                    if !l.contains(&call) || fn_name(l).as_deref() == Some(name.as_str()) {
                        continue;
                    }
                    calls += 1;
                    let pooled = caller_lines[j.saturating_sub(2)..=j]
                        .iter()
                        .any(|c| c.contains("autoreleasepool("));
                    assert!(
                        pooled,
                        "{caller}:{}: `{name}` called outside an autoreleasepool closure",
                        j + 1
                    );
                }
            }
            assert!(calls > 0, "`{name}` ({file}) is never called");
        }
    }
    // `>=`: the two in qwen35.rs, the four rows-prefill / dump / trace sites
    // and the tower's encode must be found (a site added later is pinned
    // too).
    assert!(
        sites >= 7,
        "expected at least the 7 known command-buffer sites, found {sites}"
    );
}
