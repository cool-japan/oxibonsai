//! On-device parity tests of the Qwen3.5 hybrid kernels and encoder against
//! this crate's CPU kernels: every weight decoder bitwise (one-hot GEMV), the
//! rotations bitwise, every float kernel within a relative tolerance, the
//! batched (prefill) kernels bitwise against the single-token ones, and the
//! whole encoder — prefill vs sequential decode, reset, zero-copy residency.

use super::*;
use half::f16;

use crate::gated_delta_net::{GdnDims, GdnGates, GdnHeadOrder, GdnPath};
use crate::gated_delta_net_chunk::gdn_prefill_with;
use crate::hadamard::{fwht_forward_signed, fwht_inverse_signed};
use crate::norms::{l2_norm_simd, rms_norm_gated_simd, sigmoid_mul_simd};
use crate::ssm_ops::causal_conv1d_k4_decode;

/// Deterministic xorshift64* stream.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[-1, 1)`.
    fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32() * scale).collect()
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }

    fn signs(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| if self.next_u64() & 1 == 1 { -1.0 } else { 1.0 })
            .collect()
    }

    fn scale(&mut self) -> f16 {
        f16::from_f32(0.004 + (self.next_u64() % 1000) as f32 * 3e-5)
    }
}

/// A fresh session, or `None` on a host without a Metal device. On a host
/// *with* one, a library that fails to build is a test failure, never a
/// silent skip.
fn session() -> Option<Arc<MetalGraph>> {
    metal::Device::system_default()?;
    match MetalGraph::new_session() {
        Ok(graph) => Some(graph),
        Err(e) => panic!("the combined Metal library must build on this device: {e}"),
    }
}

fn run(graph: &MetalGraph, encode: impl FnOnce(&ComputeCommandEncoderRef)) {
    let cmd = graph.command_queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(enc);
    enc.end_encoding();
    commit_and_wait(cmd, "qwen35 test").expect("command buffer completes");
}

fn buf(graph: &MetalGraph, data: &[f32]) -> Buffer {
    upload_f32(graph, data).expect("upload")
}

fn worst_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() / scale)
        .fold(0.0f32, f32::max)
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}[{i}]: gpu {a} vs cpu {b}");
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Weight formats: random blocks and their CPU dequantisation
// ─────────────────────────────────────────────────────────────────────────

/// Owned random blocks of one format.
enum Blocks {
    Pq2(Vec<BlockPQ2_0>),
    Ptq1(Vec<BlockPTQ1_0>),
    Tq2(Vec<BlockTQ2_0_g128>),
    Q2g64(Vec<BlockQ2_0G64>),
    Q1(Vec<BlockQ1_0G128>),
    F32(Vec<f32>),
}

impl Blocks {
    fn random(kind: usize, rows: usize, cols: usize, rng: &mut Rng) -> Self {
        let nsb = rows * cols / 128;
        match kind {
            0 => Self::Pq2(
                (0..nsb)
                    .map(|_| BlockPQ2_0 {
                        d: rng.scale(),
                        qs: std::array::from_fn(|_| rng.byte()),
                    })
                    .collect(),
            ),
            1 => Self::Ptq1(
                (0..nsb)
                    .map(|_| BlockPTQ1_0 {
                        qs: std::array::from_fn(|_| rng.byte()),
                        qh: std::array::from_fn(|_| rng.byte()),
                        d: rng.scale(),
                    })
                    .collect(),
            ),
            2 => Self::Tq2(
                (0..nsb)
                    .map(|_| BlockTQ2_0_g128 {
                        qs: std::array::from_fn(|_| rng.byte()),
                        d: rng.scale(),
                    })
                    .collect(),
            ),
            3 => Self::Q2g64(
                (0..nsb * 2)
                    .map(|_| BlockQ2_0G64 {
                        d: rng.scale(),
                        qs: std::array::from_fn(|_| rng.byte()),
                    })
                    .collect(),
            ),
            4 => Self::Q1(
                (0..nsb)
                    .map(|_| BlockQ1_0G128 {
                        d: rng.scale(),
                        qs: std::array::from_fn(|_| rng.byte()),
                    })
                    .collect(),
            ),
            _ => Self::F32(rng.vec(rows * cols, 0.05)),
        }
    }

    fn data(&self) -> Qwen35MatrixData<'_> {
        match self {
            Self::Pq2(b) => Qwen35MatrixData::Pq2_0(b),
            Self::Ptq1(b) => Qwen35MatrixData::Ptq1_0(b),
            Self::Tq2(b) => Qwen35MatrixData::Tq2_0G128(b),
            Self::Q2g64(b) => Qwen35MatrixData::Q2_0G64(b),
            Self::Q1(b) => Qwen35MatrixData::Q1_0G128(b),
            Self::F32(v) => Qwen35MatrixData::F32(v),
        }
    }

    /// The CPU dequantisation, row-major `[rows][cols]`.
    fn dequant(&self, rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        match self {
            Self::Pq2(b) => BlockPQ2_0::dequant(b, &mut out).expect("dequant"),
            Self::Ptq1(b) => BlockPTQ1_0::dequant(b, &mut out).expect("dequant"),
            Self::Tq2(b) => BlockTQ2_0_g128::dequant(b, &mut out).expect("dequant"),
            Self::Q2g64(b) => BlockQ2_0G64::dequant(b, &mut out).expect("dequant"),
            Self::Q1(b) => {
                for (i, block) in b.iter().enumerate() {
                    let d = block.d.to_f32();
                    for j in 0..128 {
                        let bit = (block.qs[j / 8] >> (j % 8)) & 1;
                        out[i * 128 + j] = if bit == 1 { d } else { -d };
                    }
                }
            }
            Self::F32(v) => out.copy_from_slice(v),
        }
        out
    }
}

fn device_matrix(graph: &MetalGraph, m: &Qwen35Matrix<'_>) -> DeviceMatrix {
    let mut binder = Binder {
        graph,
        region: None,
        copied_bytes: 0,
        mapped_bytes: 0,
    };
    binder.bind("test", m, m.rows, m.cols).expect("bind")
}

fn gemv(
    graph: &MetalGraph,
    m: &DeviceMatrix,
    x: &[f32],
    t_len: usize,
    init: Option<&[f32]>,
) -> Vec<f32> {
    let xb = buf(graph, x);
    let y_init = init.map_or_else(|| vec![0.0; t_len * m.rows], <[f32]>::to_vec);
    let yb = buf(graph, &y_init);
    run(graph, |enc| {
        encode_gemv(enc, m, (&xb, 0), (&yb, 0), t_len, init.is_some());
    });
    read_buffer(&yb, 0, t_len * m.rows)
}

/// Every weight decoder is exact: a one-hot input returns the dequantised
/// column (value-equal to the CPU decoder; a code of zero may come back as
/// `-0.0`), a random input matches the CPU dot product, a many-token
/// dispatch is bitwise one token at a time, and the accumulate epilogue adds
/// in place.
#[test]
fn gemv_decodes_every_format_exactly_and_matches_the_cpu() {
    let Some(graph) = session() else {
        return;
    };
    let (rows, cols) = (13usize, 384usize);
    for kind in 0..6 {
        let mut rng = Rng::new(100 + kind as u64);
        let blocks = Blocks::random(kind, rows, cols, &mut rng);
        let matrix = Qwen35Matrix {
            rows,
            cols,
            data: blocks.data(),
        };
        let name = matrix.data.name();
        let dm = device_matrix(&graph, &matrix);
        let w = blocks.dequant(rows, cols);

        // Stage-1 (0..80), stage-2 (80..120) and qh (120..128) elements of
        // the first and later super-blocks.
        for col in [
            0usize, 1, 15, 16, 79, 80, 100, 119, 120, 127, 128, 200, 250, 383,
        ] {
            let mut x = vec![0.0f32; cols];
            x[col] = 1.0;
            let y = gemv(&graph, &dm, &x, 1, None);
            for r in 0..rows {
                assert!(
                    y[r] == w[r * cols + col],
                    "{name}: one-hot column {col} row {r}: gpu {} vs cpu {}",
                    y[r],
                    w[r * cols + col]
                );
            }
        }

        let t_len = 13usize;
        let x = rng.vec(t_len * cols, 1.0);
        let batched = gemv(&graph, &dm, &x, t_len, None);
        for t in 0..t_len {
            let single = gemv(&graph, &dm, &x[t * cols..(t + 1) * cols], 1, None);
            assert_bits(&batched[t * rows..(t + 1) * rows], &single, name);
            let want: Vec<f32> = (0..rows)
                .map(|r| {
                    (0..cols)
                        .map(|k| f64::from(w[r * cols + k]) * f64::from(x[t * cols + k]))
                        .sum::<f64>() as f32
                })
                .collect();
            let err = worst_rel(&single, &want);
            assert!(err < 2e-6, "{name}: token {t} worst relative error {err:e}");
        }

        let init = rng.vec(rows, 1.0);
        let acc = gemv(&graph, &dm, &x[..cols], 1, Some(&init));
        let plain = gemv(&graph, &dm, &x[..cols], 1, None);
        for r in 0..rows {
            assert_eq!(
                acc[r].to_bits(),
                (init[r] + plain[r]).to_bits(),
                "{name}: accumulate row {r}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Rotation-fused producers
// ─────────────────────────────────────────────────────────────────────────

fn pso(graph: &MetalGraph, name: &str) -> ComputePipelineState {
    graph
        .pipeline_for(name)
        .expect("pipeline in the combined library")
}

/// RMSNorm matches the CPU, and the fused rotation of the GPU's own normed
/// row is bitwise the CPU's `fwht_forward_signed` of it (27B width and
/// block, two rows).
#[test]
fn rmsnorm_rotate_matches_the_cpu_norm_and_rotates_bitwise() {
    let Some(graph) = session() else {
        return;
    };
    let (n, block, rows) = (5120usize, 1024usize, 2usize);
    let mut rng = Rng::new(7);
    let x = rng.vec(rows * n, 2.0);
    let w = rng.vec(n, 1.5);
    let signs = rng.signs(n);
    let (xb, wb, sb) = (buf(&graph, &x), buf(&graph, &w), buf(&graph, &signs));
    let normed = zeroed(&graph, rows * n * 4).expect("normed");
    let rotated = zeroed(&graph, rows * n * 4).expect("rotated");
    let p = pso(&graph, "q35_rmsnorm_rotate");
    run(&graph, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&xb), 0);
        enc.set_buffer(1, Some(&wb), 0);
        enc.set_buffer(2, Some(&normed), 0);
        enc.set_buffer(3, Some(&rotated), 0);
        enc.set_buffer(4, Some(&sb), 0);
        set_u32(enc, 5, n as u32);
        set_f32(enc, 6, 1e-6);
        set_u32(enc, 7, block as u32);
        set_f32(enc, 8, rotation_scale(block));
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(1024, 1, 1));
    });
    let got_n = read_buffer(&normed, 0, rows * n);
    let got_r = read_buffer(&rotated, 0, rows * n);
    for r in 0..rows {
        let mut want = vec![0.0f32; n];
        crate::rms_norm_simd(&x[r * n..(r + 1) * n], &w, &mut want, 1e-6).expect("cpu norm");
        let err = worst_rel(&got_n[r * n..(r + 1) * n], &want);
        assert!(err < 1e-6, "row {r}: normed worst relative error {err:e}");
        let mut rot = got_n[r * n..(r + 1) * n].to_vec();
        fwht_forward_signed(&mut rot, &signs, block).expect("cpu fwht");
        assert_bits(&got_r[r * n..(r + 1) * n], &rot, "rotated");
    }
}

/// The standalone rotation is bitwise the CPU transform in both directions,
/// and inverse(forward(x)) returns x.
#[test]
fn standalone_rotation_is_bitwise_the_cpu_transform() {
    let Some(graph) = session() else {
        return;
    };
    let p = pso(&graph, "q35_fwht_signed");
    for (width, block, rows) in [(5120usize, 1024usize, 3usize), (384, 128, 2)] {
        let mut rng = Rng::new(width as u64);
        let x = rng.vec(rows * width, 3.0);
        let signs = rng.signs(width);
        let sb = buf(&graph, &signs);
        for inverse in [false, true] {
            let xb = buf(&graph, &x);
            run(&graph, |enc| {
                enc.set_compute_pipeline_state(&p);
                enc.set_buffer(0, Some(&xb), 0);
                enc.set_buffer(1, Some(&sb), 0);
                set_u32(enc, 2, width as u32);
                set_u32(enc, 3, block as u32);
                set_u32(enc, 4, u32::from(inverse));
                set_f32(enc, 5, rotation_scale(block));
                enc.dispatch_thread_groups(
                    MTLSize::new((width / block) as u64, rows as u64, 1),
                    MTLSize::new(block as u64, 1, 1),
                );
            });
            let got = read_buffer(&xb, 0, rows * width);
            let mut want = x.clone();
            for row in want.chunks_mut(width) {
                if inverse {
                    fwht_inverse_signed(row, &signs, block).expect("cpu inverse");
                } else {
                    fwht_forward_signed(row, &signs, block).expect("cpu forward");
                }
            }
            assert_bits(&got, &want, if inverse { "inverse" } else { "forward" });
        }
    }
}

/// Run a span producer once unfolded (`block = 0`) and once folded; the
/// unfolded output must match `cpu` within tolerance and the folded output
/// must be bitwise the CPU rotation of the unfolded one.
#[allow(clippy::too_many_arguments)]
fn check_span_producer(
    graph: &MetalGraph,
    name: &str,
    inputs: &[&Buffer],
    extra: &dyn Fn(&ComputeCommandEncoderRef, u64),
    n: usize,
    rows: usize,
    block: usize,
    cpu: &[f32],
    signs: &[f32],
    tol: f32,
) {
    let p = pso(graph, name);
    let sb = buf(graph, signs);
    let mut outs = Vec::new();
    for folded in [false, true] {
        let span = if folded { block } else { 128 };
        let out = zeroed(graph, rows * n * 4).expect("out");
        run(graph, |enc| {
            enc.set_compute_pipeline_state(&p);
            let mut idx = 0u64;
            for input in inputs {
                enc.set_buffer(idx, Some(input), 0);
                idx += 1;
            }
            enc.set_buffer(idx, Some(&out), 0);
            enc.set_buffer(idx + 1, Some(&sb), 0);
            extra(enc, idx + 2);
            let (b, s) = if folded {
                (block as u32, rotation_scale(block))
            } else {
                (0, 1.0)
            };
            let tail = idx
                + 2
                + if name == "q35_gated_norm_rotate" {
                    5
                } else if name == "q35_sigmoid_gate_rotate" {
                    2
                } else {
                    1
                };
            set_u32(enc, tail, b);
            set_f32(enc, tail + 1, s);
            set_u32(enc, tail + 2, span as u32);
            enc.dispatch_thread_groups(
                MTLSize::new((n / span) as u64, rows as u64, 1),
                MTLSize::new(span as u64, 1, 1),
            );
        });
        outs.push(read_buffer(&out, 0, rows * n));
    }
    let err = worst_rel(&outs[0], cpu);
    assert!(err < tol, "{name}: worst relative error {err:e}");
    let mut rot = outs[0].clone();
    for row in rot.chunks_mut(n) {
        fwht_forward_signed(row, signs, block).expect("cpu fwht");
    }
    assert_bits(&outs[1], &rot, name);
}

#[test]
fn swiglu_rotate_matches_the_cpu() {
    let Some(graph) = session() else {
        return;
    };
    let (n, rows, block) = (17408usize, 2usize, 1024usize);
    let mut rng = Rng::new(11);
    let g = rng.vec(rows * n, 4.0);
    let u = rng.vec(rows * n, 2.0);
    let signs = rng.signs(n);
    let mut want = vec![0.0f32; rows * n];
    crate::swiglu_simd(&g, &u, &mut want).expect("cpu swiglu");
    let (gb, ub) = (buf(&graph, &g), buf(&graph, &u));
    check_span_producer(
        &graph,
        "q35_swiglu_rotate",
        &[&gb, &ub],
        &|enc, idx| set_u32(enc, idx, n as u32),
        n,
        rows,
        block,
        &want,
        &signs,
        2e-6,
    );
}

#[test]
fn sigmoid_gate_rotate_matches_the_cpu() {
    let Some(graph) = session() else {
        return;
    };
    let (heads, hd, rows, block) = (24usize, 256usize, 2usize, 1024usize);
    let n = heads * hd;
    let mut rng = Rng::new(12);
    let attn = rng.vec(rows * n, 2.0);
    let q_all = rng.vec(rows * 2 * n, 4.0);
    let signs = rng.signs(n);
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let gate: Vec<f32> = (0..n)
            .map(|i| q_all[r * 2 * n + (i / hd) * 2 * hd + hd + i % hd])
            .collect();
        sigmoid_mul_simd(
            &attn[r * n..(r + 1) * n],
            &gate,
            &mut want[r * n..(r + 1) * n],
        )
        .expect("cpu sigmoid gate");
    }
    let (ab, qb) = (buf(&graph, &attn), buf(&graph, &q_all));
    check_span_producer(
        &graph,
        "q35_sigmoid_gate_rotate",
        &[&ab, &qb],
        &|enc, idx| {
            set_u32(enc, idx, n as u32);
            set_u32(enc, idx + 1, hd as u32);
        },
        n,
        rows,
        block,
        &want,
        &signs,
        2e-6,
    );
}

/// The Gated-DeltaNet output norm, reading `z` through the tiled → grouped
/// v-head map (27B geometry: 16 k-heads, 48 v-heads of 128).
#[test]
fn gated_norm_rotate_matches_the_cpu_per_head_norm() {
    let Some(graph) = session() else {
        return;
    };
    let (nk, nv, hv, rows, block) = (16usize, 48usize, 128usize, 2usize, 1024usize);
    let n = nv * hv;
    let rep = nv / nk;
    let eps = 1e-6f32;
    let mut rng = Rng::new(13);
    let o = rng.vec(rows * n, 2.0);
    let z = rng.vec(rows * n, 4.0);
    let w = rng.vec(hv, 1.5);
    let signs = rng.signs(n);
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        for m in 0..nv {
            let tiled = (m % rep) * nk + m / rep;
            let src = &o[r * n + m * hv..r * n + (m + 1) * hv];
            let gate = &z[r * n + tiled * hv..r * n + (tiled + 1) * hv];
            rms_norm_gated_simd(
                src,
                &w,
                gate,
                &mut want[r * n + m * hv..r * n + (m + 1) * hv],
                eps,
            )
            .expect("cpu gated norm");
        }
    }
    let (ob, zb, wb) = (buf(&graph, &o), buf(&graph, &z), buf(&graph, &w));
    // The gated norm takes its weight before the output: buffers o, z, w.
    check_span_producer(
        &graph,
        "q35_gated_norm_rotate",
        &[&ob, &zb, &wb],
        &|enc, idx| {
            set_u32(enc, idx, n as u32);
            set_u32(enc, idx + 1, hv as u32);
            set_u32(enc, idx + 2, nk as u32);
            set_u32(enc, idx + 3, rep as u32);
            set_f32(enc, idx + 4, eps);
        },
        n,
        rows,
        block,
        &want,
        &signs,
        3e-6,
    );
}

// ─────────────────────────────────────────────────────────────────────────
//  Conv, Gated DeltaNet
// ─────────────────────────────────────────────────────────────────────────

/// The conv + SiLU over five tokens matches five CPU decode steps followed
/// by SiLU, and leaves the same window behind (bitwise: it is data movement).
#[test]
fn conv1d_silu_matches_sequential_cpu_steps() {
    let Some(graph) = session() else {
        return;
    };
    let (channels, t_len) = (10240usize, 5usize);
    let mut rng = Rng::new(21);
    let x = rng.vec(t_len * channels, 2.0);
    let w = rng.vec(channels * 4, 1.0);
    let state0 = rng.vec(channels * 3, 1.0);
    let mut cpu_state = state0.clone();
    let mut want = vec![0.0f32; t_len * channels];
    let mut conv = vec![0.0f32; channels];
    for t in 0..t_len {
        causal_conv1d_k4_decode(
            &mut cpu_state,
            &x[t * channels..(t + 1) * channels],
            &w,
            &mut conv,
        )
        .expect("cpu conv");
        crate::silu_simd(&conv, &mut want[t * channels..(t + 1) * channels]).expect("cpu silu");
    }
    let (xb, wb, sb) = (buf(&graph, &x), buf(&graph, &w), buf(&graph, &state0));
    let out = zeroed(&graph, t_len * channels * 4).expect("out");
    let p = pso(&graph, "q35_conv1d_silu");
    run(&graph, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&xb), 0);
        enc.set_buffer(1, Some(&wb), 0);
        enc.set_buffer(2, Some(&sb), 0);
        enc.set_buffer(3, Some(&out), 0);
        set_u32(enc, 4, channels as u32);
        set_u32(enc, 5, t_len as u32);
        enc.dispatch_thread_groups(
            MTLSize::new(channels.div_ceil(256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });
    let err = worst_rel(&read_buffer(&out, 0, t_len * channels), &want);
    assert!(err < 2e-6, "conv+silu worst relative error {err:e}");
    assert_bits(
        &read_buffer(&sb, 0, channels * 3),
        &cpu_state,
        "conv window",
    );
}

/// The fused L2-norm + gated delta rule matches the CPU recurrence
/// (`gdn_prefill_with`, grouped order, fused path) over several tokens, both
/// in its outputs and in the state it leaves behind.
#[test]
fn gdn_matches_the_cpu_recurrence() {
    let Some(graph) = session() else {
        return;
    };
    for (nk, nv, hk, hv, t_len) in [
        (16usize, 48usize, 128usize, 128usize, 3usize),
        (2, 6, 64, 64, 5),
    ] {
        let rep = nv / nk;
        let conv_dim = 2 * nk * hk + nv * hv;
        let eps = 1e-6f32;
        let mut rng = Rng::new((nk * 1000 + hk) as u64);
        let conv_out = rng.vec(t_len * conv_dim, 1.0);
        let ab = rng.vec(t_len * 2 * nv, 3.0);
        let a_neg: Vec<f32> = (0..nv).map(|_| -(rng.f32().abs() + 0.05)).collect();
        let dt_bias = rng.vec(nv, 1.0);
        let state0 = rng.vec(nv * hv * hk, 0.3);

        // CPU: L2-normalise q/k per k-head, regroup v and the raw gates.
        let tiled = |m: usize| (m % rep) * nk + m / rep;
        let mut q = vec![0.0f32; t_len * nk * hk];
        let mut k = vec![0.0f32; t_len * nk * hk];
        let mut v = vec![0.0f32; t_len * nv * hv];
        let mut alpha = vec![0.0f32; t_len * nv];
        let mut beta = vec![0.0f32; t_len * nv];
        for t in 0..t_len {
            let row = &conv_out[t * conv_dim..(t + 1) * conv_dim];
            for h in 0..nk {
                l2_norm_simd(
                    &row[h * hk..(h + 1) * hk],
                    &mut q[(t * nk + h) * hk..(t * nk + h + 1) * hk],
                    eps,
                )
                .expect("l2 q");
                let ko = nk * hk + h * hk;
                l2_norm_simd(
                    &row[ko..ko + hk],
                    &mut k[(t * nk + h) * hk..(t * nk + h + 1) * hk],
                    eps,
                )
                .expect("l2 k");
            }
            for m in 0..nv {
                let src = 2 * nk * hk + tiled(m) * hv;
                v[(t * nv + m) * hv..(t * nv + m + 1) * hv].copy_from_slice(&row[src..src + hv]);
                alpha[t * nv + m] = ab[t * 2 * nv + tiled(m)];
                beta[t * nv + m] = ab[t * 2 * nv + nv + tiled(m)];
            }
        }
        let dims = GdnDims::new(nk, nv, hk, hv);
        let gates = GdnGates::bonsai2(&alpha, &beta, &dt_bias, &a_neg);
        let mut cpu_state = state0.clone();
        let mut want = vec![0.0f32; t_len * nv * hv];
        gdn_prefill_with(
            &mut cpu_state,
            &q,
            &k,
            &v,
            &gates,
            &mut want,
            t_len,
            &dims,
            GdnHeadOrder::Grouped,
            GdnPath::Fused,
        )
        .expect("cpu gdn");

        let bufs: Vec<Buffer> = [&conv_out, &ab, &a_neg, &dt_bias]
            .iter()
            .map(|v| buf(&graph, v))
            .collect();
        let sb = buf(&graph, &state0);
        let out = zeroed(&graph, t_len * nv * hv * 4).expect("out");
        let p = pso(&graph, "q35_gdn");
        run(&graph, |enc| {
            enc.set_compute_pipeline_state(&p);
            for (i, b) in bufs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.set_buffer(4, Some(&sb), 0);
            enc.set_buffer(5, Some(&out), 0);
            set_u32(enc, 6, nk as u32);
            set_u32(enc, 7, nv as u32);
            set_u32(enc, 8, hk as u32);
            set_u32(enc, 9, hv as u32);
            set_u32(enc, 10, conv_dim as u32);
            set_u32(enc, 11, t_len as u32);
            set_f32(enc, 12, eps);
            set_f32(enc, 13, dims.out_scale());
            enc.dispatch_thread_groups(
                MTLSize::new(nv as u64, 1, 1),
                MTLSize::new(GDN_THREADS, 1, 1),
            );
        });
        let err_out = worst_rel(&read_buffer(&out, 0, t_len * nv * hv), &want);
        let err_state = worst_rel(&read_buffer(&sb, 0, nv * hv * hk), &cpu_state);
        assert!(
            err_out < 1e-5,
            "gdn ({nk},{nv},{hk}) output worst relative error {err_out:e}"
        );
        assert!(
            err_state < 1e-5,
            "gdn ({nk},{nv},{hk}) state worst relative error {err_state:e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  The whole encoder on a small random model
// ─────────────────────────────────────────────────────────────────────────

/// A tiny random `qwen35` model: layer 0 linear, layer 1 full, the fixture
/// geometry of the model crate's synthetic GGUF (hidden 256, Hadamard block
/// 128, head_dim 64 with `n_rot` 16, 2 k-heads / 6 v-heads of 64).
struct TinyModel {
    cfg: Qwen35GpuConfig,
    mats: Vec<Vec<BlockPQ2_0>>,
    norms: Vec<Vec<f32>>,
    ab: Vec<f32>,
    conv: Vec<f32>,
    a_neg: Vec<f32>,
    dt_bias: Vec<f32>,
    signs: Vec<(usize, Vec<f32>)>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl TinyModel {
    fn new(seed: u64) -> Self {
        let cfg = Qwen35GpuConfig {
            hidden: 256,
            intermediate: 512,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 64,
            n_rot: 16,
            n_k_heads: 2,
            n_v_heads: 6,
            head_k_dim: 64,
            head_v_dim: 64,
            conv_kernel: 4,
            rms_eps: 1e-6,
            hadamard_block: Some(128),
            vocab: 512,
            max_seq_len: 32,
            max_batch: 16,
        };
        let mut rng = Rng::new(seed);
        let shapes = Self::matrix_shapes(&cfg);
        let mats = shapes
            .iter()
            .map(|&(rows, cols)| {
                (0..rows * cols / 128)
                    .map(|_| BlockPQ2_0 {
                        d: f16::from_f32(0.02 + rng.f32().abs() * 0.02),
                        qs: std::array::from_fn(|_| {
                            // Ternary codes only (00, 01, 10) — a trained model's.
                            let mut b = 0u8;
                            for lane in 0..4 {
                                b |= ((rng.next_u64() % 3) as u8) << (2 * lane);
                            }
                            b
                        }),
                    })
                    .collect()
            })
            .collect();
        let norms = (0..9)
            .map(|i| {
                let n = if i == 4 || i == 5 {
                    cfg.head_dim
                } else if i == 8 {
                    cfg.head_v_dim
                } else {
                    cfg.hidden
                };
                (0..n).map(|_| 1.0 + rng.f32() * 0.1).collect()
            })
            .collect();
        let widths = [cfg.hidden, cfg.intermediate, cfg.heads_width(), cfg.inner()];
        let signs = widths
            .iter()
            .fold(Vec::<(usize, Vec<f32>)>::new(), |mut acc, &w| {
                if !acc.iter().any(|(x, _)| *x == w) {
                    acc.push((w, rng.signs(w)));
                }
                acc
            });
        let half = cfg.n_rot / 2;
        let mut cos = vec![0.0f32; cfg.max_seq_len * half];
        let mut sin = vec![0.0f32; cfg.max_seq_len * half];
        for p in 0..cfg.max_seq_len {
            for i in 0..half {
                let theta = p as f32 * 10_000f32.powf(-2.0 * i as f32 / cfg.n_rot as f32);
                cos[p * half + i] = theta.cos();
                sin[p * half + i] = theta.sin();
            }
        }
        Self {
            ab: rng.vec(2 * cfg.n_v_heads * cfg.hidden, 0.05),
            conv: rng.vec(cfg.conv_dim() * 4, 0.5),
            a_neg: (0..cfg.n_v_heads).map(|h| -0.25 - h as f32 * 0.5).collect(),
            dt_bias: (0..cfg.n_v_heads)
                .map(|h| -2.0 + h as f32 * 0.125)
                .collect(),
            cfg,
            mats,
            norms,
            signs,
            cos,
            sin,
        }
    }

    /// `(rows, cols)` of: linear qkv, gate, ssm_out, ffn gate, up, down (x2
    /// layers) and full q, k, v, output, then the LM head.
    fn matrix_shapes(c: &Qwen35GpuConfig) -> Vec<(usize, usize)> {
        let kv = c.n_kv_heads * c.head_dim;
        vec![
            (c.conv_dim(), c.hidden),
            (c.inner(), c.hidden),
            (c.hidden, c.inner()),
            (c.intermediate, c.hidden),
            (c.intermediate, c.hidden),
            (c.hidden, c.intermediate),
            (2 * c.heads_width(), c.hidden),
            (kv, c.hidden),
            (kv, c.hidden),
            (c.hidden, c.heads_width()),
            (c.intermediate, c.hidden),
            (c.intermediate, c.hidden),
            (c.hidden, c.intermediate),
            (c.vocab, c.hidden),
        ]
    }

    fn matrix(&self, i: usize) -> Qwen35Matrix<'_> {
        let (rows, cols) = Self::matrix_shapes(&self.cfg)[i];
        Qwen35Matrix {
            rows,
            cols,
            data: Qwen35MatrixData::Pq2_0(&self.mats[i]),
        }
    }

    fn weights(&self) -> Qwen35ModelWeights<'_> {
        let linear = Qwen35LinearAttentionWeights {
            attn_norm: &self.norms[0],
            post_attention_norm: &self.norms[1],
            attn_qkv: self.matrix(0),
            attn_gate: self.matrix(1),
            ssm_alpha_beta: &self.ab,
            ssm_conv1d: &self.conv,
            a_neg: &self.a_neg,
            dt_bias: &self.dt_bias,
            ssm_norm: &self.norms[8],
            ssm_out: self.matrix(2),
            ffn_gate: self.matrix(3),
            ffn_up: self.matrix(4),
            ffn_down: self.matrix(5),
        };
        let full = Qwen35FullAttentionWeights {
            attn_norm: &self.norms[2],
            post_attention_norm: &self.norms[3],
            attn_q: self.matrix(6),
            attn_k: self.matrix(7),
            attn_v: self.matrix(8),
            attn_output: self.matrix(9),
            attn_q_norm: &self.norms[4],
            attn_k_norm: &self.norms[5],
            ffn_gate: self.matrix(10),
            ffn_up: self.matrix(11),
            ffn_down: self.matrix(12),
        };
        Qwen35ModelWeights {
            config: self.cfg.clone(),
            layers: vec![
                Qwen35LayerWeights::LinearAttention(Box::new(linear)),
                Qwen35LayerWeights::FullAttention(Box::new(full)),
            ],
            output_norm: &self.norms[6],
            lm_head: self.matrix(13),
            signs: self.signs.iter().map(|(w, s)| (*w, s.as_slice())).collect(),
            rope_cos: &self.cos,
            rope_sin: &self.sin,
        }
    }
}

/// Prefill of `T` tokens is bitwise the same as `T` single-token forwards
/// (every batched kernel runs the single-token arithmetic per column), a
/// reset really clears the recurrent state, and the sparse KV cache holds
/// one slot per full-attention layer.
#[test]
fn prefill_is_bitwise_sequential_decode_and_reset_clears_the_state() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::new(5);
    let weights = tiny.weights();
    let mut model = Qwen35GpuModel::new(&weights, Qwen35Residency::Copied).expect("model builds");
    assert_eq!(model.layer_kv_slots(), &[None, Some(0)]);
    assert_eq!(model.layer_rec_slots(), &[Some(0), None]);
    let c = tiny.cfg.clone();
    assert_eq!(
        model.kv_cache_bytes(),
        2 * (c.n_kv_heads * c.max_seq_len * c.head_dim * 2) as u64,
        "one KV slot, not one per layer"
    );
    let mut rng = Rng::new(77);
    let t_len = 11usize;
    let rows = rng.vec(t_len * c.hidden, 1.0);

    let mut prefill_logits = vec![0.0f32; c.vocab];
    model
        .forward(&rows, 0, Some(&mut prefill_logits))
        .expect("prefill");
    model.reset();
    let mut logits = vec![0.0f32; c.vocab];
    for t in 0..t_len {
        model
            .forward(
                &rows[t * c.hidden..(t + 1) * c.hidden],
                t,
                Some(&mut logits),
            )
            .expect("decode");
    }
    assert_bits(&prefill_logits, &logits, "prefill vs sequential decode");
    assert!(logits.iter().all(|v| v.is_finite()));

    model.reset();
    let mut again = vec![0.0f32; c.vocab];
    model.forward(&rows, 0, Some(&mut again)).expect("replay");
    assert_bits(&again, &prefill_logits, "replay after reset");

    // The per-layer dump ends where the fused forward ends.
    model.reset();
    let (dump, dump_logits) = model.forward_with_dump(&rows, 0).expect("dump");
    assert_eq!(dump.len(), 2);
    assert_bits(&dump_logits, &prefill_logits, "dump logits");
}

/// Zero-copy residency binds every matrix inside the mapping at its offset
/// and computes exactly what the copied model does.
#[test]
fn mapped_residency_aliases_the_weights_and_matches_the_copied_model() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::new(9);
    // Lay every quantized matrix out back to back in one page-aligned
    // allocation standing in for a file mapping.
    let page = host_page_size();
    let total: usize = tiny.mats.iter().map(|m| m.len() * 34).sum();
    let len = total.div_ceil(page) * page;
    let layout = std::alloc::Layout::from_size_align(len, page).expect("layout");
    // SAFETY: non-zero size; freed at the end of the test.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    let mut placed: Vec<&[BlockPQ2_0]> = Vec::new();
    let mut off = 0usize;
    for m in &tiny.mats {
        let bytes = m.len() * 34;
        // SAFETY: `[base + off, base + off + bytes)` lies inside the
        // allocation; `BlockPQ2_0` is 2-byte aligned and `off` is even.
        unsafe {
            std::ptr::copy_nonoverlapping(m.as_ptr().cast::<u8>(), base.add(off), bytes);
            placed.push(std::slice::from_raw_parts(
                base.add(off).cast::<BlockPQ2_0>(),
                m.len(),
            ));
        }
        off += bytes;
    }
    // SAFETY: the allocation is page-aligned, a whole number of pages and
    // outlives both models below.
    let region = unsafe { Qwen35MappedRegion::from_mapping(std::slice::from_raw_parts(base, len)) };
    let mut weights = tiny.weights();
    let shapes = TinyModel::matrix_shapes(&tiny.cfg);
    let rebind = |i: usize| Qwen35Matrix {
        rows: shapes[i].0,
        cols: shapes[i].1,
        data: Qwen35MatrixData::Pq2_0(placed[i]),
    };
    if let Qwen35LayerWeights::LinearAttention(l) = &mut weights.layers[0] {
        l.attn_qkv = rebind(0);
        l.attn_gate = rebind(1);
        l.ssm_out = rebind(2);
        l.ffn_gate = rebind(3);
        l.ffn_up = rebind(4);
        l.ffn_down = rebind(5);
    }
    if let Qwen35LayerWeights::FullAttention(f) = &mut weights.layers[1] {
        f.attn_q = rebind(6);
        f.attn_k = rebind(7);
        f.attn_v = rebind(8);
        f.attn_output = rebind(9);
        f.ffn_gate = rebind(10);
        f.ffn_up = rebind(11);
        f.ffn_down = rebind(12);
    }
    weights.lm_head = rebind(13);

    let rows = Rng::new(3).vec(4 * tiny.cfg.hidden, 1.0);
    let mut mapped_logits = vec![0.0f32; tiny.cfg.vocab];
    {
        let mut mapped =
            Qwen35GpuModel::new(&weights, Qwen35Residency::Mapped(region)).expect("mapped model");
        assert!(mapped.is_mapped());
        mapped
            .forward(&rows, 0, Some(&mut mapped_logits))
            .expect("mapped forward");
    }
    let mut copied =
        Qwen35GpuModel::new(&tiny.weights(), Qwen35Residency::Copied).expect("copied model");
    let mut copied_logits = vec![0.0f32; tiny.cfg.vocab];
    copied
        .forward(&rows, 0, Some(&mut copied_logits))
        .expect("copied forward");
    assert_bits(&mapped_logits, &copied_logits, "mapped vs copied");
    // SAFETY: allocated above with this layout; no model references it now.
    unsafe { std::alloc::dealloc(base, layout) };
}

/// Geometry the kernels cannot serve is refused with a typed error naming
/// the constraint, never run.
#[test]
fn unsupported_geometry_is_refused() {
    let base = TinyModel::new(1).cfg;
    let cases: Vec<(Qwen35GpuConfig, &str)> = vec![
        (
            Qwen35GpuConfig {
                head_dim: 512,
                ..base.clone()
            },
            "head_dim",
        ),
        (
            Qwen35GpuConfig {
                conv_kernel: 3,
                ..base.clone()
            },
            "conv_kernel",
        ),
        (
            Qwen35GpuConfig {
                hidden: 6272,
                ..base.clone()
            },
            "hidden",
        ),
        (
            Qwen35GpuConfig {
                n_rot: 3,
                ..base.clone()
            },
            "n_rot",
        ),
        (
            Qwen35GpuConfig {
                head_k_dim: 256,
                ..base.clone()
            },
            "head_k_dim",
        ),
        (
            Qwen35GpuConfig {
                hadamard_block: Some(96),
                ..base.clone()
            },
            "Hadamard",
        ),
        (
            Qwen35GpuConfig {
                n_v_heads: 5,
                ..base.clone()
            },
            "n_v_heads",
        ),
        // A non-square Gated-DeltaNet state: the kernel's output scale is
        // only the reference's for `head_k_dim == head_v_dim`.
        (
            Qwen35GpuConfig {
                head_k_dim: 32,
                ..base.clone()
            },
            "must equal head_v_dim",
        ),
        // Unfolded, 96-wide v-heads: the 128-thread gated-norm span would
        // cut a head in two and sum partial slots no simdgroup wrote.
        (
            Qwen35GpuConfig {
                hadamard_block: None,
                n_v_heads: 8,
                head_k_dim: 96,
                head_v_dim: 96,
                ..base.clone()
            },
            "gated-norm span",
        ),
    ];
    for (cfg, needle) in cases {
        let err = cfg.validate().expect_err("must be refused");
        assert!(err.to_string().contains(needle), "{needle}: {err}");
    }
    assert!(base.validate().is_ok());
    // Unfolded 64-wide v-heads tile the 128-wide span exactly: served.
    let unfolded = Qwen35GpuConfig {
        hadamard_block: None,
        ..base.clone()
    };
    assert!(unfolded.validate().is_ok());
    assert_eq!(unfolded.gated_norm_span(), 128);
}

/// The activation scratch [`Scratch::allocate`] creates is exactly what
/// [`Scratch::floats_per_token`] (and so the context-capacity budget)
/// charges per token.
#[test]
fn scratch_accounting_matches_the_allocated_buffers() {
    let Some(graph) = session() else {
        return;
    };
    let cfg = TinyModel::new(2).cfg;
    let capacity = 3usize;
    let scratch = Scratch::allocate(&graph, &cfg, capacity).expect("scratch");
    let buffers = [
        &scratch.resid,
        &scratch.normed,
        &scratch.rotated,
        &scratch.qkv,
        &scratch.conv_out,
        &scratch.z,
        &scratch.ab,
        &scratch.gdn_out,
        &scratch.gdn_rot,
        &scratch.q_all,
        &scratch.k,
        &scratch.v,
        &scratch.q_rope,
        &scratch.k_rope,
        &scratch.attn,
        &scratch.attn_rot,
        &scratch.ffn_gate,
        &scratch.ffn_up,
        &scratch.ffn_act,
    ];
    let allocated: u64 = buffers.iter().map(|b| b.length()).sum();
    assert_eq!(
        allocated,
        (capacity * Scratch::floats_per_token(&cfg) * 4) as u64
    );
}

/// The context ceiling the module docs quote for the real 27B on this M3,
/// from the device limits measured on it and the weight bytes the runner
/// binds for each band — so the documented figures cannot drift from the
/// code that enforces them.
#[test]
fn context_capacity_of_the_27b_on_the_m3_matches_the_documented_figures() {
    const M3_MAX_BUFFER_LENGTH: u64 = 14_302_248_960;
    const M3_RECOMMENDED_WORKING_SET: u64 = 19_069_665_280;
    let cfg = Qwen35GpuConfig {
        hidden: 5120,
        intermediate: 17408,
        n_heads: 24,
        n_kv_heads: 4,
        head_dim: 256,
        n_rot: 64,
        n_k_heads: 16,
        n_v_heads: 48,
        head_k_dim: 128,
        head_v_dim: 128,
        conv_kernel: 4,
        rms_eps: 1e-6,
        hadamard_block: Some(1024),
        vocab: 248_320,
        max_seq_len: 8192,
        max_batch: 512,
    };
    cfg.validate().expect("the 27B geometry is served");
    // 140 384 floats per token: 274.2 MiB of scratch at a 512-token chunk.
    assert_eq!(Scratch::floats_per_token(&cfg), 140_384);
    let capacity = |weight_bytes: u64, max_batch: usize| {
        let cfg = Qwen35GpuConfig {
            max_batch,
            ..cfg.clone()
        };
        qwen35_context_capacity(
            &cfg,
            16,
            48,
            weight_bytes,
            M3_MAX_BUFFER_LENGTH,
            M3_RECOMMENDED_WORKING_SET,
        )
    };
    // Every matrix, the LM head and the widened ssm gates, per band.
    const PQ2_0_WEIGHTS: u64 = 6_893_936_640;
    const PTQ1_0_WEIGHTS: u64 = 5_694_013_440;
    assert_eq!(capacity(PQ2_0_WEIGHTS, 512), 178_034);
    assert_eq!(capacity(PTQ1_0_WEIGHTS, 512), 196_246);
    // The scratch is charged at `max_batch` tokens, not one.
    assert!(capacity(PQ2_0_WEIGHTS, 1) > capacity(PQ2_0_WEIGHTS, 512));
    // With an unbounded working set, one K (or V) buffer binds: 14.30 GB /
    // (16 slots x 4 heads x 256 dims x 2 B).
    let by_buffer =
        qwen35_context_capacity(&cfg, 16, 48, PQ2_0_WEIGHTS, M3_MAX_BUFFER_LENGTH, u64::MAX);
    assert_eq!(by_buffer, 436_470);
}

/// A per-kernel trace captures every stage of both layer kinds, and running
/// the traced layers in order reproduces the fused forward bitwise.
#[test]
fn trace_layer_reproduces_the_forward_stage_by_stage() {
    if session().is_none() {
        return;
    }
    let tiny = TinyModel::new(21);
    let weights = tiny.weights();
    let c = tiny.cfg.clone();
    let rows = Rng::new(8).vec(2 * c.hidden, 1.0);
    let mut model = Qwen35GpuModel::new(&weights, Qwen35Residency::Copied).expect("model");
    let (dump, _) = model.forward_with_dump(&rows, 0).expect("dump");
    model.reset();
    let t0 = model.trace_layer(0, &rows, 0).expect("trace linear");
    for stage in [
        "attn_norm",
        "attn_norm_rotated",
        "attn_qkv",
        "attn_gate",
        "ssm_alpha_beta",
        "conv_silu",
        "gdn",
        "gated_norm_rotated",
        "attn_residual",
        "ffn_norm",
        "ffn_gate",
        "ffn_act_rotated",
        "residual",
    ] {
        assert!(t0.stage(stage).is_some(), "linear trace lacks {stage}");
    }
    let after0 = t0.stage("residual").expect("residual").to_vec();
    assert_bits(&after0, &dump[0], "linear layer output");
    let t1 = model.trace_layer(1, &after0, 0).expect("trace full");
    for stage in [
        "attn_q",
        "attn_k",
        "attn_v",
        "q_rope",
        "k_rope",
        "attention",
        "attn_gated_rotated",
        "residual",
    ] {
        assert!(t1.stage(stage).is_some(), "full trace lacks {stage}");
    }
    assert_bits(
        t1.stage("residual").expect("residual"),
        &dump[1],
        "full layer output",
    );
}
