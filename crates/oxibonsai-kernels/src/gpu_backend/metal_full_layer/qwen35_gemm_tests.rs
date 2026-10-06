//! The batched-prefill GEMMs (`q35_gemm_*`) against the decode GEMVs they
//! replace, on every weight format, plus the tile-position invariance of a
//! GEMM row and a micro-benchmark at the Bonsai 2 27B's projection shapes.

use std::time::Instant;

use metal::objc::rc::autoreleasepool;
use metal::objc::{msg_send, sel, sel_impl};
use metal::Buffer;

use super::prefill::{encode_gemm, GEMM_TILE};
use super::*;

/// Largest `max |gemm − gemv| / max |gemv|` the f32-activation GEMM may
/// show against the GEMV (only the summation order differs).
const PARITY_REL: f32 = 1e-5;
/// The same bound for the half-activation (`_ha`) GEMMs, whose activations
/// are rounded to `f16` (2⁻¹¹ relative) before every product.
const PARITY_REL_HALF_A: f32 = 4e-3;

/// A reproducible pseudo-random stream (an LCG), so no dependency is needed.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f32 / 2_147_483_648.0) - 1.0
    }

    fn byte_below(&mut self, bound: u32) -> u8 {
        (self.next_u32() % bound) as u8
    }

    /// An `f16` block scale in `[0.01, 0.05)`, little-endian.
    fn scale(&mut self) -> [u8; 2] {
        let d = 0.01 + 0.02 * (self.unit() + 1.0);
        half::f16::from_f32(d).to_bits().to_le_bytes()
    }
}

/// The weight formats the GEMMs serve, with the GEMV / GEMM entry points and
/// the bytes of one 128-element super-block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fmt {
    Pq2,
    Ptq1,
    Tq2,
    Q2g64,
    Q1,
    F32,
}

impl Fmt {
    const ALL: [Self; 6] = [
        Self::Pq2,
        Self::Ptq1,
        Self::Tq2,
        Self::Q2g64,
        Self::Q1,
        Self::F32,
    ];

    fn tag(self) -> &'static str {
        match self {
            Self::Pq2 => "pq2",
            Self::Ptq1 => "ptq1",
            Self::Tq2 => "tq2",
            Self::Q2g64 => "q2g64",
            Self::Q1 => "q1",
            Self::F32 => "f32",
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Pq2 | Self::Tq2 => 34,
            Self::Ptq1 => 28,
            Self::Q2g64 => 36,
            Self::Q1 => 18,
            Self::F32 => 512,
        }
    }

    /// Random, well-formed weights for a `rows × cols` matrix in this
    /// format's on-disk block layout.
    fn weights(self, rows: usize, cols: usize, rng: &mut Lcg) -> Vec<u8> {
        let nsb = cols / 128;
        let mut out = Vec::with_capacity(rows * nsb * self.block_bytes());
        for _ in 0..rows * nsb {
            match self {
                Self::Pq2 => {
                    out.extend_from_slice(&rng.scale());
                    out.extend((0..32).map(|_| rng.byte_below(256)));
                }
                Self::Tq2 => {
                    out.extend((0..32).map(|_| rng.byte_below(256)));
                    out.extend_from_slice(&rng.scale());
                }
                Self::Q2g64 => {
                    for _ in 0..2 {
                        out.extend_from_slice(&rng.scale());
                        out.extend((0..16).map(|_| rng.byte_below(256)));
                    }
                }
                Self::Q1 => {
                    out.extend_from_slice(&rng.scale());
                    out.extend((0..16).map(|_| rng.byte_below(256)));
                }
                Self::Ptq1 => {
                    // Five trits per `qs` byte (< 3^5), four per `qh` byte.
                    out.extend((0..24).map(|_| rng.byte_below(243)));
                    out.extend((0..2).map(|_| rng.byte_below(81)));
                    out.extend_from_slice(&rng.scale());
                }
                Self::F32 => {
                    for _ in 0..128 {
                        out.extend_from_slice(&(0.05 * rng.unit()).to_le_bytes());
                    }
                }
            }
        }
        out
    }
}

fn session() -> Option<Arc<MetalGraph>> {
    match MetalGraph::new_session() {
        Ok(graph) => Some(graph),
        Err(MetalGraphError::DeviceNotFound) => None,
        Err(e) => panic!("the combined Metal library must build on this device: {e}"),
    }
}

/// A matrix of `fmt` bound for both kernels (`gemm` names the GEMM entry
/// point to bind: the default or its `_ha` twin).
fn device_matrix(
    graph: &MetalGraph,
    fmt: Fmt,
    rows: usize,
    cols: usize,
    bytes: &[u8],
    gemm: &str,
) -> DeviceMatrix {
    DeviceMatrix {
        buffer: upload_bytes(graph, bytes).expect("weights upload"),
        offset: 0,
        rows,
        cols,
        pso: graph
            .pipeline_for(&format!("q35_gemv_{}", fmt.tag()))
            .expect("GEMV pipeline"),
        gemm_pso: graph.pipeline_for(gemm).expect("GEMM pipeline"),
    }
}

/// One matmul in its own command buffer; returns the GPU seconds.
fn run(
    graph: &MetalGraph,
    m: &DeviceMatrix,
    x: &Buffer,
    y: &Buffer,
    t_len: usize,
    accumulate: bool,
    gemm: bool,
) -> f64 {
    autoreleasepool(|| {
        let cmd = graph.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if gemm {
            encode_gemm(enc, m, (x, 0), (y, 0), t_len, accumulate);
        } else {
            encode_gemv(enc, m, (x, 0), (y, 0), t_len, accumulate);
        }
        enc.end_encoding();
        commit_and_wait(cmd, "q35_gemm_test").expect("command buffer");
        // SAFETY: the command buffer completed (`commit_and_wait`), so both
        // timestamps are readable.
        let (start, end): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        (end - start).max(0.0)
    })
}

fn upload(graph: &MetalGraph, values: &[f32]) -> Buffer {
    upload_f32(graph, values).expect("upload")
}

/// `max |a − b| / max |b|`.
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

/// Every format, token counts 1, 7, 64, 100 and 512 (partial and whole
/// tiles), a row count that leaves a partial weight tile, plain and
/// accumulating: the GEMM equals the GEMV to [`PARITY_REL`], and the
/// half-activation twin to [`PARITY_REL_HALF_A`].
#[test]
fn every_gemm_matches_the_gemv_it_replaces() {
    let Some(graph) = session() else {
        return;
    };
    let (rows, cols) = (200usize, 384usize);
    let mut rng = Lcg(0x9E37_79B9);
    for fmt in Fmt::ALL {
        let bytes = fmt.weights(rows, cols, &mut rng);
        let mut entries = vec![(format!("q35_gemm_{}", fmt.tag()), PARITY_REL)];
        if fmt != Fmt::F32 {
            entries.push((format!("q35_gemm_{}_ha", fmt.tag()), PARITY_REL_HALF_A));
        }
        for (entry, bound) in &entries {
            let m = device_matrix(&graph, fmt, rows, cols, &bytes, entry);
            for t_len in [1usize, 7, 64, 100, 512] {
                let x: Vec<f32> = (0..t_len * cols).map(|_| rng.unit()).collect();
                let residual: Vec<f32> = (0..t_len * rows).map(|_| rng.unit()).collect();
                let xb = upload(&graph, &x);
                for accumulate in [false, true] {
                    let init = if accumulate {
                        residual.clone()
                    } else {
                        vec![0.0; t_len * rows]
                    };
                    let (ya, yb) = (upload(&graph, &init), upload(&graph, &init));
                    run(&graph, &m, &xb, &ya, t_len, accumulate, false);
                    run(&graph, &m, &xb, &yb, t_len, accumulate, true);
                    let gemv = read_buffer(&ya, 0, t_len * rows);
                    let gemm = read_buffer(&yb, 0, t_len * rows);
                    assert!(gemm.iter().all(|v| v.is_finite()), "{entry}: non-finite");
                    let rel = rel_err(&gemm, &gemv);
                    assert!(
                        rel <= *bound,
                        "{entry} t_len {t_len} accumulate {accumulate}: GEMM vs GEMV worst \
                         relative error {rel:e} > {bound:e}"
                    );
                }
            }
        }
    }
}

/// Whether a GEMM row's result depends on where the row sits in its
/// 64-token tile: the same activation rows run at a different tile offset
/// (a sub-range of the prompt, as a different prefill chunking places them)
/// give the same bits. The GEMV has no tiles at all; the GEMM's rows are
/// independent 8-wide matrix-unit dot products, so the result is expected —
/// and here pinned — to be bitwise position-independent on this device.
#[test]
fn a_gemm_row_does_not_depend_on_its_tile_position() {
    let Some(graph) = session() else {
        return;
    };
    let (rows, cols, t_len) = (136usize, 512usize, 150usize);
    let mut rng = Lcg(0xC0FF_EE11);
    for fmt in [Fmt::Pq2, Fmt::Ptq1, Fmt::F32] {
        let bytes = fmt.weights(rows, cols, &mut rng);
        let m = device_matrix(
            &graph,
            fmt,
            rows,
            cols,
            &bytes,
            &format!("q35_gemm_{}", fmt.tag()),
        );
        let x: Vec<f32> = (0..t_len * cols).map(|_| rng.unit()).collect();
        let whole = {
            let (xb, yb) = (upload(&graph, &x), upload(&graph, &vec![0.0; t_len * rows]));
            run(&graph, &m, &xb, &yb, t_len, false, true);
            read_buffer(&yb, 0, t_len * rows)
        };
        let mut mismatched = 0usize;
        for start in [1usize, 37, 64, 101] {
            let sub = t_len - start;
            let xs = &x[start * cols..];
            let (xb, yb) = (upload(&graph, xs), upload(&graph, &vec![0.0; sub * rows]));
            run(&graph, &m, &xb, &yb, sub, false, true);
            let part = read_buffer(&yb, 0, sub * rows);
            mismatched += part
                .iter()
                .zip(&whole[start * rows..])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
        eprintln!(
            "q35_gemm_{}: {mismatched} bits differ between tile positions",
            fmt.tag()
        );
        assert_eq!(
            mismatched,
            0,
            "q35_gemm_{}: a row's result depends on its tile position",
            fmt.tag()
        );
    }
}

/// One projection shape of the 27B and how many times a token runs it.
struct Shape {
    name: &'static str,
    rows: usize,
    cols: usize,
    per_token: usize,
}

/// Brief §C.5: every folded projection of a 27B token (48 Gated-DeltaNet
/// layers, 16 full-attention layers).
const SHAPES: [Shape; 8] = [
    Shape {
        name: "attn_qkv",
        rows: 10240,
        cols: 5120,
        per_token: 48,
    },
    Shape {
        name: "attn_gate",
        rows: 6144,
        cols: 5120,
        per_token: 48,
    },
    Shape {
        name: "attn_q",
        rows: 12288,
        cols: 5120,
        per_token: 16,
    },
    Shape {
        name: "attn_k/attn_v",
        rows: 1024,
        cols: 5120,
        per_token: 32,
    },
    Shape {
        name: "ssm_out",
        rows: 5120,
        cols: 6144,
        per_token: 48,
    },
    Shape {
        name: "attn_output",
        rows: 5120,
        cols: 6144,
        per_token: 16,
    },
    Shape {
        name: "ffn_gate/ffn_up",
        rows: 17408,
        cols: 5120,
        per_token: 128,
    },
    Shape {
        name: "ffn_down",
        rows: 5120,
        cols: 17408,
        per_token: 64,
    },
];

/// Tokens per benchmarked call (the default prefill chunk).
const BENCH_M: usize = 512;

/// Micro-benchmark of the GEMMs at the 27B's projection shapes, 512 tokens
/// per call, both bands and both activation stagings: the best-of-3 GPU time
/// of each shape, its effective TFLOP/s, and the FLOP-weighted mean over one
/// token's projections (`Σ count·flops / Σ count·time`) — the rate the
/// batched prefill's projections run at. Also the GEMV's time for the same
/// call (the sequential prefill), for the ratio. Recorded, not asserted:
/// the numbers depend on what else the GPU is doing.
#[test]
fn gemm_microbench_at_the_27b_shapes() {
    let Some(graph) = session() else {
        return;
    };
    let mut rng = Lcg(0x5EED_0027);
    let x: Vec<f32> = (0..BENCH_M * 17408).map(|_| rng.unit()).collect();
    let xb = upload(&graph, &x);
    let yb = zeroed(&graph, BENCH_M * 17408 * 4).expect("output");
    for fmt in [Fmt::Pq2, Fmt::Ptq1] {
        let mut weighted = [(0.0f64, 0.0f64); 2];
        let mut gemv_time = 0.0f64;
        for shape in &SHAPES {
            let bytes = fmt.weights(shape.rows, shape.cols, &mut rng);
            let flops = 2.0 * (BENCH_M * shape.rows * shape.cols) as f64;
            let mut line = format!(
                "q35 GEMM bench {} {:<16} {:>5}x{:<5} x{:<3}",
                fmt.tag(),
                shape.name,
                shape.rows,
                shape.cols,
                shape.per_token
            );
            for (slot, suffix) in ["", "_ha"].iter().enumerate() {
                let entry = format!("q35_gemm_{}{suffix}", fmt.tag());
                let m = device_matrix(&graph, fmt, shape.rows, shape.cols, &bytes, &entry);
                run(&graph, &m, &xb, &yb, BENCH_M, false, true);
                let best = (0..3)
                    .map(|_| run(&graph, &m, &xb, &yb, BENCH_M, false, true))
                    .fold(f64::INFINITY, f64::min);
                weighted[slot].0 += shape.per_token as f64 * flops;
                weighted[slot].1 += shape.per_token as f64 * best;
                line.push_str(&format!(
                    " | {entry} {:.2} ms {:.2} TFLOP/s",
                    best * 1e3,
                    flops / best / 1e12
                ));
                if slot == 0 {
                    let wall = Instant::now();
                    let gemv = run(&graph, &m, &xb, &yb, BENCH_M, false, false);
                    gemv_time += shape.per_token as f64 * gemv;
                    line.push_str(&format!(
                        " | GEMV {:.1} ms ({:.1}x, wall {:.1} ms)",
                        gemv * 1e3,
                        gemv / best,
                        wall.elapsed().as_secs_f64() * 1e3
                    ));
                }
            }
            eprintln!("{line}");
        }
        for (slot, label) in ["f32-A", "half-A"].iter().enumerate() {
            let (flops, time) = weighted[slot];
            eprintln!(
                "q35 GEMM bench {} {label}: FLOP-weighted mean {:.3} TFLOP/s; projections {:.1} \
                 ms per token at M={BENCH_M} (GEMV {:.1} ms per token)",
                fmt.tag(),
                flops / time / 1e12,
                time / BENCH_M as f64 * 1e3,
                gemv_time / BENCH_M as f64 * 1e3,
            );
        }
    }
    const { assert!(GEMM_TILE == 64) };
}

/// Where the GEMM starts beating the per-token GEMV: both at a 27B
/// projection shape (`attn_qkv`, 10240 × 5120) for calls of 1..64 tokens,
/// best of three GPU times each — the measurement behind
/// [`Q35_GEMM_MIN_COLS`](super::Q35_GEMM_MIN_COLS). Asserts only that the
/// threshold's own call size is not slower on the GEMM than on the GEMV
/// by more than measurement noise allows.
#[test]
fn gemm_overtakes_the_gemv_at_the_min_cols_threshold() {
    let Some(graph) = session() else {
        return;
    };
    let (rows, cols) = (10240usize, 5120usize);
    let mut rng = Lcg(0xC205_50E4);
    let x: Vec<f32> = (0..64 * cols).map(|_| rng.unit()).collect();
    let xb = upload(&graph, &x);
    let yb = zeroed(&graph, 64 * rows * 4).expect("output");
    for fmt in [Fmt::Pq2, Fmt::Ptq1] {
        let bytes = fmt.weights(rows, cols, &mut rng);
        let m = device_matrix(
            &graph,
            fmt,
            rows,
            cols,
            &bytes,
            &format!("q35_gemm_{}", fmt.tag()),
        );
        let mut line = format!("q35 GEMM vs GEMV crossover {} attn_qkv:", fmt.tag());
        let mut at_threshold = (0.0f64, 0.0f64);
        for t_len in [1usize, 2, 4, 8, 16, 32, 64] {
            let best = |gemm: bool| {
                run(&graph, &m, &xb, &yb, t_len, false, gemm);
                (0..3)
                    .map(|_| run(&graph, &m, &xb, &yb, t_len, false, gemm))
                    .fold(f64::INFINITY, f64::min)
            };
            let (gemv, gemm) = (best(false), best(true));
            if t_len == super::Q35_GEMM_MIN_COLS {
                at_threshold = (gemv, gemm);
            }
            line.push_str(&format!(
                " | t={t_len} GEMV {:.2} ms GEMM {:.2} ms",
                gemv * 1e3,
                gemm * 1e3
            ));
        }
        eprintln!("{line}");
        let (gemv, gemm) = at_threshold;
        assert!(
            gemm <= gemv * 1.5,
            "{}: at the threshold the GEMM ({:.2} ms) must not lose to the GEMV ({:.2} ms)",
            fmt.tag(),
            gemm * 1e3,
            gemv * 1e3
        );
    }
}
