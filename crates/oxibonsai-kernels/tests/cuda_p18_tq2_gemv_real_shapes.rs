//! CUDA-P18 — the 2-bit GEMV geometry guard on **real** shapes.
//!
//! Checklist item (verbatim, `cuda_hybrid_parity_plan.rs` / CUDA_HANDOVER.md
//! Appendix A): "2-bit GEMV geometry guard: `encode_gemv_tq2_cached` over every
//! projection and the LM head of Ternary-Bonsai-1.7B and -8B (real shapes)
//! returns no `InvalidDimensions` and matches the CPU TQ2 GEMV. cos >= 0.999."
//!
//! Before this harness the only evidence was `cuda_tq2_gemv_parity` (synthetic
//! shapes, through `encode_lm_head_gemv_tq2`) and `encode_ternary.rs`'s own
//! synthetic test; neither ran `CudaGraph::encode_gemv_tq2_cached` — the entry
//! point whose pre-launch guards (`check_gemv_2bit_dims`,
//! `check_gemv_2bit_weight`) P18 is about — on the geometry a real model hands
//! it.
//!
//! # Which bytes
//!
//! Since 3c9993a, `oxibonsai convert … --quant tq2_0_g128` (what
//! `scripts/download_ternary.sh` runs) writes every projection, the LM head
//! and the embedding as **`TQ2_0_g128`** (ggml id 42, `qs` first); `--quant
//! pq2_0` writes PrismML **`PQ2_0`** (ggml id 142, `d` first). Files converted
//! before 3c9993a — such as the current `models/Ternary-Bonsai-{1.7B,8B}.gguf`
//! — carry `PQ2_0` (id 142); the `*-tq2_0_g128.gguf` re-encodes carry id 42.
//! The harness accepts both:
//!
//! - a `TQ2_0_g128` tensor runs the **TQ2 leg** on its own bytes;
//! - a `PQ2_0` tensor runs the **PQ2 leg** (`get_or_upload_weight_pq2_soa_for_epoch`
//!   and `encode_gemv_pq2_cached`, the same two-bit guards, against the CPU
//!   `dequant_prism::gemv_pq2_0`) and then the **TQ2 leg** on a lossless
//!   PQ2→TQ2 transcode (the FP16 scale moved behind the 32 code bytes). The
//!   transcode is exact only when the tensor holds no `0b11` (`+2`) code —
//!   `PQ2_0` decodes it as `+2`, `TQ2_0_g128` as `0` — which an absmean
//!   ternary checkpoint never emits. A tensor that does has no lossless TQ2
//!   form, so its TQ2 leg cannot run: it is named, its PQ2 leg still runs, and
//!   the file **fails** P18 once the walk ends (see "Coverage"). The CPU TQ2
//!   and CPU PQ2 outputs of a transcoded tensor are also checked against each
//!   other.
//!
//! # The TQ2 leg (P18 proper), per layer
//!
//! 1. **Per-matrix** (`q`, `k`, `v`, `o`, `gate`, `up`, `down`): blocks
//!    uploaded through `upload_weight_tq2_soa_for_epoch`, then
//!    `encode_gemv_tq2_cached` — the sequence `NativeCudaBackend::
//!    {upload_weights_ternary, gemv_tq2_g128_cached}` runs
//!    (`nativecudabackend_traits.rs:113-140`), only epoch-attributed so the
//!    harness can free each layer again.
//! 2. **Fused** (`Q‖K‖V`, `gate‖up`): the AoS byte parts concatenated and
//!    uploaded through `get_or_upload_weight_tq2_soa_lazy_for_epoch`, the
//!    buffer length checked against the TQ2 SoA size, then
//!    `encode_gemv_tq2_cached` over the summed row count — exactly what
//!    `oxibonsai-model`'s `try_cuda_gemv_ternary_fused` (`block/types/
//!    forward.rs:194-246`) does. Its CPU reference is the concatenation of the
//!    per-matrix CPU outputs (the parts share one input vector, as in the
//!    product).
//!
//! The LM head (`output.weight`, or `token_embd.weight` when the head is tied)
//! goes through `encode_gemv_tq2_cached` and through the product LM-head entry
//! point `encode_lm_head_gemv_tq2` (and `encode_gemv_pq2_cached` for a `PQ2_0`
//! head).
//!
//! The CPU oracle is `oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128` on the
//! same blocks and the same deterministic activation vector
//! (`oxibonsai_testkit::gguf_fixture::deterministic_weights`, a seeded LCG in
//! `[-1, 1]`). Every GPU result must reach `cos >= 0.999`; any
//! `CudaGraphError::InvalidDimensions` fails the test by name.
//!
//! VRAM stays bounded: every upload of a layer is attributed to one epoch from
//! `next_cuda_model_epoch`, and `release_model_epoch` frees the layer before
//! the next one is uploaded (the harness asserts it released exactly the
//! layer's entries). Peak residency is one layer's weights (under 150 MB on
//! the 8B) plus the LM head.
//!
//! # Coverage
//!
//! The checklist item asks for **every** projection and the LM head, so a
//! partial walk is a failure, never a pass. `executed: true` is written only
//! after every comparison of the model has passed **and** the TQ2 leg covered:
//!
//! - every layer of the file: the walk over `blk.{i}.attn_q.weight` must reach
//!   every `blk.*` index present, and must equal `{arch}.block_count` when the
//!   metadata carries it;
//! - all seven projections of every layer and both fused product shapes;
//! - every other two-bit `blk.*` tensor (a qwen3 GGUF has none; one would be a
//!   projection the harness does not know, so it counts as uncovered);
//! - the LM head, which must be two-bit, transcode losslessly, and pass both
//!   `encode_gemv_tq2_cached` and `encode_lm_head_gemv_tq2`;
//! - and the TQ2 GEMV count must equal `layers × 9 + 2` exactly.
//!
//! Anything uncovered (a `+2` code, a non-two-bit or absent LM head, an
//! unknown two-bit layer tensor, a layer past the walk, a `block_count`
//! mismatch) panics with the full list once the walk has ended, so the PQ2
//! legs still report; no capability is recorded for a failed run. Each model
//! self-skips (recording `cuda-hardware`, `executed: false`) only when no CUDA
//! device answers `CudaGraph::global()` or `find_model` does not locate its
//! GGUF. Run with:
//!
//! ```text
//! cargo test --release -p oxibonsai-kernels --features native-cuda \
//!     --test cuda_p18_tq2_gemv_real_shapes -- --test-threads=1 --nocapture
//! ```

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::time::Instant;

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::{count_plus_two_codes, BlockPQ2_0, BlockTQ2_0_g128, GgufTensorType};
use oxibonsai_kernels::dequant_prism::gemv_pq2_0;
use oxibonsai_kernels::gemv_ternary::gemv_tq2_0_g128;
use oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch;
use oxibonsai_kernels::{CudaGraph, CudaGraphError};
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::gguf_fixture::deterministic_weights;
use oxibonsai_testkit::parity::gpu_serial;
use oxibonsai_testkit::workspace::find_model;

/// The checklist's acceptance floor, per GEMV.
const COS_FLOOR: f64 = 0.999;
/// Two-bit group size (weights per block), TQ2_0_g128 and PQ2_0 alike.
const GROUP: usize = 128;
/// AoS / SoA bytes per block, TQ2_0_g128 and PQ2_0 alike.
const BLOCK_BYTES: usize = 34;

/// Capability-record name of [`p18_tq2_gemv_real_shapes_ternary_bonsai_1_7b`].
const TEST_1_7B: &str = "oxibonsai-kernels::cuda_p18_tq2_gemv_real_shapes::\
                         p18_tq2_gemv_real_shapes_ternary_bonsai_1_7b";
/// Capability-record name of [`p18_tq2_gemv_real_shapes_ternary_bonsai_8b`].
const TEST_8B: &str = "oxibonsai-kernels::cuda_p18_tq2_gemv_real_shapes::\
                       p18_tq2_gemv_real_shapes_ternary_bonsai_8b";

/// The seven per-layer projections, in GGUF tensor-name order.
const PROJECTIONS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];
/// Handle slot of a projection's PQ2 upload: `PQ2_SLOT_BASE + projection`.
const PQ2_SLOT_BASE: u64 = 0x10;
/// Handle slot of the fused `Q‖K‖V` upload within a layer.
const SLOT_FUSED_QKV: u64 = 7;
/// Handle slot of the fused `gate‖up` upload within a layer.
const SLOT_FUSED_GATE_UP: u64 = 8;
/// Pseudo-layer index used for the LM-head handles.
const LM_HEAD_LAYER: u64 = 0xFFFF;

/// Storage format of one real two-bit tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TwoBit {
    /// `TQ2_0_g128` (ggml id 42): `qs[32]` first, FP16 `d` last.
    Tq2,
    /// `PQ2_0` (ggml id 142): FP16 `d` first, then `qs[32]`.
    Pq2,
}

/// One real two-bit matrix out of the GGUF.
struct Matrix<'a> {
    name: String,
    format: TwoBit,
    /// Raw AoS bytes as stored in the file.
    raw: &'a [u8],
    /// Output rows (GGUF `ne1`).
    n_rows: usize,
    /// Input width (GGUF `ne0`).
    k: usize,
}

impl Matrix<'_> {
    /// The tensor as `TQ2_0_g128` blocks: its own bytes for a TQ2 tensor, the
    /// lossless transcode for a `PQ2_0` one, or `Err(n)` when `n` `+2` codes
    /// make the transcode lossy.
    fn tq2_blocks(&self) -> Result<Cow<'_, [BlockTQ2_0_g128]>, u64> {
        match self.format {
            TwoBit::Tq2 => Ok(match BlockTQ2_0_g128::slice_from_bytes(self.raw) {
                Ok(blocks) => Cow::Borrowed(blocks),
                Err(_) => Cow::Owned(
                    self.raw
                        .as_chunks::<BLOCK_BYTES>()
                        .0
                        .iter()
                        .map(|c| tq2_block(&c[..32], [c[32], c[33]]))
                        .collect(),
                ),
            }),
            TwoBit::Pq2 => {
                let chunks = self.raw.as_chunks::<BLOCK_BYTES>().0;
                let plus_two: u64 = chunks
                    .iter()
                    .map(|c| u64::from(count_plus_two_codes(&c[2..])))
                    .sum();
                if plus_two > 0 {
                    return Err(plus_two);
                }
                Ok(Cow::Owned(
                    chunks
                        .iter()
                        .map(|c| tq2_block(&c[2..], [c[0], c[1]]))
                        .collect(),
                ))
            }
        }
    }

    /// The tensor as `PQ2_0` blocks (`PQ2_0` tensors only).
    fn pq2_blocks(&self) -> Cow<'_, [BlockPQ2_0]> {
        match BlockPQ2_0::slice_from_bytes(self.raw) {
            Ok(blocks) => Cow::Borrowed(blocks),
            Err(_) => Cow::Owned(
                self.raw
                    .as_chunks::<BLOCK_BYTES>()
                    .0
                    .iter()
                    .map(|c| {
                        let mut qs = [0u8; 32];
                        qs.copy_from_slice(&c[2..]);
                        BlockPQ2_0 {
                            d: f16::from_le_bytes([c[0], c[1]]),
                            qs,
                        }
                    })
                    .collect(),
            ),
        }
    }
}

/// One `TQ2_0_g128` block from its 32 code bytes and its LE FP16 scale.
fn tq2_block(qs_bytes: &[u8], d_le: [u8; 2]) -> BlockTQ2_0_g128 {
    let mut qs = [0u8; 32];
    qs.copy_from_slice(qs_bytes);
    BlockTQ2_0_g128 {
        qs,
        d: f16::from_le_bytes(d_le),
    }
}

/// `TQ2_0_g128` AoS bytes (`qs` first, `d` last) of `blocks`.
fn tq2_aos_bytes(blocks: &[BlockTQ2_0_g128], out: &mut Vec<u8>) {
    for b in blocks {
        out.extend_from_slice(&b.qs);
        out.extend_from_slice(&b.d.to_le_bytes());
    }
}

/// Look up `name` as a two-bit matrix with a consistent size. `None` means
/// the tensor is absent; a present tensor of any other type fails the test.
fn two_bit_matrix<'a>(gguf: &GgufFile<'a>, name: &str) -> Option<Matrix<'a>> {
    let info = gguf.tensors.get(name)?;
    let format = match info.tensor_type {
        GgufTensorType::TQ2_0_g128 => TwoBit::Tq2,
        GgufTensorType::PQ2_0 => TwoBit::Pq2,
        other => panic!(
            "P18: {name} is {other:?}, neither TQ2_0_g128 nor PQ2_0 — not a two-bit ternary \
             GGUF this checklist item covers"
        ),
    };
    assert!(
        info.shape.len() >= 2,
        "P18: {name} has shape {:?}, expected [k, n_rows]",
        info.shape
    );
    let k = usize::try_from(info.shape[0]).expect("k fits usize");
    let n_rows = usize::try_from(info.shape[1]).expect("n_rows fits usize");
    let raw = gguf
        .tensor_data(name)
        .unwrap_or_else(|e| panic!("P18: tensor_data({name}): {e}"));
    assert_eq!(
        raw.len(),
        n_rows * (k / GROUP) * BLOCK_BYTES,
        "P18: {name} byte length does not match its shape [k={k}, n_rows={n_rows}]"
    );
    Some(Matrix {
        name: name.to_string(),
        format,
        raw,
        n_rows,
        k,
    })
}

/// Unwrap a CUDA result, naming the geometry-guard refusal explicitly.
fn expect_cuda<T>(r: Result<T, CudaGraphError>, what: &str, n_rows: usize, k: usize) -> T {
    match r {
        Ok(v) => v,
        Err(CudaGraphError::InvalidDimensions(msg)) => panic!(
            "P18 FAIL: the 2-bit GEMV geometry guard refused a REAL shape — {what} \
             (n_rows={n_rows}, k={k}): InvalidDimensions({msg})"
        ),
        Err(e) => panic!("P18 FAIL: {what} (n_rows={n_rows}, k={k}): {e}"),
    }
}

/// Cosine similarity and absolute error of one result against its reference.
#[derive(Debug, Clone, Copy)]
struct Parity {
    cos: f64,
    max_abs: f32,
    max_ref: f32,
    worst_row: usize,
}

fn compare(reference: &[f32], got: &[f32], what: &str) -> Parity {
    assert_eq!(
        reference.len(),
        got.len(),
        "P18: {what}: reference has {} rows, result {}",
        reference.len(),
        got.len()
    );
    let mut dot = 0.0f64;
    let mut norm_ref = 0.0f64;
    let mut norm_got = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut worst_row = 0usize;
    for (row, (&r, &g)) in reference.iter().zip(got.iter()).enumerate() {
        assert!(g.is_finite(), "P18: {what}: row {row} is not finite ({g})");
        dot += f64::from(r) * f64::from(g);
        norm_ref += f64::from(r) * f64::from(r);
        norm_got += f64::from(g) * f64::from(g);
        let d = (r - g).abs();
        if d > max_abs {
            max_abs = d;
            worst_row = row;
        }
        max_ref = max_ref.max(r.abs());
    }
    let cos = if norm_ref == 0.0 && norm_got == 0.0 {
        1.0
    } else if norm_ref == 0.0 || norm_got == 0.0 {
        0.0
    } else {
        dot / (norm_ref.sqrt() * norm_got.sqrt())
    };
    Parity {
        cos,
        max_abs,
        max_ref,
        worst_row,
    }
}

/// Running summary of one leg (TQ2 or PQ2) over a model.
struct Summary {
    label: &'static str,
    gemvs: usize,
    min_cos: f64,
    min_cos_at: String,
    max_abs: f32,
    max_abs_at: String,
}

impl Summary {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            gemvs: 0,
            min_cos: f64::INFINITY,
            min_cos_at: String::new(),
            max_abs: 0.0,
            max_abs_at: String::new(),
        }
    }

    /// Fold one comparison in, asserting the checklist floor on the spot.
    fn check(&mut self, what: &str, n_rows: usize, k: usize, cpu: &[f32], gpu: &[f32]) -> Parity {
        let p = compare(cpu, gpu, what);
        assert!(
            p.cos >= COS_FLOOR,
            "P18 FAIL: {what} (n_rows={n_rows}, k={k}): cos={:.9} < {COS_FLOOR} \
             (max|Δ|={:.4e} at row {}, cpu={} cuda={})",
            p.cos,
            p.max_abs,
            p.worst_row,
            cpu[p.worst_row],
            gpu[p.worst_row]
        );
        self.gemvs += 1;
        if p.cos < self.min_cos {
            self.min_cos = p.cos;
            self.min_cos_at = what.to_string();
        }
        if p.max_abs > self.max_abs {
            self.max_abs = p.max_abs;
            self.max_abs_at = what.to_string();
        }
        p
    }

    fn report(&self, file: &str) {
        if self.gemvs == 0 {
            println!("P18 SUMMARY {file} [{}]: not run", self.label);
        } else {
            println!(
                "P18 SUMMARY {file} [{}]: {} GEMVs, min cos={:.9} at {}, max|Δ|={:.4e} at {}, \
                 InvalidDimensions: none",
                self.label,
                self.gemvs,
                self.min_cos,
                self.min_cos_at,
                self.max_abs,
                self.max_abs_at
            );
        }
    }
}

/// Both legs' summaries plus the transcode bookkeeping of one model.
struct Report {
    tq2: Summary,
    pq2: Summary,
    /// Worst CPU-TQ2 vs CPU-PQ2 agreement over transcoded tensors.
    transcode_min_cos: f64,
    /// Tensors whose `+2` codes made the TQ2 transcode impossible.
    lossy: Vec<String>,
    /// Every matrix [`check_matrix`] was handed, whatever its TQ2 outcome.
    walked: HashSet<String>,
    /// Every shape the TQ2 leg did not cover, with the reason. Non-empty
    /// fails the model: CUDA-P18 asks for every projection and the LM head.
    uncovered: Vec<String>,
}

/// What the TQ2 leg of one matrix leaves behind for the fused step.
struct Tq2Part<'m> {
    /// The matrix as `TQ2_0_g128` blocks (own bytes or the transcode).
    blocks: Cow<'m, [BlockTQ2_0_g128]>,
    /// CPU `gemv_tq2_0_g128` output on the matrix's activation vector.
    cpu: Vec<f32>,
    /// Worst cosine over this matrix's GPU legs.
    min_cos: f64,
    /// Largest absolute deviation over this matrix's GPU legs.
    max_abs: f32,
}

/// Distinct handle id per (model, layer, slot), far from every product range.
fn handle(model_base: u64, layer: u64, slot: u64) -> u64 {
    model_base | (layer << 8) | slot
}

/// Seeded activation vector for (model, layer, role).
fn activation(k: usize, model_base: u64, layer: u64, role: u64) -> Vec<f32> {
    deterministic_weights(k, (model_base >> 32) ^ (layer << 12) ^ (role << 4) ^ 0x5018)
}

/// Run one matrix through its legs; returns its TQ2 blocks and CPU TQ2 output
/// (for the fused step) when the TQ2 leg ran. `uploads` counts the
/// epoch-attributed entries created.
#[allow(clippy::too_many_arguments)]
fn check_matrix<'m>(
    graph: &Arc<CudaGraph>,
    m: &'m Matrix<'_>,
    input: &[f32],
    h_tq2: u64,
    h_pq2: u64,
    epoch: u64,
    report: &mut Report,
    uploads: &mut usize,
) -> Option<Tq2Part<'m>> {
    report.walked.insert(m.name.clone());
    let mut cos_min = f64::INFINITY;
    let mut abs_max = 0.0f32;
    let mut cpu_pq2 = None;
    if m.format == TwoBit::Pq2 {
        let blocks = m.pq2_blocks();
        let mut cpu = vec![0.0f32; m.n_rows];
        gemv_pq2_0(&blocks, input, &mut cpu, m.n_rows, m.k)
            .unwrap_or_else(|e| panic!("P18: CPU gemv_pq2_0 on {}: {e}", m.name));
        drop(expect_cuda(
            graph.get_or_upload_weight_pq2_soa_for_epoch(h_pq2, m.raw, epoch),
            &format!("PQ2 upload {}", m.name),
            m.n_rows,
            m.k,
        ));
        *uploads += 1;
        let gpu = expect_cuda(
            graph.encode_gemv_pq2_cached(h_pq2, input, m.n_rows, m.k),
            &format!("encode_gemv_pq2_cached {}", m.name),
            m.n_rows,
            m.k,
        );
        let p = report.pq2.check(&m.name, m.n_rows, m.k, &cpu, &gpu);
        cos_min = cos_min.min(p.cos);
        abs_max = abs_max.max(p.max_abs);
        cpu_pq2 = Some(cpu);
    }
    let tq2 = match m.tq2_blocks() {
        Ok(blocks) => blocks,
        Err(plus_two) => {
            println!(
                "P18: {} holds {plus_two} `+2` code(s): the PQ2→TQ2 transcode would be lossy, \
                 so its TQ2 leg is not run — the file fails P18 once the walk ends",
                m.name
            );
            report.lossy.push(m.name.clone());
            report.uncovered.push(format!(
                "{} — TQ2 leg not run: {plus_two} `+2` code(s) make the PQ2→TQ2 transcode lossy",
                m.name
            ));
            return None;
        }
    };
    let mut cpu = vec![0.0f32; m.n_rows];
    gemv_tq2_0_g128(&tq2, input, &mut cpu, m.n_rows, m.k)
        .unwrap_or_else(|e| panic!("P18: CPU gemv_tq2_0_g128 on {}: {e}", m.name));
    if let Some(cpu_pq2) = &cpu_pq2 {
        let t = compare(cpu_pq2, &cpu, &format!("{} CPU PQ2 vs CPU TQ2", m.name));
        assert!(
            t.cos >= COS_FLOOR,
            "P18: {}: the PQ2→TQ2 transcode changed the CPU result (cos={:.9}) — harness bug",
            m.name,
            t.cos
        );
        report.transcode_min_cos = report.transcode_min_cos.min(t.cos);
    }
    expect_cuda(
        graph.upload_weight_tq2_soa_for_epoch(h_tq2, &tq2, epoch),
        &format!("TQ2 upload {}", m.name),
        m.n_rows,
        m.k,
    );
    *uploads += 1;
    let gpu = expect_cuda(
        graph.encode_gemv_tq2_cached(h_tq2, input, m.n_rows, m.k),
        &format!("encode_gemv_tq2_cached {}", m.name),
        m.n_rows,
        m.k,
    );
    let p = report.tq2.check(&m.name, m.n_rows, m.k, &cpu, &gpu);
    cos_min = cos_min.min(p.cos);
    abs_max = abs_max.max(p.max_abs);
    Some(Tq2Part {
        blocks: tq2,
        cpu,
        min_cos: cos_min,
        max_abs: abs_max,
    })
}

/// Run every layer of one ternary GGUF through both legs. Returns the layer
/// count.
fn check_layers(
    graph: &Arc<CudaGraph>,
    gguf: &GgufFile<'_>,
    model_base: u64,
    report: &mut Report,
) -> usize {
    let mut layer = 0usize;
    while gguf
        .tensors
        .get(&format!("blk.{layer}.attn_q.weight"))
        .is_some()
    {
        let l = layer as u64;
        let mats: Vec<Matrix<'_>> = PROJECTIONS
            .iter()
            .map(|p| {
                let name = format!("blk.{layer}.{p}.weight");
                two_bit_matrix(gguf, &name)
                    .unwrap_or_else(|| panic!("P18: required tensor {name} is missing"))
            })
            .collect();
        // The product feeds q/k/v and gate/up the same normed hidden vector.
        for i in [1usize, 2, 4, 5] {
            assert_eq!(
                mats[i].k, mats[0].k,
                "P18: layer {layer}: {} input width {} != q input width {}",
                mats[i].name, mats[i].k, mats[0].k
            );
        }
        let x_hidden = activation(mats[0].k, model_base, l, 0);
        let x_attn = activation(mats[3].k, model_base, l, 1);
        let x_inter = activation(mats[6].k, model_base, l, 2);
        let epoch = next_cuda_model_epoch();
        let mut uploads = 0usize;
        let mut layer_min_cos = f64::INFINITY;
        let mut layer_max_abs = 0.0f32;
        let mut fused_run = 0usize;

        // ── 1. per-matrix legs ───────────────────────────────────────────────
        let mut tq2_parts: Vec<Option<Tq2Part<'_>>> = Vec::with_capacity(PROJECTIONS.len());
        for (slot, m) in mats.iter().enumerate() {
            let input: &[f32] = match slot {
                3 => &x_attn,
                6 => &x_inter,
                _ => &x_hidden,
            };
            let s = slot as u64;
            let done = check_matrix(
                graph,
                m,
                input,
                handle(model_base, l, s),
                handle(model_base, l, PQ2_SLOT_BASE + s),
                epoch,
                report,
                &mut uploads,
            );
            if let Some(part) = &done {
                layer_min_cos = layer_min_cos.min(part.min_cos);
                layer_max_abs = layer_max_abs.max(part.max_abs);
            }
            tq2_parts.push(done);
        }

        // ── 2. fused product shapes (TQ2 leg): AoS concat + lazy upload ─────
        for (slot, label, idx) in [
            (SLOT_FUSED_QKV, "fused_qkv", &[0usize, 1, 2][..]),
            (SLOT_FUSED_GATE_UP, "fused_gate_up", &[4usize, 5][..]),
        ] {
            let what = format!("blk.{layer}.{label}");
            let parts: Option<Vec<&Tq2Part<'_>>> =
                idx.iter().map(|&i| tq2_parts[i].as_ref()).collect();
            let Some(parts) = parts else {
                let missing: Vec<&str> = idx
                    .iter()
                    .filter(|&&i| tq2_parts[i].is_none())
                    .map(|&i| mats[i].name.as_str())
                    .collect();
                println!("P18: {what} not run — part(s) {missing:?} have no lossless TQ2 form");
                report.uncovered.push(format!(
                    "{what} — fused TQ2 shape not run: part(s) {missing:?} have no lossless TQ2 form"
                ));
                continue;
            };
            let n_rows: usize = idx.iter().map(|&i| mats[i].n_rows).sum();
            let k = mats[idx[0]].k;
            let h = handle(model_base, l, slot);
            let d_weight = expect_cuda(
                graph.get_or_upload_weight_tq2_soa_lazy_for_epoch(
                    h,
                    || {
                        let mut bytes = Vec::with_capacity(n_rows * (k / GROUP) * BLOCK_BYTES);
                        for part in &parts {
                            tq2_aos_bytes(&part.blocks, &mut bytes);
                        }
                        bytes
                    },
                    epoch,
                ),
                &format!("fused upload {what}"),
                n_rows,
                k,
            );
            uploads += 1;
            assert_eq!(
                d_weight.len(),
                n_rows * (k / GROUP) * BLOCK_BYTES,
                "P18: {what}: fused SoA buffer size mismatch (the product refuses this launch)"
            );
            drop(d_weight);
            let gpu = expect_cuda(
                graph.encode_gemv_tq2_cached(h, &x_hidden, n_rows, k),
                &format!("encode_gemv_tq2_cached {what}"),
                n_rows,
                k,
            );
            let cpu: Vec<f32> = parts
                .iter()
                .flat_map(|part| part.cpu.iter().copied())
                .collect();
            let p = report.tq2.check(&what, n_rows, k, &cpu, &gpu);
            layer_min_cos = layer_min_cos.min(p.cos);
            layer_max_abs = layer_max_abs.max(p.max_abs);
            fused_run += 1;
        }

        // ── 3. free the layer before the next one is uploaded ───────────────
        let released = graph
            .release_model_epoch(epoch)
            .unwrap_or_else(|e| panic!("P18: release_model_epoch({epoch}) for layer {layer}: {e}"));
        assert_eq!(
            released, uploads,
            "P18: layer {layer}: release_model_epoch freed {released} entries, expected \
             {uploads} (F-M3 bookkeeping)"
        );
        let shapes: Vec<String> = mats
            .iter()
            .zip(PROJECTIONS.iter())
            .map(|(m, p)| format!("{p}[{}x{}]", m.n_rows, m.k))
            .collect();
        println!(
            "P18 layer {layer:>2} ({:?}): {} + {fused_run}/2 fused shapes (qkv, gate_up) — \
             {uploads} uploads, min cos={layer_min_cos:.9}, max|Δ|={layer_max_abs:.4e}",
            mats[0].format,
            shapes.join(" ")
        );
        layer += 1;
    }
    layer
}

/// LM head through every applicable entry point. Returns `true` only when both
/// TQ2 entry points (`encode_gemv_tq2_cached` and `encode_lm_head_gemv_tq2`)
/// ran on it; every other outcome (absent, not two-bit, lossy transcode) is
/// pushed to `report.uncovered` and returns `false`.
fn check_lm_head(
    graph: &Arc<CudaGraph>,
    gguf: &GgufFile<'_>,
    model_base: u64,
    report: &mut Report,
) -> bool {
    let name = if gguf.tensors.get("output.weight").is_some() {
        "output.weight"
    } else {
        println!("P18: output.weight absent — checking the tied token_embd.weight head");
        "token_embd.weight"
    };
    match gguf.tensors.get(name).map(|info| info.tensor_type) {
        Some(GgufTensorType::TQ2_0_g128 | GgufTensorType::PQ2_0) => {}
        Some(other) => {
            println!("P18: LM head {name} is {other:?}, not two-bit — no 2-bit GEMV covers it");
            report.uncovered.push(format!(
                "LM head {name} — stored as {other:?}, not TQ2_0_g128/PQ2_0, so no 2-bit GEMV \
                 covers it"
            ));
            return false;
        }
        None => {
            println!("P18: no {name} tensor — the file has no LM head to check");
            report.uncovered.push(format!(
                "LM head — neither output.weight nor {name} is present"
            ));
            return false;
        }
    }
    let Some(m) = two_bit_matrix(gguf, name) else {
        report.uncovered.push(format!(
            "LM head {name} — not found on the second lookup (harness bug)"
        ));
        return false;
    };
    let x = activation(m.k, model_base, LM_HEAD_LAYER, 3);
    let epoch = next_cuda_model_epoch();
    let mut uploads = 0usize;
    let done = check_matrix(
        graph,
        &m,
        &x,
        handle(model_base, LM_HEAD_LAYER, 0),
        handle(model_base, LM_HEAD_LAYER, PQ2_SLOT_BASE),
        epoch,
        report,
        &mut uploads,
    );
    let released = graph
        .release_model_epoch(epoch)
        .unwrap_or_else(|e| panic!("P18: release_model_epoch({epoch}) for the LM head: {e}"));
    assert_eq!(
        released, uploads,
        "P18: LM head release freed {released} entries, expected {uploads}"
    );
    let Some(Tq2Part {
        blocks,
        cpu,
        min_cos: cos,
        max_abs: abs,
    }) = done
    else {
        // `check_matrix` already pushed the lossy-transcode reason.
        return false;
    };
    // The product's GPU LM-head entry point: `CudaGraph::encode_lm_head_gemv_tq2`,
    // which `cuda_full_layer::encode_lm_head_gemv_ternary` delegates to.
    let mut aos = Vec::with_capacity(m.raw.len());
    tq2_aos_bytes(&blocks, &mut aos);
    let h_lm = handle(model_base, LM_HEAD_LAYER, 1);
    let gpu_lm = expect_cuda(
        graph.encode_lm_head_gemv_tq2(&x, h_lm, &aos, m.n_rows, m.k),
        &format!("encode_lm_head_gemv_tq2 {}", m.name),
        m.n_rows,
        m.k,
    );
    let p_lm = report.tq2.check(
        &format!("{} (encode_lm_head_gemv_tq2)", m.name),
        m.n_rows,
        m.k,
        &cpu,
        &gpu_lm,
    );
    // `encode_lm_head_gemv_tq2` caches unattributed; free it by handle.
    let released = graph
        .release_weights(&[h_lm])
        .unwrap_or_else(|e| panic!("P18: release_weights(LM head): {e}"));
    assert_eq!(released, 1, "P18: LM-head release freed {released} entries");
    println!(
        "P18 LM head {} [{}x{}] ({:?}): cached-entry legs min cos={cos:.9} max|Δ|={abs:.4e}; \
         encode_lm_head_gemv_tq2 cos={:.9} max|Δ|={:.4e} (|ref|max={:.3})",
        m.name, m.n_rows, m.k, m.format, p_lm.cos, p_lm.max_abs, p_lm.max_ref
    );
    true
}

/// Every-layer / every-projection completeness after the walk: `blk.*`
/// tensors past the walked layers, two-bit `blk.*` tensors the walk never
/// handed to [`check_matrix`], and an `{arch}.block_count` that disagrees with
/// the walk are all pushed to `report.uncovered`.
fn check_layer_coverage(gguf: &GgufFile<'_>, file: &str, n_layers: usize, report: &mut Report) {
    let mut beyond = BTreeSet::new();
    let mut unreached = Vec::new();
    for (name, info) in gguf.tensors.iter() {
        let Some(rest) = name.strip_prefix("blk.") else {
            continue;
        };
        let layer = rest.split('.').next().and_then(|s| s.parse::<usize>().ok());
        if let Some(l) = layer.filter(|&l| l >= n_layers) {
            beyond.insert(l);
            continue;
        }
        let two_bit = matches!(
            info.tensor_type,
            GgufTensorType::TQ2_0_g128 | GgufTensorType::PQ2_0
        );
        if two_bit && !report.walked.contains(name.as_str()) {
            unreached.push(format!(
                "{name} ({:?}) — a two-bit layer tensor outside the seven projections the \
                 harness walks; no TQ2 GEMV covered it",
                info.tensor_type
            ));
        }
    }
    if !beyond.is_empty() {
        report.uncovered.push(format!(
            "layer(s) {beyond:?} — blk.* tensors exist past the walk, which stopped at the \
             missing blk.{n_layers}.attn_q.weight"
        ));
    }
    unreached.sort();
    report.uncovered.extend(unreached);
    match gguf.metadata.get_string("general.architecture") {
        Ok(arch) => match gguf.metadata.get_u32(&format!("{arch}.block_count")) {
            Ok(block_count) if usize::try_from(block_count).ok() == Some(n_layers) => println!(
                "P18: {file}: {arch}.block_count = {block_count} matches the {n_layers} walked \
                 layers"
            ),
            Ok(block_count) => report.uncovered.push(format!(
                "{arch}.block_count = {block_count}, but the walk found {n_layers} layers"
            )),
            Err(e) => println!(
                "P18: {file}: no usable {arch}.block_count ({e}) — layer coverage rests on \
                 the blk.* tensor scan"
            ),
        },
        Err(e) => println!(
            "P18: {file}: no general.architecture ({e}) — layer coverage rests on the blk.* \
             tensor scan"
        ),
    }
}

/// The whole P18 matrix for one model file.
fn run_p18(file: &str, model_base: u64, test: &str) {
    let _serial = gpu_serial();
    let graph = match CudaGraph::global() {
        Ok(g) => g,
        Err(e) => {
            println!("skip: {test} — no CUDA device accessible ({e})");
            record_skipped(Capability::CudaHardware, test);
            return;
        }
    };
    let Some(path) = find_model(file) else {
        println!("skip: {test} — {file} not found under models/ (or $OXIBONSAI_MODELS_DIR)");
        record_skipped(Capability::CudaHardware, test);
        return;
    };
    let start = Instant::now();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("P18: read {path:?}: {e}"));
    let gguf = GgufFile::parse(&bytes).unwrap_or_else(|e| panic!("P18: parse {path:?}: {e}"));

    let mut report = Report {
        tq2: Summary::new("TQ2 leg: encode_gemv_tq2_cached / encode_lm_head_gemv_tq2"),
        pq2: Summary::new("PQ2 leg: encode_gemv_pq2_cached"),
        transcode_min_cos: f64::INFINITY,
        lossy: Vec::new(),
        walked: HashSet::new(),
        uncovered: Vec::new(),
    };
    let n_layers = check_layers(&graph, &gguf, model_base, &mut report);
    assert!(
        n_layers > 0,
        "P18: {file} has no blk.0.attn_q.weight — not a ternary transformer GGUF"
    );
    let lm_head_checked = check_lm_head(&graph, &gguf, model_base, &mut report);
    check_layer_coverage(&gguf, file, n_layers, &mut report);

    println!(
        "P18 SUMMARY {file}: {n_layers} layers, LM head {}, lossy-transcode tensors: {:?}, \
         uncovered shapes: {}",
        if lm_head_checked {
            "checked (both TQ2 entry points)"
        } else {
            "NOT covered"
        },
        report.lossy,
        report.uncovered.len()
    );
    if report.pq2.gemvs > 0 {
        println!(
            "P18 SUMMARY {file}: stored as PQ2_0 — TQ2 leg ran on the lossless PQ2→TQ2 \
             transcode; worst CPU PQ2 vs CPU TQ2 cos={:.12}",
            report.transcode_min_cos
        );
    }
    report.tq2.report(file);
    report.pq2.report(file);
    assert!(
        report.uncovered.is_empty(),
        "P18 FAIL: {file}: {} shape(s) were not covered by the TQ2 leg \
         (encode_gemv_tq2_cached / encode_lm_head_gemv_tq2). CUDA-P18 asks for every \
         projection of every layer and the LM head, so this run is not evidence for it and \
         no capability is recorded:\n  - {}",
        report.uncovered.len(),
        report.uncovered.join("\n  - ")
    );
    assert!(
        lm_head_checked,
        "P18 FAIL: {file}: the LM head's TQ2 legs did not both run (coverage bookkeeping bug)"
    );
    // Per layer: 7 projections + 2 fused shapes; LM head: 2 entry points.
    let expected_tq2 = n_layers * (PROJECTIONS.len() + 2) + 2;
    assert_eq!(
        report.tq2.gemvs, expected_tq2,
        "P18 FAIL: {file}: {} TQ2 GEMVs passed, expected exactly {expected_tq2} ({n_layers} \
         layers × (7 projections + 2 fused shapes) + 2 LM-head entry points)",
        report.tq2.gemvs
    );
    record_executed_timed(Capability::CudaHardware, test, start.elapsed());
}

/// CUDA-P18 on `Ternary-Bonsai-1.7B.gguf` (28 layers, hidden 2048).
#[test]
fn p18_tq2_gemv_real_shapes_ternary_bonsai_1_7b() {
    run_p18("Ternary-Bonsai-1.7B.gguf", 0x7018_0001_0000_0000, TEST_1_7B);
}

/// CUDA-P18 on `Ternary-Bonsai-8B.gguf` (36 layers, hidden 4096).
#[test]
fn p18_tq2_gemv_real_shapes_ternary_bonsai_8b() {
    run_p18("Ternary-Bonsai-8B.gguf", 0x7018_0002_0000_0000, TEST_8B);
}
