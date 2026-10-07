//! CUDA Q4_0 / Q8_0 GEMV and batch-prefill GEMM numerics on **real** weights.
//!
//! # Why
//!
//! 0.2.4 on an RTX A4000 (sm_86): `models/tb17h-q4_0.gguf` (every weight
//! matrix **and** `output.weight` in Q4_0, made by
//! `oxibonsai-model/examples/quantize_full_head.rs`) diverged from the CPU
//! binary at the **first** generated token, then decoded fluently and on
//! topic, while `tb17h-q8_0.gguf` through the same Q-std batch prefill and
//! K/V read-back was byte-identical. The in-crate CUDA tests
//! (`cuda_q_std_kernels.rs` `test_cuda_gemv_q{4,8}_0_matches_cpu`,
//! `cuda_q_std_prefill.rs` batch=12) use constant or all-zero Q4_0 blocks
//! (nibble 8 everywhere), on which any nibble permutation is invisible, so no
//! Q4_0 CUDA kernel had ever been checked against the CPU on real data.
//!
//! Root cause (fixed in the preceding commit): `gemv_q4_0_pf`
//! (`cuda_q_std_prefill_kernels.rs`), the kernel `try_cuda_prefill_q_std`
//! launches for the **LM head of the last prompt token** when the head is
//! Q4_0, paired byte `nb`'s nibbles with `x[2nb]` / `x[2nb+1]` (even/odd)
//! instead of ggml's lo-hi split `x[nb]` / `x[nb+16]` that the CPU
//! (`BlockQ4_0::dequant_to_buf`), the decode kernel `gemv_q4_0` and the batch
//! GEMMs all use. Only the first token comes from that kernel — decode runs
//! `LinearQ4_0::forward` → `cuda_gemv_q4_0` — which is exactly the observed
//! signature; the Q8_0 twin `gemv_q8_0_pf` has no nibbles.
//!
//! # What runs
//!
//! For each of `tb17h-q4_0.gguf` / `tb17h-q8_0.gguf` (found by
//! `oxibonsai_testkit::workspace::find_model`, i.e. `models/` or
//! `$OXIBONSAI_MODELS_DIR`), over the seven projections of the first
//! [`LAYERS`] layers and the LM head (`output.weight`):
//!
//! - **GEMV test** (`q_std_gemv_real_weights_*`): on one seeded activation
//!   per matrix, the decode entry point the product uses
//!   (`cuda_gemv_q4_0` / `cuda_gemv_q8_0`, called by
//!   `LinearQ{4,8}_0::forward`) **and** the batch-prefill LM-head kernel
//!   (`cuda_prefill_gemv_q_std` → `gemv_q4_0_pf` / `gemv_q8_0_pf`) against
//!   the production CPU GEMV (`gemv_q4_0` / `gemv_q8_0`) on the same blocks.
//!   For Q4_0 each GPU result is also scored against a CPU reading of the
//!   blocks with the **even/odd** nibble interleave, so a failure says at a
//!   glance whether it is that permutation (cos≈1 against it) or something
//!   else.
//! - **GEMM test** (`q_std_prefill_gemm_real_weights_*`): a [`BATCH`]-token
//!   batch (three 8-column chunks: 8 + 8 + 4) through the batch-prefill
//!   kernels the product runs — `gemm_q*` (`cuda_prefill_gemm_q_std`) on the
//!   fused `Q‖K‖V` concatenation, `attn_output`, `ffn_down` and the LM head,
//!   and `fused_gate_up_swiglu_gemm_q*`
//!   (`cuda_prefill_gate_up_swiglu_q_std`) on `gate‖up` — against the CPU
//!   GEMV per token (SwiGLU applied on the CPU side). Each token column is
//!   scored on its own, so a broken column chunk cannot hide in the average.
//!
//! Every comparison is printed (cos, max|Δ|, worst row) before anything is
//! asserted, so one hardware run yields the whole table; the test then fails
//! if any `cos < 0.999`.
//!
//! Each test self-skips (recording `cuda-hardware`, `executed: false`) when no
//! CUDA device answers `CudaGraph::global()` or its GGUF is absent;
//! `executed: true` is recorded only after every comparison passed.
//!
//! ```text
//! cargo test --release -p oxibonsai-kernels --features native-cuda \
//!     --test cuda_q_std_gemv_real_weights -- --test-threads=1 --nocapture
//! ```

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use std::borrow::Cow;
use std::time::Instant;

use half::f16;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::{BlockQ4_0, BlockQ8_0, GgufTensorType};
use oxibonsai_kernels::gemv_q4_0::gemv_q4_0;
use oxibonsai_kernels::gemv_q8_0::gemv_q8_0;
use oxibonsai_kernels::{
    cuda_gemv_q4_0, cuda_gemv_q8_0, cuda_prefill_gate_up_swiglu_q_std, cuda_prefill_gemm_q_std,
    cuda_prefill_gemv_q_std, CudaGraph, CudaGraphError,
};
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::gguf_fixture::deterministic_weights;
use oxibonsai_testkit::parity::gpu_serial;
use oxibonsai_testkit::workspace::find_model;

/// Acceptance floor per comparison.
const COS_FLOOR: f64 = 0.999;
/// Layers whose projections are checked.
const LAYERS: usize = 2;
/// Tokens in the batch-GEMM check (crosses the kernels' 8-column chunk twice).
const BATCH: usize = 20;
/// Weights per Q4_0 / Q8_0 block.
const QK: usize = 32;
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
/// The LM head tensor (the fixtures quantize it; it is never tied there).
const LM_HEAD: &str = "output.weight";

/// Q4_0 fixture file name.
const FILE_Q4_0: &str = "tb17h-q4_0.gguf";
/// Q8_0 fixture file name.
const FILE_Q8_0: &str = "tb17h-q8_0.gguf";

/// Capability-record name of [`q_std_gemv_real_weights_tb17h_q4_0`].
const TEST_GEMV_Q4_0: &str =
    "oxibonsai-kernels::cuda_q_std_gemv_real_weights::q_std_gemv_real_weights_tb17h_q4_0";
/// Capability-record name of [`q_std_gemv_real_weights_tb17h_q8_0`].
const TEST_GEMV_Q8_0: &str =
    "oxibonsai-kernels::cuda_q_std_gemv_real_weights::q_std_gemv_real_weights_tb17h_q8_0";
/// Capability-record name of [`q_std_prefill_gemm_real_weights_tb17h_q4_0`].
const TEST_GEMM_Q4_0: &str = "oxibonsai-kernels::cuda_q_std_gemv_real_weights::\
                              q_std_prefill_gemm_real_weights_tb17h_q4_0";
/// Capability-record name of [`q_std_prefill_gemm_real_weights_tb17h_q8_0`].
const TEST_GEMM_Q8_0: &str = "oxibonsai-kernels::cuda_q_std_gemv_real_weights::\
                              q_std_prefill_gemm_real_weights_tb17h_q8_0";

/// The two standard block formats this harness covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QStd {
    Q4_0,
    Q8_0,
}

impl QStd {
    fn label(self) -> &'static str {
        match self {
            QStd::Q4_0 => "Q4_0",
            QStd::Q8_0 => "Q8_0",
        }
    }

    /// AoS bytes per 32-weight block.
    fn block_bytes(self) -> usize {
        match self {
            QStd::Q4_0 => 18,
            QStd::Q8_0 => 34,
        }
    }

    fn tensor_type(self) -> GgufTensorType {
        match self {
            QStd::Q4_0 => GgufTensorType::Q4_0,
            QStd::Q8_0 => GgufTensorType::Q8_0,
        }
    }

    fn is_q4_0(self) -> bool {
        self == QStd::Q4_0
    }

    /// Distinct seed salt per format.
    fn salt(self) -> u64 {
        match self {
            QStd::Q4_0 => 0x4040,
            QStd::Q8_0 => 0x8080,
        }
    }
}

/// One real weight matrix out of the GGUF.
struct Matrix<'a> {
    name: String,
    /// Raw AoS bytes as stored in the file.
    raw: &'a [u8],
    /// Output rows (GGUF `ne1`).
    n_rows: usize,
    /// Input width (GGUF `ne0`).
    k: usize,
}

/// Look up `name` as a `quant` matrix with a consistent size. `None` means the
/// tensor is absent; a present tensor of another type fails the test (the
/// `tb17h-*` fixtures quantize every matrix, the LM head included).
fn matrix<'a>(gguf: &GgufFile<'a>, file: &str, name: &str, quant: QStd) -> Option<Matrix<'a>> {
    let info = gguf.tensors.get(name)?;
    assert_eq!(
        info.tensor_type,
        quant.tensor_type(),
        "{file}: {name} is {:?}, expected {} — regenerate the fixture with \
         `cargo run --release -p oxibonsai-model --example quantize_full_head`",
        info.tensor_type,
        quant.label()
    );
    assert!(
        info.shape.len() >= 2,
        "{file}: {name} has shape {:?}, expected [k, n_rows]",
        info.shape
    );
    let k = usize::try_from(info.shape[0]).expect("k fits usize");
    let n_rows = usize::try_from(info.shape[1]).expect("n_rows fits usize");
    assert!(
        k > 0 && k.is_multiple_of(QK),
        "{file}: {name} has k={k}, not a positive multiple of {QK}"
    );
    let raw = gguf
        .tensor_data(name)
        .unwrap_or_else(|e| panic!("{file}: tensor_data({name}): {e}"));
    assert_eq!(
        raw.len(),
        n_rows * (k / QK) * quant.block_bytes(),
        "{file}: {name} byte length does not match its shape [k={k}, n_rows={n_rows}]"
    );
    Some(Matrix {
        name: name.to_string(),
        raw,
        n_rows,
        k,
    })
}

/// The layer tensor `blk.{layer}.{proj}.weight`, which must exist.
fn layer_matrix<'a>(
    gguf: &GgufFile<'a>,
    file: &str,
    layer: usize,
    proj: &str,
    quant: QStd,
) -> Matrix<'a> {
    let name = format!("blk.{layer}.{proj}.weight");
    matrix(gguf, file, &name, quant)
        .unwrap_or_else(|| panic!("{file}: {name} is missing — not a dense transformer GGUF"))
}

/// The LM head, which must exist and be quantized (the point of the fixture).
fn lm_head<'a>(gguf: &GgufFile<'a>, file: &str, quant: QStd) -> Matrix<'a> {
    matrix(gguf, file, LM_HEAD, quant).unwrap_or_else(|| {
        panic!(
            "{file}: no {LM_HEAD} — the LM head is tied; this harness needs the \
             quantized-head fixture from quantize_full_head"
        )
    })
}

/// `raw` as Q4_0 blocks: zero-copy when aligned, decoded otherwise.
fn q4_0_blocks(raw: &[u8]) -> Cow<'_, [BlockQ4_0]> {
    match BlockQ4_0::slice_from_bytes(raw) {
        Ok(blocks) => Cow::Borrowed(blocks),
        Err(_) => Cow::Owned(
            raw.as_chunks::<18>()
                .0
                .iter()
                .map(|c| {
                    let mut qs = [0u8; 16];
                    qs.copy_from_slice(&c[2..]);
                    BlockQ4_0 {
                        d: f16::from_le_bytes([c[0], c[1]]),
                        qs,
                    }
                })
                .collect(),
        ),
    }
}

/// `raw` as Q8_0 blocks: zero-copy when aligned, decoded otherwise.
fn q8_0_blocks(raw: &[u8]) -> Cow<'_, [BlockQ8_0]> {
    match BlockQ8_0::slice_from_bytes(raw) {
        Ok(blocks) => Cow::Borrowed(blocks),
        Err(_) => Cow::Owned(
            raw.as_chunks::<34>()
                .0
                .iter()
                .map(|c| {
                    let mut qs = [0i8; 32];
                    for (q, &b) in qs.iter_mut().zip(&c[2..]) {
                        *q = i8::from_le_bytes([b]);
                    }
                    BlockQ8_0 {
                        d: f16::from_le_bytes([c[0], c[1]]),
                        qs,
                    }
                })
                .collect(),
        ),
    }
}

/// The CPU oracle: the production CPU GEMV (`gemv_q4_0` / `gemv_q8_0`) on the
/// same blocks.
fn cpu_gemv(m: &Matrix<'_>, quant: QStd, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m.n_rows];
    match quant {
        QStd::Q4_0 => gemv_q4_0(&q4_0_blocks(m.raw), input, &mut out, m.n_rows, m.k),
        QStd::Q8_0 => gemv_q8_0(&q8_0_blocks(m.raw), input, &mut out, m.n_rows, m.k),
    }
    .unwrap_or_else(|e| panic!("CPU {} GEMV on {}: {e}", quant.label(), m.name));
    out
}

/// A deliberately **wrong** CPU reading of Q4_0 blocks: byte `nb`'s low / high
/// nibble paired with `x[2nb]` / `x[2nb+1]` (the even/odd interleave the
/// pre-fix `gemv_q4_0_pf` used). Diagnostic only: a GPU result that matches
/// this instead of [`cpu_gemv`] is that permutation bug.
fn cpu_gemv_q4_0_even_odd(m: &Matrix<'_>, input: &[f32]) -> Vec<f32> {
    let blocks_per_row = m.k / QK;
    let x_blocks = input[..m.k].as_chunks::<QK>().0;
    m.raw
        .chunks_exact(blocks_per_row * 18)
        .map(|row| {
            row.as_chunks::<18>()
                .0
                .iter()
                .zip(x_blocks)
                .map(|(block, x)| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let sum: f32 = block[2..]
                        .iter()
                        .zip(x.as_chunks::<2>().0)
                        .map(|(&byte, pair)| {
                            let lo = f32::from(byte & 0x0F) - 8.0;
                            let hi = f32::from(byte >> 4) - 8.0;
                            lo * pair[0] + hi * pair[1]
                        })
                        .sum();
                    d * sum
                })
                .sum()
        })
        .collect()
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
        "{what}: reference has {} rows, result {}",
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
        assert!(g.is_finite(), "{what}: row {row} is not finite ({g})");
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

/// Running tally of one test: every comparison is printed as it happens and
/// failures are collected, so a hardware run reports the whole table before
/// the final assertion.
struct Tally {
    file: &'static str,
    checks: usize,
    min_cos: f64,
    min_cos_at: String,
    max_abs: f32,
    max_abs_at: String,
    failures: Vec<String>,
}

impl Tally {
    fn new(file: &'static str) -> Self {
        Self {
            file,
            checks: 0,
            min_cos: f64::INFINITY,
            min_cos_at: String::new(),
            max_abs: 0.0,
            max_abs_at: String::new(),
            failures: Vec::new(),
        }
    }

    /// Score `gpu` against `cpu`, print the line, remember a failure.
    fn check(&mut self, what: &str, cpu: &[f32], gpu: &[f32]) -> Parity {
        self.score(what, cpu, gpu, true)
    }

    /// As [`Self::check`]; `print` = false prints only a failing comparison.
    fn score(&mut self, what: &str, cpu: &[f32], gpu: &[f32], print: bool) -> Parity {
        let p = compare(cpu, gpu, what);
        let pass = p.cos >= COS_FLOOR;
        if print || !pass {
            println!(
                "{} {} {what}: cos={:.9} max|Δ|={:.4e} (max|ref|={:.4e}) worst row {} \
                 cpu={:.6e} cuda={:.6e}",
                if pass { "ok  " } else { "FAIL" },
                self.file,
                p.cos,
                p.max_abs,
                p.max_ref,
                p.worst_row,
                cpu.get(p.worst_row).copied().unwrap_or(f32::NAN),
                gpu.get(p.worst_row).copied().unwrap_or(f32::NAN)
            );
        }
        self.checks += 1;
        if p.cos < self.min_cos {
            self.min_cos = p.cos;
            self.min_cos_at = what.to_string();
        }
        if p.max_abs > self.max_abs {
            self.max_abs = p.max_abs;
            self.max_abs_at = what.to_string();
        }
        if !pass {
            self.failures
                .push(format!("{what}: cos={:.9} < {COS_FLOOR}", p.cos));
        }
        p
    }

    /// Print the summary and fail on any collected failure.
    fn finish(self, test: &str) {
        println!(
            "SUMMARY {} [{test}]: {} comparisons, min cos={:.9} at {}, max|Δ|={:.4e} at {}",
            self.file, self.checks, self.min_cos, self.min_cos_at, self.max_abs, self.max_abs_at
        );
        assert!(
            self.checks > 0,
            "{test}: no comparison ran on {}",
            self.file
        );
        assert!(
            self.failures.is_empty(),
            "{test}: {} of {} comparisons on {} fell below cos {COS_FLOOR}:\n  {}",
            self.failures.len(),
            self.checks,
            self.file,
            self.failures.join("\n  ")
        );
    }
}

/// Unwrap a CUDA result with the call site named.
fn expect_cuda(r: Result<(), CudaGraphError>, what: &str) {
    if let Err(e) = r {
        panic!("{what}: CUDA call failed: {e}");
    }
}

/// Seeded activation of `len` elements for (format, layer, role).
fn activation(len: usize, quant: QStd, layer: usize, role: usize) -> Vec<f32> {
    let seed = (quant.salt() << 32) ^ ((layer as u64) << 12) ^ ((role as u64) << 4) ^ 0x0A57;
    deterministic_weights(len, seed)
}

/// The fixture common to every test: the CUDA device, then the GGUF bytes.
/// `None` (after recording the skip) when either is unavailable.
fn open_fixture(file: &str, test: &str) -> Option<Vec<u8>> {
    if let Err(e) = CudaGraph::global() {
        println!("skip: {test} — no CUDA device accessible ({e})");
        record_skipped(Capability::CudaHardware, test);
        return None;
    }
    let Some(path) = find_model(file) else {
        println!("skip: {test} — {file} not found under models/ (or $OXIBONSAI_MODELS_DIR)");
        record_skipped(Capability::CudaHardware, test);
        return None;
    };
    Some(std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// GEMV leg: decode kernel + batch-prefill LM-head kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Run one matrix through the decode GEMV and the prefill LM-head GEMV.
fn gemv_matrix(m: &Matrix<'_>, quant: QStd, layer: usize, role: usize, tally: &mut Tally) {
    let input = activation(m.k, quant, layer, role);
    let started = Instant::now();
    let cpu = cpu_gemv(m, quant, &input);
    let cpu_elapsed = started.elapsed();

    let mut decode = vec![0.0f32; m.n_rows];
    let decode_call = match quant {
        QStd::Q4_0 => cuda_gemv_q4_0(m.raw, &input, &mut decode, m.n_rows, m.k),
        QStd::Q8_0 => cuda_gemv_q8_0(m.raw, &input, &mut decode, m.n_rows, m.k),
    };
    let decode_name = format!(
        "{} decode cuda_gemv_{} [{}x{}]",
        m.name,
        quant.label().to_ascii_lowercase(),
        m.n_rows,
        m.k
    );
    expect_cuda(decode_call, &decode_name);
    tally.check(&decode_name, &cpu, &decode);

    let mut prefill = vec![0.0f32; m.n_rows];
    let pf_name = format!(
        "{} prefill-LM-head gemv_{}_pf [{}x{}]",
        m.name,
        quant.label().to_ascii_lowercase(),
        m.n_rows,
        m.k
    );
    expect_cuda(
        cuda_prefill_gemv_q_std(m.raw, &input, &mut prefill, m.n_rows, m.k, quant.is_q4_0()),
        &pf_name,
    );
    tally.check(&pf_name, &cpu, &prefill);

    if quant.is_q4_0() {
        // Diagnostic: which nibble pairing does each GPU kernel follow?
        let even_odd = cpu_gemv_q4_0_even_odd(m, &input);
        let cpu_vs_eo = compare(&cpu, &even_odd, "cpu vs even/odd").cos;
        let dec_vs_eo = compare(&even_odd, &decode, "decode vs even/odd").cos;
        let pf_vs_eo = compare(&even_odd, &prefill, "prefill vs even/odd").cos;
        println!(
            "     nibble-order probe {}: cos(cpu lo-hi, even/odd)={cpu_vs_eo:.6} \
             cos(decode, even/odd)={dec_vs_eo:.6} cos(prefill-LM-head, even/odd)={pf_vs_eo:.6} \
             (≈1 against even/odd = the interleave bug)",
            m.name
        );
    }
    println!(
        "     {} CPU {} GEMV took {:.1} ms",
        m.name,
        quant.label(),
        cpu_elapsed.as_secs_f64() * 1000.0
    );
}

/// The GEMV test for one fixture.
fn run_gemv(file: &'static str, quant: QStd, test: &str) {
    let _serial = gpu_serial();
    let Some(bytes) = open_fixture(file, test) else {
        return;
    };
    let start = Instant::now();
    let gguf = GgufFile::parse(&bytes).unwrap_or_else(|e| panic!("parse {file}: {e}"));
    let mut tally = Tally::new(file);

    for layer in 0..LAYERS {
        for (role, proj) in PROJECTIONS.iter().enumerate() {
            let m = layer_matrix(&gguf, file, layer, proj, quant);
            gemv_matrix(&m, quant, layer, role, &mut tally);
        }
    }
    let head = lm_head(&gguf, file, quant);
    gemv_matrix(&head, quant, LAYERS, PROJECTIONS.len(), &mut tally);

    tally.finish(test);
    record_executed_timed(Capability::CudaHardware, test, start.elapsed());
}

// ─────────────────────────────────────────────────────────────────────────────
// GEMM leg: batch-prefill kernels over a BATCH-token batch
// ─────────────────────────────────────────────────────────────────────────────

/// CPU per-token reference of a row-concatenation of `parts` (all sharing
/// `k`): token-major `[BATCH × Σ n_rows]`.
fn cpu_batch(parts: &[&Matrix<'_>], quant: QStd, inputs: &[f32], k: usize) -> Vec<f32> {
    let n_rows: usize = parts.iter().map(|m| m.n_rows).sum();
    let mut out = Vec::with_capacity(BATCH * n_rows);
    for x in inputs.chunks_exact(k).take(BATCH) {
        for m in parts {
            out.extend_from_slice(&cpu_gemv(m, quant, x));
        }
    }
    out
}

/// Score a token-major `[BATCH × n_rows]` result one token column at a time
/// (a failing column is printed on its own), then print the worst column and
/// the whole-batch figures.
fn check_columns(tally: &mut Tally, what: &str, n_rows: usize, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), BATCH * n_rows, "{what}: CPU batch size");
    assert_eq!(gpu.len(), BATCH * n_rows, "{what}: CUDA batch size");
    let mut worst: Option<(usize, Parity)> = None;
    for (t, (c, g)) in cpu
        .chunks_exact(n_rows)
        .zip(gpu.chunks_exact(n_rows))
        .enumerate()
    {
        let p = tally.score(&format!("{what} token {t}/{BATCH}"), c, g, false);
        if worst.is_none_or(|(_, w)| p.cos < w.cos) {
            worst = Some((t, p));
        }
    }
    let whole = compare(cpu, gpu, what);
    if let Some((t, w)) = worst {
        println!(
            "     {} {what}: {BATCH} token columns, worst token {t} cos={:.9} \
             max|Δ|={:.4e}; whole batch cos={:.9} max|Δ|={:.4e} (max|ref|={:.4e})",
            tally.file, w.cos, w.max_abs, whole.cos, whole.max_abs, whole.max_ref
        );
    }
}

/// `gemm_q*` over the row-concatenation of `parts` (the fused-QKV layout the
/// product uploads when `parts` has three entries).
fn gemm_parts(parts: &[&Matrix<'_>], quant: QStd, layer: usize, role: usize, tally: &mut Tally) {
    let k = parts[0].k;
    assert!(
        parts.iter().all(|m| m.k == k),
        "concatenated parts must share k"
    );
    let n_rows: usize = parts.iter().map(|m| m.n_rows).sum();
    let mut weight = Vec::with_capacity(parts.iter().map(|m| m.raw.len()).sum());
    for m in parts {
        weight.extend_from_slice(m.raw);
    }
    let names: Vec<&str> = parts.iter().map(|m| m.name.as_str()).collect();
    let what = format!(
        "{} gemm_{} [{}x{k}, batch {BATCH}]",
        names.join("‖"),
        quant.label().to_ascii_lowercase(),
        n_rows
    );

    let inputs = activation(BATCH * k, quant, layer, 0x100 + role);
    let cpu = cpu_batch(parts, quant, &inputs, k);
    let mut gpu = vec![0.0f32; BATCH * n_rows];
    expect_cuda(
        cuda_prefill_gemm_q_std(
            &weight,
            &inputs,
            &mut gpu,
            n_rows,
            k,
            BATCH,
            quant.is_q4_0(),
        ),
        &what,
    );
    check_columns(tally, &what, n_rows, &cpu, &gpu);
}

/// `fused_gate_up_swiglu_gemm_q*` over `gate‖up` against CPU SwiGLU.
fn gate_up_swiglu(
    gate: &Matrix<'_>,
    up: &Matrix<'_>,
    quant: QStd,
    layer: usize,
    tally: &mut Tally,
) {
    assert_eq!(
        (gate.n_rows, gate.k),
        (up.n_rows, up.k),
        "gate / up shapes differ"
    );
    let (n_rows, k) = (gate.n_rows, gate.k);
    let what = format!(
        "{}‖{} fused_gate_up_swiglu_gemm_{} [{n_rows}x{k}, batch {BATCH}]",
        gate.name,
        up.name,
        quant.label().to_ascii_lowercase()
    );
    let inputs = activation(BATCH * k, quant, layer, 0x200);
    let mut cpu = Vec::with_capacity(BATCH * n_rows);
    for x in inputs.chunks_exact(k).take(BATCH) {
        let g = cpu_gemv(gate, quant, x);
        let u = cpu_gemv(up, quant, x);
        cpu.extend(g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u));
    }
    let mut gpu = vec![0.0f32; BATCH * n_rows];
    expect_cuda(
        cuda_prefill_gate_up_swiglu_q_std(
            gate.raw,
            up.raw,
            &inputs,
            &mut gpu,
            n_rows,
            k,
            BATCH,
            quant.is_q4_0(),
        ),
        &what,
    );
    check_columns(tally, &what, n_rows, &cpu, &gpu);
}

/// The batch-GEMM test for one fixture.
fn run_gemm(file: &'static str, quant: QStd, test: &str) {
    let _serial = gpu_serial();
    let Some(bytes) = open_fixture(file, test) else {
        return;
    };
    let start = Instant::now();
    let gguf = GgufFile::parse(&bytes).unwrap_or_else(|e| panic!("parse {file}: {e}"));
    let mut tally = Tally::new(file);

    for layer in 0..LAYERS {
        let get = |proj: &str| layer_matrix(&gguf, file, layer, proj, quant);
        let (q, k, v) = (get("attn_q"), get("attn_k"), get("attn_v"));
        gemm_parts(&[&q, &k, &v], quant, layer, 0, &mut tally);
        gemm_parts(&[&get("attn_output")], quant, layer, 1, &mut tally);
        gemm_parts(&[&get("ffn_down")], quant, layer, 2, &mut tally);
        gate_up_swiglu(&get("ffn_gate"), &get("ffn_up"), quant, layer, &mut tally);
    }
    // Not a product GEMM, but the largest row count the kernel can be handed.
    let head = lm_head(&gguf, file, quant);
    gemm_parts(&[&head], quant, LAYERS, 3, &mut tally);

    tally.finish(test);
    record_executed_timed(Capability::CudaHardware, test, start.elapsed());
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// Q4_0 decode GEMV + batch-prefill LM-head GEMV on `tb17h-q4_0.gguf` (the
/// regression test of the `gemv_q4_0_pf` nibble-order fix).
#[test]
fn q_std_gemv_real_weights_tb17h_q4_0() {
    run_gemv(FILE_Q4_0, QStd::Q4_0, TEST_GEMV_Q4_0);
}

/// Q8_0 decode GEMV + batch-prefill LM-head GEMV on `tb17h-q8_0.gguf`.
#[test]
fn q_std_gemv_real_weights_tb17h_q8_0() {
    run_gemv(FILE_Q8_0, QStd::Q8_0, TEST_GEMV_Q8_0);
}

/// Q4_0 batch-prefill GEMMs (fused QKV, o, down, gate‖up SwiGLU, LM head) on
/// `tb17h-q4_0.gguf`, 20 tokens.
#[test]
fn q_std_prefill_gemm_real_weights_tb17h_q4_0() {
    run_gemm(FILE_Q4_0, QStd::Q4_0, TEST_GEMM_Q4_0);
}

/// Q8_0 batch-prefill GEMMs on `tb17h-q8_0.gguf`, 20 tokens.
#[test]
fn q_std_prefill_gemm_real_weights_tb17h_q8_0() {
    run_gemm(FILE_Q8_0, QStd::Q8_0, TEST_GEMM_Q8_0);
}
