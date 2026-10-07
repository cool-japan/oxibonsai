//! Kernel-level entry points for the Q4_0 / Q8_0 batch-prefill kernels.
//!
//! [`try_cuda_prefill_q_std`](super::cuda_q_std_prefill::try_cuda_prefill_q_std)
//! only exposes whole-model results (last-token logits, K/V read-back), so a
//! wrong kernel shows up as a wrong token with no way to tell which GEMM
//! produced it. These wrappers run one prefill kernel on host data — upload,
//! launch, synchronise, read back — so each can be compared with the CPU GEMV
//! on real weights (`tests/cuda_q_std_gemv_real_weights.rs`):
//!
//! - [`cuda_prefill_gemv_q_std`] runs `gemv_q4_0_pf` / `gemv_q8_0_pf`, the
//!   last-token LM-head GEMV;
//! - [`cuda_prefill_gemm_q_std`] runs `gemm_q4_0` / `gemm_q8_0`, the batch
//!   GEMM behind the fused `Q‖K‖V`, attention-output and down projections;
//! - [`cuda_prefill_gate_up_swiglu_q_std`] runs
//!   `fused_gate_up_swiglu_gemm_q4_0` / `_q8_0`.
//!
//! Numerics probes, not hot paths: every call uploads its weight.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::CudaSlice;

use super::cuda_graph::{CudaGraph, CudaGraphError};
use super::cuda_q_std_prefill::{
    init_q_std_prefill_modules, launch_fused_gate_up_swiglu_q4_0, launch_fused_gate_up_swiglu_q8_0,
    launch_gemm_q4_0, launch_gemm_q8_0, launch_gemv_q4_0_pf, launch_gemv_q8_0_pf,
};

/// Bytes per AoS block of the Q std format: 18 for Q4_0, 34 for Q8_0.
fn q_std_block_bytes(q4_0: bool) -> usize {
    if q4_0 {
        18
    } else {
        34
    }
}

/// Validate one kernel-level probe call and return the weight byte count
/// (`n_rows * k / 32 * block_bytes`) plus `(n_rows, k, batch_size)` as `u32`.
fn check_q_std_probe(
    what: &str,
    blocks_len: usize,
    n_rows: usize,
    k: usize,
    batch_size: usize,
    q4_0: bool,
) -> Result<(usize, u32, u32, u32), CudaGraphError> {
    if k == 0 || !k.is_multiple_of(32) {
        return Err(CudaGraphError::InvalidDimensions(format!(
            "{what}: k={k} must be a positive multiple of 32"
        )));
    }
    if n_rows == 0 || batch_size == 0 {
        return Err(CudaGraphError::InvalidDimensions(format!(
            "{what}: n_rows={n_rows} and batch_size={batch_size} must be non-zero"
        )));
    }
    let to_u32 = |v: usize, name: &str| {
        u32::try_from(v).map_err(|_| {
            CudaGraphError::InvalidDimensions(format!("{what}: {name}={v} exceeds u32"))
        })
    };
    let (n_rows_u32, k_u32, batch_u32) = (
        to_u32(n_rows, "n_rows")?,
        to_u32(k, "k")?,
        to_u32(batch_size, "batch_size")?,
    );
    // The kernels index blocks with 32-bit `row * blocks_per_row + b`.
    let n_blocks = n_rows
        .checked_mul(k / 32)
        .filter(|&b| u32::try_from(b).is_ok())
        .ok_or_else(|| {
            CudaGraphError::InvalidDimensions(format!(
                "{what}: n_rows={n_rows} x k/32={} blocks overflow the kernels' u32 index",
                k / 32
            ))
        })?;
    let expected = n_blocks * q_std_block_bytes(q4_0);
    if blocks_len < expected {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{what}: weight bytes too short: {blocks_len} < {expected} \
             (n_rows={n_rows}, k={k}, {})",
            if q4_0 { "Q4_0" } else { "Q8_0" }
        )));
    }
    Ok((expected, n_rows_u32, k_u32, batch_u32))
}

/// Run the batch-prefill **LM-head** GEMV kernel on host data: `gemv_q4_0_pf`
/// (`q4_0`) or `gemv_q8_0_pf`, the kernels [`try_cuda_prefill_q_std`]
/// launches on the last prompt token's normed hidden state.
///
/// `blocks_bytes` is the raw GGUF AoS weight (`n_rows * k / 32` blocks of 18
/// or 34 bytes), `input` has `>= k` elements and `output` receives `n_rows`
/// results (written, not accumulated). Uploads, launches and synchronises
/// per call — a numerics probe for real weights, not a hot path.
///
/// # Errors
/// [`CudaGraphError::InvalidDimensions`] for a bad geometry,
/// [`CudaGraphError::WeightLayoutError`] for short buffers, and any device
/// error from the upload, launch or read-back.
///
/// [`try_cuda_prefill_q_std`]: super::cuda_q_std_prefill::try_cuda_prefill_q_std
pub fn cuda_prefill_gemv_q_std(
    blocks_bytes: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    q4_0: bool,
) -> Result<(), CudaGraphError> {
    let what = "cuda_prefill_gemv_q_std";
    let (w_bytes, n_rows_u32, k_u32, _) =
        check_q_std_probe(what, blocks_bytes.len(), n_rows, k, 1, q4_0)?;
    if input.len() < k || output.len() < n_rows {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{what}: input.len()={} < k={k} or output.len()={} < n_rows={n_rows}",
            input.len(),
            output.len()
        )));
    }
    let graph = CudaGraph::global()?;
    let mods = init_q_std_prefill_modules(&graph)?;
    let stream = graph.stream_arc();
    let d_blocks = stream
        .clone_htod(&blocks_bytes[..w_bytes])
        .map_err(|e| CudaGraphError::DriverError(format!("{what} upload weight: {e}")))?;
    let d_input = stream
        .clone_htod(&input[..k])
        .map_err(|e| CudaGraphError::DriverError(format!("{what} upload input: {e}")))?;
    let mut d_output = stream
        .alloc_zeros::<f32>(n_rows)
        .map_err(|e| CudaGraphError::DriverError(format!("{what} alloc output: {e}")))?;
    // SAFETY: every buffer was just allocated on this graph's stream with the
    // element counts the kernel reads / writes (checked above).
    unsafe {
        if q4_0 {
            launch_gemv_q4_0_pf(
                &graph,
                &mods,
                &d_blocks,
                &d_input,
                &mut d_output,
                n_rows_u32,
                k_u32,
            )?;
        } else {
            launch_gemv_q8_0_pf(
                &graph,
                &mods,
                &d_blocks,
                &d_input,
                &mut d_output,
                n_rows_u32,
                k_u32,
            )?;
        }
    }
    stream
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("{what} sync: {e}")))?;
    let host = stream
        .clone_dtoh(&d_output)
        .map_err(|e| CudaGraphError::DriverError(format!("{what} read-back: {e}")))?;
    output[..n_rows].copy_from_slice(&host);
    Ok(())
}

/// Upload a token-major `[batch_size × k]` activation batch — the prefill's
/// "column-major" `buf[token * k + element]` layout.
fn upload_probe_batch(
    graph: &CudaGraph,
    what: &str,
    inputs: &[f32],
    k: usize,
    batch_size: usize,
) -> Result<CudaSlice<f32>, CudaGraphError> {
    let n = batch_size * k;
    if inputs.len() < n {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{what}: inputs.len()={} < batch_size*k={n}",
            inputs.len()
        )));
    }
    graph
        .stream_arc()
        .clone_htod(&inputs[..n])
        .map_err(|e| CudaGraphError::DriverError(format!("{what} upload inputs: {e}")))
}

/// Synchronise and copy a token-major `[batch_size × n_rows]` result back.
fn read_back_probe_batch(
    graph: &CudaGraph,
    what: &str,
    d_outputs: &CudaSlice<f32>,
    outputs: &mut [f32],
) -> Result<(), CudaGraphError> {
    let stream = graph.stream_arc();
    stream
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("{what} sync: {e}")))?;
    let host = stream
        .clone_dtoh(d_outputs)
        .map_err(|e| CudaGraphError::DriverError(format!("{what} read-back: {e}")))?;
    outputs[..host.len()].copy_from_slice(&host);
    Ok(())
}

/// Run the batch-prefill GEMM kernel on host data: `gemm_q4_0` (`q4_0`) or
/// `gemm_q8_0`, the kernels [`try_cuda_prefill_q_std`] uses for the fused
/// `Q‖K‖V`, attention-output and down projections.
///
/// `inputs` is token-major `[batch_size × k]` (token `t` at
/// `inputs[t * k..(t + 1) * k]`, the prefill's column-major batch layout);
/// `outputs` receives token-major `[batch_size × n_rows]` (the device buffer is
/// zeroed first because the kernel accumulates with `+=`). Every
/// `batch_size`, including ones past the kernel's 8-column chunk, is valid.
///
/// # Errors
/// As [`cuda_prefill_gemv_q_std`].
///
/// [`try_cuda_prefill_q_std`]: super::cuda_q_std_prefill::try_cuda_prefill_q_std
#[allow(clippy::too_many_arguments)]
pub fn cuda_prefill_gemm_q_std(
    blocks_bytes: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_rows: usize,
    k: usize,
    batch_size: usize,
    q4_0: bool,
) -> Result<(), CudaGraphError> {
    let what = "cuda_prefill_gemm_q_std";
    let (w_bytes, n_rows_u32, k_u32, batch_u32) =
        check_q_std_probe(what, blocks_bytes.len(), n_rows, k, batch_size, q4_0)?;
    let n_out = batch_size * n_rows;
    if outputs.len() < n_out {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{what}: outputs.len()={} < batch_size*n_rows={n_out}",
            outputs.len()
        )));
    }
    let graph = CudaGraph::global()?;
    let mods = init_q_std_prefill_modules(&graph)?;
    let d_inputs = upload_probe_batch(&graph, what, inputs, k, batch_size)?;
    let stream = graph.stream_arc();
    let d_blocks = stream
        .clone_htod(&blocks_bytes[..w_bytes])
        .map_err(|e| CudaGraphError::DriverError(format!("{what} upload weight: {e}")))?;
    let mut d_outputs = stream
        .alloc_zeros::<f32>(n_out)
        .map_err(|e| CudaGraphError::DriverError(format!("{what} alloc outputs: {e}")))?;
    // SAFETY: buffers allocated above on this stream with the exact element
    // counts the kernel indexes (`batch_size * k` in, `batch_size * n_rows` out).
    unsafe {
        if q4_0 {
            launch_gemm_q4_0(
                &graph,
                &mods,
                &d_blocks,
                &d_inputs,
                &mut d_outputs,
                n_rows_u32,
                k_u32,
                batch_u32,
            )?;
        } else {
            launch_gemm_q8_0(
                &graph,
                &mods,
                &d_blocks,
                &d_inputs,
                &mut d_outputs,
                n_rows_u32,
                k_u32,
                batch_u32,
            )?;
        }
    }
    read_back_probe_batch(&graph, what, &d_outputs, outputs)
}

/// Run the batch-prefill fused gate+up SwiGLU GEMM on host data:
/// `fused_gate_up_swiglu_gemm_q4_0` (`q4_0`) or `..._q8_0`, over the
/// `gate‖up` concatenation [`try_cuda_prefill_q_std`] uploads (gate rows
/// first). `outputs[t * n_ffn_rows + r] = silu(gate_r · x_t) * (up_r · x_t)`,
/// token-major as in [`cuda_prefill_gemm_q_std`].
///
/// # Errors
/// As [`cuda_prefill_gemv_q_std`]; `gate_bytes` and `up_bytes` must each hold
/// `n_ffn_rows * k / 32` blocks.
///
/// [`try_cuda_prefill_q_std`]: super::cuda_q_std_prefill::try_cuda_prefill_q_std
#[allow(clippy::too_many_arguments)]
pub fn cuda_prefill_gate_up_swiglu_q_std(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    inputs: &[f32],
    outputs: &mut [f32],
    n_ffn_rows: usize,
    k: usize,
    batch_size: usize,
    q4_0: bool,
) -> Result<(), CudaGraphError> {
    let what = "cuda_prefill_gate_up_swiglu_q_std";
    let (w_bytes, n_rows_u32, k_u32, batch_u32) =
        check_q_std_probe(what, gate_bytes.len(), n_ffn_rows, k, batch_size, q4_0)?;
    // The up half sits behind the gate half in one buffer, so its block index
    // (`n_ffn_rows * k / 32` past the gate's) must fit the kernel's u32 too.
    check_q_std_probe(what, 2 * w_bytes, 2 * n_ffn_rows, k, batch_size, q4_0)?;
    if up_bytes.len() < w_bytes {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{what}: up weight bytes too short: {} < {w_bytes}",
            up_bytes.len()
        )));
    }
    let n_out = batch_size * n_ffn_rows;
    if outputs.len() < n_out {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "{what}: outputs.len()={} < batch_size*n_ffn_rows={n_out}",
            outputs.len()
        )));
    }
    let mut fused = Vec::with_capacity(2 * w_bytes);
    fused.extend_from_slice(&gate_bytes[..w_bytes]);
    fused.extend_from_slice(&up_bytes[..w_bytes]);

    let graph = CudaGraph::global()?;
    let mods = init_q_std_prefill_modules(&graph)?;
    let d_inputs = upload_probe_batch(&graph, what, inputs, k, batch_size)?;
    let stream = graph.stream_arc();
    let d_blocks = stream
        .clone_htod(&fused)
        .map_err(|e| CudaGraphError::DriverError(format!("{what} upload gate||up: {e}")))?;
    let mut d_outputs = stream
        .alloc_zeros::<f32>(n_out)
        .map_err(|e| CudaGraphError::DriverError(format!("{what} alloc outputs: {e}")))?;
    // SAFETY: buffers allocated above on this stream; `d_blocks` holds the
    // `2 * n_ffn_rows` rows the kernel reads, `d_outputs` the
    // `batch_size * n_ffn_rows` it writes.
    unsafe {
        if q4_0 {
            launch_fused_gate_up_swiglu_q4_0(
                &graph,
                &mods,
                &d_blocks,
                &d_inputs,
                &mut d_outputs,
                n_rows_u32,
                k_u32,
                batch_u32,
            )?;
        } else {
            launch_fused_gate_up_swiglu_q8_0(
                &graph,
                &mods,
                &d_blocks,
                &d_inputs,
                &mut d_outputs,
                n_rows_u32,
                k_u32,
                batch_u32,
            )?;
        }
    }
    read_back_probe_batch(&graph, what, &d_outputs, outputs)
}
