//! Direct Metal dispatch engine for OxiBonsai standard-quant (`Q4_0` / `Q8_0`) GEMV.
//!
//! Metal counterpart of `cuda_q_std_kernels.rs`:
//!
//! - **No private Metal state** (`MET-10`). Both pipelines, `gemv_q4_0` and
//!   `gemv_q8_0`, are resolved by name from the combined metallib that
//!   `build.rs` embeds (`ACTIVE_KERNELS` lists `MSL_GEMV_Q4_0_V1` /
//!   `MSL_GEMV_Q8_0_V1`), through [`MetalGraph::pipeline_for`] — the same
//!   embedded → disk-cached → `xcrun` → runtime-source cascade every other
//!   kernel family uses. This file used to open its own device and compile its
//!   own `MTLLibrary` from source on first use, uncached, every process start.
//! - Dispatch runs on the *current session's* command queue
//!   ([`MetalGraph::global`]), so an engine-pool replica's GEMVs stay on that
//!   replica's queue (`MET-08`).
//! - All buffers use shared storage (`MTLResourceOptions::StorageModeShared`)
//!   so CPU-side reads/writes need no explicit blit.
//!
//! Kept in its own file (not merged into `metal_graph`) to honor the 2000-line
//! refactoring policy.
//!
//! # Public API
//!
//! - [`metal_gemv_q4_0`] — `Q4_0` GEMV (18 bytes/block, 32 weights).
//! - [`metal_gemv_q8_0`] — `Q8_0` GEMV (34 bytes/block, 32 weights).

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal::MTLResourceOptions;

use super::metal_graph::{commit_and_wait, MetalGraph, MetalGraphError};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Weights per standard-quant block (`Q4_0` / `Q8_0`).
const Q_STD_BLOCK_K: usize = 32;
/// Bytes per `Q4_0` block (FP16 scale + 16 nibble bytes).
const Q4_0_BLOCK_BYTES: usize = 18;
/// Bytes per `Q8_0` block (FP16 scale + 32 int8 weights).
const Q8_0_BLOCK_BYTES: usize = 34;
/// Simdgroups per threadgroup (one output row per simdgroup).
const SIMDS_PER_TG: usize = 8;
/// Threads per threadgroup (8 simdgroups × 32 lanes).
const THREADS_PER_TG: u64 = 256;

/// Entry point of the `Q4_0` GEMV kernel in the combined metallib
/// (`kernel_sources::MSL_GEMV_Q4_0_V1`).
pub(crate) const GEMV_Q4_0_ENTRY: &str = "gemv_q4_0";
/// Entry point of the `Q8_0` GEMV kernel in the combined metallib
/// (`kernel_sources::MSL_GEMV_Q8_0_V1`).
pub(crate) const GEMV_Q8_0_ENTRY: &str = "gemv_q8_0";

// ═══════════════════════════════════════════════════════════════════════════
// Public dispatch functions
// ═══════════════════════════════════════════════════════════════════════════

/// `Q4_0` GEMV on the Metal GPU.
///
/// # Arguments
/// - `blocks`: raw AoS block bytes, length `n_rows * (k / 32) * 18`.
/// - `input`: dense FP32 input vector, length `k`.
/// - `output`: dense FP32 output vector, length `n_rows`.
/// - `n_rows`: number of output rows.
/// - `k`: input dimension (must be a positive multiple of 32).
///
/// # Errors
/// Returns [`MetalGraphError::DeviceNotFound`] on systems without a Metal device,
/// [`MetalGraphError::CompilationFailed`] if pipeline creation failed, or
/// [`MetalGraphError::EncodingFailed`] for shape/buffer mismatches.
pub fn metal_gemv_q4_0(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    dispatch_q_std_gemv(
        GEMV_Q4_0_ENTRY,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q4_0_BLOCK_BYTES,
        "Q4_0",
    )
}

/// `Q8_0` GEMV on the Metal GPU. See [`metal_gemv_q4_0`].
pub fn metal_gemv_q8_0(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    dispatch_q_std_gemv(
        GEMV_Q8_0_ENTRY,
        blocks,
        input,
        output,
        n_rows,
        k,
        Q8_0_BLOCK_BYTES,
        "Q8_0",
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_q_std_gemv(
    entry: &str,
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    block_bytes: usize,
    format: &str,
) -> Result<(), MetalGraphError> {
    // ── Validate dimensions ─────────────────────────────────────────────────
    // Shape errors are reported before the device is touched, so a malformed
    // call fails the same way on a host with no Metal GPU at all.
    if k == 0 || !k.is_multiple_of(Q_STD_BLOCK_K) {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: k = {k} must be a non-zero multiple of {Q_STD_BLOCK_K}"
        )));
    }
    let blocks_per_row = k / Q_STD_BLOCK_K;
    let expected_block_bytes = n_rows.saturating_mul(blocks_per_row) * block_bytes;
    if blocks.len() != expected_block_bytes {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: blocks.len() = {} expected {expected_block_bytes} (n_rows = {n_rows}, k = {k})",
            blocks.len()
        )));
    }
    if input.len() != k {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: input.len() = {} expected {k}",
            input.len()
        )));
    }
    if output.len() != n_rows {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{format} GEMV: output.len() = {} expected {n_rows}",
            output.len()
        )));
    }
    if n_rows == 0 {
        return Ok(());
    }

    // `MET-10`: the shared device, the *current session's* command queue, and
    // a pipeline resolved by name from the combined metallib — no private
    // device, queue or library. `MetalGraph::global()` resolves to the session
    // bound to this thread (an engine-pool replica's), else the process
    // default; `pipeline_for` caches the pipeline state by name, so after the
    // first call this is a map lookup and a refcount bump.
    let graph = MetalGraph::global()?;
    let pipeline = graph.pipeline_for(entry)?;

    // ── Allocate buffers (shared storage) ───────────────────────────────────
    let block_buf = graph.device().new_buffer_with_data(
        blocks.as_ptr() as *const std::ffi::c_void,
        blocks.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = graph.device().new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(input) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = graph.device().new_buffer(
        (n_rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    // Zero-initialise output (some drivers leave new buffers uninitialised).
    unsafe {
        std::ptr::write_bytes(output_buf.contents() as *mut f32, 0u8, n_rows);
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("n_rows = {n_rows} exceeds u32::MAX"))
    })?;
    let k_u32 = u32::try_from(k)
        .map_err(|_| MetalGraphError::EncodingFailed(format!("k = {k} exceeds u32::MAX")))?;

    // ── Encode + commit ─────────────────────────────────────────────────────
    let cmd = graph.command_queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&block_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &n_rows_u32 as *const u32 as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &k_u32 as *const u32 as *const std::ffi::c_void,
    );

    let n_tgs = n_rows.div_ceil(SIMDS_PER_TG) as u64;
    let grid = metal::MTLSize::new(n_tgs, 1, 1);
    let tg_size = metal::MTLSize::new(THREADS_PER_TG, 1, 1);
    encoder.dispatch_thread_groups(grid, tg_size);
    encoder.end_encoding();

    commit_and_wait(cmd, "metal_gemv_q_std")?;

    // ── Read output back ────────────────────────────────────────────────────
    unsafe {
        let src = output_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, output.as_mut_ptr(), n_rows);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — CI-GPU-gated parity, host-only constant checks elsewhere
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_constants_match_core() {
        assert_eq!(Q4_0_BLOCK_BYTES, oxibonsai_core::BLOCK_Q4_0_BYTES);
        assert_eq!(Q8_0_BLOCK_BYTES, oxibonsai_core::BLOCK_Q8_0_BYTES);
        assert_eq!(Q_STD_BLOCK_K, oxibonsai_core::QK_Q4_0);
        assert_eq!(Q_STD_BLOCK_K, oxibonsai_core::QK_Q8_0);
    }

    /// `k` not a multiple of 32 is rejected before any GPU work — which is
    /// now literally true: validation runs before `MetalGraph::global()`, so
    /// this holds (and runs) on a host with no Metal device at all.
    #[test]
    fn q4_0_bad_k_rejected() {
        let blocks = vec![0u8; Q4_0_BLOCK_BYTES];
        let input = vec![0.0f32; 31];
        let mut output = vec![0.0f32; 1];
        match metal_gemv_q4_0(&blocks, &input, &mut output, 1, 31) {
            Err(MetalGraphError::EncodingFailed(msg)) => {
                assert!(msg.contains("multiple of 32"), "msg = {msg}");
            }
            other => panic!("expected EncodingFailed, got {other:?}"),
        }
    }

    /// `MET-10`: both entry points resolve from the **combined** metallib
    /// (no per-family library), and a real dispatch through that pipeline
    /// matches the scalar reference.
    #[test]
    fn q_std_entries_resolve_from_the_combined_metallib_and_dispatch() {
        let Ok(graph) = MetalGraph::global() else {
            return; // no Metal device on this host
        };
        for entry in [GEMV_Q4_0_ENTRY, GEMV_Q8_0_ENTRY] {
            graph
                .pipeline_for(entry)
                .unwrap_or_else(|e| panic!("{entry} must resolve from the combined metallib: {e}"));
        }

        // One Q8_0 row of 64 weights: two blocks, scale 0.5 and 0.25,
        // quants `i - 16` so every lane is distinct and non-zero on average.
        let k = 64usize;
        let mut blocks = Vec::with_capacity(2 * Q8_0_BLOCK_BYTES);
        let mut expected = 0.0f32;
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.03 - 0.7).collect();
        for (b, scale) in [0.5f32, 0.25].iter().enumerate() {
            blocks.extend_from_slice(&half::f16::from_f32(*scale).to_le_bytes());
            for i in 0..Q_STD_BLOCK_K {
                let q = (i as i8) - 16;
                blocks.push(q as u8);
                expected += scale * f32::from(q) * input[b * Q_STD_BLOCK_K + i];
            }
        }
        let mut out = [0.0f32; 1];
        metal_gemv_q8_0(&blocks, &input, &mut out, 1, k).expect("Q8_0 GEMV via pipeline_for");
        // Only the summation order differs from the scalar loop above.
        let tol = 1e-4 * expected.abs().max(1.0);
        assert!(
            (out[0] - expected).abs() <= tol,
            "Q8_0 via the combined metallib: got {}, expected {expected}",
            out[0]
        );
        assert!(expected.abs() > 1e-3, "degenerate fixture");
    }
}
