//! Direct Metal dispatch engine for OxiBonsai FP8 (E4M3 + E5M2) GEMV.
//!
//! Phase 27 — Metal counterpart of `cuda_fp8_kernels.rs`.
//!
//! # Architecture
//!
//! - Independent singleton (own [`metal::Device`] + [`metal::CommandQueue`])
//! - Two compute pipelines: `gemv_fp8_e4m3` and `gemv_fp8_e5m2`
//! - All buffers use shared storage (`MTLResourceOptions::StorageModeShared`)
//!   so CPU-side reads/writes do not require explicit blit copies.
//!
//! Kept in its own file rather than merged into `metal_graph.rs` to honor the
//! 2000-line refactoring policy.
//!
//! # Block layout (AoS, 34 bytes/block — matches `BlockFP8E4M3` / `BlockFP8E5M2`)
//!
//! ```text
//! Block[i] = [q0, q1, ..., q31, scale_lo, scale_hi]
//! ```
//!
//! # Public API
//!
//! - [`metal_gemv_fp8_e4m3`] — FP8 E4M3FN GEMV
//! - [`metal_gemv_fp8_e5m2`] — FP8 E5M2 GEMV

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::OnceLock;

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

use super::kernel_sources::{MSL_GEMV_FP8_E4M3_V1, MSL_GEMV_FP8_E5M2_V1};
use super::metal_graph::MetalGraphError;

// ═══════════════════════════════════════════════════════════════════════════
// Singleton state
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide Metal FP8 dispatch state.
///
/// Holds the Metal device, command queue, and compiled pipelines.  Initialized
/// lazily on first call to [`metal_gemv_fp8_e4m3`] / [`metal_gemv_fp8_e5m2`].
struct MetalFp8State {
    device: Device,
    queue: CommandQueue,
    pipeline_e4m3: ComputePipelineState,
    pipeline_e5m2: ComputePipelineState,
}

// SAFETY: The underlying `metal::Device` and `metal::CommandQueue` are
// reference-counted Objective-C objects that are safe to share across threads
// once initialised.  Apple's Metal API documents these types as thread-safe.
unsafe impl Send for MetalFp8State {}
unsafe impl Sync for MetalFp8State {}

impl MetalFp8State {
    fn new() -> Result<Self, MetalGraphError> {
        let device = Device::system_default().ok_or(MetalGraphError::DeviceNotFound)?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();

        let lib_e4m3 = device
            .new_library_with_source(MSL_GEMV_FP8_E4M3_V1, &options)
            .map_err(|e| MetalGraphError::CompilationFailed(format!("FP8 E4M3 library: {e}")))?;
        let func_e4m3 = lib_e4m3
            .get_function("gemv_fp8_e4m3", None)
            .map_err(|e| MetalGraphError::CompilationFailed(format!("FP8 E4M3 function: {e}")))?;
        let pipeline_e4m3 = device
            .new_compute_pipeline_state_with_function(&func_e4m3)
            .map_err(|e| MetalGraphError::CompilationFailed(format!("FP8 E4M3 pipeline: {e}")))?;

        let lib_e5m2 = device
            .new_library_with_source(MSL_GEMV_FP8_E5M2_V1, &options)
            .map_err(|e| MetalGraphError::CompilationFailed(format!("FP8 E5M2 library: {e}")))?;
        let func_e5m2 = lib_e5m2
            .get_function("gemv_fp8_e5m2", None)
            .map_err(|e| MetalGraphError::CompilationFailed(format!("FP8 E5M2 function: {e}")))?;
        let pipeline_e5m2 = device
            .new_compute_pipeline_state_with_function(&func_e5m2)
            .map_err(|e| MetalGraphError::CompilationFailed(format!("FP8 E5M2 pipeline: {e}")))?;

        Ok(Self {
            device,
            queue,
            pipeline_e4m3,
            pipeline_e5m2,
        })
    }
}

/// Lazy process-wide singleton.
fn state() -> Result<&'static MetalFp8State, MetalGraphError> {
    static STATE: OnceLock<Result<MetalFp8State, MetalGraphError>> = OnceLock::new();
    match STATE.get_or_init(MetalFp8State::new) {
        Ok(s) => Ok(s),
        Err(e) => Err(clone_err(e)),
    }
}

fn clone_err(e: &MetalGraphError) -> MetalGraphError {
    match e {
        MetalGraphError::DeviceNotFound => MetalGraphError::DeviceNotFound,
        MetalGraphError::CompilationFailed(s) => MetalGraphError::CompilationFailed(s.clone()),
        MetalGraphError::BufferCreationFailed => MetalGraphError::BufferCreationFailed,
        MetalGraphError::EncodingFailed(s) => MetalGraphError::EncodingFailed(s.clone()),
        MetalGraphError::ExecutionFailed(s) => MetalGraphError::ExecutionFailed(s.clone()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public dispatch functions
// ═══════════════════════════════════════════════════════════════════════════

/// Block size in bytes for FP8 E4M3 / E5M2 (32 quantised weights + FP16 scale).
const FP8_BLOCK_BYTES: usize = 34;
/// Quantisation group size (number of weights per block).
const FP8_BLOCK_K: usize = 32;
/// Simdgroups per threadgroup (matches MSL kernel: 8 rows per CTA).
const SIMDS_PER_TG: usize = 8;
/// Threads per threadgroup (8 simdgroups × 32 lanes).
const THREADS_PER_TG: u64 = 256;

/// FP8 E4M3FN GEMV on Metal GPU.
///
/// # Arguments
/// - `blocks`: raw block bytes, length must equal `n_rows * (k / 32) * 34`.
/// - `input`: dense FP32 input vector, length `k`.
/// - `output`: dense FP32 output vector, length `n_rows`.
/// - `n_rows`: number of output rows.
/// - `k`: input dimension (must be a multiple of 32).
///
/// # Errors
/// Returns [`MetalGraphError::DeviceNotFound`] on systems without a Metal device,
/// [`MetalGraphError::CompilationFailed`] if pipeline creation failed, or
/// [`MetalGraphError::EncodingFailed`] for shape/buffer issues.
pub fn metal_gemv_fp8_e4m3(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    dispatch_metal_fp8_gemv(blocks, input, output, n_rows, k, Fp8Variant::E4M3)
}

/// FP8 E5M2 GEMV on Metal GPU.  See [`metal_gemv_fp8_e4m3`].
pub fn metal_gemv_fp8_e5m2(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> Result<(), MetalGraphError> {
    dispatch_metal_fp8_gemv(blocks, input, output, n_rows, k, Fp8Variant::E5M2)
}

#[derive(Copy, Clone)]
enum Fp8Variant {
    E4M3,
    E5M2,
}

fn dispatch_metal_fp8_gemv(
    blocks: &[u8],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    variant: Fp8Variant,
) -> Result<(), MetalGraphError> {
    // ── Validate dimensions ─────────────────────────────────────────────────
    if k == 0 || k % FP8_BLOCK_K != 0 {
        return Err(MetalGraphError::EncodingFailed(format!(
            "k = {k} must be a non-zero multiple of {FP8_BLOCK_K}"
        )));
    }
    let blocks_per_row = k / FP8_BLOCK_K;
    let expected_block_bytes = n_rows.saturating_mul(blocks_per_row) * FP8_BLOCK_BYTES;
    if blocks.len() != expected_block_bytes {
        return Err(MetalGraphError::EncodingFailed(format!(
            "blocks.len() = {} expected {} (n_rows = {n_rows}, k = {k})",
            blocks.len(),
            expected_block_bytes
        )));
    }
    if input.len() != k {
        return Err(MetalGraphError::EncodingFailed(format!(
            "input.len() = {} expected {k}",
            input.len()
        )));
    }
    if output.len() != n_rows {
        return Err(MetalGraphError::EncodingFailed(format!(
            "output.len() = {} expected {n_rows}",
            output.len()
        )));
    }

    let s = state()?;

    // ── Allocate buffers ────────────────────────────────────────────────────
    let block_buf = s.device.new_buffer_with_data(
        blocks.as_ptr() as *const std::ffi::c_void,
        blocks.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = s.device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (input.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = s.device.new_buffer(
        (n_rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    // Zero-initialise output (some Metal drivers leave new buffers uninitialised).
    unsafe {
        std::ptr::write_bytes(output_buf.contents() as *mut f32, 0u8, n_rows);
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| {
        MetalGraphError::EncodingFailed(format!("n_rows = {n_rows} exceeds u32::MAX"))
    })?;
    let k_u32 = u32::try_from(k)
        .map_err(|_| MetalGraphError::EncodingFailed(format!("k = {k} exceeds u32::MAX")))?;

    // ── Encode + commit ─────────────────────────────────────────────────────
    let cmd = s.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let pipeline = match variant {
        Fp8Variant::E4M3 => &s.pipeline_e4m3,
        Fp8Variant::E5M2 => &s.pipeline_e5m2,
    };
    encoder.set_compute_pipeline_state(pipeline);
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

    cmd.commit();
    cmd.wait_until_completed();

    // ── Read output back ────────────────────────────────────────────────────
    unsafe {
        let src = output_buf.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, output.as_mut_ptr(), n_rows);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — CI-GPU-gated parity tests on macOS, host-only signature checks elsewhere
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_variant_enum_compiles() {
        let _ = Fp8Variant::E4M3;
        let _ = Fp8Variant::E5M2;
    }

    #[test]
    fn block_size_constant_matches_core() {
        assert_eq!(FP8_BLOCK_BYTES, oxibonsai_core::BLOCK_FP8_BYTES);
        assert_eq!(FP8_BLOCK_K, oxibonsai_core::QK_FP8);
    }

    /// CPU-vs-GPU parity for FP8 E4M3 GEMV.
    ///
    /// Skipped silently on hosts without a Metal device (CI runners, Linux/Windows).
    #[test]
    fn metal_gemv_fp8_e4m3_matches_cpu_reference() {
        if state().is_err() {
            // No Metal device — skip on CPU-only CI hosts.
            return;
        }

        use oxibonsai_core::{BlockFP8E4M3, BLOCK_FP8_BYTES, QK_FP8};

        let n_rows = 16usize;
        let k = 128usize;
        let blocks_per_row = k / QK_FP8;

        // Build a deterministic FP8 weight matrix.
        let mut blocks_storage: Vec<BlockFP8E4M3> = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for b in 0..blocks_per_row {
                let scale_bits = ((row as u16 * 17) ^ (b as u16 * 23)) | 0x3C00; // ~1.0 around exponent
                let mut qs = [0u8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((row + b + i) as u8).wrapping_mul(13).wrapping_add(7);
                }
                // Mask the most-significant bit pattern that maps to NaN (0x7F / 0xFF)
                for q in qs.iter_mut() {
                    if *q == 0x7F || *q == 0xFF {
                        *q ^= 0x01;
                    }
                }
                blocks_storage.push(BlockFP8E4M3 {
                    qs,
                    d: half::f16::from_bits(scale_bits),
                });
            }
        }

        // Build an FP32 input vector.
        let input: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.5).collect();

        // CPU reference path.
        let mut cpu_out = vec![0.0f32; n_rows];
        crate::gemv_fp8::gemv_fp8_e4m3(&blocks_storage, &input, &mut cpu_out, n_rows, k)
            .expect("CPU FP8 E4M3 GEMV reference should succeed");

        // GPU path.
        let block_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks_storage.as_ptr().cast::<u8>(),
                blocks_storage.len() * BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; n_rows];
        metal_gemv_fp8_e4m3(block_bytes, &input, &mut gpu_out, n_rows, k)
            .expect("metal FP8 GEMV should succeed on Metal hardware");

        for i in 0..n_rows {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            let rel = diff / cpu_out[i].abs().max(1e-6);
            assert!(
                diff < 1e-3 || rel < 1e-3,
                "row {i}: cpu={} gpu={} diff={diff}",
                cpu_out[i],
                gpu_out[i]
            );
        }
    }

    /// CPU-vs-GPU parity for FP8 E5M2 GEMV. CI-GPU-gated like the E4M3 test.
    #[test]
    fn metal_gemv_fp8_e5m2_matches_cpu_reference() {
        if state().is_err() {
            return;
        }

        use oxibonsai_core::{BlockFP8E5M2, BLOCK_FP8_BYTES, QK_FP8};

        let n_rows = 17usize; // boundary: not a multiple of 8 → tests the simdgroup mask
        let k = 64usize;
        let blocks_per_row = k / QK_FP8;

        let mut blocks_storage: Vec<BlockFP8E5M2> = Vec::with_capacity(n_rows * blocks_per_row);
        for row in 0..n_rows {
            for b in 0..blocks_per_row {
                let scale_bits = ((row as u16 * 11) ^ (b as u16 * 5)) | 0x3800;
                let mut qs = [0u8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = ((row * 5 + b * 3 + i) as u8)
                        .wrapping_mul(7)
                        .wrapping_add(3);
                    // Avoid inf/NaN exponent (exp = 31): force bit 6 of exponent low when all set
                    if (*q & 0x7C) == 0x7C {
                        *q ^= 0x04;
                    }
                }
                blocks_storage.push(BlockFP8E5M2 {
                    qs,
                    d: half::f16::from_bits(scale_bits),
                });
            }
        }

        let input: Vec<f32> = (0..k).map(|i| (i as f32).sin()).collect();

        let mut cpu_out = vec![0.0f32; n_rows];
        crate::gemv_fp8::gemv_fp8_e5m2(&blocks_storage, &input, &mut cpu_out, n_rows, k)
            .expect("CPU FP8 E5M2 GEMV reference should succeed");

        let block_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks_storage.as_ptr().cast::<u8>(),
                blocks_storage.len() * BLOCK_FP8_BYTES,
            )
        };
        let mut gpu_out = vec![0.0f32; n_rows];
        metal_gemv_fp8_e5m2(block_bytes, &input, &mut gpu_out, n_rows, k)
            .expect("metal FP8 GEMV should succeed on Metal hardware");

        for i in 0..n_rows {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            let rel = diff / cpu_out[i].abs().max(1e-6);
            assert!(
                diff < 1e-3 || rel < 1e-3,
                "row {i}: cpu={} gpu={} diff={diff}",
                cpu_out[i],
                gpu_out[i]
            );
        }
    }
}
