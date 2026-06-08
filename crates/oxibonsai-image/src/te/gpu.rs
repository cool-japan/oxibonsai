//! GPU (Metal) backend for the FLUX.2 text-encoder (Qwen3-4B) f32 matmuls.
//!
//! This module routes the dominant per-layer Linears of the Qwen3-4B text
//! encoder (Q/K/V/o_proj + gate/up/down across 36 layers) onto the project's
//! f32-exact Metal GEMM kernel (`MetalGraph::encode_gemm_f32` in
//! `oxibonsai-kernels`), keeping each weight's row-major f32 bytes resident on
//! the GPU and crossing the bus only with the (small) f32 activations per
//! matmul.
//!
//! Unlike the DiT path ([`crate::gpu`]), the TE weights are **pure f32** (the
//! 4-bit MLX weights are dequantized to f32 offline by `TeWeights`), so the op
//! is a plain `out[m,n] = Σ_k input[m,k] · weight[n,k]` with **no quantization**
//! — the GPU only reassociates the sum, which keeps it cos ≈ 1.0 vs the CPU
//! `gemm_abt` reference. Parity is therefore trivially safe (the `te_parity`
//! gate stays cos ≥ 0.999).
//!
//! The whole module is gated on `cfg(all(feature = "metal", target_os =
//! "macos"))` — the same gate under which `oxibonsai-kernels` re-exports
//! `MetalGraph` — so a non-Metal / non-macOS build never references it and the
//! default Pure-Rust CPU path is entirely unaffected.
//!
//! Default OFF: unlike the DiT (`OXI_DIT_GPU`, default ON), the TE GPU path is
//! opt-in via `OXI_TE_GPU=1`. The CPU TE already tracks the goldens; the GPU
//! path is a speed optimization, enabled explicitly for A/B and production use.
//!
//! On *any* error this module returns a `TeGpuMatmulError`; the caller (the
//! `matmul` helper in [`crate::te::forward`]) swallows it and falls back to the
//! CPU [`crate::gemm::gemm_abt`], so a GPU failure can never break a forward
//! pass (no `unwrap`/`expect`/`panic!`).
//!
//! ## Weight residency (no-copy investigation)
//!
//! 36 layers of Qwen3-4B in f32 are ~16 GB once cached. The weights are ALREADY
//! f32 in host RAM (the `TeWeights` `.npy` buffers), so a zero-copy GPU alias
//! would be ideal. The `metal` crate (0.33) *does* expose
//! `Device::new_buffer_with_bytes_no_copy` (`newBufferWithBytesNoCopy:`), but
//! macOS requires the wrapped pointer to be **page-aligned** (`getpagesize()` =
//! 16 KiB on Apple Silicon) with a page-multiple length, or the call returns
//! `nil`; and the buffer would alias the host allocation, which must then
//! outlive every GPU use. The `TeWeights` weights are ordinary `Vec<f32>`
//! allocations from `.npy` parsing — not page-aligned — so a no-copy wrap would
//! be unsound/unreliable for them. We therefore use the existing
//! **upload-once-cache** path (`MetalGraph::get_or_upload_f32_weight`): each
//! weight is blitted to a `StorageModeShared` buffer the first time it is seen
//! and cached by its slice pointer for all subsequent forwards (the 36-layer
//! weight set uploads exactly once, ~16 GB resident on Apple unified memory).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use oxibonsai_kernels::{MetalGraph, MetalGraphError};

/// An error from the GPU f32-matmul path. The caller converts this into a
/// silent CPU fallback, so it never propagates out of a forward pass.
#[derive(Debug, thiserror::Error)]
pub enum TeGpuMatmulError {
    /// The process-wide Metal graph singleton could not be obtained (e.g. no
    /// Metal device, or the device failed to initialise).
    #[error("Metal graph unavailable: {0}")]
    GraphUnavailable(String),
    /// The f32-exact Metal GEMM (weight upload / encode / dispatch) failed.
    #[error("Metal f32 GEMM failed: {0}")]
    Metal(#[from] MetalGraphError),
}

/// One-time confirmation that the TE GPU path actually executed at least once
/// (used by the parity example to PROVE the GPU ran, not a silent CPU fallback).
static TE_GPU_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any [`te_matmul_gpu`] call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn te_gpu_was_used() -> bool {
    TE_GPU_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the GPU TE path. Default **OFF**; set env
/// `OXI_TE_GPU=1` to route the TE matmuls through the Metal f32 GEMM.
static TE_GPU_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the text encoder should use the GPU f32 path.
///
/// `false` unless the environment variable `OXI_TE_GPU` is set to `1`. The env
/// read is cached in a [`OnceLock`] on first call.
pub fn te_gpu_enabled() -> bool {
    *TE_GPU_ENABLED.get_or_init(|| matches!(std::env::var("OXI_TE_GPU").ok().as_deref(), Some("1")))
}

/// Compute `out[m, n] = Σ_k input[m, k] · weight[n, k]` (`x · Wᵀ`) on the GPU.
///
/// - `weight`: row-major f32 `[n, k]` (the dequantized TE Linear weight,
///   borrowed from the long-lived [`crate::te::weights::TeWeights`] registry).
/// - `input`: row-major `[m, k]`.
/// - `out`: row-major `[m, n]` (written in full).
///
/// The weight is uploaded to the GPU **once** and cached by its (stable,
/// per-weight-unique) slice pointer key, so subsequent layers/forwards reuse the
/// resident buffer and only the activations cross the bus.
///
/// # Errors
/// Returns [`TeGpuMatmulError`] if the Metal graph is unavailable or the kernel
/// upload/encode fails (e.g. a length mismatch). The caller falls back to the
/// CPU path on any error.
pub fn te_matmul_gpu(
    weight: &[f32],
    input: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), TeGpuMatmulError> {
    let graph =
        MetalGraph::global().map_err(|e| TeGpuMatmulError::GraphUnavailable(e.to_string()))?;
    // `weight` is borrowed from the run-long TeWeights registry; its base
    // address is stable and unique per weight, so it doubles as a cache key with
    // no per-Linear bookkeeping. Pointer addresses are huge and won't collide
    // with the DiT's pointer keys or the LLM's small key space.
    let key = weight.as_ptr() as u64;
    let handle = graph.get_or_upload_f32_weight(key, weight)?;
    graph.encode_gemm_f32(&handle, input, out, m, n, k)?;
    TE_GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn te_gpu_disabled_by_default_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_TE_GPU` unset → disabled). It does not mutate the env.
        if std::env::var("OXI_TE_GPU").is_err() {
            assert!(!te_gpu_enabled());
        }
    }
}
