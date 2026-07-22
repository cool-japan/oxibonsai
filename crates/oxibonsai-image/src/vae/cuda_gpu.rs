//! GPU (CUDA) backend for the FLUX.2 SMALL **VAE decoder** per-op f32
//! primitives.
//!
//! CUDA sibling of `crate::vae::gpu` (the Metal backend), authored as a
//! line-for-line mirror. It routes the heavy ops of the VAE decode path — the
//! 2-D convolutions (the prize: ~60% of the decode FLOPs and CPU-im2col-bound),
//! GroupNorm, SiLU, and nearest ×2 upsample — onto the project's parity-clean
//! f32 CUDA primitives in `oxibonsai-kernels`
//! (`CudaGraph::encode_conv2d_f32`, `CudaGraph::encode_groupnorm_f32`,
//! `CudaGraph::encode_silu_f32`, `CudaGraph::encode_upsample_nearest_f32`).
//!
//! Like the TE GPU path ([`crate::te::cuda_gpu`]) the VAE weights are **pure
//! f32** (the exported `.npy` conv/affine tensors), so every op is a plain f32
//! computation — the GPU only reassociates the sums, keeping each stage
//! cos ≈ 1.0 vs the CPU reference (the `vae_parity` gate stays cos ≥ 0.999).
//!
//! Conv weights keep the exported **MLX layout `[C_out, kH, kW, C_in]`** (which,
//! flattened row-major, is exactly the GEMM weight `[C_out, kH·kW·C_in]` the
//! kernel expects) — they are passed through verbatim, no relayout. Each weight
//! is uploaded to the GPU **once** and cached by its slice-pointer key (the
//! `Conv2d` weights are run-long `Vec<f32>` allocations, so the base address is
//! stable and unique per layer), so subsequent decodes reuse the resident buffer
//! and only the activations cross the bus.
//!
//! The whole module is gated on `cfg(all(feature = "native-cuda", any(target_os
//! = "linux", target_os = "windows")))` — the same gate under which
//! `oxibonsai-kernels` re-exports `CudaGraph` — and is `target_os`-DISJOINT
//! from the Metal gate (macOS), so a non-CUDA build never references it and the
//! default Pure-Rust CPU path is entirely unaffected.
//!
//! Default **ON** when the `native-cuda` feature is compiled (mirrors
//! `OXI_DIT_GPU`): the GPU VAE is a parity-proven win over the CPU decode. The
//! same `OXI_VAE_GPU` env var as the Metal path is reused (Metal and CUDA are
//! mutually exclusive at build by `target_os`). Set `OXI_VAE_GPU=0` to force the
//! CPU reference (for A/B parity testing without recompiling). Default-on is only
//! safe because every op silently falls back to the CPU path on any GPU error.
//!
//! On *any* error each wrapper returns a `CudaVaeGpuError`; the call sites in
//! `conv.rs` / `norm.rs` / `ops.rs` swallow it and fall back to the CPU path, so
//! a GPU failure can never break a decode (no `unwrap`/`expect`/`panic!`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use oxibonsai_kernels::{CudaGraph, CudaGraphError};

/// An error from the GPU VAE primitive path. The caller converts this into a
/// silent CPU fallback, so it never propagates out of a decode.
#[derive(Debug, thiserror::Error)]
pub enum CudaVaeGpuError {
    /// The process-wide CUDA graph singleton could not be obtained (e.g. no
    /// CUDA device, or the device failed to initialise).
    #[error("CUDA graph unavailable: {0}")]
    GraphUnavailable(String),
    /// A CUDA VAE primitive (weight upload / encode / dispatch) failed.
    #[error("CUDA VAE primitive failed: {0}")]
    Cuda(#[from] CudaGraphError),
}

/// One-time confirmation that the VAE GPU path actually executed at least once
/// (used by the parity example to PROVE the GPU ran, not a silent CPU fallback).
static VAE_GPU_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any VAE GPU primitive call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn vae_gpu_was_used() -> bool {
    VAE_GPU_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the GPU VAE path. Default **ON** when the
/// `native-cuda` feature is compiled (the GPU convs/norms/silu/upsample are a
/// parity-proven win — gated cos ≥ 0.999); set env `OXI_VAE_GPU=0` to force the
/// CPU path (for A/B parity testing without recompiling).
static VAE_GPU_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the VAE decoder should use the GPU f32 path.
///
/// `true` unless the environment variable `OXI_VAE_GPU` is set to `0`. The env
/// read is cached in a [`OnceLock`] on first call. The per-op CPU fallback in
/// `conv.rs` / `norm.rs` / `ops.rs` (a silent fall-through on any GPU `Err`)
/// makes default-on safe.
pub fn vae_gpu_enabled() -> bool {
    *VAE_GPU_ENABLED
        .get_or_init(|| !matches!(std::env::var("OXI_VAE_GPU").ok().as_deref(), Some("0")))
}

/// Run a stride-1 "same"-padded 2-D convolution on the GPU, returning the NCHW
/// output `[c_out, h_out, w_out]` (`h_out = h + 2·pad − k + 1`).
///
/// - `weight`: row-major MLX-layout `[c_out, kH, kW, c_in]` (== flattened
///   `[c_out, kH·kW·c_in]`), borrowed from the run-long [`crate::vae::conv::Conv2d`];
///   its base address is the upload-cache key.
/// - `bias`: `[c_out]`.
/// - `input`: NCHW `[c_in, h, w]`.
/// - `c_in` / `c_out` / `h` / `w` / `k` / `pad`: layer geometry.
///
/// # Errors
/// `CudaVaeGpuError` if the CUDA graph is unavailable or the kernel
/// upload/encode fails (e.g. a length/shape mismatch). The caller falls back to
/// the CPU path on any error.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_gpu(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    k: usize,
    pad: usize,
) -> Result<ConvGpuOut, CudaVaeGpuError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))?;
    // "same"-stride-1 output geometry (matches encode_conv2d_f32 / the CPU conv).
    let h_out = h + 2 * pad + 1 - k;
    let w_out = w + 2 * pad + 1 - k;
    let mut out = vec![0.0f32; c_out * h_out * w_out];
    // Upload + cache the conv weight by its stable slice-pointer key. Pointer
    // addresses are huge and won't collide with the DiT/TE/LLM key spaces.
    let key = weight.as_ptr() as u64;
    let handle = graph.get_or_upload_f32_weight(key, weight)?;
    graph.encode_conv2d_f32(&handle, input, bias, &mut out, c_in, c_out, h, w, k, pad)?;
    VAE_GPU_USED.store(true, Ordering::Relaxed);
    Ok(ConvGpuOut {
        data: out,
        h: h_out,
        w: w_out,
    })
}

/// Output of [`conv2d_gpu`]: NCHW data `[c_out, h, w]` plus the new spatial dims.
pub struct ConvGpuOut {
    /// NCHW output buffer `[c_out, h, w]`.
    pub data: Vec<f32>,
    /// Output height.
    pub h: usize,
    /// Output width.
    pub w: usize,
}

/// Run PyTorch-compatible GroupNorm on the GPU, in place over the NCHW buffer
/// `x` `[channels, hw]`.
///
/// - `weight` / `bias`: per-channel affine `[channels]`.
/// - `num_groups`: 32 in the VAE; `eps`: 1e-6 in the VAE.
///
/// # Errors
/// `CudaVaeGpuError` if the CUDA graph is unavailable or the kernel
/// upload/encode fails. The caller falls back to the CPU path on any error.
pub fn groupnorm_gpu(
    x: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    channels: usize,
    hw: usize,
    num_groups: usize,
    eps: f32,
) -> Result<(), CudaVaeGpuError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))?;
    graph.encode_groupnorm_f32(x, weight, bias, channels, hw, num_groups, eps)?;
    VAE_GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Apply element-wise SiLU (`x · sigmoid(x)`) on the GPU, in place.
///
/// # Errors
/// `CudaVaeGpuError` if the CUDA graph is unavailable or the kernel
/// upload/encode fails. The caller falls back to the CPU path on any error.
pub fn silu_gpu(x: &mut [f32]) -> Result<(), CudaVaeGpuError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))?;
    graph.encode_silu_f32(x)?;
    VAE_GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Run a nearest-neighbour ×2 upsample on the GPU, returning the NCHW output
/// `[c, 2h, 2w]`.
///
/// - `input`: NCHW `[c, h, w]`.
///
/// # Errors
/// `CudaVaeGpuError` if the CUDA graph is unavailable or the kernel
/// upload/encode fails. The caller falls back to the CPU path on any error.
pub fn upsample_gpu(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
) -> Result<UpsampleGpuOut, CudaVaeGpuError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))?;
    let h_out = h * 2;
    let w_out = w * 2;
    let mut out = vec![0.0f32; c * h_out * w_out];
    graph.encode_upsample_nearest_f32(input, &mut out, c, h, w)?;
    VAE_GPU_USED.store(true, Ordering::Relaxed);
    Ok(UpsampleGpuOut {
        data: out,
        h: h_out,
        w: w_out,
    })
}

/// Output of [`upsample_gpu`]: NCHW data `[c, 2h, 2w]` plus the new spatial dims.
pub struct UpsampleGpuOut {
    /// NCHW output buffer `[c, 2h, 2w]`.
    pub data: Vec<f32>,
    /// Output height (`2h`).
    pub h: usize,
    /// Output width (`2w`).
    pub w: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_gpu_enabled_by_default_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_VAE_GPU` unset → enabled). It does not mutate the env.
        if std::env::var("OXI_VAE_GPU").is_err() {
            assert!(vae_gpu_enabled());
        }
    }
}
