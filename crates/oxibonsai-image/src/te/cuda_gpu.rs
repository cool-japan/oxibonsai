//! GPU (CUDA) backend for the FLUX.2 text-encoder (Qwen3-4B) f32 matmuls.
//!
//! CUDA sibling of [`crate::te::gpu`] (the Metal backend), authored as a
//! line-for-line mirror. It routes the dominant per-layer Linears of the
//! Qwen3-4B text encoder (Q/K/V/o_proj + gate/up/down across 36 layers) onto the
//! project's f32-exact CUDA GEMM kernel ([`CudaGraph::encode_gemm_f32`] in
//! `oxibonsai-kernels`), keeping each weight's row-major f32 bytes resident on
//! the GPU and crossing the bus only with the (small) f32 activations per
//! matmul.
//!
//! Unlike the DiT path ([`crate::cuda_gpu`]), the TE weights are **pure f32**
//! (the 4-bit MLX weights are dequantized to f32 offline by `TeWeights`), so the
//! op is a plain `out[m,n] = Σ_k input[m,k] · weight[n,k]` with **no
//! quantization** — the GPU only reassociates the sum, which keeps it cos ≈ 1.0
//! vs the CPU `gemm_abt` reference. Parity is therefore trivially safe (the
//! `te_parity` gate stays cos ≥ 0.999).
//!
//! The whole module is gated on `cfg(all(feature = "native-cuda", any(target_os
//! = "linux", target_os = "windows")))` — the same gate under which
//! `oxibonsai-kernels` re-exports [`CudaGraph`] — and is `target_os`-DISJOINT
//! from the Metal gate (macOS), so a non-CUDA build never references it and the
//! default Pure-Rust CPU path is entirely unaffected.
//!
//! Default OFF: unlike the DiT (`OXI_DIT_GPU`, default ON), the TE GPU path is
//! opt-in via `OXI_TE_GPU=1` (the same env var as the Metal path; Metal and CUDA
//! are mutually exclusive at build by `target_os`). The CPU TE already tracks the
//! goldens; the GPU path is a speed optimization, enabled explicitly for A/B and
//! production use.
//!
//! On *any* error this module returns a [`CudaTeGpuMatmulError`]; the caller (the
//! `matmul` helper in [`crate::te::forward`]) swallows it and falls back to the
//! CPU [`crate::gemm::gemm_abt`], so a GPU failure can never break a forward
//! pass (no `unwrap`/`expect`/`panic!`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use oxibonsai_kernels::{CudaGraph, CudaGraphError};

/// An error from the GPU f32-matmul path. The caller converts this into a
/// silent CPU fallback, so it never propagates out of a forward pass.
#[derive(Debug, thiserror::Error)]
pub enum CudaTeGpuMatmulError {
    /// The process-wide CUDA graph singleton could not be obtained (e.g. no
    /// CUDA device, or the device failed to initialise).
    #[error("CUDA graph unavailable: {0}")]
    GraphUnavailable(String),
    /// The f32-exact CUDA GEMM (weight upload / encode / dispatch) failed.
    #[error("CUDA f32 GEMM failed: {0}")]
    Cuda(#[from] CudaGraphError),
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
/// `OXI_TE_GPU=1` to route the TE matmuls through the CUDA f32 GEMM.
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
/// The weight is uploaded to the GPU for this call and **evicted immediately
/// afterwards**: the 4-bit MLX `TeWeights` dequantises into a fresh transient
/// buffer per Linear, so its pointer is recycled across calls and cannot serve
/// as a long-lived cache key (see the body comment). Each matmul therefore
/// re-uploads its weight; only one TE weight is GPU-resident at a time.
///
/// # Errors
/// Returns [`CudaTeGpuMatmulError`] if the CUDA graph is unavailable or the
/// kernel upload/encode fails (e.g. a length mismatch). The caller falls back to
/// the CPU path on any error.
pub fn te_matmul_gpu(
    weight: &[f32],
    input: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaTeGpuMatmulError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaTeGpuMatmulError::GraphUnavailable(e.to_string()))?;
    // CAUTION: `weight` is NOT a stable identity. For the native 4-bit MLX
    // source, `TeWeights` dequantises each Linear into a *fresh, transient* f32
    // buffer per call (the RAM-frugal "no-cache" policy), so its base pointer is
    // recycled by the allocator across Linears. Keying the device-weight cache on
    // that pointer is therefore only sound for the duration of THIS call: a later
    // Linear can land on a freed address and otherwise collide, making
    // `get_or_upload_f32_weight` hand back a *stale* buffer (wrong weights →
    // corrupted conditioning → a blurred render). So we evict the key right after
    // the GEMM: `handle` keeps the device buffer alive across `encode_gemm_f32`,
    // and the next call re-uploads fresh. (DiT/LLM keys are long-lived weights
    // with stable addresses and are unaffected — they are never evicted here.)
    let key = weight.as_ptr() as u64;
    let handle = graph.get_or_upload_f32_weight(key, weight)?;
    graph.encode_gemm_f32(&handle, input, out, m, n, k)?;
    graph.evict_f32_weight(key)?;
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
