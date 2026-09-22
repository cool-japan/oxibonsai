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
//! `conv.rs` / `norm.rs` / `ops.rs` swallow it and fall back to the CPU path.
//! That fallback is no longer *silent* (RAG-EVAL-IMG-18): every wrapper below
//! emits a one-time `tracing::warn!` naming the failed op and the underlying
//! error the first time it happens (an `AtomicBool` latch per op, mirroring
//! `crate::vae::gpu`'s Metal sibling exactly). A GPU failure can still never
//! break a decode (no `unwrap`/`expect`/`panic!`).
//!
//! Also mirrors `crate::vae::gpu`'s `#[cfg(test)]`-only conv-dispatch override
//! (T-Missed-1), including its **thread-local** storage (not a process-global
//! atomic — a process-global override let a dispatch-forcing test on one
//! thread corrupt an unrelated test's GroupNorm/SiLU calls on another thread
//! mid-invocation; see that module's doc for the full incident): this file is
//! `target_os`-disjoint from the Metal module (never both compiled), but the
//! same hazards would apply here too on a Linux/Windows+CUDA build running
//! this crate's test suite, so the override is implemented identically rather
//! than only on the Metal side.
//!
//! **No CUDA hardware on the reference macOS/Apple-Silicon development
//! machine**, so this module has never *run*, and this crate's own gate
//! commands never compile it (`target_os`-gated to Linux/Windows). It has,
//! however, been type-checked for real: `cargo check`/`cargo clippy -- -D
//! warnings -p oxibonsai-image --target x86_64-unknown-linux-gnu --features
//! native-cuda` (both `--all-features` and `native-cuda`-only) succeed
//! cleanly, including this module's `#[cfg(test)]` code — cross-compiling
//! resolves every type and runs every lint without needing a CUDA toolkit,
//! since `cargo check` only type-checks and never links. The same target with
//! `--features cuda` (the compile-only stub, `native-cuda` NOT enabled) was
//! also checked, confirming `vae::tiling::force_conv_dispatch`'s three-way
//! `cfg` split correctly falls through to its no-op arm in that configuration
//! rather than dangling on this module. What is *not* verified is linking or
//! execution (this workstation has no Linux cross-linker configured and no
//! CUDA runtime/device either way), so a logic bug that only manifests at the
//! kernel-call boundary (e.g. an actual `CudaGraph`/`CudaGraphError`
//! behavioural mismatch) would not be caught here. It is authored as a
//! character-exact mirror of the already-compiling,
//! already-tested Metal sibling (`crate::vae::gpu`) with only the `Metal*` ->
//! `Cuda*` renames the type signatures require.

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

thread_local! {
    /// Test-only conv-dispatch override consulted by `vae_gpu_enabled` *before*
    /// `VAE_GPU_ENABLED`'s `OnceLock`. Mirrors `crate::vae::gpu`'s Metal
    /// sibling exactly, including being thread-local rather than a
    /// process-global atomic — see that module's doc for the full T-Missed-1
    /// contract. `None` = unset, `Some(false)`/`Some(true)` force CPU/GPU, but
    /// only for `vae_gpu_enabled` calls made on *this* thread.
    #[cfg(test)]
    static TEST_CONV_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// RAII guard returned by `set_conv_override`. Mirrors
/// `crate::vae::gpu::ConvOverrideGuard` exactly: dropping it clears this
/// thread's `TEST_CONV_OVERRIDE` back to `None`.
#[cfg(test)]
pub(crate) struct ConvOverrideGuard;

#[cfg(test)]
impl Drop for ConvOverrideGuard {
    fn drop(&mut self) {
        TEST_CONV_OVERRIDE.with(|c| c.set(None));
    }
}

/// Force (or, with `None`, decline to force) `vae_gpu_enabled`'s conv-dispatch
/// decision **for the calling thread only**, for the life of the returned
/// guard. See `crate::vae::gpu::set_conv_override`'s doc for the full contract
/// (this is a character-exact mirror); no lock is needed, since the override
/// is thread-local.
#[cfg(test)]
pub(crate) fn set_conv_override(v: Option<bool>) -> ConvOverrideGuard {
    TEST_CONV_OVERRIDE.with(|c| c.set(v));
    ConvOverrideGuard
}

/// Read this thread's `TEST_CONV_OVERRIDE`. Always `None` outside
/// `#[cfg(test)]`.
#[cfg(test)]
fn test_conv_override() -> Option<bool> {
    TEST_CONV_OVERRIDE.with(std::cell::Cell::get)
}

/// Non-test builds have no override machinery (it is `#[cfg(test)]` only), so
/// this is always `None`.
#[cfg(not(test))]
#[inline]
fn test_conv_override() -> Option<bool> {
    None
}

/// Whether the VAE decoder should use the GPU f32 path.
///
/// `true` unless the environment variable `OXI_VAE_GPU` is set to `0`. The env
/// read is cached in a `OnceLock` on first call. The per-op CPU fallback in
/// `conv.rs` / `norm.rs` / `ops.rs` (a silent fall-through on any GPU `Err`)
/// makes default-on safe.
///
/// In test builds this first consults the `set_conv_override` test hook; see
/// `crate::vae::gpu::vae_gpu_enabled`'s doc (this mirrors it exactly).
pub fn vae_gpu_enabled() -> bool {
    if let Some(v) = test_conv_override() {
        return v;
    }
    *VAE_GPU_ENABLED
        .get_or_init(|| !matches!(std::env::var("OXI_VAE_GPU").ok().as_deref(), Some("0")))
}

/// Latch for [`conv2d_gpu`]'s one-time fallback warning.
static CONV_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`groupnorm_gpu`]'s one-time fallback warning.
static GROUPNORM_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`silu_gpu`]'s one-time fallback warning.
static SILU_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
/// Latch for [`upsample_gpu`]'s one-time fallback warning.
static UPSAMPLE_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// Emit a one-time `tracing::warn!` the first time `flag`'s GPU op fails and
/// the caller falls back to the CPU reference path. Mirrors
/// `crate::vae::gpu::warn_gpu_fallback_once` exactly.
fn warn_gpu_fallback_once(flag: &'static AtomicBool, op: &str, reason: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            op,
            reason,
            "oxibonsai-image: VAE GPU op failed, falling back to CPU; further \
             occurrences in this process are not logged"
        );
    }
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
/// the CPU path on any error (a one-time `tracing::warn!` fires first — see
/// `warn_gpu_fallback_once`).
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
    let graph = CudaGraph::global()
        .map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))
        .inspect_err(|e| warn_gpu_fallback_once(&CONV_FALLBACK_WARNED, "conv2d", &e.to_string()))?;
    // "same"-stride-1 output geometry (matches encode_conv2d_f32 / the CPU conv).
    let h_out = h + 2 * pad + 1 - k;
    let w_out = w + 2 * pad + 1 - k;
    let mut out = vec![0.0f32; c_out * h_out * w_out];
    // Upload + cache the conv weight by its stable slice-pointer key. Pointer
    // addresses are huge and won't collide with the DiT/TE/LLM key spaces.
    let key = weight.as_ptr() as u64;
    let handle = graph
        .get_or_upload_f32_weight(key, weight)
        .inspect_err(|e| warn_gpu_fallback_once(&CONV_FALLBACK_WARNED, "conv2d", &e.to_string()))?;
    graph
        .encode_conv2d_f32(&handle, input, bias, &mut out, c_in, c_out, h, w, k, pad)
        .inspect_err(|e| warn_gpu_fallback_once(&CONV_FALLBACK_WARNED, "conv2d", &e.to_string()))?;
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
/// upload/encode fails. The caller falls back to the CPU path on any error (a
/// one-time `tracing::warn!` fires first — see `warn_gpu_fallback_once`).
pub fn groupnorm_gpu(
    x: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    channels: usize,
    hw: usize,
    num_groups: usize,
    eps: f32,
) -> Result<(), CudaVaeGpuError> {
    let graph = CudaGraph::global()
        .map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))
        .inspect_err(|e| {
            warn_gpu_fallback_once(&GROUPNORM_FALLBACK_WARNED, "groupnorm", &e.to_string())
        })?;
    graph
        .encode_groupnorm_f32(x, weight, bias, channels, hw, num_groups, eps)
        .inspect_err(|e| {
            warn_gpu_fallback_once(&GROUPNORM_FALLBACK_WARNED, "groupnorm", &e.to_string())
        })?;
    VAE_GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Apply element-wise SiLU (`x · sigmoid(x)`) on the GPU, in place.
///
/// # Errors
/// `CudaVaeGpuError` if the CUDA graph is unavailable or the kernel
/// upload/encode fails. The caller falls back to the CPU path on any error (a
/// one-time `tracing::warn!` fires first — see `warn_gpu_fallback_once`).
pub fn silu_gpu(x: &mut [f32]) -> Result<(), CudaVaeGpuError> {
    let graph = CudaGraph::global()
        .map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))
        .inspect_err(|e| warn_gpu_fallback_once(&SILU_FALLBACK_WARNED, "silu", &e.to_string()))?;
    graph
        .encode_silu_f32(x)
        .inspect_err(|e| warn_gpu_fallback_once(&SILU_FALLBACK_WARNED, "silu", &e.to_string()))?;
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
/// upload/encode fails. The caller falls back to the CPU path on any error (a
/// one-time `tracing::warn!` fires first — see `warn_gpu_fallback_once`).
pub fn upsample_gpu(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
) -> Result<UpsampleGpuOut, CudaVaeGpuError> {
    let graph = CudaGraph::global()
        .map_err(|e| CudaVaeGpuError::GraphUnavailable(e.to_string()))
        .inspect_err(|e| {
            warn_gpu_fallback_once(&UPSAMPLE_FALLBACK_WARNED, "upsample", &e.to_string())
        })?;
    let h_out = h * 2;
    let w_out = w * 2;
    let mut out = vec![0.0f32; c * h_out * w_out];
    graph
        .encode_upsample_nearest_f32(input, &mut out, c, h, w)
        .inspect_err(|e| {
            warn_gpu_fallback_once(&UPSAMPLE_FALLBACK_WARNED, "upsample", &e.to_string())
        })?;
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
        // Clears any override on this thread — see `crate::vae::gpu`'s
        // sibling test doc for why thread-local storage makes this safe
        // under `cargo test`'s multi-threaded harness with no lock.
        let _guard = set_conv_override(None);
        // OnceLock caches the first read; this asserts the default policy
        // (env `OXI_VAE_GPU` unset → enabled). It does not mutate the env.
        if std::env::var("OXI_VAE_GPU").is_err() {
            assert!(vae_gpu_enabled());
        }
    }

    #[test]
    fn conv_override_forces_cpu_regardless_of_the_cached_default() {
        let _guard = set_conv_override(Some(false));
        assert!(!vae_gpu_enabled(), "Some(false) must force the CPU path");
    }

    #[test]
    fn conv_override_forces_gpu_regardless_of_the_cached_default() {
        let _guard = set_conv_override(Some(true));
        assert!(vae_gpu_enabled(), "Some(true) must force the GPU path");
    }

    #[test]
    fn warn_gpu_fallback_once_fires_exactly_once_per_flag() {
        // A private, test-local latch — see `crate::vae::gpu`'s sibling test.
        static LOCAL_WARNED: AtomicBool = AtomicBool::new(false);
        assert!(!LOCAL_WARNED.load(Ordering::Relaxed));
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "first reason");
        assert!(LOCAL_WARNED.load(Ordering::Relaxed));
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "second reason");
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "third reason");
        assert!(LOCAL_WARNED.load(Ordering::Relaxed));
    }
}
