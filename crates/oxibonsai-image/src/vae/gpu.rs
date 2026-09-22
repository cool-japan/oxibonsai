//! GPU (Metal) backend for the FLUX.2 SMALL **VAE decoder** per-op f32
//! primitives.
//!
//! This module routes the heavy ops of the VAE decode path — the 2-D
//! convolutions (the prize: ~60% of the decode FLOPs and CPU-im2col-bound),
//! GroupNorm, SiLU, and nearest ×2 upsample — onto the project's parity-clean
//! f32 Metal primitives in `oxibonsai-kernels`
//! (`MetalGraph::encode_conv2d_f32`, `MetalGraph::encode_groupnorm_f32`,
//! `MetalGraph::encode_silu_f32`, `MetalGraph::encode_upsample_nearest_f32`).
//!
//! Like the TE GPU path ([`crate::te::gpu`]) the VAE weights are **pure f32**
//! (the exported `.npy` conv/affine tensors), so every op is a plain f32
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
//! The whole module is gated on `cfg(all(feature = "metal", target_os =
//! "macos"))` — the same gate under which `oxibonsai-kernels` re-exports
//! `MetalGraph` — so a non-Metal / non-macOS build never references it and the
//! default Pure-Rust CPU path is entirely unaffected.
//!
//! Default **ON** when the `metal` feature is compiled (mirrors `OXI_DIT_GPU`):
//! the GPU VAE is a parity-proven 3.2× win over the CPU decode, so it is the
//! normal path in a metal build. Set `OXI_VAE_GPU=0` to force the CPU reference
//! (for A/B parity testing without recompiling). Default-on is only safe because
//! every op silently falls back to the CPU path on any GPU error (below).
//!
//! On *any* error each wrapper returns a `VaeGpuError`; the call sites in
//! `conv.rs` / `norm.rs` / `ops.rs` swallow it and fall back to the CPU path.
//! That fallback is no longer *silent* (RAG-EVAL-IMG-18): every wrapper below
//! emits a one-time `tracing::warn!` naming the failed op and the underlying
//! error the first time it happens (an `AtomicBool` latch per op, so a
//! persistently-unavailable Metal device does not spam the log on every one of
//! the VAE's per-layer conv/norm/silu/upsample calls) — see
//! `warn_gpu_fallback_once`. A GPU failure can still never break a decode
//! (no `unwrap`/`expect`/`panic!`).
//!
//! ## Test-only conv-dispatch override (T-Missed-1)
//!
//! `vae_gpu_enabled` used to cache its env read in `VAE_GPU_ENABLED` (a
//! `OnceLock`) with no way to un-stick it, so whichever test in this crate's
//! one-process `cargo test` binary called it *first* fixed the answer for
//! every later test regardless of what they set `OXI_VAE_GPU` to — the exact
//! poisoning that made `vae::tiling`'s bit-exact tiled-vs-untiled tests flaky
//! (they need the CPU path; if the GPU path already won the race, its
//! shape-dependent f32 reassociation produces 1-2 ulp deltas against the CPU
//! reference). `cfg(test)`-only machinery below (`set_conv_override`) lets a
//! test force (or explicitly decline to force) the dispatch decision for its
//! own duration without touching the process environment at all, so it can
//! never stick past that one test.
//!
//! The override itself is a **thread-local** (`TEST_CONV_OVERRIDE`), not a
//! process-global atomic. An earlier version of this fix used a process-wide
//! `AtomicI8` guarded by a `Mutex` so that only one dispatch-forcing test could
//! run at a time — but the mutex only serialized the tests that opted into
//! taking it. `cargo test`'s default harness still runs *every* other test
//! concurrently on its own thread, and several of them (e.g.
//! `vae::resnet::tests::forward_tiled_is_bit_identical_across_budgets`,
//! `vae::decoder::tests::up_block_forward_tiled_matches_forward`) call
//! `vae_gpu_enabled` indirectly (via `GroupNorm::forward_inplace` /
//! `silu_inplace`) without ever taking the lock. While a `vae::tiling` test
//! held the process-global override at `Some(false)`, one of those bystander
//! tests running concurrently on another thread would read the *same* global
//! atomic mid-invocation — observing `Some(false)` for some of its several
//! GroupNorm/SiLU calls and the real (GPU) default for others — corrupting its
//! own bit-identity comparison with 1-ulp deltas, a new nondeterministic
//! failure with the same shape as the one T-Missed-1 set out to fix. A
//! thread-local closes this precisely: `cargo test` spawns a fresh OS thread
//! per test function (never a shared pool), and this crate's `vae` module tree
//! contains no `rayon`/`par_iter`/`thread::spawn` of its own, so the dispatch
//! decision for any given call is always read on the same thread that a test
//! (if any) set the override on — a value set by one test's thread is
//! therefore structurally unobservable from any other test's thread, with no
//! lock required.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use oxibonsai_kernels::{MetalGraph, MetalGraphError};

/// An error from the GPU VAE primitive path. The caller converts this into a
/// silent CPU fallback, so it never propagates out of a decode.
#[derive(Debug, thiserror::Error)]
pub enum VaeGpuError {
    /// The process-wide Metal graph singleton could not be obtained (e.g. no
    /// Metal device, or the device failed to initialise).
    #[error("Metal graph unavailable: {0}")]
    GraphUnavailable(String),
    /// A Metal VAE primitive (weight upload / encode / dispatch) failed.
    #[error("Metal VAE primitive failed: {0}")]
    Metal(#[from] MetalGraphError),
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

/// Cached runtime toggle for the GPU VAE path. Default **ON** when the `metal`
/// feature is compiled (the GPU convs/norms/silu/upsample are a parity-proven
/// win — 3.2× over CPU, gated cos ≥ 0.999); set env `OXI_VAE_GPU=0` to force the
/// CPU path (for A/B parity testing without recompiling).
static VAE_GPU_ENABLED: OnceLock<bool> = OnceLock::new();

thread_local! {
    /// Test-only conv-dispatch override consulted by `vae_gpu_enabled` *before*
    /// `VAE_GPU_ENABLED`'s `OnceLock` — see the module docs' T-Missed-1
    /// section for why this is thread-local rather than a process-global
    /// atomic. `None` = unset (fall through to the real env-cached default);
    /// `Some(false)`/`Some(true)` force the CPU/GPU path, but only for
    /// `vae_gpu_enabled` calls made on *this* thread.
    #[cfg(test)]
    static TEST_CONV_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// RAII guard returned by `set_conv_override`. Dropping it clears this
/// thread's `TEST_CONV_OVERRIDE` back to `None` regardless of what value was
/// set, so a later `vae_gpu_enabled` call on this same thread always falls
/// through to the real env-cached default again once the guard goes out of
/// scope.
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
/// guard.
///
/// - `Some(false)` forces the CPU path — for a bit-exact tiled-vs-untiled
///   comparison that must not silently run on the GPU's shape-dependent f32
///   reassociation (see `crate::vae::tiling`'s tests).
/// - `Some(true)` forces the GPU path.
/// - `None` clears this thread's override (a fresh thread-local already
///   starts at `None`; passing it explicitly documents that the caller wants
///   the real env-derived default without depending on this test running
///   first).
///
/// No lock is needed: the override lives in a thread-local, so a
/// concurrently-running test on a different thread can never observe or be
/// affected by this thread's value (see the module docs' T-Missed-1 section).
/// The returned `ConvOverrideGuard` must still be held (bound to a named
/// variable, not discarded with a bare `;`) for as long as the calling test
/// needs the override to apply — dropping it early un-forces the dispatch (on
/// this thread) before the test body that depends on it has finished running.
#[cfg(test)]
pub(crate) fn set_conv_override(v: Option<bool>) -> ConvOverrideGuard {
    TEST_CONV_OVERRIDE.with(|c| c.set(v));
    ConvOverrideGuard
}

/// Read this thread's `TEST_CONV_OVERRIDE`. Always `None` outside
/// `#[cfg(test)]` — see the sibling definition below for non-test builds.
#[cfg(test)]
fn test_conv_override() -> Option<bool> {
    TEST_CONV_OVERRIDE.with(std::cell::Cell::get)
}

/// Non-test builds have no override machinery at all (it is `#[cfg(test)]`
/// only), so this is always `None` — `vae_gpu_enabled` then always falls
/// through to the real `VAE_GPU_ENABLED` env-cached default.
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
/// In test builds this first consults the `set_conv_override` test hook
/// (see the module docs' T-Missed-1 section); when a test holds an active
/// override this returns it directly without ever touching `VAE_GPU_ENABLED`,
/// so a forced value cannot get stuck in the `OnceLock` for later tests.
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
/// the caller (`conv.rs` / `norm.rs` / `ops.rs`, none of which see this
/// warning fire directly — it happens here, at the point of failure, before
/// the `Err` is even returned to them) falls back to the CPU reference path.
///
/// `flag.swap(true, ..)` returns the *previous* value, so the message prints
/// exactly once per process per call site — every subsequent failure (e.g. a
/// GPU that stays unavailable for the life of the process) is silently
/// dropped, matching a "warn once" contract without spamming the log on a
/// hot per-layer decode loop. Mirrors the already-shipped latch pattern in
/// `crate::gpu` (the DiT Metal backend), swapping its `eprintln!` (written
/// before this crate depended on `tracing`) for `tracing::warn!` now that it
/// does.
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
/// [`VaeGpuError`] if the Metal graph is unavailable or the kernel
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
) -> Result<ConvGpuOut, VaeGpuError> {
    let graph = MetalGraph::global()
        .map_err(|e| VaeGpuError::GraphUnavailable(e.to_string()))
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
/// [`VaeGpuError`] if the Metal graph is unavailable or the kernel
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
) -> Result<(), VaeGpuError> {
    let graph = MetalGraph::global()
        .map_err(|e| VaeGpuError::GraphUnavailable(e.to_string()))
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
/// [`VaeGpuError`] if the Metal graph is unavailable or the kernel
/// upload/encode fails. The caller falls back to the CPU path on any error (a
/// one-time `tracing::warn!` fires first — see `warn_gpu_fallback_once`).
pub fn silu_gpu(x: &mut [f32]) -> Result<(), VaeGpuError> {
    let graph = MetalGraph::global()
        .map_err(|e| VaeGpuError::GraphUnavailable(e.to_string()))
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
/// [`VaeGpuError`] if the Metal graph is unavailable or the kernel
/// upload/encode fails. The caller falls back to the CPU path on any error (a
/// one-time `tracing::warn!` fires first — see `warn_gpu_fallback_once`).
pub fn upsample_gpu(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
) -> Result<UpsampleGpuOut, VaeGpuError> {
    let graph = MetalGraph::global()
        .map_err(|e| VaeGpuError::GraphUnavailable(e.to_string()))
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
        // Clears any override on this thread (a fresh thread-local already
        // starts at `None`; this is just explicit). Thread-local storage means
        // a concurrently-running forced-CPU/forced-GPU test on another thread
        // can never affect this read — see the module docs' T-Missed-1 section.
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
        // A private, test-local latch (never touched by any other test or by
        // the real GPU call sites), so this is deterministic regardless of
        // process/thread scheduling — unlike the crate-wide GPU-fallback
        // latches, this one is fully test-owned.
        static LOCAL_WARNED: AtomicBool = AtomicBool::new(false);
        assert!(
            !LOCAL_WARNED.load(Ordering::Relaxed),
            "fresh static must start false"
        );
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "first reason");
        assert!(
            LOCAL_WARNED.load(Ordering::Relaxed),
            "the first call must latch the flag"
        );
        // A second (and third) call must not panic, and must leave the latch
        // set — this is what makes the diagnostic "once per process" rather
        // than spamming the log on every per-layer VAE call.
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "second reason");
        warn_gpu_fallback_once(&LOCAL_WARNED, "test op", "third reason");
        assert!(LOCAL_WARNED.load(Ordering::Relaxed));
    }
}
