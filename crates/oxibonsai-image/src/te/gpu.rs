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
//! `matmul_inner` helper in [`crate::te::forward`]) falls back to the CPU
//! [`crate::gemm::gemm_abt`], so a GPU failure can never break a forward pass
//! (no `unwrap`/`expect`/`panic!`). That fallback used to be silent
//! (RAG-EVAL-IMG-18); `matmul_inner` now latches a one-time `tracing::warn!`
//! at its catch site the first time this fails, naming the backend and the
//! underlying error — see that function's doc for why the warning lives
//! *there* rather than in this module: `te_matmul_gpu` has exactly one
//! caller, so warning at the raise site here too would double-log a single
//! failure.
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

/// Returns `true` once any `te_matmul_gpu` call has succeeded.
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
/// read is cached in a `OnceLock` on first call.
pub fn te_gpu_enabled() -> bool {
    *TE_GPU_ENABLED.get_or_init(|| matches!(std::env::var("OXI_TE_GPU").ok().as_deref(), Some("1")))
}

/// Cached toggle for the GEMM precision on the GPU TE path. The **bf16** kernel
/// is the default (the model is natively bf16, so it is parity-clean — cos ≈ 1.0
/// — and ~3× faster than the f32 kernel on Apple GPUs). Set env
/// `OXI_TE_GEMM_F32=1` to force the bit-exact f32 kernel instead. Only consulted
/// when [`te_gpu_enabled`] is also true.
static TE_GEMM_BF16: OnceLock<bool> = OnceLock::new();

/// Whether the GPU TE path should use the bf16 GEMM kernel (default `true`).
pub fn te_gemm_bf16_enabled() -> bool {
    *TE_GEMM_BF16
        .get_or_init(|| !matches!(std::env::var("OXI_TE_GEMM_F32").ok().as_deref(), Some("1")))
}

/// Returns `true` when a weight should be persisted in the GPU cache across calls.
///
/// This is only safe when the weight pointer is stable (resident weights).
/// For non-resident weights (default `Mlx4bit` no-cache policy), the dequant
/// buffer is freed after use and the allocator recycles the address, so the
/// `as_ptr()` key is unstable. Persisting the key in that case would return a
/// stale GPU buffer on the next GEMM for a different weight at the same recycled
/// address → corrupted conditioning. Non-resident callers must evict after each
/// GEMM to prevent this.
#[inline]
fn should_persist_weight(resident: bool) -> bool {
    resident
}

/// Latch for a failed best-effort eviction of a non-resident weight from the
/// GPU cache (see [`te_matmul_gpu`]'s eviction note below): the eviction
/// itself is never fallback-worthy (the GEMM it follows already succeeded,
/// and a failed eviction only means the next upload for that key overwrites
/// the stale entry instead of finding it already gone), so this is not one of
/// RAG-EVAL-IMG-18's GPU-to-CPU fallback warnings — but it is still worth a
/// one-time diagnostic rather than a fully silent swallow.
static EVICT_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// Emit a one-time `tracing::warn!` the first time a non-resident weight's
/// best-effort GPU-cache eviction fails — see [`EVICT_FALLBACK_WARNED`]'s doc
/// for why this is a diagnostic rather than a GPU→CPU fallback warning.
/// Mirrors the `warn_*_fallback_once` latch idiom used throughout this crate
/// (e.g. `crate::vae::gpu::warn_gpu_fallback_once`) so it is unit-testable the
/// same way, via a caller-supplied `flag`.
fn warn_evict_fallback_once(flag: &'static AtomicBool, reason: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            reason,
            "oxibonsai-image: TE GPU weight-cache eviction failed (non-fatal; the \
             next upload for this key overwrites the stale entry); further \
             occurrences in this process are not logged"
        );
    }
}

/// Compute `out[m, n] = Σ_k input[m, k] · weight[n, k]` (`x · Wᵀ`) on the GPU.
///
/// - `weight`: row-major f32 `[n, k]` (the dequantized TE Linear weight,
///   borrowed from the long-lived [`crate::te::weights::TeWeights`] registry).
/// - `input`: row-major `[m, k]`.
/// - `out`: row-major `[m, n]` (written in full).
/// - `resident`: whether the host weight buffer is held resident (stable
///   `as_ptr()` identity). When `true`, the uploaded GPU buffer is kept in the
///   weight cache across calls (amortizes upload cost across prompts). When
///   `false`, the cache entry is evicted after each GEMM to prevent a stale-handle
///   hazard: the dequant buffer is freed between calls, and the allocator can
///   recycle its address so the next `get_or_upload_f32_weight` for a different
///   weight would collide with the stale key and return the wrong GPU buffer.
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
    resident: bool,
) -> Result<(), TeGpuMatmulError> {
    let graph =
        MetalGraph::global().map_err(|e| TeGpuMatmulError::GraphUnavailable(e.to_string()))?;
    // `weight` is borrowed from the run-long TeWeights registry; its base
    // address is stable and unique per weight when resident, so it doubles as a
    // cache key with no per-Linear bookkeeping. Pointer addresses are huge and
    // won't collide with the DiT's pointer keys or the LLM's small key space.
    let key = weight.as_ptr() as u64;
    let handle = graph.get_or_upload_f32_weight(key, weight)?;
    if te_gemm_bf16_enabled() && graph.bf16_gemm_available() {
        graph.encode_gemm_bf16(&handle, input, out, m, n, k)?;
    } else {
        graph.encode_gemm_f32(&handle, input, out, m, n, k)?;
    }
    // Non-resident weights have an unstable as_ptr() identity: the dequant
    // buffer is freed after this call, and the allocator can recycle the address
    // for a different weight on the next matmul. Evict now so the next
    // get_or_upload_f32_weight for a new weight at that recycled address always
    // gets a fresh upload rather than a stale handle. Eviction failure is
    // non-fatal (the next upload will simply overwrite the stale entry), but a
    // one-time diagnostic still fires (see `warn_evict_fallback_once`) rather
    // than discarding the error completely.
    if !should_persist_weight(resident) {
        if let Err(e) = graph.evict_f32_weight(key) {
            warn_evict_fallback_once(&EVICT_FALLBACK_WARNED, &e.to_string());
        }
    }
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

    #[test]
    fn te_gemm_bf16_enabled_by_default_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_TE_GEMM_F32` unset → bf16 enabled). It does not mutate the env.
        if std::env::var("OXI_TE_GEMM_F32").is_err() {
            assert!(te_gemm_bf16_enabled());
        }
    }

    #[test]
    fn test_should_persist_weight_logic() {
        assert!(
            !should_persist_weight(false),
            "non-resident must not persist"
        );
        assert!(should_persist_weight(true), "resident must persist");
    }

    #[test]
    fn warn_evict_fallback_once_fires_exactly_once_per_flag() {
        // A private, test-local latch (never touched by any other test or by
        // the real eviction call site), so this is deterministic regardless
        // of process/thread scheduling — mirrors the sibling
        // `warn_gpu_fallback_once_fires_exactly_once_per_flag` tests
        // elsewhere in this crate.
        static LOCAL_WARNED: AtomicBool = AtomicBool::new(false);
        assert!(
            !LOCAL_WARNED.load(Ordering::Relaxed),
            "fresh static must start false"
        );
        warn_evict_fallback_once(&LOCAL_WARNED, "first reason");
        assert!(
            LOCAL_WARNED.load(Ordering::Relaxed),
            "the first call must latch the flag"
        );
        // A second (and third) call must not panic, and must leave the latch
        // set — "once per process", not spammed on every non-resident GEMM.
        warn_evict_fallback_once(&LOCAL_WARNED, "second reason");
        warn_evict_fallback_once(&LOCAL_WARNED, "third reason");
        assert!(LOCAL_WARNED.load(Ordering::Relaxed));
    }
}
