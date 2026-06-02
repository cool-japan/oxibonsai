//! GPU (Metal) backend for the DiT ternary (`TQ2_0_g128`) matmuls.
//!
//! This module routes the 100 ternary Linears of the FLUX.2 Klein DiT forward
//! onto the project's fused Metal TQ2 GEMM kernel (`encode_gemm_tq2` in
//! `oxibonsai-kernels`), keeping the per-weight 2-bit codes resident on the GPU
//! and only crossing the bus with the (small) f32 activations per matmul.
//!
//! The whole module is gated on `cfg(all(feature = "metal", target_os =
//! "macos"))` — the same gate under which `oxibonsai-kernels` re-exports
//! [`MetalGraph`] — so a non-Metal / non-macOS build never references it and the
//! default Pure-Rust CPU path is entirely unaffected.
//!
//! Correctness contract (verified by the Phase-1 kernel unit test for
//! `M ∈ {1, 7, 8, 9, 100}` at ~1e-6 vs CPU): the kernel computes, for the DiT's
//! row-major `input[M, K]`, out-major weight blocks `[N, K]`, and `out[M, N]`,
//! `out[m, n] = Σ_k input[m, k] · dequant(W)[n, k]` — i.e. exactly the same
//! `A·Bᵀ` contraction as the CPU [`crate::gemm::gemm_abt`], with no transpose
//! and the identical AoS block layout the kernel's reformat expects.
//!
//! On *any* error this module returns a [`GpuMatmulError`]; the caller
//! ([`crate::math::ternary_matmul`]) swallows it and falls back to the CPU path,
//! so a GPU failure can never break a forward pass (no `unwrap`/`expect`/`panic!`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;
use oxibonsai_kernels::{MetalGraph, MetalGraphError};

/// An error from the GPU ternary-matmul path. The caller converts this into a
/// silent CPU fallback, so it never propagates out of a forward pass.
#[derive(Debug, thiserror::Error)]
pub enum GpuMatmulError {
    /// The process-wide Metal graph singleton could not be obtained (e.g. no
    /// Metal device, or the device failed to initialise).
    #[error("Metal graph unavailable: {0}")]
    GraphUnavailable(String),
    /// The fused Metal TQ2 GEMM (weight upload / encode / dispatch) failed.
    #[error("Metal TQ2 GEMM failed: {0}")]
    Metal(#[from] MetalGraphError),
}

/// Reinterpret the packed ternary blocks as their raw little-endian AoS bytes.
///
/// `BlockTQ2_0_g128` is `#[repr(C)]` and exactly 34 bytes (`qs[32] ‖ d:f16`),
/// which *is* the AoS layout the kernel's `reformat_tq2_aos_to_soa` consumes, so
/// this byte view is a valid weight upload with no conversion.
fn blocks_as_bytes(blocks: &[BlockTQ2_0_g128]) -> &[u8] {
    debug_assert_eq!(std::mem::size_of::<BlockTQ2_0_g128>(), 34);
    let len = std::mem::size_of_val(blocks);
    // SAFETY: `blocks` is a live `&[BlockTQ2_0_g128]`; `BlockTQ2_0_g128` is
    // `#[repr(C)]` with no padding (size_of == 34, the sum of its fields) and no
    // invalid byte patterns (all-bytes-valid POD: `[u8;32]` + `f16`). The
    // resulting `&[u8]` borrows `blocks` for the same lifetime and is read-only,
    // and `len` is the exact byte size of the slice.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, len) }
}

/// One-time confirmation that the GPU path actually executed at least once
/// (used by the parity example to PROVE the GPU ran, not a silent CPU fallback).
static GPU_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any [`ternary_matmul_gpu`] call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn gpu_was_used() -> bool {
    GPU_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the GPU DiT path. Default ON when the `metal`
/// feature is compiled; set env `OXI_DIT_GPU=0` to force the CPU path (for A/B
/// parity testing without recompiling).
static GPU_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the DiT should use the GPU ternary path.
///
/// `true` unless the environment variable `OXI_DIT_GPU` is set to `0`. The env
/// read is cached in a [`OnceLock`] on first call.
pub fn dit_gpu_enabled() -> bool {
    *GPU_ENABLED.get_or_init(|| !matches!(std::env::var("OXI_DIT_GPU").ok().as_deref(), Some("0")))
}

/// One-time confirmation that the GPU **joint-attention** path actually executed
/// at least once (used by the parity example to PROVE the GPU flash-attention
/// kernel ran, not a silent CPU fallback). Tracked separately from [`GPU_USED`]
/// so the attention contribution can be A/B'd independently of the ternary path.
static DIT_ATTN_GPU_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any [`joint_attention_gpu`] call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn dit_attn_gpu_was_used() -> bool {
    DIT_ATTN_GPU_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the GPU DiT **joint-attention** path. Default
/// **ON** when the `metal` feature is compiled (this flash-attention path is a
/// parity-proven win — DiT sample 1.89× — gated cos ≥ 0.999); set env
/// `OXI_DIT_ATTN_GPU=0` to force the CPU path (for A/B parity testing without
/// recompiling).
static ATTN_GPU_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the DiT should use the GPU flash-attention path for joint attention.
///
/// `true` unless the environment variable `OXI_DIT_ATTN_GPU` is set to `0`. The
/// env read is cached in a [`OnceLock`] on first call. Kept a *separate* toggle
/// from [`dit_gpu_enabled`] (which gates the ternary matmuls) so the attention
/// contribution can be measured independently for A/B parity + timing. The
/// per-op CPU fallback in [`crate::math::joint_attention`] (a silent fall-through
/// on any GPU `Err`) makes default-on safe.
pub fn dit_attn_gpu_enabled() -> bool {
    *ATTN_GPU_ENABLED
        .get_or_init(|| !matches!(std::env::var("OXI_DIT_ATTN_GPU").ok().as_deref(), Some("0")))
}

/// Compute FLUX.2 DiT joint multi-head scaled-dot-product attention on the GPU
/// via the fused Metal flash-attention kernel.
///
/// Mirrors the CPU reference [`crate::math::joint_attention`] exactly: `q`, `k`,
/// `v` are head-major `[num_heads, seq, head_dim]` f32 (RoPE already applied to
/// q,k upstream), and the returned `Vec<f32>` is the token-major attention output
/// `[seq, num_heads * head_dim]` (heads concatenated along the feature axis), with
/// `scale = 1/sqrt(head_dim)` and a non-causal softmax over keys.
///
/// Uses the **pooled** flash entry point
/// ([`MetalGraph::encode_joint_attention_flash_pooled`]), which reuses the
/// process-wide q/k/v/out buffers (zero per-call big allocation after warm-up)
/// and only crosses the bus with the (small) q/k/v upload + out download per
/// call. The CPU↔GPU transfers are negligible at the DiT shape (~0.2 ms), so this
/// captures the full ~5.5× flash-kernel win over the rayon+NEON CPU attention
/// without needing q/k/v residency.
///
/// # Errors
/// Returns [`GpuMatmulError`] if the Metal graph is unavailable or the kernel
/// encode fails (e.g. `head_dim` not a multiple of 8, `head_dim > 128`, or
/// `seq` over the kernel's compile-time cap). The caller
/// ([`crate::math::joint_attention`]) falls back to the CPU path on any error.
pub fn joint_attention_gpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Result<Vec<f32>, GpuMatmulError> {
    let graph =
        MetalGraph::global().map_err(|e| GpuMatmulError::GraphUnavailable(e.to_string()))?;
    // The kernel writes the token-major transposed result `[seq, num_heads*head_dim]`.
    let mut out = vec![0.0f32; seq * num_heads * head_dim];
    graph.encode_joint_attention_flash_pooled(q, k, v, &mut out, num_heads, seq, head_dim)?;
    DIT_ATTN_GPU_USED.store(true, Ordering::Relaxed);
    Ok(out)
}

/// Compute `out[m, n] = input[m, k] · dequant(blocks)[n, k]ᵀ` on the GPU.
///
/// - `blocks`: out-major ternary blocks (`n * (k / 128)` of them), borrowed from
///   the long-lived mmap'd GGUF in `DitWeights`.
/// - `input`: row-major `[m, k]`.
/// - `out`: row-major `[m, n]` (written in full).
///
/// The weight is uploaded to the GPU **once** and cached by its (mmap-stable,
/// per-weight-unique) pointer key, so subsequent steps reuse the resident SoA
/// buffer and only the activations cross the bus.
///
/// # Errors
/// Returns [`GpuMatmulError`] if the Metal graph is unavailable or the kernel
/// upload/encode fails (incl. `k % 128 != 0` or a length mismatch). The caller
/// falls back to the CPU path on any error.
pub fn ternary_matmul_gpu(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GpuMatmulError> {
    let graph =
        MetalGraph::global().map_err(|e| GpuMatmulError::GraphUnavailable(e.to_string()))?;
    // `blocks` is borrowed from the run-long mmap; its base address is stable and
    // unique per weight, so it doubles as a cache key with no per-Linear bookkeeping.
    // Pointer addresses are huge and won't collide with the LLM's small key space.
    let key = blocks.as_ptr() as u64;
    let handle =
        graph.get_or_upload_tq2_weight_soa_lazy(key, || blocks_as_bytes(blocks).to_vec())?;
    graph.encode_gemm_tq2(&handle, input, out, m, n, k)?;
    GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_is_thirty_four_bytes() {
        assert_eq!(std::mem::size_of::<BlockTQ2_0_g128>(), 34);
    }

    #[test]
    fn blocks_as_bytes_length_is_34_per_block() {
        // Two all-zero blocks → 68 bytes, and the view aliases the source.
        let blocks = vec![
            BlockTQ2_0_g128 {
                qs: [0u8; 32],
                d: half::f16::ZERO,
            };
            2
        ];
        let bytes = blocks_as_bytes(&blocks);
        assert_eq!(bytes.len(), 68);
        assert_eq!(bytes.as_ptr() as usize, blocks.as_ptr() as usize);
    }

    #[test]
    fn gpu_enabled_defaults_on_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_DIT_GPU` unset → enabled). It does not mutate the env.
        if std::env::var("OXI_DIT_GPU").is_err() {
            assert!(dit_gpu_enabled());
        }
    }

    #[test]
    fn attn_gpu_enabled_defaults_on_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_DIT_ATTN_GPU` unset → enabled). It does not mutate the env.
        if std::env::var("OXI_DIT_ATTN_GPU").is_err() {
            assert!(dit_attn_gpu_enabled());
        }
    }
}
