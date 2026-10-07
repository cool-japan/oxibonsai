//! [`StandardQuantKernel`] dispatch: Q4_0, Q8_0 (KERN-SOUND) and the six
//! K-quant formats Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K.
//!
//! Split out of `dispatch.rs` purely for file size, and because a single
//! `impl Trait for Type` must live in one place — this crate can define it
//! in any file, so this one holds every `StandardQuantKernel` method rather
//! than only the Q4_0/Q8_0 ones.
//!
//! ## Why the K-quant arms look different from Q4_0/Q8_0's
//!
//! Q4_0/Q8_0 have a real per-tier SIMD kernel for every CPU tier
//! (`simd_q_std_avx2`/`simd_q_std_avx512`/`simd_q_std_neon`), so their
//! `match self.tier` has one arm per tier. The K-quant GEMVs
//! (`crate::gemv_q2k`, …) are each a single free function that already
//! self-dispatches AArch64 NEON internally (K-15's `neon_fused` module in
//! `gemv_q4k.rs`/`gemv_q6k.rs`/`gemv_q8k.rs`) and has no distinct AVX2/
//! AVX-512 sibling to route to — every CPU tier therefore converges on the
//! same call, which is the honest reflection of what exists today, not a
//! shortcut (the fix owed is a
//! **dispatch policy** — GPU-arm-with-CPU-fallback wired through
//! `KernelDispatcher`, replacing the direct call the model layer used to
//! make — not new SIMD kernels).

use oxibonsai_core::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0,
};

use crate::dispatch::KernelDispatcher;
// Every `KernelTier::` use in this file sits under `#[cfg(target_arch =
// "aarch64")]`, `#[cfg(target_arch = "x86_64")]` or `#[cfg(feature =
// "gpu")]` (directly on a match arm, or via an enclosing `cpu_*_fallback`
// fn); on wasm32 with just the `wasm` feature none of those hold, so the
// name would otherwise be an unused import (wasm32 clippy leg). Gate the
// import to match rather than `#[allow(unused_imports)]`,
// which would hide the next variant that stops being used.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "gpu"))]
use crate::dispatch::KernelTier;
use crate::error::KernelResult;
use crate::traits::StandardQuantKernel;

impl StandardQuantKernel for KernelDispatcher {
    fn gemv_q4_0(
        &self,
        blocks: &[BlockQ4_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_q_std_avx512::gemv_q4_0_avx512(
                    blocks,
                    input,
                    output,
                    n_rows,
                    in_features,
                )
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_q_std_avx2::gemv_q4_0_avx2(blocks, input, output, n_rows, in_features)
            },
            // GPU tier: route to the Metal kernel on macOS; fall back to the
            // best CPU SIMD tier (K-17), not scalar, on any failure (no
            // device, shape mismatch, compile error).
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q4_0_BYTES,
                        QK_STD,
                    ) {
                        match crate::gpu_backend::metal_gemv_q4_0(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q4_0", &e),
                        }
                    }
                }
                // K-17: route to the best CPU SIMD tier, not straight to
                // scalar — mirrors the ternary/1-bit GPU arms above.
                Self::cpu_gemv_q4_0_fallback(blocks, input, output, n_rows, in_features)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_q_std_neon::gemv_q4_0_neon(blocks, input, output, n_rows, in_features)
            },
            // No Q4_0 SIMD kernel for other tiers (Reference): use scalar.
            _ => crate::gemv_q4_0::gemv_q4_0_scalar(blocks, input, output, n_rows, in_features),
        }
    }

    fn gemv_q8_0(
        &self,
        blocks: &[BlockQ8_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_q_std_avx512::gemv_q8_0_avx512(
                    blocks,
                    input,
                    output,
                    n_rows,
                    in_features,
                )
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_q_std_avx2::gemv_q8_0_avx2(blocks, input, output, n_rows, in_features)
            },
            // GPU tier: route to the Metal kernel on macOS; fall back to the
            // best CPU SIMD tier (K-17), not scalar, on any failure (no
            // device, shape mismatch, compile error).
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q8_0_BYTES,
                        QK_STD,
                    ) {
                        match crate::gpu_backend::metal_gemv_q8_0(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q8_0", &e),
                        }
                    }
                }
                // K-17: route to the best CPU SIMD tier, not straight to
                // scalar — mirrors the ternary/1-bit GPU arms above.
                Self::cpu_gemv_q8_0_fallback(blocks, input, output, n_rows, in_features)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_q_std_neon::gemv_q8_0_neon(blocks, input, output, n_rows, in_features)
            },
            _ => crate::gemv_q8_0::gemv_q8_0_scalar(blocks, input, output, n_rows, in_features),
        }
    }

    fn gemv_q2k(
        &self,
        blocks: &[BlockQ2K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q2_K_BYTES,
                        QK_K,
                    ) {
                        match crate::gpu_backend::metal_gemv_q2k(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q2_K", &e),
                        }
                    }
                }
                Self::cpu_gemv_q2k_fallback(blocks, input, output, n_rows, in_features)
            }
            // Every CPU tier converges on the one free function — see this
            // module's doc comment.
            _ => crate::gemv_q2k::gemv_q2k(blocks, input, output, n_rows, in_features),
        }
    }

    fn gemv_q3k(
        &self,
        blocks: &[BlockQ3K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q3K_BYTES,
                        QK_K,
                    ) {
                        match crate::gpu_backend::metal_gemv_q3k(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q3_K", &e),
                        }
                    }
                }
                Self::cpu_gemv_q3k_fallback(blocks, input, output, n_rows, in_features)
            }
            _ => crate::gemv_q3k::gemv_q3k(blocks, input, output, n_rows, in_features),
        }
    }

    fn gemv_q4k(
        &self,
        blocks: &[BlockQ4K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q4_K_BYTES,
                        QK_K,
                    ) {
                        match crate::gpu_backend::metal_gemv_q4k(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q4_K", &e),
                        }
                    }
                }
                Self::cpu_gemv_q4k_fallback(blocks, input, output, n_rows, in_features)
            }
            _ => crate::gemv_q4k::gemv_q4k(blocks, input, output, n_rows, in_features),
        }
    }

    fn gemv_q5k(
        &self,
        blocks: &[BlockQ5K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q5K_BYTES,
                        QK_K,
                    ) {
                        match crate::gpu_backend::metal_gemv_q5k(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q5_K", &e),
                        }
                    }
                }
                Self::cpu_gemv_q5k_fallback(blocks, input, output, n_rows, in_features)
            }
            _ => crate::gemv_q5k::gemv_q5k(blocks, input, output, n_rows, in_features),
        }
    }

    fn gemv_q6k(
        &self,
        blocks: &[BlockQ6K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q6K_BYTES,
                        QK_K,
                    ) {
                        match crate::gpu_backend::metal_gemv_q6k(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q6_K", &e),
                        }
                    }
                }
                Self::cpu_gemv_q6k_fallback(blocks, input, output, n_rows, in_features)
            }
            _ => crate::gemv_q6k::gemv_q6k(blocks, input, output, n_rows, in_features),
        }
    }

    fn gemv_q8k(
        &self,
        blocks: &[BlockQ8K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if let Some(bytes) = quant_gpu_bytes(
                        blocks,
                        input,
                        output,
                        n_rows,
                        in_features,
                        oxibonsai_core::BLOCK_Q8K_BYTES,
                        QK_K,
                    ) {
                        match crate::gpu_backend::metal_gemv_q8k(
                            bytes,
                            &input[..in_features],
                            &mut output[..n_rows],
                            n_rows,
                            in_features,
                        ) {
                            Ok(()) => return Ok(()),
                            Err(e) => warn_metal_gemv_fallback("Q8_K", &e),
                        }
                    }
                }
                Self::cpu_gemv_q8k_fallback(blocks, input, output, n_rows, in_features)
            }
            _ => crate::gemv_q8k::gemv_q8k(blocks, input, output, n_rows, in_features),
        }
    }
}

/// Group size shared by Q4_0/Q8_0 ("standard" GGUF quant formats).
#[cfg(all(feature = "metal", target_os = "macos"))]
const QK_STD: usize = 32;

/// Group size shared by every K-quant format (`QK_K` in `ggml-quants.c`).
#[cfg(all(feature = "metal", target_os = "macos"))]
const QK_K: usize = 256;

impl KernelDispatcher {
    /// Route a Q4_0 GEMV onto the best *CPU* SIMD tier (mirrors
    /// `cpu_gemv_ternary` in `dispatch.rs` for the ternary/1-bit formats).
    /// Used by the `KernelTier::Gpu` arm of `StandardQuantKernel::gemv_q4_0`
    /// so a GPU failure (no device, shape rejected by `quant_gpu_bytes`,
    /// shader compile error) degrades to the fastest CPU tier instead of
    /// dropping straight to the scalar reference (K-17).
    #[cfg(feature = "gpu")]
    pub(crate) fn cpu_gemv_q4_0_fallback(
        blocks: &[BlockQ4_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        let tier = Self::cpu_tier();
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(tier);
        // Exhaustive over `KernelTier`, not `_ => scalar` (mirrors the
        // ternary/1-bit `cpu_*` siblings): a new tier variant must fail to
        // compile here, not silently regress to scalar (K-17).
        match tier {
            KernelTier::Reference => {
                crate::gemv_q4_0::gemv_q4_0_scalar(blocks, input, output, n_rows, in_features)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_q_std_avx512::gemv_q4_0_avx512(
                    blocks,
                    input,
                    output,
                    n_rows,
                    in_features,
                )
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_q_std_avx2::gemv_q4_0_avx2(blocks, input, output, n_rows, in_features)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_q_std_neon::gemv_q4_0_neon(blocks, input, output, n_rows, in_features)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                crate::gemv_q4_0::gemv_q4_0_scalar(blocks, input, output, n_rows, in_features)
            }
        }
    }

    /// Route a Q8_0 GEMV onto the best *CPU* SIMD tier — see
    /// `cpu_gemv_q4_0_fallback` (K-17).
    #[cfg(feature = "gpu")]
    pub(crate) fn cpu_gemv_q8_0_fallback(
        blocks: &[BlockQ8_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        let tier = Self::cpu_tier();
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(tier);
        // Exhaustive over `KernelTier` — see `cpu_gemv_q4_0_fallback` above.
        match tier {
            KernelTier::Reference => {
                crate::gemv_q8_0::gemv_q8_0_scalar(blocks, input, output, n_rows, in_features)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_q_std_avx512::gemv_q8_0_avx512(
                    blocks,
                    input,
                    output,
                    n_rows,
                    in_features,
                )
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_q_std_avx2::gemv_q8_0_avx2(blocks, input, output, n_rows, in_features)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_q_std_neon::gemv_q8_0_neon(blocks, input, output, n_rows, in_features)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                crate::gemv_q8_0::gemv_q8_0_scalar(blocks, input, output, n_rows, in_features)
            }
        }
    }

    /// Route a K-quant GEMV onto the CPU (K-17): every tier
    /// converges on the one free function (see this module's doc comment),
    /// so — unlike the Q4_0/Q8_0 fallbacks above — there is no per-tier
    /// match to be exhaustive over; `Self::cpu_tier()` is still consulted
    /// (and recorded under `#[cfg(test)]`) so the K-17 regression tests can
    /// observe that a GPU failure really did route through *some* CPU tier
    /// rather than silently returning without trying.
    #[cfg(feature = "gpu")]
    fn cpu_gemv_q2k_fallback(
        blocks: &[BlockQ2K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(Self::cpu_tier());
        crate::gemv_q2k::gemv_q2k(blocks, input, output, n_rows, in_features)
    }

    /// See [`Self::cpu_gemv_q2k_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemv_q3k_fallback(
        blocks: &[BlockQ3K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(Self::cpu_tier());
        crate::gemv_q3k::gemv_q3k(blocks, input, output, n_rows, in_features)
    }

    /// See [`Self::cpu_gemv_q2k_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemv_q4k_fallback(
        blocks: &[BlockQ4K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(Self::cpu_tier());
        crate::gemv_q4k::gemv_q4k(blocks, input, output, n_rows, in_features)
    }

    /// See [`Self::cpu_gemv_q2k_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemv_q5k_fallback(
        blocks: &[BlockQ5K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(Self::cpu_tier());
        crate::gemv_q5k::gemv_q5k(blocks, input, output, n_rows, in_features)
    }

    /// See [`Self::cpu_gemv_q2k_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemv_q6k_fallback(
        blocks: &[BlockQ6K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(Self::cpu_tier());
        crate::gemv_q6k::gemv_q6k(blocks, input, output, n_rows, in_features)
    }

    /// See [`Self::cpu_gemv_q2k_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemv_q8k_fallback(
        blocks: &[BlockQ8K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()> {
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(Self::cpu_tier());
        crate::gemv_q8k::gemv_q8k(blocks, input, output, n_rows, in_features)
    }
}

/// Reinterpret the leading `n_rows * (in_features / block_size)` quant
/// blocks as a raw byte slice for the Metal GEMV kernels, or `None` if the
/// buffers are too small or `in_features` is not block-aligned (in which
/// case the scalar/CPU reference runs and reports the precise error).
///
/// `block_bytes` is the `#[repr(C)]` size of the block type (18 for `Q4_0`,
/// 34 for `Q8_0`, 84..292 for the K-quant formats); it equals
/// `size_of::<Block>()` so the leading `expected` blocks occupy exactly
/// `expected * block_bytes` contiguous, unpadded bytes. `block_size` is the
/// element count per block (32 for Q4_0/Q8_0, 256 — `QK_K` — for every
/// K-quant format).
#[cfg(all(feature = "metal", target_os = "macos"))]
fn quant_gpu_bytes<'a, T>(
    blocks: &'a [T],
    input: &[f32],
    output: &[f32],
    n_rows: usize,
    in_features: usize,
    block_bytes: usize,
    block_size: usize,
) -> Option<&'a [u8]> {
    if in_features == 0 || !in_features.is_multiple_of(block_size) {
        return None;
    }
    let blocks_per_row = in_features / block_size;
    let expected = n_rows.checked_mul(blocks_per_row)?;
    if blocks.len() < expected || input.len() < in_features || output.len() < n_rows {
        return None;
    }
    debug_assert_eq!(std::mem::size_of::<T>(), block_bytes);
    // SAFETY: `T` is a `#[repr(C)]` quant block whose size equals `block_bytes`
    // with no inter-element padding, so the leading `expected` blocks form a
    // contiguous `expected * block_bytes` byte run within `blocks`. The returned
    // slice borrows `blocks` (lifetime `'a`) and is never longer than it.
    let bytes =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), expected * block_bytes) };
    Some(bytes)
}

/// Log a one-line warning when a Metal GEMV fails and the call falls back to
/// the CPU (K-17: the best CPU SIMD tier for Q4_0/Q8_0, the single shared CPU
/// kernel for each K-quant format), suppressing the benign "no Metal-capable
/// GPU device" case.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn warn_metal_gemv_fallback(format: &str, e: &crate::gpu_backend::MetalGraphError) {
    let msg = e.to_string();
    if !msg.contains("no Metal-capable GPU device") {
        tracing::warn!(error = %e, "Metal {format} GEMV failed, falling back to the CPU kernel");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::KernelDispatcher;

    #[test]
    fn gemv_q2k_reference_tier_matches_free_function() {
        // Minimal end-to-end smoke test: the dispatcher's `Reference` arm
        // must produce the exact same result as calling the free function
        // directly (they are, in fact, the same call). `in_features = 256`
        // (one QK_K block) with `n_rows = 0` needs zero blocks and exercises
        // only the dispatch path, no K-quant fixture needed.
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let blocks: Vec<BlockQ2K> = Vec::new();
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        dispatcher
            .gemv_q2k(&blocks, &input, &mut output, 0, 256)
            .expect("zero-row Q2_K GEMV must succeed trivially");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_gemv_q5k_routes_through_a_cpu_tier() {
        // K-17 for K-quants: a `KernelTier::Gpu` dispatcher must still
        // produce a result (routing through the CPU fallback rather than
        // silently doing nothing) even with no Metal device / a rejected
        // shape.
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let blocks: Vec<BlockQ5K> = Vec::new();
        let input = vec![0.0f32; 256];
        let mut output: Vec<f32> = Vec::new();
        dispatcher
            .gemv_q5k(&blocks, &input, &mut output, 0, 256)
            .expect("zero-row Q5_K GEMV must succeed trivially on the CPU fallback");
    }
}
