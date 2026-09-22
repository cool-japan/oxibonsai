//! [`PrismKernel`] dispatch (B2-09; design doc §2.7): the three new PrismML
//! Bonsai 2 quant-format GEMV/GEMM kernels (`PQ2_0`, `PTQ1_0`, mainline
//! group-64 `Q2_0`) plus the Gated-DeltaNet/Hadamard/SSM hybrid-math
//! primitives (B2-03/04/05/06).
//!
//! Split into its own sibling file for the same reason as
//! `dispatch_std_quant.rs`: a single `impl Trait for Type` cannot itself be
//! split, and keeping every `PrismKernel` method together (rather than
//! stuffing 16 more methods into `dispatch.rs`, which is already at its
//! 2000-line budget) keeps each dispatch file a coherent, reviewable unit.
//!
//! ## GEMV/GEMM tier dispatch
//!
//! `PQ2_0`/`PTQ1_0`/group-64 `Q2_0` each have a real per-tier kernel
//! (scalar reference, NEON, AVX2 — B2-03's accept criteria only required
//! NEON parity, so there is no dedicated AVX-512 kernel; `KernelTier::Avx512`
//! therefore falls through to the AVX2 kernel, which every real AVX-512F CPU
//! also supports). None of the three has a Metal/CUDA kernel yet (B2-15/17
//! future work), so the `KernelTier::Gpu` arm routes straight to the best
//! *CPU* tier via a `cpu_*_fallback` helper — the same K-17 policy
//! `dispatch_std_quant.rs` and `dispatch.rs`'s ternary/1-bit arms use,
//! applied here so a future Metal kernel can slot into the `Gpu` arm without
//! silently regressing to scalar in the meantime.
//!
//! ## The rest (`fwht_*`, `gdn_*`, `conv1d_*`, `l2_norm`, `rms_norm_gated`,
//! `sigmoid_mul`, `softplus`, `rope_partial_splithalf`)
//!
//! See [`crate::traits::PrismKernel`]'s doc comment: each of these already
//! self-dispatches NEON/AVX2/scalar internally, so every tier converges on
//! the one free function.

use oxibonsai_core::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64};

use crate::dispatch::KernelDispatcher;
// Every `KernelTier::` use in this file sits under `#[cfg(target_arch =
// "aarch64")]`, `#[cfg(target_arch = "x86_64")]` or `#[cfg(feature =
// "gpu")]` (directly on a match arm, or via an enclosing `cpu_*_fallback`
// fn); on wasm32 with just the `wasm` feature none of those hold, so the
// name would otherwise be an unused import (wasm32 clippy leg, FIX3-BUILD
// wave 3.5). Gate the import to match rather than `#[allow(unused_imports)]`,
// which would hide the next variant that stops being used.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64", feature = "gpu"))]
use crate::dispatch::KernelTier;
use crate::error::KernelResult;
use crate::gated_delta_net::GdnState;
use crate::traits::PrismKernel;

impl PrismKernel for KernelDispatcher {
    fn gemv_pq2_0(
        &self,
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemv_pq2_0_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemv_pq2_0_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_gemv_pq2_0_fallback(blocks, input, output, n_rows, k),
            _ => crate::dequant_prism::gemv_pq2_0(blocks, input, output, n_rows, k),
        }
    }

    fn gemm_pq2_0(
        &self,
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemm_pq2_0_neon(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemm_pq2_0_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_gemm_pq2_0_fallback(blocks, input, output, m, n_rows, k),
            _ => crate::dequant_prism::gemm_pq2_0(blocks, input, output, m, n_rows, k),
        }
    }

    fn gemv_ptq1_0(
        &self,
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemv_ptq1_0_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemv_ptq1_0_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_gemv_ptq1_0_fallback(blocks, input, output, n_rows, k),
            _ => crate::gemv_ptq1::gemv_ptq1_0(blocks, input, output, n_rows, k),
        }
    }

    fn gemm_ptq1_0(
        &self,
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemm_ptq1_0_neon(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemm_ptq1_0_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_gemm_ptq1_0_fallback(blocks, input, output, m, n_rows, k),
            _ => crate::gemv_ptq1::gemm_ptq1_0(blocks, input, output, m, n_rows, k),
        }
    }

    fn gemv_q2_0_g64(
        &self,
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemv_q2_0_g64_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemv_q2_0_g64_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => Self::cpu_gemv_q2_0_g64_fallback(blocks, input, output, n_rows, k),
            _ => crate::dequant_prism::gemv_q2_0_g64(blocks, input, output, n_rows, k),
        }
    }

    fn gemm_q2_0_g64(
        &self,
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match self.tier() {
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemm_q2_0_g64_neon(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemm_q2_0_g64_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                Self::cpu_gemm_q2_0_g64_fallback(blocks, input, output, m, n_rows, k)
            }
            _ => crate::dequant_prism::gemm_q2_0_g64(blocks, input, output, m, n_rows, k),
        }
    }

    // ── Hybrid math primitives: every tier converges on one self-dispatching
    // free function (see this module's doc comment) — no per-tier match.

    fn fwht_forward_signed(&self, x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()> {
        crate::hadamard::fwht_forward_signed(x, signs, block)
    }

    fn fwht_inverse_signed(&self, x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()> {
        crate::hadamard::fwht_inverse_signed(x, signs, block)
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha_raw: &[f32],
        beta_raw: &[f32],
        a_neg: &[f32],
        dt_bias: &[f32],
        state: &mut GdnState,
        layer: usize,
        out: &mut [f32],
    ) -> KernelResult<()> {
        crate::gated_delta_net::gdn_step(
            q, k, v, alpha_raw, beta_raw, a_neg, dt_bias, state, layer, out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_chunk(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha_raw: &[f32],
        beta_raw: &[f32],
        a_neg: &[f32],
        dt_bias: &[f32],
        state: &mut GdnState,
        layer: usize,
        out: &mut [f32],
        t_len: usize,
    ) -> KernelResult<()> {
        crate::gated_delta_net_chunk::gdn_chunk(
            q, k, v, alpha_raw, beta_raw, a_neg, dt_bias, state, layer, out, t_len,
        )
    }

    fn conv1d_decode(
        &self,
        state: &mut [f32],
        x_t: &[f32],
        w: &[f32],
        out: &mut [f32],
    ) -> KernelResult<()> {
        crate::ssm_ops::causal_conv1d_k4_decode(state, x_t, w, out)
    }

    fn conv1d_prefill(
        &self,
        conv_x: &[f32],
        w: &[f32],
        out: &mut [f32],
        n_t: usize,
        d_inner: usize,
    ) -> KernelResult<()> {
        crate::ssm_ops::causal_conv1d_k4_prefill(conv_x, w, out, n_t, d_inner)
    }

    fn l2_norm(&self, input: &[f32], output: &mut [f32], eps: f32) -> KernelResult<()> {
        crate::norms::l2_norm_simd(input, output, eps)
    }

    fn rms_norm_gated(
        &self,
        input: &[f32],
        weight: &[f32],
        gate: &[f32],
        output: &mut [f32],
        eps: f32,
    ) -> KernelResult<()> {
        crate::norms::rms_norm_gated_simd(input, weight, gate, output, eps)
    }

    fn sigmoid_mul(&self, x: &[f32], gate: &[f32], output: &mut [f32]) -> KernelResult<()> {
        crate::norms::sigmoid_mul_simd(x, gate, output)
    }

    fn softplus(&self, input: &[f32], output: &mut [f32]) -> KernelResult<()> {
        crate::norms::softplus_simd(input, output)
    }

    fn rope_partial_splithalf(
        &self,
        input: &[f32],
        output: &mut [f32],
        head_dim: usize,
        n_rot: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> KernelResult<()> {
        crate::rope_mrope::rope_partial_splithalf_simd(input, output, head_dim, n_rot, cos, sin)
    }
}

impl KernelDispatcher {
    /// Route a `PQ2_0` GEMV onto the best *CPU* SIMD tier (K-17) — see
    /// `dispatch_std_quant.rs`'s `cpu_gemv_q4_0_fallback` for the pattern
    /// this mirrors. Used by `PrismKernel::gemv_pq2_0`'s `KernelTier::Gpu`
    /// arm; there is no Metal/CUDA `PQ2_0` kernel yet (B2-15/17), so this
    /// always resolves to a CPU tier today, but keeps the same shape so a
    /// future Metal attempt slots in ahead of it without changing the trait
    /// method's body shape.
    #[cfg(feature = "gpu")]
    fn cpu_gemv_pq2_0_fallback(
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        let tier = Self::cpu_tier();
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(tier);
        match tier {
            KernelTier::Reference => {
                crate::dequant_prism::gemv_pq2_0(blocks, input, output, n_rows, k)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemv_pq2_0_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemv_pq2_0_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => crate::dequant_prism::gemv_pq2_0(blocks, input, output, n_rows, k),
        }
    }

    /// See [`Self::cpu_gemv_pq2_0_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemm_pq2_0_fallback(
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => {
                crate::dequant_prism::gemm_pq2_0(blocks, input, output, m, n_rows, k)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemm_pq2_0_neon(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemm_pq2_0_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                crate::dequant_prism::gemm_pq2_0(blocks, input, output, m, n_rows, k)
            }
        }
    }

    /// See [`Self::cpu_gemv_pq2_0_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemv_ptq1_0_fallback(
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        let tier = Self::cpu_tier();
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(tier);
        match tier {
            KernelTier::Reference => {
                crate::gemv_ptq1::gemv_ptq1_0(blocks, input, output, n_rows, k)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemv_ptq1_0_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemv_ptq1_0_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => crate::gemv_ptq1::gemv_ptq1_0(blocks, input, output, n_rows, k),
        }
    }

    /// See [`Self::cpu_gemv_pq2_0_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemm_ptq1_0_fallback(
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => {
                crate::gemv_ptq1::gemm_ptq1_0(blocks, input, output, m, n_rows, k)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemm_ptq1_0_neon(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemm_ptq1_0_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => crate::gemv_ptq1::gemm_ptq1_0(blocks, input, output, m, n_rows, k),
        }
    }

    /// See [`Self::cpu_gemv_pq2_0_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemv_q2_0_g64_fallback(
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        let tier = Self::cpu_tier();
        #[cfg(test)]
        crate::dispatch::record_gpu_fallback_tier(tier);
        match tier {
            KernelTier::Reference => {
                crate::dequant_prism::gemv_q2_0_g64(blocks, input, output, n_rows, k)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemv_q2_0_g64_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemv_q2_0_g64_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                crate::dequant_prism::gemv_q2_0_g64(blocks, input, output, n_rows, k)
            }
        }
    }

    /// See [`Self::cpu_gemv_pq2_0_fallback`].
    #[cfg(feature = "gpu")]
    fn cpu_gemm_q2_0_g64_fallback(
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match Self::cpu_tier() {
            KernelTier::Reference => {
                crate::dequant_prism::gemm_q2_0_g64(blocks, input, output, m, n_rows, k)
            }
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_prism_neon::gemm_q2_0_g64_neon(blocks, input, output, m, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 | KernelTier::Avx512 => unsafe {
                crate::simd_prism_avx2::gemm_q2_0_g64_avx2(blocks, input, output, m, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                crate::dequant_prism::gemm_q2_0_g64(blocks, input, output, m, n_rows, k)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn pq2_0_block(scale: f32) -> BlockPQ2_0 {
        // All-zero `qs` decodes to code 0 -> value -1 everywhere (arithmetic
        // map `code - 1`), so a 128-wide all-ones input dots to `-128 * scale`.
        BlockPQ2_0 {
            d: f16::from_f32(scale),
            qs: [0u8; 32],
        }
    }

    #[test]
    fn gemv_pq2_0_reference_tier_matches_dequant_dot() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let blocks = vec![pq2_0_block(1.0)];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        dispatcher
            .gemv_pq2_0(&blocks, &input, &mut output, 1, 128)
            .expect("gemv_pq2_0 should succeed");
        assert!(
            (output[0] + 128.0).abs() < 1e-3,
            "expected -128.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemm_pq2_0_batches_gemv_pq2_0() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let blocks = vec![pq2_0_block(1.0)];
        let input = vec![1.0f32; 256]; // 2 rows of 128
        let mut output = vec![0.0f32; 2];
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut output, 2, 1, 128)
            .expect("gemm_pq2_0 should succeed");
        assert!((output[0] + 128.0).abs() < 1e-3);
        assert!((output[1] + 128.0).abs() < 1e-3);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_gemv_pq2_0_routes_through_a_cpu_tier() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let blocks = vec![pq2_0_block(1.0)];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        dispatcher
            .gemv_pq2_0(&blocks, &input, &mut output, 1, 128)
            .expect("Gpu-tier gemv_pq2_0 must still succeed via the CPU fallback");
        assert!((output[0] + 128.0).abs() < 1e-3);
    }

    #[test]
    fn fwht_forward_signed_involution_round_trips() {
        // Forward then inverse must return the original vector (design
        // §2.2's involution property), on any tier — exercised here on
        // whatever tier this test process auto-detects.
        let dispatcher = KernelDispatcher::auto_detect();
        let original = vec![1.0f32, 2.0, -3.0, 4.5, -0.5, 6.0, 7.0, -8.0];
        let signs = vec![1.0f32, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
        let mut x = original.clone();
        dispatcher
            .fwht_forward_signed(&mut x, &signs, 8)
            .expect("forward should succeed");
        dispatcher
            .fwht_inverse_signed(&mut x, &signs, 8)
            .expect("inverse should succeed");
        for (a, b) in original.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "expected {a}, got {b}");
        }
    }

    #[test]
    fn softplus_matches_cutoff_contract() {
        let dispatcher = KernelDispatcher::auto_detect();
        let input = vec![25.0f32, 0.0];
        let mut output = vec![0.0f32; 2];
        dispatcher
            .softplus(&input, &mut output)
            .expect("softplus should succeed");
        // Above the 20.0 cutoff, softplus(x) == x exactly (B2-05's acceptance
        // criterion).
        assert!((output[0] - 25.0).abs() < 1e-6);
        // softplus(0) == ln(2).
        assert!((output[1] - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_mul_matches_manual_computation() {
        let dispatcher = KernelDispatcher::auto_detect();
        let x = vec![2.0f32, -1.0];
        let gate = vec![0.0f32, 0.0]; // sigmoid(0) = 0.5
        let mut out = vec![0.0f32; 2];
        dispatcher
            .sigmoid_mul(&x, &gate, &mut out)
            .expect("sigmoid_mul should succeed");
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] + 0.5).abs() < 1e-5);
    }
}
