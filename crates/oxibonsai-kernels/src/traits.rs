//! Trait definitions for 1-bit, ternary, and FP8 compute kernels.
//!
//! [`OneBitKernel`] is the common interface implemented by every kernel tier
//! (reference, AVX2, AVX-512, NEON). The [`KernelDispatcher`](crate::KernelDispatcher)
//! implements this trait and delegates to the best available tier at runtime.

use crate::error::KernelResult;
use crate::weight_cache::GpuWeightHandle;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_core::{
    BlockFP8E4M3, BlockFP8E5M2, BlockPQ2_0, BlockPTQ1_0, BlockQ2K, BlockQ2_0G64, BlockQ3K,
    BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0,
};

/// Trait for Q1\_0\_g128 compute kernel implementations.
///
/// Each tier (reference, portable SIMD, platform SIMD) implements this trait.
pub trait OneBitKernel: Send + Sync {
    /// Dequantize blocks to FP32 values.
    ///
    /// For each block: `output[i] = bit[i] ? +d : -d`
    fn dequant(&self, blocks: &[BlockQ1_0G128], output: &mut [f32]) -> KernelResult<()>;

    /// Fused 1-bit matrix × FP32 vector product (GEMV).
    ///
    /// Computes `output[row] = sum_col(weight[row, col] * input[col])`
    /// where weights are Q1\_0\_g128 packed.
    ///
    /// - `blocks`: Row-major packed weight blocks, `n_rows * (k / 128)` blocks total
    /// - `input`: FP32 input vector of length `k`
    /// - `output`: FP32 output vector of length `n_rows`
    /// - `n_rows`: Number of output rows (N dimension)
    /// - `k`: Inner dimension (must be multiple of 128)
    fn gemv(
        &self,
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused 1-bit matrix × FP32 matrix product (GEMM).
    ///
    /// Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])`
    ///
    /// - `blocks`: Weight blocks in row-major order, `n_rows * (k / 128)` blocks
    /// - `input`: Row-major FP32 input [m × k]
    /// - `output`: Row-major FP32 output [m × n_rows]
    /// - `m`: Batch/sequence dimension
    /// - `n_rows`: Number of weight matrix rows (output columns)
    /// - `k`: Inner dimension (must be multiple of 128)
    fn gemm(
        &self,
        blocks: &[BlockQ1_0G128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Display name for this kernel implementation.
    fn name(&self) -> &'static str;

    /// Whether this kernel routes ops through GPU hardware.
    ///
    /// CPU-only tiers (Reference, AVX2, AVX-512, NEON) return `false`. The
    /// GPU tier returns `true`. Higher-level code (e.g. `BonsaiModel::forward`)
    /// uses this to decide whether to take fused-GPU shortcuts that bypass
    /// the per-block kernel calls.
    fn is_gpu_accelerated(&self) -> bool {
        false
    }

    /// Upload weight blocks to GPU memory for future cached GEMV/GEMM calls.
    ///
    /// Returns `Some(handle)` if the kernel supports GPU caching (i.e. the
    /// GPU tier), or `None` for CPU-only tiers.
    fn upload_weights(&self, _blocks: &[BlockQ1_0G128]) -> Option<GpuWeightHandle> {
        None
    }

    /// GEMV using a pre-uploaded weight buffer (no host→device copy for weights).
    ///
    /// Falls back to `Err(UnsupportedOperation)` by default; only the GPU tier
    /// overrides this.
    fn gemv_cached(
        &self,
        _handle: GpuWeightHandle,
        _input: &[f32],
        _output: &mut [f32],
        _n_rows: usize,
        _k: usize,
    ) -> KernelResult<()> {
        Err(crate::error::KernelError::UnsupportedOperation(
            "gemv_cached not supported by this kernel tier".into(),
        ))
    }

    /// Batch-accelerated attention input phase (RMSNorm + QKV in one command buffer).
    ///
    /// Returns `Ok(Some((q, k, v)))` if batching succeeded, or `Ok(None)` if
    /// not supported by this kernel tier.
    ///
    /// # K-20: hidden, not dead
    ///
    /// `KernelDispatcher`'s implementation of this method (`dispatch.rs`)
    /// always returns `Ok(None)` for a measured, documented reason (CPU
    /// RMSNorm + one fused GEMV beats a GPU batch of only two operations),
    /// so this trait method is currently unreachable from any live call
    /// site. It is `#[doc(hidden)]` rather than removed because
    /// `gpu_backend::scirs2_backend::Scirs2Backend`'s implementation is a
    /// real, working ~120-line Metal batch-attention path wired into
    /// `GpuBackendTrait` — deleting the trait method would delete that live
    /// backend code along with the `block/types/forward_sw.rs` /
    /// `forward_stats.rs` call sites that are its intended revival point
    /// for a future fused Bonsai 2 forward path. See the dispatcher-side
    /// note next to `dispatch.rs`'s `batch_attn_phase` impl.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn batch_attn_phase(
        &self,
        _hidden: &[f32],
        _norm_weight: &[f32],
        _norm_eps: f32,
        _qkv_handle: GpuWeightHandle,
        _q_rows: usize,
        _k_rows: usize,
        _h: usize,
    ) -> KernelResult<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
        Ok(None)
    }

    /// Batch-accelerated FFN phase (attn_proj + residual + norm + gate_up + swiglu + down + residual).
    ///
    /// Returns `Ok(true)` if batching succeeded and `hidden` was modified
    /// in-place, or `Ok(false)` if not supported.
    #[allow(clippy::too_many_arguments)]
    fn batch_ffn_phase(
        &self,
        _hidden: &mut [f32],
        _attn_out: &[f32],
        _norm_weight: &[f32],
        _norm_eps: f32,
        _attn_proj_handle: GpuWeightHandle,
        _gate_up_handle: GpuWeightHandle,
        _down_handle: GpuWeightHandle,
        _h: usize,
        _intermediate: usize,
        _attn_proj_k: usize,
    ) -> KernelResult<bool> {
        Ok(false)
    }
}

/// Ternary ({-1, 0, +1}) weight matrix kernel operations.
///
/// Parallel to [`OneBitKernel`] for TQ2\_0\_g128-format weight matrices.
/// Each kernel tier (reference, AVX2, AVX-512, NEON) implements this trait,
/// and [`crate::KernelDispatcher`] delegates to the best available tier.
pub trait TernaryKernel: Send + Sync {
    /// Dequantize TQ2\_0\_g128 blocks to FP32 values.
    ///
    /// For each block: `output[i] = scale * ternary_code[i]`
    /// where codes map as `0b00→-1`, `0b01→0`, `0b10→+1`, `0b11→0`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::KernelError::BufferTooSmall`] if `output` is shorter
    /// than `blocks.len() * 128`.
    fn dequant_ternary_g128(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
        output: &mut [f32],
    ) -> KernelResult<()>;

    /// Fused ternary matrix × FP32 vector product (GEMV).
    ///
    /// Computes `output[row] = sum_col(weight[row, col] * input[col])`
    /// where weights are TQ2\_0\_g128 packed.
    ///
    /// - `blocks`: Row-major packed weight blocks, `n_rows * (k / 128)` blocks total.
    /// - `input`: FP32 input vector of length `k`.
    /// - `output`: FP32 output vector of length `n_rows`.
    /// - `n_rows`: Number of output rows (N dimension).
    /// - `k`: Inner dimension (must be multiple of 128).
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `k % 128 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
    /// - [`crate::error::KernelError::BufferTooSmall`] if `output` is too short.
    fn gemv_ternary_g128(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused ternary matrix × FP32 matrix product (GEMM).
    ///
    /// Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])`
    ///
    /// - `blocks`: Weight blocks in row-major order, `n_rows * (k / 128)` blocks.
    /// - `input`: Row-major FP32 input [m × k].
    /// - `output`: Row-major FP32 output [m × n\_rows].
    /// - `m`: Batch/sequence dimension.
    /// - `n_rows`: Number of weight matrix rows (output columns).
    /// - `k`: Inner dimension (must be multiple of 128).
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `k % 128 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if dimensions mismatch.
    /// - [`crate::error::KernelError::BufferTooSmall`] if any buffer is too small.
    fn gemm_ternary_g128(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Upload TQ2_0_g128 weight blocks to GPU memory for future cached GEMV calls.
    ///
    /// Returns `Some(handle)` if the kernel supports GPU caching (the GPU tier),
    /// or `None` for CPU-only tiers.
    fn upload_weights_ternary(
        &self,
        _blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Option<crate::weight_cache::GpuWeightHandle> {
        None
    }

    /// GEMV using a pre-uploaded ternary weight buffer (no host→device copy for weights).
    ///
    /// Falls back to `Err(UnsupportedOperation)` by default; only the GPU tier
    /// overrides this.
    fn gemv_ternary_g128_cached(
        &self,
        _handle: crate::weight_cache::GpuWeightHandle,
        _input: &[f32],
        _output: &mut [f32],
        _n_rows: usize,
        _k: usize,
    ) -> KernelResult<()> {
        Err(crate::error::KernelError::UnsupportedOperation(
            "gemv_ternary_g128_cached not supported by this kernel tier".into(),
        ))
    }
}

/// FP8 (E4M3FN and E5M2) weight matrix kernel operations.
///
/// Parallel to [`TernaryKernel`] and [`OneBitKernel`] for FP8-quantized weight
/// matrices. Each block holds 32 weights (one byte each) plus a FP16 block
/// scale. The dequantized weight at slot `i` in block `b` is:
/// `d_b × fp8_decode(qs_b[i])`.
///
/// All tiers initially route to the scalar reference implementation. SIMD
/// specializations are a follow-on Slice.
pub trait Fp8Kernel: Send + Sync {
    /// Dequantize FP8 E4M3FN blocks to FP32 values.
    ///
    /// For each block and each slot `i`:
    /// `output[b * QK_FP8 + i] = block.d × fp8_e4m3_decode(block.qs[i])`
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::KernelError::BufferTooSmall`] if
    /// `output.len() < blocks.len() * QK_FP8`.
    fn dequant_fp8_e4m3(&self, blocks: &[BlockFP8E4M3], output: &mut [f32]) -> KernelResult<()>;

    /// Dequantize FP8 E5M2 blocks to FP32 values.
    ///
    /// For each block and each slot `i`:
    /// `output[b * QK_FP8 + i] = block.d × fp8_e5m2_decode(block.qs[i])`
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::KernelError::BufferTooSmall`] if
    /// `output.len() < blocks.len() * QK_FP8`.
    fn dequant_fp8_e5m2(&self, blocks: &[BlockFP8E5M2], output: &mut [f32]) -> KernelResult<()>;

    /// Fused FP8 E4M3FN matrix × FP32 vector product (GEMV).
    ///
    /// Computes `output[row] = dot(weight_row[row], input)` using FP8 E4M3FN
    /// quantized weights.
    ///
    /// - `blocks`: Row-major packed weight blocks, `n_rows * (k / QK_FP8)` blocks total.
    /// - `input`: FP32 input vector of length `k`.
    /// - `output`: FP32 output vector of length `n_rows`.
    /// - `n_rows`: Number of output rows (N dimension).
    /// - `k`: Inner dimension (must be multiple of `QK_FP8 = 32`).
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
    /// - [`crate::error::KernelError::BufferTooSmall`] if `output` is too short.
    fn gemv_fp8_e4m3(
        &self,
        blocks: &[BlockFP8E4M3],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused FP8 E5M2 matrix × FP32 vector product (GEMV).
    ///
    /// Same contract as [`Self::gemv_fp8_e4m3`] but for E5M2-quantized weights.
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
    /// - [`crate::error::KernelError::BufferTooSmall`] if `output` is too short.
    fn gemv_fp8_e5m2(
        &self,
        blocks: &[BlockFP8E5M2],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused FP8 E4M3FN matrix × FP32 matrix product (GEMM).
    ///
    /// Computes `output[b, r] = dot(weight_row[r], input[b])` for all
    /// batch rows `b` and weight rows `r`.
    ///
    /// - `blocks`: Weight blocks in row-major order, `n_rows * (k / QK_FP8)` blocks.
    /// - `inputs`: Row-major FP32 input \[batch × k\].
    /// - `outputs`: Row-major FP32 output \[batch × n\_rows\].
    /// - `n_rows`: Number of weight matrix rows.
    /// - `k`: Inner dimension (must be multiple of `QK_FP8 = 32`).
    /// - `batch`: Batch/sequence dimension.
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if dimensions mismatch.
    /// - [`crate::error::KernelError::BufferTooSmall`] if any buffer is too small.
    fn gemm_fp8_e4m3(
        &self,
        blocks: &[BlockFP8E4M3],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch: usize,
    ) -> KernelResult<()>;

    /// Fused FP8 E5M2 matrix × FP32 matrix product (GEMM).
    ///
    /// Same contract as [`Self::gemm_fp8_e4m3`] but for E5M2-quantized weights.
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `k % QK_FP8 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if dimensions mismatch.
    /// - [`crate::error::KernelError::BufferTooSmall`] if any buffer is too small.
    fn gemm_fp8_e5m2(
        &self,
        blocks: &[BlockFP8E5M2],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch: usize,
    ) -> KernelResult<()>;

    /// Display name for this FP8 kernel implementation.
    fn name_fp8(&self) -> &'static str {
        "fp8_reference"
    }
}

/// A kernel tier that supports **both** 1-bit and ternary fused GPU weight
/// handles (M-21).
///
/// `TransformerBlock::upload_to_gpu` (`block/types/upload.rs`)
/// currently takes `&dyn OneBitKernel`, so ternary QKV/
/// gate-up fusion cannot reach `TernaryKernel::upload_weights_ternary` — a
/// `dyn` trait object cannot gain a second trait's methods after the fact.
/// This marker trait plus its blanket impl below gives any concrete
/// dispatcher (in practice, always [`crate::KernelDispatcher`], which
/// implements every kernel trait) a single type implementing both, so a
/// `&dyn FusedKernel` can call either trait's methods. Widening
/// `upload_to_gpu`/`upload_weights_to_gpu`'s parameter type from
/// `&dyn OneBitKernel` to `&dyn FusedKernel` is the remaining step and
/// belongs to whichever package owns `block/types/upload.rs`.
pub trait FusedKernel: OneBitKernel + TernaryKernel {}

impl<T: OneBitKernel + TernaryKernel> FusedKernel for T {}

/// Standard GGUF quant-format (Q4_0, Q8_0) weight matrix kernel operations.
///
/// Parallel to [`OneBitKernel`] / [`TernaryKernel`] / [`Fp8Kernel`] for the two
/// most common distributed GGUF weight formats. [`crate::KernelDispatcher`]
/// implements this and delegates each GEMV to the best available CPU tier
/// (AVX-512 / AVX2 SIMD decode + FMA on x86-64, scalar reference otherwise).
///
/// The free functions `crate::gemv_q4_0` / `crate::gemv_q8_0` route through
/// this trait (plus Rayon row-parallelism), so callers get tiered SIMD without
/// constructing a dispatcher themselves.
pub trait StandardQuantKernel: Send + Sync {
    /// Fused Q4_0 matrix × FP32 vector product (GEMV).
    ///
    /// `output[row] = dot(weight_row[row], input)` for a row-major Q4_0 weight
    /// matrix (`n_rows * (in_features / QK_Q4_0)` blocks).
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `in_features % QK_Q4_0 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
    /// - [`crate::error::KernelError::BufferTooSmall`] if `output` is too short.
    fn gemv_q4_0(
        &self,
        blocks: &[BlockQ4_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;

    /// Fused Q8_0 matrix × FP32 vector product (GEMV).
    ///
    /// Same contract as [`Self::gemv_q4_0`] but for Q8_0 (8-bit) weights.
    fn gemv_q8_0(
        &self,
        blocks: &[BlockQ8_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;

    /// Fused Q2_K matrix × FP32 vector product (GEMV).
    ///
    /// `output[row] = dot(weight_row[row], input)` for a row-major Q2_K
    /// weight matrix (`n_rows * (in_features / 256)` blocks, `QK_K = 256`).
    /// Same error contract as [`Self::gemv_q4_0`] with `QK_K` (256) in place
    /// of `QK_Q4_0` (32).
    fn gemv_q2k(
        &self,
        blocks: &[BlockQ2K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;

    /// Fused Q3_K matrix × FP32 vector product (GEMV). See [`Self::gemv_q2k`].
    fn gemv_q3k(
        &self,
        blocks: &[BlockQ3K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;

    /// Fused Q4_K matrix × FP32 vector product (GEMV). See [`Self::gemv_q2k`].
    fn gemv_q4k(
        &self,
        blocks: &[BlockQ4K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;

    /// Fused Q5_K matrix × FP32 vector product (GEMV). See [`Self::gemv_q2k`].
    fn gemv_q5k(
        &self,
        blocks: &[BlockQ5K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;

    /// Fused Q6_K matrix × FP32 vector product (GEMV). See [`Self::gemv_q2k`].
    fn gemv_q6k(
        &self,
        blocks: &[BlockQ6K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;

    /// Fused Q8_K matrix × FP32 vector product (GEMV). See [`Self::gemv_q2k`].
    fn gemv_q8k(
        &self,
        blocks: &[BlockQ8K],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        in_features: usize,
    ) -> KernelResult<()>;
}

/// PrismML Bonsai 2 quantization-format and hybrid-math kernel operations
/// (design doc §2.7).
///
/// Covers the three new quant formats (`PQ2_0` ggml id 142, `PTQ1_0` id 143,
/// mainline group-64 `Q2_0` — disambiguated from the legacy group-128
/// `TQ2_0_g128` by [`oxibonsai_core::gguf::quant_resolve`]) plus the
/// Gated-DeltaNet / Hadamard / SSM primitives the Bonsai 2 27B hybrid
/// forward pass needs. [`crate::KernelDispatcher`] implements this and
/// delegates each op to the best available tier, exactly like
/// [`OneBitKernel`] / [`TernaryKernel`] / [`StandardQuantKernel`] / [`Fp8Kernel`].
///
/// Every non-GEMV/GEMM method here (`fwht_*`, `gdn_*`, `conv1d_*`,
/// `l2_norm`, `rms_norm_gated`, `sigmoid_mul`, `softplus`,
/// `rope_partial_splithalf`) wraps a free function in `oxibonsai-kernels`
/// (`hadamard`, `gated_delta_net`, `gated_delta_net_chunk`, `ssm_ops`,
/// `norms`, `rope_mrope`) that already self-dispatches AArch64 NEON /
/// x86-64 AVX2 internally via `#[cfg(target_arch)]` — there is
/// no separate per-tier sibling to route to yet, so [`crate::KernelDispatcher`]'s
/// implementation calls the one function directly regardless of `self.tier()`
/// today. The trait method still exists (rather than callers reaching for
/// the free function themselves) so a future Metal/CUDA kernel
/// slots in without changing any caller's call site.
pub trait PrismKernel: Send + Sync {
    /// Fused `PQ2_0` matrix × FP32 vector product (GEMV).
    ///
    /// `output[row] = dot(weight_row[row], input)` for a row-major `PQ2_0`
    /// weight matrix (`n_rows * (in_features / 128)` blocks, ggml id 142,
    /// `d` first then `qs[32]`, arithmetic code map `value = code - 1`).
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `in_features % 128 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
    /// - [`crate::error::KernelError::BufferTooSmall`] if `output` is too short.
    fn gemv_pq2_0(
        &self,
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused `PQ2_0` matrix × FP32 matrix product (GEMM): batches
    /// [`Self::gemv_pq2_0`] over `m` input rows.
    fn gemm_pq2_0(
        &self,
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused `PTQ1_0` matrix × FP32 vector product (GEMV).
    ///
    /// `output[row] = dot(weight_row[row], input)` for a row-major `PTQ1_0`
    /// weight matrix (`n_rows * (in_features / 128)` blocks, ggml id 143,
    /// 28-byte base-3 trit-packed block). Same error contract as
    /// [`Self::gemv_pq2_0`].
    fn gemv_ptq1_0(
        &self,
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused `PTQ1_0` matrix × FP32 matrix product (GEMM): batches
    /// [`Self::gemv_ptq1_0`] over `m` input rows.
    fn gemm_ptq1_0(
        &self,
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused mainline group-64 `Q2_0` matrix × FP32 vector product (GEMV).
    ///
    /// `output[row] = dot(weight_row[row], input)` for a row-major group-64
    /// `Q2_0` weight matrix (`n_rows * (in_features / 64)` blocks, `d` first
    /// then `qs[16]`, disambiguated from the legacy group-128
    /// `TQ2_0_g128`/`PQ2_0` readings of ggml id 42 by
    /// [`oxibonsai_core::gguf::quant_resolve::resolve_type_42`]).
    ///
    /// # Errors
    ///
    /// - [`crate::error::KernelError::NotBlockAligned`] if `in_features % 64 != 0`.
    /// - [`crate::error::KernelError::DimensionMismatch`] if `input` or `blocks` are too short.
    /// - [`crate::error::KernelError::BufferTooSmall`] if `output` is too short.
    fn gemv_q2_0_g64(
        &self,
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Fused group-64 `Q2_0` matrix × FP32 matrix product (GEMM): batches
    /// [`Self::gemv_q2_0_g64`] over `m` input rows.
    fn gemm_q2_0_g64(
        &self,
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()>;

    /// Blockwise forward Fast Walsh-Hadamard Transform with a fused sign
    /// flip (design §2.2): `x' = FWHT_blockwise(x ⊙ signs) / sqrt(block)`,
    /// applied independently to each contiguous `block`-wide chunk of `x`.
    /// `signs` is `±1.0` per element, the same length as `x`. In place.
    ///
    /// # Errors
    ///
    /// [`crate::error::KernelError::NamedDimensionMismatch`] if `signs.len()
    /// != x.len()`, or if `block` does not divide `x.len()`, or `block` is
    /// not a power of two.
    fn fwht_forward_signed(&self, x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()>;

    /// Inverse of [`Self::fwht_forward_signed`]: `x' = FWHT_blockwise(x) ⊙
    /// signs / sqrt(block)` — FWHT **then** signs (design §3.5, the token
    /// embedding's inverse-rotation order, the reverse of the forward hook's
    /// signs-then-FWHT). In place.
    fn fwht_inverse_signed(&self, x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()>;

    /// One decode step of the Gated DeltaNet recurrence for every v-head at
    /// once (design §2.3): updates `state`'s slab for `layer` in place and
    /// writes this token's output `o = q·S` (grouped v-head order). Mirrors
    /// [`crate::gated_delta_net::gdn_step`]'s exact parameter order.
    ///
    /// - `q`, `k`: `[n_k_heads][head_k_dim]`, already L2-normalised.
    /// - `v`: `[n_v_heads][head_v_dim]`, tiled v-head order (caller re-indexes
    ///   via `VHeadMap` before calling, per design §3.3).
    /// - `alpha_raw`, `beta_raw`: `[n_v_heads]` pre-activation gate inputs.
    /// - `a_neg`, `dt_bias`: `[n_v_heads]` (`a_neg` = `-exp(A_log)`, already
    ///   negative — rejected at load if positive).
    /// - `state`: holds every linear layer's recurrent state; `layer`
    ///   selects which slab this call updates.
    /// - `out`: `[n_v_heads][head_v_dim]`, grouped v-head order.
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
        state: &mut crate::gated_delta_net::GdnState,
        layer: usize,
        out: &mut [f32],
    ) -> KernelResult<()>;

    /// Chunked prefill of the Gated DeltaNet recurrence over `t_len` tokens —
    /// equivalent to `t_len` sequential [`Self::gdn_step`] calls, bitwise
    /// (the kernel's acceptance criterion), but processes linear layers
    /// sequentially in time within a chunk while batching everything else
    /// (design §3.10). Mirrors [`crate::gated_delta_net_chunk::gdn_chunk`]'s
    /// exact parameter order (same as [`Self::gdn_step`] plus a trailing
    /// `t_len`).
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
        state: &mut crate::gated_delta_net::GdnState,
        layer: usize,
        out: &mut [f32],
        t_len: usize,
    ) -> KernelResult<()>;

    /// One decode step (`n_t = 1`) of the causal depthwise conv1d over every
    /// channel of the concatenated GDN `qkv` stream (design §2.4, `d_conv =
    /// 4`). `state` is `[channels][3]` (oldest tap first), updated in place.
    fn conv1d_decode(
        &self,
        state: &mut [f32],
        x_t: &[f32],
        w: &[f32],
        out: &mut [f32],
    ) -> KernelResult<()>;

    /// Prefill (`n_t` tokens at once) of the causal depthwise conv1d — equal
    /// to `n_t` sequential [`Self::conv1d_decode`] calls, bitwise (the kernel's
    /// acceptance criterion). `conv_x` is the pre-assembled `d_inner x ((KC-1)
    /// + n_t)` window (history taps followed by this chunk's raw
    /// activations — no separate mutable `state`, unlike [`Self::conv1d_decode`]);
    /// `out` is token-major (`n_t x d_inner`).
    fn conv1d_prefill(
        &self,
        conv_x: &[f32],
        w: &[f32],
        out: &mut [f32],
        n_t: usize,
        d_inner: usize,
    ) -> KernelResult<()>;

    /// Per-head L2 normalisation: `output[h] = input[h] / max(||input[h]||,
    /// eps)` for each contiguous `head_dim`-wide chunk of `input` (design
    /// §2.4: GDN's joint q‖k normalisation).
    fn l2_norm(&self, input: &[f32], output: &mut [f32], eps: f32) -> KernelResult<()>;

    /// Gated RMSNorm, fused: `output[i] = weight[i] * input[i] * inv_rms *
    /// silu(gate[i])` (design §2.4, GDN's output gate). `weight` and `gate`
    /// are the same length as `input`.
    fn rms_norm_gated(
        &self,
        input: &[f32],
        weight: &[f32],
        gate: &[f32],
        output: &mut [f32],
        eps: f32,
    ) -> KernelResult<()>;

    /// `output[i] = x[i] * sigmoid(gate[i])` elementwise (design §2.6, the
    /// full-attention layer's sigmoid output gate).
    fn sigmoid_mul(&self, x: &[f32], gate: &[f32], output: &mut [f32]) -> KernelResult<()>;

    /// `output[i] = softplus(input[i])`, cutoff exactly `20.0` (the kernel's
    /// acceptance criterion: above the cutoff, `softplus(x) ≈ x` to avoid
    /// `exp` overflow) — the GDN decay-gate activation (design §2.3).
    fn softplus(&self, input: &[f32], output: &mut [f32]) -> KernelResult<()>;

    /// Partial NeoX-style RoPE for **one** `head_dim`-wide head at one
    /// position, out of place (design §2.5): rotates the first `n_rot` of
    /// `head_dim` dims (`n_rot = 64` of `head_dim = 256` for the Bonsai 2
    /// full-attention layers) and copies dims `n_rot..head_dim` through
    /// verbatim. `cos`/`sin` are the `n_rot/2`-wide table row for this one
    /// position (the caller loops over heads and positions).
    fn rope_partial_splithalf(
        &self,
        input: &[f32],
        output: &mut [f32],
        head_dim: usize,
        n_rot: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> KernelResult<()>;
}

#[cfg(test)]
mod tests {
    use super::FusedKernel;

    /// M-21: `KernelDispatcher` implements both `OneBitKernel` and
    /// `TernaryKernel`, so the blanket impl must make it a `FusedKernel`
    /// too, and a `&dyn FusedKernel` must be usable as `&dyn OneBitKernel`
    /// / `&dyn TernaryKernel` (supertrait upcasting) — the exact shape
    /// `block/types/upload.rs`'s widened `upload_to_gpu` will need.
    #[test]
    fn kernel_dispatcher_is_a_fused_kernel() {
        let dispatcher = crate::dispatch::KernelDispatcher::auto_detect();
        let fused: &dyn FusedKernel = &dispatcher;
        let _as_one_bit: &dyn super::OneBitKernel = fused;
        let _as_ternary: &dyn super::TernaryKernel = fused;
    }
}
