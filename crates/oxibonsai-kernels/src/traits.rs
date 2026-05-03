//! Trait definition for 1-bit compute kernels.
//!
//! [`OneBitKernel`] is the common interface implemented by every kernel tier
//! (reference, AVX2, AVX-512, NEON). The [`KernelDispatcher`](crate::KernelDispatcher)
//! implements this trait and delegates to the best available tier at runtime.

use crate::error::KernelResult;
use crate::weight_cache::GpuWeightHandle;
use oxibonsai_core::tensor::BlockQ1_0G128;

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
