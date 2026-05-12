//! 1-bit, ternary, FP8, Q4_0, and Q8_0 Linear layer implementations.
//!
//! Wraps the kernel GEMV/GEMM operations with a unified layer abstraction
//! that dispatches to the appropriate quantization-specific kernel.

use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::GpuWeightHandle;

use crate::error::ModelResult;

// Re-export standard quant types so callers can use `layers::linear::LinearQ4_0`.
pub use crate::layers::linear_kquant_ext::{LinearQ5K, LinearQ6K};
pub use crate::layers::linear_kquant_full::{LinearQ2K, LinearQ3K, LinearQ4K, LinearQ8K};
pub use crate::layers::linear_standard::{LinearQ4_0, LinearQ8_0};

/// A linear layer with Q1\_0\_g128 (1-bit) weights.
///
/// Computes `output = weights @ input` (without bias — Qwen3 has no bias).
/// The kernel dispatcher is stored in the struct (mirroring [`LinearTernary`])
/// so that `forward_vec` and `forward_mat` need no per-call kernel argument.
#[derive(Debug)]
pub struct Linear1Bit<'a> {
    /// Weight blocks in row-major order: [out_features × (in_features / 128)] blocks.
    blocks: &'a [BlockQ1_0G128],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns, must be multiple of 128).
    in_features: usize,
    /// GPU-resident weight handle, populated after [`upload_to_gpu()`](Self::upload_to_gpu).
    gpu_handle: Option<GpuWeightHandle>,
    /// Kernel dispatcher stored in the layer (no per-call kernel arg needed).
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
}

impl<'a> Linear1Bit<'a> {
    /// Create a 1-bit linear layer, validating block count at construction.
    ///
    /// - `blocks`: Q1\_0\_g128 weight blocks in row-major order.
    /// - `out_features`: Number of output features.
    /// - `in_features`: Number of input features (must be multiple of 128).
    /// - `kernel`: Kernel dispatcher for 1-bit GEMV/GEMM.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::ShapeMismatch`] if `in_features % 128 != 0`
    /// or `blocks.len() != out_features * (in_features / 128)`.
    pub fn new(
        blocks: &'a [BlockQ1_0G128],
        out_features: usize,
        in_features: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    ) -> crate::error::ModelResult<Self> {
        use crate::error::ModelError;

        if in_features == 0 || in_features % 128 != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "Linear1Bit".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let expected_blocks = out_features * (in_features / 128);
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "Linear1Bit".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
            gpu_handle: None,
            kernel,
        })
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Raw block references (for fused weight concatenation).
    pub fn blocks(&self) -> &[BlockQ1_0G128] {
        self.blocks
    }

    /// Access the GPU-resident weight handle, if uploaded.
    pub fn gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.gpu_handle
    }

    /// Upload weights to GPU memory if the kernel tier supports caching.
    ///
    /// After a successful upload, all subsequent [`forward_vec`](Self::forward_vec)
    /// calls will use the GPU-resident buffer instead of copying weights
    /// every time.
    pub fn upload_to_gpu(&mut self) {
        self.gpu_handle = self.kernel.upload_weights(self.blocks);
    }

    /// Forward pass: vector input (GEMV).
    ///
    /// Uses the stored kernel dispatcher — no per-call kernel argument required.
    /// Routes through `gemv_adaptive` (rayon row-parallel) for the uncached
    /// fallback, mirroring [`LinearTernary::forward`].
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    pub fn forward_vec(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        // Try the cached GPU path first (no host→device weight copy).
        if let Some(handle) = self.gpu_handle {
            if self
                .kernel
                .gemv_cached(handle, input, output, self.out_features, self.in_features)
                .is_ok()
            {
                return Ok(());
            }
        }
        // Fallback: adaptive dispatch (direct / parallel-row / parallel-tiled).
        oxibonsai_kernels::gemv_adaptive(
            &self.kernel,
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(crate::error::ModelError::Kernel)?;
        Ok(())
    }

    /// Forward pass: matrix input (GEMM) for batched/prefill operation.
    ///
    /// Uses the stored kernel dispatcher — no per-call kernel argument required.
    ///
    /// - `input`: Row-major FP32 matrix [m × in_features].
    /// - `output`: Row-major FP32 matrix [m × out_features].
    /// - `m`: Batch/sequence dimension.
    pub fn forward_mat(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        self.kernel
            .gemm(
                self.blocks,
                input,
                output,
                m,
                self.out_features,
                self.in_features,
            )
            .map_err(crate::error::ModelError::Kernel)?;
        Ok(())
    }

    /// Input dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }
}

/// A linear layer with TQ2\_0\_g128 (ternary) weights.
///
/// Computes `output = weights @ input` using ternary GEMV/GEMM kernels.
/// Unlike `Linear1Bit`, the kernel is stored in the struct and validation
/// is performed at construction time, returning an error on shape mismatch.
#[derive(Debug)]
pub struct LinearTernary<'a> {
    /// Weight blocks in row-major order: [out_features × (in_features / 128)] blocks.
    blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns, must be multiple of 128).
    in_features: usize,
    /// GPU-resident weight handle (SoA layout), populated after [`upload_to_gpu`](Self::upload_to_gpu).
    gpu_handle: Option<GpuWeightHandle>,
    /// Kernel dispatcher stored in the layer (ternary path requires no per-call kernel arg).
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
}

impl<'a> LinearTernary<'a> {
    /// Create a ternary linear layer, validating block count at construction.
    ///
    /// - `blocks`: TQ2\_0\_g128 weight blocks in row-major order.
    /// - `out_features`: Number of output features.
    /// - `in_features`: Number of input features (must be multiple of 128).
    /// - `kernel`: Kernel dispatcher for ternary GEMV/GEMM.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::ShapeMismatch`] if `in_features % 128 != 0`
    /// or `blocks.len() != out_features * (in_features / 128)`.
    pub fn new(
        blocks: &'a [oxibonsai_core::BlockTQ2_0_g128],
        out_features: usize,
        in_features: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    ) -> crate::error::ModelResult<Self> {
        use crate::error::ModelError;

        if in_features == 0 || in_features % 128 != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearTernary".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let expected_blocks = out_features * (in_features / 128);
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearTernary".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
            gpu_handle: None,
            kernel,
        })
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features (columns).
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Raw block references (for weight inspection).
    pub fn blocks(&self) -> &[oxibonsai_core::BlockTQ2_0_g128] {
        self.blocks
    }

    /// Access the GPU-resident weight handle, if uploaded.
    pub fn gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.gpu_handle
    }

    /// Upload ternary weights to GPU memory if the kernel tier supports caching.
    ///
    /// After a successful upload, all subsequent [`forward`](Self::forward) calls will use
    /// the GPU-resident buffer instead of copying weights every time.
    pub fn upload_to_gpu(&mut self) {
        use oxibonsai_kernels::TernaryKernel;
        self.gpu_handle = self.kernel.upload_weights_ternary(self.blocks);
    }

    /// Forward pass (GEMV): single input vector.
    ///
    /// Tries the GPU-cached path first; falls back to adaptive CPU SIMD.
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> crate::error::ModelResult<()> {
        use oxibonsai_kernels::TernaryKernel;
        // Try the cached GPU path first (no host→device weight copy).
        if let Some(handle) = self.gpu_handle {
            if self
                .kernel
                .gemv_ternary_g128_cached(
                    handle,
                    input,
                    output,
                    self.out_features,
                    self.in_features,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
        // Fallback: adaptive dispatch (direct / parallel-row / parallel-tiled).
        oxibonsai_kernels::gemv_adaptive_ternary(
            &self.kernel,
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(crate::error::ModelError::Kernel)?;
        Ok(())
    }

    /// Forward pass (GEMM): batched input.
    ///
    /// - `input`: Row-major FP32 matrix [batch × in_features].
    /// - `output`: Row-major FP32 matrix [batch × out_features].
    /// - `batch`: Batch/sequence dimension.
    pub fn forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
    ) -> crate::error::ModelResult<()> {
        oxibonsai_kernels::gemm_adaptive_ternary(
            &self.kernel,
            self.blocks,
            input,
            output,
            batch,
            self.out_features,
            self.in_features,
        )
        .map_err(crate::error::ModelError::Kernel)?;
        Ok(())
    }
}

/// A linear layer with FP8 E4M3FN (8-bit float) weights.
///
/// Computes `output = weights @ input` using FP8 GEMV/GEMM kernels.
/// Each block holds 32 weights (QK_FP8 = 32) + one FP16 scale.
#[derive(Debug)]
pub struct LinearFP8E4M3<'a> {
    blocks: &'a [oxibonsai_core::BlockFP8E4M3],
    out_features: usize,
    in_features: usize,
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
}

impl<'a> LinearFP8E4M3<'a> {
    /// Create an FP8 E4M3FN linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::ShapeMismatch`] if `in_features % QK_FP8 != 0`
    /// or `blocks.len() != out_features * (in_features / QK_FP8)`.
    pub fn new(
        blocks: &'a [oxibonsai_core::BlockFP8E4M3],
        out_features: usize,
        in_features: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    ) -> crate::error::ModelResult<Self> {
        use crate::error::ModelError;
        use oxibonsai_core::QK_FP8;

        if in_features == 0 || in_features % QK_FP8 != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearFP8E4M3".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_FP8;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearFP8E4M3".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
            kernel,
        })
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features (columns).
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Raw FP8 E4M3FN block references.
    pub fn blocks(&self) -> &[oxibonsai_core::BlockFP8E4M3] {
        self.blocks
    }

    /// Forward pass: vector input (GEMV).
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> crate::error::ModelResult<()> {
        oxibonsai_kernels::gemv_fp8_e4m3_par(
            &self.kernel,
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(crate::error::ModelError::Kernel)
    }

    /// Forward pass: matrix input (GEMM) for batched/prefill operation.
    ///
    /// - `input`: Row-major FP32 matrix [batch × in_features].
    /// - `output`: Row-major FP32 matrix [batch × out_features].
    /// - `batch`: Batch/sequence dimension.
    pub fn forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
    ) -> crate::error::ModelResult<()> {
        oxibonsai_kernels::gemm_fp8_e4m3_par(
            &self.kernel,
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
            batch,
        )
        .map_err(crate::error::ModelError::Kernel)
    }
}

/// A linear layer with FP8 E5M2 (8-bit float) weights.
///
/// Computes `output = weights @ input` using FP8 GEMV/GEMM kernels.
/// Each block holds 32 weights (QK_FP8 = 32) + one FP16 scale.
#[derive(Debug)]
pub struct LinearFP8E5M2<'a> {
    blocks: &'a [oxibonsai_core::BlockFP8E5M2],
    out_features: usize,
    in_features: usize,
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
}

impl<'a> LinearFP8E5M2<'a> {
    /// Create an FP8 E5M2 linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::ShapeMismatch`] if `in_features % QK_FP8 != 0`
    /// or `blocks.len() != out_features * (in_features / QK_FP8)`.
    pub fn new(
        blocks: &'a [oxibonsai_core::BlockFP8E5M2],
        out_features: usize,
        in_features: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    ) -> crate::error::ModelResult<Self> {
        use crate::error::ModelError;
        use oxibonsai_core::QK_FP8;

        if in_features == 0 || in_features % QK_FP8 != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearFP8E5M2".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_FP8;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearFP8E5M2".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
            kernel,
        })
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features (columns).
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Raw FP8 E5M2 block references.
    pub fn blocks(&self) -> &[oxibonsai_core::BlockFP8E5M2] {
        self.blocks
    }

    /// Forward pass: vector input (GEMV).
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> crate::error::ModelResult<()> {
        oxibonsai_kernels::gemv_fp8_e5m2_par(
            &self.kernel,
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(crate::error::ModelError::Kernel)
    }

    /// Forward pass: matrix input (GEMM) for batched/prefill operation.
    ///
    /// - `input`: Row-major FP32 matrix [batch × in_features].
    /// - `output`: Row-major FP32 matrix [batch × out_features].
    /// - `batch`: Batch/sequence dimension.
    pub fn forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
    ) -> crate::error::ModelResult<()> {
        oxibonsai_kernels::gemm_fp8_e5m2_par(
            &self.kernel,
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
            batch,
        )
        .map_err(crate::error::ModelError::Kernel)
    }
}

/// Sum type dispatching to Q1\_0\_g128, TQ2\_0\_g128, FP8, Q4_0, Q8_0, Q5_K, or Q6_K linear layers.
#[derive(Debug)]
pub enum LinearLayer<'a> {
    /// 1-bit (Q1\_0\_g128) linear layer.
    OneBit(Linear1Bit<'a>),
    /// Ternary (TQ2\_0\_g128) linear layer.
    Ternary(LinearTernary<'a>),
    /// FP8 E4M3FN (8-bit float) linear layer.
    FP8E4M3(LinearFP8E4M3<'a>),
    /// FP8 E5M2 (8-bit float) linear layer.
    FP8E5M2(LinearFP8E5M2<'a>),
    /// 4-bit symmetric (Q4_0) linear layer.
    Q4_0(LinearQ4_0<'a>),
    /// 8-bit symmetric (Q8_0) linear layer.
    Q8_0(LinearQ8_0<'a>),
    /// 5-bit K-quant (Q5_K) linear layer.
    Q5K(LinearQ5K<'a>),
    /// 6-bit K-quant (Q6_K) linear layer.
    Q6K(LinearQ6K<'a>),
    /// 2-bit K-quant (Q2_K) linear layer.
    Q2K(LinearQ2K<'a>),
    /// 3-bit K-quant (Q3_K) linear layer.
    Q3K(LinearQ3K<'a>),
    /// 4-bit K-quant (Q4_K) linear layer.
    Q4K(LinearQ4K<'a>),
    /// 8-bit K-quant (Q8_K) linear layer.
    Q8K(LinearQ8K<'a>),
}

impl<'a> LinearLayer<'a> {
    /// Number of output features (rows).
    pub fn out_features(&self) -> usize {
        match self {
            Self::OneBit(l) => l.out_features(),
            Self::Ternary(l) => l.out_features(),
            Self::FP8E4M3(l) => l.out_features(),
            Self::FP8E5M2(l) => l.out_features(),
            Self::Q4_0(l) => l.out_features(),
            Self::Q8_0(l) => l.out_features(),
            Self::Q5K(l) => l.out_features(),
            Self::Q6K(l) => l.out_features(),
            Self::Q2K(l) => l.out_features(),
            Self::Q3K(l) => l.out_features(),
            Self::Q4K(l) => l.out_features(),
            Self::Q8K(l) => l.out_features(),
        }
    }

    /// Number of input features (columns).
    pub fn in_features(&self) -> usize {
        match self {
            Self::OneBit(l) => l.in_features(),
            Self::Ternary(l) => l.in_features(),
            Self::FP8E4M3(l) => l.in_features(),
            Self::FP8E5M2(l) => l.in_features(),
            Self::Q4_0(l) => l.in_features(),
            Self::Q8_0(l) => l.in_features(),
            Self::Q5K(l) => l.in_features(),
            Self::Q6K(l) => l.in_features(),
            Self::Q2K(l) => l.in_features(),
            Self::Q3K(l) => l.in_features(),
            Self::Q4K(l) => l.in_features(),
            Self::Q8K(l) => l.in_features(),
        }
    }

    /// Returns the GPU weight handle, if the layer has been uploaded to GPU.
    ///
    /// FP8, Q4_0, Q8_0, Q5_K, Q6_K, and K-quant variants do not support GPU caching.
    pub fn gpu_handle(&self) -> Option<oxibonsai_kernels::GpuWeightHandle> {
        match self {
            Self::OneBit(l) => l.gpu_handle(),
            Self::Ternary(l) => l.gpu_handle(),
            Self::FP8E4M3(_)
            | Self::FP8E5M2(_)
            | Self::Q4_0(_)
            | Self::Q8_0(_)
            | Self::Q5K(_)
            | Self::Q6K(_)
            | Self::Q2K(_)
            | Self::Q3K(_)
            | Self::Q4K(_)
            | Self::Q8K(_) => None,
        }
    }

    /// Returns the Q1\_0\_g128 blocks if this is a 1-bit layer, `None` otherwise.
    pub fn blocks_1bit(&self) -> Option<&[oxibonsai_core::tensor::BlockQ1_0G128]> {
        match self {
            Self::OneBit(l) => Some(l.blocks()),
            Self::Ternary(_)
            | Self::FP8E4M3(_)
            | Self::FP8E5M2(_)
            | Self::Q4_0(_)
            | Self::Q8_0(_)
            | Self::Q5K(_)
            | Self::Q6K(_)
            | Self::Q2K(_)
            | Self::Q3K(_)
            | Self::Q4K(_)
            | Self::Q8K(_) => None,
        }
    }

    /// Returns the TQ2\_0\_g128 blocks if this is a ternary layer, `None` otherwise.
    pub fn blocks_ternary(&self) -> Option<&[oxibonsai_core::BlockTQ2_0_g128]> {
        match self {
            Self::Ternary(l) => Some(l.blocks()),
            Self::OneBit(_)
            | Self::FP8E4M3(_)
            | Self::FP8E5M2(_)
            | Self::Q4_0(_)
            | Self::Q8_0(_)
            | Self::Q5K(_)
            | Self::Q2K(_)
            | Self::Q3K(_)
            | Self::Q4K(_)
            | Self::Q8K(_)
            | Self::Q6K(_) => None,
        }
    }

    /// Returns the FP8 E4M3FN blocks if this is an FP8 E4M3 layer, `None` otherwise.
    pub fn blocks_fp8_e4m3(&self) -> Option<&[oxibonsai_core::BlockFP8E4M3]> {
        match self {
            Self::FP8E4M3(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns the FP8 E5M2 blocks if this is an FP8 E5M2 layer, `None` otherwise.
    pub fn blocks_fp8_e5m2(&self) -> Option<&[oxibonsai_core::BlockFP8E5M2]> {
        match self {
            Self::FP8E5M2(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns the Q4_0 blocks if this is a Q4_0 layer, `None` otherwise.
    pub fn blocks_q4_0(&self) -> Option<&[oxibonsai_core::BlockQ4_0]> {
        match self {
            Self::Q4_0(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns the Q8_0 blocks if this is a Q8_0 layer, `None` otherwise.
    pub fn blocks_q8_0(&self) -> Option<&[oxibonsai_core::BlockQ8_0]> {
        match self {
            Self::Q8_0(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns the Q5_K blocks if this is a Q5_K layer, `None` otherwise.
    pub fn blocks_q5k(&self) -> Option<&[oxibonsai_core::BlockQ5K]> {
        match self {
            Self::Q5K(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns the Q6_K blocks if this is a Q6_K layer, `None` otherwise.
    pub fn blocks_q6k(&self) -> Option<&[oxibonsai_core::BlockQ6K]> {
        match self {
            Self::Q6K(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns Q2_K blocks if this is a Q2_K layer, `None` otherwise.
    pub fn blocks_q2k(&self) -> Option<&[oxibonsai_core::BlockQ2K]> {
        match self {
            Self::Q2K(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns Q3_K blocks if this is a Q3_K layer, `None` otherwise.
    pub fn blocks_q3k(&self) -> Option<&[oxibonsai_core::BlockQ3K]> {
        match self {
            Self::Q3K(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns Q4_K blocks if this is a Q4_K layer, `None` otherwise.
    pub fn blocks_q4k(&self) -> Option<&[oxibonsai_core::BlockQ4K]> {
        match self {
            Self::Q4K(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns Q8_K blocks if this is a Q8_K layer, `None` otherwise.
    pub fn blocks_q8k(&self) -> Option<&[oxibonsai_core::BlockQ8K]> {
        match self {
            Self::Q8K(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Upload weights to GPU.
    ///
    /// K-quant variants are no-ops (GPU inference not yet implemented).
    pub fn upload_to_gpu(&mut self) {
        match self {
            Self::OneBit(l) => l.upload_to_gpu(),
            Self::Ternary(l) => l.upload_to_gpu(),
            Self::FP8E4M3(_)
            | Self::FP8E5M2(_)
            | Self::Q4_0(_)
            | Self::Q8_0(_)
            | Self::Q5K(_)
            | Self::Q6K(_)
            | Self::Q2K(_)
            | Self::Q3K(_)
            | Self::Q4K(_)
            | Self::Q8K(_) => {}
        }
    }

    /// Forward pass (GEMV) for a single input vector.
    pub fn forward_vec(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        match self {
            Self::OneBit(l) => l.forward_vec(input, output),
            Self::Ternary(l) => l.forward(input, output),
            Self::FP8E4M3(l) => l.forward(input, output),
            Self::FP8E5M2(l) => l.forward(input, output),
            Self::Q4_0(l) => l.forward(input, output),
            Self::Q8_0(l) => l.forward(input, output),
            Self::Q5K(l) => l.forward(input, output),
            Self::Q6K(l) => l.forward(input, output),
            Self::Q2K(l) => l.forward(input, output),
            Self::Q3K(l) => l.forward(input, output),
            Self::Q4K(l) => l.forward(input, output),
            Self::Q8K(l) => l.forward(input, output),
        }
    }

    /// Forward pass (GEMM) for a batched input.
    pub fn forward_mat(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        match self {
            Self::OneBit(l) => l.forward_mat(input, output, m),
            Self::Ternary(l) => l.forward_batch(input, output, m),
            Self::FP8E4M3(l) => l.forward_batch(input, output, m),
            Self::FP8E5M2(l) => l.forward_batch(input, output, m),
            Self::Q4_0(l) => l.forward_batch(input, output, m),
            Self::Q8_0(l) => l.forward_batch(input, output, m),
            Self::Q5K(l) => l.forward_batch(input, output, m),
            Self::Q6K(l) => l.forward_batch(input, output, m),
            Self::Q2K(l) => l.forward_batch(input, output, m),
            Self::Q3K(l) => l.forward_batch(input, output, m),
            Self::Q4K(l) => l.forward_batch(input, output, m),
            Self::Q8K(l) => l.forward_batch(input, output, m),
        }
    }
}

impl<'a> From<Linear1Bit<'a>> for LinearLayer<'a> {
    fn from(l: Linear1Bit<'a>) -> Self {
        Self::OneBit(l)
    }
}

impl<'a> From<LinearTernary<'a>> for LinearLayer<'a> {
    fn from(l: LinearTernary<'a>) -> Self {
        Self::Ternary(l)
    }
}

impl<'a> From<LinearFP8E4M3<'a>> for LinearLayer<'a> {
    fn from(l: LinearFP8E4M3<'a>) -> Self {
        Self::FP8E4M3(l)
    }
}

impl<'a> From<LinearFP8E5M2<'a>> for LinearLayer<'a> {
    fn from(l: LinearFP8E5M2<'a>) -> Self {
        Self::FP8E5M2(l)
    }
}

impl<'a> From<LinearQ4_0<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ4_0<'a>) -> Self {
        Self::Q4_0(l)
    }
}

impl<'a> From<LinearQ8_0<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ8_0<'a>) -> Self {
        Self::Q8_0(l)
    }
}

impl<'a> From<LinearQ5K<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ5K<'a>) -> Self {
        Self::Q5K(l)
    }
}

impl<'a> From<LinearQ6K<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ6K<'a>) -> Self {
        Self::Q6K(l)
    }
}
impl<'a> From<LinearQ2K<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ2K<'a>) -> Self {
        Self::Q2K(l)
    }
}
impl<'a> From<LinearQ3K<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ3K<'a>) -> Self {
        Self::Q3K(l)
    }
}
impl<'a> From<LinearQ4K<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ4K<'a>) -> Self {
        Self::Q4K(l)
    }
}
impl<'a> From<LinearQ8K<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ8K<'a>) -> Self {
        Self::Q8K(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use oxibonsai_kernels::KernelDispatcher;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn linear_1bit_gemv() {
        // 2 output features, 128 input features
        let blocks = vec![
            make_block(1.0, [0xFF; 16]), // row 0: all +1
            make_block(1.0, [0x00; 16]), // row 1: all -1
        ];
        let kernel = std::sync::Arc::new(KernelDispatcher::auto_detect());
        let layer =
            Linear1Bit::new(&blocks, 2, 128, kernel).expect("linear layer creation should succeed");

        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 2];
        layer
            .forward_vec(&input, &mut output)
            .expect("linear forward should succeed");

        assert!((output[0] - 128.0).abs() < 1.0);
        assert!((output[1] + 128.0).abs() < 1.0);
    }

    #[test]
    fn linear_ternary_forward_all_pos() {
        use oxibonsai_core::BlockTQ2_0_g128;
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        // 0xAA = 0b10101010 → every 2-bit lane is 0b10 → +1 code
        let block = BlockTQ2_0_g128 {
            qs: [0xAAu8; 32],
            d: f16::ONE,
        };
        let blocks = [block];
        let layer = LinearTernary::new(&blocks, 1, 128, kernel).expect("new should succeed");
        let input = vec![1.0f32; 128];
        let mut out = vec![0.0f32; 1];
        layer.forward(&input, &mut out).expect("fwd should succeed");
        // 128 weights × +1 × input 1.0 × scale 1.0 = 128.0
        assert!(
            (out[0] - 128.0).abs() < 1.0,
            "expected ~128, got {}",
            out[0]
        );
    }

    #[test]
    fn linear_ternary_shape_mismatch_is_err() {
        use oxibonsai_core::BlockTQ2_0_g128;
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        let block = BlockTQ2_0_g128 {
            qs: [0xAAu8; 32],
            d: f16::ONE,
        };
        // out=2, in=128 needs 2 blocks, but only 1 supplied
        let blocks = [block];
        let result = LinearTernary::new(&blocks, 2, 128, kernel);
        assert!(result.is_err(), "should error on wrong block count");
    }

    #[test]
    fn linear_1bit_new_validates_shape() {
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        // out=2, in=128 needs 2 blocks, but only 1 supplied
        let block = make_block(1.0, [0xFF; 16]);
        let blocks = [block];
        let result = Linear1Bit::new(&blocks, 2, 128, kernel.clone());
        assert!(result.is_err(), "should error on wrong block count");

        // in_features not a multiple of 128 — also invalid
        let result_bad_in = Linear1Bit::new(&blocks, 1, 64, kernel);
        assert!(
            result_bad_in.is_err(),
            "should error when in_features % 128 != 0"
        );
    }
}
