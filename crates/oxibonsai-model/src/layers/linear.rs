//! 1-bit, ternary, FP8, Q4_0, Q8_0, PrismML and dense Linear layer
//! implementations.
//!
//! Wraps the kernel GEMV/GEMM operations with a unified layer abstraction
//! that dispatches to the appropriate quantization-specific kernel.

use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_core::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64};
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::{GpuWeightHandle, PrismKernel};

use crate::error::ModelResult;

// Re-export standard quant types so callers can use `layers::linear::LinearQ4_0`.
pub use crate::layers::linear_dense::LinearDense;
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

        if in_features == 0 || !in_features.is_multiple_of(128) {
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

        if in_features == 0 || !in_features.is_multiple_of(128) {
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

        if in_features == 0 || !in_features.is_multiple_of(QK_FP8) {
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

        if in_features == 0 || !in_features.is_multiple_of(QK_FP8) {
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

/// A linear layer with `PQ2_0` (PrismML Bonsai 2 ternary, ggml id 142)
/// weights (B2-09; design §3.4).
///
/// Computes `output = weights @ input` using the `PQ2_0` GEMV/GEMM kernels.
/// **Not** the Hadamard rotation hook: per design §3.4, that lives in the
/// block forward (it is per-*activation*, memoised across several matmuls
/// sharing one), not inside `LinearLayer::forward` — `input` here is
/// expected to already be rotated when the model this layer belongs to
/// declares `prism.hadamard.*`.
#[derive(Debug)]
pub struct LinearPQ2_0<'a> {
    /// Weight blocks in row-major order: `[out_features × (in_features / 128)]` blocks.
    blocks: &'a [BlockPQ2_0],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns, must be a multiple of 128).
    in_features: usize,
    /// Kernel dispatcher stored in the layer (mirrors [`LinearTernary`]).
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
}

impl<'a> LinearPQ2_0<'a> {
    /// Create a `PQ2_0` linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::ShapeMismatch`] if
    /// `in_features % 128 != 0` or `blocks.len() != out_features *
    /// (in_features / 128)`.
    pub fn new(
        blocks: &'a [BlockPQ2_0],
        out_features: usize,
        in_features: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    ) -> crate::error::ModelResult<Self> {
        use crate::error::ModelError;

        if in_features == 0 || !in_features.is_multiple_of(QK_PQ2_0) {
            // A relationship ("in_features is a nonzero multiple of the
            // block width"), not a tensor whose own dimensions disagree
            // with what was asked for — `ShapeInvariant` (block/functions.rs
            // migrated four analogous sites the same way) says so instead
            // of a `ShapeMismatch` whose `expected`/`actual` would be
            // byte-identical `Vec<usize>`s and convey nothing.
            return Err(ModelError::ShapeInvariant {
                tensor: "LinearPQ2_0".into(),
                expected: format!("in_features a nonzero multiple of {QK_PQ2_0} (QK_PQ2_0)"),
                actual: format!("in_features = {in_features}"),
            });
        }
        let expected_blocks = out_features * (in_features / QK_PQ2_0);
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearPQ2_0".into(),
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

    /// Raw block references.
    pub fn blocks(&self) -> &[BlockPQ2_0] {
        self.blocks
    }

    /// Forward pass (GEMV): single input vector.
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> crate::error::ModelResult<()> {
        self.kernel
            .gemv_pq2_0(
                self.blocks,
                input,
                output,
                self.out_features,
                self.in_features,
            )
            .map_err(crate::error::ModelError::Kernel)
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
        self.kernel
            .gemm_pq2_0(
                self.blocks,
                input,
                output,
                batch,
                self.out_features,
                self.in_features,
            )
            .map_err(crate::error::ModelError::Kernel)
    }
}

/// A linear layer with `PTQ1_0` (PrismML Bonsai 2 1.75-bit, ggml id 143)
/// weights (B2-09; design §3.4). See [`LinearPQ2_0`] for the Hadamard-hook
/// placement note (unchanged here).
#[derive(Debug)]
pub struct LinearPTQ1_0<'a> {
    /// Weight blocks in row-major order: `[out_features × (in_features / 128)]` blocks.
    blocks: &'a [BlockPTQ1_0],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns, must be a multiple of 128).
    in_features: usize,
    /// Kernel dispatcher stored in the layer (mirrors [`LinearTernary`]).
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
}

impl<'a> LinearPTQ1_0<'a> {
    /// Create a `PTQ1_0` linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::ShapeMismatch`] if
    /// `in_features % 128 != 0` or `blocks.len() != out_features *
    /// (in_features / 128)`.
    pub fn new(
        blocks: &'a [BlockPTQ1_0],
        out_features: usize,
        in_features: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    ) -> crate::error::ModelResult<Self> {
        use crate::error::ModelError;

        if in_features == 0 || !in_features.is_multiple_of(QK_PTQ1_0) {
            // See `LinearPQ2_0::new`'s identical branch for why this is a
            // `ShapeInvariant`, not a `ShapeMismatch`.
            return Err(ModelError::ShapeInvariant {
                tensor: "LinearPTQ1_0".into(),
                expected: format!("in_features a nonzero multiple of {QK_PTQ1_0} (QK_PTQ1_0)"),
                actual: format!("in_features = {in_features}"),
            });
        }
        let expected_blocks = out_features * (in_features / QK_PTQ1_0);
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearPTQ1_0".into(),
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

    /// Raw block references.
    pub fn blocks(&self) -> &[BlockPTQ1_0] {
        self.blocks
    }

    /// Forward pass (GEMV): single input vector.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> crate::error::ModelResult<()> {
        self.kernel
            .gemv_ptq1_0(
                self.blocks,
                input,
                output,
                self.out_features,
                self.in_features,
            )
            .map_err(crate::error::ModelError::Kernel)
    }

    /// Forward pass (GEMM): batched input.
    pub fn forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
    ) -> crate::error::ModelResult<()> {
        self.kernel
            .gemm_ptq1_0(
                self.blocks,
                input,
                output,
                batch,
                self.out_features,
                self.in_features,
            )
            .map_err(crate::error::ModelError::Kernel)
    }
}

/// A linear layer with mainline group-64 `Q2_0` (ggml id 42, disambiguated
/// from the legacy group-128 `TQ2_0_g128`/`PQ2_0` readings by
/// [`oxibonsai_core::gguf::quant_resolve::resolve_type_42`]) weights
/// (B2-09; design §3.4). See [`LinearPQ2_0`] for the Hadamard-hook placement
/// note (unchanged here — the `Ternary-Bonsai-2-27B-Q2_0-prism-fork-required`
/// file folds this format under `prism.hadamard.*` too).
#[derive(Debug)]
pub struct LinearQ2_0G64<'a> {
    /// Weight blocks in row-major order: `[out_features × (in_features / 64)]` blocks.
    blocks: &'a [BlockQ2_0G64],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns, must be a multiple of 64).
    in_features: usize,
    /// Kernel dispatcher stored in the layer (mirrors [`LinearTernary`]).
    kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
}

impl<'a> LinearQ2_0G64<'a> {
    /// Create a group-64 `Q2_0` linear layer, validating block count at
    /// construction.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ModelError::ShapeMismatch`] if
    /// `in_features % 64 != 0` or `blocks.len() != out_features *
    /// (in_features / 64)`.
    pub fn new(
        blocks: &'a [BlockQ2_0G64],
        out_features: usize,
        in_features: usize,
        kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    ) -> crate::error::ModelResult<Self> {
        use crate::error::ModelError;

        if in_features == 0 || !in_features.is_multiple_of(QK_Q2_0_G64) {
            // See `LinearPQ2_0::new`'s identical branch for why this is a
            // `ShapeInvariant`, not a `ShapeMismatch`.
            return Err(ModelError::ShapeInvariant {
                tensor: "LinearQ2_0G64".into(),
                expected: format!("in_features a nonzero multiple of {QK_Q2_0_G64} (QK_Q2_0_G64)"),
                actual: format!("in_features = {in_features}"),
            });
        }
        let expected_blocks = out_features * (in_features / QK_Q2_0_G64);
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ2_0G64".into(),
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

    /// Raw block references.
    pub fn blocks(&self) -> &[BlockQ2_0G64] {
        self.blocks
    }

    /// Forward pass (GEMV): single input vector.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> crate::error::ModelResult<()> {
        self.kernel
            .gemv_q2_0_g64(
                self.blocks,
                input,
                output,
                self.out_features,
                self.in_features,
            )
            .map_err(crate::error::ModelError::Kernel)
    }

    /// Forward pass (GEMM): batched input.
    pub fn forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
    ) -> crate::error::ModelResult<()> {
        self.kernel
            .gemm_q2_0_g64(
                self.blocks,
                input,
                output,
                batch,
                self.out_features,
                self.in_features,
            )
            .map_err(crate::error::ModelError::Kernel)
    }
}

/// Sum type dispatching to Q1\_0\_g128, TQ2\_0\_g128, FP8, Q4_0, Q8_0, the
/// K-quants, the PrismML Bonsai 2 formats, or a dense `f32` matrix.
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
    /// PrismML Bonsai 2 `PQ2_0` (ggml id 142) linear layer (B2-09).
    PQ2_0(LinearPQ2_0<'a>),
    /// PrismML Bonsai 2 `PTQ1_0` (ggml id 143) linear layer (B2-09).
    PTQ1_0(LinearPTQ1_0<'a>),
    /// Mainline group-64 `Q2_0` linear layer (B2-09).
    Q2_0G64(LinearQ2_0G64<'a>),
    /// Dense (unquantized) `f32` linear layer (B2-11-FIX, gatekeeper
    /// REQUIRED #5): an `F32` / `F16` / `BF16` matrix, widened to `f32`.
    Dense(LinearDense<'a>),
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
            Self::PQ2_0(l) => l.out_features(),
            Self::PTQ1_0(l) => l.out_features(),
            Self::Q2_0G64(l) => l.out_features(),
            Self::Dense(l) => l.out_features(),
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
            Self::PQ2_0(l) => l.in_features(),
            Self::PTQ1_0(l) => l.in_features(),
            Self::Q2_0G64(l) => l.in_features(),
            Self::Dense(l) => l.in_features(),
        }
    }

    /// Returns the GPU weight handle, if the layer has been uploaded to GPU.
    ///
    /// FP8, Q4_0, Q8_0, Q5_K, Q6_K, K-quant, and PrismML variants do not
    /// support GPU caching.
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
            | Self::Q8K(_)
            | Self::PQ2_0(_)
            | Self::PTQ1_0(_)
            | Self::Q2_0G64(_)
            | Self::Dense(_) => None,
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
            | Self::Q8K(_)
            | Self::PQ2_0(_)
            | Self::PTQ1_0(_)
            | Self::Q2_0G64(_)
            | Self::Dense(_) => None,
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
            | Self::Q6K(_)
            | Self::PQ2_0(_)
            | Self::PTQ1_0(_)
            | Self::Q2_0G64(_)
            | Self::Dense(_) => None,
        }
    }

    /// Returns the `PQ2_0` blocks if this is a `PQ2_0` layer, `None` otherwise.
    pub fn blocks_pq2_0(&self) -> Option<&[BlockPQ2_0]> {
        match self {
            Self::PQ2_0(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns the `PTQ1_0` blocks if this is a `PTQ1_0` layer, `None` otherwise.
    pub fn blocks_ptq1_0(&self) -> Option<&[BlockPTQ1_0]> {
        match self {
            Self::PTQ1_0(l) => Some(l.blocks()),
            _ => None,
        }
    }

    /// Returns the group-64 `Q2_0` blocks if this is a `Q2_0G64` layer,
    /// `None` otherwise.
    pub fn blocks_q2_0_g64(&self) -> Option<&[BlockQ2_0G64]> {
        match self {
            Self::Q2_0G64(l) => Some(l.blocks()),
            _ => None,
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
    /// K-quant and PrismML variants are no-ops (GPU inference not yet
    /// implemented for them).
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
            | Self::Q8K(_)
            | Self::PQ2_0(_)
            | Self::PTQ1_0(_)
            | Self::Q2_0G64(_)
            | Self::Dense(_) => {}
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
            Self::PQ2_0(l) => l.forward(input, output),
            Self::PTQ1_0(l) => l.forward(input, output),
            Self::Q2_0G64(l) => l.forward(input, output),
            Self::Dense(l) => l.forward(input, output),
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
            Self::PQ2_0(l) => l.forward_batch(input, output, m),
            Self::PTQ1_0(l) => l.forward_batch(input, output, m),
            Self::Q2_0G64(l) => l.forward_batch(input, output, m),
            Self::Dense(l) => l.forward_batch(input, output, m),
        }
    }

    /// Returns the dense `f32` weights if this is a dense layer, `None`
    /// otherwise.
    pub fn dense_weights(&self) -> Option<&[f32]> {
        match self {
            Self::Dense(l) => Some(l.weights()),
            _ => None,
        }
    }

    /// The tensor type this layer computes in: the quantization format for
    /// a quantized layer, [`GgufTensorType::F32`] for a dense one (whose
    /// weights are widened to `f32` whatever their on-disk element type).
    pub fn quant_type(&self) -> GgufTensorType {
        match self {
            Self::OneBit(_) => GgufTensorType::Q1_0_g128,
            Self::Ternary(_) => GgufTensorType::TQ2_0_g128,
            Self::FP8E4M3(_) => GgufTensorType::F8_E4M3,
            Self::FP8E5M2(_) => GgufTensorType::F8_E5M2,
            Self::Q4_0(_) => GgufTensorType::Q4_0,
            Self::Q8_0(_) => GgufTensorType::Q8_0,
            Self::Q5K(_) => GgufTensorType::Q5_K,
            Self::Q6K(_) => GgufTensorType::Q6_K,
            Self::Q2K(_) => GgufTensorType::Q2_K,
            Self::Q3K(_) => GgufTensorType::Q3_K,
            Self::Q4K(_) => GgufTensorType::Q4_K,
            Self::Q8K(_) => GgufTensorType::Q8_K,
            Self::PQ2_0(_) => GgufTensorType::PQ2_0,
            Self::PTQ1_0(_) => GgufTensorType::PTQ1_0,
            Self::Q2_0G64(_) => GgufTensorType::Q2_0G64,
            Self::Dense(_) => GgufTensorType::F32,
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

impl<'a> From<LinearPQ2_0<'a>> for LinearLayer<'a> {
    fn from(l: LinearPQ2_0<'a>) -> Self {
        Self::PQ2_0(l)
    }
}

impl<'a> From<LinearPTQ1_0<'a>> for LinearLayer<'a> {
    fn from(l: LinearPTQ1_0<'a>) -> Self {
        Self::PTQ1_0(l)
    }
}

impl<'a> From<LinearQ2_0G64<'a>> for LinearLayer<'a> {
    fn from(l: LinearQ2_0G64<'a>) -> Self {
        Self::Q2_0G64(l)
    }
}

impl<'a> From<LinearDense<'a>> for LinearLayer<'a> {
    fn from(l: LinearDense<'a>) -> Self {
        Self::Dense(l)
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

    // ── B2-09 acceptance: LinearLayer::{PQ2_0,PTQ1_0,Q2_0G64} round-trip a
    // fixture tensor ──────────────────────────────────────────────────────

    /// `PQ2_0` fixture: `qs = 0xFF` (every 2-bit lane LSB-first is `0b11` =
    /// code 3) decodes to `value = code - 1 = +2` at every element (the
    /// arithmetic code map shared by `PQ2_0`/`Q2_0_g64`/`PTQ1_0`, distinct
    /// from the legacy ternary LUT where `0b11` means something else — see
    /// `dequant_prism`'s module doc for why the two must never be confused).
    #[test]
    fn linear_layer_pq2_0_forward_round_trips_a_fixture_tensor() {
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        let block = BlockPQ2_0 {
            d: half::f16::ONE,
            qs: [0xFFu8; 32],
        };
        let blocks = [block];
        let inner = LinearPQ2_0::new(&blocks, 1, 128, kernel).expect("new should succeed");
        let layer: LinearLayer = inner.into();
        assert_eq!(layer.out_features(), 1);
        assert_eq!(layer.in_features(), 128);

        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        layer
            .forward_vec(&input, &mut output)
            .expect("forward_vec should succeed");
        // 128 weights x +2 x input 1.0 x scale 1.0 = 256.0
        assert!(
            (output[0] - 256.0).abs() < 1.0,
            "expected ~256, got {}",
            output[0]
        );

        // Round-trip through `forward_mat` (GEMM) too, batch of 2 identical rows.
        let batched_input = vec![1.0f32; 256];
        let mut batched_output = vec![0.0f32; 2];
        layer
            .forward_mat(&batched_input, &mut batched_output, 2)
            .expect("forward_mat should succeed");
        for (i, v) in batched_output.iter().enumerate() {
            assert!(
                (v - 256.0).abs() < 1.0,
                "batch row {i}: expected ~256, got {v}"
            );
        }
    }

    /// `PTQ1_0` fixture: all-zero `qs`/`qh` decodes every trit to code 0
    /// (`value = -1`), since a zero byte's base-3 digit extraction is zero
    /// at every stage.
    #[test]
    fn linear_layer_ptq1_0_forward_round_trips_a_fixture_tensor() {
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        let block = BlockPTQ1_0 {
            d: half::f16::ONE,
            qs: [0u8; 24],
            qh: [0u8; 2],
        };
        let blocks = [block];
        let inner = LinearPTQ1_0::new(&blocks, 1, 128, kernel).expect("new should succeed");
        let layer: LinearLayer = inner.into();
        assert_eq!(layer.out_features(), 1);
        assert_eq!(layer.in_features(), 128);

        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        layer
            .forward_vec(&input, &mut output)
            .expect("forward_vec should succeed");
        // 128 weights x -1 x input 1.0 x scale 1.0 = -128.0
        assert!(
            (output[0] + 128.0).abs() < 1.0,
            "expected ~-128, got {}",
            output[0]
        );
    }

    /// Group-64 `Q2_0` fixture: same arithmetic code map as `PQ2_0`, one
    /// block covers 64 elements instead of 128.
    #[test]
    fn linear_layer_q2_0_g64_forward_round_trips_a_fixture_tensor() {
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        let block = BlockQ2_0G64 {
            d: half::f16::ONE,
            qs: [0xFFu8; 16],
        };
        let blocks = [block];
        let inner = LinearQ2_0G64::new(&blocks, 1, 64, kernel).expect("new should succeed");
        let layer: LinearLayer = inner.into();
        assert_eq!(layer.out_features(), 1);
        assert_eq!(layer.in_features(), 64);

        let input = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 1];
        layer
            .forward_vec(&input, &mut output)
            .expect("forward_vec should succeed");
        // 64 weights x +2 x input 1.0 x scale 1.0 = 128.0
        assert!(
            (output[0] - 128.0).abs() < 1.0,
            "expected ~128, got {}",
            output[0]
        );
    }

    #[test]
    fn linear_pq2_0_new_validates_shape() {
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        let block = BlockPQ2_0 {
            d: half::f16::ONE,
            qs: [0xFFu8; 32],
        };
        let blocks = [block];
        // out=2, in=128 needs 2 blocks, but only 1 supplied.
        let result = LinearPQ2_0::new(&blocks, 2, 128, kernel.clone());
        assert!(result.is_err(), "should error on wrong block count");
        // in_features not a multiple of 128 — also invalid.
        let result_bad_in = LinearPQ2_0::new(&blocks, 1, 64, kernel);
        assert!(
            result_bad_in.is_err(),
            "should error when in_features % 128 != 0"
        );
    }

    #[test]
    fn linear_q2_0_g64_new_validates_shape() {
        use std::sync::Arc;

        let kernel = Arc::new(KernelDispatcher::auto_detect());
        let block = BlockQ2_0G64 {
            d: half::f16::ONE,
            qs: [0xFFu8; 16],
        };
        let blocks = [block];
        // out=2, in=64 needs 2 blocks, but only 1 supplied.
        let result = LinearQ2_0G64::new(&blocks, 2, 64, kernel.clone());
        assert!(result.is_err(), "should error on wrong block count");
        // in_features not a multiple of 64 — also invalid.
        let result_bad_in = LinearQ2_0G64::new(&blocks, 1, 32, kernel);
        assert!(
            result_bad_in.is_err(),
            "should error when in_features % 64 != 0"
        );
    }
}
