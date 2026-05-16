//! Linear layer implementations for Q2_K, Q3_K, Q4_K, and Q8_K K-quant weight formats.
//!
//! Wraps the scalar GEMV kernels from `oxibonsai-kernels` with a layer abstraction
//! matching the pattern established by `LinearQ5K` and `LinearQ6K` in
//! `linear_kquant_ext.rs`.
//!
//! All types implement:
//! - `new()` — validates dimensions and block count.
//! - `forward()` — single-vector GEMV.
//! - `forward_batch()` — batched GEMM (sequential over batch dimension).
//! - Accessors: `out_features()`, `in_features()`, `blocks()`, `memory_bytes()`.

use oxibonsai_core::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ8K};
use oxibonsai_kernels::{gemv_q2k, gemv_q3k, gemv_q4k, gemv_q8k};

use crate::error::{ModelError, ModelResult};

// ---------------------------------------------------------------------------
// Compile-time size assertions (documenting the SAFETY invariants used in the
// unsafe `from_raw_parts` casts inside the CUDA dispatch paths below).
// ---------------------------------------------------------------------------

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
const _: () =
    assert!(std::mem::size_of::<oxibonsai_core::BlockQ2K>() == oxibonsai_core::BLOCK_Q2_K_BYTES,);
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
const _: () =
    assert!(std::mem::size_of::<oxibonsai_core::BlockQ3K>() == oxibonsai_core::BLOCK_Q3K_BYTES,);
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
const _: () =
    assert!(std::mem::size_of::<oxibonsai_core::BlockQ4K>() == oxibonsai_core::BLOCK_Q4_K_BYTES,);
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
const _: () =
    assert!(std::mem::size_of::<oxibonsai_core::BlockQ8K>() == oxibonsai_core::BLOCK_Q8K_BYTES,);

// ---------------------------------------------------------------------------
// LinearQ2K
// ---------------------------------------------------------------------------

/// A linear layer with Q2_K (2-bit K-quant) weights.
///
/// Computes `output = W × input` using the scalar Q2_K GEMV kernel.
/// `in_features` must be a multiple of 256 (QK_K).
#[derive(Debug)]
pub struct LinearQ2K<'a> {
    /// Q2_K weight blocks in row-major order: `[out_features × (in_features / 256)]` blocks.
    blocks: &'a [BlockQ2K],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns), must be a multiple of 256.
    in_features: usize,
}

impl<'a> LinearQ2K<'a> {
    /// Create a Q2_K linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % 256 != 0`, or
    /// - `blocks.len() != out_features * (in_features / 256)`.
    pub fn new(
        blocks: &'a [BlockQ2K],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        const QK_K: usize = 256;

        if in_features == 0 || in_features % QK_K != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ2K".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_K;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ2K".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
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

    /// Raw Q2_K block references.
    pub fn blocks(&self) -> &[BlockQ2K] {
        self.blocks
    }

    /// Approximate memory footprint of the weight blocks in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * oxibonsai_core::BLOCK_Q2_K_BYTES
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`:  FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q2_K GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ2K is #[repr(C)] with size BLOCK_Q2_K_BYTES (= 84).
            // The compile-time assert above guarantees this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q2_K_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q2k(
                raw,
                input,
                output,
                self.out_features,
                self.in_features,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let msg = format!("{e}");
                    if !msg.contains("no CUDA device") {
                        tracing::warn!(
                            error = %e,
                            "CUDA Q2K GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q2k(
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(ModelError::Kernel)
    }

    /// Forward pass: batched input (GEMM via sequential GEMV).
    ///
    /// - `input`:  Row-major FP32 matrix `[m × in_features]`.
    /// - `output`: Row-major FP32 matrix `[m × out_features]`.
    /// - `m`:      Batch / sequence dimension.
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        for batch in 0..m {
            let input_row = &input[batch * self.in_features..(batch + 1) * self.in_features];
            let output_row =
                &mut output[batch * self.out_features..(batch + 1) * self.out_features];
            self.forward(input_row, output_row)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LinearQ3K
// ---------------------------------------------------------------------------

/// A linear layer with Q3_K (3-bit K-quant) weights.
///
/// Computes `output = W × input` using the scalar Q3_K GEMV kernel.
/// `in_features` must be a multiple of 256 (QK_K).
#[derive(Debug)]
pub struct LinearQ3K<'a> {
    /// Q3_K weight blocks in row-major order: `[out_features × (in_features / 256)]` blocks.
    blocks: &'a [BlockQ3K],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns), must be a multiple of 256.
    in_features: usize,
}

impl<'a> LinearQ3K<'a> {
    /// Create a Q3_K linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % 256 != 0`, or
    /// - `blocks.len() != out_features * (in_features / 256)`.
    pub fn new(
        blocks: &'a [BlockQ3K],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        const QK_K: usize = 256;

        if in_features == 0 || in_features % QK_K != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ3K".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_K;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ3K".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
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

    /// Raw Q3_K block references.
    pub fn blocks(&self) -> &[BlockQ3K] {
        self.blocks
    }

    /// Approximate memory footprint of the weight blocks in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * oxibonsai_core::BLOCK_Q3K_BYTES
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`:  FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q3_K GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ3K is #[repr(C)] with size BLOCK_Q3K_BYTES (= 110).
            // The compile-time assert above guarantees this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q3K_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q3k(
                raw,
                input,
                output,
                self.out_features,
                self.in_features,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let msg = format!("{e}");
                    if !msg.contains("no CUDA device") {
                        tracing::warn!(
                            error = %e,
                            "CUDA Q3K GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q3k(
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(ModelError::Kernel)
    }

    /// Forward pass: batched input (GEMM via sequential GEMV).
    ///
    /// - `input`:  Row-major FP32 matrix `[m × in_features]`.
    /// - `output`: Row-major FP32 matrix `[m × out_features]`.
    /// - `m`:      Batch / sequence dimension.
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        for batch in 0..m {
            let input_row = &input[batch * self.in_features..(batch + 1) * self.in_features];
            let output_row =
                &mut output[batch * self.out_features..(batch + 1) * self.out_features];
            self.forward(input_row, output_row)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LinearQ4K
// ---------------------------------------------------------------------------

/// A linear layer with Q4_K (4-bit K-quant) weights.
///
/// Computes `output = W × input` using the scalar Q4_K GEMV kernel.
/// `in_features` must be a multiple of 256 (QK_K).
#[derive(Debug)]
pub struct LinearQ4K<'a> {
    /// Q4_K weight blocks in row-major order: `[out_features × (in_features / 256)]` blocks.
    blocks: &'a [BlockQ4K],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns), must be a multiple of 256.
    in_features: usize,
}

impl<'a> LinearQ4K<'a> {
    /// Create a Q4_K linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % 256 != 0`, or
    /// - `blocks.len() != out_features * (in_features / 256)`.
    pub fn new(
        blocks: &'a [BlockQ4K],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        const QK_K: usize = 256;

        if in_features == 0 || in_features % QK_K != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ4K".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_K;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ4K".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
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

    /// Raw Q4_K block references.
    pub fn blocks(&self) -> &[BlockQ4K] {
        self.blocks
    }

    /// Approximate memory footprint of the weight blocks in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * oxibonsai_core::BLOCK_Q4_K_BYTES
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`:  FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q4_K GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ4K is #[repr(C)] with size BLOCK_Q4_K_BYTES (= 144).
            // The compile-time assert above guarantees this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q4_K_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q4k(
                raw,
                input,
                output,
                self.out_features,
                self.in_features,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let msg = format!("{e}");
                    if !msg.contains("no CUDA device") {
                        tracing::warn!(
                            error = %e,
                            "CUDA Q4K GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q4k(
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(ModelError::Kernel)
    }

    /// Forward pass: batched input (GEMM via sequential GEMV).
    ///
    /// - `input`:  Row-major FP32 matrix `[m × in_features]`.
    /// - `output`: Row-major FP32 matrix `[m × out_features]`.
    /// - `m`:      Batch / sequence dimension.
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        for batch in 0..m {
            let input_row = &input[batch * self.in_features..(batch + 1) * self.in_features];
            let output_row =
                &mut output[batch * self.out_features..(batch + 1) * self.out_features];
            self.forward(input_row, output_row)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LinearQ8K
// ---------------------------------------------------------------------------

/// A linear layer with Q8_K (8-bit K-quant) weights.
///
/// Computes `output = W × input` using the scalar Q8_K GEMV kernel.
/// `in_features` must be a multiple of 256 (QK_K).
///
/// Note: Q8_K uses f32 (not f16) for the per-block scale, providing higher
/// precision than other K-quant formats at the cost of slightly larger blocks.
#[derive(Debug)]
pub struct LinearQ8K<'a> {
    /// Q8_K weight blocks in row-major order: `[out_features × (in_features / 256)]` blocks.
    blocks: &'a [BlockQ8K],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns), must be a multiple of 256.
    in_features: usize,
}

impl<'a> LinearQ8K<'a> {
    /// Create a Q8_K linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % 256 != 0`, or
    /// - `blocks.len() != out_features * (in_features / 256)`.
    pub fn new(
        blocks: &'a [BlockQ8K],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        const QK_K: usize = 256;

        if in_features == 0 || in_features % QK_K != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ8K".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_K;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ8K".into(),
                expected: vec![expected_blocks],
                actual: vec![blocks.len()],
            });
        }
        Ok(Self {
            blocks,
            out_features,
            in_features,
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

    /// Raw Q8_K block references.
    pub fn blocks(&self) -> &[BlockQ8K] {
        self.blocks
    }

    /// Approximate memory footprint of the weight blocks in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * oxibonsai_core::BLOCK_Q8K_BYTES
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`:  FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q8_K GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ8K is #[repr(C)] with size BLOCK_Q8K_BYTES (= 292).
            // The compile-time assert above guarantees this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q8K_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q8k(
                raw,
                input,
                output,
                self.out_features,
                self.in_features,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let msg = format!("{e}");
                    if !msg.contains("no CUDA device") {
                        tracing::warn!(
                            error = %e,
                            "CUDA Q8K GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q8k(
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )
        .map_err(ModelError::Kernel)
    }

    /// Forward pass: batched input (GEMM via sequential GEMV).
    ///
    /// - `input`:  Row-major FP32 matrix `[m × in_features]`.
    /// - `output`: Row-major FP32 matrix `[m × out_features]`.
    /// - `m`:      Batch / sequence dimension.
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        for batch in 0..m {
            let input_row = &input[batch * self.in_features..(batch + 1) * self.in_features];
            let output_row =
                &mut output[batch * self.out_features..(batch + 1) * self.out_features];
            self.forward(input_row, output_row)?;
        }
        Ok(())
    }
}
