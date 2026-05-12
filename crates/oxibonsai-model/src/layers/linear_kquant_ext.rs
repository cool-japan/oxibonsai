//! Linear layer implementations for Q5_K and Q6_K K-quant weight formats.
//!
//! Wraps the scalar GEMV kernels from `oxibonsai-kernels` with a layer abstraction
//! matching the pattern established by `LinearFP8E4M3` in `linear.rs`.
//!
//! Both types implement:
//! - `new()` — validates dimensions and block count.
//! - `forward()` — single-vector GEMV.
//! - `forward_batch()` — batched GEMM (sequential over batch dimension).
//! - Accessors: `out_features()`, `in_features()`, `blocks()`.

use oxibonsai_core::{BlockQ5K, BlockQ6K};
use oxibonsai_kernels::{gemv_q5k, gemv_q6k};

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
    assert!(std::mem::size_of::<oxibonsai_core::BlockQ5K>() == oxibonsai_core::BLOCK_Q5K_BYTES,);
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
const _: () =
    assert!(std::mem::size_of::<oxibonsai_core::BlockQ6K>() == oxibonsai_core::BLOCK_Q6K_BYTES,);

// ---------------------------------------------------------------------------
// LinearQ5K
// ---------------------------------------------------------------------------

/// A linear layer with Q5_K (5-bit K-quant) weights.
///
/// Computes `output = W × input` using the scalar Q5_K GEMV kernel.
/// `in_features` must be a multiple of 256 (QK_K).
#[derive(Debug)]
pub struct LinearQ5K<'a> {
    /// Q5_K weight blocks in row-major order: `[out_features × (in_features / 256)]` blocks.
    blocks: &'a [BlockQ5K],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns), must be a multiple of 256.
    in_features: usize,
}

impl<'a> LinearQ5K<'a> {
    /// Create a Q5_K linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % 256 != 0`, or
    /// - `blocks.len() != out_features * (in_features / 256)`.
    pub fn new(
        blocks: &'a [BlockQ5K],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        const QK_K: usize = 256;

        if in_features == 0 || in_features % QK_K != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ5K".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_K;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ5K".into(),
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

    /// Raw Q5_K block references.
    pub fn blocks(&self) -> &[BlockQ5K] {
        self.blocks
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`:  FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q5_K GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ5K is #[repr(C)] with size BLOCK_Q5K_BYTES (= 176).
            // The compile-time assert above guarantees this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q5K_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q5k(
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
                            "CUDA Q5K GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q5k(
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
// LinearQ6K
// ---------------------------------------------------------------------------

/// A linear layer with Q6_K (6-bit K-quant) weights.
///
/// Computes `output = W × input` using the scalar Q6_K GEMV kernel.
/// `in_features` must be a multiple of 256 (QK_K).
#[derive(Debug)]
pub struct LinearQ6K<'a> {
    /// Q6_K weight blocks in row-major order: `[out_features × (in_features / 256)]` blocks.
    blocks: &'a [BlockQ6K],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns), must be a multiple of 256.
    in_features: usize,
}

impl<'a> LinearQ6K<'a> {
    /// Create a Q6_K linear layer, validating block count at construction.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % 256 != 0`, or
    /// - `blocks.len() != out_features * (in_features / 256)`.
    pub fn new(
        blocks: &'a [BlockQ6K],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        const QK_K: usize = 256;

        if in_features == 0 || in_features % QK_K != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ6K".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let blocks_per_row = in_features / QK_K;
        let expected_blocks = out_features * blocks_per_row;
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ6K".into(),
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

    /// Raw Q6_K block references.
    pub fn blocks(&self) -> &[BlockQ6K] {
        self.blocks
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`:  FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q6_K GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ6K is #[repr(C)] with size BLOCK_Q6K_BYTES (= 210).
            // The compile-time assert above guarantees this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q6K_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q6k(
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
                            "CUDA Q6K GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q6k(
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
