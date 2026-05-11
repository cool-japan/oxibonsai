//! Standard GGUF linear layers for Q4_0 and Q8_0 quantization formats.
//!
//! These layers implement the full `forward` / `forward_batch` interface
//! using the scalar GEMV kernels from `oxibonsai-kernels`.  When the
//! `native-cuda` feature is enabled and a CUDA device is available, `forward`
//! dispatches to the NVRTC GEMV kernels; on failure it falls back to the
//! scalar CPU path.
//!
//! # Layout
//!
//! Both types follow the same convention as `LinearFP8E4M3` / `LinearFP8E5M2`:
//! - Weights are borrowed zero-copy from the GGUF memory map (`'a` lifetime).
//! - Shape validation at construction time returns `ModelError::ShapeMismatch`.
//! - `forward` dispatches to the scalar GEMV kernel.
//! - `forward_batch` loops over tokens and calls `forward` once per token.

use oxibonsai_core::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};
use oxibonsai_kernels::{gemv_q4_0, gemv_q8_0};

use crate::error::{ModelError, ModelResult};

// ---------------------------------------------------------------------------
// Compile-time size assertions (documenting the SAFETY invariants used in the
// unsafe `from_raw_parts` casts inside the CUDA dispatch paths below).
// ---------------------------------------------------------------------------

#[cfg(all(feature = "native-cuda", any(target_os = "linux", target_os = "windows")))]
const _: () = assert!(
    std::mem::size_of::<oxibonsai_core::BlockQ4_0>() == oxibonsai_core::BLOCK_Q4_0_BYTES,
);
#[cfg(all(feature = "native-cuda", any(target_os = "linux", target_os = "windows")))]
const _: () = assert!(
    std::mem::size_of::<oxibonsai_core::BlockQ8_0>() == oxibonsai_core::BLOCK_Q8_0_BYTES,
);

// ---------------------------------------------------------------------------
// LinearQ4_0
// ---------------------------------------------------------------------------

/// Linear layer with Q4_0 (4-bit symmetric, 32 weights per block) weights.
///
/// The weight matrix `W` is stored row-major as a slice of `BlockQ4_0`.
/// Row `r` occupies blocks `[r * (in_features / QK_Q4_0) .. (r+1) * …]`.
///
/// Computing: `output = W @ input` (no bias — Qwen3/Bonsai has no bias).
#[derive(Debug)]
pub struct LinearQ4_0<'a> {
    blocks: &'a [BlockQ4_0],
    out_features: usize,
    in_features: usize,
}

impl<'a> LinearQ4_0<'a> {
    /// Construct a Q4_0 linear layer with shape validation.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % QK_Q4_0 != 0`
    /// - `blocks.len() != out_features * (in_features / QK_Q4_0)`
    pub fn new(
        blocks: &'a [BlockQ4_0],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        if in_features == 0 || in_features % QK_Q4_0 != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ4_0".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let expected_blocks = out_features * (in_features / QK_Q4_0);
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ4_0".into(),
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

    /// Raw Q4_0 block references.
    pub fn blocks(&self) -> &[BlockQ4_0] {
        self.blocks
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q4_0 GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(feature = "native-cuda", any(target_os = "linux", target_os = "windows")))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ4_0 is #[repr(C)] with size BLOCK_Q4_0_BYTES (= 18).
            // The compile-time assert above and the one in oxibonsai_core::quant_std
            // both guarantee this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q4_0_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q4_0(
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
                            "CUDA Q4_0 GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q4_0(self.blocks, input, output, self.out_features, self.in_features)
            .map_err(ModelError::Kernel)
    }

    /// Forward pass: batched input (loop-over-tokens GEMM).
    ///
    /// - `input`: Row-major FP32 matrix `[m × in_features]`.
    /// - `output`: Row-major FP32 matrix `[m × out_features]`.
    /// - `m`: Number of tokens (batch size).
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        for t in 0..m {
            let inp = &input[t * self.in_features..(t + 1) * self.in_features];
            let out = &mut output[t * self.out_features..(t + 1) * self.out_features];
            self.forward(inp, out)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LinearQ8_0
// ---------------------------------------------------------------------------

/// Linear layer with Q8_0 (8-bit symmetric, 32 weights per block) weights.
///
/// The weight matrix `W` is stored row-major as a slice of `BlockQ8_0`.
/// Row `r` occupies blocks `[r * (in_features / QK_Q8_0) .. (r+1) * …]`.
///
/// Computing: `output = W @ input` (no bias).
#[derive(Debug)]
pub struct LinearQ8_0<'a> {
    blocks: &'a [BlockQ8_0],
    out_features: usize,
    in_features: usize,
}

impl<'a> LinearQ8_0<'a> {
    /// Construct a Q8_0 linear layer with shape validation.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::ShapeMismatch`] if:
    /// - `in_features == 0` or `in_features % QK_Q8_0 != 0`
    /// - `blocks.len() != out_features * (in_features / QK_Q8_0)`
    pub fn new(
        blocks: &'a [BlockQ8_0],
        out_features: usize,
        in_features: usize,
    ) -> ModelResult<Self> {
        if in_features == 0 || in_features % QK_Q8_0 != 0 {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ8_0".into(),
                expected: vec![out_features, in_features],
                actual: vec![out_features, in_features],
            });
        }
        let expected_blocks = out_features * (in_features / QK_Q8_0);
        if blocks.len() != expected_blocks {
            return Err(ModelError::ShapeMismatch {
                name: "LinearQ8_0".into(),
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

    /// Raw Q8_0 block references.
    pub fn blocks(&self) -> &[BlockQ8_0] {
        self.blocks
    }

    /// Forward pass: single input vector (GEMV).
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    ///
    /// When the `native-cuda` feature is enabled and a CUDA device is present
    /// the NVRTC Q8_0 GEMV kernel is tried first; any failure other than
    /// "no CUDA device" is logged as a warning and the CPU scalar path runs
    /// instead.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        #[cfg(all(feature = "native-cuda", any(target_os = "linux", target_os = "windows")))]
        if oxibonsai_kernels::CudaGraph::global().is_ok() {
            // SAFETY: BlockQ8_0 is #[repr(C)] with size BLOCK_Q8_0_BYTES (= 34).
            // The compile-time assert above and the one in oxibonsai_core::quant_std
            // both guarantee this layout.
            let raw = unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>(),
                    self.blocks.len() * oxibonsai_core::BLOCK_Q8_0_BYTES,
                )
            };
            match oxibonsai_kernels::cuda_gemv_q8_0(
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
                            "CUDA Q8_0 GEMV failed, falling back to CPU scalar"
                        );
                    }
                }
            }
        }
        gemv_q8_0(self.blocks, input, output, self.out_features, self.in_features)
            .map_err(ModelError::Kernel)
    }

    /// Forward pass: batched input (loop-over-tokens GEMM).
    ///
    /// - `input`: Row-major FP32 matrix `[m × in_features]`.
    /// - `output`: Row-major FP32 matrix `[m × out_features]`.
    /// - `m`: Number of tokens (batch size).
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], m: usize) -> ModelResult<()> {
        for t in 0..m {
            let inp = &input[t * self.in_features..(t + 1) * self.in_features];
            let out = &mut output[t * self.out_features..(t + 1) * self.out_features];
            self.forward(inp, out)?;
        }
        Ok(())
    }
}
