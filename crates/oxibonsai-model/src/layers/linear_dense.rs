//! Dense (unquantized) linear layer: row-major `f32` weights projected
//! through the kernel dispatcher's `f32` GEMV (B2-11-FIX, gatekeeper
//! REQUIRED #5).
//!
//! # Why a dense arm exists at all
//!
//! Every shipped model stores its projections quantized, but a GGUF may
//! carry a matrix as `F32` / `F16` / `BF16` — the synthetic `qwen35`
//! fixture's reference variants do, and so does any dequantised or
//! converted checkpoint. Before this type, [`super::linear::LinearLayer`]
//! had no variant that could hold such a matrix, so the hybrid binder had
//! to refuse the whole file. It is also the cleanest parity oracle there
//! is: an `F32` checkpoint has no quantisation noise at all.
//!
//! # One compute path
//!
//! Both `forward` (GEMV) and `forward_batch` (one GEMV per row) go through
//! [`oxibonsai_kernels::KernelDispatcher::gemv_f32`] — the same dispatcher
//! seam the dense LM head of [`crate::model::BonsaiModel`] uses — so a
//! future GPU `f32` GEMV lands behind every dense projection at once, and a
//! batched call is bit-identical to the same rows run one at a time.
//!
//! `F16` / `BF16` sources are widened to `f32` once, at bind time (every
//! half-precision value is exactly representable in `f32`, so the widening
//! is lossless); an `F32` source is either borrowed or copied, depending on
//! how the caller holds it ([`std::borrow::Cow`]).

use std::borrow::Cow;
use std::sync::Arc;

use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_kernels::KernelDispatcher;

use crate::error::{ModelError, ModelResult};

/// A linear layer with dense `f32` weights, row-major
/// `[out_features × in_features]`.
///
/// Computes `output = weights @ input` (no bias).
#[derive(Debug)]
pub struct LinearDense<'a> {
    /// Row-major `[out_features × in_features]` weights.
    weights: Cow<'a, [f32]>,
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns).
    in_features: usize,
    /// On-disk element type the weights were widened from (`F32`, `F16` or
    /// `BF16`); diagnostics only — the arithmetic is always `f32`.
    source_type: GgufTensorType,
    /// Kernel dispatcher the projections run through.
    kernel: Arc<KernelDispatcher>,
}

impl<'a> LinearDense<'a> {
    /// Create a dense layer over row-major `weights`, validating the shape.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for a zero dimension or an overflowing
    /// `out_features * in_features`; [`ModelError::ShapeMismatch`] when
    /// `weights` does not hold exactly `out_features * in_features` values.
    pub fn new(
        weights: Cow<'a, [f32]>,
        out_features: usize,
        in_features: usize,
        kernel: Arc<KernelDispatcher>,
    ) -> ModelResult<Self> {
        Self::with_source_type(
            weights,
            out_features,
            in_features,
            GgufTensorType::F32,
            kernel,
        )
    }

    /// [`LinearDense::new`] recording the on-disk element type the weights
    /// were widened from.
    ///
    /// # Errors
    ///
    /// As [`LinearDense::new`], plus [`ModelError::InvalidTensor`] when
    /// `source_type` is not one of `F32` / `F16` / `BF16`.
    pub fn with_source_type(
        weights: Cow<'a, [f32]>,
        out_features: usize,
        in_features: usize,
        source_type: GgufTensorType,
        kernel: Arc<KernelDispatcher>,
    ) -> ModelResult<Self> {
        if !matches!(
            source_type,
            GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::BF16
        ) {
            return Err(ModelError::InvalidTensor(format!(
                "LinearDense: {source_type} is not an unquantized element type"
            )));
        }
        if out_features == 0 || in_features == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: "LinearDense".to_string(),
                expected: "out_features > 0 and in_features > 0".to_string(),
                actual: format!("{out_features} x {in_features}"),
            });
        }
        let expected =
            out_features
                .checked_mul(in_features)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: "LinearDense".to_string(),
                    expected: "out_features * in_features representable as usize".to_string(),
                    actual: format!("{out_features} x {in_features}"),
                })?;
        if weights.len() != expected {
            return Err(ModelError::ShapeMismatch {
                name: "LinearDense weights".to_string(),
                expected: vec![out_features, in_features],
                actual: vec![weights.len()],
            });
        }
        Ok(Self {
            weights,
            out_features,
            in_features,
            source_type,
            kernel,
        })
    }

    /// Widen raw little-endian `F32` / `F16` / `BF16` tensor bytes into an
    /// owned dense layer.
    ///
    /// # Errors
    ///
    /// [`ModelError::InvalidTensor`] for any other `tensor_type`,
    /// [`ModelError::ShapeMismatch`] when `data` is not exactly
    /// `out_features * in_features` elements, and anything
    /// [`LinearDense::with_source_type`] returns.
    pub fn from_le_bytes(
        data: &[u8],
        tensor_type: GgufTensorType,
        out_features: usize,
        in_features: usize,
        kernel: Arc<KernelDispatcher>,
    ) -> ModelResult<LinearDense<'static>> {
        let width = match tensor_type {
            GgufTensorType::F32 => 4usize,
            GgufTensorType::F16 | GgufTensorType::BF16 => 2,
            other => {
                return Err(ModelError::InvalidTensor(format!(
                    "LinearDense: {other} is not an unquantized element type"
                )))
            }
        };
        let n =
            out_features
                .checked_mul(in_features)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: "LinearDense".to_string(),
                    expected: "out_features * in_features representable as usize".to_string(),
                    actual: format!("{out_features} x {in_features}"),
                })?;
        let needed = n
            .checked_mul(width)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "LinearDense".to_string(),
                expected: "byte length representable as usize".to_string(),
                actual: format!("{n} x {width}"),
            })?;
        if data.len() != needed {
            return Err(ModelError::ShapeMismatch {
                name: "LinearDense bytes".to_string(),
                expected: vec![needed],
                actual: vec![data.len()],
            });
        }
        let mut weights = Vec::with_capacity(n);
        match tensor_type {
            GgufTensorType::F32 => weights.extend(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            GgufTensorType::F16 => weights.extend(
                data.chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()),
            ),
            _ => weights.extend(
                data.chunks_exact(2)
                    .map(|c| oxibonsai_core::bf16::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            ),
        }
        LinearDense::with_source_type(
            Cow::Owned(weights),
            out_features,
            in_features,
            tensor_type,
            kernel,
        )
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features (columns).
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// The row-major `[out_features × in_features]` weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// The on-disk element type the weights were widened from.
    pub fn source_type(&self) -> GgufTensorType {
        self.source_type
    }

    /// Resident bytes owned by this layer (`0` when the weights are
    /// borrowed).
    pub fn resident_bytes(&self) -> usize {
        match &self.weights {
            Cow::Borrowed(_) => 0,
            Cow::Owned(v) => v.len() * std::mem::size_of::<f32>(),
        }
    }

    /// Forward pass (GEMV): `output[..out_features] = W · input[..in_features]`.
    ///
    /// # Errors
    ///
    /// [`ModelError::Kernel`] when `input` / `output` are too short for the
    /// layer's shape.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> ModelResult<()> {
        self.kernel
            .gemv_f32(
                &self.weights,
                input,
                output,
                self.out_features,
                self.in_features,
            )
            .map_err(ModelError::Kernel)
    }

    /// Forward pass (GEMM) over `batch` row-major input rows: one dispatched
    /// GEMV per row, so every row is bit-identical to [`Self::forward`] on
    /// that row alone.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `input` holds fewer than
    /// `batch * in_features` or `output` fewer than `batch * out_features`
    /// values, and anything [`Self::forward`] returns.
    pub fn forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
    ) -> ModelResult<()> {
        let in_len = batch.saturating_mul(self.in_features);
        let out_len = batch.saturating_mul(self.out_features);
        if input.len() < in_len {
            return Err(ModelError::ShapeMismatch {
                name: "LinearDense batch input".to_string(),
                expected: vec![batch, self.in_features],
                actual: vec![input.len()],
            });
        }
        if output.len() < out_len {
            return Err(ModelError::ShapeMismatch {
                name: "LinearDense batch output".to_string(),
                expected: vec![batch, self.out_features],
                actual: vec![output.len()],
            });
        }
        for (row_in, row_out) in input[..in_len]
            .chunks_exact(self.in_features)
            .zip(output[..out_len].chunks_exact_mut(self.out_features))
        {
            self.forward(row_in, row_out)?;
        }
        Ok(())
    }
}
