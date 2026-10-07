//! Dense (unquantized) linear layer: row-major `f32` weights projected
//! through the kernel dispatcher's `f32` GEMV and GEMM.
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
//! `forward` (one row) runs through
//! [`oxibonsai_kernels::KernelDispatcher::gemv_f32`] — the same dispatcher
//! seam the dense LM head of [`crate::model::BonsaiModel`] uses — and
//! `forward_batch` (many rows) through
//! [`oxibonsai_kernels::KernelDispatcher::gemm_f32`], its blocked, Rayon
//! parallel batched form. Both are the same arithmetic: every element of a
//! batched call is bit-identical to `forward` on that row alone, so batching
//! changes how fast a projection runs, never what it returns. A GPU `f32`
//! route, when the dispatcher gains one, lands behind every dense projection
//! at once; today every tier runs the CPU kernels.
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
                data.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c)),
            ),
            GgufTensorType::F16 => weights.extend(
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32()),
            ),
            _ => weights.extend(
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| oxibonsai_core::bf16::bf16_to_f32(u16::from_le_bytes(*c))),
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

    /// Forward pass (GEMM) over `batch` row-major input rows: row `r` of
    /// `output` is `W · input[r]`, all rows in one
    /// [`KernelDispatcher::gemm_f32`] call.
    ///
    /// Every row is bit-identical to [`Self::forward`] on that row alone.
    /// Buffers longer than `batch` rows are accepted: only the leading
    /// `batch * in_features` inputs are read and `batch * out_features`
    /// outputs written.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `input` holds fewer than
    /// `batch * in_features` or `output` fewer than `batch * out_features`
    /// values; [`ModelError::Kernel`] if the kernel refuses the call (it does
    /// not once those checks pass, since the constructors guarantee the
    /// weights are exactly `out_features * in_features` long).
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
        self.kernel
            .gemm_f32(
                &input[..in_len],
                &self.weights,
                None,
                batch,
                self.in_features,
                self.out_features,
                &mut output[..out_len],
            )
            .map_err(ModelError::Kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_kernels::{dot_f32, KernelTier};

    /// A bit pattern no computation here produces (a quiet NaN with a
    /// payload), so an output element the layer forgot to write is a
    /// mismatch rather than a plausible zero.
    const SENTINEL_BITS: u32 = 0x7FC0_DEAD;

    /// Deterministic pseudo-random values in `[-0.5, 0.5)`.
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 40) as u32 as f32) / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn dense_layer(
        out_features: usize,
        in_features: usize,
        tier: KernelTier,
    ) -> LinearDense<'static> {
        LinearDense::new(
            Cow::Owned(values(out_features * in_features, 0x5EED)),
            out_features,
            in_features,
            Arc::new(KernelDispatcher::with_tier(tier)),
        )
        .expect("well-formed dense layer")
    }

    /// The body `forward_batch` had before it moved onto the GEMM: one
    /// dispatched GEMV per input row. Kept here as the oracle.
    fn per_row_reference(layer: &LinearDense<'_>, input: &[f32], batch: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * layer.out_features()];
        for (row_in, row_out) in input[..batch * layer.in_features()]
            .chunks_exact(layer.in_features())
            .zip(out.chunks_exact_mut(layer.out_features()))
        {
            layer.forward(row_in, row_out).expect("per-row forward");
        }
        out
    }

    fn assert_bit_identical(context: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{context}: length");
        for (idx, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{context}: element {idx}: got {g}, want {w}"
            );
        }
    }

    /// The batched call is the per-row loop it replaced, bit for bit, for the
    /// batch sizes a caller uses — a single row, a pair, an odd size, a full
    /// chunk — on a shape that crosses the 64-row weight tile and leaves a
    /// lane remainder (`in_features` not a multiple of 8), and on one big
    /// enough that a 64-row batch takes the Rayon path.
    #[test]
    fn forward_batch_is_bit_identical_to_the_per_row_loop() {
        for tier in [KernelTier::Reference, oxibonsai_kernels::cpu_kernel_tier()] {
            for (out_features, in_features) in [(70usize, 21usize), (300, 129)] {
                let layer = dense_layer(out_features, in_features, tier);
                for batch in [1usize, 2, 7, 64] {
                    let input = values(batch * in_features, 0xA11CE + batch as u64);
                    let mut got = vec![f32::from_bits(SENTINEL_BITS); batch * out_features];
                    layer
                        .forward_batch(&input, &mut got, batch)
                        .expect("forward_batch");
                    assert_bit_identical(
                        &format!("{tier:?} {out_features}x{in_features} batch {batch}"),
                        &got,
                        &per_row_reference(&layer, &input, batch),
                    );
                }
            }
        }
    }

    /// `forward` is the dispatcher's GEMV, untouched: it equals a direct
    /// `gemv_f32` call and the plain per-row dot products.
    #[test]
    fn forward_is_the_dispatcher_gemv() {
        let (out_features, in_features) = (70usize, 21usize);
        let layer = dense_layer(out_features, in_features, KernelTier::Reference);
        let input = values(in_features, 7);
        let mut got = vec![0.0f32; out_features];
        layer.forward(&input, &mut got).expect("forward");

        let mut direct = vec![0.0f32; out_features];
        oxibonsai_kernels::gemv_f32(
            layer.weights(),
            &input,
            &mut direct,
            out_features,
            in_features,
        )
        .expect("direct gemv_f32");
        assert_bit_identical("direct gemv_f32", &got, &direct);

        let dots: Vec<f32> = layer
            .weights()
            .chunks_exact(in_features)
            .map(|row| dot_f32(row, &input))
            .collect();
        assert_bit_identical("dot_f32 per row", &got, &dots);
    }

    #[test]
    fn forward_batch_of_no_rows_writes_nothing() {
        let layer = dense_layer(5, 3, KernelTier::Reference);
        let mut out = [f32::from_bits(SENTINEL_BITS); 5];
        layer
            .forward_batch(&[], &mut out, 0)
            .expect("an empty batch is a no-op");
        assert!(out.iter().all(|v| v.to_bits() == SENTINEL_BITS));
    }

    /// Buffers longer than the batch are accepted; the extra input is not
    /// read and the extra output is not written.
    #[test]
    fn forward_batch_reads_and_writes_only_the_leading_rows() {
        let (out_features, in_features, batch) = (70usize, 21usize, 3usize);
        let layer = dense_layer(out_features, in_features, KernelTier::Reference);
        let input = values(batch * in_features, 3);
        let want = per_row_reference(&layer, &input, batch);

        let mut long_input = input.clone();
        long_input.extend([f32::NAN; 40]);
        let mut out = vec![f32::from_bits(SENTINEL_BITS); batch * out_features + 11];
        layer
            .forward_batch(&long_input, &mut out, batch)
            .expect("longer buffers");
        assert_bit_identical("prefix", &out[..batch * out_features], &want);
        assert!(
            out[batch * out_features..]
                .iter()
                .all(|v| v.to_bits() == SENTINEL_BITS),
            "the tail of `output` must not be written"
        );
    }

    #[test]
    fn forward_batch_names_the_short_buffer() {
        let layer = dense_layer(4, 3, KernelTier::Reference);
        let mut out = vec![0.0f32; 4 * 2];
        assert!(matches!(
            layer.forward_batch(&[0.0; 5], &mut out, 2),
            Err(ModelError::ShapeMismatch { ref name, .. }) if name == "LinearDense batch input"
        ));
        let mut short = vec![0.0f32; 4 * 2 - 1];
        assert!(matches!(
            layer.forward_batch(&[0.0; 6], &mut short, 2),
            Err(ModelError::ShapeMismatch { ref name, .. }) if name == "LinearDense batch output"
        ));
        // Nothing overflows: a batch too large to size is a short buffer, not a wrap.
        assert!(matches!(
            layer.forward_batch(&[0.0; 6], &mut out, usize::MAX),
            Err(ModelError::ShapeMismatch { .. })
        ));
    }
}
