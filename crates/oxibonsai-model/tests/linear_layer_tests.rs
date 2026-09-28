//! The dense `LinearLayer` arm (B2-11-FIX, gatekeeper REQUIRED #5):
//! `LinearDense` holds a row-major `f32` matrix (widened losslessly from an
//! `F32` / `F16` / `BF16` tensor) and projects through the kernel
//! dispatcher's `f32` GEMV, so a dense projection shares the perf path of
//! the dense LM head and a batched call is bit-identical to per-row calls.

use std::borrow::Cow;
use std::sync::Arc;

use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_kernels::KernelDispatcher;
use oxibonsai_model::error::ModelError;
use oxibonsai_model::layers::linear::{LinearDense, LinearLayer};

fn kernel() -> Arc<KernelDispatcher> {
    Arc::new(KernelDispatcher::auto_detect())
}

/// Deterministic weights / inputs with a spread of magnitudes.
fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 97) as f32 - 48.0) * 0.015625)
        .collect()
}

/// `f64` reference GEMV over row-major weights.
fn gemv_f64(weights: &[f32], input: &[f32], out_f: usize, in_f: usize) -> Vec<f64> {
    (0..out_f)
        .map(|r| {
            weights[r * in_f..(r + 1) * in_f]
                .iter()
                .zip(input)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum()
        })
        .collect()
}

#[test]
fn linear_dense_forward_matches_an_f64_gemv() {
    // 300 rows crosses the dispatcher GEMV's parallel-row threshold; 130
    // columns is not a multiple of any SIMD lane count.
    for &(out_f, in_f) in &[(1usize, 1usize), (7, 130), (300, 64), (33, 257)] {
        let weights = ramp(out_f * in_f, out_f);
        let input = ramp(in_f, in_f + 3);
        let layer = LinearDense::new(Cow::Borrowed(&weights), out_f, in_f, kernel())
            .expect("valid dense layer");
        assert_eq!(layer.out_features(), out_f);
        assert_eq!(layer.in_features(), in_f);
        assert_eq!(layer.resident_bytes(), 0, "borrowed weights own nothing");
        let mut out = vec![0.0f32; out_f];
        layer.forward(&input, &mut out).expect("dense GEMV");
        let reference = gemv_f64(&weights, &input, out_f, in_f);
        for (r, (&got, &want)) in out.iter().zip(&reference).enumerate() {
            let tol = 1e-5 * (1.0 + want.abs()) * (in_f as f64).sqrt();
            assert!(
                (f64::from(got) - want).abs() <= tol,
                "{out_f}x{in_f} row {r}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn linear_dense_batch_is_bit_identical_to_per_row_forward() {
    let (out_f, in_f, rows) = (300usize, 96usize, 5usize);
    let weights = ramp(out_f * in_f, 9);
    let layer = LinearDense::new(Cow::Owned(weights), out_f, in_f, kernel()).expect("layer");
    assert_eq!(layer.resident_bytes(), out_f * in_f * 4);
    let input = ramp(rows * in_f, 4);
    let mut batched = vec![0.0f32; rows * out_f];
    layer
        .forward_batch(&input, &mut batched, rows)
        .expect("batched");
    for r in 0..rows {
        let mut single = vec![0.0f32; out_f];
        layer
            .forward(&input[r * in_f..(r + 1) * in_f], &mut single)
            .expect("single row");
        assert_eq!(
            batched[r * out_f..(r + 1) * out_f]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            single.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "row {r}"
        );
    }
    // Short buffers are typed errors, never a panic.
    assert!(matches!(
        layer.forward_batch(&input[..in_f], &mut batched, rows),
        Err(ModelError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        layer.forward_batch(&input, &mut batched[..out_f], rows),
        Err(ModelError::ShapeMismatch { .. })
    ));
}

#[test]
fn linear_dense_from_le_bytes_widens_f16_and_bf16_losslessly() {
    let (out_f, in_f) = (3usize, 4usize);
    // Values exactly representable in both half formats.
    let values: Vec<f32> = vec![
        1.0, -2.0, 0.5, 0.25, -0.125, 3.0, -4.5, 6.0, 0.0, -1.5, 2.25, 8.0,
    ];
    let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f16_bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    let bf16_bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    for (bytes, ty) in [
        (&f32_bytes, GgufTensorType::F32),
        (&f16_bytes, GgufTensorType::F16),
        (&bf16_bytes, GgufTensorType::BF16),
    ] {
        let layer = LinearDense::from_le_bytes(bytes, ty, out_f, in_f, kernel())
            .unwrap_or_else(|e| panic!("{ty}: {e}"));
        assert_eq!(layer.weights(), values.as_slice(), "{ty}: widened exactly");
        assert_eq!(layer.source_type(), ty);
        let wrapped: LinearLayer<'_> = layer.into();
        assert_eq!(
            wrapped.quant_type(),
            GgufTensorType::F32,
            "{ty}: a dense layer computes in F32"
        );
        assert_eq!(wrapped.dense_weights(), Some(values.as_slice()));
    }
}

#[test]
fn linear_dense_rejects_bad_shapes_and_quantized_types() {
    let weights = vec![0.0f32; 12];
    assert!(matches!(
        LinearDense::new(Cow::Borrowed(&weights), 5, 3, kernel()),
        Err(ModelError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        LinearDense::new(Cow::Borrowed(&weights), 0, 12, kernel()),
        Err(ModelError::ShapeInvariant { .. })
    ));
    assert!(matches!(
        LinearDense::new(Cow::Borrowed(&weights), usize::MAX, 2, kernel()),
        Err(ModelError::ShapeInvariant { .. })
    ));
    assert!(matches!(
        LinearDense::from_le_bytes(&[0u8; 48], GgufTensorType::Q8_0, 3, 4, kernel()),
        Err(ModelError::InvalidTensor(_))
    ));
    assert!(matches!(
        LinearDense::from_le_bytes(&[0u8; 47], GgufTensorType::F32, 3, 4, kernel()),
        Err(ModelError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        LinearDense::with_source_type(
            Cow::Borrowed(&weights),
            3,
            4,
            GgufTensorType::PQ2_0,
            kernel()
        ),
        Err(ModelError::InvalidTensor(_))
    ));
}

#[test]
fn linear_layer_dense_arm_dispatches_and_reports_like_every_other_arm() {
    let (out_f, in_f) = (6usize, 8usize);
    let weights = ramp(out_f * in_f, 2);
    let dense = LinearDense::new(Cow::Borrowed(&weights), out_f, in_f, kernel()).expect("layer");
    let mut layer: LinearLayer<'_> = dense.into();
    assert!(matches!(layer, LinearLayer::Dense(_)));
    assert_eq!(layer.out_features(), out_f);
    assert_eq!(layer.in_features(), in_f);
    assert!(layer.gpu_handle().is_none());
    assert!(layer.blocks_1bit().is_none());
    assert!(layer.blocks_ternary().is_none());
    assert!(layer.blocks_pq2_0().is_none());
    // Uploading a dense layer is a no-op, not an error.
    layer.upload_to_gpu();
    assert!(layer.gpu_handle().is_none());

    let input = ramp(in_f, 5);
    let mut via_vec = vec![0.0f32; out_f];
    layer
        .forward_vec(&input, &mut via_vec)
        .expect("forward_vec");
    let mut via_mat = vec![0.0f32; out_f];
    layer
        .forward_mat(&input, &mut via_mat, 1)
        .expect("forward_mat");
    assert_eq!(via_vec, via_mat, "the m = 1 GEMM is the GEMV");
    let reference = gemv_f64(&weights, &input, out_f, in_f);
    for (got, want) in via_vec.iter().zip(&reference) {
        assert!((f64::from(*got) - want).abs() <= 1e-5 * (1.0 + want.abs()));
    }
}

#[test]
fn linear_layer_quant_type_names_the_quantized_formats() {
    use oxibonsai_core::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64};
    use oxibonsai_model::layers::linear::{LinearPQ2_0, LinearPTQ1_0, LinearQ2_0G64};

    let pq2 = [BlockPQ2_0 {
        d: half::f16::ONE,
        qs: [0x55u8; 32],
    }];
    let layer: LinearLayer<'_> = LinearPQ2_0::new(&pq2, 1, 128, kernel())
        .expect("pq2")
        .into();
    assert_eq!(layer.quant_type(), GgufTensorType::PQ2_0);
    assert!(layer.dense_weights().is_none());

    let ptq1 = [BlockPTQ1_0 {
        d: half::f16::ONE,
        qs: [0u8; 24],
        qh: [0u8; 2],
    }];
    let layer: LinearLayer<'_> = LinearPTQ1_0::new(&ptq1, 1, 128, kernel())
        .expect("ptq1")
        .into();
    assert_eq!(layer.quant_type(), GgufTensorType::PTQ1_0);

    let g64 = [BlockQ2_0G64 {
        d: half::f16::ONE,
        qs: [0x55u8; 16],
    }];
    let layer: LinearLayer<'_> = LinearQ2_0G64::new(&g64, 1, 64, kernel())
        .expect("q2_0 g64")
        .into();
    assert_eq!(layer.quant_type(), GgufTensorType::Q2_0G64);
}
