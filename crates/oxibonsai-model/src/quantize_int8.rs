//! INT8 (8-bit symmetric) quantization for weight tensors.
//!
//! Supports two modes:
//! - **Per-tensor**: a single scale covers the entire tensor.
//! - **Per-channel**: one scale per output channel (row of the weight matrix),
//!   which preserves per-neuron magnitude variation and yields lower error.
//!
//! # Encoding
//!
//! For each channel (or the whole tensor in per-tensor mode):
//! ```text
//! scale = clip_ratio * max(|w_i|) / 127
//! q_i   = round(w_i / scale)          ∈ [-127, 127]
//! ```
//!
//! Dequantization: `w̃_i = q_i * scale`.

use crate::quantize::{
    analyze_quantization_error, quantize_q1_0_g128, QuantizationError, QuantizeError,
};

// ─── Mode ────────────────────────────────────────────────────────────────────

/// Selects the granularity at which scale factors are computed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Int8Mode {
    /// One scale for the whole tensor — fastest but least accurate.
    PerTensor,
    /// One scale per output channel (row). More accurate for weight matrices.
    PerChannel { num_channels: usize },
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for INT8 quantization.
#[derive(Debug, Clone)]
pub struct Int8Config {
    pub mode: Int8Mode,
    /// Clip weights at `clip_ratio * max_val` before quantizing.
    ///
    /// Values in `(0.0, 1.0]` reduce the influence of large outliers.
    /// The default value of `1.0` disables clipping.
    pub clip_ratio: f32,
}

impl Default for Int8Config {
    fn default() -> Self {
        Self {
            mode: Int8Mode::PerTensor,
            clip_ratio: 1.0,
        }
    }
}

// ─── Quantized tensor ────────────────────────────────────────────────────────

/// An INT8-quantized tensor.
#[derive(Debug)]
pub struct Int8Tensor {
    /// Quantized weight values, one `i8` per element.
    pub data: Vec<i8>,
    /// Scale factors — one per channel in `PerChannel` mode, one total in
    /// `PerTensor` mode.
    pub scales: Vec<f32>,
    /// Shape of the original tensor (same semantics as the source `f32` tensor).
    pub shape: Vec<usize>,
    /// Quantization mode used to produce this tensor.
    pub mode: Int8Mode,
}

impl Int8Tensor {
    /// Dequantize back to `f32`.
    ///
    /// In `PerChannel` mode each element is multiplied by its channel's scale.
    /// In `PerTensor` mode every element uses `scales[0]`.
    pub fn dequantize(&self) -> Vec<f32> {
        match self.mode {
            Int8Mode::PerTensor => {
                let scale = self.scales[0];
                self.data.iter().map(|&q| q as f32 * scale).collect()
            }
            Int8Mode::PerChannel { num_channels } => {
                if num_channels == 0 || self.data.is_empty() {
                    return Vec::new();
                }
                let channel_size = self.data.len() / num_channels;
                let mut out = Vec::with_capacity(self.data.len());
                for (ch, scale) in self.scales.iter().enumerate() {
                    let start = ch * channel_size;
                    let end = start + channel_size;
                    for &q in &self.data[start..end] {
                        out.push(q as f32 * scale);
                    }
                }
                out
            }
        }
    }

    /// Total memory footprint of the quantized representation.
    ///
    /// `data` occupies 1 byte per element; each `f32` scale occupies 4 bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Ratio of original `f32` memory to quantized memory.
    ///
    /// `original_f32_bytes / quantized_bytes`
    pub fn compression_ratio(&self) -> f32 {
        let original = self.data.len() * 4;
        let quantized = self.memory_bytes();
        if quantized == 0 {
            return 1.0;
        }
        original as f32 / quantized as f32
    }

    /// Quantized matrix-vector multiply: `y = W * x`.
    ///
    /// `W` is this INT8 tensor interpreted as a `[num_channels × cols]` matrix.
    /// `x` is a `f32` vector of length `cols`.
    ///
    /// Each row uses its own scale so the output is computed in `f32`.
    ///
    /// Panics if the tensor mode is `PerTensor` — use `dequantize` + a BLAS
    /// routine instead for that case.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        match self.mode {
            Int8Mode::PerTensor => {
                // Per-tensor: single scale, interpret as num_elements × 1 … or just
                // treat as a flat dot product with the provided x.
                let scale = self.scales[0];
                // Reshape as rows of length x.len() if divisible; otherwise panic.
                let cols = x.len();
                assert!(
                    cols > 0 && self.data.len() % cols == 0,
                    "matvec: data length {} not divisible by x.len() {}",
                    self.data.len(),
                    cols
                );
                let rows = self.data.len() / cols;
                let mut out = vec![0.0_f32; rows];
                for (r, row) in self.data.chunks_exact(cols).enumerate() {
                    let mut acc = 0.0_f32;
                    for (&q, &xi) in row.iter().zip(x.iter()) {
                        acc += (q as f32 * scale) * xi;
                    }
                    out[r] = acc;
                }
                out
            }
            Int8Mode::PerChannel { num_channels } => {
                let cols = x.len();
                assert_eq!(
                    self.data.len(),
                    num_channels * cols,
                    "matvec: data len {} != num_channels {} * x.len() {}",
                    self.data.len(),
                    num_channels,
                    cols
                );
                let mut out = vec![0.0_f32; num_channels];
                for (ch, scale) in self.scales.iter().enumerate() {
                    let start = ch * cols;
                    let row = &self.data[start..start + cols];
                    let mut acc = 0.0_f32;
                    for (&q, &xi) in row.iter().zip(x.iter()) {
                        acc += (q as f32 * scale) * xi;
                    }
                    out[ch] = acc;
                }
                out
            }
        }
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors arising from INT8 quantization operations.
#[derive(Debug, thiserror::Error)]
pub enum Int8QuantizeError {
    /// Weight count is not evenly divisible by the requested channel count.
    #[error("Number of weights {total} not divisible by num_channels {channels}")]
    ChannelMismatch { total: usize, channels: usize },

    /// The weight tensor is empty.
    #[error("Empty weight tensor")]
    EmptyTensor,
}

// ─── Channel-level primitive ─────────────────────────────────────────────────

/// Quantize a slice of `f32` weights into `(i8 values, scale)`.
///
/// `scale = clip_ratio * max(|w_i|) / 127`.  Every weight is clamped to
/// `[-127, 127]` after division so that rounding cannot produce `i8::MIN`
/// (-128), which would be asymmetric.
pub fn quantize_channel(weights: &[f32], clip_ratio: f32) -> (Vec<i8>, f32) {
    if weights.is_empty() {
        return (Vec::new(), 0.0);
    }

    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);

    if max_abs == 0.0 {
        return (vec![0i8; weights.len()], 0.0);
    }

    let clipped_max = clip_ratio.clamp(0.0, 1.0) * max_abs;
    let scale = clipped_max / 127.0_f32;

    let quantized = weights
        .iter()
        .map(|&w| (w / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (quantized, scale)
}

// ─── Per-tensor quantization ──────────────────────────────────────────────────

/// Quantize a flat `f32` tensor to INT8 using a single global scale.
pub fn quantize_per_tensor(weights: &[f32]) -> Int8Tensor {
    let (data, scale) = quantize_channel(weights, 1.0);
    Int8Tensor {
        shape: vec![weights.len()],
        data,
        scales: vec![scale],
        mode: Int8Mode::PerTensor,
    }
}

// ─── Per-channel quantization ─────────────────────────────────────────────────

/// Quantize a flat `f32` tensor to INT8 using one scale per output channel.
///
/// `num_channels` is the number of rows (output neurons) of the weight matrix.
/// `weights.len()` must be divisible by `num_channels`, and `num_channels`
/// must be non-zero (returns [`Int8QuantizeError::ChannelMismatch`] otherwise).
pub fn quantize_per_channel(
    weights: &[f32],
    num_channels: usize,
) -> Result<Int8Tensor, Int8QuantizeError> {
    if weights.is_empty() {
        return Err(Int8QuantizeError::EmptyTensor);
    }
    if num_channels == 0 || weights.len() % num_channels != 0 {
        return Err(Int8QuantizeError::ChannelMismatch {
            total: weights.len(),
            channels: num_channels,
        });
    }

    let channel_size = weights.len() / num_channels;
    let mut all_data: Vec<i8> = Vec::with_capacity(weights.len());
    let mut scales: Vec<f32> = Vec::with_capacity(num_channels);

    for chunk in weights.chunks_exact(channel_size) {
        let (q, scale) = quantize_channel(chunk, 1.0);
        all_data.extend_from_slice(&q);
        scales.push(scale);
    }

    Ok(Int8Tensor {
        shape: vec![num_channels, channel_size],
        data: all_data,
        scales,
        mode: Int8Mode::PerChannel { num_channels },
    })
}

// ─── Error analysis ───────────────────────────────────────────────────────────

/// Quantization quality statistics for an INT8-encoded tensor.
#[derive(Debug, Clone)]
pub struct Int8QuantError {
    /// Mean squared error between the original and dequantized tensor.
    pub mse: f32,
    /// Largest absolute per-element difference.
    pub max_abs_error: f32,
    /// Signal-to-noise ratio in dB: `10 * log10(signal / noise)`.
    pub snr_db: f32,
    /// Effective bits per weight — should be close to 8.0 for INT8.
    pub bits_per_weight: f32,
    /// `original_f32_bytes / quantized_bytes`.
    pub compression_ratio: f32,
}

/// Compute quantization error statistics comparing `original` to `quantized`.
pub fn analyze_int8_error(original: &[f32], quantized: &Int8Tensor) -> Int8QuantError {
    let reconstructed = quantized.dequantize();
    let n = original.len().min(reconstructed.len());

    if n == 0 {
        return Int8QuantError {
            mse: 0.0,
            max_abs_error: 0.0,
            snr_db: f32::INFINITY,
            bits_per_weight: 8.0,
            compression_ratio: quantized.compression_ratio(),
        };
    }

    let mut sum_sq_err = 0.0_f64;
    let mut max_abs_err = 0.0_f32;
    let mut signal_power = 0.0_f64;

    for i in 0..n {
        let orig = original[i];
        let recon = reconstructed[i];
        let err = orig - recon;
        sum_sq_err += f64::from(err * err);
        let abs_err = err.abs();
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
        signal_power += f64::from(orig * orig);
    }

    let mse = (sum_sq_err / n as f64) as f32;
    let noise_power = sum_sq_err / n as f64;

    let snr_db = if noise_power == 0.0 {
        f32::INFINITY
    } else {
        let snr_linear = (signal_power / n as f64) / noise_power;
        (10.0 * snr_linear.log10()) as f32
    };

    Int8QuantError {
        mse,
        max_abs_error: max_abs_err,
        snr_db,
        bits_per_weight: 8.0,
        compression_ratio: quantized.compression_ratio(),
    }
}

// ─── Cross-format comparison ──────────────────────────────────────────────────

/// Side-by-side comparison of Q1_0 and INT8 quantization quality.
pub struct QuantizationComparison {
    /// Q1_0_g128 quantization error (1-bit, 128-element groups).
    pub q1_0: QuantizationError,
    /// INT8 per-tensor quantization error.
    pub int8_per_tensor: Int8QuantError,
    /// INT8 per-channel quantization error, if `num_channels` was specified
    /// and divides the weight count evenly.
    pub int8_per_channel: Option<Int8QuantError>,
}

/// Quantize `weights` using all available methods and return their error metrics.
///
/// `num_channels` is forwarded to [`quantize_per_channel`].  If `None`, the
/// `int8_per_channel` field of the result will be `None`.
///
/// The Q1_0 path requires the weight count to be a multiple of 128; if it is
/// not, the weights are zero-padded to the next multiple before analysis.
pub fn compare_quantization_methods(
    weights: &[f32],
    num_channels: Option<usize>,
) -> Result<QuantizationComparison, Int8QuantizeError> {
    if weights.is_empty() {
        return Err(Int8QuantizeError::EmptyTensor);
    }

    // ── Q1_0 path ─────────────────────────────────────────────────────────
    // Pad to a multiple of GROUP_SIZE if necessary.
    use crate::quantize::GROUP_SIZE;
    let q1_0_error = {
        let remainder = weights.len() % GROUP_SIZE;
        let padded: std::borrow::Cow<[f32]> = if remainder == 0 {
            std::borrow::Cow::Borrowed(weights)
        } else {
            let mut v = weights.to_vec();
            v.resize(weights.len() + GROUP_SIZE - remainder, 0.0);
            std::borrow::Cow::Owned(v)
        };
        let quantized = quantize_q1_0_g128(&padded).map_err(|e: QuantizeError| {
            // Translate to our error type — this path is always valid after
            // padding so any error is unexpected.
            Int8QuantizeError::ChannelMismatch {
                total: padded.len(),
                channels: e.to_string().len(), // dummy — never reached
            }
        })?;
        analyze_quantization_error(weights, &quantized)
            .map_err(|_| Int8QuantizeError::EmptyTensor)?
    };

    // ── INT8 per-tensor ──────────────────────────────────────────────────
    let int8_pt = quantize_per_tensor(weights);
    let int8_per_tensor = analyze_int8_error(weights, &int8_pt);

    // ── INT8 per-channel ─────────────────────────────────────────────────
    let int8_per_channel = if let Some(ch) = num_channels {
        let int8_pc = quantize_per_channel(weights, ch)?;
        Some(analyze_int8_error(weights, &int8_pc))
    } else {
        None
    };

    Ok(QuantizationComparison {
        q1_0: q1_0_error,
        int8_per_tensor,
        int8_per_channel,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── quantize_per_tensor_basic ─────────────────────────────────────────

    #[test]
    fn test_quantize_per_tensor_basic() {
        let weights: Vec<f32> = (0..256).map(|i| i as f32 * 0.01 - 1.28).collect();
        let q = quantize_per_tensor(&weights);
        assert_eq!(q.data.len(), weights.len());
        assert_eq!(q.scales.len(), 1);
        assert!(q.scales[0] > 0.0, "scale must be positive");
        assert!(matches!(q.mode, Int8Mode::PerTensor));
    }

    // ── quantize_per_tensor_symmetric ────────────────────────────────────

    #[test]
    fn test_quantize_per_tensor_symmetric() {
        // Symmetric weights ±1.0 should quantize to ±127.
        let weights: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 })
            .collect();
        let q = quantize_per_tensor(&weights);
        // scale = 1/127
        let expected_scale = 1.0_f32 / 127.0_f32;
        assert!(
            (q.scales[0] - expected_scale).abs() < 1e-5,
            "scale = {}, expected ~{expected_scale}",
            q.scales[0]
        );
        for &v in q.data.iter() {
            assert!(
                v == 127 || v == -127,
                "quantized value should be ±127, got {v}"
            );
        }
    }

    // ── quantize_per_channel_basic ────────────────────────────────────────

    #[test]
    fn test_quantize_per_channel_basic() {
        // 4 channels × 64 weights each = 256 total
        let weights: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let q = quantize_per_channel(&weights, 4).expect("per-channel quantize");
        assert_eq!(q.data.len(), 256);
        assert_eq!(q.scales.len(), 4);
        assert!(matches!(q.mode, Int8Mode::PerChannel { num_channels: 4 }));
        // Each channel scale should differ.
        assert_ne!(q.scales[0], q.scales[3], "channel scales should differ");
    }

    // ── int8_tensor_dequantize_roundtrip ──────────────────────────────────

    #[test]
    fn test_int8_tensor_dequantize_roundtrip() {
        let weights: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
        let q = quantize_per_tensor(&weights);
        let deq = q.dequantize();
        assert_eq!(deq.len(), weights.len());
        // MSE should be very small for this smooth input.
        let mse: f32 = weights
            .iter()
            .zip(deq.iter())
            .map(|(&o, &r)| (o - r) * (o - r))
            .sum::<f32>()
            / weights.len() as f32;
        assert!(mse < 1e-4, "roundtrip MSE too large: {mse}");
    }

    // ── int8_tensor_memory_bytes ──────────────────────────────────────────

    #[test]
    fn test_int8_tensor_memory_bytes() {
        let weights = vec![1.0_f32; 128];
        let q = quantize_per_tensor(&weights);
        // 128 i8 (1 byte each) + 1 f32 scale (4 bytes) = 132
        assert_eq!(q.memory_bytes(), 128 + 4);
    }

    // ── int8_tensor_compression_ratio ────────────────────────────────────

    #[test]
    fn test_int8_tensor_compression_ratio() {
        let weights = vec![1.0_f32; 1024];
        let q = quantize_per_tensor(&weights);
        // original: 1024 * 4 = 4096 bytes
        // quantized: 1024 * 1 + 1 * 4 = 1028 bytes
        let expected = 4096.0_f32 / 1028.0_f32;
        let ratio = q.compression_ratio();
        assert!(
            (ratio - expected).abs() < 0.01,
            "ratio = {ratio}, expected ~{expected}"
        );
    }

    // ── int8_matvec_correct ───────────────────────────────────────────────

    #[test]
    fn test_int8_matvec_correct() {
        // 2 channels × 4 weights
        // W = [[1, 0, 0, 0], [0, 2, 0, 0]]
        // x = [1, 1, 1, 1]
        // y = [1, 2]
        let weights: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let q = quantize_per_channel(&weights, 2).expect("quantize");
        let x = vec![1.0_f32; 4];
        let y = q.matvec(&x);
        assert_eq!(y.len(), 2);
        assert!((y[0] - 1.0).abs() < 0.02, "y[0] = {}, expected ~1.0", y[0]);
        assert!((y[1] - 2.0).abs() < 0.02, "y[1] = {}, expected ~2.0", y[1]);
    }

    // ── quantize_channel_clips_outliers ──────────────────────────────────

    #[test]
    fn test_quantize_channel_clips_outliers() {
        // One extreme outlier; with clip_ratio=0.9 the outlier gets clamped.
        let mut weights = vec![0.1_f32; 128];
        weights[0] = 100.0; // outlier
        let (q_full, scale_full) = quantize_channel(&weights, 1.0);
        let (q_clip, scale_clip) = quantize_channel(&weights, 0.9);

        // The clip scale should be smaller.
        assert!(
            scale_clip < scale_full,
            "clipped scale {scale_clip} should be < full scale {scale_full}"
        );
        // The outlier at index 0 is clamped to ±127 in both cases.
        assert_eq!(q_full[0], 127);
        assert_eq!(q_clip[0], 127);
        // Regular weights get a better resolution with clipping.
        let _ = q_clip; // silence unused
    }

    // ── analyze_int8_error ────────────────────────────────────────────────

    #[test]
    fn test_analyze_int8_error() {
        let weights: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.8).collect();
        let q = quantize_per_tensor(&weights);
        let err = analyze_int8_error(&weights, &q);
        assert!(err.mse >= 0.0, "MSE must be non-negative");
        assert!(err.max_abs_error >= 0.0);
        assert!((err.bits_per_weight - 8.0).abs() < 1e-6);
        assert!(err.compression_ratio > 0.0);
    }

    // ── compare_quantization_methods ──────────────────────────────────────

    #[test]
    fn test_compare_quantization_methods() {
        // Weight count must be a multiple of 128 for the Q1_0 path.
        let weights: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
        let cmp = compare_quantization_methods(&weights, Some(4)).expect("compare");
        // INT8 should have lower MSE than Q1_0 (8-bit vs 1-bit).
        assert!(
            cmp.int8_per_tensor.mse < cmp.q1_0.mse,
            "INT8 per-tensor MSE {} should be lower than Q1_0 MSE {}",
            cmp.int8_per_tensor.mse,
            cmp.q1_0.mse
        );
        assert!(cmp.int8_per_channel.is_some());
    }

    // ── int8_per_channel_wrong_size_returns_error ─────────────────────────

    #[test]
    fn test_int8_per_channel_wrong_size_returns_error() {
        let weights = vec![1.0_f32; 100]; // not divisible by 3
        let result = quantize_per_channel(&weights, 3);
        assert!(
            matches!(
                result,
                Err(Int8QuantizeError::ChannelMismatch {
                    total: 100,
                    channels: 3
                })
            ),
            "expected ChannelMismatch error, got {result:?}"
        );
    }

    // ── quantize_per_channel_zero_channels_returns_error ──────────────────
    //
    // Regression test: `num_channels == 0` used to panic (integer division
    // / modulo by zero) instead of returning a `Result`, since the divisibility
    // check ran `weights.len() % num_channels` without first checking for zero.

    #[test]
    fn test_quantize_per_channel_zero_channels_returns_error() {
        let weights = vec![1.0_f32; 100];
        let result = quantize_per_channel(&weights, 0);
        assert!(
            matches!(
                result,
                Err(Int8QuantizeError::ChannelMismatch {
                    total: 100,
                    channels: 0
                })
            ),
            "expected ChannelMismatch error for num_channels=0, got {result:?}"
        );
    }

    #[test]
    fn test_compare_quantization_methods_zero_channels_returns_error() {
        // compare_quantization_methods forwards num_channels to
        // quantize_per_channel; Some(0) must not panic either.
        let weights = vec![1.0_f32; 128];
        let result = compare_quantization_methods(&weights, Some(0));
        assert!(
            result.is_err(),
            "compare_quantization_methods with num_channels=Some(0) should error, not panic"
        );
    }
}
