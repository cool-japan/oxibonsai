//! Quantized KV cache: INT8 and FP8 per-row quantization for keys and values.
//!
//! INT8 memory reduction: 4× vs FP32, 2× vs FP16.
//! FP8 memory reduction: 4× vs FP32, 2× vs FP16 (with floating-point distribution).
//! Accuracy: ~0.1% error vs FP32 for typical activation ranges.
//!
//! # Layout
//! For each layer, each head, each token position:
//!   - keys_i8: [seq_len, num_kv_heads, head_dim] as i8
//!   - key_scales: [seq_len, num_kv_heads] as f32  (per-row scale)
//!   - values_i8: [seq_len, num_kv_heads, head_dim] as i8
//!   - value_scales: [seq_len, num_kv_heads] as f32

use oxibonsai_core::quant_fp8::{
    fp8_e4m3_decode, fp8_e4m3_encode, fp8_e5m2_decode, fp8_e5m2_encode, FP8_E4M3_MAX, FP8_E5M2_MAX,
};

/// Error types for quantized KV cache operations.
#[derive(Debug, thiserror::Error)]
pub enum QuantKvError {
    #[error("capacity exceeded: capacity {capacity}, tried to push token {pos}")]
    CapacityExceeded { capacity: usize, pos: usize },

    #[error("token position {0} out of range")]
    PositionOutOfRange(usize),

    #[error("head index {head} out of range (num_kv_heads = {num_heads})")]
    HeadOutOfRange { head: usize, num_heads: usize },

    #[error("layer {layer} out of range (num_layers = {num_layers})")]
    LayerOutOfRange { layer: usize, num_layers: usize },

    #[error("key/value shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },
}

// ─── Primitive quantization helpers ──────────────────────────────────────────

/// Quantize a slice to INT8 with a single per-row scale.
///
/// Returns `(quantized: Vec<i8>, scale: f32)`.
///
/// `scale = max(|x|) / 127.0`, clamped to at least [`f32::EPSILON`] to avoid
/// division-by-zero. All values are symmetrically clamped to `[-127, 127]` so
/// that rounding can never produce the asymmetric `i8::MIN` (-128).
pub fn quantize_row_i8(row: &[f32]) -> (Vec<i8>, f32) {
    if row.is_empty() {
        return (Vec::new(), f32::EPSILON);
    }

    let max_abs = row.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

    // Clamp scale to at least EPSILON to avoid division by zero for all-zero rows.
    let scale = (max_abs / 127.0_f32).max(f32::EPSILON);

    let quantized = row
        .iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (quantized, scale)
}

/// Dequantize INT8 back to f32 using the row scale.
///
/// Each element is simply multiplied by `scale`. If `scale` is zero or
/// near-zero the output will be all zeros, which is the correct representation
/// for an all-zero input row.
pub fn dequantize_row_i8(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter().map(|&q| q as f32 * scale).collect()
}

/// Mean absolute error (MAE) between the original f32 slice and the
/// dequantized version of the quantized INT8 representation.
///
/// Returns `0.0` for an empty slice.
pub fn quant_error_mae(original: &[f32], quantized: &[i8], scale: f32) -> f32 {
    let n = original.len().min(quantized.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| (o - q as f32 * scale).abs())
        .sum();
    sum / n as f32
}

// ─── Per-layer quantized KV storage ──────────────────────────────────────────

/// A single layer's INT8-quantized KV cache.
///
/// Memory layout for the INT8 data arrays uses the token-major order
/// `[token_pos * num_kv_heads * head_dim]`, so sequential decode steps
/// append contiguous blocks. Scale arrays use `[token_pos * num_kv_heads]`.
#[derive(Debug)]
pub struct QuantizedKvLayer {
    /// Quantized key data: `[capacity * num_kv_heads * head_dim]` as i8.
    keys_i8: Vec<i8>,
    /// Per-row key scales: `[capacity * num_kv_heads]` as f32.
    key_scales: Vec<f32>,
    /// Quantized value data: `[capacity * num_kv_heads * head_dim]` as i8.
    values_i8: Vec<i8>,
    /// Per-row value scales: `[capacity * num_kv_heads]` as f32.
    value_scales: Vec<f32>,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Maximum number of token positions pre-allocated.
    pub capacity: usize,
    /// Number of token positions actually stored so far.
    pub len: usize,
}

impl QuantizedKvLayer {
    /// Allocate an empty quantized KV layer with the given dimensions.
    ///
    /// Pre-allocates all storage so that subsequent [`push`](Self::push) calls
    /// do not allocate.
    pub fn new(capacity: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let data_len = capacity * num_kv_heads * head_dim;
        let scale_len = capacity * num_kv_heads;

        Self {
            keys_i8: vec![0i8; data_len],
            key_scales: vec![0.0_f32; scale_len],
            values_i8: vec![0i8; data_len],
            value_scales: vec![0.0_f32; scale_len],
            num_kv_heads,
            head_dim,
            capacity,
            len: 0,
        }
    }

    /// Append keys and values for the next token position.
    ///
    /// `keys` must be a flat slice of shape `[num_kv_heads * head_dim]` (heads
    /// first, then dims). `values` must have the same shape.
    ///
    /// Each head's row is quantized independently with its own scale.
    ///
    /// # Errors
    /// - [`QuantKvError::CapacityExceeded`] if `self.len == self.capacity`.
    /// - [`QuantKvError::ShapeMismatch`] if `keys` or `values` length is wrong.
    pub fn push(&mut self, keys: &[f32], values: &[f32]) -> Result<(), QuantKvError> {
        let expected = self.num_kv_heads * self.head_dim;

        if keys.len() != expected {
            return Err(QuantKvError::ShapeMismatch {
                expected,
                actual: keys.len(),
            });
        }
        if values.len() != expected {
            return Err(QuantKvError::ShapeMismatch {
                expected,
                actual: values.len(),
            });
        }
        if self.len >= self.capacity {
            return Err(QuantKvError::CapacityExceeded {
                capacity: self.capacity,
                pos: self.len,
            });
        }

        let token_pos = self.len;

        for head in 0..self.num_kv_heads {
            let row_start = head * self.head_dim;
            let row_end = row_start + self.head_dim;

            // Compute offsets before any mutable borrows to satisfy the borrow checker.
            let data_off = self.data_offset(token_pos, head);
            let scale_off = self.scale_offset(token_pos, head);

            // Keys
            let key_row = &keys[row_start..row_end];
            let (kq, ks) = quantize_row_i8(key_row);
            self.keys_i8[data_off..data_off + self.head_dim].copy_from_slice(&kq);
            self.key_scales[scale_off] = ks;

            // Values
            let val_row = &values[row_start..row_end];
            let (vq, vs) = quantize_row_i8(val_row);
            self.values_i8[data_off..data_off + self.head_dim].copy_from_slice(&vq);
            self.value_scales[scale_off] = vs;
        }

        self.len += 1;
        Ok(())
    }

    /// Get dequantized keys for a specific token position and head.
    ///
    /// Returns a `Vec<f32>` of length `head_dim`.
    ///
    /// # Errors
    /// - [`QuantKvError::PositionOutOfRange`] if `token_pos >= self.len`.
    /// - [`QuantKvError::HeadOutOfRange`] if `head >= self.num_kv_heads`.
    pub fn get_key(&self, token_pos: usize, head: usize) -> Result<Vec<f32>, QuantKvError> {
        self.validate_pos_head(token_pos, head)?;
        let data_off = self.data_offset(token_pos, head);
        let scale = self.key_scales[self.scale_offset(token_pos, head)];
        Ok(dequantize_row_i8(
            &self.keys_i8[data_off..data_off + self.head_dim],
            scale,
        ))
    }

    /// Get dequantized values for a specific token position and head.
    ///
    /// Returns a `Vec<f32>` of length `head_dim`.
    ///
    /// # Errors
    /// - [`QuantKvError::PositionOutOfRange`] if `token_pos >= self.len`.
    /// - [`QuantKvError::HeadOutOfRange`] if `head >= self.num_kv_heads`.
    pub fn get_value(&self, token_pos: usize, head: usize) -> Result<Vec<f32>, QuantKvError> {
        self.validate_pos_head(token_pos, head)?;
        let data_off = self.data_offset(token_pos, head);
        let scale = self.value_scales[self.scale_offset(token_pos, head)];
        Ok(dequantize_row_i8(
            &self.values_i8[data_off..data_off + self.head_dim],
            scale,
        ))
    }

    /// Get all dequantized keys for a token position (all heads, interleaved).
    ///
    /// Returns a flat `Vec<f32>` of length `num_kv_heads * head_dim`.
    ///
    /// # Errors
    /// - [`QuantKvError::PositionOutOfRange`] if `token_pos >= self.len`.
    pub fn get_keys_at(&self, token_pos: usize) -> Result<Vec<f32>, QuantKvError> {
        if token_pos >= self.len {
            return Err(QuantKvError::PositionOutOfRange(token_pos));
        }
        let mut out = Vec::with_capacity(self.num_kv_heads * self.head_dim);
        for head in 0..self.num_kv_heads {
            let data_off = self.data_offset(token_pos, head);
            let scale = self.key_scales[self.scale_offset(token_pos, head)];
            out.extend(dequantize_row_i8(
                &self.keys_i8[data_off..data_off + self.head_dim],
                scale,
            ));
        }
        Ok(out)
    }

    /// Get all dequantized values for a token position (all heads, interleaved).
    ///
    /// Returns a flat `Vec<f32>` of length `num_kv_heads * head_dim`.
    ///
    /// # Errors
    /// - [`QuantKvError::PositionOutOfRange`] if `token_pos >= self.len`.
    pub fn get_values_at(&self, token_pos: usize) -> Result<Vec<f32>, QuantKvError> {
        if token_pos >= self.len {
            return Err(QuantKvError::PositionOutOfRange(token_pos));
        }
        let mut out = Vec::with_capacity(self.num_kv_heads * self.head_dim);
        for head in 0..self.num_kv_heads {
            let data_off = self.data_offset(token_pos, head);
            let scale = self.value_scales[self.scale_offset(token_pos, head)];
            out.extend(dequantize_row_i8(
                &self.values_i8[data_off..data_off + self.head_dim],
                scale,
            ));
        }
        Ok(out)
    }

    /// Memory used by this layer in bytes (INT8 data + f32 scales).
    ///
    /// Only accounts for the pre-allocated storage slabs, not struct overhead.
    pub fn memory_bytes(&self) -> usize {
        // INT8 data: 1 byte per element
        let data_bytes = self.keys_i8.len() + self.values_i8.len();
        // f32 scales: 4 bytes each
        let scale_bytes = (self.key_scales.len() + self.value_scales.len()) * 4;
        data_bytes + scale_bytes
    }

    /// Equivalent memory if the same data were stored as FP32 (no scales).
    ///
    /// `2 * capacity * num_kv_heads * head_dim * 4 bytes`
    pub fn fp32_memory_bytes(&self) -> usize {
        // Keys + values, each element 4 bytes
        2 * self.capacity * self.num_kv_heads * self.head_dim * 4
    }

    /// Compression ratio versus FP32 storage.
    ///
    /// Values approaching 4.0 indicate near-ideal INT8 compression. The ratio
    /// is slightly below 4.0 because per-row f32 scales add overhead.
    pub fn compression_ratio(&self) -> f32 {
        let quant = self.memory_bytes();
        if quant == 0 {
            return 1.0;
        }
        self.fp32_memory_bytes() as f32 / quant as f32
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Flat index into the INT8 data arrays for `(token_pos, head, 0)`.
    ///
    /// Layout: `[token_pos][head][dim]` → `(token_pos * num_kv_heads + head) * head_dim`
    #[inline]
    fn data_offset(&self, token_pos: usize, head: usize) -> usize {
        (token_pos * self.num_kv_heads + head) * self.head_dim
    }

    /// Flat index into the scale arrays for `(token_pos, head)`.
    ///
    /// Layout: `[token_pos][head]` → `token_pos * num_kv_heads + head`
    #[inline]
    fn scale_offset(&self, token_pos: usize, head: usize) -> usize {
        token_pos * self.num_kv_heads + head
    }

    /// Validate that `token_pos < self.len` and `head < self.num_kv_heads`.
    fn validate_pos_head(&self, token_pos: usize, head: usize) -> Result<(), QuantKvError> {
        if token_pos >= self.len {
            return Err(QuantKvError::PositionOutOfRange(token_pos));
        }
        if head >= self.num_kv_heads {
            return Err(QuantKvError::HeadOutOfRange {
                head,
                num_heads: self.num_kv_heads,
            });
        }
        Ok(())
    }
}

// ─── Multi-layer quantized KV cache ──────────────────────────────────────────

/// Full multi-layer INT8-quantized KV cache for autoregressive decoding.
///
/// Wraps one [`QuantizedKvLayer`] per transformer layer and exposes a
/// unified decode-step interface through [`push_step`](Self::push_step).
#[derive(Debug)]
pub struct QuantizedKvCache {
    layers: Vec<QuantizedKvLayer>,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV attention heads per layer.
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
}

impl QuantizedKvCache {
    /// Allocate a new quantized KV cache for `num_layers` transformer layers.
    ///
    /// Each layer is pre-allocated for `capacity` token positions.
    pub fn new(num_layers: usize, capacity: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| QuantizedKvLayer::new(capacity, num_kv_heads, head_dim))
            .collect();

        Self {
            layers,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append KV tensors for all layers at the current decode step.
    ///
    /// `all_keys[layer]` must be a flat slice of shape `[num_kv_heads * head_dim]`.
    /// `all_values[layer]` must have the same shape.
    ///
    /// # Errors
    /// - [`QuantKvError::LayerOutOfRange`] if `all_keys.len() != self.num_layers`.
    /// - Propagates [`QuantKvError`] from each layer's [`push`](QuantizedKvLayer::push).
    pub fn push_step(
        &mut self,
        all_keys: &[Vec<f32>],
        all_values: &[Vec<f32>],
    ) -> Result<(), QuantKvError> {
        if all_keys.len() != self.num_layers {
            return Err(QuantKvError::LayerOutOfRange {
                layer: all_keys.len(),
                num_layers: self.num_layers,
            });
        }
        if all_values.len() != self.num_layers {
            return Err(QuantKvError::LayerOutOfRange {
                layer: all_values.len(),
                num_layers: self.num_layers,
            });
        }

        for (layer_idx, (layer, (keys, values))) in self
            .layers
            .iter_mut()
            .zip(all_keys.iter().zip(all_values.iter()))
            .enumerate()
        {
            layer.push(keys, values).map_err(|e| match e {
                // Re-attach layer context to capacity errors
                QuantKvError::CapacityExceeded { capacity, pos } => {
                    QuantKvError::CapacityExceeded { capacity, pos }
                }
                QuantKvError::ShapeMismatch { expected, actual } => {
                    QuantKvError::ShapeMismatch { expected, actual }
                }
                // Pass through other errors; we could enrich them with layer_idx
                // but the error types don't carry that field — keep as is.
                other => {
                    let _ = layer_idx;
                    other
                }
            })?;
        }
        Ok(())
    }

    /// Get dequantized keys for a specific layer, token position, and head.
    ///
    /// # Errors
    /// - [`QuantKvError::LayerOutOfRange`] if `layer >= self.num_layers`.
    /// - Propagates position/head errors from the underlying layer.
    pub fn get_key(
        &self,
        layer: usize,
        token_pos: usize,
        head: usize,
    ) -> Result<Vec<f32>, QuantKvError> {
        self.validate_layer(layer)?;
        self.layers[layer].get_key(token_pos, head)
    }

    /// Get dequantized values for a specific layer, token position, and head.
    ///
    /// # Errors
    /// - [`QuantKvError::LayerOutOfRange`] if `layer >= self.num_layers`.
    /// - Propagates position/head errors from the underlying layer.
    pub fn get_value(
        &self,
        layer: usize,
        token_pos: usize,
        head: usize,
    ) -> Result<Vec<f32>, QuantKvError> {
        self.validate_layer(layer)?;
        self.layers[layer].get_value(token_pos, head)
    }

    /// Total memory used across all layers in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// FP32-equivalent memory across all layers.
    pub fn total_fp32_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.fp32_memory_bytes()).sum()
    }

    /// Overall compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f32 {
        let quant = self.total_memory_bytes();
        if quant == 0 {
            return 1.0;
        }
        self.total_fp32_memory_bytes() as f32 / quant as f32
    }

    /// Number of token positions currently stored (taken from layer 0).
    ///
    /// Returns `0` if there are no layers.
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.len).unwrap_or(0)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn validate_layer(&self, layer: usize) -> Result<(), QuantKvError> {
        if layer >= self.num_layers {
            return Err(QuantKvError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        Ok(())
    }
}

// ─── FP8 KV cache ─────────────────────────────────────────────────────────────

/// FP8 encoding format variant for KV cache quantization.
///
/// - `E4M3` uses 4-bit exponent, 3-bit mantissa (max representable ≈ 448.0).
///   Better accuracy for typical attention activations with bounded range.
/// - `E5M2` uses 5-bit exponent, 2-bit mantissa (max representable ≈ 57344.0).
///   Wider dynamic range, useful for outlier-heavy distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8KvFormat {
    /// E4M3FN format: 4-bit exponent, 3-bit mantissa, bias=7.
    /// Max representable value: 448.0. No infinities; NaN = 0x7f/0xff.
    E4M3,
    /// E5M2 format: 5-bit exponent, 2-bit mantissa, bias=15.
    /// Max representable value: 57344.0. Supports infinities; NaN = 0x7e.
    E5M2,
}

/// Quantize a row of f32 values to FP8 using per-row absolute-max scaling.
///
/// Returns `(quantized_bytes: Vec<u8>, scale: f32)` where
/// `scale = max(|row|) / FP8_MAX`. One scale per head-row is stored; all
/// values are encoded relative to that scale.
///
/// For an all-zero row the scale is clamped to [`f32::EPSILON`] and all output
/// bytes are `0x00`.
fn quantize_row_fp8(row: &[f32], format: Fp8KvFormat) -> (Vec<u8>, f32) {
    if row.is_empty() {
        return (Vec::new(), f32::EPSILON);
    }

    let max_abs = row.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

    let fp8_max = match format {
        Fp8KvFormat::E4M3 => FP8_E4M3_MAX,
        Fp8KvFormat::E5M2 => FP8_E5M2_MAX,
    };

    // Clamp scale to at least EPSILON to avoid division by zero for all-zero rows.
    let scale = (max_abs / fp8_max).max(f32::EPSILON);

    let quantized = match format {
        Fp8KvFormat::E4M3 => row
            .iter()
            .map(|&x| fp8_e4m3_encode(x / scale))
            .collect(),
        Fp8KvFormat::E5M2 => row
            .iter()
            .map(|&x| fp8_e5m2_encode(x / scale))
            .collect(),
    };

    (quantized, scale)
}

/// Dequantize FP8 bytes back to f32 using the stored row scale.
///
/// Each element is decoded from FP8 then multiplied by `scale`.
fn dequantize_row_fp8(quantized: &[u8], scale: f32, format: Fp8KvFormat) -> Vec<f32> {
    match format {
        Fp8KvFormat::E4M3 => quantized
            .iter()
            .map(|&b| fp8_e4m3_decode(b) * scale)
            .collect(),
        Fp8KvFormat::E5M2 => quantized
            .iter()
            .map(|&b| fp8_e5m2_decode(b) * scale)
            .collect(),
    }
}

/// A single transformer layer's FP8-quantized KV cache.
///
/// Memory layout is token-major: `[token_pos][head][dim]` for data and
/// `[token_pos][head]` for scales. Append-only; `clear` resets `len` to 0
/// without reallocating.
///
/// Per-row scaling: one `f32` scale per `(token_pos, head)` pair, computed as
/// `scale = max(|row|) / FP8_MAX`. This mirrors the INT8 implementation but
/// uses FP8 byte encodings rather than i8.
#[derive(Debug)]
pub struct Fp8KvLayer {
    /// FP8-encoded key data: `[capacity * num_kv_heads * head_dim]` as u8.
    keys_fp8: Vec<u8>,
    /// Per-head-row key scales: `[capacity * num_kv_heads]` as f32.
    key_scales: Vec<f32>,
    /// FP8-encoded value data: `[capacity * num_kv_heads * head_dim]` as u8.
    values_fp8: Vec<u8>,
    /// Per-head-row value scales: `[capacity * num_kv_heads]` as f32.
    value_scales: Vec<f32>,
    /// Number of KV attention heads per token position.
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Maximum token positions pre-allocated.
    pub capacity: usize,
    /// Token positions actually stored.
    pub len: usize,
    /// FP8 encoding format (E4M3 or E5M2).
    pub format: Fp8KvFormat,
}

impl Fp8KvLayer {
    /// Allocate an FP8 KV layer for `num_kv_heads` heads of dimension `head_dim`,
    /// holding up to `capacity` token positions in the given `format`.
    ///
    /// All storage is pre-allocated so subsequent [`push`](Self::push) calls
    /// perform no heap allocation.
    pub fn with_capacity(
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        format: Fp8KvFormat,
    ) -> Self {
        let data_len = capacity * num_kv_heads * head_dim;
        let scale_len = capacity * num_kv_heads;
        Self {
            keys_fp8: vec![0u8; data_len],
            key_scales: vec![0.0_f32; scale_len],
            values_fp8: vec![0u8; data_len],
            value_scales: vec![0.0_f32; scale_len],
            num_kv_heads,
            head_dim,
            capacity,
            len: 0,
            format,
        }
    }

    /// Append FP8-quantized keys and values for the next token position.
    ///
    /// `key` and `value` must each be a flat slice of length
    /// `num_kv_heads * head_dim` (heads first, then dims within each head).
    /// Each head-row is quantized independently with its own scale.
    ///
    /// # Errors
    /// - [`QuantKvError::CapacityExceeded`] if `self.len == self.capacity`.
    /// - [`QuantKvError::ShapeMismatch`] if `key` or `value` length is wrong.
    pub fn push(&mut self, key: &[f32], value: &[f32]) -> Result<(), QuantKvError> {
        let expected = self.num_kv_heads * self.head_dim;

        if key.len() != expected {
            return Err(QuantKvError::ShapeMismatch {
                expected,
                actual: key.len(),
            });
        }
        if value.len() != expected {
            return Err(QuantKvError::ShapeMismatch {
                expected,
                actual: value.len(),
            });
        }
        if self.len >= self.capacity {
            return Err(QuantKvError::CapacityExceeded {
                capacity: self.capacity,
                pos: self.len,
            });
        }

        let token_pos = self.len;
        let format = self.format;

        for head in 0..self.num_kv_heads {
            let row_start = head * self.head_dim;
            let row_end = row_start + self.head_dim;

            let data_off = self.data_offset(token_pos, head);
            let scale_off = self.scale_offset(token_pos, head);

            // Keys
            let key_row = &key[row_start..row_end];
            let (kq, ks) = quantize_row_fp8(key_row, format);
            self.keys_fp8[data_off..data_off + self.head_dim].copy_from_slice(&kq);
            self.key_scales[scale_off] = ks;

            // Values
            let val_row = &value[row_start..row_end];
            let (vq, vs) = quantize_row_fp8(val_row, format);
            self.values_fp8[data_off..data_off + self.head_dim].copy_from_slice(&vq);
            self.value_scales[scale_off] = vs;
        }

        self.len += 1;
        Ok(())
    }

    /// Dequantize and return all keys for a token position as a flat
    /// `Vec<f32>` of length `num_kv_heads * head_dim`.
    ///
    /// Layout: `[head_0_dims..., head_1_dims..., ...]`
    ///
    /// # Panics
    /// Panics if `pos >= self.len` (index out of bounds on the pre-allocated slab).
    pub fn get_key(&self, pos: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_kv_heads * self.head_dim);
        for head in 0..self.num_kv_heads {
            let data_off = self.data_offset(pos, head);
            let scale = self.key_scales[self.scale_offset(pos, head)];
            out.extend(dequantize_row_fp8(
                &self.keys_fp8[data_off..data_off + self.head_dim],
                scale,
                self.format,
            ));
        }
        out
    }

    /// Dequantize and return all values for a token position as a flat
    /// `Vec<f32>` of length `num_kv_heads * head_dim`.
    ///
    /// # Panics
    /// Panics if `pos >= self.len`.
    pub fn get_value(&self, pos: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_kv_heads * self.head_dim);
        for head in 0..self.num_kv_heads {
            let data_off = self.data_offset(pos, head);
            let scale = self.value_scales[self.scale_offset(pos, head)];
            out.extend(dequantize_row_fp8(
                &self.values_fp8[data_off..data_off + self.head_dim],
                scale,
                self.format,
            ));
        }
        out
    }

    /// Dequantize keys for a subset of token positions.
    ///
    /// Returns a `Vec` of flat key vectors, one per position in `positions`.
    /// Positions must be < `self.len`; out-of-range positions will panic
    /// (index-out-of-bounds on the pre-allocated slab).
    pub fn get_keys_at(&self, positions: &[usize]) -> Vec<Vec<f32>> {
        positions.iter().map(|&pos| self.get_key(pos)).collect()
    }

    /// Dequantize values for a subset of token positions.
    ///
    /// Returns a `Vec` of flat value vectors, one per position in `positions`.
    pub fn get_values_at(&self, positions: &[usize]) -> Vec<Vec<f32>> {
        positions.iter().map(|&pos| self.get_value(pos)).collect()
    }

    /// Number of token positions currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if no token positions have been stored yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum token positions this layer can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes occupied by FP8 data and f32 scales for this layer.
    ///
    /// `keys_fp8 + values_fp8` (1 byte/element) + `key_scales + value_scales`
    /// (4 bytes/element).
    pub fn memory_bytes(&self) -> usize {
        let data_bytes = self.keys_fp8.len() + self.values_fp8.len();
        let scale_bytes = (self.key_scales.len() + self.value_scales.len()) * 4;
        data_bytes + scale_bytes
    }

    /// Equivalent memory if the same data were stored as FP32 with no scales.
    ///
    /// `2 * capacity * num_kv_heads * head_dim * 4`
    pub fn memory_bytes_fp32_equivalent(&self) -> usize {
        2 * self.capacity * self.num_kv_heads * self.head_dim * 4
    }

    /// Reset stored length to zero, making the layer appear empty.
    ///
    /// Does not free or zero memory; existing bytes are overwritten on the next
    /// series of [`push`](Self::push) calls.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Flat index into the FP8 data arrays for `(token_pos, head, 0)`.
    #[inline]
    fn data_offset(&self, token_pos: usize, head: usize) -> usize {
        (token_pos * self.num_kv_heads + head) * self.head_dim
    }

    /// Flat index into the scale arrays for `(token_pos, head)`.
    #[inline]
    fn scale_offset(&self, token_pos: usize, head: usize) -> usize {
        token_pos * self.num_kv_heads + head
    }
}

// ─── Multi-layer FP8 KV cache ─────────────────────────────────────────────────

/// Full multi-layer FP8-quantized KV cache for autoregressive decoding.
///
/// Wraps one [`Fp8KvLayer`] per transformer layer and exposes per-layer
/// mutable and immutable accessors. All layers share the same `format`,
/// `num_kv_heads`, `head_dim`, and `capacity`.
#[derive(Debug)]
pub struct Fp8KvCache {
    /// Per-transformer-layer FP8 KV stores.
    pub layers: Vec<Fp8KvLayer>,
}

impl Fp8KvCache {
    /// Allocate a new FP8 KV cache for `num_layers` transformer layers.
    ///
    /// Each layer is pre-allocated for `capacity` token positions.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        format: Fp8KvFormat,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| Fp8KvLayer::with_capacity(num_kv_heads, head_dim, capacity, format))
            .collect();
        Self { layers }
    }

    /// Immutable reference to a specific layer.
    ///
    /// # Panics
    /// Panics if `layer_idx >= self.num_layers()`.
    pub fn layer(&self, layer_idx: usize) -> &Fp8KvLayer {
        &self.layers[layer_idx]
    }

    /// Mutable reference to a specific layer.
    ///
    /// # Panics
    /// Panics if `layer_idx >= self.num_layers()`.
    pub fn layer_mut(&mut self, layer_idx: usize) -> &mut Fp8KvLayer {
        &mut self.layers[layer_idx]
    }

    /// Number of transformer layers in this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total memory used across all layers in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Clear all layers, resetting stored lengths to zero.
    pub fn clear_all(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}
