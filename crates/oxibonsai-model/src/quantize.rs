//! FP32 → GGUF block quantization: the shared encoder every export and
//! convert path goes through.
//!
//! Three things live here:
//!
//! * The Q1\_0\_g128 codec (below), OxiBonsai's 1-bit sign format.
//! * [`ScaleRule`], which chooses between the llama.cpp `amax` convention and
//!   the error-minimising estimators a *continuous* source needs (CQ-03),
//!   with an exact already-ternary detector so a ternary checkpoint still
//!   re-encodes bit-for-bit.
//! * [`encode_quantized_tensor`], which turns an f32 tensor into any GGUF
//!   tensor type's wire bytes — including the Bonsai 2 formats `PQ2_0`,
//!   `PTQ1_0` and mainline `Q2_0` — after checking that the tensor's first
//!   dimension is a whole number of blocks (CQ-14).
//!
//! # Q1\_0\_g128 block format
//!
//! Each block covers exactly `GROUP_SIZE` (128) weights:
//!
//! ```text
//! ┌──────────────────┬──────────────────────────────────────────────────────┐
//! │  2 bytes         │  16 bytes                                            │
//! │  FP16 scale      │  128 sign bits (1 bit per weight)                   │
//! │  = max(|w_i|)    │  bit=1 → +scale, bit=0 → −scale                     │
//! └──────────────────┴──────────────────────────────────────────────────────┘
//! ```
//!
//! This is the canonical Q1\_0\_g128 sign convention shared by every reader
//! and kernel in the workspace: [`oxibonsai_core::tensor::BlockQ1_0G128`]
//! (`w[i] = bit[i] ? +d : -d`), the CPU/CUDA/Metal GEMV and GEMM kernels in
//! `oxibonsai-kernels`, and `oxibonsai-model`'s GGUF weight loaders. The
//! encoder and decoder below MUST stay in lock-step with that convention —
//! a mismatch silently negates every 1-bit weight after export→load.
//!
//! Total block size: **18 bytes** per 128 weights → ~1.125 bits/weight.

use std::borrow::Cow;

use half::f16;
use oxibonsai_core::gguf::writer::TensorType;

/// Number of weights per quantization group.
pub const GROUP_SIZE: usize = 128;

/// Byte size of one encoded block.
///
/// Layout: `[f16 scale: 2 bytes][sign bits: 16 bytes]`
pub const BLOCK_BYTES: usize = 18; // 2 (f16) + 16 (sign bits)

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can arise during block quantization or dequantization.
///
/// Every variant is reachable and every variant names the tensor it happened
/// on (CQ-20): a bare "input length 130 is not a multiple of 128" is useless
/// in a 402-tensor conversion. Primitives that genuinely do not know the name
/// construct the error with an empty `tensor` and the caller attaches it via
/// [`QuantizeError::with_tensor`].
///
/// The historical `ZeroGroup` variant has been removed: nothing in the
/// workspace ever constructed it (an all-zero group encodes as a zero scale
/// with cleared sign bits, which is well-defined), so it was dead surface
/// area that implied a failure mode the encoder does not have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QuantizeError {
    /// Input slice length is not a multiple of the format's group size.
    #[error(
        "tensor '{}': input length {got} is not a multiple of the {group}-element group",
        tensor_label(.tensor)
    )]
    NotAligned {
        /// Tensor the failure happened on (empty when the caller had no name).
        tensor: String,
        /// The offending element count.
        got: usize,
        /// The group size that was required.
        group: usize,
    },

    /// Encoded data length is not a multiple of the format's block size.
    #[error(
        "tensor '{}': data length {got} is not a multiple of the {block}-byte block",
        tensor_label(.tensor)
    )]
    InvalidBlockData {
        /// Tensor the failure happened on (empty when the caller had no name).
        tensor: String,
        /// The offending byte count.
        got: usize,
        /// The block size that was required.
        block: usize,
    },

    /// The first (fastest-varying) dimension of a tensor is not a whole
    /// number of blocks, so ggml's per-row quantization cannot represent it.
    ///
    /// ggml quantizes each row independently and hard-rejects such a tensor
    /// (`ggml/src/gguf.cpp:721-727`), so flat-padding the whole flattened
    /// tensor — what this crate used to do — silently straddles group
    /// boundaries across rows (CQ-14).
    #[error(
        "tensor '{}': first dimension {ne0} is not a multiple of the {group}-element group; \
         ggml quantizes each row independently",
        tensor_label(.tensor)
    )]
    RowNotBlockAligned {
        /// Tensor the failure happened on (empty when the caller had no name).
        tensor: String,
        /// The offending first dimension.
        ne0: usize,
        /// The group size that was required.
        group: usize,
    },

    /// A block encoder in `oxibonsai-core` rejected the input.
    #[error("tensor '{}': {reason}", tensor_label(.tensor))]
    Encoder {
        /// Tensor the failure happened on (empty when the caller had no name).
        tensor: String,
        /// The underlying encoder message.
        reason: String,
    },
}

/// Render a possibly-empty tensor name for an error message.
fn tensor_label(name: &str) -> &str {
    if name.is_empty() {
        "<unnamed>"
    } else {
        name
    }
}

impl QuantizeError {
    /// Attach (or replace) the tensor name on an error raised by a
    /// name-less primitive.
    #[must_use]
    pub fn with_tensor(mut self, name: &str) -> Self {
        let slot = match &mut self {
            Self::NotAligned { tensor, .. }
            | Self::InvalidBlockData { tensor, .. }
            | Self::RowNotBlockAligned { tensor, .. }
            | Self::Encoder { tensor, .. } => tensor,
        };
        name.clone_into(slot);
        self
    }

    /// The tensor this error refers to, or `None` when it was raised without
    /// a name and never annotated.
    pub fn tensor(&self) -> Option<&str> {
        let name = match self {
            Self::NotAligned { tensor, .. }
            | Self::InvalidBlockData { tensor, .. }
            | Self::RowNotBlockAligned { tensor, .. }
            | Self::Encoder { tensor, .. } => tensor.as_str(),
        };
        (!name.is_empty()).then_some(name)
    }
}

// ─── Core group-level primitives ─────────────────────────────────────────────

/// Quantize a single group of exactly `GROUP_SIZE` f32 values into one block.
///
/// The group **must** have at least one non-zero element; if every element is
/// zero, the block is written with a zero scale and all sign bits cleared, but
/// no error is returned (the dequantised values will simply all be zero).
pub fn quantize_group(weights: &[f32]) -> [u8; BLOCK_BYTES] {
    quantize_group_with(weights, ScaleRule::AbsMax)
}

/// Quantize a single group of exactly `GROUP_SIZE` f32 values into one block
/// using an explicit [`ScaleRule`].
///
/// Only the *scale* depends on the rule; the sign bits are the sign of each
/// weight either way, so `AbsMean` never changes which weights are positive.
pub fn quantize_group_with(weights: &[f32], rule: ScaleRule) -> [u8; BLOCK_BYTES] {
    debug_assert_eq!(
        weights.len(),
        GROUP_SIZE,
        "quantize_group: input must be exactly {GROUP_SIZE} elements"
    );

    // ── Determine the group scale ────────────────────────────────────────
    // `AbsMax` keeps the historical `max |w_i|`. `AbsMean` uses `E|w|`,
    // which is the least-squares-optimal scale for a *sign* quantizer
    // (`argmin_d Σ (w_i − d·sign(w_i))² = Σ|w_i| / n`), but falls back to
    // absmax for an already-ternary group so re-encoding a ternary source
    // stays bit-identical (CQ-03).
    let max_abs = absmax(weights);
    let scale = match rule {
        ScaleRule::AbsMax => max_abs,
        ScaleRule::AbsMean if is_ternary_group(weights) => max_abs,
        ScaleRule::AbsMean => mean_abs(weights),
    };

    let mut block = [0u8; BLOCK_BYTES];

    // Store FP16 scale in the first two bytes.
    let scale_f16 = f16::from_f32(scale);
    let scale_bits = scale_f16.to_bits();
    block[0] = (scale_bits & 0xFF) as u8;
    block[1] = (scale_bits >> 8) as u8;

    // ── Encode sign bits ─────────────────────────────────────────────────
    // bit = 1 → positive (≥ 0), bit = 0 → negative (< 0).
    //
    // This MUST match the canonical convention used by
    // `oxibonsai_core::tensor::BlockQ1_0G128::weight` and every consumer of
    // GGUF tensor type 41 (kernels, weight loaders) — see module docs.
    for (i, &w) in weights.iter().enumerate() {
        if w >= 0.0 {
            let byte_idx = i / 8 + 2; // +2 to skip the scale bytes
            let bit_idx = i % 8;
            block[byte_idx] |= 1 << bit_idx;
        }
    }

    block
}

/// Dequantize a single 18-byte block back to `GROUP_SIZE` f32 values.
pub fn dequantize_block(block: &[u8; BLOCK_BYTES]) -> [f32; GROUP_SIZE] {
    let scale_bits = u16::from(block[0]) | (u16::from(block[1]) << 8);
    let scale = f16::from_bits(scale_bits).to_f32();

    let mut out = [0.0_f32; GROUP_SIZE];
    for (i, slot) in out.iter_mut().enumerate().take(GROUP_SIZE) {
        let byte_idx = i / 8 + 2;
        let bit_idx = i % 8;
        let sign_bit = (block[byte_idx] >> bit_idx) & 1;
        *slot = if sign_bit != 0 { scale } else { -scale };
    }
    out
}

// ─── Slice-level API ─────────────────────────────────────────────────────────

/// Quantize a slice of f32 weights to Q1\_0\_g128 format.
///
/// The input length must be a multiple of [`GROUP_SIZE`].
///
/// Returns a `Vec<u8>` whose length is
/// `(weights.len() / GROUP_SIZE) * BLOCK_BYTES`.
pub fn quantize_q1_0_g128(weights: &[f32]) -> Result<Vec<u8>, QuantizeError> {
    quantize_q1_0_g128_with(weights, ScaleRule::AbsMax)
}

/// Quantize a slice of f32 weights to Q1\_0\_g128 using an explicit
/// [`ScaleRule`].
///
/// [`ScaleRule::AbsMax`] reproduces [`quantize_q1_0_g128`] byte for byte.
/// [`ScaleRule::AbsMean`] switches continuous (dense) groups to the
/// MSE-optimal sign-quantizer scale `E|w|` while leaving already-ternary
/// groups on `amax` — see [`ScaleRule`] for why that carve-out exists.
pub fn quantize_q1_0_g128_with(weights: &[f32], rule: ScaleRule) -> Result<Vec<u8>, QuantizeError> {
    if !weights.len().is_multiple_of(GROUP_SIZE) {
        return Err(QuantizeError::NotAligned {
            tensor: String::new(),
            got: weights.len(),
            group: GROUP_SIZE,
        });
    }

    let num_blocks = weights.len() / GROUP_SIZE;
    let mut out = Vec::with_capacity(num_blocks * BLOCK_BYTES);

    for chunk in weights.as_chunks::<GROUP_SIZE>().0 {
        let block = quantize_group_with(chunk, rule);
        out.extend_from_slice(&block);
    }

    Ok(out)
}

/// Dequantize Q1\_0\_g128 bytes back to f32 weights.
///
/// The input length must be a multiple of [`BLOCK_BYTES`].
pub fn dequantize_q1_0_g128(data: &[u8]) -> Result<Vec<f32>, QuantizeError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantizeError::InvalidBlockData {
            tensor: String::new(),
            got: data.len(),
            block: BLOCK_BYTES,
        });
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(num_blocks * GROUP_SIZE);

    for block in data.as_chunks::<BLOCK_BYTES>().0 {
        let decoded = dequantize_block(block);
        out.extend_from_slice(&decoded);
    }

    Ok(out)
}

// ─── Scale rule ───────────────────────────────────────────────────────────────

/// How a per-group quantization scale is derived from the group's weights.
///
/// # Why this is a choice and not a constant (CQ-03)
///
/// `d = amax` with `q = round(w/d)` is *exactly* llama.cpp's
/// `quantize_row_tq2_0_ref` / `quantize_row_pq2_0_ref` convention, and for a
/// source that is already `{-a, 0, +a}` it is **lossless** — every shipped
/// Bonsai GGUF, the MLX affine 2-bit pack and the ONNX `MatMulNBits` path are
/// such sources, and re-encoding them must stay bit-identical.
///
/// On a *continuous* source (raw HuggingFace f32/bf16 weights) absmax is the
/// wrong estimator: one outlier sets the scale for the whole group, so
/// reconstruction over-shoots every remaining weight and layer outputs inflate
/// by roughly 3.5×. For those inputs the standard estimators are
///
/// * ternary (`{-1,0,+1}`): threshold `Δ = 0.7·E|w|`, then the least-squares
///   scale for the surviving mask, `d = Σ_{|w|≥Δ}|w| / |{|w|≥Δ}|` (TWN);
/// * sign-only (`{-1,+1}`, i.e. Q1\_0): `d = E|w|`, no threshold.
///
/// so the library default stays [`ScaleRule::AbsMax`] and only the
/// dense-source call sites (`convert_hf_to_gguf`, `convert_onnx_to_gguf`, and
/// `export`/`quantize` on a non-ternary tensor) opt into
/// [`ScaleRule::AbsMean`].
///
/// [`ScaleRule::AbsMean`] additionally *detects* an already-ternary group and
/// falls back to absmax for it, so the lossless property above survives even
/// when a caller selects `AbsMean` for a whole model whose tensors are mixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleRule {
    /// `d = max |w_i|` — llama.cpp's convention, lossless for a ternary source.
    #[default]
    AbsMax,
    /// Error-minimising scale for a continuous source, with an automatic
    /// absmax fallback for groups that are already ternary.
    AbsMean,
}

/// Threshold factor applied to `E|w|` when ternarising a continuous group.
///
/// 0.7 is the ternary-weight-network constant: it is the value that minimises
/// `||w − d·q||²` under a Gaussian weight prior.
pub const TERNARY_THRESHOLD_FACTOR: f32 = 0.7;

/// `max |w_i|` over the slice (0.0 for an empty slice).
#[inline]
fn absmax(weights: &[f32]) -> f32 {
    weights.iter().map(|w| w.abs()).fold(0.0_f32, f32::max)
}

/// `Σ|w_i| / n` over the slice (0.0 for an empty slice).
#[inline]
fn mean_abs(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let sum: f64 = weights.iter().map(|w| f64::from(w.abs())).sum();
    (sum / weights.len() as f64) as f32
}

/// Whether every weight in the group is exactly `0`, `+amax` or `−amax`.
///
/// This is the detector that makes [`ScaleRule::AbsMean`] safe to select
/// globally: for such a group the absmax encoder is already optimal *and*
/// lossless, so re-encoding must not perturb it. Exact `f32` equality is the
/// right test — a group that came out of a ternary decoder holds literally
/// `code · d` values, and anything else is a continuous group that should be
/// re-estimated.
///
/// An all-zero group counts as ternary (both rules encode it identically).
pub fn is_ternary_group(weights: &[f32]) -> bool {
    let amax = absmax(weights);
    if amax == 0.0 {
        return true;
    }
    weights.iter().all(|&w| w == 0.0 || w == amax || w == -amax)
}

/// The scale and threshold a ternary encoder should use for one group.
///
/// Returns `(scale, threshold)`: a weight is encoded as `±1` when
/// `|w| >= threshold` (matching the `>= 0.5·amax` boundary that
/// `round(w/amax)` implies) and as `0` otherwise, and the stored FP16 scale
/// is `scale`.
pub fn ternary_group_scale(weights: &[f32], rule: ScaleRule) -> (f32, f32) {
    let amax = absmax(weights);
    if amax == 0.0 {
        return (0.0, 0.0);
    }
    match rule {
        ScaleRule::AbsMax => (amax, 0.5 * amax),
        ScaleRule::AbsMean if is_ternary_group(weights) => (amax, 0.5 * amax),
        ScaleRule::AbsMean => {
            let threshold = TERNARY_THRESHOLD_FACTOR * mean_abs(weights);
            // Least-squares scale for the surviving mask. The mask can only be
            // empty if every |w| < 0.7·E|w|, which is impossible for a
            // non-zero group (the max is always ≥ the mean), but guard anyway.
            let mut sum = 0.0_f64;
            let mut count = 0usize;
            for &w in weights {
                if w.abs() >= threshold {
                    sum += f64::from(w.abs());
                    count += 1;
                }
            }
            if count == 0 {
                (amax, 0.5 * amax)
            } else {
                ((sum / count as f64) as f32, threshold)
            }
        }
    }
}

/// Rewrite `data` as its ternary reconstruction under `rule`, group by group.
///
/// Returns [`std::borrow::Cow::Borrowed`] unchanged for [`ScaleRule::AbsMax`]
/// — every downstream block encoder already implements exactly that rule, so
/// borrowing guarantees byte-identity rather than merely reproducing it.
///
/// For [`ScaleRule::AbsMean`] each group is replaced by `code · d`, i.e. a
/// value set of `{−d, 0, +d}`. Feeding that back into an absmax block encoder
/// reproduces the intended codes and scale exactly (`amax` of the rewritten
/// group *is* `d`), which is how a single verified encoder in
/// `oxibonsai-core` — including `BlockPTQ1_0`'s delicate base-3 pointer walk —
/// serves both rules without being duplicated.
///
/// Groups that are already ternary are left byte-for-byte untouched.
pub fn ternarize_groups(data: &[f32], group_size: usize, rule: ScaleRule) -> Cow<'_, [f32]> {
    if rule == ScaleRule::AbsMax || group_size == 0 {
        return Cow::Borrowed(data);
    }
    let mut out = data.to_vec();
    for group in out.chunks_mut(group_size) {
        if is_ternary_group(group) {
            continue;
        }
        let (scale, threshold) = ternary_group_scale(group, rule);
        for w in group.iter_mut() {
            *w = if *w >= threshold {
                scale
            } else if *w <= -threshold {
                -scale
            } else {
                0.0
            };
        }
    }
    Cow::Owned(out)
}

// ─── Shared block encoders ────────────────────────────────────────────────────

/// Serialise a slice of `#[repr(C)]` block structs into raw GGUF bytes.
///
/// # Safety
///
/// `T` must be `#[repr(C)]`, contain only integer / `f16` / `f32` fields (so
/// every bit pattern is valid and there is no padding), and have a
/// compile-time-asserted size equal to its on-disk block size. `BlockQ8K`'s
/// `d: f32` (unlike every other K-quant format's `f16`) is exactly this
/// case: every `f32` bit pattern, including NaN and infinities, is a valid
/// value to reinterpret as bytes, so it needs no special-casing. Every block
/// type passed here (`BlockTQ2_0_g128`, `BlockPQ2_0`, `BlockPTQ1_0`,
/// `BlockQ2_0G64`, the `quant_std` / `quant_k` / `quant_fp8` blocks) carries
/// such an assertion in `oxibonsai-core`.
unsafe fn blocks_as_bytes<T>(blocks: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(blocks);
    // SAFETY: see the contract above — `T` is repr(C) with no padding and
    // only integer-like fields, so reinterpreting the slice as bytes is
    // sound, and `byte_len` is exactly the allocation's size.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), byte_len) };
    bytes.to_vec()
}

/// Map an `oxibonsai-core` block-encoder error into a [`QuantizeError`].
fn encoder_error(e: oxibonsai_core::error::BonsaiError) -> QuantizeError {
    QuantizeError::Encoder {
        tensor: String::new(),
        reason: e.to_string(),
    }
}

/// Validate that a tensor can be quantized to `tensor_type` at all, then
/// encode it.
///
/// `ne0` is the GGUF **first** (fastest-varying) dimension — the row length.
/// ggml quantizes each row independently, so a tensor whose `ne0` is not a
/// whole number of blocks has no valid encoding in that format and is
/// rejected with [`QuantizeError::RowNotBlockAligned`] instead of being
/// flat-padded (CQ-14). Callers that want llama.cpp's behaviour — keep such a
/// tensor in F32 — should test [`row_is_block_aligned`] first.
///
/// `rule` selects the ternary scale estimator for the three ternary formats
/// and is ignored by the others (whose block encoders own their own, unrelated
/// scale derivation).
pub fn encode_quantized_tensor(
    data: &[f32],
    ne0: usize,
    tensor_type: TensorType,
    rule: ScaleRule,
) -> Result<Vec<u8>, QuantizeError> {
    use oxibonsai_core::quant_fp8::{BlockFP8E4M3, BlockFP8E5M2};
    use oxibonsai_core::quant_k::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ8K, QK_K};
    use oxibonsai_core::quant_k_ext::{BlockQ5K, BlockQ6K};
    use oxibonsai_core::quant_prism::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64};
    use oxibonsai_core::quant_std::{BlockQ4_0, BlockQ8_0, QK_Q4_0, QK_Q8_0};
    use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;

    let group = tensor_type.block_size();
    if group > 1 {
        if ne0 == 0 || !ne0.is_multiple_of(group) {
            return Err(QuantizeError::RowNotBlockAligned {
                tensor: String::new(),
                ne0,
                group,
            });
        }
        if !data.len().is_multiple_of(group) {
            return Err(QuantizeError::NotAligned {
                tensor: String::new(),
                got: data.len(),
                group,
            });
        }
    }

    let bytes = match tensor_type {
        TensorType::F32 => encode_source_bytes(SourceDtype::F32, data),
        TensorType::F16 => encode_source_bytes(SourceDtype::F16, data),
        TensorType::BF16 => encode_source_bytes(SourceDtype::BF16, data),

        // ── Ternary family: canonicalise under `rule`, then hand the
        // already-ternary values to the verified core encoder. ────────────
        TensorType::TQ2_0_g128 => {
            let src = ternarize_groups(data, group, rule);
            let blocks = BlockTQ2_0_g128::quantize(&src).map_err(encoder_error)?;
            // SAFETY: `BlockTQ2_0_g128` is repr(C), 34 bytes (asserted).
            let bytes = unsafe { blocks_as_bytes(&blocks) };
            // Self-check: this encoder must never emit the reserved 0b11
            // code, and the same screen guards data arriving from elsewhere.
            validate_ternary_codes("", &bytes, TensorType::TQ2_0_g128)?;
            bytes
        }
        TensorType::PQ2_0 => {
            let src = ternarize_groups(data, group, rule);
            let blocks = BlockPQ2_0::quantize(&src).map_err(encoder_error)?;
            // SAFETY: `BlockPQ2_0` is repr(C), 34 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::Q2_0G64 => {
            let src = ternarize_groups(data, group, rule);
            let blocks = BlockQ2_0G64::quantize(&src).map_err(encoder_error)?;
            // SAFETY: `BlockQ2_0G64` is repr(C), 18 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::PTQ1_0 => {
            let src = ternarize_groups(data, group, rule);
            let blocks = BlockPTQ1_0::quantize(&src).map_err(encoder_error)?;
            // SAFETY: `BlockPTQ1_0` is repr(C), 28 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }

        // ── Sign-only 1-bit ───────────────────────────────────────────────
        TensorType::Q1_0G128 => quantize_q1_0_g128_with(data, rule)?,

        // ── Everything else owns its own scale derivation. ────────────────
        TensorType::Q4_0 => {
            debug_assert_eq!(group, QK_Q4_0);
            let blocks = BlockQ4_0::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ4_0` is repr(C), 18 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::Q8_0 => {
            debug_assert_eq!(group, QK_Q8_0);
            let blocks = BlockQ8_0::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ8_0` is repr(C), 34 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        // Before these arms existed, `oxibonsai quantize`'s
        // writer path (this function) had no arm for these three K-quant
        // wire types at all — they fell through to the "no encoder" refusal
        // below — even though the block encoders themselves
        // (`BlockQ2K`/`BlockQ3K`/`BlockQ8K::quantize`) already existed and
        // the *reader* side (`GgufTensorType::Q2_K`/`Q3_K`/`Q8_K`,
        // `is_executable() == true`) already loaded them. Styled on the
        // existing `Q4_K` arm immediately below.
        TensorType::Q2_K => {
            debug_assert_eq!(group, QK_K);
            let blocks = BlockQ2K::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ2K` is repr(C), 84 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::Q3_K => {
            debug_assert_eq!(group, QK_K);
            let blocks = BlockQ3K::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ3K` is repr(C), 110 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::Q4_K => {
            debug_assert_eq!(group, QK_K);
            let blocks = BlockQ4K::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ4K` is repr(C), 144 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::Q5_K => {
            let blocks = BlockQ5K::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ5K` is repr(C), 176 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::Q6_K => {
            let blocks = BlockQ6K::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ6K` is repr(C), 210 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::Q8_K => {
            debug_assert_eq!(group, QK_K);
            let blocks = BlockQ8K::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockQ8K` is repr(C), 292 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::F8_E4M3 => {
            let blocks = BlockFP8E4M3::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockFP8E4M3` is repr(C), 34 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }
        TensorType::F8_E5M2 => {
            let blocks = BlockFP8E5M2::quantize(data).map_err(encoder_error)?;
            // SAFETY: `BlockFP8E5M2` is repr(C), 34 bytes (asserted).
            unsafe { blocks_as_bytes(&blocks) }
        }

        // No encoder in this workspace produces these wire forms; refuse
        // rather than emit bytes a reader would mis-slice.
        TensorType::TQ2_0 | TensorType::TQ1_0 | TensorType::MXFP4 | TensorType::NVFP4 => {
            return Err(QuantizeError::Encoder {
                tensor: String::new(),
                reason: format!("no encoder for GGUF tensor type {tensor_type:?}"),
            })
        }
    };

    Ok(bytes)
}

/// Reject a 2-bit tensor declared ternary that contains the reserved `0b11`
/// code.
///
/// The `TQ2_0` family decodes `0b11` to **0** while `PQ2_0` / `Q2_0` decode it
/// arithmetically to **+2**, so a `PQ2_0` tensor mis-declared as ggml id 42
/// (the ambiguous id) reaches the CPU ternary decoders and silently produces
/// different numbers rather than failing. `METAL-CACHE` added this screen at
/// Metal upload time; this is the same screen on the CPU convert/export path,
/// where a tensor first enters the pipeline.
///
/// `bytes` is the raw tensor payload; `tensor_type` must be one of the 2-bit
/// types, otherwise this is a no-op.
pub fn validate_ternary_codes(
    name: &str,
    bytes: &[u8],
    tensor_type: TensorType,
) -> Result<(), QuantizeError> {
    use oxibonsai_core::quant_prism::count_plus_two_codes;

    // Only the layouts whose *declared* meaning is ternary are screened.
    // `PQ2_0` and `Q2_0G64` legitimately define `0b11`, so they are exempt.
    let (block_bytes, code_range) = match tensor_type {
        // qs first, d last → codes occupy bytes 0..32 of each 34-byte block.
        TensorType::TQ2_0_g128 => (34usize, 0..32usize),
        // qs first, d last → codes occupy bytes 0..64 of each 66-byte block
        // (`BlockTQ2_0 { qs: [u8; 64], d: f16 }` in oxibonsai-core; NOT
        // d-first — a d-first read here false-rejects clean tensors whenever
        // the FP16 scale's own bytes happen to contain a `0b11` bit pair, and
        // false-negatives a real `0b11` planted in `qs[0]`/`qs[1]`).
        TensorType::TQ2_0 => (66usize, 0..64usize),
        _ => return Ok(()),
    };

    if !bytes.len().is_multiple_of(block_bytes) {
        return Err(QuantizeError::InvalidBlockData {
            tensor: name.to_string(),
            got: bytes.len(),
            block: block_bytes,
        });
    }
    for (index, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let offending = count_plus_two_codes(&block[code_range.clone()]);
        if offending != 0 {
            return Err(QuantizeError::Encoder {
                tensor: name.to_string(),
                reason: format!(
                    "block {index} of this {tensor_type:?} tensor carries {offending} reserved \
                     0b11 code(s); the TQ2_0 family decodes 0b11 as 0 while PQ2_0/Q2_0 decode it \
                     as +2, so this tensor is almost certainly PQ2_0 data mis-declared as ggml \
                     id 42"
                ),
            });
        }
    }
    Ok(())
}

/// Whether a tensor whose first dimension is `ne0` can be quantized to
/// `tensor_type` without straddling row boundaries.
///
/// llama.cpp's own quantizer keeps a tensor in its source precision when this
/// is false rather than failing the whole run, and so does
/// [`crate::export::export_to_gguf`].
#[inline]
pub fn row_is_block_aligned(ne0: usize, tensor_type: TensorType) -> bool {
    let group = tensor_type.block_size();
    group <= 1 || (ne0 != 0 && ne0.is_multiple_of(group))
}

// ─── Source element type ──────────────────────────────────────────────────────

/// The element type a tensor had *before* it was widened to `f32`.
///
/// Decoding to `f32` is lossless for every variant here, but the information
/// is not recoverable afterwards, and a converter needs it to pick an
/// appropriate output encoding: a bf16 tensor that must stay unquantized
/// should be written back as [`SourceDtype::BF16`] rather than inflated 2× to
/// `F32` (CQ-M2). It is also what lets the converter report *what it read*
/// instead of guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceDtype {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16 (CQ-M3: the read path this crate previously lacked).
    BF16,
}

impl SourceDtype {
    /// Human-readable name, matching the GGUF/ggml spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }

    /// Bytes per element in the source encoding.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Decode a little-endian byte buffer of `dtype` elements into `f32`.
///
/// This is the shared f32/f16/**bf16** read path (CQ-M3); the bf16 arm is the
/// one `quantize` previously had no way to reach. Trailing bytes that do not
/// form a whole element are rejected rather than silently dropped.
pub fn dequant_source_bytes(dtype: SourceDtype, bytes: &[u8]) -> Result<Vec<f32>, QuantizeError> {
    let width = dtype.element_bytes();
    if !bytes.len().is_multiple_of(width) {
        return Err(QuantizeError::InvalidBlockData {
            tensor: String::new(),
            got: bytes.len(),
            block: width,
        });
    }
    let out = match dtype {
        SourceDtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        SourceDtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_le_bytes(*b).to_f32())
            .collect(),
        SourceDtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
            .collect(),
    };
    Ok(out)
}

/// Re-encode `f32` weights into the source element type's little-endian bytes.
///
/// The inverse of [`dequant_source_bytes`], used when a tensor is written back
/// unquantized in the dtype it arrived in.
pub fn encode_source_bytes(dtype: SourceDtype, data: &[f32]) -> Vec<u8> {
    match dtype {
        SourceDtype::F32 => data.iter().flat_map(|f| f.to_le_bytes()).collect(),
        SourceDtype::F16 => data
            .iter()
            .flat_map(|&f| f16::from_f32(f).to_le_bytes())
            .collect(),
        SourceDtype::BF16 => data
            .iter()
            .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
            .collect(),
    }
}

// ─── Size estimation ──────────────────────────────────────────────────────────

/// Estimate how many bytes a tensor with `num_weights` elements would occupy
/// in Q1\_0\_g128 format.
///
/// Uses ceiling division so tensors whose weight count is not a multiple of
/// [`GROUP_SIZE`] are still accounted for correctly.
#[inline]
pub fn q1_0_g128_size_bytes(num_weights: usize) -> usize {
    num_weights.div_ceil(GROUP_SIZE) * BLOCK_BYTES
}

// ─── Weight statistics ────────────────────────────────────────────────────────

/// Descriptive statistics about a weight tensor before quantization.
#[derive(Debug, Clone)]
pub struct WeightStats {
    /// Minimum weight value.
    pub min: f32,
    /// Maximum weight value.
    pub max: f32,
    /// Arithmetic mean.
    pub mean: f32,
    /// Population standard deviation.
    pub std: f32,
    /// Fraction of weights considered "near zero" (|w| < 0.01).
    pub sparsity: f32,
    /// Total number of weights.
    pub num_weights: usize,
}

/// Compute descriptive statistics for a weight slice.
///
/// Returns a [`WeightStats`] with all fields set to zero / NaN-free values
/// even for empty slices.
pub fn compute_weight_stats(weights: &[f32]) -> WeightStats {
    let num_weights = weights.len();

    if num_weights == 0 {
        return WeightStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            sparsity: 0.0,
            num_weights: 0,
        };
    }

    let mut min = weights[0];
    let mut max = weights[0];
    let mut sum = 0.0_f64;
    let mut near_zero: usize = 0;

    for &w in weights {
        if w < min {
            min = w;
        }
        if w > max {
            max = w;
        }
        sum += f64::from(w);
        if w.abs() < 0.01 {
            near_zero += 1;
        }
    }

    let mean = (sum / num_weights as f64) as f32;

    // Population standard deviation
    let variance = weights
        .iter()
        .map(|&w| {
            let diff = f64::from(w) - f64::from(mean);
            diff * diff
        })
        .sum::<f64>()
        / num_weights as f64;
    let std = variance.sqrt() as f32;

    let sparsity = near_zero as f32 / num_weights as f32;

    WeightStats {
        min,
        max,
        mean,
        std,
        sparsity,
        num_weights,
    }
}

// ─── Quantization error analysis ─────────────────────────────────────────────

/// Summary statistics comparing original f32 weights to their Q1\_0\_g128
/// encoding.
#[derive(Debug, Clone)]
pub struct QuantizationError {
    /// Mean squared error between original and reconstructed weights.
    pub mse: f32,
    /// Largest absolute difference between any single original and reconstructed weight.
    pub max_abs_error: f32,
    /// Signal-to-noise ratio in dB: `10 · log10(signal_power / noise_power)`.
    /// A higher value indicates less distortion.
    pub snr_db: f32,
    /// Effective bits per weight for this format — should be ~1.125 for Q1\_0\_g128.
    pub bits_per_weight: f32,
}

/// Analyse the quantization error between `original` f32 weights and their
/// Q1\_0\_g128 encoding in `quantized`.
///
/// `quantized` must be the byte slice produced by [`quantize_q1_0_g128`] for
/// the same `original` slice.
pub fn analyze_quantization_error(
    original: &[f32],
    quantized: &[u8],
) -> Result<QuantizationError, QuantizeError> {
    let reconstructed = dequantize_q1_0_g128(quantized)?;

    // The reconstructed slice will have length = num_blocks * GROUP_SIZE, which
    // is ≥ original.len(). We only compare against original.len() elements.
    let n = original.len();
    if n == 0 {
        return Ok(QuantizationError {
            mse: 0.0,
            max_abs_error: 0.0,
            snr_db: f32::INFINITY,
            bits_per_weight: BLOCK_BYTES as f32 * 8.0 / GROUP_SIZE as f32,
        });
    }

    let mut sum_sq_error = 0.0_f64;
    let mut max_abs_error = 0.0_f32;
    let mut signal_power = 0.0_f64;

    for i in 0..n {
        let orig = original[i];
        let recon = reconstructed[i];
        let err = orig - recon;
        sum_sq_error += f64::from(err * err);
        let abs_err = err.abs();
        if abs_err > max_abs_error {
            max_abs_error = abs_err;
        }
        signal_power += f64::from(orig * orig);
    }

    let mse = (sum_sq_error / n as f64) as f32;
    let noise_power = sum_sq_error / n as f64;

    let snr_db = if noise_power == 0.0 {
        f32::INFINITY
    } else {
        let snr_linear = (signal_power / n as f64) / noise_power;
        (10.0 * snr_linear.log10()) as f32
    };

    let bits_per_weight = BLOCK_BYTES as f32 * 8.0 / GROUP_SIZE as f32;

    Ok(QuantizationError {
        mse,
        max_abs_error,
        snr_db,
        bits_per_weight,
    })
}

// ─── Round-trip helper ────────────────────────────────────────────────────────

/// Round each weight to the nearest representable Q1\_0 value for analysis.
///
/// Each weight is replaced by either `+scale` or `−scale` where `scale` is the
/// per-group maximum absolute value.  The group boundaries are determined by
/// [`GROUP_SIZE`]; any trailing weights that do not fill a complete group are
/// handled as a short group using the scale of that partial group.
pub fn round_to_q1_0(weights: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(weights.len());

    for chunk in weights.chunks(GROUP_SIZE) {
        let max_abs = chunk.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);
        let scale = f16::from_f32(max_abs).to_f32(); // apply FP16 rounding to scale

        for &w in chunk {
            out.push(if w >= 0.0 { scale } else { -scale });
        }
    }

    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a group of GROUP_SIZE f32 values.
    fn uniform_group(v: f32) -> Vec<f32> {
        vec![v; GROUP_SIZE]
    }

    // ── quantize_group ────────────────────────────────────────────────────

    #[test]
    fn test_quantize_group_basic() {
        // Mix of positive and negative values with a known max.
        let mut weights = vec![0.0_f32; GROUP_SIZE];
        weights[0] = 2.0;
        weights[1] = -1.0;
        weights[2] = 0.5;

        let block = quantize_group(&weights);

        // Verify scale == f16(2.0)
        let scale_bits = u16::from(block[0]) | (u16::from(block[1]) << 8);
        let scale = f16::from_bits(scale_bits).to_f32();
        assert!(
            (scale - 2.0).abs() < 1e-3,
            "scale should be ~2.0, got {scale}"
        );

        // index 1 is negative → bit 1 of byte 2 should be clear
        assert_eq!(block[2] & (1 << 1), 0, "weight[1] is negative");
        // index 0 is positive → bit 0 of byte 2 should be set
        assert_ne!(block[2] & 1, 0, "weight[0] is positive");
    }

    #[test]
    fn test_quantize_group_all_positive() {
        let weights = uniform_group(3.0);
        let block = quantize_group(&weights);

        // All sign bits should be 1 (positive)
        for byte in &block[2..] {
            assert_eq!(
                *byte, 0xFFu8,
                "all sign bits should be 1 for positive weights"
            );
        }
    }

    #[test]
    fn test_quantize_group_all_negative() {
        let weights = uniform_group(-1.5);
        let block = quantize_group(&weights);

        // All sign bits should be 0 (negative)
        for byte in &block[2..] {
            assert_eq!(
                *byte, 0x00,
                "all sign bits should be 0 for negative weights"
            );
        }
    }

    // ── dequantize_block ──────────────────────────────────────────────────

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Use a simple pattern: alternating +1 and -1.
        let weights: Vec<f32> = (0..GROUP_SIZE)
            .map(|i| if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 })
            .collect();

        let block = quantize_group(&weights);
        let decoded = dequantize_block(&block);

        let scale = f16::from_f32(1.0).to_f32();
        for (i, &d) in decoded.iter().enumerate() {
            let expected = if i % 2 == 0 { scale } else { -scale };
            assert!(
                (d - expected).abs() < 1e-3,
                "decoded[{i}] = {d}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_error_analysis() {
        // Weights already at ±1 — MSE should be very small after round-tripping
        // through FP16 scale.
        let weights: Vec<f32> = (0..GROUP_SIZE * 4)
            .map(|i| if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 })
            .collect();

        let quantized = quantize_q1_0_g128(&weights).expect("quantize");
        let err = analyze_quantization_error(&weights, &quantized).expect("analyze");

        // Scale = f16(1.0) = 1.0 exactly → no error expected.
        assert!(
            err.mse < 1e-6,
            "MSE should be near zero for ±1.0 weights, got {}",
            err.mse
        );
        assert!(
            (err.bits_per_weight - 1.125).abs() < 1e-6,
            "bits_per_weight should be 1.125"
        );
    }

    // ── q1_0_g128_size_bytes ──────────────────────────────────────────────

    #[test]
    fn test_q1_0_g128_size_bytes() {
        assert_eq!(q1_0_g128_size_bytes(0), 0);
        assert_eq!(q1_0_g128_size_bytes(128), BLOCK_BYTES);
        assert_eq!(q1_0_g128_size_bytes(256), 2 * BLOCK_BYTES);
        // Partial group: 129 elements → 2 blocks
        assert_eq!(q1_0_g128_size_bytes(129), 2 * BLOCK_BYTES);
    }

    // ── compute_weight_stats ──────────────────────────────────────────────

    #[test]
    fn test_weight_stats_basic() {
        // [-1, 0, 1] → mean=0, min=-1, max=1
        let weights = vec![-1.0_f32, 0.0, 1.0];
        let stats = compute_weight_stats(&weights);
        assert_eq!(stats.num_weights, 3);
        assert!((stats.min - (-1.0)).abs() < 1e-6);
        assert!((stats.max - 1.0).abs() < 1e-6);
        assert!(stats.mean.abs() < 1e-6);
    }

    #[test]
    fn test_weight_stats_sparsity() {
        // 50% of weights are near zero
        let weights: Vec<f32> = (0..100)
            .map(|i| if i < 50 { 0.005_f32 } else { 1.0_f32 })
            .collect();
        let stats = compute_weight_stats(&weights);
        assert!(
            (stats.sparsity - 0.5).abs() < 1e-6,
            "sparsity should be 0.5, got {}",
            stats.sparsity
        );
    }

    // ── analyze_quantization_error ────────────────────────────────────────

    #[test]
    fn test_analyze_quantization_error() {
        // Weights with varying magnitude — just verify the API works and
        // the reported bits_per_weight is correct.
        let weights: Vec<f32> = (0..GROUP_SIZE * 2)
            .map(|i| (i as f32) * 0.1 - 6.4)
            .collect();
        let quantized = quantize_q1_0_g128(&weights).expect("quantize");
        let err = analyze_quantization_error(&weights, &quantized).expect("analyze");

        assert!(err.mse >= 0.0, "MSE must be non-negative");
        assert!(err.max_abs_error >= 0.0);
        assert!((err.bits_per_weight - 1.125).abs() < 1e-6);
    }

    // ── round_to_q1_0 ─────────────────────────────────────────────────────

    #[test]
    fn test_round_to_q1_0() {
        let weights: Vec<f32> = vec![2.0, -2.0, 1.0, -1.0];
        // Pad to GROUP_SIZE (round_to_q1_0 uses partial groups)
        let rounded = round_to_q1_0(&weights);
        assert_eq!(rounded.len(), weights.len());

        // Scale = f16(2.0) ≈ 2.0 → positive weights → +scale, negative → -scale
        let scale = f16::from_f32(2.0).to_f32();
        assert!((rounded[0] - scale).abs() < 1e-3, "positive weight");
        assert!((rounded[1] - (-scale)).abs() < 1e-3, "negative weight");
    }

    // ── Error paths ───────────────────────────────────────────────────────

    #[test]
    fn test_quantize_wrong_length_returns_error() {
        let weights = vec![1.0_f32; 100]; // 100 is not a multiple of 128
        let result = quantize_q1_0_g128(&weights);
        assert!(
            matches!(result, Err(QuantizeError::NotAligned { got: 100, .. })),
            "expected NotAligned error"
        );
    }

    #[test]
    fn test_quantize_zero_group_handled() {
        // A group of all zeros — scale will be 0, all dequantized values will be 0.
        // The function should NOT return an error; ZeroGroup is informational only.
        let weights = vec![0.0_f32; GROUP_SIZE];
        let result = quantize_q1_0_g128(&weights);
        assert!(result.is_ok(), "all-zero group should not error");

        let bytes = result.expect("quantize");
        let decoded = dequantize_q1_0_g128(&bytes).expect("dequantize");
        for v in &decoded {
            assert_eq!(*v, 0.0, "dequantized zero group should all be zero");
        }
    }

    // ── dequantize error path ─────────────────────────────────────────────

    #[test]
    fn test_dequantize_wrong_length_returns_error() {
        let data = vec![0u8; 17]; // 17 is not a multiple of 18
        let result = dequantize_q1_0_g128(&data);
        assert!(
            matches!(result, Err(QuantizeError::InvalidBlockData { got: 17, .. })),
            "expected InvalidBlockData error"
        );
    }

    // ── empty slice edge case ─────────────────────────────────────────────

    #[test]
    fn test_compute_weight_stats_empty() {
        let stats = compute_weight_stats(&[]);
        assert_eq!(stats.num_weights, 0);
        assert_eq!(stats.sparsity, 0.0);
    }
}

// ─── Scale rule (CQ-03) ───────────────────────────────────────────────────────

#[cfg(test)]
mod scale_rule_tests {
    use super::*;
    use oxibonsai_core::quant_prism::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64};
    use oxibonsai_core::quant_ternary::{BlockTQ2_0, BlockTQ2_0_g128};

    /// A deterministic pseudo-Gaussian weight pattern — the shape a real
    /// dense f32 layer has, with a few outliers.
    fn dense(n: usize, seed: u32) -> Vec<f32> {
        // Deterministic Gaussian via Box-Muller over a 64-bit LCG, plus one
        // 6-sigma outlier per 128-element group. That is the shape of a real
        // transformer weight matrix, and the outlier is what makes absmax
        // pathological: it sets the group scale for 127 ordinary weights.
        const SIGMA: f32 = 0.3;
        let mut state = 0x2545_F491_4F6C_DD1D_u64 ^ (u64::from(seed) << 32 | u64::from(seed));
        let mut next_unit = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Top 24 bits → (0, 1); never exactly 0, so `ln` is finite.
            ((state >> 40) as f32 + 0.5) / 16_777_216.0
        };
        (0..n)
            .map(|i| {
                if i % 128 == 13 {
                    let sign = if (i / 128) % 2 == 0 { 1.0 } else { -1.0 };
                    return 6.0 * SIGMA * sign;
                }
                let u1: f32 = next_unit();
                let u2: f32 = next_unit();
                SIGMA * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect()
    }

    /// An already-ternary group: values drawn from exactly `{-a, 0, +a}`.
    fn ternary(n: usize, a: f32) -> Vec<f32> {
        (0..n)
            .map(|i| match i % 3 {
                0 => a,
                1 => -a,
                _ => 0.0,
            })
            .collect()
    }

    fn mse(original: &[f32], reconstructed: &[f32]) -> f32 {
        let n = original.len().min(reconstructed.len());
        if n == 0 {
            return 0.0;
        }
        original
            .iter()
            .zip(reconstructed)
            .take(n)
            .map(|(a, b)| {
                let d = f64::from(*a - *b);
                d * d
            })
            .sum::<f64>() as f32
            / n as f32
    }

    // ── The detector, which makes AbsMean safe to select globally ─────────

    #[test]
    fn ternary_detector_is_exact() {
        assert!(is_ternary_group(&ternary(128, 0.25)));
        assert!(is_ternary_group(&[0.0; 128]), "all-zero counts as ternary");
        assert!(is_ternary_group(&[1.0, -1.0, 0.0, 1.0]));
        // One value strictly between 0 and amax breaks it.
        assert!(!is_ternary_group(&[1.0, -1.0, 0.5, 0.0]));
        assert!(!is_ternary_group(&dense(128, 3)));
    }

    #[test]
    fn absmean_is_bit_identical_to_absmax_on_ternary_input() {
        // The lossless-case property. Every shipped Bonsai GGUF, the MLX
        // affine 2-bit pack and the ONNX MatMulNBits path all dequantize to
        // exactly this value set, so re-encoding them must not perturb a
        // single byte — `models/*.gguf` re-quantization parity depends on it.
        for a in [0.25_f32, 1.0, 0.0078125, 7.5] {
            let input = ternary(256, a);
            for tensor_type in [
                TensorType::TQ2_0_g128,
                TensorType::PQ2_0,
                TensorType::PTQ1_0,
                TensorType::Q1_0G128,
            ] {
                let max = encode_quantized_tensor(&input, 256, tensor_type, ScaleRule::AbsMax)
                    .expect("absmax encode");
                let mean = encode_quantized_tensor(&input, 256, tensor_type, ScaleRule::AbsMean)
                    .expect("absmean encode");
                assert_eq!(
                    max, mean,
                    "{tensor_type:?} with a={a}: AbsMean must fall back to absmax for an \
                     already-ternary group, byte for byte"
                );
            }
        }
    }

    #[test]
    fn q1_0_stays_bit_identical_on_ternary_input() {
        let input = ternary(512, 0.5);
        assert_eq!(
            quantize_q1_0_g128(&input).expect("absmax"),
            quantize_q1_0_g128_with(&input, ScaleRule::AbsMean).expect("absmean"),
        );
    }

    // ── The actual defect: absmax on a continuous source ──────────────────

    /// The gate CQ-03 names: on a real f32 layer, round-trip MSE must drop
    /// by at least 3×.
    ///
    /// The finding measured the symptom on `q1_0` ("inflates every layer
    /// output ~3.5×"), and that is where the effect is dramatic: a sign-only
    /// quantizer with an absmax scale reconstructs *every* weight at the
    /// group's largest magnitude, so a single 6σ outlier rescales 127
    /// ordinary weights. Measured here: ≈ 48× MSE reduction and a
    /// reconstructed L1 norm that drops from ≈ 7× the source to ≈ 1×.
    #[test]
    fn absmean_cuts_q1_0_round_trip_error_by_far_more_than_3x() {
        let input = dense(4096, 11);
        let source_l1: f64 = input.iter().map(|w| f64::from(w.abs())).sum();

        let absmax = quantize_q1_0_g128(&input).expect("absmax");
        let absmean = quantize_q1_0_g128_with(&input, ScaleRule::AbsMean).expect("absmean");
        assert_ne!(absmax, absmean, "a dense source must take the AbsMean path");

        let absmax_out = dequantize_q1_0_g128(&absmax).expect("dq absmax");
        let absmean_out = dequantize_q1_0_g128(&absmean).expect("dq absmean");
        let absmax_mse = mse(&input, &absmax_out);
        let absmean_mse = mse(&input, &absmean_out);
        assert!(
            absmean_mse * 3.0 <= absmax_mse,
            "round-trip MSE must drop by >= 3x: absmax {absmax_mse:.6}, absmean {absmean_mse:.6}"
        );

        let l1 = |v: &[f32]| v.iter().map(|w| f64::from(w.abs())).sum::<f64>() / source_l1;
        assert!(
            l1(&absmax_out) > 3.0,
            "absmax is the reported inflation: ratio {:.2}",
            l1(&absmax_out)
        );
        assert!(
            (0.8..1.25).contains(&l1(&absmean_out)),
            "absmean must preserve the signal magnitude: ratio {:.2}",
            l1(&absmean_out)
        );
    }

    /// The same defect on a *ternary* format, where it presents as the
    /// opposite symptom.
    ///
    /// With three levels and a threshold of `0.5·amax`, a 6σ outlier pushes
    /// the threshold above every ordinary weight, so absmax encodes almost
    /// the whole group as **zero** — the reconstructed L1 norm collapses to
    /// a few percent of the source instead of inflating. The improvement
    /// factor is smaller than the sign quantizer's because a ternary code
    /// cannot beat roughly 0.4·E[w²] on a Gaussian however good the scale is;
    /// measured here ≈ 2.7×.
    #[test]
    fn absmean_fixes_ternary_scale_collapse() {
        let input = dense(4096, 11);
        let source_l1: f64 = input.iter().map(|w| f64::from(w.abs())).sum();
        let mut recon = vec![0.0_f32; input.len()];

        let mut measure = |rule: ScaleRule| -> (f32, f64) {
            let bytes =
                encode_quantized_tensor(&input, 4096, TensorType::PQ2_0, rule).expect("encode");
            BlockPQ2_0::dequant(
                BlockPQ2_0::slice_from_bytes(&bytes).expect("blocks"),
                &mut recon,
            )
            .expect("dequant");
            (
                mse(&input, &recon),
                recon.iter().map(|w| f64::from(w.abs())).sum::<f64>() / source_l1,
            )
        };

        let (absmax_mse, absmax_l1) = measure(ScaleRule::AbsMax);
        let (absmean_mse, absmean_l1) = measure(ScaleRule::AbsMean);

        assert!(
            absmax_l1 < 0.5,
            "absmax on ternary collapses the signal; L1 ratio was {absmax_l1:.2}"
        );
        assert!(
            (0.6..1.4).contains(&absmean_l1),
            "absmean must restore the signal magnitude; L1 ratio was {absmean_l1:.2}"
        );
        assert!(
            absmean_mse * 2.0 <= absmax_mse,
            "ternary round-trip MSE must drop by >= 2x: absmax {absmax_mse:.6}, \
             absmean {absmean_mse:.6}"
        );
    }

    #[test]
    fn ternarize_borrows_for_absmax_and_never_touches_ternary_groups() {
        let dense_input = dense(256, 2);
        assert!(matches!(
            ternarize_groups(&dense_input, 128, ScaleRule::AbsMax),
            Cow::Borrowed(_)
        ));
        let ternary_input = ternary(256, 0.75);
        let rewritten = ternarize_groups(&ternary_input, 128, ScaleRule::AbsMean);
        assert_eq!(
            rewritten.as_ref(),
            ternary_input.as_slice(),
            "an already-ternary group must come back unchanged"
        );
    }

    // ── Block writers (CQ-05) ─────────────────────────────────────────────

    #[test]
    fn every_ternary_writer_round_trips_to_its_quantization_step() {
        let input = dense(1024, 13);
        let cases: [(TensorType, usize); 4] = [
            (TensorType::PQ2_0, 34),
            (TensorType::PTQ1_0, 28),
            (TensorType::TQ2_0_g128, 34),
            (TensorType::Q2_0G64, 18),
        ];
        for (tensor_type, block_bytes) in cases {
            let group = tensor_type.block_size();
            let bytes = encode_quantized_tensor(&input, 1024, tensor_type, ScaleRule::AbsMean)
                .expect("encode");
            assert_eq!(
                bytes.len(),
                input.len() / group * block_bytes,
                "{tensor_type:?} byte count"
            );

            let mut out = vec![0.0_f32; input.len()];
            match tensor_type {
                TensorType::PQ2_0 => BlockPQ2_0::dequant(
                    BlockPQ2_0::slice_from_bytes(&bytes).expect("blocks"),
                    &mut out,
                ),
                TensorType::PTQ1_0 => BlockPTQ1_0::dequant(
                    BlockPTQ1_0::slice_from_bytes(&bytes).expect("blocks"),
                    &mut out,
                ),
                TensorType::TQ2_0_g128 => BlockTQ2_0_g128::dequant(
                    BlockTQ2_0_g128::slice_from_bytes(&bytes).expect("blocks"),
                    &mut out,
                ),
                TensorType::Q2_0G64 => BlockQ2_0G64::dequant(
                    BlockQ2_0G64::slice_from_bytes(&bytes).expect("blocks"),
                    &mut out,
                ),
                other => panic!("unexpected {other:?}"),
            }
            .expect("dequant");

            // Every reconstructed value must be one of the three levels the
            // format represents.
            for chunk_index in 0..(input.len() / group) {
                let lo = chunk_index * group;
                let hi = lo + group;
                let scale = out[lo..hi].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                for value in &out[lo..hi] {
                    assert!(
                        *value == 0.0 || *value == scale || *value == -scale,
                        "{tensor_type:?}: value {value} is not a ternary level"
                    );
                }
            }

            // "Reproduces the source to the format's quantization step" in
            // the only form that is actually checkable for a 3-level code:
            // re-encoding the reconstruction must reproduce the exact same
            // bytes. A per-element bound cannot be asserted because a 6σ
            // outlier is by construction further than one step from any
            // ternary level.
            let reencoded = encode_quantized_tensor(&out, 1024, tensor_type, ScaleRule::AbsMean)
                .expect("re-encode");
            assert_eq!(
                reencoded, bytes,
                "{tensor_type:?}: quantize -> dequant -> quantize must be idempotent"
            );
        }
    }

    #[test]
    fn pq2_0_is_scale_first_and_ptq1_0_is_scale_last() {
        // The one property no length check can catch: both 128-wide ternary
        // blocks are the same size as the legacy layout, so a byte-swapped
        // block passes every size validation.
        let input = ternary(128, 0.5);

        let pq2 = encode_quantized_tensor(&input, 128, TensorType::PQ2_0, ScaleRule::AbsMax)
            .expect("pq2 encode");
        assert_eq!(pq2.len(), 34);
        let pq2_scale = f16::from_bits(u16::from_le_bytes([pq2[0], pq2[1]])).to_f32();
        assert_eq!(pq2_scale, 0.5, "PQ2_0 stores `d` in bytes 0..2");

        let ptq1 = encode_quantized_tensor(&input, 128, TensorType::PTQ1_0, ScaleRule::AbsMax)
            .expect("ptq1 encode");
        assert_eq!(ptq1.len(), 28);
        let ptq1_scale = f16::from_bits(u16::from_le_bytes([ptq1[26], ptq1[27]])).to_f32();
        assert_eq!(ptq1_scale, 0.5, "PTQ1_0 stores `d` in bytes 26..28");

        let legacy =
            encode_quantized_tensor(&input, 128, TensorType::TQ2_0_g128, ScaleRule::AbsMax)
                .expect("legacy encode");
        assert_eq!(legacy.len(), 34);
        let legacy_scale = f16::from_bits(u16::from_le_bytes([legacy[32], legacy[33]])).to_f32();
        assert_eq!(legacy_scale, 0.5, "TQ2_0_g128 stores `d` in bytes 32..34");
        assert_ne!(
            &pq2[..],
            &legacy[..],
            "PQ2_0 and the legacy layout must not be interchangeable"
        );
    }

    #[test]
    fn ptq1_0_transcodes_losslessly_to_pq2_0() {
        // The two Bonsai 2 formats carry identical ternary data, so a
        // PTQ1_0 export and a PQ2_0 export of the same weights must
        // dequantize to exactly the same floats.
        let input = dense(512, 17);
        let ptq1 = encode_quantized_tensor(&input, 512, TensorType::PTQ1_0, ScaleRule::AbsMean)
            .expect("ptq1");
        let pq2 = encode_quantized_tensor(&input, 512, TensorType::PQ2_0, ScaleRule::AbsMean)
            .expect("pq2");

        let mut from_ptq1 = vec![0.0_f32; 512];
        let mut from_pq2 = vec![0.0_f32; 512];
        BlockPTQ1_0::dequant(
            BlockPTQ1_0::slice_from_bytes(&ptq1).expect("ptq1 blocks"),
            &mut from_ptq1,
        )
        .expect("ptq1 dequant");
        BlockPQ2_0::dequant(
            BlockPQ2_0::slice_from_bytes(&pq2).expect("pq2 blocks"),
            &mut from_pq2,
        )
        .expect("pq2 dequant");
        assert_eq!(from_ptq1, from_pq2);
    }

    // ── Row alignment (CQ-14) ─────────────────────────────────────────────

    #[test]
    fn a_row_that_is_not_a_block_multiple_is_refused_not_padded() {
        let input = dense(260, 3);
        let err = encode_quantized_tensor(&input, 130, TensorType::PQ2_0, ScaleRule::AbsMax)
            .expect_err("130 is not a multiple of 128");
        assert!(matches!(
            err,
            QuantizeError::RowNotBlockAligned {
                ne0: 130,
                group: 128,
                ..
            }
        ));
        assert!(err.to_string().contains("row"));
        assert!(!row_is_block_aligned(130, TensorType::PQ2_0));
        assert!(row_is_block_aligned(128, TensorType::PQ2_0));
        assert!(
            row_is_block_aligned(7, TensorType::F32),
            "F32 has no blocks"
        );
    }

    // ── Error plumbing (CQ-20) ────────────────────────────────────────────

    #[test]
    fn errors_carry_the_tensor_name() {
        let err =
            encode_quantized_tensor(&dense(260, 1), 130, TensorType::PQ2_0, ScaleRule::AbsMax)
                .expect_err("must fail")
                .with_tensor("blk.7.ffn_up.weight");
        assert_eq!(err.tensor(), Some("blk.7.ffn_up.weight"));
        assert!(err.to_string().contains("blk.7.ffn_up.weight"));

        let anonymous = QuantizeError::NotAligned {
            tensor: String::new(),
            got: 3,
            group: 128,
        };
        assert_eq!(anonymous.tensor(), None);
        assert!(anonymous.to_string().contains("<unnamed>"));
    }

    // ── 0b11 screen ───────────────────────────────────

    #[test]
    fn reserved_code_in_a_ternary_tensor_is_rejected() {
        let mut bytes = encode_quantized_tensor(
            &ternary(128, 1.0),
            128,
            TensorType::TQ2_0_g128,
            ScaleRule::AbsMax,
        )
        .expect("encode");
        validate_ternary_codes("t", &bytes, TensorType::TQ2_0_g128).expect("clean tensor passes");

        // Force lane 0 of block 0 to the reserved 0b11 code.
        bytes[0] |= 0b11;
        let err = validate_ternary_codes("blk.0.attn_q.weight", &bytes, TensorType::TQ2_0_g128)
            .expect_err("0b11 must be rejected");
        assert!(err.to_string().contains("blk.0.attn_q.weight"));
        assert!(err.to_string().contains("0b11"));

        // PQ2_0 legitimately defines 0b11 (+2), so it is exempt.
        validate_ternary_codes("t", &bytes, TensorType::PQ2_0).expect("PQ2_0 is exempt");
    }

    // ── TQ2_0 (llama.cpp 256-wide, qs-first) 0b11 screen ──────────────────
    //
    // Regression coverage for the qs-first/d-last layout: a d-first reading
    // of this block both false-rejects a clean tensor (the scale's own bytes
    // can contain a `0b11` bit pair) and false-negatives a real `0b11`
    // planted in `qs[0]`/`qs[1]`, since those fall outside a `2..66` range.
    #[test]
    fn tq2_0_reserved_code_screen_uses_the_qs_first_layout() {
        // `d = 1.0` (f16 bits 0x3C00, LE bytes 00,3C) is exactly the value
        // that trips the old, wrong `2..66` range: byte 0x3C decodes to two
        // `0b11` lanes, so a d-first reading rejects this perfectly clean
        // tensor.
        let blocks = BlockTQ2_0::quantize(&ternary(256, 1.0)).expect("quantize");
        // SAFETY: `BlockTQ2_0` is repr(C), 66 bytes (asserted in oxibonsai-core).
        let mut bytes = unsafe { blocks_as_bytes(&blocks) };
        validate_ternary_codes("t", &bytes, TensorType::TQ2_0).expect(
            "a clean TQ2_0 tensor must pass — the old d-first range misread the FP16 scale's \
             own bytes as reserved codes",
        );

        // Plant a real reserved 0b11 code in qs[0] (bits 0..2 of byte 0).
        // The old, wrong `2..66` range starts two bytes into `qs` and so
        // never sees this.
        bytes[0] |= 0b11;
        let err = validate_ternary_codes("blk.0.attn_qkv.weight", &bytes, TensorType::TQ2_0)
            .expect_err("a 0b11 code planted in qs[0] must be rejected");
        assert!(err.to_string().contains("blk.0.attn_qkv.weight"));
        assert!(err.to_string().contains("0b11"));
    }

    // ── Source dtype (CQ-M2 / CQ-M3) ──────────────────────────────────────

    #[test]
    fn source_dtypes_round_trip_including_bf16() {
        let values = [1.0_f32, -2.5, 0.0, 0.5];
        for dtype in [SourceDtype::F32, SourceDtype::F16, SourceDtype::BF16] {
            let bytes = encode_source_bytes(dtype, &values);
            assert_eq!(bytes.len(), values.len() * dtype.element_bytes());
            let back = dequant_source_bytes(dtype, &bytes).expect("decode");
            assert_eq!(back.len(), values.len());
            for (a, b) in values.iter().zip(&back) {
                assert!((a - b).abs() <= 0.02, "{dtype:?}: {a} vs {b}");
            }
        }
        assert_eq!(SourceDtype::BF16.name(), "BF16");
    }

    #[test]
    fn a_truncated_source_buffer_is_an_error_not_a_silent_truncation() {
        let err = dequant_source_bytes(SourceDtype::F32, &[0u8; 6]).expect_err("6 % 4 != 0");
        assert!(matches!(
            err,
            QuantizeError::InvalidBlockData { got: 6, .. }
        ));
    }
}

// ── K-quant real-GGUF write -> parse -> dequant round trips ────────────────
//
// Split into a sibling file (`quantize/kquant_roundtrip_tests.rs`) to keep
// this file under the 2000-line policy limit; see that file's module doc
// for the full rationale. `use super::*;` there
// resolves against this module (`quantize`) exactly as it did when this was
// an inline `mod kquant_writer_roundtrip { .. }` block.
#[cfg(test)]
mod kquant_roundtrip_tests;
