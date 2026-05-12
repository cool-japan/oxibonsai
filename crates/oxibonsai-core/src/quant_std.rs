//! Standard GGUF quantization block types: Q4_0 (4-bit) and Q8_0 (8-bit).
//!
//! These are the most common quantization formats in distributed GGUF model files,
//! accounting for roughly 80% of publicly released models.
//!
//! - **Q4_0** (GGML type 2): 32 weights per block, 18 bytes total.
//!   Block scale `d: f16` + 16 bytes of packed 4-bit nibbles (2 per byte).
//!   Dequant: `w[j] = d × (nibble[j] − 8)`.
//!
//! - **Q8_0** (GGML type 8): 32 weights per block, 34 bytes total.
//!   Block scale `d: f16` + 32 bytes of `i8` weights.
//!   Dequant: `w[j] = d × qs[j]`.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of weights per Q4_0 block.
pub const QK_Q4_0: usize = 32;

/// Number of bytes per Q4_0 block (2-byte f16 scale + 16 bytes of 4-bit pairs).
pub const BLOCK_Q4_0_BYTES: usize = 18;

/// Number of weights per Q8_0 block.
pub const QK_Q8_0: usize = 32;

/// Number of bytes per Q8_0 block (2-byte f16 scale + 32 bytes of i8 weights).
pub const BLOCK_Q8_0_BYTES: usize = 34;

// ---------------------------------------------------------------------------
// BlockQ4_0
// ---------------------------------------------------------------------------

/// Q4_0 block: 32 weights quantized to 4 bits each with a shared FP16 scale.
///
/// Layout (18 bytes):
/// - `d`: FP16 block scale.
/// - `qs`: 16 bytes — 32 × 4-bit quantized weights, 2 per byte.
///   Even index `j` → low nibble `qs[j/2] & 0x0F`; odd → high nibble `qs[j/2] >> 4`.
///
/// Dequant: `w[j] = d × (nibble[j] as f32 − 8.0)` — symmetric around zero.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    /// Block scale (FP16).
    pub d: f16,
    /// 32 × 4-bit quantized weights, 2 per byte (low nibble = even index).
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == BLOCK_Q4_0_BYTES);

impl BlockQ4_0 {
    /// Dequantize a slice of Q4_0 blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_Q4_0`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_Q4_0;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_0 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_Q4_0;
            for j in 0..QK_Q4_0 {
                let nibble = if j % 2 == 0 {
                    (block.qs[j / 2] & 0x0F) as f32
                } else {
                    ((block.qs[j / 2] >> 4) & 0x0F) as f32
                };
                output[base + j] = d * (nibble - 8.0);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q4_0 blocks.
    ///
    /// Input length must be a multiple of `QK_Q4_0` (32).
    ///
    /// Scale = `max(|input|) / 7.0`; nibble = `clamp(round(x / scale + 8), 0, 15)`.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_Q4_0 != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_0 quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_Q4_0
                ),
            });
        }
        let num_blocks = input.len() / QK_Q4_0;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_Q4_0;
            let chunk = &input[base..base + QK_Q4_0];

            let max_abs = chunk
                .iter()
                .filter(|v| !v.is_nan())
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);

            if max_abs == 0.0 {
                blocks.push(BlockQ4_0 {
                    d: f16::ZERO,
                    qs: [0x88u8; 16], // nibble 8 → w = d*(8-8) = 0
                });
                continue;
            }

            // Scale maps range [-max_abs, +max_abs] to [-7, +7]; centroid at nibble 8.
            let scale = max_abs / 7.0;
            let d = f16::from_f32(scale);
            // Use the f16-rounded scale for quantization consistency.
            let scale_actual = d.to_f32();
            let inv_scale = if scale_actual == 0.0 {
                0.0
            } else {
                1.0 / scale_actual
            };

            let mut qs = [0u8; 16];
            for j in 0..QK_Q4_0 {
                let v = chunk[j];
                // Shift by 8 to make unsigned 0..15; clamp to 4-bit range.
                let nibble = (v * inv_scale + 8.5).clamp(0.0, 15.0) as u8;
                if j % 2 == 0 {
                    qs[j / 2] = nibble & 0x0F;
                } else {
                    qs[j / 2] |= (nibble & 0x0F) << 4;
                }
            }

            blocks.push(BlockQ4_0 { d, qs });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ4_0`.
    ///
    /// # Errors
    ///
    /// Returns [`BonsaiError::KQuantError`] if `data.len()` is not a multiple
    /// of `BLOCK_Q4_0_BYTES` or the pointer is not 2-byte aligned (required
    /// for the embedded `f16` field).
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q4_0_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_0 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q4_0_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q4_0 slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q4_0_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and
        // alignment checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Dequantize this single block into a 32-element f32 buffer.
    ///
    /// Used by the GEMV kernel to avoid heap allocation on the hot path.
    #[inline]
    pub fn dequant_to_buf(&self, buf: &mut [f32; 32]) {
        let d = self.d.to_f32();
        for (j, out) in buf.iter_mut().enumerate() {
            let nibble = if j % 2 == 0 {
                (self.qs[j / 2] & 0x0F) as f32
            } else {
                ((self.qs[j / 2] >> 4) & 0x0F) as f32
            };
            *out = d * (nibble - 8.0);
        }
    }
}

// ---------------------------------------------------------------------------
// BlockQ8_0
// ---------------------------------------------------------------------------

/// Q8_0 block: 32 weights quantized to 8-bit signed integers with a shared FP16 scale.
///
/// Layout (34 bytes):
/// - `d`: FP16 block scale.
/// - `qs`: 32 bytes of `i8` quantized weights.
///
/// Dequant: `w[j] = d × qs[j] as f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    /// Block scale (FP16).
    pub d: f16,
    /// 32 × int8 quantized weights.
    pub qs: [i8; 32],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == BLOCK_Q8_0_BYTES);

impl BlockQ8_0 {
    /// Dequantize a slice of Q8_0 blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_Q8_0`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_Q8_0;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q8_0 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_Q8_0;
            for (j, &q) in block.qs.iter().enumerate() {
                output[base + j] = d * (q as f32);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q8_0 blocks.
    ///
    /// Input length must be a multiple of `QK_Q8_0` (32).
    ///
    /// Scale = `max(|x|) / 127`; `qs[j] = clamp(round(x / scale), -127, 127)`.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_Q8_0 != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q8_0 quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_Q8_0
                ),
            });
        }
        let num_blocks = input.len() / QK_Q8_0;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_Q8_0;
            let chunk = &input[base..base + QK_Q8_0];

            let max_abs = chunk
                .iter()
                .filter(|v| !v.is_nan())
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);

            if max_abs == 0.0 {
                blocks.push(BlockQ8_0 {
                    d: f16::ZERO,
                    qs: [0i8; 32],
                });
                continue;
            }

            let scale = max_abs / 127.0;
            let d = f16::from_f32(scale);
            let scale_actual = d.to_f32();
            let inv_scale = if scale_actual == 0.0 {
                0.0
            } else {
                1.0 / scale_actual
            };

            let mut qs = [0i8; 32];
            for (j, &v) in chunk.iter().enumerate() {
                let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                qs[j] = q;
            }

            blocks.push(BlockQ8_0 { d, qs });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ8_0`.
    ///
    /// # Errors
    ///
    /// Returns [`BonsaiError::KQuantError`] if `data.len()` is not a multiple
    /// of `BLOCK_Q8_0_BYTES` or the pointer is not 2-byte aligned (required
    /// for the embedded `f16` field).
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q8_0_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q8_0 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q8_0_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q8_0 slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q8_0_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and
        // alignment checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Dequantize this single block into a 32-element f32 buffer.
    ///
    /// Used by the GEMV kernel to avoid heap allocation on the hot path.
    #[inline]
    pub fn dequant_to_buf(&self, buf: &mut [f32; 32]) {
        let d = self.d.to_f32();
        for (j, &q) in self.qs.iter().enumerate() {
            buf[j] = d * (q as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Block size assertions ──────────────────────────────────────────────

    #[test]
    fn q4_0_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), BLOCK_Q4_0_BYTES);
        assert_eq!(BLOCK_Q4_0_BYTES, 18);
    }

    #[test]
    fn q8_0_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), BLOCK_Q8_0_BYTES);
        assert_eq!(BLOCK_Q8_0_BYTES, 34);
    }

    #[test]
    fn qk_constants_correct() {
        assert_eq!(QK_Q4_0, 32);
        assert_eq!(QK_Q8_0, 32);
    }

    // ── Q4_0 tests ────────────────────────────────────────────────────────

    #[test]
    fn q4_0_dequant_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 7.5).collect();
        let blocks = BlockQ4_0::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 1);
        let mut output = vec![0.0f32; 32];
        BlockQ4_0::dequant(&blocks, &mut output).unwrap();
        let max_err: f32 = values
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Q4_0 has 4-bit precision; relative error should be small
        assert!(
            max_err < 1.5,
            "Q4_0 round-trip max error: {max_err} (values range ±7.5)"
        );
    }

    #[test]
    fn q4_0_all_zeros() {
        let values = vec![0.0f32; 32];
        let blocks = BlockQ4_0::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockQ4_0::dequant(&blocks, &mut output).unwrap();
        assert!(
            output.iter().all(|&x| x == 0.0),
            "all-zero input should give all-zero output"
        );
    }

    #[test]
    fn q4_0_nibble_extremes() {
        // A block with max_abs = 7.0 should produce scale = 1.0.
        // Positive 7.0 → nibble 15; negative 7.0 → nibble 1 (7*(1-8)=-7).
        let mut values = vec![0.0f32; 32];
        values[0] = 7.0;
        values[1] = -7.0;
        let blocks = BlockQ4_0::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockQ4_0::dequant(&blocks, &mut output).unwrap();
        assert!(
            (output[0] - 7.0).abs() < 1.1,
            "max weight round-trip: got {}",
            output[0]
        );
        assert!(
            (output[1] + 7.0).abs() < 1.1,
            "min weight round-trip: got {}",
            output[1]
        );
    }

    #[test]
    fn q4_0_slice_from_bytes_valid() {
        let block = BlockQ4_0 {
            d: f16::from_f32(1.0),
            qs: [0x88u8; 16],
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts((&block as *const BlockQ4_0).cast::<u8>(), BLOCK_Q4_0_BYTES)
        };
        let result = BlockQ4_0::slice_from_bytes(bytes).expect("aligned slice should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].d, f16::from_f32(1.0));
    }

    #[test]
    fn q4_0_slice_from_bytes_bad_len() {
        let data = vec![0u8; 17]; // not a multiple of 18
        assert!(
            BlockQ4_0::slice_from_bytes(&data).is_err(),
            "bad length should be rejected"
        );
    }

    #[test]
    fn q4_0_block_count_validation() {
        let values = vec![1.0f32; 96]; // 3 blocks
        let blocks = BlockQ4_0::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn q4_0_quantize_wrong_len() {
        assert!(
            BlockQ4_0::quantize(&[1.0f32; 15]).is_err(),
            "non-multiple of 32 should be rejected"
        );
    }

    #[test]
    fn q4_0_dequant_too_small_buffer() {
        let blocks = BlockQ4_0::quantize(&[1.0f32; 32]).unwrap();
        let mut out = vec![0.0f32; 10];
        assert!(
            BlockQ4_0::dequant(&blocks, &mut out).is_err(),
            "output too small should be rejected"
        );
    }

    #[test]
    fn q4_0_dequant_to_buf_matches_dequant() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
        let blocks = BlockQ4_0::quantize(&values).unwrap();
        let mut full_out = vec![0.0f32; 32];
        BlockQ4_0::dequant(&blocks, &mut full_out).unwrap();
        let mut buf = [0.0f32; 32];
        blocks[0].dequant_to_buf(&mut buf);
        for (a, b) in full_out.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-6, "dequant_to_buf must match dequant");
        }
    }

    #[test]
    fn q4_0_multi_block_no_nan() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 8.0).collect();
        let blocks = BlockQ4_0::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 2);
        let mut out = vec![0.0f32; 64];
        BlockQ4_0::dequant(&blocks, &mut out).unwrap();
        assert!(out.iter().all(|x| !x.is_nan()), "no NaN in output");
    }

    #[test]
    fn q4_0_scale_nonzero_for_nonzero_input() {
        let values = vec![1.0f32; 32];
        let blocks = BlockQ4_0::quantize(&values).unwrap();
        assert_ne!(blocks[0].d, f16::ZERO, "scale must be non-zero");
    }

    // ── Q8_0 tests ────────────────────────────────────────────────────────

    #[test]
    fn q8_0_dequant_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let blocks = BlockQ8_0::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockQ8_0::dequant(&blocks, &mut output).unwrap();
        let max_err: f32 = values
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.05,
            "Q8_0 round-trip max error: {max_err} (8-bit should be very accurate)"
        );
    }

    #[test]
    fn q8_0_all_zeros() {
        let values = vec![0.0f32; 32];
        let blocks = BlockQ8_0::quantize(&values).unwrap();
        let mut output = vec![0.0f32; 32];
        BlockQ8_0::dequant(&blocks, &mut output).unwrap();
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn q8_0_int8_extremes() {
        // Values ±127.0 * scale should clamp to qs = ±127
        let mut values = vec![0.0f32; 32];
        values[0] = 127.0;
        values[1] = -127.0;
        let blocks = BlockQ8_0::quantize(&values).unwrap();
        // With max_abs = 127.0, scale = 1.0, qs[0]=127, qs[1]=-127
        let scale = blocks[0].d.to_f32();
        assert!((scale - 1.0).abs() < 0.01, "scale should be ~1.0: {scale}");
        assert_eq!(blocks[0].qs[0], 127, "max quantized to 127");
        assert_eq!(blocks[0].qs[1], -127, "min quantized to -127");
    }

    #[test]
    fn q8_0_slice_alignment() {
        let block = BlockQ8_0 {
            d: f16::from_f32(2.0),
            qs: [0i8; 32],
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts((&block as *const BlockQ8_0).cast::<u8>(), BLOCK_Q8_0_BYTES)
        };
        let result = BlockQ8_0::slice_from_bytes(bytes).expect("aligned slice should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].d, f16::from_f32(2.0));
    }

    #[test]
    fn q8_0_quantize_scale() {
        let mut values = vec![0.0f32; 32];
        values[5] = 63.5; // max_abs = 63.5 → scale ≈ 63.5/127 ≈ 0.5
        let blocks = BlockQ8_0::quantize(&values).unwrap();
        let scale = blocks[0].d.to_f32();
        assert!(
            (scale - 0.5).abs() < 0.02,
            "scale should be ~0.5 for max=63.5, got {scale}"
        );
    }

    #[test]
    fn q8_0_slice_bad_len() {
        let data = vec![0u8; 35]; // not a multiple of 34
        assert!(BlockQ8_0::slice_from_bytes(&data).is_err());
    }

    #[test]
    fn q8_0_quantize_wrong_len() {
        assert!(BlockQ8_0::quantize(&[1.0f32; 17]).is_err());
    }

    #[test]
    fn q8_0_dequant_too_small_buffer() {
        let blocks = BlockQ8_0::quantize(&[0.0f32; 32]).unwrap();
        let mut out = vec![0.0f32; 5];
        assert!(BlockQ8_0::dequant(&blocks, &mut out).is_err());
    }

    #[test]
    fn q8_0_dequant_to_buf_matches_dequant() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32) * 3.0 - 48.0).collect();
        let blocks = BlockQ8_0::quantize(&values).unwrap();
        let mut full_out = vec![0.0f32; 32];
        BlockQ8_0::dequant(&blocks, &mut full_out).unwrap();
        let mut buf = [0.0f32; 32];
        blocks[0].dequant_to_buf(&mut buf);
        for (a, b) in full_out.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-6, "dequant_to_buf must match dequant");
        }
    }

    #[test]
    fn q8_0_positive_negative_mix() {
        let values: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
            .collect();
        let blocks = BlockQ8_0::quantize(&values).unwrap();
        let mut out = vec![0.0f32; 32];
        BlockQ8_0::dequant(&blocks, &mut out).unwrap();
        // Signs must be preserved
        for i in (2..32).step_by(2) {
            assert!(
                out[i] >= 0.0,
                "even index should be non-negative: {}",
                out[i]
            );
        }
        for i in (1..32).step_by(2) {
            assert!(
                out[i] <= 0.0,
                "odd index should be non-positive: {}",
                out[i]
            );
        }
    }

    #[test]
    fn q8_0_block_count_correct() {
        let values = vec![1.0f32; 96]; // 3 blocks
        let blocks = BlockQ8_0::quantize(&values).unwrap();
        assert_eq!(blocks.len(), 3);
    }
}
