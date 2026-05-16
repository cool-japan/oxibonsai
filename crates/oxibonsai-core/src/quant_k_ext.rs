//! K-quant block types for Q5_K and Q6_K quantization formats.
//!
//! These follow the GGML K-quant specification:
//! - **Q5_K**: 5-bit quantization with 6-bit scales, super-block of 256 weights (176 bytes)
//! - **Q6_K**: 6-bit quantization with int8 scales, super-block of 256 weights (210 bytes)
//!
//! Q5_K is an asymmetric 5-bit format using the same 12-byte scale/min packing as Q4_K.
//! Q6_K is a symmetric centered 6-bit format with per-sub-block int8 scales.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};
use crate::quant_k::QK_K;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of bytes per Q5_K block (2 + 2 + 12 + 32 + 128 = 176).
pub const BLOCK_Q5K_BYTES: usize = 176;

/// Number of bytes per Q6_K block (128 + 64 + 16 + 2 = 210).
pub const BLOCK_Q6K_BYTES: usize = 210;

// ---------------------------------------------------------------------------
// Scale-packing helpers (same layout as Q4_K in quant_k.rs)
// ---------------------------------------------------------------------------

/// Decode the 8 six-bit scale values and 8 six-bit min values from the
/// 12-byte packed `scales` array in a Q5_K block.
///
/// This is the same layout used by Q4_K in this codebase:
/// - bytes 0..3:  low 4 bits of scale[0..7] (two per byte, 4 bits each)
/// - bytes 4..7:  low 4 bits of min[0..7]   (two per byte, 4 bits each)
/// - bytes 8..9:  upper 2 bits of scale[0..7], packed 4 values per byte
/// - bytes 10..11:upper 2 bits of min[0..7],   packed 4 values per byte
fn decode_q5k_scales(scales_raw: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];

    // Low 4 bits of scales (2 per byte in bytes 0..3)
    for i in 0..4 {
        sc[2 * i] = scales_raw[i] & 0x0F;
        sc[2 * i + 1] = (scales_raw[i] >> 4) & 0x0F;
    }

    // Low 4 bits of mins (2 per byte in bytes 4..7)
    for i in 0..4 {
        mn[2 * i] = scales_raw[4 + i] & 0x0F;
        mn[2 * i + 1] = (scales_raw[4 + i] >> 4) & 0x0F;
    }

    // Upper 2 bits of scales from bytes 8..9
    for i in 0..4 {
        sc[i] |= ((scales_raw[8] >> (2 * i)) & 0x03) << 4;
        sc[4 + i] |= ((scales_raw[9] >> (2 * i)) & 0x03) << 4;
    }

    // Upper 2 bits of mins from bytes 10..11
    for i in 0..4 {
        mn[i] |= ((scales_raw[10] >> (2 * i)) & 0x03) << 4;
        mn[4 + i] |= ((scales_raw[11] >> (2 * i)) & 0x03) << 4;
    }

    (sc, mn)
}

/// Encode 8 six-bit scale values and 8 six-bit min values into the 12-byte
/// packed format used by Q5_K (same layout as Q4_K).
fn encode_q5k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];

    // Low 4 bits of scales into bytes 0..3 (two per byte)
    for i in 0..4 {
        out[i] = (sc[2 * i] & 0x0F) | ((sc[2 * i + 1] & 0x0F) << 4);
    }
    // Low 4 bits of mins into bytes 4..7 (two per byte)
    for i in 0..4 {
        out[4 + i] = (mn[2 * i] & 0x0F) | ((mn[2 * i + 1] & 0x0F) << 4);
    }
    // Upper 2 bits of scales into bytes 8..9
    for i in 0..4 {
        out[8] |= ((sc[i] >> 4) & 0x03) << (2 * i);
        out[9] |= ((sc[4 + i] >> 4) & 0x03) << (2 * i);
    }
    // Upper 2 bits of mins into bytes 10..11
    for i in 0..4 {
        out[10] |= ((mn[i] >> 4) & 0x03) << (2 * i);
        out[11] |= ((mn[4 + i] >> 4) & 0x03) << (2 * i);
    }

    out
}

// ---------------------------------------------------------------------------
// BlockQ5K
// ---------------------------------------------------------------------------

/// Q5_K super-block: 256 weights quantized to 5 bits each.
///
/// Layout (176 bytes):
/// - `d`:      FP16 super-block scale.
/// - `dmin`:   FP16 super-block minimum.
/// - `scales`: 12 bytes — packed 6-bit scale/min values for 8 sub-blocks of 32 weights.
///   Same encoding as Q4_K: low 4 bits in bytes 0..7, upper 2 bits in bytes 8..11.
/// - `qh`:     32 bytes — the high (5th) bit for each of the 256 weights (1 bit per weight).
/// - `qs`:     128 bytes — the low 4 bits for each of the 256 weights (2 per byte).
///
/// Dequant: `w[i] = d * sub_scale * q5[i] - dmin * sub_min`
/// where `q5[i] = (qs nibble) | (high bit from qh << 4)`, range [0..31].
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ5K {
    /// Super-block scale (FP16).
    pub d: f16,
    /// Super-block minimum (FP16).
    pub dmin: f16,
    /// Packed 6-bit scales for 8 sub-blocks (same layout as Q4_K).
    pub scales: [u8; 12],
    /// High bit (bit 4) for each of the 256 weights, packed 8 per byte.
    pub qh: [u8; 32],
    /// Low 4 bits for each of the 256 weights, packed 2 per byte.
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ5K>() == BLOCK_Q5K_BYTES);

impl BlockQ5K {
    /// Dequantize a slice of Q5_K blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_K` (256 per block).
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q5_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let dmin_val = block.dmin.to_f32();
            let base = block_idx * QK_K;

            let (sc, mn) = decode_q5k_scales(&block.scales);

            // 8 sub-blocks of 32 weights each
            for sub in 0..8usize {
                let sub_scale = d * (sc[sub] as f32);
                let sub_min = dmin_val * (mn[sub] as f32);
                let sub_offset = sub * 32;

                for j in 0..32usize {
                    let global_idx = sub_offset + j;

                    // Low 4 bits (nibble from qs)
                    let nibble = if global_idx % 2 == 0 {
                        block.qs[global_idx / 2] & 0x0F
                    } else {
                        (block.qs[global_idx / 2] >> 4) & 0x0F
                    };

                    // High bit (bit 4) from qh: byte = global_idx / 8, bit = global_idx % 8
                    let high_bit = (block.qh[global_idx / 8] >> (global_idx % 8)) & 1;

                    // 5-bit quantized value in [0..31]
                    let q5 = nibble | (high_bit << 4);

                    output[base + global_idx] = sub_scale * (q5 as f32) - sub_min;
                }
            }
        }
        Ok(())
    }

    /// Dequantize a single row's worth of Q5_K blocks into a pre-allocated buffer.
    ///
    /// `buf` will be extended by `blocks_for_row.len() * 256` elements.
    /// Clear or pre-size the buffer before calling if a clean start is needed.
    pub fn dequant_row_to_buf(blocks_for_row: &[Self], buf: &mut Vec<f32>) {
        let start = buf.len();
        let n = blocks_for_row.len() * QK_K;
        buf.resize(start + n, 0.0f32);
        // Buffer is always correctly sized here; result is infallible.
        let _ = Self::dequant(blocks_for_row, &mut buf[start..]);
    }

    /// Quantize f32 input into Q5_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    ///
    /// This uses an asymmetric per-sub-block quantization:
    /// - Each sub-block of 32 weights is mapped to [0..31] with an offset (min).
    /// - The 12-byte scales array stores 6-bit sub-block scale and min factors.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q5_K quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_K
                ),
            });
        }

        let num_blocks = input.len() / QK_K;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_K;
            let chunk = &input[base..base + QK_K];

            // 8 sub-blocks of 32 weights each
            let mut sub_scales = [0.0f32; 8];
            let mut sub_mins = [0.0f32; 8];

            for sub in 0..8usize {
                let sub_offset = sub * 32;
                let sub_chunk = &chunk[sub_offset..sub_offset + 32];

                let mut smin = f32::MAX;
                let mut smax = f32::MIN;
                for &v in sub_chunk {
                    if v < smin {
                        smin = v;
                    }
                    if v > smax {
                        smax = v;
                    }
                }

                // Shift minimum to zero, then scale the positive range to [0..31]
                sub_mins[sub] = if smin < 0.0 { -smin } else { 0.0 };
                let range = smax + sub_mins[sub];
                sub_scales[sub] = if range > 0.0 { range / 31.0 } else { 0.0 };
            }

            let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
            let max_min = sub_mins.iter().copied().fold(0.0f32, f32::max);

            // 6-bit sub-block factors in [0..63]
            let d = if max_scale > 0.0 {
                max_scale / 63.0
            } else {
                0.0
            };
            let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };

            let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
            let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];

            for sub in 0..8usize {
                sc[sub] = (sub_scales[sub] * inv_d + 0.5).min(63.0) as u8;
                mn[sub] = (sub_mins[sub] * inv_dmin + 0.5).min(63.0) as u8;
            }

            let scales = encode_q5k_scales(&sc, &mn);

            // Quantize weights to 5 bits (4 low bits in qs, 1 high bit in qh)
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];

            for sub in 0..8usize {
                let sub_offset = sub * 32;
                let sc_f = d * (sc[sub] as f32);
                let mn_f = dmin * (mn[sub] as f32);
                let inv_sc = if sc_f > 0.0 { 1.0 / sc_f } else { 0.0 };

                for j in 0..32usize {
                    let global_idx = sub_offset + j;
                    let val = chunk[global_idx] + mn_f;
                    let q5 = (val * inv_sc + 0.5).clamp(0.0, 31.0) as u8;

                    // Low 4 bits into qs (2 per byte)
                    let nibble = q5 & 0x0F;
                    let byte_idx = global_idx / 2;
                    if global_idx % 2 == 0 {
                        qs[byte_idx] |= nibble;
                    } else {
                        qs[byte_idx] |= nibble << 4;
                    }

                    // High bit (bit 4) into qh (1 bit per bit position in byte)
                    let high_bit = (q5 >> 4) & 1;
                    let qh_byte = global_idx / 8;
                    let qh_bit = global_idx % 8;
                    qh[qh_byte] |= high_bit << qh_bit;
                }
            }

            blocks.push(BlockQ5K {
                d: f16::from_f32(d),
                dmin: f16::from_f32(dmin),
                scales,
                qh,
                qs,
            });
        }

        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ5K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q5K_BYTES` (176)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q5K_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q5_K slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q5K_BYTES
                ),
            });
        }
        // Empty slice: no alignment check needed for zero-length data.
        if data.is_empty() {
            return Ok(&[]);
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q5_K slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q5K_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time size assert;
        // length is a multiple of BLOCK_Q5K_BYTES; pointer alignment verified above;
        // lifetime is tied to the input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

// ---------------------------------------------------------------------------
// BlockQ6K
// ---------------------------------------------------------------------------

/// Q6_K super-block: 256 weights quantized to 6 bits each.
///
/// Layout (210 bytes):
/// - `ql`:     128 bytes — low 4 bits for each of the 256 weights (2 per byte).
/// - `qh`:     64 bytes  — high 2 bits for each of the 256 weights (4 × 2-bit per byte).
/// - `scales`: 16 bytes  — int8 scale for each of 16 sub-blocks of 16 weights.
/// - `d`:      FP16 super-block scale.
///
/// Dequant: `w[i] = d * scales[i/16] * (q6[i] - 32)`
/// where `q6[i] = (ql nibble) | (high 2 bits from qh << 4)`, range [0..63],
/// centered by subtracting 32 to give a symmetric range [-32..31].
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    /// Low 4 bits for each of the 256 weights, packed 2 per byte.
    pub ql: [u8; 128],
    /// High 2 bits for each of the 256 weights, packed 4 per byte (2 bits each).
    pub qh: [u8; 64],
    /// Per-sub-block int8 scale for each of 16 sub-blocks of 16 weights.
    pub scales: [i8; 16],
    /// Super-block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockQ6K>() == BLOCK_Q6K_BYTES);

impl BlockQ6K {
    /// Dequantize a slice of Q6_K blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_K` (256 per block).
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q6_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_K;

            // 256 weights: 16 sub-blocks of 16 weights each
            for i in 0..QK_K {
                // Low 4 bits (nibble from ql)
                let nibble = if i % 2 == 0 {
                    block.ql[i / 2] & 0x0F
                } else {
                    (block.ql[i / 2] >> 4) & 0x0F
                };

                // High 2 bits from qh: each byte holds 4 × 2-bit values
                // byte index = i / 4, 2-bit lane = i % 4 (each lane is 2 bits wide)
                let high_2bits = (block.qh[i / 4] >> (2 * (i % 4))) & 0x03;

                // 6-bit quantized value in [0..63]
                let q6 = nibble | (high_2bits << 4);

                // Center around zero: [0..63] → [-32..31]
                let q_centered = q6 as i32 - 32;

                // Sub-block index: 16 sub-blocks of 16 weights
                let sub = i / 16;

                output[base + i] = d * (block.scales[sub] as f32) * (q_centered as f32);
            }
        }
        Ok(())
    }

    /// Dequantize a single row's worth of Q6_K blocks into a pre-allocated buffer.
    ///
    /// `buf` will be extended by `blocks_for_row.len() * 256` elements.
    pub fn dequant_row_to_buf(blocks_for_row: &[Self], buf: &mut Vec<f32>) {
        let start = buf.len();
        let n = blocks_for_row.len() * QK_K;
        buf.resize(start + n, 0.0f32);
        let _ = Self::dequant(blocks_for_row, &mut buf[start..]);
    }

    /// Quantize f32 input into Q6_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    ///
    /// This uses a symmetric per-sub-block quantization:
    /// - 16 sub-blocks of 16 weights each.
    /// - Each sub-block is mapped to [-32..31] via an int8 scale.
    /// - The super-block scale `d` normalizes the int8 sub-block scales.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q6_K quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_K
                ),
            });
        }

        let num_blocks = input.len() / QK_K;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_K;
            let chunk = &input[base..base + QK_K];

            // Compute per-sub-block max absolute values (16 sub-blocks of 16)
            let mut sub_max_abs = [0.0f32; 16];
            for (sub, slot) in sub_max_abs.iter_mut().enumerate() {
                let sub_offset = sub * 16;
                let sub_chunk = &chunk[sub_offset..sub_offset + 16];
                let max_abs = sub_chunk.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
                *slot = max_abs;
            }

            // The super-block scale d is chosen so that d * 127 ≈ max(sub_max_abs) / 31.
            // int8 sub-scales encode (sub_max_abs / 31) / d.
            let overall_max = sub_max_abs.iter().copied().fold(0.0f32, f32::max);
            let d = if overall_max > 0.0 {
                overall_max / (31.0 * 127.0)
            } else {
                0.0
            };

            let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

            // Quantize per-sub-block scales to int8 in [-127..127]
            let mut scales = [0i8; 16];
            for (scale_out, &max_abs) in scales.iter_mut().zip(sub_max_abs.iter()) {
                let sc_f = max_abs * inv_d / 31.0;
                *scale_out = sc_f.round().clamp(-127.0, 127.0) as i8;
            }

            // Quantize weights to 6 bits centered at 32
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];

            for (i, &w) in chunk.iter().enumerate() {
                let sub = i / 16;
                let scale_f = d * (scales[sub] as f32);
                let inv_scale = if scale_f.abs() > 1e-9 {
                    1.0 / scale_f
                } else {
                    0.0
                };

                // Centered code: add 32 to shift [-32..31] → [0..63]
                let q_centered = (w * inv_scale).round() as i32;
                let q6 = (q_centered + 32).clamp(0, 63) as u8;

                // Low 4 bits into ql (2 per byte)
                let nibble = q6 & 0x0F;
                let byte_idx = i / 2;
                if i % 2 == 0 {
                    ql[byte_idx] |= nibble;
                } else {
                    ql[byte_idx] |= nibble << 4;
                }

                // High 2 bits into qh (4 × 2-bit per byte)
                let high_2bits = (q6 >> 4) & 0x03;
                let qh_byte = i / 4;
                let qh_shift = 2 * (i % 4);
                qh[qh_byte] |= high_2bits << qh_shift;
            }

            blocks.push(BlockQ6K {
                ql,
                qh,
                scales,
                d: f16::from_f32(d),
            });
        }

        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ6K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q6K_BYTES` (210)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q6K_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q6_K slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q6K_BYTES
                ),
            });
        }
        // Empty slice: no alignment check needed for zero-length data.
        if data.is_empty() {
            return Ok(&[]);
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q6_K slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q6K_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time size assert;
        // length is a multiple of BLOCK_Q6K_BYTES; pointer alignment verified above;
        // lifetime is tied to the input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BlockQ5K tests ---

    #[test]
    fn q5k_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ5K>(), BLOCK_Q5K_BYTES);
        assert_eq!(BLOCK_Q5K_BYTES, 176);
    }

    #[test]
    fn q5k_dequant_all_zeros_input() {
        let blocks = BlockQ5K::quantize(&vec![0.0f32; 256]).expect("quantize should succeed");
        let mut out = vec![0.0f32; 256];
        BlockQ5K::dequant(&blocks, &mut out).expect("dequant should succeed");
        for &v in &out {
            assert!(
                v.abs() < 1e-4,
                "all-zero input should dequant to near-zero, got {v}"
            );
        }
    }

    #[test]
    fn q5k_dequant_output_too_small_errors() {
        let blocks = BlockQ5K::quantize(&vec![1.0f32; 256]).expect("quantize ok");
        let mut out = vec![0.0f32; 100]; // too small
        assert!(
            BlockQ5K::dequant(&blocks, &mut out).is_err(),
            "should error on too-small output buffer"
        );
    }

    #[test]
    fn q5k_quantize_non_multiple_errors() {
        assert!(
            BlockQ5K::quantize(&vec![1.0f32; 100]).is_err(),
            "should error when input len is not a multiple of 256"
        );
    }

    #[test]
    fn q5k_dequant_round_trip_accuracy() {
        // Pattern that exercises the full 5-bit space across sub-blocks.
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let blocks = BlockQ5K::quantize(&input).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ5K::dequant(&blocks, &mut out).expect("dequant ok");

        let max_err = input
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.2,
            "Q5_K round-trip max abs error {max_err} exceeds threshold 0.2"
        );
    }

    #[test]
    fn q5k_dequant_round_trip_uniform_positive() {
        // Uniform positive values — quantize then dequant should be very accurate.
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let blocks = BlockQ5K::quantize(&input).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ5K::dequant(&blocks, &mut out).expect("dequant ok");

        let max_err = input
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.2,
            "Q5_K uniform positive round-trip error {max_err} > 0.2"
        );
    }

    #[test]
    fn q5k_scale_encode_decode_round_trip() {
        // Verify that encode then decode gives back the same 6-bit values.
        let sc = [1u8, 2, 3, 4, 5, 63, 32, 0];
        let mn = [10u8, 20, 30, 40, 50, 60, 15, 7];
        let encoded = encode_q5k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q5k_scales(&encoded);
        assert_eq!(sc, sc2, "scales should survive encode-decode round trip");
        assert_eq!(mn, mn2, "mins should survive encode-decode round trip");
    }

    #[test]
    fn q5k_scale_encode_decode_all_zeros() {
        let sc = [0u8; 8];
        let mn = [0u8; 8];
        let encoded = encode_q5k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q5k_scales(&encoded);
        assert_eq!(sc, sc2);
        assert_eq!(mn, mn2);
    }

    #[test]
    fn q5k_scale_encode_decode_max_values() {
        let sc = [63u8; 8];
        let mn = [63u8; 8];
        let encoded = encode_q5k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q5k_scales(&encoded);
        assert_eq!(sc, sc2, "max scale should survive round trip");
        assert_eq!(mn, mn2, "max min should survive round trip");
    }

    #[test]
    fn q5k_slice_from_bytes_bad_length() {
        let data = vec![0u8; 100]; // not a multiple of 176
        assert!(
            BlockQ5K::slice_from_bytes(&data).is_err(),
            "should error on non-multiple length"
        );
    }

    #[test]
    fn q5k_slice_from_bytes_empty() {
        let data = vec![0u8; 0];
        let result = BlockQ5K::slice_from_bytes(&data).expect("empty slice should succeed");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn q5k_multiple_blocks_dequant() {
        // Two blocks, verify output length is 512 and round-trip is accurate.
        let input: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.005).collect();
        let blocks = BlockQ5K::quantize(&input).expect("quantize ok");
        assert_eq!(blocks.len(), 2);
        let mut out = vec![0.0f32; 512];
        BlockQ5K::dequant(&blocks, &mut out).expect("dequant ok");

        let max_err = input
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.2,
            "Q5_K two-block round-trip max error {max_err} > 0.2"
        );
    }

    #[test]
    fn q5k_dequant_row_to_buf_works() {
        let input = vec![0.5f32; 256];
        let blocks = BlockQ5K::quantize(&input).expect("quantize ok");
        let mut buf = Vec::new();
        BlockQ5K::dequant_row_to_buf(&blocks, &mut buf);
        assert_eq!(buf.len(), 256, "buf should contain 256 elements");
        // All values should be near 0.5 (quantization noise)
        for &v in &buf {
            assert!((v - 0.5).abs() < 0.1, "expected ~0.5, got {v}");
        }
    }

    // --- BlockQ6K tests ---

    #[test]
    fn q6k_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ6K>(), BLOCK_Q6K_BYTES);
        assert_eq!(BLOCK_Q6K_BYTES, 210);
    }

    #[test]
    fn q6k_dequant_all_zeros_input() {
        let blocks = BlockQ6K::quantize(&vec![0.0f32; 256]).expect("quantize should succeed");
        let mut out = vec![0.0f32; 256];
        BlockQ6K::dequant(&blocks, &mut out).expect("dequant should succeed");
        for &v in &out {
            assert!(
                v.abs() < 1e-4,
                "all-zero input should dequant to near-zero, got {v}"
            );
        }
    }

    #[test]
    fn q6k_dequant_centering() {
        // Build a block where all ql=0x00 and qh=0x00 → q6=0 → q_centered=-32.
        // scales[i]=1, d=1.0 → weight = 1.0 * 1 * (-32) = -32.0
        let block = BlockQ6K {
            ql: [0u8; 128],
            qh: [0u8; 64],
            scales: [1i8; 16],
            d: f16::from_f32(1.0),
        };
        let mut out = vec![0.0f32; 256];
        BlockQ6K::dequant(&[block], &mut out).expect("dequant ok");
        for &v in &out {
            assert!(
                (v + 32.0).abs() < 1e-3,
                "q6=0 should give weight=-32*scale, got {v}"
            );
        }
    }

    #[test]
    fn q6k_dequant_extreme_values() {
        // ql=0xFF (all nibbles=0xF=15) and qh=0xFF (all 2-bit lanes=0b11=3)
        // → q6 = 15 | (3 << 4) = 15 | 48 = 63 → q_centered = 31
        // scales=1, d=1.0 → weight = 31.0
        let block = BlockQ6K {
            ql: [0xFF; 128],
            qh: [0xFF; 64],
            scales: [1i8; 16],
            d: f16::from_f32(1.0),
        };
        let mut out = vec![0.0f32; 256];
        BlockQ6K::dequant(&[block], &mut out).expect("dequant ok");
        for &v in &out {
            assert!(
                (v - 31.0).abs() < 1e-3,
                "q6=63 should give weight=+31*scale, got {v}"
            );
        }
    }

    #[test]
    fn q6k_dequant_round_trip_accuracy() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let blocks = BlockQ6K::quantize(&input).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ6K::dequant(&blocks, &mut out).expect("dequant ok");

        let max_err = input
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.15,
            "Q6_K round-trip max abs error {max_err} exceeds threshold 0.15"
        );
    }

    #[test]
    fn q6k_quantize_non_multiple_errors() {
        assert!(
            BlockQ6K::quantize(&vec![1.0f32; 100]).is_err(),
            "should error when input len is not a multiple of 256"
        );
    }

    #[test]
    fn q6k_dequant_output_too_small_errors() {
        let blocks = BlockQ6K::quantize(&vec![1.0f32; 256]).expect("quantize ok");
        let mut out = vec![0.0f32; 100];
        assert!(
            BlockQ6K::dequant(&blocks, &mut out).is_err(),
            "should error on too-small output buffer"
        );
    }

    #[test]
    fn q6k_slice_from_bytes_bad_length() {
        let data = vec![0u8; 100]; // not a multiple of 210
        assert!(
            BlockQ6K::slice_from_bytes(&data).is_err(),
            "should error on non-multiple length"
        );
    }

    #[test]
    fn q6k_slice_from_bytes_empty() {
        let data = vec![0u8; 0];
        let result = BlockQ6K::slice_from_bytes(&data).expect("empty slice should succeed");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn q6k_quantize_scale_estimation() {
        // Constant 1.0 input should round-trip near 1.0.
        let input = vec![1.0f32; 256];
        let blocks = BlockQ6K::quantize(&input).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ6K::dequant(&blocks, &mut out).expect("dequant ok");
        for &v in &out {
            assert!(
                (v - 1.0).abs() < 0.1,
                "constant-1.0 round trip should stay near 1.0, got {v}"
            );
        }
    }

    #[test]
    fn q6k_multiple_blocks_round_trip() {
        let input: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.005).collect();
        let blocks = BlockQ6K::quantize(&input).expect("quantize ok");
        assert_eq!(blocks.len(), 2);
        let mut out = vec![0.0f32; 512];
        BlockQ6K::dequant(&blocks, &mut out).expect("dequant ok");

        let max_err = input
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.15,
            "Q6_K two-block round-trip max error {max_err} > 0.15"
        );
    }
}
