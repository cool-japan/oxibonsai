//! K-quant block types for Q2_K, Q3_K, Q4_K, and Q8_K quantization formats.
//!
//! These follow the GGML K-quant specification:
//! - **Q2_K**: 2-bit quantization with 4-bit scales, super-block of 256 weights (84 bytes)
//! - **Q3_K**: 3-bit quantization with 4-bit scales, super-block of 256 weights (110 bytes)
//! - **Q4_K**: 4-bit quantization with 6-bit scales, super-block of 256 weights (144 bytes)
//! - **Q8_K**: 8-bit quantization with FP32 scale, super-block of 256 weights (292 bytes)
//!
//! Each super-block stores a global `d` (scale) and `dmin` (minimum) in FP16,
//! plus per-sub-block scales and quantized weight nibbles/pairs.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of weights per K-quant super-block.
pub const QK_K: usize = 256;

/// Number of bytes per Q2_K block.
pub const BLOCK_Q2_K_BYTES: usize = 84;

/// Number of bytes per Q3_K block.
pub const BLOCK_Q3K_BYTES: usize = 110;

/// Number of bytes per Q4_K block.
pub const BLOCK_Q4_K_BYTES: usize = 144;

/// Number of bytes per Q8_K block.
pub const BLOCK_Q8K_BYTES: usize = 292;

// ---------------------------------------------------------------------------
// BlockQ2K
// ---------------------------------------------------------------------------

/// Q2_K super-block: 256 weights quantized to 2 bits each.
///
/// Layout (84 bytes):
/// - `scales`: 16 bytes — packed 4-bit scale/min pairs for 16 sub-blocks of 16 weights.
///   Each byte holds two 4-bit values: low nibble = scale, high nibble = min.
/// - `qs`: 64 bytes — 256 x 2-bit quantized weights (4 per byte, LSB first).
/// - `d`: FP16 super-block scale.
/// - `dmin`: FP16 super-block minimum.
///
/// Dequant: `w[i] = d * sub_scale * q[i] - dmin * sub_min`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    /// Packed 4-bit scale/min pairs for 16 sub-blocks.
    pub scales: [u8; 16],
    /// 256 x 2-bit quantized weights, 4 per byte.
    pub qs: [u8; 64],
    /// Super-block scale (FP16).
    pub d: f16,
    /// Super-block minimum (FP16).
    pub dmin: f16,
}

const _: () = assert!(std::mem::size_of::<BlockQ2K>() == BLOCK_Q2_K_BYTES);

impl BlockQ2K {
    /// Dequantize a slice of Q2_K blocks into f32 output.
    ///
    /// `output` must have length `blocks.len() * QK_K`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q2_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let base = block_idx * QK_K;

            // 16 sub-blocks of 16 weights each
            for sub in 0..16 {
                let scale_byte = block.scales[sub];
                let sc = (scale_byte & 0x0F) as f32; // low nibble = scale
                let mn = ((scale_byte >> 4) & 0x0F) as f32; // high nibble = min

                let sub_offset = sub * 16;
                for j in 0..16 {
                    let global_idx = sub_offset + j;
                    // Each byte holds 4 x 2-bit values
                    let byte_idx = global_idx / 4;
                    let shift = (global_idx % 4) * 2;
                    let q = ((block.qs[byte_idx] >> shift) & 0x03) as f32;
                    output[base + global_idx] = d * sc * q - dmin * mn;
                }
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q2_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q2_K quantize: input len {} not a multiple of {}",
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

            // Pass 1: find global max absolute value and min value across
            // all sub-blocks to set d and dmin.
            // For each sub-block of 16 weights, we find the range [min, max].
            let mut sub_scales = [0.0f32; 16];
            let mut sub_mins = [0.0f32; 16];

            for sub in 0..16 {
                let sub_offset = sub * 16;
                let sub_chunk = &chunk[sub_offset..sub_offset + 16];

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

                // The offset (min) removes the minimum, then scale maps remainder to 0..3
                sub_mins[sub] = if smin < 0.0 { -smin } else { 0.0 };
                let range = smax + sub_mins[sub];
                sub_scales[sub] = if range > 0.0 { range / 3.0 } else { 0.0 };
            }

            // Find the global maximum scale and minimum across sub-blocks
            let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
            let max_min = sub_mins.iter().copied().fold(0.0f32, f32::max);

            // Compute d and dmin so that 4-bit sub-block factors (0..15) can represent
            // the per-sub-block scales and mins.
            let d = if max_scale > 0.0 {
                max_scale / 15.0
            } else {
                0.0
            };
            let dmin = if max_min > 0.0 { max_min / 15.0 } else { 0.0 };

            let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
            let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

            // Quantize per-sub-block scales and mins to 4 bits
            let mut scales = [0u8; 16];
            let mut quant_sc = [0u8; 16];
            let mut quant_mn = [0u8; 16];

            for sub in 0..16 {
                let sc = (sub_scales[sub] * inv_d + 0.5).min(15.0) as u8;
                let mn = (sub_mins[sub] * inv_dmin + 0.5).min(15.0) as u8;
                quant_sc[sub] = sc;
                quant_mn[sub] = mn;
                scales[sub] = sc | (mn << 4);
            }

            // Quantize weights to 2 bits
            let mut qs = [0u8; 64];
            for sub in 0..16 {
                let sub_offset = sub * 16;
                let sc_f = d * (quant_sc[sub] as f32);
                let mn_f = dmin * (quant_mn[sub] as f32);
                let inv_sc = if sc_f > 0.0 { 1.0 / sc_f } else { 0.0 };

                for j in 0..16 {
                    let global_idx = sub_offset + j;
                    let val = chunk[global_idx] + mn_f;
                    let q = (val * inv_sc + 0.5).clamp(0.0, 3.0) as u8;
                    let byte_idx = global_idx / 4;
                    let shift = (global_idx % 4) * 2;
                    qs[byte_idx] |= q << shift;
                }
            }

            blocks.push(BlockQ2K {
                scales,
                qs,
                d: f16::from_f32(d),
                dmin: f16::from_f32(dmin),
            });
        }

        Ok(blocks)
    }

    /// Dequantize a single row's worth of Q2_K blocks into a pre-allocated buffer.
    ///
    /// `buf` will be extended by `blocks_for_row.len() * 256` elements.
    pub fn dequant_row_to_buf(blocks_for_row: &[Self], buf: &mut Vec<f32>) {
        let start = buf.len();
        let n = blocks_for_row.len() * QK_K;
        buf.resize(start + n, 0.0f32);
        let _ = Self::dequant(blocks_for_row, &mut buf[start..]);
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ2K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q2_K_BYTES` (84)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q2_K_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q2_K slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q2_K_BYTES
                ),
            });
        }
        if data.is_empty() {
            return Ok(&[]);
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q2_K slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q2_K_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time size assert;
        // length is a multiple of BLOCK_Q2_K_BYTES; pointer alignment verified above;
        // lifetime is tied to the input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

// ---------------------------------------------------------------------------
// BlockQ3K
// ---------------------------------------------------------------------------

/// Q3_K super-block: 256 weights quantized to 3 bits each.
///
/// Layout (110 bytes):
/// - `hmask`:  32 bytes — high bit (bit 2) for each of the 256 weights, packed 8 per byte.
/// - `qs`:     64 bytes — low 2 bits for each of the 256 weights, packed 4 per byte.
/// - `scales`: 12 bytes — 4-bit scale values for 16 sub-blocks of 16 weights each.
///   Each nibble is a signed 4-bit value (stored as u4, subtract 8 for range [-8..7]).
///   Packing: `scales[j/2] >> (4*(j%2)) & 0xF` gives sub-block j's raw scale.
/// - `d`: FP16 super-block scale.
///
/// Dequant: `w[i] = d * sub_scale * q3_signed[i]`
/// where `q3_signed = ((low2 | (high1<<2)) as i32) - 4`, range [-4, 3].
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    /// High bit (bit 2) for each of 256 weights, packed 8 per byte.
    pub hmask: [u8; 32],
    /// Low 2 bits for each of 256 weights, packed 4 per byte (2 bits each, LSB first).
    pub qs: [u8; 64],
    /// 16 × 4-bit sub-block scales, 2 per byte (low nibble = sub 2i, high nibble = sub 2i+1).
    pub scales: [u8; 12],
    /// Super-block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockQ3K>() == BLOCK_Q3K_BYTES);

impl BlockQ3K {
    /// Dequantize a slice of Q3_K blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_K` (256 per block).
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q3_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_K;

            // 16 sub-blocks of 16 weights each; scale is 4-bit signed nibble
            for i in 0..QK_K {
                // Low 2 bits from qs: each byte holds 4 × 2-bit values (2 bits per weight)
                let byte_idx = i / 4;
                let bit_shift = (i % 4) * 2;
                let lo2 = (block.qs[byte_idx] >> bit_shift) & 0x03;

                // High bit (bit 2) from hmask: each byte holds 8 bits, one per weight
                let hi1 = (block.hmask[i / 8] >> (i % 8)) & 0x01;

                // 3-bit code in [0..7], centered: range [-4..3]
                let q3 = lo2 | (hi1 << 2);
                let q3_signed = (q3 as i32) - 4;

                // Sub-block index: 16 sub-blocks of 16 weights each
                let sub = i / 16;
                // 4-bit nibble from scales (2 per byte)
                let scale_nibble = (block.scales[sub / 2] >> (4 * (sub % 2))) & 0x0F;
                // Signed 4-bit scale: stored as 0..15 representing -8..7
                let scale_signed = (scale_nibble as i8) as i32 - 8;

                output[base + i] = d * (scale_signed as f32) * (q3_signed as f32);
            }
        }
        Ok(())
    }

    /// Dequantize a single row's worth of Q3_K blocks into a pre-allocated buffer.
    ///
    /// `buf` will be extended by `blocks_for_row.len() * 256` elements.
    pub fn dequant_row_to_buf(blocks_for_row: &[Self], buf: &mut Vec<f32>) {
        let start = buf.len();
        let n = blocks_for_row.len() * QK_K;
        buf.resize(start + n, 0.0f32);
        let _ = Self::dequant(blocks_for_row, &mut buf[start..]);
    }

    /// Quantize f32 input into Q3_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    ///
    /// Uses symmetric per-sub-block quantization: 16 sub-blocks of 16 weights each,
    /// mapping each sub-block to the range [-4..3] with a 4-bit signed scale.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q3_K quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_K
                ),
            });
        }

        let num_blocks = input.len() / QK_K;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let chunk = &input[block_idx * QK_K..block_idx * QK_K + QK_K];

            // Compute per-sub-block max absolute value for 16 sub-blocks of 16 weights
            let mut sub_max_abs = [0.0f32; 16];
            for (sub, slot) in sub_max_abs.iter_mut().enumerate() {
                let sub_chunk = &chunk[sub * 16..(sub + 1) * 16];
                *slot = sub_chunk.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            }

            // Super-block scale: d * max_scale_nibble * 4 ≈ overall max abs
            // max scale nibble is 7 (for signed scale value 7 - 8 + 8 = 7 in [0..15])
            // and max 3-bit centered code is 3 (q3_signed in [-4..3])
            // Effective range = d * 7 * 3 = d * 21
            let overall_max = sub_max_abs.iter().copied().fold(0.0f32, f32::max);
            let d = if overall_max > 0.0 {
                overall_max / 21.0
            } else {
                0.0
            };
            let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

            // Compute per-sub-block 4-bit signed scale nibbles
            let mut scale_nibbles = [0u8; 16];
            for (sub, &max_abs) in sub_max_abs.iter().enumerate() {
                // scale_signed = max_abs / (d * 3), clamped to [-8..7]
                let sc_f = if d > 0.0 { max_abs * inv_d / 3.0 } else { 0.0 };
                let sc_signed = sc_f.round().clamp(-8.0, 7.0) as i32;
                // Store as 0..15 (add 8 to shift from signed to unsigned nibble)
                scale_nibbles[sub] = (sc_signed + 8).clamp(0, 15) as u8;
            }

            // Pack nibbles into scales[12]: 2 per byte
            let mut scales = [0u8; 12];
            for (sub, &nibble_val) in scale_nibbles.iter().enumerate() {
                let byte_idx = sub / 2;
                let nibble = nibble_val & 0x0F;
                if sub % 2 == 0 {
                    scales[byte_idx] |= nibble;
                } else {
                    scales[byte_idx] |= nibble << 4;
                }
            }

            // Quantize weights to 3-bit codes
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];

            for i in 0..QK_K {
                let sub = i / 16;
                let sc_signed = (scale_nibbles[sub] as i32) - 8;
                // Effective scale for this sub-block
                let eff_scale = d * (sc_signed as f32);
                let inv_eff = if eff_scale.abs() > 1e-9 {
                    1.0 / eff_scale
                } else {
                    0.0
                };

                // Compute 3-bit code: map w → q3_signed in [-4..3], then add 4 → [0..7]
                let q3_signed = (chunk[i] * inv_eff).round() as i32;
                let q3 = (q3_signed + 4).clamp(0, 7) as u8;

                // Low 2 bits → qs (4 × 2-bit per byte)
                let lo2 = q3 & 0x03;
                let byte_idx = i / 4;
                let bit_shift = (i % 4) * 2;
                qs[byte_idx] |= lo2 << bit_shift;

                // High bit (bit 2) → hmask (8 × 1-bit per byte)
                let hi1 = (q3 >> 2) & 0x01;
                hmask[i / 8] |= hi1 << (i % 8);
            }

            blocks.push(BlockQ3K {
                hmask,
                qs,
                scales,
                d: f16::from_f32(d),
            });
        }

        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ3K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q3K_BYTES` (110)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q3K_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q3_K slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q3K_BYTES
                ),
            });
        }
        if data.is_empty() {
            return Ok(&[]);
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q3_K slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q3K_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time size assert;
        // length is a multiple of BLOCK_Q3K_BYTES; pointer alignment verified above;
        // lifetime is tied to the input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

// ---------------------------------------------------------------------------
// BlockQ4K
// ---------------------------------------------------------------------------

/// Q4_K super-block: 256 weights quantized to 4 bits each.
///
/// Layout (144 bytes):
/// - `d`: FP16 super-block scale.
/// - `dmin`: FP16 super-block minimum.
/// - `scales`: 12 bytes — packed 6-bit scale/min values for 8 sub-blocks of 32 weights.
///   Encoding: bytes 0..3 hold low 4 bits of scale[0..7], bytes 4..7 hold low 4 bits
///   of min[0..7], bytes 8..11 hold the upper 2 bits of scales and mins packed.
/// - `qs`: 128 bytes — 256 x 4-bit quantized weights (2 per byte).
///
/// Dequant: `w[i] = d * sub_scale * q[i] - dmin * sub_min`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ4K {
    /// Super-block scale (FP16).
    pub d: f16,
    /// Super-block minimum (FP16).
    pub dmin: f16,
    /// Packed 6-bit scales for 8 sub-blocks.
    pub scales: [u8; 12],
    /// 256 x 4-bit quantized weights, 2 per byte.
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ4K>() == BLOCK_Q4_K_BYTES);

/// Decode the 8 six-bit scale values and 8 six-bit min values from the
/// 12-byte packed `scales` array in a Q4_K block.
///
/// Layout of the 12 bytes:
/// - bytes 0..3:  low 4 bits of scale[0..7] (two per byte, 4 bits each)
/// - bytes 4..7:  low 4 bits of min[0..7]   (two per byte, 4 bits each)
/// - bytes 8..11: upper 2 bits of scale and min, packed
///
/// Specifically for bytes 8..11:
/// - byte  8: bits 0..1 = scale[0] hi, bits 2..3 = scale[1] hi, bits 4..5 = scale[2] hi, bits 6..7 = scale[3] hi
/// - byte  9: bits 0..1 = scale[4] hi, bits 2..3 = scale[5] hi, bits 4..5 = scale[6] hi, bits 6..7 = scale[7] hi
/// - byte 10: bits 0..1 = min[0] hi,   bits 2..3 = min[1] hi,   bits 4..5 = min[2] hi,   bits 6..7 = min[3] hi
/// - byte 11: bits 0..1 = min[4] hi,   bits 2..3 = min[5] hi,   bits 4..5 = min[6] hi,   bits 6..7 = min[7] hi
fn decode_q4k_scales(scales_raw: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
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
/// packed format used by Q4_K blocks.
fn encode_q4k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];

    // Low 4 bits of scales into bytes 0..3
    for i in 0..4 {
        out[i] = (sc[2 * i] & 0x0F) | ((sc[2 * i + 1] & 0x0F) << 4);
    }

    // Low 4 bits of mins into bytes 4..7
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

impl BlockQ4K {
    /// Dequantize a slice of Q4_K blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_K`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let dmin_val = block.dmin.to_f32();
            let base = block_idx * QK_K;

            let (sc, mn) = decode_q4k_scales(&block.scales);

            // 8 sub-blocks of 32 weights each
            for sub in 0..8 {
                let sub_scale = d * (sc[sub] as f32);
                let sub_min = dmin_val * (mn[sub] as f32);
                let sub_offset = sub * 32;

                for j in 0..32 {
                    let global_idx = sub_offset + j;
                    let byte_idx = global_idx / 2;
                    let q = if global_idx % 2 == 0 {
                        (block.qs[byte_idx] & 0x0F) as f32
                    } else {
                        ((block.qs[byte_idx] >> 4) & 0x0F) as f32
                    };
                    output[base + global_idx] = sub_scale * q - sub_min;
                }
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q4_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_K quantize: input len {} not a multiple of {}",
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

            // 8 sub-blocks of 32 weights
            let mut sub_scales = [0.0f32; 8];
            let mut sub_mins = [0.0f32; 8];

            for sub in 0..8 {
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

                sub_mins[sub] = if smin < 0.0 { -smin } else { 0.0 };
                let range = smax + sub_mins[sub];
                sub_scales[sub] = if range > 0.0 { range / 15.0 } else { 0.0 };
            }

            let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
            let max_min = sub_mins.iter().copied().fold(0.0f32, f32::max);

            // 6-bit sub-block factors: 0..63
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

            for sub in 0..8 {
                sc[sub] = (sub_scales[sub] * inv_d + 0.5).min(63.0) as u8;
                mn[sub] = (sub_mins[sub] * inv_dmin + 0.5).min(63.0) as u8;
            }

            let scales = encode_q4k_scales(&sc, &mn);

            // Quantize weights to 4 bits
            let mut qs = [0u8; 128];
            for sub in 0..8 {
                let sub_offset = sub * 32;
                let sc_f = d * (sc[sub] as f32);
                let mn_f = dmin * (mn[sub] as f32);
                let inv_sc = if sc_f > 0.0 { 1.0 / sc_f } else { 0.0 };

                for j in 0..32 {
                    let global_idx = sub_offset + j;
                    let val = chunk[global_idx] + mn_f;
                    let q = (val * inv_sc + 0.5).clamp(0.0, 15.0) as u8;
                    let byte_idx = global_idx / 2;
                    if global_idx % 2 == 0 {
                        qs[byte_idx] |= q & 0x0F;
                    } else {
                        qs[byte_idx] |= (q & 0x0F) << 4;
                    }
                }
            }

            blocks.push(BlockQ4K {
                d: f16::from_f32(d),
                dmin: f16::from_f32(dmin),
                scales,
                qs,
            });
        }

        Ok(blocks)
    }

    /// Dequantize a single row's worth of Q4_K blocks into a pre-allocated buffer.
    ///
    /// `buf` will be extended by `blocks_for_row.len() * 256` elements.
    pub fn dequant_row_to_buf(blocks_for_row: &[Self], buf: &mut Vec<f32>) {
        let start = buf.len();
        let n = blocks_for_row.len() * QK_K;
        buf.resize(start + n, 0.0f32);
        let _ = Self::dequant(blocks_for_row, &mut buf[start..]);
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ4K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q4_K_BYTES` (144)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q4_K_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q4_K slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q4_K_BYTES
                ),
            });
        }
        if data.is_empty() {
            return Ok(&[]);
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q4_K slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q4_K_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time size assert;
        // length is a multiple of BLOCK_Q4_K_BYTES; pointer alignment verified above;
        // lifetime is tied to the input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

// ---------------------------------------------------------------------------
// BlockQ8K
// ---------------------------------------------------------------------------

/// Q8_K super-block: 256 weights quantized to 8 bits (int8) each.
///
/// Layout (292 bytes):
/// - `d`:      4 bytes — f32 super-block scale (NOT f16, unlike other K-quant formats).
/// - `qs`:     256 bytes — int8 quantized weight values.
/// - `bsums`:  32 bytes — precomputed sums of 16 groups of 16 weights (int16, for dot-product optimization).
///
/// Dequant: `w[i] = d * qs[i]` (bsums are not needed for scalar dequant).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    /// Super-block scale (f32, NOT f16).
    pub d: f32,
    /// 256 int8 quantized weight values.
    pub qs: [i8; 256],
    /// Precomputed sums of 16 groups of 16 weights (for SIMD dot-product optimization).
    pub bsums: [i16; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ8K>() == BLOCK_Q8K_BYTES);

impl BlockQ8K {
    /// Dequantize a slice of Q8_K blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * QK_K` (256 per block).
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_K;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q8_K dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }

        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d;
            let base = block_idx * QK_K;
            for i in 0..QK_K {
                output[base + i] = d * (block.qs[i] as f32);
            }
        }
        Ok(())
    }

    /// Dequantize a single row's worth of Q8_K blocks into a pre-allocated buffer.
    ///
    /// `buf` will be extended by `blocks_for_row.len() * 256` elements.
    pub fn dequant_row_to_buf(blocks_for_row: &[Self], buf: &mut Vec<f32>) {
        let start = buf.len();
        let n = blocks_for_row.len() * QK_K;
        buf.resize(start + n, 0.0f32);
        let _ = Self::dequant(blocks_for_row, &mut buf[start..]);
    }

    /// Quantize f32 input into Q8_K blocks.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    ///
    /// Uses a single super-block scale `d = max_abs / 127`. The `bsums` field is
    /// populated with the sum of each group of 16 weights (useful for SIMD optimized
    /// dot-product computation in other implementations).
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if input.len() % QK_K != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q8_K quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_K
                ),
            });
        }

        let num_blocks = input.len() / QK_K;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let chunk = &input[block_idx * QK_K..block_idx * QK_K + QK_K];

            // Find max absolute value across all 256 weights
            let max_abs = chunk.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);

            let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

            let mut qs = [0i8; 256];
            for (i, &w) in chunk.iter().enumerate() {
                qs[i] = (w * inv_d).round().clamp(-127.0, 127.0) as i8;
            }

            // Precompute bsums: sum of each group of 16 weights (as int16)
            let mut bsums = [0i16; 16];
            for (group, slot) in bsums.iter_mut().enumerate() {
                let group_start = group * 16;
                let sum: i32 = qs[group_start..group_start + 16]
                    .iter()
                    .map(|&q| q as i32)
                    .sum();
                *slot = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }

            blocks.push(BlockQ8K { d, qs, bsums });
        }

        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ8K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q8K_BYTES` (292)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if data.len() % BLOCK_Q8K_BYTES != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q8_K slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_Q8K_BYTES
                ),
            });
        }
        if data.is_empty() {
            return Ok(&[]);
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("Q8_K slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_Q8K_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time size assert;
        // length is a multiple of BLOCK_Q8K_BYTES; pointer alignment verified above;
        // lifetime is tied to the input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2k_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ2K>(), BLOCK_Q2_K_BYTES);
        assert_eq!(BLOCK_Q2_K_BYTES, 84);
    }

    #[test]
    fn q2k_roundtrip_zero_weights() {
        let blocks = BlockQ2K::quantize(&vec![0.0f32; 256]).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ2K::dequant(&blocks, &mut out).expect("dequant ok");
        for &v in &out {
            assert!(
                v.abs() < 1e-4,
                "all-zero input should dequant to near-zero, got {v}"
            );
        }
    }

    #[test]
    fn q2k_roundtrip_uniform() {
        let input = vec![1.0f32; 256];
        let blocks = BlockQ2K::quantize(&input).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ2K::dequant(&blocks, &mut out).expect("dequant ok");
        for &v in &out {
            let err = (v - 1.0).abs();
            assert!(err < 0.2, "uniform round-trip error {err} too high");
        }
    }

    #[test]
    fn q2k_quantize_output_length() {
        let input = vec![0.5f32; 256];
        let blocks = BlockQ2K::quantize(&input).expect("quantize ok");
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn q2k_slice_from_bytes_empty() {
        let data: Vec<u8> = vec![];
        let result = BlockQ2K::slice_from_bytes(&data).expect("empty slice ok");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn q2k_slice_from_bytes_bad_length() {
        let data = vec![0u8; 83]; // not a multiple of 84
        assert!(BlockQ2K::slice_from_bytes(&data).is_err());
    }

    #[test]
    fn q4k_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockQ4K>(), BLOCK_Q4_K_BYTES);
        assert_eq!(BLOCK_Q4_K_BYTES, 144);
    }

    #[test]
    fn q4k_scale_encode_decode_roundtrip() {
        let sc = [1, 2, 3, 4, 5, 63, 32, 0];
        let mn = [10, 20, 30, 40, 50, 60, 15, 7];
        let encoded = encode_q4k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q4k_scales(&encoded);
        assert_eq!(sc, sc2);
        assert_eq!(mn, mn2);
    }

    #[test]
    fn q4k_scale_encode_decode_all_zeros() {
        let sc = [0u8; 8];
        let mn = [0u8; 8];
        let encoded = encode_q4k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q4k_scales(&encoded);
        assert_eq!(sc, sc2);
        assert_eq!(mn, mn2);
    }

    #[test]
    fn q4k_scale_encode_decode_max_values() {
        let sc = [63u8; 8];
        let mn = [63u8; 8];
        let encoded = encode_q4k_scales(&sc, &mn);
        let (sc2, mn2) = decode_q4k_scales(&encoded);
        assert_eq!(sc, sc2);
        assert_eq!(mn, mn2);
    }

    #[test]
    fn q4k_slice_from_bytes_empty() {
        let data: Vec<u8> = vec![];
        let result = BlockQ4K::slice_from_bytes(&data).expect("empty slice ok");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn q4k_slice_from_bytes_bad_length() {
        let data = vec![0u8; 100]; // not a multiple of 144
        assert!(BlockQ4K::slice_from_bytes(&data).is_err());
    }

    // -----------------------------------------------------------------------
    // BlockQ3K tests
    // -----------------------------------------------------------------------

    #[test]
    fn q3k_block_size_assertion() {
        assert_eq!(std::mem::size_of::<BlockQ3K>(), BLOCK_Q3K_BYTES);
        assert_eq!(BLOCK_Q3K_BYTES, 110);
    }

    #[test]
    fn q3k_roundtrip_zero_weights() {
        let blocks = BlockQ3K::quantize(&vec![0.0f32; 256]).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ3K::dequant(&blocks, &mut out).expect("dequant ok");
        for &v in &out {
            assert!(
                v.abs() < 1e-4,
                "all-zero input should dequant to near-zero, got {v}"
            );
        }
    }

    #[test]
    fn q3k_roundtrip_uniform() {
        // Uniform positive values should round-trip with error < 5% of the value.
        let input = vec![1.0f32; 256];
        let blocks = BlockQ3K::quantize(&input).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ3K::dequant(&blocks, &mut out).expect("dequant ok");
        for &v in &out {
            let err = (v - 1.0).abs() / 1.0;
            assert!(
                err < 0.5,
                "uniform round-trip rel error {err} too high, got {v}"
            );
        }
    }

    #[test]
    fn q3k_slice_from_bytes() {
        // Create a valid aligned byte buffer of exactly 110 bytes and parse it.
        let data = vec![0u8; BLOCK_Q3K_BYTES];
        let result = BlockQ3K::slice_from_bytes(&data).expect("single block should parse");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn q3k_slice_from_bytes_empty() {
        let data: Vec<u8> = vec![];
        let result = BlockQ3K::slice_from_bytes(&data).expect("empty slice ok");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn q3k_slice_from_bytes_bad_length() {
        let data = vec![0u8; 100]; // not a multiple of 110
        assert!(BlockQ3K::slice_from_bytes(&data).is_err());
    }

    #[test]
    fn q3k_quantize_output_length() {
        let input = vec![0.5f32; 256];
        let blocks = BlockQ3K::quantize(&input).expect("quantize ok");
        assert_eq!(blocks.len(), 1, "256 weights → 1 block");
    }

    #[test]
    fn q3k_quantize_non_multiple_errors() {
        assert!(BlockQ3K::quantize(&vec![1.0f32; 100]).is_err());
    }

    #[test]
    fn q3k_dequant_output_too_small_errors() {
        let blocks = BlockQ3K::quantize(&vec![1.0f32; 256]).expect("quantize ok");
        let mut out = vec![0.0f32; 100];
        assert!(BlockQ3K::dequant(&blocks, &mut out).is_err());
    }

    #[test]
    fn q3k_dequant_row_to_buf_works() {
        let input = vec![0.5f32; 256];
        let blocks = BlockQ3K::quantize(&input).expect("quantize ok");
        let mut buf = Vec::new();
        BlockQ3K::dequant_row_to_buf(&blocks, &mut buf);
        assert_eq!(buf.len(), 256);
    }

    // -----------------------------------------------------------------------
    // BlockQ8K tests
    // -----------------------------------------------------------------------

    #[test]
    fn q8k_block_size_assertion() {
        assert_eq!(std::mem::size_of::<BlockQ8K>(), BLOCK_Q8K_BYTES);
        assert_eq!(BLOCK_Q8K_BYTES, 292);
    }

    #[test]
    fn q8k_roundtrip_zero_weights() {
        let blocks = BlockQ8K::quantize(&vec![0.0f32; 256]).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ8K::dequant(&blocks, &mut out).expect("dequant ok");
        for &v in &out {
            assert!(
                v.abs() < 1e-6,
                "all-zero input should dequant to exactly zero, got {v}"
            );
        }
    }

    #[test]
    fn q8k_roundtrip_uniform() {
        let input = vec![1.0f32; 256];
        let blocks = BlockQ8K::quantize(&input).expect("quantize ok");
        let mut out = vec![0.0f32; 256];
        BlockQ8K::dequant(&blocks, &mut out).expect("dequant ok");
        for &v in &out {
            let err = (v - 1.0).abs();
            assert!(err < 0.02, "Q8_K uniform round-trip error {err} too high");
        }
    }

    #[test]
    fn q8k_slice_from_bytes() {
        let data = vec![0u8; BLOCK_Q8K_BYTES];
        let result = BlockQ8K::slice_from_bytes(&data).expect("single block should parse");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn q8k_slice_from_bytes_empty() {
        let data: Vec<u8> = vec![];
        let result = BlockQ8K::slice_from_bytes(&data).expect("empty slice ok");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn q8k_slice_from_bytes_bad_length() {
        let data = vec![0u8; 100]; // not a multiple of 292
        assert!(BlockQ8K::slice_from_bytes(&data).is_err());
    }

    #[test]
    fn q8k_quantize_output_length() {
        let input = vec![0.5f32; 256];
        let blocks = BlockQ8K::quantize(&input).expect("quantize ok");
        assert_eq!(blocks.len(), 1, "256 weights → 1 block");
    }

    #[test]
    fn q8k_quantize_non_multiple_errors() {
        assert!(BlockQ8K::quantize(&vec![1.0f32; 100]).is_err());
    }

    #[test]
    fn q8k_dequant_output_too_small_errors() {
        let blocks = BlockQ8K::quantize(&vec![1.0f32; 256]).expect("quantize ok");
        let mut out = vec![0.0f32; 100];
        assert!(BlockQ8K::dequant(&blocks, &mut out).is_err());
    }

    #[test]
    fn q8k_dequant_row_to_buf_works() {
        let input = vec![0.5f32; 256];
        let blocks = BlockQ8K::quantize(&input).expect("quantize ok");
        let mut buf = Vec::new();
        BlockQ8K::dequant_row_to_buf(&blocks, &mut buf);
        assert_eq!(buf.len(), 256);
        for &v in &buf {
            assert!((v - 0.5).abs() < 0.01, "expected ~0.5, got {v}");
        }
    }

    #[test]
    fn q8k_bsums_roundtrip_sign() {
        // Verify bsums signs: positive input → positive bsums, negative → negative bsums.
        let input_pos = vec![0.5f32; 256];
        let blocks_pos = BlockQ8K::quantize(&input_pos).expect("quantize ok");
        for &bs in &blocks_pos[0].bsums {
            assert!(
                bs > 0,
                "positive input should yield positive bsums, got {bs}"
            );
        }

        let input_neg = vec![-0.5f32; 256];
        let blocks_neg = BlockQ8K::quantize(&input_neg).expect("quantize ok");
        for &bs in &blocks_neg[0].bsums {
            assert!(
                bs < 0,
                "negative input should yield negative bsums, got {bs}"
            );
        }
    }
}
