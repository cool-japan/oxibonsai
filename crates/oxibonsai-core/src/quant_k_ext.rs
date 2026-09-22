//! K-quant block types for Q5_K and Q6_K quantization formats.
//!
//! These follow the GGML K-quant specification:
//! - **Q5_K**: 5-bit quantization with 6-bit scales, super-block of 256 weights (176 bytes)
//! - **Q6_K**: 6-bit quantization with int8 scales, super-block of 256 weights (210 bytes)
//!
//! Both codecs are **verbatim transliterations** of the ggml reference
//! implementation (`ggml/src/ggml-quants.c`), keeping ggml's loop structure:
//!
//! - `dequantize_row_q5_K` (`:1786`) / `quantize_row_q5_K_ref` (`:1699`)
//! - `dequantize_row_q6_K` (`:1994`) / `quantize_row_q6_K_ref` (`:1924`)
//!
//! Q5_K shares Q4_K's 12-byte `get_scale_min_k4` scale packing and emits 32 low
//! nibbles then 32 high nibbles per 64-element group, with the `qh` bit masks
//! stepping `u1 <<= 2` / `u2 <<= 2` per group. Q6_K interleaves four lanes
//! (`y[l]`, `y[l+32]`, `y[l+64]`, `y[l+96]`) out of `ql[l]` / `ql[l+32]` and the
//! four 2-bit fields of `qh[l]`, with sub-scales `sc[is+0]`, `sc[is+2]`,
//! `sc[is+4]`, `sc[is+6]`.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};
use crate::quant_k::{
    get_scale_min_k4, make_qkx2_quants, make_qx_quants, nearest_int, pack_scales_min_k4,
    GROUP_MAX_EPS, QK_K,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of bytes per Q5_K block (2 + 2 + 12 + 32 + 128 = 176).
pub const BLOCK_Q5K_BYTES: usize = 176;

/// Number of bytes per Q6_K block (128 + 64 + 16 + 2 = 210).
pub const BLOCK_Q6K_BYTES: usize = 210;

// ---------------------------------------------------------------------------
// Scale-packing helpers (shared with Q4_K — ggml `get_scale_min_k4`)
// ---------------------------------------------------------------------------

/// Decode the 8 six-bit scale values and 8 six-bit min values from the 12-byte
/// packed `scales` array of a Q5_K block.
///
/// Q5_K uses exactly the same packing as Q4_K, so this simply calls ggml's
/// [`get_scale_min_k4`] for every sub-block (`ggml-quants.c:935`).
///
/// The hot decode path calls [`get_scale_min_k4`] directly (as ggml does), so
/// this batched form only exists for the packing round-trip tests.
#[cfg(test)]
fn decode_q5k_scales(scales_raw: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..8usize {
        let (d, m) = get_scale_min_k4(j, scales_raw);
        sc[j] = d;
        mn[j] = m;
    }
    (sc, mn)
}

/// Encode 8 six-bit scale values and 8 six-bit min values into the 12-byte
/// packed format used by Q5_K — the exact inverse of [`get_scale_min_k4`].
fn encode_q5k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    pack_scales_min_k4(sc, mn)
}

// ---------------------------------------------------------------------------
// BlockQ5K
// ---------------------------------------------------------------------------

/// Q5_K super-block: 256 weights quantized to 5 bits each.
///
/// Layout (176 bytes, `block_q5_K` in `ggml-common.h`):
/// - `d`:      FP16 super-block scale.
/// - `dmin`:   FP16 super-block minimum.
/// - `scales`: 12 bytes — eight 6-bit scales and eight 6-bit mins in ggml's
///   `get_scale_min_k4` packing (identical to Q4_K).
/// - `qh`:     32 bytes — the 5th bit of each weight. Within a 64-element group
///   `g`, `qh[l]` bit `2g` carries element `64g + l` and bit `2g + 1` carries
///   element `64g + l + 32` (the decoder's `u1`/`u2 <<= 2` masks).
/// - `qs`:     128 bytes — the low 4 bits, 32 low nibbles then 32 high nibbles
///   per 64-element group.
///
/// Dequant: `w = d * sub_scale * ((ql & 0xF) + 16·qh_bit) - dmin * sub_min`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ5K {
    /// Super-block scale (FP16).
    pub d: f16,
    /// Super-block minimum (FP16).
    pub dmin: f16,
    /// Packed 6-bit scales for 8 sub-blocks (same layout as Q4_K).
    pub scales: [u8; 12],
    /// High bit (bit 4) for each of the 256 weights, in ggml's `u1`/`u2` order.
    pub qh: [u8; 32],
    /// Low 4 bits for each of the 256 weights, packed 2 per byte.
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ5K>() == BLOCK_Q5K_BYTES);

impl BlockQ5K {
    /// Dequantize a slice of Q5_K blocks into f32 output.
    ///
    /// Byte-exact transliteration of `dequantize_row_q5_K` (`ggml-quants.c:1786`).
    ///
    /// `output` must have length >= `blocks.len() * QK_K` (256 per block).
    #[allow(clippy::needless_range_loop)]
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
            let min = block.dmin.to_f32();
            let qh = &block.qh;

            let mut y = block_idx * QK_K;
            let mut ql_off = 0usize;
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;

            for _j in (0..QK_K).step_by(64) {
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * (sc as f32);
                let m1 = min * (m as f32);
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * (sc as f32);
                let m2 = min * (m as f32);

                for l in 0..32usize {
                    let hi = if (qh[l] & u1) != 0 { 16u32 } else { 0 };
                    output[y] = d1 * (((block.qs[ql_off + l] & 0xF) as u32 + hi) as f32) - m1;
                    y += 1;
                }
                for l in 0..32usize {
                    let hi = if (qh[l] & u2) != 0 { 16u32 } else { 0 };
                    output[y] = d2 * (((block.qs[ql_off + l] >> 4) as u32 + hi) as f32) - m2;
                    y += 1;
                }

                ql_off += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
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
    /// Byte-exact transliteration of `quantize_row_q5_K_ref` (`ggml-quants.c:1699`),
    /// including the `av_x + |x|` importance weights and
    /// `make_qkx2_quants(32, 31, …, -0.5, 0.1, 15, use_mad = false)`.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    #[allow(clippy::needless_range_loop)]
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if !input.len().is_multiple_of(QK_K) {
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

        let mut l_codes = [0u8; QK_K];
        let mut laux = [0u8; 32];
        let mut weights = [0.0f32; 32];
        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        for block_idx in 0..num_blocks {
            let chunk = &input[block_idx * QK_K..block_idx * QK_K + QK_K];

            let mut max_scale = 0.0f32;
            let mut max_min = 0.0f32;
            for j in 0..(QK_K / 32) {
                let mut sum_x2 = 0.0f32;
                for l in 0..32usize {
                    sum_x2 += chunk[32 * j + l] * chunk[32 * j + l];
                }
                let av_x = (sum_x2 / 32.0).sqrt();
                for l in 0..32usize {
                    weights[l] = av_x + chunk[32 * j + l].abs();
                }
                let mut this_min = 0.0f32;
                scales[j] = make_qkx2_quants(
                    32,
                    31,
                    &chunk[32 * j..32 * j + 32],
                    &weights,
                    &mut l_codes[32 * j..32 * j + 32],
                    &mut this_min,
                    &mut laux,
                    -0.5,
                    0.1,
                    15,
                    false,
                );
                mins[j] = this_min;
                if scales[j] > max_scale {
                    max_scale = scales[j];
                }
                if mins[j] > max_min {
                    max_min = mins[j];
                }
            }

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

            let mut sc_arr = [0u8; 8];
            let mut mn_arr = [0u8; 8];
            for j in 0..(QK_K / 32) {
                sc_arr[j] = (nearest_int(inv_scale * scales[j]) as u8).min(63);
                mn_arr[j] = (nearest_int(inv_min * mins[j]) as u8).min(63);
            }
            let sc_bytes = encode_q5k_scales(&sc_arr, &mn_arr);

            let d = f16::from_f32(max_scale / 63.0);
            let dmin = f16::from_f32(max_min / 63.0);

            for j in 0..(QK_K / 32) {
                let (sc, m) = get_scale_min_k4(j, &sc_bytes);
                let dj = d.to_f32() * (sc as f32);
                if dj == 0.0 {
                    continue;
                }
                let dm = dmin.to_f32() * (m as f32);
                for ii in 0..32usize {
                    let l = nearest_int((chunk[32 * j + ii] + dm) / dj).clamp(0, 31);
                    l_codes[32 * j + ii] = l as u8;
                }
            }

            let mut qh = [0u8; QK_K / 8];
            let mut qs = [0u8; 128];
            let mut ql_off = 0usize;
            let mut m1: u8 = 1;
            let mut m2: u8 = 2;
            for n in (0..QK_K).step_by(64) {
                for j in 0..32usize {
                    let mut l1 = l_codes[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l_codes[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    qs[ql_off + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
                ql_off += 32;
            }

            blocks.push(BlockQ5K {
                d,
                dmin,
                scales: sc_bytes,
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
        if !data.len().is_multiple_of(BLOCK_Q5K_BYTES) {
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
/// Layout (210 bytes, `block_q6_K` in `ggml-common.h`):
/// - `ql`:     128 bytes — low 4 bits, in ggml's four-lane interleave.
/// - `qh`:     64 bytes  — high 2 bits, four 2-bit fields per byte.
/// - `scales`: 16 bytes  — int8 scale for each of 16 sub-blocks of 16 weights.
/// - `d`:      FP16 super-block scale.
///
/// Per 128-element half and `l in 0..32`, the decoder emits
/// `y[l]`, `y[l+32]`, `y[l+64]`, `y[l+96]` from `ql[l]`/`ql[l+32]` and the four
/// 2-bit fields of `qh[l]`, scaled by `sc[is+0]`, `sc[is+2]`, `sc[is+4]`,
/// `sc[is+6]` where `is = l/16` (`dequantize_row_q6_K`, `ggml-quants.c:1994`).
///
/// Dequant: `w = d * sc * (q6 - 32)`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    /// Low 4 bits for each of the 256 weights, in ggml's four-lane interleave.
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
    /// Byte-exact transliteration of `dequantize_row_q6_K` (`ggml-quants.c:1994`).
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

            let mut y = block_idx * QK_K;
            let mut ql_off = 0usize;
            let mut qh_off = 0usize;
            let mut sc_off = 0usize;

            for _n in (0..QK_K).step_by(128) {
                for l in 0..32usize {
                    let is = l / 16;
                    let qh = block.qh[qh_off + l];
                    let q1 = (((block.ql[ql_off + l] & 0xF) | (((qh) & 3) << 4)) as i32) - 32;
                    let q2 =
                        (((block.ql[ql_off + l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) as i32) - 32;
                    let q3 = (((block.ql[ql_off + l] >> 4) | (((qh >> 4) & 3) << 4)) as i32) - 32;
                    let q4 =
                        (((block.ql[ql_off + l + 32] >> 4) | (((qh >> 6) & 3) << 4)) as i32) - 32;

                    output[y + l] = d * (block.scales[sc_off + is] as f32) * (q1 as f32);
                    output[y + l + 32] = d * (block.scales[sc_off + is + 2] as f32) * (q2 as f32);
                    output[y + l + 64] = d * (block.scales[sc_off + is + 4] as f32) * (q3 as f32);
                    output[y + l + 96] = d * (block.scales[sc_off + is + 6] as f32) * (q4 as f32);
                }
                y += 128;
                ql_off += 64;
                qh_off += 32;
                sc_off += 8;
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
    /// Byte-exact transliteration of `quantize_row_q6_K_ref` (`ggml-quants.c:1924`),
    /// including `make_qx_quants(16, 32, …, rmse_type = 1, qw = NULL)` and the
    /// `-128.f/max_scale` sub-scale requantization.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    #[allow(clippy::needless_range_loop)]
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if !input.len().is_multiple_of(QK_K) {
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

        let mut l_codes = [0i8; QK_K];
        let mut sub_scales = [0.0f32; QK_K / 16];

        for block_idx in 0..num_blocks {
            let chunk = &input[block_idx * QK_K..block_idx * QK_K + QK_K];

            let mut max_scale = 0.0f32;
            let mut max_abs_scale = 0.0f32;
            for ib in 0..(QK_K / 16) {
                let scale = make_qx_quants(
                    16,
                    32,
                    &chunk[16 * ib..16 * ib + 16],
                    &mut l_codes[16 * ib..16 * ib + 16],
                    1,
                    None,
                );
                sub_scales[ib] = scale;
                let abs_scale = scale.abs();
                if abs_scale > max_abs_scale {
                    max_abs_scale = abs_scale;
                    max_scale = scale;
                }
            }

            if max_abs_scale < GROUP_MAX_EPS {
                blocks.push(BlockQ6K {
                    ql: [0u8; 128],
                    qh: [0u8; 64],
                    scales: [0i8; 16],
                    d: f16::from_f32(0.0),
                });
                continue;
            }

            let iscale = -128.0f32 / max_scale;
            let d = f16::from_f32(1.0 / iscale);
            let mut scales = [0i8; 16];
            for ib in 0..(QK_K / 16) {
                scales[ib] = nearest_int(iscale * sub_scales[ib]).min(127) as i8;
            }

            for j in 0..(QK_K / 16) {
                let dj = d.to_f32() * (scales[j] as f32);
                if dj == 0.0 {
                    continue;
                }
                for ii in 0..16usize {
                    let l = nearest_int(chunk[16 * j + ii] / dj).clamp(-32, 31);
                    l_codes[16 * j + ii] = (l + 32) as i8;
                }
            }

            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut ql_off = 0usize;
            let mut qh_off = 0usize;
            for j in (0..QK_K).step_by(128) {
                for l in 0..32usize {
                    let q1 = (l_codes[j + l] as u8) & 0xF;
                    let q2 = (l_codes[j + l + 32] as u8) & 0xF;
                    let q3 = (l_codes[j + l + 64] as u8) & 0xF;
                    let q4 = (l_codes[j + l + 96] as u8) & 0xF;
                    ql[ql_off + l] = q1 | (q3 << 4);
                    ql[ql_off + l + 32] = q2 | (q4 << 4);
                    qh[qh_off + l] = ((l_codes[j + l] as u8) >> 4)
                        | (((l_codes[j + l + 32] as u8) >> 4) << 2)
                        | (((l_codes[j + l + 64] as u8) >> 4) << 4)
                        | (((l_codes[j + l + 96] as u8) >> 4) << 6);
                }
                ql_off += 64;
                qh_off += 32;
            }

            blocks.push(BlockQ6K { ql, qh, scales, d });
        }

        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ6K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q6K_BYTES` (210)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if !data.len().is_multiple_of(BLOCK_Q6K_BYTES) {
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
