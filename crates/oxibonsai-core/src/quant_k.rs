//! K-quant block types for Q2_K, Q3_K, Q4_K, and Q8_K quantization formats.
//!
//! These follow the GGML K-quant specification:
//! - **Q2_K**: 2-bit quantization with 4-bit scales, super-block of 256 weights (84 bytes)
//! - **Q3_K**: 3-bit quantization with 6-bit scales, super-block of 256 weights (110 bytes)
//! - **Q4_K**: 4-bit quantization with 6-bit scales, super-block of 256 weights (144 bytes)
//! - **Q8_K**: 8-bit quantization with FP32 scale, super-block of 256 weights (292 bytes)
//!
//! Each super-block stores a global `d` (scale) and `dmin` (minimum) in FP16,
//! plus per-sub-block scales and quantized weight nibbles/pairs.
//!
//! # ggml byte-exactness
//!
//! The Q2_K / Q3_K / Q4_K codecs are **verbatim transliterations** of the ggml
//! reference implementation (`ggml/src/ggml-quants.c`), keeping ggml's loop
//! structure rather than an "equivalent" element-sequential rewrite:
//!
//! - `dequantize_row_q2_K` (`:1016`) / `quantize_row_q2_K_ref` (`:946`)
//! - `dequantize_row_q3_K` (`:1360`) / `quantize_row_q3_K_ref` (`:1284`)
//! - `dequantize_row_q4_K` (`:1584`) / `quantize_row_q4_K_ref` (`:1512`)
//! - `get_scale_min_k4` (`:935`), `nearest_int` (`:676`),
//!   `make_qkx2_quants` (`:854`), `make_q3_quants` (`:752`), `make_qx_quants` (`:683`)
//!
//! The sub-block scale/weight interleaving is load-bearing: Q2_K assigns
//! sub-block scales in `is++` order under a shift stepping 0, 2, 4, 6 per
//! 128-element half, and Q4_K emits 32 *low* nibbles then 32 *high* nibbles per
//! 64-element group. Reproducing these orders is what lets OxiBonsai read
//! third-party llama.cpp K-quant GGUFs (and write files llama.cpp can read).

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
// ggml reference primitives (ggml/src/ggml-quants.c)
// ---------------------------------------------------------------------------

/// `GROUP_MAX_EPS` (`ggml-quants.c:20`) — the "all zero" cut-off used by the
/// reference quantizers.
pub(crate) const GROUP_MAX_EPS: f32 = 1e-15;

/// ggml's `nearest_int` (`ggml-quants.c:676`).
///
/// This is deliberately **not** `f32::round`: the magic-number trick rounds
/// half-to-even, while `round()` rounds half away from zero. A single differing
/// code on a `.5` boundary breaks bit-exactness against llama.cpp, so the bit
/// trick is reproduced exactly. Matching ggml's release builds (`NDEBUG`), the
/// `|fval| <= 4194303` assertion is not enforced; every call site in this module
/// feeds values bounded by the format's `nmax`.
#[inline]
pub(crate) fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0_f32;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// ggml's `get_scale_min_k4` (`ggml-quants.c:935`).
///
/// Unpacks sub-block `j`'s 6-bit scale and 6-bit min from the 12-byte packed
/// `scales` array shared by Q4_K and Q5_K. For `j < 4` the scale is a **full
/// 6-bit value read from byte `j`** — not two 4-bit nibbles.
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Pack eight 6-bit scales and eight 6-bit mins into the 12-byte K-quant
/// `scales` array, exactly as `quantize_row_q4_K_ref` (`ggml-quants.c:1543-1556`)
/// and `quantize_row_q5_K_ref` (`:1730-1743`) do.
///
/// The ascending `j` order is load-bearing: `j < 4` *assigns* bytes `j` and
/// `j + 4`, and `j >= 4` then ORs the top two bits into those same bytes.
pub(crate) fn pack_scales_min_k4(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for j in 0..8usize {
        let ls = sc[j].min(63);
        let lm = mn[j].min(63);
        if j < 4 {
            out[j] = ls;
            out[j + 4] = lm;
        } else {
            out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            out[j - 4] |= (ls >> 4) << 6;
            out[j] |= (lm >> 4) << 6;
        }
    }
    out
}

/// ggml's `make_qx_quants` (`ggml-quants.c:683`) — RMSE-refined symmetric
/// quantization used by the Q6_K reference encoder.
///
/// `qw` is ggml's optional per-weight importance vector (`NULL` for the plain
/// `_ref` path).
#[allow(clippy::needless_range_loop)]
pub(crate) fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    l: &mut [i8],
    rmse_type: i32,
    qw: Option<&[f32]>,
) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    if rmse_type == 0 {
        for i in 0..n {
            let q = nearest_int(iscale * x[i]);
            l[i] = (nmax + q.clamp(-nmax, nmax - 1)) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let mut return_early = false;
    if rmse_type < 0 {
        rmse_type = -rmse_type;
        return_early = true;
    }
    let weight = |i: usize| -> f32 {
        match qw {
            Some(w) => w[i],
            None => match rmse_type {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(),
            },
        }
    };
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (q + nmax) as i8;
        let w = weight(i);
        sumlx += w * x[i] * (q as f32);
        suml2 += w * (q as f32) * (q as f32);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -((nmax as f32) + 0.1 * (is as f32)) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = weight(i);
            sumlx += w * x[i] * (q as f32);
            suml2 += w * (q as f32) * (q as f32);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let q = nearest_int(iscale * x[i]);
                l[i] = (nmax + q.clamp(-nmax, nmax - 1)) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// ggml's `make_q3_quants` (`ggml-quants.c:752`) — the Q3_K reference
/// sub-block quantizer (called with `do_rmse = true`).
#[allow(clippy::needless_range_loop)]
pub(crate) fn make_q3_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8], do_rmse: bool) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n {
            l[i] = 0;
        }
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            l[i] = q as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * (q as f32);
            suml2 += w * (q as f32) * (q as f32);
        }
        for _itry in 0..5 {
            let mut n_changed = 0u32;
            for i in 0..n {
                let w = x[i] * x[i];
                let mut slx = sumlx - w * x[i] * (l[i] as f32);
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * (l[i] as f32) * (l[i] as f32);
                    let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                    if new_l != l[i] as i32 {
                        slx += w * x[i] * (new_l as f32);
                        sl2 += w * (new_l as f32) * (new_l as f32);
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        for i in 0..n {
            l[i] += nmax as i8;
        }
        return if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    }
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (q + nmax) as i8;
    }
    1.0 / iscale
}

/// ggml's `make_qkx2_quants` (`ggml-quants.c:854`) — the asymmetric
/// (scale + min) sub-block quantizer used by Q2_K, Q4_K and Q5_K.
///
/// Returns the sub-block scale and writes the sub-block min (already negated,
/// matching ggml's `*the_min = -min`) into `the_min`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for i in 0..n {
            l[i] = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = (nmax as f32) / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min));
        l[i] = q.clamp(0, nmax) as u8;
        let mut diff = scale * (l[i] as f32) + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * (is as f32) + (nmax as f32)) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let w = weights[i];
            sum_l += w * (q as f32);
            sum_l2 += w * (q as f32) * (q as f32);
            sum_xl += w * (q as f32) * x[i];
        }
        let dd = sum_w * sum_l2 - sum_l * sum_l;
        if dd > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / dd;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / dd;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let mut diff = this_scale * (laux[i] as f32) + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

// ---------------------------------------------------------------------------
// BlockQ2K
// ---------------------------------------------------------------------------

/// Q2_K super-block: 256 weights quantized to 2 bits each.
///
/// Layout (84 bytes, `block_q2_K` in `ggml-common.h`):
/// - `scales`: 16 bytes — packed 4-bit scale/min pairs for 16 sub-blocks of 16 weights.
///   Each byte holds two 4-bit values: low nibble = scale, high nibble = min.
/// - `qs`: 64 bytes — 256 x 2-bit quantized weights.
/// - `d`: FP16 super-block scale.
/// - `dmin`: FP16 super-block minimum.
///
/// The 2-bit codes are **not** element-sequential. Per 128-element half, `qs`
/// byte `l` carries elements `l`, `l + 32`, `l + 64`, `l + 96` in bit lanes
/// 0-1, 2-3, 4-5, 6-7, and the sub-block scales are consumed in `is++` order
/// under a shift stepping 0, 2, 4, 6 (`dequantize_row_q2_K`, `ggml-quants.c:1016`).
///
/// Dequant: `w = d * sub_scale * q - dmin * sub_min`
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
    /// Byte-exact transliteration of `dequantize_row_q2_K` (`ggml-quants.c:1016`).
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
            let min = block.dmin.to_f32();

            let mut y = block_idx * QK_K;
            let mut is = 0usize;
            let mut q_off = 0usize;

            for _n in 0..(QK_K / 128) {
                let mut shift = 0u32;
                for _j in 0..4 {
                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * ((sc & 0xF) as f32);
                    let ml = min * ((sc >> 4) as f32);
                    for l in 0..16usize {
                        output[y] = dl * (((block.qs[q_off + l] >> shift) & 3) as f32) - ml;
                        y += 1;
                    }

                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * ((sc & 0xF) as f32);
                    let ml = min * ((sc >> 4) as f32);
                    for l in 0..16usize {
                        output[y] = dl * (((block.qs[q_off + l + 16] >> shift) & 3) as f32) - ml;
                        y += 1;
                    }

                    shift += 2;
                }
                q_off += 32;
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q2_K blocks.
    ///
    /// Byte-exact transliteration of `quantize_row_q2_K_ref` (`ggml-quants.c:946`),
    /// including `make_qkx2_quants(16, 3, …, -0.5, 0.1, 15, use_mad = true)` and
    /// the `q4scale = 15` scale/min requantization.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    #[allow(clippy::needless_range_loop)]
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if !input.len().is_multiple_of(QK_K) {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "Q2_K quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_K
                ),
            });
        }

        const Q4SCALE: f32 = 15.0;

        let num_blocks = input.len() / QK_K;
        let mut blocks = Vec::with_capacity(num_blocks);

        let mut l_codes = [0u8; QK_K];
        let mut laux = [0u8; 16];
        let mut weights = [0.0f32; 16];
        let mut mins = [0.0f32; QK_K / 16];
        let mut scales = [0.0f32; QK_K / 16];

        for block_idx in 0..num_blocks {
            let chunk = &input[block_idx * QK_K..block_idx * QK_K + QK_K];

            let mut max_scale = 0.0f32;
            let mut max_min = 0.0f32;
            for j in 0..(QK_K / 16) {
                for l in 0..16usize {
                    weights[l] = chunk[16 * j + l].abs();
                }
                let mut this_min = 0.0f32;
                scales[j] = make_qkx2_quants(
                    16,
                    3,
                    &chunk[16 * j..16 * j + 16],
                    &weights,
                    &mut l_codes[16 * j..16 * j + 16],
                    &mut this_min,
                    &mut laux,
                    -0.5,
                    0.1,
                    15,
                    true,
                );
                mins[j] = this_min;
                if scales[j] > max_scale {
                    max_scale = scales[j];
                }
                if mins[j] > max_min {
                    max_min = mins[j];
                }
            }

            // The scale half assigns `sc_bytes[j]`; the min half then ORs into
            // the same bytes, so this order is load-bearing.
            let mut sc_bytes = [0u8; 16];
            let d = if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for j in 0..(QK_K / 16) {
                    sc_bytes[j] = nearest_int(iscale * scales[j]) as u8;
                }
                f16::from_f32(max_scale / Q4SCALE)
            } else {
                f16::from_f32(0.0)
            };

            let dmin = if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for j in 0..(QK_K / 16) {
                    let l = nearest_int(iscale * mins[j]) as u8;
                    sc_bytes[j] |= l << 4;
                }
                f16::from_f32(max_min / Q4SCALE)
            } else {
                f16::from_f32(0.0)
            };

            for j in 0..(QK_K / 16) {
                let dj = d.to_f32() * ((sc_bytes[j] & 0xF) as f32);
                if dj == 0.0 {
                    continue;
                }
                let dm = dmin.to_f32() * ((sc_bytes[j] >> 4) as f32);
                for ii in 0..16usize {
                    let l = nearest_int((chunk[16 * j + ii] + dm) / dj).clamp(0, 3);
                    l_codes[16 * j + ii] = l as u8;
                }
            }

            let mut qs = [0u8; 64];
            for j in (0..QK_K).step_by(128) {
                for l in 0..32usize {
                    qs[j / 4 + l] = l_codes[j + l]
                        | (l_codes[j + l + 32] << 2)
                        | (l_codes[j + l + 64] << 4)
                        | (l_codes[j + l + 96] << 6);
                }
            }

            blocks.push(BlockQ2K {
                scales: sc_bytes,
                qs,
                d,
                dmin,
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
        if !data.len().is_multiple_of(BLOCK_Q2_K_BYTES) {
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
/// Layout (110 bytes, `block_q3_K` in `ggml-common.h`):
/// - `hmask`:  32 bytes — the *inverted* high bit for each of the 256 weights.
/// - `qs`:     64 bytes — low 2 bits for each of the 256 weights.
/// - `scales`: 12 bytes — sixteen **6-bit** sub-block scales, biased by +32 and
///   split into a 4-bit low part (bytes 0..8) and a 2-bit high part (bytes 8..12).
/// - `d`: FP16 super-block scale.
///
/// Dequant (`dequantize_row_q3_K`, `ggml-quants.c:1360`):
/// `w = d * (scale - 32) * (low2 - (hmask_bit ? 0 : 4))` — note the **inverted**
/// high-bit convention: a *set* `hmask` bit means "do not subtract 4".
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    /// Inverted high bit for each of 256 weights, packed 8 per byte.
    pub hmask: [u8; 32],
    /// Low 2 bits for each of 256 weights, packed 4 per byte.
    pub qs: [u8; 64],
    /// 16 x 6-bit sub-block scales in ggml's split 4-bit/2-bit packing.
    pub scales: [u8; 12],
    /// Super-block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockQ3K>() == BLOCK_Q3K_BYTES);

/// `kmask1` from `dequantize_row_q3_K` (`ggml-quants.c:1364`).
const Q3K_KMASK1: u32 = 0x0303_0303;
/// `kmask2` from `dequantize_row_q3_K` (`ggml-quants.c:1365`).
const Q3K_KMASK2: u32 = 0x0f0f_0f0f;

/// Unpack the 12 packed Q3_K scale bytes into 16 biased 6-bit scales, exactly
/// as `dequantize_row_q3_K` does with its `aux[4]` / `kmask1` / `kmask2` shuffle
/// (`ggml-quants.c:1374-1381`). The returned values still carry ggml's `+32`
/// bias; callers subtract 32.
fn unpack_q3k_scales(scales: &[u8; 12]) -> [u8; 16] {
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    aux[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    aux[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);

    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & Q3K_KMASK2) | (((tmp >> 4) & Q3K_KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & Q3K_KMASK2) | (((tmp >> 6) & Q3K_KMASK1) << 4);
    aux[0] = (aux[0] & Q3K_KMASK2) | (((tmp) & Q3K_KMASK1) << 4);
    aux[1] = (aux[1] & Q3K_KMASK2) | (((tmp >> 2) & Q3K_KMASK1) << 4);

    let mut out = [0u8; 16];
    for (k, word) in aux.iter().enumerate() {
        out[4 * k..4 * k + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

impl BlockQ3K {
    /// Dequantize a slice of Q3_K blocks into f32 output.
    ///
    /// Byte-exact transliteration of `dequantize_row_q3_K` (`ggml-quants.c:1360`),
    /// including the `-32` scale bias and the inverted `hmask` convention.
    ///
    /// `output` must have length >= `blocks.len() * QK_K` (256 per block).
    #[allow(clippy::needless_range_loop)]
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
            let d_all = block.d.to_f32();
            let sc = unpack_q3k_scales(&block.scales);
            let hm = &block.hmask;

            let mut y = block_idx * QK_K;
            let mut is = 0usize;
            let mut q_off = 0usize;
            let mut m: u8 = 1;

            for _n in 0..(QK_K / 128) {
                let mut shift = 0u32;
                for _j in 0..4 {
                    let dl = d_all * ((sc[is] as i32 - 32) as f32);
                    is += 1;
                    for l in 0..16usize {
                        let hi = if (hm[l] & m) != 0 { 0i32 } else { 4i32 };
                        let q = ((block.qs[q_off + l] >> shift) & 3) as i32;
                        output[y] = dl * ((q - hi) as f32);
                        y += 1;
                    }

                    let dl = d_all * ((sc[is] as i32 - 32) as f32);
                    is += 1;
                    for l in 0..16usize {
                        let hi = if (hm[l + 16] & m) != 0 { 0i32 } else { 4i32 };
                        let q = ((block.qs[q_off + l + 16] >> shift) & 3) as i32;
                        output[y] = dl * ((q - hi) as f32);
                        y += 1;
                    }

                    shift += 2;
                    m <<= 1;
                }
                q_off += 32;
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
    /// Byte-exact transliteration of `quantize_row_q3_K_ref` (`ggml-quants.c:1284`),
    /// including `make_q3_quants(16, 4, …, do_rmse = true)`, the `-32.f/max_scale`
    /// scale requantization and the inverted-`hmask` bit extraction.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    #[allow(clippy::needless_range_loop)]
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if !input.len().is_multiple_of(QK_K) {
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

        let mut l_codes = [0i8; QK_K];
        let mut scales = [0.0f32; QK_K / 16];

        for block_idx in 0..num_blocks {
            let chunk = &input[block_idx * QK_K..block_idx * QK_K + QK_K];

            let mut max_scale = 0.0f32;
            let mut amax = 0.0f32;
            for j in 0..(QK_K / 16) {
                scales[j] = make_q3_quants(
                    16,
                    4,
                    &chunk[16 * j..16 * j + 16],
                    &mut l_codes[16 * j..16 * j + 16],
                    true,
                );
                let scale = scales[j].abs();
                if scale > amax {
                    amax = scale;
                    max_scale = scales[j];
                }
            }

            let mut sc_bytes = [0u8; 12];
            let d;
            if max_scale != 0.0 {
                let iscale = -32.0f32 / max_scale;
                for j in 0..(QK_K / 16) {
                    let mut l = nearest_int(iscale * scales[j]) as i8;
                    l = l.clamp(-32, 31) + 32;
                    if j < 8 {
                        sc_bytes[j] = (l as u8) & 0xF;
                    } else {
                        sc_bytes[j - 8] |= ((l as u8) & 0xF) << 4;
                    }
                    l >>= 4;
                    sc_bytes[j % 4 + 8] |= (l as u8) << (2 * (j / 4));
                }
                d = f16::from_f32(1.0 / iscale);
            } else {
                d = f16::from_f32(0.0);
            }

            for j in 0..(QK_K / 16) {
                let low = if j < 8 {
                    sc_bytes[j] & 0xF
                } else {
                    sc_bytes[j - 8] >> 4
                };
                let high = (sc_bytes[8 + j % 4] >> (2 * (j / 4))) & 3;
                let sc = ((low | (high << 4)) as i32) - 32;
                let dj = d.to_f32() * (sc as f32);
                if dj == 0.0 {
                    continue;
                }
                for ii in 0..16usize {
                    let l = nearest_int(chunk[16 * j + ii] / dj).clamp(-4, 3);
                    l_codes[16 * j + ii] = (l + 4) as i8;
                }
            }

            // "We put the high-bit for the 1st 8 quants into bit 0, the next 8
            // into bit 1, etc." — and the bit is only set when the code is >= 4,
            // which the decoder reads as "do not subtract 4".
            let mut hmask = [0u8; QK_K / 8];
            let mut m = 0usize;
            let mut hm: u8 = 1;
            for j in 0..QK_K {
                if l_codes[j] > 3 {
                    hmask[m] |= hm;
                    l_codes[j] -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            let mut qs = [0u8; 64];
            for j in (0..QK_K).step_by(128) {
                for l in 0..32usize {
                    qs[j / 4 + l] = (l_codes[j + l] as u8)
                        | ((l_codes[j + l + 32] as u8) << 2)
                        | ((l_codes[j + l + 64] as u8) << 4)
                        | ((l_codes[j + l + 96] as u8) << 6);
                }
            }

            blocks.push(BlockQ3K {
                hmask,
                qs,
                scales: sc_bytes,
                d,
            });
        }

        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of `BlockQ3K`.
    ///
    /// Returns error if length is not a multiple of `BLOCK_Q3K_BYTES` (110)
    /// or if the pointer is not properly aligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if !data.len().is_multiple_of(BLOCK_Q3K_BYTES) {
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
/// Layout (144 bytes, `block_q4_K` in `ggml-common.h`):
/// - `d`: FP16 super-block scale.
/// - `dmin`: FP16 super-block minimum.
/// - `scales`: 12 bytes — eight **6-bit** scales and eight 6-bit mins, packed by
///   `get_scale_min_k4` (`ggml-quants.c:935`): for `j < 4` the scale is the full
///   low six bits of byte `j` and the min the low six bits of byte `j + 4`; for
///   `j >= 4` the low nibbles live in bytes `j + 4` and the high two bits in the
///   top bits of bytes `j - 4` / `j`.
/// - `qs`: 128 bytes — 256 x 4-bit quantized weights.
///
/// The nibble order is **not** element-sequential: per 64-element group the
/// decoder emits 32 *low* nibbles (`qs[l] & 0xF`) then 32 *high* nibbles
/// (`qs[l] >> 4`).
///
/// Dequant: `w = d * sub_scale * q - dmin * sub_min`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ4K {
    /// Super-block scale (FP16).
    pub d: f16,
    /// Super-block minimum (FP16).
    pub dmin: f16,
    /// Packed 6-bit scales and mins for 8 sub-blocks.
    pub scales: [u8; 12],
    /// 256 x 4-bit quantized weights, 2 per byte.
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ4K>() == BLOCK_Q4_K_BYTES);

/// Decode all eight 6-bit scales and eight 6-bit mins from a Q4_K/Q5_K
/// 12-byte `scales` array by calling ggml's [`get_scale_min_k4`] for each
/// sub-block.
///
/// The hot decode path calls [`get_scale_min_k4`] directly (as ggml does), so
/// this batched form only exists for the packing round-trip tests.
#[cfg(test)]
fn decode_q4k_scales(scales_raw: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..8usize {
        let (d, m) = get_scale_min_k4(j, scales_raw);
        sc[j] = d;
        mn[j] = m;
    }
    (sc, mn)
}

/// Encode eight 6-bit scale values and eight 6-bit min values into the 12-byte
/// packed format used by Q4_K and Q5_K — the exact inverse of
/// [`get_scale_min_k4`].
fn encode_q4k_scales(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    pack_scales_min_k4(sc, mn)
}

impl BlockQ4K {
    /// Dequantize a slice of Q4_K blocks into f32 output.
    ///
    /// Byte-exact transliteration of `dequantize_row_q4_K` (`ggml-quants.c:1584`).
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
            let min = block.dmin.to_f32();

            let mut y = block_idx * QK_K;
            let mut q_off = 0usize;
            let mut is = 0usize;

            for _j in (0..QK_K).step_by(64) {
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * (sc as f32);
                let m1 = min * (m as f32);
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * (sc as f32);
                let m2 = min * (m as f32);

                for l in 0..32usize {
                    output[y] = d1 * ((block.qs[q_off + l] & 0xF) as f32) - m1;
                    y += 1;
                }
                for l in 0..32usize {
                    output[y] = d2 * ((block.qs[q_off + l] >> 4) as f32) - m2;
                    y += 1;
                }
                q_off += 32;
                is += 2;
            }
        }
        Ok(())
    }

    /// Quantize f32 input into Q4_K blocks.
    ///
    /// Byte-exact transliteration of `quantize_row_q4_K_ref` (`ggml-quants.c:1512`),
    /// including the `av_x + |x|` importance weights and
    /// `make_qkx2_quants(32, 15, …, -1.0, 0.1, 20, use_mad = false)`.
    ///
    /// Input length must be a multiple of `QK_K` (256).
    #[allow(clippy::needless_range_loop)]
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if !input.len().is_multiple_of(QK_K) {
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
                    15,
                    &chunk[32 * j..32 * j + 32],
                    &weights,
                    &mut l_codes[32 * j..32 * j + 32],
                    &mut this_min,
                    &mut laux,
                    -1.0,
                    0.1,
                    20,
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
            let sc_bytes = encode_q4k_scales(&sc_arr, &mn_arr);

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
                    let l = nearest_int((chunk[32 * j + ii] + dm) / dj).clamp(0, 15);
                    l_codes[32 * j + ii] = l as u8;
                }
            }

            let mut qs = [0u8; 128];
            let mut q_off = 0usize;
            for j in (0..QK_K).step_by(64) {
                for l in 0..32usize {
                    qs[q_off + l] = l_codes[j + l] | (l_codes[j + l + 32] << 4);
                }
                q_off += 32;
            }

            blocks.push(BlockQ4K {
                d,
                dmin,
                scales: sc_bytes,
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
        if !data.len().is_multiple_of(BLOCK_Q4_K_BYTES) {
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
        if !input.len().is_multiple_of(QK_K) {
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
        if !data.len().is_multiple_of(BLOCK_Q8K_BYTES) {
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
        // wave-2.5 addendum (6) (Miri, pre-existing): borrow directly from a
        // real `BlockQ3K`'s own (compiler-guaranteed-aligned) stack address,
        // rather than reinterpreting a `Vec<u8>`'s buffer — `align_of::<u8>()
        // == 1`, so nothing guarantees a `Vec<u8>` allocation is aligned to
        // `align_of::<BlockQ3K>()`; real-world allocators are generous about
        // it, but Miri's is not (`cargo +nightly miri test -p
        // oxibonsai-core --no-default-features --lib` tripped the (correct)
        // alignment guard on a perfectly legitimate all-zero block). Same
        // technique as `tensor.rs::one_bit_tensor_dequantize`.
        //
        // `f16::from_bits(0)`, not `f16::from_f32(0.0)` (verifier, wave 3):
        // re-running this fix under Miri surfaced a second, unrelated wall
        // — `half`'s aarch64 `f32_to_f16` path is `asm!`-based (hardware
        // `fcvt`), which Miri's interpreter cannot execute at all
        // ("unsupported operation: inline assembly is not supported"),
        // independent of alignment. `from_bits` is a plain `u16` transmute
        // (IEEE-754 half-precision zero is bit pattern `0x0000`, exactly
        // like `f32`/`f64`), so it exercises neither the alignment guard
        // nor any hardware conversion intrinsic. Confirmed green under
        // `cargo +nightly miri test -p oxibonsai-core --no-default-features
        // --lib q3k_slice_from_bytes` after this change (was previously
        // still red, just for this different reason, even after the
        // alignment fix alone).
        let block = BlockQ3K {
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: [0u8; 12],
            d: f16::from_bits(0),
        };
        let data: &[u8] = unsafe {
            std::slice::from_raw_parts(&block as *const BlockQ3K as *const u8, BLOCK_Q3K_BYTES)
        };
        let result = BlockQ3K::slice_from_bytes(data).expect("single block should parse");
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
        // wave-2.5 addendum (6) (Miri, pre-existing): see
        // `q3k_slice_from_bytes`'s doc comment — same borrow-from-a-real-
        // aligned-value fix, not a `Vec<u8>` reinterpreted as bytes.
        let block = BlockQ8K {
            d: 0.0f32,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        };
        let data: &[u8] = unsafe {
            std::slice::from_raw_parts(&block as *const BlockQ8K as *const u8, BLOCK_Q8K_BYTES)
        };
        let result = BlockQ8K::slice_from_bytes(data).expect("single block should parse");
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
