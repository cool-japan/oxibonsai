//! Reference (naive) dequantization / GEMV / GEMM kernels for the PrismML
//! Bonsai 2 quantization formats: `PQ2_0` (ggml id 142), mainline
//! `Q2_0_g64` (ggml id 42, group 64) and `PTQ1_0` (ggml id 143).
//!
//! ## The one shared decode
//!
//! All three formats decode a small integer *code* to a signed value with
//! the **same arithmetic map**, `value = (code as i32) - 1`
//! (`ggml-quants.c:474-512` for the 2-bit family; `ggml-quants.c:2255-2285`
//! for `PTQ1_0`'s trit codes, which only ever take the values `0..=2`, a
//! strict subset of the 2-bit family's `0..=3`). That single mapping is
//! [`oxibonsai_core::q2_0_code_to_i32`] — every dequant/GEMV/GEMM function in
//! this file and in [`crate::gemv_ptq1`] funnels through it, so the crate has
//! exactly one place that could get the map wrong.
//!
//! What genuinely differs between the three formats is **how raw bytes turn
//! into a code**:
//!
//! - `PQ2_0` / `Q2_0_g64`: 4 codes per byte, LSB-first — see
//!   [`dequant_two_bit_block`] / [`dot_two_bit_block`], shared verbatim
//!   between the two formats (they differ only in `qs` length: 32 bytes / 128
//!   weights vs. 16 bytes / 64 weights, and both structs already place `d`
//!   before `qs`, so no extra "byte offset" parameter is needed — the qs
//!   slice IS the offset).
//! - `PTQ1_0`: five interleaved base-3 trits per byte plus four per `qh`
//!   byte, unpacked by [`decode_ptq1_0_codes`] (see its doc for the exact,
//!   trap-laden element order).
//!
//! **`PQ2_0` must never be confused with the legacy [`crate::dequant_ternary`]
//! `TQ2_0_g128` format.** Both are 34 bytes for 128 weights, but `TQ2_0_g128`
//! is `qs`-first with a three-level LUT (`0b11 → 0`, reserved), while
//! `PQ2_0` is `d`-first with the arithmetic map (`0b11 → +2`). Reusing either
//! format's block type or decode constant for the other silently corrupts
//! weights — see finding K-06. This file only ever touches
//! [`oxibonsai_core::BlockPQ2_0`] / [`oxibonsai_core::BlockQ2_0G64`] /
//! [`oxibonsai_core::BlockPTQ1_0`], never `BlockTQ2_0_g128`.

use oxibonsai_core::{
    q2_0_code_to_i32, BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, POW3, PTQ1_0_STAGES, QK_PQ2_0,
    QK_PTQ1_0, QK_Q2_0_G64,
};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// Shared 2-bit family helpers (PQ2_0, Q2_0_g64)
// ---------------------------------------------------------------------------

/// Decode one block's `qs` bytes (4 two-bit codes per byte, LSB-first) into
/// `out`, scaled by `d`. `out.len()` must be exactly `qs.len() * 4`.
///
/// This is the ONE two-bit decode loop in the crate; [`dequant_pq2_0`] and
/// [`dequant_q2_0_g64`] call it with their respective `qs` slice (32 bytes /
/// 128 weights or 16 bytes / 64 weights) — the loop bound comes from
/// `qs.len()`, so one function body serves both group sizes.
#[inline(always)]
fn dequant_two_bit_block(qs: &[u8], d: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), qs.len() * 4);
    for (byte_idx, &byte) in qs.iter().enumerate() {
        for lane in 0..4usize {
            let code = (byte >> (lane * 2)) & 0b11;
            out[byte_idx * 4 + lane] = q2_0_code_to_i32(code) as f32 * d;
        }
    }
}

/// Dot product of one block's decoded (but not `d`-scaled) two-bit weights
/// against `input`. `input.len()` must be at least `qs.len() * 4`.
///
/// Shared by [`gemv_pq2_0`] and [`gemv_q2_0_g64`] for the same reason as
/// [`dequant_two_bit_block`].
#[inline(always)]
fn dot_two_bit_block(qs: &[u8], input: &[f32]) -> f32 {
    debug_assert!(input.len() >= qs.len() * 4);
    let mut acc = 0.0f32;
    for (byte_idx, &byte) in qs.iter().enumerate() {
        let base = byte_idx * 4;
        for lane in 0..4usize {
            let code = (byte >> (lane * 2)) & 0b11;
            acc += q2_0_code_to_i32(code) as f32 * input[base + lane];
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// PQ2_0 (ggml id 142) — 128 weights/block, 34 bytes, d FIRST
// ---------------------------------------------------------------------------

/// Dequantize `PQ2_0` blocks (128 weights/block) into `output`.
///
/// `output[j] = ((code as i32) - 1) as f32 * d`, `00→-1 01→0 10→+1 11→+2`
/// (`dequantize_row_pq2_0`, `ggml-quants.c:490-512`).
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] if `output` is shorter than
/// `blocks.len() * 128`.
pub fn dequant_pq2_0(blocks: &[BlockPQ2_0], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_PQ2_0;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let base = bi * QK_PQ2_0;
        dequant_two_bit_block(
            &block.qs,
            block.d.to_f32(),
            &mut output[base..base + QK_PQ2_0],
        );
    }
    Ok(())
}

/// Scalar GEMV for a `PQ2_0`-quantized weight matrix.
///
/// - `blocks`: row-major weight blocks; row `i` starts at `i * (k / 128)`.
/// - `input`: FP32 input vector of length `k`.
/// - `output`: FP32 output vector of length `n_rows`.
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % 128 != 0`.
/// - [`KernelError::NamedDimensionMismatch`] if `input` is too short.
/// - [`KernelError::NamedBufferTooSmall`] if `output` or `blocks` is too short.
pub fn gemv_pq2_0(
    blocks: &[BlockPQ2_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PQ2_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PQ2_0,
        });
    }
    if input.len() < k {
        return Err(KernelError::dimension_mismatch("input", k, input.len()));
    }
    if output.len() < n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            n_rows,
            output.len(),
        ));
    }
    let blocks_per_row = k / QK_PQ2_0;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    for row in 0..n_rows {
        let mut sum = 0.0f32;
        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let input_base = bi * QK_PQ2_0;
            let input_slice = &input[input_base..input_base + QK_PQ2_0];
            sum += block.d.to_f32() * dot_two_bit_block(&block.qs, input_slice);
        }
        output[row] = sum;
    }
    Ok(())
}

/// Scalar GEMM for a `PQ2_0`-quantized weight matrix: batches [`gemv_pq2_0`]
/// over `m` input rows.
///
/// # Errors
///
/// Propagates every error [`gemv_pq2_0`] can return, plus
/// [`KernelError::NamedDimensionMismatch`] / [`KernelError::NamedBufferTooSmall`]
/// if `input` / `output` are too short for `m` rows.
pub fn gemm_pq2_0(
    blocks: &[BlockPQ2_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PQ2_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PQ2_0,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::dimension_mismatch("input", m * k, input.len()));
    }
    if output.len() < m * n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            m * n_rows,
            output.len(),
        ));
    }
    for batch in 0..m {
        let input_row = &input[batch * k..(batch + 1) * k];
        let output_row = &mut output[batch * n_rows..(batch + 1) * n_rows];
        gemv_pq2_0(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Q2_0_g64 (mainline ggml id 42, group 64) — 64 weights/block, 18 bytes, d FIRST
// ---------------------------------------------------------------------------

/// Dequantize `Q2_0_g64` blocks (64 weights/block) into `output`.
///
/// Same arithmetic map as [`dequant_pq2_0`] (`dequantize_row_q2_0`,
/// `ggml-quants.c:472-491`) at half the group size — **this is the mainline
/// `Q2_0` layout, not the legacy `TQ2_0_g128` ternary LUT**; see the module
/// doc's warning.
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] if `output` is shorter than
/// `blocks.len() * 64`.
pub fn dequant_q2_0_g64(blocks: &[BlockQ2_0G64], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_Q2_0_G64;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let base = bi * QK_Q2_0_G64;
        dequant_two_bit_block(
            &block.qs,
            block.d.to_f32(),
            &mut output[base..base + QK_Q2_0_G64],
        );
    }
    Ok(())
}

/// Scalar GEMV for a `Q2_0_g64`-quantized weight matrix.
///
/// - `blocks`: row-major weight blocks; row `i` starts at `i * (k / 64)`.
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % 64 != 0`.
/// - [`KernelError::NamedDimensionMismatch`] if `input` is too short.
/// - [`KernelError::NamedBufferTooSmall`] if `output` or `blocks` is too short.
pub fn gemv_q2_0_g64(
    blocks: &[BlockQ2_0G64],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_Q2_0_G64) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_Q2_0_G64,
        });
    }
    if input.len() < k {
        return Err(KernelError::dimension_mismatch("input", k, input.len()));
    }
    if output.len() < n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            n_rows,
            output.len(),
        ));
    }
    let blocks_per_row = k / QK_Q2_0_G64;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }

    for row in 0..n_rows {
        let mut sum = 0.0f32;
        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let input_base = bi * QK_Q2_0_G64;
            let input_slice = &input[input_base..input_base + QK_Q2_0_G64];
            sum += block.d.to_f32() * dot_two_bit_block(&block.qs, input_slice);
        }
        output[row] = sum;
    }
    Ok(())
}

/// Scalar GEMM for a `Q2_0_g64`-quantized weight matrix: batches
/// [`gemv_q2_0_g64`] over `m` input rows.
///
/// # Errors
///
/// Propagates every error [`gemv_q2_0_g64`] can return, plus
/// [`KernelError::NamedDimensionMismatch`] / [`KernelError::NamedBufferTooSmall`]
/// if `input` / `output` are too short for `m` rows.
pub fn gemm_q2_0_g64(
    blocks: &[BlockQ2_0G64],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_Q2_0_G64) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_Q2_0_G64,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::dimension_mismatch("input", m * k, input.len()));
    }
    if output.len() < m * n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            m * n_rows,
            output.len(),
        ));
    }
    for batch in 0..m {
        let input_row = &input[batch * k..(batch + 1) * k];
        let output_row = &mut output[batch * n_rows..(batch + 1) * n_rows];
        gemv_q2_0_g64(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PTQ1_0 (ggml id 143) — 128 weights/block, 28 bytes, d LAST, base-3 trits
// ---------------------------------------------------------------------------

/// Decode the 128 trit codes (`0`, `1`, `2`) of one `PTQ1_0` block.
///
/// Independent transliteration of `dequantize_row_ptq1_0`
/// (`ggml-quants.c:2255-2285`) at the byte level — deliberately re-derived
/// here rather than delegating to [`oxibonsai_core::BlockPTQ1_0::decode_codes`]
/// (a distinct implementation of the same C, in a different crate), so a
/// bitwise agreement between the two crates' outputs is a genuine
/// cross-check rather than one implementation calling the other. Three traps,
/// all reproduced:
///
/// 1. Stage `32` of `ptq1_0_stages = {32, 16, 8}` is **inert** at
///    `sizeof(qs) == 24` (`j + 32 <= 24` is never true), so the effective
///    stage sequence is `{16, 8}` with `j` carrying across stages.
/// 2. Element order is **interleaved**: loops are `n` outer / `m` inner, so
///    byte `qs[j + m]` supplies output positions `n * c + m`, *not*
///    `5 * (j + m) .. 5 * (j + m) + 5`.
/// 3. `qs[j + m] * pow3[n]` is `uint8_t` arithmetic and **wraps mod 256** —
///    [`u8::wrapping_mul`] is load-bearing; a widened multiply corrupts the
///    result the moment a product exceeds 255.
///
/// Resulting index map (`e` = output code index):
///
/// ```text
/// e[ n*16 + m ]     n in 0..5, m in 0..16  ->  trit n of qs[m]        (e[0..80))
/// e[ 80 + n*8 + m ] n in 0..5, m in 0..8   ->  trit n of qs[16 + m]   (e[80..120))
/// e[ 120 + n*2 + h] n in 0..4, h in 0..2   ->  trit n of qh[h]        (e[120..128))
/// ```
#[inline]
pub(crate) fn decode_ptq1_0_codes(block: &BlockPTQ1_0) -> [u8; QK_PTQ1_0] {
    let mut codes = [0u8; QK_PTQ1_0];
    let mut w = 0usize;
    let mut j = 0usize;
    for &c in PTQ1_0_STAGES.iter() {
        while j + c <= block.qs.len() {
            for &pow in POW3.iter().take(5) {
                for m in 0..c {
                    let q = block.qs[j + m].wrapping_mul(pow);
                    codes[w] = (((q as u16) * 3) >> 8) as u8;
                    w += 1;
                }
            }
            j += c;
        }
    }
    for &pow in POW3.iter().take(4) {
        for &byte in block.qh.iter() {
            let q = byte.wrapping_mul(pow);
            codes[w] = (((q as u16) * 3) >> 8) as u8;
            w += 1;
        }
    }
    debug_assert_eq!(w, QK_PTQ1_0);
    codes
}

/// Dequantize `PTQ1_0` blocks (128 weights/block) into `output`.
///
/// `output[j] = ((code as i32) - 1) as f32 * d`, the same map as the 2-bit
/// family applied to `PTQ1_0`'s trit codes (`0..=2`, a subset of `0..=3`).
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] if `output` is shorter than
/// `blocks.len() * 128`.
pub fn dequant_ptq1_0(blocks: &[BlockPTQ1_0], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_PTQ1_0;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let codes = decode_ptq1_0_codes(block);
        let d = block.d.to_f32();
        let base = bi * QK_PTQ1_0;
        for (j, &code) in codes.iter().enumerate() {
            output[base + j] = q2_0_code_to_i32(code) as f32 * d;
        }
    }
    Ok(())
}

/// Losslessly re-encode `PTQ1_0` blocks into the `PQ2_0` wire form (`d`
/// first, 34 bytes).
///
/// Lossless by construction: `PTQ1_0`'s trit codes only ever take the values
/// `{0, 1, 2}` (`xi ∈ {-1, 0, +1}` under the shared map), and `PQ2_0`'s
/// 2-bit codes `{0b00, 0b01, 0b10}` decode to exactly that same set under the
/// identical `code - 1` arithmetic — so the trit code becomes the 2-bit code
/// unchanged (never `0b11`) and `d` is copied bit-for-bit.
///
/// Built directly on [`decode_ptq1_0_codes`] (this crate's own trit decode),
/// not on [`oxibonsai_core::BlockPTQ1_0::transcode_to_pq2`], so that
/// `dequant_ptq1_0(blocks) == dequant_pq2_0(transcode_ptq1_0_to_pq2_0(blocks))`
/// is a real test of this crate's own repacking logic, not a tautology.
///
/// # Errors
///
/// [`KernelError::NamedBufferTooSmall`] if `out` holds fewer blocks than
/// `blocks`.
pub fn transcode_ptq1_0_to_pq2_0(
    blocks: &[BlockPTQ1_0],
    out: &mut [BlockPQ2_0],
) -> KernelResult<()> {
    if out.len() < blocks.len() {
        return Err(KernelError::buffer_too_small(
            "out",
            blocks.len(),
            out.len(),
        ));
    }
    for (bi, block) in blocks.iter().enumerate() {
        let codes = decode_ptq1_0_codes(block);
        let mut qs = [0u8; 32];
        for (j, &code) in codes.iter().enumerate() {
            qs[j / 4] |= (code & 0b11) << ((j % 4) * 2);
        }
        out[bi] = BlockPQ2_0 { d: block.d, qs };
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_pq2_block(scale: f32, qs: [u8; 32]) -> BlockPQ2_0 {
        BlockPQ2_0 {
            d: f16::from_f32(scale),
            qs,
        }
    }

    fn make_g64_block(scale: f32, qs: [u8; 16]) -> BlockQ2_0G64 {
        BlockQ2_0G64 {
            d: f16::from_f32(scale),
            qs,
        }
    }

    fn make_ptq1_block(scale: f32, qs: [u8; 24], qh: [u8; 2]) -> BlockPTQ1_0 {
        BlockPTQ1_0 {
            qs,
            qh,
            d: f16::from_f32(scale),
        }
    }

    // ── PQ2_0 dequant ────────────────────────────────────────────────────

    /// qs lane pattern `00 01 10 11` (byte 0) must decode to `-d, 0, +d, +2d`
    /// — the `11 -> +2` arithmetic branch is the entire point of this format
    /// existing separately from the legacy ternary LUT.
    #[test]
    fn pq2_0_dequant_decodes_all_four_codes_including_plus_two() {
        let mut qs = [0x55u8; 32]; // rest: code 01 -> 0
        qs[0] = 0b11_10_01_00;
        let block = make_pq2_block(2.0, qs);
        let mut out = vec![0.0f32; QK_PQ2_0];
        dequant_pq2_0(&[block], &mut out).expect("dequant");
        assert_eq!(&out[0..4], &[-2.0, 0.0, 2.0, 4.0]);
        assert!(out[4..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn pq2_0_dequant_buffer_too_small() {
        let block = make_pq2_block(1.0, [0xAA; 32]);
        let mut out = vec![0.0f32; 0];
        assert!(dequant_pq2_0(&[block], &mut out).is_err());
    }

    // ── Q2_0_g64 dequant ─────────────────────────────────────────────────

    #[test]
    fn q2_0_g64_dequant_decodes_all_four_codes_including_plus_two() {
        let mut qs = [0x55u8; 16];
        qs[0] = 0b11_10_01_00;
        let block = make_g64_block(0.5, qs);
        let mut out = vec![0.0f32; QK_Q2_0_G64];
        dequant_q2_0_g64(&[block], &mut out).expect("dequant");
        assert_eq!(&out[0..4], &[-0.5, 0.0, 0.5, 1.0]);
        assert!(out[4..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q2_0_g64_dequant_buffer_too_small() {
        let block = make_g64_block(1.0, [0xAA; 16]);
        let mut out = vec![0.0f32; 0];
        assert!(dequant_q2_0_g64(&[block], &mut out).is_err());
    }

    // ── PQ2_0 GEMV / GEMM ────────────────────────────────────────────────

    /// All codes = 0b10 (+1), d = 1.0, input = all-ones -> row sum = 128.
    #[test]
    fn gemv_pq2_0_identity() {
        let blocks = vec![make_pq2_block(1.0, [0xAA; 32]); 2];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 2];
        gemv_pq2_0(&blocks, &input, &mut output, 2, 128).expect("gemv");
        for v in output {
            assert!((v - 128.0).abs() < 1e-3, "expected 128.0, got {v}");
        }
    }

    /// All codes = 0b11 (+2) must contribute double the 0b10 (+1) case —
    /// the canary that would catch an accidental ternary-LUT reuse (which
    /// maps 0b11 to 0, giving 0.0 here instead of 256.0).
    #[test]
    fn gemv_pq2_0_plus_two_code_is_not_zero() {
        let blocks = vec![make_pq2_block(1.0, [0xFF; 32])]; // every lane = 0b11
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        gemv_pq2_0(&blocks, &input, &mut output, 1, 128).expect("gemv");
        assert!(
            (output[0] - 256.0).abs() < 1e-3,
            "0b11 must decode to +2, expected 256.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemv_pq2_0_not_block_aligned() {
        let blocks = vec![make_pq2_block(1.0, [0xAA; 32])];
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(gemv_pq2_0(&blocks, &input, &mut output, 1, 100).is_err());
    }

    #[test]
    fn gemv_pq2_0_dimension_mismatch_reports_blocks_name() {
        let blocks = vec![make_pq2_block(1.0, [0xAA; 32])]; // only 1, need 2
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 2];
        let err = gemv_pq2_0(&blocks, &input, &mut output, 2, 128).expect_err("must fail");
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    #[test]
    fn gemm_pq2_0_matches_gemv_per_batch_row() {
        let blocks = vec![
            make_pq2_block(1.0, [0xAA; 32]),
            make_pq2_block(1.5, [0x00; 32]),
        ];
        let m = 3;
        let n_rows = 2;
        let k = 128;
        let mut input = vec![0.0f32; m * k];
        for (batch, chunk) in input.chunks_mut(k).enumerate() {
            chunk.fill((batch + 1) as f32 * 0.5);
        }
        let mut gemm_out = vec![0.0f32; m * n_rows];
        gemm_pq2_0(&blocks, &input, &mut gemm_out, m, n_rows, k).expect("gemm");

        for batch in 0..m {
            let input_row = &input[batch * k..(batch + 1) * k];
            let mut gemv_out = vec![0.0f32; n_rows];
            gemv_pq2_0(&blocks, input_row, &mut gemv_out, n_rows, k).expect("gemv");
            for row in 0..n_rows {
                assert!(
                    (gemm_out[batch * n_rows + row] - gemv_out[row]).abs() < 1e-4,
                    "batch={batch} row={row}"
                );
            }
        }
    }

    // ── Q2_0_g64 GEMV / GEMM ─────────────────────────────────────────────

    #[test]
    fn gemv_q2_0_g64_identity() {
        let blocks = vec![make_g64_block(1.0, [0xAA; 16])];
        let input = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 1];
        gemv_q2_0_g64(&blocks, &input, &mut output, 1, 64).expect("gemv");
        assert!((output[0] - 64.0).abs() < 1e-3);
    }

    #[test]
    fn gemv_q2_0_g64_not_block_aligned() {
        let blocks = vec![make_g64_block(1.0, [0xAA; 16])];
        let input = vec![1.0f32; 50];
        let mut output = vec![0.0f32; 1];
        assert!(gemv_q2_0_g64(&blocks, &input, &mut output, 1, 50).is_err());
    }

    #[test]
    fn gemm_q2_0_g64_matches_gemv_per_batch_row() {
        let blocks = vec![make_g64_block(1.0, [0xAA; 16]); 2];
        let m = 2;
        let n_rows = 2;
        let k = 64;
        let input = vec![1.0f32; m * k];
        let mut gemm_out = vec![0.0f32; m * n_rows];
        gemm_q2_0_g64(&blocks, &input, &mut gemm_out, m, n_rows, k).expect("gemm");
        for v in gemm_out {
            assert!((v - 64.0).abs() < 1e-3);
        }
    }

    // ── PTQ1_0 decode / dequant ──────────────────────────────────────────

    /// Every `qs`/`qh` byte encoding trit-word `1,1,1,1,1` (all-zero trits,
    /// base-3 word `121`) must decode to all-zero codes.
    #[test]
    fn ptq1_0_all_zero_trits_decode_to_code_one() {
        // ceil(121 * 256 / 243) = 128 -> byte value whose trits are all 1.
        let zero_byte = (((121u16) * 256).div_ceil(243)) as u8;
        let block = make_ptq1_block(1.0, [zero_byte; 24], [zero_byte; 2]);
        let codes = decode_ptq1_0_codes(&block);
        assert!(codes.iter().all(|&c| c == 1), "expected all-zero trits");
        let mut out = vec![0.0f32; QK_PTQ1_0];
        dequant_ptq1_0(&[block], &mut out).expect("dequant");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ptq1_0_stage_32_is_inert_effective_sequence_is_16_then_8() {
        let mut j = 0usize;
        let mut stages = Vec::new();
        for &c in PTQ1_0_STAGES.iter() {
            while j + c <= 24 {
                stages.push((j, c));
                j += c;
            }
        }
        assert_eq!(stages, vec![(0, 16), (16, 8)]);
    }

    #[test]
    fn ptq1_0_dequant_buffer_too_small() {
        let block = make_ptq1_block(1.0, [0u8; 24], [0u8; 2]);
        let mut out = vec![0.0f32; 0];
        assert!(dequant_ptq1_0(&[block], &mut out).is_err());
    }

    /// End-to-end round trip over pseudo-random raw bytes (NOT run through a
    /// quantizer, which can never emit every byte value): every distinct
    /// `qs`/`qh` byte value must decode to a code in `0..=2`, never wrapping
    /// out of range (the trap [`decode_ptq1_0_codes`]'s doc warns about).
    #[test]
    fn ptq1_0_decode_never_leaves_the_trit_range_over_many_byte_values() {
        let mut state = 0xC0FF_EEEEu32;
        for _ in 0..256 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let qs: [u8; 24] = core::array::from_fn(|i| {
                ((state >> (i % 24)) ^ (state.rotate_left(i as u32))) as u8
            });
            let qh: [u8; 2] = [qs[0].wrapping_add(1), qs[1].wrapping_add(2)];
            let block = make_ptq1_block(1.0, qs, qh);
            let codes = decode_ptq1_0_codes(&block);
            assert!(
                codes.iter().all(|&c| c <= 2),
                "code out of trit range for state {state}"
            );
        }
    }

    // ── transcode ────────────────────────────────────────────────────────

    #[test]
    fn transcode_ptq1_0_to_pq2_0_is_lossless() {
        let mut input = vec![0.0f32; 256];
        let mut state = 0x1234_5678u32;
        for v in input.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = match (state >> 16) % 3 {
                0 => -0.75,
                1 => 0.0,
                _ => 0.75,
            };
        }
        let ptq = oxibonsai_core::BlockPTQ1_0::quantize(&input).expect("quantize");

        let mut pq = vec![BlockPQ2_0::zeroed(); ptq.len()];
        transcode_ptq1_0_to_pq2_0(&ptq, &mut pq).expect("transcode");

        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        dequant_ptq1_0(&ptq, &mut a).expect("dequant ptq");
        dequant_pq2_0(&pq, &mut b).expect("dequant pq");
        assert_eq!(a, b, "transcode must be lossless (bitwise)");
        for block in &pq {
            assert_eq!(block.count_plus_two(), 0, "ternary source cannot emit 0b11");
        }
    }

    #[test]
    fn transcode_ptq1_0_to_pq2_0_rejects_short_output() {
        let block = make_ptq1_block(1.0, [0u8; 24], [0u8; 2]);
        let mut out: Vec<BlockPQ2_0> = Vec::new();
        assert!(transcode_ptq1_0_to_pq2_0(&[block], &mut out).is_err());
    }
}
