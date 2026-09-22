//! PrismML Bonsai 2 quantization block types: `PQ2_0`, `PTQ1_0`, `Q2_0_g64`.
//!
//! These are the three block layouts the PrismML llama.cpp fork defines in
//! `ggml-common.h`, transcribed byte-for-byte:
//!
//! | type | ggml id | group | bytes | layout |
//! |---|---|---|---|---|
//! | [`BlockPQ2_0`] | 142 | 128 | 34 | `d: f16` **first**, then `qs[32]` |
//! | [`BlockPTQ1_0`] | 143 | 128 | 28 | `qs[24]`, `qh[2]`, `d: f16` **last** |
//! | [`BlockQ2_0G64`] | 42 (mainline) | 64 | 18 | `d: f16` **first**, then `qs[16]` |
//!
//! The 2-bit family (`PQ2_0`, `Q2_0_g64`) uses an **arithmetic** decode,
//! `y = ((code as i32) - 1) * d` — codes `00 → -1`, `01 → 0`, `10 → +1`,
//! `11 → +2` (`ggml-quants.c:474-512`). This is *not* the same map as the
//! legacy [`crate::quant_ternary::BlockTQ2_0_g128`], which is a three-level
//! ternary LUT with a different byte order, so the two must never share a
//! block type even though both are 34 bytes for 128 weights.
//!
//! `PTQ1_0` packs five base-3 trits per byte plus four per `qh` byte. The
//! element order inside a block is **interleaved**, not sequential — see
//! [`BlockPTQ1_0::decode_codes`] for the exact derivation.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};
use crate::quant_ternary::BlockTQ2_0_g128;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Weights per `PQ2_0` block (`QK_PQ2_0`).
pub const QK_PQ2_0: usize = 128;
/// Bytes per `PQ2_0` block.
pub const BLOCK_PQ2_0_BYTES: usize = 34;

/// Weights per `PTQ1_0` block (`QK_PTQ1_0`).
pub const QK_PTQ1_0: usize = 128;
/// Bytes per `PTQ1_0` block.
pub const BLOCK_PTQ1_0_BYTES: usize = 28;

/// Weights per mainline `Q2_0` block (`QK2_0`).
pub const QK_Q2_0_G64: usize = 64;
/// Bytes per mainline `Q2_0` block.
pub const BLOCK_Q2_0_G64_BYTES: usize = 18;

/// `PTQ1_0` stage widths (`ptq1_0_stages`, `ggml-quants.c:2203`).
///
/// At `sizeof(qs) == 24` the first stage is **inert** (`j + 32 <= 24` is never
/// true), so the effective sequence is `{16, 8}` with `j` carrying across.
/// The 32 is kept so this table is literally the reference's.
pub const PTQ1_0_STAGES: [usize; 3] = [32, 16, 8];

/// Powers of three used by the base-3 trit codec, as `u8` (the reference
/// multiplies in `uint8_t` and relies on the wrap).
pub const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

// ---------------------------------------------------------------------------
// Shared 2-bit helpers
// ---------------------------------------------------------------------------

/// Extract the 2-bit code for element `j` from a packed `qs` array.
#[inline]
pub fn two_bit_code(qs: &[u8], j: usize) -> u8 {
    let byte_index = j / 4;
    let bit_offset = (j % 4) * 2;
    (qs[byte_index] >> bit_offset) & 0x03
}

/// Arithmetic decode shared by the whole `Q2_0` family.
///
/// `00 → -1`, `01 → 0`, `10 → +1`, `11 → +2` (`ggml-quants.c:487`).
#[inline]
pub const fn q2_0_code_to_i32(code: u8) -> i32 {
    (code & 0x03) as i32 - 1
}

/// Quantize one 2-bit block body in place.
///
/// Mirrors `quantize_row_q2_0_ref` / `quantize_row_pq2_0_ref`: `d = amax`,
/// `q = round(w / d) + 1` clamped to `0..=3` (the clamp can never fire for
/// finite input, because `|w| <= amax` forces `round(w/d) ∈ {-1, 0, 1}`).
fn quantize_two_bit_body(chunk: &[f32], qs: &mut [u8]) -> f16 {
    let mut amax = 0.0f32;
    for &v in chunk {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };
    for b in qs.iter_mut() {
        *b = 0;
    }
    for (j, &w) in chunk.iter().enumerate() {
        let q = ((w * id).round() as i32 + 1).clamp(0, 3) as u8;
        let byte_index = j / 4;
        let bit_offset = (j % 4) * 2;
        qs[byte_index] |= q << bit_offset;
    }
    f16::from_f32(amax)
}

/// Count 2-bit lanes that decode to the reserved `0b11` (`+2`) code.
///
/// PrismML checkpoints are ternary, so a non-zero count on a tensor declared
/// ternary means the assumed byte order is wrong — this is the data-level
/// discriminator the id-42 resolver depends on.
#[inline]
pub fn count_plus_two_codes(qs: &[u8]) -> u32 {
    let mut n = 0u32;
    for &b in qs {
        for shift in [0u32, 2, 4, 6] {
            if (b >> shift) & 0x03 == 0x03 {
                n += 1;
            }
        }
    }
    n
}

/// Shared `slice_from_bytes` guard: alignment plus a length rule.
fn cast_blocks<'a, T>(
    data: &'a [u8],
    expected_len: usize,
    format: &'static str,
) -> BonsaiResult<&'a [T]> {
    if data.len() != expected_len {
        return Err(BonsaiError::InvalidQuantBlockSize {
            format,
            expected: expected_len,
            actual: data.len(),
        });
    }
    let align = std::mem::align_of::<T>();
    if data.as_ptr().align_offset(align) != 0 {
        return Err(BonsaiError::AlignmentError {
            expected: align,
            offset: data.as_ptr() as u64,
        });
    }
    let count = data.len() / std::mem::size_of::<T>();
    let ptr = data.as_ptr() as *const T;
    // SAFETY: `T` is `#[repr(C)]` with a compile-time-asserted size, the byte
    // length is an exact multiple of that size (checked above), the pointer is
    // correctly aligned (checked above), every bit pattern of the fields
    // (`u8`/`f16`) is valid, and the returned lifetime is tied to `data`.
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
}

// ---------------------------------------------------------------------------
// BlockPQ2_0 (ggml id 142)
// ---------------------------------------------------------------------------

/// PrismML `PQ2_0` block — ggml id 142, 128 weights in 34 bytes, `d` FIRST.
///
/// Mirrors `block_pq2_0` (`ggml-common.h:202-207`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub struct BlockPQ2_0 {
    /// Block scale (FP16), stored **before** the codes.
    pub d: f16,
    /// 128 × 2-bit codes, 4 per byte, LSB-first.
    pub qs: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<BlockPQ2_0>() == BLOCK_PQ2_0_BYTES);
const _: () = assert!(std::mem::align_of::<BlockPQ2_0>() == 2);

impl BlockPQ2_0 {
    /// A block whose codes all decode to zero.
    pub const fn zeroed() -> Self {
        // code 01 = 0 → 0b01_01_01_01 = 0x55
        Self {
            d: f16::ZERO,
            qs: [0x55; 32],
        }
    }

    /// Decode the 2-bit code of element `j` (`0..128`).
    #[inline]
    pub fn code(&self, j: usize) -> u8 {
        two_bit_code(&self.qs, j)
    }

    /// Dequantize `blocks` into `output` (`>= blocks.len() * 128` floats).
    ///
    /// `y[j] = ((code - 1) as f32) * d` — `dequantize_row_pq2_0`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        dequant_two_bit(blocks.len(), QK_PQ2_0, output, "PQ2_0", |i, j| {
            let b = &blocks[i];
            (q2_0_code_to_i32(two_bit_code(&b.qs, j)) as f32) * b.d.to_f32()
        })
    }

    /// Quantize `input` (a multiple of 128 floats) into `PQ2_0` blocks.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        require_multiple(input.len(), QK_PQ2_0, "PQ2_0")?;
        let mut blocks = Vec::with_capacity(input.len() / QK_PQ2_0);
        for chunk in input.chunks_exact(QK_PQ2_0) {
            let mut qs = [0u8; 32];
            let d = quantize_two_bit_body(chunk, &mut qs);
            blocks.push(Self { d, qs });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice whose length is a multiple of 34.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        let expected = data.len() / BLOCK_PQ2_0_BYTES * BLOCK_PQ2_0_BYTES;
        cast_blocks(data, expected, "PQ2_0")
    }

    /// Zero-copy cast of a byte slice that must hold exactly `n_blocks` blocks.
    ///
    /// The exact-length rule is the safety net for the ambiguous ggml id 42:
    /// a 34-byte tensor read as an 18-byte type (or vice versa) fails loudly
    /// here instead of decoding to noise.
    pub fn slice_from_bytes_exact(data: &[u8], n_blocks: usize) -> BonsaiResult<&[Self]> {
        cast_blocks(data, n_blocks * BLOCK_PQ2_0_BYTES, "PQ2_0")
    }

    /// Number of lanes in this block that carry the reserved `+2` code.
    pub fn count_plus_two(&self) -> u32 {
        count_plus_two_codes(&self.qs)
    }
}

// ---------------------------------------------------------------------------
// BlockQ2_0G64 (mainline ggml id 42)
// ---------------------------------------------------------------------------

/// Mainline `Q2_0` block — ggml id 42, 64 weights in 18 bytes, `d` FIRST.
///
/// Mirrors `block_q2_0` (`ggml-common.h:192-197`). Same codec as
/// [`BlockPQ2_0`] at half the group size.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub struct BlockQ2_0G64 {
    /// Block scale (FP16), stored **before** the codes.
    pub d: f16,
    /// 64 × 2-bit codes, 4 per byte, LSB-first.
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ2_0G64>() == BLOCK_Q2_0_G64_BYTES);
const _: () = assert!(std::mem::align_of::<BlockQ2_0G64>() == 2);

impl BlockQ2_0G64 {
    /// A block whose codes all decode to zero.
    pub const fn zeroed() -> Self {
        Self {
            d: f16::ZERO,
            qs: [0x55; 16],
        }
    }

    /// Decode the 2-bit code of element `j` (`0..64`).
    #[inline]
    pub fn code(&self, j: usize) -> u8 {
        two_bit_code(&self.qs, j)
    }

    /// Dequantize `blocks` into `output` (`>= blocks.len() * 64` floats).
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        dequant_two_bit(blocks.len(), QK_Q2_0_G64, output, "Q2_0_g64", |i, j| {
            let b = &blocks[i];
            (q2_0_code_to_i32(two_bit_code(&b.qs, j)) as f32) * b.d.to_f32()
        })
    }

    /// Quantize `input` (a multiple of 64 floats) into `Q2_0_g64` blocks.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        require_multiple(input.len(), QK_Q2_0_G64, "Q2_0_g64")?;
        let mut blocks = Vec::with_capacity(input.len() / QK_Q2_0_G64);
        for chunk in input.chunks_exact(QK_Q2_0_G64) {
            let mut qs = [0u8; 16];
            let d = quantize_two_bit_body(chunk, &mut qs);
            blocks.push(Self { d, qs });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice whose length is a multiple of 18.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        let expected = data.len() / BLOCK_Q2_0_G64_BYTES * BLOCK_Q2_0_G64_BYTES;
        cast_blocks(data, expected, "Q2_0_g64")
    }

    /// Zero-copy cast of a byte slice that must hold exactly `n_blocks` blocks.
    pub fn slice_from_bytes_exact(data: &[u8], n_blocks: usize) -> BonsaiResult<&[Self]> {
        cast_blocks(data, n_blocks * BLOCK_Q2_0_G64_BYTES, "Q2_0_g64")
    }

    /// Number of lanes in this block that carry the reserved `+2` code.
    pub fn count_plus_two(&self) -> u32 {
        count_plus_two_codes(&self.qs)
    }
}

// ---------------------------------------------------------------------------
// BlockPTQ1_0 (ggml id 143)
// ---------------------------------------------------------------------------

/// PrismML `PTQ1_0` block — ggml id 143, 128 weights in 28 bytes, `d` LAST.
///
/// Mirrors `block_ptq1_0` (`ggml-common.h:214-220`): five base-3 trits per
/// `qs` byte (24 bytes → 120 values) plus four per `qh` byte (2 bytes → 8
/// values).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub struct BlockPTQ1_0 {
    /// 24 bytes, 5 trits each → 120 values.
    pub qs: [u8; 24],
    /// 2 bytes, 4 trits each → 8 values.
    pub qh: [u8; 2],
    /// Block scale (FP16), stored **after** the codes.
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockPTQ1_0>() == BLOCK_PTQ1_0_BYTES);
const _: () = assert!(std::mem::align_of::<BlockPTQ1_0>() == 2);

impl BlockPTQ1_0 {
    /// Powers of three used by the trit codec (alias of [`POW3`]).
    pub const POW3: [u8; 6] = POW3;

    /// A block whose codes all decode to zero.
    pub fn zeroed() -> Self {
        // Trit code 1 means zero. Five such trits give the base-3 word
        // 3^4 + 3^3 + 3^2 + 3 + 1 = 121; four of them plus the extra `q *= 3`
        // the `qh` path applies give (((3 + 1) * 3 + 1) * 3 + 1) * 3 = 120.
        Self {
            qs: [ceil_div_243(121); 24],
            qh: [ceil_div_243(120); 2],
            d: f16::ZERO,
        }
    }

    /// Decode the 128 trit codes (`0`, `1`, `2`) of this block.
    ///
    /// Transcribed from `dequantize_row_ptq1_0` (`ggml-quants.c:2255-2285`).
    /// Three traps, all reproduced here:
    ///
    /// 1. Stage `32` of [`PTQ1_0_STAGES`] is **inert** at `sizeof(qs) == 24`
    ///    (`j + 32 <= 24` is never true), so the effective sequence is
    ///    `{16, 8}` with `j` carrying across stages.
    /// 2. The element order is **interleaved**: the loops are `n` outer /
    ///    `m` inner, so byte `qs[j + m]` supplies output positions
    ///    `n * c + m`, *not* `5 * (j + m) .. 5 * (j + m) + 5`.
    /// 3. `qs[j + m] * pow3[n]` is `uint8_t` arithmetic and **wraps mod 256**
    ///    — [`u8::wrapping_mul`], never a widened multiply.
    ///
    /// Resulting index map:
    ///
    /// ```text
    /// e[ n*16 + m ]     n in 0..5, m in 0..16  ->  trit n of qs[m]        (e[0..80))
    /// e[ 80 + n*8 + m ] n in 0..5, m in 0..8   ->  trit n of qs[16 + m]   (e[80..120))
    /// e[ 120 + n*2 + h] n in 0..4, h in 0..2   ->  trit n of qh[h]        (e[120..128))
    /// ```
    #[inline]
    pub fn decode_codes(&self, codes: &mut [u8; QK_PTQ1_0]) {
        let mut w = 0usize;
        let mut j = 0usize;
        for &c in PTQ1_0_STAGES.iter() {
            while j + c <= self.qs.len() {
                // `n` outer, `m` inner: byte `qs[j + m]` supplies output
                // position `n * c + m`.
                for &pow in POW3.iter().take(5) {
                    for m in 0..c {
                        codes[w] = trit_of(self.qs[j + m], pow);
                        w += 1;
                    }
                }
                j += c;
            }
        }
        for &pow in POW3.iter().take(4) {
            for &byte in self.qh.iter() {
                codes[w] = trit_of(byte, pow);
                w += 1;
            }
        }
        debug_assert_eq!(w, QK_PTQ1_0);
    }

    /// Dequantize `blocks` into `output` (`>= blocks.len() * 128` floats).
    ///
    /// `y = (code - 1) as f32 * d`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected = blocks.len().saturating_mul(QK_PTQ1_0);
        require_output(output.len(), expected, "PTQ1_0")?;
        let mut codes = [0u8; QK_PTQ1_0];
        for (i, block) in blocks.iter().enumerate() {
            block.decode_codes(&mut codes);
            let d = block.d.to_f32();
            let base = i * QK_PTQ1_0;
            for (j, &c) in codes.iter().enumerate() {
                output[base + j] = (c as i32 - 1) as f32 * d;
            }
        }
        Ok(())
    }

    /// Quantize `input` (a multiple of 128 floats) into `PTQ1_0` blocks.
    ///
    /// Transcribed from `quantize_row_ptq1_0_ref` (`ggml-quants.c:2205-2253`),
    /// including the source-pointer walk (`x += 5 * c` **inside** the stage
    /// loop, `x += 4 * size_of::<qh>()` after it), the extra `q *= 3` that
    /// shifts `qh`'s first value into the most significant trit, and the
    /// ceiling division `q = (q * 256 + 242) / 243`.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        require_multiple(input.len(), QK_PTQ1_0, "PTQ1_0")?;
        let mut blocks = Vec::with_capacity(input.len() / QK_PTQ1_0);
        for chunk in input.chunks_exact(QK_PTQ1_0) {
            let mut amax = 0.0f32;
            for &v in chunk {
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
            }
            let id = if amax > 0.0 { 1.0 / amax } else { 0.0 };

            let mut qs = [0u8; 24];
            // Mirrors the reference's moving `x` pointer.
            let mut base = 0usize;
            let mut j = 0usize;
            for &c in PTQ1_0_STAGES.iter() {
                while j + c <= qs.len() {
                    for m in 0..c {
                        let mut q: u8 = 0;
                        for n in 0..5usize {
                            let xi = trit_code(chunk[base + m + n * c], id);
                            q = q.wrapping_mul(3).wrapping_add(xi);
                        }
                        qs[j + m] = ceil_div_243(q);
                    }
                    base += 5 * c;
                    j += c;
                }
            }

            let mut qh = [0u8; 2];
            for h in 0..qh.len() {
                let mut q: u8 = 0;
                for m in 0..4usize {
                    let xi = trit_code(chunk[base + h + m * qh.len()], id);
                    q = q.wrapping_mul(3).wrapping_add(xi);
                }
                // Shift the first value into the most significant trit.
                q = q.wrapping_mul(3);
                qh[h] = ceil_div_243(q);
            }

            blocks.push(Self {
                qs,
                qh,
                d: f16::from_f32(amax),
            });
        }
        Ok(blocks)
    }

    /// Losslessly re-encode into the `PQ2_0` wire form (`d` first, 34 B).
    ///
    /// Lossless by construction: `PTQ1_0` only represents `{-1, 0, +1}`
    /// (`xi ∈ {0, 1, 2}`) and `PQ2_0`'s codes `{00, 01, 10}` decode to exactly
    /// that set under the same `code - 1` arithmetic, so the trit code is the
    /// 2-bit code unchanged and `d` is copied bit-for-bit.
    pub fn transcode_to_pq2(blocks: &[Self], out: &mut [BlockPQ2_0]) -> BonsaiResult<()> {
        if out.len() < blocks.len() {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "PTQ1_0 transcode: output holds {} blocks, need {}",
                    out.len(),
                    blocks.len()
                ),
            });
        }
        let mut codes = [0u8; QK_PTQ1_0];
        for (i, block) in blocks.iter().enumerate() {
            block.decode_codes(&mut codes);
            let mut qs = [0u8; 32];
            for (j, &c) in codes.iter().enumerate() {
                qs[j / 4] |= (c & 0x03) << ((j % 4) * 2);
            }
            out[i] = BlockPQ2_0 { d: block.d, qs };
        }
        Ok(())
    }

    /// Zero-copy cast of a byte slice whose length is a multiple of 28.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        let expected = data.len() / BLOCK_PTQ1_0_BYTES * BLOCK_PTQ1_0_BYTES;
        cast_blocks(data, expected, "PTQ1_0")
    }

    /// Zero-copy cast of a byte slice that must hold exactly `n_blocks` blocks.
    pub fn slice_from_bytes_exact(data: &[u8], n_blocks: usize) -> BonsaiResult<&[Self]> {
        cast_blocks(data, n_blocks * BLOCK_PTQ1_0_BYTES, "PTQ1_0")
    }
}

/// Losslessly re-encode `PTQ1_0` blocks into the legacy `TQ2_0_g128` wire form
/// (`qs` first, `d` last, 34 B) used by the existing ternary GEMV/GEMM stack.
///
/// `TQ2_0_g128`'s three-level LUT (`00 → -1`, `01 → 0`, `10 → +1`) covers
/// exactly the values `PTQ1_0` can represent, so the trit code transfers
/// unchanged and `d` is copied bit-for-bit — the transcode is lossless.
pub fn transcode_ptq1_0_to_tq2(blocks: &[BlockPTQ1_0]) -> Vec<BlockTQ2_0_g128> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut codes = [0u8; QK_PTQ1_0];
    for block in blocks {
        block.decode_codes(&mut codes);
        let mut qs = [0u8; 32];
        for (j, &c) in codes.iter().enumerate() {
            qs[j / 4] |= (c & 0x03) << ((j % 4) * 2);
        }
        out.push(BlockTQ2_0_g128 { qs, d: block.d });
    }
    out
}

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/// `int xi = lroundf(x * id) + 1` — the reference's trit code.
///
/// `|x| <= amax` and `id == 1 / amax`, so the rounded value is always in
/// `{-1, 0, 1}` and the code is in `{0, 1, 2}`; the clamp only guards against
/// a non-finite input reaching an `as u8` cast.
#[inline]
fn trit_code(x: f32, id: f32) -> u8 {
    (((x * id).round() as i32) + 1).clamp(0, 2) as u8
}

/// `q = ((uint16_t)q * 256 + 242) / 243` — ceiling division by `3^5`.
#[inline]
const fn ceil_div_243(q: u8) -> u8 {
    ((q as u16) * 256).div_ceil(243) as u8
}

/// Extract one base-3 trit: `uint8_t q = byte * pow3[n]; ((uint16_t)q * 3) >> 8`.
///
/// The multiply is `uint8_t` arithmetic in the reference and **wraps**, so
/// [`u8::wrapping_mul`] is load-bearing — a widened multiply leaves the
/// `0..=2` trit range entirely.
#[inline]
const fn trit_of(byte: u8, pow: u8) -> u8 {
    let q = byte.wrapping_mul(pow);
    (((q as u16) * 3) >> 8) as u8
}

fn require_multiple(len: usize, group: usize, format: &'static str) -> BonsaiResult<()> {
    if !len.is_multiple_of(group) {
        return Err(BonsaiError::KQuantError {
            reason: format!("{format} quantize: input length {len} is not a multiple of {group}"),
        });
    }
    Ok(())
}

fn require_output(actual: usize, expected: usize, format: &'static str) -> BonsaiResult<()> {
    if actual < expected {
        return Err(BonsaiError::KQuantError {
            reason: format!("{format} dequant: output holds {actual} floats, need {expected}"),
        });
    }
    Ok(())
}

fn dequant_two_bit<F>(
    n_blocks: usize,
    group: usize,
    output: &mut [f32],
    format: &'static str,
    value: F,
) -> BonsaiResult<()>
where
    F: Fn(usize, usize) -> f32,
{
    let expected = n_blocks.saturating_mul(group);
    require_output(output.len(), expected, format)?;
    for i in 0..n_blocks {
        let base = i * group;
        for j in 0..group {
            output[base + j] = value(i, j);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Layout ────────────────────────────────────────────────────────────

    #[test]
    fn block_sizes_match_ggml_common_h() {
        assert_eq!(std::mem::size_of::<BlockPQ2_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ2_0G64>(), 18);
        assert_eq!(std::mem::size_of::<BlockPTQ1_0>(), 28);
    }

    /// `PQ2_0` puts `d` first, `PTQ1_0` puts it last — the two new formats
    /// have OPPOSITE field orders, so one shared decoder shape would be wrong
    /// for one of them.
    #[test]
    fn field_offsets_place_the_scale_where_ggml_does() {
        let pq = BlockPQ2_0 {
            d: f16::from_f32(1.5),
            qs: [0xAB; 32],
        };
        let bytes: [u8; 34] = unsafe { std::mem::transmute(pq) };
        assert_eq!(
            u16::from_le_bytes([bytes[0], bytes[1]]),
            f16::from_f32(1.5).to_bits(),
            "PQ2_0 scale must occupy bytes 0..2"
        );
        assert_eq!(bytes[2], 0xAB);

        let pt = BlockPTQ1_0 {
            qs: [0x11; 24],
            qh: [0x22; 2],
            d: f16::from_f32(1.5),
        };
        let bytes: [u8; 28] = unsafe { std::mem::transmute(pt) };
        assert_eq!(bytes[0], 0x11);
        assert_eq!(bytes[24], 0x22);
        assert_eq!(
            u16::from_le_bytes([bytes[26], bytes[27]]),
            f16::from_f32(1.5).to_bits(),
            "PTQ1_0 scale must occupy the LAST two bytes"
        );
    }

    // ── 2-bit decode ──────────────────────────────────────────────────────

    #[test]
    fn pq2_0_decode_is_arithmetic_including_plus_two() {
        // lanes 0..4 of qs[0] = 00, 01, 10, 11 → -1, 0, +1, +2
        let mut qs = [0x55u8; 32];
        qs[0] = 0b11_10_01_00;
        let block = BlockPQ2_0 {
            d: f16::from_f32(2.0),
            qs,
        };
        let mut out = vec![0.0f32; 128];
        BlockPQ2_0::dequant(&[block], &mut out).expect("dequant");
        assert_eq!(&out[0..4], &[-2.0, 0.0, 2.0, 4.0]);
        assert_eq!(block.count_plus_two(), 1);
    }

    #[test]
    fn q2_0_g64_decode_matches_pq2_0_codec() {
        let mut qs = [0x55u8; 16];
        qs[0] = 0b11_10_01_00;
        let block = BlockQ2_0G64 {
            d: f16::from_f32(0.5),
            qs,
        };
        let mut out = vec![0.0f32; 64];
        BlockQ2_0G64::dequant(&[block], &mut out).expect("dequant");
        assert_eq!(&out[0..4], &[-0.5, 0.0, 0.5, 1.0]);
        assert_eq!(out[4], 0.0);
    }

    #[test]
    fn two_bit_quantize_roundtrip_is_exact_on_ternary_input() {
        let mut input = vec![0.0f32; 256];
        for (i, v) in input.iter_mut().enumerate() {
            *v = match i % 3 {
                0 => -0.25,
                1 => 0.0,
                _ => 0.25,
            };
        }
        let blocks = BlockPQ2_0::quantize(&input).expect("quantize");
        assert_eq!(blocks.len(), 2);
        let mut out = vec![0.0f32; 256];
        BlockPQ2_0::dequant(&blocks, &mut out).expect("dequant");
        assert_eq!(out, input, "ternary input must round-trip bit-exactly");
        for b in &blocks {
            assert_eq!(b.count_plus_two(), 0, "encoder must never emit 0b11");
        }
    }

    #[test]
    fn all_zero_block_quantizes_to_zero_scale() {
        let input = vec![0.0f32; 128];
        let blocks = BlockPQ2_0::quantize(&input).expect("quantize");
        assert_eq!(blocks[0].d, f16::ZERO);
        let mut out = vec![9.0f32; 128];
        BlockPQ2_0::dequant(&blocks, &mut out).expect("dequant");
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ── PTQ1_0 ────────────────────────────────────────────────────────────

    /// The first stage of `{32, 16, 8}` must never run at `sizeof(qs) == 24`.
    #[test]
    fn ptq1_0_stage_32_is_inert() {
        let mut j = 0usize;
        let mut executed = Vec::new();
        for &c in PTQ1_0_STAGES.iter() {
            while j + c <= 24 {
                executed.push((j, c));
                j += c;
            }
        }
        assert_eq!(
            executed,
            vec![(0, 16), (16, 8)],
            "effective stage sequence must be {{16, 8}} with j carrying"
        );
    }

    /// Byte `qs[m]` must supply output positions `n*c + m`, not `5*m + n`.
    #[test]
    fn ptq1_0_element_order_is_interleaved() {
        // A single non-zero trit: set qs[3] so its trit 0 is 2 (+1) and the
        // rest are 1 (0). Base-3 word with t0=2, t1..t4=1 → 2*81+27+9+3+1 = 202.
        let mut block = BlockPTQ1_0::zeroed();
        block.qs[3] = ceil_div_243(202);
        block.d = f16::from_f32(1.0);
        let mut codes = [0u8; 128];
        block.decode_codes(&mut codes);
        assert_eq!(codes[3], 2, "trit 0 of qs[3] must land at output index 3");
        for (i, &c) in codes.iter().enumerate() {
            if i == 3 {
                continue;
            }
            assert_eq!(c, 1, "index {i} must stay at code 1 (zero)");
        }
        // Trit 1 of qs[3] lands at 1*16+3 = 19, not 5*3+1 = 16.
        let mut block = BlockPTQ1_0::zeroed();
        // t0=1, t1=2, t2..t4=1 → 81 + 2*27 + 9 + 3 + 1 = 148
        block.qs[3] = ceil_div_243(148);
        block.decode_codes(&mut codes);
        assert_eq!(codes[19], 2, "trit 1 of qs[3] must land at index 1*16+3");
        assert_eq!(codes[16], 1);
    }

    /// `qs[j+m] * pow3[n]` is `uint8_t` arithmetic and **wraps mod 256**.
    /// With `pow3[4] = 81`, byte 200 gives `200 * 81 = 16200`, which
    /// truncates to `72`; a widened multiply keeps 16200 and the extracted
    /// "trit" leaves the `0..=2` range entirely.
    #[test]
    fn ptq1_0_trit_extraction_wraps_in_u8() {
        let b = 200u8;
        let p = POW3[4];
        assert_eq!(p, 81);
        let wrapped = b.wrapping_mul(p);
        assert_eq!(wrapped, 72, "uint8_t truncation of 200 * 81");
        let trit = ((wrapped as u16) * 3) >> 8;
        assert!(trit <= 2, "the wrapped form stays a trit: {trit}");

        let widened = ((b as u16) * (p as u16) * 3) >> 8;
        assert!(
            widened > 2,
            "a widened multiply leaves the trit range ({widened}) — that is the trap"
        );
    }

    /// End-to-end guard for the same trap: a decoder that widened the
    /// multiply would corrupt most blocks, so a pseudo-random ternary
    /// round-trip over many distinct `qs` byte values must be exact.
    #[test]
    fn ptq1_0_roundtrip_exercises_many_distinct_qs_bytes() {
        let mut input = vec![0.0f32; 128 * 64];
        let mut state = 0x9E37_79B9u32;
        for v in input.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = match (state >> 13) % 3 {
                0 => -2.0,
                1 => 0.0,
                _ => 2.0,
            };
        }
        let blocks = BlockPTQ1_0::quantize(&input).expect("quantize");
        let distinct: std::collections::BTreeSet<u8> =
            blocks.iter().flat_map(|b| b.qs.iter().copied()).collect();
        assert!(
            distinct.len() > 60,
            "fixture must exercise many qs byte values, saw {}",
            distinct.len()
        );
        let mut out = vec![0.0f32; input.len()];
        BlockPTQ1_0::dequant(&blocks, &mut out).expect("dequant");
        assert_eq!(out, input);
    }

    #[test]
    fn ptq1_0_quantize_dequant_roundtrip_is_exact_on_ternary_input() {
        let mut input = vec![0.0f32; 256];
        for (i, v) in input.iter_mut().enumerate() {
            *v = match (i * 7) % 3 {
                0 => -1.5,
                1 => 0.0,
                _ => 1.5,
            };
        }
        let blocks = BlockPTQ1_0::quantize(&input).expect("quantize");
        assert_eq!(blocks.len(), 2);
        let mut out = vec![0.0f32; 256];
        BlockPTQ1_0::dequant(&blocks, &mut out).expect("dequant");
        assert_eq!(out, input, "ternary input must round-trip bit-exactly");
    }

    #[test]
    fn ptq1_0_zeroed_decodes_to_all_zero() {
        let block = BlockPTQ1_0::zeroed();
        let mut codes = [0u8; 128];
        block.decode_codes(&mut codes);
        assert!(
            codes.iter().all(|&c| c == 1),
            "zeroed() must decode to the all-zero trit code"
        );
    }

    #[test]
    fn transcode_preserves_every_value_bitwise() {
        let mut input = vec![0.0f32; 512];
        let mut state = 0x1234_5678u32;
        for v in input.iter_mut() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = match (state >> 16) % 3 {
                0 => -0.75,
                1 => 0.0,
                _ => 0.75,
            };
        }
        let ptq = BlockPTQ1_0::quantize(&input).expect("quantize");

        let mut pq = vec![BlockPQ2_0::zeroed(); ptq.len()];
        BlockPTQ1_0::transcode_to_pq2(&ptq, &mut pq).expect("transcode");

        let mut a = vec![0.0f32; 512];
        let mut b = vec![0.0f32; 512];
        BlockPTQ1_0::dequant(&ptq, &mut a).expect("dequant ptq");
        BlockPQ2_0::dequant(&pq, &mut b).expect("dequant pq");
        assert_eq!(a, b, "transcode to PQ2_0 must be lossless");

        let tq = transcode_ptq1_0_to_tq2(&ptq);
        let mut c = vec![0.0f32; 512];
        BlockTQ2_0_g128::dequant(&tq, &mut c).expect("dequant tq");
        assert_eq!(a, c, "transcode to TQ2_0_g128 must be lossless");
        for blk in &pq {
            assert_eq!(blk.count_plus_two(), 0, "ternary source cannot emit 0b11");
        }
    }

    // ── slice_from_bytes guards ───────────────────────────────────────────

    #[test]
    fn slice_from_bytes_rejects_wrong_claimed_block_count() {
        // 34 bytes of a PQ2_0 tensor claimed as 2 blocks (needs 68).
        let buf = vec![0u8; 34];
        let err = BlockPQ2_0::slice_from_bytes_exact(&buf, 2).expect_err("must reject");
        match err {
            BonsaiError::InvalidQuantBlockSize {
                format,
                expected,
                actual,
            } => {
                assert_eq!(format, "PQ2_0");
                assert_eq!(expected, 68);
                assert_eq!(actual, 34);
            }
            other => panic!("expected InvalidQuantBlockSize, got {other:?}"),
        }
        // …and the g64 reading of the same bytes is refused too.
        assert!(BlockQ2_0G64::slice_from_bytes_exact(&buf, 2).is_err());
        assert!(BlockPTQ1_0::slice_from_bytes_exact(&buf, 1).is_err());
    }

    #[test]
    fn slice_from_bytes_rejects_non_multiple_length() {
        let buf = vec![0u8; 35];
        assert!(BlockPQ2_0::slice_from_bytes(&buf).is_err());
        let buf = vec![0u8; 19];
        assert!(BlockQ2_0G64::slice_from_bytes(&buf).is_err());
        let buf = vec![0u8; 29];
        assert!(BlockPTQ1_0::slice_from_bytes(&buf).is_err());
    }

    #[test]
    fn slice_from_bytes_rejects_misaligned_pointer() {
        let backing = [0u8; 69];
        let misaligned = &backing[1..69]; // 68 bytes at an odd address
        if misaligned.as_ptr().align_offset(2) == 0 {
            // The allocator happened to hand back an odd base; skip rather
            // than assert the wrong thing.
            return;
        }
        let err = BlockPQ2_0::slice_from_bytes(misaligned).expect_err("must reject");
        assert!(matches!(
            err,
            BonsaiError::AlignmentError { expected: 2, .. }
        ));
    }

    #[test]
    fn slice_from_bytes_accepts_well_formed_buffer() {
        let blocks = BlockPQ2_0::quantize(&vec![0.5f32; 256]).expect("quantize");
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * BLOCK_PQ2_0_BYTES,
            )
        };
        let back = BlockPQ2_0::slice_from_bytes_exact(bytes, 2).expect("cast");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0], blocks[0]);
    }

    #[test]
    fn quantize_rejects_non_multiple_input() {
        assert!(BlockPQ2_0::quantize(&vec![0.0f32; 127]).is_err());
        assert!(BlockQ2_0G64::quantize(&vec![0.0f32; 63]).is_err());
        assert!(BlockPTQ1_0::quantize(&vec![0.0f32; 129]).is_err());
    }

    #[test]
    fn dequant_rejects_short_output() {
        let blocks = BlockPQ2_0::quantize(&vec![0.5f32; 128]).expect("quantize");
        let mut out = vec![0.0f32; 127];
        assert!(BlockPQ2_0::dequant(&blocks, &mut out).is_err());
    }
}
