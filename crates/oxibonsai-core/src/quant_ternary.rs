//! Ternary quantization block types for TQ2_0_g128 and TQ2_0 formats.
//!
//! Two ternary formats: `BlockTQ2_0_g128` (128 weights, 34 bytes, PrismML)
//! and `BlockTQ2_0` (256 weights, 66 bytes, llama.cpp compat).
//! Both use 2-bit coding: `00→-1`, `01→0`, `10→+1`, 4 weights per byte LSB-first.

use half::f16;

use crate::error::{BonsaiError, BonsaiResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of weights per TQ2_0_g128 block.
pub const QK_TQ2_0_G128: usize = 128;

/// Number of weights per TQ2_0 block.
pub const QK_TQ2_0: usize = 256;

/// Number of bytes per TQ2_0_g128 block.
pub const BLOCK_TQ2_0_G128_BYTES: usize = 34;

/// Number of bytes per TQ2_0 block.
pub const BLOCK_TQ2_0_BYTES: usize = 66;

// ---------------------------------------------------------------------------
// TernaryCode
// ---------------------------------------------------------------------------

/// Ternary weight code for 2-bit encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TernaryCode {
    /// Negative weight (-1): bit pattern `0b00`.
    Neg = 0b00,
    /// Zero weight (0): bit pattern `0b01`.
    Zero = 0b01,
    /// Positive weight (+1): bit pattern `0b10`.
    Pos = 0b10,
}

impl TernaryCode {
    /// Convert to integer representation: Neg→-1, Zero→0, Pos→+1.
    pub fn to_i8(self) -> i8 {
        match self {
            Self::Neg => -1,
            Self::Zero => 0,
            Self::Pos => 1,
        }
    }
}

/// The single decode table for the **`TQ2_0` family** (OxiBonsai's `TQ2_0_g128`
/// and llama.cpp's 256-wide `TQ2_0`): `0b00 → -1`, `0b01 → 0`, `0b10 → +1`,
/// `0b11 → 0`.
///
/// `0b11` is unreachable from any ggml encoder (`q = round(w/amax) + 1` is
/// always in `0..=2`), and six GPU decoders plus the CPU/Metal byte-parity
/// guard hard-code the map to zero, so it stays zero here. This is deliberately
/// **not** the `Q2_0`/`PQ2_0` map, which is arithmetic (`code - 1`, so
/// `0b11 → +2`) — see [`crate::quant_prism::q2_0_code_to_i32`].
///
/// Only the low two bits of `code` are read, so callers may pass a shifted
/// byte directly.
#[inline]
pub const fn ternary_code_to_i8(code: u8) -> i8 {
    [-1i8, 0, 1, 0][(code & 0x03) as usize]
}

// ---------------------------------------------------------------------------
// BlockTQ2_0_g128
// ---------------------------------------------------------------------------

/// TQ2_0_g128 block: 128 weights at 2 bits each, PrismML format.
///
/// Layout (34 bytes): `qs[32]` packed codes + `d` FP16 scale.
/// Bit coding: `00→-1`, `01→0`, `10→+1`, 4 weights per byte LSB-first.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockTQ2_0_g128 {
    /// 128 × 2-bit quantized weights, 4 per byte, LSB-first.
    pub qs: [u8; 32],
    /// Block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockTQ2_0_g128>() == BLOCK_TQ2_0_G128_BYTES);

impl BlockTQ2_0_g128 {
    /// Dequantize a slice of TQ2_0_g128 blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * 128`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_TQ2_0_G128;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_TQ2_0_G128;
            for j in 0..QK_TQ2_0_G128 {
                let byte_idx = j / 4;
                let lane = j % 4;
                let code_val = Self::ternary_decode(block.qs[byte_idx], lane);
                output[base + j] = d * (code_val as f32);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into TQ2_0_g128 blocks.
    ///
    /// Input length must be a multiple of 128.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if !input.len().is_multiple_of(QK_TQ2_0_G128) {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_TQ2_0_G128
                ),
            });
        }
        let num_blocks = input.len() / QK_TQ2_0_G128;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_TQ2_0_G128;
            let chunk = &input[base..base + QK_TQ2_0_G128];

            let absmax = chunk
                .iter()
                .copied()
                .fold(0.0f32, |acc, x| acc.max(x.abs()));

            let mut qs = [0u8; 32];

            if absmax == 0.0 {
                // All zero: code = 0b01 (Zero), qs bytes = 0b01_01_01_01 = 0x55
                for b in qs.iter_mut() {
                    *b = 0x55;
                }
                blocks.push(BlockTQ2_0_g128 { qs, d: f16::ZERO });
                continue;
            }

            let threshold = 0.5 * absmax;
            for (j, &x) in chunk.iter().enumerate() {
                let code: u8 = if x >= threshold {
                    TernaryCode::Pos as u8 // 0b10
                } else if x <= -threshold {
                    TernaryCode::Neg as u8 // 0b00
                } else {
                    TernaryCode::Zero as u8 // 0b01
                };
                let byte_idx = j / 4;
                let shift = (j % 4) * 2;
                qs[byte_idx] |= code << shift;
            }

            blocks.push(BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(absmax),
            });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of TQ2_0_g128 blocks.
    ///
    /// Returns error if length is not a multiple of 34 or pointer is misaligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if !data.len().is_multiple_of(BLOCK_TQ2_0_G128_BYTES) {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_TQ2_0_G128_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0_g128 slice_from_bytes: pointer not {}-byte aligned",
                    align
                ),
            });
        }
        let count = data.len() / BLOCK_TQ2_0_G128_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and alignment
        // checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Decode a 2-bit code at `lane` (0..4) from `byte`, returning the weight as i8.
    ///
    /// Code map: `0b00→-1`, `0b01→0`, `0b10→+1`, `0b11→0` (reserved treated as zero).
    /// Routed through the single shared [`ternary_code_to_i8`] table.
    pub fn ternary_decode(byte: u8, lane: usize) -> i8 {
        let shift = lane * 2;
        ternary_code_to_i8(byte >> shift)
    }
}

// ---------------------------------------------------------------------------
// BlockTQ2_0
// ---------------------------------------------------------------------------

/// TQ2_0 block: 256 weights at 2 bits each, llama.cpp compat format.
///
/// Layout (66 bytes): `qs[64]` packed codes + `d` FP16 scale.
/// Same 2-bit coding as TQ2_0_g128.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockTQ2_0 {
    /// 256 × 2-bit quantized weights, 4 per byte, LSB-first.
    pub qs: [u8; 64],
    /// Block scale (FP16).
    pub d: f16,
}

const _: () = assert!(std::mem::size_of::<BlockTQ2_0>() == BLOCK_TQ2_0_BYTES);

impl BlockTQ2_0 {
    /// Dequantize a slice of TQ2_0 blocks into f32 output.
    ///
    /// `output` must have length >= `blocks.len() * 256`.
    pub fn dequant(blocks: &[Self], output: &mut [f32]) -> BonsaiResult<()> {
        let expected_len = blocks.len() * QK_TQ2_0;
        if output.len() < expected_len {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0 dequant: output len {} < expected {}",
                    output.len(),
                    expected_len
                ),
            });
        }
        for (block_idx, block) in blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let base = block_idx * QK_TQ2_0;
            for j in 0..QK_TQ2_0 {
                let byte_idx = j / 4;
                let lane = j % 4;
                let code_val = ternary_decode_g256(block.qs[byte_idx], lane);
                output[base + j] = d * (code_val as f32);
            }
        }
        Ok(())
    }

    /// Quantize f32 input into TQ2_0 blocks.
    ///
    /// Input length must be a multiple of 256.
    pub fn quantize(input: &[f32]) -> BonsaiResult<Vec<Self>> {
        if !input.len().is_multiple_of(QK_TQ2_0) {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0 quantize: input len {} not a multiple of {}",
                    input.len(),
                    QK_TQ2_0
                ),
            });
        }
        let num_blocks = input.len() / QK_TQ2_0;
        let mut blocks = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let base = block_idx * QK_TQ2_0;
            let chunk = &input[base..base + QK_TQ2_0];

            let absmax = chunk
                .iter()
                .copied()
                .fold(0.0f32, |acc, x| acc.max(x.abs()));

            let mut qs = [0u8; 64];

            if absmax == 0.0 {
                for b in qs.iter_mut() {
                    *b = 0x55;
                }
                blocks.push(BlockTQ2_0 { qs, d: f16::ZERO });
                continue;
            }

            let threshold = 0.5 * absmax;
            for (j, &x) in chunk.iter().enumerate() {
                let code: u8 = if x >= threshold {
                    TernaryCode::Pos as u8
                } else if x <= -threshold {
                    TernaryCode::Neg as u8
                } else {
                    TernaryCode::Zero as u8
                };
                let byte_idx = j / 4;
                let shift = (j % 4) * 2;
                qs[byte_idx] |= code << shift;
            }

            blocks.push(BlockTQ2_0 {
                qs,
                d: f16::from_f32(absmax),
            });
        }
        Ok(blocks)
    }

    /// Zero-copy cast of a byte slice to a slice of TQ2_0 blocks.
    ///
    /// Returns error if length is not a multiple of 66 or pointer is misaligned.
    pub fn slice_from_bytes(data: &[u8]) -> BonsaiResult<&[Self]> {
        if !data.len().is_multiple_of(BLOCK_TQ2_0_BYTES) {
            return Err(BonsaiError::KQuantError {
                reason: format!(
                    "TQ2_0 slice_from_bytes: byte len {} not a multiple of {}",
                    data.len(),
                    BLOCK_TQ2_0_BYTES
                ),
            });
        }
        let align = std::mem::align_of::<Self>();
        if data.as_ptr().align_offset(align) != 0 {
            return Err(BonsaiError::KQuantError {
                reason: format!("TQ2_0 slice_from_bytes: pointer not {}-byte aligned", align),
            });
        }
        let count = data.len() / BLOCK_TQ2_0_BYTES;
        let ptr = data.as_ptr() as *const Self;
        // SAFETY: repr(C) layout validated by compile-time assert; length and alignment
        // checked above; lifetime tied to input slice.
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }
}

/// Decode a 2-bit code at `lane` (0..4) from `byte` for BlockTQ2_0.
///
/// Code map: `0b00→-1`, `0b01→0`, `0b10→+1`, `0b11→0` (reserved treated as zero).
/// Routed through the single shared [`ternary_code_to_i8`] table.
fn ternary_decode_g256(byte: u8, lane: usize) -> i8 {
    let shift = lane * 2;
    ternary_code_to_i8(byte >> shift)
}

// ---------------------------------------------------------------------------
// Load-time 2-bit layout sniff (design §1.3)
// ---------------------------------------------------------------------------

/// The three on-disk readings a 2-bit block tensor can have.
///
/// ggml type id 42 ships in all three; ids 142 (`PQ2_0`) and 35 (`TQ2_0`) are
/// unambiguous, but the same structural test is what proves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoBitLayout {
    /// 34-byte block, `d` FIRST (`block_pq2_0` / PrismML gen-1 id 42).
    DFirst34,
    /// 34-byte block, `qs` FIRST and `d` LAST (legacy OxiBonsai id 42).
    QsFirst34,
    /// 18-byte block, `d` FIRST (mainline `block_q2_0`, group 64).
    DFirst18,
    /// No single hypothesis is structurally clean, or more than one is.
    ///
    /// An all-zero or tiny sample is legitimately ambiguous; so is a genuinely
    /// corrupt tensor. The caller decides which, and must never guess.
    Ambiguous,
}

impl TwoBitLayout {
    /// Bytes per block under this hypothesis.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::DFirst34 | Self::QsFirst34 => 34,
            Self::DFirst18 => 18,
            Self::Ambiguous => 0,
        }
    }

    /// Weights per block under this hypothesis.
    pub const fn block_size(self) -> usize {
        match self {
            Self::DFirst34 | Self::QsFirst34 => 128,
            Self::DFirst18 => 64,
            Self::Ambiguous => 0,
        }
    }

    /// Byte offset of the FP16 scale inside a block.
    const fn scale_offset(self) -> usize {
        match self {
            Self::DFirst34 | Self::DFirst18 => 0,
            Self::QsFirst34 => 32,
            Self::Ambiguous => 0,
        }
    }

    /// Byte offset of the packed codes inside a block.
    const fn codes_offset(self) -> usize {
        match self {
            Self::DFirst34 | Self::DFirst18 => 2,
            Self::QsFirst34 => 0,
            Self::Ambiguous => 0,
        }
    }

    /// Number of code bytes inside a block.
    const fn codes_len(self) -> usize {
        match self {
            Self::DFirst34 | Self::QsFirst34 => 32,
            Self::DFirst18 => 16,
            Self::Ambiguous => 0,
        }
    }

    /// Human-readable name used in error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::DFirst34 => "d-first/34B/group-128",
            Self::QsFirst34 => "qs-first/34B/group-128",
            Self::DFirst18 => "d-first/18B/group-64",
            Self::Ambiguous => "ambiguous",
        }
    }
}

/// The three hypotheses the sniff evaluates, in a stable order.
pub const TWO_BIT_LAYOUT_CANDIDATES: [TwoBitLayout; 3] = [
    TwoBitLayout::DFirst34,
    TwoBitLayout::QsFirst34,
    TwoBitLayout::DFirst18,
];

/// Structural evidence gathered for one layout hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutScore {
    /// Which hypothesis this score belongs to.
    pub candidate: TwoBitLayout,
    /// How many blocks were actually examined.
    pub blocks_sampled: usize,
    /// How many 2-bit lanes decoded to the reserved `0b11` (`+2`) code.
    ///
    /// PrismML checkpoints are ternary, so a correct reading gives exactly 0.
    pub lanes_eq_three: u64,
    /// How many block scales were non-finite or negative.
    pub bad_scales: u64,
    /// `false` when the buffer length is not a whole number of blocks for
    /// this hypothesis, which rules it out on its own.
    pub length_compatible: bool,
}

impl LayoutScore {
    /// A hypothesis is clean when it examined real blocks and found neither a
    /// reserved code nor an implausible scale.
    pub fn is_clean(&self) -> bool {
        self.length_compatible
            && self.blocks_sampled > 0
            && self.lanes_eq_three == 0
            && self.bad_scales == 0
    }
}

/// Default number of blocks to sample; the design calls for `>= 2000`, which
/// separates the real files by 0 vs 450–5153 reserved codes.
pub const SNIFF_DEFAULT_BLOCKS: usize = 2000;

/// A byte length that is an exact multiple of every [`TwoBitLayout`]
/// candidate's block size (34, 34 and 18 bytes) — `18 * 34`.
///
/// `sniff_two_bit_layout_scores` only ever inspects **whole** blocks
/// (`available = bytes.len() / blk`), so `LayoutScore::length_compatible`'s
/// exact-multiple check is genuine evidence only when `bytes` is a tensor's
/// real, untruncated data: a true group-64 tensor's total is `k * 18`
/// bytes, which happens to also be a multiple of 34 only for roughly one
/// `k` in seventeen, so a wrong-geometry candidate usually fails this check
/// honestly. But a *caller-chosen* truncation length that is a multiple of
/// one candidate's block size and not another's manufactures that same
/// "fail" for a reason that has nothing to do with the bytes — only with
/// where the cut landed. Capping any truncation at a multiple of this
/// constant keeps every candidate's block boundary intact no matter which
/// one is actually correct.
const SNIFF_SAMPLE_COMMON_MULTIPLE_BYTES: usize = 612;

/// Round `n_blocks` worth of the widest candidate's block size (34 bytes) up
/// to a multiple of `SNIFF_SAMPLE_COMMON_MULTIPLE_BYTES`, for use as a
/// byte cap before calling [`sniff_two_bit_layout`] (or
/// [`sniff_two_bit_layout_scores`]) on a tensor that may be far larger than
/// the sample the sniff actually needs.
///
/// Rounding **up** rather than down still guarantees every candidate —
/// including the smaller-block [`TwoBitLayout::DFirst18`] — sees at least
/// `n_blocks` whole blocks, matching design §1.3's "over >= 2000 sampled
/// blocks": `sniff_sample_byte_cap(2000) == 68544`, which is 2016 blocks of
/// 34 bytes and 3808 blocks of 18 bytes, both comfortably over 2000.
///
/// Without this, truncating a large tensor's sample to exactly
/// `n_blocks * 34` bytes (68 000 for the default) is not a multiple of 18,
/// so it silently disqualifies a genuinely clean [`TwoBitLayout::DFirst18`]
/// reading via `length_compatible` — the mandatory group-64 discriminator
/// (design §1.3: "MANDATORY, not a backstop") would degrade to a no-op for
/// every group-64 tensor above the cap.
pub fn sniff_sample_byte_cap(n_blocks: usize) -> usize {
    let widest_candidate_budget = n_blocks.saturating_mul(BLOCK_TQ2_0_G128_BYTES);
    widest_candidate_budget
        .div_ceil(SNIFF_SAMPLE_COMMON_MULTIPLE_BYTES)
        .saturating_mul(SNIFF_SAMPLE_COMMON_MULTIPLE_BYTES)
}

/// Score every 2-bit layout hypothesis against a tensor's raw bytes.
///
/// For each of `{d-first-34, qs-first-34, d-first-18}` this counts illegal
/// `0b11` lanes and checks that the FP16 block scale is finite and
/// non-negative, over up to `n_blocks` blocks.
pub fn sniff_two_bit_layout_scores(bytes: &[u8], n_blocks: usize) -> [LayoutScore; 3] {
    let mut scores = [LayoutScore {
        candidate: TwoBitLayout::Ambiguous,
        blocks_sampled: 0,
        lanes_eq_three: 0,
        bad_scales: 0,
        length_compatible: false,
    }; 3];

    for (slot, candidate) in TWO_BIT_LAYOUT_CANDIDATES.iter().enumerate() {
        let blk = candidate.block_bytes();
        let available = bytes.len() / blk;
        let sampled = available.min(n_blocks);
        let mut lanes_eq_three = 0u64;
        let mut bad_scales = 0u64;

        for i in 0..sampled {
            let base = i * blk;
            let s = base + candidate.scale_offset();
            let d = f16::from_le_bytes([bytes[s], bytes[s + 1]]).to_f32();
            if !d.is_finite() || d < 0.0 {
                bad_scales += 1;
            }
            let c = base + candidate.codes_offset();
            for &b in &bytes[c..c + candidate.codes_len()] {
                for shift in [0u32, 2, 4, 6] {
                    if (b >> shift) & 0x03 == 0x03 {
                        lanes_eq_three += 1;
                    }
                }
            }
        }

        scores[slot] = LayoutScore {
            candidate: *candidate,
            blocks_sampled: sampled,
            lanes_eq_three,
            bad_scales,
            length_compatible: bytes.len().is_multiple_of(blk) && available > 0,
        };
    }

    scores
}

/// Decide which 2-bit layout a tensor's raw bytes actually use.
///
/// Mainline llama.cpp loads a gen-2 `Q2_0` file "without a warning and outputs
/// gibberish" (`MODEL-FORMATS.md`); this is the structural check that lets
/// OxiBonsai fail loudly instead. It returns [`TwoBitLayout::Ambiguous`]
/// unless **exactly one** length-compatible hypothesis is clean — an all-zero
/// or too-small sample is genuinely undecidable and must not be guessed.
///
/// Measured separation on the real files (2000 blocks of the first quantized
/// tensor): the correct reading scores 0 reserved codes and 0 bad scales; the
/// wrong ones score 450–5153 reserved codes and 508–652 bad scales.
pub fn sniff_two_bit_layout(bytes: &[u8], n_blocks: usize) -> TwoBitLayout {
    let scores = sniff_two_bit_layout_scores(bytes, n_blocks);
    let mut winner = TwoBitLayout::Ambiguous;
    let mut clean = 0usize;
    for score in scores.iter() {
        if score.is_clean() {
            clean += 1;
            winner = score.candidate;
        }
    }
    if clean == 1 {
        winner
    } else {
        TwoBitLayout::Ambiguous
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tq2_0_g128_block_size_correct() {
        assert_eq!(
            std::mem::size_of::<BlockTQ2_0_g128>(),
            BLOCK_TQ2_0_G128_BYTES
        );
        assert_eq!(BLOCK_TQ2_0_G128_BYTES, 34);
    }

    #[test]
    fn tq2_0_block_size_correct() {
        assert_eq!(std::mem::size_of::<BlockTQ2_0>(), BLOCK_TQ2_0_BYTES);
        assert_eq!(BLOCK_TQ2_0_BYTES, 66);
    }

    #[test]
    fn tq2_0_g128_roundtrip_uniform() {
        // Alternating 0.5, -0.5, 0.0 pattern for 128 values.
        let mut input = [0.0f32; 128];
        for (i, x) in input.iter_mut().enumerate() {
            *x = match i % 3 {
                0 => 0.5,
                1 => -0.5,
                _ => 0.0,
            };
        }
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        let mut output = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");
        let mse: f32 = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 128.0;
        assert!(mse < 1e-3, "MSE {mse} too high");
    }

    #[test]
    fn tq2_0_g128_all_zero_input() {
        let input = [0.0f32; 128];
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].d, f16::ZERO);
        let mut output = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");
        for &v in &output {
            assert_eq!(v, 0.0, "all outputs should be zero");
        }
    }

    #[test]
    fn tq2_0_g128_all_positive() {
        let input = [1.0f32; 128];
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        assert_eq!(blocks.len(), 1);
        // absmax = 1.0 → d = f16(1.0)
        assert!(
            (blocks[0].d.to_f32() - 1.0).abs() < 1e-3,
            "d should be ~1.0"
        );
        // All codes should be Pos (0b10), so each byte = 0b10101010 = 0xAA
        for &b in &blocks[0].qs {
            assert_eq!(b, 0xAA, "all bytes should be 0xAA for all-positive");
        }
    }

    #[test]
    fn tq2_0_g128_all_negative() {
        let input = [-1.0f32; 128];
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        assert_eq!(blocks.len(), 1);
        // absmax = 1.0 → d = f16(1.0)
        assert!(
            (blocks[0].d.to_f32() - 1.0).abs() < 1e-3,
            "d should be ~1.0"
        );
        // All codes should be Neg (0b00), so each byte = 0b00000000 = 0x00
        for &b in &blocks[0].qs {
            assert_eq!(b, 0x00, "all bytes should be 0x00 for all-negative");
        }
    }

    #[test]
    fn tq2_0_g128_mixed_threshold() {
        // Pattern: [2.0, 0.9, 0.0, -0.9, -2.0] repeating to fill 128 elements.
        // absmax=2.0, threshold=1.0:
        //   2.0 ≥ 1.0  → Pos (+d = 2.0)
        //   0.9 < 1.0  → Zero (0.0)
        //   0.0 < 1.0  → Zero (0.0)
        //  -0.9: abs=0.9 < 1.0 → Zero (0.0)
        //  -2.0 ≤ -1.0 → Neg (-d = -2.0)
        let mut input = [0.0f32; 128];
        let pattern = [2.0f32, 0.9, 0.0, -0.9, -2.0];
        for i in 0..128 {
            input[i] = pattern[i % 5];
        }
        let blocks = BlockTQ2_0_g128::quantize(&input).expect("quantize should succeed");
        let mut output = vec![0.0f32; 128];
        BlockTQ2_0_g128::dequant(&blocks, &mut output).expect("dequant should succeed");

        let expected_pattern = [2.0f32, 0.0, 0.0, 0.0, -2.0];
        for i in 0..128 {
            let expected = expected_pattern[i % 5];
            assert!(
                (output[i] - expected).abs() < 1e-3,
                "index {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn tq2_0_g128_slice_from_bytes_misaligned() {
        // 35 bytes is not a multiple of 34 → should return Err.
        let data = vec![0u8; 35];
        let result = BlockTQ2_0_g128::slice_from_bytes(&data);
        assert!(result.is_err(), "35-byte slice should fail");
    }

    #[test]
    fn tq2_0_g128_slice_from_bytes_aligned() {
        // Build a real block and reinterpret as bytes (guaranteed alignment).
        let block = BlockTQ2_0_g128 {
            qs: [0u8; 32],
            d: f16::from_f32(1.0),
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &block as *const BlockTQ2_0_g128 as *const u8,
                BLOCK_TQ2_0_G128_BYTES,
            )
        };
        let result =
            BlockTQ2_0_g128::slice_from_bytes(bytes).expect("aligned slice should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].d, f16::from_f32(1.0));
    }

    #[test]
    fn tq2_0_roundtrip_random() {
        // 256 values oscillating in [-1, 1].
        let mut input = [0.0f32; 256];
        for (i, x) in input.iter_mut().enumerate() {
            *x = ((i as f32) / 128.0 - 1.0).clamp(-1.0, 1.0);
        }
        let blocks = BlockTQ2_0::quantize(&input).expect("quantize should succeed");
        let mut output = vec![0.0f32; 256];
        BlockTQ2_0::dequant(&blocks, &mut output).expect("dequant should succeed");
        let mse: f32 = input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 256.0;
        // TQ2_0 is a 3-level ternary quantizer; on a continuous ramp in [-1,1]
        // a large fraction of values are zeroed (|x| < 0.5 * absmax), so MSE
        // around 0.08–0.10 is expected.  Require < 0.15 to catch regressions.
        assert!(mse < 0.15, "MSE {mse} too high for TQ2_0 roundtrip");
    }

    #[test]
    fn ternary_decode_all_lanes() {
        // Construct a byte to test all four lanes:
        //   lane 0 (bits 1:0): 0b00 → -1
        //   lane 1 (bits 3:2): 0b11 → 0 (reserved)
        //   lane 2 (bits 5:4): 0b01 → 0
        //   lane 3 (bits 7:6): 0b10 → +1
        // Byte = 0b10_01_11_00 = 0b10011100 = 0x9C
        let byte: u8 = 0b10011100;
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 0),
            -1,
            "lane 0: 0b00 → -1"
        );
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 1),
            0,
            "lane 1: 0b11 → 0 (reserved)"
        );
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 2),
            0,
            "lane 2: 0b01 → 0"
        );
        assert_eq!(
            BlockTQ2_0_g128::ternary_decode(byte, 3),
            1,
            "lane 3: 0b10 → +1"
        );
    }

    // ── Shared decode table ───────────────────────────────────────────────

    /// The TQ2_0 family keeps `0b11 → 0`. Six GPU decoders and the
    /// CPU/Metal byte-parity guard depend on it; the arithmetic `code - 1`
    /// map (`0b11 → +2`) belongs to the Q2_0/PQ2_0 family only.
    #[test]
    fn ternary_code_table_is_the_three_level_lut() {
        assert_eq!(ternary_code_to_i8(0b00), -1);
        assert_eq!(ternary_code_to_i8(0b01), 0);
        assert_eq!(ternary_code_to_i8(0b10), 1);
        assert_eq!(ternary_code_to_i8(0b11), 0);
        // Only the low two bits are read, so a shifted byte works directly.
        assert_eq!(ternary_code_to_i8(0b1111_1110), 1);
    }

    #[test]
    fn both_decoders_route_through_the_shared_table() {
        for byte in 0u8..=255 {
            for lane in 0..4usize {
                let expect = ternary_code_to_i8(byte >> (lane * 2));
                assert_eq!(BlockTQ2_0_g128::ternary_decode(byte, lane), expect);
                assert_eq!(ternary_decode_g256(byte, lane), expect);
            }
        }
    }

    // ── Layout sniff ──────────────────────────────────────────────────────

    /// Build `n` blocks of ternary codes under a chosen layout so the sniff
    /// has something structurally clean to find.
    fn synth_two_bit(layout: TwoBitLayout, n: usize, scale: f32) -> Vec<u8> {
        let blk = layout.block_bytes();
        let mut out = vec![0u8; n * blk];
        let d = f16::from_f32(scale).to_le_bytes();
        for i in 0..n {
            let base = i * blk;
            let s = base + layout.scale_offset();
            out[s] = d[0];
            out[s + 1] = d[1];
            let c = base + layout.codes_offset();
            for k in 0..layout.codes_len() {
                // 0b10_01_00_01 — only legal ternary codes.
                out[c + k] = 0b10_01_00_01;
            }
        }
        out
    }

    #[test]
    fn sniff_identifies_qs_first_34_layout() {
        let buf = synth_two_bit(TwoBitLayout::QsFirst34, 256, 0.0415);
        assert_eq!(sniff_two_bit_layout(&buf, 2000), TwoBitLayout::QsFirst34);
    }

    #[test]
    fn sniff_identifies_d_first_34_layout() {
        let buf = synth_two_bit(TwoBitLayout::DFirst34, 256, 0.0109);
        assert_eq!(sniff_two_bit_layout(&buf, 2000), TwoBitLayout::DFirst34);
    }

    #[test]
    fn sniff_identifies_d_first_18_layout() {
        let buf = synth_two_bit(TwoBitLayout::DFirst18, 256, 0.0109);
        assert_eq!(sniff_two_bit_layout(&buf, 2000), TwoBitLayout::DFirst18);
    }

    /// The wrong hypothesis must accumulate reserved codes and/or implausible
    /// scales — that separation is the only discriminator the id-42 resolver
    /// has for d-first vs qs-first.
    #[test]
    fn sniff_scores_separate_the_hypotheses() {
        let buf = synth_two_bit(TwoBitLayout::QsFirst34, 256, 0.0415);
        let scores = sniff_two_bit_layout_scores(&buf, 2000);
        let qs = scores
            .iter()
            .find(|s| s.candidate == TwoBitLayout::QsFirst34)
            .expect("candidate present");
        let df = scores
            .iter()
            .find(|s| s.candidate == TwoBitLayout::DFirst34)
            .expect("candidate present");
        assert!(qs.is_clean(), "correct reading must be clean: {qs:?}");
        assert!(!df.is_clean(), "wrong reading must not be clean: {df:?}");
        assert!(df.lanes_eq_three > 0 || df.bad_scales > 0);
    }

    /// Regression guard: the naive "`n_blocks` of the widest (34-byte)
    /// candidate" truncation is not a multiple of 18, so it silently
    /// disqualifies an otherwise-clean [`TwoBitLayout::DFirst18`] reading via
    /// `length_compatible` — even though every block the sniff actually
    /// examined (`available = bytes.len() / 18`) was intact.
    /// [`sniff_sample_byte_cap`] exists to prevent exactly this.
    #[test]
    fn naive_widest_candidate_cut_wrongly_disqualifies_d_first_18() {
        // Comfortably more than 2000 blocks' worth of clean DFirst18 data at
        // either cut below.
        let buf = synth_two_bit(TwoBitLayout::DFirst18, 4000, 0.0109);
        assert_eq!(buf.len(), 4000 * 18);

        // The pre-fix cut: `SNIFF_DEFAULT_BLOCKS * 34` bytes.
        let naive_cut = SNIFF_DEFAULT_BLOCKS * 34;
        assert!(naive_cut < buf.len(), "fixture must exceed the naive cut");
        assert_ne!(
            naive_cut % TwoBitLayout::DFirst18.block_bytes(),
            0,
            "the naive cut must NOT be an 18-byte multiple, or it doesn't reproduce the bug"
        );
        let scores_naive = sniff_two_bit_layout_scores(&buf[..naive_cut], SNIFF_DEFAULT_BLOCKS);
        let df18_naive = scores_naive
            .iter()
            .find(|s| s.candidate == TwoBitLayout::DFirst18)
            .expect("candidate present");
        assert!(
            !df18_naive.length_compatible,
            "documents the bug: the naive cut disqualifies a clean candidate on length alone: \
             {df18_naive:?}"
        );

        // The fixed cap keeps every candidate's block size whole.
        let safe_cut = sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS);
        assert!(safe_cut < buf.len(), "fixture must exceed the safe cut too");
        assert_eq!(safe_cut % TwoBitLayout::DFirst18.block_bytes(), 0);
        assert_eq!(safe_cut % TwoBitLayout::QsFirst34.block_bytes(), 0);
        let scores_safe = sniff_two_bit_layout_scores(&buf[..safe_cut], SNIFF_DEFAULT_BLOCKS);
        let df18_safe = scores_safe
            .iter()
            .find(|s| s.candidate == TwoBitLayout::DFirst18)
            .expect("candidate present");
        assert!(
            df18_safe.is_clean(),
            "the mandatory sniff must stay live for a large, genuinely clean group-64 sample: \
             {df18_safe:?}"
        );
    }

    /// An all-zero buffer is clean under every hypothesis, so the sniff must
    /// say so rather than pick one.
    #[test]
    fn sniff_reports_ambiguous_when_nothing_separates() {
        // 34*18 = 612 = 18*34, so both block sizes divide it exactly.
        let buf = vec![0u8; 34 * 18];
        assert_eq!(sniff_two_bit_layout(&buf, 2000), TwoBitLayout::Ambiguous);
    }

    #[test]
    fn sniff_reports_ambiguous_on_empty_input() {
        assert_eq!(sniff_two_bit_layout(&[], 2000), TwoBitLayout::Ambiguous);
    }

    #[test]
    fn layout_geometry_matches_the_probe_table() {
        assert_eq!(TwoBitLayout::DFirst34.block_bytes(), 34);
        assert_eq!(TwoBitLayout::DFirst34.block_size(), 128);
        assert_eq!(TwoBitLayout::QsFirst34.block_bytes(), 34);
        assert_eq!(TwoBitLayout::QsFirst34.block_size(), 128);
        assert_eq!(TwoBitLayout::DFirst18.block_bytes(), 18);
        assert_eq!(TwoBitLayout::DFirst18.block_size(), 64);
    }
}
