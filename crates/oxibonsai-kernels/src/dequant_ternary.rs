//! Reference (naive) dequantization kernels for ternary TQ2\_0\_g128 and TQ2\_0 formats.
//!
//! These are the correctness reference implementations — pure scalar Rust,
//! no SIMD, no unsafe. Each 2-bit code maps as: `00→-1`, `01→0`, `10→+1`,
//! `11→0` (reserved), via the single shared
//! [`oxibonsai_core::ternary_code_to_i8`] table (K-01): every NEON/AVX2/
//! AVX-512 ternary kernel in this crate, plus `BlockTQ2_0_g128::dequant` /
//! `ternary_decode` and `BlockTQ2_0::dequant` in `oxibonsai-core`, route
//! through that same table, so this format's decode cannot drift across the
//! two crates or across kernel tiers. `tests/ternary_cross_tier.rs` asserts
//! the whole matrix agrees.

use oxibonsai_core::{ternary_code_to_i8, BlockTQ2_0, BlockTQ2_0_g128, QK_TQ2_0, QK_TQ2_0_G128};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Decode a single 2-bit ternary code at `lane` (0..4) in `byte` to f32.
///
/// Code map: `0b00→-1.0`, `0b01→0.0`, `0b10→+1.0`, `0b11→0.0` (reserved),
/// via the shared [`ternary_code_to_i8`] table (K-01).
#[inline]
fn decode_code_f32(byte: u8, lane: usize) -> f32 {
    ternary_code_to_i8(byte >> (lane * 2)) as f32
}

// ---------------------------------------------------------------------------
// TQ2_0_g128 — 128 weights per block, 32 qs bytes
// ---------------------------------------------------------------------------

/// Dequantize TQ2\_0\_g128 blocks (128 weights/block) into f32 output.
///
/// For each block: `output[i] = scale * ternary_code[i]`
///
/// This is the reference implementation for correctness verification.
///
/// # Errors
///
/// Returns [`KernelError::BufferTooSmall`] if `output` is shorter than
/// `blocks.len() * 128`.
pub fn dequant_tq2_0_g128(blocks: &[BlockTQ2_0_g128], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_TQ2_0_G128;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let base = bi * QK_TQ2_0_G128;
        // 32 bytes × 4 lanes = 128 weights
        for byte_idx in 0..32 {
            let byte = block.qs[byte_idx];
            for lane in 0..4_usize {
                output[base + byte_idx * 4 + lane] = d * decode_code_f32(byte, lane);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TQ2_0 — 256 weights per block, 64 qs bytes
// ---------------------------------------------------------------------------

/// Dequantize TQ2\_0 blocks (256 weights/block) into f32 output.
///
/// Same 2-bit coding as TQ2\_0\_g128 but with 64 qs bytes and 256 weights per block.
///
/// # Errors
///
/// Returns [`KernelError::BufferTooSmall`] if `output` is shorter than
/// `blocks.len() * 256`.
pub fn dequant_tq2_0(blocks: &[BlockTQ2_0], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_TQ2_0;
    if output.len() < needed {
        return Err(KernelError::buffer_too_small(
            "output",
            needed,
            output.len(),
        ));
    }

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let base = bi * QK_TQ2_0;
        // 64 bytes × 4 lanes = 256 weights
        for byte_idx in 0..64 {
            let byte = block.qs[byte_idx];
            for lane in 0..4_usize {
                output[base + byte_idx * 4 + lane] = d * decode_code_f32(byte, lane);
            }
        }
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

    fn make_g128_block(scale: f32, qs: [u8; 32]) -> BlockTQ2_0_g128 {
        BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    fn make_g256_block(scale: f32, qs: [u8; 64]) -> BlockTQ2_0 {
        BlockTQ2_0 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    // --- TQ2_0_g128 tests ---

    /// All qs bytes = 0x55 → all 2-bit codes = 0b01 (Zero) → output all 0.0.
    #[test]
    fn tq2_0_g128_dequant_all_zero() {
        let block = make_g128_block(1.0, [0x55; 32]);
        let mut output = vec![0.0f32; QK_TQ2_0_G128];
        dequant_tq2_0_g128(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-6, "index {i}: expected 0.0, got {v}",);
        }
    }

    /// All qs bytes = 0xAA → all 2-bit codes = 0b10 (Pos) → output all +d.
    #[test]
    fn tq2_0_g128_dequant_all_pos() {
        let block = make_g128_block(2.0, [0xAA; 32]);
        let mut output = vec![0.0f32; QK_TQ2_0_G128];
        dequant_tq2_0_g128(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-4, "index {i}: expected 2.0, got {v}",);
        }
    }

    /// All qs bytes = 0x00 → all 2-bit codes = 0b00 (Neg) → output all -d.
    #[test]
    fn tq2_0_g128_dequant_all_neg() {
        let block = make_g128_block(2.0, [0x00; 32]);
        let mut output = vec![0.0f32; QK_TQ2_0_G128];
        dequant_tq2_0_g128(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!((v + 2.0).abs() < 1e-4, "index {i}: expected -2.0, got {v}",);
        }
    }

    /// Output slice too short → BufferTooSmall error.
    #[test]
    fn tq2_0_g128_dequant_buffer_too_small() {
        let block = make_g128_block(1.0, [0xAA; 32]);
        let mut output = vec![0.0f32; 0];
        let result = dequant_tq2_0_g128(&[block], &mut output);
        assert!(result.is_err(), "expected BufferTooSmall error");
    }

    /// First byte = 0b10_01_00_10 → lane 0: Pos(+d), lane 1: Neg(-d), lane 2: Zero(0), lane 3: Pos(+d).
    ///
    /// Byte layout LSB-first: bits[1:0]=lane0, bits[3:2]=lane1, bits[5:4]=lane2, bits[7:6]=lane3.
    /// 0b10_01_00_10:
    ///   lane 0 (bits[1:0]) = 0b10 → Pos → +d
    ///   lane 1 (bits[3:2]) = 0b00 → Neg → -d
    ///   lane 2 (bits[5:4]) = 0b01 → Zero → 0
    ///   lane 3 (bits[7:6]) = 0b10 → Pos → +d
    #[test]
    fn tq2_0_g128_dequant_mixed() {
        let mut qs = [0x55u8; 32]; // rest all zero (Zero codes)
        qs[0] = 0b10_01_00_10; // binary: bits[7:6]=10 bits[5:4]=01 bits[3:2]=00 bits[1:0]=10
        let block = make_g128_block(3.0, qs);
        let mut output = vec![0.0f32; QK_TQ2_0_G128];
        dequant_tq2_0_g128(&[block], &mut output).expect("dequant should succeed");

        // lane 0: code=0b10 (Pos) → +3.0
        assert!(
            (output[0] - 3.0).abs() < 1e-4,
            "output[0]: expected 3.0, got {}",
            output[0]
        );
        // lane 1: code=0b00 (Neg) → -3.0
        assert!(
            (output[1] + 3.0).abs() < 1e-4,
            "output[1]: expected -3.0, got {}",
            output[1]
        );
        // lane 2: code=0b01 (Zero) → 0.0
        assert!(
            output[2].abs() < 1e-6,
            "output[2]: expected 0.0, got {}",
            output[2]
        );
        // lane 3: code=0b10 (Pos) → +3.0
        assert!(
            (output[3] - 3.0).abs() < 1e-4,
            "output[3]: expected 3.0, got {}",
            output[3]
        );
        // remaining bytes (index 1..32) were 0x55 → all zero
        for (offset, val) in output[4..].iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "output[{}]: expected 0.0, got {}",
                offset + 4,
                val
            );
        }
    }

    // --- TQ2_0 tests ---

    /// All qs bytes = 0x55 → all Zero → output all 0.0.
    #[test]
    fn tq2_0_dequant_all_zero() {
        let block = make_g256_block(1.0, [0x55; 64]);
        let mut output = vec![0.0f32; QK_TQ2_0];
        dequant_tq2_0(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-6, "index {i}: expected 0.0, got {v}");
        }
    }

    /// All qs bytes = 0xAA → all Pos → output all +d.
    #[test]
    fn tq2_0_dequant_all_pos() {
        let block = make_g256_block(2.0, [0xAA; 64]);
        let mut output = vec![0.0f32; QK_TQ2_0];
        dequant_tq2_0(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-4, "index {i}: expected 2.0, got {v}",);
        }
    }

    /// Output slice too short → BufferTooSmall error.
    #[test]
    fn tq2_0_dequant_buffer_too_small() {
        let block = make_g256_block(1.0, [0xAA; 64]);
        let mut output = vec![0.0f32; 0];
        let result = dequant_tq2_0(&[block], &mut output);
        assert!(result.is_err(), "expected BufferTooSmall error");
    }
}
