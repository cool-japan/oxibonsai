//! FP8 dequantization reference kernels (E4M3FN and E5M2).
//!
//! Each block has 32 weights and a FP16 scale. The dequantized value
//! for weight i in block b is: `d_b × fp8_decode(qs_b[i])`.
//!
//! These are pure scalar Rust correctness-reference implementations — no SIMD,
//! no unsafe. SIMD specializations are a follow-on Slice.

use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, QK_FP8, fp8_e4m3_decode, fp8_e5m2_decode};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// E4M3FN dequantization
// ---------------------------------------------------------------------------

/// Scalar dequantization for FP8 E4M3FN blocks.
///
/// For each block `b` and each weight slot `i`:
/// `output[b * QK_FP8 + i] = block.d × fp8_e4m3_decode(block.qs[i])`
///
/// # Errors
///
/// Returns [`KernelError::BufferTooSmall`] if `output.len() < blocks.len() * QK_FP8`.
pub fn dequant_fp8_e4m3(blocks: &[BlockFP8E4M3], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_FP8;
    if output.len() < needed {
        return Err(KernelError::BufferTooSmall {
            needed,
            available: output.len(),
        });
    }

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let base = bi * QK_FP8;
        for i in 0..QK_FP8 {
            output[base + i] = d * fp8_e4m3_decode(block.qs[i]);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// E5M2 dequantization
// ---------------------------------------------------------------------------

/// Scalar dequantization for FP8 E5M2 blocks.
///
/// For each block `b` and each weight slot `i`:
/// `output[b * QK_FP8 + i] = block.d × fp8_e5m2_decode(block.qs[i])`
///
/// # Errors
///
/// Returns [`KernelError::BufferTooSmall`] if `output.len() < blocks.len() * QK_FP8`.
pub fn dequant_fp8_e5m2(blocks: &[BlockFP8E5M2], output: &mut [f32]) -> KernelResult<()> {
    let needed = blocks.len() * QK_FP8;
    if output.len() < needed {
        return Err(KernelError::BufferTooSmall {
            needed,
            available: output.len(),
        });
    }

    for (bi, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let base = bi * QK_FP8;
        for i in 0..QK_FP8 {
            output[base + i] = d * fp8_e5m2_decode(block.qs[i]);
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

    fn make_e4m3_block(scale: f32, qs: [u8; 32]) -> BlockFP8E4M3 {
        BlockFP8E4M3 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    fn make_e5m2_block(scale: f32, qs: [u8; 32]) -> BlockFP8E5M2 {
        BlockFP8E5M2 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    // --- E4M3 dequantization tests ---

    /// Zero qs bytes → E4M3 code 0x00 = +0.0; output is all zero regardless of scale.
    #[test]
    fn e4m3_dequant_all_zeros() {
        let block = make_e4m3_block(2.0, [0x00u8; 32]);
        let mut output = vec![0.0f32; QK_FP8];
        dequant_fp8_e4m3(&[block], &mut output).expect("dequant should succeed");
        // 0x00 decodes to 0.0 in E4M3; scale × 0.0 = 0.0
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-6, "index {i}: expected 0.0, got {v}");
        }
    }

    /// qs = 0x38 → E4M3 normal: exp=7, man=0 → 2^(7-7) × 1.0 = 1.0.  Scale=3.0 → output=3.0.
    #[test]
    fn e4m3_dequant_ones_with_scale() {
        // fp8_e4m3_decode(0x38): byte=0x38=0b00111000, sign=0, exp=0b0111=7, man=0b000=0
        // value = 2^(7-7) * (1+0/8) = 1.0
        let block = make_e4m3_block(3.0, [0x38u8; 32]);
        let mut output = vec![0.0f32; QK_FP8];
        dequant_fp8_e4m3(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 0.02,
                "index {i}: expected ~3.0, got {v}"
            );
        }
    }

    /// Negative scale: d=-1.0, qs=0x38 (decodes +1.0) → output=-1.0.
    #[test]
    fn e4m3_dequant_negative_scale() {
        let block = make_e4m3_block(-1.0, [0x38u8; 32]);
        let mut output = vec![0.0f32; QK_FP8];
        dequant_fp8_e4m3(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v + 1.0).abs() < 0.02,
                "index {i}: expected ~-1.0, got {v}"
            );
        }
    }

    /// Two blocks with different scales are dequantized independently.
    #[test]
    fn e4m3_dequant_two_blocks_independent() {
        // Block 0: scale=1.0, qs=0x38 → all 1.0
        // Block 1: scale=2.0, qs=0x38 → all 2.0
        let blocks = vec![
            make_e4m3_block(1.0, [0x38u8; 32]),
            make_e4m3_block(2.0, [0x38u8; 32]),
        ];
        let mut output = vec![0.0f32; QK_FP8 * 2];
        dequant_fp8_e4m3(&blocks, &mut output).expect("dequant should succeed");
        for i in 0..QK_FP8 {
            assert!(
                (output[i] - 1.0).abs() < 0.02,
                "block0[{i}]: expected ~1.0, got {}",
                output[i]
            );
        }
        for i in QK_FP8..2 * QK_FP8 {
            assert!(
                (output[i] - 2.0).abs() < 0.02,
                "block1[{i}]: expected ~2.0, got {}",
                output[i]
            );
        }
    }

    /// Zero-length block slice → zero output needed → always succeeds.
    #[test]
    fn e4m3_dequant_empty_blocks() {
        let mut output = vec![];
        dequant_fp8_e4m3(&[], &mut output).expect("empty dequant should succeed");
    }

    /// Buffer too small → BufferTooSmall error.
    #[test]
    fn e4m3_dequant_buffer_too_small() {
        let block = make_e4m3_block(1.0, [0x38u8; 32]);
        let mut output = vec![0.0f32; QK_FP8 - 1];
        let result = dequant_fp8_e4m3(&[block], &mut output);
        assert!(
            matches!(result, Err(KernelError::BufferTooSmall { .. })),
            "expected BufferTooSmall, got {result:?}"
        );
    }

    /// Buffer exactly the right size → success.
    #[test]
    fn e4m3_dequant_exact_buffer_size() {
        let block = make_e4m3_block(1.0, [0x38u8; 32]);
        let mut output = vec![0.0f32; QK_FP8];
        dequant_fp8_e4m3(&[block], &mut output).expect("exact-size buffer should succeed");
    }

    /// Oversized buffer → success, no out-of-bounds write.
    #[test]
    fn e4m3_dequant_oversized_buffer() {
        let block = make_e4m3_block(1.0, [0x38u8; 32]);
        let mut output = vec![99.0f32; QK_FP8 + 10];
        dequant_fp8_e4m3(&[block], &mut output).expect("oversized buffer should succeed");
        // Trailing elements should be untouched
        for i in QK_FP8..output.len() {
            assert_eq!(output[i], 99.0, "trailing element {i} was modified");
        }
    }

    /// qs = 0xB8 → E4M3 byte 0xB8 = sign=1, exp=7, man=0 → -1.0.  Scale=1.0 → output=-1.0.
    #[test]
    fn e4m3_dequant_negative_weights() {
        // 0xB8 = 0b10111000: sign=1, exp=0b0111=7, man=0b000=0 → -(2^0 × 1.0) = -1.0
        let block = make_e4m3_block(1.0, [0xB8u8; 32]);
        let mut output = vec![0.0f32; QK_FP8];
        dequant_fp8_e4m3(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v + 1.0).abs() < 0.02,
                "index {i}: expected ~-1.0, got {v}"
            );
        }
    }

    // --- E5M2 dequantization tests ---

    /// Zero qs → E5M2 code 0x00 = 0.0; output all zero.
    #[test]
    fn e5m2_dequant_all_zeros() {
        let block = make_e5m2_block(5.0, [0x00u8; 32]);
        let mut output = vec![0.0f32; QK_FP8];
        dequant_fp8_e5m2(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-6, "index {i}: expected 0.0, got {v}");
        }
    }

    /// qs = 0x3C → E5M2 normal: exp=15, man=0 → 2^(15-15)×1.0=1.0.  Scale=2.0 → output=2.0.
    #[test]
    fn e5m2_dequant_ones_with_scale() {
        // 0x3C = 0b00111100: sign=0, exp=0b01111=15, man=0b00=0 → 2^(15-15) = 1.0
        let block = make_e5m2_block(2.0, [0x3Cu8; 32]);
        let mut output = vec![0.0f32; QK_FP8];
        dequant_fp8_e5m2(&[block], &mut output).expect("dequant should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 0.02,
                "index {i}: expected ~2.0, got {v}"
            );
        }
    }

    /// Buffer too small → BufferTooSmall error.
    #[test]
    fn e5m2_dequant_buffer_too_small() {
        let block = make_e5m2_block(1.0, [0x00u8; 32]);
        let mut output = vec![0.0f32; 0];
        let result = dequant_fp8_e5m2(&[block], &mut output);
        assert!(
            matches!(result, Err(KernelError::BufferTooSmall { .. })),
            "expected BufferTooSmall, got {result:?}"
        );
    }

    /// Two blocks with different scales are dequantized correctly.
    #[test]
    fn e5m2_dequant_two_blocks_independent() {
        let blocks = vec![
            make_e5m2_block(1.0, [0x3Cu8; 32]), // block 0: all 1.0
            make_e5m2_block(4.0, [0x3Cu8; 32]), // block 1: all 4.0
        ];
        let mut output = vec![0.0f32; QK_FP8 * 2];
        dequant_fp8_e5m2(&blocks, &mut output).expect("dequant should succeed");
        for i in 0..QK_FP8 {
            assert!(
                (output[i] - 1.0).abs() < 0.02,
                "block0[{i}]: expected ~1.0, got {}",
                output[i]
            );
        }
        for i in QK_FP8..2 * QK_FP8 {
            assert!(
                (output[i] - 4.0).abs() < 0.02,
                "block1[{i}]: expected ~4.0, got {}",
                output[i]
            );
        }
    }
}
