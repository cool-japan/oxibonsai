//! Reference (naive) GEMV kernel for Q1\_0\_g128.
//!
//! Computes `output = weight_matrix @ input_vector` where the weight matrix
//! is stored in Q1\_0\_g128 format.
//!
//! **Algorithmic insight:** For 1-bit weights, the inner loop reduces to:
//! ```text
//! For each group of 128 weights:
//!     sum_positive = sum of input elements where bit = 1
//!     sum_negative = sum of input elements where bit = 0
//!     result += scale * (sum_positive - sum_negative)
//! ```

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

use crate::error::{KernelError, KernelResult};

/// Fused 1-bit GEMV: `output[row] = dot(weight_row, input)`.
///
/// - `blocks`: Row-major weight blocks. Row `i` starts at `i * blocks_per_row`.
/// - `input`: FP32 input vector of length `k`.
/// - `output`: FP32 output vector of length `n_rows`.
/// - `n_rows`: Number of output rows.
/// - `k`: Inner dimension (must be divisible by 128).
pub fn gemv_1bit_g128(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < k {
        return Err(KernelError::DimensionMismatch {
            expected: k,
            got: input.len(),
        });
    }
    if output.len() < n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: n_rows,
            available: output.len(),
        });
    }

    let blocks_per_row = k / QK1_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: expected_blocks,
            available: blocks.len(),
        });
    }

    for row in 0..n_rows {
        let mut sum = 0.0f32;
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];

        for (bi, block) in row_blocks.iter().enumerate() {
            let d = block.d.to_f32();
            let input_base = bi * QK1_0_G128;

            // Accumulate: for 1-bit, signed sum then scale
            let mut block_sum = 0.0f32;
            for j in 0..QK1_0_G128 {
                let byte_index = j / 8;
                let bit_offset = j % 8;
                let sign = if (block.qs[byte_index] >> bit_offset) & 1 != 0 {
                    1.0f32
                } else {
                    -1.0f32
                };
                block_sum += sign * input[input_base + j];
            }
            sum += d * block_sum;
        }

        output[row] = sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn gemv_identity_like() {
        // Single row, single block, all bits = 1, scale=1.0
        // dot(+1*128, input) = sum(input)
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 1];

        gemv_1bit_g128(&blocks, &input, &mut output, 1, 128).expect("gemv should succeed");

        let expected: f32 = (0..128).map(|i| i as f32).sum();
        assert!(
            (output[0] - expected).abs() < 1.0,
            "expected ~{expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn gemv_negative_weights() {
        // All bits = 0 → all -d → dot = -d * sum(input)
        let blocks = vec![make_block(2.0, [0x00; 16])];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];

        gemv_1bit_g128(&blocks, &input, &mut output, 1, 128).expect("gemv should succeed");

        let expected = -2.0 * 128.0;
        assert!(
            (output[0] - expected).abs() < 1.0,
            "expected {expected}, got {}",
            output[0]
        );
    }

    #[test]
    fn gemv_multiple_rows() {
        // 2 rows, each 128 elements
        let blocks = vec![
            make_block(1.0, [0xFF; 16]), // row 0: all +1
            make_block(1.0, [0x00; 16]), // row 1: all -1
        ];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 2];

        gemv_1bit_g128(&blocks, &input, &mut output, 2, 128).expect("gemv should succeed");

        assert!((output[0] - 128.0).abs() < 1.0);
        assert!((output[1] + 128.0).abs() < 1.0);
    }

    #[test]
    fn gemv_dimension_validation() {
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input = vec![1.0f32; 64]; // too small
        let mut output = vec![0.0f32; 1];

        let result = gemv_1bit_g128(&blocks, &input, &mut output, 1, 128);
        assert!(result.is_err());
    }

    #[test]
    fn gemv_not_block_aligned() {
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];

        let result = gemv_1bit_g128(&blocks, &input, &mut output, 1, 100);
        assert!(result.is_err());
    }
}
