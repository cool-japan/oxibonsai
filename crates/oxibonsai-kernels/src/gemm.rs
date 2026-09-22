//! Reference (naive) GEMM kernel for Q1\_0\_g128.
//!
//! Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])` where
//! weights are Q1\_0\_g128 packed. This is used for prompt prefill
//! (batch matmul).

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

use crate::error::{KernelError, KernelResult};

/// Fused 1-bit GEMM: `output[m, n] = dot(weight_row_n, input_row_m)`.
///
/// - `blocks`: Weight blocks, row-major. Row `n` starts at `n * blocks_per_row`.
/// - `input`: Row-major FP32 input matrix [m × k].
/// - `output`: Row-major FP32 output matrix [m × n_rows].
/// - `m`: Batch/sequence dimension.
/// - `n_rows`: Number of weight matrix rows (output columns).
/// - `k`: Inner dimension (must be divisible by 128).
pub fn gemm_1bit_g128(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    if input.len() < m * k {
        return Err(KernelError::DimensionMismatch {
            expected: m * k,
            got: input.len(),
        });
    }
    if output.len() < m * n_rows {
        return Err(KernelError::BufferTooSmall {
            needed: m * n_rows,
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

    // For each input row (batch element)
    for mi in 0..m {
        let input_row = &input[mi * k..(mi + 1) * k];

        // For each weight row (output column)
        for ni in 0..n_rows {
            let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
            let mut sum = 0.0f32;

            for (bi, block) in row_blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let input_base = bi * QK1_0_G128;

                let mut block_sum = 0.0f32;
                for j in 0..QK1_0_G128 {
                    let byte_index = j / 8;
                    let bit_offset = j % 8;
                    let sign = if (block.qs[byte_index] >> bit_offset) & 1 != 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    block_sum += sign * input_row[input_base + j];
                }
                sum += d * block_sum;
            }

            output[mi * n_rows + ni] = sum;
        }
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
    fn gemm_single_batch() {
        // m=1, n=1, k=128 — reduces to a GEMV
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];

        gemm_1bit_g128(&blocks, &input, &mut output, 1, 1, 128).expect("gemm should succeed");

        assert!(
            (output[0] - 128.0).abs() < 1.0,
            "expected ~128.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemm_batch_of_two() {
        // m=2, n=1, k=128
        let blocks = vec![make_block(1.0, [0xFF; 16])];
        let mut input = vec![0.0f32; 256];
        // Row 0: all 1.0
        for v in &mut input[..128] {
            *v = 1.0;
        }
        // Row 1: all 2.0
        for v in &mut input[128..] {
            *v = 2.0;
        }

        let mut output = vec![0.0f32; 2];
        gemm_1bit_g128(&blocks, &input, &mut output, 2, 1, 128).expect("gemm should succeed");

        assert!((output[0] - 128.0).abs() < 1.0);
        assert!((output[1] - 256.0).abs() < 1.0);
    }

    #[test]
    fn gemm_multiple_output_rows() {
        // m=1, n=2, k=128
        let blocks = vec![
            make_block(1.0, [0xFF; 16]), // row 0: all +1
            make_block(1.0, [0x00; 16]), // row 1: all -1
        ];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 2];

        gemm_1bit_g128(&blocks, &input, &mut output, 1, 2, 128).expect("gemm should succeed");

        assert!((output[0] - 128.0).abs() < 1.0);
        assert!((output[1] + 128.0).abs() < 1.0);
    }
}
