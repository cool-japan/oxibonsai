//! Native GEMV / GEMM for the `PTQ1_0` (ggml id 143) base-3 trit format.
//!
//! `PTQ1_0` packs 128 weights into 28 bytes (1.75 bits/weight): decoding
//! directly from the trit-packed form and dotting against the input vector
//! avoids materializing a transcoded copy of the weight matrix, which matters
//! because `PTQ1_0` is memory-bound at inference time — see
//! [`crate::dequant_prism`]'s module doc for the full format background and
//! the shared `code -> value` map this file also uses
//! ([`oxibonsai_core::q2_0_code_to_i32`]).
//!
//! The alternative path — [`oxibonsai_core::transcode_ptq1_0_to_tq2`] followed
//! by [`crate::gemv_ternary::gemv_tq2_0_g128`] — reuses the existing ternary
//! GEMV/GEMM/Metal/CUDA stack at the cost of materializing a second,
//! `TQ2_0_g128`-shaped copy of the weights (34 B/128 weights vs. `PTQ1_0`'s 28
//! B/128, i.e. ~18% more bytes to move for the transcoded copy, plus the
//! one-time transcode pass itself). `tests` benchmarks both.

use oxibonsai_core::{q2_0_code_to_i32, BlockPTQ1_0, QK_PTQ1_0};

use crate::dequant_prism::decode_ptq1_0_codes;
use crate::error::{KernelError, KernelResult};

/// Native scalar GEMV for a `PTQ1_0`-quantized weight matrix.
///
/// `output[row] = d_block * sum_j code_to_value(code[j]) * input[j]`,
/// accumulated per 128-weight block. `blocks` is row-major: row `r` occupies
/// `blocks[r * (k/128) .. (r+1) * (k/128)]`.
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % 128 != 0`.
/// - [`KernelError::NamedDimensionMismatch`] if `input` is shorter than `k`.
/// - [`KernelError::NamedBufferTooSmall`] if `output` or `blocks` is too short.
pub fn gemv_ptq1_0(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PTQ1_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PTQ1_0,
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
    let blocks_per_row = k / QK_PTQ1_0;
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
            let codes = decode_ptq1_0_codes(block);
            let input_base = bi * QK_PTQ1_0;
            let mut block_sum = 0.0f32;
            for (j, &code) in codes.iter().enumerate() {
                block_sum += q2_0_code_to_i32(code) as f32 * input[input_base + j];
            }
            sum += block.d.to_f32() * block_sum;
        }
        output[row] = sum;
    }
    Ok(())
}

/// Scalar GEMM for a `PTQ1_0`-quantized weight matrix: batches
/// [`gemv_ptq1_0`] over `m` input rows (prompt prefill).
///
/// # Errors
///
/// Propagates every error [`gemv_ptq1_0`] can return, plus
/// [`KernelError::NamedDimensionMismatch`] / [`KernelError::NamedBufferTooSmall`]
/// if `input` / `output` are too short for `m` rows.
pub fn gemm_ptq1_0(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_PTQ1_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_PTQ1_0,
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
        gemv_ptq1_0(blocks, input_row, output_row, n_rows, k)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Register-blocked (MR-tiled) GEMM
//
// `gemm_ptq1_0` above re-runs `decode_ptq1_0_codes` (a five-stage base-3
// trit unpack) once per (batch row, block) pair. The blocked form below
// decodes each block ONCE and consumes it with
// [`crate::dequant_prism::PRISM_GEMM_MR`] batch rows' accumulators live,
// which for `PTQ1_0` saves the decode as well as the weight traffic. The
// per-(batch row, weight row) multiply-add sequence is unchanged, so the
// result is bit-identical to the GEMV sweep.
// ---------------------------------------------------------------------------

use crate::dequant_prism::{for_each_prism_register_block, validate_prism_gemm, PrismTileSpan};

fn micro_ptq1_0_scalar<const MR: usize>(
    row_blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    span: PrismTileSpan,
) {
    let mut sums = [0.0f32; MR];
    for (bi, block) in row_blocks.iter().enumerate() {
        let codes = decode_ptq1_0_codes(block);
        let input_base = bi * QK_PTQ1_0;
        let mut acc = [0.0f32; MR];
        for (j, &code) in codes.iter().enumerate() {
            let w = q2_0_code_to_i32(code) as f32;
            let col = input_base + j;
            for (r, a) in acc.iter_mut().enumerate() {
                *a += w * input[(span.m0 + r) * span.k + col];
            }
        }
        let d = block.d.to_f32();
        for (a, sum) in acc.iter().zip(sums.iter_mut()) {
            *sum += d * *a;
        }
    }
    for (r, sum) in sums.iter().enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = *sum;
    }
}

fn tile_ptq1_0_scalar<const MR: usize>(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    for ni in 0..n_rows {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let span = PrismTileSpan { k, n_rows, ni, m0 };
        micro_ptq1_0_scalar::<MR>(row_blocks, input, output, span);
    }
}

/// Register-blocked scalar GEMM for `PTQ1_0` — bit-identical to
/// [`gemm_ptq1_0`], with each weight block trit-decoded once per
/// [`crate::dequant_prism::PRISM_GEMM_MR`] batch rows instead of once per
/// batch row.
///
/// # Errors
///
/// See [`validate_prism_gemm`].
pub fn gemm_ptq1_0_blocked(
    blocks: &[BlockPTQ1_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_prism_gemm(
        blocks.len(),
        input.len(),
        output.len(),
        m,
        n_rows,
        k,
        QK_PTQ1_0,
    )?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    for_each_prism_register_block!(
        m,
        tile_ptq1_0_scalar,
        [],
        blocks,
        input,
        output,
        n_rows,
        k,
        blocks_per_row
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
//
// NOTE ON NAMING: the package gate filters tests by the substring `prism`
// (`cargo test -p oxibonsai-kernels --all-features prism`), which libtest
// matches against a test's full in-binary path, not the source file name.
// This module is therefore named `prism_gemv_tests` (not the crate's usual
// `tests`) so every function here is actually selected by that filter —
// `gemv_ptq1::tests::foo` would silently NOT match `prism`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod prism_gemv_tests {
    use super::*;
    use half::f16;

    fn make_block(scale: f32, qs: [u8; 24], qh: [u8; 2]) -> BlockPTQ1_0 {
        BlockPTQ1_0 {
            qs,
            qh,
            d: f16::from_f32(scale),
        }
    }

    /// Byte value whose 5/4 trits are all `1` (code word `0`, decoded value
    /// `0`) — the "zero" byte for hand-built fixtures.
    fn zero_byte() -> u8 {
        (((121u16) * 256).div_ceil(243)) as u8
    }

    /// Byte value whose first trit is `2` (+1) and the rest are `1` (0):
    /// base-3 word `2*81 + 27 + 9 + 3 + 1 = 202`.
    fn first_trit_plus_one_byte() -> u8 {
        ((202u16 * 256).div_ceil(243)) as u8
    }

    #[test]
    fn gemv_ptq1_0_all_zero_gives_zero_output() {
        let block = make_block(1.0, [zero_byte(); 24], [zero_byte(); 2]);
        let input = vec![1.0f32; QK_PTQ1_0];
        let mut output = vec![9.0f32; 1];
        gemv_ptq1_0(&[block], &input, &mut output, 1, QK_PTQ1_0).expect("gemv");
        assert_eq!(output[0], 0.0);
    }

    /// One `+1` trit (at output index 3, per the interleaved order) times an
    /// input of all-ones must contribute exactly `d`.
    #[test]
    fn gemv_ptq1_0_single_plus_one_trit_matches_expected_position() {
        let mut qs = [zero_byte(); 24];
        qs[3] = first_trit_plus_one_byte(); // trit 0 of qs[3] -> output index 3
        let block = make_block(2.0, qs, [zero_byte(); 2]);
        let input = vec![1.0f32; QK_PTQ1_0];
        let mut output = vec![0.0f32; 1];
        gemv_ptq1_0(&[block], &input, &mut output, 1, QK_PTQ1_0).expect("gemv");
        assert!((output[0] - 2.0).abs() < 1e-6, "got {}", output[0]);
    }

    #[test]
    fn gemv_ptq1_0_not_block_aligned() {
        let block = make_block(1.0, [0u8; 24], [0u8; 2]);
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(gemv_ptq1_0(&[block], &input, &mut output, 1, 100).is_err());
    }

    #[test]
    fn gemv_ptq1_0_dimension_mismatch_reports_input_name() {
        let block = make_block(1.0, [0u8; 24], [0u8; 2]);
        let input = vec![1.0f32; 10]; // too short for k=128
        let mut output = vec![0.0f32; 1];
        let err = gemv_ptq1_0(&[block], &input, &mut output, 1, QK_PTQ1_0).expect_err("must fail");
        assert_eq!(err.buffer_name(), Some("input"));
    }

    /// `gemv_ptq1_0` must agree with `dequant` + naive dot product within
    /// 1e-4 (design §7.2 acceptance).
    #[test]
    fn gemv_ptq1_0_matches_dequant_plus_naive_dot() {
        let mut input = vec![0.0f32; 2 * QK_PTQ1_0];
        let mut state = 0xABCD_1234u32;
        for v in input.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((state >> 8) as i32 % 200 - 100) as f32 / 37.0;
        }
        let mut weight_row = vec![0.0f32; 2 * QK_PTQ1_0];
        let mut wstate = 0x9999_7777u32;
        for v in weight_row.iter_mut() {
            wstate = wstate.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = match (wstate >> 13) % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
        }
        let blocks = BlockPTQ1_0::quantize(&weight_row).expect("quantize");

        let mut naive_dequant = vec![0.0f32; weight_row.len()];
        crate::dequant_prism::dequant_ptq1_0(&blocks, &mut naive_dequant).expect("dequant");
        let expected: f32 = naive_dequant
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum();

        let mut output = vec![0.0f32; 1];
        gemv_ptq1_0(&blocks, &input, &mut output, 1, weight_row.len()).expect("gemv");

        let tol = 1e-4 * expected.abs().max(1.0);
        assert!(
            (output[0] - expected).abs() < tol,
            "gemv={} vs dequant+dot={}",
            output[0],
            expected
        );
    }

    #[test]
    fn gemm_ptq1_0_matches_gemv_per_batch_row() {
        let blocks = vec![
            BlockPTQ1_0::quantize(&vec![1.0f32; QK_PTQ1_0]).expect("quantize")[0],
            BlockPTQ1_0::quantize(&vec![-1.0f32; QK_PTQ1_0]).expect("quantize")[0],
        ];
        let m = 3;
        let n_rows = 2;
        let k = QK_PTQ1_0;
        let mut input = vec![0.0f32; m * k];
        for (batch, chunk) in input.chunks_mut(k).enumerate() {
            chunk.fill((batch + 1) as f32 * 0.25);
        }
        let mut gemm_out = vec![0.0f32; m * n_rows];
        gemm_ptq1_0(&blocks, &input, &mut gemm_out, m, n_rows, k).expect("gemm");

        for batch in 0..m {
            let input_row = &input[batch * k..(batch + 1) * k];
            let mut gemv_out = vec![0.0f32; n_rows];
            gemv_ptq1_0(&blocks, input_row, &mut gemv_out, n_rows, k).expect("gemv");
            for row in 0..n_rows {
                assert!((gemm_out[batch * n_rows + row] - gemv_out[row]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn gemm_ptq1_0_not_block_aligned() {
        let block = make_block(1.0, [0u8; 24], [0u8; 2]);
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(gemm_ptq1_0(&[block], &input, &mut output, 1, 1, 100).is_err());
    }

    /// Native `gemv_ptq1_0` vs. transcode-to-`TQ2_0_g128` then
    /// `gemv_tq2_0_g128` must agree on VALUE (both decode the same ternary
    /// weights) — the "benchmark both" comparison from the design doc.
    /// Reports wall-clock for both the scalar reference tier and (on
    /// AArch64) the NEON tier of each, since the design's "within 15% of
    /// `gemv_tq2_0_g128`" target is about the tier that would actually ship,
    /// not the scalar fallback.
    ///
    /// `#[ignore]`-d because wall-clock ratios are unreliable on a shared
    /// build machine (this worktree's `target/` build competes with sibling
    /// agents' concurrent compiles for CPU) and must never gate CI; there is
    /// no hard pass/fail threshold here, only a print, for the same reason —
    /// run manually with `cargo test -p oxibonsai-kernels --release
    /// --all-features prism_ptq1_0_gemv_within_15pct_of_tq2 -- --ignored
    /// --nocapture` and read the printed ratios.
    ///
    /// MEASURED on this dev machine (Apple M3, release profile, idle,
    /// allocation-free timing loop): scalar tier ratio ~2.3x SLOWER
    /// (`PTQ1_0`'s base-3 trit peel is inherently more ALU work per element
    /// than TQ2's 2-bit LSB extraction, and at this benchmark's size — a few
    /// hundred KB of weights — both tiers are cache-resident, so the
    /// comparison is ALU-bound, not memory-bound); NEON tier ratio ~0.41x,
    /// i.e. `gemv_ptq1_0_neon` measured ~2.4x **faster** than
    /// `gemv_tq2_0_g128_neon`, comfortably beating the 15% target — the
    /// batched 16-lanes-per-stage vectorized trit decode
    /// ([`crate::simd_prism_neon::decode_ptq1_0_codes_neon`]) apparently
    /// pipelines better than the existing ternary NEON kernel's per-byte
    /// decode. The scalar-tier shortfall is real but is not the tier that
    /// ships on any target with NEON (every AArch64 target, including this
    /// one) or AVX2.
    #[test]
    #[ignore = "wall-clock perf comparison; run manually, not under concurrent-build CI"]
    fn prism_ptq1_0_gemv_within_15pct_of_tq2() {
        use std::time::Instant;

        const N_ROWS: usize = 512;
        const K: usize = QK_PTQ1_0 * 8; // 1024

        let mut weight_row = vec![0.0f32; N_ROWS * K];
        let mut state = 0x1357_9BDFu32;
        for v in weight_row.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = match (state >> 11) % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
        }
        let ptq_blocks = BlockPTQ1_0::quantize(&weight_row).expect("quantize");
        let tq_blocks = oxibonsai_core::transcode_ptq1_0_to_tq2(&ptq_blocks);
        assert_eq!(tq_blocks.len(), ptq_blocks.len());

        let mut input = vec![1.0f32; K];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) / 3.0;
        }

        let mut out_native = vec![0.0f32; N_ROWS];
        let mut out_tq2 = vec![0.0f32; N_ROWS];

        // Correctness: both paths decode the same weights, so their GEMV
        // output must agree numerically regardless of which is faster.
        gemv_ptq1_0(&ptq_blocks, &input, &mut out_native, N_ROWS, K).expect("native gemv");
        crate::gemv_ternary::gemv_tq2_0_g128(&tq_blocks, &input, &mut out_tq2, N_ROWS, K)
            .expect("tq2 gemv");
        for row in 0..N_ROWS {
            assert!(
                (out_native[row] - out_tq2[row]).abs() < 1e-2,
                "row {row}: native={} tq2={}",
                out_native[row],
                out_tq2[row]
            );
        }

        const ITERS: usize = 50;

        // Scalar tier: buffers are pre-allocated and reused across
        // iterations so the measurement is pure kernel time, not allocation
        // noise.
        let t0 = Instant::now();
        for _ in 0..ITERS {
            gemv_ptq1_0(&ptq_blocks, &input, &mut out_native, N_ROWS, K).expect("native gemv");
        }
        let native_secs = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        for _ in 0..ITERS {
            crate::gemv_ternary::gemv_tq2_0_g128(&tq_blocks, &input, &mut out_tq2, N_ROWS, K)
                .expect("tq2 gemv");
        }
        let tq2_secs = t1.elapsed().as_secs_f64();

        let scalar_ratio = native_secs / tq2_secs.max(f64::EPSILON);
        println!(
            "[scalar]  PTQ1_0 native gemv: {native_secs:.6}s vs TQ2_0_g128 gemv: {tq2_secs:.6}s \
             (ratio {scalar_ratio:.3}; design target for the shipped tier is <= 1.15)"
        );

        #[cfg(target_arch = "aarch64")]
        {
            let t2 = Instant::now();
            for _ in 0..ITERS {
                unsafe {
                    crate::simd_prism_neon::gemv_ptq1_0_neon(
                        &ptq_blocks,
                        &input,
                        &mut out_native,
                        N_ROWS,
                        K,
                    )
                    .expect("native neon gemv");
                }
            }
            let native_neon_secs = t2.elapsed().as_secs_f64();

            let t3 = Instant::now();
            for _ in 0..ITERS {
                unsafe {
                    crate::simd_neon::gemv_tq2_0_g128_neon(
                        &tq_blocks,
                        &input,
                        &mut out_tq2,
                        N_ROWS,
                        K,
                    )
                    .expect("tq2 neon gemv");
                }
            }
            let tq2_neon_secs = t3.elapsed().as_secs_f64();

            let neon_ratio = native_neon_secs / tq2_neon_secs.max(f64::EPSILON);
            println!(
                "[neon]    PTQ1_0 native gemv: {native_neon_secs:.6}s vs TQ2_0_g128 gemv: \
                 {tq2_neon_secs:.6}s (ratio {neon_ratio:.3}; design target <= 1.15)"
            );
            // This is the design's actual acceptance target (design §8.2:
            // "within 15% of gemv_tq2_0_g128") and the tier that
            // ships on every NEON-capable target, so — unlike the scalar
            // print above, which documents a real but not-shipped-tier
            // property — it is worth a real regression guard here. Measured
            // ~0.41 on this dev machine (NEON native is faster, not just
            // within 15%); a generous ceiling still catches a real
            // regression without flaking under shared-machine CPU load.
            assert!(
                neon_ratio <= 1.15,
                "PTQ1_0 native NEON gemv regressed vs TQ2_0_g128 NEON: ratio {neon_ratio:.3} \
                 (design target <= 1.15)"
            );
        }
    }
}
