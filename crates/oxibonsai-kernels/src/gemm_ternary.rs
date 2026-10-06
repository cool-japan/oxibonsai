//! Reference (naive) GEMM kernels for ternary TQ2\_0\_g128 and TQ2\_0 formats.
//!
//! Computes `output[m, n] = sum_k(weight[n, k] * input[m, k])` where
//! weights are ternary-quantized. Used for prompt prefill (batch matmul).
//!
//! Each batch row is processed as an independent GEMV call.

use oxibonsai_core::{BlockTQ2_0, BlockTQ2_0_g128, QK_TQ2_0, QK_TQ2_0_G128};

use crate::error::{KernelError, KernelResult};
use crate::gemv_ternary::{gemv_tq2_0, gemv_tq2_0_g128};

// ---------------------------------------------------------------------------
// TQ2_0_g128 — 128 weights per block
// ---------------------------------------------------------------------------

/// Scalar GEMM for TQ2\_0\_g128-quantized weight matrix.
///
/// Computes `output[batch, row] = dot(weight_row, input[batch])` for all
/// batch rows and weight rows.
///
/// - `blocks`: Weight blocks, row-major [n\_rows × blocks\_per\_row].
/// - `input`: Row-major FP32 input matrix [m × k].
/// - `output`: Row-major FP32 output matrix [m × n\_rows].
/// - `m`: Batch/sequence dimension.
/// - `n_rows`: Number of weight matrix rows (output columns).
/// - `k`: Inner dimension (must be divisible by 128).
///
/// # Errors
///
/// Propagates all errors from [`gemv_tq2_0_g128`]:
/// - [`KernelError::NotBlockAligned`] if `k % 128 != 0`.
/// - [`KernelError::DimensionMismatch`] if dimensions mismatch.
/// - [`KernelError::BufferTooSmall`] if any buffer is too small.
pub fn gemm_tq2_0_g128(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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

    for batch in 0..m {
        let input_row = &input[batch * k..(batch + 1) * k];
        let output_row = &mut output[batch * n_rows..(batch + 1) * n_rows];
        gemv_tq2_0_g128(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// TQ2_0 — 256 weights per block
// ---------------------------------------------------------------------------

/// Scalar GEMM for TQ2\_0-quantized weight matrix.
///
/// Computes `output[batch, row] = dot(weight_row, input[batch])` for all
/// batch rows and weight rows.
///
/// - `blocks`: Weight blocks, row-major [n\_rows × blocks\_per\_row].
/// - `input`: Row-major FP32 input matrix [m × k].
/// - `output`: Row-major FP32 output matrix [m × n\_rows].
/// - `m`: Batch/sequence dimension.
/// - `n_rows`: Number of weight matrix rows (output columns).
/// - `k`: Inner dimension (must be divisible by 256).
///
/// # Errors
///
/// Propagates all errors from [`gemv_tq2_0`]:
/// - [`KernelError::NotBlockAligned`] if `k % 256 != 0`.
/// - [`KernelError::DimensionMismatch`] if dimensions mismatch.
/// - [`KernelError::BufferTooSmall`] if any buffer is too small.
pub fn gemm_tq2_0(
    blocks: &[BlockTQ2_0],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    if !k.is_multiple_of(QK_TQ2_0) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0,
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

    for batch in 0..m {
        let input_row = &input[batch * k..(batch + 1) * k];
        let output_row = &mut output[batch * n_rows..(batch + 1) * n_rows];
        gemv_tq2_0(blocks, input_row, output_row, n_rows, k)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv_ternary::gemv_tq2_0_g128;
    use half::f16;

    fn make_g128_block(scale: f32, qs: [u8; 32]) -> BlockTQ2_0_g128 {
        BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(scale),
        }
    }

    /// gemm output must match 4 individual gemv calls for each batch row.
    ///
    /// m=4, n_rows=2, k=128.
    #[test]
    fn gemm_tq2_0_g128_matches_gemv() {
        // 2 rows of weights, k=128 → 2 blocks total
        let blocks = vec![
            make_g128_block(1.0, [0xAA; 32]), // row 0: all +1
            make_g128_block(1.5, [0x00; 32]), // row 1: all -1 (scale 1.5)
        ];
        // 4 different input rows
        let m = 4;
        let n_rows = 2;
        let k = 128;
        let mut input = vec![0.0_f32; m * k];
        for batch in 0..m {
            for j in 0..k {
                input[batch * k + j] = (batch + 1) as f32 * 0.5;
            }
        }

        // Run GEMM
        let mut gemm_out = vec![0.0_f32; m * n_rows];
        gemm_tq2_0_g128(&blocks, &input, &mut gemm_out, m, n_rows, k).expect("gemm should succeed");

        // Run 4 individual GEMVs and compare
        for batch in 0..m {
            let input_row = &input[batch * k..(batch + 1) * k];
            let mut gemv_out = vec![0.0_f32; n_rows];
            gemv_tq2_0_g128(&blocks, input_row, &mut gemv_out, n_rows, k)
                .expect("gemv should succeed");

            for row in 0..n_rows {
                let gemm_val = gemm_out[batch * n_rows + row];
                let gemv_val = gemv_out[row];
                assert!(
                    (gemm_val - gemv_val).abs() < 1e-4,
                    "batch={batch} row={row}: gemm={gemm_val} vs gemv={gemv_val}",
                );
            }
        }
    }

    /// Batch GEMM with all-positive weights: every output row should equal k * d.
    #[test]
    fn gemm_tq2_0_g128_all_positive() {
        let m = 3;
        let n_rows = 4;
        let k = 128;
        let blocks = vec![make_g128_block(1.0, [0xAA; 32]); n_rows];
        let input = vec![1.0_f32; m * k];
        let mut output = vec![0.0_f32; m * n_rows];

        gemm_tq2_0_g128(&blocks, &input, &mut output, m, n_rows, k).expect("gemm should succeed");

        for batch in 0..m {
            for row in 0..n_rows {
                let v = output[batch * n_rows + row];
                assert!(
                    (v - 128.0).abs() < 0.5,
                    "batch={batch} row={row}: expected 128.0, got {v}",
                );
            }
        }
    }

    /// k=100 is not a multiple of 128 → NotBlockAligned error.
    #[test]
    fn gemm_tq2_0_g128_not_block_aligned() {
        let blocks = vec![make_g128_block(1.0, [0xAA; 32])];
        let input = vec![1.0_f32; 100];
        let mut output = vec![0.0_f32; 1];

        let result = gemm_tq2_0_g128(&blocks, &input, &mut output, 1, 1, 100);
        assert!(result.is_err(), "expected NotBlockAligned error");
    }
}

// ---------------------------------------------------------------------------
// K-18 — register-blocked ternary GEMM
// ---------------------------------------------------------------------------
//
// The two functions above are the *reference* shape the K-18 finding names:
// a loop of GEMVs, so the whole quantized weight matrix is streamed and
// re-decoded once per batch row. They stay exactly as they are — every
// cross-tier parity test in the crate (`tests/ternary_cross_tier.rs`,
// `tests/proptest_kernels.rs`) compares a SIMD tier against them, and the
// reference must keep its single, obvious accumulation order.
//
// What follows is the production path: the loop nest inverted to
// `for m_group { for n { for k_block { decode ONCE; FMA against MR rows } } }`,
// so a decoded 128-weight block is consumed by `MR` batch rows before the
// next block is touched. That divides both the decode work and the weight
// traffic by `MR` ([`TERNARY_GEMM_MR`], 8): for Bonsai 2's
// `ffn_up [5120, 17408]` (696 320 blocks = 23 MB) at `m = 512` the weight
// matrix is streamed 64 times instead of 512.
//
// **Bit-exactness (the K-18 verdict's caution).** The finding warns that
// inverting the loop nest changes the per-row reduction order and that the
// result therefore stops being bit-identical. That is true of the *obvious*
// rewrite, but it is avoidable and this implementation avoids it: the
// register-blocked kernels below keep, per `(batch_row, weight_row, block)`,
// exactly the accumulator layout and FMA order of the tier kernel they
// replace — one `float32x4_t` accumulator per block swept over the 32 `qs`
// bytes in order, then `row_sum += d * hsum(acc)` (NEON, mirroring
// `simd_neon::gemm_tq2_0_g128_neon`); one `__m256` accumulator per block
// swept over 16 two-byte chunks (AVX2, mirroring
// `simd_avx2::gemm_tq2_0_g128_avx2`); a strictly sequential scalar sweep
// (reference tier, mirroring `gemv_tq2_0_g128` above). Batch rows never
// interact, so blocking over `m` cannot perturb a row's reduction. The
// speedup comes from *reuse* of the decoded block, not from reassociation —
// and `blocked_tests::ternary_blocked_is_bit_identical_to_the_tier_gemm`
// locks that in — per tier, for every tier this host can construct, via
// `blocked_tests::every_tier_blocked_gemm_is_bit_identical_to_that_tiers_own_gemm`.
// No bit-exactness assertion had to be relaxed *because of the register
// blocking*. One was relaxed in `oxibonsai-model` at the same time and for a
// different reason — `model::types::tests`'s chunked-vs-single-shot prefill
// comparison, where a chunk split that leaves a one-token tail makes
// `forward_prefill_cpu` decline that chunk (`CPU_PREFILL_MIN_TOKENS = 2`) so
// its logits come from the per-token GEMV sweep instead of from this kernel.
// That is a code-path split in the caller, not reassociation here; the
// no-tail case is asserted bit-for-bit again by that module's
// `chunked_prefill_matches_the_single_shot_prefill_bit_for_bit`.
//
// Different *tiers* are, of course, not bit-identical to each other and never
// were: `Reference` against NEON on the same ternary GEMM differs by up to
// 3.6e-7 at k=128 and 1.05e-5 at k=1024, for these blocked kernels and for
// the plain tier kernels alike and by the same amount to every digit
// measured, because a
// 4-lane accumulator plus a horizontal sum is a different (equally valid)
// summation order from a scalar sweep
// (`blocked_tests::blocked_gemm_tiers_agree_within_float_noise`).
//
// The decode itself is the second half of perf-08: every tier's decode
// helper builds a 4-lane `[f32; 4]` on the stack and reloads it
// (`simd_neon::decode_byte_neon_to_f32x4`,
// `simd_avx2::decode_2bytes_avx2_to_f32x8`) — a store-to-load-forwarding
// round trip per 4 weights. [`TERNARY_BYTE_LUT`] replaces it with one
// 4 KiB, L1-resident table lookup per `qs` byte, producing bit-identical
// values (it is generated from the same
// [`oxibonsai_core::ternary_code_to_i8`] table at compile time, and
// `blocked_tests::ternary_byte_lut_matches_the_shared_table_exhaustively`
// proves all 256 x 4 entries agree). The remaining factor — int8
// dot-product (`vdotq_s32` / VNNI) — is K-14 and is
// deliberately NOT attempted here: it would change results bit-for-bit and
// must land as its own selectable tier.

use crate::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_core::ternary_code_to_i8;

/// Register-blocking factor for the ternary GEMM: how many batch rows a
/// decoded weight block is consumed by before the next block is touched.
///
/// 8 keeps the working set inside the NEON register file — 8 `float32x4_t`
/// accumulators plus the decoded weight vector and one input vector, well
/// under AArch64's 32 vector registers — while dividing the decode work and
/// the streamed weight bytes by 8.
pub const TERNARY_GEMM_MR: usize = 8;

/// Compile-time expansion of [`oxibonsai_core::ternary_code_to_i8`]: one
/// packed `qs` byte to its four decoded f32 weights (LSB-first lanes).
///
/// This is not a second decode table — it is *generated from* the shared
/// one by a `const fn`, so K-01's "one table" invariant holds by
/// construction, and
/// `blocked_tests::ternary_byte_lut_matches_the_shared_table_exhaustively`
/// re-checks all 1024 values at test time anyway.
static TERNARY_BYTE_LUT: [[f32; 4]; 256] = build_ternary_byte_lut();

const fn build_ternary_byte_lut() -> [[f32; 4]; 256] {
    let mut table = [[0.0f32; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let b = byte as u8;
        table[byte] = [
            ternary_code_to_i8(b) as f32,
            ternary_code_to_i8(b >> 2) as f32,
            ternary_code_to_i8(b >> 4) as f32,
            ternary_code_to_i8(b >> 6) as f32,
        ];
        byte += 1;
    }
    table
}

/// Where one register-blocked micro-kernel call writes its results.
///
/// Bundled into a struct rather than passed as four more parameters so the
/// micro-kernels stay inside clippy's `too_many_arguments` budget.
#[derive(Debug, Clone, Copy)]
struct TileSpan {
    /// Inner dimension of the GEMM.
    k: usize,
    /// Output row stride (number of weight rows in the full matrix).
    n_rows: usize,
    /// Index of the weight row this call computes.
    ni: usize,
    /// First batch row of the register block.
    m0: usize,
}

/// Validate the shared GEMM preconditions and return `blocks_per_row`.
fn validate_ternary_gemm(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &[f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<usize> {
    if !k.is_multiple_of(QK_TQ2_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK_TQ2_0_G128,
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
    let blocks_per_row = k / QK_TQ2_0_G128;
    let expected_blocks = n_rows * blocks_per_row;
    if blocks.len() < expected_blocks {
        return Err(KernelError::buffer_too_small(
            "blocks",
            expected_blocks,
            blocks.len(),
        ));
    }
    Ok(blocks_per_row)
}

/// Register-blocked ternary GEMM (K-18): `output[m, n] = weight[n, :] . input[m, :]`.
///
/// Numerically identical, bit for bit, to
/// `TernaryKernel::gemm_ternary_g128` on the same dispatcher — see the
/// module-level note above — but it decodes each weight block once per
/// [`TERNARY_GEMM_MR`] batch rows instead of once per batch row.
///
/// `dispatcher` selects which tier's arithmetic is reproduced. Tiers this
/// file cannot mirror without editing a kernel module it does not own
/// (AVX-512) call straight into that module's own (non-register-blocked)
/// GEMM instead, so this entry point is always safe to call, never escapes
/// to the GPU, and never changes results.
///
/// - `blocks`: Weight blocks, row-major [n\_rows x blocks\_per\_row].
/// - `input`: Row-major FP32 input matrix [m x k].
/// - `output`: Row-major FP32 output matrix [m x n\_rows].
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k % 128 != 0`.
/// - [`KernelError::DimensionMismatch`] if `input` is shorter than `m * k`.
/// - [`KernelError::BufferTooSmall`] if `output` or `blocks` are too short.
pub fn gemm_tq2_0_g128_blocked(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_ternary_gemm(blocks, input, output, m, n_rows, k)?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    match effective_blocked_tier(dispatcher) {
        BlockedTier::Scalar => {
            gemm_tq2_blocked_scalar(blocks, input, output, m, n_rows, k, blocks_per_row);
            Ok(())
        }
        #[cfg(target_arch = "aarch64")]
        BlockedTier::Neon => {
            // SAFETY: NEON is baseline on AArch64, and every index the kernel
            // forms is bounded by the validation above.
            unsafe {
                gemm_tq2_blocked_neon(blocks, input, output, m, n_rows, k, blocks_per_row);
            }
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        BlockedTier::Avx2 => {
            // SAFETY: `effective_blocked_tier` only yields `Avx2` when the
            // dispatcher already detected AVX2+FMA; indices are validated.
            unsafe {
                gemm_tq2_blocked_avx2(blocks, input, output, m, n_rows, k, blocks_per_row);
            }
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        BlockedTier::Delegate => {
            // SAFETY: `Delegate` is only produced by `effective_blocked_tier`
            // when `dispatcher.tier()` is already `Avx512`, or by
            // `cpu_blocked_tier` after `is_x86_feature_detected!` confirmed
            // avx512f+avx512bw+avx512vl -- in both cases this host supports
            // the target features `gemm_tq2_0_g128_avx512` requires.
            //
            // This calls the AVX-512 module's own (non-register-blocked)
            // GEMM directly rather than re-entering
            // `dispatcher`'s `TernaryKernel::gemm_ternary_g128`: `dispatcher`
            // can be `KernelTier::Gpu` here (ternary has no GPU GEMM, so
            // `effective_blocked_tier` CPU-falls-back through
            // `cpu_blocked_tier`), and re-entering a GPU-tier dispatcher's
            // own GEMM would route this call back onto the GPU instead of
            // the CPU tier `Delegate` was chosen to reproduce.
            unsafe {
                crate::simd_avx512::gemm_tq2_0_g128_avx512(blocks, input, output, m, n_rows, k)
            }
        }
    }
}

/// Which register-blocked arithmetic a dispatcher's tier maps onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockedTier {
    /// Pure scalar, bit-identical to the reference GEMV sweep.
    Scalar,
    /// NEON, bit-identical to the `simd_neon` tier GEMM.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// AVX2+FMA, bit-identical to the `simd_avx2` tier GEMM.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// No blocked variant for this tier in this crate's owned modules —
    /// call straight into `simd_avx512`'s own (non-blocked) GEMM so results
    /// never change and the call never re-enters a (possibly GPU-tier)
    /// dispatcher.
    ///
    /// Only AVX-512 needs this today (its tier kernels live in
    /// `simd_avx512.rs`), so the variant
    /// exists only where it can be constructed.
    #[cfg(target_arch = "x86_64")]
    Delegate,
}

/// Map a dispatcher tier onto the blocked kernel that reproduces it.
///
/// `KernelTier::Gpu` is mapped through the same CPU-tier selection the
/// dispatcher's own `cpu_gemm_ternary` fallback uses, because a GPU-tier
/// dispatcher runs ternary GEMM on the CPU regardless (there is no ternary
/// GPU GEMM kernel).
pub(crate) fn effective_blocked_tier(dispatcher: &KernelDispatcher) -> BlockedTier {
    match dispatcher.tier() {
        KernelTier::Reference => BlockedTier::Scalar,
        #[cfg(target_arch = "aarch64")]
        KernelTier::Neon => BlockedTier::Neon,
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx2 => BlockedTier::Avx2,
        // AVX-512's tier kernels live in `simd_avx512.rs`, which this
        // package does not own; delegating keeps x86-64 AVX-512 hosts on
        // today's exact numerics instead of silently demoting them.
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx512 => BlockedTier::Delegate,
        #[cfg(feature = "gpu")]
        KernelTier::Gpu => cpu_blocked_tier(),
    }
}

/// The blocked tier a CPU fallback should use, mirroring
/// `KernelDispatcher::cpu_tier`.
#[cfg(feature = "gpu")]
fn cpu_blocked_tier() -> BlockedTier {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            return BlockedTier::Delegate;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return BlockedTier::Avx2;
        }
        BlockedTier::Scalar
    }
    #[cfg(target_arch = "aarch64")]
    {
        BlockedTier::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        BlockedTier::Scalar
    }
}

/// Walk the batch dimension in register blocks, calling `$tile` with the
/// largest supported block that fits the remaining rows.
///
/// Four monomorphizations (8/4/2/1) cover every `m`; each is fully
/// unrolled over its batch rows so the accumulators stay in registers.
macro_rules! for_each_register_block {
    ($m:expr, $tile:ident, $($arg:expr),* $(,)?) => {{
        let mut m0 = 0usize;
        while m0 < $m {
            let remaining = $m - m0;
            if remaining >= TERNARY_GEMM_MR {
                $tile::<TERNARY_GEMM_MR>($($arg,)* m0);
                m0 += TERNARY_GEMM_MR;
            } else if remaining >= 4 {
                $tile::<4>($($arg,)* m0);
                m0 += 4;
            } else if remaining >= 2 {
                $tile::<2>($($arg,)* m0);
                m0 += 2;
            } else {
                $tile::<1>($($arg,)* m0);
                m0 += 1;
            }
        }
    }};
}

// ─── Scalar (reference-tier) register-blocked kernel ────────────────────

/// Scalar register-blocked GEMM, bit-identical to [`gemm_tq2_0_g128`].
#[allow(clippy::too_many_arguments)]
fn gemm_tq2_blocked_scalar(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
) {
    for_each_register_block!(
        m,
        tile_scalar,
        blocks,
        input,
        output,
        n_rows,
        k,
        blocks_per_row
    );
}

#[allow(clippy::too_many_arguments)]
fn tile_scalar<const MR: usize>(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    for ni in 0..n_rows {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let span = TileSpan { k, n_rows, ni, m0 };
        micro_scalar::<MR>(row_blocks, input, output, span);
    }
}

fn micro_scalar<const MR: usize>(
    row_blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    span: TileSpan,
) {
    let mut sums = [0.0f32; MR];
    for (bi, block) in row_blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let inp_base = bi * QK_TQ2_0_G128;
        let mut block_sum = [0.0f32; MR];
        for (byte_idx, &code) in block.qs.iter().enumerate() {
            let lanes = &TERNARY_BYTE_LUT[code as usize];
            for (lane, &w) in lanes.iter().enumerate() {
                let col = inp_base + byte_idx * 4 + lane;
                for (r, acc) in block_sum.iter_mut().enumerate() {
                    *acc += w * input[(span.m0 + r) * span.k + col];
                }
            }
        }
        for (acc, sum) in block_sum.iter().zip(sums.iter_mut()) {
            *sum += d * acc;
        }
    }
    for (r, sum) in sums.iter().enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = *sum;
    }
}

// ─── NEON register-blocked kernel ───────────────────────────────────────

/// Horizontal sum matching `simd_neon::hsum_neon` exactly.
///
/// Replicated here rather than imported because `hsum_neon` is private to
/// `simd_neon`; the two must stay
/// identical, which
/// `blocked_tests::ternary_blocked_is_bit_identical_to_the_tier_gemm`
/// enforces on every run.
///
/// # Safety
/// NEON is baseline on AArch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_neon_blocked(v: std::arch::aarch64::float32x4_t) -> f32 {
    use std::arch::aarch64::{vgetq_lane_f32, vpaddq_f32};
    let pair = vpaddq_f32(v, v);
    let sum = vpaddq_f32(pair, pair);
    vgetq_lane_f32::<0>(sum)
}

/// NEON register-blocked GEMM, bit-identical to
/// `simd_neon::gemm_tq2_0_g128_neon`.
///
/// # Safety
/// NEON is baseline on AArch64; all indices are pre-validated.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tq2_blocked_neon(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
) {
    for_each_register_block!(
        m,
        tile_neon,
        blocks,
        input,
        output,
        n_rows,
        k,
        blocks_per_row
    );
}

/// # Safety
/// See [`gemm_tq2_blocked_neon`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_neon<const MR: usize>(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    for ni in 0..n_rows {
        if ni + 1 < n_rows {
            let next_row = blocks.as_ptr().add((ni + 1) * blocks_per_row) as *const i8;
            // SAFETY: prefetch is a hint; the macro is a no-op off-nightly.
            crate::aarch64_prefetch!(next_row, 0, 3);
        }
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let span = TileSpan { k, n_rows, ni, m0 };
        micro_neon::<MR>(row_blocks, input, output, span);
    }
}

/// # Safety
/// See [`gemm_tq2_blocked_neon`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn micro_neon<const MR: usize>(
    row_blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    span: TileSpan,
) {
    use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let mut sums = [0.0f32; MR];
    let lut = TERNARY_BYTE_LUT.as_ptr() as *const f32;
    for (bi, block) in row_blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let inp_base = bi * QK_TQ2_0_G128;
        let mut acc = [vdupq_n_f32(0.0f32); MR];
        for (byte_idx, &code) in block.qs.iter().enumerate() {
            // One L1-resident table load replaces the per-byte stack round
            // trip the tier decoder does (perf-08); the four lanes are the
            // same values, so the FMA below is bit-identical.
            let w = vld1q_f32(lut.add(code as usize * 4));
            let col = inp_base + byte_idx * 4;
            for (r, a) in acc.iter_mut().enumerate() {
                let x = vld1q_f32(input.as_ptr().add((span.m0 + r) * span.k + col));
                *a = vfmaq_f32(*a, w, x);
            }
        }
        for (a, sum) in acc.iter().zip(sums.iter_mut()) {
            *sum += d * hsum_neon_blocked(*a);
        }
    }
    for (r, sum) in sums.iter().enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = *sum;
    }
}

// ─── AVX2 register-blocked kernel ───────────────────────────────────────

/// Horizontal sum matching `simd_avx2::hsum_avx2` exactly.
///
/// # Safety
/// Requires AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn hsum_avx2_blocked(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
        _mm_movehdup_ps, _mm_movehl_ps,
    };
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

/// AVX2 register-blocked GEMM, bit-identical to
/// `simd_avx2::gemm_tq2_0_g128_avx2`.
///
/// # Safety
/// Requires AVX2+FMA; all indices are pre-validated.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tq2_blocked_avx2(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
) {
    for_each_register_block!(
        m,
        tile_avx2,
        blocks,
        input,
        output,
        n_rows,
        k,
        blocks_per_row
    );
}

/// # Safety
/// See [`gemm_tq2_blocked_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_avx2<const MR: usize>(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    blocks_per_row: usize,
    m0: usize,
) {
    for ni in 0..n_rows {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let span = TileSpan { k, n_rows, ni, m0 };
        micro_avx2::<MR>(row_blocks, input, output, span);
    }
}

/// # Safety
/// See [`gemm_tq2_blocked_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn micro_avx2<const MR: usize>(
    row_blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    output: &mut [f32],
    span: TileSpan,
) {
    use std::arch::x86_64::{
        _mm256_castps128_ps256, _mm256_fmadd_ps, _mm256_insertf128_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_loadu_ps,
    };

    let mut sums = [0.0f32; MR];
    let lut = TERNARY_BYTE_LUT.as_ptr() as *const f32;
    for (bi, block) in row_blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let inp_base = bi * QK_TQ2_0_G128;
        let mut acc = [_mm256_setzero_ps(); MR];
        for chunk in 0..16 {
            let b0 = block.qs[chunk * 2] as usize;
            let b1 = block.qs[chunk * 2 + 1] as usize;
            let lo = _mm_loadu_ps(lut.add(b0 * 4));
            let hi = _mm_loadu_ps(lut.add(b1 * 4));
            let w = _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
            let col = inp_base + chunk * 8;
            for (r, a) in acc.iter_mut().enumerate() {
                let x = _mm256_loadu_ps(input.as_ptr().add((span.m0 + r) * span.k + col));
                *a = _mm256_fmadd_ps(w, x, *a);
            }
        }
        for (a, sum) in acc.iter().zip(sums.iter_mut()) {
            *sum += d * hsum_avx2_blocked(*a);
        }
    }
    for (r, sum) in sums.iter().enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = *sum;
    }
}

// ---------------------------------------------------------------------------
// Tests for the register-blocked path (K-18)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod blocked_tests {
    use super::*;
    use crate::traits::TernaryKernel;
    use half::f16;

    fn block(seed: usize) -> BlockTQ2_0_g128 {
        let mut qs = [0u8; 32];
        for (i, q) in qs.iter_mut().enumerate() {
            // Deliberately includes the reserved `0b11` code so the K-01
            // table's mapping of it to 0 is exercised by every case below.
            *q = ((seed * 31 + i * 17) & 0xFF) as u8;
        }
        BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(0.25 + (seed % 7) as f32 * 0.125),
        }
    }

    fn matrix(n_rows: usize, k: usize) -> Vec<BlockTQ2_0_g128> {
        let bpr = k / QK_TQ2_0_G128;
        (0..n_rows * bpr).map(block).collect()
    }

    fn activations(m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|i| ((i % 251) as f32 * 0.0137) - 1.7)
            .collect()
    }

    /// The compile-time LUT must agree with the one shared decode table
    /// (K-01) for every byte and every lane — no shadow copy.
    #[test]
    fn ternary_byte_lut_matches_the_shared_table_exhaustively() {
        for byte in 0..=255u8 {
            for (lane, &got) in TERNARY_BYTE_LUT[byte as usize].iter().enumerate() {
                let expected = ternary_code_to_i8(byte >> (lane * 2)) as f32;
                assert_eq!(
                    got, expected,
                    "LUT[{byte:#04x}][{lane}] diverged from ternary_code_to_i8"
                );
            }
        }
    }

    /// K-18's caution, answered: the register-blocked GEMM is **bit
    /// identical** to the dispatcher's own GEMM for every batch size,
    /// including the 8/4/2/1 register-block tail splits. Nothing on the
    /// prefill path had to have a bit-exactness assertion relaxed.
    #[test]
    fn ternary_blocked_is_bit_identical_to_the_tier_gemm() {
        let dispatcher = KernelDispatcher::auto_detect();
        for &(m, n_rows, k) in &[
            (1usize, 3usize, 128usize),
            (2, 5, 256),
            (3, 4, 128),
            (7, 9, 384),
            (8, 8, 256),
            (13, 6, 512),
            (17, 3, 128),
        ] {
            let blocks = matrix(n_rows, k);
            let input = activations(m, k);
            let mut expected = vec![0.0f32; m * n_rows];
            let mut got = vec![0.0f32; m * n_rows];
            dispatcher
                .gemm_ternary_g128(&blocks, &input, &mut expected, m, n_rows, k)
                .expect("tier gemm should succeed");
            gemm_tq2_0_g128_blocked(&dispatcher, &blocks, &input, &mut got, m, n_rows, k)
                .expect("blocked gemm should succeed");
            assert_eq!(
                got, expected,
                "blocked ternary gemm diverged at m={m} n_rows={n_rows} k={k}"
            );
        }
    }

    /// The scalar blocked kernel is bit-identical to the reference
    /// loop-of-GEMVs, so the `Reference` tier keeps its exact semantics.
    #[test]
    fn ternary_blocked_scalar_is_bit_identical_to_the_reference() {
        let (m, n_rows, k) = (11usize, 7usize, 256usize);
        let blocks = matrix(n_rows, k);
        let input = activations(m, k);
        let mut expected = vec![0.0f32; m * n_rows];
        let mut got = vec![0.0f32; m * n_rows];
        gemm_tq2_0_g128(&blocks, &input, &mut expected, m, n_rows, k).expect("reference gemm");
        gemm_tq2_blocked_scalar(&blocks, &input, &mut got, m, n_rows, k, k / QK_TQ2_0_G128);
        assert_eq!(got, expected, "scalar blocked gemm diverged from reference");
    }

    /// `m = 1` must still equal a plain GEMV: the register-block tail path
    /// is the one a decode step would take.
    #[test]
    fn ternary_blocked_m1_matches_gemv() {
        let dispatcher = KernelDispatcher::auto_detect();
        let (n_rows, k) = (9usize, 384usize);
        let blocks = matrix(n_rows, k);
        let input = activations(1, k);
        let mut gemv_out = vec![0.0f32; n_rows];
        let mut blocked = vec![0.0f32; n_rows];
        crate::gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut gemv_out, n_rows, k)
            .expect("gemv");
        gemm_tq2_0_g128_blocked(&dispatcher, &blocks, &input, &mut blocked, 1, n_rows, k)
            .expect("blocked gemm");
        for (i, (g, b)) in gemv_out.iter().zip(blocked.iter()).enumerate() {
            assert!(
                (g - b).abs() <= 1e-4 * g.abs().max(1.0),
                "row {i}: gemv={g} blocked={b}"
            );
        }
    }

    /// **Every tier's** register-blocked GEMM is bit-identical to **that
    /// tier's own** GEMM — not just the auto-detected one.
    ///
    /// [`ternary_blocked_is_bit_identical_to_the_tier_gemm`] pins the same
    /// property for `KernelDispatcher::auto_detect()` alone, which is one
    /// tier per host and, on a `gpu` build, is the GPU tier; nothing
    /// exercised `BlockedTier::Scalar` against the scalar tier on a NEON
    /// machine. This sweeps both the `Reference` tier and this host's native
    /// CPU tier ([`crate::cpu_kernel_tier`]) across the `m` values the
    /// batched CPU prefill actually produces, including the
    /// [`TERNARY_GEMM_MR`] tail shapes (`m = 13` is 8 + 4 + 1).
    ///
    /// This is the invariant K-18 could silently have broken. It was
    /// measured holding end to end on the real `Ternary-Bonsai-1.7B.gguf`
    /// — `Reference` vs the auto CPU tier, batched prefill, 64 self-generated
    /// greedy steps on three prompt slots: **exactly 0.0 at every step**,
    /// identical token chains — and asserted it nowhere.
    #[test]
    fn every_tier_blocked_gemm_is_bit_identical_to_that_tiers_own_gemm() {
        let native = crate::cpu_kernel_tier();
        let mut tiers = vec![KernelTier::Reference];
        if native != KernelTier::Reference {
            tiers.push(native);
        }
        for tier in tiers {
            let dispatcher = KernelDispatcher::with_tier(tier);
            for &(m, n_rows, k) in &[
                (1usize, 3usize, 128usize),
                (4, 8, 256),
                (8, 8, 256),
                (13, 6, 512),
                (64, 5, 1024),
            ] {
                let blocks = matrix(n_rows, k);
                let input = activations(m, k);
                let mut expected = vec![0.0f32; m * n_rows];
                let mut got = vec![0.0f32; m * n_rows];
                dispatcher
                    .gemm_ternary_g128(&blocks, &input, &mut expected, m, n_rows, k)
                    .expect("tier gemm should succeed");
                gemm_tq2_0_g128_blocked(&dispatcher, &blocks, &input, &mut got, m, n_rows, k)
                    .expect("blocked gemm should succeed");
                assert_eq!(
                    got, expected,
                    "{tier:?}: blocked ternary gemm diverged from that tier's own gemm \
                     at m={m} n_rows={n_rows} k={k}"
                );
            }
        }
    }

    /// The `Reference` tier and this host's native CPU tier are **not**
    /// bit-identical to each other — this pins how far apart they are.
    ///
    /// Stated explicitly because the model-level cross-tier test
    /// (`oxibonsai-model`'s
    /// `forward_prefill_is_bit_exact_across_the_reference_and_native_cpu_tiers`)
    /// *is* bit-exact, and it would be easy to read that as a claim about
    /// these kernels. It is not: it is a claim about **routing** — a ternary
    /// model never sends its arithmetic through the caller's dispatcher — and
    /// the two tiers' arithmetic genuinely differs, by a 4-lane horizontal
    /// sum's worth of reassociation.
    ///
    /// Measured on this M3 (absolute worst |delta| over the whole output
    /// matrix): 3.58e-7 at `k = 128`, 5.72e-6 at `k = 256`, 7.63e-6 at
    /// `k = 512`, 1.05e-5 at `k = 1024` — the same figures, to every digit
    /// printed, for the blocked kernels and for the plain tier kernels,
    /// which is itself the evidence that register blocking adds nothing to
    /// the gap. The ceiling below is
    /// relative to the output's own scale, so it stays meaningful on a host
    /// whose native tier is AVX2 or AVX-512 rather than NEON.
    #[test]
    fn blocked_gemm_tiers_agree_within_float_noise() {
        let native = crate::cpu_kernel_tier();
        if native == KernelTier::Reference {
            // Nothing to compare: this host has no SIMD tier to differ from.
            return;
        }
        let d_ref = KernelDispatcher::with_tier(KernelTier::Reference);
        let d_native = KernelDispatcher::with_tier(native);
        for &(m, n_rows, k) in &[(1usize, 3usize, 128usize), (13, 6, 512), (64, 5, 1024)] {
            let blocks = matrix(n_rows, k);
            let input = activations(m, k);
            let mut a = vec![0.0f32; m * n_rows];
            let mut b = vec![0.0f32; m * n_rows];
            gemm_tq2_0_g128_blocked(&d_ref, &blocks, &input, &mut a, m, n_rows, k)
                .expect("reference blocked gemm");
            gemm_tq2_0_g128_blocked(&d_native, &blocks, &input, &mut b, m, n_rows, k)
                .expect("native blocked gemm");
            let scale = a
                .iter()
                .chain(b.iter())
                .fold(0.0f32, |acc, v| acc.max(v.abs()))
                .max(1.0);
            let worst = a
                .iter()
                .zip(b.iter())
                .fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
            assert!(
                worst <= 1e-4 * scale,
                "Reference vs {native:?} diverged by {worst:e} (scale {scale:e}) at \
                 m={m} n_rows={n_rows} k={k} — far more than a reassociation difference"
            );
        }
    }

    /// Validation errors are reported before any arithmetic runs.
    #[test]
    fn ternary_blocked_rejects_bad_shapes() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = matrix(1, 128);
        let input = vec![0.0f32; 100];
        let mut out = vec![0.0f32; 1];
        assert!(
            gemm_tq2_0_g128_blocked(&dispatcher, &blocks, &input, &mut out, 1, 1, 100).is_err(),
            "k=100 is not block aligned"
        );
        let input = vec![0.0f32; 128];
        assert!(
            gemm_tq2_0_g128_blocked(&dispatcher, &blocks, &input, &mut out, 4, 1, 128).is_err(),
            "input too short for m=4"
        );
    }

    /// An empty batch or an empty weight matrix is a no-op, not a panic.
    #[test]
    fn ternary_blocked_handles_empty_dimensions() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = matrix(1, 128);
        let input: Vec<f32> = Vec::new();
        let mut out: Vec<f32> = Vec::new();
        gemm_tq2_0_g128_blocked(&dispatcher, &blocks, &input, &mut out, 0, 1, 128)
            .expect("m=0 is a no-op");
        gemm_tq2_0_g128_blocked(&dispatcher, &blocks, &input, &mut out, 0, 0, 128)
            .expect("n_rows=0 is a no-op");
    }
}
