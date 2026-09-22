//! Register-blocked 1-bit Q1\_0\_g128 GEMM (K-18).
//!
//! [`crate::gemm::gemm_1bit_g128`] and every SIMD tier's 1-bit GEMM
//! (`simd_neon::gemm_1bit_g128_neon_prefetch`,
//! `simd_avx2::gemm_1bit_g128_avx2_prefetch`,
//! `simd_avx512::gemm_1bit_g128_avx512`) share one shape:
//! `for mi { for ni { decode the whole row } }`. The quantized weight matrix
//! is therefore streamed and re-decoded once per batch row, so a prefill of
//! `m` tokens does `m` full passes over the weights with zero reuse.
//!
//! This module inverts the nest to
//! `for m_group { for ni { for k_block { decode ONCE; FMA against MR rows } } }`.
//! A decoded 128-weight block is consumed by [`ONEBIT_GEMM_MR`] batch rows
//! before the next block is touched, dividing both the sign-decode work and
//! the streamed weight bytes by `MR`.
//!
//! **Bit-exactness.** Each `(batch_row, weight_row)` pair keeps exactly the
//! accumulator layout and FMA order of the tier kernel being replaced —
//! `acc0` over the low nibble of every `qs` byte, `acc1` over the high
//! nibble, both swept in byte order across blocks in order, then
//! `hsum(acc0 + acc1)` (NEON); even blocks into `acc0` and odd blocks into
//! `acc1` in two-block pairs (AVX2); a strictly sequential 128-element
//! scalar sweep (reference). Batch rows never interact, so blocking over
//! `m` cannot perturb a row's reduction, and
//! [`blocked_tests::onebit_blocked_is_bit_identical_to_the_tier_gemm`]
//! asserts byte equality rather than a tolerance. That is what lets
//! [`crate::tiled::gemm_tiled`] and [`crate::parallel::gemm_1bit_g128_par`]
//! adopt this kernel without relaxing a single existing assertion.
//!
//! The sign decode itself is perf-08's other half: `bits_to_signs_neon` /
//! `bits_to_signs_avx2` rebuild a `±1` vector per 4 (resp. 8) weights out
//! of lane inserts, compares and selects. [`ONEBIT_SIGN_LUT`] replaces that
//! with one 8 KiB, L1-resident table load per `qs` byte, producing exactly
//! the same `±1.0` values.

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

use crate::dispatch::KernelDispatcher;
use crate::error::{KernelError, KernelResult};
use crate::gemm_ternary::BlockedTier;

/// Register-blocking factor for the 1-bit GEMM: how many batch rows a
/// decoded weight block is consumed by before the next block is touched.
///
/// The 1-bit kernels carry two accumulators per row (low/high nibble), so 8
/// rows means 16 live vector accumulators plus two sign vectors and a scale
/// — still inside AArch64's 32 vector registers.
///
/// **Measured, not assumed (PERF-CPU-PREFILL verifier pass).** An earlier
/// revision of this doc quoted a specific MR=4-vs-8 speedup and specific
/// ternary sequential/parallel multipliers; neither number is reproducible
/// from the harness it cited, and no run recording them shipped with this
/// package, so they are removed rather than repeated unverified. What *is*
/// reproducible is `parallel_tests::gemm_register_blocking_speedup`'s own
/// table (loop-of-GEMV vs this register-blocked kernel vs the
/// Rayon-parallel driver, at `n_rows=k=1024`). One real run on this M3
/// (`cargo test -p oxibonsai-kernels --release --lib
/// gemm_register_blocking_speedup -- --ignored --nocapture`, 2026-09-22)
/// printed:
///
/// ```text
/// K-18 register-blocked GEMM, n_rows=1024 k=1024, tier=Q1_0_g128 NEON (128-bit)
///           m  loop-of-gemv       blocked   blocked+par     seq x     par x
/// ternary   1       0.821ms       0.563ms       0.103ms     1.46x     7.98x
/// ternary   4       1.981ms       0.190ms       0.317ms    10.42x     6.25x
/// ternary   8       5.088ms       0.333ms       0.319ms    15.26x    15.94x
/// ternary  64      31.544ms       4.833ms       2.964ms     6.53x    10.64x
/// ternary 512     303.548ms      34.313ms      25.619ms     8.85x    11.85x
/// 1-bit     1       0.440ms       0.192ms       0.186ms     2.29x     2.37x
/// 1-bit     4       1.750ms       0.406ms       0.415ms     4.31x     4.22x
/// 1-bit     8       3.763ms       0.914ms       0.903ms     4.12x     4.17x
/// 1-bit    64      11.723ms       5.155ms       4.773ms     2.27x     2.46x
/// 1-bit   512     121.837ms      29.033ms      20.752ms     4.20x     5.87x
/// ```
///
/// This is one run's wall-clock numbers on shared, CI-class hardware —
/// order-of-magnitude evidence that register-blocking (and the Rayon slab
/// split on top of it) win at every `m` tried, for both formats, not a
/// pinned regression contract. Re-run the ignored test for fresh numbers
/// rather than trusting this comment on a different machine. `8` rows was
/// chosen because it keeps every accumulator live in registers without
/// spilling on AArch64 (16 accumulators + 2 sign vectors + a scale, of 32
/// available); a from-scratch MR=4-vs-8 comparison was not part of this run
/// and is not asserted here. The 1-bit gain is modest by construction:
/// `bits_to_signs_*` is far cheaper than the ternary table decode, so
/// there is much less decode work to amortize — the table above shows the
/// ternary column reaching noticeably higher multipliers than the 1-bit
/// one at the same `m`, consistent with that, though the exact ratio
/// varies by `m` and run.
pub const ONEBIT_GEMM_MR: usize = 8;

/// Compile-time sign table: one packed `qs` byte to its eight `±1.0`
/// weights, LSB-first (bit `j` drives lane `j`), matching
/// [`crate::dequant::dequant_1bit_g128`]'s `bit ? +d : -d` exactly.
///
/// Lanes `0..4` reproduce `bits_to_signs_neon(byte, 0)` and lanes `4..8`
/// reproduce `bits_to_signs_neon(byte, 4)`; all eight reproduce
/// `bits_to_signs_avx2(byte)`.
static ONEBIT_SIGN_LUT: [[f32; 8]; 256] = build_onebit_sign_lut();

const fn build_onebit_sign_lut() -> [[f32; 8]; 256] {
    let mut table = [[0.0f32; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut bit = 0usize;
        while bit < 8 {
            table[byte][bit] = if (byte >> bit) & 1 != 0 { 1.0 } else { -1.0 };
            bit += 1;
        }
        byte += 1;
    }
    table
}

/// Where one register-blocked micro-kernel call writes its results.
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
fn validate_onebit_gemm(
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &[f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<usize> {
    if !k.is_multiple_of(QK1_0_G128) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
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
    let blocks_per_row = k / QK1_0_G128;
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

/// Register-blocked 1-bit GEMM (K-18): `output[m, n] = weight[n, :] . input[m, :]`.
///
/// Numerically identical, bit for bit, to
/// [`crate::traits::OneBitKernel::gemm`] on the same dispatcher, but each
/// weight block is decoded once per [`ONEBIT_GEMM_MR`] batch rows instead
/// of once per batch row.
///
/// Tiers this package cannot mirror without editing a kernel module it does
/// not own (AVX-512) call straight into that module's own
/// (non-register-blocked) GEMM instead, so this entry point is always safe
/// to call, never escapes to the GPU, and never changes results.
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
pub fn gemm_1bit_g128_blocked(
    dispatcher: &KernelDispatcher,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let blocks_per_row = validate_onebit_gemm(blocks, input, output, m, n_rows, k)?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    match crate::gemm_ternary::effective_blocked_tier(dispatcher) {
        BlockedTier::Scalar => {
            gemm_1bit_blocked_scalar(blocks, input, output, m, n_rows, k, blocks_per_row);
            Ok(())
        }
        #[cfg(target_arch = "aarch64")]
        BlockedTier::Neon => {
            // SAFETY: NEON is baseline on AArch64; every index the kernel
            // forms is bounded by the validation above.
            unsafe {
                gemm_1bit_blocked_neon(blocks, input, output, m, n_rows, k, blocks_per_row);
            }
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        BlockedTier::Avx2 => {
            // SAFETY: this arm is only reached when AVX2+FMA was detected.
            unsafe {
                gemm_1bit_blocked_avx2(blocks, input, output, m, n_rows, k, blocks_per_row);
            }
            Ok(())
        }
        #[cfg(target_arch = "x86_64")]
        BlockedTier::Delegate => {
            // SAFETY: `Delegate` is only produced by `effective_blocked_tier`
            // when `dispatcher.tier()` is already `Avx512`, or by
            // `cpu_blocked_tier` after `is_x86_feature_detected!` confirmed
            // avx512f+avx512bw+avx512vl -- in both cases this host supports
            // the target features `gemm_1bit_g128_avx512` requires.
            //
            // This calls the AVX-512 module's own (non-register-blocked)
            // GEMM directly rather than re-entering `dispatcher`'s
            // `OneBitKernel::gemm`: `dispatcher` can be `KernelTier::Gpu`
            // here (see `crate::gemm_ternary::effective_blocked_tier`'s
            // `KernelTier::Gpu` arm), and re-entering a GPU-tier
            // dispatcher's own `gemm` would route this call back onto the
            // GPU instead of the CPU tier `Delegate` was chosen to
            // reproduce.
            unsafe {
                crate::simd_avx512::gemm_1bit_g128_avx512(blocks, input, output, m, n_rows, k)
            }
        }
    }
}

/// Walk the batch dimension in register blocks, calling `$tile` with the
/// largest supported block that fits the remaining rows.
macro_rules! for_each_register_block {
    ($m:expr, $tile:ident, $($arg:expr),* $(,)?) => {{
        let mut m0 = 0usize;
        while m0 < $m {
            let remaining = $m - m0;
            if remaining >= ONEBIT_GEMM_MR {
                $tile::<ONEBIT_GEMM_MR>($($arg,)* m0);
                m0 += ONEBIT_GEMM_MR;
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

/// Scalar register-blocked GEMM, bit-identical to
/// [`crate::gemm::gemm_1bit_g128`].
#[allow(clippy::too_many_arguments)]
fn gemm_1bit_blocked_scalar(
    blocks: &[BlockQ1_0G128],
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
    blocks: &[BlockQ1_0G128],
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
    row_blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    span: TileSpan,
) {
    let mut sums = [0.0f32; MR];
    for (bi, block) in row_blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let input_base = bi * QK1_0_G128;
        let mut block_sum = [0.0f32; MR];
        for (byte_idx, &bits) in block.qs.iter().enumerate() {
            let signs = &ONEBIT_SIGN_LUT[bits as usize];
            for (bit, &sign) in signs.iter().enumerate() {
                let col = input_base + byte_idx * 8 + bit;
                for (r, acc) in block_sum.iter_mut().enumerate() {
                    *acc += sign * input[(span.m0 + r) * span.k + col];
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
/// `simd_neon::gemm_1bit_g128_neon_prefetch`.
///
/// # Safety
/// NEON is baseline on AArch64; all indices are pre-validated.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_1bit_blocked_neon(
    blocks: &[BlockQ1_0G128],
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
/// See [`gemm_1bit_blocked_neon`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_neon<const MR: usize>(
    blocks: &[BlockQ1_0G128],
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
/// See [`gemm_1bit_blocked_neon`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn micro_neon<const MR: usize>(
    row_blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    span: TileSpan,
) {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32};

    let lut = ONEBIT_SIGN_LUT.as_ptr() as *const f32;
    let mut acc_lo = [vdupq_n_f32(0.0); MR];
    let mut acc_hi = [vdupq_n_f32(0.0); MR];
    for (bi, block) in row_blocks.iter().enumerate() {
        let scale = vdupq_n_f32(block.d.to_f32());
        let input_base = bi * QK1_0_G128;
        for (byte_idx, &bits) in block.qs.iter().enumerate() {
            // Two L1-resident table loads replace the per-byte lane-insert /
            // compare / select sequence (perf-08); the values are the same
            // `±1.0`, so the FMAs below are bit-identical.
            let signs_lo = vld1q_f32(lut.add(bits as usize * 8));
            let signs_hi = vld1q_f32(lut.add(bits as usize * 8 + 4));
            let col = input_base + byte_idx * 8;
            for r in 0..MR {
                let base = (span.m0 + r) * span.k + col;
                let i_lo = vld1q_f32(input.as_ptr().add(base));
                acc_lo[r] = vfmaq_f32(acc_lo[r], scale, vmulq_f32(signs_lo, i_lo));
                let i_hi = vld1q_f32(input.as_ptr().add(base + 4));
                acc_hi[r] = vfmaq_f32(acc_hi[r], scale, vmulq_f32(signs_hi, i_hi));
            }
        }
    }
    for (r, (lo, hi)) in acc_lo.iter().zip(acc_hi.iter()).enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = hsum_neon_blocked(vaddq_f32(*lo, *hi));
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
/// `simd_avx2::gemm_1bit_g128_avx2_prefetch`.
///
/// # Safety
/// Requires AVX2+FMA; all indices are pre-validated.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_1bit_blocked_avx2(
    blocks: &[BlockQ1_0G128],
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
/// See [`gemm_1bit_blocked_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_avx2<const MR: usize>(
    blocks: &[BlockQ1_0G128],
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
/// See [`gemm_1bit_blocked_avx2`].
///
/// Mirrors the tier kernel's two-block pairing exactly: even blocks
/// accumulate into `acc0`, odd blocks into `acc1`, and a trailing odd block
/// goes to `acc0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn micro_avx2<const MR: usize>(
    row_blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    span: TileSpan,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_setzero_ps,
    };

    let lut = ONEBIT_SIGN_LUT.as_ptr() as *const f32;
    let mut acc0 = [_mm256_setzero_ps(); MR];
    let mut acc1 = [_mm256_setzero_ps(); MR];
    let blocks_per_row = row_blocks.len();
    let pairs = blocks_per_row / 2;
    let remainder = blocks_per_row % 2;

    for pair_idx in 0..pairs {
        let bi0 = pair_idx * 2;
        let block0 = &row_blocks[bi0];
        let block1 = &row_blocks[bi0 + 1];
        let scale0 = _mm256_set1_ps(block0.d.to_f32());
        let scale1 = _mm256_set1_ps(block1.d.to_f32());
        let base0 = bi0 * QK1_0_G128;
        let base1 = (bi0 + 1) * QK1_0_G128;

        for chunk in 0..16 {
            let signs0 = _mm256_loadu_ps(lut.add(block0.qs[chunk] as usize * 8));
            let signs1 = _mm256_loadu_ps(lut.add(block1.qs[chunk] as usize * 8));
            let col0 = base0 + chunk * 8;
            let col1 = base1 + chunk * 8;
            for r in 0..MR {
                let row_base = (span.m0 + r) * span.k;
                let inp0 = _mm256_loadu_ps(input.as_ptr().add(row_base + col0));
                acc0[r] = _mm256_fmadd_ps(scale0, _mm256_mul_ps(signs0, inp0), acc0[r]);
                let inp1 = _mm256_loadu_ps(input.as_ptr().add(row_base + col1));
                acc1[r] = _mm256_fmadd_ps(scale1, _mm256_mul_ps(signs1, inp1), acc1[r]);
            }
        }
    }

    if remainder == 1 {
        let bi = pairs * 2;
        let block = &row_blocks[bi];
        let scale = _mm256_set1_ps(block.d.to_f32());
        let base = bi * QK1_0_G128;
        for chunk in 0..16 {
            let signs = _mm256_loadu_ps(lut.add(block.qs[chunk] as usize * 8));
            let col = base + chunk * 8;
            for (r, a0) in acc0.iter_mut().enumerate() {
                let inp = _mm256_loadu_ps(input.as_ptr().add((span.m0 + r) * span.k + col));
                *a0 = _mm256_fmadd_ps(scale, _mm256_mul_ps(signs, inp), *a0);
            }
        }
    }

    for (r, (a0, a1)) in acc0.iter().zip(acc1.iter()).enumerate() {
        output[(span.m0 + r) * span.n_rows + span.ni] = hsum_avx2_blocked(_mm256_add_ps(*a0, *a1));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod blocked_tests {
    use super::*;
    use crate::traits::OneBitKernel;
    use half::f16;

    fn block(seed: usize) -> BlockQ1_0G128 {
        let mut qs = [0u8; 16];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((seed * 29 + i * 23) & 0xFF) as u8;
        }
        BlockQ1_0G128 {
            d: f16::from_f32(0.125 + (seed % 5) as f32 * 0.25),
            qs,
        }
    }

    fn matrix(n_rows: usize, k: usize) -> Vec<BlockQ1_0G128> {
        let bpr = k / QK1_0_G128;
        (0..n_rows * bpr).map(block).collect()
    }

    fn activations(m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|i| ((i % 197) as f32 * 0.0191) - 1.9)
            .collect()
    }

    /// The compile-time sign table must reproduce the reference dequant's
    /// `bit ? +1 : -1`, LSB-first, for every byte.
    #[test]
    fn onebit_sign_lut_matches_the_reference_dequant_exhaustively() {
        for byte in 0..=255u8 {
            let b = BlockQ1_0G128 {
                d: f16::ONE,
                qs: [byte; 16],
            };
            let mut reference = vec![0.0f32; QK1_0_G128];
            crate::dequant::dequant_1bit_g128(&[b], &mut reference).expect("reference dequant");
            for bit in 0..8usize {
                assert_eq!(
                    ONEBIT_SIGN_LUT[byte as usize][bit], reference[bit],
                    "sign LUT[{byte:#04x}][{bit}] diverged from the reference dequant"
                );
            }
        }
    }

    /// K-18 without a relaxed assertion: the register-blocked 1-bit GEMM is
    /// **bit identical** to the dispatcher's own GEMM at every batch size,
    /// including the 4/2/1 register-block tail splits.
    #[test]
    fn onebit_blocked_is_bit_identical_to_the_tier_gemm() {
        let dispatcher = KernelDispatcher::auto_detect();
        for &(m, n_rows, k) in &[
            (1usize, 3usize, 128usize),
            (2, 5, 256),
            (3, 4, 384),
            (4, 8, 256),
            (7, 6, 512),
            (11, 3, 128),
            (16, 9, 384),
        ] {
            let blocks = matrix(n_rows, k);
            let input = activations(m, k);
            let mut expected = vec![0.0f32; m * n_rows];
            let mut got = vec![0.0f32; m * n_rows];
            dispatcher
                .gemm(&blocks, &input, &mut expected, m, n_rows, k)
                .expect("tier gemm should succeed");
            gemm_1bit_g128_blocked(&dispatcher, &blocks, &input, &mut got, m, n_rows, k)
                .expect("blocked gemm should succeed");
            assert_eq!(
                got, expected,
                "blocked 1-bit gemm diverged at m={m} n_rows={n_rows} k={k}"
            );
        }
    }

    /// The scalar blocked kernel is bit-identical to the reference GEMM, so
    /// the `Reference` tier keeps its exact semantics.
    #[test]
    fn onebit_blocked_scalar_is_bit_identical_to_the_reference() {
        let (m, n_rows, k) = (9usize, 7usize, 384usize);
        let blocks = matrix(n_rows, k);
        let input = activations(m, k);
        let mut expected = vec![0.0f32; m * n_rows];
        let mut got = vec![0.0f32; m * n_rows];
        crate::gemm::gemm_1bit_g128(&blocks, &input, &mut expected, m, n_rows, k)
            .expect("reference gemm");
        gemm_1bit_blocked_scalar(&blocks, &input, &mut got, m, n_rows, k, k / QK1_0_G128);
        assert_eq!(got, expected, "scalar blocked gemm diverged from reference");
    }

    /// Validation runs before any arithmetic.
    #[test]
    fn onebit_blocked_rejects_bad_shapes() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = matrix(1, 128);
        let input = vec![0.0f32; 100];
        let mut out = vec![0.0f32; 1];
        assert!(
            gemm_1bit_g128_blocked(&dispatcher, &blocks, &input, &mut out, 1, 1, 100).is_err(),
            "k=100 is not block aligned"
        );
        let input = vec![0.0f32; 128];
        assert!(
            gemm_1bit_g128_blocked(&dispatcher, &blocks, &input, &mut out, 4, 1, 128).is_err(),
            "input too short for m=4"
        );
        let short_blocks = matrix(1, 128);
        let input = vec![0.0f32; 256];
        let mut out = vec![0.0f32; 2];
        assert!(
            gemm_1bit_g128_blocked(&dispatcher, &short_blocks, &input, &mut out, 1, 2, 128)
                .is_err(),
            "not enough weight blocks for n_rows=2"
        );
    }

    /// An empty batch or weight matrix is a no-op, not a panic.
    #[test]
    fn onebit_blocked_handles_empty_dimensions() {
        let dispatcher = KernelDispatcher::auto_detect();
        let blocks = matrix(1, 128);
        let input: Vec<f32> = Vec::new();
        let mut out: Vec<f32> = Vec::new();
        gemm_1bit_g128_blocked(&dispatcher, &blocks, &input, &mut out, 0, 1, 128)
            .expect("m=0 is a no-op");
        gemm_1bit_g128_blocked(&dispatcher, &blocks, &input, &mut out, 0, 0, 128)
            .expect("n_rows=0 is a no-op");
    }
}
