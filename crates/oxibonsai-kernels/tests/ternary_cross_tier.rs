//! Full cross-tier parity matrix for the ternary `TQ2_0_g128` kernels
//! (K-04 / T-04 / T-10).
//!
//! Covers `{dequant, gemv, gemv_prefetch, gemm} x {Reference, NEON, AVX2,
//! AVX-512}`, driven by a shared generator whose `qs` bytes span the full
//! `0..=255` range so every 2-bit code -- including the reserved `0b11` --
//! appears. It also extends the matrix to the `oxibonsai-core` decoders
//! (`BlockTQ2_0_g128::dequant` / `ternary_decode`, and `BlockTQ2_0::dequant`,
//! which is the only public surface of the private `ternary_decode_g256`):
//! K-01's whole point is that five independent decode implementations exist
//! across three crates, and this file is the first thing in the repo that
//! actually calls all of them side by side.
//!
//! Before the K-01 fix, `simd_neon::gemm_tq2_0_g128_neon` (and the AVX2 /
//! AVX-512 dequant/gemv/gemm siblings -- three unmasked inline copies per
//! ISA, not just GEMM) inlined an *unmasked* copy of the decode that mapped
//! `0b11` to `+1` instead of the project-wide `0`, silently disagreeing with
//! every other tier and with the six GPU decoders that hardcode `0b11 -> 0`.
//! `ternary_cross_tier_gemm_neon_reproduces_k01_canary` below reproduces the
//! exact byte pattern from the finding and failed red on the pre-fix tree
//! (`gemm_tq2_0_g128_neon` returned `1.0` where every other tier returned
//! `0.0`); `ternary_cross_tier_gemm_all_tiers_match_reference` failed red
//! too, on ordinary random data, once `qs` bytes could contain `0b11`.
//!
//! x86 arms are gated on `is_x86_feature_detected!`, so on this session's
//! Apple Silicon host they compile -- and are clippy-clean under
//! `--target x86_64-apple-darwin` -- but never execute: Rosetta 2 does not
//! emulate AVX2/AVX-512, so there is no runtime parity evidence for the
//! AVX2/AVX-512 kernels on this host. The NEON arm is gated on
//! `cfg(target_arch = "aarch64")` and does run here.
//!
//! Every `#[test]` fn in this file is named with a `ternary_cross_tier_`
//! prefix so it is picked up by the package gate's release-mode leg
//! (`cargo test ... ternary_cross_tier`), which filters by *test name
//! substring*, not by `--test <binary>` -- see the package's `deviations`
//! for why that distinction mattered here.

use half::f16;
use oxibonsai_core::{ternary_code_to_i8, BlockTQ2_0, BlockTQ2_0_g128, QK_TQ2_0, QK_TQ2_0_G128};
use oxibonsai_kernels::{dequant_ternary, gemm_ternary, gemv_ternary};

// ─────────────────────────── Shared generator ────────────────────────────

/// Deterministic LCG (Numerical Recipes constants) so a failure is
/// reproducible without a `proptest` regression file.
fn lcg_next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// A `BlockTQ2_0_g128` whose `qs` bytes are drawn from the full `0..=255`
/// range, so every 2-bit code (including the reserved `0b11`) appears with
/// high probability in every block.
fn arb_ternary_g128_block(seed: &mut u32) -> BlockTQ2_0_g128 {
    let mut qs = [0u8; 32];
    for b in qs.iter_mut() {
        *b = (lcg_next(seed) >> 16) as u8;
    }
    // Deterministic, finite, non-zero, non-power-of-two scale so a
    // decode-only bug can't hide behind a rounding coincidence.
    let raw_scale = 0.031_25 + ((lcg_next(seed) >> 20) as f32 / 4096.0) * 0.25;
    BlockTQ2_0_g128 {
        qs,
        d: f16::from_f32(raw_scale),
    }
}

fn arb_ternary_g128_blocks(seed: &mut u32, count: usize) -> Vec<BlockTQ2_0_g128> {
    (0..count).map(|_| arb_ternary_g128_block(seed)).collect()
}

fn arb_input(seed: &mut u32, k: usize) -> Vec<f32> {
    (0..k)
        .map(|_| ((lcg_next(seed) >> 8) as f32 / u32::MAX as f32) * 6.0 - 3.0)
        .collect()
}

/// House-style relative tolerance (mirrors `simd_q_std_neon.rs`'s boundary
/// tests): tight near zero, proportionally scaled for larger accumulated
/// sums, so it stays meaningful across the whole `k` range exercised below
/// without being flaky when tiers reduce in a different order.
fn rel_tol(reference: f32) -> f32 {
    1e-4 * reference.abs().max(1.0)
}

/// 8 blocks x 32 `qs` bytes = 256 bytes; byte value == global index, so
/// EVERY `u8` value -- hence every 2-bit code in every one of its 4 lanes --
/// appears at least once, deterministically (not merely "with high
/// probability" like the randomized generator above).
fn exhaustive_byte_coverage_blocks() -> Vec<BlockTQ2_0_g128> {
    let n_blocks = 8;
    let mut blocks = Vec::with_capacity(n_blocks);
    for block_idx in 0..n_blocks {
        let mut qs = [0u8; 32];
        for (j, b) in qs.iter_mut().enumerate() {
            *b = (block_idx * 32 + j) as u8;
        }
        blocks.push(BlockTQ2_0_g128 {
            qs,
            d: f16::from_f32(1.0),
        });
    }
    blocks
}

/// Shapes as `(n_rows, k)`; `k` values are multiples of 128 covering 1..4
/// blocks per row plus a wider row.
const GEMV_SHAPES: &[(usize, usize)] = &[(1, 128), (2, 128), (3, 256), (4, 384), (8, 128)];

/// Batch sizes for the GEMM matrix.
const GEMM_M: &[usize] = &[1, 2, 4];

// ───────────────────────── K-01 canary (exact repro) ──────────────────────

/// `qs[0] = 0b11_10_01_00` (lane0=Neg, lane1=Zero, lane2=Pos, lane3=reserved
/// `0b11`), `d = 1.0`; every other weight is `0x55` (all-Zero) so it
/// contributes nothing. Only `input[3]` (which lane 3 of byte 0 multiplies)
/// is non-zero, isolating the reserved code's contribution.
///
/// Reference gemv/gemm (and, after the K-01 fix, every SIMD tier) must
/// report `0.0` for the whole row (`0b11 -> 0`); before the fix,
/// `gemm_tq2_0_g128_neon` reported `1.0` (lane 3 decoded as `+1`).
#[test]
fn ternary_cross_tier_gemm_neon_reproduces_k01_canary() {
    let mut qs = [0x55u8; 32]; // rest of the block: all-Zero codes
    qs[0] = 0b11_10_01_00; // bits[7:6] (lane 3) = 0b11 reserved
    let block = BlockTQ2_0_g128 {
        qs,
        d: f16::from_f32(1.0),
    };
    let mut input = [0.0f32; QK_TQ2_0_G128];
    input[3] = 1.0; // isolates lane 3's decoded value

    let mut out_ref_gemv = [0.0f32; 1];
    gemv_ternary::gemv_tq2_0_g128(&[block], &input, &mut out_ref_gemv, 1, QK_TQ2_0_G128)
        .expect("reference gemv must succeed");
    assert_eq!(
        out_ref_gemv[0], 0.0,
        "reference gemv: 0b11 must decode to 0"
    );

    let mut out_ref_gemm = [0.0f32; 1];
    gemm_ternary::gemm_tq2_0_g128(&[block], &input, &mut out_ref_gemm, 1, 1, QK_TQ2_0_G128)
        .expect("reference gemm must succeed");
    assert_eq!(
        out_ref_gemm[0], 0.0,
        "reference gemm: 0b11 must decode to 0"
    );

    #[cfg(target_arch = "aarch64")]
    {
        let mut out_neon_gemv = [0.0f32; 1];
        unsafe {
            oxibonsai_kernels::simd_neon::gemv_tq2_0_g128_neon(
                &[block],
                &input,
                &mut out_neon_gemv,
                1,
                QK_TQ2_0_G128,
            )
            .expect("neon gemv must succeed");
        }
        assert_eq!(
            out_neon_gemv[0], out_ref_gemv[0],
            "gemv_tq2_0_g128_neon disagrees with the reference decode of 0b11"
        );

        let mut out_neon_gemm = [0.0f32; 1];
        unsafe {
            oxibonsai_kernels::simd_neon::gemm_tq2_0_g128_neon(
                &[block],
                &input,
                &mut out_neon_gemm,
                1,
                1,
                QK_TQ2_0_G128,
            )
            .expect("neon gemm must succeed");
        }
        assert_eq!(
            out_neon_gemm[0], out_ref_gemm[0],
            "K-01: gemm_tq2_0_g128_neon must agree with the masked reference \
             decode (0b11 -> 0); this is the exact regression the finding \
             reproduced (unmasked inline decode returned 1.0 here)"
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut out_avx2_gemm = [0.0f32; 1];
            unsafe {
                oxibonsai_kernels::simd_avx2::gemm_tq2_0_g128_avx2(
                    &[block],
                    &input,
                    &mut out_avx2_gemm,
                    1,
                    1,
                    QK_TQ2_0_G128,
                )
                .expect("avx2 gemm must succeed");
            }
            assert_eq!(out_avx2_gemm[0], out_ref_gemm[0], "K-01 on AVX2 gemm");
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512_gemm = [0.0f32; 1];
            unsafe {
                oxibonsai_kernels::simd_avx512::gemm_tq2_0_g128_avx512(
                    &[block],
                    &input,
                    &mut out_avx512_gemm,
                    1,
                    1,
                    QK_TQ2_0_G128,
                )
                .expect("avx512 gemm must succeed");
            }
            assert_eq!(out_avx512_gemm[0], out_ref_gemm[0], "K-01 on AVX-512 gemm");
        }
    }
}

// ───────────────────────────────── dequant ────────────────────────────────

/// One (n_blocks, qs-source) case shared by both dequant tests below.
fn check_dequant_all_tiers(blocks: &[BlockTQ2_0_g128]) {
    let needed = blocks.len() * QK_TQ2_0_G128;

    let mut out_ref = vec![0.0f32; needed];
    dequant_ternary::dequant_tq2_0_g128(blocks, &mut out_ref).expect("reference dequant");

    // oxibonsai-core's own decoder (production load path) must be bit-exact
    // vs. the kernels-crate reference -- this is the "two crates" half of
    // K-01's "five independent decoders" claim.
    let mut out_core = vec![0.0f32; needed];
    BlockTQ2_0_g128::dequant(blocks, &mut out_core).expect("core dequant");
    assert_eq!(
        out_ref, out_core,
        "oxibonsai-core::BlockTQ2_0_g128::dequant diverges from the kernels-crate reference"
    );

    // Direct scalar-decoder parity: BlockTQ2_0_g128::ternary_decode must be
    // exactly oxibonsai_core::ternary_code_to_i8 for every lane of every
    // byte actually present (not just a sampled subset).
    for block in blocks {
        for (byte_idx, &byte) in block.qs.iter().enumerate() {
            for lane in 0..4_usize {
                let expected = ternary_code_to_i8(byte >> (lane * 2));
                let actual = BlockTQ2_0_g128::ternary_decode(byte, lane);
                assert_eq!(
                    actual, expected,
                    "BlockTQ2_0_g128::ternary_decode(byte={byte:#04x}, lane={lane}) \
                     [qs byte index {byte_idx}] disagrees with ternary_code_to_i8"
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut out_neon = vec![0.0f32; needed];
        unsafe {
            oxibonsai_kernels::simd_neon::dequant_tq2_0_g128_neon(blocks, &mut out_neon)
                .expect("neon dequant");
        }
        assert_eq!(
            out_ref, out_neon,
            "NEON dequant diverges from the reference decode (must be bit-exact)"
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut out_avx2 = vec![0.0f32; needed];
            unsafe {
                oxibonsai_kernels::simd_avx2::dequant_tq2_0_g128_avx2(blocks, &mut out_avx2)
                    .expect("avx2 dequant");
            }
            assert_eq!(
                out_ref, out_avx2,
                "AVX2 dequant diverges from the reference decode (must be bit-exact)"
            );
        }
        if is_x86_feature_detected!("avx512f") {
            let mut out_avx512 = vec![0.0f32; needed];
            unsafe {
                oxibonsai_kernels::simd_avx512::dequant_tq2_0_g128_avx512(blocks, &mut out_avx512)
                    .expect("avx512 dequant");
            }
            assert_eq!(
                out_ref, out_avx512,
                "AVX-512 dequant diverges from the reference decode (must be bit-exact)"
            );
        }
    }
}

#[test]
fn ternary_cross_tier_dequant_all_tiers_match_reference_randomized() {
    let mut seed = 0xC0FF_EE42u32;
    for &n_blocks in &[1usize, 2, 3, 5, 8] {
        let blocks = arb_ternary_g128_blocks(&mut seed, n_blocks);
        check_dequant_all_tiers(&blocks);
    }
}

/// Deterministic: guarantees every `u8` value (hence every 2-bit code,
/// including `0b11`, in every lane position) is exercised at least once --
/// not merely "with high probability" the way the randomized case is.
#[test]
fn ternary_cross_tier_dequant_all_tiers_match_reference_exhaustive_byte_coverage() {
    let blocks = exhaustive_byte_coverage_blocks();
    // Sanity: the fixture really does contain a 0b11 code (byte 3 has
    // lane0 = 3 & 0b11 = 0b11; more generally every byte b with b & 3 == 3
    // contributes one).
    assert!(
        blocks.iter().any(|b| b
            .qs
            .iter()
            .any(|&byte| (0..4).any(|l| (byte >> (l * 2)) & 0b11 == 0b11))),
        "fixture bug: exhaustive byte coverage must include the reserved 0b11 code"
    );
    check_dequant_all_tiers(&blocks);
}

/// `BlockTQ2_0` (256-wide llama.cpp-compat format) dequant must also match
/// hand-computed expectations from `ternary_code_to_i8` -- this is the only
/// public surface of the private `ternary_decode_g256`, so it's the only
/// way to cover it from outside `oxibonsai-core`.
#[test]
fn ternary_cross_tier_tq2_0_g256_core_dequant_matches_ternary_code_to_i8() {
    let mut qs = [0u8; 64];
    for (i, b) in qs.iter_mut().enumerate() {
        *b = (i * 37 + 11) as u8; // arbitrary full-byte-range coverage
    }
    let scale = 0.75_f32;
    let block = BlockTQ2_0 {
        qs,
        d: f16::from_f32(scale),
    };

    let mut out = vec![0.0f32; QK_TQ2_0];
    BlockTQ2_0::dequant(&[block], &mut out).expect("core g256 dequant");

    for (j, &got) in out.iter().enumerate().take(QK_TQ2_0) {
        let byte_idx = j / 4;
        let lane = j % 4;
        let expected =
            f16::from_f32(scale).to_f32() * ternary_code_to_i8(qs[byte_idx] >> (lane * 2)) as f32;
        assert_eq!(
            got, expected,
            "BlockTQ2_0::dequant[{j}] (byte {byte_idx:#04x}, lane {lane}) disagrees with ternary_code_to_i8"
        );
    }
}

// ────────────────────────────────── gemv ──────────────────────────────────

#[test]
fn ternary_cross_tier_gemv_all_tiers_match_reference() {
    let mut seed = 0x5EED_1234u32;
    for &(n_rows, k) in GEMV_SHAPES {
        let blocks_per_row = k / QK_TQ2_0_G128;
        let blocks = arb_ternary_g128_blocks(&mut seed, n_rows * blocks_per_row);
        let input = arb_input(&mut seed, k);

        let mut out_ref = vec![0.0f32; n_rows];
        gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv");

        #[cfg(target_arch = "aarch64")]
        {
            let mut out_neon = vec![0.0f32; n_rows];
            unsafe {
                oxibonsai_kernels::simd_neon::gemv_tq2_0_g128_neon(
                    &blocks,
                    &input,
                    &mut out_neon,
                    n_rows,
                    k,
                )
                .expect("neon gemv");
            }
            for i in 0..n_rows {
                let tol = rel_tol(out_ref[i]);
                assert!(
                    (out_ref[i] - out_neon[i]).abs() < tol,
                    "NEON gemv row {i} (n_rows={n_rows}, k={k}): ref={}, neon={}, tol={tol}",
                    out_ref[i],
                    out_neon[i]
                );
            }

            let mut out_neon_pf = vec![0.0f32; n_rows];
            unsafe {
                oxibonsai_kernels::simd_neon::gemv_tq2_0_g128_neon_prefetch(
                    &blocks,
                    &input,
                    &mut out_neon_pf,
                    n_rows,
                    k,
                )
                .expect("neon gemv_prefetch");
            }
            for i in 0..n_rows {
                let tol = rel_tol(out_ref[i]);
                assert!(
                    (out_ref[i] - out_neon_pf[i]).abs() < tol,
                    "NEON gemv_prefetch row {i} (n_rows={n_rows}, k={k}): ref={}, neon_pf={}, tol={tol}",
                    out_ref[i],
                    out_neon_pf[i]
                );
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut out_avx2 = vec![0.0f32; n_rows];
                unsafe {
                    oxibonsai_kernels::simd_avx2::gemv_tq2_0_g128_avx2(
                        &blocks,
                        &input,
                        &mut out_avx2,
                        n_rows,
                        k,
                    )
                    .expect("avx2 gemv");
                }
                for i in 0..n_rows {
                    let tol = rel_tol(out_ref[i]);
                    assert!(
                        (out_ref[i] - out_avx2[i]).abs() < tol,
                        "AVX2 gemv row {i} (n_rows={n_rows}, k={k}): ref={}, avx2={}, tol={tol}",
                        out_ref[i],
                        out_avx2[i]
                    );
                }

                let mut out_avx2_pf = vec![0.0f32; n_rows];
                unsafe {
                    oxibonsai_kernels::simd_avx2::gemv_tq2_0_g128_avx2_prefetch(
                        &blocks,
                        &input,
                        &mut out_avx2_pf,
                        n_rows,
                        k,
                    )
                    .expect("avx2 gemv_prefetch");
                }
                for i in 0..n_rows {
                    let tol = rel_tol(out_ref[i]);
                    assert!(
                        (out_ref[i] - out_avx2_pf[i]).abs() < tol,
                        "AVX2 gemv_prefetch row {i} (n_rows={n_rows}, k={k}): ref={}, avx2_pf={}, tol={tol}",
                        out_ref[i],
                        out_avx2_pf[i]
                    );
                }
            }
            if is_x86_feature_detected!("avx512f") {
                let mut out_avx512 = vec![0.0f32; n_rows];
                unsafe {
                    oxibonsai_kernels::simd_avx512::gemv_tq2_0_g128_avx512(
                        &blocks,
                        &input,
                        &mut out_avx512,
                        n_rows,
                        k,
                    )
                    .expect("avx512 gemv");
                }
                for i in 0..n_rows {
                    let tol = rel_tol(out_ref[i]);
                    assert!(
                        (out_ref[i] - out_avx512[i]).abs() < tol,
                        "AVX-512 gemv row {i} (n_rows={n_rows}, k={k}): ref={}, avx512={}, tol={tol}",
                        out_ref[i],
                        out_avx512[i]
                    );
                }

                let mut out_avx512_pf = vec![0.0f32; n_rows];
                unsafe {
                    oxibonsai_kernels::simd_avx512::gemv_tq2_0_g128_avx512_prefetch(
                        &blocks,
                        &input,
                        &mut out_avx512_pf,
                        n_rows,
                        k,
                    )
                    .expect("avx512 gemv_prefetch");
                }
                for i in 0..n_rows {
                    let tol = rel_tol(out_ref[i]);
                    assert!(
                        (out_ref[i] - out_avx512_pf[i]).abs() < tol,
                        "AVX-512 gemv_prefetch row {i} (n_rows={n_rows}, k={k}): ref={}, avx512_pf={}, tol={tol}",
                        out_ref[i],
                        out_avx512_pf[i]
                    );
                }
            }
        }
    }
}

/// Same shapes, but every block's `qs` is the exhaustive-byte-coverage
/// pattern truncated/repeated to the row's block count, guaranteeing `0b11`
/// participates in every GEMV tier's accumulation (not just dequant's).
#[test]
fn ternary_cross_tier_gemv_all_tiers_match_reference_exhaustive_byte_coverage() {
    let exhaustive = exhaustive_byte_coverage_blocks();
    let mut seed = 0xABCD_EF01u32;
    for &(n_rows, k) in GEMV_SHAPES {
        let blocks_per_row = k / QK_TQ2_0_G128;
        let needed = n_rows * blocks_per_row;
        let blocks: Vec<BlockTQ2_0_g128> = (0..needed)
            .map(|i| exhaustive[i % exhaustive.len()])
            .collect();
        let input = arb_input(&mut seed, k);

        let mut out_ref = vec![0.0f32; n_rows];
        gemv_ternary::gemv_tq2_0_g128(&blocks, &input, &mut out_ref, n_rows, k)
            .expect("reference gemv");

        #[cfg(target_arch = "aarch64")]
        {
            let mut out_neon = vec![0.0f32; n_rows];
            unsafe {
                oxibonsai_kernels::simd_neon::gemv_tq2_0_g128_neon(
                    &blocks,
                    &input,
                    &mut out_neon,
                    n_rows,
                    k,
                )
                .expect("neon gemv");
            }
            for i in 0..n_rows {
                let tol = rel_tol(out_ref[i]);
                assert!(
                    (out_ref[i] - out_neon[i]).abs() < tol,
                    "NEON gemv (exhaustive qs) row {i} (n_rows={n_rows}, k={k}): ref={}, neon={}, tol={tol}",
                    out_ref[i],
                    out_neon[i]
                );
            }
        }
    }
}

// ────────────────────────────────── gemm ──────────────────────────────────

#[test]
fn ternary_cross_tier_gemm_all_tiers_match_reference() {
    let mut seed = 0x9E37_79B9u32;
    for &(n_rows, k) in GEMV_SHAPES {
        for &m in GEMM_M {
            let blocks_per_row = k / QK_TQ2_0_G128;
            let blocks = arb_ternary_g128_blocks(&mut seed, n_rows * blocks_per_row);
            let input = arb_input(&mut seed, m * k);

            let mut out_ref = vec![0.0f32; m * n_rows];
            gemm_ternary::gemm_tq2_0_g128(&blocks, &input, &mut out_ref, m, n_rows, k)
                .expect("reference gemm");

            #[cfg(target_arch = "aarch64")]
            {
                let mut out_neon = vec![0.0f32; m * n_rows];
                unsafe {
                    oxibonsai_kernels::simd_neon::gemm_tq2_0_g128_neon(
                        &blocks,
                        &input,
                        &mut out_neon,
                        m,
                        n_rows,
                        k,
                    )
                    .expect("neon gemm");
                }
                for i in 0..(m * n_rows) {
                    let tol = rel_tol(out_ref[i]);
                    assert!(
                        (out_ref[i] - out_neon[i]).abs() < tol,
                        "NEON gemm idx {i} (m={m}, n_rows={n_rows}, k={k}): ref={}, neon={}, tol={tol}",
                        out_ref[i],
                        out_neon[i]
                    );
                }
            }

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let mut out_avx2 = vec![0.0f32; m * n_rows];
                    unsafe {
                        oxibonsai_kernels::simd_avx2::gemm_tq2_0_g128_avx2(
                            &blocks,
                            &input,
                            &mut out_avx2,
                            m,
                            n_rows,
                            k,
                        )
                        .expect("avx2 gemm");
                    }
                    for i in 0..(m * n_rows) {
                        let tol = rel_tol(out_ref[i]);
                        assert!(
                            (out_ref[i] - out_avx2[i]).abs() < tol,
                            "AVX2 gemm idx {i} (m={m}, n_rows={n_rows}, k={k}): ref={}, avx2={}, tol={tol}",
                            out_ref[i],
                            out_avx2[i]
                        );
                    }
                }
                if is_x86_feature_detected!("avx512f") {
                    let mut out_avx512 = vec![0.0f32; m * n_rows];
                    unsafe {
                        oxibonsai_kernels::simd_avx512::gemm_tq2_0_g128_avx512(
                            &blocks,
                            &input,
                            &mut out_avx512,
                            m,
                            n_rows,
                            k,
                        )
                        .expect("avx512 gemm");
                    }
                    for i in 0..(m * n_rows) {
                        let tol = rel_tol(out_ref[i]);
                        assert!(
                            (out_ref[i] - out_avx512[i]).abs() < tol,
                            "AVX-512 gemm idx {i} (m={m}, n_rows={n_rows}, k={k}): ref={}, avx512={}, tol={tol}",
                            out_ref[i],
                            out_avx512[i]
                        );
                    }
                }
            }
        }
    }
}
