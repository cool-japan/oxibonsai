//! Integration tests for [`oxibonsai_kernels::gemm_f32`], the blocked, Rayon
//! parallel dense FP32 GEMM.
//!
//! The contract under test is that every element is **bit-identical** to what
//! [`oxibonsai_kernels::gemv_f32`] produces for that row of `a` (plus the bias,
//! added to the finished dot product): the tiling, the parallel split and the
//! thread count only decide where and when a value is computed, never what it
//! is. So the oracle throughout is the per-row `gemv_f32` loop the GEMM
//! replaces, compared with `to_bits()`.
//!
//! Every output buffer starts filled with a sentinel bit pattern, so an
//! element the kernel forgot to write shows up as a mismatch, not as a
//! plausible zero.
//!
//! The Rayon pool is pinned explicitly (`with_pool`) wherever the *path* a
//! call takes matters, so the coverage does not depend on how many cores the
//! test machine has.

#![cfg(not(target_arch = "wasm32"))]

use oxibonsai_kernels::gemm_f32::PAR_MIN_MACS;
use oxibonsai_kernels::{
    cpu_kernel_tier, dot_f32, gemm_f32, gemv_f32, KernelDispatcher, KernelError, KernelTier,
};

/// A bit pattern no computation here produces: a quiet NaN with a payload.
const SENTINEL_BITS: u32 = 0x7FC0_DEAD;

fn sentinel() -> f32 {
    f32::from_bits(SENTINEL_BITS)
}

/// Deterministic pseudo-random values in `[-0.5, 0.5)`.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 40) as u32 as f32) / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// Run `f` on a Rayon pool of exactly `threads` threads.
fn with_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build a rayon pool")
        .install(f)
}

/// The oracle: `gemv_f32` once per row of `a`.
fn per_row_gemv(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        gemv_f32(
            w,
            &a[i * k..(i + 1) * k],
            &mut out[i * n..(i + 1) * n],
            n,
            k,
        )
        .expect("per-row gemv on well-formed shapes");
    }
    out
}

/// The oracle with a bias: each finished GEMV element plus `bias[j]`.
fn add_bias(rows: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
    rows.iter()
        .enumerate()
        .map(|(idx, &value)| value + bias[idx % n])
        .collect()
}

fn assert_bit_identical(context: &str, got: &[f32], want: &[f32], n: usize) {
    assert_eq!(got.len(), want.len(), "{context}: output length");
    if let Some(idx) = got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits())
    {
        panic!(
            "{context}: first mismatch at row {} column {}: got {} ({:#010x}), want {} ({:#010x})",
            idx / n,
            idx % n,
            got[idx],
            got[idx].to_bits(),
            want[idx],
            want[idx].to_bits()
        );
    }
}

/// Two values agree when their bits do, or both are NaN (a NaN's payload is
/// whatever the hardware propagates and is not part of the contract).
fn same_or_both_nan(g: f32, w: f32) -> bool {
    g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan())
}

/// `gemm_f32` into a sentinel-filled buffer of exactly `m * n`.
fn run_gemm(a: &[f32], w: &[f32], bias: Option<&[f32]>, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![sentinel(); m * n];
    gemm_f32(a, w, bias, m, k, n, &mut out).expect("gemm_f32 on well-formed shapes");
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// What it computes
// ─────────────────────────────────────────────────────────────────────────────

/// A product small enough to check by hand (every value an exact integer in
/// `f32`), so the layout is pinned independently of any other kernel: `a` is
/// `2 × 3` row-major, `w` is `4 × 3` row-major (one weight row per output
/// feature), the output is `2 × 4`, and the bias is per output column.
#[test]
fn a_hand_computed_product_pins_the_layout_and_the_bias() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = [
        1.0f32, 0.0, 0.0, // picks a[.][0]
        0.0, 1.0, 0.0, // picks a[.][1]
        0.0, 0.0, 1.0, // picks a[.][2]
        1.0, 1.0, 1.0, // the row sum
    ];
    let plain = run_gemm(&a, &w, None, 2, 3, 4);
    assert_eq!(plain, [1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    let bias = [10.0f32, 20.0, 30.0, 40.0];
    let biased = run_gemm(&a, &w, Some(&bias), 2, 3, 4);
    assert_eq!(biased, [11.0, 22.0, 33.0, 46.0, 14.0, 25.0, 36.0, 55.0]);
}

/// Against an independent oracle — the same product accumulated in `f64` —
/// so a defect the per-row `gemv_f32` shares with the GEMM (a wrong dot, a
/// swapped operand) cannot hide behind their agreement. The tolerance is the
/// `f32` rounding of a length-`k` sum, not a slack that could mask a bug.
#[test]
fn the_result_is_the_matrix_product_to_f32_rounding() {
    let (m, k, n) = (19usize, 300usize, 131usize);
    let a = values(m * k, 61);
    let w = values(n * k, 62);
    let bias = values(n, 63);
    let got = run_gemm(&a, &w, Some(&bias), m, k, n);
    for i in 0..m {
        for j in 0..n {
            let exact: f64 = (0..k)
                .map(|p| f64::from(a[i * k + p]) * f64::from(w[j * k + p]))
                .sum::<f64>()
                + f64::from(bias[j]);
            let error = (f64::from(got[i * n + j]) - exact).abs();
            assert!(
                error <= 1e-4,
                "element ({i}, {j}): got {}, exact {exact}, error {error}",
                got[i * n + j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-identity to the per-row GEMV
// ─────────────────────────────────────────────────────────────────────────────

/// The specified matrix — every `m` in {1, 2, 3, 8, 9, 64, 65} against every
/// `k` in {1, 16, 1152, 4304} and `n` in {1, 5, 1152}, with and without a bias
/// — plus `k = 13` and `k = 1155`, the two lengths that are not a multiple of
/// the eight accumulator lanes and so exercise the remainder loop (the
/// specified `k`s other than 1 are all multiples of eight). Run on a pinned
/// four-thread pool so the parallel path is the one exercised for the big
/// shapes on any host.
///
/// Rows of `a` are independent, so one oracle computed for the largest `m`
/// serves every smaller `m` as a prefix.
#[test]
fn matches_the_per_row_gemv_across_the_specified_shape_matrix() {
    const MS: [usize; 7] = [1, 2, 3, 8, 9, 64, 65];
    const KS: [usize; 6] = [1, 13, 16, 1152, 1155, 4304];
    const NS: [usize; 3] = [1, 5, 1152];
    let m_max = MS[MS.len() - 1];
    with_pool(4, || {
        for &k in &KS {
            let a = values(m_max * k, 0xA000 + k as u64);
            for &n in &NS {
                let w = values(n * k, 0xB000 + (k * 7 + n) as u64);
                let bias = values(n, 0xC000 + n as u64);
                let oracle = per_row_gemv(&a, &w, m_max, k, n);
                let oracle_biased = add_bias(&oracle, &bias, n);
                for &m in &MS {
                    let context = format!("m={m} k={k} n={n}");
                    let got = run_gemm(&a[..m * k], &w, None, m, k, n);
                    assert_bit_identical(&context, &got, &oracle[..m * n], n);
                    let got = run_gemm(&a[..m * k], &w, Some(&bias), m, k, n);
                    assert_bit_identical(
                        &format!("{context} with bias"),
                        &got,
                        &oracle_biased[..m * n],
                        n,
                    );
                }
            }
        }
    });
}

/// Shapes chosen around the macro tile (8 rows of `a` × 64 rows of `w`) and
/// the register tiles (4×2, 2×4, 1×8): every remainder a block can leave, in
/// both directions, on the sequential path and on the parallel one.
#[test]
fn matches_the_per_row_gemv_on_partial_tiles_and_odd_shapes() {
    let ms = [
        1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 23, 24, 25, 33,
    ];
    let ks = [2usize, 7, 9, 17, 100];
    let ns = [2usize, 3, 7, 8, 9, 63, 64, 65, 66, 71, 127, 128, 129, 200];
    for threads in [1usize, 4] {
        with_pool(threads, || {
            for &k in &ks {
                let a = values(33 * k, 0xD000 + k as u64);
                for &n in &ns {
                    let w = values(n * k, 0xE000 + (k * 31 + n) as u64);
                    let bias = values(n, 0xF000 + n as u64);
                    let oracle = per_row_gemv(&a, &w, 33, k, n);
                    let oracle_biased = add_bias(&oracle, &bias, n);
                    for &m in &ms {
                        let context = format!("threads={threads} m={m} k={k} n={n}");
                        let got = run_gemm(&a[..m * k], &w, None, m, k, n);
                        assert_bit_identical(&context, &got, &oracle[..m * n], n);
                        let got = run_gemm(&a[..m * k], &w, Some(&bias), m, k, n);
                        assert_bit_identical(
                            &format!("{context} with bias"),
                            &got,
                            &oracle_biased[..m * n],
                            n,
                        );
                    }
                }
            }
        });
    }
}

/// A shape large enough that the split by macro tile is many-to-one with the
/// pool (hundreds of blocks), and awkward in both directions.
#[test]
fn a_many_block_parallel_call_is_bit_identical() {
    let (m, k, n) = (131usize, 517usize, 389usize);
    assert!(m * n * k >= PAR_MIN_MACS, "this shape must take the pool");
    let a = values(m * k, 1);
    let w = values(n * k, 2);
    let bias = values(n, 3);
    let oracle = per_row_gemv(&a, &w, m, k, n);
    let oracle_biased = add_bias(&oracle, &bias, n);
    with_pool(8, || {
        assert_bit_identical("no bias", &run_gemm(&a, &w, None, m, k, n), &oracle, n);
        assert_bit_identical(
            "bias",
            &run_gemm(&a, &w, Some(&bias), m, k, n),
            &oracle_biased,
            n,
        );
    });
}

/// With no input features every dot product is empty: the result is `0.0`
/// (plus the bias), and `w` and `a` may be empty.
#[test]
fn zero_input_features_give_zeros_plus_the_bias() {
    let (m, n) = (3usize, 5usize);
    let plain = run_gemm(&[], &[], None, m, 0, n);
    assert!(plain.iter().all(|v| v.to_bits() == 0.0f32.to_bits()));
    let bias = values(n, 9);
    let biased = run_gemm(&[], &[], Some(&bias), m, 0, n);
    for (idx, &value) in biased.iter().enumerate() {
        assert_eq!(value.to_bits(), (0.0f32 + bias[idx % n]).to_bits());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bias
// ─────────────────────────────────────────────────────────────────────────────

/// The bias is one addition to the finished dot product — not a seed for the
/// accumulators, which would round differently (and turn `-0.0 + -0.0` into
/// `-0.0` where `dot + bias` gives `+0.0`).
#[test]
fn the_bias_is_added_to_the_finished_dot_product() {
    let (m, k, n) = (9usize, 21usize, 70usize);
    let a = values(m * k, 11);
    let w = values(n * k, 12);
    let mut bias = values(n, 13);
    // Values that expose an accumulator-seeding implementation: a huge bias
    // (absorbs the dot product's low bits when added first) and both zeros.
    bias[0] = 1.0e9;
    bias[1] = -1.0e9;
    bias[2] = -0.0;
    bias[3] = 0.0;
    let got = run_gemm(&a, &w, Some(&bias), m, k, n);
    for i in 0..m {
        for j in 0..n {
            let dot = dot_f32(&a[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
            assert_eq!(
                got[i * n + j].to_bits(),
                (dot + bias[j]).to_bits(),
                "element ({i}, {j})"
            );
        }
    }

    // All-zero inputs make every dot exactly `+0.0`, so the sum is the bias
    // itself except that `+0.0 + -0.0` is `+0.0`.
    let zeros = vec![0.0f32; m * k];
    let got = run_gemm(&zeros, &w, Some(&bias), m, k, n);
    for i in 0..m {
        assert_eq!(got[i * n + 2].to_bits(), 0.0f32.to_bits(), "-0.0 bias");
        assert_eq!(got[i * n + 1].to_bits(), (-1.0e9f32).to_bits());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors and degenerate shapes
// ─────────────────────────────────────────────────────────────────────────────

/// `m == 0` is a no-op that reports success and inspects nothing else: no
/// operand is required to be the right length, and nothing is written.
#[test]
fn m_zero_is_a_no_op_whatever_the_other_arguments_are() {
    let mut out = [sentinel(); 4];
    gemm_f32(&[], &[], None, 0, 7, 3, &mut out).expect("m = 0");
    gemm_f32(
        &[1.0],
        &[2.0; 3],
        Some(&[]),
        0,
        usize::MAX,
        usize::MAX,
        &mut [],
    )
    .expect("m = 0");
    assert!(out.iter().all(|v| v.to_bits() == SENTINEL_BITS));
}

/// Every length and overflow arm returns its own typed error, names the
/// offending operand, and leaves `out` untouched. None of them panics.
#[test]
fn every_error_arm_is_typed_and_leaves_the_output_untouched() {
    let (m, k, n) = (3usize, 5usize, 4usize);
    let a = values(m * k, 1);
    let w = values(n * k, 2);
    let bias = values(n, 3);
    let untouched = |out: &[f32]| out.iter().all(|v| v.to_bits() == SENTINEL_BITS);

    // input too short
    let mut out = vec![sentinel(); m * n];
    match gemm_f32(&a[..m * k - 1], &w, None, m, k, n, &mut out) {
        Err(KernelError::NamedDimensionMismatch {
            name: "input",
            expected,
            got,
        }) => assert_eq!((expected, got), (m * k, m * k - 1)),
        other => panic!("input too short: {other:?}"),
    }
    assert!(untouched(&out));

    // weights too short
    match gemm_f32(&a, &w[..n * k - 1], None, m, k, n, &mut out) {
        Err(KernelError::NamedDimensionMismatch {
            name: "weights",
            expected,
            got,
        }) => assert_eq!((expected, got), (n * k, n * k - 1)),
        other => panic!("weights too short: {other:?}"),
    }
    assert!(untouched(&out));

    // bias too short
    match gemm_f32(&a, &w, Some(&bias[..n - 1]), m, k, n, &mut out) {
        Err(KernelError::NamedDimensionMismatch {
            name: "bias",
            expected,
            got,
        }) => assert_eq!((expected, got), (n, n - 1)),
        other => panic!("bias too short: {other:?}"),
    }
    assert!(untouched(&out));

    // output too small
    let mut small = vec![sentinel(); m * n - 1];
    match gemm_f32(&a, &w, None, m, k, n, &mut small) {
        Err(KernelError::NamedBufferTooSmall {
            name: "output",
            needed,
            available,
        }) => assert_eq!((needed, available), (m * n, m * n - 1)),
        other => panic!("output too small: {other:?}"),
    }
    assert!(untouched(&small));

    // Overflowing extents are reported as such (and before any length check).
    let mut none: [f32; 0] = [];
    for (label, mm, kk, nn) in [
        ("m * k", 2usize, usize::MAX, 1usize),
        ("n * k", 1, usize::MAX / 2 + 1, 2),
        ("m * n", usize::MAX / 2 + 1, 0, 2),
    ] {
        match gemm_f32(&[], &[], None, mm, kk, nn, &mut none) {
            Err(KernelError::UnsupportedOperation(message)) => {
                assert!(message.contains("overflows"), "{label}: {message}");
            }
            other => panic!("{label} overflow: {other:?}"),
        }
    }
}

/// Unlike `gemv_f32`, where an empty weight matrix is the deliberate all-zero
/// projection of a weightless model, an empty `w` here is a length mismatch:
/// a batched matmul that quietly returned zeros for a weight matrix that
/// failed to load would hide the bug.
#[test]
fn empty_weights_are_a_length_mismatch_not_the_zero_projection() {
    let a = values(2 * 3, 5);
    let mut out = vec![sentinel(); 2 * 4];
    match gemm_f32(&a, &[], None, 2, 3, 4, &mut out) {
        Err(KernelError::NamedDimensionMismatch {
            name: "weights",
            expected: 12,
            got: 0,
        }) => {}
        other => panic!("empty weights: {other:?}"),
    }
    assert!(out.iter().all(|v| v.to_bits() == SENTINEL_BITS));
    // The GEMV keeps its convention.
    let mut row = vec![sentinel(); 4];
    gemv_f32(&[], &a[..3], &mut row, 4, 3).expect("gemv_f32 zero projection");
    assert!(row.iter().all(|&v| v == 0.0));
}

/// With no output columns there is nothing to compute or write, but `a` is
/// still checked (it is an operand of the call).
#[test]
fn zero_output_columns_write_nothing_and_still_validate_the_input() {
    let a = values(3 * 4, 6);
    let mut out: [f32; 0] = [];
    gemm_f32(&a, &[], None, 3, 4, 0, &mut out).expect("n = 0");
    assert!(matches!(
        gemm_f32(&a[..11], &[], None, 3, 4, 0, &mut out),
        Err(KernelError::NamedDimensionMismatch { name: "input", .. })
    ));
}

/// Longer slices are accepted (the `gemv_f32` convention): only the leading
/// `m * k` / `n * k` / `n` values are read and only the leading `m * n`
/// outputs are written.
#[test]
fn slices_longer_than_the_shapes_are_accepted_and_their_tails_left_alone() {
    let (m, k, n) = (9usize, 13usize, 70usize);
    let a = values(m * k, 21);
    let w = values(n * k, 22);
    let bias = values(n, 23);
    let want = run_gemm(&a, &w, Some(&bias), m, k, n);

    let a_long: Vec<f32> = a.iter().copied().chain([f32::NAN; 17]).collect();
    let w_long: Vec<f32> = w.iter().copied().chain([f32::NAN; 5]).collect();
    let bias_long: Vec<f32> = bias.iter().copied().chain([f32::NAN; 3]).collect();
    let mut out = vec![sentinel(); m * n + 9];
    gemm_f32(&a_long, &w_long, Some(&bias_long), m, k, n, &mut out).expect("longer slices");
    assert_bit_identical("prefix", &out[..m * n], &want, n);
    assert!(
        out[m * n..].iter().all(|v| v.to_bits() == SENTINEL_BITS),
        "the tail of `out` past m * n must not be written"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-finite values
// ─────────────────────────────────────────────────────────────────────────────

/// A NaN in row `i` of `a` poisons output row `i`; a NaN in row `j` of `w`, or
/// in `bias[j]`, poisons output column `j`; every other element is bit-for-bit
/// the clean result. Infinities follow IEEE and agree with the per-row GEMV.
/// Run on both sides of the parallel threshold.
#[test]
fn nan_and_infinity_propagate_by_row_and_column_like_the_per_row_gemv() {
    for (m, k, n, threads) in [(13usize, 37usize, 70usize, 1usize), (40, 300, 130, 4)] {
        with_pool(threads, || {
            let a = values(m * k, 31);
            let w = values(n * k, 32);
            let bias = values(n, 33);
            let clean = run_gemm(&a, &w, Some(&bias), m, k, n);

            let (nan_row, nan_col, nan_bias_col) = (m / 2, n / 3, n - 2);
            let mut a_nan = a.clone();
            a_nan[nan_row * k + k / 2] = f32::NAN;
            let mut w_nan = w.clone();
            w_nan[nan_col * k + 1] = f32::NAN;
            let mut bias_nan = bias.clone();
            bias_nan[nan_bias_col] = f32::NAN;

            let got = run_gemm(&a_nan, &w_nan, Some(&bias_nan), m, k, n);
            for i in 0..m {
                for j in 0..n {
                    let poisoned = i == nan_row || j == nan_col || j == nan_bias_col;
                    let value = got[i * n + j];
                    if poisoned {
                        assert!(value.is_nan(), "({i}, {j}) must be NaN, got {value}");
                    } else {
                        assert_eq!(
                            value.to_bits(),
                            clean[i * n + j].to_bits(),
                            "({i}, {j}) must be untouched by the NaNs elsewhere"
                        );
                    }
                }
            }

            // Infinities: same result as the per-row GEMV (plus bias) on every
            // element; an infinity meeting its opposite is NaN in both.
            let mut a_inf = a.clone();
            a_inf[3 * k] = f32::INFINITY;
            a_inf[(m - 1) * k + 2] = f32::NEG_INFINITY;
            let mut w_inf = w.clone();
            w_inf[5 * k + 2] = f32::INFINITY;
            w_inf[(n - 1) * k] = f32::NEG_INFINITY;
            let oracle = add_bias(&per_row_gemv(&a_inf, &w_inf, m, k, n), &bias, n);
            let got = run_gemm(&a_inf, &w_inf, Some(&bias), m, k, n);
            for (idx, (&g, &o)) in got.iter().zip(&oracle).enumerate() {
                assert!(
                    same_or_both_nan(g, o),
                    "infinity case, element ({}, {}): got {g}, want {o}",
                    idx / n,
                    idx % n
                );
            }
            assert!(got.iter().any(|v| v.is_infinite()), "no infinity survived");
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread-count independence and the parallel threshold
// ─────────────────────────────────────────────────────────────────────────────

/// One thread, two, an odd count that splits the blocks differently, many, and
/// the default pool must all produce the same bits — on a shape below the
/// parallel threshold and on one above it (so both code paths run under every
/// pool size).
#[test]
fn results_do_not_depend_on_the_thread_count() {
    let shapes = [(9usize, 64usize, 70usize), (40, 300, 130), (65, 257, 193)];
    assert!(
        shapes[0].0 * shapes[0].1 * shapes[0].2 < PAR_MIN_MACS,
        "the first shape must stay on the calling thread"
    );
    for &(m, k, n) in &shapes[1..] {
        assert!(m * n * k >= PAR_MIN_MACS, "{m}x{k}x{n} must take the pool");
    }
    for (m, k, n) in shapes {
        let a = values(m * k, 41);
        let w = values(n * k, 42);
        let bias = values(n, 43);
        let oracle = add_bias(&per_row_gemv(&a, &w, m, k, n), &bias, n);
        let default_pool = run_gemm(&a, &w, Some(&bias), m, k, n);
        assert_bit_identical(
            &format!("default pool, {m}x{k}x{n}"),
            &default_pool,
            &oracle,
            n,
        );
        for threads in [1usize, 2, 3, 8] {
            let got = with_pool(threads, || run_gemm(&a, &w, Some(&bias), m, k, n));
            assert_bit_identical(
                &format!("{threads} thread(s), {m}x{k}x{n}"),
                &got,
                &default_pool,
                n,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// `KernelDispatcher::gemm_f32` is the free function on every tier —
/// including the GPU tier, which has no dense FP32 GEMM route of its own and
/// runs the CPU kernel — and agrees with `KernelDispatcher::gemv_f32` per row.
#[test]
fn the_dispatcher_entry_runs_the_same_kernel_on_every_tier() {
    let (m, k, n) = (17usize, 45usize, 133usize);
    let a = values(m * k, 51);
    let w = values(n * k, 52);
    let bias = values(n, 53);
    let want = run_gemm(&a, &w, Some(&bias), m, k, n);

    let tiers = [
        KernelTier::Reference,
        cpu_kernel_tier(),
        // The GPU tier exists only in a `gpu` build; it must still answer
        // through the CPU kernel.
        #[cfg(feature = "gpu")]
        KernelTier::Gpu,
    ];
    for tier in tiers {
        // `with_tier` clamps a tier this CPU cannot run down to one it can.
        let dispatcher = KernelDispatcher::with_tier(tier);
        let mut got = vec![sentinel(); m * n];
        dispatcher
            .gemm_f32(&a, &w, Some(&bias), m, k, n, &mut got)
            .expect("dispatcher gemm_f32");
        assert_bit_identical(&format!("tier {tier:?}"), &got, &want, n);

        // And per row through the dispatcher's own GEMV.
        let mut per_row = vec![sentinel(); m * n];
        for i in 0..m {
            dispatcher
                .gemv_f32(
                    &w,
                    &a[i * k..(i + 1) * k],
                    &mut per_row[i * n..(i + 1) * n],
                    n,
                    k,
                )
                .expect("dispatcher gemv_f32");
        }
        assert_bit_identical(
            &format!("tier {tier:?} vs per-row dispatcher gemv"),
            &got,
            &add_bias(&per_row, &bias, n),
            n,
        );

        // Typed errors surface through the dispatcher unchanged.
        assert!(matches!(
            dispatcher.gemm_f32(&a[..1], &w, None, m, k, n, &mut got),
            Err(KernelError::NamedDimensionMismatch { name: "input", .. })
        ));
    }
}
