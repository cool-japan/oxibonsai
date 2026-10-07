//! K-02 / M-01 / K-M3 regression contract for the `simd_float_ops` entry
//! points (`rms_norm_simd`, `silu_simd`, `swiglu_simd`, `rope_apply_simd`).
//!
//! # Why this file must run under `--release`
//!
//! Before the fix, these four functions validated their cross-slice length
//! invariants with `debug_assert!` only, then indexed `weight`/`output`
//! through raw-pointer SIMD loads/stores bounded solely by `input.len()`.
//! `debug_assert!` compiles to nothing in a release build, so a caller
//! passing a too-short `weight`/`output`/`cos_table`/`sin_table` slice would
//! silently read and write out of bounds — undefined behavior — in exactly
//! the build profile OxiBonsai ships. A `cargo test` run in the default
//! (debug) profile cannot observe that defect at all: the same bad call
//! would have tripped the `debug_assert!` and panicked (a passing "test
//! caught a bug" signal that has nothing to do with what actually ships).
//! The fix replaces the `debug_assert!`s with real, always-on `if` checks
//! that return a `KernelError`, so these tests behave identically in debug
//! and release — but they are collected in their own file specifically so
//! the gate can run them under `--release` as an explicit, permanent check
//! that no future change quietly reintroduces a `debug_assert!`-guarded
//! (i.e. release-mode-unchecked) length contract here.
//!
//! Run: `cargo test -p oxibonsai-kernels --release --all-features --test
//! simd_float_ops_contract`.

use oxibonsai_kernels::error::KernelError;
use oxibonsai_kernels::{rms_norm_simd, rope_apply_simd, silu_simd, swiglu_simd};

// ═══════════════════════════════════════════════════════════════════════
//  K-02 / M-01: real length-contract errors, not a debug_assert
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn rms_norm_simd_rejects_mismatched_weight_length() {
    // A weight tensor loaded from a malformed/truncated GGUF file could be
    // shorter than `hidden_size` — this must be a clean `Err`, never an
    // out-of-bounds read of `weight` in the NEON/AVX2 kernels.
    let input = vec![1.0f32; 4096];
    let weight = vec![1.0f32; 8]; // far short of input.len()
    let mut output = vec![0.0f32; 4096];

    let result = rms_norm_simd(&input, &weight, &mut output, 1e-6);
    match result {
        Err(KernelError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, 4096);
            assert_eq!(got, 8);
        }
        other => panic!("expected Err(DimensionMismatch), got {other:?}"),
    }
}

#[test]
fn rms_norm_simd_rejects_short_output() {
    let input = vec![1.0f32; 4096];
    let weight = vec![1.0f32; 4096];
    let mut output = vec![0.0f32; 8]; // far short of input.len()

    let result = rms_norm_simd(&input, &weight, &mut output, 1e-6);
    match result {
        Err(KernelError::BufferTooSmall { needed, available }) => {
            assert_eq!(needed, 4096);
            assert_eq!(available, 8);
        }
        other => panic!("expected Err(BufferTooSmall), got {other:?}"),
    }
}

#[test]
fn silu_simd_rejects_short_output() {
    let input = vec![1.0f32; 17408]; // FFN-intermediate-sized buffer
    let mut output = vec![0.0f32; 4];

    let result = silu_simd(&input, &mut output);
    match result {
        Err(KernelError::BufferTooSmall { needed, available }) => {
            assert_eq!(needed, 17408);
            assert_eq!(available, 4);
        }
        other => panic!("expected Err(BufferTooSmall), got {other:?}"),
    }
}

#[test]
fn swiglu_simd_rejects_mismatched_up_length() {
    let gate = vec![1.0f32; 17408];
    let up = vec![1.0f32; 4]; // far short of gate.len()
    let mut output = vec![0.0f32; 17408];

    let result = swiglu_simd(&gate, &up, &mut output);
    match result {
        Err(KernelError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, 17408);
            assert_eq!(got, 4);
        }
        other => panic!("expected Err(DimensionMismatch), got {other:?}"),
    }
}

#[test]
fn swiglu_simd_rejects_short_output() {
    let gate = vec![1.0f32; 17408];
    let up = vec![1.0f32; 17408];
    let mut output = vec![0.0f32; 4];

    let result = swiglu_simd(&gate, &up, &mut output);
    match result {
        Err(KernelError::BufferTooSmall { needed, available }) => {
            assert_eq!(needed, 17408);
            assert_eq!(available, 4);
        }
        other => panic!("expected Err(BufferTooSmall), got {other:?}"),
    }
}

#[test]
fn rope_apply_simd_rejects_mismatched_cos_table_length() {
    let input = vec![1.0f32; 256]; // head_dim=256 → half_dim=128
    let cos_t = vec![1.0f32; 4]; // far short of half_dim
    let sin_t = vec![0.0f32; 128];
    let mut output = vec![0.0f32; 256];

    let result = rope_apply_simd(&input, &mut output, &cos_t, &sin_t);
    match result {
        Err(KernelError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, 128);
            assert_eq!(got, 4);
        }
        other => panic!("expected Err(DimensionMismatch), got {other:?}"),
    }
}

#[test]
fn rope_apply_simd_rejects_mismatched_sin_table_length() {
    let input = vec![1.0f32; 256];
    let cos_t = vec![1.0f32; 128];
    let sin_t = vec![0.0f32; 4]; // far short of half_dim
    let mut output = vec![0.0f32; 256];

    let result = rope_apply_simd(&input, &mut output, &cos_t, &sin_t);
    match result {
        Err(KernelError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, 128);
            assert_eq!(got, 4);
        }
        other => panic!("expected Err(DimensionMismatch), got {other:?}"),
    }
}

#[test]
fn rope_apply_simd_rejects_short_output() {
    let input = vec![1.0f32; 256];
    let cos_t = vec![1.0f32; 128];
    let sin_t = vec![0.0f32; 128];
    let mut output = vec![0.0f32; 4]; // far short of input.len()

    let result = rope_apply_simd(&input, &mut output, &cos_t, &sin_t);
    match result {
        Err(KernelError::BufferTooSmall { needed, available }) => {
            assert_eq!(needed, 256);
            assert_eq!(available, 4);
        }
        other => panic!("expected Err(BufferTooSmall), got {other:?}"),
    }
}

#[test]
fn valid_calls_still_succeed_after_the_length_contract_change() {
    // The point of K-02/M-01 is to reject bad lengths, not to become
    // stricter than before for well-formed calls — every existing
    // (correctly-shaped) caller must keep working unchanged.
    let input = vec![2.0f32; 64];
    let weight = vec![1.0f32; 64];
    let mut rms_out = vec![0.0f32; 64];
    rms_norm_simd(&input, &weight, &mut rms_out, 1e-6).expect("matching lengths must succeed");

    let mut silu_out = vec![0.0f32; 64];
    silu_simd(&input, &mut silu_out).expect("matching lengths must succeed");

    let up = vec![3.0f32; 64];
    let mut swiglu_out = vec![0.0f32; 64];
    swiglu_simd(&input, &up, &mut swiglu_out).expect("matching lengths must succeed");

    let half = 32;
    let cos_t = vec![1.0f32; half];
    let sin_t = vec![0.0f32; half];
    let mut rope_out = vec![0.0f32; 64];
    rope_apply_simd(&input, &mut rope_out, &cos_t, &sin_t).expect("matching lengths must succeed");
}

// ═══════════════════════════════════════════════════════════════════════
//  K-M3: body/tail rounding consistency (vectorized exp + division)
// ═══════════════════════════════════════════════════════════════════════

/// `silu_simd(x)[i]` must be bit-identical for the same `x` embedded at
/// different offsets and different total buffer lengths — i.e. regardless
/// of whether it falls inside the SIMD "body" (a multiple-of-4 prefix, on
/// NEON) or the previously-scalar "tail" (the last 1-3 elements). Before
/// K-M3, the NEON body approximated the sigmoid reciprocal with
/// `vrecpeq_f32` + two Newton-Raphson steps while the tail used exact
/// scalar division, so the *same* logical value could silently round
/// differently depending only on where in the buffer it landed — a real
/// hazard for a project that gates CPU-vs-Metal determinism on buffer
/// length.
#[test]
fn silu_simd_is_bit_identical_regardless_of_offset_and_length() {
    let probe_values = [0.734_f32, -2.5, 5.0, -0.001, 12.25, -30.0];

    for &x in &probe_values {
        let mut reference: Option<f32> = None;
        for len in 1..=13usize {
            for pos in 0..len {
                let mut input = vec![0.1f32; len];
                input[pos] = x;
                let mut output = vec![0.0f32; len];
                silu_simd(&input, &mut output).expect("matching lengths");
                let got = output[pos];
                match reference {
                    None => reference = Some(got),
                    Some(r) => assert_eq!(
                        got.to_bits(),
                        r.to_bits(),
                        "silu({x}) at len={len} pos={pos} = {got} ({:#x}), \
                         expected bit-identical to the first observation {r} ({:#x})",
                        got.to_bits(),
                        r.to_bits()
                    ),
                }
            }
        }
    }
}

/// Same property as above for `swiglu_simd`'s gate input (the `up` factor
/// is a plain multiply after `silu`, so it shares the identical hazard).
#[test]
fn swiglu_simd_is_bit_identical_regardless_of_offset_and_length() {
    let probe_pairs = [(-1.375_f32, 2.5_f32), (3.0, -0.5), (0.02, 7.0)];

    for &(g, u) in &probe_pairs {
        let mut reference: Option<f32> = None;
        for len in 1..=13usize {
            for pos in 0..len {
                let mut gate = vec![0.5f32; len];
                let up = vec![u; len];
                gate[pos] = g;
                let mut output = vec![0.0f32; len];
                swiglu_simd(&gate, &up, &mut output).expect("matching lengths");
                let got = output[pos];
                match reference {
                    None => reference = Some(got),
                    Some(r) => assert_eq!(
                        got.to_bits(),
                        r.to_bits(),
                        "swiglu({g},{u}) at len={len} pos={pos} = {got}, \
                         expected bit-identical to {r}"
                    ),
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  K-M3: rms_norm / rope body/tail bit-identity (scalar tails)
// ═══════════════════════════════════════════════════════════════════════
//
// `rms_norm_simd` and `rope_apply_simd` finish a non-multiple-of-the-vector
// length with plain scalar code, which must evaluate the vector body's exact
// operation sequence. Lengths 9..=16 give the AVX2 kernels one 8-lane body
// plus every tail size (and a body-only 16), and the NEON kernels two to four
// 4-lane bodies plus their tails: index 0 is always in the body, the last
// index in the tail unless the length is a whole number of vectors.

/// With a uniform `input` every element shares one `inv_rms`, so a uniform
/// `weight` must make the whole output one value: `output[0]` (body) and
/// `output[len - 1]` (tail) bit-identical. Before the fix the AVX2 and NEON
/// tails computed `(weight * input) * inv_rms` against the body's
/// `weight * (input * inv_rms)`, which rounds differently.
#[test]
fn rms_norm_simd_body_and_tail_agree_bitwise_for_uniform_buffers() {
    let probes: [(f32, f32, f32); 6] = [
        (0.37, 0.913, 1e-5),
        (-1.3, 1.7, 1e-6),
        (2.75, -0.45, 1e-5),
        (1.0e-3, 3.1, 1e-6),
        (17.0, 0.0125, 1e-5),
        (-0.731, 1.234_5, 0.0),
    ];
    for (x, w, eps) in probes {
        for len in 9..=16usize {
            let input = vec![x; len];
            let weight = vec![w; len];
            let mut output = vec![f32::NAN; len];
            rms_norm_simd(&input, &weight, &mut output, eps).expect("matching lengths");
            assert_eq!(
                output[0].to_bits(),
                output[len - 1].to_bits(),
                "rms_norm_simd len={len} x={x} w={w} eps={eps}: output[0] = {:e} (body) but \
                 output[{}] = {:e} (tail)",
                output[0],
                len - 1,
                output[len - 1]
            );
        }
    }
}

/// With uniform `x0` (first half), `x1` (second half), `cos` and `sin`,
/// every rotation pair is the same computation: `output[0]` and
/// `output[half_dim - 1]` (the rotated first halves), and `output[half_dim]`
/// and `output[2 * half_dim - 1]` (the rotated second halves), must agree
/// bit for bit. Before the fix the AVX2 tail rounded `x0*cos - x1*sin` and
/// `x0*sin + x1*cos` product by product, against the body's fused
/// multiply-adds.
#[test]
fn rope_apply_simd_body_and_tail_agree_bitwise_for_uniform_buffers() {
    let probes: [(f32, f32, f32); 5] = [
        (0.37, -0.81, 0.3),
        (-1.3, 1.9, 1.1),
        (2.75, 0.05, 2.7),
        (-17.5, 4.25, 0.785),
        (1.0e-3, -6.0, 3.0),
    ];
    for (x0, x1, angle) in probes {
        let (c, s) = (angle.cos(), angle.sin());
        for half_dim in 9..=16usize {
            let mut input = vec![x0; 2 * half_dim];
            input[half_dim..].fill(x1);
            let cos_table = vec![c; half_dim];
            let sin_table = vec![s; half_dim];
            let mut output = vec![f32::NAN; 2 * half_dim];
            rope_apply_simd(&input, &mut output, &cos_table, &sin_table).expect("matching lengths");
            for (first, last) in [(0, half_dim - 1), (half_dim, 2 * half_dim - 1)] {
                assert_eq!(
                    output[first].to_bits(),
                    output[last].to_bits(),
                    "rope_apply_simd half_dim={half_dim} x0={x0} x1={x1} angle={angle}: \
                     output[{first}] = {:e} (body) but output[{last}] = {:e} (tail)",
                    output[first],
                    output[last]
                );
            }
        }
    }
}
