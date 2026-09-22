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
