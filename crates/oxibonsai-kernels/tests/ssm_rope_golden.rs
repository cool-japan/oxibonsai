//! B2-06 acceptance gate: SSM conv1d + norms + partial/M-RoPE.
//!
//! Restates, as public-API integration tests, the exact acceptance
//! criteria from the package spec (design §8.2 B2-06):
//!
//! 1. `causal_conv1d_k4_prefill` == `T` sequential `causal_conv1d_k4_decode`
//!    calls, **bitwise**.
//! 2. `PartialRopeTable` vs `MropeTable` degeneracy, **bitwise**, over
//!    10 000 positions (the kernel-level half — `mrope_build_tables` vs
//!    `partial_rope_build_table` — since this crate's `tests/` cannot
//!    depend on `oxibonsai-model`; the model-crate structs themselves are
//!    covered by `crates/oxibonsai-model/src/layers/rope_mrope.rs`'s own
//!    inline test of the same shape).
//! 3. Partial RoPE leaves dims `[n_rot, head_dim)` **bitwise** untouched.
//! 4. Golden cos/sin tables and rotated vectors from an independent
//!    transliteration of `ggml_rope_multi` (`fork/models/ops.cpp`), for
//!    `n_rot=64`, `head_dim=256`, `sections=[11,11,10,0]`, `freq_base=1e7`,
//!    at 8 positions. Compared with a **tolerance**, not bitwise: `powf`
//!    (used for `theta_scale`) is a transcendental function, and two
//!    independently type-written expressions computing "the same" `powf`
//!    are not guaranteed to be bit-identical `f32` values (an earlier
//!    version of a sibling test in this crate hit exactly this — 1 ULP —
//!    comparing a hand-written reference loop against the library's own
//!    runtime `powf` call). Golden-vs-implementation cross-checks
//!    throughout this design use a tolerance for the same reason (e.g. FWHT
//!    "vs naive ... <= 1e-5", GDN "vs naive ... <= 1e-6"); bitwise is
//!    reserved for two calls of the *same* underlying code path.
//! 5. `l2_norm_simd` matches `scale = 1/max(sqrt(sum(x^2)), eps)` (not
//!    `1/sqrt(sum+eps)`), including a near-zero-vector case.

use oxibonsai_kernels::norms::l2_norm_simd;
use oxibonsai_kernels::rope_mrope::{
    mrope_build_tables, partial_rope_build_table, rope_partial_splithalf_simd,
};
use oxibonsai_kernels::ssm_ops::{causal_conv1d_k4_decode, causal_conv1d_k4_prefill};

/// Deterministic xorshift64* generator — no external RNG dependency needed
/// for a self-contained golden test.
struct XorShift64(u64);
impl XorShift64 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

const KC: usize = 4;

// ═══════════════════════════════════════════════════════════════════
//  1. conv prefill == T x step, bitwise
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gate_conv_prefill_equals_t_sequential_decode_calls_bitwise() {
    let channels = 53; // not a multiple of 4 or 8
    let n_t = 23; // not a multiple of 4 or 8
    let mut rng = XorShift64(0x6011_de17_60a1_de17);

    let initial_state: Vec<f32> = (0..channels * (KC - 1)).map(|_| rng.next_f32()).collect();
    let w: Vec<f32> = (0..channels * KC).map(|_| rng.next_f32()).collect();
    let xs: Vec<Vec<f32>> = (0..n_t)
        .map(|_| (0..channels).map(|_| rng.next_f32()).collect())
        .collect();

    let mut state = initial_state.clone();
    let mut decode_out = vec![0.0f32; n_t * channels];
    for (t, x_row) in xs.iter().enumerate() {
        let mut step_out = vec![0.0f32; channels];
        causal_conv1d_k4_decode(&mut state, x_row, &w, &mut step_out).expect("valid decode");
        decode_out[t * channels..(t + 1) * channels].copy_from_slice(&step_out);
    }

    let ncs = (KC - 1) + n_t;
    let mut conv_x = vec![0.0f32; channels * ncs];
    for c in 0..channels {
        conv_x[c * ncs..c * ncs + (KC - 1)]
            .copy_from_slice(&initial_state[c * (KC - 1)..(c + 1) * (KC - 1)]);
        for (t, x_row) in xs.iter().enumerate() {
            conv_x[c * ncs + (KC - 1) + t] = x_row[c];
        }
    }
    let mut prefill_out = vec![0.0f32; n_t * channels];
    causal_conv1d_k4_prefill(&conv_x, &w, &mut prefill_out, n_t, channels).expect("valid prefill");

    for i in 0..n_t * channels {
        assert_eq!(
            decode_out[i].to_bits(),
            prefill_out[i].to_bits(),
            "token {} channel {}: decode={} prefill={}",
            i / channels,
            i % channels,
            decode_out[i],
            prefill_out[i]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
//  2. kernel-level M-RoPE degeneracy, bitwise, over 10 000 positions
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gate_mrope_degeneracy_bitwise_over_10000_positions() {
    let n_rot = 64;
    let sections = [11u32, 11, 10, 0];
    let freq_base = 1.0e7f32;
    let half = n_rot / 2;

    let mut cos_mrope = vec![0.0f32; half];
    let mut sin_mrope = vec![0.0f32; half];
    let mut cos_partial = vec![0.0f32; half];
    let mut sin_partial = vec![0.0f32; half];

    for p in 0..10_000i32 {
        mrope_build_tables(
            [p, p, p],
            sections,
            n_rot,
            freq_base,
            &mut cos_mrope,
            &mut sin_mrope,
        )
        .expect("valid: [11,11,10,0] never needs the e axis");
        partial_rope_build_table(p, n_rot, freq_base, &mut cos_partial, &mut sin_partial)
            .expect("valid");

        for k in 0..half {
            assert_eq!(
                cos_mrope[k].to_bits(),
                cos_partial[k].to_bits(),
                "pos={p} k={k}"
            );
            assert_eq!(
                sin_mrope[k].to_bits(),
                sin_partial[k].to_bits(),
                "pos={p} k={k}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  3. partial RoPE leaves dims [n_rot, head_dim) bitwise untouched
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gate_partial_rope_tail_bitwise_untouched() {
    let head_dim = 256;
    let n_rot = 64;
    let half = n_rot / 2;
    let mut rng = XorShift64(0x7a11_7a11_7a11_7a11);

    let input: Vec<f32> = (0..head_dim).map(|_| rng.next_f32()).collect();
    let cos: Vec<f32> = (0..half).map(|_| rng.next_f32().cos()).collect();
    let sin: Vec<f32> = (0..half).map(|_| rng.next_f32().sin()).collect();
    let mut output = vec![f32::NAN; head_dim];

    rope_partial_splithalf_simd(&input, &mut output, head_dim, n_rot, &cos, &sin)
        .expect("valid call");

    for i in n_rot..head_dim {
        assert_eq!(
            output[i].to_bits(),
            input[i].to_bits(),
            "dim {i} in [n_rot, head_dim) must be bitwise untouched"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
//  4. golden cos/sin + rotated vectors vs an independent ggml_rope_multi
//     transliteration, n_rot=64, head_dim=256, sections=[11,11,10,0],
//     freq_base=1e7, at 8 positions
// ═══════════════════════════════════════════════════════════════════

/// Direct, from-scratch transliteration of `ggml_mrope_cache_init` +
/// `rotate_pairs` (`fork/models/ops.cpp:5872-5959`), for the IMROPE mode
/// with `ext_factor=0` (no YaRN — Bonsai 2 sets none), specialised to the
/// text case `p_t=p_h=p_w=p` this module documents. Independent of
/// `oxibonsai_kernels::rope_mrope`'s implementation (does not call it), so
/// this is a genuine cross-check, not a self-comparison.
fn golden_ggml_rope_multi_text(
    input: &[f32],
    p: i32,
    head_dim: usize,
    n_rot: usize,
    sections: [u32; 4],
    freq_base: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let half = n_rot / 2;
    let sect_dims = (sections[0] + sections[1] + sections[2] + sections[3]) as usize;

    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
    let mut theta_t = p as f32;
    let mut theta_h = p as f32;
    let mut theta_w = p as f32;

    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];

    for k in 0..half {
        let sector = k % sect_dims;
        // ops.cpp:5908-5917 (is_imrope branch).
        let theta = if sector % 3 == 1 && sector < 3 * sections[1] as usize {
            theta_h
        } else if sector % 3 == 2 && sector < 3 * sections[2] as usize {
            theta_w
        } else if sector.is_multiple_of(3) && sector < 3 * sections[0] as usize {
            theta_t
        } else {
            panic!("golden transliteration: sector {sector} needs the vision e axis");
        };
        cos_table[k] = theta.cos();
        sin_table[k] = theta.sin();
        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
    }

    // rotate_pairs<T>(n_dims=n_rot, n_offset=half, cache, src, dst, scale=2)
    // (ops.cpp:5943-5960), applied to the first n_rot dims, then the
    // remaining [n_rot, head_dim) copied through verbatim
    // (ops.cpp:6102-6114).
    let mut output = input.to_vec();
    for k in 0..half {
        let x0 = input[k];
        let x1 = input[k + half];
        output[k] = x0 * cos_table[k] - x1 * sin_table[k];
        output[k + half] = x0 * sin_table[k] + x1 * cos_table[k];
    }
    output[n_rot..head_dim].copy_from_slice(&input[n_rot..head_dim]);

    (cos_table, sin_table, output)
}

#[test]
fn gate_golden_rope_multi_cos_sin_and_rotated_vectors_at_8_positions() {
    let head_dim = 256;
    let n_rot = 64;
    let half = n_rot / 2;
    let sections = [11u32, 11, 10, 0];
    let freq_base = 1.0e7f32;
    let positions = [0i32, 1, 2, 5, 17, 100, 1000, 65535];

    let mut rng = XorShift64(0x9017_de17_9017_de17);
    let input: Vec<f32> = (0..head_dim).map(|_| rng.next_f32()).collect();

    for &p in &positions {
        let (golden_cos, golden_sin, golden_out) =
            golden_ggml_rope_multi_text(&input, p, head_dim, n_rot, sections, freq_base);

        let mut cos_impl = vec![0.0f32; half];
        let mut sin_impl = vec![0.0f32; half];
        mrope_build_tables(
            [p, p, p],
            sections,
            n_rot,
            freq_base,
            &mut cos_impl,
            &mut sin_impl,
        )
        .expect("valid");

        for k in 0..half {
            assert!(
                (golden_cos[k] - cos_impl[k]).abs() < 1e-5,
                "pos={p} k={k}: golden cos {} vs impl {}",
                golden_cos[k],
                cos_impl[k]
            );
            assert!(
                (golden_sin[k] - sin_impl[k]).abs() < 1e-5,
                "pos={p} k={k}: golden sin {} vs impl {}",
                golden_sin[k],
                sin_impl[k]
            );
        }

        let mut out_impl = vec![0.0f32; head_dim];
        rope_partial_splithalf_simd(&input, &mut out_impl, head_dim, n_rot, &cos_impl, &sin_impl)
            .expect("valid");

        for i in 0..head_dim {
            assert!(
                (golden_out[i] - out_impl[i]).abs() < 1e-4,
                "pos={p} dim={i}: golden {} vs impl {}",
                golden_out[i],
                out_impl[i]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  5. l2_norm matches 1/max(sqrt(sum), eps), including near-zero
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gate_l2_norm_matches_eps_floors_denominator_formula() {
    let mut rng = XorShift64(0x1201_0e17_1201_0e17);
    let input: Vec<f32> = (0..128).map(|_| rng.next_f32() * 4.0).collect();
    let eps = 1e-6f32;

    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let expected_scale = 1.0 / sum_sq.sqrt().max(eps);

    let mut output = vec![0.0f32; 128];
    l2_norm_simd(&input, &mut output, eps).expect("valid");

    for i in 0..128 {
        let expected = input[i] * expected_scale;
        assert!(
            (output[i] - expected).abs() < 1e-4,
            "l2_norm[{i}] = {}, expected {expected}",
            output[i]
        );
    }
}

#[test]
fn gate_l2_norm_near_zero_vector_floors_at_1_over_eps() {
    // sqrt(sum) << eps here, so the correct formula's scale is exactly
    // 1/eps -- the wrong formula (1/sqrt(sum+eps)) would give ~1/sqrt(eps),
    // three orders of magnitude off. See `norms.rs`'s own test of this
    // shape for the worked-out numbers.
    let n = 64;
    let input = vec![1e-20f32; n];
    let eps = 1e-6f32;
    let mut output = vec![0.0f32; n];

    l2_norm_simd(&input, &mut output, eps).expect("valid");

    let expected = 1e-20f32 * (1.0 / eps);
    for &v in &output {
        assert!(
            (v - expected).abs() < expected * 0.01,
            "l2_norm near-zero = {v}, expected ~= {expected} (1/eps floor)"
        );
    }
}
