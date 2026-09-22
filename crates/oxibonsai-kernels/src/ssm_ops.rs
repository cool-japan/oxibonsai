//! Causal depthwise conv1d, kernel size 4 (`d_conv = 4`), for the Bonsai 2
//! Gated DeltaNet SSM path (K-09).
//!
//! Reference: `fork/models/ssm_conv_cpu.cpp.txt`
//! (`ggml_compute_forward_ssm_conv_f32`): `nc = d_conv = 4;
//! ncs = d_conv - 1 + n_t; nr = d_inner; sumf = Σ_{i0<nc} s[i0 + i1*ncs] *
//! c[i0 + i1*nc]`, applied over the concatenated qkv stream. SiLU and the
//! joint q‖k L2-norm that follow in the real forward graph are **not**
//! applied here — `qwen35.cpp:494-498` runs `ggml_ssm_conv` then a
//! *separate* `ggml_silu`, and B2-06's `norms::l2_norm_simd` is the
//! separate L2-norm step (K-10). This module is the conv only.
//!
//! # Memory layouts
//!
//! Both entry points are **channel-major on every input**: a channel's
//! samples (whether history, weight taps, or a combined window) are
//! contiguous, and channels are the outer/slow dimension — the same layout
//! `ggml_compute_forward_ssm_conv_f32` itself uses for `conv_x`/`src0` and
//! for the `conv1d.weight`/`src1` tensor (`ssm_conv1d.weight` GGUF `ne =
//! [4, channels]`, i.e. 4 contiguous floats per channel).
//!
//! [`causal_conv1d_k4_prefill`]'s **output**, however, is *token*-major
//! (`[n_t][channels]`, channel fastest) — this is deliberate, not an
//! inconsistency: it mirrors `ggml_compute_forward_ssm_conv_f32`'s own
//! `dst` tensor layout (`{d_inner, n_t, n_s}`, `d_inner` = `ne0` = fastest)
//! and matches every other per-token activation in this model's forward
//! pass (one contiguous row per token), so a caller can feed a prefill
//! chunk's output directly into the next per-token op (L2-norm, GDN, …)
//! without a transpose.

use crate::error::{KernelError, KernelResult};

/// Depthwise conv1d kernel width. `d_conv = 4` is fixed by the Bonsai 2
/// GGUF (`ssm.conv_kernel = 4`) and by the `_k4` in these functions' names —
/// it is a compile-time constant, not a runtime parameter, so the NEON tap
/// loop below can be fully unrolled.
const KC: usize = 4;

/// One decode step (`n_t = 1`) of the causal depthwise conv1d for **every**
/// channel at once.
///
/// - `state`: `[channels][KC-1]` (channel-major, oldest tap first), updated
///   **in place** to the post-step window. `channels` is taken from
///   `x_t.len()` (unambiguous, unlike `causal_conv1d_k4_prefill` — see its
///   doc comment).
/// - `x_t`: `[channels]`, this token's pre-conv activation.
/// - `w`: `[channels][KC]` (channel-major; GGUF `ssm_conv1d.weight`, `ne =
///   [4, channels]`).
/// - `out`: `[channels]`, this token's post-conv activation (SiLU **not**
///   applied — apply `norms`/`simd_float_ops::silu_simd` separately).
///
/// `out[c] = Σ_{i<KC-1} state[c][i]*w[c][i] + x_t[c]*w[c][KC-1]`, then
/// `state[c]` shifts to `[state[c][1], state[c][2], x_t[c]]`.
///
/// The shift is fused into the same pass that computes `out` (one read of
/// `state[c]`, one write of `state[c]`, no separate shift pass over the
/// buffer) — a naive implementation that shifted the *whole* per-layer
/// state as a distinct step before computing would touch the full 120 KiB
/// conv state (48 layers × 120 KiB = 5.8 MiB) an extra time per token for
/// no reason (K-09).
///
/// # Errors
///
/// - [`KernelError::NamedBufferTooSmall`] if `state.len() < 3 *
///   x_t.len()`, `w.len() < 4 * x_t.len()`, or `out.len() < x_t.len()`.
#[inline]
pub fn causal_conv1d_k4_decode(
    state: &mut [f32],
    x_t: &[f32],
    w: &[f32],
    out: &mut [f32],
) -> KernelResult<()> {
    let channels = x_t.len();
    let needed_state = channels * (KC - 1);
    if state.len() < needed_state {
        return Err(KernelError::buffer_too_small(
            "state",
            needed_state,
            state.len(),
        ));
    }
    let needed_w = channels * KC;
    if w.len() < needed_w {
        return Err(KernelError::buffer_too_small("w", needed_w, w.len()));
    }
    if out.len() < channels {
        return Err(KernelError::buffer_too_small("out", channels, out.len()));
    }

    #[cfg(target_arch = "aarch64")]
    {
        causal_conv1d_k4_decode_neon(state, x_t, w, out, channels);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA support just confirmed.
            unsafe { causal_conv1d_k4_decode_avx2(state, x_t, w, out, channels) };
        } else {
            causal_conv1d_k4_decode_scalar(state, x_t, w, out, channels);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        causal_conv1d_k4_decode_scalar(state, x_t, w, out, channels);
    }

    Ok(())
}

/// Windowed (prefill) form of the causal depthwise conv1d, for `n_t` tokens
/// of `d_inner` channels at once.
///
/// - `conv_x`: `[d_inner][KC-1+n_t]` (channel-major) — the **combined**
///   window: for each channel, `KC-1` history taps (oldest first) followed
///   by `n_t` new samples. This is exactly `ggml`'s own `conv_x`/`src0`
///   layout, so the caller (a later package's forward driver) builds it by
///   concatenating the persistent per-channel conv state with this
///   prefill chunk's new activations, the same way `qwen35.cpp`'s
///   `build_conv_state` does.
/// - `w`: `[d_inner][KC]` (channel-major).
/// - `out`: `[n_t][d_inner]` (**token**-major — see the module doc comment).
/// - `n_t`, `d_inner`: both required explicitly. Unlike
///   [`causal_conv1d_k4_decode`] (where `x_t.len()` alone unambiguously
///   gives `channels`), `conv_x.len() == d_inner * (KC-1+n_t)` has many
///   `(d_inner, n_t)` factorisations, so both dimensions must be named.
///
/// **State is not carried by this function.** After a prefill call, the
/// caller must copy the *trailing* `KC-1` columns of each channel's
/// `conv_x` row (i.e. `conv_x[c*(KC-1+n_t) + n_t .. c*(KC-1+n_t) + n_t +
/// KC-1]`, the last `KC-1` samples fed in) into the persistent per-channel
/// conv state, so a subsequent [`causal_conv1d_k4_decode`] call continues
/// the same causal window. This mirrors `gdn_prefill_f32`'s state handling
/// and is deliberate: it keeps this kernel a pure function of its inputs,
/// with no hidden dependency on how the cache is owned (the recurrent-state
/// cache is B2-10's territory).
///
/// Internally parallelised over **tokens** (`rayon`, `d_inner`-wide chunks
/// of the token-major `out`) rather than literally over the `d_inner`
/// (channel) axis: channels are the mathematically independent axis (no
/// channel depends on another), but `out`'s memory layout makes a *token*
/// split the one with disjoint, contiguous, safely-`&mut`-able chunks
/// (`out.par_chunks_mut(d_inner)`); reads of the shared `conv_x`/`w` slices
/// are `&[f32]`, so any number of channel-strided concurrent reads across
/// those per-token tasks is race-free regardless of which axis is chunked.
/// Each per-token task still vectorises over channels internally (NEON/AVX2
/// tiers), so the channel axis's parallelism is exploited at the SIMD
/// level even though the thread-level split is by token.
///
/// # Errors
///
/// - [`KernelError::NamedBufferTooSmall`] if `conv_x.len() < d_inner *
///   (KC-1+n_t)`, `w.len() < d_inner * KC`, or `out.len() < n_t * d_inner`.
#[inline]
pub fn causal_conv1d_k4_prefill(
    conv_x: &[f32],
    w: &[f32],
    out: &mut [f32],
    n_t: usize,
    d_inner: usize,
) -> KernelResult<()> {
    let ncs = (KC - 1) + n_t;
    let needed_conv_x = d_inner * ncs;
    if conv_x.len() < needed_conv_x {
        return Err(KernelError::buffer_too_small(
            "conv_x",
            needed_conv_x,
            conv_x.len(),
        ));
    }
    let needed_w = d_inner * KC;
    if w.len() < needed_w {
        return Err(KernelError::buffer_too_small("w", needed_w, w.len()));
    }
    let needed_out = n_t * d_inner;
    if out.len() < needed_out {
        return Err(KernelError::buffer_too_small("out", needed_out, out.len()));
    }

    if n_t == 0 || d_inner == 0 {
        return Ok(());
    }

    // Below this many tokens, thread-spawn overhead is not worth it —
    // matches the "sequential fallback for small batches" convention
    // already used by `parallel.rs`'s `*_par` wrappers.
    const PAR_PREFILL_MIN_TOKENS: usize = 8;

    if n_t < PAR_PREFILL_MIN_TOKENS {
        for t in 0..n_t {
            conv1d_prefill_one_token(
                conv_x,
                w,
                &mut out[t * d_inner..(t + 1) * d_inner],
                t,
                ncs,
                d_inner,
            );
        }
        return Ok(());
    }

    // On WASM: no rayon threads available — fall back to sequential.
    #[cfg(target_arch = "wasm32")]
    {
        for t in 0..n_t {
            conv1d_prefill_one_token(
                conv_x,
                w,
                &mut out[t * d_inner..(t + 1) * d_inner],
                t,
                ncs,
                d_inner,
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out[..needed_out]
            .par_chunks_mut(d_inner)
            .enumerate()
            .for_each(|(t, out_row)| {
                conv1d_prefill_one_token(conv_x, w, out_row, t, ncs, d_inner);
            });
    }

    Ok(())
}

// ═════════════════════════════════════════════════════════════════
//  Shared dot product (bitwise consistency between decode and prefill)
// ═════════════════════════════════════════════════════════════════

/// The one 4-tap dot product every tier (scalar tail *and* SIMD body, in
/// *both* [`causal_conv1d_k4_decode`] and [`causal_conv1d_k4_prefill`])
/// must compute with this **exact** operand order: one multiply followed
/// by three fused multiply-adds, tap 0 to tap 3.
///
/// This is what makes "prefill == T sequential decode calls, bitwise" (the
/// package's acceptance gate) provable rather than merely likely: decode
/// call `t`'s window `[state_after_t_steps, x_t]` is, by construction
/// (state shift is an exact copy, no arithmetic), identical to prefill's
/// window `conv_x[c][t..t+KC]` for the same channel and token — so if every
/// tier reduces that identical 8-tuple `(w0..w3, x0..x3)` through this same
/// sequence, the two call paths cannot diverge by even one rounding step.
/// `f32::mul_add` maps to a genuine hardware FMA (single rounding) on both
/// NEON (`vfmaq_f32`) and AVX2+FMA (`_mm256_fmadd_ps`) — matching this
/// scalar helper bit-for-bit is *why* the SIMD tiers use the same 4-step
/// shape instead of e.g. a pairwise-sum tree.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn conv_dot4(w0: f32, w1: f32, w2: f32, w3: f32, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    x3.mul_add(w3, x2.mul_add(w2, x1.mul_add(w1, x0 * w0)))
}

// ═════════════════════════════════════════════════════════════════
//  Scalar fallbacks (also the reference in tests)
// ═════════════════════════════════════════════════════════════════

#[allow(dead_code)]
#[inline]
fn causal_conv1d_k4_decode_scalar(
    state: &mut [f32],
    x_t: &[f32],
    w: &[f32],
    out: &mut [f32],
    channels: usize,
) {
    for c in 0..channels {
        let sb = c * (KC - 1);
        let wb = c * KC;
        let s0 = state[sb];
        let s1 = state[sb + 1];
        let s2 = state[sb + 2];
        let x = x_t[c];
        out[c] = conv_dot4(w[wb], w[wb + 1], w[wb + 2], w[wb + 3], s0, s1, s2, x);
        state[sb] = s1;
        state[sb + 1] = s2;
        state[sb + 2] = x;
    }
}

/// `#[allow(dead_code)]`: on aarch64 (this crate's primary CI target) the
/// runtime dispatch in `conv1d_prefill_one_token` always takes the NEON
/// branch, so in a plain (non-`cfg(test)`) library build this scalar
/// reference has no caller at all outside the test module — same rationale
/// as the scalar fallbacks in `simd_float_ops.rs`. It IS exercised, by
/// `prefill_one_token_matches_scalar_reference` below.
#[allow(dead_code)]
#[inline]
fn conv1d_prefill_one_token_scalar(
    conv_x: &[f32],
    w: &[f32],
    out_row: &mut [f32],
    t: usize,
    ncs: usize,
    d_inner: usize,
) {
    for (c, out_val) in out_row.iter_mut().enumerate().take(d_inner) {
        let base = c * ncs + t;
        let wb = c * KC;
        *out_val = conv_dot4(
            w[wb],
            w[wb + 1],
            w[wb + 2],
            w[wb + 3],
            conv_x[base],
            conv_x[base + 1],
            conv_x[base + 2],
            conv_x[base + 3],
        );
    }
}

/// Dispatch one prefill token's `d_inner`-wide row to the best available
/// tier. Not itself gated on length (the public entry point already
/// validated everything); `out_row.len() == d_inner` is an invariant of how
/// [`causal_conv1d_k4_prefill`] slices its output before calling this.
#[inline]
fn conv1d_prefill_one_token(
    conv_x: &[f32],
    w: &[f32],
    out_row: &mut [f32],
    t: usize,
    ncs: usize,
    d_inner: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        conv1d_prefill_one_token_neon(conv_x, w, out_row, t, ncs, d_inner);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA support just confirmed.
            unsafe { conv1d_prefill_one_token_avx2(conv_x, w, out_row, t, ncs, d_inner) };
        } else {
            conv1d_prefill_one_token_scalar(conv_x, w, out_row, t, ncs, d_inner);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        conv1d_prefill_one_token_scalar(conv_x, w, out_row, t, ncs, d_inner);
    }
}

// ═════════════════════════════════════════════════════════════════
//  AArch64 NEON implementations
// ═════════════════════════════════════════════════════════════════

/// # Safety
///
/// The caller (`causal_conv1d_k4_decode`) must guarantee `state.len() >= 3
/// * channels`, `w.len() >= 4 * channels`, `x_t.len() >= channels` and
/// `out.len() >= channels`.
///
/// `KC == 4` is a compile-time constant, so the tap loop is fully unrolled
/// and the access is transposed via NEON's structured loads: `vld3q_f32`
/// deinterleaves `state`'s `[channels][3]` layout directly into 3 per-tap
/// vectors (one contiguous 12-float load covering 4 channels, not a
/// stride-3 gather), `vld4q_f32` does the same for `w`'s `[channels][4]`
/// layout, and `vst3q_f32` writes the shifted 3-tap state back with the
/// same single-store transpose. This turns what would otherwise be a
/// stride-3/stride-4 gather across channels into contiguous loads.
#[cfg(target_arch = "aarch64")]
#[inline]
fn causal_conv1d_k4_decode_neon(
    state: &mut [f32],
    x_t: &[f32],
    w: &[f32],
    out: &mut [f32],
    channels: usize,
) {
    use std::arch::aarch64::*;

    let mut c = 0;
    // SAFETY: NEON is always available on aarch64; every access below is
    // bounded by `channels`, which the caller has already validated against
    // `state`/`w`/`x_t`/`out`'s lengths.
    unsafe {
        while c + 4 <= channels {
            let s = vld3q_f32(state.as_ptr().add(c * (KC - 1)));
            let wv = vld4q_f32(w.as_ptr().add(c * KC));
            let xv = vld1q_f32(x_t.as_ptr().add(c));

            // conv_dot4's exact order: mul then 3 fma's, tap 0 to tap 3.
            let mut acc = vmulq_f32(s.0, wv.0);
            acc = vfmaq_f32(acc, s.1, wv.1);
            acc = vfmaq_f32(acc, s.2, wv.2);
            acc = vfmaq_f32(acc, xv, wv.3);
            vst1q_f32(out.as_mut_ptr().add(c), acc);

            // Fused shift: new state = [old tap1, old tap2, x] — a single
            // structured store, no separate pass over the buffer (K-09).
            let new_s = float32x4x3_t(s.1, s.2, xv);
            vst3q_f32(state.as_mut_ptr().add(c * (KC - 1)), new_s);

            c += 4;
        }
    }

    for cc in c..channels {
        let sb = cc * (KC - 1);
        let wb = cc * KC;
        let s0 = state[sb];
        let s1 = state[sb + 1];
        let s2 = state[sb + 2];
        let x = x_t[cc];
        out[cc] = conv_dot4(w[wb], w[wb + 1], w[wb + 2], w[wb + 3], s0, s1, s2, x);
        state[sb] = s1;
        state[sb + 1] = s2;
        state[sb + 2] = x;
    }
}

/// Processes one token's `d_inner`-wide row, 4 channels at a time.
///
/// Unlike the decode path, `conv_x`'s channels are `ncs` apart (not a
/// tight interleave `vld4q_f32` can deinterleave directly, since `ncs !=
/// KC` in general), so the 4-channel window is gathered through a small
/// stack buffer instead; `w`'s `[channels][KC]` layout *is* a tight
/// interleave regardless of `ncs`, so its load still uses `vld4q_f32`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn conv1d_prefill_one_token_neon(
    conv_x: &[f32],
    w: &[f32],
    out_row: &mut [f32],
    t: usize,
    ncs: usize,
    d_inner: usize,
) {
    use std::arch::aarch64::*;

    let mut c = 0;
    // SAFETY: bounded by `d_inner`; every `conv_x`/`w` index reached is
    // `< d_inner * ncs` / `< d_inner * KC`, both caller-validated bounds
    // (via `causal_conv1d_k4_prefill`), and `out_row.len() == d_inner`.
    unsafe {
        while c + 4 <= d_inner {
            let mut tap_buf = [[0.0f32; 4]; KC];
            for (k, tap_k) in tap_buf.iter_mut().enumerate() {
                for (j, slot) in tap_k.iter_mut().enumerate() {
                    *slot = conv_x[(c + j) * ncs + t + k];
                }
            }
            let tap0 = vld1q_f32(tap_buf[0].as_ptr());
            let tap1 = vld1q_f32(tap_buf[1].as_ptr());
            let tap2 = vld1q_f32(tap_buf[2].as_ptr());
            let tap3 = vld1q_f32(tap_buf[3].as_ptr());
            let wv = vld4q_f32(w.as_ptr().add(c * KC));

            let mut acc = vmulq_f32(tap0, wv.0);
            acc = vfmaq_f32(acc, tap1, wv.1);
            acc = vfmaq_f32(acc, tap2, wv.2);
            acc = vfmaq_f32(acc, tap3, wv.3);
            vst1q_f32(out_row.as_mut_ptr().add(c), acc);

            c += 4;
        }
    }

    for (cc, out_val) in out_row.iter_mut().enumerate().take(d_inner).skip(c) {
        let base = cc * ncs + t;
        let wb = cc * KC;
        *out_val = conv_dot4(
            w[wb],
            w[wb + 1],
            w[wb + 2],
            w[wb + 3],
            conv_x[base],
            conv_x[base + 1],
            conv_x[base + 2],
            conv_x[base + 3],
        );
    }
}

// ═════════════════════════════════════════════════════════════════
//  x86_64 AVX2+FMA implementations
// ═════════════════════════════════════════════════════════════════

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. In addition, the
/// caller (`causal_conv1d_k4_decode`) must guarantee `state.len() >= 3 *
/// channels`, `w.len() >= 4 * channels`, `x_t.len() >= channels` and
/// `out.len() >= channels`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn causal_conv1d_k4_decode_avx2(
    state: &mut [f32],
    x_t: &[f32],
    w: &[f32],
    out: &mut [f32],
    channels: usize,
) {
    use std::arch::x86_64::*;

    let mut c = 0;
    while c + 8 <= channels {
        let mut s0b = [0.0f32; 8];
        let mut s1b = [0.0f32; 8];
        let mut s2b = [0.0f32; 8];
        let mut w0b = [0.0f32; 8];
        let mut w1b = [0.0f32; 8];
        let mut w2b = [0.0f32; 8];
        let mut w3b = [0.0f32; 8];
        for j in 0..8 {
            let sb = (c + j) * (KC - 1);
            s0b[j] = state[sb];
            s1b[j] = state[sb + 1];
            s2b[j] = state[sb + 2];
            let wb = (c + j) * KC;
            w0b[j] = w[wb];
            w1b[j] = w[wb + 1];
            w2b[j] = w[wb + 2];
            w3b[j] = w[wb + 3];
        }
        let s0 = _mm256_loadu_ps(s0b.as_ptr());
        let s1 = _mm256_loadu_ps(s1b.as_ptr());
        let s2 = _mm256_loadu_ps(s2b.as_ptr());
        let xv = _mm256_loadu_ps(x_t.as_ptr().add(c));
        let wv0 = _mm256_loadu_ps(w0b.as_ptr());
        let wv1 = _mm256_loadu_ps(w1b.as_ptr());
        let wv2 = _mm256_loadu_ps(w2b.as_ptr());
        let wv3 = _mm256_loadu_ps(w3b.as_ptr());

        let mut acc = _mm256_mul_ps(s0, wv0);
        acc = _mm256_fmadd_ps(s1, wv1, acc);
        acc = _mm256_fmadd_ps(s2, wv2, acc);
        acc = _mm256_fmadd_ps(xv, wv3, acc);
        _mm256_storeu_ps(out.as_mut_ptr().add(c), acc);

        for j in 0..8 {
            let sb = (c + j) * (KC - 1);
            state[sb] = s1b[j];
            state[sb + 1] = s2b[j];
            state[sb + 2] = x_t[c + j];
        }
        c += 8;
    }

    for cc in c..channels {
        let sb = cc * (KC - 1);
        let wb = cc * KC;
        let s0 = state[sb];
        let s1 = state[sb + 1];
        let s2 = state[sb + 2];
        let x = x_t[cc];
        out[cc] = conv_dot4(w[wb], w[wb + 1], w[wb + 2], w[wb + 3], s0, s1, s2, x);
        state[sb] = s1;
        state[sb + 1] = s2;
        state[sb + 2] = x;
    }
}

/// # Safety
///
/// Caller must have confirmed `avx2` + `fma` support. Bounds are the same
/// as [`conv1d_prefill_one_token_neon`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv1d_prefill_one_token_avx2(
    conv_x: &[f32],
    w: &[f32],
    out_row: &mut [f32],
    t: usize,
    ncs: usize,
    d_inner: usize,
) {
    use std::arch::x86_64::*;

    let mut c = 0;
    while c + 8 <= d_inner {
        let mut tap_buf = [[0.0f32; 8]; KC];
        for (k, tap_k) in tap_buf.iter_mut().enumerate() {
            for (j, slot) in tap_k.iter_mut().enumerate() {
                *slot = conv_x[(c + j) * ncs + t + k];
            }
        }
        let mut wbuf = [[0.0f32; 8]; KC];
        for (k, w_k) in wbuf.iter_mut().enumerate() {
            for (j, slot) in w_k.iter_mut().enumerate() {
                *slot = w[(c + j) * KC + k];
            }
        }

        let tap0 = _mm256_loadu_ps(tap_buf[0].as_ptr());
        let tap1 = _mm256_loadu_ps(tap_buf[1].as_ptr());
        let tap2 = _mm256_loadu_ps(tap_buf[2].as_ptr());
        let tap3 = _mm256_loadu_ps(tap_buf[3].as_ptr());
        let wv0 = _mm256_loadu_ps(wbuf[0].as_ptr());
        let wv1 = _mm256_loadu_ps(wbuf[1].as_ptr());
        let wv2 = _mm256_loadu_ps(wbuf[2].as_ptr());
        let wv3 = _mm256_loadu_ps(wbuf[3].as_ptr());

        let mut acc = _mm256_mul_ps(tap0, wv0);
        acc = _mm256_fmadd_ps(tap1, wv1, acc);
        acc = _mm256_fmadd_ps(tap2, wv2, acc);
        acc = _mm256_fmadd_ps(tap3, wv3, acc);
        _mm256_storeu_ps(out_row.as_mut_ptr().add(c), acc);

        c += 8;
    }

    for (cc, out_val) in out_row.iter_mut().enumerate().take(d_inner).skip(c) {
        let base = cc * ncs + t;
        let wb = cc * KC;
        *out_val = conv_dot4(
            w[wb],
            w[wb + 1],
            w[wb + 2],
            w[wb + 3],
            conv_x[base],
            conv_x[base + 1],
            conv_x[base + 2],
            conv_x[base + 3],
        );
    }
}

// ═════════════════════════════════════════════════════════════════
//  Tests
// ═════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* generator so tests need no external RNG
    /// dependency (this crate's `owned_files` for this package do not
    /// include `Cargo.toml`).
    struct XorShift64(u64);
    impl XorShift64 {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // Map the top 24 bits to roughly [-0.5, 0.5).
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }
    }

    // ── basic decode correctness ───────────────────────────────

    #[test]
    fn decode_matches_hand_computed_single_channel() {
        // 1 channel, state = [1.0, 2.0, 3.0], x_t = 4.0, w = [10, 20, 30, 40].
        // out = 1*10 + 2*20 + 3*30 + 4*40 = 10+40+90+160 = 300.
        let mut state = vec![1.0f32, 2.0, 3.0];
        let x_t = vec![4.0f32];
        let w = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32];

        causal_conv1d_k4_decode(&mut state, &x_t, &w, &mut out).expect("valid call");

        assert!((out[0] - 300.0).abs() < 1e-3, "out = {}", out[0]);
        // state shifts to [2.0, 3.0, 4.0].
        assert_eq!(state, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn decode_state_shift_is_a_pure_fifo() {
        let mut state = vec![0.0f32, 0.0, 0.0];
        let w = vec![0.0f32, 0.0, 0.0, 1.0]; // out = x_t (identity tap)
        let mut out = vec![0.0f32];

        for (step, &x) in [1.0f32, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            causal_conv1d_k4_decode(&mut state, &[x], &w, &mut out).expect("valid call");
            if step >= 3 {
                // After 4 steps the window is fully populated with real data.
                assert_eq!(out[0], x);
            }
        }
        // After 5 steps: state holds the last 3 inputs [3,4,5].
        assert_eq!(state, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn decode_matches_scalar_reference_many_channels() {
        let channels = 41; // not a multiple of 4
        let mut rng = XorShift64(0xdead_beef_1234_5678);
        let mut state_simd: Vec<f32> = (0..channels * 3).map(|_| rng.next_f32()).collect();
        let mut state_scalar = state_simd.clone();
        let w: Vec<f32> = (0..channels * 4).map(|_| rng.next_f32()).collect();
        let x_t: Vec<f32> = (0..channels).map(|_| rng.next_f32()).collect();
        let mut out_simd = vec![0.0f32; channels];
        let mut out_scalar = vec![0.0f32; channels];

        causal_conv1d_k4_decode(&mut state_simd, &x_t, &w, &mut out_simd).expect("valid");
        causal_conv1d_k4_decode_scalar(&mut state_scalar, &x_t, &w, &mut out_scalar, channels);

        for i in 0..channels {
            assert!(
                (out_simd[i] - out_scalar[i]).abs() < 1e-4,
                "out[{i}]: {} vs {}",
                out_simd[i],
                out_scalar[i]
            );
        }
        for i in 0..channels * 3 {
            assert!(
                (state_simd[i] - state_scalar[i]).abs() < 1e-6,
                "state[{i}]: {} vs {}",
                state_simd[i],
                state_scalar[i]
            );
        }
    }

    #[test]
    fn decode_rejects_short_state() {
        let mut state = vec![0.0f32; 2]; // needs 3
        let x_t = vec![1.0f32];
        let w = vec![0.0f32; 4];
        let mut out = vec![0.0f32];
        assert!(causal_conv1d_k4_decode(&mut state, &x_t, &w, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_short_weight() {
        let mut state = vec![0.0f32; 3];
        let x_t = vec![1.0f32];
        let w = vec![0.0f32; 3]; // needs 4
        let mut out = vec![0.0f32];
        assert!(causal_conv1d_k4_decode(&mut state, &x_t, &w, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_short_output() {
        let mut state = vec![0.0f32; 3];
        let x_t = vec![1.0f32];
        let w = vec![0.0f32; 4];
        let mut out: Vec<f32> = vec![]; // needs 1
        assert!(causal_conv1d_k4_decode(&mut state, &x_t, &w, &mut out).is_err());
    }

    // ── prefill validation ─────────────────────────────────────

    #[test]
    fn prefill_rejects_short_conv_x() {
        let conv_x = vec![0.0f32; 5]; // needs 4*(3+2)=20 for d_inner=4,n_t=2
        let w = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 8];
        assert!(causal_conv1d_k4_prefill(&conv_x, &w, &mut out, 2, 4).is_err());
    }

    #[test]
    fn prefill_rejects_short_weight() {
        let conv_x = vec![0.0f32; 20];
        let w = vec![0.0f32; 3]; // needs 16
        let mut out = vec![0.0f32; 8];
        assert!(causal_conv1d_k4_prefill(&conv_x, &w, &mut out, 2, 4).is_err());
    }

    #[test]
    fn prefill_rejects_short_output() {
        let conv_x = vec![0.0f32; 20];
        let w = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 3]; // needs 8
        assert!(causal_conv1d_k4_prefill(&conv_x, &w, &mut out, 2, 4).is_err());
    }

    #[test]
    fn prefill_zero_tokens_is_a_harmless_no_op() {
        let conv_x = vec![0.0f32; 4 * 3];
        let w = vec![0.0f32; 4 * 4];
        let mut out: Vec<f32> = vec![];
        causal_conv1d_k4_prefill(&conv_x, &w, &mut out, 0, 4).expect("n_t=0 must not error");
    }

    // ── the acceptance gate: prefill == T sequential decode calls ──

    /// `channels` and `n_t` are deliberately **not** multiples of 4, so a
    /// `(channel, token)` pair can land in the NEON body on one call path
    /// and the scalar tail on the other — the only way to actually exercise
    /// `conv_dot4`'s ordering guarantee across that boundary (this crate's
    /// existing K-M3 body/tail bugs were exactly this class of hazard).
    #[test]
    fn prefill_matches_t_sequential_decode_calls_bitwise() {
        let channels = 37;
        let n_t = 11;
        let mut rng = XorShift64(0x0ddc_0ffe_e15b_ad00);

        let initial_state: Vec<f32> = (0..channels * (KC - 1)).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..channels * KC).map(|_| rng.next_f32()).collect();
        let xs: Vec<Vec<f32>> = (0..n_t)
            .map(|_| (0..channels).map(|_| rng.next_f32()).collect())
            .collect();

        // Path A: T sequential decode calls, threading `state` through.
        let mut state = initial_state.clone();
        let mut decode_out = vec![0.0f32; n_t * channels];
        for (t, x_row) in xs.iter().enumerate() {
            let mut step_out = vec![0.0f32; channels];
            causal_conv1d_k4_decode(&mut state, x_row, &w, &mut step_out).expect("valid decode");
            decode_out[t * channels..(t + 1) * channels].copy_from_slice(&step_out);
        }

        // Path B: one prefill call. conv_x[c] = initial_state[c] ++ xs[..][c].
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
        causal_conv1d_k4_prefill(&conv_x, &w, &mut prefill_out, n_t, channels)
            .expect("valid prefill");

        for i in 0..n_t * channels {
            assert_eq!(
                decode_out[i].to_bits(),
                prefill_out[i].to_bits(),
                "mismatch at token {} channel {}: decode={} prefill={}",
                i / channels,
                i % channels,
                decode_out[i],
                prefill_out[i]
            );
        }
    }

    /// Same gate, but with dimensions that ARE multiples of 4 (the
    /// "everything lands in the SIMD body" case) — belt-and-braces
    /// alongside the ragged-dimensions version above.
    #[test]
    fn prefill_matches_t_sequential_decode_calls_bitwise_aligned_dims() {
        let channels = 32;
        let n_t = 16;
        let mut rng = XorShift64(0xa11a_11ed_00d5_eed5);

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
        causal_conv1d_k4_prefill(&conv_x, &w, &mut prefill_out, n_t, channels)
            .expect("valid prefill");

        for i in 0..n_t * channels {
            assert_eq!(decode_out[i].to_bits(), prefill_out[i].to_bits());
        }
    }

    /// Exercises the `>= PAR_PREFILL_MIN_TOKENS` rayon-parallel path
    /// specifically (the two tests above use small `n_t` and stay on the
    /// sequential fallback).
    #[test]
    fn prefill_parallel_path_matches_sequential_decode_bitwise() {
        let channels = 19;
        let n_t = 64; // comfortably above PAR_PREFILL_MIN_TOKENS
        let mut rng = XorShift64(0x5eed_5eed_5eed_5eed);

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
        causal_conv1d_k4_prefill(&conv_x, &w, &mut prefill_out, n_t, channels)
            .expect("valid prefill");

        for i in 0..n_t * channels {
            assert_eq!(decode_out[i].to_bits(), prefill_out[i].to_bits());
        }
    }

    /// Cross-tier check for the prefill path (the decode path already has
    /// `decode_matches_scalar_reference_many_channels`): the dispatched
    /// (NEON, on this machine) per-token computation must agree with the
    /// scalar reference within float tolerance. Also keeps
    /// `conv1d_prefill_one_token_scalar` genuinely exercised rather than
    /// merely reachable-in-principle on a target where it is dead code.
    #[test]
    fn prefill_one_token_matches_scalar_reference() {
        let d_inner = 43; // not a multiple of 4 or 8
        let n_t = 5;
        let t = 2;
        let ncs = (KC - 1) + n_t;
        let mut rng = XorShift64(0xfeed_face_dead_c0de);

        let conv_x: Vec<f32> = (0..d_inner * ncs).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..d_inner * KC).map(|_| rng.next_f32()).collect();

        let mut dispatched = vec![0.0f32; d_inner];
        conv1d_prefill_one_token(&conv_x, &w, &mut dispatched, t, ncs, d_inner);

        let mut scalar = vec![0.0f32; d_inner];
        conv1d_prefill_one_token_scalar(&conv_x, &w, &mut scalar, t, ncs, d_inner);

        for i in 0..d_inner {
            assert!(
                (dispatched[i] - scalar[i]).abs() < 1e-4,
                "channel {i}: dispatched={} scalar={}",
                dispatched[i],
                scalar[i]
            );
        }
    }

    #[test]
    fn conv_dot4_matches_naive_sum() {
        let (w0, w1, w2, w3) = (1.5f32, -2.0, 0.25, 3.0);
        let (x0, x1, x2, x3) = (0.5f32, 1.0, -1.5, 2.0);
        let naive = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
        let fused = conv_dot4(w0, w1, w2, w3, x0, x1, x2, x3);
        assert!((naive - fused).abs() < 1e-6, "{naive} vs {fused}");
    }
}
