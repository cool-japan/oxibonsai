//! Partial NeoX-style RoPE and interleaved M-RoPE (`IMROPE`) table
//! construction for the Bonsai 2 hybrid full-attention layers (K-11 / M-06).
//!
//! # Correction (K-11)
//!
//! An earlier draft of this fix proposed `(2i, 2i+1)` element pairing. That
//! is wrong. `fork/models/ops.cpp:6086-6094` dispatches
//! `GGML_ROPE_TYPE_MROPE` / `GGML_ROPE_TYPE_IMROPE` (and plain `NEOX`) to
//! the *same* `rotate_pairs<T>(n_dims, n_dims/2, cache, ..)` — i.e. **NeoX
//! split-half pairing** `(j, j + n_rot/2)` for `j` in `0..n_rot/2`, over
//! only the first `n_rot` of `head_dim` channels, with `:6102-6113` copying
//! dims `[n_rot, head_dim)` through **verbatim**. "Interleaved" in `IMROPE`
//! names how the `t`/`h`/`w` position axes interleave across frequency
//! *sectors* (`ggml_mrope_cache_init`, `:5906-5917`, `sector % 3`), not
//! element pairing. `qwen35` maps to `LLAMA_ROPE_TYPE_IMROPE`
//! (`fork/src_llama-model.cpp:3203-3206`).
//!
//! # Proof: for text (`p_t = p_h = p_w`), IMROPE degenerates to standard NeoX RoPE
//!
//! `ggml_mrope_cache_init` (`ops.cpp:5872-5940`) tracks four running angles
//! `theta_{t,h,w,e}`, all four initialised from the *given* position and all
//! four multiplied by the **same** `theta_scale` on *every* iteration,
//! regardless of which one gets selected for that iteration's rotation
//! (`:5935-5938`). So if `p_t == p_h == p_w`, then at every step
//! `theta_t == theta_h == theta_w` — the `sector`-based selection is a
//! provable no-op, and the whole cache equals what
//! `ggml_rope_cache_init(p, ..)` (plain single-axis RoPE) would have
//! produced at the same `theta_scale`. With `sections = [11,11,10,0]`,
//! `sect_dims = 32 = n_rot/2` (no wraparound) and the `is_imrope` selection
//! rule below assigns *every* sector in `0..32` to `t`, `h` or `w` — `theta_e`
//! is never reached, consistent with the 4th (vision) section being
//! width-0 for text.
//!
//! [`rope_partial_splithalf_simd`] and [`mrope_build_tables`] are the two
//! primitives; `crates/oxibonsai-model/src/layers/rope_mrope.rs` wraps them
//! into `PartialRopeTable` / `MropeTable`.

use crate::error::{KernelError, KernelResult};

/// Rotate the first `n_rot` of a `head_dim`-wide vector using NeoX
/// split-half pairing, and copy the remaining `[n_rot, head_dim)` channels
/// through unchanged.
///
/// For `j in 0..n_rot/2`:
/// ```text
/// output[j]          = input[j]*cos[j] - input[j+n_rot/2]*sin[j]
/// output[j+n_rot/2]   = input[j]*sin[j] + input[j+n_rot/2]*cos[j]
/// ```
/// and `output[i] = input[i]` for `i in n_rot..head_dim` — a **real** copy
/// (not a no-op assumption), since `input` and `output` are always distinct
/// buffers at this API boundary.
///
/// This is exactly [`crate::simd_float_ops::rope_apply_simd`]'s rotation
/// (which already implements split-half pairing, deriving `half_dim` from
/// its input slice's length) applied to the first `n_rot` elements, plus
/// the pass-through tail `rope_apply_simd` has no notion of — so the
/// rotation itself is reused verbatim rather than re-implemented, and this
/// function owns only the "partial" framing around it.
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `n_rot` is odd.
/// - [`KernelError::NamedBufferTooSmall`] if `head_dim < n_rot`, or if
///   `input.len() < head_dim` or `output.len() < head_dim`.
/// - [`KernelError::NamedDimensionMismatch`] if `cos.len() < n_rot/2` or
///   `sin.len() < n_rot/2` — propagated from the underlying
///   [`crate::simd_float_ops::rope_apply_simd`] call.
#[inline]
pub fn rope_partial_splithalf_simd(
    input: &[f32],
    output: &mut [f32],
    head_dim: usize,
    n_rot: usize,
    cos: &[f32],
    sin: &[f32],
) -> KernelResult<()> {
    if !n_rot.is_multiple_of(2) {
        return Err(KernelError::NotBlockAligned {
            count: n_rot,
            block_size: 2,
        });
    }
    if head_dim < n_rot {
        return Err(KernelError::buffer_too_small("head_dim", n_rot, head_dim));
    }
    if input.len() < head_dim {
        return Err(KernelError::buffer_too_small(
            "input",
            head_dim,
            input.len(),
        ));
    }
    if output.len() < head_dim {
        return Err(KernelError::buffer_too_small(
            "output",
            head_dim,
            output.len(),
        ));
    }

    if n_rot > 0 {
        crate::simd_float_ops::rope_apply_simd(&input[..n_rot], &mut output[..n_rot], cos, sin)?;
    }
    if n_rot < head_dim {
        output[n_rot..head_dim].copy_from_slice(&input[n_rot..head_dim]);
    }

    Ok(())
}

/// Which position axis an `IMROPE` frequency sector selects.
#[derive(Debug, Clone, Copy)]
enum MropeAxis {
    T,
    H,
    W,
    /// The vision "extra" axis (`ggml`'s `theta_e`) — not representable by
    /// [`mrope_build_tables`]'s 3-axis `pos` argument.
    E,
}

/// `fork/models/ops.cpp:5908-5917`'s `is_imrope` sector-to-axis rule,
/// transliterated directly (only `sections[0..3]` are read; `sections[3]`
/// never gates a `T`/`H`/`W` branch, only whether the fallback is reached).
#[inline]
fn imrope_axis_for_sector(sector: usize, sections: [u32; 4]) -> MropeAxis {
    if sector % 3 == 1 && sector < 3 * sections[1] as usize {
        MropeAxis::H
    } else if sector % 3 == 2 && sector < 3 * sections[2] as usize {
        MropeAxis::W
    } else if sector.is_multiple_of(3) && sector < 3 * sections[0] as usize {
        MropeAxis::T
    } else {
        MropeAxis::E
    }
}

/// Build the `cos`/`sin` tables for one token's `IMROPE` rotation, given its
/// 3-axis position.
///
/// - `pos`: `[p_t, p_h, p_w]`. For text, all three are the plain token
///   index — this is exactly what makes the degeneracy proof in the module
///   doc comment apply (see also `PartialRopeTable` vs `MropeTable` in the
///   model crate).
/// - `sections`: `ggml`'s `rope.dimension_sections`, e.g. `[11,11,10,0]`.
/// - `n_rot`: number of rotated dims (`64` for Bonsai 2); `cos_out`/
///   `sin_out` each receive `n_rot/2` entries.
/// - `freq_base`: `1e7` for Bonsai 2. `theta_scale = freq_base^(-2/n_rot)`,
///   computed once, then applied by repeated multiplication — **not** by
///   `powf` per index — matching `ops.cpp:5935-5938`, which updates
///   `theta_t`/`theta_h`/`theta_w` by repeated multiplication in exactly
///   this way, on *every* iteration regardless of which axis gets selected
///   that iteration. See [`partial_rope_build_table`] for why a
///   *single*-axis caller must route through this exact function (not an
///   independently-typed equivalent) to get bit-identical results when the
///   three axes coincide.
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `n_rot` is odd.
/// - [`KernelError::UnsupportedOperation`] if `sections` sum to zero, or if
///   the sector selection would need the vision `e` axis (this function's
///   `pos` has no `p_e` slot) — returned as soon as such a sector is
///   reached, so a returned `Err` means `cos_out`/`sin_out` are only
///   partially written and must not be used.
/// - [`KernelError::NamedBufferTooSmall`] if `cos_out.len() < n_rot/2` or
///   `sin_out.len() < n_rot/2`.
#[inline]
pub fn mrope_build_tables(
    pos: [i32; 3],
    sections: [u32; 4],
    n_rot: usize,
    freq_base: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) -> KernelResult<()> {
    if !n_rot.is_multiple_of(2) {
        return Err(KernelError::NotBlockAligned {
            count: n_rot,
            block_size: 2,
        });
    }
    let half = n_rot / 2;

    let sect_dims =
        sections[0] as usize + sections[1] as usize + sections[2] as usize + sections[3] as usize;
    if sect_dims == 0 {
        return Err(KernelError::UnsupportedOperation(
            "mrope_build_tables: sections sum to zero".to_string(),
        ));
    }

    if cos_out.len() < half {
        return Err(KernelError::buffer_too_small(
            "cos_out",
            half,
            cos_out.len(),
        ));
    }
    if sin_out.len() < half {
        return Err(KernelError::buffer_too_small(
            "sin_out",
            half,
            sin_out.len(),
        ));
    }

    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
    let mut theta_t = pos[0] as f32;
    let mut theta_h = pos[1] as f32;
    let mut theta_w = pos[2] as f32;

    for k in 0..half {
        let sector = k % sect_dims;
        let theta = match imrope_axis_for_sector(sector, sections) {
            MropeAxis::T => theta_t,
            MropeAxis::H => theta_h,
            MropeAxis::W => theta_w,
            MropeAxis::E => {
                return Err(KernelError::UnsupportedOperation(format!(
                    "mrope_build_tables: sections {sections:?} select the vision \
                     'e' axis at sector {sector} (rotation index {k} of {half}), \
                     which this text-only API (pos = [t,h,w]) cannot represent"
                )));
            }
        };
        cos_out[k] = theta.cos();
        sin_out[k] = theta.sin();
        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
    }

    Ok(())
}

/// A sections array that maps *every* sector in `0..n_rot/2` to a real axis
/// (`T`, `H` or `W` — never the unrepresentable vision `E` axis), for any
/// `n_rot`. Used by [`partial_rope_build_table`] to drive
/// [`mrope_build_tables`] for a genuinely single-axis table.
///
/// Correctness: `3 * half >= 3` always exceeds every `sector < half`, so
/// whichever residue class (`sector % 3`) a sector falls into, its
/// corresponding branch's range check always passes.
#[inline]
fn single_axis_sections(n_rot: usize) -> [u32; 4] {
    let half = (n_rot / 2).max(1) as u32;
    [half, half, half, 0]
}

/// Build the `cos`/`sin` table for a single position (no separate `t`/`h`/
/// `w` axes) by calling [`mrope_build_tables`] with `pos = [p, p, p]`.
///
/// # Why this exists, not just "call `mrope_build_tables` with `[p,p,p]`
/// from the model crate"
///
/// `f32::powf` (used internally for `theta_scale`) is a transcendental
/// function: two **independently type-written** expressions computing "the
/// same" `powf(freq_base, -2/n_rot)" are not guaranteed to produce a
/// bit-identical `f32` — verified empirically while writing this module's
/// tests (a hand-written reference loop with `const` inputs, which the
/// compiler is free to constant-fold via its own host-time evaluator,
/// differed from this module's runtime `powf` call by 1 ULP at one
/// position). `oxibonsai-model`'s `layers::rope_mrope::PartialRopeTable`
/// (a downstream crate this one does not depend on, hence plain text, not
/// an intra-doc link) therefore builds its table by calling *this exact
/// function*, not by re-deriving the same formula — so it and
/// `layers::rope_mrope::MropeTable` (which also routes through
/// [`mrope_build_tables`]) are bit-identical **by construction** (same
/// compiled code, run with runtime-equal arguments) rather than "by
/// coincidence of two textually similar implementations".
///
/// This is mathematically exact, not an approximation: when `mrope_build_tables`
/// is called with `pos = [p, p, p]`, `theta_t`, `theta_h` and `theta_w`
/// start at the identical bit pattern and are updated by the identical
/// `*= theta_scale` on *every* iteration regardless of which one gets
/// selected for a given sector (`ops.cpp:5935-5938`) — so the value written
/// to `cos_out[k]`/`sin_out[k]` at each `k` cannot depend on the sections
/// array's specific shape, only on whether it is valid (covers every
/// sector without needing the `e` axis). [`single_axis_sections`] is one
/// such valid choice.
///
/// # Errors
///
/// Only [`KernelError::NamedBufferTooSmall`] for `cos_out`/`sin_out` (too
/// short) can occur in practice — `single_axis_sections` never triggers the
/// "sections sum to zero" or "needs the e axis" errors of
/// [`mrope_build_tables`] for any `n_rot >= 1`.
#[inline]
pub fn partial_rope_build_table(
    p: i32,
    n_rot: usize,
    freq_base: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) -> KernelResult<()> {
    mrope_build_tables(
        [p, p, p],
        single_axis_sections(n_rot),
        n_rot,
        freq_base,
        cos_out,
        sin_out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── rope_partial_splithalf_simd ─────────────────────────────

    #[test]
    fn partial_rope_rotates_only_first_n_rot_dims() {
        let head_dim = 8;
        let n_rot = 4; // half = 2
        let input: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
        let cos = vec![1.0f32, 1.0]; // angle 0 -> identity rotation
        let sin = vec![0.0f32, 0.0];
        let mut output = vec![0.0f32; head_dim];

        rope_partial_splithalf_simd(&input, &mut output, head_dim, n_rot, &cos, &sin)
            .expect("valid call");

        // Identity rotation -> rotated part unchanged too.
        assert_eq!(output, input);
    }

    #[test]
    fn partial_rope_leaves_tail_bitwise_untouched() {
        let head_dim = 256;
        let n_rot = 64;
        let half = n_rot / 2;
        let input: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let cos: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).cos()).collect();
        let sin: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut output = vec![f32::NAN; head_dim]; // poison, so a missed copy is obvious

        rope_partial_splithalf_simd(&input, &mut output, head_dim, n_rot, &cos, &sin)
            .expect("valid call");

        for i in n_rot..head_dim {
            assert_eq!(
                output[i].to_bits(),
                input[i].to_bits(),
                "tail dim {i} must be bitwise untouched"
            );
        }
        // The rotated part must actually have changed (sanity: this isn't
        // passing only because the whole buffer got copied through).
        // Index 0 uses angle 0 (cos=1, sin=0), an identity rotation, so
        // check an index with a genuinely non-trivial angle instead.
        assert_ne!(output[1], input[1]);
    }

    #[test]
    fn partial_rope_matches_direct_rotation_formula() {
        let head_dim = 16;
        let n_rot = 8;
        let half = n_rot / 2;
        let input: Vec<f32> = (0..head_dim).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let cos: Vec<f32> = (0..half).map(|i| (i as f32 * 0.2).cos()).collect();
        let sin: Vec<f32> = (0..half).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut output = vec![0.0f32; head_dim];

        rope_partial_splithalf_simd(&input, &mut output, head_dim, n_rot, &cos, &sin)
            .expect("valid call");

        for j in 0..half {
            let expected_lo = input[j] * cos[j] - input[j + half] * sin[j];
            let expected_hi = input[j] * sin[j] + input[j + half] * cos[j];
            assert!((output[j] - expected_lo).abs() < 1e-5);
            assert!((output[j + half] - expected_hi).abs() < 1e-5);
        }
    }

    #[test]
    fn partial_rope_rejects_odd_n_rot() {
        let input = vec![0.0f32; 8];
        let mut output = vec![0.0f32; 8];
        let cos = vec![1.0f32; 4];
        let sin = vec![0.0f32; 4];
        assert!(rope_partial_splithalf_simd(&input, &mut output, 8, 3, &cos, &sin).is_err());
    }

    #[test]
    fn partial_rope_rejects_n_rot_exceeding_head_dim() {
        let input = vec![0.0f32; 4];
        let mut output = vec![0.0f32; 4];
        let cos = vec![1.0f32; 4];
        let sin = vec![0.0f32; 4];
        assert!(rope_partial_splithalf_simd(&input, &mut output, 4, 8, &cos, &sin).is_err());
    }

    #[test]
    fn partial_rope_rejects_short_input() {
        let input = vec![0.0f32; 4]; // head_dim claims 8
        let mut output = vec![0.0f32; 8];
        let cos = vec![1.0f32; 2];
        let sin = vec![0.0f32; 2];
        assert!(rope_partial_splithalf_simd(&input, &mut output, 8, 4, &cos, &sin).is_err());
    }

    #[test]
    fn partial_rope_rejects_short_output() {
        let input = vec![0.0f32; 8];
        let mut output = vec![0.0f32; 4]; // head_dim claims 8
        let cos = vec![1.0f32; 2];
        let sin = vec![0.0f32; 2];
        assert!(rope_partial_splithalf_simd(&input, &mut output, 8, 4, &cos, &sin).is_err());
    }

    // ── mrope_build_tables ──────────────────────────────────────

    const BONSAI2_SECTIONS: [u32; 4] = [11, 11, 10, 0];
    const BONSAI2_N_ROT: usize = 64;
    const BONSAI2_FREQ_BASE: f32 = 1.0e7;

    #[test]
    fn mrope_bonsai2_sections_never_need_the_e_axis() {
        let half = BONSAI2_N_ROT / 2;
        let mut cos_out = vec![0.0f32; half];
        let mut sin_out = vec![0.0f32; half];
        for pos in [0i32, 1, 5, 100, 65535] {
            mrope_build_tables(
                [pos, pos, pos],
                BONSAI2_SECTIONS,
                BONSAI2_N_ROT,
                BONSAI2_FREQ_BASE,
                &mut cos_out,
                &mut sin_out,
            )
            .expect("[11,11,10,0] must cover every sector with t/h/w, never e");
        }
    }

    #[test]
    fn mrope_degenerates_to_standard_rope_for_text_positions() {
        // For p_t=p_h=p_w=p, the result cannot depend on the sections
        // array's shape (only on whether it is valid) -- see
        // `partial_rope_build_table`'s doc comment for why this is checked
        // by calling the shared underlying code path rather than against
        // an independently-typed closed-form reference loop (`powf`
        // constant-folded at compile time in a hand-written test loop is
        // NOT guaranteed bit-identical to the same formula's runtime libm
        // call inside `mrope_build_tables` -- verified empirically: an
        // earlier version of this test compared against such a loop and
        // failed by 1 ULP at pos=1, k=3).
        let half = BONSAI2_N_ROT / 2;
        let mut cos_bonsai2 = vec![0.0f32; half];
        let mut sin_bonsai2 = vec![0.0f32; half];
        let mut cos_single_axis = vec![0.0f32; half];
        let mut sin_single_axis = vec![0.0f32; half];

        for &p in &[0i32, 1, 2, 17, 1000, 65535] {
            mrope_build_tables(
                [p, p, p],
                BONSAI2_SECTIONS,
                BONSAI2_N_ROT,
                BONSAI2_FREQ_BASE,
                &mut cos_bonsai2,
                &mut sin_bonsai2,
            )
            .expect("valid");

            partial_rope_build_table(
                p,
                BONSAI2_N_ROT,
                BONSAI2_FREQ_BASE,
                &mut cos_single_axis,
                &mut sin_single_axis,
            )
            .expect("valid");

            for k in 0..half {
                assert_eq!(
                    cos_bonsai2[k].to_bits(),
                    cos_single_axis[k].to_bits(),
                    "pos={p} k={k}: cos mismatch between BONSAI2_SECTIONS and single_axis_sections"
                );
                assert_eq!(
                    sin_bonsai2[k].to_bits(),
                    sin_single_axis[k].to_bits(),
                    "pos={p} k={k}: sin mismatch between BONSAI2_SECTIONS and single_axis_sections"
                );
            }

            // Also sanity-check (tolerance, not bitwise -- see above) against
            // the direct closed-form theta_k = p * freq_base^(-k/half).
            let theta_scale = BONSAI2_FREQ_BASE.powf(-2.0 / BONSAI2_N_ROT as f32);
            let mut theta = p as f32;
            for k in 0..half {
                assert!(
                    (cos_bonsai2[k] - theta.cos()).abs() < 1e-4,
                    "pos={p} k={k}: cos {} vs closed-form {}",
                    cos_bonsai2[k],
                    theta.cos()
                );
                assert!(
                    (sin_bonsai2[k] - theta.sin()).abs() < 1e-4,
                    "pos={p} k={k}: sin {} vs closed-form {}",
                    sin_bonsai2[k],
                    theta.sin()
                );
                theta *= theta_scale;
            }
        }
    }

    /// The actual package acceptance gate: over 10 000 positions,
    /// `mrope_build_tables` at `[p,p,p]` (any valid sections) must match
    /// [`partial_rope_build_table`] bitwise. This is the kernel-level half
    /// of "`PartialRopeTable` vs `MropeTable` degeneracy bitwise over 10 000
    /// positions" -- the model-crate structs build on exactly these two
    /// functions (see `crates/oxibonsai-model/src/layers/rope_mrope.rs`).
    #[test]
    fn mrope_vs_partial_axis_degeneracy_over_10000_positions() {
        let half = BONSAI2_N_ROT / 2;
        let mut cos_a = vec![0.0f32; half];
        let mut sin_a = vec![0.0f32; half];
        let mut cos_b = vec![0.0f32; half];
        let mut sin_b = vec![0.0f32; half];

        for p in 0..10_000i32 {
            mrope_build_tables(
                [p, p, p],
                BONSAI2_SECTIONS,
                BONSAI2_N_ROT,
                BONSAI2_FREQ_BASE,
                &mut cos_a,
                &mut sin_a,
            )
            .expect("valid");
            partial_rope_build_table(p, BONSAI2_N_ROT, BONSAI2_FREQ_BASE, &mut cos_b, &mut sin_b)
                .expect("valid");

            for k in 0..half {
                assert_eq!(cos_a[k].to_bits(), cos_b[k].to_bits(), "pos={p} k={k}");
                assert_eq!(sin_a[k].to_bits(), sin_b[k].to_bits(), "pos={p} k={k}");
            }
        }
    }

    #[test]
    fn mrope_rejects_zero_sections() {
        let mut cos_out = vec![0.0f32; 4];
        let mut sin_out = vec![0.0f32; 4];
        assert!(
            mrope_build_tables([0, 0, 0], [0, 0, 0, 0], 8, 1e4, &mut cos_out, &mut sin_out)
                .is_err()
        );
    }

    #[test]
    fn mrope_rejects_odd_n_rot() {
        let mut cos_out = vec![0.0f32; 4];
        let mut sin_out = vec![0.0f32; 4];
        assert!(
            mrope_build_tables([0, 0, 0], [1, 1, 1, 0], 7, 1e4, &mut cos_out, &mut sin_out)
                .is_err()
        );
    }

    #[test]
    fn mrope_rejects_short_cos_out() {
        let mut cos_out = vec![0.0f32; 1]; // needs 4
        let mut sin_out = vec![0.0f32; 4];
        assert!(
            mrope_build_tables([0, 0, 0], [1, 1, 1, 0], 8, 1e4, &mut cos_out, &mut sin_out)
                .is_err()
        );
    }

    /// A sections config that genuinely requires the vision `e` axis:
    /// `sections[0] = 0` means the `T`-branch's `sector < 3*sections[0] ==
    /// 0` guard can never pass, so sector 0 (which matches neither the `H`
    /// nor `W` branch either, both requiring `sector % 3 != 0`) falls
    /// through to `e`.
    #[test]
    fn mrope_rejects_sections_that_need_the_e_axis() {
        let mut cos_out = vec![0.0f32; 4];
        let mut sin_out = vec![0.0f32; 4];
        let err = mrope_build_tables([1, 2, 3], [0, 1, 1, 0], 8, 1e4, &mut cos_out, &mut sin_out)
            .expect_err("sector 0 must require the unsupported e axis");
        assert!(matches!(err, KernelError::UnsupportedOperation(_)));
    }
}
