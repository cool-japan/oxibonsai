//! RoPE (Rotary Position Embedding) scaling variants for extended context.
//!
//! Strategies:
//! - `None`: standard RoPE — no scaling applied.
//! - `Linear`: scale all frequencies by 1/s (simple, fast, loses high-freq info).
//! - `DynamicNtk`: NTK-aware scaling applied dynamically at inference time.
//! - `Llama31`: LLaMA 3.1 / LLaMA 3.2 scaling with low/high frequency blending.
//! - `LongRope`: LongRoPE with per-dimension rescale factors.
//!
//! ## Standard RoPE Frequency Convention
//!
//! For dimension index `i` in `[0, head_dim/2)`:
//!
//! ```text
//! freq_i = 1.0 / base^(2*i / head_dim)
//! ```
//!
//! The actual rotation angle at position `p` is `theta_i * p = freq_i * p`.

use thiserror::Error;

// ─── RopeScalingStrategy ─────────────────────────────────────────────────────

/// A RoPE scaling strategy for extended context inference.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingStrategy {
    /// No scaling — standard RoPE with unmodified frequencies.
    None,

    /// Linear scaling: divide all frequencies by `scale_factor`.
    ///
    /// Equivalent to multiplying the effective sequence length by `scale_factor`.
    /// Fast and simple but degrades quality for high-frequency dimensions.
    Linear {
        /// Must be >= 1.0.
        scale_factor: f32,
    },

    /// Dynamic NTK scaling: scales the RoPE base frequency at inference time.
    ///
    /// The effective base is computed as:
    /// ```text
    /// effective_base = base * (s * max_pos / orig_max_pos - (s - 1))^(d / (d - 2))
    /// ```
    /// where `s = current_seq_len / original_max_position`.
    ///
    /// When `current_seq_len <= original_max_position`, standard frequencies are used.
    DynamicNtk {
        /// The maximum sequence length used during pretraining.
        original_max_position: usize,
        /// Base frequency (e.g. 10000.0 for most models, 500000.0 for LLaMA 3.1).
        base: f32,
    },

    /// LLaMA 3.1 / LLaMA 3.2 scaling: blends original and scaled frequencies
    /// per dimension based on wavelength thresholds.
    ///
    /// Dimensions with long wavelengths (low frequency) are interpolated;
    /// dimensions with short wavelengths (high frequency) are left unmodified;
    /// intermediate dimensions are smoothly blended.
    Llama31 {
        /// The maximum sequence length used during pretraining.
        original_max_position: usize,
        /// Context extension factor (e.g. 8.0 for 8× context extension).
        scale_factor: f32,
        /// Low-frequency threshold factor (default 1.0 in LLaMA 3.1).
        low_freq_factor: f32,
        /// High-frequency threshold factor (default 4.0 in LLaMA 3.1).
        high_freq_factor: f32,
        /// Base frequency.
        base: f32,
    },

    /// LongRoPE: per-dimension rescale based on externally computed factors.
    ///
    /// Each of the `head_dim/2` frequency dimensions is divided by the
    /// corresponding rescale factor. Factors are typically derived from an
    /// evolutionary search optimising perplexity on long documents.
    LongRope {
        /// Per-dimension rescale factors; length must equal `head_dim / 2`.
        rescale_factors: Vec<f32>,
        /// Original pretraining context length (used for reference only).
        original_max_position: usize,
    },

    /// YaRN (Yet Another RoPE extensioN): NTK-by-parts frequency
    /// interpolation plus an attention-temperature correction ("mscale").
    ///
    /// This is the algorithm GGUF's `<arch>.rope.scaling.type = "yarn"`
    /// metadata implies (M-08) — the same one implemented by `ggml`/
    /// llama.cpp's `rope_yarn` / `ggml_rope_yarn_corr_dims` and by HF
    /// `transformers`' `_compute_yarn_parameters`. Unlike [`Self::Linear`]
    /// (which divides every frequency by `scale_factor` uniformly) and
    /// [`Self::DynamicNtk`] (which inflates the base), YaRN blends, per
    /// frequency-pair dimension `i`, between the unscaled ("extrapolated")
    /// frequency and the linearly-interpolated (`freq / factor`) frequency,
    /// using a ramp over a correction-dimension range derived from
    /// `beta_fast` / `beta_slow`: dimensions with short wavelengths (high
    /// frequency) keep their original frequency, dimensions with long
    /// wavelengths (low frequency) are fully interpolated, and dimensions in
    /// between are smoothly blended. A separate temperature multiplier
    /// ("mscale", see [`yarn_mscale`]) is folded directly into the
    /// `RopeTable` cos/sin table rather than into the frequency vector —
    /// see `RopeTable::new_with_scaling` in `crate::layers::rope`.
    Yarn {
        /// Pretraining context length before extension (GGUF
        /// `original_context_length`).
        original_max_position: usize,
        /// Context-extension factor `s = extended / original` (GGUF
        /// `factor`). Must be `>= 1.0`.
        factor: f32,
        /// NTK-by-parts low correction-dimension parameter (GGUF
        /// `beta_fast`). Upstream default: `32.0`.
        beta_fast: f32,
        /// NTK-by-parts high correction-dimension parameter (GGUF
        /// `beta_slow`). Upstream default: `1.0`.
        beta_slow: f32,
        /// Attention-temperature multiplier override (GGUF `attn_factor`).
        /// `None` means "compute the standard default" — see [`yarn_mscale`].
        attn_factor: Option<f32>,
    },
}

// ─── RopeScalingError ────────────────────────────────────────────────────────

/// Errors produced by RoPE scaling operations.
#[derive(Debug, Error)]
pub enum RopeScalingError {
    /// `head_dim` was zero or odd; must be a positive even integer.
    #[error("head_dim {0} must be even and > 0")]
    InvalidHeadDim(usize),

    /// The length of `rescale_factors` does not match `head_dim / 2`.
    #[error("rescale_factors length {got} != head_dim/2 = {expected}")]
    RescaleFactorLengthMismatch { got: usize, expected: usize },

    /// `scale_factor` was less than 1.0; scaling must not compress the context.
    #[error("scale_factor must be >= 1.0, got {0}")]
    InvalidScaleFactor(f32),

    /// The length of the `q` or `k` slice did not match `head_dim`.
    #[error("q/k length {got} != head_dim {expected}")]
    VecLengthMismatch { got: usize, expected: usize },
}

// ─── compute_rope_frequencies ────────────────────────────────────────────────

/// Compute RoPE frequencies given a scaling strategy.
///
/// Returns `head_dim / 2` angular frequency values (θ_i for i = 0..head_dim/2).
/// The rotation angle at absolute position `p` is `θ_i * p`.
///
/// # `RopeScalingStrategy::Yarn` drops `attn_factor` (minor finding)
///
/// This function's `Vec<f32>` return type carries *frequencies only* — there
/// is nowhere in that shape to also return YaRN's attention-temperature
/// multiplier ("mscale"), so the `Yarn` arm silently ignores the
/// `attn_factor` field entirely (it destructures it away with `..`). A
/// caller that pairs this function's output with [`apply_rope_with_freqs`]
/// for a `Yarn` strategy therefore applies the *frequency* blend correctly
/// but silently loses the mscale multiplier — query/key pairs come out
/// unrotated-in-magnitude, only rotated-in-angle.
/// [`RopeTable::new_with_scaling`](crate::layers::rope::RopeTable::new_with_scaling)
/// does **not** have this gap: it calls `yarn_inv_freq_f64` (`pub(crate)`,
/// not part of this module's public API, hence plain code text rather than
/// a doc link here) and [`yarn_mscale`] directly — bypassing this function
/// for the `Yarn` case specifically — and folds mscale into the
/// precomputed cos/sin table instead, which is the path GGUF
/// `<arch>.rope.scaling.type = "yarn"` metadata is actually routed through
/// end to end. Any *new* caller that wants both the frequencies and the
/// mscale from this function's `Yarn` arm must additionally call
/// [`yarn_mscale`] itself with the same `factor`/`attn_factor` and apply it
/// to its own rotated output.
///
/// # Errors
///
/// - [`RopeScalingError::InvalidHeadDim`] if `head_dim` is zero or odd.
/// - [`RopeScalingError::InvalidScaleFactor`] if `scale_factor < 1.0` (Linear strategy).
/// - [`RopeScalingError::RescaleFactorLengthMismatch`] if `rescale_factors.len() != head_dim/2` (LongRope strategy).
pub fn compute_rope_frequencies(
    head_dim: usize,
    base: f32,
    strategy: &RopeScalingStrategy,
    current_seq_len: usize,
) -> Result<Vec<f32>, RopeScalingError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(RopeScalingError::InvalidHeadDim(head_dim));
    }

    match strategy {
        RopeScalingStrategy::None => Ok(standard_frequencies(head_dim, base)),

        RopeScalingStrategy::Linear { scale_factor } => {
            if *scale_factor < 1.0 {
                return Err(RopeScalingError::InvalidScaleFactor(*scale_factor));
            }
            let freqs = standard_frequencies(head_dim, base);
            Ok(freqs.into_iter().map(|f| f / scale_factor).collect())
        }

        RopeScalingStrategy::DynamicNtk {
            original_max_position,
            base: ntk_base,
        } => {
            let effective_base =
                dynamic_ntk_base(*ntk_base, head_dim, *original_max_position, current_seq_len);
            Ok(standard_frequencies(head_dim, effective_base))
        }

        RopeScalingStrategy::Llama31 {
            original_max_position,
            scale_factor,
            low_freq_factor,
            high_freq_factor,
            base: llama_base,
        } => {
            if *scale_factor < 1.0 {
                return Err(RopeScalingError::InvalidScaleFactor(*scale_factor));
            }
            Ok(llama31_frequencies(
                head_dim,
                *llama_base,
                *original_max_position,
                *scale_factor,
                *low_freq_factor,
                *high_freq_factor,
            ))
        }

        RopeScalingStrategy::LongRope {
            rescale_factors,
            original_max_position: _,
        } => {
            let half_dim = head_dim / 2;
            if rescale_factors.len() != half_dim {
                return Err(RopeScalingError::RescaleFactorLengthMismatch {
                    got: rescale_factors.len(),
                    expected: half_dim,
                });
            }
            let freqs = standard_frequencies(head_dim, base);
            Ok(freqs
                .into_iter()
                .zip(rescale_factors.iter())
                .map(|(f, &r)| {
                    // Guard against zero rescale factors to avoid division by zero.
                    if r.abs() < f32::EPSILON {
                        f
                    } else {
                        f / r
                    }
                })
                .collect())
        }

        RopeScalingStrategy::Yarn {
            original_max_position,
            factor,
            beta_fast,
            beta_slow,
            ..
        } => {
            if *factor < 1.0 {
                return Err(RopeScalingError::InvalidScaleFactor(*factor));
            }
            Ok(yarn_inv_freq(
                head_dim,
                base,
                *original_max_position,
                *factor,
                *beta_fast,
                *beta_slow,
            ))
        }
    }
}

// ─── apply_rope_with_freqs ───────────────────────────────────────────────────

/// Apply standard RoPE rotation to a query/key vector at position `pos`.
///
/// Uses precomputed frequencies from [`compute_rope_frequencies`].
/// Rotates pairs `(v[i], v[i + half_dim])` in-place for both `q` and `k`.
///
/// At position 0 the rotation is the identity (cos(0)=1, sin(0)=0).
///
/// # Errors
///
/// - [`RopeScalingError::InvalidHeadDim`] if `freqs.len() * 2` is zero or odd.
/// - [`RopeScalingError::VecLengthMismatch`] if `q.len()` or `k.len()` ≠ `freqs.len() * 2`.
pub fn apply_rope_with_freqs(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    freqs: &[f32],
) -> Result<(), RopeScalingError> {
    let half = freqs.len();
    let head_dim = half * 2;

    if half == 0 {
        return Err(RopeScalingError::InvalidHeadDim(0));
    }

    if q.len() != head_dim {
        return Err(RopeScalingError::VecLengthMismatch {
            got: q.len(),
            expected: head_dim,
        });
    }
    if k.len() != head_dim {
        return Err(RopeScalingError::VecLengthMismatch {
            got: k.len(),
            expected: head_dim,
        });
    }

    for i in 0..half {
        let angle = pos as f32 * freqs[i];
        let (sin_a, cos_a) = angle.sin_cos();

        // Rotate query pair
        let q0 = q[i];
        let q1 = q[half + i];
        q[i] = q0 * cos_a - q1 * sin_a;
        q[half + i] = q0 * sin_a + q1 * cos_a;

        // Rotate key pair
        let k0 = k[i];
        let k1 = k[half + i];
        k[i] = k0 * cos_a - k1 * sin_a;
        k[half + i] = k0 * sin_a + k1 * cos_a;
    }

    Ok(())
}

// ─── dynamic_ntk_base ────────────────────────────────────────────────────────

/// Compute the effective base frequency for Dynamic NTK scaling.
///
/// When `current_seq_len <= original_max_position`, returns `base` unchanged
/// (no scaling needed). Otherwise, the effective base is inflated so that the
/// higher-order (lower-frequency) dimensions can represent longer sequences
/// without aliasing.
///
/// Formula:
/// ```text
/// s  = current_seq_len / original_max_position
/// effective_base = base * s^(d / (d - 2))
/// ```
///
/// where `d = head_dim`. This matches the formulation in Su et al. 2023
/// ("Scaling RoPE beyond Training Context") and the HuggingFace implementation.
pub fn dynamic_ntk_base(
    base: f32,
    head_dim: usize,
    original_max_position: usize,
    current_seq_len: usize,
) -> f32 {
    if current_seq_len <= original_max_position || original_max_position == 0 {
        return base;
    }

    let s = current_seq_len as f32 / original_max_position as f32;

    // NTK exponent: d / (d - 2); fallback to 1.0 for tiny head dims.
    let ntk_exp = if head_dim > 2 {
        head_dim as f32 / (head_dim as f32 - 2.0)
    } else {
        1.0
    };

    base * s.powf(ntk_exp)
}

// ─── llama31_frequencies ─────────────────────────────────────────────────────

/// Compute LLaMA 3.1 per-dimension frequencies.
///
/// Implements the frequency blending described in the LLaMA 3.1 technical
/// report (Meta AI, 2024). For each frequency dimension `i`:
///
/// 1. Compute the standard frequency `f_i = 1 / base^(2i/d)`.
/// 2. Compute the wavelength `λ = 2π / f_i`.
/// 3. Determine low/high wavelength thresholds from the original context:
///    - `low_thresh  = original_max_position / high_freq_factor`
///    - `high_thresh = original_max_position / low_freq_factor`
/// 4. Blend:
///    - `λ < low_thresh`  → use `f_i` unchanged (high frequency, no scaling).
///    - `λ > high_thresh` → divide `f_i` by `scale_factor` (pure interpolation).
///    - otherwise         → smooth ramp between the two.
///
/// When `scale_factor == 1.0` this returns exactly the standard frequencies.
pub fn llama31_frequencies(
    head_dim: usize,
    base: f32,
    original_max_position: usize,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let orig = original_max_position as f32;
    let two_pi = 2.0 * std::f32::consts::PI;

    // Wavelength thresholds
    // high_freq dimensions have wavelength < low_thresh  → not scaled
    // low_freq  dimensions have wavelength > high_thresh → scaled by 1/scale_factor
    let low_thresh = if high_freq_factor.abs() > f32::EPSILON {
        orig / high_freq_factor
    } else {
        f32::MAX
    };
    let high_thresh = if low_freq_factor.abs() > f32::EPSILON {
        orig / low_freq_factor
    } else {
        f32::MAX
    };

    (0..half_dim)
        .map(|i| {
            let freq = standard_freq(i, head_dim, base);
            if (scale_factor - 1.0).abs() < f32::EPSILON {
                // scale_factor == 1 → no change regardless of wavelength
                return freq;
            }

            let wavelength = if freq > f32::EPSILON {
                two_pi / freq
            } else {
                f32::MAX
            };

            if wavelength < low_thresh {
                // High-frequency dimension — leave unchanged.
                freq
            } else if wavelength > high_thresh {
                // Low-frequency dimension — apply full linear scaling.
                freq / scale_factor
            } else {
                // Intermediate — smooth linear blend.
                // ramp ∈ [0, 1]: 0 at high_thresh boundary, 1 at low_thresh boundary.
                let range = high_thresh - low_thresh;
                let ramp = if range > f32::EPSILON {
                    (wavelength - low_thresh) / range
                } else {
                    0.5
                };
                // ramp=0 → not scaled; ramp=1 → fully scaled.
                // Blend between unscaled (1.0 weight at ramp=0) and scaled (ramp=1).
                let scaled_freq = freq / scale_factor;
                (1.0 - ramp) * freq + ramp * scaled_freq
            }
        })
        .collect()
}

// ─── YaRN (NTK-by-parts) frequency blend ────────────────────────────────────

/// "Correction dimension" boundary used by YaRN's NTK-by-parts ramp: the
/// frequency-pair index at which a rotation with `num_rotations` full turns
/// over `original_max_position` tokens occurs, in the standard RoPE
/// frequency schedule for `(head_dim, base)`.
///
/// Matches `ggml_rope_yarn_corr_dim` / HF `transformers`'
/// `find_correction_dim` exactly:
/// `head_dim * ln(original_max_position / (num_rotations * 2π)) / (2 * ln(base))`.
fn yarn_correction_dim_f64(
    head_dim: usize,
    base: f64,
    original_max_position: usize,
    num_rotations: f64,
) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (head_dim as f64 * (original_max_position as f64 / (num_rotations * two_pi)).ln())
        / (2.0 * base.ln())
}

/// `f64`-precision core of the YaRN NTK-by-parts inverse-frequency blend —
/// the single source of truth for the formula; [`yarn_inv_freq`] (`f32`) is
/// a thin cast wrapper around this.
///
/// `RopeTable::new_with_scaling` (`crate::layers::rope`) calls this directly
/// — rather than going through the `f32`-returning [`compute_rope_frequencies`]
/// — so it can carry full precision through the `pos * freq` angle
/// computation used to build the cos/sin table. Once a frequency is rounded
/// to `f32`, the resulting *angle*'s absolute error at position `p` is
/// bounded by roughly `p * freq * 2⁻²⁴` (the `f32` machine epsilon): already
/// on the order of `1e-3` at `p ≈ 16384` for a high-frequency dimension —
/// far too coarse for a "matches a reference to 1e-6" check (M-08). `cos`
/// and `sin` themselves are always bounded in `[-1, 1]`, so rounding *them*
/// to `f32` at the very end costs only `~6e-8` absolute, independent of
/// position — hence threading `f64` through to that point rather than
/// through the frequency alone.
///
/// Returns `head_dim / 2` frequencies. Degenerates exactly to the standard
/// (unscaled) frequency for every dimension when `factor == 1.0` — the blend
/// formula collapses to `freq_extrap` on its own for any ramp value, so no
/// special case is needed for it. Guarded against `original_max_position ==
/// 0` (which would otherwise divide by zero / take `ln` of zero inside the
/// correction-dimension formula) and `head_dim < 2` (no frequency pairs to
/// blend).
pub(crate) fn yarn_inv_freq_f64(
    head_dim: usize,
    base: f64,
    original_max_position: usize,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let half_dim = head_dim / 2;
    let standard_freq_f64 = |i: usize| -> f64 { 1.0 / base.powf(2.0 * i as f64 / head_dim as f64) };

    if original_max_position == 0 || half_dim == 0 {
        return (0..half_dim).map(standard_freq_f64).collect();
    }

    // Bounds fix (finding B3): `low` and `high` are correction-
    // dimension *indices into the standard RoPE frequency schedule*, whose
    // valid range is `0..head_dim` conceptually (ggml computes them against
    // `n_dims = head_dim`), even though only the first `half_dim` of them
    // are ever used to index `freq_extrap`/`freq_interp` below. Both
    // `ggml_rope_yarn_corr_dims` (`fork/ggml_src_ggml.c:4434-4435`:
    // `dims[0] = MAX(0, start)` — no upper clamp on `low` — and `dims[1] =
    // MIN(n_dims - 1, end)` with `n_dims = head_dim`) and HF transformers'
    // `find_correction_range` (`max(low, 0)`, `min(high, dim - 1)` with
    // `dim = head_dim`) agree on this exactly. Clamping `high` to
    // `half_dim - 1` instead of `head_dim - 1` (the bug this replaces)
    // silently drags the ramp's upper boundary inward whenever the raw
    // correction dimension falls in `(half_dim - 1, head_dim - 1]` —
    // unreachable for Bonsai-8B and the 27B (their raw `high` stays well
    // under `half_dim - 1`) but a real formula error in the general case,
    // caught by `yarn_high_correction_dim_clamps_to_head_dim_not_half_dim`
    // below.
    let low = yarn_correction_dim_f64(head_dim, base, original_max_position, beta_fast)
        .floor()
        .max(0.0);
    let high = yarn_correction_dim_f64(head_dim, base, original_max_position, beta_slow)
        .ceil()
        .min((head_dim - 1) as f64);
    // Reference implementations nudge the denominator away from zero for the
    // degenerate case where the two correction dims coincide, rather than
    // dividing by zero.
    let denom = (high - low).max(0.001);

    (0..half_dim)
        .map(|i| {
            let freq_extrap = standard_freq_f64(i);
            let freq_interp = freq_extrap / factor;
            // `ramp == 0` at/before `low`  → fully extrapolated (unscaled,
            //                                 short-wavelength / high-freq).
            // `ramp == 1` at/after  `high` → fully interpolated (`/factor`,
            //                                 long-wavelength / low-freq).
            let ramp = ((i as f64 - low) / denom).clamp(0.0, 1.0);
            freq_interp * ramp + freq_extrap * (1.0 - ramp)
        })
        .collect()
}

/// `f32` wrapper around [`yarn_inv_freq_f64`] (the canonical formula) for
/// [`compute_rope_frequencies`]'s uniform `Vec<f32>` contract. See
/// [`yarn_inv_freq_f64`]'s doc comment for why `RopeTable::new_with_scaling`
/// calls the `f64` core directly instead of using this wrapper's output.
fn yarn_inv_freq(
    head_dim: usize,
    base: f32,
    original_max_position: usize,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    yarn_inv_freq_f64(
        head_dim,
        base as f64,
        original_max_position,
        factor as f64,
        beta_fast as f64,
        beta_slow as f64,
    )
    .into_iter()
    .map(|f| f as f32)
    .collect()
}

/// YaRN attention-temperature multiplier ("mscale"): compensates for the
/// attention-entropy increase that context extension causes by uniformly
/// scaling the rotated query/key pair.
///
/// `RopeTable::new_with_scaling` (`crate::layers::rope`) folds this directly
/// into the precomputed cos/sin table, matching `ggml`'s `rope_yarn`, which
/// multiplies both `cos_theta` and `sin_theta` by the same value — so no
/// attention or kernel code needs to change to pick it up.
///
/// `attn_factor_override` corresponds to GGUF's
/// `<arch>.rope.scaling.attn_factor`; when `None`, it seeds the multiplier
/// at the neutral default of `1.0`.
///
/// **Convention: multiply, not replace (minor finding).** GGUF's
/// `attn_factor` key is defined by llama.cpp, which *seeds* `mscale` with
/// `hparams.rope_attn_factor` (default `1.0`) and then multiplies in the
/// standard correction: `mscale *= 1.0 + 0.1 * ln(1.0 / freq_scale)`, i.e.
/// `mscale *= 1.0 + 0.1 * ln(factor)` (`fork/src_llama-hparams.h:135`,
/// `fork/src_llama-model.cpp:1448`, `fork/models/ops.cpp:5849`). HF
/// `transformers` instead *replaces* the computed value outright when an
/// override is given. This function follows llama.cpp's multiply
/// convention because GGUF's `attn_factor` key is llama.cpp's, not HF's.
/// With `attn_factor_override = None` (the seed defaults to `1.0`) the two
/// conventions are identical — in particular, unreachable on Bonsai-8B,
/// which carries no `attn_factor` key at all.
pub fn yarn_mscale(factor: f32, attn_factor_override: Option<f32>) -> f32 {
    let seed = attn_factor_override.unwrap_or(1.0);
    if factor <= 1.0 {
        seed
    } else {
        seed * (0.1 * factor.ln() + 1.0)
    }
}

// ─── FreqStats ───────────────────────────────────────────────────────────────

/// Statistics summarising a set of RoPE frequencies.
#[derive(Debug, Clone)]
pub struct FreqStats {
    /// Smallest frequency value in the set.
    pub min_freq: f32,
    /// Largest frequency value in the set.
    pub max_freq: f32,
    /// Arithmetic mean of all frequency values.
    pub mean_freq: f32,
    /// Approximate maximum representable context: `1 / min_freq`.
    ///
    /// The lowest frequency completes one full rotation in roughly this many
    /// tokens, giving an upper bound on useful positional distinguishability.
    pub effective_context: f32,
}

impl FreqStats {
    /// Compute statistics from a slice of frequencies.
    ///
    /// Returns zeroed stats for an empty slice.
    pub fn compute(freqs: &[f32]) -> Self {
        if freqs.is_empty() {
            return Self {
                min_freq: 0.0,
                max_freq: 0.0,
                mean_freq: 0.0,
                effective_context: 0.0,
            };
        }

        let mut min_freq = freqs[0];
        let mut max_freq = freqs[0];
        let mut sum = 0.0_f64;

        for &f in freqs {
            if f < min_freq {
                min_freq = f;
            }
            if f > max_freq {
                max_freq = f;
            }
            sum += f as f64;
        }

        let mean_freq = (sum / freqs.len() as f64) as f32;
        let effective_context = if min_freq > f32::EPSILON {
            1.0 / min_freq
        } else {
            f32::INFINITY
        };

        Self {
            min_freq,
            max_freq,
            mean_freq,
            effective_context,
        }
    }

    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "FreqStats {{ min={:.6e}, max={:.6e}, mean={:.6e}, effective_ctx={:.1} }}",
            self.min_freq, self.max_freq, self.mean_freq, self.effective_context
        )
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Standard RoPE frequency for dimension `i`:
/// `freq = 1 / base^(2i / head_dim)`.
#[inline]
fn standard_freq(i: usize, head_dim: usize, base: f32) -> f32 {
    1.0_f32 / base.powf(2.0 * i as f32 / head_dim as f32)
}

/// Compute standard (unscaled) RoPE frequencies for all `head_dim/2` pairs.
fn standard_frequencies(head_dim: usize, base: f32) -> Vec<f32> {
    let half_dim = head_dim / 2;
    (0..half_dim)
        .map(|i| standard_freq(i, head_dim, base))
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: f32 = 10_000.0;
    const HEAD_DIM: usize = 64;

    fn standard_freqs_ref(head_dim: usize, base: f32) -> Vec<f32> {
        let half = head_dim / 2;
        (0..half)
            .map(|i| 1.0_f32 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect()
    }

    // ── no_scaling_standard_freqs ────────────────────────────────────────────

    #[test]
    fn no_scaling_standard_freqs() {
        let freqs = compute_rope_frequencies(HEAD_DIM, BASE, &RopeScalingStrategy::None, 4096)
            .expect("None strategy should succeed");

        let expected = standard_freqs_ref(HEAD_DIM, BASE);
        assert_eq!(freqs.len(), expected.len());
        for (i, (got, exp)) in freqs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "freq[{i}]: got {got}, expected {exp}"
            );
        }
    }

    // ── linear_scaling_divides_freqs ─────────────────────────────────────────

    #[test]
    fn linear_scaling_divides_freqs() {
        let scale = 4.0_f32;
        let freqs = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::Linear {
                scale_factor: scale,
            },
            4096,
        )
        .expect("Linear strategy should succeed");

        let standard = standard_freqs_ref(HEAD_DIM, BASE);
        for (i, (got, std_f)) in freqs.iter().zip(standard.iter()).enumerate() {
            let expected = std_f / scale;
            assert!(
                (got - expected).abs() < 1e-6,
                "freq[{i}]: got {got}, expected {expected}"
            );
        }
    }

    // ── linear_scaling_scale_1_unchanged ─────────────────────────────────────

    #[test]
    fn linear_scaling_scale_1_unchanged() {
        let freqs_linear = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::Linear { scale_factor: 1.0 },
            4096,
        )
        .expect("Linear scale=1 should succeed");

        let freqs_none = compute_rope_frequencies(HEAD_DIM, BASE, &RopeScalingStrategy::None, 4096)
            .expect("None strategy should succeed");

        for (i, (a, b)) in freqs_linear.iter().zip(freqs_none.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "freq[{i}]: linear scale=1 got {a}, None got {b}"
            );
        }
    }

    // ── dynamic_ntk_longer_seq_higher_base ───────────────────────────────────

    #[test]
    fn dynamic_ntk_longer_seq_higher_base() {
        let orig = 4096_usize;
        let base_short = dynamic_ntk_base(BASE, HEAD_DIM, orig, orig);
        let base_long = dynamic_ntk_base(BASE, HEAD_DIM, orig, orig * 4);

        assert!(
            base_long > base_short,
            "longer sequence should produce higher effective base: short={base_short}, long={base_long}"
        );
    }

    // ── dynamic_ntk_at_orig_len_unchanged ────────────────────────────────────

    #[test]
    fn dynamic_ntk_at_orig_len_unchanged() {
        let orig = 4096_usize;
        let effective = dynamic_ntk_base(BASE, HEAD_DIM, orig, orig);
        assert!(
            (effective - BASE).abs() < 1e-3,
            "at original length, effective base should equal base: {effective} vs {BASE}"
        );

        // Also verify via compute_rope_frequencies
        let freqs_ntk = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::DynamicNtk {
                original_max_position: orig,
                base: BASE,
            },
            orig,
        )
        .expect("DynamicNtk at orig len should succeed");

        let freqs_none = standard_freqs_ref(HEAD_DIM, BASE);
        for (i, (a, b)) in freqs_ntk.iter().zip(freqs_none.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "freq[{i}]: NTK at orig len got {a}, standard got {b}"
            );
        }
    }

    // ── llama31_freqs_length ──────────────────────────────────────────────────

    #[test]
    fn llama31_freqs_length() {
        let freqs = llama31_frequencies(HEAD_DIM, BASE, 8192, 8.0, 1.0, 4.0);
        assert_eq!(
            freqs.len(),
            HEAD_DIM / 2,
            "llama31_frequencies must return head_dim/2 values"
        );
    }

    // ── llama31_freqs_positive ────────────────────────────────────────────────

    #[test]
    fn llama31_freqs_positive() {
        let freqs = llama31_frequencies(HEAD_DIM, BASE, 8192, 8.0, 1.0, 4.0);
        for (i, &f) in freqs.iter().enumerate() {
            assert!(f > 0.0, "freq[{i}] = {f} is not positive");
        }
    }

    // ── llama31_scale_1_unchanged ─────────────────────────────────────────────

    #[test]
    fn llama31_scale_1_unchanged() {
        let freqs_scaled = llama31_frequencies(HEAD_DIM, BASE, 8192, 1.0, 1.0, 4.0);
        let freqs_standard = standard_freqs_ref(HEAD_DIM, BASE);

        for (i, (got, exp)) in freqs_scaled.iter().zip(freqs_standard.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "freq[{i}]: scale=1 got {got}, standard got {exp}"
            );
        }
    }

    // ── longrope_freqs_uses_factors ───────────────────────────────────────────

    #[test]
    fn longrope_freqs_uses_factors() {
        let half = HEAD_DIM / 2;
        let factors: Vec<f32> = (0..half).map(|i| 1.0 + i as f32 * 0.1).collect();

        let freqs = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::LongRope {
                rescale_factors: factors.clone(),
                original_max_position: 4096,
            },
            8192,
        )
        .expect("LongRope should succeed");

        let standard = standard_freqs_ref(HEAD_DIM, BASE);
        for (i, ((got, std_f), &r)) in freqs
            .iter()
            .zip(standard.iter())
            .zip(factors.iter())
            .enumerate()
        {
            let expected = std_f / r;
            assert!(
                (got - expected).abs() < 1e-6,
                "freq[{i}]: got {got}, expected {expected} (std={std_f}, factor={r})"
            );
        }
    }

    // ── longrope_wrong_factor_count_error ─────────────────────────────────────

    #[test]
    fn longrope_wrong_factor_count_error() {
        let wrong_factors = vec![1.0_f32; 10]; // head_dim/2 = 32, not 10
        let result = compute_rope_frequencies(
            HEAD_DIM,
            BASE,
            &RopeScalingStrategy::LongRope {
                rescale_factors: wrong_factors,
                original_max_position: 4096,
            },
            8192,
        );
        assert!(
            matches!(
                result,
                Err(RopeScalingError::RescaleFactorLengthMismatch {
                    got: 10,
                    expected: 32
                })
            ),
            "expected RescaleFactorLengthMismatch, got: {result:?}"
        );
    }

    // ── apply_rope_zero_pos_identity ──────────────────────────────────────────

    #[test]
    fn apply_rope_zero_pos_identity() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let mut q: Vec<f32> = (0..HEAD_DIM).map(|x| x as f32 * 0.1).collect();
        let mut k: Vec<f32> = (0..HEAD_DIM).map(|x| x as f32 * 0.2 + 1.0).collect();
        let q_orig = q.clone();
        let k_orig = k.clone();

        apply_rope_with_freqs(&mut q, &mut k, 0, &freqs).expect("apply at pos=0 should succeed");

        for i in 0..HEAD_DIM {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-5,
                "q[{i}] should be unchanged at pos=0: {} → {}",
                q_orig[i],
                q[i]
            );
            assert!(
                (k[i] - k_orig[i]).abs() < 1e-5,
                "k[{i}] should be unchanged at pos=0: {} → {}",
                k_orig[i],
                k[i]
            );
        }
    }

    // ── apply_rope_changes_at_pos1 ────────────────────────────────────────────

    #[test]
    fn apply_rope_changes_at_pos1() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let mut q: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.5).collect();
        let mut k: Vec<f32> = (0..HEAD_DIM).map(|x| (x as f32 + 1.0) * 0.3).collect();
        let q_orig = q.clone();

        apply_rope_with_freqs(&mut q, &mut k, 1, &freqs).expect("apply at pos=1 should succeed");

        // At least some values should have changed
        let changed = q
            .iter()
            .zip(q_orig.iter())
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(
            changed,
            "apply_rope_with_freqs at pos=1 should modify values"
        );
    }

    // ── apply_rope_invalid_head_dim_error ─────────────────────────────────────

    #[test]
    fn apply_rope_invalid_head_dim_error() {
        // Build a freq slice of length 0 to trigger InvalidHeadDim(0).
        let freqs: Vec<f32> = vec![];
        let mut q = vec![1.0_f32];
        let mut k = vec![1.0_f32];
        let result = apply_rope_with_freqs(&mut q, &mut k, 0, &freqs);
        assert!(
            matches!(result, Err(RopeScalingError::InvalidHeadDim(0))),
            "empty freqs should return InvalidHeadDim(0), got: {result:?}"
        );
    }

    // ── freq_stats_min_max_ordering ───────────────────────────────────────────

    #[test]
    fn freq_stats_min_max_ordering() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let stats = FreqStats::compute(&freqs);
        assert!(
            stats.min_freq <= stats.mean_freq,
            "min ({}) should be <= mean ({})",
            stats.min_freq,
            stats.mean_freq
        );
        assert!(
            stats.mean_freq <= stats.max_freq,
            "mean ({}) should be <= max ({})",
            stats.mean_freq,
            stats.max_freq
        );
    }

    // ── freq_stats_effective_context_positive ─────────────────────────────────

    #[test]
    fn freq_stats_effective_context_positive() {
        let freqs = standard_freqs_ref(HEAD_DIM, BASE);
        let stats = FreqStats::compute(&freqs);
        assert!(
            stats.effective_context > 0.0,
            "effective_context should be positive, got {}",
            stats.effective_context
        );
    }

    // ── compute_freqs_invalid_dim_error ───────────────────────────────────────

    #[test]
    fn compute_freqs_invalid_dim_error() {
        let result = compute_rope_frequencies(0, BASE, &RopeScalingStrategy::None, 4096);
        assert!(
            matches!(result, Err(RopeScalingError::InvalidHeadDim(0))),
            "head_dim=0 should return InvalidHeadDim(0), got: {result:?}"
        );

        let result_odd = compute_rope_frequencies(3, BASE, &RopeScalingStrategy::None, 4096);
        assert!(
            matches!(result_odd, Err(RopeScalingError::InvalidHeadDim(3))),
            "head_dim=3 (odd) should return InvalidHeadDim(3), got: {result_odd:?}"
        );
    }

    // ── M-08: YaRN scaling ───────────────────────────────────────────────────

    fn bonsai_8b_yarn() -> RopeScalingStrategy {
        // The exact values `models/Bonsai-8B.gguf` ships (M-08): a real,
        // shipped model whose long-context output was silently wrong before
        // YaRN scaling was wired in, because it was parsed nowhere.
        RopeScalingStrategy::Yarn {
            original_max_position: 16384,
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: None,
        }
    }

    /// Discriminating test for the NTK-by-parts ramp *direction* (the part
    /// of this algorithm most likely to be transcribed backwards): at the
    /// very first frequency-pair index the ramp must be fully "extrapolated"
    /// (unscaled — high frequency, short wavelength), and at the very last
    /// index it must be fully "interpolated" (`freq / factor` — low
    /// frequency, long wavelength). These two endpoints are simple closed
    /// forms that do not depend on the correction-dimension machinery being
    /// exercised, so this check is independent of the ramp/blend
    /// implementation itself.
    #[test]
    fn yarn_ramp_direction_boundary_frequencies() {
        let strategy = bonsai_8b_yarn();
        let freqs = compute_rope_frequencies(128, 1_000_000.0, &strategy, 4096)
            .expect("Yarn strategy should succeed");
        let standard = standard_freqs_ref(128, 1_000_000.0);
        let half_dim = freqs.len();

        assert!(
            (freqs[0] - standard[0]).abs() < 1e-9,
            "dim 0 (highest frequency) must stay unscaled: got {}, standard {}",
            freqs[0],
            standard[0]
        );
        let last = half_dim - 1;
        let expected_last = standard[last] / 4.0;
        assert!(
            (freqs[last] - expected_last).abs() < 1e-9,
            "last dim (lowest frequency) must be fully interpolated (/factor): \
             got {}, expected {}",
            freqs[last],
            expected_last
        );
    }

    // ── yarn_factor_one_matches_standard_frequencies ─────────────────────────

    #[test]
    fn yarn_factor_one_matches_standard_frequencies() {
        let strategy = RopeScalingStrategy::Yarn {
            original_max_position: 16384,
            factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: None,
        };
        let freqs = compute_rope_frequencies(HEAD_DIM, BASE, &strategy, 4096)
            .expect("factor=1.0 Yarn should succeed");
        let standard = standard_freqs_ref(HEAD_DIM, BASE);
        for (i, (got, exp)) in freqs.iter().zip(standard.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "freq[{i}]: factor=1.0 got {got}, standard {exp}"
            );
        }
    }

    // ── yarn_rejects_sub_unity_factor ────────────────────────────────────────

    #[test]
    fn yarn_rejects_sub_unity_factor() {
        let strategy = RopeScalingStrategy::Yarn {
            original_max_position: 16384,
            factor: 0.5,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: None,
        };
        let result = compute_rope_frequencies(HEAD_DIM, BASE, &strategy, 4096);
        assert!(
            matches!(result, Err(RopeScalingError::InvalidScaleFactor(f)) if (f - 0.5).abs() < 1e-9),
            "factor < 1.0 should be rejected, got: {result:?}"
        );
    }

    // ── yarn_frequencies_bounded_between_interpolated_and_extrapolated ───────

    #[test]
    fn yarn_frequencies_bounded_between_interpolated_and_extrapolated() {
        let strategy = bonsai_8b_yarn();
        let freqs = compute_rope_frequencies(128, 1_000_000.0, &strategy, 4096)
            .expect("Yarn strategy should succeed");
        // `standard_freqs_ref` computes in `f32` throughout, whereas
        // `yarn_inv_freq` (which `freqs` above went through) computes its
        // `f64` core (`yarn_inv_freq_f64`) and rounds to `f32` only once at
        // the end — a few ULPs more precise, not a different formula. `1e-6`
        // absolute comfortably covers that gap at these frequencies'
        // magnitude (~0.05-0.9) while still being 4-5 orders of magnitude
        // tighter than a real ramp/blend bug would produce.
        let epsilon = 1e-6_f32;
        let standard = standard_freqs_ref(128, 1_000_000.0);
        for (i, (&yarn_f, &std_f)) in freqs.iter().zip(standard.iter()).enumerate() {
            let interp = std_f / 4.0;
            let (lo, hi) = if interp < std_f {
                (interp, std_f)
            } else {
                (std_f, interp)
            };
            assert!(
                yarn_f >= lo - epsilon && yarn_f <= hi + epsilon,
                "freq[{i}] = {yarn_f} should be between interpolated {interp} \
                 and extrapolated {std_f}"
            );
        }
    }

    // ── B3: high correction dim must clamp to head_dim-1, not half_dim-1 ─────

    #[test]
    fn yarn_high_correction_dim_clamps_to_head_dim_not_half_dim() {
        // Materiality point: at these
        // parameters the RAW high correction dimension is ~37 (see the
        // assertion below), which is *larger* than `half_dim - 1` (= 31 for
        // head_dim=64) but still within `head_dim - 1` (= 63). Both ggml
        // (`ggml_rope_yarn_corr_dims`) and HF transformers
        // (`find_correction_range`) clamp `high` against `head_dim - 1`; the
        // bug this test guards against instead clamped against `half_dim -
        // 1`, which — only in configurations like this one — silently drags
        // `high` down to 31 and wrongly forces every dimension from there
        // down towards `low` closer to "fully interpolated" than the true
        // ramp allows. `top` (the last valid frequency-pair index) is the
        // sharpest witness: under the bug it is forced to ramp == 1.0
        // (freq == freq_extrap / factor, exactly); under the fix its true
        // ramp is well short of 1.0 (a partial blend).
        //
        // This scenario is unreachable for Bonsai-8B (head_dim=128) and the
        // 27B, so it is deliberately a synthetic head_dim=64 shape chosen to
        // actually exercise the clamp, not a real-model regression.
        const HEAD_DIM: usize = 64;
        const BASE: f64 = 10_000.0;
        const ORIGINAL_MAX_POSITION: usize = 262_144;
        const BETA_FAST: f64 = 32.0;
        const BETA_SLOW: f64 = 1.0;
        const FACTOR: f32 = 4.0;
        let half_dim = HEAD_DIM / 2;
        let top = half_dim - 1;

        // Confirm this scenario actually exercises the bug (self-checking,
        // so a future change to the formula that made the bug unreachable
        // here would fail loudly instead of leaving a silently-vacuous
        // test).
        let raw_high = yarn_correction_dim_f64(HEAD_DIM, BASE, ORIGINAL_MAX_POSITION, BETA_SLOW);
        assert!(
            raw_high.ceil() > (half_dim - 1) as f64,
            "test setup no longer exercises the half_dim-vs-head_dim clamp bug: \
             raw high correction dim {raw_high} <= half_dim-1 ({})",
            half_dim - 1
        );
        assert!(
            raw_high.ceil() <= (HEAD_DIM - 1) as f64,
            "test setup: raw high correction dim {raw_high} must still be within \
             head_dim-1 ({}), or even the fixed clamp would engage and this test \
             would no longer isolate the half_dim-vs-head_dim distinction",
            HEAD_DIM - 1
        );

        let strategy = RopeScalingStrategy::Yarn {
            original_max_position: ORIGINAL_MAX_POSITION,
            factor: FACTOR,
            beta_fast: BETA_FAST as f32,
            beta_slow: BETA_SLOW as f32,
            attn_factor: None,
        };
        let freqs = compute_rope_frequencies(HEAD_DIM, BASE as f32, &strategy, 4096)
            .expect("Yarn strategy should succeed");
        let standard = standard_freqs_ref(HEAD_DIM, BASE as f32);
        let fully_interpolated = standard[top] / FACTOR;

        let relative_gap = (freqs[top] - fully_interpolated).abs() / fully_interpolated;
        assert!(
            relative_gap > 0.1,
            "dim {top} should not be fully interpolated (raw high correction dim \
             is {raw_high}, well past half_dim-1={}): got freq={}, fully-interpolated \
             would be {} (relative gap {relative_gap} — this must be large if the \
             head_dim-1 clamp is in effect instead of the buggy half_dim-1 one)",
            half_dim - 1,
            freqs[top],
            fully_interpolated
        );
    }

    // ── yarn_mscale ───────────────────────────────────────────────────────────

    #[test]
    fn yarn_mscale_no_scaling_is_identity() {
        assert!((yarn_mscale(1.0, None) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn yarn_mscale_matches_standard_formula() {
        let factor = 4.0_f32;
        let expected = 0.1 * factor.ln() + 1.0;
        let got = yarn_mscale(factor, None);
        assert!(
            (got - expected).abs() < 1e-6,
            "yarn_mscale({factor}, None) = {got}, expected {expected}"
        );
        // The real Bonsai-8B mscale should be a modest boost (~1.14), not a
        // no-op and not something wildly large.
        assert!(
            (1.0..1.5).contains(&got),
            "mscale {got} outside sane range for factor={factor}"
        );
    }

    #[test]
    fn yarn_mscale_override_is_respected() {
        // Multiply convention (minor finding — see `yarn_mscale`'s
        // doc comment): the override *seeds* mscale, it does not replace the
        // computed correction outright. `2.5` is not the expected result on
        // its own; `2.5 * (0.1 * ln(4.0) + 1.0)` is.
        let got = yarn_mscale(4.0, Some(2.5));
        let expected = 2.5 * (0.1 * 4.0f32.ln() + 1.0);
        assert!(
            (got - expected).abs() < 1e-6,
            "explicit attn_factor override should seed (multiply into) the \
             standard correction, got {got}, expected {expected}"
        );
    }

    #[test]
    fn yarn_mscale_override_at_factor_one_is_the_override_itself() {
        // At factor <= 1.0 there is no `0.1 * ln(factor) + 1.0` correction to
        // multiply in (matches `yarn_mscale_no_scaling_is_identity`'s
        // `None` case at the neutral seed `1.0`), so an explicit override
        // passes through unchanged.
        let got = yarn_mscale(1.0, Some(2.5));
        assert!(
            (got - 2.5).abs() < 1e-9,
            "at factor<=1.0 the override should pass through as-is, got {got}"
        );
    }
}
