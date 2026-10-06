//! SwiGLU activation function.
//!
//! `SwiGLU(x) = SiLU(gate(x)) * up(x)`
//! where `SiLU(x) = x * sigmoid(x)`.
//!
//! Used in the MLP (feed-forward) blocks of Qwen3.

use crate::error::ModelResult;

/// Apply SiLU (Swish) activation: `silu(x) = x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SwiGLU: `swiglu(gate, up) = silu(gate) * up`, propagating a real
/// length-contract violation (K-02/M-01) instead of asserting.
///
/// - `gate`: Output of gate projection (length n).
/// - `up`: Output of up projection (length n).
/// - `output`: Result buffer (length n).
///
/// Delegates to the SIMD-accelerated implementation in `oxibonsai_kernels`.
///
/// This is the *only* entry point: the production call sites
/// (`block/types/forward.rs`, `forward_sw.rs`, `forward_stats.rs`) all
/// propagate this with `?` from a `ModelResult`-returning function, so a
/// length-contract violation aborts that forward pass instead of silently
/// computing a plausible-looking wrong logit vector from a stale
/// `swiglu_out` buffer (K-02/M-01). There is intentionally no infallible
/// wrapper around this — swallowing the error was exactly the defect.
///
/// # Errors
///
/// Returns [`crate::error::ModelError::Kernel`] if `up.len() != gate.len()`
/// or `output.len() < gate.len()`.
pub fn try_swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) -> ModelResult<()> {
    oxibonsai_kernels::swiglu_simd(gate, up, output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_at_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn silu_positive() {
        // silu(1.0) = 1.0 / (1.0 + exp(-1.0)) ≈ 0.7311
        let result = silu(1.0);
        assert!((result - 0.7311).abs() < 0.001);
    }

    #[test]
    fn swiglu_basic() {
        let gate = vec![1.0, 0.0, -1.0];
        let up = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];

        try_swiglu(&gate, &up, &mut output).expect("matching-length buffers must succeed");

        assert!((output[0] - silu(1.0) * 2.0).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5); // silu(0)*3 = 0
        assert!((output[2] - silu(-1.0) * 4.0).abs() < 1e-5);
    }

    // ── K-02 / M-01: real length-contract errors, not a debug_assert ────

    #[test]
    fn try_swiglu_rejects_mismatched_up_length() {
        let gate = vec![1.0, 2.0, 3.0];
        let up = vec![1.0, 2.0]; // too short
        let mut output = vec![0.0; 3];

        let result = try_swiglu(&gate, &up, &mut output);
        assert!(
            result.is_err(),
            "try_swiglu must reject a gate/up length mismatch instead of \
             reading out of bounds in release (K-02/M-01)"
        );
    }

    #[test]
    fn try_swiglu_rejects_short_output_buffer() {
        let gate = vec![1.0, 2.0, 3.0];
        let up = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 1]; // too short

        let result = try_swiglu(&gate, &up, &mut output);
        assert!(
            result.is_err(),
            "try_swiglu must reject a too-short output buffer instead of \
             writing out of bounds in release (K-02/M-01)"
        );
    }

    // Note: an earlier revision kept an infallible `swiglu()`
    // wrapper around `try_swiglu` for source compatibility with call sites
    // that predated the fallible kernel contract, plus a test
    // (`swiglu_leaves_output_untouched_on_length_mismatch`) asserting that it
    // logged-and-ignored a length-contract violation. Both are gone: the
    // three production call sites now propagate the error with `?` (see the
    // module doc on `try_swiglu`), so there is no longer an infallible
    // caller whose contract that test could describe. Removing it is not a
    // weakened test — `try_swiglu_rejects_mismatched_up_length` and
    // `try_swiglu_rejects_short_output_buffer` above already cover the
    // length-contract-violation behavior of the one remaining entry point.
}
