//! The implementation of the synthetic hybrid GGUF fixture generator and its
//! independent f64 reference model; `../hybrid_gguf.rs` documents what the
//! fixture is and re-exports this module's public API unchanged.
//!
//! This file holds the deterministic RNG and wires the submodules together,
//! one per concern:
//!
//! - `spec` — the fixed dimensions, [`HybridFixtureSpec`], the 24-variant
//!   matrix and the derived [`Dims`];
//! - `wide_variant` — the 1024-wide Hadamard variant's constants and spec;
//! - `plan` — the tensor plan (every GGUF tensor with its pre-quantization
//!   values, the Hadamard sign vectors, the folded-name set);
//! - `encode` — the wire encodings of a planned tensor;
//! - `writer` — writing the GGUF, parsing it back, the negative fixture;
//! - `math` — the f64 scalar primitives;
//! - `reference` — the f64 reference forward of the whole stack;
//! - `cross_checks` — the self-checks of the primitives.

#[path = "cross_checks.rs"]
mod cross_checks;
#[path = "encode.rs"]
mod encode;
#[path = "math.rs"]
mod math;
#[path = "plan.rs"]
mod plan;
#[path = "reference.rs"]
mod reference;
#[path = "spec.rs"]
mod spec;
#[path = "wide_variant.rs"]
mod wide_variant;
#[path = "writer.rs"]
mod writer;

pub use cross_checks::{
    fwht_round_trip_check, gdn_cross_check, partial_rope_check, GdnCrossCheckResult,
};
pub use math::{
    apply_partial_rope_f64, causal_conv1d_step_f64, fwht_forward_signed_f64,
    gated_rms_norm_head_f64, gdn_step_fused_f64, l2_norm_f64, rms_norm_f64, VHeadMap,
};
pub use reference::ReferenceForward;
pub use spec::{
    all_variant_specs, Dims, HybridFixtureSpec, ALL_QUANT_TYPES, BLOCK_COUNT, CONTEXT_LENGTH, FFN,
    FULL_ATTENTION_INTERVAL, HADAMARD_BLOCK_SIZE, HEAD_DIM, HIDDEN, N_HEAD, N_KV, RMS_EPS,
    ROPE_DIM, ROPE_FREQ_BASE, ROPE_SECTIONS, SSM_CONV_KERNEL, SSM_STATE_SIZE, SSM_TIME_STEP_RANK,
    T_TOKENS, VOCAB,
};
pub use wide_variant::{
    hadamard_1024_variant_spec, FFN_WIDE, HADAMARD_BLOCK_SIZE_WIDE, HEAD_DIM_WIDE, HIDDEN_WIDE,
};
pub use writer::{build, build_invalid_ungrouped_fixture, HybridFixture, InvalidUngroupedFixture};

// ═════════════════════════════════════════════════════════════════════════
// 1. Deterministic RNG
// ═════════════════════════════════════════════════════════════════════════

/// `xorshift64*`: small, dependency-free, and fully deterministic given a
/// seed — exactly what a reproducible fixture needs (`oxibonsai-model` has no
/// `rand` dependency, and a fixture must not depend on one's stream staying
/// stable across releases).
pub struct Xorshift64Star(u64);

impl Xorshift64Star {
    pub fn new(seed: u64) -> Self {
        // A zero state is a fixed point of xorshift; fold the seed with a
        // nonzero odd constant so `seed == 0` still produces a real stream.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn next_unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)`.
    pub fn next_range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_unit_f64() * (hi - lo)
    }

    /// One of `{-1.0, 0.0, 1.0}`, uniformly.
    fn next_ternary(&mut self) -> f32 {
        match self.next_u64() % 3 {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        }
    }

    /// One of `{-1.0, 1.0}`, uniformly (for the 1-bit format, which has no
    /// zero code).
    fn next_binary(&mut self) -> f32 {
        if self.next_u64().is_multiple_of(2) {
            -1.0
        } else {
            1.0
        }
    }

    /// A small-magnitude value for a plain (unquantized) `F32` tensor.
    fn next_continuous(&mut self) -> f32 {
        self.next_range_f64(-2.0, 2.0) as f32
    }

    fn next_usize_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
