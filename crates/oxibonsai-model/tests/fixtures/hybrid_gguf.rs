//! Synthetic hybrid (`qwen35`) GGUF fixture generator + independent f64
//! scalar reference model (`bonsai2-design.md` §7.1/§7.2/§8.2).
//!
//! This is the workhorse that lets every other Bonsai 2 package be tested
//! without 7 GB of real weights. It has two halves:
//!
//! 1. [`build`] emits a COMPLETE, valid, tiny hybrid GGUF: a real
//!    `general.architecture = "qwen35"` file with the full `qwen35.*` /
//!    `ssm.*` key set, a real `prism.hadamard.*` contract, and every tensor
//!    the hybrid loader binds, in any of six quantization formats (`F32`,
//!    `PQ2_0`, `PTQ1_0`, `Q2_0_g64`, `TQ2_0_g128`, `Q1_0_g128`) crossed with
//!    Hadamard on/off and V-head grouping on/off, at either the narrow
//!    ([`HIDDEN`]) or the real 27B's own 1024-wide ([`HIDDEN_WIDE`])
//!    Hadamard block geometry (see [`HybridFixtureSpec`]).
//! 2. An independent, from-scratch **f64 scalar reference model** of the
//!    same graph (embedding + inverse Hadamard, per-layer full-attention and
//!    Gated-DeltaNet linear-attention, final norm + LM head), built
//!    alongside the generator over the exact same pre-quantization weight
//!    values. It shares **no code** with `oxibonsai-kernels`/
//!    `oxibonsai-model`'s production kernels on purpose: it is deliberately
//!    slow and obviously correct, so it is ground truth that would catch a
//!    wrong V-head permutation or a wrong RoPE pairing — both of which pass
//!    every per-kernel norm check.
//!
//! `pub` dev-only API: exposed so a shared test-support crate can reuse this
//! builder instead of writing another one-off GGUF fixture function. Every
//! generated file lands in [`std::env::temp_dir`] and is removed when the
//! returned [`HybridFixture`] is dropped; nothing here ever hardcodes an
//! absolute path.
//!
//! ## What "matches the f64 reference model" means here
//!
//! `bonsai2-design.md` §8.2's acceptance line ("all variants build, load
//! through the real hybrid loader and match the embedded f64 scalar
//! model") is exercised end to end:
//! `oxibonsai_model::hybrid::model::HybridModel` (the real hybrid loader)
//! lives in this same crate, and `hybrid_forward_parity_tests.rs` loads
//! every variant this file's [`build`]/[`all_variant_specs`]/
//! [`hadamard_1024_variant_spec`] produce through it, teacher-forced
//! against this file's own f64 reference — both the final logits and, via
//! [`ReferenceForward::per_layer_hidden`], every intermediate layer's
//! residual stream. `hybrid_fixture_tests.rs` (this file's own test
//! binary) additionally cross-checks the f64 primitives directly against
//! their `oxibonsai_kernels` counterparts (`fwht_forward_signed`,
//! `gdn_step_with`, `causal_conv1d_k4_decode`, `l2_norm_simd`,
//! `rms_norm_gated_simd`, `rope_partial_splithalf_simd`) on deterministic
//! random inputs, and checks:
//!
//! - the generator's byte layout round-trips exactly through the real,
//!   already-merged block codecs (`BlockPQ2_0`/`BlockPTQ1_0`/
//!   `BlockQ2_0G64`/`BlockTQ2_0_g128`/`BlockQ1_0G128::dequant`) and the
//!   real GGUF parser/config types (`GgufFile::parse`,
//!   `HybridConfig::from_metadata`, `HadamardConfig::from_metadata`);
//! - the f64 reference model is internally self-consistent: independently
//!   re-derived forms of the Fused-Weighted-Hadamard-Transform (involution
//!   identity), the Gated-DeltaNet recurrence (fused single-pass vs. a
//!   naive 3-pass form vs. an O(T²) closed-form expansion) and partial RoPE
//!   (untouched-tail invariant) all agree with each other;
//! - the generator and the reference model are deterministic given a seed,
//!   and every temp file is cleaned up.
// Every test binary that includes this fixture uses a subset of its public
// API, so both the unused items and the unused re-exports below are expected.
#![allow(dead_code, unused_imports)]

// The implementation lives in `hybrid_gguf/`, split by concern (see the
// doc comment of `hybrid_gguf/mod.rs`); this file keeps the fixture's public
// path — each consumer loads it with `#[path = "fixtures/hybrid_gguf.rs"]` —
// and re-exports the whole public API unchanged.
#[path = "hybrid_gguf/mod.rs"]
mod imp;

pub use imp::*;
