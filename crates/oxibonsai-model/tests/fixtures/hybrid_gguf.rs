//! Synthetic hybrid (`qwen35`) GGUF fixture generator + independent f64
//! scalar reference model — B2-16 (`bonsai2-design.md` §7.1/§7.2/§8.2).
//!
//! This is the workhorse that lets every other Bonsai 2 package be tested
//! without 7 GB of real weights. It has two halves:
//!
//! 1. [`build`] emits a COMPLETE, valid, tiny hybrid GGUF: a real
//!    `general.architecture = "qwen35"` file with the full `qwen35.*` /
//!    `ssm.*` key set, a real `prism.hadamard.*` contract, and every tensor
//!    the hybrid loader binds, in any of six quantization formats (`F32`,
//!    `PQ2_0`, `PTQ1_0`, `Q2_0_g64`, `TQ2_0_g128`, `Q1_0_g128`) crossed with
//!    Hadamard on/off and V-head grouping on/off.
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
//! `pub` dev-only API: exposed so a shared test-support crate (T-07) can
//! reuse this builder instead of writing an 18th one-off GGUF fixture
//! function. Every generated file lands in [`std::env::temp_dir`] and is
//! removed when the returned [`HybridFixture`] is dropped; nothing here
//! ever hardcodes an absolute path.
//!
//! ## What "matches the f64 reference model" means in this package
//!
//! `bonsai2-design.md` §8.2's acceptance line for this package
//! ("all 16 variants build, load through the real hybrid loader and match
//! the embedded f64 scalar model") describes the **release-gate** shape of
//! this fixture, which is exercised once `oxibonsai_model::hybrid` exists
//! (owned by the sibling B2-09/B2-10/B2-11 packages) and wired up by B2-18
//! (whose spec explicitly depends on B2-16 **and** B2-11). That model does
//! not exist in this package's dependency graph. What this package
//! delivers and *can* verify on its own:
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
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;

use oxibonsai_core::bf16::{bf16_to_f32, f32_to_bf16};
use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_core::error::{BonsaiError, BonsaiResult};
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::hadamard_config::HadamardConfig;
use oxibonsai_core::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, BlockTQ2_0_g128};

// ═════════════════════════════════════════════════════════════════════════
// 1. Deterministic RNG
// ═════════════════════════════════════════════════════════════════════════

/// `xorshift64*`: small, dependency-free, and fully deterministic given a
/// seed — exactly what a reproducible fixture needs (no `rand` crate
/// dependency, which `oxibonsai-model`'s `Cargo.toml` is not owned by this
/// package to add).
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
    fn next_range_f64(&mut self, lo: f64, hi: f64) -> f64 {
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

// ═════════════════════════════════════════════════════════════════════════
// 2. Small f64 scalar math primitives (independent of every production
//    kernel — see the module doc's "no shared code" note)
// ═════════════════════════════════════════════════════════════════════════

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_f64(x: f64) -> f64 {
    x * sigmoid_f64(x)
}

/// `x > 20 ? x : ln(1 + exp(x))` — the exact ggml softplus cutoff (design
/// §2.3), verified against the already-merged `softplus_scalar_elem`
/// (`oxibonsai-kernels/src/norms.rs`) during this package's design pass.
fn softplus_f64(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Standard RMSNorm: `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
///
/// This is the "ordinary" form (mean-based, `eps` under the radical), used
/// for every plain norm layer (`attn_norm`, `post_attention_norm`,
/// `output_norm`, `attn_q_norm`, `attn_k_norm`) and — per-head — for the
/// Gated-DeltaNet output norm. It is a *different* formula from
/// [`l2_norm_heads_f64`] below; conflating the two is a real, documented
/// trap in this codebase's history (see `oxibonsai-kernels/src/norms.rs`'s
/// `l2_norm_simd` doc comment).
fn rms_norm_f64(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| v * inv_rms * w)
        .collect()
}

/// `L2Norm`: `out[i] = x[i] / max(sqrt(sum(x^2)), eps)` — no mean, no
/// weight. `eps` floors the denominator; it is not added under the
/// radical. Applied per-head over the joint Q/K region in Gated DeltaNet.
fn l2_norm_f64(x: &[f64], eps: f64) -> Vec<f64> {
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    let denom = sum_sq.sqrt().max(eps);
    x.iter().map(|&v| v / denom).collect()
}

/// Gated RMSNorm for one V-head: `out[j] = weight[j] * o[j] * inv_rms *
/// silu(z[j])`, `inv_rms` from the *mean*-based RMSNorm formula over this
/// head's slice only (mirrors `rms_norm_gated_scalar` in
/// `oxibonsai-kernels/src/norms.rs`, read for verification, not called).
fn gated_rms_norm_head_f64(o: &[f64], z: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    let n = o.len() as f64;
    let sum_sq: f64 = o.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n + eps).sqrt();
    (0..o.len())
        .map(|j| weight[j] * o[j] * inv_rms * silu_f64(z[j]))
        .collect()
}

/// Partial NeoX RoPE over the first `n_rot` of `head_dim` dimensions.
/// Pairs `(ic, ic + n_rot/2)` for `ic in 0..n_rot/2`; dims `n_rot..head_dim`
/// are copied unchanged (design §2.5's proof that text-only IMROPE
/// degenerates to standard partial RoPE).
fn apply_partial_rope_f64(x: &mut [f64], pos: usize, n_rot: usize, freq_base: f64) {
    let half = n_rot / 2;
    for ic in 0..half {
        let theta = pos as f64 * freq_base.powf(-2.0 * ic as f64 / n_rot as f64);
        let (sin_v, cos_v) = theta.sin_cos();
        let x0 = x[ic];
        let x1 = x[ic + half];
        x[ic] = x0 * cos_v - x1 * sin_v;
        x[ic + half] = x0 * sin_v + x1 * cos_v;
    }
    // dims n_rot..head_dim are left untouched by construction (never
    // written above) — asserted explicitly by
    // `partial_rope_leaves_the_tail_untouched` in `hybrid_fixture_tests.rs`.
}

// ═════════════════════════════════════════════════════════════════════════
// 3. Blockwise FWHT with fused sign flip (design §2.2), f64
// ═════════════════════════════════════════════════════════════════════════

/// Unsigned, unnormalized in-place Walsh-Hadamard butterfly network.
/// `buf.len()` must be a power of two. Applying this twice multiplies the
/// input by `buf.len()` (the involution identity
/// [`fwht_round_trip_is_involution`] in `hybrid_fixture_tests.rs` checks).
fn fwht_butterfly_f64(buf: &mut [f64]) {
    let n = buf.len();
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in 0..len {
                let u = buf[i + j];
                let v = buf[i + len + j];
                buf[i + j] = u + v;
                buf[i + len + j] = u - v;
            }
            i += 2 * len;
        }
        len <<= 1;
    }
}

/// Forward fold: `x <- FWHT_block(x ⊙ signs) * (1/sqrt(block))`, applied
/// independently to each `block`-wide slice. `signs` and `x` are the same
/// length (a whole number of `block`-wide chunks).
fn fwht_forward_signed_f64(x: &[f64], signs: &[f64], block: usize) -> Vec<f64> {
    let inv_sqrt = 1.0 / (block as f64).sqrt();
    let mut out = x.to_vec();
    for (chunk_idx, chunk) in out.chunks_mut(block).enumerate() {
        for (j, v) in chunk.iter_mut().enumerate() {
            *v *= signs[chunk_idx * block + j] * inv_sqrt;
        }
        fwht_butterfly_f64(chunk);
    }
    out
}

/// Inverse fold (used once, after the embedding lookup):
/// `x <- FWHT_block(x) * (1/sqrt(block))`, then `x ⊙ signs`.
fn fwht_inverse_signed_f64(x: &[f64], signs: &[f64], block: usize) -> Vec<f64> {
    let inv_sqrt = 1.0 / (block as f64).sqrt();
    let mut out = x.to_vec();
    for (chunk_idx, chunk) in out.chunks_mut(block).enumerate() {
        fwht_butterfly_f64(chunk);
        for (j, v) in chunk.iter_mut().enumerate() {
            *v = *v * inv_sqrt * signs[chunk_idx * block + j];
        }
    }
    out
}

// ═════════════════════════════════════════════════════════════════════════
// 4. V-head grouping (design §3.3)
// ═════════════════════════════════════════════════════════════════════════

/// Maps a GROUPED V-head index (the order `ssm_out`'s folded columns and
/// this fixture's recurrent state use) to the TILED index the GGUF's raw
/// V-indexed rows use (`ssm_alpha`/`ssm_beta`/`ssm_a`/`ssm_dt.bias` outputs
/// and the V-slice of `attn_qkv`/`attn_gate`).
#[derive(Debug, Clone, Copy)]
pub struct VHeadMap {
    pub n_k_heads: usize,
    pub v_per_k: usize,
}

impl VHeadMap {
    pub fn new(n_k_heads: usize, n_v_heads: usize) -> Self {
        let v_per_k = n_v_heads / n_k_heads.max(1);
        Self { n_k_heads, v_per_k }
    }

    /// GROUPED index `m` -> TILED index it reads from:
    /// `(m % rep) * nk + (m / rep)`.
    #[inline]
    pub fn tiled(&self, grouped: usize) -> usize {
        (grouped % self.v_per_k) * self.n_k_heads + (grouped / self.v_per_k)
    }

    /// GROUPED index `m` -> the K-head it reads (`m / rep`).
    #[inline]
    pub fn k_head(&self, grouped: usize) -> usize {
        grouped / self.v_per_k
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        self.v_per_k == 1
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 5. Gated DeltaNet (design §2.3), f64 — fused, naive-3-pass, and an O(T²)
//    closed-form cross-check.
// ═════════════════════════════════════════════════════════════════════════

/// One decode step, fused single-pass form (the form used by the actual
/// forward driver below). `state` is `[head_v_dim][head_k_dim]`
/// (row-major, row `j` = the mathematical state's column `j`, matching
/// design §2.3's chosen memory order). Returns `(out, decay, delta)` so the
/// caller can build the O(T²) cross-check.
#[allow(clippy::too_many_arguments)]
fn gdn_step_fused_f64(
    state: &mut [f64],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    alpha_raw: f64,
    beta_raw: f64,
    dt_bias: f64,
    a_neg: f64,
    head_k_dim: usize,
    head_v_dim: usize,
) -> (Vec<f64>, f64, Vec<f64>) {
    let beta = sigmoid_f64(beta_raw);
    let sp = softplus_f64(alpha_raw + dt_bias);
    let g = a_neg * sp;
    let decay = g.exp();
    let scale = 1.0 / (head_v_dim as f64).sqrt();

    let mut out = vec![0.0f64; head_v_dim];
    let mut delta = vec![0.0f64; head_v_dim];
    let mut tmp = vec![0.0f64; head_k_dim];
    for j in 0..head_v_dim {
        let row = &mut state[j * head_k_dim..(j + 1) * head_k_dim];
        let mut sum = 0.0f64;
        for i in 0..head_k_dim {
            let s = row[i] * decay;
            tmp[i] = s;
            sum += s * k[i];
        }
        let d = (v[j] - sum) * beta;
        delta[j] = d;
        for i in 0..head_k_dim {
            row[i] = tmp[i] + k[i] * d;
        }
        out[j] = row.iter().zip(q).map(|(&s, &qi)| s * qi).sum::<f64>() * scale;
    }
    (out, decay, delta)
}

/// The same step, as an explicit non-fused 3-pass form (§7.2 "vs a naive
/// non-fused 3-pass reference"): `S *= decay` as its own pass, then a pass
/// computing `delta`, then a pass updating `S` and the output.
#[allow(clippy::too_many_arguments)]
fn gdn_step_naive_3pass_f64(
    state: &mut [f64],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    alpha_raw: f64,
    beta_raw: f64,
    dt_bias: f64,
    a_neg: f64,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Vec<f64> {
    let beta = sigmoid_f64(beta_raw);
    let sp = softplus_f64(alpha_raw + dt_bias);
    let decay = (a_neg * sp).exp();
    let scale = 1.0 / (head_v_dim as f64).sqrt();

    // Pass 1: S *= decay (whole slab).
    for s in state.iter_mut() {
        *s *= decay;
    }
    // Pass 2: delta[j] = (v[j] - dot(S_row_j, k)) * beta.
    let mut delta = vec![0.0f64; head_v_dim];
    for j in 0..head_v_dim {
        let row = &state[j * head_k_dim..(j + 1) * head_k_dim];
        let dot: f64 = row.iter().zip(k).map(|(&s, &ki)| s * ki).sum();
        delta[j] = (v[j] - dot) * beta;
    }
    // Pass 3: S_row_j += k * delta[j] (axpy), then out[j] = dot(S_row_j, q).
    let mut out = vec![0.0f64; head_v_dim];
    for j in 0..head_v_dim {
        let row = &mut state[j * head_k_dim..(j + 1) * head_k_dim];
        for i in 0..head_k_dim {
            row[i] += k[i] * delta[j];
        }
        out[j] = row.iter().zip(q).map(|(&s, &qi)| s * qi).sum::<f64>() * scale;
    }
    out
}

/// O(T²) closed-form cross-check (§7.2's third GDN row): given the
/// per-step `decay`/`delta`/`k`/`q` history already produced by a
/// sequential run, reconstruct every step's output directly from
/// `o_t[j] = scale * sum_{s<=t} (prod_{u=s+1..t} decay_u) * (k_s . q_t) *
/// delta_s[j]` and compare to the sequential recurrence's own output. This
/// is the closed form of unrolling `S_t = S_{t-1}*decay_t + k_t (x)
/// delta_t`, `S_0 = 0` — an algebraic identity independent of how the
/// per-step `decay`/`delta` values themselves were computed, so it
/// specifically guards the *state-update wiring* (outer-product axis
/// order, decay placement, summation bounds), not the per-step formula.
fn gdn_quadratic_reconstruction_f64(
    decay_hist: &[f64],
    delta_hist: &[Vec<f64>],
    k_hist: &[Vec<f64>],
    q_hist: &[Vec<f64>],
    head_v_dim: usize,
) -> Vec<Vec<f64>> {
    let t_len = decay_hist.len();
    let scale = 1.0 / (head_v_dim as f64).sqrt();
    let mut out = vec![vec![0.0f64; head_v_dim]; t_len];
    for t in 0..t_len {
        for s in 0..=t {
            let mut prod = 1.0f64;
            for decay in decay_hist.iter().take(t + 1).skip(s + 1) {
                prod *= decay;
            }
            let kq: f64 = k_hist[s]
                .iter()
                .zip(&q_hist[t])
                .map(|(&ki, &qi)| ki * qi)
                .sum();
            let coeff = prod * kq * scale;
            for j in 0..head_v_dim {
                out[t][j] += coeff * delta_hist[s][j];
            }
        }
    }
    out
}

// ═════════════════════════════════════════════════════════════════════════
// 6. Causal depthwise conv1d (design §2.4), f64
// ═════════════════════════════════════════════════════════════════════════

/// One decode step of the causal depthwise conv1d over `channels` streams,
/// kernel width `kc`. `state` is `[channels][kc-1]`, oldest-first, updated
/// in place. `w` is `[channels][kc]` (matches GGUF `ssm_conv1d.weight`'s
/// `ne=[kc, channels]` row-major-per-channel layout).
fn causal_conv1d_step_f64(
    state: &mut [Vec<f64>],
    x: &[f64],
    w: &[Vec<f64>],
    channels: usize,
    kc: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; channels];
    for c in 0..channels {
        let mut acc = 0.0f64;
        for i in 0..kc - 1 {
            acc += state[c][i] * w[c][i];
        }
        acc += x[c] * w[c][kc - 1];
        y[c] = acc;
        state[c].rotate_left(1);
        let last = state[c].len() - 1;
        state[c][last] = x[c];
    }
    y
}

// ═════════════════════════════════════════════════════════════════════════
// 7. Fixture specification and derived dimensions
// ═════════════════════════════════════════════════════════════════════════

/// Number of decoder layers in every generated fixture. `full_attention_interval
/// == 4` selects layers 3 and 7 as full-attention (16 % / 84 % split, the
/// same ratio class as the real 27B) and the rest as Gated-DeltaNet.
pub const BLOCK_COUNT: usize = 8;
pub const FULL_ATTENTION_INTERVAL: usize = 4;
pub const HIDDEN: usize = 256;
pub const FFN: usize = 384;
pub const N_HEAD: usize = 4;
pub const N_KV: usize = 2;
pub const HEAD_DIM: usize = 32;
pub const ROPE_DIM: usize = 16;
pub const ROPE_SECTIONS: [i32; 4] = [3, 3, 2, 0];
pub const ROPE_FREQ_BASE: f64 = 10_000.0;
pub const SSM_CONV_KERNEL: usize = 4;
pub const SSM_STATE_SIZE: usize = 32; // head_k_dim == head_v_dim
pub const SSM_TIME_STEP_RANK: usize = 4; // n_v_heads
pub const VOCAB: usize = 64;
pub const CONTEXT_LENGTH: usize = 512;
pub const HADAMARD_BLOCK_SIZE: usize = 128;
pub const RMS_EPS: f64 = 1e-6;
/// Number of tokens the embedded reference forward runs over — long enough
/// to exceed the conv1d kernel width and exercise multi-position causal
/// attention and multi-step GDN recurrence.
pub const T_TOKENS: usize = 6;

/// The six quantization formats a hybrid loader must bind (design §1.1 +
/// the B2-16 work order's format list).
pub const ALL_QUANT_TYPES: [TensorType; 6] = [
    TensorType::F32,
    TensorType::PQ2_0,
    TensorType::PTQ1_0,
    TensorType::Q2_0G64,
    TensorType::TQ2_0_g128,
    TensorType::Q1_0G128,
];

/// One fixture variant: a quantization format crossed with Hadamard
/// on/off and V-head grouping on/off (design §7.1's `HybridFixtureSpec`).
#[derive(Debug, Clone, Copy)]
pub struct HybridFixtureSpec {
    pub quant: TensorType,
    pub hadamard: bool,
    pub gdn_v_grouped: bool,
    pub seed: u64,
}

/// Every canonical (quant, hadamard, grouped) combination — the full cross
/// product (`6 x 2 x 2 = 24`), a strict superset of both design §7.1's
/// original 16-variant matrix (4 quant formats it named explicitly) and
/// this package's work order's 6-format list.
pub fn all_variant_specs(base_seed: u64) -> Vec<HybridFixtureSpec> {
    let mut specs = Vec::with_capacity(ALL_QUANT_TYPES.len() * 4);
    for (qi, &quant) in ALL_QUANT_TYPES.iter().enumerate() {
        for (hi, &hadamard) in [false, true].iter().enumerate() {
            for (gi, &gdn_v_grouped) in [false, true].iter().enumerate() {
                let mix = (qi as u64) * 100 + (hi as u64) * 10 + (gi as u64);
                let seed = base_seed ^ mix.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                specs.push(HybridFixtureSpec {
                    quant,
                    hadamard,
                    gdn_v_grouped,
                    seed,
                });
            }
        }
    }
    specs
}

/// Every dimension derivable from a [`HybridFixtureSpec`]. `gdn_v_grouped`
/// selects `n_k_heads`: `2` (so `v_per_k == 2`, the real interesting case)
/// when grouped, or `4` (`v_per_k == 1`, the trivial/identity case) when
/// not — a V-head map with `rep > 1` and `gdn_v_grouped == false` is
/// exactly the configuration design §3.3 says a hybrid loader must
/// *refuse*, so the ordinary parity matrix never constructs it (see
/// [`build_invalid_ungrouped_fixture`] for the dedicated negative fixture).
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub inner_size: usize,
    pub conv_dim: usize,
}

impl Dims {
    pub fn from_spec(spec: &HybridFixtureSpec) -> Self {
        let n_v_heads = SSM_TIME_STEP_RANK;
        let n_k_heads = if spec.gdn_v_grouped { 2 } else { n_v_heads };
        let head_k_dim = SSM_STATE_SIZE;
        let head_v_dim = SSM_STATE_SIZE;
        let inner_size = n_v_heads * head_v_dim;
        let conv_dim = 2 * head_k_dim * n_k_heads + inner_size;
        Self {
            n_k_heads,
            n_v_heads,
            head_k_dim,
            head_v_dim,
            inner_size,
            conv_dim,
        }
    }

    pub fn v_per_k(&self) -> usize {
        self.n_v_heads / self.n_k_heads.max(1)
    }

    pub fn is_full_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL)
    }

    pub fn attn_output_input_width(&self) -> usize {
        N_HEAD * HEAD_DIM
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 8. Planned tensors: one record per GGUF tensor, carrying both its wire
//    encoding and the exact pre-quantization f32 values the reference
//    model widens to f64.
// ═════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorKind {
    /// Encoded with the variant's chosen `quant` format.
    Quant,
    /// Always `BF16`, never folded (`ssm_alpha`/`ssm_beta`).
    Bf16,
    /// Always plain `F32` regardless of the variant (norms, conv1d, `ssm_a`,
    /// `ssm_dt.bias`).
    PlainF32,
}

struct PlannedTensor {
    shape: Vec<u64>,
    kind: TensorKind,
    /// Row-major, length `product(shape)`. For [`TensorKind::Bf16`], these
    /// are already the bf16-round-tripped values (so the reference model
    /// widens exactly what the file stores).
    values: Vec<f32>,
}

/// Every named tensor a `qwen35` hybrid checkpoint binds, plus enough
/// metadata to drive both the GGUF writer and the f64 reference model.
struct FixturePlan {
    dims: Dims,
    tensors: BTreeMap<String, PlannedTensor>,
    /// `width -> ±1.0 vector`, only populated when `spec.hadamard`.
    signs: BTreeMap<usize, Vec<f64>>,
    folded_names: Vec<String>,
}

fn tname(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

/// Quantizable tensor kinds' foldable-suffix set (design §1.7 rule 7 /
/// `hadamard_config.rs::FOLDABLE_BLOCK_SUFFIXES`).
const FOLDABLE_SUFFIXES: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_gate",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ssm_out",
];

fn build_plan(spec: &HybridFixtureSpec) -> FixturePlan {
    let dims = Dims::from_spec(spec);
    let mut rng = Xorshift64Star::new(spec.seed);
    let mut tensors = BTreeMap::new();

    let push_quant = |tensors: &mut BTreeMap<String, PlannedTensor>,
                      rng: &mut Xorshift64Star,
                      name: &str,
                      ne0: usize,
                      ne1: usize| {
        let n = ne0 * ne1;
        let values: Vec<f32> = (0..n)
            .map(|_| match spec.quant {
                TensorType::F32 => rng.next_continuous(),
                TensorType::Q1_0G128 => rng.next_binary(),
                _ => rng.next_ternary(),
            })
            .collect();
        tensors.insert(
            name.to_string(),
            PlannedTensor {
                shape: vec![ne0 as u64, ne1 as u64],
                kind: TensorKind::Quant,
                values,
            },
        );
    };
    let push_plain_f32 = |tensors: &mut BTreeMap<String, PlannedTensor>,
                          rng: &mut Xorshift64Star,
                          name: &str,
                          shape: Vec<u64>,
                          range: (f64, f64)| {
        let n: u64 = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|_| rng.next_range_f64(range.0, range.1) as f32)
            .collect();
        tensors.insert(
            name.to_string(),
            PlannedTensor {
                shape,
                kind: TensorKind::PlainF32,
                values,
            },
        );
    };
    let push_bf16 = |tensors: &mut BTreeMap<String, PlannedTensor>,
                     rng: &mut Xorshift64Star,
                     name: &str,
                     ne0: usize,
                     ne1: usize| {
        let n = ne0 * ne1;
        let values: Vec<f32> = (0..n)
            .map(|_| {
                let raw = rng.next_range_f64(-1.0, 1.0) as f32;
                bf16_to_f32(f32_to_bf16(raw))
            })
            .collect();
        tensors.insert(
            name.to_string(),
            PlannedTensor {
                shape: vec![ne0 as u64, ne1 as u64],
                kind: TensorKind::Bf16,
                values,
            },
        );
    };

    // ── Globals ────────────────────────────────────────────────────────
    push_quant(&mut tensors, &mut rng, "token_embd.weight", HIDDEN, VOCAB);
    push_plain_f32(
        &mut tensors,
        &mut rng,
        "output_norm.weight",
        vec![HIDDEN as u64],
        (0.5, 1.5),
    );
    push_quant(&mut tensors, &mut rng, "output.weight", HIDDEN, VOCAB);

    for layer in 0..BLOCK_COUNT {
        push_plain_f32(
            &mut tensors,
            &mut rng,
            &tname(layer, "attn_norm.weight"),
            vec![HIDDEN as u64],
            (0.5, 1.5),
        );
        push_plain_f32(
            &mut tensors,
            &mut rng,
            &tname(layer, "post_attention_norm.weight"),
            vec![HIDDEN as u64],
            (0.5, 1.5),
        );
        push_quant(
            &mut tensors,
            &mut rng,
            &tname(layer, "ffn_gate.weight"),
            HIDDEN,
            FFN,
        );
        push_quant(
            &mut tensors,
            &mut rng,
            &tname(layer, "ffn_up.weight"),
            HIDDEN,
            FFN,
        );
        push_quant(
            &mut tensors,
            &mut rng,
            &tname(layer, "ffn_down.weight"),
            FFN,
            HIDDEN,
        );

        if dims.is_full_attention(layer) {
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_q.weight"),
                HIDDEN,
                N_HEAD * HEAD_DIM * 2,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_k.weight"),
                HIDDEN,
                N_KV * HEAD_DIM,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_v.weight"),
                HIDDEN,
                N_KV * HEAD_DIM,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_output.weight"),
                dims.attn_output_input_width(),
                HIDDEN,
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_q_norm.weight"),
                vec![HEAD_DIM as u64],
                (0.5, 1.5),
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_k_norm.weight"),
                vec![HEAD_DIM as u64],
                (0.5, 1.5),
            );
        } else {
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_qkv.weight"),
                HIDDEN,
                dims.conv_dim,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_gate.weight"),
                HIDDEN,
                dims.inner_size,
            );
            push_bf16(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_alpha.weight"),
                HIDDEN,
                dims.n_v_heads,
            );
            push_bf16(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_beta.weight"),
                HIDDEN,
                dims.n_v_heads,
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_conv1d.weight"),
                vec![SSM_CONV_KERNEL as u64, dims.conv_dim as u64],
                (-0.5, 0.5),
            );
            // ssm_a: negative (A = -exp(A_log)), consumed as-is.
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_a"),
                vec![dims.n_v_heads as u64],
                (-1.0, -0.1),
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_dt.bias"),
                vec![dims.n_v_heads as u64],
                (-0.5, 0.5),
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_norm.weight"),
                vec![dims.head_v_dim as u64],
                (0.5, 1.5),
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_out.weight"),
                dims.inner_size,
                HIDDEN,
            );
        }
    }

    // ── Hadamard sign vectors + folded-name set ─────────────────────────
    let mut signs = BTreeMap::new();
    let mut folded_names = Vec::new();
    if spec.hadamard {
        for &width in &[HIDDEN, dims.attn_output_input_width(), FFN] {
            signs.entry(width).or_insert_with(|| {
                (0..width)
                    .map(|_| {
                        if rng.next_u64().is_multiple_of(2) {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                    .collect::<Vec<f64>>()
            });
        }
        // `ssm_out`'s input width is `inner_size`, which coincides with
        // `attn_output_input_width()` by this fixture's chosen dimensions
        // (mirrors the real 27B, where both are 6144) — already covered
        // by the loop above, asserted in `hybrid_fixture_tests.rs`.
        folded_names.push("output.weight".to_string());
        for layer in 0..BLOCK_COUNT {
            let suffixes: &[&str] = if dims.is_full_attention(layer) {
                &["attn_q", "attn_k", "attn_v", "attn_output"]
            } else {
                &["attn_qkv", "attn_gate", "ssm_out"]
            };
            for suffix in suffixes {
                debug_assert!(FOLDABLE_SUFFIXES.contains(suffix));
                folded_names.push(tname(layer, &format!("{suffix}.weight")));
            }
            folded_names.push(tname(layer, "ffn_gate.weight"));
            folded_names.push(tname(layer, "ffn_up.weight"));
            folded_names.push(tname(layer, "ffn_down.weight"));
        }
    }

    FixturePlan {
        dims,
        tensors,
        signs,
        folded_names,
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 9. GGUF byte encoding
// ═════════════════════════════════════════════════════════════════════════

/// Reinterpret a `#[repr(C)]` `Copy` block slice as raw little-endian bytes
/// (every block type here packs `u8`/`f16` fields with no implicit padding,
/// verified by each type's own `size_of` const-assert upstream).
///
/// # Safety
/// `T` must be `#[repr(C)]`, `Copy`, and free of padding bytes — true for
/// every block type passed to this function (`BlockPQ2_0`, `BlockPTQ1_0`,
/// `BlockQ2_0G64`, `BlockTQ2_0_g128`), each of which carries its own
/// `size_of::<Self>() == BLOCK_*_BYTES` const-assert in `oxibonsai-core`.
fn blocks_to_bytes<T: Copy>(blocks: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(blocks);
    // SAFETY: see the function doc; `blocks` outlives the `slice::from_raw_parts`
    // call and the resulting slice is copied into an owned `Vec` before
    // `blocks` could be dropped or mutated.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, byte_len) }.to_vec()
}

/// Hand-pack a `Q1_0_g128` tensor: `d: f16` first, then 128 sign bits per
/// block (`bit == 1 -> +d`). No library `quantize()` exists for this
/// 1-bit-only format (design §1.4 covers `PQ2_0`/`PTQ1_0`/`Q2_0_g64` only),
/// so this fixture supplies its own — every value here is exactly `+1.0`
/// or `-1.0`, so `d = 1.0` (exact in `f16`) makes the encoding lossless.
fn encode_q1_0_g128(values: &[f32]) -> BonsaiResult<Vec<u8>> {
    if !values.len().is_multiple_of(128) {
        return Err(BonsaiError::KQuantError {
            reason: format!(
                "Q1_0_g128 fixture encode: length {} is not a multiple of 128",
                values.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(values.len() / 128 * 18);
    for chunk in values.chunks_exact(128) {
        out.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut qs = [0u8; 16];
        for (i, &v) in chunk.iter().enumerate() {
            if v > 0.0 {
                qs[i / 8] |= 1 << (i % 8);
            }
        }
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    out
}

fn quantize_matrix(quant: TensorType, values: &[f32]) -> BonsaiResult<Vec<u8>> {
    match quant {
        TensorType::F32 => Ok(encode_f32(values)),
        TensorType::PQ2_0 => BlockPQ2_0::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::PTQ1_0 => BlockPTQ1_0::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::Q2_0G64 => BlockQ2_0G64::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::TQ2_0_g128 => BlockTQ2_0_g128::quantize(values).map(|b| blocks_to_bytes(&b)),
        TensorType::Q1_0G128 => encode_q1_0_g128(values),
        other => Err(BonsaiError::KQuantError {
            reason: format!("hybrid fixture: {other:?} is not a supported quant choice"),
        }),
    }
}

fn encode_tensor(
    kind: TensorKind,
    quant: TensorType,
    values: &[f32],
) -> BonsaiResult<(TensorType, Vec<u8>)> {
    match kind {
        TensorKind::PlainF32 => Ok((TensorType::F32, encode_f32(values))),
        TensorKind::Bf16 => Ok((TensorType::BF16, encode_bf16(values))),
        TensorKind::Quant => Ok((quant, quantize_matrix(quant, values)?)),
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 10. The built fixture
// ═════════════════════════════════════════════════════════════════════════

static FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_temp_path(tag: &str, seed: u64) -> PathBuf {
    let n = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "oxibonsai_hybrid_fixture_{tag}_{}_{seed}_{n}.gguf",
        std::process::id()
    ))
}

/// Precomputed f64 forward-pass outputs (the reference model's answer for
/// this fixture's fixed token sequence), stored once at build time so a
/// consuming test never has to re-derive it.
pub struct ReferenceForward {
    pub token_ids: Vec<usize>,
    /// `[T_TOKENS][vocab]` logits.
    pub logits: Vec<Vec<f64>>,
    /// `[T_TOKENS][hidden]` pre-final-norm residual stream, for diagnostics.
    pub final_hidden: Vec<Vec<f64>>,
}

/// A generated hybrid GGUF fixture: the file on disk, its parsed
/// configuration, the exact pre-quantization weight values (for
/// byte-fidelity checks), and the independent f64 reference forward.
pub struct HybridFixture {
    pub path: PathBuf,
    pub spec: HybridFixtureSpec,
    pub dims: Dims,
    pub cfg: HybridConfig,
    pub hadamard: Option<HadamardConfig>,
    tensors: BTreeMap<String, PlannedTensor>,
    pub reference: ReferenceForward,
}

impl HybridFixture {
    /// The exact row-major pre-quantization `f32` values written for
    /// tensor `name` (already bf16-rounded for `ssm_alpha`/`ssm_beta`), or
    /// `None` if no such tensor was planned.
    pub fn planned_values(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|t| t.values.as_slice())
    }

    pub fn planned_shape(&self, name: &str) -> Option<&[u64]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

impl Drop for HybridFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A metadata-only fixture for the one configuration design §3.3 says a
/// hybrid loader must *refuse*: `gdn_v_grouped == false` (columns folded
/// tiled) with `v_per_k > 1` (an ungrouped fold cannot be re-derived from
/// a tiled read). No `ReferenceForward` is computed — there is no correct
/// answer for a configuration the spec says to reject, so this only
/// carries what `build()` and `parse()` need to exist and be consistent.
pub struct InvalidUngroupedFixture {
    pub path: PathBuf,
    pub v_per_k: usize,
}

impl Drop for InvalidUngroupedFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Build the negative fixture described on [`InvalidUngroupedFixture`]:
/// hadamard on (`gdn_v_grouped` only exists as a key at all when the
/// Hadamard contract is present), but `gdn_v_grouped = false` in the
/// metadata while the weight layout still uses `n_k_heads = 2` (`v_per_k ==
/// 2`) — exactly the combination the design says must be rejected once a
/// loader exists (B2-10/B2-18).
pub fn build_invalid_ungrouped_fixture(
    seed: u64,
    quant: TensorType,
) -> BonsaiResult<InvalidUngroupedFixture> {
    // `gdn_v_grouped: true` here only selects `n_k_heads == 2` (`v_per_k ==
    // 2`) via `Dims::from_spec`; the metadata's own `gdn_v_grouped` value
    // is overridden to `false` below, producing the inconsistent pair.
    let spec = HybridFixtureSpec {
        quant,
        hadamard: true,
        gdn_v_grouped: true,
        seed,
    };
    let plan = build_plan(&spec);
    let path = unique_temp_path("invalid_ungrouped", seed);
    write_gguf(&spec, &plan, &path, false)?;
    Ok(InvalidUngroupedFixture {
        path,
        v_per_k: plan.dims.v_per_k(),
    })
}

/// Build one hybrid GGUF fixture and its embedded f64 reference forward.
pub fn build(spec: &HybridFixtureSpec) -> BonsaiResult<HybridFixture> {
    let plan = build_plan(spec);
    let path = unique_temp_path("hybrid", spec.seed);
    write_gguf(spec, &plan, &path, spec.gdn_v_grouped)?;

    let bytes = std::fs::read(&path).map_err(BonsaiError::MmapError)?;
    let file = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes)?;
    let cfg = HybridConfig::from_metadata(&file.metadata)?;
    let hadamard = HadamardConfig::from_metadata(&file.metadata)?;
    if let Some(had) = &hadamard {
        had.validate_against_tensors(&file.tensors)?;
    }

    let reference = run_reference_forward(spec, &plan);

    Ok(HybridFixture {
        path,
        spec: *spec,
        dims: plan.dims,
        cfg,
        hadamard,
        tensors: plan.tensors,
        reference,
    })
}

fn write_gguf(
    spec: &HybridFixtureSpec,
    plan: &FixturePlan,
    path: &std::path::Path,
    gdn_v_grouped_metadata: bool,
) -> BonsaiResult<()> {
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen35".to_string()),
    );
    w.add_metadata(
        "general.name",
        MetadataWriteValue::Str("hybrid-fixture".to_string()),
    );
    w.add_metadata(
        "qwen35.embedding_length",
        MetadataWriteValue::U32(HIDDEN as u32),
    );
    w.add_metadata(
        "qwen35.block_count",
        MetadataWriteValue::U32(BLOCK_COUNT as u32),
    );
    w.add_metadata(
        "qwen35.attention.head_count",
        MetadataWriteValue::U32(N_HEAD as u32),
    );
    w.add_metadata(
        "qwen35.attention.head_count_kv",
        MetadataWriteValue::U32(N_KV as u32),
    );
    w.add_metadata(
        "qwen35.attention.key_length",
        MetadataWriteValue::U32(HEAD_DIM as u32),
    );
    w.add_metadata(
        "qwen35.attention.value_length",
        MetadataWriteValue::U32(HEAD_DIM as u32),
    );
    w.add_metadata(
        "qwen35.feed_forward_length",
        MetadataWriteValue::U32(FFN as u32),
    );
    w.add_metadata("qwen35.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
    w.add_metadata(
        "qwen35.context_length",
        MetadataWriteValue::U32(CONTEXT_LENGTH as u32),
    );
    w.add_metadata(
        "qwen35.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(RMS_EPS as f32),
    );
    w.add_metadata(
        "qwen35.rope.freq_base",
        MetadataWriteValue::F32(ROPE_FREQ_BASE as f32),
    );
    w.add_metadata(
        "qwen35.full_attention_interval",
        MetadataWriteValue::U32(FULL_ATTENTION_INTERVAL as u32),
    );
    w.add_metadata(
        "qwen35.rope.dimension_count",
        MetadataWriteValue::U32(ROPE_DIM as u32),
    );
    w.add_metadata(
        "qwen35.rope.dimension_sections",
        MetadataWriteValue::ArrayI32(ROPE_SECTIONS.to_vec()),
    );
    w.add_metadata(
        "qwen35.ssm.conv_kernel",
        MetadataWriteValue::U32(SSM_CONV_KERNEL as u32),
    );
    w.add_metadata(
        "qwen35.ssm.state_size",
        MetadataWriteValue::U32(SSM_STATE_SIZE as u32),
    );
    w.add_metadata(
        "qwen35.ssm.group_count",
        MetadataWriteValue::U32(plan.dims.n_k_heads as u32),
    );
    w.add_metadata(
        "qwen35.ssm.time_step_rank",
        MetadataWriteValue::U32(SSM_TIME_STEP_RANK as u32),
    );
    w.add_metadata(
        "qwen35.ssm.inner_size",
        MetadataWriteValue::U32(plan.dims.inner_size as u32),
    );
    w.add_metadata("general.sampling.temp", MetadataWriteValue::F32(1.0));
    w.add_metadata("general.sampling.top_p", MetadataWriteValue::F32(0.95));
    w.add_metadata("general.sampling.top_k", MetadataWriteValue::U32(20));

    // ── id-42 disambiguation metadata (design §7.2 "id-42 resolution" row,
    // wave-2.5 addendum item 4): `general.quantization_version` is the only
    // real discriminator between the on-disk readings of wire id 42
    // (`oxibonsai_core::gguf::quant_resolve` module docs) — the mainline
    // `Q2_0_g64` reading carries the numeric `u32 2` PrismML files use,
    // while OxiBonsai's own legacy `TQ2_0_g128` writer emits the **string**
    // `"TQ2_0_G128"` instead. Give each wire-id-42 fixture variant its real
    // spelling so `resolve_type_42_with_sample` can be exercised against
    // these fixtures, not only against real model files. Other quant
    // formats (F32/PQ2_0/PTQ1_0/Q1_0_g128) are never ambiguous at the
    // wire-id level, so this key is left unset for them.
    match spec.quant {
        TensorType::Q2_0G64 => {
            w.add_metadata("general.quantization_version", MetadataWriteValue::U32(2));
        }
        TensorType::TQ2_0_g128 => {
            w.add_metadata(
                "general.quantization_version",
                MetadataWriteValue::Str("TQ2_0_G128".to_string()),
            );
        }
        _ => {}
    }

    // ── Tiny tokenizer block (enough for HybridConfig/GgufFile to parse a
    // "complete" file; the tokenizer loader itself is a different
    // package's scope) ─────────────────────────────────────────────────
    w.add_metadata(
        "tokenizer.ggml.model",
        MetadataWriteValue::Str("gpt2".to_string()),
    );
    let tokens: Vec<String> = (0..VOCAB).map(|i| format!("<tok{i}>")).collect();
    w.add_metadata(
        "tokenizer.ggml.tokens",
        MetadataWriteValue::ArrayStr(tokens),
    );
    w.add_metadata(
        "tokenizer.ggml.eos_token_id",
        MetadataWriteValue::U32((VOCAB - 1) as u32),
    );
    w.add_metadata("tokenizer.ggml.bos_token_id", MetadataWriteValue::U32(0));
    w.add_metadata(
        "tokenizer.ggml.add_bos_token",
        MetadataWriteValue::Bool(false),
    );

    // ── Hadamard contract ────────────────────────────────────────────────
    if spec.hadamard {
        w.add_metadata("prism.hadamard.version", MetadataWriteValue::U32(1));
        w.add_metadata(
            "prism.hadamard.block_size",
            MetadataWriteValue::U32(HADAMARD_BLOCK_SIZE as u32),
        );
        w.add_metadata(
            "prism.hadamard.transform",
            MetadataWriteValue::Str("normalized-sylvester-walsh-hadamard".to_string()),
        );
        w.add_metadata(
            "prism.hadamard.axis",
            MetadataWriteValue::Str("input-last-dimension".to_string()),
        );
        w.add_metadata(
            "prism.hadamard.sign_mode",
            MetadataWriteValue::Str("explicit".to_string()),
        );
        let widths: Vec<i32> = plan.signs.keys().map(|&w| w as i32).collect();
        let mut values: Vec<i32> = Vec::new();
        for width in plan.signs.keys() {
            for &s in &plan.signs[width] {
                values.push(s as i32);
            }
        }
        w.add_metadata(
            "prism.hadamard.sign_widths",
            MetadataWriteValue::ArrayI32(widths),
        );
        w.add_metadata(
            "prism.hadamard.sign_values",
            MetadataWriteValue::ArrayI32(values),
        );
        w.add_metadata(
            "prism.hadamard.weight_names",
            MetadataWriteValue::ArrayStr(plan.folded_names.clone()),
        );
        w.add_metadata(
            "prism.hadamard.inverse_weight_names",
            MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
        );
        w.add_metadata(
            "prism.hadamard.gdn_v_grouped",
            MetadataWriteValue::Bool(gdn_v_grouped_metadata),
        );
    }

    for (name, planned) in &plan.tensors {
        let (wire_type, bytes) = encode_tensor(planned.kind, spec.quant, &planned.values)?;
        w.add_tensor(TensorEntry {
            name: name.clone(),
            shape: planned.shape.clone(),
            tensor_type: wire_type,
            data: bytes,
        });
    }

    let mut file = std::fs::File::create(path).map_err(BonsaiError::MmapError)?;
    w.write(&mut file).map_err(|e| BonsaiError::KQuantError {
        reason: format!("hybrid fixture write failed: {e}"),
    })?;
    Ok(())
}

// ═════════════════════════════════════════════════════════════════════════
// 11. The f64 reference forward driver
// ═════════════════════════════════════════════════════════════════════════

/// Fetch tensor `name`'s planned values as `f64` (exact widening).
///
/// Panics (rather than silently returning `vec![]`) if `name` was never
/// planned: this is the f64 reference model's own ground truth, so a
/// name typo in the forward driver below must fail loudly, not degrade
/// into an all-zero output that `zip`-truncated arithmetic would silently
/// propagate. Every call site is reached only from a branch that matches
/// `build_plan`'s own construction of that name (full-attention names
/// only under `is_full_attention`, `ssm_*` names only under its `else`),
/// so this is a real invariant, not merely convenient.
fn f64_values(plan: &FixturePlan, name: &str) -> Vec<f64> {
    let planned = plan.tensors.get(name).unwrap_or_else(|| {
        panic!("hybrid fixture reference forward: tensor '{name}' was never planned by build_plan")
    });
    planned.values.iter().map(|&v| v as f64).collect()
}

/// `y = W^T x` for a row-major `[in, out]`-shaped weight (`W` stored as
/// `out` rows of `in` values each, matching GGUF's `[ne0=in, ne1=out]`
/// convention): `y[o] = sum_i W[o][i] * x[i]`.
fn matmul_row_major(w: &[f64], in_dim: usize, out_dim: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        y[o] = row.iter().zip(x).map(|(&wi, &xi)| wi * xi).sum();
    }
    y
}

struct HadamardRuntime {
    block_size: usize,
    signs: BTreeMap<usize, Vec<f64>>,
}

impl HadamardRuntime {
    /// Panics on a width with no configured sign vector, rather than
    /// silently skipping the rotation: `rotate` is only ever called on the
    /// three widths `build_plan` populates `signs` for
    /// (`HIDDEN`/`attn_output_input_width` (== `inner_size` by this
    /// fixture's chosen dims) /`FFN`), so reaching `None` here means a call
    /// site and `build_plan`'s sign-width set have drifted apart — exactly
    /// the class of silent-wrong-math bug design §3.4 warns a skipped
    /// rotation is, so it must fail loudly rather than let the caller run
    /// un-rotated data through a folded matmul unnoticed.
    fn rotate(&self, x: &[f64]) -> Vec<f64> {
        let width = x.len();
        match self.signs.get(&width) {
            Some(s) => fwht_forward_signed_f64(x, s, self.block_size),
            None => panic!(
                "hybrid fixture reference forward: no Hadamard sign vector configured for \
                 width {width}; rotate() must only be called on a folded activation's width"
            ),
        }
    }

    fn inverse_embedding(&self, x: &[f64]) -> Vec<f64> {
        let width = x.len();
        match self.signs.get(&width) {
            Some(s) => fwht_inverse_signed_f64(x, s, self.block_size),
            None => panic!(
                "hybrid fixture reference forward: no Hadamard sign vector configured for \
                 width {width}; inverse_embedding() must only be called on the embedding width"
            ),
        }
    }
}

fn run_reference_forward(spec: &HybridFixtureSpec, plan: &FixturePlan) -> ReferenceForward {
    let dims = plan.dims;
    let hadamard = if spec.hadamard {
        Some(HadamardRuntime {
            block_size: HADAMARD_BLOCK_SIZE,
            signs: plan.signs.clone(),
        })
    } else {
        None
    };
    let rotate = |x: &[f64]| -> Vec<f64> {
        match &hadamard {
            Some(h) => h.rotate(x),
            None => x.to_vec(),
        }
    };

    let mut rng = Xorshift64Star::new(spec.seed ^ 0xC0FF_EE00_D15E_A5E5);
    let token_ids: Vec<usize> = (0..T_TOKENS).map(|_| rng.next_usize_below(VOCAB)).collect();

    let full_output_in = dims.attn_output_input_width();
    let vhead_map = VHeadMap::new(dims.n_k_heads, dims.n_v_heads);

    // Per-linear-layer recurrent state.
    let linear_layers: Vec<usize> = (0..BLOCK_COUNT)
        .filter(|&l| !dims.is_full_attention(l))
        .collect();
    let mut conv_state: BTreeMap<usize, Vec<Vec<f64>>> = linear_layers
        .iter()
        .map(|&l| (l, vec![vec![0.0f64; SSM_CONV_KERNEL - 1]; dims.conv_dim]))
        .collect();
    let mut gdn_state: BTreeMap<usize, Vec<f64>> = linear_layers
        .iter()
        .map(|&l| {
            (
                l,
                vec![0.0f64; dims.n_v_heads * dims.head_v_dim * dims.head_k_dim],
            )
        })
        .collect();

    // Per-full-layer KV history.
    let full_layers: Vec<usize> = (0..BLOCK_COUNT)
        .filter(|&l| dims.is_full_attention(l))
        .collect();
    let mut k_hist: BTreeMap<usize, Vec<Vec<Vec<f64>>>> = full_layers
        .iter()
        .map(|&l| (l, vec![Vec::new(); N_KV]))
        .collect();
    let mut v_hist: BTreeMap<usize, Vec<Vec<Vec<f64>>>> = full_layers
        .iter()
        .map(|&l| (l, vec![Vec::new(); N_KV]))
        .collect();

    let token_embd = f64_values(plan, "token_embd.weight");
    let output_norm_w = f64_values(plan, "output_norm.weight");
    let output_w = f64_values(plan, "output.weight");

    let mut logits = Vec::with_capacity(T_TOKENS);
    let mut final_hidden = Vec::with_capacity(T_TOKENS);

    for (pos, &tok) in token_ids.iter().enumerate() {
        let embed_row: Vec<f64> = token_embd[tok * HIDDEN..(tok + 1) * HIDDEN].to_vec();
        let mut h = match &hadamard {
            Some(hr) => hr.inverse_embedding(&embed_row),
            None => embed_row,
        };

        for layer in 0..BLOCK_COUNT {
            let residual = h.clone();
            let attn_norm_w = f64_values(plan, &tname(layer, "attn_norm.weight"));
            let normed = rms_norm_f64(&h, &attn_norm_w, RMS_EPS);

            if dims.is_full_attention(layer) {
                let a_rot = rotate(&normed);
                let wq = f64_values(plan, &tname(layer, "attn_q.weight"));
                let wk = f64_values(plan, &tname(layer, "attn_k.weight"));
                let wv = f64_values(plan, &tname(layer, "attn_v.weight"));
                let qfull = matmul_row_major(&wq, HIDDEN, N_HEAD * HEAD_DIM * 2, &a_rot);
                let kfull = matmul_row_major(&wk, HIDDEN, N_KV * HEAD_DIM, &a_rot);
                let vfull = matmul_row_major(&wv, HIDDEN, N_KV * HEAD_DIM, &a_rot);

                let q_norm_w = f64_values(plan, &tname(layer, "attn_q_norm.weight"));
                let k_norm_w = f64_values(plan, &tname(layer, "attn_k_norm.weight"));

                // K/V for this position, per KV head, RMS-normed + RoPE'd.
                for kv in 0..N_KV {
                    let mut k_h = kfull[kv * HEAD_DIM..(kv + 1) * HEAD_DIM].to_vec();
                    k_h = rms_norm_f64(&k_h, &k_norm_w, RMS_EPS);
                    apply_partial_rope_f64(&mut k_h, pos, ROPE_DIM, ROPE_FREQ_BASE);
                    let v_h = vfull[kv * HEAD_DIM..(kv + 1) * HEAD_DIM].to_vec();
                    // `k_hist`/`v_hist` were pre-populated with one N_KV-slot
                    // vector for every layer satisfying `is_full_attention`,
                    // using that same predicate; we are inside the branch
                    // that predicate selected, so `layer` is always present,
                    // and `kv < N_KV` by the loop bound above.
                    k_hist
                        .get_mut(&layer)
                        .expect("full-attention layer must be present in k_hist")
                        .get_mut(kv)
                        .expect("kv head index must be < N_KV")
                        .push(k_h);
                    v_hist
                        .get_mut(&layer)
                        .expect("full-attention layer must be present in v_hist")
                        .get_mut(kv)
                        .expect("kv head index must be < N_KV")
                        .push(v_h);
                }

                let group_size = N_HEAD / N_KV;
                let scale = 1.0 / (HEAD_DIM as f64).sqrt();
                let mut attn_out = vec![0.0f64; N_HEAD * HEAD_DIM];
                let mut gate_all = vec![0.0f64; N_HEAD * HEAD_DIM];
                for head in 0..N_HEAD {
                    let base = head * 2 * HEAD_DIM;
                    let mut q_h = qfull[base..base + HEAD_DIM].to_vec();
                    let gate_h = &qfull[base + HEAD_DIM..base + 2 * HEAD_DIM];
                    q_h = rms_norm_f64(&q_h, &q_norm_w, RMS_EPS);
                    apply_partial_rope_f64(&mut q_h, pos, ROPE_DIM, ROPE_FREQ_BASE);

                    let kv_head = head / group_size;
                    let ks = &k_hist[&layer][kv_head];
                    let vs = &v_hist[&layer][kv_head];
                    let scores: Vec<f64> = ks
                        .iter()
                        .map(|k| scale * k.iter().zip(&q_h).map(|(&ki, &qi)| ki * qi).sum::<f64>())
                        .collect();
                    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                    let sum_s: f64 = exp_s.iter().sum();
                    let weights: Vec<f64> = exp_s.iter().map(|&e| e / sum_s).collect();
                    let mut o_h = vec![0.0f64; HEAD_DIM];
                    for (t, v) in vs.iter().enumerate() {
                        for d in 0..HEAD_DIM {
                            o_h[d] += weights[t] * v[d];
                        }
                    }
                    attn_out[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&o_h);
                    gate_all[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(gate_h);
                }
                debug_assert_eq!(attn_out.len(), full_output_in);
                let gated: Vec<f64> = attn_out
                    .iter()
                    .zip(&gate_all)
                    .map(|(&a, &g)| a * sigmoid_f64(g))
                    .collect();
                let o_rot = rotate(&gated);
                let wo = f64_values(plan, &tname(layer, "attn_output.weight"));
                let attn_result = matmul_row_major(&wo, full_output_in, HIDDEN, &o_rot);
                h = residual
                    .iter()
                    .zip(&attn_result)
                    .map(|(&r, &a)| r + a)
                    .collect();
            } else {
                let a_rot = rotate(&normed);
                let wqkv = f64_values(plan, &tname(layer, "attn_qkv.weight"));
                let wgate = f64_values(plan, &tname(layer, "attn_gate.weight"));
                let mut qkv = matmul_row_major(&wqkv, HIDDEN, dims.conv_dim, &a_rot);
                let z_tiled = matmul_row_major(&wgate, HIDDEN, dims.inner_size, &a_rot);

                // ssm_alpha/ssm_beta are NOT folded: consume the unrotated
                // `normed` (design §3.4's "easy-to-miss" trap).
                let w_alpha = f64_values(plan, &tname(layer, "ssm_alpha.weight"));
                let w_beta = f64_values(plan, &tname(layer, "ssm_beta.weight"));
                let alpha_raw_tiled = matmul_row_major(&w_alpha, HIDDEN, dims.n_v_heads, &normed);
                let beta_raw_tiled = matmul_row_major(&w_beta, HIDDEN, dims.n_v_heads, &normed);

                let conv_w_flat = f64_values(plan, &tname(layer, "ssm_conv1d.weight"));
                // ne=[kc, conv_dim] row-major-per-channel: channel c's kc
                // weights are conv_w_flat[c*kc .. c*kc+kc].
                let conv_w: Vec<Vec<f64>> = (0..dims.conv_dim)
                    .map(|c| conv_w_flat[c * SSM_CONV_KERNEL..(c + 1) * SSM_CONV_KERNEL].to_vec())
                    .collect();
                // `conv_state` was pre-populated for every layer NOT
                // satisfying `is_full_attention`, using that same
                // predicate; we are inside the `else` of that branch.
                let state = conv_state
                    .get_mut(&layer)
                    .expect("linear-attention layer must be present in conv_state");
                qkv = causal_conv1d_step_f64(state, &qkv, &conv_w, dims.conv_dim, SSM_CONV_KERNEL);
                qkv = qkv.into_iter().map(silu_f64).collect();

                let q_width = dims.n_k_heads * dims.head_k_dim;
                let k_width = dims.n_k_heads * dims.head_k_dim;
                let (q_all, rest) = qkv.split_at(q_width);
                let (k_all, v_tiled) = rest.split_at(k_width);

                // L2-norm the joint Q/K region, per K-head.
                let mut q_normed = vec![0.0f64; q_width];
                let mut k_normed = vec![0.0f64; k_width];
                for kh in 0..dims.n_k_heads {
                    let qs = &q_all[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let ks = &k_all[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let qn = l2_norm_f64(qs, 1e-6);
                    let kn = l2_norm_f64(ks, 1e-6);
                    q_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim].copy_from_slice(&qn);
                    k_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim].copy_from_slice(&kn);
                }

                let ssm_a = f64_values(plan, &tname(layer, "ssm_a"));
                let ssm_dt_bias = f64_values(plan, &tname(layer, "ssm_dt.bias"));

                // Same invariant as `conv_state` above: `gdn_state` is
                // pre-populated for every linear-attention layer.
                let state = gdn_state
                    .get_mut(&layer)
                    .expect("linear-attention layer must be present in gdn_state");
                let mut out_grouped = vec![0.0f64; dims.n_v_heads * dims.head_v_dim];
                for m in 0..dims.n_v_heads {
                    let tiled_j = vhead_map.tiled(m);
                    let kh = vhead_map.k_head(m);
                    let q_h = &q_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let k_h = &k_normed[kh * dims.head_k_dim..(kh + 1) * dims.head_k_dim];
                    let v_h = &v_tiled[tiled_j * dims.head_v_dim..(tiled_j + 1) * dims.head_v_dim];
                    let state_slab = &mut state[m * dims.head_v_dim * dims.head_k_dim
                        ..(m + 1) * dims.head_v_dim * dims.head_k_dim];
                    let (out_h, _decay, _delta) = gdn_step_fused_f64(
                        state_slab,
                        q_h,
                        k_h,
                        v_h,
                        alpha_raw_tiled[tiled_j],
                        beta_raw_tiled[tiled_j],
                        ssm_dt_bias[tiled_j],
                        ssm_a[tiled_j],
                        dims.head_k_dim,
                        dims.head_v_dim,
                    );
                    out_grouped[m * dims.head_v_dim..(m + 1) * dims.head_v_dim]
                        .copy_from_slice(&out_h);
                }

                // z is TILED (same as v); re-index to GROUPED to align with
                // `out_grouped` before the gated norm (design §3.4).
                let ssm_norm_w = f64_values(plan, &tname(layer, "ssm_norm.weight"));
                let mut gated_out = vec![0.0f64; dims.n_v_heads * dims.head_v_dim];
                for m in 0..dims.n_v_heads {
                    let tiled_j = vhead_map.tiled(m);
                    let o_h = &out_grouped[m * dims.head_v_dim..(m + 1) * dims.head_v_dim];
                    let z_h = &z_tiled[tiled_j * dims.head_v_dim..(tiled_j + 1) * dims.head_v_dim];
                    let g_h = gated_rms_norm_head_f64(o_h, z_h, &ssm_norm_w, RMS_EPS);
                    gated_out[m * dims.head_v_dim..(m + 1) * dims.head_v_dim].copy_from_slice(&g_h);
                }

                // The recurrent state is kept GROUPED (design §3.6), so
                // `ssm_out`'s fold degenerates to a plain `rotate`.
                let o_rot = rotate(&gated_out);
                let w_out = f64_values(plan, &tname(layer, "ssm_out.weight"));
                let ssm_result = matmul_row_major(&w_out, dims.inner_size, HIDDEN, &o_rot);
                h = residual
                    .iter()
                    .zip(&ssm_result)
                    .map(|(&r, &s)| r + s)
                    .collect();
            }

            // ── Shared FFN ───────────────────────────────────────────────
            let residual2 = h.clone();
            let post_norm_w = f64_values(plan, &tname(layer, "post_attention_norm.weight"));
            let f = rms_norm_f64(&h, &post_norm_w, RMS_EPS);
            let f_rot = rotate(&f);
            let w_gate = f64_values(plan, &tname(layer, "ffn_gate.weight"));
            let w_up = f64_values(plan, &tname(layer, "ffn_up.weight"));
            let gate_out = matmul_row_major(&w_gate, HIDDEN, FFN, &f_rot);
            let up_out = matmul_row_major(&w_up, HIDDEN, FFN, &f_rot);
            let m_vec: Vec<f64> = gate_out
                .iter()
                .zip(&up_out)
                .map(|(&g, &u)| silu_f64(g) * u)
                .collect();
            let m_rot = rotate(&m_vec);
            let w_down = f64_values(plan, &tname(layer, "ffn_down.weight"));
            let down_out = matmul_row_major(&w_down, FFN, HIDDEN, &m_rot);
            h = residual2
                .iter()
                .zip(&down_out)
                .map(|(&r, &d)| r + d)
                .collect();
        }

        final_hidden.push(h.clone());
        let normed_final = rms_norm_f64(&h, &output_norm_w, RMS_EPS);
        let final_rot = rotate(&normed_final);
        let token_logits = matmul_row_major(&output_w, HIDDEN, VOCAB, &final_rot);
        logits.push(token_logits);
    }

    ReferenceForward {
        token_ids,
        logits,
        final_hidden,
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 12. Internal self-checks — exposed as `pub fn`s so
//     `hybrid_fixture_tests.rs` can assert them without duplicating the
//     math, but independent of `build()`/`run_reference_forward` so a
//     wiring bug in the forward driver cannot mask a bug in a primitive.
// ═════════════════════════════════════════════════════════════════════════

/// `inverse(forward(x)) == x` (design §7.2's FWHT round-trip row), for a
/// deterministic pseudo-random `x` and sign vector of the given width.
pub fn fwht_round_trip_check(seed: u64, width: usize, block: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Xorshift64Star::new(seed);
    let x: Vec<f64> = (0..width).map(|_| rng.next_range_f64(-3.0, 3.0)).collect();
    let signs: Vec<f64> = (0..width)
        .map(|_| {
            if rng.next_u64().is_multiple_of(2) {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let folded = fwht_forward_signed_f64(&x, &signs, block);
    let recovered = fwht_inverse_signed_f64(&folded, &signs, block);
    (x, recovered)
}

/// `(fused_two_steps, naive_two_steps, quadratic_two_steps)`, each
/// `[2][head_v_dim]` — see [`gdn_cross_check`].
pub type GdnCrossCheckResult = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// One GDN step computed three independent ways: fused single-pass, naive
/// 3-pass, and (via a 2-step sequential run feeding the quadratic
/// reconstruction) the O(T²) closed form. Returns
/// [`GdnCrossCheckResult`] for the caller to compare pairwise.
pub fn gdn_cross_check(seed: u64, head_k_dim: usize, head_v_dim: usize) -> GdnCrossCheckResult {
    let mut rng = Xorshift64Star::new(seed);
    let mut gen_vec =
        |n: usize| -> Vec<f64> { (0..n).map(|_| rng.next_range_f64(-1.0, 1.0)).collect() };

    let q_steps: Vec<Vec<f64>> = (0..2).map(|_| gen_vec(head_k_dim)).collect();
    let k_steps: Vec<Vec<f64>> = (0..2).map(|_| gen_vec(head_k_dim)).collect();
    let v_steps: Vec<Vec<f64>> = (0..2).map(|_| gen_vec(head_v_dim)).collect();
    let alpha_steps = gen_vec(2);
    let beta_steps = gen_vec(2);
    let dt_bias = rng.next_range_f64(-0.5, 0.5);
    let a_neg = -rng.next_range_f64(0.1, 1.0);

    let mut state_fused = vec![0.0f64; head_v_dim * head_k_dim];
    let mut state_naive = vec![0.0f64; head_v_dim * head_k_dim];
    let mut fused_out = Vec::with_capacity(2);
    let mut naive_out = Vec::with_capacity(2);
    let mut decay_hist = Vec::with_capacity(2);
    let mut delta_hist = Vec::with_capacity(2);
    for t in 0..2 {
        let (out, decay, delta) = gdn_step_fused_f64(
            &mut state_fused,
            &q_steps[t],
            &k_steps[t],
            &v_steps[t],
            alpha_steps[t],
            beta_steps[t],
            dt_bias,
            a_neg,
            head_k_dim,
            head_v_dim,
        );
        fused_out.push(out);
        decay_hist.push(decay);
        delta_hist.push(delta);

        let out2 = gdn_step_naive_3pass_f64(
            &mut state_naive,
            &q_steps[t],
            &k_steps[t],
            &v_steps[t],
            alpha_steps[t],
            beta_steps[t],
            dt_bias,
            a_neg,
            head_k_dim,
            head_v_dim,
        );
        naive_out.push(out2);
    }

    let quadratic_out =
        gdn_quadratic_reconstruction_f64(&decay_hist, &delta_hist, &k_steps, &q_steps, head_v_dim);

    (fused_out, naive_out, quadratic_out)
}

/// Applies partial RoPE and returns `(before, after)` so a caller can
/// assert the untouched tail (`n_rot..head_dim`) is bit-identical while
/// the rotated head (`0..n_rot`) changed.
pub fn partial_rope_check(
    seed: u64,
    pos: usize,
    n_rot: usize,
    head_dim: usize,
    freq_base: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Xorshift64Star::new(seed);
    let before: Vec<f64> = (0..head_dim)
        .map(|_| rng.next_range_f64(-3.0, 3.0))
        .collect();
    let mut after = before.clone();
    apply_partial_rope_f64(&mut after, pos, n_rot, freq_base);
    (before, after)
}
