//! The independent f64 scalar primitives of the reference model: norms,
//! partial RoPE, the blockwise Walsh-Hadamard transform with its fused sign
//! flip, the V-head grouping map, the Gated DeltaNet step in three forms and
//! the causal depthwise conv1d. Nothing here calls `oxibonsai-kernels`; the
//! module doc of the parent explains why.

// ═════════════════════════════════════════════════════════════════════════
// 2. Small f64 scalar math primitives (independent of every production
//    kernel — see the module doc's "no shared code" note)
// ═════════════════════════════════════════════════════════════════════════

pub(super) fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub(super) fn silu_f64(x: f64) -> f64 {
    x * sigmoid_f64(x)
}

/// `x > 20 ? x : ln(1 + exp(x))` — the exact ggml softplus cutoff (design
/// §2.3), matching the kernels' own `softplus_scalar_elem`
/// (`oxibonsai-kernels/src/norms.rs`).
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
pub fn rms_norm_f64(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
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
pub fn l2_norm_f64(x: &[f64], eps: f64) -> Vec<f64> {
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    let denom = sum_sq.sqrt().max(eps);
    x.iter().map(|&v| v / denom).collect()
}

/// Gated RMSNorm for one V-head: `out[j] = weight[j] * o[j] * inv_rms *
/// silu(z[j])`, `inv_rms` from the *mean*-based RMSNorm formula over this
/// head's slice only (mirrors `rms_norm_gated_scalar` in
/// `oxibonsai-kernels/src/norms.rs`, read for verification, not called).
pub fn gated_rms_norm_head_f64(o: &[f64], z: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
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
pub fn apply_partial_rope_f64(x: &mut [f64], pos: usize, n_rot: usize, freq_base: f64) {
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
pub fn fwht_forward_signed_f64(x: &[f64], signs: &[f64], block: usize) -> Vec<f64> {
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
pub(super) fn fwht_inverse_signed_f64(x: &[f64], signs: &[f64], block: usize) -> Vec<f64> {
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
pub fn gdn_step_fused_f64(
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
pub(super) fn gdn_step_naive_3pass_f64(
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
pub(super) fn gdn_quadratic_reconstruction_f64(
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
pub fn causal_conv1d_step_f64(
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
