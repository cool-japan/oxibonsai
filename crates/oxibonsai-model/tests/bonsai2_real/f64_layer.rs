//! An independent `f64` evaluation of single `qwen35` layers, straight from
//! a GGUF's bytes — design §8.2 **G3**: "f64-reference self-consistency of
//! our hybrid forward on the real 27B at layers 0, 3, 7, 31, 63".
//!
//! # Independence
//!
//! Nothing here calls production model or kernel code. Weights are decoded
//! from the raw tensor bytes by this file's own `PQ2_0` / `F32` / `F16` /
//! `BF16` readers (the fork's `dequantize_row_pq2_0` layout: `d` first, then
//! 2-bit codes LSB-first, `code - 1`), the Hadamard fold is re-derived from
//! the `prism.hadamard.*` metadata with this file's own Sylvester butterfly,
//! and every norm, the partial NeoX RoPE, the GQA attention, the causal
//! conv, the L2 norm and the gated delta rule are written out in `f64` from
//! the design text (§2.3–§2.6, §3.2–§3.5). The whole reference is itself
//! validated end to end against the synthetic fixture's own independent
//! `f64` model (`hybrid_f64_layer_reference_matches_the_fixture_bonsai2`).
//!
//! # Memory
//!
//! Weights are never materialised: every matrix is decoded one output row
//! at a time (per rayon worker) and applied to all positions at once, so a
//! whole 27B layer is evaluated with a few hundred KiB of `f64` scratch.

#![allow(dead_code)]

use std::collections::HashMap;

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::hadamard_config::HadamardConfig;
use rayon::prelude::*;

/// `blk.{layer}.{suffix}`.
#[must_use]
pub fn tensor_name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

// ─────────────────────────────────────────────────────────────────────────
//  Raw tensor decoding
// ─────────────────────────────────────────────────────────────────────────

fn f16_bits_to_f64(bits: u16) -> f64 {
    f64::from(half::f16::from_bits(bits).to_f32())
}

fn bf16_bits_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// Decode row `r` (`row_len` elements) of a `[row_len, rows]` tensor.
fn decode_row(ty: GgufTensorType, data: &[u8], row_len: usize, r: usize, out: &mut [f64]) {
    match ty {
        GgufTensorType::F32 => {
            let base = r * row_len * 4;
            for (i, o) in out.iter_mut().enumerate() {
                let c = &data[base + i * 4..base + i * 4 + 4];
                *o = f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        GgufTensorType::F16 => {
            let base = r * row_len * 2;
            for (i, o) in out.iter_mut().enumerate() {
                let c = &data[base + i * 2..base + i * 2 + 2];
                *o = f16_bits_to_f64(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgufTensorType::BF16 => {
            let base = r * row_len * 2;
            for (i, o) in out.iter_mut().enumerate() {
                let c = &data[base + i * 2..base + i * 2 + 2];
                *o = bf16_bits_to_f64(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgufTensorType::PQ2_0 => {
            // 34-byte blocks of 128: `d` (f16) FIRST, then `qs[32]`, element
            // j at qs[j / 4] >> (2 * (j % 4)), value (code - 1) * d.
            const QK: usize = 128;
            const BYTES: usize = 34;
            let blocks_per_row = row_len / QK;
            let base = r * blocks_per_row * BYTES;
            for b in 0..blocks_per_row {
                let blk = &data[base + b * BYTES..base + (b + 1) * BYTES];
                let d = f16_bits_to_f64(u16::from_le_bytes([blk[0], blk[1]]));
                let qs = &blk[2..BYTES];
                for j in 0..QK {
                    let code = (qs[j / 4] >> (2 * (j % 4))) & 0x03;
                    out[b * QK + j] = (f64::from(code) - 1.0) * d;
                }
            }
        }
        other => panic!("f64 reference: no decoder for {other}"),
    }
}

/// Read access to a GGUF's tensors, decoding to `f64` on the fly.
pub struct Weights<'g, 'a> {
    gguf: &'g GgufFile<'a>,
}

impl<'g, 'a> Weights<'g, 'a> {
    /// Wrap a parsed GGUF.
    #[must_use]
    pub fn new(gguf: &'g GgufFile<'a>) -> Self {
        Self { gguf }
    }

    fn tensor(&self, name: &str) -> (GgufTensorType, Vec<u64>, &'a [u8]) {
        let info = self
            .gguf
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("f64 reference: missing tensor {name}"));
        let data = self
            .gguf
            .tensor_data(name)
            .unwrap_or_else(|e| panic!("f64 reference: {name}: {e}"));
        (info.tensor_type, info.shape.clone(), data)
    }

    /// Every element of a (small) tensor, in storage order.
    #[must_use]
    pub fn all(&self, name: &str) -> Vec<f64> {
        let (ty, shape, data) = self.tensor(name);
        let n = usize::try_from(shape.iter().product::<u64>()).expect("element count");
        let row_len = usize::try_from(shape[0]).expect("row length");
        let rows = n / row_len;
        let mut out = vec![0.0f64; n];
        for (r, chunk) in out.chunks_exact_mut(row_len).enumerate().take(rows) {
            decode_row(ty, data, row_len, r, chunk);
        }
        out
    }

    /// Row `r` of the `[in, out]` matrix `name` (length `in`).
    #[must_use]
    pub fn row(&self, name: &str, r: usize) -> Vec<f64> {
        let (ty, shape, data) = self.tensor(name);
        let row_len = usize::try_from(shape[0]).expect("row length");
        let mut out = vec![0.0f64; row_len];
        decode_row(ty, data, row_len, r, &mut out);
        out
    }

    /// `y_t = W x_t` for every `x_t` in `xs`, where `W` is the GGUF matrix
    /// `name` of shape `[in, out]`. Each weight row is decoded once and
    /// applied to all positions; rows are spread over rayon workers.
    #[must_use]
    pub fn matvec(&self, name: &str, xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (ty, shape, data) = self.tensor(name);
        let in_f = usize::try_from(shape[0]).expect("in features");
        let out_f = usize::try_from(shape[1]).expect("out features");
        for x in xs {
            assert_eq!(x.len(), in_f, "{name}: input width");
        }
        let per_row: Vec<Vec<f64>> = (0..out_f)
            .into_par_iter()
            .map_init(
                || vec![0.0f64; in_f],
                |row, r| {
                    decode_row(ty, data, in_f, r, row);
                    xs.iter()
                        .map(|x| row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>())
                        .collect()
                },
            )
            .collect();
        (0..xs.len())
            .map(|t| per_row.iter().map(|col| col[t]).collect())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Hadamard fold, f64
// ─────────────────────────────────────────────────────────────────────────

/// The `prism.hadamard.*` fold, re-derived in `f64`.
pub struct Fold {
    block: usize,
    signs: HashMap<usize, Vec<f64>>,
    config: HadamardConfig,
}

fn butterfly(buf: &mut [f64]) {
    let n = buf.len();
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let (u, v) = (buf[j], buf[j + len]);
                buf[j] = u + v;
                buf[j + len] = u - v;
            }
            i += 2 * len;
        }
        len *= 2;
    }
}

impl Fold {
    /// The fold a GGUF declares, or `None` for an unfolded file.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Option<Self> {
        let config = HadamardConfig::from_metadata(&gguf.metadata)
            .expect("prism.hadamard metadata parses")?;
        let signs = config
            .signs
            .iter()
            .map(|(width, s)| (*width, s.iter().map(|&v| f64::from(v)).collect()))
            .collect();
        Some(Self {
            block: config.block_size,
            signs,
            config,
        })
    }

    /// Whether matrix `name` was folded (its input must be rotated).
    #[must_use]
    pub fn is_folded(&self, name: &str) -> bool {
        self.config.is_folded(name)
    }

    /// Forward rotation of an activation: `FWHT_block(x ⊙ s) / sqrt(block)`.
    #[must_use]
    pub fn rotate(&self, x: &[f64]) -> Vec<f64> {
        let signs = self
            .signs
            .get(&x.len())
            .unwrap_or_else(|| panic!("no Hadamard signs for width {}", x.len()));
        let inv = 1.0 / (self.block as f64).sqrt();
        let mut out: Vec<f64> = x.iter().zip(signs).map(|(v, s)| v * s).collect();
        for chunk in out.chunks_mut(self.block) {
            butterfly(chunk);
            for v in chunk.iter_mut() {
                *v *= inv;
            }
        }
        out
    }

    /// Inverse (the embedding's): `FWHT_block(x) / sqrt(block)`, then `⊙ s`.
    #[must_use]
    pub fn inverse(&self, x: &[f64]) -> Vec<f64> {
        let signs = self
            .signs
            .get(&x.len())
            .unwrap_or_else(|| panic!("no Hadamard signs for width {}", x.len()));
        let inv = 1.0 / (self.block as f64).sqrt();
        let mut out = x.to_vec();
        for chunk in out.chunks_mut(self.block) {
            butterfly(chunk);
        }
        out.iter_mut()
            .zip(signs)
            .for_each(|(v, s)| *v = *v * inv * s);
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Scalar f64 building blocks
// ─────────────────────────────────────────────────────────────────────────

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// ggml's softplus: `x > 20 ? x : ln(1 + e^x)`.
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `x / sqrt(mean(x²) + eps) ⊙ w`.
fn rms_norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let mean = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// `x / max(||x||, eps)` (ggml `l2_norm`: eps floors the norm).
fn l2_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(eps);
    x.iter().map(|v| v / norm).collect()
}

/// Partial NeoX RoPE: pairs `(j, j + n_rot/2)` for `j < n_rot/2`, angle
/// `pos · base^(-2j/n_rot)`; dims `n_rot..` untouched.
fn rope(x: &mut [f64], pos: usize, n_rot: usize, base: f64) {
    let half = n_rot / 2;
    for j in 0..half {
        let theta = pos as f64 * base.powf(-2.0 * j as f64 / n_rot as f64);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[j], x[j + half]);
        x[j] = a * c - b * s;
        x[j + half] = a * s + b * c;
    }
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Project `xs` through `name`, rotating the inputs first when the matrix
/// is folded (one rotation per activation, computed by the caller).
fn project(
    w: &Weights<'_, '_>,
    fold: Option<&Fold>,
    name: &str,
    plain: &[Vec<f64>],
    rotated: Option<&[Vec<f64>]>,
) -> Vec<Vec<f64>> {
    match (fold, rotated) {
        (Some(f), Some(rot)) if f.is_folded(name) => w.matvec(name, rot),
        _ => w.matvec(name, plain),
    }
}

fn rotate_all(fold: Option<&Fold>, xs: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    fold.map(|f| xs.iter().map(|x| f.rotate(x)).collect())
}

/// The SwiGLU FFN half shared by both layer kinds, applied in place.
fn ffn(w: &Weights<'_, '_>, fold: Option<&Fold>, layer: usize, eps: f64, h: &mut [Vec<f64>]) {
    let norm_w = w.all(&tensor_name(layer, "post_attention_norm.weight"));
    let f: Vec<Vec<f64>> = h.iter().map(|x| rms_norm(x, &norm_w, eps)).collect();
    let f_rot = rotate_all(fold, &f);
    let gate = project(
        w,
        fold,
        &tensor_name(layer, "ffn_gate.weight"),
        &f,
        f_rot.as_deref(),
    );
    let up = project(
        w,
        fold,
        &tensor_name(layer, "ffn_up.weight"),
        &f,
        f_rot.as_deref(),
    );
    let m: Vec<Vec<f64>> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g.iter().zip(u).map(|(a, b)| silu(*a) * b).collect())
        .collect();
    let m_rot = rotate_all(fold, &m);
    let down = project(
        w,
        fold,
        &tensor_name(layer, "ffn_down.weight"),
        &m,
        m_rot.as_deref(),
    );
    for (x, d) in h.iter_mut().zip(&down) {
        *x = add(x, d);
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  The two layer kinds
// ─────────────────────────────────────────────────────────────────────────

/// One full-attention layer over positions `0..xs.len()` (fresh KV).
#[must_use]
pub fn full_layer(
    w: &Weights<'_, '_>,
    cfg: &HybridConfig,
    fold: Option<&Fold>,
    layer: usize,
    xs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let eps = f64::from(cfg.base.rms_norm_eps);
    let (n_head, n_kv, hd) = (
        cfg.base.num_attention_heads,
        cfg.base.num_kv_heads,
        cfg.base.head_dim,
    );
    let n_rot = cfg.rope_dimension_count;
    let base = f64::from(cfg.base.rope_freq_base);
    let group = n_head / n_kv;

    let attn_norm = w.all(&tensor_name(layer, "attn_norm.weight"));
    let a: Vec<Vec<f64>> = xs.iter().map(|x| rms_norm(x, &attn_norm, eps)).collect();
    let a_rot = rotate_all(fold, &a);
    let qg = project(
        w,
        fold,
        &tensor_name(layer, "attn_q.weight"),
        &a,
        a_rot.as_deref(),
    );
    let k = project(
        w,
        fold,
        &tensor_name(layer, "attn_k.weight"),
        &a,
        a_rot.as_deref(),
    );
    let v = project(
        w,
        fold,
        &tensor_name(layer, "attn_v.weight"),
        &a,
        a_rot.as_deref(),
    );
    let q_norm = w.all(&tensor_name(layer, "attn_q_norm.weight"));
    let k_norm = w.all(&tensor_name(layer, "attn_k_norm.weight"));

    // Per position: normed + RoPE'd keys, raw values.
    let keys: Vec<Vec<Vec<f64>>> = (0..xs.len())
        .map(|t| {
            (0..n_kv)
                .map(|kh| {
                    let mut kk = rms_norm(&k[t][kh * hd..(kh + 1) * hd], &k_norm, eps);
                    rope(&mut kk, t, n_rot, base);
                    kk
                })
                .collect()
        })
        .collect();
    let scale = 1.0 / (hd as f64).sqrt();

    let mut out = Vec::with_capacity(xs.len());
    for (t, qg_t) in qg.iter().enumerate() {
        let mut gated = vec![0.0f64; n_head * hd];
        for h in 0..n_head {
            let lo = h * 2 * hd;
            let mut q = rms_norm(&qg_t[lo..lo + hd], &q_norm, eps);
            rope(&mut q, t, n_rot, base);
            let gate = &qg_t[lo + hd..lo + 2 * hd];
            let kh = h / group;
            let scores: Vec<f64> = (0..=t)
                .map(|s| scale * q.iter().zip(&keys[s][kh]).map(|(a, b)| a * b).sum::<f64>())
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..hd {
                let o: f64 = (0..=t).map(|s| weights[s] * v[s][kh * hd + d]).sum::<f64>() / total;
                gated[h * hd + d] = o * sigmoid(gate[d]);
            }
        }
        out.push(gated);
    }
    let out_rot = rotate_all(fold, &out);
    let proj = project(
        w,
        fold,
        &tensor_name(layer, "attn_output.weight"),
        &out,
        out_rot.as_deref(),
    );
    let mut h: Vec<Vec<f64>> = xs.iter().zip(&proj).map(|(x, p)| add(x, p)).collect();
    ffn(w, fold, layer, eps, &mut h);
    h
}

/// One Gated-DeltaNet layer over positions `0..xs.len()` (fresh state).
#[must_use]
pub fn linear_layer(
    w: &Weights<'_, '_>,
    cfg: &HybridConfig,
    fold: Option<&Fold>,
    layer: usize,
    xs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let eps = f64::from(cfg.base.rms_norm_eps);
    let (nk, nv) = (cfg.n_k_heads(), cfg.n_v_heads());
    let (hk, hv) = (cfg.head_k_dim(), cfg.head_v_dim());
    let conv_dim = cfg.conv_dim();
    let kc = cfg.ssm_conv_kernel;
    let rep = nv / nk;
    let tiled = |m: usize| (m % rep) * nk + m / rep;

    let attn_norm = w.all(&tensor_name(layer, "attn_norm.weight"));
    let a: Vec<Vec<f64>> = xs.iter().map(|x| rms_norm(x, &attn_norm, eps)).collect();
    let a_rot = rotate_all(fold, &a);
    let qkv = project(
        w,
        fold,
        &tensor_name(layer, "attn_qkv.weight"),
        &a,
        a_rot.as_deref(),
    );
    let z = project(
        w,
        fold,
        &tensor_name(layer, "attn_gate.weight"),
        &a,
        a_rot.as_deref(),
    );
    // Never folded: they read the UN-rotated activation.
    let alpha = w.matvec(&tensor_name(layer, "ssm_alpha.weight"), &a);
    let beta = w.matvec(&tensor_name(layer, "ssm_beta.weight"), &a);
    let conv_w = w.all(&tensor_name(layer, "ssm_conv1d.weight"));
    let a_neg = w.all(&tensor_name(layer, "ssm_a"));
    let dt_bias = w.all(&tensor_name(layer, "ssm_dt.bias"));
    let ssm_norm = w.all(&tensor_name(layer, "ssm_norm.weight"));

    let mut window = vec![vec![0.0f64; kc - 1]; conv_dim];
    // State per grouped v-head, `[hv][hk]`.
    let mut state = vec![vec![0.0f64; hv * hk]; nv];
    let scale = 1.0 / (hk as f64).sqrt();
    let mut outs = Vec::with_capacity(xs.len());
    for t in 0..xs.len() {
        // Depthwise causal conv (oldest tap first), then SiLU on all channels.
        let y: Vec<f64> = (0..conv_dim)
            .map(|c| {
                let taps = &conv_w[c * kc..(c + 1) * kc];
                let mut acc = qkv[t][c] * taps[kc - 1];
                for i in 0..kc - 1 {
                    acc += window[c][i] * taps[i];
                }
                window[c].rotate_left(1);
                window[c][kc - 2] = qkv[t][c];
                silu(acc)
            })
            .collect();
        let q: Vec<Vec<f64>> = (0..nk)
            .map(|h| l2_norm(&y[h * hk..(h + 1) * hk], eps))
            .collect();
        let k: Vec<Vec<f64>> = (0..nk)
            .map(|h| l2_norm(&y[(nk + h) * hk..(nk + h + 1) * hk], eps))
            .collect();
        let v_base = 2 * nk * hk;
        let mut gated = vec![0.0f64; nv * hv];
        for (m, s) in state.iter_mut().enumerate() {
            let tj = tiled(m);
            let kh = m / rep;
            let decay = (a_neg[tj] * softplus(alpha[t][tj] + dt_bias[tj])).exp();
            let b = sigmoid(beta[t][tj]);
            let v = &y[v_base + tj * hv..v_base + (tj + 1) * hv];
            let mut o = vec![0.0f64; hv];
            for j in 0..hv {
                let row = &mut s[j * hk..(j + 1) * hk];
                for x in row.iter_mut() {
                    *x *= decay;
                }
                let kv: f64 = row.iter().zip(&k[kh]).map(|(a, b)| a * b).sum();
                let delta = (v[j] - kv) * b;
                for (x, kk) in row.iter_mut().zip(&k[kh]) {
                    *x += kk * delta;
                }
                o[j] = row.iter().zip(&q[kh]).map(|(a, b)| a * b).sum::<f64>() * scale;
            }
            // Gated RMSNorm with the TILED z of the same head.
            let zh = &z[t][tj * hv..(tj + 1) * hv];
            let mean = o.iter().map(|x| x * x).sum::<f64>() / hv as f64;
            let inv = 1.0 / (mean + eps).sqrt();
            for j in 0..hv {
                gated[m * hv + j] = o[j] * inv * ssm_norm[j] * silu(zh[j]);
            }
        }
        outs.push(gated);
    }
    let outs_rot = rotate_all(fold, &outs);
    let proj = project(
        w,
        fold,
        &tensor_name(layer, "ssm_out.weight"),
        &outs,
        outs_rot.as_deref(),
    );
    let mut h: Vec<Vec<f64>> = xs.iter().zip(&proj).map(|(x, p)| add(x, p)).collect();
    ffn(w, fold, layer, eps, &mut h);
    h
}

/// The layer `layer` of either kind.
#[must_use]
pub fn layer(
    w: &Weights<'_, '_>,
    cfg: &HybridConfig,
    fold: Option<&Fold>,
    layer: usize,
    xs: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    if cfg.is_full_attention(layer) {
        full_layer(w, cfg, fold, layer, xs)
    } else {
        linear_layer(w, cfg, fold, layer, xs)
    }
}

/// The embedding rows of `tokens` (inverse-rotated when the file folds
/// `token_embd.weight`).
#[must_use]
pub fn embed(w: &Weights<'_, '_>, fold: Option<&Fold>, tokens: &[u32]) -> Vec<Vec<f64>> {
    tokens
        .iter()
        .map(|&t| {
            let row = w.row(
                "token_embd.weight",
                usize::try_from(t).expect("token fits usize"),
            );
            match fold {
                Some(f) if f.config.is_inverse("token_embd.weight") => f.inverse(&row),
                _ => row,
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────
//  Comparison metrics
// ─────────────────────────────────────────────────────────────────────────

/// How an `f32` activation block compares with its `f64` reference.
#[derive(Debug, Clone, Copy)]
pub struct Agreement {
    /// Cosine over the whole `[t × hidden]` block.
    pub cos: f64,
    /// Smallest per-position cosine.
    pub min_row_cos: f64,
    /// Cosine of the layer's *contribution* (`output - input`), which the
    /// residual pass-through cannot inflate.
    pub delta_cos: f64,
    /// `max |a - b|`.
    pub max_abs: f64,
    /// `max |b|` (the reference's scale).
    pub ref_max: f64,
    /// `||a - b|| / ||b||`.
    pub rel_l2: f64,
}

fn cosine(a: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Compare the `f32` rows `ours` with the reference `want`, both being the
/// outputs of a layer whose input was `input`.
#[must_use]
pub fn agreement(ours: &[Vec<f32>], want: &[Vec<f64>], input: &[Vec<f64>]) -> Agreement {
    let pairs = || {
        ours.iter()
            .zip(want)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (f64::from(*x), *y)))
    };
    let cos = cosine(pairs());
    let min_row_cos = ours
        .iter()
        .zip(want)
        .map(|(a, b)| cosine(a.iter().zip(b).map(|(x, y)| (f64::from(*x), *y))))
        .fold(f64::INFINITY, f64::min);
    let delta_cos = cosine(ours.iter().zip(want).zip(input).flat_map(|((a, b), x)| {
        a.iter()
            .zip(b)
            .zip(x)
            .map(|((p, q), r)| (f64::from(*p) - r, q - r))
    }));
    let (mut max_abs, mut ref_max, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in pairs() {
        max_abs = max_abs.max((x - y).abs());
        ref_max = ref_max.max(y.abs());
        num += (x - y) * (x - y);
        den += y * y;
    }
    Agreement {
        cos,
        min_row_cos,
        delta_cos,
        max_abs,
        ref_max,
        rel_l2: if den == 0.0 {
            num.sqrt()
        } else {
            (num / den).sqrt()
        },
    }
}
