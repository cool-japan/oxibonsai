//! Numeric primitives for the FLUX.2 Klein DiT forward pass.
//!
//! Everything here operates on flat `Vec<f32>` / `&[f32]` buffers with explicit
//! shapes (no `ndarray`), matching the convention of the rest of `oxibonsai`.
//! The reference (golden) megakernel computes blocks in bf16; this crate
//! computes in f32 throughout and is validated against the bf16 goldens with a
//! tolerance (see the parity example).
//!
//! Conventions:
//! - A `Linear[out, in]` weight is row-major `w[n * in + k]` (= `w[n][k]`), and
//!   `dense_matmul` computes `out[m, n] = Σ_k in[m, k] * w[n, k]` — the same
//!   contraction as the ternary GEMM, so the two linear flavours are
//!   interchangeable at call sites.
//! - LayerNorm is **affine = false** (mean-centred, no weight/bias), eps = 1e-6.
//! - QK-RMSNorm uses a per-head weight over `head_dim`, eps = 1e-6.
//! - RoPE is interleaved (adjacent pairs), 4-axis, theta = 2000.

use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;
use oxibonsai_kernels::softmax_simd;

use crate::error::{DitError, DitResult};
use crate::gemm::gemm_abt;

/// Dequantise a ternary weight to a row-major `[out, in]` f32 buffer.
///
/// The ternary blocks are out-major (`n` rows × `k/128` blocks each), so the
/// dequantised buffer is exactly the row-major `[out, in]` matrix the SIMD f32
/// [`gemm_abt`] consumes. This is mathematically exact (ternary code × per-block
/// fp16 scale, identical coding to the kernels' GEMV), so it does not change
/// results — only the throughput (the per-call bit-unpacking ternary GEMV in
/// `oxibonsai-kernels` is ~0.2 GMAC/s; dequant + [`gemm_abt`] is ~30–60 GMAC/s).
///
/// Done per call (not cached) to bound peak memory: an f32 cache of all ternary
/// weights would be ~15 GB, so each weight is dequantised, used, and freed.
fn dequant_weight(blocks: &[BlockTQ2_0_g128], n: usize, k: usize) -> DitResult<Vec<f32>> {
    let mut buf = vec![0.0f32; n * k];
    BlockTQ2_0_g128::dequant(blocks, &mut buf).map_err(DitError::Gguf)?;
    Ok(buf)
}

/// Dense f32 matmul for a bf16-decoded `Linear[out, in]` (bias = false).
///
/// `out[m, n] = Σ_k input[m, k] * weight[n, k]`.
///
/// - `input`: row-major `[m, k]`.
/// - `weight`: row-major `[n, k]` (the `to_f32_vec()` of a `[out, in]` tensor).
/// - returns row-major `[m, n]`.
///
/// # Errors
/// [`DitError::Shape`] if any buffer length disagrees with `m`, `n`, `k`.
pub fn dense_matmul(
    input: &[f32],
    weight: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> DitResult<Vec<f32>> {
    if input.len() != m * k {
        return Err(DitError::Shape(format!(
            "dense_matmul input len {} != m*k {}",
            input.len(),
            m * k
        )));
    }
    if weight.len() != n * k {
        return Err(DitError::Shape(format!(
            "dense_matmul weight len {} != n*k {}",
            weight.len(),
            n * k
        )));
    }
    let mut out = vec![0.0f32; m * n];
    // GPU-first dense f32 GEMM (native-cuda; same `OXI_DIT_GPU` gate as the
    // ternary path). The DiT stage0 `context_embedder` [512×7680→3072] dominates
    // the DiT wall on the CPU; routing the dense Linears through `encode_gemm_f32`
    // is the largest stage0 win. Falls back to the CPU SIMD gemm on any GPU error.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        if crate::cuda_gpu::dit_gpu_enabled()
            && crate::cuda_gpu::dense_matmul_gpu(weight, input, &mut out, m, n, k).is_ok()
        {
            return Ok(out);
        }
    }
    gemm_abt(input, weight, &mut out, m, n, k);
    Ok(out)
}

/// Run `body(row, &mut out[row])` for every one of `rows` length-`width` output
/// rows, split across the available CPUs via scoped threads. Falls back to a
/// serial loop for tiny problems (where thread spawn cost dominates).
fn par_rows<F>(out: &mut [f32], rows: usize, width: usize, body: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    debug_assert_eq!(out.len(), rows * width);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows.max(1));
    if threads <= 1 || rows < 8 {
        for (r, chunk) in out.chunks_mut(width).enumerate() {
            body(r, chunk);
        }
        return;
    }
    // Even chunk of rows per thread (last takes the remainder).
    let per = rows.div_ceil(threads);
    let body_ref = &body;
    std::thread::scope(|scope| {
        let mut base = 0usize;
        for chunk in out.chunks_mut(per * width) {
            let start = base;
            let chunk_rows = chunk.len() / width;
            base += chunk_rows;
            scope.spawn(move || {
                for r in 0..chunk_rows {
                    let row = &mut chunk[r * width..(r + 1) * width];
                    body_ref(start + r, row);
                }
            });
        }
    });
}

/// Ternary (`TQ2_0_g128`) matmul wrapper with the same contraction semantics as
/// [`dense_matmul`]: `out[m, n] = Σ_k input[m, k] * w[n, k]`.
///
/// - `blocks`: out-major ternary blocks (`n * (k / 128)` of them).
/// - returns row-major `[m, n]`.
///
/// # Errors
/// Propagates the kernel error (e.g. `k % 128 != 0`) as [`DitError::Kernel`].
pub fn ternary_matmul(
    blocks: &[BlockTQ2_0_g128],
    input: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> DitResult<Vec<f32>> {
    if input.len() != m * k {
        return Err(DitError::Shape(format!(
            "ternary_matmul input len {} != m*k {}",
            input.len(),
            m * k
        )));
    }
    if k % 128 != 0 {
        return Err(DitError::Shape(format!(
            "ternary_matmul k {k} not a multiple of 128"
        )));
    }
    let mut out = vec![0.0f32; m * n];
    // GPU-first: route through the fused Metal TQ2 GEMM when the `metal` feature
    // is compiled (macOS) and not disabled via `OXI_DIT_GPU=0`. On any GPU error
    // we fall through to the CPU path below — never panic. The kernel computes
    // the identical `out[m,n] = Σ_k input[m,k]·dequant(W)[n,k]` contraction.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if crate::gpu::dit_gpu_enabled() {
            match crate::gpu::ternary_matmul_gpu(blocks, input, &mut out, m, n, k) {
                Ok(()) => return Ok(out),
                Err(_e) => {
                    // Fall through to the CPU path. `out` is reused (overwritten
                    // in full by `gemm_abt`), so a partial GPU write is harmless.
                }
            }
        }
    }
    // CUDA sibling of the Metal block above (target_os-disjoint: Linux/Windows).
    // Same `OXI_DIT_GPU` toggle, same identical contraction; on any GPU error we
    // fall through to the CPU path below — never panic.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        if crate::cuda_gpu::dit_gpu_enabled() {
            match crate::cuda_gpu::ternary_matmul_gpu(blocks, input, &mut out, m, n, k) {
                Ok(()) => return Ok(out),
                Err(_e) => {
                    // Fall through to the CPU path. `out` is reused (overwritten
                    // in full by `gemm_abt`), so a partial GPU write is harmless.
                }
            }
        }
    }
    // ── CPU path: dequantise the weight to f32, then run the fast SIMD GEMM. ──
    let weight = dequant_weight(blocks, n, k)?;
    gemm_abt(input, &weight, &mut out, m, n, k);
    Ok(out)
}

/// In-place LayerNorm with **affine = false** (mean-centred), eps as given,
/// applied independently to every length-`dim` row of `x` (`[rows, dim]`).
///
/// `y = (x - mean) / sqrt(var + eps)`, `var` = population variance (`/ dim`).
pub fn layer_norm_inplace(x: &mut [f32], rows: usize, dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), rows * dim);
    let inv_dim = 1.0f32 / dim as f32;
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let mut mean = 0.0f32;
        for &v in row.iter() {
            mean += v;
        }
        mean *= inv_dim;
        let mut var = 0.0f32;
        for &v in row.iter() {
            let d = v - mean;
            var += d * d;
        }
        var *= inv_dim;
        let inv_std = 1.0f32 / (var + eps).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * inv_std;
        }
    }
}

/// Apply modulation `y = (1 + scale) * LN(x) + shift` to `x` `[rows, dim]`,
/// where `scale`/`shift` are length-`dim` (broadcast over rows). `x` must
/// already be LayerNormed (affine = false). Done in-place.
pub fn modulate_inplace(x: &mut [f32], rows: usize, dim: usize, shift: &[f32], scale: &[f32]) {
    debug_assert_eq!(scale.len(), dim);
    debug_assert_eq!(shift.len(), dim);
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        for (i, v) in row.iter_mut().enumerate() {
            *v = (1.0 + scale[i]) * *v + shift[i];
        }
    }
}

/// Per-head QK-RMSNorm: normalise every contiguous `head_dim` chunk of `x`
/// (`[rows, head_dim]`, where `rows = num_heads * seq`) by its own RMS and
/// scale by the shared per-head `weight` (length `head_dim`), eps as given.
///
/// `y = weight * x / sqrt(mean(x^2) + eps)`.
pub fn rms_norm_heads_inplace(
    x: &mut [f32],
    rows: usize,
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) {
    debug_assert_eq!(weight.len(), head_dim);
    let inv_dim = 1.0f32 / head_dim as f32;
    for r in 0..rows {
        let row = &mut x[r * head_dim..(r + 1) * head_dim];
        let mut ms = 0.0f32;
        for &v in row.iter() {
            ms += v * v;
        }
        ms *= inv_dim;
        let inv_rms = 1.0f32 / (ms + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = weight[i] * *v * inv_rms;
        }
    }
}

/// SiLU activation: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SiLU element-wise, in-place.
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// SwiGLU over a `[rows, 2 * half]` buffer: for each row split into
/// `(gate, up)` halves and return `[rows, half]` = `silu(gate) * up`.
pub fn swiglu(x: &[f32], rows: usize, half: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * 2 * half);
    let full = 2 * half;
    let mut out = vec![0.0f32; rows * half];
    for r in 0..rows {
        let src = &x[r * full..r * full + full];
        let dst = &mut out[r * half..(r + 1) * half];
        let (gate, up) = src.split_at(half);
        for i in 0..half {
            dst[i] = silu(gate[i]) * up[i];
        }
    }
    out
}

/// The interleaved RoPE cos/sin tables for a joint sequence.
///
/// `cos`/`sin` are each `[seq, half]` where `half = rope_dim / 2` (here 64).
/// Built from per-axis frequencies (see [`build_rope_tables`]).
#[derive(Debug, Clone)]
pub struct RopeTables {
    /// `[seq, half]` cos table.
    pub cos: Vec<f32>,
    /// `[seq, half]` sin table.
    pub sin: Vec<f32>,
    /// Sequence length.
    pub seq: usize,
    /// Half the rope dim (number of rotated pairs per token).
    pub half: usize,
}

/// Build the 4-axis interleaved RoPE tables from position ids.
///
/// - `ids`: row-major `[seq, num_axes]` position ids (cast to f32).
/// - `axes_dims`: per-axis rope dim (each even); Σ = rope_dim (here 128).
/// - `theta`: rope base (here 2000).
///
/// For axis `a` with dim `d_a`, there are `d_a / 2` frequencies
/// `omega_i = theta^(-(2 i) / d_a)`, and `freq[token, off + i] = ids[token, a] *
/// omega_i`, concatenated across axes in order. `cos`/`sin` are `[seq, half]`.
pub fn build_rope_tables(
    ids: &[f32],
    seq: usize,
    num_axes: usize,
    axes_dims: &[u32],
    theta: f32,
) -> DitResult<RopeTables> {
    if ids.len() != seq * num_axes {
        return Err(DitError::Shape(format!(
            "build_rope_tables ids len {} != seq*num_axes {}",
            ids.len(),
            seq * num_axes
        )));
    }
    if axes_dims.len() != num_axes {
        return Err(DitError::Shape(format!(
            "build_rope_tables axes_dims len {} != num_axes {}",
            axes_dims.len(),
            num_axes
        )));
    }
    let half: usize = axes_dims.iter().map(|&d| (d / 2) as usize).sum();
    // Precompute per-axis omega tables.
    let mut omegas: Vec<Vec<f32>> = Vec::with_capacity(num_axes);
    for &dim in axes_dims {
        let pairs = (dim / 2) as usize;
        let mut om = Vec::with_capacity(pairs);
        for i in 0..pairs {
            // scale = (2 i) / dim ; omega = 1 / theta^scale = theta^(-scale)
            let scale = (2 * i) as f32 / dim as f32;
            om.push(theta.powf(-scale));
        }
        omegas.push(om);
    }
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for t in 0..seq {
        let mut off = 0usize;
        for (a, om) in omegas.iter().enumerate() {
            let pos = ids[t * num_axes + a];
            for (i, &omega) in om.iter().enumerate() {
                let angle = pos * omega;
                cos[t * half + off + i] = angle.cos();
                sin[t * half + off + i] = angle.sin();
            }
            off += om.len();
        }
    }
    Ok(RopeTables {
        cos,
        sin,
        seq,
        half,
    })
}

/// Apply interleaved RoPE in-place to a `[num_heads, seq, head_dim]` buffer.
///
/// Pairs are **adjacent**: `(x[2 i], x[2 i + 1])` rotate by frequency `i`,
/// using `cos`/`sin` `[seq, half]` (broadcast over heads). `head_dim = 2 *
/// half`.
///
/// ```text
/// out0 = real * cos - imag * sin
/// out1 = imag * cos + real * sin
/// ```
pub fn apply_rope_inplace(
    x: &mut [f32],
    num_heads: usize,
    seq: usize,
    head_dim: usize,
    rope: &RopeTables,
) -> DitResult<()> {
    if rope.seq != seq {
        return Err(DitError::Shape(format!(
            "apply_rope seq {} != rope.seq {}",
            seq, rope.seq
        )));
    }
    if rope.half * 2 != head_dim {
        return Err(DitError::Shape(format!(
            "apply_rope head_dim {} != 2*rope.half {}",
            head_dim,
            2 * rope.half
        )));
    }
    let half = rope.half;
    for h in 0..num_heads {
        for t in 0..seq {
            let base = (h * seq + t) * head_dim;
            let crow = &rope.cos[t * half..(t + 1) * half];
            let srow = &rope.sin[t * half..(t + 1) * half];
            let row = &mut x[base..base + head_dim];
            for i in 0..half {
                let real = row[2 * i];
                let imag = row[2 * i + 1];
                let c = crow[i];
                let s = srow[i];
                row[2 * i] = real * c - imag * s;
                row[2 * i + 1] = imag * c + real * s;
            }
        }
    }
    Ok(())
}

/// Multi-head scaled-dot-product attention over a joint sequence.
///
/// `q`, `k`, `v` are `[num_heads, seq, head_dim]` (head-major). Returns the
/// attention output reassembled to `[seq, num_heads * head_dim]` (token-major,
/// heads concatenated along the feature axis — the transpose+reshape the spec
/// performs after SDPA). `scale = 1 / sqrt(head_dim)`, softmax in f32 over keys.
pub fn joint_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq: usize,
    head_dim: usize,
) -> DitResult<Vec<f32>> {
    let expect = num_heads * seq * head_dim;
    if q.len() != expect || k.len() != expect || v.len() != expect {
        return Err(DitError::Shape(format!(
            "joint_attention q/k/v len mismatch (expect {expect})"
        )));
    }
    // GPU-first: route through the fused Metal flash-attention kernel when the
    // `metal` feature is compiled (macOS) and explicitly enabled via
    // `OXI_DIT_ATTN_GPU=1`. On any GPU error we fall through to the CPU path
    // below — never panic. The kernel matches this reference exactly: head-major
    // q/k/v in, token-major `[seq, num_heads*head_dim]` out, non-causal softmax,
    // `scale = 1/sqrt(head_dim)`.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if crate::gpu::dit_attn_gpu_enabled() {
            match crate::gpu::joint_attention_gpu(q, k, v, num_heads, seq, head_dim) {
                Ok(out) => return Ok(out),
                Err(_e) => {
                    // Fall through to the CPU path (silent fallback / reference).
                }
            }
        }
    }
    // CUDA sibling of the Metal block above (target_os-disjoint: Linux/Windows).
    // Same `OXI_DIT_ATTN_GPU` toggle, same head-major→token-major contract; on
    // any GPU error we fall through to the CPU path below — never panic.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        if crate::cuda_gpu::dit_attn_gpu_enabled() {
            match crate::cuda_gpu::joint_attention_gpu(q, k, v, num_heads, seq, head_dim) {
                Ok(out) => return Ok(out),
                Err(_e) => {
                    // Fall through to the CPU path (silent fallback / reference).
                }
            }
        }
    }
    let inner = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // Per-head output stored head-major [num_heads, seq, head_dim], computed in
    // parallel (each head is independent), then transposed to [seq, inner].
    let mut head_out = vec![0.0f32; num_heads * seq * head_dim];
    let attend_head = |h: usize, dst: &mut [f32]| {
        let head_off = h * seq * head_dim;
        let mut scores = vec![0.0f32; seq];
        for qi in 0..seq {
            let q_row = &q[head_off + qi * head_dim..head_off + (qi + 1) * head_dim];
            for (ki, score) in scores.iter_mut().enumerate() {
                let k_row = &k[head_off + ki * head_dim..head_off + (ki + 1) * head_dim];
                *score = crate::gemm::dot(q_row, k_row, head_dim) * scale;
            }
            softmax_simd(&mut scores);
            let o = &mut dst[qi * head_dim..(qi + 1) * head_dim];
            for d in o.iter_mut() {
                *d = 0.0;
            }
            for (ki, &w) in scores.iter().enumerate() {
                let v_row = &v[head_off + ki * head_dim..head_off + (ki + 1) * head_dim];
                for d in 0..head_dim {
                    o[d] += w * v_row[d];
                }
            }
        }
    };
    par_rows(&mut head_out, num_heads, seq * head_dim, attend_head);
    // Transpose [num_heads, seq, head_dim] → [seq, num_heads*head_dim].
    let mut out = vec![0.0f32; seq * inner];
    for h in 0..num_heads {
        for qi in 0..seq {
            let src = &head_out[(h * seq + qi) * head_dim..(h * seq + qi + 1) * head_dim];
            let dst = &mut out[qi * inner + h * head_dim..qi * inner + (h + 1) * head_dim];
            dst.copy_from_slice(src);
        }
    }
    Ok(out)
}

/// Reshape `[seq, inner]` token-major (inner = heads*head_dim) into
/// `[num_heads, seq, head_dim]` head-major, the layout attention/RoPE expect.
pub fn to_heads(x: &[f32], seq: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
    let inner = num_heads * head_dim;
    debug_assert_eq!(x.len(), seq * inner);
    let mut out = vec![0.0f32; seq * inner];
    for t in 0..seq {
        for h in 0..num_heads {
            let src = &x[t * inner + h * head_dim..t * inner + (h + 1) * head_dim];
            let dst = &mut out[(h * seq + t) * head_dim..(h * seq + t + 1) * head_dim];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// Sinusoidal timestep embedding (`flip_sin_to_cos = true`), returning `[dim]`.
///
/// `half = dim / 2`; `freqs_i = exp(-ln(10000) * i / half)` for `i` in
/// `0..half`; `args_i = t * freqs_i`; `emb = concat([sin(args), cos(args)])`;
/// flipped to `concat([cos(args), sin(args)])`.
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let log10000 = (10000.0f32).ln();
    let mut emb = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (-log10000 * i as f32 / half as f32).exp();
        let arg = t * freq;
        // flipped: [cos(args) (half), sin(args) (half)]
        emb[i] = arg.cos();
        emb[half + i] = arg.sin();
    }
    emb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        layer_norm_inplace(&mut x, 1, 4, 1e-6);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn dense_matmul_identity() {
        // input [2,3], weight = identity-ish [3,3] -> out == input
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut weight = vec![0.0f32; 9];
        for i in 0..3 {
            weight[i * 3 + i] = 1.0;
        }
        let out = dense_matmul(&input, &weight, 2, 3, 3).expect("matmul");
        assert_eq!(out, input);
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        // pos=0 everywhere -> cos=1,sin=0 -> rope is identity.
        let ids = vec![0.0f32; 4];
        let tables = build_rope_tables(&ids, 1, 4, &[32, 32, 32, 32], 2000.0).expect("rope");
        assert_eq!(tables.half, 64);
        assert!(tables.cos.iter().all(|&c| (c - 1.0).abs() < 1e-9));
        assert!(tables.sin.iter().all(|&s| s.abs() < 1e-9));
        let mut x: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let orig = x.clone();
        apply_rope_inplace(&mut x, 1, 1, 128, &tables).expect("apply");
        assert_eq!(x, orig);
    }

    #[test]
    fn swiglu_matches_manual() {
        let x = vec![1.0, -1.0, 2.0, 3.0]; // gate=[1,-1], up=[2,3]
        let out = swiglu(&x, 1, 2);
        assert!((out[0] - silu(1.0) * 2.0).abs() < 1e-6);
        assert!((out[1] - silu(-1.0) * 3.0).abs() < 1e-6);
    }
}
