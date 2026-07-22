//! Pure-Rust Qwen3-4B **encoder** forward pass (single full causal pass, no KV
//! cache, no generation) capturing per-layer hidden states.
//!
//! Mirrors the reference `Qwen3TextEncoder.__call__` (mflux):
//! - `embed_tokens` lookup → `hidden_states_list[0]` (no norm).
//! - 36 decoder layers, each:
//!   `h = h + o_proj(attn(qk_norm(rope(q,k)), v))` over `input_layernorm(h)`,
//!   `h = h + down_proj(silu(gate_proj(x)) * up_proj(x))` over
//!   `post_attention_layernorm(h)` — appended to `hidden_states_list`.
//! - GQA: 32 query heads / 8 kv heads (each kv head shared by 4 query heads),
//!   head_dim 128, QK-RMSNorm (eps 1e-6) on q and k per head, RoPE θ=1e6.
//! - attention mask = causal (lower-triangular) + padding (`mask==0 → -inf`),
//!   SDPA in f32 with scale `1/sqrt(head_dim)`.
//!
//! Everything runs in f32 on flat `Vec<f32>` buffers (batch 1). The matmuls
//! reuse the shared SIMD GEMM [`crate::gemm::gemm_abt`] (`x · Wᵀ`); the per-head
//! dot products reuse [`crate::gemm::dot`]; softmax reuses
//! [`oxibonsai_kernels::softmax_simd`]; SiLU reuses [`crate::math::silu`].

use half::bf16;
use oxibonsai_kernels::softmax_simd;

use crate::gemm::{dot, gemm_abt};
use crate::math::silu;
use crate::te::config::{TeConfig, STACK_LAYERS};
use crate::te::error::{TeError, TeResult};
use crate::te::rope::Qwen3Rope;
use crate::te::weights::TeWeights;

/// Numeric precision policy for the encoder forward.
///
/// The reference MLX model stores every residual-stream tensor (and every
/// `nn.Linear` / norm / RoPE / SDPA output) in **bf16**, upcasting only inside
/// reductions (RMSNorm variance, SDPA). [`Precision::Bf16Storage`] reproduces
/// that "bf16 storage, f32 reductions" behaviour by rounding to bf16 at each of
/// those boundaries, which tracks the goldens to the deepest layers.
/// [`Precision::F32`] keeps everything in f32 (faster, slightly looser on the
/// deepest pre-final-norm layers but still ≥ 0.999 cosine on the stacked cond).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Pure f32 throughout.
    F32,
    /// Emulate MLX bf16 storage (round to bf16 at op boundaries; reduce in f32).
    Bf16Storage,
}

/// Round every element of `x` to bf16 and back to f32 (MLX `astype(bf16)`).
/// Parallelised across CPUs (this is the TE's most-called element-wise op —
/// ~18×/layer over arrays up to `[512, 9728]`); each element is independent so
/// the result is bit-identical to the serial loop.
fn round_bf16(x: &mut [f32]) {
    par_flat(x, 1 << 16, |chunk| {
        for v in chunk.iter_mut() {
            *v = bf16::from_f32(*v).to_f32();
        }
    });
}

/// Apply `body` to ~`min(cpus, len/min_chunk)` contiguous chunks of `data` in
/// parallel via scoped threads (serial for small inputs / a single core). Valid
/// only for element-wise work (each chunk is processed independently). Uses
/// std threads (not rayon) to stay `wasm`-compatible, matching [`par_heads`].
fn par_flat<F>(data: &mut [f32], min_chunk: usize, body: F)
where
    F: Fn(&mut [f32]) + Sync,
{
    let n = data.len();
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1);
    if threads <= 1 || n < 2 * min_chunk {
        body(data);
        return;
    }
    let nthreads = threads.min(n / min_chunk).max(1);
    let per = n.div_ceil(nthreads);
    let body_ref = &body;
    std::thread::scope(|scope| {
        for chunk in data.chunks_mut(per) {
            scope.spawn(move || body_ref(chunk));
        }
    });
}

/// The captured output of a Qwen3 encoder forward pass.
pub struct TeOutput {
    /// `hidden_states_list[0..=num_layers]` — index 0 is the token embeddings,
    /// index `i` (`1..=num_layers`) is the residual-stream output of decoder
    /// layer `i-1`, all **pre** the final RMSNorm. Each is `[seq, hidden]`.
    pub hidden_states: Vec<Vec<f32>>,
    /// Sequence length.
    pub seq: usize,
    /// Hidden size.
    pub hidden: usize,
}

impl TeOutput {
    /// The `[seq, 3*hidden]` conditioning: per position, the layer
    /// [`STACK_LAYERS`] hidden states concatenated along the feature axis
    /// (`stack(axis=1) → transpose(0,2,1,3) → reshape`).
    ///
    /// # Errors
    /// [`TeError::Shape`] if a stacked layer index is out of range.
    pub fn cond_7680(&self) -> TeResult<Vec<f32>> {
        let seq = self.seq;
        let hidden = self.hidden;
        for &l in &STACK_LAYERS {
            if l >= self.hidden_states.len() {
                return Err(TeError::Shape(format!(
                    "stack layer {l} out of range (have {})",
                    self.hidden_states.len()
                )));
            }
        }
        let groups = STACK_LAYERS.len();
        let mut out = vec![0.0f32; seq * groups * hidden];
        for t in 0..seq {
            for (g, &l) in STACK_LAYERS.iter().enumerate() {
                let src = &self.hidden_states[l][t * hidden..(t + 1) * hidden];
                let dst_base = t * groups * hidden + g * hidden;
                out[dst_base..dst_base + hidden].copy_from_slice(src);
            }
        }
        Ok(out)
    }
}

/// A Qwen3-4B encoder bound to a loaded weight registry.
pub struct TextEncoder<'w> {
    weights: &'w TeWeights,
    cfg: TeConfig,
    precision: Precision,
}

impl<'w> TextEncoder<'w> {
    /// Bind to a weight registry (cloning the small config), defaulting to
    /// [`Precision::F32`] — the spec-mandated pure-f32 path, which matches the
    /// stacked cond (the DiT input) at cosine ≥ 0.999.
    pub fn new(weights: &'w TeWeights) -> Self {
        Self::with_precision(weights, Precision::F32)
    }

    /// Bind with an explicit [`Precision`] policy.
    pub fn with_precision(weights: &'w TeWeights, precision: Precision) -> Self {
        let cfg = weights.config().clone();
        Self {
            weights,
            cfg,
            precision,
        }
    }

    /// The encoder configuration.
    pub fn config(&self) -> &TeConfig {
        &self.cfg
    }

    /// The active precision policy.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Round in-place to bf16 iff the bf16-storage policy is active.
    #[inline]
    fn quantize(&self, x: &mut [f32]) {
        if self.precision == Precision::Bf16Storage {
            round_bf16(x);
        }
    }

    /// Run the full causal forward over `input_ids` (length `seq`) with the
    /// 0/1 `attention_mask` (length `seq`), capturing all hidden states.
    ///
    /// # Errors
    /// [`TeError::Shape`] on a length mismatch, or a weight-loading error.
    pub fn forward(&self, input_ids: &[u32], attention_mask: &[i32]) -> TeResult<TeOutput> {
        let seq = input_ids.len();
        if attention_mask.len() != seq {
            return Err(TeError::Shape(format!(
                "attention_mask len {} != input_ids len {seq}",
                attention_mask.len()
            )));
        }
        let hidden = self.cfg.hidden_size;
        let head_dim = self.cfg.head_dim;
        let n_q = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let eps = self.cfg.rms_norm_eps;

        // ── embed_tokens lookup → hidden[0] ──
        // Gather only the `seq` token rows of the `[vocab, hidden]` table.
        // (For the 4-bit source this dequantises just those rows — no 1.5 GB
        // full-table f32 spike — yet the values are byte-identical to a full
        // dequant-then-gather; the f32 `.npy` source still full-loads + gathers.
        // The token-id bounds check `id < vocab` lives inside `embed_gather`.)
        let mut h = self
            .weights
            .embed_gather("embed_tokens", input_ids, hidden)?;
        // MLX embed_tokens output is bf16; the dequant rows are already
        // bf16-valued, but round for exactness under the bf16-storage policy.
        self.quantize(&mut h);

        let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(self.cfg.num_layers + 1);
        hidden_states.push(h.clone());

        // ── RoPE tables + additive attention mask ──
        let rope = Qwen3Rope::new(seq, head_dim, self.cfg.rope_theta);
        let mask = build_mask(attention_mask, seq);

        // ── decoder layers ──
        let timing = std::env::var("OXI_IMAGE_TIMING").is_ok();
        TE_MATMUL_NS.store(0, std::sync::atomic::Ordering::Relaxed);
        TE_ATTN_NS.store(0, std::sync::atomic::Ordering::Relaxed);
        for layer in 0..self.cfg.num_layers {
            self.decoder_layer(
                &mut h, layer, seq, &rope, &mask, eps, hidden, head_dim, n_q, n_kv,
            )?;
            hidden_states.push(h.clone());
        }
        if timing {
            let mm = TE_MATMUL_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
            let at = TE_ATTN_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
            eprintln!(
                "[timing]   TE matmul(GPU)={mm:.2}s  attention(CPU)={at:.2}s  \
                 (remainder = norms/rope/quantize/reshape/embed)"
            );
        }

        Ok(TeOutput {
            hidden_states,
            seq,
            hidden,
        })
    }

    /// One Qwen3 decoder layer, applied in-place to the residual stream `h`.
    #[allow(clippy::too_many_arguments)]
    fn decoder_layer(
        &self,
        h: &mut [f32],
        layer: usize,
        seq: usize,
        rope: &Qwen3Rope,
        mask: &[f32],
        eps: f32,
        hidden: usize,
        head_dim: usize,
        n_q: usize,
        n_kv: usize,
    ) -> TeResult<()> {
        let pfx = format!("layers.{layer}");
        let q_dim = self.cfg.q_dim();
        let kv_dim = self.cfg.kv_dim();

        // ---- self-attention block ----
        // input_layernorm: f32 reduction, bf16-valued output (RMSNorm casts back).
        let in_ln = self
            .weights
            .vec1(&format!("{pfx}.input_layernorm"), hidden)?;
        let mut x = rms_norm(h, seq, hidden, &in_ln.data, eps);
        self.quantize(&mut x);

        let wq = self
            .weights
            .linear(&format!("{pfx}.self_attn.q_proj"), q_dim, hidden)?;
        let wk = self
            .weights
            .linear(&format!("{pfx}.self_attn.k_proj"), kv_dim, hidden)?;
        let wv = self
            .weights
            .linear(&format!("{pfx}.self_attn.v_proj"), kv_dim, hidden)?;
        // [seq, q_dim] / [seq, kv_dim], token-major (heads concatenated).
        // nn.Linear outputs bf16.
        let resident = self.weights.is_resident();
        let mut q = matmul(&x, &wq.data, seq, q_dim, hidden, resident)?;
        let mut k = matmul(&x, &wk.data, seq, kv_dim, hidden, resident)?;
        let mut v = matmul(&x, &wv.data, seq, kv_dim, hidden, resident)?;
        self.quantize(&mut q);
        self.quantize(&mut k);
        self.quantize(&mut v);

        // QK-RMSNorm per head over head_dim (applied while q/k are still
        // [seq, n_heads, head_dim] = token-major: each contiguous head_dim chunk
        // is one (token, head) vector). f32 reduction, bf16-valued output.
        let q_norm = self
            .weights
            .vec1(&format!("{pfx}.self_attn.q_norm"), head_dim)?;
        let k_norm = self
            .weights
            .vec1(&format!("{pfx}.self_attn.k_norm"), head_dim)?;
        rms_norm_heads(&mut q, seq * n_q, head_dim, &q_norm.data, eps);
        rms_norm_heads(&mut k, seq * n_kv, head_dim, &k_norm.data, eps);
        self.quantize(&mut q);
        self.quantize(&mut k);

        // To head-major [n_heads, seq, head_dim] for RoPE + attention.
        let mut qh = token_to_head_major(&q, seq, n_q, head_dim);
        let mut kh = token_to_head_major(&k, seq, n_kv, head_dim);
        let vh = token_to_head_major(&v, seq, n_kv, head_dim);
        // RoPE: bf16 cos/sin applied to bf16 q/k → bf16.
        rope.apply(&mut qh, n_q, seq);
        rope.apply(&mut kh, n_kv, seq);
        self.quantize(&mut qh);
        self.quantize(&mut kh);

        // GQA attention with the additive causal+padding mask → [seq, q_dim].
        // SDPA upcasts q/k/v to f32 internally; output cast back to bf16.
        let t_attn = std::time::Instant::now();
        let mut attn = self.gqa_attention(&qh, &kh, &vh, mask, seq, head_dim, n_q, n_kv)?;
        TE_ATTN_NS.fetch_add(
            t_attn.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.quantize(&mut attn);

        let wo = self
            .weights
            .linear(&format!("{pfx}.self_attn.o_proj"), hidden, q_dim)?;
        let mut attn_out = matmul(&attn, &wo.data, seq, hidden, q_dim, resident)?;
        self.quantize(&mut attn_out);
        for (hv, av) in h.iter_mut().zip(attn_out.iter()) {
            *hv += *av;
        }
        // residual add stored in bf16.
        self.quantize(h);

        // ---- MLP (SwiGLU) block ----
        // post_attention_layernorm: f32 reduction, bf16-valued output.
        let post_ln = self
            .weights
            .vec1(&format!("{pfx}.post_attention_layernorm"), hidden)?;
        x = rms_norm(h, seq, hidden, &post_ln.data, eps);
        self.quantize(&mut x);
        let inter = self.cfg.intermediate_size;
        let wgate = self
            .weights
            .linear(&format!("{pfx}.mlp.gate_proj"), inter, hidden)?;
        let wup = self
            .weights
            .linear(&format!("{pfx}.mlp.up_proj"), inter, hidden)?;
        let mut gate = matmul(&x, &wgate.data, seq, inter, hidden, resident)?;
        let mut up = matmul(&x, &wup.data, seq, inter, hidden, resident)?;
        // nn.Linear outputs bf16.
        self.quantize(&mut gate);
        self.quantize(&mut up);
        // silu(gate) (bf16) * up (bf16) → bf16.
        let mut act = vec![0.0f32; seq * inter];
        // SwiGLU activation (exp-heavy); per-row independent → parallel.
        par_heads(&mut act, seq, inter, |r, dst| {
            let base = r * inter;
            for (j, d) in dst.iter_mut().enumerate() {
                *d = silu(gate[base + j]) * up[base + j];
            }
        });
        self.quantize(&mut act);
        let wdown = self
            .weights
            .linear(&format!("{pfx}.mlp.down_proj"), hidden, inter)?;
        let mut down = matmul(&act, &wdown.data, seq, hidden, inter, resident)?;
        self.quantize(&mut down);
        for (hv, dv) in h.iter_mut().zip(down.iter()) {
            *hv += *dv;
        }
        // residual add stored in bf16.
        self.quantize(h);
        Ok(())
    }

    /// GQA scaled-dot-product attention with an additive `[seq, seq]` mask.
    ///
    /// `q` is `[n_q, seq, head_dim]`, `k`/`v` are `[n_kv, seq, head_dim]`;
    /// query head `h` uses kv head `h / kv_group`. Returns `[seq, n_q*head_dim]`
    /// token-major (heads concatenated) — the layout `o_proj` consumes.
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &[f32],
        seq: usize,
        head_dim: usize,
        n_q: usize,
        n_kv: usize,
    ) -> TeResult<Vec<f32>> {
        let kv_group = n_q / n_kv;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let inner = n_q * head_dim;
        let mut out = vec![0.0f32; seq * inner];
        // Per query head (independent). Parallelised over heads via the shared
        // scoped-thread row splitter.
        let attend = |hq: usize, dst: &mut [f32]| {
            let kv = hq / kv_group;
            let q_off = hq * seq * head_dim;
            let kv_off = kv * seq * head_dim;
            let mut scores = vec![0.0f32; seq];
            for qi in 0..seq {
                let q_row = &q[q_off + qi * head_dim..q_off + (qi + 1) * head_dim];
                let mrow = &mask[qi * seq..(qi + 1) * seq];
                for (ki, score) in scores.iter_mut().enumerate() {
                    // A masked key (causal future or padding) carries an additive
                    // `-inf`, so its softmax weight is `exp(-inf) = 0` no matter
                    // the score. Skip the dot entirely and stamp `-inf` directly —
                    // byte-identical to computing it. For the fixed 512-token
                    // padded sequence this is the bulk of the work on a short
                    // prompt (only the real, non-future keys survive). `mask` only
                    // ever holds `0.0` or `-inf`, so `is_infinite()` is exactly the
                    // masked set.
                    if mrow[ki].is_infinite() {
                        *score = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_row = &k[kv_off + ki * head_dim..kv_off + (ki + 1) * head_dim];
                    *score = dot(q_row, k_row, head_dim) * scale + mrow[ki];
                }
                softmax_simd(&mut scores);
                // dst is this head's [seq, head_dim] slab.
                let o = &mut dst[qi * head_dim..(qi + 1) * head_dim];
                for d in o.iter_mut() {
                    *d = 0.0;
                }
                for (ki, &w) in scores.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    let v_row = &v[kv_off + ki * head_dim..kv_off + (ki + 1) * head_dim];
                    crate::gemm::axpy(o, w, v_row, head_dim);
                }
            }
        };
        // Compute head-major then transpose to token-major.
        let mut head_out = vec![0.0f32; n_q * seq * head_dim];
        par_heads(&mut head_out, n_q, seq * head_dim, attend);
        for hq in 0..n_q {
            for qi in 0..seq {
                let src = &head_out[(hq * seq + qi) * head_dim..(hq * seq + qi + 1) * head_dim];
                let dst = &mut out[qi * inner + hq * head_dim..qi * inner + (hq + 1) * head_dim];
                dst.copy_from_slice(src);
            }
        }
        Ok(out)
    }
}

/// Run `body(head, &mut out[head])` for every one of `heads` length-`width`
/// slabs, split across CPUs via scoped threads (serial for tiny problems).
fn par_heads<F>(out: &mut [f32], heads: usize, width: usize, body: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    debug_assert_eq!(out.len(), heads * width);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(heads.max(1));
    if threads <= 1 || heads < 4 {
        for (hidx, chunk) in out.chunks_mut(width).enumerate() {
            body(hidx, chunk);
        }
        return;
    }
    let per = heads.div_ceil(threads);
    let body_ref = &body;
    std::thread::scope(|scope| {
        let mut base = 0usize;
        for chunk in out.chunks_mut(per * width) {
            let start = base;
            let chunk_heads = chunk.len() / width;
            base += chunk_heads;
            scope.spawn(move || {
                for r in 0..chunk_heads {
                    let slab = &mut chunk[r * width..(r + 1) * width];
                    body_ref(start + r, slab);
                }
            });
        }
    });
}

/// Build the additive `[seq, seq]` attention mask: `mask[i, j] = 0` if query `i`
/// may attend to key `j` (causal `j <= i` AND key `j` unmasked), else `-inf`.
fn build_mask(attention_mask: &[i32], seq: usize) -> Vec<f32> {
    let neg_inf = f32::NEG_INFINITY;
    let mut mask = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            // causal: future keys masked; padding: keys with mask==0 masked.
            let blocked = j > i || attention_mask[j] == 0;
            mask[i * seq + j] = if blocked { neg_inf } else { 0.0 };
        }
    }
    mask
}

/// `out[m, n] = Σ_k input[m, k] · weight[n, k]` (`x · Wᵀ`).
///
/// GPU-first: when the `metal` feature is compiled (macOS) and enabled via
/// `OXI_TE_GPU=1`, the matmul is routed through the f32-exact Metal GEMM
/// ([`crate::te::gpu::te_matmul_gpu`]), which is numerically equivalent to the
/// CPU `gemm_abt` (cos ≈ 1.0 — reassociated f32 sums only). On *any* GPU error
/// it silently falls through to the CPU [`gemm_abt`] path (never panics), so a
/// GPU failure can never break a forward pass. Applied to every TE Linear
/// (Q/K/V, o_proj, gate/up/down) since all call this helper.
///
/// `resident` controls the GPU weight cache residency policy: when `true`, the
/// uploaded GPU buffer is kept across calls (amortizes upload cost across
/// prompts); when `false`, it is evicted after each GEMM to prevent a
/// stale-handle hazard from pointer-key recycling (see [`crate::te::gpu::te_matmul_gpu`]).
static TE_MATMUL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TE_ATTN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn matmul(
    input: &[f32],
    weight: &[f32],
    m: usize,
    n: usize,
    k: usize,
    resident: bool,
) -> TeResult<Vec<f32>> {
    let t = std::time::Instant::now();
    let r = matmul_inner(input, weight, m, n, k, resident);
    TE_MATMUL_NS.fetch_add(
        t.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    r
}

fn matmul_inner(
    input: &[f32],
    weight: &[f32],
    m: usize,
    n: usize,
    k: usize,
    _resident: bool,
) -> TeResult<Vec<f32>> {
    if input.len() != m * k {
        return Err(TeError::Shape(format!(
            "matmul input len {} != m*k {}",
            input.len(),
            m * k
        )));
    }
    if weight.len() != n * k {
        return Err(TeError::Shape(format!(
            "matmul weight len {} != n*k {}",
            weight.len(),
            n * k
        )));
    }
    let mut out = vec![0.0f32; m * n];
    // GPU-first f32 path (opt-in via OXI_TE_GPU=1). The kernel computes the
    // identical `out[m,n] = Σ_k input[m,k]·weight[n,k]` contraction; on any GPU
    // error we fall through to the CPU path below — never panic. `out` is reused
    // (overwritten in full by `gemm_abt`), so a partial GPU write is harmless.
    // `resident` is threaded through so the GPU path can evict non-resident keys
    // from the weight cache after each GEMM (preventing stale-handle hazards).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if crate::te::gpu::te_gpu_enabled() {
            match crate::te::gpu::te_matmul_gpu(weight, input, &mut out, m, n, k, _resident) {
                Ok(()) => return Ok(out),
                Err(_e) => {
                    // Fall through to the CPU GEMM.
                }
            }
        }
    }
    // CUDA sibling of the Metal block above (target_os-disjoint: Linux/Windows).
    // Same `OXI_TE_GPU` toggle (opt-in `=1`); on any GPU error we fall through to
    // the CPU GEMM — never panic.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        if crate::te::cuda_gpu::te_gpu_enabled() {
            match crate::te::cuda_gpu::te_matmul_gpu(weight, input, &mut out, m, n, k, _resident) {
                Ok(()) => return Ok(out),
                Err(_e) => {
                    // Fall through to the CPU GEMM.
                }
            }
        }
    }
    gemm_abt(input, weight, &mut out, m, n, k);
    Ok(out)
}

/// Weighted RMSNorm over the last (`dim`) axis of `x` `[rows, dim]`, eps as
/// given, returning a fresh buffer: `y = weight * x / sqrt(mean(x²) + eps)`.
/// (Variance computed in f32 — matches the reference, which upcasts to f32.)
fn rms_norm(x: &[f32], rows: usize, dim: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * dim);
    debug_assert_eq!(weight.len(), dim);
    let inv_dim = 1.0f32 / dim as f32;
    let mut out = vec![0.0f32; rows * dim];
    // Per-row independent → parallel across CPUs (bit-identical to serial).
    par_heads(&mut out, rows, dim, |r, dst| {
        let src = &x[r * dim..(r + 1) * dim];
        let mut ms = 0.0f32;
        for &v in src {
            ms += v * v;
        }
        ms *= inv_dim;
        let inv_rms = 1.0f32 / (ms + eps).sqrt();
        for i in 0..dim {
            dst[i] = weight[i] * src[i] * inv_rms;
        }
    });
    out
}

/// In-place weighted RMSNorm over each contiguous `head_dim` chunk of `x`
/// (`[rows, head_dim]`, `rows = num_heads * seq`), eps as given.
fn rms_norm_heads(x: &mut [f32], rows: usize, head_dim: usize, weight: &[f32], eps: f32) {
    debug_assert_eq!(weight.len(), head_dim);
    let inv_dim = 1.0f32 / head_dim as f32;
    // Per-row independent → parallel across CPUs (bit-identical to serial).
    par_heads(x, rows, head_dim, |_, row| {
        let mut ms = 0.0f32;
        for &v in row.iter() {
            ms += v * v;
        }
        ms *= inv_dim;
        let inv_rms = 1.0f32 / (ms + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = weight[i] * *v * inv_rms;
        }
    });
}

/// Reshape `[seq, num_heads*head_dim]` token-major into `[num_heads, seq,
/// head_dim]` head-major.
fn token_to_head_major(x: &[f32], seq: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
    let inner = num_heads * head_dim;
    debug_assert_eq!(x.len(), seq * inner);
    let mut out = vec![0.0f32; seq * inner];
    // Output is head-major, so each head's `[seq, head_dim]` block is contiguous
    // → parallel over heads (bit-identical; just a strided gather/copy).
    par_heads(&mut out, num_heads, seq * head_dim, |hh, head_block| {
        for t in 0..seq {
            let src = &x[t * inner + hh * head_dim..t * inner + (hh + 1) * head_dim];
            head_block[t * head_dim..(t + 1) * head_dim].copy_from_slice(src);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weight_normalises() {
        let x = vec![3.0f32, 4.0, 0.0, 0.0];
        let w = vec![1.0f32; 4];
        let y = rms_norm(&x, 1, 4, &w, 0.0);
        // ms = (9+16)/4 = 6.25, rms = 2.5; y = x/2.5
        assert!((y[0] - 1.2).abs() < 1e-6);
        assert!((y[1] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn mask_is_causal_and_padding() {
        // seq=4, last token padded.
        let am = [1, 1, 1, 0];
        let m = build_mask(&am, 4);
        // row 0: only key 0 (causal); keys 1..3 future -> -inf
        assert_eq!(m[0], 0.0);
        assert!(m[1].is_infinite());
        // row 2: keys 0,1,2 allowed (causal), but key 3 is future anyway
        assert_eq!(m[2 * 4], 0.0);
        assert_eq!(m[2 * 4 + 2], 0.0);
        // padding: key 3 masked everywhere it would otherwise be visible (row 3)
        assert!(m[3 * 4 + 3].is_infinite());
    }

    #[test]
    fn token_head_roundtrip() {
        // seq=2, heads=2, head_dim=2 -> token-major [t,h,d]
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let hm = token_to_head_major(&x, 2, 2, 2);
        // head0: tokens (0,1)->[0,1],[4,5]; head1: [2,3],[6,7]
        assert_eq!(hm, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }
}
