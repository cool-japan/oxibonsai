//! The Qwen3-VL ViT tower and merger (bonsai2-design.md §6.2): the graph of
//! the PrismML llama.cpp fork's `clip_graph_qwen3vl::build`, evaluated in
//! `f32` on the CPU.
//!
//! # One block
//!
//! ```text
//! Q | K | V = split(attn_qkv(LayerNorm_ln1(x)) + b)     16 heads x 72
//! Q, K      = RoPE2D(Q), RoPE2D(K)
//! h         = x + attn_out(softmax(Q Kᵀ / sqrt(72)) V) + b   (full, no mask)
//! x'        = h + ffn_down(GELU(ffn_up(LayerNorm_ln2(h)) + b)) + b
//! ```
//!
//! There is no pre-block LayerNorm, no class token, no window attention
//! and no deepstack merger in this projector (the loader refuses a file
//! that has any of them). After the last block: `v.post_ln`, then every
//! four consecutive rows — one 2 x 2 merge window, see
//! [`super::patch_embed`] — are concatenated into one `4 * hidden` token and
//! projected by `mm.0` -> GELU -> `mm.2` to the language model's width.
//!
//! # 2-D rotary embedding
//!
//! Each block rotates Q and K with ggml's `GGML_ROPE_TYPE_VISION` mode
//! (`ggml_rope_multi(.., n_dims = head_dim / 2, sections = {head_dim / 4;
//! 4}, VISION, .., freq_base 10000, freq_scale 1, ext_factor 0,
//! attn_factor 1)`) with the per-patch position `(y, x, y, x)`. Vision mode
//! pairs channel `j` with `j + head_dim / 2` across the **whole** head
//! (`rotate_pairs(ne0, n_dims)`), and restarts the angle at every section
//! boundary: pair `j < head_dim / 4` turns by `y * base^(-2j / n_dims)`,
//! pair `j >= head_dim / 4` by `x * base^(-2(j - head_dim/4) / n_dims)`.
//! That is not the text model's interleaved M-RoPE
//! ([`oxibonsai_kernels::rope_mrope::mrope_build_tables`] refuses the
//! vision axis by design), so `vision_rope_table` builds the table here,
//! reproducing `ggml_mrope_cache_init`'s `f32` arithmetic step for step;
//! the rotation itself is the kernels crate's
//! [`rope_partial_splithalf_simd`] with `n_rot = head_dim`.
//!
//! # GELU
//!
//! The file sets `clip.use_gelu = true`, which the reference maps to
//! `FFN_GELU` = `ggml_gelu`, the **tanh** approximation
//! `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`; the merger MLP is
//! hard-coded to the same op. [`gelu_tanh`] evaluates that formula exactly
//! (ggml's Metal kernel does the same; its CPU backend reads a 16-bit
//! lookup table of the same function).
//!
//! # Attention
//!
//! Full bidirectional attention over up to a few thousand patches: per
//! head and per block of [`ATTN_QUERY_BLOCK`] queries, `Q Kᵀ` and `P V` are
//! two [`KernelDispatcher::gemm_f32`] calls (V is transposed once per
//! layer so both products use the GEMM's `a · wᵀ` form) with a row softmax
//! in between. The work items are independent, so the result is
//! bit-identical for any Rayon thread count.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd;
use oxibonsai_kernels::{cpu_kernel_tier, softmax_simd, KernelDispatcher};
use rayon::prelude::*;

use super::clip_loader::{
    load_vision_weights, VisionBlockWeights, VisionConfig, VisionWeights, SPATIAL_MERGE,
};
use super::patch_embed::{normalize_planar, window_order, PatchEmbed};
use super::{GridSize, ImageRgb8};
use crate::error::{ModelError, ModelResult};

/// Queries per attention work item. 64 queries against 2304 keys is a
/// 576 KiB score block — cache-resident, and enough rows for the GEMM's
/// register tiles.
pub const ATTN_QUERY_BLOCK: usize = 64;

/// `GELU_COEF_A` of ggml's tanh GELU.
const GELU_COEF_A: f32 = 0.044_715;

/// `SQRT_2_OVER_PI` of ggml's tanh GELU.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// The tanh-approximated GELU of the reference graph (`ggml_gelu_f32`):
/// `0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))`.
#[inline]
#[must_use]
pub fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// Build one token's `cos`/`sin` table for ggml's `GGML_ROPE_TYPE_VISION`
/// multi-section RoPE: `head_dim / 2` rotation pairs, pair `j` coupling
/// channels `j` and `j + head_dim / 2`.
///
/// Reproduces `ggml_mrope_cache_init` with `indep_sects = true`,
/// `is_imrope = false`, `freq_scale = 1`, `ext_factor = 0`, `attn_factor = 1`
/// and no frequency factors, in the same `f32` arithmetic:
/// `theta_scale = freq_base^(-2 / n_dims)` with `n_dims = head_dim / 2`,
/// four running angles started at `pos[0..4]` and multiplied by
/// `theta_scale` after every pair, the angle of section `s` restarted at
/// `pos[s]` when pair index `j % sum(sections)` enters it, and pair `j`
/// using the angle of the section it falls in.
///
/// With the tower's sections (`head_dim / 4` each) and `pos = (y, x, y, x)`
/// the first `head_dim / 4` pairs rotate by the row position and the rest
/// by the column position.
///
/// # Errors
///
/// [`ModelError::ShapeInvariant`] for a `head_dim` that is not a positive
/// multiple of 4, sections summing to zero or to more than `head_dim`, or a
/// `cos_out`/`sin_out` shorter than `head_dim / 2`.
pub(crate) fn vision_rope_table(
    pos: [i32; 4],
    sections: [usize; 4],
    head_dim: usize,
    freq_base: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) -> ModelResult<()> {
    let bad = |expected: &str, actual: String| ModelError::ShapeInvariant {
        tensor: "vision rope table".to_string(),
        expected: expected.to_string(),
        actual,
    };
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(bad(
            "a positive head_dim divisible by 4",
            format!("{head_dim}"),
        ));
    }
    let n_pairs = head_dim / 2;
    if cos_out.len() < n_pairs || sin_out.len() < n_pairs {
        return Err(bad(
            "cos/sin buffers of at least head_dim / 2 entries",
            format!(
                "{} / {} for head_dim {head_dim}",
                cos_out.len(),
                sin_out.len()
            ),
        ));
    }
    let sect_dims: usize = sections.iter().sum();
    if sect_dims == 0 || sect_dims > head_dim {
        return Err(bad(
            "sections summing to 1..=head_dim",
            format!("{sections:?} for head_dim {head_dim}"),
        ));
    }
    let sec_w = sections[0] + sections[1];
    let sec_e = sec_w + sections[2];
    let n_dims = n_pairs as f32;
    let theta_scale = freq_base.powf(-2.0 / n_dims);
    let base = pos.map(|p| p as f32);
    let (mut theta_t, mut theta_h, mut theta_w, mut theta_e) = (base[0], base[1], base[2], base[3]);
    for j in 0..n_pairs {
        let sector = j % sect_dims;
        if sector == 0 {
            theta_t = base[0];
        } else if sector == sections[0] {
            theta_h = base[1];
        } else if sector == sec_w {
            theta_w = base[2];
        } else if sector == sec_e {
            theta_e = base[3];
        }
        let theta = if sector >= sections[0] && sector < sec_w {
            theta_h
        } else if sector >= sec_w && sector < sec_w + sections[2] {
            theta_w
        } else if sector >= sec_w + sections[2] {
            theta_e
        } else {
            theta_t
        };
        cos_out[j] = theta.cos();
        sin_out[j] = theta.sin();
        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
    Ok(())
}

/// Per-token RoPE tables for one image, in merge-window order:
/// `[n_patches x head_dim / 2]` each.
struct VisionRope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    n_pairs: usize,
}

impl VisionRope {
    fn for_grid(cfg: &VisionConfig, grid_h: usize, grid_w: usize) -> ModelResult<Self> {
        let order = window_order(grid_h, grid_w);
        let n_pairs = cfg.head_dim / 2;
        let mut cos = vec![0.0f32; order.len() * n_pairs];
        let mut sin = vec![0.0f32; order.len() * n_pairs];
        let sections = cfg.rope_sections();
        cos.par_chunks_mut(n_pairs)
            .zip(sin.par_chunks_mut(n_pairs))
            .zip(order.par_iter())
            .try_for_each(|((c, s), &(y, x))| {
                let y = position_i32(y)?;
                let x = position_i32(x)?;
                vision_rope_table(
                    [y, x, y, x],
                    sections,
                    cfg.head_dim,
                    cfg.rope_freq_base,
                    c,
                    s,
                )
            })?;
        Ok(Self { cos, sin, n_pairs })
    }

    fn row(&self, token: usize) -> (&[f32], &[f32]) {
        let range = token * self.n_pairs..(token + 1) * self.n_pairs;
        (&self.cos[range.clone()], &self.sin[range])
    }
}

fn position_i32(p: usize) -> ModelResult<i32> {
    i32::try_from(p).map_err(|_| ModelError::ShapeInvariant {
        tensor: "vision rope position".to_string(),
        expected: "a patch coordinate that fits in i32".to_string(),
        actual: format!("{p}"),
    })
}

/// Scratch buffers reused by every block of one `encode` call.
struct BlockScratch {
    normed: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    up: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    vt: Vec<f32>,
    heads_out: Vec<f32>,
}

impl BlockScratch {
    fn new(n: usize, cfg: &VisionConfig) -> Self {
        let h = cfg.hidden;
        Self {
            normed: vec![0.0; n * h],
            qkv: vec![0.0; n * 3 * h],
            attn: vec![0.0; n * h],
            proj: vec![0.0; n * h],
            up: vec![0.0; n * cfg.ffn],
            q: vec![0.0; n * h],
            k: vec![0.0; n * h],
            vt: vec![0.0; n * h],
            heads_out: vec![0.0; n * h],
        }
    }
}

/// The Qwen3-VL vision tower of a Bonsai 2 `mmproj`: 27 ViT blocks and the
/// merger, every weight resident as `f32` (see the module docs for the
/// graph).
///
/// Built once with [`VisionTower::from_mmproj`]; [`VisionTower::encode`] is
/// `&self` and may be called from several threads.
pub struct VisionTower {
    config: VisionConfig,
    patch: PatchEmbed,
    blocks: Vec<VisionBlockWeights>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    mm0_w: Vec<f32>,
    mm0_b: Vec<f32>,
    mm2_w: Vec<f32>,
    mm2_b: Vec<f32>,
    bound_tensors: usize,
    dispatcher: KernelDispatcher,
}

impl std::fmt::Debug for VisionTower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionTower")
            .field("config", &self.config)
            .field("blocks", &self.blocks.len())
            .field("bound_tensors", &self.bound_tensors)
            .field("resident_bytes", &self.resident_bytes())
            .field("dispatcher", &self.dispatcher)
            .finish()
    }
}

impl VisionTower {
    /// Load a `qwen3vl_merger` vision projector from its parsed GGUF.
    ///
    /// Validates the `clip.*` metadata, binds every tensor of every block
    /// (refusing a missing, mis-shaped or unexpected one — see
    /// [`super::clip_loader`]) and dequantises all of it to `f32`. The
    /// returned tower owns its weights and does not borrow `gguf`.
    ///
    /// # Errors
    ///
    /// Whatever [`VisionConfig::from_gguf`] and the tensor loader report —
    /// every one names the offending key or tensor.
    pub fn from_mmproj(gguf: &GgufFile<'_>) -> ModelResult<Self> {
        let config = VisionConfig::from_gguf(gguf)?;
        let VisionWeights {
            patch_w0,
            patch_w1,
            patch_b,
            pos_embd,
            blocks,
            post_ln_w,
            post_ln_b,
            mm0_w,
            mm0_b,
            mm2_w,
            mm2_b,
            bound_tensors,
        } = load_vision_weights(gguf, &config)?;
        let patch = PatchEmbed::new(&config, patch_w0, &patch_w1, patch_b, pos_embd)?;
        Ok(Self {
            config,
            patch,
            blocks,
            post_ln_w,
            post_ln_b,
            mm0_w,
            mm0_b,
            mm2_w,
            mm2_b,
            bound_tensors,
            dispatcher: KernelDispatcher::with_tier(cpu_kernel_tier()),
        })
    }

    /// The projector's hyper-parameters.
    #[must_use]
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// The number of ViT blocks bound (27 for Bonsai 2).
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// The number of GGUF tensors bound — always the file's whole
    /// inventory (334 for Bonsai 2: 12 per block plus 10).
    #[must_use]
    pub fn bound_tensor_count(&self) -> usize {
        self.bound_tensors
    }

    /// Bytes held by the resident `f32` weights (about 1.84 GB for the
    /// Bonsai 2 projector).
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let f32_bytes = |v: &Vec<f32>| std::mem::size_of_val(v.as_slice());
        self.patch.resident_bytes()
            + self
                .blocks
                .iter()
                .map(VisionBlockWeights::resident_bytes)
                .sum::<usize>()
            + f32_bytes(&self.post_ln_w)
            + f32_bytes(&self.post_ln_b)
            + f32_bytes(&self.mm0_w)
            + f32_bytes(&self.mm0_b)
            + f32_bytes(&self.mm2_w)
            + f32_bytes(&self.mm2_b)
    }

    /// The merged grid an image of `width x height` pixels produces, after
    /// checking that it can be encoded as-is.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] when a side is zero or not a multiple
    /// of [`VisionConfig::merge_unit`] (32), or when
    /// `(height / 32) * (width / 32)` exceeds `max_tokens`.
    pub fn merged_grid(
        &self,
        width: usize,
        height: usize,
        max_tokens: usize,
    ) -> ModelResult<GridSize> {
        let unit = self.config.merge_unit();
        if width == 0 || height == 0 || !width.is_multiple_of(unit) || !height.is_multiple_of(unit)
        {
            return Err(ModelError::ShapeInvariant {
                tensor: "image".to_string(),
                expected: format!(
                    "a non-empty image whose sides are multiples of {unit} \
                     (patch {} x spatial merge {}); resize before encoding",
                    self.config.patch_size, self.config.spatial_merge
                ),
                actual: format!("{width} x {height}"),
            });
        }
        let grid = GridSize {
            h: height / unit,
            w: width / unit,
        };
        let tokens = grid.h.saturating_mul(grid.w);
        if tokens > max_tokens {
            return Err(ModelError::ShapeInvariant {
                tensor: "image tokens".to_string(),
                expected: format!("at most {max_tokens} merged tokens"),
                actual: format!(
                    "{tokens} ({} x {} merged grid of a {width} x {height} image)",
                    grid.h, grid.w
                ),
            });
        }
        Ok(grid)
    }

    /// Encode an RGB8 image into `[n_merged x projection_dim]` embedding
    /// rows (row-major over the returned merged grid) in the language
    /// model's unrotated embedding space.
    ///
    /// The image must already be sized for the tower: both sides multiples
    /// of 32 and at most `max_tokens` merged tokens (see
    /// [`Self::merged_grid`]); this method normalises the pixels itself
    /// with the file's `image_mean` / `image_std`, and never resizes.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] for a buffer that does not match the
    /// declared size, [`Self::merged_grid`]'s geometry errors, and any
    /// kernel shape error.
    pub fn encode(&self, img: &ImageRgb8, max_tokens: usize) -> ModelResult<(Vec<f32>, GridSize)> {
        img.validate()?;
        let grid = self.merged_grid(img.width, img.height, max_tokens)?;
        let planar = normalize_planar(img, self.config.image_mean, self.config.image_std);
        let rows = self.run(&planar, img.width, img.height)?;
        Ok((rows, grid))
    }

    /// [`Self::encode`] for pixels that are already normalised: `planar`
    /// holds `3 * width * height` values, channel-major (`[3][height]
    /// [width]`, R then G then B), exactly what the reference graph's input
    /// tensor holds after preprocessing.
    ///
    /// # Errors
    ///
    /// As [`Self::encode`].
    pub fn encode_normalized(
        &self,
        planar: &[f32],
        width: usize,
        height: usize,
        max_tokens: usize,
    ) -> ModelResult<(Vec<f32>, GridSize)> {
        let grid = self.merged_grid(width, height, max_tokens)?;
        let rows = self.run(planar, width, height)?;
        Ok((rows, grid))
    }

    /// The whole graph on a validated, normalised planar image.
    fn run(&self, planar: &[f32], width: usize, height: usize) -> ModelResult<Vec<f32>> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let mut x = self
            .patch
            .embed_planar(planar, width, height, &self.dispatcher)?;
        let n = x.len() / h;
        let rope = VisionRope::for_grid(cfg, height / cfg.patch_size, width / cfg.patch_size)?;
        let mut scratch = BlockScratch::new(n, cfg);
        for block in &self.blocks {
            self.block_forward(block, &mut x, n, &rope, &mut scratch)?;
        }

        layer_norm_rows(
            &x,
            &self.post_ln_w,
            &self.post_ln_b,
            cfg.eps,
            h,
            &mut scratch.normed,
        );
        let window = SPATIAL_MERGE * SPATIAL_MERGE;
        let n_merged = n / window;
        let mut mid = vec![0.0f32; n_merged * cfg.merger_hidden];
        self.dispatcher.gemm_f32(
            &scratch.normed,
            &self.mm0_w,
            Some(&self.mm0_b),
            n_merged,
            cfg.merged_width(),
            cfg.merger_hidden,
            &mut mid,
        )?;
        gelu_tanh_inplace(&mut mid);
        let mut out = vec![0.0f32; n_merged * cfg.projection_dim];
        self.dispatcher.gemm_f32(
            &mid,
            &self.mm2_w,
            Some(&self.mm2_b),
            n_merged,
            cfg.merger_hidden,
            cfg.projection_dim,
            &mut out,
        )?;
        Ok(out)
    }

    /// One pre-LayerNorm ViT block, updating the residual stream `x`
    /// (`[n x hidden]`) in place.
    fn block_forward(
        &self,
        w: &VisionBlockWeights,
        x: &mut [f32],
        n: usize,
        rope: &VisionRope,
        s: &mut BlockScratch,
    ) -> ModelResult<()> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let d = &self.dispatcher;

        layer_norm_rows(x, &w.ln1_w, &w.ln1_b, cfg.eps, h, &mut s.normed);
        d.gemm_f32(&s.normed, &w.qkv_w, Some(&w.qkv_b), n, h, 3 * h, &mut s.qkv)?;
        apply_rope_qk(&mut s.qkv, cfg, rope)?;
        self.attention(n, s)?;
        d.gemm_f32(&s.attn, &w.out_w, Some(&w.out_b), n, h, h, &mut s.proj)?;
        add_inplace(x, &s.proj);

        layer_norm_rows(x, &w.ln2_w, &w.ln2_b, cfg.eps, h, &mut s.normed);
        d.gemm_f32(&s.normed, &w.up_w, Some(&w.up_b), n, h, cfg.ffn, &mut s.up)?;
        gelu_tanh_inplace(&mut s.up);
        d.gemm_f32(
            &s.up,
            &w.down_w,
            Some(&w.down_b),
            n,
            cfg.ffn,
            h,
            &mut s.proj,
        )?;
        add_inplace(x, &s.proj);
        Ok(())
    }

    /// Full bidirectional multi-head attention over `s.qkv`
    /// (`[n x (Q | K | V)]`, RoPE already applied), writing the
    /// concatenated head outputs to `s.attn` (`[n x hidden]`).
    fn attention(&self, n: usize, s: &mut BlockScratch) -> ModelResult<()> {
        let cfg = &self.config;
        let (h, heads, hd) = (cfg.hidden, cfg.heads, cfg.head_dim);
        let row = 3 * h;
        let scale = 1.0 / (hd as f32).sqrt();
        let per_head = n * hd;
        let qkv = &s.qkv;

        // Pack each head contiguously: Q and K as `[n x hd]`, V transposed
        // to `[hd x n]` so `P V` is the GEMM's `a . w^T` with `w = V^T`.
        s.q.par_chunks_mut(per_head)
            .zip(s.k.par_chunks_mut(per_head))
            .zip(s.vt.par_chunks_mut(per_head))
            .enumerate()
            .for_each(|(head, ((qh, kh), vh))| {
                for t in 0..n {
                    let base = t * row + head * hd;
                    qh[t * hd..(t + 1) * hd].copy_from_slice(&qkv[base..base + hd]);
                    kh[t * hd..(t + 1) * hd].copy_from_slice(&qkv[base + h..base + h + hd]);
                    for j in 0..hd {
                        vh[j * n + t] = qkv[base + 2 * h + j];
                    }
                }
            });

        let (q, k, vt) = (&s.q, &s.k, &s.vt);
        let dispatcher = &self.dispatcher;
        s.heads_out
            .par_chunks_mut(per_head)
            .enumerate()
            .try_for_each(|(head, out_h)| -> ModelResult<()> {
                let qh = &q[head * per_head..(head + 1) * per_head];
                let kh = &k[head * per_head..(head + 1) * per_head];
                let vh = &vt[head * per_head..(head + 1) * per_head];
                out_h
                    .par_chunks_mut(ATTN_QUERY_BLOCK * hd)
                    .enumerate()
                    .try_for_each_init(Vec::new, |scores: &mut Vec<f32>, (blk, out_b)| {
                        let rows = out_b.len() / hd;
                        let q0 = blk * ATTN_QUERY_BLOCK;
                        scores.clear();
                        scores.resize(rows * n, 0.0);
                        dispatcher.gemm_f32(
                            &qh[q0 * hd..(q0 + rows) * hd],
                            kh,
                            None,
                            rows,
                            hd,
                            n,
                            scores,
                        )?;
                        for srow in scores.chunks_mut(n) {
                            for v in srow.iter_mut() {
                                *v *= scale;
                            }
                            softmax_simd(srow);
                        }
                        dispatcher.gemm_f32(scores, vh, None, rows, n, hd, out_b)?;
                        Ok::<(), ModelError>(())
                    })
            })?;

        let heads_out = &s.heads_out;
        s.attn.par_chunks_mut(h).enumerate().for_each(|(t, dst)| {
            for head in 0..heads {
                let src = (head * n + t) * hd;
                dst[head * hd..(head + 1) * hd].copy_from_slice(&heads_out[src..src + hd]);
            }
        });
        Ok(())
    }
}

/// Rotate the Q and K parts of every `[Q | K | V]` row with the token's
/// vision RoPE table.
fn apply_rope_qk(qkv: &mut [f32], cfg: &VisionConfig, rope: &VisionRope) -> ModelResult<()> {
    let (h, heads, hd) = (cfg.hidden, cfg.heads, cfg.head_dim);
    qkv.par_chunks_mut(3 * h).enumerate().try_for_each_init(
        || vec![0.0f32; hd],
        |tmp, (t, row)| -> ModelResult<()> {
            let (cos, sin) = rope.row(t);
            for part in [0, h] {
                for head in 0..heads {
                    let off = part + head * hd;
                    let dst = &mut row[off..off + hd];
                    tmp.copy_from_slice(dst);
                    rope_partial_splithalf_simd(tmp, dst, hd, hd, cos, sin)?;
                }
            }
            Ok(())
        },
    )
}

/// LayerNorm every `dim`-wide row of `x` into `out`:
/// `(x - mean) / sqrt(var + eps) * w + b`, with the mean and the (biased)
/// variance accumulated in `f64`.
pub(crate) fn layer_norm_rows(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    eps: f32,
    dim: usize,
    out: &mut [f32],
) {
    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .for_each(|(o, xi)| {
            let n = xi.len() as f64;
            let mean = xi.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let var = xi
                .iter()
                .map(|&v| {
                    let c = f64::from(v) - mean;
                    c * c
                })
                .sum::<f64>()
                / n;
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            for (((dst, &v), &wi), &bi) in o.iter_mut().zip(xi).zip(w).zip(b) {
                *dst = ((f64::from(v) - mean) * inv) as f32 * wi + bi;
            }
        });
}

/// [`gelu_tanh`] applied in place.
fn gelu_tanh_inplace(x: &mut [f32]) {
    x.par_chunks_mut(1 << 14).for_each(|chunk| {
        for v in chunk {
            *v = gelu_tanh(*v);
        }
    });
}

/// `x += y`, element-wise.
fn add_inplace(x: &mut [f32], y: &[f32]) {
    x.par_chunks_mut(1 << 14)
        .zip(y.par_chunks(1 << 14))
        .for_each(|(xc, yc)| {
            for (a, b) in xc.iter_mut().zip(yc) {
                *a += *b;
            }
        });
}
