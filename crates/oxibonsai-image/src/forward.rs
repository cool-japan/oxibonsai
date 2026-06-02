//! Top-level FLUX.2 Klein 4B DiT forward pass (Pure Rust, f32).
//!
//! [`DitForward`] borrows a loaded [`DitWeights`] and runs the verified
//! `Flux2KleinFastTransformer` math: embeddings → time embedding → 4-axis RoPE →
//! 5 dual-stream blocks → 20 single-stream blocks → `AdaLayerNormContinuous`
//! head → `proj_out`. It also exposes a Euler sampler loop ([`DitForward::sample`])
//! over golden sigmas, and per-stage taps ([`Stage0`], [`run_stage0`]) used by the
//! parity harness to localise discrepancies.
//!
//! All linear layers are bias-free; the 100 listed projections are ternary
//! (`TQ2_0_g128`), the embedders / modulation / head / time-MLP are bf16 dense.

use crate::blocks::{DoubleBlock, DoubleMod, ModTriple, SingleBlock};
use crate::error::DitResult;
use crate::math::{
    build_rope_tables, dense_matmul, layer_norm_inplace, silu_inplace, timestep_embedding,
    RopeTables,
};
use crate::weights::DitWeights;

/// The two QK-RMSNorm weights for a single block (shared across its heads).
#[derive(Debug, Clone)]
pub struct QkvNorm {
    /// Query norm weight (length `head_dim`).
    pub q: Vec<f32>,
    /// Key norm weight (length `head_dim`).
    pub k: Vec<f32>,
}

/// Number of channels in the sinusoidal timestep embedding before the MLP.
const TIME_EMBED_CHANNELS: usize = 256;

/// A forward-pass runner over a borrowed DiT weight registry.
pub struct DitForward<'w> {
    weights: &'w DitWeights,
}

/// Per-stage S0 taps (embeddings, time embedding, RoPE, modulation params).
pub struct Stage0 {
    /// `x_embedder(hidden_states)` → `[seq_img, hidden]`.
    pub x_emb: Vec<f32>,
    /// `context_embedder(encoder_hidden_states)` → `[seq_txt, hidden]`.
    pub ctx_emb: Vec<f32>,
    /// Joint RoPE tables ([seq_txt + seq_img, half]).
    pub rope: RopeTables,
    /// Image-stream modulation (MSA + MLP triples).
    pub mod_img: DoubleMod,
    /// Text-stream modulation (MSA + MLP triples).
    pub mod_txt: DoubleMod,
    /// Single-stream modulation triple.
    pub mod_single: ModTriple,
    /// Time embedding `temb` ([hidden]).
    pub temb: Vec<f32>,
}

impl<'w> DitForward<'w> {
    /// Wrap a loaded weight registry.
    pub fn new(weights: &'w DitWeights) -> Self {
        Self { weights }
    }

    /// Hidden size (= heads * head_dim).
    fn hidden(&self) -> usize {
        self.weights.config().hidden_size() as usize
    }

    /// Compute the timestep embedding `temb` ([hidden]) from a scalar `t`.
    ///
    /// `t` is fed directly (the goldens are already on the [0, 1000] scale).
    /// `temb = linear_2(silu(linear_1(sinusoidal(t))))`.
    pub fn time_embedding(&self, t: f32) -> DitResult<Vec<f32>> {
        let hidden = self.hidden();
        let sinus = timestep_embedding(t, TIME_EMBED_CHANNELS);
        let l1 = self
            .weights
            .bf16_tensor("time_guidance_embed.timestep_embedder.linear_1.weight")?;
        let (out1, in1) = (l1.shape()[0] as usize, l1.shape()[1] as usize);
        let mut h = dense_matmul(&sinus, &l1.to_f32_vec(), 1, out1, in1)?;
        silu_inplace(&mut h);
        let l2 = self
            .weights
            .bf16_tensor("time_guidance_embed.timestep_embedder.linear_2.weight")?;
        let (out2, in2) = (l2.shape()[0] as usize, l2.shape()[1] as usize);
        let temb = dense_matmul(&h, &l2.to_f32_vec(), 1, out2, in2)?;
        debug_assert_eq!(temb.len(), hidden);
        Ok(temb)
    }

    /// Project `temb` through a modulation Linear and chunk it into `sets`
    /// `(shift, scale, gate)` triples of width `hidden`.
    fn modulation(&self, name: &str, temb: &[f32], sets: usize) -> DitResult<Vec<ModTriple>> {
        let hidden = self.hidden();
        // mod = linear(silu(temb))
        let mut act = temb.to_vec();
        silu_inplace(&mut act);
        let w = self.weights.bf16_tensor(name)?;
        let (out, inp) = (w.shape()[0] as usize, w.shape()[1] as usize);
        let proj = dense_matmul(&act, &w.to_f32_vec(), 1, out, inp)?;
        // proj is [3*sets*hidden]; split into 3*sets chunks of `hidden`, group
        // each consecutive 3 as (shift, scale, gate).
        let mut triples = Vec::with_capacity(sets);
        for s in 0..sets {
            let base = s * 3 * hidden;
            let shift = proj[base..base + hidden].to_vec();
            let scale = proj[base + hidden..base + 2 * hidden].to_vec();
            let gate = proj[base + 2 * hidden..base + 3 * hidden].to_vec();
            triples.push(ModTriple { shift, scale, gate });
        }
        Ok(triples)
    }

    /// Compute the S0 stage (embeddings, RoPE, modulation, time embedding).
    ///
    /// - `hidden_states`: `[seq_img, in_channels]`.
    /// - `encoder_hidden_states`: `[seq_txt, joint_attention_dim]`.
    /// - `img_ids`: `[seq_img, num_axes]`; `txt_ids`: `[seq_txt, num_axes]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_stage0(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        img_ids: &[f32],
        txt_ids: &[f32],
        seq_img: usize,
        seq_txt: usize,
        timestep: f32,
    ) -> DitResult<Stage0> {
        let cfg = self.weights.config();
        let hidden = self.hidden();
        let in_channels = cfg.in_channels as usize;
        let joint_dim = cfg.joint_attention_dim as usize;
        let num_axes = cfg.axes_dims_rope.len();

        // Embeddings (bf16 dense).
        let st = std::env::var("OXI_IMAGE_TIMING").is_ok();
        let t_xe = std::time::Instant::now();
        let xe = self.weights.bf16_tensor("x_embedder.weight")?;
        let x_emb = dense_matmul(
            hidden_states,
            &xe.to_f32_vec(),
            seq_img,
            hidden,
            in_channels,
        )?;
        if st {
            eprintln!("[timing]     x_emb: {:.3}s", t_xe.elapsed().as_secs_f64());
        }
        let t_ce = std::time::Instant::now();
        let ce = self.weights.bf16_tensor("context_embedder.weight")?;
        let ctx_emb = dense_matmul(
            encoder_hidden_states,
            &ce.to_f32_vec(),
            seq_txt,
            hidden,
            joint_dim,
        )?;
        if st {
            eprintln!("[timing]     ctx_emb: {:.3}s", t_ce.elapsed().as_secs_f64());
        }

        // RoPE: txt-first, then img.
        let txt_rope = build_rope_tables(
            txt_ids,
            seq_txt,
            num_axes,
            &cfg.axes_dims_rope,
            cfg.rope_theta,
        )?;
        let img_rope = build_rope_tables(
            img_ids,
            seq_img,
            num_axes,
            &cfg.axes_dims_rope,
            cfg.rope_theta,
        )?;
        let rope = concat_rope(&txt_rope, &img_rope);

        // Time embedding + modulation projections.
        let temb = self.time_embedding(timestep)?;
        let mut mi = self.modulation("double_stream_modulation_img.linear.weight", &temb, 2)?;
        let mut mt = self.modulation("double_stream_modulation_txt.linear.weight", &temb, 2)?;
        let ms = self.modulation("single_stream_modulation.linear.weight", &temb, 1)?;

        // Pop in reverse to avoid clones / index churn.
        let img_mlp = mi.pop().unwrap_or_default_triple(hidden);
        let img_msa = mi.pop().unwrap_or_default_triple(hidden);
        let txt_mlp = mt.pop().unwrap_or_default_triple(hidden);
        let txt_msa = mt.pop().unwrap_or_default_triple(hidden);
        let mod_single = ms
            .into_iter()
            .next()
            .unwrap_or_else(|| ModTriple::zeros(hidden));

        Ok(Stage0 {
            x_emb,
            ctx_emb,
            rope,
            mod_img: DoubleMod {
                msa: img_msa,
                mlp: img_mlp,
            },
            mod_txt: DoubleMod {
                msa: txt_msa,
                mlp: txt_mlp,
            },
            mod_single,
            temb,
        })
    }

    /// Run the full DiT forward and return the noise prediction
    /// `[seq_img, in_channels]`.
    ///
    /// Inputs as in [`Self::run_stage0`]. `taps`, when `Some`, is filled with
    /// per-block intermediates for the parity harness.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &[f32],
        encoder_hidden_states: &[f32],
        img_ids: &[f32],
        txt_ids: &[f32],
        seq_img: usize,
        seq_txt: usize,
        timestep: f32,
        mut taps: Option<&mut ForwardTaps>,
    ) -> DitResult<Vec<f32>> {
        let cfg = self.weights.config();
        let hidden = self.hidden();
        let num_heads = cfg.num_attention_heads as usize;
        let head_dim = cfg.attention_head_dim as usize;
        let ffn_inner = cfg.ffn_inner_size() as usize;
        let eps = cfg.eps;
        let in_channels = cfg.in_channels as usize;

        let dit_timed = std::env::var("OXI_IMAGE_TIMING").is_ok();
        let t_s0 = std::time::Instant::now();
        let s0 = self.run_stage0(
            hidden_states,
            encoder_hidden_states,
            img_ids,
            txt_ids,
            seq_img,
            seq_txt,
            timestep,
        )?;
        if dit_timed {
            eprintln!(
                "[timing]   stage0 total: {:.3}s",
                t_s0.elapsed().as_secs_f64()
            );
        }
        if let Some(t) = taps.as_deref_mut() {
            t.stage0 = Some(Stage0 {
                x_emb: s0.x_emb.clone(),
                ctx_emb: s0.ctx_emb.clone(),
                rope: s0.rope.clone(),
                mod_img: s0.mod_img.clone(),
                mod_txt: s0.mod_txt.clone(),
                mod_single: s0.mod_single.clone(),
                temb: s0.temb.clone(),
            });
        }

        let mut hidden_buf = s0.x_emb;
        let mut enc_buf = s0.ctx_emb;

        let timed = std::env::var("DIT_TIME_BLOCKS").is_ok();

        // ── 5 dual-stream blocks ──
        for i in 0..cfg.num_layers {
            let t0 = std::time::Instant::now();
            DoubleBlock::new(i).forward(
                self.weights,
                &mut hidden_buf,
                &mut enc_buf,
                seq_img,
                seq_txt,
                hidden,
                num_heads,
                head_dim,
                ffn_inner,
                eps,
                &s0.rope,
                &s0.mod_img,
                &s0.mod_txt,
            )?;
            if timed {
                eprintln!("  double block {i}: {:.2}s", t0.elapsed().as_secs_f64());
            }
            if let Some(t) = taps.as_deref_mut() {
                t.double_enc.push(enc_buf.clone());
                t.double_h.push(hidden_buf.clone());
            }
        }

        // ── Concat [txt; img] → joint stream [seq_joint, hidden] ──
        let seq_joint = seq_txt + seq_img;
        let mut joint = vec![0.0f32; seq_joint * hidden];
        joint[..seq_txt * hidden].copy_from_slice(&enc_buf);
        joint[seq_txt * hidden..].copy_from_slice(&hidden_buf);
        if let Some(t) = taps.as_deref_mut() {
            t.single_in = Some(joint.clone());
        }

        // Pre-resolve single-block QK norms.
        let mut single_norms = Vec::with_capacity(cfg.num_single_layers as usize);
        for j in 0..cfg.num_single_layers {
            let p = format!("single_transformer_blocks.{j}");
            single_norms.push(QkvNorm {
                q: self
                    .weights
                    .bf16_tensor(&format!("{p}.attn.norm_q.weight"))?
                    .to_f32_vec(),
                k: self
                    .weights
                    .bf16_tensor(&format!("{p}.attn.norm_k.weight"))?
                    .to_f32_vec(),
            });
        }

        // ── 20 single-stream blocks ──
        // Resident fast path (native-cuda): upload `joint` once, run the whole
        // single-block stack on the GPU, download once — collapsing the per-block
        // PCIe round-trip + sync. Skipped when capturing per-block taps (parity
        // mode needs each block's intermediate); falls back to the per-block loop
        // on any GPU error.
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        let single_done = taps.is_none()
            && crate::cuda_gpu::dit_gpu_enabled()
            && crate::cuda_gpu::dit_fused_enabled()
            && crate::cuda_gpu::single_blocks_gpu(
                self.weights,
                cfg.num_single_layers as usize,
                &mut joint,
                seq_joint,
                hidden,
                num_heads,
                head_dim,
                ffn_inner,
                eps,
                &s0.rope,
                &s0.mod_single,
                &single_norms,
            )
            .is_ok();
        #[cfg(not(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        )))]
        let single_done = false;

        if !single_done {
            for j in 0..cfg.num_single_layers {
                let t0 = std::time::Instant::now();
                SingleBlock::new(j).forward(
                    self.weights,
                    &mut joint,
                    seq_joint,
                    hidden,
                    num_heads,
                    head_dim,
                    ffn_inner,
                    eps,
                    &s0.rope,
                    &s0.mod_single,
                    &single_norms[j as usize],
                )?;
                if timed {
                    eprintln!("  single block {j}: {:.2}s", t0.elapsed().as_secs_f64());
                }
                if let Some(t) = taps.as_deref_mut() {
                    t.single_h.push(joint.clone());
                }
            }
        }

        // ── Drop txt → image rows only ──
        let img_part = joint[seq_txt * hidden..].to_vec();

        // ── AdaLayerNormContinuous head ──
        let t_final = std::time::Instant::now();
        let normed = self.ada_layer_norm_continuous(&img_part, &s0.temb, seq_img, hidden, eps)?;
        if let Some(t) = taps {
            t.norm_out = Some(normed.clone());
        }

        // ── proj_out → [seq_img, in_channels] ──
        let po = self.weights.bf16_tensor("proj_out.weight")?;
        let noise = dense_matmul(&normed, &po.to_f32_vec(), seq_img, in_channels, hidden)?;
        if dit_timed {
            eprintln!(
                "[timing]   final(ada+proj): {:.3}s",
                t_final.elapsed().as_secs_f64()
            );
        }
        Ok(noise)
    }

    /// `AdaLayerNormContinuous`: `te = linear(silu(temb))` ([6144]); split into
    /// `scale = te[:hidden]`, `shift = te[hidden:]`; return
    /// `LN(x) * (1 + scale) + shift`.
    fn ada_layer_norm_continuous(
        &self,
        x: &[f32],
        temb: &[f32],
        seq: usize,
        hidden: usize,
        eps: f32,
    ) -> DitResult<Vec<f32>> {
        let mut act = temb.to_vec();
        silu_inplace(&mut act);
        let w = self.weights.bf16_tensor("norm_out.linear.weight")?;
        let (out, inp) = (w.shape()[0] as usize, w.shape()[1] as usize);
        let te = dense_matmul(&act, &w.to_f32_vec(), 1, out, inp)?;
        let scale = &te[..hidden];
        let shift = &te[hidden..2 * hidden];
        let mut y = x.to_vec();
        layer_norm_inplace(&mut y, seq, hidden, eps);
        for r in 0..seq {
            let row = &mut y[r * hidden..(r + 1) * hidden];
            for i in 0..hidden {
                row[i] = row[i] * (1.0 + scale[i]) + shift[i];
            }
        }
        Ok(y)
    }

    /// Euler sampler loop over the golden sigmas.
    ///
    /// Starting from `latents` `[seq_img, in_channels]`, for each step `t`:
    /// `noise = forward(latents, cond, ids, timesteps[t]);
    ///  latents += (sigmas[t+1] - sigmas[t]) * noise`. Returns the final
    /// latents. If `per_step` is `Some`, each step's `(noise, latents)` is
    /// recorded.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        init_latents: &[f32],
        encoder_hidden_states: &[f32],
        img_ids: &[f32],
        txt_ids: &[f32],
        seq_img: usize,
        seq_txt: usize,
        timesteps: &[f32],
        sigmas: &[f32],
        mut per_step: Option<&mut Vec<StepTap>>,
    ) -> DitResult<Vec<f32>> {
        let mut latents = init_latents.to_vec();
        for (step, &t) in timesteps.iter().enumerate() {
            let noise = self.forward(
                &latents,
                encoder_hidden_states,
                img_ids,
                txt_ids,
                seq_img,
                seq_txt,
                t,
                None,
            )?;
            let dt = sigmas[step + 1] - sigmas[step];
            for (l, &n) in latents.iter_mut().zip(noise.iter()) {
                *l += dt * n;
            }
            if let Some(ps) = per_step.as_deref_mut() {
                ps.push(StepTap {
                    noise,
                    latents: latents.clone(),
                });
            }
        }
        Ok(latents)
    }
}

/// One sampler step's taps.
pub struct StepTap {
    /// The DiT noise prediction for this step.
    pub noise: Vec<f32>,
    /// The latents after applying the Euler update for this step.
    pub latents: Vec<f32>,
}

/// Optional per-block intermediates captured during a forward pass.
#[derive(Default)]
pub struct ForwardTaps {
    /// S0 stage taps.
    pub stage0: Option<Stage0>,
    /// `enc` after each double block (5).
    pub double_enc: Vec<Vec<f32>>,
    /// `hidden` after each double block (5).
    pub double_h: Vec<Vec<f32>>,
    /// The joint `[txt; img]` stream entering the single blocks.
    pub single_in: Option<Vec<f32>>,
    /// `hidden` after each single block (20).
    pub single_h: Vec<Vec<f32>>,
    /// The AdaLN head output (before `proj_out`).
    pub norm_out: Option<Vec<f32>>,
}

/// Concatenate two RoPE tables along the sequence axis (a-first).
fn concat_rope(a: &RopeTables, b: &RopeTables) -> RopeTables {
    debug_assert_eq!(a.half, b.half);
    let half = a.half;
    let seq = a.seq + b.seq;
    let mut cos = Vec::with_capacity(seq * half);
    let mut sin = Vec::with_capacity(seq * half);
    cos.extend_from_slice(&a.cos);
    cos.extend_from_slice(&b.cos);
    sin.extend_from_slice(&a.sin);
    sin.extend_from_slice(&b.sin);
    RopeTables {
        cos,
        sin,
        seq,
        half,
    }
}

impl ModTriple {
    /// A zero-filled triple of width `hidden` (defensive fallback only).
    fn zeros(hidden: usize) -> Self {
        ModTriple {
            shift: vec![0.0; hidden],
            scale: vec![0.0; hidden],
            gate: vec![0.0; hidden],
        }
    }
}

/// Small helper to keep the `pop()` flow free of `unwrap`.
trait OrDefaultTriple {
    fn unwrap_or_default_triple(self, hidden: usize) -> ModTriple;
}

impl OrDefaultTriple for Option<ModTriple> {
    fn unwrap_or_default_triple(self, hidden: usize) -> ModTriple {
        self.unwrap_or_else(|| ModTriple::zeros(hidden))
    }
}
