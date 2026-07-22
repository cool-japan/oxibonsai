//! The FLUX.2 SMALL VAE decoder driver: builds every layer from a
//! [`VaeWeights`] registry and runs the untiled decode path
//! (`decode_packed_latents`), with optional per-stage taps for parity checks.
//!
//! Pipeline (NCHW throughout, batch 1), for the SMALL decoder
//! (channels `[96, 192, 384, 384]`, reversed for decode → `[384,384,192,96]`):
//!
//! ```text
//! packed[1,128,32,32]
//!   → bn_denorm        [128,32,32]
//!   → unpatchify       [32,64,64]
//!   → post_quant_conv  [32,64,64]   (k=1)
//!   → conv_in          [384,64,64]  (k=3)
//!   → mid_block        [384,64,64]  (resnet, attention, resnet)
//!   → up_block 0       [384,128,128](3 resnets + upsample)
//!   → up_block 1       [384,256,256]
//!   → up_block 2       [192,512,512](resnet0 has 1×1 shortcut)
//!   → up_block 3       [96,512,512] (resnet0 has 1×1 shortcut, NO upsample)
//!   → conv_norm_out    [96,512,512] (GroupNorm, pre-silu tap)
//!   → silu
//!   → conv_out         [3,512,512]  (k=3)
//! ```

use crate::vae::attention::{AttentionBlock, Linear};
use crate::vae::conv::Conv2d;
use crate::vae::error::{VaeError, VaeResult};
use crate::vae::norm::GroupNorm;
use crate::vae::ops::{bn_denorm, silu_inplace, unpatchify, upsample_nearest2x};
use crate::vae::resnet::ResnetBlock2D;
use crate::vae::weights::VaeWeights;

/// GroupNorm groups used throughout the VAE.
const NUM_GROUPS: usize = 32;
/// GroupNorm epsilon (pytorch_compatible).
const GN_EPS: f32 = 1e-6;
/// BatchNorm-stats denorm epsilon.
const BN_EPS: f32 = 1e-4;
/// Latent channels.
const LATENT_CH: usize = 32;

/// An NCHW tensor carried between decode stages.
#[derive(Clone)]
pub struct Map {
    /// NCHW data `[c, h, w]`.
    pub data: Vec<f32>,
    /// Channels.
    pub c: usize,
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
}

impl Map {
    fn new(data: Vec<f32>, c: usize, h: usize, w: usize) -> Self {
        Self { data, c, h, w }
    }

    /// Crate-visible constructor, used by the tiling helpers in
    /// [`crate::vae::tiling`].
    pub(crate) fn new_pub(data: Vec<f32>, c: usize, h: usize, w: usize) -> Self {
        Self { data, c, h, w }
    }
}

/// Optional capture of every per-stage intermediate (for parity validation).
#[derive(Default)]
pub struct DecodeTaps {
    /// After BatchNorm-stats denorm, `[128,32,32]`.
    pub bn_denorm: Option<Vec<f32>>,
    /// After unpatchify, `[32,64,64]`.
    pub unpatchified: Option<Vec<f32>>,
    /// After `post_quant_conv`, `[32,64,64]`.
    pub post_quant_conv: Option<Vec<f32>>,
    /// After `conv_in`, `[384,64,64]`.
    pub conv_in: Option<Vec<f32>>,
    /// After the mid block, `[384,64,64]`.
    pub mid: Option<Vec<f32>>,
    /// After each up block (`up[i]`).
    pub up: Vec<Vec<f32>>,
    /// After `conv_norm_out` (BEFORE the final SiLU), `[96,512,512]`.
    pub conv_norm_out: Option<Vec<f32>>,
}

/// A decoder up-block: `num_layers` resnets then an optional upsampler.
struct UpBlock {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<Conv2d>,
}

/// The mid block: resnet, attention, resnet.
struct MidBlock {
    resnet0: ResnetBlock2D,
    attention: AttentionBlock,
    resnet1: ResnetBlock2D,
}

/// The full SMALL VAE decoder, with all weights loaded.
pub struct VaeDecoder {
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
    post_quant_conv: Conv2d,
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl VaeDecoder {
    /// Build the decoder from an exported-weights registry.
    ///
    /// # Errors
    /// [`VaeError`] if any required tensor is missing or has an unexpected shape.
    pub fn from_weights(w: &VaeWeights) -> VaeResult<Self> {
        let bn_mean = w.vec1("bn.running_mean")?.data.clone();
        let bn_var = w.vec1("bn.running_var")?.data.clone();
        let post_quant_conv = load_conv(w, "post_quant_conv", 0)?;
        let conv_in = load_conv(w, "decoder.conv_in", 1)?;
        let mid_block = load_mid_block(w, "decoder.mid_block")?;

        // 4 up blocks; blocks 0,1,2 have an upsampler, block 3 does not.
        let mut up_blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let prefix = format!("decoder.up_blocks.{i}");
            let has_upsample = i < 3;
            up_blocks.push(load_up_block(w, &prefix, 3, has_upsample)?);
        }

        let conv_norm_out = load_group_norm(w, "decoder.conv_norm_out")?;
        let conv_out = load_conv(w, "decoder.conv_out", 1)?;

        Ok(Self {
            bn_mean,
            bn_var,
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    /// Decode packed latents `[1, 128, 32, 32]` (flat NCHW, batch 1) to an RGB
    /// map `[3, 512, 512]` (untiled).
    ///
    /// `taps`, if supplied, captures every per-stage intermediate.
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch or a propagated layer error.
    pub fn decode_packed_latents(
        &self,
        packed: &[f32],
        ph: usize,
        pw: usize,
        mut taps: Option<&mut DecodeTaps>,
    ) -> VaeResult<Map> {
        let packed_ch = 4 * LATENT_CH; // 128
        if packed.len() != packed_ch * ph * pw {
            return Err(VaeError::Shape(format!(
                "decode_packed input len {} != 128*{ph}*{pw}",
                packed.len()
            )));
        }
        // 1. BatchNorm-stats denorm.
        let denorm = bn_denorm(
            packed,
            &self.bn_mean,
            &self.bn_var,
            packed_ch,
            ph,
            pw,
            BN_EPS,
        )?;
        if let Some(t) = taps.as_deref_mut() {
            t.bn_denorm = Some(denorm.clone());
        }
        // 2. Unpatchify [128,h,w] → [32, 2h, 2w].
        let up = unpatchify(&denorm, packed_ch, ph, pw)?;
        if let Some(t) = taps.as_deref_mut() {
            t.unpatchified = Some(up.data.clone());
        }
        let latents = Map::new(up.data, up.c, up.h, up.w);
        self.decode(&latents, taps)
    }

    /// Decode unpatchified latents `[32, 64, 64]` to RGB `[3, 512, 512]`.
    ///
    /// # Errors
    /// As [`Self::decode_packed_latents`].
    pub fn decode(&self, latents: &Map, mut taps: Option<&mut DecodeTaps>) -> VaeResult<Map> {
        // post_quant_conv (k=1): scaling_factor=1, shift_factor=0 → no-op scale.
        let pqc = self
            .post_quant_conv
            .forward(&latents.data, latents.h, latents.w)?;
        let mut cur = Map::new(pqc.data, self.post_quant_conv.out_ch, pqc.h, pqc.w);
        if let Some(t) = taps.as_deref_mut() {
            t.post_quant_conv = Some(cur.data.clone());
        }
        // conv_in (k=3).
        let ci = self.conv_in.forward(&cur.data, cur.h, cur.w)?;
        cur = Map::new(ci.data, self.conv_in.out_ch, ci.h, ci.w);
        if let Some(t) = taps.as_deref_mut() {
            t.conv_in = Some(cur.data.clone());
        }
        // mid block.
        cur = self.mid_block.forward(&cur)?;
        if let Some(t) = taps.as_deref_mut() {
            t.mid = Some(cur.data.clone());
        }
        // up blocks.
        for ub in &self.up_blocks {
            cur = ub.forward(&cur)?;
            if let Some(t) = taps.as_deref_mut() {
                t.up.push(cur.data.clone());
            }
        }
        // conv_norm_out (GroupNorm), tap pre-silu. This is the last tap, so
        // `taps` is consumed directly (no reborrow needed).
        self.conv_norm_out
            .forward_inplace(&mut cur.data, cur.h, cur.w)?;
        if let Some(t) = taps {
            t.conv_norm_out = Some(cur.data.clone());
        }
        // silu → conv_out (k=3) → [3,512,512].
        silu_inplace(&mut cur.data);
        let out = self.conv_out.forward(&cur.data, cur.h, cur.w)?;
        Ok(Map::new(out.data, self.conv_out.out_ch, out.h, out.w))
    }

    /// Run the decode prologue through the mid block: `bn_denorm → unpatchify →
    /// post_quant_conv → conv_in → mid_block`. The mid block runs whole (it has
    /// global attention, so it cannot be spatially tiled), returning the map
    /// `[C, H, W]` that enters the up-block stack.
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch or a propagated layer error.
    fn decode_through_mid(&self, packed: &[f32], ph: usize, pw: usize) -> VaeResult<Map> {
        let packed_ch = 4 * LATENT_CH; // 128
        if packed.len() != packed_ch * ph * pw {
            return Err(VaeError::Shape(format!(
                "decode_packed input len {} != 128*{ph}*{pw}",
                packed.len()
            )));
        }
        // 1. BatchNorm-stats denorm.
        let denorm = bn_denorm(
            packed,
            &self.bn_mean,
            &self.bn_var,
            packed_ch,
            ph,
            pw,
            BN_EPS,
        )?;
        // 2. Unpatchify [128,h,w] → [32, 2h, 2w].
        let up = unpatchify(&denorm, packed_ch, ph, pw)?;
        let mut cur = Map::new(up.data, up.c, up.h, up.w);
        // 3. post_quant_conv (k=1).
        let pqc = self.post_quant_conv.forward(&cur.data, cur.h, cur.w)?;
        cur = Map::new(pqc.data, self.post_quant_conv.out_ch, pqc.h, pqc.w);
        // 4. conv_in (k=3).
        let ci = self.conv_in.forward(&cur.data, cur.h, cur.w)?;
        cur = Map::new(ci.data, self.conv_in.out_ch, ci.h, ci.w);
        // 5. mid block (whole — global attention).
        self.mid_block.forward(&cur)
    }

    /// Run decode through `conv_norm_out` (GroupNorm applied), stopping **before**
    /// the final `silu_inplace + conv_out.forward`, with the up-blocks run
    /// **whole**. Returns the post-GroupNorm plane `[96, H, W]` (or whatever the
    /// actual output-stage channel count is).
    ///
    /// Shared entry-point used by [`Self::decode_packed_latents_tiled`] so the
    /// `AfterConvNormOut` boundary can be tiled without duplicating the decode
    /// prologue.
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch or a propagated layer error.
    fn decode_before_conv_out(&self, packed: &[f32], ph: usize, pw: usize) -> VaeResult<Map> {
        let mut cur = self.decode_through_mid(packed, ph, pw)?;
        // 6. up blocks (whole).
        for ub in &self.up_blocks {
            cur = ub.forward(&cur)?;
        }
        // 7. conv_norm_out (GroupNorm — global).
        self.conv_norm_out
            .forward_inplace(&mut cur.data, cur.h, cur.w)?;
        // Caller applies silu + conv_out (tiled or not).
        Ok(cur)
    }

    /// `AfterMid` variant of [`Self::decode_before_conv_out`]: the up-blocks run
    /// through [`UpBlock::forward_tiled`], which bounds each up-block convolution's
    /// im2col scratch to `max_tile_pixels` output pixels. The activation planes
    /// between blocks stay whole (they are small relative to the im2col) so the
    /// GroupNorm statistics remain exactly global and the output is **bit-identical
    /// to the untiled CPU decode** — only the multi-GB up-block im2col peak is
    /// capped (from the untiled ~3.6 GB up-block peak at 512² down to
    /// `O(max_tile_pixels · k·k·in_ch)`, e.g. ≈1 GB at a 256²-pixel budget).
    ///
    /// `conv_norm_out` runs whole (global GroupNorm stats). The up-block convs run
    /// on the CPU, matching [`Conv2d::forward_tiled`] (the GPU conv path has no
    /// host im2col to bound); vs a GPU untiled decode the result matches to
    /// `cos ≈ 1`, like the `AfterConvNormOut` tail tiling.
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch or a propagated layer error.
    fn decode_before_conv_out_tiled(
        &self,
        packed: &[f32],
        ph: usize,
        pw: usize,
        max_tile_pixels: usize,
    ) -> VaeResult<Map> {
        let mut cur = self.decode_through_mid(packed, ph, pw)?;
        // 6. up blocks (im2col bounded per conv; planes whole → stats global).
        for ub in &self.up_blocks {
            cur = ub.forward_tiled(&cur, max_tile_pixels)?;
        }
        // 7. conv_norm_out (GroupNorm — global).
        self.conv_norm_out
            .forward_inplace(&mut cur.data, cur.h, cur.w)?;
        Ok(cur)
    }

    /// Tiled decode: numerically equivalent to [`Self::decode_packed_latents`]
    /// (bit-exact on the CPU path; `cos ≈ 1` vs the GPU path, whose sum
    /// reassociation depends on tile size), with peak scratch memory bounded by
    /// `cfg.tile_px` instead of the full output size.
    ///
    /// Two boundaries (see [`crate::vae::tiling::TileBoundary`]):
    ///
    /// * [`AfterConvNormOut`](crate::vae::tiling::TileBoundary::AfterConvNormOut)
    ///   (default): everything through `conv_norm_out` runs whole, then only the
    ///   final `silu → conv_out` is tiled (halo = 1 px). Per-op Metal/CUDA dispatch
    ///   of the upstream ops is unchanged.
    /// * [`AfterMid`](crate::vae::tiling::TileBoundary::AfterMid): additionally bounds the up-block convolution
    ///   im2col — the real ~3.6 GB up-block peak at 512² — by running each
    ///   up-block conv through [`Conv2d::forward_tiled`] with a
    ///   `cfg.tile_px²`-pixel budget. Activation planes stay whole (GroupNorm
    ///   stats global), so it is bit-identical to the untiled CPU decode. The
    ///   up-block convs run on the CPU under this boundary (the host im2col is the
    ///   thing being bounded); `conv_norm_out` and the `silu → conv_out` tail are
    ///   handled as in `AfterConvNormOut`.
    ///
    /// Auto-tiling in [`crate::pipeline`] and [`crate::session`] activates only
    /// when `16 * ph > 512` (i.e., output width > 512 px), so 512-px renders
    /// are unaffected.
    ///
    /// # Errors
    /// [`VaeError::Shape`] on a length mismatch or a propagated layer error.
    pub fn decode_packed_latents_tiled(
        &self,
        packed: &[f32],
        ph: usize,
        pw: usize,
        cfg: crate::vae::tiling::TileConfig,
    ) -> VaeResult<Map> {
        use crate::vae::tiling::{tile_silu_conv_out, TileBoundary};
        // Both boundaries share the tiled `silu → conv_out` tail; they differ only
        // in whether the up-blocks are run whole (`AfterConvNormOut`) or with the
        // im2col bounded (`AfterMid`). `conv_norm_out` always runs whole so the
        // final GroupNorm keeps globally-correct per-group statistics.
        let pre_silu = match cfg.boundary {
            TileBoundary::AfterConvNormOut => self.decode_before_conv_out(packed, ph, pw)?,
            TileBoundary::AfterMid => {
                // Budget the up-block im2col to ~`tile_px²` output pixels, matching
                // the spatial-tile memory scale of the `silu → conv_out` tail.
                let max_tile_pixels = cfg.tile_px.saturating_mul(cfg.tile_px).max(1);
                self.decode_before_conv_out_tiled(packed, ph, pw, max_tile_pixels)?
            }
        };
        tile_silu_conv_out(
            &pre_silu.data,
            pre_silu.c,
            pre_silu.h,
            pre_silu.w,
            &self.conv_out,
            cfg.tile_px,
        )
    }
}

impl MidBlock {
    fn forward(&self, x: &Map) -> VaeResult<Map> {
        let r0 = self.resnet0.forward(&x.data, x.h, x.w)?;
        let cur = Map::new(r0, self.resnet0.out_ch, x.h, x.w);
        let attn = self.attention.forward(&cur.data, cur.h, cur.w)?;
        let cur = Map::new(attn, cur.c, cur.h, cur.w);
        let r1 = self.resnet1.forward(&cur.data, cur.h, cur.w)?;
        Ok(Map::new(r1, self.resnet1.out_ch, cur.h, cur.w))
    }
}

impl UpBlock {
    fn forward(&self, x: &Map) -> VaeResult<Map> {
        let mut cur = x.clone();
        for resnet in &self.resnets {
            let data = resnet.forward(&cur.data, cur.h, cur.w)?;
            cur = Map::new(data, resnet.out_ch, cur.h, cur.w);
        }
        if let Some(conv) = self.upsampler.as_ref() {
            let upsampled = upsample_nearest2x(&cur.data, cur.c, cur.h, cur.w)?;
            let out = conv.forward(&upsampled.data, upsampled.h, upsampled.w)?;
            cur = Map::new(out.data, conv.out_ch, out.h, out.w);
        }
        Ok(cur)
    }

    /// Memory-bounded variant of [`Self::forward`]: each resnet and the upsampler
    /// convolution run with their im2col scratch capped at `max_tile_pixels`
    /// output pixels ([`crate::vae::resnet::ResnetBlock2D::forward_tiled`] /
    /// [`Conv2d::forward_tiled`]). Activation planes stay whole, so the result is
    /// bit-identical to [`Self::forward`]. Used by the `AfterMid` tiling path.
    fn forward_tiled(&self, x: &Map, max_tile_pixels: usize) -> VaeResult<Map> {
        let mut cur = x.clone();
        for resnet in &self.resnets {
            let data = resnet.forward_tiled(&cur.data, cur.h, cur.w, max_tile_pixels)?;
            cur = Map::new(data, resnet.out_ch, cur.h, cur.w);
        }
        if let Some(conv) = self.upsampler.as_ref() {
            let upsampled = upsample_nearest2x(&cur.data, cur.c, cur.h, cur.w)?;
            let out =
                conv.forward_tiled(&upsampled.data, upsampled.h, upsampled.w, max_tile_pixels)?;
            cur = Map::new(out.data, conv.out_ch, out.h, out.w);
        }
        Ok(cur)
    }
}

// ── builders ──────────────────────────────────────────────────────────────

/// Load a conv (`<prefix>.weight` `[out,kH,kW,in]`, `<prefix>.bias` `[out]`).
fn load_conv(w: &VaeWeights, prefix: &str, pad: usize) -> VaeResult<Conv2d> {
    let weight = w.get(&format!("{prefix}.weight"))?;
    let bias = w.get(&format!("{prefix}.bias"))?;
    Conv2d::from_weights(&weight.data, &weight.shape, &bias.data, pad)
}

/// Load a Linear (`<prefix>.weight` `[out,in]`, `<prefix>.bias` `[out]`).
fn load_linear(w: &VaeWeights, prefix: &str) -> VaeResult<Linear> {
    let weight = w.get(&format!("{prefix}.weight"))?;
    let bias = w.get(&format!("{prefix}.bias"))?;
    Linear::from_weights(&weight.data, &weight.shape, &bias.data)
}

/// Load a GroupNorm (`<prefix>.weight`/`.bias` `[C]`).
fn load_group_norm(w: &VaeWeights, prefix: &str) -> VaeResult<GroupNorm> {
    let weight = w.vec1(&format!("{prefix}.weight"))?;
    let bias = w.vec1(&format!("{prefix}.bias"))?;
    GroupNorm::new(&weight.data, &bias.data, NUM_GROUPS, GN_EPS)
}

/// Load a resnet block (`<prefix>.{norm1,conv1,norm2,conv2[,conv_shortcut]}`).
fn load_resnet(w: &VaeWeights, prefix: &str) -> VaeResult<ResnetBlock2D> {
    let norm1 = load_group_norm(w, &format!("{prefix}.norm1"))?;
    let conv1 = load_conv(w, &format!("{prefix}.conv1"), 1)?;
    let norm2 = load_group_norm(w, &format!("{prefix}.norm2"))?;
    let conv2 = load_conv(w, &format!("{prefix}.conv2"), 1)?;
    let in_ch = conv1.in_ch;
    let out_ch = conv2.out_ch;
    // 1×1 shortcut present only when in != out channels.
    let conv_shortcut = if in_ch != out_ch {
        Some(load_conv(w, &format!("{prefix}.conv_shortcut"), 0)?)
    } else {
        None
    };
    Ok(ResnetBlock2D {
        norm1,
        conv1,
        norm2,
        conv2,
        conv_shortcut,
        in_ch,
        out_ch,
    })
}

/// Load the mid block (`<prefix>.resnets.{0,1}`, `<prefix>.attentions.0`).
fn load_mid_block(w: &VaeWeights, prefix: &str) -> VaeResult<MidBlock> {
    let resnet0 = load_resnet(w, &format!("{prefix}.resnets.0"))?;
    let resnet1 = load_resnet(w, &format!("{prefix}.resnets.1"))?;
    let attention = load_attention(w, &format!("{prefix}.attentions.0"))?;
    Ok(MidBlock {
        resnet0,
        attention,
        resnet1,
    })
}

/// Load an attention block (`<prefix>.{group_norm,to_q,to_k,to_v,to_out}`).
fn load_attention(w: &VaeWeights, prefix: &str) -> VaeResult<AttentionBlock> {
    let group_norm = load_group_norm(w, &format!("{prefix}.group_norm"))?;
    let to_q = load_linear(w, &format!("{prefix}.to_q"))?;
    let to_k = load_linear(w, &format!("{prefix}.to_k"))?;
    let to_v = load_linear(w, &format!("{prefix}.to_v"))?;
    let to_out = load_linear(w, &format!("{prefix}.to_out"))?;
    let channels = group_norm.channels;
    Ok(AttentionBlock {
        group_norm,
        to_q,
        to_k,
        to_v,
        to_out,
        channels,
    })
}

/// Load an up block (`<prefix>.resnets.{0..num_layers}`, optional
/// `<prefix>.upsamplers.0.conv`).
fn load_up_block(
    w: &VaeWeights,
    prefix: &str,
    num_layers: usize,
    has_upsample: bool,
) -> VaeResult<UpBlock> {
    let mut resnets = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        resnets.push(load_resnet(w, &format!("{prefix}.resnets.{i}"))?);
    }
    let upsampler = if has_upsample {
        Some(load_conv(w, &format!("{prefix}.upsamplers.0.conv"), 1)?)
    } else {
        None
    };
    Ok(UpBlock { resnets, upsampler })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny same-channel ResnetBlock2D (no shortcut) with `groups`-group
    /// GroupNorms, for the up-block tiling parity check.
    fn tiny_resnet(ch: usize, groups: usize, seed: f32) -> ResnetBlock2D {
        let gn = |off: f32| {
            let weight: Vec<f32> = (0..ch).map(|i| 1.0 + 0.02 * (i as f32 + off)).collect();
            let bias: Vec<f32> = (0..ch).map(|i| -0.01 * (i as f32 + off)).collect();
            GroupNorm::new(&weight, &bias, groups, GN_EPS).expect("groupnorm")
        };
        let conv = |off: f32| {
            let weight: Vec<f32> = (0..ch * 3 * 3 * ch)
                .map(|n| ((n as f32 + off) * 0.011).sin() * 0.25)
                .collect();
            let bias: Vec<f32> = (0..ch).map(|n| 0.005 * n as f32).collect();
            Conv2d::from_weights(&weight, &[ch, 3, 3, ch], &bias, 1).expect("conv")
        };
        ResnetBlock2D {
            norm1: gn(seed),
            conv1: conv(seed + 1.0),
            norm2: gn(seed + 2.0),
            conv2: conv(seed + 3.0),
            conv_shortcut: None,
            in_ch: ch,
            out_ch: ch,
        }
    }

    /// `UpBlock::forward_tiled` (the `AfterMid` per-block lever) must give a
    /// bit-identical result at every pixel budget — including through the
    /// nearest-2× upsampler convolution. The whole-plane `forward_tiled` reference
    /// runs the same CPU im2col + GEMM as the untiled `UpBlock::forward` (which
    /// routes its convs to the GPU under `native-cuda`), so the row-blocking
    /// contract is validated CPU-side, independent of `OXI_VAE_GPU`.
    #[test]
    fn up_block_forward_tiled_matches_forward() {
        let ch = 32usize; // divisible by NUM_GROUPS
        let upsampler_weight: Vec<f32> = (0..ch * 3 * 3 * ch)
            .map(|n| (n as f32 * 0.007).cos() * 0.2)
            .collect();
        let upsampler =
            Conv2d::from_weights(&upsampler_weight, &[ch, 3, 3, ch], &vec![0.0f32; ch], 1)
                .expect("upsampler");
        let block = UpBlock {
            resnets: vec![
                tiny_resnet(ch, NUM_GROUPS, 0.0),
                tiny_resnet(ch, NUM_GROUPS, 10.0),
            ],
            upsampler: Some(upsampler),
        };
        let (h, w) = (5usize, 4usize);
        let data: Vec<f32> = (0..ch * h * w)
            .map(|i| (i as f32 * 0.019).sin() * 0.9 + 0.05)
            .collect();
        let x = Map::new(data, ch, h, w);
        // Whole-plane reference: a budget past the upsampled 2h×2w resolution runs
        // every conv in a single chunk (== the untiled CPU im2col path).
        let reference = block
            .forward_tiled(&x, (2 * h) * (2 * w))
            .expect("whole-plane reference");
        for &budget in &[1usize, w, 4 * w, (2 * h) * (2 * w)] {
            let tiled = block.forward_tiled(&x, budget).expect("forward_tiled");
            assert_eq!(tiled.c, reference.c);
            assert_eq!((tiled.h, tiled.w), (reference.h, reference.w));
            assert_eq!(
                tiled.data, reference.data,
                "UpBlock forward_tiled(budget={budget}) diverged from whole-plane"
            );
        }
    }
}
