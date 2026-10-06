//! The Qwen3-VL vision tower of a Bonsai 2 `mmproj` on the Metal GPU
//! (bonsai2-design.md §6): every ViT block, the post-norm and the merger in
//! one command buffer per image, every weight and activation resident.
//!
//! # What runs where
//!
//! The host normalises the pixels, gathers the 16 × 16 × 3 patches in
//! 2 × 2 merge-window order, resizes the learned position grid to the
//! image's patch grid and builds the 2-D rotary rows — cheap, per-image
//! work that the CPU tower does the same way — and hands them over
//! ([`VisionGpuModel::encode`]). The device runs the rest: the patch GEMM
//! (`f32` kernel, summed temporal slices, bias) plus the position rows, then
//! per block LayerNorm → QKV GEMM → 2-D RoPE and head-major pack →
//! flash attention → output GEMM with the residual add → LayerNorm → up
//! GEMM with the tanh GELU → down GEMM with the residual add, then
//! `v.post_ln` and the merger (`mm.0` + GELU, `mm.2`) over the merged rows.
//! The kernels are `kernel_sources::vision`'s and the DiT's flash attention
//! (`joint_attention_flash_f32`).
//!
//! # Weights
//!
//! Every matrix stays in the file's own storage, so no weight is rounded
//! ([`VisionWeightFormat`]): the `Q8_0` blocks as they are, read exactly by
//! `vit_gemm_q80` (a per-block `f32` partial against the int8 values,
//! scaled by the block's `f16` scale — the value of the dequantised
//! weights `d · q`); the `F16` matrices as `f16` (`vit_gemm_f16w`); any
//! other storage as `f32` (`vit_gemm_f32w`). For Bonsai 2 (`Q8_0` except
//! the `F16` `ffn_down`) that is 0.61 GB of weights against the CPU
//! tower's 1.84 GB of `f32`. Every GEMM adds a fresh per-slice `f32`
//! partial into its accumulator, which keeps a long `K` (the 4304-wide
//! FFN) at the CPU GEMM's own accuracy (about 1.7e-7 relative against
//! `f64`, where one running chain of products reached 1e-6).
//!
//! Measured on the M3 at a 768 × 768 image's shapes, the three run at
//! about the same rate — 1.03 s of GEMMs per image with the exact `Q8_0`
//! blocks, 1.04 s with every weight rounded to `f16`, 1.11 s with `f32`
//! weights (`tests::exact_q8_0_weights_stay_close_to_f16_speed_at_the_vit_shapes`)
//! — so the blocks, the smallest in memory, are held as they are: rounding
//! them to `f16` would cost the real projector a per-row relative error of
//! about 1.5e-3 against the CPU tower, where the exact blocks stay within
//! about 3e-5. The vectors, the norms and the patch kernel stay `f32`.
//! [`VisionGpuBuilder`] writes each tensor into its device buffer as it is
//! loaded, so no whole-tower `f32` copy ever exists.
//!
//! # Attention bound
//!
//! The flash kernel loops over key tiles, so its sequence length is a
//! validation constant, not a kernel limit: the tower accepts up to
//! [`VIT_ATTN_MAX_SEQ`] patches (the largest image budget, 16 384 merged
//! tokens, × 4) and checks its own bounds rather than the DiT's. The
//! activation scratch is sized for the budget a tower is built with (1024
//! merged tokens by default, 4096 patches).
//!
//! # Sessions
//!
//! A model owns its own [`MetalGraph`] session (its own command queue);
//! concurrent callers serialise on `&mut self`, which the model crate wraps
//! in a mutex so every engine replica shares one tower. Beside a Metal
//! hybrid engine that is a second live session in the process (the
//! runner's is the first), counted by [`MetalGraph::live_session_count`].
//! [`MetalGraph::max_sessions`] sizes engine pools but is not enforced when
//! a session opens, so opening the tower's session past it is reported
//! ([`session_ceiling_warning`], logged once per tower) rather than refused.

use std::sync::Arc;

use metal::{Buffer, ComputePipelineState, MTLResourceOptions};

use crate::gpu_backend::metal_graph::{alloc_buf, MetalGraph, MetalGraphError};

mod encode;

#[cfg(test)]
mod tests;

/// Most patches one encode takes: the largest image budget the projector's
/// loader accepts (16 384 merged tokens), four patches each — every index
/// the kernels form stays inside `u32` at this length. Well above the DiT's
/// own flash-attention cap, which is a validation constant of that caller
/// (the kernel loops over key tiles).
pub const VIT_ATTN_MAX_SEQ: usize = 65_536;

/// Largest head width the flash kernel stages (`FA_BK × 128` floats of
/// threadgroup memory).
pub const VIT_ATTN_MAX_HEAD_DIM: usize = 128;

/// Threads of one LayerNorm row.
const NORM_THREADS: u64 = 256;
/// Output-tile edge of the GEMMs.
const GEMM_TILE: usize = 64;
/// Threads per GEMM threadgroup.
const GEMM_THREADS: u64 = 128;
/// Epilogue flag: add the bias.
const FLAG_BIAS: u32 = 1;
/// Epilogue flag: apply the tanh GELU.
const FLAG_GELU: u32 = 2;
/// Epilogue flag: add into the output (the residual add).
const FLAG_ACCUMULATE: u32 = 4;

/// How the device holds one of the tower's matrices (see the module docs,
/// "Weights"): the file's own storage, never a rounded copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisionWeightFormat {
    /// The file's `Q8_0` blocks (an `f16` scale, then 32 int8), read exactly
    /// by `vit_gemm_q80`; the matrix's inner dimension must be a multiple of
    /// 32.
    Q8_0,
    /// `f16` values, read by `vit_gemm_f16w` — for tensors the file
    /// **stores** as `F16` only, whose values `f16` holds exactly. It is
    /// never a rounding of `f32` (or `Q8_0`) weights on the product path:
    /// the model crate's loader (`vision::metal`) maps `Q8_0` → `Q8_0`,
    /// `F16` → `F16` and every other storage → `F32`. Holding other
    /// matrices as `F16` (every weight rounded to `f16`) exists only as the
    /// speed comparison of this module's tests.
    F16,
    /// `f32` values — any other storage — read by `vit_gemm_f32w`.
    F32,
}

impl VisionWeightFormat {
    /// Device bytes of `values` weights in this format.
    #[must_use]
    pub const fn bytes(self, values: usize) -> usize {
        match self {
            Self::Q8_0 => values / 32 * 34,
            Self::F16 => values * 2,
            Self::F32 => values * 4,
        }
    }
}

/// The storage of the tower's six matrix kinds; every block's matrix of a
/// kind shares one.
///
/// Each entry is the file's own storage of that matrix kind (`Q8_0` →
/// [`VisionWeightFormat::Q8_0`], `F16` → [`VisionWeightFormat::F16`],
/// anything else → [`VisionWeightFormat::F32`]): an `F16` entry is for
/// `F16`-stored tensors only, never a rounding of `f32` weights on the
/// product path. [`VisionMatrixFormats::uniform`] with `F16` over matrices
/// stored otherwise is a test benchmark, not a load configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionMatrixFormats {
    /// `attn_qkv` (inner dimension `hidden`).
    pub qkv: VisionWeightFormat,
    /// `attn_out` (inner dimension `hidden`).
    pub out: VisionWeightFormat,
    /// `ffn_up` (inner dimension `hidden`).
    pub up: VisionWeightFormat,
    /// `ffn_down` (inner dimension `ffn`).
    pub down: VisionWeightFormat,
    /// `mm.0` (inner dimension `4 · hidden`).
    pub mm0: VisionWeightFormat,
    /// `mm.2` (inner dimension `merger_hidden`).
    pub mm2: VisionWeightFormat,
}

impl VisionMatrixFormats {
    /// Every matrix in `format`.
    #[must_use]
    pub const fn uniform(format: VisionWeightFormat) -> Self {
        Self {
            qkv: format,
            out: format,
            up: format,
            down: format,
            mm0: format,
            mm2: format,
        }
    }
}

/// Geometry of a vision tower on the device.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionGpuConfig {
    /// ViT width (1152).
    pub hidden: usize,
    /// Attention heads (16).
    pub heads: usize,
    /// Per-head width, `hidden / heads` (72).
    pub head_dim: usize,
    /// MLP width (4304).
    pub ffn: usize,
    /// ViT blocks (27).
    pub blocks: usize,
    /// LayerNorm epsilon.
    pub eps: f32,
    /// Values per patch, `3 · patch²` (768).
    pub patch_len: usize,
    /// Output width of `mm.0` (4608).
    pub merger_hidden: usize,
    /// Output width of `mm.2` — the language model's hidden size (5120).
    pub projection_dim: usize,
    /// Most patches one encode takes (a multiple of 4, the merge window).
    pub max_patches: usize,
    /// How the device holds each matrix kind.
    pub formats: VisionMatrixFormats,
}

impl VisionGpuConfig {
    /// Width of one merged token before the merger, `4 · hidden`.
    #[must_use]
    pub const fn merged_width(&self) -> usize {
        4 * self.hidden
    }

    /// Reject a geometry the kernels cannot serve, naming the constraint.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for the first violated
    /// constraint.
    pub fn validate(&self) -> Result<(), MetalGraphError> {
        let bad = |what: String| {
            Err(MetalGraphError::InvalidDimensions(format!(
                "vision GPU: {what}"
            )))
        };
        for (name, value) in [
            ("hidden", self.hidden),
            ("heads", self.heads),
            ("ffn", self.ffn),
            ("blocks", self.blocks),
            ("patch_len", self.patch_len),
            ("merger_hidden", self.merger_hidden),
            ("projection_dim", self.projection_dim),
        ] {
            if value == 0 {
                return bad(format!("{name} must be non-zero"));
            }
        }
        if self.heads * self.head_dim != self.hidden {
            return bad(format!(
                "heads {} x head_dim {} must equal hidden {}",
                self.heads, self.head_dim, self.hidden
            ));
        }
        if !self.head_dim.is_multiple_of(8) || self.head_dim > VIT_ATTN_MAX_HEAD_DIM {
            return bad(format!(
                "head_dim {} must be a multiple of 8 no larger than {VIT_ATTN_MAX_HEAD_DIM} (the \
                 flash attention's tile)",
                self.head_dim
            ));
        }
        // Every GEMM reads its activations a float4 at a time.
        for (name, value) in [
            ("hidden", self.hidden),
            ("ffn", self.ffn),
            ("patch_len", self.patch_len),
            ("merger_hidden", self.merger_hidden),
        ] {
            if !value.is_multiple_of(4) {
                return bad(format!(
                    "{name} {value} must be a multiple of 4 (a GEMM's inner dimension)"
                ));
            }
        }
        if self.max_patches < 4
            || !self.max_patches.is_multiple_of(4)
            || self.max_patches > VIT_ATTN_MAX_SEQ
        {
            return bad(format!(
                "max_patches {} must be a multiple of 4 in 4..={VIT_ATTN_MAX_SEQ}",
                self.max_patches
            ));
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return bad(format!("eps {} must be finite and positive", self.eps));
        }
        let f = &self.formats;
        for (name, format, inner) in [
            ("attn_qkv", f.qkv, self.hidden),
            ("attn_out", f.out, self.hidden),
            ("ffn_up", f.up, self.hidden),
            ("ffn_down", f.down, self.ffn),
            ("mm.0", f.mm0, self.merged_width()),
            ("mm.2", f.mm2, self.merger_hidden),
        ] {
            if format == VisionWeightFormat::Q8_0 && !inner.is_multiple_of(32) {
                return bad(format!(
                    "{name} is held as Q8_0 blocks, but its inner dimension {inner} is not a \
                     multiple of 32"
                ));
            }
        }
        Ok(())
    }
}

/// Bytes a [`VisionGpuModel`] keeps resident, computed without a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionGpuFootprint {
    /// Every weight: the matrices in their [`VisionWeightFormat`]s and the
    /// `f32` vectors and patch kernel.
    pub weight_bytes: u64,
    /// The activation scratch at `max_patches`.
    pub scratch_bytes: u64,
}

impl VisionGpuFootprint {
    /// Everything the model allocates.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.weight_bytes.saturating_add(self.scratch_bytes)
    }
}

/// One weight of the tower, by role (`block` indexes the ViT blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisionTensor {
    /// `v.blk.N.ln1.weight`, `[hidden]`.
    Ln1Weight(usize),
    /// `v.blk.N.ln1.bias`, `[hidden]`.
    Ln1Bias(usize),
    /// `v.blk.N.attn_qkv.weight`, `[3 · hidden][hidden]`.
    QkvWeight(usize),
    /// `v.blk.N.attn_qkv.bias`, `[3 · hidden]`.
    QkvBias(usize),
    /// `v.blk.N.attn_out.weight`, `[hidden][hidden]`.
    OutWeight(usize),
    /// `v.blk.N.attn_out.bias`, `[hidden]`.
    OutBias(usize),
    /// `v.blk.N.ln2.weight`, `[hidden]`.
    Ln2Weight(usize),
    /// `v.blk.N.ln2.bias`, `[hidden]`.
    Ln2Bias(usize),
    /// `v.blk.N.ffn_up.weight`, `[ffn][hidden]`.
    UpWeight(usize),
    /// `v.blk.N.ffn_up.bias`, `[ffn]`.
    UpBias(usize),
    /// `v.blk.N.ffn_down.weight`, `[hidden][ffn]`.
    DownWeight(usize),
    /// `v.blk.N.ffn_down.bias`, `[hidden]`.
    DownBias(usize),
    /// The patch kernel (both temporal slices summed), `[hidden][patch_len]`.
    PatchKernel,
    /// `v.patch_embd.bias`, `[hidden]`.
    PatchBias,
    /// `v.post_ln.weight`, `[hidden]`.
    PostLnWeight,
    /// `v.post_ln.bias`, `[hidden]`.
    PostLnBias,
    /// `mm.0.weight`, `[merger_hidden][4 · hidden]`.
    Mm0Weight,
    /// `mm.0.bias`, `[merger_hidden]`.
    Mm0Bias,
    /// `mm.2.weight`, `[projection_dim][merger_hidden]`.
    Mm2Weight,
    /// `mm.2.bias`, `[projection_dim]`.
    Mm2Bias,
}

impl VisionTensor {
    /// Values the tensor holds under `cfg`, and how the device keeps them:
    /// a matrix in its kind's [`VisionWeightFormat`], everything else (the
    /// vectors and the patch kernel) as `f32`.
    fn shape(self, cfg: &VisionGpuConfig) -> (usize, VisionWeightFormat) {
        let (h, f) = (cfg.hidden, cfg.ffn);
        let fm = &cfg.formats;
        let vector = VisionWeightFormat::F32;
        match self {
            Self::Ln1Weight(_)
            | Self::Ln1Bias(_)
            | Self::OutBias(_)
            | Self::Ln2Weight(_)
            | Self::Ln2Bias(_)
            | Self::DownBias(_)
            | Self::PatchBias
            | Self::PostLnWeight
            | Self::PostLnBias => (h, vector),
            Self::QkvWeight(_) => (3 * h * h, fm.qkv),
            Self::QkvBias(_) => (3 * h, vector),
            Self::OutWeight(_) => (h * h, fm.out),
            Self::UpWeight(_) => (f * h, fm.up),
            Self::UpBias(_) => (f, vector),
            Self::DownWeight(_) => (h * f, fm.down),
            Self::PatchKernel => (h * cfg.patch_len, vector),
            Self::Mm0Weight => (cfg.merger_hidden * cfg.merged_width(), fm.mm0),
            Self::Mm0Bias => (cfg.merger_hidden, vector),
            Self::Mm2Weight => (cfg.projection_dim * cfg.merger_hidden, fm.mm2),
            Self::Mm2Bias => (cfg.projection_dim, vector),
        }
    }

    /// Every tensor of a tower with `blocks` ViT blocks, in load order.
    #[must_use]
    pub fn all(blocks: usize) -> Vec<Self> {
        let mut out = Vec::with_capacity(12 * blocks + 8);
        for b in 0..blocks {
            out.extend([
                Self::Ln1Weight(b),
                Self::Ln1Bias(b),
                Self::QkvWeight(b),
                Self::QkvBias(b),
                Self::OutWeight(b),
                Self::OutBias(b),
                Self::Ln2Weight(b),
                Self::Ln2Bias(b),
                Self::UpWeight(b),
                Self::UpBias(b),
                Self::DownWeight(b),
                Self::DownBias(b),
            ]);
        }
        out.extend([
            Self::PatchKernel,
            Self::PatchBias,
            Self::PostLnWeight,
            Self::PostLnBias,
            Self::Mm0Weight,
            Self::Mm0Bias,
            Self::Mm2Weight,
            Self::Mm2Bias,
        ]);
        out
    }

    fn block(self) -> Option<usize> {
        match self {
            Self::Ln1Weight(b)
            | Self::Ln1Bias(b)
            | Self::QkvWeight(b)
            | Self::QkvBias(b)
            | Self::OutWeight(b)
            | Self::OutBias(b)
            | Self::Ln2Weight(b)
            | Self::Ln2Bias(b)
            | Self::UpWeight(b)
            | Self::UpBias(b)
            | Self::DownWeight(b)
            | Self::DownBias(b) => Some(b),
            _ => None,
        }
    }
}

/// One weight on the device: its buffer and how it holds the values.
#[derive(Clone)]
struct DeviceTensor {
    buffer: Buffer,
    format: VisionWeightFormat,
    set: bool,
}

/// One matrix of the built tower: its buffer and the GEMM that reads it.
struct GpuMatrix {
    buffer: Buffer,
    format: VisionWeightFormat,
}

/// Pipelines of the combined library the tower dispatches.
struct VitPipelines {
    layer_norm: ComputePipelineState,
    rope_pack: ComputePipelineState,
    add_rows: ComputePipelineState,
    gemm_f16w: ComputePipelineState,
    gemm_f32w: ComputePipelineState,
    gemm_q80: ComputePipelineState,
}

impl VitPipelines {
    /// The GEMM that reads weights held in `format`.
    fn gemm_for(&self, format: VisionWeightFormat) -> &ComputePipelineState {
        match format {
            VisionWeightFormat::Q8_0 => &self.gemm_q80,
            VisionWeightFormat::F16 => &self.gemm_f16w,
            VisionWeightFormat::F32 => &self.gemm_f32w,
        }
    }
}

/// Activation buffers for up to `max_patches` patches.
struct VitScratch {
    patches: Buffer,
    pos: Buffer,
    rope_cos: Buffer,
    rope_sin: Buffer,
    x: Buffer,
    normed: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    up: Buffer,
    mid: Buffer,
    out: Buffer,
}

impl VitScratch {
    /// Floats one patch occupies across every scratch buffer.
    fn floats_per_patch(cfg: &VisionGpuConfig) -> usize {
        cfg.patch_len // patches
            + cfg.hidden // pos
            + cfg.head_dim // rope cos + sin (head_dim / 2 each)
            + 3 * cfg.hidden // x, normed, attn
            + 3 * cfg.hidden // qkv
            + 3 * cfg.hidden // q, k, v head-major
            + cfg.ffn // up
            + (cfg.merger_hidden + cfg.projection_dim).div_ceil(4) // mid, out per merged row
    }

    fn allocate(graph: &MetalGraph, cfg: &VisionGpuConfig) -> Result<Self, MetalGraphError> {
        let n = cfg.max_patches;
        let f = |floats: usize| -> Result<Buffer, MetalGraphError> {
            alloc_buf(
                &graph.device,
                (floats.max(1) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let merged = n / 4;
        Ok(Self {
            patches: f(n * cfg.patch_len)?,
            pos: f(n * cfg.hidden)?,
            rope_cos: f(n * cfg.head_dim / 2)?,
            rope_sin: f(n * cfg.head_dim / 2)?,
            x: f(n * cfg.hidden)?,
            normed: f(n * cfg.hidden)?,
            qkv: f(n * 3 * cfg.hidden)?,
            q: f(n * cfg.hidden)?,
            k: f(n * cfg.hidden)?,
            v: f(n * cfg.hidden)?,
            attn: f(n * cfg.hidden)?,
            up: f(n * cfg.ffn)?,
            mid: f(merged * cfg.merger_hidden)?,
            out: f(merged * cfg.projection_dim)?,
        })
    }
}

/// Allocates every weight buffer of a tower and fills them one tensor at a
/// time — [`Self::set`] for values, [`Self::set_q8_0`] for a matrix held as
/// its file's `Q8_0` blocks — straight into their device buffers;
/// [`Self::build`] checks that every tensor was set and allocates the
/// scratch.
pub struct VisionGpuBuilder {
    graph: Arc<MetalGraph>,
    cfg: VisionGpuConfig,
    tensors: std::collections::HashMap<VisionTensor, DeviceTensor>,
}

impl std::fmt::Debug for VisionGpuBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionGpuBuilder")
            .field("config", &self.cfg)
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl VisionGpuBuilder {
    /// Validate `cfg`, open a Metal session on the process-shared device and
    /// allocate every weight buffer.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a geometry the kernels do
    /// not serve; [`MetalGraphError::DeviceNotFound`] without a device;
    /// allocation errors.
    pub fn new(cfg: VisionGpuConfig) -> Result<Self, MetalGraphError> {
        cfg.validate()?;
        let graph = MetalGraph::new_session()?;
        if let Some(warning) =
            session_ceiling_warning(MetalGraph::live_session_count(), MetalGraph::max_sessions())
        {
            tracing::warn!(
                live_sessions = MetalGraph::live_session_count(),
                max_sessions = MetalGraph::max_sessions(),
                "{warning}"
            );
        }
        let mut tensors = std::collections::HashMap::new();
        for tensor in VisionTensor::all(cfg.blocks) {
            let (len, format) = tensor.shape(&cfg);
            let bytes = format.bytes(len).max(4) as u64;
            let buffer = alloc_buf(&graph.device, bytes, MTLResourceOptions::StorageModeShared)?;
            tensors.insert(
                tensor,
                DeviceTensor {
                    buffer,
                    format,
                    set: false,
                },
            );
        }
        Ok(Self {
            graph,
            cfg,
            tensors,
        })
    }

    /// The geometry being built.
    #[must_use]
    pub fn config(&self) -> &VisionGpuConfig {
        &self.cfg
    }

    /// The device slot of `tensor`, after checking its block index.
    fn slot(&mut self, tensor: VisionTensor) -> Result<&mut DeviceTensor, MetalGraphError> {
        if let Some(b) = tensor.block() {
            if b >= self.cfg.blocks {
                return Err(MetalGraphError::InvalidDimensions(format!(
                    "vision GPU: {tensor:?} past the {} blocks",
                    self.cfg.blocks
                )));
            }
        }
        self.tensors.get_mut(&tensor).ok_or_else(|| {
            MetalGraphError::InvalidDimensions(format!("vision GPU: no buffer for {tensor:?}"))
        })
    }

    /// Write `values` (row-major, `f32`) into `tensor`'s device buffer: as
    /// they are for an `f32` tensor, as `f16` for a matrix held in
    /// [`VisionWeightFormat::F16`] (exact for the file's `F16` values, which
    /// is what that format is for).
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a block past the config, a
    /// length that disagrees with the tensor's shape, or a matrix held as
    /// `Q8_0` blocks (set with [`Self::set_q8_0`]).
    pub fn set(&mut self, tensor: VisionTensor, values: &[f32]) -> Result<(), MetalGraphError> {
        let (len, format) = tensor.shape(&self.cfg);
        if format == VisionWeightFormat::Q8_0 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "vision GPU: {tensor:?} is held as Q8_0 blocks; set it with set_q8_0"
            )));
        }
        if values.len() != len {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "vision GPU: {tensor:?} has {} values, expected {len}",
                values.len()
            )));
        }
        let slot = self.slot(tensor)?;
        if format == VisionWeightFormat::F16 {
            // SAFETY: the buffer was allocated for exactly `len` f16 values
            // (shared storage, non-null contents) and no GPU work uses it
            // yet; `u16` and `f16` bits have no invalid patterns.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(slot.buffer.contents().cast::<u16>(), len)
            };
            for (d, &v) in dst.iter_mut().zip(values) {
                *d = half::f16::from_f32(v).to_bits();
            }
        } else {
            // SAFETY: as above, for `len` f32 values; the ranges cannot
            // overlap (`values` is a host slice).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    slot.buffer.contents().cast::<f32>(),
                    len,
                );
            }
        }
        slot.set = true;
        Ok(())
    }

    /// Copy `blocks` — the matrix's `Q8_0` blocks exactly as the file holds
    /// them, rows of `inner / 32` 34-byte blocks — into `tensor`'s device
    /// buffer.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a block past the config, a
    /// tensor not held as [`VisionWeightFormat::Q8_0`], or a byte count
    /// other than the matrix's.
    pub fn set_q8_0(&mut self, tensor: VisionTensor, blocks: &[u8]) -> Result<(), MetalGraphError> {
        let (len, format) = tensor.shape(&self.cfg);
        if format != VisionWeightFormat::Q8_0 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "vision GPU: {tensor:?} is held as {format:?}, not Q8_0 blocks"
            )));
        }
        let bytes = format.bytes(len);
        if blocks.len() != bytes {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "vision GPU: {tensor:?} has {} bytes of Q8_0 blocks, expected {bytes}",
                blocks.len()
            )));
        }
        let slot = self.slot(tensor)?;
        // SAFETY: the buffer was allocated for exactly `bytes` bytes (shared
        // storage, non-null contents), no GPU work uses it yet, and the
        // ranges cannot overlap (`blocks` is a host slice).
        unsafe {
            std::ptr::copy_nonoverlapping(
                blocks.as_ptr(),
                slot.buffer.contents().cast::<u8>(),
                bytes,
            );
        }
        slot.set = true;
        Ok(())
    }

    /// Finish: every tensor must have been set; the scratch is allocated
    /// for `max_patches`.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] naming the first tensor never
    /// set; allocation and pipeline errors.
    pub fn build(self) -> Result<VisionGpuModel, MetalGraphError> {
        let cfg = self.cfg;
        for tensor in VisionTensor::all(cfg.blocks) {
            match self.tensors.get(&tensor) {
                Some(slot) if slot.set => {}
                _ => {
                    return Err(MetalGraphError::InvalidDimensions(format!(
                        "vision GPU: {tensor:?} was never set"
                    )))
                }
            }
        }
        let graph = self.graph;
        let tensors = self.tensors;
        let take = |tensor: VisionTensor| -> Result<Buffer, MetalGraphError> {
            tensors
                .get(&tensor)
                .map(|slot| slot.buffer.clone())
                .ok_or_else(|| {
                    MetalGraphError::InvalidDimensions(format!(
                        "vision GPU: no buffer for {tensor:?}"
                    ))
                })
        };
        let matrix = |tensor: VisionTensor| -> Result<GpuMatrix, MetalGraphError> {
            tensors
                .get(&tensor)
                .map(|slot| GpuMatrix {
                    buffer: slot.buffer.clone(),
                    format: slot.format,
                })
                .ok_or_else(|| {
                    MetalGraphError::InvalidDimensions(format!(
                        "vision GPU: no buffer for {tensor:?}"
                    ))
                })
        };
        let mut blocks = Vec::with_capacity(cfg.blocks);
        for b in 0..cfg.blocks {
            blocks.push(GpuBlock {
                ln1_w: take(VisionTensor::Ln1Weight(b))?,
                ln1_b: take(VisionTensor::Ln1Bias(b))?,
                qkv_w: matrix(VisionTensor::QkvWeight(b))?,
                qkv_b: take(VisionTensor::QkvBias(b))?,
                out_w: matrix(VisionTensor::OutWeight(b))?,
                out_b: take(VisionTensor::OutBias(b))?,
                ln2_w: take(VisionTensor::Ln2Weight(b))?,
                ln2_b: take(VisionTensor::Ln2Bias(b))?,
                up_w: matrix(VisionTensor::UpWeight(b))?,
                up_b: take(VisionTensor::UpBias(b))?,
                down_w: matrix(VisionTensor::DownWeight(b))?,
                down_b: take(VisionTensor::DownBias(b))?,
            });
        }
        let weight_bytes = tensors.values().map(|t| t.buffer.length()).sum();
        let pipes = VitPipelines {
            layer_norm: graph.pipeline_for("vit_layer_norm")?,
            rope_pack: graph.pipeline_for("vit_qkv_rope_pack")?,
            add_rows: graph.pipeline_for("vit_add_rows")?,
            gemm_f16w: graph.pipeline_for("vit_gemm_f16w")?,
            gemm_f32w: graph.pipeline_for("vit_gemm_f32w")?,
            gemm_q80: graph.pipeline_for("vit_gemm_q80")?,
        };
        let scratch = VitScratch::allocate(&graph, &cfg)?;
        Ok(VisionGpuModel {
            patch_kernel: take(VisionTensor::PatchKernel)?,
            patch_bias: take(VisionTensor::PatchBias)?,
            post_ln_w: take(VisionTensor::PostLnWeight)?,
            post_ln_b: take(VisionTensor::PostLnBias)?,
            mm0_w: matrix(VisionTensor::Mm0Weight)?,
            mm0_b: take(VisionTensor::Mm0Bias)?,
            mm2_w: matrix(VisionTensor::Mm2Weight)?,
            mm2_b: take(VisionTensor::Mm2Bias)?,
            graph,
            cfg,
            pipes,
            blocks,
            scratch,
            weight_bytes,
            last_gpu_seconds: 0.0,
        })
    }
}

/// One ViT block's weights on the device.
struct GpuBlock {
    ln1_w: Buffer,
    ln1_b: Buffer,
    qkv_w: GpuMatrix,
    qkv_b: Buffer,
    out_w: GpuMatrix,
    out_b: Buffer,
    ln2_w: Buffer,
    ln2_b: Buffer,
    up_w: GpuMatrix,
    up_b: Buffer,
    down_w: GpuMatrix,
    down_b: Buffer,
}

/// A vision tower resident on the GPU (see the module docs), built by
/// [`VisionGpuBuilder`].
pub struct VisionGpuModel {
    graph: Arc<MetalGraph>,
    cfg: VisionGpuConfig,
    pipes: VitPipelines,
    blocks: Vec<GpuBlock>,
    patch_kernel: Buffer,
    patch_bias: Buffer,
    post_ln_w: Buffer,
    post_ln_b: Buffer,
    mm0_w: GpuMatrix,
    mm0_b: Buffer,
    mm2_w: GpuMatrix,
    mm2_b: Buffer,
    scratch: VitScratch,
    weight_bytes: u64,
    last_gpu_seconds: f64,
}

// Every Metal object type is `Send + Sync` in the `metal` crate and the rest
// is plain data: the compiler derives `Send`; this keeps it from silently
// disappearing (the model crate shares one tower across engine replicas).
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<VisionGpuModel>();
};

impl std::fmt::Debug for VisionGpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionGpuModel")
            .field("config", &self.cfg)
            .field("weight_bytes", &self.weight_bytes)
            .finish()
    }
}

impl VisionGpuModel {
    /// What a model of geometry `cfg` keeps resident, without a device.
    #[must_use]
    pub fn footprint(cfg: &VisionGpuConfig) -> VisionGpuFootprint {
        let weight_bytes = VisionTensor::all(cfg.blocks)
            .into_iter()
            .map(|t| {
                let (len, format) = t.shape(cfg);
                format.bytes(len).max(4) as u64
            })
            .sum();
        let scratch_bytes = (cfg.max_patches * VitScratch::floats_per_patch(cfg) * 4) as u64;
        VisionGpuFootprint {
            weight_bytes,
            scratch_bytes,
        }
    }

    /// The geometry the model was built with.
    #[must_use]
    pub fn config(&self) -> &VisionGpuConfig {
        &self.cfg
    }

    /// Bytes of weights on the device.
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Everything the model allocated: weights and scratch.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        Self::footprint(&self.cfg)
            .scratch_bytes
            .saturating_add(self.weight_bytes)
    }

    /// GPU time of the last [`Self::encode`]'s command buffer, in seconds
    /// (`0.0` before the first).
    #[must_use]
    pub fn last_gpu_seconds(&self) -> f64 {
        self.last_gpu_seconds
    }
}

/// The tower's own bounds on a flash-attention call of `seq` patches with
/// heads of `head_dim` (see the module docs, "Attention bound").
///
/// # Errors
///
/// [`MetalGraphError::InvalidDimensions`] when `seq` is zero or past
/// [`VIT_ATTN_MAX_SEQ`], or `head_dim` is not a multiple of 8 in
/// `8..=VIT_ATTN_MAX_HEAD_DIM`.
pub fn validate_vit_attention(seq: usize, head_dim: usize) -> Result<(), MetalGraphError> {
    if seq == 0 || seq > VIT_ATTN_MAX_SEQ {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "vision GPU attention: {seq} patches (1..={VIT_ATTN_MAX_SEQ})"
        )));
    }
    if head_dim == 0 || !head_dim.is_multiple_of(8) || head_dim > VIT_ATTN_MAX_HEAD_DIM {
        return Err(MetalGraphError::InvalidDimensions(format!(
            "vision GPU attention: head_dim {head_dim} must be a multiple of 8 in \
             8..={VIT_ATTN_MAX_HEAD_DIM}"
        )));
    }
    Ok(())
}

/// What opening a vision tower's session should report when it leaves
/// `live` Metal sessions in the process against a ceiling of `max`
/// ([`MetalGraph::max_sessions`]): `None` within the ceiling, else the
/// warning naming both numbers. The ceiling sizes engine pools but is not
/// enforced when a session opens (see the module docs, "Sessions"), so the
/// tower opens its session regardless.
#[must_use]
pub fn session_ceiling_warning(live: usize, max: usize) -> Option<String> {
    (live > max).then(|| {
        format!(
            "the Metal vision tower opened a session of its own, leaving {live} live Metal \
             sessions in this process against a ceiling of {max} \
             (OXIBONSAI_METAL_MAX_SESSIONS); every session keeps its own queue and buffers, and \
             the ceiling is not enforced when a session opens"
        )
    })
}
