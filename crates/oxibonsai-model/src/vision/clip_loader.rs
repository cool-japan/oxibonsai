//! Loader for the Qwen3-VL vision projector (`mmproj`): a GGUF with
//! `general.architecture = "clip"` and `clip.projector_type =
//! "qwen3vl_merger"` (bonsai2-design.md §6.1).
//!
//! [`VisionConfig::from_gguf`] reads and validates the `clip.*` metadata;
//! `load_vision_weights` binds every tensor the `qwen3vl_merger` graph
//! reads, checks each one's shape against the configuration and
//! dequantises it to `f32`. Nothing is left mapped: the tower owns plain
//! `Vec<f32>` matrices afterwards.
//!
//! # Strictness
//!
//! Every problem is an error, never a warning:
//!
//! - a tensor the graph reads is missing ([`ModelError::MissingTensor`]) or
//!   has the wrong shape ([`ModelError::ShapeMismatch`], naming the expected
//!   and the actual shape);
//! - a tensor is present that this graph does not read
//!   ([`ModelError::ShapeInvariant`] listing every one). `v.pre_ln.*`, a
//!   `v.blk.N.ffn_gate.*` and a `v.deepstack.*` merger each change the
//!   reference graph when present (the fork's `clip_graph_qwen3vl::build`
//!   applies them), and `v.class_embd` is one it asserts is absent, so
//!   running without them would produce plausible but wrong embeddings; any
//!   other stray tensor means the file is not the projector this loader
//!   implements, which is refused rather than silently ignored;
//! - the metadata describes a variant this graph does not cover: another
//!   projector type, a spatial merge other than 2, a non-GELU MLP, a
//!   deepstack layer, grouped-KV attention, a head width not divisible by
//!   four (the 2-D rotary embedding splits each head into four equal
//!   sections).
//!
//! # Weight layout
//!
//! GGUF stores a matrix as `[in, out]` in ggml order (`ne0 = in` is the
//! fastest-varying dimension), i.e. one contiguous row of `in` values per
//! output feature — exactly the `w[n x k]` layout
//! [`oxibonsai_kernels::gemm_f32()`] consumes, so no transpose is needed.
//! The patch kernel `[p, p, 3, hidden]` is likewise one row of `3 * p * p`
//! values per output channel, ordered channel-major, then kernel row, then
//! kernel column.

use std::collections::HashSet;

use oxibonsai_core::bf16::bf16_to_f32;
use oxibonsai_core::gguf::metadata::MetadataStore;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::{BlockQ8_0, BonsaiError, QK_Q8_0};
use rayon::prelude::*;

use crate::error::{ModelError, ModelResult};

/// `general.architecture` of a vision projector GGUF.
pub const CLIP_ARCHITECTURE: &str = "clip";

/// The only `clip.projector_type` this loader implements.
pub const QWEN3VL_MERGER_PROJECTOR: &str = "qwen3vl_merger";

/// Frequency base of the ViT's 2-D rotary embedding. The reference graph
/// hard-codes it (`ggml_rope_multi(..., 10000, ...)` in
/// `clip_graph_qwen3vl::build`); it is not stored in the file.
pub const VISION_ROPE_FREQ_BASE: f32 = 10_000.0;

/// The spatial merge the `qwen3vl_merger` graph is written for: its
/// patch-reordering reshapes and its position table hard-code 2 x 2 windows.
pub const SPATIAL_MERGE: usize = 2;

/// Metadata keys read by [`VisionConfig::from_gguf`].
pub mod keys {
    /// `general.architecture`.
    pub const ARCHITECTURE: &str = "general.architecture";
    /// `clip.has_vision_encoder`.
    pub const HAS_VISION_ENCODER: &str = "clip.has_vision_encoder";
    /// `clip.projector_type`.
    pub const PROJECTOR_TYPE: &str = "clip.projector_type";
    /// `clip.vision.projector_type` — the per-modality spelling a
    /// mixed-modality projector uses when `clip.projector_type` is absent.
    pub const VISION_PROJECTOR_TYPE: &str = "clip.vision.projector_type";
    /// `clip.use_gelu`.
    pub const USE_GELU: &str = "clip.use_gelu";
    /// `clip.use_silu`.
    pub const USE_SILU: &str = "clip.use_silu";
    /// `clip.vision.image_size`.
    pub const IMAGE_SIZE: &str = "clip.vision.image_size";
    /// `clip.vision.patch_size`.
    pub const PATCH_SIZE: &str = "clip.vision.patch_size";
    /// `clip.vision.embedding_length`.
    pub const EMBEDDING_LENGTH: &str = "clip.vision.embedding_length";
    /// `clip.vision.feed_forward_length`.
    pub const FEED_FORWARD_LENGTH: &str = "clip.vision.feed_forward_length";
    /// `clip.vision.block_count`.
    pub const BLOCK_COUNT: &str = "clip.vision.block_count";
    /// `clip.vision.projection_dim`.
    pub const PROJECTION_DIM: &str = "clip.vision.projection_dim";
    /// `clip.vision.attention.head_count`.
    pub const HEAD_COUNT: &str = "clip.vision.attention.head_count";
    /// `clip.vision.attention.head_count_kv`.
    pub const HEAD_COUNT_KV: &str = "clip.vision.attention.head_count_kv";
    /// `clip.vision.attention.head_dim`.
    pub const HEAD_DIM: &str = "clip.vision.attention.head_dim";
    /// `clip.vision.attention.layer_norm_epsilon`.
    pub const LAYER_NORM_EPSILON: &str = "clip.vision.attention.layer_norm_epsilon";
    /// `clip.vision.spatial_merge_size`.
    pub const SPATIAL_MERGE_SIZE: &str = "clip.vision.spatial_merge_size";
    /// `clip.vision.image_mean`.
    pub const IMAGE_MEAN: &str = "clip.vision.image_mean";
    /// `clip.vision.image_std`.
    pub const IMAGE_STD: &str = "clip.vision.image_std";
    /// `clip.vision.is_deepstack_layers`.
    pub const IS_DEEPSTACK_LAYERS: &str = "clip.vision.is_deepstack_layers";
}

/// Tensor names of the `qwen3vl_merger` projector.
pub mod names {
    /// First temporal slice of the patch convolution, `[p, p, 3, hidden]`.
    pub const PATCH_EMBD: &str = "v.patch_embd.weight";
    /// Second temporal slice of the patch convolution, `[p, p, 3, hidden]`.
    pub const PATCH_EMBD_1: &str = "v.patch_embd.weight.1";
    /// Patch-embedding bias, `[hidden]`.
    pub const PATCH_BIAS: &str = "v.patch_embd.bias";
    /// Learned position embedding, `[hidden, side * side]`.
    pub const POSITION_EMBD: &str = "v.position_embd.weight";
    /// Post-block LayerNorm scale (the merger's input norm), `[hidden]`.
    pub const POST_LN_WEIGHT: &str = "v.post_ln.weight";
    /// Post-block LayerNorm bias, `[hidden]`.
    pub const POST_LN_BIAS: &str = "v.post_ln.bias";
    /// First merger layer, `[4 * hidden, merger_hidden]`.
    pub const MM0_WEIGHT: &str = "mm.0.weight";
    /// First merger layer bias, `[merger_hidden]`.
    pub const MM0_BIAS: &str = "mm.0.bias";
    /// Second merger layer, `[merger_hidden, projection_dim]`.
    pub const MM2_WEIGHT: &str = "mm.2.weight";
    /// Second merger layer bias, `[projection_dim]`.
    pub const MM2_BIAS: &str = "mm.2.bias";

    /// `v.blk.{layer}.{suffix}`.
    #[must_use]
    pub fn block(layer: usize, suffix: &str) -> String {
        format!("v.blk.{layer}.{suffix}")
    }
}

/// The per-block tensor suffixes, in load order.
const BLOCK_SUFFIXES: [&str; 12] = [
    "ln1.weight",
    "ln1.bias",
    "attn_qkv.weight",
    "attn_qkv.bias",
    "attn_out.weight",
    "attn_out.bias",
    "ln2.weight",
    "ln2.bias",
    "ffn_up.weight",
    "ffn_up.bias",
    "ffn_down.weight",
    "ffn_down.bias",
];

/// The non-block tensors the graph reads.
const GLOBAL_TENSORS: [&str; 10] = [
    names::PATCH_EMBD,
    names::PATCH_EMBD_1,
    names::PATCH_BIAS,
    names::POSITION_EMBD,
    names::POST_LN_WEIGHT,
    names::POST_LN_BIAS,
    names::MM0_WEIGHT,
    names::MM0_BIAS,
    names::MM2_WEIGHT,
    names::MM2_BIAS,
];

/// Hyper-parameters of a `qwen3vl_merger` vision projector.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// `clip.vision.image_size` — the native training resolution (768).
    /// Informational: the tower accepts any size whose sides are multiples
    /// of [`Self::merge_unit`].
    pub image_size: usize,
    /// `clip.vision.patch_size` (16).
    pub patch_size: usize,
    /// `clip.vision.spatial_merge_size` (always [`SPATIAL_MERGE`]).
    pub spatial_merge: usize,
    /// `clip.vision.embedding_length` (1152).
    pub hidden: usize,
    /// `clip.vision.feed_forward_length` (4304).
    pub ffn: usize,
    /// `clip.vision.attention.head_count` (16).
    pub heads: usize,
    /// Per-head width, `hidden / heads` (72).
    pub head_dim: usize,
    /// `clip.vision.block_count` (27).
    pub blocks: usize,
    /// `clip.vision.attention.layer_norm_epsilon` (1e-6).
    pub eps: f32,
    /// `clip.vision.projection_dim` (5120) — the width of every output row,
    /// equal to the language model's hidden size.
    pub projection_dim: usize,
    /// Output width of `mm.0` (4608).
    pub merger_hidden: usize,
    /// Side of the stored square position-embedding grid (48).
    pub pos_grid: usize,
    /// `clip.vision.image_mean`, per RGB channel.
    pub image_mean: [f32; 3],
    /// `clip.vision.image_std`, per RGB channel.
    pub image_std: [f32; 3],
    /// Frequency base of the 2-D rotary embedding
    /// ([`VISION_ROPE_FREQ_BASE`]).
    pub rope_freq_base: f32,
}

impl VisionConfig {
    /// Read and validate the `clip.*` metadata of `gguf`, plus the two
    /// dimensions only the tensors carry (the stored position grid and the
    /// merger's hidden width).
    ///
    /// # Errors
    ///
    /// - [`ModelError::Core`] with `MissingConfigKey` / `InvalidMetadata`
    ///   for an absent or mistyped required key;
    /// - [`ModelError::ShapeInvariant`] for a file this graph does not
    ///   cover (see the module docs);
    /// - [`ModelError::MissingTensor`] / [`ModelError::ShapeMismatch`] when
    ///   `v.position_embd.weight` or `mm.0.weight` is absent or not 2-D.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> ModelResult<Self> {
        let meta = &gguf.metadata;

        let arch = meta.get_string(keys::ARCHITECTURE)?;
        if arch != CLIP_ARCHITECTURE {
            return Err(invariant(
                keys::ARCHITECTURE,
                format!("\"{CLIP_ARCHITECTURE}\" (a vision projector GGUF)"),
                format!("\"{arch}\""),
            ));
        }
        if !optional_bool(meta, keys::HAS_VISION_ENCODER)?.unwrap_or(false) {
            return Err(invariant(
                keys::HAS_VISION_ENCODER,
                "true".to_string(),
                "false or absent: the file carries no vision encoder".to_string(),
            ));
        }
        let projector = projector_type(meta)?;
        if projector != QWEN3VL_MERGER_PROJECTOR {
            return Err(invariant(
                keys::PROJECTOR_TYPE,
                format!("\"{QWEN3VL_MERGER_PROJECTOR}\""),
                format!("\"{projector}\""),
            ));
        }
        let use_gelu = optional_bool(meta, keys::USE_GELU)?.unwrap_or(false);
        let use_silu = optional_bool(meta, keys::USE_SILU)?.unwrap_or(false);
        if !use_gelu || use_silu {
            return Err(invariant(
                keys::USE_GELU,
                "clip.use_gelu = true and clip.use_silu unset (a GELU MLP)".to_string(),
                format!("use_gelu = {use_gelu}, use_silu = {use_silu}"),
            ));
        }

        let image_size = required_usize(meta, keys::IMAGE_SIZE)?;
        let patch_size = required_usize(meta, keys::PATCH_SIZE)?;
        let hidden = required_usize(meta, keys::EMBEDDING_LENGTH)?;
        let ffn = required_usize(meta, keys::FEED_FORWARD_LENGTH)?;
        let blocks = required_usize(meta, keys::BLOCK_COUNT)?;
        let projection_dim = required_usize(meta, keys::PROJECTION_DIM)?;
        let heads = required_usize(meta, keys::HEAD_COUNT)?;
        let eps = meta.get_f32(keys::LAYER_NORM_EPSILON)?;
        let spatial_merge = match meta.get(keys::SPATIAL_MERGE_SIZE) {
            None => SPATIAL_MERGE,
            Some(value) => usize_of(keys::SPATIAL_MERGE_SIZE, value.as_u32())?,
        };

        for (key, value) in [
            (keys::PATCH_SIZE, patch_size),
            (keys::EMBEDDING_LENGTH, hidden),
            (keys::FEED_FORWARD_LENGTH, ffn),
            (keys::BLOCK_COUNT, blocks),
            (keys::PROJECTION_DIM, projection_dim),
            (keys::HEAD_COUNT, heads),
        ] {
            if value == 0 {
                return Err(invariant(
                    key,
                    "a positive value".to_string(),
                    "0".to_string(),
                ));
            }
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(invariant(
                keys::LAYER_NORM_EPSILON,
                "a finite positive epsilon".to_string(),
                format!("{eps}"),
            ));
        }
        if spatial_merge != SPATIAL_MERGE {
            return Err(invariant(
                keys::SPATIAL_MERGE_SIZE,
                format!("{SPATIAL_MERGE} (the qwen3vl_merger graph merges 2 x 2 windows)"),
                format!("{spatial_merge}"),
            ));
        }
        if !hidden.is_multiple_of(heads) {
            return Err(invariant(
                keys::HEAD_COUNT,
                format!("a divisor of the hidden width {hidden}"),
                format!("{heads}"),
            ));
        }
        let head_dim = hidden / heads;
        if let Some(value) = meta.get(keys::HEAD_DIM) {
            let declared = usize_of(keys::HEAD_DIM, value.as_u32())?;
            if declared != head_dim {
                return Err(invariant(
                    keys::HEAD_DIM,
                    format!("hidden / heads = {head_dim} (the fused QKV rows are split per head)"),
                    format!("{declared}"),
                ));
            }
        }
        if !head_dim.is_multiple_of(4) {
            return Err(invariant(
                keys::HEAD_DIM,
                "a multiple of 4 (the 2-D rotary embedding uses four equal sections)".to_string(),
                format!("{head_dim}"),
            ));
        }
        if let Some(value) = meta.get(keys::HEAD_COUNT_KV) {
            let kv = usize_of(keys::HEAD_COUNT_KV, value.as_u32())?;
            if kv != heads {
                return Err(invariant(
                    keys::HEAD_COUNT_KV,
                    format!("{heads} (the vision tower has no grouped-KV attention)"),
                    format!("{kv}"),
                ));
            }
        }
        if let Some(value) = meta.get(keys::IS_DEEPSTACK_LAYERS) {
            let flags = value.as_array().ok_or_else(|| {
                ModelError::Core(BonsaiError::InvalidMetadata {
                    key: keys::IS_DEEPSTACK_LAYERS.to_string(),
                    reason: format!("expected a bool array, found {}", value.type_name()),
                })
            })?;
            for (layer, flag) in flags.iter().enumerate() {
                match flag.as_bool() {
                    Some(false) => {}
                    Some(true) => {
                        return Err(invariant(
                            keys::IS_DEEPSTACK_LAYERS,
                            "false for every layer (no deepstack merger)".to_string(),
                            format!("true at layer {layer}"),
                        ));
                    }
                    None => {
                        return Err(ModelError::Core(BonsaiError::InvalidMetadata {
                            key: keys::IS_DEEPSTACK_LAYERS.to_string(),
                            reason: format!(
                                "element {layer} is not a bool, found {}",
                                flag.type_name()
                            ),
                        }));
                    }
                }
            }
        }
        let image_mean = rgb_triple(meta, keys::IMAGE_MEAN)?;
        let image_std = rgb_triple(meta, keys::IMAGE_STD)?;
        if let Some(bad) = image_std.iter().find(|s| !(s.is_finite() && **s != 0.0)) {
            return Err(invariant(
                keys::IMAGE_STD,
                "three finite non-zero values".to_string(),
                format!("{bad}"),
            ));
        }
        if let Some(bad) = image_mean.iter().find(|m| !m.is_finite()) {
            return Err(invariant(
                keys::IMAGE_MEAN,
                "three finite values".to_string(),
                format!("{bad}"),
            ));
        }

        // The stored position grid: `[hidden, side * side]`.
        let pos_shape = tensor_shape(gguf, names::POSITION_EMBD)?;
        let n_pos = match pos_shape.as_slice() {
            [h, n] if *h == hidden => *n,
            _ => {
                return Err(ModelError::ShapeMismatch {
                    name: names::POSITION_EMBD.to_string(),
                    expected: vec![hidden, 0],
                    actual: pos_shape,
                });
            }
        };
        let pos_grid = integer_sqrt(n_pos);
        if pos_grid == 0 || pos_grid * pos_grid != n_pos {
            return Err(invariant(
                names::POSITION_EMBD,
                "a square number of positions (side * side)".to_string(),
                format!("{n_pos}"),
            ));
        }

        // The merger's hidden width: `mm.0.weight` is `[4 * hidden, mid]`.
        let merged = hidden * SPATIAL_MERGE * SPATIAL_MERGE;
        let mm0_shape = tensor_shape(gguf, names::MM0_WEIGHT)?;
        let merger_hidden = match mm0_shape.as_slice() {
            [k, mid] if *k == merged && *mid > 0 => *mid,
            _ => {
                return Err(ModelError::ShapeMismatch {
                    name: names::MM0_WEIGHT.to_string(),
                    expected: vec![merged, 0],
                    actual: mm0_shape,
                });
            }
        };

        Ok(Self {
            image_size,
            patch_size,
            spatial_merge,
            hidden,
            ffn,
            heads,
            head_dim,
            blocks,
            eps,
            projection_dim,
            merger_hidden,
            pos_grid,
            image_mean,
            image_std,
            rope_freq_base: VISION_ROPE_FREQ_BASE,
        })
    }

    /// Pixels per side of one merge window: `patch_size * spatial_merge`
    /// (32). Both image sides must be a multiple of this.
    #[must_use]
    pub const fn merge_unit(&self) -> usize {
        self.patch_size * self.spatial_merge
    }

    /// Width of one merged token before the merger:
    /// `hidden * spatial_merge^2` (4608).
    #[must_use]
    pub const fn merged_width(&self) -> usize {
        self.hidden * self.spatial_merge * self.spatial_merge
    }

    /// Values per patch: `3 * patch_size^2` (768).
    #[must_use]
    pub const fn patch_len(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    /// The four rotary sections, `head_dim / 4` each — ggml's
    /// `mrope_sections = {d_head/4, d_head/4, d_head/4, d_head/4}`.
    #[must_use]
    pub const fn rope_sections(&self) -> [usize; 4] {
        let s = self.head_dim / 4;
        [s, s, s, s]
    }

    /// The number of tensors a complete projector with this configuration
    /// carries: 12 per block plus the 10 global ones.
    #[must_use]
    pub const fn expected_tensor_count(&self) -> usize {
        BLOCK_SUFFIXES.len() * self.blocks + GLOBAL_TENSORS.len()
    }
}

/// One ViT block's weights, dequantised to `f32`. Every matrix is
/// row-major with one row per output feature (`[out x in]`).
#[derive(Debug, Clone)]
pub(crate) struct VisionBlockWeights {
    pub(crate) ln1_w: Vec<f32>,
    pub(crate) ln1_b: Vec<f32>,
    /// `[3 * hidden x hidden]`: rows `0..hidden` are Q, then K, then V.
    pub(crate) qkv_w: Vec<f32>,
    pub(crate) qkv_b: Vec<f32>,
    pub(crate) out_w: Vec<f32>,
    pub(crate) out_b: Vec<f32>,
    pub(crate) ln2_w: Vec<f32>,
    pub(crate) ln2_b: Vec<f32>,
    pub(crate) up_w: Vec<f32>,
    pub(crate) up_b: Vec<f32>,
    pub(crate) down_w: Vec<f32>,
    pub(crate) down_b: Vec<f32>,
}

impl VisionBlockWeights {
    /// Resident bytes of every buffer.
    pub(crate) fn resident_bytes(&self) -> usize {
        [
            &self.ln1_w,
            &self.ln1_b,
            &self.qkv_w,
            &self.qkv_b,
            &self.out_w,
            &self.out_b,
            &self.ln2_w,
            &self.ln2_b,
            &self.up_w,
            &self.up_b,
            &self.down_w,
            &self.down_b,
        ]
        .iter()
        .map(|v| std::mem::size_of_val(v.as_slice()))
        .sum()
    }
}

/// Every weight of a projector, dequantised to `f32`.
#[derive(Debug, Clone)]
pub(crate) struct VisionWeights {
    /// `v.patch_embd.weight`: `[hidden x 3 * p * p]`.
    pub(crate) patch_w0: Vec<f32>,
    /// `v.patch_embd.weight.1`: `[hidden x 3 * p * p]`.
    pub(crate) patch_w1: Vec<f32>,
    pub(crate) patch_b: Vec<f32>,
    /// `v.position_embd.weight`: `[side * side x hidden]`, raster order.
    pub(crate) pos_embd: Vec<f32>,
    pub(crate) blocks: Vec<VisionBlockWeights>,
    pub(crate) post_ln_w: Vec<f32>,
    pub(crate) post_ln_b: Vec<f32>,
    /// `[merger_hidden x 4 * hidden]`.
    pub(crate) mm0_w: Vec<f32>,
    pub(crate) mm0_b: Vec<f32>,
    /// `[projection_dim x merger_hidden]`.
    pub(crate) mm2_w: Vec<f32>,
    pub(crate) mm2_b: Vec<f32>,
    /// The number of GGUF tensors bound (always the file's whole inventory).
    pub(crate) bound_tensors: usize,
}

/// Bind and dequantise every tensor of the projector described by `cfg`.
///
/// Blocks load in parallel; the result does not depend on the thread count.
///
/// # Errors
///
/// [`ModelError::MissingTensor`], [`ModelError::ShapeMismatch`] and
/// [`ModelError::InvalidTensor`] (an unsupported storage type) naming the
/// offending tensor, and [`ModelError::ShapeInvariant`] listing every tensor
/// the graph does not read (see the module docs).
pub(crate) fn load_vision_weights(
    gguf: &GgufFile<'_>,
    cfg: &VisionConfig,
) -> ModelResult<VisionWeights> {
    check_inventory(gguf, cfg)?;

    let h = cfg.hidden;
    let p = cfg.patch_size;
    let patch_shape = [p, p, 3, h];
    let patch_w0 = load_tensor_f32(gguf, names::PATCH_EMBD, &patch_shape)?;
    let patch_w1 = load_tensor_f32(gguf, names::PATCH_EMBD_1, &patch_shape)?;
    let patch_b = load_tensor_f32(gguf, names::PATCH_BIAS, &[h])?;
    let pos_embd = load_tensor_f32(
        gguf,
        names::POSITION_EMBD,
        &[h, cfg.pos_grid * cfg.pos_grid],
    )?;

    let blocks = (0..cfg.blocks)
        .into_par_iter()
        .map(|layer| load_block(gguf, cfg, layer))
        .collect::<ModelResult<Vec<_>>>()?;

    let post_ln_w = load_tensor_f32(gguf, names::POST_LN_WEIGHT, &[h])?;
    let post_ln_b = load_tensor_f32(gguf, names::POST_LN_BIAS, &[h])?;
    let merged = cfg.merged_width();
    let mm0_w = load_tensor_f32(gguf, names::MM0_WEIGHT, &[merged, cfg.merger_hidden])?;
    let mm0_b = load_tensor_f32(gguf, names::MM0_BIAS, &[cfg.merger_hidden])?;
    let mm2_w = load_tensor_f32(
        gguf,
        names::MM2_WEIGHT,
        &[cfg.merger_hidden, cfg.projection_dim],
    )?;
    let mm2_b = load_tensor_f32(gguf, names::MM2_BIAS, &[cfg.projection_dim])?;

    Ok(VisionWeights {
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
        bound_tensors: cfg.expected_tensor_count(),
    })
}

/// Load block `layer`'s twelve tensors.
fn load_block(
    gguf: &GgufFile<'_>,
    cfg: &VisionConfig,
    layer: usize,
) -> ModelResult<VisionBlockWeights> {
    let h = cfg.hidden;
    let f = cfg.ffn;
    let load =
        |suffix: &str, shape: &[usize]| load_tensor_f32(gguf, &names::block(layer, suffix), shape);
    Ok(VisionBlockWeights {
        ln1_w: load("ln1.weight", &[h])?,
        ln1_b: load("ln1.bias", &[h])?,
        qkv_w: load("attn_qkv.weight", &[h, 3 * h])?,
        qkv_b: load("attn_qkv.bias", &[3 * h])?,
        out_w: load("attn_out.weight", &[h, h])?,
        out_b: load("attn_out.bias", &[h])?,
        ln2_w: load("ln2.weight", &[h])?,
        ln2_b: load("ln2.bias", &[h])?,
        up_w: load("ffn_up.weight", &[h, f])?,
        up_b: load("ffn_up.bias", &[f])?,
        down_w: load("ffn_down.weight", &[f, h])?,
        down_b: load("ffn_down.bias", &[h])?,
    })
}

/// Check that the file carries every tensor the graph reads and nothing
/// else.
///
/// Missing tensors are reported first (the first one, by name); then every
/// unexpected tensor at once, sorted, so a file with a stray gate or
/// deepstack merger is diagnosed in one pass.
fn check_inventory(gguf: &GgufFile<'_>, cfg: &VisionConfig) -> ModelResult<()> {
    let mut expected: Vec<String> = GLOBAL_TENSORS.iter().map(|s| (*s).to_string()).collect();
    for layer in 0..cfg.blocks {
        expected.extend(BLOCK_SUFFIXES.iter().map(|s| names::block(layer, s)));
    }
    for name in &expected {
        if gguf.tensors.get(name).is_none() {
            return Err(ModelError::MissingTensor { name: name.clone() });
        }
    }
    let expected: HashSet<&str> = expected.iter().map(String::as_str).collect();
    let mut unexpected: Vec<&str> = gguf
        .tensors
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| !expected.contains(name))
        .collect();
    if unexpected.is_empty() {
        return Ok(());
    }
    unexpected.sort_unstable();
    Err(invariant(
        "mmproj tensor inventory",
        format!(
            "exactly the {} tensors of the {QWEN3VL_MERGER_PROJECTOR} graph \
             ({} blocks; no pre_ln, class_embd, ffn_gate or deepstack tensors)",
            cfg.expected_tensor_count(),
            cfg.blocks
        ),
        format!(
            "{} unexpected tensor(s): {}",
            unexpected.len(),
            unexpected.join(", ")
        ),
    ))
}

/// Load tensor `name`, check that its shape is `expected` (GGUF order,
/// trailing unit dimensions ignored) and dequantise it to `f32`.
///
/// Supported storage types: `F32`, `F16`, `BF16` and `Q8_0` — the types a
/// llama.cpp vision projector is written in. `Q8_0` goes through
/// [`BlockQ8_0::dequant`]; the flat types are plain element conversions.
///
/// # Errors
///
/// - [`ModelError::MissingTensor`] if the tensor is absent;
/// - [`ModelError::ShapeMismatch`] if its shape differs from `expected`;
/// - [`ModelError::InvalidTensor`] for any other storage type, or a `Q8_0`
///   row that is not a whole number of 32-element blocks;
/// - [`ModelError::Core`] if the bytes lie outside the file or a `Q8_0`
///   buffer is misaligned.
pub(crate) fn load_tensor_f32(
    gguf: &GgufFile<'_>,
    name: &str,
    expected: &[usize],
) -> ModelResult<Vec<f32>> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_string(),
        })?;
    let actual = trimmed_shape(&info.shape);
    let wanted = trim_unit_dims(expected);
    if actual != wanted {
        return Err(ModelError::ShapeMismatch {
            name: name.to_string(),
            expected: wanted,
            actual,
        });
    }
    let n = expected
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            invariant(
                name,
                "an element count representable in usize".to_string(),
                format!("{expected:?} overflows"),
            )
        })?;
    let data = gguf.tensor_data(name)?;
    let values = match info.tensor_type {
        GgufTensorType::F32 => flat(name, data, n, 4, |c| {
            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
        })?,
        GgufTensorType::F16 => flat(name, data, n, 2, |c| {
            half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()
        })?,
        GgufTensorType::BF16 => flat(name, data, n, 2, |c| {
            bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))
        })?,
        GgufTensorType::Q8_0 => {
            info.validate_row_blocking()?;
            let blocks = BlockQ8_0::slice_from_bytes(data)?;
            if blocks.len().checked_mul(QK_Q8_0) != Some(n) {
                return Err(ModelError::InvalidTensor(format!(
                    "{name}: {} Q8_0 blocks decode to {} values, the shape needs {n}",
                    blocks.len(),
                    blocks.len().saturating_mul(QK_Q8_0)
                )));
            }
            let mut out = vec![0.0f32; n];
            out.par_chunks_mut(QK_Q8_0 * 1024)
                .zip(blocks.par_chunks(1024))
                .try_for_each(|(dst, src)| BlockQ8_0::dequant(src, dst))?;
            out
        }
        other => {
            return Err(ModelError::InvalidTensor(format!(
                "{name}: storage type {} is not supported by the vision loader \
                 (expected F32, F16, BF16 or Q8_0)",
                other.name()
            )));
        }
    };
    Ok(values)
}

/// Decode a flat tensor of `n` elements, `elem_bytes` bytes each.
fn flat(
    name: &str,
    data: &[u8],
    n: usize,
    elem_bytes: usize,
    decode: impl Fn(&[u8]) -> f32 + Sync,
) -> ModelResult<Vec<f32>> {
    if Some(data.len()) != n.checked_mul(elem_bytes) {
        return Err(ModelError::ShapeMismatch {
            name: format!("{name} (bytes)"),
            expected: vec![n.saturating_mul(elem_bytes)],
            actual: vec![data.len()],
        });
    }
    let mut out = vec![0.0f32; n];
    out.par_chunks_mut(1 << 16)
        .zip(data.par_chunks(elem_bytes << 16))
        .for_each(|(dst, src)| {
            for (value, bytes) in dst.iter_mut().zip(src.chunks_exact(elem_bytes)) {
                *value = decode(bytes);
            }
        });
    Ok(out)
}

/// The GGUF-order shape of tensor `name`, trailing unit dimensions dropped.
fn tensor_shape(gguf: &GgufFile<'_>, name: &str) -> ModelResult<Vec<usize>> {
    let info = gguf
        .tensors
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor {
            name: name.to_string(),
        })?;
    Ok(trimmed_shape(&info.shape))
}

/// `shape` as `usize`s (saturating), trailing unit dimensions dropped.
fn trimmed_shape(shape: &[u64]) -> Vec<usize> {
    let dims: Vec<usize> = shape
        .iter()
        .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
        .collect();
    trim_unit_dims(&dims)
}

/// `dims` with trailing `1`s removed (a `[n, 1]` bias is the same tensor as
/// `[n]`); a scalar stays `[1]`.
fn trim_unit_dims(dims: &[usize]) -> Vec<usize> {
    let mut out = dims.to_vec();
    while out.len() > 1 && out.last() == Some(&1) {
        out.pop();
    }
    out
}

/// The largest `r` with `r * r <= n`.
fn integer_sqrt(n: usize) -> usize {
    let mut r = (n as f64).sqrt() as usize;
    while r.saturating_mul(r) > n {
        r -= 1;
    }
    while (r + 1).saturating_mul(r + 1) <= n {
        r += 1;
    }
    r
}

/// `clip.projector_type`, or `clip.vision.projector_type` when the former
/// is absent or empty (the per-modality spelling of mixed projectors).
fn projector_type(meta: &MetadataStore) -> ModelResult<String> {
    for key in [keys::PROJECTOR_TYPE, keys::VISION_PROJECTOR_TYPE] {
        if let Some(value) = meta.get(key) {
            let text = value.as_str().ok_or_else(|| {
                ModelError::Core(BonsaiError::InvalidMetadata {
                    key: key.to_string(),
                    reason: format!("expected a string, found {}", value.type_name()),
                })
            })?;
            if !text.is_empty() {
                return Ok(text.to_string());
            }
        }
    }
    Err(ModelError::Core(BonsaiError::MissingConfigKey {
        key: keys::PROJECTOR_TYPE.to_string(),
    }))
}

/// An optional bool key: `None` when absent, an error when mistyped.
fn optional_bool(meta: &MetadataStore, key: &str) -> ModelResult<Option<bool>> {
    match meta.get(key) {
        None => Ok(None),
        Some(value) => value.as_bool().map(Some).ok_or_else(|| {
            ModelError::Core(BonsaiError::InvalidMetadata {
                key: key.to_string(),
                reason: format!("expected bool, found {}", value.type_name()),
            })
        }),
    }
}

/// A required unsigned integer key as `usize`.
fn required_usize(meta: &MetadataStore, key: &str) -> ModelResult<usize> {
    let value = meta.get_u32(key)?;
    usize_of(key, Some(value))
}

fn usize_of(key: &str, value: Option<u32>) -> ModelResult<usize> {
    let value = value.ok_or_else(|| {
        ModelError::Core(BonsaiError::InvalidMetadata {
            key: key.to_string(),
            reason: "expected an unsigned integer".to_string(),
        })
    })?;
    usize::try_from(value).map_err(|_| {
        ModelError::Core(BonsaiError::InvalidMetadata {
            key: key.to_string(),
            reason: format!("{value} does not fit in usize"),
        })
    })
}

/// A required array of at least three floats (the first three are used,
/// as the reference loader does).
fn rgb_triple(meta: &MetadataStore, key: &str) -> ModelResult<[f32; 3]> {
    let values = meta.get_array(key)?;
    if values.len() < 3 {
        return Err(ModelError::Core(BonsaiError::InvalidMetadata {
            key: key.to_string(),
            reason: format!("expected at least 3 values, found {}", values.len()),
        }));
    }
    let mut out = [0.0f32; 3];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value.as_f32().ok_or_else(|| {
            ModelError::Core(BonsaiError::InvalidMetadata {
                key: key.to_string(),
                reason: format!("expected a float, found {}", value.type_name()),
            })
        })?;
    }
    Ok(out)
}

/// A [`ModelError::ShapeInvariant`] for `what`.
fn invariant(what: &str, expected: String, actual: String) -> ModelError {
    ModelError::ShapeInvariant {
        tensor: what.to_string(),
        expected,
        actual,
    }
}
