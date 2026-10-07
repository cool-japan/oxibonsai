//! Qwen3-VL vision tower for the Bonsai 2 vision projector (`mmproj`),
//! bonsai2-design.md §6.
//!
//! Vision is optional: text-only inference never loads, touches or needs
//! anything in this module, and nothing outside it depends on it.
//!
//! # The file
//!
//! The projector ships as a separate GGUF with `general.architecture =
//! "clip"` and `clip.projector_type = "qwen3vl_merger"`. For Bonsai 2 27B it
//! is `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (629 246 976 bytes): 27 ViT
//! blocks over a hidden width of 1152 with 16 heads, 16 x 16 patches, a
//! learned position embedding over a 48 x 48 grid, and a two-layer merger
//! MLP (4608 -> 4608 -> 5120) that lands in the language model's embedding
//! space. [`clip_loader`] reads and validates it; everything is dequantised
//! to `f32` once at load.
//!
//! # The graph
//!
//! [`VisionTower::encode`] evaluates the graph of the PrismML llama.cpp
//! fork's `clip_graph_qwen3vl::build` (tools/mtmd/models/qwen3vl.cpp):
//!
//! 1. normalise the RGB8 pixels to `(x / 255 - mean) / std` per channel;
//! 2. patchify with **both** temporal slices of the patch convolution (a
//!    still image is a two-frame temporal patch, so the two kernels are
//!    summed), add `v.patch_embd.bias`;
//! 3. reorder the patches into 2 x 2 merge-window order and add the learned
//!    position embedding, bilinearly resized (align-corners) from the stored
//!    grid to the image's patch grid and reordered the same way
//!    ([`patch_embed`]);
//! 4. run the pre-LayerNorm ViT blocks — fused QKV, a 2-D rotary embedding
//!    (ggml's `GGML_ROPE_TYPE_VISION`: row position on the first half of
//!    each head's rotation pairs, column position on the second), full
//!    bidirectional attention, GELU MLP ([`tower`]);
//! 5. `v.post_ln`, concatenate each merge window's four tokens, and apply
//!    `mm.0` -> GELU -> `mm.2`.
//!
//! The rows [`VisionTower::encode`] returns are the final merged embeddings,
//! one per 2 x 2 window, in row-major order over the merged grid. They are
//! in the language model's **unrotated** embedding basis (bonsai2-design.md
//! §3.5): the vision tower is a plain, unfolded network, so its rows must
//! bypass the inverse Hadamard transform that `token_embd` lookups need.
//!
//! # The input boundary
//!
//! [`VisionTower::encode`] takes an [`ImageRgb8`] that has **already** been
//! resized so that both sides are multiples of `patch_size * spatial_merge`
//! (32 for Bonsai 2) and the merged-token count fits the caller's budget; it
//! refuses anything else with a typed error rather than resizing. Resizing
//! (the fork's smart-resize) belongs to the caller's preprocessing; the
//! mean/std normalisation belongs here, because the tower owns the metadata
//! that defines it.
//!
//! # From a reference to rows in the prompt
//!
//! | step | module |
//! |---|---|
//! | `data:` URI / file -> PNG or JPEG bytes -> RGB8 | [`image_decode`] |
//! | `http(s)` reference -> the opt-in, the fetcher seam, the address policy | [`remote`] |
//! | smart resize + Pillow bicubic into the box, token budget | [`preprocess`] |
//! | ViT + merger -> merged rows (unrotated basis) | [`tower`] |
//! | one `<|image_pad|>` -> `h * w` rows, bracket checks | [`merger`] |
//! | 3-axis M-RoPE positions of the spliced prompt | [`mrope`] |
//!
//! The model side of the splice — text rows through the inverse Hadamard
//! transform, image rows around it — is
//! [`crate::hybrid::vision_prefill`].
//!
//! # Two executors
//!
//! [`VisionTower`] runs on the CPU; `metal::VisionTowerMetal` (Metal builds) runs the
//! same graph on the Metal GPU (with the matrices in the file's own `Q8_0`
//! / `F16` storage, read exactly). A caller that
//! serves both holds a [`VisionEncoder`], whose surface is the CPU tower's
//! (`encode`, `encode_normalized`, `merged_grid`, `config`,
//! `resident_bytes`), and builds exactly one of them.
//!
//! # Numerics
//!
//! On the CPU tower everything runs in `f32`. Every matrix product — the patch
//! convolution, QKV, the attention's `QKᵀ` and `PV`, the output projection,
//! the MLP and the merger — goes through
//! [`oxibonsai_kernels::KernelDispatcher::gemm_f32`], so the result does not
//! depend on the Rayon thread count. The test suite checks the whole graph
//! against an independent `f64` evaluation of the same graph
//! (`f64_reference.rs`) on a synthetic projector and on the real one.

pub mod clip_loader;
#[cfg(test)]
mod f64_reference;
pub mod image_decode;
pub mod merger;
pub mod metal;
pub mod mrope;
pub mod patch_embed;
pub mod preprocess;
pub mod remote;
pub mod tower;
#[cfg(test)]
mod tower_tests;

pub use clip_loader::VisionConfig;
pub use image_decode::{
    decode_image, load_image_bytes, load_image_source, parse_data_uri, ImageInputError,
    ImageSourcePolicy,
};
pub use merger::{plan_splice, SpliceError, SplicePlan, SpliceSegment, VisionTokenIds};
pub use mrope::{image_positions, image_rope_span, MropeCursor, MropePos};
pub use preprocess::{
    prepare_image, PreparedImage, PreprocessConfig, ResizeFilter, DEFAULT_IMAGE_MAX_TOKENS,
    MAX_IMAGE_MAX_TOKENS,
};
pub use remote::{RemoteImageAccess, RemoteImageFetcher, SharedRemoteImageFetcher};
pub use tower::VisionTower;

use crate::error::{ModelError, ModelResult};

/// An 8-bit RGB image: `width * height` pixels, row-major, three bytes per
/// pixel, no row padding — pixel `(x, y)` channel `c` is
/// `data[(y * width + x) * 3 + c]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRgb8 {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` bytes, row-major RGB.
    pub data: Vec<u8>,
}

impl ImageRgb8 {
    /// Wrap `data` as a `width x height` RGB image, checking its length.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when `data.len() != width * height * 3`
    /// (or that product overflows).
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> ModelResult<Self> {
        let image = Self {
            width,
            height,
            data,
        };
        image.validate()?;
        Ok(image)
    }

    /// Check that `data` holds exactly `width * height * 3` bytes.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] naming the image buffer, with the
    /// expected and actual byte counts.
    pub fn validate(&self) -> ModelResult<()> {
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "image".to_string(),
                expected: "width * height * 3 representable in usize".to_string(),
                actual: format!("{} x {} overflows", self.width, self.height),
            })?;
        if self.data.len() != expected {
            return Err(ModelError::ShapeMismatch {
                name: format!("image data ({} x {} RGB8)", self.width, self.height),
                expected: vec![expected],
                actual: vec![self.data.len()],
            });
        }
        Ok(())
    }

    /// The RGB value of pixel `(x, y)`, or `None` outside the image (or when
    /// the buffer is shorter than the declared size).
    #[must_use]
    pub fn pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let base = y.checked_mul(self.width)?.checked_add(x)?.checked_mul(3)?;
        let px = self.data.get(base..base.checked_add(3)?)?;
        Some([px[0], px[1], px[2]])
    }
}

/// A vision tower on one executor, with the CPU tower's surface: the
/// runtime builds the one its engine's decode runs beside (the Metal tower
/// for a Metal engine, never both).
#[derive(Debug)]
pub enum VisionEncoder {
    /// The CPU tower (`f32`, every matrix product on the CPU).
    Cpu(VisionTower),
    /// The Metal tower (the file's `Q8_0` / `F16` matrices as stored, one
    /// command buffer per image).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(metal::VisionTowerMetal),
}

impl VisionEncoder {
    /// `"cpu"` or `"metal"`.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(_) => "metal",
        }
    }

    /// Whether this is the Metal tower.
    #[must_use]
    pub fn is_metal(&self) -> bool {
        self.backend_name() == "metal"
    }

    /// The projector's hyper-parameters.
    #[must_use]
    pub fn config(&self) -> &VisionConfig {
        match self {
            Self::Cpu(tower) => tower.config(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(tower) => tower.config(),
        }
    }

    /// The number of ViT blocks bound.
    #[must_use]
    pub fn block_count(&self) -> usize {
        match self {
            Self::Cpu(tower) => tower.block_count(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(tower) => tower.block_count(),
        }
    }

    /// The number of GGUF tensors bound.
    #[must_use]
    pub fn bound_tensor_count(&self) -> usize {
        match self {
            Self::Cpu(tower) => tower.bound_tensor_count(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(tower) => tower.bound_tensor_count(),
        }
    }

    /// Bytes the tower keeps resident (`f32` weights on the CPU; device
    /// weights, scratch and the host position grid on Metal).
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Cpu(tower) => tower.resident_bytes(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(tower) => tower.resident_bytes(),
        }
    }

    /// The merged grid an image of `width x height` pixels produces (see
    /// [`VisionTower::merged_grid`]).
    ///
    /// # Errors
    ///
    /// The tower's geometry errors.
    pub fn merged_grid(
        &self,
        width: usize,
        height: usize,
        max_tokens: usize,
    ) -> ModelResult<GridSize> {
        match self {
            Self::Cpu(tower) => tower.merged_grid(width, height, max_tokens),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(tower) => tower.merged_grid(width, height, max_tokens),
        }
    }

    /// Encode an RGB8 image sized for the tower (see
    /// [`VisionTower::encode`]).
    ///
    /// # Errors
    ///
    /// The tower's errors.
    pub fn encode(&self, img: &ImageRgb8, max_tokens: usize) -> ModelResult<(Vec<f32>, GridSize)> {
        match self {
            Self::Cpu(tower) => tower.encode(img, max_tokens),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(tower) => tower.encode(img, max_tokens),
        }
    }

    /// Encode already-normalised planar pixels (see
    /// [`VisionTower::encode_normalized`]).
    ///
    /// # Errors
    ///
    /// The tower's errors.
    pub fn encode_normalized(
        &self,
        planar: &[f32],
        width: usize,
        height: usize,
        max_tokens: usize,
    ) -> ModelResult<(Vec<f32>, GridSize)> {
        match self {
            Self::Cpu(tower) => tower.encode_normalized(planar, width, height, max_tokens),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(tower) => tower.encode_normalized(planar, width, height, max_tokens),
        }
    }
}

/// The merged-token grid of one encoded image: `h` rows by `w` columns of
/// 2 x 2 merge windows, i.e. `(height / 32, width / 32)` for Bonsai 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridSize {
    /// Merged-grid rows (`height / (patch_size * spatial_merge)`).
    pub h: usize,
    /// Merged-grid columns (`width / (patch_size * spatial_merge)`).
    pub w: usize,
}

impl GridSize {
    /// The number of merged tokens (`h * w`) — the number of embedding rows
    /// [`VisionTower::encode`] returns and the number of `<|image_pad|>`
    /// positions the image occupies in the prompt.
    #[must_use]
    pub const fn n_tokens(self) -> usize {
        self.h * self.w
    }
}
