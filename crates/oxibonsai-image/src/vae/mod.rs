//! Pure-Rust FLUX.2 SMALL VAE **decoder** (latent → RGB), validated
//! stage-by-stage against golden MLX tensors.
//!
//! This is the decode-only half of the FLUX.2 (`bonsai-image`) VAE: it takes the
//! DiT's packed latent `[1, 128, 32, 32]`, denormalises it with stored
//! BatchNorm statistics, unpatchifies it to `[1, 32, 64, 64]`, and runs the
//! convolutional decoder (`post_quant_conv → conv_in → mid_block → up_blocks →
//! conv_norm_out → conv_out`) to produce `[1, 3, 512, 512]`. Everything is
//! computed in f32 on flat NCHW (batch-1) buffers.
//!
//! Weights are loaded from the per-tensor `.npy` files exported by
//! `/tmp/bonsai_vae_export_weights.py` via [`weights::VaeWeights`]. The conv
//! weight layout is the MLX `[out, kH, kW, in]` (see [`weights`]).
//!
//! The decoder supports both **untiled** (single forward pass,
//! `decode_packed_latents`) and **tiled** (`decode_packed_latents_tiled`)
//! modes. The tiled path tiles only the final `silu→conv_out` stage with a
//! 1-pixel halo (TileBoundary::AfterConvNormOut) so peak activation is
//! O(tile_size) instead of O(output_resolution²). Results are numerically
//! identical (cos ≥ 0.999 vs the untiled path). See [`tiling`] for details.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use oxibonsai_image::vae::{VaeWeights, VaeDecoder};
//!
//! let weights = VaeWeights::open(Path::new("/tmp/bonsai_golden/vae/weights"))
//!     .expect("open vae weights");
//! let decoder = VaeDecoder::from_weights(&weights).expect("build decoder");
//! // packed: flat NCHW [1,128,32,32] latent from the DiT.
//! let packed = vec![0.0f32; 128 * 32 * 32];
//! let rgb = decoder
//!     .decode_packed_latents(&packed, 32, 32, None)
//!     .expect("decode");
//! assert_eq!((rgb.c, rgb.h, rgb.w), (3, 512, 512));
//! ```

pub mod attention;
pub mod conv;
/// GPU (CUDA) f32 backend for the VAE decoder convs/norms/silu/upsample. The
/// `target_os`-disjoint sibling of `gpu`; gated on `cfg(all(feature =
/// "native-cuda", any(target_os = "linux", target_os = "windows")))`. Reuses the
/// same `OXI_VAE_GPU` env var as the Metal path.
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_gpu;
pub mod decoder;
pub mod error;
/// GPU (Metal) f32 backend for the VAE decoder convs/norms/silu/upsample. Gated
/// on `cfg(all(feature = "metal", target_os = "macos"))`; opt-in via
/// `OXI_VAE_GPU=1`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod gpu;
pub mod norm;
pub mod ops;
pub mod resnet;
/// Pure-Rust loader for the FLUX.2 `AutoencoderKLFlux2` VAE weights read directly
/// from the canonical diffusers `*.safetensors` (bf16→f32, conv-weight
/// transpose, `to_out.0` un-nesting). The self-serve alternative to the `.npy`
/// golden dump; selected automatically by [`weights::VaeWeights::open`] for a
/// `.safetensors` path.
pub mod safetensors;
/// Spatial tiling helpers for memory-bounded VAE decode at resolutions > 512 px.
///
/// The tiled path (`VaeDecoder::decode_packed_latents_tiled`) has two boundaries.
/// The default (`TileBoundary::AfterConvNormOut`) tiles only the final
/// `silu → conv_out` stage (k=3, pad=1, halo=1 px). `TileBoundary::AfterMid`
/// additionally bounds the up-block convolution im2col (the real multi-GB peak)
/// via `Conv2d::forward_tiled`. Both keep every GroupNorm plane whole (correct
/// global per-group statistics), so results are bit-identical to the untiled CPU
/// path (and `cos ≈ 1` vs the GPU path, whose sum reassociation depends on tile
/// size).
pub mod tiling;
pub mod weights;

pub use decoder::{DecodeTaps, Map, VaeDecoder};
pub use error::{VaeError, VaeResult};
pub use tiling::{TileBoundary, TileConfig};
pub use weights::{Tensor, VaeWeights};
