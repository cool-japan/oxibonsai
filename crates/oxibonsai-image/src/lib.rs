//! # oxibonsai-image
//!
//! FLUX.2 DiT (`bonsai-image`) GGUF **weight loader and configuration** for the
//! OxiBonsai text-to-image port.
//!
//! This crate (Phase 2 of the OxiBonsai → Bonsai-Image port) provides:
//!
//! - [`DitConfig`] — the diffusion-transformer architecture configuration,
//!   parsed entirely from the `bonsai-image.*` GGUF metadata namespace written
//!   by the `oxibonsai-model` MLX→GGUF converter.
//! - [`DitWeights`] — a flat, typed registry over every tensor in a
//!   `bonsai-image` DiT GGUF file: quantized linears as ternary
//!   [`oxibonsai_core::quant_ternary::BlockTQ2_0_g128`] blocks (looked up by
//!   diffusers base name), and plain tensors as BF16 (a borrowed raw-byte
//!   slice plus on-demand `u16`-bits / `f32` decoders that allocate).
//!
//! It does **not** implement the forward pass or build the nested transformer
//! block hierarchy; that is a later phase, gated on a golden-reference oracle.
//! The registry here is designed so block structs can be built on top of it.
//!
//! ## Storage conventions honoured
//!
//! 1. Quantized linear → looked up under its base name (no `.weight`), GGUF
//!    type `TQ2_0_g128`.
//! 2. Plain tensor → looked up under its full name, GGUF type `BF16`.
//! 3. All tensors are stored with their logical shape **reversed**; the
//!    accessors recover the logical shape (and, for quantized linears, the
//!    `(out, in)` dimensions).
//!
//! ## Example
//!
//! ```no_run
//! use std::path::Path;
//! use oxibonsai_image::DitWeights;
//!
//! let weights = DitWeights::open(Path::new("/path/to/bonsai-image-dit.gguf"))
//!     .expect("load DiT");
//! let cfg = weights.config();
//! assert_eq!(cfg.hidden_size(), cfg.num_attention_heads * cfg.attention_head_dim);
//!
//! // Quantized linear by diffusers base name.
//! let q = weights
//!     .quantized_linear("transformer_blocks.0.attn.to_q")
//!     .expect("to_q present");
//! assert_eq!(q.blocks.len() as u64, q.expected_block_count());
//!
//! // Plain BF16 tensor by full name, decoded on demand.
//! let emb = weights.bf16_tensor("x_embedder.weight").expect("x_embedder present");
//! let _values: Vec<f32> = emb.to_f32_vec();
//! ```

pub mod blocks;
pub mod config;
/// GPU (CUDA) backend for the DiT ternary matmuls + joint flash-attention. The
/// `target_os`-disjoint sibling of [`gpu`]; gated on `cfg(all(feature =
/// "native-cuda", any(target_os = "linux", target_os = "windows")))`. Reuses the
/// same `OXI_DIT_GPU` / `OXI_DIT_ATTN_GPU` env vars as the Metal path.
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_gpu;
pub mod error;
pub mod forward;
pub mod gemm;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod gpu;
pub mod math;
/// End-to-end text-to-image orchestration ([`pipeline::text_to_image`]): the
/// single library entry point shared by the `generate` example and the
/// `oxibonsai image` CLI subcommand (DiT → VAE → PNG, native or golden-parity).
pub mod pipeline;
pub mod png;
/// FLUX.2 native sampling scaffolding — byte-exact Pure-Rust port of MLX's
/// Threefry random generator ([`sample::mlx_rng`]) plus the FLUX.2 initial-noise,
/// position-id, and flow-match schedule generators. Makes the text-to-image
/// pipeline self-sufficient (no golden `.npy` dump for scaffolding).
pub mod sample;
/// Resident multi-prompt session: load the pipeline once, render many prompts.
pub mod session;
pub mod te;
pub mod vae;
pub mod weights;

pub use config::{DitConfig, ARCHITECTURE, DEFAULT_EPS};
pub use error::{DitError, DitResult};
pub use forward::{DitForward, ForwardTaps, QkvNorm, Stage0, StepTap};
pub use pipeline::{
    text_to_image, GoldenOverride, PipelineError, TeSource, TextToImageCfg, TextToImageOut,
};
pub use png::{encode_rgb8, PngError, PngResult};
pub use session::{ImageSession, RenderOutcome, RenderParams, StageTimings};
pub use weights::{Bf16Tensor, DitWeights, QuantizedLinear};
