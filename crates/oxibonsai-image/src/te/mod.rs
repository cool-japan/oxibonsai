//! Pure-Rust Qwen3-4B **text encoder** for Bonsai-Image (FLUX.2 Klein).
//!
//! Produces the `[seq, 7680]` conditioning the DiT `context_embedder` consumes,
//! from token ids, by running a single full causal Qwen3-4B forward and stacking
//! the pre-final-norm hidden states of decoder layers 8/17/26 (1-indexed
//! `hidden_states_list[9/18/27]`).
//!
//! The 4-bit (mlx packed-affine, bits=4, group_size=64) weights are dequantised
//! to f32 offline (`/tmp/bonsai_te_export_weights.py`) and loaded via
//! [`TeWeights`]; the forward runs entirely in f32, validated against golden
//! MLX tensors (cosine ≥ 0.999 per layer and on the stacked cond) by the
//! `te_parity` example.
//!
//! # Pipeline position
//!
//! ```text
//! prompt → [tokenizer] → input_ids[512] ─┐
//!                                          ├─→ TextEncoder::forward → cond[512,7680]
//!          attention_mask[512] ───────────┘                              │
//!                                                            DiT context_embedder
//! ```
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use oxibonsai_image::te::{TeWeights, TextEncoder};
//!
//! let weights = TeWeights::open(Path::new("/tmp/bonsai_golden/te/weights"))
//!     .expect("open TE weights");
//! let encoder = TextEncoder::new(&weights);
//! let ids: Vec<u32> = vec![151644, 872, 198 /* … */];
//! let mask: Vec<i32> = vec![1; ids.len()];
//! let out = encoder.forward(&ids, &mask).expect("forward");
//! let cond = out.cond_7680().expect("cond"); // [seq * 7680]
//! ```

pub mod config;
/// GPU (CUDA) f32 matmul backend for the TE Linears. The `target_os`-disjoint
/// sibling of `gpu` (the Metal module, macOS-only); gated on `cfg(all(feature = "native-cuda", any(target_os =
/// "linux", target_os = "windows")))`. Reuses the same `OXI_TE_GPU` env var as
/// the Metal path (opt-in `=1`).
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_gpu;
pub mod error;
pub mod forward;
/// GPU (Metal) f32 matmul backend for the TE Linears. Gated on
/// `cfg(all(feature = "metal", target_os = "macos"))`; opt-in via `OXI_TE_GPU=1`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod gpu;
/// Pure-Rust MLX 4-bit (packed-affine) weight loader: loads the native ~2.1 GB
/// 4-bit safetensors directly instead of the ~15 GB dequantised f32 `.npy` dump.
pub mod mlx4bit;
pub mod rope;
pub mod tokenizer;
pub mod weights;

pub use config::{TeConfig, STACK_LAYERS};
pub use error::{TeError, TeResult};
pub use forward::{Precision, TeOutput, TextEncoder};
pub use mlx4bit::{dequantize_mlx_4bit_affine, Mlx4bitModel};
pub use rope::Qwen3Rope;
pub use tokenizer::{Qwen3Tokenizer, TokenizerOutput};
pub use weights::{read_npy_f32, TeWeights, Tensor};
