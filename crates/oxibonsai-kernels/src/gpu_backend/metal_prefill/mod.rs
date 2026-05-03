//! Batched prefill (multi-token) GPU dispatch for OxiBonsai.
//!
//! Split into:
//! - `types`     — `PrefillBuffers`, `LayerWeightRefs`, `LayerConfig` (`pub(crate)`)
//! - `functions` — `MetalGraph` impl: encoder helpers + `encode_full_forward_prefill*`
//! - `functions_2` — public `try_metal_full_forward_prefill*` entry points

pub(crate) mod types;
pub(crate) mod functions;
pub(crate) mod functions_2;

pub(crate) use types::*;

pub use functions_2::*;
