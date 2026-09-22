//! Batched prefill (multi-token) GPU dispatch for OxiBonsai.
//!
//! Split into:
//! - `types`     — `PrefillBuffers`, `LayerWeightRefs`, `LayerConfig` (`pub(crate)`)
//! - `attention` — batched attention pipelines/dispatch + the bucketed
//!   `PrefillBufferCache` (perf-01 / perf-M3)
//! - `functions` — `MetalGraph` impl: encoder helpers + `encode_full_forward_prefill*`
//! - `functions_2` — public `try_metal_full_forward_prefill*` entry points

pub(crate) mod attention;
pub(crate) mod functions;
pub(crate) mod functions_2;
pub(crate) mod types;

// `MetalGraph`'s resident prefill buffers are held as the capacity-tracking
// `PrefillBufferCache` (perf-M3); `types::*` is consumed inside this module
// only, so it is no longer re-exported.
pub(crate) use attention::PrefillBufferCache;

pub use functions_2::*;
