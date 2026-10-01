//! Batched prefill (multi-token) GPU dispatch for OxiBonsai.
//!
//! Split into:
//! - `types`     — `PrefillBuffers`, `LayerWeightRefs`, `LayerConfig` (`pub(crate)`)
//! - `attention` — batched attention pipelines/dispatch + the bucketed
//!   `PrefillBufferCache` (perf-01 / perf-M3)
//! - `functions` — `MetalGraph` impl: the micro-batched prefill runner with
//!   its bounded command-buffer waits (M-18), the per-layer encoders and the
//!   `encode_full_forward_prefill*` entry points
//! - `functions_2` — public `try_metal_full_forward_prefill*` entry points
//! - `hidden` — the head-free hidden-state prefill for embeddings, on a
//!   request-scoped device KV cache (MET-05)

pub(crate) mod attention;
pub(crate) mod functions;
pub(crate) mod functions_2;
pub(crate) mod hidden;
pub(crate) mod types;

// `MetalGraph`'s resident prefill buffers are held as the capacity-tracking
// `PrefillBufferCache` (perf-M3); `types::*` is consumed inside this module
// only, so it is no longer re-exported.
pub(crate) use attention::PrefillBufferCache;

pub use functions::PREFILL_LOGITS_MICRO_BATCH;
pub use functions_2::*;
pub use hidden::{
    try_metal_full_forward_prefill_hidden, try_metal_full_forward_prefill_hidden_cached,
    try_metal_full_forward_prefill_hidden_ternary, HiddenPrefillInput, HiddenPrefillShape,
    HIDDEN_PREFILL_MICRO_BATCH,
};
