//! CUDA GPU forward-pass methods for `BonsaiModel`.
//!
//! Phase 32 split the monolithic `forward_cuda.rs` (1801 lines) into focused
//! sub-modules grouped by quant family.  Each sub-module contributes its own
//! `impl<'a> BonsaiModel<'a> { ... }` block (Rust supports multiple inherent
//! impl blocks across files for the same type).
//!
//! Sub-modules:
//!   - [`byte_helpers`]: free `#[repr(C)]` block → `&[u8]` zero-copy casts for
//!     every K-quant and standard-quant block format.
//!   - [`q1`]: 1-bit (Q1) helper builders plus all top-level dispatch entry
//!     points (`try_cuda_full_forward_inner`, `try_cuda_full_forward_with_lm_head`,
//!     `try_cuda_prefill_with_lm_head`, `try_cuda_prefill_verify`).  The
//!     dispatchers route to ternary / Q-std / FP8 / K-quant paths as needed.
//!   - [`ternary`]: TQ2 ternary helper builders and the dedicated ternary
//!     batch-prefill methods.
//!   - [`q_std`]: Q4_0 / Q8_0 helper builders and the dedicated Q-std
//!     batch-prefill methods.
//!   - [`k_quant`]: Q2K / Q3K / Q4K / Q5K / Q6K / Q8K helper builders and the
//!     K-quant batch-prefill methods.
//!
//! FP8 batch-prefill methods live in the sibling `forward_cuda_fp8` module
//! (`crates/oxibonsai-model/src/model/types/forward_cuda_fp8.rs`).

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

mod byte_helpers;
mod k_quant;
mod q1;
mod q_std;
mod ternary;

use super::BonsaiModel;

/// Environment variable that used to force the split-KV-cache CUDA batch
/// prefill on (finding **F6**). It is read only to warn that it no longer does.
const FORCE_SPLIT_PREFILL_ENV: &str = "OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL";

/// One-shot latch for the "override is no longer honoured" diagnostic.
static SPLIT_PREFILL_OVERRIDE_NOTICE: std::sync::Once = std::sync::Once::new();

/// Whether a split-KV-cache CUDA batch prefill may run — finding **F6**.
///
/// Always `false`: no kernels-side entry point can hand the prompt's K/V back
/// to the host yet (`try_cuda_prefill_q_std` / `_k_quant` / `_fp8` keep their
/// device KV cache in a module-private slot and expose no read-back), so there
/// is no configuration in which these paths produce grounded output.
///
/// This is a runtime predicate rather than an unconditional `return Err` so the
/// batch-prefill implementations below it stay compiled and type-checked
/// instead of rotting behind dead code — they are complete, and only the KV
/// hand-off is missing. When the read-back lands, this is the single place that
/// turns all six entry points back on.
pub(super) fn cuda_split_prefill_allowed() -> bool {
    false
}

/// The error the six refusing entry points return — finding **F6**.
///
/// Refuse a split-KV-cache CUDA batch prefill, and say why.
///
/// Q4_0/Q8_0 (`q_std`), every K-quant (`k_quant`) and FP8 (`forward_cuda_fp8`)
/// each drive a **GPU-private** device KV cache during batch prefill
/// (`acquire_q_std_kv_cache` / `acquire_k_quant_kv_cache` /
/// `acquire_fp8_kv_cache` — three separate process-global slots), while DECODE
/// for those three families runs CPU attention over `self.kv_cache`. Nothing
/// ever copies the prompt's K/V from the device cache back to the host one, so
/// a "successful" GPU prefill leaves decode attending over all-zero prompt KV:
/// fluent, confident, entirely ungrounded output, with no error anywhere. The
/// Q1/ternary path does not have this problem because its prefill and decode
/// share one device KV cache (`cuda_full_layer::acquire_kv_cache`).
///
/// Until a GPU→host read-back exists (see the package's recorded deviations for
/// the exact signature change the three kernels-side entry points need), these
/// six entry points return `Err` and `forward_prefill` falls back to the
/// bit-correct sequential per-token path, which populates `self.kv_cache`.
///
/// `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL` previously turned the guard off and ran
/// the GPU path anyway, "for throughput microbenchmarks only". It is **no
/// longer honoured**: the only thing it could produce was a fast wrong answer,
/// and nothing downstream distinguished that from a real one. Setting it now
/// logs once at `error` level and changes nothing else.
pub(super) fn cuda_split_prefill_disabled(
    family: &str,
    verify: bool,
) -> Box<dyn std::error::Error> {
    if std::env::var_os(FORCE_SPLIT_PREFILL_ENV).is_some() {
        SPLIT_PREFILL_OVERRIDE_NOTICE.call_once(|| {
            tracing::error!(
                "{FORCE_SPLIT_PREFILL_ENV} is set but is NO LONGER HONOURED. It used to run the \
                 Q4_0/Q8_0, K-quant and FP8 CUDA batch prefill against a GPU-private KV cache \
                 that CPU decode never reads, so generation continued against all-zero prompt \
                 K/V — fast, and silently wrong. The bit-correct sequential prefill runs instead."
            );
        });
    }
    let what = if verify {
        "batch prefill verify"
    } else {
        "batch prefill"
    };
    format!(
        "{family} CUDA {what} disabled: this path writes a GPU-private KV cache that CPU decode \
         never reads back, so its output would be silently ungrounded; using the bit-correct \
         sequential fallback"
    )
    .into()
}

/// Release this model's CUDA-resident weight buffers when it is dropped
/// (CUDA-SAFETY F-M3, model half).
///
/// The CUDA weight cache is process-global and keyed by
/// `(model_epoch, handle)`, so without this an unloaded model's buffers stay
/// resident for the life of the process — a hot-reload or an engine-pool
/// churn leaks the whole model's weights every cycle. `BonsaiModel` allocates
/// its epoch once at load (`next_cuda_model_epoch()`), and this hands that
/// epoch back to
/// [`CudaGraph::release_model_epoch`](oxibonsai_kernels::CudaGraph::release_model_epoch),
/// which frees everything registered under it.
///
/// Mirrors the shape of the Metal branch in `gpu_cache.rs`: the two are
/// mutually exclusive by `target_os`, so only one `Drop` impl for
/// `BonsaiModel` is ever compiled.
///
/// **Never panics.** `CudaGraph::global()` is fallible (no device, no driver,
/// a poisoned lock), and a `Drop` that unwraps would abort the process during
/// unwinding. Both fallible calls are logged and swallowed.
///
/// **Compile-blind.** This host has no CUDA, so this branch has been written
/// against the API and mirrored from the Metal path but never compiled or
/// run; see the package's recorded deviations.
impl Drop for BonsaiModel<'_> {
    fn drop(&mut self) {
        // Gate on "did this process ever open a CUDA context".
        //
        // `CudaGraph::global()` CONSTRUCTS the singleton on first call — it
        // would open the device and build the context for a model that never
        // touched CUDA, just to find nothing registered under its epoch — so
        // the gate must not be that call.
        //
        // It also must not be `cuda_qkv_cache.is_some()`, which is what it used
        // to be (wave-2.5 deviation #14): only `q1::get_or_build_cuda_qkv_cache`
        // populates that field, while the ternary, Q4_0/Q8_0, K-quant and FP8
        // CUDA paths build their QKV concatenation locally. A model that used
        // one of those skipped its release entirely and leaked its GPU weights
        // for the life of the process — the exact leak F-M3 is about.
        //
        // `cuda_context_is_live` reads the lock-free mirror of the singleton's
        // state: exact for every path, and it constructs nothing.
        if !oxibonsai_kernels::gpu_backend::cuda_graph::functions::cuda_context_is_live() {
            return;
        }

        let epoch = self.cuda_model_epoch;
        match oxibonsai_kernels::CudaGraph::global() {
            Ok(graph) => match graph.release_model_epoch(epoch) {
                Ok(released) => {
                    if released > 0 {
                        tracing::debug!(
                            epoch,
                            released,
                            "released this model's CUDA weight buffers on drop"
                        );
                    }
                }
                Err(e) => tracing::warn!(
                    error = %e,
                    epoch,
                    "BonsaiModel::drop: releasing the CUDA weight cache failed; its GPU \
                     buffers may outlive it"
                ),
            },
            // No CUDA context was ever created, so there is nothing cached
            // under this epoch and nothing to release.
            Err(e) => tracing::debug!(
                error = %e,
                epoch,
                "BonsaiModel::drop: no CUDA context; nothing to release"
            ),
        }
    }
}
