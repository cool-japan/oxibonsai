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

/// Whether a split-KV-cache CUDA batch prefill may run for a family whose
/// caller stores **no** K/V read-back into `self.kv_cache` — finding **F6**.
///
/// Always `false`. K-quant (`k_quant`) and FP8 (`forward_cuda_fp8`) drive a
/// GPU-private device KV cache during batch prefill while their decode
/// attends over the host cache, and their callers do not yet write the
/// prompt's K/V back into it (the kernels-side `kv_readback_out` of
/// `try_cuda_prefill_k_quant` / `try_cuda_prefill_fp8_with_kv_readback` is
/// there; the model side passes `None`), so there is no configuration in
/// which those paths produce grounded output. The Q4_0/Q8_0 family, whose
/// caller does store the read-back, is gated by
/// [`cuda_split_prefill_allowed_with_readback`] instead.
///
/// This is a runtime predicate rather than an unconditional `return Err` so
/// the batch-prefill implementations below it stay compiled and
/// type-checked instead of rotting behind dead code.
pub(super) fn cuda_split_prefill_allowed() -> bool {
    false
}

/// Whether a split-KV-cache CUDA batch prefill whose caller writes the
/// device K/V read-back into `self.kv_cache` may run at `pos_start` —
/// finding **F6** (the Q4_0/Q8_0 family, `q_std`).
///
/// Only at `pos_start == 0`. Token `t` of the batch attends over device
/// positions `[0, pos_start + t]`, and the family's device KV cache is
/// GPU-private and process-global (keyed on geometry alone): positions
/// before `pos_start` hold whatever an earlier call left there — possibly
/// for another sequence or model, and never the positions the host-KV
/// decode path or a sequential fallback wrote — and nothing records which.
/// At `pos_start == 0` the call itself writes every position it reads, so
/// the logits and the read-back are both grounded; any later chunk or turn
/// takes the sequential path, which reads the host cache the read-back
/// filled.
pub(super) fn cuda_split_prefill_allowed_with_readback(pos_start: usize) -> bool {
    pos_start == 0
}

/// The error the refusing split-KV-cache entry points return — finding
/// **F6** — for a family whose caller stores no read-back (see
/// [`cuda_split_prefill_allowed`]).
///
/// Q4_0/Q8_0 (`q_std`), every K-quant (`k_quant`) and FP8
/// (`forward_cuda_fp8`) each drive a **GPU-private** device KV cache during
/// batch prefill (`acquire_q_std_kv_cache` / `acquire_k_quant_kv_cache` /
/// `acquire_fp8_kv_cache` — three separate process-global slots), while
/// DECODE for those three families runs CPU attention over `self.kv_cache`.
/// Without a read-back, a "successful" GPU prefill leaves decode attending
/// over all-zero prompt KV: fluent, confident, entirely ungrounded output,
/// with no error anywhere. The Q1/ternary path does not have this problem
/// because its prefill and decode share one device KV cache
/// (`cuda_full_layer::acquire_kv_cache`).
///
/// The Q4_0/Q8_0 entry points read the device K/V back
/// (`cuda_full_layer::read_back_kv_cache`) and write it into
/// `self.kv_cache` ([`BonsaiModel::store_cuda_kv_readback`]), so they
/// refuse only a `pos_start > 0` call, through
/// [`cuda_split_prefill_needs_history`]. The K-quant and FP8 entry points
/// still return this error on every call, and `forward_prefill` falls back
/// to the bit-correct sequential per-token path, which populates
/// `self.kv_cache` directly.
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
    warn_on_ignored_force_override();
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

/// The error a read-back-storing split-KV-cache entry point (Q4_0/Q8_0)
/// returns for a call at `pos_start > 0` — finding **F6**, see
/// [`cuda_split_prefill_allowed_with_readback`].
///
/// The message carries the same "writes a GPU-private KV cache" clause as
/// [`cuda_split_prefill_disabled`], which is what `prefill_dispatch`'s
/// fallback logger keys its `debug!` level on: this is a by-construction
/// refusal, not a failure.
pub(super) fn cuda_split_prefill_needs_history(
    family: &str,
    verify: bool,
    pos_start: usize,
) -> Box<dyn std::error::Error> {
    warn_on_ignored_force_override();
    let what = if verify {
        "batch prefill verify"
    } else {
        "batch prefill"
    };
    format!(
        "{family} CUDA {what} at pos_start {pos_start} disabled: this path writes a GPU-private \
         KV cache that holds no history before pos_start (earlier positions live only in the \
         host cache); using the bit-correct sequential fallback"
    )
    .into()
}

/// Log, once per process, that `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL` is set
/// but no longer does anything.
fn warn_on_ignored_force_override() {
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
}

impl BonsaiModel<'_> {
    /// Write a CUDA batch prefill's device K/V read-back (finding **F6**)
    /// into `self.kv_cache` at `[pos_start, pos_start + batch_size)`, one
    /// `(keys, values)` pair per block in `self.blocks` order (the order the
    /// batch-prefill entry points lay their layers out in), each stored
    /// under the block's own `layer_index()` — see
    /// `prefill_dispatch::store_kv_readback` for the checks.
    ///
    /// Run on hardware (RTX A4000, CUDA 12.0, 2026-10-07): CUDA-P11 passed on
    /// real Q4_0 / Q8_0 weights through this store (read-back against the
    /// sequential host KV, min cos 1.000000); the host-side store is also
    /// unit-tested.
    ///
    /// # Errors
    /// A layer-count / shape mismatch, a window past the cache's limit, or
    /// a failed cache growth.
    pub(super) fn store_cuda_kv_readback(
        &mut self,
        readback: &[(Vec<f32>, Vec<f32>)],
        pos_start: usize,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let layer_indices: Vec<usize> = self.blocks.iter().map(|b| b.layer_index()).collect();
        super::prefill_dispatch::store_kv_readback(
            &mut self.kv_cache,
            &layer_indices,
            readback,
            pos_start,
            batch_size,
        )?;
        Ok(())
    }
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
/// It then releases the model's **Q1 slot namespace** too (`MET-02`, CUDA
/// half — `q1.rs`'s module docs): every norm / final-norm / LM-head /
/// weight-fallback slot composed over this model's epoch, from all three
/// CUDA caches — plus the ternary, Q4_0/Q8_0 and K-quant layouts of the same
/// namespace
/// (`SlotNamespace::cuda_ternary_keys`,
/// `SlotNamespace::cuda_std_quant_keys`,
/// `SlotNamespace::cuda_k_quant_keys`).
/// Those slots are unique to this model (no other model can compose them),
/// so evicting them can only ever free this model's buffers; without it,
/// per-load slots would leak one copy of the norms and LM head per
/// load/unload cycle (the old shared literals were at least reused).
///
/// Mirrors the shape of the Metal branch in `gpu_cache.rs`: the two are
/// mutually exclusive by `target_os`, so only one `Drop` impl for
/// `BonsaiModel` is ever compiled.
///
/// **Never panics.** `CudaGraph::global()` is fallible (no device, no driver,
/// a poisoned lock), and a `Drop` that unwraps would abort the process during
/// unwinding. Both fallible calls are logged and swallowed.
///
/// Hardware status (RTX A4000, CUDA 12.0, 2026-10-07): F-M3 was observed
/// only partially — nothing was evicted while in use, and VRAM was back to
/// idle at process exit, but the release on drop itself was not observable
/// in-process.
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
        // to be: only `q1::get_or_build_cuda_qkv_cache`
        // populates that field, while the ternary path caches its
        // concatenation in its own field (`cuda_ternary_qkv_cache`) and the
        // Q4_0/Q8_0, K-quant and FP8 CUDA paths build theirs locally. A model
        // that used one of those skipped its release entirely and leaked its
        // GPU weights for the life of the process — the exact leak F-M3 is
        // about.
        //
        // `cuda_context_is_live` reads the lock-free mirror of the singleton's
        // state: exact for every path, and it constructs nothing.
        if !oxibonsai_kernels::gpu_backend::cuda_graph::functions::cuda_context_is_live() {
            return;
        }

        let epoch = self.cuda_model_epoch;
        match oxibonsai_kernels::CudaGraph::global() {
            Ok(graph) => {
                match graph.release_model_epoch(epoch) {
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
                }
                let slots = self.cuda_q1_slots();
                let mut slot_keys = slots.cuda_keys(self.blocks.len());
                slot_keys.extend(slots.cuda_ternary_keys(self.blocks.len()));
                slot_keys.extend(slots.cuda_std_quant_keys(self.blocks.len()));
                slot_keys.extend(slots.cuda_k_quant_keys(self.blocks.len()));
                match graph.release_weights(&slot_keys) {
                    Ok(0) => {}
                    Ok(released) => tracing::debug!(
                        epoch,
                        released,
                        "released this model's CUDA Q1 slot namespace on drop"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        epoch,
                        "BonsaiModel::drop: releasing the CUDA Q1 slot namespace failed; its \
                         norm / LM-head buffers may outlive it"
                    ),
                }
            }
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
