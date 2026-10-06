//! Model types: `BonsaiModel` struct, accessors, and the module tree that
//! holds its constructors and forward paths.
//!
//! | file | role |
//! |---|---|
//! | `mod.rs` | the struct, per-sequence state, accessors |
//! | `constructors.rs` | `from_gguf*`, `new*`, context sizing |
//! | `context.rs` | on-demand growth of the host caches |
//! | `decode.rs` | `forward` / `forward_into` and the per-block loop |
//! | `prefill_dispatch.rs` | `forward_prefill*` / verify dispatch ladder |
//! | `forward_metal.rs` | the fused Metal decode / prefill entry points and the M-18 prefill router |
//! | `forward_metal_hidden.rs` | the head-free Metal prefill `forward_hidden` tries first |
//! | `q1_slots.rs` | the per-GGUF-mapping GPU slot namespace + registry |
//! | `testing_fixture.rs` | `new_for_testing_with_blocks` + the Q1 replica fixture |

#[cfg(test)]
use super::weight_loaders::dominant_from_counts;
use crate::block::TransformerBlock;
use crate::error::{ModelError, ModelResult};
use crate::kv_cache::KvCache;
use crate::layers::linear::{Linear1Bit, LinearFP8E4M3, LinearFP8E5M2, LinearTernary};
use crate::layers::linear_kquant_ext::{LinearQ5K, LinearQ6K};
use crate::layers::linear_kquant_full::{LinearQ2K, LinearQ3K, LinearQ4K, LinearQ8K};
use crate::layers::linear_standard::{LinearQ4_0, LinearQ8_0};
use crate::layers::rms_norm::RmsNorm;
use crate::layers::rope::RopeTable;
use crate::model_registry::ModelVariant;
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::traits::FusedKernel;

mod constructors;
mod context;
mod decode;
mod embedding;
mod forward_hidden;
mod lm_head;
mod prefill_cpu;
mod prefill_dispatch;
pub(crate) mod q1_slots;
mod testing_fixture;
#[cfg(test)]
mod tests;

#[cfg(test)]
use crate::model::weight_loaders::load_f32_tensor;
/// The wire-id-42 layout helpers the hybrid (`qwen35`) loader shares with
/// the dense loader, re-exported crate-wide from here because the
/// parent `model` module keeps `weight_loaders` private: one resolver, so
/// the two load paths can never disagree about a file's layout.
pub(crate) use crate::model::weight_loaders::{
    apply_resolved_type, resolve_id42_once, tensor_data_resolved,
};
#[cfg(test)]
use constructors::{effective_context, parse_force_cpu_after, prealloc_context};
use decode::{in_autorelease_pool, run_blocks};
use embedding::EmbeddingTable;

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod forward_cuda;
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
mod forward_cuda_fp8;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod forward_metal;
#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod forward_metal_tests;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use forward_metal::Q1MetalSlots;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod forward_metal_fp8;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod forward_metal_hidden;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu_cache;
/// The uncached ternary binding (`BonsaiModel::ternary_gpu_binding`) and its
/// tail, nameable outside the crate — the decode-throughput A/B drives the
/// uncached path with them.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use gpu_cache::{TernaryGpuBinding, TernaryTailBinding};

/// The release hook of a GPU slot namespace on this build: on Metal,
/// `MetalGraph::release_model(epoch)` through the live session (never opening
/// a device); `None` elsewhere, where nothing is keyed on the namespace's
/// epoch (the CUDA build releases its tagged slots in `BonsaiModel`'s drop).
pub(crate) fn gpu_release_hook() -> Option<q1_slots::ReleaseHook> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        Some(forward_metal::release_metal_mapping)
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        None
    }
}

/// A fresh, never-reused epoch for a GPU slot namespace nobody else joins (a
/// standalone `TransformerBlock`): from the CUDA model-epoch counter on a
/// CUDA build — where every model's slots are composed over its
/// `cuda_model_epoch`, so drawing from the same counter keeps a standalone
/// block's slots disjoint from every model's — and from the Metal /
/// generic GPU counter otherwise.
pub(crate) fn fresh_slot_epoch() -> u64 {
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch()
    }
    #[cfg(not(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    )))]
    {
        oxibonsai_kernels::gpu_backend::next_gpu_model_epoch()
    }
}

/// Growth window for a model that pre-allocates lazily but has no smaller
/// budget of its own, and the fallback context when nothing declares one.
///
/// A **loaded** model does *not* use this as a ceiling: it pre-allocates its
/// full effective context, because that is exactly what the caller asked for
/// (`--max-seq-len`, already reduced by `config.max_context_length` and by the
/// RAM-derived cap in [`effective_context`]) and because its device-side KV
/// cache is allocated from the same geometry and cannot be re-geometried later.
/// Bonsai 2 27B's `qwen35.context_length = 262144` — 16 full-attention layers ×
/// 4 KV heads × 262 144 × 256 × 4 B × 2 ≈ **137 GB** — is bounded by those two
/// inputs, not by this constant; a position past the effective context is a
/// typed [`ModelError::SequenceTooLong`] rather than a RoPE table read past its
/// last row.
const MAX_PREALLOC_CONTEXT: usize = 4096;

/// Pre-allocated context for the weight-less, config-only constructors (M-33).
///
/// [`BonsaiModel::new`] is a *test/mock* seam (`InferenceEngine::new`, the
/// engine pool's placeholder replicas, router tests). It used to pre-allocate a
/// 4096-position KV cache unconditionally — 1.2 GB for the 8B config — on top
/// of two `vocab × hidden` f32 tables. It now starts at this length and grows
/// on demand ([`BonsaiModel::ensure_context_capacity`]), keeping the whole
/// constructor under 64 MiB for every shipped config.
const WEIGHTLESS_PREALLOC_CONTEXT: usize = 128;

/// [`ModelError::error_code`] of the distinguished "GPU→CPU fallback needs a
/// KV-cache rebuild" error (MET-05); see
/// [`ModelError::GpuFallbackRequiresCacheRebuild`]. Public API the runtime may
/// key on, but the seam is the two functions below — construct with
/// [`gpu_fallback_requires_cache_rebuild`], match with
/// [`gpu_fallback_cache_rebuild_pos`], never by comparing codes by hand.
pub const GPU_FALLBACK_CACHE_REBUILD_CODE: &str = "GPU_FALLBACK_REQUIRES_CACHE_REBUILD";

/// Build the distinguished GPU-fallback error for position `pos`.
///
/// See [`GPU_FALLBACK_CACHE_REBUILD_CODE`].
pub fn gpu_fallback_requires_cache_rebuild(pos: usize) -> ModelError {
    ModelError::GpuFallbackRequiresCacheRebuild { pos }
}

/// Recover the position from an error built by
/// [`gpu_fallback_requires_cache_rebuild`], or `None` for any other error.
pub fn gpu_fallback_cache_rebuild_pos(err: &ModelError) -> Option<usize> {
    match err {
        ModelError::GpuFallbackRequiresCacheRebuild { pos } => Some(*pos),
        _ => None,
    }
}

/// Per-model scratch buffers reused by every `forward` (M-22).
///
/// `forward` used to allocate three `Vec`s per token — the embedding row, the
/// normalized hidden state and the logits (607 KB for the shipped 151 669-token
/// vocabulary, 993 KB for Bonsai 2's 248 320) — plus one more on the fused GPU
/// path. They now live here, sized once on the first call and reused verbatim
/// afterwards; `forward_into` makes **zero** heap allocations per token, which
/// `tests::forward_into_is_allocation_free_per_token` asserts with the shared
/// counting allocator.
///
/// Held behind the same `&mut self` as the caches (never a `Mutex`): a hybrid
/// block will need `&mut` recurrent state, and a `&self` GPU path that could
/// also reach the scratch would be a re-entrancy hazard.
#[derive(Default)]
struct ModelScratch {
    /// Hidden state, `[hidden_size]`.
    hidden: Vec<f32>,
    /// Output-norm result, `[hidden_size]`.
    normed: Vec<f32>,
    /// Logits staging buffer for the fused GPU path, `[vocab_size]`.
    ///
    /// The Metal full-forward entry point takes `&mut Vec<f32>` (it may resize
    /// it), so the caller's `&mut [f32]` cannot be handed to it directly.
    gpu_logits: Vec<f32>,
}

/// The complete Bonsai model (Qwen3 architecture) with loaded weights.
///
/// Lifetime `'a` is tied to the memory-mapped GGUF data.
pub struct BonsaiModel<'a> {
    config: Qwen3Config,
    /// Token embedding table: `[vocab_size × hidden_size]`.
    ///
    /// Kept in its on-disk quantized form and dequantized one row at a time
    /// into [`ModelScratch::hidden`] (M-02 / M-32 / perf-04) — see
    /// [`embedding::EmbeddingTable`] for the resident-set table and for the
    /// dense escape hatch the batched GPU gather paths still use.
    token_embd: EmbeddingTable<'a>,
    /// Shared FP32 handle for the engine-pool seam.
    ///
    /// Non-empty only when the embedding really is a dense FP32 table; a
    /// quantized table is borrowed from the memory map, so replicas share the
    /// mapping itself and there is nothing to hand out. Replicas built through
    /// [`from_gguf_with_embd`](Self::from_gguf_with_embd) keep the *same*
    /// `Arc` either way, so the pool's identity check still holds.
    shared_embd: std::sync::Arc<[f32]>,
    /// Transformer blocks.
    pub(crate) blocks: Vec<TransformerBlock<'a>>,
    /// Final output RMSNorm.
    output_norm: RmsNorm,
    /// Output (LM head) weight blocks.
    output_weight: OutputWeight<'a>,
    /// RoPE precomputed tables.
    rope: RopeTable,
    /// KV cache.
    kv_cache: KvCache,
    /// Dominant tensor quantization type, detected at load time for variant
    /// identification (M-25: weight tensors only, deterministic on ties).
    dominant_quant_type: oxibonsai_core::GgufTensorType,
    /// Whether the file declares `prism.hadamard.*` metadata:
    /// distinguishes the Hadamard-folded Bonsai 2 27B family
    /// (`TernaryBonsai227b{Pq2,Ptq1,Q2g64}`) from the un-folded gen-1
    /// `Bonsai27B`, which `dominant_quant_type` alone cannot (both can
    /// resolve to the same quant type, e.g. `PQ2_0`). See
    /// [`ModelVariant::detect_qwen35_27b`].
    has_hadamard: bool,
    /// Reusable per-token buffers (M-22).
    scratch: ModelScratch,
    /// Highest context position this model will accept, `pos < max_context`
    /// (sec-11): the configured context length, clamped by the caller's
    /// request and by an optional RAM-derived cap.
    max_context: usize,
    /// Number of leading positions this model has itself written into the host
    /// [`KvCache`]. Combined with [`KvCache::seq_len`] (which external writers
    /// such as the prefix cache advertise) it is the coherence watermark
    /// MET-05 checks before letting a CPU block path attend over history.
    host_kv_written: usize,
    /// `true` once a fused GPU path that maintains its own **device** KV cache
    /// has run for the current sequence (MET-05).
    ///
    /// Atomic rather than a plain `bool` because the latch must be settable
    /// from `&self`: `forward_greedy_gpu` — the Metal decode entry point the
    /// engine calls directly — takes `&self`, and widening it would be a
    /// source-breaking API change. `Relaxed` is sufficient: the flag guards a
    /// decision on the thread that owns this model, not a data handoff.
    gpu_path_active: std::sync::atomic::AtomicBool,
    /// Prompt length above which [`Self::forward_prefill`] chunks (M-18).
    prefill_chunk_tokens: usize,
    /// Dispatcher the dense-FP32 LM head projects through (K-12/M-23).
    ///
    /// Every other projection reaches `oxibonsai-kernels` through the
    /// `Arc<KernelDispatcher>` its own [`LinearLayer`] holds; the
    /// [`OutputWeight::Fp32`] head has no `LinearLayer`, so it carries the
    /// model's dispatcher here instead of calling a crate-private copy of the
    /// kernel. A GGUF-loaded model shares the very `Arc` its blocks were built
    /// with, so this costs one refcount and no extra detection.
    lm_head_kernel: std::sync::Arc<oxibonsai_kernels::KernelDispatcher>,
    /// Debug/test seam: force the CPU path from this position onward.
    ///
    /// Read once at construction from `OXIBONSAI_FORCE_CPU_DECODE_AFTER` (the
    /// same variable `InferenceEngine::generate` reads for the greedy loop) so
    /// the *sampled* path — which goes through [`Self::forward`], not through
    /// the greedy loop — can be driven into its GPU→CPU transition without a
    /// real GPU fault. Never read per token: that would allocate.
    force_cpu_decode_after: Option<usize>,
    /// Cached GPU weight handles for zero-overhead Metal decode path.
    /// Populated on first GPU forward pass, reused on subsequent calls.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gpu_weight_cache: std::sync::Mutex<Option<oxibonsai_kernels::CachedModelWeights>>,
    /// The Metal weight-cache namespace of this model's GGUF **mapping**
    /// (MET-02): joined at construction, shared by every replica of the
    /// mapping (and handed to every block), released with its last replica —
    /// see `forward_metal::Q1MetalSlots` and `q1_slots`.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    metal_q1_slots: forward_metal::Q1MetalSlots,
    /// Cached per-layer QKV concatenated bytes for CUDA path (built once, reused).
    /// Avoids repeated heap allocation on every token during CUDA decode.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    cuda_qkv_cache: std::sync::Mutex<Option<std::sync::Arc<Vec<Vec<u8>>>>>,
    /// This load's CUDA weight-cache epoch (CUDA-SAFETY F-M3, model half):
    /// allocated once here, released by `Drop` in `forward_cuda`.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    cuda_model_epoch: u64,
}

impl<'a> BonsaiModel<'a> {
    /// Get model configuration.
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Cheaply clone a handle to the shared token-embedding table.
    ///
    /// The engine pool calls this on replica `#1` and passes the clone to
    /// [`from_gguf_with_embd`](Self::from_gguf_with_embd) when building further
    /// replicas. This is an atomic refcount bump, never a data copy, and it
    /// never materializes anything:
    ///
    /// * **Dense (FP32/F16) embedding** — returns the one shared `Arc<[f32]>`
    ///   every replica indexes, exactly as before.
    /// * **Quantized embedding** (every shipped Bonsai model: `token_embd` is
    ///   `TQ2_0_g128`/`Q1_0_g128`) — returns the empty handle. There is no
    ///   dense allocation to share because each replica reads rows straight out
    ///   of the same memory-mapped GGUF, which is strictly cheaper than sharing
    ///   one dequantized copy. Passing that handle back into
    ///   `from_gguf_with_embd` is the documented "load it from the GGUF" case,
    ///   so all replicas still report one identical handle.
    pub fn shared_token_embd(&self) -> std::sync::Arc<[f32]> {
        match self.token_embd.dense_handle() {
            Some(dense) => dense,
            None => std::sync::Arc::clone(&self.shared_embd),
        }
    }

    /// Get mutable reference to KV cache.
    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }

    /// Read-only access to the KV cache.
    ///
    /// Used by the prefix-cache-aware engine to extract previously computed
    /// blocks after a prefill so they can be inserted into the prefix-cache trie.
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    /// Reset all per-sequence state for a new conversation.
    ///
    /// Clears the host KV cache, the device-side (Metal) KV cache, the
    /// backend-coherence latch of MET-05 and the recurrent state seam
    /// (M-Missed-3). Everything that survives is immutable weight data.
    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.host_kv_written = 0;
        // Drop the MET-05 backend latch: the next sequence starts at position 0,
        // where the CPU path needs no history and is allowed again. This is also
        // what makes the device-resident KV cache harmless across a reset —
        // attention at `pos` reads only `0..=pos`, every entry of which the new
        // sequence rewrites before reading.
        self.set_gpu_path_active(false);
        self.reset_gpu_kv_cache();
        self.reset_recurrent();
    }

    /// Reset the KV cache (alias for `reset`).
    pub fn reset_cache(&mut self) {
        self.reset();
    }

    /// Release the device-resident KV cache of the fused GPU decode path
    /// (M-Missed-3).
    ///
    /// The fused Metal path keeps its KV in the process-global
    /// `MetalGraph::kv_cache` (`metal_full_layer::GpuKvCache`), which
    /// `BonsaiModel::reset` never touched on its own. Two halves:
    ///
    /// * **Correctness** — handled in [`reset`](Self::reset) itself: the
    ///   backend latch and the host watermark are cleared, and a new sequence
    ///   rewrites every position before attending to it, so no stale device
    ///   entry is ever read as history even while this method is a no-op.
    /// * **Footprint / hygiene** — handled here, via the public
    ///   `MetalGraph::clear_global_kv_cache_if_present()` seam added in
    ///   `oxibonsai-kernels` for this: it releases the buffers without ever
    ///   constructing a `MetalGraph` for a model that never touched the fused
    ///   GPU path (that singleton's `global()` would otherwise open the
    ///   default Metal device and compile every pipeline just to find nothing
    ///   to clear). A poisoned lock is logged, not propagated: `reset` cannot
    ///   fail, and the host-side reset above already keeps the model correct
    ///   regardless of whether the device buffers actually shrink.
    fn reset_gpu_kv_cache(&mut self) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Err(e) = oxibonsai_kernels::MetalGraph::clear_global_kv_cache_if_present() {
            tracing::warn!(
                error = %e,
                "failed to release the device-resident Metal KV cache on reset"
            );
        }
    }

    /// Reset recurrent (linear-attention / Gated-DeltaNet) state.
    ///
    /// A dense Qwen3 stack has **no** recurrent layers, so there is no state
    /// here to clear: this method exists so the dense and hybrid models
    /// present one reset seam (RT-28). Recurrent state lives where the
    /// recurrent layers do — in [`crate::hybrid::HybridModel`], whose
    /// `reset_recurrent` zeroes every Gated-DeltaNet state and conv window,
    /// and which [`crate::hybrid::LoadedModel::reset_recurrent`] /
    /// [`crate::hybrid::LoadedModel::set_recurrent_state`] dispatch to. A
    /// `qwen35` file can never load as a `BonsaiModel` (see
    /// [`crate::hybrid::LoadedModel`]), so this can never be the no-op
    /// standing in for state that exists.
    pub fn reset_recurrent(&mut self) {}

    /// Whether a fused GPU path with its own device KV cache has run for the
    /// current sequence (MET-05).
    pub fn gpu_path_active(&self) -> bool {
        self.gpu_path_active
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Set the MET-05 backend latch.
    fn set_gpu_path_active(&self, active: bool) {
        self.gpu_path_active
            .store(active, std::sync::atomic::Ordering::Relaxed);
    }

    /// Tell the model that the host KV cache now holds `positions` valid
    /// leading entries (MET-05).
    ///
    /// The runtime calls this after replaying the committed prefix through a
    /// CPU dispatcher in response to
    /// [`gpu_fallback_requires_cache_rebuild`]; afterwards the CPU path is
    /// allowed to continue from `positions`.
    pub fn mark_host_kv_rebuilt(&mut self, positions: usize) {
        self.host_kv_written = self.host_kv_written.max(positions);
        self.set_gpu_path_active(false);
    }

    /// Highest accepted position + 1 for this model (sec-11).
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Prompt length above which [`Self::forward_prefill`] chunks (M-18).
    pub fn prefill_chunk_tokens(&self) -> usize {
        self.prefill_chunk_tokens
    }

    /// Set the chunking threshold; `0` disables chunking entirely (M-18).
    ///
    /// See [`crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS`] for why
    /// the default is deliberately large enough to be a no-op.
    pub fn set_prefill_chunk_tokens(&mut self, tokens: usize) {
        self.prefill_chunk_tokens = tokens;
    }

    /// Resident bytes this model owns: caches and any dense weight copies.
    ///
    /// Excludes everything borrowed from the memory-mapped GGUF (all quantized
    /// weights, and the token embedding unless a dense consumer materialized
    /// it). This is the number the M-02 / M-33 memory acceptance tests read.
    pub fn footprint_bytes(&self) -> usize {
        let rope_bytes = if self.rope.max_seq_len() > 0 {
            self.rope.cos_at(0).len() * self.rope.max_seq_len() * 2 * std::mem::size_of::<f32>()
        } else {
            0
        };
        self.token_embd.resident_bytes()
            + self.output_weight.resident_bytes()
            + self.kv_cache.memory_bytes()
            + rope_bytes
            + (self.scratch.hidden.len()
                + self.scratch.normed.len()
                + self.scratch.gpu_logits.len())
                * std::mem::size_of::<f32>()
    }

    /// Upload all weight matrices across every Transformer block to GPU memory.
    ///
    /// Should be called once after model loading and before the first
    /// forward pass. If the kernel does not support GPU caching (e.g. CPU
    /// tiers), this is a cheap no-op.
    ///
    /// # M-21: `&dyn FusedKernel`, not `&dyn OneBitKernel`
    ///
    /// Widened so [`TransformerBlock::upload_to_gpu`] can reach
    /// [`TernaryKernel::upload_weights_ternary`](oxibonsai_kernels::TernaryKernel::upload_weights_ternary)
    /// as well as
    /// [`OneBitKernel::upload_weights`](oxibonsai_kernels::OneBitKernel::upload_weights)
    /// — a `dyn OneBitKernel` trait object cannot, and that erasure is what
    /// pinned the fused QKV / gate-up handles to the 1-bit path. `FusedKernel`
    /// has a blanket impl for every `OneBitKernel + TernaryKernel`, so every
    /// existing caller — all of which pass the concrete `KernelDispatcher` —
    /// compiles unchanged.
    pub fn upload_weights_to_gpu(&mut self, kernel: &dyn FusedKernel) {
        let n_blocks = self.blocks.len();
        if n_blocks == 0 {
            return;
        }
        tracing::info!(blocks = n_blocks, "uploading model weights to GPU");
        // On a Metal build the uploads go through the scirs2-core GPU backend,
        // whose Metal objects are autoreleased: pooled here so they are freed
        // when the upload returns, not when the calling thread exits.
        in_autorelease_pool(|| {
            for block in &mut self.blocks {
                block.upload_to_gpu(kernel);
            }
            match self.output_weight {
                OutputWeight::OneBit(ref mut linear) => linear.upload_to_gpu(),
                OutputWeight::Ternary(ref mut linear) => linear.upload_to_gpu(),
                OutputWeight::FP8E4M3(_)
                | OutputWeight::FP8E5M2(_)
                | OutputWeight::Q4_0(_)
                | OutputWeight::Q8_0(_)
                | OutputWeight::Q5K(_)
                | OutputWeight::Q6K(_)
                | OutputWeight::Q2K(_)
                | OutputWeight::Q3K(_)
                | OutputWeight::Q4K(_)
                | OutputWeight::Q8K(_) => {}
                OutputWeight::Fp32 { .. } => {}
            }
        });
        tracing::info!("GPU weight upload complete");
    }

    /// Detect the model variant from the loaded configuration and dominant
    /// tensor type. Tries the Bonsai 2 27B / `qwen35` family first (M-14),
    /// falling through to the legacy 8B/4B/1.7B-family detection.
    pub fn variant(&self) -> ModelVariant {
        ModelVariant::from_config_and_resolved_sample(
            &self.config,
            self.dominant_quant_type,
            self.has_hadamard,
        )
    }

    /// Approximate total number of parameters in the model.
    pub fn num_parameters(&self) -> u64 {
        self.variant().param_count()
    }

    /// Approximate model size in bytes (on disk).
    pub fn model_size_bytes(&self) -> u64 {
        self.variant().expected_model_size_bytes()
    }

    /// Maximum context length from the configuration.
    pub fn context_length(&self) -> usize {
        self.config.max_context_length
    }

    /// Dominant weight quantization type detected at load time (M-25) —
    /// what `KernelDispatcher::kernel_label` needs to name the kernel family
    /// actually running (cli-16) instead of a hardcoded one.
    pub fn dominant_quant_type(&self) -> oxibonsai_core::GgufTensorType {
        self.dominant_quant_type
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Hidden dimension size.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Current KV cache memory usage in bytes.
    pub fn kv_cache_memory_bytes(&self) -> usize {
        self.kv_cache.memory_bytes()
    }

    /// Load a model from GGUF with auto-detected variant.
    ///
    /// Same as `from_gguf` but also logs the detected model variant.
    pub fn from_gguf_auto(gguf: &'a GgufFile<'a>, max_seq_len: usize) -> ModelResult<Self> {
        let model = Self::from_gguf(gguf, max_seq_len)?;
        let variant = model.variant();
        tracing::info!(
            variant = variant.name(),
            params = variant.param_count(),
            "auto-detected model variant"
        );
        Ok(model)
    }

    /// Number of host-KV positions that are known to be valid (MET-05).
    ///
    /// The maximum of what this model wrote itself and what an external writer
    /// (the prefix cache's `inject_block` + `set_seq_len`, speculative decode's
    /// cursor moves) has advertised through [`KvCache::seq_len`].
    fn host_kv_valid_until(&self) -> usize {
        self.host_kv_written.max(self.kv_cache.seq_len())
    }

    /// Refuse to run a **host**-KV attention step whose history the host cache
    /// does not actually hold (MET-05).
    ///
    /// Returns the distinguished [`gpu_fallback_requires_cache_rebuild`] error
    /// when a fused GPU path has been maintaining a device-side KV cache for
    /// this sequence and the host cache is short of `pos`. The alternative —
    /// what this code did before — is to attend over zeros and return
    /// `Ok(logits)` that look entirely plausible.
    fn require_host_kv_coherent(&self, pos: usize) -> ModelResult<()> {
        if self.gpu_path_active() && pos > self.host_kv_valid_until() {
            return Err(gpu_fallback_requires_cache_rebuild(pos));
        }
        Ok(())
    }

    /// Record that a host-KV path wrote positions `..=pos`.
    fn note_host_kv_written(&mut self, pos: usize) {
        self.host_kv_written = self.host_kv_written.max(pos + 1);
        self.set_gpu_path_active(false);
    }

    /// Record that a device-KV (fused GPU) path handled positions `..=pos`.
    ///
    /// Takes `&self` so the `&self` GPU entry points — notably
    /// [`forward_greedy_gpu`](Self::forward_greedy_gpu) — can latch it too
    /// (MET-05 runtime half): that path maintained the device KV cache without
    /// ever telling the model, under-reporting GPU activity on the most-used
    /// decode path.
    pub(super) fn note_device_kv_used(&self) {
        self.set_gpu_path_active(true);
    }

    /// Whether the GPU paths must be skipped for `pos` (test seam, MET-05).
    fn force_cpu_at(&self, pos: usize) -> bool {
        self.force_cpu_decode_after
            .is_some_and(|after| pos >= after)
    }
}

/// Output projection can be 1-bit, ternary, FP8, Q4_0, Q8_0, Q5_K, Q6_K, or FP32.
pub(super) enum OutputWeight<'a> {
    OneBit(Linear1Bit<'a>),
    Ternary(LinearTernary<'a>),
    FP8E4M3(LinearFP8E4M3<'a>),
    FP8E5M2(LinearFP8E5M2<'a>),
    /// 4-bit symmetric (Q4_0) output projection.
    Q4_0(LinearQ4_0<'a>),
    /// 8-bit symmetric (Q8_0) output projection.
    Q8_0(LinearQ8_0<'a>),
    /// 5-bit K-quant (Q5_K) output projection.
    Q5K(LinearQ5K<'a>),
    /// 6-bit K-quant (Q6_K) output projection.
    Q6K(LinearQ6K<'a>),
    /// 2-bit K-quant (Q2_K) output projection.
    Q2K(LinearQ2K<'a>),
    /// 3-bit K-quant (Q3_K) output projection.
    Q3K(LinearQ3K<'a>),
    /// 4-bit K-quant (Q4_K) output projection.
    Q4K(LinearQ4K<'a>),
    /// 8-bit K-quant (Q8_K) output projection.
    Q8K(LinearQ8K<'a>),
    /// Dense FP32 output projection.
    ///
    /// An **empty** `weights` slice is the compact representation of the
    /// all-zero LM head used by the weight-less constructors (M-33): it
    /// produces exactly the zero logits a materialized zero matrix would,
    /// without the `vocab × hidden` allocation (2.5 GB for the 8B config).
    Fp32 {
        weights: Vec<f32>,
        out_features: usize,
        in_features: usize,
    },
}
