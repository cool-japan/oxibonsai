//! Model types: `BonsaiModel` struct, constructors, accessors, and main forward pass.

#[cfg(test)]
use super::weight_loaders::dominant_from_counts;
use super::weight_loaders::{
    build_rope_table, build_rope_table_or_unscaled, load_f32_tensor, load_output_weight,
    load_transformer_block, resolve_id42_once, resolved_dominant_weight_quant_type,
    validate_config_shapes,
};
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
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(
        feature = "native-cuda",
        not(all(feature = "metal", target_os = "macos")),
        any(target_os = "linux", target_os = "windows")
    )
))]
use lm_head::copy_logits;
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::tensor_names;
use oxibonsai_kernels::traits::{FusedKernel, OneBitKernel};

mod embedding;
mod lm_head;
mod prefill_cpu;
#[cfg(test)]
mod tests;

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
#[cfg(all(feature = "metal", target_os = "macos"))]
mod forward_metal_fp8;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu_cache;

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
    /// Whether the file declares `prism.hadamard.*` metadata (B2-09):
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
    /// Whether the KV cache / RoPE table may grow towards [`Self::max_context`]
    /// on demand. Only the config-only constructors enable it: a GGUF-loaded
    /// model sizes its caches once, because the device-side KV cache is
    /// allocated from `kv_cache.max_seq_len()` and cannot be re-geometried.
    kv_growth: bool,
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
    /// Load a model from a parsed GGUF file.
    ///
    /// Extracts configuration from metadata, then maps all tensor data
    /// into the layer structures (zero-copy for quantized weights, including
    /// the token-embedding table).
    pub fn from_gguf(gguf: &'a GgufFile<'a>, max_seq_len: usize) -> ModelResult<Self> {
        Self::from_gguf_with_embd(gguf, max_seq_len, std::sync::Arc::from(Vec::new()))
    }

    /// Load a model from a parsed GGUF file, reusing a pre-loaded, shared token
    /// embedding table.
    ///
    /// Identical to [`from_gguf`](Self::from_gguf) in every respect (blocks,
    /// output weight, RMSNorms, RoPE, KV cache, config) **except** that the
    /// caller may supply the `token_embd` table instead of it being read from
    /// the GGUF. This is the seam the engine pool uses to share one
    /// `Arc<[f32]>` across all replicas.
    ///
    /// * A **non-empty** `token_embd` MUST be the dequantized
    ///   [`token_embd.weight`](tensor_names::TOKEN_EMBD) tensor for this exact
    ///   GGUF (`vocab_size × hidden_size` FP32, row-major); anything else
    ///   changes the model output, so a length mismatch is rejected.
    /// * An **empty** `token_embd` means "load it from this GGUF": the table is
    ///   then kept quantized and looked up row-wise (M-02), and the passed
    ///   `Arc` is retained as this model's shared handle so that every replica
    ///   built from one [`shared_token_embd`](Self::shared_token_embd) handle
    ///   still reports the same allocation.
    ///
    /// The vocab-size reconciliation below reads the GGUF tensor shape (not the
    /// slice), so the resolved `config.vocab_size` is identical either way.
    pub fn from_gguf_with_embd(
        gguf: &'a GgufFile<'a>,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
    ) -> ModelResult<Self> {
        Self::from_gguf_with_embd_and_cap(gguf, max_seq_len, token_embd, None)
    }

    /// [`from_gguf_with_embd`](Self::from_gguf_with_embd) with an explicit
    /// RAM-derived context cap (sec-11).
    ///
    /// `context_cap` is the largest context the host can afford — the value
    /// `max_context_for_budget` supplies. The effective context is
    /// `min(max_seq_len, config.max_context_length, context_cap)`; positions at
    /// or beyond it are rejected by [`forward`](Self::forward) with
    /// [`ModelError::SequenceTooLong`] instead of reading past the RoPE table.
    pub fn from_gguf_with_embd_and_cap(
        gguf: &'a GgufFile<'a>,
        max_seq_len: usize,
        token_embd: std::sync::Arc<[f32]>,
        context_cap: Option<usize>,
    ) -> ModelResult<Self> {
        let mut config = Qwen3Config::from_metadata(&gguf.metadata)?;
        if let Some(embd_info) = gguf.tensors.get(tensor_names::TOKEN_EMBD) {
            if embd_info.shape.len() >= 2 {
                let tensor_vocab = embd_info.shape[1] as usize;
                if tensor_vocab != config.vocab_size {
                    tracing::warn!(
                        metadata_vocab = config.vocab_size, tensor_vocab,
                        "vocab_size mismatch: GGUF metadata says {} but token_embd tensor has {} rows; using tensor dimension",
                        config.vocab_size, tensor_vocab,
                    );
                    config.vocab_size = tensor_vocab;
                }
            }
        }
        // M-12, load-time half: reject a head geometry that cannot produce a
        // coherent KV cache BEFORE anything is allocated from it (a zero
        // `num_kv_heads` reaches `KvCache::new` and then the GQA divide).
        // `load_transformer_block` re-checks per layer against real tensor
        // widths; this catches a file with no layers at all.
        validate_config_shapes(&config)?;
        // B2-09: resolve ggml wire id 42's on-disk layout ONCE for the whole
        // file (`Ok(None)` when the file has no wire-id-42 tensor at all) and
        // thread it into every loader below, instead of each one re-deriving
        // it from the raw parse-time guess (which is always `TQ2_0_g128`
        // regardless of the file's real layout) or -- worse -- re-resolving
        // it once per layer.
        let resolved_42 = resolve_id42_once(gguf)?;
        let dominant_quant_type = resolved_dominant_weight_quant_type(gguf)?;
        let has_hadamard = gguf.metadata.get("prism.hadamard.version").is_some();
        tracing::info!(
            layers = config.num_layers,
            hidden = config.hidden_size,
            heads = config.num_attention_heads,
            kv_heads = config.num_kv_heads,
            vocab = config.vocab_size,
            "loading BonsaiModel from GGUF"
        );
        let (token_embd_table, shared_embd) = if token_embd.is_empty() {
            // "Load it from this GGUF": quantized and row-wise where possible.
            // The passed (empty) handle is retained so that every replica built
            // from one `shared_token_embd()` still reports the same allocation.
            (
                EmbeddingTable::from_gguf(gguf, config.vocab_size, config.hidden_size)?,
                token_embd,
            )
        } else {
            let needed = config.vocab_size.saturating_mul(config.hidden_size);
            if token_embd.len() < needed {
                return Err(ModelError::ShapeMismatch {
                    name: "shared token_embd".to_string(),
                    expected: vec![config.vocab_size, config.hidden_size],
                    actual: vec![token_embd.len()],
                });
            }
            // The table itself holds the one reference; `shared_token_embd`
            // hands out clones of *that*, so keeping a second clone here would
            // inflate the engine pool's reference count for no reason.
            (
                EmbeddingTable::dense(token_embd, config.vocab_size, config.hidden_size),
                std::sync::Arc::from(Vec::new()),
            )
        };
        if token_embd_table.is_empty() {
            return Err(ModelError::ShapeMismatch {
                name: tensor_names::TOKEN_EMBD.to_string(),
                expected: vec![config.vocab_size, config.hidden_size],
                actual: vec![0],
            });
        }
        let output_norm_w = load_f32_tensor(gguf, tensor_names::OUTPUT_NORM)?;
        let output_norm = RmsNorm::new(output_norm_w, config.rms_norm_eps);
        let kernel = std::sync::Arc::new(oxibonsai_kernels::KernelDispatcher::auto_detect());
        let output_weight = load_output_weight(gguf, &config, &kernel, resolved_42)?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let block = load_transformer_block(gguf, &config, layer_idx, &kernel, resolved_42)?;
            blocks.push(block);
        }
        let max_context = effective_context(&config, Some(max_seq_len), context_cap);
        // A loaded model pre-allocates its whole effective context: it may not
        // grow later (the device KV cache is allocated from this geometry), so
        // anything smaller would silently deny the caller the context they
        // asked for. `effective_context` has already reduced `max_seq_len` by
        // the configured context length and the RAM-derived cap.
        let prealloc = prealloc_context(max_context, None);
        if prealloc < max_seq_len {
            tracing::warn!(
                requested = max_seq_len,
                effective = max_context,
                allocated = prealloc,
                config_context = config.max_context_length,
                "requested max_seq_len clamped to the model's context budget"
            );
        }
        // M-08: honour `<arch>.rope.scaling.*` — `models/Bonsai-8B.gguf`
        // declares YaRN (factor 4.0, original context 16384).
        let rope = build_rope_table(&config, prealloc)?;
        let kv_cache = KvCache::new(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            prealloc,
        );
        tracing::info!(
            blocks = blocks.len(),
            embd_elements = token_embd_table.len(),
            embd_resident_bytes = token_embd_table.resident_bytes(),
            lm_head = output_weight.kind(),
            quant = ?dominant_quant_type,
            max_seq_len = prealloc,
            max_context,
            "model loaded successfully"
        );
        Ok(Self {
            config,
            token_embd: token_embd_table,
            shared_embd,
            blocks,
            output_norm,
            output_weight,
            rope,
            kv_cache,
            dominant_quant_type,
            has_hadamard,
            scratch: ModelScratch::default(),
            max_context,
            kv_growth: false,
            host_kv_written: 0,
            gpu_path_active: std::sync::atomic::AtomicBool::new(false),
            prefill_chunk_tokens: crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS,
            lm_head_kernel: kernel,
            force_cpu_decode_after: force_cpu_decode_after_from_env(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch:
                oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch(),
        })
    }

    /// Create a model from configuration only (no weights), for testing.
    ///
    /// Allocates **O(1)** memory (M-33): the `vocab × hidden` embedding and LM
    /// head are synthesized as all-zero on demand rather than materialized
    /// (2 × 2.5 GB for the 8B config, 2 × 4.74 GiB for Bonsai 2 27B), and the
    /// KV cache starts at [`WEIGHTLESS_PREALLOC_CONTEXT`] positions and grows
    /// towards `config.max_context_length` as the sequence does. A forward pass
    /// behaves exactly as before — all-zero weights produce all-zero logits.
    pub fn new(config: Qwen3Config) -> Self {
        Self::new_with_context_cap(config, None)
    }

    /// [`new`](Self::new) with an explicit RAM-derived context cap (sec-11).
    ///
    /// The effective context is `min(config.max_context_length, context_cap)`;
    /// `None` means "no cap beyond the configuration".
    pub fn new_with_context_cap(config: Qwen3Config, context_cap: Option<usize>) -> Self {
        let h = config.hidden_size;
        let max_context = effective_context(&config, None, context_cap);
        let prealloc = prealloc_context(max_context, Some(WEIGHTLESS_PREALLOC_CONTEXT));
        let kv_cache = KvCache::new(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            prealloc,
        );
        // M-08: a config-only model gets the same table a GGUF-loaded one
        // does. Infallible constructor, so an unusable declaration degrades to
        // an unscaled table with a `tracing::error!` rather than a panic.
        let rope = build_rope_table_or_unscaled(&config, prealloc);
        Self {
            token_embd: EmbeddingTable::constant(0.0, config.vocab_size, h),
            shared_embd: std::sync::Arc::from(Vec::new()),
            blocks: Vec::new(),
            output_norm: RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
            output_weight: OutputWeight::zero_fp32(config.vocab_size, h),
            rope,
            kv_cache,
            dominant_quant_type: oxibonsai_core::GgufTensorType::Q1_0_g128,
            has_hadamard: false,
            scratch: ModelScratch::default(),
            max_context,
            kv_growth: true,
            host_kv_written: 0,
            gpu_path_active: std::sync::atomic::AtomicBool::new(false),
            prefill_chunk_tokens: crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS,
            lm_head_kernel: lm_head::weightless_lm_head_kernel(),
            force_cpu_decode_after: force_cpu_decode_after_from_env(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch:
                oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch(),
            config,
        }
    }

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
    /// Seam for the hybrid `qwen35` path (RT-28 / B2-10): Bonsai 2's 48 linear
    /// layers carry a `[48 × 128 × 128]` state matrix plus a `[3 × 10240]`
    /// convolution window per sequence, which must be cleared alongside the KV
    /// cache of the 16 full-attention layers. No block in this tree owns
    /// recurrent state yet, so this is a no-op that [`reset`](Self::reset)
    /// already calls — the hybrid block only has to fill it in.
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
    /// as well as [`OneBitKernel::upload_weights`] — a `dyn OneBitKernel`
    /// trait object cannot, and that erasure is what pinned the fused
    /// QKV / gate-up handles to the 1-bit path. `FusedKernel` has a blanket
    /// impl for every `OneBitKernel + TernaryKernel`, so every existing
    /// caller — all of which pass the concrete `KernelDispatcher` — compiles
    /// unchanged.
    pub fn upload_weights_to_gpu(&mut self, kernel: &dyn FusedKernel) {
        let n_blocks = self.blocks.len();
        if n_blocks == 0 {
            return;
        }
        tracing::info!(blocks = n_blocks, "uploading model weights to GPU");
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

    /// Ensure the caches can serve position `pos`, growing them if allowed
    /// (sec-11).
    ///
    /// # Errors
    ///
    /// [`ModelError::SequenceTooLong`] when `pos` is beyond the effective
    /// context, or beyond the allocated caches of a model that may not grow.
    fn ensure_context_capacity(&mut self, pos: usize) -> ModelResult<()> {
        if pos >= self.max_context {
            return Err(ModelError::SequenceTooLong {
                seq_len: pos + 1,
                max_ctx: self.max_context,
            });
        }
        let allocated = self.kv_cache.max_seq_len();
        if pos < allocated {
            return Ok(());
        }
        // A model whose device KV cache has already been allocated from the
        // current geometry must not silently change that geometry underneath
        // it; report the allocated limit instead.
        if !self.kv_growth || self.gpu_path_active() {
            return Err(ModelError::SequenceTooLong {
                seq_len: pos + 1,
                max_ctx: allocated,
            });
        }
        let target = allocated
            .saturating_mul(2)
            .max(pos + 1)
            .max(WEIGHTLESS_PREALLOC_CONTEXT)
            .min(self.max_context);
        self.grow_context(target)
    }

    /// Reallocate the KV cache and RoPE table to `new_len` positions, keeping
    /// every cached position.
    fn grow_context(&mut self, new_len: usize) -> ModelResult<()> {
        let layers = self.kv_cache.num_layers();
        let heads = self.kv_cache.num_kv_heads();
        let head_dim = self.kv_cache.head_dim();
        let keep = self.host_kv_valid_until().min(self.kv_cache.max_seq_len());
        let mut grown = KvCache::new(layers, heads, head_dim, new_len);
        for layer in 0..layers {
            for head in 0..heads {
                let keys = self.kv_cache.keys_for(layer, head, keep);
                let values = self.kv_cache.values_for(layer, head, keep);
                for p in 0..keep {
                    let span = p * head_dim..(p + 1) * head_dim;
                    grown.try_store_key(layer, head, p, &keys[span.clone()])?;
                    grown.try_store_value(layer, head, p, &values[span])?;
                }
            }
        }
        grown.set_seq_len(self.kv_cache.seq_len());
        self.kv_cache = grown;
        // M-08: rebuild through the same scaling-aware path the constructors
        // use — a plain `RopeTable::new` here would silently drop YaRN the
        // moment a sequence outgrew its first allocation.
        self.rope = build_rope_table(&self.config, new_len)?;
        tracing::debug!(
            new_len,
            kept = keep,
            "grew the KV cache and RoPE table on demand"
        );
        Ok(())
    }

    /// Process multiple prompt tokens in a single batch forward pass on GPU.
    ///
    /// Uses GEMM instead of GEMV for projections (processing all tokens at once),
    /// with sequential per-token attention. Only the last token's logits are
    /// returned (for generation to start). The GPU KV cache is populated for
    /// all positions.
    ///
    /// Falls back to sequential single-token forward if the GPU batch path
    /// is unavailable.
    ///
    /// A prompt longer than [`prefill_chunk_tokens`](Self::prefill_chunk_tokens)
    /// is split into non-overlapping chunks and driven through
    /// [`crate::chunked_prefill::run_chunked_prefill`] (M-18); anything that
    /// fits in one chunk takes the single-shot path unchanged, so the fused
    /// Metal batch path's dispatch granularity is unaffected for short prompts.
    pub fn forward_prefill(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        let Some(config) =
            crate::chunked_prefill::prefill_chunk_plan(token_ids.len(), self.prefill_chunk_tokens)
        else {
            return self.forward_prefill_unchunked(token_ids, pos_start, kernel);
        };
        crate::chunked_prefill::run_chunked_prefill(
            token_ids,
            config,
            |chunk| {
                self.forward_prefill_unchunked(&chunk.tokens, pos_start + chunk.start_pos, kernel)
            },
            // `PrefillFirst` never yields, so this is never called; the
            // executor also serves the interleaved priorities, where a caller
            // supplies real decode work.
            || Ok(()),
        )?
        .ok_or_else(|| {
            ModelError::Internal(
                "forward_prefill: the chunked executor produced no logits for a non-empty prompt"
                    .to_string(),
            )
        })
    }

    /// Single-shot batched prefill: the whole prompt in one call.
    ///
    /// This is exactly what [`forward_prefill`](Self::forward_prefill) was
    /// before M-18 wired the chunk scheduler in, and it is what each chunk
    /// runs through.
    fn forward_prefill_unchunked(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(ModelError::MissingTensor {
                name: "forward_prefill: empty token_ids".into(),
            });
        }
        if token_ids.len() == 1 {
            return self.forward(token_ids[0], pos_start, kernel);
        }
        // M-17: every batched prefill path below — GPU and the batched CPU one
        // — computes full causal attention. A model that declared
        // `<arch>.attention.sliding_window` must not silently get it, so it
        // falls through to `forward_sequential`, whose per-token
        // `forward_into` honours the window. See `forward_into`.
        let windowed = self.config.sliding_window.is_some();
        let _gpu_kernel = kernel.is_gpu_accelerated() && !self.force_cpu_at(pos_start) && !windowed;
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel && token_ids.len() <= 16 {
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel
            && !matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
        {
            // Fused Metal prefill supports OneBit + Ternary today. It maintains
            // the DEVICE KV cache only (MET-05).
            match self.try_metal_prefill_with_lm_head(token_ids, pos_start) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal batch prefill failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
        {
            // FP8 hybrid batch prefill (Phase 28.B): batched FP8 GEMM projections
            // on the GPU, attention + K/V store on the CPU against `self.kv_cache`
            // — the same cache the per-token FP8 decode path reads. Correct by
            // construction (no split KV cache), and a HOST-KV path for MET-05.
            let is_e4m3 = matches!(&self.output_weight, OutputWeight::FP8E4M3(_));
            match self.try_metal_prefill_with_lm_head_fp8(token_ids, pos_start, is_e4m3) {
                Ok(logits) => {
                    self.note_host_kv_written(pos_start + token_ids.len() - 1);
                    return Ok(logits);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal FP8 batch prefill failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // FP8 batch GEMM prefill (Phase 26).
            let is_e4m3 = matches!(&self.output_weight, OutputWeight::FP8E4M3(_));
            match self.try_cuda_prefill_with_lm_head_fp8(token_ids, pos_start, is_e4m3) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda FP8 batch prefill failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token CUDA GEMV.
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q4_0(_) | OutputWeight::Q8_0(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            let q4_0 = matches!(&self.output_weight, OutputWeight::Q4_0(_));
            match self.try_cuda_prefill_with_lm_head_q_std(token_ids, pos_start, q4_0) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda Q4_0/Q8_0 batch prefill failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q2K(_)
                    | OutputWeight::Q3K(_)
                    | OutputWeight::Q4K(_)
                    | OutputWeight::Q5K(_)
                    | OutputWeight::Q6K(_)
                    | OutputWeight::Q8K(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // K-quant batch GEMM prefill (Phase 25).
            let fmt = self.output_weight.k_quant_format().ok_or_else(|| {
                ModelError::Internal(format!(
                    "forward_prefill: K-quant branch entered with a {} LM head",
                    self.output_weight.kind()
                ))
            })?;
            match self.try_cuda_prefill_with_lm_head_k_quant(token_ids, pos_start, fmt) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda K-quant batch prefill failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token forward using CUDA GEMV.
            return self.forward_sequential(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel {
            match self.try_cuda_prefill_with_lm_head(token_ids, pos_start) {
                Ok(logits) => {
                    self.note_device_kv_used();
                    return Ok(logits);
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("LM head not supported on CUDA prefill path") {
                        tracing::debug!(
                            error = % e,
                            "cuda batch prefill skipped (LM head dtype not supported), using sequential"
                        );
                    } else {
                        tracing::warn!(
                            error = % e,
                            "cuda batch prefill failed, falling back to sequential"
                        );
                    }
                }
            }
        }
        // perf-M2: try the batched CPU prefill before falling back to the
        // per-token loop -- this is the hook that makes `prefill_cpu` reachable.
        // Skipped for a windowed model (M-17): `forward_prefill_cpu`'s
        // `gqa_attention_row` is unconditionally full-causal.
        if !windowed {
            if let Some(logits) = self.forward_prefill_cpu(token_ids, pos_start)? {
                return Ok(logits);
            }
        }
        self.forward_sequential(token_ids, pos_start, kernel)
    }

    /// Sequential per-token prefill fallback shared by every batched path.
    ///
    /// Guards the MET-05 transition once, up front: looping into
    /// [`forward`](Self::forward) from a sequence whose history lives in the
    /// device KV cache would attend over an all-zero host cache.
    fn forward_sequential(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        self.require_host_kv_coherent(pos_start)?;
        let mut last_logits = Vec::new();
        for (i, &token_id) in token_ids.iter().enumerate() {
            last_logits = self.forward(token_id, pos_start + i, kernel)?;
        }
        Ok(last_logits)
    }

    /// Sequential per-token verify fallback (argmax at every position).
    fn forward_sequential_verify(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<u32>> {
        self.require_host_kv_coherent(pos_start)?;
        let mut token_ids_out = Vec::with_capacity(token_ids.len());
        for (i, &token_id) in token_ids.iter().enumerate() {
            let logits = self.forward(token_id, pos_start + i, kernel)?;
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (j, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = j as u32;
                }
            }
            token_ids_out.push(best_idx);
        }
        Ok(token_ids_out)
    }

    /// Forward pass for speculative decode verification.
    ///
    /// Processes multiple tokens in batch via GPU prefill, then runs the LM head
    /// and argmax on ALL positions (not just the last). Returns the greedy
    /// argmax token ID for each input position.
    ///
    /// If GPU batch path is unavailable, falls back to sequential CPU forward
    /// with argmax at each position.
    pub fn forward_prefill_verify(
        &mut self,
        token_ids: &[u32],
        pos_start: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<u32>> {
        if token_ids.is_empty() {
            return Ok(vec![]);
        }
        // M-17: as in `forward_prefill` — every batched verify path is
        // full-causal, so a windowed model takes the sequential one.
        let _gpu_kernel = kernel.is_gpu_accelerated()
            && !self.force_cpu_at(pos_start)
            && self.config.sliding_window.is_none();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel
            && !matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
        {
            // Fused Metal prefill verify supports OneBit + Ternary today; FP8
            // falls through to per-token sequential, which dispatches through
            // `KernelDispatcher::gemv_fp8_*` (Metal GPU via Phase 27).
            match self.try_metal_prefill_verify(token_ids, pos_start) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "metal batch prefill verify failed, falling back to sequential"
                    );
                }
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // FP8 batch GEMM prefill verify (Phase 26).
            let is_e4m3 = matches!(&self.output_weight, OutputWeight::FP8E4M3(_));
            match self.try_cuda_prefill_verify_fp8(token_ids, pos_start, is_e4m3) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda FP8 batch prefill verify failed, falling back to sequential",
                ),
            }
            return self.forward_sequential_verify(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q2K(_)
                    | OutputWeight::Q3K(_)
                    | OutputWeight::Q4K(_)
                    | OutputWeight::Q5K(_)
                    | OutputWeight::Q6K(_)
                    | OutputWeight::Q8K(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // K-quant batch GEMM prefill verify (Phase 25).
            let fmt = self.output_weight.k_quant_format().ok_or_else(|| {
                ModelError::Internal(format!(
                    "forward_prefill_verify: K-quant branch entered with a {} LM head",
                    self.output_weight.kind()
                ))
            })?;
            match self.try_cuda_prefill_verify_k_quant(token_ids, pos_start, fmt) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => cuda_prefill_fallback_log(
                    &e.to_string(),
                    "cuda K-quant batch prefill verify failed, falling back to sequential",
                ),
            }
            // Fallback: sequential token-by-token with argmax.
            return self.forward_sequential_verify(token_ids, pos_start, kernel);
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel {
            match self.try_cuda_prefill_verify(token_ids, pos_start) {
                Ok(ids) => {
                    self.note_device_kv_used();
                    return Ok(ids);
                }
                Err(e) => {
                    tracing::warn!(
                        error = % e,
                        "cuda batch prefill verify failed, falling back to sequential"
                    );
                }
            }
        }
        self.forward_sequential_verify(token_ids, pos_start, kernel)
    }

    /// Forward pass for a single token at position `pos`.
    ///
    /// Returns logits over the vocabulary `[vocab_size]`. Allocates the result
    /// vector; [`forward_into`](Self::forward_into) writes into a caller-owned
    /// buffer and allocates nothing at all.
    #[tracing::instrument(skip(self, kernel), fields(token_id, pos))]
    pub fn forward(
        &mut self,
        token_id: u32,
        pos: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<Vec<f32>> {
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.forward_into(token_id, pos, kernel, &mut logits)?;
        Ok(logits)
    }

    /// Forward pass for a single token at position `pos`, writing
    /// `[vocab_size]` logits into `logits`.
    ///
    /// Allocation-free after the first call: every intermediate lives in
    /// [`ModelScratch`] (M-22).
    ///
    /// # Errors
    ///
    /// * [`ModelError::SequenceTooLong`] — `pos` is beyond the effective
    ///   context (sec-11).
    /// * [`gpu_fallback_requires_cache_rebuild`] — the GPU decode path has been
    ///   maintaining a device KV cache for this sequence and cannot fall back
    ///   to the CPU without the host cache being rebuilt first (MET-05).
    /// * [`ModelError::ShapeMismatch`] — `logits` is shorter than `vocab_size`.
    pub fn forward_into(
        &mut self,
        token_id: u32,
        pos: usize,
        kernel: &dyn OneBitKernel,
        logits: &mut [f32],
    ) -> ModelResult<()> {
        let vocab = self.config.vocab_size;
        if logits.len() < vocab {
            return Err(ModelError::ShapeMismatch {
                name: "forward logits".to_string(),
                expected: vec![vocab],
                actual: vec![logits.len()],
            });
        }
        self.ensure_context_capacity(pos)?;
        // Move the scratch out so the `&self` GPU entry points can be called
        // while its buffers are borrowed mutably. `take` moves the `Vec`s (no
        // allocation) and the buffers are put back on every exit path.
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = self.forward_core(token_id, pos, kernel, &mut scratch, logits);
        self.scratch = scratch;
        result
    }

    /// Body of [`forward_into`](Self::forward_into), with the scratch detached.
    fn forward_core(
        &mut self,
        token_id: u32,
        pos: usize,
        kernel: &dyn OneBitKernel,
        scratch: &mut ModelScratch,
        logits: &mut [f32],
    ) -> ModelResult<()> {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        if scratch.hidden.len() != h {
            scratch.hidden.resize(h, 0.0);
        }
        if scratch.normed.len() != h {
            scratch.normed.resize(h, 0.0);
        }
        self.token_embd.copy_row(token_id, &mut scratch.hidden)?;
        let t_blocks_start = std::time::Instant::now();
        // M-17: read once, up front. Every `self.blocks` loop below needs it
        // while `self.kv_cache` is mutably borrowed, so it must already be a
        // plain `Option<usize>` (it is `Copy`) rather than a field access.
        let sliding_window = self.config.sliding_window;
        // M-17: the fused whole-layer GPU paths below all compute FULL causal
        // attention — none of them takes a window argument. Claiming them for
        // a model that declared `<arch>.attention.sliding_window` would return
        // confident output that ignores the very constraint the file declares,
        // so a windowed model routes through the per-block path (which honours
        // it) instead. `None` — every shipped model — leaves this expression
        // exactly as it was.
        //
        // NOT covered here: `forward_greedy_gpu` (`forward_metal.rs`, a
        // different package's file this wave) is a *second* fused-Metal decode
        // entry point that `engine_greedy.rs` calls directly, never through
        // `forward_into`, so this gate cannot reach it. It needs the same
        // refusal at its own head — returning `Err` is enough, because its one
        // production caller already falls back to the CPU path on `Err`. The
        // exact change is recorded in this package's `deviations`.
        let _gpu_kernel =
            kernel.is_gpu_accelerated() && !self.force_cpu_at(pos) && sliding_window.is_none();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if _gpu_kernel {
            if scratch.gpu_logits.len() != vocab {
                scratch.gpu_logits.resize(vocab, 0.0);
            }
            match self.try_metal_full_forward_with_lm_head(
                &mut scratch.hidden,
                pos,
                &mut scratch.gpu_logits,
            ) {
                Ok(()) => {
                    self.note_device_kv_used();
                    // `try_metal_full_forward_with_lm_head` owns the `Vec` and
                    // may resize it; copy what it produced rather than indexing
                    // a length it did not promise (M-29: no panics here).
                    copy_logits(&scratch.gpu_logits, &mut logits[..vocab]);
                    let t_elapsed = t_blocks_start.elapsed();
                    tracing::debug!(
                        target : "fwd_profile",
                        "pos={pos} fused_gpu={:.1}ms (metal layers+norm+lm_head)", t_elapsed
                        .as_secs_f64() * 1000.0,
                    );
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused Metal forward+LM head unavailable, trying the next path"
                    );
                }
            }
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // FP8 models use CUDA-accelerated GEMV via KernelTier::Gpu block dispatch.
            // Skip the Q1/TQ2 fused CUDA graph paths (they only handle 1-bit/ternary
            // weights). This is a HOST-KV path: the blocks read and write
            // `self.kv_cache`.
            self.require_host_kv_coherent(pos)?;
            run_blocks(
                &self.blocks,
                sliding_window,
                &mut scratch.hidden,
                pos,
                &mut self.kv_cache,
                &self.rope,
                kernel,
            )?;
            self.note_host_kv_written(pos);
            let t_blocks_elapsed = t_blocks_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} fp8_cuda_dispatch={:.1}ms (cuda gemv via block dispatch)",
                t_blocks_elapsed.as_secs_f64() * 1000.0,
            );
            let t_norm_start = std::time::Instant::now();
            self.output_norm
                .forward(&scratch.hidden, &mut scratch.normed)?;
            let t_norm_elapsed = t_norm_start.elapsed();
            let t_lm_start = std::time::Instant::now();
            self.apply_lm_head(&scratch.normed, &mut logits[..vocab])?;
            let t_lm_elapsed = t_lm_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} norm={:.2}ms lm_head={:.2}ms",
                t_norm_elapsed.as_secs_f64() * 1000.0,
                t_lm_elapsed.as_secs_f64() * 1000.0,
            );
            return Ok(());
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel
            && matches!(
                &self.output_weight,
                OutputWeight::Q4_0(_)
                    | OutputWeight::Q8_0(_)
                    | OutputWeight::Q2K(_)
                    | OutputWeight::Q3K(_)
                    | OutputWeight::Q4K(_)
                    | OutputWeight::Q5K(_)
                    | OutputWeight::Q6K(_)
                    | OutputWeight::Q8K(_)
            )
            && oxibonsai_kernels::CudaGraph::global().is_ok()
        {
            // Q4_0/Q8_0 and K-quant models: skip the Q1 fused CUDA graph path.
            // Each layer GEMV dispatches to CUDA via LinearQ*::forward(), against
            // the HOST KV cache.
            self.require_host_kv_coherent(pos)?;
            run_blocks(
                &self.blocks,
                sliding_window,
                &mut scratch.hidden,
                pos,
                &mut self.kv_cache,
                &self.rope,
                kernel,
            )?;
            self.note_host_kv_written(pos);
            let t_blocks_elapsed = t_blocks_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} quant_cuda_dispatch={:.1}ms (cuda gemv via block dispatch)",
                t_blocks_elapsed.as_secs_f64() * 1000.0,
            );
            let t_norm_start = std::time::Instant::now();
            self.output_norm
                .forward(&scratch.hidden, &mut scratch.normed)?;
            let t_norm_elapsed = t_norm_start.elapsed();
            let t_lm_start = std::time::Instant::now();
            self.apply_lm_head(&scratch.normed, &mut logits[..vocab])?;
            let t_lm_elapsed = t_lm_start.elapsed();
            tracing::debug!(
                target: "fwd_profile",
                "pos={pos} norm={:.2}ms lm_head={:.2}ms",
                t_norm_elapsed.as_secs_f64() * 1000.0,
                t_lm_elapsed.as_secs_f64() * 1000.0,
            );
            return Ok(());
        }
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        if _gpu_kernel {
            match self.try_cuda_full_forward_with_lm_head(&scratch.hidden, pos) {
                Ok(fused_logits) => {
                    self.note_device_kv_used();
                    copy_logits(&fused_logits, &mut logits[..vocab]);
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused CUDA forward+LM head unavailable, trying the next path"
                    );
                }
            }
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let did_full_forward = if _gpu_kernel {
            match self.try_metal_full_forward_inner(&mut scratch.hidden, pos) {
                Ok(()) => true,
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused Metal Q1 full-layer forward unavailable, trying ternary"
                    );
                    match self.try_metal_full_forward_ternary_inner(&mut scratch.hidden, pos) {
                        Ok(()) => true,
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                pos,
                                "fused Metal ternary full-layer forward unavailable, \
                                 falling back to per-block CPU/GPU dispatch"
                            );
                            false
                        }
                    }
                }
            }
        } else {
            false
        };
        #[cfg(all(
            feature = "native-cuda",
            not(all(feature = "metal", target_os = "macos")),
            any(target_os = "linux", target_os = "windows")
        ))]
        let did_full_forward = if _gpu_kernel {
            match self.try_cuda_full_forward_inner(&scratch.hidden, pos) {
                Ok(new_hidden) => {
                    let n = new_hidden.len().min(scratch.hidden.len());
                    scratch.hidden[..n].copy_from_slice(&new_hidden[..n]);
                    true
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        pos,
                        "fused CUDA full-layer forward unavailable, falling back to \
                         per-block CPU/GPU dispatch"
                    );
                    false
                }
            }
        } else {
            false
        };
        #[cfg(not(any(
            all(feature = "metal", target_os = "macos"),
            all(
                feature = "native-cuda",
                not(all(feature = "metal", target_os = "macos")),
                any(target_os = "linux", target_os = "windows")
            )
        )))]
        let did_full_forward = false;
        if did_full_forward {
            // The fused GPU layer path maintains its own device KV cache.
            self.note_device_kv_used();
        } else {
            // Host-KV path: every block reads the history out of `self.kv_cache`,
            // so it must actually contain that history (MET-05).
            self.require_host_kv_coherent(pos)?;
            run_blocks(
                &self.blocks,
                sliding_window,
                &mut scratch.hidden,
                pos,
                &mut self.kv_cache,
                &self.rope,
                kernel,
            )?;
            self.note_host_kv_written(pos);
        }
        let t_blocks_elapsed = t_blocks_start.elapsed();
        let t_norm_start = std::time::Instant::now();
        self.output_norm
            .forward(&scratch.hidden, &mut scratch.normed)?;
        let t_norm_elapsed = t_norm_start.elapsed();
        let t_lm_start = std::time::Instant::now();
        self.apply_lm_head(&scratch.normed, &mut logits[..vocab])?;
        let t_lm_elapsed = t_lm_start.elapsed();
        tracing::debug!(
            target : "fwd_profile",
            "pos={pos} blocks={:.1}ms norm={:.1}ms lm_head={:.1}ms gpu={}",
            t_blocks_elapsed.as_secs_f64() * 1000.0, t_norm_elapsed.as_secs_f64() *
            1000.0, t_lm_elapsed.as_secs_f64() * 1000.0, did_full_forward,
        );
        Ok(())
    }
}

/// Run every Transformer block over `hidden`, honouring the configured
/// sliding-window span (M-17).
///
/// A free function rather than a method because all three call sites hold
/// disjoint borrows of `self` at once (`&self.blocks`, `&mut self.kv_cache`,
/// `&self.rope`), which only stays legal while they are separate arguments.
///
/// `sliding_window` is `Some(w)` exactly when the GGUF declared
/// `<arch>.attention.sliding_window = w`. In that case each block runs
/// [`TransformerBlock::forward_with_sliding_window`] over a local window of
/// the `w` most recent key positions; otherwise it runs the unchanged
/// [`TransformerBlock::forward`], so every model shipped today (none declares
/// the key) keeps a byte-identical forward pass.
///
/// # Why `sink_tokens: 0`
///
/// [`SlidingWindowConfig`] can also pin the first *N* positions into the
/// window as attention sinks. The GGUF key carries a width only — it is
/// llama.cpp's `n_swa`, a pure causal local window with no sinks — so
/// declaring any sink here would silently attend to positions the reference
/// implementation does not. Sinks stay reachable through
/// [`crate::layers::sliding_window`] for callers that genuinely want them.
fn run_blocks(
    blocks: &[TransformerBlock<'_>],
    sliding_window: Option<usize>,
    hidden: &mut [f32],
    pos: usize,
    kv_cache: &mut KvCache,
    rope: &RopeTable,
    kernel: &dyn OneBitKernel,
) -> ModelResult<()> {
    match sliding_window {
        None => {
            for block in blocks {
                block.forward(hidden, pos, kv_cache, rope, kernel)?;
            }
        }
        Some(window) => {
            let sw = crate::layers::sliding_window::SlidingWindowConfig::new(window, 0);
            for block in blocks {
                block.forward_with_sliding_window(
                    hidden,
                    pos,
                    kv_cache,
                    rope,
                    kernel,
                    Some(&sw),
                )?;
            }
        }
    }
    Ok(())
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

/// Effective context length for a model (sec-11).
///
/// `min(requested, config.max_context_length, cap)`, with a zero or missing
/// value treated as "no constraint from that source" and a floor of 1 so the
/// caches are never degenerate.
fn effective_context(config: &Qwen3Config, requested: Option<usize>, cap: Option<usize>) -> usize {
    let mut limit = usize::MAX;
    for value in [requested, Some(config.max_context_length), cap]
        .into_iter()
        .flatten()
    {
        if value > 0 {
            limit = limit.min(value);
        }
    }
    if limit == usize::MAX {
        MAX_PREALLOC_CONTEXT
    } else {
        limit.max(1)
    }
}

/// Positions to pre-allocate for a model whose effective context is
/// `max_context` (sec-11).
///
/// `growth_window` is `None` for a model that may **not** grow its caches — it
/// gets its whole effective context up front, because nothing can enlarge it
/// later. A model that may grow ([`BonsaiModel::ensure_context_capacity`])
/// starts at `min(max_context, window)` instead, so a mock built from a
/// production config costs kilobytes rather than gigabytes.
fn prealloc_context(max_context: usize, growth_window: Option<usize>) -> usize {
    match growth_window {
        Some(window) => max_context.clamp(1, window.max(1)),
        None => max_context.max(1),
    }
}

/// Log one CUDA batch-prefill fallback at the level it deserves (**F15.4**).
///
/// The Q4_0/Q8_0, K-quant and FP8 CUDA batch-prefill families are disabled by
/// construction — `forward_cuda::cuda_split_prefill_allowed()` is an
/// unconditional `false` while no kernels-side entry point can hand the
/// prompt's K/V back to the host — so all six entry points return the same
/// refusal on **every** prompt. Reporting that at `warn!` filled logs with a
/// line the operator can do nothing about and which does not describe a
/// failure: the bit-correct sequential prefill runs and the answer is right.
///
/// This keeps the message text unchanged and changes only its level: the
/// by-construction refusal is `debug!`, anything else — a real dispatch
/// failure, once the guard is lifted — stays `warn!`. Matching on the
/// refusal's own wording mirrors the idiom already used for the "LM head not
/// supported on CUDA prefill path" skip a few branches below.
#[cfg(all(
    feature = "native-cuda",
    not(all(feature = "metal", target_os = "macos")),
    any(target_os = "linux", target_os = "windows")
))]
fn cuda_prefill_fallback_log(error: &str, message: &'static str) {
    // The distinguishing clause of `forward_cuda::cuda_split_prefill_disabled`.
    const DISABLED_BY_CONSTRUCTION: &str = "writes a GPU-private KV cache";
    if error.contains(DISABLED_BY_CONSTRUCTION) {
        tracing::debug!(error, "{message}");
    } else {
        tracing::warn!(error, "{message}");
    }
}

/// Read the `OXIBONSAI_FORCE_CPU_DECODE_AFTER` debug/test seam once.
fn force_cpu_decode_after_from_env() -> Option<usize> {
    parse_force_cpu_after(std::env::var("OXIBONSAI_FORCE_CPU_DECODE_AFTER").ok())
}

/// Parse the `OXIBONSAI_FORCE_CPU_DECODE_AFTER` value; anything unparsable
/// means "never force the CPU", exactly as `InferenceEngine::generate` treats
/// the same variable.
fn parse_force_cpu_after(raw: Option<String>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse().ok())
}

impl BonsaiModel<'static> {
    /// Build a model with real (but tiny, deterministic) Transformer blocks
    /// for testing the prefix-cache path.
    ///
    /// Unlike [`BonsaiModel::new`] (which leaves `blocks` empty), this
    /// constructor instantiates `config.num_layers` real
    /// [`crate::block::TransformerBlock`]s backed by leaked weight
    /// allocations. The leaked memory is acceptable in tests, where the
    /// process is short-lived. The resulting model writes its KV cache via
    /// the standard CPU forward path, allowing the prefix cache to be
    /// exercised end-to-end.
    pub fn new_for_testing_with_blocks(config: Qwen3Config) -> Self {
        use crate::block::TransformerBlock;
        use crate::layers::linear::{Linear1Bit, LinearLayer};
        use half::f16;
        use oxibonsai_core::tensor::BlockQ1_0G128;
        use oxibonsai_kernels::{KernelDispatcher, KernelTier};
        use std::sync::Arc;

        let h = config.hidden_size;
        let hd = config.head_dim;
        let nq = config.num_attention_heads;
        let nkv = config.num_kv_heads;
        let inter = config.intermediate_size;

        // Q1_0_g128 packs 128 weights per block → blocks_per_in_row = in_features / 128.
        // We require in_features % 128 == 0.
        assert!(
            h.is_multiple_of(128),
            "test fixture requires hidden_size to be a multiple of 128"
        );
        assert!(
            inter.is_multiple_of(128),
            "test fixture requires intermediate_size to be a multiple of 128"
        );

        let h_bpr = h / 128;
        let inter_bpr = inter / 128;

        // Force the Reference (CPU) tier so the CPU `KvCache` is populated by
        // the forward path. With auto_detect on a GPU host the dispatcher
        // would route through Metal/CUDA, leaving the CPU cache empty and
        // breaking prefix-cache tests that round-trip through it.
        let kernel_arc = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));
        let max_context = effective_context(&config, None, None);
        let prealloc = prealloc_context(max_context, Some(MAX_PREALLOC_CONTEXT));
        let kv_cache = KvCache::new(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            prealloc,
        );
        let rope = build_rope_table_or_unscaled(&config, prealloc);

        // Helper: build a leaked Vec<BlockQ1_0G128> with deterministic data so
        // the test fixture is reproducible. Returns a 'static slice.
        fn make_blocks_static(n: usize, scale: f32, pattern: u8) -> &'static [BlockQ1_0G128] {
            let v: Vec<BlockQ1_0G128> = (0..n)
                .map(|i| BlockQ1_0G128 {
                    d: f16::from_f32(scale),
                    qs: [pattern.wrapping_add((i & 0xff) as u8); 16],
                })
                .collect();
            // Leak the allocation so the slice lives for 'static. Acceptable in tests.
            Box::leak(v.into_boxed_slice())
        }

        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            // Per-block weight allocations (leaked).
            let q_blk = make_blocks_static(nq * hd * h_bpr, 0.01, 0xA5);
            let k_blk = make_blocks_static(nkv * hd * h_bpr, 0.01, 0x5A);
            let v_blk = make_blocks_static(nkv * hd * h_bpr, 0.01, 0x33);
            let o_blk = make_blocks_static(h * (nq * hd / 128).max(1), 0.01, 0xCC);
            let g_blk = make_blocks_static(inter * h_bpr, 0.01, 0x77);
            let u_blk = make_blocks_static(inter * h_bpr, 0.01, 0x88);
            let d_blk = make_blocks_static(h * inter_bpr, 0.01, 0x99);

            let attn_q: LinearLayer<'static> =
                Linear1Bit::new(q_blk, nq * hd, h, kernel_arc.clone())
                    .expect("q proj")
                    .into();
            let attn_k: LinearLayer<'static> =
                Linear1Bit::new(k_blk, nkv * hd, h, kernel_arc.clone())
                    .expect("k proj")
                    .into();
            let attn_v: LinearLayer<'static> =
                Linear1Bit::new(v_blk, nkv * hd, h, kernel_arc.clone())
                    .expect("v proj")
                    .into();
            let attn_out: LinearLayer<'static> =
                Linear1Bit::new(o_blk, h, nq * hd, kernel_arc.clone())
                    .expect("o proj")
                    .into();
            let ffn_gate: LinearLayer<'static> =
                Linear1Bit::new(g_blk, inter, h, kernel_arc.clone())
                    .expect("gate proj")
                    .into();
            let ffn_up: LinearLayer<'static> = Linear1Bit::new(u_blk, inter, h, kernel_arc.clone())
                .expect("up proj")
                .into();
            let ffn_down: LinearLayer<'static> =
                Linear1Bit::new(d_blk, h, inter, kernel_arc.clone())
                    .expect("down proj")
                    .into();

            let block = TransformerBlock::new(
                layer_idx,
                RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
                attn_q,
                attn_k,
                attn_v,
                attn_out,
                RmsNorm::new(vec![1.0; hd], config.rms_norm_eps),
                RmsNorm::new(vec![1.0; hd], config.rms_norm_eps),
                RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
                ffn_gate,
                ffn_up,
                ffn_down,
                nq,
                nkv,
                hd,
                h,
            );
            blocks.push(block);
        }

        Self {
            // The fixture's embedding is a constant 0.01 in every element, as
            // before — now synthesized per row instead of materialized.
            token_embd: EmbeddingTable::constant(0.01, config.vocab_size, h),
            shared_embd: std::sync::Arc::from(Vec::new()),
            blocks,
            output_norm: RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
            output_weight: OutputWeight::zero_fp32(config.vocab_size, h),
            rope,
            kv_cache,
            dominant_quant_type: oxibonsai_core::GgufTensorType::Q1_0_g128,
            has_hadamard: false,
            scratch: ModelScratch::default(),
            max_context,
            kv_growth: true,
            host_kv_written: 0,
            gpu_path_active: std::sync::atomic::AtomicBool::new(false),
            prefill_chunk_tokens: crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS,
            lm_head_kernel: kernel_arc,
            force_cpu_decode_after: force_cpu_decode_after_from_env(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch:
                oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch(),
            config,
        }
    }
}
