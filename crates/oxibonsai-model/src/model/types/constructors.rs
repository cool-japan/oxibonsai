//! `BonsaiModel` constructors: GGUF loading and the config-only (weight-less)
//! test/mock seam, plus the context-sizing helpers they share.
//!
//! Split out of `model/types/mod.rs` (that file had reached the
//! 2000-line ceiling). Every item keeps its original body; only its home
//! moved.

use super::embedding::EmbeddingTable;
use super::lm_head;
use super::WEIGHTLESS_PREALLOC_CONTEXT;
use super::{BonsaiModel, ModelScratch, OutputWeight, MAX_PREALLOC_CONTEXT};
use crate::block::TransformerBlock;
use crate::error::{ModelError, ModelResult};
use crate::kv_cache::{KvCache, KvCacheBacking, GROWTH_CHUNK_POSITIONS};
use crate::layers::rms_norm::RmsNorm;
use crate::model::weight_loaders::{
    build_rope_table, build_rope_table_or_unscaled, load_f32_tensor, load_output_weight,
    load_transformer_block, resolve_id42_once, resolved_dominant_weight_quant_type,
    validate_config_shapes,
};
use oxibonsai_core::config::Qwen3Config;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::tensor_names;

/// The GPU slot namespaces a new model takes and hands to its blocks
/// (`MET-02`).
///
/// - **Metal:** the model **joins** the namespace of the GGUF mapping its
///   weights are borrowed from (`super::q1_slots`) — one epoch shared by every
///   replica of the mapping — and every block gets that same state, so the
///   per-layer block path keys its norms exactly like the fused paths.
///   Registration happens here, at construction, so a model's slots are fixed
///   for its whole life (see the `q1_slots` module docs for why).
/// - **CUDA:** the model mints its own `cuda_model_epoch` (CUDA pools hold one
///   replica), composes its Q1 slots over it, and hands its blocks a private
///   namespace under that epoch; the model's drop releases the tagged slots.
/// - **Neither:** nothing is keyed on a namespace; blocks keep the standalone
///   namespace `TransformerBlock::new` gave them.
pub(super) struct ModelGpuSlots {
    /// The mapping's Metal namespace.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) metal_q1_slots: super::forward_metal::Q1MetalSlots,
    /// The model's CUDA weight-cache epoch.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) cuda_model_epoch: u64,
}

impl ModelGpuSlots {
    /// Take the namespaces for a model made of `blocks` and `output`, and give
    /// every block its share.
    pub(super) fn attach(blocks: &mut [TransformerBlock<'_>], output: &OutputWeight<'_>) -> Self {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let metal_q1_slots = super::forward_metal::Q1MetalSlots::for_mapping(
                super::q1_slots::mapping_anchor(blocks, output),
            );
            for block in blocks.iter_mut() {
                block.set_slot_namespace(std::sync::Arc::clone(metal_q1_slots.state()));
            }
            Self { metal_q1_slots }
        }
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            let _ = output;
            let cuda_model_epoch =
                oxibonsai_kernels::gpu_backend::cuda_graph_slot::next_cuda_model_epoch();
            let state = super::q1_slots::MappingState::private(cuda_model_epoch, None);
            for block in blocks.iter_mut() {
                block.set_slot_namespace(std::sync::Arc::clone(&state));
            }
            Self { cuda_model_epoch }
        }
        #[cfg(not(any(
            all(feature = "metal", target_os = "macos"),
            all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            )
        )))]
        {
            let _ = (blocks, output);
            Self {}
        }
    }
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
        // Resolve ggml wire id 42's on-disk layout ONCE for the whole
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
        // MET-02: join this GGUF mapping's GPU slot namespace (shared with
        // every replica of it) and hand it to the blocks.
        #[cfg_attr(
            not(any(
                all(feature = "metal", target_os = "macos"),
                all(
                    feature = "native-cuda",
                    any(target_os = "linux", target_os = "windows")
                )
            )),
            allow(unused_variables)
        )]
        let gpu_slots = ModelGpuSlots::attach(&mut blocks, &output_weight);
        let max_context = effective_context(&config, Some(max_seq_len), context_cap);
        // The whole effective context is the model's LOGICAL window: the RoPE
        // table covers it from load (a few MB), and the host KV cache reports
        // it as its fixed `max_seq_len()` — which is what every device-side KV
        // cache is sized from, so that geometry never changes mid-sequence.
        // `effective_context` has already reduced `max_seq_len` by the
        // configured context length and the RAM-derived cap.
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
        // The host KV cache is `f16` (half the bytes of
        // the old `f32` default, and the element type the fused GPU paths
        // and the reference implementations keep their KV in) and LAZY —
        // one growth chunk resident at load, grown by the decode loop
        // (`ensure_context_capacity` -> `try_ensure_capacity(pos + 1)`) only
        // as positions are actually reached on a host-KV path. A decode that
        // runs entirely on the fused GPU path never grows it at all.
        let kv_cache = KvCache::try_new_lazy(
            KvCacheBacking::DenseF16,
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            prealloc,
            GROWTH_CHUNK_POSITIONS,
        )?;
        tracing::info!(
            blocks = blocks.len(),
            embd_elements = token_embd_table.len(),
            embd_resident_bytes = token_embd_table.resident_bytes(),
            lm_head = output_weight.kind(),
            quant = ?dominant_quant_type,
            max_seq_len = prealloc,
            max_context,
            kv_resident_bytes = kv_cache.memory_bytes(),
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
            host_kv_written: 0,
            gpu_path_active: std::sync::atomic::AtomicBool::new(false),
            prefill_chunk_tokens: crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS,
            lm_head_kernel: kernel,
            force_cpu_decode_after: force_cpu_decode_after_from_env(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal_q1_slots: gpu_slots.metal_q1_slots,
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_ternary_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch: gpu_slots.cuda_model_epoch,
        })
    }

    /// Create a model from configuration only (no weights), for testing.
    ///
    /// Allocates **O(1)** memory (M-33): the `vocab × hidden` embedding and LM
    /// head are synthesized as all-zero on demand rather than materialized
    /// (2 × 2.5 GB for the 8B config, 2 × 4.74 GiB for Bonsai 2 27B), and the
    /// `f16` KV cache and the RoPE table start at
    /// `WEIGHTLESS_PREALLOC_CONTEXT` positions and grow towards
    /// `config.max_context_length` as the sequence does. A forward pass
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
        // Same host-KV default as a loaded model (lazy `f16`, limit =
        // effective context); infallible constructor, so the infallible
        // form — an absurd limit surfaces later as a typed growth error.
        let kv_cache = KvCache::new_lazy_f16(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            max_context,
            prealloc,
        );
        // M-08: a config-only model gets the same table a GGUF-loaded one
        // does. Infallible constructor, so an unusable declaration degrades to
        // an unscaled table with a `tracing::error!` rather than a panic.
        let rope = build_rope_table_or_unscaled(&config, prealloc);
        // No mapped weights: a private GPU slot namespace (see `ModelGpuSlots`).
        let output_weight = OutputWeight::zero_fp32(config.vocab_size, h);
        let mut blocks: Vec<TransformerBlock<'a>> = Vec::new();
        #[cfg_attr(
            not(any(
                all(feature = "metal", target_os = "macos"),
                all(
                    feature = "native-cuda",
                    any(target_os = "linux", target_os = "windows")
                )
            )),
            allow(unused_variables)
        )]
        let gpu_slots = ModelGpuSlots::attach(&mut blocks, &output_weight);
        Self {
            token_embd: EmbeddingTable::constant(0.0, config.vocab_size, h),
            shared_embd: std::sync::Arc::from(Vec::new()),
            blocks,
            output_norm: RmsNorm::new(vec![1.0; h], config.rms_norm_eps),
            output_weight,
            rope,
            kv_cache,
            dominant_quant_type: oxibonsai_core::GgufTensorType::Q1_0_g128,
            has_hadamard: false,
            scratch: ModelScratch::default(),
            max_context,
            host_kv_written: 0,
            gpu_path_active: std::sync::atomic::AtomicBool::new(false),
            prefill_chunk_tokens: crate::chunked_prefill::DEFAULT_PREFILL_CHUNK_TOKENS,
            lm_head_kernel: lm_head::weightless_lm_head_kernel(),
            force_cpu_decode_after: force_cpu_decode_after_from_env(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu_weight_cache: std::sync::Mutex::new(None),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            metal_q1_slots: gpu_slots.metal_q1_slots,
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_ternary_qkv_cache: std::sync::Mutex::new(None),
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            cuda_model_epoch: gpu_slots.cuda_model_epoch,
            config,
        }
    }
}

/// Effective context length for a model (sec-11).
///
/// `min(requested, config.max_context_length, cap)`, with a zero or missing
/// value treated as "no constraint from that source" and a floor of 1 so the
/// caches are never degenerate.
pub(super) fn effective_context(
    config: &Qwen3Config,
    requested: Option<usize>,
    cap: Option<usize>,
) -> usize {
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

/// RoPE rows (and the config-only models' first KV allocation) for a model
/// whose effective context is `max_context` (sec-11).
///
/// `growth_window` is `None` for a GGUF-loaded model: its RoPE table covers
/// the whole effective context from load, so every `&self` GPU entry point
/// (which cannot grow anything) can serve any position the model accepts.
/// The config-only constructors pass a window and start at
/// `min(max_context, window)` instead, growing on demand
/// ([`BonsaiModel::ensure_context_capacity`]), so a mock built from a
/// production config costs kilobytes rather than gigabytes.
pub(super) fn prealloc_context(max_context: usize, growth_window: Option<usize>) -> usize {
    match growth_window {
        Some(window) => max_context.clamp(1, window.max(1)),
        None => max_context.max(1),
    }
}

/// Read the `OXIBONSAI_FORCE_CPU_DECODE_AFTER` debug/test seam once.
pub(super) fn force_cpu_decode_after_from_env() -> Option<usize> {
    parse_force_cpu_after(std::env::var("OXIBONSAI_FORCE_CPU_DECODE_AFTER").ok())
}

/// Parse the `OXIBONSAI_FORCE_CPU_DECODE_AFTER` value; anything unparsable
/// means "never force the CPU", exactly as `InferenceEngine::generate` treats
/// the same variable.
pub(super) fn parse_force_cpu_after(raw: Option<String>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse().ok())
}
