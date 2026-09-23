//! [`HybridModel`] — construction, layer split and reset for a `qwen35`
//! (PrismML Bonsai 2) stack (design §3.1 / §3.10).
//!
//! # What is here
//!
//! B2-10 landed the **skeleton**: every tensor bound and shape-checked,
//! both caches allocated, the v-head map, the Hadamard hook and the layer
//! split. B2-11 landed the **forward**: [`HybridModel::forward`],
//! [`HybridModel::forward_prefill`] and [`HybridModel::forward_with_dump`],
//! all three of which are one call into
//! [`crate::hybrid::forward::run_chunk`] parameterised by batch, plus the
//! RoPE table and the chunk-wide activation scratch they run in.
//!
//! # Memory: `max_seq_len` is a parameter, never the model's context length
//!
//! The 27B declares `context_length = 262144`. Allocating the KV cache for
//! that would be 16 slots × 4 kv heads × 256 dims × 2 (K+V) × 262144 × 4 B =
//! **34 GB** — an OOM kill on a 24 GB machine before the first token. So
//! [`HybridModel::from_gguf`] takes `max_seq_len` explicitly and
//! [`DEFAULT_MAX_SEQ_LEN`] is 8192, not the model maximum; the context guard
//! that turns a RAM budget into a ceiling is B2-12's.
//!
//! The host cache is the **f16**, layer-sparse `KvCache::try_new_sparse` of
//! design §3.7 — 16 slots × 4 kv heads × 256 dims × 2 (K+V) × 2 B =
//! 64 KiB/token for the 27B, and the same `f16` KV the PrismML fork's own
//! goldens were produced with. [`KvPrecision::F32`] doubles that and exists
//! so a differential test can tell `f16` rounding apart from an arithmetic
//! bug.

use std::sync::Arc;

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::hadamard_config::HadamardConfig;
use oxibonsai_kernels::KernelDispatcher;

use crate::error::{ModelError, ModelResult};
use crate::hybrid::block::{
    FullAttnBlock, FullScratch, HybridBlock, LinearAttnBlock, LinearScratch,
};
use crate::hybrid::forward::{
    run_chunk, ForwardCtx, HybridScratch, LayerDump, RopeTables, DEFAULT_PREFILL_CHUNK,
};
use crate::hybrid::hadamard::{HadamardHook, HadamardScratch};
use crate::hybrid::recurrent_cache::RecurrentCache;
use crate::hybrid::vhead_map::VHeadMap;
use crate::hybrid::weights::{
    bind_conv1d, bind_embedding, bind_gate_projection, bind_gdn_gates, bind_linear, bind_lm_head,
    block_tensor, load_norm, names, resolve_id42, HybridEmbedding,
};
use crate::kv_cache::KvCache;
use crate::layers::linear::LinearLayer;
use crate::layers::rms_norm::RmsNorm;
use crate::model_registry::ModelVariant;

/// The KV window a hybrid model is built with unless the caller says
/// otherwise — the shipped default of design §5.6, **not** the model's
/// 262 144-token maximum (see the module docs).
pub const DEFAULT_MAX_SEQ_LEN: usize = 8192;

/// Which layers are full attention, which are recurrent, and each layer's
/// cache slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSplit {
    /// `block_count`.
    n_layers: usize,
    /// Indices of the full-attention layers, ascending (27B: `{3, 7, …,
    /// 63}`).
    full_layers: Vec<usize>,
    /// Indices of the Gated-DeltaNet layers, ascending.
    linear_layers: Vec<usize>,
    /// `kv_slot_of[i]` for a full layer, `None` for a linear one.
    kv_slot_of: Vec<Option<usize>>,
    /// `rec_slot_of[i]` for a linear layer, `None` for a full one.
    rec_slot_of: Vec<Option<usize>>,
}

impl LayerSplit {
    /// Derive the split from a config.
    ///
    /// Slots are assigned in **layer order**: the first full layer gets
    /// `kv_slot` 0, the second 1, and so on — so a slot index never depends
    /// on iteration order elsewhere.
    #[must_use]
    pub fn new(config: &HybridConfig) -> Self {
        let n_layers = config.base.num_layers;
        let mut full_layers = Vec::new();
        let mut linear_layers = Vec::new();
        let mut kv_slot_of = vec![None; n_layers];
        let mut rec_slot_of = vec![None; n_layers];
        for layer in 0..n_layers {
            if config.is_full_attention(layer) {
                kv_slot_of[layer] = Some(full_layers.len());
                full_layers.push(layer);
            } else {
                rec_slot_of[layer] = Some(linear_layers.len());
                linear_layers.push(layer);
            }
        }
        Self {
            n_layers,
            full_layers,
            linear_layers,
            kv_slot_of,
            rec_slot_of,
        }
    }

    /// Total layers.
    #[inline]
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Full-attention layer indices, ascending.
    #[inline]
    #[must_use]
    pub fn full_layers(&self) -> &[usize] {
        &self.full_layers
    }

    /// Gated-DeltaNet layer indices, ascending.
    #[inline]
    #[must_use]
    pub fn linear_layers(&self) -> &[usize] {
        &self.linear_layers
    }

    /// KV-cache slot of layer `layer`, or `None` when it is recurrent.
    #[inline]
    #[must_use]
    pub fn kv_slot(&self, layer: usize) -> Option<usize> {
        self.kv_slot_of.get(layer).copied().flatten()
    }

    /// Recurrent-cache slot of layer `layer`, or `None` when it is full
    /// attention.
    #[inline]
    #[must_use]
    pub fn rec_slot(&self, layer: usize) -> Option<usize> {
        self.rec_slot_of.get(layer).copied().flatten()
    }
}

/// A loaded Qwen3.5 / Bonsai 2 hybrid model.
///
/// Borrows every quantized weight from the mmap'd GGUF (`'a`); only the
/// norms, the small `ssm_*` vectors and the two caches are owned.
#[derive(Debug)]
pub struct HybridModel<'a> {
    config: HybridConfig,
    split: LayerSplit,
    blocks: Vec<HybridBlock<'a>>,
    embedding: HybridEmbedding<'a>,
    output_norm: RmsNorm,
    lm_head: LinearLayer<'a>,
    hadamard: Option<HadamardHook>,
    vhead_map: VHeadMap,
    kv_cache: KvCache,
    recurrent: RecurrentCache,
    rope: RopeTables,
    scratch: HybridScratch,
    prefill_chunk: usize,
    max_seq_len: usize,
    quant_type: GgufTensorType,
    variant: Option<ModelVariant>,
}

/// Element type of a hybrid model's KV cache.
///
/// Design §3.7 ships `f16` (64 KiB/token for the 27B, and what the PrismML
/// fork's own goldens were generated with — `llama-cli` defaults to an
/// `f16` KV with no `-ctk`/`-ctv`). `F32` doubles the footprint and exists
/// for differential testing against an exact reference, where the `f16`
/// rounding of stored keys and values would otherwise be indistinguishable
/// from an arithmetic bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvPrecision {
    /// Half precision, layer-sparse (design §3.7) — the shipped default.
    #[default]
    F16,
    /// Single precision, dense-backed: 128 KiB/token for the 27B.
    F32,
}

impl<'a> HybridModel<'a> {
    /// Parse a `qwen35` configuration from a GGUF's metadata.
    ///
    /// # Errors
    ///
    /// [`ModelError::Core`] with `UnsupportedArchitecture` for anything but
    /// `qwen35`, or `MissingConfigKey` for a missing hyper-parameter.
    pub fn config_from_gguf(gguf: &GgufFile<'_>) -> ModelResult<HybridConfig> {
        HybridConfig::from_metadata(&gguf.metadata).map_err(ModelError::Core)
    }

    /// Load a hybrid model from an already-parsed GGUF.
    ///
    /// `max_seq_len` sizes the KV cache and is a deliberate parameter — see
    /// the module docs on why it must not default to the model's declared
    /// context length. [`DEFAULT_MAX_SEQ_LEN`] is the shipped default.
    ///
    /// Validation performed here, all fatal:
    ///
    /// * the architecture and every `qwen35.*` hyper-parameter
    ///   ([`HybridConfig::from_metadata`], which also runs its own structural
    ///   checks);
    /// * the `prism.hadamard.*` contract, if present — including that every
    ///   folded/inverse name really is a tensor in this file, and that every
    ///   rotated width has a sign vector of exactly that length (M-15);
    /// * `gdn_v_grouped = false` with more v-heads than k-heads, which is
    ///   refused outright (design §3.3);
    /// * every tensor's presence and shape, per layer kind (design §3.8);
    /// * `ssm_a <= 0` on every linear layer, at load (B2-05);
    /// * the presence of an explicit `output.weight` (a tied head is
    ///   refused, design §3.5).
    ///
    /// # Errors
    ///
    /// Any of the above, as the named [`ModelError`] variant.
    pub fn from_gguf(gguf: &'a GgufFile<'a>, max_seq_len: usize) -> ModelResult<Self> {
        let config = Self::config_from_gguf(gguf)?;
        let kernel = Arc::new(KernelDispatcher::auto_detect());
        Self::from_gguf_with(gguf, config, max_seq_len, &kernel)
    }

    /// [`HybridModel::from_gguf`] with an explicit config and kernel
    /// dispatcher — what a test or a pinned-tier caller uses.
    ///
    /// # Errors
    ///
    /// As [`HybridModel::from_gguf`].
    pub fn from_gguf_with(
        gguf: &'a GgufFile<'a>,
        config: HybridConfig,
        max_seq_len: usize,
        kernel: &Arc<KernelDispatcher>,
    ) -> ModelResult<Self> {
        Self::from_gguf_with_precision(gguf, config, max_seq_len, kernel, KvPrecision::default())
    }

    /// [`HybridModel::from_gguf_with`] with an explicit KV element type.
    ///
    /// # Errors
    ///
    /// As [`HybridModel::from_gguf`].
    pub fn from_gguf_with_precision(
        gguf: &'a GgufFile<'a>,
        config: HybridConfig,
        max_seq_len: usize,
        kernel: &Arc<KernelDispatcher>,
        kv_precision: KvPrecision,
    ) -> ModelResult<Self> {
        if max_seq_len == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: "max_seq_len".to_string(),
                expected: "> 0".to_string(),
                actual: "0".to_string(),
            });
        }
        config.validate().map_err(ModelError::Core)?;

        // ── Hadamard contract ───────────────────────────────────────────
        let hadamard_config =
            HadamardConfig::from_metadata(&gguf.metadata).map_err(ModelError::Core)?;
        let hadamard = match hadamard_config {
            Some(spec) => {
                spec.validate_against_tensors(&gguf.tensors)
                    .map_err(ModelError::Core)?;
                let hook = HadamardHook::new(Arc::new(spec));
                hook.validate_widths(&config)?;
                Some(hook)
            }
            // A gen-1 `qwen35` file (no `prism.hadamard.version`) is not
            // folded at all: no rotation, and nothing constrains the
            // v-head column order.
            None => None,
        };
        let gdn_v_grouped = hadamard
            .as_ref()
            .is_none_or(|hook| hook.config().gdn_v_grouped);
        let vhead_map = VHeadMap::for_fold(config.n_k_heads(), config.n_v_heads(), gdn_v_grouped)?;

        // ── Globals ─────────────────────────────────────────────────────
        let resolved_42 = resolve_id42(gguf)?;
        let hidden = config.base.hidden_size;
        let vocab = config.base.vocab_size;
        let embedding = bind_embedding(gguf, hidden, vocab, resolved_42)?;
        let output_norm = load_norm(gguf, names::OUTPUT_NORM, hidden, config.base.rms_norm_eps)?;
        let lm_head = bind_lm_head(gguf, hidden, vocab, resolved_42, kernel)?;

        // ── Layers ──────────────────────────────────────────────────────
        let split = LayerSplit::new(&config);
        let mut blocks = Vec::with_capacity(split.n_layers());
        for layer in 0..split.n_layers() {
            let block = match (split.kv_slot(layer), split.rec_slot(layer)) {
                (Some(kv_slot), _) => HybridBlock::Full(bind_full_layer(
                    gguf,
                    &config,
                    layer,
                    kv_slot,
                    resolved_42,
                    kernel,
                )?),
                (None, Some(rec_slot)) => HybridBlock::Linear(bind_linear_layer(
                    gguf,
                    &config,
                    layer,
                    rec_slot,
                    resolved_42,
                    kernel,
                    &vhead_map,
                )?),
                // `LayerSplit::new` assigns exactly one slot to every layer.
                (None, None) => {
                    return Err(ModelError::ShapeInvariant {
                        tensor: format!("layer {layer}"),
                        expected: "a KV slot or a recurrent slot".to_string(),
                        actual: "neither".to_string(),
                    })
                }
            };
            blocks.push(block);
        }

        // ── Caches ──────────────────────────────────────────────────────
        // Indexed by `kv_slot` (0..16 for the 27B), never by `layer_idx`,
        // and allocated through the *fallible* constructors: a 27B at the
        // model's declared 262 144-token context is 16 GiB of KV even at
        // `f16`, which must surface as an error rather than an abort.
        let kv_cache = match kv_precision {
            KvPrecision::F16 => KvCache::try_new_sparse(
                split.full_layers().len(),
                config.base.num_kv_heads,
                config.base.head_dim,
                max_seq_len,
            )?,
            KvPrecision::F32 => KvCache::try_new(
                split.full_layers().len(),
                config.base.num_kv_heads,
                config.base.head_dim,
                max_seq_len,
            )?,
        };
        let recurrent = RecurrentCache::new(&config)?;
        let rope = RopeTables::new(
            config.rope_dimension_count,
            max_seq_len,
            config.base.rope_freq_base,
        )?;
        // One token's worth to start with: `run_chunk` grows it in place the
        // first time a prefill asks for more.
        let scratch = HybridScratch::new(&config, 1);

        let quant_type = crate::hybrid::weights::apply_resolved_type(
            gguf.tensors
                .require(names::OUTPUT)
                .map_err(ModelError::Core)?
                .tensor_type,
            resolved_42,
        );
        let variant = ModelVariant::detect_qwen35_27b(
            &config.base.architecture,
            config.base.num_layers as u64,
            quant_type,
            hadamard.is_some(),
        );

        Ok(Self {
            config,
            split,
            blocks,
            embedding,
            output_norm,
            lm_head,
            hadamard,
            vhead_map,
            kv_cache,
            recurrent,
            rope,
            scratch,
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
            max_seq_len,
            quant_type,
            variant,
        })
    }

    /// The parsed `qwen35` configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// The layer split and its cache slots.
    #[inline]
    #[must_use]
    pub fn split(&self) -> &LayerSplit {
        &self.split
    }

    /// Every decoder layer, in stack order.
    #[inline]
    #[must_use]
    pub fn blocks(&self) -> &[HybridBlock<'a>] {
        &self.blocks
    }

    /// One decoder layer.
    #[inline]
    #[must_use]
    pub fn block(&self, layer: usize) -> Option<&HybridBlock<'a>> {
        self.blocks.get(layer)
    }

    /// The token-embedding table (rows are decoded on lookup and are still
    /// in the rotated basis for a folded model).
    #[inline]
    #[must_use]
    pub fn embedding(&self) -> &HybridEmbedding<'a> {
        &self.embedding
    }

    /// The final RMSNorm before the LM head.
    #[inline]
    #[must_use]
    pub fn output_norm(&self) -> &RmsNorm {
        &self.output_norm
    }

    /// The LM head (`output.weight`, folded).
    #[inline]
    #[must_use]
    pub fn lm_head(&self) -> &LinearLayer<'a> {
        &self.lm_head
    }

    /// The Hadamard hook, or `None` for an unfolded (gen-1) file.
    #[inline]
    #[must_use]
    pub fn hadamard(&self) -> Option<&HadamardHook> {
        self.hadamard.as_ref()
    }

    /// The tiled ↔ grouped v-head index map.
    #[inline]
    #[must_use]
    pub fn vhead_map(&self) -> &VHeadMap {
        &self.vhead_map
    }

    /// KV cache for the full-attention layers, indexed by `kv_slot`.
    #[inline]
    #[must_use]
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    /// Mutable KV cache.
    #[inline]
    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }

    /// Recurrent state for the Gated-DeltaNet layers, indexed by `rec_slot`.
    #[inline]
    #[must_use]
    pub fn recurrent(&self) -> &RecurrentCache {
        &self.recurrent
    }

    /// Mutable recurrent state.
    #[inline]
    pub fn recurrent_mut(&mut self) -> &mut RecurrentCache {
        &mut self.recurrent
    }

    /// Take ownership of the recurrent state, e.g. to hand it to the
    /// runtime's reset seam (`InferenceEngine::set_recurrent_state`, RT-28).
    ///
    /// Leaves a freshly zeroed cache of the same geometry behind, so the
    /// model stays usable and no caller can end up with a model whose state
    /// silently vanished.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] if the replacement cannot be allocated
    /// (a degenerate config, which [`HybridModel::from_gguf_with`] already
    /// rejects).
    pub fn take_recurrent(&mut self) -> ModelResult<RecurrentCache> {
        let replacement = RecurrentCache::new(&self.config)?;
        Ok(std::mem::replace(&mut self.recurrent, replacement))
    }

    /// KV window this model was built with.
    #[inline]
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// The resolved on-disk quantization of the weight matrices.
    #[inline]
    #[must_use]
    pub fn quant_type(&self) -> GgufTensorType {
        self.quant_type
    }

    /// The detected model variant, when this file is one of the known 27B
    /// builds.
    #[inline]
    #[must_use]
    pub fn variant(&self) -> Option<ModelVariant> {
        self.variant
    }

    /// Folded tensor count from the Hadamard contract (27B: 401); `0` for an
    /// unfolded file.
    #[inline]
    #[must_use]
    pub fn folded_count(&self) -> usize {
        self.hadamard.as_ref().map_or(0, HadamardHook::folded_count)
    }

    /// Allocate the rotation scratch a forward needs
    /// ([`HadamardScratch`], design §3.4).
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for a degenerate width.
    pub fn hadamard_scratch(&self) -> ModelResult<HadamardScratch> {
        HadamardScratch::new(&self.config)
    }

    /// Tokens per prefill chunk (design SS3.10; `--prefill-chunk`).
    #[inline]
    #[must_use]
    pub fn prefill_chunk(&self) -> usize {
        self.prefill_chunk
    }

    /// Set the prefill chunk size; `0` is rejected.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for a zero chunk.
    pub fn set_prefill_chunk(&mut self, chunk: usize) -> ModelResult<()> {
        if chunk == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: "prefill chunk".to_string(),
                expected: "> 0".to_string(),
                actual: "0".to_string(),
            });
        }
        self.prefill_chunk = chunk;
        Ok(())
    }

    /// Whether the KV cache is the `f16`, layer-sparse one of design SS3.7.
    #[inline]
    #[must_use]
    pub fn kv_is_f16(&self) -> bool {
        self.kv_cache.is_sparse()
    }

    /// Borrow every part one forward needs, field by field, so the driver
    /// can hold disjoint `&`/`&mut` borrows of one model.
    fn ctx(&mut self) -> ForwardCtx<'_, 'a> {
        ForwardCtx {
            config: &self.config,
            blocks: &self.blocks,
            embedding: &self.embedding,
            output_norm: &self.output_norm,
            lm_head: &self.lm_head,
            hadamard: self.hadamard.as_ref(),
            vhead_map: &self.vhead_map,
            rope: &self.rope,
            kv: &mut self.kv_cache,
            recurrent: &mut self.recurrent,
            scratch: &mut self.scratch,
        }
    }

    /// Single-token decode at absolute position `pos`, writing
    /// `[vocab_size]` logits into `logits` (design SS3.10).
    ///
    /// This is [`HybridModel::forward_prefill`] with a one-token chunk --
    /// literally the same body -- so a decode step and the last token of a
    /// prefill chunk cannot diverge.
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] past the KV window,
    /// [`ModelError::ShapeMismatch`] for a short `logits`, and anything the
    /// blocks or kernels return.
    pub fn forward(&mut self, token: u32, pos: usize, logits: &mut [f32]) -> ModelResult<()> {
        let mut ctx = self.ctx();
        run_chunk(&mut ctx, &[token], pos, Some(logits), None)?;
        self.kv_cache.set_seq_len(pos + 1);
        Ok(())
    }

    /// Single-token decode that also allocates its logit row.
    ///
    /// # Errors
    ///
    /// As [`HybridModel::forward`].
    pub fn forward_alloc(&mut self, token: u32, pos: usize) -> ModelResult<Vec<f32>> {
        let mut logits = vec![0.0f32; self.config.base.vocab_size];
        self.forward(token, pos, &mut logits)?;
        Ok(logits)
    }

    /// Chunked prefill of `tokens` starting at `start_pos`, writing the
    /// **last** token's `[vocab_size]` logits into `last_logits`.
    ///
    /// Long prompts are split into [`HybridModel::prefill_chunk`]-token
    /// chunks; each chunk advances the recurrent state exactly once per
    /// token, in order, and stores every key/value at its absolute
    /// position, so the result is the same as feeding the tokens one at a
    /// time (design SS8.2 G5).
    ///
    /// # Errors
    ///
    /// As [`HybridModel::forward`].
    pub fn forward_prefill(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        last_logits: &mut [f32],
    ) -> ModelResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let chunk = self.prefill_chunk.max(1);
        let total = tokens.len();
        let mut offset = 0usize;
        while offset < total {
            let end = (offset + chunk).min(total);
            let is_last = end == total;
            let slice = tokens
                .get(offset..end)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: "prefill chunk".to_string(),
                    expected: format!("{offset}..{end} within {total} tokens"),
                    actual: "out of range".to_string(),
                })?;
            let mut ctx = self.ctx();
            if is_last {
                run_chunk(&mut ctx, slice, start_pos + offset, Some(last_logits), None)?;
            } else {
                run_chunk(&mut ctx, slice, start_pos + offset, None, None)?;
            }
            offset = end;
        }
        self.kv_cache.set_seq_len(start_pos + total);
        Ok(())
    }

    /// [`HybridModel::forward_prefill`] over a single chunk, recording every
    /// block's output (design SS8.2 G11: the CPU reference dump the later
    /// Metal parity gate diffs against).
    ///
    /// Deliberately **not** chunked: a dump is a diagnostic over a known,
    /// bounded token list, and stitching per-chunk dumps together would hide
    /// exactly the chunk-boundary behaviour it exists to expose.
    ///
    /// # Errors
    ///
    /// As [`HybridModel::forward`].
    pub fn forward_with_dump(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        last_logits: Option<&mut [f32]>,
    ) -> ModelResult<LayerDump> {
        let mut dump = LayerDump::default();
        let end = start_pos + tokens.len();
        {
            let mut ctx = self.ctx();
            run_chunk(&mut ctx, tokens, start_pos, last_logits, Some(&mut dump))?;
        }
        self.kv_cache.set_seq_len(end);
        Ok(dump)
    }

    /// Bytes held by the activation scratch at its current chunk capacity.
    #[inline]
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.memory_bytes()
    }

    /// Clear both caches: the KV cursor **and** the recurrent state (RT-28).
    ///
    /// The recurrent half is the part with no positional masking — a stale
    /// `S` contaminates the next request's very first token rather than
    /// being overwritten — so a reset that clears only the KV cache is not a
    /// reset at all for a hybrid model.
    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.recurrent.reset();
    }

    /// One-line summary for `oxibonsai info` (integration gate G2).
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "{} | {} layers ({} full / {} linear) | hidden {} | {} v-heads over {} k-heads | \
             {} folded tensors | quant {}",
            self.config.base.architecture,
            self.split.n_layers(),
            self.split.full_layers().len(),
            self.split.linear_layers().len(),
            self.config.base.hidden_size,
            self.config.n_v_heads(),
            self.config.n_k_heads(),
            self.folded_count(),
            self.quant_type.name(),
        )
    }
}

/// Bind one full-attention layer (design §3.8).
fn bind_full_layer<'a>(
    gguf: &'a GgufFile<'a>,
    config: &HybridConfig,
    layer: usize,
    kv_slot: usize,
    resolved_42: Option<GgufTensorType>,
    kernel: &Arc<KernelDispatcher>,
) -> ModelResult<FullAttnBlock<'a>> {
    let hidden = config.base.hidden_size;
    let heads_width = config.base.num_attention_heads * config.base.head_dim;
    let kv_width = config.base.num_kv_heads * config.base.head_dim;
    let inter = config.base.intermediate_size;
    let eps = config.base.rms_norm_eps;
    let blk = |suffix: &str| block_tensor(layer, suffix);

    Ok(FullAttnBlock {
        layer_idx: layer,
        attn_norm: load_norm(gguf, &blk(names::ATTN_NORM), hidden, eps)?,
        post_attn_norm: load_norm(gguf, &blk(names::POST_ATTENTION_NORM), hidden, eps)?,
        // `attn_q` is [hidden, 2 * heads_width]: per head `[q | gate]`.
        attn_q: bind_linear(
            gguf,
            &blk(names::ATTN_Q),
            hidden,
            heads_width * 2,
            resolved_42,
            kernel,
        )?,
        attn_k: bind_linear(
            gguf,
            &blk(names::ATTN_K),
            hidden,
            kv_width,
            resolved_42,
            kernel,
        )?,
        attn_v: bind_linear(
            gguf,
            &blk(names::ATTN_V),
            hidden,
            kv_width,
            resolved_42,
            kernel,
        )?,
        attn_output: bind_linear(
            gguf,
            &blk(names::ATTN_OUTPUT),
            heads_width,
            hidden,
            resolved_42,
            kernel,
        )?,
        attn_q_norm: load_norm(gguf, &blk(names::ATTN_Q_NORM), config.base.head_dim, eps)?,
        attn_k_norm: load_norm(gguf, &blk(names::ATTN_K_NORM), config.base.head_dim, eps)?,
        ffn_gate: bind_linear(
            gguf,
            &blk(names::FFN_GATE),
            hidden,
            inter,
            resolved_42,
            kernel,
        )?,
        ffn_up: bind_linear(
            gguf,
            &blk(names::FFN_UP),
            hidden,
            inter,
            resolved_42,
            kernel,
        )?,
        ffn_down: bind_linear(
            gguf,
            &blk(names::FFN_DOWN),
            inter,
            hidden,
            resolved_42,
            kernel,
        )?,
        kv_slot,
        scratch: std::sync::Mutex::new(FullScratch::new(config)),
    })
}

/// Bind one Gated-DeltaNet layer (design §3.8).
fn bind_linear_layer<'a>(
    gguf: &'a GgufFile<'a>,
    config: &HybridConfig,
    layer: usize,
    rec_slot: usize,
    resolved_42: Option<GgufTensorType>,
    kernel: &Arc<KernelDispatcher>,
    vhead_map: &VHeadMap,
) -> ModelResult<LinearAttnBlock<'a>> {
    let hidden = config.base.hidden_size;
    let inner = config.ssm_inner_size;
    let inter = config.base.intermediate_size;
    let heads = config.n_v_heads();
    let eps = config.base.rms_norm_eps;
    let blk = |suffix: &str| block_tensor(layer, suffix);

    Ok(LinearAttnBlock {
        layer_idx: layer,
        attn_norm: load_norm(gguf, &blk(names::ATTN_NORM), hidden, eps)?,
        post_attn_norm: load_norm(gguf, &blk(names::POST_ATTENTION_NORM), hidden, eps)?,
        attn_qkv: bind_linear(
            gguf,
            &blk(names::ATTN_QKV),
            hidden,
            config.conv_dim(),
            resolved_42,
            kernel,
        )?,
        attn_gate: bind_linear(
            gguf,
            &blk(names::ATTN_GATE),
            hidden,
            inner,
            resolved_42,
            kernel,
        )?,
        ssm_alpha: bind_gate_projection(gguf, layer, names::SSM_ALPHA, hidden, heads, resolved_42)?,
        ssm_beta: bind_gate_projection(gguf, layer, names::SSM_BETA, hidden, heads, resolved_42)?,
        ssm_conv1d: bind_conv1d(gguf, layer, config)?,
        // `ssm_a` and `ssm_dt.bias` bound by name (never positionally,
        // B2-05 / gatekeeper REQUIRED #8), `validate_a_neg`-checked at load,
        // and re-indexed tiled -> grouped through `vhead_map` so they land
        // in the same v-head order as every other v-indexed quantity this
        // block produces (gatekeeper blocking finding on this package).
        gates: bind_gdn_gates(gguf, layer, heads, vhead_map)?,
        ssm_norm: load_norm(gguf, &blk(names::SSM_NORM), config.head_v_dim(), eps)?,
        ssm_out: bind_linear(
            gguf,
            &blk(names::SSM_OUT),
            inner,
            hidden,
            resolved_42,
            kernel,
        )?,
        ffn_gate: bind_linear(
            gguf,
            &blk(names::FFN_GATE),
            hidden,
            inter,
            resolved_42,
            kernel,
        )?,
        ffn_up: bind_linear(
            gguf,
            &blk(names::FFN_UP),
            hidden,
            inter,
            resolved_42,
            kernel,
        )?,
        ffn_down: bind_linear(
            gguf,
            &blk(names::FFN_DOWN),
            inter,
            hidden,
            resolved_42,
            kernel,
        )?,
        rec_slot,
        scratch: std::sync::Mutex::new(LinearScratch::new(config)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::tests_support::{
        bonsai2_config, synthetic_gguf, FixtureOptions, FixtureShape,
    };

    fn kernel() -> Arc<KernelDispatcher> {
        Arc::new(KernelDispatcher::with_tier(
            oxibonsai_kernels::KernelTier::Reference,
        ))
    }

    #[test]
    fn layer_split_is_the_27b_s_16_full_and_48_linear() {
        let split = LayerSplit::new(&bonsai2_config());
        assert_eq!(split.n_layers(), 64);
        assert_eq!(split.full_layers().len(), 16);
        assert_eq!(split.linear_layers().len(), 48);
        assert_eq!(split.full_layers()[0], 3);
        assert_eq!(split.full_layers()[15], 63);
        assert!(split.full_layers().iter().all(|i| (i + 1) % 4 == 0));

        // Slots are assigned in layer order and cover both ranges exactly
        // once.
        assert_eq!(split.kv_slot(3), Some(0));
        assert_eq!(split.kv_slot(7), Some(1));
        assert_eq!(split.kv_slot(63), Some(15));
        assert_eq!(split.kv_slot(0), None);
        assert_eq!(split.rec_slot(0), Some(0));
        assert_eq!(split.rec_slot(1), Some(1));
        assert_eq!(split.rec_slot(3), None);
        let mut kv: Vec<usize> = (0..64).filter_map(|i| split.kv_slot(i)).collect();
        kv.sort_unstable();
        assert_eq!(kv, (0..16).collect::<Vec<_>>());
        let mut rec: Vec<usize> = (0..64).filter_map(|i| split.rec_slot(i)).collect();
        rec.sort_unstable();
        assert_eq!(rec, (0..48).collect::<Vec<_>>());
    }

    #[test]
    fn loads_a_synthetic_hybrid_file_end_to_end() {
        let shape = FixtureShape::default();
        let bytes = synthetic_gguf(shape, FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let model = HybridModel::from_gguf(&gguf, 64).expect("model loads");

        assert_eq!(model.config().base.architecture, "qwen35");
        assert_eq!(model.split().n_layers(), shape.n_layers);
        assert_eq!(model.split().full_layers(), &[3, 7]);
        assert_eq!(model.split().linear_layers(), &[0, 1, 2, 4, 5, 6]);
        assert_eq!(model.blocks().len(), shape.n_layers);

        // Every block is the kind the split says, with the slot the split
        // says.
        for (layer, block) in model.blocks().iter().enumerate() {
            assert_eq!(block.layer_idx(), layer);
            assert_eq!(block.is_full(), shape.is_full(layer));
            assert_eq!(block.kv_slot(), model.split().kv_slot(layer));
            assert_eq!(block.rec_slot(), model.split().rec_slot(layer));
        }

        // Bindings: the widths of the two layer kinds.
        let full = model
            .block(3)
            .and_then(HybridBlock::as_full)
            .expect("layer 3 is full");
        assert_eq!(full.attn_q().out_features(), shape.heads_width() * 2);
        assert_eq!(
            full.attn_k().out_features(),
            shape.n_kv_heads * shape.head_dim
        );
        assert_eq!(full.attn_output().in_features(), shape.heads_width());
        assert_eq!(full.attn_q_norm().hidden_size(), shape.head_dim);

        let linear = model
            .block(0)
            .and_then(HybridBlock::as_linear)
            .expect("layer 0 is linear");
        assert_eq!(linear.attn_qkv().out_features(), shape.conv_dim());
        assert_eq!(linear.attn_gate().out_features(), shape.inner());
        assert_eq!(linear.ssm_out().in_features(), shape.inner());
        assert_eq!(linear.ssm_alpha().out_features(), shape.n_v_heads);
        assert_eq!(linear.ssm_beta().in_features(), shape.hidden);
        assert_eq!(
            linear.ssm_conv1d().len(),
            shape.conv_kernel * shape.conv_dim()
        );
        assert_eq!(linear.ssm_norm().hidden_size(), shape.state_size);
        assert!(linear.gates().a_neg().iter().all(|a| *a <= 0.0));

        // Caches: sized by slot count, not by layer count.
        assert_eq!(model.recurrent().n_layers(), 6);
        assert_eq!(model.max_seq_len(), 64);
        assert_eq!(model.kv_cache().seq_len(), 0);

        // The fold contract, and its exclusions.
        let hook = model.hadamard().expect("the fixture is folded");
        assert!(hook.is_folded("blk.0.attn_qkv.weight"));
        assert!(!hook.is_folded("blk.0.ssm_alpha.weight"));
        assert_eq!(model.folded_count(), 1 + 6 * 6 + 2 * 7);
        assert!(
            model.describe().contains("6 linear"),
            "{}",
            model.describe()
        );
    }

    #[test]
    fn an_unfolded_file_loads_without_a_hook() {
        let bytes = synthetic_gguf(
            FixtureShape::default(),
            FixtureOptions {
                hadamard: false,
                ..FixtureOptions::default()
            },
        );
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let model = HybridModel::from_gguf(&gguf, 32).expect("gen-1 file loads");
        assert!(model.hadamard().is_none());
        assert_eq!(model.folded_count(), 0);
        // The grouped v-head convention still applies: nothing was folded,
        // so nothing constrains the column order.
        assert_eq!(model.vhead_map().v_per_k(), 3);
    }

    #[test]
    fn refuses_an_ungrouped_fold() {
        let bytes = synthetic_gguf(
            FixtureShape::default(),
            FixtureOptions {
                gdn_v_grouped: false,
                ..FixtureOptions::default()
            },
        );
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let err = HybridModel::from_gguf(&gguf, 32)
            .expect_err("gdn_v_grouped=false with nv != nk must be refused");
        assert_eq!(err.error_code(), "UNGROUPED_FOLDED_GDN_OUTPUT");
    }

    #[test]
    fn reset_clears_both_caches() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");

        model.recurrent_mut().ssm_mut(0).expect("slot 0")[0] = 1.0;
        model.recurrent_mut().conv_mut(2).expect("slot 2")[1] = -1.0;
        model.recurrent_mut().advance(4);

        model.reset();
        assert_eq!(model.recurrent().token_count(), 0);
        assert!(model
            .recurrent()
            .ssm(0)
            .expect("slot 0")
            .iter()
            .all(|v| *v == 0.0));
        assert!(model
            .recurrent()
            .conv(2)
            .expect("slot 2")
            .iter()
            .all(|v| *v == 0.0));
        assert_eq!(model.kv_cache().seq_len(), 0);
    }

    #[test]
    fn take_recurrent_leaves_a_fresh_cache_behind() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf(&gguf, 32).expect("model loads");
        model.recurrent_mut().advance(3);

        let taken = model.take_recurrent().expect("replacement allocates");
        assert_eq!(taken.token_count(), 3);
        assert_eq!(model.recurrent().token_count(), 0);
        assert_eq!(model.recurrent().n_layers(), taken.n_layers());
    }

    #[test]
    fn rejects_a_zero_kv_window() {
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let config = HybridModel::config_from_gguf(&gguf).expect("config");
        let err = HybridModel::from_gguf_with(&gguf, config, 0, &kernel())
            .expect_err("a zero KV window must be refused");
        assert_eq!(err.error_code(), "SHAPE_INVARIANT");
    }

    #[test]
    fn refuses_a_dense_qwen3_file() {
        // The dense path's own architecture must not load here.
        let mut config = bonsai2_config();
        config.base.architecture = "qwen3".to_string();
        assert!(config.validate().is_ok(), "the geometry itself is fine");
        // `from_metadata` is the guard: a `qwen3` file has no `qwen35.*`
        // keys at all, so it cannot reach `from_gguf_with`.
        let bytes = synthetic_gguf(FixtureShape::default(), FixtureOptions::default());
        let mut gguf_bytes = bytes.clone();
        // Flip the architecture string in place (same length) so the file
        // still parses but no longer declares `qwen35`.
        if let Some(pos) = find_bytes(&gguf_bytes, b"qwen35") {
            gguf_bytes[pos..pos + 6].copy_from_slice(b"qwen3x");
        }
        let gguf = GgufFile::parse(&gguf_bytes).expect("fixture still parses");
        let err = HybridModel::config_from_gguf(&gguf)
            .expect_err("an architecture other than qwen35 must be refused");
        assert_eq!(err.error_code(), "CORE_ERROR");
    }

    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }
}

/// The package's acceptance gate against the **real** Bonsai 2 27B files
/// (design §7.3 gate G2).
///
/// Model weights are not in the repository, so each case skips when its file
/// is absent. Set `OXI_REQUIRE_MODEL_FILES=1` to turn a missing file into a
/// hard failure instead — a green run over an empty `models/` executes zero
/// assertions, and CI that really mounts the weights must be able to demand
/// that every case ran. (Same convention as
/// `model_registry::tests::detect_qwen35_27b_on_real_files`.)
#[cfg(test)]
mod real_model_tests {
    use super::*;
    use crate::hybrid::weights::{block_tensor, load_dense_f32, names};
    use oxibonsai_core::gguf::reader::mmap_gguf_file;

    /// Verified constants of both 27B language files
    /// (`gguf_headers_summary.txt`).
    const N_LAYERS: usize = 64;
    const N_FULL: usize = 16;
    const N_LINEAR: usize = 48;
    const HIDDEN: usize = 5120;
    const INTERMEDIATE: usize = 17_408;
    const HEADS_WIDTH: usize = 6144;
    const KV_WIDTH: usize = 1024;
    const CONV_DIM: usize = 10_240;
    const N_V_HEADS: usize = 48;
    const N_K_HEADS: usize = 16;
    const HEAD_DIM: usize = 256;
    const FOLDED_TENSORS: usize = 401;
    const VOCAB: usize = 248_320;
    /// 48 linear layers x 48 v-heads x 128 x 128 x 4 B + 48 x 10240 x 3 x 4 B.
    const RECURRENT_BYTES: usize = 150_994_944 + 5_898_240;

    fn require_real_files() -> bool {
        std::env::var("OXI_REQUIRE_MODEL_FILES")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    fn models_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
    }

    /// Load one real file and assert the whole skeleton.
    ///
    /// `max_seq_len` is deliberately tiny: the KV cache is allocated eagerly
    /// and the model's own `context_length` is 262 144, which at f32 would be
    /// 34 GB.
    fn check_real_file(filename: &str, expected_quant: GgufTensorType, expected: ModelVariant) {
        let path = models_dir().join(filename);
        let Ok(mmap) = mmap_gguf_file(&path) else {
            assert!(
                !require_real_files(),
                "OXI_REQUIRE_MODEL_FILES=1: {filename} must be present at {}",
                path.display()
            );
            eprintln!("skipping {filename}: not present at {}", path.display());
            return;
        };
        let gguf = GgufFile::parse(&mmap)
            .unwrap_or_else(|e| panic!("{filename}: real file must parse cleanly: {e}"));

        let model = HybridModel::from_gguf(&gguf, 64)
            .unwrap_or_else(|e| panic!("{filename}: real file must load: {e}"));

        // -- config ------------------------------------------------------
        let config = model.config();
        assert_eq!(config.base.architecture, "qwen35", "{filename}");
        assert_eq!(config.base.num_layers, N_LAYERS, "{filename}");
        assert_eq!(config.base.hidden_size, HIDDEN, "{filename}");
        assert_eq!(config.base.intermediate_size, INTERMEDIATE, "{filename}");
        assert_eq!(config.base.head_dim, HEAD_DIM, "{filename}");
        assert_eq!(config.base.vocab_size, VOCAB, "{filename}");
        assert_eq!(config.n_v_heads(), N_V_HEADS, "{filename}");
        assert_eq!(config.n_k_heads(), N_K_HEADS, "{filename}");
        assert_eq!(config.conv_dim(), CONV_DIM, "{filename}");
        assert_eq!(config.ssm_inner_size, HEADS_WIDTH, "{filename}");

        // -- layer split (G2, half 1) ------------------------------------
        let split = model.split();
        assert_eq!(split.n_layers(), N_LAYERS, "{filename}");
        assert_eq!(split.full_layers().len(), N_FULL, "{filename}");
        assert_eq!(split.linear_layers().len(), N_LINEAR, "{filename}");
        let expected_full: Vec<usize> = (0..N_LAYERS).filter(|i| (i + 1) % 4 == 0).collect();
        assert_eq!(
            split.full_layers(),
            expected_full.as_slice(),
            "{filename}: full layers must be every 4th, ending at 63"
        );
        assert_eq!(split.kv_slot(3), Some(0), "{filename}");
        assert_eq!(split.kv_slot(63), Some(15), "{filename}");
        assert_eq!(split.rec_slot(0), Some(0), "{filename}");
        assert_eq!(split.rec_slot(62), Some(47), "{filename}");
        assert_eq!(model.blocks().len(), N_LAYERS, "{filename}");
        for (layer, block) in model.blocks().iter().enumerate() {
            assert_eq!(block.layer_idx(), layer, "{filename}");
            assert_eq!(
                block.is_full(),
                (layer + 1) % 4 == 0,
                "{filename} layer {layer}"
            );
            assert_eq!(
                block.kv_slot(),
                split.kv_slot(layer),
                "{filename} layer {layer}"
            );
            assert_eq!(
                block.rec_slot(),
                split.rec_slot(layer),
                "{filename} layer {layer}"
            );
        }

        // -- tensor bindings (G2, half 2) --------------------------------
        for layer in split.full_layers().iter().copied() {
            let full = model
                .block(layer)
                .and_then(HybridBlock::as_full)
                .unwrap_or_else(|| panic!("{filename}: layer {layer} must be full attention"));
            // attn_q is [5120, 12288] = 24 heads x 256 x 2 (q|gate).
            assert_eq!(full.attn_q().in_features(), HIDDEN);
            assert_eq!(full.attn_q().out_features(), HEADS_WIDTH * 2);
            assert_eq!(full.attn_k().out_features(), KV_WIDTH);
            assert_eq!(full.attn_v().out_features(), KV_WIDTH);
            assert_eq!(full.attn_output().in_features(), HEADS_WIDTH);
            assert_eq!(full.attn_output().out_features(), HIDDEN);
            assert_eq!(full.attn_q_norm().hidden_size(), HEAD_DIM);
            assert_eq!(full.attn_k_norm().hidden_size(), HEAD_DIM);
            assert_eq!(full.attn_norm().hidden_size(), HIDDEN);
            assert_eq!(full.post_attn_norm().hidden_size(), HIDDEN);
            assert_eq!(full.ffn_gate().out_features(), INTERMEDIATE);
            assert_eq!(full.ffn_down().in_features(), INTERMEDIATE);
        }
        for layer in split.linear_layers().iter().copied() {
            let linear = model
                .block(layer)
                .and_then(HybridBlock::as_linear)
                .unwrap_or_else(|| panic!("{filename}: layer {layer} must be linear attention"));
            // attn_qkv is [5120, 10240] = q 2048 | k 2048 | v 6144.
            assert_eq!(linear.attn_qkv().in_features(), HIDDEN);
            assert_eq!(linear.attn_qkv().out_features(), CONV_DIM);
            assert_eq!(linear.attn_gate().out_features(), HEADS_WIDTH);
            assert_eq!(linear.ssm_out().in_features(), HEADS_WIDTH);
            assert_eq!(linear.ssm_out().out_features(), HIDDEN);
            assert_eq!(linear.ssm_alpha().in_features(), HIDDEN);
            assert_eq!(linear.ssm_alpha().out_features(), N_V_HEADS);
            assert_eq!(linear.ssm_beta().out_features(), N_V_HEADS);
            assert_eq!(linear.ssm_conv1d().len(), 4 * CONV_DIM);
            assert_eq!(linear.ssm_norm().hidden_size(), 128);
            assert_eq!(linear.gates().a_neg().len(), N_V_HEADS);
            assert_eq!(linear.gates().dt_bias().len(), N_V_HEADS);
            // B2-05: every ssm_a was validated at load, not mid-decode.
            assert!(
                linear.gates().a_neg().iter().all(|a| *a <= 0.0),
                "{filename} layer {layer}: ssm_a must be A = -exp(A_log)"
            );
            // Gatekeeper's blocking finding, pinned on the REAL file (this is
            // the exact layer/tensor the verifier's mutation probe measured
            // 45/48 wrong heads on): `gates().a_neg()`/`dt_bias()` must be in
            // GROUPED order, i.e. index `m` must equal the raw GGUF row
            // `vhead_map.tiled(m)`, not the raw row `m` itself.
            let raw_a = load_dense_f32(&gguf, &block_tensor(layer, names::SSM_A), N_V_HEADS)
                .unwrap_or_else(|e| panic!("{filename} layer {layer}: raw ssm_a: {e}"));
            let raw_dt = load_dense_f32(&gguf, &block_tensor(layer, names::SSM_DT_BIAS), N_V_HEADS)
                .unwrap_or_else(|e| panic!("{filename} layer {layer}: raw ssm_dt.bias: {e}"));
            for m in 0..N_V_HEADS {
                let tiled = model.vhead_map().tiled(m);
                assert_eq!(
                    linear.gates().a_neg()[m],
                    raw_a[tiled],
                    "{filename} layer {layer}: grouped v-head {m} (raw GGUF row {tiled}): a_neg"
                );
                assert_eq!(
                    linear.gates().dt_bias()[m],
                    raw_dt[tiled],
                    "{filename} layer {layer}: grouped v-head {m} (raw GGUF row {tiled}): dt_bias"
                );
            }
        }

        // -- globals -----------------------------------------------------
        assert_eq!(model.output_norm().hidden_size(), HIDDEN);
        assert_eq!(model.lm_head().in_features(), HIDDEN);
        assert_eq!(model.lm_head().out_features(), VOCAB);
        let mut row = vec![0.0f32; HIDDEN];
        model
            .embedding()
            .row(0, HIDDEN, &mut row)
            .unwrap_or_else(|e| panic!("{filename}: embedding row 0: {e}"));
        assert!(
            row.iter().any(|v| *v != 0.0),
            "{filename}: embedding row 0 is all zero"
        );

        // -- Hadamard contract -------------------------------------------
        let hook = model
            .hadamard()
            .unwrap_or_else(|| panic!("{filename} is a folded file and must have a hook"));
        assert_eq!(hook.block_size(), 1024, "{filename}");
        assert_eq!(hook.folded_count(), FOLDED_TENSORS, "{filename}");
        assert!(hook.is_inverse(names::TOKEN_EMBD), "{filename}");
        assert!(hook.is_folded(names::OUTPUT), "{filename}");
        assert_eq!(hook.config().inverse.len(), 1, "{filename}");
        assert!(hook.config().gdn_v_grouped, "{filename}");
        // The exclusions B2-11 must honour: feeding these the ROTATED
        // activation is silent wrong math (design §3.4).
        for layer in split.linear_layers().iter().copied() {
            for suffix in [
                names::SSM_ALPHA,
                names::SSM_BETA,
                names::SSM_CONV1D,
                names::SSM_A,
                names::SSM_DT_BIAS,
                names::SSM_NORM,
                names::ATTN_NORM,
                names::POST_ATTENTION_NORM,
            ] {
                let name = block_tensor(layer, suffix);
                assert!(
                    !hook.is_folded(&name),
                    "{filename}: {name} must NOT be folded"
                );
            }
            for suffix in [names::ATTN_QKV, names::ATTN_GATE, names::SSM_OUT] {
                let name = block_tensor(layer, suffix);
                assert!(hook.is_folded(&name), "{filename}: {name} must be folded");
            }
        }
        // Every rotated width has a full sign vector.
        let scratch = model.hadamard_scratch().expect("scratch allocates");
        assert_eq!(scratch.widths(), vec![HIDDEN, HEADS_WIDTH, INTERMEDIATE]);

        // -- v-head map and caches ---------------------------------------
        assert_eq!(model.vhead_map().n_v_heads(), N_V_HEADS);
        assert_eq!(model.vhead_map().v_per_k(), 3);
        assert_eq!(model.recurrent().n_layers(), N_LINEAR);
        assert_eq!(
            model.recurrent().memory_bytes(),
            RECURRENT_BYTES,
            "{filename}"
        );
        assert_eq!(model.max_seq_len(), 64);

        // -- reporting (what the CLI prints) -----------------------------
        assert_eq!(model.quant_type(), expected_quant, "{filename}");
        assert_eq!(model.variant(), Some(expected), "{filename}");
        let description = model.describe();
        for fragment in ["qwen35", "64 layers", "16 full", "48 linear", "401 folded"] {
            assert!(description.contains(fragment), "{filename}: {description}");
        }
    }

    #[test]
    fn loads_the_real_27b_pq2_0() {
        check_real_file(
            "Ternary-Bonsai-2-27B-PQ2_0.gguf",
            GgufTensorType::PQ2_0,
            ModelVariant::TernaryBonsai227bPq2,
        );
    }

    #[test]
    fn loads_the_real_27b_ptq1_0() {
        check_real_file(
            "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            GgufTensorType::PTQ1_0,
            ModelVariant::TernaryBonsai227bPtq1,
        );
    }
}
