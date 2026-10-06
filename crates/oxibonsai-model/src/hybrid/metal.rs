//! Metal runner for a loaded `qwen35` / PrismML Bonsai 2 [`HybridModel`]
//! (MET-09, design §4.2): binds the model's weights to the hybrid Metal
//! encoder and drives it token by token or chunk by chunk.
//!
//! # What runs where
//!
//! Everything per layer runs on the GPU, in one command buffer per call:
//! the RMSNorms with their fused Hadamard rotation, every projection, the
//! depthwise conv, the Gated-DeltaNet recurrence, partial RoPE, the sparse
//! `f16` KV cache and its attention, the output gates, the FFN and the LM
//! head. The host does only the embedding lookup of the prompt tokens and
//! its inverse rotation — one row per token, exactly
//! [`crate::hybrid::HadamardHook::inverse_embedding`]'s arithmetic — and
//! hands the rows over.
//!
//! # State
//!
//! The runner owns its own device state: the KV cache (one slot per
//! full-attention layer, indexed like [`crate::hybrid::LayerSplit::kv_slot`]),
//! the recurrent state and the conv windows. It shares nothing mutable with
//! the [`HybridModel`] it was built from — only the model's immutable weight
//! slices, which the device reads in place when the runner is built with
//! [`HybridMetalRunner::new_mapped`] or [`HybridMetalRunner::new_in_place`] —
//! so the CPU model stays usable next to it, e.g. as a parity reference or
//! for the embedding pass.
//!
//! The runner counts the positions its state has consumed
//! ([`HybridMetalRunner::token_count`]) the way the CPU model's recurrent
//! cache does, so a caller can hold it to the same contiguous-position
//! contract. It also keeps the sequence's M-RoPE offsets in the CPU model's
//! own position-keyed log (`RopeOffsets`, design §6.2): after an image
//! prefilled through [`HybridMetalRunner::forward_prefill_rows`] every later
//! text token rotates at its sequence position minus the offset in force
//! there ([`HybridMetalRunner::rope_delta`]), a forward at `pos` forgets the
//! offsets of a continuation abandoned past `pos`, and a reset clears them —
//! exactly the CPU model's bookkeeping. The recurrent state can be copied out and back
//! ([`HybridMetalRunner::snapshot_state`] /
//! [`HybridMetalRunner::restore_state`], `metal_state.rs`): an exact
//! rollback point, like the CPU model's `RecurrentCache::snapshot`. A GPU
//! call that fails abandons the sequence — the state is cleared and the
//! count returns to zero — since a command buffer that failed part-way may
//! have advanced some layers and not others.
//!
//! # Context capacity
//!
//! Construction refuses a KV window the device cannot keep resident
//! (`Qwen35GpuModel::max_context`: 178 032 positions for the 27B's `PQ2_0`
//! band and 196 244 for `PTQ1_0` on a 24 GB M3). Keeping the CPU model
//! beside the runner doubles the per-position cost (two KV caches, two
//! recurrent states), so under the runtime's RAM budget such a process fits
//! ≈ 84 K (`PQ2_0`) / ≈ 93 K (`PTQ1_0`) positions — the derivation is in the
//! `metal_full_layer::qwen35` module docs.
//!
//! # Prefill
//!
//! A multi-token call runs its projections as tiled GEMMs
//! ([`Qwen35PrefillMode::Batched`], the default): each weight is decoded
//! once per 64 tokens instead of once per token, so a prefill runs several
//! times faster than decode. The GEMM sums in another order than the decode
//! GEMV, so a batched prefill agrees with feeding the tokens one at a time
//! to rounding; [`Qwen35PrefillMode::Sequential`]
//! ([`HybridMetalRunner::set_prefill_mode`]) makes it bit-identical at the
//! decode rate. Decode is the GEMV in both modes.
//!
//! # Selection
//!
//! The runner is an explicit object: nothing here, and nothing in
//! [`HybridModel`], consults the environment. Choosing Metal or CPU is the
//! caller's backend decision — the runtime engine builds a runner for
//! `--backend metal` and, when a device exists and the model is served, for
//! `--backend auto`; [`HybridMetalRunner::check_supported`] tells a caller,
//! without touching the device, whether a model's geometry and weight
//! formats are ones the Metal kernels serve, and
//! [`HybridMetalRunner::footprint`] what a runner would keep resident.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use oxibonsai_kernels::error::KernelError;
use oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::{
    Qwen35FullAttentionWeights, Qwen35GpuConfig, Qwen35GpuModel, Qwen35LayerTrace,
    Qwen35LayerWeights, Qwen35LinearAttentionWeights, Qwen35MappedRegion, Qwen35Matrix,
    Qwen35MatrixData, Qwen35MatrixId, Qwen35ModelWeights, Qwen35PrefillMode, Qwen35Residency,
    Qwen35Rope,
};
use oxibonsai_kernels::hadamard::fwht_inverse_signed;
use oxibonsai_kernels::rope_mrope::{mrope_build_tables, partial_rope_build_table};
use oxibonsai_kernels::MetalGraphError;

use crate::error::{ModelError, ModelResult};
use crate::hybrid::block::{FullAttnBlock, HybridBlock, LinearAttnBlock};
use crate::hybrid::forward::RopeTables;
use crate::hybrid::model::HybridModel;
use crate::hybrid::vision_prefill::RopeOffsets;
use crate::hybrid::weights::{Bf16Matrix, HybridEmbedding};
use crate::layers::linear::LinearLayer;
use crate::layers::rms_norm::RmsNorm;
use crate::layers::rope_mrope::MropePos;
use crate::vision::mrope::next_rope_position;

/// A [`HybridModel`] running on the Metal GPU.
///
/// `'a` is the lifetime of the model's weight storage (the mapped GGUF):
/// the runner reads the quantized blocks in place for as long as it lives.
pub struct HybridMetalRunner<'a> {
    gpu: Qwen35GpuModel<'a>,
    embedding: HybridEmbedding<'a>,
    /// `token_embd`'s sign vector when the checkpoint is Hadamard-folded.
    inverse_signs: Option<Arc<[f32]>>,
    /// FWHT block of the fold (`0` when unfolded).
    hadamard_block: usize,
    hidden: usize,
    vocab: usize,
    max_seq_len: usize,
    max_batch: usize,
    /// Positions the recurrent state has consumed since the last reset
    /// (see [`Self::token_count`]).
    token_count: usize,
    /// The sequence's M-RoPE offsets by the sequence position each takes
    /// effect at (see the module docs).
    rope_offsets: RopeOffsets,
    /// `rope.dimension_sections`, for the 3-axis angles of image rows.
    rope_sections: [u32; 4],
    /// Rotated dimensions per head.
    n_rot: usize,
    /// RoPE frequency base.
    rope_freq_base: f32,
    /// Host staging for the embedded rows of one call, `[t][hidden]`.
    rows: Vec<f32>,
}

impl std::fmt::Debug for HybridMetalRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridMetalRunner")
            .field("gpu", &self.gpu)
            .field("folded", &self.inverse_signs.is_some())
            .field("hidden", &self.hidden)
            .field("vocab", &self.vocab)
            .field("max_seq_len", &self.max_seq_len)
            .field("max_batch", &self.max_batch)
            .field("token_count", &self.token_count)
            .field("rope_offsets", &self.rope_offsets)
            .field("prefill_mode", &self.gpu.prefill_mode())
            .finish()
    }
}

impl<'a> HybridMetalRunner<'a> {
    /// Build a runner whose device buffers hold their own copy of every
    /// weight matrix.
    ///
    /// The KV window is the model's own [`HybridModel::max_seq_len`] and one
    /// GPU call takes at most [`HybridModel::prefill_chunk`] tokens (longer
    /// prefills are chunked).
    ///
    /// # Errors
    ///
    /// [`ModelError::Kernel`] carrying [`KernelError::UnsupportedOperation`]
    /// for a geometry or a weight format the Metal kernels cannot serve (see
    /// [`Self::check_supported`]) or a KV window past the device's capacity,
    /// and [`KernelError::GpuError`] when there is no Metal device or an
    /// allocation fails.
    pub fn new(model: &HybridModel<'a>) -> ModelResult<Self> {
        Self::build(model, Qwen35Residency::Copied, None)
    }

    /// Build a runner that reads the weight matrices **in place** from the
    /// file mapping they live in: the whole mapping becomes one no-copy
    /// Metal buffer and each matrix is bound at its offset, so the model's
    /// weights cost no extra memory and no upload time. Matrices outside the
    /// mapping (none, for a model loaded from `mapping`) are copied.
    ///
    /// `mapping` must be the mapping of the GGUF `model` was loaded from and
    /// start on a page boundary — which a whole-file `memmap2::Mmap` always
    /// does; any other start is refused rather than read.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus [`KernelError::UnsupportedOperation`] for a
    /// mapping that does not start on a page boundary.
    pub fn new_mapped(model: &HybridModel<'a>, mapping: &'a memmap2::Mmap) -> ModelResult<Self> {
        // SAFETY: `mapping` is a live `memmap2::Mmap` borrowed for `'a`, so
        // every page of it stays mapped for as long as the runner (which
        // cannot outlive `'a`) may read it. The page-alignment half of the
        // contract is checked by the encoder before the no-copy buffer is
        // created: an unaligned start is refused, never wrapped.
        let region = unsafe { Qwen35MappedRegion::from_mapping(&mapping[..]) };
        Self::build(model, Qwen35Residency::Mapped(region), None)
    }

    /// Build a runner over `file_bytes`, the bytes of the GGUF `model` was
    /// loaded from: the weights are read **in place** (as
    /// [`Self::new_mapped`]) when `file_bytes` starts on a host page
    /// boundary — which the whole-file memory mapping every GGUF loader
    /// hands out always does — and copied (as [`Self::new`]) when it does
    /// not, e.g. a GGUF image assembled in a heap buffer.
    ///
    /// This is the constructor for a caller that holds the parsed GGUF
    /// (`GgufFile::data`) rather than its `memmap2::Mmap`.
    /// [`Self::is_mapped`] tells which residency was chosen.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn new_in_place(model: &HybridModel<'a>, file_bytes: &'a [u8]) -> ModelResult<Self> {
        Self::new_in_place_with_batch(model, file_bytes, model.prefill_chunk())
    }

    /// [`Self::new_in_place`] with calls of up to `max_batch` tokens instead
    /// of the model's prefill chunk — what a caller that budgeted the
    /// activation scratch for a larger (or smaller) prefill chunk builds.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], plus [`KernelError::UnsupportedOperation`] for a
    /// zero `max_batch`.
    pub fn new_in_place_with_batch(
        model: &HybridModel<'a>,
        file_bytes: &'a [u8],
        max_batch: usize,
    ) -> ModelResult<Self> {
        let residency = match Qwen35MappedRegion::page_aligned(file_bytes) {
            Some(region) => Qwen35Residency::Mapped(region),
            None => Qwen35Residency::Copied,
        };
        Self::build(model, residency, Some(max_batch))
    }

    /// Whether `model` is one the Metal kernels can run, checked without
    /// touching the device: its geometry ([`Qwen35GpuConfig::validate`]),
    /// every projection's weight format, the norm epsilons, the `v`-head
    /// order and the KV window's `u32` range.
    ///
    /// # Errors
    ///
    /// [`ModelError::Kernel`] carrying [`KernelError::UnsupportedOperation`]
    /// naming the first constraint `model` violates.
    pub fn check_supported(model: &HybridModel<'_>) -> ModelResult<()> {
        let cfg = gpu_config(model)?;
        cfg.validate().map_err(from_gpu)?;
        check_vhead_map(model, &cfg)?;
        let eps = cfg.rms_eps;
        for block in model.blocks() {
            match block {
                HybridBlock::Full(b) => {
                    for (name, norm) in full_norms(b) {
                        check_eps(b.layer_idx(), name, norm, eps)?;
                    }
                    for (name, layer) in full_matrices(b) {
                        gpu_matrix(b.layer_idx(), name, layer)?;
                    }
                }
                HybridBlock::Linear(b) => {
                    for (name, norm) in linear_norms(b) {
                        check_eps(b.layer_idx(), name, norm, eps)?;
                    }
                    for (name, layer) in linear_matrices(b) {
                        gpu_matrix(b.layer_idx(), name, layer)?;
                    }
                    for (name, gate) in [("ssm_alpha", b.ssm_alpha()), ("ssm_beta", b.ssm_beta())] {
                        check_gate_shape(b.layer_idx(), name, gate, &cfg)?;
                    }
                }
            }
        }
        check_eps(usize::MAX, "output_norm", model.output_norm(), eps)?;
        gpu_matrix(usize::MAX, "output", model.lm_head())?;
        Ok(())
    }

    fn build(
        model: &HybridModel<'a>,
        residency: Qwen35Residency<'a>,
        max_batch: Option<usize>,
    ) -> ModelResult<Self> {
        Self::check_supported(model)?;
        let mut cfg = gpu_config(model)?;
        if let Some(batch) = max_batch {
            if batch == 0 {
                return Err(unsupported(
                    "qwen35 Metal runner: max_batch must be at least 1".to_string(),
                ));
            }
            cfg.max_batch = batch;
        }
        let config = model.config();

        // Partial-RoPE angles: the CPU forward's own table, flattened.
        let rope = RopeTables::new(cfg.n_rot, cfg.max_seq_len, config.base.rope_freq_base)?;
        let half = cfg.n_rot / 2;
        let mut rope_cos = Vec::with_capacity(cfg.max_seq_len * half);
        let mut rope_sin = Vec::with_capacity(cfg.max_seq_len * half);
        for pos in 0..cfg.max_seq_len {
            let (cos, sin) = rope.angles(pos)?;
            rope_cos.extend_from_slice(cos);
            rope_sin.extend_from_slice(sin);
        }

        // `ssm_alpha` / `ssm_beta` widened to one `[2 * n_v][hidden]` f32
        // matrix per linear layer (tiled row order, as stored). They are
        // copied to the device, so the widened copies live only until the
        // model is built.
        let mut alpha_beta: Vec<Vec<f32>> = Vec::new();
        for block in model.blocks() {
            if let HybridBlock::Linear(b) = block {
                alpha_beta.push(widen_gates(b, &cfg));
            }
        }

        let mut layers = Vec::with_capacity(model.blocks().len());
        let mut next_linear = 0usize;
        for block in model.blocks() {
            layers.push(match block {
                HybridBlock::Full(b) => {
                    Qwen35LayerWeights::FullAttention(Box::new(full_weights(b)?))
                }
                HybridBlock::Linear(b) => {
                    let ab = alpha_beta.get(next_linear).ok_or_else(|| {
                        ModelError::Internal(format!(
                            "layer {}: no widened ssm_alpha/ssm_beta",
                            b.layer_idx()
                        ))
                    })?;
                    next_linear += 1;
                    Qwen35LayerWeights::LinearAttention(Box::new(linear_weights(b, ab)?))
                }
            });
        }

        let mut signs: Vec<(usize, &[f32])> = Vec::new();
        let (inverse_signs, hadamard_block) = match model.hadamard() {
            Some(hook) => {
                let spec = hook.config();
                for width in [cfg.hidden, cfg.intermediate, cfg.heads_width(), cfg.inner()] {
                    if signs.iter().all(|(w, _)| *w != width) {
                        signs.push((width, spec.signs_for(width).map_err(ModelError::Core)?));
                    }
                }
                let inverse = spec.signs.get(&cfg.hidden).cloned().ok_or_else(|| {
                    unsupported(format!(
                        "qwen35 Metal runner: the fold has no sign vector for the \
                         embedding width {}",
                        cfg.hidden
                    ))
                })?;
                (Some(inverse), hook.block_size())
            }
            None => (None, 0),
        };

        let weights = Qwen35ModelWeights {
            config: cfg.clone(),
            layers,
            output_norm: model.output_norm().weight(),
            lm_head: gpu_matrix(usize::MAX, "output", model.lm_head())?,
            signs,
            rope_cos: &rope_cos,
            rope_sin: &rope_sin,
        };
        let gpu = Qwen35GpuModel::new(&weights, residency).map_err(from_gpu)?;

        // The encoder assigns KV / recurrent slots in layer order, exactly as
        // `LayerSplit` does; a disagreement would read another layer's
        // history, so it is checked rather than assumed.
        for (layer, slot) in gpu.layer_kv_slots().iter().enumerate() {
            let expected = model
                .split()
                .kv_slot(layer)
                .and_then(|s| u32::try_from(s).ok());
            if *slot != expected {
                return Err(ModelError::Internal(format!(
                    "layer {layer}: Metal KV slot {slot:?} disagrees with the model's {expected:?}"
                )));
            }
        }

        Ok(Self {
            gpu,
            embedding: rebind_embedding(model.embedding()),
            inverse_signs,
            hadamard_block,
            hidden: cfg.hidden,
            vocab: cfg.vocab,
            max_seq_len: cfg.max_seq_len,
            max_batch: cfg.max_batch,
            token_count: 0,
            rope_offsets: RopeOffsets::default(),
            rope_sections: config.rope_sections,
            n_rot: cfg.n_rot,
            rope_freq_base: config.base.rope_freq_base,
            rows: Vec::new(),
        })
    }

    /// The geometry the runner was built with.
    #[must_use]
    pub fn config(&self) -> &Qwen35GpuConfig {
        self.gpu.config()
    }

    /// KV window, in positions.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Tokens one GPU call takes; longer prefills are chunked.
    #[must_use]
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Change the most tokens one GPU call takes (the activation scratch
    /// follows at the next call that needs it).
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedOperation`] for a zero batch or one whose
    /// scratch would push the KV window past this device's capacity (the
    /// message names both numbers); nothing changes then.
    pub fn set_max_batch(&mut self, max_batch: usize) -> ModelResult<()> {
        self.gpu.set_max_batch(max_batch).map_err(from_gpu)?;
        self.max_batch = max_batch;
        Ok(())
    }

    /// How multi-token calls run their projections (see the module docs).
    #[must_use]
    pub fn prefill_mode(&self) -> Qwen35PrefillMode {
        self.gpu.prefill_mode()
    }

    /// Select how multi-token calls run their projections: the tiled GEMM
    /// ([`Qwen35PrefillMode::Batched`], the default) or the decode GEMV once
    /// per token ([`Qwen35PrefillMode::Sequential`], bit-identical to
    /// feeding the tokens one at a time).
    pub fn set_prefill_mode(&mut self, mode: Qwen35PrefillMode) {
        self.gpu.set_prefill_mode(mode);
    }

    /// Tokens a call needs before the batched mode runs its projections on
    /// the GEMM (`Q35_GEMM_MIN_COLS`, 16, unless changed).
    #[must_use]
    pub fn gemm_min_cols(&self) -> usize {
        self.gpu.gemm_min_cols()
    }

    /// Change the call size from which the batched mode takes the GEMM
    /// (at least 2: decode is always the GEMV) — a parity check puts every
    /// prefill of a short prompt on the GEMM this way.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedOperation`] below 2.
    pub fn set_gemm_min_cols(&mut self, min_cols: usize) -> ModelResult<()> {
        self.gpu.set_gemm_min_cols(min_cols).map_err(from_gpu)
    }

    /// Vocabulary size (logit row length).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// Whether the weights are read in place from the file mapping.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.gpu.is_mapped()
    }

    /// Weight bytes the device reads (mapped plus copied).
    #[must_use]
    pub fn weight_bytes(&self) -> u64 {
        self.gpu.weight_bytes()
    }

    /// Bytes of the device KV cache (K and V together).
    #[must_use]
    pub fn kv_cache_bytes(&self) -> u64 {
        self.gpu.kv_cache_bytes()
    }

    /// GPU execution time of the last forward's command buffer, in seconds.
    #[must_use]
    pub fn last_gpu_seconds(&self) -> f64 {
        self.gpu.last_gpu_seconds()
    }

    /// Clear the recurrent state and the conv windows (a new sequence), and
    /// the position count and the M-RoPE offsets with them.
    ///
    /// The KV cache needs no clearing: every position is written before any
    /// query at or after it reads it.
    pub fn reset(&mut self) {
        self.gpu.reset();
        self.token_count = 0;
        self.rope_offsets.clear();
    }

    /// The M-RoPE offset of the current sequence: its next token sits at
    /// sequence position `p` ([`Self::token_count`]) but rotates at the text
    /// position `p - rope_delta()` (design §6.2) — `0` for any text-only
    /// sequence, `h * w - max(h, w)` more after each image of a merged
    /// `h x w` grid. The CPU model's `HybridModel::rope_delta`, kept the
    /// same way.
    #[must_use]
    pub fn rope_delta(&self) -> usize {
        self.rope_offsets.at(self.token_count)
    }

    /// The M-RoPE offset in force at sequence position `pos` of the current
    /// sequence (the offset the token there rotates behind; `0` before any
    /// image) — what a caller checks a rollback point against.
    #[must_use]
    pub fn rope_offset_at(&self, pos: usize) -> usize {
        self.rope_offsets.at(pos)
    }

    /// The text rotary position a prompt starting at sequence position
    /// `start_pos` begins at, without changing any state: `0` for a new
    /// sequence (`start_pos == 0`), else `start_pos` minus the offset in
    /// force there — what a caller assembling a multimodal prompt for this
    /// runner passes as its rope start.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for a `start_pos` before the offset in
    /// force there (no sequence this runner holds has such a position).
    pub fn rope_start_for(&self, start_pos: usize) -> ModelResult<usize> {
        if start_pos == 0 {
            return Ok(0);
        }
        let delta = self.rope_offsets.at(start_pos);
        start_pos
            .checked_sub(delta)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "rotary position".to_string(),
                expected: format!("a sequence position at or past the M-RoPE offset {delta}"),
                actual: start_pos.to_string(),
            })
    }

    /// Positions the recurrent state has consumed since the last reset: the
    /// end of the last successful [`Self::forward_into`] /
    /// [`Self::forward_prefill`] / [`Self::forward_with_dump`] call (or the
    /// position of a restored snapshot) — the next position of a contiguous
    /// sequence, exactly as `RecurrentCache::token_count` is on the CPU.
    ///
    /// [`Self::trace_layer`] advances a single layer's state and leaves the
    /// count alone: a traced runner must be reset before it decodes again.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Abandon the sequence after a failed GPU call: a command buffer that
    /// failed part-way may have advanced some layers' state and not others,
    /// so nothing short of a reset (which also clears the M-RoPE offsets) is
    /// a known state.
    fn abandon(&mut self, error: MetalGraphError) -> ModelError {
        self.reset();
        from_gpu(error)
    }

    /// The embedding rows of `tokens` after the inverse rotation,
    /// `[t][hidden]` — the input of layer 0, bit-identical to the CPU
    /// forward's.
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] for a token past the table, and a
    /// kernel error from the inverse transform.
    pub fn embed(&mut self, tokens: &[u32]) -> ModelResult<Vec<f32>> {
        self.embed_rows(tokens)?;
        Ok(self.rows.clone())
    }

    fn embed_rows(&mut self, tokens: &[u32]) -> ModelResult<()> {
        let hidden = self.hidden;
        self.rows.clear();
        self.rows.resize(tokens.len() * hidden, 0.0);
        for (t, &token) in tokens.iter().enumerate() {
            let row = self
                .rows
                .get_mut(t * hidden..(t + 1) * hidden)
                .ok_or_else(|| ModelError::Internal("embedding staging too short".to_string()))?;
            self.embedding.row(token, hidden, row)?;
            if let Some(signs) = &self.inverse_signs {
                fwht_inverse_signed(row, signs, self.hadamard_block).map_err(ModelError::Kernel)?;
            }
        }
        Ok(())
    }

    fn check_window(&self, start_pos: usize, t_len: usize) -> ModelResult<()> {
        let end = start_pos
            .checked_add(t_len)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "chunk end position".to_string(),
                expected: "start_pos + t_len representable as usize".to_string(),
                actual: format!("start_pos = {start_pos}, t_len = {t_len}"),
            })?;
        if end > self.max_seq_len {
            return Err(ModelError::PositionOutOfRange {
                pos: end.saturating_sub(1),
                max: self.max_seq_len,
            });
        }
        Ok(())
    }

    fn check_logits(&self, logits: &[f32]) -> ModelResult<()> {
        if logits.len() < self.vocab {
            return Err(ModelError::ShapeMismatch {
                name: "logits".to_string(),
                expected: vec![self.vocab],
                actual: vec![logits.len()],
            });
        }
        Ok(())
    }

    /// Single-token decode at absolute position `pos`, returning the
    /// `[vocab]` logits.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_into`].
    pub fn forward(&mut self, token: u32, pos: usize) -> ModelResult<Vec<f32>> {
        let mut logits = vec![0.0f32; self.vocab];
        self.forward_into(token, pos, &mut logits)?;
        Ok(logits)
    }

    /// Single-token decode at absolute position `pos`, writing the `[vocab]`
    /// logits into `logits`.
    ///
    /// # Errors
    ///
    /// [`ModelError::PositionOutOfRange`] past the KV window or for a token
    /// past the vocabulary, [`ModelError::ShapeMismatch`] for a short
    /// `logits` — all refused before any GPU work, leaving the sequence as
    /// it was — and [`KernelError::GpuError`] for a failed command buffer,
    /// which abandons the sequence (see the module docs).
    pub fn forward_into(&mut self, token: u32, pos: usize, logits: &mut [f32]) -> ModelResult<()> {
        self.check_logits(logits)?;
        self.check_window(pos, 1)?;
        let rope_start = self.rope_start_for(pos)?;
        self.embed_rows(&[token])?;
        // The sequence continues at `pos`: the offsets of a continuation
        // abandoned past it no longer apply.
        self.rope_offsets.continue_at(pos);
        let rope = Qwen35Rope::Contiguous { rope_start };
        if let Err(e) = self.gpu.forward_rows(&self.rows, pos, rope, Some(logits)) {
            return Err(self.abandon(e));
        }
        self.token_count = pos + 1;
        Ok(())
    }

    /// Prefill `tokens` from absolute position `start_pos`, writing the
    /// **last** token's `[vocab]` logits into `last_logits`.
    ///
    /// Runs in [`Self::max_batch`]-token chunks; every chunk advances the
    /// recurrent state once per token, in order, and stores every key and
    /// value at its absolute position. Token `i` rotates at `start_pos + i -
    /// rope_delta` ([`Self::rope_delta`]). In
    /// [`Qwen35PrefillMode::Sequential`] the result is bit-identical to
    /// feeding the tokens one at a time; in the default
    /// [`Qwen35PrefillMode::Batched`] the projections run as tiled GEMMs and
    /// agree with it to rounding — and do not depend on the chunking as long
    /// as every chunk is at least
    /// `oxibonsai_kernels::gpu_backend::metal_full_layer::qwen35::Q35_GEMM_MIN_COLS`
    /// tokens.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_into`].
    pub fn forward_prefill(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        last_logits: &mut [f32],
    ) -> ModelResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        self.check_logits(last_logits)?;
        self.check_window(start_pos, tokens.len())?;
        // Every id is checked before the first chunk runs, so a bad token
        // late in a long prompt is refused without advancing the state.
        let vocab = u32::try_from(self.vocab).unwrap_or(u32::MAX);
        if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab) {
            return Err(ModelError::PositionOutOfRange {
                pos: bad as usize,
                max: self.vocab,
            });
        }
        let rope_start = self.rope_start_for(start_pos)?;
        self.rope_offsets.continue_at(start_pos);
        let n_chunks = tokens.len().div_ceil(self.max_batch);
        for (i, chunk) in tokens.chunks(self.max_batch).enumerate() {
            self.embed_rows(chunk)?;
            let offset = i * self.max_batch;
            let logits = if i + 1 == n_chunks {
                Some(&mut *last_logits)
            } else {
                None
            };
            let rope = Qwen35Rope::Contiguous {
                rope_start: rope_start + offset,
            };
            if let Err(e) = self
                .gpu
                .forward_rows(&self.rows, start_pos + offset, rope, logits)
            {
                return Err(self.abandon(e));
            }
            self.token_count = start_pos + offset + chunk.len();
        }
        Ok(())
    }

    /// Run `tokens` (at most [`Self::max_batch`]) layer by layer, returning
    /// the residual stream after every layer (`[layer][t * hidden]`) and the
    /// last token's logits — the GPU counterpart of
    /// [`HybridModel::forward_with_dump`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward_into`], plus [`ModelError::ShapeInvariant`] for
    /// more than [`Self::max_batch`] tokens.
    pub fn forward_with_dump(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> ModelResult<(Vec<Vec<f32>>, Vec<f32>)> {
        self.check_batch(tokens.len())?;
        self.check_window(start_pos, tokens.len())?;
        let rope_start = self.rope_start_for(start_pos)?;
        self.embed_rows(tokens)?;
        self.rope_offsets.continue_at(start_pos);
        let rope = Qwen35Rope::Contiguous { rope_start };
        match self.gpu.forward_with_dump_rows(&self.rows, start_pos, rope) {
            Ok(dump) => {
                self.token_count = start_pos + tokens.len();
                Ok((dump.layers, dump.logits))
            }
            Err(e) => Err(self.abandon(e)),
        }
    }

    /// Prefill caller-supplied rows starting at sequence position
    /// `start_pos`, writing the **last** row's `[vocab]` logits into
    /// `logits_out` when given — the runner's twin of the CPU
    /// `HybridModel::forward_prefill_rows`, same contract.
    ///
    /// `rows` is `positions.len()` rows of `hidden` floats **already in the
    /// basis layer 0 reads**: they enter the residual stream verbatim, with
    /// no embedding lookup and no inverse Hadamard transform — text rows as
    /// `HybridModel::embed_token_rows` produces them, image rows straight
    /// from the vision tower (whose rows are in the unrotated basis already;
    /// transforming them would be a silent correctness bug). Row `t` is
    /// stored at KV position `start_pos + t`, advances the recurrent state
    /// once, and rotates in the full-attention layers at `positions[t]`
    /// (3-axis M-RoPE, design §6.2) — angles built on the host with the
    /// arithmetic of the CPU model's `RopeTables::fill_angles`, so a text
    /// position rotates exactly like the resident table's row. Runs in
    /// [`Self::max_batch`]-row chunks. Attention stays causal in sequence
    /// order, which for an image whose rows are row-major over its merged
    /// grid is the reference's 2-D causal mask.
    ///
    /// Afterwards the next token sits at sequence position `start_pos +
    /// positions.len()` and rotates at one past the largest axis
    /// `positions` used; the difference is recorded as the M-RoPE offset
    /// ([`Self::rope_delta`]) for every later forward. An empty `positions`
    /// is a no-op.
    ///
    /// # Errors
    ///
    /// All checked before any GPU work, leaving the sequence as it was:
    /// [`ModelError::ShapeMismatch`] for `rows` of the wrong length or a
    /// short `logits_out`, [`ModelError::PositionOutOfRange`] past the KV
    /// window (or an axis past the angle range), and
    /// [`ModelError::ShapeInvariant`] for rotary positions that run ahead of
    /// the sequence. A failed command buffer abandons the sequence.
    pub fn forward_prefill_rows(
        &mut self,
        rows: &[f32],
        positions: &[MropePos],
        start_pos: usize,
        mut logits_out: Option<&mut [f32]>,
    ) -> ModelResult<()> {
        let Some(rope_next) = self.check_rows(rows, positions, start_pos)? else {
            return Ok(());
        };
        if let Some(out) = logits_out.as_deref() {
            self.check_logits(out)?;
        }
        let (cos, sin) = self.row_angles(positions)?;
        let (hidden, half, n) = (self.hidden, self.n_rot / 2, positions.len());
        self.rope_offsets.continue_at(start_pos);
        let n_chunks = n.div_ceil(self.max_batch);
        for i in 0..n_chunks {
            let lo = i * self.max_batch;
            let hi = (lo + self.max_batch).min(n);
            let (chunk, cos, sin) = match (
                rows.get(lo * hidden..hi * hidden),
                cos.get(lo * half..hi * half),
                sin.get(lo * half..hi * half),
            ) {
                (Some(chunk), Some(cos), Some(sin)) => (chunk, cos, sin),
                _ => {
                    return Err(ModelError::Internal(format!(
                        "rows prefill chunk {lo}..{hi} outside the validated rows"
                    )))
                }
            };
            let logits = if i + 1 == n_chunks {
                logits_out.as_deref_mut()
            } else {
                None
            };
            let rope = Qwen35Rope::PerRow { cos, sin };
            if let Err(e) = self.gpu.forward_rows(chunk, start_pos + lo, rope, logits) {
                return Err(self.abandon(e));
            }
            self.token_count = start_pos + hi;
        }
        self.record_rope(start_pos, start_pos + n, rope_next)
    }

    /// [`Self::forward_prefill_rows`] as one call (at most
    /// [`Self::max_batch`] rows), recording what layer 0 read — read back
    /// from the device, the counterpart of the CPU `LayerDump::embedding` —
    /// every layer's output and the last row's logits.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_prefill_rows`], plus [`ModelError::ShapeInvariant`]
    /// for an empty prompt or more than [`Self::max_batch`] rows.
    pub fn forward_prefill_rows_with_dump(
        &mut self,
        rows: &[f32],
        positions: &[MropePos],
        start_pos: usize,
    ) -> ModelResult<HybridMetalRowsDump> {
        let rope_next = self
            .check_rows(rows, positions, start_pos)?
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "prefill rows".to_string(),
                expected: "at least one row to record".to_string(),
                actual: "0".to_string(),
            })?;
        self.check_batch(positions.len())?;
        let (cos, sin) = self.row_angles(positions)?;
        self.rope_offsets.continue_at(start_pos);
        let rope = Qwen35Rope::PerRow {
            cos: &cos,
            sin: &sin,
        };
        let dump = match self.gpu.forward_with_dump_rows(rows, start_pos, rope) {
            Ok(dump) => dump,
            Err(e) => return Err(self.abandon(e)),
        };
        self.token_count = start_pos + positions.len();
        self.record_rope(start_pos, start_pos + positions.len(), rope_next)?;
        Ok(HybridMetalRowsDump {
            embedding: dump.input,
            layers: dump.layers,
            logits: dump.logits,
        })
    }

    /// Validate a rows prefill before touching any state; `Ok(None)` for an
    /// empty one, else the rotary position the token after it takes (the
    /// CPU model's own checks).
    fn check_rows(
        &self,
        rows: &[f32],
        positions: &[MropePos],
        start_pos: usize,
    ) -> ModelResult<Option<usize>> {
        let Some(rope_next) = next_rope_position(positions) else {
            return if rows.is_empty() {
                Ok(None)
            } else {
                Err(ModelError::ShapeMismatch {
                    name: "prefill rows".to_string(),
                    expected: vec![0],
                    actual: vec![rows.len()],
                })
            };
        };
        let n = positions.len();
        if rows.len() != n.saturating_mul(self.hidden) {
            return Err(ModelError::ShapeMismatch {
                name: "prefill rows".to_string(),
                expected: vec![n, self.hidden],
                actual: vec![rows.len()],
            });
        }
        self.check_window(start_pos, n)?;
        let end = start_pos + n;
        if rope_next > end {
            return Err(ModelError::ShapeInvariant {
                tensor: "rotary positions".to_string(),
                expected: format!(
                    "no axis past its row's sequence position (next sequence position {end})"
                ),
                actual: format!("next rotary position {rope_next}"),
            });
        }
        Ok(Some(rope_next))
    }

    /// The `[n][n_rot / 2]` cos and sin rows of `positions`, with the CPU
    /// model's `RopeTables::fill_angles` arithmetic: a text position (`t ==
    /// h == w`) is the single-axis table row (`partial_rope_build_table`,
    /// which built the resident table), any other the interleaved M-RoPE of
    /// the model's sections (`mrope_build_tables`).
    fn row_angles(&self, positions: &[MropePos]) -> ModelResult<(Vec<f32>, Vec<f32>)> {
        let half = self.n_rot / 2;
        let mut cos = vec![0.0f32; positions.len() * half];
        let mut sin = vec![0.0f32; positions.len() * half];
        let axis = |v: u32| {
            i32::try_from(v).map_err(|_| ModelError::ShapeInvariant {
                tensor: "rope position".to_string(),
                expected: "representable as i32".to_string(),
                actual: v.to_string(),
            })
        };
        for ((pos, c), s) in positions
            .iter()
            .zip(cos.chunks_mut(half.max(1)))
            .zip(sin.chunks_mut(half.max(1)))
        {
            let axis_max = pos.t.max(pos.h).max(pos.w) as usize;
            if axis_max >= self.max_seq_len {
                return Err(ModelError::PositionOutOfRange {
                    pos: axis_max,
                    max: self.max_seq_len,
                });
            }
            let built = if pos.t == pos.h && pos.t == pos.w {
                partial_rope_build_table(axis(pos.t)?, self.n_rot, self.rope_freq_base, c, s)
            } else {
                mrope_build_tables(
                    [axis(pos.t)?, axis(pos.h)?, axis(pos.w)?],
                    self.rope_sections,
                    self.n_rot,
                    self.rope_freq_base,
                    c,
                    s,
                )
            };
            built.map_err(ModelError::Kernel)?;
        }
        Ok((cos, sin))
    }

    /// Record the M-RoPE offset a rows prefill that began at `start_pos`
    /// leaves: the next token sits at `seq_end` and rotates at `rope_next`.
    fn record_rope(
        &mut self,
        start_pos: usize,
        seq_end: usize,
        rope_next: usize,
    ) -> ModelResult<()> {
        let delta = seq_end
            .checked_sub(rope_next)
            .ok_or_else(|| ModelError::ShapeInvariant {
                tensor: "rotary positions".to_string(),
                expected: format!(
                    "no rotary position past the sequence position (next sequence position {seq_end})"
                ),
                actual: format!("next rotary position {rope_next}"),
            })?;
        self.rope_offsets.record(start_pos, seq_end, delta);
        Ok(())
    }

    /// Run layer `layer` alone on `hidden_rows` (its input residual rows,
    /// `[t][hidden]`), one kernel per command buffer, capturing every
    /// intermediate activation — the per-kernel parity harness. The layer's
    /// KV slot or recurrent state advances as in a forward.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_into`], plus an out-of-range `layer`.
    pub fn trace_layer(
        &mut self,
        layer: usize,
        hidden_rows: &[f32],
        start_pos: usize,
    ) -> ModelResult<Qwen35LayerTrace> {
        let t_len = hidden_rows.len() / self.hidden.max(1);
        self.check_batch(t_len)?;
        self.check_window(start_pos, t_len)?;
        self.gpu
            .trace_layer(layer, hidden_rows, start_pos)
            .map_err(from_gpu)
    }

    /// `y = W x` on the GPU for one matrix of layer `layer` (`layer ==
    /// n_layers` with [`Qwen35MatrixId::LmHead`] for the LM head): a one-hot
    /// `x` returns the dequantized column, the bitwise check of every weight
    /// decoder.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedOperation`] for a matrix the layer does not
    /// have or a wrong-length `x`; [`KernelError::GpuError`] for a failed
    /// command buffer.
    pub fn gemv_probe(
        &mut self,
        layer: usize,
        matrix: Qwen35MatrixId,
        x: &[f32],
    ) -> ModelResult<Vec<f32>> {
        self.gpu.gemv_probe(layer, matrix, x).map_err(from_gpu)
    }

    /// The fold's blockwise rotation of whole `width`-wide rows of `x`, on
    /// the GPU: forward (signs, then FWHT) or inverse (FWHT, then signs).
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedOperation`] for an unfolded model, a width
    /// without a sign vector or a partial row.
    pub fn rotate(&mut self, x: &[f32], width: usize, inverse: bool) -> ModelResult<Vec<f32>> {
        self.gpu.rotate(x, width, inverse).map_err(from_gpu)
    }

    fn check_batch(&self, t_len: usize) -> ModelResult<()> {
        if t_len == 0 || t_len > self.max_batch {
            return Err(ModelError::ShapeInvariant {
                tensor: "Metal runner call".to_string(),
                expected: format!("1..={} tokens", self.max_batch),
                actual: t_len.to_string(),
            });
        }
        Ok(())
    }
}

/// What [`HybridMetalRunner::forward_prefill_rows_with_dump`] records — the
/// runner's counterpart of the CPU `LayerDump`'s `embedding` and per-layer
/// rows.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridMetalRowsDump {
    /// What layer 0 read, read back from the device: `[rows][hidden]`.
    pub embedding: Vec<f32>,
    /// The residual stream after every layer (`[layer][rows * hidden]`).
    pub layers: Vec<Vec<f32>>,
    /// The last row's `[vocab]` logits.
    pub logits: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────
//  Binding helpers
// ─────────────────────────────────────────────────────────────────────────

/// A geometry or format the Metal kernels do not serve.
fn unsupported(msg: String) -> ModelError {
    ModelError::Kernel(KernelError::UnsupportedOperation(msg))
}

/// Map an encoder error: a refused geometry stays an unsupported
/// operation, everything else (no device, allocation, a failed command
/// buffer) is a GPU error.
fn from_gpu(e: MetalGraphError) -> ModelError {
    match e {
        MetalGraphError::InvalidDimensions(msg) => unsupported(msg),
        other => ModelError::Kernel(KernelError::GpuError(format!(
            "qwen35 Metal runner: {other}"
        ))),
    }
}

/// The encoder's view of `model`'s geometry.
fn gpu_config(model: &HybridModel<'_>) -> ModelResult<Qwen35GpuConfig> {
    let c = model.config();
    if c.base.value_length != c.base.head_dim {
        return Err(unsupported(format!(
            "qwen35 Metal runner: value_length {} differs from key_length {}",
            c.base.value_length, c.base.head_dim
        )));
    }
    Ok(Qwen35GpuConfig {
        hidden: c.base.hidden_size,
        intermediate: c.base.intermediate_size,
        n_heads: c.base.num_attention_heads,
        n_kv_heads: c.base.num_kv_heads,
        head_dim: c.base.head_dim,
        n_rot: c.rope_dimension_count,
        n_k_heads: c.n_k_heads(),
        n_v_heads: c.n_v_heads(),
        head_k_dim: c.head_k_dim(),
        head_v_dim: c.head_v_dim(),
        conv_kernel: c.ssm_conv_kernel,
        rms_eps: c.base.rms_norm_eps,
        hadamard_block: model.hadamard().map(|hook| hook.block_size()),
        vocab: c.base.vocab_size,
        max_seq_len: model.max_seq_len(),
        max_batch: model.prefill_chunk().max(1),
    })
}

/// The encoder reads `z`, `alpha`, `beta` and the `v` block of `attn_qkv`
/// through `tiled(m) = (m % rep) * n_k + m / rep`; the model's map must be
/// exactly that (it always is for a loadable fold — checked, not assumed).
fn check_vhead_map(model: &HybridModel<'_>, cfg: &Qwen35GpuConfig) -> ModelResult<()> {
    let map = model.vhead_map();
    let (nk, nv) = (cfg.n_k_heads, cfg.n_v_heads);
    if map.n_k_heads() != nk || map.n_v_heads() != nv || nk == 0 {
        return Err(unsupported(format!(
            "qwen35 Metal runner: v-head map is {} over {}, the config {nv} over {nk}",
            map.n_v_heads(),
            map.n_k_heads()
        )));
    }
    let rep = nv / nk;
    for grouped in 0..nv {
        let expected = (grouped % rep) * nk + grouped / rep;
        if map.tiled(grouped) != expected {
            return Err(unsupported(format!(
                "qwen35 Metal runner: grouped v-head {grouped} maps to tiled {} (the kernels \
                 read tiled {expected})",
                map.tiled(grouped)
            )));
        }
    }
    Ok(())
}

fn check_eps(layer: usize, name: &str, norm: &RmsNorm, eps: f32) -> ModelResult<()> {
    if norm.eps().to_bits() == eps.to_bits() {
        Ok(())
    } else {
        Err(unsupported(format!(
            "qwen35 Metal runner: {} {name} has epsilon {} (the kernels use {eps} throughout)",
            layer_label(layer),
            norm.eps()
        )))
    }
}

fn check_gate_shape(
    layer: usize,
    name: &str,
    gate: &Bf16Matrix<'_>,
    cfg: &Qwen35GpuConfig,
) -> ModelResult<()> {
    if gate.out_features() == cfg.n_v_heads && gate.in_features() == cfg.hidden {
        Ok(())
    } else {
        Err(unsupported(format!(
            "qwen35 Metal runner: {} {name} is {}x{}, expected {}x{}",
            layer_label(layer),
            gate.out_features(),
            gate.in_features(),
            cfg.n_v_heads,
            cfg.hidden
        )))
    }
}

fn layer_label(layer: usize) -> String {
    if layer == usize::MAX {
        "the model".to_string()
    } else {
        format!("layer {layer}")
    }
}

/// The on-disk format name of a projection.
fn format_name(layer: &LinearLayer<'_>) -> &'static str {
    match layer {
        LinearLayer::OneBit(_) => "Q1_0_g128",
        LinearLayer::Ternary(_) => "TQ2_0_g128",
        LinearLayer::FP8E4M3(_) => "FP8_E4M3",
        LinearLayer::FP8E5M2(_) => "FP8_E5M2",
        LinearLayer::Q4_0(_) => "Q4_0",
        LinearLayer::Q8_0(_) => "Q8_0",
        LinearLayer::Q5K(_) => "Q5_K",
        LinearLayer::Q6K(_) => "Q6_K",
        LinearLayer::Q2K(_) => "Q2_K",
        LinearLayer::Q3K(_) => "Q3_K",
        LinearLayer::Q4K(_) => "Q4_K",
        LinearLayer::Q8K(_) => "Q8_K",
        LinearLayer::PQ2_0(_) => "PQ2_0",
        LinearLayer::PTQ1_0(_) => "PTQ1_0",
        LinearLayer::Q2_0G64(_) => "Q2_0_g64",
        LinearLayer::Dense(_) => "F32",
    }
}

/// Borrow a projection's stored blocks for the encoder.
fn gpu_matrix<'w>(
    layer: usize,
    name: &str,
    projection: &'w LinearLayer<'_>,
) -> ModelResult<Qwen35Matrix<'w>> {
    let data = match projection {
        LinearLayer::PQ2_0(l) => Qwen35MatrixData::Pq2_0(l.blocks()),
        LinearLayer::PTQ1_0(l) => Qwen35MatrixData::Ptq1_0(l.blocks()),
        LinearLayer::Ternary(l) => Qwen35MatrixData::Tq2_0G128(l.blocks()),
        LinearLayer::Q2_0G64(l) => Qwen35MatrixData::Q2_0G64(l.blocks()),
        LinearLayer::OneBit(l) => Qwen35MatrixData::Q1_0G128(l.blocks()),
        LinearLayer::Dense(l) => Qwen35MatrixData::F32(l.weights()),
        other => {
            return Err(unsupported(format!(
                "qwen35 Metal runner: {} {name} is {}, which has no Metal GEMV here \
                 (PQ2_0, PTQ1_0, TQ2_0_g128, Q2_0_g64, Q1_0_g128 and F32 do)",
                layer_label(layer),
                format_name(other)
            )))
        }
    };
    Ok(Qwen35Matrix {
        rows: projection.out_features(),
        cols: projection.in_features(),
        data,
    })
}

fn full_norms<'b>(b: &'b FullAttnBlock<'_>) -> [(&'static str, &'b RmsNorm); 4] {
    [
        ("attn_norm", b.attn_norm()),
        ("post_attention_norm", b.post_attn_norm()),
        ("attn_q_norm", b.attn_q_norm()),
        ("attn_k_norm", b.attn_k_norm()),
    ]
}

fn linear_norms<'b>(b: &'b LinearAttnBlock<'_>) -> [(&'static str, &'b RmsNorm); 3] {
    [
        ("attn_norm", b.attn_norm()),
        ("post_attention_norm", b.post_attn_norm()),
        ("ssm_norm", b.ssm_norm()),
    ]
}

fn full_matrices<'b, 'a>(b: &'b FullAttnBlock<'a>) -> [(&'static str, &'b LinearLayer<'a>); 7] {
    [
        ("attn_q", b.attn_q()),
        ("attn_k", b.attn_k()),
        ("attn_v", b.attn_v()),
        ("attn_output", b.attn_output()),
        ("ffn_gate", b.ffn_gate()),
        ("ffn_up", b.ffn_up()),
        ("ffn_down", b.ffn_down()),
    ]
}

fn linear_matrices<'b, 'a>(b: &'b LinearAttnBlock<'a>) -> [(&'static str, &'b LinearLayer<'a>); 6] {
    [
        ("attn_qkv", b.attn_qkv()),
        ("attn_gate", b.attn_gate()),
        ("ssm_out", b.ssm_out()),
        ("ffn_gate", b.ffn_gate()),
        ("ffn_up", b.ffn_up()),
        ("ffn_down", b.ffn_down()),
    ]
}

fn full_weights<'w>(b: &'w FullAttnBlock<'_>) -> ModelResult<Qwen35FullAttentionWeights<'w>> {
    let layer = b.layer_idx();
    Ok(Qwen35FullAttentionWeights {
        attn_norm: b.attn_norm().weight(),
        post_attention_norm: b.post_attn_norm().weight(),
        attn_q: gpu_matrix(layer, "attn_q", b.attn_q())?,
        attn_k: gpu_matrix(layer, "attn_k", b.attn_k())?,
        attn_v: gpu_matrix(layer, "attn_v", b.attn_v())?,
        attn_output: gpu_matrix(layer, "attn_output", b.attn_output())?,
        attn_q_norm: b.attn_q_norm().weight(),
        attn_k_norm: b.attn_k_norm().weight(),
        ffn_gate: gpu_matrix(layer, "ffn_gate", b.ffn_gate())?,
        ffn_up: gpu_matrix(layer, "ffn_up", b.ffn_up())?,
        ffn_down: gpu_matrix(layer, "ffn_down", b.ffn_down())?,
    })
}

fn linear_weights<'w>(
    b: &'w LinearAttnBlock<'_>,
    alpha_beta: &'w [f32],
) -> ModelResult<Qwen35LinearAttentionWeights<'w>> {
    let layer = b.layer_idx();
    Ok(Qwen35LinearAttentionWeights {
        attn_norm: b.attn_norm().weight(),
        post_attention_norm: b.post_attn_norm().weight(),
        attn_qkv: gpu_matrix(layer, "attn_qkv", b.attn_qkv())?,
        attn_gate: gpu_matrix(layer, "attn_gate", b.attn_gate())?,
        ssm_alpha_beta: alpha_beta,
        ssm_conv1d: b.ssm_conv1d(),
        a_neg: b.gates().a_neg(),
        dt_bias: b.gates().dt_bias(),
        ssm_norm: b.ssm_norm().weight(),
        ssm_out: gpu_matrix(layer, "ssm_out", b.ssm_out())?,
        ffn_gate: gpu_matrix(layer, "ffn_gate", b.ffn_gate())?,
        ffn_up: gpu_matrix(layer, "ffn_up", b.ffn_up())?,
        ffn_down: gpu_matrix(layer, "ffn_down", b.ffn_down())?,
    })
}

/// `ssm_alpha` rows then `ssm_beta` rows, widened to `f32` in their stored
/// (tiled) order: `[2 * n_v_heads][hidden]`. Shapes were checked by
/// [`HybridMetalRunner::check_supported`].
fn widen_gates(b: &LinearAttnBlock<'_>, cfg: &Qwen35GpuConfig) -> Vec<f32> {
    let (rows, cols) = (cfg.n_v_heads, cfg.hidden);
    let mut out = Vec::with_capacity(2 * rows * cols);
    for gate in [b.ssm_alpha(), b.ssm_beta()] {
        for row in 0..rows {
            out.extend((0..cols).map(|col| gate.at(row, col)));
        }
    }
    out
}

/// A second handle on the same embedding table, never a copy of it: the
/// quantized variants re-borrow the mapped blocks, and the unquantized
/// variant clones its shared `Arc<[f32]>` handle.
fn rebind_embedding<'a>(embedding: &HybridEmbedding<'a>) -> HybridEmbedding<'a> {
    match embedding {
        HybridEmbedding::Pq2_0(blocks) => HybridEmbedding::Pq2_0(blocks),
        HybridEmbedding::Ptq1_0(blocks) => HybridEmbedding::Ptq1_0(blocks),
        HybridEmbedding::Q2_0G64(blocks) => HybridEmbedding::Q2_0G64(blocks),
        HybridEmbedding::Ternary(blocks) => HybridEmbedding::Ternary(blocks),
        HybridEmbedding::OneBit(blocks) => HybridEmbedding::OneBit(blocks),
        HybridEmbedding::Dense(table) => HybridEmbedding::Dense(table.clone()),
    }
}

#[path = "metal_state.rs"]
mod state;

pub use state::{HybridGpuSnapshot, HybridMetalFootprint};

#[cfg(test)]
#[path = "metal_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "metal_rows_tests.rs"]
mod rows_tests;
