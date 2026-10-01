//! Metal GPU forward-pass methods for `BonsaiModel`.
//!
//! # One namespace per GGUF mapping (`MET-02`)
//!
//! Every buffer a fused Metal path keys on a model-owned slot — the Q1
//! RMSNorm weights, final norm and LM head, and every ternary weight — is
//! keyed `WeightKey::new(epoch, kind, slot)` under the **mapping epoch** held
//! by [`Q1MetalSlots`]: one epoch per GGUF mapping, shared by every
//! engine-pool replica of it and by every block of those replicas, distinct
//! across mappings (see [`super::q1_slots`]). So:
//!
//! - two *different* models in one process can never be served each other's
//!   norms or LM head;
//! - N replicas of one model hold those buffers **once**;
//! - within one model, the prefill, the fused decode, the cached greedy
//!   builder (`gpu_cache.rs`) and the per-layer block path all resolve the
//!   same slot to the same buffer;
//! - the whole set leaves the GPU in one `MetalGraph::release_model(epoch)`
//!   when the last replica drops.
//!
//! The Q1 **projections** are the exception on purpose: they are keyed by
//! the scirs2 upload-handle ids, which the backend deduplicates across
//! replicas and evicts itself when their last replica releases them.
//!
//! # Ternary routing (`MET-03`)
//!
//! Every ternary decode / prefill / verify path the model takes dispatches
//! through the **cached** entry points of [`super::gpu_cache`] (eight GPU
//! handles per layer resolved once). The uncached binding survives as the
//! reference those paths are proven bit-identical against
//! ([`BonsaiModel::forward_logits_gpu_ternary_uncached`],
//! [`BonsaiModel::prefill_logits_gpu_ternary_uncached`],
//! [`BonsaiModel::prefill_verify_gpu_ternary_uncached`]) and as the only
//! route for a ternary body under a non-ternary LM head, whose cache has no
//! tail.
//!
//! # Autorelease pools
//!
//! Every forward this module sends to the Metal kernels — the fused decode
//! and greedy decode, the fused and verify batch prefills, the per-layer
//! fallbacks and the uncached references — runs inside its own Objective-C
//! autorelease pool
//! ([`oxibonsai_kernels::gpu_backend::with_autorelease_pool`]). A forward's
//! command buffers and encoders are autoreleased objects, and a decode
//! thread without a pool would keep ~1.8 KiB of them per token until it
//! exits. The pool around each call releases them as soon as the call
//! returns, whichever entry point it reaches and whether or not that entry
//! point drains a pool of its own.

use super::q1_slots::{MappingRegistration, MappingState, SlotNamespace};
use super::{BonsaiModel, OutputWeight};
use crate::block::blocks_as_bytes;
use oxibonsai_kernels::gpu_backend::metal_full_layer::types::{next_model_epoch, WeightKind};
use oxibonsai_kernels::gpu_backend::{
    try_metal_full_forward_prefill_q1_cached, with_autorelease_pool, MetalPrefillPolicy,
    PrefillCostSnapshot, PrefillRoute,
};
use oxibonsai_kernels::{FullForwardLayerParams, GpuWeightHandle, MetalGraph, MetalGraphError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// Convenience alias for the boxed error every Metal entry point here returns.
type GpuResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Whether `e` is a batched-prefill wait that missed its deadline (M-18) —
/// the typed timeout of `MetalGraphError::is_command_buffer_timeout`, however
/// deep in the Metal entry points it was boxed.
pub(super) fn is_prefill_timeout(e: &(dyn std::error::Error + 'static)) -> bool {
    e.downcast_ref::<MetalGraphError>()
        .is_some_and(MetalGraphError::is_command_buffer_timeout)
}

/// Release every buffer keyed under `epoch` from the Metal weight cache —
/// the [`super::q1_slots::ReleaseHook`] of the Metal build, run once when the
/// last holder of a mapping that touched the GPU drops.
///
/// **Never opens a device.** It releases through the session bound to this
/// thread when there is one (a model used inside a private or isolated
/// session is dropped inside it), else through the process-default session
/// **iff** the shared device already has live sessions. When neither holds,
/// no Metal state exists in this process on the shared device, so nothing can
/// be resident under the epoch there.
pub(crate) fn release_metal_mapping(epoch: u64) {
    // The mapping's M-18 prefill cost model leaves with it.
    MetalPrefillPolicy::forget(epoch);
    let graph = match MetalGraph::current_session() {
        Some(session) => session,
        None if MetalGraph::live_session_count() > 0 => match MetalGraph::global() {
            Ok(graph) => graph,
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    epoch,
                    "could not reach the Metal graph to release a dropped mapping's buffers"
                );
                return;
            }
        },
        None => return,
    };
    match graph.release_model(epoch) {
        Ok(0) => {}
        Ok(released) => tracing::debug!(
            epoch,
            released,
            "released the last replica's Metal weight buffers"
        ),
        Err(e) => tracing::debug!(
            error = %e,
            epoch,
            "releasing a dropped mapping's Metal weight buffers failed"
        ),
    }
}

/// The Metal weight-cache namespace of one model's GGUF **mapping**
/// (`MET-02`): the Q1 norm / final-norm / LM-head slots and the epoch every
/// model-owned Metal buffer (Q1 and ternary) is keyed under.
///
/// The kernels' Q1 entry points take the norm and LM-head slots as `u64`s
/// composed `TAG | epoch << 24 | local` (the historical locals —
/// `1_000_000 + layer * 10 + k`, `2_000_000`, `3_000_000` — namespaced by the
/// epoch; see `q1_slots::SlotNamespace`) and key each upload
/// `WeightKey::new(epoch, kind, slot)`.
///
/// # Sharing and lifetime
///
/// - `Q1MetalSlots::for_mapping` — what every loaded model uses — **joins** the
///   namespace of the mapping its weights are borrowed from: every
///   engine-pool replica of one GGUF gets the same epoch and therefore the
///   same buffers, and a different mapping always gets a different epoch.
/// - [`Self::fresh`] / [`Self::with_epoch`] build a **private** namespace
///   nobody else can join (a weight-less model; a test control that aims a
///   model at another's epoch on purpose).
///
/// The buffers are released with `MetalGraph::release_model(epoch)` when the
/// **last** holder of the namespace drops — the last replica of a shared
/// mapping (dropping any earlier replica leaves them resident for its
/// siblings), or the sole owner of a private one — and only if some path put
/// buffers under it. [`Self::release`] releases them on demand, for every
/// holder at once: a surviving replica simply re-uploads on its next miss.
/// Evicting only removes the cache entries; a dispatch already encoded keeps
/// its buffers alive through its own `Arc`s.
#[derive(Debug)]
pub struct Q1MetalSlots {
    registration: MappingRegistration,
}

impl Q1MetalSlots {
    /// A private namespace under a new, never-reused epoch.
    #[must_use]
    pub fn fresh() -> Self {
        Self::with_epoch(next_model_epoch())
    }

    /// A private namespace under `epoch`: it addresses exactly the buffers
    /// every other holder of that epoch addresses, but joins no mapping —
    /// its own release (or its last holder's drop) evicts the epoch's buffers
    /// for all of them. Meant for a caller that already holds an epoch.
    #[must_use]
    pub fn with_epoch(epoch: u64) -> Self {
        Self {
            registration: MappingRegistration::private(epoch, Some(release_metal_mapping)),
        }
    }

    /// Join the namespace of the mapping anchored at `anchor` (shared with
    /// every live replica of that mapping), or take a private one for `None`.
    #[must_use]
    pub(crate) fn for_mapping(anchor: Option<u64>) -> Self {
        Self {
            registration: MappingRegistration::join(
                anchor,
                next_model_epoch,
                Some(release_metal_mapping),
            ),
        }
    }

    /// The epoch these slots — and every model-owned Metal buffer of the
    /// mapping — are keyed under.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.registration.epoch()
    }

    /// The pure slot composition of [`Self::epoch`].
    fn slots(&self) -> SlotNamespace {
        self.registration.slots()
    }

    /// Base of layer `layer`'s four norm slots (`+0` attn, `+1` q, `+2` k,
    /// `+3` ffn).
    #[must_use]
    pub fn norm_base(&self, layer: usize) -> u64 {
        self.slots().norm_base(layer)
    }

    /// Slot of the final `output_norm`.
    #[must_use]
    pub fn final_norm(&self) -> u64 {
        self.slots().final_norm()
    }

    /// Slot of the 1-bit LM head.
    #[must_use]
    pub fn lm_head(&self) -> u64 {
        self.slots().lm_head()
    }

    /// Record that a fused path is about to hand (or has handed) these slots
    /// to the kernels for an `n_layers`-layer model. Every path calls it
    /// before **and** after its dispatch, so an explicit [`Self::release`]
    /// racing a sibling replica's upload can never leave a resident buffer
    /// behind with the in-use flag cleared.
    pub(super) fn mark_used(&self, n_layers: usize) {
        debug_assert!(
            SlotNamespace::check_layer_count(n_layers).is_ok(),
            "layer counts are checked before any slot is derived"
        );
        self.registration.state().mark_gpu_used();
    }

    /// Whether a fused path has put buffers under this namespace since it was
    /// created or last released — by this model or by a sibling replica of
    /// the same mapping.
    #[must_use]
    pub fn is_in_use(&self) -> bool {
        self.registration.state().is_gpu_used()
    }

    /// Whether this namespace is a GGUF mapping's shared one (as opposed to
    /// a private namespace).
    #[must_use]
    pub fn is_shared_mapping(&self) -> bool {
        self.registration.is_shared_mapping()
    }

    /// Live models holding this namespace's mapping (1 for a private one).
    #[must_use]
    pub fn replicas(&self) -> usize {
        self.registration.replicas()
    }

    /// The shared mapping state (cloned into the model's blocks so the
    /// per-layer path keys on the same epoch).
    pub(crate) fn state(&self) -> &Arc<MappingState> {
        self.registration.state()
    }

    /// Every `(slot, kind)` the Q1 fused paths can populate for an
    /// `n_layers`-layer model: four norms per layer, the final norm and the
    /// LM head.
    #[must_use]
    pub fn cache_keys(&self, n_layers: usize) -> Vec<(u64, WeightKind)> {
        self.slots()
            .keys(n_layers)
            .into_iter()
            .map(|(slot, cache)| {
                let kind = match cache {
                    super::q1_slots::SlotCache::Norm => WeightKind::RawF32,
                    super::q1_slots::SlotCache::Quant => WeightKind::Q1Soa,
                };
                (slot, kind)
            })
            .collect()
    }

    /// Evict every buffer keyed under this namespace's epoch from the Metal
    /// weight cache of the session this thread dispatches into
    /// ([`MetalGraph::global`]), returning how many were released — `Ok(0)`,
    /// without touching the Metal graph, when no path has put buffers under
    /// it since it was created or last released.
    ///
    /// The namespace is shared by every replica of the mapping, so this
    /// evicts **their** buffers too; a surviving replica re-uploads on its
    /// next miss (and a replica's cached handle set keeps its old buffers
    /// alive until it is dropped or rebuilt).
    ///
    /// # Errors
    ///
    /// The Metal graph cannot be reached, or its cache lock is poisoned; the
    /// namespace then stays marked in use, so a later call (or the last
    /// holder's drop) releases it again.
    pub fn release(&self) -> Result<usize, oxibonsai_kernels::MetalGraphError> {
        let state = self.registration.state();
        if !state.take_gpu_used() {
            return Ok(0);
        }
        let released = MetalGraph::global().and_then(|graph| graph.release_model(state.epoch()));
        if released.is_err() {
            state.mark_gpu_used();
        }
        released
    }
}

/// Test-only override for [`force_ternary_tail_failure`].
static FORCE_TERNARY_TAIL_FAIL: AtomicBool = AtomicBool::new(false);

/// `OXIBONSAI_FORCE_METAL_TAIL_FAIL` as read once at first use.
///
/// Read through a `OnceLock` rather than per call so the decode hot path never
/// pays for a `getenv`.
static FORCE_TERNARY_TAIL_FAIL_ENV: OnceLock<bool> = OnceLock::new();

/// Whether the ternary fused forward must fail at its final-norm → LM-head
/// tail, after every layer's weights are already GPU-resident.
///
/// This is the MET-02 fault-injection seam: the double-upload it guards against
/// is reachable through *any* deterministic tail failure (a missing TQ2 LM-head
/// pipeline, an `lm_head_out_features` mismatch, a logits-buffer allocation
/// failure, a GPU fault surfaced by MET-04), none of which can be provoked on
/// healthy hardware. Set `OXIBONSAI_FORCE_METAL_TAIL_FAIL` to reproduce the
/// state that used to send `BonsaiModel::forward()` into a second handle
/// namespace and duplicate the whole model on the GPU.
fn force_ternary_tail_failure() -> bool {
    *FORCE_TERNARY_TAIL_FAIL_ENV
        .get_or_init(|| std::env::var_os("OXIBONSAI_FORCE_METAL_TAIL_FAIL").is_some())
        || FORCE_TERNARY_TAIL_FAIL.load(Ordering::Relaxed)
}

/// Turn the [`force_ternary_tail_failure`] seam on or off from a test.
///
/// Process-global, like the `MetalGraph` singleton it is used against: callers
/// must hold the GPU test lock and clear the flag before releasing it.
#[cfg(test)]
pub(super) fn set_force_ternary_tail_failure(on: bool) {
    FORCE_TERNARY_TAIL_FAIL.store(on, Ordering::Relaxed);
}

/// The id of a block's upload handle, or a named error.
fn handle_id(handle: Option<GpuWeightHandle>, layer: usize, what: &str) -> GpuResult<u64> {
    handle
        .map(|hnd| hnd.id())
        .ok_or_else(|| format!("missing GPU handle for layer {layer} {what}").into())
}

impl<'a> BonsaiModel<'a> {
    /// The epoch every model-owned Metal buffer of this model's GGUF mapping
    /// is keyed under — shared by every replica of the mapping, distinct
    /// across mappings (`MET-02`).
    #[must_use]
    pub fn gpu_mapping_epoch(&self) -> u64 {
        self.metal_q1_slots.epoch()
    }

    /// Refuse the Q1 fused paths unless every block holds its four GPU
    /// upload handles — checked **before** anything is concatenated, so a
    /// ternary (or never-uploaded) model falls through for the price of a
    /// few `Option` checks.
    fn require_q1_gpu_handles(&self) -> GpuResult<()> {
        for block in &self.blocks {
            if block.fused_qkv_gpu_handle().is_none()
                || block.attn_output_gpu_handle().is_none()
                || block.fused_gate_up_gpu_handle().is_none()
                || block.ffn_down_gpu_handle().is_none()
            {
                return Err("missing GPU handle".into());
            }
        }
        Ok(())
    }

    /// Every layer's Q‖K‖V concatenation — the one Q1 layout that cannot be
    /// borrowed from the mapping.
    pub(super) fn q1_qkv_concats(&self) -> GpuResult<Vec<Vec<u8>>> {
        let mut qkv_concats: Vec<Vec<u8>> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let q_bytes =
                blocks_as_bytes(block.attn_q_blocks().ok_or("attn_q: not a 1-bit layer")?);
            let k_bytes =
                blocks_as_bytes(block.attn_k_blocks().ok_or("attn_k: not a 1-bit layer")?);
            let v_bytes =
                blocks_as_bytes(block.attn_v_blocks().ok_or("attn_v: not a 1-bit layer")?);
            let mut concat = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            concat.extend_from_slice(q_bytes);
            concat.extend_from_slice(k_bytes);
            concat.extend_from_slice(v_bytes);
            qkv_concats.push(concat);
        }
        Ok(qkv_concats)
    }

    /// Per-layer Q1 dispatch parameters — the **single** place every Q1
    /// fused path (decode, decode + LM head, prefill, prefill-verify, and the
    /// cached greedy builder in `gpu_cache.rs`) derives its slots from, so
    /// they agree by construction: the norms from this model's mapping
    /// namespace (under its epoch), the projections from the blocks' upload
    /// handles (`MET-02`).
    ///
    /// Marks the namespace in use: the caller dispatches next.
    pub(super) fn q1_layer_params<'b>(
        &'b self,
        qkv_concats: &'b [Vec<u8>],
    ) -> GpuResult<Vec<FullForwardLayerParams<'b>>> {
        SlotNamespace::check_layer_count(self.blocks.len())?;
        let epoch = self.metal_q1_slots.epoch();
        let mut layer_params: Vec<FullForwardLayerParams<'b>> =
            Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let layer = block.layer_index();
            let norm_handle_base = self.metal_q1_slots.norm_base(layer);
            layer_params.push(FullForwardLayerParams {
                model_epoch: epoch,
                attn_norm_handle: norm_handle_base,
                attn_norm_bytes: block.attn_norm_weight(),
                fused_qkv_handle: handle_id(block.fused_qkv_gpu_handle(), layer, "fused_qkv")?,
                fused_qkv_bytes: qkv_concats
                    .get(i)
                    .ok_or("fewer Q‖K‖V concatenations than layers")?,
                q_norm_handle: norm_handle_base + 1,
                q_norm_bytes: block.q_norm_weight(),
                k_norm_handle: norm_handle_base + 2,
                k_norm_bytes: block.k_norm_weight(),
                attn_proj_handle: handle_id(block.attn_output_gpu_handle(), layer, "attn_proj")?,
                attn_proj_bytes: blocks_as_bytes(
                    block
                        .attn_output_blocks()
                        .ok_or("attn_output: not a 1-bit layer")?,
                ),
                ffn_norm_handle: norm_handle_base + 3,
                ffn_norm_bytes: block.ffn_norm_weight(),
                gate_up_handle: handle_id(block.fused_gate_up_gpu_handle(), layer, "gate_up")?,
                gate_bytes: blocks_as_bytes(
                    block
                        .ffn_gate_blocks()
                        .ok_or("ffn_gate: not a 1-bit layer")?,
                ),
                up_bytes: blocks_as_bytes(
                    block.ffn_up_blocks().ok_or("ffn_up: not a 1-bit layer")?,
                ),
                down_handle: handle_id(block.ffn_down_gpu_handle(), layer, "down")?,
                down_bytes: blocks_as_bytes(
                    block
                        .ffn_down_blocks()
                        .ok_or("ffn_down: not a 1-bit layer")?,
                ),
            });
        }
        self.metal_q1_slots.mark_used(self.blocks.len());
        Ok(layer_params)
    }

    /// Embed a batch column-major and build its per-position RoPE tables.
    fn q1_prefill_inputs(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> GpuResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let batch_size = token_ids.len();
        let h = self.config.hidden_size;
        let half_dim = self.config.head_dim / 2;
        let mut hidden_batch = vec![0.0f32; batch_size * h];
        // M-02: gather only the rows this batch references, decoding each
        // straight out of the quantized table. The deleted dense
        // `Index<Range<usize>>` hatch materialized the whole
        // `vocab x hidden` FP32 table here (1.16 / 2.31 / 4.74 GiB for the
        // 1.7B / 8B / 27B) on the first multi-token prompt.
        self.token_embd.copy_rows(token_ids, &mut hidden_batch)?;
        let mut cos_table = vec![0.0f32; batch_size * half_dim];
        let mut sin_table = vec![0.0f32; batch_size * half_dim];
        for t in 0..batch_size {
            let pos = pos_start + t;
            cos_table[t * half_dim..(t + 1) * half_dim]
                .copy_from_slice(self.rope.cos_at_checked(pos)?);
            sin_table[t * half_dim..(t + 1) * half_dim]
                .copy_from_slice(self.rope.sin_at_checked(pos)?);
        }
        Ok((hidden_batch, cos_table, sin_table))
    }

    /// Attempt to run all transformer layers in a single Metal command buffer.
    ///
    /// On success, `hidden` is updated in-place through all layers. The GPU
    /// manages its own KV cache. Returns `Err` if any precondition is not
    /// met or the dispatch fails.
    pub(super) fn try_metal_full_forward_inner(
        &self,
        hidden: &mut [f32],
        pos: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        self.require_q1_gpu_handles()?;
        let qkv_concats = self.q1_qkv_concats()?;
        let layer_params = self.q1_layer_params(&qkv_concats)?;
        let rope_cos = self.rope.cos_at_checked(pos)?;
        let rope_sin = self.rope.sin_at_checked(pos)?;
        let result = with_autorelease_pool(|| {
            oxibonsai_kernels::try_metal_full_forward(
                hidden,
                pos,
                n_layers,
                &layer_params,
                rope_cos,
                rope_sin,
                h,
                inter,
                nq,
                nkv,
                hd,
                eps,
                max_seq_len,
                None,
                None,
                eps,
                None,
                None,
                0,
                None,
                None,
            )
        });
        self.metal_q1_slots.mark_used(n_layers);
        result.map_err(|e| {
            tracing::warn!(
                error = % e, "full-forward GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// Attempt to run every transformer layer on Metal for a ternary
    /// (TQ2_0_g128) model, encoding all layers into a single command buffer.
    ///
    /// Mirrors [`Self::try_metal_full_forward_inner`] for the TQ2 GEMV kernel.
    /// Returns `Err` if any block is not ternary or the Metal dispatch fails —
    /// in which case the caller falls back to the CPU per-layer path.
    ///
    /// This is the path `BonsaiModel::forward()` drops into when the fused
    /// final-norm → LM-head route fails. It used to carry its own weight-handle
    /// namespace, so that fallback uploaded a **second full copy** of the
    /// model (MET-02). With a ternary LM head it now binds the model's
    /// **cached** weight set (`MET-03`) — the very buffers the failed route
    /// made resident — and runs no lookup at all; a ternary body under a
    /// non-ternary head has no cached tail, so it keeps the uncached binding,
    /// which resolves to the same mapping-epoch slots.
    pub(super) fn try_metal_full_forward_ternary_inner(
        &self,
        hidden: &mut [f32],
        pos: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        let rope_cos = self.rope.cos_at_checked(pos)?;
        let rope_sin = self.rope.sin_at_checked(pos)?;
        let result = if matches!(self.output_weight, OutputWeight::Ternary(_)) {
            self.with_ternary_gpu_cache(|cached| {
                with_autorelease_pool(|| {
                    oxibonsai_kernels::try_metal_full_forward_ternary_cached(
                        hidden,
                        pos,
                        cached,
                        rope_cos,
                        rope_sin,
                        h,
                        inter,
                        nq,
                        nkv,
                        hd,
                        eps,
                        max_seq_len,
                        eps,
                        None,
                        None,
                    )
                })
            })?
        } else {
            let binding = self.ternary_gpu_binding()?;
            with_autorelease_pool(|| {
                oxibonsai_kernels::try_metal_full_forward_ternary(
                    hidden,
                    pos,
                    n_layers,
                    &binding.layer_params,
                    rope_cos,
                    rope_sin,
                    h,
                    inter,
                    nq,
                    nkv,
                    hd,
                    eps,
                    max_seq_len,
                    None,
                    None,
                    eps,
                    None,
                    None,
                    0,
                    None,
                    None,
                )
            })
        };
        result.map_err(|e| {
            tracing::warn!(
                error = % e, "ternary full-forward GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// Attempt to run all transformer layers + final RMSNorm + LM head GEMV
    /// in a single Metal command buffer.
    ///
    /// On success, `logits` is filled with the output logits and `hidden` is
    /// NOT updated (the GPU handles everything end-to-end). Returns `Err` if
    /// any precondition is not met (missing GPU handles, FP32 LM head, etc.).
    ///
    /// Ternary models are delegated to the cached ternary route.
    ///
    /// Every successful step is timed into the model's M-18 prefill cost
    /// model: this is the decode `forward` runs per token on the GPU route,
    /// i.e. the sequential path a fused prefill is compared against.
    pub(super) fn try_metal_full_forward_with_lm_head(
        &self,
        hidden: &mut [f32],
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let started = Instant::now();
        self.try_metal_full_forward_with_lm_head_untimed(hidden, pos, logits)?;
        MetalPrefillPolicy::record_decode(self.gpu_mapping_epoch(), started.elapsed());
        Ok(())
    }

    /// Body of [`Self::try_metal_full_forward_with_lm_head`].
    fn try_metal_full_forward_with_lm_head_untimed(
        &self,
        hidden: &mut [f32],
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }

        // ── Ternary path ──────────────────────────────────────────────────────
        if matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return self.try_metal_full_forward_with_lm_head_ternary(hidden, pos, logits);
        }

        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head reached the Q1 fused GPU path".into())
            }
            OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_) => {
                return Err("FP8 GPU inference not yet supported; use CPU path".into());
            }
            OutputWeight::Q4_0(_)
            | OutputWeight::Q8_0(_)
            | OutputWeight::Q5K(_)
            | OutputWeight::Q6K(_)
            | OutputWeight::Q2K(_)
            | OutputWeight::Q3K(_)
            | OutputWeight::Q4K(_)
            | OutputWeight::Q8K(_) => {
                return Err(
                    "K-quant / Q-type LM head not yet supported on fused Metal path; use CPU path"
                        .into(),
                );
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on fused GPU path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        self.require_q1_gpu_handles()?;
        let qkv_concats = self.q1_qkv_concats()?;
        let layer_params = self.q1_layer_params(&qkv_concats)?;
        let rope_cos = self.rope.cos_at_checked(pos)?;
        let rope_sin = self.rope.sin_at_checked(pos)?;
        let final_norm_handle = self.metal_q1_slots.final_norm();
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = self.metal_q1_slots.lm_head();
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let result = with_autorelease_pool(|| {
            oxibonsai_kernels::try_metal_full_forward(
                hidden,
                pos,
                n_layers,
                &layer_params,
                rope_cos,
                rope_sin,
                h,
                inter,
                nq,
                nkv,
                hd,
                eps,
                max_seq_len,
                Some(final_norm_handle),
                Some(final_norm_bytes),
                final_norm_eps,
                Some(lm_head_handle),
                Some(lm_head_bytes),
                lm_head_out_features,
                Some(logits),
                None,
            )
        });
        self.metal_q1_slots.mark_used(n_layers);
        result.map_err(|e| {
            tracing::warn!(
                error = % e, "full-forward+lm_head GPU dispatch failed, falling back"
            );
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// GPU batch prefill: all layers + final norm + LM head, behind the M-18
    /// guard. This is the call `forward_prefill` makes on the Metal route.
    ///
    /// # Route
    ///
    /// The model's measured prefill cost model (`MetalPrefillPolicy`, keyed by
    /// the mapping epoch) picks the route: the fused batch prefill
    /// ([`Self::try_metal_prefill_with_lm_head_fused`]) unless fused calls of a
    /// comparable size were measured slower per token than the single-token
    /// decode, in which case the prompt runs through
    /// [`Self::metal_prefill_sequential`] — exactly the per-token fused
    /// decode `forward` performs, so the logits, the device KV cache and the
    /// position are those of a sequential prefill. A model with no
    /// measurement yet always takes the fused path.
    ///
    /// # Deadline
    ///
    /// The fused route runs under a deadline of
    /// [`oxibonsai_kernels::gpu_backend::FUSED_BUDGET_FACTOR`] times the
    /// predicted sequential time. A fused prefill that misses it stops
    /// committing work (at most one micro-batch is still running on the GPU),
    /// logs one warning, records the timeout against the cost model and
    /// prefills the prompt sequentially instead — so no prompt waits on the GPU
    /// without a bound, and the next prompt of that size goes straight to the
    /// cheaper route.
    pub fn try_metal_prefill_with_lm_head(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let epoch = self.gpu_mapping_epoch();
        let tokens = token_ids.len();
        let decision = MetalPrefillPolicy::decide(epoch, tokens, self.decode_prior_s_per_token());
        if decision.route == PrefillRoute::Sequential {
            tracing::debug!(
                tokens,
                pos_start,
                predicted_fused_s = ?decision.predicted_fused_s,
                predicted_sequential_s = decision.predicted_sequential_s,
                "metal prefill: the cost model routes this prompt through sequential decode"
            );
            return self.metal_prefill_sequential(token_ids, pos_start);
        }
        let started = Instant::now();
        let fused = {
            let _deadline = MetalGraph::prefill_deadline_scope(started + decision.fused_budget);
            self.try_metal_prefill_with_lm_head_fused(token_ids, pos_start)
        };
        match fused {
            Ok(logits) => {
                MetalPrefillPolicy::record_fused(epoch, tokens, started.elapsed());
                Ok(logits)
            }
            Err(e) if is_prefill_timeout(e.as_ref()) => {
                let waited = started.elapsed();
                MetalPrefillPolicy::record_fused_timeout(epoch, tokens, waited);
                tracing::warn!(
                    tokens,
                    pos_start,
                    waited_ms = waited.as_millis() as u64,
                    budget_ms = decision.fused_budget.as_millis() as u64,
                    error = %e,
                    "fused Metal prefill missed its deadline; prefilling the prompt through \
                     sequential decode instead"
                );
                self.metal_prefill_sequential(token_ids, pos_start)
            }
            Err(e) => Err(e),
        }
    }

    /// The fused Metal batch prefill, **strictly**: no route decision, no
    /// fallback — a dispatch failure or a missed deadline comes back as the
    /// error (a timeout satisfies
    /// `MetalGraphError::is_command_buffer_timeout`).
    ///
    /// Both 1-bit and ternary (TQ2_0_g128) LM-head models are supported; the
    /// ternary one runs the cached ternary prefill, and a 1-bit model whose
    /// fused weight cache is resident binds it instead of rebuilding its
    /// Q‖K‖V concatenation per call (perf-03).
    ///
    /// Marked `pub` so parity tests can invoke the fused path directly,
    /// bypassing both the M-18 router and the silent fallback in
    /// [`Self::forward_prefill`] that masks GPU dispatch failures.
    pub fn try_metal_prefill_with_lm_head_fused(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        // Context-length guard (mirrors the single-token `forward()` check at
        // model/types/mod.rs and the CUDA prefill guards in forward_cuda/*).
        // The batched RoPE gather below reads `self.rope.cos_at_checked(pos_start
        // + t)`, and `RopeTable` is sized to exactly `max_seq_len` rows. The
        // checked accessor is the backstop (it returns
        // `ModelError::PositionOutOfRange` where the old unchecked slice
        // panicked); this guard is what produces the message callers match on and
        // what makes `forward_prefill` fall back to the sequential path, whose
        // per-token `forward()` returns a clean `ModelError::SequenceTooLong`.
        if pos_start + batch_size > self.kv_cache.max_seq_len() {
            return Err(format!(
                "prefill sequence too long: {batch_size} tokens at pos {pos_start} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                // Ternary batch prefill (TQ2_0_g128 weights end-to-end).
                return self.try_metal_prefill_with_lm_head_ternary(token_ids, pos_start);
            }
            OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_) => {
                return Err("FP8 GPU inference not yet supported; use CPU path".into());
            }
            OutputWeight::Q4_0(_)
            | OutputWeight::Q8_0(_)
            | OutputWeight::Q5K(_)
            | OutputWeight::Q6K(_)
            | OutputWeight::Q2K(_)
            | OutputWeight::Q3K(_)
            | OutputWeight::Q4K(_)
            | OutputWeight::Q8K(_) => {
                return Err(
                    "K-quant / Q-type LM head not yet supported on Metal prefill path; use CPU path"
                        .into(),
                );
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on GPU prefill path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        self.require_q1_gpu_handles()?;
        let (hidden_batch, cos_table, sin_table) = self.q1_prefill_inputs(token_ids, pos_start)?;
        let final_norm_eps = self.output_norm.eps();
        let lm_head_out_features = lm_head_linear.out_features();
        let mut logits = vec![0.0f32; lm_head_out_features];
        let cached_result = {
            let guard = self
                .gpu_weight_cache
                .lock()
                .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
            match guard.as_ref() {
                Some(cached @ oxibonsai_kernels::CachedModelWeights::Q1(_)) => {
                    Some(with_autorelease_pool(|| {
                        try_metal_full_forward_prefill_q1_cached(
                            &hidden_batch,
                            batch_size,
                            pos_start,
                            cached,
                            &cos_table,
                            &sin_table,
                            h,
                            inter,
                            nq,
                            nkv,
                            hd,
                            eps,
                            max_seq_len,
                            final_norm_eps,
                            lm_head_out_features,
                            Some(&mut logits),
                            None,
                        )
                    }))
                }
                _ => None,
            }
        };
        let result = match cached_result {
            Some(result) => result,
            None => {
                let qkv_concats = self.q1_qkv_concats()?;
                let layer_params = self.q1_layer_params(&qkv_concats)?;
                let final_norm_handle = self.metal_q1_slots.final_norm();
                let final_norm_bytes = self.output_norm.weight();
                let lm_head_handle = self.metal_q1_slots.lm_head();
                let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
                with_autorelease_pool(|| {
                    oxibonsai_kernels::try_metal_full_forward_prefill(
                        &hidden_batch,
                        batch_size,
                        pos_start,
                        n_layers,
                        &layer_params,
                        &cos_table,
                        &sin_table,
                        h,
                        inter,
                        nq,
                        nkv,
                        hd,
                        eps,
                        max_seq_len,
                        Some(final_norm_handle),
                        Some(final_norm_bytes),
                        final_norm_eps,
                        Some(lm_head_handle),
                        Some(lm_head_bytes),
                        lm_head_out_features,
                        Some(&mut logits),
                        None,
                    )
                })
            }
        };
        self.metal_q1_slots.mark_used(n_layers);
        result.map_err(|e| {
            // A missed deadline is reported once, by the router.
            if !e.is_command_buffer_timeout() {
                tracing::warn!(error = % e, "batch prefill GPU dispatch failed");
            }
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(logits)
    }

    /// The **sequential** Metal prefill route: every prompt token through
    /// the single-token fused decode + LM head, one command buffer each.
    ///
    /// Exactly what `forward` does per token on the GPU route — the same entry
    /// point (`try_metal_full_forward_with_lm_head`) on the same
    /// hidden state — so the returned last-position logits, the device KV
    /// cache and the MET-05 latch equal those of `forward` called token by
    /// token. It is the route the M-18 guard takes when the fused batch
    /// prefill is measured (or forced) slower, and the fallback after a fused
    /// prefill misses its deadline.
    ///
    /// # Errors
    ///
    /// An empty prompt, a position past the context, or any Metal failure of
    /// a decode step.
    pub fn metal_prefill_sequential(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if token_ids.is_empty() {
            return Err("metal_prefill_sequential: empty prompt".into());
        }
        if pos_start + token_ids.len() > self.kv_cache.max_seq_len() {
            return Err(format!(
                "prefill sequence too long: {} tokens at pos {pos_start} exceeds max_seq_len {}",
                token_ids.len(),
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        let mut hidden = vec![0.0f32; self.config.hidden_size];
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for (i, &token) in token_ids.iter().enumerate() {
            let pos = pos_start + i;
            self.token_embd.copy_row(token, &mut hidden)?;
            self.try_metal_full_forward_with_lm_head(&mut hidden, pos, &mut logits)?;
            // As `forward` does after every fused step (MET-05).
            self.note_device_kv_used();
        }
        Ok(logits)
    }

    /// The model's decode-cost prior for the M-18 router, seconds per token:
    /// a fused decode step streams every quantized weight once, so its cost
    /// is the weight bytes at a conservative 20 GB/s (Bonsai-8B: 58 ms,
    /// Ternary-Bonsai-1.7B: 27 ms against the ~45 / ~20 ms measured). It only
    /// stands in until the first real decode step is measured.
    fn decode_prior_s_per_token(&self) -> f64 {
        const PRIOR_BYTES_PER_SECOND: f64 = 20.0e9;
        let c = &self.config;
        let qkv = (c.num_attention_heads + 2 * c.num_kv_heads) * c.head_dim;
        let attn = c.num_attention_heads * c.head_dim;
        let per_layer =
            c.hidden_size * qkv + attn * c.hidden_size + 3 * c.hidden_size * c.intermediate_size;
        let weights = (per_layer * c.num_layers + c.vocab_size * c.hidden_size) as f64;
        let bytes_per_weight = match &self.output_weight {
            OutputWeight::OneBit(_) => 18.0 / 128.0,
            OutputWeight::Ternary(_) => 34.0 / 128.0,
            _ => 0.5,
        };
        weights * bytes_per_weight / PRIOR_BYTES_PER_SECOND
    }

    /// Pin this model's Metal prefill route (`None` returns it to the
    /// measured decision) — for every replica of its GGUF mapping, which share
    /// the cost model (M-18). Tests and measurement harnesses use it to take a
    /// route regardless of what the cost model has seen.
    pub fn force_metal_prefill_route(&self, route: Option<PrefillRoute>) {
        MetalPrefillPolicy::force_route(self.gpu_mapping_epoch(), route);
    }

    /// The M-18 prefill cost model of this model's mapping, for diagnostics.
    #[must_use]
    pub fn metal_prefill_cost(&self) -> PrefillCostSnapshot {
        MetalPrefillPolicy::snapshot(self.gpu_mapping_epoch(), self.decode_prior_s_per_token())
    }

    /// GPU batch prefill verify: all layers + final norm + LM head + per-position argmax.
    ///
    /// Both 1-bit and ternary (TQ2_0_g128) LM-head models are supported; the
    /// ternary one runs the cached ternary verify.
    ///
    /// Marked `pub` so parity tests can invoke this **strict** path
    /// directly, bypassing the silent fallback in
    /// [`Self::forward_prefill_verify`] that masks GPU dispatch failures.
    pub fn try_metal_prefill_verify(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        let n_layers = self.blocks.len();
        if n_layers == 0 {
            return Err("no blocks".into());
        }
        // Context-length guard: prevents an out-of-bounds RoPE slice panic on
        // prompts longer than the context (mirrors `forward()` and the CUDA
        // prefill-verify guards).
        if pos_start + batch_size > self.kv_cache.max_seq_len() {
            return Err(format!(
                "prefill-verify sequence too long: {batch_size} tokens at pos {pos_start} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        let lm_head_linear = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear,
            OutputWeight::Ternary(_) => {
                // Ternary batch prefill verify (TQ2_0_g128 weights end-to-end).
                return self.try_metal_prefill_verify_ternary_path(token_ids, pos_start);
            }
            OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_) => {
                return Err("FP8 GPU inference not yet supported; use CPU path".into());
            }
            OutputWeight::Q4_0(_)
            | OutputWeight::Q8_0(_)
            | OutputWeight::Q5K(_)
            | OutputWeight::Q6K(_)
            | OutputWeight::Q2K(_)
            | OutputWeight::Q3K(_)
            | OutputWeight::Q4K(_)
            | OutputWeight::Q8K(_) => {
                return Err(
                    "K-quant / Q-type LM head not yet supported on Metal prefill-verify path; use CPU path"
                        .into(),
                );
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on GPU prefill verify path".into());
            }
        };
        let eps = self.blocks[0].attn_norm_eps();
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        self.require_q1_gpu_handles()?;
        let (hidden_batch, cos_table, sin_table) = self.q1_prefill_inputs(token_ids, pos_start)?;
        let qkv_concats = self.q1_qkv_concats()?;
        let layer_params = self.q1_layer_params(&qkv_concats)?;
        let final_norm_handle = self.metal_q1_slots.final_norm();
        let final_norm_bytes = self.output_norm.weight();
        let final_norm_eps = self.output_norm.eps();
        let lm_head_handle = self.metal_q1_slots.lm_head();
        let lm_head_bytes = blocks_as_bytes(lm_head_linear.blocks());
        let lm_head_out_features = lm_head_linear.out_features();
        let mut batch_token_ids: Vec<u32> = Vec::with_capacity(batch_size);
        let result = with_autorelease_pool(|| {
            oxibonsai_kernels::try_metal_full_forward_prefill_verify(
                &hidden_batch,
                batch_size,
                pos_start,
                n_layers,
                &layer_params,
                &cos_table,
                &sin_table,
                h,
                inter,
                nq,
                nkv,
                hd,
                eps,
                max_seq_len,
                Some(final_norm_handle),
                Some(final_norm_bytes),
                final_norm_eps,
                Some(lm_head_handle),
                Some(lm_head_bytes),
                lm_head_out_features,
                &mut batch_token_ids,
            )
        });
        self.metal_q1_slots.mark_used(n_layers);
        result.map_err(|e| {
            tracing::warn!(error = % e, "batch prefill verify GPU dispatch failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        Ok(batch_token_ids)
    }

    /// Greedy forward pass: runs all layers + LM head + argmax entirely on GPU.
    ///
    /// Instead of downloading the full logits vector (~607KB), this performs
    /// argmax on the GPU and downloads only the resulting token ID (4 bytes).
    /// This eliminates ~607KB of GPU→CPU bandwidth per token and removes
    /// CPU-side sampling overhead for greedy (temperature=0) decoding.
    ///
    /// On the first call, all weight handles are cached in `gpu_weight_cache`
    /// (keyed exactly like the uncached fused paths, so a model that already
    /// prefilled uploads nothing more). Subsequent calls skip ALL byte
    /// concatenation, weight upload, and HashMap lookups — passing
    /// pre-cached handles directly to the GPU.
    ///
    /// Supports both Q1 (1-bit) and ternary (TQ2_0_g128) models. FP32 LM head
    /// is not supported and returns `Err`.
    ///
    /// Returns the token ID directly, or `Err` if the GPU path is not available.
    pub fn forward_greedy_gpu(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        // Context-length guard (mirrors the single-token `forward()` check at
        // model/types/mod.rs). `self.rope.cos_at_checked(pos)` below reads a
        // `RopeTable` sized to exactly `max_seq_len` rows and errors rather than
        // panicking past it; this guard returns the named Err first, which lets
        // the engine's decode loop fall back to the guarded sequential path.
        if pos >= self.kv_cache.max_seq_len() {
            return Err(format!(
                "greedy sequence too long: pos {pos} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        // M-17 gate (out-of-owned-files fix; see FIX3-MODEL wave-3.5 notes):
        // this fused-Metal entry point attends over the *full* KV cache with
        // no windowing, and is reached directly from
        // `engine_greedy::greedy_decode_token_with_fallback` rather than
        // through `BonsaiModel::forward`/`forward_into`, so the
        // `sliding_window.is_none()` gate there cannot cover it. Refuse here
        // so a windowed model falls back to the guarded CPU sequential path
        // (which already replays the committed tokens and recovers on any
        // `Err`) instead of silently decoding with full causal attention.
        // Also covers `forward_greedy_gpu_ternary`, called only from below.
        if self.config.sliding_window.is_some() {
            return Err(
                "sliding-window model: fused Metal greedy decode is full-causal; falling back to CPU"
                    .into(),
            );
        }
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        let eps = if self.blocks.is_empty() {
            return Err("no blocks".into());
        } else {
            self.blocks[0].attn_norm_eps()
        };
        let final_norm_eps = self.output_norm.eps();

        // Ternary models run the cached ternary greedy path.
        if matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return self.forward_greedy_gpu_ternary(token_id, pos);
        }

        let lm_head_out_features = match &self.output_weight {
            OutputWeight::OneBit(linear) => linear.out_features(),
            OutputWeight::Ternary(_) => {
                return Err("ternary LM head reached the Q1 greedy GPU path".into())
            }
            OutputWeight::FP8E4M3(_) | OutputWeight::FP8E5M2(_) => {
                return Err("FP8 GPU inference not yet supported; use CPU path".into());
            }
            OutputWeight::Q4_0(_)
            | OutputWeight::Q8_0(_)
            | OutputWeight::Q5K(_)
            | OutputWeight::Q6K(_)
            | OutputWeight::Q2K(_)
            | OutputWeight::Q3K(_)
            | OutputWeight::Q4K(_)
            | OutputWeight::Q8K(_) => {
                return Err(
                    "K-quant / Q-type LM head not yet supported on Metal greedy path; use CPU path"
                        .into(),
                );
            }
            OutputWeight::Fp32 { .. } => {
                return Err("FP32 LM head not supported on greedy GPU path".into());
            }
        };
        self.get_or_create_gpu_cache()?;
        // M-02: decode exactly this one row out of the quantized table
        // instead of indexing a materialized dense view of all of it.
        let mut hidden = vec![0.0f32; h];
        self.token_embd.copy_row(token_id, &mut hidden)?;
        let rope_cos = self.rope.cos_at_checked(pos)?;
        let rope_sin = self.rope.sin_at_checked(pos)?;
        let mut greedy_token_id: u32 = 0;
        let guard = self
            .gpu_weight_cache
            .lock()
            .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
        let cached = guard.as_ref().ok_or("GPU weight cache not populated")?;
        with_autorelease_pool(|| {
            oxibonsai_kernels::try_metal_full_forward_cached(
                &mut hidden,
                pos,
                cached,
                rope_cos,
                rope_sin,
                h,
                inter,
                nq,
                nkv,
                hd,
                eps,
                max_seq_len,
                final_norm_eps,
                lm_head_out_features,
                None,
                Some(&mut greedy_token_id),
            )
        })
        .map_err(|e| {
            tracing::warn!(error = % e, "cached greedy GPU forward failed");
            Box::new(e) as Box<dyn std::error::Error>
        })?;
        // MET-05 (runtime half): this path maintains the DEVICE KV cache and
        // never writes the host one, so latch the backend before returning —
        // otherwise a later CPU step would attend over an all-zero history and
        // return confident garbage instead of asking for a cache rebuild.
        self.note_device_kv_used();
        Ok(greedy_token_id)
    }

    /// Ternary-model greedy decode: all transformer layers + ternary LM head +
    /// GPU argmax, through the **cached** ternary weight set (`MET-03`) — the
    /// eight GPU handles per layer `get_or_create_gpu_cache` resolved once,
    /// bound directly (no per-token lookup, no host copy of any weight).
    fn forward_greedy_gpu_ternary(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        // Context-length guard: `self.rope.cos_at_checked(pos)` reads a
        // `RopeTable` sized to exactly `max_seq_len` rows and errors past it.
        // Mirrors the single-token `forward()` check, which owns the message.
        if pos >= self.kv_cache.max_seq_len() {
            return Err(format!(
                "ternary greedy sequence too long: pos {pos} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        with_autorelease_pool(|| self.forward_greedy_gpu_ternary_cached(token_id, pos)).map_err(
            |e| {
                tracing::warn!(error = % e, "ternary greedy GPU forward failed");
                e
            },
        )
    }

    /// Ternary fused forward + LM head (single token, non-greedy sampling).
    ///
    /// Routes the decode hot-path for ternary models through the **cached**
    /// ternary weight set (`MET-03`). This is the call `BonsaiModel::forward()`
    /// makes first; when it fails, `forward()` retries through
    /// [`Self::try_metal_full_forward_ternary_inner`], which binds the very
    /// same cached buffers (MET-02).
    ///
    /// `pub(super)` so the MET-02 regression test can drive it and the fallback
    /// directly, in the order `forward()` does.
    pub(super) fn try_metal_full_forward_with_lm_head_ternary(
        &self,
        hidden: &mut [f32],
        pos: usize,
        logits: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nq = self.config.num_attention_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let max_seq_len = self.kv_cache.max_seq_len();
        let eps = if self.blocks.is_empty() {
            return Err("no blocks".into());
        } else {
            self.blocks[0].attn_norm_eps()
        };
        let final_norm_eps = self.output_norm.eps();

        // Every weight resident first (the cached shape resolves them all)...
        self.get_or_create_gpu_cache()?;

        // MET-02 fault-injection seam. Every layer's weights are resident by
        // now and only the final-norm → LM-head tail is left, which is exactly
        // the state a real deterministic tail failure (missing TQ2 LM-head
        // pipeline, logits-buffer allocation failure, GPU fault) leaves behind —
        // the state that used to send `forward()` into a second, separate
        // handle namespace and duplicate the whole model on the GPU. Returning
        // here reproduces it without needing a broken GPU.
        if force_ternary_tail_failure() {
            return Err(
                "ternary fused GPU tail failure forced by OXIBONSAI_FORCE_METAL_TAIL_FAIL".into(),
            );
        }

        let rope_cos = self.rope.cos_at_checked(pos)?;
        let rope_sin = self.rope.sin_at_checked(pos)?;
        self.with_ternary_gpu_cache(|cached| {
            with_autorelease_pool(|| {
                oxibonsai_kernels::try_metal_prefill_ternary_cached(
                    hidden,
                    pos,
                    cached,
                    rope_cos,
                    rope_sin,
                    h,
                    inter,
                    nq,
                    nkv,
                    hd,
                    eps,
                    max_seq_len,
                    final_norm_eps,
                    logits,
                )
            })
        })?
        .map_err(|e| {
            tracing::warn!(error = % e, "ternary fused GPU forward failed");
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    /// GPU batch prefill — ternary (TQ2_0_g128) variant, through the
    /// **cached** ternary weight set ([`Self::prefill_logits_gpu_ternary_cached`]).
    /// Only the last token's logits are returned.
    ///
    /// Marked `pub` so parity tests can invoke this **strict** path
    /// directly, bypassing the silent fallback in
    /// [`Self::forward_prefill`] that masks GPU dispatch failures.
    pub fn try_metal_prefill_with_lm_head_ternary(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        if self.blocks.is_empty() {
            return Err("no blocks".into());
        }
        // Context-length guard: prevents an out-of-bounds RoPE slice panic on
        // prompts longer than the context (mirrors `forward()` and the CUDA
        // ternary prefill guard).
        if pos_start + batch_size > self.kv_cache.max_seq_len() {
            return Err(format!(
                "ternary prefill sequence too long: {batch_size} tokens at pos {pos_start} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        if !matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return Err("ternary prefill called on non-ternary model".into());
        }
        with_autorelease_pool(|| self.prefill_logits_gpu_ternary_cached(token_ids, pos_start))
            .map_err(|e| {
                // A missed deadline is reported once, by the M-18 router.
                if !is_prefill_timeout(e.as_ref()) {
                    tracing::warn!(error = % e, "ternary batch prefill GPU dispatch failed");
                }
                e
            })
    }

    /// GPU batch prefill verify — ternary (TQ2_0_g128) variant, through the
    /// **cached** ternary weight set ([`Self::prefill_verify_gpu_ternary_cached`]).
    /// Returns the per-position greedy argmax token IDs.
    ///
    /// Marked `pub` so parity tests can invoke this **strict** path
    /// directly, bypassing the silent fallback in
    /// [`Self::forward_prefill_verify`] that masks GPU dispatch failures.
    pub fn try_metal_prefill_verify_ternary_path(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let batch_size = token_ids.len();
        if self.blocks.is_empty() {
            return Err("no blocks".into());
        }
        // Context-length guard: prevents an out-of-bounds RoPE slice panic on
        // prompts longer than the context (mirrors `forward()` and the CUDA
        // ternary prefill-verify guard).
        if pos_start + batch_size > self.kv_cache.max_seq_len() {
            return Err(format!(
                "ternary prefill-verify sequence too long: {batch_size} tokens at pos {pos_start} exceeds max_seq_len {}",
                self.kv_cache.max_seq_len()
            )
            .into());
        }
        if !matches!(&self.output_weight, OutputWeight::Ternary(_)) {
            return Err("ternary prefill verify called on non-ternary model".into());
        }
        with_autorelease_pool(|| self.prefill_verify_gpu_ternary_cached(token_ids, pos_start))
            .map_err(|e| {
                tracing::warn!(error = % e, "ternary batch prefill verify GPU dispatch failed");
                e
            })
    }

    // ── Uncached ternary reference entries (the MET-03 A/B arm) ──────────────

    /// Logits of one token through the **uncached** ternary GPU path, strictly
    /// (no CPU fallback): the per-call `FullForwardLayerParamsTernary` binding
    /// ([`Self::ternary_gpu_binding`]) and eight weight-cache lookups per
    /// layer in the kernels.
    ///
    /// Nothing on the model's decode route takes this any more; it is the
    /// reference the cached shape is proven against — the MET-03 parity
    /// evidence compares it with [`Self::forward_logits_gpu_ternary_cached`]
    /// bit for bit — and the explicit "no fallback, no cache" entry a caller
    /// can use to tell a GPU failure from a CPU answer.
    ///
    /// # Errors
    ///
    /// A non-ternary model, a position past the context, a sliding-window
    /// model, or any Metal failure — never a CPU fallback.
    pub fn forward_logits_gpu_ternary_uncached(
        &self,
        token_id: u32,
        pos: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (mut hidden, rope_cos, rope_sin) = self.ternary_decode_prologue(token_id, pos)?;
        let binding = self.ternary_gpu_binding()?;
        let tail = binding
            .tail
            .ok_or("the uncached ternary logits path requires a ternary LM head")?;
        let mut logits = Vec::new();
        let result = with_autorelease_pool(|| {
            oxibonsai_kernels::try_metal_prefill_ternary(
                &mut hidden,
                pos,
                self.blocks.len(),
                &binding.layer_params,
                rope_cos,
                rope_sin,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.blocks[0].attn_norm_eps(),
                self.kv_cache.max_seq_len(),
                Some(tail.final_norm_handle),
                Some(tail.final_norm_bytes),
                self.output_norm.eps(),
                Some(tail.lm_head_handle),
                Some(tail.lm_head_bytes),
                tail.lm_head_out_features,
                &mut logits,
            )
        });
        self.metal_q1_slots.mark_used(self.blocks.len());
        result?;
        self.note_device_kv_used();
        Ok(logits)
    }

    /// Batched prefill through the **uncached** ternary GPU path, strictly:
    /// the per-call binding plus the kernels' batched ternary prefill, which
    /// resolves all eight buffers per layer and the tail under the layers'
    /// `model_epoch` (the mapping epoch). The reference
    /// [`Self::prefill_logits_gpu_ternary_cached`] is proven against; returns
    /// the last position's logits.
    ///
    /// # Errors
    ///
    /// A non-ternary model, an empty or over-long batch, or any Metal
    /// failure — never a CPU fallback.
    pub fn prefill_logits_gpu_ternary_uncached(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (hidden_batch, cos_table, sin_table) =
            self.ternary_prefill_inputs(token_ids, pos_start)?;
        let binding = self.ternary_gpu_binding()?;
        let tail = binding
            .tail
            .ok_or("the uncached ternary prefill requires a ternary LM head")?;
        let mut logits = vec![0.0f32; tail.lm_head_out_features];
        let result = with_autorelease_pool(|| {
            oxibonsai_kernels::try_metal_full_forward_prefill_ternary(
                &hidden_batch,
                token_ids.len(),
                pos_start,
                self.blocks.len(),
                &binding.layer_params,
                &cos_table,
                &sin_table,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.blocks[0].attn_norm_eps(),
                self.kv_cache.max_seq_len(),
                Some(tail.final_norm_handle),
                Some(tail.final_norm_bytes),
                self.output_norm.eps(),
                Some(tail.lm_head_handle),
                Some(tail.lm_head_bytes),
                tail.lm_head_out_features,
                Some(&mut logits),
                None,
            )
        });
        self.metal_q1_slots.mark_used(self.blocks.len());
        result?;
        self.note_device_kv_used();
        Ok(logits)
    }

    /// Batched speculative-verify prefill through the **uncached** ternary GPU
    /// path, strictly — the twin of [`Self::prefill_logits_gpu_ternary_uncached`]
    /// returning every position's greedy argmax, and the reference
    /// [`Self::prefill_verify_gpu_ternary_cached`] is proven against.
    ///
    /// # Errors
    ///
    /// As [`Self::prefill_logits_gpu_ternary_uncached`].
    pub fn prefill_verify_gpu_ternary_uncached(
        &self,
        token_ids: &[u32],
        pos_start: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let (hidden_batch, cos_table, sin_table) =
            self.ternary_prefill_inputs(token_ids, pos_start)?;
        let binding = self.ternary_gpu_binding()?;
        let tail = binding
            .tail
            .ok_or("the uncached ternary prefill verify requires a ternary LM head")?;
        let mut ids: Vec<u32> = Vec::with_capacity(token_ids.len());
        let result = with_autorelease_pool(|| {
            oxibonsai_kernels::try_metal_full_forward_prefill_verify_ternary(
                &hidden_batch,
                token_ids.len(),
                pos_start,
                self.blocks.len(),
                &binding.layer_params,
                &cos_table,
                &sin_table,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_kv_heads,
                self.config.head_dim,
                self.blocks[0].attn_norm_eps(),
                self.kv_cache.max_seq_len(),
                Some(tail.final_norm_handle),
                Some(tail.final_norm_bytes),
                self.output_norm.eps(),
                Some(tail.lm_head_handle),
                Some(tail.lm_head_bytes),
                tail.lm_head_out_features,
                &mut ids,
            )
        });
        self.metal_q1_slots.mark_used(self.blocks.len());
        result?;
        self.note_device_kv_used();
        Ok(ids)
    }
}

impl BonsaiModel<'_> {
    /// This model's Metal weight-cache namespace: the Q1 norm / LM-head slots
    /// and the mapping epoch (MET-02).
    #[must_use]
    pub fn q1_metal_slots(&self) -> &Q1MetalSlots {
        &self.metal_q1_slots
    }

    /// Evict every Metal buffer keyed under this model's mapping epoch — for
    /// a Q1 model its norms, final norm and LM head — returning how many
    /// were released, and drop the model's cached Q1 handle set so the freed
    /// buffers are not kept alive by it (the next greedy call rebuilds it). A
    /// model whose fused paths never ran (or were already released) returns
    /// `Ok(0)` without touching (or initialising) the Metal graph.
    ///
    /// **Siblings.** The namespace belongs to the GGUF mapping, so the
    /// buffers of every engine-pool replica of this model go too: a surviving
    /// replica re-uploads them on its next miss (its own cached handle set
    /// keeps the old copies alive until it is rebuilt or dropped). Dropping a
    /// model releases nothing while a sibling replica is alive — the last
    /// replica's drop releases the lot — so call this only to free the
    /// buffers of a mapping early.
    ///
    /// # Errors
    ///
    /// The Metal graph cannot be reached, or a lock is poisoned.
    pub fn release_q1_metal_slots(&self) -> Result<usize, Box<dyn std::error::Error>> {
        {
            let mut guard = self
                .gpu_weight_cache
                .lock()
                .map_err(|e| format!("gpu_weight_cache lock: {e}"))?;
            if matches!(
                guard.as_ref(),
                Some(oxibonsai_kernels::CachedModelWeights::Q1(_))
            ) {
                *guard = None;
            }
        }
        Ok(self.metal_q1_slots.release()?)
    }
}
