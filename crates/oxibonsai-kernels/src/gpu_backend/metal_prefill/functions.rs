//! The batched prefill runner on `MetalGraph`: per-layer Q1 / ternary
//! encoders, micro-batched command buffers with bounded waits, and the four
//! tails (last-row logits or greedy id, every-row argmax, every-row
//! final-normed hidden state, none).
//!
//! # Micro-batches (M-18)
//!
//! A prompt is encoded in micro-batches of at most `micro_batch` rows, one
//! command buffer each: all layers for those rows, positions continuing
//! where the previous micro-batch stopped (its keys and values are already
//! in the KV cache, so attention sees the whole history). Every command
//! buffer is waited on against a deadline
//! ([`super::super::metal_graph::commit_and_wait_bounded`]); a prefill that
//! misses it stops committing, parks the one in-flight buffer in the session
//! and returns a timeout, so the work still running after a timeout is at
//! most one micro-batch. Each micro-batch's command buffer and encoder are
//! autoreleased objects, so the whole micro-batch — creation, encode,
//! commit, wait — runs inside one `autoreleasepool`: a long prompt, or a
//! server thread prefilling request after request, keeps none of them.
//!
//! The GEMM kernel family is chosen **once per request** ([`PrefillGemm`]),
//! never per micro-batch: every kernel here computes a row independently of
//! the other rows of its batch, so a micro-batched prefill is bit-identical
//! to a single-batch one exactly when every micro-batch runs the same
//! kernels.

use metal::objc::rc::autoreleasepool;
use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, MutexGuard};
use std::time::Instant;

use super::super::metal_full_layer::GpuKvCache;
use super::super::metal_graph::{
    alloc_buf, commit_and_wait_bounded, default_prefill_budget, div_ceil, download_f32,
    effective_prefill_deadline, set_scalar, upload_f32, wait_bounded, MetalGraph, MetalGraphError,
    MetalWeightHandle, PrefillWorkShape,
};

use super::attention::{
    batched_attention_supported, prefill_batch_capacity, PrefillAttnDims, PrefillAttnPipelines,
    PrefillBufferCache,
};
use super::types::{LayerConfig, LayerWeightRefs, PrefillBuffers};

/// Smallest prefill batch routed through the tiled simdgroup ternary GEMM
/// (`gemm_tq2_g128_simdgroup`, or `gemm_tq2_g128_v10_simdgroup` when that
/// cannot be built; perf-02).
///
/// The tiled kernels sweep the weight matrix once per 64-column tile, so
/// their cost is flat below one tile, while the row-wise `v7` kernel
/// re-reads the weights once per 8-column chunk. Per layer (QKV + gate‖up +
/// down, GPU time) the tiled kernel already wins at 8 rows — 8.67 ms against
/// 13.80 ms for `v7` on the Ternary-Bonsai-8B geometry, 2.90 ms against
/// 5.90 ms on the 1.7B one — and the gap widens with the batch (at 64 rows:
/// 9.36 against 161.2 ms, 2.66 against 38.96 ms). Batches below 8 rows —
/// speculative verify's 4-token batch — stay on `v7`.
const PREFILL_TQ2_TILED_MIN_BATCH: usize = 8;

/// Smallest prefill batch routed through the tiled simdgroup Q1 GEMM
/// (`gemm_q1_g128_simdgroup`, M-18).
///
/// The row-wise `v7` kernel re-reads every input column once per weight row;
/// measured per layer on the Bonsai-8B geometry it costs 60 / 40 / 268 / 229 ms
/// (QKV / O / gate‖up / down) for a 256-token batch against 5.1 / 3.4 / 20.8 /
/// 11.2 ms for the tiled kernel, and its per-token cost keeps growing with the
/// batch (84 ms/token at 256 rows, 271 at 4096) — the super-linear prefill
/// M-18 measured. The tiled kernel's cost is flat below one 64-column tile and
/// already wins at 8 rows (9.5 against 18.3 ms per layer on the same
/// geometry); batches below 8 rows — speculative verify's 4-token batch —
/// stay on the row-wise kernel.
pub(crate) const PREFILL_Q1_TILED_MIN_BATCH: usize = 8;

/// Rows per command buffer of a logits / verify prefill.
///
/// The tiled GEMMs are already at their per-token floor at 128-512 rows
/// (6.2 / 6.0 / 5.7 ms per token on the 8B geometry), and 512 rows keep a
/// micro-batch's command buffer to a few seconds of GPU time on the largest
/// dense model — the most a timed-out prefill can leave running.
pub const PREFILL_LOGITS_MICRO_BATCH: usize = 512;

/// Process-wide count of batched prefill runs that completed.
static PREFILL_FUSED_CALL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Weight family of a prefill run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefillFormat {
    /// `Q1_0_g128` SoA weights.
    OneBit,
    /// `TQ2_0_g128` SoA weights.
    Ternary,
}

/// GEMM kernel family of one prefill request (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefillGemm {
    /// `gemm_q1_g128_v7` / `gemm_tq2_g128_v7`: one weight row per simdgroup.
    Rowwise,
    /// `gemm_q1_g128_simdgroup` / `gemm_tq2_g128_simdgroup` (its bit-identical
    /// `gemm_tq2_g128_v10_simdgroup` when it cannot be built): 64x64 tiles on
    /// the matrix units.
    Tiled,
}

/// What a prefill run produces after its last layer.
pub(crate) enum PrefillTail<'a> {
    /// Nothing: the layers only (their KV cache writes are the product).
    LayersOnly,
    /// Final RMSNorm + LM head over the **last** row: the logits, or with
    /// `greedy_out` their argmax only.
    LastRow {
        final_norm: &'a Buffer,
        eps: f32,
        lm_head: &'a Buffer,
        out_features: usize,
        logits_out: Option<&'a mut Vec<f32>>,
        greedy_out: Option<&'a mut u32>,
    },
    /// Final RMSNorm + LM head + argmax over **every** row (speculative
    /// verify).
    EveryRowArgmax {
        final_norm: &'a Buffer,
        eps: f32,
        lm_head: &'a Buffer,
        out_features: usize,
        ids_out: &'a mut Vec<u32>,
    },
    /// Final RMSNorm over **every** row, downloaded row-major
    /// `[batch x hidden]` (the embedding pass).
    HiddenRows {
        final_norm: &'a Buffer,
        eps: f32,
        rows_out: &'a mut Vec<f32>,
    },
}

/// One batched prefill request.
pub(crate) struct PrefillRun<'a> {
    pub(crate) format: PrefillFormat,
    pub(crate) hidden_batch: &'a [f32],
    pub(crate) cos_table: &'a [f32],
    pub(crate) sin_table: &'a [f32],
    pub(crate) pos_start: usize,
    pub(crate) batch_size: usize,
    pub(crate) layers: &'a [LayerWeightRefs<'a>],
    pub(crate) config: LayerConfig,
    pub(crate) micro_batch: usize,
    /// Label of every command buffer this run commits.
    pub(crate) label: &'static str,
}

/// The per-layer weight handles of the public `encode_full_forward_prefill*`
/// entry points, in their historical 8-tuple order.
pub(crate) type PrefillLayerHandles<'h> = (
    &'h Arc<MetalWeightHandle>,
    &'h Arc<MetalWeightHandle>,
    &'h Arc<MetalWeightHandle>,
    &'h Arc<MetalWeightHandle>,
    &'h Arc<MetalWeightHandle>,
    &'h Arc<MetalWeightHandle>,
    &'h Arc<MetalWeightHandle>,
    &'h Arc<MetalWeightHandle>,
);

/// Borrow the eight buffers of each layer as [`LayerWeightRefs`].
pub(crate) fn layer_refs<'h>(layers: &[PrefillLayerHandles<'h>]) -> Vec<LayerWeightRefs<'h>> {
    layers
        .iter()
        .map(|w| LayerWeightRefs {
            attn_norm: &w.0.buffer,
            qkv: &w.1.buffer,
            q_norm: &w.2.buffer,
            k_norm: &w.3.buffer,
            output_proj: &w.4.buffer,
            ffn_norm: &w.5.buffer,
            gate_up: &w.6.buffer,
            down: &w.7.buffer,
        })
        .collect()
}

/// Session-owned output buffers a tail needs, locked for the whole run.
struct TailBuffers<'g> {
    logits: Option<MutexGuard<'g, Option<Buffer>>>,
    token_ids: Option<MutexGuard<'g, Option<Buffer>>>,
    /// Shared-storage staging of a micro-batch's normed rows (the prefill's
    /// own `normed_buf` is GPU-private).
    hidden_rows: Option<Buffer>,
}

fn logits_of<'b>(t: &'b TailBuffers<'_>) -> Result<&'b Buffer, MetalGraphError> {
    t.logits
        .as_ref()
        .and_then(|g| g.as_ref())
        .ok_or(MetalGraphError::BufferCreationFailed)
}

fn token_ids_of<'b>(t: &'b TailBuffers<'_>) -> Result<&'b Buffer, MetalGraphError> {
    t.token_ids
        .as_ref()
        .and_then(|g| g.as_ref())
        .ok_or(MetalGraphError::BufferCreationFailed)
}

impl MetalGraph {
    /// Number of batched prefill runs completed since process start (M-18):
    /// what a caller checks to prove its prefill took the fused path rather
    /// than a fallback.
    #[must_use]
    pub fn prefill_fused_call_count() -> u64 {
        PREFILL_FUSED_CALL_COUNT.load(Ordering::Relaxed)
    }

    /// Batched prefill runs **this session** completed — the per-session
    /// twin of [`Self::prefill_fused_call_count`], immune to prefills other
    /// sessions (other tests, other engines) run concurrently.
    #[must_use]
    pub fn prefill_run_count(&self) -> u64 {
        self.prefill_runs.load(Ordering::Relaxed)
    }

    /// The tiled simdgroup Q1 GEMM pipeline, resolved from the combined
    /// metallib on first use; `None` (never an error) where it cannot be
    /// built, in which case the Q1 prefill keeps the row-wise kernel.
    pub(crate) fn prefill_q1_tiled_pipeline(&self) -> Option<&ComputePipelineState> {
        self.prefill_q1_tiled
            .get_or_init(
                || match autoreleasepool(|| self.pipeline_for("gemm_q1_g128_simdgroup")) {
                    Ok(pso) => Some(pso),
                    Err(e) => {
                        tracing::info!(
                            "tiled Q1 prefill GEMM unavailable ({e}); using gemm_q1_g128_v7"
                        );
                        None
                    }
                },
            )
            .as_ref()
    }

    /// The prefill's tiled ternary GEMM pipeline (`gemm_tq2_g128_simdgroup`,
    /// bit-identical to `gemm_tq2_g128_v10_simdgroup` with a cheaper staging
    /// step), resolved on first use; `None` keeps the tiled ternary prefill
    /// on `v10`.
    pub(crate) fn prefill_tq2_tiled_pipeline(&self) -> Option<&ComputePipelineState> {
        self.prefill_tq2_tiled
            .get_or_init(|| {
                match autoreleasepool(|| self.pipeline_for("gemm_tq2_g128_simdgroup")) {
                    Ok(pso) => Some(pso),
                    Err(e) => {
                        tracing::info!(
                            "tiled ternary prefill GEMM unavailable ({e}); using \
                         gemm_tq2_g128_v10_simdgroup"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    /// The GEMM family a `batch_size`-row request of `format` runs with.
    pub(crate) fn choose_prefill_gemm(
        &self,
        format: PrefillFormat,
        batch_size: usize,
    ) -> PrefillGemm {
        match format {
            PrefillFormat::Ternary if batch_size >= PREFILL_TQ2_TILED_MIN_BATCH => {
                // Resolve the tiled kernel now, before any lock or deadline.
                let _ = self.prefill_tq2_tiled_pipeline();
                PrefillGemm::Tiled
            }
            PrefillFormat::OneBit
                if batch_size >= PREFILL_Q1_TILED_MIN_BATCH
                    && self.prefill_q1_tiled_pipeline().is_some() =>
            {
                PrefillGemm::Tiled
            }
            _ => PrefillGemm::Rowwise,
        }
    }

    /// Acquire the prefill buffer set, allocating if needed.
    ///
    /// **Bucketed and grow-only** (perf-M3). The set used to be cached
    /// exact-match on `batch_size`, so every distinct prompt length threw away
    /// and reallocated ~172 MB of Metal buffers. It is now allocated for a
    /// capacity rounded up to [`prefill_batch_capacity`] and reused by any
    /// smaller batch at the same model dimensions; when it must grow, the new
    /// capacity is the larger of the resident one and the requested bucket, so
    /// a sweep of prompt lengths never oscillates. The column-major layout
    /// makes an oversized set exactly equivalent to a tight one — a batch of
    /// `n` reads and writes only the first `n` columns.
    ///
    /// A prefill command buffer parked by a timed-out wait is drained first,
    /// bounded by `deadline` (M-18): the buffers handed out are then idle.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn acquire_prefill_buffers(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
        deadline: Instant,
    ) -> Result<MutexGuard<'_, Option<PrefillBufferCache>>, MetalGraphError> {
        let mut guard = self.prefill_buffers.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("prefill_buffers lock poisoned".into())
        })?;
        self.drain_inflight_prefill(deadline)?;
        // `None` ⇒ the resident set already serves this call; `Some(cap)` ⇒
        // allocate at `cap`, never below what is already resident for these
        // same dimensions.
        let needed_capacity = match guard.as_ref() {
            Some(cache)
                if cache.serves(
                    batch_size,
                    hidden_size,
                    intermediate_size,
                    nq,
                    nkv,
                    head_dim,
                    max_seq,
                ) =>
            {
                None
            }
            Some(cache) => Some(prefill_batch_capacity(batch_size).max(cache.growth_floor(
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            ))),
            None => Some(prefill_batch_capacity(batch_size)),
        };
        if let Some(capacity) = needed_capacity {
            *guard = Some(PrefillBufferCache::allocate(
                &self.device,
                capacity,
                hidden_size,
                intermediate_size,
                nq,
                nkv,
                head_dim,
                max_seq,
            )?);
        }
        Ok(guard)
    }

    /// Wait (until `deadline`) for a prefill command buffer an earlier
    /// timed-out run parked in this session. Called with the prefill-buffer
    /// lock held, which is also the only place a buffer is parked.
    fn drain_inflight_prefill(&self, deadline: Instant) -> Result<(), MetalGraphError> {
        let mut slot = self.prefill_inflight.lock().map_err(|_| {
            MetalGraphError::ExecutionFailed("prefill_inflight lock poisoned".into())
        })?;
        // The wait and the release of the parked buffer run in one pool.
        autoreleasepool(|| {
            let Some(cmd) = slot.as_ref() else {
                return Ok(());
            };
            match wait_bounded(cmd, "prefill_inflight_drain", deadline, false) {
                Err(e) if e.is_command_buffer_timeout() => Err(e),
                // Completed, or failed: either way it no longer touches the
                // buffers, and a failure belonged to the request that timed out.
                Ok(()) | Err(_) => {
                    *slot = None;
                    Ok(())
                }
            }
        })
    }

    /// Park a command buffer whose bounded wait gave up (see
    /// [`Self::drain_inflight_prefill`]).
    fn park_inflight_prefill(&self, cmd: &metal::CommandBufferRef) {
        match self.prefill_inflight.lock() {
            Ok(mut slot) => *slot = Some(cmd.to_owned()),
            Err(_) => tracing::error!(
                "prefill_inflight lock poisoned: a timed-out prefill command buffer could not \
                 be parked"
            ),
        }
    }

    /// Run one batched prefill request (see the module docs).
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::EncodingFailed`] for inputs shorter than the request
    /// or a layer-count mismatch; a timeout
    /// ([`MetalGraphError::is_command_buffer_timeout`]) when a micro-batch
    /// misses the deadline; any allocation or command-buffer failure.
    pub(crate) fn run_prefill(
        &self,
        run: &PrefillRun<'_>,
        mut tail: PrefillTail<'_>,
    ) -> Result<(), MetalGraphError> {
        let config = &run.config;
        let h = config.hidden_size;
        let half_dim = config.head_dim / 2;
        let batch = run.batch_size;
        if batch == 0 {
            return Err(MetalGraphError::EncodingFailed(
                "prefill needs at least one token".into(),
            ));
        }
        check_len("hidden_batch", run.hidden_batch.len(), batch * h)?;
        check_len("cos_table", run.cos_table.len(), batch * half_dim)?;
        check_len("sin_table", run.sin_table.len(), batch * half_dim)?;
        // Compile (once per process) the pipelines BEFORE taking the
        // prefill-buffer and KV-cache locks and before the deadline clock
        // starts: a first-run MSL compile under load takes seconds.
        // `None` ⇒ unsupported geometry or no pipelines ⇒ per-token fallback.
        let attn =
            if batched_attention_supported(config.n_q_heads, config.n_kv_heads, config.head_dim) {
                self.prefill_attn_pipelines()
            } else {
                None
            };
        let gemm = self.choose_prefill_gemm(run.format, batch);
        let shape = PrefillWorkShape {
            n_layers: run.layers.len(),
            hidden: h,
            intermediate: config.intermediate_size,
            nq: config.n_q_heads,
            nkv: config.n_kv_heads,
            head_dim: config.head_dim,
        };
        let deadline = effective_prefill_deadline(
            Instant::now(),
            default_prefill_budget(&shape, batch, run.pos_start),
        );
        let micro = run.micro_batch.clamp(1, batch);
        let pb_guard = self.acquire_prefill_buffers(
            micro,
            h,
            config.intermediate_size,
            config.n_q_heads,
            config.n_kv_heads,
            config.head_dim,
            config.max_seq_len,
            deadline,
        )?;
        let bufs = &pb_guard
            .as_ref()
            .ok_or_else(|| {
                MetalGraphError::ExecutionFailed("prefill_buffers not allocated".into())
            })?
            .bufs;
        let kv_guard = self.acquire_kv_cache(
            run.layers.len(),
            config.n_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )?;
        let kv = kv_guard
            .as_ref()
            .ok_or_else(|| MetalGraphError::ExecutionFailed("kv_cache not allocated".into()))?;
        let tail_bufs = self.acquire_tail_buffers(&tail, micro, batch, h)?;

        let mut m0 = 0usize;
        while m0 < batch {
            let m1 = (m0 + micro).min(batch);
            let rows = m1 - m0;
            // One micro-batch per pool: its command buffer and encoder are
            // autoreleased, and a parked (timed-out) buffer is retained by
            // the session before the pool drains.
            autoreleasepool(|| {
                // SAFETY: the three buffers are shared-storage and sized for at
                // least `micro >= rows` rows; the prefill-buffer lock (held) and the
                // drain above guarantee no command buffer is still using them.
                unsafe {
                    upload_f32(&bufs.hidden_buf, &run.hidden_batch[m0 * h..m1 * h]);
                    upload_f32(&bufs.cos_buf, &run.cos_table[m0 * half_dim..m1 * half_dim]);
                    upload_f32(&bufs.sin_buf, &run.sin_table[m0 * half_dim..m1 * half_dim]);
                }
                let cmd = self.command_queue.new_command_buffer();
                let encoder = cmd.new_compute_command_encoder();
                for (layer_idx, layer) in run.layers.iter().enumerate() {
                    match run.format {
                        PrefillFormat::OneBit => self.encode_layer_prefill(
                            encoder,
                            bufs,
                            kv,
                            layer,
                            layer_idx,
                            rows,
                            run.pos_start + m0,
                            config,
                            attn,
                            gemm,
                        )?,
                        PrefillFormat::Ternary => self.encode_layer_prefill_ternary(
                            encoder,
                            bufs,
                            kv,
                            layer,
                            layer_idx,
                            rows,
                            run.pos_start + m0,
                            config,
                            attn,
                            gemm,
                        )?,
                    }
                }
                let is_last = m1 == batch;
                self.encode_prefill_tail(
                    encoder, run.format, bufs, &tail_bufs, &tail, rows, m0, is_last,
                )?;
                encoder.end_encoding();
                if let Err(e) = commit_and_wait_bounded(cmd, run.label, deadline) {
                    if e.is_command_buffer_timeout() {
                        self.park_inflight_prefill(cmd);
                    }
                    return Err(e);
                }
                finish_prefill_tail(&mut tail, &tail_bufs, rows, m0, is_last, h)
            })?;
            m0 = m1;
        }
        PREFILL_FUSED_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        self.prefill_runs.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Lock and size the session buffers `tail` writes, for the whole run.
    fn acquire_tail_buffers(
        &self,
        tail: &PrefillTail<'_>,
        micro: usize,
        batch: usize,
        h: usize,
    ) -> Result<TailBuffers<'_>, MetalGraphError> {
        let f = std::mem::size_of::<f32>();
        let u = std::mem::size_of::<u32>();
        let mut out = TailBuffers {
            logits: None,
            token_ids: None,
            hidden_rows: None,
        };
        match tail {
            PrefillTail::LayersOnly => {}
            PrefillTail::LastRow {
                out_features,
                greedy_out,
                ..
            } => {
                out.logits = Some(self.lock_sized(&self.logits_buf, (out_features * f) as u64)?);
                if greedy_out.is_some() {
                    out.token_ids = Some(self.lock_sized(&self.token_id_buf, u as u64)?);
                }
            }
            PrefillTail::EveryRowArgmax { out_features, .. } => {
                out.logits =
                    Some(self.lock_sized(&self.logits_buf, (micro * out_features * f) as u64)?);
                out.token_ids = Some(self.lock_sized(&self.token_id_buf, (batch * u) as u64)?);
            }
            PrefillTail::HiddenRows { .. } => {
                out.hidden_rows = Some(alloc_buf(
                    &self.device,
                    (micro * h * f) as u64,
                    MTLResourceOptions::StorageModeShared,
                )?);
            }
        }
        Ok(out)
    }

    /// Lock a session output buffer, (re)allocating it to at least `bytes`.
    fn lock_sized<'g>(
        &self,
        slot: &'g std::sync::Mutex<Option<Buffer>>,
        bytes: u64,
    ) -> Result<MutexGuard<'g, Option<Buffer>>, MetalGraphError> {
        let mut guard = slot
            .lock()
            .map_err(|_| MetalGraphError::ExecutionFailed("output buffer lock poisoned".into()))?;
        if guard.as_ref().is_none_or(|b| b.length() < bytes) {
            *guard = Some(alloc_buf(
                &self.device,
                bytes.max(4),
                MTLResourceOptions::StorageModeShared,
            )?);
        }
        Ok(guard)
    }

    /// Encode `tail` for one micro-batch of `rows` rows starting at batch row
    /// `m0` into `encoder`.
    #[allow(clippy::too_many_arguments)]
    fn encode_prefill_tail(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        format: PrefillFormat,
        bufs: &PrefillBuffers,
        tail_bufs: &TailBuffers<'_>,
        tail: &PrefillTail<'_>,
        rows: usize,
        m0: usize,
        is_last: bool,
    ) -> Result<(), MetalGraphError> {
        let f = std::mem::size_of::<f32>() as u64;
        match tail {
            PrefillTail::LayersOnly => {}
            PrefillTail::LastRow {
                final_norm,
                eps,
                lm_head,
                out_features,
                greedy_out,
                ..
            } => {
                if !is_last {
                    return Ok(());
                }
                let h = bufs.hidden_size() as u32;
                let last_row_offset = (rows as u64 - 1) * u64::from(h) * f;
                encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_weighted_v2);
                encoder.set_buffer(0, Some(&bufs.hidden_buf), last_row_offset);
                encoder.set_buffer(1, Some(final_norm), 0);
                encoder.set_buffer(2, Some(&bufs.normed_buf), 0);
                // SAFETY: scalar bindings at the kernel's declared indices.
                unsafe {
                    set_scalar(encoder, 3, eps);
                    set_scalar(encoder, 4, &h);
                }
                encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                let logits = logits_of(tail_bufs)?;
                let n = *out_features as u32;
                match format {
                    PrefillFormat::OneBit => {
                        self.dispatch_gemv_q1(encoder, lm_head, &bufs.normed_buf, logits, n, h)
                    }
                    PrefillFormat::Ternary => {
                        self.dispatch_gemv_tq2(encoder, lm_head, &bufs.normed_buf, logits, n, h)
                    }
                }
                if greedy_out.is_some() {
                    self.dispatch_argmax(encoder, logits, token_ids_of(tail_bufs)?, n);
                }
            }
            PrefillTail::EveryRowArgmax {
                final_norm,
                eps,
                lm_head,
                out_features,
                ..
            } => {
                let h = bufs.hidden_size() as u32;
                self.dispatch_batched_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    final_norm,
                    &bufs.normed_buf,
                    *eps,
                    h,
                    rows as u32,
                );
                let logits = logits_of(tail_bufs)?;
                let vocab = *out_features as u32;
                match format {
                    PrefillFormat::OneBit => self.dispatch_gemm_q1_v7(
                        encoder,
                        lm_head,
                        &bufs.normed_buf,
                        logits,
                        vocab,
                        h,
                        rows as u32,
                    ),
                    PrefillFormat::Ternary => {
                        for col in 0..rows as u64 {
                            encoder.set_compute_pipeline_state(&self.pipelines.gemv_tq2_g128_v1);
                            encoder.set_buffer(0, Some(lm_head), 0);
                            encoder.set_buffer(1, Some(&bufs.normed_buf), col * u64::from(h) * f);
                            encoder.set_buffer(2, Some(logits), col * u64::from(vocab) * f);
                            // SAFETY: scalar bindings at the kernel's indices.
                            unsafe {
                                set_scalar(encoder, 3, &vocab);
                                set_scalar(encoder, 4, &h);
                            }
                            encoder.dispatch_thread_groups(
                                MTLSize::new(div_ceil(vocab as usize, 8) as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }
                    }
                }
                let ids = token_ids_of(tail_bufs)?;
                let u = std::mem::size_of::<u32>() as u64;
                for col in 0..rows as u64 {
                    encoder.set_compute_pipeline_state(&self.pipelines.argmax);
                    encoder.set_buffer(0, Some(logits), col * u64::from(vocab) * f);
                    encoder.set_buffer(1, Some(ids), (m0 as u64 + col) * u);
                    // SAFETY: scalar binding at the kernel's index.
                    unsafe {
                        set_scalar(encoder, 2, &vocab);
                    }
                    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
                }
            }
            PrefillTail::HiddenRows {
                final_norm, eps, ..
            } => {
                let staging = tail_bufs
                    .hidden_rows
                    .as_ref()
                    .ok_or(MetalGraphError::BufferCreationFailed)?;
                self.dispatch_batched_rmsnorm(
                    encoder,
                    &bufs.hidden_buf,
                    final_norm,
                    staging,
                    *eps,
                    bufs.hidden_size() as u32,
                    rows as u32,
                );
            }
        }
        Ok(())
    }

    /// Encode ALL transformer layers for batch prefill.
    ///
    /// Like `encode_full_forward`, except:
    /// - `hidden_batch` is `batch_size × hidden_size` (all prompt tokens)
    /// - `cos_table`/`sin_table` are `batch_size × half_dim` (all positions' RoPE)
    /// - Each layer uses GEMM (not GEMV) for batched projections
    /// - Attention is batch-wide: two dispatches per layer (perf-01),
    ///   falling back to the per-token loop only for geometries the
    ///   batched kernels cannot serve
    /// - After all layers, only the LAST token feeds into final norm + LM head
    /// - The prompt runs in micro-batches of [`PREFILL_LOGITS_MICRO_BATCH`]
    ///   rows, each waited on against the prefill deadline (M-18)
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_forward_prefill(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
    ) -> Result<(), MetalGraphError> {
        self.encode_prefill_last_row(
            PrefillFormat::OneBit,
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            layer_weights,
            cos_table,
            sin_table,
            LayerConfig {
                hidden_size,
                intermediate_size,
                n_q_heads: nq,
                n_kv_heads: nkv,
                head_dim,
                eps,
                max_seq_len,
            },
            final_norm_w,
            final_norm_eps,
            lm_head_w,
            lm_head_out_features,
            logits_out,
            greedy_token_id_out,
        )
    }

    /// Encode full-forward prefill for **verification** (speculative decoding).
    ///
    /// Identical to `encode_full_forward_prefill` for the layer loop, but the
    /// tail runs final RMSNorm + LM-head GEMM on **all** batch positions and
    /// returns per-position argmax token IDs instead of single-token logits.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_forward_prefill_verify(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        batch_token_ids_out: &mut Vec<u32>,
    ) -> Result<(), MetalGraphError> {
        self.encode_prefill_every_row_argmax(
            PrefillFormat::OneBit,
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            layer_weights,
            cos_table,
            sin_table,
            LayerConfig {
                hidden_size,
                intermediate_size,
                n_q_heads: nq,
                n_kv_heads: nkv,
                head_dim,
                eps,
                max_seq_len,
            },
            final_norm_w,
            final_norm_eps,
            lm_head_w,
            lm_head_out_features,
            batch_token_ids_out,
        )
    }

    /// Encode ALL transformer layers for batch prefill — ternary
    /// (TQ2_0_g128) variant.
    ///
    /// Mirror of [`Self::encode_full_forward_prefill`] but with weight GEMMs
    /// routed through the TQ2 batched kernels. The final-norm + LM-head tail
    /// also uses TQ2 GEMV (`dispatch_gemv_tq2`) for the LM head.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_forward_prefill_ternary(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
    ) -> Result<(), MetalGraphError> {
        self.encode_prefill_last_row(
            PrefillFormat::Ternary,
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            layer_weights,
            cos_table,
            sin_table,
            LayerConfig {
                hidden_size,
                intermediate_size,
                n_q_heads: nq,
                n_kv_heads: nkv,
                head_dim,
                eps,
                max_seq_len,
            },
            final_norm_w,
            final_norm_eps,
            lm_head_w,
            lm_head_out_features,
            logits_out,
            greedy_token_id_out,
        )
    }

    /// Encode ALL transformer layers for batch prefill — **verify** variant
    /// (speculative decoding), ternary (TQ2_0_g128) weights.
    ///
    /// Mirror of [`Self::encode_full_forward_prefill_verify`] but with TQ2
    /// GEMM for projections and TQ2 GEMV for the per-position LM head.
    /// The LM head is dispatched per column because the existing TQ2 GEMV
    /// is not batched; this matches what the per-position fused tail does
    /// internally.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_forward_prefill_verify_ternary(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        batch_token_ids_out: &mut Vec<u32>,
    ) -> Result<(), MetalGraphError> {
        self.encode_prefill_every_row_argmax(
            PrefillFormat::Ternary,
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            layer_weights,
            cos_table,
            sin_table,
            LayerConfig {
                hidden_size,
                intermediate_size,
                n_q_heads: nq,
                n_kv_heads: nkv,
                head_dim,
                eps,
                max_seq_len,
            },
            final_norm_w,
            final_norm_eps,
            lm_head_w,
            lm_head_out_features,
            batch_token_ids_out,
        )
    }

    /// Encode ALL transformer layers for batch prefill and return the
    /// final-normed hidden state of **every** row — the head-free prefill of
    /// the embedding pass (`Q1_0_g128` weights).
    ///
    /// The same per-layer encoding as [`Self::encode_full_forward_prefill`],
    /// but the final RMSNorm runs over all `batch_size` rows (the logits
    /// entry points norm only the last one) and no LM head is dispatched;
    /// `hidden_out` receives the `[batch x hidden]` rows, row-major, and the
    /// rows run in `micro_batch`-row command buffers, positions continuing
    /// across them. The KV cache written is **this session's** — the public
    /// entry points (`try_metal_full_forward_prefill_hidden*`) call this on a
    /// request-scoped sibling session so no decode's cache is touched
    /// (MET-05).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_forward_prefill_hidden(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: &MetalWeightHandle,
        final_norm_eps: f32,
        micro_batch: usize,
        hidden_out: &mut Vec<f32>,
    ) -> Result<(), MetalGraphError> {
        self.encode_prefill_hidden_rows(
            PrefillFormat::OneBit,
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            layer_weights,
            cos_table,
            sin_table,
            LayerConfig {
                hidden_size,
                intermediate_size,
                n_q_heads: nq,
                n_kv_heads: nkv,
                head_dim,
                eps,
                max_seq_len,
            },
            final_norm_w,
            final_norm_eps,
            micro_batch,
            hidden_out,
        )
    }

    /// The ternary (`TQ2_0_g128`) twin of
    /// [`Self::encode_full_forward_prefill_hidden`].
    #[allow(clippy::too_many_arguments)]
    pub fn encode_full_forward_prefill_hidden_ternary(
        &self,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        eps: f32,
        max_seq_len: usize,
        final_norm_w: &MetalWeightHandle,
        final_norm_eps: f32,
        micro_batch: usize,
        hidden_out: &mut Vec<f32>,
    ) -> Result<(), MetalGraphError> {
        self.encode_prefill_hidden_rows(
            PrefillFormat::Ternary,
            hidden_batch,
            pos_start,
            batch_size,
            n_layers,
            layer_weights,
            cos_table,
            sin_table,
            LayerConfig {
                hidden_size,
                intermediate_size,
                n_q_heads: nq,
                n_kv_heads: nkv,
                head_dim,
                eps,
                max_seq_len,
            },
            final_norm_w,
            final_norm_eps,
            micro_batch,
            hidden_out,
        )
    }

    /// Shared body of the two hidden-row entry points.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_prefill_hidden_rows(
        &self,
        format: PrefillFormat,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        config: LayerConfig,
        final_norm_w: &MetalWeightHandle,
        final_norm_eps: f32,
        micro_batch: usize,
        hidden_out: &mut Vec<f32>,
    ) -> Result<(), MetalGraphError> {
        check_layer_handles(n_layers, layer_weights.len())?;
        let layers = layer_refs(layer_weights);
        let run = PrefillRun {
            format,
            hidden_batch,
            cos_table,
            sin_table,
            pos_start,
            batch_size,
            layers: &layers,
            config,
            micro_batch,
            label: match format {
                PrefillFormat::OneBit => "prefill_q1_hidden",
                PrefillFormat::Ternary => "prefill_ternary_hidden",
            },
        };
        self.run_prefill(
            &run,
            PrefillTail::HiddenRows {
                final_norm: &final_norm_w.buffer,
                eps: final_norm_eps,
                rows_out: hidden_out,
            },
        )
    }

    /// Shared body of the two last-row (logits / greedy) entry points.
    #[allow(clippy::too_many_arguments)]
    fn encode_prefill_last_row(
        &self,
        format: PrefillFormat,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        config: LayerConfig,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        logits_out: Option<&mut Vec<f32>>,
        greedy_token_id_out: Option<&mut u32>,
    ) -> Result<(), MetalGraphError> {
        check_layer_handles(n_layers, layer_weights.len())?;
        let layers = layer_refs(layer_weights);
        let greedy = greedy_token_id_out.is_some();
        let tail = match (final_norm_w, lm_head_w) {
            (Some(fnorm), Some(lm)) if lm_head_out_features > 0 => PrefillTail::LastRow {
                final_norm: &fnorm.buffer,
                eps: final_norm_eps,
                lm_head: &lm.buffer,
                out_features: lm_head_out_features,
                logits_out,
                greedy_out: greedy_token_id_out,
            },
            _ => PrefillTail::LayersOnly,
        };
        let label = match (format, &tail, greedy) {
            (_, PrefillTail::LayersOnly, _) => match format {
                PrefillFormat::OneBit => "prefill_q1_no_lm_head",
                PrefillFormat::Ternary => "prefill_ternary_no_lm_head",
            },
            (PrefillFormat::OneBit, _, true) => "prefill_q1_greedy",
            (PrefillFormat::OneBit, _, false) => "prefill_q1_logits",
            (PrefillFormat::Ternary, _, true) => "prefill_ternary_greedy",
            (PrefillFormat::Ternary, _, false) => "prefill_ternary_logits",
        };
        let run = PrefillRun {
            format,
            hidden_batch,
            cos_table,
            sin_table,
            pos_start,
            batch_size,
            layers: &layers,
            config,
            micro_batch: PREFILL_LOGITS_MICRO_BATCH,
            label,
        };
        self.run_prefill(&run, tail)
    }

    /// Shared body of the two verify (every-row argmax) entry points.
    #[allow(clippy::too_many_arguments)]
    fn encode_prefill_every_row_argmax(
        &self,
        format: PrefillFormat,
        hidden_batch: &[f32],
        pos_start: usize,
        batch_size: usize,
        n_layers: usize,
        layer_weights: &[PrefillLayerHandles<'_>],
        cos_table: &[f32],
        sin_table: &[f32],
        config: LayerConfig,
        final_norm_w: Option<&Arc<MetalWeightHandle>>,
        final_norm_eps: f32,
        lm_head_w: Option<&Arc<MetalWeightHandle>>,
        lm_head_out_features: usize,
        batch_token_ids_out: &mut Vec<u32>,
    ) -> Result<(), MetalGraphError> {
        check_layer_handles(n_layers, layer_weights.len())?;
        let layers = layer_refs(layer_weights);
        let (tail, label) = match (final_norm_w, lm_head_w) {
            (Some(fnorm), Some(lm)) if lm_head_out_features > 0 => (
                PrefillTail::EveryRowArgmax {
                    final_norm: &fnorm.buffer,
                    eps: final_norm_eps,
                    lm_head: &lm.buffer,
                    out_features: lm_head_out_features,
                    ids_out: batch_token_ids_out,
                },
                match format {
                    PrefillFormat::OneBit => "prefill_q1_verify",
                    PrefillFormat::Ternary => "prefill_ternary_verify",
                },
            ),
            _ => (
                PrefillTail::LayersOnly,
                match format {
                    PrefillFormat::OneBit => "prefill_q1_verify_no_lm_head",
                    PrefillFormat::Ternary => "prefill_ternary_verify_no_lm_head",
                },
            ),
        };
        let run = PrefillRun {
            format,
            hidden_batch,
            cos_table,
            sin_table,
            pos_start,
            batch_size,
            layers: &layers,
            config,
            micro_batch: PREFILL_LOGITS_MICRO_BATCH,
            label,
        };
        self.run_prefill(&run, tail)
    }

    /// Encode a single transformer layer for batch prefill (multiple tokens).
    ///
    /// Non-attention operations (RMSNorm, QKV projection, FFN) use batched GEMM
    /// kernels to process all tokens in parallel. Attention is batch-wide too
    /// since perf-01: `prefill_qkv_prepare` + `prefill_flash_attention`, two
    /// dispatches for the whole prompt, with the causal mask applied in-kernel.
    /// [`Self::encode_attention_per_token`] remains the fallback for geometries
    /// the batched kernels cannot serve.
    ///
    /// The hidden state is read from and written to `bufs.hidden_buf` in-place
    /// via residual-add GEMM variants.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_prefill(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        layer_idx: usize,
        batch_size: usize,
        pos_start: usize,
        config: &LayerConfig,
        attn: Option<&PrefillAttnPipelines>,
        gemm: PrefillGemm,
    ) -> Result<(), MetalGraphError> {
        let h = config.hidden_size;
        let nq = config.n_q_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_size;
        let qkv_out = (nq + 2 * nkv) * hd;
        let bs = batch_size as u32;
        let inv_sqrt_hd = 1.0f32 / (hd as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.attn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_gemm_q1_prefill(
            encoder,
            gemm,
            layer_weights.qkv,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_out as u32,
            h as u32,
            bs,
            None,
        )?;
        // ── Attention phase ───────────────────────────────────────────────
        // perf-01: the whole prompt in TWO batch-wide dispatches, replacing
        // the `6 x batch` tiny dispatches this used to encode. The per-token
        // fallback is the historical path and runs only when the batched
        // kernels are unavailable for this geometry or device.
        let attn_dims = PrefillAttnDims {
            nq: nq as u32,
            nkv: nkv as u32,
            heads_per_group,
            head_dim: hd as u32,
            eps: config.eps,
            max_seq: config.max_seq_len as u32,
            pos_start: pos_start as u32,
            batch_size: bs,
            scale: inv_sqrt_hd,
            layer_offset: cache_layer_offset,
        };
        match attn {
            Some(pipelines) => self.encode_attention_batched(
                encoder,
                bufs,
                kv,
                layer_weights,
                pipelines,
                &attn_dims,
            ),
            None => self.encode_attention_per_token(
                encoder,
                bufs,
                kv,
                layer_weights,
                batch_size,
                pos_start,
                config,
                cache_layer_offset,
            ),
        }
        self.dispatch_gemm_q1_prefill(
            encoder,
            gemm,
            layer_weights.output_proj,
            &bufs.attn_out_buf,
            &bufs.hidden_buf,
            h as u32,
            (nq * hd) as u32,
            bs,
            Some(&bufs.hidden_buf),
        )?;
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.ffn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_gate_up_q1_prefill(
            encoder,
            gemm,
            layer_weights.gate_up,
            bufs,
            inter as u32,
            h as u32,
            bs,
        )?;
        self.dispatch_gemm_q1_prefill(
            encoder,
            gemm,
            layer_weights.down,
            &bufs.swiglu_buf,
            &bufs.hidden_buf,
            h as u32,
            inter as u32,
            bs,
            Some(&bufs.hidden_buf),
        )?;
        Ok(())
    }

    /// Encode a single transformer layer for batch prefill — ternary
    /// (TQ2_0_g128) variant.
    ///
    /// Mirrors [`Self::encode_layer_prefill`] step-for-step: same RMSNorm,
    /// same batch-wide attention phase, same residual structure. The only
    /// difference is that every weight GEMM dispatches through
    /// [`Self::dispatch_gemm_tq2_prefill`] (`v10` for tiled requests, `v7`
    /// otherwise), and
    /// because TQ2 has no fused residual / fused gate+up+SwiGLU GEMM kernel
    /// the corresponding sites expand to two dispatches each:
    ///
    /// - Q1 `gemm_q1_v7_residual(W, x, hidden)` →
    ///   `gemm_tq2(W, x, normed_buf)` + `residual_add(hidden, normed_buf)`.
    /// - Q1 `fused_gate_up_swiglu_gemm_q1(W, x, swiglu_buf)` →
    ///   `gemm_tq2(W, x, gate_up_buf [n_rows = 2·inter])`
    ///   + `batched_swiglu(gate_up_buf, swiglu_buf)`.
    ///
    /// `normed_buf` is reused as the scratch destination for both residual
    /// adds — it is overwritten by the next layer's RMSNorm anyway, and
    /// during the final layer the residual_add is the last write so its
    /// staleness does not matter.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_prefill_ternary(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        layer_idx: usize,
        batch_size: usize,
        pos_start: usize,
        config: &LayerConfig,
        attn: Option<&PrefillAttnPipelines>,
        gemm: PrefillGemm,
    ) -> Result<(), MetalGraphError> {
        let h = config.hidden_size;
        let nq = config.n_q_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_size;
        let qkv_out = (nq + 2 * nkv) * hd;
        let bs = batch_size as u32;
        let inv_sqrt_hd = 1.0f32 / (hd as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let cache_layer_offset = kv.layer_offset_elements(layer_idx);
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.attn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_gemm_tq2_prefill(
            encoder,
            gemm,
            layer_weights.qkv,
            &bufs.normed_buf,
            &bufs.qkv_buf,
            qkv_out as u32,
            h as u32,
            bs,
        );
        // ── Attention phase ───────────────────────────────────────────────
        // perf-01: the whole prompt in TWO batch-wide dispatches, replacing
        // the `6 x batch` tiny dispatches this used to encode. The per-token
        // fallback is the historical path and runs only when the batched
        // kernels are unavailable for this geometry or device.
        let attn_dims = PrefillAttnDims {
            nq: nq as u32,
            nkv: nkv as u32,
            heads_per_group,
            head_dim: hd as u32,
            eps: config.eps,
            max_seq: config.max_seq_len as u32,
            pos_start: pos_start as u32,
            batch_size: bs,
            scale: inv_sqrt_hd,
            layer_offset: cache_layer_offset,
        };
        match attn {
            Some(pipelines) => self.encode_attention_batched(
                encoder,
                bufs,
                kv,
                layer_weights,
                pipelines,
                &attn_dims,
            ),
            None => self.encode_attention_per_token(
                encoder,
                bufs,
                kv,
                layer_weights,
                batch_size,
                pos_start,
                config,
                cache_layer_offset,
            ),
        }
        self.dispatch_gemm_tq2_prefill(
            encoder,
            gemm,
            layer_weights.output_proj,
            &bufs.attn_out_buf,
            &bufs.normed_buf,
            h as u32,
            (nq * hd) as u32,
            bs,
        );
        self.dispatch_residual_add(
            encoder,
            &bufs.hidden_buf,
            &bufs.normed_buf,
            (batch_size * h) as u32,
        );
        self.dispatch_batched_rmsnorm(
            encoder,
            &bufs.hidden_buf,
            layer_weights.ffn_norm,
            &bufs.normed_buf,
            config.eps,
            h as u32,
            bs,
        );
        self.dispatch_gemm_tq2_prefill(
            encoder,
            gemm,
            layer_weights.gate_up,
            &bufs.normed_buf,
            &bufs.gate_up_buf,
            (2 * inter) as u32,
            h as u32,
            bs,
        );
        self.dispatch_batched_swiglu(
            encoder,
            &bufs.gate_up_buf,
            &bufs.swiglu_buf,
            inter as u32,
            bs,
        );
        self.dispatch_gemm_tq2_prefill(
            encoder,
            gemm,
            layer_weights.down,
            &bufs.swiglu_buf,
            &bufs.normed_buf,
            h as u32,
            inter as u32,
            bs,
        );
        self.dispatch_residual_add(
            encoder,
            &bufs.hidden_buf,
            &bufs.normed_buf,
            (batch_size * h) as u32,
        );
        Ok(())
    }

    /// Ternary (TQ2_0_g128) GEMM for the prefill path: the tiled
    /// `simdgroup_matrix` kernel for a tiled request, `v7` for a row-wise one
    /// (perf-02).
    ///
    /// The tiled kernel is `gemm_tq2_g128_simdgroup`: the tile, K slicing and
    /// matrix-unit MAC order of `gemm_tq2_g128_v10_simdgroup` (the kernel the
    /// DiT path ships, which it falls back to when it cannot be built), so it
    /// produces the same bits, with a cheaper staging step. Both read the
    /// **same** SoA weight buffer and the same column-major
    /// `inputs[col*k + elem]` / `outputs[col*n_rows + row]` layout as `v7`, so
    /// the swap is argument-for-argument. Below
    /// [`PREFILL_TQ2_TILED_MIN_BATCH`] columns `v7` runs instead, which keeps
    /// the two small-batch callers on `v7` by construction: single-token
    /// decode (a GEMV, not this path at all) and speculative verify
    /// (batch = 4). The choice is made once per request
    /// ([`Self::choose_prefill_gemm`]), so every micro-batch of a request runs
    /// the same kernel.
    ///
    /// The tiled kernels stage the dequantized weight as `half` — exact for
    /// ternary `code x scale` — and accumulate in f32 in a different order
    /// from `v7`, so results are numerically equivalent (cos >= 0.999,
    /// max-abs ~1e-5) but not bit-identical to it.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm_tq2_prefill(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gemm: PrefillGemm,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
    ) {
        match (gemm, self.prefill_tq2_tiled_pipeline()) {
            (PrefillGemm::Tiled, Some(pso)) => {
                encoder.set_compute_pipeline_state(pso);
                encoder.set_buffer(0, Some(blocks), 0);
                encoder.set_buffer(1, Some(inputs), 0);
                encoder.set_buffer(2, Some(outputs), 0);
                // SAFETY: scalar bindings at the kernel's declared indices.
                unsafe {
                    set_scalar(encoder, 3, &n_rows);
                    set_scalar(encoder, 4, &batch_size);
                    set_scalar(encoder, 5, &k);
                }
                encoder.dispatch_thread_groups(
                    MTLSize::new(
                        div_ceil(n_rows as usize, 64) as u64,
                        div_ceil(batch_size as usize, 64) as u64,
                        1,
                    ),
                    MTLSize::new(128, 1, 1),
                );
            }
            (PrefillGemm::Tiled, None) => {
                self.dispatch_gemm_tq2_v10(encoder, blocks, inputs, outputs, n_rows, k, batch_size)
            }
            (PrefillGemm::Rowwise, _) => {
                self.dispatch_gemm_tq2_v7(encoder, blocks, inputs, outputs, n_rows, k, batch_size)
            }
        }
    }

    /// Q1_0_g128 GEMM for the prefill path, `outputs = residual + W·x` when
    /// `residual` is given (it may alias `outputs`), `W·x` otherwise:
    /// `gemm_q1_g128_simdgroup` for a tiled request, `gemm_q1_g128_v7{,_residual}`
    /// for a row-wise one (M-18; see [`PREFILL_Q1_TILED_MIN_BATCH`]).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm_q1_prefill(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gemm: PrefillGemm,
        blocks: &Buffer,
        inputs: &Buffer,
        outputs: &Buffer,
        n_rows: u32,
        k: u32,
        batch_size: u32,
        residual: Option<&Buffer>,
    ) -> Result<(), MetalGraphError> {
        match (gemm, residual) {
            (PrefillGemm::Tiled, _) => {
                let pso = self.prefill_q1_tiled_pipeline().ok_or_else(|| {
                    MetalGraphError::EncodingFailed(
                        "tiled Q1 prefill GEMM chosen without its pipeline".into(),
                    )
                })?;
                encoder.set_compute_pipeline_state(pso);
                encoder.set_buffer(0, Some(blocks), 0);
                encoder.set_buffer(1, Some(inputs), 0);
                encoder.set_buffer(2, Some(outputs), 0);
                encoder.set_buffer(6, Some(residual.unwrap_or(outputs)), 0);
                let mode = u32::from(residual.is_some());
                // SAFETY: scalar bindings at the kernel's declared indices.
                unsafe {
                    set_scalar(encoder, 3, &n_rows);
                    set_scalar(encoder, 4, &batch_size);
                    set_scalar(encoder, 5, &k);
                    set_scalar(encoder, 7, &mode);
                }
                encoder.dispatch_thread_groups(
                    MTLSize::new(
                        div_ceil(n_rows as usize, 64) as u64,
                        div_ceil(batch_size as usize, 64) as u64,
                        1,
                    ),
                    MTLSize::new(128, 1, 1),
                );
            }
            (PrefillGemm::Rowwise, None) => {
                self.dispatch_gemm_q1_v7(encoder, blocks, inputs, outputs, n_rows, k, batch_size)
            }
            (PrefillGemm::Rowwise, Some(residual)) => self.dispatch_gemm_q1_v7_residual(
                encoder, blocks, inputs, outputs, n_rows, k, batch_size, residual,
            ),
        }
        Ok(())
    }

    /// The Q1 gate‖up projection + SwiGLU into `bufs.swiglu_buf`: the tiled
    /// GEMM over the concatenated `2·inter` rows into `gate_up_buf` followed by
    /// `batched_swiglu` (the ternary path's shape), or the row-wise fused
    /// `fused_gate_up_swiglu_gemm_q1` kernel.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_gate_up_q1_prefill(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        gemm: PrefillGemm,
        weight: &Buffer,
        bufs: &PrefillBuffers,
        inter: u32,
        k: u32,
        batch_size: u32,
    ) -> Result<(), MetalGraphError> {
        match gemm {
            PrefillGemm::Tiled => {
                self.dispatch_gemm_q1_prefill(
                    encoder,
                    gemm,
                    weight,
                    &bufs.normed_buf,
                    &bufs.gate_up_buf,
                    2 * inter,
                    k,
                    batch_size,
                    None,
                )?;
                self.dispatch_batched_swiglu(
                    encoder,
                    &bufs.gate_up_buf,
                    &bufs.swiglu_buf,
                    inter,
                    batch_size,
                );
            }
            PrefillGemm::Rowwise => self.dispatch_fused_gate_up_swiglu_gemm(
                encoder,
                weight,
                &bufs.normed_buf,
                &bufs.swiglu_buf,
                inter,
                k,
                batch_size,
            ),
        }
        Ok(())
    }

    /// Batched attention phase for one prefill layer — **two dispatches for
    /// the whole prompt** (perf-01).
    ///
    /// 1. `prefill_qkv_prepare` RMSNorms and rotates every token's Q and K
    ///    heads and writes K/V straight into the KV cache. Q is rewritten **in
    ///    place** over the Q section of `qkv_buf`, which is why this path needs
    ///    no buffer the per-token path did not already have.
    /// 2. `prefill_flash_attention` computes causal GQA attention for every
    ///    `(token, head)` with an online softmax, so the `batch x n_ctx` score
    ///    matrix is never materialised, and writes `attn_out_buf`.
    ///
    /// Both dispatches go into the caller's encoder. Metal serialises
    /// dispatches within one compute encoder (the default
    /// `MTLDispatchTypeSerial`), so the flash kernel is guaranteed to see the
    /// keys and values the prepare kernel just stored — the same ordering
    /// guarantee the per-token path relied on between `fused_kv_store` and
    /// `batched_attention_scores_v2`.
    fn encode_attention_batched(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        pipelines: &PrefillAttnPipelines,
        dims: &PrefillAttnDims,
    ) {
        self.dispatch_prefill_qkv_prepare(
            pipelines,
            encoder,
            &bufs.qkv_buf,
            layer_weights.q_norm,
            layer_weights.k_norm,
            &bufs.cos_buf,
            &bufs.sin_buf,
            &kv.k_cache,
            &kv.v_cache,
            dims,
        );
        self.dispatch_prefill_flash_attention(
            pipelines,
            encoder,
            &bufs.qkv_buf,
            &kv.k_cache,
            &kv.v_cache,
            &bufs.attn_out_buf,
            dims,
        );
        Self::note_batched_attention_layer();
    }

    /// Historical per-token attention phase: six single-token dispatches per
    /// prompt token (`fused_qk_norm`, `fused_qk_rope`, `fused_kv_store`,
    /// `batched_attention_scores_v2`, `batched_softmax`,
    /// `batched_attention_weighted_sum`).
    ///
    /// Retained as the fallback for geometries the batched kernels cannot
    /// serve ([`batched_attention_supported`]) and for devices/toolchains
    /// where they fail to compile. It is the reference the batched path is
    /// parity-tested against, so it must stay byte-for-byte what it was.
    #[allow(clippy::too_many_arguments)]
    fn encode_attention_per_token(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        bufs: &PrefillBuffers,
        kv: &GpuKvCache,
        layer_weights: &LayerWeightRefs<'_>,
        batch_size: usize,
        pos_start: usize,
        config: &LayerConfig,
        cache_layer_offset: u64,
    ) {
        let nq = config.n_q_heads;
        let nkv = config.n_kv_heads;
        let hd = config.head_dim;
        let qkv_out = (nq + 2 * nkv) * hd;
        let half_dim = hd / 2;
        let inv_sqrt_hd = 1.0f32 / (hd as f32).sqrt();
        let heads_per_group = (nq / nkv) as u32;
        let f = std::mem::size_of::<f32>();
        for t in 0..batch_size {
            let pos = pos_start + t;
            let seq_len = (pos + 1) as u32;
            let qkv_col_byte_offset = (t * qkv_out * f) as u64;
            let q_byte_offset = qkv_col_byte_offset;
            let k_byte_offset = qkv_col_byte_offset + (nq * hd * f) as u64;
            let v_byte_offset = qkv_col_byte_offset + ((nq + nkv) * hd * f) as u64;
            self.dispatch_fused_qk_norm(
                encoder,
                &bufs.qkv_buf,
                q_byte_offset,
                &bufs.qkv_buf,
                k_byte_offset,
                &bufs.q_normed_buf,
                &bufs.k_normed_buf,
                layer_weights.q_norm,
                layer_weights.k_norm,
                nq as u32,
                nkv as u32,
                hd as u32,
                config.eps,
            );
            let rope_byte_offset = (t * half_dim * f) as u64;
            {
                encoder.set_compute_pipeline_state(&self.pipelines.fused_qk_rope);
                encoder.set_buffer(0, Some(&bufs.q_normed_buf), 0);
                encoder.set_buffer(1, Some(&bufs.k_normed_buf), 0);
                encoder.set_buffer(2, Some(&bufs.q_rope_buf), 0);
                encoder.set_buffer(3, Some(&bufs.k_rope_buf), 0);
                encoder.set_buffer(4, Some(&bufs.cos_buf), rope_byte_offset);
                encoder.set_buffer(5, Some(&bufs.sin_buf), rope_byte_offset);
                unsafe {
                    set_scalar(encoder, 6, &(nq as u32));
                    set_scalar(encoder, 7, &(nkv as u32));
                    set_scalar(encoder, 8, &(half_dim as u32));
                }
                let tg_x = div_ceil(half_dim, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, (nq + nkv) as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
            self.dispatch_fused_kv_store(
                encoder,
                &bufs.k_rope_buf,
                &bufs.qkv_buf,
                v_byte_offset,
                &kv.k_cache,
                &kv.v_cache,
                nkv as u32,
                hd as u32,
                config.max_seq_len as u32,
                pos as u32,
                cache_layer_offset,
            );
            {
                self.dispatch_attention_scores_v2(
                    encoder,
                    &bufs.q_rope_buf,
                    &kv.k_cache,
                    &bufs.scores_buf,
                    hd as u32,
                    nq as u32,
                    nkv as u32,
                    heads_per_group,
                    config.max_seq_len as u32,
                    seq_len,
                    inv_sqrt_hd,
                    cache_layer_offset,
                );
            }
            {
                encoder.set_compute_pipeline_state(&self.pipelines.batched_softmax);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                unsafe {
                    set_scalar(encoder, 1, &(nq as u32));
                    set_scalar(encoder, 2, &(config.max_seq_len as u32));
                    set_scalar(encoder, 3, &seq_len);
                }
                encoder
                    .dispatch_thread_groups(MTLSize::new(nq as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            {
                let attn_col_byte_offset = (t * nq * hd * f) as u64;
                encoder.set_compute_pipeline_state(&self.pipelines.batched_attention_weighted_sum);
                encoder.set_buffer(0, Some(&bufs.scores_buf), 0);
                encoder.set_buffer(1, Some(&kv.v_cache), 0);
                encoder.set_buffer(2, Some(&bufs.attn_out_buf), attn_col_byte_offset);
                unsafe {
                    set_scalar(encoder, 3, &(hd as u32));
                    set_scalar(encoder, 4, &(nq as u32));
                    set_scalar(encoder, 5, &(nkv as u32));
                    set_scalar(encoder, 6, &heads_per_group);
                    set_scalar(encoder, 7, &(config.max_seq_len as u32));
                    set_scalar(encoder, 8, &seq_len);
                    set_scalar(encoder, 9, &cache_layer_offset);
                }
                let tg_x = div_ceil(hd, 64) as u64;
                encoder.dispatch_thread_groups(
                    MTLSize::new(tg_x, nq as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            }
        }
    }
}

/// Read back what `tail` produced for one micro-batch, after its command
/// buffer completed.
fn finish_prefill_tail(
    tail: &mut PrefillTail<'_>,
    tail_bufs: &TailBuffers<'_>,
    rows: usize,
    m0: usize,
    is_last: bool,
    h: usize,
) -> Result<(), MetalGraphError> {
    match tail {
        PrefillTail::LayersOnly => {}
        PrefillTail::LastRow {
            out_features,
            logits_out,
            greedy_out,
            ..
        } => {
            if !is_last {
                return Ok(());
            }
            if let Some(out) = greedy_out.as_deref_mut() {
                let ids = token_ids_of(tail_bufs)?;
                // SAFETY: shared-storage buffer of at least one u32, written by
                // the argmax of the completed command buffer.
                *out = unsafe { *(ids.contents() as *const u32) };
            } else if let Some(out) = logits_out.as_deref_mut() {
                out.resize(*out_features, 0.0);
                // SAFETY: shared-storage logits buffer of at least
                // `out_features` floats, written by the completed LM head.
                unsafe { download_f32(logits_of(tail_bufs)?, out) };
            }
        }
        PrefillTail::EveryRowArgmax { ids_out, .. } => {
            if m0 == 0 {
                ids_out.clear();
            }
            let ids = token_ids_of(tail_bufs)?;
            // SAFETY: shared-storage buffer sized for the whole batch; rows
            // `m0..m0+rows` were written by the completed argmax dispatches.
            unsafe {
                let ptr = ids.contents() as *const u32;
                for col in 0..rows {
                    ids_out.push(*ptr.add(m0 + col));
                }
            }
        }
        PrefillTail::HiddenRows { rows_out, .. } => {
            if m0 == 0 {
                rows_out.clear();
            }
            let staging = tail_bufs
                .hidden_rows
                .as_ref()
                .ok_or(MetalGraphError::BufferCreationFailed)?;
            let start = rows_out.len();
            rows_out.resize(start + rows * h, 0.0);
            // SAFETY: shared-storage staging buffer sized for `micro >= rows`
            // rows of `h` floats, written by the completed final norm.
            unsafe { download_f32(staging, &mut rows_out[start..]) };
        }
    }
    Ok(())
}

/// Reject an input shorter than the request needs.
fn check_len(what: &str, got: usize, need: usize) -> Result<(), MetalGraphError> {
    if got < need {
        return Err(MetalGraphError::EncodingFailed(format!(
            "{what} too short: need {need}, got {got}"
        )));
    }
    Ok(())
}

/// Reject a per-layer handle slice whose length disagrees with `n_layers`.
fn check_layer_handles(n_layers: usize, given: usize) -> Result<(), MetalGraphError> {
    if given != n_layers {
        return Err(MetalGraphError::EncodingFailed(format!(
            "layer_weights length mismatch: need {n_layers}, got {given}"
        )));
    }
    Ok(())
}
