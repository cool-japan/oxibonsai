//! Batched prefill of the Qwen3.5 hybrid encoder (a child module of
//! `metal_full_layer::qwen35`): the tiled GEMM that replaces the per-token
//! GEMV for every projection of a multi-token call, the object setting that
//! selects it ([`Qwen35PrefillMode`]), and the batch-capacity control.
//!
//! # What changes between the two modes
//!
//! Only the projections. [`Qwen35PrefillMode::Sequential`] runs every
//! projection of a `t`-token call as the decode GEMV with one grid row per
//! token — each token re-streams the whole weight matrix, so a prefill runs
//! at the decode rate, but a token's logits are bit-identical to feeding the
//! tokens one at a time. [`Qwen35PrefillMode::Batched`] runs the projections
//! of every call of at least [`Q35_GEMM_MIN_COLS`] tokens through the
//! `q35_gemm_*` kernels (`kernel_sources::qwen35_gemm`): each weight is
//! decoded once per 64 tokens and multiplied on the 8 × 8 matrix units. The
//! norms, rotations, the conv, the Gated-DeltaNet recurrence, the KV store,
//! the attention and the LM head run the same kernels in both modes, and a
//! single-token call (decode) always takes the GEMV — so decode is bitwise
//! unchanged. The GEMM sums a projection in a different order, so a batched
//! prefill's logits agree with the sequential ones to rounding (the tests
//! band them), not bitwise.
//!
//! The runner consults no environment: the mode is an explicit setting on
//! the model ([`Qwen35GpuModel::set_prefill_mode`]), `Batched` by default.
//!
//! # Batch capacity
//!
//! `Qwen35GpuConfig::max_batch` bounds the tokens one call takes and sizes
//! the activation scratch (the per-token floats of `Scratch`), which is part
//! of what [`qwen35_footprint`](super::qwen35_footprint) charges against the
//! device's working set. [`Qwen35GpuModel::set_max_batch`] moves it after
//! construction — refusing a batch whose scratch would push the KV window
//! past the device ceiling — so a caller can honour a prefill chunk larger
//! than the one the model was built with.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::{
    encode_gemv, qwen35_context_capacity, set_u32, DeviceMatrix, MetalGraphError, Qwen35GpuModel,
    Qwen35MatrixData, Scratch,
};

/// Tokens a call must have before [`Qwen35PrefillMode::Batched`] routes its
/// projections through the GEMM. A GEMM pass decodes every weight of a
/// 64-token tile whatever the call's size, so a short call pays for the
/// whole tile: measured on the M3 at `attn_qkv` (10240 × 5120), one pass
/// costs 2.6 ms (`PQ2_0`) / 3.7 ms (`PTQ1_0`) against 0.24 / 0.23 ms per
/// token for the GEMV — the GEMM wins from about 11 (`PQ2_0`) and 16
/// (`PTQ1_0`) tokens. Decode (one token) is always the GEMV.
pub const Q35_GEMM_MIN_COLS: usize = 16;

/// Output-tile edge of the `q35_gemm_*` kernels (`Q35G_TM` / `Q35G_TN`).
pub(super) const GEMM_TILE: usize = 64;
/// Threads per `q35_gemm_*` threadgroup (4 simdgroups).
pub(super) const GEMM_THREADS: u64 = 128;

/// Whether the quantized GEMMs stage the activations as `half` (the `_ha`
/// entry points) instead of `f32`. `f32` keeps the activations exact at the
/// throughput the batched prefill needs (see the kernel module docs).
pub(super) const GEMM_HALF_ACTIVATIONS: bool = false;

/// How a multi-token call runs its projections (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Qwen35PrefillMode {
    /// Every projection is the decode GEMV, once per token: bit-identical to
    /// feeding the tokens one at a time, at the decode rate.
    Sequential,
    /// Projections of a call of at least [`Q35_GEMM_MIN_COLS`] tokens run
    /// through the tiled GEMM; decode (one token) stays on the GEMV.
    #[default]
    Batched,
}

impl Qwen35PrefillMode {
    /// Stable lower-case name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Batched => "batched",
        }
    }
}

impl std::fmt::Display for Qwen35PrefillMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Qwen35MatrixData<'_> {
    /// The batched-prefill GEMM entry point for this format.
    pub(super) fn gemm_kernel(&self) -> &'static str {
        match (self, GEMM_HALF_ACTIVATIONS) {
            (Self::Pq2_0(_), false) => "q35_gemm_pq2",
            (Self::Ptq1_0(_), false) => "q35_gemm_ptq1",
            (Self::Tq2_0G128(_), false) => "q35_gemm_tq2",
            (Self::Q2_0G64(_), false) => "q35_gemm_q2g64",
            (Self::Q1_0G128(_), false) => "q35_gemm_q1",
            (Self::Pq2_0(_), true) => "q35_gemm_pq2_ha",
            (Self::Ptq1_0(_), true) => "q35_gemm_ptq1_ha",
            (Self::Tq2_0G128(_), true) => "q35_gemm_tq2_ha",
            (Self::Q2_0G64(_), true) => "q35_gemm_q2g64_ha",
            (Self::Q1_0G128(_), true) => "q35_gemm_q1_ha",
            (Self::F32(_), _) => "q35_gemm_f32",
        }
    }
}

/// `y = W x` (or `y += W x`) for `t_len` columns through the tiled GEMM:
/// the GEMV's buffers and scalars, a `[ceil(rows / 64), ceil(t_len / 64)]`
/// grid of 128-thread threadgroups.
pub(super) fn encode_gemm(
    enc: &ComputeCommandEncoderRef,
    m: &DeviceMatrix,
    x: (&Buffer, u64),
    y: (&Buffer, u64),
    t_len: usize,
    accumulate: bool,
) {
    enc.set_compute_pipeline_state(&m.gemm_pso);
    enc.set_buffer(0, Some(&m.buffer), m.offset);
    enc.set_buffer(1, Some(x.0), x.1);
    enc.set_buffer(2, Some(y.0), y.1);
    set_u32(enc, 3, m.rows as u32);
    set_u32(enc, 4, m.cols as u32);
    set_u32(enc, 5, t_len as u32);
    set_u32(enc, 6, u32::from(accumulate));
    enc.dispatch_thread_groups(
        MTLSize::new(
            m.rows.div_ceil(GEMM_TILE) as u64,
            t_len.div_ceil(GEMM_TILE) as u64,
            1,
        ),
        MTLSize::new(GEMM_THREADS, 1, 1),
    );
}

/// Whether a call of `t_len` tokens runs its projections through the GEMM
/// under `mode`, the GEMM taking calls of at least `min_cols` tokens.
#[must_use]
pub(super) fn uses_gemm(mode: Qwen35PrefillMode, t_len: usize, min_cols: usize) -> bool {
    mode == Qwen35PrefillMode::Batched && t_len >= min_cols
}

impl Qwen35GpuModel<'_> {
    /// How multi-token calls run their projections (see the module docs).
    #[must_use]
    pub fn prefill_mode(&self) -> Qwen35PrefillMode {
        self.prefill_mode
    }

    /// Select how multi-token calls run their projections. Takes effect on
    /// the next call; decode (one token) is the GEMV in both modes.
    pub fn set_prefill_mode(&mut self, mode: Qwen35PrefillMode) {
        self.prefill_mode = mode;
    }

    /// Tokens a call needs before [`Qwen35PrefillMode::Batched`] runs its
    /// projections on the GEMM ([`Q35_GEMM_MIN_COLS`] unless changed).
    #[must_use]
    pub fn gemm_min_cols(&self) -> usize {
        self.gemm_min_cols
    }

    /// Change the call size from which [`Qwen35PrefillMode::Batched`] takes
    /// the GEMM — e.g. `2` to put every multi-token call on it, so a parity
    /// check exercises the GEMM on a short prompt. Decode (one token) stays
    /// on the GEMV.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] below 2; nothing changes then.
    pub fn set_gemm_min_cols(&mut self, min_cols: usize) -> Result<(), MetalGraphError> {
        if min_cols < 2 {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: the GEMM threshold {min_cols} would put decode on the GEMM (at least \
                 2)"
            )));
        }
        self.gemm_min_cols = min_cols;
        Ok(())
    }

    /// `y = W x` (or `y += W x`) for `t_len` columns: the GEMM when the
    /// prefill mode and the call size select it, the decode GEMV otherwise.
    pub(super) fn encode_matmul(
        &self,
        enc: &ComputeCommandEncoderRef,
        m: &DeviceMatrix,
        x: (&Buffer, u64),
        y: (&Buffer, u64),
        t_len: usize,
        accumulate: bool,
    ) {
        if uses_gemm(self.prefill_mode, t_len, self.gemm_min_cols) {
            encode_gemm(enc, m, x, y, t_len, accumulate);
        } else {
            encode_gemv(enc, m, x, y, t_len, accumulate);
        }
    }

    /// Change the most tokens one call takes to `max_batch`.
    ///
    /// The activation scratch grows to the new batch at the next call that
    /// needs it (and is released now when the batch shrinks); the resident
    /// footprint changes by `max_batch × Scratch::floats_per_token` floats,
    /// which is why the batch is part of the device-capacity bound.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a zero batch, or one whose
    /// scratch would push the model's KV window past this device's capacity
    /// (the message names both numbers); nothing changes then.
    pub fn set_max_batch(&mut self, max_batch: usize) -> Result<(), MetalGraphError> {
        if max_batch == 0 {
            return Err(MetalGraphError::InvalidDimensions(
                "qwen35 GPU: max_batch must be at least 1".to_string(),
            ));
        }
        let mut cfg = self.cfg.clone();
        cfg.max_batch = max_batch;
        let n_full = self.layer_kv_slot.iter().filter(|s| s.is_some()).count();
        let n_linear = self.layer_rec_slot.iter().filter(|s| s.is_some()).count();
        let device = &self.graph.device;
        let cap = qwen35_context_capacity(
            &cfg,
            n_full,
            n_linear,
            self.weight_bytes,
            device.max_buffer_length(),
            device.recommended_max_working_set_size(),
        );
        if cfg.max_seq_len > cap {
            return Err(MetalGraphError::InvalidDimensions(format!(
                "qwen35 GPU: a {max_batch}-token batch leaves room for {cap} KV positions on this \
                 device, below the model's window of {} (current batch {})",
                cfg.max_seq_len, self.cfg.max_batch
            )));
        }
        if max_batch < self.scratch.capacity {
            self.scratch = Scratch::allocate(&self.graph, &cfg, 1)?;
        }
        self.cfg = cfg;
        Ok(())
    }
}
