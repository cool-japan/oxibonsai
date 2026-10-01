//! What a [`Qwen35GpuModel`] keeps resident on the device, and how its
//! per-sequence state leaves and re-enters it.
//!
//! # Footprint
//!
//! [`qwen35_footprint`] splits everything the encoder allocates besides the
//! weights into a part that does not depend on the KV window (the
//! Gated-DeltaNet state, the conv windows, the logits row and the
//! activation scratch at `max_batch` tokens) and a per-position part (the
//! `f16` K + V cache over every full-attention slot, the rope angles and
//! one row of attention scores). [`qwen35_context_capacity`] and every
//! caller that budgets a window — the engine that keeps a CPU model beside
//! the runner, `oxibonsai info` — read the same two numbers, so the
//! arithmetic exists once.
//!
//! # Recurrent state snapshots
//!
//! The recurrent state lives in shared-storage buffers. Every call that
//! encodes GPU work waits for its command buffer before it returns, so
//! between calls no work is in flight and the state can be copied out and
//! back with plain host copies: [`Qwen35GpuModel::snapshot_state`] /
//! [`Qwen35GpuModel::restore_state`]. The KV cache is not part of a
//! snapshot: a position is always written before any query at or after it
//! reads it, so a sequence rolled back to position `p` only ever writes at
//! `p` or later and the stored keys and values below `p` stay valid.

use super::{
    qwen35_context_capacity, read_buffer, MetalGraph, MetalGraphError, Qwen35GpuConfig,
    Qwen35GpuModel, Scratch,
};

/// Bytes of one `f32`.
const F32_BYTES: u64 = 4;
/// Bytes of one `f16`.
const F16_BYTES: u64 = 2;
/// Past samples each conv window holds (the 4-tap kernel's three).
const CONV_TAPS_KEPT: u64 = 3;

/// What a [`Qwen35GpuModel`] of one geometry keeps resident besides its
/// weights (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Footprint {
    /// Bytes independent of the KV window: the recurrent state, the logits
    /// row and the activation scratch at `max_batch` tokens (the most one
    /// forward grows it to).
    pub fixed_bytes: u64,
    /// The recurrent-state share of [`Self::fixed_bytes`]: every
    /// Gated-DeltaNet slab and conv window.
    pub recurrent_bytes: u64,
    /// Bytes per KV position: the `f16` K and V of every full-attention
    /// slot, the rope angles and one row of attention scores.
    pub per_position_bytes: u64,
    /// The `f16` K + V share of [`Self::per_position_bytes`].
    pub kv_bytes_per_position: u64,
    /// One K (or V) buffer's bytes per position — what `maxBufferLength`
    /// bounds.
    pub kv_buffer_bytes_per_position: u64,
}

impl Qwen35Footprint {
    /// Bytes the model allocates for a KV window of `positions` (the
    /// per-position buffers only; add [`Self::fixed_bytes`] and the weights
    /// for the whole model).
    #[must_use]
    pub fn window_bytes(&self, positions: usize) -> u64 {
        (positions as u64).saturating_mul(self.per_position_bytes)
    }

    /// Everything the model allocates besides its weights for a KV window of
    /// `positions`.
    #[must_use]
    pub fn resident_bytes(&self, positions: usize) -> u64 {
        self.fixed_bytes
            .saturating_add(self.window_bytes(positions))
    }
}

/// The resident footprint of a model of geometry `cfg` with `n_full`
/// full-attention and `n_linear` Gated-DeltaNet layers (see
/// [`Qwen35Footprint`]).
///
/// The per-position K/V terms count at least one slot, exactly as the
/// encoder allocates at least one; the recurrent terms count the layers
/// there are.
#[must_use]
pub fn qwen35_footprint(cfg: &Qwen35GpuConfig, n_full: usize, n_linear: usize) -> Qwen35Footprint {
    let bytes = |elements: usize, width: u64| (elements as u64).saturating_mul(width);
    let slots = n_full.max(1);
    let kv_buffer_bytes_per_position = bytes(slots * cfg.n_kv_heads * cfg.head_dim, F16_BYTES);
    let kv_bytes_per_position = kv_buffer_bytes_per_position.saturating_mul(2);
    let per_position_bytes = kv_bytes_per_position
        .saturating_add(bytes(cfg.n_rot, F32_BYTES))
        .saturating_add(bytes(cfg.n_heads, F32_BYTES));
    let recurrent_bytes = bytes(
        n_linear * cfg.n_v_heads * cfg.head_v_dim * cfg.head_k_dim,
        F32_BYTES,
    )
    .saturating_add(bytes(n_linear * cfg.conv_dim(), F32_BYTES).saturating_mul(CONV_TAPS_KEPT));
    let fixed_bytes = recurrent_bytes
        .saturating_add(bytes(cfg.vocab, F32_BYTES))
        .saturating_add(
            bytes(cfg.max_batch.max(1), F32_BYTES)
                .saturating_mul(Scratch::floats_per_token(cfg) as u64),
        );
    Qwen35Footprint {
        fixed_bytes,
        recurrent_bytes,
        per_position_bytes,
        kv_bytes_per_position,
        kv_buffer_bytes_per_position,
    }
}

/// The two limits of this host's Metal device the capacity bound reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35DeviceLimits {
    /// `MTLDevice.maxBufferLength`, in bytes.
    pub max_buffer_length: u64,
    /// `MTLDevice.recommendedMaxWorkingSetSize`, in bytes.
    pub recommended_working_set: u64,
}

impl Qwen35DeviceLimits {
    /// The limits of the process-shared Metal device, opening it (and
    /// building its library) on first use.
    ///
    /// This is also the "is there a Metal device at all" probe: the runner
    /// builds its session on the same shared device.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::DeviceNotFound`] when the host has no Metal
    /// device — the only error that means that — and the library build
    /// error when the combined Metal library does not compile.
    pub fn of_shared_device() -> Result<Self, MetalGraphError> {
        let shared = MetalGraph::shared_device()?;
        Ok(Self {
            max_buffer_length: shared.device.max_buffer_length(),
            recommended_working_set: shared.device.recommended_max_working_set_size(),
        })
    }

    /// [`qwen35_context_capacity`] under these limits.
    #[must_use]
    pub fn context_capacity(
        &self,
        cfg: &Qwen35GpuConfig,
        n_full: usize,
        n_linear: usize,
        weight_bytes: u64,
    ) -> usize {
        qwen35_context_capacity(
            cfg,
            n_full,
            n_linear,
            weight_bytes,
            self.max_buffer_length,
            self.recommended_working_set,
        )
    }
}

/// A host copy of a [`Qwen35GpuModel`]'s recurrent state: every
/// Gated-DeltaNet slab and every conv window, bit for bit.
///
/// Restoring it ([`Qwen35GpuModel::restore_state`]) returns the recurrence
/// to exactly the point it was taken at; see the module docs for why the KV
/// cache needs no copy.
#[derive(Clone, PartialEq)]
pub struct Qwen35RecurrentSnapshot {
    /// Every Gated-DeltaNet slab, slot-major.
    pub(super) ssm: Vec<f32>,
    /// Every conv window, slot-major.
    pub(super) conv: Vec<f32>,
}

impl std::fmt::Debug for Qwen35RecurrentSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35RecurrentSnapshot")
            .field("ssm_floats", &self.ssm.len())
            .field("conv_floats", &self.conv.len())
            .finish()
    }
}

impl Qwen35RecurrentSnapshot {
    /// Bytes the snapshot holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        ((self.ssm.len() + self.conv.len()) as u64).saturating_mul(F32_BYTES)
    }
}

impl Qwen35GpuModel<'_> {
    /// Bytes of the device recurrent state (Gated-DeltaNet slabs and conv
    /// windows together).
    #[must_use]
    pub fn recurrent_state_bytes(&self) -> u64 {
        self.ssm_state.length() + self.conv_state.length()
    }

    /// Copy the recurrent state to the host.
    ///
    /// Takes `&self`: every call that encodes GPU work waits for its command
    /// buffer, so no work can be writing the state while a shared borrow of
    /// the model exists.
    #[must_use]
    pub fn snapshot_state(&self) -> Qwen35RecurrentSnapshot {
        let floats = |buf: &metal::Buffer| (buf.length() / F32_BYTES) as usize;
        Qwen35RecurrentSnapshot {
            ssm: read_buffer(&self.ssm_state, 0, floats(&self.ssm_state)),
            conv: read_buffer(&self.conv_state, 0, floats(&self.conv_state)),
        }
    }

    /// Put a [`Self::snapshot_state`] copy back on the device.
    ///
    /// # Errors
    ///
    /// [`MetalGraphError::InvalidDimensions`] for a snapshot of another
    /// geometry (its slabs or windows are not this model's size); nothing is
    /// written then.
    pub fn restore_state(
        &mut self,
        snapshot: &Qwen35RecurrentSnapshot,
    ) -> Result<(), MetalGraphError> {
        for (what, buf, values) in [
            ("ssm_state", &self.ssm_state, &snapshot.ssm),
            ("conv_state", &self.conv_state, &snapshot.conv),
        ] {
            let want = (buf.length() / F32_BYTES) as usize;
            if values.len() != want {
                return Err(MetalGraphError::InvalidDimensions(format!(
                    "qwen35 GPU: a {what} snapshot of {} floats cannot restore a {want}-float \
                     state",
                    values.len()
                )));
            }
        }
        for (buf, values) in [
            (&self.ssm_state, &snapshot.ssm),
            (&self.conv_state, &snapshot.conv),
        ] {
            // SAFETY: `buf` is a shared-storage buffer of exactly
            // `values.len()` floats (checked above), owned by this model;
            // `&mut self` excludes every other use of it, and every forward
            // waited for its command buffer, so no GPU work is in flight.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buf.contents().cast::<f32>(),
                    values.len(),
                );
            }
        }
        Ok(())
    }

    /// Bytes the Metal device reports as currently allocated by this
    /// process (`MTLDevice.currentAllocatedSize`) — every session's buffers,
    /// including no-copy buffers over a file mapping.
    #[must_use]
    pub fn device_allocated_bytes(&self) -> u64 {
        self.graph.device.current_allocated_size()
    }

    /// The resident footprint of this model ([`qwen35_footprint`] of its
    /// own geometry and layer split).
    #[must_use]
    pub fn footprint(&self) -> Qwen35Footprint {
        let n_full = self.layer_kv_slot.iter().filter(|s| s.is_some()).count();
        let n_linear = self.layer_rec_slot.iter().filter(|s| s.is_some()).count();
        qwen35_footprint(&self.cfg, n_full, n_linear)
    }
}
