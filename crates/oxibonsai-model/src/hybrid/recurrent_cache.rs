//! Per-sequence recurrent state for every Gated-DeltaNet layer of a hybrid
//! (`qwen35`) stack — the analogue of the KV cache for the 48 linear layers
//! (M-05, design §3.6).
//!
//! # What is held
//!
//! Per linear-attention layer:
//!
//! * the **Gated-DeltaNet state** `S`, `[n_v_heads][head_v_dim][head_k_dim]`
//!   f32 in **grouped** v-head order (design §3.3 — the order `ssm_out`
//!   consumes, so the kernel's output needs no scatter). 27B: 48 × 128 × 128
//!   = 786 432 floats = 3 MiB per layer;
//! * the **causal conv window**, `[conv_dim][conv_kernel - 1]` f32,
//!   channel-major, oldest tap first — exactly the layout
//!   `oxibonsai_kernels::ssm_ops::causal_conv1d_k4_decode` updates in place.
//!   27B: 10 240 × 3 = 30 720 floats = 120 KiB per layer.
//!
//! Allocated **once per sequence**: for the 27B the state is ~150 MB, so a
//! per-token allocation would dominate decode.
//!
//! # Why there is no `truncate(pos)`
//!
//! A KV cache rolls back by moving a cursor. A recurrence cannot: `S` after
//! `n` tokens is not recoverable from `S` after `n + k`. [`RecurrentCache`]
//! therefore offers [`RecurrentCache::snapshot`]/[`RecurrentCache::restore`]
//! (one deep copy, ~150 MB — fine for a single speculative-decode rollback
//! window) and refuses [`RecurrentCache::truncate`] outright with
//! [`ModelError::RecurrentRollbackUnsupported`]. The last-K checkpoint ring
//! that would make arbitrary rollback cheap is design §8.4 item 2.
//!
//! # RT-28
//!
//! [`RecurrentCache::reset`] is the runtime reset seam. The runtime's
//! `RecurrentState` trait lives in `oxibonsai-runtime`, which depends on this
//! crate, so the `impl` itself cannot live here (orphan rule); the three
//! methods it needs are [`RecurrentCache::reset`],
//! [`RecurrentCache::memory_bytes`] and [`RECURRENT_NAME`], and the
//! forwarding impl is a five-line addition to
//! `oxibonsai-runtime/src/engine_control.rs` (recorded as this package's
//! deviation).

use oxibonsai_core::config_hybrid::HybridConfig;
use oxibonsai_kernels::gated_delta_net::GdnDims;

use crate::error::{ModelError, ModelResult};

/// `RecurrentState::recurrent_name` for this cache.
pub const RECURRENT_NAME: &str = "gated-delta-net";

/// Per-sequence recurrent state for all Gated-DeltaNet layers of one model.
///
/// Indexed by **`rec_slot`** (`0..n_linear_layers`), not by `layer_idx`:
/// only 48 of the 27B's 64 layers are recurrent, and slotting avoids
/// allocating 16 unused slabs.
#[derive(Debug, Clone)]
pub struct RecurrentCache {
    /// `[n_linear_layers][n_v_heads * head_v_dim * head_k_dim]`, grouped
    /// v-head order.
    ssm: Vec<Vec<f32>>,
    /// `[n_linear_layers][conv_dim * (conv_kernel - 1)]`, channel-major,
    /// oldest tap first, in the GGUF's **raw (tiled)** channel order --
    /// NOT the grouped v-head order `ssm` above uses. The conv runs over
    /// all `conv_dim` channels of the concatenated `q|k|v` stream
    /// untouched; only the `v` *slice* of its output is re-indexed
    /// tiled -> grouped afterwards (design SS3.3).
    conv: Vec<Vec<f32>>,
    /// Kernel geometry of one layer's `S`.
    dims: GdnDims,
    /// Channels the depthwise conv runs over (27B: 10 240).
    conv_dim: usize,
    /// History taps kept per channel (`conv_kernel - 1`; 27B: 3).
    conv_taps: usize,
    /// Tokens folded into this state since the last [`RecurrentCache::reset`].
    tokens: usize,
}

/// A deep copy of a [`RecurrentCache`], for a single rollback point.
///
/// ~150 MB for the 27B — take one, not a ring (design §3.6: the prefix cache
/// refuses hybrid models in v1 for exactly this reason).
#[derive(Debug, Clone)]
pub struct RecurrentSnapshot {
    ssm: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
    tokens: usize,
}

impl RecurrentSnapshot {
    /// Tokens consumed into the state at the moment of the snapshot.
    #[inline]
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.tokens
    }

    /// Recurrent layers captured.
    #[inline]
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.ssm.len()
    }
}

impl RecurrentCache {
    /// Allocate a zeroed cache for every Gated-DeltaNet layer of `config`.
    ///
    /// A zeroed state is exactly the start-of-sequence state (an empty
    /// prefix contributes nothing to the recurrence), so no separate "primed"
    /// step is needed.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] if the geometry is degenerate (a zero
    /// dimension, or `n_v_heads` not a multiple of `n_k_heads`) or if the
    /// allocation size overflows `usize`.
    pub fn new(config: &HybridConfig) -> ModelResult<Self> {
        let dims = GdnDims::new(
            config.n_k_heads(),
            config.n_v_heads(),
            config.head_k_dim(),
            config.head_v_dim(),
        );
        dims.validate().map_err(ModelError::Kernel)?;
        let conv_dim = config.conv_dim();
        let conv_taps =
            config
                .ssm_conv_kernel
                .checked_sub(1)
                .ok_or_else(|| ModelError::ShapeInvariant {
                    tensor: "qwen35.ssm.conv_kernel".to_string(),
                    expected: ">= 1".to_string(),
                    actual: config.ssm_conv_kernel.to_string(),
                })?;
        if conv_dim == 0 || conv_taps == 0 {
            return Err(ModelError::ShapeInvariant {
                tensor: "RecurrentCache conv window".to_string(),
                expected: "conv_dim > 0 and conv_kernel > 1".to_string(),
                actual: format!(
                    "conv_dim = {conv_dim}, conv_kernel = {}",
                    config.ssm_conv_kernel
                ),
            });
        }
        let n_layers = config.num_linear_layers();
        let state_len = dims.state_len();
        let conv_len = conv_dim
            .checked_mul(conv_taps)
            .ok_or_else(|| Self::overflow("conv_dim * (conv_kernel - 1)"))?;
        // Prove the whole allocation is representable before touching the
        // allocator, so a bad header reports a shape error instead of an
        // abort inside `vec!`.
        for (what, per_layer) in [("ssm", state_len), ("conv", conv_len)] {
            per_layer
                .checked_mul(n_layers)
                .and_then(|n| n.checked_mul(core::mem::size_of::<f32>()))
                .ok_or_else(|| Self::overflow(what))?;
        }
        Ok(Self {
            ssm: vec![vec![0.0; state_len]; n_layers],
            conv: vec![vec![0.0; conv_len]; n_layers],
            dims,
            conv_dim,
            conv_taps,
            tokens: 0,
        })
    }

    fn overflow(what: &str) -> ModelError {
        ModelError::ShapeInvariant {
            tensor: format!("RecurrentCache {what}"),
            expected: "allocation size representable as usize".to_string(),
            actual: "overflow".to_string(),
        }
    }

    /// Recurrent (linear-attention) layers held.
    #[inline]
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.ssm.len()
    }

    /// Kernel geometry of one layer's Gated-DeltaNet state.
    #[inline]
    #[must_use]
    pub fn dims(&self) -> GdnDims {
        self.dims
    }

    /// Channels of the depthwise causal conv (27B: 10 240).
    #[inline]
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        self.conv_dim
    }

    /// History taps kept per conv channel (`conv_kernel - 1`; 27B: 3).
    #[inline]
    #[must_use]
    pub fn conv_taps(&self) -> usize {
        self.conv_taps
    }

    /// Tokens folded into this state since the last [`RecurrentCache::reset`].
    #[inline]
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.tokens
    }

    /// Record that `n` more tokens have been folded into the state.
    ///
    /// Saturates rather than wrapping: the count drives diagnostics and the
    /// [`ModelError::RecurrentRollbackUnsupported`] message, never indexing.
    #[inline]
    pub fn advance(&mut self, n: usize) {
        self.tokens = self.tokens.saturating_add(n);
    }

    /// Zero every tensor — the start-of-sequence state (RT-28 seam).
    ///
    /// Idempotent, and keeps the allocation: the engine calls this on every
    /// request reset, including for a sequence that never ran.
    pub fn reset(&mut self) {
        for slab in &mut self.ssm {
            slab.fill(0.0);
        }
        for window in &mut self.conv {
            window.fill(0.0);
        }
        self.tokens = 0;
    }

    /// One layer's Gated-DeltaNet state, grouped v-head order.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] naming the slot when `slot` is out of
    /// range. (Design §3.6 sketches an infallible accessor; a fallible one
    /// is used here for the same reason `GdnState::layer_mut` is fallible —
    /// this crate does not panic on a caller's index.)
    pub fn ssm(&self, slot: usize) -> ModelResult<&[f32]> {
        self.ssm
            .get(slot)
            .map(Vec::as_slice)
            .ok_or_else(|| self.bad_slot("ssm", slot))
    }

    /// Mutable form of [`RecurrentCache::ssm`] — what the Gated-DeltaNet
    /// kernel updates in place.
    ///
    /// # Errors
    ///
    /// As [`RecurrentCache::ssm`].
    pub fn ssm_mut(&mut self, slot: usize) -> ModelResult<&mut [f32]> {
        let n_layers = self.ssm.len();
        self.ssm
            .get_mut(slot)
            .map(Vec::as_mut_slice)
            .ok_or_else(|| Self::slot_error("ssm", slot, n_layers))
    }

    /// One layer's conv window, channel-major, oldest tap first.
    ///
    /// # Errors
    ///
    /// As [`RecurrentCache::ssm`].
    pub fn conv(&self, slot: usize) -> ModelResult<&[f32]> {
        self.conv
            .get(slot)
            .map(Vec::as_slice)
            .ok_or_else(|| self.bad_slot("conv", slot))
    }

    /// Mutable form of [`RecurrentCache::conv`] — what
    /// `causal_conv1d_k4_decode` shifts in place.
    ///
    /// # Errors
    ///
    /// As [`RecurrentCache::ssm`].
    pub fn conv_mut(&mut self, slot: usize) -> ModelResult<&mut [f32]> {
        let n_layers = self.conv.len();
        self.conv
            .get_mut(slot)
            .map(Vec::as_mut_slice)
            .ok_or_else(|| Self::slot_error("conv", slot, n_layers))
    }

    /// Both of one layer's tensors at once (`(ssm, conv)`), which a layer
    /// forward needs simultaneously and cannot get from two `&mut self`
    /// calls.
    ///
    /// # Errors
    ///
    /// As [`RecurrentCache::ssm`].
    pub fn layer_mut(&mut self, slot: usize) -> ModelResult<(&mut [f32], &mut [f32])> {
        let n_layers = self.ssm.len();
        let ssm = self
            .ssm
            .get_mut(slot)
            .map(Vec::as_mut_slice)
            .ok_or_else(|| Self::slot_error("ssm", slot, n_layers))?;
        let conv = self
            .conv
            .get_mut(slot)
            .map(Vec::as_mut_slice)
            .ok_or_else(|| Self::slot_error("conv", slot, n_layers))?;
        Ok((ssm, conv))
    }

    fn bad_slot(&self, what: &str, slot: usize) -> ModelError {
        Self::slot_error(what, slot, self.ssm.len())
    }

    fn slot_error(what: &str, slot: usize, n_layers: usize) -> ModelError {
        ModelError::ShapeMismatch {
            name: format!("RecurrentCache {what} slot"),
            expected: vec![n_layers],
            actual: vec![slot],
        }
    }

    /// Deep copy of the whole state, for a single rollback point.
    #[must_use]
    pub fn snapshot(&self) -> RecurrentSnapshot {
        RecurrentSnapshot {
            ssm: self.ssm.clone(),
            conv: self.conv.clone(),
            tokens: self.tokens,
        }
    }

    /// Restore a [`RecurrentCache::snapshot`] taken from **this** cache's
    /// geometry.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] when the snapshot's layer count or any
    /// slab length differs — a snapshot from another model must not be
    /// silently truncated into this one.
    pub fn restore(&mut self, snapshot: &RecurrentSnapshot) -> ModelResult<()> {
        if snapshot.ssm.len() != self.ssm.len() || snapshot.conv.len() != self.conv.len() {
            return Err(ModelError::ShapeMismatch {
                name: "RecurrentSnapshot layers".to_string(),
                expected: vec![self.ssm.len(), self.conv.len()],
                actual: vec![snapshot.ssm.len(), snapshot.conv.len()],
            });
        }
        for (slot, (dst, src)) in self.ssm.iter_mut().zip(snapshot.ssm.iter()).enumerate() {
            if dst.len() != src.len() {
                return Err(ModelError::ShapeMismatch {
                    name: format!("RecurrentSnapshot ssm slot {slot}"),
                    expected: vec![dst.len()],
                    actual: vec![src.len()],
                });
            }
            dst.copy_from_slice(src);
        }
        for (slot, (dst, src)) in self.conv.iter_mut().zip(snapshot.conv.iter()).enumerate() {
            if dst.len() != src.len() {
                return Err(ModelError::ShapeMismatch {
                    name: format!("RecurrentSnapshot conv slot {slot}"),
                    expected: vec![dst.len()],
                    actual: vec![src.len()],
                });
            }
            dst.copy_from_slice(src);
        }
        self.tokens = snapshot.tokens;
        Ok(())
    }

    /// Always an error: a recurrence cannot be rolled back by moving a
    /// cursor (M-05).
    ///
    /// `pos == 0` is refused too, deliberately: routing it to
    /// [`RecurrentCache::reset`] would make "truncate to 0" quietly work and
    /// "truncate to 1" quietly fail, which is precisely the kind of
    /// position-dependent behaviour that hides a rollback bug until it
    /// reaches a long sequence. Callers that mean "start over" call `reset`.
    ///
    /// # Errors
    ///
    /// Always [`ModelError::RecurrentRollbackUnsupported`].
    pub fn truncate(&mut self, pos: usize) -> ModelResult<()> {
        Err(ModelError::RecurrentRollbackUnsupported {
            pos,
            tokens: self.tokens,
        })
    }

    /// Bytes held by the Gated-DeltaNet states.
    ///
    /// 27B: 48 linear layers × 48 v-heads × 128 × 128 × 4 B =
    /// **150 994 944** (144 MiB), the same figure as
    /// `GdnState::with_layers(GdnDims::bonsai2(), 48).bytes()` and design
    /// §3.6's "144 MiB".
    #[inline]
    #[must_use]
    pub fn ssm_bytes(&self) -> usize {
        self.ssm.len() * self.dims.state_len() * core::mem::size_of::<f32>()
    }

    /// Bytes held by the causal-conv windows.
    ///
    /// 27B: 48 × 10 240 × 3 × 4 B = **5 898 240** (5.625 MiB).
    #[inline]
    #[must_use]
    pub fn conv_bytes(&self) -> usize {
        self.conv.len() * self.conv_dim * self.conv_taps * core::mem::size_of::<f32>()
    }

    /// Total resident bytes, `ssm_bytes() + conv_bytes()`.
    ///
    /// 27B: 150 994 944 + 5 898 240 = **156 893 184** (~157 MB), which is
    /// the "~157 MB of state" the runtime's `RecurrentState` doc quotes.
    #[inline]
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.ssm_bytes() + self.conv_bytes()
    }
}

/// `oxibonsai-runtime`'s `RecurrentState` trait requires `Send` (the engine
/// pool moves replicas across threads with `spawn_blocking`), and the
/// forwarding impl has to live in that crate because of the orphan rule.
/// Assert the bound here so a field added to [`RecurrentCache`] that breaks
/// it fails in this package rather than in someone else's.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<RecurrentCache>();
    assert_send::<RecurrentSnapshot>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::tests_support::bonsai2_config;

    #[test]
    fn bonsai2_geometry_and_exact_memory() {
        let config = bonsai2_config();
        let cache = RecurrentCache::new(&config).expect("valid config");

        assert_eq!(cache.n_layers(), 48, "48 of the 64 layers are recurrent");
        assert_eq!(cache.dims(), GdnDims::bonsai2());
        assert_eq!(cache.conv_dim(), 10_240);
        assert_eq!(cache.conv_taps(), 3);

        // Derived, not hard-coded: 48 layers x 48 v-heads x 128 x 128 x 4 B.
        assert_eq!(cache.ssm_bytes(), 48 * 48 * 128 * 128 * 4);
        assert_eq!(cache.ssm_bytes(), 150_994_944);
        // 48 layers x 10240 channels x 3 taps x 4 B.
        assert_eq!(cache.conv_bytes(), 48 * 10_240 * 3 * 4);
        assert_eq!(cache.conv_bytes(), 5_898_240);
        assert_eq!(cache.memory_bytes(), 156_893_184);

        // Cross-check the state half against the kernels crate's own
        // allocation for the same model, so this figure is pinned to another
        // package's arithmetic rather than to this file's comment.
        let kernel_state =
            oxibonsai_kernels::gated_delta_net::GdnState::with_layers(GdnDims::bonsai2(), 48)
                .expect("valid dims");
        assert_eq!(cache.ssm_bytes(), kernel_state.bytes());
    }

    #[test]
    fn slabs_are_zeroed_and_reset_restores_that() {
        let config = bonsai2_config();
        let mut cache = RecurrentCache::new(&config).expect("valid config");
        assert!(cache.ssm(0).expect("slot 0").iter().all(|v| *v == 0.0));
        assert!(cache.conv(47).expect("slot 47").iter().all(|v| *v == 0.0));

        cache.ssm_mut(5).expect("slot 5")[17] = 3.5;
        cache.conv_mut(5).expect("slot 5")[2] = -1.25;
        cache.advance(12);
        assert_eq!(cache.token_count(), 12);

        cache.reset();
        assert_eq!(cache.token_count(), 0);
        assert!(cache.ssm(5).expect("slot 5").iter().all(|v| *v == 0.0));
        assert!(cache.conv(5).expect("slot 5").iter().all(|v| *v == 0.0));
        // Idempotent.
        cache.reset();
        assert_eq!(cache.token_count(), 0);
    }

    #[test]
    fn out_of_range_slots_error_rather_than_panic() {
        let config = bonsai2_config();
        let mut cache = RecurrentCache::new(&config).expect("valid config");
        assert!(cache.ssm(48).is_err());
        assert!(cache.conv(48).is_err());
        assert!(cache.ssm_mut(usize::MAX).is_err());
        assert!(cache.conv_mut(48).is_err());
        let err = cache.layer_mut(100).expect_err("slot 100 does not exist");
        assert_eq!(err.error_code(), "SHAPE_MISMATCH");
    }

    #[test]
    fn layer_mut_hands_out_both_tensors_at_once() {
        let config = bonsai2_config();
        let mut cache = RecurrentCache::new(&config).expect("valid config");
        let (ssm, conv) = cache.layer_mut(3).expect("slot 3");
        assert_eq!(ssm.len(), 48 * 128 * 128);
        assert_eq!(conv.len(), 10_240 * 3);
        ssm[0] = 1.0;
        conv[0] = 2.0;
        assert_eq!(cache.ssm(3).expect("slot 3")[0], 1.0);
        assert_eq!(cache.conv(3).expect("slot 3")[0], 2.0);
    }

    #[test]
    fn snapshot_and_restore_round_trip() {
        let mut config = bonsai2_config();
        // Small geometry: a full 27B snapshot is 150 MB and this test only
        // needs the copy semantics.
        config.base.num_layers = 8;
        config.ssm_group_count = 2;
        config.ssm_time_step_rank = 6;
        config.ssm_inner_size = 6 * 4;
        config.ssm_state_size = 4;
        let mut cache = RecurrentCache::new(&config).expect("valid config");

        cache.ssm_mut(0).expect("slot 0")[0] = 7.0;
        cache.conv_mut(1).expect("slot 1")[3] = -2.0;
        cache.advance(9);
        let snap = cache.snapshot();
        assert_eq!(snap.token_count(), 9);
        assert_eq!(snap.n_layers(), cache.n_layers());

        cache.ssm_mut(0).expect("slot 0")[0] = 0.0;
        cache.conv_mut(1).expect("slot 1")[3] = 0.0;
        cache.advance(5);
        cache.restore(&snap).expect("same geometry");
        assert_eq!(cache.ssm(0).expect("slot 0")[0], 7.0);
        assert_eq!(cache.conv(1).expect("slot 1")[3], -2.0);
        assert_eq!(cache.token_count(), 9);
    }

    #[test]
    fn restore_rejects_a_foreign_snapshot() {
        let config = bonsai2_config();
        let mut small = config.clone();
        small.base.num_layers = 8;
        let mut cache = RecurrentCache::new(&small).expect("valid config");
        let other = {
            let mut bigger = config.clone();
            bigger.base.num_layers = 16;
            RecurrentCache::new(&bigger)
                .expect("valid config")
                .snapshot()
        };
        let err = cache
            .restore(&other)
            .expect_err("a snapshot with a different layer count must be refused");
        assert_eq!(err.error_code(), "SHAPE_MISMATCH");
    }

    #[test]
    fn truncate_is_always_an_error_including_position_zero() {
        let mut config = bonsai2_config();
        config.base.num_layers = 4;
        let mut cache = RecurrentCache::new(&config).expect("valid config");
        cache.advance(31);
        for pos in [0usize, 1, 30] {
            let err = cache
                .truncate(pos)
                .expect_err("recurrent rollback is refused");
            assert_eq!(err.error_code(), "RECURRENT_ROLLBACK_UNSUPPORTED");
            let message = err.to_string();
            assert!(message.contains(&format!("position {pos}")), "{message}");
            assert!(message.contains("31 tokens"), "{message}");
        }
        // Refusing must not have mutated anything.
        assert_eq!(cache.token_count(), 31);
    }

    #[test]
    fn rejects_a_degenerate_geometry() {
        let mut config = bonsai2_config();
        config.ssm_conv_kernel = 1;
        assert!(RecurrentCache::new(&config).is_err());

        let mut config = bonsai2_config();
        config.ssm_state_size = 0;
        assert!(RecurrentCache::new(&config).is_err());
    }
}
