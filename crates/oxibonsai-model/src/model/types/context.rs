//! On-demand growth of the host caches (sec-11, M-07),
//! split out of `model/types/mod.rs`.
//!
//! # What grows, and what never changes
//!
//! * The host KV cache is a **lazy** `f16` cache whose logical limit —
//!   [`KvCache::max_seq_len`](crate::kv_cache::KvCache::max_seq_len) — is the
//!   model's effective context, fixed at construction. Only its *allocation*
//!   grows, through [`KvCache::try_ensure_capacity`](crate::kv_cache::KvCache::try_ensure_capacity)
//!   (whole chunks, doubling once large).
//! * Every device-side KV cache (the fused Metal / CUDA paths, the per-block
//!   Metal layer) is sized from that fixed `max_seq_len()`, so host growth
//!   can never re-geometry a device cache mid-sequence. The old rule "a
//!   GGUF-loaded model may not grow, and nothing may grow once a GPU path
//!   ran" existed only because the device geometry used to be read from the
//!   host *allocation*; it is gone, and so is the bespoke rebuild-and-copy
//!   `grow_context` it needed.
//! * The RoPE table of a GGUF-loaded model covers the whole effective
//!   context from load. The config-only constructors (M-33) start it small;
//!   it grows here, with the cache, through the scaling-aware builder.

use super::BonsaiModel;
use crate::error::{ModelError, ModelResult};
use crate::model::weight_loaders::build_rope_table;

impl BonsaiModel<'_> {
    /// Ensure the caches can serve position `pos`: the decode loop's
    /// `try_ensure_capacity(pos + 1)` (lazy KV growth), plus RoPE rows when
    /// a config-only model's table is shorter.
    ///
    /// Called before any layer runs, so an allocation failure is a typed
    /// error with nothing half-written.
    ///
    /// # Errors
    ///
    /// [`ModelError::SequenceTooLong`] when `pos` is beyond the effective
    /// context; [`ModelError::KvAllocation`] when the KV growth cannot be
    /// allocated; [`ModelError::RopeScaling`] from a RoPE rebuild.
    pub(super) fn ensure_context_capacity(&mut self, pos: usize) -> ModelResult<()> {
        if pos >= self.max_context {
            return Err(ModelError::SequenceTooLong {
                seq_len: pos + 1,
                max_ctx: self.max_context,
            });
        }
        let before = self.kv_cache.allocated_seq_len();
        self.kv_cache.try_ensure_capacity(pos + 1)?;
        if self.rope.max_seq_len() <= pos {
            let rows = self
                .kv_cache
                .allocated_seq_len()
                .max(pos + 1)
                .min(self.max_context);
            // M-08: rebuild through the same scaling-aware path the
            // constructors use — a plain `RopeTable::new` would silently drop
            // YaRN the moment a sequence outgrew the first allocation.
            self.rope = build_rope_table(&self.config, rows)?;
        }
        let after = self.kv_cache.allocated_seq_len();
        if after != before {
            tracing::debug!(
                pos,
                before,
                after,
                limit = self.kv_cache.max_seq_len(),
                resident_bytes = self.kv_cache.memory_bytes(),
                "grew the host KV cache on demand"
            );
        }
        Ok(())
    }
}
