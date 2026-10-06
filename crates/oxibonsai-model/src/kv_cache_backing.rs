//! [`KvCachePolicy`] and [`KvCacheBacking`]: the storage-format vocabulary of
//! the KV cache, split out of `kv_cache.rs` (that file had
//! reached the 2000-line ceiling). Re-exported from [`crate::kv_cache`], so
//! every existing `crate::kv_cache::KvCacheBacking` path is unchanged.

use half::f16;

/// Policy for KV cache storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCachePolicy {
    /// Standard FP32 cache (contiguous allocation).
    #[default]
    Standard,
    /// FP16 cache (half the memory of Standard).
    Fp16,
    /// Sliding window cache: only retain the most recent N positions.
    SlidingWindow(usize),
}

/// Selects which concrete backing a [`crate::model::BonsaiModel`] should
/// build/use for its KV cache (RT-14 / M-13).
///
/// This is the data half of the "ideal fix" the M-13/RT-14 review
/// describes: the sibling `oxibonsai-runtime` crate's
/// `kv_cache_policy` module (not a dependency of this crate, so named here
/// only in prose, not as a doc link) has a `KvCachePolicy` whose
/// `observe`/`with_action`/`set_action` already fire a real callback on
/// every genuine tier transition (see that module's docs) — what is still
/// missing is a `BonsaiModel::set_kv_backing(&mut self, backing:
/// KvCacheBacking)` seam for that callback to call, which would require
/// changing the block-forward signature across `block/types/forward.rs`,
/// `forward_metal.rs` and `forward_cuda/*` — a partial wire-up would not be
/// the real fix, so none is attempted ("this is real work, not a one-line
/// wire-up").
/// `oxibonsai_runtime::kv_cache_policy` implements `From<KvCacheLevel> for
/// KvCacheBacking` so that, once such a seam exists, wiring it is exactly
/// `policy.set_action(move |level| model.set_kv_backing(level.into()))`.
///
/// Variants mirror that crate's `KvCacheLevel`'s four pressure-driven tiers
/// 1:1 (`Fp16`/`Q8`/`Fp8`/`Q4`, each `Dense*`
/// because every existing standalone implementation of that tier —
/// [`KvCacheFp16`](crate::kv_cache_fp16::KvCacheFp16),
/// [`kv_cache_quant::QuantizedKvCache`](crate::kv_cache_quant::QuantizedKvCache),
/// [`kv_cache_quant::Fp8KvCache`](crate::kv_cache_quant::Fp8KvCache) — is
/// dense), plus the two variants a *dynamic* precision policy cannot
/// recommend because they are load-time, architecture-driven facts rather
/// than pressure-driven choices: [`Self::DenseF32`] (full precision, one
/// tier *above* the policy's own `Fp16` baseline — what callers wanting an
/// exact `f32` host cache ask for) and [`Self::SparseF16`] (M-07 — only
/// reachable for a hybrid model, which `KvCacheLevel` has no notion of).
/// `is_sparse`/precision are therefore
/// deliberately coupled one level too coarsely for a hybrid model running a
/// *dynamically re-quantized sparse* cache to be representable — that
/// combination does not exist anywhere in this codebase yet either, so
/// `KvCacheBacking` does not claim to model it; widen this enum (or split it
/// into an orthogonal `{layout, precision}` pair) when it does.
///
/// # What the models build
///
/// * `BonsaiModel` (dense transformer): [`Self::DenseF16`], **bounded and
///   lazily allocated** — every constructor (`from_gguf`, the config-only
///   `new*` seams and the test fixture) builds it through
///   [`KvCache::try_new_lazy`](crate::kv_cache::KvCache::try_new_lazy) /
///   [`KvCache::new_lazy_f16`](crate::kv_cache::KvCache::new_lazy_f16) with
///   the effective context as the logical limit, and the decode / prefill
///   loops grow it chunk by chunk through
///   [`KvCache::try_ensure_capacity`](crate::kv_cache::KvCache::try_ensure_capacity).
///   This matches the policy's own `Fp16` baseline, so the host cache and
///   the policy now agree on where "no pressure" sits.
/// * `HybridModel` (Bonsai 2): [`Self::SparseF16`] by default (bounded and
///   lazy the same way), or [`Self::DenseF32`] when `KvPrecision::F32` is
///   requested (the arithmetic-isolating parity gates use that) — built
///   with one slot per KV-bearing layer either way, since the hybrid model
///   maps its full-attention layers to slots itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KvCacheBacking {
    /// Dense `f32` storage for every transformer layer — [`KvCache::new`](crate::kv_cache::KvCache::new) /
    /// [`KvCache::try_new`](crate::kv_cache::KvCache::try_new), or lazily via
    /// [`KvCache::try_new_lazy`](crate::kv_cache::KvCache::try_new_lazy). Full precision; no model
    /// constructor defaults to it any more.
    DenseF32,
    /// Dense, `f16`-per-element storage — half the memory of
    /// [`Self::DenseF32`] at equal geometry. Mirrors
    /// `KvCacheLevel::Fp16`. The host KV cache every `BonsaiModel`
    /// constructor builds (bounded, lazily allocated).
    DenseF16,
    /// Dense, INT8-quantized storage. Mirrors `KvCacheLevel::Q8`.
    DenseQ8,
    /// Dense, FP8-quantized storage. Mirrors `KvCacheLevel::Fp8`.
    DenseFp8,
    /// Dense, INT4-quantized storage. Mirrors `KvCacheLevel::Q4`.
    DenseQ4,
    /// Sparse storage: only the KV-bearing layers, `f16` per element —
    /// [`KvCache::new_sparse`](crate::kv_cache::KvCache::new_sparse) / [`KvCache::try_new_sparse`](crate::kv_cache::KvCache::try_new_sparse) (M-07). The
    /// backing a hybrid model (Bonsai 2) needs; not reachable from
    /// [`From<KvCacheLevel>`](#impl-From<KvCacheLevel>-for-KvCacheBacking)
    /// since the policy has no notion of "this model is hybrid".
    SparseF16,
}

impl KvCacheBacking {
    /// Bytes per **code** element this backing stores — `f32`/`f16` are
    /// exact (no separate scale exists); `Q8`/`Fp8`/`Q4` report their code
    /// width only (1, 1, and 1-rounded-up-from-0.5 bytes respectively) and
    /// omit each format's shared per-block scale, which is amortized across
    /// many elements rather than a fixed per-element cost.
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::DenseF32 => std::mem::size_of::<f32>(),
            Self::DenseF16 | Self::SparseF16 => std::mem::size_of::<f16>(),
            Self::DenseQ8 | Self::DenseFp8 | Self::DenseQ4 => 1,
        }
    }

    /// Whether this backing allocates only the KV-bearing layers
    /// ([`Self::SparseF16`]) rather than every transformer layer.
    pub const fn is_sparse(self) -> bool {
        matches!(self, Self::SparseF16)
    }
}
