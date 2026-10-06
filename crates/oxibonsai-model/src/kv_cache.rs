//! KV Cache for autoregressive generation.
//!
//! Stores key and value tensors for each layer to avoid recomputation
//! during token-by-token generation.
//!
//! # Element type, layout and lazy growth (M-07)
//!
//! A [`KvCache`] is described by three independent facts:
//!
//! * **element type** — `f32` (4 B) or `f16` (2 B), [`KvCache::is_f16`];
//! * **layout** — *dense* (one slot per transformer layer) or *sparse* (one
//!   slot per KV-bearing layer of a hybrid stack, the caller mapping
//!   `layer_idx -> kv_slot`), [`KvCache::is_sparse`];
//! * **capacity** — the **logical** limit [`KvCache::max_seq_len`] (the most
//!   positions this cache may ever hold, what every caller sizing a context
//!   or a device-side cache reads) and the **allocated** capacity
//!   [`KvCache::allocated_seq_len`] (what is resident right now).
//!
//! The legacy constructors ([`KvCache::new`], [`KvCache::try_new`],
//! [`KvCache::new_sparse`], [`KvCache::try_new_sparse`]) allocate their
//! whole limit up front, exactly as before, and [`KvCache::ensure_capacity`]
//! still raises *both* numbers on such a cache. [`KvCache::try_new_lazy`]
//! builds a **bounded** cache instead: it starts at a small allocation and
//! grows it — by whole [`GROWTH_CHUNK_POSITIONS`] chunks, doubling once it is
//! large — towards a fixed logical limit, either explicitly
//! ([`KvCache::try_ensure_capacity`], which the model decode loops call with
//! `pos + 1` before running any layer) or implicitly on a store past the
//! allocation. `max_seq_len()` never moves on a bounded cache, so a
//! device-side cache sized from it keeps its geometry while the host grows.
//!
//! # Reading back
//!
//! Attention reads the cache in place through [`KvCache::attend_group`],
//! which never materialises a history copy for either element type.
//!
//! [`KvCache::keys_for`]/[`KvCache::values_for`] keep their `&[f32]`
//! signature and are correct for every storage mode (they used to return an
//! **empty** slice for `f16` storage — the release-build-silent footgun the
//! review flagged, guarded only by a `debug_assert!`). An `f32`
//! cache is borrowed zero-copy; an `f16` cache is served from an `f32`
//! read-back mirror built on the first such read after a write and dropped
//! by the next write. They are diagnostic / inspection accessors: no hot
//! path calls them, so the mirror is never built during inference.
//!
//! The standalone page-based cache lives in [`PagedKvCache`]; the storage
//! vocabulary ([`KvCacheBacking`], [`KvCachePolicy`]) in its own file.

use std::sync::OnceLock;

use half::f16;

use crate::error::{ModelError, ModelResult};

#[path = "kv_cache_attention.rs"]
mod attention;
#[path = "kv_cache_backing.rs"]
mod backing;
#[cfg(test)]
#[path = "kv_cache_lazy_tests.rs"]
mod lazy_tests;
#[path = "kv_cache_paged.rs"]
mod paged;
#[cfg(test)]
#[path = "kv_cache_tests.rs"]
mod tests;

pub use attention::attend_f16_group;
pub use backing::{KvCacheBacking, KvCachePolicy};
pub use paged::PagedKvCache;

/// Backing element storage for a [`KvCache`].
///
/// Private: callers see the element type through [`KvCache::is_f16`] and
/// the full description through [`KvCache::backing`].
#[derive(Debug)]
enum KvStorage {
    /// One `f32` per element (4 bytes).
    F32 { keys: Vec<f32>, values: Vec<f32> },
    /// One [`f16`] per element (2 bytes): half the footprint, and the
    /// element type the PrismML fork's own goldens were produced with.
    /// Converts to/from `f32` at the store / read boundary.
    F16 { keys: Vec<f16>, values: Vec<f16> },
}

impl KvStorage {
    /// Total element count across both key and value buffers.
    fn total_elements(&self) -> usize {
        match self {
            Self::F32 { keys, values } => keys.len() + values.len(),
            Self::F16 { keys, values } => keys.len() + values.len(),
        }
    }

    /// Bytes per stored element.
    fn element_bytes(&self) -> usize {
        match self {
            Self::F32 { .. } => std::mem::size_of::<f32>(),
            Self::F16 { .. } => std::mem::size_of::<f16>(),
        }
    }

    /// Resident byte count, honest about the element width (M-07's "exactly
    /// 64 KiB/token" gate depends on this being 2 bytes/element for `f16`).
    fn memory_bytes(&self) -> usize {
        self.total_elements() * self.element_bytes()
    }

    /// Allocate zeroed storage of `total` elements per buffer, of the same
    /// element type as `self` (or `f16` when `f16` is set).
    fn try_alloc(f16_elements: bool, total: usize, requested_bytes: usize) -> ModelResult<Self> {
        if f16_elements {
            Ok(Self::F16 {
                keys: try_alloc_zeroed(total, f16::ZERO, requested_bytes)?,
                values: try_alloc_zeroed(total, f16::ZERO, requested_bytes)?,
            })
        } else {
            Ok(Self::F32 {
                keys: try_alloc_zeroed(total, 0.0f32, requested_bytes)?,
                values: try_alloc_zeroed(total, 0.0f32, requested_bytes)?,
            })
        }
    }
}

/// Which layers a cache has slots for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvLayout {
    /// One slot per transformer layer.
    Dense,
    /// One slot per KV-bearing layer of a hybrid stack (Bonsai 2: 16 of 64);
    /// the caller maps `layer_idx -> kv_slot` before every access.
    Sparse,
}

/// Per-layer KV cache storing key and value vectors — see the module docs
/// for the element-type / layout / capacity model.
#[derive(Debug)]
pub struct KvCache {
    /// Number of Transformer layers (dense) or KV slots (sparse).
    num_layers: usize,
    /// Number of KV heads per layer.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// **Logical** capacity: the most positions this cache may hold.
    max_seq_len: usize,
    /// **Allocated** capacity: positions resident right now, and the
    /// per-(layer, head) stride of the storage. `capacity <= max_seq_len`.
    capacity: usize,
    /// `true` for a [`KvCache::try_new_lazy`] cache, whose `max_seq_len` is a
    /// fixed ceiling; `false` for the legacy eager caches, which growth may
    /// enlarge past their original size.
    bounded: bool,
    /// Current sequence length (number of tokens cached).
    seq_len: usize,
    /// Slot layout (dense or sparse).
    layout: KvLayout,
    /// Backing storage (`f32` or `f16` — see [`KvStorage`]).
    storage: KvStorage,
    /// `f32` read-back mirror of an `f16` key buffer, built by the first
    /// [`KvCache::keys_for`] after a write and dropped by the next write
    /// (see the module docs). Always empty for `f32` storage.
    key_mirror: OnceLock<Vec<f32>>,
    /// Value counterpart of `key_mirror`.
    value_mirror: OnceLock<Vec<f32>>,
}

/// Growth increment, in positions, for [`KvCache::ensure_capacity`] (M-07:
/// "lazy growth in chunks rather than `max_seq_len` up front"), and the
/// default first allocation of a lazy cache. Matches the page size of
/// [`PagedKvCache`] for consistency, not because the two are otherwise
/// related.
pub const GROWTH_CHUNK_POSITIONS: usize = 256;

/// Compute `num_layers * num_kv_heads * max_seq_len * head_dim`, saturating
/// on overflow instead of panicking or wrapping (a `usize::MAX` component —
/// always attacker- or corruption-reachable from a config/GGUF value — used
/// to reach a plain `*` chain here, which panics on overflow in a debug
/// build and silently wraps to a small, plausible-looking count in release).
fn checked_element_count(
    num_layers: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
) -> Option<usize> {
    num_layers
        .checked_mul(num_kv_heads)?
        .checked_mul(max_seq_len)?
        .checked_mul(head_dim)
}

/// Allocate a `len`-element `Vec<T>` filled with `zero`, surfacing a real
/// allocator failure as a [`ModelError::KvAllocation`] instead of aborting the
/// process (M-07: `try_new`/`try_new_sparse` were
/// "fallible" only in the overflow-detecting sense — both used to finish
/// with `Self::new(...)`/`Self::new_sparse(...)`, whose plain `vec![zero;
/// len]` still calls `handle_alloc_error` and aborts on an allocator `Err`).
/// `Vec::try_reserve_exact` performs the one allocation this needs up front
/// as a `Result`; the following `resize` cannot itself need to grow the
/// buffer again since the exact capacity was just reserved, so it cannot
/// introduce a second, unchecked allocation point.
fn try_alloc_zeroed<T: Clone>(len: usize, zero: T, requested_bytes: usize) -> ModelResult<Vec<T>> {
    let mut buf: Vec<T> = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|e| ModelError::KvAllocation {
            requested_bytes: Some(requested_bytes),
            detail: format!("allocator refused {requested_bytes} bytes: {e}"),
        })?;
    buf.resize(len, zero);
    Ok(buf)
}

/// Validate a geometry and return `(elements per buffer, total bytes)` for
/// `element_bytes`-wide elements (keys + values).
fn checked_geometry(
    what: &str,
    num_layers: usize,
    num_kv_heads: usize,
    positions: usize,
    head_dim: usize,
    element_bytes: usize,
) -> ModelResult<(usize, usize)> {
    let total =
        checked_element_count(num_layers, num_kv_heads, positions, head_dim).ok_or_else(|| {
            ModelError::KvAllocation {
                requested_bytes: None,
                detail: format!(
                    "{what}{num_layers} layers x {num_kv_heads} kv_heads x \
                 {positions} max_seq_len x {head_dim} head_dim overflows usize"
                ),
            }
        })?;
    let requested_bytes = total
        .checked_mul(2) // keys + values
        .and_then(|n| n.checked_mul(element_bytes))
        .ok_or_else(|| ModelError::KvAllocation {
            requested_bytes: None,
            detail: format!("{what}{total} elements overflow a byte count"),
        })?;
    // A byte count past `isize::MAX` cannot be a valid `Vec` allocation on
    // any target this crate ships for; fail cleanly with the computed size
    // named instead of letting the allocator abort the process with
    // "capacity overflow".
    if requested_bytes > isize::MAX as usize {
        return Err(ModelError::KvAllocation {
            requested_bytes: Some(requested_bytes),
            detail: "exceeds the largest representable allocation".to_string(),
        });
    }
    Ok((total, requested_bytes))
}

impl KvCache {
    /// Create a new, dense, FP32-backed KV cache, fully allocated.
    ///
    /// Infallible: a combination of dimensions whose element count would
    /// overflow `usize` saturates to `usize::MAX` elements instead of
    /// panicking on the multiplication, which then fails in the allocator
    /// instead — the same outcome the pre-existing plain-`*` version had for
    /// any input a caller would actually pass, with no behavior change for
    /// every real (non-overflowing) input. Use [`KvCache::try_new`] to
    /// detect the overflow case and react instead of aborting the process.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let total = checked_element_count(num_layers, num_kv_heads, max_seq_len, head_dim)
            .unwrap_or(usize::MAX);
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            capacity: max_seq_len,
            bounded: false,
            seq_len: 0,
            layout: KvLayout::Dense,
            storage: KvStorage::F32 {
                keys: vec![0.0; total],
                values: vec![0.0; total],
            },
            key_mirror: OnceLock::new(),
            value_mirror: OnceLock::new(),
        }
    }

    /// Fallible counterpart of [`KvCache::new`] (M-07).
    ///
    /// Fallible in **two** senses: an overflowing geometry is rejected, and a
    /// real allocator failure on an in-range request is surfaced the same
    /// way instead of aborting the process.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::KvAllocation`] naming the number of bytes that
    /// would have been requested when `num_layers * num_kv_heads *
    /// max_seq_len * head_dim` overflows `usize`, when the resulting byte
    /// count would exceed the largest representable allocation, or when the
    /// allocator itself reports it cannot satisfy the (in-range) request.
    pub fn try_new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> ModelResult<Self> {
        let (total, requested_bytes) = checked_geometry(
            "",
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            std::mem::size_of::<f32>(),
        )?;
        Ok(Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            capacity: max_seq_len,
            bounded: false,
            seq_len: 0,
            layout: KvLayout::Dense,
            storage: KvStorage::try_alloc(false, total, requested_bytes)?,
            key_mirror: OnceLock::new(),
            value_mirror: OnceLock::new(),
        })
    }

    /// Allocate a **sparse** KV cache: only for the layers that actually
    /// carry a KV cache at all (hybrid architectures such as Bonsai 2, whose
    /// 64 transformer layers include only 16 full-attention layers — the
    /// other 48 are recurrent Gated-DeltaNet layers with no KV
    /// representation), stored as `f16` rather than `f32` (M-07).
    ///
    /// # Parameters
    ///
    /// - `n_slots` — number of KV-bearing layers (16 for Bonsai 2 27B), i.e.
    ///   the cache's `num_layers()` after construction. **Not** the model's
    ///   raw transformer layer count: the caller maps a real layer index
    ///   (e.g. `3, 7, 11, ..., 63` for Bonsai 2's `(i+1) % 4 == 0` rule) down
    ///   to its slot `0..n_slots` before every store or read — this cache
    ///   does not itself hold that mapping.
    /// - `n_kv_heads`, `head_dim`, `max_seq` — as for [`KvCache::new`].
    ///
    /// # Memory
    ///
    /// Exactly `n_slots * n_kv_heads * head_dim * 2 (K+V) * 2 bytes (f16)`
    /// per token: for the Bonsai 2 27B (`n_slots=16, n_kv_heads=4,
    /// head_dim=256`) that is **65 536 bytes = 64 KiB/token** exactly.
    pub fn new_sparse(n_slots: usize, n_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let total =
            checked_element_count(n_slots, n_kv_heads, max_seq, head_dim).unwrap_or(usize::MAX);
        Self {
            num_layers: n_slots,
            num_kv_heads: n_kv_heads,
            head_dim,
            max_seq_len: max_seq,
            capacity: max_seq,
            bounded: false,
            seq_len: 0,
            layout: KvLayout::Sparse,
            storage: KvStorage::F16 {
                keys: vec![f16::ZERO; total],
                values: vec![f16::ZERO; total],
            },
            key_mirror: OnceLock::new(),
            value_mirror: OnceLock::new(),
        }
    }

    /// Fallible counterpart of [`KvCache::new_sparse`] (M-07): the same
    /// overflow / oversize / allocator-failure detection as
    /// [`KvCache::try_new`], sized for 2-byte (`f16`) elements.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::KvAllocation`] naming the requested byte count when
    /// the element count or byte count would overflow, when it would exceed
    /// the largest representable allocation, or when the allocator reports
    /// it cannot satisfy an otherwise in-range request.
    pub fn try_new_sparse(
        n_slots: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> ModelResult<Self> {
        let (total, requested_bytes) = checked_geometry(
            "sparse: ",
            n_slots,
            n_kv_heads,
            max_seq,
            head_dim,
            std::mem::size_of::<f16>(),
        )?;
        Ok(Self {
            num_layers: n_slots,
            num_kv_heads: n_kv_heads,
            head_dim,
            max_seq_len: max_seq,
            capacity: max_seq,
            bounded: false,
            seq_len: 0,
            layout: KvLayout::Sparse,
            storage: KvStorage::try_alloc(true, total, requested_bytes)?,
            key_mirror: OnceLock::new(),
            value_mirror: OnceLock::new(),
        })
    }

    /// Build a **bounded, lazily allocated** cache (M-07).
    ///
    /// `backing` selects element type and layout:
    ///
    /// | backing | elements | layout |
    /// |---|---|---|
    /// | [`KvCacheBacking::DenseF32`] | `f32` | one slot per layer |
    /// | [`KvCacheBacking::DenseF16`] | `f16` | one slot per layer |
    /// | [`KvCacheBacking::SparseF16`] | `f16` | one slot per KV-bearing layer |
    ///
    /// `max_seq_len` is the fixed logical limit reported by
    /// [`KvCache::max_seq_len`]; only `initial_capacity` positions (clamped
    /// to `1..=max_seq_len`) are allocated now, and the rest on demand —
    /// see [`KvCache::try_ensure_capacity`].
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeInvariant`] for a backing this type does not
    /// implement (the quantized `Dense{Q8,Fp8,Q4}` tiers live in
    /// [`crate::kv_cache_quant`]); [`ModelError::KvAllocation`] when the
    /// limit's geometry overflows or the first allocation fails.
    pub fn try_new_lazy(
        backing: KvCacheBacking,
        n_slots: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        initial_capacity: usize,
    ) -> ModelResult<Self> {
        let (f16_elements, layout) = match backing {
            KvCacheBacking::DenseF32 => (false, KvLayout::Dense),
            KvCacheBacking::DenseF16 => (true, KvLayout::Dense),
            KvCacheBacking::SparseF16 => (true, KvLayout::Sparse),
            other => {
                return Err(ModelError::ShapeInvariant {
                    tensor: "KvCache backing".to_string(),
                    expected: "DenseF32, DenseF16 or SparseF16".to_string(),
                    actual: format!("{other:?} (see crate::kv_cache_quant)"),
                })
            }
        };
        let element_bytes = backing.bytes_per_element();
        // The whole limit must be representable, or growth could overflow
        // later; checked now, without allocating it.
        checked_geometry(
            "lazy limit: ",
            n_slots,
            n_kv_heads,
            max_seq_len,
            head_dim,
            element_bytes,
        )?;
        let capacity = initial_capacity
            .clamp(1, max_seq_len.max(1))
            .min(max_seq_len);
        let (total, requested_bytes) = checked_geometry(
            "lazy: ",
            n_slots,
            n_kv_heads,
            capacity,
            head_dim,
            element_bytes,
        )?;
        Ok(Self {
            num_layers: n_slots,
            num_kv_heads: n_kv_heads,
            head_dim,
            max_seq_len,
            capacity,
            bounded: true,
            seq_len: 0,
            layout,
            storage: KvStorage::try_alloc(f16_elements, total, requested_bytes)?,
            key_mirror: OnceLock::new(),
            value_mirror: OnceLock::new(),
        })
    }

    /// Infallible, dense-`f16` counterpart of [`KvCache::try_new_lazy`] for
    /// constructors that cannot return an error (the config-only
    /// `BonsaiModel::new*` seams), mirroring [`KvCache::new`]'s contract: a
    /// geometry whose *initial allocation* overflows saturates and fails in
    /// the allocator. The limit itself is never allocated here — an absurd
    /// limit only surfaces later, as a typed [`ModelError::KvAllocation`]
    /// from the growth that would need it.
    pub fn new_lazy_f16(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        initial_capacity: usize,
    ) -> Self {
        let capacity = initial_capacity
            .clamp(1, max_seq_len.max(1))
            .min(max_seq_len);
        let total = checked_element_count(num_layers, num_kv_heads, capacity, head_dim)
            .unwrap_or(usize::MAX);
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            capacity,
            bounded: true,
            seq_len: 0,
            layout: KvLayout::Dense,
            storage: KvStorage::F16 {
                keys: vec![f16::ZERO; total],
                values: vec![f16::ZERO; total],
            },
            key_mirror: OnceLock::new(),
            value_mirror: OnceLock::new(),
        }
    }

    /// Whether this cache has one slot per KV-bearing layer of a hybrid
    /// stack ([`KvCache::new_sparse`], [`KvCacheBacking::SparseF16`]) rather
    /// than one per transformer layer. A layout fact — see
    /// [`KvCache::is_f16`] for the element type.
    pub fn is_sparse(&self) -> bool {
        self.layout == KvLayout::Sparse
    }

    /// Whether elements are stored as `f16` (2 bytes) rather than `f32`.
    pub fn is_f16(&self) -> bool {
        matches!(self.storage, KvStorage::F16 { .. })
    }

    /// Whether this is a bounded, lazily allocated cache
    /// ([`KvCache::try_new_lazy`]).
    pub fn is_lazy(&self) -> bool {
        self.bounded
    }

    /// The [`KvCacheBacking`] this cache implements.
    pub fn backing(&self) -> KvCacheBacking {
        match (self.layout, self.is_f16()) {
            (KvLayout::Sparse, _) => KvCacheBacking::SparseF16,
            (KvLayout::Dense, true) => KvCacheBacking::DenseF16,
            (KvLayout::Dense, false) => KvCacheBacking::DenseF32,
        }
    }

    /// Grow the allocation to at least `min_seq_len` positions if it is not
    /// already that large, rounding **up** to the next multiple of
    /// [`GROWTH_CHUNK_POSITIONS`] (and doubling once the cache is already
    /// large) — a caller advancing one position at a time reallocates
    /// rarely, never on every token, and never allocates its whole limit up
    /// front just because it *might* reach it.
    ///
    /// A no-op when `min_seq_len <= self.allocated_seq_len()`. Infallible:
    /// see [`try_ensure_capacity`](Self::try_ensure_capacity) to detect and
    /// react to a refused growth instead of leaving the cache unchanged.
    pub fn ensure_capacity(&mut self, min_seq_len: usize) {
        if let Err(e) = self.try_ensure_capacity(min_seq_len) {
            tracing::warn!(error = %e, min_seq_len, "KV cache growth refused");
        }
    }

    /// Fallible counterpart of [`ensure_capacity`](Self::ensure_capacity).
    ///
    /// On an eager cache (the legacy constructors) growth raises the
    /// allocation *and* [`max_seq_len`](Self::max_seq_len), as it always
    /// has. On a lazy cache ([`KvCache::try_new_lazy`]) `max_seq_len` is a
    /// ceiling: growth is clamped to it and a request beyond it is refused.
    ///
    /// # Errors
    ///
    /// [`ModelError::SequenceTooLong`] when a lazy cache is asked for more
    /// than its limit; [`ModelError::KvAllocation`] when the grown geometry
    /// overflows or the allocator refuses it. The cache is left completely
    /// unchanged in either case.
    pub fn try_ensure_capacity(&mut self, min_seq_len: usize) -> ModelResult<()> {
        if min_seq_len <= self.capacity {
            return Ok(());
        }
        if self.bounded && min_seq_len > self.max_seq_len {
            return Err(ModelError::SequenceTooLong {
                seq_len: min_seq_len,
                max_ctx: self.max_seq_len,
            });
        }
        let chunks = min_seq_len.div_ceil(GROWTH_CHUNK_POSITIONS).max(1);
        // Growing by exactly one chunk at a time makes `try_grow_to`'s
        // full-copy cost O(n^2) in total bytes moved once the cache is much
        // larger than a chunk; past that point double instead (standard
        // amortized growth). Only ever widens, so `min_seq_len` is always
        // satisfied.
        let mut new_capacity = chunks
            .saturating_mul(GROWTH_CHUNK_POSITIONS)
            .max(min_seq_len)
            .max(self.capacity.saturating_mul(2));
        if self.bounded {
            new_capacity = new_capacity.min(self.max_seq_len);
        }
        self.try_grow_to(new_capacity)
    }

    /// Reallocate this cache to exactly `new_capacity` positions, preserving
    /// every allocated position of every layer and head.
    /// [`seq_len`](Self::seq_len) is unchanged — the caller still owns
    /// advancing it. A no-op when `new_capacity <= allocated_seq_len()`
    /// (this method never shrinks; see [`truncate`](Self::truncate) /
    /// [`clear`](Self::clear) for the cursor). Stays in the same element
    /// type and layout it started in.
    ///
    /// Infallible; see [`try_grow_to`](Self::try_grow_to).
    pub fn grow_to(&mut self, new_capacity: usize) {
        if let Err(e) = self.try_grow_to(new_capacity) {
            tracing::warn!(error = %e, new_capacity, "KV cache growth refused");
        }
    }

    /// Fallible counterpart of [`grow_to`](Self::grow_to).
    ///
    /// Copies every **allocated** position, not only `0..seq_len`: a
    /// batched prefill stores a whole chunk before it advances the cursor,
    /// so a growth triggered mid-chunk must not drop the positions it has
    /// already written.
    ///
    /// # Errors
    ///
    /// [`ModelError::SequenceTooLong`] past a lazy cache's limit;
    /// [`ModelError::KvAllocation`] when the new geometry overflows or the
    /// allocator refuses it. The cache is left completely unchanged (the old
    /// storage is never dropped until the new one is fully built).
    pub fn try_grow_to(&mut self, new_capacity: usize) -> ModelResult<()> {
        if new_capacity <= self.capacity {
            return Ok(());
        }
        if self.bounded && new_capacity > self.max_seq_len {
            return Err(ModelError::SequenceTooLong {
                seq_len: new_capacity,
                max_ctx: self.max_seq_len,
            });
        }
        let (total, requested_bytes) = checked_geometry(
            "grow: ",
            self.num_layers,
            self.num_kv_heads,
            new_capacity,
            self.head_dim,
            self.storage.element_bytes(),
        )?;
        let mut grown = KvStorage::try_alloc(self.is_f16(), total, requested_bytes)?;
        // Positions `0..capacity` are one contiguous run of `capacity *
        // head_dim` elements for a fixed `(layer, head)` in both the old and
        // the new storage (`cache_offset` varies `pos` fastest): copy each
        // run directly, element type unchanged (f32->f32, f16->f16).
        let run_len = self.capacity * self.head_dim;
        if run_len > 0 {
            let slabs = self.num_layers * self.num_kv_heads;
            let old_stride = run_len;
            let new_stride = new_capacity * self.head_dim;
            match (&self.storage, &mut grown) {
                (
                    KvStorage::F32 {
                        keys: sk,
                        values: sv,
                    },
                    KvStorage::F32 {
                        keys: gk,
                        values: gv,
                    },
                ) => copy_runs(sk, sv, gk, gv, slabs, old_stride, new_stride, run_len),
                (
                    KvStorage::F16 {
                        keys: sk,
                        values: sv,
                    },
                    KvStorage::F16 {
                        keys: gk,
                        values: gv,
                    },
                ) => copy_runs(sk, sv, gk, gv, slabs, old_stride, new_stride, run_len),
                _ => {
                    return Err(ModelError::Internal(
                        "KvCache::try_grow_to: grown storage changed element type".to_string(),
                    ))
                }
            }
        }
        self.storage = grown;
        self.drop_mirrors();
        self.capacity = new_capacity;
        if !self.bounded {
            self.max_seq_len = new_capacity;
        }
        Ok(())
    }

    /// Current number of cached tokens.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// **Logical** capacity: the most positions this cache may hold. Stable
    /// on a lazy cache (sizing a context or a device-side cache from it is
    /// safe); equal to the allocation on an eager one.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// **Allocated** capacity: positions resident right now.
    pub fn allocated_seq_len(&self) -> usize {
        self.capacity
    }

    /// Validate a `try_store_key`/`try_store_value` call and make sure `pos` is
    /// allocated (growing a lazy cache on demand).
    ///
    /// Mirrors the checks performed by [`crate::kv_cache_fp16::KvCacheFp16::store`]
    /// so both caches fail the same way for the same inputs.
    fn prepare_store(
        &mut self,
        layer: usize,
        head: usize,
        pos: usize,
        len: usize,
    ) -> ModelResult<()> {
        if layer >= self.num_layers {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache layer".to_string(),
                expected: vec![self.num_layers],
                actual: vec![layer],
            });
        }
        if head >= self.num_kv_heads {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache head".to_string(),
                expected: vec![self.num_kv_heads],
                actual: vec![head],
            });
        }
        if pos >= self.max_seq_len {
            return Err(ModelError::SequenceTooLong {
                seq_len: pos + 1,
                max_ctx: self.max_seq_len,
            });
        }
        if len != self.head_dim {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache key/value dim".to_string(),
                expected: vec![self.head_dim],
                actual: vec![len],
            });
        }
        if pos >= self.capacity {
            self.try_ensure_capacity(pos + 1)?;
        }
        Ok(())
    }

    /// Write `src` at `offset` into the key (`is_key`) or value buffer,
    /// converting to `f16` where the storage is `f16`. The caller has
    /// validated `offset..offset + head_dim`.
    fn write_row(&mut self, is_key: bool, offset: usize, src: &[f32]) {
        self.drop_mirrors();
        let hd = self.head_dim;
        match &mut self.storage {
            KvStorage::F32 { keys, values } => {
                let dst = if is_key { keys } else { values };
                dst[offset..offset + hd].copy_from_slice(src);
            }
            KvStorage::F16 { keys, values } => {
                let dst = if is_key { keys } else { values };
                for (d, &s) in dst[offset..offset + hd].iter_mut().zip(src) {
                    *d = f16::from_f32(s);
                }
            }
        }
    }

    /// Store a key vector, **silently ignoring** an out-of-range call.
    ///
    /// This form exists only for callers that deliberately want the
    /// pre-[`try_store_key`](Self::try_store_key) contract (M-26): it drops
    /// the error instead of returning it. Prefer `try_store_key` for every
    /// new call site — in particular for a sparse cache, where a dropped
    /// store loses an entire full-attention layer's history for the position.
    ///
    /// The error-swallowing `store_key` / `store_value` forwarders this used
    /// to back are gone: every caller either propagates the error
    /// (`try_store_*`) or names this lossy form explicitly.
    pub fn store_key_lossy(&mut self, layer: usize, head: usize, pos: usize, key: &[f32]) {
        let _ = self.try_store_key(layer, head, pos, key);
    }

    /// Store a key vector for `(layer, head, pos)`, growing a lazy cache's
    /// allocation when `pos` is past it.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] for an out-of-range `layer`, `head`, or
    /// a `key` length that doesn't match `head_dim`;
    /// [`ModelError::SequenceTooLong`] when `pos >= max_seq_len`;
    /// [`ModelError::KvAllocation`] when the on-demand growth fails.
    pub fn try_store_key(
        &mut self,
        layer: usize,
        head: usize,
        pos: usize,
        key: &[f32],
    ) -> ModelResult<()> {
        self.prepare_store(layer, head, pos, key.len())?;
        let offset = self.cache_offset(layer, head, pos);
        self.write_row(true, offset, key);
        Ok(())
    }

    /// Store a value vector, silently ignoring an out-of-range call — see
    /// [`store_key_lossy`](Self::store_key_lossy).
    pub fn store_value_lossy(&mut self, layer: usize, head: usize, pos: usize, value: &[f32]) {
        let _ = self.try_store_value(layer, head, pos, value);
    }

    /// Fallible value store; same contract as
    /// [`try_store_key`](Self::try_store_key).
    ///
    /// # Errors
    ///
    /// As [`try_store_key`](Self::try_store_key).
    pub fn try_store_value(
        &mut self,
        layer: usize,
        head: usize,
        pos: usize,
        value: &[f32],
    ) -> ModelResult<()> {
        self.prepare_store(layer, head, pos, value.len())?;
        let offset = self.cache_offset(layer, head, pos);
        self.write_row(false, offset, value);
        Ok(())
    }

    /// Drop the `f16` read-back mirrors (every write and every growth calls
    /// this). Free when they were never built.
    fn drop_mirrors(&mut self) {
        self.key_mirror.take();
        self.value_mirror.take();
    }

    /// Rows `0..seq_len` of the key or value buffer of `(layer, head)`.
    ///
    /// `seq_len` is clamped to the allocated capacity: rows past the
    /// allocation were never stored, and are not fabricated. An out-of-range
    /// `layer`/`head` yields an empty slice rather than a panic.
    fn rows(&self, is_key: bool, layer: usize, head: usize, seq_len: usize) -> &[f32] {
        if layer >= self.num_layers || head >= self.num_kv_heads {
            return &[];
        }
        let hd = self.head_dim;
        let start = self.cache_offset(layer, head, 0);
        let end = start + seq_len.min(self.capacity) * hd;
        match &self.storage {
            KvStorage::F32 { keys, values } => {
                let src = if is_key { keys } else { values };
                &src[start..end]
            }
            KvStorage::F16 { keys, values } => {
                let (src, mirror) = if is_key {
                    (keys, &self.key_mirror)
                } else {
                    (values, &self.value_mirror)
                };
                let widened = mirror.get_or_init(|| src.iter().map(|h| h.to_f32()).collect());
                &widened[start..end]
            }
        }
    }

    /// Cached keys of `(layer, head)` for positions `0..seq_len`,
    /// `[seq_len × head_dim]` row-major (`seq_len` clamped to the
    /// allocation).
    ///
    /// Zero-copy for an `f32` cache. For an `f16` cache the rows come from
    /// the `f32` read-back mirror (see the module docs): exact, since every
    /// `f16` value is representable in `f32`, and built at most once between
    /// writes. Hot paths use [`attend_group`](Self::attend_group) instead,
    /// which reads either storage in place.
    pub fn keys_for(&self, layer: usize, head: usize, seq_len: usize) -> &[f32] {
        self.rows(true, layer, head, seq_len)
    }

    /// Cached values of `(layer, head)` for positions `0..seq_len`. See
    /// [`keys_for`](Self::keys_for).
    pub fn values_for(&self, layer: usize, head: usize, seq_len: usize) -> &[f32] {
        self.rows(false, layer, head, seq_len)
    }

    /// Gather the key rows at `positions` of `(layer, head)` into one owned
    /// `[positions.len() × head_dim]` buffer — only those rows are read (and,
    /// for an `f16` cache, widened), which is what a windowed attention
    /// needs. Never-allocated positions read as zeros; an out-of-range
    /// `layer`/`head` yields zeros too.
    pub fn gather_keys(&self, layer: usize, head: usize, positions: &[usize]) -> Vec<f32> {
        self.gather(true, layer, head, positions)
    }

    /// Value counterpart of [`gather_keys`](Self::gather_keys).
    pub fn gather_values(&self, layer: usize, head: usize, positions: &[usize]) -> Vec<f32> {
        self.gather(false, layer, head, positions)
    }

    fn gather(&self, is_key: bool, layer: usize, head: usize, positions: &[usize]) -> Vec<f32> {
        let hd = self.head_dim;
        let mut out = vec![0.0f32; positions.len() * hd];
        if layer >= self.num_layers || head >= self.num_kv_heads {
            return out;
        }
        for (dst, &pos) in out.chunks_exact_mut(hd.max(1)).zip(positions) {
            if pos >= self.capacity {
                continue;
            }
            let src = self.cache_offset(layer, head, pos);
            match &self.storage {
                KvStorage::F32 { keys, values } => {
                    let buf = if is_key { keys } else { values };
                    dst.copy_from_slice(&buf[src..src + hd]);
                }
                KvStorage::F16 { keys, values } => {
                    let buf = if is_key { keys } else { values };
                    for (d, s) in dst.iter_mut().zip(&buf[src..src + hd]) {
                        *d = s.to_f32();
                    }
                }
            }
        }
        out
    }

    /// Cached keys of `(layer, head)` for positions `0..seq_len` as an owned
    /// buffer, widened row by row for an `f16` cache (no whole-buffer
    /// mirror); `seq_len` is clamped to the allocation.
    pub fn keys_for_owned(&self, layer: usize, head: usize, seq_len: usize) -> Vec<f32> {
        let rows: Vec<usize> = (0..seq_len.min(self.capacity)).collect();
        self.gather(true, layer, head, &rows)
    }

    /// Value counterpart of [`keys_for_owned`](Self::keys_for_owned).
    pub fn values_for_owned(&self, layer: usize, head: usize, seq_len: usize) -> Vec<f32> {
        let rows: Vec<usize> = (0..seq_len.min(self.capacity)).collect();
        self.gather(false, layer, head, &rows)
    }

    /// Advance the sequence position by one token.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Reset the cache cursor (clear all stored KV pairs). The allocation is
    /// kept, so the next sequence re-grows nothing it already reached.
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Roll the cache cursor back to `new_len`, logically discarding all cached
    /// positions at or beyond `new_len`. Positions `0..new_len` are retained.
    /// `new_len` is clamped down to the current `seq_len` (never grows).
    /// No buffer zeroing — identical convention to `clear`.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.seq_len {
            self.seq_len = new_len;
        }
    }

    /// Compute flat offset into the storage (the stride is the allocated
    /// capacity, not the logical limit).
    fn cache_offset(&self, layer: usize, head: usize, pos: usize) -> usize {
        ((layer * self.num_kv_heads + head) * self.capacity + pos) * self.head_dim
    }

    /// Bytes resident right now (the allocation, honest about the element
    /// width: 4 bytes/element for `f32`, 2 for `f16`).
    pub fn memory_bytes(&self) -> usize {
        self.storage.memory_bytes()
    }

    /// Bytes this cache would hold at its full logical limit.
    pub fn max_memory_bytes(&self) -> usize {
        checked_element_count(
            self.num_layers,
            self.num_kv_heads,
            self.max_seq_len,
            self.head_dim,
        )
        .and_then(|n| n.checked_mul(2 * self.storage.element_bytes()))
        .unwrap_or(usize::MAX)
    }

    // `utilization_ratio` (M-19): removed — zero production callers, and
    // `pub(crate)` fails this crate's `-D warnings --all-targets` gate via
    // `dead_code`. Re-add it (`seq_len / max_seq_len`) if a real consumer
    // (e.g. an admin endpoint reporting live KV pressure) needs it.

    /// Number of layers (dense) or KV slots (sparse) in this cache.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of KV heads per layer.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Manually set the cached sequence length.
    ///
    /// Used by the prefix-cache integration when restoring previously
    /// computed KV blocks: after [`inject_block`](Self::inject_block) writes
    /// the block contents, the consumer must call this to advertise the
    /// number of valid positions to subsequent attention computations.
    ///
    /// `n` is clamped to `max_seq_len`.
    pub fn set_seq_len(&mut self, n: usize) {
        self.seq_len = n.min(self.max_seq_len);
    }

    /// Extract one prefix-cache block worth of KV for a single layer.
    ///
    /// Reads `block_size` consecutive positions starting at `start_pos` for
    /// every KV head in `layer` and returns them in `[head][pos_in_block][dim]`
    /// order, packed as a flat `Vec<f32>` of length
    /// `num_kv_heads * block_size * head_dim`.
    ///
    /// Mirrors the layout used by [`crate::prefix_cache::CacheBlock`].
    ///
    /// Returns `(keys, values)`. Positions past the allocation (never
    /// stored) read as zeros; so does an out-of-range `layer`.
    pub fn extract_block(
        &self,
        layer: usize,
        start_pos: usize,
        block_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let hd = self.head_dim;
        let per_layer = self.num_kv_heads * block_size * hd;
        let mut keys = vec![0.0f32; per_layer];
        let mut values = vec![0.0f32; per_layer];
        if layer >= self.num_layers {
            return (keys, values);
        }
        for head in 0..self.num_kv_heads {
            for off in 0..block_size {
                let pos = start_pos + off;
                if pos >= self.capacity {
                    continue;
                }
                let src = self.cache_offset(layer, head, pos);
                let dst = (head * block_size + off) * hd;
                match &self.storage {
                    KvStorage::F32 {
                        keys: src_keys,
                        values: src_values,
                    } => {
                        keys[dst..dst + hd].copy_from_slice(&src_keys[src..src + hd]);
                        values[dst..dst + hd].copy_from_slice(&src_values[src..src + hd]);
                    }
                    KvStorage::F16 {
                        keys: src_keys,
                        values: src_values,
                    } => {
                        for (d, s) in keys[dst..dst + hd].iter_mut().zip(&src_keys[src..src + hd]) {
                            *d = s.to_f32();
                        }
                        for (d, s) in values[dst..dst + hd]
                            .iter_mut()
                            .zip(&src_values[src..src + hd])
                        {
                            *d = s.to_f32();
                        }
                    }
                }
            }
        }
        (keys, values)
    }

    /// Inject a previously extracted block back into the cache for a single
    /// layer, growing a lazy cache's allocation to cover it.
    ///
    /// `keys` and `values` must have the `[head][pos_in_block][dim]` layout
    /// produced by [`extract_block`](Self::extract_block), of length
    /// `num_kv_heads * block_size * head_dim`.
    ///
    /// # Errors
    ///
    /// [`ModelError::ShapeMismatch`] for an out-of-range `layer` or wrongly
    /// sized `keys`/`values`; [`ModelError::KvAllocation`] when the growth
    /// fails. Positions at or past [`max_seq_len`](Self::max_seq_len) are
    /// skipped (the block straddles the end of the context).
    pub fn try_inject_block(
        &mut self,
        layer: usize,
        start_pos: usize,
        block_size: usize,
        keys: &[f32],
        values: &[f32],
    ) -> ModelResult<()> {
        let hd = self.head_dim;
        let per_layer = self.num_kv_heads * block_size * hd;
        if layer >= self.num_layers {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache inject layer".to_string(),
                expected: vec![self.num_layers],
                actual: vec![layer],
            });
        }
        if keys.len() != per_layer || values.len() != per_layer {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache inject block".to_string(),
                expected: vec![per_layer],
                actual: vec![keys.len(), values.len()],
            });
        }
        let end = start_pos.saturating_add(block_size).min(self.max_seq_len);
        if end > self.capacity {
            self.try_ensure_capacity(end)?;
        }
        for head in 0..self.num_kv_heads {
            for off in 0..block_size {
                let pos = start_pos + off;
                if pos >= self.max_seq_len {
                    continue;
                }
                let src = (head * block_size + off) * hd;
                let dst = self.cache_offset(layer, head, pos);
                self.write_row(true, dst, &keys[src..src + hd]);
                self.write_row(false, dst, &values[src..src + hd]);
            }
        }
        Ok(())
    }

    /// Infallible form of [`try_inject_block`](Self::try_inject_block) (the
    /// prefix cache's historical API): a rejected injection is logged at
    /// `warn` and leaves the cache unchanged.
    pub fn inject_block(
        &mut self,
        layer: usize,
        start_pos: usize,
        block_size: usize,
        keys: &[f32],
        values: &[f32],
    ) {
        if let Err(e) = self.try_inject_block(layer, start_pos, block_size, keys, values) {
            tracing::warn!(error = %e, layer, start_pos, block_size, "KV block injection refused");
        }
    }
}

/// Copy `slabs` runs of `run_len` elements from `old_stride`-strided key /
/// value buffers into `new_stride`-strided ones.
#[allow(clippy::too_many_arguments)]
fn copy_runs<T: Copy>(
    src_keys: &[T],
    src_values: &[T],
    dst_keys: &mut [T],
    dst_values: &mut [T],
    slabs: usize,
    old_stride: usize,
    new_stride: usize,
    run_len: usize,
) {
    for slab in 0..slabs {
        let (old_start, new_start) = (slab * old_stride, slab * new_stride);
        dst_keys[new_start..new_start + run_len]
            .copy_from_slice(&src_keys[old_start..old_start + run_len]);
        dst_values[new_start..new_start + run_len]
            .copy_from_slice(&src_values[old_start..old_start + run_len]);
    }
}
