//! PagedAttention / vLLM-style paged KV cache.
//!
//! This module implements a block-based key-value cache that mirrors the
//! PagedAttention design from vLLM.  Physical memory is divided into fixed-size
//! *pages* (blocks), each holding `block_size` token slots.  Logical sequences
//! are given a [`BlockTable`] that maps their logical block indices to physical
//! page indices obtained from a shared [`BlockPool`].  Allocation is *lazy*:
//! pages are handed out on demand as sequences grow, and are returned to the
//! pool when a sequence is dropped.
//!
//! # Architecture overview
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  PagedKvCache                                    │
//! │                                                  │
//! │  pool: BlockPool  ◄──── free_list (Vec<usize>)  │
//! │  sequences: HashMap<seq_id, BlockTable>          │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! Each [`KvPage`] stores keys and values for *one* transformer layer and
//! `block_size` token positions.  A [`BlockTable`] therefore holds
//! `num_layers` independent block lists, one per layer.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of token slots per KV page.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// KvPage
// ---------------------------------------------------------------------------

/// A single KV page that holds `block_size` token slots for **one** layer.
///
/// Both `keys` and `values` are stored in row-major order with the logical
/// layout `[block_size, num_kv_heads, head_dim]`.
#[derive(Debug, Clone)]
pub struct KvPage {
    /// Flattened key tensor: `[block_size × num_kv_heads × head_dim]` f32 elements.
    pub keys: Vec<f32>,
    /// Flattened value tensor: same shape as `keys`.
    pub values: Vec<f32>,
}

impl KvPage {
    /// Allocate a zeroed page for the given dimensions.
    fn new(block_size: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let len = block_size * num_kv_heads * head_dim;
        Self {
            keys: vec![0.0_f32; len],
            values: vec![0.0_f32; len],
        }
    }
}

// ---------------------------------------------------------------------------
// BlockPool
// ---------------------------------------------------------------------------

/// Pre-allocated pool of [`KvPage`]s shared across all sequences.
///
/// Pages are handed out via [`BlockPool::allocate`] and returned via
/// [`BlockPool::free`].  The pool never grows beyond its initial capacity.
pub struct BlockPool {
    /// All pages ever created (indexed by physical block index).
    pages: Vec<KvPage>,
    /// Indices of currently unused pages.
    free_list: Vec<usize>,
    /// Token slots per page.
    block_size: usize,
    /// Number of transformer layers each page covers.
    num_layers: usize,
    /// Number of KV-attention heads.
    num_kv_heads: usize,
    /// Dimensionality of each attention head.
    head_dim: usize,
}

impl BlockPool {
    /// Create a pool with `capacity` pages.
    ///
    /// Every page is pre-allocated and zeroed at construction time so that
    /// subsequent allocations are O(1) pointer-hand-offs from the free list.
    ///
    /// # Arguments
    ///
    /// * `capacity`    – total number of physical pages.
    /// * `block_size`  – token slots per page.
    /// * `num_layers`  – number of transformer layers (informational; each
    ///   logical layer has its own independent block list in [`BlockTable`]).
    /// * `num_kv_heads` – number of KV attention heads.
    /// * `head_dim`    – per-head dimension.
    pub fn new(
        capacity: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let mut pages = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);
        for idx in 0..capacity {
            pages.push(KvPage::new(block_size, num_kv_heads, head_dim));
            free_list.push(idx);
        }
        Self {
            pages,
            free_list,
            block_size,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Allocate one page and return its physical index.
    ///
    /// Returns `None` when the pool is exhausted (out-of-memory).
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    /// Return page `idx` to the pool.
    ///
    /// The page contents are **not** zeroed on release; callers must overwrite
    /// every slot they intend to read.
    pub fn free(&mut self, idx: usize) {
        self.free_list.push(idx);
    }

    /// Number of pages currently available for allocation.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of pages in the pool (constant after construction).
    pub fn total_count(&self) -> usize {
        self.pages.len()
    }

    /// Fraction of pages currently in use, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when the pool is empty (capacity == 0).
    pub fn utilization(&self) -> f32 {
        let total = self.total_count();
        if total == 0 {
            return 0.0;
        }
        let used = total - self.free_count();
        used as f32 / total as f32
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Return an immutable reference to the page at physical index `idx`.
    fn page(&self, idx: usize) -> &KvPage {
        &self.pages[idx]
    }

    /// Return a mutable reference to the page at physical index `idx`.
    fn page_mut(&mut self, idx: usize) -> &mut KvPage {
        &mut self.pages[idx]
    }

    /// Number of f32 elements per token slot within a page.
    fn slot_len(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

// ---------------------------------------------------------------------------
// BlockTable
// ---------------------------------------------------------------------------

/// Maps a sequence's logical block indices to physical page indices.
///
/// Each transformer layer has its own independent list of physical blocks so
/// that cross-layer sharing is straightforward to reason about.
pub struct BlockTable {
    /// Token slots per block (must match the [`BlockPool`]'s `block_size`).
    block_size: usize,
    /// `blocks[layer][logical_block]` → physical page index.
    blocks: Vec<Vec<usize>>,
    /// Number of transformer layers.
    num_layers: usize,
    /// Highest `token_pos + 1` actually written via [`PagedKvCache::write_kv`]
    /// across any layer. Distinct from [`token_capacity`](Self::token_capacity),
    /// which reports *allocated* (block-rounded) capacity, not the number of
    /// positions that hold real data.
    written_len: usize,
}

impl BlockTable {
    /// Create an empty block table for `num_layers` layers.
    pub fn new(num_layers: usize, block_size: usize) -> Self {
        Self {
            block_size,
            blocks: vec![Vec::new(); num_layers],
            num_layers,
            written_len: 0,
        }
    }

    /// Record that `token_pos` has been written to, extending the table's
    /// externally-visible write cursor (see [`PagedKvCache::sequence_length`])
    /// if `token_pos` is the highest position written so far.
    fn record_write(&mut self, token_pos: usize) {
        self.written_len = self.written_len.max(token_pos + 1);
    }

    /// Append a newly-allocated physical page to `layer`'s block list.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `layer >= num_layers`.
    pub fn append_block(&mut self, layer: usize, physical_idx: usize) {
        debug_assert!(layer < self.num_layers);
        self.blocks[layer].push(physical_idx);
    }

    /// Look up the physical page index for logical block `logical_block` in
    /// `layer`.
    ///
    /// Returns `None` if either the layer or the logical block index is out of
    /// range.
    pub fn get_block(&self, layer: usize, logical_block: usize) -> Option<usize> {
        self.blocks.get(layer)?.get(logical_block).copied()
    }

    /// Number of physical blocks currently mapped for `layer`.
    ///
    /// Returns `0` for out-of-range `layer`.
    pub fn num_blocks(&self, layer: usize) -> usize {
        self.blocks.get(layer).map_or(0, |v| v.len())
    }

    /// Total token capacity (may include unused trailing slots) for `layer`.
    ///
    /// This is the *allocated* capacity rounded up to `block_size`, **not**
    /// the number of positions actually written — use
    /// [`PagedKvCache::sequence_length`] for the real write cursor.
    pub fn token_capacity(&self, layer: usize) -> usize {
        self.num_blocks(layer) * self.block_size
    }
}

// ---------------------------------------------------------------------------
// PagedKvError
// ---------------------------------------------------------------------------

/// Errors returned by [`PagedKvCache`] operations.
#[derive(Debug, thiserror::Error)]
pub enum PagedKvError {
    /// The requested sequence ID does not exist in the cache.
    #[error("sequence {0} not found")]
    SequenceNotFound(u64),

    /// The pool has no free pages left.
    #[error("out of memory: no free KV blocks")]
    OutOfMemory,

    /// The token position exceeds the sequence's allocated capacity.
    #[error("token position {pos} out of range for sequence {seq_id}")]
    PositionOutOfRange { seq_id: u64, pos: usize },

    /// A key or value slice has the wrong length.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
}

// ---------------------------------------------------------------------------
// PagedKvCache
// ---------------------------------------------------------------------------

/// Orchestrates a [`BlockPool`] and per-sequence [`BlockTable`]s.
///
/// This is the primary entry-point for vLLM-style paged KV management.
/// Sequences are identified by opaque `u64` IDs assigned by
/// [`PagedKvCache::create_sequence`].
///
/// # Example
///
/// ```rust
/// use oxibonsai_model::paged_kv_cache::{PagedKvCache};
///
/// let mut cache = PagedKvCache::new(
///     /*capacity=*/ 128,
///     /*num_layers=*/ 32,
///     /*num_kv_heads=*/ 8,
///     /*head_dim=*/ 128,
/// );
///
/// let seq = cache.create_sequence();
/// cache.ensure_capacity(seq, 1).expect("failed to ensure capacity");
///
/// let key = vec![1.0_f32; 8 * 128];   // num_kv_heads * head_dim
/// let val = vec![2.0_f32; 8 * 128];
/// cache.write_kv(seq, 0, 0, &key, &val).expect("failed to write kv");
///
/// let (k, v) = cache.read_kv(seq, 0, 0).expect("failed to read kv");
/// assert_eq!(k, key.as_slice());
/// ```
pub struct PagedKvCache {
    pool: BlockPool,
    sequences: HashMap<u64, BlockTable>,
    next_seq_id: u64,
}

impl PagedKvCache {
    /// Create a cache with [`DEFAULT_BLOCK_SIZE`] token slots per page.
    ///
    /// # Arguments
    ///
    /// * `capacity`    – maximum number of physical KV pages.
    /// * `num_layers`  – number of transformer layers.
    /// * `num_kv_heads` – number of KV attention heads.
    /// * `head_dim`    – per-head feature dimension.
    pub fn new(capacity: usize, num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self::new_with_block_size(
            capacity,
            DEFAULT_BLOCK_SIZE,
            num_layers,
            num_kv_heads,
            head_dim,
        )
    }

    /// Create a cache with a custom `block_size`.
    ///
    /// All other parameters are the same as [`PagedKvCache::new`].
    pub fn new_with_block_size(
        capacity: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            pool: BlockPool::new(capacity, block_size, num_layers, num_kv_heads, head_dim),
            sequences: HashMap::new(),
            next_seq_id: 0,
        }
    }

    // ------------------------------------------------------------------
    // Sequence lifecycle
    // ------------------------------------------------------------------

    /// Register a new sequence and return its ID.
    ///
    /// Sequences start with no allocated pages.  Call [`ensure_capacity`] or
    /// [`write_kv`] to trigger lazy allocation.
    ///
    /// [`ensure_capacity`]: PagedKvCache::ensure_capacity
    /// [`write_kv`]: PagedKvCache::write_kv
    pub fn create_sequence(&mut self) -> u64 {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        let num_layers = self.pool.num_layers;
        let block_size = self.pool.block_size;
        self.sequences
            .insert(id, BlockTable::new(num_layers, block_size));
        id
    }

    /// Drop a sequence, returning all its physical pages to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`PagedKvError::SequenceNotFound`] if `seq_id` is unknown.
    pub fn drop_sequence(&mut self, seq_id: u64) -> Result<(), PagedKvError> {
        let table = self
            .sequences
            .remove(&seq_id)
            .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

        for layer_blocks in &table.blocks {
            for &phys_idx in layer_blocks {
                self.pool.free(phys_idx);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Capacity management
    // ------------------------------------------------------------------

    /// Ensure the sequence can hold at least `num_tokens` positions in every
    /// layer, allocating new pages from the pool as needed.
    ///
    /// This is the primary lazy-allocation entry-point.  [`write_kv`] calls
    /// this internally, so explicit calls are optional.
    ///
    /// # Errors
    ///
    /// * [`PagedKvError::SequenceNotFound`] – unknown `seq_id`.
    /// * [`PagedKvError::OutOfMemory`] – pool exhausted before the request
    ///   could be satisfied; partially-allocated pages are **not** rolled back
    ///   (vLLM behaviour — callers should drop the sequence on OOM).
    ///
    /// [`write_kv`]: PagedKvCache::write_kv
    pub fn ensure_capacity(&mut self, seq_id: u64, num_tokens: usize) -> Result<(), PagedKvError> {
        // We need to look up immutable fields before taking the mutable borrow.
        let num_layers = self.pool.num_layers;
        let block_size = self.pool.block_size;

        // Compute the number of blocks needed to cover `num_tokens`.
        let blocks_needed = num_tokens.div_ceil(block_size);

        // We must not hold a reference into `self.sequences` while mutating
        // `self.pool`, so we collect the per-layer deficits first.
        let deficits: Vec<usize> = {
            let table = self
                .sequences
                .get(&seq_id)
                .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

            (0..num_layers)
                .map(|layer| {
                    let have = table.num_blocks(layer);
                    blocks_needed.saturating_sub(have)
                })
                .collect()
        };

        // Allocate the required pages and record them in the block table.
        for (layer, deficit) in deficits.into_iter().enumerate() {
            for _ in 0..deficit {
                let phys = self.pool.allocate().ok_or(PagedKvError::OutOfMemory)?;
                let table = self
                    .sequences
                    .get_mut(&seq_id)
                    .ok_or(PagedKvError::SequenceNotFound(seq_id))?;
                table.append_block(layer, phys);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // KV I/O
    // ------------------------------------------------------------------

    /// Write a key-value pair for token at position `token_pos` in `layer`.
    ///
    /// `key` and `value` must each have exactly `num_kv_heads * head_dim`
    /// elements.  Capacity is extended automatically if the token position
    /// falls outside the current allocation.
    ///
    /// # Errors
    ///
    /// * [`PagedKvError::SequenceNotFound`]
    /// * [`PagedKvError::OutOfMemory`]
    /// * [`PagedKvError::DimMismatch`] – wrong slice length.
    pub fn write_kv(
        &mut self,
        seq_id: u64,
        layer: usize,
        token_pos: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), PagedKvError> {
        let slot_len = self.pool.slot_len();

        if key.len() != slot_len {
            return Err(PagedKvError::DimMismatch {
                expected: slot_len,
                actual: key.len(),
            });
        }
        if value.len() != slot_len {
            return Err(PagedKvError::DimMismatch {
                expected: slot_len,
                actual: value.len(),
            });
        }

        // Ensure we have enough blocks for token_pos + 1.
        self.ensure_capacity(seq_id, token_pos + 1)?;

        let block_size = self.pool.block_size;
        let logical_block = token_pos / block_size;
        let slot_in_block = token_pos % block_size;

        let phys = {
            let table = self
                .sequences
                .get_mut(&seq_id)
                .ok_or(PagedKvError::SequenceNotFound(seq_id))?;
            let phys =
                table
                    .get_block(layer, logical_block)
                    .ok_or(PagedKvError::PositionOutOfRange {
                        seq_id,
                        pos: token_pos,
                    })?;
            table.record_write(token_pos);
            phys
        };

        let offset = slot_in_block * slot_len;
        let page = self.pool.page_mut(phys);
        page.keys[offset..offset + slot_len].copy_from_slice(key);
        page.values[offset..offset + slot_len].copy_from_slice(value);
        Ok(())
    }

    /// Read the key-value pair for token at position `token_pos` in `layer`.
    ///
    /// Returns `(&key_slice, &value_slice)` where each slice has
    /// `num_kv_heads * head_dim` elements.
    ///
    /// # Errors
    ///
    /// * [`PagedKvError::SequenceNotFound`]
    /// * [`PagedKvError::PositionOutOfRange`] – token has not been written yet.
    pub fn read_kv(
        &self,
        seq_id: u64,
        layer: usize,
        token_pos: usize,
    ) -> Result<(&[f32], &[f32]), PagedKvError> {
        let block_size = self.pool.block_size;
        let slot_len = self.pool.slot_len();

        let table = self
            .sequences
            .get(&seq_id)
            .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

        let logical_block = token_pos / block_size;
        let slot_in_block = token_pos % block_size;

        let phys =
            table
                .get_block(layer, logical_block)
                .ok_or(PagedKvError::PositionOutOfRange {
                    seq_id,
                    pos: token_pos,
                })?;

        let offset = slot_in_block * slot_len;
        let page = self.pool.page(phys);
        Ok((
            &page.keys[offset..offset + slot_len],
            &page.values[offset..offset + slot_len],
        ))
    }

    // ------------------------------------------------------------------
    // Metrics
    // ------------------------------------------------------------------

    /// Fraction of pool pages currently in use (`[0.0, 1.0]`).
    pub fn pool_utilization(&self) -> f32 {
        self.pool.utilization()
    }

    /// Number of token positions actually written to `seq_id` via
    /// [`write_kv`](Self::write_kv) (the highest `token_pos + 1` seen across
    /// any layer), **not** the block-rounded allocated capacity.
    ///
    /// Returns `0` for unknown sequences or sequences that have not had any
    /// tokens written yet.
    pub fn sequence_length(&self, seq_id: u64) -> usize {
        self.sequences.get(&seq_id).map_or(0, |t| t.written_len)
    }
}
