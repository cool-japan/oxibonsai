//! The experimental page-based KV cache ([`PagedKvCache`]), split out of
//! `kv_cache.rs` (B2-11-FIX). Re-exported from [`crate::kv_cache`], so the
//! `crate::kv_cache::PagedKvCache` path is unchanged.

/// Default number of positions per page.
const DEFAULT_PAGE_SIZE: usize = 256;

/// A single page in the paged KV cache.
///
/// Each page holds `page_size` positions worth of key and value data
/// for a single layer and head.
#[derive(Debug, Clone)]
struct KvPage {
    /// Key data: [page_size * head_dim] floats.
    keys: Vec<f32>,
    /// Value data: [page_size * head_dim] floats.
    values: Vec<f32>,
    /// Number of positions actually used in this page.
    used: usize,
}

impl KvPage {
    fn new(page_size: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![0.0; page_size * head_dim],
            values: vec![0.0; page_size * head_dim],
            used: 0,
        }
    }
}

/// Page-based KV cache for memory-efficient allocation.
///
/// Instead of pre-allocating the full `max_seq_len` contiguously,
/// pages of `page_size` positions are allocated on demand. This is
/// beneficial when the actual sequence length is much shorter than
/// `max_seq_len`.
///
/// **Experimental.** `BonsaiModel` never builds this type — it holds a
/// single, always-dense [`KvCache`](crate::kv_cache::KvCache) (or, since M-07, [`KvCache::new_sparse`](crate::kv_cache::KvCache::new_sparse)
/// for a hybrid model) — so nothing here is on a production forward path
/// today. Implemented: lazy per-page allocation, `store_key`/`store_value`/
/// `keys_for`/`values_for`, utilization accounting. **Not** implemented:
/// PagedAttention's actual GPU/kernel-side benefit (a fused attention kernel
/// that reads pages directly) and eviction/preemption — this is the
/// primitive-only building block, not a wired-up PagedAttention scheduler
/// (README.md's "Known Limitations" documents this as shipped-but-unwired,
/// not advertised-but-missing). A second, more elaborate PagedAttention/vLLM
/// -style implementation with its own `BlockPool`/`BlockTable`/`KvPage`
/// lives in [`crate::paged_kv_cache`] under the *same* type name
/// `PagedKvCache` — the two are unrelated and neither is a re-export of the
/// other; do not conflate them when reading call sites. The real plan for
/// either to become selectable lives with the M-13/RT-14 `KvCacheBacking`
/// work (see [`KvCacheBacking`](crate::kv_cache::KvCacheBacking)): wire behind a bit-exactness parity test
/// against the dense `KvCache` before either becomes reachable from
/// `BonsaiModel`.
#[derive(Debug)]
pub struct PagedKvCache {
    /// Pages indexed as [layer][head][page_index].
    pages: Vec<Vec<Vec<KvPage>>>,
    /// Number of transformer layers.
    num_layers: usize,
    /// Number of KV heads per layer.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Positions per page.
    page_size: usize,
    /// Maximum sequence length (total capacity).
    max_seq_len: usize,
    /// Current sequence length.
    seq_len: usize,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    ///
    /// Pages are allocated lazily as positions are stored.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self::with_page_size(
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            DEFAULT_PAGE_SIZE,
        )
    }

    /// Create a new paged KV cache with a custom page size.
    pub fn with_page_size(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        page_size: usize,
    ) -> Self {
        let pages = (0..num_layers)
            .map(|_| (0..num_kv_heads).map(|_| Vec::new()).collect())
            .collect();

        Self {
            pages,
            num_layers,
            num_kv_heads,
            head_dim,
            page_size,
            max_seq_len,
            seq_len: 0,
        }
    }

    /// Store a key vector for a specific layer, head, and position.
    pub fn store_key(&mut self, layer: usize, head: usize, pos: usize, key: &[f32]) {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_kv_heads);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(key.len(), self.head_dim);

        let page_idx = pos / self.page_size;
        let offset_in_page = pos % self.page_size;

        self.ensure_page(layer, head, page_idx);

        let page = &mut self.pages[layer][head][page_idx];
        let start = offset_in_page * self.head_dim;
        page.keys[start..start + self.head_dim].copy_from_slice(key);
        if offset_in_page >= page.used {
            page.used = offset_in_page + 1;
        }
    }

    /// Store a value vector for a specific layer, head, and position.
    pub fn store_value(&mut self, layer: usize, head: usize, pos: usize, value: &[f32]) {
        debug_assert!(layer < self.num_layers);
        debug_assert!(head < self.num_kv_heads);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(value.len(), self.head_dim);

        let page_idx = pos / self.page_size;
        let offset_in_page = pos % self.page_size;

        self.ensure_page(layer, head, page_idx);

        let page = &mut self.pages[layer][head][page_idx];
        let start = offset_in_page * self.head_dim;
        page.values[start..start + self.head_dim].copy_from_slice(value);
        if offset_in_page >= page.used {
            page.used = offset_in_page + 1;
        }
    }

    /// Get all cached keys for a layer and head up to `seq_len`, assembled into a contiguous buffer.
    pub fn keys_for(&self, layer: usize, head: usize, seq_len: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(seq_len * self.head_dim);
        let head_pages = &self.pages[layer][head];

        for pos in 0..seq_len {
            let page_idx = pos / self.page_size;
            let offset_in_page = pos % self.page_size;

            if page_idx < head_pages.len() {
                let page = &head_pages[page_idx];
                let start = offset_in_page * self.head_dim;
                result.extend_from_slice(&page.keys[start..start + self.head_dim]);
            } else {
                // Page not yet allocated; fill with zeros
                result.extend(std::iter::repeat_n(0.0f32, self.head_dim));
            }
        }

        result
    }

    /// Get all cached values for a layer and head up to `seq_len`.
    pub fn values_for(&self, layer: usize, head: usize, seq_len: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(seq_len * self.head_dim);
        let head_pages = &self.pages[layer][head];

        for pos in 0..seq_len {
            let page_idx = pos / self.page_size;
            let offset_in_page = pos % self.page_size;

            if page_idx < head_pages.len() {
                let page = &head_pages[page_idx];
                let start = offset_in_page * self.head_dim;
                result.extend_from_slice(&page.values[start..start + self.head_dim]);
            } else {
                result.extend(std::iter::repeat_n(0.0f32, self.head_dim));
            }
        }

        result
    }

    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Advance the sequence position by one token.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Reset the cache (deallocate all pages).
    pub fn clear(&mut self) {
        self.seq_len = 0;
        for layer_pages in &mut self.pages {
            for head_pages in layer_pages.iter_mut() {
                head_pages.clear();
            }
        }
    }

    /// Total memory currently allocated by this cache in bytes.
    ///
    /// Only counts allocated pages, not the full capacity.
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total_pages = 0usize;
        for layer_pages in &self.pages {
            for head_pages in layer_pages {
                total_pages += head_pages.len();
            }
        }
        // Each page has keys + values, each of page_size * head_dim floats
        total_pages * self.page_size * self.head_dim * std::mem::size_of::<f32>() * 2
    }

    // `utilization_ratio` (M-19): removed — same reasoning as
    // `KvCache::utilization_ratio` above (zero production callers, and
    // `pub(crate)` fails this crate's own `-D warnings --all-targets` gate
    // via `dead_code`, verified empirically).

    /// Total number of pages allocated.
    pub fn total_pages(&self) -> usize {
        let mut count = 0usize;
        for layer_pages in &self.pages {
            for head_pages in layer_pages {
                count += head_pages.len();
            }
        }
        count
    }

    /// Page size (positions per page).
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Ensure a page exists at the given index, allocating it if needed.
    fn ensure_page(&mut self, layer: usize, head: usize, page_idx: usize) {
        let head_pages = &mut self.pages[layer][head];
        while head_pages.len() <= page_idx {
            head_pages.push(KvPage::new(self.page_size, self.head_dim));
        }
    }
}
