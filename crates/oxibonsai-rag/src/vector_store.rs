//! In-memory flat vector store with configurable distance metric.
//!
//! The [`VectorStore`] holds a flat list of [`VectorEntry`] items.  Search
//! is performed with a brute-force linear scan over all entries, evaluating
//! the configured [`Distance`] metric against the query vector, and keeping
//! only the top-`k` results in a bounded min-heap (no full sort of every
//! candidate — see `VectorStore::scored`).  Memory is `entries x dim x 4`
//! bytes of `f32` vectors (50 000 entries of 384 dimensions is 76.8 MB, about
//! 73 MiB) plus the chunk text, and every query costs `entries x dim`
//! multiply-adds (about 19 million for that corpus) plus an `O(n log k)`
//! top-`k` heap; there is no index structure, so latency grows linearly with
//! the corpus. A hierarchical-navigable-small-world (HNSW) index behind an
//! `ann` feature is a deliberately deferred, project-sized item (see
//! `TODO.md`); until then, larger corpora are out of scope for this crate.
//!
//! # Scoring and the `min_score` sign convention
//!
//! Scoring semantics are unified by [`Distance::to_score`]: similarity
//! metrics (Cosine, DotProduct) use their raw value as the score, whereas
//! true distances (Euclidean, Angular, Hamming) are *negated* so that
//! "higher is better" sorting always yields the closest match first.
//!
//! This has a consequence callers must know before setting `min_score`:
//! for a similarity metric, scores are
//! naturally in a "the bigger the more alike" range you would expect
//! (`[-1, 1]` for Cosine, unbounded for DotProduct). For a true-distance
//! metric, every achievable score is **`<= 0`** (it is `-distance`, and a
//! distance is never negative), so a `min_score` picked with similarity
//! metrics in mind — e.g. the ergonomic-looking `0.0` — silently rejects
//! every result except an exact (`distance == 0`) duplicate. There is no
//! single threshold that means the same thing across metrics, which is why
//! [`crate::retriever::RetrieverConfig::min_score`] no longer defaults to
//! `0.0`; see that field's documentation for the chosen default and how to
//! pick a metric-appropriate threshold.
//!
//! # Normalisation
//!
//! Only [`Distance::Cosine`] pre-normalises stored/query vectors — it is
//! the metric's defining transform. Every other metric, **including
//! [`Distance::DotProduct`]**, sees vectors exactly as supplied: magnitude
//! is the whole point of choosing DotProduct over Cosine, so silently
//! normalising it would make the two indistinguishable.
//!
//! # NaN / Inf guards
//!
//! Any non-finite value in an inserted vector or the query vector is
//! rejected with [`RagError::NonFinite`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::chunker::Chunk;
use crate::distance::Distance;
use crate::error::RagError;
use crate::metadata_filter::MetadataFilter;

// ─────────────────────────────────────────────────────────────────────────────
// Math primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length slices.
///
/// Returns 0.0 if either slice is empty or they have different lengths.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalise `v` in place.
///
/// If the Euclidean norm is smaller than `1e-10` the vector is left
/// unchanged to prevent NaN propagation.
#[inline]
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Both vectors are assumed to be *unit vectors* (L2-normalised).  Under
/// that assumption, cosine similarity == dot product and the denominator
/// can be skipped.
///
/// Returns a value in `[-1.0, 1.0]`.  Returns `0.0` for empty or mismatched
/// inputs rather than panicking.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    dot_product(a, b).clamp(-1.0, 1.0)
}

/// Whether `distance` requires its stored/query vectors to be pre-normalised.
///
/// Only [`Distance::Cosine`] does: it is mathematically *defined* as the dot
/// product of unit vectors. Every other metric — most importantly
/// [`Distance::DotProduct`], whose entire purpose is to be sensitive to
/// vector magnitude — must see vectors exactly as supplied.
///
/// [`Distance`] lives in a sibling module, so this
/// predicate is `VectorStore`-local (a free function, not a method on
/// `Distance`) rather than the `Distance::normalizes_inputs()` the finder
/// proposed; it is used consistently at both call sites below ([`VectorStore::insert`]
/// and [`VectorStore::scored`]) so the classification cannot drift between them.
#[inline]
fn requires_normalized_inputs(distance: Distance) -> bool {
    matches!(distance, Distance::Cosine)
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorEntry & SearchResult
// ─────────────────────────────────────────────────────────────────────────────

/// A single indexed entry in the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Unique identifier assigned at insertion time. Stable across
    /// `delete`/`update` calls (ids are never reused or shifted), so a
    /// [`SearchResult::id`] captured before a mutation stays meaningful
    /// after it.
    pub id: usize,
    /// Stored embedding vector. Only [`Distance::Cosine`] L2-normalises it
    /// on insert; every other metric (`DotProduct`, `Euclidean`,
    /// `Angular`, `Hamming`) stores it exactly as supplied — see
    /// `requires_normalized_inputs` in this module.
    pub vector: Vec<f32>,
    /// The chunk this entry was derived from.
    pub chunk: Chunk,
}

/// A result returned by a similarity / distance search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unified "higher is better" score (see [`Distance::to_score`]).  For
    /// similarity metrics this equals the raw similarity; for true
    /// distances it equals the negative of the raw distance — see this
    /// module's documentation for what that means for `min_score`.
    pub score: f32,
    /// The chunk associated with this result.
    pub chunk: Chunk,
    /// The entry's unique identifier in the store.
    pub id: usize,
}

/// Order [`SearchResult`]s by score for use in a bounded min-heap: the
/// *smallest* score is the `BinaryHeap`'s greatest element, so
/// `BinaryHeap::pop` evicts the current top-k's weakest candidate.  Uses
/// `f32::total_cmp` for a real total order (scores are already guaranteed
/// finite by the NaN/Inf guards in [`VectorStore::scored`], but this keeps
/// the `Ord` impl correct regardless).
struct HeapEntry(SearchResult);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.score == other.0.score
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.score.total_cmp(&self.0.score)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorStore
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory flat vector store backed by a `Vec<VectorEntry>`.
///
/// The configured [`Distance`] controls both how vectors are *stored*
/// (only [`Distance::Cosine`] pre-normalises; every other metric stores
/// verbatim) and how queries are scored.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VectorStore {
    entries: Vec<VectorEntry>,
    dim: usize,
    #[serde(default)]
    distance: Distance,
    /// Monotonically increasing counter for the next assigned id. Kept
    /// separate from `entries.len()` so that ids stay unique even after
    /// [`VectorStore::delete`] shrinks `entries`: reusing
    /// `entries.len()` as the next id would let a post-delete `insert`
    /// collide with a surviving entry's id.
    #[serde(default)]
    next_id: usize,
}

impl VectorStore {
    /// Create an empty cosine-similarity store for vectors of dim `dim`.
    pub fn new(dim: usize) -> Self {
        Self::new_with_distance(dim, Distance::default())
    }

    /// Create an empty store with a specific [`Distance`] metric.
    pub fn new_with_distance(dim: usize, distance: Distance) -> Self {
        Self {
            entries: Vec::new(),
            dim,
            distance,
            next_id: 0,
        }
    }

    /// Insert a vector+chunk pair into the store.
    ///
    /// Behaviour depends on the store's [`Distance`] — see
    /// `requires_normalized_inputs` and this module's documentation.
    ///
    /// Returns the assigned entry id (unique for the lifetime of the store,
    /// never reused even after a [`VectorStore::delete`]). Errors:
    ///
    /// - [`RagError::DimensionMismatch`] for wrong-size vectors.
    /// - [`RagError::NonFinite`] for `NaN` / `±∞` entries.
    pub fn insert(&mut self, mut vector: Vec<f32>, chunk: Chunk) -> Result<usize, RagError> {
        if vector.len() != self.dim {
            return Err(RagError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(RagError::NonFinite);
        }
        if requires_normalized_inputs(self.distance) {
            l2_normalize(&mut vector);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(VectorEntry { id, vector, chunk });
        Ok(id)
    }

    /// Remove the entry with `id` from the store.
    ///
    /// Scans for `id` rather than assuming `id == position` (ids and
    /// positions can diverge after any prior `delete`). Returns the removed
    /// entry, or `None` if no entry has `id`. `O(n)`.
    pub fn delete(&mut self, id: usize) -> Option<VectorEntry> {
        let pos = self.entries.iter().position(|e| e.id == id)?;
        Some(self.entries.remove(pos))
    }

    /// Remove every entry whose [`Chunk::doc_id`] equals `doc_id`.
    ///
    /// Returns the number of entries removed. Intended for GDPR/APPI-style
    /// deletion requests and re-crawling a changed source document, which
    /// previously required a full `clear()` and re-embed of the whole
    /// corpus.
    pub fn delete_by_doc_id(&mut self, doc_id: usize) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.chunk.doc_id != doc_id);
        before - self.entries.len()
    }

    /// Replace the vector and chunk of the entry with `id`, in place.
    ///
    /// The entry keeps its original `id`. `vector` is validated and
    /// normalised exactly as in [`VectorStore::insert`]. Returns `Ok(true)`
    /// if an entry with `id` was found and updated, `Ok(false)` if no entry
    /// has `id`, or an `Err` if `vector` is invalid.
    pub fn update(
        &mut self,
        id: usize,
        mut vector: Vec<f32>,
        chunk: Chunk,
    ) -> Result<bool, RagError> {
        if vector.len() != self.dim {
            return Err(RagError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(RagError::NonFinite);
        }
        if requires_normalized_inputs(self.distance) {
            l2_normalize(&mut vector);
        }
        match self.entries.iter_mut().find(|e| e.id == id) {
            Some(entry) => {
                entry.vector = vector;
                entry.chunk = chunk;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Return the top-`top_k` entries by score.
    ///
    /// The query vector is normalised internally when the metric is
    /// [`Distance::Cosine`]; it is not mutated. Results are returned in
    /// descending score order (see this module's documentation for
    /// polarity).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        self.search_with_threshold(query, top_k, f32::NEG_INFINITY)
    }

    /// Like [`Self::search`] but discards results whose score is below
    /// `min_score`. See this module's documentation for what `min_score`
    /// means for a non-similarity [`Distance`].
    pub fn search_with_threshold(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        self.scored(query, top_k, min_score, None)
    }

    /// Search filtered by a [`MetadataFilter`], honouring `min_score`
    /// exactly like [`Self::search_with_threshold`] (a
    /// prior version hard-coded `f32::NEG_INFINITY` here, so adding a
    /// metadata filter silently *widened* the result set past what
    /// `min_score` alone would ever allow).
    ///
    /// Filter evaluation is interleaved with scoring: the metric is
    /// evaluated only for entries that pass the filter.
    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: f32,
        filter: &MetadataFilter,
    ) -> Result<Vec<SearchResult>, RagError> {
        filter.validate()?;
        Ok(self.scored(query, top_k, min_score, Some(filter)))
    }

    /// Core scoring loop shared by every search entry point.
    ///
    /// Builds each [`SearchResult`] directly from the borrowed [`VectorEntry`]
    /// while scanning, using a bounded min-heap of size `top_k`
    /// (`O(n log k)`) instead of collecting every passing candidate and
    /// fully sorting it (`O(n log n)`) — this is both a
    /// performance fix and, because it never re-indexes `self.entries` by
    /// id afterwards, a correctness fix: the previous
    /// implementation collected `(score, id)` pairs and looked the chunk
    /// back up via `self.entries[id]` *after* sorting, which assumed
    /// `id == position` and both panicked (id out of bounds) and — the
    /// sharper failure — silently returned the *wrong* chunk for a
    /// correct score whenever a restored or edited snapshot's ids were
    /// dense-but-permuted.
    fn scored(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: f32,
        filter: Option<&MetadataFilter>,
    ) -> Vec<SearchResult> {
        if self.entries.is_empty() || top_k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Vec::new();
        }

        // Prepare the query according to metric semantics.
        let prepared: Vec<f32> = if requires_normalized_inputs(self.distance) {
            let mut q = query.to_vec();
            l2_normalize(&mut q);
            q
        } else {
            query.to_vec()
        };

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_k.saturating_add(1));
        for entry in &self.entries {
            if let Some(f) = filter {
                if !f.matches(&entry.chunk.metadata) {
                    continue;
                }
            }
            let raw = match self.distance.compute(&prepared, &entry.vector) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let score = self.distance.to_score(raw);
            if score < min_score {
                continue;
            }

            if heap.len() < top_k {
                heap.push(HeapEntry(SearchResult {
                    score,
                    chunk: entry.chunk.clone(),
                    id: entry.id,
                }));
            } else if let Some(worst) = heap.peek() {
                if score > worst.0.score {
                    heap.pop();
                    heap.push(HeapEntry(SearchResult {
                        score,
                        chunk: entry.chunk.clone(),
                        id: entry.id,
                    }));
                }
            }
        }

        let mut results: Vec<SearchResult> = heap.into_iter().map(|h| h.0).collect();
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    /// Number of entries currently in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries from the store (preserves the configured
    /// dimension and distance metric; does **not** reset the id counter,
    /// so ids assigned before and after a `clear()` never collide).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Approximate heap memory used by the stored vectors and chunk
    /// texts.  This is a lower-bound estimate: it counts vector bytes and
    /// chunk-text bytes but ignores allocator overhead and struct
    /// padding.
    pub fn memory_usage_bytes(&self) -> usize {
        self.entries.iter().fold(0usize, |acc, e| {
            acc + e.vector.len() * std::mem::size_of::<f32>()
                + e.chunk.text.len()
                + std::mem::size_of::<VectorEntry>()
        })
    }

    /// The embedding dimensionality this store was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The active distance metric.
    pub fn distance(&self) -> Distance {
        self.distance
    }

    /// Borrow the internal entries (used by the persistence layer).
    pub(crate) fn entries(&self) -> &[VectorEntry] {
        &self.entries
    }

    /// Replace the internal entries (used by the persistence layer).
    ///
    /// Also resynchronises the id counter used by [`VectorStore::insert`]
    /// to one past the highest id among `entries`, so a subsequent insert
    /// can never collide with a restored entry's id — even if those ids
    /// are sparse (e.g. the snapshot was taken after some `delete` calls).
    pub(crate) fn set_entries(&mut self, entries: Vec<VectorEntry>) {
        self.next_id = entries.iter().map(|e| e.id).max().map_or(0, |m| m + 1);
        self.entries = entries;
    }

    /// Move every entry from `other` into `self`, re-assigning ids from
    /// `self`'s own id counter so they cannot collide with `self`'s
    /// existing entries. Used by [`crate::retriever::Retriever`] to commit a
    /// staged (temporary) store only after every chunk of a document
    /// embedded and inserted successfully.
    pub(crate) fn merge_from(&mut self, other: VectorStore) {
        for mut entry in other.entries {
            entry.id = self.next_id;
            self.next_id += 1;
            self.entries.push(entry);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str) -> Chunk {
        Chunk::new(text.to_string(), 0, 0, 0)
    }

    // ── RAG-EVAL-IMG-08: DotProduct is not silently normalised ──────────────

    #[test]
    fn dot_product_store_is_sensitive_to_magnitude() {
        let mut store = VectorStore::new_with_distance(2, Distance::DotProduct);
        store
            .insert(vec![10.0, 0.0], chunk("long"))
            .expect("insert");
        store
            .insert(vec![1.0, 0.0], chunk("short"))
            .expect("insert");
        let results = store.search(&[1.0, 0.0], 2);
        assert_eq!(
            results[0].chunk.text, "long",
            "the higher-norm vector must score higher under DotProduct"
        );
        assert!(
            (results[0].score - 10.0).abs() < 1e-4,
            "expected raw (unnormalised) dot product ~10.0, got {}",
            results[0].score
        );
    }

    #[test]
    fn cosine_store_still_normalises() {
        let mut store = VectorStore::new_with_distance(2, Distance::Cosine);
        store
            .insert(vec![10.0, 0.0], chunk("long"))
            .expect("insert");
        let results = store.search(&[1.0, 0.0], 1);
        assert!(
            (results[0].score - 1.0).abs() < 1e-4,
            "cosine score of a co-directional vector must be ~1.0 regardless of magnitude, got {}",
            results[0].score
        );
    }

    // ── RAG-EVAL-IMG-07: search_filtered honours min_score ──────────────────

    #[test]
    fn search_filtered_honours_min_score() {
        let mut store = VectorStore::new(2);
        store
            .insert(vec![1.0, 0.0], chunk("aligned"))
            .expect("insert");
        store
            .insert(vec![-1.0, 0.0], chunk("opposite"))
            .expect("insert");
        let filter = MetadataFilter::All(vec![]);
        let results = store
            .search_filtered(&[1.0, 0.0], 10, 0.5, &filter)
            .expect("search_filtered");
        assert_eq!(
            results.len(),
            1,
            "min_score=0.5 must exclude the opposite vector"
        );
        assert_eq!(results[0].chunk.text, "aligned");
    }

    #[test]
    fn search_filtered_with_all_empty_matches_search_with_threshold() {
        let mut store = VectorStore::new(2);
        store.insert(vec![1.0, 0.0], chunk("a")).expect("insert");
        store.insert(vec![0.0, 1.0], chunk("b")).expect("insert");
        let unfiltered = store.search_with_threshold(&[1.0, 0.0], 10, -1.0);
        let filtered = store
            .search_filtered(&[1.0, 0.0], 10, -1.0, &MetadataFilter::All(vec![]))
            .expect("search_filtered");
        assert_eq!(unfiltered.len(), filtered.len());
        for (u, f) in unfiltered.iter().zip(filtered.iter()) {
            assert_eq!(u.id, f.id);
            assert!((u.score - f.score).abs() < 1e-6);
        }
    }

    // ── RAG-EVAL-IMG-22: delete / update / id stability ──────────────────────

    #[test]
    fn delete_removes_by_id_not_position() {
        // Euclidean (not the default Cosine): with 1-D vectors, Cosine
        // normalisation collapses any positive value to the same unit
        // vector `[1.0]`, making `a`/`b`/`c` indistinguishable by score and
        // defeating the point of this test. Euclidean distance actually
        // discriminates them.
        let mut store = VectorStore::new_with_distance(1, Distance::Euclidean);
        let a = store.insert(vec![1.0], chunk("a")).expect("insert a");
        let b = store.insert(vec![2.0], chunk("b")).expect("insert b");
        let c = store.insert(vec![3.0], chunk("c")).expect("insert c");
        assert!(store.delete(b).is_some());
        assert_eq!(store.len(), 2);
        // `c`'s position shifted down after removing `b`, but its id (and
        // the chunk `scored()` returns for it) must be unaffected.
        let results = store.search(&[3.0], 1);
        assert_eq!(results[0].id, c);
        assert_eq!(results[0].chunk.text, "c");
        assert!(store.delete(a).is_some());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn delete_unknown_id_returns_none() {
        let mut store = VectorStore::new(1);
        store.insert(vec![1.0], chunk("a")).expect("insert");
        assert!(store.delete(999).is_none());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn insert_after_delete_never_collides_with_a_live_id() {
        let mut store = VectorStore::new(1);
        let a = store.insert(vec![1.0], chunk("a")).expect("insert a");
        let b = store.insert(vec![2.0], chunk("b")).expect("insert b");
        store.delete(a).expect("delete a");
        // Before the id-counter fix, the next id was `entries.len()` (== 1
        // after the delete), which collides with `b`'s existing id.
        let d = store.insert(vec![4.0], chunk("d")).expect("insert d");
        assert_ne!(d, b, "new id must not collide with a surviving entry's id");
        let ids: Vec<usize> = store.entries().iter().map(|e| e.id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "ids must stay unique: {ids:?}");
    }

    #[test]
    fn update_replaces_vector_and_chunk_keeping_id() {
        let mut store = VectorStore::new(1);
        let a = store.insert(vec![1.0], chunk("a")).expect("insert");
        let updated = store
            .update(a, vec![5.0], chunk("a-v2"))
            .expect("update should not error");
        assert!(updated);
        let results = store.search(&[5.0], 1);
        assert_eq!(results[0].id, a);
        assert_eq!(results[0].chunk.text, "a-v2");
    }

    #[test]
    fn update_unknown_id_returns_false() {
        let mut store = VectorStore::new(1);
        store.insert(vec![1.0], chunk("a")).expect("insert");
        let updated = store.update(999, vec![1.0], chunk("x")).expect("update");
        assert!(!updated);
    }

    #[test]
    fn update_rejects_wrong_dimension() {
        let mut store = VectorStore::new(2);
        let a = store.insert(vec![1.0, 0.0], chunk("a")).expect("insert");
        let err = store.update(a, vec![1.0], chunk("bad"));
        assert!(matches!(err, Err(RagError::DimensionMismatch { .. })));
    }

    #[test]
    fn delete_by_doc_id_removes_every_matching_chunk() {
        let mut store = VectorStore::new(1);
        store
            .insert(vec![1.0], Chunk::new("a1".into(), 1, 0, 0))
            .expect("insert");
        store
            .insert(vec![2.0], Chunk::new("a2".into(), 1, 1, 0))
            .expect("insert");
        store
            .insert(vec![3.0], Chunk::new("b1".into(), 2, 0, 0))
            .expect("insert");
        let removed = store.delete_by_doc_id(1);
        assert_eq!(removed, 2);
        assert_eq!(store.len(), 1);
        assert_eq!(store.search(&[3.0], 1)[0].chunk.text, "b1");
    }

    // ── merge_from ────────────────────────────────────────────────────────

    #[test]
    fn merge_from_reassigns_ids_from_the_destination_counter() {
        let mut dest = VectorStore::new(1);
        dest.insert(vec![1.0], chunk("d0")).expect("insert");
        dest.insert(vec![2.0], chunk("d1")).expect("insert");

        let mut staged = VectorStore::new(1);
        staged.insert(vec![3.0], chunk("s0")).expect("insert"); // staged id 0
        staged.insert(vec![4.0], chunk("s1")).expect("insert"); // staged id 1

        dest.merge_from(staged);
        assert_eq!(dest.len(), 4);
        let ids: Vec<usize> = dest.entries().iter().map(|e| e.id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            4,
            "merged ids must not collide with destination ids: {ids:?}"
        );
    }

    // ── RAG-EVAL-IMG-23: bounded top-k matches full-sort semantics ──────────

    #[test]
    fn bounded_top_k_matches_expected_order_and_count() {
        // Distance::Euclidean from a point near the middle: closest points
        // should win regardless of insertion order.
        let mut store = VectorStore::new_with_distance(1, Distance::Euclidean);
        for i in 0..50i32 {
            store
                .insert(vec![i as f32], Chunk::new(format!("v{i}"), 0, 0, 0))
                .expect("insert");
        }
        let results = store.search(&[25.0], 3);
        assert_eq!(results.len(), 3);
        let texts: Vec<&str> = results.iter().map(|r| r.chunk.text.as_str()).collect();
        assert!(texts.contains(&"v25"));
        // Scores must be in descending order.
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    // ── Pre-existing behaviour (regression guards) ──────────────────────────

    #[test]
    fn insert_and_search_unchanged() {
        let mut store = VectorStore::new(4);
        let v1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0, 0.0];
        store
            .insert(v1.clone(), chunk("chunk one"))
            .expect("insert v1");
        store
            .insert(v2.clone(), chunk("chunk two"))
            .expect("insert v2");
        assert_eq!(store.len(), 2);
        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.text, "chunk one");
    }

    #[test]
    fn search_threshold_unchanged() {
        let mut store = VectorStore::new(2);
        store
            .insert(vec![1.0, 0.0], chunk("positive"))
            .expect("insert");
        store
            .insert(vec![-1.0, 0.0], chunk("negative"))
            .expect("insert");
        let results = store.search_with_threshold(&[1.0, 0.0], 10, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "positive");
    }

    #[test]
    fn dimension_mismatch_unchanged() {
        let mut store = VectorStore::new(4);
        let result = store.insert(vec![1.0, 2.0], chunk("wrong dim"));
        assert!(matches!(
            result,
            Err(RagError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
    }
}
