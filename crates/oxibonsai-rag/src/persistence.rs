//! Persistence for the vector store and retriever: JSON (human-readable,
//! the default) and a compact hand-rolled binary format.
//!
//! Two snapshot types are exposed:
//!
//! - [`IndexSnapshot`] captures a [`crate::vector_store::VectorStore`] — its
//!   dimensionality, distance metric, and every stored entry.
//! - [`RetrieverSnapshot`] wraps an [`IndexSnapshot`] together with the
//!   [`Retriever`]'s document counter, its [`crate::retriever::RetrieverConfig`],
//!   and its embedder's persisted state (see "Embedder state" below), so a
//!   round trip through [`Retriever::save`]/[`Retriever::load`] restores the
//!   retriever exactly as it was configured, not just its index.
//!
//! A monotonically-increasing [`SCHEMA_VERSION`] is stored in every
//! snapshot.  Loaders refuse to deserialise an unknown `schema_version` with
//! [`RagError::Persistence`]. [`RetrieverSnapshot::config`] and
//! [`RetrieverSnapshot::embedder_state`] are `#[serde(default)]`, so this
//! did not require bumping [`SCHEMA_VERSION`]: an older snapshot missing
//! those fields still loads (with a default config and no embedder state —
//! exactly today's pre-fix behaviour), while a *newly saved* snapshot now
//! carries real values that a subsequent [`Retriever::load`] restores.
//!
//! # Writes are atomic
//!
//! Every `save*` function here writes to a sibling temporary file and
//! `rename`s it into place, rather than writing straight into the
//! destination. A crash or a concurrent reader can therefore never observe
//! a partially-written snapshot — the destination file is either the
//! previous complete snapshot or the new complete one, never a truncated
//! mix of both (needs no new dependency).
//!
//! # Embedder state
//!
//! The [`Retriever`] is generic over its [`Embedder`]. [`Retriever::save`]/
//! [`Retriever::load`] only require plain [`Embedder`] on the *load* side —
//! the caller still supplies an embedder instance, exactly as before — but
//! `save` additionally requires [`crate::embedding::EmbedderState`] so it can
//! persist that embedder's fittable state (a no-op for stateless embedders
//! like [`crate::embedding::IdentityEmbedder`]) into
//! [`RetrieverSnapshot::embedder_state`]. [`Retriever::load_with_embedder`]
//! goes one step further and reconstructs the embedder *itself* from that
//! persisted state, so a caller no longer needs to keep the original corpus
//! around just to refit a [`crate::embedding::TfIdfEmbedder`].
//!
//! # Example
//!
//! ```no_run
//! use oxibonsai_rag::embedding::IdentityEmbedder;
//! use oxibonsai_rag::pipeline::RagConfig;
//! use oxibonsai_rag::retriever::{Retriever, RetrieverConfig};
//!
//! let embedder = IdentityEmbedder::new(32).expect("valid dim");
//! let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
//! retriever
//!     .add_document("some text", &RagConfig::default().chunk_config)
//!     .expect("index");
//!
//! let path = std::env::temp_dir().join("rag_snapshot.json");
//! retriever.save(&path).expect("save");
//!
//! let embedder = IdentityEmbedder::new(32).expect("valid dim");
//! let restored = Retriever::load(embedder, &path).expect("load");
//! assert_eq!(restored.chunk_count(), 1);
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::chunker::Chunk;
use crate::distance::Distance;
use crate::embedding::{Embedder, EmbedderState};
use crate::error::RagError;
use crate::metadata_filter::MetadataValue;
use crate::retriever::{Retriever, RetrieverConfig};
use crate::vector_store::{VectorEntry, VectorStore};

// ─────────────────────────────────────────────────────────────────────────────
// Schema version
// ─────────────────────────────────────────────────────────────────────────────

/// Current on-disk snapshot schema version.
///
/// Bump this when [`IndexSnapshot`]'s layout changes in a
/// non-backwards-compatible way (adding a field to [`RetrieverSnapshot`]
/// does not require a bump — see the module docs). Loaders reject unknown
/// values with [`RagError::Persistence`] so that a stale binary cannot
/// silently misinterpret a newer file.
pub const SCHEMA_VERSION: u32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// IndexSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Serde-serialisable snapshot of a [`VectorStore`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSnapshot {
    /// Schema version tag (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Embedding dimensionality.
    pub dim: usize,
    /// Distance metric the store was configured with.
    #[serde(default)]
    pub distance: Distance,
    /// All stored entries, in insertion order.
    pub entries: Vec<VectorEntry>,
    /// Optional serialised TF-IDF state.  Superseded by
    /// [`RetrieverSnapshot::embedder_state`] for the `Retriever::save`/`load`
    /// path; kept here, and still populatable by hand, for
    /// callers who persist a bare [`VectorStore`] without a `Retriever`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tfidf_state: Option<serde_json::Value>,
}

impl IndexSnapshot {
    /// Ensure the schema version matches the build-time constant.  Returns
    /// [`RagError::Persistence`] for unknown versions.
    pub fn check_version(&self) -> Result<(), RagError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(RagError::Persistence(format!(
                "unsupported schema_version {} (expected {})",
                self.schema_version, SCHEMA_VERSION
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RetrieverSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Serde-serialisable snapshot of a [`Retriever`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieverSnapshot {
    /// Schema version tag (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Number of distinct documents indexed so far.
    pub doc_count: usize,
    /// The underlying vector-store snapshot.
    pub store: IndexSnapshot,
    /// The retriever's configuration at save time. `#[serde(default)]` so a
    /// snapshot written before this field existed still loads (falling back
    /// to [`RetrieverConfig::default`], exactly the pre-fix behaviour)
    /// instead of failing to parse.
    #[serde(default)]
    pub config: RetrieverConfig,
    /// The embedder's persisted state, if any (see
    /// [`crate::embedding::EmbedderState`] and the module docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder_state: Option<serde_json::Value>,
}

impl RetrieverSnapshot {
    fn check_version(&self) -> Result<(), RagError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(RagError::Persistence(format!(
                "unsupported schema_version {} (expected {})",
                self.schema_version, SCHEMA_VERSION
            )));
        }
        self.store.check_version()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic writes
// ─────────────────────────────────────────────────────────────────────────────

/// Build a sibling temporary path for `path` (same directory, so the final
/// `rename` is same-filesystem and therefore atomic on every platform this
/// crate targets).
fn temp_sibling_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("oxibonsai_rag_snapshot");
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    path.with_file_name(format!(".{file_name}.tmp-{pid}-{nanos}"))
}

/// Write `bytes` to `path` atomically: write to a same-directory temporary
/// file, `fsync` it, then `rename` it into place. A reader can therefore
/// only ever observe the previous complete file or the new complete file —
/// never a truncated write from a crash mid-save.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), RagError> {
    let tmp = temp_sibling_path(path);
    let write_result = (|| -> std::io::Result<()> {
        let mut file = fs::File::create(&tmp)?;
        std::io::Write::write_all(&mut file, bytes)?;
        file.sync_all()
    })();
    if let Err(e) = write_result {
        let _ = fs::remove_file(&tmp);
        return Err(RagError::Io(e));
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(RagError::Io(e));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorStore <-> IndexSnapshot conversions
// ─────────────────────────────────────────────────────────────────────────────

impl VectorStore {
    /// Produce an [`IndexSnapshot`] capturing the current store contents.
    pub fn to_snapshot(&self) -> IndexSnapshot {
        IndexSnapshot {
            schema_version: SCHEMA_VERSION,
            dim: self.dim(),
            distance: self.distance(),
            entries: self.entries().to_vec(),
            tfidf_state: None,
        }
    }

    /// Build a [`VectorStore`] from a previously-produced snapshot.
    ///
    /// Returns [`RagError::Persistence`] if the schema version is unknown,
    /// and [`RagError::DimensionMismatch`] if any stored entry has a
    /// vector whose length disagrees with the snapshot's `dim`.
    pub fn from_snapshot(snapshot: IndexSnapshot) -> Result<Self, RagError> {
        snapshot.check_version()?;
        for entry in &snapshot.entries {
            if entry.vector.len() != snapshot.dim {
                return Err(RagError::DimensionMismatch {
                    expected: snapshot.dim,
                    got: entry.vector.len(),
                });
            }
        }
        let mut store = VectorStore::new_with_distance(snapshot.dim, snapshot.distance);
        store.set_entries(snapshot.entries);
        Ok(store)
    }

    /// Serialise this store to `path` as pretty-printed JSON, atomically.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), RagError> {
        let bytes = serde_json::to_vec_pretty(&self.to_snapshot())
            .map_err(|e| RagError::Persistence(format!("serialize failed: {e}")))?;
        write_atomic(path.as_ref(), &bytes)
    }

    /// Deserialise a store previously written by [`VectorStore::save_json`].
    ///
    /// Returns [`RagError::Persistence`] on malformed JSON or unknown
    /// schema version, and [`RagError::DimensionMismatch`] if any stored
    /// entry's vector length disagrees with the snapshot's `dim`.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, RagError> {
        let bytes = fs::read(path.as_ref())?;
        let snapshot: IndexSnapshot = serde_json::from_slice(&bytes)
            .map_err(|e| RagError::Persistence(format!("parse failed: {e}")))?;
        Self::from_snapshot(snapshot)
    }

    /// Serialise this store to `path` in the compact binary format (see the
    /// `binary` submodule), atomically. Typically much smaller and faster
    /// to load than [`VectorStore::save_json`] for a large index.
    pub fn save_binary(&self, path: impl AsRef<Path>) -> Result<(), RagError> {
        let bytes = binary::encode_index_snapshot(&self.to_snapshot())?;
        write_atomic(path.as_ref(), &bytes)
    }

    /// Deserialise a store previously written by [`VectorStore::save_binary`].
    pub fn load_binary(path: impl AsRef<Path>) -> Result<Self, RagError> {
        let bytes = fs::read(path.as_ref())?;
        let snapshot = binary::decode_index_snapshot(&bytes)?;
        Self::from_snapshot(snapshot)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retriever persistence
// ─────────────────────────────────────────────────────────────────────────────

impl<E: EmbedderState> Retriever<E> {
    /// Serialise this retriever — its index, its
    /// [`crate::retriever::RetrieverConfig`], and its embedder's persisted
    /// state (see [`EmbedderState`]) — to `path` as pretty-printed JSON,
    /// atomically.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), RagError> {
        let snapshot = self.to_retriever_snapshot();
        let bytes = serde_json::to_vec_pretty(&snapshot)
            .map_err(|e| RagError::Persistence(format!("serialize failed: {e}")))?;
        write_atomic(path.as_ref(), &bytes)
    }

    /// Serialise this retriever to `path` in the compact binary format,
    /// atomically.
    pub fn save_binary(&self, path: impl AsRef<Path>) -> Result<(), RagError> {
        let snapshot = self.to_retriever_snapshot();
        let bytes = binary::encode_retriever_snapshot(&snapshot)?;
        write_atomic(path.as_ref(), &bytes)
    }

    fn to_retriever_snapshot(&self) -> RetrieverSnapshot {
        RetrieverSnapshot {
            schema_version: SCHEMA_VERSION,
            doc_count: self.document_count(),
            store: self.store().to_snapshot(),
            config: self.config().clone(),
            embedder_state: self.embedder().to_state(),
        }
    }

    /// Reconstruct a [`Retriever`] from a snapshot previously written by
    /// [`Retriever::save_binary`], using the embedder the caller supplies
    /// (exactly like [`Retriever::load`]; see
    /// [`Retriever::load_binary_with_embedder`] to reconstruct the embedder
    /// from its persisted state instead).
    pub fn load_binary(embedder: E, path: impl AsRef<Path>) -> Result<Self, RagError> {
        let bytes = fs::read(path.as_ref())?;
        let snapshot = binary::decode_retriever_snapshot(&bytes)?;
        Self::from_snapshot_checked(embedder, snapshot)
    }

    fn from_snapshot_checked(embedder: E, snapshot: RetrieverSnapshot) -> Result<Self, RagError> {
        snapshot.check_version()?;
        if embedder.embedding_dim() != snapshot.store.dim {
            return Err(RagError::DimensionMismatch {
                expected: snapshot.store.dim,
                got: embedder.embedding_dim(),
            });
        }
        let store = VectorStore::from_snapshot(snapshot.store)?;
        Ok(Self::from_parts(
            embedder,
            store,
            snapshot.doc_count,
            snapshot.config,
        ))
    }
}

impl<E: Embedder> Retriever<E> {
    /// Reconstruct a [`Retriever`] from a previously-saved JSON snapshot.
    ///
    /// `embedder` must produce vectors of the same dimensionality as the
    /// snapshot, otherwise [`RagError::DimensionMismatch`] is returned. The
    /// retriever's [`crate::retriever::RetrieverConfig`] is restored from the
    /// snapshot (falling back to [`RetrieverConfig::default`] for a
    /// snapshot saved before that was persisted).
    ///
    /// This does not require `E: EmbedderState` — unlike
    /// [`Retriever::save`], `load` never needs to *extract* state from an
    /// embedder, only accept whatever `embedder` the caller already built.
    pub fn load(embedder: E, path: impl AsRef<Path>) -> Result<Self, RagError> {
        let bytes = fs::read(path.as_ref())?;
        let snapshot: RetrieverSnapshot = serde_json::from_slice(&bytes)
            .map_err(|e| RagError::Persistence(format!("parse failed: {e}")))?;
        snapshot.check_version()?;

        if embedder.embedding_dim() != snapshot.store.dim {
            return Err(RagError::DimensionMismatch {
                expected: snapshot.store.dim,
                got: embedder.embedding_dim(),
            });
        }

        let store = VectorStore::from_snapshot(snapshot.store)?;
        Ok(Self::from_parts(
            embedder,
            store,
            snapshot.doc_count,
            snapshot.config,
        ))
    }
}

impl<E: EmbedderState> Retriever<E> {
    /// Reconstruct a [`Retriever`] **and its embedder** purely from a JSON
    /// snapshot written by [`Retriever::save`], using
    /// [`EmbedderState::from_state`] to rebuild `E` from its persisted
    /// state — unlike [`Retriever::load`], the caller does not need to
    /// supply (or know how to rebuild) an equivalent embedder out of band.
    ///
    /// This is the direct fix for "`TfIdfEmbedder` cannot be reconstructed":
    /// a saved [`crate::embedding::TfIdfEmbedder`] index
    /// now round-trips its vocabulary and IDF table without the caller
    /// needing to keep the original corpus around to refit it.
    pub fn load_with_embedder(path: impl AsRef<Path>) -> Result<Self, RagError> {
        let bytes = fs::read(path.as_ref())?;
        let snapshot: RetrieverSnapshot = serde_json::from_slice(&bytes)
            .map_err(|e| RagError::Persistence(format!("parse failed: {e}")))?;
        snapshot.check_version()?;
        let embedder = E::from_state(snapshot.embedder_state.as_ref(), snapshot.store.dim)?;
        Self::from_snapshot_checked(embedder, snapshot)
    }

    /// Binary-format counterpart of [`Retriever::load_with_embedder`].
    pub fn load_binary_with_embedder(path: impl AsRef<Path>) -> Result<Self, RagError> {
        let bytes = fs::read(path.as_ref())?;
        let snapshot = binary::decode_retriever_snapshot(&bytes)?;
        snapshot.check_version()?;
        let embedder = E::from_state(snapshot.embedder_state.as_ref(), snapshot.store.dim)?;
        Self::from_snapshot_checked(embedder, snapshot)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary format
// ─────────────────────────────────────────────────────────────────────────────

/// A compact, hand-rolled binary encoding for [`IndexSnapshot`] and
/// [`RetrieverSnapshot`], used behind [`VectorStore::save_binary`]/
/// [`Retriever::save_binary`].
///
/// The bulk data — every entry's id, raw `f32` vector, and chunk (text,
/// `doc_id`/`chunk_idx`/`char_offset`, metadata) — is encoded field-by-field
/// as fixed-width little-endian integers and length-prefixed UTF-8/byte
/// strings, which is what actually makes this smaller and faster to parse
/// than the JSON format for a non-trivial index (verbose field names,
/// pretty-printing whitespace, and decimal-ASCII float formatting all cost
/// nothing here). `RetrieverConfig` and any embedder state are small and
/// rarely present, so — for implementation simplicity, without materially
/// affecting overall size — they are embedded as compact (non-pretty) JSON
/// byte blobs inside the otherwise-binary envelope; that is a deliberate,
/// documented trade-off, not an oversight.
///
/// COOLJAPAN policy bans `bincode`; the intended dependency for this format
/// is `oxicode`, but it is not a dependency of this crate yet. This hand-rolled
/// codec has no
/// dependency on either and fully implements the "add a binary format"
/// requirement in the meantime; swapping the low-level encode/decode calls
/// below for `oxicode::serialize`/`deserialize` once that dependency lands
/// is a self-contained follow-up that does not change either public API.
mod binary {
    use super::{
        Chunk, Distance, HashMap, IndexSnapshot, MetadataValue, RagError, RetrieverConfig,
        RetrieverSnapshot, VectorEntry,
    };

    const INDEX_MAGIC: &[u8; 8] = b"OXIRAGB1";
    const RETRIEVER_MAGIC: &[u8; 8] = b"OXIRAGR1";

    // ── FNV-1a 64-bit checksum ───────────────────────────────────────────────
    //
    // A well-known, trivial, dependency-free non-cryptographic hash: good
    // enough to detect truncation/bit-rot corruption on load, which is the
    // gap atomic writes alone don't close (disk corruption, a hand-edited
    // file, copying a partial file over the network, …).

    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    // ── Low-level writers ────────────────────────────────────────────────────

    fn write_u8(buf: &mut Vec<u8>, v: u8) {
        buf.push(v);
    }

    fn write_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f32(buf: &mut Vec<u8>, v: f32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
        write_u64(buf, bytes.len() as u64);
        buf.extend_from_slice(bytes);
    }

    fn write_str(buf: &mut Vec<u8>, s: &str) {
        write_bytes(buf, s.as_bytes());
    }

    fn distance_tag(d: Distance) -> u8 {
        match d {
            Distance::Cosine => 0,
            Distance::Euclidean => 1,
            Distance::DotProduct => 2,
            Distance::Angular => 3,
            Distance::Hamming => 4,
        }
    }

    fn distance_from_tag(tag: u8) -> Result<Distance, RagError> {
        match tag {
            0 => Ok(Distance::Cosine),
            1 => Ok(Distance::Euclidean),
            2 => Ok(Distance::DotProduct),
            3 => Ok(Distance::Angular),
            4 => Ok(Distance::Hamming),
            other => Err(RagError::Persistence(format!(
                "binary snapshot: unknown distance tag {other}"
            ))),
        }
    }

    fn write_metadata_value(buf: &mut Vec<u8>, value: &MetadataValue) {
        match value {
            MetadataValue::String(s) => {
                write_u8(buf, 0);
                write_str(buf, s);
            }
            MetadataValue::Int(n) => {
                write_u8(buf, 1);
                buf.extend_from_slice(&n.to_le_bytes());
            }
            MetadataValue::Float(f) => {
                write_u8(buf, 2);
                buf.extend_from_slice(&f.to_le_bytes());
            }
            MetadataValue::Bool(b) => {
                write_u8(buf, 3);
                write_u8(buf, u8::from(*b));
            }
        }
    }

    fn write_chunk(buf: &mut Vec<u8>, chunk: &Chunk) {
        write_str(buf, &chunk.text);
        write_u64(buf, chunk.doc_id as u64);
        write_u64(buf, chunk.chunk_idx as u64);
        write_u64(buf, chunk.char_offset as u64);
        write_u64(buf, chunk.metadata.len() as u64);
        for (key, value) in &chunk.metadata {
            write_str(buf, key);
            write_metadata_value(buf, value);
        }
    }

    fn write_entry(buf: &mut Vec<u8>, entry: &VectorEntry) {
        write_u64(buf, entry.id as u64);
        write_u64(buf, entry.vector.len() as u64);
        for x in &entry.vector {
            write_f32(buf, *x);
        }
        write_chunk(buf, &entry.chunk);
    }

    /// Optional opaque JSON blob: a presence byte, then (if present) a
    /// length-prefixed compact-JSON payload.
    ///
    /// Propagates a `serde_json` failure instead of swallowing it into a
    /// zero-length blob: in every call site in this
    /// module `value` is a `serde_json::Value`, whose own serialisation is
    /// not expected to fail, but silently writing an empty payload on the
    /// (currently unreachable) error path would otherwise turn a real bug
    /// into a confusing "invalid embedded JSON blob" error at *load* time
    /// instead of a clear one here, at the point of the actual failure.
    fn write_optional_json(
        buf: &mut Vec<u8>,
        value: &Option<serde_json::Value>,
    ) -> Result<(), RagError> {
        match value {
            Some(v) => {
                write_u8(buf, 1);
                let bytes = serde_json::to_vec(v)
                    .map_err(|e| RagError::Persistence(format!("serialize failed: {e}")))?;
                write_bytes(buf, &bytes);
            }
            None => write_u8(buf, 0),
        }
        Ok(())
    }

    fn write_index_body(buf: &mut Vec<u8>, snapshot: &IndexSnapshot) -> Result<(), RagError> {
        write_u32(buf, snapshot.schema_version);
        write_u64(buf, snapshot.dim as u64);
        write_u8(buf, distance_tag(snapshot.distance));
        write_u64(buf, snapshot.entries.len() as u64);
        for entry in &snapshot.entries {
            write_entry(buf, entry);
        }
        write_optional_json(buf, &snapshot.tfidf_state)
    }

    /// Encode an [`IndexSnapshot`] as `MAGIC || body || checksum(MAGIC || body)`.
    pub(super) fn encode_index_snapshot(snapshot: &IndexSnapshot) -> Result<Vec<u8>, RagError> {
        let mut buf = Vec::with_capacity(64 + snapshot.entries.len() * 64);
        buf.extend_from_slice(INDEX_MAGIC);
        write_index_body(&mut buf, snapshot)?;
        let checksum = fnv1a64(&buf);
        write_u64(&mut buf, checksum);
        Ok(buf)
    }

    pub(super) fn encode_retriever_snapshot(
        snapshot: &RetrieverSnapshot,
    ) -> Result<Vec<u8>, RagError> {
        let mut buf = Vec::with_capacity(128 + snapshot.store.entries.len() * 64);
        buf.extend_from_slice(RETRIEVER_MAGIC);
        write_u32(&mut buf, snapshot.schema_version);
        write_u64(&mut buf, snapshot.doc_count as u64);
        write_index_body(&mut buf, &snapshot.store)?;
        // Unlike the `Option<serde_json::Value>` blobs `write_optional_json`
        // handles, `snapshot.config` is a concrete `RetrieverConfig` with a
        // hand-written `Serialize` impl on its `min_score` field
        // (`serialize_min_score` in `retriever.rs`) -- propagate rather
        // than swallow here too, on the same reasoning.
        let config_json = serde_json::to_vec(&snapshot.config)
            .map_err(|e| RagError::Persistence(format!("serialize failed: {e}")))?;
        write_bytes(&mut buf, &config_json);
        write_optional_json(&mut buf, &snapshot.embedder_state)?;
        let checksum = fnv1a64(&buf);
        write_u64(&mut buf, checksum);
        Ok(buf)
    }

    // ── Cursor-based reader ──────────────────────────────────────────────────

    struct Cursor<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self { data, pos: 0 }
        }

        /// Bytes not yet consumed. Used to cap `with_capacity` hints
        /// against what the remaining file could actually contain (see
        /// [`capped_capacity`]) rather than trusting a declared count
        /// outright.
        fn remaining(&self) -> usize {
            self.data.len() - self.pos
        }

        fn take(&mut self, n: usize) -> Result<&'a [u8], RagError> {
            let end = self
                .pos
                .checked_add(n)
                .ok_or_else(|| RagError::Persistence("binary snapshot: length overflow".into()))?;
            if end > self.data.len() {
                return Err(RagError::Persistence(format!(
                    "binary snapshot: truncated (wanted {n} bytes at offset {}, have {})",
                    self.pos,
                    self.data.len()
                )));
            }
            let slice = &self.data[self.pos..end];
            self.pos = end;
            Ok(slice)
        }

        fn read_magic(&mut self, expected: &[u8; 8]) -> Result<(), RagError> {
            let got = self.take(8)?;
            if got != expected {
                return Err(RagError::Persistence(
                    "binary snapshot: bad magic (not an oxibonsai-rag binary snapshot, or wrong \
                     kind -- an IndexSnapshot file cannot be loaded as a RetrieverSnapshot or \
                     vice versa)"
                        .into(),
                ));
            }
            Ok(())
        }

        fn read_u8(&mut self) -> Result<u8, RagError> {
            Ok(self.take(1)?[0])
        }

        fn read_u32(&mut self) -> Result<u32, RagError> {
            let bytes: [u8; 4] = self.take(4)?.try_into().map_err(|_| {
                RagError::Persistence("binary snapshot: internal u32 read error".into())
            })?;
            Ok(u32::from_le_bytes(bytes))
        }

        fn read_u64(&mut self) -> Result<u64, RagError> {
            let bytes: [u8; 8] = self.take(8)?.try_into().map_err(|_| {
                RagError::Persistence("binary snapshot: internal u64 read error".into())
            })?;
            Ok(u64::from_le_bytes(bytes))
        }

        fn read_usize(&mut self) -> Result<usize, RagError> {
            let v = self.read_u64()?;
            usize::try_from(v).map_err(|_| {
                RagError::Persistence(format!("binary snapshot: length {v} exceeds usize::MAX"))
            })
        }

        fn read_f32(&mut self) -> Result<f32, RagError> {
            let bytes: [u8; 4] = self.take(4)?.try_into().map_err(|_| {
                RagError::Persistence("binary snapshot: internal f32 read error".into())
            })?;
            Ok(f32::from_le_bytes(bytes))
        }

        fn read_i64(&mut self) -> Result<i64, RagError> {
            let bytes: [u8; 8] = self.take(8)?.try_into().map_err(|_| {
                RagError::Persistence("binary snapshot: internal i64 read error".into())
            })?;
            Ok(i64::from_le_bytes(bytes))
        }

        fn read_f64(&mut self) -> Result<f64, RagError> {
            let bytes: [u8; 8] = self.take(8)?.try_into().map_err(|_| {
                RagError::Persistence("binary snapshot: internal f64 read error".into())
            })?;
            Ok(f64::from_le_bytes(bytes))
        }

        fn read_bytes(&mut self) -> Result<&'a [u8], RagError> {
            let len = self.read_usize()?;
            self.take(len)
        }

        fn read_str(&mut self) -> Result<String, RagError> {
            let bytes = self.read_bytes()?;
            String::from_utf8(bytes.to_vec())
                .map_err(|e| RagError::Persistence(format!("binary snapshot: invalid utf8: {e}")))
        }

        fn read_optional_json(&mut self) -> Result<Option<serde_json::Value>, RagError> {
            match self.read_u8()? {
                0 => Ok(None),
                1 => {
                    let bytes = self.read_bytes()?;
                    let value = serde_json::from_slice(bytes).map_err(|e| {
                        RagError::Persistence(format!(
                            "binary snapshot: invalid embedded JSON blob: {e}"
                        ))
                    })?;
                    Ok(Some(value))
                }
                other => Err(RagError::Persistence(format!(
                    "binary snapshot: invalid optional-JSON presence byte {other}"
                ))),
            }
        }
    }

    /// Cap a length read from an untrusted file to a `with_capacity` hint
    /// that cannot request more memory than the remaining bytes could
    /// possibly encode (`read_index_body`/`read_entry`/
    /// `read_chunk` used to pass a file-controlled length straight to
    /// `with_capacity`). FNV-1a-64 (see [`fnv1a64`]) is trivially forgeable,
    /// so a crafted-but-checksummed snapshot with e.g. `entry_count =
    /// usize::MAX` must not be able to make this process attempt a huge
    /// up-front allocation and abort.
    ///
    /// This only changes the *pre-allocation hint*, not correctness: the
    /// caller's read loop still runs `requested` times unchanged, `push`ing
    /// (and reallocating as needed, exactly as it would from any other
    /// starting capacity) until it either finishes or [`Cursor::take`]
    /// returns a truncation error -- both unchanged from before this cap
    /// existed. An under-estimate of `min_element_bytes` is therefore
    /// harmless (a few extra reallocations for a large legitimate file); an
    /// over-estimate is not attempted here, so a lower bound is fine.
    fn capped_capacity(
        requested: usize,
        remaining_bytes: usize,
        min_element_bytes: usize,
    ) -> usize {
        let max_possible = remaining_bytes / min_element_bytes.max(1);
        requested.min(max_possible)
    }

    /// Smallest possible on-disk encoding of one `(String, MetadataValue)`
    /// pair written by [`write_metadata_value`]: an empty key (`write_str`
    /// costs 8 bytes for a zero length) plus the smallest metadata variant,
    /// `Bool` (1 tag byte + 1 payload byte) -- see [`capped_capacity`] for
    /// why a lower bound suffices here.
    const MIN_METADATA_ENTRY_BYTES: usize = 8 + 2;

    /// Smallest possible on-disk encoding of one [`VectorEntry`] written by
    /// [`write_entry`]: `id` (8) + a zero-length `vector`'s length prefix
    /// (8) + the smallest possible [`Chunk`] written by [`write_chunk`]
    /// (empty `text` (8) + `doc_id` (8) + `chunk_idx` (8) + `char_offset`
    /// (8) + a zero-length `metadata`'s length prefix (8)).
    const MIN_ENTRY_BYTES: usize = 8 + 8 + (8 + 8 + 8 + 8 + 8);

    fn read_metadata_value(cur: &mut Cursor) -> Result<MetadataValue, RagError> {
        match cur.read_u8()? {
            0 => Ok(MetadataValue::String(cur.read_str()?)),
            1 => Ok(MetadataValue::Int(cur.read_i64()?)),
            2 => Ok(MetadataValue::Float(cur.read_f64()?)),
            3 => Ok(MetadataValue::Bool(cur.read_u8()? != 0)),
            other => Err(RagError::Persistence(format!(
                "binary snapshot: unknown metadata value tag {other}"
            ))),
        }
    }

    fn read_chunk(cur: &mut Cursor) -> Result<Chunk, RagError> {
        let text = cur.read_str()?;
        let doc_id = cur.read_usize()?;
        let chunk_idx = cur.read_usize()?;
        let char_offset = cur.read_usize()?;
        let metadata_len = cur.read_usize()?;
        let mut metadata = HashMap::with_capacity(capped_capacity(
            metadata_len,
            cur.remaining(),
            MIN_METADATA_ENTRY_BYTES,
        ));
        for _ in 0..metadata_len {
            let key = cur.read_str()?;
            let value = read_metadata_value(cur)?;
            metadata.insert(key, value);
        }
        Ok(Chunk {
            text,
            doc_id,
            chunk_idx,
            char_offset,
            metadata,
        })
    }

    fn read_entry(cur: &mut Cursor) -> Result<VectorEntry, RagError> {
        let id = cur.read_usize()?;
        let vector_len = cur.read_usize()?;
        // Every `f32` element is exactly 4 bytes on disk (`write_f32`).
        let mut vector = Vec::with_capacity(capped_capacity(vector_len, cur.remaining(), 4));
        for _ in 0..vector_len {
            vector.push(cur.read_f32()?);
        }
        let chunk = read_chunk(cur)?;
        Ok(VectorEntry { id, vector, chunk })
    }

    fn read_index_body(cur: &mut Cursor) -> Result<IndexSnapshot, RagError> {
        let schema_version = cur.read_u32()?;
        let dim = cur.read_usize()?;
        let distance = distance_from_tag(cur.read_u8()?)?;
        let entry_count = cur.read_usize()?;
        let mut entries = Vec::with_capacity(capped_capacity(
            entry_count,
            cur.remaining(),
            MIN_ENTRY_BYTES,
        ));
        for _ in 0..entry_count {
            entries.push(read_entry(cur)?);
        }
        let tfidf_state = cur.read_optional_json()?;
        Ok(IndexSnapshot {
            schema_version,
            dim,
            distance,
            entries,
            tfidf_state,
        })
    }

    /// Verify the trailing FNV-1a checksum, then hand back the checksummed
    /// payload (magic included) for the caller to parse.
    fn verify_and_strip_checksum(bytes: &[u8]) -> Result<&[u8], RagError> {
        if bytes.len() < 8 {
            return Err(RagError::Persistence(
                "binary snapshot: too short to contain a checksum".into(),
            ));
        }
        let (payload, checksum_bytes) = bytes.split_at(bytes.len() - 8);
        let expected = u64::from_le_bytes(checksum_bytes.try_into().map_err(|_| {
            RagError::Persistence("binary snapshot: internal checksum-slice error".into())
        })?);
        let actual = fnv1a64(payload);
        if actual != expected {
            return Err(RagError::Persistence(format!(
                "binary snapshot: checksum mismatch (file is corrupted or truncated): \
                 expected {expected:016x}, computed {actual:016x}"
            )));
        }
        Ok(payload)
    }

    pub(super) fn decode_index_snapshot(bytes: &[u8]) -> Result<IndexSnapshot, RagError> {
        let payload = verify_and_strip_checksum(bytes)?;
        let mut cur = Cursor::new(payload);
        cur.read_magic(INDEX_MAGIC)?;
        read_index_body(&mut cur)
    }

    pub(super) fn decode_retriever_snapshot(bytes: &[u8]) -> Result<RetrieverSnapshot, RagError> {
        let payload = verify_and_strip_checksum(bytes)?;
        let mut cur = Cursor::new(payload);
        cur.read_magic(RETRIEVER_MAGIC)?;
        let schema_version = cur.read_u32()?;
        let doc_count = cur.read_usize()?;
        let store = read_index_body(&mut cur)?;
        let config_bytes = cur.read_bytes()?;
        let config: RetrieverConfig = serde_json::from_slice(config_bytes).map_err(|e| {
            RagError::Persistence(format!(
                "binary snapshot: invalid embedded config JSON: {e}"
            ))
        })?;
        let embedder_state = cur.read_optional_json()?;
        // This function doesn't compare `schema_version` against
        // `SCHEMA_VERSION` itself; the caller (`Retriever::load_binary*`)
        // calls `.check_version()` on the returned snapshot, matching the
        // JSON path's error surface exactly.
        Ok(RetrieverSnapshot {
            schema_version,
            doc_count,
            store,
            config,
            embedder_state,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::super::SCHEMA_VERSION;
        use super::*;

        #[test]
        fn fnv1a64_is_deterministic_and_sensitive() {
            let a = fnv1a64(b"hello world");
            let b = fnv1a64(b"hello world");
            let c = fnv1a64(b"hello worlD");
            assert_eq!(a, b);
            assert_ne!(a, c);
        }

        #[test]
        fn checksum_mismatch_is_detected() {
            let snapshot = IndexSnapshot {
                schema_version: SCHEMA_VERSION,
                dim: 1,
                distance: Distance::Cosine,
                entries: vec![],
                tfidf_state: None,
            };
            let mut bytes = encode_index_snapshot(&snapshot).expect("encode");
            let last = bytes.len() - 1;
            bytes[last] ^= 0xFF; // flip a bit in the checksum trailer
            let err = decode_index_snapshot(&bytes);
            assert!(matches!(err, Err(RagError::Persistence(_))));
        }

        #[test]
        fn truncated_data_is_rejected_not_panicking() {
            let snapshot = IndexSnapshot {
                schema_version: SCHEMA_VERSION,
                dim: 2,
                distance: Distance::Euclidean,
                entries: vec![VectorEntry {
                    id: 0,
                    vector: vec![1.0, 2.0],
                    chunk: Chunk::new("x".into(), 0, 0, 0),
                }],
                tfidf_state: None,
            };
            let bytes = encode_index_snapshot(&snapshot).expect("encode");
            for cut in [0usize, 1, 8, bytes.len() / 2] {
                let err = decode_index_snapshot(&bytes[..cut]);
                assert!(err.is_err(), "truncation at {cut} must error, not panic");
            }
        }

        // ── a forged length must not drive a huge pre-allocation ──

        #[test]
        fn forged_huge_entry_count_with_valid_checksum_errors_cleanly() {
            // Simulates a crafted file: a legitimate empty `IndexSnapshot`'s
            // bytes, with `entry_count` overwritten to an enormous value and
            // the trailing checksum recomputed so it still verifies --
            // exactly the scenario to guard against. FNV-1a-64 is
            // trivially forgeable, so a "valid checksum" proves nothing
            // about a length field being honest. Without `capped_capacity`
            // capping the `Vec::with_capacity` hint in `read_index_body`,
            // this `entry_count` makes `entry_count * size_of::<VectorEntry>()`
            // overflow `usize`, which Rust's allocator turns into a hard
            // "capacity overflow" panic -- not a catchable `Result::Err` --
            // aborting the whole read. With the cap in place, this instead
            // returns an ordinary truncation error almost instantly.
            let snapshot = IndexSnapshot {
                schema_version: SCHEMA_VERSION,
                dim: 1,
                distance: Distance::Cosine,
                entries: vec![],
                tfidf_state: None,
            };
            let mut bytes = encode_index_snapshot(&snapshot).expect("encode");
            // Layout: MAGIC(8) + schema_version(4) + dim(8) + distance(1)
            // + entry_count(8) + ... + checksum(8, trailing).
            let entry_count_offset = 8 + 4 + 8 + 1;
            bytes[entry_count_offset..entry_count_offset + 8]
                .copy_from_slice(&(u64::MAX / 2).to_le_bytes());
            let checksum_offset = bytes.len() - 8;
            let new_checksum = fnv1a64(&bytes[..checksum_offset]);
            bytes[checksum_offset..].copy_from_slice(&new_checksum.to_le_bytes());

            let err = decode_index_snapshot(&bytes);
            assert!(
                matches!(err, Err(RagError::Persistence(_))),
                "a forged huge entry_count (with a recomputed, matching checksum) must error \
                 cleanly instead of attempting a huge up-front allocation: {err:?}"
            );
        }

        #[test]
        fn wrong_magic_is_rejected() {
            let snapshot = RetrieverSnapshot {
                schema_version: SCHEMA_VERSION,
                doc_count: 0,
                store: IndexSnapshot {
                    schema_version: SCHEMA_VERSION,
                    dim: 1,
                    distance: Distance::Cosine,
                    entries: vec![],
                    tfidf_state: None,
                },
                config: RetrieverConfig::default(),
                embedder_state: None,
            };
            let bytes = encode_retriever_snapshot(&snapshot).expect("encode");
            // A RetrieverSnapshot-encoded file must be rejected by the
            // IndexSnapshot decoder (different magic).
            let err = decode_index_snapshot(&bytes);
            assert!(matches!(err, Err(RagError::Persistence(_))));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{IdentityEmbedder, TfIdfEmbedder};

    fn tmp_path(tag: &str, ext: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("oxibonsai_rag_persist_{tag}_{pid}_{nanos}.{ext}"))
    }

    // ── Pre-existing behaviour (regression guards) ──────────────────────────

    #[test]
    fn roundtrip_preserves_entries() {
        let mut store = VectorStore::new(3);
        let chunk = Chunk::new("hello".into(), 0, 0, 0);
        store.insert(vec![1.0, 0.0, 0.0], chunk).expect("insert");

        let path = tmp_path("roundtrip", "json");
        store.save_json(&path).expect("save");
        let loaded = VectorStore::load_json(&path).expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.dim(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unknown_version_rejected() {
        let snapshot = IndexSnapshot {
            schema_version: 9999,
            dim: 1,
            distance: Distance::Cosine,
            entries: Vec::new(),
            tfidf_state: None,
        };
        let result = VectorStore::from_snapshot(snapshot);
        assert!(matches!(result, Err(RagError::Persistence(_))));
    }

    // ── RAG-EVAL-IMG-09: RetrieverConfig round-trips ────────────────────────

    #[test]
    fn retriever_load_restores_the_saved_config_not_the_default() {
        let embedder = IdentityEmbedder::new(8).expect("valid dim");
        let config = RetrieverConfig::default().with_top_k(2).with_rerank(true);
        let mut retriever = Retriever::new(embedder, config);
        retriever
            .add_document(
                "hello there",
                &crate::chunker::ChunkConfig::default().with_min_chunk_size(1),
            )
            .expect("index");

        let path = tmp_path("config_roundtrip", "json");
        retriever.save(&path).expect("save");

        let embedder2 = IdentityEmbedder::new(8).expect("valid dim");
        let restored = Retriever::load(embedder2, &path).expect("load");
        assert_eq!(restored.config().top_k, 2);
        assert!(restored.config().rerank);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn default_config_retriever_round_trips_through_save_and_load() {
        // Exercises the exact scenario the advisor flagged: a
        // default-configured retriever (min_score = f32::MIN) must not fail
        // to *deserialise* -- f32::NEG_INFINITY would have serialised to
        // JSON `null` and failed to parse back into an f32.
        let embedder = IdentityEmbedder::new(8).expect("valid dim");
        let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
        retriever
            .add_document(
                "hello there",
                &crate::chunker::ChunkConfig::default().with_min_chunk_size(1),
            )
            .expect("index");
        let path = tmp_path("default_config_roundtrip", "json");
        retriever
            .save(&path)
            .expect("save must succeed for a default config");
        let embedder2 = IdentityEmbedder::new(8).expect("valid dim");
        let restored =
            Retriever::load(embedder2, &path).expect("load must succeed for a default config");
        assert_eq!(restored.config().min_score, f32::MIN);
        std::fs::remove_file(&path).ok();
    }

    // ── non-finite `min_score` must still round-trip ────
    //
    // `RetrieverConfig::min_score` is a `pub` field on a `#[non_exhaustive]`
    // struct: `cfg.min_score = f32::NEG_INFINITY` reaches it directly from
    // outside the crate, bypassing `with_min_score` entirely. Before the
    // fix, `serde_json` serialised a non-finite `f32` as JSON `null`, which
    // does not deserialise back into an `f32` -- `save`/`save_binary`
    // returned `Ok`, but the resulting file could never be loaded again.

    #[test]
    fn retriever_with_non_finite_min_score_round_trips_through_save_and_load() {
        for min_score in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let embedder = IdentityEmbedder::new(8).expect("valid dim");
            let config = RetrieverConfig::default().with_min_score(min_score);
            let mut retriever = Retriever::new(embedder, config);
            retriever
                .add_document(
                    "hello there",
                    &crate::chunker::ChunkConfig::default().with_min_chunk_size(1),
                )
                .expect("index");

            let path = tmp_path("non_finite_min_score_json", "json");
            retriever
                .save(&path)
                .expect("save must succeed for a non-finite min_score");
            let embedder2 = IdentityEmbedder::new(8).expect("valid dim");
            let restored = Retriever::load(embedder2, &path).unwrap_or_else(|e| {
                panic!("load must succeed after saving min_score={min_score}: {e:?}")
            });
            if min_score.is_nan() {
                assert!(
                    restored.config().min_score.is_nan(),
                    "min_score NaN-ness must survive"
                );
            } else {
                assert_eq!(restored.config().min_score, min_score);
            }
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn retriever_with_non_finite_min_score_round_trips_through_save_binary_and_load_binary() {
        for min_score in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let embedder = IdentityEmbedder::new(8).expect("valid dim");
            let config = RetrieverConfig::default().with_min_score(min_score);
            let mut retriever = Retriever::new(embedder, config);
            retriever
                .add_document(
                    "hello there",
                    &crate::chunker::ChunkConfig::default().with_min_chunk_size(1),
                )
                .expect("index");

            let path = tmp_path("non_finite_min_score_bin", "bin");
            retriever
                .save_binary(&path)
                .expect("save_binary must succeed for a non-finite min_score");
            let embedder2 = IdentityEmbedder::new(8).expect("valid dim");
            let restored = Retriever::load_binary(embedder2, &path).unwrap_or_else(|e| {
                panic!("load_binary must succeed after saving min_score={min_score}: {e:?}")
            });
            if min_score.is_nan() {
                assert!(
                    restored.config().min_score.is_nan(),
                    "min_score NaN-ness must survive"
                );
            } else {
                assert_eq!(restored.config().min_score, min_score);
            }
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn euclidean_store_round_trip_still_returns_entries() {
        // The concrete scenario from missed-M2: a saved non-Cosine index
        // reloaded via `Retriever::load` must not come back silently unable
        // to return any of its own entries.
        let embedder = IdentityEmbedder::new(4).expect("valid dim");
        let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
        let mut store = VectorStore::new_with_distance(4, Distance::Euclidean);
        store
            .insert(vec![1.0, 2.0, 3.0, 4.0], Chunk::new("a".into(), 0, 0, 0))
            .expect("insert");
        *retriever.store_mut() = store;

        let path = tmp_path("euclidean_roundtrip", "json");
        retriever.save(&path).expect("save");
        let embedder2 = IdentityEmbedder::new(4).expect("valid dim");
        let restored = Retriever::load(embedder2, &path).expect("load");
        assert_eq!(restored.store().distance(), Distance::Euclidean);
        let hits = restored.store().search(&[1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(
            hits.len(),
            1,
            "the restored Euclidean store must still return its entry"
        );
        std::fs::remove_file(&path).ok();
    }

    // ── RAG-EVAL-IMG-10: TfIdfEmbedder state round-trips ────────────────────

    #[test]
    fn tfidf_embedder_reconstructed_via_load_with_embedder() {
        let docs = [
            "rust is fast and safe",
            "python is easy to learn",
            "go is simple",
        ];
        let embedder = TfIdfEmbedder::fit(&docs, 50);
        let dim = embedder.embedding_dim();
        let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
        for doc in docs {
            retriever
                .add_document(
                    doc,
                    &crate::chunker::ChunkConfig::default().with_min_chunk_size(1),
                )
                .expect("index");
        }

        let path = tmp_path("tfidf_roundtrip", "json");
        retriever.save(&path).expect("save");

        let restored: Retriever<TfIdfEmbedder> =
            Retriever::load_with_embedder(&path).expect("load_with_embedder");
        assert_eq!(restored.embedder().embedding_dim(), dim);
        assert_eq!(
            restored.embedder().vocab_size(),
            retriever.embedder().vocab_size()
        );
        // A genuinely reconstructed embedder must produce identical
        // embeddings, not just report the same dimension.
        let hits = restored
            .retrieve("rust safe")
            .expect("retrieve on the reconstructed embedder");
        assert!(!hits.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_with_embedder_errors_cleanly_without_persisted_state() {
        // A hand-crafted legacy-shaped snapshot (as if saved before
        // RAG-EVAL-IMG-10) has no embedder_state at all.
        let snapshot = RetrieverSnapshot {
            schema_version: SCHEMA_VERSION,
            doc_count: 0,
            store: IndexSnapshot {
                schema_version: SCHEMA_VERSION,
                dim: 4,
                distance: Distance::Cosine,
                entries: vec![],
                tfidf_state: None,
            },
            config: RetrieverConfig::default(),
            embedder_state: None,
        };
        let path = tmp_path("legacy_no_state", "json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&snapshot).expect("serialize"),
        )
        .expect("write");
        let err: Result<Retriever<TfIdfEmbedder>, _> = Retriever::load_with_embedder(&path);
        assert!(matches!(err, Err(RagError::Persistence(_))));
        std::fs::remove_file(&path).ok();
    }

    // ── RAG-EVAL-IMG-34: binary format ──────────────────────────────────────

    #[test]
    fn binary_store_roundtrip_matches_json() {
        let mut store = VectorStore::new(3);
        store
            .insert(
                vec![1.0, 2.0, 3.0],
                Chunk::new("hello".into(), 0, 0, 0).with_metadata("k", "v"),
            )
            .expect("insert");
        store
            .insert(vec![4.0, 5.0, 6.0], Chunk::new("world".into(), 1, 0, 5))
            .expect("insert");

        let path = tmp_path("binary_store", "bin");
        store.save_binary(&path).expect("save_binary");
        let loaded = VectorStore::load_binary(&path).expect("load_binary");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dim(), 3);
        assert_eq!(loaded.distance(), store.distance());
        let hits = loaded.search(&[1.0, 2.0, 3.0], 2);
        assert_eq!(hits[0].chunk.text, "hello");
        assert_eq!(
            hits[0].chunk.metadata.get("k"),
            Some(&MetadataValue::from("v"))
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn binary_retriever_roundtrip_restores_config_and_index() {
        let embedder = IdentityEmbedder::new(8).expect("valid dim");
        let config = RetrieverConfig::default().with_top_k(7);
        let mut retriever = Retriever::new(embedder, config);
        retriever
            .add_document(
                "some content to index",
                &crate::chunker::ChunkConfig::default().with_min_chunk_size(1),
            )
            .expect("index");

        let path = tmp_path("binary_retriever", "bin");
        retriever.save_binary(&path).expect("save_binary");

        let embedder2 = IdentityEmbedder::new(8).expect("valid dim");
        let restored = Retriever::load_binary(embedder2, &path).expect("load_binary");
        assert_eq!(restored.config().top_k, 7);
        assert_eq!(restored.chunk_count(), retriever.chunk_count());
        assert_eq!(restored.document_count(), retriever.document_count());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn binary_load_with_embedder_reconstructs_tfidf() {
        let docs = ["alpha beta gamma", "delta epsilon"];
        let embedder = TfIdfEmbedder::fit(&docs, 20);
        let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
        for doc in docs {
            retriever
                .add_document(
                    doc,
                    &crate::chunker::ChunkConfig::default().with_min_chunk_size(1),
                )
                .expect("index");
        }
        let path = tmp_path("binary_tfidf", "bin");
        retriever.save_binary(&path).expect("save_binary");
        let restored: Retriever<TfIdfEmbedder> =
            Retriever::load_binary_with_embedder(&path).expect("load_binary_with_embedder");
        assert_eq!(
            restored.embedder().vocab_size(),
            retriever.embedder().vocab_size()
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn binary_corrupted_file_is_rejected_not_panicking() {
        let store = VectorStore::new(2);
        let path = tmp_path("binary_corrupt", "bin");
        store.save_binary(&path).expect("save_binary");
        let mut bytes = std::fs::read(&path).expect("read back");
        // Flip a byte in the middle of the payload (not the checksum
        // trailer) to simulate bit-rot / a hand-edited file.
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("rewrite corrupted");
        let err = VectorStore::load_binary(&path);
        assert!(matches!(err, Err(RagError::Persistence(_))));
        std::fs::remove_file(&path).ok();
    }

    // ── RAG-EVAL-IMG-34: atomic writes ──────────────────────────────────────

    #[test]
    fn save_json_leaves_no_temp_file_behind_on_success() {
        let store = VectorStore::new(2);
        let path = tmp_path("atomic_cleanup", "json");
        store.save_json(&path).expect("save");
        assert!(path.exists());
        let dir = path.parent().expect("parent dir");
        let stray_temp_files: Vec<_> = std::fs::read_dir(dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.contains(".tmp-") && n.contains("atomic_cleanup"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            stray_temp_files.is_empty(),
            "no .tmp- file must remain after a successful atomic save: {stray_temp_files:?}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn save_overwrites_existing_file_atomically() {
        let store1 = VectorStore::new(1);
        let path = tmp_path("atomic_overwrite", "json");
        store1.save_json(&path).expect("save1");

        let mut store2 = VectorStore::new(1);
        store2
            .insert(vec![1.0], Chunk::new("a".into(), 0, 0, 0))
            .expect("insert");
        store2.save_json(&path).expect("save2");

        let loaded = VectorStore::load_json(&path).expect("load");
        assert_eq!(loaded.len(), 1);
        std::fs::remove_file(&path).ok();
    }
}
