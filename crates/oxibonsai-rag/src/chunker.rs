//! Document chunking strategies for the RAG pipeline.
//!
//! This module provides three plain chunking functions:
//!
//! 1. [`chunk_document`] — fixed-size character windows with configurable overlap.
//! 2. [`chunk_by_sentences`] — group consecutive sentences up to a maximum count.
//! 3. [`chunk_by_paragraphs`] — split on blank lines (paragraph boundaries).
//!
//! It also defines [`Chunker`], a trait-object-safe seam that lets
//! [`crate::retriever::RetrieverBuilder`] and [`crate::pipeline::RagPipeline`]
//! plug in *any* of the crate's chunking strategies — the three functions
//! above, or the `RichChunk`-based family in [`crate::advanced_chunker`], or
//! [`crate::code_chunker::CodeChunker`] — rather than being hard-wired to
//! [`chunk_document`] (RAG-EVAL-IMG-11 / RAG-EVAL-IMG-12).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::advanced_chunker::{ChunkStrategy, RichChunk};
use crate::code_chunker::CodeChunker;
use crate::error::RagError;
use crate::metadata_filter::MetadataValue;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the sliding-window character chunker.
///
/// Marked `#[non_exhaustive]` so new knobs can be added in patch releases
/// without breaking downstream code.  External crates must construct
/// instances via [`ChunkConfig::default`] plus the `with_*` builders
/// rather than a struct literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChunkConfig {
    /// Maximum number of characters per chunk (default 512).
    pub chunk_size: usize,
    /// Number of characters that consecutive chunks share (default 64).
    pub overlap: usize,
    /// Chunks shorter than this are discarded (default 32).
    pub min_chunk_size: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
            min_chunk_size: 32,
        }
    }
}

impl ChunkConfig {
    /// Validate the configuration, returning an error message if it is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_size == 0 {
            return Err("chunk_size must be > 0".into());
        }
        if self.overlap >= self.chunk_size {
            return Err(format!(
                "overlap ({}) must be < chunk_size ({})",
                self.overlap, self.chunk_size
            ));
        }
        if self.min_chunk_size > self.chunk_size {
            // `chunk_document` never produces a window longer than
            // `chunk_size`, so if `min_chunk_size` exceeds it, every window
            // is discarded by the min-length filter and every document
            // silently yields zero chunks forever.
            return Err(format!(
                "min_chunk_size ({}) must be <= chunk_size ({})",
                self.min_chunk_size, self.chunk_size
            ));
        }
        Ok(())
    }

    /// Set [`ChunkConfig::chunk_size`] (builder).
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set [`ChunkConfig::overlap`] (builder).
    #[must_use]
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set [`ChunkConfig::min_chunk_size`] (builder).
    #[must_use]
    pub fn with_min_chunk_size(mut self, min_chunk_size: usize) -> Self {
        self.min_chunk_size = min_chunk_size;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunk
// ─────────────────────────────────────────────────────────────────────────────

/// A contiguous slice of a document, produced by one of the chunking functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// The text content of this chunk.
    pub text: String,
    /// Zero-based index of the source document in the corpus.
    pub doc_id: usize,
    /// Zero-based index of this chunk within its document.
    pub chunk_idx: usize,
    /// Byte offset of the first character of this chunk in the original
    /// document (despite the field's name, every chunker in this crate that
    /// can recover it uses a *byte* offset, since that is what every
    /// `text[chunk.char_offset..]` deep-link/citation call site in this
    /// codebase actually needs; see [`chunk_document`] and
    /// [`chunk_by_sentences`]).
    pub char_offset: usize,
    /// Arbitrary key/value metadata attached to this chunk.  Consumed by
    /// [`crate::metadata_filter::MetadataFilter`] for targeted retrieval.
    ///
    /// Defaults to an empty map when deserialising older snapshots.
    #[serde(default)]
    pub metadata: HashMap<String, MetadataValue>,
}

impl Chunk {
    /// Construct a chunk with no metadata.
    pub fn new(text: String, doc_id: usize, chunk_idx: usize, char_offset: usize) -> Self {
        Self {
            text,
            doc_id,
            chunk_idx,
            char_offset,
            metadata: HashMap::new(),
        }
    }

    /// Insert a metadata key/value pair (builder).
    #[must_use]
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build a [`Chunk`] from a [`RichChunk`] (as produced by any
    /// [`crate::advanced_chunker::ChunkStrategy`]: `MarkdownChunker`,
    /// `RecursiveCharSplitter`, `SentenceChunker`, `SlidingWindowChunker`),
    /// attaching it to `doc_id` (RAG-EVAL-IMG-12).
    ///
    /// # Offset semantics (RAG-EVAL-IMG-12 verifier follow-up)
    ///
    /// `RichChunk::char_start`/`char_end` are *character* offsets (see
    /// `advanced_chunker.rs`'s own `RichChunk::len`, which counts
    /// `self.text.chars().count()`), whereas [`Chunk::char_offset`] is
    /// documented and used everywhere else in this crate (see
    /// [`chunk_document`], [`chunk_by_sentences`], [`chunk_by_paragraphs`])
    /// as a *byte* offset into the source text — the unit every
    /// `doc[chunk.char_offset..].starts_with(&chunk.text)` deep-link/citation
    /// call site actually needs. A prior version of this function copied
    /// `char_start` into `char_offset` verbatim, which only happened to be
    /// correct for ASCII documents (where 1 char == 1 byte); on a CJK
    /// document a citation offset landed roughly a third of the way into the
    /// wrong text. `index` — built once per document by the caller (e.g.
    /// [`StrategyChunker::chunk`], which already holds `text`) — converts
    /// `char_start` to the true byte offset in O(1), so the O(n) table build
    /// happens once per document rather than once per chunk (an O(n·k)
    /// re-scan would reintroduce the exact class of blow-up
    /// `chunk_document` was already fixed for; see RAG-EVAL-IMG-14).
    pub fn from_rich(rich: RichChunk, doc_id: usize, index: &CharByteIndex) -> Self {
        let RichChunk {
            text,
            char_start,
            chunk_index,
            metadata,
            ..
        } = rich;
        Self {
            text,
            doc_id,
            chunk_idx: chunk_index,
            char_offset: index.byte_offset(char_start),
            metadata: convert_rich_metadata(metadata),
        }
    }
}

/// Convert a [`RichChunk`]'s string-only metadata map into a [`Chunk`]'s
/// [`MetadataValue`]-typed one.
///
/// Shared by [`Chunk::from_rich`] and the [`From<RichChunk>`] impl below,
/// which otherwise duplicated this loop verbatim: the two conversions differ
/// only in how they compute `char_offset` (one has a [`CharByteIndex`] to
/// recover a true byte offset, the other does not — see the
/// [`From<RichChunk>`] impl's own docs), not in how they handle metadata.
fn convert_rich_metadata(metadata: HashMap<String, String>) -> HashMap<String, MetadataValue> {
    let mut converted = HashMap::with_capacity(metadata.len());
    for (key, value) in metadata {
        converted.insert(key, MetadataValue::from(value));
    }
    converted
}

/// A precomputed character-index → byte-offset lookup table for one source
/// document.
///
/// [`Chunk::from_rich`] needs this conversion because
/// [`crate::advanced_chunker::RichChunk`] reports positions in *characters*
/// while [`Chunk::char_offset`] is a *byte* offset. Recomputing it with a
/// fresh `source.char_indices().nth(char_idx)` scan inside `from_rich` for
/// every chunk would cost O(n) per chunk — O(n·k) for a k-chunk document,
/// the same quadratic-in-spirit trap `chunk_document` was fixed for
/// (RAG-EVAL-IMG-14). Building this table once per document and indexing it
/// per chunk is O(n + k).
#[derive(Debug, Clone)]
pub struct CharByteIndex {
    /// `offsets[i]` is the byte offset of the source's `i`-th character.
    /// Carries one extra trailing sentinel equal to the source's total byte
    /// length, so [`CharByteIndex::byte_offset`] never needs a branch for an
    /// index at (or past) the end of the document.
    offsets: Vec<usize>,
}

impl CharByteIndex {
    /// Build the table for `source`. O(n) in `source`'s length; call this
    /// once per document, not once per chunk.
    pub fn new(source: &str) -> Self {
        let mut offsets: Vec<usize> = source.char_indices().map(|(byte, _)| byte).collect();
        offsets.push(source.len());
        Self { offsets }
    }

    /// The byte offset of the `char_idx`-th character of the source this
    /// table was built from, in O(1).
    ///
    /// `char_idx` at or past the end of the document (including
    /// `source.chars().count()` itself, the exact value a `RichChunk`
    /// spanning to the end of the document reports) resolves to the
    /// source's total byte length rather than panicking or silently
    /// wrapping.
    pub fn byte_offset(&self, char_idx: usize) -> usize {
        // `offsets` always has >= 1 entry: even an empty source produces a
        // single sentinel (`"".char_indices()` yields nothing, then the
        // `push(0)` above adds the one entry), so this index is safe without
        // a fallible `.get(..)`.
        let last = self.offsets.len() - 1;
        self.offsets[char_idx.min(last)]
    }
}

impl From<RichChunk> for Chunk {
    /// Single-document convenience conversion: `doc_id` is always `0`, since
    /// [`RichChunk`] carries no document identity of its own.
    ///
    /// Unlike [`Chunk::from_rich`], this conversion has no access to the
    /// source document (a bare `RichChunk` does not carry one), so it
    /// **cannot** build a [`CharByteIndex`] and recover a true byte offset:
    /// `char_start` is carried over verbatim into [`Chunk::char_offset`],
    /// which is only byte-accurate for ASCII text. Multi-document callers,
    /// and anyone who needs a correct offset on non-ASCII text, should use
    /// [`Chunk::from_rich`] directly (see [`StrategyChunker::chunk`] for the
    /// pattern), or index through
    /// [`crate::retriever::RetrieverBuilder::with_chunker`], which threads
    /// both the real `doc_id` and a real [`CharByteIndex`] through
    /// automatically.
    fn from(rich: RichChunk) -> Self {
        let RichChunk {
            text,
            char_start,
            chunk_index,
            metadata,
            ..
        } = rich;
        Chunk {
            text,
            doc_id: 0,
            chunk_idx: chunk_index,
            char_offset: char_start,
            metadata: convert_rich_metadata(metadata),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunker trait — the pluggable-strategy seam
// ─────────────────────────────────────────────────────────────────────────────

/// Pluggable document-chunking strategy, usable via
/// [`crate::retriever::RetrieverBuilder::with_chunker`].
///
/// This is the trait-object-safe seam that lets [`crate::retriever::Retriever`]
/// and [`crate::pipeline::RagPipeline`] accept *any* of the crate's chunking
/// strategies instead of being hard-wired to [`chunk_document`]
/// (RAG-EVAL-IMG-11 / RAG-EVAL-IMG-12). Implementations must be `Send + Sync`
/// so a boxed chunker can be stored on a `Retriever` shared across threads.
pub trait Chunker: Send + Sync {
    /// Split `text` (the `doc_id`-th document) into chunks.
    fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError>;
}

/// [`Chunker`] adapter for the fixed-size sliding-window strategy
/// ([`chunk_document`]) — equivalent to the strategy `Retriever` uses when no
/// custom [`Chunker`] is configured at all.
#[derive(Debug, Clone, Default)]
pub struct FixedWindowChunker(pub ChunkConfig);

impl Chunker for FixedWindowChunker {
    fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError> {
        self.0.validate().map_err(RagError::InvalidChunkConfig)?;
        Ok(chunk_document(text, doc_id, &self.0))
    }
}

/// [`Chunker`] adapter for [`chunk_by_sentences`]; the wrapped `usize` is
/// `max_sentences`.
#[derive(Debug, Clone, Copy)]
pub struct SentenceGroupChunker(pub usize);

impl Chunker for SentenceGroupChunker {
    fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError> {
        Ok(chunk_by_sentences(text, doc_id, self.0))
    }
}

/// [`Chunker`] adapter for [`chunk_by_paragraphs`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ParagraphChunker;

impl Chunker for ParagraphChunker {
    fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError> {
        Ok(chunk_by_paragraphs(text, doc_id))
    }
}

/// [`Chunker`] adapter for any [`ChunkStrategy`] (`MarkdownChunker`,
/// `RecursiveCharSplitter`, `advanced_chunker::SentenceChunker`,
/// `SlidingWindowChunker`) — this whole family previously had no path into
/// `Retriever`/`RagPipeline` at all (RAG-EVAL-IMG-12). Builds one
/// [`CharByteIndex`] for `text` and reuses it across every produced chunk, so
/// [`Chunk::from_rich`] reports a real byte offset (correct for non-ASCII
/// documents too) in O(n + k) rather than the O(n·k) an index-per-chunk scan
/// would cost.
pub struct StrategyChunker<S>(pub S);

impl<S: ChunkStrategy> Chunker for StrategyChunker<S> {
    fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError> {
        let index = CharByteIndex::new(text);
        Ok(self
            .0
            .chunk(text)
            .into_iter()
            .map(|rich| Chunk::from_rich(rich, doc_id, &index))
            .collect())
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError> {
        // Disambiguates to the inherent `CodeChunker::chunk` (inherent
        // methods always take priority over same-named trait methods for a
        // concrete receiver type), so this is not recursive.
        CodeChunker::chunk(self, text, doc_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-size sliding-window chunker
// ─────────────────────────────────────────────────────────────────────────────

/// Split `text` into overlapping fixed-size character windows.
///
/// The step between consecutive windows is `config.chunk_size - config.overlap`.
/// Chunks smaller than `config.min_chunk_size` characters are discarded —
/// including the whole document when it is itself shorter than
/// `min_chunk_size`; see [`crate::retriever::Retriever::add_document`] for
/// the indexing-level guarantee that a short *document* is never dropped
/// from the corpus even though this low-level window function still can
/// silently *skip* (never emit) a final window shorter than
/// `min_chunk_size` when the document length isn't an exact multiple of
/// the step size — the loop always terminates once a window reaches the
/// end of `text`, so no chunk shorter than `min_chunk_size` is ever pushed,
/// zero-length or otherwise.
///
/// Runs in O(n) time and O(n) extra space (one `usize` byte-offset per
/// character, no repeated re-scans): a prior version recomputed the byte
/// offset of every window's start from byte 0 on every iteration, which made
/// the whole function O(n²) (RAG-EVAL-IMG-14: 1 MiB took 1.39 s, 2 MiB took
/// 5.66 s — a 4.06× blow-up for a 2× input).
///
/// Returns an empty `Vec` if `text` is empty.
pub fn chunk_document(text: &str, doc_id: usize, config: &ChunkConfig) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }

    // Precompute the byte offset of every character exactly once. Slicing
    // `text[offsets[start]..offsets[end]]` below is then O(window length)
    // instead of the old `s.char_indices().nth(n)` rescan-from-0 per chunk.
    let offsets: Vec<usize> = text.char_indices().map(|(byte, _)| byte).collect();
    let total = offsets.len();

    if total < config.min_chunk_size {
        // The whole document is a single chunk only if it is not below min.
        return Vec::new();
    }

    let step = config.chunk_size.saturating_sub(config.overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < total {
        let end = (start + config.chunk_size).min(total);

        if end - start >= config.min_chunk_size {
            let byte_start = offsets[start];
            let byte_end = if end < total {
                offsets[end]
            } else {
                text.len()
            };
            let chunk_idx = chunks.len();
            chunks.push(Chunk::new(
                text[byte_start..byte_end].to_string(),
                doc_id,
                chunk_idx,
                byte_start,
            ));
        }

        if end == total {
            break;
        }
        start += step;
    }

    chunks
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentence-based chunker
// ─────────────────────────────────────────────────────────────────────────────

/// Split `text` into chunks of at most `max_sentences` consecutive sentences.
///
/// Sentence boundaries are detected by a lightweight heuristic: a sentence ends
/// at `.`, `!`, or `?` followed by optional whitespace.  This is intentionally
/// simple — production use would wire in a proper sentence-boundary detector.
///
/// Each chunk's text is the verbatim source span from the start of its first
/// sentence to the end of its last sentence (including whatever original
/// separators appeared between them), and `Chunk::char_offset` is that
/// span's real byte offset — so `doc[chunk.char_offset..]` always starts
/// with `chunk.text` by construction (RAG-EVAL-IMG-19: a prior version
/// rebuilt each chunk by `join(" ")`-ing trimmed sentences and tracked the
/// offset with a `+1`-per-sentence approximation, which desynchronised the
/// two the moment a real separator was not exactly one space).
///
/// Returns an empty `Vec` if `text` is empty.
pub fn chunk_by_sentences(text: &str, doc_id: usize, max_sentences: usize) -> Vec<Chunk> {
    if text.is_empty() || max_sentences == 0 {
        return Vec::new();
    }

    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut i = 0usize;

    while i < sentences.len() {
        let batch_end = (i + max_sentences).min(sentences.len());
        let batch = &sentences[i..batch_end];
        // SAFETY-free invariant: `batch` is non-empty because `i < batch_end`.
        let first_start = batch[0].0;
        let last_end = batch[batch.len() - 1].1;

        let chunk_idx = chunks.len();
        chunks.push(Chunk::new(
            text[first_start..last_end].to_string(),
            doc_id,
            chunk_idx,
            first_start,
        ));

        i = batch_end;
    }

    chunks
}

/// Lightweight sentence splitter based on terminal punctuation.
///
/// Returns each sentence's `(byte_start, byte_end)` span in `text`, with
/// leading/trailing whitespace trimmed from the span itself (rather than
/// returning an already-trimmed `&str`, so the caller can slice the
/// *original* text — including any interior separators — across a run of
/// consecutive sentences and get back an exact substring; see
/// [`chunk_by_sentences`]).
fn split_sentences(text: &str) -> Vec<(usize, usize)> {
    let mut sentences = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' || b == b'!' || b == b'?' {
            // Consume trailing punctuation (e.g. "..." or "?!")
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b'.' || bytes[j] == b'!' || bytes[j] == b'?') {
                j += 1;
            }
            // Skip whitespace after terminal punctuation
            while j < bytes.len()
                && (bytes[j] == b' ' || bytes[j] == b'\t' || bytes[j] == b'\n' || bytes[j] == b'\r')
            {
                j += 1;
            }
            if let Some(span) = trimmed_span(text, start, j) {
                sentences.push(span);
            }
            start = j;
            i = j;
        } else {
            i += 1;
        }
    }
    // Trailing text without terminal punctuation
    if let Some(span) = trimmed_span(text, start, text.len()) {
        sentences.push(span);
    }
    sentences
}

/// Return the byte span of `text[start..end]` with leading/trailing
/// (Unicode) whitespace trimmed, or `None` if the trimmed span is empty.
/// Uses only length differences (no pointer arithmetic) to recover the
/// trimmed span's absolute byte offsets.
fn trimmed_span(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let slice = &text[start..end];
    let after_start_trim = slice.trim_start();
    let leading_trimmed = slice.len() - after_start_trim.len();
    let trimmed = after_start_trim.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let span_start = start + leading_trimmed;
    let span_end = span_start + trimmed.len();
    Some((span_start, span_end))
}

// ─────────────────────────────────────────────────────────────────────────────
// Paragraph-based chunker
// ─────────────────────────────────────────────────────────────────────────────

/// Split `text` into chunks at paragraph boundaries (one or more blank lines).
///
/// Each non-empty paragraph (after whitespace-trimming) becomes a separate
/// chunk, and `Chunk::char_offset` is that trimmed span's real byte offset
/// -- so `doc[chunk.char_offset..]` always starts with `chunk.text` by
/// construction (RAG-CORE-blocking-1: a prior version tracked a
/// `byte_cursor` set to the *separator*'s own byte offset, so every chunk
/// after the first pointed one blank line too early; the error grew with
/// `\r\n` endings and with runs of more than one blank line).
/// Returns an empty `Vec` if `text` is empty.
pub fn chunk_by_paragraphs(text: &str, doc_id: usize) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }

    // Split on sequences of two or more newlines (blank line separator)
    let mut chunks = Vec::new();

    // We iterate over the text looking for blank lines manually so we can track
    // byte offsets accurately.
    let mut para_start = 0usize;
    let mut prev_line_empty = false;
    let mut line_start = 0usize;

    let text_bytes = text.as_bytes();
    let mut i = 0usize;

    while i <= text_bytes.len() {
        // At end-of-line (or end-of-string), check whether this line is blank
        let is_eot = i == text_bytes.len();
        let is_newline = !is_eot && (text_bytes[i] == b'\n');

        if is_newline || is_eot {
            let line = text[line_start..i].trim();
            let is_blank = line.is_empty();

            if is_blank && !prev_line_empty {
                // We have just hit the first blank line -- emit the
                // paragraph spanning [para_start, line_start), trimmed.
                // `trimmed_span` recovers the trimmed span's own absolute
                // byte offsets directly, instead of the old `byte_cursor`
                // (which pointed at this *separator*'s byte offset, not
                // the next paragraph's own start).
                if let Some((span_start, span_end)) = trimmed_span(text, para_start, line_start) {
                    let chunk_idx = chunks.len();
                    chunks.push(Chunk::new(
                        text[span_start..span_end].to_string(),
                        doc_id,
                        chunk_idx,
                        span_start,
                    ));
                }
                para_start = i + 1;
            } else if !is_blank {
                // Non-blank line resets the blank-line run
                if prev_line_empty {
                    // First non-blank line after a blank: update para_start
                    para_start = line_start;
                }
            }

            prev_line_empty = is_blank;
            line_start = i + 1;
        }

        if is_eot {
            // Emit any remaining paragraph. `para_start` can already equal
            // `text.len() + 1` here: when the text ends in exactly one
            // `\n`, the EOT iteration's trailing empty "line" is itself
            // classified blank, so the blank-line branch above already
            // fired once (correctly emitting the final paragraph) and
            // advanced `para_start` past the end of the string. Clamp to
            // `text.len()` so that case slices the empty `text.len()
            // .. text.len()` range (which `trimmed_span` correctly turns
            // into `None`, emitting nothing further) instead of the
            // out-of-bounds `text.len() + 1 .. text.len()` that used to
            // panic. `para_start` is otherwise always `0` or a previously
            // assigned `line_start`, both `<= text.len()`, so the `.min()`
            // is a no-op in every other case and never truncates a real
            // paragraph.
            if let Some((span_start, span_end)) =
                trimmed_span(text, para_start.min(text.len()), text.len())
            {
                let chunk_idx = chunks.len();
                chunks.push(Chunk::new(
                    text[span_start..span_end].to_string(),
                    doc_id,
                    chunk_idx,
                    span_start,
                ));
            }
            break;
        }

        i += 1;
    }

    chunks
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced_chunker::SlidingWindowChunker;

    // ── RAG-EVAL-IMG-14: linear scaling ─────────────────────────────────────

    /// Reproduce-first: this is the exact shape of the finder's repro
    /// (default `ChunkConfig`, plain ASCII filler).  Asserts the
    /// *scaling* property `t(2n) < 3*t(n)` rather than an absolute bound, so
    /// it is robust to debug-build overhead and a busy build machine; an
    /// O(n²) implementation fails this regardless of profile or load
    /// (measured on this crate's old implementation: 1 MiB=1.39s,
    /// 2 MiB=5.66s => 4.06x, at opt-level 2). An absolute release-profile
    /// ceiling (4 MiB < 200ms) is asserted separately, only outside debug
    /// builds, where machine load cannot produce a false failure.
    #[test]
    fn chunk_document_scales_linearly_not_quadratically() {
        let config = ChunkConfig::default();
        let time_for_mib = |mib: usize| -> std::time::Duration {
            let text = "a".repeat(mib * 1024 * 1024);
            let start = std::time::Instant::now();
            let chunks = chunk_document(&text, 0, &config);
            assert!(!chunks.is_empty(), "sanity: must actually produce chunks");
            start.elapsed()
        };

        let t1 = time_for_mib(1);
        let t2 = time_for_mib(2);
        assert!(
            t2 < t1 * 3 + std::time::Duration::from_millis(50),
            "chunk_document does not scale linearly: t(1 MiB)={t1:?}, t(2 MiB)={t2:?}"
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn chunk_document_4mib_under_200ms_release() {
        let config = ChunkConfig::default();
        let text = "a".repeat(4 * 1024 * 1024);
        let start = std::time::Instant::now();
        let chunks = chunk_document(&text, 0, &config);
        let elapsed = start.elapsed();
        assert!(!chunks.is_empty());
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "4 MiB chunk_document took {elapsed:?}, expected < 200ms in a release build \
             (RAG-EVAL-IMG-14 gate)"
        );
    }

    // ── RAG-EVAL-IMG-19: char_offset correctness ────────────────────────────

    #[test]
    fn chunk_by_sentences_offsets_point_at_the_real_text() {
        // The finder's own reproduction: multi-space separators, which the
        // old `join(" ")` + `+1`-per-sentence approximation could not track.
        let text = "First sentence.  Second one!  Third here?  Tail without punctuation";
        let chunks = chunk_by_sentences(text, 0, 1);
        assert!(
            chunks.len() >= 4,
            "expected >= 4 chunks, got {}",
            chunks.len()
        );
        for chunk in &chunks {
            assert!(
                text[chunk.char_offset..].starts_with(&chunk.text),
                "char_offset {} does not point at chunk text {:?} \
                 (text there is {:?})",
                chunk.char_offset,
                chunk.text,
                &text[chunk.char_offset..(chunk.char_offset + chunk.text.len()).min(text.len())]
            );
        }
    }

    #[test]
    fn chunk_by_sentences_offsets_hold_for_multi_sentence_batches() {
        let text = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five.";
        let chunks = chunk_by_sentences(text, 0, 2);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(
                text[chunk.char_offset..].starts_with(&chunk.text),
                "char_offset {} does not point at chunk text {:?}",
                chunk.char_offset,
                chunk.text
            );
        }
    }

    // ── RAG-CORE-blocking-1: chunk_by_paragraphs char_offset correctness ────
    //
    // `chunk_by_paragraphs` used to track a `byte_cursor` set to the blank
    // line's own byte offset, so every chunk after the first pointed one
    // separator too early. Extends the same `doc[c.char_offset..
    // ].starts_with(&c.text)` property `chunk_by_sentences` already carries
    // above to `chunk_by_paragraphs`, across every corner the finder's
    // evidence called out.

    /// Assert the `char_offset` property for every chunk of `text` and
    /// return the chunks, so each scenario below can add its own further
    /// checks on top of the shared property.
    fn assert_paragraph_offsets_correct(text: &str) -> Vec<Chunk> {
        let chunks = chunk_by_paragraphs(text, 0);
        for chunk in &chunks {
            let Some(at_offset) = text.get(chunk.char_offset..) else {
                panic!(
                    "char_offset {} is not even a valid byte boundary in the source \
                     document, for chunk text {:?}",
                    chunk.char_offset, chunk.text
                );
            };
            // `.get()` (rather than direct indexing) for the diagnostic
            // preview: a *wrong* `char_offset` need not land on a UTF-8
            // char boundary `chunk.text.len()` bytes later, and the
            // preview must never itself panic while reporting the bug.
            let preview_end = (chunk.char_offset + chunk.text.len()).min(text.len());
            let preview = text
                .get(chunk.char_offset..preview_end)
                .unwrap_or(at_offset);
            assert!(
                at_offset.starts_with(&chunk.text),
                "char_offset {} does not point at chunk text {:?} (text there is {:?})",
                chunk.char_offset,
                chunk.text,
                preview
            );
        }
        chunks
    }

    #[test]
    fn chunk_by_paragraphs_offsets_correct_for_plain_lf_blank_lines() {
        // The finder's own reproduction: chunk 1 (0-based) used to get
        // `char_offset=18` where "Para two..." actually starts at 19.
        let text = "Para one is here.\n\nPara two is here.\n\nPara three is here.";
        let chunks = assert_paragraph_offsets_correct(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Para one is here.");
        assert_eq!(chunks[1].text, "Para two is here.");
        assert_eq!(chunks[2].text, "Para three is here.");
    }

    #[test]
    fn chunk_by_paragraphs_offsets_correct_for_crlf_blank_lines() {
        // `\r\n\r\n` blank-line separators: the finder notes the error
        // "grows with `\r\n` endings".
        let text = "Para one.\r\n\r\nPara two.\r\n\r\nPara three.";
        let chunks = assert_paragraph_offsets_correct(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Para one.");
        assert_eq!(chunks[1].text, "Para two.");
        assert_eq!(chunks[2].text, "Para three.");
    }

    #[test]
    fn chunk_by_paragraphs_offsets_correct_with_three_or_more_blank_lines() {
        let text = "Para one.\n\n\n\nPara two.\n\n\n\n\nPara three.";
        let chunks = assert_paragraph_offsets_correct(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Para one.");
        assert_eq!(chunks[1].text, "Para two.");
        assert_eq!(chunks[2].text, "Para three.");
    }

    #[test]
    fn chunk_by_paragraphs_offsets_correct_with_leading_and_trailing_blank_lines() {
        let text = "\n\n\nPara one.\n\nPara two.\n\n\n";
        let chunks = assert_paragraph_offsets_correct(text);
        assert_eq!(
            chunks.len(),
            2,
            "leading/trailing blank runs must not emit empty chunks"
        );
        assert_eq!(chunks[0].text, "Para one.");
        assert_eq!(chunks[1].text, "Para two.");
    }

    #[test]
    fn chunk_by_paragraphs_offsets_correct_with_single_trailing_newline() {
        // RAG-CORE-blocking-1 (verifier follow-up): a document whose last
        // line is non-blank and ends with exactly one `\n` used to panic
        // inside `trimmed_span` at end-of-text. At the EOT iteration the
        // trailing empty "line" is classified blank, the blank-line branch
        // fires (correctly emitting the final paragraph) and advances
        // `para_start` to `text.len() + 1`; the *separate* end-of-text
        // emission block then ran unconditionally afterwards and sliced
        // `text[text.len() + 1 .. text.len()]`, which panics
        // ("start byte index ... is out of bounds"). This is the single
        // most common real input shape (any file saved with a trailing
        // newline), so it must not panic and must not double-emit the
        // already-emitted final paragraph.
        let text = "Para one.\n";
        let chunks = assert_paragraph_offsets_correct(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Para one.");
    }

    #[test]
    fn chunk_by_paragraphs_offsets_correct_with_two_paragraphs_and_trailing_newline() {
        // Same defect, with a real blank-line separator present too, so the
        // end-of-text branch is reached with `para_start` already updated by
        // an *earlier* blank line as well as the EOT one.
        let text = "A\n\nB\n";
        let chunks = assert_paragraph_offsets_correct(text);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "A");
        assert_eq!(chunks[1].text, "B");
    }

    #[test]
    fn chunk_by_paragraphs_offsets_correct_for_cjk_document() {
        // Non-ASCII text: a byte-offset bug that happens to look right on
        // ASCII input (1 byte == 1 char) would still be wrong here.
        let text = "最初の段落です。\n\n二番目の段落です。\n\n三番目の段落です。";
        let chunks = assert_paragraph_offsets_correct(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "最初の段落です。");
        assert_eq!(chunks[1].text, "二番目の段落です。");
        assert_eq!(chunks[2].text, "三番目の段落です。");
    }

    #[test]
    fn paragraph_chunker_through_retriever_end_to_end_offsets_are_correct() {
        // Same property, exercised through the public
        // `RetrieverBuilder::with_chunker(ParagraphChunker)` path rather
        // than calling `chunk_by_paragraphs` directly, since that is the
        // route the crate's own doctest (retriever.rs) actually ships.
        use crate::embedding::IdentityEmbedder;
        use crate::retriever::RetrieverBuilder;

        let text = "Para one is here.\n\nPara two is here.\n\nPara three is here.";
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut retriever = RetrieverBuilder::new(emb)
            .with_chunker(Box::new(ParagraphChunker))
            .build();
        retriever
            .add_document(text, &ChunkConfig::default())
            .expect("index");
        assert_eq!(retriever.chunk_count(), 3);

        for entry in retriever.store().entries() {
            assert!(
                text[entry.chunk.char_offset..].starts_with(&entry.chunk.text),
                "char_offset {} does not point at chunk text {:?} through \
                 RetrieverBuilder::with_chunker(ParagraphChunker)",
                entry.chunk.char_offset,
                entry.chunk.text
            );
        }
    }

    // ── Chunk::from_rich / From<RichChunk> ──────────────────────────────────

    #[test]
    fn from_rich_carries_text_and_index() {
        let source = "hello world";
        let index = CharByteIndex::new(source);
        let rich = RichChunk::new("hello world".to_string(), 0, 11, 3);
        let chunk = Chunk::from_rich(rich, 7, &index);
        assert_eq!(chunk.text, "hello world");
        assert_eq!(chunk.doc_id, 7);
        assert_eq!(chunk.chunk_idx, 3);
        assert_eq!(chunk.char_offset, 0);
    }

    #[test]
    fn from_rich_folds_string_metadata_into_metadata_value() {
        let index = CharByteIndex::new("x");
        let mut rich = RichChunk::new("x".to_string(), 0, 1, 0);
        rich.metadata
            .insert("source".to_string(), "readme.md".to_string());
        let chunk = Chunk::from_rich(rich, 0, &index);
        assert_eq!(
            chunk.metadata.get("source"),
            Some(&MetadataValue::from("readme.md"))
        );
    }

    #[test]
    fn from_trait_defaults_doc_id_to_zero() {
        let rich = RichChunk::new("y".to_string(), 0, 1, 2);
        let chunk: Chunk = rich.into();
        assert_eq!(chunk.doc_id, 0);
        assert_eq!(chunk.chunk_idx, 2);
    }

    // ── CharByteIndex / Chunk::from_rich byte-vs-char offset (wave-1.5
    //    addendum: `RichChunk::char_start` is a character count, but
    //    `Chunk::char_offset` must be a byte offset) ───────────────────────

    #[test]
    fn char_byte_index_ascii_identity() {
        // For ASCII text, char index == byte index everywhere.
        let index = CharByteIndex::new("hello world");
        for i in 0..=11 {
            assert_eq!(index.byte_offset(i), i);
        }
    }

    #[test]
    fn char_byte_index_cjk_converts_char_count_to_byte_offset() {
        // Each of these CJK characters is 3 bytes in UTF-8.
        let source = "最初の段落です。二番目の段落です。";
        let index = CharByteIndex::new(source);
        // Character 5 is 'で' (0-based: 最,初,の,段,落,で,...), i.e. 5 * 3 = 15
        // bytes in. A prior version of `Chunk::from_rich` would have reported
        // `5` here (the character count itself, copied verbatim) instead of
        // `15`, landing roughly a third of the way into the wrong text.
        assert_eq!(index.byte_offset(5), 15);
        assert_eq!(
            &source[index.byte_offset(5)..index.byte_offset(5) + 3],
            "で"
        );
        assert_eq!(index.byte_offset(0), 0);
    }

    #[test]
    fn char_byte_index_out_of_range_resolves_to_source_len() {
        let source = "最初の段落です。"; // 8 chars, 24 bytes
        let index = CharByteIndex::new(source);
        assert_eq!(index.byte_offset(8), source.len());
        assert_eq!(index.byte_offset(1000), source.len());
    }

    #[test]
    fn char_byte_index_empty_source() {
        let index = CharByteIndex::new("");
        assert_eq!(index.byte_offset(0), 0);
        assert_eq!(index.byte_offset(5), 0);
    }

    #[test]
    fn from_rich_reports_a_true_byte_offset_on_cjk_text() {
        // Direct reproduction of the addendum's failure scenario: a
        // `RichChunk` whose `char_start` (5, a *character* count) is copied
        // verbatim into `Chunk::char_offset` (a *byte* offset) is wrong for
        // any document containing a multi-byte character before the chunk.
        let source = "最初の段落です。二番目の段落です。";
        let index = CharByteIndex::new(source);
        let rich = RichChunk::new("です。二番目".to_string(), 5, 11, 1);
        let chunk = Chunk::from_rich(rich, 0, &index);
        assert_eq!(
            chunk.char_offset, 15,
            "char_offset must be a byte offset (15), not the raw character \
             count (5)"
        );
        assert!(
            source[chunk.char_offset..].starts_with(&chunk.text),
            "char_offset {} does not point at chunk text {:?} in source {source:?}",
            chunk.char_offset,
            chunk.text
        );
    }

    /// Same property FIX-03's chunk-offset sweep asserts for the byte-offset
    /// chunkers (`chunk_by_sentences_offsets_point_at_the_real_text`,
    /// `chunk_by_paragraphs_offsets_correct_for_cjk_document`, above),
    /// exercised here for the `RichChunk`-derived family through
    /// `StrategyChunker`, across CJK, mixed-script and emoji (4-byte UTF-8)
    /// text and a range of window sizes.
    ///
    /// `SlidingWindowChunker` is used deliberately: its `chunk_text` is
    /// always an exact `chars[start..end]` slice of the source (never
    /// reconstructed via `join`), so this isolates the byte/char offset fix
    /// itself rather than any of `advanced_chunker.rs`'s own text-assembly
    /// behaviour (e.g. `SentenceChunker` rejoins sentences with a literal
    /// `" "`, which is a separate concern from the one this test — and the
    /// addendum — targets).
    #[test]
    fn strategy_chunker_offsets_satisfy_starts_with_property_across_scripts_and_windows() {
        let texts = [
            "最初の段落です。二番目の段落です。三番目の段落です。",
            "Mixed ASCII and 日本語 in one 文字列 with a few more 漢字 words.",
            "한국어 텍스트도 포함합니다 for good measure, testing Hangul too.",
            "🎉🎌🎊 repeated emoji only 🎉🎌🎊 repeated emoji only 🎉🎌🎊",
            "", // degenerate: must not panic and must produce zero chunks
        ];
        let windows = [1usize, 2, 3, 5, 7, 11, 16];
        let mut total_chunks_checked = 0usize;

        for text in texts {
            for &window in &windows {
                let chunker = StrategyChunker(SlidingWindowChunker::non_overlapping(window));
                let chunks = chunker.chunk(text, 0).expect("chunk must not error");
                for c in &chunks {
                    let Some(at_offset) = text.get(c.char_offset..) else {
                        panic!(
                            "char_offset {} is not a valid UTF-8 boundary in \
                             {text:?} (window={window}) for chunk text {:?}",
                            c.char_offset, c.text
                        );
                    };
                    assert!(
                        at_offset.starts_with(&c.text),
                        "char_offset {} does not point at chunk text {:?} \
                         (text there is {:?}) for source {text:?}, window={window}",
                        c.char_offset,
                        c.text,
                        &at_offset[..c.text.len().min(at_offset.len())]
                    );
                    total_chunks_checked += 1;
                }
            }
        }
        assert!(
            total_chunks_checked > 50,
            "sweep should have exercised well over 50 chunks, got {total_chunks_checked}"
        );
    }

    /// Same property, exercised through the public
    /// `RetrieverBuilder::with_chunker(StrategyChunker(..))` path — the real
    /// integration surface RAG-EVAL-IMG-12 added — rather than calling the
    /// `Chunker` trait directly, mirroring
    /// `paragraph_chunker_through_retriever_end_to_end_offsets_are_correct`
    /// above for the `RichChunk` family.
    #[test]
    fn strategy_chunker_through_retriever_end_to_end_offsets_are_correct_on_cjk() {
        use crate::embedding::IdentityEmbedder;
        use crate::retriever::RetrieverBuilder;

        let text = "最初の段落です。二番目の段落です。三番目の段落です。";
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut retriever = RetrieverBuilder::new(emb)
            .with_chunker(Box::new(StrategyChunker(SlidingWindowChunker::new(6, 6))))
            .build();
        retriever
            .add_document(text, &ChunkConfig::default())
            .expect("index");
        assert!(retriever.chunk_count() >= 2, "expected multiple CJK chunks");

        for entry in retriever.store().entries() {
            assert!(
                text[entry.chunk.char_offset..].starts_with(&entry.chunk.text),
                "char_offset {} does not point at chunk text {:?} through \
                 RetrieverBuilder::with_chunker(StrategyChunker(..)) on CJK text",
                entry.chunk.char_offset,
                entry.chunk.text
            );
        }
    }

    // ── Chunker adapters ─────────────────────────────────────────────────────

    #[test]
    fn fixed_window_chunker_matches_chunk_document() {
        let text = "a".repeat(600);
        let chunker = FixedWindowChunker::default();
        let via_trait = chunker.chunk(&text, 0).expect("chunk");
        let via_fn = chunk_document(&text, 0, &ChunkConfig::default());
        assert_eq!(via_trait.len(), via_fn.len());
        assert_eq!(via_trait[0].text, via_fn[0].text);
    }

    #[test]
    fn sentence_group_chunker_matches_chunk_by_sentences() {
        let text = "One. Two. Three.";
        let chunker = SentenceGroupChunker(1);
        let via_trait = chunker.chunk(text, 0).expect("chunk");
        assert_eq!(via_trait.len(), 3);
    }

    #[test]
    fn paragraph_chunker_matches_chunk_by_paragraphs() {
        let text = "Para one.\n\nPara two.";
        let chunker = ParagraphChunker;
        let via_trait = chunker.chunk(text, 0).expect("chunk");
        assert_eq!(via_trait.len(), 2);
    }

    #[test]
    fn strategy_chunker_wraps_chunk_strategy_family() {
        let chunker = StrategyChunker(SlidingWindowChunker::non_overlapping(5));
        let chunks = chunker.chunk("abcdefghij", 3).expect("chunk");
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|c| c.doc_id == 3));
    }

    #[test]
    fn code_chunker_implements_chunker_trait() {
        use crate::code_chunker::Language;
        let chunker = CodeChunker::new(Language::Plain).with_min_chunk_chars(1);
        let chunks = Chunker::chunk(&chunker, "hello world", 1).expect("chunk");
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].doc_id, 1);
    }

    // ── Pre-existing behaviour (regression guards) ──────────────────────────

    #[test]
    fn chunk_document_basic_unchanged() {
        let text = "a".repeat(600);
        let config = ChunkConfig::default();
        let chunks = chunk_document(&text, 0, &config);
        assert!(!chunks.is_empty());
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.doc_id, 0);
            assert_eq!(chunk.chunk_idx, i);
            assert!(text[chunk.char_offset..].starts_with(&chunk.text));
        }
    }

    #[test]
    fn chunk_document_short_text_still_discarded_at_window_level() {
        // `chunk_document` itself keeps its documented per-window contract;
        // the "always index at least one chunk per document" guarantee for
        // RAG-EVAL-IMG-13 lives one layer up, in
        // `Retriever::produce_chunks`, so this free function's behaviour for
        // a lone caller (and the crate's own long-standing unit test of it)
        // is unchanged.
        let text = "hi";
        let config = ChunkConfig::default();
        let chunks = chunk_document(text, 0, &config);
        assert!(chunks.is_empty());
    }
}
