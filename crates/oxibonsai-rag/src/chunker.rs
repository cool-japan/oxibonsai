//! Document chunking strategies for the RAG pipeline.
//!
//! This module provides three chunking approaches:
//!
//! 1. [`chunk_document`] — fixed-size character windows with configurable overlap.
//! 2. [`chunk_by_sentences`] — group consecutive sentences up to a maximum count.
//! 3. [`chunk_by_paragraphs`] — split on blank lines (paragraph boundaries).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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
    /// Byte offset of the first character of this chunk in the original document.
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
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-size sliding-window chunker
// ─────────────────────────────────────────────────────────────────────────────

/// Split `text` into overlapping fixed-size character windows.
///
/// The step between consecutive windows is `config.chunk_size - config.overlap`.
/// Chunks smaller than `config.min_chunk_size` characters are discarded.
///
/// Returns an empty `Vec` if `text` is empty.
pub fn chunk_document(text: &str, doc_id: usize, config: &ChunkConfig) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }

    // Collect Unicode scalar values so we can slice by char index cleanly
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total < config.min_chunk_size {
        // The whole document is a single chunk only if it is not below min
        return Vec::new();
    }

    let step = config.chunk_size.saturating_sub(config.overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < total {
        let end = (start + config.chunk_size).min(total);
        let chunk_chars = &chars[start..end];
        let chunk_text: String = chunk_chars.iter().collect();

        if chunk_text.chars().count() >= config.min_chunk_size {
            // Compute byte offset of `start`-th char in the original string
            let byte_offset = byte_offset_of_char(text, start);
            let chunk_idx = chunks.len();
            chunks.push(Chunk::new(chunk_text, doc_id, chunk_idx, byte_offset));
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
    let mut sentence_start_byte = 0usize;
    let mut i = 0usize;

    while i < sentences.len() {
        let batch_start = i;
        let batch_end = (i + max_sentences).min(sentences.len());
        let batch: Vec<&str> = sentences[batch_start..batch_end].to_vec();
        let chunk_text = batch.join(" ");

        let chunk_idx = chunks.len();
        chunks.push(Chunk::new(
            chunk_text,
            doc_id,
            chunk_idx,
            sentence_start_byte,
        ));

        // Advance byte offset by the bytes consumed in this batch
        for s in &batch {
            sentence_start_byte += s.len();
            // Account for the whitespace that was between sentences in the original
            sentence_start_byte += 1; // rough approximation
        }

        i = batch_end;
    }

    chunks
}

/// Lightweight sentence splitter based on terminal punctuation.
fn split_sentences(text: &str) -> Vec<&str> {
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
            let sentence = text[start..j].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = j;
            i = j;
        } else {
            i += 1;
        }
    }
    // Trailing text without terminal punctuation
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

// ─────────────────────────────────────────────────────────────────────────────
// Paragraph-based chunker
// ─────────────────────────────────────────────────────────────────────────────

/// Split `text` into chunks at paragraph boundaries (one or more blank lines).
///
/// Each non-empty paragraph (after whitespace-trimming) becomes a separate chunk.
/// Returns an empty `Vec` if `text` is empty.
pub fn chunk_by_paragraphs(text: &str, doc_id: usize) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }

    // Split on sequences of two or more newlines (blank line separator)
    let mut chunks = Vec::new();
    let mut byte_cursor = 0usize;

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
                // We have just hit the first blank line — emit the paragraph
                let para = text[para_start..line_start].trim();
                if !para.is_empty() {
                    let chunk_idx = chunks.len();
                    chunks.push(Chunk::new(para.to_string(), doc_id, chunk_idx, byte_cursor));
                    byte_cursor = i;
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
            // Emit any remaining paragraph
            let para = text[para_start..].trim();
            if !para.is_empty() {
                let chunk_idx = chunks.len();
                chunks.push(Chunk::new(para.to_string(), doc_id, chunk_idx, byte_cursor));
            }
            break;
        }

        i += 1;
    }

    chunks
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the byte offset of the `n`-th Unicode scalar value in `s`.
fn byte_offset_of_char(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map(|(b, _)| b).unwrap_or(s.len())
}
