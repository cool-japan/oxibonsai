//! Advanced document chunking strategies for RAG pipelines.
//!
//! Beyond simple fixed-size chunking, this module implements:
//! - Sentence-aware chunking (split on sentence boundaries)
//! - Recursive character text splitting (LangChain-style)
//! - Semantic chunking (group similar consecutive sentences)
//! - Sliding window with overlap
//! - Markdown/code-aware splitting

use std::collections::HashMap;

// ── RichChunk ─────────────────────────────────────────────────────────────────

/// A more detailed chunk with positional metadata.
#[derive(Debug, Clone)]
pub struct RichChunk {
    pub text: String,
    pub char_start: usize,
    pub char_end: usize,
    pub chunk_index: usize,
    pub metadata: HashMap<String, String>,
}

impl RichChunk {
    /// Create a new `RichChunk`.
    pub fn new(text: String, start: usize, end: usize, index: usize) -> Self {
        Self {
            text,
            char_start: start,
            char_end: end,
            chunk_index: index,
            metadata: HashMap::new(),
        }
    }

    /// Character count of the chunk text.
    pub fn len(&self) -> usize {
        self.text.chars().count()
    }

    /// Returns `true` if the chunk text is empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Approximate word count (whitespace-split).
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

// ── ChunkStrategy trait ───────────────────────────────────────────────────────

/// Strategy for chunking documents.
pub trait ChunkStrategy: Send + Sync {
    fn chunk(&self, text: &str) -> Vec<RichChunk>;
    fn name(&self) -> &'static str;
}

// ── SentenceChunker ───────────────────────────────────────────────────────────

/// Sentence-aware chunker: splits on [.!?] boundaries, then groups sentences
/// into chunks of at most `max_chars` characters.
pub struct SentenceChunker {
    pub max_chars: usize,
    pub overlap_sentences: usize,
}

impl SentenceChunker {
    /// Create with a maximum character limit per chunk and no overlap.
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            overlap_sentences: 0,
        }
    }

    /// Configure sentence overlap between consecutive chunks.
    pub fn with_overlap(mut self, sentences: usize) -> Self {
        self.overlap_sentences = sentences;
        self
    }
}

impl ChunkStrategy for SentenceChunker {
    fn name(&self) -> &'static str {
        "sentence"
    }

    fn chunk(&self, text: &str) -> Vec<RichChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks: Vec<RichChunk> = Vec::new();
        let mut i = 0usize;

        while i < sentences.len() {
            let mut group: Vec<&str> = Vec::new();
            let mut total_chars = 0usize;

            // Build a group of sentences that fit within max_chars.
            //
            // Counts *characters* (`RichChunk::len()`'s own unit), not
            // bytes: `s.len()` used to be compared directly against
            // `self.max_chars`, so a CJK sentence's multi-byte-per-char
            // encoding made the group fill up (and stop packing further
            // sentences in) at roughly 1/3 of the configured budget.
            let mut j = i;
            while j < sentences.len() {
                let s = sentences[j];
                let added = if group.is_empty() {
                    s.chars().count()
                } else {
                    s.chars().count() + 1
                };
                if !group.is_empty() && total_chars + added > self.max_chars {
                    break;
                }
                group.push(s);
                total_chars += added;
                j += 1;
            }

            if group.is_empty() {
                // Single sentence exceeds max_chars — emit it as-is
                group.push(sentences[i]);
                j = i + 1;
            }

            let chunk_text = group.join(" ");

            // Compute char_start by summing all sentences before this group
            let char_start: usize = sentences[..i].iter().map(|s| s.chars().count() + 1).sum();
            let char_end = char_start + chunk_text.chars().count();

            chunks.push(RichChunk::new(
                chunk_text,
                char_start,
                char_end,
                chunks.len(),
            ));

            // Advance, considering overlap
            let consumed = j - i;
            let step = if consumed > self.overlap_sentences {
                consumed - self.overlap_sentences
            } else {
                1
            };
            i += step.max(1);
        }

        chunks
    }
}

// ── RecursiveCharSplitter ─────────────────────────────────────────────────────

/// Recursive character splitter: tries separators in order until all chunks
/// are ≤ max_chars.
pub struct RecursiveCharSplitter {
    pub max_chars: usize,
    pub overlap: usize,
    pub separators: Vec<String>,
}

impl RecursiveCharSplitter {
    /// Create with default separators and no overlap.
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            overlap: 0,
            separators: Self::default_separators(),
        }
    }

    /// Configure character overlap between adjacent chunks.
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Override the separator list.
    pub fn with_separators(mut self, seps: Vec<String>) -> Self {
        self.separators = seps;
        self
    }

    /// Default separators: paragraph, line, space, character.
    pub fn default_separators() -> Vec<String> {
        vec![
            "\n\n".to_string(),
            "\n".to_string(),
            " ".to_string(),
            "".to_string(),
        ]
    }

    /// Recursively split text using the given separators.
    fn split_recursive(&self, text: &str, seps: &[String]) -> Vec<String> {
        if text.chars().count() <= self.max_chars {
            return vec![text.to_string()];
        }

        let sep = match seps.first() {
            Some(s) => s,
            None => {
                // No separators left — split by character
                return split_by_chars(text, self.max_chars, self.overlap);
            }
        };

        let remaining_seps = &seps[1..];

        if sep.is_empty() {
            // Split by individual characters
            return split_by_chars(text, self.max_chars, self.overlap);
        }

        let parts: Vec<&str> = text.split(sep.as_str()).collect();

        let mut result: Vec<String> = Vec::new();
        let mut current_group: Vec<&str> = Vec::new();
        let mut current_len = 0usize;

        for part in &parts {
            let part_len = part.chars().count();
            let sep_len = if current_group.is_empty() {
                0
            } else {
                sep.chars().count()
            };

            if current_len + sep_len + part_len > self.max_chars && !current_group.is_empty() {
                // Flush current group
                let joined = current_group.join(sep.as_str());
                if joined.chars().count() > self.max_chars {
                    // Need to recurse with the next separator
                    let sub = self.split_recursive(&joined, remaining_seps);
                    result.extend(sub);
                } else {
                    result.push(joined);
                }

                // Start new group with overlap
                if self.overlap > 0 {
                    // Keep last few items that fit within overlap
                    let mut overlap_items: Vec<&str> = Vec::new();
                    let mut overlap_len = 0usize;
                    for &item in current_group.iter().rev() {
                        let item_len = item.chars().count() + sep.chars().count();
                        if overlap_len + item_len > self.overlap {
                            break;
                        }
                        overlap_items.push(item);
                        overlap_len += item_len;
                    }
                    overlap_items.reverse();
                    current_group = overlap_items;
                    current_len = current_group
                        .iter()
                        .map(|s| s.chars().count())
                        .sum::<usize>()
                        + if current_group.len() > 1 {
                            (current_group.len() - 1) * sep.chars().count()
                        } else {
                            0
                        };
                } else {
                    current_group.clear();
                    current_len = 0;
                }
            }

            if part_len > self.max_chars {
                // This single part is too large — recurse on it
                if !current_group.is_empty() {
                    result.push(current_group.join(sep.as_str()));
                    current_group.clear();
                    current_len = 0;
                }
                let sub = self.split_recursive(part, remaining_seps);
                result.extend(sub);
            } else {
                let sep_add = if current_group.is_empty() {
                    0
                } else {
                    sep.chars().count()
                };
                current_len += sep_add + part_len;
                current_group.push(part);
            }
        }

        if !current_group.is_empty() {
            let joined = current_group.join(sep.as_str());
            if joined.chars().count() > self.max_chars {
                let sub = self.split_recursive(&joined, remaining_seps);
                result.extend(sub);
            } else {
                result.push(joined);
            }
        }

        result
    }
}

impl ChunkStrategy for RecursiveCharSplitter {
    fn name(&self) -> &'static str {
        "recursive"
    }

    fn chunk(&self, text: &str) -> Vec<RichChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let pieces = self.split_recursive(text, &self.separators.clone());

        let mut chunks: Vec<RichChunk> = Vec::new();
        let mut char_cursor = 0usize;
        let text_chars: Vec<char> = text.chars().collect();
        let total_chars = text_chars.len();

        for piece in pieces {
            if piece.is_empty() {
                continue;
            }
            let piece_len = piece.chars().count();
            // Find piece start in original text by scanning forward
            let start = find_substring_char_offset(&text_chars, &piece, char_cursor);
            let start = start.unwrap_or(char_cursor);
            let end = (start + piece_len).min(total_chars);

            chunks.push(RichChunk::new(piece, start, end, chunks.len()));
            char_cursor = start + piece_len;
        }

        chunks
    }
}

// ── SlidingWindowChunker ──────────────────────────────────────────────────────

/// Sliding window chunker with configurable step size.
pub struct SlidingWindowChunker {
    /// Window size in characters.
    pub window_size: usize,
    /// How far to advance each step.
    pub step_size: usize,
}

impl SlidingWindowChunker {
    /// Create with explicit window and step sizes.
    pub fn new(window_size: usize, step_size: usize) -> Self {
        Self {
            window_size,
            step_size: step_size.max(1),
        }
    }

    /// Non-overlapping: step equals window size.
    pub fn non_overlapping(size: usize) -> Self {
        Self::new(size, size)
    }

    /// 50% overlap: step is half the window size.
    pub fn with_50pct_overlap(size: usize) -> Self {
        let step = (size / 2).max(1);
        Self::new(size, step)
    }
}

impl ChunkStrategy for SlidingWindowChunker {
    fn name(&self) -> &'static str {
        "sliding_window"
    }

    fn chunk(&self, text: &str) -> Vec<RichChunk> {
        if text.is_empty() || self.window_size == 0 {
            return Vec::new();
        }

        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();
        let mut chunks: Vec<RichChunk> = Vec::new();
        let mut start = 0usize;

        while start < total {
            let end = (start + self.window_size).min(total);
            let chunk_text: String = chars[start..end].iter().collect();
            chunks.push(RichChunk::new(chunk_text, start, end, chunks.len()));

            if end == total {
                break;
            }
            start += self.step_size;
        }

        chunks
    }
}

// ── MarkdownChunker ───────────────────────────────────────────────────────────

/// Markdown-aware chunker: splits on headings (#, ##, ###).
pub struct MarkdownChunker {
    pub max_chars: usize,
    /// 1 = #, 2 = ##, etc. Only split on headings at or below this level.
    pub min_heading_level: u8,
}

impl MarkdownChunker {
    /// Create with `max_chars` limit and split on any heading level (1–6).
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            min_heading_level: 1,
        }
    }

    /// Determine the heading level of a line (0 if not a heading).
    fn heading_level(line: &str) -> u8 {
        let trimmed = line.trim_start();
        let count = trimmed.bytes().take_while(|&b| b == b'#').count();
        if count == 0 || count > 6 {
            return 0;
        }
        // Must be followed by a space or end of line
        let after = &trimmed[count..];
        if after.is_empty() || after.starts_with(' ') {
            count as u8
        } else {
            0
        }
    }
}

impl ChunkStrategy for MarkdownChunker {
    fn name(&self) -> &'static str {
        "markdown"
    }

    fn chunk(&self, text: &str) -> Vec<RichChunk> {
        if text.is_empty() {
            return Vec::new();
        }

        // Split text into sections at heading boundaries
        let lines: Vec<&str> = text.lines().collect();
        let mut sections: Vec<String> = Vec::new();
        let mut current: Vec<&str> = Vec::new();

        for line in &lines {
            let level = Self::heading_level(line);
            if level > 0 && level <= self.min_heading_level + 5 {
                // This is a heading line — start a new section
                if !current.is_empty() {
                    sections.push(current.join("\n"));
                    current.clear();
                }
            }
            current.push(line);
        }
        if !current.is_empty() {
            sections.push(current.join("\n"));
        }

        // Now apply max_chars within each section using recursive splitting
        let splitter = RecursiveCharSplitter::new(self.max_chars);
        let mut chunks: Vec<RichChunk> = Vec::new();
        let mut char_cursor = 0usize;

        for section in &sections {
            if section.trim().is_empty() {
                char_cursor += section.chars().count() + 1; // +1 for newline between sections
                continue;
            }

            let section_len = section.chars().count();

            if section_len <= self.max_chars {
                let start = char_cursor;
                let end = start + section_len;
                chunks.push(RichChunk::new(section.clone(), start, end, chunks.len()));
            } else {
                // Recursively split the section
                let sub_chunks = splitter.chunk(section);
                for mut sc in sub_chunks {
                    sc.char_start += char_cursor;
                    sc.char_end += char_cursor;
                    sc.chunk_index = chunks.len();
                    chunks.push(sc);
                }
            }

            char_cursor += section_len + 1; // +1 for the newline between sections
        }

        chunks
    }
}

// ── ChunkerRegistry ───────────────────────────────────────────────────────────

/// A chunker registry for selecting strategies by name.
pub struct ChunkerRegistry {
    strategies: HashMap<String, Box<dyn ChunkStrategy>>,
}

impl ChunkerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
        }
    }

    /// Register a strategy; the strategy's `name()` is used as the key.
    pub fn register(&mut self, strategy: Box<dyn ChunkStrategy>) {
        self.strategies
            .insert(strategy.name().to_string(), strategy);
    }

    /// Chunk text using the named strategy, returning `None` if not found.
    pub fn chunk(&self, name: &str, text: &str) -> Option<Vec<RichChunk>> {
        self.strategies.get(name).map(|s| s.chunk(text))
    }

    /// List all registered strategy names.
    pub fn available_strategies(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.strategies.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Build a registry pre-loaded with Sentence, Recursive, SlidingWindow, and Markdown.
    pub fn default_registry() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(SentenceChunker::new(512)));
        registry.register(Box::new(RecursiveCharSplitter::new(512)));
        registry.register(Box::new(SlidingWindowChunker::non_overlapping(512)));
        registry.register(Box::new(MarkdownChunker::new(512)));
        registry
    }
}

impl Default for ChunkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Split text on sentence-ending punctuation followed by whitespace or end.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' || b == b'!' || b == b'?' {
            // Consume consecutive punctuation
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b'.' || bytes[j] == b'!' || bytes[j] == b'?') {
                j += 1;
            }
            // Check that it's followed by whitespace or end-of-string
            let at_end = j >= bytes.len();
            let followed_by_space = !at_end
                && (bytes[j] == b' '
                    || bytes[j] == b'\t'
                    || bytes[j] == b'\n'
                    || bytes[j] == b'\r');

            if at_end || followed_by_space {
                // Skip whitespace
                while j < bytes.len()
                    && (bytes[j] == b' '
                        || bytes[j] == b'\t'
                        || bytes[j] == b'\n'
                        || bytes[j] == b'\r')
                {
                    j += 1;
                }
                let sentence = text[start..j].trim();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }

    // Trailing text without terminal punctuation
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    sentences
}

/// Split text into fixed-size character windows with optional overlap.
fn split_by_chars(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total == 0 {
        return Vec::new();
    }

    let step = max_chars.saturating_sub(overlap).max(1);
    let mut result = Vec::new();
    let mut start = 0usize;

    while start < total {
        let end = (start + max_chars).min(total);
        let chunk: String = chars[start..end].iter().collect();
        result.push(chunk);
        if end == total {
            break;
        }
        start += step;
    }

    result
}

/// Find the character-offset of `needle` in `haystack_chars` starting from `from`.
fn find_substring_char_offset(haystack: &[char], needle: &str, from: usize) -> Option<usize> {
    let needle_chars: Vec<char> = needle.chars().collect();
    let needle_len = needle_chars.len();

    if needle_len == 0 {
        return Some(from);
    }

    let limit = if haystack.len() >= needle_len {
        haystack.len() - needle_len + 1
    } else {
        return None;
    };

    for i in from..limit {
        if haystack[i..i + needle_len] == needle_chars[..] {
            return Some(i);
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod inline_tests {
    use super::*;

    #[test]
    fn rich_chunk_len_and_words() {
        let chunk = RichChunk::new("hello world".to_string(), 0, 11, 0);
        assert_eq!(chunk.len(), 11);
        assert_eq!(chunk.word_count(), 2);
    }

    #[test]
    fn sentence_chunker_splits() {
        let chunker = SentenceChunker::new(200);
        let text = "Hello world. This is a test. Another sentence here.";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn sentence_chunker_packs_cjk_sentences_by_char_count_not_byte_count() {
        // Each sentence is 8 *chars* (7 CJK codepoints, each 3 UTF-8 bytes,
        // plus an ASCII '.') but 22 *bytes* -- the same byte-vs-char class
        // as RAG-EVAL-IMG-20, reproduced inside `SentenceChunker::chunk`
        // (RichChunk::len() counts chars, but the accumulation used to
        // compare `s.len()` (bytes) against `max_chars`).
        let sentence = "日本語のテスト.";
        assert_eq!(sentence.chars().count(), 8, "sanity: 8 chars");
        assert_eq!(sentence.len(), 22, "sanity: 22 bytes");
        let text = format!("{sentence} {sentence} {sentence}");

        let chunker = SentenceChunker::new(20);
        let chunks = chunker.chunk(&text);

        // Every chunk must respect the budget in CHARS (RichChunk::len()).
        for c in &chunks {
            assert!(
                c.len() <= 20,
                "chunk {:?} has {} chars, exceeds max_chars=20",
                c.text,
                c.len()
            );
        }

        // With a correct char-based budget, two 8-char sentences (17 chars
        // joined by one space) fit comfortably under 20 and must be packed
        // into the same chunk. A byte-based accumulation instead sees ~22
        // "chars" from the very first sentence alone -- already over
        // budget -- and emits one sentence per chunk (3 chunks, not 2).
        assert_eq!(
            chunks.len(),
            2,
            "expected two 8-char CJK sentences to pack into one 20-char-budget \
             chunk; got {} chunks: {:?}",
            chunks.len(),
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
        assert_eq!(
            chunks[0].len(),
            17,
            "first chunk should contain both packed sentences (17 chars)"
        );
    }

    #[test]
    fn recursive_splitter_short() {
        let splitter = RecursiveCharSplitter::new(1000);
        let text = "short text";
        let chunks = splitter.chunk(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }

    #[test]
    fn sliding_window_basic() {
        let chunker = SlidingWindowChunker::non_overlapping(5);
        let chunks = chunker.chunk("abcdefghij");
        assert_eq!(chunks.len(), 2);
    }
}
