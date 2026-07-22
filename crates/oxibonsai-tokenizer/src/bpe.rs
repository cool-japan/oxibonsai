//! Byte-Pair Encoding (BPE) merge table and encoding routines.
//!
//! This module implements:
//! - [`BpeMerges`] — a merge table mapping symbol pairs to merged-token IDs
//! - [`bpe_encode`] — greedy BPE encoding of a pre-tokenized word
//! - [`pretokenize`] — GPT-2–style whitespace/punctuation split
//! - [`byte_fallback_id`] — produce a `<0xHH>` token name for unknown bytes

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::vocab::Vocabulary;

// ── BpeMerges ────────────────────────────────────────────────────────────────

/// A single merge-table entry: its priority rank and the merged token's ID.
///
/// Storing the priority directly next to the result ID makes both
/// [`BpeMerges::get_merge_priority`] and [`BpeMerges::get_merge_result`] true
/// `O(1)` `HashMap` look-ups.  The previous implementation re-derived the
/// priority with a linear `Vec::position` scan over the whole ordered merge
/// list, which was `O(total-merge-count)` per candidate pair and made encoding
/// with a production-scale vocabulary (e.g. Qwen3's ~150K merges) impractically
/// slow.
#[derive(Debug, Clone, Copy)]
struct MergeEntry {
    /// 0-based priority rank (insertion order); lower = higher priority.
    rank: u32,
    /// ID of the merged token.
    result_id: u32,
}

/// BPE merge table: a set of (A, B) → merged-ID rules ordered by priority.
///
/// Lower priority index = earlier merge (higher priority).
#[derive(Debug, Clone, Default)]
pub struct BpeMerges {
    /// Maps a symbol pair to its `(rank, merged-token-ID)`.
    merges: HashMap<(String, String), MergeEntry>,
    /// Rank to assign to the next distinct pair inserted (monotonically
    /// increasing, so insertion order defines priority).
    next_rank: u32,
}

impl BpeMerges {
    /// Create an empty merge table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a merge rule: `a` + `b` → token with ID `result_id`.
    ///
    /// Duplicate entries (same pair) preserve the original priority rank while
    /// overwriting only the result ID (matching the previous behaviour, where
    /// the order-slot was kept and just the mapped ID replaced).
    pub fn add_merge(&mut self, a: &str, b: &str, result_id: u32) {
        match self.merges.entry((a.to_owned(), b.to_owned())) {
            Entry::Occupied(mut occupied) => {
                // Preserve the existing rank; overwrite only the result ID.
                occupied.get_mut().result_id = result_id;
            }
            Entry::Vacant(vacant) => {
                let rank = self.next_rank;
                self.next_rank += 1;
                vacant.insert(MergeEntry { rank, result_id });
            }
        }
    }

    /// Return the 0-based priority index for a merge pair, if it exists.
    ///
    /// Lower index = higher priority (applied first during encoding).
    pub fn get_merge_priority(&self, a: &str, b: &str) -> Option<usize> {
        self.rank(a, b).map(|rank| rank as usize)
    }

    /// Return the raw priority rank for a merge pair, if it exists.
    ///
    /// This is the internal `O(1)` hot-path used by the BPE merge loop; it
    /// avoids the `usize` widening done by [`Self::get_merge_priority`].
    fn rank(&self, a: &str, b: &str) -> Option<u32> {
        // `HashMap<(String, String), _>` cannot be probed with a `(&str, &str)`
        // borrow, so we build an owned key.  This mirrors the reference Qwen3
        // encoder in `oxibonsai-image` and is dwarfed by the eliminated linear
        // scan.
        self.merges
            .get(&(a.to_owned(), b.to_owned()))
            .map(|entry| entry.rank)
    }

    /// Return the merged token ID for a pair, if a rule exists.
    pub fn get_merge_result(&self, a: &str, b: &str) -> Option<u32> {
        self.merges
            .get(&(a.to_owned(), b.to_owned()))
            .map(|entry| entry.result_id)
    }

    /// Number of merge rules in the table.
    pub fn len(&self) -> usize {
        self.merges.len()
    }

    /// Returns `true` if the merge table is empty.
    pub fn is_empty(&self) -> bool {
        self.merges.is_empty()
    }
}

// ── Pre-tokenizer ─────────────────────────────────────────────────────────────

/// Split text into pre-tokens using a GPT-2–style rule:
/// words (optionally preceded by a space) are separated from punctuation and
/// standalone whitespace runs.
///
/// The returned strings are the raw Unicode chunks to be BPE-encoded
/// individually.
pub fn pretokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut last_was_space = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            last_was_space = true;
        } else if ch.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            // Punctuation gets its own token; add the leading space prefix if
            // the previous character was whitespace (GPT-2 convention: Ġ prefix).
            let mut tok = String::new();
            if last_was_space {
                tok.push('\u{0120}'); // Ġ — GPT-2 space prefix
            }
            tok.push(ch);
            tokens.push(tok);
            last_was_space = false;
        } else {
            if last_was_space && !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            if last_was_space && current.is_empty() {
                current.push('\u{0120}'); // Leading Ġ prefix
            }
            current.push(ch);
            last_was_space = false;
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// GPT-2 / ByteLevel pre-tokenization that **preserves whitespace-run
/// structure** (tabs, newlines, carriage returns, and runs of multiple spaces).
///
/// This is a hand-written port of the canonical GPT-2 pre-tokenization regex
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// used (with `Isolated` behaviour) by the HuggingFace `tokenizers` ByteLevel
/// pipeline for Qwen3 / Llama-3 / GPT-2.  Unlike [`pretokenize`], which folds
/// every whitespace run into a single `Ġ` marker (silently discarding tabs,
/// newlines and repeated spaces), each returned piece keeps its literal
/// whitespace characters so that — after the bytes→unicode remap — they match
/// the `Ċ`/`Ġ`-bearing vocabulary entries real `tokenizer.json` files ship.
///
/// The returned pieces still contain raw UTF-8; the caller is responsible for
/// applying the bytes→unicode map (see [`bpe_encode_bytelevel`]).
pub fn pretokenize_gpt2(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;

    let is_letter = |c: char| c.is_alphabetic();
    let is_number = |c: char| c.is_numeric();
    let is_ws = |c: char| c.is_whitespace();
    let is_nl = |c: char| c == '\r' || c == '\n';

    while i < n {
        let c = chars[i];

        // 1) contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == '\'' && i + 1 < n {
            if let Some(len) = match_contraction(&chars[i..]) {
                out.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }

        // 2) [^\r\n\p{L}\p{N}]?\p{L}+  (optional single non-nl/non-alnum, then letters)
        {
            let mut j = i;
            if !is_nl(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                // Only consume the leading char if a letter follows.
                if j + 1 < n && is_letter(chars[j + 1]) {
                    j += 1;
                }
            }
            if j < n && is_letter(chars[j]) {
                while j < n && is_letter(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // 3) \p{N}  (a single number character)
        if is_number(c) {
            out.push(c.to_string());
            i += 1;
            continue;
        }

        // 4)  ?[^\s\p{L}\p{N}]+[\r\n]*  (optional space, run of symbols, trailing newlines)
        {
            let mut j = i;
            if chars[j] == ' '
                && j + 1 < n
                && !is_ws(chars[j + 1])
                && !is_letter(chars[j + 1])
                && !is_number(chars[j + 1])
            {
                j += 1;
            }
            if j < n && !is_ws(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                while j < n && !is_ws(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                    j += 1;
                }
                while j < n && is_nl(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // 5) \s*[\r\n]+  (whitespace ending in newlines)
        if is_ws(c) {
            let mut j = i;
            while j < n && is_ws(chars[j]) && !is_nl(chars[j]) {
                j += 1;
            }
            if j < n && is_nl(chars[j]) {
                while j < n && is_nl(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // 6) \s+(?!\S)  and  7) \s+
        if is_ws(c) {
            let mut j = i;
            while j < n && is_ws(chars[j]) {
                j += 1;
            }
            if j == n {
                // Trailing whitespace run: take it all (\s+(?!\S)).
                out.push(chars[i..j].iter().collect());
                i = j;
            } else if j - i >= 2 {
                // Leave the final space to attach to the following token
                // (GPT-2 keeps a single leading space with the next word).
                out.push(chars[i..j - 1].iter().collect());
                i = j - 1;
            } else {
                // A lone space before a word that branch 2/4 did not consume:
                // emit it standalone (it byte-encodes to 'Ġ').
                out.push(chars[i..j].iter().collect());
                i = j;
            }
            continue;
        }

        // Fallback: emit the single char (unreachable for valid text).
        out.push(c.to_string());
        i += 1;
    }

    out
}

/// Match a leading contraction (`'s 't 're 've 'm 'll 'd`, case-insensitive),
/// returning its char length if present.
fn match_contraction(rest: &[char]) -> Option<usize> {
    // `rest[0] == '\''` is guaranteed by the caller.
    let lower = |c: char| c.to_ascii_lowercase();
    if rest.len() >= 2 {
        let c1 = lower(rest[1]);
        // 3-char: 're 've 'll
        if rest.len() >= 3 {
            let c2 = lower(rest[2]);
            if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                return Some(3);
            }
        }
        // 2-char: 's 't 'm 'd
        if matches!(c1, 's' | 't' | 'm' | 'd') {
            return Some(2);
        }
    }
    None
}

// ── BPE encoder ──────────────────────────────────────────────────────────────

/// Greedy BPE encode a single pre-tokenized word.
///
/// Algorithm:
/// 1. Split the word into individual Unicode characters as the initial symbol
///    sequence.
/// 2. Repeatedly find the pair with the lowest priority index in `merges` and
///    merge it.
/// 3. Continue until no more merges apply.
/// 4. Map each remaining symbol to its vocabulary ID; use byte-fallback for
///    any symbol not found in the vocabulary.
pub fn bpe_encode(word: &str, vocab: &Vocabulary, merges: &BpeMerges) -> Vec<u32> {
    if word.is_empty() {
        return Vec::new();
    }

    // Map symbols → token IDs, falling back to `<0xHH>` byte tokens.
    bpe_merge_symbols(word, merges)
        .iter()
        .flat_map(|sym| symbol_to_ids(sym, vocab))
        .collect()
}

/// Greedy BPE-encode a byte-level pre-tokenized piece.
///
/// This is the ByteLevel (GPT-2 / Qwen3 / Llama-3) counterpart of
/// [`bpe_encode`].  The caller is expected to have already remapped the raw
/// UTF-8 bytes of the piece through the GPT-2 bytes→unicode table (see
/// [`crate::hf_format::bytes_to_unicode_string`]), so every resulting symbol is
/// a printable-unicode code point that a byte-level vocabulary carries directly.
///
/// Unlike [`bpe_encode`], no `<0xHH>` byte-fallback is attempted: byte-level
/// vocabularies do not ship those SentencePiece-style tokens, and every single
/// byte already has its own vocabulary entry.  Any symbol that is genuinely
/// absent maps to `unk_id`.
pub fn bpe_encode_bytelevel(
    byte_level_piece: &str,
    vocab: &Vocabulary,
    merges: &BpeMerges,
    unk_id: u32,
) -> Vec<u32> {
    if byte_level_piece.is_empty() {
        return Vec::new();
    }

    bpe_merge_symbols(byte_level_piece, merges)
        .iter()
        .map(|sym| vocab.get_id(sym).unwrap_or(unk_id))
        .collect()
}

/// Apply the BPE merge loop to a piece and return the final symbol strings.
///
/// Algorithm:
/// 1. Split the piece into individual Unicode characters as the initial symbol
///    sequence.
/// 2. Repeatedly find the pair with the lowest priority rank and merge its
///    left-most occurrence.
/// 3. Continue until no more merges apply.
///
/// The per-pair priority look-up is `O(1)` (a single `HashMap` probe), so the
/// loop is `O(word_len^2)` in the pathological case regardless of how large the
/// merge table is.
fn bpe_merge_symbols(word: &str, merges: &BpeMerges) -> Vec<String> {
    // Initialise the symbol sequence as individual characters.
    let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

    // Iteratively apply the highest-priority merge.
    loop {
        if symbols.len() < 2 {
            break;
        }

        // Find the pair with the best (lowest rank) priority.
        let best = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(pos, pair)| merges.rank(&pair[0], &pair[1]).map(|rank| (rank, pos)))
            .min_by_key(|&(rank, _)| rank);

        match best {
            None => break, // No more applicable merges.
            Some((_, pos)) => {
                // Merge symbols[pos] and symbols[pos+1].
                let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
                symbols[pos] = merged;
                symbols.remove(pos + 1);
            }
        }
    }

    symbols
}

/// Convert a symbol to one or more token IDs.
///
/// If the symbol is directly in the vocabulary, return its single ID.
/// Otherwise attempt UTF-8 byte fallback: each byte is encoded as `<0xHH>`.
fn symbol_to_ids(sym: &str, vocab: &Vocabulary) -> Vec<u32> {
    if let Some(id) = vocab.get_id(sym) {
        return vec![id];
    }

    // Byte fallback.
    sym.as_bytes()
        .iter()
        .filter_map(|&b| {
            let fallback = byte_fallback_id(b);
            vocab.get_id(&fallback)
        })
        .collect()
}

// ── Byte fallback ─────────────────────────────────────────────────────────────

/// Return the byte-fallback token name for a single byte value.
///
/// Format: `<0xHH>` where `HH` is the uppercase hexadecimal byte value.
///
/// # Example
/// ```
/// use oxibonsai_tokenizer::bpe::byte_fallback_id;
/// assert_eq!(byte_fallback_id(0x20), "<0x20>");
/// assert_eq!(byte_fallback_id(0xFF), "<0xFF>");
/// ```
pub fn byte_fallback_id(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocabulary;

    fn make_vocab_with_merges() -> (Vocabulary, BpeMerges) {
        let mut vocab = Vocabulary::new();
        // Individual characters
        vocab.insert("h", 10);
        vocab.insert("e", 11);
        vocab.insert("l", 12);
        vocab.insert("o", 13);
        // Merged tokens
        vocab.insert("he", 20);
        vocab.insert("hel", 21);
        vocab.insert("hell", 22);
        vocab.insert("hello", 23);
        vocab.insert("lo", 24);

        let mut merges = BpeMerges::new();
        merges.add_merge("h", "e", 20);
        merges.add_merge("he", "l", 21);
        merges.add_merge("hel", "l", 22);
        merges.add_merge("hell", "o", 23);
        merges.add_merge("l", "o", 24);

        (vocab, merges)
    }

    #[test]
    fn byte_fallback_format() {
        assert_eq!(byte_fallback_id(0x00), "<0x00>");
        assert_eq!(byte_fallback_id(0x20), "<0x20>");
        assert_eq!(byte_fallback_id(0xFF), "<0xFF>");
        assert_eq!(byte_fallback_id(0x0A), "<0x0A>");
    }

    #[test]
    fn bpe_merges_priority() {
        let mut m = BpeMerges::new();
        m.add_merge("a", "b", 1);
        m.add_merge("b", "c", 2);
        assert_eq!(m.get_merge_priority("a", "b"), Some(0));
        assert_eq!(m.get_merge_priority("b", "c"), Some(1));
        assert_eq!(m.get_merge_priority("x", "y"), None);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn bpe_encode_hello() {
        let (vocab, merges) = make_vocab_with_merges();
        let ids = bpe_encode("hello", &vocab, &merges);
        // Should merge all the way to "hello" → id 23
        assert_eq!(ids, vec![23]);
    }

    #[test]
    fn bpe_encode_empty() {
        let (vocab, merges) = make_vocab_with_merges();
        let ids = bpe_encode("", &vocab, &merges);
        assert!(ids.is_empty());
    }

    #[test]
    fn pretokenize_simple_sentence() {
        let tokens = pretokenize("hello world");
        assert!(!tokens.is_empty());
        // Should split into at least "hello" and "Ġworld"
        assert!(tokens.iter().any(|t| t.contains("hello") || t == "hello"));
    }

    #[test]
    fn pretokenize_empty() {
        assert!(pretokenize("").is_empty());
    }

    #[test]
    fn pretokenize_punctuation_splits() {
        let tokens = pretokenize("hi,there");
        // Should split around the comma
        assert!(tokens.len() >= 2);
    }

    // ── Finding 34: O(1) merge-priority correctness ─────────────────────────

    #[test]
    fn merge_priority_is_correct_for_late_inserted_pair() {
        // The rank must equal the insertion order even for the last pair added
        // to a large table (this is the case the old linear scan handled slowly
        // but the new HashMap must still get numerically right).
        let mut m = BpeMerges::new();
        for i in 0..1_000u32 {
            m.add_merge(&format!("l{i}"), &format!("r{i}"), i);
        }
        assert_eq!(m.get_merge_priority("l0", "r0"), Some(0));
        assert_eq!(m.get_merge_priority("l999", "r999"), Some(999));
        assert_eq!(m.get_merge_result("l999", "r999"), Some(999));
        assert_eq!(m.len(), 1_000);
    }

    #[test]
    fn duplicate_merge_preserves_rank_overwrites_result() {
        let mut m = BpeMerges::new();
        m.add_merge("a", "b", 10);
        m.add_merge("c", "d", 11);
        // Re-insert (a,b) with a new result id — rank must stay 0.
        m.add_merge("a", "b", 99);
        assert_eq!(m.get_merge_priority("a", "b"), Some(0));
        assert_eq!(m.get_merge_result("a", "b"), Some(99));
        assert_eq!(m.get_merge_priority("c", "d"), Some(1));
        assert_eq!(m.len(), 2);
    }

    // ── Finding 14: whitespace-preserving GPT-2 pre-tokenizer ────────────────

    #[test]
    fn pretokenize_gpt2_preserves_newline() {
        let pieces = pretokenize_gpt2("hello\nworld");
        assert_eq!(pieces, vec!["hello", "\n", "world"]);
    }

    #[test]
    fn pretokenize_gpt2_preserves_repeated_spaces() {
        // Two spaces between words: GPT-2 keeps one space attached to the next
        // word and emits the extra space as its own piece.
        let pieces = pretokenize_gpt2("a  b");
        let joined: String = pieces.concat();
        assert_eq!(joined, "a  b", "no whitespace may be lost: {pieces:?}");
    }

    #[test]
    fn pretokenize_gpt2_leading_space_attaches_to_word() {
        let pieces = pretokenize_gpt2("a tiny bonsai");
        assert_eq!(pieces, vec!["a", " tiny", " bonsai"]);
    }

    #[test]
    fn pretokenize_gpt2_is_lossless() {
        for text in [
            "hello\tworld",
            "line1\r\nline2",
            "  spaced  out  ",
            "café 中文 😀!",
            "",
        ] {
            let joined: String = pretokenize_gpt2(text).concat();
            assert_eq!(joined, text, "pretokenize_gpt2 dropped bytes for {text:?}");
        }
    }

    // ── Finding 9: byte-level encode uses unk, not <0xHH> fallback ────────────

    #[test]
    fn bpe_encode_bytelevel_maps_and_falls_back_to_unk() {
        let mut vocab = Vocabulary::new();
        vocab.insert("a", 1);
        vocab.insert("b", 2);
        // Note: no `<0xHH>` byte-fallback tokens are registered.
        let merges = BpeMerges::new();
        // "ab" — both present.
        assert_eq!(bpe_encode_bytelevel("ab", &vocab, &merges, 0), vec![1, 2]);
        // "aZ" — 'Z' absent, must become unk (7), NOT a `<0x5A>` fallback.
        assert_eq!(bpe_encode_bytelevel("aZ", &vocab, &merges, 7), vec![1, 7]);
    }
}
