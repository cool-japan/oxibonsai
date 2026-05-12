//! WordPiece tokenizer — greedy longest-match segmentation for BERT/RoBERTa/DeBERTa.
//!
//! The WordPiece algorithm (Schuster & Nakamura, 2012) pre-tokenizes input by
//! splitting on whitespace and then greedily segments each word into the longest
//! vocabulary subwords, using a `##` prefix for continuation (non-initial)
//! pieces.
//!
//! ## Example
//!
//! Vocabulary: `["play", "##ing", "##s", "hello"]`
//!
//! - `"playing"` → `["play", "##ing"]`
//! - `"hello"` → `["hello"]`
//! - `"xyz"` (not in vocab) → `["[UNK]"]`
//!
//! ## UNK handling
//!
//! If a word exceeds `max_input_chars_per_word` (default 100 characters), or if
//! no segmentation can cover all characters in the word, the whole word is
//! replaced with a single UNK token.
//!
//! ## UTF-8 safety
//!
//! The algorithm works at Unicode character boundaries, not byte offsets, so
//! multi-byte codepoints are never split in the middle.

use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

/// The `##` prefix that marks WordPiece continuation (non-initial) tokens.
///
/// When a subword does not start a whitespace-separated word, it is stored and
/// looked up with this prefix prepended.  For example, `"ing"` inside
/// `"playing"` is represented as `"##ing"`.
pub const WORDPIECE_CONTINUATION_PREFIX: &str = "##";

// ── WordPieceError ────────────────────────────────────────────────────────────

/// Errors that can occur while constructing a [`WordPieceVocab`].
#[derive(Debug, Clone, PartialEq)]
pub enum WordPieceError {
    /// The supplied token list was empty.
    EmptyVocab,
    /// The `unk_id` falls outside the valid token-ID range.
    UnkOutOfRange {
        /// The unk_id that was provided.
        unk_id: u32,
        /// The number of tokens in the vocabulary.
        vocab_len: usize,
    },
    /// Two or more tokens have the same string.
    DuplicateToken(String),
}

impl std::fmt::Display for WordPieceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyVocab => write!(f, "vocabulary is empty"),
            Self::UnkOutOfRange { unk_id, vocab_len } => {
                write!(
                    f,
                    "unk_id {unk_id} is out of range for vocabulary of size {vocab_len}"
                )
            }
            Self::DuplicateToken(t) => write!(f, "duplicate token in WordPiece vocabulary: {t:?}"),
        }
    }
}

impl std::error::Error for WordPieceError {}

// ── WordPieceVocab ────────────────────────────────────────────────────────────

/// WordPiece vocabulary with greedy longest-match-first tokenization.
///
/// Used by BERT, RoBERTa, DeBERTa, DistilBERT, and ALBERT model families.
///
/// Tokens are indexed by position in the construction list: `tokens[i]` has
/// ID `i`.  Continuation tokens carry the `##` prefix in the vocabulary, e.g.
/// `"##ing"` is a distinct vocabulary entry from `"ing"`.
///
/// # Thread safety
///
/// [`WordPieceVocab`] is `Send + Sync` and can safely be shared across threads.
#[derive(Debug, Clone)]
pub struct WordPieceVocab {
    /// Forward map: token string → integer ID.
    token_to_id: HashMap<String, u32>,
    /// Reverse map: integer ID → token string (indexed by position).
    id_to_token: Vec<String>,
    /// Token ID emitted when a word cannot be segmented.
    unk_id: u32,
    /// Maximum number of Unicode scalar values a word may contain before it is
    /// immediately replaced with UNK (default: 100).
    max_input_chars_per_word: usize,
}

impl WordPieceVocab {
    /// Construct a [`WordPieceVocab`] from an ordered list of token strings.
    ///
    /// The position of each token determines its integer ID: `tokens[0]` has
    /// ID `0`, `tokens[1]` has ID `1`, and so on.
    ///
    /// # Errors
    ///
    /// - [`WordPieceError::EmptyVocab`] — if `tokens` is empty.
    /// - [`WordPieceError::UnkOutOfRange`] — if `unk_id >= tokens.len()`.
    /// - [`WordPieceError::DuplicateToken`] — if any token string appears more
    ///   than once.
    pub fn new(tokens: Vec<String>, unk_id: u32) -> Result<Self, WordPieceError> {
        if tokens.is_empty() {
            return Err(WordPieceError::EmptyVocab);
        }
        if unk_id as usize >= tokens.len() {
            return Err(WordPieceError::UnkOutOfRange {
                unk_id,
                vocab_len: tokens.len(),
            });
        }

        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            if token_to_id.insert(token.clone(), i as u32).is_some() {
                return Err(WordPieceError::DuplicateToken(token.clone()));
            }
        }

        Ok(Self {
            id_to_token: tokens,
            token_to_id,
            unk_id,
            max_input_chars_per_word: 100,
        })
    }

    /// Override the maximum number of Unicode scalar values per word.
    ///
    /// Words longer than this limit are immediately emitted as a single UNK
    /// token without attempting segmentation.  The HuggingFace default is 100.
    ///
    /// Returns `self` for builder-style chaining.
    pub fn with_max_input_chars(mut self, max: usize) -> Self {
        self.max_input_chars_per_word = max;
        self
    }

    /// Encode `text` into a sequence of token IDs using greedy WordPiece
    /// segmentation.
    ///
    /// The text is first split on ASCII whitespace.  Each resulting word is
    /// independently segmented via greedy longest-match-first with `##`-prefixed
    /// continuation tokens.  Words that cannot be fully segmented are replaced
    /// by a single UNK token.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        for word in text.split_whitespace() {
            self.tokenize_word_into(word, &mut result);
        }
        result
    }

    /// Greedy-longest-match segmentation of a single whitespace-delimited word.
    ///
    /// Algorithm (Schuster & Nakamura 2012):
    /// 1. Collect Unicode character boundaries for the word.
    /// 2. Starting at character `start = 0`, try every span `[start, end]` from
    ///    longest to shortest:
    ///    - If `start == 0`, look up the bare substring.
    ///    - Otherwise look up `"##" + substring`.
    ///    - On a hit, record the token ID and advance `start = end`.
    /// 3. If no single-character span (or `##`-char) exists, mark the word as
    ///    bad: remove all tokens pushed so far and emit one UNK.
    fn tokenize_word_into(&self, word: &str, out: &mut Vec<u32>) {
        let char_count = word.chars().count();
        if char_count > self.max_input_chars_per_word {
            out.push(self.unk_id);
            return;
        }

        // Collect the byte index of each character boundary plus the end sentinel.
        // This allows O(1) extraction of the substring corresponding to any
        // character-index range [i, j).
        let char_boundaries: Vec<usize> = word
            .char_indices()
            .map(|(byte_idx, _)| byte_idx)
            .chain(std::iter::once(word.len()))
            .collect();

        let n_chars = char_boundaries.len() - 1; // == char_count
        let mut start_char: usize = 0;
        let mut is_bad = false;
        let checkpoint = out.len(); // restore point if the word fails

        'outer: while start_char < n_chars {
            let byte_start = char_boundaries[start_char];
            let mut end_char = n_chars;

            loop {
                let byte_end = char_boundaries[end_char];
                // Build the candidate token: bare for the first subword, ##-prefixed
                // for all subsequent subwords within the same whitespace-delimited word.
                let candidate: String = if start_char == 0 {
                    word[byte_start..byte_end].to_owned()
                } else {
                    format!(
                        "{}{}",
                        WORDPIECE_CONTINUATION_PREFIX,
                        &word[byte_start..byte_end]
                    )
                };

                if let Some(&id) = self.token_to_id.get(&candidate) {
                    out.push(id);
                    start_char = end_char;
                    continue 'outer;
                }

                // No match — shrink the span by one character from the right.
                if end_char == start_char + 1 {
                    // The single-character span also failed → whole word is bad.
                    is_bad = true;
                    break 'outer;
                }
                end_char -= 1;
            }
        }

        if is_bad {
            // Discard any partial tokens we pushed for this word and emit UNK.
            out.truncate(checkpoint);
            out.push(self.unk_id);
        }
    }

    /// Decode a sequence of token IDs back to a human-readable string.
    ///
    /// Continuation tokens (those whose vocabulary entry begins with `##`) have
    /// their prefix stripped and are appended without a separator.  Non-
    /// continuation tokens (the first subword of a word) are preceded by a
    /// space, except at the very beginning of the output.
    ///
    /// IDs that fall outside the vocabulary range are silently ignored.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in ids {
            let token = match self.id_to_token.get(id as usize) {
                Some(t) => t.as_str(),
                None => continue,
            };
            if let Some(cont) = token.strip_prefix(WORDPIECE_CONTINUATION_PREFIX) {
                // Continuation piece — no separator, strip the "##".
                result.push_str(cont);
            } else {
                // New word — add a space before it (unless this is the first token).
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(token);
            }
        }
        result
    }

    /// Look up the string representation of a single token ID.
    ///
    /// Returns `None` if `id` is out of range.
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    /// Return the total number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Return the UNK token ID configured for this vocabulary.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Return the maximum number of Unicode scalar values allowed per word.
    ///
    /// Words longer than this limit are directly emitted as UNK.
    pub fn max_input_chars_per_word(&self) -> usize {
        self.max_input_chars_per_word
    }
}

// ── Inline tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal BERT-style vocabulary for tests.
    ///
    /// IDs:
    ///  0 → [PAD]
    ///  1 → [UNK]
    ///  2 → [CLS]
    ///  3 → [SEP]
    ///  4 → hello
    ///  5 → world
    ///  6 → play
    ///  7 → ##ing
    ///  8 → ##s
    ///  9 → foo
    /// 10 → ##bar
    fn make_vocab() -> WordPieceVocab {
        let tokens: Vec<String> = vec![
            "[PAD]".into(),
            "[UNK]".into(),
            "[CLS]".into(),
            "[SEP]".into(),
            "hello".into(),
            "world".into(),
            "play".into(),
            "##ing".into(),
            "##s".into(),
            "foo".into(),
            "##bar".into(),
        ];
        WordPieceVocab::new(tokens, 1).expect("make_vocab must succeed")
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn error_empty_vocab() {
        let err = WordPieceVocab::new(vec![], 0).unwrap_err();
        assert_eq!(err, WordPieceError::EmptyVocab);
    }

    #[test]
    fn error_unk_out_of_range() {
        let err = WordPieceVocab::new(vec!["a".into()], 5).unwrap_err();
        assert!(
            matches!(
                err,
                WordPieceError::UnkOutOfRange {
                    unk_id: 5,
                    vocab_len: 1
                }
            ),
            "unexpected variant: {err:?}"
        );
    }

    #[test]
    fn error_duplicate_token() {
        let err = WordPieceVocab::new(vec!["a".into(), "a".into()], 0).unwrap_err();
        assert!(matches!(err, WordPieceError::DuplicateToken(ref t) if t == "a"));
    }

    #[test]
    fn vocab_size_matches() {
        let vocab = make_vocab();
        assert_eq!(vocab.vocab_size(), 11);
    }

    #[test]
    fn unk_id_accessor() {
        let vocab = make_vocab();
        assert_eq!(vocab.unk_id(), 1);
    }

    #[test]
    fn max_input_chars_default() {
        let vocab = make_vocab();
        assert_eq!(vocab.max_input_chars_per_word(), 100);
    }

    #[test]
    fn max_input_chars_builder() {
        let vocab = make_vocab().with_max_input_chars(42);
        assert_eq!(vocab.max_input_chars_per_word(), 42);
    }

    // ── Encoding ──────────────────────────────────────────────────────────────

    #[test]
    fn encode_empty_string() {
        let vocab = make_vocab();
        assert_eq!(vocab.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn encode_known_word() {
        let vocab = make_vocab();
        assert_eq!(vocab.encode("hello"), vec![4]);
    }

    #[test]
    fn encode_word_with_continuation() {
        let vocab = make_vocab();
        // "playing" → "play"(6) + "##ing"(7)
        assert_eq!(vocab.encode("playing"), vec![6, 7]);
    }

    #[test]
    fn encode_unknown_word_becomes_unk() {
        let vocab = make_vocab();
        // "xyz" is not in the vocabulary → single UNK
        assert_eq!(vocab.encode("xyz"), vec![1]);
    }

    #[test]
    fn encode_multi_word() {
        let vocab = make_vocab();
        assert_eq!(vocab.encode("hello world"), vec![4, 5]);
    }

    #[test]
    fn encode_word_too_long_becomes_unk() {
        let vocab = WordPieceVocab::new(vec!["[UNK]".into(), "a".into()], 0)
            .expect("vocab ok")
            .with_max_input_chars(3);
        // "aaaa" = 4 chars > 3 → UNK immediately
        assert_eq!(vocab.encode("aaaa"), vec![0]);
    }

    #[test]
    fn encode_at_exact_char_limit_is_not_unk() {
        // Exactly max_input_chars_per_word chars should still be attempted.
        let vocab = WordPieceVocab::new(vec!["[UNK]".into(), "aaa".into()], 0)
            .expect("vocab ok")
            .with_max_input_chars(3);
        // "aaa" = 3 chars == limit → segmentation attempted → token found
        assert_eq!(vocab.encode("aaa"), vec![1]);
    }

    #[test]
    fn foobar_segmentation() {
        let vocab = make_vocab();
        // "foobar" = "foo"(9) + "##bar"(10)
        assert_eq!(vocab.encode("foobar"), vec![9, 10]);
    }

    #[test]
    fn partial_bad_word_is_fully_replaced() {
        let vocab = make_vocab();
        // "fooxyz" — "foo" matches but "##xyz" doesn't; whole word → UNK
        assert_eq!(vocab.encode("fooxyz"), vec![1]);
    }

    #[test]
    fn encode_multibyte_unicode_word() {
        // "café" contains é (2 bytes) — must not split mid-codepoint.
        let tokens: Vec<String> = vec!["[UNK]".into(), "caf".into(), "##é".into()];
        let vocab = WordPieceVocab::new(tokens, 0).expect("vocab ok");
        let ids = vocab.encode("café");
        // "caf" + "##é" = [1, 2]
        assert_eq!(ids, vec![1, 2]);
    }

    // ── Decoding ──────────────────────────────────────────────────────────────

    #[test]
    fn decode_strips_continuation_prefix() {
        let vocab = make_vocab();
        // [play, ##ing] → "playing"
        assert_eq!(vocab.decode(&[6, 7]), "playing");
    }

    #[test]
    fn decode_multi_word() {
        let vocab = make_vocab();
        assert_eq!(vocab.decode(&[4, 5]), "hello world");
    }

    #[test]
    fn decode_empty_slice() {
        let vocab = make_vocab();
        assert_eq!(vocab.decode(&[] as &[u32]), "");
    }

    #[test]
    fn decode_unknown_ids_silently_ignored() {
        let vocab = make_vocab();
        // ID 999 is out of range — ignored, not an error
        assert_eq!(vocab.decode(&[4, 999, 5]), "hello world");
    }

    #[test]
    fn decode_token_known_id() {
        let vocab = make_vocab();
        assert_eq!(vocab.decode_token(4), Some("hello"));
        assert_eq!(vocab.decode_token(7), Some("##ing"));
    }

    #[test]
    fn decode_token_out_of_range() {
        let vocab = make_vocab();
        assert_eq!(vocab.decode_token(999), None);
    }

    // ── Error display ─────────────────────────────────────────────────────────

    #[test]
    fn display_empty_vocab_error() {
        let s = format!("{}", WordPieceError::EmptyVocab);
        assert!(s.contains("empty"));
    }

    #[test]
    fn display_unk_out_of_range_error() {
        let s = format!(
            "{}",
            WordPieceError::UnkOutOfRange {
                unk_id: 7,
                vocab_len: 3
            }
        );
        assert!(s.contains("7"));
        assert!(s.contains("3"));
    }

    #[test]
    fn display_duplicate_token_error() {
        let s = format!("{}", WordPieceError::DuplicateToken("hello".to_string()));
        assert!(s.contains("hello"));
    }
}
