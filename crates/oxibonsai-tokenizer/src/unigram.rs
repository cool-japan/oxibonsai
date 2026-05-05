//! Viterbi-based Unigram tokenizer for OxiBonsai.
//!
//! This module implements the Unigram language model tokenization algorithm as
//! used by SentencePiece and HuggingFace models (e.g. BERT-Japanese, T5,
//! mBERT, XLNet, ...).  Unlike BPE, Unigram selects the segmentation that
//! maximises the sum of token log-probabilities under a trained vocabulary.
//!
//! The core algorithm is the **Viterbi forward-pass** over byte positions,
//! followed by a simple backtracking step to recover the best token sequence.
//!
//! ## UTF-8 safety
//!
//! All candidate spans are checked with `str::is_char_boundary` before being
//! looked up in the vocabulary, so multibyte codepoints (e.g. CJK, emoji) are
//! never split in the middle.
//!
//! ## UNK fallback
//!
//! When no vocabulary token covers position `i`, the algorithm consumes exactly
//! **one byte** and emits `unk_id` with a large negative penalty (`-1e6`).
//! This guarantees that `encode` always terminates for any UTF-8 input.

use std::collections::HashMap;

// ── UnigramError ─────────────────────────────────────────────────────────────

/// Errors that can occur when building a [`UnigramVocab`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnigramError {
    /// The vocabulary entry list was empty.
    EmptyVocab,
    /// The `unk_id` given to [`UnigramVocab::new`] is beyond the vocabulary
    /// size.
    UnkOutOfRange {
        /// The unk_id that was provided.
        unk_id: u32,
        /// The actual number of entries in the vocabulary.
        vocab_len: usize,
    },
    /// Two or more entries share the same token string.
    DuplicateToken(String),
}

impl std::fmt::Display for UnigramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyVocab => write!(f, "unigram vocabulary must not be empty"),
            Self::UnkOutOfRange { unk_id, vocab_len } => write!(
                f,
                "unk_id {unk_id} is out of range for vocabulary of size {vocab_len}"
            ),
            Self::DuplicateToken(tok) => {
                write!(f, "duplicate token in unigram vocabulary: {tok:?}")
            }
        }
    }
}

impl std::error::Error for UnigramError {}

// ── UnigramVocab ─────────────────────────────────────────────────────────────

/// Log-probability vocabulary for Unigram language-model tokenization.
///
/// Tokens are indexed by a contiguous `u32` ID starting at `0`.  The ID space
/// is determined entirely by the order of the `entries` slice passed to
/// [`UnigramVocab::new`].
#[derive(Debug)]
pub struct UnigramVocab {
    /// `entries[id]` = `(token_string, log_prob)`.
    entries: Vec<(String, f64)>,
    /// Fast token-string → ID lookup.
    token_to_id: HashMap<String, u32>,
    /// `token_scores[id]` = `log_prob` — kept flat for cache-friendly Viterbi.
    token_scores: Vec<f64>,
    /// Token ID that represents an unknown / out-of-vocabulary segment.
    unk_id: u32,
    /// Maximum byte length of any token string (≥ 1).  Used to bound the inner
    /// loop of the Viterbi forward pass.
    max_token_byte_len: usize,
}

impl UnigramVocab {
    /// Construct a [`UnigramVocab`] from an ordered list of `(token, log_prob)`
    /// pairs.
    ///
    /// The position of each pair in `entries` determines its token ID: the
    /// first pair has ID `0`, the second has ID `1`, and so on.
    ///
    /// # Errors
    ///
    /// Returns [`UnigramError::EmptyVocab`] if `entries` is empty.
    /// Returns [`UnigramError::UnkOutOfRange`] if `unk_id >= entries.len()`.
    /// Returns [`UnigramError::DuplicateToken`] if any token string appears
    /// more than once.
    pub fn new(entries: Vec<(String, f64)>, unk_id: u32) -> Result<Self, UnigramError> {
        if entries.is_empty() {
            return Err(UnigramError::EmptyVocab);
        }

        if unk_id as usize >= entries.len() {
            return Err(UnigramError::UnkOutOfRange {
                unk_id,
                vocab_len: entries.len(),
            });
        }

        // Duplicate-check and build the token→id map.
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(entries.len());
        let mut token_scores: Vec<f64> = Vec::with_capacity(entries.len());
        let mut max_token_byte_len: usize = 1;

        for (idx, (token, score)) in entries.iter().enumerate() {
            if token_to_id.insert(token.clone(), idx as u32).is_some() {
                return Err(UnigramError::DuplicateToken(token.clone()));
            }
            token_scores.push(*score);
            let byte_len = token.len();
            if byte_len > max_token_byte_len {
                max_token_byte_len = byte_len;
            }
        }

        Ok(Self {
            entries,
            token_to_id,
            token_scores,
            unk_id,
            max_token_byte_len,
        })
    }

    /// Viterbi best-path segmentation.
    ///
    /// Returns the token ID sequence that maximises the sum of log-probabilities
    /// over all possible segmentations of `text`.  When no vocabulary token
    /// covers the current position, a single byte is consumed with ID
    /// `unk_id` and a large negative penalty.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        const UNK_PENALTY: f64 = -1e6;

        let n = text.len(); // byte length
        if n == 0 {
            return Vec::new();
        }

        // best_score[i] = max log-prob to reach byte position i from position 0.
        let mut best_score: Vec<f64> = vec![f64::NEG_INFINITY; n + 1];
        // best_back[i] = Some((token_id, token_byte_len)) for backtracking.
        let mut best_back: Vec<Option<(u32, usize)>> = vec![None; n + 1];
        best_score[0] = 0.0;

        for i in 0..n {
            if best_score[i] == f64::NEG_INFINITY {
                continue;
            }

            // If `i` is not a UTF-8 character boundary we cannot form any
            // valid token starting here; propagate the UNK single-byte step
            // to carry the path forward one more byte.
            if !text.is_char_boundary(i) {
                if i < n {
                    let cand = best_score[i] + UNK_PENALTY;
                    if cand > best_score[i + 1] {
                        best_score[i + 1] = cand;
                        best_back[i + 1] = Some((self.unk_id, 1));
                    }
                }
                continue;
            }

            let mut found_any = false;
            let max_len = self.max_token_byte_len.min(n - i);

            for len in 1..=max_len {
                // Guard against a span ending mid-codepoint.
                if !text.is_char_boundary(i + len) {
                    continue;
                }
                // Safety: both `i` and `i + len` are char boundaries, so this
                // slice is a valid UTF-8 string.
                let substr = &text[i..i + len];
                if let Some(&tok_id) = self.token_to_id.get(substr) {
                    let score = self.token_scores[tok_id as usize];
                    let cand = best_score[i] + score;
                    if cand > best_score[i + len] {
                        best_score[i + len] = cand;
                        best_back[i + len] = Some((tok_id, len));
                        found_any = true;
                    }
                }
            }

            // UNK fallback: consume exactly one byte with a heavy penalty so
            // that the algorithm always makes forward progress.
            if (!found_any || best_score[i + 1] == f64::NEG_INFINITY) && i < n {
                let cand = best_score[i] + UNK_PENALTY;
                if cand > best_score[i + 1] {
                    best_score[i + 1] = cand;
                    best_back[i + 1] = Some((self.unk_id, 1));
                }
            }
        }

        // Backtrack from position n to position 0.
        let mut tokens: Vec<u32> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            match best_back[pos] {
                Some((tok_id, len)) => {
                    tokens.push(tok_id);
                    pos -= len;
                }
                None => {
                    // Should not be reachable due to the UNK fallback above,
                    // but handle gracefully to avoid a panic.
                    break;
                }
            }
        }
        tokens.reverse();
        tokens
    }

    /// Return the total number of entries in the vocabulary.
    pub fn token_count(&self) -> usize {
        self.entries.len()
    }

    /// Decode a single token ID back to its string slice, or `None` if the ID
    /// is out of range.
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.entries.get(id as usize).map(|(s, _)| s.as_str())
    }

    /// Decode a sequence of token IDs back to a `String` by concatenating the
    /// corresponding token strings.
    ///
    /// Unknown IDs (beyond the vocabulary) are silently replaced with the
    /// Unicode replacement character `\u{FFFD}`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            match self.decode_token(id) {
                Some(s) => out.push_str(s),
                None => out.push('\u{FFFD}'),
            }
        }
        out
    }

    /// Return the UNK token ID configured for this vocabulary.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }
}

// ── Inline tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab(entries: &[(&str, f64)], unk_id: u32) -> UnigramVocab {
        UnigramVocab::new(
            entries
                .iter()
                .map(|(s, p)| (s.to_string(), *p))
                .collect(),
            unk_id,
        )
        .unwrap()
    }

    // ── Construction error paths ──────────────────────────────────────────────

    #[test]
    fn empty_vocab_errors() {
        let err = UnigramVocab::new(vec![], 0).unwrap_err();
        assert_eq!(err, UnigramError::EmptyVocab);
    }

    #[test]
    fn unk_out_of_range_errors() {
        let entries = vec![("<unk>".to_string(), 0.0), ("a".to_string(), -1.0)];
        let err = UnigramVocab::new(entries, 5).unwrap_err();
        assert!(matches!(
            err,
            UnigramError::UnkOutOfRange {
                unk_id: 5,
                vocab_len: 2
            }
        ));
    }

    #[test]
    fn duplicate_token_errors() {
        let entries = vec![
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.5),
            ("a".to_string(), -2.0), // duplicate
        ];
        let err = UnigramVocab::new(entries, 0).unwrap_err();
        match err {
            UnigramError::DuplicateToken(tok) => assert_eq!(tok, "a"),
            other => panic!("expected DuplicateToken, got {other:?}"),
        }
    }

    // ── Encoding paths ────────────────────────────────────────────────────────

    #[test]
    fn single_token_encodes() {
        // Vocabulary: id=0 "<unk>", id=1 "hello"
        let vocab = make_vocab(&[("<unk>", 0.0), ("hello", -1.0)], 0);
        let ids = vocab.encode("hello");
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn ambiguous_prefers_higher_score() {
        // "ab" score -1.0 vs "a"(-2.0) + "b"(-2.0) = -4.0.  "ab" should win.
        let vocab = make_vocab(&[("<unk>", 0.0), ("a", -2.0), ("b", -2.0), ("ab", -1.0)], 0);
        let ids = vocab.encode("ab");
        // Expect single token "ab" (id=3).
        assert_eq!(ids, vec![3]);
    }

    #[test]
    fn lower_score_path_loses_to_higher() {
        // "a" + "b" each score -0.5 → -1.0 total; "ab" scores -1.5.
        // "a"+"b" should win.
        let vocab = make_vocab(&[("<unk>", 0.0), ("a", -0.5), ("b", -0.5), ("ab", -1.5)], 0);
        let ids = vocab.encode("ab");
        // Expect two tokens: "a"(id=1) + "b"(id=2).
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn multibyte_utf8_boundary_respected() {
        // "café" is 5 bytes: 'c'=1, 'a'=1, 'f'=1, 'é'=2 (U+00E9 = 0xC3 0xA9).
        // Vocabulary knows "é" but not the individual bytes — UNK fallback must
        // not split the 2-byte sequence of 'é'.
        let vocab = make_vocab(
            &[("<unk>", 0.0), ("c", -1.0), ("a", -1.0), ("f", -1.0), ("é", -1.0)],
            0,
        );
        let ids = vocab.encode("café");
        // Should not panic and should produce a non-empty result.
        assert!(!ids.is_empty());
        // 'é' must appear as its own token (id=4), not split across UNK tokens.
        assert!(ids.contains(&4));
    }

    #[test]
    fn unk_fallback_for_unknown_byte() {
        // Only "<unk>" and "a" are in the vocabulary.  'z' has no match.
        let vocab = make_vocab(&[("<unk>", 0.0), ("a", -1.0)], 0);
        let ids = vocab.encode("z");
        // 'z' is unknown → should be emitted as unk_id (0).
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn decode_roundtrip() {
        let vocab = make_vocab(
            &[("<unk>", 0.0), ("hello", -1.0), (" ", -0.5), ("world", -1.0)],
            0,
        );
        let text = "hello world";
        let ids = vocab.encode(text);
        let decoded = vocab.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn empty_string_encodes_to_empty() {
        let vocab = make_vocab(&[("<unk>", 0.0), ("a", -1.0)], 0);
        let ids = vocab.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn decode_unknown_id_produces_replacement_char() {
        let vocab = make_vocab(&[("<unk>", 0.0), ("a", -1.0)], 0);
        let decoded = vocab.decode(&[999]);
        assert_eq!(decoded, "\u{FFFD}");
    }

    #[test]
    fn decode_token_out_of_range_returns_none() {
        let vocab = make_vocab(&[("<unk>", 0.0), ("a", -1.0)], 0);
        assert!(vocab.decode_token(999).is_none());
    }

    #[test]
    fn token_count_matches_entries() {
        let vocab = make_vocab(&[("<unk>", 0.0), ("a", -1.0), ("b", -2.0)], 0);
        assert_eq!(vocab.token_count(), 3);
    }

    #[test]
    fn error_display_empty_vocab() {
        let err = UnigramError::EmptyVocab;
        let s = format!("{err}");
        assert!(s.contains("empty"));
    }

    #[test]
    fn error_display_unk_out_of_range() {
        let err = UnigramError::UnkOutOfRange {
            unk_id: 10,
            vocab_len: 3,
        };
        let s = format!("{err}");
        assert!(s.contains("10"));
        assert!(s.contains("3"));
    }

    #[test]
    fn error_display_duplicate_token() {
        let err = UnigramError::DuplicateToken("foo".to_string());
        let s = format!("{err}");
        assert!(s.contains("foo"));
    }
}
