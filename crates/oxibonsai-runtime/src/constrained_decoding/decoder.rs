//! Shared token-text decoding index for text-oriented constraints.
//!
//! [`JsonConstraint`](super::json::JsonConstraint) and
//! [`RegexConstraint`](super::regex::RegexConstraint) operate on the **text**
//! that a token decodes to, not on the raw token id.  For any real subword
//! tokenizer the token id has no relationship to the character(s) it emits, so
//! a constraint that inspects the id directly is meaningless.
//!
//! [`TokenTextIndex`] precomputes, once at constraint-construction time, the
//! decoded character sequence for every token id in `0..vocab_size` via a
//! caller-supplied `decode_fn: Fn(u32) -> Option<String>`.  It mirrors the
//! first-byte-index acceleration used by
//! [`grammar::GrammarConstraint`](crate::grammar::GrammarConstraint): tokens are
//! grouped by their first decoded character so that a constraint can probe only
//! the tokens whose first character is currently valid.
//!
//! * `token_chars[id]` — the decoded characters for token `id` (empty when the
//!   decode function returned `None` or an empty string).
//! * `first_char_index[c]` — the token ids whose first decoded character is `c`.
//! * `empty_token_ids` — token ids that decode to no text (EOS / padding /
//!   special tokens).  These contribute no characters to the constrained stream
//!   and are only ever allowed when the constraint is already in a terminal
//!   (accepting) state.

use std::collections::HashMap;

/// Precomputed mapping from token id to decoded text, plus a first-character
/// index for fast candidate filtering.
///
/// Construct with [`TokenTextIndex::build`]; the decode function is invoked
/// exactly `vocab_size` times during construction and is **not** retained, so
/// later `allowed_tokens` / `advance` calls never decode again.
pub(crate) struct TokenTextIndex {
    /// Decoded character sequence for each token id in `0..vocab_size`.
    token_chars: Vec<Vec<char>>,
    /// Maps a first decoded character to the token ids that begin with it.
    first_char_index: HashMap<char, Vec<u32>>,
    /// Token ids that decode to no characters (special / EOS / padding tokens).
    empty_token_ids: Vec<u32>,
}

impl TokenTextIndex {
    /// Build the index by decoding every token id in `0..vocab_size`.
    ///
    /// A token that decodes to `None` or to the empty string is recorded in
    /// [`empty_token_ids`](Self::empty_token_ids); every other token is grouped
    /// by its first character.
    pub(crate) fn build(decode_fn: impl Fn(u32) -> Option<String>, vocab_size: usize) -> Self {
        let mut token_chars: Vec<Vec<char>> = Vec::with_capacity(vocab_size);
        let mut first_char_index: HashMap<char, Vec<u32>> = HashMap::new();
        let mut empty_token_ids: Vec<u32> = Vec::new();

        for id in 0..vocab_size as u32 {
            let chars: Vec<char> = match decode_fn(id) {
                Some(text) => text.chars().collect(),
                None => Vec::new(),
            };
            match chars.first() {
                Some(&first) => first_char_index.entry(first).or_default().push(id),
                None => empty_token_ids.push(id),
            }
            token_chars.push(chars);
        }

        Self {
            token_chars,
            first_char_index,
            empty_token_ids,
        }
    }

    /// Decoded characters for token `id`, or an empty slice if `id` is out of
    /// range for the precomputed index.
    pub(crate) fn token_chars(&self, id: u32) -> &[char] {
        self.token_chars
            .get(id as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Token ids that decode to no characters (special / EOS / padding tokens).
    pub(crate) fn empty_token_ids(&self) -> &[u32] {
        &self.empty_token_ids
    }

    /// Token ids whose first decoded character equals `c` (empty slice if none).
    pub(crate) fn ids_with_first_char(&self, c: char) -> &[u32] {
        self.first_char_index
            .get(&c)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Iterate over `(first_char, token_ids)` groups.
    ///
    /// Used by constraints whose next valid character set is not enumerable in
    /// advance (e.g. a regex `.` or a JSON string interior), where every group
    /// must be probed.
    pub(crate) fn first_char_groups(&self) -> impl Iterator<Item = (&char, &Vec<u32>)> {
        self.first_char_index.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy tokenizer: id 0 → "", 1 → "ab", 2 → "a", 3 → "cd", 4 → "" (EOS).
    fn toy_decode(id: u32) -> Option<String> {
        match id {
            0 => Some(String::new()),
            1 => Some("ab".to_string()),
            2 => Some("a".to_string()),
            3 => Some("cd".to_string()),
            4 => None,
            _ => None,
        }
    }

    #[test]
    fn build_groups_by_first_char() {
        let index = TokenTextIndex::build(toy_decode, 5);
        // Tokens 1 ("ab") and 2 ("a") both start with 'a'.
        let a_ids = index.ids_with_first_char('a');
        assert!(a_ids.contains(&1));
        assert!(a_ids.contains(&2));
        // Token 3 ("cd") starts with 'c'.
        assert_eq!(index.ids_with_first_char('c'), &[3]);
        // No token starts with 'z'.
        assert!(index.ids_with_first_char('z').is_empty());
    }

    #[test]
    fn empty_tokens_recorded() {
        let index = TokenTextIndex::build(toy_decode, 5);
        // Token 0 (empty string) and token 4 (None) both decode to no chars.
        let empties = index.empty_token_ids();
        assert!(empties.contains(&0));
        assert!(empties.contains(&4));
        assert!(!empties.contains(&1));
    }

    #[test]
    fn token_chars_returns_decoded_sequence() {
        let index = TokenTextIndex::build(toy_decode, 5);
        assert_eq!(index.token_chars(1), &['a', 'b']);
        assert_eq!(index.token_chars(3), &['c', 'd']);
        assert!(index.token_chars(0).is_empty());
        // Out-of-range id yields an empty slice, never panics.
        assert!(index.token_chars(999).is_empty());
    }

    #[test]
    fn first_char_groups_cover_all_non_empty_tokens() {
        let index = TokenTextIndex::build(toy_decode, 5);
        let mut seen: Vec<u32> = index
            .first_char_groups()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2, 3]);
    }

    #[test]
    fn multibyte_first_char_grouped_correctly() {
        // A token that decodes to a non-ASCII string must be grouped under its
        // first Unicode scalar value, not a byte.
        let index = TokenTextIndex::build(
            |id| match id {
                0 => Some("é-tude".to_string()),
                _ => None,
            },
            1,
        );
        assert_eq!(index.ids_with_first_char('é'), &[0]);
        assert_eq!(index.token_chars(0), &['é', '-', 't', 'u', 'd', 'e']);
    }
}
