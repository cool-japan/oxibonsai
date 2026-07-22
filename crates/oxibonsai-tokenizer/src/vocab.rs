//! Vocabulary management for the OxiBonsai tokenizer.
//!
//! Provides bidirectional mapping between token strings and integer IDs,
//! plus a separate special-token registry used for BOS/EOS/PAD/UNK handling.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{TokenizerError, TokenizerResult};

/// Bidirectional token ↔ ID vocabulary with special-token support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Vocabulary {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
    /// Tokens that must be atomically carved out of the input *before*
    /// pre-tokenization/model segmentation runs during `encode`, mirroring
    /// HuggingFace's `AddedVocabulary` semantics: **every** added token —
    /// whether flagged `special` in `tokenizer.json` or not — is protected
    /// from being shredded by generic segmentation (e.g. `<tool_call>`,
    /// `<|fim_prefix|>`).  This is a **superset** of `special_tokens`: every
    /// special token is also protected, but not every protected token is
    /// "special" (only `special_tokens` drives decode-skip / BOS-EOS-style
    /// semantics).
    protected_tokens: HashMap<String, u32>,
}

impl Vocabulary {
    /// Create an empty vocabulary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a regular (non-special) token ↔ ID pair.
    ///
    /// If the ID or token already exists, the mapping is silently overwritten.
    pub fn insert(&mut self, token: &str, id: u32) {
        self.token_to_id.insert(token.to_owned(), id);
        self.id_to_token.insert(id, token.to_owned());
    }

    /// Register a special token (e.g. `<s>`, `</s>`, `<unk>`, `<pad>`).
    ///
    /// Special tokens are also inserted into the main maps so they can be
    /// looked up by the standard `get_id` / `get_token` interface.
    pub fn add_special(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_owned(), id);
        self.protected_tokens.insert(token.to_owned(), id);
        self.insert(token, id);
    }

    /// Register a non-special "added token" that must still be atomically
    /// protected from pre-tokenization during `encode` (e.g. `<tool_call>`,
    /// `<|fim_prefix|>`), per HuggingFace `AddedVocabulary` semantics: every
    /// added token — special or not — is carved out before model
    /// segmentation runs; only the `special` flag governs default
    /// decode-skipping and similar normalization behaviour.
    ///
    /// Inserted into the main token↔id maps (so lookups work) and into the
    /// protected-token registry consulted by the encoder's leftmost-longest
    /// carve-out, but **not** into the narrower [`Self::special_tokens`]
    /// registry — so [`Self::is_special_token`] / [`Self::is_special_id`]
    /// keep reflecting only genuinely `special == true` tokens.
    pub fn add_protected(&mut self, token: &str, id: u32) {
        self.protected_tokens.insert(token.to_owned(), id);
        self.insert(token, id);
    }

    /// Look up the integer ID for a token string.
    ///
    /// Returns `None` if the token is not present in the vocabulary.
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Look up the token string for a given integer ID.
    ///
    /// Returns `None` if the ID is not present.
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Total number of tokens in the vocabulary (including special tokens).
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Returns `true` if the vocabulary contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Returns `true` if the given token string is registered as a special token.
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    /// Returns `true` if the given ID corresponds to a special token.
    pub fn is_special_id(&self, id: u32) -> bool {
        self.special_tokens.values().any(|&v| v == id)
    }

    /// Iterate over all (token, id) pairs (regular + special).
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.token_to_id.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// Iterate over the registered special tokens as `(token, id)` pairs.
    ///
    /// These are the tokens promoted via [`Self::add_special`] (e.g. the HF
    /// `added_tokens` marked `special == true`, such as `<|im_start|>`).  They
    /// are used by the encoder to carve out atomic special-token substrings
    /// before generic pre-tokenization.
    pub fn special_tokens(&self) -> impl Iterator<Item = (&str, u32)> {
        self.special_tokens.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// Iterate over ALL atomically-protected added tokens: the union of
    /// [`Self::special_tokens`] and any tokens registered via
    /// [`Self::add_protected`].  This is the set the encoder's
    /// leftmost-longest carve-out draws from, matching HuggingFace's
    /// `AddedVocabulary` (every added token is protected from
    /// pre-tokenization, not just `special == true` ones).
    pub fn protected_tokens(&self) -> impl Iterator<Item = (&str, u32)> {
        self.protected_tokens.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// Deserialize a vocabulary from a JSON object mapping token → id.
    ///
    /// The JSON must be a flat object: `{ "<token>": <id>, ... }`.
    /// Special tokens (those whose names start and end with `<` / `>`) are
    /// automatically promoted to the special-token registry.
    pub fn from_json(json: &str) -> TokenizerResult<Self> {
        let raw: HashMap<String, u32> =
            serde_json::from_str(json).map_err(|e| TokenizerError::InvalidJson(e.to_string()))?;

        if raw.is_empty() {
            return Err(TokenizerError::InvalidVocab(
                "vocabulary JSON must not be empty".into(),
            ));
        }

        let mut vocab = Self::new();
        for (token, id) in raw {
            // Heuristic: treat tokens that look like `<something>` as special.
            if token.starts_with('<') && token.ends_with('>') {
                vocab.add_special(&token, id);
            } else {
                vocab.insert(&token, id);
            }
        }
        Ok(vocab)
    }

    /// Serialize the vocabulary to a compact JSON object (token → id).
    ///
    /// The output is always sorted by token string for determinism.
    pub fn to_json(&self) -> String {
        // Collect and sort for deterministic output.
        let mut entries: Vec<(&str, u32)> = self.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        let mut out = String::from('{');
        for (i, (token, id)) in entries.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            // Cheap manual JSON escaping for token strings (printable ASCII assumed).
            out.push('"');
            for ch in token.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            out.push_str("\":");
            out.push_str(&id.to_string());
        }
        out.push('}');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let mut v = Vocabulary::new();
        v.insert("hello", 1);
        v.insert("world", 2);
        assert_eq!(v.get_id("hello"), Some(1));
        assert_eq!(v.get_id("world"), Some(2));
        assert_eq!(v.get_token(1), Some("hello"));
        assert_eq!(v.get_token(99), None);
    }

    #[test]
    fn special_tokens_are_found_in_main_maps() {
        let mut v = Vocabulary::new();
        v.add_special("<bos>", 0);
        assert_eq!(v.get_id("<bos>"), Some(0));
        assert!(v.is_special_token("<bos>"));
        assert!(v.is_special_id(0));
    }

    #[test]
    fn json_roundtrip() {
        let mut v = Vocabulary::new();
        v.insert("a", 3);
        v.insert("b", 4);
        v.add_special("<unk>", 0);
        let json = v.to_json();
        let v2 = Vocabulary::from_json(&json).expect("parse should succeed");
        assert_eq!(v2.get_id("a"), Some(3));
        assert_eq!(v2.get_id("b"), Some(4));
        assert_eq!(v2.get_id("<unk>"), Some(0));
    }

    #[test]
    fn empty_json_fails() {
        assert!(Vocabulary::from_json("{}").is_err());
    }

    #[test]
    fn invalid_json_fails() {
        assert!(Vocabulary::from_json("not json").is_err());
    }
}
