//! HuggingFace `tokenizer.json` format parser (BPE and Unigram models).
//!
//! This module provides a faithful, dependency-free parser for the tokenizer
//! JSON format emitted by the HuggingFace `tokenizers` library.  It supports:
//!
//! - BPE `model` with `vocab` and `merges` (both string `"a b"` form and
//!   array `["a","b"]` form)
//! - Unigram `model` with `vocab` as `[[token, score], ...]` and `unk_id`
//! - `added_tokens` (marked special if `special == true`)
//! - `pre_tokenizer` type detection (GPT-2 ByteLevel vs. Whitespace)
//! - `decoder` type detection (ByteLevel is the default for modern models)
//!
//! The result of parsing is an [`HfTokenizerJson`] struct that can be
//! converted into a fully-configured [`crate::OxiTokenizer`] via
//! [`HfTokenizerJson::into_tokenizer`].
//!
//! ## GPT-2 bytes-to-unicode mapping
//!
//! Modern BPE tokenizers (Qwen3, Llama-3, Mistral, ...) encode raw bytes as
//! visible Unicode characters so that whitespace and control bytes can
//! participate in merges.  The mapping is:
//!
//! - Bytes `0x21..=0x7E` (`!` … `~`) map to themselves.
//! - Bytes `0xA1..=0xAC` and `0xAE..=0xFF` map to themselves.
//! - The remaining 68 bytes (`0x00..=0x20`, `0x7F..=0xA0`, `0xAD`) are
//!   remapped to Unicode code points `0x100..=0x143`.
//!
//! See <https://github.com/openai/gpt-2/blob/master/src/encoder.py> for the
//! canonical reference.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    bpe::BpeMerges,
    error::{TokenizerError, TokenizerResult},
    tokenizer::{OxiTokenizer, TokenizerConfig},
    vocab::Vocabulary,
};

// ── Bytes-to-unicode map ─────────────────────────────────────────────────────

/// Build the canonical GPT-2 bytes-to-unicode table.
///
/// Returns an array indexed by byte value containing the Unicode code point
/// for that byte.  The inverse map is built by [`bytes_to_unicode_inverse`].
fn build_bytes_to_unicode() -> [char; 256] {
    // Bytes that map to themselves.
    let mut printable: Vec<u8> = Vec::with_capacity(188);
    for b in 0x21u8..=0x7Eu8 {
        printable.push(b);
    }
    for b in 0xA1u8..=0xACu8 {
        printable.push(b);
    }
    for b in 0xAEu8..=0xFFu8 {
        printable.push(b);
    }

    // Remaining bytes get remapped to code points 0x100 and onward.
    let mut table: [char; 256] = ['\0'; 256];
    let mut n: u32 = 0;
    for b in 0u16..=255u16 {
        if printable.contains(&(b as u8)) {
            table[b as usize] = char::from_u32(b as u32).unwrap_or('\u{FFFD}');
        } else {
            let cp = 0x100u32 + n;
            table[b as usize] = char::from_u32(cp).unwrap_or('\u{FFFD}');
            n += 1;
        }
    }
    table
}

/// Return the public 256-entry GPT-2 bytes-to-unicode mapping.
///
/// The returned array is indexed by byte value.  Element `i` is the Unicode
/// character used to represent byte `i` in the HF ByteLevel pre-tokenizer.
pub fn bytes_to_unicode_map() -> [char; 256] {
    build_bytes_to_unicode()
}

/// Return the Unicode character for the given byte, per the GPT-2 map.
pub fn byte_to_unicode(b: u8) -> char {
    bytes_to_unicode_map()[b as usize]
}

/// Inverse of [`byte_to_unicode`]: return the byte value for a Unicode char,
/// or `None` if `ch` is not part of the 256-entry table.
pub fn unicode_to_byte(ch: char) -> Option<u8> {
    let table = bytes_to_unicode_map();
    for (idx, &c) in table.iter().enumerate() {
        if c == ch {
            return Some(idx as u8);
        }
    }
    None
}

/// Build a `HashMap<char, u8>` inverse map (faster for long decode paths).
pub fn bytes_to_unicode_inverse() -> HashMap<char, u8> {
    let table = build_bytes_to_unicode();
    let mut out = HashMap::with_capacity(256);
    for (idx, &ch) in table.iter().enumerate() {
        out.insert(ch, idx as u8);
    }
    out
}

/// Apply the GPT-2 bytes-to-unicode map to a UTF-8 string, producing the
/// pre-tokenizer output used by `tokenizer.json` ByteLevel models.
pub fn bytes_to_unicode_string(s: &str) -> String {
    let table = bytes_to_unicode_map();
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        out.push(table[*b as usize]);
    }
    out
}

// ── HfModelType ──────────────────────────────────────────────────────────────

/// Model type discriminator parsed from `model.type` in a HuggingFace
/// `tokenizer.json` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HfModelType {
    /// BPE (Byte-Pair Encoding) model — uses `vocab` (object) + `merges`.
    Bpe,
    /// Unigram language model — uses `vocab` as `[[token, score], ...]` and
    /// `unk_id`.
    Unigram,
    /// Any other (currently unrecognised) model type.  Stored for diagnostics.
    Other(String),
}

impl HfModelType {
    fn from_str(s: &str) -> Self {
        match s {
            "BPE" => Self::Bpe,
            "Unigram" => Self::Unigram,
            other => Self::Other(other.to_owned()),
        }
    }
}

// ── HfTokenizerJson ──────────────────────────────────────────────────────────

/// Parsed representation of a HuggingFace `tokenizer.json` file.
///
/// All text-expressing fields are plain Rust strings — no base64, no escaping
/// tricks beyond what `serde_json` already handles.
#[derive(Debug, Clone)]
pub struct HfTokenizerJson {
    /// Discriminator for the model type declared in `model.type`.
    pub model_type: HfModelType,
    /// Token string → integer ID (includes added/special tokens).
    ///
    /// For BPE models this is populated from `model.vocab` (an object).
    /// For Unigram models this is derived from the ordered `model.vocab` array
    /// so that decode (ID → string) continues to work via the standard path.
    pub vocab: HashMap<String, u32>,
    /// Ordered list of BPE merge pairs `(left, right)`.
    ///
    /// Order defines priority — first pair is highest priority.
    /// Always empty for Unigram models.
    pub merges: Vec<(String, String)>,
    /// For Unigram models: ordered `(token, log_prob)` pairs from `model.vocab`.
    ///
    /// The position of each pair in the vector determines its token ID.
    /// `None` for BPE models.
    pub unigram_vocab: Option<Vec<(String, f64)>>,
    /// For Unigram models: the UNK token ID from `model.unk_id`.
    ///
    /// `None` for BPE models.
    pub unigram_unk_id: Option<u32>,
    /// Tokens flagged as `special == true` in `added_tokens`.
    pub special_tokens: HashMap<String, u32>,
    /// BOS token string, if present.
    pub bos_token: Option<String>,
    /// EOS token string, if present.
    pub eos_token: Option<String>,
    /// UNK token string, if present.
    pub unk_token: Option<String>,
    /// PAD token string, if present.
    pub pad_token: Option<String>,
    /// `true` if the tokenizer uses the GPT-2 ByteLevel pre-tokenizer.
    pub byte_level: bool,
}

impl HfTokenizerJson {
    /// Parse a HuggingFace `tokenizer.json` document from an in-memory string.
    pub fn parse(json: &str) -> TokenizerResult<Self> {
        let root: Value = serde_json::from_str(json)
            .map_err(|e| TokenizerError::HfFormat(format!("invalid JSON: {e}")))?;

        let model = root
            .get("model")
            .ok_or_else(|| TokenizerError::HfFormat("missing `model` field".to_owned()))?;

        // ── 0. model type ────────────────────────────────────────────────────
        let model_type = model
            .get("type")
            .and_then(Value::as_str)
            .map(HfModelType::from_str)
            .unwrap_or(HfModelType::Bpe);

        // ── 1. vocab + merges (dispatched by model type) ─────────────────────
        let (mut vocab, merges, unigram_vocab, unigram_unk_id) = match &model_type {
            HfModelType::Unigram => {
                let (v, m, uv, uid) = parse_unigram_model(model)?;
                (v, m, uv, uid)
            }
            HfModelType::Bpe | HfModelType::Other(_) => {
                let (v, m) = parse_bpe_model(model)?;
                (v, m, None, None)
            }
        };

        // ── 2. added_tokens → special_tokens ────────────────────────────────
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        if let Some(added) = root.get("added_tokens").and_then(Value::as_array) {
            for token_obj in added {
                let content = token_obj
                    .get("content")
                    .and_then(Value::as_str)
                    .map(|s| s.to_owned());
                let id = token_obj
                    .get("id")
                    .and_then(Value::as_u64)
                    .map(|n| n as u32);
                let is_special = token_obj
                    .get("special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if let (Some(content), Some(id)) = (content, id) {
                    // Even non-special added tokens go into the vocab if
                    // missing, so that encode/decode can see them.
                    vocab.entry(content.clone()).or_insert(id);
                    if is_special {
                        special_tokens.insert(content, id);
                    }
                }
            }
        }

        // ── 3. BOS/EOS/UNK/PAD hints ────────────────────────────────────────
        let bos_token = extract_special_token(&root, "bos_token");
        let eos_token = extract_special_token(&root, "eos_token");
        let unk_token = extract_special_token(&root, "unk_token").or_else(|| {
            model
                .get("unk_token")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
        let pad_token = extract_special_token(&root, "pad_token");

        // ── 4. ByteLevel detection ───────────────────────────────────────────
        let byte_level = detect_byte_level(&root);

        Ok(Self {
            model_type,
            vocab,
            merges,
            unigram_vocab,
            unigram_unk_id,
            special_tokens,
            bos_token,
            eos_token,
            unk_token,
            pad_token,
            byte_level,
        })
    }

    /// Convert the parsed document into a ready-to-use [`OxiTokenizer`].
    ///
    /// For `"BPE"` (and unrecognised) model types the existing BPE path is
    /// used.  For `"Unigram"` a [`crate::unigram::UnigramVocab`] is built and
    /// [`OxiTokenizer::with_unigram`] is called.
    pub fn into_tokenizer(self) -> TokenizerResult<OxiTokenizer> {
        // Build the shared vocabulary used for decode in all paths.
        let mut vocabulary = Vocabulary::new();
        for (token, id) in &self.vocab {
            if self.special_tokens.contains_key(token) {
                vocabulary.add_special(token, *id);
            } else {
                vocabulary.insert(token, *id);
            }
        }

        // Resolve BOS/EOS/UNK/PAD IDs.
        let bos_id = self
            .bos_token
            .as_ref()
            .and_then(|t| self.vocab.get(t).copied());
        let eos_id = self
            .eos_token
            .as_ref()
            .and_then(|t| self.vocab.get(t).copied());
        let unk_id_from_token = self
            .unk_token
            .as_ref()
            .and_then(|t| self.vocab.get(t).copied());
        let pad_id = self
            .pad_token
            .as_ref()
            .and_then(|t| self.vocab.get(t).copied());

        let mut config = TokenizerConfig {
            byte_level_decode: self.byte_level,
            ..Default::default()
        };
        if let Some(id) = bos_id {
            config.bos_token_id = id;
        }
        if let Some(id) = eos_id {
            config.eos_token_id = id;
        }
        if let Some(id) = unk_id_from_token {
            config.unk_token_id = id;
        }
        if let Some(id) = pad_id {
            config.pad_token_id = id;
        }

        match self.model_type {
            HfModelType::Bpe | HfModelType::Other(_) => {
                // Build the merge table.  For each (a, b) in priority order
                // we need the merged token's ID — by HF convention this is
                // `vocab[a ++ b]`.
                let mut merges = BpeMerges::new();
                for (a, b) in &self.merges {
                    let merged = format!("{a}{b}");
                    let merged_id = match self.vocab.get(&merged) {
                        Some(&id) => id,
                        None => {
                            // Some HF dumps omit the final merged token from
                            // the vocab (e.g. when the merge never actually
                            // applies in the training corpus).  Skip silently.
                            continue;
                        }
                    };
                    merges.add_merge(a, b, merged_id);
                }
                Ok(OxiTokenizer::new(vocabulary, merges, config))
            }
            HfModelType::Unigram => {
                let entries = self.unigram_vocab.ok_or_else(|| {
                    TokenizerError::HfFormat(
                        "Unigram model requires `model.vocab` array".to_owned(),
                    )
                })?;
                // Prefer unk_id from `model.unk_id`; fall back to the id
                // resolved from the `unk_token` string.
                let effective_unk_id = self
                    .unigram_unk_id
                    .unwrap_or(config.unk_token_id);
                let unigram_vocab = crate::unigram::UnigramVocab::new(entries, effective_unk_id)
                    .map_err(|e| {
                        TokenizerError::HfFormat(format!("invalid Unigram vocab: {e}"))
                    })?;
                Ok(OxiTokenizer::with_unigram(vocabulary, unigram_vocab, config))
            }
        }
    }
}

// ── Private parse helpers ────────────────────────────────────────────────────

/// Parse a BPE `model` section: returns `(vocab_map, merges)`.
#[allow(clippy::type_complexity)]
fn parse_bpe_model(
    model: &Value,
) -> TokenizerResult<(HashMap<String, u32>, Vec<(String, String)>)> {
    // vocab: required, must be an object.
    let vocab_val = model
        .get("vocab")
        .ok_or_else(|| TokenizerError::HfFormat("missing `model.vocab` field".to_owned()))?;
    let mut vocab: HashMap<String, u32> = HashMap::new();
    match vocab_val {
        Value::Object(map) => {
            for (token, id_val) in map {
                let id = id_val.as_u64().ok_or_else(|| {
                    TokenizerError::HfFormat(format!(
                        "vocab entry {token:?} has non-integer id"
                    ))
                })? as u32;
                vocab.insert(token.clone(), id);
            }
        }
        _ => {
            return Err(TokenizerError::HfFormat(
                "`model.vocab` must be an object".to_owned(),
            ));
        }
    }

    // merges: required, must be an array.
    // HF supports two shapes:
    //   "merges": ["a b", "c d"]
    //   "merges": [["a","b"], ["c","d"]]
    let merges_val = model
        .get("merges")
        .ok_or_else(|| TokenizerError::HfFormat("missing `model.merges` field".to_owned()))?;
    let mut merges: Vec<(String, String)> = Vec::new();
    match merges_val {
        Value::Array(list) => {
            for (idx, entry) in list.iter().enumerate() {
                let pair = parse_merge_entry(entry).ok_or_else(|| {
                    TokenizerError::HfFormat(format!("malformed merge entry #{idx}: {entry:?}"))
                })?;
                merges.push(pair);
            }
        }
        _ => {
            return Err(TokenizerError::HfFormat(
                "`model.merges` must be an array".to_owned(),
            ));
        }
    }

    Ok((vocab, merges))
}

/// Parse a Unigram `model` section.
///
/// Returns `(vocab_map, merges=[], unigram_entries, unk_id)`.
/// `vocab_map` maps token → ID derived from the position in `model.vocab` so
/// that decode (ID → string) continues to work through the standard
/// [`Vocabulary`] path.
#[allow(clippy::type_complexity)]
fn parse_unigram_model(
    model: &Value,
) -> TokenizerResult<(
    HashMap<String, u32>,
    Vec<(String, String)>,
    Option<Vec<(String, f64)>>,
    Option<u32>,
)> {
    let vocab_val = model
        .get("vocab")
        .ok_or_else(|| TokenizerError::HfFormat("missing `model.vocab` field".to_owned()))?;

    let arr = vocab_val.as_array().ok_or_else(|| {
        TokenizerError::HfFormat(
            "Unigram `model.vocab` must be an array of [token, score] pairs".to_owned(),
        )
    })?;

    let mut entries: Vec<(String, f64)> = Vec::with_capacity(arr.len());
    let mut vocab_map: HashMap<String, u32> = HashMap::with_capacity(arr.len());

    for (idx, item) in arr.iter().enumerate() {
        let pair = item.as_array().ok_or_else(|| {
            TokenizerError::HfFormat(format!(
                "Unigram vocab entry #{idx} must be a [token, score] array"
            ))
        })?;
        if pair.len() != 2 {
            return Err(TokenizerError::HfFormat(format!(
                "Unigram vocab entry #{idx} must have exactly 2 elements, got {}",
                pair.len()
            )));
        }
        let token = pair[0].as_str().ok_or_else(|| {
            TokenizerError::HfFormat(format!(
                "Unigram vocab entry #{idx}: first element must be a string"
            ))
        })?;
        let score = pair[1].as_f64().ok_or_else(|| {
            TokenizerError::HfFormat(format!(
                "Unigram vocab entry #{idx}: second element must be a number"
            ))
        })?;
        vocab_map.insert(token.to_owned(), idx as u32);
        entries.push((token.to_owned(), score));
    }

    let unk_id = model
        .get("unk_id")
        .and_then(Value::as_u64)
        .map(|n| n as u32);

    Ok((vocab_map, vec![], Some(entries), unk_id))
}

/// Parse a single `model.merges` entry into a `(left, right)` pair.
fn parse_merge_entry(entry: &Value) -> Option<(String, String)> {
    match entry {
        // Form 1: "a b"
        Value::String(s) => {
            let mut parts = s.splitn(2, ' ');
            let a = parts.next()?.to_owned();
            let b = parts.next()?.to_owned();
            Some((a, b))
        }
        // Form 2: ["a", "b"]
        Value::Array(arr) if arr.len() == 2 => {
            let a = arr[0].as_str()?.to_owned();
            let b = arr[1].as_str()?.to_owned();
            Some((a, b))
        }
        _ => None,
    }
}

/// Extract a special-token string from multiple possible top-level locations.
fn extract_special_token(root: &Value, key: &str) -> Option<String> {
    // First look in the top-level `added_tokens_decoder`-style block some
    // models use.  Then check the very top level.
    if let Some(v) = root.get(key) {
        if let Some(s) = v.as_str() {
            return Some(s.to_owned());
        }
        if let Some(inner) = v.get("content").and_then(Value::as_str) {
            return Some(inner.to_owned());
        }
    }
    None
}

/// Return `true` if the tokenizer pre-tokenizer or decoder is ByteLevel.
fn detect_byte_level(root: &Value) -> bool {
    let has_bl = |field: &str| -> bool {
        match root.get(field) {
            Some(Value::Object(map)) => map
                .get("type")
                .and_then(Value::as_str)
                .map(|t| t == "ByteLevel")
                .unwrap_or(false),
            Some(Value::Array(list)) => list.iter().any(|entry| {
                entry
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|t| t == "ByteLevel")
                    .unwrap_or(false)
            }),
            _ => false,
        }
    };
    has_bl("pre_tokenizer") || has_bl("decoder")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_is_256_entries() {
        let table = bytes_to_unicode_map();
        // All 256 entries must be populated and distinct.
        let mut seen = std::collections::HashSet::new();
        for &ch in table.iter() {
            assert_ne!(ch, '\0', "map entry must not be NUL");
            assert!(seen.insert(ch), "map entries must be distinct");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn map_printable_ascii_passthrough() {
        // All bytes in 0x21..=0x7E should map to themselves.
        for b in 0x21u8..=0x7Eu8 {
            assert_eq!(byte_to_unicode(b), char::from(b));
        }
    }

    #[test]
    fn map_latin1_passthrough() {
        for b in 0xA1u8..=0xACu8 {
            assert_eq!(byte_to_unicode(b), char::from(b));
        }
        for b in 0xAEu8..=0xFFu8 {
            assert_eq!(byte_to_unicode(b), char::from(b));
        }
    }

    #[test]
    fn map_space_remapped() {
        // Space (0x20) becomes Ġ = U+0120.
        assert_eq!(byte_to_unicode(0x20), '\u{0120}');
    }

    #[test]
    fn map_newline_remapped() {
        // LF (0x0A) becomes Ċ = U+010A.
        assert_eq!(byte_to_unicode(0x0A), '\u{010A}');
    }

    #[test]
    fn map_inverse_roundtrip() {
        for b in 0u16..=255u16 {
            let b = b as u8;
            let ch = byte_to_unicode(b);
            assert_eq!(
                unicode_to_byte(ch),
                Some(b),
                "roundtrip failed for byte {b:#x}"
            );
        }
    }

    #[test]
    fn bytes_to_unicode_string_basic() {
        let out = bytes_to_unicode_string(" hello");
        // Leading space becomes Ġ, rest are ASCII passthrough.
        assert!(out.starts_with('\u{0120}'));
        assert!(out.contains("hello"));
    }

    #[test]
    fn parse_minimal_tokenizer_json() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3},
                "merges": ["a b"]
            }
        }"#;
        let parsed = HfTokenizerJson::parse(json).expect("minimal parse ok");
        assert_eq!(parsed.vocab.len(), 4);
        assert_eq!(parsed.merges.len(), 1);
        assert_eq!(parsed.merges[0], ("a".to_owned(), "b".to_owned()));
    }

    #[test]
    fn parse_array_merges() {
        let json = r#"{
            "model": {
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": [["a", "b"]]
            }
        }"#;
        let parsed = HfTokenizerJson::parse(json).expect("array merges ok");
        assert_eq!(parsed.merges[0], ("a".to_owned(), "b".to_owned()));
    }

    #[test]
    fn parse_detects_byte_level() {
        let json = r#"{
            "pre_tokenizer": {"type": "ByteLevel"},
            "model": {
                "vocab": {"a": 0},
                "merges": []
            }
        }"#;
        let parsed = HfTokenizerJson::parse(json).expect("parse ok");
        assert!(parsed.byte_level);
    }

    #[test]
    fn parse_missing_model_errors() {
        let json = r#"{"foo": "bar"}"#;
        let err = HfTokenizerJson::parse(json).expect_err("should fail");
        match err {
            TokenizerError::HfFormat(msg) => assert!(msg.contains("model")),
            other => panic!("expected HfFormat, got {other:?}"),
        }
    }

    #[test]
    fn parse_picks_up_special_tokens() {
        let json = r#"{
            "added_tokens": [
                {"id": 100, "content": "<|im_start|>", "special": true},
                {"id": 101, "content": "foo", "special": false}
            ],
            "model": {
                "vocab": {"a": 0},
                "merges": []
            }
        }"#;
        let parsed = HfTokenizerJson::parse(json).expect("parse ok");
        assert!(parsed.special_tokens.contains_key("<|im_start|>"));
        assert!(!parsed.special_tokens.contains_key("foo"));
        // But `foo` should still be in the vocab.
        assert_eq!(parsed.vocab.get("foo"), Some(&101));
    }

    #[test]
    fn into_tokenizer_roundtrip() {
        // Build a byte-level-like fixture so decode goes through unicode_to_byte.
        let json = r#"{
            "pre_tokenizer": {"type": "ByteLevel"},
            "model": {
                "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3},
                "merges": ["a b"]
            }
        }"#;
        let parsed = HfTokenizerJson::parse(json).expect("parse ok");
        let tok = parsed.into_tokenizer().expect("to tokenizer ok");
        assert!(tok.vocab_size() >= 4);
    }

    #[test]
    fn malformed_merge_entry_errors() {
        let json = r#"{
            "model": {
                "vocab": {"a": 0},
                "merges": [{"not": "a pair"}]
            }
        }"#;
        let err = HfTokenizerJson::parse(json).expect_err("should fail");
        assert!(matches!(err, TokenizerError::HfFormat(_)));
    }

    #[test]
    fn vocab_non_integer_id_errors() {
        let json = r#"{
            "model": {
                "vocab": {"a": "not an int"},
                "merges": []
            }
        }"#;
        let err = HfTokenizerJson::parse(json).expect_err("should fail");
        assert!(matches!(err, TokenizerError::HfFormat(_)));
    }

    #[test]
    fn inverse_map_len() {
        let inv = bytes_to_unicode_inverse();
        assert_eq!(inv.len(), 256);
    }
}
