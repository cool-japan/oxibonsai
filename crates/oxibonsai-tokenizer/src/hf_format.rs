//! HuggingFace `tokenizer.json` format parser (BPE, Unigram, and WordPiece models).
//!
//! This module provides a faithful, dependency-free parser for the tokenizer
//! JSON format emitted by the HuggingFace `tokenizers` library.  It supports:
//!
//! - BPE `model` with `vocab` and `merges` (both string `"a b"` form and
//!   array `["a","b"]` form)
//! - Unigram `model` with `vocab` as `[[token, score], ...]` and `unk_id`
//! - WordPiece `model` with `vocab` as `{"token": id, ...}`, `unk_token`, and
//!   optional `max_input_chars_per_word` (BERT/RoBERTa/DeBERTa family)
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
use std::sync::OnceLock;

use serde_json::Value;

use crate::{
    bpe::{BpeMerges, PreTokenizerKind},
    error::{TokenizerError, TokenizerResult},
    tokenizer::{OxiTokenizer, TokenizerConfig},
    vocab::Vocabulary,
    wordpiece::WordPieceVocab,
};

// ── Bytes-to-unicode map ─────────────────────────────────────────────────────

/// Build the canonical GPT-2 bytes-to-unicode table.
///
/// Returns an array indexed by byte value containing the Unicode code point
/// for that byte.  The inverse map is built by [`bytes_to_unicode_inverse`].
///
/// `O(256 × 188)` (a linear `contains` scan per byte) — cheap once, but see
/// [`bytes_to_unicode_map`], which is the function every caller should
/// actually use: it caches this result process-wide (TOK-14) so repeated
/// construction (e.g. one `OxiTokenizer` per request) does not pay this
/// cost more than once per process.
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

/// Process-wide cache for [`bytes_to_unicode_map`] (TOK-14): the table is
/// deterministic and process-global, so building it once with
/// [`build_bytes_to_unicode`]'s `O(256×188)` linear scan and reusing the
/// `Copy` result is strictly better than re-running that scan on every
/// call — which, before this cache, happened once per constructed
/// [`crate::tokenizer::OxiTokenizer`] (`OxiTokenizer::new` /
/// `with_unigram` / `with_wordpiece` each call this to populate their
/// `byte_to_unicode` field) and, via [`bytes_to_unicode_string`], on every
/// single legacy (non-cached) byte-level string conversion.
static BYTES_TO_UNICODE_CACHE: OnceLock<[char; 256]> = OnceLock::new();

/// Return the public 256-entry GPT-2 bytes-to-unicode mapping.
///
/// The returned array is indexed by byte value.  Element `i` is the Unicode
/// character used to represent byte `i` in the HF ByteLevel pre-tokenizer.
/// `O(1)` after the first call in the process (TOK-14).
pub fn bytes_to_unicode_map() -> [char; 256] {
    *BYTES_TO_UNICODE_CACHE.get_or_init(build_bytes_to_unicode)
}

/// Return the Unicode character for the given byte, per the GPT-2 map.
pub fn byte_to_unicode(b: u8) -> char {
    bytes_to_unicode_map()[b as usize]
}

/// Process-wide cache for the `char -> byte` inverse map, built once on
/// first use. This backs [`unicode_to_byte`], which is called once per
/// decoded `char` on the server's per-token streaming decode hot path
/// (see [`crate::tokenizer::OxiTokenizer::decode_id_into`] /
/// [`crate::streaming::StreamingDecoder::push_token`]); rebuilding the
/// 256-entry table (itself an O(256×188) linear-scan construction) on every
/// call would make that hot path needlessly quadratic.
static UNICODE_TO_BYTE_CACHE: OnceLock<HashMap<char, u8>> = OnceLock::new();

/// The maximum unicode codepoint the bytes-to-unicode table ever produces
/// (`0x100 + 67`, the last of the 68 remapped bytes; see
/// [`build_bytes_to_unicode`]). Sized with headroom for
/// [`UNICODE_TO_BYTE_ARRAY_CACHE`]'s flat lookup array.
const UNICODE_TO_BYTE_ARRAY_LEN: usize = 512;

/// Process-wide flat-array reverse cache (TOK-14 fix-shape: "`OnceLock<[char;
/// 256]>` plus a reverse `[u8; 512]`"): every codepoint the forward table can
/// produce is `< 512` (`0x143` is the highest, `0x100..=0x143` covers the 68
/// remapped bytes; everything else is a printable-ASCII/Latin-1 passthrough
/// well under `0x100`), so a direct array index avoids even the hashing
/// [`UNICODE_TO_BYTE_CACHE`] still costs. `u16::MAX` marks a codepoint that
/// is not part of the table (byte values fit in `u8`, so this is
/// unambiguous). [`unicode_to_byte`] uses this as its fast path and falls
/// back to the hashmap cache only for `ch as u32 >= 512` (impossible for a
/// valid entry, but keeps the function correct for any `char`).
static UNICODE_TO_BYTE_ARRAY_CACHE: OnceLock<[u16; UNICODE_TO_BYTE_ARRAY_LEN]> = OnceLock::new();

fn build_unicode_to_byte_array() -> [u16; UNICODE_TO_BYTE_ARRAY_LEN] {
    let table = build_bytes_to_unicode();
    let mut out = [u16::MAX; UNICODE_TO_BYTE_ARRAY_LEN];
    for (byte, &ch) in table.iter().enumerate() {
        let cp = ch as u32;
        if (cp as usize) < UNICODE_TO_BYTE_ARRAY_LEN {
            out[cp as usize] = byte as u16;
        }
    }
    out
}

/// Inverse of [`byte_to_unicode`]: return the byte value for a Unicode char,
/// or `None` if `ch` is not part of the 256-entry table.
///
/// `O(1)` after the first call in the process, via a direct array index
/// (TOK-14) rather than a hash lookup for the overwhelmingly common case
/// (every codepoint this table ever produces is `< 512`).
pub fn unicode_to_byte(ch: char) -> Option<u8> {
    let cp = ch as u32;
    if (cp as usize) < UNICODE_TO_BYTE_ARRAY_LEN {
        let table = UNICODE_TO_BYTE_ARRAY_CACHE.get_or_init(build_unicode_to_byte_array);
        let v = table[cp as usize];
        return if v == u16::MAX { None } else { Some(v as u8) };
    }
    // Unreachable for any codepoint the table actually produces, but kept
    // as a correct (if slower) fallback rather than assuming the array
    // bound above can never change.
    UNICODE_TO_BYTE_CACHE
        .get_or_init(bytes_to_unicode_inverse)
        .get(&ch)
        .copied()
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
    /// WordPiece model (BERT/RoBERTa/DeBERTa family) — uses `vocab` (object),
    /// `unk_token`, and optional `max_input_chars_per_word`.
    WordPiece,
    /// Any other (currently unrecognised) model type.  Stored for diagnostics.
    Other(String),
}

impl HfModelType {
    fn from_str(s: &str) -> Self {
        match s {
            "BPE" => Self::Bpe,
            "Unigram" => Self::Unigram,
            "WordPiece" => Self::WordPiece,
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
    /// For WordPiece models this is populated from `model.vocab` (an object).
    pub vocab: HashMap<String, u32>,
    /// Ordered list of BPE merge pairs `(left, right)`.
    ///
    /// Order defines priority — first pair is highest priority.
    /// Always empty for Unigram and WordPiece models.
    pub merges: Vec<(String, String)>,
    /// For Unigram models: ordered `(token, log_prob)` pairs from `model.vocab`.
    ///
    /// The position of each pair in the vector determines its token ID.
    /// `None` for BPE and WordPiece models.
    pub unigram_vocab: Option<Vec<(String, f64)>>,
    /// For Unigram models: the UNK token ID from `model.unk_id`.
    ///
    /// `None` for BPE and WordPiece models.
    pub unigram_unk_id: Option<u32>,
    /// For WordPiece models: the `max_input_chars_per_word` from `model`.
    ///
    /// `None` for BPE and Unigram models (defaults to 100 when absent).
    pub wordpiece_max_chars: Option<usize>,
    /// Tokens flagged as `special == true` in `added_tokens`.
    pub special_tokens: HashMap<String, u32>,
    /// ALL tokens present in `added_tokens`, regardless of the JSON
    /// `special` flag — a superset of `special_tokens`.
    ///
    /// HuggingFace's `AddedVocabulary` atomically protects *every* added
    /// token from pre-tokenization/model segmentation during encode, not
    /// just the ones marked `special == true` (e.g. Qwen's `<tool_call>`,
    /// `<|fim_prefix|>` are `special == false` yet must still encode
    /// atomically). `special_tokens` remains the narrower set used for
    /// decode-skip / BOS-EOS-style semantics.
    pub protected_tokens: HashMap<String, u32>,
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
    /// `true` when `normalizer` declares (bare, or nested one level inside a
    /// `Sequence`) `{"type": "NFC"}` (TOK-09). Applied by
    /// [`crate::tokenizer::OxiTokenizer::encode`] via
    /// [`crate::tokenizer::TokenizerConfig::normalize_nfc`].
    pub normalize_nfc: bool,
    /// The verbatim `Split`-stage regex pattern, when `pre_tokenizer` is a
    /// `Sequence` containing a `Split` node with a `Regex` pattern (the
    /// shape Qwen2/Qwen3/Bonsai's real `tokenizer.json` files use).
    ///
    /// `None` means: if [`Self::byte_level`] is set, fall back to the
    /// built-in canonical GPT-2/Qwen2 pattern (a bare `ByteLevel`
    /// pre-tokenizer with no preceding explicit `Split` applies HF's own
    /// default internal regex, which *is* that canonical pattern) — see
    /// [`crate::bpe::PreTokenizerKind::Gpt2`]. Kept separate from the
    /// enum-driven GGUF path deliberately (TOK-04 verdict correction):
    /// taking the pattern verbatim here, rather than re-deriving it from a
    /// `PreTokenizerKind`, means a `tokenizer.json` is never re-split by a
    /// second, possibly-different default pattern.
    pub pretokenize_pattern: Option<String>,
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
            HfModelType::WordPiece => {
                // WordPiece vocab is an object {"token": id, ...} — re-use the
                // BPE object-vocab parser for the model.vocab field; merges are
                // not used by WordPiece so we accept an empty/absent merges list.
                let vocab_val = model.get("vocab").ok_or_else(|| {
                    TokenizerError::HfFormat("WordPiece model.vocab missing".into())
                })?;
                let mut wp_vocab: HashMap<String, u32> = HashMap::new();
                match vocab_val {
                    Value::Object(map) => {
                        for (token, id_val) in map {
                            let id = id_val.as_u64().ok_or_else(|| {
                                TokenizerError::HfFormat(format!(
                                    "WordPiece vocab id for '{token}' is not an integer"
                                ))
                            })? as u32;
                            wp_vocab.insert(token.clone(), id);
                        }
                    }
                    _ => {
                        return Err(TokenizerError::HfFormat(
                            "WordPiece model.vocab must be an object".into(),
                        ));
                    }
                }
                (wp_vocab, vec![], None, None)
            }
            HfModelType::Bpe | HfModelType::Other(_) => {
                let (v, m) = parse_bpe_model(model)?;
                (v, m, None, None)
            }
        };

        // ── 2. added_tokens → special_tokens / protected_tokens ─────────────
        // Every entry in `added_tokens` — special or not — must be tracked in
        // `protected_tokens` so it gets atomic leftmost-longest carve-out
        // protection during encode (HF `AddedVocabulary` semantics). Only
        // entries with `special == true` additionally go into the narrower
        // `special_tokens` set.
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        let mut protected_tokens: HashMap<String, u32> = HashMap::new();
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
                    protected_tokens.insert(content.clone(), id);
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

        // ── 4b. Normalizer (TOK-09) ──────────────────────────────────────────
        let normalize_nfc = parse_normalizer(root.get("normalizer"))?;

        // ── 4c. Explicit `Split` pre-tokenizer pattern (TOK-04 correction) ───
        let pretokenize_pattern = root
            .get("pre_tokenizer")
            .map(find_split_regex_pattern)
            .transpose()?
            .flatten();

        // ── 4d. Post-processor (TOK-09 gap closure) ──────────────────────────
        // Validate-only: no post-processor type this crate implements needs
        // a stored field, but a declared, token-stream-altering type must
        // not be silently dropped -- see `validate_post_processor`.
        validate_post_processor(root.get("post_processor"))?;

        // ── 5. WordPiece-specific: max_input_chars_per_word ─────────────────
        let wordpiece_max_chars = if model_type == HfModelType::WordPiece {
            model
                .get("max_input_chars_per_word")
                .and_then(Value::as_u64)
                .map(|n| n as usize)
        } else {
            None
        };

        Ok(Self {
            model_type,
            vocab,
            merges,
            unigram_vocab,
            unigram_unk_id,
            wordpiece_max_chars,
            special_tokens,
            protected_tokens,
            bos_token,
            eos_token,
            unk_token,
            pad_token,
            byte_level,
            normalize_nfc,
            pretokenize_pattern,
        })
    }

    /// Convert the parsed document into a ready-to-use [`OxiTokenizer`].
    ///
    /// For `"BPE"` (and unrecognised) model types the existing BPE path is
    /// used.  For `"Unigram"` a [`crate::unigram::UnigramVocab`] is built and
    /// [`OxiTokenizer::with_unigram`] is called.  For `"WordPiece"` a
    /// [`WordPieceVocab`] is built and [`OxiTokenizer::with_wordpiece`] is
    /// called.
    pub fn into_tokenizer(self) -> TokenizerResult<OxiTokenizer> {
        // Build the shared vocabulary used for decode in all paths.
        let mut vocabulary = Vocabulary::new();
        for (token, id) in &self.vocab {
            if self.special_tokens.contains_key(token) {
                vocabulary.add_special(token, *id);
            } else if self.protected_tokens.contains_key(token) {
                // Non-special added token (e.g. `<tool_call>`,
                // `<|fim_prefix|>`): still protected from pre-tokenization
                // during encode, but not treated as "special" for
                // decode-skip purposes.
                vocabulary.add_protected(token, *id);
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
            normalize_nfc: self.normalize_nfc,
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

        let pretokenize_pattern = self.pretokenize_pattern.clone();
        let byte_level = self.byte_level;

        let tok = match self.model_type {
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
                OxiTokenizer::new(vocabulary, merges, config)
            }
            HfModelType::Unigram => {
                let entries = self.unigram_vocab.ok_or_else(|| {
                    TokenizerError::HfFormat(
                        "Unigram model requires `model.vocab` array".to_owned(),
                    )
                })?;
                // Prefer unk_id from `model.unk_id`; fall back to the id
                // resolved from the `unk_token` string.
                let effective_unk_id = self.unigram_unk_id.unwrap_or(config.unk_token_id);
                let unigram_vocab = crate::unigram::UnigramVocab::new(entries, effective_unk_id)
                    .map_err(|e| TokenizerError::HfFormat(format!("invalid Unigram vocab: {e}")))?;
                OxiTokenizer::with_unigram(vocabulary, unigram_vocab, config)
            }
            HfModelType::WordPiece => {
                // Build the ordered token list from the vocab map.
                let wp_vocab = build_wordpiece_vocab_from_map(
                    &self.vocab,
                    self.unk_token.as_deref(),
                    self.wordpiece_max_chars,
                    config.unk_token_id,
                )?;
                OxiTokenizer::with_wordpiece(vocabulary, wp_vocab, config)
            }
        };

        // Attach the byte-level `Split` pre-tokenizer pattern (TOK-03/04).
        // A verbatim pattern from the JSON itself takes priority; otherwise
        // fall back to the built-in canonical pattern whenever ByteLevel was
        // detected (matching a bare `ByteLevel` pre-tokenizer's own default
        // internal regex — see `HfTokenizerJson::pretokenize_pattern`'s
        // docs for why this can never double-split).
        let tok = match pretokenize_pattern {
            Some(pattern) => tok.with_pretokenizer_pattern(&pattern)?,
            None if byte_level => tok.with_pretokenizer_kind(PreTokenizerKind::Gpt2),
            None => tok,
        };

        Ok(tok)
    }
}

// ── Private parse helpers ────────────────────────────────────────────────────

/// Build a [`WordPieceVocab`] from the flat token→id map obtained during
/// parsing of a WordPiece `model` section.
///
/// The map may come from the parsed `model.vocab` object.  This function:
/// 1. Sorts entries by ID to produce a contiguous, ordered token list.
/// 2. Validates that IDs are contiguous starting from 0.
/// 3. Resolves the UNK token ID from `unk_token_str` (falling back to
///    `fallback_unk_id` from the config if the string is absent or not found).
/// 4. Applies the optional `max_chars` limit.
fn build_wordpiece_vocab_from_map(
    vocab_map: &HashMap<String, u32>,
    unk_token_str: Option<&str>,
    max_chars: Option<usize>,
    fallback_unk_id: u32,
) -> TokenizerResult<WordPieceVocab> {
    // Sort by ID so that `tokens[id] == token_string`.
    let mut pairs: Vec<(&str, u32)> = vocab_map.iter().map(|(k, &v)| (k.as_str(), v)).collect();
    pairs.sort_by_key(|(_, id)| *id);

    // Validate contiguity.
    for (i, (_, id)) in pairs.iter().enumerate() {
        if *id as usize != i {
            return Err(TokenizerError::HfFormat(format!(
                "WordPiece vocab IDs are not contiguous: expected {i}, found {id}"
            )));
        }
    }

    let tokens: Vec<String> = pairs.into_iter().map(|(t, _)| t.to_owned()).collect();

    // Resolve the UNK token ID.
    let unk_id: u32 = unk_token_str
        .and_then(|s| vocab_map.get(s).copied())
        .unwrap_or(fallback_unk_id);

    let wp = WordPieceVocab::new(tokens, unk_id)
        .map_err(|e| TokenizerError::HfFormat(format!("invalid WordPiece vocab: {e}")))?;

    Ok(if let Some(max) = max_chars {
        wp.with_max_input_chars(max)
    } else {
        wp
    })
}

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
    // Track which token string first claimed each ID so that a malformed
    // vocab with two distinct token strings sharing one numeric ID is
    // rejected up front, rather than silently picking a non-deterministic
    // "winner" later based on `HashMap` iteration order (which uses a
    // randomized `RandomState` and is not stable across process runs).
    let mut seen_ids: HashMap<u32, String> =
        HashMap::with_capacity(vocab_val.as_object().map(|m| m.len()).unwrap_or(0));
    match vocab_val {
        Value::Object(map) => {
            for (token, id_val) in map {
                let id = id_val.as_u64().ok_or_else(|| {
                    TokenizerError::HfFormat(format!("vocab entry {token:?} has non-integer id"))
                })? as u32;
                if let Some(prev_token) = seen_ids.get(&id) {
                    if prev_token != token {
                        return Err(TokenizerError::HfFormat(format!(
                            "BPE vocab id {id} is assigned to both {prev_token:?} and {token:?}; vocab IDs must be unique"
                        )));
                    }
                }
                seen_ids.insert(id, token.clone());
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

/// Return `true` if `value`'s `type` field equals `"ByteLevel"`.
fn is_byte_level_entry(value: &Value) -> bool {
    value
        .get("type")
        .and_then(Value::as_str)
        .map(|t| t == "ByteLevel")
        .unwrap_or(false)
}

/// Return `true` if `value` is (or, when `Sequence`-wrapped, nests) a
/// `ByteLevel` pre_tokenizer/decoder entry.
///
/// In the real HuggingFace `tokenizers` schema a `Sequence` pre_tokenizer or
/// decoder is always a JSON **object** of the shape
/// `{"type":"Sequence","pretokenizers":[...]}` (pre-tokenizers) or
/// `{"type":"Sequence","decoders":[...]}` (decoders) — never a bare
/// top-level array. This recurses into that nested array (checked under
/// both possible field names since callers pass either shape) before
/// giving up.
fn contains_byte_level(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            if is_byte_level_entry(value) {
                return true;
            }
            let is_sequence = map
                .get("type")
                .and_then(Value::as_str)
                .map(|t| t == "Sequence")
                .unwrap_or(false);
            if is_sequence {
                for nested_field in ["pretokenizers", "decoders"] {
                    if let Some(Value::Array(list)) = map.get(nested_field) {
                        if list.iter().any(contains_byte_level) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        Value::Array(list) => list.iter().any(contains_byte_level),
        _ => false,
    }
}

/// Return `true` if the tokenizer pre-tokenizer or decoder is ByteLevel.
fn detect_byte_level(root: &Value) -> bool {
    let has_bl =
        |field: &str| -> bool { root.get(field).map(contains_byte_level).unwrap_or(false) };
    has_bl("pre_tokenizer") || has_bl("decoder")
}

// ── Normalizer (TOK-09) ────────────────────────────────────────────────────────

/// Parse the top-level `normalizer` field, returning whether NFC
/// normalization must be applied.
///
/// Absent or JSON `null` means "no normalizer declared" (`Ok(false)`,
/// not an error — the overwhelming majority of `tokenizer.json` files omit
/// this field entirely). Any *declared* normalizer this crate does not
/// implement is a hard error (spec: "ERROR on an unsupported normalizer
/// rather than silently skipping it") rather than a silent no-op, since
/// silently ignoring a real normalizer step is exactly the proven TOK-09
/// divergence (HF composing `"e" + U+0301` into `"é"` before encoding;
/// this crate previously encoded the decomposed form as three separate
/// tokens).
fn parse_normalizer(value: Option<&Value>) -> TokenizerResult<bool> {
    match value {
        None => Ok(false),
        Some(v) if v.is_null() => Ok(false),
        Some(v) => normalizer_needs_nfc(v),
    }
}

/// Recursive helper for [`parse_normalizer`]: `true` if `value` is (or, for
/// a `Sequence`, contains) an `NFC` normalizer; errors on any other
/// declared type.
fn normalizer_needs_nfc(value: &Value) -> TokenizerResult<bool> {
    let ty = value.get("type").and_then(Value::as_str).ok_or_else(|| {
        TokenizerError::HfFormat("normalizer entry is missing its `type` field".to_owned())
    })?;
    match ty {
        "NFC" => Ok(true),
        "Sequence" => {
            let list = value
                .get("normalizers")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    TokenizerError::HfFormat(
                        "Sequence normalizer is missing its `normalizers` array".to_owned(),
                    )
                })?;
            let mut needs_nfc = false;
            for entry in list {
                if normalizer_needs_nfc(entry)? {
                    needs_nfc = true;
                }
            }
            Ok(needs_nfc)
        }
        other => Err(TokenizerError::HfFormat(format!(
            "unsupported tokenizer.json normalizer type {other:?}: only \"NFC\" \
             (optionally nested inside a \"Sequence\") is implemented; refusing \
             to silently mis-tokenize rather than ignoring it"
        ))),
    }
}

// ── Split pre-tokenizer pattern extraction (TOK-04 correction) ─────────────────

/// Search a `pre_tokenizer` value (bare or `Sequence`-wrapped, matching
/// [`contains_byte_level`]'s recursion shape) for a `Split` node carrying a
/// `Regex` pattern, returning that pattern verbatim if found.
///
/// This is how Qwen2/Qwen3/Bonsai's real `tokenizer.json` files declare
/// their pre-tokenizer: `{"type": "Sequence", "pretokenizers": [{"type":
/// "Split", "pattern": {"Regex": "..."}, "behavior": "Isolated", ...},
/// {"type": "ByteLevel", "use_regex": false, ...}]}`. Taking the pattern
/// verbatim from here — rather than re-deriving it from a
/// [`PreTokenizerKind`] — means the `ByteLevel` stage's own default regex
/// is never *also* applied, which would silently re-split already-atomic
/// pieces a second time.
///
/// # Errors
/// Propagates [`validate_split_behavior`]'s error for a `Split` node that
/// declares a `behavior` other than `"Isolated"`, or `invert: true` —
/// either would mean this crate's [`crate::bpe::pretokenize_regex`] (which
/// always implements plain `Isolated`/`invert=false`) taking the pattern
/// verbatim no longer reproduces the source tokenizer's actual split.
fn find_split_regex_pattern(value: &Value) -> TokenizerResult<Option<String>> {
    match value {
        Value::Object(map) => {
            let ty = map.get("type").and_then(Value::as_str);
            if ty == Some("Split") {
                validate_split_behavior(value)?;
                return Ok(map
                    .get("pattern")
                    .and_then(|p| p.get("Regex"))
                    .and_then(Value::as_str)
                    .map(str::to_owned));
            }
            if ty == Some("Sequence") {
                if let Some(Value::Array(list)) = map.get("pretokenizers") {
                    for entry in list {
                        if let Some(pattern) = find_split_regex_pattern(entry)? {
                            return Ok(Some(pattern));
                        }
                    }
                }
            }
            Ok(None)
        }
        Value::Array(list) => {
            for entry in list {
                if let Some(pattern) = find_split_regex_pattern(entry)? {
                    return Ok(Some(pattern));
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

/// Validate a `Split` pre-tokenizer node's `behavior`/`invert` fields
/// (a minor finding filed alongside TOK-04's
/// pattern-extraction correction above).
///
/// [`crate::bpe::pretokenize_regex`] always implements HF's `Isolated`
/// behaviour (every regex match becomes its own piece; non-matches pass
/// through unchanged) with `invert = false` (matches are kept as the
/// pieces, never treated as the separators). Silently taking a `Split`
/// pattern verbatim while ignoring an *explicitly declared* different
/// `behavior`, or `invert: true`, would change what the pattern actually
/// produces and mis-tokenize without warning. **Absent** fields keep
/// today's behaviour (`Isolated`/`false`, HF's own defaults, and what
/// every hand-written fixture already in this crate assumes when it omits
/// them) — only an *explicit* non-conforming declaration is an error, so
/// this stays fail-loud without becoming fail-loud-by-default.
fn validate_split_behavior(value: &Value) -> TokenizerResult<()> {
    if let Some(behavior) = value.get("behavior").and_then(Value::as_str) {
        if behavior != "Isolated" {
            return Err(TokenizerError::HfFormat(format!(
                "unsupported tokenizer.json Split behavior {behavior:?}: only \
                 \"Isolated\" is implemented by this crate's `pretokenize_regex`"
            )));
        }
    }
    if value
        .get("invert")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(TokenizerError::HfFormat(
            "unsupported tokenizer.json Split invert=true: this crate's \
             `pretokenize_regex` only implements invert=false (regex matches \
             are kept as the pieces, never treated as separators)"
                .to_owned(),
        ));
    }
    Ok(())
}

// ── Post-processor (TOK-09 gap closure) ─────────────────────────────────────

/// Validate the top-level `post_processor` field.
///
/// Absent or JSON `null` is fine (`Ok(())`) — most Qwen-family
/// `tokenizer.json` files declare none; this crate's BOS/EOS insertion is
/// driven independently by [`crate::tokenizer::TokenizerConfig::add_bos`] /
/// `add_eos`. A bare, or `Sequence`-nested, `ByteLevel` post-processor is
/// also accepted as a documented no-op: HF's `ByteLevel` post-processor
/// only repairs the character-offset bookkeeping of the (unused by this
/// crate) `Encoding::offsets` field — it never inserts, removes or renames
/// a token id, so taking no action for it is correct, not a gap.
///
/// Any other declared type — `TemplateProcessing`, `BertProcessing`,
/// `RobertaProcessing` — genuinely changes the emitted token *stream*
/// (typically BOS/EOS/CLS/SEP insertion per a template), which this crate
/// does not implement. Silently ignoring a real, stream-altering
/// post-processor would reproduce exactly the class of defect TOK-09 fixed
/// for `normalizer` (a declared step silently skipped rather than honoured
/// or refused) — so an unsupported type is a hard, fail-loud error here
/// too, matching [`normalizer_needs_nfc`]'s policy, rather than a silent
/// no-op that could leave e.g. a `TemplateProcessing`-declared BOS/EOS
/// insertion invisibly missing.
fn validate_post_processor(value: Option<&Value>) -> TokenizerResult<()> {
    match value {
        None => Ok(()),
        Some(v) if v.is_null() => Ok(()),
        Some(v) => post_processor_is_supported(v),
    }
}

/// Recursive helper for [`validate_post_processor`]: errors on any declared
/// `post_processor` type this crate does not treat as an identity no-op
/// over the token stream.
fn post_processor_is_supported(value: &Value) -> TokenizerResult<()> {
    let ty = value.get("type").and_then(Value::as_str).ok_or_else(|| {
        TokenizerError::HfFormat("post_processor entry is missing its `type` field".to_owned())
    })?;
    match ty {
        "ByteLevel" => Ok(()),
        "Sequence" => {
            let list = value
                .get("processors")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    TokenizerError::HfFormat(
                        "Sequence post_processor is missing its `processors` array".to_owned(),
                    )
                })?;
            for entry in list {
                post_processor_is_supported(entry)?;
            }
            Ok(())
        }
        other => Err(TokenizerError::HfFormat(format!(
            "unsupported tokenizer.json post_processor type {other:?}: only \"ByteLevel\" \
             (optionally nested inside a \"Sequence\") is an identity no-op for this \
             crate's token stream; \"TemplateProcessing\"/\"BertProcessing\"/\
             \"RobertaProcessing\" would insert or alter tokens (e.g. BOS/EOS/CLS/SEP) \
             that this crate does not apply — refusing to silently mis-tokenize rather \
             than ignoring it"
        ))),
    }
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
        // And, per HF `AddedVocabulary` semantics, `foo` (special: false)
        // must still land in the broader protected-tokens set so it gets
        // atomic encode-time carve-out even though it is not "special".
        assert!(parsed.protected_tokens.contains_key("<|im_start|>"));
        assert!(parsed.protected_tokens.contains_key("foo"));
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
