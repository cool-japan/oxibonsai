//! Tokenizer serialization: save and load tokenizer state to/from text files.
//!
//! Format (plain text, UTF-8):
//! Line 1: "oxitokenizer v1"
//! Line 2: `"vocab_size <N>"`
//! Line 3: `"merges <M>"`
//! Lines 4..(4+N): `"tok_id <id> <token_text_base64>"`
//! Lines (4+N)..(4+N+M): `"merge <left_id> <right_id> <merged_id>"`
//! Special tokens (if any): `"special <token_text_base64> <id>"`

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Header magic string.
pub const FORMAT_MAGIC: &str = "oxitokenizer v1";

// ── Base64 implementation ─────────────────────────────────────────────────────

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes to standard base64 (3 bytes → 4 chars, padded with '=').
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = Vec::with_capacity(bytes.len().div_ceil(3) * 4);

    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };

        let combined = (b0 << 16) | (b1 << 8) | b2;

        out.push(BASE64_CHARS[((combined >> 18) & 0x3F) as usize]);
        out.push(BASE64_CHARS[((combined >> 12) & 0x3F) as usize]);

        if chunk.len() > 1 {
            out.push(BASE64_CHARS[((combined >> 6) & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }

        if chunk.len() > 2 {
            out.push(BASE64_CHARS[(combined & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
    }

    // SAFETY: `BASE64_CHARS` is ASCII-only and `=` is ASCII, so every byte we
    // pushed is a valid ASCII character.  `String::from_utf8` therefore cannot
    // fail — but rather than `unwrap`, we fall back to the empty string
    // (keeps the function panic-free, matching the no-unwrap policy).  The
    // caller's "decode failed" path will flag any such inconsistency cleanly.
    String::from_utf8(out).unwrap_or_default()
}

/// Decode a standard base64 string back to bytes.
pub fn base64_decode(s: &str) -> Result<Vec<u8>, SerializationError> {
    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity((s.len() * 3) / 4 + 1);

    let decode_char = |c: u8| -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };

    let chars: Vec<u8> = s.bytes().collect();

    for chunk in chars.chunks(4) {
        match chunk.len() {
            4 => {
                let v0 = decode_char(chunk[0]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[0] as char))
                })?;
                let v1 = decode_char(chunk[1]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[1] as char))
                })?;
                let v2 = decode_char(chunk[2]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[2] as char))
                })?;
                let v3 = decode_char(chunk[3]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[3] as char))
                })?;
                let combined = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
                out.push(((combined >> 16) & 0xFF) as u8);
                out.push(((combined >> 8) & 0xFF) as u8);
                out.push((combined & 0xFF) as u8);
            }
            3 => {
                let v0 = decode_char(chunk[0]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[0] as char))
                })?;
                let v1 = decode_char(chunk[1]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[1] as char))
                })?;
                let v2 = decode_char(chunk[2]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[2] as char))
                })?;
                let combined = (v0 << 18) | (v1 << 12) | (v2 << 6);
                out.push(((combined >> 16) & 0xFF) as u8);
                out.push(((combined >> 8) & 0xFF) as u8);
            }
            2 => {
                let v0 = decode_char(chunk[0]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[0] as char))
                })?;
                let v1 = decode_char(chunk[1]).ok_or_else(|| {
                    SerializationError::Base64Error(format!("invalid char '{}'", chunk[1] as char))
                })?;
                let combined = (v0 << 18) | (v1 << 12);
                out.push(((combined >> 16) & 0xFF) as u8);
            }
            1 => {
                return Err(SerializationError::Base64Error(
                    "truncated base64 group of 1 char".to_string(),
                ));
            }
            _ => {}
        }
    }

    Ok(out)
}

// ── SerializationError ────────────────────────────────────────────────────────

/// Errors that can occur during tokenizer serialization/deserialization.
#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid format magic: expected '{expected}', got '{got}'")]
    InvalidMagic { expected: String, got: String },

    #[error("parse error on line {line}: {msg}")]
    ParseError { line: usize, msg: String },

    #[error("base64 decode error: {0}")]
    Base64Error(String),

    #[error("duplicate token id {0}")]
    DuplicateId(u32),
}

// ── ConfigSnapshot ───────────────────────────────────────────────────────────

/// The subset of [`crate::tokenizer::TokenizerConfig`] that changes encode
/// and decode *behaviour* — as opposed to the vocabulary/merges content
/// already carried by [`TokenizerState`] — and therefore must round-trip
/// through [`TokenizerState::save`] / [`TokenizerState::load`], or a
/// byte-level tokenizer silently decodes through the wrong path after a
/// save+load cycle (missed finding, TOK-01 second route).
///
/// Defaults to exactly [`crate::tokenizer::TokenizerConfig::default`]'s
/// values, so a state saved by a version of this crate that predates this
/// field (or one that never calls [`TokenizerState::with_config`]) still
/// loads with today's observed behaviour — no format break.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigSnapshot {
    /// See [`crate::tokenizer::TokenizerConfig::byte_level_decode`].
    pub byte_level_decode: bool,
    /// See [`crate::tokenizer::TokenizerConfig::add_bos`].
    pub add_bos: bool,
    /// See [`crate::tokenizer::TokenizerConfig::add_eos`].
    pub add_eos: bool,
    /// See [`crate::tokenizer::TokenizerConfig::bos_token_id`].
    pub bos_token_id: u32,
    /// See [`crate::tokenizer::TokenizerConfig::eos_token_id`].
    pub eos_token_id: u32,
    /// See [`crate::tokenizer::TokenizerConfig::unk_token_id`].
    pub unk_token_id: u32,
    /// See [`crate::tokenizer::TokenizerConfig::pad_token_id`].
    pub pad_token_id: u32,
}

impl Default for ConfigSnapshot {
    fn default() -> Self {
        // Mirrors `TokenizerConfig::default()` field-for-field.
        Self {
            byte_level_decode: false,
            add_bos: false,
            add_eos: false,
            bos_token_id: 1,
            eos_token_id: 2,
            unk_token_id: 0,
            pad_token_id: 3,
        }
    }
}

// ── TokenizerState ────────────────────────────────────────────────────────────

/// A serializable snapshot of a trained tokenizer.
///
/// Marked `#[non_exhaustive]` (added alongside the `config` field, TOK-01)
/// so a future field can be added the same way without breaking downstream
/// full-struct-literal construction — matching the same precaution already
/// taken on [`crate::tokenizer::TokenizerConfig`]. Build one with
/// [`TokenizerState::new`] or [`TokenizerState::from_trained`] and
/// [`TokenizerState::with_config`].
#[derive(Debug)]
#[non_exhaustive]
pub struct TokenizerState {
    /// id → token string
    pub vocab: HashMap<u32, String>,
    /// (left_id, right_id, merged_id)
    pub merges: Vec<(u32, u32, u32)>,
    /// special token name → id (e.g. `"<BOS>"` → 1)
    pub special_tokens: HashMap<String, u32>,
    /// The encode/decode-affecting config to restore on load (TOK-01).
    /// Defaults to [`ConfigSnapshot::default`]; attach a real tokenizer's
    /// config with [`Self::with_config`] before saving.
    pub config: ConfigSnapshot,
}

impl TokenizerState {
    /// Create an empty state.
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            merges: Vec::new(),
            special_tokens: HashMap::new(),
            config: ConfigSnapshot::default(),
        }
    }

    /// Build a `TokenizerState` from a [`crate::trainer::TrainedTokenizer`].
    ///
    /// `TrainedTokenizer` (the `BpeTrainer`'s output) carries no
    /// byte-level/BOS-EOS configuration of its own, so `config` is left at
    /// [`ConfigSnapshot::default`] here; a caller that trained a byte-level
    /// model should call [`Self::with_config`] with the real config before
    /// saving.
    pub fn from_trained(trained: &crate::trainer::TrainedTokenizer) -> Self {
        let mut state = Self::new();

        for (&id, token) in &trained.vocab {
            if token.starts_with('<') && token.ends_with('>') {
                state.special_tokens.insert(token.clone(), id);
            }
            state.vocab.insert(id, token.clone());
        }

        for rule in &trained.merges {
            state.merges.push((rule.left, rule.right, rule.merged));
        }

        state
    }

    /// Attach an encode/decode config to this state before saving (TOK-01).
    ///
    /// Additive builder method: callers that already have a
    /// [`crate::tokenizer::OxiTokenizer`] (via
    /// [`crate::tokenizer::OxiTokenizer::config`]) can carry its *real*
    /// `byte_level_decode`/BOS/EOS/UNK/PAD settings through a save+load
    /// round trip instead of [`Self::to_oxi_tokenizer`] silently
    /// reconstructing [`crate::tokenizer::TokenizerConfig::default`].
    #[must_use]
    pub fn with_config(mut self, config: ConfigSnapshot) -> Self {
        self.config = config;
        self
    }

    /// Number of vocabulary entries.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Save to a writer.
    ///
    /// The format is deterministic: vocab entries are written sorted by id,
    /// merges in their original order, special tokens sorted by name.
    pub fn save_to<W: Write>(&self, writer: &mut W) -> Result<(), SerializationError> {
        // Line 1: magic
        writeln!(writer, "{}", FORMAT_MAGIC)?;

        // Line 2: vocab_size
        writeln!(writer, "vocab_size {}", self.vocab.len())?;

        // Line 3: merges count
        writeln!(writer, "merges {}", self.merges.len())?;

        // Lines 4..(4+N): tok_id entries sorted by id for determinism
        let mut vocab_entries: Vec<(u32, &str)> =
            self.vocab.iter().map(|(&id, s)| (id, s.as_str())).collect();
        vocab_entries.sort_by_key(|(id, _)| *id);

        for (id, token) in &vocab_entries {
            let encoded = base64_encode(token.as_bytes());
            writeln!(writer, "tok_id {id} {encoded}")?;
        }

        // Merge rules in original order
        for &(left, right, merged) in &self.merges {
            writeln!(writer, "merge {left} {right} {merged}")?;
        }

        // Special tokens sorted by name
        let mut special_entries: Vec<(&str, u32)> = self
            .special_tokens
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        special_entries.sort_by_key(|(name, _)| *name);

        for (name, id) in &special_entries {
            let encoded = base64_encode(name.as_bytes());
            writeln!(writer, "special {encoded} {id}")?;
        }

        // Config snapshot (TOK-01): always written, in a fixed field order,
        // so the format stays deterministic. A pre-this-fix reader ignores
        // these lines only if it treats unrecognised trailing lines as
        // errors it doesn't hit — new `load_from` handles them explicitly;
        // see the format-compat note on `ConfigSnapshot`.
        writeln!(
            writer,
            "config byte_level_decode {}",
            self.config.byte_level_decode
        )?;
        writeln!(writer, "config add_bos {}", self.config.add_bos)?;
        writeln!(writer, "config add_eos {}", self.config.add_eos)?;
        writeln!(writer, "config bos_token_id {}", self.config.bos_token_id)?;
        writeln!(writer, "config eos_token_id {}", self.config.eos_token_id)?;
        writeln!(writer, "config unk_token_id {}", self.config.unk_token_id)?;
        writeln!(writer, "config pad_token_id {}", self.config.pad_token_id)?;

        Ok(())
    }

    /// Load from a reader.
    pub fn load_from<R: BufRead>(reader: &mut R) -> Result<Self, SerializationError> {
        let mut lines = reader.lines();
        let mut line_no: usize = 0;

        // Helper to read the next non-empty line
        let mut next_line = |line_no: &mut usize| -> Result<String, SerializationError> {
            *line_no += 1;
            match lines.next() {
                Some(Ok(l)) => Ok(l),
                Some(Err(e)) => Err(SerializationError::Io(e)),
                None => Err(SerializationError::ParseError {
                    line: *line_no,
                    msg: "unexpected end of file".to_string(),
                }),
            }
        };

        // Line 1: magic
        let magic_line = next_line(&mut line_no)?;
        if magic_line.trim() != FORMAT_MAGIC {
            return Err(SerializationError::InvalidMagic {
                expected: FORMAT_MAGIC.to_string(),
                got: magic_line.trim().to_string(),
            });
        }

        // Line 2: vocab_size <N>
        let vocab_size_line = next_line(&mut line_no)?;
        let vocab_size = parse_count_line(&vocab_size_line, "vocab_size", line_no)?;

        // Line 3: merges <M>
        let merges_line = next_line(&mut line_no)?;
        let merges_count = parse_count_line(&merges_line, "merges", line_no)?;

        // Read vocab entries
        let mut vocab: HashMap<u32, String> = HashMap::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let l = next_line(&mut line_no)?;
            let parts: Vec<&str> = l.trim().splitn(3, ' ').collect();
            if parts.len() != 3 || parts[0] != "tok_id" {
                return Err(SerializationError::ParseError {
                    line: line_no,
                    msg: format!("expected 'tok_id <id> <b64>', got '{l}'"),
                });
            }
            let id: u32 = parts[1]
                .parse()
                .map_err(|_| SerializationError::ParseError {
                    line: line_no,
                    msg: format!("invalid token id '{}'", parts[1]),
                })?;
            let token_bytes = base64_decode(parts[2])?;
            let token =
                String::from_utf8(token_bytes).map_err(|_| SerializationError::ParseError {
                    line: line_no,
                    msg: "token text is not valid UTF-8".to_string(),
                })?;
            if vocab.contains_key(&id) {
                return Err(SerializationError::DuplicateId(id));
            }
            vocab.insert(id, token);
        }

        // Read merge rules
        let mut merges: Vec<(u32, u32, u32)> = Vec::with_capacity(merges_count);
        for _ in 0..merges_count {
            let l = next_line(&mut line_no)?;
            let parts: Vec<&str> = l.trim().splitn(4, ' ').collect();
            if parts.len() != 4 || parts[0] != "merge" {
                return Err(SerializationError::ParseError {
                    line: line_no,
                    msg: format!("expected 'merge <left> <right> <merged>', got '{l}'"),
                });
            }
            let left: u32 = parts[1]
                .parse()
                .map_err(|_| SerializationError::ParseError {
                    line: line_no,
                    msg: format!("invalid merge left id '{}'", parts[1]),
                })?;
            let right: u32 = parts[2]
                .parse()
                .map_err(|_| SerializationError::ParseError {
                    line: line_no,
                    msg: format!("invalid merge right id '{}'", parts[2]),
                })?;
            let merged: u32 = parts[3]
                .parse()
                .map_err(|_| SerializationError::ParseError {
                    line: line_no,
                    msg: format!("invalid merge merged id '{}'", parts[3]),
                })?;
            merges.push((left, right, merged));
        }

        // Read remaining lines as either "special" or "config" entries
        // (both optional sections; order-independent). A file saved before
        // the TOK-01 config-snapshot fix has no "config" lines at all, and
        // `config` simply stays at `ConfigSnapshot::default()` — the exact
        // behaviour such a file already had, so this is purely additive.
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        let mut config = ConfigSnapshot::default();
        for maybe_line in lines {
            line_no += 1;
            let l = maybe_line.map_err(SerializationError::Io)?;
            let l = l.trim();
            if l.is_empty() {
                continue;
            }
            let parts: Vec<&str> = l.splitn(3, ' ').collect();
            match parts.first().copied() {
                Some("special") => {
                    if parts.len() != 3 {
                        return Err(SerializationError::ParseError {
                            line: line_no,
                            msg: format!("expected 'special <b64> <id>', got '{l}'"),
                        });
                    }
                    let name_bytes = base64_decode(parts[1])?;
                    let name = String::from_utf8(name_bytes).map_err(|_| {
                        SerializationError::ParseError {
                            line: line_no,
                            msg: "special token name is not valid UTF-8".to_string(),
                        }
                    })?;
                    let id: u32 = parts[2]
                        .parse()
                        .map_err(|_| SerializationError::ParseError {
                            line: line_no,
                            msg: format!("invalid special token id '{}'", parts[2]),
                        })?;
                    special_tokens.insert(name, id);
                }
                Some("config") => {
                    if parts.len() != 3 {
                        return Err(SerializationError::ParseError {
                            line: line_no,
                            msg: format!("expected 'config <key> <value>', got '{l}'"),
                        });
                    }
                    apply_config_field(&mut config, parts[1], parts[2], line_no)?;
                }
                _ => {
                    return Err(SerializationError::ParseError {
                        line: line_no,
                        msg: format!("expected 'special ...' or 'config ...', got '{l}'"),
                    });
                }
            }
        }

        Ok(TokenizerState {
            vocab,
            merges,
            special_tokens,
            config,
        })
    }

    /// Save to a file path.
    pub fn save(&self, path: &Path) -> Result<(), SerializationError> {
        let file = File::create(path).map_err(SerializationError::Io)?;
        let mut writer = BufWriter::new(file);
        self.save_to(&mut writer)?;
        writer.flush().map_err(SerializationError::Io)?;
        Ok(())
    }

    /// Load from a file path.
    pub fn load(path: &Path) -> Result<Self, SerializationError> {
        let file = File::open(path).map_err(SerializationError::Io)?;
        let mut reader = BufReader::new(file);
        Self::load_from(&mut reader)
    }

    /// Convert to an [`crate::OxiTokenizer`] using our vocab, merges and the
    /// saved [`ConfigSnapshot`] (TOK-01: this used to hardcode
    /// `TokenizerConfig::default()`, silently forcing `byte_level_decode =
    /// false` regardless of what the saved tokenizer actually was — which
    /// both mis-decodes a byte-level vocabulary through the legacy
    /// `Ġ`-stripping path and, via `OxiTokenizer::new`'s
    /// `build_special_ids`, previously reintroduced the exact
    /// punctuation-deletion bug through this second constructor route).
    pub fn to_oxi_tokenizer(&self) -> crate::OxiTokenizer {
        use crate::{
            bpe::BpeMerges,
            tokenizer::{OxiTokenizer, TokenizerConfig},
            vocab::Vocabulary,
        };

        let mut vocabulary = Vocabulary::new();
        for (&id, token) in &self.vocab {
            if self.special_tokens.contains_key(token.as_str()) {
                vocabulary.add_special(token, id);
            } else {
                vocabulary.insert(token, id);
            }
        }

        let mut bpe_merges = BpeMerges::new();
        for &(left_id, right_id, merged_id) in &self.merges {
            let left_str = self.vocab.get(&left_id).map(|s| s.as_str()).unwrap_or("");
            let right_str = self.vocab.get(&right_id).map(|s| s.as_str()).unwrap_or("");
            bpe_merges.add_merge(left_str, right_str, merged_id);
        }

        let config = TokenizerConfig {
            byte_level_decode: self.config.byte_level_decode,
            add_bos: self.config.add_bos,
            add_eos: self.config.add_eos,
            bos_token_id: self.config.bos_token_id,
            eos_token_id: self.config.eos_token_id,
            unk_token_id: self.config.unk_token_id,
            pad_token_id: self.config.pad_token_id,
            ..Default::default()
        };
        OxiTokenizer::new(vocabulary, bpe_merges, config)
    }
}

/// Parse one `config <key> <value>` line into the matching
/// [`ConfigSnapshot`] field.
fn apply_config_field(
    config: &mut ConfigSnapshot,
    key: &str,
    value: &str,
    line_no: usize,
) -> Result<(), SerializationError> {
    fn parse_bool(value: &str, line_no: usize) -> Result<bool, SerializationError> {
        value.parse().map_err(|_| SerializationError::ParseError {
            line: line_no,
            msg: format!("invalid bool value '{value}'"),
        })
    }
    fn parse_u32(value: &str, line_no: usize) -> Result<u32, SerializationError> {
        value.parse().map_err(|_| SerializationError::ParseError {
            line: line_no,
            msg: format!("invalid u32 value '{value}'"),
        })
    }

    match key {
        "byte_level_decode" => config.byte_level_decode = parse_bool(value, line_no)?,
        "add_bos" => config.add_bos = parse_bool(value, line_no)?,
        "add_eos" => config.add_eos = parse_bool(value, line_no)?,
        "bos_token_id" => config.bos_token_id = parse_u32(value, line_no)?,
        "eos_token_id" => config.eos_token_id = parse_u32(value, line_no)?,
        "unk_token_id" => config.unk_token_id = parse_u32(value, line_no)?,
        "pad_token_id" => config.pad_token_id = parse_u32(value, line_no)?,
        other => {
            return Err(SerializationError::ParseError {
                line: line_no,
                msg: format!("unknown config key '{other}'"),
            });
        }
    }
    Ok(())
}

impl Default for TokenizerState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse a line of the form `<keyword> <count>`.
fn parse_count_line(
    line: &str,
    keyword: &str,
    line_no: usize,
) -> Result<usize, SerializationError> {
    let parts: Vec<&str> = line.trim().splitn(2, ' ').collect();
    if parts.len() != 2 || parts[0] != keyword {
        return Err(SerializationError::ParseError {
            line: line_no,
            msg: format!("expected '{keyword} <N>', got '{line}'"),
        });
    }
    parts[1]
        .parse::<usize>()
        .map_err(|_| SerializationError::ParseError {
            line: line_no,
            msg: format!("invalid count value '{}'", parts[1]),
        })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod inline_tests {
    use super::*;

    #[test]
    fn base64_encode_decode_hello() {
        let original = b"Hello, World!";
        let encoded = base64_encode(original);
        let decoded = base64_decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64_empty() {
        let encoded = base64_encode(b"");
        assert_eq!(encoded, "");
        let decoded = base64_decode("").expect("decode empty");
        assert!(decoded.is_empty());
    }

    #[test]
    fn tokenizer_state_roundtrip_basic() {
        let mut state = TokenizerState::new();
        state.vocab.insert(0, "<unk>".to_string());
        state.vocab.insert(1, "a".to_string());
        state.merges.push((0, 1, 2));

        let mut buf = Vec::new();
        state.save_to(&mut buf).expect("save should succeed");

        let mut reader = std::io::BufReader::new(buf.as_slice());
        let loaded = TokenizerState::load_from(&mut reader).expect("load should succeed");

        assert_eq!(loaded.vocab.get(&0), Some(&"<unk>".to_string()));
        assert_eq!(loaded.vocab.get(&1), Some(&"a".to_string()));
        assert_eq!(loaded.merges, vec![(0, 1, 2)]);
        // A state saved without an explicit config carries the default
        // through the round trip.
        assert_eq!(loaded.config, ConfigSnapshot::default());
    }

    // ── TOK-01 second route: config round-trips through save/load ────────

    #[test]
    fn byte_level_config_round_trips_through_save_and_load() {
        let mut state = TokenizerState::new();
        // A byte-level vocab whose real GPT-2-remapped `!`=0 would, under
        // the pre-fix `TokenizerConfig::default()` (unk=0), be silently
        // deleted on decode if `to_oxi_tokenizer` ignored the real config.
        state.vocab.insert(0, "!".to_string());
        state.vocab.insert(1, "a".to_string());
        state = state.with_config(ConfigSnapshot {
            byte_level_decode: true,
            unk_token_id: 999, // deliberately far from any real vocab id
            ..ConfigSnapshot::default()
        });

        let mut buf = Vec::new();
        state.save_to(&mut buf).expect("save should succeed");
        let mut reader = std::io::BufReader::new(buf.as_slice());
        let loaded = TokenizerState::load_from(&mut reader).expect("load should succeed");

        assert!(loaded.config.byte_level_decode);
        assert_eq!(loaded.config.unk_token_id, 999);

        let tok = loaded.to_oxi_tokenizer();
        assert!(
            tok.config().byte_level_decode,
            "byte_level_decode must survive the round trip"
        );
        // '!' (id 0) must decode normally: it does not collide with the
        // real (relocated) unk id 999, and is not itself special.
        let decoded = tok.decode(&[0]).expect("decode should succeed");
        assert_eq!(decoded, "!");
    }

    #[test]
    fn config_snapshot_defaults_match_tokenizer_config_defaults() {
        // `ConfigSnapshot::default()` must mirror
        // `TokenizerConfig::default()` field-for-field, or an old
        // (pre-config-lines) saved file would silently change behaviour
        // when loaded by this version of the crate.
        let snapshot = ConfigSnapshot::default();
        let config = crate::tokenizer::TokenizerConfig::default();
        assert_eq!(snapshot.byte_level_decode, config.byte_level_decode);
        assert_eq!(snapshot.add_bos, config.add_bos);
        assert_eq!(snapshot.add_eos, config.add_eos);
        assert_eq!(snapshot.bos_token_id, config.bos_token_id);
        assert_eq!(snapshot.eos_token_id, config.eos_token_id);
        assert_eq!(snapshot.unk_token_id, config.unk_token_id);
        assert_eq!(snapshot.pad_token_id, config.pad_token_id);
    }

    #[test]
    fn pre_config_lines_file_format_still_loads_with_defaults() {
        // A file saved by a version of this crate that predates the
        // config-snapshot fix has no "config ..." lines at all -- only
        // "special ..." lines (or none). `load_from` must still parse it
        // cleanly, defaulting the config rather than erroring.
        let mut old_format = String::new();
        old_format.push_str(&format!("{FORMAT_MAGIC}\n"));
        old_format.push_str("vocab_size 1\n");
        old_format.push_str("merges 0\n");
        old_format.push_str(&format!("tok_id 0 {}\n", base64_encode(b"a")));
        old_format.push_str(&format!("special {} 0\n", base64_encode(b"<unk>")));

        let mut reader = std::io::BufReader::new(old_format.as_bytes());
        let loaded = TokenizerState::load_from(&mut reader).expect("legacy format should load");
        assert_eq!(loaded.config, ConfigSnapshot::default());
        assert_eq!(loaded.special_tokens.get("<unk>"), Some(&0));
    }

    #[test]
    fn unknown_config_key_is_a_parse_error() {
        let mut old_format = String::new();
        old_format.push_str(&format!("{FORMAT_MAGIC}\n"));
        old_format.push_str("vocab_size 0\n");
        old_format.push_str("merges 0\n");
        old_format.push_str("config not_a_real_field true\n");

        let mut reader = std::io::BufReader::new(old_format.as_bytes());
        let err = match TokenizerState::load_from(&mut reader) {
            Err(e) => e,
            Ok(_) => panic!("expected a parse error for an unknown config key"),
        };
        assert!(matches!(err, SerializationError::ParseError { .. }));
    }

    #[test]
    fn with_config_is_a_builder_that_returns_self() {
        let state = TokenizerState::new().with_config(ConfigSnapshot {
            add_bos: true,
            ..ConfigSnapshot::default()
        });
        assert!(state.config.add_bos);
    }
}
