//! `tokenizer.ggml.*` metadata: the write side of the embedded tokenizer.
//!
//! Every GGUF this crate used to convert contained **no tokenizer at all**
//! (CQ-08), so a converted model could only be run by also shipping a
//! side-car `tokenizer.json` and hoping the caller picked the matching one.
//! Real GGUFs — including all six shipped Bonsai 2 27 B files — embed the
//! whole vocabulary:
//!
//! ```text
//! tokenizer.ggml.model             "gpt2"
//! tokenizer.ggml.pre               "qwen35"
//! tokenizer.ggml.tokens            arr[str]  (248320 entries)
//! tokenizer.ggml.token_type        arr[i32]  (248320 entries)
//! tokenizer.ggml.merges            arr[str]  (247587 entries, "a b")
//! tokenizer.ggml.bos_token_id      u32
//! tokenizer.ggml.eos_token_id      u32
//! tokenizer.ggml.padding_token_id  u32
//! tokenizer.ggml.add_bos_token     bool
//! tokenizer.chat_template          str
//! ```
//!
//! This module reads a HuggingFace tokenizer directory (`tokenizer.json` plus
//! the optional `tokenizer_config.json` / `special_tokens_map.json`) and emits
//! exactly that block. It deliberately does **not** depend on
//! `oxibonsai-tokenizer`: the write side only needs the vocabulary as data,
//! and `oxibonsai-model` must not grow a tokenizer dependency to convert a
//! model.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};
use serde_json::Value;

/// GGUF token-type codes (`llama_token_attr` / `gguf_token_type`).
pub mod token_type {
    /// Undefined / unset.
    pub const UNDEFINED: i32 = 0;
    /// An ordinary vocabulary entry.
    pub const NORMAL: i32 = 1;
    /// The unknown-token entry.
    pub const UNKNOWN: i32 = 2;
    /// A control token (`<|im_start|>`, `<|endoftext|>`, …).
    pub const CONTROL: i32 = 3;
    /// A user-defined added token that is not a control token.
    pub const USER_DEFINED: i32 = 4;
    /// A byte-fallback entry.
    pub const BYTE: i32 = 6;
}

/// Metadata keys this module writes.
pub mod keys {
    /// Tokenizer family (`"gpt2"` for byte-level BPE).
    pub const MODEL: &str = "tokenizer.ggml.model";
    /// Pre-tokenizer regex identifier (`"qwen2"`, `"qwen35"`, …).
    pub const PRE: &str = "tokenizer.ggml.pre";
    /// Vocabulary, indexed by token id.
    pub const TOKENS: &str = "tokenizer.ggml.tokens";
    /// Per-token type codes, parallel to [`TOKENS`].
    pub const TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
    /// BPE merge rules, `"left right"` per entry.
    pub const MERGES: &str = "tokenizer.ggml.merges";
    /// Beginning-of-sequence token id.
    pub const BOS_TOKEN_ID: &str = "tokenizer.ggml.bos_token_id";
    /// End-of-sequence token id.
    pub const EOS_TOKEN_ID: &str = "tokenizer.ggml.eos_token_id";
    /// Padding token id.
    pub const PADDING_TOKEN_ID: &str = "tokenizer.ggml.padding_token_id";
    /// Unknown token id.
    pub const UNKNOWN_TOKEN_ID: &str = "tokenizer.ggml.unknown_token_id";
    /// Whether BOS is prepended automatically.
    pub const ADD_BOS_TOKEN: &str = "tokenizer.ggml.add_bos_token";
    /// Whether EOS is appended automatically.
    pub const ADD_EOS_TOKEN: &str = "tokenizer.ggml.add_eos_token";
    /// Jinja chat template.
    pub const CHAT_TEMPLATE: &str = "tokenizer.chat_template";
}

/// Errors raised while reading a HuggingFace tokenizer directory.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerMetaError {
    /// `tokenizer.json` could not be read.
    #[error("reading tokenizer at {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// `tokenizer.json` is not valid JSON.
    #[error("parsing {path}: {source}")]
    Json {
        /// The file that could not be parsed.
        path: PathBuf,
        /// The underlying parse error.
        #[source]
        source: serde_json::Error,
    },

    /// `tokenizer.json` does not have the byte-level-BPE shape this writer
    /// understands.
    #[error("tokenizer at {path}: {reason}")]
    Unsupported {
        /// The file that was rejected.
        path: PathBuf,
        /// Why it was rejected.
        reason: String,
    },
}

/// A tokenizer vocabulary ready to be written into a GGUF file.
#[derive(Debug, Clone, Default)]
pub struct TokenizerMetadata {
    /// `tokenizer.ggml.model` — `"gpt2"` for byte-level BPE.
    pub model: String,
    /// `tokenizer.ggml.pre` — the pre-tokenizer regex identifier.
    pub pre: String,
    /// Vocabulary indexed by token id; every id in `0..len` is present.
    pub tokens: Vec<String>,
    /// Per-token type codes, parallel to `tokens`.
    pub token_types: Vec<i32>,
    /// BPE merges in `"left right"` form.
    pub merges: Vec<String>,
    /// Beginning-of-sequence token id.
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token id.
    pub eos_token_id: Option<u32>,
    /// Padding token id.
    pub padding_token_id: Option<u32>,
    /// Unknown token id.
    pub unknown_token_id: Option<u32>,
    /// Whether BOS is prepended automatically.
    pub add_bos_token: Option<bool>,
    /// Whether EOS is appended automatically.
    pub add_eos_token: Option<bool>,
    /// Jinja chat template.
    pub chat_template: Option<String>,
}

impl TokenizerMetadata {
    /// Number of vocabulary entries.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Whether anything was actually loaded.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Locate and load a tokenizer next to a model.
    ///
    /// Searches `dir`, then its parent and grandparent (the ONNX layout puts
    /// `model.onnx` two levels below the repository root). Returns `Ok(None)`
    /// when no `tokenizer.json` exists anywhere in that chain — a model
    /// without a tokenizer is unusual but not an error the converter should
    /// invent.
    pub fn discover(dir: &Path, pre: &str) -> Result<Option<Self>, TokenizerMetaError> {
        let mut candidate = Some(dir);
        for _ in 0..3 {
            let Some(d) = candidate else { break };
            let path = d.join("tokenizer.json");
            if path.is_file() {
                return Self::from_dir(d, pre).map(Some);
            }
            candidate = d.parent();
        }
        Ok(None)
    }

    /// Load `tokenizer.json` (plus the optional sibling config files) from
    /// `dir`.
    pub fn from_dir(dir: &Path, pre: &str) -> Result<Self, TokenizerMetaError> {
        let tokenizer_path = dir.join("tokenizer.json");
        let root = read_json(&tokenizer_path)?;
        let mut meta = Self::from_tokenizer_json(&root, &tokenizer_path, pre)?;

        let config_path = dir.join("tokenizer_config.json");
        if config_path.is_file() {
            let config = read_json(&config_path)?;
            meta.apply_tokenizer_config(&config);
        }

        let special_path = dir.join("special_tokens_map.json");
        if special_path.is_file() {
            let special = read_json(&special_path)?;
            meta.apply_special_tokens_map(&special);
        }

        Ok(meta)
    }

    /// Build the vocabulary from an already-parsed `tokenizer.json`.
    pub fn from_tokenizer_json(
        root: &Value,
        path: &Path,
        pre: &str,
    ) -> Result<Self, TokenizerMetaError> {
        let unsupported = |reason: &str| TokenizerMetaError::Unsupported {
            path: path.to_path_buf(),
            reason: reason.to_string(),
        };

        let model = root
            .get("model")
            .and_then(Value::as_object)
            .ok_or_else(|| unsupported("missing `model` object"))?;

        let model_type = model.get("type").and_then(Value::as_str).unwrap_or("BPE");
        if !model_type.eq_ignore_ascii_case("bpe") {
            return Err(unsupported(&format!(
                "only byte-level BPE tokenizers can be embedded, found `{model_type}`"
            )));
        }

        let vocab = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| unsupported("missing `model.vocab` object"))?;

        // `vocab` is token -> id; invert it into an id-indexed table. A
        // BTreeMap keyed by id keeps the inversion deterministic and makes a
        // duplicate id detectable rather than silently last-wins.
        let mut by_id: BTreeMap<u64, String> = BTreeMap::new();
        for (token, id) in vocab {
            let id = id
                .as_u64()
                .ok_or_else(|| unsupported("a `model.vocab` id is not an integer"))?;
            if by_id.insert(id, token.clone()).is_some() {
                return Err(unsupported(&format!("duplicate token id {id} in vocab")));
            }
        }

        // Added tokens can extend past the base vocabulary.
        let added = root
            .get("added_tokens")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let mut special_ids: BTreeMap<u64, bool> = BTreeMap::new();
        for entry in added {
            let Some(id) = entry.get("id").and_then(Value::as_u64) else {
                continue;
            };
            let Some(content) = entry.get("content").and_then(Value::as_str) else {
                continue;
            };
            let is_special = entry
                .get("special")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            by_id.insert(id, content.to_string());
            special_ids.insert(id, is_special);
        }

        let max_id = by_id.keys().copied().next_back().unwrap_or(0);
        let vocab_len = usize::try_from(max_id)
            .map_err(|_| unsupported("token id exceeds the addressable range"))?
            .saturating_add(1);

        // A GGUF vocabulary is a dense array, so every hole has to be filled
        // with an explicit unused placeholder rather than shifting ids.
        let mut tokens = vec![String::new(); vocab_len];
        let mut token_types = vec![token_type::UNDEFINED; vocab_len];
        for (id, token) in by_id {
            let idx = id as usize;
            let is_special = special_ids.get(&id).copied().unwrap_or(false);
            token_types[idx] = if is_special {
                token_type::CONTROL
            } else if special_ids.contains_key(&id) {
                token_type::USER_DEFINED
            } else {
                token_type::NORMAL
            };
            tokens[idx] = token;
        }
        for (idx, slot) in tokens.iter_mut().enumerate() {
            if slot.is_empty() && token_types[idx] == token_type::UNDEFINED {
                *slot = format!("[PAD{idx}]");
                token_types[idx] = token_type::USER_DEFINED;
            }
        }

        let merges = parse_merges(model.get("merges"))
            .ok_or_else(|| unsupported("missing or malformed `model.merges`"))?;

        Ok(Self {
            model: "gpt2".to_string(),
            pre: pre.to_string(),
            tokens,
            token_types,
            merges,
            ..Default::default()
        })
    }

    /// Apply `tokenizer_config.json` (special tokens, flags, chat template).
    pub fn apply_tokenizer_config(&mut self, config: &Value) {
        self.bos_token_id = self
            .lookup_special(config.get("bos_token"))
            .or(self.bos_token_id);
        self.eos_token_id = self
            .lookup_special(config.get("eos_token"))
            .or(self.eos_token_id);
        self.padding_token_id = self
            .lookup_special(config.get("pad_token"))
            .or(self.padding_token_id);
        self.unknown_token_id = self
            .lookup_special(config.get("unk_token"))
            .or(self.unknown_token_id);

        if let Some(v) = config.get("add_bos_token").and_then(Value::as_bool) {
            self.add_bos_token = Some(v);
        }
        if let Some(v) = config.get("add_eos_token").and_then(Value::as_bool) {
            self.add_eos_token = Some(v);
        }
        if let Some(t) = extract_chat_template(config.get("chat_template")) {
            self.chat_template = Some(t);
        }
    }

    /// Apply `special_tokens_map.json` for fields the config did not set.
    pub fn apply_special_tokens_map(&mut self, special: &Value) {
        if self.bos_token_id.is_none() {
            self.bos_token_id = self.lookup_special(special.get("bos_token"));
        }
        if self.eos_token_id.is_none() {
            self.eos_token_id = self.lookup_special(special.get("eos_token"));
        }
        if self.padding_token_id.is_none() {
            self.padding_token_id = self.lookup_special(special.get("pad_token"));
        }
        if self.unknown_token_id.is_none() {
            self.unknown_token_id = self.lookup_special(special.get("unk_token"));
        }
    }

    /// Resolve a HuggingFace special-token entry (a bare string, or an
    /// `AddedToken` object with a `content` field) to a vocabulary id.
    fn lookup_special(&self, entry: Option<&Value>) -> Option<u32> {
        let text = match entry? {
            Value::String(s) => s.as_str(),
            Value::Object(o) => o.get("content")?.as_str()?,
            _ => return None,
        };
        self.token_id(text)
    }

    /// Look up a token's id by its exact text.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.tokens
            .iter()
            .position(|t| t == text)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Write the `tokenizer.*` block.
    ///
    /// Keys whose value is unknown are omitted rather than defaulted: a wrong
    /// `eos_token_id` makes a model generate forever, which is worse than an
    /// absent one the loader can complain about.
    pub fn write(&self, writer: &mut GgufWriter<'_>) {
        writer.add_metadata(keys::MODEL, MetadataWriteValue::Str(self.model.clone()));
        if !self.pre.is_empty() {
            writer.add_metadata(keys::PRE, MetadataWriteValue::Str(self.pre.clone()));
        }
        writer.add_metadata(
            keys::TOKENS,
            MetadataWriteValue::ArrayStr(self.tokens.clone()),
        );
        writer.add_metadata(
            keys::TOKEN_TYPE,
            MetadataWriteValue::ArrayI32(self.token_types.clone()),
        );
        if !self.merges.is_empty() {
            writer.add_metadata(
                keys::MERGES,
                MetadataWriteValue::ArrayStr(self.merges.clone()),
            );
        }
        for (key, id) in [
            (keys::BOS_TOKEN_ID, self.bos_token_id),
            (keys::EOS_TOKEN_ID, self.eos_token_id),
            (keys::PADDING_TOKEN_ID, self.padding_token_id),
            (keys::UNKNOWN_TOKEN_ID, self.unknown_token_id),
        ] {
            if let Some(v) = id {
                writer.add_metadata(key, MetadataWriteValue::U32(v));
            }
        }
        for (key, flag) in [
            (keys::ADD_BOS_TOKEN, self.add_bos_token),
            (keys::ADD_EOS_TOKEN, self.add_eos_token),
        ] {
            if let Some(v) = flag {
                writer.add_metadata(key, MetadataWriteValue::Bool(v));
            }
        }
        if let Some(ref template) = self.chat_template {
            writer.add_metadata(
                keys::CHAT_TEMPLATE,
                MetadataWriteValue::Str(template.clone()),
            );
        }
    }
}

/// The `tokenizer.ggml.pre` identifier for a GGUF architecture.
///
/// `qwen35` adds `\p{M}` (combining marks) in two places relative to `qwen2`;
/// every other Qwen3-family model uses the `qwen2` regex.
pub fn pre_tokenizer_for_arch(arch: &str) -> &'static str {
    match arch {
        "qwen35" => "qwen35",
        _ => "qwen2",
    }
}

/// Parse `model.merges`, accepting both `tokenizers` encodings: the classic
/// `"left right"` strings and the newer `["left", "right"]` pairs.
fn parse_merges(value: Option<&Value>) -> Option<Vec<String>> {
    let array = value?.as_array()?;
    let mut out = Vec::with_capacity(array.len());
    for entry in array {
        match entry {
            Value::String(s) => out.push(s.clone()),
            Value::Array(pair) if pair.len() == 2 => {
                let left = pair[0].as_str()?;
                let right = pair[1].as_str()?;
                out.push(format!("{left} {right}"));
            }
            _ => return None,
        }
    }
    Some(out)
}

/// Extract a chat template, accepting both the plain-string form and the
/// list-of-named-templates form (where `"default"` wins).
fn extract_chat_template(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Array(entries) => {
            let mut fallback: Option<String> = None;
            for entry in entries {
                let name = entry.get("name").and_then(Value::as_str).unwrap_or("");
                let template = entry.get("template").and_then(Value::as_str)?;
                if name == "default" {
                    return Some(template.to_string());
                }
                fallback.get_or_insert_with(|| template.to_string());
            }
            fallback
        }
        _ => None,
    }
}

fn read_json(path: &Path) -> Result<Value, TokenizerMetaError> {
    let raw = std::fs::read_to_string(path).map_err(|source| TokenizerMetaError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| TokenizerMetaError::Json {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::reader::GgufFile;
    use serde_json::json;

    fn tokenizer_json() -> Value {
        json!({
            "model": {
                "type": "BPE",
                "vocab": { "a": 0, "b": 1, "ab": 2 },
                "merges": ["a b"],
            },
            "added_tokens": [
                { "id": 3, "content": "<|endoftext|>", "special": true },
                { "id": 4, "content": "<|im_end|>", "special": true },
            ],
        })
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "oxibonsai_tokmeta_{tag}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn vocab_is_dense_and_ordered_by_id() {
        let meta = TokenizerMetadata::from_tokenizer_json(
            &tokenizer_json(),
            Path::new("tokenizer.json"),
            "qwen2",
        )
        .expect("parse");
        assert_eq!(
            meta.tokens,
            vec!["a", "b", "ab", "<|endoftext|>", "<|im_end|>"]
        );
        assert_eq!(
            meta.token_types,
            vec![
                token_type::NORMAL,
                token_type::NORMAL,
                token_type::NORMAL,
                token_type::CONTROL,
                token_type::CONTROL
            ]
        );
        assert_eq!(meta.merges, vec!["a b"]);
        assert_eq!(meta.vocab_size(), 5);
    }

    #[test]
    fn pair_form_merges_are_accepted() {
        let mut root = tokenizer_json();
        root["model"]["merges"] = json!([["a", "b"], ["ab", "a"]]);
        let meta =
            TokenizerMetadata::from_tokenizer_json(&root, Path::new("tokenizer.json"), "qwen2")
                .expect("parse");
        assert_eq!(meta.merges, vec!["a b", "ab a"]);
    }

    #[test]
    fn gaps_in_the_id_space_are_filled_not_shifted() {
        let mut root = tokenizer_json();
        root["added_tokens"] = json!([{ "id": 6, "content": "<|im_end|>", "special": true }]);
        let meta =
            TokenizerMetadata::from_tokenizer_json(&root, Path::new("tokenizer.json"), "qwen2")
                .expect("parse");
        assert_eq!(meta.vocab_size(), 7);
        assert_eq!(meta.tokens[6], "<|im_end|>");
        // ids 3..6 were absent — placeholders keep the array dense so token 6
        // still resolves to id 6.
        assert_eq!(meta.token_types[4], token_type::USER_DEFINED);
    }

    #[test]
    fn non_bpe_tokenizer_is_refused() {
        let root = json!({ "model": { "type": "Unigram", "vocab": {} } });
        let err =
            TokenizerMetadata::from_tokenizer_json(&root, Path::new("tokenizer.json"), "qwen2")
                .expect_err("must refuse");
        assert!(matches!(err, TokenizerMetaError::Unsupported { .. }));
    }

    #[test]
    fn config_resolves_special_token_ids_and_template() {
        let mut meta = TokenizerMetadata::from_tokenizer_json(
            &tokenizer_json(),
            Path::new("tokenizer.json"),
            "qwen35",
        )
        .expect("parse");
        meta.apply_tokenizer_config(&json!({
            "bos_token": "<|endoftext|>",
            "eos_token": { "content": "<|im_end|>" },
            "add_bos_token": false,
            "chat_template": "{{ bos_token }}",
        }));
        assert_eq!(meta.bos_token_id, Some(3));
        assert_eq!(meta.eos_token_id, Some(4));
        assert_eq!(meta.add_bos_token, Some(false));
        assert_eq!(meta.chat_template.as_deref(), Some("{{ bos_token }}"));
    }

    #[test]
    fn list_form_chat_template_prefers_default() {
        assert_eq!(
            extract_chat_template(Some(&json!([
                {"name": "tool_use", "template": "T"},
                {"name": "default", "template": "D"},
            ]))),
            Some("D".to_string())
        );
    }

    #[test]
    fn round_trips_through_a_real_gguf() {
        let dir = scratch_dir("roundtrip");
        std::fs::create_dir_all(&dir).expect("create dir");
        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string(&tokenizer_json()).expect("serialise"),
        )
        .expect("write tokenizer.json");
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_string(&json!({
                "bos_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
                "add_bos_token": false,
                "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
            }))
            .expect("serialise"),
        )
        .expect("write tokenizer_config.json");

        let meta = TokenizerMetadata::discover(&dir, "qwen35")
            .expect("discover")
            .expect("tokenizer present");
        let _ = std::fs::remove_dir_all(&dir);

        let mut w = GgufWriter::new();
        meta.write(&mut w);
        let bytes = w.to_bytes().expect("serialise gguf");
        let gguf = GgufFile::parse(&bytes).expect("parse gguf");

        assert_eq!(gguf.metadata.get_string(keys::MODEL).ok(), Some("gpt2"));
        assert_eq!(gguf.metadata.get_string(keys::PRE).ok(), Some("qwen35"));
        assert_eq!(gguf.metadata.get_u32(keys::EOS_TOKEN_ID).ok(), Some(4));
        assert_eq!(gguf.metadata.get_u32(keys::BOS_TOKEN_ID).ok(), Some(3));
        let tokens = gguf
            .metadata
            .get_string_array(keys::TOKENS)
            .expect("tokens array");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[4], "<|im_end|>");
        assert!(gguf.metadata.get_string(keys::CHAT_TEMPLATE).is_ok());
    }

    #[test]
    fn discover_returns_none_without_a_tokenizer() {
        let dir = scratch_dir("absent");
        std::fs::create_dir_all(&dir).expect("create dir");
        let found = TokenizerMetadata::discover(&dir, "qwen2").expect("discover");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(found.is_none());
    }

    #[test]
    fn pre_tokenizer_mapping() {
        assert_eq!(pre_tokenizer_for_arch("qwen35"), "qwen35");
        assert_eq!(pre_tokenizer_for_arch("qwen3"), "qwen2");
    }
}
