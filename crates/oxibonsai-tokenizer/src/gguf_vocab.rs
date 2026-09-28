//! GGUF-embedded tokenizer loader (TOK-05).
//!
//! Bonsai 2 27B (and any other `qwen35`/`gpt2`-pre GGUF checkpoint) ships
//! its vocabulary, BPE merges, special-token ids and chat template *only*
//! inside the GGUF's own key-value metadata — there is no accompanying
//! `tokenizer.json`. This module reads `tokenizer.ggml.*` through
//! [`oxibonsai_core::MetadataStore`]'s typed accessors and builds the exact
//! same kind of [`crate::OxiTokenizer`] the `tokenizer.json` path
//! ([`crate::hf_format::HfTokenizerJson::into_tokenizer`]) produces.
//!
//! Keys read (verified against the real Bonsai 2 27B GGUF headers):
//!
//! | key | GGUF type | required |
//! |---|---|---|
//! | `tokenizer.ggml.model` | string | yes (must be `"gpt2"`, i.e. byte-level BPE) |
//! | `tokenizer.ggml.pre` | string | no (default: `"gpt2"`) |
//! | `tokenizer.ggml.tokens` | array\<string\> | yes |
//! | `tokenizer.ggml.merges` | array\<string\> (`"left right"`) | no |
//! | `tokenizer.ggml.token_type` | array\<int32\> | no |
//! | `tokenizer.ggml.bos_token_id` | uint32 | no |
//! | `tokenizer.ggml.eos_token_id` | uint32 | yes |
//! | `tokenizer.ggml.unknown_token_id` | uint32 | no |
//! | `tokenizer.ggml.padding_token_id` | uint32 | no |
//! | `tokenizer.ggml.add_bos_token` | bool | no (default `false`) |
//! | `tokenizer.ggml.add_eos_token` | bool | no (default `false`) |
//! | `tokenizer.chat_template` | string | no |
//!
//! Only `tokenizer.ggml.model = "gpt2"` (byte-level BPE) is implemented —
//! the only model type Bonsai 2 and the rest of this crate's target models
//! use; a SentencePiece/Unigram GGUF vocab is out of this finding's scope
//! and errors clearly rather than silently mis-loading.

use oxibonsai_core::MetadataStore;

use crate::{
    bpe::{BpeMerges, PreTokenizerKind},
    error::{TokenizerError, TokenizerResult},
    tokenizer::{OxiTokenizer, TokenizerConfig},
    vocab::Vocabulary,
};

// ── Metadata keys ────────────────────────────────────────────────────────────

pub const KEY_MODEL: &str = "tokenizer.ggml.model";
pub const KEY_PRE: &str = "tokenizer.ggml.pre";
pub const KEY_TOKENS: &str = "tokenizer.ggml.tokens";
pub const KEY_MERGES: &str = "tokenizer.ggml.merges";
pub const KEY_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
pub const KEY_BOS_TOKEN_ID: &str = "tokenizer.ggml.bos_token_id";
pub const KEY_EOS_TOKEN_ID: &str = "tokenizer.ggml.eos_token_id";
pub const KEY_UNK_TOKEN_ID: &str = "tokenizer.ggml.unknown_token_id";
pub const KEY_PAD_TOKEN_ID: &str = "tokenizer.ggml.padding_token_id";
pub const KEY_ADD_BOS_TOKEN: &str = "tokenizer.ggml.add_bos_token";
pub const KEY_ADD_EOS_TOKEN: &str = "tokenizer.ggml.add_eos_token";
pub const KEY_CHAT_TEMPLATE: &str = "tokenizer.chat_template";

/// `llama.cpp`'s `LLAMA_TOKEN_TYPE_*` values, as read from the (optional)
/// `tokenizer.ggml.token_type` int32 array.
///
/// Verified against the real Bonsai 2 27B histogram (TOK-05 verdict
/// correction): `{NORMAL: 248044, UNUSED: 243, CONTROL: 27, USER_DEFINED:
/// 6}` — no `BYTE` entries, consistent with `model = "gpt2"` (byte-level;
/// no SentencePiece-style `<0xHH>` synthetic byte tokens exist in this
/// vocab). The 27 `CONTROL` ids are the `<|...|>`-bracketed markers
/// (`<|im_start|>`, `<|endoftext|>`, ...) and the 6 `USER_DEFINED` ids are
/// the single-angle-bracket reasoning/tool markers (`<think>`, `</think>`,
/// `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`) —
/// genuine content the model emits, not decode-time noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufTokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl GgufTokenType {
    fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => return None,
        })
    }
}

/// Return `true` if `token` has the conventional special-token marker shape
/// (`<...>` or `<|...|>`) rather than being ordinary vocabulary content.
///
/// Deliberately excludes the `<0xHH>` byte-fallback shape (see
/// [`crate::bpe::byte_fallback_id`]): those are ordinary, always-decodable
/// raw-byte tokens and must never be promoted to special even if a
/// (malformed) GGUF's declared bos/eos/unk/pad id happens to coincide with
/// one. Used only as the narrow gate for the `token_type`-absent defensive
/// fallback in [`tokenizer_from_gguf_metadata`] — mirrors the existing
/// `<...>` special-token heuristic in [`crate::vocab::Vocabulary::from_json`].
fn looks_like_special_marker(token: &str) -> bool {
    let is_bracketed = token.len() > 2 && token.starts_with('<') && token.ends_with('>');
    is_bracketed && !is_byte_fallback_shape(token)
}

/// `true` for the exact `<0xHH>` shape [`crate::bpe::byte_fallback_id`]
/// produces (two ASCII hex digits, either case — matching the tolerance of
/// `u8::from_str_radix(_, 16)`, which is how it is parsed back on decode).
fn is_byte_fallback_shape(token: &str) -> bool {
    token
        .strip_prefix("<0x")
        .and_then(|s| s.strip_suffix('>'))
        .is_some_and(|inner| inner.len() == 2 && inner.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Build an [`OxiTokenizer`] directly from a GGUF file's metadata store.
///
/// See the module docs for the key set. Every failure is surfaced through
/// the dedicated [`TokenizerError::GgufFormat`] variant (`TOK-17`): a
/// missing or mistyped `tokenizer.ggml.*` key, an unsupported model type,
/// or a malformed entry — the GGUF-container counterpart of
/// [`TokenizerError::HfFormat`], which stays reserved for a malformed
/// `tokenizer.json`, so a caller can tell the two sources apart by variant
/// rather than by a message prefix.
///
/// # Errors
/// Returns `Err` if `tokenizer.ggml.model` is missing, is not `"gpt2"`,
/// `tokenizer.ggml.tokens` is missing/empty, `tokenizer.ggml.eos_token_id`
/// is missing, `tokenizer.ggml.token_type`'s length disagrees with
/// `tokenizer.ggml.tokens`' length, or a `tokenizer.ggml.merges` entry is
/// not of the form `"left right"`.
pub fn tokenizer_from_gguf_metadata(md: &MetadataStore) -> TokenizerResult<OxiTokenizer> {
    let model = md
        .get_string(KEY_MODEL)
        .map_err(|e| TokenizerError::GgufFormat(e.to_string()))?;
    if model != "gpt2" {
        return Err(TokenizerError::GgufFormat(format!(
            "unsupported tokenizer.ggml.model {model:?} (only the byte-level \"gpt2\" BPE \
             vocab is implemented)"
        )));
    }

    let tokens = md
        .get_string_array(KEY_TOKENS)
        .map_err(|e| TokenizerError::GgufFormat(e.to_string()))?;
    if tokens.is_empty() {
        return Err(TokenizerError::GgufFormat(
            "tokenizer.ggml.tokens is empty".to_owned(),
        ));
    }

    let token_types: Option<Vec<i32>> = md.get_i32_array(KEY_TOKEN_TYPE).ok();
    if let Some(types) = &token_types {
        if types.len() != tokens.len() {
            return Err(TokenizerError::GgufFormat(format!(
                "tokenizer.ggml.token_type has {} entries, expected {} (one per \
                 tokenizer.ggml.tokens entry)",
                types.len(),
                tokens.len()
            )));
        }
    }

    // Read the special-token ids up front (pure metadata reads, independent
    // of vocabulary construction) so the vocabulary loop below can consult
    // them for the `token_type`-absent fallback immediately below.
    let bos_id = md.get(KEY_BOS_TOKEN_ID).and_then(|v| v.as_u32());
    let eos_id = md
        .get_u32(KEY_EOS_TOKEN_ID)
        .map_err(|e| TokenizerError::GgufFormat(e.to_string()))?;
    let unk_id = md.get(KEY_UNK_TOKEN_ID).and_then(|v| v.as_u32());
    let pad_id = md.get(KEY_PAD_TOKEN_ID).and_then(|v| v.as_u32());

    // ── Vocabulary ──────────────────────────────────────────────────────
    //
    // Defensive fallback (minor finding, wave-2 B2-08 review): when
    // `tokenizer.ggml.token_type` is absent entirely (the real Bonsai 2 27B
    // GGUF always ships it — verified CONTROL=27/USER_DEFINED=6 — but a
    // hand-rolled or otherwise atypical GGUF might not), a declared
    // bos/eos/unk/pad id would otherwise land as a perfectly ordinary
    // `vocab.insert` entry: `OxiTokenizer::new`'s `build_special_ids` only
    // trusts a config id when the vocabulary *disagrees* that it is
    // ordinary content (the TOK-01 fix), so an id the vocabulary confirms
    // is present-and-plain is correctly left alone by that generic
    // post-hoc guard — and would render verbatim on decode instead of
    // being skipped.
    //
    // Recovered here, narrowly: when `token_type` is absent AND the
    // specific token text at a declared special id looks like a
    // conventional marker (`<...>`/`<|...|>`, never the `<0xHH>`
    // byte-fallback shape or an ordinary word), mark it via `add_special`
    // where we have strictly more information than the generic
    // `build_special_ids` guard — a dedicated, unambiguous GGUF key, not a
    // coincidental default the way TOK-01's `unk=0/bos=1/eos=2/pad=3` was.
    // Deliberately narrow: promoting an arbitrary *plain-word* token to
    // special (and therefore into the atomic encode-time carve-out set)
    // on nothing but an id coincidence would reproduce a weaker form of
    // the exact bug TOK-01 fixed, so an ordinary-looking token at a
    // declared id is left alone rather than guessed at.
    let token_type_absent = token_types.is_none();
    let fallback_special_ids = [bos_id, Some(eos_id), unk_id, pad_id];

    let mut vocab = Vocabulary::new();
    for (idx, token) in tokens.iter().enumerate() {
        let id = u32::try_from(idx).map_err(|_| {
            TokenizerError::GgufFormat(
                "tokenizer.ggml.tokens has more entries than fit in a u32".to_owned(),
            )
        })?;
        let ty = token_types
            .as_ref()
            .and_then(|types| types.get(idx))
            .copied()
            .and_then(GgufTokenType::from_i32);
        let is_fallback_special = token_type_absent
            && fallback_special_ids.contains(&Some(id))
            && looks_like_special_marker(token);
        match ty {
            // CONTROL: genuinely special — carved out atomically on encode
            // AND skipped on decode (e.g. `<|im_start|>`, `<|endoftext|>`).
            Some(GgufTokenType::Control) => vocab.add_special(token, id),
            // USER_DEFINED: carved out atomically on encode (so
            // `<think>` / `<tool_call>` map to their trained ids rather
            // than being shredded by pre-tokenization) but NOT skipped on
            // decode — this is genuine model output a caller needs to see
            // (the reasoning splitter, the tool-call parser).
            Some(GgufTokenType::UserDefined) => vocab.add_protected(token, id),
            // No token_type array at all, and this exact id is a declared
            // bos/eos/unk/pad marker-shaped token: see the fallback comment
            // above.
            _ if is_fallback_special => vocab.add_special(token, id),
            // NORMAL / UNUSED / BYTE / unrecognised type / no token_type
            // array at all: an ordinary, always-decodable vocabulary entry.
            _ => vocab.insert(token, id),
        }
    }

    // ── Merges ──────────────────────────────────────────────────────────
    // GGUF stores each merge as a single "left right" string — the same
    // shape `tokenizer.json`'s string-form `model.merges` entries use.
    let mut merges = BpeMerges::new();
    if let Ok(raw_merges) = md.get_string_array(KEY_MERGES) {
        for (idx, entry) in raw_merges.iter().enumerate() {
            let Some((a, b)) = entry.split_once(' ') else {
                return Err(TokenizerError::GgufFormat(format!(
                    "malformed tokenizer.ggml.merges entry #{idx}: {entry:?} (expected \
                     \"left right\")"
                )));
            };
            let merged = format!("{a}{b}");
            // Mirrors `HfTokenizerJson::into_tokenizer`'s BPE branch: a
            // merge whose result never made it into the final vocab is
            // silently skipped rather than treated as an error.
            if let Some(merged_id) = vocab.get_id(&merged) {
                merges.add_merge(a, b, merged_id);
            }
        }
    }

    // ── Config ────────────────────────────────────────────────────────
    // `add_bos_token` / `add_eos_token` default to `false` when absent —
    // matching `TokenizerConfig::default()` and Bonsai 2's own explicit
    // `add_bos_token = False` (design doc Appendix A.1 / acceptance
    // criterion: "`add_bos = false` respected").
    let add_bos = md
        .get(KEY_ADD_BOS_TOKEN)
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let add_eos = md
        .get(KEY_ADD_EOS_TOKEN)
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut config = TokenizerConfig {
        byte_level_decode: true,
        add_bos,
        add_eos,
        eos_token_id: eos_id,
        ..Default::default()
    };
    if let Some(id) = bos_id {
        config.bos_token_id = id;
    }
    if let Some(id) = unk_id {
        config.unk_token_id = id;
    }
    if let Some(id) = pad_id {
        config.pad_token_id = id;
    }

    let pre = md.get_string(KEY_PRE).unwrap_or("gpt2");
    let kind = PreTokenizerKind::from_gguf_pre(pre);

    Ok(OxiTokenizer::new(vocab, merges, config).with_pretokenizer_kind(kind))
}

/// Return the model's embedded `tokenizer.chat_template`, if present.
///
/// Kept separate from [`tokenizer_from_gguf_metadata`] since the chat
/// template is consumed by a different subsystem
/// (`crate::chat_templates::ChatTemplateKind`, wired by a later package)
/// and is orthogonal to building the tokenizer itself.
pub fn chat_template_from_gguf_metadata(md: &MetadataStore) -> Option<String> {
    md.get(KEY_CHAT_TEMPLATE)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::types::GgufValueType;

    /// Build a minimal, well-formed metadata block covering every key this
    /// module reads, byte-identical in shape to the real GGUF wire format
    /// (so this also exercises `MetadataStore::parse` end-to-end).
    fn make_metadata(extra_bool_add_bos: bool) -> MetadataStore {
        fn kv_string(key: &str, value: &str) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&str_bytes(value));
            bytes
        }
        fn kv_u32(key: &str, value: u32) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes
        }
        fn kv_bool(key: &str, value: bool) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Bool as u32).to_le_bytes());
            bytes.push(u8::from(value));
            bytes
        }
        fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                bytes.extend_from_slice(&str_bytes(v));
            }
            bytes
        }
        fn kv_i32_array(key: &str, values: &[i32]) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        }
        fn str_bytes(s: &str) -> Vec<u8> {
            let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(s.as_bytes());
            bytes
        }

        // Vocab: bytes-as-chars 0..4 ("a","b","c","ab"), then two CONTROL
        // specials and one USER_DEFINED protected token.
        let tokens = [
            "a",
            "b",
            "c",
            "ab",
            "<|im_start|>",
            "<|endoftext|>",
            "<think>",
        ];
        let token_types = [1i32, 1, 1, 1, 3, 3, 4]; // NORMAL x4, CONTROL x2, USER_DEFINED x1

        let mut data = Vec::new();
        let mut count = 0u64;
        macro_rules! push {
            ($bytes:expr) => {{
                data.extend_from_slice(&$bytes);
                count += 1;
            }};
        }
        push!(kv_string(KEY_MODEL, "gpt2"));
        push!(kv_string(KEY_PRE, "qwen35"));
        push!(kv_string_array(KEY_TOKENS, &tokens));
        push!(kv_i32_array(KEY_TOKEN_TYPE, &token_types));
        push!(kv_string_array(KEY_MERGES, &["a b"]));
        push!(kv_u32(KEY_BOS_TOKEN_ID, 5)); // <|endoftext|>
        push!(kv_u32(KEY_EOS_TOKEN_ID, 5));
        push!(kv_bool(KEY_ADD_BOS_TOKEN, extra_bool_add_bos));
        push!(kv_string(KEY_CHAT_TEMPLATE, "{{ messages }}"));

        let (store, _) = MetadataStore::parse(&data, 0, count).expect("well-formed metadata block");
        store
    }

    #[test]
    fn builds_tokenizer_with_control_and_user_defined_split() {
        let md = make_metadata(false);
        let tok = tokenizer_from_gguf_metadata(&md).expect("gguf tokenizer should build");

        assert_eq!(tok.vocab_size(), 7);
        assert!(tok.config().byte_level_decode);
        assert!(
            !tok.config().add_bos,
            "add_bos_token=false must be respected"
        );
        assert_eq!(tok.bos_id(), 5);
        assert_eq!(tok.eos_id(), 5);

        // CONTROL id must be classified special (decode-skip).
        assert!(tok.is_special(5));
        // USER_DEFINED id must NOT be classified special (it must render).
        assert!(!tok.is_special(6));

        // Merge applied: "a"+"b" -> "ab" (id 3).
        let ids = tok.encode("ab").expect("encode");
        assert_eq!(ids, vec![3]);
    }

    #[test]
    fn control_token_is_skipped_user_defined_is_rendered_on_decode() {
        let md = make_metadata(false);
        let tok = tokenizer_from_gguf_metadata(&md).expect("gguf tokenizer should build");

        // id 4 = "<|im_start|>" (CONTROL) must vanish on decode.
        let decoded = tok.decode(&[4, 0]).expect("decode");
        assert_eq!(decoded, "a", "CONTROL token must be skipped on decode");

        // id 6 = "<think>" (USER_DEFINED) must render verbatim.
        let decoded2 = tok.decode(&[6, 0]).expect("decode");
        assert_eq!(
            decoded2, "<think>a",
            "USER_DEFINED token must NOT be skipped on decode"
        );
    }

    #[test]
    fn add_bos_token_true_is_respected() {
        let md = make_metadata(true);
        let tok = tokenizer_from_gguf_metadata(&md).expect("gguf tokenizer should build");
        assert!(tok.config().add_bos);
        let ids = tok.encode("a").expect("encode");
        assert_eq!(ids.first().copied(), Some(5), "BOS must be prepended");
    }

    // ── `token_type`-absent defensive fallback (minor finding, wave-2 review) ──

    /// Build a metadata block with NO `tokenizer.ggml.token_type` array at
    /// all — the scenario the fallback exists for. `bos_token_id` and
    /// `eos_token_id` both point at the bracket-shaped `"<|endoftext|>"`
    /// (id 0); `pad_token_id` points at the ORDINARY word `"b"` (id 2) —
    /// the regression guard proving the fallback does not over-promote.
    fn make_metadata_without_token_type() -> MetadataStore {
        fn kv_string(key: &str, value: &str) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&str_bytes(value));
            bytes
        }
        fn kv_u32(key: &str, value: u32) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes
        }
        fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                bytes.extend_from_slice(&str_bytes(v));
            }
            bytes
        }
        fn str_bytes(s: &str) -> Vec<u8> {
            let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(s.as_bytes());
            bytes
        }

        // [0]="<|endoftext|>" (bos+eos), [1]="a", [2]="b" (pad_id points
        // here — an ordinary word, must NOT be promoted), [3]="<|im_end|>"
        // (bracket-shaped but NOT any declared id — must stay untouched).
        let tokens = ["<|endoftext|>", "a", "b", "<|im_end|>"];

        let mut data = Vec::new();
        let mut count = 0u64;
        macro_rules! push {
            ($bytes:expr) => {{
                data.extend_from_slice(&$bytes);
                count += 1;
            }};
        }
        push!(kv_string(KEY_MODEL, "gpt2"));
        push!(kv_string_array(KEY_TOKENS, &tokens));
        // NOTE: KEY_TOKEN_TYPE is deliberately omitted.
        push!(kv_u32(KEY_BOS_TOKEN_ID, 0));
        push!(kv_u32(KEY_EOS_TOKEN_ID, 0));
        push!(kv_u32(KEY_PAD_TOKEN_ID, 2));

        let (store, _) = MetadataStore::parse(&data, 0, count).expect("well-formed metadata block");
        store
    }

    #[test]
    fn token_type_absent_fallback_promotes_bracketed_special() {
        let md = make_metadata_without_token_type();
        let tok = tokenizer_from_gguf_metadata(&md).expect("gguf tokenizer should build");

        // id 0 = "<|endoftext|>" (both bos and eos here) must be recovered
        // as special even though no token_type array exists at all.
        assert!(
            tok.is_special(0),
            "a declared bos/eos id shaped like a marker must be promoted to \
             special by the token_type-absent fallback"
        );
        let decoded = tok.decode(&[0, 1]).expect("decode");
        assert_eq!(
            decoded, "a",
            "the recovered special token must be skipped on decode"
        );
    }

    #[test]
    fn token_type_absent_fallback_does_not_promote_ordinary_word() {
        // Regression guard: `pad_token_id` (2) points at the ORDINARY word
        // "b". The fallback must NOT promote it -- doing so on nothing but
        // an id coincidence would reproduce a weaker form of the exact
        // TOK-01 bug this crate already fixed once (an ordinary content
        // token silently vanishing from decode / getting carved out of
        // encode).
        let md = make_metadata_without_token_type();
        let tok = tokenizer_from_gguf_metadata(&md).expect("gguf tokenizer should build");

        assert!(
            !tok.is_special(2),
            "an ordinary word token must not be promoted just because it \
             coincides with a declared pad_token_id"
        );
        assert_eq!(tok.decode(&[2]).expect("decode"), "b");
    }

    #[test]
    fn token_type_absent_fallback_ignores_unrelated_bracketed_token() {
        // "<|im_end|>" (id 3) is bracket-shaped but is not any of the
        // declared bos/eos/unk/pad ids in this fixture — it must be left as
        // an ordinary, decodable, non-special entry.
        let md = make_metadata_without_token_type();
        let tok = tokenizer_from_gguf_metadata(&md).expect("gguf tokenizer should build");
        assert!(!tok.is_special(3));
        assert_eq!(tok.decode(&[3]).expect("decode"), "<|im_end|>");
    }

    #[test]
    fn token_type_absent_fallback_never_promotes_byte_fallback_shaped_token() {
        // Even if a declared special id coincidentally points at a
        // `<0xHH>`-shaped byte-fallback token, the fallback must not
        // promote it — those are ordinary, always-decodable raw bytes (see
        // `crate::bpe::byte_fallback_id`), never special markers.
        fn kv_string(key: &str, value: &str) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&str_bytes(value));
            bytes
        }
        fn kv_u32(key: &str, value: u32) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes
        }
        fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                bytes.extend_from_slice(&str_bytes(v));
            }
            bytes
        }
        fn str_bytes(s: &str) -> Vec<u8> {
            let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(s.as_bytes());
            bytes
        }

        let tokens = ["<0x41>", "a"];
        let mut data = Vec::new();
        let mut count = 0u64;
        macro_rules! push {
            ($bytes:expr) => {{
                data.extend_from_slice(&$bytes);
                count += 1;
            }};
        }
        push!(kv_string(KEY_MODEL, "gpt2"));
        push!(kv_string_array(KEY_TOKENS, &tokens));
        push!(kv_u32(KEY_EOS_TOKEN_ID, 0)); // points at "<0x41>"

        let (store, _) = MetadataStore::parse(&data, 0, count).expect("well-formed metadata block");
        let tok = tokenizer_from_gguf_metadata(&store).expect("gguf tokenizer should build");

        assert!(
            !tok.is_special(0),
            "a <0xHH> byte-fallback token must never be promoted to special, \
             even if it coincidentally sits at a declared eos_token_id"
        );
        assert_eq!(tok.decode(&[0]).expect("decode"), "A");
    }

    #[test]
    fn pretokenizer_kind_is_qwen35_from_pre_metadata() {
        // Verified indirectly: qwen35 admits combining marks in a letter
        // run where the plain gpt2/qwen2 pattern would split them off. We
        // can't easily probe the private `pretokenizer_kind` field from
        // here, so this is re-verified end-to-end in `hf_parity_tests.rs`
        // instead; this test only pins that loading succeeds with `pre =
        // "qwen35"` present (the plumbing path).
        let md = make_metadata(false);
        assert!(tokenizer_from_gguf_metadata(&md).is_ok());
    }

    #[test]
    fn missing_model_key_errors() {
        let store = MetadataStore::new();
        match tokenizer_from_gguf_metadata(&store) {
            Err(TokenizerError::GgufFormat(msg)) => assert!(
                msg.contains(KEY_MODEL),
                "the error must name the missing key: {msg}"
            ),
            Err(other) => panic!("expected GgufFormat, got: {other}"),
            Ok(_) => panic!("expected an error for a missing model key"),
        }
    }

    #[test]
    fn non_gpt2_model_errors_clearly() {
        fn str_bytes(s: &str) -> Vec<u8> {
            let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(s.as_bytes());
            bytes
        }
        let mut data = str_bytes(KEY_MODEL);
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        data.extend_from_slice(&str_bytes("llama"));
        let (store, _) = MetadataStore::parse(&data, 0, 1).expect("parses");
        match tokenizer_from_gguf_metadata(&store) {
            Err(TokenizerError::GgufFormat(msg)) => assert!(msg.contains("llama")),
            Err(other) => panic!("expected GgufFormat, got {other}"),
            Ok(_) => panic!("expected an error for a non-gpt2 model"),
        }
    }

    /// `TOK-17`: every GGUF-side failure — a missing key, an empty or
    /// inconsistent array, a malformed merge — is the dedicated
    /// `GgufFormat` variant, never the `tokenizer.json` one, and its
    /// message no longer carries the old `"GGUF: "` prefix (the variant's
    /// own `Display` names the container).
    #[test]
    fn every_gguf_metadata_failure_is_the_gguf_format_variant() {
        fn str_bytes(s: &str) -> Vec<u8> {
            let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(s.as_bytes());
            bytes
        }
        fn kv_string(key: &str, value: &str) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&str_bytes(value));
            bytes
        }
        fn kv_u32(key: &str, value: u32) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
            bytes
        }
        fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                bytes.extend_from_slice(&str_bytes(v));
            }
            bytes
        }
        fn kv_i32_array(key: &str, values: &[i32]) -> Vec<u8> {
            let mut bytes = str_bytes(key);
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        }
        fn store(entries: &[Vec<u8>]) -> MetadataStore {
            let data: Vec<u8> = entries.iter().flatten().copied().collect();
            let (store, _) = MetadataStore::parse(&data, 0, entries.len() as u64).expect("parses");
            store
        }

        let cases: Vec<(&str, MetadataStore, &str)> = vec![
            (
                "missing tokens",
                store(&[kv_string(KEY_MODEL, "gpt2")]),
                KEY_TOKENS,
            ),
            (
                "empty tokens",
                store(&[
                    kv_string(KEY_MODEL, "gpt2"),
                    kv_string_array(KEY_TOKENS, &[]),
                ]),
                "is empty",
            ),
            (
                "token_type length mismatch",
                store(&[
                    kv_string(KEY_MODEL, "gpt2"),
                    kv_string_array(KEY_TOKENS, &["a", "b"]),
                    kv_i32_array(KEY_TOKEN_TYPE, &[1]),
                ]),
                "token_type has 1 entries",
            ),
            (
                "missing eos",
                store(&[
                    kv_string(KEY_MODEL, "gpt2"),
                    kv_string_array(KEY_TOKENS, &["a", "b"]),
                ]),
                KEY_EOS_TOKEN_ID,
            ),
            (
                "malformed merge",
                store(&[
                    kv_string(KEY_MODEL, "gpt2"),
                    kv_string_array(KEY_TOKENS, &["a", "b"]),
                    kv_u32(KEY_EOS_TOKEN_ID, 0),
                    kv_string_array(KEY_MERGES, &["ab"]),
                ]),
                "malformed tokenizer.ggml.merges entry #0",
            ),
        ];
        for (label, md, needle) in cases {
            match tokenizer_from_gguf_metadata(&md) {
                Err(TokenizerError::GgufFormat(msg)) => {
                    assert!(
                        msg.contains(needle),
                        "{label}: {msg:?} must name {needle:?}"
                    );
                    assert!(
                        !msg.starts_with("GGUF: "),
                        "{label}: the variant already names the container: {msg:?}"
                    );
                }
                Err(other) => panic!("{label}: expected GgufFormat, got {other:?}"),
                Ok(_) => panic!("{label}: expected an error"),
            }
        }
    }

    #[test]
    fn chat_template_extraction() {
        let md = make_metadata(false);
        let tmpl = chat_template_from_gguf_metadata(&md);
        assert_eq!(tmpl.as_deref(), Some("{{ messages }}"));
    }

    #[test]
    fn chat_template_absent_is_none() {
        let store = MetadataStore::new();
        assert_eq!(chat_template_from_gguf_metadata(&store), None);
    }
}
