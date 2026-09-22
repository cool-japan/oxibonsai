//! Tokenizer bridge: Pure-Rust native backend, optional HuggingFace backend.
//!
//! [`TokenizerBridge`] is an enum over two interchangeable backends:
//!
//! * [`TokenizerBridge::Native`] — the workspace's own Pure-Rust BPE
//!   ([`oxibonsai_tokenizer::OxiTokenizer`]).  Always compiled in, including
//!   on `wasm32` targets and in a `--no-default-features` build.
//! * [`TokenizerBridge::Hf`] — HuggingFace `tokenizers`.  Compiled only when
//!   the crate feature `hf-tokenizer` is enabled **and** the target is not
//!   `wasm32` (the dependency itself is declared under
//!   `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]`, so the
//!   feature alone does not make the crate linkable — hence the
//!   `all(feature = "hf-tokenizer", not(target_arch = "wasm32"))` predicate
//!   repeated throughout this file).
//!
//! `hf-tokenizer` stays a *default* feature and [`TokenizerBridge::from_file`]
//! keeps preferring the HF backend whenever it is compiled in, so a default
//! build behaves exactly as it did before the backend split
//! (deps-07 / TOK-13).  Turning the feature off no longer removes the type:
//! the native backend covers every operation, so `--no-default-features`
//! compiles and serves real traffic instead of returning "unavailable"
//! errors (the pre-split `wasm32` stubs did exactly that).

use crate::error::{RuntimeError, RuntimeResult};
use oxibonsai_tokenizer::OxiTokenizer;
use std::collections::HashMap;

/// Which backend a [`TokenizerBridge`] is currently using.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenizerBackendKind {
    /// Pure-Rust [`oxibonsai_tokenizer::OxiTokenizer`].
    Native,
    /// HuggingFace `tokenizers`.
    Hf,
}

/// One entry of a tokenizer's *added vocabulary*, in a backend-neutral shape.
///
/// Mirrors the two fields of HuggingFace's `AddedToken` that this workspace
/// consumes (`content` and `special`); the native backend fills them from
/// [`oxibonsai_tokenizer::Vocabulary`]'s protected/special registries, which
/// carry the same `AddedVocabulary` semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedTokenInfo {
    /// The literal token text, e.g. `<|im_start|>`.
    pub content: String,
    /// Whether the token was flagged `special` by the vocabulary.
    pub special: bool,
}

/// Backend-neutral read-only view of a bridge's vocabulary.
///
/// Returned by [`TokenizerBridge::inner`].  It exposes the vocabulary queries
/// the server needs (notably `get_added_tokens_decoder`, which drives
/// `server::sanitize::SpecialTokenGuard`) with one signature that is valid in
/// every feature configuration.  Callers that need the *whole* HuggingFace
/// tokenizer can still reach it through [`TokenizerBridge::hf`].
pub struct TokenizerVocabView<'a> {
    bridge: &'a TokenizerBridge,
}

impl TokenizerVocabView<'_> {
    /// The tokenizer's added vocabulary, keyed by token id.
    ///
    /// HuggingFace semantics: *every* added token is returned, whether or not
    /// it is flagged `special` — the flag is carried in
    /// [`AddedTokenInfo::special`].
    pub fn get_added_tokens_decoder(&self) -> HashMap<u32, AddedTokenInfo> {
        match self.bridge {
            TokenizerBridge::Native(tok) => {
                let vocab = tok.vocab();
                vocab
                    .protected_tokens()
                    .map(|(content, id)| {
                        (
                            id,
                            AddedTokenInfo {
                                content: content.to_owned(),
                                special: vocab.is_special_token(content),
                            },
                        )
                    })
                    .collect()
            }
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBridge::Hf(tok) => tok
                .get_added_tokens_decoder()
                .into_iter()
                .map(|(id, added)| {
                    (
                        id,
                        AddedTokenInfo {
                            content: added.content,
                            special: added.special,
                        },
                    )
                })
                .collect(),
        }
    }

    /// Total vocabulary size, added tokens included.
    pub fn vocab_size(&self) -> usize {
        self.bridge.vocab_size()
    }

    /// Resolve a token string to its id, if the vocabulary contains it.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self.bridge {
            TokenizerBridge::Native(tok) => tok.vocab().get_id(token),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBridge::Hf(tok) => tok.token_to_id(token),
        }
    }

    /// Resolve a token id to its literal token string, if it exists.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        match self.bridge {
            TokenizerBridge::Native(tok) => tok.vocab().get_token(id).map(ToOwned::to_owned),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBridge::Hf(tok) => tok.id_to_token(id),
        }
    }
}

/// Tokenizer used by the inference engine, server and CLI.
///
/// See the [module docs](self) for the backend split.
pub enum TokenizerBridge {
    /// Pure-Rust BPE backend — always available.
    Native(Box<OxiTokenizer>),
    /// HuggingFace `tokenizers` backend (feature `hf-tokenizer`, non-wasm).
    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    Hf(Box<tokenizers::Tokenizer>),
}

/// Per-stream UTF-8-safe decode state. Owned by the caller.
///
/// BPE / byte-level tokenizers (Qwen3, GPT-2, etc.) sometimes emit a single
/// token that carries only **part** of a multi-byte UTF-8 character (e.g. one
/// byte of a CJK ideograph or emoji).  Decoding tokens one-at-a-time without
/// buffering breaks those multi-byte sequences and produces `U+FFFD`
/// replacement characters in the output stream.  This state mirrors what the
/// HuggingFace `tokenizers::DecodeStream` keeps internally so that we can own
/// it externally and feed tokens through [`TokenizerBridge::step_decode`] —
/// the native backend runs the *same* windowing algorithm over the same four
/// fields, so both backends stream identically.
///
/// Use one `DecodeStreamState` per generation request; reset (or drop &
/// re-create) it between independent requests.
#[derive(Default)]
pub struct DecodeStreamState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    skip_special_tokens: bool,
}

impl DecodeStreamState {
    /// Construct a fresh decode-stream state.
    ///
    /// `skip_special_tokens` matches the existing `decode()` behavior — pass
    /// `true` to drop sentinel tokens (e.g. `<|im_end|>`) from the output.
    pub fn new(skip_special_tokens: bool) -> Self {
        Self {
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            skip_special_tokens,
        }
    }

    /// Reset the state, preserving the original `skip_special_tokens` flag.
    pub fn reset(&mut self) {
        *self = Self::new(self.skip_special_tokens);
    }
}

impl TokenizerBridge {
    /// Load a tokenizer from a HuggingFace-format `tokenizer.json` file.
    ///
    /// Uses the HuggingFace backend when it is compiled in (the default
    /// feature set), otherwise the Pure-Rust native backend.  Use
    /// [`Self::native_from_file`] to pin the native backend explicitly.
    pub fn from_file(path: &str) -> RuntimeResult<Self> {
        #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
        {
            let inner = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
            Ok(Self::Hf(Box::new(inner)))
        }
        #[cfg(not(all(feature = "hf-tokenizer", not(target_arch = "wasm32"))))]
        {
            Self::native_from_file(path)
        }
    }

    /// Load a tokenizer from a HuggingFace-format `tokenizer.json` file using
    /// the Pure-Rust native backend, regardless of which features are on.
    pub fn native_from_file(path: &str) -> RuntimeResult<Self> {
        let inner = OxiTokenizer::from_json_file(path)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        Ok(Self::Native(Box::new(inner)))
    }

    /// Load a tokenizer from the *contents* of a HuggingFace-format
    /// `tokenizer.json`, using the Pure-Rust native backend.
    ///
    /// The filesystem-free variant of [`Self::native_from_file`] — the form
    /// `wasm32` builds and embedded fixtures need.
    pub fn native_from_json_str(json: &str) -> RuntimeResult<Self> {
        let inner = OxiTokenizer::from_hf_tokenizer_json(json)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        Ok(Self::Native(Box::new(inner)))
    }

    /// Wrap an already-constructed native tokenizer.
    pub fn from_native_tokenizer(tokenizer: OxiTokenizer) -> Self {
        Self::Native(Box::new(tokenizer))
    }

    /// Wrap an already-constructed HuggingFace tokenizer.
    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    pub fn from_hf_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self::Hf(Box::new(tokenizer))
    }

    /// Which backend this bridge is using.
    pub fn backend(&self) -> TokenizerBackendKind {
        match self {
            Self::Native(_) => TokenizerBackendKind::Native,
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(_) => TokenizerBackendKind::Hf,
        }
    }

    /// Encode text to token IDs (no special tokens added).
    pub fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        match self {
            Self::Native(tok) => tok
                .encode(text)
                .map_err(|e| RuntimeError::Tokenizer(e.to_string())),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(tok) => {
                let encoding = tok
                    .encode(text, false)
                    .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
                Ok(encoding.get_ids().to_vec())
            }
        }
    }

    /// Decode token IDs to text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> RuntimeResult<String> {
        match self {
            Self::Native(tok) => Ok(native_decode(tok, ids, true)),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(tok) => tok
                .decode(ids, true)
                .map_err(|e| RuntimeError::Tokenizer(e.to_string())),
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Native(tok) => tok.vocab_size(),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(tok) => tok.get_vocab_size(true),
        }
    }

    /// A backend-neutral read-only view of the loaded vocabulary.
    ///
    /// This is what `server::sanitize::SpecialTokenGuard` builds its control
    /// token set from, so it must stay valid in every feature configuration —
    /// see [`TokenizerVocabView`].
    pub fn inner(&self) -> TokenizerVocabView<'_> {
        TokenizerVocabView { bridge: self }
    }

    /// The underlying HuggingFace tokenizer, when this bridge is HF-backed.
    ///
    /// Returns `None` for a native-backed bridge.  Only compiled when the
    /// `hf-tokenizer` feature is on and the target can link `tokenizers`.
    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    pub fn hf(&self) -> Option<&tokenizers::Tokenizer> {
        match self {
            Self::Hf(tok) => Some(tok),
            Self::Native(_) => None,
        }
    }

    /// The underlying native tokenizer, when this bridge is native-backed.
    pub fn native(&self) -> Option<&OxiTokenizer> {
        match self {
            Self::Native(tok) => Some(tok),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(_) => None,
        }
    }

    /// Whether `id` is a special/control token in the loaded vocabulary
    /// (TOK-15) — e.g. `<|im_start|>`, `<|endoftext|>`. Driven from the
    /// vocabulary actually loaded, not a hardcoded id range: Qwen3-family
    /// specials sit in one block and Bonsai 2's sit in another
    /// (`248044..248076`), and this works for either because it asks the
    /// backend's own added-vocabulary registry rather than assuming a range.
    pub fn is_special(&self, id: u32) -> bool {
        match self {
            Self::Native(tok) => tok.vocab().is_special_id(id),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(tok) => match tok.id_to_token(id) {
                Some(token) => tok.get_added_vocabulary().is_special_token(&token),
                None => false,
            },
        }
    }

    /// The raw output bytes a single token id decodes to (TOK-15) — what
    /// TOK-M1's logprobs fix needs to emit OpenAI-shaped `bytes` fields
    /// without round-tripping a single id through the streaming decoder
    /// (which corrupts multi-byte characters split across ids, e.g.
    /// `"日本語処理"` decoded id-by-id comes back as
    /// `["日本","語","<?>","<?>","理"]`).
    ///
    /// Returns an **owned** `Vec<u8>` rather than a borrowed `&[u8]`: a
    /// byte-level vocabulary entry (e.g. GPT-2's `"Ġ"` standing for a literal
    /// space) is not itself the output byte sequence — it must be unmapped
    /// through the byte-level alphabet first — so there is nothing to borrow
    /// from without adding a cache. An out-of-vocabulary id decodes to the
    /// UTF-8 bytes of `U+FFFD`, matching [`Self::decode`]'s behaviour on
    /// invalid ids.
    ///
    /// Unlike [`Self::decode`]'s multi-token concatenation (which only
    /// resolves a leading byte-level space marker to a literal space when it
    /// is *not* the very first thing decoded, to avoid emitting a spurious
    /// leading space), a standalone piece has no such context to consult, so
    /// a leading space marker always resolves to a literal space here.
    pub fn piece(&self, id: u32) -> Vec<u8> {
        match self {
            Self::Native(tok) => native_token_piece_bytes(tok, id),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(tok) => match tok.id_to_token(id) {
                Some(token) => hf_token_piece_bytes(&token, hf_decoder_is_byte_level(tok)),
                None => "\u{FFFD}".as_bytes().to_vec(),
            },
        }
    }

    /// The set of token ids that end a generation (TOK-15 / RT-18's
    /// tokenizer-level building block).
    ///
    /// Driven from the loaded vocabulary rather than a hardcoded
    /// Qwen3-specific constant: the native backend's own configured
    /// `eos_token_id` (when the loaded `tokenizer.json` declared one *and*
    /// the vocabulary flags that id special per [`Self::is_special`]) is
    /// always first, followed by any additional well-known
    /// end-of-text/end-of-turn marker spellings
    /// ([`END_OF_TEXT_MARKER_SPELLINGS`]) that this vocabulary also defines
    /// as a *special* added token, in the order listed, deduplicated. Chat
    /// models routinely stop on more than one id (e.g. Qwen-family
    /// `<|im_end|>` **and** `<|endoftext|>`) — a caller with more authoritative
    /// information (a GGUF's `tokenizer.ggml.eos_token_id` plus any sibling
    /// stop-token metadata) should still prefer that source and merge it
    /// with this one; this method only reports what the tokenizer layer
    /// itself can determine.
    ///
    /// Returns an **owned** `Vec<u32>` — computed fresh from vocabulary
    /// lookups rather than cached, since it is expected to be called once
    /// per engine/session setup rather than per token.
    ///
    /// **Known limitation, closed in practice by the specialness check**:
    /// `oxibonsai_tokenizer::TokenizerConfig::eos_token_id` is a plain `u32`,
    /// not an `Option<u32>`, so once a `tokenizer.json` is loaded there is no
    /// direct way to tell "the file explicitly declared this eos id" apart
    /// from "nothing declared one and this is just
    /// `TokenizerConfig::default()`'s built-in `2`". Requiring the id to
    /// both resolve to a real vocabulary entry *and* be flagged special
    /// closes this for every real vocabulary this bridge loads: an unset
    /// default that happens to alias a real, unrelated token — as
    /// `TokenizerConfig::default()`'s `2` does on the repo's own
    /// `models/tokenizer.json` (real Qwen3), where id 2 is the ordinary
    /// token `"#"` — is now filtered out rather than reported, because no
    /// real `tokenizer.json` flags an ordinary word special. Only a
    /// vocabulary that deliberately marks *its own* default-aliasing id
    /// special would still be misread as "declared"; closing that residual
    /// case fully requires `oxibonsai_tokenizer` itself to carry an explicit
    /// "was this set" flag, which is outside this package's owned files.
    pub fn eos_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        // `match`, not `if let`: with the `hf-tokenizer` feature off,
        // `Native` is this enum's only variant, and an `if let` against it
        // would be flagged `irrefutable_let_patterns` under `-D warnings`.
        match self {
            Self::Native(tok) => {
                let configured = tok.config().eos_token_id;
                // The native backend's `TokenizerConfig::eos_token_id`
                // defaults to `2` (see
                // `oxibonsai_tokenizer::TokenizerConfig::default`) even when
                // nothing in the loaded `tokenizer.json` set it, so mere
                // vocabulary resolvability is not enough to trust it: on the
                // repo's own `models/tokenizer.json` (real Qwen3, which
                // declares no top-level `eos_token`), id 2 resolves to the
                // ordinary token `"#"` — exactly the "wrong non-Qwen
                // fallback" RT-18 warns about. Requiring `is_special` too
                // (the same predicate the marker-spelling loop below already
                // applies) closes it: a real `tokenizer.json` never flags an
                // ordinary word special, so only a genuinely-declared eos id
                // survives.
                if tok.vocab().get_token(configured).is_some() && self.is_special(configured) {
                    ids.push(configured);
                }
            }
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(_) => {
                // `tokenizers::Tokenizer` carries no `eos_token_id` config of
                // its own (that lives in a sibling `tokenizer_config.json`
                // this bridge does not parse) — only the marker-spelling
                // scan below applies to this backend.
            }
        }
        for marker in END_OF_TEXT_MARKER_SPELLINGS {
            if let Some(id) = self.inner().token_to_id(marker) {
                if self.is_special(id) && !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Create a fresh decode-stream state for one generation request.
    ///
    /// See [`DecodeStreamState`] and [`Self::step_decode`] for the streaming
    /// decode protocol.  Use this instead of repeatedly calling
    /// [`Self::decode`] with single-token slices, which mishandles tokens that
    /// straddle UTF-8 codepoint boundaries.
    pub fn new_decode_stream(&self, skip_special_tokens: bool) -> DecodeStreamState {
        DecodeStreamState::new(skip_special_tokens)
    }

    /// Advance the decode stream by one token.
    ///
    /// Returns `Ok(Some(text))` only when the buffered bytes form a complete
    /// UTF-8 chunk (which may span several previous tokens for CJK / emoji);
    /// returns `Ok(None)` when more tokens are needed before any well-formed
    /// text can be emitted.  Callers must **not** print the empty string when
    /// `Ok(None)` is returned — wait for the next token.
    pub fn step_decode(
        &self,
        state: &mut DecodeStreamState,
        id: u32,
    ) -> RuntimeResult<Option<String>> {
        match self {
            Self::Native(tok) => native_step_decode(tok, state, id),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            Self::Hf(tok) => tokenizers::step_decode_stream(
                tok.as_ref(),
                vec![id],
                state.skip_special_tokens,
                &mut state.ids,
                &mut state.prefix,
                &mut state.prefix_index,
            )
            .map_err(|e| RuntimeError::Tokenizer(e.to_string())),
        }
    }
}

impl std::fmt::Debug for TokenizerBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerBridge")
            .field("backend", &self.backend())
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

// ── Native decode ────────────────────────────────────────────────────────────
//
// `oxibonsai_tokenizer::OxiTokenizer::decode` cannot be used directly here for
// two reasons, both of which matter for server output correctness:
//
// 1. It has no `skip_special_tokens` switch — it always drops the four ids in
//    `TokenizerConfig` (`bos/eos/unk/pad`).
// 2. Those four ids default to `1/2/0/3` and a HuggingFace `tokenizer.json`
//    that declares no `unk_token`/`pad_token` (Qwen3's does not) leaves the
//    defaults in place.  In a GPT-2-style byte-level vocabulary ids 0..=3 are
//    the ordinary tokens `!`, `"`, `#`, `$`, so `decode` silently deletes
//    those characters from generated text.
//
// The loop below therefore reproduces `OxiTokenizer::decode_id_into`'s
// byte-level / byte-fallback / legacy-`Ġ` behaviour on top of the crate's
// public API (`vocab()`, `config()`, `unicode_to_byte`) while deciding what
// counts as "special" from the *vocabulary* (`special == true` in
// `added_tokens`), which is exactly HuggingFace's `skip_special_tokens` rule.
// The upstream fix is recorded in this package's deviations.

/// GPT-2 byte-level marker for an encoded space (`U+0120`, "Ġ").
const BYTE_LEVEL_SPACE_MARKER: char = '\u{0120}';

/// Well-known end-of-text/end-of-turn marker spellings [`TokenizerBridge::eos_ids`]
/// checks for beyond the native backend's own configured `eos_token_id`.
/// These are the literal token strings real chat-model vocabularies use
/// across the GPT-2/ChatML/Llama family (Qwen3's `tokenizer.json` declares
/// `<|endoftext|>` *and* `<|im_end|>` as separate specials; Bonsai 2's does
/// the same). A marker only counts if the loaded vocabulary both contains it
/// *and* flags it `special` — an ordinary model that happens to have, say,
/// `</s>` as a normal in-vocabulary word would not match, since normal words
/// are never flagged special.
const END_OF_TEXT_MARKER_SPELLINGS: &[&str] =
    &["<|im_end|>", "<|endoftext|>", "<|end|>", "</s>", "<eos>"];

/// Parse a `<0xHH>` byte-fallback token into the byte it stands for.
fn parse_byte_fallback(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Append the bytes of one token id to `out`.
fn native_decode_id_into(
    tok: &OxiTokenizer,
    id: u32,
    skip_special_tokens: bool,
    out: &mut Vec<u8>,
) {
    let vocab = tok.vocab();
    if skip_special_tokens && vocab.is_special_id(id) {
        return;
    }
    let token = match vocab.get_token(id) {
        Some(t) => t,
        None => {
            out.extend_from_slice("\u{FFFD}".as_bytes());
            return;
        }
    };

    if let Some(byte) = parse_byte_fallback(token) {
        out.push(byte);
        return;
    }

    if tok.config().byte_level_decode {
        for ch in token.chars() {
            match oxibonsai_tokenizer::unicode_to_byte(ch) {
                Some(b) => out.push(b),
                None => {
                    // Not part of the GPT-2 byte-level alphabet — emit the
                    // character's own UTF-8 bytes verbatim.
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    } else {
        let stripped = token.trim_start_matches(BYTE_LEVEL_SPACE_MARKER);
        if token.starts_with(BYTE_LEVEL_SPACE_MARKER) && !out.is_empty() {
            out.push(b' ');
        }
        out.extend_from_slice(stripped.as_bytes());
    }
}

/// Decode one token id to its own, context-free raw output bytes — the
/// native-backend half of [`TokenizerBridge::piece`].
///
/// Deliberately a near-duplicate of [`native_decode_id_into`]'s body rather
/// than a shared helper with a flag: the two have genuinely different
/// concatenation semantics (this one has no "is this the very start of a
/// longer decoded string" context to consult, so a leading byte-level space
/// marker always resolves to a literal space — see [`TokenizerBridge::piece`]'s
/// doc), and folding both into one function with a boolean parameter would
/// obscure that difference rather than clarify it. Special-token skipping is
/// intentionally **not** applied here (unlike `native_decode_id_into`):
/// `piece()` reports the raw bytes for *any* id the caller asks about (e.g.
/// TOK-M1's logprobs use includes ids skipped by `decode`).
fn native_token_piece_bytes(tok: &OxiTokenizer, id: u32) -> Vec<u8> {
    let vocab = tok.vocab();
    let token = match vocab.get_token(id) {
        Some(t) => t,
        None => return "\u{FFFD}".as_bytes().to_vec(),
    };

    if let Some(byte) = parse_byte_fallback(token) {
        return vec![byte];
    }

    let mut out = Vec::with_capacity(token.len());
    if tok.config().byte_level_decode {
        for ch in token.chars() {
            match oxibonsai_tokenizer::unicode_to_byte(ch) {
                Some(b) => out.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    } else if let Some(rest) = token.strip_prefix(BYTE_LEVEL_SPACE_MARKER) {
        out.push(b' ');
        out.extend_from_slice(rest.as_bytes());
    } else {
        out.extend_from_slice(token.as_bytes());
    }
    out
}

/// Whether an HF `tokenizers::Tokenizer`'s decoder is (or contains) a
/// byte-level decoder — the gate [`hf_token_piece_bytes`] needs to decide
/// whether unmapping through the GPT-2 byte-level alphabet is even correct
/// for this vocabulary. A `Sequence` decoder (some `tokenizer.json` files
/// wrap `ByteLevel` alongside e.g. a `Fuse`/`Strip` stage) counts if any of
/// its stages is `ByteLevel`, checked recursively in case a `Sequence`
/// itself nests another `Sequence`.
#[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
fn hf_decoder_is_byte_level(tok: &tokenizers::Tokenizer) -> bool {
    fn is_byte_level(decoder: &tokenizers::DecoderWrapper) -> bool {
        match decoder {
            tokenizers::DecoderWrapper::ByteLevel(_) => true,
            tokenizers::DecoderWrapper::Sequence(seq) => {
                seq.get_decoders().iter().any(is_byte_level)
            }
            _ => false,
        }
    }
    tok.get_decoder().is_some_and(is_byte_level)
}

/// HF-backend half of [`TokenizerBridge::piece`]: `tokenizers::Tokenizer`
/// already hands back the token in the same GPT-2 byte-level alphabet when
/// its decoder is byte-level ([`hf_decoder_is_byte_level`]), so the
/// unmapping step is identical to the native backend's — reuse the
/// `unicode_to_byte` table from `oxibonsai_tokenizer` rather than
/// reimplementing the 256-entry alphabet a second time. Returns the token's
/// own UTF-8 bytes as-is for a non-byte-level vocabulary (e.g. a
/// WordPiece/Unigram `tokenizer.json`), matching how `tokenizers::Tokenizer`
/// itself would emit it. `byte_level` must come from
/// [`hf_decoder_is_byte_level`] rather than being assumed true: mapping
/// every char through `unicode_to_byte` unconditionally mis-decodes a
/// non-byte-level vocabulary's non-ASCII tokens (`"café"` would come back as
/// `[c, a, f, 0xE9]` — a lone UTF-8 continuation byte — instead of
/// `[c, a, f, 0xC3, 0xA9]`).
#[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
fn hf_token_piece_bytes(token: &str, byte_level: bool) -> Vec<u8> {
    if let Some(byte) = parse_byte_fallback(token) {
        return vec![byte];
    }
    if !byte_level {
        return token.as_bytes().to_vec();
    }
    let mut out = Vec::with_capacity(token.len());
    for ch in token.chars() {
        match oxibonsai_tokenizer::unicode_to_byte(ch) {
            Some(b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

/// Decode a token-id slice with the native backend.
///
/// Byte sequences that are not (yet) valid UTF-8 — a token carrying only part
/// of a multi-byte character — are rendered with `U+FFFD`, matching the
/// HuggingFace backend, so [`native_step_decode`]'s "is this chunk complete?"
/// test can be the same one HuggingFace uses.
fn native_decode(tok: &OxiTokenizer, ids: &[u32], skip_special_tokens: bool) -> String {
    let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
    for &id in ids {
        native_decode_id_into(tok, id, skip_special_tokens, &mut bytes);
    }
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

/// Native-backend counterpart of `tokenizers::step_decode_stream`.
///
/// Same algorithm, same state fields: keep a sliding window of ids, decode the
/// whole window each step, and emit only the part that extends the previously
/// emitted prefix — and only once it no longer ends in a replacement
/// character, i.e. once the trailing multi-byte sequence is complete.
///
/// Caveat, unreachable through [`TokenizerBridge::native_from_file`] /
/// [`TokenizerBridge::native_from_json_str`] (both set
/// `byte_level_decode = true`): a tokenizer built by
/// [`TokenizerBridge::from_native_tokenizer`] from `OxiTokenizer::from_json`
/// decodes with the legacy `Ġ` rule, whose leading-space emission depends on
/// whether the output buffer is already non-empty. For such a tokenizer the
/// window reset below can drop one space at a window boundary. Byte-level
/// decoding — every HuggingFace `tokenizer.json` — is context-free and
/// unaffected.
fn native_step_decode(
    tok: &OxiTokenizer,
    state: &mut DecodeStreamState,
    id: u32,
) -> RuntimeResult<Option<String>> {
    let skip = state.skip_special_tokens;

    if state.prefix.is_empty() && !state.ids.is_empty() {
        let new_prefix = native_decode(tok, &state.ids, skip);
        if !new_prefix.ends_with('\u{FFFD}') {
            state.prefix = new_prefix;
            state.prefix_index = state.ids.len();
        }
    }

    state.ids.push(id);
    let string = native_decode(tok, &state.ids, skip);
    if string.len() > state.prefix.len() && !string.ends_with('\u{FFFD}') {
        if !string.starts_with(&state.prefix) {
            return Err(RuntimeError::Tokenizer(format!(
                "decode stream desynchronized on token {id}: expected prefix {:?}, decoded {:?}",
                state.prefix, string
            )));
        }
        let new_text = string[state.prefix.len()..].to_owned();
        let new_prefix_index = state.ids.len() - state.prefix_index;
        state.ids = state.ids.split_off(state.prefix_index);
        state.prefix = native_decode(tok, &state.ids, skip);
        state.prefix_index = new_prefix_index;
        Ok(Some(new_text))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Path to the project's bundled Qwen3 tokenizer.  Tests that need a real
    /// BPE tokenizer skip themselves when this fixture is missing so that
    /// freshly-cloned working trees still pass `cargo test`.
    const FIXTURE_TOKENIZER: &str = "../../models/tokenizer.json";

    /// A minimal byte-level HuggingFace `tokenizer.json`.
    ///
    /// Deliberately assigns ids 0..=3 to the ordinary characters `!`, `"`,
    /// `#`, `$` — the ids `TokenizerConfig` defaults to for `unk`, `bos`,
    /// `eos` and `pad` — which is what makes it a regression fixture for the
    /// "native decode eats punctuation" defect.  `Ã`/`©` are the GPT-2
    /// byte-level spellings of `0xC3`/`0xA9`, i.e. the two halves of `é`, so
    /// streaming across them exercises the incomplete-UTF-8 path.  `Ġ` is the
    /// byte-level spelling of a space.
    const TINY_TOKENIZER_JSON: &str = r##"{
        "model": {
            "type": "BPE",
            "vocab": {
                "!": 0, "\"": 1, "#": 2, "$": 3,
                "H": 4, "i": 5, "Ġ": 6, "a": 7, "b": 8,
                "Ã": 9, "©": 10, "Hi": 11
            },
            "merges": ["H i"]
        },
        "added_tokens": [
            { "id": 12, "content": "<|im_start|>", "special": true },
            { "id": 13, "content": "<tool_call>", "special": false }
        ],
        "pre_tokenizer": { "type": "ByteLevel" },
        "decoder": { "type": "ByteLevel" }
    }"##;

    fn tiny_native_bridge() -> TokenizerBridge {
        TokenizerBridge::native_from_json_str(TINY_TOKENIZER_JSON)
            .expect("tiny tokenizer fixture should load")
    }

    fn maybe_load_fixture() -> Option<TokenizerBridge> {
        if !Path::new(FIXTURE_TOKENIZER).exists() {
            eprintln!(
                "skipped: tokenizer fixture not found at {FIXTURE_TOKENIZER} \
                 (run scripts/download_tokenizer.sh to enable)",
            );
            return None;
        }
        match TokenizerBridge::from_file(FIXTURE_TOKENIZER) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("skipped: failed to load tokenizer fixture: {e}");
                None
            }
        }
    }

    fn maybe_load_native_fixture() -> Option<TokenizerBridge> {
        if !Path::new(FIXTURE_TOKENIZER).exists() {
            eprintln!("skipped: tokenizer fixture not found at {FIXTURE_TOKENIZER}");
            return None;
        }
        match TokenizerBridge::native_from_file(FIXTURE_TOKENIZER) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("skipped: failed to load native tokenizer fixture: {e}");
                None
            }
        }
    }

    /// Drive every id through `step_decode` and concatenate the well-formed
    /// chunks.  Mirrors what the CLI / SSE code paths do.
    fn stream_through(tok: &TokenizerBridge, ids: &[u32]) -> RuntimeResult<String> {
        let mut state = tok.new_decode_stream(true);
        let mut out = String::new();
        for &id in ids {
            if let Some(chunk) = tok.step_decode(&mut state, id)? {
                out.push_str(&chunk);
            }
        }
        Ok(out)
    }

    // ── Native backend (never gated on `hf-tokenizer`) ────────────────────

    #[test]
    fn native_backend_is_always_available() {
        let tok = tiny_native_bridge();
        assert_eq!(tok.backend(), TokenizerBackendKind::Native);
        assert!(tok.native().is_some());
        assert!(tok.vocab_size() >= 12);
    }

    #[test]
    fn native_decode_keeps_tokens_that_alias_default_special_ids() -> RuntimeResult<()> {
        let tok = tiny_native_bridge();

        // Ids 0..=3 are `!`, `"`, `#`, `$` here and collide with
        // TokenizerConfig's default unk/bos/eos/pad ids.  A decoder that
        // trusts those defaults silently deletes the characters.
        let text = "!\"#$";
        let ids = tok.encode(text)?;
        assert_eq!(ids, vec![0, 1, 2, 3], "byte-level encode of {text:?}");
        assert_eq!(tok.decode(&ids)?, text, "punctuation must survive decode");
        assert_eq!(stream_through(&tok, &ids)?, text, "…and streaming decode");
        Ok(())
    }

    #[test]
    fn native_round_trips_bpe_merges_and_spaces() -> RuntimeResult<()> {
        let tok = tiny_native_bridge();
        let text = "Hi ab!";
        let ids = tok.encode(text)?;
        assert_eq!(tok.decode(&ids)?, text);
        assert_eq!(stream_through(&tok, &ids)?, text);
        Ok(())
    }

    #[test]
    fn native_streaming_waits_for_complete_utf8() -> RuntimeResult<()> {
        let tok = tiny_native_bridge();
        let mut state = tok.new_decode_stream(true);

        // `é` is 0xC3 0xA9: the first token alone is not a valid character.
        assert_eq!(
            tok.step_decode(&mut state, 9)?,
            None,
            "half a codepoint must not be emitted"
        );
        assert_eq!(
            tok.step_decode(&mut state, 10)?,
            Some("é".to_string()),
            "the character is emitted once it is complete"
        );
        assert_eq!(tok.step_decode(&mut state, 4)?, Some("H".to_string()));
        Ok(())
    }

    #[test]
    fn native_skips_special_tokens_only_when_asked() -> RuntimeResult<()> {
        let tok = tiny_native_bridge();
        let ids = vec![4, 5, 12];

        // `<|im_start|>` is flagged `special`, so the default decode drops it.
        assert_eq!(tok.decode(&ids)?, "Hi");

        // …and is preserved when the stream is told to keep special tokens.
        let mut state = tok.new_decode_stream(false);
        let mut kept = String::new();
        for id in ids {
            if let Some(chunk) = tok.step_decode(&mut state, id)? {
                kept.push_str(&chunk);
            }
        }
        assert_eq!(kept, "Hi<|im_start|>");
        Ok(())
    }

    #[test]
    fn native_added_tokens_view_carries_the_special_flag() {
        let tok = tiny_native_bridge();
        let added = tok.inner().get_added_tokens_decoder();

        let start = added.get(&12).expect("<|im_start|> must be an added token");
        assert_eq!(start.content, "<|im_start|>");
        assert!(start.special, "declared special == true in the fixture");

        let tool = added.get(&13).expect("<tool_call> must be an added token");
        assert_eq!(tool.content, "<tool_call>");
        assert!(
            !tool.special,
            "declared special == false: protected from pre-tokenization, not a control token"
        );

        assert_eq!(tok.inner().token_to_id("Hi"), Some(11));
        assert_eq!(tok.inner().id_to_token(11).as_deref(), Some("Hi"));
        assert_eq!(tok.inner().vocab_size(), tok.vocab_size());
    }

    #[test]
    fn native_encode_is_unaffected_by_added_token_text() -> RuntimeResult<()> {
        let tok = tiny_native_bridge();
        // An added token embedded in text keeps its atomic id (HF
        // `AddedVocabulary` semantics), which is what the server's prompt
        // sanitizer relies on.
        let ids = tok.encode("Hi<|im_start|>")?;
        assert!(ids.contains(&12), "added token should encode atomically");
        Ok(())
    }

    #[test]
    fn native_fixture_round_trips_ascii_and_cjk() -> RuntimeResult<()> {
        let Some(tok) = maybe_load_native_fixture() else {
            return Ok(());
        };
        assert_eq!(tok.backend(), TokenizerBackendKind::Native);

        for input in [
            "Hello, world! Streaming ASCII works fine.",
            "日本語処理を専門",
            "The quick brown fox — jumps over 🦊 the lazy dog.",
        ] {
            let ids = tok.encode(input)?;
            assert!(!ids.is_empty(), "encoding {input:?} yielded no ids");
            assert_eq!(tok.decode(&ids)?, input, "native decode of {input:?}");
            let streamed = stream_through(&tok, &ids)?;
            assert!(!streamed.contains('\u{FFFD}'), "U+FFFD in {streamed:?}");
            assert_eq!(streamed, input, "native streaming decode of {input:?}");
        }
        Ok(())
    }

    // ── Backend selection ─────────────────────────────────────────────────

    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    #[test]
    fn from_file_prefers_the_hf_backend_when_compiled_in() {
        let Some(tok) = maybe_load_fixture() else {
            return;
        };
        assert_eq!(
            tok.backend(),
            TokenizerBackendKind::Hf,
            "default builds must keep using the HuggingFace backend"
        );
        assert!(tok.hf().is_some());
        assert!(tok.native().is_none());
    }

    #[cfg(not(all(feature = "hf-tokenizer", not(target_arch = "wasm32"))))]
    #[test]
    fn from_file_falls_back_to_the_native_backend() {
        let Some(tok) = maybe_load_fixture() else {
            return;
        };
        assert_eq!(tok.backend(), TokenizerBackendKind::Native);
    }

    // ── Streaming decode against the real Qwen3 vocabulary ────────────────

    #[test]
    fn streaming_decode_cjk_no_replacement_chars() -> RuntimeResult<()> {
        let Some(tok) = maybe_load_fixture() else {
            return Ok(());
        };

        // Mix of Japanese ideographs and hiragana exercising multi-byte UTF-8
        // (3 bytes per char) that BPE byte-level tokenization typically splits
        // across two or three tokens.
        let input = "日本語処理を専門";
        let ids = tok.encode(input)?;
        assert!(!ids.is_empty(), "encoding yielded no token ids");

        let streamed = stream_through(&tok, &ids)?;

        assert!(
            !streamed.contains('\u{FFFD}'),
            "streaming decode produced U+FFFD replacement char(s); output: {streamed:?}",
        );
        assert_eq!(
            streamed, input,
            "streaming decode did not reconstruct the original CJK input",
        );
        Ok(())
    }

    #[test]
    fn streaming_decode_ascii_passes_through() -> RuntimeResult<()> {
        let Some(tok) = maybe_load_fixture() else {
            return Ok(());
        };

        let input = "Hello, world! Streaming ASCII works fine.";
        let ids = tok.encode(input)?;
        let streamed = stream_through(&tok, &ids)?;
        assert!(!streamed.contains('\u{FFFD}'));
        assert_eq!(streamed, input);
        Ok(())
    }

    #[test]
    fn streaming_decode_handles_empty_input() -> RuntimeResult<()> {
        let Some(tok) = maybe_load_fixture() else {
            return Ok(());
        };

        // Driving zero ids must yield no output and must not panic.
        let streamed = stream_through(&tok, &[])?;
        assert!(
            streamed.is_empty(),
            "empty token stream should yield empty output, got {streamed:?}",
        );

        // Resetting a fresh state is a no-op; the state is still usable
        // afterwards (verified by re-running the empty-input drive).
        let mut state = tok.new_decode_stream(true);
        state.reset();
        let still_empty = stream_through(&tok, &[])?;
        assert!(still_empty.is_empty());
        Ok(())
    }

    #[test]
    fn byte_fallback_tokens_parse() {
        assert_eq!(parse_byte_fallback("<0x41>"), Some(b'A'));
        assert_eq!(parse_byte_fallback("<0x0a>"), Some(b'\n'));
        assert_eq!(parse_byte_fallback("<0xZZ>"), None);
        assert_eq!(parse_byte_fallback("<0x412>"), None);
        assert_eq!(parse_byte_fallback("hello"), None);
    }

    // ── TOK-15: is_special / piece / eos_ids ────────────────────────────────

    #[test]
    fn is_special_reflects_the_loaded_vocabulary_not_a_hardcoded_range() {
        let tok = tiny_native_bridge();
        // id 12 = "<|im_start|>", declared `special: true` in the fixture.
        assert!(tok.is_special(12));
        // id 13 = "<tool_call>", declared `special: false`.
        assert!(!tok.is_special(13));
        // Ordinary ids (including ones that alias other tokenizers' default
        // special-id ranges, e.g. 0..=3 here) are not special.
        assert!(!tok.is_special(0));
        assert!(!tok.is_special(4));
        // An out-of-vocabulary id is not special either.
        assert!(!tok.is_special(9_999));
    }

    #[test]
    fn piece_returns_raw_output_bytes_not_the_vocabulary_label() {
        let tok = tiny_native_bridge();
        // id 6 = "Ġ" (byte-level space marker) — the raw output byte is a
        // literal space, not the two-byte UTF-8 encoding of 'Ġ' itself.
        assert_eq!(tok.piece(6), b" ".to_vec());
        // id 4 = "H", an ordinary printable-ASCII byte-level token.
        assert_eq!(tok.piece(4), b"H".to_vec());
        // Out-of-vocabulary id decodes to U+FFFD, matching `decode`.
        assert_eq!(tok.piece(9_999), "\u{FFFD}".as_bytes().to_vec());
    }

    #[test]
    fn piece_reports_special_tokens_too_unlike_decode() {
        // decode() skips special tokens by default; piece() must still
        // report their real bytes (TOK-M1 needs this for ids that logprobs
        // reports even when they are control tokens).
        let tok = tiny_native_bridge();
        assert_eq!(tok.piece(12), b"<|im_start|>".to_vec());
    }

    #[test]
    fn piece_concatenation_reconstructs_multi_token_text() {
        // "é" = 0xC3 0xA9 split across ids 9 and 10 in the fixture — proves
        // `piece()` is genuinely returning raw output bytes usable for
        // OpenAI-style base64 `bytes` reporting, not just single ASCII
        // chars.
        let tok = tiny_native_bridge();
        let mut bytes = tok.piece(9);
        bytes.extend(tok.piece(10));
        assert_eq!(String::from_utf8(bytes).expect("valid utf8"), "é");
    }

    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    #[test]
    fn hf_token_piece_bytes_honors_the_byte_level_flag() {
        // Pins the exact scenario the rustdoc describes and the finding
        // caught it *not* actually implementing: for a non-byte-level
        // vocabulary, `unicode_to_byte('é')` resolves to `Some(0xE9)` (the
        // GPT-2 byte-level alphabet happens to keep this char as itself),
        // so mapping every char through it unconditionally would mis-decode
        // "café" as `[c, a, f, 0xE9]` — a lone UTF-8 continuation byte, not
        // valid UTF-8 on its own — instead of the correct
        // `[c, a, f, 0xC3, 0xA9]`.
        assert_eq!(
            hf_token_piece_bytes("café", false),
            "café".as_bytes().to_vec(),
            "byte_level=false must return the token's own UTF-8 bytes as-is"
        );
        assert_eq!(
            hf_token_piece_bytes("café", true),
            vec![b'c', b'a', b'f', 0xE9],
            "byte_level=true must unmap through the GPT-2 alphabet (this is \
             the behaviour a real byte-level vocabulary like Qwen3's needs)"
        );
    }

    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    #[test]
    fn hf_decoder_is_byte_level_detects_the_decoder_shape() {
        use tokenizers::decoders::byte_level::ByteLevel;
        use tokenizers::decoders::sequence::Sequence;
        use tokenizers::decoders::wordpiece::WordPiece;
        use tokenizers::models::bpe::BPE;
        use tokenizers::DecoderWrapper;

        // No decoder configured at all.
        let no_decoder = tokenizers::Tokenizer::new(BPE::default());
        assert!(!hf_decoder_is_byte_level(&no_decoder));

        // A non-byte-level decoder (WordPiece/Unigram-style vocabularies).
        let mut word_piece = tokenizers::Tokenizer::new(BPE::default());
        word_piece.with_decoder(Some(WordPiece::default()));
        assert!(!hf_decoder_is_byte_level(&word_piece));

        // The byte-level decoder itself.
        let mut byte_level = tokenizers::Tokenizer::new(BPE::default());
        byte_level.with_decoder(Some(ByteLevel::default()));
        assert!(hf_decoder_is_byte_level(&byte_level));

        // A `Sequence` that *contains* a `ByteLevel` stage (some real
        // `tokenizer.json` files wrap it alongside e.g. `Fuse`/`Strip`) must
        // still count — the recursive case.
        let mut sequence_with_byte_level = tokenizers::Tokenizer::new(BPE::default());
        sequence_with_byte_level.with_decoder(Some(DecoderWrapper::Sequence(Sequence::new(vec![
            DecoderWrapper::WordPiece(WordPiece::default()),
            DecoderWrapper::ByteLevel(ByteLevel::default()),
        ]))));
        assert!(hf_decoder_is_byte_level(&sequence_with_byte_level));

        // A `Sequence` that contains no `ByteLevel` stage must not.
        let mut sequence_without_byte_level = tokenizers::Tokenizer::new(BPE::default());
        sequence_without_byte_level.with_decoder(Some(DecoderWrapper::Sequence(Sequence::new(
            vec![DecoderWrapper::WordPiece(WordPiece::default())],
        ))));
        assert!(!hf_decoder_is_byte_level(&sequence_without_byte_level));
    }

    #[test]
    fn eos_ids_rejects_an_unset_default_that_aliases_an_ordinary_token() {
        // The tiny fixture never declares an eos_token in its JSON, so the
        // native backend keeps `TokenizerConfig::default()`'s `eos_token_id`
        // (2), which in this fixture aliases the ordinary, non-special token
        // "#" (id 2 is `#` here too, see `TINY_TOKENIZER_JSON`). Id 2 is
        // *not* flagged special, so `eos_ids` must not report it — this is
        // the regression test for the bug where mere vocabulary
        // resolvability (without checking `is_special`) let an unset
        // default alias a real, unrelated, non-special token through.
        let tok = tiny_native_bridge();
        assert!(
            !tok.is_special(2),
            "id 2 (\"#\") must not be special in this fixture"
        );
        let ids = tok.eos_ids();
        assert!(
            !ids.contains(&2),
            "an unset default eos_token_id that aliases an ordinary, non-special \
             token must not be reported as an eos id; got {ids:?}"
        );
    }

    #[test]
    fn eos_ids_on_the_real_qwen3_fixture_excludes_the_unset_default() {
        // `models/tokenizer.json` (the bundled real Qwen3 fixture) declares
        // no top-level `eos_token`, so the native backend keeps
        // `TokenizerConfig::default()`'s `eos_token_id` (2), which in this
        // real vocabulary resolves to the ordinary token "#" — not a
        // control token. `eos_ids()` must report exactly the two real
        // Qwen3 end-of-text specials (`<|im_end|>` = 151645,
        // `<|endoftext|>` = 151643) and must not contain 2.
        let Some(tok) = maybe_load_native_fixture() else {
            return;
        };
        assert!(
            !tok.is_special(2),
            "id 2 (\"#\") must not be special in the real fixture"
        );
        assert_eq!(
            tok.eos_ids(),
            vec![151645, 151643],
            "eos_ids() on the real Qwen3 fixture must be exactly [151645, 151643]"
        );
    }

    #[test]
    fn eos_ids_rejects_a_configured_eos_that_resolves_to_no_token() {
        const TINY_VOCAB_JSON: &str = r##"{
            "model": { "type": "BPE", "vocab": { "H": 0, "i": 1 }, "merges": [] },
            "added_tokens": [],
            "pre_tokenizer": { "type": "ByteLevel" },
            "decoder": { "type": "ByteLevel" }
        }"##;
        // A 2-entry vocabulary: `TokenizerConfig::default()`'s eos id (2) is
        // out of range and must not be reported as if it were real — this
        // is the one case `eos_ids` *can* and does filter.
        let tok = TokenizerBridge::native_from_json_str(TINY_VOCAB_JSON).expect("should load");
        let ids = tok.eos_ids();
        assert!(
            !ids.contains(&2),
            "an eos_token_id with no corresponding vocabulary entry must not be reported; got {ids:?}"
        );
    }

    #[test]
    fn eos_ids_finds_declared_end_of_text_markers() {
        const TOKENIZER_WITH_EOS_MARKERS: &str = r##"{
            "model": {
                "type": "BPE",
                "vocab": { "H": 0, "i": 1 },
                "merges": []
            },
            "added_tokens": [
                { "id": 100, "content": "<|im_end|>", "special": true },
                { "id": 101, "content": "<|endoftext|>", "special": true },
                { "id": 102, "content": "<tool_call>", "special": false }
            ],
            "pre_tokenizer": { "type": "ByteLevel" },
            "decoder": { "type": "ByteLevel" }
        }"##;
        let tok = TokenizerBridge::native_from_json_str(TOKENIZER_WITH_EOS_MARKERS)
            .expect("fixture should load");
        let ids = tok.eos_ids();
        assert!(
            ids.contains(&100),
            "<|im_end|> should be an eos id; got {ids:?}"
        );
        assert!(
            ids.contains(&101),
            "<|endoftext|> should be an eos id; got {ids:?}"
        );
        assert!(
            !ids.contains(&102),
            "a non-special added token must not be treated as an eos id"
        );
    }

    #[test]
    fn eos_ids_deduplicates() {
        // If a fixture's configured `eos_token_id` happens to coincide with
        // one of the marker spellings, it must appear only once.
        const TOKENIZER_EOS_ALIASES_MARKER: &str = r##"{
            "model": {
                "type": "BPE",
                "vocab": { "H": 0, "i": 1, "<|endoftext|>": 2 },
                "merges": []
            },
            "added_tokens": [
                { "id": 2, "content": "<|endoftext|>", "special": true }
            ],
            "pre_tokenizer": { "type": "ByteLevel" },
            "decoder": { "type": "ByteLevel" }
        }"##;
        let tok = TokenizerBridge::native_from_json_str(TOKENIZER_EOS_ALIASES_MARKER)
            .expect("fixture should load");
        let ids = tok.eos_ids();
        let count_of_2 = ids.iter().filter(|&&id| id == 2).count();
        assert_eq!(count_of_2, 1, "id 2 must not be listed twice; got {ids:?}");
    }

    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    #[test]
    fn is_special_and_piece_agree_across_backends_on_the_real_fixture() {
        // Native-vs-HF parity is only meaningful once both backends see the
        // exact same real vocabulary; this test is skip-safe like the
        // existing fixture-backed tests above when `models/tokenizer.json`
        // is absent (e.g. a freshly cloned tree or this sandboxed worktree).
        let Some(hf) = maybe_load_fixture() else {
            return;
        };
        let Some(native) = maybe_load_native_fixture() else {
            return;
        };
        assert_eq!(hf.backend(), TokenizerBackendKind::Hf);
        assert_eq!(native.backend(), TokenizerBackendKind::Native);

        for input in [
            "Hello, world!",
            "日本語処理を専門",
            "The quick brown fox jumps over the lazy dog.",
        ] {
            let hf_ids = hf.encode(input).expect("hf encode");
            let native_ids = native.encode(input).expect("native encode");
            assert_eq!(
                hf_ids, native_ids,
                "native and HF encode must agree on {input:?} before native can \
                 become the default backend (wave-1 addendum item (c))"
            );
            for &id in &hf_ids {
                assert_eq!(
                    hf.is_special(id),
                    native.is_special(id),
                    "is_special must agree on id {id} for {input:?}"
                );
                assert_eq!(
                    hf.piece(id),
                    native.piece(id),
                    "piece must agree on id {id} for {input:?}"
                );
            }
        }
    }
}
