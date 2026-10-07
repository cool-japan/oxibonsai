//! Tokenizer bridge: Pure-Rust native backend, optional HuggingFace backend.
//!
//! [`TokenizerBridge`] is a struct wrapping a private `TokenizerBackend`
//! enum over two interchangeable backends (a struct rather than a bare public
//! enum so it can carry the chat-template/special-id fields below —
//! [`TokenizerBridge::backend`] is the read-only accessor for which one is
//! active):
//!
//! * The native backend — the workspace's own Pure-Rust BPE
//!   ([`oxibonsai_tokenizer::OxiTokenizer`]).  Always compiled in, including
//!   on `wasm32` targets and in a `--no-default-features` build.
//! * The HF backend — HuggingFace `tokenizers`.  Compiled only when
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
//!
//! On top of the backend, this type also carries the model's resolved chat
//! template, its `<think>`/`</think>`/`<tool_call>`/`</tool_call>` special
//! ids, the class the vocabulary assigns each added token
//! ([`TokenizerBridge::token_class`]) and the reasoning-parser rules its
//! template implies ([`TokenizerBridge::reasoning_format`]): see
//! [`TokenizerBridge::native_from_gguf_metadata`],
//! [`TokenizerBridge::with_chat_template`] and
//! [`TokenizerBridge::resolved_chat_template`].

use crate::error::{RuntimeError, RuntimeResult};
use crate::reasoning::ReasoningFormat;
use oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate;
use oxibonsai_tokenizer::OxiTokenizer;
use std::collections::HashMap;

/// Shared chat-prompt rendering pipeline, built on top of
/// [`TokenizerBridge`] — see that module's own doc. `pub(crate)` (not
/// `pub`): an internal seam `server::chat` and `api_extensions` share,
/// not part of this crate's public API surface.
///
/// `#[cfg(feature = "server")]`: unlike the rest of this file,
/// `chat_render` reaches into `crate::server` (gated the same way,
/// `lib.rs`) and uses `serde_json`'s `raw_value` feature (pulled in only
/// through `axum`/the `server` feature's own dependency set) — this module
/// alone would otherwise break the `--no-default-features` / `wasm32`
/// build this file's own module doc promises stays available
/// (`runtime_compiles_for_wasm32_unknown_unknown_no_default_features`).
#[cfg(feature = "server")]
pub(crate) mod chat_render;

/// The class a vocabulary assigns one token: llama.cpp's
/// `LLAMA_TOKEN_TYPE_*` for a vocabulary built from GGUF metadata
/// (`tokenizer.ggml.token_type`), or the `added_tokens` flags of a
/// `tokenizer.json` (an added token flagged `special` is [`Self::Control`],
/// any other added token [`Self::UserDefined`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocabTokenClass {
    /// Ordinary vocabulary text.
    Normal,
    /// The unknown-token placeholder.
    Unknown,
    /// A control marker (`<|im_start|>`, `<|endoftext|>`, `<|image_pad|>`,
    /// …): carved out atomically on encode and skipped on decode.
    Control,
    /// A user-defined added token (`<think>`, `<tool_call>`, …): carved out
    /// atomically on encode, but ordinary text on decode.
    UserDefined,
    /// A reserved slot the model was never trained on.
    Unused,
    /// A raw-byte fallback token.
    Byte,
}

impl VocabTokenClass {
    /// The class of llama.cpp token type `value` (`None` for a value outside
    /// the six defined types).
    pub fn from_gguf_token_type(value: i32) -> Option<Self> {
        Some(match value {
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
/// tokenizer can still reach it through `TokenizerBridge::hf` (available with
/// the `hf-tokenizer` feature).
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
        match &self.bridge.backend {
            TokenizerBackend::Native(tok) => {
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
            TokenizerBackend::Hf(tok) => tok
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
        match &self.bridge.backend {
            TokenizerBackend::Native(tok) => tok.vocab().get_id(token),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => tok.token_to_id(token),
        }
    }

    /// Resolve a token id to its literal token string, if it exists.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        match &self.bridge.backend {
            TokenizerBackend::Native(tok) => tok.vocab().get_token(id).map(ToOwned::to_owned),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => tok.id_to_token(id),
        }
    }
}

/// The two interchangeable tokenizer backends — see the [module docs](self).
///
/// Private to this module: [`TokenizerBridge`] (the public type every other
/// crate/module constructs and calls) wraps this rather than being this
/// enum directly, so it can carry a resolved chat template and special-token
/// ids *alongside* whichever backend is loaded (RT-09 / cli-11) —
/// see [`TokenizerBridge`]'s own doc for why.
enum TokenizerBackend {
    /// Pure-Rust BPE backend — always available.
    Native(Box<OxiTokenizer>),
    /// HuggingFace `tokenizers` backend (feature `hf-tokenizer`, non-wasm).
    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    Hf(Box<tokenizers::Tokenizer>),
}

/// Tokenizer used by the inference engine, server and CLI.
///
/// See the [module docs](self) for the backend split (`TokenizerBackend`).
///
/// Beyond the raw encode/decode backend, a bridge optionally carries the
/// loaded model's own resolved chat template and its `<think>`/`</think>`/
/// `<tool_call>`/`</tool_call>` special-token ids — the model's
/// `ResolvedChatTemplate` resolved once per loaded model and carried here
/// (cli-11: the single runtime entry point the CLI's `--think` /
/// `--no-think` / `--reasoning-effort` / `--tools` flags need).
/// [`Self::with_chat_template`] attaches one explicitly (the
/// seam a caller with a `MetadataStore` — e.g. GGUF-loading code — plugs
/// into); [`Self::native_from_gguf_metadata`] resolves both the tokenizer
/// AND the template from the same GGUF metadata in one call, with the
/// named ChatML/Qwen3 fallback for models that ship none (via
/// [`Self::resolved_chat_template`], which never returns `None` to a
/// caller). A bridge built through any of the other constructors — which
/// have no `MetadataStore` to resolve a template from — still resolves its
/// think/tool-call ids from whichever vocabulary it loaded (`None` for a
/// vocabulary that defines no such tokens — RT-10's "models without
/// `<think>`" correction; the shipped Qwen3 1.7B/8B `tokenizer.json` does
/// define them) and falls back to the built-in template on
/// [`Self::resolved_chat_template`].
pub struct TokenizerBridge {
    backend: TokenizerBackend,
    /// The loaded model's own resolved chat template, when one has been
    /// attached via [`Self::with_chat_template`] /
    /// [`Self::native_from_gguf_metadata`]. `None` for every other
    /// constructor (no `MetadataStore` was available to resolve one from);
    /// [`Self::resolved_chat_template`] is the caller-facing accessor that
    /// never exposes this `None` state directly.
    chat_template: Option<ResolvedChatTemplate>,
    /// This vocabulary's `<think>` token id, if it defines one.
    think_open_id: Option<u32>,
    /// This vocabulary's `</think>` token id, if it defines one.
    think_close_id: Option<u32>,
    /// This vocabulary's `<tool_call>` token id, if it defines one.
    tool_call_open_id: Option<u32>,
    /// This vocabulary's `</tool_call>` token id, if it defines one.
    tool_call_close_id: Option<u32>,
    /// Every token id the vocabulary classifies as anything but
    /// [`VocabTokenClass::Normal`] ([`Self::token_class`]).
    token_classes: HashMap<u32, VocabTokenClass>,
    /// The reasoning-parser rules the resolved chat template implies
    /// ([`Self::reasoning_format`]).
    reasoning_format: ReasoningFormat,
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
    /// Finish building a bridge from an already-constructed backend: resolve
    /// this vocabulary's `<think>`/`</think>`/`<tool_call>`/`</tool_call>`
    /// ids (`None` for any the vocabulary does not define — RT-10) and
    /// start with no chat template attached. Every constructor below funnels
    /// through this one place so the id-resolution logic lives exactly once.
    fn from_backend(backend: TokenizerBackend) -> Self {
        let mut bridge = Self {
            backend,
            chat_template: None,
            think_open_id: None,
            think_close_id: None,
            tool_call_open_id: None,
            tool_call_close_id: None,
            token_classes: HashMap::new(),
            reasoning_format: fallback_reasoning_format(),
        };
        bridge.think_open_id = bridge.inner().token_to_id("<think>");
        bridge.think_close_id = bridge.inner().token_to_id("</think>");
        bridge.tool_call_open_id = bridge.inner().token_to_id("<tool_call>");
        bridge.tool_call_close_id = bridge.inner().token_to_id("</tool_call>");
        bridge.token_classes = bridge
            .inner()
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, added)| {
                let class = if added.special {
                    VocabTokenClass::Control
                } else {
                    VocabTokenClass::UserDefined
                };
                (id, class)
            })
            .collect();
        bridge
    }

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
            Ok(Self::from_backend(TokenizerBackend::Hf(Box::new(inner))))
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
        Ok(Self::from_backend(TokenizerBackend::Native(Box::new(
            inner,
        ))))
    }

    /// Load a tokenizer from the *contents* of a HuggingFace-format
    /// `tokenizer.json`, using the Pure-Rust native backend.
    ///
    /// The filesystem-free variant of [`Self::native_from_file`] — the form
    /// `wasm32` builds and embedded fixtures need.
    pub fn native_from_json_str(json: &str) -> RuntimeResult<Self> {
        let inner = OxiTokenizer::from_hf_tokenizer_json(json)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        Ok(Self::from_backend(TokenizerBackend::Native(Box::new(
            inner,
        ))))
    }

    /// Build a bridge over the native backend directly from a GGUF's own
    /// embedded tokenizer metadata (`tokenizer.ggml.*`) — the same source a
    /// loaded model's weights come from — and additionally resolve its
    /// `tokenizer.chat_template` (cli-11).
    ///
    /// This is the single entry point that closes RT-09/TOK-07/cli-03's "no
    /// runtime API hands \[the caller\] the loaded model's template" gap:
    /// [`Self::resolved_chat_template`] on the returned bridge never falls
    /// through to `None` (a model shipping none still gets the named
    /// Qwen3/ChatML fallback), and [`Self::think_open_id`] /
    /// [`Self::think_close_id`] / [`Self::tool_call_open_id`] /
    /// [`Self::tool_call_close_id`] expose the vocabulary's own special ids
    /// for `<think>`/`</think>`/`<tool_call>`/`</tool_call>` — everything
    /// the CLI's `--think`/`--no-think`/`--reasoning-effort`/`--tools` flags
    /// and this crate's own chat-completion handlers need.
    ///
    /// # Errors
    /// Propagates a tokenizer-construction failure, or a shipped
    /// `tokenizer.chat_template` this engine's Jinja subset cannot compile
    /// (never silently substitutes the fallback for a template that DID
    /// ship; see [`ResolvedChatTemplate::from_gguf`]).
    pub fn native_from_gguf_metadata(md: &oxibonsai_core::MetadataStore) -> RuntimeResult<Self> {
        let inner = OxiTokenizer::from_gguf_metadata(md)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        let template = ResolvedChatTemplate::from_gguf(md)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        let mut bridge = Self::from_backend(TokenizerBackend::Native(Box::new(inner)))
            .with_chat_template(template);
        bridge.apply_gguf_token_types(md);
        Ok(bridge)
    }

    /// Overlay the vocabulary's own `tokenizer.ggml.token_type` classes, when
    /// the metadata carries the array: every id it types as anything but
    /// `NORMAL` gets that class (so `UNUSED` and `BYTE` slots are classified
    /// too), and an added token it types `NORMAL` loses the class its
    /// added-token flags implied. Without the array the added-token flags
    /// stand.
    fn apply_gguf_token_types(&mut self, md: &oxibonsai_core::MetadataStore) {
        let Ok(types) = md.get_i32_array(oxibonsai_tokenizer::gguf_vocab::KEY_TOKEN_TYPE) else {
            return;
        };
        for (index, value) in types.into_iter().enumerate() {
            let Ok(id) = u32::try_from(index) else {
                break;
            };
            match VocabTokenClass::from_gguf_token_type(value) {
                Some(VocabTokenClass::Normal) => {
                    self.token_classes.remove(&id);
                }
                Some(class) => {
                    self.token_classes.insert(id, class);
                }
                None => {}
            }
        }
    }

    /// Wrap an already-constructed native tokenizer.
    pub fn from_native_tokenizer(tokenizer: OxiTokenizer) -> Self {
        Self::from_backend(TokenizerBackend::Native(Box::new(tokenizer)))
    }

    /// Wrap an already-constructed HuggingFace tokenizer.
    #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
    pub fn from_hf_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self::from_backend(TokenizerBackend::Hf(Box::new(tokenizer)))
    }

    /// Attach a resolved chat template (e.g. from
    /// [`ResolvedChatTemplate::from_gguf`], when the caller has a
    /// `MetadataStore` this constructor's own [`Self::native_from_gguf_metadata`]
    /// did not build the bridge from). Builder-style; does not touch the
    /// already-resolved think/tool-call ids (those come from the
    /// vocabulary, not the template).
    #[must_use]
    pub fn with_chat_template(mut self, template: ResolvedChatTemplate) -> Self {
        self.reasoning_format = reasoning_format_of(&template);
        self.chat_template = Some(template);
        self
    }

    /// The reasoning-parser rules the resolved chat template implies: the
    /// reference server parses output with its Qwen3-Coder parser for a
    /// template that teaches the XML tool-call form (Bonsai 2's own), and
    /// with its generic tagged parser otherwise — see
    /// [`crate::reasoning::ReasoningFormat`] for what each strips.
    pub fn reasoning_format(&self) -> ReasoningFormat {
        self.reasoning_format
    }

    /// The class the vocabulary assigns token `id` ([`VocabTokenClass`]),
    /// or `None` for an id outside the vocabulary.
    pub fn token_class(&self, id: u32) -> Option<VocabTokenClass> {
        if let Some(class) = self.token_classes.get(&id) {
            return Some(*class);
        }
        self.inner()
            .id_to_token(id)
            .map(|_| VocabTokenClass::Normal)
    }

    /// Every token id the vocabulary classifies as anything but
    /// [`VocabTokenClass::Normal`], with its class (unordered).
    pub fn token_classes(&self) -> impl Iterator<Item = (u32, VocabTokenClass)> + '_ {
        self.token_classes.iter().map(|(id, class)| (*id, *class))
    }

    /// The model's own resolved chat template, or the named Qwen3/ChatML
    /// fallback when none was ever attached (the ChatML/Qwen3 template, for
    /// models that ship none). Never `None` — this is the accessor every
    /// prompt-rendering
    /// call site should use, rather than matching on `Self::chat_template`'s
    /// private `Option` directly. Cheap to call per request: the returned
    /// value's expensive part (a compiled [`oxibonsai_tokenizer::jinja::JinjaTemplate`])
    /// is behind an `Arc` that this only clones.
    pub fn resolved_chat_template(&self) -> ResolvedChatTemplate {
        self.chat_template
            .clone()
            .unwrap_or_else(ResolvedChatTemplate::default_fallback)
    }

    /// This vocabulary's `<think>` token id (single-token, design Appendix
    /// A.1), or `None` when it defines no such token (RT-10's "models
    /// without `<think>`"). Bonsai 2 resolves 248068; the shipped Qwen3
    /// 1.7B/8B `tokenizer.json` resolves 151667 (a non-`special` added
    /// token, which such a model emits itself under the ChatML fallback).
    pub fn think_open_id(&self) -> Option<u32> {
        self.think_open_id
    }

    /// This vocabulary's `</think>` token id, or `None`. See
    /// [`Self::think_open_id`].
    pub fn think_close_id(&self) -> Option<u32> {
        self.think_close_id
    }

    /// This vocabulary's `<tool_call>` token id, or `None` when it defines
    /// no such token.
    pub fn tool_call_open_id(&self) -> Option<u32> {
        self.tool_call_open_id
    }

    /// This vocabulary's `</tool_call>` token id, or `None`. See
    /// [`Self::tool_call_open_id`].
    pub fn tool_call_close_id(&self) -> Option<u32> {
        self.tool_call_close_id
    }

    /// Which backend this bridge is using.
    pub fn backend(&self) -> TokenizerBackendKind {
        match &self.backend {
            TokenizerBackend::Native(_) => TokenizerBackendKind::Native,
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(_) => TokenizerBackendKind::Hf,
        }
    }

    /// Encode text to token IDs (no special tokens added).
    pub fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        match &self.backend {
            TokenizerBackend::Native(tok) => tok
                .encode(text)
                .map_err(|e| RuntimeError::Tokenizer(e.to_string())),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => {
                let encoding = tok
                    .encode(text, false)
                    .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
                Ok(encoding.get_ids().to_vec())
            }
        }
    }

    /// Decode token IDs to text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> RuntimeResult<String> {
        match &self.backend {
            TokenizerBackend::Native(tok) => Ok(native_decode(tok, ids, true)),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => tok
                .decode(ids, true)
                .map_err(|e| RuntimeError::Tokenizer(e.to_string())),
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match &self.backend {
            TokenizerBackend::Native(tok) => tok.vocab_size(),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => tok.get_vocab_size(true),
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
        match &self.backend {
            TokenizerBackend::Hf(tok) => Some(tok),
            TokenizerBackend::Native(_) => None,
        }
    }

    /// The underlying native tokenizer, when this bridge is native-backed.
    pub fn native(&self) -> Option<&OxiTokenizer> {
        match &self.backend {
            TokenizerBackend::Native(tok) => Some(tok),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(_) => None,
        }
    }

    /// Whether `id` is a special/control token in the loaded vocabulary
    /// (TOK-15) — e.g. `<|im_start|>`, `<|endoftext|>`. Driven from the
    /// vocabulary actually loaded, not a hardcoded id range: Qwen3-family
    /// specials sit in one block and Bonsai 2's sit in another
    /// (`248044..248076`), and this works for either because it asks the
    /// backend's own added-vocabulary registry rather than assuming a range.
    pub fn is_special(&self, id: u32) -> bool {
        match &self.backend {
            TokenizerBackend::Native(tok) => tok.vocab().is_special_id(id),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => match tok.id_to_token(id) {
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
        match &self.backend {
            TokenizerBackend::Native(tok) => native_token_piece_bytes(tok, id),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => match tok.id_to_token(id) {
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
    /// (`END_OF_TEXT_MARKER_SPELLINGS`) that this vocabulary also defines
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
    /// "was this set" flag.
    pub fn eos_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        // `match`, not `if let`: with the `hf-tokenizer` feature off,
        // `Native` is this enum's only variant, and an `if let` against it
        // would be flagged `irrefutable_let_patterns` under `-D warnings`.
        match &self.backend {
            TokenizerBackend::Native(tok) => {
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
            TokenizerBackend::Hf(_) => {
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
        match &self.backend {
            TokenizerBackend::Native(tok) => native_step_decode(tok, state, id),
            #[cfg(all(feature = "hf-tokenizer", not(target_arch = "wasm32")))]
            TokenizerBackend::Hf(tok) => tokenizers::step_decode_stream(
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
            .field("has_chat_template", &self.chat_template.is_some())
            .field("think_ids", &(self.think_open_id, self.think_close_id))
            .field(
                "tool_call_ids",
                &(self.tool_call_open_id, self.tool_call_close_id),
            )
            .field("reasoning_format", &self.reasoning_format)
            .finish()
    }
}

/// The reasoning format `template` implies — see
/// [`TokenizerBridge::reasoning_format`].
fn reasoning_format_of(template: &ResolvedChatTemplate) -> ReasoningFormat {
    if template.uses_xml_tool_calls() {
        ReasoningFormat::Qwen3Coder
    } else {
        ReasoningFormat::Tagged
    }
}

/// [`reasoning_format_of`] the built-in fallback template (the template a
/// bridge renders until one is attached), classified once per process.
fn fallback_reasoning_format() -> ReasoningFormat {
    static FALLBACK: std::sync::OnceLock<ReasoningFormat> = std::sync::OnceLock::new();
    *FALLBACK.get_or_init(|| reasoning_format_of(&ResolvedChatTemplate::default_fallback()))
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
mod tests;
