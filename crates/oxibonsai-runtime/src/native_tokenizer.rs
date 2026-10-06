//! Native tokenizer bridge using oxibonsai-tokenizer (pure Rust BPE).
//!
//! This bridge allows the inference engine to use the project-native tokenizer
//! without any C/FFI dependencies, making it fully WASM-compatible.
//!
//! ## Overview
//!
//! [`NativeTokenizerBridge`] wraps an [`OxiTokenizer`] instance and optionally
//! a [`ChatTemplate`] to provide a unified encode/decode/chat-format API that
//! mirrors [`crate::tokenizer_bridge::TokenizerBridge`] but requires zero
//! C extensions and compiles to `wasm32-unknown-unknown`.
//!
//! ## Quick start
//!
//! ```rust
//! use oxibonsai_runtime::native_tokenizer::NativeTokenizerBridge;
//!
//! // Character-level fallback — no vocab file needed, great for testing.
//! let bridge = NativeTokenizerBridge::char_level_fallback();
//! let ids = bridge.encode("hello").expect("encode should succeed");
//! assert!(!ids.is_empty());
//! let text = bridge.decode(&ids).expect("decode should succeed");
//! assert_eq!(text, "hello");
//! ```

use oxibonsai_tokenizer::utils::ChatTemplate;
use oxibonsai_tokenizer::{OxiTokenizer, TokenizerConfig};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise from [`NativeTokenizerBridge`] operations.
#[derive(Debug, thiserror::Error)]
pub enum NativeTokenizerError {
    /// Wraps an error propagated from the underlying [`OxiTokenizer`].
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] oxibonsai_tokenizer::TokenizerError),

    /// Returned when [`NativeTokenizerBridge::format_chat`] is called but no
    /// chat template was configured at construction time.
    #[error("no chat template configured")]
    NoChatTemplate,

    /// Encoding failed for a reason not covered by the underlying tokenizer
    /// error type (e.g. an internal invariant violation).
    #[error("encode failed: {0}")]
    EncodeFailed(String),
}

// ── NativeTokenizerBridge ─────────────────────────────────────────────────────

/// Bridge between the inference engine and [`OxiTokenizer`].
///
/// Provides encode/decode/chat-format operations backed by the project-native
/// pure Rust BPE tokenizer.  The bridge is `Send + Sync` and holds no mutable
/// state after construction.
pub struct NativeTokenizerBridge {
    tokenizer: OxiTokenizer,
    chat_template: Option<ChatTemplate>,
}

impl NativeTokenizerBridge {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Create a bridge wrapping the provided [`OxiTokenizer`], with no chat
    /// template.
    ///
    /// Use [`NativeTokenizerBridge::with_chatml`] if you need ChatML
    /// formatting (e.g. for Qwen3 models).
    pub fn new(tokenizer: OxiTokenizer) -> Self {
        Self {
            tokenizer,
            chat_template: None,
        }
    }

    /// Create a minimal char-level fallback tokenizer.
    ///
    /// This uses [`OxiTokenizer::char_level_stub`] with a 512-token vocabulary
    /// and attaches no chat template.  Useful for unit tests and smoke-checks
    /// where a real vocab file is not required.
    pub fn char_level_fallback() -> Self {
        Self::new(OxiTokenizer::char_level_stub(512))
    }

    /// Create a bridge with a ChatML template pre-configured.
    ///
    /// This is the correct constructor for Qwen3 / OxiBonsai models, which
    /// use the `<|im_start|>role\ncontent<|im_end|>` format.
    pub fn with_chatml(tokenizer: OxiTokenizer) -> Self {
        Self {
            tokenizer,
            chat_template: Some(ChatTemplate::chatml()),
        }
    }

    /// Create a char-level fallback tokenizer with a ChatML template.
    ///
    /// Convenience constructor that combines `char_level_fallback` and
    /// `with_chatml` — handy for tests that exercise the chat-formatting
    /// path without a real vocab file.
    pub fn char_level_fallback_with_chatml() -> Self {
        Self::with_chatml(OxiTokenizer::char_level_stub(512))
    }

    /// Create a bridge from a JSON-serialized vocabulary and merge table,
    /// using the supplied configuration.
    ///
    /// `vocab_json`: `{ "token": id, … }`
    /// `merges_json`: `[["a", "b"], …]` (highest-priority merge first)
    pub fn from_json(
        vocab_json: &str,
        merges_json: &str,
        config: TokenizerConfig,
    ) -> Result<Self, NativeTokenizerError> {
        let tokenizer = OxiTokenizer::from_json(vocab_json, merges_json, config)?;
        Ok(Self::new(tokenizer))
    }

    // ── Core encode / decode ──────────────────────────────────────────────

    /// Encode a text string into a sequence of token IDs.
    ///
    /// Delegates directly to [`OxiTokenizer::encode`].
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, NativeTokenizerError> {
        self.tokenizer
            .encode(text)
            .map_err(NativeTokenizerError::Tokenizer)
    }

    /// Decode a sequence of token IDs back into a UTF-8 string.
    ///
    /// Special tokens (BOS, EOS, PAD, UNK) are silently skipped.
    /// Unknown IDs produce `\u{FFFD}` (replacement character).
    pub fn decode(&self, ids: &[u32]) -> Result<String, NativeTokenizerError> {
        self.tokenizer
            .decode(ids)
            .map_err(NativeTokenizerError::Tokenizer)
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> Result<String, NativeTokenizerError> {
        self.tokenizer
            .decode_token(id)
            .map_err(NativeTokenizerError::Tokenizer)
    }

    /// Encode a batch of texts, returning one `Vec<u32>` per input.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, NativeTokenizerError> {
        self.tokenizer
            .encode_batch(texts)
            .map_err(NativeTokenizerError::Tokenizer)
    }

    // ── Chat template ─────────────────────────────────────────────────────

    /// Format a list of `(role, content)` pairs into a single prompt string
    /// using the configured chat template.
    ///
    /// Returns [`NativeTokenizerError::NoChatTemplate`] if no template was
    /// provided at construction time.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxibonsai_runtime::native_tokenizer::NativeTokenizerBridge;
    ///
    /// let bridge = NativeTokenizerBridge::char_level_fallback_with_chatml();
    /// let prompt = bridge
    ///     .format_chat(&[("user", "Hello!")])
    ///     .expect("format_chat should succeed");
    /// assert!(prompt.contains("<|im_start|>user"));
    /// ```
    pub fn format_chat(&self, messages: &[(&str, &str)]) -> Result<String, NativeTokenizerError> {
        match &self.chat_template {
            Some(tmpl) => Ok(tmpl.format(messages)),
            None => Err(NativeTokenizerError::NoChatTemplate),
        }
    }

    // ── Informational helpers ─────────────────────────────────────────────

    /// Return the total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Return the BOS token ID from the underlying tokenizer configuration.
    pub fn bos_id(&self) -> u32 {
        self.tokenizer.bos_id()
    }

    /// Return the EOS token ID from the underlying tokenizer configuration.
    pub fn eos_id(&self) -> u32 {
        self.tokenizer.eos_id()
    }

    /// Return `true` if the given token ID is a special token (BOS/EOS/PAD/UNK).
    pub fn is_special(&self, id: u32) -> bool {
        self.tokenizer.is_special(id)
    }

    /// Return a reference to the underlying [`OxiTokenizer`].
    pub fn inner(&self) -> &OxiTokenizer {
        &self.tokenizer
    }

    /// Return a reference to the configured [`ChatTemplate`], if any.
    pub fn chat_template(&self) -> Option<&ChatTemplate> {
        self.chat_template.as_ref()
    }
}

// ── std::fmt::Debug ───────────────────────────────────────────────────────────

impl std::fmt::Debug for NativeTokenizerBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeTokenizerBridge")
            .field("vocab_size", &self.vocab_size())
            .field("has_chat_template", &self.chat_template.is_some())
            .finish()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_level_fallback_encode_decode() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        let ids = bridge.encode("hello").expect("encode should succeed");
        let text = bridge.decode(&ids).expect("decode should succeed");
        assert_eq!(text, "hello");
    }

    #[test]
    fn char_level_fallback_nonempty() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        let ids = bridge.encode("hello").expect("encode should succeed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn char_level_fallback_roundtrip_long() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        // Single word — no GPT-2 space-prefix complications.
        let original = "thequickbrownfox";
        let ids = bridge.encode(original).expect("encode should succeed");
        let decoded = bridge.decode(&ids).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn native_tokenizer_vocab_size_positive() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        assert!(bridge.vocab_size() > 0);
    }

    #[test]
    fn char_level_encode_consistent() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        let ids1 = bridge.encode("consistent").expect("first encode");
        let ids2 = bridge.encode("consistent").expect("second encode");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn char_level_special_chars() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        // Spaces, newlines, and basic Unicode should not panic.
        let ids = bridge.encode("hello world\nhow are you").expect("encode");
        assert!(!ids.is_empty());
    }

    #[test]
    fn native_tokenizer_decode_empty() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        let text = bridge.decode(&[]).expect("decode empty should succeed");
        assert_eq!(text, "");
    }

    #[test]
    fn native_tokenizer_format_chat_no_template() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        let result = bridge.format_chat(&[("user", "Hello!")]);
        assert!(matches!(result, Err(NativeTokenizerError::NoChatTemplate)));
    }

    #[test]
    fn native_tokenizer_with_chatml_format() {
        let bridge = NativeTokenizerBridge::char_level_fallback_with_chatml();
        let prompt = bridge
            .format_chat(&[("user", "Hello!")])
            .expect("format_chat should succeed");
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("<|im_end|>"));
    }

    #[test]
    fn native_tokenizer_encode_empty() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        // Encoding an empty string should succeed (may be empty or BOS-only).
        let ids = bridge.encode("").expect("encode empty should succeed");
        // No assertion on content — just that it does not error.
        let _ = ids;
    }

    #[test]
    fn debug_impl_shows_vocab_size() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        let dbg = format!("{bridge:?}");
        assert!(dbg.contains("vocab_size"));
        assert!(dbg.contains("has_chat_template"));
    }

    #[test]
    fn bos_eos_ids_accessible() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        assert_eq!(bridge.bos_id(), 1);
        assert_eq!(bridge.eos_id(), 2);
    }

    #[test]
    fn special_token_detection() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        assert!(bridge.is_special(0)); // UNK
        assert!(bridge.is_special(1)); // BOS
        assert!(bridge.is_special(2)); // EOS
        assert!(bridge.is_special(3)); // PAD
        assert!(!bridge.is_special(4)); // first real token
    }

    #[test]
    fn inner_ref_returns_tokenizer() {
        let bridge = NativeTokenizerBridge::char_level_fallback();
        // We can call inner() and get a consistent vocab_size.
        assert_eq!(bridge.inner().vocab_size(), bridge.vocab_size());
    }

    // ── Mandatory regression test ────────────────────────────────────────
    //
    // `TokenizerConfig::default()` sets `unk=0, bos=1, eos=2, pad=3`. A real
    // byte-level vocabulary that declares no explicit unk/bos/eos/pad token
    // (like Qwen3's real `tokenizer.json`) routinely assigns those exact ids
    // to ordinary punctuation ('!','"','#','$'). Before the root-cause fix
    // in `oxibonsai_tokenizer::tokenizer::build_special_ids` (deriving the
    // special-id set from the vocabulary rather than blindly trusting the
    // config quartet), those four ordinary bytes were silently classified
    // special and dropped by every decode path — including this bridge's,
    // since `NativeTokenizerBridge::decode` (above) calls
    // `OxiTokenizer::decode` directly. This test exercises that exact
    // collision at BOTH the `OxiTokenizer` level (via `from_json`, which
    // this bridge's own `from_json` constructor wraps) and the
    // `NativeTokenizerBridge::decode` level, so the fix is validated
    // through the actual type the qwen35 / Bonsai 2 vocabulary is routed
    // through, not just through `oxibonsai-tokenizer`'s own test suite.
    #[test]
    fn decode_survives_punctuation_colliding_with_default_special_ids() {
        // `Vocabulary::from_json`-compatible plain vocab: no `<...>`-shaped
        // entries, so nothing is auto-promoted to "special" — matching a
        // real byte-level vocab where '!'/'"'/'#'/'$' are ordinary tokens.
        // Two-hash raw-string delimiter: the JSON's `"#":2` entry contains
        // the literal byte sequence `"#`, which would otherwise prematurely
        // close a single-hash `r#"..."#` raw string right there.
        let vocab_json = r##"{"!":0,"\"":1,"#":2,"$":3,"a":4,"b":5}"##;
        let merges_json = "[]";
        let config = TokenizerConfig::default();
        assert_eq!(config.unk_token_id, 0);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.pad_token_id, 3);

        // Level 1: `OxiTokenizer::from_json` directly (what this bridge's
        // `from_json` wraps).
        let direct = OxiTokenizer::from_json(vocab_json, merges_json, TokenizerConfig::default())
            .expect("OxiTokenizer::from_json should succeed");
        let ids = direct.encode("!\"#$ab").expect("encode should succeed");
        let decoded = direct.decode(&ids).expect("decode should succeed");
        assert_eq!(
            decoded, "!\"#$ab",
            "OxiTokenizer::decode must not drop punctuation colliding with \
             TokenizerConfig::default()'s unk/bos/eos/pad ids"
        );

        // Level 2: through the bridge this crate actually routes the
        // native/GGUF-vocab tokenizer through.
        let bridge = NativeTokenizerBridge::from_json(vocab_json, merges_json, config)
            .expect("from_json should succeed");
        let ids = bridge.encode("!\"#$ab").expect("encode should succeed");
        let decoded = bridge.decode(&ids).expect("decode should succeed");
        assert_eq!(
            decoded, "!\"#$ab",
            "NativeTokenizerBridge::decode must not drop punctuation \
             colliding with the default special-id quartet"
        );
    }
}
