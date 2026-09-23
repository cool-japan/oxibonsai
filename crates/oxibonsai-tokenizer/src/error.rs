//! Error types for the OxiBonsai tokenizer.

use thiserror::Error;

/// All errors that can occur during tokenization operations.
///
/// This enum is marked `#[non_exhaustive]` so that new variants can be added
/// in future minor releases without a breaking semver change.  Consumers must
/// always include a catch-all arm when matching on [`TokenizerError`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenizerError {
    /// A token string was not found in the vocabulary.
    #[error("unknown token: {0:?}")]
    UnknownToken(String),

    /// The vocabulary data is malformed or inconsistent.
    #[error("invalid vocabulary: {0}")]
    InvalidVocab(String),

    /// Encoding of input text failed.
    #[error("encode failed: {0}")]
    EncodeFailed(String),

    /// Decoding of token IDs failed.
    #[error("decode failed: {0}")]
    DecodeFailed(String),

    /// JSON deserialization failed.
    #[error("invalid JSON: {0}")]
    InvalidJson(String),

    /// A HuggingFace `tokenizer.json` file could not be parsed or interpreted.
    ///
    /// Includes missing required fields (`model`, `vocab`, `merges`), unsupported
    /// BPE types, and malformed merge entries.
    #[error("HF tokenizer format error: {0}")]
    HfFormat(String),

    /// A GGUF-embedded tokenizer definition (`tokenizer.ggml.*`) could not be
    /// parsed or interpreted (B2-08/TOK-05's format, distinct from
    /// [`Self::HfFormat`]'s `tokenizer.json`: both describe "the serialized
    /// tokenizer definition could not be parsed", just from a different
    /// container).
    ///
    /// Added by B2-13 per the wave-2.5 addendum: `gguf_vocab.rs`'s own
    /// loader was written before this variant existed (that file was
    /// outside its owning package's `owned_files`, so it could not add the
    /// variant itself) and reused `HfFormat` with a `"GGUF: "` message
    /// prefix instead — `gguf_vocab.rs` is not in *this* package's
    /// `owned_files` either, so its two `.map_err` call sites still emit
    /// `HfFormat("GGUF: …")` rather than this variant; recorded as a
    /// deviation with the exact one-line diff each site needs
    /// (`.map_err(|e| TokenizerError::HfFormat(format!("GGUF: {e}")))` →
    /// `.map_err(|e| TokenizerError::GgufFormat(e.to_string()))`).
    #[error("GGUF tokenizer format error: {0}")]
    GgufFormat(String),

    /// A streaming decoder received token IDs that together do not form a
    /// complete UTF-8 sequence and further bytes are required to finish.
    ///
    /// This variant is primarily returned by [`crate::streaming::StreamingDecoder::finish`]
    /// when the stream ends mid-character.
    #[error("incomplete UTF-8 sequence at end of stream")]
    IncompleteUtf8,

    /// Rendering a chat-template failed (missing variable, bad syntax, ...).
    #[error("template render failed: {0}")]
    TemplateRender(String),

    /// An underlying I/O operation (file read, etc.) failed.
    ///
    /// We wrap the `io::Error` as a `String` so that `TokenizerError` can
    /// continue to derive `Clone, PartialEq, Eq` — `std::io::Error` itself
    /// does not implement those traits.
    #[error("I/O error: {0}")]
    Io(String),
}

impl From<std::io::Error> for TokenizerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

/// Convenience result alias for tokenizer operations.
pub type TokenizerResult<T> = Result<T, TokenizerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_unknown_token() {
        let e = TokenizerError::UnknownToken("foo".to_owned());
        let s = format!("{e}");
        assert!(s.contains("foo"));
    }

    #[test]
    fn display_hf_format() {
        let e = TokenizerError::HfFormat("bad merges".to_owned());
        let s = format!("{e}");
        assert!(s.contains("bad merges"));
        assert!(s.contains("HF"));
    }

    #[test]
    fn display_gguf_format() {
        let e = TokenizerError::GgufFormat("missing tokenizer.ggml.tokens".to_owned());
        let s = format!("{e}");
        assert!(s.contains("missing tokenizer.ggml.tokens"));
        assert!(s.contains("GGUF"));
    }

    #[test]
    fn gguf_format_is_distinct_from_hf_format() {
        let a = TokenizerError::GgufFormat("x".to_owned());
        let b = TokenizerError::HfFormat("x".to_owned());
        assert_ne!(a, b);
    }

    #[test]
    fn display_incomplete_utf8() {
        let e = TokenizerError::IncompleteUtf8;
        let s = format!("{e}");
        assert!(s.to_ascii_lowercase().contains("utf-8"));
    }

    #[test]
    fn display_template_render() {
        let e = TokenizerError::TemplateRender("no such var".to_owned());
        let s = format!("{e}");
        assert!(s.contains("no such var"));
    }

    #[test]
    fn io_error_conversion_preserves_message() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let tok_err: TokenizerError = io_err.into();
        match tok_err {
            TokenizerError::Io(msg) => assert!(msg.contains("missing")),
            other => panic!("expected Io variant, got {other:?}"),
        }
    }

    #[test]
    fn tokenizer_error_is_clone() {
        let e = TokenizerError::InvalidVocab("oops".to_owned());
        let c = e.clone();
        assert_eq!(e, c);
    }

    #[test]
    fn tokenizer_error_equality() {
        let a = TokenizerError::EncodeFailed("x".to_owned());
        let b = TokenizerError::EncodeFailed("x".to_owned());
        let c = TokenizerError::EncodeFailed("y".to_owned());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
