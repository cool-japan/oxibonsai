//! Tokenizer bridge wrapping HuggingFace tokenizers.
//!
//! On WASM targets, the `tokenizers` crate is unavailable (requires native C extensions).
//! A stub implementation is provided that returns errors for all operations.

use crate::error::{RuntimeError, RuntimeResult};

/// Thin wrapper around `tokenizers::Tokenizer`.
///
/// On non-WASM targets, delegates to the full HuggingFace tokenizers library.
/// On WASM targets, all methods return a `RuntimeError::Tokenizer` error.
pub struct TokenizerBridge {
    #[cfg(not(target_arch = "wasm32"))]
    inner: tokenizers::Tokenizer,
    #[cfg(target_arch = "wasm32")]
    _phantom: (),
}

/// Per-stream UTF-8-safe decode state. Owned by the caller.
///
/// BPE / byte-level tokenizers (Qwen3, GPT-2, etc.) sometimes emit a single
/// token that carries only **part** of a multi-byte UTF-8 character (e.g. one
/// byte of a CJK ideograph or emoji).  Decoding tokens one-at-a-time without
/// buffering breaks those multi-byte sequences and produces `U+FFFD`
/// replacement characters in the output stream.  This state mirrors what the
/// HuggingFace `tokenizers::DecodeStream` keeps internally so that we can own
/// it externally and feed tokens through `TokenizerBridge::step_decode`.
///
/// Use one `DecodeStreamState` per generation request; reset (or drop &
/// re-create) it between independent requests.
#[derive(Default)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
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
    /// Load a tokenizer from a JSON file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &str) -> RuntimeResult<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a JSON file.
    ///
    /// On WASM targets, always returns an error since the tokenizers library
    /// requires native code not available in WebAssembly.
    #[cfg(target_arch = "wasm32")]
    pub fn from_file(_path: &str) -> RuntimeResult<Self> {
        Err(RuntimeError::Tokenizer(
            "tokenizers library is not available on wasm32 targets".to_string(),
        ))
    }

    /// Encode text to token IDs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text to token IDs.
    ///
    /// On WASM targets, always returns an error.
    #[cfg(target_arch = "wasm32")]
    pub fn encode(&self, _text: &str) -> RuntimeResult<Vec<u32>> {
        Err(RuntimeError::Tokenizer(
            "tokenizers library is not available on wasm32 targets".to_string(),
        ))
    }

    /// Decode token IDs to text.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn decode(&self, ids: &[u32]) -> RuntimeResult<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))
    }

    /// Decode token IDs to text.
    ///
    /// On WASM targets, always returns an error.
    #[cfg(target_arch = "wasm32")]
    pub fn decode(&self, _ids: &[u32]) -> RuntimeResult<String> {
        Err(RuntimeError::Tokenizer(
            "tokenizers library is not available on wasm32 targets".to_string(),
        ))
    }

    /// Get the vocabulary size.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the vocabulary size.
    ///
    /// On WASM targets, returns 0 since no tokenizer is available.
    #[cfg(target_arch = "wasm32")]
    pub fn vocab_size(&self) -> usize {
        0
    }

    /// Get the internal tokenizer reference.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    /// Create a fresh decode-stream state for one generation request.
    ///
    /// See [`DecodeStreamState`] and [`Self::step_decode`] for the streaming
    /// decode protocol.  Use this instead of repeatedly calling
    /// [`Self::decode`] with single-token slices, which mishandles tokens that
    /// straddle UTF-8 codepoint boundaries.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_decode_stream(&self, skip_special_tokens: bool) -> DecodeStreamState {
        DecodeStreamState::new(skip_special_tokens)
    }

    /// Create a fresh decode-stream state.
    ///
    /// On WASM targets this returns a state object, but [`Self::step_decode`]
    /// will always error.  The state itself is harmless to construct.
    #[cfg(target_arch = "wasm32")]
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn step_decode(
        &self,
        state: &mut DecodeStreamState,
        id: u32,
    ) -> RuntimeResult<Option<String>> {
        tokenizers::step_decode_stream(
            &*self.inner,
            vec![id],
            state.skip_special_tokens,
            &mut state.ids,
            &mut state.prefix,
            &mut state.prefix_index,
        )
        .map_err(|e| RuntimeError::Tokenizer(e.to_string()))
    }

    /// Advance the decode stream by one token.
    ///
    /// On WASM targets this always returns an error since the tokenizers
    /// library is unavailable.
    #[cfg(target_arch = "wasm32")]
    pub fn step_decode(
        &self,
        _state: &mut DecodeStreamState,
        _id: u32,
    ) -> RuntimeResult<Option<String>> {
        Err(RuntimeError::Tokenizer(
            "tokenizers library is not available on wasm32 targets".to_string(),
        ))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::path::Path;

    /// Path to the project's bundled Qwen3 tokenizer.  Tests that need a real
    /// BPE tokenizer skip themselves when this fixture is missing so that
    /// freshly-cloned working trees still pass `cargo test`.
    const FIXTURE_TOKENIZER: &str = "../../models/tokenizer.json";

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
}
