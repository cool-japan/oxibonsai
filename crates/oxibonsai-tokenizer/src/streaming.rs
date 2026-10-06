//! UTF-8-safe streaming decoder.
//!
//! When a server emits tokens one at a time, naive `decode(&[id])` can return
//! strings with invalid UTF-8 because a single BPE token may hold *part* of a
//! multi-byte codepoint (common for CJK / emoji output).  The decoder in this
//! module keeps a small byte buffer across calls and only flushes characters
//! that form a complete UTF-8 sequence.
//!
//! ## Robustness (TOK-02)
//!
//! A single structurally-invalid byte (not merely an as-yet-incomplete
//! multi-byte prefix) is recovered from immediately: it is replaced with
//! `\u{FFFD}` and dropped, rather than being held in `pending` forever
//! waiting for a completion that can never come (the previous behaviour,
//! which let an adversarial or corrupt byte stream grow `pending` without
//! bound and poison the rest of the generation). `Self::finish` always
//! returns a best-effort `String` — it can no longer fail; any genuinely
//! incomplete trailing sequence (the stream legitimately ended mid-character,
//! e.g. truncated at `max_tokens`) is replaced with `\u{FFFD}` the same way
//! `Self::finish_lossy` already did.
//!
//! ## Usage
//!
//! ```rust
//! use oxibonsai_tokenizer::OxiTokenizer;
//!
//! let tok = OxiTokenizer::char_level_stub(256);
//! let ids = tok.encode("Hello!").expect("encode");
//! let mut dec = tok.streaming_decoder();
//! let mut out = String::new();
//! for id in &ids {
//!     if let Some(piece) = dec.push_token(*id) {
//!         out.push_str(&piece);
//!     }
//! }
//! // `finish()` never fails — it always returns the best-effort tail.
//! out.push_str(&dec.finish().unwrap_or_default());
//! assert_eq!(out, "Hello!");
//! ```

use crate::{error::TokenizerResult, tokenizer::OxiTokenizer};

/// The longest a legal UTF-8 sequence can be (bytes). Used as the trigger
/// for forcibly recovering from a `pending` buffer that has grown at least
/// this large while `str::from_utf8` still reports a structurally-undecided
/// prefix (`valid_up_to() == 0`, `error_len() == None`) — such a prefix
/// cannot legitimately be "waiting for more continuation bytes" past this
/// length, so treating it as corrupt (rather than pinning `pending` open
/// forever) is the only sound interpretation (TOK-02 verdict correction).
const MAX_UTF8_SEQUENCE_LEN: usize = 4;

/// A streaming decoder that yields well-formed UTF-8 slices as tokens arrive.
///
/// The decoder holds a reference to its parent [`OxiTokenizer`] so that
/// special-token handling, vocabulary lookup and byte-level decoding remain
/// consistent with [`OxiTokenizer::decode`].
pub struct StreamingDecoder<'a> {
    tokenizer: &'a OxiTokenizer,
    /// Bytes that have been decoded but not yet emitted because they are
    /// part of an incomplete UTF-8 sequence.
    pending: Vec<u8>,
    /// Total bytes the decoder has seen across the stream (for diagnostics).
    total_bytes: usize,
    /// Total tokens the decoder has seen across the stream.
    total_tokens: usize,
}

impl<'a> StreamingDecoder<'a> {
    /// Create a fresh decoder tied to `tokenizer`.
    pub fn new(tokenizer: &'a OxiTokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::with_capacity(8),
            total_bytes: 0,
            total_tokens: 0,
        }
    }

    /// Push a single token ID and return the next well-formed UTF-8 slice, if
    /// any.  Returns `None` when the token's bytes do not extend any
    /// previously-pending prefix into a full UTF-8 character.
    ///
    /// The returned `String` contains all characters that became complete as
    /// a result of this push — may be multiple characters if the token
    /// carries several whole code points.
    pub fn push_token(&mut self, id: u32) -> Option<String> {
        self.total_tokens += 1;
        let mut scratch: Vec<u8> = Vec::with_capacity(8);
        self.tokenizer.decode_id_into(id, &mut scratch);
        if scratch.is_empty() {
            return None;
        }
        self.total_bytes += scratch.len();
        self.pending.extend_from_slice(&scratch);
        self.flush_complete()
    }

    /// Push many tokens at once.  Equivalent to repeatedly calling
    /// [`Self::push_token`] but only returns once, with all complete
    /// characters concatenated.
    pub fn push_tokens(&mut self, ids: &[u32]) -> Option<String> {
        let mut out = String::new();
        for &id in ids {
            if let Some(piece) = self.push_token(id) {
                out.push_str(&piece);
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Finish the stream and return the best-effort remaining text.
    ///
    /// This **never fails** (TOK-02): any trailing bytes that still form an
    /// incomplete UTF-8 sequence when the stream ends (e.g. generation was
    /// truncated mid-character) are replaced with `\u{FFFD}`, identically to
    /// [`Self::finish_lossy`] — indeed this delegates to it. The `Result`
    /// return type is kept for source compatibility with existing callers
    /// (`?`/`.expect(...)` continue to work; the `Err` arm is simply never
    /// constructed any more), rather than changing this method's signature.
    pub fn finish(self) -> TokenizerResult<String> {
        Ok(self.finish_lossy())
    }

    /// Finish the stream, replacing any trailing invalid bytes with
    /// `\u{FFFD}`.  Never fails.
    pub fn finish_lossy(mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let bytes = std::mem::take(&mut self.pending);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Finish the stream, requiring it to have ended on a complete UTF-8
    /// boundary (the opposite of [`Self::finish`]'s TOK-02-mandated lenient
    /// default).
    ///
    /// Additive method for a caller with an independent reason to treat
    /// "the stream ended mid-character" as an error — e.g. a test harness
    /// or a protocol that guarantees complete characters and wants to
    /// surface a violation of that guarantee rather than silently
    /// substituting `\u{FFFD}`. By construction (see `Self::flush_complete`,
    /// which drains any *definitely* invalid byte immediately on push,
    /// every push), any bytes still in `pending` when this is called are
    /// always a genuinely incomplete-but-valid-so-far sequence, never a
    /// corrupt one — so the only error this can return is
    /// [`crate::error::TokenizerError::IncompleteUtf8`].
    ///
    /// # Errors
    /// Returns [`crate::error::TokenizerError::IncompleteUtf8`] if the
    /// stream ended with an incomplete trailing UTF-8 sequence still
    /// pending.
    pub fn finish_strict(self) -> TokenizerResult<String> {
        if self.pending.is_empty() {
            Ok(String::new())
        } else {
            Err(crate::error::TokenizerError::IncompleteUtf8)
        }
    }

    /// Number of bytes currently held in the pending buffer.
    ///
    /// A non-zero value after a `push_token` call indicates that the last
    /// token ended mid-UTF-8-sequence.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Reset the decoder state without destroying the `OxiTokenizer`
    /// reference — useful when processing multiple independent streams.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.total_bytes = 0;
        self.total_tokens = 0;
    }

    /// Total bytes processed since construction or last [`Self::reset`].
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Total tokens processed since construction or last [`Self::reset`].
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Pull all complete/recoverable UTF-8 text out of `pending`, leaving
    /// only a genuinely-incomplete trailing sequence (if any, and only up to
    /// [`MAX_UTF8_SEQUENCE_LEN`] bytes of one) behind.
    ///
    /// Three cases, distinguished via [`std::str::Utf8Error::error_len`]
    /// (TOK-02 — the previous implementation only ever consulted
    /// `valid_up_to`, which cannot tell "waiting for more bytes" apart from
    /// "these bytes are simply invalid" and therefore let a single corrupt
    /// lead byte pin `pending` open forever):
    /// 1. A confirmed-valid prefix (`valid_up_to > 0`): emit it and keep
    ///    scanning the remainder — there may be more decodable text, or
    ///    another error, right after it.
    /// 2. A definitely-invalid sequence at the front (`error_len() ==
    ///    Some(n)`): emit `\u{FFFD}`, drop exactly those `n` bytes, and keep
    ///    scanning. This is what makes a flood of invalid bytes drain
    ///    immediately instead of growing `pending` without bound.
    /// 3. A structurally-undecided prefix (`error_len() == None`, i.e. a
    ///    valid lead byte whose continuation bytes have not all arrived
    ///    yet): keep waiting, *unless* `pending` has already grown past
    ///    [`MAX_UTF8_SEQUENCE_LEN`] bytes while still undecided, which is
    ///    impossible for a legal UTF-8 lead byte and therefore means the
    ///    supposed "lead byte" was corrupt; recover the same way as case 2.
    fn flush_complete(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }

        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    if !s.is_empty() {
                        out.push_str(s);
                    }
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid_up_to = e.valid_up_to();
                    if valid_up_to > 0 {
                        // SAFETY-free: `pending[..valid_up_to]` is exactly
                        // the confirmed-valid prefix per `Utf8Error`'s
                        // contract, so this can never fail; still route
                        // through `String::from_utf8` (not `unwrap`) and
                        // simply skip on the unreachable `Err` arm.
                        if let Ok(s) = String::from_utf8(self.pending[..valid_up_to].to_vec()) {
                            out.push_str(&s);
                        }
                        self.pending.drain(..valid_up_to);
                        continue;
                    }

                    match e.error_len() {
                        Some(bad_len) => {
                            // Case 2: definite invalid sequence.
                            out.push('\u{FFFD}');
                            self.pending.drain(..bad_len.max(1));
                            continue;
                        }
                        None if self.pending.len() > MAX_UTF8_SEQUENCE_LEN => {
                            // Case 3 overflow: cannot legitimately still be
                            // "incomplete" past the longest valid UTF-8
                            // sequence length; the lead byte was corrupt.
                            // Drop just it and keep scanning the rest.
                            out.push('\u{FFFD}');
                            self.pending.remove(0);
                            continue;
                        }
                        None => break, // Case 3: genuinely incomplete; wait.
                    }
                }
            }
        }

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::{
        bpe::{byte_fallback_id, BpeMerges},
        tokenizer::TokenizerConfig,
        vocab::Vocabulary,
        OxiTokenizer,
    };

    /// A tokenizer whose vocabulary is exactly the 256 `<0xHH>` byte-fallback
    /// tokens, ids `0..256`. Pushing token id `b` decodes to the single raw
    /// byte `b` — this lets tests drive the streaming decoder with
    /// *arbitrary* byte sequences (including structurally-invalid UTF-8
    /// lead/continuation bytes), independent of any real BPE vocabulary.
    fn byte_stream_tokenizer() -> OxiTokenizer {
        let mut vocab = Vocabulary::new();
        for b in 0u16..=255 {
            vocab.insert(&byte_fallback_id(b as u8), u32::from(b));
        }
        OxiTokenizer::new(vocab, BpeMerges::new(), TokenizerConfig::default())
    }

    // ── TOK-02 regressions ───────────────────────────────────────────────

    #[test]
    fn single_invalid_byte_recovers_immediately_not_forever() {
        // Reproduces the exact finding: 1 invalid lead byte (0x80, a lone
        // continuation byte — invalid at a lead position) followed by 9
        // ASCII bytes. Before the fix: pending grew to 9, emitted text was
        // "", and `finish()` returned `Err`. After the fix: the invalid
        // byte is replaced with U+FFFD immediately and the ASCII bytes flow
        // through normally.
        let tok = byte_stream_tokenizer();
        let mut dec = tok.streaming_decoder();
        let mut out = String::new();
        if let Some(piece) = dec.push_token(0x80) {
            out.push_str(&piece);
        }
        assert_eq!(
            dec.pending_len(),
            0,
            "a single invalid lead byte must be recovered immediately, not \
             left pending"
        );
        for &b in b"Hi there!" {
            if let Some(piece) = dec.push_token(u32::from(b)) {
                out.push_str(&piece);
            }
        }
        out.push_str(&dec.finish().expect("finish never fails"));
        assert_eq!(out, "\u{FFFD}Hi there!");
    }

    #[test]
    fn flood_of_invalid_bytes_never_grows_pending_unbounded() {
        // Before the fix this drove `pending` to 10_000. After the fix each
        // invalid byte is drained the moment it is pushed.
        let tok = byte_stream_tokenizer();
        let mut dec = tok.streaming_decoder();
        for _ in 0..10_000 {
            dec.push_token(0xFF); // 0xFF is never a valid UTF-8 byte at all.
            assert!(
                dec.pending_len() <= 4,
                "pending must stay bounded, got {}",
                dec.pending_len()
            );
        }
        // Never panics/errors, and produces a best-effort string.
        let tail = dec.finish().expect("finish never fails");
        assert!(tail.is_empty() || tail.chars().all(|c| c == '\u{FFFD}'));
    }

    #[test]
    fn finish_strict_errors_on_genuinely_incomplete_tail() {
        // The opt-in strict counterpart to `finish()`'s TOK-02-mandated
        // lenient default -- preserves the pre-fix error behaviour for a
        // caller that explicitly wants it.
        let tok = byte_stream_tokenizer();
        let mut dec = tok.streaming_decoder();
        dec.push_token(0xE6); // lead byte of a 3-byte sequence, incomplete.
        let err = match dec.finish_strict() {
            Err(e) => e,
            Ok(s) => panic!("expected IncompleteUtf8, got Ok({s:?})"),
        };
        assert!(matches!(err, crate::error::TokenizerError::IncompleteUtf8));
    }

    #[test]
    fn finish_strict_succeeds_on_complete_stream() {
        let tok = byte_stream_tokenizer();
        let mut dec = tok.streaming_decoder();
        for &b in b"ok" {
            dec.push_token(u32::from(b));
        }
        assert_eq!(dec.finish_strict().expect("finish_strict ok"), "");
    }

    #[test]
    fn finish_never_errors_on_genuinely_incomplete_tail() {
        // A CJK character's lead byte (0xE6, the first of a 3-byte
        // sequence) with no continuation bytes ever arriving — a legitimate
        // "generation truncated mid-character" scenario, NOT corruption.
        // `finish()` must still return `Ok`, with the incomplete tail
        // rendered as U+FFFD (same as `finish_lossy`).
        let tok = byte_stream_tokenizer();
        let mut dec = tok.streaming_decoder();
        let piece = dec.push_token(0xE6); // lead byte of e.g. U+4E2D ("中").
        assert_eq!(piece, None, "a bare lead byte must not flush yet");
        assert_eq!(dec.pending_len(), 1);
        let tail = dec.finish().expect("finish must never return Err");
        assert_eq!(tail, "\u{FFFD}");
    }

    #[test]
    fn valid_multibyte_sequence_across_pushes_reconstructs_exactly() {
        // The ordinary, non-corrupt case must be unaffected: a 3-byte CJK
        // character delivered one byte per push still reconstructs exactly
        // once complete.
        let tok = byte_stream_tokenizer();
        let mut dec = tok.streaming_decoder();
        let bytes = "中".as_bytes(); // 3-byte UTF-8 sequence.
        assert_eq!(bytes.len(), 3);
        let mut out = String::new();
        for &b in bytes {
            if let Some(piece) = dec.push_token(u32::from(b)) {
                out.push_str(&piece);
            }
        }
        out.push_str(&dec.finish().expect("finish ok"));
        assert_eq!(out, "中");
    }

    #[test]
    fn pending_never_exceeds_four_bytes_across_a_mixed_stream() {
        // A stream mixing valid multi-byte prefixes with invalid bytes must
        // never let `pending` exceed the longest possible UTF-8 sequence
        // length, regardless of the exact byte pattern.
        let tok = byte_stream_tokenizer();
        let mut dec = tok.streaming_decoder();
        let stream: &[u8] = &[
            0xE6, 0x80, // 2 of 3 bytes of a CJK lead sequence (incomplete)
            0xFF, // definitely invalid — must trigger recovery
            0xF0, 0x9F, 0x98, 0x80, // a full 4-byte emoji sequence
            0x80, 0x80, 0x80, 0x80, 0x80, // five lone continuation bytes
        ];
        for &b in stream {
            dec.push_token(u32::from(b));
            assert!(dec.pending_len() <= 4, "pending exceeded 4 bytes");
        }
        let _ = dec.finish().expect("finish never fails");
    }

    // ── TOK-02 property tests ────────────────────────────────────────────

    #[cfg(test)]
    mod proptests {
        use super::byte_stream_tokenizer;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// For ANY byte sequence (valid UTF-8 or not), pushing it one
            /// byte at a time must never panic, must keep `pending` bounded
            /// to at most 4 bytes after every push, and `finish()` must
            /// never return `Err`.
            #[test]
            fn arbitrary_bytes_never_panic_and_stay_bounded(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
                let tok = byte_stream_tokenizer();
                let mut dec = tok.streaming_decoder();
                let mut out = String::new();
                for &b in &bytes {
                    if let Some(piece) = dec.push_token(u32::from(b)) {
                        out.push_str(&piece);
                    }
                    prop_assert!(dec.pending_len() <= 4);
                }
                let tail = dec.finish();
                prop_assert!(tail.is_ok(), "finish() must never fail");
                out.push_str(&tail.unwrap_or_default());
                // The result is, by construction, always a valid `String` —
                // the real assertion is simply that we got here without
                // panicking on any byte pattern proptest can generate.
                let _ = out;
            }

            /// For any *valid* UTF-8 string, feeding its bytes through the
            /// streaming decoder one byte at a time (i.e. worst-case byte
            /// fragmentation) must reconstruct it exactly — there is no
            /// corruption to recover from, only genuine incompleteness that
            /// resolves once enough bytes have arrived.
            #[test]
            fn valid_utf8_byte_by_byte_reconstructs_exactly(s in "\\PC{0,128}") {
                let tok = byte_stream_tokenizer();
                let mut dec = tok.streaming_decoder();
                let mut out = String::new();
                for &b in s.as_bytes() {
                    if let Some(piece) = dec.push_token(u32::from(b)) {
                        out.push_str(&piece);
                    }
                }
                out.push_str(&dec.finish().expect("finish never fails"));
                prop_assert_eq!(out, s);
            }
        }
    }

    #[test]
    fn ascii_passthrough() {
        let tok = OxiTokenizer::char_level_stub(256);
        let ids = tok.encode("abc").expect("encode");
        let mut dec = tok.streaming_decoder();
        let mut out = String::new();
        for id in &ids {
            if let Some(piece) = dec.push_token(*id) {
                out.push_str(&piece);
            }
        }
        out.push_str(&dec.finish().expect("finish ok"));
        assert_eq!(out, "abc");
    }

    #[test]
    fn reset_clears_state() {
        let tok = OxiTokenizer::char_level_stub(256);
        let mut dec = tok.streaming_decoder();
        let ids = tok.encode("abc").expect("encode");
        for id in &ids {
            dec.push_token(*id);
        }
        dec.reset();
        assert_eq!(dec.pending_len(), 0);
        assert_eq!(dec.total_bytes(), 0);
        assert_eq!(dec.total_tokens(), 0);
    }

    #[test]
    fn push_tokens_batch() {
        let tok = OxiTokenizer::char_level_stub(256);
        let mut dec = tok.streaming_decoder();
        let ids = tok.encode("hello").expect("encode");
        let out = dec.push_tokens(&ids).unwrap_or_default();
        // Non-empty because char-level stub emits one char per token.
        assert!(!out.is_empty());
    }

    #[test]
    fn finish_on_empty_is_ok() {
        let tok = OxiTokenizer::char_level_stub(256);
        let dec = tok.streaming_decoder();
        let out = dec.finish().expect("empty finish ok");
        assert_eq!(out, "");
    }

    #[test]
    fn finish_lossy_never_fails() {
        let tok = OxiTokenizer::char_level_stub(256);
        let dec = tok.streaming_decoder();
        let out = dec.finish_lossy();
        assert_eq!(out, "");
    }

    #[test]
    fn counters_advance() {
        let tok = OxiTokenizer::char_level_stub(256);
        let mut dec = tok.streaming_decoder();
        let ids = tok.encode("ab").expect("encode");
        for id in &ids {
            dec.push_token(*id);
        }
        assert!(dec.total_tokens() >= ids.len());
        assert!(dec.total_bytes() > 0);
    }
}
