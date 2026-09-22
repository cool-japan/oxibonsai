//! Integration tests for [`StreamingDecoder`].
//!
//! Covers:
//! - single-byte ASCII tokens
//! - multi-byte UTF-8 split across byte-fallback tokens (e.g. `あ`)
//! - CJK runs
//! - emoji (4-byte codepoints)
//! - combining marks
//! - no replacement char mid-sequence
//! - `finish()` behaviour (strict vs lossy)

use oxibonsai_tokenizer::{byte_fallback_id, BpeMerges, OxiTokenizer, TokenizerConfig, Vocabulary};

// ── Byte-fallback fixture ────────────────────────────────────────────────────

/// Build a tokenizer whose vocabulary contains all 256 `<0xHH>` byte-fallback
/// tokens (IDs 4..=259) plus the four standard specials at IDs 0..3.
///
/// The encoder will emit one `<0xHH>` token per raw byte, which lets us feed
/// any multi-byte UTF-8 sequence through `StreamingDecoder` and verify
/// byte-exact reassembly.
fn byte_fallback_tokenizer() -> OxiTokenizer {
    let mut vocab = Vocabulary::new();
    vocab.add_special("<unk>", 0);
    vocab.add_special("<bos>", 1);
    vocab.add_special("<eos>", 2);
    vocab.add_special("<pad>", 3);
    for byte in 0u16..=255u16 {
        let b = byte as u8;
        let token = byte_fallback_id(b);
        vocab.insert(&token, 4 + b as u32);
    }
    let config = TokenizerConfig::default();
    let merges = BpeMerges::new();
    OxiTokenizer::new(vocab, merges, config)
}

// ── ASCII path ───────────────────────────────────────────────────────────────

#[test]
fn ascii_single_byte_roundtrip() {
    let tok = OxiTokenizer::char_level_stub(256);
    let ids = tok.encode("abc").expect("encode");
    let mut dec = tok.streaming_decoder();
    let mut out = String::new();
    for id in &ids {
        if let Some(piece) = dec.push_token(*id) {
            out.push_str(&piece);
        }
    }
    out.push_str(&dec.finish().expect("finish"));
    assert_eq!(out, "abc");
}

#[test]
fn ascii_push_tokens_batch() {
    let tok = OxiTokenizer::char_level_stub(256);
    let ids = tok.encode("hello").expect("encode");
    let mut dec = tok.streaming_decoder();
    let piece = dec.push_tokens(&ids).unwrap_or_default();
    let rest = dec.finish().expect("finish");
    assert_eq!(format!("{piece}{rest}"), "hello");
}

// ── Multi-byte UTF-8 spanning tokens ─────────────────────────────────────────

#[test]
fn utf8_hiragana_split_across_byte_tokens() {
    // 'あ' = U+3042 encoded as E3 81 82 (3 bytes).
    let tok = byte_fallback_tokenizer();
    let ids: Vec<u32> = "あ".bytes().map(|b| 4 + b as u32).collect();
    assert_eq!(ids.len(), 3);

    let mut dec = tok.streaming_decoder();
    let first = dec.push_token(ids[0]);
    assert!(first.is_none(), "1/3 bytes should not yield a char");
    let second = dec.push_token(ids[1]);
    assert!(second.is_none(), "2/3 bytes should not yield a char");
    let third = dec.push_token(ids[2]);
    assert_eq!(third.as_deref(), Some("あ"));
}

#[test]
fn utf8_cjk_run_five_chars() {
    // A short Japanese phrase: "こんにちは"
    let phrase = "こんにちは";
    let tok = byte_fallback_tokenizer();
    let ids: Vec<u32> = phrase.bytes().map(|b| 4 + b as u32).collect();
    let mut dec = tok.streaming_decoder();
    let mut out = String::new();
    for id in ids {
        if let Some(piece) = dec.push_token(id) {
            out.push_str(&piece);
        }
    }
    out.push_str(&dec.finish().expect("finish"));
    assert_eq!(out, phrase);
}

#[test]
fn utf8_emoji_4byte() {
    // '👋' = U+1F44B encoded as F0 9F 91 8B (4 bytes).
    let tok = byte_fallback_tokenizer();
    let ids: Vec<u32> = "👋".bytes().map(|b| 4 + b as u32).collect();
    assert_eq!(ids.len(), 4);

    let mut dec = tok.streaming_decoder();
    assert!(dec.push_token(ids[0]).is_none());
    assert!(dec.push_token(ids[1]).is_none());
    assert!(dec.push_token(ids[2]).is_none());
    assert_eq!(dec.push_token(ids[3]).as_deref(), Some("👋"));
}

#[test]
fn utf8_combining_marks_preserved() {
    // 'e' + U+0301 (combining acute) = "é" decomposed form.
    let s = "e\u{0301}";
    let tok = byte_fallback_tokenizer();
    let ids: Vec<u32> = s.bytes().map(|b| 4 + b as u32).collect();

    let mut dec = tok.streaming_decoder();
    let mut out = String::new();
    for id in ids {
        if let Some(piece) = dec.push_token(id) {
            out.push_str(&piece);
        }
    }
    out.push_str(&dec.finish().expect("finish"));
    assert_eq!(out, s);
    // Must still be decomposed (2 code points, not 1 precomposed 'é').
    assert_eq!(out.chars().count(), 2);
}

#[test]
fn no_replacement_char_while_pending() {
    // Feed the first byte of 'あ' only.  The output so far must not contain
    // U+FFFD — the decoder must HOLD rather than emit garbage.
    let tok = byte_fallback_tokenizer();
    let bytes = "あ".as_bytes();
    let mut dec = tok.streaming_decoder();
    let first = dec.push_token(4 + bytes[0] as u32);
    assert!(first.is_none());
    assert_eq!(dec.pending_len(), 1);
}

#[test]
fn pending_len_tracks_incomplete() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    let bytes = "あ".as_bytes();
    dec.push_token(4 + bytes[0] as u32);
    assert_eq!(dec.pending_len(), 1);
    dec.push_token(4 + bytes[1] as u32);
    assert_eq!(dec.pending_len(), 2);
    dec.push_token(4 + bytes[2] as u32);
    assert_eq!(dec.pending_len(), 0);
}

// ── finish() semantics ───────────────────────────────────────────────────────

#[test]
fn finish_strict_on_complete_stream_succeeds() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    for b in "ok".bytes() {
        dec.push_token(4 + b as u32);
    }
    let rest = dec.finish().expect("finish");
    // Residue should be empty — flush_complete consumes any ASCII eagerly.
    assert_eq!(rest, "");
}

#[test]
fn finish_strict_on_incomplete_stream_errors() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    let bytes = "あ".as_bytes();
    dec.push_token(4 + bytes[0] as u32);
    dec.push_token(4 + bytes[1] as u32);
    // Third byte missing — the strict counterpart must error.
    //
    // Recorded reason (ACCEPTANCE: "existing tokenizer tests unchanged OR
    // UPDATED WITH A RECORDED REASON"): TOK-02 made `finish()` itself
    // infallible (it now always returns a best-effort `Ok` string, per
    // `crate::streaming::StreamingDecoder::finish`'s docs), so the old
    // `dec.finish().expect_err(...)` call here now panics. This test's own
    // name (`finish_strict_on_incomplete_stream_errors`) already names the
    // right method — swap in `finish_strict()`, the additive opt-in
    // counterpart that preserves the pre-TOK-02 error behaviour.
    let err = dec.finish_strict().expect_err("must fail");
    use oxibonsai_tokenizer::TokenizerError;
    assert!(matches!(err, TokenizerError::IncompleteUtf8));
}

#[test]
fn finish_lossy_on_incomplete_stream_returns_replacement() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    let bytes = "あ".as_bytes();
    dec.push_token(4 + bytes[0] as u32);
    let lossy = dec.finish_lossy();
    // Must contain U+FFFD (replacement character).
    assert!(lossy.chars().any(|c| c == '\u{FFFD}'));
}

#[test]
fn finish_on_empty_stream_ok() {
    let tok = OxiTokenizer::char_level_stub(256);
    let dec = tok.streaming_decoder();
    assert_eq!(dec.finish().expect("finish"), "");
}

// ── Counters / reset ─────────────────────────────────────────────────────────

#[test]
fn total_counters_advance() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    for b in "hello".bytes() {
        dec.push_token(4 + b as u32);
    }
    assert!(dec.total_tokens() >= 5);
    assert!(dec.total_bytes() >= 5);
}

#[test]
fn reset_clears_counters_and_pending() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    let bytes = "あ".as_bytes();
    dec.push_token(4 + bytes[0] as u32);
    assert!(dec.pending_len() > 0);
    dec.reset();
    assert_eq!(dec.pending_len(), 0);
    assert_eq!(dec.total_bytes(), 0);
    assert_eq!(dec.total_tokens(), 0);
}

#[test]
fn special_ids_produce_no_output() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    // IDs 0..=3 are the BOS/EOS/PAD/UNK specials.
    for id in 0u32..=3u32 {
        let piece = dec.push_token(id);
        assert!(piece.is_none());
    }
    assert_eq!(dec.pending_len(), 0);
}

#[test]
fn unknown_id_emits_replacement_char() {
    let tok = byte_fallback_tokenizer();
    let mut dec = tok.streaming_decoder();
    let piece = dec.push_token(999_999);
    // Replacement char is its own complete UTF-8 sequence, so the decoder
    // should emit it immediately.
    assert_eq!(piece.as_deref(), Some("\u{FFFD}"));
}

#[test]
fn alternating_ascii_and_multibyte() {
    let s = "a あ b";
    let tok = byte_fallback_tokenizer();
    let ids: Vec<u32> = s.bytes().map(|b| 4 + b as u32).collect();
    let mut dec = tok.streaming_decoder();
    let mut out = String::new();
    for id in ids {
        if let Some(piece) = dec.push_token(id) {
            out.push_str(&piece);
        }
    }
    out.push_str(&dec.finish().expect("finish"));
    assert_eq!(out, s);
}

#[test]
fn streaming_decoder_vs_decode_match() {
    let s = "あ";
    let tok = byte_fallback_tokenizer();
    let ids: Vec<u32> = s.bytes().map(|b| 4 + b as u32).collect();

    let batch = tok.decode(&ids).expect("decode");

    let mut dec = tok.streaming_decoder();
    let mut streamed = String::new();
    for id in ids {
        if let Some(piece) = dec.push_token(id) {
            streamed.push_str(&piece);
        }
    }
    streamed.push_str(&dec.finish().expect("finish"));
    assert_eq!(batch, streamed);
}
