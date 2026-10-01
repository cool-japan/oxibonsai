//! Unit tests for [`crate::tokenizer_bridge`].
//!
//! Attached to `tokenizer_bridge.rs` as its `#[cfg(test)] mod tests`, so they
//! keep full access to the module's private items; kept in their own file to
//! hold `tokenizer_bridge.rs` under the workspace's 2000-line ceiling.

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

/// `FIXTURE_TOKENIZER`'s default resolves relative to this crate's own
/// directory (`cargo test`'s cwd), which this worktree's `models/` does
/// not populate (`.gitkeep` only) — `OXI_TOKENIZER`, when set, overrides
/// it with an absolute path instead, the same override
/// `server::tests::gpu_argmax_routing` and
/// `completions::stream::tests::real_model_router` already establish
/// for the real GGUF itself. Additive: every existing call site that
/// never sets the variable keeps resolving `FIXTURE_TOKENIZER` exactly
/// as before.
fn fixture_tokenizer_path() -> String {
    std::env::var("OXI_TOKENIZER").unwrap_or_else(|_| FIXTURE_TOKENIZER.to_string())
}

fn maybe_load_fixture() -> Option<TokenizerBridge> {
    let path = fixture_tokenizer_path();
    if !Path::new(&path).exists() {
        eprintln!(
            "skipped: tokenizer fixture not found at {path} \
             (run scripts/download_tokenizer.sh, or set OXI_TOKENIZER, to enable)",
        );
        return None;
    }
    match TokenizerBridge::from_file(&path) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("skipped: failed to load tokenizer fixture: {e}");
            None
        }
    }
}

fn maybe_load_native_fixture() -> Option<TokenizerBridge> {
    let path = fixture_tokenizer_path();
    if !Path::new(&path).exists() {
        eprintln!("skipped: tokenizer fixture not found at {path}");
        return None;
    }
    match TokenizerBridge::native_from_file(&path) {
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
    sequence_without_byte_level.with_decoder(Some(DecoderWrapper::Sequence(Sequence::new(vec![
        DecoderWrapper::WordPiece(WordPiece::default()),
    ]))));
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
             become the default backend"
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

// ── Chat template + think/tool-call ids carried on the bridge ────────

/// A GGUF metadata block carrying a full `tokenizer.ggml.*` vocabulary
/// (so [`OxiTokenizer::from_gguf_metadata`] succeeds) that ALSO defines
/// `<think>`, `</think>`, `<tool_call>` and `</tool_call>` as
/// `USER_DEFINED` added tokens, plus `tokenizer.chat_template` — the
/// same byte-identical wire-format construction
/// `gguf_vocab.rs::tests::make_metadata` uses (that helper is private to
/// its own module, so this crate builds its own minimal one).
fn gguf_metadata_with_think_and_tools(chat_template: &str) -> oxibonsai_core::MetadataStore {
    use oxibonsai_core::gguf::types::GgufValueType;

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
    fn str_bytes(s: &str) -> Vec<u8> {
        let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(s.as_bytes());
        bytes
    }

    let tokens = [
        "a",
        "b",
        "c",
        "<|endoftext|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
    ];
    // NORMAL x3, then <|endoftext|> is CONTROL, the four reasoning/tool
    // markers are USER_DEFINED (matches the real Bonsai 2 histogram's
    // token_type split documented in `gguf_vocab.rs`).
    let token_types = [1i32, 1, 1, 3, 4, 4, 4, 4];

    let mut data = Vec::new();
    let mut count = 0u64;
    macro_rules! push {
        ($bytes:expr) => {{
            data.extend_from_slice(&$bytes);
            count += 1;
        }};
    }
    push!(kv_string("tokenizer.ggml.model", "gpt2"));
    push!(kv_string_array("tokenizer.ggml.tokens", &tokens));
    push!(kv_i32_array("tokenizer.ggml.token_type", &token_types));
    push!(kv_u32("tokenizer.ggml.eos_token_id", 3)); // <|endoftext|>
    push!(kv_string("tokenizer.chat_template", chat_template));

    let (store, _) =
        oxibonsai_core::MetadataStore::parse(&data, 0, count).expect("well-formed metadata");
    store
}

/// A minimal, definitely-compiling real-Jinja template (the subset
/// `chat_templates.rs`'s own fallback templates already demonstrate:
/// `{% for %}`, `{{ dotted.field }}`).
const TINY_VALID_JINJA_TEMPLATE: &str =
    "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}";

#[test]
fn native_from_gguf_metadata_builds_a_working_tokenizer_and_attaches_the_template() {
    let md = gguf_metadata_with_think_and_tools(TINY_VALID_JINJA_TEMPLATE);
    let bridge =
        TokenizerBridge::native_from_gguf_metadata(&md).expect("well-formed GGUF metadata");
    assert_eq!(bridge.backend(), TokenizerBackendKind::Native);
    // The tokenizer itself works (round-trips ordinary vocabulary).
    let ids = bridge.encode("ab").expect("encode");
    assert!(!ids.is_empty());

    // The GGUF's own template is attached and actually rendered
    // through, not the built-in fallback.
    let rendered = bridge
        .resolved_chat_template()
        .render_with(
            &[oxibonsai_tokenizer::chat_templates::RenderMessage::new(
                "user", "hi",
            )],
            &oxibonsai_tokenizer::chat_templates::RenderOptions::default(),
        )
        .expect("render");
    assert_eq!(rendered, "user:hi;");
}

#[test]
fn native_from_gguf_metadata_resolves_think_and_tool_call_ids() {
    let md = gguf_metadata_with_think_and_tools(TINY_VALID_JINJA_TEMPLATE);
    let bridge = TokenizerBridge::native_from_gguf_metadata(&md).expect("well-formed metadata");
    assert_eq!(bridge.think_open_id(), Some(4));
    assert_eq!(bridge.think_close_id(), Some(5));
    assert_eq!(bridge.tool_call_open_id(), Some(6));
    assert_eq!(bridge.tool_call_close_id(), Some(7));
}

#[test]
fn native_from_gguf_metadata_errors_on_an_uncompilable_shipped_template() {
    // B5: a shipped-but-uncompilable template must error, not silently
    // substitute the fallback.
    let md = gguf_metadata_with_think_and_tools("{% this is not valid jinja %}");
    assert!(TokenizerBridge::native_from_gguf_metadata(&md).is_err());
}

#[test]
fn resolved_chat_template_falls_back_when_none_was_ever_attached() {
    // A bridge built through any constructor OTHER than
    // `native_from_gguf_metadata` / `with_chat_template` has no
    // GgufMetadata-derived template -- `resolved_chat_template` must
    // still return something usable (the named ChatML/Qwen3 fallback),
    // never a `None` the caller has to special-case.
    let bridge = tiny_native_bridge();
    let out = bridge
        .resolved_chat_template()
        .render_with(
            &[oxibonsai_tokenizer::chat_templates::RenderMessage::new(
                "user", "hi",
            )],
            &oxibonsai_tokenizer::chat_templates::RenderOptions {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .expect("fallback must render");
    assert_eq!(
        out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        "must be the named Qwen3/ChatML fallback"
    );
}

#[test]
fn a_vocabulary_with_no_think_tokens_resolves_no_think_ids() {
    // RT-10 correction: a vocabulary that defines no `<think>`/`</think>`
    // at all resolves no think ids -- `tiny_native_bridge`'s fixture has
    // no reasoning markers. (The shipped Qwen3 1.7B/8B `tokenizer.json`
    // is NOT such a vocabulary: it defines `<think>` = 151667 and
    // `</think>` = 151668 as non-special added tokens.)
    let bridge = tiny_native_bridge();
    assert_eq!(bridge.think_open_id(), None);
    assert_eq!(bridge.think_close_id(), None);
}

#[test]
fn with_chat_template_attaches_without_disturbing_already_resolved_ids() {
    let md = gguf_metadata_with_think_and_tools(TINY_VALID_JINJA_TEMPLATE);
    // Build the tokenizer half without the metadata-driven template...
    let tok = oxibonsai_tokenizer::OxiTokenizer::from_gguf_metadata(&md).expect("tokenizer build");
    let bridge = TokenizerBridge::from_native_tokenizer(tok);
    assert_eq!(
        bridge.think_open_id(),
        Some(4),
        "ids resolve from the vocabulary regardless of how the template was attached"
    );
    // ...then attach a template explicitly, mirroring a caller that
    // resolves `ResolvedChatTemplate::from_gguf` itself.
    let template = oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::from_gguf(&md)
        .expect("compiles");
    let bridge = bridge.with_chat_template(template);
    assert_eq!(
        bridge.think_open_id(),
        Some(4),
        "unchanged by with_chat_template"
    );
    let rendered = bridge
        .resolved_chat_template()
        .render_with(
            &[oxibonsai_tokenizer::chat_templates::RenderMessage::new(
                "user", "hi",
            )],
            &oxibonsai_tokenizer::chat_templates::RenderOptions::default(),
        )
        .expect("render");
    assert_eq!(rendered, "user:hi;");
}

#[test]
fn debug_impl_reports_the_new_chat_fields() {
    let bridge = tiny_native_bridge();
    let dbg = format!("{bridge:?}");
    assert!(dbg.contains("has_chat_template"));
    assert!(dbg.contains("think_ids"));
    assert!(dbg.contains("tool_call_ids"));
}

// ── a special-flagged close marker must still reach the reasoning
//    splitter ────────────────────────────────────────────────────────
//
// Both chat endpoints decode every generated id through the response
// pipeline (`server/response_pipeline.rs`), which feeds each decoded piece
// to `reasoning::ReasoningSplitter::push`. A token flagged `special` in
// the vocabulary — real chat-model `<|...|>` markers commonly are, and
// `<think>`/`</think>` themselves could be for a given model — decodes to
// `Ok(None)` (no bytes of its own). An earlier revision treated `Ok(None)`
// as a reason to skip the token entirely, so the splitter never learned
// that id occurred at all: a special-flagged `</think>` would never be
// seen, and every token after it would stay misclassified as reasoning
// for the rest of the response. An empty piece flows through instead, so
// the splitter still sees every id. This test pins the two building
// blocks: `step_decode` really does return `Ok(None)` for a
// special-flagged id, and feeding that id (with empty text) to the
// splitter still correctly detects the boundary.
#[test]
fn step_decode_returns_none_for_a_special_flagged_id_yet_the_splitter_still_sees_it() {
    let tok = tiny_native_bridge();
    // `<|im_start|>` (id 12) is flagged `"special": true` in
    // `TINY_TOKENIZER_JSON` — stands in for a model whose `</think>` is
    // also special; only the flag matters here, not the token's own
    // text.
    let close_id = 12u32;
    let mut decode_state = tok.new_decode_stream(true);

    // Sanity: confirm the precondition this whole fix exists for.
    let close_piece = tok
        .step_decode(&mut decode_state, close_id)
        .expect("decode must not error");
    assert_eq!(
        close_piece, None,
        "a special-flagged id must decode to no visible text"
    );

    // Mirrors the fixed handler pattern exactly: every id is pushed to
    // the splitter, using an empty string when `step_decode` returned
    // `None`, regardless of whether that id is the close marker.
    let mut splitter = crate::reasoning::ReasoningSplitter::new(true, Some(close_id));
    assert!(splitter.in_reasoning());

    match splitter.push(close_id, close_piece.as_deref().unwrap_or("")) {
        crate::reasoning::ReasoningChunk::Boundary => {}
        other => {
            panic!("a special-flagged close id must still be seen as the boundary, got {other:?}")
        }
    }
    assert!(
        !splitter.in_reasoning(),
        "the splitter must have left the reasoning phase"
    );

    // And a real, decodable token right after it is ordinary content —
    // proving the splitter did not just get stuck, but genuinely
    // resumed normal classification.
    let mut decode_state = tok.new_decode_stream(true);
    let piece = tok
        .step_decode(&mut decode_state, 7) // "a"
        .expect("decode must not error")
        .unwrap_or_default();
    match splitter.push(7, &piece) {
        crate::reasoning::ReasoningChunk::Content(s) => assert_eq!(s, "a"),
        other => panic!("expected Content(\"a\") after the boundary, got {other:?}"),
    }
}

/// `render_chat_prompt` (whole-prompt Jinja rendering, B1) replaced
/// `server::sanitize::encode_chat_prompt` (the old hardcoded
/// ChatML-segment builder) everywhere — that function has no callers left
/// in the crate at all. A deployment whose model ships no
/// `tokenizer.chat_template` renders through
/// `ResolvedChatTemplate::default_fallback` — the exact template
/// `render_chat_prompt` uses here with no `.with_chat_template` call. For
/// a plain system/user/assistant conversation (no tools, no reasoning —
/// the overwhelmingly common case, and the only shape the old builder
/// ever handled), the two paths must therefore produce byte-identical
/// prompt token ids on the REAL shipped 1.7B/8B tokenizer, or the rewrite
/// silently changed live chat output for those deployments.
#[cfg(feature = "server")]
#[test]
fn fallback_render_chat_prompt_matches_the_old_chatml_builder_on_the_real_tokenizer() {
    let Some(tok) = maybe_load_fixture() else {
        return;
    };
    let guard = crate::server::sanitize::SpecialTokenGuard::from_tokenizer(&tok);
    let messages = vec![
        crate::server::ChatMessage::text("system", "You are a helpful assistant."),
        crate::server::ChatMessage::text("user", "Hello, who are you?"),
        crate::server::ChatMessage::text("assistant", "I am OxiBonsai."),
        crate::server::ChatMessage::text("user", "Nice to meet you."),
    ];

    let old_ids = crate::server::sanitize::encode_chat_prompt(&tok, &messages, &guard, true)
        .expect("old builder must encode");

    let render_messages = crate::tokenizer_bridge::chat_render::to_render_messages(&messages, &[]);
    let opts = oxibonsai_tokenizer::chat_templates::RenderOptions {
        add_generation_prompt: true,
        ..Default::default()
    };
    let (_rendered, new_ids) = crate::tokenizer_bridge::chat_render::render_chat_prompt(
        &tok,
        &guard,
        &render_messages,
        &opts,
        true,
    )
    .expect("new pipeline must render");

    assert_eq!(
        old_ids, new_ids,
        "the new whole-prompt Jinja rendering must reproduce the old ChatML-segment \
         builder's exact token ids for a plain conversation on the real tokenizer"
    );
}

/// Real-data proof on the Bonsai 2 27B vocabulary (`OXI_BONSAI2_PQ2_GGUF`):
/// reads ONLY the header, the metadata KV table and the tensor descriptors
/// (`GgufFile::parse` over an `mmap`, never the multi-GB tensor bytes — the
/// same reader every real load goes through, without the weight load),
/// builds a `TokenizerBridge` from that metadata alone, and checks what the
/// server depends on for this model: the shipped template resolves (an
/// uncompilable one errors loudly — this is the suite's only check that the
/// real 27B template compiles against this engine), the think ids are the
/// design's own 248068/248069 (design App. A.1), and whether
/// `</think>`/`<|im_end|>` are vocabulary-flagged `special` — the
/// precondition `step_decode`'s `Ok(None)` for a special-flagged id (this
/// file's
/// `step_decode_returns_none_for_a_special_flagged_id_yet_the_splitter_still_sees_it`)
/// exists for. Without the variable the test records a skip.
#[test]
fn real_27b_gguf_metadata_resolves_a_real_template_and_the_design_think_ids() {
    const TEST: &str = "oxibonsai-runtime::lib::\
                        real_27b_gguf_metadata_resolves_a_real_template_and_the_design_think_ids";
    use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
    let Some(gguf_path) = std::env::var("OXI_BONSAI2_PQ2_GGUF")
        .ok()
        .filter(|path| !path.is_empty())
    else {
        eprintln!(
            "capability report: {TEST} SKIPPED — set OXI_BONSAI2_PQ2_GGUF to the \
             Ternary-Bonsai-2-27B-PQ2_0.gguf path"
        );
        record_skipped(Capability::Bonsai2Models, TEST);
        return;
    };
    let start = std::time::Instant::now();
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(Path::new(&gguf_path))
        .expect("OXI_BONSAI2_PQ2_GGUF must name a readable GGUF file");
    let file = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)
        .expect("header + metadata + tensor descriptors must parse");
    eprintln!(
        "27B GGUF: {} metadata entries, {} tensors",
        file.metadata.len(),
        file.tensors.len()
    );

    let bridge = TokenizerBridge::native_from_gguf_metadata(&file.metadata)
        .expect("vocab + chat template must both load from the real 27B metadata");

    let template = bridge.resolved_chat_template();
    assert!(
        matches!(
            template,
            oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::Jinja(_)
        ),
        "the shipped model's own tokenizer.chat_template must resolve, not silently fall \
         back, for a real model that does ship one"
    );

    assert_eq!(
        bridge.think_open_id(),
        Some(248068),
        "the design's own <think> id (Appendix A.1) must resolve against the real vocabulary"
    );
    assert_eq!(
        bridge.think_close_id(),
        Some(248069),
        "the design's own </think> id (Appendix A.1) must resolve against the real vocabulary"
    );

    eprintln!(
        "</think> (248069) flagged special: {}",
        bridge.is_special(248069)
    );
    if let Some(im_end_id) = bridge
        .encode("<|im_end|>")
        .ok()
        .and_then(|ids| (ids.len() == 1).then_some(ids[0]))
    {
        eprintln!(
            "<|im_end|> ({im_end_id}) flagged special: {}",
            bridge.is_special(im_end_id)
        );
    }
    record_executed_timed(Capability::Bonsai2Models, TEST, start.elapsed());
}
