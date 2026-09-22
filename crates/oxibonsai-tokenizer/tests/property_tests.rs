//! Property-based tests for the OxiBonsai tokenizer.
//!
//! Uses [`proptest`] to generate random inputs and verify invariants that
//! must hold for every valid UTF-8 input — not just the hand-crafted ones.

use oxibonsai_tokenizer::{
    bpe_encode, byte_fallback_id, bytes_to_unicode_map, pretokenize, BpeMerges, OxiTokenizer,
    TokenizerConfig, Vocabulary,
};
use proptest::prelude::*;

// ── Fixture: full 256-byte-fallback tokenizer ────────────────────────────────

fn byte_fallback_tokenizer() -> OxiTokenizer {
    let mut vocab = Vocabulary::new();
    vocab.add_special("<unk>", 0);
    vocab.add_special("<bos>", 1);
    vocab.add_special("<eos>", 2);
    vocab.add_special("<pad>", 3);
    for byte in 0u16..=255u16 {
        let b = byte as u8;
        vocab.insert(&byte_fallback_id(b), 4 + b as u32);
    }
    OxiTokenizer::new(vocab, BpeMerges::new(), TokenizerConfig::default())
}

/// Encode a `&str` as a raw byte sequence (every byte maps to its
/// `<0xHH>` fallback token).
fn encode_bytes_direct(s: &str) -> Vec<u32> {
    s.bytes().map(|b| 4 + b as u32).collect()
}

// ── Fixture: byte-level BPE tokenizer with REAL merge rules ─────────────────
//
// T-10: `decode_encode_roundtrip_utf8` above is synthetic-only in a second
// sense beyond not touching a real `models/tokenizer.json` — its
// `BpeMerges::new()` is empty, so it never runs a single merge round; every
// input takes the trivial per-byte-fallback path. This fixture is the same
// full 0..=255 ByteLevel vocabulary (`bytes_to_unicode_map`, matching
// `crates/oxibonsai-tokenizer/tests/native_bpe_fidelity_tests.rs`'s own
// production-shaped fixture) but with real, chained merge rules over
// printable ASCII (`bytes_to_unicode_map`'s GPT-2 scheme keeps printable
// ASCII bytes — including 'a'..'d' — mapped to themselves, so plain ASCII
// merge rules are valid byte-level merge rules), so arbitrary UTF-8 that
// happens to contain "ab"/"cd"/"abcd" substrings actually exercises
// `bpe_merge_symbols`'s merge loop before the round-trip is checked.
fn byte_level_bpe_tokenizer_with_merges() -> OxiTokenizer {
    let table = bytes_to_unicode_map();
    let mut vocab = Vocabulary::new();
    for b in 0u16..=255 {
        vocab.insert(&table[b as usize].to_string(), u32::from(b));
    }
    // Chained two levels deep, mirroring
    // `native_bpe_fidelity_tests.rs::qwen3_fixture_json`'s
    // `["a b", "c d", "ab cd"]` merge table.
    let mut merges = BpeMerges::new();
    merges.add_merge("a", "b", 300);
    merges.add_merge("c", "d", 301);
    merges.add_merge("ab", "cd", 302);
    vocab.insert("ab", 300);
    vocab.insert("cd", 301);
    vocab.insert("abcd", 302);

    let mut config = TokenizerConfig::default();
    config.byte_level_decode = true;
    OxiTokenizer::new(vocab, merges, config)
}

// ── Properties ───────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Property: `decode(encode(s)) == s` for arbitrary UTF-8 input ≤ 256 bytes,
    /// via the byte-fallback path.
    #[test]
    fn decode_encode_roundtrip_utf8(s in "\\PC{0,64}") {
        let tok = byte_fallback_tokenizer();
        // Skip inputs whose UTF-8 byte length exceeds 256.
        prop_assume!(s.len() <= 256);

        let ids = encode_bytes_direct(&s);
        let decoded = tok.decode(&ids).expect("decode should succeed");
        prop_assert_eq!(decoded, s);
    }

    /// T-10: the same round-trip property, through `OxiTokenizer::encode`
    /// itself (not a hand-built byte-fallback id list) on a tokenizer with
    /// real merge rules — see [`byte_level_bpe_tokenizer_with_merges`]. A
    /// byte-level BPE tokenizer's round-trip guarantee must hold regardless
    /// of *how many* merge rounds ran, since `decode` only ever reassembles
    /// the underlying bytes of whichever tokens `encode` chose.
    #[test]
    fn decode_encode_roundtrip_utf8_with_real_bpe_merges(s in "\\PC{0,64}") {
        prop_assume!(s.len() <= 256);
        let tok = byte_level_bpe_tokenizer_with_merges();
        let ids = tok.encode(&s).expect("encode should succeed");
        let decoded = tok.decode(&ids).expect("decode should succeed");
        prop_assert_eq!(decoded, s);
    }

    /// Property: The char-level tokenizer's `encode` is deterministic.
    #[test]
    fn char_level_encode_deterministic(s in "\\PC{0,32}") {
        let tok = OxiTokenizer::char_level_stub(256);
        let a = tok.encode(&s).expect("encode a");
        let b = tok.encode(&s).expect("encode b");
        prop_assert_eq!(a, b);
    }

    /// Property: BPE on the byte-fallback tokenizer is deterministic.
    #[test]
    fn byte_fallback_encode_deterministic(s in "\\PC{0,64}") {
        let tok = byte_fallback_tokenizer();
        let a = tok.encode(&s).expect("encode a");
        let b = tok.encode(&s).expect("encode b");
        prop_assert_eq!(a, b);
    }

    /// Property: `byte_fallback_id(b)` always has the format `<0xHH>`.
    #[test]
    fn byte_fallback_format(b in any::<u8>()) {
        let token = byte_fallback_id(b);
        prop_assert!(token.starts_with("<0x"));
        prop_assert!(token.ends_with('>'));
        prop_assert_eq!(token.len(), 6);
    }

    /// Property: Every byte maps to a distinct fallback token name.
    #[test]
    fn byte_fallback_injection(b1 in any::<u8>(), b2 in any::<u8>()) {
        let t1 = byte_fallback_id(b1);
        let t2 = byte_fallback_id(b2);
        if b1 == b2 {
            prop_assert_eq!(t1, t2);
        } else {
            prop_assert_ne!(t1, t2);
        }
    }

    /// Property: `pretokenize` never produces an empty string inside its output.
    #[test]
    fn pretokenize_no_empty_chunks(s in ".{0,64}") {
        let tokens = pretokenize(&s);
        for t in &tokens {
            prop_assert!(!t.is_empty(), "pretokenize emitted empty chunk");
        }
    }

    /// Property: `pretokenize` is idempotent when re-joined on the empty string
    /// (preserves all non-whitespace bytes at a minimum).
    #[test]
    fn pretokenize_preserves_non_space_chars(s in "[a-zA-Z0-9]{0,32}") {
        let tokens = pretokenize(&s);
        // Join without separators — for purely alphanumeric input there are
        // no whitespace or punctuation splits, so the joined output must
        // match exactly (minus a possible Ġ prefix — but for ASCII-only
        // inputs with no leading space there is none).
        let joined: String = tokens.join("");
        prop_assert_eq!(joined, s);
    }

    /// Property: When the vocab has all byte-fallbacks, `encode` never
    /// produces UNK (id 0) for pure byte input.
    #[test]
    fn no_unk_with_full_byte_fallback(s in "[\\x20-\\x7E]{0,32}") {
        let tok = byte_fallback_tokenizer();
        let ids = tok.encode(&s).expect("encode");
        for id in ids {
            prop_assert_ne!(id, 0, "UNK produced for ASCII input {:?}", s);
        }
    }

    /// Property: `decode` of an empty id list is the empty string.
    #[test]
    fn decode_empty_is_empty(_dummy in any::<u8>()) {
        let tok = OxiTokenizer::char_level_stub(256);
        let s = tok.decode(&[]).expect("decode empty");
        prop_assert_eq!(s, "");
    }

    /// Property: IDs returned by the char-level tokenizer are all < vocab_size.
    #[test]
    fn char_level_ids_in_range(s in "[a-zA-Z0-9 ]{0,32}") {
        let tok = OxiTokenizer::char_level_stub(256);
        let vs = tok.vocab_size() as u32;
        let ids = tok.encode(&s).expect("encode");
        for id in ids {
            prop_assert!(id < vs, "id {id} out of range (vocab_size {vs})");
        }
    }

    /// Property: Running `bpe_encode` on a word without any merges returns
    /// one id per character when all chars are in the vocab.
    #[test]
    fn bpe_encode_no_merges_is_char_split(word in "[a-z]{1,16}") {
        let mut vocab = Vocabulary::new();
        for (i, ch) in word.chars().enumerate() {
            vocab.insert(&ch.to_string(), i as u32 + 100);
        }
        let merges = BpeMerges::new();
        let ids = bpe_encode(&word, &vocab, &merges);
        prop_assert_eq!(ids.len(), word.chars().count());
    }

    /// Property: `char_level_stub(n)` never exceeds `n` tokens in size.
    #[test]
    fn char_level_vocab_size_bounded(n in 4usize..512) {
        let tok = OxiTokenizer::char_level_stub(n);
        prop_assert!(tok.vocab_size() <= n);
        prop_assert!(tok.vocab_size() >= 4);
    }

    /// Property: Vocabulary IDs inserted via `insert` roundtrip via `get_id`
    /// and `get_token`.
    #[test]
    fn vocab_insert_roundtrip(tok in "[a-zA-Z]{1,8}", id in 0u32..10_000u32) {
        let mut vocab = Vocabulary::new();
        vocab.insert(&tok, id);
        prop_assert_eq!(vocab.get_id(&tok), Some(id));
        prop_assert_eq!(vocab.get_token(id).map(String::from), Some(tok));
    }

    /// Property: The BPE merge-result lookup returns exactly what was inserted.
    #[test]
    fn bpe_merge_result_matches(
        a in "[a-z]{1,4}",
        b in "[a-z]{1,4}",
        id in 0u32..10_000u32
    ) {
        let mut merges = BpeMerges::new();
        merges.add_merge(&a, &b, id);
        prop_assert_eq!(merges.get_merge_result(&a, &b), Some(id));
        prop_assert!(merges.get_merge_priority(&a, &b).is_some());
    }

    /// Property: `decode(encode(s))` never panics for arbitrary strings
    /// containing printable ASCII (char-level tokenizer).
    #[test]
    fn char_level_no_panic(s in "[\\x20-\\x7E]{0,48}") {
        let tok = OxiTokenizer::char_level_stub(256);
        let ids = tok.encode(&s).expect("encode");
        let _ = tok.decode(&ids).expect("decode");
    }
}
