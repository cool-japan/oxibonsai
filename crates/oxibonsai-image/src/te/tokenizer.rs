//! Qwen3 ByteLevel-BPE tokenizer for the Bonsai-Image text encoder.
//!
//! Reproduces the HuggingFace `tokenizers` pipeline used by the reference
//! (`tokenizer.json`): **normalize (ASCII-only, see below) → GPT-2
//! pre-tokenization regex → ByteLevel byte→unicode mapping → BPE merges**, then
//! splices the fixed Qwen3 chat-template special-id prefix/suffix and
//! right-pads to `max_len`.
//!
//! Chat template (`enable_thinking=False`) for a prompt `P`:
//!
//! ```text
//! <|im_start|>user\n{P}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
//! ```
//!
//! which, as token ids, is the fixed prefix `[151644, 872, 198]`
//! (`<|im_start|>`, `user`, `\n`), then `BPE(P)`, then the fixed suffix
//! `[151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]`
//! (`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`, `<think>`, `\n\n`,
//! `</think>`, `\n\n`); right-padded to `max_len` with `151643` (`<|endoftext|>`).
//! The attention mask is 1 for the real tokens, 0 for padding.
//!
//! Validated: for `P = "a tiny bonsai tree in a ceramic pot"` the produced ids
//! equal the golden `input_ids` exactly (see the `tokenizer_*` tests / the
//! `te_parity` example's tokenizer check).
//!
//! ## Normalization scope (no real NFC)
//!
//! `normalize_ascii_only` is a **plain identity pass-through**, not Unicode
//! NFC normalization — it does not compose or decompose anything. It is
//! correct only because image prompts are overwhelmingly ASCII/Latin (already
//! in NFC by construction) and the canonical goldens are ASCII. Full Unicode
//! NFC needs composition tables this crate does not implement; a non-ASCII
//! prompt that requires real recomposition will tokenize slightly differently
//! from HF. An earlier revision named this function `nfc_ascii`, which implied
//! it performed some (if scope-limited) real NFC work; it did not, and the
//! rename removes that false impression (RAG-EVAL-IMG-18 / TOK-10 audit).
//! `oxibonsai-tokenizer` does have real NFC support
//! ([`oxibonsai_tokenizer::tokenizer::TokenizerConfig::normalize_nfc`], via
//! `OxiTokenizer::encode`), but this module bypasses the full `OxiTokenizer`
//! pipeline (see below) in favour of calling the lower-level BPE primitives
//! directly, so it does not go through that path either — tracked as a
//! follow-up alongside real NFC support for this text encoder, not addressed
//! by this fix (TOK-10's scope is de-duplication, not new normalization
//! behaviour).
//!
//! ## Byte-level BPE pipeline (TOK-10: no longer a duplicate)
//!
//! This module used to carry its own independent, byte-for-byte duplicate of
//! the GPT-2 byte-level BPE pipeline (`pre_tokenize`, `match_contraction`,
//! `Qwen3Tokenizer::bpe`, `build_byte_to_unicode`, `parse_merge`) instead of
//! depending on `oxibonsai-tokenizer`'s canonical implementation, so this
//! crate's own pre-tokenizer fixes (e.g. TOK-03's `fancy-regex`-based
//! `Split` engine, which corrected divergences from HF on Roman numerals and
//! blank-line whitespace) never reached it. `encode_prompt`/`from_json_str`
//! below now build an [`oxibonsai_tokenizer::Vocabulary`] +
//! [`oxibonsai_tokenizer::BpeMerges`] from the parsed `tokenizer.json` and
//! call [`oxibonsai_tokenizer::bpe::pretokenize_gpt2`] +
//! [`oxibonsai_tokenizer::bpe::bpe_encode_bytelevel`] directly, rather than
//! going through the full [`oxibonsai_tokenizer::OxiTokenizer`] type: this
//! text encoder's chat-template wrapping is a **fixed, pre-captured id
//! array** (`PREFIX`/`SUFFIX` below, validated against the mflux golden),
//! spliced in after encoding the bare prompt body — it has no use for
//! `OxiTokenizer`'s general special-token carve-out, BOS/EOS injection, or
//! decode path. Calling the BPE primitives directly also means
//! `OxiTokenizer`'s protected-token carve-out (`build_special_pieces`,
//! `oxibonsai-tokenizer`'s `tokenizer.rs:607-616`) is never in this code
//! path at all, so it cannot re-tokenize `PREFIX`/`SUFFIX` differently (the
//! wave-1 addendum's mandatory pre-check for this rewrite).

use std::path::Path;

use oxibonsai_tokenizer::bpe::{bpe_encode_bytelevel, pretokenize_gpt2};
use oxibonsai_tokenizer::hf_format::bytes_to_unicode_map;
use oxibonsai_tokenizer::{BpeMerges, HfTokenizerJson, Vocabulary};

use crate::te::error::{TeError, TeResult};

/// `<|im_start|>` id.
const IM_START: u32 = 151644;
/// `<|im_end|>` id.
const IM_END: u32 = 151645;
/// `<|endoftext|>` id (padding).
pub const PAD_ID: u32 = 151643;
/// `user` token id.
const USER: u32 = 872;
/// `assistant` token id.
const ASSISTANT: u32 = 77091;
/// `\n` token id.
const NL: u32 = 198;
/// `\n\n` token id.
const NL2: u32 = 271;
/// `<think>` token id.
const THINK_OPEN: u32 = 151667;
/// `</think>` token id.
const THINK_CLOSE: u32 = 151668;

/// The fixed chat-template prefix: `<|im_start|>user\n`.
const PREFIX: [u32; 3] = [IM_START, USER, NL];
/// The fixed chat-template suffix:
/// `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
const SUFFIX: [u32; 9] = [
    IM_END,
    NL,
    IM_START,
    ASSISTANT,
    NL,
    THINK_OPEN,
    NL2,
    THINK_CLOSE,
    NL2,
];

/// The tokenized output: padded `input_ids` and the 0/1 `attention_mask`.
#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    /// Padded token ids (`max_len` of them).
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<i32>,
}

/// Qwen3 ByteLevel-BPE tokenizer (loaded from `tokenizer.json`).
///
/// A thin wrapper over `oxibonsai-tokenizer`'s BPE primitives (see the
/// module docs' "Byte-level BPE pipeline" section for why this does not use
/// the full `OxiTokenizer` type).
pub struct Qwen3Tokenizer {
    /// Byte-level token string ↔ id, and the atomically-protected id sets
    /// (unused by this module — `oxibonsai_tokenizer::Vocabulary` always
    /// carries them, but nothing here calls the carve-out APIs that
    /// consult them).
    vocabulary: Vocabulary,
    /// Ordered BPE merge table.
    merges: BpeMerges,
    /// Fallback id for a byte-level piece with no vocabulary entry. Real
    /// byte-level vocabularies always cover all 256 remapped byte
    /// characters, so this should be unreachable in practice; resolved from
    /// the parsed `tokenizer.json`'s `unk_token` when declared, else `0`.
    unk_id: u32,
}

impl Qwen3Tokenizer {
    /// Load from a directory containing `tokenizer.json`.
    ///
    /// # Errors
    /// [`TeError::Io`] / [`TeError::Tokenizer`] on a missing or malformed file.
    pub fn open(dir: &Path) -> TeResult<Self> {
        let path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path).map_err(|e| TeError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        Self::from_json_str(&text)
    }

    /// Parse from the raw `tokenizer.json` contents.
    ///
    /// Delegates all parsing to
    /// [`oxibonsai_tokenizer::hf_format::HfTokenizerJson::parse`] (handling
    /// both merge shapes — `["a","b"]` and `"a b"` — and both `vocab`
    /// object forms) rather than re-implementing a parser, then builds a
    /// plain [`Vocabulary`] + [`BpeMerges`] from its `vocab`/`merges`
    /// fields. This tokenizer has no use for `HfTokenizerJson`'s
    /// added-token/special-token bookkeeping (see the module docs), so it
    /// is intentionally not carried over.
    ///
    /// # Errors
    /// [`TeError::Tokenizer`] if the JSON is malformed or missing the
    /// `model.vocab` / `model.merges` fields, or if a merge's resulting
    /// token is absent from the vocabulary.
    pub fn from_json_str(text: &str) -> TeResult<Self> {
        let parsed = HfTokenizerJson::parse(text).map_err(|e| TeError::Tokenizer(e.to_string()))?;

        let mut vocabulary = Vocabulary::new();
        for (token, id) in &parsed.vocab {
            vocabulary.insert(token, *id);
        }

        let mut merges = BpeMerges::new();
        for (a, b) in &parsed.merges {
            let merged = format!("{a}{b}");
            let merged_id = parsed.vocab.get(&merged).copied().ok_or_else(|| {
                TeError::Tokenizer(format!("merged token {merged:?} not in vocab"))
            })?;
            merges.add_merge(a, b, merged_id);
        }

        let unk_id = parsed
            .unk_token
            .as_deref()
            .and_then(|t| parsed.vocab.get(t).copied())
            .unwrap_or(0);

        Ok(Self {
            vocabulary,
            merges,
            unk_id,
        })
    }

    /// Tokenize a prompt into padded ids + attention mask of length `max_len`,
    /// applying the Qwen3 chat-template wrapping.
    ///
    /// If the wrapped sequence exceeds `max_len` it is truncated to `max_len`
    /// (matching `truncation`); the mask then has no padding zeros.
    ///
    /// # Errors
    /// Propagates any error from [`Self::encode_prompt`].
    pub fn tokenize(&self, prompt: &str, max_len: usize) -> TeResult<TokenizerOutput> {
        let mut ids: Vec<u32> = Vec::with_capacity(max_len);
        ids.extend_from_slice(&PREFIX);
        ids.extend(self.encode_prompt(prompt)?);
        ids.extend_from_slice(&SUFFIX);

        // Truncate then pad to max_len.
        if ids.len() > max_len {
            ids.truncate(max_len);
        }
        let real = ids.len();
        let mut attention_mask = vec![1i32; real];
        if real < max_len {
            ids.resize(max_len, PAD_ID);
            attention_mask.resize(max_len, 0);
        }
        Ok(TokenizerOutput {
            input_ids: ids,
            attention_mask,
        })
    }

    /// BPE-encode the bare prompt `P` (no chat-template wrapping) to ids.
    ///
    /// Pre-tokenizes with [`pretokenize_gpt2`] (the canonical HuggingFace
    /// `ByteLevel` `Split` regex, driven by `fancy-regex` — TOK-03), remaps
    /// each piece's UTF-8 bytes through the GPT-2 bytes→unicode table, and
    /// BPE-merges with [`bpe_encode_bytelevel`]. A byte-level piece absent
    /// from the vocabulary maps to `unk_id` (this type's private fallback
    /// field) rather than erroring —
    /// unreachable for a well-formed byte-level `tokenizer.json` (every one
    /// of the 256 remapped byte characters always has an entry), matching
    /// `bpe_encode_bytelevel`'s documented contract.
    ///
    /// # Errors
    /// Currently infallible (kept as a `Result` for API stability and
    /// forward-compatibility with a stricter validation mode).
    pub fn encode_prompt(&self, prompt: &str) -> TeResult<Vec<u32>> {
        let byte_to_unicode = bytes_to_unicode_map();
        let mut ids = Vec::new();
        for piece in pretokenize_gpt2(prompt) {
            let mut byte_level = String::with_capacity(piece.len());
            for &b in piece.as_bytes() {
                byte_level.push(byte_to_unicode[b as usize]);
            }
            ids.extend(bpe_encode_bytelevel(
                &byte_level,
                &self.vocabulary,
                &self.merges,
                self.unk_id,
            ));
        }
        Ok(ids)
    }
}

/// Identity pass-through, **not** Unicode NFC normalization (see the module
/// docs' "Normalization scope" section for why the name no longer claims
/// otherwise). ASCII input is already in NFC, so this is correct for the
/// ASCII/Latin prompts this tokenizer targets; non-ASCII input that would need
/// real composition/decomposition passes through unrecomposed.
#[allow(dead_code)] // Kept for documentation/API-history purposes; see module docs.
fn normalize_ascii_only(s: &str) -> String {
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_to_unicode_known_points() {
        // Wiring smoke-check (TOK-10): the same well-known GPT-2 mapping
        // values, now sourced from `oxibonsai_tokenizer` instead of a local
        // duplicate table.
        let t = bytes_to_unicode_map();
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        assert_eq!(t[b'a' as usize], 'a');
    }

    #[test]
    fn pretokenize_canonical_prompt() {
        // Wiring smoke-check: `pretokenize_gpt2` (fancy-regex-driven, owned
        // by `oxibonsai-tokenizer`) must still split the canonical example
        // prompt exactly as the old local scanner did.
        let p = pretokenize_gpt2("a tiny bonsai tree in a ceramic pot");
        assert_eq!(
            p,
            vec!["a", " tiny", " bonsai", " tree", " in", " a", " ceramic", " pot"]
        );
    }

    #[test]
    fn bpe_merges_by_rank() {
        // Tiny vocab: chars a,b,c plus merges (a,b)->ab rank0, (ab,c)->abc rank1.
        let json = r#"{
            "model": {
                "vocab": {"a":0,"b":1,"c":2,"ab":3,"abc":4},
                "merges": [["a","b"],["ab","c"]]
            }
        }"#;
        let tok = Qwen3Tokenizer::from_json_str(json).expect("parse");
        // "abc" byte-level is "abc" (all printable ASCII) -> merge to ["abc"] (id 4).
        assert_eq!(tok.encode_prompt("abc").expect("encode"), vec![4]);
        // "ab" -> [3]; "ba" -> no merge -> [1, 0] ("b" then "a").
        assert_eq!(tok.encode_prompt("ab").expect("encode"), vec![3]);
        assert_eq!(tok.encode_prompt("ba").expect("encode"), vec![1, 0]);
    }

    /// Merge-order equivalence regression net (TOK-10 mandatory pre-check).
    ///
    /// The deleted local `Qwen3Tokenizer::bpe` merged **all** non-overlapping
    /// occurrences of the single best-ranked pair per round before
    /// re-scanning (the classic reference BPE algorithm — its own removed
    /// doc comment asserted this explicitly). `oxibonsai_tokenizer::bpe`'s
    /// replacement merges **one** occurrence per min-heap pop, with ties at
    /// equal rank broken leftmost-first. These are two different
    /// *implementations* of the same underlying process (a heap
    /// pop-and-relink can never let a later occurrence of the current
    /// best-ranked pair be pre-empted by anything higher-priority that
    /// wasn't already queued ahead of it), and the values below were
    /// captured empirically from the OLD implementation, on an adversarial
    /// corpus specifically designed to exercise repeated/overlapping pair
    /// occurrences within one word (a run of identical characters, and a
    /// chain of distinct competing pairs), before it was deleted — so this
    /// pins that the rewrite did not silently change segmentation.
    #[test]
    fn merge_order_equivalence_regression_net() {
        let json = r#"{
            "model": {
                "vocab": {
                    "a":1,"b":2,"c":3,"d":4,"e":5,
                    "aa":20,"aaaa":21,
                    "bc":22,"cd":23,"bcd":24,"de":25
                },
                "merges": [["a","a"],["aa","aa"],["b","c"],["c","d"],["bc","d"],["d","e"]]
            }
        }"#;
        let tok = Qwen3Tokenizer::from_json_str(json).expect("parse");

        let cases: &[(&str, &[u32])] = &[
            ("aaaa", &[21]),         // "aaaa" (single merged token)
            ("aaaaaa", &[21, 20]),   // "aaaa","aa"
            ("aaa", &[20, 1]),       // "aa","a"
            ("abab", &[1, 2, 1, 2]), // no (a,b)/(b,a) rule at all: unchanged
            ("bcde", &[24, 5]),      // "bcd","e"
            ("bcda", &[24, 1]),      // "bcd","a"
        ];
        for (word, expected_ids) in cases {
            let ids = tok.encode_prompt(word).expect("encode");
            assert_eq!(
                &ids, expected_ids,
                "merge-order divergence for {word:?}: got {ids:?}, expected {expected_ids:?} \
                 (captured from the pre-rewrite reference implementation)"
            );
        }
    }

    #[test]
    fn chat_template_wraps_and_pads() {
        // vocab with just 'a' (id 64, matching the real 'a').
        let json = r#"{ "model": { "vocab": {"a":64}, "merges": [] } }"#;
        let tok = Qwen3Tokenizer::from_json_str(json).expect("parse");
        let out = tok.tokenize("a", 8).expect("tok");
        // prefix(3) + [64] + suffix(9) = 13 > 8 -> truncated to 8, no padding.
        assert_eq!(out.input_ids.len(), 8);
        assert_eq!(&out.input_ids[..4], &[IM_START, USER, NL, 64]);
        assert!(out.attention_mask.iter().all(|&m| m == 1));

        // With a larger max_len, the tail pads with PAD_ID and mask zeros.
        let out2 = tok.tokenize("a", 20).expect("tok");
        assert_eq!(out2.input_ids.len(), 20);
        assert_eq!(
            out2.input_ids[13..]
                .iter()
                .filter(|&&x| x == PAD_ID)
                .count(),
            7
        );
        assert_eq!(out2.attention_mask.iter().filter(|&&m| m == 0).count(), 7);
        assert_eq!(out2.attention_mask.iter().filter(|&&m| m == 1).count(), 13);
    }

    #[test]
    fn normalize_ascii_only_is_identity() {
        assert_eq!(normalize_ascii_only("hello world"), "hello world");
    }

    /// Full golden match — gated on the real `tokenizer.json` being present.
    /// Point `OXI_TE_TOKENIZER_DIR` at the `text_encoder-mlx-4bit` dir to run;
    /// skipped otherwise.
    #[test]
    fn golden_ids_match_when_available() {
        let dir_buf = match std::env::var("OXI_TE_TOKENIZER_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => {
                eprintln!("skip: set OXI_TE_TOKENIZER_DIR to the text_encoder dir");
                return;
            }
        };
        let dir = dir_buf.as_path();
        if !dir.join("tokenizer.json").exists() {
            eprintln!("skip: tokenizer.json not present");
            return;
        }
        let tok = Qwen3Tokenizer::open(dir).expect("open tokenizer");
        let out = tok
            .tokenize("a tiny bonsai tree in a ceramic pot", 512)
            .expect("tokenize");
        let expect_head: [u32; 21] = [
            151644, 872, 198, 64, 13673, 81034, 2143, 4916, 304, 264, 42024, 3338, 151645, 198,
            151644, 77091, 198, 151667, 271, 151668, 271,
        ];
        assert_eq!(&out.input_ids[..21], &expect_head);
        assert!(out.input_ids[21..].iter().all(|&x| x == PAD_ID));
        assert_eq!(out.attention_mask.iter().filter(|&&m| m == 1).count(), 21);
    }
}
