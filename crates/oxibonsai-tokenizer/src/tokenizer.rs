//! High-level OxiBonsai tokenizer: BPE + Unigram + WordPiece + char-level fallback.
//!
//! [`OxiTokenizer`] ties together a [`Vocabulary`], a [`BpeMerges`] table, and
//! a [`TokenizerConfig`] into a complete encode/decode API that is
//! `no_std`-friendly and WASM-compatible.
//!
//! When a [`crate::unigram::UnigramVocab`] is attached via
//! [`OxiTokenizer::with_unigram`], encoding switches to Viterbi segmentation
//! instead of BPE.
//!
//! When a [`crate::wordpiece::WordPieceVocab`] is attached via
//! [`OxiTokenizer::with_wordpiece`], encoding switches to greedy WordPiece
//! segmentation, which is the algorithm used by BERT, RoBERTa, DeBERTa,
//! DistilBERT, and ALBERT.

use std::collections::HashSet;

use tracing::debug;

use crate::{
    bpe::{
        bpe_encode, bpe_encode_bytelevel, byte_fallback_id, pretokenize, pretokenize_gpt2,
        BpeMerges,
    },
    error::{TokenizerError, TokenizerResult},
    hf_format::bytes_to_unicode_map,
    vocab::Vocabulary,
};

// ── TokenizerConfig ───────────────────────────────────────────────────────────

/// Configuration knobs for an [`OxiTokenizer`].
///
/// Marked `#[non_exhaustive]` so that new optional knobs can be added in
/// future minor releases without breaking downstream code.  Inside this crate
/// struct literals with `..Default::default()` continue to work.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TokenizerConfig {
    /// Whether to prepend a BOS (beginning-of-sequence) token.
    pub add_bos: bool,
    /// Whether to append an EOS (end-of-sequence) token.
    pub add_eos: bool,
    /// Token ID used for BOS.
    pub bos_token_id: u32,
    /// Token ID used for EOS.
    pub eos_token_id: u32,
    /// Token ID used for unknown tokens (fallback).
    pub unk_token_id: u32,
    /// Token ID used for padding.
    pub pad_token_id: u32,
    /// Optional maximum output length (tokens are truncated, not padded).
    pub max_length: Option<usize>,
    /// When `true`, the decoder applies the GPT-2 **bytes ↔ unicode** inverse
    /// map to every token string before emitting bytes (see
    /// [`crate::hf_format`]).  When `false`, the legacy `Ġ`-stripping path is
    /// used (same behaviour as 0.1.x).
    ///
    /// `from_json_file` / `OxiTokenizer::from_hf_tokenizer_json` set this to
    /// `true` automatically; hand-built configs default to `false` for
    /// backwards compatibility.
    pub byte_level_decode: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
            bos_token_id: 1,
            eos_token_id: 2,
            unk_token_id: 0,
            pad_token_id: 3,
            max_length: None,
            byte_level_decode: false,
        }
    }
}

// ── OxiTokenizer ─────────────────────────────────────────────────────────────

/// Pure Rust BPE / Unigram / WordPiece tokenizer compatible with MeCrab and the WASM target.
///
/// The tokenizer supports:
/// - Standard BPE encoding via a merge table
/// - Viterbi Unigram encoding (HuggingFace `"Unigram"` model type)
/// - Greedy WordPiece encoding (HuggingFace `"WordPiece"` model type — BERT family)
/// - Optional BOS/EOS injection
/// - Byte-fallback for out-of-vocabulary bytes
/// - Character-level mode (no trained vocab needed — useful in tests)
pub struct OxiTokenizer {
    vocab: Vocabulary,
    merges: BpeMerges,
    config: TokenizerConfig,
    /// The set of special token IDs for quick membership tests.
    special_ids: HashSet<u32>,
    /// Registered special/added-token strings (e.g. `<|im_start|>`) paired with
    /// their trained IDs, sorted by **descending byte length** so that the
    /// left-most **longest** match wins during the pre-split.
    ///
    /// These substrings are carved out of the input atomically before any
    /// model-specific segmentation runs, so chat-template markers map to their
    /// trained IDs instead of being shredded into UNK by pre-tokenization.
    special_pieces: Vec<(String, u32)>,
    /// GPT-2 byte→unicode table, precomputed once so the ByteLevel encode path
    /// does not rebuild the 256-entry map for every byte.
    byte_to_unicode: [char; 256],
    /// Optional Unigram vocabulary for Viterbi-based segmentation.
    ///
    /// When `Some`, the tokenizer dispatches to Unigram encoding instead of
    /// BPE.  When `None`, the BPE or WordPiece path is used.
    unigram: Option<crate::unigram::UnigramVocab>,
    /// Optional WordPiece vocabulary for BERT-style greedy segmentation.
    ///
    /// When `Some`, the tokenizer dispatches to WordPiece encoding.  This
    /// takes precedence over the BPE path but is checked after Unigram.
    /// When `None`, the BPE path (or Unigram if attached) is used.
    wordpiece: Option<crate::wordpiece::WordPieceVocab>,
}

impl OxiTokenizer {
    /// Construct a tokenizer from pre-built components.
    ///
    /// Sets `unigram` and `wordpiece` to `None` — the BPE path is used for
    /// encoding.
    pub fn new(vocab: Vocabulary, merges: BpeMerges, config: TokenizerConfig) -> Self {
        let special_ids = build_special_ids(&config);
        let special_pieces = build_special_pieces(&vocab);
        Self {
            vocab,
            merges,
            config,
            special_ids,
            special_pieces,
            byte_to_unicode: bytes_to_unicode_map(),
            unigram: None,
            wordpiece: None,
        }
    }

    /// Construct a Unigram tokenizer from pre-built components.
    ///
    /// The `unigram_vocab` is used for Viterbi-based segmentation; the `vocab`
    /// is kept for decode operations (ID → token string).  An empty
    /// [`BpeMerges`] table is stored for API consistency.
    pub fn with_unigram(
        vocab: Vocabulary,
        unigram_vocab: crate::unigram::UnigramVocab,
        config: TokenizerConfig,
    ) -> Self {
        let special_ids = build_special_ids(&config);
        let special_pieces = build_special_pieces(&vocab);
        Self {
            vocab,
            merges: BpeMerges::new(),
            config,
            special_ids,
            special_pieces,
            byte_to_unicode: bytes_to_unicode_map(),
            unigram: Some(unigram_vocab),
            wordpiece: None,
        }
    }

    /// Construct a WordPiece tokenizer from pre-built components.
    ///
    /// The `wordpiece_vocab` is used for greedy longest-match-first
    /// segmentation (BERT/RoBERTa/DeBERTa model family); the `vocab` is kept
    /// for decode operations (ID → token string).  An empty [`BpeMerges`]
    /// table is stored for API consistency.
    pub fn with_wordpiece(
        vocab: Vocabulary,
        wordpiece_vocab: crate::wordpiece::WordPieceVocab,
        config: TokenizerConfig,
    ) -> Self {
        let special_ids = build_special_ids(&config);
        let special_pieces = build_special_pieces(&vocab);
        Self {
            vocab,
            merges: BpeMerges::new(),
            config,
            special_ids,
            special_pieces,
            byte_to_unicode: bytes_to_unicode_map(),
            unigram: None,
            wordpiece: Some(wordpiece_vocab),
        }
    }

    /// Return `true` if this tokenizer uses Unigram (Viterbi) segmentation.
    pub fn is_unigram(&self) -> bool {
        self.unigram.is_some()
    }

    /// Return `true` if this tokenizer uses WordPiece (BERT-family) segmentation.
    pub fn is_wordpiece(&self) -> bool {
        self.wordpiece.is_some()
    }

    /// Encode a single text string into a sequence of token IDs.
    ///
    /// Steps:
    /// 1. Carve out any registered special/added tokens (e.g. `<|im_start|>`)
    ///    as atomic IDs, so chat-template markers are not shredded by generic
    ///    pre-tokenization.
    /// 2. Pre-tokenize each remaining segment into words.
    /// 3. Encode each word via WordPiece, Unigram Viterbi, or BPE.
    /// 4. Optionally prepend BOS and append EOS.
    /// 5. Optionally truncate to `config.max_length`.
    pub fn encode(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        debug!(text_len = text.len(), "encoding text");

        let mut ids: Vec<u32> = Vec::new();

        if self.config.add_bos {
            ids.push(self.config.bos_token_id);
        }

        self.encode_with_special_tokens(text, &mut ids);

        if self.config.add_eos {
            ids.push(self.config.eos_token_id);
        }

        // Truncate if configured.
        if let Some(max) = self.config.max_length {
            ids.truncate(max);
        }

        Ok(ids)
    }

    /// Split `text` on registered special/added tokens (longest-match-first)
    /// and encode the non-special segments with the active model path.
    ///
    /// Special tokens are emitted verbatim as their trained IDs; only the
    /// segments *between* them are handed to the pre-tokenizer / BPE / Unigram /
    /// WordPiece encoder.  When no special tokens are registered this collapses
    /// to a single call on the whole input.
    fn encode_with_special_tokens(&self, text: &str, ids: &mut Vec<u32>) {
        if self.special_pieces.is_empty() {
            self.encode_segment(text, ids);
            return;
        }

        let mut rest = text;
        while !rest.is_empty() {
            match self.find_leftmost_special(rest) {
                Some((start, len, special_id)) => {
                    if start > 0 {
                        self.encode_segment(&rest[..start], ids);
                    }
                    ids.push(special_id);
                    rest = &rest[start + len..];
                }
                None => {
                    self.encode_segment(rest, ids);
                    break;
                }
            }
        }
    }

    /// Find the left-most (and, at a tie, longest) registered special token in
    /// `hay`, returning `(byte_start, byte_len, id)`.
    fn find_leftmost_special(&self, hay: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (piece, id) in &self.special_pieces {
            if let Some(pos) = hay.find(piece.as_str()) {
                let len = piece.len();
                let better = match best {
                    None => true,
                    Some((best_start, best_len, _)) => {
                        pos < best_start || (pos == best_start && len > best_len)
                    }
                };
                if better {
                    best = Some((pos, len, *id));
                }
            }
        }
        best
    }

    /// Encode a single non-special segment with the active model path.
    fn encode_segment(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }

        if let Some(wp) = &self.wordpiece {
            // WordPiece path: greedy longest-match-first segmentation of the
            // full text (the WordPieceVocab splits on whitespace internally).
            ids.extend_from_slice(&wp.encode(text));
        } else if let Some(unigram) = &self.unigram {
            // Unigram / SentencePiece path: Viterbi segmentation on the
            // *full* text without any GPT-2-style pretokenisation.
            //
            // SentencePiece callers (e.g. GemmaTokenizer) perform their own
            // ▁-normalisation before calling `encode`; splitting on whitespace
            // would silently discard newlines and other control characters that
            // have explicit vocabulary entries (e.g. token 107 = `\n`).
            ids.extend_from_slice(&unigram.encode(text));
        } else if self.config.byte_level_decode {
            // ByteLevel BPE path (GPT-2 / Qwen3 / Llama-3): whitespace-preserving
            // pre-tokenization, then remap every UTF-8 byte through the GPT-2
            // bytes→unicode table before running the merge table.  This is the
            // only path that correctly encodes non-ASCII text (CJK, emoji,
            // accented Latin) and non-space whitespace (tabs, newlines, repeated
            // spaces) against a byte-level vocabulary.
            for piece in pretokenize_gpt2(text) {
                let mut byte_level = String::with_capacity(piece.len());
                for &byte in piece.as_bytes() {
                    byte_level.push(self.byte_to_unicode[byte as usize]);
                }
                let piece_ids = bpe_encode_bytelevel(
                    &byte_level,
                    &self.vocab,
                    &self.merges,
                    self.config.unk_token_id,
                );
                ids.extend_from_slice(&piece_ids);
            }
        } else {
            // Legacy (non-byte-level) BPE path: GPT-2-lookalike pretokenisation
            // + merge-table encoding with `<0xHH>` byte fallback.
            for word in &pretokenize(text) {
                let word_ids = bpe_encode(word, &self.vocab, &self.merges);
                if word_ids.is_empty() {
                    // Byte-fallback path: encode each UTF-8 byte explicitly.
                    for byte in word.as_bytes() {
                        let fallback = byte_fallback_id(*byte);
                        let fallback_id = self.vocab.get_id(&fallback);
                        ids.push(fallback_id.unwrap_or(self.config.unk_token_id));
                    }
                } else {
                    ids.extend_from_slice(&word_ids);
                }
            }
        }
    }

    /// Encode a batch of texts in sequence (returns one `Vec<u32>` per input).
    pub fn encode_batch(&self, texts: &[&str]) -> TokenizerResult<Vec<Vec<u32>>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Decode a sequence of token IDs back into a string.
    ///
    /// Special tokens (BOS, EOS, PAD, UNK) are silently skipped.
    /// Byte-fallback tokens (`<0xHH>`) are decoded back to their original byte.
    /// Unknown IDs that are not in the vocabulary produce `\u{FFFD}` (replacement
    /// character) rather than an error, to be maximally robust.
    ///
    /// When `config.byte_level_decode` is `true`, tokens are run through the
    /// full 256-entry GPT-2 **unicode → byte** inverse map (see
    /// [`crate::hf_format`]).  Otherwise the legacy `Ġ`-stripping path is used.
    pub fn decode(&self, ids: &[u32]) -> TokenizerResult<String> {
        let bytes = self.decode_to_bytes(ids);
        String::from_utf8(bytes).map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    /// Decode to raw bytes — used by both [`Self::decode`] and the streaming
    /// decoder so that the two paths stay byte-for-byte identical.
    pub(crate) fn decode_to_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);

        for &id in ids {
            self.decode_id_into(id, &mut bytes);
        }

        bytes
    }

    /// Append the UTF-8 bytes for a single token ID to `bytes`.
    ///
    /// Special tokens are silently dropped.  Unknown IDs produce `\u{FFFD}`.
    pub(crate) fn decode_id_into(&self, id: u32, bytes: &mut Vec<u8>) {
        if self.special_ids.contains(&id) {
            return;
        }

        let token = match self.vocab.get_token(id) {
            Some(t) => t,
            None => {
                bytes.extend_from_slice("\u{FFFD}".as_bytes());
                return;
            }
        };

        // Byte-fallback tokens: `<0xHH>` → raw byte.
        if let Some(byte) = parse_byte_fallback(token) {
            bytes.push(byte);
            return;
        }

        if self.config.byte_level_decode {
            // Full GPT-2 bytes-to-unicode inverse mapping.
            for ch in token.chars() {
                if let Some(b) = crate::hf_format::unicode_to_byte(ch) {
                    bytes.push(b);
                } else {
                    // Non-byte-level character — emit UTF-8 verbatim.
                    let mut buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut buf);
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
        } else {
            // Legacy `Ġ`-stripping path — kept bit-for-bit identical to 0.1.x.
            let stripped = token.trim_start_matches('\u{0120}');
            if token.starts_with('\u{0120}') && !bytes.is_empty() {
                bytes.push(b' ');
            }
            bytes.extend_from_slice(stripped.as_bytes());
        }
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> TokenizerResult<String> {
        self.vocab
            .get_token(id)
            .map(|s| s.to_owned())
            .ok_or_else(|| TokenizerError::DecodeFailed(format!("unknown token id {id}")))
    }

    /// Return the total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    /// Construct a tokenizer from JSON-encoded vocabulary and merge lists.
    ///
    /// `vocab_json`: `{ "token": id, ... }`
    /// `merges_json`: `[["a", "b"], ...]` (ordered from highest to lowest priority)
    pub fn from_json(
        vocab_json: &str,
        merges_json: &str,
        config: TokenizerConfig,
    ) -> TokenizerResult<Self> {
        let vocab = Vocabulary::from_json(vocab_json)?;

        let raw_merges: Vec<(String, String)> = serde_json::from_str(merges_json)
            .map_err(|e| TokenizerError::InvalidJson(e.to_string()))?;

        let mut merges = BpeMerges::new();
        for (a, b) in &raw_merges {
            // The merged token name is the concatenation.
            let merged = format!("{a}{b}");
            let result_id = vocab.get_id(&merged).ok_or_else(|| {
                TokenizerError::InvalidVocab(format!("merged token {merged:?} not in vocabulary"))
            })?;
            merges.add_merge(a, b, result_id);
        }

        Ok(Self::new(vocab, merges, config))
    }

    /// Load a tokenizer from a HuggingFace-style `tokenizer.json` file.
    ///
    /// This routes through [`crate::hf_format::HfTokenizerJson`] which:
    ///
    /// 1. Parses the `model.vocab` map (token → id).
    /// 2. Parses the `model.merges` list (both string-pair and array-pair forms).
    /// 3. Picks up the `added_tokens` / `special_tokens` block.
    /// 4. Sets `byte_level_decode = true` on the returned config so that
    ///    decode() correctly reverses the GPT-2 bytes-to-unicode map.
    ///
    /// Any field not expressible in [`TokenizerConfig`] (truncation policy,
    /// normalizer variants, ...) is ignored but does not cause an error so
    /// that loading a live HF file "just works".
    pub fn from_json_file(path: impl AsRef<std::path::Path>) -> TokenizerResult<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_hf_tokenizer_json(&json)
    }

    /// In-memory variant of [`Self::from_json_file`] that takes the JSON as a
    /// `&str`.  Useful for WASM builds and for tests that embed a tokenizer
    /// fixture verbatim.
    pub fn from_hf_tokenizer_json(json: &str) -> TokenizerResult<Self> {
        let parsed = crate::hf_format::HfTokenizerJson::parse(json)?;
        parsed.into_tokenizer()
    }

    /// Begin streaming decode.  Returns a [`crate::streaming::StreamingDecoder`]
    /// that keeps UTF-8 state across `push_token` calls — essential for server
    /// code that emits one token at a time.
    pub fn streaming_decoder(&self) -> crate::streaming::StreamingDecoder<'_> {
        crate::streaming::StreamingDecoder::new(self)
    }

    /// Access the tokenizer configuration (read-only).
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Access the vocabulary (read-only).
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Access the merge table (read-only).
    pub fn merges(&self) -> &BpeMerges {
        &self.merges
    }

    /// Create a character-level tokenizer (no trained merges) for testing
    /// and examples.
    ///
    /// Assigns IDs 4..vocab_size to printable ASCII characters (space = 4,
    /// '!' = 5, ...) with IDs 0-3 reserved for UNK/BOS/EOS/PAD.
    ///
    /// This tokenizer has no BPE merges: each character is its own token.
    /// The `_stub` suffix is retained for API compatibility.
    pub fn char_level_stub(vocab_size: usize) -> Self {
        assert!(
            vocab_size >= 4,
            "char_level_stub requires vocab_size >= 4 for special tokens"
        );

        let mut vocab = Vocabulary::new();
        vocab.add_special("<unk>", 0);
        vocab.add_special("<bos>", 1);
        vocab.add_special("<eos>", 2);
        vocab.add_special("<pad>", 3);

        // Fill remaining slots with printable ASCII characters.
        let mut next_id = 4u32;
        for byte in 0x20u8..=0x7Eu8 {
            if next_id as usize >= vocab_size {
                break;
            }
            let ch = char::from(byte);
            vocab.insert(&ch.to_string(), next_id);
            next_id += 1;
        }

        // Also populate byte-fallback tokens for any remaining slots.
        for byte in 0u8..=255u8 {
            if next_id as usize >= vocab_size {
                break;
            }
            let fallback = byte_fallback_id(byte);
            if vocab.get_id(&fallback).is_none() {
                vocab.insert(&fallback, next_id);
                next_id += 1;
            }
        }

        let config = TokenizerConfig {
            add_bos: false,
            add_eos: false,
            bos_token_id: 1,
            eos_token_id: 2,
            unk_token_id: 0,
            pad_token_id: 3,
            max_length: None,
            byte_level_decode: false,
        };

        let merges = BpeMerges::new();
        // Use Self::new which initialises both unigram and wordpiece to None.
        Self::new(vocab, merges, config)
    }

    // ── Special token helpers ─────────────────────────────────────────────

    /// Return the BOS token ID from the configuration.
    pub fn bos_id(&self) -> u32 {
        self.config.bos_token_id
    }

    /// Return the EOS token ID from the configuration.
    pub fn eos_id(&self) -> u32 {
        self.config.eos_token_id
    }

    /// Return `true` if `id` is one of the configured special token IDs.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Build the set of special token IDs from a config.
fn build_special_ids(config: &TokenizerConfig) -> HashSet<u32> {
    let mut set = HashSet::new();
    set.insert(config.bos_token_id);
    set.insert(config.eos_token_id);
    set.insert(config.unk_token_id);
    set.insert(config.pad_token_id);
    set
}

/// Build the ordered list of atomically-protected token substrings used by
/// the encoder to carve out added tokens (e.g. `<|im_start|>`,
/// `<tool_call>`, `<|fim_prefix|>`) before segmentation.
///
/// Draws from [`Vocabulary::protected_tokens`] — the union of `special ==
/// true` tokens and non-special `added_tokens` — matching HuggingFace's
/// `AddedVocabulary` semantics where *every* added token is protected from
/// pre-tokenization, not just the ones flagged `special`.
///
/// The list is sorted by **descending byte length** so that a longest-match
/// wins when two protected tokens share a common prefix.
fn build_special_pieces(vocab: &Vocabulary) -> Vec<(String, u32)> {
    let mut pieces: Vec<(String, u32)> = vocab
        .protected_tokens()
        .filter(|(tok, _)| !tok.is_empty())
        .map(|(tok, id)| (tok.to_owned(), id))
        .collect();
    // Longest first; ties broken by token string for deterministic ordering.
    pieces.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
    pieces
}

/// Parse a byte-fallback token like `<0x41>` and return the byte value.
///
/// Returns `None` if the token is not in the `<0xHH>` format.
fn parse_byte_fallback(token: &str) -> Option<u8> {
    let inner = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if inner.len() != 2 {
        return None;
    }
    u8::from_str_radix(inner, 16).ok()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_level_stub_encode_ascii() {
        let tok = OxiTokenizer::char_level_stub(200);
        let ids = tok.encode("ab").expect("encode should succeed");
        // Each char should map to a consistent non-zero ID.
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], 0); // not UNK
        assert_ne!(ids[1], 0);
        assert_ne!(ids[0], ids[1]); // 'a' ≠ 'b'
    }

    #[test]
    fn char_level_stub_bos_eos() {
        let mut tok = OxiTokenizer::char_level_stub(200);
        tok.config.add_bos = true;
        tok.config.add_eos = true;
        tok.special_ids = build_special_ids(&tok.config);
        let ids = tok.encode("hi").expect("encode should succeed");
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(*ids.last().expect("must have last element"), 2); // EOS
    }

    #[test]
    fn char_level_stub_vocab_size() {
        let tok = OxiTokenizer::char_level_stub(50);
        assert!(tok.vocab_size() <= 50);
        assert!(tok.vocab_size() >= 4); // at least special tokens
    }

    #[test]
    fn special_token_detection() {
        let tok = OxiTokenizer::char_level_stub(200);
        assert!(tok.is_special(0)); // UNK
        assert!(tok.is_special(1)); // BOS
        assert!(tok.is_special(2)); // EOS
        assert!(tok.is_special(3)); // PAD
        assert!(!tok.is_special(4)); // first real token
    }

    #[test]
    fn bos_eos_ids_match_config() {
        let tok = OxiTokenizer::char_level_stub(200);
        assert_eq!(tok.bos_id(), 1);
        assert_eq!(tok.eos_id(), 2);
    }

    #[test]
    fn decode_token_roundtrip() {
        let tok = OxiTokenizer::char_level_stub(200);
        // 'a' should map to some ID; we can look it up.
        let ids = tok.encode("a").expect("should encode");
        if let Some(&id) = ids.first() {
            let s = tok.decode_token(id).expect("decode_token should succeed");
            assert_eq!(s, "a");
        }
    }

    #[test]
    fn decode_unknown_id_returns_error() {
        let tok = OxiTokenizer::char_level_stub(50);
        let result = tok.decode_token(99_999);
        assert!(result.is_err());
    }

    #[test]
    fn max_length_truncates() {
        let mut tok = OxiTokenizer::char_level_stub(200);
        tok.config.max_length = Some(3);
        tok.special_ids = build_special_ids(&tok.config);
        let ids = tok.encode("hello world").expect("encode should succeed");
        assert!(ids.len() <= 3);
    }

    #[test]
    fn encode_batch_consistency() {
        let tok = OxiTokenizer::char_level_stub(200);
        let texts = ["ab", "cd", "ef"];
        let batch = tok
            .encode_batch(&texts)
            .expect("batch encode should succeed");
        assert_eq!(batch.len(), 3);
        for (i, ids) in batch.iter().enumerate() {
            let single = tok.encode(texts[i]).expect("single encode should succeed");
            assert_eq!(*ids, single);
        }
    }

    #[test]
    fn parse_byte_fallback_valid() {
        assert_eq!(parse_byte_fallback("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_fallback("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_fallback("<0xFF>"), Some(0xFF));
    }

    #[test]
    fn parse_byte_fallback_invalid() {
        assert_eq!(parse_byte_fallback("hello"), None);
        assert_eq!(parse_byte_fallback("<0x>"), None);
        assert_eq!(parse_byte_fallback("<0x1>"), None);
    }

    #[test]
    fn from_json_roundtrip() {
        let vocab_json = r#"{"a":10,"b":11,"ab":20,"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3}"#;
        let merges_json = r#"[["a","b"]]"#;
        let config = TokenizerConfig::default();
        let tok = OxiTokenizer::from_json(vocab_json, merges_json, config)
            .expect("from_json should succeed");
        assert_eq!(tok.vocab_size(), 7);
        // Encoding "ab" should produce a single merged token 20.
        let ids = tok.encode("ab").expect("encode should succeed");
        assert!(ids.contains(&20));
    }

    #[test]
    fn is_unigram_false_for_bpe() {
        let tok = OxiTokenizer::char_level_stub(200);
        assert!(!tok.is_unigram());
    }
}
