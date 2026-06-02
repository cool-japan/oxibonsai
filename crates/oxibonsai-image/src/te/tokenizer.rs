//! Qwen3 ByteLevel-BPE tokenizer for the Bonsai-Image text encoder.
//!
//! Reproduces the HuggingFace `tokenizers` pipeline used by the reference
//! (`tokenizer.json`): **NFC normalize → GPT-2 pre-tokenization regex → ByteLevel
//! byte→unicode mapping → BPE merges**, then splices the fixed Qwen3 chat-template
//! special-id prefix/suffix and right-pads to `max_len`.
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
//! ## NFC scope
//!
//! Normalization is applied for the ASCII range (identity) and is otherwise a
//! pass-through: image prompts are overwhelmingly ASCII/Latin and the canonical
//! goldens are ASCII. Full Unicode NFC (which would need composition tables) is a
//! deliberate non-goal here; non-ASCII prompts that require recomposition may
//! tokenize slightly differently from HF.

use std::collections::HashMap;
use std::path::Path;

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
pub struct Qwen3Tokenizer {
    /// Byte-level token string → id.
    vocab: HashMap<String, u32>,
    /// Ordered merge rule `(left, right)` → rank (lower = higher priority).
    merge_ranks: HashMap<(String, String), u32>,
    /// GPT-2 byte → unicode-char table.
    byte_to_unicode: [char; 256],
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
    /// # Errors
    /// [`TeError::Tokenizer`] if the JSON is malformed or missing the
    /// `model.vocab` / `model.merges` fields.
    pub fn from_json_str(text: &str) -> TeResult<Self> {
        let root: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| TeError::Tokenizer(format!("json parse: {e}")))?;
        let model = root
            .get("model")
            .ok_or_else(|| TeError::Tokenizer("no model field".into()))?;

        // vocab: { token: id }
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TeError::Tokenizer("no model.vocab object".into()))?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        for (k, v) in vocab_obj {
            let id = v
                .as_u64()
                .ok_or_else(|| TeError::Tokenizer(format!("vocab id for {k:?} not an int")))?;
            vocab.insert(k.clone(), id as u32);
        }

        // merges: either ["a b", ...] (older) or [["a","b"], ...] (newer).
        let merges = model
            .get("merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TeError::Tokenizer("no model.merges array".into()))?;
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = parse_merge(m)
                .ok_or_else(|| TeError::Tokenizer(format!("bad merge entry at {rank}")))?;
            merge_ranks.insert((a, b), rank as u32);
        }

        Ok(Self {
            vocab,
            merge_ranks,
            byte_to_unicode: build_byte_to_unicode(),
        })
    }

    /// Tokenize a prompt into padded ids + attention mask of length `max_len`,
    /// applying the Qwen3 chat-template wrapping.
    ///
    /// If the wrapped sequence exceeds `max_len` it is truncated to `max_len`
    /// (matching `truncation`); the mask then has no padding zeros.
    ///
    /// # Errors
    /// [`TeError::Tokenizer`] if a byte-level piece is not present in the vocab
    /// (which would indicate a corrupt `tokenizer.json`).
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
    /// # Errors
    /// As [`Self::tokenize`].
    pub fn encode_prompt(&self, prompt: &str) -> TeResult<Vec<u32>> {
        let normalized = nfc_ascii(prompt);
        let mut ids = Vec::new();
        for piece in pre_tokenize(&normalized) {
            // ByteLevel: map each UTF-8 byte of the piece to its unicode char.
            let mut byte_level = String::with_capacity(piece.len());
            for &b in piece.as_bytes() {
                byte_level.push(self.byte_to_unicode[b as usize]);
            }
            for tok in self.bpe(&byte_level) {
                let id = self
                    .vocab
                    .get(&tok)
                    .ok_or_else(|| TeError::Tokenizer(format!("token {tok:?} not in vocab")))?;
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    /// Apply BPE merges to a byte-level token string: start from single chars and
    /// repeatedly merge the adjacent pair with the lowest rank until none remain.
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() < 2 {
            return word;
        }
        loop {
            // Find the lowest-rank adjacent pair.
            let mut best_rank = u32::MAX;
            let mut best_idx: Option<usize> = None;
            for i in 0..word.len() - 1 {
                if let Some(&r) = self
                    .merge_ranks
                    .get(&(word[i].clone(), word[i + 1].clone()))
                {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(idx) = best_idx else { break };
            // Merge ALL non-overlapping occurrences of that best pair (the HF
            // reference rebuilds the word merging every occurrence of the chosen
            // pair left-to-right before re-scanning).
            let (a, b) = (word[idx].clone(), word[idx + 1].clone());
            let mut merged: Vec<String> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
            if word.len() < 2 {
                break;
            }
        }
        word
    }
}

/// Parse a merge entry: `["a","b"]` (array) or `"a b"` (space-joined string).
fn parse_merge(m: &serde_json::Value) -> Option<(String, String)> {
    if let Some(arr) = m.as_array() {
        if arr.len() == 2 {
            return Some((arr[0].as_str()?.to_string(), arr[1].as_str()?.to_string()));
        }
        return None;
    }
    let s = m.as_str()?;
    let sp = s.find(' ')?;
    Some((s[..sp].to_string(), s[sp + 1..].to_string()))
}

/// Build the GPT-2 byte→unicode table: printable ASCII/Latin map to themselves,
/// the remaining bytes map to `256 + n` codepoints. (Identical to the table used
/// by `tokenizers`' ByteLevel.)
fn build_byte_to_unicode() -> [char; 256] {
    // The "directly printable" byte ranges (kept as-is).
    let mut keep: Vec<u32> = Vec::new();
    keep.extend(b'!' as u32..=b'~' as u32);
    keep.extend(0xA1u32..=0xACu32);
    keep.extend(0xAEu32..=0xFFu32);

    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0u32..256 {
        if keep.contains(&b) {
            table[b as usize] = char::from_u32(b).unwrap_or('\0');
        } else {
            let c = char::from_u32(256 + n).unwrap_or('\0');
            table[b as usize] = c;
            n += 1;
        }
    }
    table
}

/// NFC normalization restricted to the ASCII range (identity). Non-ASCII passes
/// through unchanged (see module docs for the scope note).
fn nfc_ascii(s: &str) -> String {
    // ASCII is already in NFC; we do not recompose non-ASCII here.
    s.to_string()
}

/// GPT-2 pre-tokenization, a hand-written port of the regex
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` with `Isolated` behavior.
///
/// Emits the matched substrings in order (the whole input is consumed).
fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;

    let is_letter = |c: char| c.is_alphabetic();
    let is_number = |c: char| c.is_numeric();
    let is_ws = |c: char| c.is_whitespace();
    let is_nl = |c: char| c == '\r' || c == '\n';

    while i < n {
        let c = chars[i];

        // 1) contractions: '(?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == '\'' && i + 1 < n {
            if let Some(len) = match_contraction(&chars[i..]) {
                out.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }

        // 2) [^\r\n\p{L}\p{N}]?\p{L}+   (optional single non-nl/non-alnum, then letters)
        {
            let mut j = i;
            // optional leading char that is not \r,\n and not letter/number
            if !is_nl(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                // only consume it if a letter follows
                if j + 1 < n && is_letter(chars[j + 1]) {
                    j += 1;
                }
            }
            if j < n && is_letter(chars[j]) {
                while j < n && is_letter(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // 3) \p{N}  (a single number char — \p{N} with no +)
        if is_number(c) {
            out.push(c.to_string());
            i += 1;
            continue;
        }

        // 4)  ?[^\s\p{L}\p{N}]+[\r\n]*  (optional space, run of non-ws/non-alnum, trailing newlines)
        {
            let mut j = i;
            let mut consumed_space = false;
            if chars[j] == ' ' {
                // only if followed by a non-ws/non-alnum symbol
                if j + 1 < n
                    && !is_ws(chars[j + 1])
                    && !is_letter(chars[j + 1])
                    && !is_number(chars[j + 1])
                {
                    j += 1;
                    consumed_space = true;
                }
            }
            if j < n && !is_ws(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                while j < n && !is_ws(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                    j += 1;
                }
                while j < n && is_nl(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            // if we tentatively consumed a space but no symbol followed, fall
            // through to the whitespace branches with i unchanged.
            let _ = consumed_space;
        }

        // 5) \s*[\r\n]+   (whitespace ending in newlines)
        if is_ws(c) {
            // look ahead: a run of whitespace that contains a newline -> branch 5
            let mut j = i;
            while j < n && is_ws(chars[j]) && !is_nl(chars[j]) {
                j += 1;
            }
            if j < n && is_nl(chars[j]) {
                while j < n && is_nl(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // 6) \s+(?!\S)  and  7) \s+
        if is_ws(c) {
            let mut j = i;
            while j < n && is_ws(chars[j]) {
                j += 1;
            }
            // \s+(?!\S): if the whitespace run reaches end-of-text, take it all.
            // Otherwise \s+ but leaving the last space for a following word
            // (GPT-2 keeps a single leading space with the next token).
            if j == n {
                out.push(chars[i..j].iter().collect());
                i = j;
            } else if j - i >= 2 {
                // leave the final space to attach to the next token
                out.push(chars[i..j - 1].iter().collect());
                i = j - 1;
            } else {
                // single space followed by a non-space: it belongs to the next
                // token (handled by branch 2/4's optional leading space), so emit
                // nothing here and let the next iteration consume it. To avoid an
                // infinite loop, attach it as its own piece only if the next char
                // is itself whitespace (cannot happen here) — so advance by
                // pushing the space (it will byte-encode to 'Ġ').
                // In practice branch 2/4 already consumed the leading space, so
                // reaching here with a lone space means a space before a word that
                // those branches did not take; emit it standalone.
                out.push(chars[i..j].iter().collect());
                i = j;
            }
            continue;
        }

        // Fallback: emit the single char (should be unreachable for valid text).
        out.push(c.to_string());
        i += 1;
    }
    out
}

/// Match a leading contraction (`'s 't 're 've 'm 'll 'd`, case-insensitive),
/// returning its char length if present.
fn match_contraction(rest: &[char]) -> Option<usize> {
    // rest[0] == '\''
    let lower = |c: char| c.to_ascii_lowercase();
    if rest.len() >= 2 {
        let c1 = lower(rest[1]);
        // 3-char: 're 've 'll
        if rest.len() >= 3 {
            let c2 = lower(rest[2]);
            let three = matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l'));
            if three {
                return Some(3);
            }
        }
        // 2-char: 's 't 'm 'd
        if matches!(c1, 's' | 't' | 'm' | 'd') {
            return Some(2);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_to_unicode_known_points() {
        let t = build_byte_to_unicode();
        // space (0x20) is not in the printable-kept set → maps to 256+n; the
        // first remapped byte 0x00 → 256 ('Ā'), and 0x20 → 'Ġ' (0x120).
        assert_eq!(t[b' ' as usize], 'Ġ');
        assert_eq!(t[b'\n' as usize], 'Ċ');
        // 'a' stays 'a'.
        assert_eq!(t[b'a' as usize], 'a');
    }

    #[test]
    fn pretokenize_canonical_prompt() {
        let p = pre_tokenize("a tiny bonsai tree in a ceramic pot");
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
        // "abc" byte-level is "abc" (all printable ASCII) -> merge to ["abc"].
        let out = tok.bpe("abc");
        assert_eq!(out, vec!["abc".to_string()]);
        // "ab" -> ["ab"]; "ba" -> no merge -> ["b","a"].
        assert_eq!(tok.bpe("ab"), vec!["ab".to_string()]);
        assert_eq!(tok.bpe("ba"), vec!["b".to_string(), "a".to_string()]);
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
