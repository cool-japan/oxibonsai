//! JSON-grammar [`TokenConstraint`] implementation.
//!
//! Hosts [`JsonParseState`], the internal [`JsonMachine`] state machine, and the
//! public [`JsonConstraint`] that restricts generation to syntactically valid
//! JSON.
//!
//! # Two operating modes
//!
//! [`JsonConstraint`] can be constructed in two very different modes:
//!
//! * [`JsonConstraint::with_decoder`] — **the real mode.** The caller supplies a
//!   `decode_fn: Fn(u32) -> Option<String>` mapping each token id to the text it
//!   emits.  `allowed_tokens` / `advance` then operate on that decoded text, so
//!   the constraint is correct for any real subword tokenizer.
//! * [`JsonConstraint::new`] — **demonstration / toy mode only.** With no decoder
//!   the constraint treats each raw token id as a Unicode code point
//!   (`char::from_u32(id)`).  This is meaningful *only* for a synthetic vocabulary
//!   where `token_id == codepoint` (e.g. an ASCII byte vocab in a unit test).  For
//!   a real tokenizer it produces masks and accept/reject decisions that have no
//!   relationship to the generated text — do **not** use it in production.  Prefer
//!   [`JsonConstraint::with_decoder`], or the byte-correct
//!   [`crate::grammar::GrammarConstraint`] compiled from a JSON schema.

use super::decoder::TokenTextIndex;
use super::error_trait::TokenConstraint;

// ─────────────────────────────────────────────────────────────────────────────
// JsonParseState
// ─────────────────────────────────────────────────────────────────────────────

/// Internal parser state for [`JsonConstraint`].
#[derive(Debug, Clone, PartialEq)]
pub enum JsonParseState {
    /// Before any character has been emitted.
    Start,
    /// Inside a JSON object `{`, waiting for a key or `}`.
    InObject,
    /// Inside a string that is an object key.
    InObjectKey,
    /// After an object key, expecting `:`.
    AfterKey,
    /// After `:`, waiting for a value.
    InObjectValue,
    /// Inside a JSON array `[`, waiting for a value or `]`.
    InArray,
    /// After a value inside an array, waiting for `,` or `]`.
    InArrayValue,
    /// Inside a string value (or key).
    InString,
    /// Immediately after a `\` inside a string.
    InStringEscape,
    /// Inside a number literal.
    InNumber,
    /// Inside a boolean keyword (`true` / `false`).
    InBool,
    /// Inside `null`.
    InNull,
    /// Top-level value is complete.
    Complete,
    /// An error has been encountered.
    Error,
}

// ─────────────────────────────────────────────────────────────────────────────
// JsonMachine — the character-level parse state machine
// ─────────────────────────────────────────────────────────────────────────────

/// The character-by-character JSON parse state machine.
///
/// This is intentionally small and cheap to [`Clone`] so that
/// [`JsonConstraint::allowed_tokens`] can speculatively feed a candidate token's
/// full text through a clone without disturbing the live state.
#[derive(Debug, Clone, PartialEq)]
struct JsonMachine {
    state: JsonParseState,
    depth: usize,
    expecting_comma_or_close: bool,
    /// Keyword / number accumulator (bounded; cleared on value completion).
    keyword_buf: String,
    /// Stack of context: `'o'` = object, `'a'` = array.
    context_stack: Vec<char>,
}

impl JsonMachine {
    fn new() -> Self {
        Self {
            state: JsonParseState::Start,
            depth: 0,
            expecting_comma_or_close: false,
            keyword_buf: String::new(),
            context_stack: Vec::new(),
        }
    }

    /// Returns `true` if we are currently inside a string.
    fn is_in_string(&self) -> bool {
        matches!(
            self.state,
            JsonParseState::InString | JsonParseState::InStringEscape
        )
    }

    /// Returns `true` when the current state accepts *any* character as the next
    /// character, so the valid first-character set cannot be enumerated.
    ///
    /// This mirrors the catch-all arms of [`feed_char`](Self::feed_char): inside
    /// a string (key or value) any character continues the string, and directly
    /// after a `\` any character is accepted as an escape.
    fn accepts_arbitrary_char(&self) -> bool {
        matches!(
            self.state,
            JsonParseState::InString | JsonParseState::InObjectKey | JsonParseState::InStringEscape
        )
    }

    /// Returns the set of ASCII characters that are valid as the *next* character
    /// given the current parse state.
    ///
    /// For states where [`accepts_arbitrary_char`](Self::accepts_arbitrary_char)
    /// is `true` this list is *not* exhaustive (non-ASCII characters are also
    /// accepted); callers must consult `accepts_arbitrary_char` first.
    fn valid_next_chars(&self) -> Vec<char> {
        match &self.state {
            JsonParseState::Start => {
                vec![
                    '{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f',
                    'n', ' ', '\t', '\n',
                ]
            }
            JsonParseState::InObject => {
                if self.expecting_comma_or_close {
                    vec![',', '}', ' ', '\t', '\n']
                } else {
                    vec!['"', '}', ' ', '\t', '\n']
                }
            }
            JsonParseState::InObjectKey => {
                // Any printable ASCII except " (which closes) and \ (handled separately).
                let mut v: Vec<char> = (0x20u8..0x7fu8)
                    .filter(|&c| c != b'"')
                    .map(|c| c as char)
                    .collect();
                v.push('"'); // closing quote
                v.push('\\');
                v
            }
            JsonParseState::AfterKey => vec![':', ' ', '\t'],
            JsonParseState::InObjectValue
            | JsonParseState::InArrayValue
            | JsonParseState::InArray => {
                // Start of any JSON value.
                if self.expecting_comma_or_close {
                    if self.context_stack.last() == Some(&'o') {
                        vec![',', '}', ' ', '\t', '\n']
                    } else {
                        vec![',', ']', ' ', '\t', '\n']
                    }
                } else {
                    vec![
                        '{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't',
                        'f', 'n', ' ', '\t', '\n',
                    ]
                }
            }
            JsonParseState::InString => {
                let mut v: Vec<char> = (0x20u8..0x7fu8)
                    .filter(|&c| c != b'"')
                    .map(|c| c as char)
                    .collect();
                v.push('"');
                v.push('\\');
                v
            }
            JsonParseState::InStringEscape => {
                vec!['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
            }
            JsonParseState::InNumber => {
                vec![
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'E', '+', '-', ',',
                    '}', ']', ' ', '\t', '\n',
                ]
            }
            JsonParseState::InBool | JsonParseState::InNull => {
                // Allow letters that could continue the keyword.
                vec![
                    'r', 'u', 'e', 'a', 'l', 's', 'i', 'o', 'n', 't', 'f', ',', '}', ']', ' ',
                    '\t', '\n',
                ]
            }
            JsonParseState::Complete => {
                // After a complete value, allow whitespace.
                vec![' ', '\t', '\n']
            }
            JsonParseState::Error => vec![],
        }
    }

    /// Feed a single character through the state machine.
    fn feed_char(&mut self, ch: char) {
        match &self.state.clone() {
            JsonParseState::Error | JsonParseState::Complete => {
                // In Complete state whitespace is ok; anything else is an error.
                if self.state == JsonParseState::Complete && !ch.is_whitespace() {
                    self.state = JsonParseState::Error;
                }
            }
            JsonParseState::Start => {
                if ch.is_whitespace() {
                    return;
                }
                match ch {
                    '{' => {
                        self.depth += 1;
                        self.context_stack.push('o');
                        self.state = JsonParseState::InObject;
                        self.expecting_comma_or_close = false;
                    }
                    '[' => {
                        self.depth += 1;
                        self.context_stack.push('a');
                        self.state = JsonParseState::InArray;
                        self.expecting_comma_or_close = false;
                    }
                    '"' => {
                        self.state = JsonParseState::InString;
                    }
                    '-' | '0'..='9' => {
                        self.state = JsonParseState::InNumber;
                        self.keyword_buf.clear();
                        self.keyword_buf.push(ch);
                    }
                    't' | 'f' => {
                        self.state = JsonParseState::InBool;
                        self.keyword_buf.clear();
                        self.keyword_buf.push(ch);
                    }
                    'n' => {
                        self.state = JsonParseState::InNull;
                        self.keyword_buf.clear();
                        self.keyword_buf.push(ch);
                    }
                    _ => {
                        self.state = JsonParseState::Error;
                    }
                }
            }
            JsonParseState::InObject => {
                if ch.is_whitespace() {
                    return;
                }
                if self.expecting_comma_or_close {
                    match ch {
                        ',' => {
                            self.expecting_comma_or_close = false;
                        }
                        '}' => {
                            self.close_context();
                        }
                        _ => {
                            self.state = JsonParseState::Error;
                        }
                    }
                } else {
                    match ch {
                        '"' => {
                            self.state = JsonParseState::InObjectKey;
                        }
                        '}' => {
                            self.close_context();
                        }
                        _ => {
                            self.state = JsonParseState::Error;
                        }
                    }
                }
            }
            JsonParseState::InObjectKey => match ch {
                '"' => {
                    self.state = JsonParseState::AfterKey;
                }
                '\\' => {
                    self.state = JsonParseState::InStringEscape;
                }
                _ => {} // Any other char stays in key
            },
            JsonParseState::AfterKey => {
                if ch.is_whitespace() {
                    return;
                }
                if ch == ':' {
                    self.state = JsonParseState::InObjectValue;
                    self.expecting_comma_or_close = false;
                } else {
                    self.state = JsonParseState::Error;
                }
            }
            JsonParseState::InObjectValue => {
                if ch.is_whitespace() {
                    return;
                }
                self.start_value(ch, 'o');
            }
            JsonParseState::InArray => {
                if ch.is_whitespace() {
                    return;
                }
                if self.expecting_comma_or_close {
                    match ch {
                        ',' => {
                            self.expecting_comma_or_close = false;
                        }
                        ']' => {
                            self.close_context();
                        }
                        _ => {
                            self.state = JsonParseState::Error;
                        }
                    }
                } else {
                    match ch {
                        ']' => {
                            self.close_context();
                        }
                        _ => {
                            self.start_value(ch, 'a');
                        }
                    }
                }
            }
            JsonParseState::InArrayValue => {
                if ch.is_whitespace() {
                    return;
                }
                if self.expecting_comma_or_close {
                    if self.context_stack.last() == Some(&'a') {
                        match ch {
                            ',' => {
                                self.expecting_comma_or_close = false;
                                self.state = JsonParseState::InArray;
                            }
                            ']' => {
                                self.close_context();
                            }
                            _ => {
                                self.state = JsonParseState::Error;
                            }
                        }
                    } else {
                        match ch {
                            ',' => {
                                self.expecting_comma_or_close = false;
                                self.state = JsonParseState::InObject;
                            }
                            '}' => {
                                self.close_context();
                            }
                            _ => {
                                self.state = JsonParseState::Error;
                            }
                        }
                    }
                } else {
                    self.start_value(ch, *self.context_stack.last().unwrap_or(&'a'));
                }
            }
            JsonParseState::InString => match ch {
                '"' => {
                    self.finish_string();
                }
                '\\' => {
                    self.state = JsonParseState::InStringEscape;
                }
                _ => {} // Any other char stays in string
            },
            JsonParseState::InStringEscape => {
                // Accept any valid escape char; fall back to InString.
                self.state = JsonParseState::InString;
            }
            JsonParseState::InNumber => match ch {
                '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => {
                    self.keyword_buf.push(ch);
                }
                _ => {
                    // Number ended — treat `ch` as the next character after value.
                    self.finish_value();
                    self.feed_char(ch);
                }
            },
            JsonParseState::InBool => {
                self.keyword_buf.push(ch);
                let kb = self.keyword_buf.clone();
                if kb == "true" || kb == "false" {
                    self.keyword_buf.clear();
                    self.finish_value();
                } else if !"true".starts_with(kb.as_str()) && !"false".starts_with(kb.as_str()) {
                    self.state = JsonParseState::Error;
                }
            }
            JsonParseState::InNull => {
                self.keyword_buf.push(ch);
                let kb = self.keyword_buf.clone();
                if kb == "null" {
                    self.keyword_buf.clear();
                    self.finish_value();
                } else if !"null".starts_with(kb.as_str()) {
                    self.state = JsonParseState::Error;
                }
            }
        }
    }

    /// Begin parsing a new JSON value starting with `ch`.
    fn start_value(&mut self, ch: char, ctx: char) {
        match ch {
            '{' => {
                self.depth += 1;
                self.context_stack.push('o');
                self.state = JsonParseState::InObject;
                self.expecting_comma_or_close = false;
            }
            '[' => {
                self.depth += 1;
                self.context_stack.push('a');
                self.state = JsonParseState::InArray;
                self.expecting_comma_or_close = false;
            }
            '"' => {
                self.state = JsonParseState::InString;
            }
            '-' | '0'..='9' => {
                self.state = JsonParseState::InNumber;
                self.keyword_buf.clear();
                self.keyword_buf.push(ch);
                let _ = ctx; // context noted but not needed here
            }
            't' | 'f' => {
                self.state = JsonParseState::InBool;
                self.keyword_buf.clear();
                self.keyword_buf.push(ch);
            }
            'n' => {
                self.state = JsonParseState::InNull;
                self.keyword_buf.clear();
                self.keyword_buf.push(ch);
            }
            _ => {
                self.state = JsonParseState::Error;
            }
        }
    }

    /// A scalar value (string/number/bool/null) has been completed.
    fn finish_value(&mut self) {
        self.expecting_comma_or_close = true;
        match self.context_stack.last() {
            Some(&'o') => {
                self.state = JsonParseState::InObject;
            }
            Some(&'a') => {
                self.state = JsonParseState::InArray;
            }
            None => {
                self.state = JsonParseState::Complete;
            }
            _ => {
                self.state = JsonParseState::Error;
            }
        }
    }

    /// A `"` was seen — close the current string.
    fn finish_string(&mut self) {
        match self.context_stack.last() {
            Some(&'o') => {
                self.state = JsonParseState::InObject;
                self.expecting_comma_or_close = true;
            }
            Some(&'a') => {
                self.state = JsonParseState::InArray;
                self.expecting_comma_or_close = true;
            }
            None => {
                self.state = JsonParseState::Complete;
            }
            _ => {
                self.state = JsonParseState::Error;
            }
        }
    }

    /// Close the current object or array context.
    fn close_context(&mut self) {
        if let Some(ctx) = self.context_stack.pop() {
            if ctx == 'o' || ctx == 'a' {
                self.depth = self.depth.saturating_sub(1);
            }
        }
        self.expecting_comma_or_close = true;
        match self.context_stack.last() {
            Some(&'o') => {
                self.state = JsonParseState::InObject;
            }
            Some(&'a') => {
                self.state = JsonParseState::InArray;
            }
            None => {
                self.state = JsonParseState::Complete;
            }
            _ => {
                self.state = JsonParseState::Error;
            }
        }
    }

    /// Speculatively feed `chars` through a clone of the machine and report
    /// whether the whole sequence is accepted (never reaches [`JsonParseState::Error`]).
    ///
    /// An empty `chars` slice is reported as *not* accepted; empty tokens are
    /// handled separately (allowed only when the machine is already complete).
    fn would_accept(&self, chars: &[char]) -> bool {
        if self.state == JsonParseState::Error || chars.is_empty() {
            return false;
        }
        let mut probe = self.clone();
        for &ch in chars {
            probe.feed_char(ch);
            if probe.state == JsonParseState::Error {
                return false;
            }
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JsonConstraint
// ─────────────────────────────────────────────────────────────────────────────

/// Constrains generation to syntactically valid JSON.
///
/// Tracks nesting depth and parse state character by character.
///
/// See the [`constrained_decoding`](crate::constrained_decoding) module documentation for the difference between the real
/// [`with_decoder`](Self::with_decoder) mode and the demonstration-only
/// [`new`](Self::new) (token-id-as-codepoint) mode.
pub struct JsonConstraint {
    machine: JsonMachine,
    /// Precomputed token→text index.  `Some` in real mode
    /// ([`with_decoder`](Self::with_decoder)); `None` in demonstration/toy mode
    /// ([`new`](Self::new)).
    index: Option<TokenTextIndex>,
}

impl JsonConstraint {
    /// Create a new **demonstration / toy** `JsonConstraint`.
    ///
    /// # Warning: not for real tokenizers
    ///
    /// With no decoder, `allowed_tokens` and `advance` treat each raw token id as
    /// a Unicode code point (`char::from_u32(id)`).  This is only meaningful for a
    /// synthetic vocabulary where `token_id == codepoint` (e.g. an ASCII byte
    /// vocab used in unit tests).  For a real subword tokenizer it produces masks
    /// and accept/reject decisions unrelated to the generated text.
    ///
    /// Use [`JsonConstraint::with_decoder`] (or the byte-correct
    /// [`crate::grammar::GrammarConstraint`]) for production.
    pub fn new() -> Self {
        Self {
            machine: JsonMachine::new(),
            index: None,
        }
    }

    /// Create a real `JsonConstraint` driven by a token decode function.
    ///
    /// `decode_fn` maps a token id to the text it emits (`None` for
    /// EOS / padding / special tokens that emit no text).  It is invoked once per
    /// token id in `0..vocab_size` at construction time and is not retained.
    ///
    /// After construction, `allowed_tokens` masks exactly the tokens whose decoded
    /// text keeps the JSON parse state valid, and `advance` feeds a committed
    /// token's decoded text through the state machine — correct for any real
    /// tokenizer.
    ///
    /// ```rust
    /// use oxibonsai_runtime::constrained_decoding::{JsonConstraint, TokenConstraint};
    ///
    /// // Toy vocab: 0→"{", 1→"}", 2→"\"", 3→"true".
    /// let decode = |id: u32| match id {
    ///     0 => Some("{".to_string()),
    ///     1 => Some("}".to_string()),
    ///     2 => Some("\"".to_string()),
    ///     3 => Some("true".to_string()),
    ///     _ => None,
    /// };
    /// let mut c = JsonConstraint::with_decoder(decode, 4);
    /// let mask = c.allowed_tokens(&[], 4).unwrap();
    /// assert!(mask[0]); // "{" starts an object
    /// assert!(!mask[1]); // "}" cannot start a document
    /// ```
    pub fn with_decoder(decode_fn: impl Fn(u32) -> Option<String>, vocab_size: usize) -> Self {
        Self {
            machine: JsonMachine::new(),
            index: Some(TokenTextIndex::build(decode_fn, vocab_size)),
        }
    }

    /// Current parse state.
    pub fn current_state(&self) -> &JsonParseState {
        &self.machine.state
    }

    /// Current nesting depth.
    pub fn depth(&self) -> usize {
        self.machine.depth
    }

    /// Returns `true` if we are currently inside a string.
    pub fn is_in_string(&self) -> bool {
        self.machine.is_in_string()
    }

    /// Returns the set of ASCII characters that are valid as the *next* character
    /// given the current parse state.
    ///
    /// Note: in string-interior states this list omits non-ASCII characters that
    /// are nonetheless accepted; see the real [`with_decoder`](Self::with_decoder)
    /// path, which probes full token text rather than relying on this list.
    pub fn valid_next_chars(&self) -> Vec<char> {
        self.machine.valid_next_chars()
    }

    /// Real-mode mask: probe candidate tokens' decoded text against the machine.
    fn allowed_tokens_real(&self, index: &TokenTextIndex, vocab_size: usize) -> Vec<bool> {
        let mut mask = vec![false; vocab_size];

        if self.machine.state == JsonParseState::Error {
            return mask;
        }

        // Empty-text tokens (EOS / special): allowed only when the document is
        // already complete.
        if self.machine.state == JsonParseState::Complete {
            for &id in index.empty_token_ids() {
                if (id as usize) < vocab_size {
                    mask[id as usize] = true;
                }
            }
        }

        let probe_ids = |ids: &[u32], mask: &mut Vec<bool>| {
            for &id in ids {
                let idx = id as usize;
                if idx >= vocab_size || mask[idx] {
                    continue;
                }
                if self.machine.would_accept(index.token_chars(id)) {
                    mask[idx] = true;
                }
            }
        };

        if self.machine.accepts_arbitrary_char() {
            // The valid first-character set is unbounded (any char continues a
            // string / escape): every non-empty token must be probed.
            for (_first, ids) in index.first_char_groups() {
                probe_ids(ids, &mut mask);
            }
        } else {
            // Only tokens whose first character is currently valid can match.
            for ch in self.machine.valid_next_chars() {
                probe_ids(index.ids_with_first_char(ch), &mut mask);
            }
        }

        mask
    }

    /// Toy-mode mask: treat each token id as a Unicode code point.
    fn allowed_tokens_toy(&self, vocab_size: usize) -> Vec<bool> {
        if self.machine.state == JsonParseState::Error {
            return vec![false; vocab_size];
        }
        let valid = self.machine.valid_next_chars();
        (0..vocab_size)
            .map(|id| {
                let ch = char::from_u32(id as u32).unwrap_or('\u{FFFD}');
                // Allow if valid_next_chars contains it, or if the token is
                // non-ASCII (cannot tell without a vocab table — be conservative).
                ch as u32 > 127 || valid.contains(&ch)
            })
            .collect()
    }
}

impl Default for JsonConstraint {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenConstraint for JsonConstraint {
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        match &self.index {
            Some(index) => Some(self.allowed_tokens_real(index, vocab_size)),
            None => Some(self.allowed_tokens_toy(vocab_size)),
        }
    }

    fn advance(&mut self, token: u32) -> bool {
        if self.machine.state == JsonParseState::Error {
            return false;
        }
        match &self.index {
            Some(index) => {
                let chars = index.token_chars(token).to_vec();
                if chars.is_empty() {
                    // EOS / special / out-of-range: valid only when complete.
                    return self.machine.state == JsonParseState::Complete;
                }
                // Clone-and-commit: a rejected token leaves the live state intact
                // rather than stranding the machine in a partial `Error` state.
                let mut probe = self.machine.clone();
                for ch in chars {
                    probe.feed_char(ch);
                    if probe.state == JsonParseState::Error {
                        return false;
                    }
                }
                self.machine = probe;
                true
            }
            None => {
                // Toy mode: treat the token id as a code point.
                if let Some(ch) = char::from_u32(token) {
                    self.machine.feed_char(ch);
                }
                self.machine.state != JsonParseState::Error
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.machine.state == JsonParseState::Complete
    }

    fn reset(&mut self) {
        self.machine = JsonMachine::new();
    }

    fn name(&self) -> &str {
        "JsonConstraint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Toy / state-machine tests (token id == codepoint) ────────────────────

    #[test]
    fn json_constraint_initial_state() {
        let jc = JsonConstraint::new();
        assert_eq!(*jc.current_state(), JsonParseState::Start);
        assert_eq!(jc.depth(), 0);
    }

    #[test]
    fn json_constraint_valid_object_chars() {
        let jc = JsonConstraint::new();
        let valid = jc.valid_next_chars();
        assert!(valid.contains(&'{'));
        assert!(valid.contains(&'['));
        assert!(valid.contains(&'"'));
    }

    #[test]
    fn json_constraint_tracks_depth() {
        let mut jc = JsonConstraint::new();
        jc.advance('{' as u32);
        assert_eq!(jc.depth(), 1);
        jc.advance('"' as u32);
        jc.advance('k' as u32);
        jc.advance('"' as u32);
        jc.advance(':' as u32);
        jc.advance('{' as u32);
        assert_eq!(jc.depth(), 2);
        jc.advance('}' as u32);
        assert_eq!(jc.depth(), 1);
    }

    #[test]
    fn json_constraint_detects_completion() {
        let mut jc = JsonConstraint::new();
        assert!(!jc.is_complete());
        // Feed `{}`
        jc.advance('{' as u32);
        jc.advance('}' as u32);
        assert!(jc.is_complete());
    }

    #[test]
    fn json_constraint_in_string_state() {
        let mut jc = JsonConstraint::new();
        jc.advance('"' as u32);
        assert!(jc.is_in_string());
        jc.advance('"' as u32);
        assert!(!jc.is_in_string());
    }

    // ── Real decoder-driven tests (multi-char tokens, id != codepoint) ────────

    /// A small realistic tokenizer where ids bear NO relation to code points:
    /// multi-character tokens and out-of-order ids.
    fn json_vocab(id: u32) -> Option<String> {
        match id {
            10 => Some("{".to_string()),
            11 => Some("}".to_string()),
            12 => Some("[".to_string()),
            13 => Some("]".to_string()),
            14 => Some("\"".to_string()),
            15 => Some("key".to_string()),
            16 => Some(":".to_string()),
            17 => Some("true".to_string()),
            18 => Some("false".to_string()),
            19 => Some("123".to_string()),
            20 => Some(",".to_string()),
            21 => Some("\":".to_string()), // closing-quote + colon
            99 => None,                    // EOS
            _ => None,
        }
    }

    const JSON_VOCAB: usize = 100;

    #[test]
    fn json_decoder_masks_start_correctly() {
        let c = JsonConstraint::with_decoder(json_vocab, JSON_VOCAB);
        let mask = c.allowed_tokens(&[], JSON_VOCAB).unwrap();
        assert!(mask[10], "'{{' should open a document");
        assert!(mask[12], "'[' should open an array");
        assert!(mask[14], "'\"' should open a string");
        assert!(mask[17], "'true' is a valid document");
        assert!(mask[19], "'123' is a valid document");
        // These cannot start a JSON document.
        assert!(!mask[11], "'}}' cannot start a document");
        assert!(!mask[13], "']' cannot start a document");
        assert!(!mask[16], "':' cannot start a document");
        assert!(!mask[20], "',' cannot start a document");
        // EOS not allowed at start (not complete).
        assert!(!mask[99], "EOS not allowed before any value");
    }

    #[test]
    fn json_decoder_multichar_token_advances_state() {
        let mut c = JsonConstraint::with_decoder(json_vocab, JSON_VOCAB);
        // Build {"key":true}
        assert!(c.advance(10), "'{{'");
        assert_eq!(c.depth(), 1);
        assert_eq!(*c.current_state(), JsonParseState::InObject);
        assert!(c.advance(14), "'\"' opens key");
        assert_eq!(*c.current_state(), JsonParseState::InObjectKey);
        assert!(c.advance(15), "'key' multichar token inside key string");
        assert_eq!(
            *c.current_state(),
            JsonParseState::InObjectKey,
            "still parsing the key after 'key'"
        );
        assert!(c.advance(21), "'\":' closes key and starts value");
        assert!(c.advance(17), "'true' value");
        assert!(c.advance(11), "'}}' closes object");
        assert!(c.is_complete(), "{{\"key\":true}} is complete");
        assert_eq!(c.depth(), 0);
    }

    #[test]
    fn json_decoder_rejects_invalid_multichar() {
        let mut c = JsonConstraint::with_decoder(json_vocab, JSON_VOCAB);
        assert!(c.advance(10), "'{{'");
        // After '{' only a key-opening '"' or '}' is valid; 'true' (17) is not.
        let mask = c.allowed_tokens(&[], JSON_VOCAB).unwrap();
        assert!(!mask[17], "'true' cannot follow '{{'");
        assert!(mask[14], "'\"' can follow '{{'");
        assert!(mask[11], "'}}' can close empty object");
        // advance with the invalid token must be rejected.
        assert!(!c.advance(17), "advancing 'true' after '{{' must fail");
    }

    #[test]
    fn json_decoder_eos_only_when_complete() {
        let mut c = JsonConstraint::with_decoder(json_vocab, JSON_VOCAB);
        // Not complete initially → EOS blocked.
        let mask = c.allowed_tokens(&[], JSON_VOCAB).unwrap();
        assert!(!mask[99]);
        // Complete a value.
        assert!(c.advance(17), "'true'");
        assert!(c.is_complete());
        let mask = c.allowed_tokens(&[], JSON_VOCAB).unwrap();
        assert!(mask[99], "EOS allowed once the document is complete");
    }

    #[test]
    fn json_decoder_string_interior_allows_arbitrary_text() {
        // A token that decodes to arbitrary text must be allowed inside a string.
        let decode = |id: u32| match id {
            0 => Some("\"".to_string()),
            1 => Some("hello world!".to_string()),
            2 => Some("café ☕".to_string()), // non-ASCII inside string
            3 => Some("\"".to_string()),
            _ => None,
        };
        let mut c = JsonConstraint::with_decoder(decode, 4);
        assert!(c.advance(0), "open string");
        assert!(c.is_in_string());
        let mask = c.allowed_tokens(&[], 4).unwrap();
        assert!(mask[1], "arbitrary ASCII text allowed in string");
        assert!(mask[2], "non-ASCII text allowed in string");
        assert!(c.advance(2), "commit non-ASCII text");
        assert!(c.is_in_string(), "still in string");
        assert!(c.advance(3), "close string");
        assert!(c.is_complete(), "\"...\" is a complete document");
    }

    #[test]
    fn json_decoder_reset_restores_initial_state() {
        let mut c = JsonConstraint::with_decoder(json_vocab, JSON_VOCAB);
        assert!(c.advance(10));
        assert_eq!(c.depth(), 1);
        c.reset();
        assert_eq!(c.depth(), 0);
        assert_eq!(*c.current_state(), JsonParseState::Start);
        assert!(!c.is_complete());
    }
}
