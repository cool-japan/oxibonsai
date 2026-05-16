//! JSON-grammar [`TokenConstraint`] implementation.
//!
//! Hosts [`JsonParseState`] and the [`JsonConstraint`] state machine that
//! restricts generation to syntactically valid JSON.

use super::error_trait::TokenConstraint;

// ─────────────────────────────────────────────────────────────────────────────
// JsonConstraint
// ─────────────────────────────────────────────────────────────────────────────

/// Internal parser state for `JsonConstraint`.
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

/// Constrains generation to syntactically valid JSON.
///
/// Tracks nesting depth and parse state character by character.
pub struct JsonConstraint {
    state: JsonParseState,
    depth: usize,
    buffer: String,
    expecting_comma_or_close: bool,
    // For keyword tracking (true/false/null).
    keyword_buf: String,
    // Stack of context: 'o' = object, 'a' = array.
    context_stack: Vec<char>,
}

impl JsonConstraint {
    /// Create a new `JsonConstraint` in its initial state.
    pub fn new() -> Self {
        Self {
            state: JsonParseState::Start,
            depth: 0,
            buffer: String::new(),
            expecting_comma_or_close: false,
            keyword_buf: String::new(),
            context_stack: Vec::new(),
        }
    }

    /// Current parse state.
    pub fn current_state(&self) -> &JsonParseState {
        &self.state
    }

    /// Current nesting depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns `true` if we are currently inside a string.
    pub fn is_in_string(&self) -> bool {
        matches!(
            self.state,
            JsonParseState::InString | JsonParseState::InStringEscape
        )
    }

    /// Returns the set of ASCII characters that are valid as the *next* character
    /// given the current parse state.
    pub fn valid_next_chars(&self) -> Vec<char> {
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
                return;
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
            JsonParseState::InObjectKey => {
                match ch {
                    '"' => {
                        self.state = JsonParseState::AfterKey;
                    }
                    '\\' => {
                        self.state = JsonParseState::InStringEscape;
                    }
                    _ => {} // Any other char stays in key
                }
            }
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
            JsonParseState::InString => {
                match ch {
                    '"' => {
                        self.finish_string();
                    }
                    '\\' => {
                        self.state = JsonParseState::InStringEscape;
                    }
                    _ => {} // Any other char stays in string
                }
            }
            JsonParseState::InStringEscape => {
                // Accept any valid escape char; fall back to InString.
                self.state = JsonParseState::InString;
            }
            JsonParseState::InNumber => {
                match ch {
                    '0'..='9' | '.' | 'e' | 'E' | '+' | '-' => {
                        self.keyword_buf.push(ch);
                    }
                    _ => {
                        // Number ended — treat `ch` as the next character after value.
                        self.finish_value();
                        self.feed_char(ch);
                    }
                }
            }
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
        self.buffer.push(ch);
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
}

impl Default for JsonConstraint {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenConstraint for JsonConstraint {
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        if self.state == JsonParseState::Error {
            return Some(vec![false; vocab_size]);
        }
        // Conservative: for each token id in [0, vocab_size) check if its first
        // ASCII character (treating the id as codepoint) is in valid_next_chars.
        let valid = self.valid_next_chars();
        let mask: Vec<bool> = (0..vocab_size)
            .map(|id| {
                // Map token id to a char for a simplified single-char check.
                let ch = char::from_u32(id as u32).unwrap_or('\u{FFFD}');
                // Allow if valid_next_chars contains it, or if the token is non-ASCII
                // (we can't tell without a vocab table — be conservative and allow).
                ch as u32 > 127 || valid.contains(&ch)
            })
            .collect();
        Some(mask)
    }

    fn advance(&mut self, token: u32) -> bool {
        if self.state == JsonParseState::Error {
            return false;
        }
        // Treat token id as a codepoint.
        if let Some(ch) = char::from_u32(token) {
            self.feed_char(ch);
        }
        self.state != JsonParseState::Error
    }

    fn is_complete(&self) -> bool {
        self.state == JsonParseState::Complete
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn name(&self) -> &str {
        "JsonConstraint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
