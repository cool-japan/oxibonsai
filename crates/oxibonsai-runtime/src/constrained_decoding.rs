//! Grammar-constrained decoding for token-by-token generation.
//!
//! This module provides the [`TokenConstraint`] trait and concrete implementations
//! that restrict which tokens the model can emit at each decoding step:
//!
//! - [`NoConstraint`] — passthrough, all tokens allowed
//! - [`RegexConstraint`] — restricts output to strings matching a regex pattern
//! - [`JsonConstraint`] — restricts output to syntactically valid JSON
//!
//! The [`ConstrainedSampler`] wraps a [`crate::sampling_advanced::SamplerChain`] and
//! applies a mask to logits before sampling so that only valid continuations are drawn.
//!
//! ## Example
//! ```rust
//! use oxibonsai_runtime::constrained_decoding::{ConstrainedSamplerBuilder, TokenConstraint};
//!
//! let mut sampler = ConstrainedSamplerBuilder::new(128, 42)
//!     .with_json_constraint();
//! assert!(!sampler.is_complete());
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise when building or running a token constraint.
#[derive(Debug, thiserror::Error)]
pub enum ConstraintError {
    /// The supplied regex pattern was syntactically invalid.
    #[error("Invalid regex pattern: {0}")]
    InvalidPattern(String),

    /// The supplied JSON schema was invalid (reserved for future schema-based constraints).
    #[error("Invalid JSON schema: {0}")]
    InvalidSchema(String),

    /// The constraint was violated at a specific token.
    #[error("Constraint violated at token {token}: {reason}")]
    Violated { token: u32, reason: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that restricts which tokens are valid at each decoding step.
///
/// Implementors maintain internal state representing how far through the
/// constrained structure the generation has progressed.
pub trait TokenConstraint: Send + Sync {
    /// Given the tokens generated so far, return a bitmask of allowed next tokens.
    ///
    /// `vocab_size` is the total vocabulary size.  Returns `None` if all tokens
    /// are allowed (no active constraint).
    fn allowed_tokens(&self, generated: &[u32], vocab_size: usize) -> Option<Vec<bool>>;

    /// Called after a token is committed.
    ///
    /// Returns `false` if the constraint is now violated (generation should stop).
    fn advance(&mut self, token: u32) -> bool;

    /// Returns `true` if the current state is a valid terminal state.
    fn is_complete(&self) -> bool;

    /// Reset the constraint to its initial state.
    fn reset(&mut self);

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────────────
// NoConstraint — passthrough
// ─────────────────────────────────────────────────────────────────────────────

/// A passthrough constraint that places no restriction on the vocabulary.
pub struct NoConstraint;

impl TokenConstraint for NoConstraint {
    fn allowed_tokens(&self, _generated: &[u32], _vocab_size: usize) -> Option<Vec<bool>> {
        None
    }

    fn advance(&mut self, _token: u32) -> bool {
        true
    }

    fn is_complete(&self) -> bool {
        true
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "NoConstraint"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal NFA-based regex engine
// ─────────────────────────────────────────────────────────────────────────────

/// One NFA state.
#[derive(Debug, Clone)]
enum NfaState {
    /// Matches a specific character then transitions to `next`.
    Literal(char, usize),
    /// Matches any character then transitions to `next`.
    Any(usize),
    /// ε-transition fork (used for `|`, `?`, `*`, `+`).
    Split(usize, usize),
    /// Character class `[...]`.  `negated` inverts the match.
    Class {
        chars: Vec<char>,
        ranges: Vec<(char, char)>,
        negated: bool,
        next: usize,
    },
    /// The accepting state.
    Accept,
}

/// Simple NFA compiled from a regex pattern.
#[derive(Debug, Clone)]
struct RegexNfa {
    states: Vec<NfaState>,
    start: usize,
    accept_state: usize,
}

/// A fragment of NFA states returned by the compiler — holds start index and
/// a list of "dangling" out-arrows that must be patched to the next fragment.
struct Fragment {
    start: usize,
    /// Indices of states whose outgoing arrow is "open" (needs patching).
    outs: Vec<usize>,
}

impl RegexNfa {
    /// Build an NFA from a regex pattern.
    fn from_pattern(pattern: &str) -> Result<Self, ConstraintError> {
        let mut nfa = RegexNfa {
            states: Vec::new(),
            start: 0,
            accept_state: 0,
        };
        let chars: Vec<char> = pattern.chars().collect();
        let frag = nfa
            .compile(&chars, 0)
            .map_err(ConstraintError::InvalidPattern)?;
        // Add accept state.
        let accept = nfa.push(NfaState::Accept);
        nfa.accept_state = accept;
        nfa.patch(&frag.outs, accept);
        nfa.start = frag.start;
        Ok(nfa)
    }

    fn push(&mut self, state: NfaState) -> usize {
        let idx = self.states.len();
        self.states.push(state);
        idx
    }

    /// Patch all dangling out-arrows in `outs` to point to `target`.
    fn patch(&mut self, outs: &[usize], target: usize) {
        for &idx in outs {
            match &mut self.states[idx] {
                NfaState::Literal(_, ref mut n)
                | NfaState::Any(ref mut n)
                | NfaState::Class {
                    next: ref mut n, ..
                } => *n = target,
                NfaState::Split(ref mut a, ref mut b) => {
                    // Patch every open slot (usize::MAX means "unset").
                    if *a == usize::MAX {
                        *a = target;
                    }
                    if *b == usize::MAX {
                        *b = target;
                    }
                }
                NfaState::Accept => {}
            }
        }
    }

    /// Recursive-descent compiler; returns a Fragment.
    fn compile(&mut self, chars: &[char], mut pos: usize) -> Result<Fragment, String> {
        // Parse a sequence of alternation alternatives: e1 | e2 | ...
        let mut alt_frags: Vec<Fragment> = Vec::new();
        let mut cur_frags: Vec<Fragment> = Vec::new();

        while pos < chars.len() {
            let ch = chars[pos];

            // Handle alternation `|`
            if ch == '|' {
                let seq = Self::concat_fragments(&mut self.states, cur_frags);
                alt_frags.push(seq);
                cur_frags = Vec::new();
                pos += 1;
                continue;
            }

            // End of group
            if ch == ')' {
                break;
            }

            // Parse one atom (possibly followed by a quantifier)
            let (atom, new_pos) = self.parse_atom(chars, pos)?;
            pos = new_pos;

            // Check for quantifier
            let quantified = if pos < chars.len() {
                match chars[pos] {
                    '?' => {
                        pos += 1;
                        self.quantifier_optional(atom)
                    }
                    '*' => {
                        pos += 1;
                        self.quantifier_star(atom)
                    }
                    '+' => {
                        pos += 1;
                        self.quantifier_plus(atom)
                    }
                    _ => atom,
                }
            } else {
                atom
            };

            cur_frags.push(quantified);
        }

        // Concatenate remaining sequence
        let seq = Self::concat_fragments(&mut self.states, cur_frags);
        alt_frags.push(seq);

        // Build alternation if needed
        let result = if alt_frags.len() == 1 {
            alt_frags.remove(0)
        } else {
            self.alternation(alt_frags)
        };

        Ok(result)
    }

    /// Parse one atom starting at `pos`, return (Fragment, new_pos).
    fn parse_atom(&mut self, chars: &[char], pos: usize) -> Result<(Fragment, usize), String> {
        if pos >= chars.len() {
            return Err("Unexpected end of pattern".to_string());
        }
        let ch = chars[pos];
        match ch {
            '(' => {
                // Grouped sub-expression
                let inner = self.compile(chars, pos + 1)?;
                // Find matching ')'
                let mut depth = 1usize;
                let mut i = pos + 1;
                while i < chars.len() {
                    match chars[i] {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        '\\' => {
                            i += 1;
                        } // skip escaped
                        _ => {}
                    }
                    i += 1;
                }
                let new_pos = if i < chars.len() && chars[i] == ')' {
                    i + 1
                } else {
                    i
                };
                Ok((inner, new_pos))
            }
            '[' => {
                let (frag, new_pos) = self.parse_class(chars, pos)?;
                Ok((frag, new_pos))
            }
            '.' => {
                let idx = self.push(NfaState::Any(usize::MAX));
                Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 1,
                ))
            }
            '\\' => {
                let (frag, new_pos) = self.parse_escape(chars, pos)?;
                Ok((frag, new_pos))
            }
            _ if ch == '*' || ch == '+' || ch == '?' => {
                Err(format!("Unexpected quantifier '{ch}' at position {pos}"))
            }
            _ => {
                let idx = self.push(NfaState::Literal(ch, usize::MAX));
                Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 1,
                ))
            }
        }
    }

    /// Parse a character class `[...]`.
    fn parse_class(&mut self, chars: &[char], start: usize) -> Result<(Fragment, usize), String> {
        // start points to '['
        let mut pos = start + 1;
        let negated = if pos < chars.len() && chars[pos] == '^' {
            pos += 1;
            true
        } else {
            false
        };

        let mut class_chars: Vec<char> = Vec::new();
        let mut ranges: Vec<(char, char)> = Vec::new();

        while pos < chars.len() && chars[pos] != ']' {
            if chars[pos] == '\\' && pos + 1 < chars.len() {
                // Escape inside class
                let escaped = chars[pos + 1];
                match escaped {
                    'd' => ranges.push(('0', '9')),
                    'w' => {
                        ranges.push(('a', 'z'));
                        ranges.push(('A', 'Z'));
                        ranges.push(('0', '9'));
                        class_chars.push('_');
                    }
                    's' => {
                        class_chars.extend_from_slice(&[' ', '\t', '\n', '\r']);
                    }
                    _ => class_chars.push(escaped),
                }
                pos += 2;
            } else if pos + 2 < chars.len() && chars[pos + 1] == '-' && chars[pos + 2] != ']' {
                ranges.push((chars[pos], chars[pos + 2]));
                pos += 3;
            } else {
                class_chars.push(chars[pos]);
                pos += 1;
            }
        }

        let new_pos = if pos < chars.len() && chars[pos] == ']' {
            pos + 1
        } else {
            pos
        };

        let idx = self.push(NfaState::Class {
            chars: class_chars,
            ranges,
            negated,
            next: usize::MAX,
        });
        Ok((
            Fragment {
                start: idx,
                outs: vec![idx],
            },
            new_pos,
        ))
    }

    /// Parse a backslash escape at `pos` (e.g., `\d`, `\w`, `\s`).
    fn parse_escape(&mut self, chars: &[char], pos: usize) -> Result<(Fragment, usize), String> {
        if pos + 1 >= chars.len() {
            return Err("Trailing backslash in pattern".to_string());
        }
        let escaped = chars[pos + 1];
        let (class_chars, ranges): (Vec<char>, Vec<(char, char)>) = match escaped {
            'd' => (vec![], vec![('0', '9')]),
            'D' => {
                // non-digit — represented as negated class [^0-9]
                let idx = self.push(NfaState::Class {
                    chars: vec![],
                    ranges: vec![('0', '9')],
                    negated: true,
                    next: usize::MAX,
                });
                return Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 2,
                ));
            }
            'w' => (vec!['_'], vec![('a', 'z'), ('A', 'Z'), ('0', '9')]),
            'W' => {
                let idx = self.push(NfaState::Class {
                    chars: vec!['_'],
                    ranges: vec![('a', 'z'), ('A', 'Z'), ('0', '9')],
                    negated: true,
                    next: usize::MAX,
                });
                return Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 2,
                ));
            }
            's' => (vec![' ', '\t', '\n', '\r'], vec![]),
            'S' => {
                let idx = self.push(NfaState::Class {
                    chars: vec![' ', '\t', '\n', '\r'],
                    ranges: vec![],
                    negated: true,
                    next: usize::MAX,
                });
                return Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 2,
                ));
            }
            'n' => {
                let idx = self.push(NfaState::Literal('\n', usize::MAX));
                return Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 2,
                ));
            }
            'r' => {
                let idx = self.push(NfaState::Literal('\r', usize::MAX));
                return Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 2,
                ));
            }
            't' => {
                let idx = self.push(NfaState::Literal('\t', usize::MAX));
                return Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 2,
                ));
            }
            _ => {
                // Treat as literal escape (e.g., `\.`)
                let idx = self.push(NfaState::Literal(escaped, usize::MAX));
                return Ok((
                    Fragment {
                        start: idx,
                        outs: vec![idx],
                    },
                    pos + 2,
                ));
            }
        };
        let idx = self.push(NfaState::Class {
            chars: class_chars,
            ranges,
            negated: false,
            next: usize::MAX,
        });
        Ok((
            Fragment {
                start: idx,
                outs: vec![idx],
            },
            pos + 2,
        ))
    }

    // ── Quantifiers ──────────────────────────────────────────────────────────

    /// `e?` — zero or one.
    fn quantifier_optional(&mut self, frag: Fragment) -> Fragment {
        let split = self.push(NfaState::Split(frag.start, usize::MAX));
        let mut outs = frag.outs;
        outs.push(split); // the second arm of Split is still open
        Fragment { start: split, outs }
    }

    /// `e*` — zero or more.
    fn quantifier_star(&mut self, frag: Fragment) -> Fragment {
        let split = self.push(NfaState::Split(frag.start, usize::MAX));
        // Patch all fragment outs back to the split (loop).
        self.patch(&frag.outs, split);
        Fragment {
            start: split,
            outs: vec![split],
        }
    }

    /// `e+` — one or more.
    fn quantifier_plus(&mut self, frag: Fragment) -> Fragment {
        let split = self.push(NfaState::Split(frag.start, usize::MAX));
        self.patch(&frag.outs, split);
        Fragment {
            start: frag.start,
            outs: vec![split],
        }
    }

    /// Build alternation from multiple fragments (`e1 | e2 | ...`).
    fn alternation(&mut self, frags: Vec<Fragment>) -> Fragment {
        if frags.is_empty() {
            let split = self.push(NfaState::Split(usize::MAX, usize::MAX));
            return Fragment {
                start: split,
                outs: vec![split],
            };
        }
        let mut iter = frags.into_iter();
        let mut current = iter.next().expect("non-empty checked above");
        for next_frag in iter {
            let split = self.push(NfaState::Split(current.start, next_frag.start));
            let mut outs = current.outs;
            outs.extend(next_frag.outs);
            current = Fragment { start: split, outs };
        }
        current
    }

    /// Concatenate a sequence of fragments into one.
    fn concat_fragments(states: &mut Vec<NfaState>, frags: Vec<Fragment>) -> Fragment {
        if frags.is_empty() {
            // ε-fragment: a split pointing nowhere used as a placeholder
            let idx = states.len();
            states.push(NfaState::Split(usize::MAX, usize::MAX));
            return Fragment {
                start: idx,
                outs: vec![idx],
            };
        }
        let mut iter = frags.into_iter();
        let first = iter.next().expect("non-empty checked above");
        iter.fold(first, |acc, next| {
            // Patch all open outs of acc to point to start of next
            for &idx in &acc.outs {
                match &mut states[idx] {
                    NfaState::Literal(_, ref mut n)
                    | NfaState::Any(ref mut n)
                    | NfaState::Class {
                        next: ref mut n, ..
                    } => {
                        if *n == usize::MAX {
                            *n = next.start;
                        }
                    }
                    NfaState::Split(ref mut a, ref mut b) => {
                        if *a == usize::MAX {
                            *a = next.start;
                        } else if *b == usize::MAX {
                            *b = next.start;
                        }
                    }
                    NfaState::Accept => {}
                }
            }
            Fragment {
                start: acc.start,
                outs: next.outs,
            }
        })
    }

    // ── Simulation ───────────────────────────────────────────────────────────

    /// Compute the ε-closure of a set of states.
    fn epsilon_closure(&self, states: Vec<usize>) -> Vec<usize> {
        let mut closure: Vec<usize> = Vec::new();
        let mut stack = states;
        let mut visited = std::collections::HashSet::new();
        while let Some(s) = stack.pop() {
            if s == usize::MAX || !visited.insert(s) {
                continue;
            }
            closure.push(s);
            if let Some(NfaState::Split(a, b)) = self.states.get(s) {
                if *a != usize::MAX {
                    stack.push(*a);
                }
                if *b != usize::MAX {
                    stack.push(*b);
                }
            }
        }
        closure
    }

    /// Advance the NFA by consuming character `ch` from state set `states`.
    fn step(&self, states: &[usize], ch: char) -> Vec<usize> {
        let mut next = Vec::new();
        for &s in states {
            if s == usize::MAX {
                continue;
            }
            if let Some(state) = self.states.get(s) {
                match state {
                    NfaState::Literal(c, n) => {
                        if *c == ch && *n != usize::MAX {
                            next.push(*n);
                        }
                    }
                    NfaState::Any(n) => {
                        if *n != usize::MAX {
                            next.push(*n);
                        }
                    }
                    NfaState::Class {
                        chars,
                        ranges,
                        negated,
                        next: n,
                    } => {
                        let matched = chars.contains(&ch)
                            || ranges.iter().any(|&(lo, hi)| ch >= lo && ch <= hi);
                        let effective = if *negated { !matched } else { matched };
                        if effective && *n != usize::MAX {
                            next.push(*n);
                        }
                    }
                    NfaState::Split(_, _) | NfaState::Accept => {}
                }
            }
        }
        self.epsilon_closure(next)
    }

    /// Returns `true` if any of `states` is the accept state.
    fn is_accepting(&self, states: &[usize]) -> bool {
        states.contains(&self.accept_state)
    }

    /// Check whether `text` is fully matched by the NFA.
    fn is_full_match(&self, text: &str) -> bool {
        let initial = self.epsilon_closure(vec![self.start]);
        let final_states = text.chars().fold(initial, |s, ch| self.step(&s, ch));
        self.is_accepting(&final_states)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RegexConstraint
// ─────────────────────────────────────────────────────────────────────────────

/// Constrains generation to strings that match a regular expression.
///
/// Uses a minimal NFA engine (no external crate). Supported syntax:
/// - Literals, `.` (any char), `*`, `+`, `?`
/// - Alternation `|`
/// - Grouping `(...)`
/// - Character classes `[abc]`, `[a-z]`, `[^x]`
/// - Escapes: `\d`, `\D`, `\w`, `\W`, `\s`, `\S`, `\n`, `\r`, `\t`
pub struct RegexConstraint {
    pattern: String,
    nfa: RegexNfa,
    current_states: Vec<usize>,
    matched_so_far: String,
}

impl RegexConstraint {
    /// Build a new constraint from `pattern`.
    pub fn new(pattern: &str) -> Result<Self, ConstraintError> {
        let nfa = RegexNfa::from_pattern(pattern)?;
        let current_states = nfa.epsilon_closure(vec![nfa.start]);
        Ok(Self {
            pattern: pattern.to_string(),
            nfa,
            current_states,
            matched_so_far: String::new(),
        })
    }

    /// Test whether `text` fully matches `pattern`.
    pub fn is_match(pattern: &str, text: &str) -> bool {
        match RegexNfa::from_pattern(pattern) {
            Ok(nfa) => nfa.is_full_match(text),
            Err(_) => false,
        }
    }

    /// The text matched so far.
    pub fn current_partial(&self) -> &str {
        &self.matched_so_far
    }

    /// Check whether character `ch` would keep the NFA in a live (non-dead) state.
    pub fn char_is_valid(&self, ch: char) -> bool {
        let next = self.nfa.step(&self.current_states, ch);
        !next.is_empty()
    }
}

impl TokenConstraint for RegexConstraint {
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        // If already in a dead state, nothing is allowed.
        if self.current_states.is_empty() {
            return Some(vec![false; vocab_size]);
        }
        // We cannot map token ids to characters without a real vocabulary table,
        // so we return None (allow all) as a safe conservative choice.
        // The constraint is enforced via `advance` which rejects invalid tokens.
        None
    }

    fn advance(&mut self, token: u32) -> bool {
        // Treat the token id as a codepoint for demonstration purposes.
        // In a real integration the caller would pass token bytes/text.
        let ch = char::from_u32(token).unwrap_or('\u{FFFD}');
        let next = self.nfa.step(&self.current_states, ch);
        if next.is_empty() {
            return false;
        }
        self.current_states = next;
        self.matched_so_far.push(ch);
        true
    }

    fn is_complete(&self) -> bool {
        self.nfa.is_accepting(&self.current_states)
    }

    fn reset(&mut self) {
        self.current_states = self.nfa.epsilon_closure(vec![self.nfa.start]);
        self.matched_so_far.clear();
    }

    fn name(&self) -> &str {
        &self.pattern
    }
}

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

// ─────────────────────────────────────────────────────────────────────────────
// ConstrainedSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`crate::sampling_advanced::SamplerChain`] with a [`TokenConstraint`].
///
/// Before each sampling step the logits for disallowed tokens are masked to
/// `-1e9` so they are effectively excluded from the distribution.
pub struct ConstrainedSampler {
    inner: crate::sampling_advanced::SamplerChain,
    constraint: Box<dyn TokenConstraint>,
    generated: Vec<u32>,
    vocab_size: usize,
}

impl ConstrainedSampler {
    /// Create a new `ConstrainedSampler`.
    pub fn new(
        sampler: crate::sampling_advanced::SamplerChain,
        constraint: Box<dyn TokenConstraint>,
        vocab_size: usize,
    ) -> Self {
        Self {
            inner: sampler,
            constraint,
            generated: Vec::new(),
            vocab_size,
        }
    }

    /// Sample the next token, masking logits for disallowed tokens first.
    ///
    /// Steps:
    /// 1. Query the constraint for an allowed-token mask.
    /// 2. Set `logits[i] = -1e9` for every `false` entry in the mask.
    /// 3. Delegate to the inner sampler chain.
    /// 4. Call `constraint.advance(token)`.
    /// 5. Track the token in `self.generated`.
    pub fn sample(&mut self, logits: &mut Vec<f32>) -> u32 {
        // Apply constraint mask.
        if let Some(mask) = self
            .constraint
            .allowed_tokens(&self.generated, self.vocab_size)
        {
            for (i, allowed) in mask.iter().enumerate() {
                if i < logits.len() && !allowed {
                    logits[i] = -1e9;
                }
            }
        }
        let token = self.inner.sample(logits) as u32;
        self.constraint.advance(token);
        self.generated.push(token);
        token
    }

    /// Returns `true` if the constraint considers the current output complete.
    pub fn is_complete(&self) -> bool {
        self.constraint.is_complete()
    }

    /// Reset both the inner sampler state and the constraint.
    pub fn reset(&mut self) {
        self.generated.clear();
        self.constraint.reset();
    }

    /// Number of tokens generated so far.
    pub fn generated_text_len(&self) -> usize {
        self.generated.len()
    }

    /// Human-readable name of the active constraint.
    pub fn constraint_name(&self) -> &str {
        self.constraint.name()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstrainedSamplerBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Ergonomic builder for [`ConstrainedSampler`].
pub struct ConstrainedSamplerBuilder {
    vocab_size: usize,
    seed: u64,
}

impl ConstrainedSamplerBuilder {
    /// Create a new builder.
    pub fn new(vocab_size: usize, seed: u64) -> Self {
        Self { vocab_size, seed }
    }

    fn default_chain(&self) -> crate::sampling_advanced::SamplerChain {
        crate::sampling_advanced::SamplerChain::new(self.seed)
    }

    /// Build a `ConstrainedSampler` with a `JsonConstraint`.
    pub fn with_json_constraint(self) -> ConstrainedSampler {
        ConstrainedSampler::new(
            self.default_chain(),
            Box::new(JsonConstraint::new()),
            self.vocab_size,
        )
    }

    /// Build a `ConstrainedSampler` with a `RegexConstraint`.
    pub fn with_regex_constraint(
        self,
        pattern: &str,
    ) -> Result<ConstrainedSampler, ConstraintError> {
        let constraint = RegexConstraint::new(pattern)?;
        let chain = self.default_chain();
        Ok(ConstrainedSampler::new(
            chain,
            Box::new(constraint),
            self.vocab_size,
        ))
    }

    /// Build an unconstrained `ConstrainedSampler` (passthrough).
    pub fn unconstrained(self) -> ConstrainedSampler {
        ConstrainedSampler::new(
            self.default_chain(),
            Box::new(NoConstraint),
            self.vocab_size,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AllowListConstraint — force output to be one of a finite set of sequences
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that forces the generated token sequence to exactly match one
/// of a finite set of allowed token-id sequences (e.g., multiple-choice answers).
///
/// At each step only the union of next tokens across all still-active candidates
/// is permitted.  A candidate becomes inactive the moment any token in the prefix
/// fails to match.  Generation is considered complete once the full token sequence
/// of at least one candidate has been consumed.
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::constrained_decoding::{AllowListConstraint, TokenConstraint};
///
/// // Two candidates: [10, 20] and [10, 30]
/// let mut c = AllowListConstraint::new(vec![vec![10, 20], vec![10, 30]]);
/// // First token: only 10 is allowed (shared prefix)
/// let mask = c.allowed_tokens(&[], 50).unwrap();
/// assert!(mask[10]);
/// assert!(!mask[20]);
/// assert!(!mask[30]);
/// ```
pub struct AllowListConstraint {
    /// All allowed sequences.
    candidates: Vec<Vec<u32>>,
    /// Which candidates still match the current prefix.
    active: Vec<bool>,
    /// Number of tokens consumed so far.
    position: usize,
}

impl AllowListConstraint {
    /// Create a new `AllowListConstraint` from a list of allowed token sequences.
    pub fn new(candidates: Vec<Vec<u32>>) -> Self {
        let n = candidates.len();
        Self {
            candidates,
            active: vec![true; n],
            position: 0,
        }
    }

    /// Returns the number of candidate sequences that are still active.
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }
}

impl TokenConstraint for AllowListConstraint {
    /// Returns a bitmask of tokens that are valid next tokens across all still-active
    /// candidates at the current position.  Always returns `Some` (never unconstrained).
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        let mut mask = vec![false; vocab_size];
        for (i, active) in self.active.iter().enumerate() {
            if !active {
                continue;
            }
            let seq = &self.candidates[i];
            if self.position < seq.len() {
                let tok = seq[self.position] as usize;
                if tok < vocab_size {
                    mask[tok] = true;
                }
            }
        }
        Some(mask)
    }

    /// Commits `token` at the current position.
    ///
    /// Any candidate where `candidates[i][position] != token` (or the candidate is
    /// already exhausted) is deactivated.  Returns `true` when at least one candidate
    /// remains active **or** a candidate was just completed at this position.
    fn advance(&mut self, token: u32) -> bool {
        let mut just_completed = false;
        for (i, active) in self.active.iter_mut().enumerate() {
            if !*active {
                continue;
            }
            let seq = &self.candidates[i];
            if self.position >= seq.len() {
                // Candidate was already completed; further tokens deactivate it.
                *active = false;
            } else if seq[self.position] == token {
                // Token matches; check if this completes the candidate.
                if self.position + 1 == seq.len() {
                    just_completed = true;
                }
                // Keep active — will be filtered by is_complete / future advance calls.
            } else {
                *active = false;
            }
        }
        self.position += 1;
        // Return true if at least one candidate is still active or one just completed.
        just_completed || self.active.iter().any(|&a| a)
    }

    /// Returns `true` when the consumed token sequence fully matches at least one
    /// candidate, i.e. `position == candidates[i].len()` for some active `i`.
    fn is_complete(&self) -> bool {
        self.candidates
            .iter()
            .enumerate()
            .any(|(i, seq)| self.active[i] && self.position == seq.len())
    }

    /// Reset to initial state (all candidates active, position zero).
    fn reset(&mut self) {
        self.active.fill(true);
        self.position = 0;
    }

    fn name(&self) -> &str {
        "AllowListConstraint"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SequenceConstraint — force output to follow a specific token sequence exactly
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that forces the generated output to reproduce a specific,
/// pre-determined token sequence.
///
/// While `position < target.len()`, only `target[position]` is allowed.
/// Once the target has been fully reproduced (`position >= target.len()`) the
/// constraint is satisfied and all tokens become allowed again (returns `None`).
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::constrained_decoding::{SequenceConstraint, TokenConstraint};
///
/// let mut c = SequenceConstraint::new(vec![5, 6, 7]);
/// let mask = c.allowed_tokens(&[], 10).unwrap();
/// assert!(mask[5]);
/// assert!(!mask[6]);
/// assert_eq!(c.advance(5), true);
/// ```
pub struct SequenceConstraint {
    /// The token sequence that must be reproduced.
    target: Vec<u32>,
    /// Number of tokens consumed (next expected index into `target`).
    position: usize,
    /// Set to `true` if a mismatched token was ever committed.
    failed: bool,
}

impl SequenceConstraint {
    /// Create a new `SequenceConstraint` for the given target sequence.
    pub fn new(target: Vec<u32>) -> Self {
        Self {
            target,
            position: 0,
            failed: false,
        }
    }

    /// Whether the constraint has been violated (a wrong token was committed).
    pub fn is_failed(&self) -> bool {
        self.failed
    }
}

impl TokenConstraint for SequenceConstraint {
    /// Returns a bitmask allowing only the next expected token, or `None` once the
    /// full sequence has been reproduced.
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        if self.position >= self.target.len() {
            // Sequence fully consumed — no further restriction.
            return None;
        }
        let mut mask = vec![false; vocab_size];
        let next = self.target[self.position] as usize;
        if next < vocab_size {
            mask[next] = true;
        }
        Some(mask)
    }

    /// Commits `token`.  Returns `false` (and sets the failed flag) if `token`
    /// does not match the expected token at the current position.
    fn advance(&mut self, token: u32) -> bool {
        if self.position < self.target.len() && token != self.target[self.position] {
            self.failed = true;
            self.position += 1;
            return false;
        }
        self.position += 1;
        true
    }

    /// Returns `true` once all tokens in the target sequence have been consumed.
    fn is_complete(&self) -> bool {
        self.position >= self.target.len()
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.position = 0;
        self.failed = false;
    }

    fn name(&self) -> &str {
        "SequenceConstraint"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LengthConstraint — enforce hard minimum and maximum generation lengths
// ─────────────────────────────────────────────────────────────────────────────

/// A constraint that enforces hard minimum and maximum token-count limits.
///
/// - While `count < min_len`: if a `stop_token` is configured it is excluded from
///   the mask (cannot stop early).
/// - While `count >= max_len`: if a `stop_token` is configured only that token is
///   allowed; otherwise an all-`false` mask is returned, signalling the caller to
///   halt generation externally.
/// - Between `min_len` and `max_len`: all tokens are allowed (`None`).
///
/// Completion is defined as either reaching `max_len` OR generating `min_len` or
/// more tokens followed by the `stop_token`.
///
/// # Example
/// ```rust
/// use oxibonsai_runtime::constrained_decoding::{LengthConstraint, TokenConstraint};
///
/// // Must generate at least 2 tokens, stop token is 1 (EOS), max 10.
/// let mut c = LengthConstraint::new(2, 10, Some(1));
/// // Before min_len: stop_token excluded
/// let mask = c.allowed_tokens(&[], 4).unwrap();
/// assert!(!mask[1]);  // stop token blocked
/// assert!(mask[0]);   // other tokens allowed
/// ```
pub struct LengthConstraint {
    /// Minimum number of tokens that must be generated before stopping.
    min_len: usize,
    /// Hard upper bound on generated token count.
    max_len: usize,
    /// Optional end-of-sequence token; treated specially for early-stop control.
    stop_token: Option<u32>,
    /// Number of tokens committed via `advance` so far.
    count: usize,
    /// True once the `stop_token` has been committed.
    stop_seen: bool,
}

impl LengthConstraint {
    /// Create a new `LengthConstraint`.
    ///
    /// `min_len` must be `<= max_len`.
    pub fn new(min_len: usize, max_len: usize, stop_token: Option<u32>) -> Self {
        Self {
            min_len,
            max_len,
            stop_token,
            count: 0,
            stop_seen: false,
        }
    }

    /// Current token count.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl TokenConstraint for LengthConstraint {
    fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
        if self.count < self.min_len {
            // Cannot stop early — exclude stop_token if one is configured.
            if let Some(stop) = self.stop_token {
                let mut mask = vec![true; vocab_size];
                let stop_idx = stop as usize;
                if stop_idx < vocab_size {
                    mask[stop_idx] = false;
                }
                return Some(mask);
            }
            // No stop token: no restriction below min_len.
            return None;
        }

        if self.count >= self.max_len {
            // Must stop now.
            if let Some(stop) = self.stop_token {
                let mut mask = vec![false; vocab_size];
                let stop_idx = stop as usize;
                if stop_idx < vocab_size {
                    mask[stop_idx] = true;
                }
                return Some(mask);
            }
            // No stop token: emit an all-false mask to force external termination.
            return Some(vec![false; vocab_size]);
        }

        // Between min and max: unconstrained.
        None
    }

    /// Commits `token`, updating `count` and `stop_seen`.  Always returns `true`.
    fn advance(&mut self, token: u32) -> bool {
        if let Some(stop) = self.stop_token {
            if token == stop {
                self.stop_seen = true;
            }
        }
        self.count += 1;
        true
    }

    /// Returns `true` when at least `min_len` tokens have been generated AND either
    /// the `stop_token` was seen or `max_len` has been reached.
    fn is_complete(&self) -> bool {
        if self.count < self.min_len {
            return false;
        }
        self.count >= self.max_len || self.stop_seen
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.count = 0;
        self.stop_seen = false;
    }

    fn name(&self) -> &str {
        "LengthConstraint"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NoConstraint ─────────────────────────────────────────────────────────

    #[test]
    fn no_constraint_allows_all() {
        let nc = NoConstraint;
        assert!(nc.allowed_tokens(&[], 10).is_none());
    }

    // ── JsonConstraint ───────────────────────────────────────────────────────

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

    // ── RegexNfa ─────────────────────────────────────────────────────────────

    #[test]
    fn regex_nfa_literal_match() {
        let nfa = RegexNfa::from_pattern("abc").expect("valid pattern");
        assert!(nfa.is_full_match("abc"));
        assert!(!nfa.is_full_match("ab"));
        assert!(!nfa.is_full_match("abcd"));
    }

    #[test]
    fn regex_nfa_dot_match() {
        let nfa = RegexNfa::from_pattern("a.c").expect("valid pattern");
        assert!(nfa.is_full_match("abc"));
        assert!(nfa.is_full_match("axc"));
        assert!(!nfa.is_full_match("ac"));
    }

    #[test]
    fn regex_nfa_star_quantifier() {
        let nfa = RegexNfa::from_pattern("ab*c").expect("valid pattern");
        assert!(nfa.is_full_match("ac"));
        assert!(nfa.is_full_match("abc"));
        assert!(nfa.is_full_match("abbc"));
        assert!(!nfa.is_full_match("xbc"));
    }

    #[test]
    fn regex_nfa_alternation() {
        let nfa = RegexNfa::from_pattern("cat|dog").expect("valid pattern");
        assert!(nfa.is_full_match("cat"));
        assert!(nfa.is_full_match("dog"));
        assert!(!nfa.is_full_match("cow"));
    }

    // ── RegexConstraint ──────────────────────────────────────────────────────

    #[test]
    fn regex_constraint_is_match() {
        assert!(RegexConstraint::is_match("he+llo", "hello"));
        assert!(RegexConstraint::is_match("he+llo", "heeeello"));
        assert!(!RegexConstraint::is_match("he+llo", "hllo"));
    }

    #[test]
    fn regex_constraint_allows_valid_chars() {
        let rc = RegexConstraint::new("abc").expect("valid");
        // 'a' (97) should be valid as first char
        assert!(rc.char_is_valid('a'));
        assert!(!rc.char_is_valid('b')); // 'b' is not valid before 'a'
    }

    // ── ConstrainedSampler ───────────────────────────────────────────────────

    #[test]
    fn constrained_sampler_masks_logits() {
        // vocab_size = 4; mask allows only tokens 0 and 2
        struct AllowEvens;
        impl TokenConstraint for AllowEvens {
            fn allowed_tokens(&self, _: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
                Some((0..vocab_size).map(|i| i % 2 == 0).collect())
            }
            fn advance(&mut self, _: u32) -> bool {
                true
            }
            fn is_complete(&self) -> bool {
                true
            }
            fn reset(&mut self) {}
            fn name(&self) -> &str {
                "AllowEvens"
            }
        }

        let chain = crate::sampling_advanced::SamplerChain::greedy();
        let mut sampler = ConstrainedSampler::new(chain, Box::new(AllowEvens), 4);
        // Make token 1 have highest logit; after masking token 0 should win.
        let mut logits = vec![2.0_f32, 10.0, 1.0, 0.5];
        // token 1 is masked → token 0 wins (highest among allowed)
        let tok = sampler.sample(&mut logits);
        assert_eq!(tok, 0);
    }

    #[test]
    fn constrained_sampler_greedy_json() {
        let chain = crate::sampling_advanced::SamplerChain::greedy();
        let mut sampler = ConstrainedSampler::new(chain, Box::new(JsonConstraint::new()), 256);
        assert!(!sampler.is_complete());
        // Feed '{' then '}'
        let mut logits_open = vec![0.0_f32; 256];
        logits_open['{' as usize] = 100.0;
        sampler.sample(&mut logits_open);

        let mut logits_close = vec![0.0_f32; 256];
        logits_close['}' as usize] = 100.0;
        sampler.sample(&mut logits_close);

        assert!(sampler.is_complete());
        assert_eq!(sampler.generated_text_len(), 2);
    }

    #[test]
    fn constrained_sampler_reset() {
        let chain = crate::sampling_advanced::SamplerChain::greedy();
        let mut sampler = ConstrainedSampler::new(chain, Box::new(JsonConstraint::new()), 256);
        let mut logits = vec![0.0_f32; 256];
        logits['{' as usize] = 100.0;
        sampler.sample(&mut logits);
        assert_eq!(sampler.generated_text_len(), 1);
        sampler.reset();
        assert_eq!(sampler.generated_text_len(), 0);
    }

    #[test]
    fn constrained_sampler_builder_json() {
        let sampler = ConstrainedSamplerBuilder::new(256, 42).with_json_constraint();
        assert_eq!(sampler.constraint_name(), "JsonConstraint");
    }

    #[test]
    fn constrained_sampler_builder_unconstrained() {
        let sampler = ConstrainedSamplerBuilder::new(256, 42).unconstrained();
        assert_eq!(sampler.constraint_name(), "NoConstraint");
        assert!(sampler.is_complete());
    }

    #[test]
    fn constraint_error_display() {
        let e = ConstraintError::InvalidPattern("bad".to_string());
        assert!(e.to_string().contains("bad"));
        let e2 = ConstraintError::Violated {
            token: 5,
            reason: "oops".to_string(),
        };
        assert!(e2.to_string().contains("5"));
        assert!(e2.to_string().contains("oops"));
    }
}
