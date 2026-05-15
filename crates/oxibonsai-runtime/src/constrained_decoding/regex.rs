//! Regex-based [`TokenConstraint`] implementation backed by a minimal NFA
//! engine.
//!
//! This sub-module hosts the NFA compiler/simulator (`NfaState`, `RegexNfa`,
//! `Fragment`) plus the public [`RegexConstraint`] type.

use super::error_trait::{ConstraintError, TokenConstraint};

// ─────────────────────────────────────────────────────────────────────────────
// Minimal NFA-based regex engine
// ─────────────────────────────────────────────────────────────────────────────

/// One NFA state.
#[derive(Debug, Clone)]
pub(super) enum NfaState {
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
pub(super) struct RegexNfa {
    states: Vec<NfaState>,
    start: usize,
    accept_state: usize,
}

/// A fragment of NFA states returned by the compiler — holds start index and
/// a list of "dangling" out-arrows that must be patched to the next fragment.
pub(super) struct Fragment {
    start: usize,
    /// Indices of states whose outgoing arrow is "open" (needs patching).
    outs: Vec<usize>,
}

impl RegexNfa {
    /// Build an NFA from a regex pattern.
    pub(super) fn from_pattern(pattern: &str) -> Result<Self, ConstraintError> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
