//! Tokenizer utilities: normalization, special token handling, chat templates.
//!
//! This module provides:
//! - [`TextNormalizer`] — pure-Rust Unicode normalization (lowercase, accent stripping,
//!   whitespace collapsing) without any ICU / C-library dependency.
//! - [`ChatTemplate`] — Jinja2-like chat template engine for multi-turn conversations.
//! - [`TruncationSide`] / [`PaddingStrategy`] — truncation and padding enums.
//! - [`BatchEncoder`] — batch tokenization with optional truncation and padding.
//! - [`BatchEncoding`] — the resulting padded token ID matrix plus attention masks.

use unicode_normalization::UnicodeNormalization;

use crate::{error::TokenizerResult, tokenizer::OxiTokenizer};

// ── TextNormalizer ────────────────────────────────────────────────────────────

/// Configurable Unicode text normalizer (pure Rust, no ICU dependency).
///
/// The normalizer applies a chain of transforms in this order:
/// 1. Optional ASCII lowercasing.
/// 2. Optional combining-character (accent) stripping.
/// 3. Optional leading/trailing whitespace stripping.
/// 4. Optional whitespace collapsing (multiple spaces → one space).
#[derive(Debug, Clone)]
pub struct TextNormalizer {
    /// Convert ASCII alphabetic characters to lowercase.
    pub lowercase: bool,
    /// Remove Unicode combining characters (strips accents from base letters).
    pub strip_accents: bool,
    /// Strip leading and trailing whitespace.
    pub strip_whitespace: bool,
    /// Collapse interior whitespace runs into a single space.
    pub collapse_whitespace: bool,
}

impl Default for TextNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextNormalizer {
    /// Create a normalizer with all transforms disabled.
    pub fn new() -> Self {
        Self {
            lowercase: false,
            strip_accents: false,
            strip_whitespace: false,
            collapse_whitespace: false,
        }
    }

    /// Convenience constructor: lowercase only.
    pub fn lowercase_only() -> Self {
        Self {
            lowercase: true,
            ..Self::new()
        }
    }

    /// Convenience constructor: strip + collapse whitespace.
    pub fn whitespace_only() -> Self {
        Self {
            strip_whitespace: true,
            collapse_whitespace: true,
            ..Self::new()
        }
    }

    /// Apply the configured normalization pipeline to `text`.
    pub fn normalize(&self, text: &str) -> String {
        // Step 1: lowercase.
        let mut result: String = if self.lowercase {
            text.chars()
                .map(|c| {
                    // Only lowercase ASCII letters to avoid locale-sensitive surprises.
                    if c.is_ascii_alphabetic() {
                        c.to_ascii_lowercase()
                    } else {
                        c
                    }
                })
                .collect()
        } else {
            text.to_owned()
        };

        // Step 2: strip combining characters (accents).
        //
        // Decompose to NFD first (TOK-09 missed-finding fix): a precomposed
        // character like "é" (U+00E9) has no combining mark to strip at all
        // under a raw code-point scan, while its decomposed form "e" +
        // U+0301 does — so without decomposing first, `strip_accents`
        // silently disagreed on the two Unicode-equivalent spellings of the
        // same text. Decomposing unconditionally makes both forms produce
        // the same stripped output.
        if self.strip_accents {
            result = result.nfd().filter(|&c| !is_combining(c)).collect();
        }

        // Step 3: collapse interior whitespace.
        if self.collapse_whitespace {
            let mut collapsed = String::with_capacity(result.len());
            let mut prev_was_space = false;
            for c in result.chars() {
                if c.is_whitespace() {
                    if !prev_was_space {
                        collapsed.push(' ');
                    }
                    prev_was_space = true;
                } else {
                    collapsed.push(c);
                    prev_was_space = false;
                }
            }
            result = collapsed;
        }

        // Step 4: strip leading/trailing whitespace.
        if self.strip_whitespace {
            result = result.trim().to_owned();
        }

        result
    }
}

/// Returns `true` if `c` is a Unicode combining character (general category
/// `Mn` Non-spacing Mark, `Mc` Spacing Mark, or `Me` Enclosing Mark).
///
/// TOK-09 missed-finding fix: the previous implementation was a hand-rolled
/// 4-block range check that covered only a handful of `Mn` blocks (Latin
/// combining diacritics) and — despite a doc comment claiming otherwise —
/// missed `Mc` entirely and every combining mark outside those four blocks:
/// Cyrillic (U+0483–0489), Hebrew points (U+0591–05C7), Arabic diacritics
/// (U+064B–065F, U+0670), Devanagari matras/virama (U+0900–0903,
/// U+093A–094F), Thai (U+0E31–0E3A), and more. `unicode-normalization`'s
/// `is_combining_mark` is a complete, table-driven general-category check
/// (Pure Rust, already a workspace dependency for NFC — see
/// `crate::tokenizer::TokenizerConfig::normalize_nfc`), so this now
/// delegates to it rather than re-deriving an incomplete range list.
fn is_combining(c: char) -> bool {
    unicode_normalization::char::is_combining_mark(c)
}

// ── ChatTemplate ──────────────────────────────────────────────────────────────

/// A minimal Jinja2-like chat template engine.
///
/// Supported syntax:
/// - `{{ role }}` / `{{ content }}` — variable substitution inside `{% for %}`
/// - `{% for message in messages %}` … `{% endfor %}` — loop over messages
/// - `{% if role == "system" %}` … `{% endif %}` — simple equality condition
/// - `{% if role == "system" %}` … `{% else %}` … `{% endif %}` — else branch
///
/// The template is processed sequentially; nesting is **not** supported.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template: String,
}

impl ChatTemplate {
    /// Construct a template from a raw template string.
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_owned(),
        }
    }

    /// ChatML template — used by Qwen3 / OxiBonsai.
    ///
    /// ```text
    /// <|im_start|>{{ role }}
    /// {{ content }}<|im_end|>
    /// ```
    pub fn chatml() -> Self {
        Self::new(
            "{% for message in messages %}<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n{% endfor %}",
        )
    }

    /// Llama-3 style template.
    pub fn llama3() -> Self {
        Self::new(concat!(
            "<|begin_of_text|>",
            "{% for message in messages %}",
            "<|start_header_id|>{{ role }}<|end_header_id|>\n\n",
            "{{ content }}<|eot_id|>",
            "{% endfor %}",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ))
    }

    /// Format a list of `(role, content)` pairs into a single prompt string.
    ///
    /// The template is evaluated for each message in the list.
    pub fn format(&self, messages: &[(&str, &str)]) -> String {
        render_template(&self.template, messages)
    }

    /// Extract the last user message from a formatted ChatML prompt.
    ///
    /// Scans backwards for the last `<|im_start|>user\n…<|im_end|>` block.
    pub fn extract_user_message(prompt: &str) -> Option<String> {
        // Walk in reverse looking for the last user block.
        let mut search = prompt;
        let marker = "<|im_start|>user\n";
        let end_marker = "<|im_end|>";

        let mut last_user: Option<String> = None;

        while let Some(start_pos) = search.find(marker) {
            let after_marker = &search[start_pos + marker.len()..];
            if let Some(end_pos) = after_marker.find(end_marker) {
                last_user = Some(after_marker[..end_pos].to_owned());
            }
            // Advance past this marker.
            search = &search[start_pos + marker.len()..];
        }

        last_user
    }
}

/// Evaluate the template string against the given messages.
///
/// The evaluator is a simple state machine that handles `{% for %}`, `{% if %}`,
/// `{% else %}`, `{% endif %}`, `{% endfor %}`, and `{{ var }}` tags.
///
/// Exposed at crate level so that [`crate::chat_templates`] can share it.
pub(crate) fn render_template(template: &str, messages: &[(&str, &str)]) -> String {
    // Split the template into tokens: literals and tags.
    let tokens = tokenize_template(template);

    let mut output = String::new();

    // Find the for-loop body boundaries.
    // We support exactly one top-level {% for message in messages %} loop.
    let for_start = tokens.iter().position(|t| {
        matches!(t, TemplateToken::Tag(s) if s.trim().starts_with("for ") && s.contains("messages"))
    });
    let for_end = tokens
        .iter()
        .position(|t| matches!(t, TemplateToken::Tag(s) if s.trim() == "endfor"));

    match (for_start, for_end) {
        (Some(fs), Some(fe)) if fs < fe => {
            // Render content before the loop.
            for tok in &tokens[..fs] {
                if let TemplateToken::Literal(lit) = tok {
                    output.push_str(lit);
                }
            }

            // Render the loop body for each message.
            let body_tokens = &tokens[fs + 1..fe];
            for (role, content) in messages {
                output.push_str(&render_body(body_tokens, role, content));
            }

            // Render content after the loop.
            for tok in &tokens[fe + 1..] {
                if let TemplateToken::Literal(lit) = tok {
                    output.push_str(lit);
                }
            }
        }
        _ => {
            // No loop — treat the whole template as the body for the first message.
            if let Some((role, content)) = messages.first() {
                output.push_str(&render_body(&tokens, role, content));
            }
        }
    }

    output
}

/// Render one iteration of the loop body (or the full template body) for a
/// single `(role, content)` pair, handling `{% if %}` / `{% else %}` / `{% endif %}`.
fn render_body(tokens: &[TemplateToken], role: &str, content: &str) -> String {
    let mut output = String::new();
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            TemplateToken::Literal(lit) => {
                output.push_str(lit);
                i += 1;
            }
            TemplateToken::Variable(var) => {
                let val = resolve_variable(var.trim(), role, content);
                output.push_str(&val);
                i += 1;
            }
            TemplateToken::Tag(tag) => {
                let tag_trimmed = tag.trim();
                if tag_trimmed.starts_with("if ") {
                    // Parse condition: `if role == "system"` etc.
                    let condition_met = evaluate_condition(tag_trimmed, role, content);

                    // Collect the if-body, else-body, and skip to endif.
                    let (if_body, else_body, skip) = collect_if_bodies(&tokens[i + 1..]);

                    if condition_met {
                        output.push_str(&render_body(&if_body, role, content));
                    } else {
                        output.push_str(&render_body(&else_body, role, content));
                    }

                    i += 1 + skip;
                } else {
                    // Ignore unknown/structural tags (endfor, endif, else handled above).
                    i += 1;
                }
            }
        }
    }

    output
}

/// Collect tokens belonging to the if-body and (optionally) else-body.
///
/// Returns `(if_tokens, else_tokens, tokens_consumed)`.
fn collect_if_bodies(tokens: &[TemplateToken]) -> (Vec<TemplateToken>, Vec<TemplateToken>, usize) {
    let mut if_body = Vec::new();
    let mut else_body = Vec::new();
    let mut in_else = false;
    let mut depth = 1usize; // nesting depth (for nested ifs, not fully supported but safe)
    let mut consumed = 0;

    for (idx, tok) in tokens.iter().enumerate() {
        consumed = idx + 1;
        match tok {
            TemplateToken::Tag(tag) => {
                let t = tag.trim();
                if t.starts_with("if ") {
                    depth += 1;
                    if in_else {
                        else_body.push(tok.clone());
                    } else {
                        if_body.push(tok.clone());
                    }
                } else if t == "endif" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    if in_else {
                        else_body.push(tok.clone());
                    } else {
                        if_body.push(tok.clone());
                    }
                } else if t == "else" && depth == 1 {
                    in_else = true;
                } else if in_else {
                    else_body.push(tok.clone());
                } else {
                    if_body.push(tok.clone());
                }
            }
            other => {
                if in_else {
                    else_body.push(other.clone());
                } else {
                    if_body.push(other.clone());
                }
            }
        }
    }

    (if_body, else_body, consumed)
}

/// Evaluate a simple `if` condition of the form `if VAR == "VALUE"`.
fn evaluate_condition(tag: &str, role: &str, content: &str) -> bool {
    // Strip the leading "if ".
    let rest = tag.trim_start_matches("if ").trim();

    // Split on "==" — very basic parser.
    if let Some((lhs, rhs)) = rest.split_once("==") {
        let lhs = lhs.trim();
        let rhs = rhs.trim().trim_matches('"').trim_matches('\'');
        let lhs_val = resolve_variable(lhs, role, content);
        return lhs_val == rhs;
    }

    // Unknown condition — default to false.
    false
}

/// Resolve a template variable name to its string value.
fn resolve_variable(var: &str, role: &str, content: &str) -> String {
    match var {
        "role" | "message.role" => role.to_owned(),
        "content" | "message.content" => content.to_owned(),
        _ => String::new(),
    }
}

/// A single token in the parsed template.
#[derive(Debug, Clone)]
enum TemplateToken {
    /// Plain text literal.
    Literal(String),
    /// `{{ variable }}` expression.
    Variable(String),
    /// `{% tag %}` control tag (without the `{% %}` delimiters).
    Tag(String),
}

/// Tokenize a template string into [`TemplateToken`]s.
fn tokenize_template(template: &str) -> Vec<TemplateToken> {
    let mut tokens = Vec::new();
    let mut rest = template;

    while !rest.is_empty() {
        // Look for the next tag or variable opener.
        let var_pos = rest.find("{{");
        let tag_pos = rest.find("{%");

        // Determine which delimiter ({{ or {%) appears first.
        let next = match (var_pos, tag_pos) {
            (None, None) => {
                // No more delimiters — everything remaining is a literal.
                tokens.push(TemplateToken::Literal(rest.to_owned()));
                break;
            }
            (Some(vp), None) => Some(('v', vp)),
            (None, Some(tp)) => Some(('t', tp)),
            (Some(vp), Some(tp)) => {
                if vp <= tp {
                    Some(('v', vp))
                } else {
                    Some(('t', tp))
                }
            }
        };

        match next {
            None => break, // handled above
            Some(('v', vp)) => {
                // Variable `{{ … }}` comes first.
                if vp > 0 {
                    tokens.push(TemplateToken::Literal(rest[..vp].to_owned()));
                }
                let after_open = &rest[vp + 2..];
                if let Some(close) = after_open.find("}}") {
                    tokens.push(TemplateToken::Variable(after_open[..close].to_owned()));
                    rest = &after_open[close + 2..];
                } else {
                    tokens.push(TemplateToken::Literal(rest.to_owned()));
                    break;
                }
            }
            Some(('t', tp)) => {
                // Tag `{% … %}` comes first.
                if tp > 0 {
                    tokens.push(TemplateToken::Literal(rest[..tp].to_owned()));
                }
                let after_open = &rest[tp + 2..];
                if let Some(close) = after_open.find("%}") {
                    tokens.push(TemplateToken::Tag(after_open[..close].to_owned()));
                    rest = &after_open[close + 2..];
                } else {
                    tokens.push(TemplateToken::Literal(rest.to_owned()));
                    break;
                }
            }
            Some(_) => break, // unreachable, satisfies exhaustiveness
        }
    }

    tokens
}

// ── TruncationSide ────────────────────────────────────────────────────────────

/// Which side of a token sequence to truncate from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationSide {
    /// Remove tokens from the left (beginning) of the sequence.
    Left,
    /// Remove tokens from the right (end) of the sequence.
    Right,
}

// ── PaddingStrategy ───────────────────────────────────────────────────────────

/// How to pad sequences in a batch to the same length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad all sequences to exactly `n` tokens.
    Fixed(usize),
    /// Pad all sequences to the length of the longest sequence in the batch.
    Longest,
}

// ── BatchEncoder ──────────────────────────────────────────────────────────────

/// Batch tokenizer that encodes multiple texts with optional truncation and padding.
///
/// Build with the builder-pattern setters, then call [`BatchEncoder::encode_batch`].
pub struct BatchEncoder<'a> {
    tokenizer: &'a OxiTokenizer,
    /// Hard maximum token count (before padding).
    pub max_length: Option<usize>,
    /// Which side to truncate from if a sequence exceeds `max_length`.
    pub truncation: Option<TruncationSide>,
    /// Padding strategy applied after all sequences are encoded.
    pub padding: Option<PaddingStrategy>,
    /// Token ID used for padding positions.
    pub pad_token_id: u32,
}

impl<'a> BatchEncoder<'a> {
    /// Create a new `BatchEncoder` wrapping `tokenizer`.
    ///
    /// Truncation, padding, and `max_length` are all disabled by default.
    /// The pad token ID defaults to `3` (matching `OxiTokenizer`'s default config).
    pub fn new(tokenizer: &'a OxiTokenizer) -> Self {
        Self {
            tokenizer,
            max_length: None,
            truncation: None,
            padding: None,
            pad_token_id: 3,
        }
    }

    /// Set the maximum sequence length.
    pub fn with_max_length(mut self, n: usize) -> Self {
        self.max_length = Some(n);
        self
    }

    /// Set the truncation side.
    pub fn with_truncation(mut self, side: TruncationSide) -> Self {
        self.truncation = Some(side);
        self
    }

    /// Set the padding strategy.
    pub fn with_padding(mut self, strategy: PaddingStrategy) -> Self {
        self.padding = Some(strategy);
        self
    }

    /// Encode a batch of text strings, applying truncation and padding as configured.
    ///
    /// Returns a [`BatchEncoding`] with `input_ids`, `attention_mask`, and `lengths`.
    pub fn encode_batch(&self, texts: &[&str]) -> TokenizerResult<BatchEncoding> {
        if texts.is_empty() {
            return Ok(BatchEncoding {
                input_ids: Vec::new(),
                attention_mask: Vec::new(),
                lengths: Vec::new(),
            });
        }

        // Step 1: encode each text.
        let mut encoded: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| self.tokenizer.encode(t))
            .collect::<TokenizerResult<_>>()?;

        // Step 2: truncate.
        if let Some(max) = self.max_length {
            for seq in &mut encoded {
                if seq.len() > max {
                    match self.truncation.unwrap_or(TruncationSide::Right) {
                        TruncationSide::Right => {
                            seq.truncate(max);
                        }
                        TruncationSide::Left => {
                            let excess = seq.len() - max;
                            seq.drain(..excess);
                        }
                    }
                }
            }
        }

        // Step 3: record actual lengths.
        let lengths: Vec<usize> = encoded.iter().map(Vec::len).collect();

        // Step 4: determine target pad length.
        let target_len = match self.padding {
            None => None,
            Some(PaddingStrategy::Longest) => lengths.iter().copied().max(),
            Some(PaddingStrategy::Fixed(n)) => Some(n),
        };

        // Step 5: build attention masks and pad input_ids.
        let mut input_ids: Vec<Vec<u32>> = Vec::with_capacity(encoded.len());
        let mut attention_mask: Vec<Vec<u32>> = Vec::with_capacity(encoded.len());

        for (seq, &len) in encoded.iter().zip(lengths.iter()) {
            match target_len {
                None => {
                    // No padding — mask is all-ones.
                    let mask = vec![1u32; len];
                    input_ids.push(seq.clone());
                    attention_mask.push(mask);
                }
                Some(pad_to) => {
                    let mut ids = seq.clone();
                    let pad_count = pad_to.saturating_sub(len);
                    ids.extend(std::iter::repeat_n(self.pad_token_id, pad_count));

                    let mut mask = vec![1u32; len];
                    mask.extend(std::iter::repeat_n(0u32, pad_count));

                    input_ids.push(ids);
                    attention_mask.push(mask);
                }
            }
        }

        Ok(BatchEncoding {
            input_ids,
            attention_mask,
            lengths,
        })
    }
}

// ── BatchEncoding ─────────────────────────────────────────────────────────────

/// The result of batch-encoding multiple texts.
pub struct BatchEncoding {
    /// Token ID matrix — shape `[batch_size][padded_seq_len]`.
    pub input_ids: Vec<Vec<u32>>,
    /// Attention mask — `1` for real tokens, `0` for padding.
    pub attention_mask: Vec<Vec<u32>>,
    /// Actual (non-padded) token count per sequence.
    pub lengths: Vec<usize>,
}

impl BatchEncoding {
    /// Number of sequences in the batch.
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Length of the longest (padded) sequence.
    pub fn max_seq_len(&self) -> usize {
        self.input_ids.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Returns `true` if any sequence is shorter than `max_seq_len` (i.e., padding was applied).
    pub fn is_padded(&self) -> bool {
        let max = self.max_seq_len();
        self.lengths.iter().any(|&l| l < max)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OxiTokenizer;

    // ── TextNormalizer ──

    #[test]
    fn test_text_normalizer_lowercase() {
        let n = TextNormalizer::lowercase_only();
        assert_eq!(n.normalize("Hello World"), "hello world");
        assert_eq!(n.normalize("ABC123"), "abc123");
        assert_eq!(n.normalize("already lower"), "already lower");
    }

    #[test]
    fn test_text_normalizer_collapse_whitespace() {
        let n = TextNormalizer::whitespace_only();
        assert_eq!(n.normalize("  hello   world  "), "hello world");
        assert_eq!(n.normalize("a  b  c"), "a b c");
        assert_eq!(n.normalize("no extra"), "no extra");
    }

    #[test]
    fn test_text_normalizer_strip_accents() {
        let n = TextNormalizer {
            strip_accents: true,
            ..TextNormalizer::new()
        };
        // U+0301 is COMBINING ACUTE ACCENT — should be stripped from "e\u{0301}"
        let input = "caf\u{0065}\u{0301}";
        let result = n.normalize(input);
        assert!(
            !result.contains('\u{0301}'),
            "combining accent should be removed"
        );
    }

    #[test]
    fn test_text_normalizer_strip_accents_precomposed_and_decomposed_agree() {
        // TOK-09 missed-finding fix: precomposed "é" (U+00E9) and decomposed
        // "e" + U+0301 must strip to the SAME result now that NFD runs
        // first — before the fix, only the decomposed form had anything to
        // strip at all.
        let n = TextNormalizer {
            strip_accents: true,
            ..TextNormalizer::new()
        };
        let precomposed = n.normalize("caf\u{00e9}");
        let decomposed = n.normalize("cafe\u{0301}");
        assert_eq!(precomposed, decomposed);
        assert_eq!(precomposed, "cafe");
    }

    #[test]
    fn test_text_normalizer_strip_accents_covers_non_latin_scripts() {
        // TOK-09 missed-finding fix: the old 4-block range list covered
        // only a few Latin-combining-mark blocks. These are the exact
        // codepoints the finding named as missed: Devanagari virama,
        // Arabic fatha, Hebrew qamats, Cyrillic combining breve.
        let n = TextNormalizer {
            strip_accents: true,
            ..TextNormalizer::new()
        };
        // Devanagari "न" + U+094D (VIRAMA) -> bare "न".
        assert_eq!(n.normalize("न\u{094d}"), "न");
        // Arabic "ب" + U+064E (FATHA) -> bare "ب".
        assert_eq!(n.normalize("ب\u{064e}"), "ب");
        // Hebrew "ש" + U+05B8 (QAMATS) -> bare "ש".
        assert_eq!(n.normalize("ש\u{05b8}"), "ש");
        // Cyrillic "а" + U+0483 (COMBINING CYRILLIC TITLO) -> bare "а".
        assert_eq!(n.normalize("а\u{0483}"), "а");
    }

    #[test]
    fn test_is_combining_covers_spacing_marks_mc() {
        // TOK-09 missed-finding fix: the old implementation missed `Mc`
        // (Spacing Mark) entirely. U+0903 DEVANAGARI SIGN VISARGA is `Mc`.
        assert!(is_combining('\u{0903}'));
    }

    #[test]
    fn test_text_normalizer_combined() {
        let n = TextNormalizer {
            lowercase: true,
            strip_whitespace: true,
            collapse_whitespace: true,
            ..TextNormalizer::new()
        };
        assert_eq!(n.normalize("  HELLO   WORLD  "), "hello world");
    }

    // ── ChatTemplate ──

    #[test]
    fn test_chat_template_chatml_format() {
        let tmpl = ChatTemplate::chatml();
        let messages = [("user", "Hello!")];
        let out = tmpl.format(&messages);
        assert!(
            out.contains("<|im_start|>user"),
            "should contain user start token"
        );
        assert!(out.contains("Hello!"), "should contain message content");
        assert!(out.contains("<|im_end|>"), "should contain end token");
    }

    #[test]
    fn test_chat_template_multi_turn() {
        let tmpl = ChatTemplate::chatml();
        let messages = [
            ("system", "You are helpful."),
            ("user", "What is 2+2?"),
            ("assistant", "4"),
        ];
        let out = tmpl.format(&messages);
        assert!(out.contains("<|im_start|>system"), "system role present");
        assert!(out.contains("<|im_start|>user"), "user role present");
        assert!(
            out.contains("<|im_start|>assistant"),
            "assistant role present"
        );
        assert!(out.contains("You are helpful."), "system content present");
        assert!(out.contains("What is 2+2?"), "user content present");
        assert!(out.contains('4'), "assistant content present");
    }

    #[test]
    fn test_chat_template_extract_user_message() {
        let tmpl = ChatTemplate::chatml();
        let messages = [("user", "Find me a recipe.")];
        let formatted = tmpl.format(&messages);
        let extracted = ChatTemplate::extract_user_message(&formatted);
        assert_eq!(extracted, Some("Find me a recipe.".to_owned()));
    }

    #[test]
    fn test_chat_template_extract_user_message_multi_turn() {
        let tmpl = ChatTemplate::chatml();
        let messages = [
            ("user", "First question"),
            ("assistant", "First answer"),
            ("user", "Second question"),
        ];
        let formatted = tmpl.format(&messages);
        let extracted = ChatTemplate::extract_user_message(&formatted);
        // Should return the LAST user message.
        assert_eq!(extracted, Some("Second question".to_owned()));
    }

    // ── BatchEncoder ──

    fn make_tokenizer() -> OxiTokenizer {
        OxiTokenizer::char_level_stub(256)
    }

    #[test]
    fn test_batch_encoder_basic() {
        let tok = make_tokenizer();
        let enc = BatchEncoder::new(&tok);
        let result = enc
            .encode_batch(&["hello", "world"])
            .expect("batch encode should succeed");
        assert_eq!(result.batch_size(), 2);
        assert!(!result.input_ids[0].is_empty());
        assert!(!result.input_ids[1].is_empty());
    }

    #[test]
    fn test_batch_encoder_truncation_right() {
        let tok = make_tokenizer();
        let enc = BatchEncoder::new(&tok)
            .with_max_length(3)
            .with_truncation(TruncationSide::Right);
        let result = enc
            .encode_batch(&["hello world"])
            .expect("encode should succeed");
        assert_eq!(result.lengths[0], 3, "should be truncated to 3 tokens");
        assert_eq!(result.input_ids[0].len(), 3);
    }

    #[test]
    fn test_batch_encoder_truncation_left() {
        let tok = make_tokenizer();
        // Encode without truncation to find the full token IDs.
        let full = tok.encode("hello").expect("encode");
        let full_len = full.len();

        let enc = BatchEncoder::new(&tok)
            .with_max_length(2)
            .with_truncation(TruncationSide::Left);
        let result = enc.encode_batch(&["hello"]).expect("encode should succeed");

        if full_len >= 2 {
            // The kept tokens should be the last 2 of the full sequence.
            assert_eq!(result.input_ids[0], full[full_len - 2..]);
        }
        assert!(result.lengths[0] <= 2);
    }

    #[test]
    fn test_batch_encoder_padding_fixed() {
        let tok = make_tokenizer();
        let enc = BatchEncoder::new(&tok).with_padding(PaddingStrategy::Fixed(10));
        let result = enc
            .encode_batch(&["hi", "hello"])
            .expect("encode should succeed");
        // Both sequences should be padded to exactly 10.
        for ids in &result.input_ids {
            assert_eq!(ids.len(), 10);
        }
    }

    #[test]
    fn test_batch_encoding_attention_mask() {
        let tok = make_tokenizer();
        let enc = BatchEncoder::new(&tok).with_padding(PaddingStrategy::Longest);
        let result = enc
            .encode_batch(&["hi", "hello world"])
            .expect("encode should succeed");

        let max_len = result.max_seq_len();
        for (i, mask) in result.attention_mask.iter().enumerate() {
            assert_eq!(mask.len(), max_len, "mask length matches padded seq len");
            let real_len = result.lengths[i];
            // Real tokens → 1.
            for &m in &mask[..real_len] {
                assert_eq!(m, 1u32, "real token position should have mask=1");
            }
            // Padding → 0.
            for &m in &mask[real_len..] {
                assert_eq!(m, 0u32, "padding position should have mask=0");
            }
        }
    }

    #[test]
    fn test_batch_encoding_empty() {
        let tok = make_tokenizer();
        let enc = BatchEncoder::new(&tok);
        let result = enc.encode_batch(&[]).expect("empty batch should succeed");
        assert_eq!(result.batch_size(), 0);
        assert_eq!(result.max_seq_len(), 0);
        assert!(!result.is_padded());
    }
}
