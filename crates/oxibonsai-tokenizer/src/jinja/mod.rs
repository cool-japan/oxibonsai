//! A real Jinja **subset** engine for chat templates.
//!
//! Chat templates shipped inside GGUF files (`tokenizer.chat_template`) are
//! genuine Jinja programs: Bonsai 2's template alone uses macros with default
//! arguments, namespaces, reverse slices, `loop.previtem`/`nextitem`, the
//! `tojson`/`items`/`selectattr` filters, `is string` / `is mapping` /
//! `is iterable` tests and `raise_exception`.  Pattern-matching such a
//! template with string substitution produces plausible-looking but wrong
//! prompts, which is why this module exists.
//!
//! Two properties are load-bearing:
//!
//! 1. **Never render garbage.**  Anything outside the supported subset is a
//!    hard [`JinjaError`] — unknown statements, filters and tests are rejected
//!    at *compile* time with a line and column; unsupported runtime operations
//!    are rejected at render time.  `raise_exception(msg)` in a template
//!    surfaces as [`JinjaError::TemplateRaise`], never as a panic.
//! 2. **Never hang or panic.**  The lexer and parser are total, recursion and
//!    loop iteration counts are bounded, arithmetic is checked, and the output
//!    size is capped — this engine renders attacker-influenced data on the
//!    serve path.
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_tokenizer::jinja::{JinjaTemplate, Value};
//!
//! let template = JinjaTemplate::compile(
//!     "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor -%}",
//! )
//! .expect("template compiles");
//!
//! let context = Value::from_json_str(
//!     r#"{"messages": [{"role": "user", "content": "hi"}]}"#,
//! )
//! .expect("context parses");
//!
//! assert_eq!(
//!     template.render(&context).expect("render"),
//!     "<|im_start|>user\nhi<|im_end|>\n"
//! );
//! ```
//!
//! # Whitespace
//!
//! The defaults match the environment both HuggingFace `transformers` and the
//! llama.cpp chat-template renderer use: `trim_blocks` and `lstrip_blocks` on,
//! a single trailing newline dropped, and CRLF normalised to LF.  The `{%-`,
//! `-%}`, `{{-` and `-}}` markers work as in Jinja and override the defaults.
//!
//! # Supported subset
//!
//! * statements: `if` / `elif` / `else`, `for` … `else` (with an inline
//!   `if` filter), `set` (plain, tuple-unpacking, `namespace` attribute and
//!   block form), `macro`, `call`, `generation`, `break`, `continue`,
//!   comments;
//! * expressions: literals (including list, tuple and dict), attribute and
//!   item access, Python slices, calls with keyword arguments, `and`/`or`/
//!   `not`, chained comparisons, `in` / `not in`, arithmetic, `~` concat and
//!   the inline conditional;
//! * filters: see [`parser::SUPPORTED_FILTERS`];
//! * tests: see [`parser::SUPPORTED_TESTS`];
//! * globals: `raise_exception`, `namespace`, `range`, `dict`;
//! * string methods `startswith`, `endswith`, `strip`, `lstrip`, `rstrip`,
//!   `upper`, `lower`, `replace`, `split`, `join`; mapping methods `get`,
//!   `keys`, `values`, `items`.

pub mod ast;
pub mod eval;
pub mod lexer;
pub mod parser;
pub mod value;

use thiserror::Error;

use crate::error::TokenizerError;
use ast::Node;
use lexer::Lexer;
use parser::Parser;

pub use value::{Value, ValueMap};

/// Everything that can go wrong compiling or rendering a template.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JinjaError {
    /// The template is malformed, or uses a construct outside the supported
    /// subset.  Always carries a 1-based line and column.
    #[error("jinja syntax error at line {line}, column {column}: {message}")]
    Syntax {
        /// 1-based line of the offending token.
        line: usize,
        /// 1-based column of the offending token.
        column: usize,
        /// What went wrong.
        message: String,
    },

    /// The template is well-formed but the operation is not valid on the
    /// values it was given (type mismatch, division by zero, ...).
    #[error("jinja runtime error: {0}")]
    Runtime(String),

    /// The template itself called `raise_exception(msg)`.
    #[error("{0}")]
    TemplateRaise(String),

    /// A safety bound was hit (recursion depth, loop iterations, output size).
    #[error("jinja limit exceeded: {0}")]
    Limit(String),
}

impl JinjaError {
    /// Build a [`JinjaError::Runtime`].
    pub fn runtime(message: impl Into<String>) -> Self {
        JinjaError::Runtime(message.into())
    }

    /// Build a [`JinjaError::Limit`].
    pub fn limit(message: impl Into<String>) -> Self {
        JinjaError::Limit(message.into())
    }

    /// The `raise_exception` message, if this error came from the template.
    pub fn template_raise_message(&self) -> Option<&str> {
        match self {
            JinjaError::TemplateRaise(msg) => Some(msg),
            _ => None,
        }
    }
}

impl From<JinjaError> for TokenizerError {
    fn from(error: JinjaError) -> Self {
        TokenizerError::TemplateRender(error.to_string())
    }
}

/// Compile- and render-time settings, including the safety bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JinjaOptions {
    /// Remove the first newline after a block tag (Jinja's `trim_blocks`).
    pub trim_blocks: bool,
    /// Remove leading whitespace from the start of a line to a block tag
    /// (Jinja's `lstrip_blocks`).
    pub lstrip_blocks: bool,
    /// Keep a single trailing newline in the source instead of dropping it.
    pub keep_trailing_newline: bool,
    /// Maximum syntactic nesting accepted by the parser.
    pub max_parse_depth: usize,
    /// Maximum block/expression nesting evaluated at render time.
    pub max_render_depth: usize,
    /// Maximum total loop iterations for one render.
    pub max_loop_iterations: usize,
    /// Maximum number of bytes one render may emit.
    pub max_output_bytes: usize,
}

impl Default for JinjaOptions {
    fn default() -> Self {
        Self {
            trim_blocks: true,
            lstrip_blocks: true,
            keep_trailing_newline: false,
            max_parse_depth: 128,
            max_render_depth: 256,
            max_loop_iterations: 1_000_000,
            max_output_bytes: 32 * 1024 * 1024,
        }
    }
}

/// A compiled template.
///
/// Compilation is done once (it is the expensive half); rendering takes an
/// immutable borrow, so a template can be shared behind an `Arc` across
/// requests.
#[derive(Debug, Clone)]
pub struct JinjaTemplate {
    root: Vec<Node>,
    options: JinjaOptions,
}

impl JinjaTemplate {
    /// Compile a template with the chat-template defaults.
    pub fn compile(source: &str) -> Result<Self, JinjaError> {
        Self::compile_with(source, JinjaOptions::default())
    }

    /// Compile a template with explicit options.
    pub fn compile_with(source: &str, options: JinjaOptions) -> Result<Self, JinjaError> {
        let tokens = Lexer::new(source, &options).tokenize()?;
        let root = Parser::new(&tokens, &options).parse_template()?;
        Ok(Self { root, options })
    }

    /// Render the template against a context, which must be a mapping (or a
    /// namespace); `Undefined`/`None` render against an empty context.
    pub fn render(&self, context: &Value) -> Result<String, JinjaError> {
        eval::render(&self.root, context, &self.options)
    }

    /// Convenience wrapper: parse a JSON object into a context and render.
    ///
    /// The JSON is parsed with [`Value::from_json_str`], so object key order
    /// is preserved and `tojson` reproduces it exactly.
    pub fn render_json(&self, context_json: &str) -> Result<String, JinjaError> {
        let context = Value::from_json_str(context_json)?;
        self.render(&context)
    }

    /// The options this template was compiled with.
    pub fn options(&self) -> &JinjaOptions {
        &self.options
    }

    /// The parsed root nodes (useful for diagnostics and tests).
    pub fn nodes(&self) -> &[Node] {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_reports_line_and_column() {
        let err = JinjaTemplate::compile("ok\nok\n{% if %}").expect_err("must fail");
        match err {
            JinjaError::Syntax { line, column, .. } => {
                assert_eq!(line, 3);
                assert!(column >= 1);
            }
            other => panic!("expected a syntax error, got {other:?}"),
        }
    }

    #[test]
    fn errors_convert_into_tokenizer_errors() {
        let err = TokenizerError::from(JinjaError::TemplateRaise("nope".to_string()));
        assert!(matches!(err, TokenizerError::TemplateRender(ref m) if m == "nope"));
    }

    #[test]
    fn template_is_reusable_and_shareable() {
        // A compiled template must be `Send + Sync`: B2-13 keeps it in a
        // `ChatTemplateKind::Jinja(Arc<JinjaTemplate>)` shared by every server
        // thread.  Runtime `Value`s deliberately are not — they never outlive
        // a single `render` call.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JinjaTemplate>();
        assert_send_sync::<JinjaError>();
        assert_send_sync::<JinjaOptions>();

        let shared = std::sync::Arc::new(JinjaTemplate::compile("{{ a }}").expect("compile"));
        let worker = {
            let shared = std::sync::Arc::clone(&shared);
            std::thread::spawn(move || shared.render_json(r#"{"a": 1}"#))
        };
        assert_eq!(worker.join().expect("thread joins").expect("render"), "1");
        assert_eq!(shared.render_json(r#"{"a": "x"}"#).expect("render"), "x");
    }

    #[test]
    fn options_round_trip() {
        let options = JinjaOptions {
            trim_blocks: false,
            lstrip_blocks: false,
            ..JinjaOptions::default()
        };
        let template = JinjaTemplate::compile_with("a\n    {% if true %}\nb{% endif %}", options)
            .expect("compile");
        assert_eq!(
            template.render(&Value::Undefined).expect("render"),
            "a\n    \nb"
        );
        assert!(!template.options().trim_blocks);
    }
}
