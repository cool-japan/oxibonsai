//! Canned chat-template registry covering the five major open-weight
//! instruction-tuned families, plus the real Jinja-subset rendering path
//! (B2-13 / bonsai2-design.md §5.2) used for a model's own
//! `tokenizer.chat_template` and for a from-scratch, tool-call-aware
//! ChatML/Qwen3 replacement.
//!
//! The five canned kinds ([`ChatTemplateKind`] / [`ChatMessage`] /
//! [`ChatTemplateKind::render`]) are unchanged from before this package:
//! they use only `{{ role }}` / `{{ content }}` and one `if role == "user"`,
//! which the crate's minimal [`crate::utils::render_template`] evaluator
//! renders correctly today (TOK-07 verdict correction — do not rewrite
//! them). What is new is [`ResolvedChatTemplate`], which renders through
//! the real Jinja subset engine ([`crate::jinja`]) against either a model's
//! own GGUF-embedded template or a hand-authored, real-Jinja-syntax
//! fallback per family that — unlike the canned string-substitution
//! templates above — understands `tool` role turns, `tool_calls` and
//! `reasoning_content` (RT-07/RT-09/RT-10).
//!
//! | Family   | Roles supported                  | Reference tag(s)                            |
//! |----------|----------------------------------|---------------------------------------------|
//! | ChatML   | system / user / assistant / tool | `<|im_start|>` / `<|im_end|>`               |
//! | Llama3   | system / user / assistant        | `<|start_header_id|>` / `<|eot_id|>`        |
//! | Mistral  | user / assistant                 | `[INST]` / `[/INST]`                        |
//! | Gemma    | user / assistant                 | `<start_of_turn>` / `<end_of_turn>`         |
//! | Qwen     | system / user / assistant / tool | `<|im_start|>` / `<|im_end|>` + `<|endoftext|>` |

use std::sync::{Arc, OnceLock};

use crate::jinja::{JinjaError, JinjaTemplate, Value, ValueMap};
use crate::{error::TokenizerResult, utils::render_template};

// ── ChatMessage ──────────────────────────────────────────────────────────────

/// A single chat-turn used by [`ChatTemplateKind::render`].
///
/// The lifetime ties the message to the caller's storage so no extra
/// allocations are needed during rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMessage<'a> {
    /// Conventional role: `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: &'a str,
    /// Message content (raw, not pre-tokenized).
    pub content: &'a str,
}

impl<'a> ChatMessage<'a> {
    /// Convenience constructor.
    pub fn new(role: &'a str, content: &'a str) -> Self {
        Self { role, content }
    }

    /// Short-hand for a user message.
    pub fn user(content: &'a str) -> Self {
        Self::new("user", content)
    }

    /// Short-hand for an assistant message.
    pub fn assistant(content: &'a str) -> Self {
        Self::new("assistant", content)
    }

    /// Short-hand for a system message.
    pub fn system(content: &'a str) -> Self {
        Self::new("system", content)
    }
}

// ── ChatTemplateKind ─────────────────────────────────────────────────────────

/// Identifies one of the built-in chat-template families.
///
/// Use [`Self::render`] to format a sequence of [`ChatMessage`]s into the
/// canonical prompt string for that family.  The returned string is ready to
/// be passed to [`crate::OxiTokenizer::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChatTemplateKind {
    /// OpenAI-style ChatML: `<|im_start|>role\ncontent<|im_end|>\n…`.
    ChatML,
    /// Llama-3 Instruct: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`.
    Llama3,
    /// Mistral Instruct: `<s>[INST] user [/INST] assistant </s>`.
    Mistral,
    /// Gemma Instruct: `<start_of_turn>role\ncontent<end_of_turn>`.
    Gemma,
    /// Qwen series: Same tags as ChatML plus `<|endoftext|>` trailer.
    Qwen,
}

impl ChatTemplateKind {
    /// Return the raw Jinja-lite template string for this family.
    pub fn template(&self) -> &'static str {
        match self {
            Self::ChatML => CHATML_TEMPLATE,
            Self::Llama3 => LLAMA3_TEMPLATE,
            Self::Mistral => MISTRAL_TEMPLATE,
            Self::Gemma => GEMMA_TEMPLATE,
            Self::Qwen => QWEN_TEMPLATE,
        }
    }

    /// Render a list of [`ChatMessage`]s into a prompt string.
    pub fn render(&self, messages: &[ChatMessage<'_>]) -> String {
        let pairs: Vec<(&str, &str)> = messages.iter().map(|m| (m.role, m.content)).collect();
        render_template(self.template(), &pairs)
    }

    /// Render messages and append an assistant-generation prompt (the opener
    /// that tells the model "your turn").  For ChatML this is
    /// `<|im_start|>assistant\n`; for Llama-3 `<|start_header_id|>assistant<|end_header_id|>\n\n`.
    pub fn render_with_generation_prompt(&self, messages: &[ChatMessage<'_>]) -> String {
        let mut out = self.render(messages);
        out.push_str(self.generation_prompt());
        out
    }

    /// The tag that follows the last user message to invite an assistant
    /// response.  Empty for families that don't need one.
    pub fn generation_prompt(&self) -> &'static str {
        match self {
            Self::ChatML | Self::Qwen => "<|im_start|>assistant\n",
            Self::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            Self::Mistral => "",
            Self::Gemma => "<start_of_turn>model\n",
        }
    }

    /// Tokenize this family's prompt directly with a tokenizer.
    ///
    /// Equivalent to `tok.encode(&kind.render(msgs))` but kept as a helper
    /// method for symmetry with the HF `apply_chat_template` API.
    pub fn encode(
        &self,
        tokenizer: &crate::OxiTokenizer,
        messages: &[ChatMessage<'_>],
    ) -> TokenizerResult<Vec<u32>> {
        tokenizer.encode(&self.render(messages))
    }

    /// List all kinds that this crate knows about.  Useful for testing and
    /// for building UI pickers.
    pub fn all() -> &'static [ChatTemplateKind] {
        &[
            Self::ChatML,
            Self::Llama3,
            Self::Mistral,
            Self::Gemma,
            Self::Qwen,
        ]
    }

    /// Infer a template kind from a model name heuristic.  Returns `None` if
    /// no family is recognised.
    pub fn infer_from_name(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase();
        if n.contains("llama-3") || n.contains("llama3") {
            Some(Self::Llama3)
        } else if n.contains("mistral") {
            Some(Self::Mistral)
        } else if n.contains("gemma") {
            Some(Self::Gemma)
        } else if n.contains("qwen") {
            Some(Self::Qwen)
        } else if n.contains("chatml") {
            Some(Self::ChatML)
        } else {
            None
        }
    }

    /// The real-Jinja-syntax replacement for this family, used by
    /// [`ResolvedChatTemplate::render_with`] — unlike [`Self::render`]
    /// (the minimal evaluator above), this understands `tool`-role turns
    /// and `message.tool_calls` (RT-07), matching the shape the real
    /// Bonsai 2 template teaches the model to emit
    /// (`<tool_call><function=NAME><parameter=KEY>…`).
    ///
    /// Compiled once per family and cached: every one of these constants is
    /// fixed and hand-verified (see the `fallback_*_compiles` tests), so a
    /// compile failure here can only be this crate's own regression, not
    /// attacker- or model-supplied input.
    pub fn fallback_jinja_template(&self) -> Arc<JinjaTemplate> {
        fn cached(
            cell: &'static OnceLock<Arc<JinjaTemplate>>,
            source: &'static str,
        ) -> Arc<JinjaTemplate> {
            Arc::clone(cell.get_or_init(|| {
                Arc::new(JinjaTemplate::compile(source).expect(
                    "built-in fallback chat templates are fixed, hand-verified constants \
                     covered by fallback_*_compiles tests",
                ))
            }))
        }
        static CHATML_QWEN: OnceLock<Arc<JinjaTemplate>> = OnceLock::new();
        static LLAMA3: OnceLock<Arc<JinjaTemplate>> = OnceLock::new();
        static MISTRAL: OnceLock<Arc<JinjaTemplate>> = OnceLock::new();
        static GEMMA: OnceLock<Arc<JinjaTemplate>> = OnceLock::new();
        match self {
            Self::ChatML | Self::Qwen => cached(&CHATML_QWEN, FALLBACK_CHATML_JINJA),
            Self::Llama3 => cached(&LLAMA3, FALLBACK_LLAMA3_JINJA),
            Self::Mistral => cached(&MISTRAL, FALLBACK_MISTRAL_JINJA),
            Self::Gemma => cached(&GEMMA, FALLBACK_GEMMA_JINJA),
        }
    }
}

// ── Template strings ─────────────────────────────────────────────────────────

const CHATML_TEMPLATE: &str =
    "{% for message in messages %}<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n{% endfor %}";

const LLAMA3_TEMPLATE: &str = concat!(
    "<|begin_of_text|>",
    "{% for message in messages %}",
    "<|start_header_id|>{{ role }}<|end_header_id|>\n\n",
    "{{ content }}<|eot_id|>",
    "{% endfor %}"
);

// Mistral needs an if/else to distinguish user from assistant turns.
// The minimal evaluator supports `{% if role == "user" %} … {% else %} … {% endif %}`.
const MISTRAL_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "{% if role == \"user\" %}<s>[INST] {{ content }} [/INST]{% else %} {{ content }}</s>{% endif %}",
    "{% endfor %}"
);

const GEMMA_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "<start_of_turn>{{ role }}\n{{ content }}<end_of_turn>\n",
    "{% endfor %}"
);

const QWEN_TEMPLATE: &str =
    "{% for message in messages %}<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n{% endfor %}";

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kinds_yield_a_template() {
        for k in ChatTemplateKind::all() {
            assert!(!k.template().is_empty(), "template for {k:?} empty");
        }
    }

    #[test]
    fn chatml_renders_basic() {
        let out = ChatTemplateKind::ChatML.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("hi"));
        assert!(out.contains("<|im_end|>"));
    }

    #[test]
    fn llama3_renders_basic() {
        let out = ChatTemplateKind::Llama3.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(out.contains("<|eot_id|>"));
    }

    #[test]
    fn mistral_renders_basic() {
        let out = ChatTemplateKind::Mistral
            .render(&[ChatMessage::user("hi"), ChatMessage::assistant("there")]);
        assert!(out.contains("[INST] hi [/INST]"));
        assert!(out.contains("there"));
    }

    #[test]
    fn gemma_renders_basic() {
        let out = ChatTemplateKind::Gemma.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<start_of_turn>user"));
        assert!(out.contains("<end_of_turn>"));
    }

    #[test]
    fn qwen_renders_basic() {
        let out = ChatTemplateKind::Qwen.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("<|im_end|>"));
    }

    #[test]
    fn generation_prompt_chatml() {
        let p = ChatTemplateKind::ChatML.generation_prompt();
        assert!(p.contains("assistant"));
    }

    #[test]
    fn render_with_generation_prompt() {
        let out =
            ChatTemplateKind::ChatML.render_with_generation_prompt(&[ChatMessage::user("hi")]);
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn infer_from_name_known() {
        assert_eq!(
            ChatTemplateKind::infer_from_name("Qwen3-1.7B"),
            Some(ChatTemplateKind::Qwen)
        );
        assert_eq!(
            ChatTemplateKind::infer_from_name("Meta-Llama-3-8B-Instruct"),
            Some(ChatTemplateKind::Llama3)
        );
        assert_eq!(
            ChatTemplateKind::infer_from_name("mistral-7b"),
            Some(ChatTemplateKind::Mistral)
        );
        assert_eq!(
            ChatTemplateKind::infer_from_name("gemma-2b"),
            Some(ChatTemplateKind::Gemma)
        );
    }

    #[test]
    fn infer_from_name_unknown() {
        assert_eq!(ChatTemplateKind::infer_from_name("bert-base"), None);
    }

    #[test]
    fn encode_works_with_stub() {
        let tok = crate::OxiTokenizer::char_level_stub(256);
        let ids = ChatTemplateKind::ChatML
            .encode(&tok, &[ChatMessage::user("hi")])
            .expect("encode ok");
        assert!(!ids.is_empty());
    }

    #[test]
    fn chat_message_constructors() {
        let u = ChatMessage::user("x");
        assert_eq!(u.role, "user");
        let a = ChatMessage::assistant("y");
        assert_eq!(a.role, "assistant");
        let s = ChatMessage::system("z");
        assert_eq!(s.role, "system");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Real Jinja-subset rendering (B2-13, bonsai2-design.md §5.2)
// ═══════════════════════════════════════════════════════════════════════════

/// One chat turn for [`ResolvedChatTemplate::render_with`].
///
/// Distinct from [`ChatMessage`] (which the five canned, string-substitution
/// templates use and must keep its `Copy`, two-field shape — existing
/// `chat_template_tests.rs` construct and compare it directly): the real
/// Jinja engine needs the fuller OpenAI message shape the design's own
/// `apply_template.json` golden exercises (`reasoning_content`, assistant
/// `tool_calls`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderMessage {
    /// Conventional role: `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// Message text. Never `None` at this layer — a tool-calls-only
    /// assistant turn with no natural-language preamble uses `""`, which
    /// renders identically to the reference template's own `content is
    /// none` branch (both emit nothing for the content slot).
    pub content: String,
    /// The `<think>…</think>` span already produced for this turn (a past
    /// assistant message being re-rendered back into a follow-up prompt).
    pub reasoning_content: Option<String>,
    /// Raw JSON **text** of this assistant message's OpenAI-shaped
    /// `tool_calls` array (e.g.
    /// `[{"type":"function","function":{"name":"f","arguments":{...}}}]`),
    /// if any. Kept as text — never a `serde_json::Value` — for the same
    /// key-order reason as [`RenderOptions::tools`]; see that field's doc.
    pub tool_calls: Option<String>,
    /// The id of the tool call this `tool`-role message answers (unused by
    /// every template this crate renders today — the real Bonsai 2
    /// template keys tool-response merging on adjacency, not on id — but
    /// carried for symmetry with `crate::server::ChatMessage` and for a
    /// future template that does key off it).
    pub tool_call_id: Option<String>,
}

impl RenderMessage {
    /// Construct a plain text turn.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    /// Attach reasoning content (assistant turns only; ignored by every
    /// template's role-specific branch otherwise).
    #[must_use]
    pub fn with_reasoning_content(mut self, reasoning_content: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning_content.into());
        self
    }

    /// Attach a `tool_calls` array as raw JSON text (see the field doc for
    /// why text, not a `serde_json::Value`).
    #[must_use]
    pub fn with_tool_calls_json(mut self, tool_calls_json: impl Into<String>) -> Self {
        self.tool_calls = Some(tool_calls_json.into());
        self
    }

    /// Attach the id of the tool call this message answers.
    #[must_use]
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }
}

/// Options controlling one [`ResolvedChatTemplate::render_with`] call
/// (bonsai2-design.md §5.2's `RenderOptions`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderOptions {
    /// Append the assistant-turn opener (`true` for inference; `false` when
    /// re-rendering a transcript for e.g. training-data export).
    pub add_generation_prompt: bool,
    /// `chat_template_kwargs.enable_thinking`. `None` = template default
    /// (the real Bonsai 2 template treats an absent value as `true`).
    pub enable_thinking: Option<bool>,
    /// `chat_template_kwargs.reasoning_effort`: `"xhigh"` / `"medium"` /
    /// `"low"`. `None` = template default (`"xhigh"`).
    pub reasoning_effort: Option<String>,
    /// `chat_template_kwargs.preserve_thinking`. `None` = template default
    /// (`true`).
    pub preserve_thinking: Option<bool>,
    /// Whether multi-image/video turns get a `"Picture N: "` / `"Video N:
    /// "` label (vision, phase 2 — accepted here for API completeness;
    /// no template this crate renders today has image/video content parts
    /// to label).
    pub add_vision_id: bool,
    /// Raw JSON **text** of the OpenAI-shaped `tools` array (e.g.
    /// `[{"type":"function","function":{"name":...,"description":...,
    /// "parameters":{...}}}]`), fed to [`Value::from_json_str`] so the
    /// template's `tool | tojson` reproduces the caller's own key order
    /// byte-for-byte.
    ///
    /// **Design-signature deviation, recorded deliberately.**
    /// bonsai2-design.md §5.2 types this field `Option<serde_json::Value>`.
    /// That type cannot pass this package's own G7 gate: the workspace's
    /// `serde_json` (root `Cargo.toml`, not in this package's `owned_files`)
    /// has neither `preserve_order` nor `raw_value` enabled, so
    /// `serde_json::Value`'s `Map` is a `BTreeMap` — going through it
    /// *always* re-sorts object keys alphabetically, regardless of how
    /// carefully a caller builds the `Value` (see
    /// `crate::jinja::value::Value`'s `From<&serde_json::Value>` doc, which
    /// documents this exact hazard and is pinned by
    /// `jinja_tool_key_order_requires_from_json_str`). Storing the JSON as
    /// **text** and parsing it with `Value::from_json_str` — this crate's
    /// own hand-written, order-preserving `Deserialize` — is the only way
    /// to reproduce the reference renderer's byte output without either (a)
    /// a workspace-wide `Cargo.toml` edit outside this package's
    /// `owned_files` (turning on `serde_json/preserve_order`, which would
    /// also reorder every other consumer's GGUF-metadata `Value`s — a
    /// bigger, cross-cutting change than one package should make
    /// unilaterally) or (b) `serde_json::value::RawValue`, which needs the
    /// `raw_value` feature the same way. Recorded in this package's
    /// `deviations`.
    ///
    /// **Residual gap this does not close**, also recorded: a live HTTP
    /// request's `tools[].function.parameters` (an arbitrary, caller-supplied
    /// JSON Schema) is deserialized into a typed Rust request struct
    /// (`crate::api_types`-equivalent in `oxibonsai-runtime`) *before* it
    /// ever reaches this field, and that deserialization step already went
    /// through a `serde_json::Value` for the schema body — so by the time
    /// the caller re-serializes it back to text for this field, the
    /// *nested* schema's own key order is already unrecoverable (the
    /// top-level `type`/`function`/`name`/`description`/`parameters`
    /// ordering is still correct, because those come from a Rust struct's
    /// fixed field-declaration order, not from a re-sorted map). Only a
    /// workspace-wide `preserve_order`, or capturing the raw request body
    /// bytes ahead of typed deserialization, closes that residual gap.
    pub tools: Option<String>,
}

/// A template resolved and ready to render through the real Jinja engine —
/// either a model's own `tokenizer.chat_template`, or a real-Jinja-syntax
/// stand-in for one of the five canned families above.
///
/// Kept **separate** from [`ChatTemplateKind`] rather than adding a
/// `Jinja(Arc<JinjaTemplate>)` variant to it (bonsai2-design.md §5.2's
/// literal signature): `ChatTemplateKind` derives `Copy, PartialEq, Eq` and
/// `chat_template_tests.rs` (not owned by this package) relies on all
/// three — `assert_eq!(ChatTemplateKind::infer_from_name(..),
/// Some(ChatTemplateKind::Qwen))`, `ChatTemplateKind::all()` returning a
/// `&'static [ChatTemplateKind]` slice by value, etc. `Arc<JinjaTemplate>`
/// has no `PartialEq` (a `JinjaTemplate` is not meaningfully comparable) and
/// is not `Copy`, so adding it as a variant would force dropping all three
/// derives and break that test file, which this package may not weaken.
/// Recorded as a design-signature deviation.
#[derive(Debug, Clone)]
pub enum ResolvedChatTemplate {
    /// One of the five canned families, rendered through its real-Jinja
    /// [`ChatTemplateKind::fallback_jinja_template`] rather than the
    /// minimal evaluator (so `tool`/`tool_calls` render correctly).
    Canned(ChatTemplateKind),
    /// A model's own compiled `tokenizer.chat_template`.
    Jinja(Arc<JinjaTemplate>),
}

impl ResolvedChatTemplate {
    /// Prefer the GGUF's own `tokenizer.chat_template`; fall back to the
    /// built-in Qwen3/ChatML replacement only for a model that ships **no**
    /// `tokenizer.chat_template` at all.
    ///
    /// **Never renders garbage** (TOK-07/RT-09, spec item 1 "an unsupported
    /// construct must ERROR"): a template this engine's Jinja subset
    /// **cannot compile** is a loud, caller-visible [`Err`], never a silent
    /// substitution — B5 correction. An earlier revision of this function
    /// swapped in the fallback whenever compilation failed, with only a
    /// `tracing::warn!`; that conflated two very different situations
    /// ("no template shipped" — a legitimate, expected case the fallback
    /// exists for) and ("a template shipped but this engine's Jinja subset
    /// rejects it" — an engine-capability gap that must surface as an
    /// operator-visible error, not a prompt silently rendered against the
    /// *wrong* template the model was never tuned against). Both shipped
    /// real templates (Bonsai 2 27B and the legacy Bonsai-8B Qwen3 one)
    /// compile cleanly against this engine, so this tightening is safe for
    /// every model this crate ships against today.
    pub fn from_gguf(md: &oxibonsai_core::MetadataStore) -> Result<Self, JinjaError> {
        match crate::gguf_vocab::chat_template_from_gguf_metadata(md) {
            Some(text) => {
                let tpl = JinjaTemplate::compile(&text)?;
                Ok(ResolvedChatTemplate::Jinja(Arc::new(tpl)))
            }
            None => Ok(Self::default_fallback()),
        }
    }

    /// The named fallback for a model that ships no `tokenizer.chat_template`
    /// (bonsai2-design.md §5.2, spec item 1): the real-Jinja-syntax
    /// ChatML/Qwen3 replacement, which — unlike the minimal
    /// [`ChatTemplateKind::Qwen`] evaluator path — renders `tool` turns and
    /// `tool_calls` correctly (RT-07's correction: "default to the Qwen3
    /// form for the 1.7B/8B models actually shipped").
    pub fn default_fallback() -> Self {
        ResolvedChatTemplate::Canned(ChatTemplateKind::Qwen)
    }

    /// The compiled template this will render through.
    fn template(&self) -> Arc<JinjaTemplate> {
        match self {
            ResolvedChatTemplate::Jinja(tpl) => Arc::clone(tpl),
            ResolvedChatTemplate::Canned(kind) => kind.fallback_jinja_template(),
        }
    }

    /// Render `messages` under `opts` through the real Jinja engine.
    ///
    /// Returns `Err` — never a partial or "best effort" string — when the
    /// template raises (e.g. an empty `messages` list: every template this
    /// module renders raises `raise_exception('No messages provided.')`, a
    /// [`JinjaError::TemplateRaise`], for that case) or when `opts.tools` /
    /// a message's `tool_calls` is not valid JSON.
    pub fn render_with(
        &self,
        messages: &[RenderMessage],
        opts: &RenderOptions,
    ) -> Result<String, JinjaError> {
        let ctx = build_context(messages, opts)?;
        self.template().render(&ctx)
    }
}

/// Build the Jinja render context: `{"messages": [...], "tools": [...],
/// "enable_thinking": ..., "reasoning_effort": ..., "preserve_thinking":
/// ..., "add_vision_id": ..., "add_generation_prompt": ...}`.
///
/// A key is omitted entirely (rather than set to `Value::None`) whenever
/// the corresponding [`RenderOptions`] / [`RenderMessage`] field is `None`,
/// so the template's own `is undefined` / bare-truthiness checks (`{%- if
/// enable_thinking is undefined or ... %}`, `{%- if message.tool_calls and
/// ... %}`) see the same "not provided at all" state the reference renderer
/// does — `Value::None` and "absent" are deliberately different states in
/// this value model (see `crate::jinja::value`).
fn build_context(messages: &[RenderMessage], opts: &RenderOptions) -> Result<Value, JinjaError> {
    let mut msg_values = Vec::with_capacity(messages.len());
    for message in messages {
        let mut entry = ValueMap::new();
        entry.insert("role", Value::str(&message.role));
        entry.insert("content", Value::str(&message.content));
        if let Some(reasoning_content) = &message.reasoning_content {
            entry.insert("reasoning_content", Value::str(reasoning_content));
        }
        if let Some(tool_calls_json) = &message.tool_calls {
            entry.insert("tool_calls", Value::from_json_str(tool_calls_json)?);
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            entry.insert("tool_call_id", Value::str(tool_call_id));
        }
        msg_values.push(Value::map(entry));
    }

    let mut ctx = ValueMap::new();
    ctx.insert("messages", Value::list(msg_values));
    if let Some(tools_json) = &opts.tools {
        ctx.insert("tools", Value::from_json_str(tools_json)?);
    }
    if let Some(enable_thinking) = opts.enable_thinking {
        ctx.insert("enable_thinking", Value::Bool(enable_thinking));
    }
    if let Some(reasoning_effort) = &opts.reasoning_effort {
        ctx.insert("reasoning_effort", Value::str(reasoning_effort));
    }
    if let Some(preserve_thinking) = opts.preserve_thinking {
        ctx.insert("preserve_thinking", Value::Bool(preserve_thinking));
    }
    ctx.insert("add_vision_id", Value::Bool(opts.add_vision_id));
    ctx.insert(
        "add_generation_prompt",
        Value::Bool(opts.add_generation_prompt),
    );
    Ok(Value::map(ctx))
}

// ── Fallback Jinja templates ─────────────────────────────────────────────────
//
// Real Jinja syntax (not the minimal evaluator's Jinja-lite subset above),
// compiled once and cached by `ChatTemplateKind::fallback_jinja_template`.
// The ChatML/Qwen one mirrors the real Bonsai 2 template's `tool` / assistant
// `tool_calls` handling (`fork`'s reference shape, §5.4) exactly, so a
// non-Bonsai-2 ChatML/Qwen model that ships no `tokenizer.chat_template`
// still gets correct tool-call round-tripping instead of the dropped/
// mis-rendered turns RT-07 found. Llama-3/Mistral/Gemma keep the simpler
// system/user/assistant shape their canned constants already had — no
// finding in this package's scope requires more from those three families.

const FALLBACK_CHATML_JINJA: &str = r#"{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == "system" %}
        {{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\n' + message.content }}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if message.content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' + message.content + '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

const FALLBACK_LLAMA3_JINJA: &str = r#"{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{{- '<|begin_of_text|>' }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message.role + '<|end_header_id|>\n\n' + message.content + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"#;

const FALLBACK_MISTRAL_JINJA: &str = r#"{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<s>[INST] ' + message.content + ' [/INST]' }}
    {%- else %}
        {{- ' ' + message.content + '</s>' }}
    {%- endif %}
{%- endfor %}"#;

const FALLBACK_GEMMA_JINJA: &str = r#"{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- for message in messages %}
    {{- '<start_of_turn>' + message.role + '\n' + message.content + '<end_of_turn>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<start_of_turn>model\n' }}
{%- endif %}"#;

// ── Tests: real Jinja rendering, G7 golden, fallbacks ────────────────────────

#[cfg(test)]
mod resolved_template_tests {
    use super::*;

    // bonsai2-design.md §5.2's real `chat_template.jinja` (176 lines,
    // `scratchpad/hf/mlx/chat_template.jinja` — a session-scratchpad path
    // that will not exist once this patch is merged) and the 5 golden
    // input/output pairs of `scratchpad/golden2/apply_template.json` (G7),
    // embedded verbatim below rather than read from either path at test
    // time (no-absolute-paths / no-scratchpad-dependency policy).
    // Regenerated byte-for-byte from the reference files by a small script
    // (`json.dumps`/literal-escaping), not hand-transcribed, to rule out
    // transcription error in a 176-line template and 5 multi-hundred-byte
    // expected strings.

    pub(crate) const BONSAI2_REAL_TEMPLATE_FIXTURE: &str = r######"{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"######;

    pub(crate) const GOLDEN_CASE_0_EXPECTED: &str = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n";

    pub(crate) const GOLDEN_CASE_1_EXPECTED: &str =
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    pub(crate) const GOLDEN_CASE_2_EXPECTED: &str = "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n";

    pub(crate) const GOLDEN_CASE_3_EXPECTED: &str = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nuser greets\n</think>\n\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n";

    pub(crate) const GOLDEN_CASE_4_EXPECTED: &str = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\n<think>\n";
    pub(crate) const GOLDEN_CASE_4_TOOLS_JSON: &str = "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}]";

    fn compile_real_template() -> JinjaTemplate {
        JinjaTemplate::compile(BONSAI2_REAL_TEMPLATE_FIXTURE)
            .expect("the real Bonsai 2 template must compile against this engine (G7)")
    }

    // ── G7: byte-identical rendering for all 5 golden cases ──────────────

    #[test]
    fn g7_case_0_default_xhigh_reasoning() {
        let tpl = ResolvedChatTemplate::Jinja(Arc::new(compile_real_template()));
        let messages = vec![RenderMessage::new("user", "What is 2+2? Answer briefly.")];
        let opts = RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        assert_eq!(out, GOLDEN_CASE_0_EXPECTED);
    }

    #[test]
    fn g7_case_1_enable_thinking_false() {
        let tpl = ResolvedChatTemplate::Jinja(Arc::new(compile_real_template()));
        let messages = vec![RenderMessage::new("user", "What is 2+2?")];
        let opts = RenderOptions {
            add_generation_prompt: true,
            enable_thinking: Some(false),
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        assert_eq!(out, GOLDEN_CASE_1_EXPECTED);
    }

    #[test]
    fn g7_case_2_reasoning_effort_low() {
        let tpl = ResolvedChatTemplate::Jinja(Arc::new(compile_real_template()));
        let messages = vec![RenderMessage::new("user", "What is 2+2?")];
        let opts = RenderOptions {
            add_generation_prompt: true,
            reasoning_effort: Some("low".to_string()),
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        assert_eq!(out, GOLDEN_CASE_2_EXPECTED);
    }

    #[test]
    fn g7_case_3_multi_turn_with_reasoning_content() {
        let tpl = ResolvedChatTemplate::Jinja(Arc::new(compile_real_template()));
        let messages = vec![
            RenderMessage::new("system", "You are a helpful assistant"),
            RenderMessage::new("user", "Hi"),
            RenderMessage::new("assistant", "Hello!").with_reasoning_content("user greets"),
            RenderMessage::new("user", "Bye"),
        ];
        let opts = RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        assert_eq!(out, GOLDEN_CASE_3_EXPECTED);
    }

    #[test]
    fn g7_case_4_tools_preserve_key_order() {
        let tpl = ResolvedChatTemplate::Jinja(Arc::new(compile_real_template()));
        let messages = vec![RenderMessage::new("user", "weather?")];
        let opts = RenderOptions {
            add_generation_prompt: true,
            tools: Some(GOLDEN_CASE_4_TOOLS_JSON.to_string()),
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        assert_eq!(out, GOLDEN_CASE_4_EXPECTED);
    }

    #[test]
    fn real_template_raises_on_empty_messages() {
        let tpl = ResolvedChatTemplate::Jinja(Arc::new(compile_real_template()));
        let opts = RenderOptions::default();
        let err = tpl.render_with(&[], &opts).expect_err("must raise");
        assert_eq!(
            err.template_raise_message(),
            Some("No messages provided."),
            "got: {err:?}"
        );
    }

    // ── Fallback templates compile (invariant `fallback_jinja_template`'s
    //    `.expect()` relies on) and render tool/tool_calls correctly ───────

    #[test]
    fn fallback_chatml_compiles() {
        JinjaTemplate::compile(FALLBACK_CHATML_JINJA).expect("must compile");
    }

    #[test]
    fn fallback_llama3_compiles() {
        JinjaTemplate::compile(FALLBACK_LLAMA3_JINJA).expect("must compile");
    }

    #[test]
    fn fallback_mistral_compiles() {
        JinjaTemplate::compile(FALLBACK_MISTRAL_JINJA).expect("must compile");
    }

    #[test]
    fn fallback_gemma_compiles() {
        JinjaTemplate::compile(FALLBACK_GEMMA_JINJA).expect("must compile");
    }

    #[test]
    fn fallback_jinja_template_is_cached_same_arc() {
        let a = ChatTemplateKind::Qwen.fallback_jinja_template();
        let b = ChatTemplateKind::Qwen.fallback_jinja_template();
        assert!(Arc::ptr_eq(&a, &b), "must return the cached instance");
    }

    #[test]
    fn default_fallback_renders_qwen_style() {
        let tpl = ResolvedChatTemplate::default_fallback();
        let messages = vec![RenderMessage::new("user", "hi")];
        let opts = RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        assert_eq!(
            out,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn fallback_renders_empty_error() {
        let tpl = ResolvedChatTemplate::default_fallback();
        let err = tpl
            .render_with(&[], &RenderOptions::default())
            .expect_err("must raise");
        assert!(matches!(err, JinjaError::TemplateRaise(_)));
    }

    // ── RT-07: tool role + assistant tool_calls round-trip through the
    //    fallback (the actually-shipped 1.7B/8B Qwen3 models' path) ───────

    #[test]
    fn fallback_renders_tool_response_wrapped_in_user_turn() {
        let tpl = ResolvedChatTemplate::default_fallback();
        let messages = vec![
            RenderMessage::new("user", "weather in Tokyo?"),
            RenderMessage::new("assistant", "").with_tool_calls_json(
                r#"[{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]"#,
            ),
            RenderMessage::new("tool", "{\"temp_c\": 21}"),
        ];
        let opts = RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        // Never the old, bare (no ChatML wrapper) `_` arm rendering.
        assert!(
            out.contains(
                "<|im_start|>user\n<tool_response>\n{\"temp_c\": 21}\n</tool_response><|im_end|>\n"
            ),
            "tool role must render as a wrapped user turn, got: {out:?}"
        );
        assert!(
            out.contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"),
            "assistant tool_calls must round-trip into the XML shape, got: {out:?}"
        );
    }

    #[test]
    fn fallback_merges_consecutive_tool_responses_into_one_user_turn() {
        let tpl = ResolvedChatTemplate::default_fallback();
        let messages = vec![
            RenderMessage::new("user", "compare weather"),
            RenderMessage::new("assistant", "").with_tool_calls_json(
                r#"[{"function":{"name":"a","arguments":{}}},{"function":{"name":"b","arguments":{}}}]"#,
            ),
            RenderMessage::new("tool", "A=1"),
            RenderMessage::new("tool", "B=2"),
        ];
        let opts = RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        };
        let out = tpl.render_with(&messages, &opts).expect("render");
        // Exactly one opener for the two consecutive tool turns, not one per turn.
        assert_eq!(out.matches("<|im_start|>user\n<tool_response>").count(), 1);
        assert!(out.contains("A=1"));
        assert!(out.contains("B=2"));
    }

    // ── RenderMessage / RenderOptions plumbing ────────────────────────────

    #[test]
    fn render_message_builders_round_trip() {
        let m = RenderMessage::new("assistant", "hi")
            .with_reasoning_content("because")
            .with_tool_calls_json("[]")
            .with_tool_call_id("call_1");
        assert_eq!(m.role, "assistant");
        assert_eq!(m.content, "hi");
        assert_eq!(m.reasoning_content.as_deref(), Some("because"));
        assert_eq!(m.tool_calls.as_deref(), Some("[]"));
        assert_eq!(m.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn render_options_default_is_no_generation_prompt() {
        let opts = RenderOptions::default();
        assert!(!opts.add_generation_prompt);
        assert!(opts.enable_thinking.is_none());
        assert!(opts.tools.is_none());
    }

    #[test]
    fn build_context_errors_on_invalid_tools_json() {
        let messages = vec![RenderMessage::new("user", "hi")];
        let opts = RenderOptions {
            tools: Some("not json".to_string()),
            ..Default::default()
        };
        let err = build_context(&messages, &opts).expect_err("must error");
        assert!(matches!(err, JinjaError::Runtime(_)));
    }

    #[test]
    fn build_context_errors_on_invalid_tool_calls_json() {
        let messages = vec![RenderMessage::new("assistant", "x").with_tool_calls_json("{bad")];
        let err = build_context(&messages, &RenderOptions::default()).expect_err("must error");
        assert!(matches!(err, JinjaError::Runtime(_)));
    }

    // ── from_gguf ──────────────────────────────────────────────────────────

    #[test]
    fn from_gguf_falls_back_when_no_template_present() {
        let md = oxibonsai_core::MetadataStore::new();
        let resolved =
            ResolvedChatTemplate::from_gguf(&md).expect("no template -> fallback, not an error");
        let out = resolved
            .render_with(
                &[RenderMessage::new("user", "hi")],
                &RenderOptions {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .expect("render");
        assert_eq!(
            out,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn from_gguf_uses_the_real_template_when_present() {
        // Exercises `gguf_vocab::chat_template_from_gguf_metadata` end to end
        // (the seam this function is built on) with the real Bonsai 2
        // template, proving `from_gguf` picks it up rather than only ever
        // falling back.
        let md = build_metadata_with_chat_template(BONSAI2_REAL_TEMPLATE_FIXTURE);
        let resolved =
            ResolvedChatTemplate::from_gguf(&md).expect("the real template must compile");
        assert!(matches!(resolved, ResolvedChatTemplate::Jinja(_)));
        let out = resolved
            .render_with(
                &[RenderMessage::new("user", "What is 2+2? Answer briefly.")],
                &RenderOptions {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .expect("render");
        assert_eq!(out, GOLDEN_CASE_0_EXPECTED);
    }

    #[test]
    fn from_gguf_errors_on_uncompilable_template() {
        // B5 correction: a model that SHIPS a `tokenizer.chat_template` this
        // engine's Jinja subset cannot compile must surface a loud `Err`
        // (spec item 1: "an unsupported construct must ERROR"), never
        // silently substitute the fallback template — that would render a
        // real prompt against a template the model was never tuned
        // against, with only a log line as the (easily-missed) signal.
        let md = build_metadata_with_chat_template("{% this is not valid jinja %}");
        let err = ResolvedChatTemplate::from_gguf(&md)
            .expect_err("an uncompilable shipped template must error, not silently fall back");
        // A real compile failure, not some other unrelated error shape.
        assert!(
            !err.to_string().is_empty(),
            "the error must carry a real diagnostic"
        );
    }

    /// Build a minimal, well-formed `MetadataStore` carrying only
    /// `tokenizer.chat_template = template`, mirroring
    /// `gguf_vocab.rs`'s own `make_metadata` test helper's wire-format
    /// construction (that helper is private to its module, so this crate
    /// builds its own minimal one rather than depending on it).
    fn build_metadata_with_chat_template(template: &str) -> oxibonsai_core::MetadataStore {
        use oxibonsai_core::gguf::types::GgufValueType;

        let mut data = Vec::new();
        // key
        let key = crate::gguf_vocab::KEY_CHAT_TEMPLATE.as_bytes();
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        // value type: string
        data.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        // value: string (len + bytes)
        let value = template.as_bytes();
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value);

        let (store, _) = oxibonsai_core::MetadataStore::parse(&data, 0, 1)
            .expect("well-formed single-entry metadata block");
        store
    }
}
