//! Shared chat-prompt rendering pipeline (B2-13, bonsai2-design.md §5.2-§5.3).
//!
//! Renders a model's own chat template ([`TokenizerBridge::resolved_chat_template`])
//! through the real Jinja engine and encodes the result in ONE whole-prompt
//! `encode` call, matching the reference renderer's own tokenization
//! exactly (a template can emit structure — e.g. the `<tool_call>` XML
//! block — that straddles what a per-ChatML-segment encoder would treat as
//! separate boundaries; G7 byte-identity needs the WHOLE rendered string
//! tokenized as one unit, the same way `oxibonsai chat` / any reference
//! `apply_chat_template` + `encode` pipeline does).
//!
//! # TOK-M2, adapted for whole-prompt rendering
//!
//! `server::sanitize::encode_chat_prompt` (the previous, ChatML-only
//! prompt builder this module's callers now bypass) got its TOK-M2
//! guarantee — a client's message content can never resolve to a real
//! control-token id — "for free" by construction: it encodes each
//! `PromptSegment::Template` and `PromptSegment::Content` SEPARATELY, so a
//! content segment's ids are always run through [`SpecialTokenGuard`]
//! before being spliced next to the (separately-encoded, always-trusted)
//! template ids. A single combined encode of the WHOLE rendered string —
//! which real per-model chat-template rendering fundamentally requires,
//! since the template interleaves literal markup and message content
//! character-by-character — loses that separation: nothing would stop a
//! user's literal `<tool_call>` (a real added token in the shipped
//! vocabularies) from being encoded as the model's genuine control-token
//! id once it sits inside one big string next to real template markup.
//!
//! [`render_chat_prompt`] restores the guarantee at the PER-MESSAGE-CONTENT
//! level instead of the per-ChatML-segment one: [`neutralize_message_text`]
//! validates (and, if needed, neutralizes) each message's raw text BEFORE
//! it is ever substituted into the template, so the eventual single encode
//! of the fully-rendered prompt can never let that text collapse into a
//! control-token id the surrounding template text did not itself emit. See
//! that function's own doc for the exact two-layer mechanism (the
//! `<|...|>` raw-text guard, then an id-level check against the
//! tokenizer's own added-vocabulary control ids, mirroring
//! `encode_chat_prompt`'s two layers).
//!
//! Shared between `server::chat` (the base `/v1/chat/completions`
//! endpoint) and `api_extensions` (the extended endpoint), which each
//! build the same [`RenderableMessage`] shape from their own,
//! differently-typed request message lists.

use crate::server::api_error::ApiError;
use crate::server::sanitize::{neutralize_special_markers, SpecialTokenGuard};
use crate::tokenizer_bridge::TokenizerBridge;
use oxibonsai_tokenizer::chat_templates::{RenderMessage, RenderOptions};
use oxibonsai_tokenizer::jinja::JinjaError;
use serde::Serialize;

/// Extra request fields the base (`server::chat`) and extended
/// (`api_extensions`) endpoints must both read but cannot add to their own
/// typed request structs this wave: `chat_template_kwargs.enable_thinking`
/// / `.reasoning_effort` / `.preserve_thinking` (also accepted un-nested,
/// matching real vLLM-family servers) — `ChatCompletionRequest` (`server.rs`)
/// is owned by ENGINE-SEAM this wave; `ExtendedChatRequest` (`api_types.rs`,
/// owned) COULD take these as real typed fields, but sharing one
/// implementation with the base endpoint (which cannot) is simpler than
/// two diverging ones — plus `tools` as raw JSON **text** (B4) rather than
/// a typed `Vec<ToolDefinition>`/`Vec<Tool>` field: re-serializing either
/// through `serde_json::Value` (this workspace's `serde_json` has neither
/// `preserve_order` nor is a manifest edit in either package's
/// `owned_files`) re-sorts each tool's `parameters` schema keys — measured
/// diverging from G7 case 5 at byte 393 — because the schema is ALREADY a
/// `serde_json::Value` (a `BTreeMap`-backed `Value::Object`) by the time it
/// is deserialized into the typed field, before any handler code runs; the
/// key order is unrecoverable from that point on regardless of what
/// happens downstream.
///
/// Deserialized from the SAME raw JSON bytes the typed request struct
/// itself is built from (see each endpoint's extractor) — never a second,
/// independent parse of a re-serialized value.
#[derive(Debug, Default, serde::Deserialize)]
pub(crate) struct ChatRequestExtras {
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargsWire>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// Raw JSON text of the `tools` array, when present — see the struct
    /// doc. `raw_value` is already an active `serde_json` feature for
    /// this crate (pulled in transitively through `axum`; verified via
    /// `cargo tree -e features`), so this needs no manifest change.
    #[serde(default)]
    tools: Option<Box<serde_json::value::RawValue>>,
}

#[derive(Debug, Default, Clone, serde::Deserialize)]
struct ChatTemplateKwargsWire {
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    preserve_thinking: Option<bool>,
}

impl ChatRequestExtras {
    /// `chat_template_kwargs.enable_thinking` wins over a bare top-level
    /// `enable_thinking` when both are somehow present.
    pub(crate) fn effective_enable_thinking(&self) -> Option<bool> {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking)
            .or(self.enable_thinking)
    }

    pub(crate) fn effective_reasoning_effort(&self) -> Option<String> {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.reasoning_effort.clone())
            .or_else(|| self.reasoning_effort.clone())
    }

    pub(crate) fn effective_preserve_thinking(&self) -> Option<bool> {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.preserve_thinking)
    }

    /// The `tools` array's raw JSON text, if present (B4).
    pub(crate) fn tools_raw_json(&self) -> Option<String> {
        self.tools.as_ref().map(|rv| rv.get().to_string())
    }
}

/// Patch one extra string field into a serialized chunk/response's
/// `choices[0].message` or `choices[0].delta` object — the seam that lets
/// `reasoning_content` reach the client without either endpoint's chunk/
/// message type declaring the field.
pub(crate) fn with_extra_delta_field<T: Serialize>(
    value: &T,
    pointer: &str,
    key: &str,
    text: String,
) -> String {
    let mut v = serde_json::to_value(value).unwrap_or_default();
    if let Some(obj) = v.pointer_mut(pointer).and_then(|p| p.as_object_mut()) {
        obj.insert(key.to_string(), serde_json::Value::String(text));
    }
    serde_json::to_string(&v).unwrap_or_default()
}

/// One assistant-turn tool call in the shape this pipeline needs: a
/// function name plus its RAW (still JSON-text, OpenAI wire-format)
/// arguments string — never a parsed `serde_json::Value`. See
/// [`tool_calls_to_render_json`]'s doc for why keeping it as text matters.
#[derive(Debug, Clone)]
pub(crate) struct RenderableToolCall {
    pub name: String,
    pub arguments_json_text: String,
}

/// One message in the shape this pipeline needs, independent of which
/// concrete request type (`server::ChatMessage` / the extended endpoint's
/// message list) the caller actually has.
#[derive(Debug, Clone, Default)]
pub(crate) struct RenderableMessage {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<RenderableToolCall>,
    pub tool_call_id: Option<String>,
}

/// Convert the shared `crate::server::ChatMessage` wire type (used by BOTH
/// the base `/v1/chat/completions` endpoint's own `ChatCompletionRequest`
/// and the extended endpoint's `ExtendedChatRequest`) into the
/// backend-neutral shape [`render_chat_prompt`] needs.
///
/// `reasoning_content` comes from `reasoning_contents`, not from `messages`
/// itself: `ChatMessage` (`server.rs`, owned by ENGINE-SEAM this wave) has
/// no such field to read from a replayed assistant turn, so the caller
/// recovers it from the request's own raw JSON first
/// ([`preprocess_message_content_and_reasoning`]) and passes it in here,
/// indexed the same way `messages` is (`reasoning_contents[i]` for
/// `messages[i]`; a short or absent side list — no tokenizer attached at
/// all skips the whole pre-pass — just means every message past its end
/// gets `None`, same as today). `tool_calls`' `arguments` is threaded
/// through as the RAW string OpenAI's own wire format already gives it in
/// (`FunctionCallResult::arguments: String`) — never re-parsed into a
/// `serde_json::Value`, for the same key-order reason
/// [`tool_calls_to_render_json`] documents.
pub(crate) fn to_render_messages(
    messages: &[crate::server::ChatMessage],
    reasoning_contents: &[Option<String>],
) -> Vec<RenderableMessage> {
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| RenderableMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
            reasoning_content: reasoning_contents.get(i).cloned().flatten(),
            tool_calls: m
                .tool_calls
                .as_ref()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|tc| RenderableToolCall {
                            name: tc.function.name.clone(),
                            arguments_json_text: tc.function.arguments.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect()
}

/// Pre-parse `raw_json` (a chat request body, generically, BEFORE it is
/// deserialized into either endpoint's typed request struct) to recover
/// two things `ChatMessage`'s `content: Option<String>` field (`server.rs`,
/// unowned this wave) cannot represent on its own, closing the rest of
/// `B11`/`SV-11` entirely in owned code:
///
/// 1. **Vision-shaped `content` arrays.** A client sending the OpenAI
///    multipart shape (`content: [{"type":"text",...}, {"type":"image_url",...}]`)
///    would otherwise fail `ChatMessage`'s own deserialization with an
///    opaque schema error before any handler code runs. Each message's
///    `content`, if it is a JSON array, is parsed as
///    [`crate::api_types::MessageContent`] and flattened with
///    [`crate::api_types::MessageContent::into_text_only`] — an honest
///    `400` naming the `image_url` part specifically (never a silent drop)
///    if one is present — and the flattened string is spliced back into
///    the JSON value in `content`'s place, so the typed parse that follows
///    sees the plain string it already knows how to handle.
/// 2. **Per-message `reasoning_content`.** A client replaying assistant
///    history from a reasoning-capable response includes this field on
///    that turn; `ChatMessage` has no such field, so it is captured here
///    into a side list indexed the same way `messages` is, for
///    [`to_render_messages`] to merge back in.
///
/// Returns `(rewritten_json_text, reasoning_contents)`. The rewritten text
/// is what the caller then feeds to `serde_json::from_str::<ChatCompletionRequest>`
/// (or `ExtendedChatRequest`) in place of the original body bytes; nothing
/// else in the value is touched, and callers still parse
/// [`ChatRequestExtras`] (`tools`, `chat_template_kwargs`, ...) from the
/// ORIGINAL, un-rewritten bytes — `tools_raw_json`'s key-order guarantee
/// depends on that text never round-tripping through `serde_json::Value`,
/// which re-sorts object keys (`B4`'s own finding, unrelated to this
/// rewrite's `messages`-only scope).
///
/// A body with no `messages` array at all, or a non-array `messages`, is
/// left completely untouched (`reasoning_contents` comes back empty) —
/// `ChatCompletionRequest`'s own deserialization is still what rejects
/// that shape, with its own message, not this function.
pub(crate) fn preprocess_message_content_and_reasoning(
    raw_json: &str,
) -> Result<(String, Vec<Option<String>>), ApiError> {
    let mut value: serde_json::Value = match serde_json::from_str(raw_json) {
        Ok(v) => v,
        // Malformed JSON syntax: leave it for the typed parse right after
        // this call to report with its own, already-established error
        // shape rather than duplicating that here.
        Err(_) => return Ok((raw_json.to_string(), Vec::new())),
    };
    let mut reasoning_contents = Vec::new();
    if let Some(messages) = value.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for msg in messages.iter_mut() {
            let Some(obj) = msg.as_object_mut() else {
                reasoning_contents.push(None);
                continue;
            };
            reasoning_contents.push(
                obj.get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            );
            let is_array = obj.get("content").is_some_and(serde_json::Value::is_array);
            if is_array {
                let parts_value = obj.get("content").cloned().unwrap_or_default();
                let content: crate::api_types::MessageContent = serde_json::from_value(parts_value)
                    .map_err(|e| {
                        ApiError::bad_request(format!("invalid message content: {e}"), "messages")
                    })?;
                let text = content
                    .into_text_only()
                    .map_err(|e| ApiError::bad_request(e, "messages"))?;
                obj.insert("content".to_string(), serde_json::Value::String(text));
            }
        }
    }
    let rewritten = serde_json::to_string(&value)
        .map_err(|e| ApiError::internal(format!("failed to re-serialize request body: {e}")))?;
    Ok((rewritten, reasoning_contents))
}

/// Build one assistant tool_calls array's JSON **text** from
/// [`RenderableToolCall`]s, splicing each call's raw arguments text
/// directly into the array rather than parsing it into a
/// `serde_json::Value` and re-serializing (B1(d) / B4).
///
/// Two independent reasons this must stay text-level, both measured:
///
/// 1. **Key order (B4).** This workspace's `serde_json` (root `Cargo.toml`)
///    has neither `preserve_order` nor is a parse-then-reserialize round
///    trip order-preserving without it — `serde_json::Map` is a
///    `BTreeMap`, so parsing `arguments` into a `Value` and re-emitting it
///    would re-sort its keys alphabetically. OpenAI's real wire format
///    already gives `arguments` to us as a JSON-encoded **string** (not an
///    object) at the point a client — or a previous assistant turn being
///    replayed back as history — hands it to this crate, so its own
///    content is *already* exactly the object's JSON text, in the
///    original key order, with nothing to re-sort. Splicing that text
///    verbatim (after normalizing an empty/whitespace-only string to
///    `{}`, the documented "no arguments" convention
///    [`crate::tool_calling::xml_tool_calls_to_openai`] itself produces)
///    reproduces it byte-for-byte.
/// 2. **`arguments|items` needs a mapping (B1(d)).** The real template
///    iterates `tool_call.arguments|items` — `.items()` on a JSON STRING
///    raises `'str' object has no items()` in both Python jinja2 and this
///    crate's own engine (verified: `chat_templates.rs`'s embedded fixture
///    against Python jinja2 on the differential case). Splicing the
///    argument text so that `Value::from_json_str`
///    ([`oxibonsai_tokenizer::jinja::Value`]'s own order-preserving
///    parser, used by [`RenderMessage::with_tool_calls_json`]'s consumer,
///    `build_context`) sees a genuine embedded JSON OBJECT is what makes
///    `arguments` a mapping to the template, instead of the doubly-quoted
///    string OpenAI's outer wire format itself uses.
///
/// A `raw_arguments` that is not valid JSON at all (the client sent
/// genuinely malformed history) is spliced as-is too: the WHOLE resulting
/// `tool_calls` array text then fails to parse in
/// [`RenderMessage::with_tool_calls_json`]'s consumer, which surfaces as an
/// honest [`JinjaError::Runtime`] — an ERROR, never garbage (spec item 1)
/// — rather than this function guessing at a recovery.
pub(crate) fn tool_calls_to_render_json(calls: &[RenderableToolCall]) -> String {
    let mut out = String::from("[");
    for (i, call) in calls.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let name_json = serde_json::to_string(&call.name).unwrap_or_else(|_| "\"\"".to_string());
        let trimmed = call.arguments_json_text.trim();
        let args_json = if trimmed.is_empty() { "{}" } else { trimmed };
        out.push_str(&format!(
            r#"{{"type":"function","function":{{"name":{name_json},"arguments":{args_json}}}}}"#
        ));
    }
    out.push(']');
    out
}

/// TOK-M2, adapted for whole-prompt Jinja rendering — see the module doc's
/// "TOK-M2, adapted for whole-prompt rendering" section for the full
/// rationale. Validates (and, if `sanitize` is `true`, neutralizes) one
/// message's raw free-text field (`content` or `reasoning_content`) BEFORE
/// it is substituted into the template.
///
/// Mirrors `server::sanitize::SpecialTokenGuard::encode_content`'s own
/// two-layer defense exactly, just applied to one message field instead of
/// one `PromptSegment::Content`:
///
/// 1. [`neutralize_special_markers`] (the `<|...|>` raw-text guard) first.
/// 2. The cleaned text is encoded ALONE (isolated from any template text)
///    and [`SpecialTokenGuard::neutralize`] strips any id that belongs to
///    the tokenizer's added/control vocabulary — this is what actually
///    catches `<think>`/`<tool_call>`/`<tool_response>`, none of which the
///    `<|...|>`-shaped raw-text guard alone would ever match.
///
/// If nothing was dropped, the cleaned text is used as-is (no lossy
/// decode round-trip for the overwhelmingly common case). If something
/// WAS dropped, the neutralized id sequence is decoded back to text and
/// then re-encoded ONE more time as a verification pass: only once THAT
/// re-encoding also contains no control ids is the decoded text accepted.
/// This closes the "reveal" class a naive drop-then-decode could otherwise
/// open (dropping an id can leave text whose surrounding characters
/// combine into a NEW control-shaped token once re-tokenized as part of
/// the larger prompt) — the same class [`neutralize_special_markers`]'s
/// own seam-space insertion defends against for the `<|...|>` family, done
/// here for the id-level family instead.
///
/// # Errors
/// Only when content still resolves to a control-token id after one
/// neutralize-decode-reencode pass — an adversarially-crafted "reveal"
/// (e.g. `<th<think>ink>`), not ordinary text.
pub(crate) fn neutralize_message_text(
    tokenizer: &TokenizerBridge,
    guard: &SpecialTokenGuard,
    text: &str,
    sanitize: bool,
) -> Result<String, ApiError> {
    if !sanitize || text.is_empty() {
        return Ok(text.to_string());
    }
    let cleaned = neutralize_special_markers(text);
    let mut ids = tokenizer
        .encode(&cleaned)
        .map_err(|e| ApiError::internal(format!("failed to validate message content: {e}")))?;
    let dropped = guard.neutralize(&mut ids);
    if dropped == 0 {
        return Ok(cleaned);
    }
    tracing::debug!(
        dropped,
        "dropped control tokens injected in client message content (chat-template rendering path)"
    );
    let decoded = tokenizer
        .decode(&ids)
        .map_err(|e| ApiError::internal(format!("failed to validate message content: {e}")))?;
    let mut reencoded = tokenizer
        .encode(&decoded)
        .map_err(|e| ApiError::internal(format!("failed to validate message content: {e}")))?;
    if guard.neutralize(&mut reencoded) > 0 {
        return Err(ApiError::bad_request(
            "message content could not be safely rendered: it still resolves to a control \
             token after neutralization"
                .to_string(),
            "messages",
        ));
    }
    Ok(decoded)
}

/// Map a template render failure to an honest client-facing `400` (spec
/// item 1: "an unsupported construct must ERROR") rather than a `500` —
/// every documented `raise_exception` (empty messages, an unsupported
/// `reasoning_effort`, a system message not first, no user query found,
/// ...) and every malformed-JSON runtime error (invalid `tools` /
/// `tool_calls` text) is a property of the REQUEST, not a server fault.
pub(crate) fn api_error_from_jinja(err: JinjaError) -> ApiError {
    let message = match err.template_raise_message() {
        Some(msg) => msg.to_string(),
        None => err.to_string(),
    };
    ApiError::bad_request(message, "messages")
}

/// Render `messages` through `tokenizer`'s own resolved chat template and
/// encode the result to prompt tokens in one whole-prompt `encode` call.
/// See the module doc for why one call, and [`neutralize_message_text`]
/// for how TOK-M2 is preserved despite it.
///
/// Returns `(rendered_prompt_text, prompt_token_ids)` — the text is kept
/// (rather than discarded) because callers need it to resolve
/// [`super::TokenizerBridge::think_open_id`] / `think_close_id` against
/// the actual encoded ids (`started_in_think`, below).
/// Guard every free-text field of one turn's tool calls before they are
/// spliced into the rendered prompt (see [`render_chat_prompt`]'s TOK-M2
/// note): both `name` and the raw `arguments_json_text` are attacker
/// reachable the same way `content` is — a replayed assistant history turn
/// is exactly a prior tool call a caller supplies back to us — so both go
/// through the same [`neutralize_message_text`] two-layer defense.
///
/// Guarding `arguments_json_text` as opaque text rather than parsing it
/// first is deliberate and safe: [`neutralize_special_markers`] only ever
/// inserts a plain ASCII space, and a `<|...|>`-shaped substring can only
/// occur inside a JSON string's own character content (nowhere else is
/// free text possible in a JSON value), so the result is still exactly as
/// JSON-shaped as the input. In the rare case the id-level guard's
/// decode/re-encode round trip (only reached when a control id was
/// actually found) does leave the text no longer valid JSON, that already
/// falls through to the same honest `JinjaError` `tool_calls_to_render_json`'s
/// own doc describes for any malformed `arguments` — an error, never
/// silently-wrong output.
fn guard_tool_calls(
    tokenizer: &TokenizerBridge,
    guard: &SpecialTokenGuard,
    calls: &[RenderableToolCall],
    sanitize: bool,
) -> Result<Vec<RenderableToolCall>, ApiError> {
    calls
        .iter()
        .map(|call| {
            Ok(RenderableToolCall {
                name: neutralize_message_text(tokenizer, guard, &call.name, sanitize)?,
                arguments_json_text: neutralize_message_text(
                    tokenizer,
                    guard,
                    &call.arguments_json_text,
                    sanitize,
                )?,
            })
        })
        .collect()
}

pub(crate) fn render_chat_prompt(
    tokenizer: &TokenizerBridge,
    guard: &SpecialTokenGuard,
    messages: &[RenderableMessage],
    opts: &RenderOptions,
    sanitize: bool,
) -> Result<(String, Vec<u32>), ApiError> {
    let mut render_messages = Vec::with_capacity(messages.len());
    for msg in messages {
        let content = neutralize_message_text(tokenizer, guard, &msg.content, sanitize)?;
        let mut rm = RenderMessage::new(msg.role.clone(), content);
        if let Some(reasoning) = &msg.reasoning_content {
            let cleaned = neutralize_message_text(tokenizer, guard, reasoning, sanitize)?;
            if !cleaned.is_empty() {
                rm = rm.with_reasoning_content(cleaned);
            }
        }
        if !msg.tool_calls.is_empty() {
            // TOK-M2: `name`/`arguments` are as attacker-reachable as
            // `content` (a replayed assistant tool call is client-supplied
            // history) but previously went straight into the rendered
            // prompt unguarded — see `guard_tool_calls`'s doc.
            let guarded_calls = guard_tool_calls(tokenizer, guard, &msg.tool_calls, sanitize)?;
            rm = rm.with_tool_calls_json(tool_calls_to_render_json(&guarded_calls));
        }
        if let Some(id) = &msg.tool_call_id {
            rm = rm.with_tool_call_id(id.clone());
        }
        render_messages.push(rm);
    }

    // TOK-M2: the `tools` schema (names/descriptions/parameter text) is
    // exactly as caller-controlled as any message and, before this, went
    // straight into the rendered prompt unguarded — the same
    // `<|...|>`/control-id injection `neutralize_message_text` closes for
    // `content` applies verbatim here, treating the whole raw JSON text as
    // one opaque string (see `guard_tool_calls`'s doc for why that is
    // JSON-safe). A no-op for ordinary tool schemas: G7's own golden tools
    // fixture carries no control-shaped text, so its rendering is
    // unaffected byte-for-byte.
    let guarded_tools = match &opts.tools {
        Some(raw) => Some(neutralize_message_text(tokenizer, guard, raw, sanitize)?),
        None => None,
    };
    let owned_opts;
    let opts: &RenderOptions = if guarded_tools == opts.tools {
        opts
    } else {
        owned_opts = RenderOptions {
            tools: guarded_tools,
            ..opts.clone()
        };
        &owned_opts
    };

    let template = tokenizer.resolved_chat_template();
    let rendered = template
        .render_with(&render_messages, opts)
        .map_err(api_error_from_jinja)?;
    let prompt_tokens = tokenizer
        .encode(&rendered)
        .map_err(|e| ApiError::internal(format!("failed to tokenize the rendered prompt: {e}")))?;
    Ok((rendered, prompt_tokens))
}

/// Whether the rendered+encoded prompt ends inside an open `<think>` span
/// (B3 / RT-10): the MOST RECENT think-marker id in `prompt_tokens` —
/// scanning backward past any trailing non-marker ids (e.g. the newline
/// token(s) after `enable_thinking: false`'s own `</think>\n\n`) — decides
/// it, never just the prompt's own last id.
///
/// The real template's generation-prompt tail is exactly one of:
/// * `enable_thinking` true/default: `...<think>\n` — ends WITH an open
///   think span, so generation starts already inside `reasoning_content`.
/// * `enable_thinking: false`: `...<think>\n\n</think>\n\n` — the close
///   marker is the more recent of the two, so generation starts in
///   `content` even though `<think>`'s id is still present earlier in the
///   very same tail.
///
/// `false` when neither id resolved for this vocabulary (RT-10's "models
/// without `<think>`" — [`ReasoningSplitter`](crate::reasoning::ReasoningSplitter)
/// is permanent pass-through in that case regardless of this function's
/// result, but callers still call it uniformly).
pub(crate) fn started_in_think(
    prompt_tokens: &[u32],
    think_open_id: Option<u32>,
    think_close_id: Option<u32>,
) -> bool {
    prompt_tokens
        .iter()
        .rev()
        .find_map(|&id| {
            if Some(id) == think_open_id {
                Some(true)
            } else if Some(id) == think_close_id {
                Some(false)
            } else {
                None
            }
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::sanitize::SpecialTokenGuard;

    fn native_tokenizer_with_specials() -> TokenizerBridge {
        const TOKENIZER_JSON: &str = r##"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "!": 0, "\"": 1, "#": 2, "$": 3,
                    "H": 4, "i": 5, "Ġ": 6, "a": 7, "b": 8, "n": 9, "k": 10, "t": 11, "o": 12,
                    "<": 13, ">": 14, "T": 15, "h": 16, "T2": 17
                },
                "merges": []
            },
            "added_tokens": [
                { "id": 100, "content": "<think>", "special": false },
                { "id": 101, "content": "</think>", "special": false },
                { "id": 102, "content": "<tool_call>", "special": false },
                { "id": 103, "content": "</tool_call>", "special": false }
            ],
            "pre_tokenizer": { "type": "ByteLevel" },
            "decoder": { "type": "ByteLevel" }
        }"##;
        TokenizerBridge::native_from_json_str(TOKENIZER_JSON).expect("fixture must load")
    }

    #[test]
    fn tool_calls_to_render_json_splices_raw_arguments_text_verbatim() {
        let calls = vec![RenderableToolCall {
            name: "get_weather".to_string(),
            // Deliberately in non-alphabetical key order: proves nothing
            // re-sorts it.
            arguments_json_text: r#"{"zone":"jst","city":"Tokyo"}"#.to_string(),
        }];
        let json = tool_calls_to_render_json(&calls);
        assert_eq!(
            json,
            r#"[{"type":"function","function":{"name":"get_weather","arguments":{"zone":"jst","city":"Tokyo"}}}]"#
        );
        // Must still be valid JSON overall.
        let _: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    }

    #[test]
    fn tool_calls_to_render_json_normalizes_empty_arguments_to_an_empty_object() {
        let calls = vec![RenderableToolCall {
            name: "ping".to_string(),
            arguments_json_text: String::new(),
        }];
        let json = tool_calls_to_render_json(&calls);
        assert_eq!(
            json,
            r#"[{"type":"function","function":{"name":"ping","arguments":{}}}]"#
        );
    }

    #[test]
    fn tool_calls_to_render_json_multiple_calls() {
        let calls = vec![
            RenderableToolCall {
                name: "a".to_string(),
                arguments_json_text: "{}".to_string(),
            },
            RenderableToolCall {
                name: "b".to_string(),
                arguments_json_text: r#"{"x":1}"#.to_string(),
            },
        ];
        let json = tool_calls_to_render_json(&calls);
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v.as_array().expect("array").len(), 2);
    }

    #[test]
    fn neutralize_message_text_passes_ordinary_text_through_unchanged() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let out = neutralize_message_text(&tok, &guard, "Hi Hi Hi", true).expect("ok");
        assert_eq!(out, "Hi Hi Hi");
    }

    #[test]
    fn neutralize_message_text_is_a_no_op_when_sanitize_is_false() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let out =
            neutralize_message_text(&tok, &guard, "<think>anything</think>", false).expect("ok");
        assert_eq!(out, "<think>anything</think>");
    }

    #[test]
    fn neutralize_message_text_strips_an_injected_control_token() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        // "<think>" is a real added token in this fixture's vocabulary; a
        // client sending it as plain content must not have it survive as
        // the real control-token id once the whole prompt is re-encoded.
        let out = neutralize_message_text(&tok, &guard, "<think>", true).expect("must not error");
        let ids = tok.encode(&out).expect("encode");
        assert!(
            !ids.contains(&100),
            "the real <think> id must not survive neutralization; got ids {ids:?} for {out:?}"
        );
    }

    #[test]
    fn neutralize_message_text_strips_an_injected_tool_call_marker() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let out =
            neutralize_message_text(&tok, &guard, "<tool_call>", true).expect("must not error");
        let ids = tok.encode(&out).expect("encode");
        assert!(!ids.contains(&102), "got ids {ids:?} for {out:?}");
    }

    #[test]
    fn neutralize_message_text_empty_string_is_untouched() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let out = neutralize_message_text(&tok, &guard, "", true).expect("ok");
        assert_eq!(out, "");
    }

    #[test]
    fn started_in_think_true_when_open_is_the_most_recent_marker() {
        // "...<think>\n" (enable_thinking default/true): the ONLY marker
        // present is the open one.
        assert!(started_in_think(&[1, 2, 100, 5], Some(100), Some(101)));
    }

    #[test]
    fn started_in_think_false_when_close_is_more_recent_than_open() {
        // "...<think>\n\n</think>\n\n" (enable_thinking: false): both
        // markers appear, but close is LATER (more recent).
        assert!(!started_in_think(
            &[1, 100, 9, 9, 101, 9, 9],
            Some(100),
            Some(101)
        ));
    }

    #[test]
    fn started_in_think_false_when_no_markers_resolved() {
        // RT-10: a vocabulary with no <think>/</think> at all.
        assert!(!started_in_think(&[1, 2, 3], None, None));
    }

    #[test]
    fn started_in_think_false_on_an_empty_prompt() {
        assert!(!started_in_think(&[], Some(100), Some(101)));
    }

    #[test]
    fn render_chat_prompt_renders_and_encodes_the_fallback_template() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let messages = vec![RenderableMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            ..Default::default()
        }];
        let opts = RenderOptions {
            add_generation_prompt: true,
            ..Default::default()
        };
        let (text, ids) =
            render_chat_prompt(&tok, &guard, &messages, &opts, true).expect("must render");
        assert!(text.starts_with("<|im_start|>user\n"));
        assert!(!ids.is_empty());
        // Round-trips to the same rendered text (whole-prompt encode).
        assert_eq!(
            tok.decode(&ids).unwrap_or_default().is_empty(),
            text.is_empty()
        );
    }

    #[test]
    fn render_chat_prompt_errors_honestly_on_empty_messages() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let err = render_chat_prompt(&tok, &guard, &[], &RenderOptions::default(), true)
            .expect_err("must error, not render garbage");
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn render_chat_prompt_keeps_tool_calls_argument_key_order() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let messages = vec![
            RenderableMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
                ..Default::default()
            },
            RenderableMessage {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: vec![RenderableToolCall {
                    name: "f".to_string(),
                    arguments_json_text: r#"{"z":1,"a":2}"#.to_string(),
                }],
                ..Default::default()
            },
        ];
        let (text, _) = render_chat_prompt(
            &tok,
            &guard,
            &messages,
            &RenderOptions {
                add_generation_prompt: false,
                ..Default::default()
            },
            true,
        )
        .expect("must render");
        // The fallback template's tojson-free `<parameter=KEY>` shape
        // preserves argument order directly in the rendered text.
        let z_pos = text.find("<parameter=z>").expect("z parameter present");
        let a_pos = text.find("<parameter=a>").expect("a parameter present");
        assert!(
            z_pos < a_pos,
            "z must render before a (original order): {text:?}"
        );
    }

    // ── TOK-M2 injection surfaces opened by whole-prompt rendering ────────
    //
    // `content`/`reasoning_content` were always guarded; `tool_calls`
    // (`name`/`arguments`) and `tools` are exactly as client-reachable (a
    // replayed assistant tool call, or the tool schema itself) but went
    // straight into the rendered prompt unguarded before `guard_tool_calls`
    // and `render_chat_prompt`'s own `opts.tools` guard.

    #[test]
    fn render_chat_prompt_neutralizes_an_injected_control_token_in_a_tool_call_name() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let messages = vec![RenderableMessage {
            role: "assistant".to_string(),
            content: String::new(),
            // "<think>" is a real added token (id 100) in this fixture.
            // Unguarded, this would let a replayed tool-call name inject a
            // real control-token id into the middle of the rendered prompt.
            tool_calls: vec![RenderableToolCall {
                name: "<think>".to_string(),
                arguments_json_text: "{}".to_string(),
            }],
            ..Default::default()
        }];
        let (_, ids) = render_chat_prompt(&tok, &guard, &messages, &RenderOptions::default(), true)
            .expect("must render, not error, for this non-adversarial-JSON case");
        assert!(
            !ids.contains(&100),
            "an injected <think> id must not survive into the rendered prompt via a tool \
             call's own name: {ids:?}"
        );
    }

    /// `<tool_call>` (unlike the `<|...|>` family) is only caught by the
    /// id-level layer, whose drop-then-decode can itself leave a JSON
    /// value's surrounding quote/brace no longer valid JSON — the template
    /// then honestly fails to parse `arguments` back out
    /// (`tool_calls_to_render_json`'s own doc anticipates exactly this).
    /// Either outcome is acceptable and is asserted here: a successful
    /// render must never carry the real id, and any error must be a clean
    /// `400`, never a panic or a `500`.
    #[test]
    fn render_chat_prompt_neutralizes_an_injected_control_token_in_tool_call_arguments() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        let messages = vec![RenderableMessage {
            role: "assistant".to_string(),
            content: String::new(),
            // "<tool_call>" (id 102) sits inside the JSON *value*, exactly
            // where a client-supplied argument string would put it.
            tool_calls: vec![RenderableToolCall {
                name: "f".to_string(),
                arguments_json_text: r#"{"x":"<tool_call>"}"#.to_string(),
            }],
            ..Default::default()
        }];
        match render_chat_prompt(&tok, &guard, &messages, &RenderOptions::default(), true) {
            Ok((_, ids)) => assert!(
                !ids.contains(&102),
                "an injected <tool_call> id must not survive into the rendered prompt via a \
                 tool call's own arguments: {ids:?}"
            ),
            Err(e) => assert_eq!(
                e.status(),
                axum::http::StatusCode::BAD_REQUEST,
                "a neutralization that corrupts the argument JSON must fail honestly \
                 (400), not panic or 500: {e:?}"
            ),
        }
    }

    #[test]
    fn render_chat_prompt_neutralizes_an_injected_control_token_in_the_tools_schema() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        // The fallback template never references `tools` at all (see
        // `chat_tests.rs::chat_request_extras_tools_raw_json_preserves_key_order`'s
        // identical note) — attach a minimal real-Jinja template that does.
        const TOOLS_TEMPLATE: &str = "{% for m in messages %}{{ m.content }}{% endfor %}\
             {% if tools %}{% for t in tools %}{{ t | tojson }}{% endfor %}{% endif %}";
        let tok = tok.with_chat_template(
            oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::Jinja(std::sync::Arc::new(
                oxibonsai_tokenizer::jinja::JinjaTemplate::compile(TOOLS_TEMPLATE)
                    .expect("must compile"),
            )),
        );
        let messages = vec![RenderableMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
            ..Default::default()
        }];
        // A tool *name* carrying the real added-token text, spliced by the
        // template's `t | tojson` the same way the real Bonsai 2 template
        // does for a client-supplied tools array.
        let opts = RenderOptions {
            tools: Some(r#"[{"name":"<think>"}]"#.to_string()),
            ..Default::default()
        };
        // Same acceptable-outcomes shape as the tool-call-arguments case
        // above: `<think>` is only caught by the id-level layer, whose
        // drop-then-decode can leave the schema no longer valid JSON for
        // the template's own `tools` re-parse — an honest 400, never a
        // silently-injected id.
        match render_chat_prompt(&tok, &guard, &messages, &opts, true) {
            Ok((_, ids)) => assert!(
                !ids.contains(&100),
                "an injected <think> id must not survive into the rendered prompt via the \
                 tools schema: {ids:?}"
            ),
            Err(e) => assert_eq!(
                e.status(),
                axum::http::StatusCode::BAD_REQUEST,
                "a neutralization that corrupts the tools JSON must fail honestly (400), \
                 not panic or 500: {e:?}"
            ),
        }
    }

    #[test]
    fn render_chat_prompt_tools_guard_is_a_no_op_when_sanitize_is_false() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        const ECHO_TEMPLATE: &str =
            "{% if tools %}{% for t in tools %}{{ t | tojson }}{% endfor %}{% endif %}";
        let tok = tok.with_chat_template(
            oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::Jinja(std::sync::Arc::new(
                oxibonsai_tokenizer::jinja::JinjaTemplate::compile(ECHO_TEMPLATE)
                    .expect("must compile"),
            )),
        );
        let opts = RenderOptions {
            tools: Some(r#"[{"name":"Hi"}]"#.to_string()),
            ..Default::default()
        };
        let (text, _) = render_chat_prompt(&tok, &guard, &[], &opts, false).expect("must render");
        // `tojson` itself reformats (a space after `:`, matching Python
        // `json.dumps`'s convention — see `tool_calls_to_render_json`'s own
        // doc) regardless of the guard, so this checks the *value* survived
        // untouched, not exact byte spacing.
        assert!(
            text.contains(r#""name": "Hi"#),
            "sanitize:false must leave the tools text untouched: {text:?}"
        );
    }

    // ── B11/SV-11: reasoning_content + vision-content-array pre-pass ──────

    #[test]
    fn preprocess_extracts_reasoning_content_by_index() {
        let raw = r#"{"messages": [
            {"role": "assistant", "content": "hi", "reasoning_content": "thinking"},
            {"role": "user", "content": "thanks"}
        ]}"#;
        let (_, reasoning_contents) =
            preprocess_message_content_and_reasoning(raw).expect("must succeed");
        assert_eq!(reasoning_contents, vec![Some("thinking".to_string()), None]);
    }

    #[test]
    fn preprocess_flattens_a_text_only_content_array() {
        let raw = r#"{"messages": [
            {"role": "user", "content": [{"type":"text","text":"hello"},{"type":"text","text":" world"}]}
        ]}"#;
        let (rewritten, _) = preprocess_message_content_and_reasoning(raw).expect("must succeed");
        let v: serde_json::Value = serde_json::from_str(&rewritten).expect("valid JSON");
        assert_eq!(
            v["messages"][0]["content"],
            serde_json::json!("hello world")
        );
    }

    #[test]
    fn preprocess_rejects_an_image_url_part_honestly() {
        let raw = r#"{"messages": [
            {"role": "user", "content": [{"type":"image_url","image_url":{"url":"http://x"}}]}
        ]}"#;
        let err = preprocess_message_content_and_reasoning(raw)
            .expect_err("an image_url part must be rejected, not silently dropped");
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn preprocess_leaves_plain_string_content_untouched() {
        let raw = r#"{"messages": [{"role": "user", "content": "hi"}]}"#;
        let (rewritten, reasoning_contents) =
            preprocess_message_content_and_reasoning(raw).expect("must succeed");
        let v: serde_json::Value = serde_json::from_str(&rewritten).expect("valid JSON");
        assert_eq!(v["messages"][0]["content"], serde_json::json!("hi"));
        assert_eq!(reasoning_contents, vec![None]);
    }

    #[test]
    fn preprocess_leaves_malformed_json_for_the_typed_parse_to_reject() {
        let raw = "{not json";
        let (rewritten, reasoning_contents) =
            preprocess_message_content_and_reasoning(raw).expect("must not itself error");
        assert_eq!(rewritten, raw);
        assert!(reasoning_contents.is_empty());
    }

    #[test]
    fn preprocess_leaves_a_missing_messages_array_untouched() {
        let raw = r#"{"model": "x"}"#;
        let (rewritten, reasoning_contents) =
            preprocess_message_content_and_reasoning(raw).expect("must succeed");
        let v: serde_json::Value = serde_json::from_str(&rewritten).expect("valid JSON");
        assert_eq!(v["model"], serde_json::json!("x"));
        assert!(reasoning_contents.is_empty());
    }

    #[test]
    fn to_render_messages_merges_reasoning_content_by_index() {
        let messages = vec![
            crate::server::ChatMessage::text("assistant", "hi"),
            crate::server::ChatMessage::text("user", "thanks"),
        ];
        let reasoning_contents = vec![Some("thinking".to_string()), None];
        let render_messages = to_render_messages(&messages, &reasoning_contents);
        assert_eq!(
            render_messages[0].reasoning_content.as_deref(),
            Some("thinking")
        );
        assert_eq!(render_messages[1].reasoning_content, None);
    }

    #[test]
    fn to_render_messages_tolerates_a_shorter_reasoning_contents_list() {
        // No tokenizer attached at all skips the whole pre-pass in the
        // caller, so `reasoning_contents` can legitimately be `&[]` even
        // when `messages` is not.
        let messages = vec![crate::server::ChatMessage::text("user", "hi")];
        let render_messages = to_render_messages(&messages, &[]);
        assert_eq!(render_messages[0].reasoning_content, None);
    }

    /// End-to-end through the real pre-pass, merge, and render — not just
    /// each piece in isolation — using a template that echoes
    /// `reasoning_content` the way the real Bonsai 2 template does
    /// (`chat_templates.rs`'s own `g7_case_3_multi_turn_with_reasoning_content`
    /// pins that template's exact formatting; this pins that the HTTP-layer
    /// wiring actually delivers a client's `reasoning_content` to it at
    /// all, which before this fix it never did — B11).
    #[test]
    fn reasoning_content_survives_the_whole_pipeline_from_raw_body_to_rendered_prompt() {
        let tok = native_tokenizer_with_specials();
        let guard = SpecialTokenGuard::from_tokenizer(&tok);
        const ECHO_TEMPLATE: &str =
            "{% for m in messages %}{{ m.role }}[{{ m.reasoning_content }}]:{{ m.content }};{% endfor %}";
        let tok = tok.with_chat_template(
            oxibonsai_tokenizer::chat_templates::ResolvedChatTemplate::Jinja(std::sync::Arc::new(
                oxibonsai_tokenizer::jinja::JinjaTemplate::compile(ECHO_TEMPLATE)
                    .expect("must compile"),
            )),
        );

        let raw = r#"{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "user greets"}
        ]}"#;
        let (rewritten, reasoning_contents) =
            preprocess_message_content_and_reasoning(raw).expect("must succeed");
        let value: serde_json::Value = serde_json::from_str(&rewritten).expect("valid JSON");
        let messages: Vec<crate::server::ChatMessage> =
            serde_json::from_value(value["messages"].clone()).expect("must deserialize");
        let render_messages = to_render_messages(&messages, &reasoning_contents);

        let (text, _) = render_chat_prompt(
            &tok,
            &guard,
            &render_messages,
            &RenderOptions::default(),
            true,
        )
        .expect("must render");
        assert_eq!(text, "user[]:hi;assistant[user greets]:Hello!;");
    }
}
