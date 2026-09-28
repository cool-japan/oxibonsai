//! Incremental tool-call extraction over one response's content channel.
//!
//! A child of [`crate::tool_calling`] (declared there through `#[path]`),
//! kept in its own file so `tool_calling.rs` stays under the workspace's
//! 2000-line ceiling.
//!
//! [`ToolCallStreamExtractor`] is the single implementation both chat
//! endpoints run over a response's content, streamed or not: it is fed one
//! generated token at a time (its id and its decoded text) and releases
//! [`ToolStreamEvent`]s — ordinary content as soon as it provably cannot be
//! part of a call, and each tool call the moment its block closes. The
//! non-streaming handlers feed the same tokens through the same extractor,
//! so a streamed response and its non-streamed twin agree on content,
//! calls and `finish_reason` by construction.
//!
//! # The protocol it enforces
//!
//! * **Openers.** When the vocabulary has a `<tool_call>` token, only that
//!   token opens a block (design App. A.1): the text `<tool_call>` spelled
//!   out of ordinary tokens — a model writing *about* the tag — is content.
//!   A vocabulary without one falls back to matching the text, holding back
//!   a trailing partial `<tool_call` until it resolves either way.
//! * **Blocks.** A block runs from its opener to the first `</tool_call>`
//!   (the vocabulary's closing token, or the text). It is parsed in the XML
//!   form the Bonsai 2 template teaches ([`super::XmlToolCallStreamParser`])
//!   or, failing that, the Qwen3 JSON form (`{"name": …, "arguments": …}`,
//!   [`crate::api_types::parse_json_tool_call_body`]).
//! * **Text before the first call** is content; whitespace immediately in
//!   front of an opener is dropped (it only separated the prose from the
//!   call).
//! * **After a call** only whitespace and further calls may follow (the
//!   template's `<IMPORTANT>` rule: reasoning goes before a call, never
//!   after). Any other text is a protocol violation: it — and everything
//!   after it — is dropped with a warning, exactly as the whole-text parser
//!   [`super::parse_xml_tool_calls`] drops it.
//! * **A block that closes but does not parse** ends extraction: its text,
//!   and everything after it, is ordinary content.
//! * **A block still open when the response ends** is not a call: its text
//!   is released as content ([`ToolStreamEnd::unclosed`]).

use super::{make_tool_call, new_tool_call_id, XmlToolCallStreamParser};
use super::{TOOL_CALL_CLOSE, TOOL_CALL_OPEN};
use crate::api_types::ToolCall;

/// What [`ToolCallStreamExtractor`] releases, in response order.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolStreamEvent {
    /// Assistant text that is not part of a tool call.
    Content(String),
    /// A tool call whose block has just closed and parsed — OpenAI shape,
    /// with a freshly minted id and `arguments` in the model's own key order.
    Call(ToolCall),
}

/// How a response's content channel ended, as far as tool calls go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolStreamEnd {
    /// Tool calls released over the whole response.
    pub calls: usize,
    /// A block was still open when the response ended; its text was released
    /// as content.
    pub unclosed: bool,
    /// A closed block did not parse as a call; its text, and everything
    /// after it, was released as content.
    pub malformed: bool,
}

/// Where the extractor is in the response (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// No call yet: text is content.
    Content,
    /// Inside a block, from its opener up to `</tool_call>`.
    Block,
    /// A call just closed: whitespace and further calls may follow.
    AfterCall,
    /// Text followed a call: everything from here on is dropped.
    Dropping,
    /// A block failed to parse: everything from here on is content.
    Passthrough,
}

/// Incremental, token-level tool-call extractor — see the module doc.
#[derive(Debug, Clone)]
pub struct ToolCallStreamExtractor {
    /// The vocabulary's `<tool_call>` id; `None` matches the opener as text.
    open_id: Option<u32>,
    /// The vocabulary's `</tool_call>` id, when it has one.
    close_id: Option<u32>,
    phase: Phase,
    /// Content not yet released: its trailing whitespace (dropped if a call
    /// opens next) and, when openers are text-matched, a trailing partial
    /// `<tool_call` prefix.
    held: String,
    /// Whitespace dropped in front of the current block's opener, released
    /// again with the block if the block never becomes a call.
    pre_block: String,
    /// The current block's text from its opener on.
    block: String,
    calls: usize,
    unclosed: bool,
    malformed: bool,
    /// Bytes dropped after a call ([`Phase::Dropping`]).
    dropped: usize,
}

/// Whitespace in the sense `str::trim` uses.
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

impl ToolCallStreamExtractor {
    /// An extractor for one response. `open_id` / `close_id` are the
    /// vocabulary's `<tool_call>` / `</tool_call>` token ids (`None` when it
    /// has no such token: that marker is then matched as text).
    pub fn new(open_id: Option<u32>, close_id: Option<u32>) -> Self {
        Self {
            open_id,
            close_id,
            phase: Phase::Content,
            held: String::new(),
            pre_block: String::new(),
            block: String::new(),
            calls: 0,
            unclosed: false,
            malformed: false,
            dropped: 0,
        }
    }

    /// Tool calls released so far.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Whether the extractor is inside a block (text is being held back as a
    /// possible call rather than released as content).
    pub fn in_block(&self) -> bool {
        self.phase == Phase::Block
    }

    /// Feed one generated token: its id and its decoded text (possibly empty
    /// — a token completing no character yet, or a special-flagged one).
    /// Returns what is now safe to release, in order.
    pub fn push(&mut self, id: u32, text: &str) -> Vec<ToolStreamEvent> {
        let mut out = Vec::new();
        self.feed(Some(id), text, &mut out);
        out
    }

    /// End the response: release whatever is still held back, and report
    /// how extraction ended.
    pub fn finish(&mut self) -> (Vec<ToolStreamEvent>, ToolStreamEnd) {
        let mut out = Vec::new();
        match self.phase {
            Phase::Content => {
                let held = std::mem::take(&mut self.held);
                push_content(&mut out, held);
            }
            Phase::Block => {
                self.unclosed = true;
                let mut text = std::mem::take(&mut self.pre_block);
                text.push_str(&self.block);
                self.block.clear();
                push_content(&mut out, text);
                self.phase = Phase::Passthrough;
            }
            Phase::AfterCall => self.held.clear(),
            Phase::Dropping | Phase::Passthrough => {}
        }
        (
            out,
            ToolStreamEnd {
                calls: self.calls,
                unclosed: self.unclosed,
                malformed: self.malformed,
            },
        )
    }

    /// Route `text` (the decoded text of token `id`, or `None` for text
    /// that is the remainder of an earlier token) through the current phase.
    fn feed(&mut self, id: Option<u32>, text: &str, out: &mut Vec<ToolStreamEvent>) {
        match self.phase {
            Phase::Content => self.feed_content(id, text, out),
            Phase::Block => self.feed_block(id, text, out),
            Phase::AfterCall => self.feed_after_call(id, text, out),
            Phase::Dropping => {
                self.dropped = self.dropped.saturating_add(text.len());
            }
            Phase::Passthrough => push_content(out, text.to_string()),
        }
    }

    fn is_open(&self, id: Option<u32>) -> bool {
        id.is_some() && id == self.open_id
    }

    fn feed_content(&mut self, id: Option<u32>, text: &str, out: &mut Vec<ToolStreamEvent>) {
        if self.open_id.is_some() {
            if self.is_open(id) {
                self.pre_block = std::mem::take(&mut self.held);
                self.open_block(out);
                return;
            }
            self.held.push_str(text);
            self.release_held_up_to(self.held.len(), out);
            return;
        }
        // Text-matched openers.
        self.held.push_str(text);
        if let Some(pos) = self.held.find(TOOL_CALL_OPEN) {
            let rest = self.held.split_off(pos + TOOL_CALL_OPEN.len());
            self.held.truncate(pos);
            let content_end = self.held.trim_end_matches(is_space).len();
            let before = self.held.split_off(content_end);
            let released = std::mem::take(&mut self.held);
            push_content(out, released);
            self.pre_block = before;
            self.open_block(out);
            if !rest.is_empty() {
                self.feed(None, &rest, out);
            }
            return;
        }
        let partial = (1..TOOL_CALL_OPEN.len())
            .rev()
            .find(|&k| self.held.ends_with(&TOOL_CALL_OPEN[..k]))
            .unwrap_or(0);
        let safe_end = self.held.len() - partial;
        self.release_held_up_to(safe_end, out);
    }

    /// Release `held[..limit]` up to its last non-whitespace character; the
    /// trailing whitespace (and everything from `limit` on) stays held.
    fn release_held_up_to(&mut self, limit: usize, out: &mut Vec<ToolStreamEvent>) {
        let release_end = self.held[..limit].trim_end_matches(is_space).len();
        if release_end > 0 {
            let rest = self.held.split_off(release_end);
            let released = std::mem::replace(&mut self.held, rest);
            push_content(out, released);
        }
    }

    /// Start a block at an opener (the opener's own text is the canonical
    /// `<tool_call>`, whatever its token decoded to).
    fn open_block(&mut self, out: &mut Vec<ToolStreamEvent>) {
        self.block.clear();
        self.block.push_str(TOOL_CALL_OPEN);
        self.phase = Phase::Block;
        self.try_close_block(out);
    }

    fn feed_block(&mut self, id: Option<u32>, text: &str, out: &mut Vec<ToolStreamEvent>) {
        if id.is_some() && id == self.close_id {
            self.block.push_str(TOOL_CALL_CLOSE);
        } else {
            self.block.push_str(text);
        }
        self.try_close_block(out);
    }

    /// Close the current block if its `</tool_call>` has arrived: release
    /// the call (or, when it does not parse, the block's text as content)
    /// and route whatever followed the close through the next phase.
    fn try_close_block(&mut self, out: &mut Vec<ToolStreamEvent>) {
        let Some(rel) = self.block[TOOL_CALL_OPEN.len()..].find(TOOL_CALL_CLOSE) else {
            return;
        };
        let close_end = TOOL_CALL_OPEN.len() + rel + TOOL_CALL_CLOSE.len();
        let rest = self.block.split_off(close_end);
        let block = std::mem::take(&mut self.block);
        match parse_block(&block) {
            Some(call) => {
                self.calls += 1;
                self.pre_block.clear();
                out.push(ToolStreamEvent::Call(call));
                self.phase = Phase::AfterCall;
            }
            None => {
                self.malformed = true;
                let mut text = std::mem::take(&mut self.pre_block);
                text.push_str(&block);
                push_content(out, text);
                self.phase = Phase::Passthrough;
            }
        }
        if !rest.is_empty() {
            self.feed(None, &rest, out);
        }
    }

    fn feed_after_call(&mut self, id: Option<u32>, text: &str, out: &mut Vec<ToolStreamEvent>) {
        if self.open_id.is_some() {
            if self.is_open(id) {
                self.held.clear();
                self.pre_block.clear();
                self.open_block(out);
                return;
            }
            self.held.push_str(text);
            if !self.held.chars().all(is_space) {
                self.drop_trailing();
            }
            return;
        }
        self.held.push_str(text);
        let trimmed = self.held.trim_start_matches(is_space);
        if let Some(after_open) = trimmed.strip_prefix(TOOL_CALL_OPEN) {
            let rest = after_open.to_string();
            self.held.clear();
            self.pre_block.clear();
            self.open_block(out);
            if !rest.is_empty() {
                self.feed(None, &rest, out);
            }
        } else if !TOOL_CALL_OPEN.starts_with(trimmed) {
            self.drop_trailing();
        }
    }

    /// Text followed a completed call: drop it and everything after it.
    fn drop_trailing(&mut self) {
        self.dropped = self.dropped.saturating_add(self.held.len());
        self.held.clear();
        self.phase = Phase::Dropping;
        tracing::warn!(
            target: "oxibonsai_runtime::tool_calling",
            "text after the final </tool_call> is a protocol violation (reasoning must precede \
             a tool call, never follow it); dropping it and the rest of the response"
        );
    }
}

/// Parse one closed block (`<tool_call>…</tool_call>`): the XML form first,
/// then the JSON form. `None` when it is neither.
fn parse_block(block: &str) -> Option<ToolCall> {
    let mut xml = XmlToolCallStreamParser::new();
    if let Some(call) = xml.feed(block).into_iter().next() {
        let arguments = call.arguments_json();
        return Some(make_tool_call(new_tool_call_id(), call.name, arguments));
    }
    let interior = block
        .strip_prefix(TOOL_CALL_OPEN)
        .and_then(|rest| rest.strip_suffix(TOOL_CALL_CLOSE))?;
    crate::api_types::parse_json_tool_call_body(interior, &new_tool_call_id())
}

/// Push `text` as a content event, merging with a directly preceding
/// content event and skipping empty text.
fn push_content(out: &mut Vec<ToolStreamEvent>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(ToolStreamEvent::Content(previous)) = out.last_mut() {
        previous.push_str(&text);
        return;
    }
    out.push(ToolStreamEvent::Content(text));
}

#[cfg(test)]
#[path = "tool_calling_extract_tests.rs"]
mod tests;
