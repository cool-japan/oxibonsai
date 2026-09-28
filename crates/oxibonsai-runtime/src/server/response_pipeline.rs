//! One chat response's generated tokens → its OpenAI channels, streamed or
//! not.
//!
//! Both chat endpoints (`/v1/chat/completions` and
//! `/v1/chat/completions/extended`) run every response — streamed or not —
//! through one [`ResponsePipeline`], token by token:
//!
//! 1. **decode** — [`PieceDecoder`] (UTF-8-safe; nothing is shown when no
//!    tokenizer is attached);
//! 2. **reasoning split** — [`ReasoningSplitter`], configured by
//!    [`ResponseShape`] from the vocabulary, the rendered prompt and the
//!    template's reasoning format;
//! 3. **tool-call extraction** — [`ToolCallStreamExtractor`], on the content
//!    channel only and only when the request advertised tools;
//! 4. **stop sequences** — the endpoint's [`ContentStop`] stage, on the
//!    content that survives extraction (a stop string inside the reasoning
//!    or inside a tool call's arguments stops nothing).
//!
//! A streamed response sends each [`ResponseEvent`] as it is released
//! ([`StreamDriver`]); a non-streamed one collects the same events
//! ([`ResponsePipeline::collect`]). Content, tool calls and `finish_reason`
//! therefore agree between a stream and its non-streamed twin by
//! construction.
//!
//! **After a stop sequence matches** nothing more is released — no content
//! and no tool call — and a stream cancels its generation.
//! **`finish_reason`** ([`ResponseEnd::finish_reason`]): `"tool_calls"` when
//! at least one call was released, else `"stop"` for a stop-sequence match,
//! else `"length"` when generation used its whole token budget, else
//! `"stop"`. A tool-call block still open when the response ends is not a
//! call: its text is released as content and the finish reason is the
//! natural one.

use std::sync::Arc;

use crate::api_types::ToolCall;
use crate::engine_control::CancellationToken;
use crate::reasoning::{ReasoningChunk, ReasoningFormat, ReasoningSplitter};
use crate::server::AppState;
use crate::tokenizer_bridge::chat_render::{self, DecodedPiece, PieceDecoder};
use crate::tokenizer_bridge::TokenizerBridge;
use crate::tool_calling::{ToolCallStreamExtractor, ToolStreamEvent};

// ── Events ───────────────────────────────────────────────────────────────────

/// What a [`ResponsePipeline`] releases, in response order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResponseEvent {
    /// `reasoning_content` text.
    Reasoning(String),
    /// `content` text.
    Content(String),
    /// A complete tool call.
    ToolCall(ToolCall),
}

/// How a response ended, as far as its `finish_reason` goes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ResponseEnd {
    /// Tool calls released.
    pub(crate) calls: usize,
    /// A stop sequence matched.
    pub(crate) stopped: bool,
    /// A tool-call block was still open when the response ended (its text
    /// was released as content).
    pub(crate) unclosed: bool,
}

impl ResponseEnd {
    /// The OpenAI `finish_reason` of a response that generated `generated`
    /// tokens against a budget of `max_tokens` — see the module doc.
    pub(crate) fn finish_reason(&self, generated: usize, max_tokens: usize) -> &'static str {
        if self.calls > 0 {
            "tool_calls"
        } else if self.stopped {
            "stop"
        } else if generated >= max_tokens {
            "length"
        } else {
            "stop"
        }
    }
}

// ── Stop stage ───────────────────────────────────────────────────────────────

/// An endpoint's stop-sequence stage over the content channel.
///
/// Text is held back while it could still grow into a stop sequence and
/// released once it provably cannot; a match releases everything before it
/// and stops the stage for good.
pub(crate) trait ContentStop {
    /// Feed content text; returns what is now safe to release (possibly
    /// empty). May stop the stage.
    fn push_text(&mut self, text: &str) -> String;

    /// Whether generated token `id` is itself a configured stop sequence
    /// (a stop string that is one token's whole text). Checked without
    /// changing state; the pipeline then ends the content and calls
    /// [`Self::mark_stopped`].
    fn is_stop_id(&self, _id: u32) -> bool {
        false
    }

    /// Release everything still held back — the content has ended, or a
    /// tool call follows it, so the held text can no longer grow into a
    /// match. Returns an empty string once stopped.
    fn release_held(&mut self) -> String;

    /// Stop the stage (a stop token matched by id).
    fn mark_stopped(&mut self);

    /// Whether a stop sequence has matched.
    fn stopped(&self) -> bool;
}

// ── Shape ────────────────────────────────────────────────────────────────────

/// Everything that decides how one response's tokens are split: the
/// vocabulary's marker ids, what the rendered prompt already did, the
/// template's reasoning format and whether tools are active.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ResponseShape {
    /// The rendered prompt ends inside an open think span
    /// ([`chat_render::started_in_think`]).
    pub(crate) started_in_think: bool,
    /// `<think>` recognised as a model-emitted opener at the very start of
    /// the output — `None` when the vocabulary has none, and when the
    /// prompt's generation tail already carries a closed think block (the
    /// reference parser's one optional reasoning block was consumed by the
    /// prompt itself).
    pub(crate) watched_think_open_id: Option<u32>,
    /// `</think>`.
    pub(crate) think_close_id: Option<u32>,
    /// The reference parser whose whitespace rules the split follows.
    pub(crate) reasoning_format: ReasoningFormat,
    /// `<tool_call>`.
    pub(crate) tool_call_open_id: Option<u32>,
    /// `</tool_call>`.
    pub(crate) tool_call_close_id: Option<u32>,
    /// Tool-call extraction runs (the request advertised tools and did not
    /// opt out with `tool_choice: "none"`).
    pub(crate) tools_active: bool,
}

impl ResponseShape {
    /// Resolve the shape of a response to `prompt_tokens` (rendered as
    /// `rendered`, when a template rendered it) on a server whose tokenizer
    /// is `tokenizer`. Without a tokenizer there is no marker to split on:
    /// everything is content (and none of it can be shown).
    pub(crate) fn resolve(
        tokenizer: Option<&TokenizerBridge>,
        prompt_tokens: &[u32],
        rendered: Option<&str>,
        tools_active: bool,
    ) -> Self {
        let Some(tok) = tokenizer else {
            return Self {
                tools_active,
                ..Self::default()
            };
        };
        let think_open_id = tok.think_open_id();
        let think_close_id = tok.think_close_id();
        let tail_closed = rendered.is_some_and(prompt_tail_closes_think);
        Self {
            started_in_think: chat_render::started_in_think(
                prompt_tokens,
                think_open_id,
                think_close_id,
            ),
            watched_think_open_id: if tail_closed { None } else { think_open_id },
            think_close_id,
            reasoning_format: tok.reasoning_format(),
            tool_call_open_id: tok.tool_call_open_id(),
            tool_call_close_id: tok.tool_call_close_id(),
            tools_active,
        }
    }

    /// The reasoning splitter this shape implies.
    fn splitter(&self) -> ReasoningSplitter {
        ReasoningSplitter::with_ids(
            self.started_in_think,
            self.watched_think_open_id,
            self.think_close_id,
        )
        .with_format(self.reasoning_format)
        .ending_at_tool_call(self.tool_call_open_id)
    }
}

/// Whether a rendered prompt's generation tail already closed a think block
/// — its text ends with `</think>` plus whitespace, the shape of a template's
/// `enable_thinking: false` generation prompt (`…<think>\n\n</think>\n\n`).
pub(crate) fn prompt_tail_closes_think(rendered: &str) -> bool {
    rendered.trim_end().ends_with("</think>")
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// One response's decode → reasoning split → tool-call extraction → stop
/// pipeline — see the module doc.
pub(crate) struct ResponsePipeline<S> {
    decoder: PieceDecoder,
    splitter: ReasoningSplitter,
    extractor: Option<ToolCallStreamExtractor>,
    stop: S,
    calls: usize,
    unclosed: bool,
    finished: bool,
}

/// A whole non-streamed response, collected from its pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct CollectedResponse {
    /// The reasoning channel; `None` when the model did not reason (or
    /// reasoned only whitespace).
    pub(crate) reasoning_content: Option<String>,
    /// The content channel.
    pub(crate) content: String,
    /// The tool calls, in order.
    pub(crate) tool_calls: Vec<ToolCall>,
    /// How the response ended.
    pub(crate) end: ResponseEnd,
}

impl<S: ContentStop> ResponsePipeline<S> {
    /// A pipeline for one response of `shape`, stopping on `stop`.
    pub(crate) fn new(shape: &ResponseShape, tokenizer: Option<&TokenizerBridge>, stop: S) -> Self {
        Self {
            decoder: PieceDecoder::new(tokenizer),
            splitter: shape.splitter(),
            extractor: shape.tools_active.then(|| {
                ToolCallStreamExtractor::new(shape.tool_call_open_id, shape.tool_call_close_id)
            }),
            stop,
            calls: 0,
            unclosed: false,
            finished: false,
        }
    }

    /// Whether a stop sequence has matched (nothing more is released).
    pub(crate) fn stopped(&self) -> bool {
        self.stop.stopped()
    }

    /// Feed one generated token; returns what is now safe to release.
    pub(crate) fn push(
        &mut self,
        tokenizer: Option<&TokenizerBridge>,
        id: u32,
    ) -> Vec<ResponseEvent> {
        let mut out = Vec::new();
        if self.finished || self.stop.stopped() {
            return out;
        }
        // An id with no text of its own (a special-flagged marker, a partial
        // UTF-8 byte, no tokenizer) still reaches the splitter and the
        // extractor by id.
        let text = match self.decoder.next(tokenizer, id) {
            DecodedPiece::Text(text) => text,
            DecodedPiece::Pending | DecodedPiece::Omitted => String::new(),
        };
        if !self.splitter.in_reasoning() && self.stop.is_stop_id(id) {
            self.end_content(&mut out);
            self.stop.mark_stopped();
            return out;
        }
        match self.splitter.push(id, &text) {
            ReasoningChunk::Reasoning(text) => {
                if !text.is_empty() {
                    out.push(ResponseEvent::Reasoning(text));
                }
            }
            ReasoningChunk::Boundary => {}
            ReasoningChunk::Content(text) => {
                let events = match self.extractor.as_mut() {
                    Some(extractor) => extractor.push(id, &text),
                    None if text.is_empty() => Vec::new(),
                    None => vec![ToolStreamEvent::Content(text)],
                };
                self.route(events, &mut out);
            }
        }
        out
    }

    /// End the response: release whatever is still held back and report how
    /// it ended. Idempotent.
    pub(crate) fn finish(&mut self) -> (Vec<ResponseEvent>, ResponseEnd) {
        let mut out = Vec::new();
        if !self.finished && !self.stop.stopped() {
            self.end_content(&mut out);
        }
        self.finished = true;
        (
            out,
            ResponseEnd {
                calls: self.calls,
                stopped: self.stop.stopped(),
                unclosed: self.unclosed,
            },
        )
    }

    /// Run every id of a finished generation through the pipeline and
    /// collect the response (the non-streaming twin of feeding a stream).
    pub(crate) fn collect(
        mut self,
        tokenizer: Option<&TokenizerBridge>,
        ids: &[u32],
    ) -> CollectedResponse {
        let mut reasoning = String::new();
        let mut collected = CollectedResponse::default();
        let mut absorb = |events: Vec<ResponseEvent>, collected: &mut CollectedResponse| {
            for event in events {
                match event {
                    ResponseEvent::Reasoning(text) => reasoning.push_str(&text),
                    ResponseEvent::Content(text) => collected.content.push_str(&text),
                    ResponseEvent::ToolCall(call) => collected.tool_calls.push(call),
                }
            }
        };
        for &id in ids {
            let events = self.push(tokenizer, id);
            absorb(events, &mut collected);
        }
        let (events, end) = self.finish();
        absorb(events, &mut collected);
        collected.end = end;
        // The splitter never releases a whitespace-only span; the check keeps
        // the reference parser's rule explicit where the field is decided.
        collected.reasoning_content = (!reasoning
            .chars()
            .all(|c| matches!(c, ' ' | '\n' | '\r' | '\t')))
        .then_some(reasoning);
        collected
    }

    /// The content channel has ended (the response ended, or a stop token
    /// matched): release the extractor's held text and an unclosed block as
    /// content, then everything the stop stage still holds.
    fn end_content(&mut self, out: &mut Vec<ResponseEvent>) {
        if let Some(extractor) = self.extractor.as_mut() {
            let (events, end) = extractor.finish();
            if end.unclosed {
                self.unclosed = true;
                tracing::debug!(
                    "a <tool_call> block was still open when the response ended; its text is \
                     released as content"
                );
            }
            self.route(events, out);
        }
        let held = self.stop.release_held();
        push_content(out, held);
    }

    /// Route the extractor's events through the stop stage.
    fn route(&mut self, events: Vec<ToolStreamEvent>, out: &mut Vec<ResponseEvent>) {
        for event in events {
            if self.stop.stopped() {
                return;
            }
            match event {
                ToolStreamEvent::Content(text) => {
                    let released = self.stop.push_text(&text);
                    push_content(out, released);
                }
                ToolStreamEvent::Call(call) => {
                    // The content before a call is complete: nothing held
                    // back can grow into a stop sequence any more.
                    let held = self.stop.release_held();
                    push_content(out, held);
                    self.calls += 1;
                    out.push(ResponseEvent::ToolCall(call));
                }
            }
        }
    }
}

/// Push `text` as a content event, merging with a directly preceding content
/// event and skipping empty text.
fn push_content(out: &mut Vec<ResponseEvent>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(ResponseEvent::Content(previous)) = out.last_mut() {
        previous.push_str(&text);
        return;
    }
    out.push(ResponseEvent::Content(text));
}

// ── Streaming ────────────────────────────────────────────────────────────────

/// A blocking generation's result as the stream sees it: the number of
/// tokens generated, or the error's message.
pub(crate) type GenerationOutcome = Result<usize, String>;

/// An endpoint's SSE chunk shapes.
pub(crate) trait StreamChunks: Send + 'static {
    /// The first chunk: the `assistant` role delta.
    fn role(&self) -> String;
    /// A `reasoning_content` delta.
    fn reasoning(&self, text: String) -> String;
    /// A `content` delta.
    fn content(&self, text: String) -> String;
    /// A `tool_calls` delta carrying call number `index` (0-based) whole.
    fn tool_call(&self, index: usize, call: &ToolCall) -> String;
    /// The payloads that end the stream: for `Ok((finish_reason,
    /// generated))` the final chunk carrying `finish_reason`, followed by
    /// the usage chunk when the client asked for it; for `Err(message)` the
    /// error object a failed generation ends the stream with.
    fn terminal(&self, outcome: Result<(&'static str, usize), String>) -> Vec<String>;
}

/// Drives one streamed response: feeds each generated token through the
/// [`ResponsePipeline`], sends each released event as its SSE chunk, then
/// the final chunk (and the usage chunk) once the generation reports its
/// outcome. Runs as its own task, so it keeps going however fast the client
/// reads, and cancels the generation when a stop sequence matches or the
/// client goes away.
pub(crate) struct StreamDriver<S, C> {
    /// The server (its tokenizer decodes each token).
    pub(crate) state: Arc<AppState>,
    /// The response's pipeline.
    pub(crate) pipeline: ResponsePipeline<S>,
    /// The endpoint's chunk shapes.
    pub(crate) chunks: C,
    /// Cancels the generation.
    pub(crate) cancel: CancellationToken,
    /// The request's token budget (for `finish_reason: "length"`).
    pub(crate) max_tokens: usize,
    /// The per-request rate tracker fed one sample per generated token.
    pub(crate) rate_tracker:
        Option<Arc<std::sync::Mutex<crate::request_metrics::RequestRateTracker>>>,
}

impl<S: ContentStop, C: StreamChunks> StreamDriver<S, C> {
    /// Run the stream to its end. `token_rx` carries the generated ids,
    /// `outcome_rx` the generation's outcome, and `payload_tx` takes each
    /// SSE payload in order.
    pub(crate) async fn run(
        mut self,
        mut token_rx: tokio::sync::mpsc::UnboundedReceiver<u32>,
        outcome_rx: tokio::sync::oneshot::Receiver<GenerationOutcome>,
        payload_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) {
        if payload_tx.send(self.chunks.role()).is_err() {
            self.cancel.cancel();
            return;
        }
        let mut calls_sent = 0usize;
        while let Some(id) = token_rx.recv().await {
            self.record_token();
            // Inside a held-back tool-call block nothing is sent for a while,
            // so a vanished client is noticed here, not only on a send.
            if payload_tx.is_closed() {
                self.cancel.cancel();
                return;
            }
            let was_stopped = self.pipeline.stopped();
            let events = self.pipeline.push(self.state.tokenizer(), id);
            if !self.send_events(&payload_tx, events, &mut calls_sent) {
                self.cancel.cancel();
                return;
            }
            if !was_stopped && self.pipeline.stopped() {
                self.cancel.cancel();
            }
        }
        let (events, end) = self.pipeline.finish();
        if !self.send_events(&payload_tx, events, &mut calls_sent) {
            return;
        }
        let outcome = outcome_rx.await.unwrap_or_else(|_| {
            Err("the generation task ended without reporting an outcome".to_string())
        });
        if let Err(message) = &outcome {
            tracing::error!(error = %message, "streaming generation failed mid-stream");
        }
        let max_tokens = self.max_tokens;
        let terminal = self.chunks.terminal(
            outcome.map(|generated| (end.finish_reason(generated, max_tokens), generated)),
        );
        for payload in terminal {
            if payload_tx.send(payload).is_err() {
                return;
            }
        }
    }

    /// Record one generated token on the rate tracker.
    fn record_token(&self) {
        let Some(tracker) = self.rate_tracker.as_ref() else {
            return;
        };
        if let Ok(mut tracker) = tracker.lock() {
            if tracker.tokens_emitted() == 0 {
                tracker.record_first_token();
            } else {
                tracker.record_token();
            }
        }
    }

    /// Send `events` as chunks; `false` once the client is gone.
    fn send_events(
        &self,
        payload_tx: &tokio::sync::mpsc::UnboundedSender<String>,
        events: Vec<ResponseEvent>,
        calls_sent: &mut usize,
    ) -> bool {
        for event in events {
            let payload = match event {
                ResponseEvent::Reasoning(text) => self.chunks.reasoning(text),
                ResponseEvent::Content(text) => self.chunks.content(text),
                ResponseEvent::ToolCall(call) => {
                    let payload = self.chunks.tool_call(*calls_sent, &call);
                    *calls_sent += 1;
                    payload
                }
            };
            if payload_tx.send(payload).is_err() {
                return false;
            }
        }
        true
    }
}

/// The `delta.tool_calls` entry of a streamed tool call: the whole call in
/// one delta — `{"index", "id", "type": "function", "function": {"name",
/// "arguments"}}` — at its 0-based `index` among the response's calls.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct StreamToolCallDelta {
    /// The call's position among the response's calls.
    pub(crate) index: usize,
    /// The call itself (`id`, `type`, `function`).
    #[serde(flatten)]
    pub(crate) call: ToolCall,
}

#[cfg(test)]
#[path = "response_pipeline_tests.rs"]
mod tests;
