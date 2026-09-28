//! `<think>` reasoning-content splitting (bonsai2-design.md §5.3).
//!
//! Bonsai 2's chat template always starts an assistant generation with a
//! bare `<think>\n` (single token id **248068**) and the model's own output
//! later closes it with `</think>` (single token id **248069**) before its
//! real answer. [`ReasoningSplitter`] turns the raw per-token generation
//! stream into the two channels OpenAI's `reasoning_content`/`content`
//! response fields need, switching on **token id**, never a substring
//! match — `<think>`/`</think>` are atomic tokens (design Appendix A.1), so
//! a boundary can never be split across a decode chunk the way a
//! string-matched stop sequence could (RT-06/TOK-M2's chunk-boundary-leak
//! class does not exist here by construction: `push` classifies exactly one
//! already-decoded token at a time, so the caller can never observe a
//! "half a marker" state to leak).
//!
//! # Vocabularies without `<think>` (RT-10 correction)
//!
//! A vocabulary that defines no `<think>`/`</think>` tokens at all must not
//! be pushed into "reasoning" mode just because the *caller* asked for
//! thinking — there is no way for such a model to ever close it.
//! [`ReasoningSplitter::new`] takes `close_id: Option<u32>`: `None` (the ids
//! did not resolve against the loaded tokenizer) puts the splitter into
//! permanent pass-through, where every [`ReasoningSplitter::push`] call
//! returns [`ReasoningChunk::Content`] and [`ReasoningSplitter::in_reasoning`]
//! is always `false`. Resolve the ids once per loaded model (the
//! vocabulary's `<think>`/`</think>` piece lookup) and gate on **both** that
//! resolution *and* the rendered prompt before ever constructing a splitter
//! that starts `started_in_think = true`.
//!
//! The shipped Qwen3 1.7B/8B vocabulary (`models/tokenizer.json`) is **not**
//! such a vocabulary: it defines `<think>` = 151667 and `</think>` = 151668
//! as ordinary (non-`special`) added tokens. Under a template that does not
//! open a think span itself (the ChatML fallback), such a model emits its
//! **own** `<think>` as the first token of its answer;
//! [`ReasoningSplitter::with_ids`] / [`ReasoningSplitter::watching_open`]
//! recognise that model-emitted opener — only at the very start of the
//! output, exactly like the reference parser's optional leading
//! `<think>…</think>` block — and route the span to `reasoning_content`.
//!
//! # Whitespace: exactly what the reference parser strips
//!
//! The reference server (PrismML's llama.cpp fork) picks its output parser
//! from the chat template, and the two parsers this crate's templates meet
//! differ in how much whitespace they consume around the think span.
//! [`ReasoningFormat`] names them; [`ReasoningSplitter::with_format`] picks
//! one (the bridge classifies each template —
//! `TokenizerBridge::reasoning_format`).
//!
//! * [`ReasoningFormat::Tagged`] (the default) — the generic tagged parser
//!   (`common/chat-auto-parser-generator.cpp`,
//!   `optional(optspace(start) + reasoning(until(end)) + optspace(end))`,
//!   with `optspace` in `common/chat-peg-parser.cpp`): `optspace` consumes
//!   a marker's own surrounding whitespace characters, each optional, and
//!   nothing more. After a model-emitted `<think>` that is its one framing
//!   `\n`; after `</think>` it is the end marker's two `\n` (the templates'
//!   `'\n</think>\n\n' + content` convention), counted across however many
//!   tokens they arrive in. Any other leading whitespace of the reasoning,
//!   or of the answer, is kept.
//! * [`ReasoningFormat::Qwen3Coder`] — the fork's dedicated parser for
//!   templates that teach the `<tool_call><function=…><parameter=…>` XML
//!   form (`common/chat.cpp`, `common_chat_params_init_qwen3_coder`, which
//!   Bonsai 2's own template selects): `"<think>" + space()` drops **every**
//!   leading whitespace character of the reasoning, the reasoning also ends
//!   (unconsumed) at `<tool_call>`, and `reasoning << content` — whose `<<`
//!   is `space()` — drops every leading whitespace character of the answer,
//!   whether or not a reasoning block preceded it. `space()` is C
//!   `isspace`: space, `\t`, `\n`, `\v`, `\f`, `\r`.
//!
//! In both formats reasoning that consists only of whitespace (space, `\n`,
//! `\r`, `\t` — the reference parser's own set) is discarded, never reported
//! (for example the empty `<think>\n\n</think>` a template pre-fills), and
//! the trailing whitespace of the reasoning (the `\n` before `</think>`) is
//! kept. The splitter implements all of this once, for the streaming and the
//! non-streaming path alike: whitespace a format may still keep is held
//! back, released together with the next non-whitespace text, and discarded
//! if the span ends (or the generation ends) before any arrives — so the
//! concatenation of a stream's deltas always equals [`split_reasoning`]'s
//! result, and a stream never sends a reasoning delta for a response whose
//! non-streaming twin has none.
//!
//! The G8 golden (the reference server's own `/v1/chat/completions` response
//! for "What is 2+2? Answer briefly." on the real 27B, replayed by
//! `g8_real_trace_*` below) keeps `reasoning_content`'s trailing
//! `"...Final: 4.\n"` and reports `content` as the bare `"4"`: the `\n\n`
//! after `</think>` reaches neither channel, in both formats.

// ── Chunk classification ────────────────────────────────────────────────────

/// Which channel one [`ReasoningSplitter::push`] call's piece belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningChunk {
    /// Reasoning-channel text (accumulates into `reasoning_content`).
    Reasoning(String),
    /// Answer-channel text (accumulates into `content`).
    Content(String),
    /// A structural boundary: a think marker itself, or whitespace the
    /// reference parser consumes as markup (see the module doc). Contributes
    /// to neither channel and is safe to drop.
    Boundary,
}

impl ReasoningChunk {
    /// The text this chunk carries (`""` for [`Self::Boundary`]).
    pub fn as_str(&self) -> &str {
        match self {
            ReasoningChunk::Reasoning(s) | ReasoningChunk::Content(s) => s.as_str(),
            ReasoningChunk::Boundary => "",
        }
    }

    /// `true` for [`Self::Reasoning`].
    pub fn is_reasoning(&self) -> bool {
        matches!(self, ReasoningChunk::Reasoning(_))
    }

    /// `true` for [`Self::Content`].
    pub fn is_content(&self) -> bool {
        matches!(self, ReasoningChunk::Content(_))
    }
}

/// Which reference output parser's whitespace rules a [`ReasoningSplitter`]
/// follows — see the module doc's "Whitespace" section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// The generic tagged parser: only the markers' own framing newlines are
    /// markup.
    #[default]
    Tagged,
    /// The Qwen3-Coder-family parser: every leading whitespace character of
    /// the reasoning and of the answer is markup, and `<tool_call>` also
    /// ends the reasoning.
    Qwen3Coder,
}

// ── Splitter ─────────────────────────────────────────────────────────────────

/// Internal phase of a [`ReasoningSplitter`]. Not `pub`: the only observable
/// states are "in reasoning" ([`ReasoningSplitter::in_reasoning`]) and
/// "not".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Nothing has been emitted yet and the model may still open its own
    /// think span ([`ReasoningSplitter::watching_open`]): the open id starts
    /// reasoning, any non-empty piece settles into the answer.
    AwaitingOpen,
    /// A model-emitted `<think>` was just consumed; its framing whitespace
    /// (see [`ReasoningFormat`]) is still to be swallowed.
    JustOpened,
    /// Emitting into the reasoning channel.
    Reasoning,
    /// The close token was just consumed; the next piece(s) may still be
    /// swallowed by the post-boundary newline skip, up to the fixed budget
    /// carried here (see [`POST_CLOSE_NEWLINE_BUDGET`]; tagged format only).
    /// Once a piece exhausts the remaining budget, or the budget is already
    /// `0`, every further piece — newlines included — is real content.
    JustClosed(u8),
    /// The answer has not shown a non-whitespace character yet, and the
    /// format drops its leading whitespace (Qwen3-Coder format only).
    LeadingContent,
    /// Emitting into the content channel.
    Content,
}

/// How many leading newline characters immediately after `</think>` the
/// tagged format swallows, **in total**, however many separate decode tokens
/// they are split across: the end marker's own two (`'\n</think>\n\n'`), the
/// bound of the reference parser's `optspace(end)`. Anything beyond that is
/// real model output, not template boilerplate, even if it happens to also be
/// a bare newline.
const POST_CLOSE_NEWLINE_BUDGET: u8 = 2;

/// Whether `c` is whitespace in the reference parser's sense for discarding
/// a whitespace-only `reasoning_content` (space, `\n`, `\r`, `\t` —
/// deliberately not Unicode `White_Space`).
fn is_reference_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\n' | '\r' | '\t')
}

/// Whether `c` is whitespace in the sense of the reference parser's
/// `space()` (C `isspace`): space, `\t`, `\n`, `\v`, `\f`, `\r`.
fn is_parser_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r')
}

/// Whether `text` is empty or consists only of reference whitespace.
fn is_whitespace_only(text: &str) -> bool {
    text.chars().all(is_reference_whitespace)
}

/// Splits a raw, per-token generation stream into `reasoning_content` and
/// `content`. See the module docs for the id-keyed design, the
/// pass-through contract for vocabularies without `<think>`, and the
/// whitespace rules of each [`ReasoningFormat`].
#[derive(Debug, Clone)]
pub struct ReasoningSplitter {
    /// `Some(id)` only when both `<think>`/`</think>` resolved against the
    /// loaded model's vocabulary; `None` is permanent pass-through.
    close_id: Option<u32>,
    /// `Some(id)` of `<think>` when a model-emitted opener is recognised at
    /// the start of the output ([`Self::watching_open`]).
    open_id: Option<u32>,
    /// `Some(id)` of `<tool_call>`, which also ends the reasoning in the
    /// Qwen3-Coder format ([`Self::ending_at_tool_call`]).
    tool_call_id: Option<u32>,
    format: ReasoningFormat,
    phase: Phase,
    /// The reasoning span's leading whitespace-only run, held back until
    /// the span proves to carry non-whitespace text (then released in front
    /// of it) or ends without any (then discarded) — see the module doc.
    held_whitespace: String,
    /// Whether the current reasoning span has released any text yet (after
    /// which whitespace is ordinary reasoning text, never held back).
    reasoning_released: bool,
}

impl ReasoningSplitter {
    /// Build a splitter.
    ///
    /// * `started_in_think` — whether the rendered prompt already ends
    ///   inside a think span (Bonsai 2 with thinking enabled: always
    ///   `true`, since the generation prompt itself ends with `<think>\n`).
    /// * `close_id` — `Some(id)` of the model's `</think>` token only when
    ///   the vocabulary actually has one; otherwise `None` (RT-10
    ///   correction — see the module doc).
    ///
    /// A splitter built this way never recognises a model-emitted `<think>`;
    /// use [`Self::with_ids`] (or [`Self::watching_open`]) for that. It
    /// follows [`ReasoningFormat::Tagged`] until [`Self::with_format`] says
    /// otherwise.
    pub fn new(started_in_think: bool, close_id: Option<u32>) -> Self {
        let phase = if close_id.is_some() && started_in_think {
            Phase::Reasoning
        } else {
            Phase::Content
        };
        Self {
            close_id,
            open_id: None,
            tool_call_id: None,
            format: ReasoningFormat::Tagged,
            phase,
            held_whitespace: String::new(),
            reasoning_released: false,
        }
    }

    /// Build a splitter that also recognises a model-emitted `<think>`
    /// opener: [`Self::new`]`(started_in_think, close_id)` followed by
    /// [`Self::watching_open`]`(open_id)`.
    pub fn with_ids(started_in_think: bool, open_id: Option<u32>, close_id: Option<u32>) -> Self {
        Self::new(started_in_think, close_id).watching_open(open_id)
    }

    /// Also recognise a model-emitted `<think>` (`open_id`) at the very
    /// start of the output (builder).
    ///
    /// Takes effect only when the prompt did **not** already open a think
    /// span (`started_in_think == false`) and both `open_id` and the close
    /// id resolved (and differ). The opener is honoured only before any
    /// non-empty content has been classified — the reference parser's
    /// optional reasoning block sits at the very start of the output, so a
    /// `<think>` appearing mid-answer stays ordinary content. After the
    /// opener, its framing whitespace is swallowed (per [`ReasoningFormat`]),
    /// and the span then closes on `</think>` exactly like a prompt-opened
    /// one. A prompt whose generation tail already carries a *closed* think
    /// block must not watch: the reference parser's one optional reasoning
    /// block was consumed by the prompt itself.
    #[must_use]
    pub fn watching_open(mut self, open_id: Option<u32>) -> Self {
        let usable = matches!((open_id, self.close_id), (Some(open), Some(close)) if open != close);
        if usable && matches!(self.phase, Phase::Content | Phase::LeadingContent) {
            self.open_id = open_id;
            self.phase = Phase::AwaitingOpen;
        }
        self
    }

    /// Follow `format`'s whitespace rules (builder); see [`ReasoningFormat`].
    #[must_use]
    pub fn with_format(mut self, format: ReasoningFormat) -> Self {
        self.format = format;
        if self.phase == Phase::Content {
            // The answer starts right away (no reasoning span can open):
            // under the Qwen3-Coder rules its leading whitespace is markup.
            self.phase = self.content_start();
        }
        self
    }

    /// End the reasoning at the vocabulary's `<tool_call>` token too, in the
    /// Qwen3-Coder format (builder): that token is classified as content
    /// (it opens the call the answer carries), exactly like the reference
    /// parser's `peek("<tool_call>")`. No effect in the tagged format.
    #[must_use]
    pub fn ending_at_tool_call(mut self, tool_call_id: Option<u32>) -> Self {
        self.tool_call_id = tool_call_id;
        self
    }

    /// The whitespace rules this splitter follows.
    pub fn format(&self) -> ReasoningFormat {
        self.format
    }

    /// Leave the reasoning span on its close marker: whatever whitespace is
    /// still held back belongs to a whitespace-only span and is discarded.
    fn close_reasoning(&mut self) -> ReasoningChunk {
        self.held_whitespace.clear();
        self.reasoning_released = false;
        self.phase = match self.format {
            ReasoningFormat::Tagged => Phase::JustClosed(0),
            ReasoningFormat::Qwen3Coder => Phase::LeadingContent,
        };
        ReasoningChunk::Boundary
    }

    /// Whether `id` ends the reasoning without being consumed (the
    /// Qwen3-Coder format's `<tool_call>`).
    fn ends_reasoning_unconsumed(&self, id: u32) -> bool {
        self.format == ReasoningFormat::Qwen3Coder && self.tool_call_id == Some(id)
    }

    /// Leave the reasoning at a `<tool_call>` (Qwen3-Coder format): the
    /// token itself opens the answer.
    fn end_reasoning_at_tool_call(&mut self, piece: &str) -> ReasoningChunk {
        self.held_whitespace.clear();
        self.reasoning_released = false;
        self.phase = Phase::Content;
        ReasoningChunk::Content(piece.to_string())
    }

    /// Classify one reasoning piece under the whitespace rules: the
    /// Qwen3-Coder format drops the span's leading whitespace; the tagged
    /// format holds a leading whitespace-only run back and releases it with
    /// the span's first non-whitespace text.
    fn reasoning_piece(&mut self, piece: &str) -> ReasoningChunk {
        if self.reasoning_released {
            return ReasoningChunk::Reasoning(piece.to_string());
        }
        if self.format == ReasoningFormat::Qwen3Coder {
            let trimmed = piece.trim_start_matches(is_parser_space);
            if trimmed.is_empty() {
                return ReasoningChunk::Reasoning(String::new());
            }
            self.reasoning_released = true;
            return ReasoningChunk::Reasoning(trimmed.to_string());
        }
        if is_whitespace_only(piece) {
            self.held_whitespace.push_str(piece);
            return ReasoningChunk::Reasoning(String::new());
        }
        self.reasoning_released = true;
        let mut released = std::mem::take(&mut self.held_whitespace);
        released.push_str(piece);
        ReasoningChunk::Reasoning(released)
    }

    /// Classify one answer piece while its leading whitespace is still being
    /// dropped (Qwen3-Coder format). An empty piece (an id with no text of
    /// its own) passes through as empty content, so the caller still sees
    /// its id.
    fn leading_content_piece(&mut self, piece: &str) -> ReasoningChunk {
        if piece.is_empty() {
            return ReasoningChunk::Content(String::new());
        }
        let trimmed = piece.trim_start_matches(is_parser_space);
        if trimmed.is_empty() {
            return ReasoningChunk::Boundary;
        }
        self.phase = Phase::Content;
        ReasoningChunk::Content(trimmed.to_string())
    }

    /// The phase a response's answer starts in when no reasoning span opens.
    fn content_start(&self) -> Phase {
        match self.format {
            ReasoningFormat::Tagged => Phase::Content,
            ReasoningFormat::Qwen3Coder => Phase::LeadingContent,
        }
    }

    /// Convenience: build a pass-through splitter (every push is `Content`).
    /// Equivalent to `Self::new(false, None)`, spelled out for callers that
    /// already know they have no `<think>` ids for this model.
    pub fn pass_through() -> Self {
        Self::new(false, None)
    }

    /// Whether this splitter can ever split at all (`close_id` resolved),
    /// as opposed to permanent pass-through.
    pub fn is_active(&self) -> bool {
        self.close_id.is_some()
    }

    /// Whether the splitter is currently inside the reasoning span. Used to
    /// gate other per-token decisions on the *unsplit* raw stream — cli-11's
    /// correction: a stop-sequence match must be evaluated **after** this
    /// split, never before, or a stop string occurring inside the reasoning
    /// span would truncate the real answer before it is even reached.
    pub fn in_reasoning(&self) -> bool {
        matches!(self.phase, Phase::Reasoning | Phase::JustOpened)
    }

    /// Feed one generated token (its id and already-decoded piece text) and
    /// classify it.
    ///
    /// `piece` must be the correctly decoded text for `id` alone (e.g. via
    /// [`oxibonsai_tokenizer::StreamingDecoder`], which handles a
    /// multi-byte UTF-8 character split across token boundaries) — this
    /// function does not itself decode anything.
    ///
    /// A [`ReasoningChunk::Reasoning`] or [`ReasoningChunk::Content`] may
    /// carry an empty string (whitespace held back or dropped under the
    /// format's rules, or an id with no text of its own) — callers skip
    /// empty chunks, exactly as they already do for an empty decoded piece.
    pub fn push(&mut self, id: u32, piece: &str) -> ReasoningChunk {
        match self.phase {
            Phase::AwaitingOpen => {
                if self.open_id == Some(id) {
                    self.phase = Phase::JustOpened;
                    return ReasoningChunk::Boundary;
                }
                if piece.is_empty() {
                    return ReasoningChunk::Content(String::new());
                }
                // Any real output settles the answer: a `<think>` after it
                // is text, not a reasoning opener.
                self.phase = self.content_start();
                if self.phase == Phase::LeadingContent {
                    return self.leading_content_piece(piece);
                }
                ReasoningChunk::Content(piece.to_string())
            }
            Phase::JustOpened => {
                if self.close_id == Some(id) {
                    return self.close_reasoning();
                }
                if self.ends_reasoning_unconsumed(id) {
                    return self.end_reasoning_at_tool_call(piece);
                }
                if piece.is_empty() {
                    return ReasoningChunk::Reasoning(String::new());
                }
                // The opener's own framing newline (`<think>\n`) is markup,
                // exactly like the `\n` a template writes after its own
                // `<think>` in the prompt; the Qwen3-Coder format drops every
                // leading whitespace character of the span.
                self.phase = Phase::Reasoning;
                let rest = match self.format {
                    ReasoningFormat::Tagged => piece.strip_prefix('\n').unwrap_or(piece),
                    ReasoningFormat::Qwen3Coder => piece,
                };
                self.reasoning_piece(rest)
            }
            Phase::Reasoning => {
                if self.close_id == Some(id) {
                    return self.close_reasoning();
                }
                if self.ends_reasoning_unconsumed(id) {
                    return self.end_reasoning_at_tool_call(piece);
                }
                self.reasoning_piece(piece)
            }
            Phase::JustClosed(swallowed) => {
                if piece.is_empty() {
                    // An id with no text of its own neither consumes the
                    // budget nor ends the gap; it still reaches the caller.
                    return ReasoningChunk::Content(String::new());
                }
                let budget = POST_CLOSE_NEWLINE_BUDGET.saturating_sub(swallowed);
                if budget == 0 {
                    // The fixed post-boundary budget is already spent by
                    // earlier tokens; this piece — even if it is itself
                    // more bare newlines — is real content from here on.
                    self.phase = Phase::Content;
                    return ReasoningChunk::Content(piece.to_string());
                }
                let leading = piece.chars().take_while(|&c| c == '\n').count();
                let take = leading.min(budget as usize);
                let trimmed = &piece[take..];
                if trimmed.is_empty() {
                    // Entirely swallowed by the post-boundary newline skip,
                    // within budget; stay in `JustClosed` (with the updated
                    // remaining budget) in case more leading newlines arrive
                    // as separate tokens.
                    self.phase = Phase::JustClosed(swallowed + take as u8);
                    return ReasoningChunk::Boundary;
                }
                self.phase = Phase::Content;
                ReasoningChunk::Content(trimmed.to_string())
            }
            Phase::LeadingContent => self.leading_content_piece(piece),
            Phase::Content => ReasoningChunk::Content(piece.to_string()),
        }
    }
}

/// Split an already-decoded `(id, piece)` sequence into `(reasoning_content,
/// content)` in one call — the shape a non-streaming response handler wants.
///
/// `reasoning_content` is `None` when nothing was ever classified as
/// reasoning (pass-through mode, or a splitter that never actually entered
/// the reasoning phase) and when the reasoning was whitespace-only (the
/// module doc's rule, applied by the same [`ReasoningSplitter`] the
/// streaming path uses), matching the OpenAI response contract of omitting
/// the field rather than sending an empty string.
///
/// Recognises no model-emitted `<think>`; see [`split_reasoning_with_ids`].
pub fn split_reasoning<'a, I>(
    tokens: I,
    started_in_think: bool,
    close_id: Option<u32>,
) -> (Option<String>, String)
where
    I: IntoIterator<Item = (u32, &'a str)>,
{
    split_with(ReasoningSplitter::new(started_in_think, close_id), tokens)
}

/// [`split_reasoning`] with a model-emitted `<think>` (`open_id`) also
/// recognised at the start of the output — the non-streaming twin of
/// [`ReasoningSplitter::with_ids`].
pub fn split_reasoning_with_ids<'a, I>(
    tokens: I,
    started_in_think: bool,
    open_id: Option<u32>,
    close_id: Option<u32>,
) -> (Option<String>, String)
where
    I: IntoIterator<Item = (u32, &'a str)>,
{
    split_with(
        ReasoningSplitter::with_ids(started_in_think, open_id, close_id),
        tokens,
    )
}

/// Drive `splitter` over `tokens`, collecting both channels — the
/// non-streaming twin of feeding the same splitter token by token.
pub fn split_with<'a, I>(mut splitter: ReasoningSplitter, tokens: I) -> (Option<String>, String)
where
    I: IntoIterator<Item = (u32, &'a str)>,
{
    let mut reasoning = String::new();
    let mut content = String::new();
    for (id, piece) in tokens {
        match splitter.push(id, piece) {
            ReasoningChunk::Reasoning(s) => reasoning.push_str(&s),
            ReasoningChunk::Content(s) => content.push_str(&s),
            ReasoningChunk::Boundary => {}
        }
    }
    // The splitter never releases a whitespace-only span, so a reasoning
    // channel that is still whitespace-only here is empty; the check keeps
    // the reference parser's rule explicit at the one place the field is
    // decided.
    let reasoning = if is_whitespace_only(&reasoning) {
        None
    } else {
        Some(reasoning)
    };
    (reasoning, content)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic phase transitions ───────────────────────────────────────────

    #[test]
    fn pass_through_mode_is_always_content() {
        let mut splitter = ReasoningSplitter::pass_through();
        assert!(!splitter.is_active());
        assert!(!splitter.in_reasoning());
        assert_eq!(
            splitter.push(999, "hello"),
            ReasoningChunk::Content("hello".to_string())
        );
        assert_eq!(
            splitter.push(1000, "\nworld"),
            ReasoningChunk::Content("\nworld".to_string()),
            "pass-through mode must never swallow a leading newline — only \
             the real post-close boundary does that"
        );
    }

    #[test]
    fn active_but_not_started_in_think_is_content_from_the_start() {
        let mut splitter = ReasoningSplitter::new(false, Some(248069));
        assert!(splitter.is_active());
        assert!(!splitter.in_reasoning());
        assert_eq!(
            splitter.push(1, "hi"),
            ReasoningChunk::Content("hi".to_string())
        );
    }

    #[test]
    fn started_in_think_classifies_until_close_id() {
        let mut splitter = ReasoningSplitter::new(true, Some(248069));
        assert!(splitter.in_reasoning());
        assert_eq!(
            splitter.push(1, "thinking "),
            ReasoningChunk::Reasoning("thinking ".to_string())
        );
        assert!(splitter.in_reasoning());
        assert_eq!(splitter.push(248069, "</think>"), ReasoningChunk::Boundary);
        assert!(!splitter.in_reasoning());
        assert_eq!(
            splitter.push(2, "answer"),
            ReasoningChunk::Content("answer".to_string())
        );
    }

    #[test]
    fn close_id_never_matches_when_disabled() {
        // With `close_id: None`, an id that would coincidentally equal a
        // real close-think id must not be treated specially — nothing to
        // compare against.
        let mut splitter = ReasoningSplitter::new(true, None);
        assert!(!splitter.in_reasoning(), "never active without a close_id");
        assert_eq!(
            splitter.push(248069, "</think>"),
            ReasoningChunk::Content("</think>".to_string())
        );
    }

    // ── Post-boundary newline swallow (G8) ────────────────────────────────

    #[test]
    fn swallows_leading_newlines_once_after_close_only() {
        let mut splitter = ReasoningSplitter::new(true, Some(248069));
        let _ = splitter.push(1, "r");
        assert_eq!(splitter.push(248069, "</think>"), ReasoningChunk::Boundary);
        // A run of pure-newline tokens right after the boundary is dropped,
        // up to the fixed 2-newline budget (matching the reference fork's
        // `optspace(end)` bound, not an unbounded fixpoint strip)…
        assert_eq!(splitter.push(2, "\n"), ReasoningChunk::Boundary);
        assert_eq!(splitter.push(3, "\n"), ReasoningChunk::Boundary);
        // …the budget (2 newlines) is already spent by tokens 2 and 3, so a
        // THIRD bare-newline token is real content, not swallowed…
        assert_eq!(
            splitter.push(4, "\n\n4"),
            ReasoningChunk::Content("\n\n4".to_string()),
            "the post-boundary budget is per-splitter, not per-token: once spent, \
             even more newlines are real content"
        );
        // …and a later, real newline (mid-answer) is content, unstripped.
        assert_eq!(
            splitter.push(5, "\nmore"),
            ReasoningChunk::Content("\nmore".to_string())
        );
    }

    #[test]
    fn swallows_at_most_two_newlines_within_a_single_piece() {
        // The common case (G8's own shape): both newlines arrive in ONE
        // token immediately after the close, then content starts cleanly.
        let mut splitter = ReasoningSplitter::new(true, Some(248069));
        assert_eq!(splitter.push(248069, "</think>"), ReasoningChunk::Boundary);
        assert_eq!(splitter.push(1, "\n\n"), ReasoningChunk::Boundary);
        assert_eq!(
            splitter.push(2, "4"),
            ReasoningChunk::Content("4".to_string())
        );
    }

    #[test]
    fn a_single_piece_with_more_than_two_leading_newlines_keeps_the_extra_ones_as_content() {
        let mut splitter = ReasoningSplitter::new(true, Some(248069));
        assert_eq!(splitter.push(248069, "</think>"), ReasoningChunk::Boundary);
        // 3 leading newlines in one piece: only 2 are budget, the 3rd (and
        // the real text after it) is content.
        assert_eq!(
            splitter.push(1, "\n\n\nreal"),
            ReasoningChunk::Content("\nreal".to_string())
        );
    }

    #[test]
    fn content_with_no_leading_newline_after_close_is_untouched() {
        let mut splitter = ReasoningSplitter::new(true, Some(248069));
        assert_eq!(splitter.push(248069, "</think>"), ReasoningChunk::Boundary);
        assert_eq!(
            splitter.push(1, "4"),
            ReasoningChunk::Content("4".to_string())
        );
    }

    // ── G8: reproduces the reference server's prompt-1 response structurally ─
    //
    // `reasoning_content` == "We need to answer user: \"What is 2+2? Answer
    // briefly.\" Simple. Final: 4.\n" (kept, trailing newline and all) and
    // `content` == "4" exactly. This test proves the *splitter* reproduces
    // the documented pass/fail criterion (design §7.3 G8: "content ==
    // \"4\", reasoning non-empty") against a hand-built, collapsed token
    // stream shaped like the golden's own text; the real golden's actual 29
    // separate `(id, token)` pairs (`Ternary-Bonsai-2-27B-*.gguf` is on
    // disk in this environment as of 2026-09-21) are replayed verbatim,
    // both as one batch call and one token at a time, by
    // `g8_real_trace_split_reasoning_matches_the_golden_response_exactly` /
    // `g8_real_trace_streaming_replay_matches_the_golden_response_exactly`
    // below.
    #[test]
    fn g8_reasoning_split_matches_golden_shape() {
        const CLOSE_THINK_ID: u32 = 248069;
        let reasoning_text =
            "We need to answer user: \"What is 2+2? Answer briefly.\" Simple. Final: 4.\n";
        let tokens: Vec<(u32, &str)> = vec![
            (100, reasoning_text),
            (CLOSE_THINK_ID, "</think>"),
            (101, "\n\n"),
            (102, "4"),
        ];
        let (reasoning, content) = split_reasoning(tokens, true, Some(CLOSE_THINK_ID));
        assert_eq!(content, "4", "G8: content must be exactly \"4\"");
        assert_eq!(
            reasoning.as_deref(),
            Some(reasoning_text),
            "G8: reasoning_content must be non-empty and byte-identical to the \
             pre-boundary text, trailing newline included"
        );
    }

    /// The real golden trace behind G8 (the reference server's own
    /// `/v1/chat/completions` response,
    /// `choices[0].logprobs.content`, a real Bonsai 2 27B server response to
    /// "What is 2+2? Answer briefly."): the model's own 29 separate
    /// `(id, token)` pairs, extracted verbatim, not the single collapsed
    /// `(id, whole_reasoning_text)` shape
    /// [`g8_reasoning_split_matches_golden_shape`] above uses. That
    /// collapsed shape cannot exercise anything about how MANY separate
    /// tokens the post-boundary newline swallow spans, or what a
    /// special-flagged, empty-decoding EOS-shaped id (`248046`, the real
    /// trace's own last token, decoding to `""`) does to the split — this
    /// replay can. `close_id = 248069` and `248068` (the `<think>` opener)
    /// never appears among the 29: the real prompt already ends with
    /// `<think>\n`, so every one of these ids is a genuinely GENERATED
    /// token, `started_in_think: true` from the first one.
    const G8_REAL_TRACE: &[(u32, &str)] = &[
        (1596, "We"),
        (1144, " need"),
        (310, " to"),
        (4087, " answer"),
        (1156, " user"),
        (25, ":"),
        (328, " \""),
        (3710, "What"),
        (369, " is"),
        (220, " "),
        (17, "2"),
        (10, "+"),
        (17, "2"),
        (30, "?"),
        (21134, " Answer"),
        (25899, " briefly"),
        (1149, ".\""),
        (8722, " Simple"),
        (13, "."),
        (12650, " Final"),
        (25, ":"),
        (220, " "),
        (19, "4"),
        (13, "."),
        (198, "\n"),
        (248069, "</think>"),
        (271, "\n\n"),
        (19, "4"),
        (248046, ""),
    ];
    const G8_REAL_TRACE_REASONING: &str =
        "We need to answer user: \"What is 2+2? Answer briefly.\" Simple. Final: 4.\n";
    const G8_REAL_TRACE_CONTENT: &str = "4";

    #[test]
    fn g8_real_trace_split_reasoning_matches_the_golden_response_exactly() {
        let (reasoning, content) =
            split_reasoning(G8_REAL_TRACE.iter().copied(), true, Some(248069));
        assert_eq!(
            reasoning.as_deref(),
            Some(G8_REAL_TRACE_REASONING),
            "must byte-match the real server response's own reasoning_content"
        );
        assert_eq!(
            content, G8_REAL_TRACE_CONTENT,
            "must byte-match the real server response's own content"
        );
    }

    /// Same trace, replayed one token at a time through `push` — the shape
    /// `server/chat.rs`'s and `api_extensions.rs`'s streaming decode loops
    /// actually use, as opposed to [`split_reasoning`]'s one-shot batch
    /// call. In particular this exercises the newline swallow spanning
    /// exactly one piece (`(271, "\n\n")`, both newlines in a single token,
    /// budget exactly exhausted) and confirms a token that decodes to `""`
    /// (`248046`, standing in for a special-flagged token whose own
    /// `step_decode` returned `None` — the decode loops feed exactly this
    /// shape to `push`) is a
    /// harmless no-op once already in `Phase::Content`.
    #[test]
    fn g8_real_trace_streaming_replay_matches_the_golden_response_exactly() {
        let mut splitter = ReasoningSplitter::new(true, Some(248069));
        assert!(splitter.in_reasoning());
        let mut reasoning = String::new();
        let mut content = String::new();
        for &(id, piece) in G8_REAL_TRACE {
            match splitter.push(id, piece) {
                ReasoningChunk::Reasoning(s) => reasoning.push_str(&s),
                ReasoningChunk::Content(s) => content.push_str(&s),
                ReasoningChunk::Boundary => {}
            }
        }
        assert!(
            !splitter.in_reasoning(),
            "must have left the reasoning phase by the end"
        );
        assert_eq!(
            reasoning, G8_REAL_TRACE_REASONING,
            "streamed reasoning_content must byte-match the real server response"
        );
        assert_eq!(
            content, G8_REAL_TRACE_CONTENT,
            "streamed content must byte-match the real server response"
        );
    }

    #[test]
    fn split_reasoning_pass_through_has_no_reasoning_content() {
        let tokens: Vec<(u32, &str)> = vec![(1, "hello "), (2, "world")];
        let (reasoning, content) = split_reasoning(tokens, false, None);
        assert_eq!(reasoning, None);
        assert_eq!(content, "hello world");
    }

    #[test]
    fn split_reasoning_never_started_in_think_is_all_content() {
        let tokens: Vec<(u32, &str)> = vec![(1, "hello")];
        let (reasoning, content) = split_reasoning(tokens, false, Some(248069));
        assert_eq!(reasoning, None);
        assert_eq!(content, "hello");
    }

    #[test]
    fn split_reasoning_whitespace_only_reasoning_is_none() {
        // Matches the reference fork's parser: reasoning that is present
        // but carries only whitespace is discarded, the same as if there
        // had been no reasoning at all — never an OpenAI `reasoning_content`
        // field that is technically `Some("")`-ish but shows nothing.
        const CLOSE: u32 = 248069;
        let tokens: Vec<(u32, &str)> = vec![(1, "   \n  "), (CLOSE, "</think>"), (2, "answer")];
        let (reasoning, content) = split_reasoning(tokens, true, Some(CLOSE));
        assert_eq!(reasoning, None, "whitespace-only reasoning must be None");
        assert_eq!(content, "answer");
    }

    // ── Whitespace-only reasoning: one rule, both paths ──────────────────

    const OPEN: u32 = 248068;
    const CLOSE: u32 = 248069;

    /// Stream `tokens` through `splitter` exactly as the server's SSE loops
    /// do: collect every non-empty reasoning delta, the content, and the
    /// order the two channels first appeared in.
    fn stream(
        mut splitter: ReasoningSplitter,
        tokens: &[(u32, &str)],
    ) -> (Vec<String>, String, Option<bool>) {
        let mut deltas = Vec::new();
        let mut content = String::new();
        let mut reasoning_first = None;
        for &(id, piece) in tokens {
            match splitter.push(id, piece) {
                ReasoningChunk::Reasoning(s) if !s.is_empty() => {
                    reasoning_first.get_or_insert(true);
                    deltas.push(s);
                }
                ReasoningChunk::Content(s) if !s.is_empty() => {
                    reasoning_first.get_or_insert(false);
                    content.push_str(&s);
                }
                _ => {}
            }
        }
        (deltas, content, reasoning_first)
    }

    #[test]
    fn a_whitespace_only_reasoning_span_is_never_streamed() {
        let tokens: &[(u32, &str)] = &[(1, " "), (2, "\n"), (3, "\t\r\n"), (CLOSE, ""), (4, "hi")];
        let (deltas, content, _) = stream(ReasoningSplitter::new(true, Some(CLOSE)), tokens);
        assert!(
            deltas.is_empty(),
            "no reasoning delta may be sent for a whitespace-only span: {deltas:?}"
        );
        assert_eq!(content, "hi");
        let (reasoning, batch_content) = split_reasoning(tokens.iter().copied(), true, Some(CLOSE));
        assert_eq!(reasoning, None);
        assert_eq!(batch_content, "hi");
    }

    #[test]
    fn leading_whitespace_is_released_with_the_first_real_reasoning_text() {
        let mut splitter = ReasoningSplitter::new(true, Some(CLOSE));
        assert_eq!(
            splitter.push(1, " "),
            ReasoningChunk::Reasoning(String::new())
        );
        assert_eq!(
            splitter.push(2, "\n"),
            ReasoningChunk::Reasoning(String::new())
        );
        assert_eq!(
            splitter.push(3, "plan"),
            ReasoningChunk::Reasoning(" \nplan".to_string()),
            "the held whitespace is released in front of the first real text"
        );
        // Once the span has real text, whitespace is ordinary reasoning.
        assert_eq!(
            splitter.push(4, "\n"),
            ReasoningChunk::Reasoning("\n".to_string())
        );
    }

    /// Unicode whitespace outside the reference parser's four characters is
    /// real text (the old `trim()` check treated U+00A0 as whitespace).
    #[test]
    fn only_the_reference_whitespace_set_counts() {
        let tokens: &[(u32, &str)] = &[(1, "\u{a0}"), (CLOSE, ""), (2, "x")];
        let (reasoning, _) = split_reasoning(tokens.iter().copied(), true, Some(CLOSE));
        assert_eq!(reasoning.as_deref(), Some("\u{a0}"));
    }

    /// The streamed reasoning deltas concatenate to exactly the
    /// non-streaming `reasoning_content` (and `None` iff nothing streamed),
    /// and reasoning always precedes content, over a table of shapes.
    #[test]
    fn streaming_and_non_streaming_agree_on_every_shape() {
        let cases: Vec<Vec<(u32, &str)>> = vec![
            vec![(1, "a"), (CLOSE, ""), (2, "\n\nb")],
            vec![(1, "\n"), (2, "a"), (3, " "), (CLOSE, ""), (4, "b")],
            vec![(1, "  "), (CLOSE, "")],
            vec![(1, "  ")],
            vec![(CLOSE, ""), (1, "b")],
            vec![(1, "a"), (2, "  "), (3, "b")],
            vec![(OPEN, ""), (1, "\n"), (2, "x"), (CLOSE, ""), (3, "y")],
        ];
        for tokens in cases {
            for (started, open) in [(true, None), (false, Some(OPEN))] {
                let splitter = ReasoningSplitter::with_ids(started, open, Some(CLOSE));
                let (deltas, content, first) = stream(splitter, &tokens);
                let (reasoning, batch_content) =
                    split_reasoning_with_ids(tokens.iter().copied(), started, open, Some(CLOSE));
                let streamed = deltas.concat();
                assert_eq!(
                    reasoning,
                    (!streamed.is_empty()).then(|| streamed.clone()),
                    "{tokens:?} started_in_think={started}"
                );
                assert_eq!(content, batch_content, "{tokens:?}");
                if !deltas.is_empty() && !content.is_empty() {
                    assert_eq!(first, Some(true), "reasoning must precede content");
                }
            }
        }
    }

    // ── A model-emitted `<think>` ──────────────────────────────────────

    #[test]
    fn a_model_emitted_opener_at_the_start_routes_its_span_to_reasoning() {
        let tokens: &[(u32, &str)] = &[
            (OPEN, "<think>"),
            (1, "\n"),
            (2, "Okay"),
            (3, ", done.\n"),
            (CLOSE, "</think>"),
            (4, "\n\n"),
            (5, "Answer"),
        ];
        let (reasoning, content) =
            split_reasoning_with_ids(tokens.iter().copied(), false, Some(OPEN), Some(CLOSE));
        assert_eq!(
            reasoning.as_deref(),
            Some("Okay, done.\n"),
            "the opener and its framing newline are markup, not reasoning"
        );
        assert_eq!(content, "Answer");

        let mut splitter = ReasoningSplitter::with_ids(false, Some(OPEN), Some(CLOSE));
        assert!(!splitter.in_reasoning());
        assert_eq!(splitter.push(OPEN, "<think>"), ReasoningChunk::Boundary);
        assert!(splitter.in_reasoning());
    }

    #[test]
    fn an_opener_after_real_content_is_ordinary_text() {
        let tokens: &[(u32, &str)] =
            &[(1, "Hi "), (OPEN, "<think>"), (2, "x"), (CLOSE, "</think>")];
        let (reasoning, content) =
            split_reasoning_with_ids(tokens.iter().copied(), false, Some(OPEN), Some(CLOSE));
        assert_eq!(reasoning, None);
        assert_eq!(content, "Hi <think>x</think>");
    }

    #[test]
    fn empty_pieces_before_the_opener_do_not_settle_the_answer() {
        // A special-flagged token decodes to "" and must not stop the
        // splitter from recognising the opener that follows it.
        let tokens: &[(u32, &str)] = &[(9, ""), (OPEN, ""), (2, "r"), (CLOSE, ""), (3, "c")];
        let (reasoning, content) =
            split_reasoning_with_ids(tokens.iter().copied(), false, Some(OPEN), Some(CLOSE));
        assert_eq!(reasoning.as_deref(), Some("r"));
        assert_eq!(content, "c");
    }

    #[test]
    fn new_keeps_ignoring_a_model_emitted_opener() {
        // `ReasoningSplitter::new` is unchanged: no opener watching.
        let mut splitter = ReasoningSplitter::new(false, Some(CLOSE));
        assert_eq!(
            splitter.push(OPEN, "<think>"),
            ReasoningChunk::Content("<think>".to_string())
        );
        let tokens: &[(u32, &str)] = &[(OPEN, "<think>"), (1, "x")];
        let (reasoning, content) = split_reasoning(tokens.iter().copied(), false, Some(CLOSE));
        assert_eq!(reasoning, None);
        assert_eq!(content, "<think>x");
    }

    #[test]
    fn opener_watching_needs_distinct_resolved_ids_and_a_closed_prompt() {
        // Missing / identical ids: nothing to watch.
        for (open, close) in [
            (None, Some(CLOSE)),
            (Some(OPEN), None),
            (Some(CLOSE), Some(CLOSE)),
        ] {
            let mut splitter = ReasoningSplitter::with_ids(false, open, close);
            assert_eq!(
                splitter.push(OPEN, "<think>"),
                ReasoningChunk::Content("<think>".to_string()),
                "open={open:?} close={close:?}"
            );
        }
        // A prompt that already opened the span: the splitter starts in
        // reasoning, and a second opener is reasoning text.
        let mut splitter = ReasoningSplitter::with_ids(true, Some(OPEN), Some(CLOSE));
        assert!(splitter.in_reasoning());
        assert_eq!(
            splitter.push(OPEN, "<think>"),
            ReasoningChunk::Reasoning("<think>".to_string())
        );
    }

    /// The Qwen3 shape on the real shipped vocabulary (`OXI_TOKENIZER`,
    /// e.g. `models/tokenizer.json`): the vocabulary resolves `<think>` /
    /// `</think>` to 151667 / 151668, and a model-emitted span decoded
    /// token by token splits into `reasoning_content` and `content`.
    /// Self-skips (with a note) when the variable is unset.
    #[test]
    fn a_qwen3_model_emitted_span_splits_on_the_real_vocabulary() {
        let Some(path) = std::env::var_os("OXI_TOKENIZER").filter(|p| !p.is_empty()) else {
            eprintln!(
                "skipped: a_qwen3_model_emitted_span_splits_on_the_real_vocabulary needs \
                 OXI_TOKENIZER (the shipped Qwen3 tokenizer.json)"
            );
            return;
        };
        let path = path.to_string_lossy().into_owned();
        let tok = crate::tokenizer_bridge::TokenizerBridge::from_file(&path)
            .expect("the real tokenizer loads");
        assert_eq!(tok.think_open_id(), Some(151_667));
        assert_eq!(tok.think_close_id(), Some(151_668));
        let ids = tok
            .encode("<think>\nThe user greets me.\n</think>\n\nHello!")
            .expect("encode");
        assert_eq!(ids.first().copied(), Some(151_667), "{ids:?}");
        let mut state = tok.new_decode_stream(true);
        let mut pieces = Vec::new();
        for &id in &ids {
            let piece = tok
                .step_decode(&mut state, id)
                .expect("decode")
                .unwrap_or_default();
            pieces.push((id, piece));
        }
        let (reasoning, content) = split_reasoning_with_ids(
            pieces.iter().map(|(id, p)| (*id, p.as_str())),
            false,
            tok.think_open_id(),
            tok.think_close_id(),
        );
        assert_eq!(reasoning.as_deref(), Some("The user greets me.\n"));
        assert_eq!(content, "Hello!");
    }

    // ── The Qwen3-Coder format (the reference parser Bonsai 2's template
    //    selects): `"<think>" + space()` and `reasoning << content` ──────

    const TOOL_CALL: u32 = 248058;

    fn qwen3_coder(started_in_think: bool, open: Option<u32>) -> ReasoningSplitter {
        ReasoningSplitter::with_ids(started_in_think, open, Some(CLOSE))
            .with_format(ReasoningFormat::Qwen3Coder)
            .ending_at_tool_call(Some(TOOL_CALL))
    }

    /// Batch and token-by-token results of one splitter configuration.
    fn both_ways(
        splitter: ReasoningSplitter,
        tokens: &[(u32, &str)],
    ) -> ((Option<String>, String), (Vec<String>, String)) {
        let batch = split_with(splitter.clone(), tokens.iter().copied());
        let (deltas, content, _) = stream(splitter, tokens);
        (batch, (deltas, content))
    }

    #[test]
    fn qwen3_coder_drops_every_leading_whitespace_character_of_the_reasoning() {
        let tokens: &[(u32, &str)] = &[(1, "\n"), (2, " \t\u{0B}plan"), (3, "\n"), (CLOSE, "")];
        let ((reasoning, _), (deltas, _)) = both_ways(qwen3_coder(true, None), tokens);
        assert_eq!(
            reasoning.as_deref(),
            Some("plan\n"),
            "trailing whitespace is kept"
        );
        assert_eq!(deltas.concat(), "plan\n");
        // The tagged format keeps the model's own leading whitespace.
        let (tagged, _) = both_ways(ReasoningSplitter::new(true, Some(CLOSE)), tokens);
        assert_eq!(tagged.0.as_deref(), Some("\n \t\u{0B}plan\n"));
    }

    #[test]
    fn qwen3_coder_drops_every_leading_whitespace_character_after_the_close() {
        let tokens: &[(u32, &str)] = &[
            (1, "r"),
            (CLOSE, "</think>"),
            (2, "\n\n"),
            (3, "\n \t"),
            (4, "\r\n4"),
            (5, " \n"),
        ];
        let ((_, content), (_, streamed)) = both_ways(qwen3_coder(true, None), tokens);
        assert_eq!(content, "4 \n", "only the leading run is markup");
        assert_eq!(streamed, content);
        // The tagged format swallows only the end marker's own two newlines.
        let (tagged, _) = both_ways(ReasoningSplitter::new(true, Some(CLOSE)), tokens);
        assert_eq!(tagged.1, "\n \t\r\n4 \n");
    }

    #[test]
    fn qwen3_coder_tool_call_ends_the_reasoning_without_being_consumed() {
        let tokens: &[(u32, &str)] = &[
            (OPEN, "<think>"),
            (1, "\n"),
            (TOOL_CALL, "<tool_call>"),
            (2, "\n{\"name\": \"f\"}\n"),
        ];
        let ((reasoning, content), (deltas, streamed)) =
            both_ways(qwen3_coder(false, Some(OPEN)), tokens);
        assert_eq!(reasoning, None, "the whitespace-only span is discarded");
        assert!(deltas.is_empty());
        assert_eq!(content, "<tool_call>\n{\"name\": \"f\"}\n");
        assert_eq!(streamed, content);

        let mut splitter = qwen3_coder(true, None);
        assert_eq!(
            splitter.push(9, "thinking"),
            ReasoningChunk::Reasoning("thinking".to_string())
        );
        assert_eq!(
            splitter.push(TOOL_CALL, "<tool_call>"),
            ReasoningChunk::Content("<tool_call>".to_string())
        );
        assert!(!splitter.in_reasoning());

        // The tagged format keeps a `<tool_call>` inside the span as reasoning.
        let mut tagged =
            ReasoningSplitter::new(true, Some(CLOSE)).ending_at_tool_call(Some(TOOL_CALL));
        assert_eq!(
            tagged.push(TOOL_CALL, "<tool_call>"),
            ReasoningChunk::Reasoning("<tool_call>".to_string())
        );
    }

    #[test]
    fn qwen3_coder_strips_the_answers_leading_whitespace_without_any_reasoning_block() {
        // A prompt that closed its own think block (`enable_thinking: false`)
        // or a vocabulary without `<think>`: `reasoning << content` still
        // drops the answer's leading whitespace.
        for splitter in [
            ReasoningSplitter::new(false, Some(CLOSE)).with_format(ReasoningFormat::Qwen3Coder),
            ReasoningSplitter::pass_through().with_format(ReasoningFormat::Qwen3Coder),
        ] {
            let tokens: &[(u32, &str)] = &[(1, "\n"), (2, "  Hi"), (3, "\nthere")];
            let ((reasoning, content), (_, streamed)) = both_ways(splitter, tokens);
            assert_eq!(reasoning, None);
            assert_eq!(content, "Hi\nthere");
            assert_eq!(streamed, content);
        }
    }

    #[test]
    fn qwen3_coder_whitespace_before_a_model_opener_settles_the_answer() {
        // The reference parser's optional reasoning block must start the
        // output: after leading whitespace a `<think>` is ordinary text.
        let tokens: &[(u32, &str)] = &[(1, "\n"), (OPEN, "<think>"), (2, "x")];
        let ((reasoning, content), _) = both_ways(qwen3_coder(false, Some(OPEN)), tokens);
        assert_eq!(reasoning, None);
        assert_eq!(content, "<think>x");
    }

    #[test]
    fn watching_open_and_the_format_builder_compose_in_either_order() {
        let a = ReasoningSplitter::new(false, Some(CLOSE))
            .with_format(ReasoningFormat::Qwen3Coder)
            .watching_open(Some(OPEN));
        let b = ReasoningSplitter::new(false, Some(CLOSE))
            .watching_open(Some(OPEN))
            .with_format(ReasoningFormat::Qwen3Coder);
        for mut splitter in [a, b] {
            assert_eq!(splitter.format(), ReasoningFormat::Qwen3Coder);
            assert_eq!(splitter.push(OPEN, "<think>"), ReasoningChunk::Boundary);
            assert!(splitter.in_reasoning());
        }
    }

    #[test]
    fn g8_real_trace_matches_the_golden_response_in_the_qwen3_coder_format_too() {
        let splitter = ReasoningSplitter::new(true, Some(248069))
            .with_format(ReasoningFormat::Qwen3Coder)
            .ending_at_tool_call(Some(248058));
        let ((reasoning, content), (deltas, streamed)) = both_ways(splitter, G8_REAL_TRACE);
        assert_eq!(reasoning.as_deref(), Some(G8_REAL_TRACE_REASONING));
        assert_eq!(content, G8_REAL_TRACE_CONTENT);
        assert_eq!(deltas.concat(), G8_REAL_TRACE_REASONING);
        assert_eq!(streamed, G8_REAL_TRACE_CONTENT);
    }

    #[test]
    fn an_id_without_text_right_after_the_close_still_reaches_the_caller() {
        let mut splitter = ReasoningSplitter::new(true, Some(CLOSE));
        assert_eq!(splitter.push(CLOSE, ""), ReasoningChunk::Boundary);
        assert_eq!(
            splitter.push(TOOL_CALL, ""),
            ReasoningChunk::Content(String::new()),
            "an empty piece neither consumes the newline budget nor disappears"
        );
        assert_eq!(
            splitter.push(1, "\n\nx"),
            ReasoningChunk::Content("x".to_string())
        );
    }

    #[test]
    fn streaming_and_non_streaming_agree_in_the_qwen3_coder_format() {
        let cases: Vec<Vec<(u32, &str)>> = vec![
            vec![(1, " a"), (CLOSE, ""), (2, "\n\n\n b")],
            vec![(1, "\n"), (2, "a"), (3, " "), (CLOSE, ""), (4, "b")],
            vec![(1, "  "), (CLOSE, "")],
            vec![(1, "  ")],
            vec![(CLOSE, ""), (1, " \n"), (2, "b")],
            vec![
                (OPEN, ""),
                (1, "\n"),
                (2, "x"),
                (TOOL_CALL, "<tool_call>"),
                (3, "y"),
            ],
        ];
        for tokens in cases {
            for (started, open) in [(true, None), (false, Some(OPEN))] {
                let splitter = qwen3_coder(started, open);
                let ((reasoning, content), (deltas, streamed)) = both_ways(splitter, &tokens);
                let concat = deltas.concat();
                assert_eq!(
                    reasoning,
                    (!concat.is_empty()).then(|| concat.clone()),
                    "{tokens:?} started_in_think={started}"
                );
                assert_eq!(content, streamed, "{tokens:?}");
            }
        }
    }

    // ── ReasoningChunk helpers ─────────────────────────────────────────────

    #[test]
    fn reasoning_chunk_as_str_and_predicates() {
        let r = ReasoningChunk::Reasoning("a".to_string());
        let c = ReasoningChunk::Content("b".to_string());
        let b = ReasoningChunk::Boundary;
        assert_eq!(r.as_str(), "a");
        assert!(r.is_reasoning());
        assert!(!r.is_content());
        assert_eq!(c.as_str(), "b");
        assert!(c.is_content());
        assert!(!c.is_reasoning());
        assert_eq!(b.as_str(), "");
        assert!(!b.is_reasoning());
        assert!(!b.is_content());
    }
}
