//! `<think>` reasoning-content splitting (B2-13, bonsai2-design.md §5.3).
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
//! # Models without `<think>` (RT-10 correction)
//!
//! A GGUF whose vocabulary has no `<think>`/`</think>` tokens at all (the
//! shipped Qwen3 1.7B/8B models) must not be pushed into "reasoning" mode
//! just because the *caller* asked for thinking — there is no way for such
//! a model to ever close it. [`ReasoningSplitter::new`] takes `close_id:
//! Option<u32>`: `None` (the ids did not resolve against the loaded
//! tokenizer) puts the splitter into permanent pass-through, where every
//! [`ReasoningSplitter::push`] call returns [`ReasoningChunk::Content`] and
//! [`ReasoningSplitter::in_reasoning`] is always `false`. Resolve the ids
//! once per loaded model (e.g. via the vocabulary's `<think>`/`</think>`
//! piece lookup) and gate on **both** that resolution *and* the
//! request/template's `enable_thinking` before ever constructing a splitter
//! that starts `started_in_think = true`.
//!
//! # The post-boundary gap (G8)
//!
//! `scratchpad/golden2/chat.prompt1.server.json` (the reference server's own
//! `/v1/chat/completions` response) shows `reasoning_content` keeping its
//! trailing `"...Final: 4.\n"` newline untouched, while `content` is the
//! bare `"4"` — the `\n\n` the model emits immediately after `</think>`
//! (matching the template's own historical assistant re-rendering
//! convention, `'\n</think>\n\n' + content`) never reaches either channel.
//! [`ReasoningSplitter`] reproduces this by swallowing a leading run of
//! `'\n'` characters **once**, immediately after the close boundary, and
//! nowhere else — a newline appearing later, mid-answer, is real content.

// ── Chunk classification ────────────────────────────────────────────────────

/// Which channel one [`ReasoningSplitter::push`] call's piece belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningChunk {
    /// Reasoning-channel text (accumulates into `reasoning_content`).
    Reasoning(String),
    /// Answer-channel text (accumulates into `content`).
    Content(String),
    /// A structural boundary: either the `</think>` token itself, or a
    /// leading newline immediately following it that neither channel shows
    /// (see the module doc's "post-boundary gap" section). Contributes to
    /// neither channel and is safe to drop.
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

// ── Splitter ─────────────────────────────────────────────────────────────────

/// Internal phase of a [`ReasoningSplitter`]. Not `pub`: the only observable
/// states are "in reasoning" ([`ReasoningSplitter::in_reasoning`]) and
/// "not" — `JustClosed` is a one-token transient the caller never needs to
/// distinguish from `Content`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Emitting into the reasoning channel.
    Reasoning,
    /// The close token was just consumed; the next piece(s) may still be
    /// swallowed by the post-boundary newline skip (see the module doc),
    /// up to the fixed budget carried here (see [`POST_CLOSE_NEWLINE_BUDGET`]).
    /// Once a piece exhausts the remaining budget, or the budget is already
    /// `0`, every further piece — newlines included — is real content.
    JustClosed(u8),
    /// Emitting into the content channel.
    Content,
}

/// How many leading newline characters immediately after `</think>` the
/// post-boundary gap swallows, **in total**, however many separate decode
/// tokens they are split across.
///
/// Matches the reference fork's own `optspace(end)` bound rather than an
/// unbounded fixpoint strip: the template's historical assistant
/// re-rendering convention is exactly `'\n</think>\n\n' + content` — two
/// newlines — so anything beyond that is real model output, not template
/// boilerplate, even if it happens to also be a bare newline.
const POST_CLOSE_NEWLINE_BUDGET: u8 = 2;

/// Splits a raw, per-token generation stream into `reasoning_content` and
/// `content`. See the module docs for the id-keyed design and the
/// pass-through contract for models without `<think>`.
#[derive(Debug, Clone)]
pub struct ReasoningSplitter {
    /// `Some(id)` only when both `<think>`/`</think>` resolved against the
    /// loaded model's vocabulary; `None` is permanent pass-through.
    close_id: Option<u32>,
    phase: Phase,
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
    pub fn new(started_in_think: bool, close_id: Option<u32>) -> Self {
        let phase = if close_id.is_some() && started_in_think {
            Phase::Reasoning
        } else {
            Phase::Content
        };
        Self { close_id, phase }
    }

    /// Reconstruct the current phase's `JustClosed` budget as `Phase::Content`
    /// once it hits zero — shared by every `push` arm that discovers the
    /// budget is already exhausted, so the "budget spent" transition happens
    /// in exactly one place.
    fn just_closed_budget_used_up(&mut self, piece: &str) -> ReasoningChunk {
        self.phase = Phase::Content;
        ReasoningChunk::Content(piece.to_string())
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
        self.phase == Phase::Reasoning
    }

    /// Feed one generated token (its id and already-decoded piece text) and
    /// classify it.
    ///
    /// `piece` must be the correctly decoded text for `id` alone (e.g. via
    /// [`oxibonsai_tokenizer::StreamingDecoder`], which handles a
    /// multi-byte UTF-8 character split across token boundaries) — this
    /// function does not itself decode anything.
    pub fn push(&mut self, id: u32, piece: &str) -> ReasoningChunk {
        match self.phase {
            Phase::Reasoning => {
                if self.close_id == Some(id) {
                    self.phase = Phase::JustClosed(0);
                    return ReasoningChunk::Boundary;
                }
                ReasoningChunk::Reasoning(piece.to_string())
            }
            Phase::JustClosed(swallowed) => {
                let budget = POST_CLOSE_NEWLINE_BUDGET.saturating_sub(swallowed);
                if budget == 0 {
                    // The fixed post-boundary budget is already spent by
                    // earlier tokens; this piece — even if it is itself
                    // more bare newlines — is real content from here on.
                    return self.just_closed_budget_used_up(piece);
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
            Phase::Content => ReasoningChunk::Content(piece.to_string()),
        }
    }
}

/// Split an already-decoded `(id, piece)` sequence into `(reasoning_content,
/// content)` in one call — the shape a non-streaming response handler wants.
///
/// `reasoning_content` is `None` when nothing was ever classified as
/// reasoning (pass-through mode, or a splitter that never actually entered
/// the reasoning phase), matching the OpenAI response contract of omitting
/// the field rather than sending an empty string.
pub fn split_reasoning<'a, I>(
    tokens: I,
    started_in_think: bool,
    close_id: Option<u32>,
) -> (Option<String>, String)
where
    I: IntoIterator<Item = (u32, &'a str)>,
{
    let mut splitter = ReasoningSplitter::new(started_in_think, close_id);
    let mut reasoning = String::new();
    let mut content = String::new();
    for (id, piece) in tokens {
        match splitter.push(id, piece) {
            ReasoningChunk::Reasoning(s) => reasoning.push_str(&s),
            ReasoningChunk::Content(s) => content.push_str(&s),
            ReasoningChunk::Boundary => {}
        }
    }
    // Whitespace-only reasoning is treated the same as none at all —
    // matching the reference fork's parser, which discards it rather than
    // reporting an OpenAI `reasoning_content` field that is present but
    // carries nothing a client could ever show.
    let reasoning = if reasoning.trim().is_empty() {
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

    // ── G8: reproduces `golden2/chat.prompt1.server.json` structurally ────
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
    // below (post-verifier-review).
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

    /// The real golden trace behind G8 (`scratchpad/golden2/chat.prompt1.server.json`,
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
    /// `step_decode` returned `None` — the post-verifier-review fix
    /// elsewhere in this crate feeds exactly this shape to `push`) is a
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
