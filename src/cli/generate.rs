//! Generation helpers shared by `oxibonsai run` and `oxibonsai chat`:
//!
//! * [`ChatContract`] — the Bonsai 2 chat-contract options (`--think` /
//!   `--no-think`, `--reasoning-effort`, `--tools`; cli-11), rendered
//!   through the model's own chat template ([`render_prompt`]) so the
//!   prompt the model sees is byte-identical to the reference renderer's
//!   (`golden2/apply_template.json`, B2-13's G7 contract). Tool definitions
//!   are kept as raw JSON **text** end to end — never a `serde_json::Value`,
//!   whose `BTreeMap` would re-sort the schema's keys.
//! * [`TokenPrinter`] — streams decoded tokens to the terminal, routing a
//!   model's `<think>` reasoning (split on the `</think>` token id, design
//!   §5.3) to stderr or dropping it (`--show-reasoning` /
//!   `--hide-reasoning`), and collecting the final `(reasoning, content)`
//!   pair for a chat history.
//! * [`decode_with_sampler`] — a token-by-token decode loop driven by the
//!   CLI's own [`Sampler`]. `InferenceEngine` exposes no `min_p` setter, so a
//!   sampled request with `--min-p > 0` decodes through this loop, which
//!   applies temperature → top-k → min-p → top-p and the penalty history
//!   with the very same [`Sampler`] math the engine's own loop uses.

use std::io::{self, Write};

use oxibonsai_runtime::config::{RenderMessage, RenderOptions, ResolvedChatTemplate};
use oxibonsai_runtime::reasoning::{ReasoningChunk, ReasoningSplitter};
use oxibonsai_runtime::sampling::{PenaltyParams, Sampler, SamplingParams};
use oxibonsai_runtime::tokenizer_bridge::DecodeStreamState;
use oxibonsai_runtime::TokenizerBridge;

// ──────────────────────────────────────────────────────────────────────────
// Chat contract (cli-11)
// ──────────────────────────────────────────────────────────────────────────

/// The chat-contract options one command resolved (flags, then `--config`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ChatContract {
    /// `--think` → `Some(true)`, `--no-think` → `Some(false)`, neither →
    /// `None` (the template's own undefined branch — part of its byte-exact
    /// contract, so it is passed through as undefined, never defaulted).
    pub(crate) enable_thinking: Option<bool>,
    /// `--reasoning-effort low|medium|xhigh`.
    pub(crate) reasoning_effort: Option<String>,
    /// The `--tools` file's raw JSON text (validated as a JSON array, kept
    /// byte-for-byte).
    pub(crate) tools_json: Option<String>,
}

impl ChatContract {
    /// Resolve the contract, loading `--tools` from disk.
    ///
    /// # Errors
    ///
    /// An unreadable `--tools` file, or one that is not a JSON array.
    pub(crate) fn from_flags(
        enable_thinking: Option<bool>,
        reasoning_effort: Option<String>,
        tools_path: Option<&str>,
    ) -> anyhow::Result<Self> {
        let tools_json = tools_path.map(load_tools_file).transpose()?;
        Ok(Self {
            enable_thinking,
            reasoning_effort,
            tools_json,
        })
    }

    /// `true` when no chat-contract option was given at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.enable_thinking.is_none()
            && self.reasoning_effort.is_none()
            && self.tools_json.is_none()
    }

    /// The template [`RenderOptions`] this contract means.
    pub(crate) fn render_options(&self, add_generation_prompt: bool) -> RenderOptions {
        RenderOptions {
            add_generation_prompt,
            enable_thinking: self.enable_thinking,
            reasoning_effort: self.reasoning_effort.clone(),
            tools: self.tools_json.clone(),
            ..RenderOptions::default()
        }
    }
}

/// Read a `--tools` file as raw text, checking that it parses as a JSON
/// array. The parse result is discarded — the text itself is what the
/// template consumes, so the operator's key order and number formatting
/// survive untouched.
///
/// # Errors
///
/// An unreadable file, invalid JSON, or JSON that is not an array.
pub(crate) fn load_tools_file(path: &str) -> anyhow::Result<String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read --tools file '{path}': {e}"))?;
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("--tools file '{path}' is not valid JSON: {e}"))?;
    if !parsed.is_array() {
        anyhow::bail!(
            "--tools file '{path}' must contain a JSON array of OpenAI-style tool definitions \
             (e.g. [{{\"type\": \"function\", \"function\": {{...}}}}])"
        );
    }
    Ok(text.trim_end().to_string())
}

/// Render `messages` through `template` under `contract`, with the
/// assistant generation prompt appended.
///
/// # Errors
///
/// The template's own render error (e.g. an unsupported reasoning effort,
/// or malformed tools JSON).
pub(crate) fn render_prompt(
    template: &ResolvedChatTemplate,
    messages: &[RenderMessage],
    contract: &ChatContract,
) -> anyhow::Result<String> {
    template
        .render_with(messages, &contract.render_options(true))
        .map_err(|e| anyhow::anyhow!("failed to render the prompt through the chat template: {e}"))
}

/// `--show-reasoning` / `--hide-reasoning`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ReasoningDisplay {
    /// Print reasoning to stderr while the answer streams to stdout.
    #[default]
    Show,
    /// Drop the reasoning; print only the answer.
    Hide,
}

impl ReasoningDisplay {
    /// Resolve the two flags (`clap` already makes them mutually
    /// exclusive). Returns the display and whether it was set explicitly.
    pub(crate) fn from_flags(show: bool, hide: bool) -> (Self, bool) {
        if hide {
            (Self::Hide, true)
        } else {
            (Self::Show, show)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Token printing with reasoning split (design §5.3)
// ──────────────────────────────────────────────────────────────────────────

/// Streams generated tokens to the terminal.
pub(crate) enum TokenPrinter<'t> {
    /// Decode with the tokenizer; reasoning is split off on the
    /// `</think>` token id.
    Text(Box<TextPrinter<'t>>),
    /// No tokenizer: print raw token ids.
    Raw {
        /// Tokens printed so far.
        count: usize,
    },
}

/// The tokenizer-backed half of [`TokenPrinter`].
pub(crate) struct TextPrinter<'t> {
    tok: &'t TokenizerBridge,
    state: DecodeStreamState,
    splitter: ReasoningSplitter,
    display: ReasoningDisplay,
    reasoning: String,
    content: String,
    /// Stream reasoning/content to the terminal as tokens arrive (`false`
    /// for the buffered constrained/stop loop, which prints once at the end).
    live: bool,
}

impl<'t> TokenPrinter<'t> {
    /// A printer for one generation. `started_in_think` is whether the
    /// rendered prompt left the model inside an open `<think>` block (always
    /// `false` for a raw, un-templated prompt).
    pub(crate) fn new(
        tok: Option<&'t TokenizerBridge>,
        started_in_think: bool,
        display: ReasoningDisplay,
        live: bool,
    ) -> Self {
        match tok {
            Some(tok) => {
                let splitter = if started_in_think {
                    ReasoningSplitter::new(true, tok.think_close_id())
                } else {
                    ReasoningSplitter::pass_through()
                };
                Self::Text(Box::new(TextPrinter {
                    tok,
                    state: tok.new_decode_stream(true),
                    splitter,
                    display,
                    reasoning: String::new(),
                    content: String::new(),
                    live,
                }))
            }
            None => Self::Raw { count: 0 },
        }
    }

    /// Feed one generated token.
    ///
    /// # Errors
    ///
    /// A tokenizer decode error.
    pub(crate) fn push(&mut self, token_id: u32) -> anyhow::Result<()> {
        match self {
            Self::Text(printer) => printer.push(token_id),
            Self::Raw { count } => {
                *count += 1;
                if *count == 1 {
                    print!("Tokens:");
                }
                print!(" {token_id}");
                let _ = io::stdout().flush();
                Ok(())
            }
        }
    }

    /// The text generated so far (reasoning + content, in order) — what
    /// `--stop` matches against.
    pub(crate) fn full_text(&self) -> String {
        match self {
            Self::Text(printer) => format!("{}{}", printer.reasoning, printer.content),
            Self::Raw { .. } => String::new(),
        }
    }

    /// Finish the generation: `(reasoning, content)`. For a buffered
    /// printer, `stop_truncate` first cuts the content at a matched stop
    /// sequence and both channels are printed now.
    pub(crate) fn finish(
        self,
        stop_truncate: Option<&dyn Fn(&str) -> String>,
    ) -> (Option<String>, String) {
        match self {
            Self::Text(printer) => {
                let TextPrinter {
                    reasoning,
                    content,
                    display,
                    live,
                    ..
                } = *printer;
                let content = match stop_truncate {
                    Some(truncate) => truncate(&content),
                    None => content,
                };
                if !live {
                    if display == ReasoningDisplay::Show && !reasoning.is_empty() {
                        eprintln!("{reasoning}");
                    }
                    print!("{content}");
                    let _ = io::stdout().flush();
                }
                // Whitespace-only reasoning counts as none — exactly what the
                // server's `split_reasoning` reports, so a chat history
                // re-renders the same turn the server would.
                let reasoning = (!reasoning.trim().is_empty()).then_some(reasoning);
                (reasoning, content)
            }
            Self::Raw { .. } => (None, String::new()),
        }
    }
}

impl TextPrinter<'_> {
    fn push(&mut self, token_id: u32) -> anyhow::Result<()> {
        let piece = self
            .tok
            .step_decode(&mut self.state, token_id)?
            .unwrap_or_default();
        match self.splitter.push(token_id, &piece) {
            ReasoningChunk::Reasoning(text) => {
                if self.live && self.display == ReasoningDisplay::Show {
                    eprint!("{text}");
                    let _ = io::stderr().flush();
                }
                self.reasoning.push_str(&text);
            }
            ReasoningChunk::Content(text) => {
                if self.live {
                    print!("{text}");
                    let _ = io::stdout().flush();
                }
                self.content.push_str(&text);
            }
            ReasoningChunk::Boundary => {
                if self.live
                    && self.display == ReasoningDisplay::Show
                    && !self.reasoning.is_empty()
                    && !self.reasoning.ends_with('\n')
                {
                    eprintln!();
                }
            }
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────
// CLI-owned sampled decode loop (min-p)
// ──────────────────────────────────────────────────────────────────────────

/// Build the CLI's own [`Sampler`] for a request: the same parameters and
/// seed the engine was built with, plus the penalties and `min_p`.
pub(crate) fn cli_sampler(
    params: SamplingParams,
    seed: u64,
    penalties: PenaltyParams,
    min_p: f32,
) -> Sampler {
    let mut sampler = Sampler::new(params, seed);
    sampler.set_penalties(penalties);
    sampler.set_min_p(min_p);
    sampler
}

/// Whether a request must decode through [`decode_with_sampler`] rather than
/// the engine's own loop: only a sampled request (`temperature > 0`) with
/// `min_p > 0` needs it — greedy decoding ignores min-p, and `min_p == 0`
/// is exactly what the engine's own sampler already does.
pub(crate) fn needs_cli_sampler(temperature: f32, min_p: f32) -> bool {
    temperature > 0.0 && min_p > 0.0
}

/// Prefill `prompt_tokens` from position 0, then decode up to `max_tokens`
/// tokens with `sampler` (history-aware, so repetition/frequency/presence
/// penalties apply), stopping at the engine's EOS set or when `on_token`
/// returns `false`. Mirrors `InferenceEngine::generate_streaming_sync`'s own
/// loop order: sample → EOS check → emit → record history → forward.
///
/// Returns the number of tokens emitted.
///
/// # Errors
///
/// Engine prefill/forward errors, sampler errors, or `on_token`'s error.
pub(crate) fn decode_with_sampler(
    engine: &mut oxibonsai_runtime::InferenceEngine<'_>,
    prompt_tokens: &[u32],
    max_tokens: usize,
    sampler: &mut Sampler,
    mut on_token: impl FnMut(u32) -> anyhow::Result<bool>,
) -> anyhow::Result<usize> {
    if prompt_tokens.is_empty() || max_tokens == 0 {
        return Ok(0);
    }
    engine.reset();
    let mut logits = engine.prefill_from_pos(prompt_tokens, 0)?;
    let mut history: Vec<u32> = Vec::new();
    let mut generated = 0usize;
    for (pos, _) in (prompt_tokens.len()..).zip(0..max_tokens) {
        let token = sampler.sample_with_history(&logits, &history)?;
        if engine.is_eos(token) {
            break;
        }
        generated += 1;
        let keep_going = on_token(token)?;
        history.push(token);
        if !keep_going {
            break;
        }
        logits = engine.decode_step(token, pos)?;
    }
    Ok(generated)
}

#[cfg(test)]
#[path = "generate_tests.rs"]
mod tests;
