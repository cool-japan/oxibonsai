//! Generation helpers shared by `oxibonsai run` and `oxibonsai chat`:
//!
//! * [`ChatContract`] — the Bonsai 2 chat-contract options (`--think` /
//!   `--no-think`, `--reasoning-effort`, `--tools`; cli-11), rendered
//!   through the model's own chat template ([`render_prompt`]) so the
//!   prompt the model sees is byte-identical to the PrismML fork's own
//!   reference renderer. Tool definitions are kept as raw JSON **text** end
//!   to end — never a `serde_json::Value`, whose `BTreeMap` would re-sort
//!   the schema's keys.
//! * [`TokenPrinter`] — streams decoded tokens to the terminal, routing a
//!   model's `<think>` reasoning (split on the `</think>` token id, design
//!   §5.3) to stderr or dropping it (`--show-reasoning` /
//!   `--hide-reasoning`), and collecting the final `(reasoning, content)`
//!   pair for a chat history.
//! * [`cli_sampler`] — the CLI's own [`Sampler`], built with the same
//!   parameters/seed/penalties/`min_p` the engine's own decode path uses.
//!   `run`/`chat`'s ordinary sampled decoding goes through the engine
//!   itself now (`InferenceEngine::set_min_p`): only
//!   `run_constrained_or_stopped_with` (the `--grammar`/`--stop` path, which
//!   cannot run through the engine's own loop) still builds its plain
//!   sampler from this.

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
// The CLI's own sampler (grammar / `--stop` path only)
// ──────────────────────────────────────────────────────────────────────────
//
// Previously, a sampled request with `min_p > 0` also decoded through a
// dedicated CLI-owned loop (`decode_with_sampler`) built on this same
// sampler, because the engine had no `min_p` setter. `InferenceEngine::set_min_p`
// closed that gap: `run`/`chat` now call it once at engine
// construction (`cmd_run::load_engine`) and route every sampled request
// through the engine's own decode path uniformly
// (`cmd_run_tests::min_p_engine_path_matches_the_cli_loop` proves the two
// were token-for-token identical on both the tiny testkit fixture and the
// real 1.7B before the CLI loop was retired). [`cli_sampler`] itself stays:
// `run_constrained_or_stopped_with` (the `--grammar`/`--stop` path, which
// cannot run through the engine's own loop — a grammar mask or a
// stop-sequence check has no engine-side seam) still builds its plain
// (non-constrained) sampler from it.

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

#[cfg(test)]
#[path = "generate_tests.rs"]
mod tests;
