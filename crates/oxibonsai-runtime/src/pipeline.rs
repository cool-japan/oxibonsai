//! High-level inference pipeline API for OxiBonsai.
//!
//! The pipeline composes token healing, context management, sampling strategies,
//! beam search, and constrained decoding into a single fluent builder that
//! produces a configured [`InferencePipeline`] ready to run.
//!
//! ## Quick Start
//!
//! ```rust
//! use oxibonsai_runtime::pipeline::{PipelineBuilder, greedy_pipeline};
//! use oxibonsai_runtime::context_manager::TruncationStrategy;
//!
//! // Pre-built convenience preset. The trailing argument is an optional
//! // detokenizer (`Option<Arc<dyn Fn(&[u32]) -> RuntimeResult<String> + Send + Sync>>`);
//! // pass `None` to only ever consume `token_ids` from the output.
//! let pipeline = greedy_pipeline(32, None);
//! assert_eq!(pipeline.max_tokens(), 32);
//! assert!(!pipeline.has_healing());
//!
//! // Custom pipeline via builder. Attaching a detokenizer is what makes
//! // `PipelineOutput::text` and real-text stop-sequence matching work; a
//! // pipeline built without one only ever gets `token_ids` (`text` stays
//! // empty and `text_available` is `false` -- see [`PipelineOutput`]).
//! use oxibonsai_runtime::token_healing::TokenHealingConfig;
//! use oxibonsai_runtime::error::RuntimeResult;
//! let custom = PipelineBuilder::new()
//!     .max_tokens(128)
//!     .with_token_healing(TokenHealingConfig::default())
//!     .stop_on(vec!["<|end|>".to_string()])
//!     .with_detokenizer(|ids: &[u32]| -> RuntimeResult<String> {
//!         // A real caller would decode through its tokenizer here (e.g.
//!         // `TokenizerBridge::decode`); this toy version is just for the
//!         // doctest.
//!         Ok(ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" "))
//!     })
//!     .build();
//! assert!(custom.has_healing());
//! assert_eq!(custom.stop_sequences(), &["<|end|>"]);
//! ```

use std::sync::Arc;
use std::time::Instant;

use crate::beam_search::{BeamSearchConfig, BeamSearchEngine};
use crate::constrained_decoding::TokenConstraint;
use crate::context_manager::{ContextWindow, TruncationStrategy};
use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling_advanced::{LcgRng, SamplerChain, SamplerStep};
use crate::token_healing::{TokenHealer, TokenHealingConfig};

/// Signature shared by every detokenizer callback the pipeline accepts:
/// decode an arbitrary slice of token ids to text.
///
/// Kept as a type alias so `PipelineConfig`, [`PipelineBuilder`] and the
/// convenience constructors all name the exact same (rather lengthy) trait
/// object type.
pub type Detokenizer = dyn Fn(&[u32]) -> RuntimeResult<String> + Send + Sync;

// ─────────────────────────────────────────────────────────────────────────────
// GenerationStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// How the pipeline generates tokens at each step.
pub enum GenerationStrategy {
    /// Standard autoregressive sampling via a composable sampler chain.
    Sampling(SamplerChain),
    /// Beam search — deterministic search over the top-`beam_width` candidates.
    BeamSearch(BeamSearchConfig),
    /// Greedy decoding — always pick the highest-logit token.
    Greedy,
}

// ─────────────────────────────────────────────────────────────────────────────
// StopReason
// ─────────────────────────────────────────────────────────────────────────────

/// Why generation terminated.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// The `max_tokens` budget was exhausted.
    MaxTokens,
    /// A user-supplied stop sequence was encountered in the output.
    StopSequence(String),
    /// The model emitted an end-of-sequence token.
    EndOfSequence,
    /// The active [`TokenConstraint`] reported completion.
    ConstraintComplete,
    /// The active [`TokenConstraint`] eliminated every legal next token (an
    /// all-disallowed mask) before a single token could be generated for
    /// this step.
    ///
    /// Deliberately distinct from [`ConstraintComplete`](Self::ConstraintComplete):
    /// without this variant, a constraint that finishes normally (e.g. a
    /// satisfied JSON schema) and one that is simply unsatisfiable (e.g. a
    /// buggy grammar that masks every token) would both be reported as
    /// `ConstraintComplete` with zero generated tokens -- byte-identical and
    /// therefore silently indistinguishable to a caller.
    ConstraintUnsatisfiable,
    /// Generation aborted because the underlying engine returned an error.
    ///
    /// Only produced by the infallible [`InferencePipeline::run`] entry point;
    /// the accompanying string is the display form of the underlying
    /// [`crate::error::RuntimeError`]. The fallible
    /// [`InferencePipeline::try_run`] surfaces the same failure as `Err`.
    Error(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineOutput
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a complete pipeline run.
#[derive(Debug)]
pub struct PipelineOutput {
    /// Decoded text of the generated tokens.
    ///
    /// Populated by decoding [`token_ids`](Self::token_ids) through the
    /// [`Detokenizer`] attached via [`PipelineBuilder::with_detokenizer`].
    /// When no detokenizer is attached this is always the empty string --
    /// check [`text_available`](Self::text_available) rather than
    /// `text.is_empty()` alone, since an empty completion also has an empty
    /// `text` even with a detokenizer attached. This field never contains a
    /// space-separated dump of raw token-id decimals.
    pub text: String,
    /// Whether [`text`](Self::text) was actually decoded.
    ///
    /// `false` means either no detokenizer was attached, or one was
    /// attached but decoding failed after generation completed; the two
    /// cases are not distinguishable from this flag alone -- a caller that
    /// needs to tell them apart should use
    /// [`InferencePipeline::try_run`], whose `Err` surfaces a decode
    /// failure directly (the infallible [`InferencePipeline::run`] instead
    /// reports it as `text_available: false` alongside
    /// [`StopReason::Error`]).
    pub text_available: bool,
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens (after healing/context management).
    pub prompt_tokens: usize,
    /// Number of generated (completion) tokens.
    pub completion_tokens: usize,
    /// Reason generation ended.
    pub stop_reason: StopReason,
    /// Whether token healing was applied and changed the prompt.
    pub healing_applied: bool,
    /// Wall-clock time for the entire pipeline run in milliseconds.
    pub elapsed_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineConfig  (private)
// ─────────────────────────────────────────────────────────────────────────────

struct PipelineConfig {
    max_tokens: usize,
    strategy: GenerationStrategy,
    healing_config: Option<TokenHealingConfig>,
    constraint: Option<Box<dyn TokenConstraint>>,
    context_max_tokens: usize,
    truncation: TruncationStrategy,
    stop_sequences: Vec<String>,
    /// Decodes generated token ids to text for [`PipelineOutput::text`] and
    /// for real-text stop-sequence matching (see [`StopSequenceMatcher`]).
    /// `None` means the pipeline only ever produces `token_ids`.
    detokenizer: Option<Arc<Detokenizer>>,
    /// Stored for reproducibility and future use by strategies that need a
    /// standalone RNG (e.g. beam search with stochastic expansion).
    #[allow(dead_code)]
    seed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder that composes all inference options into an [`InferencePipeline`].
pub struct PipelineBuilder {
    max_tokens: usize,
    strategy: Option<GenerationStrategy>,
    healing_config: Option<TokenHealingConfig>,
    constraint: Option<Box<dyn TokenConstraint>>,
    context_max_tokens: usize,
    truncation: TruncationStrategy,
    stop_sequences: Vec<String>,
    detokenizer: Option<Arc<Detokenizer>>,
    seed: u64,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    /// Create a new builder with sensible defaults.
    ///
    /// Defaults:
    /// - `max_tokens` = 256
    /// - strategy = `Greedy`
    /// - no healing, no constraint
    /// - `context_max_tokens` = 2048, `TruncationStrategy::TruncateLeft`
    /// - no stop sequences
    /// - `seed` = 0
    pub fn new() -> Self {
        Self {
            max_tokens: 256,
            strategy: None,
            healing_config: None,
            constraint: None,
            context_max_tokens: 2048,
            truncation: TruncationStrategy::TruncateLeft,
            stop_sequences: Vec::new(),
            detokenizer: None,
            seed: 0,
        }
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Use greedy (argmax) decoding.
    pub fn greedy(mut self) -> Self {
        self.strategy = Some(GenerationStrategy::Greedy);
        self
    }

    /// Use a [`SamplerChain`] for token selection.
    pub fn with_sampling(mut self, chain: SamplerChain) -> Self {
        self.strategy = Some(GenerationStrategy::Sampling(chain));
        self
    }

    /// Use beam search with the supplied configuration.
    pub fn with_beam_search(mut self, config: BeamSearchConfig) -> Self {
        self.strategy = Some(GenerationStrategy::BeamSearch(config));
        self
    }

    /// Enable token healing with the supplied configuration.
    pub fn with_token_healing(mut self, config: TokenHealingConfig) -> Self {
        self.healing_config = Some(config);
        self
    }

    /// Attach a token constraint (e.g. JSON or regex).
    pub fn with_constraint(mut self, c: Box<dyn TokenConstraint>) -> Self {
        self.constraint = Some(c);
        self
    }

    /// Stop generation when any of the given string sequences appear in the output.
    ///
    /// Matching happens against real decoded text (see
    /// [`with_detokenizer`](Self::with_detokenizer)); without a detokenizer
    /// attached, configured stop sequences cannot be evaluated and are
    /// skipped (a warning is logged once, generation is unaffected -- see
    /// [`InferencePipeline::run`]'s module-level notes). The same leniency
    /// applies if an attached detokenizer's decode call errors while
    /// checking a stop sequence mid-run: that check is skipped (also warned
    /// once) rather than aborting the whole run, though a decode failure
    /// while producing the final [`PipelineOutput::text`] remains fatal --
    /// see [`InferencePipeline::run`]'s notes for the distinction.
    pub fn stop_on(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }

    /// Attach a detokenizer used to decode generated token ids to text.
    ///
    /// This is what makes [`PipelineOutput::text`] non-empty and lets
    /// [`stop_on`](Self::stop_on) match against real text instead of raw
    /// token ids. A typical caller wraps a [`crate::tokenizer_bridge::TokenizerBridge`]
    /// (or `oxibonsai_tokenizer::OxiTokenizer`) `decode` method:
    ///
    /// ```rust
    /// use oxibonsai_runtime::pipeline::PipelineBuilder;
    /// use oxibonsai_runtime::error::RuntimeResult;
    ///
    /// let pipeline = PipelineBuilder::new()
    ///     .with_detokenizer(|ids: &[u32]| -> RuntimeResult<String> {
    ///         Ok(format!("{ids:?}"))
    ///     })
    ///     .build();
    /// assert!(pipeline.has_detokenizer());
    /// ```
    pub fn with_detokenizer<F>(mut self, f: F) -> Self
    where
        F: Fn(&[u32]) -> RuntimeResult<String> + Send + Sync + 'static,
    {
        self.detokenizer = Some(Arc::new(f));
        self
    }

    /// Like [`with_detokenizer`](Self::with_detokenizer), but takes an
    /// already-shared `Arc<Detokenizer>` directly (or `None` to clear it).
    /// Used internally by the convenience constructors
    /// ([`chat_pipeline`], [`code_pipeline`], [`greedy_pipeline`]), which
    /// thread an optional detokenizer straight through without forcing
    /// every caller to build a fresh closure.
    pub fn with_detokenizer_opt(mut self, f: Option<Arc<Detokenizer>>) -> Self {
        self.detokenizer = f;
        self
    }

    /// Configure the context window size and truncation strategy.
    pub fn context_window(mut self, max_tokens: usize, strategy: TruncationStrategy) -> Self {
        self.context_max_tokens = max_tokens;
        self.truncation = strategy;
        self
    }

    /// Set the random seed used by sampling strategies.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Consume the builder and produce an [`InferencePipeline`].
    pub fn build(self) -> InferencePipeline {
        let strategy = self.strategy.unwrap_or(GenerationStrategy::Greedy);
        InferencePipeline {
            config: PipelineConfig {
                max_tokens: self.max_tokens,
                strategy,
                healing_config: self.healing_config,
                constraint: self.constraint,
                context_max_tokens: self.context_max_tokens,
                truncation: self.truncation,
                stop_sequences: self.stop_sequences,
                detokenizer: self.detokenizer,
                seed: self.seed,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferencePipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A fully configured inference pipeline.
///
/// Obtain one via [`PipelineBuilder`] or one of the convenience constructors
/// ([`chat_pipeline`], [`code_pipeline`], [`greedy_pipeline`]).
pub struct InferencePipeline {
    config: PipelineConfig,
}

impl InferencePipeline {
    /// Run the pipeline against the supplied engine.
    ///
    /// The pipeline:
    ///
    /// 1. Applies token healing to the prompt (if configured).
    /// 2. Trims the prompt to `context_max_tokens` using the configured truncation.
    /// 3. Generates tokens according to the selected [`GenerationStrategy`],
    ///    honouring any attached [`TokenConstraint`].
    /// 4. Stops at `max_tokens`, an EOS token, a stop sequence, or constraint
    ///    completion — whichever comes first.
    ///
    /// This is the infallible convenience entry point: a forward-pass, sampler,
    /// or constraint failure is surfaced through [`StopReason::Error`] on the
    /// returned [`PipelineOutput`] (with an empty completion) instead of
    /// panicking. Callers that need to react to the underlying
    /// [`crate::error::RuntimeError`] should use
    /// [`InferencePipeline::try_run`] instead.
    ///
    /// The engine API works with raw token IDs (no vocabulary metadata is
    /// available at this layer), so [`PipelineOutput::text`] is only
    /// populated when a [`Detokenizer`] is attached via
    /// [`PipelineBuilder::with_detokenizer`]; without one, `text` is empty
    /// and [`PipelineOutput::text_available`] is `false` (never a
    /// space-separated dump of token-id decimals -- see the field docs on
    /// [`PipelineOutput`]). The same applies to `stop_on`: a configured stop
    /// sequence can only be matched against real decoded text, so without a
    /// detokenizer attached it is evaluated as if no stop sequences were
    /// configured (a `tracing::warn!` is logged once per run so this is
    /// observable rather than silent). An *attached* detokenizer that
    /// returns `Err` while checking a stop sequence mid-run degrades the
    /// same way: that step's check is skipped (again warned once, not per
    /// step) and generation continues, rather than discarding every
    /// already-generated token over a single failed stop-sequence probe.
    /// This is distinct from the detokenizer failing on the *final* decode
    /// that produces [`PipelineOutput::text`] once generation has finished
    /// -- that failure is not swallowed and is still surfaced as
    /// [`StopReason::Error`] (`run`) or `Err` (`try_run`).
    pub fn run(
        &mut self,
        prompt_token_ids: Vec<u32>,
        engine: &mut InferenceEngine,
    ) -> PipelineOutput {
        let wall_start = Instant::now();
        match self.try_run_inner(prompt_token_ids, engine, wall_start) {
            Ok(output) => output,
            Err(e) => {
                tracing::error!(error = %e, "inference pipeline run failed");
                PipelineOutput {
                    text: String::new(),
                    text_available: false,
                    token_ids: Vec::new(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    stop_reason: StopReason::Error(e.to_string()),
                    healing_applied: false,
                    elapsed_ms: wall_start.elapsed().as_millis() as u64,
                }
            }
        }
    }

    /// Fallible twin of [`InferencePipeline::run`].
    ///
    /// Behaves identically but propagates any engine/forward-pass failure as
    /// `Err(RuntimeError)` instead of encoding it in the output's
    /// [`StopReason`]. Prefer this when the caller must distinguish a genuine
    /// generation failure from a normal early stop.
    pub fn try_run(
        &mut self,
        prompt_token_ids: Vec<u32>,
        engine: &mut InferenceEngine,
    ) -> RuntimeResult<PipelineOutput> {
        let wall_start = Instant::now();
        self.try_run_inner(prompt_token_ids, engine, wall_start)
    }

    /// Shared implementation behind [`run`](Self::run) and
    /// [`try_run`](Self::try_run).
    fn try_run_inner(
        &mut self,
        prompt_token_ids: Vec<u32>,
        engine: &mut InferenceEngine,
        wall_start: Instant,
    ) -> RuntimeResult<PipelineOutput> {
        // ── 1. Token healing ────────────────────────────────────────────────
        let (healed_prompt, healing_applied) = self.apply_healing(prompt_token_ids, engine)?;

        // ── 2. Context window management ────────────────────────────────────
        let mut window = ContextWindow::new(self.config.context_max_tokens, self.config.truncation);
        window.append(&healed_prompt);
        let context_tokens = window.tokens();
        let prompt_tokens = context_tokens.len();

        // ── 3. Generation ───────────────────────────────────────────────────
        // Clone the (small) beam config out of `self.config` up front so the
        // strategy-discriminant borrow is released before the decode loop takes
        // the mutable borrows it needs.
        let beam_cfg = if let GenerationStrategy::BeamSearch(cfg) = &self.config.strategy {
            Some(cfg.clone())
        } else {
            None
        };
        let (generated, stop_reason) = match beam_cfg {
            Some(cfg) => self.run_beam_search(&context_tokens, cfg, engine)?,
            None => self.run_autoregressive(&context_tokens, engine)?,
        };

        // ── 4. Build output ──────────────────────────────────────────────────
        // Decode through the attached detokenizer (RT-01): `text` must never
        // fall back to a token-id-decimal dump. Without a detokenizer there
        // is nothing to decode with, so `text` stays empty and callers are
        // told so via `text_available`.
        let (text, text_available) = match self.config.detokenizer.as_ref() {
            Some(detok) => (detok(&generated)?, true),
            None => (String::new(), false),
        };

        let elapsed_ms = wall_start.elapsed().as_millis() as u64;

        Ok(PipelineOutput {
            text,
            text_available,
            completion_tokens: generated.len(),
            token_ids: generated,
            prompt_tokens,
            stop_reason,
            healing_applied,
            elapsed_ms,
        })
    }

    /// Apply token healing to the prompt, driving the real model for the
    /// prefix re-score when a [`TokenHealingConfig`] is attached.
    ///
    /// Returns the (possibly healed) prompt and whether healing changed it.
    /// Any forward-pass error raised while scoring the prefix is propagated.
    fn apply_healing(
        &self,
        prompt_token_ids: Vec<u32>,
        engine: &mut InferenceEngine,
    ) -> RuntimeResult<(Vec<u32>, bool)> {
        let healing_cfg = match &self.config.healing_config {
            Some(cfg) => cfg.clone(),
            None => return Ok((prompt_token_ids, false)),
        };

        let healer = TokenHealer::new(healing_cfg);
        let vocab_size = engine.vocab_size();

        // The healer calls back once with the prefix; feed it real model
        // logits. A forward-pass error is captured and surfaced after the
        // (closure-bounded) mutable borrow of `engine` is released, since the
        // callback signature cannot itself return a `Result`.
        let mut heal_err: Option<RuntimeError> = None;
        let result = {
            let err_slot = &mut heal_err;
            healer.heal(&prompt_token_ids, vocab_size, |prefix| {
                if err_slot.is_some() || prefix.is_empty() {
                    return Vec::new();
                }
                engine.reset();
                match engine.prefill_from_pos(prefix, 0) {
                    Ok(logits) => logits,
                    Err(e) => {
                        *err_slot = Some(e);
                        Vec::new()
                    }
                }
            })
        };
        if let Some(e) = heal_err {
            return Err(e);
        }

        Ok((result.healed_tokens, result.changed))
    }

    /// Autoregressive generation (greedy or via a configured [`SamplerChain`]),
    /// honouring any attached [`TokenConstraint`], the EOS token, and stop
    /// sequences.
    ///
    /// Drives the engine's public prefill/decode primitives directly (rather
    /// than delegating to [`InferenceEngine::generate`]) so the pipeline's own
    /// sampler chain and constraint actually determine each token.
    fn run_autoregressive(
        &mut self,
        context_tokens: &[u32],
        engine: &mut InferenceEngine,
    ) -> RuntimeResult<(Vec<u32>, StopReason)> {
        let vocab_size = engine.vocab_size();
        let max = self.config.max_tokens;
        let context_len = context_tokens.len();

        if context_tokens.is_empty() || max == 0 {
            return Ok((Vec::new(), StopReason::MaxTokens));
        }

        // Disjoint field borrows so the sampler chain and constraint can be
        // driven statefully across the whole decode loop.
        let strategy = &mut self.config.strategy;
        let constraint = &mut self.config.constraint;
        let stop_matcher = StopSequenceMatcher::new(&self.config.stop_sequences);
        let detokenizer = self.config.detokenizer.clone();
        if !stop_matcher.is_empty() && detokenizer.is_none() {
            // RT-02: stop sequences can only be matched against real decoded
            // text. Without a detokenizer there is nothing to decode with --
            // rather than resurrect the old token-id-decimal match (which
            // could never fire correctly) or hard-fail a request that may
            // not care about `text` at all, skip stop-sequence matching and
            // say so once, loudly.
            tracing::warn!(
                "InferencePipeline: stop_on(...) is configured but no detokenizer is \
                 attached (PipelineBuilder::with_detokenizer); stop sequences cannot be \
                 matched against token ids and will not fire for this run"
            );
        }

        // Start from a clean KV cache (healing may have populated it), then
        // prefill the full (healed) prompt.
        engine.reset();
        let mut logits = engine.prefill_from_pos(context_tokens, 0)?;

        let mut generated: Vec<u32> = Vec::with_capacity(max);
        let mut stop_reason = StopReason::MaxTokens;
        // Warn-once flag for a detokenizer that errors mid-run while
        // checking a stop sequence (see the loop body below): logged at
        // most once per run so a persistently-broken detokenizer does not
        // spam a warning on every remaining step.
        let mut detok_error_warned = false;

        for step in 0..max {
            // `SV-09`: cooperative cancellation, once per decode step -- this
            // loop drives the engine's primitives directly rather than going
            // through `InferenceEngine::generate`/`generate_streaming`, so it
            // did not inherit their cancellation check and an armed token was
            // silently ignored for the whole pipeline path.
            if engine.is_cancelled() {
                tracing::debug!(
                    step,
                    "InferencePipeline: autoregressive generation cancelled"
                );
                break;
            }

            // ── Constraint mask (RT-20) ───────────────────────────────────
            // Excluded tokens get an exact `f32::NEG_INFINITY` rather than a
            // large-but-finite sentinel (which could leak nonzero
            // probability at extreme temperatures), applied once to a
            // scratch copy so the model's raw logits are never mutated in
            // place -- only allocated when a constraint is actually
            // attached, so the unconstrained hot path pays nothing extra.
            let next: u32 = if let Some(c) = constraint.as_ref() {
                match c.allowed_tokens(&generated, vocab_size) {
                    // Bound the "is anything allowed" scan to what `logits`
                    // can actually reflect, exactly like
                    // `apply_constraint_mask` (below) is already bounded by
                    // `logits.get_mut(i)`: a mask longer than `logits.len()`
                    // whose only allowed index sits past the end could
                    // otherwise report "satisfiable" here while every
                    // in-range (reachable) logit still gets masked to
                    // `-inf` by `apply_constraint_mask` -- the exact
                    // NaN-driven path this guard exists to prevent.
                    Some(mask)
                        if !mask[..logits.len().min(mask.len())]
                            .iter()
                            .any(|&allowed| allowed) =>
                    {
                        // The constraint has eliminated every possible next
                        // token. Feeding an all-`NEG_INFINITY` vector into
                        // the sampler would hit exactly the NaN-driven
                        // arbitrary-token path RT-21 describes for
                        // `sampling.rs`'s softmax (and the equivalent
                        // latent path in `sampling_advanced.rs`'s
                        // `softmax_inplace` -> `categorical_sample`, since
                        // `(-inf) - (-inf) == NaN`) -- switching from the
                        // old finite `-1e9` sentinel to an exact `-inf`
                        // makes that reachable where it silently wasn't
                        // before.
                        // Stop here, deterministically and honestly,
                        // instead of ever constructing that input --
                        // reported as the distinct `ConstraintUnsatisfiable`
                        // (never `ConstraintComplete`) so a constraint that
                        // is simply impossible to satisfy is never confused
                        // with one that finished normally; see that
                        // variant's doc comment.
                        stop_reason = StopReason::ConstraintUnsatisfiable;
                        break;
                    }
                    Some(mask) => {
                        let mut masked = logits.clone();
                        apply_constraint_mask(&mut masked, &mask);
                        select_next(strategy, &mut masked)
                    }
                    None => select_next(strategy, &mut logits),
                }
            } else {
                select_next(strategy, &mut logits)
            };

            // ── EOS ──────────────────────────────────────────────────────
            // `RT-18`: check membership in the engine's whole resolved EOS
            // *set* (`tokenizer.ggml.eos_token_id` plus any additional
            // terminator ids such as `eot_token_id` or `<|endoftext|>`)
            // rather than comparing only against the single primary
            // `eos_token_id()`, which a model with more than one terminator
            // (Bonsai 2's chat contract stops on both `<|im_end|>` and
            // `<|endoftext|>`) would otherwise miss.
            if engine.is_eos(next) {
                stop_reason = StopReason::EndOfSequence;
                break;
            }

            // ── Stop-sequence check (RT-02); the triggering token is
            // excluded on a match ─────────────────────────────────────────
            if !stop_matcher.is_empty() {
                if let Some(detok) = detokenizer.as_ref() {
                    let mut candidate = generated.clone();
                    candidate.push(next);
                    match detok(&candidate) {
                        Ok(text) => {
                            if let StopMatch::Found { sequence, start } = stop_matcher.check(&text)
                            {
                                let keep = StopSequenceMatcher::keep_len(
                                    &candidate,
                                    start,
                                    detok.as_ref(),
                                )?;
                                generated.truncate(keep);
                                stop_reason = StopReason::StopSequence(sequence);
                                break;
                            }
                        }
                        Err(e) => {
                            // A decode failure here is scoped to *this
                            // step's* stop-sequence check, not to the run as
                            // a whole -- propagating it via `?` would (per
                            // `InferencePipeline::run`) discard every
                            // already-generated token merely because the
                            // stop-sequence probe on the latest candidate
                            // failed, which is far more destructive than
                            // simply not being able to check for a stop
                            // match this step. Degrade the same way the
                            // missing-detokenizer case above does: warn once
                            // and keep generating as if this step's check
                            // never fired. The *final* `text` decode in
                            // `try_run_inner` still propagates a persistent
                            // failure normally.
                            if !detok_error_warned {
                                tracing::warn!(
                                    error = %e,
                                    "InferencePipeline: detokenizer returned an error while \
                                     checking a stop sequence; skipping the stop-sequence check \
                                     for the rest of this run instead of discarding \
                                     already-generated tokens"
                                );
                                detok_error_warned = true;
                            }
                        }
                    }
                }
            }

            // ── Commit the token ─────────────────────────────────────────
            generated.push(next);

            // ── Advance the constraint; stop on completion or violation ──
            if let Some(c) = constraint.as_mut() {
                let still_valid = c.advance(next);
                if c.is_complete() || !still_valid {
                    stop_reason = StopReason::ConstraintComplete;
                    break;
                }
            }

            // ── Feed the token back for the next step's logits ───────────
            if generated.len() < max {
                logits = engine.decode_step(next, context_len + step)?;
            }
        }

        Ok((generated, stop_reason))
    }

    /// Beam-search generation.
    ///
    /// Supplies the beam engine a real `get_logits` closure so the configured
    /// [`BeamSearchConfig`] actually explores candidates instead of stalling
    /// on an empty logit vector. Also honours any attached [`TokenConstraint`]
    /// (masking each beam before expansion, exactly like `run_autoregressive`
    /// does) and inherits the engine's GGUF-resolved EOS id unless the caller
    /// explicitly overrode [`BeamSearchConfig::eos_token_id`].
    ///
    /// ## Cost (RT-33)
    ///
    /// All live beams at a given step share one [`InferenceEngine`] (and
    /// therefore one KV cache), evaluated one at a time. Rather than
    /// resetting and re-prefilling each beam's **entire** history from
    /// position 0 on every call (`O(beams × steps × prompt_len)` forward
    /// passes -- for `B` beams, `T` steps and prompt length `L`,
    /// `Σ_{t=0..T} B·(L+t) ≈ B·T·(L+T/2)`, e.g. ~544K token-forwards instead
    /// of ~1.5K for `B=4, T=128, L=1000`), the cache is rewound
    /// ([`InferenceEngine::rewind_cache`]) to the longest common prefix with
    /// whichever beam was evaluated immediately before it, and only the
    /// divergent suffix is re-prefilled. Since every live beam within a
    /// round has the same depth (all rounds extend every surviving beam by
    /// exactly one token), that prefix is usually long, so most calls cost a
    /// single-token forward. Worst case (beams fully diverge from token 0
    /// every call) this is `O(L + B·T·avg_divergence)`; it can pathologically
    /// approach the old bound if beams share nothing, but never exceeds it,
    /// and it is exact -- not an approximation -- because
    /// [`InferenceEngine::prefill_from_pos`] always returns logits reflecting
    /// the beam's complete token history. A caller that must guarantee the
    /// tighter `O(L + B·T)` bound regardless of divergence needs `B`
    /// independent KV caches (e.g. `paged_kv_cache.rs`'s per-sequence ids),
    /// which is out of scope here -- cap `beam_width` for latency-sensitive
    /// callers in the meantime.
    fn run_beam_search(
        &mut self,
        context_tokens: &[u32],
        beam_cfg: BeamSearchConfig,
        engine: &mut InferenceEngine,
    ) -> RuntimeResult<(Vec<u32>, StopReason)> {
        let vocab_size = engine.vocab_size();

        // Inherit the engine's real (GGUF-resolved) EOS id when the caller
        // left `eos_token_id` at its default sentinel; an explicit override
        // (e.g. a test pinning a synthetic EOS id) is always respected.
        let beam_cfg = beam_cfg.inherit_eos_if_default(engine.eos_token_id());

        let beam_engine = BeamSearchEngine::new(beam_cfg);

        // Prime the cache exactly once; every beam below shares it, rewound
        // to its common prefix with whatever the cache currently holds (see
        // the cost note above -- RT-33).
        engine.reset();

        // The beam engine calls `get_logits` for each live beam at each step.
        // A forward-pass error is captured and surfaced after the search
        // returns (the closure signature cannot return a `Result`).
        let mut search_err: Option<RuntimeError> = None;
        // Tokens currently reflected in the shared KV cache. Kept in lock
        // step with the cache: every successful `prefill_from_pos` below
        // updates it to match, so `common_prefix_len(&cached_tokens, ..)`
        // never overstates how much of the next beam is already cached.
        let mut cached_tokens: Vec<u32> = Vec::new();
        let result = {
            let err_slot = &mut search_err;
            let constraint = self.config.constraint.as_deref_mut();
            beam_engine.search_with_constraint(
                context_tokens.to_vec(),
                vocab_size,
                |beam_tokens, _step| {
                    // Once any call has failed, every later call must be a
                    // pure no-op: `cached_tokens` is no longer trustworthy
                    // (the failed `prefill_from_pos` below may have left the
                    // cache at neither its old nor its intended new length),
                    // and touching `rewind_cache`/`prefill_from_pos` again
                    // would build on that unknown state. This guard running
                    // first is what keeps that safe -- do not reorder it
                    // after the cache operations.
                    if err_slot.is_some() || beam_tokens.is_empty() {
                        return Vec::new();
                    }
                    // Rewind to the longest prefix `beam_tokens` still shares
                    // with what is currently cached, but always keep at
                    // least one token pending so a beam identical to the
                    // previous one still gets freshly computed logits
                    // instead of a stale/absent forward pass.
                    let mut common =
                        common_prefix_len(&cached_tokens, beam_tokens).min(beam_tokens.len() - 1);
                    if engine.recurrent_rollback_supported() {
                        engine.rewind_cache(common);
                    } else if common != cached_tokens.len() {
                        // A hybrid (recurrent) engine cannot move
                        // a cursor back -- its Gated-DeltaNet state after the
                        // cached tokens does not determine the state after the
                        // shorter common prefix. Unless the beam purely
                        // extends what is cached, replay it from scratch:
                        // exact (the "reset and replay" rollback), just not
                        // KV-reusing.
                        engine.reset();
                        common = 0;
                    }
                    match engine.prefill_from_pos(&beam_tokens[common..], common) {
                        Ok(logits) => {
                            cached_tokens = beam_tokens.to_vec();
                            logits
                        }
                        Err(e) => {
                            *err_slot = Some(e);
                            Vec::new()
                        }
                    }
                },
                constraint,
            )
        };
        if let Some(e) = search_err {
            return Err(e);
        }

        let best = result.best().to_vec();
        // Strip the prompt prefix from the beam result.
        let generated = if best.len() > context_tokens.len() {
            best[context_tokens.len()..].to_vec()
        } else {
            Vec::new()
        };

        // If a constraint is attached, replay it over the winning sequence
        // once more. This both (a) leaves the pipeline's live constraint
        // state consistent with the emitted output -- matching the
        // autoregressive path's contract -- and (b) tells us whether
        // completion was genuinely due to the constraint (rather than simply
        // running out of beam-search budget), so the reported `StopReason`
        // is honest instead of always `MaxTokens`/`EndOfSequence`.
        let constraint_completed = match self.config.constraint.as_deref_mut() {
            Some(c) if !generated.is_empty() => {
                c.reset();
                let mut still_valid = true;
                for &t in &generated {
                    if !c.advance(t) {
                        still_valid = false;
                        break;
                    }
                }
                still_valid && c.is_complete()
            }
            _ => false,
        };

        let (final_tokens, stop_reason) = self.check_stop_sequences(generated)?;
        let stop_reason = if constraint_completed
            && matches!(
                stop_reason,
                StopReason::EndOfSequence | StopReason::MaxTokens
            ) {
            StopReason::ConstraintComplete
        } else {
            stop_reason
        };

        Ok((final_tokens, stop_reason))
    }

    /// Post-hoc pass over a completed (e.g. beam-search) token sequence:
    /// find the first configured stop sequence in its real decoded text
    /// (RT-02) and truncate to just before it. Returns the tokens to keep
    /// plus the stop reason.
    ///
    /// Shares [`StopSequenceMatcher`] with `run_autoregressive`'s inline
    /// check rather than duplicating the match-then-map-to-token-index
    /// logic; the two differ only in that this one decodes the whole
    /// sequence in a single call instead of incrementally.
    fn check_stop_sequences(&self, tokens: Vec<u32>) -> RuntimeResult<(Vec<u32>, StopReason)> {
        let matcher = StopSequenceMatcher::new(&self.config.stop_sequences);
        let fallback_reason = |tokens: &[u32]| {
            if tokens.len() >= self.config.max_tokens {
                StopReason::MaxTokens
            } else {
                StopReason::EndOfSequence
            }
        };

        if matcher.is_empty() || tokens.is_empty() {
            let stop = fallback_reason(&tokens);
            return Ok((tokens, stop));
        }

        let Some(detok) = self.config.detokenizer.as_ref() else {
            // See `run_autoregressive`'s identical rationale: without a
            // detokenizer, stop sequences cannot be evaluated against real
            // text, so behave as if none were configured instead of
            // resurrecting a token-id-decimal scan.
            tracing::warn!(
                "InferencePipeline: stop_on(...) is configured but no detokenizer is \
                 attached; stop sequences cannot be matched for this beam-search run"
            );
            let stop = fallback_reason(&tokens);
            return Ok((tokens, stop));
        };

        let text = detok(&tokens)?;
        if let StopMatch::Found { sequence, start } = matcher.check(&text) {
            let keep = StopSequenceMatcher::keep_len(&tokens, start, detok.as_ref())?;
            return Ok((tokens[..keep].to_vec(), StopReason::StopSequence(sequence)));
        }

        let stop = fallback_reason(&tokens);
        Ok((tokens, stop))
    }

    /// Maximum number of tokens this pipeline will generate.
    pub fn max_tokens(&self) -> usize {
        self.config.max_tokens
    }

    /// Returns `true` if token healing is configured.
    pub fn has_healing(&self) -> bool {
        self.config.healing_config.is_some()
    }

    /// Returns `true` if a token constraint is attached.
    pub fn has_constraint(&self) -> bool {
        self.config.constraint.is_some()
    }

    /// Returns `true` if a [`Detokenizer`] is attached (see
    /// [`PipelineBuilder::with_detokenizer`]) -- i.e. whether this pipeline's
    /// output will ever populate [`PipelineOutput::text`] or be able to
    /// match configured stop sequences against real text.
    pub fn has_detokenizer(&self) -> bool {
        self.config.detokenizer.is_some()
    }

    /// The list of stop sequences that will halt generation early.
    pub fn stop_sequences(&self) -> &[String] {
        &self.config.stop_sequences
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Build a standard chat pipeline.
///
/// Settings:
/// - Temperature = 0.7, top-p = 0.9, min-p = 0.05
/// - Context window = 4096 tokens (TruncateLeft)
/// - No healing, no constraint
///
/// `detokenizer` is threaded straight to
/// [`PipelineBuilder::with_detokenizer_opt`]; pass `None` if the caller only
/// consumes `token_ids` (then [`PipelineOutput::text`] stays empty -- see
/// [`PipelineOutput::text_available`]).
pub fn chat_pipeline(
    seed: u64,
    max_tokens: usize,
    detokenizer: Option<Arc<Detokenizer>>,
) -> InferencePipeline {
    let chain = SamplerChain::new(seed)
        .add(SamplerStep::Temperature(0.7))
        .add(SamplerStep::TopP(0.9))
        .add(SamplerStep::MinP(0.05));

    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .with_sampling(chain)
        .context_window(4096, TruncationStrategy::TruncateLeft)
        .with_detokenizer_opt(detokenizer)
        .seed(seed)
        .build()
}

/// Build a code-generation pipeline.
///
/// Settings:
/// - Temperature = 0.2, top-k = 40
/// - Token healing enabled (default config)
/// - Stop on `"\n\n"` (blank line) -- only takes effect when `detokenizer`
///   is `Some`; see [`PipelineBuilder::stop_on`].
///
/// `detokenizer` is threaded straight to
/// [`PipelineBuilder::with_detokenizer_opt`].
pub fn code_pipeline(
    seed: u64,
    max_tokens: usize,
    detokenizer: Option<Arc<Detokenizer>>,
) -> InferencePipeline {
    let chain = SamplerChain::new(seed)
        .add(SamplerStep::Temperature(0.2))
        .add(SamplerStep::TopK(40));

    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .with_sampling(chain)
        .with_token_healing(TokenHealingConfig::default())
        .stop_on(vec!["\n\n".to_string()])
        .with_detokenizer_opt(detokenizer)
        .seed(seed)
        .build()
}

/// Build a greedy (deterministic) pipeline.
///
/// `detokenizer` is threaded straight to
/// [`PipelineBuilder::with_detokenizer_opt`].
pub fn greedy_pipeline(
    max_tokens: usize,
    detokenizer: Option<Arc<Detokenizer>>,
) -> InferencePipeline {
    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .greedy()
        .with_detokenizer_opt(detokenizer)
        .build()
}

// ─────────────────────────────────────────────────────────────────────────────
// StopSequenceMatcher (RT-02 / RT-06)
// ─────────────────────────────────────────────────────────────────────────────

/// Detects configured stop sequences inside text decoded from generated
/// tokens.
///
/// Used by both of [`InferencePipeline`]'s generation paths
/// (`run_autoregressive`'s inline per-step check and
/// `check_stop_sequences`'s post-hoc pass over a completed beam-search
/// result) so they cannot drift into two different notions of "stopped".
/// Kept `pub(crate)` rather than private so `api_extensions`'s streaming SSE
/// path (RT-06 -- a hold-back buffer in front of a live token stream, which
/// has the same "did a stop sequence appear, and where" question this type
/// already answers) can reuse it instead of growing a third implementation.
///
/// Matches against real decoded text obtained from a caller-supplied
/// [`Detokenizer`] rather than a proxy such as token-id decimals (RT-01's
/// defect, and the reason RT-02's stop-sequence matching never fired: see
/// the module's former behaviour). Because matching always runs against a
/// freshly decoded, complete token sequence rather than an incrementally
/// appended buffer, a stop sequence spanning multiple tokens (e.g. `"<|end"`
/// then `"|>"` from a BPE tokenizer) is found correctly regardless of where
/// the token boundaries happen to fall -- there is no risk of a chunk-
/// boundary leak in this (non-streaming) use, since nothing is ever emitted
/// to a consumer before the whole run finishes.
pub(crate) struct StopSequenceMatcher {
    /// Configured stop sequences, with empty strings dropped (an empty
    /// sequence would trivially "match" every position, which is never a
    /// caller's intent).
    sequences: Vec<String>,
    /// Longest configured sequence, in bytes; `0` when `sequences` is empty.
    max_stop_len: usize,
}

/// Outcome of [`StopSequenceMatcher::check`].
pub(crate) enum StopMatch {
    /// No configured stop sequence appears in the checked text.
    None,
    /// `sequence` was found starting at byte offset `start` of the checked
    /// text (the earliest match across all configured sequences).
    Found { sequence: String, start: usize },
}

impl StopSequenceMatcher {
    /// Build a matcher for the given stop sequences.
    pub(crate) fn new(sequences: &[String]) -> Self {
        let sequences: Vec<String> = sequences
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        let max_stop_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            sequences,
            max_stop_len,
        }
    }

    /// `true` when there are no (non-empty) configured stop sequences, i.e.
    /// [`check`](Self::check) can never report a match.
    pub(crate) fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// The longest configured stop sequence, in bytes. Not consumed by
    /// production code directly — the streaming caller (RT-06) only ever
    /// needs [`hold_back_len`](Self::hold_back_len), which applies this
    /// value's `- 1` cap internally — so this accessor exists purely for
    /// tests to assert on the value directly rather than reaching for the
    /// private field. Narrowed (like `hold_back_len`'s own `allow`) to only
    /// where it is actually dead: a non-test build has no caller at all, a
    /// test build does (this module's own tests), so the lint still runs
    /// for real there.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn max_stop_len(&self) -> usize {
        self.max_stop_len
    }

    /// Scan already-decoded `text` for the earliest occurrence of any
    /// configured stop sequence.
    pub(crate) fn check(&self, text: &str) -> StopMatch {
        self.sequences
            .iter()
            .filter_map(|seq| text.find(seq.as_str()).map(|start| (seq, start)))
            .min_by_key(|&(_, start)| start)
            .map(|(seq, start)| StopMatch::Found {
                sequence: seq.clone(),
                start,
            })
            .unwrap_or(StopMatch::None)
    }

    /// Longest suffix of `text` that could still be the unfinished prefix of
    /// a configured stop sequence -- i.e. it might grow into a full match
    /// once more text arrives. A streaming caller must hold back exactly
    /// this many trailing bytes before flushing anything to its consumer;
    /// the cut point `text.len() - hold_back_len(text)` always lands on a
    /// UTF-8 char boundary; per RT-06's corrected fix, slicing there instead
    /// of at a raw byte offset is what avoids panicking mid-codepoint on
    /// multi-byte (e.g. CJK) output.
    ///
    /// `pipeline.rs`'s own matching never needs this: it always holds the
    /// complete decoded text and only ever reports a result once the whole
    /// run has finished, so nothing is "flushed" early. It is provided
    /// here, on the shared matcher, for the streaming SSE path (RT-06),
    /// which does need it and would otherwise have to reimplement this
    /// exact suffix-matching logic against a second copy of `sequences`.
    ///
    /// Consumed by `api_extensions::extended_chat_completions_stream` (RT-06).
    /// That module is `#[cfg(feature = "server")]` while this one is not, so
    /// a `--no-default-features` build (this crate explicitly supports one,
    /// see `hf-tokenizer`'s Cargo.toml docs) has no caller at all; the
    /// `allow` only applies there; a default (`server`-enabled) build still
    /// gets the dead-code check on this method for free.
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    pub(crate) fn hold_back_len(&self, text: &str) -> usize {
        if self.sequences.is_empty() || text.is_empty() {
            return 0;
        }
        let cap = self.max_stop_len.saturating_sub(1).min(text.len());
        for len in (1..=cap).rev() {
            let start = text.len() - len;
            if !text.is_char_boundary(start) {
                continue;
            }
            let suffix = &text[start..];
            if self.sequences.iter().any(|seq| seq.starts_with(suffix)) {
                return len;
            }
        }
        0
    }

    /// Determine how many leading tokens of `tokens` to keep so that their
    /// decoded text ends at or before byte offset `match_start` -- i.e.
    /// drop every token whose decoded contribution starts at or after a
    /// stop-sequence match found at that offset.
    ///
    /// Binary searches over the candidate prefix length rather than
    /// scanning it linearly, so `detok` is called `O(log tokens.len())`
    /// times instead of up to `tokens.len()` times. This relies on the same
    /// monotonicity assumption a linear scan would also need -- decoding a
    /// longer token-id prefix never yields *shorter* text, which holds for
    /// any real (streaming/append-only) tokenizer decoder. That cost is
    /// only paid once, on the single step a stop sequence is actually found
    /// (which ends generation), never on the per-token hot path.
    pub(crate) fn keep_len(
        tokens: &[u32],
        match_start: usize,
        detok: &Detokenizer,
    ) -> RuntimeResult<usize> {
        // Largest `k` in `0..=tokens.len()` with `detok(&tokens[..k]).len()
        // <= match_start`. `lo = 0` is always a valid starting answer:
        // decoding zero tokens yields `""`, and `0 <= match_start` always
        // holds for a byte offset.
        let mut lo = 0usize;
        let mut hi = tokens.len();
        while lo < hi {
            // `mid` sits strictly in `(lo, hi]`, so each branch below
            // strictly narrows the range and the loop always terminates.
            let mid = lo + (hi - lo).div_ceil(2);
            let cumulative_len = detok(&tokens[..mid])?.len();
            if cumulative_len <= match_start {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Ok(lo)
    }
}

/// Length of the longest common prefix shared by two token sequences.
///
/// Used by `run_beam_search` (RT-33) to find how much of a beam's history is
/// already reflected in the shared KV cache from evaluating the previous
/// beam, so only the divergent suffix needs to be re-prefilled.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Force every disallowed logit to `f32::NEG_INFINITY` -- an exact
/// exclusion that holds at any temperature, unlike a large-but-finite
/// sentinel such as `-1e9` (RT-20), which can retain nonzero post-softmax
/// probability at extreme logit ranges / high temperature.
fn apply_constraint_mask(logits: &mut [f32], mask: &[bool]) {
    for (i, &allowed) in mask.iter().enumerate() {
        if !allowed {
            if let Some(logit) = logits.get_mut(i) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

/// Select the next token per the configured [`GenerationStrategy`], from
/// the scratch copy when a constraint is attached, otherwise the live
/// buffer coming straight from the engine -- see `apply_constraint_mask`'s
/// caller, which only clones into a scratch copy when a constraint is
/// actually attached. Either way, `logits` may be mutated in place (e.g.
/// `SamplerChain::sample`'s in-place softmax), so callers must not assume
/// it is unchanged afterwards.
fn select_next(strategy: &mut GenerationStrategy, logits: &mut Vec<f32>) -> u32 {
    match strategy {
        GenerationStrategy::Sampling(chain) => chain.sample(logits) as u32,
        // Greedy (and the unreachable beam arm) fall back to argmax.
        _ => argmax_logits(logits),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: unused but part of internal plumbing
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy argmax over a logit slice.
///
/// First-wins tie-break and NaN-robust (RT-22): the previous
/// `.max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))` returns the LAST of
/// several equal maxima and can let a NaN entry win depending on fold
/// direction. This matches `sampling.rs::argmax`'s already-fixed
/// convention (and `engine.rs`'s hand-rolled `generate_greedy_gpu` loop),
/// so all three greedy paths in this crate now agree on exact ties --
/// otherwise the 27B plan's temp-0 byte-parity gate against the PrismML
/// fork would surface a tie-break mismatch as a phantom kernel bug.
fn argmax_logits(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Build a greedy sampler chain (single Greedy step).
#[allow(dead_code)]
fn greedy_chain(seed: u64) -> SamplerChain {
    SamplerChain::new(seed).add(SamplerStep::Greedy)
}

/// LCG-based sampler: temperature + weighted draw, no external deps.
#[allow(dead_code)]
fn sample_from_logits(logits: &[f32], temperature: f32, rng: &mut LcgRng) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature < 1e-6 {
        return argmax_logits(logits);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|&v| ((v - max) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return 0;
    }
    let target = rng.next_f32() * sum;
    let mut cum = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        cum += e;
        if cum >= target {
            return i as u32;
        }
    }
    (exps.len() - 1) as u32
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
