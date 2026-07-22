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
//! // Pre-built convenience preset
//! let pipeline = greedy_pipeline(32);
//! assert_eq!(pipeline.max_tokens(), 32);
//! assert!(!pipeline.has_healing());
//!
//! // Custom pipeline via builder
//! use oxibonsai_runtime::token_healing::TokenHealingConfig;
//! let custom = PipelineBuilder::new()
//!     .max_tokens(128)
//!     .with_token_healing(TokenHealingConfig::default())
//!     .stop_on(vec!["<|end|>".to_string()])
//!     .build();
//! assert!(custom.has_healing());
//! assert_eq!(custom.stop_sequences(), &["<|end|>"]);
//! ```

use std::time::Instant;

use crate::beam_search::{BeamSearchConfig, BeamSearchEngine};
use crate::constrained_decoding::TokenConstraint;
use crate::context_manager::{ContextWindow, TruncationStrategy};
use crate::engine::InferenceEngine;
use crate::error::{RuntimeError, RuntimeResult};
use crate::sampling_advanced::{LcgRng, SamplerChain, SamplerStep};
use crate::token_healing::{TokenHealer, TokenHealingConfig};

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
    /// In the absence of a real tokenizer the token IDs are serialised as
    /// space-separated decimal strings.
    pub text: String,
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
    pub fn stop_on(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
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
    /// Because the engine API works with raw token IDs (no vocabulary metadata is
    /// available at this layer), the `text` field of the returned [`PipelineOutput`]
    /// encodes token IDs as space-separated decimal strings.
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
        let text: String = generated
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let elapsed_ms = wall_start.elapsed().as_millis() as u64;

        Ok(PipelineOutput {
            text,
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
        let vocab_size = engine.model().config().vocab_size;

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
        let vocab_size = engine.model().config().vocab_size;
        let max = self.config.max_tokens;
        let context_len = context_tokens.len();

        if context_tokens.is_empty() || max == 0 {
            return Ok((Vec::new(), StopReason::MaxTokens));
        }

        // Disjoint field borrows so the sampler chain and constraint can be
        // driven statefully across the whole decode loop.
        let strategy = &mut self.config.strategy;
        let constraint = &mut self.config.constraint;
        let stop_sequences = &self.config.stop_sequences;

        // Start from a clean KV cache (healing may have populated it), then
        // prefill the full (healed) prompt.
        engine.reset();
        let mut logits = engine.prefill_from_pos(context_tokens, 0)?;

        let mut generated: Vec<u32> = Vec::with_capacity(max);
        let mut committed_text = String::new();
        let mut stop_reason = StopReason::MaxTokens;

        for step in 0..max {
            // ── Constraint mask ──────────────────────────────────────────
            if let Some(c) = constraint.as_ref() {
                if let Some(mask) = c.allowed_tokens(&generated, vocab_size) {
                    for (i, &allowed) in mask.iter().enumerate() {
                        if !allowed && i < logits.len() {
                            logits[i] = -1e9;
                        }
                    }
                }
            }

            // ── Select the next token per the configured strategy ────────
            let next: u32 = match strategy {
                GenerationStrategy::Sampling(chain) => chain.sample(&mut logits) as u32,
                // Greedy (and the unreachable beam arm) fall back to argmax.
                _ => argmax_logits(&logits),
            };

            // ── EOS ──────────────────────────────────────────────────────
            // Use the engine's GGUF-resolved EOS id (falls back to the
            // engine-wide default when the loaded model has no explicit
            // `tokenizer.ggml.eos_token_id`) rather than the hardcoded
            // constant, so autoregressive decoding recognizes the real
            // model's end-of-sequence token.
            if next == engine.eos_token_id() {
                stop_reason = StopReason::EndOfSequence;
                break;
            }

            // ── Stop-sequence check (the triggering token is excluded) ───
            if !stop_sequences.is_empty() {
                let mut candidate = committed_text.clone();
                candidate.push_str(&next.to_string());
                candidate.push(' ');
                if let Some(seq) = stop_sequences
                    .iter()
                    .find(|s| candidate.contains(s.as_str()))
                {
                    stop_reason = StopReason::StopSequence(seq.clone());
                    break;
                }
                committed_text = candidate;
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
    /// Supplies the beam engine a real `get_logits` closure that re-runs prefill
    /// over each beam's full token sequence from a reset KV cache, so the
    /// configured [`BeamSearchConfig`] actually explores candidates instead of
    /// stalling on an empty logit vector. Also honours any attached
    /// [`TokenConstraint`] (masking each beam before expansion, exactly like
    /// `run_autoregressive` does) and inherits the engine's GGUF-resolved
    /// EOS id unless the caller explicitly overrode
    /// [`BeamSearchConfig::eos_token_id`].
    fn run_beam_search(
        &mut self,
        context_tokens: &[u32],
        beam_cfg: BeamSearchConfig,
        engine: &mut InferenceEngine,
    ) -> RuntimeResult<(Vec<u32>, StopReason)> {
        let vocab_size = engine.model().config().vocab_size;

        // Inherit the engine's real (GGUF-resolved) EOS id when the caller
        // left `eos_token_id` at its default sentinel; an explicit override
        // (e.g. a test pinning a synthetic EOS id) is always respected.
        let beam_cfg = beam_cfg.inherit_eos_if_default(engine.eos_token_id());

        let beam_engine = BeamSearchEngine::new(beam_cfg);

        // The beam engine calls `get_logits` for each live beam at each step.
        // Each call re-runs prefill over the beam's full token sequence from a
        // reset cache. A forward-pass error is captured and surfaced after the
        // search returns (the closure signature cannot return a `Result`).
        let mut search_err: Option<RuntimeError> = None;
        let result = {
            let err_slot = &mut search_err;
            let constraint = self.config.constraint.as_deref_mut();
            beam_engine.search_with_constraint(
                context_tokens.to_vec(),
                vocab_size,
                |beam_tokens, _step| {
                    if err_slot.is_some() || beam_tokens.is_empty() {
                        return Vec::new();
                    }
                    engine.reset();
                    match engine.prefill_from_pos(beam_tokens, 0) {
                        Ok(logits) => logits,
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

        let (final_tokens, stop_reason) = self.check_stop_sequences(generated);
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

    /// Walk `tokens`, checking whether any stop sequence appears in the partial
    /// decoded text.  Returns the tokens up to (but not including) the stop
    /// sequence, plus the stop reason.
    fn check_stop_sequences(&self, tokens: Vec<u32>) -> (Vec<u32>, StopReason) {
        if self.config.stop_sequences.is_empty() {
            let stop = if tokens.len() >= self.config.max_tokens {
                StopReason::MaxTokens
            } else {
                StopReason::EndOfSequence
            };
            return (tokens, stop);
        }

        // Build the text token-by-token and scan for stop sequences.
        let mut text_so_far = String::new();
        for (i, &tok) in tokens.iter().enumerate() {
            text_so_far.push_str(&tok.to_string());
            text_so_far.push(' ');

            for seq in &self.config.stop_sequences {
                if text_so_far.contains(seq.as_str()) {
                    return (tokens[..i].to_vec(), StopReason::StopSequence(seq.clone()));
                }
            }
        }

        let stop = if tokens.len() >= self.config.max_tokens {
            StopReason::MaxTokens
        } else {
            StopReason::EndOfSequence
        };
        (tokens, stop)
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
pub fn chat_pipeline(seed: u64, max_tokens: usize) -> InferencePipeline {
    let chain = SamplerChain::new(seed)
        .add(SamplerStep::Temperature(0.7))
        .add(SamplerStep::TopP(0.9))
        .add(SamplerStep::MinP(0.05));

    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .with_sampling(chain)
        .context_window(4096, TruncationStrategy::TruncateLeft)
        .seed(seed)
        .build()
}

/// Build a code-generation pipeline.
///
/// Settings:
/// - Temperature = 0.2, top-k = 40
/// - Token healing enabled (default config)
/// - Stop on `"\n\n"` (blank line)
pub fn code_pipeline(seed: u64, max_tokens: usize) -> InferencePipeline {
    let chain = SamplerChain::new(seed)
        .add(SamplerStep::Temperature(0.2))
        .add(SamplerStep::TopK(40));

    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .with_sampling(chain)
        .with_token_healing(TokenHealingConfig::default())
        .stop_on(vec!["\n\n".to_string()])
        .seed(seed)
        .build()
}

/// Build a greedy (deterministic) pipeline.
pub fn greedy_pipeline(max_tokens: usize) -> InferencePipeline {
    PipelineBuilder::new()
        .max_tokens(max_tokens)
        .greedy()
        .build()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: unused but part of internal plumbing
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy argmax over a logit slice.
fn argmax_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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
mod tests {
    use super::*;
    use crate::sampling::SamplingParams;

    // ── Builder tests ────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_builder_default() {
        let pipeline = PipelineBuilder::new().build();
        assert_eq!(pipeline.max_tokens(), 256);
        assert!(!pipeline.has_healing());
        assert!(!pipeline.has_constraint());
        assert!(pipeline.stop_sequences().is_empty());
    }

    #[test]
    fn test_pipeline_builder_max_tokens() {
        let pipeline = PipelineBuilder::new().max_tokens(512).build();
        assert_eq!(pipeline.max_tokens(), 512);
    }

    #[test]
    fn test_pipeline_builder_greedy() {
        let pipeline = PipelineBuilder::new().greedy().build();
        assert!(matches!(
            pipeline.config.strategy,
            GenerationStrategy::Greedy
        ));
    }

    #[test]
    fn test_pipeline_builder_stop_sequences() {
        let stops = vec!["<|end|>".to_string(), "STOP".to_string()];
        let pipeline = PipelineBuilder::new().stop_on(stops.clone()).build();
        assert_eq!(pipeline.stop_sequences(), stops.as_slice());
    }

    #[test]
    fn test_pipeline_builder_with_healing() {
        let cfg = TokenHealingConfig {
            lookback: 2,
            min_prob: 0.1,
            enabled: true,
        };
        let pipeline = PipelineBuilder::new().with_token_healing(cfg).build();
        assert!(pipeline.has_healing());
    }

    // ── Output / StopReason tests ────────────────────────────────────────────

    #[test]
    fn test_pipeline_output_stop_reason() {
        let output = PipelineOutput {
            text: "hello".to_string(),
            token_ids: vec![1, 2, 3],
            prompt_tokens: 5,
            completion_tokens: 3,
            stop_reason: StopReason::StopSequence("STOP".to_string()),
            healing_applied: false,
            elapsed_ms: 10,
        };
        assert_eq!(
            output.stop_reason,
            StopReason::StopSequence("STOP".to_string())
        );
        assert_eq!(output.completion_tokens, 3);
        assert_eq!(output.prompt_tokens, 5);
    }

    // ── Preset tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_chat_pipeline_preset() {
        let pipeline = chat_pipeline(42, 256);
        assert_eq!(pipeline.max_tokens(), 256);
        assert!(!pipeline.has_healing());
        assert!(pipeline.stop_sequences().is_empty());
        // Context window should be 4096
        assert_eq!(pipeline.config.context_max_tokens, 4096);
    }

    #[test]
    fn test_code_pipeline_preset() {
        let pipeline = code_pipeline(0, 128);
        assert_eq!(pipeline.max_tokens(), 128);
        assert!(pipeline.has_healing());
        assert_eq!(pipeline.stop_sequences(), &["\n\n"]);
    }

    #[test]
    fn test_greedy_pipeline_preset() {
        let pipeline = greedy_pipeline(64);
        assert_eq!(pipeline.max_tokens(), 64);
        assert!(!pipeline.has_healing());
        assert!(!pipeline.has_constraint());
        assert!(matches!(
            pipeline.config.strategy,
            GenerationStrategy::Greedy
        ));
    }

    // ── Full run test ────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_run_basic() {
        use oxibonsai_core::config::Qwen3Config;

        let config = Qwen3Config::tiny_test();
        let mut engine = InferenceEngine::new(
            config,
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            42,
        );

        let mut pipeline = PipelineBuilder::new().max_tokens(5).greedy().build();

        let output = pipeline.run(vec![151644u32, 872], &mut engine);
        // We care that the pipeline runs without panic and produces a result.
        assert_eq!(output.prompt_tokens, 2);
        assert!(output.elapsed_ms < 60_000, "should finish in under 60s");
    }

    // ── Strategy / constraint / healing wiring tests ─────────────────────────
    //
    // The `tiny_test` model has no transformer blocks and zero weights, so every
    // forward pass yields an all-zero logit vector on any backend. That makes
    // greedy argmax pick the last vocab index deterministically and lets these
    // tests assert exact behaviour without a real checkpoint.

    /// Build a deterministic greedy engine over the tiny test config.
    fn greedy_engine() -> InferenceEngine<'static> {
        use oxibonsai_core::config::Qwen3Config;
        InferenceEngine::new(
            Qwen3Config::tiny_test(),
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            42,
        )
    }

    /// Test constraint that allows exactly one token id at every step.
    struct OnlyToken(u32);
    impl TokenConstraint for OnlyToken {
        fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
            let mut mask = vec![false; vocab_size];
            let idx = self.0 as usize;
            if idx < vocab_size {
                mask[idx] = true;
            }
            Some(mask)
        }
        fn advance(&mut self, _token: u32) -> bool {
            true
        }
        fn is_complete(&self) -> bool {
            false
        }
        fn reset(&mut self) {}
        fn name(&self) -> &str {
            "OnlyToken"
        }
    }

    /// Test constraint that reports completion after `n` committed tokens.
    struct CompleteAfterN {
        n: usize,
        count: usize,
    }
    impl TokenConstraint for CompleteAfterN {
        fn allowed_tokens(&self, _generated: &[u32], _vocab_size: usize) -> Option<Vec<bool>> {
            None
        }
        fn advance(&mut self, _token: u32) -> bool {
            self.count += 1;
            true
        }
        fn is_complete(&self) -> bool {
            self.count >= self.n
        }
        fn reset(&mut self) {
            self.count = 0;
        }
        fn name(&self) -> &str {
            "CompleteAfterN"
        }
    }

    #[test]
    fn test_pipeline_run_greedy_deterministic() {
        let prompt = vec![151644u32, 872];

        let mut e1 = greedy_engine();
        let out1 = PipelineBuilder::new()
            .max_tokens(5)
            .greedy()
            .build()
            .run(prompt.clone(), &mut e1);

        let mut e2 = greedy_engine();
        let out2 = PipelineBuilder::new()
            .max_tokens(5)
            .greedy()
            .build()
            .run(prompt.clone(), &mut e2);

        assert_eq!(out1.token_ids.len(), 5);
        assert_eq!(out1.stop_reason, StopReason::MaxTokens);
        assert_eq!(
            out1.token_ids, out2.token_ids,
            "greedy decoding must be deterministic"
        );
        // Identical logits every step → greedy repeats the same token.
        assert!(out1.token_ids.iter().all(|&t| t == out1.token_ids[0]));
    }

    #[test]
    fn test_pipeline_run_sampling_honors_chain() {
        use oxibonsai_core::config::Qwen3Config;
        let vocab = Qwen3Config::tiny_test().vocab_size;
        let prompt = vec![151644u32, 872];

        let mut ge = greedy_engine();
        let out_greedy = PipelineBuilder::new()
            .max_tokens(5)
            .greedy()
            .build()
            .run(prompt.clone(), &mut ge);

        // A stochastic chain must drive selection itself. If the pipeline
        // ignored it (the pre-fix bug), the output would match pure argmax.
        let chain = SamplerChain::default_chat(7);
        let mut se = greedy_engine();
        let out_sampling = PipelineBuilder::new()
            .max_tokens(5)
            .with_sampling(chain)
            .build()
            .run(prompt.clone(), &mut se);

        assert_ne!(
            out_sampling.token_ids, out_greedy.token_ids,
            "the configured SamplerChain must be honored, not argmax/the engine sampler"
        );
        assert!(out_sampling.token_ids.iter().all(|&t| (t as usize) < vocab));
    }

    #[test]
    fn test_pipeline_run_beam_search_generates() {
        let cfg = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 3,
            eos_token_id: 999_999,
            early_stopping: false,
            ..Default::default()
        };
        let mut engine = greedy_engine();
        let output = PipelineBuilder::new()
            .with_beam_search(cfg)
            .build()
            .run(vec![151644u32, 872], &mut engine);

        // Regression: the beam path used to stall on empty logits and return
        // nothing; it must now actually explore and produce tokens.
        assert!(
            !output.token_ids.is_empty(),
            "beam search must produce tokens"
        );
        assert_eq!(output.token_ids.len(), 3);
    }

    #[test]
    fn test_pipeline_run_beam_search_honors_constraint() {
        // Regression (runtime-engine-01): InferencePipeline::run_beam_search
        // used to silently ignore any attached TokenConstraint -- combining
        // `.with_constraint(...)` with `.with_beam_search(...)` produced
        // fully unconstrained output. The constraint must now mask every
        // beam's candidates before top-k expansion, exactly like the
        // autoregressive path already does.
        let cfg = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 4,
            eos_token_id: 999_999,
            early_stopping: false,
            ..Default::default()
        };
        let mut engine = greedy_engine();
        let output = PipelineBuilder::new()
            .with_beam_search(cfg)
            .with_constraint(Box::new(OnlyToken(5)))
            .build()
            .run(vec![151644u32, 872], &mut engine);

        assert!(
            !output.token_ids.is_empty(),
            "constrained beam search must still produce tokens"
        );
        assert!(
            output.token_ids.iter().all(|&t| t == 5),
            "the attached constraint must be honored on the beam-search path too, got {:?}",
            output.token_ids
        );
    }

    #[test]
    fn test_pipeline_run_constraint_masks_tokens() {
        let mut engine = greedy_engine();
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(4)
            .greedy()
            .with_constraint(Box::new(OnlyToken(5)))
            .build();

        let output = pipeline.run(vec![151644u32, 872], &mut engine);
        assert_eq!(
            output.token_ids,
            vec![5u32, 5, 5, 5],
            "only the constraint-allowed token may be emitted"
        );
        assert_eq!(output.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_pipeline_run_constraint_completes() {
        let mut engine = greedy_engine();
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(5)
            .greedy()
            .with_constraint(Box::new(CompleteAfterN { n: 2, count: 0 }))
            .build();

        let output = pipeline.run(vec![151644u32, 872], &mut engine);
        assert_eq!(output.stop_reason, StopReason::ConstraintComplete);
        assert_eq!(
            output.token_ids.len(),
            2,
            "generation must stop when the constraint reports completion"
        );
    }

    #[test]
    fn test_pipeline_run_healing_applies() {
        // lookback=1, min_prob=0.0: the model's argmax over the prefix (the last
        // vocab index) differs from the final prompt token, so healing must fire
        // — proving it is actually driven against the model rather than a no-op.
        let mut engine = greedy_engine();
        let mut pipeline = PipelineBuilder::new()
            .max_tokens(3)
            .greedy()
            .with_token_healing(TokenHealingConfig::default())
            .build();

        let output = pipeline.run(vec![151644u32, 872], &mut engine);
        assert!(
            output.healing_applied,
            "token healing must actually run against the model"
        );
        assert_eq!(output.prompt_tokens, 2);
    }

    #[test]
    fn test_pipeline_try_run_ok() {
        let mut engine = greedy_engine();
        let mut pipeline = PipelineBuilder::new().max_tokens(4).greedy().build();

        let output = pipeline
            .try_run(vec![151644u32, 872], &mut engine)
            .expect("happy-path try_run must return Ok");
        assert_eq!(output.token_ids.len(), 4);
        assert!(matches!(output.stop_reason, StopReason::MaxTokens));
    }
}
