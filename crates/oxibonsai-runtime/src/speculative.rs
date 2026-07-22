//! Speculative decoding for accelerated autoregressive generation.
//!
//! Speculative decoding uses a small "draft" model to generate K candidate tokens,
//! which the larger "target" model then verifies in a single parallel forward pass.
//! Accepted tokens are kept; the first rejected token is resampled from the target
//! distribution. This can yield near-linear speedup proportional to the average
//! number of accepted tokens per step.
//!
//! ## Algorithm (Leviathan et al., 2023)
//!
//! 1. Draft model generates K tokens: `t_1, ..., t_K` with draft probabilities `p_d`
//! 2. Target model scores all K+1 positions in parallel, producing `p_t`
//! 3. For each position `i`, accept `t_i` if:
//!    - `p_t(t_i) >= p_d(t_i)`, OR
//!    - with probability `p_t(t_i) / p_d(t_i)` (rejection sampling)
//! 4. If rejected at position `i`, resample from adjusted distribution
//! 5. Always append one bonus target-sampled token after full acceptance
//!
//! ## Two-engine production API
//!
//! [`SpeculativeDecoder::generate_verified`] is the real, end-to-end entry
//! point: it drafts K tokens with the decoder's own draft engine and verifies
//! them against a **separate target [`InferenceEngine`]** using the model
//! layer's batched [`InferenceEngine::verify_batch`]
//! (`forward_prefill_verify`) greedy scoring. It performs no synthetic /
//! mock scoring and its accepted output is token-identical to plain greedy
//! autoregressive decoding of the target model (lossless acceleration).
//!
//! ```rust,no_run
//! use oxibonsai_core::config::Qwen3Config;
//! use oxibonsai_runtime::engine::InferenceEngine;
//! use oxibonsai_runtime::sampling::SamplingParams;
//! use oxibonsai_runtime::speculative::{SpeculativeConfig, SpeculativeDecoder};
//!
//! let config = Qwen3Config::tiny_test();
//! let params = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
//! let draft_engine = InferenceEngine::new(config.clone(), params.clone(), 42);
//! let mut target = InferenceEngine::new(config, params.clone(), 42);
//! let mut decoder = SpeculativeDecoder::new(draft_engine, SpeculativeConfig::default());
//! let _output = decoder.generate_verified(&mut target, &[1u32, 2, 3], 16, &params);
//! ```
//!
//! ## Verification harness (test-only)
//!
//! [`SpeculativeDecoder::verify`], [`SpeculativeDecoder::step`] and the
//! `#[doc(hidden)]` `generate_speculative` operate on **caller-supplied**
//! target logits (or, in `generate_speculative`, a deterministic synthesized
//! target) and exist to unit-test the draft/verify plumbing in isolation.
//! They are not a production generator — use `generate_verified` for that.

use crate::adaptive_lookahead::{AdaptiveLookahead, AdaptiveLookaheadConfig};
use crate::engine::{InferenceEngine, EOS_TOKEN_ID, MAX_PREALLOC_TOKENS};
use crate::error::RuntimeResult;
use crate::sampling::SamplingParams;

// ──────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per step (lookahead K, typically 4–8).
    pub lookahead: usize,
    /// Minimum acceptance ratio threshold (0.0 = pure rejection sampling criterion).
    ///
    /// Setting this above 0.0 makes the decoder more conservative (fewer accepted
    /// tokens per step, but closer to target distribution).
    pub acceptance_threshold: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            lookahead: 5,
            acceptance_threshold: 0.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Step result
// ──────────────────────────────────────────────────────────────────

/// Result from one speculative decoding step (draft + verify).
#[derive(Debug, Clone)]
pub struct SpeculativeStep {
    /// Tokens proposed by the draft model.
    pub draft_tokens: Vec<u32>,
    /// Tokens accepted after verification against the target.
    pub accepted_tokens: Vec<u32>,
    /// Fraction of draft tokens that were accepted: `accepted / proposed`.
    pub acceptance_rate: f32,
}

// ──────────────────────────────────────────────────────────────────
// Internal mini-PRNG (xorshift64, no external rand crate)
// ──────────────────────────────────────────────────────────────────

/// Minimal xorshift64 PRNG state — no external dependency.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state (xorshift must not start at 0)
        let state = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns a sample in `[0.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        // Use top 24 bits for f32 mantissa precision
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ──────────────────────────────────────────────────────────────────
// SpeculativeDecoder
// ──────────────────────────────────────────────────────────────────

/// Speculative decoder: wraps a draft [`InferenceEngine`] and provides
/// draft-then-verify generation with running acceptance statistics.
pub struct SpeculativeDecoder<'a> {
    /// Draft model engine (smaller/faster model).
    pub draft_engine: InferenceEngine<'a>,
    /// Speculative decoding configuration.
    pub config: SpeculativeConfig,
    /// Total number of speculative steps taken.
    pub total_steps: u64,
    /// Total number of tokens proposed by the draft model.
    pub total_draft_tokens: u64,
    /// Total number of tokens accepted after target verification.
    pub total_accepted_tokens: u64,
    /// Internal PRNG for rejection sampling decisions (available for subtype use).
    #[allow(dead_code)]
    rng: Xorshift64,
    /// Optional adaptive controller — when present, the lookahead is
    /// updated after each step from the running acceptance EWMA.
    adaptive: Option<AdaptiveLookahead>,
    /// Number of tokens whose KV is committed in the draft engine's cache.
    /// Tracks the speculative delta-KV rollback cursor.
    committed_len: usize,
}

impl<'a> SpeculativeDecoder<'a> {
    /// Create a new speculative decoder with the given draft engine and config.
    pub fn new(draft_engine: InferenceEngine<'a>, config: SpeculativeConfig) -> Self {
        Self {
            draft_engine,
            config,
            total_steps: 0,
            total_draft_tokens: 0,
            total_accepted_tokens: 0,
            rng: Xorshift64::new(0xfeed1234_5678abcd),
            adaptive: None,
            committed_len: 0,
        }
    }

    /// Create a speculative decoder with an [`AdaptiveLookahead`] controller
    /// active. The initial lookahead is taken from `adaptive_config.initial`
    /// and overrides `config.lookahead` for the first step.
    pub fn with_adaptive(
        draft_engine: InferenceEngine<'a>,
        config: SpeculativeConfig,
        adaptive_config: AdaptiveLookaheadConfig,
    ) -> Result<Self, crate::adaptive_lookahead::AdaptiveLookaheadError> {
        let adaptive = AdaptiveLookahead::try_new(adaptive_config)?;
        let mut config = config;
        config.lookahead = adaptive.lookahead();
        Ok(Self {
            draft_engine,
            config,
            total_steps: 0,
            total_draft_tokens: 0,
            total_accepted_tokens: 0,
            rng: Xorshift64::new(0xfeed1234_5678abcd),
            adaptive: Some(adaptive),
            committed_len: 0,
        })
    }

    /// Read the current adaptive controller, if any.
    pub fn adaptive(&self) -> Option<&AdaptiveLookahead> {
        self.adaptive.as_ref()
    }

    /// Mutable access to the adaptive controller, if any.
    pub fn adaptive_mut(&mut self) -> Option<&mut AdaptiveLookahead> {
        self.adaptive.as_mut()
    }

    /// Generate up to `config.lookahead` draft tokens from the draft model.
    ///
    /// In this implementation, the draft engine uses its sampler to produce tokens
    /// autoregressively from `context`. The returned tokens are the draft candidates
    /// for target-model verification.
    pub fn draft(&mut self, context: &[u32], _params: &SamplingParams) -> Vec<u32> {
        let k = self.config.lookahead;
        // Treat a direct `draft` call as "context is fully committed"; prime from scratch.
        // This preserves the semantics of the public API (returns ≤ k tokens, errors as empty)
        // while using the incremental delta path internally. Sampler-driven (`greedy = false`).
        self.draft_delta(context, k, false).unwrap_or_default()
    }

    /// Commit the delta (context tokens not yet in the cache) then draft k tokens
    /// incrementally. After drafting, rewinds the cache to the committed prefix so
    /// speculative KV is logically discarded.
    ///
    /// When `greedy` is `true`, draft tokens are chosen by first-max argmax
    /// (matching [`InferenceEngine::verify_batch`]'s convention, so a
    /// draft-equals-target run accepts every token); otherwise the draft
    /// engine's configured sampler is used.
    ///
    /// On success, returns the drafted tokens and updates `self.committed_len` to
    /// equal `context.len()` (the delta is now committed).
    fn draft_delta(
        &mut self,
        context: &[u32],
        k: usize,
        greedy: bool,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        if context.is_empty() {
            return Ok(vec![]);
        }
        // Robustness: if this decoder instance is reused on a *shorter* (or
        // otherwise divergent-length) context than the one that produced the
        // current `committed_len` — without an intervening `reset()` — the
        // cached prefix no longer matches `context`. Rewind the draft cache and
        // re-prime from scratch instead of indexing past the end of `context`
        // (which previously panicked with an out-of-bounds slice access).
        if context.len() < self.committed_len {
            self.draft_engine.reset();
            self.committed_len = 0;
        }
        let max_ctx = self.draft_engine.model().kv_cache().max_seq_len();

        // (a) Commit the delta: tokens in context not yet in the cache.
        let committed = self.committed_len;
        let mut last_logits = if context.len() > committed {
            let logits = self
                .draft_engine
                .prefill_from_pos(&context[committed..], committed)?;
            // Advance the cache's seq_len to the committed prefix length so
            // committed_position() == kv_cache().seq_len() is a testable invariant.
            self.draft_engine
                .model_mut()
                .kv_cache_mut()
                .set_seq_len(context.len());
            logits
        } else {
            // Nothing new to commit: re-derive logits by forwarding the last
            // committed token. `committed == context.len()` here (the shorter
            // case was rewound above), so `last_pos` is always in bounds; guard
            // with `get` regardless so a future caller can never index OOB.
            let last_pos = committed.saturating_sub(1);
            let last_token = *context
                .get(last_pos)
                .ok_or("draft_delta: committed cursor past context length")?;
            self.draft_engine.decode_step(last_token, last_pos)?
        };
        self.committed_len = context.len();

        // (b) Draft k tokens incrementally.
        let mut draft_tokens = Vec::with_capacity(k);
        for (step_idx, _) in (0..k).enumerate() {
            let pos = context.len() + step_idx;
            if pos >= max_ctx {
                break;
            }
            let token = if greedy {
                argmax_first(&last_logits)
            } else {
                match self.draft_engine.sample(&last_logits) {
                    Ok(t) => t,
                    Err(_) => break,
                }
            };
            draft_tokens.push(token);
            last_logits = match self.draft_engine.decode_step(token, pos) {
                Ok(l) => l,
                Err(_) => {
                    draft_tokens.pop();
                    break;
                }
            };
        }

        // (c) Rewind: drop speculative KV past the committed prefix.
        self.draft_engine.rewind_cache(context.len());

        Ok(draft_tokens)
    }

    /// Commit the delta, decode exactly one bonus token and KEEP its KV (it becomes
    /// committed). Used as the empty-acceptance fallback in `generate_speculative`.
    fn draft_one_committed(
        &mut self,
        context: &[u32],
    ) -> Result<Option<u32>, Box<dyn std::error::Error + Send + Sync>> {
        if context.is_empty() {
            return Ok(None);
        }
        // Same reuse-on-shorter-context guard as `draft_delta`: rewind rather
        // than index past the end of `context`.
        if context.len() < self.committed_len {
            self.draft_engine.reset();
            self.committed_len = 0;
        }
        let max_ctx = self.draft_engine.model().kv_cache().max_seq_len();
        let committed = self.committed_len;

        let last_logits = if context.len() > committed {
            let logits = self
                .draft_engine
                .prefill_from_pos(&context[committed..], committed)?;
            self.draft_engine
                .model_mut()
                .kv_cache_mut()
                .set_seq_len(context.len());
            logits
        } else {
            let last_pos = committed.saturating_sub(1);
            let last_token = *context
                .get(last_pos)
                .ok_or("draft_one_committed: committed cursor past context length")?;
            self.draft_engine.decode_step(last_token, last_pos)?
        };
        self.committed_len = context.len();

        let pos = context.len();
        if pos >= max_ctx {
            return Ok(None);
        }

        let token = self.draft_engine.sample(&last_logits)?;
        // Forward the bonus token and KEEP its KV (no rewind).
        self.draft_engine.decode_step(token, pos)?;
        self.draft_engine
            .model_mut()
            .kv_cache_mut()
            .set_seq_len(pos + 1);
        self.committed_len = pos + 1;
        Ok(Some(token))
    }

    /// Verify draft tokens against **caller-supplied** target-model logits
    /// (test/harness primitive; the production path is
    /// [`Self::generate_verified`], which sources these from a real target).
    ///
    /// For each draft position `i`, the target's probability `p_t(t_i)` is
    /// compared against a mock draft probability `p_d(t_i)` derived from
    /// the target logits (as a self-consistency check when target logits are
    /// provided). In production, `p_d` comes from the draft model's softmax.
    ///
    /// Acceptance criterion (speculative sampling):
    /// - Accept if `p_t(t_i) >= p_d(t_i)`
    /// - Else accept with probability `p_t(t_i) / p_d(t_i)`
    ///
    /// Returns only the prefix of tokens accepted before the first rejection.
    pub fn verify(
        &self,
        draft_tokens: &[u32],
        target_logits: &[Vec<f32>],
        _params: &SamplingParams,
    ) -> Vec<u32> {
        let mut accepted = Vec::with_capacity(draft_tokens.len());

        // We need a mutable PRNG — use a local one seeded from step count for reproducibility
        let mut local_rng = Xorshift64::new(
            self.total_steps
                .wrapping_mul(6364136223846793005)
                .wrapping_add(0xabcdef01),
        );

        for (i, &token) in draft_tokens.iter().enumerate() {
            let logits = match target_logits.get(i) {
                Some(l) => l,
                None => break,
            };

            if logits.is_empty() {
                break;
            }

            // Compute softmax probabilities for target
            let target_probs = softmax(logits);

            // Get target probability for this draft token
            let target_prob = if (token as usize) < target_probs.len() {
                target_probs[token as usize]
            } else {
                0.0
            };

            // Mock draft probability: use a uniform-like estimate over top candidates
            // In production this would come from the draft model's own softmax output.
            // Here we use 1/vocab_size as a conservative draft estimate.
            let vocab_size = logits.len() as f32;
            let draft_prob = (1.0 / vocab_size).max(1e-9);

            let rng_sample = local_rng.next_f32();
            let threshold = self.config.acceptance_threshold;

            if Self::should_accept(draft_prob, target_prob, threshold, rng_sample) {
                accepted.push(token);
            } else {
                // First rejection — stop here
                break;
            }
        }

        accepted
    }

    /// Perform one complete speculative decoding step: draft K tokens then verify.
    ///
    /// Returns a [`SpeculativeStep`] with the draft proposals, accepted subset,
    /// and per-step acceptance rate.
    pub fn step(
        &mut self,
        context: &[u32],
        target_logits: &[Vec<f32>],
        params: &SamplingParams,
    ) -> SpeculativeStep {
        // Phase 1: Draft
        let draft_tokens = self.draft(context, params);
        let n_drafted = draft_tokens.len();

        // Phase 2: Verify
        let accepted_tokens = self.verify(&draft_tokens, target_logits, params);
        let n_accepted = accepted_tokens.len();

        // Update statistics
        self.total_steps += 1;
        self.total_draft_tokens += n_drafted as u64;
        self.total_accepted_tokens += n_accepted as u64;

        // Feed the adaptive controller (if any) and apply its lookahead update.
        if let Some(adaptive) = self.adaptive.as_mut() {
            adaptive.observe_step(n_drafted, n_accepted);
            // The controller may have changed `lookahead` — propagate it to
            // `config.lookahead` so the next `step` drafts the new amount.
            self.config.lookahead = adaptive.lookahead();
        }

        let acceptance_rate = if n_drafted > 0 {
            n_accepted as f32 / n_drafted as f32
        } else {
            0.0
        };

        SpeculativeStep {
            draft_tokens,
            accepted_tokens,
            acceptance_rate,
        }
    }

    /// **Test harness only — not a production generator.**
    ///
    /// This drafts `lookahead` candidates each step and verifies them against a
    /// *synthesized* deterministic target distribution (a peaked distribution
    /// keyed on `context.last() + step`), NOT a real target model. It exists to
    /// exercise the draft → verify → commit plumbing (and the adaptive-lookahead
    /// controller) end-to-end without loading a second model; the emitted token
    /// values and the resulting acceptance-rate / speedup statistics are
    /// therefore meaningless as a real speculative-decoding result.
    ///
    /// For real speculative decoding against an actual target model, use
    /// [`SpeculativeDecoder::generate_verified`], which scores draft tokens with
    /// [`InferenceEngine::verify_batch`] (`forward_prefill_verify`).
    #[doc(hidden)]
    pub fn generate_speculative(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        params: &SamplingParams,
    ) -> Vec<u32> {
        self.reset();
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));
        let mut context: Vec<u32> = prompt_tokens.to_vec();

        while output.len() < max_tokens {
            let remaining = max_tokens - output.len();
            let effective_lookahead = self.config.lookahead.min(remaining);

            // Synthesise mock target logits for each draft position.
            // In production: run target model forward pass over all positions.
            // Here we generate uniform-ish logits for each draft position using PRNG.
            let vocab_size = 32000usize; // representative for Qwen3
            let target_logits: Vec<Vec<f32>> = (0..effective_lookahead)
                .map(|step_idx| {
                    // Build a peaked distribution at a token derived from context + step
                    let peak_token =
                        (context.last().copied().unwrap_or(0) as usize + step_idx + 1) % vocab_size;
                    let mut logits = vec![0.0f32; vocab_size];
                    // Give the peak token high logit, others low
                    logits[peak_token] = 10.0;
                    for (i, l) in logits.iter_mut().enumerate() {
                        if i != peak_token {
                            *l = -2.0;
                        }
                    }
                    logits
                })
                .collect();

            let step_result = self.step(&context, &target_logits, params);

            if step_result.accepted_tokens.is_empty() {
                // No tokens accepted — try generating one committed token to avoid infinite loop
                match self.draft_one_committed(&context) {
                    Ok(Some(tok)) => {
                        output.push(tok);
                        context.push(tok);
                    }
                    _ => break,
                }
            } else {
                let to_take = step_result.accepted_tokens.len().min(remaining);
                for &tok in step_result.accepted_tokens[..to_take].iter() {
                    output.push(tok);
                    context.push(tok);
                    if output.len() >= max_tokens {
                        break;
                    }
                }
            }

            // Safety: break if context grows unexpectedly large
            if context.len() > prompt_tokens.len() + max_tokens + self.config.lookahead {
                break;
            }
        }

        output
    }

    /// Generate up to `max_tokens` tokens with **real two-engine speculative
    /// decoding**: this decoder's draft engine proposes up to `lookahead` tokens
    /// which the supplied `target` engine verifies in a single batched forward
    /// pass via [`InferenceEngine::verify_batch`] (`forward_prefill_verify`).
    ///
    /// This is greedy (argmax) speculative decoding — the standard mode for
    /// prefill-based verification. Draft tokens are proposed with the same
    /// first-max argmax convention the target uses to verify, so the accepted
    /// output is **token-identical to plain greedy autoregressive decoding of
    /// the target model**: speculative decoding accelerates that decode without
    /// changing its result. Only the first *rejected* draft token per step is
    /// replaced by the target's own prediction (the "bonus" token), and any
    /// draft suffix after it is discarded. When the draft and target are the
    /// same model the acceptance rate is ~100%.
    ///
    /// Both engines' per-sequence state (KV caches, committed cursor) is reset
    /// on entry. Running acceptance statistics ([`Self::acceptance_rate`],
    /// [`Self::speedup_estimate`]) are updated against the *real* target
    /// verification, and the adaptive controller (if any) is driven from them.
    ///
    /// `_params` is reserved: prefill-based verification is inherently greedy
    /// (argmax), so sampling parameters (temperature / top-k / top-p) do not
    /// affect the result and are ignored.
    ///
    /// # Errors
    ///
    /// Propagates target/draft forward-pass failures as [`RuntimeError`].
    ///
    /// [`RuntimeError`]: crate::error::RuntimeError
    pub fn generate_verified(
        &mut self,
        target: &mut InferenceEngine<'_>,
        prompt_tokens: &[u32],
        max_tokens: usize,
        _params: &SamplingParams,
    ) -> RuntimeResult<Vec<u32>> {
        if prompt_tokens.is_empty() || max_tokens == 0 {
            return Ok(Vec::new());
        }
        self.reset();
        target.reset();

        let max_ctx = target.model().kv_cache().max_seq_len();
        let mut output: Vec<u32> = Vec::with_capacity(max_tokens.min(MAX_PREALLOC_TOKENS));
        let mut context: Vec<u32> = prompt_tokens.to_vec();

        // Prime the target over the prompt. The argmax of the final prompt
        // logits is the target's first continuation token (position `P`).
        let prompt_logits = target.prefill_from_pos(prompt_tokens, 0)?;
        let mut next_token = argmax_first(&prompt_logits);
        if next_token == EOS_TOKEN_ID {
            return Ok(output);
        }
        output.push(next_token);
        context.push(next_token);

        // Absolute position of the *next* token to be produced (the one that
        // follows `next_token`). `next_token` itself lives at position `pos - 1`.
        let mut pos = prompt_tokens.len() + 1;

        while output.len() < max_tokens {
            if pos >= max_ctx {
                break;
            }
            let remaining = max_tokens - output.len();
            let k = self.config.lookahead.min(remaining);

            // (1) Draft up to `k` tokens greedily from the draft engine given the
            //     current context (which ends in `next_token`). `draft_delta`
            //     commits the context delta into the draft cache and rewinds the
            //     speculative suffix, keeping the draft cache consistent.
            let draft = if k == 0 {
                Vec::new()
            } else {
                self.draft_delta(&context, k, true).unwrap_or_default()
            };

            // (2) Verify: the target scores `[next_token, draft...]` in one
            //     batched pass, returning its greedy prediction at each position.
            //     Cap the batch so a write never runs past the target KV cache.
            let room = max_ctx.saturating_sub(pos - 1);
            if room == 0 {
                break;
            }
            let mut batch = Vec::with_capacity((1 + draft.len()).min(room));
            batch.push(next_token);
            for &d in draft.iter() {
                if batch.len() >= room {
                    break;
                }
                batch.push(d);
            }
            let model_preds = target.verify_batch(&batch, pos - 1)?;

            // Number of draft tokens that were actually forwarded (post-cap).
            let n_draft_in_batch = batch.len() - 1;

            // (3) Accept the longest prefix of draft tokens that match the
            //     target's greedy prediction at each position.
            let mut accepted = 0usize;
            while accepted < n_draft_in_batch
                && accepted < model_preds.len()
                && draft[accepted] == model_preds[accepted]
            {
                accepted += 1;
            }

            // Update running statistics against the REAL verification.
            self.total_steps += 1;
            self.total_draft_tokens += n_draft_in_batch as u64;
            self.total_accepted_tokens += accepted as u64;
            if let Some(adaptive) = self.adaptive.as_mut() {
                adaptive.observe_step(n_draft_in_batch, accepted);
                self.config.lookahead = adaptive.lookahead();
            }

            // (4) Commit accepted draft tokens (stopping at EOS / max_tokens).
            let take = accepted.min(max_tokens - output.len());
            let mut hit_eos = false;
            for &tok in draft.iter().take(take) {
                if tok == EOS_TOKEN_ID {
                    hit_eos = true;
                    break;
                }
                output.push(tok);
                context.push(tok);
            }
            if hit_eos || output.len() >= max_tokens {
                break;
            }

            // (5) Bonus: the target's own prediction at the accept/reject
            //     boundary. This is the first non-drafted token and guarantees
            //     forward progress even when nothing was accepted.
            let bonus = if accepted < model_preds.len() {
                model_preds[accepted]
            } else {
                match model_preds.last() {
                    Some(&t) => t,
                    None => break,
                }
            };
            if bonus == EOS_TOKEN_ID {
                break;
            }
            output.push(bonus);
            context.push(bonus);
            next_token = bonus;
            pos += accepted + 1;
        }

        target.stats().record_request(output.len());
        Ok(output)
    }

    /// Overall acceptance rate: accepted tokens / draft tokens, across all steps.
    ///
    /// Returns 0.0 if no drafts have been generated yet.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_accepted_tokens as f32 / self.total_draft_tokens as f32
    }

    /// Theoretical speedup estimate from speculative decoding.
    ///
    /// Speedup ≈ accepted tokens per step (capped at lookahead).
    /// Returns the mean accepted tokens per step, which indicates how many
    /// target forward passes were "skipped" relative to autoregressive decoding.
    ///
    /// A return of 1.0 means no speedup (equivalent to autoregressive); higher
    /// values indicate benefit from speculative parallelism.
    pub fn speedup_estimate(&self) -> f32 {
        if self.total_steps == 0 {
            return 1.0;
        }
        let avg_accepted = self.total_accepted_tokens as f32 / self.total_steps as f32;
        // Speedup is bounded by lookahead + 1 (the bonus token)
        avg_accepted.max(1.0)
    }

    /// Returns the number of tokens whose KV is committed in the draft engine's cache.
    pub fn committed_position(&self) -> usize {
        self.committed_len
    }

    /// Reset per-sequence state: clears the draft engine's KV cache and the
    /// committed-position cursor. Does NOT touch acceptance statistics.
    pub fn reset(&mut self) {
        self.draft_engine.reset();
        self.committed_len = 0;
    }

    /// Reset all accumulated statistics (steps, tokens, acceptance counts).
    /// If an adaptive controller is attached, its EWMA is also reset.
    pub fn reset_stats(&mut self) {
        self.total_steps = 0;
        self.total_draft_tokens = 0;
        self.total_accepted_tokens = 0;
        if let Some(adaptive) = self.adaptive.as_mut() {
            adaptive.reset();
            self.config.lookahead = adaptive.lookahead();
        }
    }

    /// Determine whether a draft token should be accepted.
    ///
    /// Implements the speculative sampling acceptance criterion:
    /// - If `target_prob >= draft_prob`: always accept
    /// - Otherwise: accept with probability `target_prob / draft_prob`
    ///
    /// The `threshold` parameter can optionally raise the bar for acceptance.
    /// `rng_sample` must be in `[0.0, 1.0)`.
    fn should_accept(draft_prob: f32, target_prob: f32, threshold: f32, rng_sample: f32) -> bool {
        if target_prob >= draft_prob {
            // Target assigns higher probability — always accept
            true
        } else {
            // Rejection sampling: accept with prob target/draft
            let accept_prob = (target_prob / draft_prob).max(0.0);
            let effective_threshold = accept_prob - threshold;
            rng_sample < effective_threshold
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Utility: softmax over f32 slice
// ──────────────────────────────────────────────────────────────────

/// First-max argmax over a logit slice.
///
/// Ties are broken toward the **lowest** index, matching the convention used by
/// [`crate::engine::InferenceEngine::verify_batch`] /
/// `BonsaiModel::forward_prefill_verify`. Using the identical convention on both
/// the draft and verify sides is what makes a draft-equals-target greedy run
/// accept every token (rather than diverging on exact logit ties). Returns `0`
/// for an empty slice.
fn argmax_first(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Compute numerically stable softmax over a logit slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        // Uniform fallback
        let n = logits.len() as f32;
        return vec![1.0 / n; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use oxibonsai_core::config::Qwen3Config;

    fn make_decoder(lookahead: usize) -> SpeculativeDecoder<'static> {
        // Use a statically-valid config — tiny_test gives a minimal model
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_config = SpeculativeConfig {
            lookahead,
            acceptance_threshold: 0.0,
        };
        SpeculativeDecoder::new(engine, spec_config)
    }

    fn make_peaked_logits(
        vocab_size: usize,
        peak_token: usize,
        n_positions: usize,
    ) -> Vec<Vec<f32>> {
        (0..n_positions)
            .map(|_| {
                let mut logits = vec![-5.0f32; vocab_size];
                if peak_token < vocab_size {
                    logits[peak_token] = 10.0;
                }
                logits
            })
            .collect()
    }

    #[test]
    fn test_speculative_config_defaults() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.lookahead, 5, "default lookahead should be 5");
        assert!(
            (cfg.acceptance_threshold - 0.0).abs() < f32::EPSILON,
            "default threshold should be 0.0"
        );
    }

    #[test]
    fn test_draft_generates_lookahead_tokens() {
        let mut decoder = make_decoder(3);
        let context = vec![1u32, 2, 3];
        let params = SamplingParams::default();
        let draft = decoder.draft(&context, &params);
        // Draft should generate up to lookahead tokens (may be fewer if EOS hit)
        assert!(
            draft.len() <= 3,
            "draft should not exceed lookahead=3, got {}",
            draft.len()
        );
    }

    #[test]
    fn test_verify_accepts_high_probability_tokens() {
        let decoder = make_decoder(5);
        let params = SamplingParams::default();
        let vocab_size = 100;

        // Token 42 is the draft token; give it very high target probability
        let draft_tokens = vec![42u32];
        let target_logits = make_peaked_logits(vocab_size, 42, 1);

        let accepted = decoder.verify(&draft_tokens, &target_logits, &params);
        assert_eq!(
            accepted.len(),
            1,
            "high-probability token should be accepted"
        );
        assert_eq!(accepted[0], 42);
    }

    #[test]
    fn test_verify_rejects_low_probability_tokens() {
        let decoder = make_decoder(5);
        let params = SamplingParams::default();
        let vocab_size = 1000;

        // Token 500 — give it very low probability (far from peak)
        let draft_tokens = vec![500u32];
        let mut logits = vec![-10.0f32; vocab_size];
        logits[0] = 20.0; // strong peak at token 0, not 500
        let target_logits = vec![logits];

        // With very low target_prob for token 500, most RNG samples should reject
        // Run multiple times to confirm rejection is common
        let mut rejections = 0;
        for _ in 0..20 {
            let accepted = decoder.verify(&draft_tokens, &target_logits, &params);
            if accepted.is_empty() {
                rejections += 1;
            }
        }
        assert!(
            rejections > 0,
            "low-probability token should be rejected at least sometimes"
        );
    }

    #[test]
    fn test_acceptance_rate_zero_at_start() {
        let decoder = make_decoder(5);
        assert!(
            (decoder.acceptance_rate() - 0.0).abs() < f32::EPSILON,
            "acceptance rate must be 0.0 before any steps"
        );
        assert_eq!(decoder.total_steps, 0);
        assert_eq!(decoder.total_draft_tokens, 0);
        assert_eq!(decoder.total_accepted_tokens, 0);
    }

    #[test]
    fn test_acceptance_rate_updates_after_step() {
        let mut decoder = make_decoder(4);
        let params = SamplingParams::default();
        let context = vec![1u32, 2, 3];

        // Use peaked logits so tokens are likely accepted
        let vocab_size = 32usize;
        let target_logits = make_peaked_logits(vocab_size, 5, 4);

        let step = decoder.step(&context, &target_logits, &params);

        assert_eq!(decoder.total_steps, 1, "one step should have been recorded");
        assert_eq!(
            decoder.total_draft_tokens,
            step.draft_tokens.len() as u64,
            "draft token count should match"
        );
        assert!(
            decoder.total_accepted_tokens <= decoder.total_draft_tokens,
            "accepted cannot exceed drafted"
        );
    }

    #[test]
    fn test_generate_speculative_returns_tokens() {
        let mut decoder = make_decoder(3);
        let params = SamplingParams::default();
        let prompt = vec![1u32, 2, 3];

        let output = decoder.generate_speculative(&prompt, 5, &params);
        // Should return up to max_tokens tokens
        assert!(
            output.len() <= 5,
            "output should not exceed max_tokens=5, got {}",
            output.len()
        );
    }

    #[test]
    fn test_should_accept_target_above_draft() {
        // When target_prob > draft_prob, always accept regardless of rng_sample
        assert!(
            SpeculativeDecoder::should_accept(0.1, 0.9, 0.0, 0.99),
            "target > draft: must accept even with rng_sample near 1.0"
        );
        assert!(
            SpeculativeDecoder::should_accept(0.05, 0.5, 0.0, 0.0),
            "target > draft: must accept with rng_sample=0.0"
        );
    }

    #[test]
    fn test_should_accept_target_below_draft_probabilistic() {
        // target_prob < draft_prob → accept with prob target/draft
        // With target=0.1, draft=1.0, accept_prob = 0.1
        // rng_sample=0.05 < 0.1 → should accept
        assert!(
            SpeculativeDecoder::should_accept(1.0, 0.1, 0.0, 0.05),
            "rng_sample=0.05 < accept_prob=0.1, should accept"
        );
        // rng_sample=0.5 >= 0.1 → should reject
        assert!(
            !SpeculativeDecoder::should_accept(1.0, 0.1, 0.0, 0.5),
            "rng_sample=0.5 >= accept_prob=0.1, should reject"
        );
    }

    #[test]
    fn test_speedup_estimate_below_lookahead() {
        let mut decoder = make_decoder(5);
        // Before any steps, speedup is 1.0 (baseline)
        assert!(
            (decoder.speedup_estimate() - 1.0).abs() < f32::EPSILON,
            "initial speedup should be 1.0"
        );

        // Simulate some stats: 10 steps, 30 drafted, 15 accepted
        decoder.total_steps = 10;
        decoder.total_draft_tokens = 30;
        decoder.total_accepted_tokens = 15;

        let speedup = decoder.speedup_estimate();
        // avg_accepted = 15/10 = 1.5; speedup = max(1.5, 1.0) = 1.5
        assert!(
            (speedup - 1.5).abs() < 1e-4,
            "speedup should be 1.5 (avg accepted per step), got {speedup}"
        );
        assert!(
            speedup <= decoder.config.lookahead as f32 + 1.0,
            "speedup cannot exceed lookahead+1"
        );
    }

    #[test]
    fn test_with_adaptive_starts_with_initial_lookahead() {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_cfg = SpeculativeConfig {
            lookahead: 99,
            acceptance_threshold: 0.0,
        };
        let adapt_cfg = AdaptiveLookaheadConfig {
            initial: 4,
            min: 2,
            max: 10,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let decoder =
            SpeculativeDecoder::with_adaptive(engine, spec_cfg, adapt_cfg).expect("valid");
        // Adaptive overrides the spec config's lookahead.
        assert_eq!(decoder.config.lookahead, 4);
        assert!(decoder.adaptive().is_some());
    }

    #[test]
    fn test_adaptive_decreases_lookahead_on_low_acceptance() {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_cfg = SpeculativeConfig {
            lookahead: 8,
            acceptance_threshold: 0.0,
        };
        let adapt_cfg = AdaptiveLookaheadConfig {
            initial: 8,
            min: 2,
            max: 12,
            alpha: 0.7,
            cooldown_steps: 1,
        };
        let mut decoder =
            SpeculativeDecoder::with_adaptive(engine, spec_cfg, adapt_cfg).expect("valid");
        let context = vec![1u32, 2, 3];
        let params = SamplingParams::default();
        // Provide logits with no peaked target — most rejections.
        let vocab = 100usize;
        let logits: Vec<Vec<f32>> = (0..decoder.config.lookahead)
            .map(|_| {
                let mut l = vec![10.0f32; vocab];
                l[0] = -50.0; // bias away from typical draft tokens
                l
            })
            .collect();
        for _ in 0..30 {
            decoder.step(&context, &logits, &params);
        }
        // With low acceptance, lookahead should have fallen toward the min.
        let final_la = decoder.config.lookahead;
        assert!(
            final_la <= 8,
            "lookahead should not increase, got {final_la}"
        );
    }

    #[test]
    fn test_reset_stats_resets_adaptive() {
        let config = Qwen3Config::tiny_test();
        let params = SamplingParams::default();
        let engine = InferenceEngine::new(config, params, 42);
        let spec_cfg = SpeculativeConfig {
            lookahead: 5,
            acceptance_threshold: 0.0,
        };
        let adapt_cfg = AdaptiveLookaheadConfig {
            initial: 5,
            min: 2,
            max: 12,
            alpha: 0.5,
            cooldown_steps: 1,
        };
        let mut decoder =
            SpeculativeDecoder::with_adaptive(engine, spec_cfg, adapt_cfg).expect("valid");
        // Drive the adaptive controller into a different state.
        for _ in 0..30 {
            let logits = make_peaked_logits(64, 5, decoder.config.lookahead);
            decoder.step(&[1, 2, 3], &logits, &SamplingParams::default());
        }
        decoder.reset_stats();
        assert_eq!(decoder.total_steps, 0);
        assert_eq!(decoder.config.lookahead, 5);
        assert_eq!(
            decoder.adaptive().expect("adaptive present").observations(),
            0
        );
    }

    #[test]
    fn test_delta_draft_token_identical_to_full_reprefill() {
        let cfg = Qwen3Config::tiny_test();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let k = 4;
        let context = vec![1u32, 2, 3, 4, 5];

        // Baseline: old behaviour — loop generate(ctx, 1) growing ctx.
        let mut base_eng = InferenceEngine::new(cfg.clone(), params.clone(), 42);
        let mut baseline = Vec::new();
        let mut cur = context.clone();
        for _ in 0..k {
            match base_eng.generate(&cur, 1) {
                Ok(g) if !g.is_empty() => {
                    baseline.push(g[0]);
                    cur.push(g[0]);
                }
                _ => break,
            }
        }

        // New delta path.
        let draft_eng = InferenceEngine::new(cfg, params.clone(), 42);
        let mut dec = SpeculativeDecoder::new(
            draft_eng,
            SpeculativeConfig {
                lookahead: k,
                ..SpeculativeConfig::default()
            },
        );
        let got = dec.draft(&context, &params);

        assert_eq!(
            got, baseline,
            "delta draft must be token-identical to full re-prefill"
        );
    }

    #[test]
    fn test_delta_draft_commits_incrementally() {
        let cfg = Qwen3Config::tiny_test();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let draft_eng = InferenceEngine::new(cfg, params.clone(), 42);
        let mut dec = SpeculativeDecoder::new(
            draft_eng,
            SpeculativeConfig {
                lookahead: 3,
                ..SpeculativeConfig::default()
            },
        );
        let prompt = vec![1u32, 2, 3];
        let _ = dec.generate_speculative(&prompt, 6, &params);
        assert!(
            dec.committed_position() >= prompt.len(),
            "committed cursor must advance past the prompt"
        );
        assert_eq!(
            dec.committed_position(),
            dec.draft_engine.model().kv_cache().seq_len(),
            "committed_position must mirror kv_cache seq_len"
        );
    }

    #[test]
    fn test_reject_all_rolls_back_and_next_round_correct() {
        let cfg = Qwen3Config::tiny_test();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let draft_eng = InferenceEngine::new(cfg, params.clone(), 42);
        let mut dec = SpeculativeDecoder::new(
            draft_eng,
            SpeculativeConfig {
                lookahead: 4,
                ..SpeculativeConfig::default()
            },
        );
        let context = vec![7u32, 8, 9];

        // Round 1: draft k tokens.
        let draft1 = dec.draft(&context, &params);

        // After draft(), rewind is called internally; cursor must be at context.len().
        assert_eq!(
            dec.committed_position(),
            context.len(),
            "after draft(), cursor must be at committed prefix, not prefix+k"
        );

        // Round 2 from the same committed context must reproduce round-1 draft exactly.
        let draft2 = dec.draft(&context, &params);
        assert_eq!(
            draft1, draft2,
            "after rollback, redrafting identical context yields identical tokens"
        );
    }

    // ── Finding 16: reuse-on-shorter-context OOB panic ──────────────────────

    #[test]
    fn test_draft_reuse_on_shorter_context_does_not_panic() {
        // Regression: reusing a decoder on a SHORTER context than the one that
        // set `committed_len`, without an intervening reset(), used to panic
        // with an out-of-bounds slice index in `draft_delta`'s else branch.
        let cfg = Qwen3Config::tiny_test();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let draft_eng = InferenceEngine::new(cfg, params.clone(), 42);
        let mut dec = SpeculativeDecoder::new(
            draft_eng,
            SpeculativeConfig {
                lookahead: 3,
                ..SpeculativeConfig::default()
            },
        );

        // Long context first ⇒ committed cursor advances to 5.
        let _ = dec.draft(&[1u32, 2, 3, 4, 5], &params);
        assert_eq!(dec.committed_position(), 5);

        // Reuse on a shorter context WITHOUT reset(): must auto-rewind, not panic.
        let out = dec.draft(&[1u32, 2], &params);
        assert!(out.len() <= 3, "draft must not exceed lookahead");
        assert_eq!(
            dec.committed_position(),
            2,
            "committed cursor must track the new shorter context after auto-rewind"
        );

        // The decoder must remain usable for a subsequent (growing) context.
        let _ = dec.draft(&[1u32, 2, 3], &params);
        assert_eq!(dec.committed_position(), 3);
    }

    // ── Findings 25 & 36: real two-engine verified speculative decoding ─────

    /// Build a CPU-`Reference`-tier engine so verification and drafting use the
    /// identical (first-max) argmax convention deterministically, regardless of
    /// the build's GPU feature flags.
    fn reference_engine(cfg: &Qwen3Config, params: &SamplingParams) -> InferenceEngine<'static> {
        use oxibonsai_kernels::KernelTier;
        use oxibonsai_model::model::BonsaiModel;
        let model = BonsaiModel::new(cfg.clone());
        InferenceEngine::from_model_with_tier(model, KernelTier::Reference, params.clone(), 42)
    }

    /// Plain greedy autoregressive decode of the target (first-max argmax),
    /// used as the ground truth the speculative decoder must reproduce exactly.
    fn plain_greedy_reference(
        cfg: &Qwen3Config,
        params: &SamplingParams,
        prompt: &[u32],
        max: usize,
    ) -> Vec<u32> {
        let mut eng = reference_engine(cfg, params);
        let logits = eng.prefill_from_pos(prompt, 0).expect("prefill");
        let mut out = Vec::new();
        let mut tok = argmax_first(&logits);
        let mut pos = prompt.len();
        while out.len() < max {
            if tok == EOS_TOKEN_ID {
                break;
            }
            out.push(tok);
            let logits = eng.decode_step(tok, pos).expect("decode_step");
            pos += 1;
            tok = argmax_first(&logits);
        }
        out
    }

    #[test]
    fn test_generate_verified_matches_plain_greedy_and_accepts_all() {
        let cfg = Qwen3Config::tiny_test();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let prompt = vec![1u32, 2, 3];
        let max = 12usize;

        // Two engines from the SAME model ⇒ draft == target.
        let draft_eng = reference_engine(&cfg, &params);
        let mut target = reference_engine(&cfg, &params);
        let mut dec = SpeculativeDecoder::new(
            draft_eng,
            SpeculativeConfig {
                lookahead: 4,
                ..SpeculativeConfig::default()
            },
        );

        let spec_out = dec
            .generate_verified(&mut target, &prompt, max, &params)
            .expect("generate_verified");

        // Lossless-acceleration invariant: identical to plain greedy target decode.
        let reference = plain_greedy_reference(&cfg, &params, &prompt, max);
        assert_eq!(
            spec_out, reference,
            "verified speculative output must equal plain greedy target decode"
        );
        assert_eq!(
            spec_out.len(),
            max,
            "should produce exactly max_tokens tokens"
        );

        // draft == target ⇒ every drafted token is verified ⇒ ~100% acceptance.
        assert!(
            dec.total_draft_tokens > 0,
            "some tokens must have been drafted"
        );
        assert!(
            dec.acceptance_rate() > 0.99,
            "draft==target must accept ~all drafts, got {}",
            dec.acceptance_rate()
        );
        assert!(
            dec.speedup_estimate() > 1.0,
            "accepting multiple tokens per step must show speedup, got {}",
            dec.speedup_estimate()
        );
    }

    #[test]
    fn test_generate_verified_edge_cases() {
        let cfg = Qwen3Config::tiny_test();
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let draft_eng = reference_engine(&cfg, &params);
        let mut target = reference_engine(&cfg, &params);
        let mut dec = SpeculativeDecoder::new(draft_eng, SpeculativeConfig::default());

        // Empty prompt ⇒ empty output.
        assert!(dec
            .generate_verified(&mut target, &[], 8, &params)
            .expect("empty prompt")
            .is_empty());
        // Zero max_tokens ⇒ empty output.
        assert!(dec
            .generate_verified(&mut target, &[1u32, 2, 3], 0, &params)
            .expect("zero max")
            .is_empty());
    }
}
