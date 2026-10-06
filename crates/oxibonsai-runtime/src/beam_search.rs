//! Beam search decoding for OxiBonsai.
//!
//! Beam search maintains `beam_width` candidate sequences simultaneously,
//! expanding each at every step and keeping the top-`beam_width` by
//! cumulative log-probability (with optional length penalty).
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_runtime::beam_search::{BeamSearchConfig, BeamSearchEngine};
//!
//! let config = BeamSearchConfig {
//!     beam_width: 2,
//!     max_tokens: 10,
//!     eos_token_id: 2,
//!     ..Default::default()
//! };
//! let engine = BeamSearchEngine::new(config);
//!
//! // Mock logits: always prefer token 5
//! let result = engine.search(vec![1, 2], 10, |_tokens, _step| {
//!     let mut logits = vec![0.0f32; 10];
//!     logits[5] = 10.0;
//!     logits[2] = -10.0; // EOS gets low score
//!     logits
//! });
//!
//! assert!(!result.best().is_empty());
//! ```
//!
//! # Constrained beam search
//!
//! [`BeamSearchEngine::search_with_constraint`] applies a [`TokenConstraint`]
//! to every live beam before top-k expansion (masking disallowed tokens) and
//! stops a beam early once the constraint reports completion, mirroring the
//! masking [`crate::pipeline::InferencePipeline::run`] already applies on the
//! autoregressive path. Because a [`TokenConstraint`] is stateful and cannot
//! generally be cloned, each beam's state is rebuilt from scratch (`reset` +
//! `advance` over that beam's own generated-so-far tokens) rather than being
//! tracked incrementally -- see the method docs for details.

use crate::constrained_decoding::TokenConstraint;

/// A [`TokenConstraint`] trait-object reference, with its object-lifetime
/// bound pinned to `'static` (matching the default bound `Box<dyn
/// TokenConstraint>` already carries -- every real implementation owns its
/// state rather than borrowing). Writing this out explicitly, instead of
/// relying on `&mut dyn TokenConstraint` elision (which ties the object bound
/// to the *reference's* lifetime instead), lets callers reborrow a boxed
/// constraint and pass it through multiple function boundaries without the
/// resulting invariance forcing every intermediate borrow to be `'static`.
type ConstraintRef<'a> = &'a mut (dyn TokenConstraint + 'static);

/// Outcome of replaying a beam's generated suffix through a
/// [`TokenConstraint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintReplay {
    /// The suffix contains a token the constraint refused in `advance`: the
    /// extension is illegal and must not become a beam.
    Violated,
    /// Legal so far, but not a valid terminal sequence yet.
    Incomplete,
    /// Legal and a valid terminal sequence: the beam is done.
    Complete,
}

// ─── Config ────────────────────────────────────────────────────────────────

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of parallel beams to maintain (typical: 4–8).
    pub beam_width: usize,
    /// Maximum tokens to generate per beam.
    pub max_tokens: usize,
    /// Length penalty exponent α: `score = log_prob / len^α`.
    ///
    /// Values in [0.6, 1.0] are typical. α = 1.0 is neutral; α < 1.0
    /// rewards longer sequences; α > 1.0 penalises them.
    pub length_penalty: f32,
    /// Block any token that would create a repeated n-gram of this size.
    /// Set to 0 to disable (default).
    pub no_repeat_ngram_size: usize,
    /// Stop as soon as the best beam generates an EOS token.
    pub early_stopping: bool,
    /// Token ID that marks end of sequence.
    pub eos_token_id: u32,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_tokens: 256,
            length_penalty: 0.6,
            no_repeat_ngram_size: 0,
            early_stopping: true,
            // Matches `crate::engine::EOS_TOKEN_ID`, the engine-wide fallback
            // EOS id used by every other decode path. Callers that know the
            // real (GGUF-resolved) EOS id should still set it explicitly;
            // `InferencePipeline::run` overrides this default with the live
            // engine's resolved EOS id automatically.
            eos_token_id: crate::engine::EOS_TOKEN_ID,
        }
    }
}

impl BeamSearchConfig {
    /// Returns this config with `eos_token_id` replaced by `engine_eos`, but
    /// *only* when it is still sitting at the library default sentinel
    /// (i.e. the caller never customised it). An explicit caller override --
    /// including one that happens to equal the default's numeric value by
    /// coincidence -- is always preserved.
    ///
    /// This is how [`crate::pipeline::InferencePipeline`] lets
    /// `BeamSearchConfig::default()` pick up the engine's real,
    /// GGUF-resolved EOS id (via [`InferenceEngine::eos_token_id`]) instead
    /// of silently decoding against the wrong id, while still letting a
    /// caller who knows better pin an explicit value.
    ///
    /// [`InferenceEngine::eos_token_id`]: crate::engine::InferenceEngine::eos_token_id
    pub fn inherit_eos_if_default(mut self, engine_eos: u32) -> Self {
        if self.eos_token_id == Self::default().eos_token_id {
            self.eos_token_id = engine_eos;
        }
        self
    }
}

// ─── Beam ──────────────────────────────────────────────────────────────────

/// One candidate sequence in the beam search.
#[derive(Debug, Clone)]
pub struct Beam {
    /// All token IDs in this candidate (prompt + generated so far).
    pub tokens: Vec<u32>,
    /// Cumulative log-probability of this sequence.
    pub log_prob: f64,
    /// Whether this beam has hit an EOS token and is finished.
    pub is_done: bool,
}

impl Beam {
    /// Create a new beam seeded with the given initial tokens.
    pub fn new(initial_tokens: Vec<u32>) -> Self {
        Self {
            tokens: initial_tokens,
            log_prob: 0.0,
            is_done: false,
        }
    }

    /// Length-normalised score used for beam ranking.
    ///
    /// `score = log_prob / (len ^ length_penalty)`
    ///
    /// Avoids division-by-zero by treating a zero-length sequence as length 1.
    pub fn score(&self, length_penalty: f32) -> f64 {
        let len = self.tokens.len().max(1) as f64;
        self.log_prob / len.powf(length_penalty as f64)
    }

    /// Extend the beam with one more token, returning a new beam.
    pub fn extend(&self, token: u32, log_prob: f64) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);
        Self {
            tokens,
            log_prob: self.log_prob + log_prob,
            is_done: false,
        }
    }

    /// Total number of tokens in this beam.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` when the beam contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ─── Result ────────────────────────────────────────────────────────────────

/// Output of a beam search run.
#[derive(Debug)]
pub struct BeamSearchResult {
    /// All completed sequences, ordered best-first.
    pub sequences: Vec<Vec<u32>>,
    /// Length-normalised score for each sequence.
    pub scores: Vec<f64>,
    /// Number of generation steps taken.
    pub num_steps: usize,
}

impl BeamSearchResult {
    /// The highest-scoring token sequence.
    pub fn best(&self) -> &[u32] {
        self.sequences.first().map(|s| s.as_slice()).unwrap_or(&[])
    }

    /// Score of the highest-scoring sequence.
    pub fn best_score(&self) -> f64 {
        self.scores.first().copied().unwrap_or(f64::NEG_INFINITY)
    }
}

// ─── Engine ────────────────────────────────────────────────────────────────

/// Beam search engine.
///
/// Decoupled from the model via a `get_logits` closure so it can be used
/// with any inference backend.
pub struct BeamSearchEngine {
    /// Search configuration.
    pub config: BeamSearchConfig,
}

impl BeamSearchEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    /// Run beam search with no [`TokenConstraint`] attached.
    ///
    /// `get_logits(beam_tokens, step)` is called for every live beam at every
    /// step and must return a logit vector of length `vocab_size`.
    ///
    /// Equivalent to [`search_with_constraint`](Self::search_with_constraint)
    /// with `constraint = None`.
    pub fn search<F>(
        &self,
        initial_tokens: Vec<u32>,
        vocab_size: usize,
        get_logits: F,
    ) -> BeamSearchResult
    where
        F: FnMut(&[u32], usize) -> Vec<f32>,
    {
        self.search_with_constraint(initial_tokens, vocab_size, get_logits, None)
    }

    /// Run beam search, optionally honouring an attached [`TokenConstraint`].
    ///
    /// `get_logits(beam_tokens, step)` is called for every live beam at every
    /// step and must return a logit vector of length `vocab_size`.
    ///
    /// When `constraint` is `Some`, every live beam's logits are masked
    /// (disallowed tokens forced to a large negative value) *before* top-k
    /// expansion, and a beam stops growing as soon as its candidate extension
    /// makes the constraint report completion -- exactly mirroring the
    /// masking + [`ConstraintComplete`](crate::pipeline::StopReason::ConstraintComplete)
    /// behaviour of the autoregressive path.
    ///
    /// Because beams genuinely diverge (each explores a different token
    /// sequence) and [`TokenConstraint`] is stateful but not `Clone`, the
    /// constraint's state is *not* tracked incrementally per beam. Instead,
    /// for every mask/completion check the constraint is `reset()` and
    /// replayed (`advance()`) over that beam's own generated-so-far tokens
    /// from scratch. This is O(depth) extra work per beam per step, which is
    /// negligible next to the full-sequence `get_logits` re-prefill already
    /// paid per beam per step by every caller in this crate.
    pub fn search_with_constraint<F>(
        &self,
        initial_tokens: Vec<u32>,
        vocab_size: usize,
        mut get_logits: F,
        mut constraint: Option<ConstraintRef<'_>>,
    ) -> BeamSearchResult
    where
        F: FnMut(&[u32], usize) -> Vec<f32>,
    {
        let cfg = &self.config;
        let bw = cfg.beam_width.max(1);
        // Every beam's `tokens` is `initial_tokens` (the prompt) followed by
        // whatever it has generated; this never shrinks, so `prompt_len` is a
        // stable split point for the "generated so far" suffix the
        // constraint operates on.
        let prompt_len = initial_tokens.len();

        // Initialise with a single beam
        let mut beams: Vec<Beam> = vec![Beam::new(initial_tokens)];
        let mut completed: Vec<Beam> = Vec::new();
        let mut steps = 0;

        for step in 0..cfg.max_tokens {
            steps = step + 1;

            // Collect live (non-done) beams
            let live: Vec<Beam> = beams.iter().filter(|b| !b.is_done).cloned().collect();

            if live.is_empty() {
                steps = step;
                break;
            }

            // Expand every live beam
            let mut candidates: Vec<Beam> = Vec::new();

            for beam in &live {
                // Extensions this parent contributed to the round, whether
                // they landed in `candidates` or (under `early_stopping`)
                // straight in `completed`. A parent that contributes none is
                // re-queued frozen below, so after this loop **every** live
                // beam is accounted for exactly once -- which is what lets
                // the `candidates.is_empty()` break drop the stale parents.
                let mut produced = 0usize;
                let mut logits = get_logits(&beam.tokens, step);

                // Apply no-repeat-ngram masking if configured
                if cfg.no_repeat_ngram_size > 0 {
                    Self::apply_no_repeat_ngram(
                        &mut logits,
                        &beam.tokens,
                        cfg.no_repeat_ngram_size,
                    );
                }

                // Apply the constraint mask (if any) before top-k expansion.
                if let Some(bc) = constraint.as_deref_mut() {
                    let suffix = &beam.tokens[prompt_len..];
                    if let Some(mask) = Self::replay_constraint_mask(bc, suffix, vocab_size) {
                        // Bound the "is anything still allowed" scan to
                        // what `logits` can actually reflect (mirrors
                        // `InferencePipeline::run_autoregressive`'s
                        // RT-PIPELINE guard in `pipeline.rs`): a mask
                        // longer than `logits` whose only allowed index
                        // sits past the end can never be honoured by
                        // masking `logits` alone, so it must be treated as
                        // fully disallowed here too.
                        let reachable = mask.len().min(logits.len());
                        if !mask[..reachable].iter().any(|&allowed| allowed) {
                            // Every token this beam could legally emit
                            // next is disallowed. Masking to an all-`-inf`
                            // vector and ranking it in `top_k_log_probs` is
                            // exactly the NaN-producing path RT-21
                            // describes for softmax (`(-inf) - (-inf)`
                            // once the log-softmax denominator is also
                            // `-inf`) -- and the old finite `-1e9`
                            // sentinel was no better: with every logit
                            // pinned to the *same* value, log-softmax
                            // hands every token an identical, meaningless
                            // log-probability and top-k fabricates a
                            // continuation that violates the very
                            // constraint that produced the mask. Detect
                            // the dead end here, before either vector is
                            // ever built, and freeze this beam instead of
                            // extending it.
                            //
                            // Freeze, don't drop (RAG-EVAL-IMG follow-up):
                            // an earlier version of this fix did a bare
                            // `continue`, which contributes nothing to
                            // `candidates` at all. That is harmless when
                            // *every* live beam stalls in the same round --
                            // `candidates` stays wholly empty, the "no
                            // candidates this round" fallback below fires,
                            // and `beams` (never reassigned) flows into
                            // `completed` untouched. But when only *some*
                            // live beams stall in a `beam_width > 1` round,
                            // `candidates` is non-empty from the siblings
                            // that did extend, so that fallback never
                            // fires: `beams = prune_beams(candidates, ...)`
                            // then unconditionally overwrites `beams`, and
                            // a `continue`d beam -- which was never added
                            // to `candidates` and is not `is_done` (so the
                            // "already-done beams from the previous round"
                            // drain above does not catch it either) --
                            // vanishes for good, however good its score.
                            //
                            // Marking it done and pushing it into
                            // `candidates` (rather than `completed`
                            // directly) reuses the existing, already-
                            // correct lifecycle a naturally-completed
                            // (EOS) beam gets when `early_stopping` is
                            // false: it competes fairly in this round's
                            // `prune_beams` alongside every real
                            // extension, then either survives into
                            // `beams` (draining into `completed` on a
                            // later round via the done-beams drain, or at
                            // the final "gather all remaining live beams"
                            // step) or is pruned out on merit like any
                            // other candidate -- never by an unconditional,
                            // score-blind drop. Pushing straight into
                            // `completed` instead (skipping `prune_beams`
                            // entirely) would lose it that fair comparison
                            // and, before this round's per-parent
                            // accounting existed, would also have been
                            // re-gathered a second time by the "no
                            // candidates this round" fallback below.
                            candidates.push(Self::frozen(beam));
                            continue;
                        }
                        Self::apply_constraint_mask(&mut logits, &mask);
                    }
                }

                // Get top-k (token, log_prob) candidates from this beam
                let top = Self::top_k_log_probs(&logits, bw);

                for (token, lp) in top {
                    let mut new_beam = beam.extend(token, lp);
                    let mut done = token == cfg.eos_token_id;

                    if !done {
                        if let Some(bc) = constraint.as_deref_mut() {
                            let mut ext_suffix: Vec<u32> = beam.tokens[prompt_len..].to_vec();
                            ext_suffix.push(token);
                            match Self::replay_constraint_status(bc, &ext_suffix) {
                                // The extension itself is illegal. A mask
                                // cannot always prevent this: a constraint
                                // may legitimately report itself
                                // unconstrained at this position
                                // (`allowed_tokens` -> `None`) while still
                                // refusing a specific token in `advance`,
                                // and the previous replay could only ask
                                // "is this complete?", whose `false`
                                // conflates "not yet" with "never". Reject
                                // the candidate instead of committing a
                                // beam the constraint has already refused.
                                ConstraintReplay::Violated => continue,
                                ConstraintReplay::Complete => done = true,
                                ConstraintReplay::Incomplete => {}
                            }
                        }
                    }

                    produced += 1;
                    if done {
                        new_beam.is_done = true;
                        if cfg.early_stopping {
                            completed.push(new_beam);
                            continue;
                        }
                    }
                    candidates.push(new_beam);
                }

                if produced == 0 {
                    // Nothing this beam could legally emit survived (every
                    // top-k extension was rejected by the constraint, or
                    // `get_logits` returned nothing to rank). Freeze it and
                    // let it compete on merit, exactly like a beam whose
                    // mask disallowed everything -- never drop it silently.
                    candidates.push(Self::frozen(beam));
                }
            }

            // Keep any already-done beams from the previous round
            // Use drain to avoid moving `beams` so we can still use it after break
            let done_indices: Vec<usize> = beams
                .iter()
                .enumerate()
                .filter(|(_, b)| b.is_done)
                .map(|(i, _)| i)
                .collect();
            // Remove done beams in reverse index order to preserve indices
            for &idx in done_indices.iter().rev() {
                completed.push(beams.remove(idx));
            }

            if candidates.is_empty() {
                // Every live beam was expanded this round and each one
                // either contributed an extension (to `candidates`, or --
                // under `early_stopping` -- directly to `completed`) or was
                // re-queued frozen by the `produced == 0` guard above. So an
                // empty `candidates` here means the round's whole output
                // went to `completed`, and the parents still sitting in
                // `beams` are stale, un-extended copies whose successors are
                // already recorded. Gathering them below would return a
                // pre-EOS prefix *next to* its own EOS-terminated
                // continuation -- and, carrying `log_prob == 0.0`, that
                // prefix outscores the continuation and is returned instead
                // of it.
                beams.clear();
                break;
            }

            // Prune to beam_width
            beams = Self::prune_beams(candidates, bw, cfg.length_penalty);

            // Early-stop when best completed beam outscores every live beam
            if cfg.early_stopping && !completed.is_empty() {
                let best_completed_score = completed
                    .iter()
                    .map(|b| b.score(cfg.length_penalty))
                    .fold(f64::NEG_INFINITY, f64::max);

                let best_live_score = beams
                    .iter()
                    .map(|b| b.score(cfg.length_penalty))
                    .fold(f64::NEG_INFINITY, f64::max);

                if best_completed_score >= best_live_score {
                    steps = step + 1;
                    break;
                }
            }
        }

        // Gather all remaining live beams as completed
        for b in beams {
            completed.push(b);
        }

        // Sort by score descending
        completed.sort_by(|a, b| {
            b.score(cfg.length_penalty)
                .partial_cmp(&a.score(cfg.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep at most beam_width results
        completed.truncate(bw);

        let scores: Vec<f64> = completed
            .iter()
            .map(|b| b.score(cfg.length_penalty))
            .collect();
        let sequences: Vec<Vec<u32>> = completed.into_iter().map(|b| b.tokens).collect();

        BeamSearchResult {
            sequences,
            scores,
            num_steps: steps,
        }
    }

    /// Rebuild constraint state from scratch and compute the allowed-token
    /// mask for `suffix` (a beam's generated-so-far tokens, prompt excluded).
    ///
    /// Returns `None` when the constraint reports itself unconstrained at
    /// this position (mirrors [`TokenConstraint::allowed_tokens`]).
    fn replay_constraint_mask(
        constraint: ConstraintRef<'_>,
        suffix: &[u32],
        vocab_size: usize,
    ) -> Option<Vec<bool>> {
        constraint.reset();
        for &t in suffix {
            if !constraint.advance(t) {
                // The beam's own history already violates the constraint
                // (should not happen in practice since violating tokens are
                // never committed -- see `search_with_constraint`'s
                // completion check) -- forbid every token defensively rather
                // than silently falling back to "unconstrained".
                return Some(vec![false; vocab_size]);
            }
        }
        constraint.allowed_tokens(suffix, vocab_size)
    }

    /// Rebuild constraint state from scratch and classify `suffix` (a beam's
    /// generated-so-far tokens, prompt excluded, including the candidate
    /// token under consideration).
    ///
    /// Three outcomes, not two: the previous boolean ("is it complete?")
    /// made a rejected token indistinguishable from a merely-unfinished one,
    /// so an extension the constraint refused in `advance` was still pushed
    /// as a live beam.
    fn replay_constraint_status(constraint: ConstraintRef<'_>, suffix: &[u32]) -> ConstraintReplay {
        constraint.reset();
        for &t in suffix {
            if !constraint.advance(t) {
                return ConstraintReplay::Violated;
            }
        }
        if constraint.is_complete() {
            ConstraintReplay::Complete
        } else {
            ConstraintReplay::Incomplete
        }
    }

    /// A done-marked clone of `beam`, used to re-queue a parent that could
    /// not legally extend so it competes in `prune_beams` on merit instead
    /// of vanishing.
    fn frozen(beam: &Beam) -> Beam {
        let mut frozen = beam.clone();
        frozen.is_done = true;
        frozen
    }

    /// Zero out (set to −∞) any token that would create a repeated n-gram.
    ///
    /// For each position in `tokens` where the last `ngram_size - 1` tokens
    /// match the trailing `ngram_size - 1` tokens of the current sequence,
    /// the following token is forbidden.
    pub fn apply_no_repeat_ngram(logits: &mut [f32], tokens: &[u32], ngram_size: usize) {
        if ngram_size == 0 || tokens.len() < ngram_size {
            return;
        }

        // The suffix we want to avoid repeating is the last (ngram_size - 1) tokens
        let prefix_len = ngram_size - 1;
        let suffix = &tokens[tokens.len() - prefix_len..];

        // Scan all valid n-gram starting positions in the existing token sequence
        for start in 0..tokens.len().saturating_sub(prefix_len) {
            let window = &tokens[start..start + prefix_len];
            if window == suffix {
                // The token that would complete the n-gram is at `start + prefix_len`
                let banned_token = tokens[start + prefix_len] as usize;
                if banned_token < logits.len() {
                    logits[banned_token] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Force every disallowed logit to `f32::NEG_INFINITY` -- an exact
    /// exclusion that holds at any temperature, unlike a large-but-finite
    /// sentinel such as `-1e9` (RT-20 class), which can retain nonzero
    /// post-softmax probability at extreme logit ranges. Mirrors
    /// `pipeline.rs`'s private `apply_constraint_mask`, exposed here as a
    /// `pub fn` (like this struct's own [`Self::apply_no_repeat_ngram`])
    /// so the exact write can be asserted directly in a test instead of
    /// only inferred from `search_with_constraint`'s black-box output --
    /// which cannot actually distinguish this sentinel from a finite one
    /// whenever the surviving allowed logit is realistic, since `exp()`
    /// underflows to precisely `0.0` for *either* sentinel once shifted
    /// far enough below the max (see the direct test below).
    ///
    /// Bounded by `logits.get_mut(i)`, exactly like `pipeline.rs`'s
    /// version: a `mask` longer than `logits` is handled by simply never
    /// reaching those extra entries, never by indexing out of bounds.
    ///
    /// **Visibility (decided for 0.2.4):** `pub(crate)`, not `pub`. It was
    /// extracted purely so the exact `-inf` write could be asserted in a
    /// test; `pub(crate)` keeps all of that test value (the tests live in
    /// this crate) without committing the project to a public API whose
    /// sibling in `pipeline.rs` is private. No caller outside this crate
    /// used it.
    pub(crate) fn apply_constraint_mask(logits: &mut [f32], mask: &[bool]) {
        for (i, &allowed) in mask.iter().enumerate() {
            if !allowed {
                if let Some(logit) = logits.get_mut(i) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Return the top-`k` `(token_id, log_prob)` pairs from a logit vector.
    ///
    /// Logits are converted to log-probabilities via log-softmax.
    pub fn top_k_log_probs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
        if logits.is_empty() {
            return Vec::new();
        }

        // Numerical stability: subtract max before exp
        let max_logit = logits
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);

        // Compute log-softmax: log_prob_i = logit_i - max - log(sum(exp(logit_j - max)))
        let shifted: Vec<f32> = logits
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    v - max_logit
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();

        let log_sum_exp = shifted.iter().copied().map(|v| v.exp()).sum::<f32>().ln();

        let mut indexed: Vec<(u32, f64)> = shifted
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, (v - log_sum_exp) as f64))
            .collect();

        // Sort by log-prob descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Keep the top `beam_width` beams by length-normalised score.
    pub fn prune_beams(mut beams: Vec<Beam>, beam_width: usize, length_penalty: f32) -> Vec<Beam> {
        beams.sort_by(|a, b| {
            b.score(length_penalty)
                .partial_cmp(&a.score(length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        beams.truncate(beam_width);
        beams
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Beam unit tests ────────────────────────────────────────────────────

    #[test]
    fn test_beam_new_initial() {
        let tokens = vec![1u32, 2, 3];
        let beam = Beam::new(tokens.clone());
        assert_eq!(beam.tokens, tokens);
        assert!((beam.log_prob - 0.0).abs() < f64::EPSILON);
        assert!(!beam.is_done);
        assert_eq!(beam.len(), 3);
        assert!(!beam.is_empty());
    }

    #[test]
    fn test_beam_score_length_penalty() {
        let beam = Beam {
            tokens: vec![1, 2, 3, 4],
            log_prob: -4.0,
            is_done: false,
        };
        // score = -4.0 / 4^0.6
        let expected = -4.0_f64 / (4.0_f64.powf(0.6));
        let score = beam.score(0.6);
        assert!(
            (score - expected).abs() < 1e-6,
            "score={score}, expected={expected}"
        );
    }

    #[test]
    fn test_beam_score_zero_length() {
        // An empty beam should not panic — treated as length 1
        let beam = Beam {
            tokens: vec![],
            log_prob: -1.0,
            is_done: false,
        };
        let score = beam.score(0.6);
        assert!((score - -1.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_beam_extend() {
        let beam = Beam {
            tokens: vec![1, 2],
            log_prob: -1.5,
            is_done: false,
        };
        let extended = beam.extend(3, -0.5);
        assert_eq!(extended.tokens, vec![1, 2, 3]);
        assert!((extended.log_prob - -2.0).abs() < 1e-10);
        assert!(!extended.is_done);
    }

    // ── top_k_log_probs tests ──────────────────────────────────────────────

    #[test]
    fn test_top_k_log_probs_returns_k_best() {
        // logits with clear winner at index 3
        let logits = vec![0.0f32, 1.0, 2.0, 10.0, 0.5];
        let result = BeamSearchEngine::top_k_log_probs(&logits, 2);
        assert_eq!(result.len(), 2);
        // Best token should be index 3
        assert_eq!(result[0].0, 3);
        // Log-probs should be in descending order
        assert!(result[0].1 >= result[1].1);
    }

    #[test]
    fn test_top_k_log_probs_k_larger_than_vocab() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let result = BeamSearchEngine::top_k_log_probs(&logits, 10);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_top_k_log_probs_empty() {
        let result = BeamSearchEngine::top_k_log_probs(&[], 4);
        assert!(result.is_empty());
    }

    // ── prune_beams tests ──────────────────────────────────────────────────

    #[test]
    fn test_prune_beams_keeps_best() {
        let beams = vec![
            Beam {
                tokens: vec![1],
                log_prob: -10.0,
                is_done: false,
            },
            Beam {
                tokens: vec![2],
                log_prob: -1.0,
                is_done: false,
            },
            Beam {
                tokens: vec![3],
                log_prob: -5.0,
                is_done: false,
            },
            Beam {
                tokens: vec![4],
                log_prob: -2.0,
                is_done: false,
            },
        ];
        let pruned = BeamSearchEngine::prune_beams(beams, 2, 1.0);
        assert_eq!(pruned.len(), 2);
        // Best beam has log_prob = -1.0 → tokens = [2]
        assert_eq!(pruned[0].tokens, vec![2]);
        // Second-best has log_prob = -2.0 → tokens = [4]
        assert_eq!(pruned[1].tokens, vec![4]);
    }

    #[test]
    fn test_prune_beams_fewer_than_width() {
        let beams = vec![Beam {
            tokens: vec![1],
            log_prob: -3.0,
            is_done: false,
        }];
        let pruned = BeamSearchEngine::prune_beams(beams, 4, 0.6);
        assert_eq!(pruned.len(), 1);
    }

    // ── apply_no_repeat_ngram tests ───────────────────────────────────────

    #[test]
    fn test_apply_no_repeat_ngram_blocks_repeated() {
        // tokens = [1, 2, 3]; ngram_size = 2 → last prefix is [3]
        // If [3] appeared before at position 1 (tokens[1]=2≠3), skip.
        // If [3] appeared before at position 2 (tokens[2]=3), following token is tokens[3] — but
        // tokens only has length 3, so that would be out of bounds. Let's use a longer sequence.
        //
        // tokens = [1, 2, 1, 2]; ngram_size = 2 → suffix = [2]
        // position 1: tokens[1]=2 matches; next token = tokens[2]=1 → ban token 1
        let tokens = vec![1u32, 2, 1, 2];
        let mut logits = vec![0.0f32; 5];
        BeamSearchEngine::apply_no_repeat_ngram(&mut logits, &tokens, 2);
        assert_eq!(logits[1], f32::NEG_INFINITY, "token 1 should be banned");
        // token 2 not yet banned (the last occurrence of [2] is at the very end,
        // no following token exists in history)
        assert!(logits[2].is_finite());
    }

    #[test]
    fn test_no_repeat_ngram_no_effect_when_disabled() {
        let tokens = vec![1u32, 2, 1, 2];
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut logits = original.clone();
        BeamSearchEngine::apply_no_repeat_ngram(&mut logits, &tokens, 0);
        assert_eq!(
            logits, original,
            "ngram_size=0 should leave logits unchanged"
        );
    }

    #[test]
    fn test_no_repeat_ngram_too_short_sequence() {
        // Sequence shorter than ngram_size → no banning
        let tokens = vec![1u32];
        let mut logits = vec![1.0f32; 5];
        BeamSearchEngine::apply_no_repeat_ngram(&mut logits, &tokens, 3);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    // ── BeamSearchEngine::search integration tests ────────────────────────

    #[test]
    fn test_beam_search_greedy_equivalent_width1() {
        // With beam_width=1 and greedy logits, beam search is equivalent to greedy decoding.
        let config = BeamSearchConfig {
            beam_width: 1,
            max_tokens: 5,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            early_stopping: false,
            eos_token_id: 99, // Never generated
        };
        let engine = BeamSearchEngine::new(config);

        // Always return token 7 as the best
        let result = engine.search(vec![0u32], 10, |_tokens, _step| {
            let mut logits = vec![0.0f32; 10];
            logits[7] = 100.0;
            logits
        });

        assert_eq!(result.num_steps, 5);
        let best = result.best();
        // First token is initial (0), remaining should all be 7
        assert!(best.iter().skip(1).all(|&t| t == 7));
    }

    #[test]
    fn test_beam_search_with_eos() {
        // Beam search should stop early when EOS is generated (early_stopping=true).
        let eos = 3u32;
        let config = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 20,
            length_penalty: 0.6,
            no_repeat_ngram_size: 0,
            early_stopping: true,
            eos_token_id: eos,
        };
        let engine = BeamSearchEngine::new(config);

        let step_counter = std::cell::Cell::new(0usize);
        let result = engine.search(vec![1u32], 5, |_tokens, _step| {
            step_counter.set(step_counter.get() + 1);
            // After 2 calls produce EOS as best token
            let mut logits = vec![0.0f32; 5];
            if step_counter.get() >= 2 {
                logits[eos as usize] = 100.0;
            } else {
                logits[1] = 5.0;
            }
            logits
        });

        // Should not have run all 20 steps
        assert!(
            result.num_steps < 20,
            "expected early stop, got {} steps",
            result.num_steps
        );
        assert!(!result.sequences.is_empty());
    }

    // ── BeamSearchConfig::default() EOS regression (runtime-engine-04) ─────

    #[test]
    fn test_beam_search_config_default_eos_matches_engine_wide_constant() {
        // Regression: BeamSearchConfig::default() used to hardcode
        // eos_token_id = 2, which never matches any real model's resolved
        // EOS (e.g. 151645), so default-config beam search never recognized
        // EOS and always ran to `max_tokens`. It must track the same
        // engine-wide fallback every other decode path uses.
        assert_eq!(
            BeamSearchConfig::default().eos_token_id,
            crate::engine::EOS_TOKEN_ID
        );
        assert_ne!(
            BeamSearchConfig::default().eos_token_id,
            2,
            "must not regress to the old hardcoded default"
        );
    }

    #[test]
    fn test_inherit_eos_if_default_overrides_the_sentinel() {
        // A config left at its default sentinel picks up the engine's real,
        // GGUF-resolved EOS id.
        let cfg = BeamSearchConfig::default().inherit_eos_if_default(151_643);
        assert_eq!(cfg.eos_token_id, 151_643);
    }

    #[test]
    fn test_inherit_eos_if_default_preserves_explicit_override() {
        // A caller who explicitly pinned a (non-default) eos_token_id keeps
        // it -- the inheritance must never clobber an intentional override.
        let cfg = BeamSearchConfig {
            eos_token_id: 999_999,
            ..Default::default()
        }
        .inherit_eos_if_default(151_643);
        assert_eq!(
            cfg.eos_token_id, 999_999,
            "explicit eos_token_id override must not be replaced"
        );
    }

    // ── Constrained beam search (runtime-engine-01) ─────────────────────────

    #[test]
    fn test_beam_search_with_constraint_masks_disallowed_tokens() {
        use crate::constrained_decoding::{JsonConstraint, TokenConstraint};

        // Toy-mode JsonConstraint: token id == ASCII code point.
        let vocab_size = 128usize;
        let config = BeamSearchConfig {
            beam_width: 3,
            max_tokens: 3,
            length_penalty: 0.6,
            no_repeat_ngram_size: 0,
            early_stopping: false,
            eos_token_id: 999_999, // never generated; isolates constraint behaviour
        };
        let engine = BeamSearchEngine::new(config);

        // Always most strongly prefer '}' -- a character that is *never*
        // valid as the opening character of a JSON document. An
        // unconstrained search would immediately emit it; a genuinely
        // applied constraint must mask it out of the top-k selection.
        let mut constraint = JsonConstraint::new();
        let result = engine.search_with_constraint(
            Vec::new(),
            vocab_size,
            |_tokens, _step| {
                let mut logits = vec![0.0f32; vocab_size];
                logits['}' as usize] = 100.0;
                logits['{' as usize] = 50.0;
                logits
            },
            Some(&mut constraint),
        );

        assert!(
            !result.sequences.is_empty(),
            "constrained beam search must still produce sequences"
        );
        for seq in &result.sequences {
            assert_ne!(
                seq.first().copied(),
                Some(b'}' as u32),
                "constraint must mask '}}' as an opening token: {seq:?}"
            );

            // Replay every returned beam through a fresh constraint instance:
            // every returned sequence must be a genuinely valid JSON prefix,
            // i.e. `advance` must never report a violation.
            let mut replay = JsonConstraint::new();
            for &tok in seq {
                assert!(
                    replay.advance(tok),
                    "beam {seq:?} contains a token the JSON constraint rejects"
                );
            }
        }
    }

    #[test]
    fn test_apply_constraint_mask_writes_exact_neg_infinity_not_finite_sentinel() {
        // Pins the RT-20-class sentinel fix at a granularity
        // `test_beam_search_with_constraint_masks_disallowed_tokens` (above)
        // cannot: that test only asserts a disallowed token is *absent*
        // from the output, which a finite `-1e9` sentinel (<< the allowed
        // token's logit) already guaranteed just as well as an exact
        // `-inf` does -- it passes identically under either sentinel, so it
        // does not actually pin which one `search_with_constraint` uses.
        // Assert directly on the masked buffer instead. The disallowed
        // index starts from an extreme value (1e30) precisely so that *if*
        // masking ever regressed to leaving the raw value untouched (or to
        // a sentinel that does not dominate it), that regression would be
        // impossible to miss.
        let mut logits = vec![50.0f32, 1e30, 10.0, -5.0];
        let mask = vec![true, false, true, true];
        BeamSearchEngine::apply_constraint_mask(&mut logits, &mask);

        assert_eq!(
            logits[1],
            f32::NEG_INFINITY,
            "a disallowed logit -- even one starting from an extreme value -- must become \
             exactly f32::NEG_INFINITY, not a finite sentinel such as -1e9: {logits:?}"
        );
        assert_eq!(logits[0], 50.0, "allowed entries must be untouched");
        assert_eq!(logits[2], 10.0, "allowed entries must be untouched");
        assert_eq!(logits[3], -5.0, "allowed entries must be untouched");

        // Feed the masked buffer through the same log-softmax
        // `search_with_constraint` actually ranks with: the disallowed
        // entry must carry exactly zero probability, not merely a small
        // one, and no NaN may appear anywhere in the ranking.
        let ranked = BeamSearchEngine::top_k_log_probs(&logits, logits.len());
        assert!(
            ranked.iter().all(|(_, lp)| !lp.is_nan()),
            "no ranked log-probability may be NaN: {ranked:?}"
        );
        let disallowed = ranked
            .iter()
            .find(|(token, _)| *token == 1)
            .expect("k == vocab_size returns every entry, including the disallowed one");
        assert_eq!(
            disallowed.1.exp(),
            0.0,
            "log-softmax must assign the masked entry exactly zero probability, got {}",
            disallowed.1
        );
    }

    #[test]
    fn test_beam_search_with_constraint_stops_beam_on_completion() {
        use crate::constrained_decoding::JsonConstraint;

        // `{}` is a complete JSON document after two tokens. With
        // early_stopping the beam must stop growing right there instead of
        // being forced to keep emitting (whitespace-only-valid) tokens up to
        // max_tokens.
        let vocab_size = 128usize;
        let config = BeamSearchConfig {
            beam_width: 1,
            max_tokens: 10,
            length_penalty: 0.6,
            no_repeat_ngram_size: 0,
            early_stopping: true,
            eos_token_id: 999_999, // never generated; isolates constraint behaviour
        };
        let engine = BeamSearchEngine::new(config);

        let mut constraint = JsonConstraint::new();
        let result = engine.search_with_constraint(
            Vec::new(),
            vocab_size,
            |tokens, _step| {
                let mut logits = vec![0.0f32; vocab_size];
                // Prefer '{' first, then '}' -- both always the single
                // strongest signal, so beam_width=1 greedily builds `{}`.
                if tokens.is_empty() {
                    logits['{' as usize] = 100.0;
                } else {
                    logits['}' as usize] = 100.0;
                }
                logits
            },
            Some(&mut constraint),
        );

        assert_eq!(
            result.best(),
            &[b'{' as u32, b'}' as u32],
            "beam must stop as soon as the constraint reports `{{}}` complete, not run to max_tokens"
        );
        assert!(
            result.num_steps < 10,
            "expected constraint-driven early stop, got {} steps",
            result.num_steps
        );
    }

    // ── RT-20 class sibling: an all-disallowed mask must stop the beam ──────
    // ── cleanly instead of ranking an all-`-inf`/uniform logit vector ───────

    /// Test-only constraint that reports every token disallowed, from the
    /// very first step, unconditionally -- gives full, deterministic
    /// control over the "fully unsatisfiable" scenario without depending
    /// on `JsonConstraint`'s own state machine ever reaching one.
    struct AlwaysUnsatisfiable;
    impl TokenConstraint for AlwaysUnsatisfiable {
        fn allowed_tokens(&self, _generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
            Some(vec![false; vocab_size])
        }
        fn advance(&mut self, _token: u32) -> bool {
            false
        }
        fn is_complete(&self) -> bool {
            false
        }
        fn reset(&mut self) {}
        fn name(&self) -> &str {
            "always-unsatisfiable"
        }
    }

    #[test]
    fn test_beam_search_constraint_fully_unsatisfiable_stops_cleanly_without_garbage() {
        // Reproduces the same finite-sentinel defect RT-20 fixed in
        // `pipeline.rs`, here in `search_with_constraint`: the old
        // `logits[i] = -1e9` sets every logit to the *same* finite value
        // when the whole vocabulary is disallowed. `top_k_log_probs`'s
        // log-softmax then hands every token an identical, meaningless
        // log-probability, and the beam gets extended with a low-index
        // token that the constraint never actually allowed -- a fabricated
        // continuation, not a NaN, but "garbage" all the same. (Swapping
        // to `f32::NEG_INFINITY` alone would be *worse*: log-softmax then
        // computes `(-inf) - (-inf) = NaN` for every entry once the
        // denominator is also `-inf`.) The fix must detect the
        // all-disallowed mask before ranking anything and drop the beam
        // from this round instead.
        let vocab_size = 16usize;
        let config = BeamSearchConfig {
            beam_width: 1,
            max_tokens: 3,
            eos_token_id: 999_999,
            early_stopping: false,
            ..Default::default()
        };
        let engine = BeamSearchEngine::new(config);
        let mut constraint = AlwaysUnsatisfiable;

        let initial = vec![1u32, 2];
        let result = engine.search_with_constraint(
            initial.clone(),
            vocab_size,
            |_tokens, _step| {
                // Any real-looking logits; the constraint must mask them
                // all away regardless of what the model would have said.
                (0..vocab_size).map(|i| i as f32).collect()
            },
            Some(&mut constraint),
        );

        assert!(
            result.scores.iter().all(|s| !s.is_nan()),
            "no score may be NaN, got {:?}",
            result.scores
        );
        assert_eq!(
            result.best(),
            initial.as_slice(),
            "a beam that can never legally extend must stay frozen at its last \
             valid length, not be extended with a fabricated (disallowed) token; \
             got {:?}",
            result.best()
        );
    }

    /// Test-only constraint whose allowed set is a pure function of the
    /// generated-so-far suffix (never mutable internal state), so a single
    /// shared instance -- reset and replayed independently per beam by
    /// `replay_constraint_mask` -- can hand two *different* live beams in
    /// the same round genuinely different masks: a beam that has generated
    /// `[0]` is a dead end (nothing ever legal again); a beam that has
    /// generated `[1]` may continue with token `2`.
    struct StallsAfterTokenZero;
    impl TokenConstraint for StallsAfterTokenZero {
        fn allowed_tokens(&self, generated: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
            let mut mask = vec![false; vocab_size];
            match generated {
                [] => {
                    mask[0] = true;
                    mask[1] = true;
                }
                [0] => {
                    // Dead end: no token is ever legal after `0`.
                }
                [1] => {
                    mask[2] = true;
                }
                _ => {}
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
            "stalls-after-token-zero"
        }
    }

    #[test]
    fn test_beam_search_stalled_beam_survives_alongside_beams_that_keep_extending() {
        // Reproduce first: with `beam_width > 1`, if exactly one of several
        // *live* beams hits a fully-disallowed mask while at least one
        // sibling does not, the old code's bare `continue` drops the
        // stalled beam from `candidates` without marking it `is_done`, so
        // it is neither carried into `completed` via the "already-done
        // beams from the previous round" drain (it was never marked done)
        // nor preserved by the "no candidates this round" fallback (that
        // only fires when *every* live beam stalls, making `candidates`
        // wholly empty). `beams = prune_beams(candidates, ...)` then
        // overwrites `beams` with only the siblings that produced a
        // candidate, and the stalled beam -- however good its score --
        // silently vanishes from every subsequent round and from the final
        // ranking.
        //
        // Round 1: both token 0 (logit 100) and token 1 (logit 50) are
        // allowed, so with beam_width=2 both become live beams. Round 2:
        // the `[0]`-suffixed beam's mask disallows everything (dead end);
        // the `[1]`-suffixed beam's mask allows only token 2, so it keeps
        // extending to `[1, 2]`. Both beams represent legally-generated,
        // constraint-valid prefixes and must both reach the final result.
        let vocab_size = 8usize;
        let config = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 2,
            length_penalty: 0.6,
            no_repeat_ngram_size: 0,
            early_stopping: false,
            eos_token_id: 999_999, // never generated; isolates constraint behaviour
        };
        let engine = BeamSearchEngine::new(config);
        let mut constraint = StallsAfterTokenZero;

        let result = engine.search_with_constraint(
            Vec::new(),
            vocab_size,
            |tokens, _step| {
                let mut logits = vec![0.0f32; vocab_size];
                if tokens.is_empty() {
                    logits[0] = 100.0;
                    logits[1] = 50.0;
                } else if tokens == [1] {
                    logits[2] = 100.0;
                }
                // tokens == [0]: irrelevant, the mask disallows everything.
                logits
            },
            Some(&mut constraint),
        );

        assert_eq!(
            result.sequences.len(),
            2,
            "the beam that stalled on a fully-disallowed mask must survive to the final \
             ranking alongside the beam that kept extending, not vanish; got {:?}",
            result.sequences
        );
        assert!(
            result.sequences.contains(&vec![0u32]),
            "the stalled beam (frozen at its last valid length) must be present; got {:?}",
            result.sequences
        );
        assert!(
            result.sequences.contains(&vec![1u32, 2]),
            "the beam that kept extending must also be present; got {:?}",
            result.sequences
        );
    }

    // ── An extension the constraint REJECTS must
    // ── never become a live beam ────────────────────────────────────────

    /// A constraint that masks nothing (`allowed_tokens` → `None`, i.e. "no
    /// active restriction at this position") but refuses token `3` in
    /// `advance`, and never reports itself complete.
    ///
    /// This is the shape that reaches the missing rejection path: with no
    /// mask there is nothing to stop the token being *selected*, and a
    /// replay that reports "not complete" says nothing about legality — so
    /// the violating token used to be committed as a live beam.
    struct RejectsTokenThree;
    impl TokenConstraint for RejectsTokenThree {
        fn allowed_tokens(&self, _generated: &[u32], _vocab_size: usize) -> Option<Vec<bool>> {
            None
        }
        fn advance(&mut self, token: u32) -> bool {
            token != 3
        }
        fn is_complete(&self) -> bool {
            false
        }
        fn reset(&mut self) {}
        fn name(&self) -> &str {
            "rejects-token-three"
        }
    }

    #[test]
    fn test_beam_search_rejects_a_constraint_violating_extension() {
        let vocab_size = 8usize;
        let config = BeamSearchConfig {
            beam_width: 1,
            max_tokens: 3,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            early_stopping: false,
            eos_token_id: 999_999, // never generated; isolates constraint behaviour
        };
        let engine = BeamSearchEngine::new(config);
        let mut constraint = RejectsTokenThree;

        let initial = vec![1u32, 2];
        let result = engine.search_with_constraint(
            initial.clone(),
            vocab_size,
            |_tokens, _step| {
                // The model wants token 3 above everything else.
                let mut logits = vec![0.0f32; vocab_size];
                logits[3] = 100.0;
                logits[4] = 1.0;
                logits
            },
            Some(&mut constraint),
        );

        for seq in &result.sequences {
            assert!(
                !seq[initial.len()..].contains(&3u32),
                "a token the constraint rejected in `advance` must never be committed \
                 to a beam; got {seq:?}"
            );
        }
        assert!(
            !result.sequences.is_empty(),
            "the parent beam must survive as a frozen candidate, not vanish"
        );
        assert!(
            result.scores.iter().all(|s| !s.is_nan()),
            "no score may be NaN, got {:?}",
            result.scores
        );
    }

    #[test]
    fn test_beam_search_rejection_keeps_the_legal_sibling_extension() {
        // With beam_width = 2 the top-2 are {3 (illegal), 4 (legal)}: the
        // illegal one is dropped and the legal one still extends, so the
        // rejection path must not freeze a beam that had a valid option.
        let vocab_size = 8usize;
        let config = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 1,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            early_stopping: false,
            eos_token_id: 999_999,
        };
        let engine = BeamSearchEngine::new(config);
        let mut constraint = RejectsTokenThree;

        let result = engine.search_with_constraint(
            Vec::new(),
            vocab_size,
            |_tokens, _step| {
                let mut logits = vec![0.0f32; vocab_size];
                logits[3] = 100.0;
                logits[4] = 50.0;
                logits
            },
            Some(&mut constraint),
        );

        assert!(
            result.sequences.contains(&vec![4u32]),
            "the legal extension must survive; got {:?}",
            result.sequences
        );
        assert!(
            !result.sequences.iter().any(|s| s.contains(&3u32)),
            "the illegal extension must not; got {:?}",
            result.sequences
        );
    }

    // ── When every live beam finishes in one round
    // ── under `early_stopping`, the stale parents must not be re-gathered ─

    #[test]
    fn test_beam_search_early_stopping_does_not_duplicate_stale_parents() {
        // beam_width = 1 and EOS ranked top: the single live beam's only
        // extension is EOS, which `early_stopping` sends straight to
        // `completed`. `candidates` is then empty, the loop breaks *before*
        // `beams` is reassigned, and the trailing "gather all remaining live
        // beams" step used to re-push the stale, un-extended, pre-EOS parent
        // — which, carrying `log_prob == 0.0`, outscores the real
        // EOS-terminated sequence and is returned instead of it.
        let vocab_size = 8usize;
        let config = BeamSearchConfig {
            beam_width: 1,
            max_tokens: 4,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            early_stopping: true,
            eos_token_id: 5,
        };
        let engine = BeamSearchEngine::new(config);

        let initial = vec![1u32, 2];
        // A *soft* distribution on purpose: EOS wins, but with a real
        // (negative) log-probability. A near-deterministic distribution would
        // give the EOS beam `log_prob ≈ 0`, tying it with the stale parent's
        // literal `0.0` and hiding the defect behind a stable sort.
        let result = engine.search(initial.clone(), vocab_size, |_tokens, _step| {
            let mut logits = vec![0.0f32; vocab_size];
            logits[5] = 2.0; // EOS wins every step
            logits[4] = 1.0;
            logits
        });

        assert!(
            !result.sequences.contains(&initial),
            "the stale pre-EOS parent must not be returned as a result; got {:?}",
            result.sequences
        );
        assert_eq!(
            result.best(),
            &[1u32, 2, 5],
            "the EOS-terminated sequence is the only completed beam; got {:?}",
            result.sequences
        );
    }

    /// A constraint that allows everything and calls any non-empty
    /// generation complete: every extension of every beam terminates in the
    /// round it is created.
    struct CompleteAfterOneToken;
    impl TokenConstraint for CompleteAfterOneToken {
        fn allowed_tokens(&self, _generated: &[u32], _vocab_size: usize) -> Option<Vec<bool>> {
            None
        }
        fn advance(&mut self, _token: u32) -> bool {
            true
        }
        fn is_complete(&self) -> bool {
            true
        }
        fn reset(&mut self) {}
        fn name(&self) -> &str {
            "complete-after-one-token"
        }
    }

    #[test]
    fn test_beam_search_early_stopping_all_beams_finish_without_duplicates() {
        // The same defect at beam_width = 2, driven by a constraint that
        // completes every extension: both of the round's candidates go
        // straight to `completed`, `candidates` stays empty, and the stale
        // parent used to be gathered alongside them — outscoring both, since
        // it carries `log_prob == 0.0`.
        let vocab_size = 8usize;
        let config = BeamSearchConfig {
            beam_width: 2,
            max_tokens: 4,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            early_stopping: true,
            eos_token_id: 999_999, // never generated; the constraint terminates
        };
        let engine = BeamSearchEngine::new(config);
        let mut constraint = CompleteAfterOneToken;

        let initial = vec![7u32];
        let result = engine.search_with_constraint(
            initial.clone(),
            vocab_size,
            |_tokens, _step| {
                let mut logits = vec![0.0f32; vocab_size];
                logits[1] = 2.0;
                logits[2] = 1.5;
                logits
            },
            Some(&mut constraint),
        );

        assert!(
            !result.sequences.contains(&initial),
            "the stale, un-extended parent must not be gathered next to its own \
             completed extensions; got {:?}",
            result.sequences
        );
        let mut deduped = result.sequences.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(
            deduped.len(),
            result.sequences.len(),
            "no sequence may appear twice; got {:?}",
            result.sequences
        );
        for seq in &result.sequences {
            assert_eq!(
                seq.len(),
                initial.len() + 1,
                "every returned sequence must be a completed extension; got {:?}",
                result.sequences
            );
        }
    }

    #[test]
    fn test_beam_search_result_best() {
        let result = BeamSearchResult {
            sequences: vec![vec![1, 2, 3], vec![4, 5, 6]],
            scores: vec![-0.5, -1.0],
            num_steps: 3,
        };
        assert_eq!(result.best(), &[1, 2, 3]);
        assert!((result.best_score() - -0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_beam_search_result_empty() {
        let result = BeamSearchResult {
            sequences: vec![],
            scores: vec![],
            num_steps: 0,
        };
        assert_eq!(result.best(), &[] as &[u32]);
        assert_eq!(result.best_score(), f64::NEG_INFINITY);
    }
}
