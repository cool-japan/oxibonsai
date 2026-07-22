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
                        for (i, &allowed) in mask.iter().enumerate() {
                            if !allowed && i < logits.len() {
                                logits[i] = -1e9;
                            }
                        }
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
                            done = Self::replay_constraint_complete(bc, &ext_suffix);
                        }
                    }

                    if done {
                        new_beam.is_done = true;
                        if cfg.early_stopping {
                            completed.push(new_beam);
                            continue;
                        }
                    }
                    candidates.push(new_beam);
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

    /// Rebuild constraint state from scratch and report whether `suffix` (a
    /// beam's generated-so-far tokens, prompt excluded, including the
    /// candidate token under consideration) is now a complete, valid
    /// terminal sequence.
    fn replay_constraint_complete(constraint: ConstraintRef<'_>, suffix: &[u32]) -> bool {
        constraint.reset();
        for &t in suffix {
            if !constraint.advance(t) {
                return false;
            }
        }
        constraint.is_complete()
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
