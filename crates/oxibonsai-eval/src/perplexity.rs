//! Perplexity evaluator.
//!
//! Perplexity (PPL) measures how well a probability distribution predicted a
//! text sample. Lower is better.
//!
//! PPL = exp(−(1/N) · Σ log p(xᵢ | x<ᵢ))
//!
//! This module also provides bits-per-byte (BPB), an alternative metric
//! normalised by the number of UTF-8 bytes in the corpus.
//!
//! [`PerplexityEvaluator::compute_sliding`] implements the sliding-window
//! algorithm for sequences longer than a model's context window (see that
//! method's doc for why this needs *per-window* log-probabilities rather
//! than a single flat array, and for what `stride` actually controls).

use serde::Serialize;

use crate::error::EvalError;

// ──────────────────────────────────────────────────────────────────────────────
// PerplexityResult
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics from a batch perplexity evaluation.
#[derive(Debug, Serialize)]
pub struct PerplexityResult {
    /// Mean perplexity across all samples.
    pub mean_ppl: f32,
    /// Minimum perplexity across all samples.
    pub min_ppl: f32,
    /// Maximum perplexity across all samples.
    pub max_ppl: f32,
    /// Population standard deviation of perplexity values.
    pub std_ppl: f32,
    /// Number of samples evaluated.
    pub n_samples: usize,
    /// Total number of tokens processed.
    pub total_tokens: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// PerplexityEvaluator
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluator that computes perplexity from model log-probabilities.
pub struct PerplexityEvaluator {
    /// Sliding-window stride (in tokens) between the start of one scored
    /// context window and the next, consumed by
    /// [`Self::compute_sliding`] (default: 512). Unused by [`Self::compute`],
    /// [`Self::compute_batch`], [`Self::from_logits`] and
    /// [`Self::bits_per_byte`], which each already receive one
    /// log-probability per token and have no windows to stride between.
    pub stride: usize,
    /// Optional maximum sequence length to consider; [`Self::compute`] and
    /// [`Self::bits_per_byte`] truncate their input to this many tokens.
    /// [`Self::compute_sliding`] does *not* re-truncate its input: it
    /// expects each window the caller passes in to already be at most this
    /// long (i.e. this is a statement about how the windows were built,
    /// not something `compute_sliding` itself enforces).
    pub max_length: Option<usize>,
}

impl Default for PerplexityEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerplexityEvaluator {
    /// Create a new evaluator with sensible defaults (stride = 512, no max length).
    pub fn new() -> Self {
        Self {
            stride: 512,
            max_length: None,
        }
    }

    /// Create an evaluator with the specified sliding-window stride.
    pub fn with_stride(stride: usize) -> Self {
        Self {
            stride,
            max_length: None,
        }
    }

    /// Compute perplexity for a single sequence of log-probabilities.
    ///
    /// Each element of `log_probs` is the natural log-probability of the token
    /// at that position given all preceding tokens.
    ///
    /// Returns `f32::INFINITY` when `log_probs` is empty (undefined PPL).
    pub fn compute(&self, log_probs: &[f32]) -> f32 {
        let probs = match self.max_length {
            Some(max) => &log_probs[..log_probs.len().min(max)],
            None => log_probs,
        };

        if probs.is_empty() {
            return f32::INFINITY;
        }

        let n = probs.len() as f32;
        let avg_neg_log_prob = -probs.iter().copied().sum::<f32>() / n;
        avg_neg_log_prob.exp()
    }

    /// Compute perplexity statistics for a batch of log-probability sequences.
    ///
    /// Each inner `Vec<f32>` corresponds to one sample.
    /// Empty sequences are silently skipped.
    pub fn compute_batch(&self, log_probs_batch: &[Vec<f32>]) -> PerplexityResult {
        let ppls: Vec<f32> = log_probs_batch
            .iter()
            .filter(|lp| !lp.is_empty())
            .map(|lp| self.compute(lp))
            .collect();

        let total_tokens: usize = log_probs_batch.iter().map(Vec::len).sum();

        if ppls.is_empty() {
            return PerplexityResult {
                mean_ppl: f32::INFINITY,
                min_ppl: f32::INFINITY,
                max_ppl: f32::INFINITY,
                std_ppl: 0.0,
                n_samples: 0,
                total_tokens,
            };
        }

        let n = ppls.len() as f32;
        let mean_ppl = ppls.iter().copied().sum::<f32>() / n;
        let min_ppl = ppls.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_ppl = ppls.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let variance = ppls.iter().map(|p| (p - mean_ppl).powi(2)).sum::<f32>() / n;
        let std_ppl = variance.sqrt();

        PerplexityResult {
            mean_ppl,
            min_ppl,
            max_ppl,
            std_ppl,
            n_samples: ppls.len(),
            total_tokens,
        }
    }

    /// Compute perplexity from raw logits and the ground-truth token IDs.
    ///
    /// `logits[i]` is the vocabulary-wide logit vector at position `i`.
    /// `token_ids[i]` is the token that was actually observed at position `i`.
    ///
    /// The function applies the log-softmax over each logit vector and selects
    /// the log-prob corresponding to the ground-truth token.
    ///
    /// Returns `Err` — never panics — on caller data that cannot be scored:
    /// `logits` and `token_ids` of different lengths, an empty logit vector,
    /// a `token_id` out of range for its logit vector (previously an
    /// out-of-bounds-index *panic*, RAG-EVAL-IMG-28), or a logit row whose
    /// maximum is non-finite (an all-`-inf`/`NaN` row, which would otherwise
    /// silently poison the result with `NaN` instead of erroring). Mirrors
    /// [`crate::calibration::nll_from_logits`]'s error shape.
    pub fn from_logits(&self, logits: &[Vec<f32>], token_ids: &[u32]) -> Result<f32, EvalError> {
        if logits.len() != token_ids.len() {
            return Err(EvalError::MetricMismatch {
                expected: "equal-length logits and token_ids arrays",
                got: format!("{} vs {}", logits.len(), token_ids.len()),
            });
        }
        if logits.is_empty() {
            return Ok(f32::INFINITY);
        }

        let mut log_probs = Vec::with_capacity(logits.len());
        for (logit_vec, &token_id) in logits.iter().zip(token_ids.iter()) {
            if logit_vec.is_empty() {
                return Err(EvalError::Numerical("empty logit vector".to_string()));
            }
            let tid = token_id as usize;
            if tid >= logit_vec.len() {
                return Err(EvalError::MetricMismatch {
                    expected: "token_id < vocab size",
                    got: format!("token_id={tid} but only {} logits", logit_vec.len()),
                });
            }
            let max_logit = logit_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if !max_logit.is_finite() {
                return Err(EvalError::Numerical(
                    "non-finite max logit encountered (all -inf or NaN row)".to_string(),
                ));
            }
            let exp_sum: f32 = logit_vec.iter().map(|&l| (l - max_logit).exp()).sum();
            if !exp_sum.is_finite() || exp_sum <= 0.0 {
                return Err(EvalError::Numerical(
                    "log-sum-exp produced a non-finite or non-positive sum".to_string(),
                ));
            }
            let log_sum_exp = max_logit + exp_sum.ln();
            log_probs.push(logit_vec[tid] - log_sum_exp);
        }

        Ok(self.compute(&log_probs))
    }

    /// Sliding-window perplexity for a sequence longer than a model's
    /// context window (the Hugging Face `run_clm.py` recipe).
    ///
    /// `windows[i]` is the model's per-token log-probabilities for the
    /// i-th context window, where window `i` starts at absolute token
    /// position `i * self.stride` in the underlying token sequence (`self`'s
    /// `stride`, clamped to at least 1) and covers `windows[i].len()` tokens
    /// from there (normally [`Self::max_length`](PerplexityEvaluator::max_length)
    /// tokens, possibly fewer for the last window). `windows` must be the
    /// *complete, contiguous* sequence of windows starting at index 0 — a
    /// caller that omits a window (e.g. passes only windows 0 and 2) shifts
    /// every subsequent window's assumed start position, since this method
    /// has no way to tell "index 1 in this slice" from "the sequence's true
    /// window 1" apart from the slice index itself.
    ///
    /// Only the tokens beyond what an *earlier* window already covered
    /// contribute to the aggregate — so every position in the original
    /// sequence is counted exactly once, using the log-probability computed
    /// with the most leading context available, and no position is silently
    /// dropped just because the whole document does not fit in one context
    /// window. `stride < window length` gives overlapping windows (extra
    /// context, no extra counting); `stride >= window length` degenerates to
    /// non-overlapping chunks.
    ///
    /// Note on why this takes `&[Vec<f32>]` rather than a single flat
    /// `&[f32]`: once a document has already been scored end-to-end in one
    /// pass (one flat array of per-token log-probs), re-partitioning that
    /// *same, already-fixed* array into windows cannot change any value in
    /// it — windowing only changes results when different windows are
    /// scored with different amounts of leading context, which requires one
    /// model call per window. [`Self::compute`] already handles the
    /// single-pass case; this method is for the case a single pass cannot
    /// even produce (the sequence exceeds the model's context length), so
    /// it must accept one log-probability vector per window.
    ///
    /// Returns `f32::INFINITY` if `windows` is empty or contributes no
    /// tokens at all (e.g. every window is empty).
    pub fn compute_sliding(&self, windows: &[Vec<f32>]) -> f32 {
        let stride = self.stride.max(1);
        let mut neg_log_prob_sum = 0.0f64;
        let mut count = 0usize;
        let mut prev_end = 0usize; // first absolute position NOT yet counted

        for (i, window) in windows.iter().enumerate() {
            if window.is_empty() {
                continue;
            }
            let begin = i * stride;
            let end = begin + window.len();
            let new_start = prev_end.max(begin);
            if new_start >= end {
                // Every position in this window was already covered by an
                // earlier, at-least-as-context-rich window.
                continue;
            }
            let local_start = new_start - begin;
            for &lp in &window[local_start..] {
                neg_log_prob_sum -= f64::from(lp);
                count += 1;
            }
            prev_end = end;
        }

        if count == 0 {
            return f32::INFINITY;
        }
        (neg_log_prob_sum / count as f64).exp() as f32
    }

    /// Corpus-level perplexity: `exp(Σ(-log p) / Σ tokens)` pooled across
    /// every sample in `log_probs_batch`, applying the same
    /// [`Self::max_length`](PerplexityEvaluator::max_length) truncation
    /// [`Self::compute`] uses.
    ///
    /// This is a different statistic from [`PerplexityResult::mean_ppl`]
    /// (produced by [`Self::compute_batch`]), which is the *arithmetic mean
    /// of each sample's own PPL* — a macro-average that over-weights short
    /// samples relative to long ones. Pooling every token's
    /// log-probability before exponentiating once, as this method does, is
    /// the standard "corpus perplexity" definition. Returns `f32::INFINITY`
    /// for an empty batch, or a batch whose samples are all empty.
    pub fn corpus_perplexity(&self, log_probs_batch: &[Vec<f32>]) -> f32 {
        let mut neg_log_prob_sum = 0.0f64;
        let mut count = 0usize;
        for lp in log_probs_batch {
            let probs = match self.max_length {
                Some(max) => &lp[..lp.len().min(max)],
                None => lp.as_slice(),
            };
            for &v in probs {
                neg_log_prob_sum -= f64::from(v);
                count += 1;
            }
        }
        if count == 0 {
            return f32::INFINITY;
        }
        (neg_log_prob_sum / count as f64).exp() as f32
    }

    /// Compute bits-per-byte (BPB).
    ///
    /// BPB normalises perplexity by the number of bytes in the corpus:
    ///
    /// BPB = (−Σ log₂ p(xᵢ | x<ᵢ)) / n_bytes
    ///
    /// `log_probs` must be natural-log probabilities. Returns `f32::INFINITY`
    /// when `n_bytes == 0` or `log_probs` is empty.
    pub fn bits_per_byte(&self, log_probs: &[f32], n_bytes: usize) -> f32 {
        let probs = match self.max_length {
            Some(max) => &log_probs[..log_probs.len().min(max)],
            None => log_probs,
        };

        if probs.is_empty() || n_bytes == 0 {
            return f32::INFINITY;
        }

        // Convert nats to bits: log₂(x) = ln(x) / ln(2)
        let log2_e: f32 = std::f32::consts::E.log2();
        let neg_sum_log2_prob: f32 = probs.iter().map(|&lp| -lp * log2_e).sum();
        neg_sum_log2_prob / n_bytes as f32
    }
}
