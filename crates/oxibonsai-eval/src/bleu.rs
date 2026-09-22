//! BLEU (Bilingual Evaluation Understudy) implementation.
//!
//! Follows Papineni et al. (2002) — *BLEU: a Method for Automatic Evaluation
//! of Machine Translation* — with additional smoothing variants for
//! sentence-level BLEU on sparse n-grams, inspired by (but not verified
//! byte-for-byte against) Chen & Cherry (2014); see [`SmoothingMethod`] for
//! exactly what each variant computes and how confident that attribution is.
//!
//! ## Algorithm
//!
//! For each n ∈ `[1..=max_n]`:
//!
//! ```text
//! p_n = Σ_{ngram ∈ candidate} min(count_cand(ngram), max_ref_count(ngram))
//!       / Σ_{ngram ∈ candidate} count_cand(ngram)
//! ```
//!
//! Brevity penalty (BP):
//!
//! ```text
//! BP = 1                 if c > r
//!    = exp(1 - r/c)      if 0 < c ≤ r
//!    = 0                 if c == 0
//! ```
//!
//! where `c = sum of candidate lengths` and `r = sum of *closest* reference
//! lengths` (with shortest-ref tie-break).
//!
//! Final score:
//!
//! ```text
//! BLEU = BP · exp( Σ_n (1/N) · log p_n )
//! ```
//!
//! ## Smoothing (sparse sentence-level)
//!
//! - [`SmoothingMethod::None`] — unsmoothed (zero if any p_n = 0).
//! - [`SmoothingMethod::AddOne`] — add 1 to every order's numerator and
//!   denominator (Lidstone-1 / "Laplace" smoothing, the formula commonly
//!   cited as Chen & Cherry 2014's method 2 / NLTK's `method2`).
//! - [`SmoothingMethod::ExpDecay`] — geometric decay `1 / (2^k · c)` on a
//!   zero-count order (see the variant's own doc for exactly what `k` and
//!   `c` are, and a caveat on its literature attribution).
//!
//! ## Empty candidate
//!
//! Returns `BleuScore { bleu: 0.0, precisions: [0.0; N], brevity_penalty: 0.0,
//! length_ratio: 0.0 }`.

use std::collections::HashMap;

use crate::rouge::{ngram_counts, tokenize, TokenSeq};

/// Smoothing strategy for sentence-level / sparse BLEU.
///
/// Corpus-level BLEU aggregates counts first and should generally use
/// [`SmoothingMethod::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmoothingMethod {
    /// No smoothing (classic Papineni — geometric mean may go to 0).
    #[default]
    None,
    /// Add 1 to both the numerator and the denominator of **every** order's
    /// modified precision, i.e. `p_n = (matches_n + 1) / (total_n + 1)` for
    /// every `n`, not only the orders where `matches_n == 0`.
    ///
    /// This is Lidstone-1 / "Laplace" smoothing, commonly cited as Chen &
    /// Cherry (2014)'s method 2 / NLTK's `SmoothingFunction.method2`, by
    /// analogy with the formula's simplicity — but, like the `ExpDecay`
    /// variant below, this has **not** been checked against a live NLTK or
    /// sacreBLEU install in this environment, so treat the name as a
    /// description of what the code computes, not a verified match to the
    /// literature.
    AddOne,
    /// Exponentially decaying smoothing for a zero-count order: when
    /// `matches_n == 0`, use `1 / (2^k · c)` instead of `0`, where `c` is the
    /// candidate's token length and `k` is a 1-based counter of how many
    /// zero-count orders (including this one) have been seen so far,
    /// starting over at each fresh evaluation (`k` resets whenever a
    /// non-zero order is seen, so it is *not* the NIST "keep counting for
    /// the rest of the sentence" rule below).
    ///
    /// This is inspired by the NIST geometric-sequence smoothing usually
    /// cited as Chen & Cherry (2014)'s method 3 / NLTK's `method3`, but is
    /// **not verified to be numerically identical to it**: the reference
    /// formula's denominator is the order's own n-gram *count*
    /// (`total_n`, so trigrams and 4-grams get different denominators), not
    /// the flat candidate length `c` used here, and its `k` counter never
    /// resets once a zero-count order is seen. Confirming (and, if needed,
    /// correcting) that difference needs golden vectors from a live
    /// NLTK/sacreBLEU install, which was not available in the environment
    /// this was implemented in (see `deviations` in the accompanying
    /// package report). Until then, treat the name as "an exponential
    /// zero-order smoother", not a literal implementation of the cited
    /// paper's method 3.
    ExpDecay,
}

/// One BLEU score for a (candidate, references) pair.
#[derive(Debug, Clone)]
pub struct BleuScore {
    /// Overall BLEU in `[0, 1]`.
    pub bleu: f32,
    /// Per-order modified precisions p_1, p_2, …, p_N.
    pub precisions: Vec<f32>,
    /// Brevity penalty factor applied.
    pub brevity_penalty: f32,
    /// Ratio `c / r` (candidate length over effective reference length).
    pub length_ratio: f32,
}

impl BleuScore {
    fn zero(max_n: usize) -> Self {
        Self {
            bleu: 0.0,
            precisions: vec![0.0; max_n],
            brevity_penalty: 0.0,
            length_ratio: 0.0,
        }
    }
}

/// BLEU configuration.
///
/// Marked `#[non_exhaustive]` so future fields (e.g. custom n-gram weights)
/// can be added without breaking downstream code. Construct via
/// [`BleuConfig::default`] or [`BleuConfig::new`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BleuConfig {
    /// Maximum n-gram order (`max_n`); default 4.
    pub max_n: usize,
    /// Smoothing method for sparse p_n (default [`SmoothingMethod::None`]).
    pub smoothing: SmoothingMethod,
}

impl Default for BleuConfig {
    fn default() -> Self {
        Self {
            max_n: 4,
            smoothing: SmoothingMethod::None,
        }
    }
}

impl BleuConfig {
    /// Build a config with explicit parameters.
    pub fn new(max_n: usize, smoothing: SmoothingMethod) -> Self {
        Self {
            max_n: max_n.max(1),
            smoothing,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sentence-level BLEU
// ──────────────────────────────────────────────────────────────────────────────

/// Sentence BLEU for a single candidate against one-or-more references.
///
/// Tokenization is word-level (see [`crate::rouge::tokenize`]).
pub fn sentence_bleu(candidate: &str, references: &[&str], cfg: &BleuConfig) -> BleuScore {
    let cand = tokenize(candidate);
    let refs: Vec<TokenSeq> = references.iter().map(|r| tokenize(r)).collect();
    sentence_bleu_tokens(&cand, &refs, cfg)
}

/// Sentence BLEU from pre-tokenised inputs.
pub fn sentence_bleu_tokens(
    candidate: &TokenSeq,
    references: &[TokenSeq],
    cfg: &BleuConfig,
) -> BleuScore {
    if candidate.is_empty() {
        return BleuScore::zero(cfg.max_n);
    }
    if references.is_empty() {
        return BleuScore::zero(cfg.max_n);
    }

    let c_len = candidate.len();
    let r_len = closest_ref_length(c_len, references);

    let mut precisions = Vec::with_capacity(cfg.max_n);
    let mut log_precision_sum = 0.0f64;
    let mut zero_streak = 0usize;
    let mut collapsed = false;

    for n in 1..=cfg.max_n {
        let (matches, total) = match_counts_sentence(candidate, references, n);
        let (p_n, used_total) =
            apply_smoothing(matches, total, cfg.smoothing, c_len, &mut zero_streak);
        precisions.push(p_n);

        if used_total == 0 || p_n <= 0.0 {
            collapsed = true;
            log_precision_sum = f64::NEG_INFINITY;
        } else if !collapsed {
            log_precision_sum += (p_n as f64).ln();
        }
    }

    let bp = brevity_penalty(c_len, r_len);
    let length_ratio = if r_len == 0 {
        0.0
    } else {
        c_len as f32 / r_len as f32
    };

    let bleu = if collapsed {
        0.0
    } else {
        let n = cfg.max_n as f64;
        let geo = (log_precision_sum / n).exp();
        (bp as f64 * geo) as f32
    };

    BleuScore {
        bleu,
        precisions,
        brevity_penalty: bp,
        length_ratio,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Corpus-level BLEU
// ──────────────────────────────────────────────────────────────────────────────

/// Corpus BLEU: aggregate modified precision counts across all sentences
/// before computing the geometric mean.
///
/// `references[i]` is the list of reference translations for the i-th
/// candidate. At least one reference per candidate is required.
pub fn corpus_bleu(candidates: &[&str], references: &[Vec<&str>], cfg: &BleuConfig) -> BleuScore {
    let cands: Vec<TokenSeq> = candidates.iter().map(|c| tokenize(c)).collect();
    let refs: Vec<Vec<TokenSeq>> = references
        .iter()
        .map(|refs_i| refs_i.iter().map(|r| tokenize(r)).collect())
        .collect();
    corpus_bleu_tokens(&cands, &refs, cfg)
}

/// Corpus BLEU from pre-tokenised inputs.
pub fn corpus_bleu_tokens(
    candidates: &[TokenSeq],
    references: &[Vec<TokenSeq>],
    cfg: &BleuConfig,
) -> BleuScore {
    if candidates.is_empty() || candidates.iter().all(|c| c.is_empty()) {
        return BleuScore::zero(cfg.max_n);
    }
    let n_eff = candidates.len().min(references.len());
    if n_eff == 0 {
        return BleuScore::zero(cfg.max_n);
    }

    let mut total_c_len = 0usize;
    let mut total_r_len = 0usize;
    let mut match_by_n = vec![0u64; cfg.max_n];
    let mut total_by_n = vec![0u64; cfg.max_n];

    for i in 0..n_eff {
        let cand = &candidates[i];
        let refs = &references[i];
        if cand.is_empty() || refs.is_empty() {
            continue;
        }
        total_c_len += cand.len();
        total_r_len += closest_ref_length(cand.len(), refs);

        for n in 1..=cfg.max_n {
            let (m, t) = match_counts_sentence(cand, refs, n);
            match_by_n[n - 1] += m as u64;
            total_by_n[n - 1] += t as u64;
        }
    }

    if total_c_len == 0 {
        return BleuScore::zero(cfg.max_n);
    }

    // Corpus BLEU uses a single smoothing decision per n across the whole
    // corpus. For `ExpDecay`, the "zero streak" restarts per corpus evaluation.
    let mut precisions = Vec::with_capacity(cfg.max_n);
    let mut log_sum = 0.0f64;
    let mut collapsed = false;
    let mut zero_streak = 0usize;

    for n in 0..cfg.max_n {
        let m = match_by_n[n] as usize;
        let t = total_by_n[n] as usize;
        let (p_n, used_total) = apply_smoothing(m, t, cfg.smoothing, total_c_len, &mut zero_streak);
        precisions.push(p_n);
        if used_total == 0 || p_n <= 0.0 {
            collapsed = true;
            log_sum = f64::NEG_INFINITY;
        } else if !collapsed {
            log_sum += (p_n as f64).ln();
        }
    }

    let bp = brevity_penalty(total_c_len, total_r_len);
    let length_ratio = if total_r_len == 0 {
        0.0
    } else {
        total_c_len as f32 / total_r_len as f32
    };

    let bleu = if collapsed {
        0.0
    } else {
        let nn = cfg.max_n as f64;
        (bp as f64 * (log_sum / nn).exp()) as f32
    };

    BleuScore {
        bleu,
        precisions,
        brevity_penalty: bp,
        length_ratio,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internals
// ──────────────────────────────────────────────────────────────────────────────

/// Compute clipped match count and total candidate n-gram count for a single sentence.
fn match_counts_sentence(cand: &TokenSeq, refs: &[TokenSeq], n: usize) -> (usize, usize) {
    let cand_counts = ngram_counts(cand, n);
    let total: usize = cand_counts.values().sum();
    if total == 0 {
        return (0, 0);
    }

    // Maximum reference count per n-gram across all references.
    let mut max_ref: HashMap<Vec<String>, usize> = HashMap::new();
    for r in refs {
        let rc = ngram_counts(r, n);
        for (k, v) in rc {
            let e = max_ref.entry(k).or_insert(0);
            if v > *e {
                *e = v;
            }
        }
    }

    let mut matches = 0usize;
    for (ngram, &cand_count) in &cand_counts {
        if let Some(&rc) = max_ref.get(ngram) {
            matches += cand_count.min(rc);
        }
    }
    (matches, total)
}

// `ngram_counts` itself lives in `crate::rouge` (RAG-EVAL-IMG-33's dedup
// note: this module used to carry a byte-identical private copy).

/// Find the length of the reference *closest* to the candidate (shortest tie-break).
fn closest_ref_length(c_len: usize, refs: &[TokenSeq]) -> usize {
    let mut best: Option<(usize, usize)> = None; // (abs_diff, len)
    for r in refs {
        let r_len = r.len();
        let diff = r_len.max(c_len) - r_len.min(c_len);
        match best {
            None => best = Some((diff, r_len)),
            Some((bd, bl)) => {
                if diff < bd || (diff == bd && r_len < bl) {
                    best = Some((diff, r_len));
                }
            }
        }
    }
    best.map(|(_, l)| l).unwrap_or(0)
}

fn brevity_penalty(c_len: usize, r_len: usize) -> f32 {
    if c_len == 0 {
        return 0.0;
    }
    if c_len > r_len {
        return 1.0;
    }
    (1.0f64 - r_len as f64 / c_len as f64).exp() as f32
}

/// Returns `(p_n, effective_denominator)`; if the denominator is 0,
/// caller treats the score as collapsed.
fn apply_smoothing(
    matches: usize,
    total: usize,
    method: SmoothingMethod,
    c_len: usize,
    zero_streak: &mut usize,
) -> (f32, usize) {
    match method {
        SmoothingMethod::None => {
            if total == 0 {
                (0.0, 0)
            } else {
                (matches as f32 / total as f32, total)
            }
        }
        SmoothingMethod::AddOne => {
            if total == 0 {
                (0.0, 0)
            } else {
                // Uniform add-1 on every order (see `SmoothingMethod::AddOne`
                // doc): `matches == 0` naturally falls out of this same
                // formula as `1 / (total + 1)`, so there is no separate case.
                ((matches as f32 + 1.0) / (total as f32 + 1.0), total + 1)
            }
        }
        SmoothingMethod::ExpDecay => {
            if total == 0 {
                return (0.0, 0);
            }
            if matches == 0 {
                *zero_streak += 1;
                let k = *zero_streak as f32;
                let denom = (2.0f32).powf(k) * c_len.max(1) as f32;
                (1.0 / denom, total)
            } else {
                *zero_streak = 0;
                (matches as f32 / total as f32, total)
            }
        }
    }
}
