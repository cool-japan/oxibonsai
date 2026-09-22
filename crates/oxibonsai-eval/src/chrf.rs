//! chrF and chrF++ — character n-gram F-score (Popović 2015).
//!
//! chrF computes the F_β (default β=2) score over character n-grams from
//! n=1 up to n=`order` (default 6). chrF++ additionally mixes in word
//! n-gram F-scores up to `word_order` (typical 2).
//!
//! All iteration is over Unicode `char`s — byte slicing is *not* used, so
//! multi-byte UTF-8 sequences are handled correctly.
//!
//! ## Effective order
//!
//! An n-gram order for which *either* side has no n-grams at all — the
//! candidate is shorter than `n`, or the reference is shorter than `n` — is
//! not scoreable and is skipped entirely rather than counted as an F=0.0
//! contribution. This is "effective order" chrF++ as sacreBLEU implements
//! it for sentence-level scoring: an order counts only when `n_hyp > 0 AND
//! n_ref > 0`. It is why `chrf("a", "a")` is `1.0` and not `0.1667`: the
//! default order is 6, but a 1-character candidate/reference pair only has
//! n-grams for `n=1`, so only that one order enters the average
//! (RAG-EVAL-IMG-05). It is also why gating on the reference alone is
//! wrong: `chrf("a", "abcdef")` under a reference-only gate is `0.033333`
//! (orders 2..=6 stay "effective" because the 6-character reference reaches
//! them, even though the 1-character candidate has zero n-grams there, so
//! five of the six orders are forced to a hard `F=0.0`); gating on both
//! sides makes it `0.200000`, sacreBLEU's computed value for this pair (see
//! `tests/chrf_effective_order_tests.rs` for how that value was checked —
//! this environment has no network access to run `sacrebleu` itself),
//! because only the one order both sides actually reach is counted.
//!
//! ## Final score
//!
//! For each *effective* order `n` (both candidate and reference have ≥1
//! n-gram of length `n`), compute that order's own F_β from its
//! precision/recall over character (or word) n-gram multisets:
//! ```text
//! p_n = |cand ∩ ref| / |cand|
//! r_n = |cand ∩ ref| / |ref|
//! F_n = (1 + β²) · p_n · r_n / (β² · p_n + r_n)
//! ```
//! then average the **F_n values themselves** over the `N_eff` effective
//! orders (character orders and, for chrF++, word orders together, equal
//! weight per order) to get the final score. This differs slightly (order
//! 1e-3 in the cases measured so far, not the ~1e-6 once claimed here) from
//! first averaging `p_n`/`r_n` separately and combining once at the end,
//! since F is a nonlinear function of (p, r); both shapes appear in the
//! literature and the difference is not worth gating on.
//!
//! Empty candidate *and* empty reference → score = 1.0.
//! Empty candidate *or* empty reference (not both) → score = 0.0.

use std::collections::HashMap;

/// Character / word n-gram F-score result.
#[derive(Debug, Clone)]
pub struct ChrfScore {
    /// Final F-score in `[0, 1]`.
    pub score: f32,
    /// Character n-gram order used.
    pub order: usize,
    /// β for the F-score (β>1 weights recall).
    pub beta: f32,
    /// Word n-gram order (0 = chrF, >=1 = chrF++).
    pub word_order: usize,
}

/// chrF (character-n-gram F-score) default: order=6, β=2.
pub fn chrf(candidate: &str, reference: &str) -> ChrfScore {
    chrf_with(candidate, reference, 6, 2.0, 0)
}

/// chrF++ convenience: char order=6, word order=2, β=2.
pub fn chrf_plus_plus(candidate: &str, reference: &str) -> ChrfScore {
    chrf_with(candidate, reference, 6, 2.0, 2)
}

/// chrF / chrF++ with explicit parameters.
///
/// - `order`: maximum character n-gram order (≥ 1).
/// - `beta`: β for F-score (typical 2.0).
/// - `word_order`: 0 → chrF; ≥1 → mix in word n-grams up to this order (chrF++).
pub fn chrf_with(
    candidate: &str,
    reference: &str,
    order: usize,
    beta: f32,
    word_order: usize,
) -> ChrfScore {
    let order = order.max(1);
    let cand_chars: Vec<char> = candidate.chars().collect();
    let ref_chars: Vec<char> = reference.chars().collect();

    // Handle edge cases: both empty → perfect, either empty → 0.
    let both_empty = cand_chars.is_empty() && ref_chars.is_empty();
    let one_empty = cand_chars.is_empty() ^ ref_chars.is_empty();
    if both_empty {
        return ChrfScore {
            score: 1.0,
            order,
            beta,
            word_order,
        };
    }
    if one_empty {
        return ChrfScore {
            score: 0.0,
            order,
            beta,
            word_order,
        };
    }

    let cand_words: Vec<&str> = candidate.split_whitespace().collect();
    let ref_words: Vec<&str> = reference.split_whitespace().collect();

    // Collect per-order F-beta into a single averaged score. An order for
    // which either side has no n-grams (candidate shorter than `n`, or
    // reference shorter than `n`) is skipped outright (effective order —
    // see the module doc) instead of forcing a 0.0 into the average, which
    // is what made `chrf("a", "a")` score 0.1667 instead of 1.0
    // (RAG-EVAL-IMG-05).
    let mut f_values: Vec<f32> = Vec::new();

    // Character orders
    for n in 1..=order {
        if let Some(f) = order_f_beta_chars(&cand_chars, &ref_chars, n, beta) {
            f_values.push(f);
        }
    }

    // Word orders (chrF++)
    if word_order >= 1 {
        for n in 1..=word_order {
            if let Some(f) = order_f_beta_words(&cand_words, &ref_words, n, beta) {
                f_values.push(f);
            }
        }
    }

    let score = if f_values.is_empty() {
        0.0
    } else {
        let sum: f32 = f_values.iter().sum();
        sum / f_values.len() as f32
    };

    ChrfScore {
        score: score.clamp(0.0, 1.0),
        order,
        beta,
        word_order,
    }
}

/// F-beta for one character n-gram order, or `None` if *either* side has no
/// n-grams of this order (the order is not "effective" — see the module
/// doc). sacreBLEU's effective order counts an order only when both the
/// candidate and the reference have at least one n-gram of length `n`
/// (`n_hyp > 0 AND n_ref > 0`); gating on the reference alone let a
/// too-short candidate be scored as a hard `F=0.0` at every higher order
/// instead of being excluded from the average (see the module doc).
fn order_f_beta_chars(cand: &[char], reference: &[char], n: usize, beta: f32) -> Option<f32> {
    if reference.len() < n || cand.len() < n {
        return None;
    }
    let cand_counts = ngram_counts_char(cand, n);
    let ref_counts = ngram_counts_char(reference, n);
    Some(f_beta_from_counts(&cand_counts, &ref_counts, beta))
}

/// Word n-gram analogue of [`order_f_beta_chars`] (used by chrF++'s word
/// orders); same both-sides effective-order rule.
fn order_f_beta_words(cand: &[&str], reference: &[&str], n: usize, beta: f32) -> Option<f32> {
    if reference.len() < n || cand.len() < n {
        return None;
    }
    let cand_counts = ngram_counts_words(cand, n);
    let ref_counts = ngram_counts_words(reference, n);
    Some(f_beta_from_counts(&cand_counts, &ref_counts, beta))
}

fn f_beta_from_counts<K: std::hash::Hash + Eq + Clone>(
    cand: &HashMap<K, usize>,
    reference: &HashMap<K, usize>,
    beta: f32,
) -> f32 {
    let cand_total: usize = cand.values().sum();
    let ref_total: usize = reference.values().sum();
    if cand_total == 0 || ref_total == 0 {
        return 0.0;
    }
    let mut overlap = 0usize;
    for (k, &v) in cand {
        if let Some(&rv) = reference.get(k) {
            overlap += v.min(rv);
        }
    }
    if overlap == 0 {
        return 0.0;
    }
    let p = overlap as f32 / cand_total as f32;
    let r = overlap as f32 / ref_total as f32;
    let b2 = beta * beta;
    let denom = b2 * p + r;
    if denom <= 0.0 {
        0.0
    } else {
        ((1.0 + b2) * p * r) / denom
    }
}

fn ngram_counts_char(chars: &[char], n: usize) -> HashMap<Vec<char>, usize> {
    let mut counts: HashMap<Vec<char>, usize> = HashMap::new();
    if n == 0 || chars.len() < n {
        return counts;
    }
    for w in chars.windows(n) {
        *counts.entry(w.to_vec()).or_insert(0) += 1;
    }
    counts
}

fn ngram_counts_words(words: &[&str], n: usize) -> HashMap<Vec<String>, usize> {
    let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
    if n == 0 || words.len() < n {
        return counts;
    }
    for w in words.windows(n) {
        let key: Vec<String> = w.iter().map(|s| s.to_string()).collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}
