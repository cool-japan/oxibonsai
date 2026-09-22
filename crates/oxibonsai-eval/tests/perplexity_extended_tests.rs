//! - RAG-EVAL-IMG-28: `PerplexityEvaluator::from_logits` panicked (index out
//!   of bounds) on a caller-supplied out-of-range token id; it now returns
//!   `Result`. A non-finite logit row is also rejected rather than
//!   silently poisoning the perplexity with `NaN`.
//! - RAG-EVAL-IMG-15: `stride` was a dead field — nothing read it.
//!   `PerplexityEvaluator::compute_sliding` now implements the sliding-
//!   window algorithm for sequences longer than a model's context window,
//!   and `corpus_perplexity` adds the properly token-pooled statistic next
//!   to the pre-existing (macro-averaged) `PerplexityResult::mean_ppl`.

use oxibonsai_eval::{EvalError, PerplexityEvaluator};

// ──────────────────────────────────────────────────────────────────────────────
// from_logits — no panics on caller data
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn from_logits_errors_instead_of_panicking_on_out_of_range_token_id() {
    // Exact reproduction from the verified finding: this used to panic with
    // "index out of bounds: the len is 3 but the index is 99".
    let eval = PerplexityEvaluator::new();
    let result = eval.from_logits(&[vec![0.1, 0.2, 0.3]], &[99]);
    assert!(result.is_err(), "expected Err, got {result:?}");
    assert!(matches!(result, Err(EvalError::MetricMismatch { .. })));
}

#[test]
fn from_logits_errors_on_length_mismatch() {
    let eval = PerplexityEvaluator::new();
    let result = eval.from_logits(&[vec![0.1, 0.2]], &[0, 1]);
    assert!(result.is_err());
}

#[test]
fn from_logits_errors_on_empty_logit_vector() {
    let eval = PerplexityEvaluator::new();
    let result = eval.from_logits(&[vec![]], &[0]);
    assert!(result.is_err());
}

#[test]
fn from_logits_errors_on_all_negative_infinity_row() {
    // An all -inf row has a non-finite max logit; without this guard the
    // log-sum-exp would silently produce NaN instead of erroring.
    let eval = PerplexityEvaluator::new();
    let result = eval.from_logits(&[vec![f32::NEG_INFINITY, f32::NEG_INFINITY]], &[0]);
    assert!(result.is_err(), "expected Err, got {result:?}");
}

#[test]
fn from_logits_computes_correct_value_for_valid_input() {
    // Two equal logits ⇒ softmax = [0.5, 0.5] ⇒ log p(token 0) = ln(0.5)
    // ⇒ PPL = exp(-ln(0.5)) = 2.0.
    let eval = PerplexityEvaluator::new();
    let ppl = eval
        .from_logits(&[vec![0.0, 0.0]], &[0])
        .expect("valid input must not error");
    assert!((ppl - 2.0).abs() < 1e-4, "got {ppl}");
}

#[test]
fn from_logits_empty_input_is_infinity_not_an_error() {
    let eval = PerplexityEvaluator::new();
    let ppl = eval
        .from_logits(&[], &[])
        .expect("empty input is not an error");
    assert_eq!(ppl, f32::INFINITY);
}

// ──────────────────────────────────────────────────────────────────────────────
// compute_sliding — the `stride` field is no longer a dead knob
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn compute_sliding_non_overlapping_windows_counts_every_position_once() {
    let eval = PerplexityEvaluator::with_stride(4);
    // Two 4-token windows, no overlap (stride == window length), all
    // "perfect" log-probs (0.0) ⇒ PPL = 1.0.
    let windows = vec![vec![0.0f32; 4], vec![0.0f32; 4]];
    let ppl = eval.compute_sliding(&windows);
    assert!((ppl - 1.0).abs() < 1e-5, "got {ppl}");
}

#[test]
fn compute_sliding_overlap_only_counts_the_new_tail_of_each_window() {
    // window0 covers absolute positions 0..4 with log p = 0 everywhere.
    // window1 (stride=2) covers absolute positions 2..6; its first two
    // entries (absolute 2, 3) deliberately hold implausible values that
    // must be IGNORED (already counted, with more context, by window0);
    // only its last two entries (absolute 4, 5, log p = -1 and -2) are new.
    let eval = PerplexityEvaluator::with_stride(2);
    let windows = vec![vec![0.0, 0.0, 0.0, 0.0], vec![-1000.0, -1000.0, -1.0, -2.0]];
    let ppl = eval.compute_sliding(&windows);
    // count = 6, neg_log_prob_sum = 0*4 + 1.0 + 2.0 = 3.0, mean = 0.5.
    let expected = 0.5f32.exp();
    assert!(
        (ppl - expected).abs() < 1e-4,
        "expected {expected} (only the two new tokens should count), got {ppl}"
    );
}

#[test]
fn compute_sliding_result_actually_depends_on_stride() {
    // Direct proof that `stride` is no longer dead: the same two windows
    // scored with two different strides must give two different results.
    let windows = vec![vec![0.0, 0.0, 0.0, 0.0], vec![-1.0, -1.0, -1.0, -1.0]];

    let no_overlap = PerplexityEvaluator::with_stride(4).compute_sliding(&windows);
    let overlap = PerplexityEvaluator::with_stride(2).compute_sliding(&windows);

    assert!(
        (no_overlap - overlap).abs() > 1e-3,
        "stride=4 gave {no_overlap}, stride=2 gave {overlap} — these must differ"
    );
    // stride=4: all 8 positions counted, mean = (0 + 4*1.0)/8 = 0.5.
    assert!((no_overlap - 0.5f32.exp()).abs() < 1e-4, "got {no_overlap}");
    // stride=2: window1 contributes only its last 2 (of 4) entries;
    // count=6, sum=2.0, mean=1/3.
    assert!(
        (overlap - (1.0f32 / 3.0).exp()).abs() < 1e-4,
        "got {overlap}"
    );
}

#[test]
fn compute_sliding_skips_a_window_fully_covered_by_an_earlier_one() {
    // A short, ragged window entirely inside a range an earlier, longer
    // window already covered must contribute nothing (and must not panic
    // on the resulting empty slice).
    let eval = PerplexityEvaluator::with_stride(1);
    let windows = vec![vec![0.0f32; 10], vec![-99.0, -99.0]];
    let ppl = eval.compute_sliding(&windows);
    assert!((ppl - 1.0).abs() < 1e-5, "got {ppl}");
}

#[test]
fn compute_sliding_empty_windows_is_infinity_not_a_panic() {
    let eval = PerplexityEvaluator::new();
    assert_eq!(eval.compute_sliding(&[]), f32::INFINITY);
    assert_eq!(eval.compute_sliding(&[vec![], vec![]]), f32::INFINITY);
}

// ──────────────────────────────────────────────────────────────────────────────
// corpus_perplexity — token-pooled statistic distinct from mean_ppl
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn corpus_perplexity_differs_from_macro_averaged_mean_ppl() {
    let eval = PerplexityEvaluator::new();
    let sample_a = vec![0.0f32]; // 1 token, PPL = 1.0
    let sample_b = vec![-(2.0f32.ln()); 9]; // 9 tokens, PPL = 2.0 each

    let batch_result = eval.compute_batch(&[sample_a.clone(), sample_b.clone()]);
    assert!(
        (batch_result.mean_ppl - 1.5).abs() < 1e-4,
        "mean_ppl={}",
        batch_result.mean_ppl
    );

    let corpus_ppl = eval.corpus_perplexity(&[sample_a, sample_b]);
    // exp( (0*1 + ln(2)*9) / 10 ) = 2^0.9.
    let expected = 2.0f32.powf(0.9);
    assert!(
        (corpus_ppl - expected).abs() < 1e-3,
        "expected {expected}, got {corpus_ppl}"
    );
    assert!(
        (corpus_ppl - batch_result.mean_ppl).abs() > 0.1,
        "corpus_perplexity must be a genuinely different statistic from mean_ppl"
    );
}

#[test]
fn corpus_perplexity_empty_batch_is_infinity() {
    let eval = PerplexityEvaluator::new();
    assert_eq!(eval.corpus_perplexity(&[]), f32::INFINITY);
}
