//! Property tests for [`oxibonsai_runtime::sampling::Sampler`] covering the
//! RT-21 (zero-mass fallback), RT-22 (first-index greedy tie-break), and
//! RT-25 (seed-0 PRNG degeneracy) fixes, plus the observable contract of the
//! perf-12 top-p optimization.
//!
//! These exercise only the public API (`Sampler::new` / `Sampler::sample`).
//! The exact-set-equality check for perf-12's windowed top-p selection
//! against a full-sort reference lives in `sampling.rs`'s own `#[cfg(test)]`
//! module instead, since it needs the crate-private `truncate_to_top_p`
//! helper — see `top_p_fast_path_matches_reference_full_sort` there.

use std::collections::HashSet;

use oxibonsai_runtime::sampling::{Sampler, SamplingParams};
use proptest::prelude::*;

fn params(temperature: f32, top_k: usize, top_p: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k,
        top_p,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

/// Brute-force reference: index of the maximum value, ties broken toward
/// the first (lowest) index; a `NaN` entry is never chosen over a finite
/// value. Written from scratch (not re-exported from `sampling::argmax`) so
/// this test validates the sampler's *contract*, not merely that its
/// internals still call themselves.
fn reference_first_argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

proptest! {
    /// ACCEPTANCE: across the full (vocab_size, temperature, top_p, top_k,
    /// seed) space, `sample` never errors on well-formed input and never
    /// returns an index outside `0..vocab_size`.
    #[test]
    fn sample_never_panics_and_is_always_in_bounds(
        logits in prop::collection::vec(-50.0f32..50.0, 1..500),
        temperature in 0.0f32..5.0,
        top_p in 0.0f32..=1.0,
        top_k in 0usize..600,
        seed in any::<u64>(),
    ) {
        let mut sampler = Sampler::new(params(temperature, top_k, top_p), seed);
        let token = sampler
            .sample(&logits)
            .expect("sample must not error for well-formed input");
        prop_assert!(
            (token as usize) < logits.len(),
            "token {token} out of bounds for len {}",
            logits.len()
        );
    }

    /// ACCEPTANCE: greedy == first argmax, including for tie-heavy logits
    /// (a small integer range forces frequent exact ties).
    #[test]
    fn greedy_matches_first_argmax(
        logits in prop::collection::vec(0i32..6, 1..200)
            .prop_map(|v| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>()),
        top_k in 0usize..10,
        top_p in 0.0f32..=1.0,
        seed in any::<u64>(),
    ) {
        // top_k / top_p are irrelevant once temperature routes to the greedy
        // (argmax) path, but are varied here too to confirm they really are
        // ignored on that path.
        let mut sampler = Sampler::new(params(0.0, top_k, top_p), seed);
        let token = sampler.sample(&logits).expect("greedy sample must not error");
        prop_assert_eq!(token as usize, reference_first_argmax(&logits));
    }
}

/// RT-25: seeds `0..=64` must all produce pairwise-distinct sample
/// sequences. The literal reachable bug was every seed collapsing to the
/// *same* degenerate stream (in particular, seed 0 always returning
/// whichever token sat first in `probs_buf`, regardless of the logits).
#[test]
fn seeds_zero_to_64_produce_distinct_streams() {
    let mut logits = vec![0.0f32; 4096];
    for (i, v) in logits.iter_mut().enumerate() {
        *v = (i as f32 * 0.0173).sin() * 8.0;
    }

    let streams: Vec<Vec<u32>> = (0u64..=64)
        .map(|seed| {
            let mut sampler = Sampler::new(params(1.0, 0, 0.95), seed);
            (0..32)
                .map(|_| sampler.sample(&logits).expect("sample"))
                .collect()
        })
        .collect();

    let distinct: HashSet<&Vec<u32>> = streams.iter().collect();
    assert_eq!(
        distinct.len(),
        streams.len(),
        "seeds 0..=64 must all produce distinct sample streams (found {} distinct out of {})",
        distinct.len(),
        streams.len()
    );
}

/// RT-25 regression, mirroring the original bug report precisely: seed 0
/// with `top_k=0`/`top_p=1.0` (the natural "no filtering" configuration)
/// must not collapse to always returning whichever token sits first in
/// insertion order.
#[test]
fn seed_zero_is_not_pinned_to_first_token() {
    let mut logits = vec![-20.0f32; 32];
    logits[17] = 5.0;
    let mut sampler = Sampler::new(params(1.0, 0, 1.0), 0);
    let saw_non_zero = (0..64).any(|_| sampler.sample(&logits).expect("sample") != 0);
    assert!(saw_non_zero, "seed 0 must not always return token 0");
}

/// RT-25: `PipelineBuilder::new` documents `seed: 0` as its default, so this
/// exact seed is a live, user-reachable configuration, not a hypothetical.
#[test]
fn seed_zero_produces_a_varied_stream_like_any_other_seed() {
    let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut sampler = Sampler::new(params(1.0, 0, 1.0), 0);
    let mut seen = HashSet::new();
    for _ in 0..200 {
        seen.insert(sampler.sample(&logits).expect("sample"));
    }
    assert!(
        seen.len() > 1,
        "seed 0 must sample from more than a single fixed token over 200 draws, got {seen:?}"
    );
}

/// RT-21: every candidate masked to `-inf` (a fully-constrained decode with
/// no legal continuation) must not panic and must still return an in-range
/// index.
#[test]
fn all_negative_infinity_logits_do_not_panic() {
    let logits = vec![f32::NEG_INFINITY; 64];
    let mut sampler = Sampler::new(params(1.0, 0, 0.9), 7);
    let token = sampler
        .sample(&logits)
        .expect("an all-masked candidate set must still return Ok, never panic");
    assert!((token as usize) < logits.len());
}

/// RT-21: an all-`NaN` logit vector (a fully diverged forward pass) must
/// not panic and must still return an in-range index.
#[test]
fn all_nan_logits_do_not_panic() {
    let logits = vec![f32::NAN; 64];
    let mut sampler = Sampler::new(params(1.0, 0, 0.9), 7);
    let token = sampler
        .sample(&logits)
        .expect("an all-NaN candidate set must still return Ok, never panic");
    assert!((token as usize) < logits.len());
}

/// RT-21: a `NaN` reaching the sampler alongside otherwise well-formed
/// logits (e.g. one unstable element from a partially-diverged forward
/// pass) must not silently propagate — the true best candidate must still
/// be recovered, not merely "some in-range index".
#[test]
fn nan_logit_recovers_true_best_candidate() {
    let mut logits = vec![-1.0f32; 300];
    logits[123] = 99.0;
    logits[5] = f32::NAN;
    logits[288] = f32::NAN;
    let mut sampler = Sampler::new(params(0.8, 0, 0.9), 55);
    let token = sampler
        .sample(&logits)
        .expect("a NaN-poisoned candidate set must still return Ok, never panic");
    assert_eq!(
        token, 123,
        "the zero-mass fallback must recover the true argmax despite the NaN entries"
    );
}

/// RT-22, greedy path with a repeated-history call: `sample_with_history`'s
/// no-penalty fast path must agree with plain `sample` on the same
/// first-index tie-break, since it delegates to the same `sample_core`.
#[test]
fn greedy_tie_break_is_first_index_via_sample_with_history_too() {
    let params = params(0.0, 0, 1.0);
    let mut sampler = Sampler::new(params, 123);
    let logits = vec![2.0f32, 2.0, 2.0, 2.0];
    let token = sampler
        .sample_with_history(&logits, &[])
        .expect("greedy sample_with_history");
    assert_eq!(token, 0, "greedy tie-break must return the FIRST index");
}
