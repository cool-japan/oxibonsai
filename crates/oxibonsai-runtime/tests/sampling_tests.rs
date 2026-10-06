//! Tests for the sampling module.
//!
//! Covers temperature, top-k, top-p, greedy, and distribution verification.

use oxibonsai_runtime::sampling::{Sampler, SamplingParams};

fn greedy_params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    }
}

// ══════════════════════════════════════════════════════════════
// Greedy / Temperature=0
// ══════════════════════════════════════════════════════════════

#[test]
fn temperature_zero_always_returns_argmax() {
    let params = greedy_params();
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![0.1, 0.5, 0.9, 0.3, 0.7];

    for _ in 0..20 {
        let token = sampler.sample(&logits).expect("should sample");
        assert_eq!(
            token, 2,
            "temperature=0 should always pick index 2 (max=0.9)"
        );
    }
}

#[test]
fn temperature_zero_with_negative_logits() {
    let params = greedy_params();
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![-10.0, -5.0, -1.0, -20.0];

    let token = sampler.sample(&logits).expect("should sample");
    assert_eq!(token, 2, "should pick -1.0 as the max");
}

#[test]
fn temperature_zero_with_ties_returns_a_maximum() {
    let params = greedy_params();
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![1.0, 1.0, 1.0];

    let token = sampler.sample(&logits).expect("should sample");
    // argmax picks one of the tied maximum values
    assert!(token < 3, "should return a valid index among tied maxima");
    // The value at the selected index should be the maximum
    assert!(
        (logits[token as usize] - 1.0).abs() < f32::EPSILON,
        "selected token should have maximum value"
    );
}

// ══════════════════════════════════════════════════════════════
// Temperature=1.0 sampling
// ══════════════════════════════════════════════════════════════

#[test]
fn temperature_1_samples_from_distribution() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut sampler = Sampler::new(params, 12345);
    // With a peaked distribution, most samples should be the max
    let logits = vec![0.0, 0.0, 10.0, 0.0, 0.0];

    let mut count_2 = 0;
    for _ in 0..100 {
        let token = sampler.sample(&logits).expect("should sample");
        assert!((token as usize) < 5);
        if token == 2 {
            count_2 += 1;
        }
    }
    assert!(
        count_2 > 50,
        "peak logit should be sampled frequently: count_2={count_2}"
    );
}

// ══════════════════════════════════════════════════════════════
// Top-k filtering
// ══════════════════════════════════════════════════════════════

#[test]
fn top_k_1_always_returns_top_element() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![0.1, 0.9, 0.5, 0.3];

    for _ in 0..20 {
        let token = sampler.sample(&logits).expect("should sample");
        assert_eq!(token, 1, "top_k=1 should always pick max (index 1)");
    }
}

#[test]
fn top_k_reduces_candidate_set() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 2,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut sampler = Sampler::new(params, 42);
    // logits: indices 2 and 4 are the top 2
    let logits = vec![0.0, 0.0, 10.0, 0.0, 9.0];

    for _ in 0..100 {
        let token = sampler.sample(&logits).expect("should sample");
        assert!(
            token == 2 || token == 4,
            "top_k=2 should only sample from top 2, got {token}"
        );
    }
}

// ══════════════════════════════════════════════════════════════
// Top-p filtering
// ══════════════════════════════════════════════════════════════

#[test]
fn top_p_near_zero_returns_top_element() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.01, // very small
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![0.0, 0.0, 10.0, 0.0, 0.0];

    for _ in 0..20 {
        let token = sampler.sample(&logits).expect("should sample");
        assert_eq!(token, 2, "very low top_p should pick the peak");
    }
}

#[test]
fn top_p_1_considers_all_tokens() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut sampler = Sampler::new(params, 42);
    // Uniform logits
    let logits = [1.0; 10];

    let mut seen = [false; 10];
    for _ in 0..500 {
        let token = sampler.sample(&logits).expect("should sample") as usize;
        if token < 10 {
            seen[token] = true;
        }
    }
    let num_seen = seen.iter().filter(|&&s| s).count();
    assert!(
        num_seen >= 5,
        "top_p=1.0 with uniform should hit many tokens: seen={num_seen}"
    );
}

// ══════════════════════════════════════════════════════════════
// Repetition penalty
// ══════════════════════════════════════════════════════════════

#[test]
fn repetition_penalty_1_has_no_effect() {
    let params1 = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let params2 = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut s1 = Sampler::new(params1, 42);
    let mut s2 = Sampler::new(params2, 42);
    let logits = vec![0.1, 0.5, 0.9, 0.3];

    let t1 = s1.sample(&logits).expect("s1");
    let t2 = s2.sample(&logits).expect("s2");
    assert_eq!(t1, t2, "penalty=1.0 should have no effect");
}

// ══════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════

#[test]
fn very_large_logits_no_overflow() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![1000.0, 999.0, 998.0];

    let token = sampler.sample(&logits).expect("should not overflow");
    assert!(token < 3, "should return valid index");
}

#[test]
fn all_negative_logits() {
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![-100.0, -50.0, -200.0];

    let token = sampler.sample(&logits).expect("should handle negatives");
    assert_eq!(token, 1, "should pick -50.0 as the max");
}

#[test]
fn empty_logits_returns_zero() {
    let params = SamplingParams::default();
    let mut sampler = Sampler::new(params, 42);
    let logits: Vec<f32> = vec![];

    let token = sampler.sample(&logits).expect("empty should return 0");
    assert_eq!(token, 0);
}

#[test]
fn single_logit() {
    let params = SamplingParams::default();
    let mut sampler = Sampler::new(params, 42);
    let logits = vec![42.0];

    let token = sampler.sample(&logits).expect("single should work");
    assert_eq!(token, 0);
}

// ══════════════════════════════════════════════════════════════
// Statistical distribution test
// ══════════════════════════════════════════════════════════════

#[test]
fn statistical_distribution_roughly_correct() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 128,
    };
    let mut sampler = Sampler::new(params, 42);

    // Two logits: 0.0 and 0.0 -> equal probability
    let logits = vec![0.0, 0.0];
    let n = 1000;
    let mut counts = [0usize; 2];

    for _ in 0..n {
        let token = sampler.sample(&logits).expect("should sample") as usize;
        if token < 2 {
            counts[token] += 1;
        }
    }

    // Each should be roughly 50% (within reasonable variance)
    let ratio_0 = counts[0] as f64 / n as f64;
    let ratio_1 = counts[1] as f64 / n as f64;
    assert!(
        (ratio_0 - 0.5).abs() < 0.15,
        "token 0 ratio should be ~0.5, got {ratio_0}"
    );
    assert!(
        (ratio_1 - 0.5).abs() < 0.15,
        "token 1 ratio should be ~0.5, got {ratio_1}"
    );
}

#[test]
fn sampling_params_default_values() {
    let params = SamplingParams::default();
    assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 40);
    assert!((params.top_p - 0.9).abs() < f32::EPSILON);
    // The previous
    // `1.1` default silently disqualified every plain `temperature: 0`
    // request from the fused GPU argmax path (`greedy_gpu_eligible`
    // requires `repetition_penalty == 1.0`); the fixed default carries no
    // hidden penalty.
    assert!((params.repetition_penalty - 1.0).abs() < f32::EPSILON);
}

#[test]
fn sampler_params_accessible() {
    let params = SamplingParams {
        temperature: 0.5,
        top_k: 10,
        top_p: 0.8,
        repetition_penalty: 1.2,
        max_tokens: 128,
    };
    let sampler = Sampler::new(params, 42);
    let p = sampler.params();
    assert!((p.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(p.top_k, 10);
}
