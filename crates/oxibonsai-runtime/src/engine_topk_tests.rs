//! Tests for the sampled top-k route (`perf-11`, sampled half) of
//! [`crate::engine_greedy`].
//!
//! Four layers:
//!
//! * the **byte identity** the route ships on: a sampled request decodes
//!   exactly like an independently spelled-out classic loop (prefill, then a
//!   standalone `Sampler::new(params, seed)` draw over every full logit row)
//!   with the route opted in and with it off (the default) — on a synthetic
//!   CPU engine here, on a fused Metal fixture and on the real 1.7B
//!   (`OXI_MODEL`) below — and a candidate draw equals the classic draw
//!   draw-for-draw on tie-heavy rows;
//! * host-only tests of the CPU candidate extraction ([`top_k_candidates`]),
//!   the configuration, and the route's distribution;
//! * Metal tests of the route (on the fused synthetic ternary model,
//!   and on the real 1.7B via `OXI_MODEL`) proving that (a) the GPU
//!   `topk_f32` download over the resident logits equals the CPU extraction
//!   of the same row bit for bit, and (b) a sampled generation through the
//!   GPU candidates is token-for-token identical to the full-row candidate
//!   reference mode, with every step counted where it was served. Every
//!   engine in them that exercises the route opts in explicitly
//!   (`set_sampled_topk(SampledTopKConfig::gpu_candidates())`): the route is
//!   off by default;
//! * the **default** (`default_route`) and the throughput guard that keeps
//!   it from silently regressing (`throughput`): a default-constructed
//!   engine and a default server request take the full-row path and never
//!   move the route's counters, and on the real 1.7B default-sampled decode
//!   keeps at least 0.6x the greedy rate in the same process.
//!
//! The `perf11_*` tests pin what landing the canonical survivor order and
//! the route left unchanged: route on == route off across a 576-case sweep
//! of `top_k` × `top_p` × min-p × temperature × seed, `top_k == 0`
//! realisations (pinned ids captured on the sampler before the canonical
//! order existed), greedy output, and the route's counters once it is
//! opted in.

use std::collections::HashMap;

use super::*;
use crate::engine::InferenceEngine;
use crate::sampling::{Sampler, SamplingParams};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Reference extraction by a full sort (the obviously-correct baseline).
fn full_sort_reference(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut all: Vec<(u32, f32)> = row
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > f32::NEG_INFINITY)
        .map(|(i, v)| (i as u32, *v))
        .collect();
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut ids: Vec<u32> = all.iter().take(k).map(|(i, _)| *i).collect();
    let mut values: Vec<f32> = all.iter().take(k).map(|(_, v)| *v).collect();
    while ids.len() < k {
        ids.push(0);
        values.push(f32::NEG_INFINITY);
    }
    (ids, values)
}

/// The classic full-row decode loop, spelled out independently of the
/// engine's own `generate`: prefill the prompt, then per step draw with a
/// standalone `Sampler::new(params, seed)` over the full logit row, stop on
/// an EOS id, and forward the drawn token at the next position. A sampled
/// request on the default configuration must equal this token for token.
fn classic_sampler_loop(
    engine: &mut InferenceEngine<'_>,
    params: &SamplingParams,
    seed: u64,
    prompt: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    classic_sampler_loop_with_min_p(engine, params, seed, 0.0, prompt, max_tokens)
}

/// [`classic_sampler_loop`] with the standalone sampler's min-p set to
/// `min_p`.
fn classic_sampler_loop_with_min_p(
    engine: &mut InferenceEngine<'_>,
    params: &SamplingParams,
    seed: u64,
    min_p: f32,
    prompt: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    let mut sampler = Sampler::new(params.clone(), seed);
    sampler.set_min_p(min_p);
    let mut row = engine.prefill_from_pos(prompt, 0).expect("classic prefill");
    let mut tokens = Vec::new();
    for pos in prompt.len()..prompt.len() + max_tokens {
        let token = sampler.sample(&row).expect("classic draw");
        if engine.is_eos(token) {
            break;
        }
        tokens.push(token);
        row = engine.decode_step(token, pos).expect("classic decode step");
    }
    tokens
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn top_k_candidates_orders_by_value_then_by_the_lower_id() {
    let row = [0.5f32, 2.0, 2.0, -1.0, 3.0, 2.0];
    let (ids, values) = top_k_candidates(&row, 4);
    assert_eq!(ids, vec![4, 1, 2, 5]);
    assert_eq!(values, vec![3.0, 2.0, 2.0, 2.0]);
}

#[test]
fn top_k_candidates_never_selects_nan_or_negative_infinity_and_pads() {
    let row = [f32::NAN, 1.0, f32::NEG_INFINITY, 0.5];
    let (ids, values) = top_k_candidates(&row, 4);
    assert_eq!(ids, vec![1, 3, 0, 0]);
    assert_eq!(values[..2], [1.0, 0.5]);
    assert!(values[2..].iter().all(|v| *v == f32::NEG_INFINITY));
    let (ids, values) = top_k_candidates(&[], 2);
    assert_eq!(ids, vec![0, 0]);
    assert!(values.iter().all(|v| *v == f32::NEG_INFINITY));
}

#[test]
fn top_k_candidates_equals_a_full_sort_on_random_rows_with_ties() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for trial in 0..40usize {
        let n = 50 + trial * 37;
        let row: Vec<f32> = (0..n)
            .map(|_| {
                // Coarse quantisation forces plenty of exact ties.
                ((xorshift(&mut state) >> 40) % 64) as f32 * 0.25 - 8.0
            })
            .collect();
        for k in [1usize, 5, 20, 64, n + 3] {
            let got = top_k_candidates(&row, k);
            let want = full_sort_reference(&row, k);
            assert_eq!(got.0, want.0, "trial {trial} k {k}: ids");
            assert_eq!(bits(&got.1), bits(&want.1), "trial {trial} k {k}: values");
        }
    }
}

/// The route ships **off**: it is byte-identical to the classic sampler, but
/// its selection kernel costs more per token than the full-row read it
/// replaces. [`SampledTopKConfig::gpu_candidates`] names the mode explicitly,
/// so it is the opt-in; it carries the same candidate count, which clamps to
/// the kernel cap and the vocabulary.
#[test]
fn the_route_is_off_by_default_and_candidates_clamp() {
    let default = SampledTopKConfig::default();
    assert_eq!(default.mode, SampledTopKMode::Off);
    assert_eq!(SampledTopKMode::default(), SampledTopKMode::Off);
    assert_eq!(default.candidates, DEFAULT_SAMPLED_TOPK_CANDIDATES);
    let opt_in = SampledTopKConfig::gpu_candidates();
    assert_ne!(
        default, opt_in,
        "gpu_candidates() is the opt-in, not the default"
    );
    assert_eq!(opt_in.mode, SampledTopKMode::GpuCandidates);
    assert_eq!(opt_in.candidates, DEFAULT_SAMPLED_TOPK_CANDIDATES);
    assert_eq!(opt_in.candidates, default.candidates);
    assert_eq!(opt_in.effective_candidates(248_320), 64);
    assert_eq!(opt_in.effective_candidates(32), 32);
    let huge = SampledTopKConfig {
        candidates: 100_000,
        ..opt_in
    };
    assert_eq!(
        huge.effective_candidates(248_320),
        oxibonsai_kernels::gpu_backend::MAX_RESIDENT_TOPK
    );
    let zero = SampledTopKConfig {
        candidates: 0,
        ..opt_in
    };
    assert_eq!(zero.effective_candidates(248_320), 1);

    let mut engine = InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        SamplingParams::default(),
        1,
    );
    assert_eq!(engine.sampled_topk(), SampledTopKConfig::default());
    assert_eq!(engine.sampled_topk().mode, SampledTopKMode::Off);
    assert!(!engine.sampled_topk_eligible(false));
    engine.set_sampled_topk(opt_in);
    assert_eq!(engine.sampled_topk(), opt_in);
}

#[test]
fn a_synthetic_engine_is_never_eligible_and_counts_nothing() {
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.9,
        repetition_penalty: 1.0,
        max_tokens: 4,
    };
    let mut engine =
        InferenceEngine::new(oxibonsai_core::config::Qwen3Config::tiny_test(), params, 42);
    // Not the fused GPU route: even with the route on, the classic full-row sampler
    // runs, and a non-fused engine is not counted as a "fused full-row
    // request".
    engine.set_sampled_topk(SampledTopKConfig::gpu_candidates());
    assert!(!engine.sampled_topk_eligible(false));
    let _ = engine.generate(&[1, 2, 3], 3).expect("generate");
    assert_eq!(engine.stats().sampled_topk_steps(), 0);
    assert_eq!(engine.stats().sampled_full_row_requests(), 0);
    engine.set_sampled_topk(SampledTopKConfig {
        mode: SampledTopKMode::Off,
        candidates: 8,
    });
    assert_eq!(engine.sampled_topk().mode, SampledTopKMode::Off);
}

/// The shipped default path (CPU engine): a seeded sampled `generate`, the
/// streaming entry point and `generate_with_seed` all equal the
/// independently spelled-out classic loop token for token.
#[test]
fn default_sampled_generation_equals_the_classic_sampler_loop() {
    let config = oxibonsai_core::config::Qwen3Config::tiny_test;
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.9,
        repetition_penalty: 1.0,
        max_tokens: 8,
    };
    let prompt = [11u32, 22, 33];

    let mut engine = InferenceEngine::new(config(), params.clone(), 77);
    let via_generate = engine.generate(&prompt, 8).expect("generate");
    let mut reference = InferenceEngine::new(config(), params.clone(), 77);
    let classic = classic_sampler_loop(&mut reference, &params, 77, &prompt, 8);
    assert_eq!(via_generate, classic);

    let mut streaming = InferenceEngine::new(config(), params.clone(), 77);
    let (tx, rx) = std::sync::mpsc::channel();
    streaming
        .generate_streaming_sync(&prompt, 8, &tx)
        .expect("streaming");
    drop(tx);
    assert_eq!(rx.into_iter().collect::<Vec<u32>>(), classic);

    let mut seeded = InferenceEngine::new(config(), params.clone(), 5);
    let via_seed = seeded
        .generate_with_seed(&prompt, 8, 1234, &params)
        .expect("seeded");
    let mut reference = InferenceEngine::new(config(), params.clone(), 5);
    assert_eq!(
        via_seed,
        classic_sampler_loop(&mut reference, &params, 1234, &prompt, 8)
    );
}

/// The perf-11 acceptance property, draw for draw: a seeded candidate draw
/// over the top-64 sub-row and a seeded classic `Sampler::sample` over the
/// full row pick the same token on every one of 200 tie-heavy 1000-wide rows
/// (0.001-grid values: exact ties everywhere, including across the top-k
/// boundary), for `top_k` 1, 5, 20 and 64 (the candidate count itself), at
/// `top_p` 1.0 and 0.9, with and without min-p (set through the engine's
/// [`InferenceEngine::set_min_p`]). Before the sampler's canonical survivor
/// order this differed on 125 / 7 of 200 rows at `top_k 20`, `top_p` 1.0 /
/// 0.9. Every configuration prints its `N/200 rows differ` count.
#[test]
fn candidate_draws_equal_the_classic_sampler_draw_for_draw() {
    const ROWS: usize = 200;
    for top_k in [1usize, 5, 20, DEFAULT_SAMPLED_TOPK_CANDIDATES] {
        for (top_p, min_p) in [(1.0f32, 0.0f32), (0.9, 0.0), (0.95, 0.05)] {
            let params = SamplingParams {
                temperature: 0.8,
                top_k,
                top_p,
                repetition_penalty: 1.0,
                max_tokens: 16,
            };
            let mut engine = InferenceEngine::new(
                oxibonsai_core::config::Qwen3Config::tiny_test(),
                params.clone(),
                11,
            );
            engine.set_min_p(min_p);
            let mut classic = Sampler::new(params, 11);
            classic.set_min_p(min_p);
            let mut state = 0x1234_5678_9abc_def0u64;
            let mut differ = 0usize;
            for _ in 0..ROWS {
                let row: Vec<f32> = (0..1000)
                    .map(|_| ((xorshift(&mut state) >> 40) % 10_000) as f32 / 1000.0)
                    .collect();
                let a = classic.sample(&row).expect("classic draw");
                let b = engine
                    .sample_row_candidates(&row, DEFAULT_SAMPLED_TOPK_CANDIDATES)
                    .expect("candidate draw");
                if a != b {
                    differ += 1;
                }
            }
            println!(
                "candidate_draws_equal_the_classic_sampler_draw_for_draw: top_k {top_k} \
                 top_p {top_p} min_p {min_p}: {differ}/{ROWS} rows differ"
            );
            assert_eq!(
                differ, 0,
                "top_k {top_k} top_p {top_p} min_p {min_p}: candidate draws must equal \
                 classic draws"
            );
        }
    }
}

#[test]
fn sample_candidates_maps_the_winning_index_back_to_its_token_id() {
    let greedy = SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 4,
    };
    let mut engine =
        InferenceEngine::new(oxibonsai_core::config::Qwen3Config::tiny_test(), greedy, 42);
    // Greedy over the sub-row picks index 0 -> id 900.
    let id = engine
        .sample_candidates(&[900, 12, 7], &[5.0, 1.0, 0.5])
        .expect("sample");
    assert_eq!(id, 900);
    let id = engine
        .sample_row_candidates(&[0.1, 9.0, 0.2, 4.0], 2)
        .expect("sample row");
    assert_eq!(id, 1);
}

/// The route's candidate draw and the classic full-row sampler draw from the
/// **same distribution** — the analytic one: temperature, top-k, softmax,
/// top-p, renormalise — so the route never changes what a sampled request
/// can produce or how likely it is (and, with the sampler's canonical
/// survivor order, not even which token a given seed lands on:
/// `candidate_draws_equal_the_classic_sampler_draw_for_draw`). Both
/// samplers' empirical frequencies over many seeded draws
/// must match that distribution within 5 standard errors, and neither may
/// ever leave its support.
#[test]
fn candidate_draws_follow_the_classic_sampler_distribution() {
    const DRAWS: usize = 12_000;
    const WIDTH: usize = 1000;
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 20,
        top_p: 0.9,
        repetition_penalty: 1.0,
        max_tokens: 1,
    };
    // Distinct logits: a 24-bit uniform draw scaled to [0, 6) plus a tiny
    // strictly increasing offset, so no exact tie can blur the top-k or
    // nucleus boundaries.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let row: Vec<f32> = (0..WIDTH)
        .map(|i| {
            let unit = (xorshift(&mut state) >> 40) as f32 / (1u64 << 24) as f32;
            unit * 6.0 + i as f32 * 1e-5
        })
        .collect();

    // The analytic distribution, in f64.
    let mut ranked: Vec<(usize, f64)> = row
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, f64::from(v)))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    ranked.truncate(params.top_k);
    let temperature = f64::from(params.temperature);
    let max_scaled = ranked[0].1 / temperature;
    let weights: Vec<f64> = ranked
        .iter()
        .map(|(_, v)| (v / temperature - max_scaled).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut expected: Vec<(usize, f64)> = Vec::new();
    let mut cumulative = 0.0f64;
    for ((id, _), weight) in ranked.iter().zip(&weights) {
        let p = weight / total;
        cumulative += p;
        expected.push((*id, p));
        if cumulative > f64::from(params.top_p) {
            // The nucleus boundary must not be a float coin-flip for the
            // f32 samplers: it sits well clear of `top_p`.
            assert!(
                (cumulative - f64::from(params.top_p)).abs() > 1e-4
                    && (cumulative - p - f64::from(params.top_p)).abs() > 1e-4,
                "test row puts the nucleus boundary too close to top_p"
            );
            break;
        }
    }
    let nucleus_mass: f64 = expected.iter().map(|(_, p)| p).sum();
    let expected: HashMap<usize, f64> = expected
        .into_iter()
        .map(|(id, p)| (id, p / nucleus_mass))
        .collect();
    assert!(expected.len() >= 3, "a non-trivial nucleus");

    let mut classic = Sampler::new(params.clone(), 0xC1A5_51C0);
    let mut engine = InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        params,
        0x0717_0717,
    );
    let mut classic_hist: HashMap<usize, usize> = HashMap::new();
    let mut route_hist: HashMap<usize, usize> = HashMap::new();
    for _ in 0..DRAWS {
        let a = classic.sample(&row).expect("classic draw") as usize;
        *classic_hist.entry(a).or_insert(0) += 1;
        let b = engine
            .sample_row_candidates(&row, DEFAULT_SAMPLED_TOPK_CANDIDATES)
            .expect("candidate draw") as usize;
        *route_hist.entry(b).or_insert(0) += 1;
    }
    for (name, hist) in [("classic", &classic_hist), ("candidates", &route_hist)] {
        for id in hist.keys() {
            assert!(
                expected.contains_key(id),
                "{name} drew token {id}, outside the top-k/top-p support"
            );
        }
        for (&id, &p) in &expected {
            let freq = *hist.get(&id).unwrap_or(&0) as f64 / DRAWS as f64;
            let sigma = (p * (1.0 - p) / DRAWS as f64).sqrt();
            assert!(
                (freq - p).abs() <= 5.0 * sigma + 1e-3,
                "{name}: token {id} drawn with frequency {freq:.4}, expected {p:.4}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// perf-11: what the canonical survivor order leaves unchanged
// ─────────────────────────────────────────────────────────────────────────────

/// Width of [`synthetic_row`]'s rows.
const SYNTHETIC_WIDTH: usize = 256;

/// One logit row of a deterministic synthetic "model": a pure function of
/// the decode `step` and the previously drawn token `prev`, so a chain of
/// draws over it behaves like a generation (each drawn token shapes the
/// next row) with no model numerics involved — the pins below hold on every
/// host.
///
/// Values lie on a 0.5-spaced grid of 16 levels in `[-4.0, 3.5]`: with
/// `tie_free` off, every level is shared by ~16 ids (a tie-heavy row, where
/// an index-order walk and a rank-order walk visit tied ids differently);
/// with it on, each id adds its own `id · 1e-4` (< one grid step), so no two
/// values are equal and a probability sort has exactly one result.
fn synthetic_row(step: usize, prev: u32, tie_free: bool) -> Vec<f32> {
    let mut state = 0x5EED_0000_0000_0001u64
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(prev).wrapping_mul(0xD1B5_4A32_D192_ED03);
    if state == 0 {
        state = 1;
    }
    (0..SYNTHETIC_WIDTH)
        .map(|i| {
            let level = ((xorshift(&mut state) >> 40) % 16) as f32;
            let tie_break = if tie_free { i as f32 * 1e-4 } else { 0.0 };
            level * 0.5 - 4.0 + tie_break
        })
        .collect()
}

/// A seeded 24-token sampled "request" over [`synthetic_row`]: a standalone
/// `Sampler::new(params, seed)` (temperature 0.8) draws each token from the
/// row of its step, starting after prompt token `7`.
fn synthetic_request(seed: u64, top_k: usize, top_p: f32, tie_free: bool) -> Vec<u32> {
    const TOKENS: usize = 24;
    let params = SamplingParams {
        temperature: 0.8,
        top_k,
        top_p,
        repetition_penalty: 1.0,
        max_tokens: TOKENS,
    };
    let mut sampler = Sampler::new(params, seed);
    let mut prev = 7u32;
    let mut out = Vec::with_capacity(TOKENS);
    for step in 0..TOKENS {
        let row = synthetic_row(step, prev, tie_free);
        let token = sampler.sample(&row).expect("synthetic draw");
        out.push(token);
        prev = token;
    }
    out
}

/// `top_k == 0` requests keep the index-order sampler path: their seeded
/// realisations are exactly the ones the sampler produced before the
/// canonical survivor order existed.
///
/// The pinned ids were captured by this very harness on the unmodified
/// sampler (seeds 1–4, temperature 0.8, 24 tokens) and are
/// host-independent: [`synthetic_row`] involves no model. Two legs:
///
/// * tie-heavy rows at `top_p = 1.0` — the full row is walked in index
///   order, so every tie is visited by index; a sampler that ranked the row
///   (the canonical order, which `top_k > 0` uses) would visit ties — and
///   draw tokens — differently, which the last assertion proves;
/// * tie-free rows at `top_p = 0.9` — the windowed nucleus selection, whose
///   walk order is then fully determined by the values.
///
/// The Metal twin pins a `top_k == 0` request through the fused engine,
/// route on and off (`metal::perf11_top_k_zero_realisations_are_unchanged_on_the_fused_fixture`).
#[test]
fn perf11_top_k_zero_realisations_are_unchanged() {
    const TIE_HEAVY_TOP_P_1: [[u32; 24]; 4] = [
        [
            153, 197, 72, 224, 48, 112, 207, 107, 162, 120, 248, 59, 47, 103, 218, 152, 42, 53,
            253, 207, 54, 87, 172, 253,
        ],
        [
            251, 94, 101, 10, 211, 5, 179, 24, 105, 218, 121, 230, 57, 1, 191, 219, 117, 70, 12, 7,
            142, 107, 27, 250,
        ],
        [
            72, 21, 235, 250, 73, 163, 93, 93, 141, 212, 35, 101, 233, 222, 88, 191, 39, 192, 55,
            115, 28, 244, 134, 2,
        ],
        [
            38, 18, 98, 238, 65, 128, 131, 130, 47, 6, 60, 13, 134, 154, 188, 156, 195, 182, 219,
            180, 211, 254, 139, 177,
        ],
    ];
    const TIE_FREE_TOP_P_09: [[u32; 24]; 4] = [
        [
            247, 227, 140, 157, 185, 101, 252, 173, 97, 98, 181, 145, 192, 252, 40, 249, 120, 99,
            127, 135, 171, 109, 12, 11,
        ],
        [
            248, 82, 87, 207, 231, 195, 109, 222, 46, 4, 37, 125, 149, 235, 134, 194, 150, 57, 238,
            222, 221, 2, 180, 97,
        ],
        [
            162, 233, 178, 80, 65, 62, 54, 113, 27, 66, 188, 46, 16, 88, 111, 66, 71, 60, 117, 149,
            167, 220, 45, 248,
        ],
        [
            206, 191, 54, 246, 149, 177, 161, 43, 149, 245, 67, 230, 173, 119, 35, 213, 228, 35,
            193, 199, 130, 28, 216, 124,
        ],
    ];

    for (label, top_p, tie_free, pinned) in [
        ("tie-heavy, top_p 1.0", 1.0f32, false, &TIE_HEAVY_TOP_P_1),
        ("tie-free, top_p 0.9", 0.9, true, &TIE_FREE_TOP_P_09),
    ] {
        for (seed, want) in (1..=4u64).zip(pinned.iter()) {
            let got = synthetic_request(seed, 0, top_p, tie_free);
            assert_eq!(
                got.as_slice(),
                want.as_slice(),
                "{label}, seed {seed}: a top_k == 0 realisation changed"
            );
        }
    }

    // The pin discriminates: walking the same tie-heavy rows in rank order
    // (`top_k` = the row width ranks the whole row without cutting it)
    // draws other tokens for the same seeds.
    let ranked_differs = (1..=4u64).any(|seed| {
        synthetic_request(seed, SYNTHETIC_WIDTH, 1.0, false) != TIE_HEAVY_TOP_P_1[seed as usize - 1]
    });
    assert!(
        ranked_differs,
        "an index-order and a rank-order walk of tie-heavy rows must differ somewhere, or this \
         pin could not tell them apart"
    );
}

/// Greedy decoding is untouched by the canonical order and by the route: a
/// temperature-0 draw is the first-index argmax of the raw row whatever
/// `top_k` / `top_p` / min-p say (the greedy branch runs before any
/// ranking), and a temperature-0 generation is the same with the route opted
/// in and off (the default), and equals an independently spelled-out argmax
/// loop. The Metal twin runs the fused route's GPU argmax
/// (`metal::perf11_greedy_is_unchanged_on_the_fused_route`).
#[test]
fn perf11_greedy_is_unchanged() {
    let mut state = 0x0DDB_A11C_0FFE_E000u64;
    for top_k in [0usize, 1, 5, 20, DEFAULT_SAMPLED_TOPK_CANDIDATES] {
        for (top_p, min_p) in [(1.0f32, 0.0f32), (0.9, 0.0), (0.95, 0.5)] {
            let params = SamplingParams {
                temperature: 0.0,
                top_k,
                top_p,
                repetition_penalty: 1.0,
                max_tokens: 1,
            };
            let mut sampler = Sampler::new(params, 3);
            sampler.set_min_p(min_p);
            for _ in 0..50 {
                // 16 levels over 300 ids: the maximum is almost always tied.
                let row: Vec<f32> = (0..300)
                    .map(|_| ((xorshift(&mut state) >> 40) % 16) as f32 * 0.5)
                    .collect();
                assert_eq!(
                    sampler.sample(&row).expect("greedy draw"),
                    argmax_first(&row),
                    "top_k {top_k} top_p {top_p} min_p {min_p}: greedy must be the \
                     first-index argmax"
                );
            }
        }
    }

    let config = oxibonsai_core::config::Qwen3Config::tiny_test;
    let greedy = SamplingParams {
        temperature: 0.0,
        top_k: 20,
        top_p: 0.9,
        repetition_penalty: 1.0,
        max_tokens: 8,
    };
    let prompt = [3u32, 1, 4];
    let mut on = InferenceEngine::new(config(), greedy.clone(), 5);
    on.set_sampled_topk(SampledTopKConfig::gpu_candidates());
    assert_eq!(on.sampled_topk().mode, SampledTopKMode::GpuCandidates);
    let mut off = InferenceEngine::new(config(), greedy.clone(), 6);
    off.set_sampled_topk(SampledTopKConfig {
        mode: SampledTopKMode::Off,
        ..SampledTopKConfig::default()
    });
    let via_on = on.generate(&prompt, 8).expect("route on");
    let via_off = off.generate(&prompt, 8).expect("route off");
    assert_eq!(via_on, via_off);

    let mut reference = InferenceEngine::new(config(), greedy, 7);
    let mut row = reference.prefill_from_pos(&prompt, 0).expect("prefill");
    let mut classic = Vec::new();
    for pos in prompt.len()..prompt.len() + 8 {
        let token = argmax_first(&row);
        if reference.is_eos(token) {
            break;
        }
        classic.push(token);
        row = reference.decode_step(token, pos).expect("decode step");
    }
    assert_eq!(via_on, classic);
}

// ─────────────────────────────────────────────────────────────────────────────
// The default (route off) and the sampled-throughput guard that protects it
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "engine_topk_tests/default_route.rs"]
mod default_route;

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "engine_topk_tests/throughput.rs"]
mod throughput;

// ─────────────────────────────────────────────────────────────────────────────
// Metal: the GPU half
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use half::f16;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
    use oxibonsai_kernels::gpu_backend::{
        metal_resident_logits_download, metal_resident_logits_topk,
    };
    use oxibonsai_kernels::MetalGraph;
    use oxibonsai_testkit::gguf_fixture::Lcg;

    pub(super) const MAX_SEQ: usize = 128;
    const VOCAB: usize = 32;

    fn tq2_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
        let num_blocks = num_weights / 128;
        let mut data = Vec::with_capacity(num_blocks * 34);
        let mut lcg = Lcg::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        for _ in 0..num_blocks {
            for _ in 0..32 {
                data.push(lcg.next_valid_tq2_byte());
            }
            let scale =
                0.25_f32 + ((lcg.next_u64() >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
            data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        }
        data
    }

    fn f32_pattern(n: usize, scale: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            let phase = (i as f32) * 0.013_f32;
            v.extend_from_slice(&(scale * (1.0 + 0.25 * phase.sin())).to_le_bytes());
        }
        v
    }

    /// The shared known-good fused-ternary fixture (h=128, 2 layers, vocab 32,
    /// every projection and the LM head `TQ2_0_g128`) — the same one the
    /// Metal greedy / cross-backend suites use.
    fn fused_ternary_gguf() -> Vec<u8> {
        fused_ternary_gguf_with_vocab(VOCAB)
    }

    /// [`fused_ternary_gguf`] with a `vocab`-token vocabulary (`vocab * 128`
    /// is a whole number of `TQ2_0_g128` blocks for any `vocab`): the
    /// 32-token default is below the shipped sampling default `top_k` of 40,
    /// so a test of the default configuration needs a wider one for the
    /// route to be eligible at all.
    pub(super) fn fused_ternary_gguf_with_vocab(vocab: usize) -> Vec<u8> {
        let (h, inter, layers, nq, nkv, hd) = (128usize, 256usize, 2usize, 4usize, 2usize, 32);
        let mut w = GgufWriter::new();
        w.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen3".into()),
        );
        w.add_metadata(
            "general.name",
            MetadataWriteValue::Str("SampledTopKTest".into()),
        );
        w.add_metadata("qwen3.embedding_length", MetadataWriteValue::U32(h as u32));
        w.add_metadata("qwen3.block_count", MetadataWriteValue::U32(layers as u32));
        w.add_metadata(
            "qwen3.attention.head_count",
            MetadataWriteValue::U32(nq as u32),
        );
        w.add_metadata(
            "qwen3.attention.head_count_kv",
            MetadataWriteValue::U32(nkv as u32),
        );
        w.add_metadata(
            "qwen3.feed_forward_length",
            MetadataWriteValue::U32(inter as u32),
        );
        w.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(vocab as u32));
        w.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
        w.add_metadata(
            "qwen3.attention.layer_norm_rms_epsilon",
            MetadataWriteValue::F32(1e-6),
        );
        w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
        w.add_tensor(TensorEntry {
            name: "token_embd.weight".into(),
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(vocab * h, 0.5),
        });
        w.add_tensor(TensorEntry {
            name: "output_norm.weight".into(),
            shape: vec![h as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(h, 1.0),
        });
        w.add_tensor(TensorEntry {
            name: "output.weight".into(),
            shape: vec![h as u64, vocab as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_pattern(vocab * h, 0xCAFE_BABE),
        });
        for layer in 0..layers {
            let p = format!("blk.{layer}");
            for (name, n) in [
                ("attn_norm", h),
                ("ffn_norm", h),
                ("attn_q_norm", hd),
                ("attn_k_norm", hd),
            ] {
                w.add_tensor(TensorEntry {
                    name: format!("{p}.{name}.weight"),
                    shape: vec![n as u64],
                    tensor_type: TensorType::F32,
                    data: f32_pattern(n, 1.0),
                });
            }
            let seed = 0x1000_0000_u64.wrapping_add((layer as u64) << 16);
            for (i, (name, ne0, ne1)) in [
                ("attn_q", h, nq * hd),
                ("attn_k", h, nkv * hd),
                ("attn_v", h, nkv * hd),
                ("attn_output", nq * hd, h),
                ("ffn_gate", h, inter),
                ("ffn_up", h, inter),
                ("ffn_down", inter, h),
            ]
            .into_iter()
            .enumerate()
            {
                w.add_tensor(TensorEntry {
                    name: format!("{p}.{name}.weight"),
                    shape: vec![ne0 as u64, ne1 as u64],
                    tensor_type: TensorType::TQ2_0_g128,
                    data: tq2_pattern(ne0 * ne1, seed.wrapping_add(i as u64)),
                });
            }
        }
        w.to_bytes().expect("fixture serialises")
    }

    fn sampled_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.9,
            repetition_penalty: 1.0,
            max_tokens: 16,
        }
    }

    /// An engine over `gguf` with the given route configuration.
    fn engine_with<'a>(
        gguf: &'a GgufFile<'a>,
        params: SamplingParams,
        seed: u64,
        max_seq: usize,
        route: SampledTopKConfig,
    ) -> InferenceEngine<'a> {
        let mut engine =
            InferenceEngine::from_gguf(gguf, params, seed, max_seq).expect("engine builds");
        engine.set_sampled_topk(route);
        engine
    }

    /// The full-row candidate reference configuration.
    fn full_row_reference() -> SampledTopKConfig {
        SampledTopKConfig {
            mode: SampledTopKMode::FullRowCandidates,
            ..SampledTopKConfig::gpu_candidates()
        }
    }

    /// Per step, the GPU candidate download over the resident logits must be
    /// bit-identical to the CPU extraction of the same (downloaded) row, and
    /// its winner must be the fused forward's own argmax.
    fn assert_gpu_candidates_match_cpu(engine: &mut InferenceEngine<'_>, prompt: &[u32], k: usize) {
        let vocab = engine.vocab_size();
        let row = engine.prefill_from_pos(prompt, 0).expect("prefill");
        let mut token = crate::engine_greedy::argmax_first(&row);
        for (step, pos) in (prompt.len()..prompt.len() + 8).enumerate() {
            let argmax_id = engine
                .dense_model()
                .expect("dense")
                .forward_greedy_gpu(token, pos)
                .unwrap_or_else(|e| panic!("step {step}: fused forward: {e}"));
            let gpu = metal_resident_logits_topk(vocab, k).expect("gpu top-k");
            let full = metal_resident_logits_download(vocab).expect("resident row");
            let (ids, values) = top_k_candidates(&full, k);
            assert_eq!(gpu.ids, ids, "step {step}: candidate ids");
            assert_eq!(bits(&gpu.values), bits(&values), "step {step}: values");
            assert_eq!(gpu.ids[0], argmax_id, "step {step}: winner vs argmax");
            token = argmax_id;
        }
    }

    #[test]
    fn gpu_candidates_equal_the_cpu_extraction_of_the_resident_row() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let mut engine =
            InferenceEngine::from_gguf(&gguf, sampled_params(), 7, MAX_SEQ).expect("engine");
        assert!(engine.uses_fused_gpu_decode(), "fixture must be fused");
        assert_gpu_candidates_match_cpu(&mut engine, &[1, 2, 3, 4], VOCAB);
    }

    /// An opted-in fused engine: the route is on and serves every decode
    /// step from GPU candidates, and `generate`, the streaming entry point
    /// and `generate_with_seed` are token-for-token the classic full-row
    /// sampler — and the route switched off.
    #[test]
    fn opt_in_fused_engine_samples_exactly_like_the_classic_sampler() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];
        let params = sampled_params();
        let opt_in = SampledTopKConfig::gpu_candidates();

        let mut engine = engine_with(&gguf, params.clone(), 11, MAX_SEQ, opt_in);
        assert!(engine.uses_fused_gpu_decode(), "fixture must be fused");
        assert_eq!(engine.sampled_topk().mode, SampledTopKMode::GpuCandidates);
        assert!(engine.sampled_topk_eligible(false));
        let via_engine = engine.generate(&prompt, 16).expect("generate");
        assert_eq!(engine.stats().sampled_full_row_requests(), 0);
        assert_eq!(engine.stats().sampled_topk_steps(), 15);
        assert_eq!(engine.stats().sampled_topk_full_row_steps(), 0);

        let mut reference =
            InferenceEngine::from_gguf(&gguf, params.clone(), 11, MAX_SEQ).expect("engine");
        let classic = classic_sampler_loop(&mut reference, &params, 11, &prompt, 16);
        assert_eq!(classic.len(), 16, "no EOS inside the fixture's vocabulary");
        assert_eq!(
            via_engine, classic,
            "the opted-in fused engine must sample exactly like the classic sampler"
        );

        let mut off = engine_with(
            &gguf,
            params.clone(),
            11,
            MAX_SEQ,
            SampledTopKConfig {
                mode: SampledTopKMode::Off,
                ..SampledTopKConfig::default()
            },
        );
        assert_eq!(off.generate(&prompt, 16).expect("route off"), classic);
        assert_eq!(off.stats().sampled_full_row_requests(), 1);

        let mut streaming = engine_with(&gguf, params.clone(), 11, MAX_SEQ, opt_in);
        let (tx, rx) = std::sync::mpsc::channel();
        streaming
            .generate_streaming_sync(&prompt, 16, &tx)
            .expect("streaming");
        drop(tx);
        assert_eq!(rx.into_iter().collect::<Vec<u32>>(), classic);
        assert_eq!(streaming.stats().sampled_topk_steps(), 15);

        let mut seeded = engine_with(&gguf, params.clone(), 999, MAX_SEQ, opt_in);
        let via_seed = seeded
            .generate_with_seed(&prompt, 16, 23, &params)
            .expect("seeded");
        let mut reference =
            InferenceEngine::from_gguf(&gguf, params.clone(), 999, MAX_SEQ).expect("engine");
        assert_eq!(
            via_seed,
            classic_sampler_loop(&mut reference, &params, 23, &prompt, 16)
        );
    }

    /// The route: sampled output through the GPU candidates is
    /// byte-identical to the full-row candidate reference, each engine counts
    /// where its steps were served, and the whole output — the first token
    /// drawn from the prefill row by the classic sampler on both paths —
    /// equals the route switched off.
    #[test]
    fn opt_in_gpu_topk_route_is_byte_identical_to_the_full_row_candidate_reference() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];
        let opt_in = SampledTopKConfig::gpu_candidates();

        let mut gpu = engine_with(&gguf, sampled_params(), 11, MAX_SEQ, opt_in);
        assert!(gpu.sampled_topk_eligible(false));
        let via_gpu = gpu.generate(&prompt, 16).expect("gpu route");

        let mut reference = engine_with(&gguf, sampled_params(), 11, MAX_SEQ, full_row_reference());
        assert!(reference.sampled_topk_eligible(false));
        let via_full_row = reference.generate(&prompt, 16).expect("full-row route");

        assert_eq!(via_gpu.len(), 16);
        assert_eq!(
            via_gpu, via_full_row,
            "GPU top-k candidates must reproduce the full-row candidate draw exactly"
        );
        assert_eq!(gpu.stats().sampled_topk_steps(), 15);
        assert_eq!(gpu.stats().sampled_topk_full_row_steps(), 0);
        assert_eq!(reference.stats().sampled_topk_steps(), 0);
        assert_eq!(reference.stats().sampled_topk_full_row_steps(), 15);

        // The streaming entry point takes the same route.
        let mut streaming = engine_with(&gguf, sampled_params(), 11, MAX_SEQ, opt_in);
        let (tx, rx) = std::sync::mpsc::channel();
        streaming
            .generate_streaming_sync(&prompt, 16, &tx)
            .expect("streaming");
        drop(tx);
        assert_eq!(rx.into_iter().collect::<Vec<u32>>(), via_gpu);
        assert_eq!(streaming.stats().sampled_topk_steps(), 15);

        // Step 0 is the classic draw over the prefill row on both paths, and
        // with the canonical survivor order every later step agrees too. (The
        // route-off reference is spelled out; it is also what an engine does
        // without any configuration.)
        let mut route_off = engine_with(
            &gguf,
            sampled_params(),
            11,
            MAX_SEQ,
            SampledTopKConfig {
                mode: SampledTopKMode::Off,
                ..opt_in
            },
        );
        let via_route_off = route_off.generate(&prompt, 16).expect("route off");
        assert_eq!(via_route_off.first(), via_gpu.first());
        assert_eq!(via_route_off, via_gpu);
    }

    /// `top_k` at or above the vocabulary is not a real top-k selection on
    /// the full row, so the route refuses it (counted); one below it
    /// is served.
    #[test]
    fn a_top_k_at_or_above_the_vocabulary_is_not_eligible() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let opt_in = SampledTopKConfig::gpu_candidates();

        for top_k in [VOCAB, VOCAB + 5] {
            let params = SamplingParams {
                top_k,
                ..sampled_params()
            };
            let mut engine = engine_with(&gguf, params, 3, MAX_SEQ, opt_in);
            assert_eq!(engine.vocab_size(), VOCAB);
            assert!(!engine.sampled_topk_eligible(false), "top_k {top_k}");
            let _ = engine.generate(&[1, 2, 3], 4).expect("generate");
            assert_eq!(engine.stats().sampled_full_row_requests(), 1);
            assert_eq!(engine.stats().sampled_topk_steps(), 0);
        }

        let below = SamplingParams {
            top_k: VOCAB - 1,
            ..sampled_params()
        };
        let mut engine = engine_with(&gguf, below, 3, MAX_SEQ, opt_in);
        assert!(engine.sampled_topk_eligible(false));
        let tokens = engine.generate(&[1, 2, 3], 4).expect("generate");
        assert_eq!(tokens.len(), 4);
        assert_eq!(engine.stats().sampled_topk_steps(), 3);
        assert_eq!(engine.stats().sampled_full_row_requests(), 0);
    }

    /// A request the route cannot serve exactly — a penalty (it must
    /// see every logit), `top_k = 0` (top-p over the whole vocabulary), the
    /// route switched off — decodes the full row and is counted as such.
    #[test]
    fn ineligible_sampled_requests_are_counted_as_full_row() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let opt_in = SampledTopKConfig::gpu_candidates();

        let mut penalised = engine_with(&gguf, sampled_params(), 3, MAX_SEQ, opt_in);
        penalised.set_penalties(crate::sampling::PenaltyParams::new(0.5, 0.0));
        assert!(!penalised.sampled_topk_eligible(false));
        let _ = penalised.generate(&[1, 2, 3], 4).expect("generate");
        assert_eq!(penalised.stats().sampled_full_row_requests(), 1);
        assert_eq!(penalised.stats().sampled_topk_steps(), 0);

        let open_vocab = SamplingParams {
            top_k: 0,
            ..sampled_params()
        };
        let mut unbounded = engine_with(&gguf, open_vocab, 3, MAX_SEQ, opt_in);
        assert!(!unbounded.sampled_topk_eligible(false));
        let _ = unbounded.generate(&[1, 2, 3], 4).expect("generate");
        assert_eq!(unbounded.stats().sampled_full_row_requests(), 1);

        let mut off = engine_with(
            &gguf,
            sampled_params(),
            3,
            MAX_SEQ,
            SampledTopKConfig {
                mode: SampledTopKMode::Off,
                ..opt_in
            },
        );
        assert!(!off.sampled_topk_eligible(false));
        let _ = off.generate(&[1, 2, 3], 4).expect("generate");
        assert_eq!(off.stats().sampled_full_row_requests(), 1);
        assert_eq!(off.stats().sampled_topk_steps(), 0);

        // Greedy never counts as a sampled request, and logprobs always need
        // the full row.
        let greedy = SamplingParams {
            temperature: 0.0,
            ..sampled_params()
        };
        let engine = engine_with(&gguf, greedy, 3, MAX_SEQ, opt_in);
        assert!(!engine.sampled_topk_eligible(false));
        let sampled = engine_with(&gguf, sampled_params(), 3, MAX_SEQ, opt_in);
        assert!(sampled.sampled_topk_eligible(false));
        assert!(!sampled.sampled_topk_eligible(true));
    }

    /// perf-11 metrics: with an `InferenceMetrics` attached, the three
    /// Prometheus counters move exactly with their `EngineStats` twins on the
    /// fused fixture — a full-row request (route off), GPU-candidate steps
    /// (route on) and full-row steps (the full-row reference mode) — and the
    /// rendered exposition carries the live values.
    #[test]
    fn sampled_route_counters_reach_the_prometheus_exposition() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];
        let metrics = std::sync::Arc::new(crate::metrics::InferenceMetrics::new());

        let off = SampledTopKConfig {
            mode: SampledTopKMode::Off,
            ..SampledTopKConfig::gpu_candidates()
        };
        let mut full_row_request = engine_with(&gguf, sampled_params(), 11, MAX_SEQ, off);
        full_row_request.set_metrics(std::sync::Arc::clone(&metrics));
        let _ = full_row_request.generate(&prompt, 8).expect("route off");
        assert_eq!(full_row_request.stats().sampled_full_row_requests(), 1);
        assert_eq!(metrics.sampled_full_row_requests_total.get(), 1);

        let mut gpu = engine_with(
            &gguf,
            sampled_params(),
            11,
            MAX_SEQ,
            SampledTopKConfig::gpu_candidates(),
        );
        gpu.set_metrics(std::sync::Arc::clone(&metrics));
        let tokens = gpu.generate(&prompt, 8).expect("route on");
        assert_eq!(tokens.len(), 8);
        assert_eq!(gpu.stats().sampled_topk_steps(), 7);
        assert_eq!(
            metrics.sampled_topk_steps_total.get(),
            gpu.stats().sampled_topk_steps()
        );

        let mut reference = engine_with(&gguf, sampled_params(), 11, MAX_SEQ, full_row_reference());
        reference.set_metrics(std::sync::Arc::clone(&metrics));
        let _ = reference.generate(&prompt, 8).expect("full-row reference");
        assert_eq!(reference.stats().sampled_topk_full_row_steps(), 7);
        assert_eq!(
            metrics.sampled_topk_full_row_steps_total.get(),
            reference.stats().sampled_topk_full_row_steps()
        );

        let exposition = metrics.render_prometheus();
        for line in [
            "oxibonsai_sampled_full_row_requests_total 1",
            "oxibonsai_sampled_topk_steps_total 7",
            "oxibonsai_sampled_topk_full_row_steps_total 7",
        ] {
            assert!(exposition.contains(line), "missing `{line}`:\n{exposition}");
        }
    }

    /// The route switched off, spelled out (it is also the default).
    fn route_off() -> SampledTopKConfig {
        SampledTopKConfig {
            mode: SampledTopKMode::Off,
            ..SampledTopKConfig::default()
        }
    }

    /// The `perf-11` acceptance sweep on the fused fixture: with the canonical
    /// survivor order, a seeded sampled request with the route opted in is
    /// token-for-token the same request with the route switched off (the
    /// default), for every `top_k` in {1, 5, 20, 64} × `top_p`
    /// in {1.0, 0.9, 0.95} × min-p in {0.0, 0.05} × temperature in
    /// {0.3, 0.8, 1.2} × seeds 0..8 (576 cases).
    ///
    /// Not vacuous: every eligible case (`top_k` below the 32-token
    /// vocabulary) is served from GPU candidates at every decode step, the
    /// `top_k` 64 cases (at/above the vocabulary) are counted as full-row
    /// requests, and min-p 0.05 changes the route-off realisation of at least
    /// one case (so the min-p axis really reaches the sampler — through
    /// [`InferenceEngine::set_min_p`] and `generate_with_seed`, which carries
    /// it into its per-call sampler).
    #[test]
    fn perf11_route_on_equals_route_off_sweep() {
        const TOKENS: usize = 12;
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];

        let mut on = engine_with(
            &gguf,
            sampled_params(),
            0,
            MAX_SEQ,
            SampledTopKConfig::gpu_candidates(),
        );
        assert!(on.uses_fused_gpu_decode(), "fixture must be fused");
        assert_eq!(on.sampled_topk(), SampledTopKConfig::gpu_candidates());
        assert_eq!(on.sampled_topk().mode, SampledTopKMode::GpuCandidates);
        let mut off = engine_with(&gguf, sampled_params(), 0, MAX_SEQ, route_off());

        let started = std::time::Instant::now();
        let mut cases = 0usize;
        let mut routed_cases = 0usize;
        let mut min_p_changed = 0usize;
        for top_k in [1usize, 5, 20, 64] {
            for top_p in [1.0f32, 0.9, 0.95] {
                for temperature in [0.3f32, 0.8, 1.2] {
                    let params = SamplingParams {
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty: 1.0,
                        max_tokens: TOKENS,
                    };
                    for seed in 0..8u64 {
                        let mut route_off_by_min_p = Vec::with_capacity(2);
                        for min_p in [0.0f32, 0.05] {
                            let case = format!(
                                "top_k {top_k} top_p {top_p} min_p {min_p} t {temperature} \
                                 seed {seed}"
                            );
                            on.set_min_p(min_p);
                            off.set_min_p(min_p);
                            let topk_steps = on.stats().sampled_topk_steps();
                            let full_row_steps = on.stats().sampled_topk_full_row_steps();
                            let full_row_requests = on.stats().sampled_full_row_requests();

                            on.reset();
                            let via_on = on
                                .generate_with_seed(&prompt, TOKENS, seed, &params)
                                .expect("route on");
                            off.reset();
                            let via_off = off
                                .generate_with_seed(&prompt, TOKENS, seed, &params)
                                .expect("route off");
                            assert_eq!(via_on.len(), TOKENS, "{case}: no EOS in the fixture");
                            assert_eq!(via_on, via_off, "{case}: route on != route off");

                            let served = on.stats().sampled_topk_steps() - topk_steps;
                            let fell_back =
                                on.stats().sampled_topk_full_row_steps() - full_row_steps;
                            let full_row =
                                on.stats().sampled_full_row_requests() - full_row_requests;
                            if top_k < VOCAB {
                                assert_eq!(served, (TOKENS - 1) as u64, "{case}: GPU steps");
                                assert_eq!(fell_back, 0, "{case}: fallback steps");
                                assert_eq!(full_row, 0, "{case}: full-row requests");
                                routed_cases += 1;
                            } else {
                                assert_eq!(served, 0, "{case}: GPU steps");
                                assert_eq!(full_row, 1, "{case}: full-row requests");
                            }
                            route_off_by_min_p.push(via_off);
                            cases += 1;
                        }
                        if route_off_by_min_p[0] != route_off_by_min_p[1] {
                            min_p_changed += 1;
                        }
                    }
                }
            }
        }
        on.set_min_p(0.0);
        off.set_min_p(0.0);
        println!(
            "perf11_route_on_equals_route_off_sweep: {cases} cases identical ({routed_cases} \
             served from GPU candidates), min-p 0.05 changed {min_p_changed} of {} \
             realisations, {:.1} s",
            cases / 2,
            started.elapsed().as_secs_f64()
        );
        assert_eq!(cases, 576);
        assert_eq!(routed_cases, 432);
        assert!(
            min_p_changed > 0,
            "min-p 0.05 never changed a route-off realisation: the min-p axis proves nothing"
        );
    }

    /// The fused-engine twin of the host-level
    /// `perf11_top_k_zero_realisations_are_unchanged`: a seeded `top_k == 0`
    /// sampled request (temperature 0.8, seeds 1–4, 24 tokens, `top_p` 1.0
    /// and 0.9) produces exactly the ids the fused engine produced before the
    /// canonical survivor order existed (captured on that unmodified
    /// engine) — with the route opted in (`top_k == 0` is never eligible, so
    /// it decodes the full row and is counted as such) and with it off (the
    /// default), and equal to an independently spelled-out classic loop.
    ///
    /// The pin is of this fixture's Metal logits: a change to the fused
    /// kernels' arithmetic that moves a draw across a probability boundary
    /// would need a re-capture (the classic-loop assertion stays valid
    /// either way).
    #[test]
    fn perf11_top_k_zero_realisations_are_unchanged_on_the_fused_fixture() {
        const TOP_P_1: [[u32; 24]; 4] = [
            [
                21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 24, 21, 17, 21, 21, 21, 21, 21, 28, 21, 21,
                21, 21, 29,
            ],
            [
                28, 21, 21, 21, 21, 11, 21, 17, 21, 21, 21, 21, 21, 5, 21, 21, 21, 21, 11, 11, 21,
                21, 11, 28,
            ],
            [
                21, 13, 28, 29, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 17,
                24, 21, 11,
            ],
            [
                21, 21, 21, 24, 21, 21, 21, 21, 21, 11, 21, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                29, 21, 21,
            ],
        ];
        const TOP_P_09: [[u32; 24]; 4] = [
            [
                21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 11, 21, 21, 21, 21, 21, 21, 21, 11, 21, 21,
                21, 21, 11,
            ],
            [
                11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                21, 21, 11,
            ],
            [
                21, 21, 11, 11, 21, 21, 21, 21, 21, 21, 21, 21, 11, 11, 21, 21, 21, 21, 21, 21, 21,
                11, 21, 21,
            ],
            [
                21, 21, 21, 11, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 11, 21, 21,
                11, 21, 21,
            ],
        ];
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];

        for (top_p, pinned) in [(1.0f32, &TOP_P_1), (0.9, &TOP_P_09)] {
            let params = SamplingParams {
                temperature: 0.8,
                top_k: 0,
                top_p,
                repetition_penalty: 1.0,
                max_tokens: 24,
            };
            let mut on = engine_with(
                &gguf,
                params.clone(),
                0,
                MAX_SEQ,
                SampledTopKConfig::gpu_candidates(),
            );
            assert_eq!(on.sampled_topk().mode, SampledTopKMode::GpuCandidates);
            assert!(
                !on.sampled_topk_eligible(false),
                "top_k 0 is never eligible"
            );
            let mut off = engine_with(&gguf, params.clone(), 0, MAX_SEQ, route_off());
            for (seed, want) in (1..=4u64).zip(pinned.iter()) {
                on.reset();
                let via_on = on
                    .generate_with_seed(&prompt, 24, seed, &params)
                    .expect("route on");
                off.reset();
                let via_off = off
                    .generate_with_seed(&prompt, 24, seed, &params)
                    .expect("route off");
                assert_eq!(
                    via_on.as_slice(),
                    want.as_slice(),
                    "top_p {top_p} seed {seed}"
                );
                assert_eq!(
                    via_off.as_slice(),
                    want.as_slice(),
                    "top_p {top_p} seed {seed}"
                );
                let mut reference =
                    InferenceEngine::from_gguf(&gguf, params.clone(), 0, MAX_SEQ).expect("engine");
                let classic = classic_sampler_loop(&mut reference, &params, seed, &prompt, 24);
                assert_eq!(
                    classic.as_slice(),
                    want.as_slice(),
                    "top_p {top_p} seed {seed}"
                );
            }
            assert_eq!(on.stats().sampled_full_row_requests(), 4);
            assert_eq!(on.stats().sampled_topk_steps(), 0);
        }
    }

    /// Greedy output on the fused route is unchanged by the canonical order
    /// and by the route: a temperature-0 request takes the GPU
    /// argmax with the route opted in and off alike — through `generate`, the
    /// streaming entry point, `generate_with_seed` and the explicit
    /// `generate_greedy_gpu` — and equals an independently spelled-out loop
    /// taking the first-index argmax of every full logit row; a penalised
    /// greedy request (full row, penalty before the argmax) is also the same
    /// either way. No sampled-route counter moves.
    #[test]
    fn perf11_greedy_is_unchanged_on_the_fused_route() {
        const TOKENS: usize = 16;
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];
        let greedy = SamplingParams {
            temperature: 0.0,
            top_k: 20,
            top_p: 0.9,
            repetition_penalty: 1.0,
            max_tokens: TOKENS,
        };

        let mut reference =
            InferenceEngine::from_gguf(&gguf, greedy.clone(), 3, MAX_SEQ).expect("engine");
        let mut row = reference.prefill_from_pos(&prompt, 0).expect("prefill");
        let mut classic = Vec::new();
        for pos in prompt.len()..prompt.len() + TOKENS {
            let token = argmax_first(&row);
            if reference.is_eos(token) {
                break;
            }
            classic.push(token);
            row = reference.decode_step(token, pos).expect("decode step");
        }
        assert_eq!(
            classic.len(),
            TOKENS,
            "no EOS inside the fixture's vocabulary"
        );

        let mut on = engine_with(
            &gguf,
            greedy.clone(),
            3,
            MAX_SEQ,
            SampledTopKConfig::gpu_candidates(),
        );
        assert_eq!(on.sampled_topk().mode, SampledTopKMode::GpuCandidates);
        let mut off = engine_with(&gguf, greedy.clone(), 3, MAX_SEQ, route_off());
        for (name, engine) in [("route on", &mut on), ("route off", &mut off)] {
            assert!(
                engine.greedy_gpu_eligible(false),
                "{name}: GPU argmax route"
            );
            engine.reset();
            let via_generate = engine.generate(&prompt, TOKENS).expect("generate");
            assert_eq!(via_generate, classic, "{name}: generate");

            engine.reset();
            let (tx, rx) = std::sync::mpsc::channel();
            engine
                .generate_streaming_sync(&prompt, TOKENS, &tx)
                .expect("streaming");
            drop(tx);
            assert_eq!(
                rx.into_iter().collect::<Vec<u32>>(),
                classic,
                "{name}: streaming"
            );

            engine.reset();
            let via_seed = engine
                .generate_with_seed(&prompt, TOKENS, 99, &greedy)
                .expect("seeded");
            assert_eq!(via_seed, classic, "{name}: generate_with_seed");

            engine.reset();
            let via_explicit = engine
                .generate_greedy_gpu(&prompt, TOKENS)
                .expect("generate_greedy_gpu");
            assert_eq!(via_explicit, classic, "{name}: generate_greedy_gpu");

            assert_eq!(engine.stats().sampled_topk_steps(), 0, "{name}");
            assert_eq!(engine.stats().sampled_topk_full_row_steps(), 0, "{name}");
            assert_eq!(engine.stats().sampled_full_row_requests(), 0, "{name}");
        }

        let penalised = SamplingParams {
            repetition_penalty: 1.3,
            ..greedy
        };
        on.reset();
        off.reset();
        let via_on = on
            .generate_with_params(&prompt, TOKENS, &penalised)
            .expect("penalised, route on");
        let via_off = off
            .generate_with_params(&prompt, TOKENS, &penalised)
            .expect("penalised, route off");
        assert_eq!(via_on, via_off, "penalised greedy");
        assert_eq!(on.stats().sampled_full_row_requests(), 0);
    }

    /// With the route opted in, the three sampled-route Prometheus counters
    /// move exactly with their `EngineStats` twins on the fused fixture: an
    /// eligible sampled request is served from GPU candidates at every decode
    /// step, a `top_k == 0` request is one full-row request, and decode steps
    /// forced off the GPU mid-request (the MET-05 CPU replay, through the
    /// `force_cpu_decode_after` seam) are full-row steps.
    #[test]
    fn perf11_route_counters_move_when_the_route_is_opted_in() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];
        let metrics = std::sync::Arc::new(crate::metrics::InferenceMetrics::new());

        let mut engine = engine_with(
            &gguf,
            sampled_params(),
            11,
            MAX_SEQ,
            SampledTopKConfig::gpu_candidates(),
        );
        assert_eq!(engine.sampled_topk(), SampledTopKConfig::gpu_candidates());
        engine.set_metrics(std::sync::Arc::clone(&metrics));

        let tokens = engine.generate(&prompt, 8).expect("eligible request");
        assert_eq!(tokens.len(), 8);
        assert_eq!(metrics.sampled_topk_steps_total.get(), 7);
        assert_eq!(metrics.sampled_topk_full_row_steps_total.get(), 0);
        assert_eq!(metrics.sampled_full_row_requests_total.get(), 0);

        let open_vocabulary = SamplingParams {
            top_k: 0,
            ..sampled_params()
        };
        engine.reset();
        let tokens = engine
            .generate_with_params(&prompt, 8, &open_vocabulary)
            .expect("top_k 0 request");
        assert_eq!(tokens.len(), 8);
        assert_eq!(metrics.sampled_full_row_requests_total.get(), 1);
        assert_eq!(metrics.sampled_topk_steps_total.get(), 7);

        // Steps 1 and 2 on the GPU, steps 3..=7 forced onto the CPU replay.
        engine.set_speculative(crate::engine_control::SpeculativeConfig {
            force_cpu_decode_after: Some(3),
            ..crate::engine_control::SpeculativeConfig::default()
        });
        engine.reset();
        let tokens = engine.generate(&prompt, 8).expect("forced CPU tail");
        assert_eq!(tokens.len(), 8);
        assert_eq!(metrics.sampled_topk_steps_total.get(), 9);
        assert_eq!(metrics.sampled_topk_full_row_steps_total.get(), 5);
        assert_eq!(metrics.sampled_full_row_requests_total.get(), 1);

        let stats = engine.stats();
        assert_eq!(
            stats.sampled_topk_steps(),
            metrics.sampled_topk_steps_total.get()
        );
        assert_eq!(
            stats.sampled_topk_full_row_steps(),
            metrics.sampled_topk_full_row_steps_total.get()
        );
        assert_eq!(
            stats.sampled_full_row_requests(),
            metrics.sampled_full_row_requests_total.get()
        );
        let exposition = metrics.render_prometheus();
        for line in [
            "oxibonsai_sampled_topk_steps_total 9",
            "oxibonsai_sampled_topk_full_row_steps_total 5",
            "oxibonsai_sampled_full_row_requests_total 1",
        ] {
            assert!(exposition.contains(line), "missing `{line}`:\n{exposition}");
        }
    }

    /// Map `OXI_MODEL` (the real ternary 1.7B) or report the skip.
    fn real_model(test: &str) -> Option<memmap2::Mmap> {
        use oxibonsai_testkit::capability::{record_skipped, Capability};
        let Some(path) = std::env::var_os("OXI_MODEL").filter(|p| !p.is_empty()) else {
            eprintln!(
                "capability report: {test} SKIPPED — OXI_MODEL is not set (point it at \
                 models/Ternary-Bonsai-1.7B.gguf)"
            );
            record_skipped(Capability::LegacyModels, test);
            return None;
        };
        Some(
            oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&path))
                .expect("OXI_MODEL maps"),
        )
    }

    /// `The capital of Japan is` in the Qwen3 vocabulary.
    const REAL_PROMPT: [u32; 5] = [785, 6722, 315, 6323, 374];

    /// The route on the real ternary 1.7B (`OXI_MODEL`), where the
    /// vocabulary (151 936) is far above the candidate count, so the GPU
    /// kernel really does select 64 of 151 936: the GPU download equals the
    /// CPU extraction, and generation equals the full-row candidate
    /// reference.
    #[test]
    fn real_model_gpu_topk_route_matches_the_full_row_candidate_reference() {
        use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
        const TEST: &str = "oxibonsai-runtime::lib::\
             real_model_gpu_topk_route_matches_the_full_row_candidate_reference";
        let Some(mmap) = real_model(TEST) else {
            return;
        };
        let gate_start = std::time::Instant::now();
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let gguf = GgufFile::parse(&mmap).expect("OXI_MODEL parses");
        let opt_in = SampledTopKConfig::gpu_candidates();

        let mut probe = engine_with(&gguf, sampled_params(), 5, 512, opt_in);
        if !probe.uses_fused_gpu_decode() {
            eprintln!(
                "capability report: {TEST} SKIPPED — OXI_MODEL is not on the fused GPU route; \
                 the sampled top-k route does not apply to it"
            );
            record_skipped(Capability::LegacyModels, TEST);
            return;
        }
        let k = probe
            .sampled_topk()
            .effective_candidates(probe.vocab_size());
        assert_eq!(k, DEFAULT_SAMPLED_TOPK_CANDIDATES);
        assert_gpu_candidates_match_cpu(&mut probe, &REAL_PROMPT, k);
        drop(probe);

        let mut gpu = engine_with(&gguf, sampled_params(), 5, 512, opt_in);
        let via_gpu = gpu.generate(&REAL_PROMPT, 24).expect("gpu route");
        let mut reference = engine_with(&gguf, sampled_params(), 5, 512, full_row_reference());
        let via_full_row = reference
            .generate(&REAL_PROMPT, 24)
            .expect("full-row route");
        eprintln!(
            "real-model sampled top-k: gpu {:?} | full-row {:?} | gpu steps {} / full-row \
             steps {}",
            via_gpu,
            via_full_row,
            gpu.stats().sampled_topk_steps(),
            reference.stats().sampled_topk_full_row_steps()
        );
        assert_eq!(via_gpu, via_full_row);
        assert!(gpu.stats().sampled_topk_steps() > 0);
        assert_eq!(gpu.stats().sampled_topk_full_row_steps(), 0);
        eprintln!(
            "real-model sampled top-k: gpu == full-row, {} GPU-candidate steps / {} full-row \
             steps, {} tokens",
            gpu.stats().sampled_topk_steps(),
            reference.stats().sampled_topk_full_row_steps(),
            via_gpu.len()
        );
        record_executed_timed(Capability::LegacyModels, TEST, gate_start.elapsed());
    }

    /// The opted-in route on the real ternary 1.7B (`OXI_MODEL`): a seeded
    /// sampled request on the fused route — served from GPU top-k candidates
    /// — is token-for-token the classic full-row sampler, through `generate`
    /// and through the streaming entry point, and so is the same request
    /// with the route switched off (the default); with min-p 0.05 set through
    /// [`InferenceEngine::set_min_p`], the route's seeded draw equals the
    /// classic sampler's with the same min-p.
    #[test]
    fn real_model_opt_in_sampled_decode_matches_the_classic_sampler() {
        use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
        const TEST: &str = "oxibonsai-runtime::lib::\
             real_model_opt_in_sampled_decode_matches_the_classic_sampler";
        const MIN_P: f32 = 0.05;
        let opt_in = SampledTopKConfig::gpu_candidates();
        let Some(mmap) = real_model(TEST) else {
            return;
        };
        let gate_start = std::time::Instant::now();
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let gguf = GgufFile::parse(&mmap).expect("OXI_MODEL parses");
        let params = sampled_params();

        let mut engine = engine_with(&gguf, params.clone(), 5, 512, opt_in);
        if !engine.uses_fused_gpu_decode() {
            eprintln!(
                "capability report: {TEST} SKIPPED — OXI_MODEL is not on the fused GPU route"
            );
            record_skipped(Capability::LegacyModels, TEST);
            return;
        }
        assert_eq!(engine.sampled_topk().mode, SampledTopKMode::GpuCandidates);
        let via_engine = engine.generate(&REAL_PROMPT, 24).expect("generate");
        assert!(engine.stats().sampled_topk_steps() > 0);
        assert_eq!(engine.stats().sampled_full_row_requests(), 0);
        let steps_before_min_p = engine.stats().sampled_topk_steps();
        engine.reset();
        engine.set_min_p(MIN_P);
        let via_engine_min_p = engine
            .generate_with_seed(&REAL_PROMPT, 24, 5, &params)
            .expect("generate with min-p");
        assert!(engine.stats().sampled_topk_steps() > steps_before_min_p);
        drop(engine);

        let mut off = engine_with(&gguf, params.clone(), 5, 512, route_off());
        let via_route_off = off.generate(&REAL_PROMPT, 24).expect("route off");
        assert_eq!(off.stats().sampled_topk_steps(), 0);
        drop(off);

        let mut reference =
            InferenceEngine::from_gguf(&gguf, params.clone(), 5, 512).expect("real engine");
        let classic = classic_sampler_loop(&mut reference, &params, 5, &REAL_PROMPT, 24);
        reference.reset();
        let classic_min_p =
            classic_sampler_loop_with_min_p(&mut reference, &params, 5, MIN_P, &REAL_PROMPT, 24);
        drop(reference);
        eprintln!(
            "real-model opt-in sampled decode: engine {via_engine:?} | route off \
             {via_route_off:?} | classic {classic:?} | min-p {MIN_P}: engine \
             {via_engine_min_p:?} | classic {classic_min_p:?}"
        );
        assert!(!classic.is_empty());
        assert_eq!(via_engine, classic);
        assert_eq!(via_route_off, classic);
        assert_eq!(via_engine_min_p, classic_min_p);

        let mut streaming = engine_with(&gguf, params.clone(), 5, 512, opt_in);
        let (tx, rx) = std::sync::mpsc::channel();
        streaming
            .generate_streaming_sync(&REAL_PROMPT, 24, &tx)
            .expect("streaming");
        drop(tx);
        assert_eq!(rx.into_iter().collect::<Vec<u32>>(), classic);
        assert!(streaming.stats().sampled_topk_steps() > 0);
        eprintln!(
            "real-model opt-in sampled decode: engine == classic == route off, {} tokens {:?}; \
             min-p {MIN_P}: engine == classic, {} tokens",
            classic.len(),
            classic,
            classic_min_p.len()
        );
        record_executed_timed(Capability::LegacyModels, TEST, gate_start.elapsed());
    }
}
