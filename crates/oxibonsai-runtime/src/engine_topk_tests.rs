//! Tests for the sampled top-k route (`perf-11`, sampled half) of
//! [`crate::engine_greedy`].
//!
//! Three layers:
//!
//! * the **shipped default**: the route is off, and a sampled request decodes
//!   exactly like an independently spelled-out classic loop (prefill, then a
//!   standalone `Sampler::new(params, seed)` draw over every full logit row)
//!   — on a synthetic CPU engine here, on a fused Metal fixture and on the
//!   real 1.7B (`OXI_MODEL`) below;
//! * host-only tests of the CPU candidate extraction ([`top_k_candidates`]),
//!   the configuration, and the opt-in route's *distribution* (it draws from
//!   exactly the classic sampler's distribution; only a seed's realisation
//!   differs, which is why it is opt-in);
//! * Metal tests of the opt-in route (on the fused synthetic ternary model,
//!   and on the real 1.7B via `OXI_MODEL`) proving that (a) the GPU
//!   `topk_f32` download over the resident logits equals the CPU extraction
//!   of the same row bit for bit, and (b) a sampled generation through the
//!   GPU candidates is token-for-token identical to the full-row candidate
//!   reference mode, with every step counted where it was served.

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
    let mut sampler = Sampler::new(params.clone(), seed);
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

/// The route ships **off** (a seeded request keeps its pre-route output);
/// `gpu_candidates()` is the one-call opt-in with the default candidate
/// count, and the count clamps to the kernel cap and the vocabulary.
#[test]
fn the_route_is_off_by_default_and_candidates_clamp() {
    let default = SampledTopKConfig::default();
    assert_eq!(default.mode, SampledTopKMode::Off);
    assert_eq!(SampledTopKMode::default(), SampledTopKMode::Off);
    assert_eq!(default.candidates, DEFAULT_SAMPLED_TOPK_CANDIDATES);
    let opt_in = SampledTopKConfig::gpu_candidates();
    assert_eq!(opt_in.mode, SampledTopKMode::GpuCandidates);
    assert_eq!(opt_in.candidates, DEFAULT_SAMPLED_TOPK_CANDIDATES);
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

    let engine = InferenceEngine::new(
        oxibonsai_core::config::Qwen3Config::tiny_test(),
        SamplingParams::default(),
        1,
    );
    assert_eq!(engine.sampled_topk(), SampledTopKConfig::default());
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
    // Not the fused GPU route: even opted in, the classic full-row sampler
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

/// The opt-in route's candidate draw and the classic full-row sampler draw
/// from the **same distribution** — the analytic one: temperature, top-k,
/// softmax, top-p, renormalise — so the route never changes what a sampled
/// request can produce or how likely it is, only which token a given seed
/// lands on. Both samplers' empirical frequencies over many seeded draws
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

    const MAX_SEQ: usize = 128;
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
        w.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(VOCAB as u32));
        w.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
        w.add_metadata(
            "qwen3.attention.layer_norm_rms_epsilon",
            MetadataWriteValue::F32(1e-6),
        );
        w.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
        w.add_tensor(TensorEntry {
            name: "token_embd.weight".into(),
            shape: vec![h as u64, VOCAB as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(VOCAB * h, 0.5),
        });
        w.add_tensor(TensorEntry {
            name: "output_norm.weight".into(),
            shape: vec![h as u64],
            tensor_type: TensorType::F32,
            data: f32_pattern(h, 1.0),
        });
        w.add_tensor(TensorEntry {
            name: "output.weight".into(),
            shape: vec![h as u64, VOCAB as u64],
            tensor_type: TensorType::TQ2_0_g128,
            data: tq2_pattern(VOCAB * h, 0xCAFE_BABE),
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

    /// The shipped default on the fused route: the route is off, the request
    /// is counted as a full-row request, and `generate`, the streaming entry
    /// point and `generate_with_seed` are token-for-token the classic
    /// full-row sampler — exactly what a sampled request did before the
    /// route existed.
    #[test]
    fn default_fused_engine_samples_exactly_like_the_classic_sampler() {
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let bytes = fused_ternary_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let prompt = [1u32, 5, 9, 2];
        let params = sampled_params();

        let mut engine =
            InferenceEngine::from_gguf(&gguf, params.clone(), 11, MAX_SEQ).expect("engine");
        assert!(engine.uses_fused_gpu_decode(), "fixture must be fused");
        assert_eq!(engine.sampled_topk().mode, SampledTopKMode::Off);
        assert!(!engine.sampled_topk_eligible(false));
        let via_engine = engine.generate(&prompt, 16).expect("generate");
        assert_eq!(engine.stats().sampled_full_row_requests(), 1);
        assert_eq!(engine.stats().sampled_topk_steps(), 0);
        assert_eq!(engine.stats().sampled_topk_full_row_steps(), 0);

        let mut reference =
            InferenceEngine::from_gguf(&gguf, params.clone(), 11, MAX_SEQ).expect("engine");
        let classic = classic_sampler_loop(&mut reference, &params, 11, &prompt, 16);
        assert_eq!(classic.len(), 16, "no EOS inside the fixture's vocabulary");
        assert_eq!(
            via_engine, classic,
            "the default fused engine must sample exactly like the classic sampler"
        );

        let mut streaming =
            InferenceEngine::from_gguf(&gguf, params.clone(), 11, MAX_SEQ).expect("engine");
        let (tx, rx) = std::sync::mpsc::channel();
        streaming
            .generate_streaming_sync(&prompt, 16, &tx)
            .expect("streaming");
        drop(tx);
        assert_eq!(rx.into_iter().collect::<Vec<u32>>(), classic);

        let mut seeded =
            InferenceEngine::from_gguf(&gguf, params.clone(), 999, MAX_SEQ).expect("engine");
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

    /// The opt-in route: sampled output through the GPU candidates is
    /// byte-identical to the full-row candidate reference, each engine counts
    /// where its steps were served, and the first token — drawn from the
    /// prefill row by the classic sampler — equals the default path's.
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

        // Step 0 is the classic draw over the prefill row on both paths.
        let mut default_engine =
            InferenceEngine::from_gguf(&gguf, sampled_params(), 11, MAX_SEQ).expect("engine");
        let via_default = default_engine.generate(&prompt, 16).expect("default");
        assert_eq!(via_default.first(), via_gpu.first());
    }

    /// `top_k` at or above the vocabulary is not a real top-k selection on
    /// the full row, so the opt-in route refuses it (counted); one below it
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

    /// A request the opt-in route cannot serve exactly — a penalty (it must
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

    /// The opt-in route on the real ternary 1.7B (`OXI_MODEL`), where the
    /// vocabulary (151 936) is far above the candidate count, so the GPU
    /// kernel really does select 64 of 151 936: the GPU download equals the
    /// CPU extraction, and generation equals the full-row candidate
    /// reference.
    #[test]
    fn real_model_gpu_topk_route_matches_the_full_row_candidate_reference() {
        use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};
        const TEST: &str = "engine_greedy::topk_tests::metal::\
             real_model_gpu_topk_route_matches_the_full_row_candidate_reference";
        let Some(mmap) = real_model(TEST) else {
            return;
        };
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
        record_executed(Capability::LegacyModels, TEST);
    }

    /// The shipped default on the real ternary 1.7B (`OXI_MODEL`): a seeded
    /// sampled request on the fused route is token-for-token the classic
    /// full-row sampler, through `generate` and through the streaming entry
    /// point.
    #[test]
    fn real_model_default_sampled_decode_matches_the_classic_sampler() {
        use oxibonsai_testkit::capability::{record_executed, record_skipped, Capability};
        const TEST: &str = "engine_greedy::topk_tests::metal::\
             real_model_default_sampled_decode_matches_the_classic_sampler";
        let Some(mmap) = real_model(TEST) else {
            return;
        };
        let _session = MetalGraph::bind_new_session().expect("metal session");
        let gguf = GgufFile::parse(&mmap).expect("OXI_MODEL parses");
        let params = sampled_params();

        let mut engine =
            InferenceEngine::from_gguf(&gguf, params.clone(), 5, 512).expect("real engine");
        if !engine.uses_fused_gpu_decode() {
            eprintln!(
                "capability report: {TEST} SKIPPED — OXI_MODEL is not on the fused GPU route"
            );
            record_skipped(Capability::LegacyModels, TEST);
            return;
        }
        assert_eq!(engine.sampled_topk().mode, SampledTopKMode::Off);
        let via_engine = engine.generate(&REAL_PROMPT, 24).expect("generate");
        assert_eq!(engine.stats().sampled_full_row_requests(), 1);
        drop(engine);

        let mut reference =
            InferenceEngine::from_gguf(&gguf, params.clone(), 5, 512).expect("real engine");
        let classic = classic_sampler_loop(&mut reference, &params, 5, &REAL_PROMPT, 24);
        drop(reference);
        eprintln!("real-model default sampled decode: engine {via_engine:?} | classic {classic:?}");
        assert!(!classic.is_empty());
        assert_eq!(via_engine, classic);

        let mut streaming =
            InferenceEngine::from_gguf(&gguf, params.clone(), 5, 512).expect("real engine");
        let (tx, rx) = std::sync::mpsc::channel();
        streaming
            .generate_streaming_sync(&REAL_PROMPT, 24, &tx)
            .expect("streaming");
        drop(tx);
        assert_eq!(rx.into_iter().collect::<Vec<u32>>(), classic);
        record_executed(Capability::LegacyModels, TEST);
    }
}
