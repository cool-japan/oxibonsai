//! Design §8.2 **G3**: "f64-reference self-consistency of our hybrid forward on the real 27B at
//! layers 0, 3, 7, 31, 63 (cos >= 0.99999 f32 vs f64 on the same weights)".
//!
//! The fork has no per-layer activation dump to compare against (its build
//! carries no eval-callback binary), so G3 is: take the **production** `f32`
//! forward's own per-layer residual stream (`HybridModel::forward_with_dump`,
//! teacher-forced over prompt 1 plus the fork's first golden tokens), feed
//! each selected layer's *input* rows to the independent `f64` evaluation
//! in [`crate::f64_layer`], and compare its *output* rows with the `f32`
//! path's. Layer 0 is a Gated-DeltaNet layer; 3, 7, 31 and 63 are
//! full-attention layers (the first, the second, a middle and the last).
//!
//! # What is asserted
//!
//! * `cos >= 0.99999` over each layer's whole `[t × 5120]` output block and
//!   for every position separately (the ruled quantity);
//! * the cosine of each layer's *contribution* (`output - input`) at least
//!   [`G3_DELTA_COS_MIN`] — the residual pass-through makes the output
//!   cosine easy, the contribution cosine is what a wrong projection or a
//!   wrong recurrence would actually move;
//! * `max |f32 - f64| <= G3_MAX_ABS_REL * max |f64|` per layer — the
//!   documented band (see [`G3_MAX_ABS_REL`] for the measured values: the
//!   worst layer sits at 2.7e-7 of its activation scale, every cosine
//!   rounds to 1.000000000).
//!
//! Memory: the model is mmapped and never widened to `f32`
//! weights; the `f64` reference dequantizes one weight row at a time. The
//! measured run peaks at a 380 MB process footprint (7.2 GB resident, all
//! of it the shared file mapping).
//!
//! The model is loaded with an `f32` KV cache here: G3 measures arithmetic,
//! and the shipped `f16` KV rounding (exercised by G1/G4 against the fork)
//! would otherwise be indistinguishable from an arithmetic error.
//!
//! The reference itself is validated first, on the synthetic fixture, by
//! chaining it through every layer and matching the fixture's own
//! independent `f64` model
//! (`hybrid_f64_layer_reference_matches_the_fixture_bonsai2`).

use std::sync::Arc;
use std::time::Instant;

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::writer::TensorType;
use oxibonsai_kernels::KernelDispatcher;
use oxibonsai_model::hybrid::model::{HybridModel, KvPrecision};

use crate::f64_layer::{self, agreement, Fold, Weights};
use crate::harness::{
    golden_dir, locate_model, parse_golden_steps, parse_prompt_tokens, read_golden,
    real_model_serial, record_capability_timed, PQ2_ENV, PQ2_FILE,
};
use crate::hybrid_gguf::{all_variant_specs, build};

/// G3's per-layer activation cosine floor, f32 vs f64.
const G3_COS_MIN: f64 = 0.99999;

/// Floor on the cosine of a layer's contribution (`output - input`).
const G3_DELTA_COS_MIN: f64 = 0.9999;

/// Band on `max |f32 - f64|`, relative to the layer's largest reference
/// activation — G3's documented max-abs band.
///
/// Measured on the real `PQ2_0` 27B (12 teacher-forced positions of
/// prompt 1, this box's NEON kernels):
///
/// | layer | kind | max \|f32 − f64\| | max \|ref\| | ratio | cos (block / worst row / contribution) |
/// |---|---|---|---|---|---|
/// | 0 | GDN | 2.95e-6 | 13.6 | 2.17e-7 | 1.000000000 / 1.000000000 / 1.000000000 |
/// | 3 | full | 8.82e-6 | 32.6 | 2.71e-7 | 1.000000000 / 1.000000000 / 1.000000000 |
/// | 7 | full | 4.40e-6 | 42.5 | 1.04e-7 | 1.000000000 / 1.000000000 / 1.000000000 |
/// | 31 | full | 3.62e-6 | 102.3 | 3.54e-8 | 1.000000000 / 1.000000000 / 1.000000000 |
/// | 63 | full | 8.97e-5 | 499.2 | 1.80e-7 | 1.000000000 / 1.000000000 / 1.000000000 |
///
/// i.e. a few `f32` ulps of the layer's scale. The band leaves ~37x
/// headroom over the worst measured ratio — room for another CPU tier's
/// summation order (AVX2 / AVX-512 reduce in a different order than NEON,
/// at the same ~1e-7 relative magnitude) — while a wrong projection, a
/// wrong recurrence step or a lossy intermediate (an `f16` / INT8 round
/// trip is already ~1e-3 relative) lands orders of magnitude outside it.
const G3_MAX_ABS_REL: f64 = 1e-5;

/// Layers of the ruling: the first GDN layer, the first two full-attention
/// layers, a middle one and the last.
const G3_LAYERS: [usize; 5] = [0, 3, 7, 31, 63];

/// Positions evaluated: prompt 1 (5 tokens) plus the fork's first seven
/// greedy tokens — twelve, above the ruling's ">= 8".
const G3_POSITIONS: usize = 12;

fn to_f64_rows(flat: &[f32], hidden: usize) -> Vec<Vec<f64>> {
    flat.chunks_exact(hidden)
        .map(|r| r.iter().map(|&v| f64::from(v)).collect())
        .collect()
}

fn to_f32_rows(flat: &[f32], hidden: usize) -> Vec<Vec<f32>> {
    flat.chunks_exact(hidden).map(<[f32]>::to_vec).collect()
}

/// The reference, validated against the fixture's own independent f64
/// model: chained
/// from the embedding through every layer of the folded, grouped `PQ2_0`
/// fixture (and its unfolded twin), it must reproduce that model's final
/// residual stream; and fed the production path's per-layer inputs, it
/// must agree with the production path's outputs.
#[test]
fn hybrid_f64_layer_reference_matches_the_fixture_bonsai2() {
    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let specs: Vec<_> = all_variant_specs(0xB211_F164_0000_0003)
        .into_iter()
        .filter(|s| matches!(s.quant, TensorType::PQ2_0 | TensorType::F32) && s.gdn_v_grouped)
        .collect();
    assert!(!specs.is_empty(), "the PQ2_0 / F32 grouped variants exist");
    for spec in &specs {
        let fixture = build(spec).expect("fixture builds");
        let bytes = std::fs::read(&fixture.path).expect("fixture readable");
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let cfg = fixture.cfg.clone();
        let hidden = cfg.base.hidden_size;
        let weights = Weights::new(&gguf);
        let fold = Fold::from_gguf(&gguf);
        assert_eq!(fold.is_some(), spec.hadamard, "{spec:?}: fold presence");
        let tokens: Vec<u32> = fixture
            .reference
            .token_ids
            .iter()
            .map(|&t| u32::try_from(t).expect("token fits"))
            .collect();

        // (1) Chained through every layer: the independent f64 models agree.
        let mut h = f64_layer::embed(&weights, fold.as_ref(), &tokens);
        for layer in 0..cfg.base.num_layers {
            h = f64_layer::layer(&weights, &cfg, fold.as_ref(), layer, &h);
        }
        for (t, (ours, theirs)) in h.iter().zip(&fixture.reference.final_hidden).enumerate() {
            let num: f64 = ours
                .iter()
                .zip(theirs)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let den: f64 = theirs.iter().map(|b| b * b).sum();
            let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
            assert!(
                rel <= 1e-9,
                "{spec:?} token {t}: the f64 layer reference disagrees with the fixture's own \
                 f64 model (relative L2 {rel:.3e})"
            );
        }

        // (2) Teacher-forced per layer against the production f32 path.
        let mut model = HybridModel::from_gguf_with_precision(
            &gguf,
            cfg.clone(),
            64,
            &kernel,
            KvPrecision::F32,
        )
        .expect("fixture model loads");
        let dump = model
            .forward_with_dump(&tokens, 0, None)
            .expect("dump forward");
        for layer in 0..cfg.base.num_layers {
            let input = if layer == 0 {
                to_f64_rows(&dump.embedding, hidden)
            } else {
                to_f64_rows(&dump.layers[layer - 1], hidden)
            };
            let want = f64_layer::layer(&weights, &cfg, fold.as_ref(), layer, &input);
            let ours = to_f32_rows(&dump.layers[layer], hidden);
            let a = agreement(&ours, &want, &input);
            assert!(
                a.cos >= G3_COS_MIN
                    && a.min_row_cos >= G3_COS_MIN
                    && a.delta_cos >= G3_DELTA_COS_MIN,
                "{spec:?} layer {layer}: {a:?}"
            );
        }
    }
}

/// G3 on the real `PQ2_0` 27B.
#[test]
fn hybrid_real_27b_layers_match_the_f64_reference_bonsai2() {
    const TEST: &str = "oxibonsai-model::hybrid_forward_parity_tests::\
                        hybrid_real_27b_layers_match_the_f64_reference_bonsai2";
    let Some(path) = locate_model(PQ2_ENV, PQ2_FILE, TEST) else {
        return;
    };
    let gate_start = Instant::now();
    let _one_real_model_at_a_time = real_model_serial();
    let mmap = mmap_gguf_file(&path).unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()));
    let gguf = GgufFile::parse(&mmap).expect("27B PQ2_0 GGUF parses");
    let cfg = HybridModel::config_from_gguf(&gguf).expect("qwen35 config");
    let hidden = cfg.base.hidden_size;
    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let mut model =
        HybridModel::from_gguf_with_precision(&gguf, cfg.clone(), 64, &kernel, KvPrecision::F32)
            .expect("27B loads");

    // Teacher-forced tokens: prompt 1 + the fork's greedy continuation.
    let golden = golden_dir();
    let mut tokens = parse_prompt_tokens(&read_golden(
        &golden,
        "Ternary-Bonsai-2-27B-PQ2_0.prompt1.prompt_tokens.txt",
    ));
    let steps = parse_golden_steps(&read_golden(&golden, "PQ2_0.prompt1.server.json"));
    tokens.extend(steps.iter().map(|s| s.id));
    tokens.truncate(G3_POSITIONS);
    assert_eq!(tokens.len(), G3_POSITIONS);

    let t0 = Instant::now();
    let dump = model
        .forward_with_dump(&tokens, 0, None)
        .expect("f32 forward with per-layer dump");
    eprintln!(
        "G3: f32 forward over {} positions with dump in {:.2}s",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );
    let weights = Weights::new(&gguf);
    let fold = Fold::from_gguf(&gguf);
    assert!(fold.is_some(), "the real 27B is folded");

    let mut failures = Vec::new();
    for &layer in &G3_LAYERS {
        let t = Instant::now();
        let input = if layer == 0 {
            to_f64_rows(&dump.embedding, hidden)
        } else {
            to_f64_rows(&dump.layers[layer - 1], hidden)
        };
        let want = f64_layer::layer(&weights, &cfg, fold.as_ref(), layer, &input);
        let ours = to_f32_rows(&dump.layers[layer], hidden);
        let a = agreement(&ours, &want, &input);
        eprintln!(
            "G3 layer {layer:>2} ({}): cos {:.9} min-row cos {:.9} contribution cos {:.9} \
             max|d| {:.3e} (max|ref| {:.3e}, rel {:.3e}) rel-L2 {:.3e} [{:.1}s]",
            if cfg.is_full_attention(layer) {
                "full"
            } else {
                "GDN "
            },
            a.cos,
            a.min_row_cos,
            a.delta_cos,
            a.max_abs,
            a.ref_max,
            a.max_abs / a.ref_max.max(f64::MIN_POSITIVE),
            a.rel_l2,
            t.elapsed().as_secs_f64(),
        );
        if a.cos < G3_COS_MIN || a.min_row_cos < G3_COS_MIN {
            failures.push(format!(
                "layer {layer}: activation cosine below {G3_COS_MIN}: {a:?}"
            ));
        }
        if a.delta_cos < G3_DELTA_COS_MIN {
            failures.push(format!(
                "layer {layer}: contribution cosine below {G3_DELTA_COS_MIN}: {a:?}"
            ));
        }
        if a.max_abs > G3_MAX_ABS_REL * a.ref_max {
            failures.push(format!(
                "layer {layer}: max|f32 - f64| outside {G3_MAX_ABS_REL:e} x max|ref|: {a:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "G3:\n{}", failures.join("\n"));
    record_capability_timed(true, TEST, Some(gate_start.elapsed()));
}
