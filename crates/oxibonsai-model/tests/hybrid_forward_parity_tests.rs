//! CPU parity gates for the `qwen35` hybrid forward driver (B2-11,
//! `bonsai2-design.md` §8.2 gates **G3 / G5 / G11** on the synthetic
//! fixtures and **G1 / G4** on the real 27B when it is present).
//!
//! # Why this file consumes B2-16's fixture rather than its own reference
//!
//! `tests/fixtures/hybrid_gguf.rs` carries an **independent** `f64` scalar
//! model of the whole `qwen35` stack (`ReferenceForward`), written by a
//! different package against the same design text. A reference written
//! here, beside the implementation, would share its misconceptions — the
//! rotation order, the tiled↔grouped v-head indexing and the un-rotated
//! `ssm_alpha`/`ssm_beta` inputs are all things a single author gets
//! consistently wrong. That fixture lives under `tests/`, so it is only
//! reachable from an integration test, which is why this file exists.
//!
//! # Why the quantized variants are exact oracles
//!
//! The fixture's quantized weights are drawn from `{-1, 0, +1}` (`{-1, +1}`
//! for the 1-bit format), so every block's absmax scale is exactly `1.0`
//! (representable in `f16`) and quantize→dequantize is **lossless**. The
//! `f64` reference's pre-quantization values are therefore the bytes on
//! disk, and a 1e-4 agreement is a statement about the *arithmetic*, not
//! about a codec's noise floor.
//!
//! Every test name contains both `hybrid` and `bonsai2` so the package
//! gate's two substring filters select the whole file.

#[path = "fixtures/hybrid_gguf.rs"]
mod hybrid_gguf;

#[path = "bonsai2_real/harness.rs"]
mod harness;

#[path = "bonsai2_real/greedy_gates.rs"]
mod greedy_gates;

#[path = "bonsai2_real/f64_layer.rs"]
mod f64_layer;

#[path = "bonsai2_real/layer_gate.rs"]
mod layer_gate;

use std::sync::Arc;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::TensorType;
use oxibonsai_kernels::KernelDispatcher;
use oxibonsai_model::hybrid::model::{HybridModel, KvPrecision};

use harness::{log_softmax, parse_golden_steps, parse_prompt_tokens, top_n};
use hybrid_gguf::{all_variant_specs, build, HybridFixtureSpec, T_TOKENS};

const BASE_SEED: u64 = 0xB211_0F0A_5E11_0000;

/// Tolerance for "matches the f64 scalar model", design §8.2's 1e-4.
///
/// Applied as `|a - b| <= ATOL + RTOL * |b|`: the logit rows of the
/// fixture reach |x| ~ 10, where a pure absolute 1e-4 would demand ~24
/// bits of agreement from an `f32` pipeline whose own unit in the last
/// place is already ~1e-6 at that magnitude.
const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-4;

/// `true` for the four `F32` fixture variants, which bind every projection
/// through the dense `LinearLayer::Dense` arm (gatekeeper REQUIRED #5).
fn is_dense_variant(spec: &HybridFixtureSpec) -> bool {
    matches!(spec.quant, TensorType::F32)
}

/// Largest `|a - b| - (ATOL + RTOL * |b|)` over the pair, i.e. the amount
/// by which the worst element misses the tolerance (negative = passes).
fn worst_miss(actual: &[f32], reference: &[f64]) -> (f64, usize) {
    let mut worst = f64::NEG_INFINITY;
    let mut index = 0;
    for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        let miss = (f64::from(a) - b).abs() - (ATOL + RTOL * b.abs());
        if miss > worst {
            worst = miss;
            index = i;
        }
    }
    (worst, index)
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

fn argmax_f64(values: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

// ═════════════════════════════════════════════════════════════════════════
//  G3 — every synthetic variant matches the f64 scalar model
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_forward_matches_f64_reference_on_every_bonsai2_variant() {
    let specs = all_variant_specs(BASE_SEED);
    assert_eq!(
        specs.len(),
        24,
        "the fixture generator's variant matrix changed shape"
    );

    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let mut checked = 0usize;
    let mut checked_dense = 0usize;

    for spec in &specs {
        let fixture = build(spec).expect("fixture builds");
        let bytes = std::fs::read(&fixture.path).expect("fixture file readable");
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf_with_precision(
            &gguf,
            fixture.cfg.clone(),
            64,
            &kernel,
            // The reference keeps its KV history in exact f64; the shipped
            // f16 cache is pinned separately by
            // `hybrid_f16_kv_agrees_with_f32_kv_bonsai2`.
            KvPrecision::F32,
        )
        .unwrap_or_else(|e| panic!("{spec:?}: hybrid model must load: {e}"));

        let vocab = fixture.cfg.base.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        for (t, &token) in fixture.reference.token_ids.iter().enumerate() {
            let token = u32::try_from(token).expect("fixture token fits u32");
            model
                .forward(token, t, &mut logits)
                .unwrap_or_else(|e| panic!("{spec:?}: forward at pos {t}: {e}"));
            let reference = &fixture.reference.logits[t];
            assert_eq!(reference.len(), vocab, "reference row width");
            let (miss, index) = worst_miss(&logits, reference);
            assert!(
                miss <= 0.0,
                "{spec:?}: token {t} logit[{index}] = {} vs reference {} (misses the 1e-4 \
                 tolerance by {miss:.3e})",
                logits[index],
                reference[index],
            );
            assert_eq!(
                argmax(&logits),
                argmax_f64(reference),
                "{spec:?}: token {t} greedy choice diverged"
            );
        }
        checked += 1;
        if is_dense_variant(spec) {
            checked_dense += 1;
        }
    }

    assert_eq!(
        checked, 24,
        "every variant — quantized and dense — must load and be checked"
    );
    assert_eq!(
        checked_dense, 4,
        "the four F32 variants run through the dense LinearLayer arm"
    );
}

// ═════════════════════════════════════════════════════════════════════════
//  G5 — prefill(T) == T sequential decode steps
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_prefill_equals_sequential_decode_bonsai2() {
    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let specs = all_variant_specs(BASE_SEED ^ 0x5151_5151);

    for spec in &specs {
        let fixture = build(spec).expect("fixture builds");
        let bytes = std::fs::read(&fixture.path).expect("fixture file readable");
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let vocab = fixture.cfg.base.vocab_size;
        let tokens: Vec<u32> = fixture
            .reference
            .token_ids
            .iter()
            .map(|&t| u32::try_from(t).expect("token fits u32"))
            .collect();

        // (a) T sequential decode steps.
        let mut sequential = HybridModel::from_gguf_with_precision(
            &gguf,
            fixture.cfg.clone(),
            64,
            &kernel,
            KvPrecision::F32,
        )
        .expect("model loads");
        let mut step_logits = vec![0.0f32; vocab];
        for (t, &token) in tokens.iter().enumerate() {
            sequential
                .forward(token, t, &mut step_logits)
                .expect("decode step");
        }

        // (b) one prefill over the whole prompt, chunked small enough that
        //     the chunk boundary lands mid-prompt (T_TOKENS is 6).
        let mut prefilled = HybridModel::from_gguf_with_precision(
            &gguf,
            fixture.cfg.clone(),
            64,
            &kernel,
            KvPrecision::F32,
        )
        .expect("model loads");
        prefilled.set_prefill_chunk(4).expect("chunk 4 accepted");
        let mut prefill_logits = vec![0.0f32; vocab];
        prefilled
            .forward_prefill(&tokens, 0, &mut prefill_logits)
            .expect("prefill");

        for (i, (&a, &b)) in step_logits.iter().zip(&prefill_logits).enumerate() {
            let tol = 1e-5 + 1e-5 * f64::from(b.abs());
            assert!(
                (f64::from(a) - f64::from(b)).abs() <= tol,
                "{spec:?}: logit[{i}] decode {a} vs prefill {b}"
            );
        }

        // The recurrent state must agree too — logits alone would pass even
        // if the last chunk's state had been rebuilt from scratch.
        let linear_layers = sequential.split().linear_layers().len();
        for slot in 0..linear_layers {
            let a = sequential.recurrent().ssm(slot).expect("ssm slot");
            let b = prefilled.recurrent().ssm(slot).expect("ssm slot");
            assert_eq!(a.len(), b.len(), "state width");
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                assert!(
                    (f64::from(x) - f64::from(y)).abs() <= 1e-5 + 1e-5 * f64::from(y.abs()),
                    "{spec:?}: slot {slot} S[{i}] decode {x} vs prefill {y}"
                );
            }
            let a = sequential.recurrent().conv(slot).expect("conv slot");
            let b = prefilled.recurrent().conv(slot).expect("conv slot");
            assert_eq!(a, b, "{spec:?}: slot {slot} conv window must be identical");
        }

        // …and so must the KV cache of every full-attention layer.
        let full_layers = sequential.split().full_layers().len();
        let n_kv = fixture.cfg.base.num_kv_heads;
        for slot in 0..full_layers {
            for head in 0..n_kv {
                let a = sequential.kv_cache().keys_for_owned(slot, head, T_TOKENS);
                let b = prefilled.kv_cache().keys_for_owned(slot, head, T_TOKENS);
                assert_eq!(a.len(), b.len(), "kv width");
                for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
                    assert!(
                        (f64::from(x) - f64::from(y)).abs() <= 1e-5 + 1e-5 * f64::from(y.abs()),
                        "{spec:?}: slot {slot} head {head} K[{i}] decode {x} vs prefill {y}"
                    );
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  The shipped f16 KV cache
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_f16_kv_agrees_with_f32_kv_bonsai2() {
    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let spec = all_variant_specs(BASE_SEED ^ 0xF16F_16F1)
        .into_iter()
        .find(|s| matches!(s.quant, TensorType::PQ2_0) && s.hadamard && s.gdn_v_grouped)
        .expect("the folded, grouped PQ2_0 variant exists");
    let fixture = build(&spec).expect("fixture builds");
    let bytes = std::fs::read(&fixture.path).expect("fixture file readable");
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let vocab = fixture.cfg.base.vocab_size;
    let tokens: Vec<u32> = fixture
        .reference
        .token_ids
        .iter()
        .map(|&t| u32::try_from(t).expect("token fits u32"))
        .collect();

    let mut exact = HybridModel::from_gguf_with_precision(
        &gguf,
        fixture.cfg.clone(),
        64,
        &kernel,
        KvPrecision::F32,
    )
    .expect("f32 model loads");
    assert!(!exact.kv_is_f16());
    let mut shipped = HybridModel::from_gguf(&gguf, 64).expect("default model loads");
    assert!(
        shipped.kv_is_f16(),
        "design §3.7: the shipped hybrid KV cache is f16 and layer-sparse"
    );

    let mut a = vec![0.0f32; vocab];
    let mut b = vec![0.0f32; vocab];
    for (t, &token) in tokens.iter().enumerate() {
        exact.forward(token, t, &mut a).expect("f32 decode");
        shipped.forward(token, t, &mut b).expect("f16 decode");
    }
    // f16 keys/values round to ~5e-4 relative, so the two paths agree to
    // about three decimal digits — but must still pick the same token,
    // which is what greedy parity against the fork actually needs.
    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
        let tol = 5e-2 + 5e-2 * f64::from(x.abs());
        assert!(
            (f64::from(x) - f64::from(y)).abs() <= tol,
            "logit[{i}] f32-KV {x} vs f16-KV {y}"
        );
    }
    assert_eq!(argmax(&a), argmax(&b), "f16 KV changed the greedy choice");
}

// ═════════════════════════════════════════════════════════════════════════
//  G11 — the per-layer dump the Metal parity gate diffs against
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_layer_dump_records_every_block_bonsai2() {
    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let spec = all_variant_specs(BASE_SEED ^ 0x0D0D_0D0D)
        .into_iter()
        .find(|s| matches!(s.quant, TensorType::PQ2_0) && s.hadamard && s.gdn_v_grouped)
        .expect("variant exists");
    let fixture = build(&spec).expect("fixture builds");
    let bytes = std::fs::read(&fixture.path).expect("fixture readable");
    let gguf = GgufFile::parse(&bytes).expect("fixture parses");
    let mut model = HybridModel::from_gguf_with_precision(
        &gguf,
        fixture.cfg.clone(),
        64,
        &kernel,
        KvPrecision::F32,
    )
    .expect("model loads");

    let tokens: Vec<u32> = fixture
        .reference
        .token_ids
        .iter()
        .map(|&t| u32::try_from(t).expect("token fits u32"))
        .collect();
    let vocab = fixture.cfg.base.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    let dump = model
        .forward_with_dump(&tokens, 0, Some(&mut logits))
        .expect("dump forward");

    let hidden = fixture.cfg.base.hidden_size;
    assert_eq!(dump.tokens, tokens);
    assert_eq!(dump.start_pos, 0);
    assert_eq!(dump.hidden, hidden);
    assert_eq!(dump.layers.len(), fixture.cfg.base.num_layers);
    assert_eq!(dump.embedding.len(), tokens.len() * hidden);
    assert_eq!(dump.final_norm.len(), hidden);
    for layer in 0..fixture.cfg.base.num_layers {
        for t in 0..tokens.len() {
            let row = dump
                .layer_row(layer, t)
                .unwrap_or_else(|| panic!("layer {layer} row {t} present"));
            assert_eq!(row.len(), hidden);
            assert!(
                row.iter().all(|v| v.is_finite()),
                "layer {layer} row {t} has a non-finite element"
            );
        }
    }

    // The dump's last row, re-normalised and pushed through the LM head, is
    // the same logit row the call returned — i.e. the dump is of the real
    // forward, not of a parallel bookkeeping path.
    let last = dump
        .layer_row(fixture.cfg.base.num_layers - 1, tokens.len() - 1)
        .expect("last row");
    assert!(last.iter().any(|v| *v != 0.0));
    assert!(logits.iter().all(|v| v.is_finite()));

    // The reference's own final hidden state for the same token, compared
    // the way design SS8.2's G3 specifies a *per-layer activation* be
    // compared -- cosine similarity >= 0.999, not the elementwise 1e-4 that
    // G3/G4 apply to logits. The distinction is not a loosening: the
    // residual stream reaches |x| ~ 60 here after eight accumulated layers,
    // where an f32 pipeline's own representable step is already ~4e-6 and a
    // 1e-4 *relative* band is below the noise floor of the accumulation
    // order alone, while the logits it feeds are RMS-normalised first and
    // do hold to 1e-4 (asserted by the G3 test above).
    let reference_final = &fixture.reference.final_hidden[tokens.len() - 1];
    assert_eq!(reference_final.len(), hidden);
    let cos = cosine_similarity(last, reference_final);
    assert!(
        cos >= 0.999,
        "final hidden cosine similarity {cos:.9} < 0.999 vs the f64 reference"
    );
    // …and the relative L2 error, which cosine alone cannot see (it is
    // scale-invariant, so a uniformly scaled activation would pass).
    let rel = relative_l2(last, reference_final);
    assert!(
        rel <= 1e-3,
        "final hidden relative L2 error {rel:.3e} > 1e-3 vs the f64 reference"
    );
}

/// Cosine similarity between an `f32` activation and its `f64` reference —
/// design SS8.2 G3's per-layer criterion.
fn cosine_similarity(actual: &[f32], reference: &[f64]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&a, &b) in actual.iter().zip(reference) {
        let a = f64::from(a);
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// `||actual - reference|| / ||reference||`.
fn relative_l2(actual: &[f32], reference: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &b) in actual.iter().zip(reference) {
        let d = f64::from(a) - b;
        num += d * d;
        den += b * b;
    }
    if den == 0.0 {
        return num.sqrt();
    }
    (num / den).sqrt()
}

// ═════════════════════════════════════════════════════════════════════════
//  The dense arm: an F32 qwen35 file loads and runs (gatekeeper REQUIRED #5)
// ═════════════════════════════════════════════════════════════════════════

/// The `F32` variants used to be refused outright (no dense `LinearLayer`
/// existed). They now bind every projection as `LinearLayer::Dense`, run,
/// and track the f64 reference within design §8.2's 1e-4 band — the same
/// band the quantized variants are held to (the residual error is `f32`
/// accumulation order, which a dense matrix does not remove).
#[test]
fn hybrid_dense_f32_variant_binds_the_dense_arm_and_matches_the_reference_bonsai2() {
    use oxibonsai_core::gguf::types::GgufTensorType;
    use oxibonsai_model::hybrid::HybridBlock;
    use oxibonsai_model::layers::linear::LinearLayer;

    let kernel = Arc::new(KernelDispatcher::auto_detect());
    let dense_specs: Vec<HybridFixtureSpec> = all_variant_specs(BASE_SEED ^ 0x3232_3232)
        .into_iter()
        .filter(is_dense_variant)
        .collect();
    assert_eq!(dense_specs.len(), 4, "four F32 variants");

    for spec in &dense_specs {
        let fixture = build(spec).expect("fixture builds");
        let bytes = std::fs::read(&fixture.path).expect("fixture readable");
        let gguf = GgufFile::parse(&bytes).expect("fixture parses");
        let mut model = HybridModel::from_gguf_with_precision(
            &gguf,
            fixture.cfg.clone(),
            64,
            &kernel,
            KvPrecision::F32,
        )
        .unwrap_or_else(|e| panic!("{spec:?}: an F32 qwen35 file must load: {e}"));

        // Every projection is the dense arm, reporting F32.
        let assert_dense = |layer: &LinearLayer<'_>, what: &str| {
            assert!(
                matches!(layer, LinearLayer::Dense(_)),
                "{spec:?}: {what} must bind as LinearLayer::Dense"
            );
            assert_eq!(layer.quant_type(), GgufTensorType::F32, "{spec:?}: {what}");
            assert!(layer.gpu_handle().is_none(), "{spec:?}: {what}");
            let weights = layer.dense_weights().expect("dense weights");
            assert_eq!(
                weights.len(),
                layer.out_features() * layer.in_features(),
                "{spec:?}: {what} weight count"
            );
        };
        assert_dense(model.lm_head(), "output.weight");
        for block in model.blocks() {
            match block {
                HybridBlock::Full(full) => {
                    for (layer, what) in [
                        (full.attn_q(), "attn_q"),
                        (full.attn_k(), "attn_k"),
                        (full.attn_v(), "attn_v"),
                        (full.attn_output(), "attn_output"),
                        (full.ffn_gate(), "ffn_gate"),
                        (full.ffn_up(), "ffn_up"),
                        (full.ffn_down(), "ffn_down"),
                    ] {
                        assert_dense(layer, what);
                    }
                }
                HybridBlock::Linear(linear) => {
                    for (layer, what) in [
                        (linear.attn_qkv(), "attn_qkv"),
                        (linear.attn_gate(), "attn_gate"),
                        (linear.ssm_out(), "ssm_out"),
                        (linear.ffn_gate(), "ffn_gate"),
                        (linear.ffn_up(), "ffn_up"),
                        (linear.ffn_down(), "ffn_down"),
                    ] {
                        assert_dense(layer, what);
                    }
                }
            }
        }

        // Batched projection == per-row projection, bit for bit (one
        // dispatched GEMV per row either way).
        let head = model.lm_head();
        let (out_f, in_f) = (head.out_features(), head.in_features());
        let rows = 3usize;
        let input: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
            .collect();
        let mut batched = vec![0.0f32; rows * out_f];
        head.forward_mat(&input, &mut batched, rows)
            .expect("dense batched projection");
        for r in 0..rows {
            let mut single = vec![0.0f32; out_f];
            head.forward_vec(&input[r * in_f..(r + 1) * in_f], &mut single)
                .expect("dense single-row projection");
            assert_eq!(
                &batched[r * out_f..(r + 1) * out_f],
                single.as_slice(),
                "{spec:?}: batched row {r} must equal the single-row projection bit for bit"
            );
        }

        // The whole model runs and tracks the f64 reference within the
        // design's 1e-4 band.
        let vocab = fixture.cfg.base.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        for (t, &token) in fixture.reference.token_ids.iter().enumerate() {
            let token = u32::try_from(token).expect("fixture token fits u32");
            model
                .forward(token, t, &mut logits)
                .unwrap_or_else(|e| panic!("{spec:?}: forward at pos {t}: {e}"));
            let reference = &fixture.reference.logits[t];
            let (miss, index) = worst_miss(&logits, reference);
            assert!(
                miss <= 0.0,
                "{spec:?}: token {t} logit[{index}] = {} vs reference {} (misses the 1e-4 \
                 tolerance by {miss:.3e})",
                logits[index],
                reference[index],
            );
            assert_eq!(
                argmax(&logits),
                argmax_f64(reference),
                "{spec:?}: token {t} greedy choice"
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  The G1/G4 harness itself, verified without the 7.2 GB weight file
// ═════════════════════════════════════════════════════════════════════════

/// The real goldens are in the session scratchpad, not the repository, so
/// the ignored 27B gate could otherwise fail on a *parsing* mistake that
/// looks like a forward-pass divergence. These fixtures reproduce the two
/// formats exactly (an abridged `llama-server` `/completion` body and a
/// `llama-cli` tokenisation dump) so the harness is proven here, in the
/// ordinary gate, and only the weights are missing over there.
#[test]
fn hybrid_golden_harness_parses_the_fork_formats_bonsai2() {
    let server = r#"{
 "index": 0,
 "content": " Tokyo.",
 "tokens_predicted": 2,
 "tokens_evaluated": 5,
 "completion_probabilities": [
  {
   "id": 25358,
   "token": " Tokyo",
   "bytes": [32, 84, 111, 107, 121, 111],
   "logprob": -0.41223224997520447,
   "top_logprobs": [
    {"id": 25358, "token": " Tokyo", "logprob": -0.41223224997520447},
    {"id": 13, "token": ".", "logprob": -1.5}
   ]
  },
  {
   "id": 13,
   "token": ".",
   "bytes": [46],
   "logprob": -0.125,
   "top_logprobs": [
    {"id": 13, "token": ".", "logprob": -0.125},
    {"id": 198, "token": "\n", "logprob": -2.25}
   ]
  }
 ]
}"#;
    let steps = parse_golden_steps(server);
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].id, 25358);
    assert_eq!(steps[1].id, 13);
    assert_eq!(steps[0].top.len(), 2);
    assert_eq!(steps[0].top[0].0, 25358);
    assert!((steps[0].top[0].1 - -0.41223224997520447).abs() < 1e-12);
    assert_eq!(steps[1].top[1], (198, -2.25));

    // `llama-cli`'s tokenisation dump: a timestamped log line per token,
    // the id between the level flag and the `->`.
    let dump = "0.04.926.888 I    760 -> 'The'\n\
                0.04.926.892 I   6511 -> ' capital'\n\
                0.04.926.895 I    314 -> ' of'\n\
                0.04.926.900 I   6124 -> ' Japan'\n\
                0.04.926.900 I    369 -> ' is'\n\
                not a token line at all\n";
    assert_eq!(parse_prompt_tokens(dump), vec![760, 6511, 314, 6124, 369]);
    assert!(parse_prompt_tokens("").is_empty());
}

/// The vendored oracle (`tests/fixtures/bonsai2_golden{,_cpu}/`) is
/// complete and self-consistent, so the real-model gates never depend on a
/// session scratchpad — checked on every run, model files or not. Reads the
/// vendored copies directly, whatever `OXI_BONSAI2_GOLDEN_DIR` says.
#[test]
fn hybrid_vendored_fork_goldens_are_complete_and_consistent_bonsai2() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let metal_dir = fixtures.join("bonsai2_golden");
    let cpu_dir = fixtures.join("bonsai2_golden_cpu");
    let read = |dir: &std::path::Path, name: &str| -> String {
        let path = dir.join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    for prompt_index in 1..=3usize {
        let prompt = harness::PROMPTS[prompt_index - 1];
        let mut greedy_ids = Vec::new();
        for (dir, backend) in [(&metal_dir, "metal"), (&cpu_dir, "cpu")] {
            let server = read(dir, &format!("PQ2_0.prompt{prompt_index}.server.json"));
            // Redacted capture path: no local path of the capture machine.
            assert!(
                !server.contains("/Users/") && !server.contains("/home/"),
                "{backend} prompt {prompt_index}: an absolute path was vendored"
            );
            let root: serde_json::Value = serde_json::from_str(&server).expect("golden parses");
            assert_eq!(
                root.get("model").and_then(serde_json::Value::as_str),
                Some(harness::PQ2_FILE),
                "{backend} prompt {prompt_index}: the dump names the release file"
            );
            let steps = parse_golden_steps(&server);
            assert_eq!(steps.len(), 24, "{backend} prompt {prompt_index}: 24 steps");
            for (step, golden) in steps.iter().enumerate() {
                assert_eq!(
                    golden.top.len(),
                    harness::TOP_N,
                    "{backend} prompt {prompt_index} step {step}: top-{}",
                    harness::TOP_N
                );
                assert_eq!(
                    golden.top[0].0, golden.id,
                    "{backend} prompt {prompt_index} step {step}: greedy = top-1"
                );
                assert!(
                    golden.top.windows(2).all(|w| w[0].1 >= w[1].1),
                    "{backend} prompt {prompt_index} step {step}: logprobs descend"
                );
            }
            if backend == "metal" {
                greedy_ids = steps.iter().map(|s| s.id).collect();
                // The server's 24-token content is the prefix of the
                // `llama-cli` 32-token dumps of both builds.
                let content = root
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .expect("server content");
                for build in ["PQ2_0", "PTQ1_0"] {
                    let text = read(
                        &metal_dir,
                        &format!("Ternary-Bonsai-2-27B-{build}.prompt{prompt_index}.txt"),
                    );
                    assert!(
                        text.starts_with(content) && text.ends_with("\n\n"),
                        "{build} prompt {prompt_index}: text dump vs server content"
                    );
                }
            }
        }
        assert_eq!(greedy_ids.len(), 24);
        // Both builds tokenise the raw prompt identically, into no more ids
        // than it has bytes.
        let pq2 = parse_prompt_tokens(&read(
            &metal_dir,
            &format!("Ternary-Bonsai-2-27B-PQ2_0.prompt{prompt_index}.prompt_tokens.txt"),
        ));
        let ptq1 = parse_prompt_tokens(&read(
            &metal_dir,
            &format!("Ternary-Bonsai-2-27B-PTQ1_0.prompt{prompt_index}.prompt_tokens.txt"),
        ));
        assert!(!pq2.is_empty(), "prompt {prompt_index}: tokens");
        assert_eq!(
            pq2, ptq1,
            "prompt {prompt_index}: {prompt:?} tokenises alike"
        );
        assert!(
            pq2.len() <= prompt.len(),
            "prompt {prompt_index}: token count"
        );
    }
}

#[test]
fn hybrid_golden_harness_logprobs_and_ranking_agree_with_the_fork_bonsai2() {
    // log_softmax is shift-invariant and sums to 1 in probability space.
    let logits = [1.0f32, 3.0, 2.0, -5.0];
    let lp = log_softmax(&logits);
    let total: f64 = lp.iter().map(|x| x.exp()).sum();
    assert!((total - 1.0).abs() < 1e-12, "probabilities sum to {total}");
    let shifted: Vec<f32> = logits.iter().map(|l| l + 17.0).collect();
    for (a, b) in lp.iter().zip(log_softmax(&shifted)) {
        assert!((a - b).abs() < 1e-9, "log_softmax must be shift-invariant");
    }
    // The largest logit is the largest logprob, and the gap is preserved.
    assert!((lp[1] - lp[2] - 1.0).abs() < 1e-9);

    // top_n ranks by logprob, descending, ties broken by id.
    let ranked = top_n(&lp, 3);
    assert_eq!(ranked[0].0, 1);
    assert_eq!(ranked[1].0, 2);
    assert_eq!(ranked[2].0, 0);
    let tied = top_n(&[0.0, 0.0, -1.0], 2);
    assert_eq!(tied[0].0, 0);
    assert_eq!(tied[1].0, 1);

    // argmax and the top-1 of the ranking are the same token, which is what
    // makes the G1 and G4 assertions consistent with each other.
    assert_eq!(
        u32::try_from(argmax(&logits)).expect("fits"),
        top_n(&lp, 1)[0].0
    );
}
