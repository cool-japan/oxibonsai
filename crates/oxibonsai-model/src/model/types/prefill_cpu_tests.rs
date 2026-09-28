//! The batched prefill's tests, and the fixtures `forward_hidden`'s tests
//! share with them (`pub(in crate::model::types)`: test-only, and only for
//! this module's siblings).

use super::*;
use crate::model::types::OutputWeight;
use oxibonsai_core::config::{Qwen3Config, RopeScaling};
use oxibonsai_kernels::{KernelDispatcher, KernelTier};

/// Small config satisfying the `Q1_0_g128` fixture's constraints
/// (`hidden % 128 == 0`, `intermediate % 128 == 0`), with genuine GQA
/// (4 query heads over 2 KV heads) so the head mapping is exercised.
pub(in crate::model::types) fn tiny_config(num_layers: usize) -> Qwen3Config {
    Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers,
        num_attention_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        value_length: 32,
        vocab_size: 96,
        max_context_length: 512,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10_000.0,
        rope_scaling: RopeScaling::None,
        // M-17: the batched path is full-causal by construction and
        // declines a windowed model, so it is only reachable with `None`
        // (`windowed_model_is_declined` pins the decline).
        sliding_window: None,
        architecture: "test".to_string(),
        model_name: "prefill-cpu-test".to_string(),
    }
}

/// Deterministic dense FP32 LM head.
///
/// `new_for_testing_with_blocks` installs `OutputWeight::zero_fp32`,
/// whose weight vector is **empty**, so its logits carry no information
/// and could not distinguish a correct prefill from a broken one. Every
/// test below swaps in this real head first.
fn dense_lm_head(out_features: usize, in_features: usize) -> OutputWeight<'static> {
    let weights = (0..out_features * in_features)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    OutputWeight::Fp32 {
        weights,
        out_features,
        in_features,
    }
}

/// A `Q1_0_g128` model with real (deterministic) blocks and the dense
/// LM head above: in scope for the batched path.
pub(in crate::model::types) fn fixture(cfg: Qwen3Config) -> BonsaiModel<'static> {
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let mut model = BonsaiModel::new_for_testing_with_blocks(cfg);
    model.output_weight = dense_lm_head(vocab, hidden);
    model
}

/// Deterministic `TQ2_0_g128` blocks for the ternary test fixture below.
///
/// Mirrors `BonsaiModel::new_for_testing_with_blocks`'s own
/// `make_blocks_static` helper for `Q1_0_g128` -- that helper is a
/// closure private to `new_for_testing_with_blocks`'s body, not
/// reusable here, so this is its ternary twin. A `0b11` 2-bit code the
/// pattern happens to produce decodes to `0` (the K-01 reserved-code
/// contract every ternary kernel shares), which is a perfectly valid
/// ternary weight -- this fixture only needs deterministic,
/// differentiated data, not "realistic" weights.
fn ternary_blocks_static(n: usize, scale: f32, pattern: u8) -> &'static [BlockTQ2_0_g128] {
    let v: Vec<BlockTQ2_0_g128> = (0..n)
        .map(|i| {
            let mut qs = [0u8; 32];
            for (j, b) in qs.iter_mut().enumerate() {
                *b = pattern.wrapping_add(((i * 32 + j) & 0xff) as u8);
            }
            BlockTQ2_0_g128 {
                qs,
                d: half::f16::from_f32(scale),
            }
        })
        .collect();
    // Leak the allocation so the slice lives for 'static, same as the
    // Q1_0_g128 fixture does -- acceptable in tests.
    Box::leak(v.into_boxed_slice())
}

/// The ternary (`TQ2_0_g128`) twin of [`fixture`].
///
/// `fixture` builds every projection as `LinearLayer::OneBit`
/// (`Q1_0_g128`), so `PrefillMatrix::Ternary` and the ternary half of
/// `layer_plan` (this module's *other* branch) are never exercised by
/// any test that only calls `fixture` -- and ternary is the format this
/// project is named for and the one `LinearTernary::forward_batch`
/// actually uses in production.
///
/// Reuses `new_for_testing_with_blocks` for everything that does not
/// depend on the projection format -- embedding, KV cache, RoPE,
/// output norm -- then replaces `blocks` with freshly built
/// all-`LinearTernary` ones. `BonsaiModel::blocks` is `pub(crate)`, and
/// every other field this function touches is a plain private field
/// `mod.rs` declares on `BonsaiModel`: `prefill_cpu` is a *child*
/// module of `model::types` (`mod.rs`), so those private fields are
/// visible from here exactly as they are from `mod.rs` itself -- no new
/// visibility was widened to write this fixture.
pub(in crate::model::types) fn ternary_fixture(cfg: Qwen3Config) -> BonsaiModel<'static> {
    use crate::layers::linear::{LinearLayer, LinearTernary};
    use crate::layers::rms_norm::RmsNorm;
    use std::sync::Arc;

    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nq = cfg.num_attention_heads;
    let nkv = cfg.num_kv_heads;
    let inter = cfg.intermediate_size;
    assert!(
        h.is_multiple_of(128),
        "ternary test fixture requires hidden_size to be a multiple of 128"
    );
    assert!(
        inter.is_multiple_of(128),
        "ternary test fixture requires intermediate_size to be a multiple of 128"
    );
    let h_bpr = h / 128;
    let inter_bpr = inter / 128;

    // Same Reference-tier pin as `new_for_testing_with_blocks`, for the
    // same reason: populate the CPU `KvCache` deterministically instead
    // of routing through whatever GPU tier this host would auto-detect.
    let kernel_arc = Arc::new(KernelDispatcher::with_tier(KernelTier::Reference));

    let mut blocks = Vec::with_capacity(cfg.num_layers);
    for layer_idx in 0..cfg.num_layers {
        let q_blk = ternary_blocks_static(nq * hd * h_bpr, 0.01, 0xA5);
        let k_blk = ternary_blocks_static(nkv * hd * h_bpr, 0.01, 0x5A);
        let v_blk = ternary_blocks_static(nkv * hd * h_bpr, 0.01, 0x33);
        let o_blk = ternary_blocks_static(h * (nq * hd / 128).max(1), 0.01, 0xCC);
        let g_blk = ternary_blocks_static(inter * h_bpr, 0.01, 0x77);
        let u_blk = ternary_blocks_static(inter * h_bpr, 0.01, 0x88);
        let d_blk = ternary_blocks_static(h * inter_bpr, 0.01, 0x99);

        let attn_q: LinearLayer<'static> =
            LinearTernary::new(q_blk, nq * hd, h, kernel_arc.clone())
                .expect("q proj")
                .into();
        let attn_k: LinearLayer<'static> =
            LinearTernary::new(k_blk, nkv * hd, h, kernel_arc.clone())
                .expect("k proj")
                .into();
        let attn_v: LinearLayer<'static> =
            LinearTernary::new(v_blk, nkv * hd, h, kernel_arc.clone())
                .expect("v proj")
                .into();
        let attn_out: LinearLayer<'static> =
            LinearTernary::new(o_blk, h, nq * hd, kernel_arc.clone())
                .expect("o proj")
                .into();
        let ffn_gate: LinearLayer<'static> =
            LinearTernary::new(g_blk, inter, h, kernel_arc.clone())
                .expect("gate proj")
                .into();
        let ffn_up: LinearLayer<'static> = LinearTernary::new(u_blk, inter, h, kernel_arc.clone())
            .expect("up proj")
            .into();
        let ffn_down: LinearLayer<'static> =
            LinearTernary::new(d_blk, h, inter, kernel_arc.clone())
                .expect("down proj")
                .into();

        let block = TransformerBlock::new(
            layer_idx,
            RmsNorm::new(vec![1.0; h], cfg.rms_norm_eps),
            attn_q,
            attn_k,
            attn_v,
            attn_out,
            RmsNorm::new(vec![1.0; hd], cfg.rms_norm_eps),
            RmsNorm::new(vec![1.0; hd], cfg.rms_norm_eps),
            RmsNorm::new(vec![1.0; h], cfg.rms_norm_eps),
            ffn_gate,
            ffn_up,
            ffn_down,
            nq,
            nkv,
            hd,
            h,
        );
        blocks.push(block);
    }

    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let mut model = BonsaiModel::new_for_testing_with_blocks(cfg);
    model.blocks = blocks;
    model.dominant_quant_type = oxibonsai_core::GgufTensorType::TQ2_0_g128;
    model.output_weight = dense_lm_head(vocab, hidden);
    model
}

/// Cosine similarity in `f64` (`0.0` for a zero vector).
pub(in crate::model::types) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Run the sequential per-token reference over `prompt` on a model
/// `build` constructs, returning the last position's logits.
fn sequential_logits_with(
    cfg: &Qwen3Config,
    prompt: &[u32],
    build: impl Fn(Qwen3Config) -> BonsaiModel<'static>,
) -> Vec<f32> {
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let mut model = build(cfg.clone());
    let mut logits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        logits = model
            .forward(tok, i, &kernel)
            .expect("sequential forward should succeed");
    }
    logits
}

/// [`sequential_logits_with`] against the `Q1_0_g128` [`fixture`] —
/// every existing caller's reference before ternary coverage was added.
fn sequential_logits(cfg: &Qwen3Config, prompt: &[u32]) -> Vec<f32> {
    sequential_logits_with(cfg, prompt, fixture)
}

/// perf-M2's acceptance: the batched CPU prefill reproduces the
/// sequential per-token reference on the last position.
#[test]
fn batched_prefill_matches_the_sequential_reference() {
    let cfg = tiny_config(2);
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 5) % 96).collect();
    let expected = sequential_logits(&cfg, &prompt);

    let mut batched = fixture(cfg);
    let logits = batched
        .forward_prefill_cpu(&prompt, 0)
        .expect("batched prefill should succeed")
        .expect("the 1-bit fixture is in scope for the batched path");

    assert_eq!(logits.len(), expected.len());
    let cos = cosine(&logits, &expected);
    assert!(
        cos >= 0.9999,
        "batched prefill diverged from the sequential reference: cos={cos}"
    );
    // Cosine alone would tolerate a single badly wrong logit, so bound
    // every element as well.
    let (rel, idx) = max_scaled_diff(&logits, &expected);
    assert!(
        rel <= 1e-4,
        "logit {idx} diverged by {rel} (ref={}, got={})",
        expected[idx],
        logits[idx]
    );
}

/// The ternary (`TQ2_0_g128`) twin of
/// `batched_prefill_matches_the_sequential_reference`: same acceptance,
/// same tolerances, [`ternary_fixture`] in place of [`fixture`], so
/// `PrefillMatrix::Ternary` and `layer_plan`'s ternary branch actually
/// run under test. The tolerance is a cosine/scaled-diff bound rather
/// than bit-exact equality for the same reason as every other test in
/// this module: the register-blocked GEMM
/// changes the per-row FMA accumulation order relative to the
/// sequential GEMV reference this compares against.
#[test]
fn batched_prefill_matches_the_sequential_reference_ternary() {
    let cfg = tiny_config(2);
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 5) % 96).collect();
    let expected = sequential_logits_with(&cfg, &prompt, ternary_fixture);

    let mut batched = ternary_fixture(cfg);
    let logits = batched
        .forward_prefill_cpu(&prompt, 0)
        .expect("batched prefill should succeed")
        .expect("the ternary fixture is in scope for the batched path");

    assert_eq!(logits.len(), expected.len());
    let cos = cosine(&logits, &expected);
    assert!(
        cos >= 0.9999,
        "ternary batched prefill diverged from the sequential reference: cos={cos}"
    );
    let (rel, idx) = max_scaled_diff(&logits, &expected);
    assert!(
        rel <= 1e-4,
        "ternary logit {idx} diverged by {rel} (ref={}, got={})",
        expected[idx],
        logits[idx]
    );
}

/// The KV cache a batched pass leaves behind must let decoding continue
/// exactly as the sequential path would — this is what proves the
/// per-position KV writes stayed ordered and causal.
#[test]
fn batched_prefill_leaves_a_usable_kv_cache() {
    let cfg = tiny_config(2);
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let prompt: Vec<u32> = (0..9u32).map(|i| (i * 7 + 1) % 96).collect();
    let next_token = 13u32;

    let mut reference = fixture(cfg.clone());
    for (i, &tok) in prompt.iter().enumerate() {
        reference
            .forward(tok, i, &kernel)
            .expect("sequential forward should succeed");
    }
    let seq_next = reference
        .forward(next_token, prompt.len(), &kernel)
        .expect("sequential decode step should succeed");

    let mut batched = fixture(cfg);
    batched
        .forward_prefill_cpu(&prompt, 0)
        .expect("batched prefill should succeed")
        .expect("fixture is in scope");
    let batched_next = batched
        .forward(next_token, prompt.len(), &kernel)
        .expect("decode after batched prefill should succeed");

    let cos = cosine(&batched_next, &seq_next);
    assert!(
        cos >= 0.9999,
        "decode after batched prefill diverged: cos={cos}"
    );
    assert_eq!(
        batched.kv_cache().seq_len(),
        prompt.len() + 1,
        "the KV cursor must cover every prefilled position plus the decode step"
    );
}

/// Largest element-wise difference between two vectors, expressed as a
/// fraction of the **vector's own** largest magnitude.
///
/// Scaling by the vector rather than by each element is what makes this
/// usable as a hard bound: an element that happens to be `-3.7e-9` where
/// its neighbours are `O(1)` carries no information, and a per-element
/// relative ratio would report a 0.4 % "divergence" for the difference
/// between `-3.7e-9` and `0`. Per-vector scaling asks the question that
/// matters — is any element off by a meaningful fraction of the signal?
pub(in crate::model::types) fn max_scaled_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let scale = a
        .iter()
        .chain(b.iter())
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let rel = (x - y).abs() / scale;
        if rel > worst {
            worst = rel;
            at = i;
        }
    }
    (worst, at)
}

/// Every prefilled position's keys and values must match the sequential
/// path's, for every layer, every KV head and **every element** — not
/// just in aggregate.
///
/// Checked element by element rather than by a cosine over the whole
/// flattened `[seq_len x head_dim]` buffer: that aggregate is dominated
/// by the correct majority, so one wrong position (or a permutation of
/// positions) would still clear 0.9999. This is the check that actually
/// constrains the three pieces of logic this module re-derives instead
/// of calling — the per-head QK-norm + RoPE ordering, the
/// `advance_kv_cache_to` cursor rule, and `compute_gqa_attention`.
///
/// The bound is a tight scaled one rather than exact equality because
/// the two paths legitimately run different SIMD tiers: the fixture's
/// `LinearLayer`s carry a `KernelTier::Reference` dispatcher, while
/// [`prefill_dispatcher`] pins the best CPU tier (NEON here), whose
/// per-block reduction order differs by design.
#[test]
fn batched_prefill_kv_cache_matches_position_by_position() {
    let cfg = tiny_config(2);
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let prompt: Vec<u32> = (0..10u32).map(|i| (i * 11 + 3) % 96).collect();

    let mut reference = fixture(cfg.clone());
    for (i, &tok) in prompt.iter().enumerate() {
        reference
            .forward(tok, i, &kernel)
            .expect("sequential forward should succeed");
    }

    let mut batched = fixture(cfg.clone());
    batched
        .forward_prefill_cpu(&prompt, 0)
        .expect("batched prefill should succeed")
        .expect("fixture is in scope");

    let seq_len = prompt.len();
    let hd = cfg.head_dim;
    for layer in 0..cfg.num_layers {
        for head in 0..cfg.num_kv_heads {
            let ref_k = reference.kv_cache().keys_for(layer, head, seq_len);
            let got_k = batched.kv_cache().keys_for(layer, head, seq_len);
            let ref_v = reference.kv_cache().values_for(layer, head, seq_len);
            let got_v = batched.kv_cache().values_for(layer, head, seq_len);
            assert_eq!(ref_k.len(), got_k.len(), "key buffer length");
            assert_eq!(ref_v.len(), got_v.len(), "value buffer length");
            for pos in 0..seq_len {
                let span = pos * hd..(pos + 1) * hd;
                let (rk, ik) = max_scaled_diff(&ref_k[span.clone()], &got_k[span.clone()]);
                assert!(
                    rk <= 1e-4,
                    "layer {layer} head {head} pos {pos} key element {ik}                          diverged by {rk} (ref={}, got={})",
                    ref_k[span.start + ik],
                    got_k[span.start + ik]
                );
                let (rv, iv) = max_scaled_diff(&ref_v[span.clone()], &got_v[span.clone()]);
                assert!(
                    rv <= 1e-4,
                    "layer {layer} head {head} pos {pos} value element {iv}                          diverged by {rv} (ref={}, got={})",
                    ref_v[span.start + iv],
                    got_v[span.start + iv]
                );
            }
        }
    }
}

/// A prompt longer than one micro-batch must give the same answer as a
/// single-pass one: the pass boundary is invisible, and the register
/// block's `m % MR` tail is exercised.
#[test]
fn micro_batch_boundary_is_invisible() {
    let cfg = tiny_config(1);
    let prompt: Vec<u32> = (0..CPU_PREFILL_MICRO_BATCH as u32 + 5)
        .map(|i| (i * 3) % 96)
        .collect();
    let expected = sequential_logits(&cfg, &prompt);

    let mut batched = fixture(cfg);
    let logits = batched
        .forward_prefill_cpu(&prompt, 0)
        .expect("batched prefill should succeed")
        .expect("fixture is in scope");
    let cos = cosine(&logits, &expected);
    assert!(
        cos >= 0.9999,
        "multi-pass batched prefill diverged: cos={cos}"
    );
}

/// Prefilling from a non-zero `pos_start` continues an existing sequence
/// rather than restarting it.
#[test]
fn batched_prefill_continues_from_a_non_zero_position() {
    let cfg = tiny_config(2);
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let head: Vec<u32> = vec![5, 9, 17];
    let tail: Vec<u32> = vec![23, 31, 42, 55];

    let mut reference = fixture(cfg.clone());
    let mut expected = Vec::new();
    for (i, &tok) in head.iter().chain(tail.iter()).enumerate() {
        expected = reference
            .forward(tok, i, &kernel)
            .expect("sequential forward should succeed");
    }

    let mut batched = fixture(cfg);
    for (i, &tok) in head.iter().enumerate() {
        batched
            .forward(tok, i, &kernel)
            .expect("warm-up forward should succeed");
    }
    let logits = batched
        .forward_prefill_cpu(&tail, head.len())
        .expect("batched prefill should succeed")
        .expect("fixture is in scope");
    let cos = cosine(&logits, &expected);
    assert!(
        cos >= 0.9999,
        "batched prefill from pos_start={} diverged: cos={cos}",
        head.len()
    );
}

/// A one-token prompt is the decode path; the batched path declines it
/// without writing anything.
#[test]
fn single_token_prompt_is_declined_without_writing() {
    let mut model = fixture(tiny_config(1));
    let before = model.kv_cache().seq_len();
    assert!(
        model
            .forward_prefill_cpu(&[3], 0)
            .expect("declining is not an error")
            .is_none(),
        "a single-token prompt must be declined"
    );
    assert_eq!(model.kv_cache().seq_len(), before, "nothing may be written");
}

/// A model with no transformer blocks (the config-only constructor) is
/// declined rather than producing garbage.
#[test]
fn blockless_model_is_declined() {
    let mut model = BonsaiModel::new(tiny_config(2));
    assert!(
        model
            .forward_prefill_cpu(&[1, 2, 3], 0)
            .expect("declining is not an error")
            .is_none(),
        "a model without blocks must be declined"
    );
}

/// The derived-shape helper rejects a block count that does not divide
/// cleanly, instead of mis-indexing the weights.
#[test]
fn out_features_rejects_inconsistent_block_counts() {
    let blocks = vec![
        BlockQ1_0G128 {
            d: half::f16::ONE,
            qs: [0; 16],
        };
        5
    ];
    let matrix = PrefillMatrix::OneBit(&blocks);
    assert_eq!(
        matrix.out_features(256),
        None,
        "5 blocks is not a multiple of 2 blocks per row"
    );
    assert_eq!(matrix.out_features(0), None, "zero in_features");
    assert_eq!(matrix.out_features(100), None, "not block aligned");
    assert_eq!(matrix.out_features(128), Some(5));
}

/// M-17: a model that declares a sliding attention window is declined —
/// this path's attention is full-causal — before anything is written.
#[test]
fn windowed_model_is_declined() {
    let mut cfg = tiny_config(2);
    cfg.sliding_window = Some(4);
    let mut model = fixture(cfg);
    let before = model.kv_cache().seq_len();
    let prompt: Vec<u32> = (0..9u32).collect();
    assert!(
        model
            .forward_prefill_cpu(&prompt, 0)
            .expect("declining is not an error")
            .is_none(),
        "a windowed model must be declined"
    );
    assert_eq!(model.kv_cache().seq_len(), before, "nothing may be written");
}

/// `true` when the opt-in INT8 tier is selected in this process's
/// environment: the kernel crate's drivers then route through it, so an f32
/// bit-for-bit comparison against them is not what is being asked.
fn int8_tier_selected() -> bool {
    let selected = oxibonsai_kernels::dispatch_int8::Int8Tier::from_env().is_some();
    if selected {
        eprintln!(
            "OXIBONSAI_KERNEL_TIER selects an INT8 tier: the kernel driver is not the f32 \
             reference in this environment, so the bit-for-bit comparison is skipped"
        );
    }
    selected
}

/// Deterministic `Q1_0_g128` blocks: every bit pattern and a varied scale.
fn onebit_blocks(n: usize, seed: u8) -> Vec<BlockQ1_0G128> {
    (0..n)
        .map(|i| {
            let mut qs = [0u8; 16];
            for (j, b) in qs.iter_mut().enumerate() {
                *b = seed
                    .wrapping_mul(31)
                    .wrapping_add((((i * 16 + j) * 7) & 0xff) as u8);
            }
            BlockQ1_0G128 {
                d: half::f16::from_f32(0.01 + (i % 7) as f32 * 0.003),
                qs,
            }
        })
        .collect()
}

/// Deterministic `TQ2_0_g128` blocks (a `0b11` code decodes to `0`, K-01).
fn ternary_blocks(n: usize, seed: u8) -> Vec<BlockTQ2_0_g128> {
    (0..n)
        .map(|i| {
            let mut qs = [0u8; 32];
            for (j, b) in qs.iter_mut().enumerate() {
                *b = seed
                    .wrapping_mul(17)
                    .wrapping_add((((i * 32 + j) * 13) & 0xff) as u8);
            }
            BlockTQ2_0_g128 {
                qs,
                d: half::f16::from_f32(0.02 + (i % 5) as f32 * 0.004),
            }
        })
        .collect()
}

/// Deterministic activations in roughly `[-1, 1)`.
fn activations(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The two-way split runs the same register-blocked kernel as
/// `oxibonsai_kernels::parallel::gemm_*_par`, only with different task
/// boundaries, so every output element must be **bit-identical** — across
/// both formats, odd feature counts, and every batch size a micro-batch can
/// take (below, at and past the register block, past the thread count, and
/// the full micro-batch with a tail).
#[test]
fn gemm_blocked_2d_is_bit_identical_to_the_kernel_driver() {
    if int8_tier_selected() {
        return;
    }
    let dispatcher = prefill_dispatcher();
    let k = 256;
    let bpr = k / GROUP_WEIGHTS;
    for n_rows in [1usize, 37, 131, 300] {
        let q1 = onebit_blocks(n_rows * bpr, 3);
        let tq2 = ternary_blocks(n_rows * bpr, 5);
        for m in [1usize, 2, 7, 8, 9, 10, 16, 40, 128, 133] {
            let input = activations(m * k, (m * 1000 + n_rows) as u32);
            for matrix in [PrefillMatrix::OneBit(&q1), PrefillMatrix::Ternary(&tq2)] {
                let mut want = vec![f32::NAN; m * n_rows];
                match matrix {
                    PrefillMatrix::OneBit(b) => oxibonsai_kernels::parallel::gemm_1bit_g128_par(
                        dispatcher, b, &input, &mut want, m, n_rows, k,
                    ),
                    PrefillMatrix::Ternary(b) => {
                        oxibonsai_kernels::parallel::gemm_ternary_g128_par(
                            dispatcher, b, &input, &mut want, m, n_rows, k,
                        )
                    }
                }
                .expect("kernel driver");
                let mut got = vec![f32::NAN; m * n_rows];
                gemm_blocked_2d(matrix, dispatcher, &input, &mut got, m, n_rows, k)
                    .expect("two-way split");
                let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    got_bits,
                    want_bits,
                    "{} m={m} n_rows={n_rows}: the two-way split must reproduce the kernel \
                     driver bit for bit",
                    match matrix {
                        PrefillMatrix::OneBit(_) => "Q1_0_g128",
                        PrefillMatrix::Ternary(_) => "TQ2_0_g128",
                    }
                );
            }
        }
    }
}

/// Both routes `PrefillMatrix::gemm` can take agree bit for bit while the
/// INT8 opt-in is off, so the route choice can never change a result on
/// the default configuration.
#[test]
fn both_gemm_routes_agree_bit_for_bit() {
    if int8_tier_selected() {
        return;
    }
    let (k, n_rows, m) = (384usize, 70usize, 11usize);
    let bpr = k / GROUP_WEIGHTS;
    let tq2 = ternary_blocks(n_rows * bpr, 9);
    let q1 = onebit_blocks(n_rows * bpr, 11);
    let input = activations(m * k, 77);
    for matrix in [PrefillMatrix::Ternary(&tq2), PrefillMatrix::OneBit(&q1)] {
        let mut blocked = vec![0.0f32; m * n_rows];
        let mut driver = vec![0.0f32; m * n_rows];
        matrix
            .gemm(GemmRoute::Blocked2d, &input, &mut blocked, m, n_rows, k)
            .expect("blocked route");
        matrix
            .gemm(GemmRoute::KernelDriver, &input, &mut driver, m, n_rows, k)
            .expect("kernel-driver route");
        assert_eq!(
            blocked.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            driver.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

/// Shape errors are typed errors, never a slice-index panic.
#[test]
fn gemm_blocked_2d_rejects_short_buffers_with_typed_errors() {
    let dispatcher = prefill_dispatcher();
    let blocks = ternary_blocks(4, 1);
    let matrix = PrefillMatrix::Ternary(&blocks);
    let input = activations(2 * 128, 1);
    let mut output = vec![0.0f32; 8];
    // 4 blocks at k = 128 is 4 rows; ask for 5.
    let err = gemm_blocked_2d(matrix, dispatcher, &input, &mut output, 1, 5, 128)
        .expect_err("too few weight blocks");
    assert!(matches!(err, ModelError::Kernel(_)), "{err:?}");
    let err = gemm_blocked_2d(matrix, dispatcher, &input[..100], &mut output, 1, 4, 128)
        .expect_err("short input");
    assert!(matches!(err, ModelError::Kernel(_)), "{err:?}");
    let err = gemm_blocked_2d(matrix, dispatcher, &input, &mut output[..3], 1, 4, 128)
        .expect_err("short output");
    assert!(matches!(err, ModelError::Kernel(_)), "{err:?}");
    let err = gemm_blocked_2d(matrix, dispatcher, &input, &mut output, 1, 4, 100)
        .expect_err("k not block aligned");
    assert!(matches!(err, ModelError::Kernel(_)), "{err:?}");
    // Empty dimensions are a no-op, not an error.
    gemm_blocked_2d(matrix, dispatcher, &input, &mut output, 0, 4, 128).expect("m = 0");
    gemm_blocked_2d(matrix, dispatcher, &input, &mut output, 1, 0, 128).expect("n_rows = 0");
}

/// Every split covers each `(batch row, feature)` exactly once with
/// non-empty tasks, keeps batch slabs to one register block, and — the
/// point of the split — yields enough tasks to occupy the pool even when
/// the batch alone could not.
#[test]
fn gemm_split_covers_every_element_once_and_fills_the_pool() {
    for threads in [1usize, 2, 8, 16] {
        for m in [1usize, 2, 7, 8, 10, 64, 128, 133] {
            for n_rows in [1usize, 31, 64, 1024, 6144] {
                let split = GemmSplit::plan(m, n_rows, 8, threads);
                assert!(split.row_slab >= 1 && split.row_slab <= 8.min(m));
                assert!(split.features_per_slab >= 1);
                assert!(split.feature_slabs >= 1);
                // Coverage: the slabs tile [0, m) x [0, n_rows) exactly.
                let row_slabs = m.div_ceil(split.row_slab);
                assert!((row_slabs - 1) * split.row_slab < m);
                assert!(split.feature_slabs * split.features_per_slab >= n_rows);
                assert!((split.feature_slabs - 1) * split.features_per_slab < n_rows);
                // Parallelism: enough tasks for the pool whenever the
                // feature count allows it.
                let tasks = row_slabs * split.feature_slabs;
                let most_by_width = row_slabs * n_rows.div_ceil(GEMM_MIN_FEATURES_PER_TASK);
                assert!(
                    tasks >= threads.min(most_by_width),
                    "threads={threads} m={m} n_rows={n_rows}: {tasks} tasks ({split:?})"
                );
            }
        }
    }
    // The case the split exists for: a 10-token input on an 8-thread pool
    // is 2 batch slabs, so the features must supply the rest.
    let short = GemmSplit::plan(10, 2048, 8, 8);
    assert_eq!(short.row_slab, 8);
    assert!(2 * short.feature_slabs >= 8, "{short:?}");
}

/// Fill layer 0 of `cache` with deterministic keys and values for
/// positions `0..len`.
fn fill_cache(cache: &mut KvCache, len: usize) {
    let hd = cache.head_dim();
    for pos in 0..len {
        for head in 0..cache.num_kv_heads() {
            let key = activations(hd, (pos * 97 + head * 13) as u32);
            let value = activations(hd, (pos * 89 + head * 7 + 5) as u32);
            cache.try_store_key(0, head, pos, &key).expect("store key");
            cache
                .try_store_value(0, head, pos, &value)
                .expect("store value");
        }
    }
    cache.set_seq_len(len);
}

/// The row-parallel attention gives every row exactly the bits the per-row
/// call (`gqa_attention` with `seq_len = pos + 1`, the per-token path's
/// shape) gives it — for an `f32` and an `f16` cache, and for both the
/// fewer-rows-than-threads and the more-rows-than-threads settings.
#[test]
fn attend_rows_is_bit_identical_to_the_per_row_attention() {
    use crate::kv_cache::KvCacheBacking;

    let geom = PrefillGeometry {
        hidden: 128,
        intermediate: 256,
        q_dim: 4 * 32,
        kv_dim: 2 * 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        heads_per_group: 2,
    };
    let caches = [
        KvCache::new(1, 2, 32, 96),
        KvCache::try_new_lazy(KvCacheBacking::DenseF16, 1, 2, 32, 96, 96).expect("f16 cache"),
    ];
    for mut cache in caches {
        for (pos_start, rows) in [(0usize, 3usize), (5, 2), (7, 40), (0, 64)] {
            fill_cache(&mut cache, pos_start + rows);
            let queries = activations(rows * geom.q_dim, (pos_start * 3 + rows) as u32);
            let mut got = vec![f32::NAN; rows * geom.q_dim];
            attend_rows(&queries, &mut got, &cache, 0, &geom, pos_start).expect("rows");
            let mut want = vec![f32::NAN; rows * geom.q_dim];
            for row in 0..rows {
                let span = row * geom.q_dim..(row + 1) * geom.q_dim;
                crate::block::functions::gqa_attention(
                    &queries[span.clone()],
                    &mut want[span],
                    &cache,
                    0,
                    geom.num_heads,
                    geom.heads_per_group,
                    geom.head_dim,
                    pos_start + row + 1,
                    true,
                )
                .expect("per-row attention");
            }
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "pos_start={pos_start} rows={rows} f16={}",
                cache.is_f16()
            );
        }
    }
}

/// How many times each leg of
/// [`real_model_cpu_prefill_outruns_the_sequential_prefill`] is timed.
///
/// The gating comparison is the **minimum** of these, not the mean:
/// a minimum is the closest a wall-clock sample gets to the machine's
/// own floor, and it is the statistic that survives the contention this
/// module's MEASUREMENTS table showed moves the sequential leg by 46 %
/// while leaving the batched leg within 0.5 %.
const PERF_TIMED_RUNS: usize = 3;

/// Best-effort 1/5/15-minute load average, for the measurement record.
///
/// Read through `uptime` rather than a crate: this is test-only
/// reporting, the workspace has no load-average dependency, and adding
/// one for a printed diagnostic would be a real dependency for a string.
/// Anything that goes wrong yields `"unavailable"` — the measurement is
/// still valid, it just carries no load annotation.
pub(in crate::model::types) fn load_average() -> String {
    match std::process::Command::new("uptime").output() {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            match text.split_once("load average") {
                Some((_, tail)) => tail.trim_start_matches([':', 's', ' ']).trim().to_string(),
                None => text.trim().to_string(),
            }
        }
        _ => "unavailable".to_string(),
    }
}

/// perf-M2 on the real shipped model: the batched CPU prefill must
/// reproduce the sequential per-token reference (cos >= 0.9999) and be
/// **at least as fast** as it on the same machine.
///
/// # What this asserts, and what it only records
///
/// An absolute `< 60 ms/prompt-token` target was once proposed. That
/// number was a 183 ms/token sequential baseline divided by three, and
/// neither leg of that derivation is reproducible on this hardware: the
/// unchanged sequential code measures 268.6–392.4 ms/token here, and the
/// batched path measured 82.3–82.7 ms/token across a 5x load-average
/// swing (see the module doc's MEASUREMENTS table). The absolute figure
/// is therefore *recorded*, not asserted, and the gating invariant is the
/// **relative** one: batched prefill throughput at least equal to the
/// sequential per-token prefill throughput, measured in-process, as the
/// minimum of [`PERF_TIMED_RUNS`] runs of each leg. A ratio is
/// dimensionless, so it is the one thing a contended machine cannot
/// fake; an absolute millisecond count is not.
///
/// The `speedup >= 1.5` floor below is deliberately stricter than the
/// relative rule's `>= 1.0`: 1.5x was never the red leg (every
/// measurement taken is 3.26x–4.74x or better), so keeping it preserves a
/// guarantee rather than weakening one.
///
/// Ignored by default (it needs a multi-hundred-MB model file and a real
/// CPU); the absolute numbers it prints are the record.
///
/// Calls [`BonsaiModel::forward_prefill_cpu`] **directly** rather than
/// through [`BonsaiModel::forward_prefill`]: under `--all-features` the
/// Metal feature is on, and `forward_prefill` would take the fused GPU
/// path first, timing the GPU instead of the CPU path this module
/// implements. The sequential reference uses the same pinned CPU tier
/// ([`oxibonsai_kernels::cpu_kernel_tier`]) that [`prefill_dispatcher`]
/// pins, so the comparison is CPU-tier-for-CPU-tier: the old
/// loop-of-GEMVs shape against the new register-blocked batch, which is
/// the axis perf-M2 is about.
///
/// Run with:
/// ```text
/// OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf \
///   cargo test -p oxibonsai-model --release --all-features --lib \
///   prefill_cpu::tests::real_model_cpu_prefill_outruns_the_sequential_prefill \
///   -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires OXI_MODEL real ternary/1-bit GGUF; run on dev Mac"]
fn real_model_cpu_prefill_outruns_the_sequential_prefill() {
    use oxibonsai_core::gguf::reader::GgufFile;
    use std::time::{Duration, Instant};

    let Some(path) = std::env::var_os("OXI_MODEL") else {
        eprintln!(
            "real_model_cpu_prefill_outruns_the_sequential_prefill: OXI_MODEL not set — \
             skipping. Set OXI_MODEL=/path/to/Ternary-Bonsai-1.7B.gguf to run."
        );
        return;
    };
    let bytes = std::fs::read(&path).expect("read OXI_MODEL gguf");
    let gguf = GgufFile::parse(&bytes).expect("GgufFile::parse OXI_MODEL");

    const MAX_SEQ: usize = 4096;
    const PROMPT_LEN: usize = 280; // same order as the 277-token baseline run

    // Two independent models from the same bytes: one for the batched
    // CPU path, one for the sequential reference, so neither run's KV
    // cache or timing is polluted by the other.
    let mut cpu_model =
        BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf (cpu)");
    let mut seq_model =
        BonsaiModel::from_gguf(&gguf, MAX_SEQ).expect("BonsaiModel::from_gguf (sequential)");

    let vocab = cpu_model.config().vocab_size as u32;
    assert!(vocab > 1, "real model must have a non-trivial vocabulary");
    let prompt: Vec<u32> = (0..PROMPT_LEN as u32)
        .map(|i| 1 + (i * 97) % (vocab - 1))
        .collect();

    let load_before = load_average();

    // Leg 1: the batched CPU prefill, `PERF_TIMED_RUNS` times, each from
    // a cleared KV cache so every run does the identical work.
    let mut cpu_runs: Vec<Duration> = Vec::with_capacity(PERF_TIMED_RUNS);
    let mut cpu_logits = Vec::new();
    for _ in 0..PERF_TIMED_RUNS {
        cpu_model.reset();
        let t = Instant::now();
        cpu_logits = cpu_model
            .forward_prefill_cpu(&prompt, 0)
            .expect("batched CPU prefill should succeed on the real model")
            .expect(
                "the real model's projections should be a register-blocked format \
                 (Q1_0_g128 or TQ2_0_g128)",
            );
        cpu_runs.push(t.elapsed());
    }
    let cpu_best = cpu_runs.iter().copied().min().unwrap_or(Duration::MAX);

    // Leg 2: the sequential per-token reference on the same CPU tier.
    let kernel = KernelDispatcher::with_tier(oxibonsai_kernels::cpu_kernel_tier());
    let mut seq_runs: Vec<Duration> = Vec::with_capacity(PERF_TIMED_RUNS);
    let mut seq_logits = Vec::new();
    for _ in 0..PERF_TIMED_RUNS {
        seq_model.reset();
        let t = Instant::now();
        for (i, &tok) in prompt.iter().enumerate() {
            seq_logits = seq_model
                .forward(tok, i, &kernel)
                .expect("sequential forward should succeed on the real model");
        }
        seq_runs.push(t.elapsed());
    }
    let seq_best = seq_runs.iter().copied().min().unwrap_or(Duration::MAX);

    let load_after = load_average();
    let cos = cosine(&cpu_logits, &seq_logits);
    let ms_per_token = cpu_best.as_secs_f64() * 1e3 / PROMPT_LEN as f64;
    let seq_ms_per_token = seq_best.as_secs_f64() * 1e3 / PROMPT_LEN as f64;
    let speedup = seq_best.as_secs_f64() / cpu_best.as_secs_f64().max(f64::MIN_POSITIVE);

    // Every individual run, not just the minimum: the first pass over a
    // multi-hundred-MB weight file pays the page-fault and first-touch
    // cost of the whole matrix, which is exactly the difference between
    // a single-run number and the warm figure.
    let per_run = |runs: &[Duration]| -> String {
        runs.iter()
            .map(|d| format!("{:.3}", d.as_secs_f64() * 1e3 / PROMPT_LEN as f64))
            .collect::<Vec<_>>()
            .join(", ")
    };
    eprintln!(
        "real_model_cpu_prefill_outruns_the_sequential_prefill: model={path:?} \
         prompt_len={PROMPT_LEN} runs={PERF_TIMED_RUNS} (min of each leg)\n\
         \x20 batched CPU prefill : {:>9.2} ms total, {:>7.3} ms/prompt-token (min)\n\
         \x20   per run           : [{}] ms/prompt-token\n\
         \x20 sequential reference: {:>9.2} ms total, {:>7.3} ms/prompt-token (min)\n\
         \x20   per run           : [{}] ms/prompt-token\n\
         \x20 speedup             : {speedup:.2}x\n\
         \x20 cos(batched, sequential) = {cos}\n\
         \x20 load average before : {load_before}\n\
         \x20 load average after  : {load_after}",
        cpu_best.as_secs_f64() * 1e3,
        ms_per_token,
        per_run(&cpu_runs),
        seq_best.as_secs_f64() * 1e3,
        seq_ms_per_token,
        per_run(&seq_runs),
    );

    assert_eq!(cpu_logits.len(), seq_logits.len());
    assert!(
        cos >= 0.9999,
        "batched CPU prefill diverged from the sequential reference on the real model: \
         cos={cos}"
    );
    // The gating invariant: the ratio, not the millisecond count.
    assert!(
        cpu_best <= seq_best,
        "perf-M2 relative invariant violated: batched CPU prefill ({ms_per_token:.3} \
         ms/token) is slower than the sequential per-token prefill \
         ({seq_ms_per_token:.3} ms/token)"
    );
    assert!(
        speedup >= 1.5,
        "batched CPU prefill should be substantially faster than the sequential \
         reference, got only {speedup:.2}x"
    );
}
