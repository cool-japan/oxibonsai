//! Tests for model construction and forward pass behavior.

use half::f16;
use oxibonsai_core::config::{Qwen3Config, RopeScaling};
use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::dispatch::{KernelDispatcher, KernelTier};
use oxibonsai_model::block::TransformerBlock;
use oxibonsai_model::kv_cache::KvCache;
use oxibonsai_model::layers::linear::Linear1Bit;
use oxibonsai_model::layers::rms_norm::RmsNorm;
use oxibonsai_model::layers::rope::RopeTable;
use oxibonsai_model::model::BonsaiModel;

fn make_blocks(n: usize, scale: f32, pattern: u8) -> Vec<BlockQ1_0G128> {
    (0..n)
        .map(|_| BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: [pattern; 16],
        })
        .collect()
}

fn ref_kernel() -> std::sync::Arc<KernelDispatcher> {
    std::sync::Arc::new(KernelDispatcher::with_tier(KernelTier::Reference))
}

/// Build a small Transformer block for testing with given dimensions.
#[allow(clippy::too_many_arguments)]
fn build_test_block<'a>(
    layer_idx: usize,
    h: usize,
    hd: usize,
    nq: usize,
    nkv: usize,
    inter: usize,
    scale: f32,
    pattern: u8,
    q_blocks: &'a [BlockQ1_0G128],
    k_blocks: &'a [BlockQ1_0G128],
    v_blocks: &'a [BlockQ1_0G128],
    o_blocks: &'a [BlockQ1_0G128],
    gate_blocks: &'a [BlockQ1_0G128],
    up_blocks: &'a [BlockQ1_0G128],
    down_blocks: &'a [BlockQ1_0G128],
) -> TransformerBlock<'a> {
    let _ = (scale, pattern); // used indirectly through blocks
    let kernel = ref_kernel();
    TransformerBlock::new(
        layer_idx,
        RmsNorm::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(q_blocks, nq * hd, h, kernel.clone())
            .expect("q")
            .into(),
        Linear1Bit::new(k_blocks, nkv * hd, h, kernel.clone())
            .expect("k")
            .into(),
        Linear1Bit::new(v_blocks, nkv * hd, h, kernel.clone())
            .expect("v")
            .into(),
        Linear1Bit::new(o_blocks, h, nq * hd, kernel.clone())
            .expect("o")
            .into(),
        RmsNorm::new(vec![1.0; hd], 1e-6),
        RmsNorm::new(vec![1.0; hd], 1e-6),
        RmsNorm::new(vec![1.0; h], 1e-6),
        Linear1Bit::new(gate_blocks, inter, h, kernel.clone())
            .expect("gate")
            .into(),
        Linear1Bit::new(up_blocks, inter, h, kernel.clone())
            .expect("up")
            .into(),
        Linear1Bit::new(down_blocks, h, inter, kernel.clone())
            .expect("down")
            .into(),
        nq,
        nkv,
        hd,
        h,
    )
}

// ══════════════════════════════════════════════════════════════
// Model creation tests
// ══════════════════════════════════════════════════════════════

// HOTFIX-TESTMEM: `BonsaiModel::new(Qwen3Config::bonsai_8b())` allocates ~5 GB
// of token_embd + output_weight tables (plus a ~1.2 GB KV cache) — for tests
// that only check the config was carried through unchanged. `tiny_test()`
// exercises the identical constructor for a few tens of MB.

#[test]
fn model_new_creates_valid_model() {
    let config = Qwen3Config::tiny_test();
    let model = BonsaiModel::new(config);
    assert_eq!(model.config().hidden_size, 64);
    assert_eq!(model.config().num_layers, 2);
    assert_eq!(model.config().vocab_size, 151936);
}

#[test]
fn model_new_has_empty_blocks() {
    let config = Qwen3Config::tiny_test();
    let model = BonsaiModel::new(config);
    // BonsaiModel::new creates an empty blocks vec for testing
    // We verify the config is correct
    assert_eq!(model.config().num_attention_heads, 4);
    assert_eq!(model.config().num_kv_heads, 2);
}

#[test]
fn model_forward_produces_logits_of_vocab_size() {
    // Use a small config for testing
    let config = Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 0, // no blocks for speed
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        value_length: 64,
        vocab_size: 100,
        max_context_length: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "test".to_string(),
        model_name: "test".to_string(),
    };

    let mut model = BonsaiModel::new(config);
    let kernel = ref_kernel();

    // With 0 blocks, forward should still work (embedding + norm + output projection)
    let logits = model
        .forward(0, 0, kernel.as_ref())
        .expect("forward should succeed with empty blocks");
    assert_eq!(logits.len(), 100, "logits should match vocab_size");
}

#[test]
fn model_forward_deterministic() {
    let config = Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 0,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        value_length: 64,
        vocab_size: 50,
        max_context_length: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "test".to_string(),
        model_name: "test".to_string(),
    };

    let mut model1 = BonsaiModel::new(config.clone());
    let mut model2 = BonsaiModel::new(config);
    let kernel = ref_kernel();

    let logits1 = model1.forward(0, 0, kernel.as_ref()).expect("forward 1");
    let logits2 = model2.forward(0, 0, kernel.as_ref()).expect("forward 2");

    assert_eq!(logits1.len(), logits2.len());
    for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "logits should be identical: index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn model_reset_clears_kv_cache() {
    let config = Qwen3Config {
        hidden_size: 128,
        intermediate_size: 256,
        num_layers: 0,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        value_length: 64,
        vocab_size: 50,
        max_context_length: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        rope_scaling: RopeScaling::None,
        sliding_window: None,
        architecture: "test".to_string(),
        model_name: "test".to_string(),
    };

    let mut model = BonsaiModel::new(config);
    let kernel = ref_kernel();

    let _ = model.forward(0, 0, kernel.as_ref()).expect("forward");
    model.reset();
    assert_eq!(model.kv_cache_mut().seq_len(), 0);
}

// ══════════════════════════════════════════════════════════════
// TransformerBlock forward tests
// ══════════════════════════════════════════════════════════════

#[test]
fn transformer_block_forward_changes_hidden_state() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_h = h / 128;

    let q_blocks = make_blocks(nq * hd * blocks_per_h, 0.01, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_h, 0.01, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

    let block = build_test_block(
        0,
        h,
        hd,
        nq,
        nkv,
        inter,
        0.01,
        0xFF,
        &q_blocks,
        &k_blocks,
        &v_blocks,
        &o_blocks,
        &gate_blocks,
        &up_blocks,
        &down_blocks,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = ref_kernel();
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let original = hidden.clone();

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
        .expect("block forward");

    let max_diff = hidden
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-6,
        "forward should change hidden state, max_diff={max_diff}"
    );
}

#[test]
fn transformer_block_residual_connection_preserves_input_contribution() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_h = h / 128;

    // Use very small scale so sublayer outputs are small
    let q_blocks = make_blocks(nq * hd * blocks_per_h, 0.001, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_h, 0.001, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_h, 0.001, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_h, 0.001, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_h, 0.001, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_h, 0.001, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.001, 0xFF);

    let block = build_test_block(
        0,
        h,
        hd,
        nq,
        nkv,
        inter,
        0.001,
        0xFF,
        &q_blocks,
        &k_blocks,
        &v_blocks,
        &o_blocks,
        &gate_blocks,
        &up_blocks,
        &down_blocks,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = ref_kernel();
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    let mut hidden: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let original = hidden.clone();

    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
        .expect("block forward");

    // With very small weights, the residual contribution from input dominates
    // So output should be close to input (but not identical)
    let correlation: f32 = hidden
        .iter()
        .zip(original.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>()
        / (hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
            * original.iter().map(|x| x * x).sum::<f32>().sqrt());

    assert!(
        correlation > 0.9,
        "with small weights, output should be highly correlated with input: corr={correlation}"
    );
}

#[test]
fn rope_produces_position_dependent_qk_vectors() {
    // Test that RoPE at different positions produces different Q/K vectors,
    // which is the mechanism by which Transformer blocks distinguish positions.
    let hd = 64;
    let rope = RopeTable::new(hd, 16, 10000.0);

    let input = vec![1.0f32; hd];
    let mut out_pos0 = vec![0.0f32; hd];
    let mut out_pos5 = vec![0.0f32; hd];

    rope.apply(&input, &mut out_pos0, 0).expect("pos 0");
    rope.apply(&input, &mut out_pos5, 5).expect("pos 5");

    let diff: f32 = out_pos0
        .iter()
        .zip(out_pos5.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff > 1e-3,
        "RoPE at different positions should produce different outputs: diff={diff}"
    );
}

#[test]
fn kv_cache_accumulates_across_forward_calls() {
    let h = 128;
    let hd = 64;
    let nq = 2;
    let nkv = 1;
    let inter = 256;
    let blocks_per_h = h / 128;

    let q_blocks = make_blocks(nq * hd * blocks_per_h, 0.01, 0xFF);
    let k_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let v_blocks = make_blocks(nkv * hd * blocks_per_h, 0.01, 0xFF);
    let o_blocks = make_blocks(h * blocks_per_h, 0.01, 0xFF);
    let gate_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let up_blocks = make_blocks(inter * blocks_per_h, 0.01, 0xFF);
    let down_blocks = make_blocks(h * (inter / 128), 0.01, 0xFF);

    let block = build_test_block(
        0,
        h,
        hd,
        nq,
        nkv,
        inter,
        0.01,
        0xFF,
        &q_blocks,
        &k_blocks,
        &v_blocks,
        &o_blocks,
        &gate_blocks,
        &up_blocks,
        &down_blocks,
    );

    let rope = RopeTable::new(hd, 16, 10000.0);
    let kernel = ref_kernel();
    let mut kv_cache = KvCache::new(1, nkv, hd, 16);

    // Forward at position 0
    let mut hidden = vec![0.1f32; h];
    block
        .forward(&mut hidden, 0, &mut kv_cache, &rope, kernel.as_ref())
        .expect("pos 0");

    // Check that keys were stored at position 0
    let keys_after_0 = kv_cache.keys_for(0, 0, 1);
    assert_eq!(keys_after_0.len(), hd);
    // At least some values should be non-zero
    let has_nonzero = keys_after_0.iter().any(|&v| v.abs() > 1e-10);
    assert!(
        has_nonzero,
        "KV cache should have non-zero values after forward"
    );

    // Forward at position 1
    let mut hidden2 = vec![0.2f32; h];
    block
        .forward(&mut hidden2, 1, &mut kv_cache, &rope, kernel.as_ref())
        .expect("pos 1");

    // Now KV cache should have 2 positions
    let keys_after_1 = kv_cache.keys_for(0, 0, 2);
    assert_eq!(keys_after_1.len(), 2 * hd);
}

#[test]
fn model_config_bonsai_8b_defaults() {
    let config = Qwen3Config::bonsai_8b();
    assert_eq!(config.hidden_size, 4096);
    // Real GGUF header values (models/*Bonsai-8B.gguf): intermediate 12288,
    // vocab 151669.
    assert_eq!(config.intermediate_size, 12288);
    assert_eq!(config.num_layers, 36);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 151_669);
    assert!((config.rms_norm_eps - 1e-6).abs() < 1e-10);
    assert!((config.rope_freq_base - 1_000_000.0).abs() < 1.0);
}

// ══════════════════════════════════════════════════════════════
// M-17: `<arch>.attention.sliding_window`, wired end to end
//
// These are the production-path tests for the sliding-window forward: a real
// GGUF declaring the key must drive
// `TransformerBlock::forward_with_sliding_window` through
// `BonsaiModel::forward`, and a GGUF without it must keep the fully-causal
// path it has always taken.
// ══════════════════════════════════════════════════════════════

/// Hidden size of the sliding-window fixture (a multiple of 128, as
/// `Q1_0_g128` requires).
const SW_H: usize = 128;
/// FFN width of the sliding-window fixture (multiple of 128).
const SW_INTER: usize = 256;
/// Layers in the sliding-window fixture.
const SW_LAYERS: usize = 2;
/// Query heads of the sliding-window fixture.
const SW_NQ: usize = 4;
/// KV heads of the sliding-window fixture (genuine GQA: 4 over 2).
const SW_NKV: usize = 2;
/// Head dimension of the sliding-window fixture.
const SW_HD: usize = 32;
/// Vocabulary of the sliding-window fixture.
const SW_VOCAB: usize = 32;

/// Deterministic `Q1_0_g128` weight bytes: `[scale f16 LE][qs 16 B]` per 128
/// weights, in the on-disk AoS layout the GGUF reader expects.
fn sw_q1_pattern(num_weights: usize, seed: u64) -> Vec<u8> {
    assert_eq!(num_weights % 128, 0);
    let mut data = Vec::with_capacity(num_weights / 128 * 18);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..num_weights / 128 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let scale = 0.25_f32 + ((state >> 33) as u32 as f32) / (u32::MAX as f32) * 0.5_f32;
        data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for _ in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            data.push((state >> 33) as u8);
        }
    }
    data
}

/// Deterministic FP32 tensor bytes.
fn sw_f32_pattern(n: usize, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    for i in 0..n {
        let value = scale * (1.0_f32 + 0.25_f32 * ((i as f32) * 0.013_f32).sin());
        v.extend_from_slice(&value.to_le_bytes());
    }
    v
}

/// Build a small dense-Qwen3 GGUF, optionally declaring
/// `qwen3.attention.sliding_window`.
///
/// Everything except that one key is byte-identical between the two variants,
/// so any difference in the forward pass is attributable to the window alone.
fn build_sliding_window_gguf(sliding_window: Option<u32>) -> Vec<u8> {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};

    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".to_string()),
    );
    writer.add_metadata(
        "general.name",
        MetadataWriteValue::Str("SlidingWindowFixture".to_string()),
    );
    writer.add_metadata(
        "qwen3.embedding_length",
        MetadataWriteValue::U32(SW_H as u32),
    );
    writer.add_metadata(
        "qwen3.block_count",
        MetadataWriteValue::U32(SW_LAYERS as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count",
        MetadataWriteValue::U32(SW_NQ as u32),
    );
    writer.add_metadata(
        "qwen3.attention.head_count_kv",
        MetadataWriteValue::U32(SW_NKV as u32),
    );
    writer.add_metadata(
        "qwen3.feed_forward_length",
        MetadataWriteValue::U32(SW_INTER as u32),
    );
    writer.add_metadata("qwen3.vocab_size", MetadataWriteValue::U32(SW_VOCAB as u32));
    writer.add_metadata("qwen3.context_length", MetadataWriteValue::U32(512));
    writer.add_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        MetadataWriteValue::F32(1e-6),
    );
    writer.add_metadata("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0));
    if let Some(window) = sliding_window {
        writer.add_metadata(
            "qwen3.attention.sliding_window",
            MetadataWriteValue::U32(window),
        );
    }

    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![SW_H as u64, SW_VOCAB as u64],
        tensor_type: TensorType::F32,
        data: sw_f32_pattern(SW_VOCAB * SW_H, 0.5),
    });
    writer.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![SW_H as u64],
        tensor_type: TensorType::F32,
        data: sw_f32_pattern(SW_H, 1.0),
    });
    writer.add_tensor(TensorEntry {
        name: "output.weight".to_string(),
        shape: vec![SW_H as u64, SW_VOCAB as u64],
        tensor_type: TensorType::F32,
        data: sw_f32_pattern(SW_VOCAB * SW_H, 0.05),
    });

    for layer in 0..SW_LAYERS {
        let pfx = format!("blk.{layer}");
        for (name, len) in [
            (format!("{pfx}.attn_norm.weight"), SW_H),
            (format!("{pfx}.ffn_norm.weight"), SW_H),
            (format!("{pfx}.attn_q_norm.weight"), SW_HD),
            (format!("{pfx}.attn_k_norm.weight"), SW_HD),
        ] {
            writer.add_tensor(TensorEntry {
                name,
                shape: vec![len as u64],
                tensor_type: TensorType::F32,
                data: sw_f32_pattern(len, 1.0),
            });
        }
        let seed = 0x3000_0000_u64.wrapping_add((layer as u64) << 16);
        for (name, shape, weights, bump) in [
            (
                format!("{pfx}.attn_q.weight"),
                vec![SW_H as u64, (SW_NQ * SW_HD) as u64],
                SW_NQ * SW_HD * SW_H,
                0,
            ),
            (
                format!("{pfx}.attn_k.weight"),
                vec![SW_H as u64, (SW_NKV * SW_HD) as u64],
                SW_NKV * SW_HD * SW_H,
                1,
            ),
            (
                format!("{pfx}.attn_v.weight"),
                vec![SW_H as u64, (SW_NKV * SW_HD) as u64],
                SW_NKV * SW_HD * SW_H,
                2,
            ),
            (
                format!("{pfx}.attn_output.weight"),
                vec![(SW_NQ * SW_HD) as u64, SW_H as u64],
                SW_H * SW_NQ * SW_HD,
                3,
            ),
            (
                format!("{pfx}.ffn_gate.weight"),
                vec![SW_H as u64, SW_INTER as u64],
                SW_INTER * SW_H,
                4,
            ),
            (
                format!("{pfx}.ffn_up.weight"),
                vec![SW_H as u64, SW_INTER as u64],
                SW_INTER * SW_H,
                5,
            ),
            (
                format!("{pfx}.ffn_down.weight"),
                vec![SW_INTER as u64, SW_H as u64],
                SW_H * SW_INTER,
                6,
            ),
        ] {
            writer.add_tensor(TensorEntry {
                name,
                shape,
                tensor_type: TensorType::Q1_0G128,
                data: sw_q1_pattern(weights, seed.wrapping_add(bump)),
            });
        }
    }

    writer.to_bytes().expect("GgufWriter::to_bytes")
}

/// Run `prompt` token-by-token through a fresh model built from `bytes`,
/// returning the last position's logits.
fn sw_sequential_logits(bytes: &[u8], prompt: &[u32]) -> Vec<f32> {
    use oxibonsai_core::gguf::reader::GgufFile;
    let gguf = GgufFile::parse(bytes).expect("parse sliding-window fixture");
    let mut model = BonsaiModel::from_gguf(&gguf, 256).expect("from_gguf");
    let kernel = ref_kernel();
    let mut last = Vec::new();
    for (i, &tid) in prompt.iter().enumerate() {
        last = model.forward(tid, i, kernel.as_ref()).expect("forward");
    }
    last
}

/// `forward_prefill` over the whole prompt, returning the last position's
/// logits.
fn sw_prefill_logits(bytes: &[u8], prompt: &[u32]) -> Vec<f32> {
    use oxibonsai_core::gguf::reader::GgufFile;
    let gguf = GgufFile::parse(bytes).expect("parse sliding-window fixture");
    let mut model = BonsaiModel::from_gguf(&gguf, 256).expect("from_gguf");
    let kernel = ref_kernel();
    model
        .forward_prefill(prompt, 0, kernel.as_ref())
        .expect("forward_prefill")
}

fn sw_max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// A GGUF **without** the key keeps `sliding_window == None`, so the forward
/// pass takes the unchanged fully-causal branch.
#[test]
fn gguf_without_sliding_window_key_stays_fully_causal() {
    use oxibonsai_core::gguf::reader::GgufFile;
    let bytes = build_sliding_window_gguf(None);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let model = BonsaiModel::from_gguf(&gguf, 256).expect("from_gguf");
    assert_eq!(model.config().sliding_window, None);
}

/// A GGUF **with** the key carries it into the config, which is what
/// `BonsaiModel::forward` branches on.
#[test]
fn gguf_with_sliding_window_key_carries_it_into_the_config() {
    use oxibonsai_core::gguf::reader::GgufFile;
    let bytes = build_sliding_window_gguf(Some(4));
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let model = BonsaiModel::from_gguf(&gguf, 256).expect("from_gguf");
    assert_eq!(model.config().sliding_window, Some(4));
}

/// The acceptance test: a window **smaller than the prompt** must change the
/// output, which can only happen if `forward_with_sliding_window` actually
/// ran — reached through `BonsaiModel::forward`, i.e. in production, not only
/// from a block-level unit test.
#[test]
fn small_sliding_window_changes_the_logits_in_production() {
    let prompt: Vec<u32> = (0..12_u32).map(|i| (i * 5 + 3) % SW_VOCAB as u32).collect();

    let unwindowed = sw_sequential_logits(&build_sliding_window_gguf(None), &prompt);
    let windowed = sw_sequential_logits(&build_sliding_window_gguf(Some(3)), &prompt);

    assert_eq!(unwindowed.len(), SW_VOCAB);
    let diff = sw_max_abs_diff(&unwindowed, &windowed);
    assert!(
        diff > 1e-6,
        "a 3-token window over a 12-token prompt must change the logits \
         (max |delta| = {diff}); if it does not, `forward_with_sliding_window` \
         is not being reached from `BonsaiModel::forward`"
    );
}

/// Narrowing the window further must keep changing the result: this rules out
/// "some unrelated difference" as the explanation for the test above.
#[test]
fn narrower_sliding_windows_keep_changing_the_logits() {
    let prompt: Vec<u32> = (0..12_u32).map(|i| (i * 5 + 3) % SW_VOCAB as u32).collect();
    let w2 = sw_sequential_logits(&build_sliding_window_gguf(Some(2)), &prompt);
    let w6 = sw_sequential_logits(&build_sliding_window_gguf(Some(6)), &prompt);
    assert!(
        sw_max_abs_diff(&w2, &w6) > 1e-6,
        "window 2 and window 6 must not produce the same logits"
    );
}

/// A window at least as wide as the sequence admits every key position, so it
/// must reproduce full causal attention (up to the reassociation the windowed
/// path's gathered-buffer attention introduces).
#[test]
fn sliding_window_wider_than_the_prompt_matches_full_attention() {
    let prompt: Vec<u32> = (0..8_u32).map(|i| (i * 7 + 1) % SW_VOCAB as u32).collect();
    let unwindowed = sw_sequential_logits(&build_sliding_window_gguf(None), &prompt);
    let windowed = sw_sequential_logits(&build_sliding_window_gguf(Some(64)), &prompt);
    let diff = sw_max_abs_diff(&unwindowed, &windowed);
    // Relative bound (tightened from an absolute `1e-3` during wave-3.5
    // verifier review): an off-by-one in the gather-and-attend window path
    // must not be able to hide under a loose absolute tolerance. The review
    // independently re-verified `<= 1e-5 * max|logit|` against this exact
    // fixture family (plus a 1-token/window=1 bit-exact check and a
    // window=1-vs-window=2 divergence check) before recommending it.
    let scale = unwindowed.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let bound = 1e-5 * scale;
    assert!(
        diff <= bound,
        "a window wider than the sequence must reproduce full attention, \
         max |delta| = {diff} (bound = {bound}, max |logit| = {scale})"
    );
}

/// The batched prefill paths are all fully causal, so a windowed model must
/// not take them: `forward_prefill` has to agree with the *windowed*
/// per-token loop exactly, and must not produce the fully-causal answer.
#[test]
fn windowed_prefill_falls_back_to_the_windowed_sequential_path() {
    let prompt: Vec<u32> = (0..12_u32).map(|i| (i * 5 + 3) % SW_VOCAB as u32).collect();
    let bytes = build_sliding_window_gguf(Some(3));

    let sequential = sw_sequential_logits(&bytes, &prompt);
    let prefilled = sw_prefill_logits(&bytes, &prompt);
    for (i, (a, b)) in sequential.iter().zip(prefilled.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "windowed prefill must be bit-identical to the windowed sequential \
             forward at index {i}: {a} vs {b}"
        );
    }

    // ... and it must NOT match the fully-causal answer, which is what a
    // batched prefill path would have produced had it claimed the prompt.
    let full = sw_sequential_logits(&build_sliding_window_gguf(None), &prompt);
    assert!(
        sw_max_abs_diff(&full, &prefilled) > 1e-6,
        "windowed prefill produced the fully-causal logits: a batched prefill \
         path claimed a windowed model"
    );
}

/// A present-but-zero window is a corrupt declaration and must be rejected at
/// load, not silently promoted to fully-causal attention.
#[test]
fn zero_sliding_window_is_rejected_at_load() {
    use oxibonsai_core::gguf::reader::GgufFile;
    let bytes = build_sliding_window_gguf(Some(0));
    let gguf = GgufFile::parse(&bytes).expect("parse");
    // `BonsaiModel` is not `Debug`, so unwrap the `Result` by hand rather
    // than through `expect_err`.
    let text = match BonsaiModel::from_gguf(&gguf, 256) {
        Ok(_) => panic!("a zero-width sliding window must not load"),
        Err(e) => e.to_string(),
    };
    assert!(
        text.contains("sliding_window"),
        "the error must name the offending key: {text}"
    );
}

/// M-18 + M-17 interaction: `forward_prefill` chunks *above*
/// `forward_prefill_unchunked`, and the window gate lives *inside* it, so each
/// chunk is gated independently. A chunked windowed prefill must therefore
/// still be bit-identical to the windowed per-token forward.
#[test]
fn chunked_windowed_prefill_still_honours_the_window() {
    use oxibonsai_core::gguf::reader::GgufFile;

    let prompt: Vec<u32> = (0..12_u32).map(|i| (i * 5 + 3) % SW_VOCAB as u32).collect();
    let bytes = build_sliding_window_gguf(Some(3));

    let sequential = sw_sequential_logits(&bytes, &prompt);

    let gguf = GgufFile::parse(&bytes).expect("parse");
    let mut model = BonsaiModel::from_gguf(&gguf, 256).expect("from_gguf");
    // Four tokens per chunk over a 12-token prompt: three real chunks.
    model.set_prefill_chunk_tokens(4);
    let kernel = ref_kernel();
    let chunked = model
        .forward_prefill(&prompt, 0, kernel.as_ref())
        .expect("chunked forward_prefill");

    for (i, (a, b)) in sequential.iter().zip(chunked.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "chunked windowed prefill diverged from the windowed sequential \
             forward at index {i}: {a} vs {b}"
        );
    }
}
