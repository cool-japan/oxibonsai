//! Self-contained synthetic-GGUF test for the `oxibonsai-image` DiT loader.
//!
//! Builds a tiny `bonsai-image` GGUF in memory (one fake quantized module + one
//! BF16 tensor + the full `bonsai-image.*` metadata) with the OxiBonsai core
//! `GgufWriter`, then loads it through [`DitWeights`]. Runs without the 1.3 GB
//! `/tmp/parity.gguf` and validates every code path: config parsing, the
//! base-name quantized lookup (including `.weight`-suffix fallback), reversed
//! shape recovery, ternary block count, and BF16 decode.

use half::bf16;

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::quant_ternary::BlockTQ2_0_g128;

use oxibonsai_image::{DitWeights, DEFAULT_EPS};

/// Logical output features for the synthetic quantized linear.
const SYN_OUT: u64 = 2;
/// Logical input features for the synthetic quantized linear (128-divisible).
const SYN_IN: u64 = 128;
/// Base module name a quantized linear is stored under (no `.weight`).
const QUANT_NAME: &str = "transformer_blocks.0.attn.to_q";
/// Full name a plain BF16 tensor is stored under.
const BF16_NAME: &str = "x_embedder.weight";

/// Build a tiny in-memory `bonsai-image` GGUF and the f32 weights we expect the
/// quantized linear to dequantize to.
fn build_synthetic_gguf() -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    // ── Quantized module: (out=2, in=128) → 2 ternary blocks (68 bytes). ──
    // Fill each 128-element group with a deterministic ternary pattern so the
    // dequantized values are predictable.
    let n_blocks = (SYN_OUT * (SYN_IN / 128)) as usize;
    let mut q_input = vec![0.0f32; (SYN_OUT * SYN_IN) as usize];
    for (i, v) in q_input.iter_mut().enumerate() {
        // -1, 0, +1 repeating, scaled by 0.5 so absmax (the stored scale) = 0.5.
        *v = match i % 3 {
            0 => -0.5,
            1 => 0.0,
            _ => 0.5,
        };
    }
    let blocks = BlockTQ2_0_g128::quantize(&q_input).expect("quantize synthetic weight");
    assert_eq!(blocks.len(), n_blocks);
    // What the loader should dequantize back to (lossy: snapped to {-d,0,+d}).
    let mut q_expected = vec![0.0f32; (SYN_OUT * SYN_IN) as usize];
    BlockTQ2_0_g128::dequant(&blocks, &mut q_expected).expect("dequant synthetic weight");

    let mut q_bytes = Vec::with_capacity(blocks.len() * std::mem::size_of::<BlockTQ2_0_g128>());
    for b in &blocks {
        q_bytes.extend_from_slice(&b.qs);
        q_bytes.extend_from_slice(&b.d.to_le_bytes());
    }

    // ── Plain BF16 tensor: logical [3072, 128] would be huge; use a small ──
    // logical [3, 4] = 12 elements, stored reversed as ne = [4, 3].
    let bf16_logical: Vec<f32> = (0..12).map(|i| (i as f32) * 0.25).collect();
    let bf16_bytes: Vec<u8> = bf16_logical
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect();

    // ── Writer + metadata (the full bonsai-image.* namespace). ──
    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("bonsai-image".to_string()),
    );
    w.add_metadata("bonsai-image.num_layers", MetadataWriteValue::U32(5));
    w.add_metadata(
        "bonsai-image.num_single_layers",
        MetadataWriteValue::U32(20),
    );
    w.add_metadata(
        "bonsai-image.attention.head_count",
        MetadataWriteValue::U32(24),
    );
    w.add_metadata(
        "bonsai-image.attention.head_dim",
        MetadataWriteValue::U32(128),
    );
    w.add_metadata(
        "bonsai-image.joint_attention_dim",
        MetadataWriteValue::U32(7680),
    );
    w.add_metadata("bonsai-image.in_channels", MetadataWriteValue::U32(128));
    w.add_metadata("bonsai-image.mlp_ratio", MetadataWriteValue::F32(3.0));
    w.add_metadata(
        "bonsai-image.rope.axes_dims",
        MetadataWriteValue::ArrayU32(vec![32, 32, 32, 32]),
    );
    w.add_metadata("bonsai-image.rope.theta", MetadataWriteValue::F32(2000.0));
    w.add_metadata(
        "bonsai-image.guidance_embeds",
        MetadataWriteValue::Bool(false),
    );

    // Quantized tensor stored under BASE name with reversed shape ne = [in, out].
    w.add_tensor(TensorEntry {
        name: QUANT_NAME.to_string(),
        shape: vec![SYN_IN, SYN_OUT],
        tensor_type: TensorType::TQ2_0_g128,
        data: q_bytes,
    });
    // Plain BF16 tensor under full name, reversed shape ne = [4, 3].
    w.add_tensor(TensorEntry {
        name: BF16_NAME.to_string(),
        shape: vec![4, 3],
        tensor_type: TensorType::BF16,
        data: bf16_bytes,
    });

    let bytes = w.to_bytes().expect("serialise synthetic gguf");
    (bytes, q_expected, bf16_logical)
}

#[test]
fn synthetic_gguf_loads_and_validates() {
    let (gguf_bytes, q_expected, bf16_logical) = build_synthetic_gguf();
    let weights = DitWeights::from_bytes(gguf_bytes).expect("load synthetic DiT");

    // ── Config parsed from metadata. ──
    let cfg = weights.config();
    assert_eq!(cfg.num_layers, 5);
    assert_eq!(cfg.num_single_layers, 20);
    assert_eq!(cfg.num_attention_heads, 24);
    assert_eq!(cfg.attention_head_dim, 128);
    assert_eq!(cfg.hidden_size(), 3072);
    assert_eq!(cfg.ffn_inner_size(), 9216);
    assert_eq!(cfg.joint_attention_dim, 7680);
    assert_eq!(cfg.in_channels, 128);
    assert_eq!(cfg.mlp_ratio, 3.0);
    assert_eq!(cfg.axes_dims_rope, vec![32, 32, 32, 32]);
    assert_eq!(cfg.rope_dim(), 128);
    assert_eq!(cfg.rope_theta, 2000.0);
    assert!(!cfg.guidance_embeds);
    assert_eq!(cfg.eps, DEFAULT_EPS);

    // ── Tensor counts / type partitioning. ──
    assert_eq!(weights.tensor_count(), 2);
    assert_eq!(weights.quantized_names(), vec![QUANT_NAME]);
    assert_eq!(weights.bf16_names(), vec![BF16_NAME]);

    // ── Quantized lookup by base name. ──
    let q = weights
        .quantized_linear(QUANT_NAME)
        .expect("quantized linear by base name");
    assert_eq!(q.out_features, SYN_OUT, "recovered out = ne[1]");
    assert_eq!(q.in_features, SYN_IN, "recovered in = ne[0]");
    assert_eq!(q.expected_block_count(), q.blocks.len() as u64);

    // Dequantize and compare to reference.
    let mut deq = vec![0.0f32; (SYN_OUT * SYN_IN) as usize];
    BlockTQ2_0_g128::dequant(q.blocks, &mut deq).expect("dequant loaded blocks");
    assert_eq!(deq, q_expected, "loaded ternary weights match reference");

    // ── Quantized lookup via .weight-suffix fallback. ──
    let q2 = weights
        .quantized_linear(&format!("{QUANT_NAME}.weight"))
        .expect("quantized linear via .weight fallback");
    assert_eq!(q2.blocks.as_ptr(), q.blocks.as_ptr(), "same blocks");

    // ── BF16 lookup, reversed-shape recovery, and decode. ──
    let bf = weights
        .bf16_tensor(BF16_NAME)
        .expect("bf16 tensor by full name");
    assert_eq!(bf.shape(), vec![3, 4], "logical shape = reverse(ne=[4,3])");
    assert_eq!(bf.element_count(), 12);
    let decoded = bf.to_f32_vec();
    assert_eq!(decoded, bf16_logical, "bf16 decodes to reference values");
    let bits = bf.bits().expect("bf16 bits");
    assert_eq!(bits.len(), 12);

    // ── Error paths. ──
    assert!(
        weights.quantized_linear("does.not.exist").is_err(),
        "missing quantized tensor errors"
    );
    assert!(
        weights.bf16_tensor("does.not.exist").is_err(),
        "missing bf16 tensor errors"
    );
    // Asking for the quantized tensor as BF16 must report a type mismatch.
    assert!(
        weights.bf16_tensor(QUANT_NAME).is_err(),
        "type-mismatch on quantized-as-bf16"
    );
}
