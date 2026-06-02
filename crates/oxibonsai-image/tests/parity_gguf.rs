//! Validation of the `oxibonsai-image` DiT loader against the real, validated
//! `bonsai-image` GGUF produced by the MLX→GGUF converter.
//!
//! This test is **gated** behind the `OXIBONSAI_DIT_GGUF` environment variable,
//! which must point at the GGUF file (e.g. `/tmp/parity.gguf`). When the
//! variable is unset (CI, or the 1.3 GB file absent) the test is skipped, so it
//! never fails a clean checkout. Run with:
//!
//! ```text
//! OXIBONSAI_DIT_GGUF=/tmp/parity.gguf cargo test -p oxibonsai-image --test parity_gguf -- --nocapture
//! ```

use std::path::PathBuf;

use oxibonsai_image::DitWeights;

/// One entry in the expected quantized-linear inventory.
struct QuantSpec {
    /// Storage (base) module name.
    name: String,
    /// Logical output features.
    out: u64,
    /// Logical input features.
    inp: u64,
}

/// Build the full 100-module quantized inventory from the converter design.
fn quantized_inventory() -> Vec<QuantSpec> {
    let mut v = Vec::new();

    // Double-stream blocks 0..5.
    for layer in 0..5u32 {
        let p = format!("transformer_blocks.{layer}");
        // (3072, 3072) attention projections — 8 per block.
        for m in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
        ] {
            v.push(QuantSpec {
                name: format!("{p}.{m}"),
                out: 3072,
                inp: 3072,
            });
        }
        // (18432, 3072) feed-forward input projections.
        for m in ["ff.linear_in", "ff_context.linear_in"] {
            v.push(QuantSpec {
                name: format!("{p}.{m}"),
                out: 18432,
                inp: 3072,
            });
        }
        // (3072, 9216) feed-forward output projections.
        for m in ["ff.linear_out", "ff_context.linear_out"] {
            v.push(QuantSpec {
                name: format!("{p}.{m}"),
                out: 3072,
                inp: 9216,
            });
        }
    }

    // Single-stream blocks 0..20.
    for layer in 0..20u32 {
        let p = format!("single_transformer_blocks.{layer}");
        // (27648, 3072) fused QKV+MLP projection.
        v.push(QuantSpec {
            name: format!("{p}.attn.to_qkv_mlp_proj"),
            out: 27648,
            inp: 3072,
        });
        // (3072, 12288) output projection.
        v.push(QuantSpec {
            name: format!("{p}.attn.to_out"),
            out: 3072,
            inp: 12288,
        });
    }

    v
}

/// BF16 spot-check entries: (full name, logical shape).
fn bf16_spot_checks() -> Vec<(&'static str, Vec<u64>)> {
    vec![
        ("x_embedder.weight", vec![3072, 128]),
        ("context_embedder.weight", vec![3072, 7680]),
        ("proj_out.weight", vec![128, 3072]),
        ("norm_out.linear.weight", vec![6144, 3072]),
        (
            "double_stream_modulation_img.linear.weight",
            vec![18432, 3072],
        ),
        (
            "double_stream_modulation_txt.linear.weight",
            vec![18432, 3072],
        ),
        ("single_stream_modulation.linear.weight", vec![9216, 3072]),
    ]
}

#[test]
fn parity_gguf_full_validation() {
    let Ok(path) = std::env::var("OXIBONSAI_DIT_GGUF") else {
        eprintln!("skipping: OXIBONSAI_DIT_GGUF not set");
        return;
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        eprintln!("skipping: {} does not exist", path.display());
        return;
    }

    let weights = DitWeights::open(&path).expect("load real DiT GGUF");

    // ── Config reads back the expected dims. ──
    let cfg = weights.config();
    assert_eq!(cfg.num_layers, 5, "num_layers");
    assert_eq!(cfg.num_single_layers, 20, "num_single_layers");
    assert_eq!(cfg.num_attention_heads, 24, "num_attention_heads");
    assert_eq!(cfg.attention_head_dim, 128, "attention_head_dim");
    assert_eq!(cfg.hidden_size(), 3072, "hidden_size derived");
    assert_eq!(cfg.joint_attention_dim, 7680, "joint_attention_dim");
    assert_eq!(cfg.in_channels, 128, "in_channels");
    assert_eq!(cfg.mlp_ratio, 3.0, "mlp_ratio");
    assert_eq!(cfg.axes_dims_rope, vec![32, 32, 32, 32], "axes_dims_rope");
    assert_eq!(cfg.rope_theta, 2000.0, "rope_theta");
    assert!(!cfg.guidance_embeds, "guidance_embeds");

    // ── Type partition: 100 TQ2_0_g128 + 69 BF16 = 169 total. ──
    let quant_names = weights.quantized_names();
    let bf16_names = weights.bf16_names();
    println!(
        "tensor_count={} quantized={} bf16={}",
        weights.tensor_count(),
        quant_names.len(),
        bf16_names.len()
    );
    assert_eq!(quant_names.len(), 100, "expected 100 quantized tensors");
    assert_eq!(bf16_names.len(), 69, "expected 69 bf16 tensors");
    assert_eq!(weights.tensor_count(), 169, "expected 169 total tensors");

    // ── Every quantized module from the inventory present with correct ──
    // recovered (out, in) and block count = out * (in / 128).
    let inventory = quantized_inventory();
    assert_eq!(inventory.len(), 100, "inventory enumerates 100 modules");
    for spec in &inventory {
        let q = weights
            .quantized_linear(&spec.name)
            .unwrap_or_else(|e| panic!("missing quantized module {}: {e}", spec.name));
        assert_eq!(
            q.out_features, spec.out,
            "{} recovered out_features",
            spec.name
        );
        assert_eq!(
            q.in_features, spec.inp,
            "{} recovered in_features",
            spec.name
        );
        let want_blocks = spec.out * (spec.inp / 128);
        assert_eq!(
            q.blocks.len() as u64,
            want_blocks,
            "{} block count = out*(in/128)",
            spec.name
        );
        assert_eq!(
            q.expected_block_count(),
            q.blocks.len() as u64,
            "{} expected_block_count agrees",
            spec.name
        );
    }

    // ── BF16 spot-checks: present, correct logical (reversed-ne) shape. ──
    for (name, logical_shape) in bf16_spot_checks() {
        let bf = weights
            .bf16_tensor(name)
            .unwrap_or_else(|e| panic!("missing bf16 tensor {name}: {e}"));
        assert_eq!(bf.shape(), logical_shape, "{name} logical shape");
        let expected_elems: u64 = logical_shape.iter().product();
        assert_eq!(bf.element_count(), expected_elems, "{name} element count");
        // Confirm a real decode works (touch the first chunk only is implicit
        // via to_f32_vec, which must produce element_count values).
        assert_eq!(
            bf.to_f32_vec().len() as u64,
            expected_elems,
            "{name} decodes to element_count f32 values"
        );
    }

    // ── Sanity: the union of the two name sets is disjoint and complete. ──
    for q in &quant_names {
        assert!(
            !bf16_names.contains(q),
            "tensor {q} classified as both quantized and bf16"
        );
    }

    // Print the BF16 inventory so a reviewer can confirm the 69 set (e.g. the
    // presence of time_guidance_embed and the absence of time_text_embed).
    println!("--- BF16 tensors ({}) ---", bf16_names.len());
    for n in &bf16_names {
        println!("{n}");
    }
}
