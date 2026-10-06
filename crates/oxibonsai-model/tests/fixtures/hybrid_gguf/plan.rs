//! The tensor plan: one record per GGUF tensor of a variant, carrying both
//! its wire encoding kind and the exact pre-quantization `f32` values the
//! reference model widens to f64, plus the Hadamard sign vectors and the
//! folded-tensor name set.

use std::collections::BTreeMap;

use oxibonsai_core::bf16::{bf16_to_f32, f32_to_bf16};
use oxibonsai_core::gguf::writer::TensorType;

use super::spec::{Dims, HybridFixtureSpec, BLOCK_COUNT, N_HEAD, N_KV, SSM_CONV_KERNEL, VOCAB};
use super::Xorshift64Star;

// ═════════════════════════════════════════════════════════════════════════
// 8. Planned tensors: one record per GGUF tensor, carrying both its wire
//    encoding and the exact pre-quantization f32 values the reference
//    model widens to f64.
// ═════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TensorKind {
    /// Encoded with the variant's chosen `quant` format.
    Quant,
    /// Always `BF16`, never folded (`ssm_alpha`/`ssm_beta`).
    Bf16,
    /// Always plain `F32` regardless of the variant (norms, conv1d, `ssm_a`,
    /// `ssm_dt.bias`).
    PlainF32,
}

pub(super) struct PlannedTensor {
    pub(super) shape: Vec<u64>,
    pub(super) kind: TensorKind,
    /// Row-major, length `product(shape)`. For [`TensorKind::Bf16`], these
    /// are already the bf16-round-tripped values (so the reference model
    /// widens exactly what the file stores).
    pub(super) values: Vec<f32>,
}

/// Every named tensor a `qwen35` hybrid checkpoint binds, plus enough
/// metadata to drive both the GGUF writer and the f64 reference model.
pub(super) struct FixturePlan {
    pub(super) dims: Dims,
    pub(super) tensors: BTreeMap<String, PlannedTensor>,
    /// `width -> ±1.0 vector`, only populated when `spec.hadamard`.
    pub(super) signs: BTreeMap<usize, Vec<f64>>,
    pub(super) folded_names: Vec<String>,
}

pub(super) fn tname(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}

/// Quantizable tensor kinds' foldable-suffix set (design §1.7 rule 7 /
/// `hadamard_config.rs::FOLDABLE_BLOCK_SUFFIXES`).
const FOLDABLE_SUFFIXES: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_gate",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ssm_out",
];

pub(super) fn build_plan(spec: &HybridFixtureSpec) -> FixturePlan {
    let dims = Dims::from_spec(spec);
    let mut rng = Xorshift64Star::new(spec.seed);
    let mut tensors = BTreeMap::new();

    let push_quant = |tensors: &mut BTreeMap<String, PlannedTensor>,
                      rng: &mut Xorshift64Star,
                      name: &str,
                      ne0: usize,
                      ne1: usize| {
        let n = ne0 * ne1;
        let values: Vec<f32> = (0..n)
            .map(|_| match spec.quant {
                TensorType::F32 => rng.next_continuous(),
                TensorType::Q1_0G128 => rng.next_binary(),
                _ => rng.next_ternary(),
            })
            .collect();
        tensors.insert(
            name.to_string(),
            PlannedTensor {
                shape: vec![ne0 as u64, ne1 as u64],
                kind: TensorKind::Quant,
                values,
            },
        );
    };
    let push_plain_f32 = |tensors: &mut BTreeMap<String, PlannedTensor>,
                          rng: &mut Xorshift64Star,
                          name: &str,
                          shape: Vec<u64>,
                          range: (f64, f64)| {
        let n: u64 = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|_| rng.next_range_f64(range.0, range.1) as f32)
            .collect();
        tensors.insert(
            name.to_string(),
            PlannedTensor {
                shape,
                kind: TensorKind::PlainF32,
                values,
            },
        );
    };
    let push_bf16 = |tensors: &mut BTreeMap<String, PlannedTensor>,
                     rng: &mut Xorshift64Star,
                     name: &str,
                     ne0: usize,
                     ne1: usize| {
        let n = ne0 * ne1;
        let values: Vec<f32> = (0..n)
            .map(|_| {
                let raw = rng.next_range_f64(-1.0, 1.0) as f32;
                bf16_to_f32(f32_to_bf16(raw))
            })
            .collect();
        tensors.insert(
            name.to_string(),
            PlannedTensor {
                shape: vec![ne0 as u64, ne1 as u64],
                kind: TensorKind::Bf16,
                values,
            },
        );
    };

    // ── Globals ────────────────────────────────────────────────────────
    push_quant(
        &mut tensors,
        &mut rng,
        "token_embd.weight",
        dims.hidden,
        VOCAB,
    );
    push_plain_f32(
        &mut tensors,
        &mut rng,
        "output_norm.weight",
        vec![dims.hidden as u64],
        (0.5, 1.5),
    );
    push_quant(&mut tensors, &mut rng, "output.weight", dims.hidden, VOCAB);

    for layer in 0..BLOCK_COUNT {
        push_plain_f32(
            &mut tensors,
            &mut rng,
            &tname(layer, "attn_norm.weight"),
            vec![dims.hidden as u64],
            (0.5, 1.5),
        );
        push_plain_f32(
            &mut tensors,
            &mut rng,
            &tname(layer, "post_attention_norm.weight"),
            vec![dims.hidden as u64],
            (0.5, 1.5),
        );
        push_quant(
            &mut tensors,
            &mut rng,
            &tname(layer, "ffn_gate.weight"),
            dims.hidden,
            dims.ffn,
        );
        push_quant(
            &mut tensors,
            &mut rng,
            &tname(layer, "ffn_up.weight"),
            dims.hidden,
            dims.ffn,
        );
        push_quant(
            &mut tensors,
            &mut rng,
            &tname(layer, "ffn_down.weight"),
            dims.ffn,
            dims.hidden,
        );

        if dims.is_full_attention(layer) {
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_q.weight"),
                dims.hidden,
                N_HEAD * dims.head_dim * 2,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_k.weight"),
                dims.hidden,
                N_KV * dims.head_dim,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_v.weight"),
                dims.hidden,
                N_KV * dims.head_dim,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_output.weight"),
                dims.attn_output_input_width(),
                dims.hidden,
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_q_norm.weight"),
                vec![dims.head_dim as u64],
                (0.5, 1.5),
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_k_norm.weight"),
                vec![dims.head_dim as u64],
                (0.5, 1.5),
            );
        } else {
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_qkv.weight"),
                dims.hidden,
                dims.conv_dim,
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "attn_gate.weight"),
                dims.hidden,
                dims.inner_size,
            );
            push_bf16(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_alpha.weight"),
                dims.hidden,
                dims.n_v_heads,
            );
            push_bf16(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_beta.weight"),
                dims.hidden,
                dims.n_v_heads,
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_conv1d.weight"),
                vec![SSM_CONV_KERNEL as u64, dims.conv_dim as u64],
                (-0.5, 0.5),
            );
            // ssm_a: negative (A = -exp(A_log)), consumed as-is.
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_a"),
                vec![dims.n_v_heads as u64],
                (-1.0, -0.1),
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_dt.bias"),
                vec![dims.n_v_heads as u64],
                (-0.5, 0.5),
            );
            push_plain_f32(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_norm.weight"),
                vec![dims.head_v_dim as u64],
                (0.5, 1.5),
            );
            push_quant(
                &mut tensors,
                &mut rng,
                &tname(layer, "ssm_out.weight"),
                dims.inner_size,
                dims.hidden,
            );
        }
    }

    // ── Hadamard sign vectors + folded-name set ─────────────────────────
    let mut signs = BTreeMap::new();
    let mut folded_names = Vec::new();
    if spec.hadamard {
        for &width in &[dims.hidden, dims.attn_output_input_width(), dims.ffn] {
            signs.entry(width).or_insert_with(|| {
                (0..width)
                    .map(|_| {
                        if rng.next_u64().is_multiple_of(2) {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                    .collect::<Vec<f64>>()
            });
        }
        // `ssm_out`'s input width is `inner_size`, which coincides with
        // `attn_output_input_width()` by this fixture's chosen dimensions
        // (mirrors the real 27B, where both are 6144) — already covered
        // by the loop above, asserted in `hybrid_fixture_tests.rs`.
        folded_names.push("output.weight".to_string());
        for layer in 0..BLOCK_COUNT {
            let suffixes: &[&str] = if dims.is_full_attention(layer) {
                &["attn_q", "attn_k", "attn_v", "attn_output"]
            } else {
                &["attn_qkv", "attn_gate", "ssm_out"]
            };
            for suffix in suffixes {
                debug_assert!(FOLDABLE_SUFFIXES.contains(suffix));
                folded_names.push(tname(layer, &format!("{suffix}.weight")));
            }
            folded_names.push(tname(layer, "ffn_gate.weight"));
            folded_names.push(tname(layer, "ffn_up.weight"));
            folded_names.push(tname(layer, "ffn_down.weight"));
        }
    }

    FixturePlan {
        dims,
        tensors,
        signs,
        folded_names,
    }
}
