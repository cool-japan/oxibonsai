//! Qwen3.5 / PrismML **Bonsai 2** hybrid model (`general.architecture =
//! "qwen35"`) — blocks, recurrent cache, v-head map, Hadamard hook and
//! weight binding (B2-10; design §3).
//!
//! # Why a sibling module tree and not a wider `TransformerBlock`
//!
//! A `qwen35` stack interleaves two *structurally different* decoder layers:
//!
//! * **full attention** (layer `i` where `(i + 1) % full_attention_interval
//!   == 0` — 16 of the 27B's 64): GQA with `head_dim` 256, a `q|gate`
//!   interleave inside one `attn_q [5120, 12288]` projection, per-head q/k
//!   RMSNorm, a sigmoid output gate and partial M-RoPE;
//! * **linear attention** (the other 48): a recurrent Gated DeltaNet with a
//!   depthwise causal conv, per-v-head gates and a `[48][128][128]` state
//!   that persists across tokens.
//!
//! [`crate::block::TransformerBlock`] can express neither (M-04, M-16), and
//! widening it in place would put a hybrid discriminant on the hot path of
//! every shipping Qwen3 model. The design instead puts the whole hybrid
//! stack here, so existing models stay byte-identical; nothing in this
//! module is reachable from [`crate::model::BonsaiModel`].
//!
//! # What this package (B2-10) lands
//!
//! The *skeleton*: every weight bound and validated, every cache allocated,
//! the v-head index map, the Hadamard hook and its scratch. The per-layer
//! `forward` bodies are B2-11 and are deliberately **absent** rather than
//! present-and-`todo!()` — a method that exists must work.
//!
//! | file | role |
//! |---|---|
//! | [`block`] | [`HybridBlock`] enum + the two layer kinds + their scratch |
//! | [`hadamard`] | [`HadamardHook`] / [`HadamardScratch`] (design §3.4/§3.5) |
//! | [`model`] | [`HybridModel`]: construction, layer split, reset |
//! | [`recurrent_cache`] | [`RecurrentCache`]: GDN state + conv window (§3.6) |
//! | [`vhead_map`] | [`VHeadMap`]: tiled ↔ grouped v-head indices (§3.3) |
//! | [`weights`] | GGUF name → field binding, with hard shape errors (§3.8) |

pub mod block;
pub mod hadamard;
pub mod model;
pub mod recurrent_cache;
pub mod vhead_map;
pub mod weights;

pub use block::{FullAttnBlock, FullScratch, HybridBlock, LinearAttnBlock, LinearScratch};
pub use hadamard::{rotated_widths, HadamardHook, HadamardScratch};
pub use model::{HybridModel, LayerSplit, DEFAULT_MAX_SEQ_LEN};
pub use recurrent_cache::{RecurrentCache, RecurrentSnapshot, RECURRENT_NAME};
pub use vhead_map::{VHeadMap, GDN_HEAD_ORDER};
pub use weights::{
    block_tensor, names as tensor_names, Bf16Matrix, GdnGateWeights, HybridEmbedding,
    FULL_LAYER_TENSORS, LINEAR_LAYER_TENSORS, SHARED_LAYER_TENSORS,
};

/// Shared fixtures for this module's unit tests: the real 27B
/// hyper-parameters, a synthetic Hadamard contract, and a *complete*
/// synthetic `qwen35` GGUF small enough to build in-process.
///
/// The synthetic file is what lets the binding, layer-split and refusal
/// paths be tested on a machine with no 27B weights at all (the real-file
/// tests skip when `models/` is empty); it carries every tensor name,
/// dtype and shape relationship the real file does, at 1/20th the width.
#[cfg(test)]
pub(crate) mod tests_support {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use oxibonsai_core::config_hybrid::HybridConfig;
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
    use oxibonsai_core::hadamard_config::HadamardConfig;

    use crate::quantize::{encode_quantized_tensor, ScaleRule};

    /// The real Bonsai 2 27B hyper-parameters (`gguf_headers_summary.txt`).
    pub(crate) fn bonsai2_config() -> HybridConfig {
        let mut base = oxibonsai_core::config::Qwen3Config::bonsai_8b();
        base.architecture = "qwen35".to_string();
        base.model_name = "Ternary-Bonsai-2-27B".to_string();
        base.hidden_size = 5120;
        base.intermediate_size = 17408;
        base.num_layers = 64;
        base.num_attention_heads = 24;
        base.num_kv_heads = 4;
        base.head_dim = 256;
        base.value_length = 256;
        base.vocab_size = 248_320;
        base.max_context_length = 262_144;
        base.rms_norm_eps = 1e-6;
        base.rope_freq_base = 1e7;
        HybridConfig {
            base,
            full_attention_interval: 4,
            rope_dimension_count: 64,
            rope_sections: [11, 11, 10, 0],
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 48,
            ssm_inner_size: 6144,
            nextn_predict_layers: 0,
            sampling_top_k: Some(20),
            sampling_top_p: Some(0.95),
            sampling_temperature: Some(1.0),
        }
    }

    /// A `prism.hadamard.*` contract with the 27B's three sign widths and
    /// the fold membership of the real files (every folded kind, none of the
    /// `ssm_*` gates).
    pub(crate) fn bonsai2_hadamard() -> HadamardConfig {
        let mut signs: HashMap<usize, Arc<[f32]>> = HashMap::new();
        for width in [5120usize, 6144, 17408] {
            let values: Vec<f32> = (0..width)
                .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            signs.insert(width, values.into());
        }
        let mut folded: HashSet<String> = HashSet::new();
        folded.insert("output.weight".to_string());
        for layer in 0..64usize {
            for suffix in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "attn_qkv",
                "attn_gate",
                "ssm_out",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                folded.insert(format!("blk.{layer}.{suffix}.weight"));
            }
        }
        let mut inverse = HashSet::new();
        inverse.insert("token_embd.weight".to_string());
        HadamardConfig {
            block_size: 1024,
            signs,
            folded,
            inverse,
            gdn_v_grouped: true,
        }
    }

    /// Geometry of the synthetic `qwen35` fixture: the 27B's shape
    /// relationships, scaled down so a whole file fits in a unit test.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct FixtureShape {
        pub hidden: usize,
        pub intermediate: usize,
        pub n_layers: usize,
        pub n_heads: usize,
        pub n_kv_heads: usize,
        pub head_dim: usize,
        pub vocab: usize,
        pub full_attention_interval: usize,
        pub conv_kernel: usize,
        pub state_size: usize,
        pub n_k_heads: usize,
        pub n_v_heads: usize,
        pub hadamard_block: usize,
    }

    impl Default for FixtureShape {
        fn default() -> Self {
            Self {
                hidden: 256,
                intermediate: 512,
                n_layers: 8,
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: 64,
                vocab: 512,
                full_attention_interval: 4,
                conv_kernel: 4,
                state_size: 64,
                n_k_heads: 2,
                n_v_heads: 6,
                hadamard_block: 128,
            }
        }
    }

    impl FixtureShape {
        /// `ssm.inner_size` = `n_v_heads * head_v_dim`.
        pub fn inner(&self) -> usize {
            self.n_v_heads * self.state_size
        }

        /// `attn_qkv`'s output width.
        pub fn conv_dim(&self) -> usize {
            2 * self.state_size * self.n_k_heads + self.inner()
        }

        /// Width of the concatenated attention heads (`attn_output`'s input).
        pub fn heads_width(&self) -> usize {
            self.n_heads * self.head_dim
        }

        pub fn is_full(&self, layer: usize) -> bool {
            (layer + 1).is_multiple_of(self.full_attention_interval)
        }

        /// The three activation widths a fold covers.
        pub fn sign_widths(&self) -> Vec<usize> {
            let mut widths = vec![
                self.hidden,
                self.heads_width(),
                self.inner(),
                self.intermediate,
            ];
            widths.sort_unstable();
            widths.dedup();
            widths
        }
    }

    /// How a fixture file is quantized and whether it carries a fold.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct FixtureOptions {
        pub quant: TensorType,
        pub hadamard: bool,
        pub gdn_v_grouped: bool,
        /// Omit `output.weight`, i.e. a tied LM head — which a `qwen35`
        /// loader must refuse (design §3.5).
        pub omit_lm_head: bool,
    }

    impl Default for FixtureOptions {
        fn default() -> Self {
            Self {
                quant: TensorType::PQ2_0,
                hadamard: true,
                gdn_v_grouped: true,
                omit_lm_head: false,
            }
        }
    }

    /// Deterministic pseudo-random weights — a cheap LCG so a fixture is
    /// reproducible without a dependency.
    fn ramp(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    fn quant_tensor(name: &str, shape: [usize; 2], quant: TensorType, seed: u64) -> TensorEntry {
        let [ne0, ne1] = shape;
        let values = ramp(ne0 * ne1, seed);
        let data = encode_quantized_tensor(&values, ne0, quant, ScaleRule::AbsMax)
            .unwrap_or_else(|e| panic!("fixture {name}: {e}"));
        TensorEntry {
            name: name.to_string(),
            shape: vec![ne0 as u64, ne1 as u64],
            tensor_type: quant,
            data,
        }
    }

    fn f32_tensor(name: &str, shape: &[usize], values: Vec<f32>) -> TensorEntry {
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        TensorEntry {
            name: name.to_string(),
            shape: shape.iter().map(|d| *d as u64).collect(),
            tensor_type: TensorType::F32,
            data,
        }
    }

    fn bf16_tensor(name: &str, shape: [usize; 2], seed: u64) -> TensorEntry {
        let [ne0, ne1] = shape;
        let values = ramp(ne0 * ne1, seed);
        let mut data = Vec::with_capacity(values.len() * 2);
        for v in &values {
            // Round-to-nearest-even bf16, matching `pack::f32_to_bf16`.
            let bits = v.to_bits();
            let rounded = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
            data.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
        }
        TensorEntry {
            name: name.to_string(),
            shape: vec![ne0 as u64, ne1 as u64],
            tensor_type: TensorType::BF16,
            data,
        }
    }

    /// Build a complete synthetic `qwen35` GGUF in memory.
    pub(crate) fn synthetic_gguf(shape: FixtureShape, options: FixtureOptions) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        let u32v = |v: usize| MetadataWriteValue::U32(v as u32);
        writer
            .add_metadata(
                "general.architecture",
                MetadataWriteValue::Str("qwen35".into()),
            )
            .add_metadata(
                "general.name",
                MetadataWriteValue::Str("synthetic-qwen35".into()),
            )
            .add_metadata("qwen35.embedding_length", u32v(shape.hidden))
            .add_metadata("qwen35.feed_forward_length", u32v(shape.intermediate))
            .add_metadata("qwen35.block_count", u32v(shape.n_layers))
            .add_metadata("qwen35.attention.head_count", u32v(shape.n_heads))
            .add_metadata("qwen35.attention.head_count_kv", u32v(shape.n_kv_heads))
            .add_metadata("qwen35.attention.key_length", u32v(shape.head_dim))
            .add_metadata("qwen35.attention.value_length", u32v(shape.head_dim))
            .add_metadata("qwen35.vocab_size", u32v(shape.vocab))
            .add_metadata("qwen35.context_length", u32v(4096))
            .add_metadata(
                "qwen35.attention.layer_norm_rms_epsilon",
                MetadataWriteValue::F32(1e-6),
            )
            .add_metadata("qwen35.rope.freq_base", MetadataWriteValue::F32(1e7))
            .add_metadata("qwen35.rope.dimension_count", u32v(16))
            .add_metadata(
                "qwen35.rope.dimension_sections",
                MetadataWriteValue::ArrayU32(vec![3, 3, 2, 0]),
            )
            .add_metadata(
                "qwen35.full_attention_interval",
                u32v(shape.full_attention_interval),
            )
            .add_metadata("qwen35.ssm.conv_kernel", u32v(shape.conv_kernel))
            .add_metadata("qwen35.ssm.state_size", u32v(shape.state_size))
            .add_metadata("qwen35.ssm.group_count", u32v(shape.n_k_heads))
            .add_metadata("qwen35.ssm.time_step_rank", u32v(shape.n_v_heads))
            .add_metadata("qwen35.ssm.inner_size", u32v(shape.inner()));

        if options.hadamard {
            let widths = shape.sign_widths();
            let mut sign_values: Vec<i32> = Vec::new();
            for width in &widths {
                for i in 0..*width {
                    sign_values.push(if i % 3 == 0 { -1 } else { 1 });
                }
            }
            // A name listed in the fold contract must exist as a tensor
            // (`validate_against_tensors`), so a fixture without an LM head
            // must not list one either — otherwise the contract check fires
            // before the tied-head refusal under test.
            let mut weight_names = if options.omit_lm_head {
                Vec::new()
            } else {
                vec!["output.weight".to_string()]
            };
            for layer in 0..shape.n_layers {
                let folded: &[&str] = if shape.is_full(layer) {
                    &[
                        "attn_q",
                        "attn_k",
                        "attn_v",
                        "attn_output",
                        "ffn_gate",
                        "ffn_up",
                        "ffn_down",
                    ]
                } else {
                    &[
                        "attn_qkv",
                        "attn_gate",
                        "ssm_out",
                        "ffn_gate",
                        "ffn_up",
                        "ffn_down",
                    ]
                };
                for suffix in folded {
                    weight_names.push(format!("blk.{layer}.{suffix}.weight"));
                }
            }
            writer
                .add_metadata("prism.hadamard.version", MetadataWriteValue::U32(1))
                .add_metadata("prism.hadamard.block_size", u32v(shape.hadamard_block))
                .add_metadata(
                    "prism.hadamard.transform",
                    MetadataWriteValue::Str("normalized-sylvester-walsh-hadamard".into()),
                )
                .add_metadata(
                    "prism.hadamard.axis",
                    MetadataWriteValue::Str("input-last-dimension".into()),
                )
                .add_metadata(
                    "prism.hadamard.sign_mode",
                    MetadataWriteValue::Str("explicit".into()),
                )
                .add_metadata(
                    "prism.hadamard.sign_widths",
                    MetadataWriteValue::ArrayU32(widths.iter().map(|w| *w as u32).collect()),
                )
                .add_metadata(
                    "prism.hadamard.sign_values",
                    MetadataWriteValue::ArrayI32(sign_values),
                )
                .add_metadata(
                    "prism.hadamard.weight_names",
                    MetadataWriteValue::ArrayStr(weight_names),
                )
                .add_metadata(
                    "prism.hadamard.inverse_weight_names",
                    MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
                )
                .add_metadata(
                    "prism.hadamard.gdn_v_grouped",
                    MetadataWriteValue::Bool(options.gdn_v_grouped),
                );
        }

        let quant = options.quant;
        let mut seed = 1u64;
        let mut next_seed = || {
            seed = seed.wrapping_add(7919);
            seed
        };

        writer.add_tensor(quant_tensor(
            "token_embd.weight",
            [shape.hidden, shape.vocab],
            quant,
            next_seed(),
        ));
        if !options.omit_lm_head {
            writer.add_tensor(quant_tensor(
                "output.weight",
                [shape.hidden, shape.vocab],
                quant,
                next_seed(),
            ));
        }
        writer.add_tensor(f32_tensor(
            "output_norm.weight",
            &[shape.hidden],
            vec![1.0; shape.hidden],
        ));

        for layer in 0..shape.n_layers {
            let blk = |suffix: &str| format!("blk.{layer}.{suffix}");
            writer.add_tensor(f32_tensor(
                &blk("attn_norm.weight"),
                &[shape.hidden],
                vec![1.0; shape.hidden],
            ));
            writer.add_tensor(f32_tensor(
                &blk("post_attention_norm.weight"),
                &[shape.hidden],
                vec![1.0; shape.hidden],
            ));
            writer.add_tensor(quant_tensor(
                &blk("ffn_gate.weight"),
                [shape.hidden, shape.intermediate],
                quant,
                next_seed(),
            ));
            writer.add_tensor(quant_tensor(
                &blk("ffn_up.weight"),
                [shape.hidden, shape.intermediate],
                quant,
                next_seed(),
            ));
            writer.add_tensor(quant_tensor(
                &blk("ffn_down.weight"),
                [shape.intermediate, shape.hidden],
                quant,
                next_seed(),
            ));

            if shape.is_full(layer) {
                writer.add_tensor(quant_tensor(
                    &blk("attn_q.weight"),
                    [shape.hidden, shape.heads_width() * 2],
                    quant,
                    next_seed(),
                ));
                writer.add_tensor(quant_tensor(
                    &blk("attn_k.weight"),
                    [shape.hidden, shape.n_kv_heads * shape.head_dim],
                    quant,
                    next_seed(),
                ));
                writer.add_tensor(quant_tensor(
                    &blk("attn_v.weight"),
                    [shape.hidden, shape.n_kv_heads * shape.head_dim],
                    quant,
                    next_seed(),
                ));
                writer.add_tensor(quant_tensor(
                    &blk("attn_output.weight"),
                    [shape.heads_width(), shape.hidden],
                    quant,
                    next_seed(),
                ));
                writer.add_tensor(f32_tensor(
                    &blk("attn_q_norm.weight"),
                    &[shape.head_dim],
                    vec![1.0; shape.head_dim],
                ));
                writer.add_tensor(f32_tensor(
                    &blk("attn_k_norm.weight"),
                    &[shape.head_dim],
                    vec![1.0; shape.head_dim],
                ));
            } else {
                writer.add_tensor(quant_tensor(
                    &blk("attn_qkv.weight"),
                    [shape.hidden, shape.conv_dim()],
                    quant,
                    next_seed(),
                ));
                writer.add_tensor(quant_tensor(
                    &blk("attn_gate.weight"),
                    [shape.hidden, shape.inner()],
                    quant,
                    next_seed(),
                ));
                writer.add_tensor(quant_tensor(
                    &blk("ssm_out.weight"),
                    [shape.inner(), shape.hidden],
                    quant,
                    next_seed(),
                ));
                writer.add_tensor(bf16_tensor(
                    &blk("ssm_alpha.weight"),
                    [shape.hidden, shape.n_v_heads],
                    next_seed(),
                ));
                writer.add_tensor(bf16_tensor(
                    &blk("ssm_beta.weight"),
                    [shape.hidden, shape.n_v_heads],
                    next_seed(),
                ));
                writer.add_tensor(f32_tensor(
                    &blk("ssm_conv1d.weight"),
                    &[shape.conv_kernel, shape.conv_dim()],
                    ramp(shape.conv_kernel * shape.conv_dim(), next_seed()),
                ));
                // `ssm_a` is `A = -exp(A_log)`: strictly negative, and
                // distinct per head so a gate swap is visible.
                writer.add_tensor(f32_tensor(
                    &blk("ssm_a"),
                    &[shape.n_v_heads],
                    (0..shape.n_v_heads)
                        .map(|h| -0.25 - (h as f32) * 0.5)
                        .collect(),
                ));
                // `dt_bias` is also negative here on purpose: a swap with
                // `ssm_a` must be caught by the arithmetic, not by luck
                // (`validate_a_neg` only rejects a POSITIVE `dt_bias`).
                writer.add_tensor(f32_tensor(
                    &blk("ssm_dt.bias"),
                    &[shape.n_v_heads],
                    (0..shape.n_v_heads)
                        .map(|h| -2.0 + (h as f32) * 0.125)
                        .collect(),
                ));
                writer.add_tensor(f32_tensor(
                    &blk("ssm_norm.weight"),
                    &[shape.state_size],
                    vec![1.0; shape.state_size],
                ));
            }
        }

        writer.to_bytes().expect("synthetic fixture must serialise")
    }

    /// The default fixture: PQ2_0, folded, grouped.
    pub(crate) fn synthetic_default() -> Vec<u8> {
        synthetic_gguf(FixtureShape::default(), FixtureOptions::default())
    }
}
