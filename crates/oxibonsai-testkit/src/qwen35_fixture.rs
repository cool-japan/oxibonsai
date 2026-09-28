//! A complete, public, synthetic `qwen35` (Bonsai 2 hybrid) GGUF fixture
//! (T-07).
//!
//! [`synthetic_qwen35_gguf`] carries the full `qwen35` contract — every
//! `qwen35.*`/`ssm.*` hyperparameter, full-attention `q|gate` layers on every
//! 4th layer, Gated-DeltaNet layers with `attn_qkv`/`attn_gate`/`ssm_*`, the
//! `prism.hadamard.*` fold metadata, grouped v-heads,
//! `tokenizer.ggml.eos_token_id = 3` — at a width small enough to load and
//! forward in a debug build. It is the one hybrid fixture the workspace's
//! runtime tests share: `oxibonsai-runtime`'s own seam tests
//! (`engine_seam_tests.rs`) and its external integration tests
//! (`tests/testkit_qwen35_fixture_tests.rs`) load this same file, so a
//! crate-internal and an outside-the-crate assertion about "the synthetic
//! hybrid" are always about the same weights. Its shape is public
//! ([`HIDDEN`], [`VOCAB`], [`N_LAYERS`], …) so a test can assert geometry
//! without restating it.
//!
//! # Why not entirely through [`crate::gguf_fixture::GgufFixtureBuilder::tensor`]
//!
//! Every *folded* weight matrix (the ones `prism.hadamard.weight_names`
//! lists) is quantized PQ2_0 via [`crate::gguf_fixture::FixtureQuant::PQ2_0`]
//! and [`crate::gguf_fixture::GgufFixtureBuilder::tensor`] — that part is
//! exactly "built on `FixtureQuant::PQ2_0`" per this module's contract. Every
//! other tensor is added via
//! [`crate::gguf_fixture::GgufFixtureBuilder::tensor_raw`] with hand-computed
//! bytes instead, because the hybrid loader's own validation constrains their
//! *values*, not merely their shape, in ways [`crate::gguf_fixture::Lcg`]'s
//! generic `[-1, 1)` weights cannot satisfy:
//!
//! - `ssm_a` must be negative on every element (the loader computes
//!   `A = -exp(A_log)` and rejects a non-negative entry at load) — this
//!   fixture uses explicit negative values, never a random sign.
//! - Every norm weight (`attn_norm`, `post_attention_norm`, `attn_q_norm`,
//!   `attn_k_norm`, `ssm_norm`, `output_norm`) is a constant `1.0` (an
//!   identity gain).
//! - `ssm_alpha`/`ssm_beta` are `BF16` — a format
//!   [`crate::gguf_fixture::FixtureQuant`] has no variant for (every other
//!   consumer of that enum reads a `Block*`-quantized or plain `F32`/`F16`
//!   tensor, never a bare 16-bit float scalar), so this module hand-rolls
//!   the round-to-nearest-even `f32 -> bf16` truncation the model crate's
//!   own `BF16` packer uses.

use oxibonsai_core::gguf::writer::TensorType;
use oxibonsai_core::MetadataWriteValue;

use crate::gguf_fixture::{FixtureError, FixtureQuant, GgufFixtureBuilder, Lcg};

// ─── Shape ───────────────────────────────────────────────────────────────

/// `general.name` of the fixture.
pub const MODEL_NAME: &str = "oxibonsai-testkit-synthetic-qwen35";
/// `qwen35.embedding_length`: the residual width (and an embedding's
/// dimension).
pub const HIDDEN: usize = 256;
/// `qwen35.feed_forward_length`.
pub const INTERMEDIATE: usize = 512;
/// `qwen35.block_count`: two full-attention layers (3 and 7) and six
/// Gated-DeltaNet layers.
pub const N_LAYERS: usize = 8;
/// `qwen35.attention.head_count`.
pub const N_HEADS: usize = 4;
/// `qwen35.attention.head_count_kv`.
pub const N_KV_HEADS: usize = 2;
/// `qwen35.attention.key_length` (= `value_length`).
pub const HEAD_DIM: usize = 64;
/// `qwen35.vocab_size`.
pub const VOCAB: usize = 512;
/// `qwen35.ssm.state_size` (= the Gated-DeltaNet head width).
pub const STATE: usize = 64;
/// `qwen35.ssm.group_count` (k-heads).
pub const N_K_HEADS: usize = 2;
/// `qwen35.ssm.time_step_rank` (v-heads).
pub const N_V_HEADS: usize = 6;
/// `qwen35.ssm.conv_kernel`.
pub const CONV_KERNEL: usize = 4;
/// `prism.hadamard.block_size`.
pub const HADAMARD_BLOCK: usize = 128;
/// `qwen35.context_length` the file declares.
pub const CONTEXT_LENGTH: usize = 4096;
/// `tokenizer.ggml.eos_token_id`.
pub const EOS_TOKEN_ID: u32 = 3;

/// `attn_v`/`ssm_out`'s width: `N_V_HEADS` v-heads of `STATE` each.
#[must_use]
pub const fn inner() -> usize {
    N_V_HEADS * STATE
}

/// `attn_qkv.weight`'s output width: `q(2*STATE*N_K_HEADS) | k(same) |
/// v(inner())` concatenated — mirrors the real 27B's `attn_qkv` layout.
#[must_use]
pub const fn conv_dim() -> usize {
    2 * STATE * N_K_HEADS + inner()
}

/// Full-attention `attn_output.weight`'s input width.
#[must_use]
pub const fn heads_width() -> usize {
    N_HEADS * HEAD_DIM
}

/// Layer `layer` (0-indexed) is full-attention iff `(layer + 1) % 4 == 0` —
/// the real model's `full_attention_interval = 4` contract.
#[must_use]
pub const fn is_full(layer: usize) -> bool {
    (layer + 1).is_multiple_of(4)
}

/// Number of full-attention layers — the layers that own a KV-cache slot.
#[must_use]
pub const fn full_layer_count() -> usize {
    let mut count = 0;
    let mut layer = 0;
    while layer < N_LAYERS {
        if is_full(layer) {
            count += 1;
        }
        layer += 1;
    }
    count
}

/// A round-to-nearest-even `f32 -> bf16` encode, matching
/// `engine_seam_tests.rs::bf16_tensor`'s bit manipulation exactly: `bf16` is
/// simply the top 16 bits of an IEEE-754 `f32`, rounded rather than
/// truncated.
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let rounded = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
    (rounded >> 16) as u16
}

/// `n` deterministic values in `[-1, 1)`, seeded by `seed` — used only for
/// tensors this module encodes itself (`ssm_conv1d`, `BF16` scalars), never
/// for the PQ2_0-quantized matrices ([`GgufFixtureBuilder::tensor`] already
/// draws its own weights from the same seed).
fn ramp(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next_signed_unit_f32()).collect()
}

/// `values.len()` little-endian `f32` bytes.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values.len()` little-endian `bf16` bytes.
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f32_to_bf16_bits(*v).to_le_bytes())
        .collect()
}

/// The fallible half of [`synthetic_qwen35_gguf`] — every [`FixtureQuant::PQ2_0`]
/// tensor goes through `?` here instead of `.expect(...)`, so this module's
/// only panic site is [`synthetic_qwen35_gguf`]'s own thin wrapper (COOLJAPAN
/// policy: no `unwrap()`/`expect()` outside `#[cfg(test)]`).
///
/// # Errors
///
/// [`FixtureError::NotBlockAligned`] if a shape constant in this module were
/// ever changed to something not a multiple of `QK_PQ2_0` (128); every
/// constant as shipped satisfies this, so a normal call never returns `Err`.
/// [`FixtureError::Write`] on a duplicate tensor name, which this function's
/// own layer loop cannot produce (every name is `blk.{layer}.{suffix}` for a
/// distinct `(layer, suffix)` pair, or one of the three fixed global names).
pub fn try_synthetic_qwen35_gguf() -> Result<Vec<u8>, FixtureError> {
    let mut builder = GgufFixtureBuilder::new();
    builder
        .metadata_str("general.architecture", "qwen35")
        .metadata_str("general.name", MODEL_NAME)
        .metadata_u32("qwen35.embedding_length", HIDDEN as u32)
        .metadata_u32("qwen35.feed_forward_length", INTERMEDIATE as u32)
        .metadata_u32("qwen35.block_count", N_LAYERS as u32)
        .metadata_u32("qwen35.attention.head_count", N_HEADS as u32)
        .metadata_u32("qwen35.attention.head_count_kv", N_KV_HEADS as u32)
        .metadata_u32("qwen35.attention.key_length", HEAD_DIM as u32)
        .metadata_u32("qwen35.attention.value_length", HEAD_DIM as u32)
        .metadata_u32("qwen35.vocab_size", VOCAB as u32)
        .metadata_u32("qwen35.context_length", CONTEXT_LENGTH as u32)
        .metadata_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
        .metadata_f32("qwen35.rope.freq_base", 1e7)
        .metadata_u32("qwen35.rope.dimension_count", 16)
        .metadata(
            "qwen35.rope.dimension_sections",
            MetadataWriteValue::ArrayU32(vec![3, 3, 2, 0]),
        )
        .metadata_u32("qwen35.full_attention_interval", 4)
        .metadata_u32("qwen35.ssm.conv_kernel", CONV_KERNEL as u32)
        .metadata_u32("qwen35.ssm.state_size", STATE as u32)
        .metadata_u32("qwen35.ssm.group_count", N_K_HEADS as u32)
        .metadata_u32("qwen35.ssm.time_step_rank", N_V_HEADS as u32)
        .metadata_u32("qwen35.ssm.inner_size", inner() as u32)
        .metadata_u32("tokenizer.ggml.eos_token_id", EOS_TOKEN_ID);

    // ── `prism.hadamard.*` fold contract ────────────────────────────────
    let mut widths = vec![HIDDEN, heads_width(), inner(), INTERMEDIATE];
    widths.sort_unstable();
    widths.dedup();
    let mut sign_values: Vec<i32> = Vec::new();
    for width in &widths {
        for i in 0..*width {
            sign_values.push(if i % 3 == 0 { -1 } else { 1 });
        }
    }
    let mut weight_names = vec!["output.weight".to_string()];
    for layer in 0..N_LAYERS {
        let folded: &[&str] = if is_full(layer) {
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
    builder
        .metadata_u32("prism.hadamard.version", 1)
        .metadata_u32("prism.hadamard.block_size", HADAMARD_BLOCK as u32)
        .metadata_str(
            "prism.hadamard.transform",
            "normalized-sylvester-walsh-hadamard",
        )
        .metadata_str("prism.hadamard.axis", "input-last-dimension")
        .metadata_str("prism.hadamard.sign_mode", "explicit")
        .metadata(
            "prism.hadamard.sign_widths",
            MetadataWriteValue::ArrayU32(widths.iter().map(|w| *w as u32).collect()),
        )
        .metadata(
            "prism.hadamard.sign_values",
            MetadataWriteValue::ArrayI32(sign_values),
        )
        .metadata(
            "prism.hadamard.weight_names",
            MetadataWriteValue::ArrayStr(weight_names),
        )
        .metadata(
            "prism.hadamard.inverse_weight_names",
            MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
        )
        .metadata(
            "prism.hadamard.gdn_v_grouped",
            MetadataWriteValue::Bool(true),
        );

    // ── Tensors ──────────────────────────────────────────────────────────
    let mut seed = 1u64;
    let mut next_seed = || {
        seed = seed.wrapping_add(7919);
        seed
    };

    builder.tensor(
        "token_embd.weight",
        &[HIDDEN as u64, VOCAB as u64],
        FixtureQuant::PQ2_0,
        next_seed(),
    )?;
    builder.tensor(
        "output.weight",
        &[HIDDEN as u64, VOCAB as u64],
        FixtureQuant::PQ2_0,
        next_seed(),
    )?;
    builder.tensor_raw(
        "output_norm.weight",
        &[HIDDEN as u64],
        TensorType::F32,
        f32_bytes(&vec![1.0f32; HIDDEN]),
    );

    for layer in 0..N_LAYERS {
        let blk = |suffix: &str| format!("blk.{layer}.{suffix}");

        builder.tensor_raw(
            &blk("attn_norm.weight"),
            &[HIDDEN as u64],
            TensorType::F32,
            f32_bytes(&vec![1.0f32; HIDDEN]),
        );
        builder.tensor_raw(
            &blk("post_attention_norm.weight"),
            &[HIDDEN as u64],
            TensorType::F32,
            f32_bytes(&vec![1.0f32; HIDDEN]),
        );
        builder.tensor(
            &blk("ffn_gate.weight"),
            &[HIDDEN as u64, INTERMEDIATE as u64],
            FixtureQuant::PQ2_0,
            next_seed(),
        )?;
        builder.tensor(
            &blk("ffn_up.weight"),
            &[HIDDEN as u64, INTERMEDIATE as u64],
            FixtureQuant::PQ2_0,
            next_seed(),
        )?;
        builder.tensor(
            &blk("ffn_down.weight"),
            &[INTERMEDIATE as u64, HIDDEN as u64],
            FixtureQuant::PQ2_0,
            next_seed(),
        )?;

        if is_full(layer) {
            builder.tensor(
                &blk("attn_q.weight"),
                &[HIDDEN as u64, (heads_width() * 2) as u64],
                FixtureQuant::PQ2_0,
                next_seed(),
            )?;
            builder.tensor(
                &blk("attn_k.weight"),
                &[HIDDEN as u64, (N_KV_HEADS * HEAD_DIM) as u64],
                FixtureQuant::PQ2_0,
                next_seed(),
            )?;
            builder.tensor(
                &blk("attn_v.weight"),
                &[HIDDEN as u64, (N_KV_HEADS * HEAD_DIM) as u64],
                FixtureQuant::PQ2_0,
                next_seed(),
            )?;
            builder.tensor(
                &blk("attn_output.weight"),
                &[heads_width() as u64, HIDDEN as u64],
                FixtureQuant::PQ2_0,
                next_seed(),
            )?;
            builder.tensor_raw(
                &blk("attn_q_norm.weight"),
                &[HEAD_DIM as u64],
                TensorType::F32,
                f32_bytes(&vec![1.0f32; HEAD_DIM]),
            );
            builder.tensor_raw(
                &blk("attn_k_norm.weight"),
                &[HEAD_DIM as u64],
                TensorType::F32,
                f32_bytes(&vec![1.0f32; HEAD_DIM]),
            );
        } else {
            builder.tensor(
                &blk("attn_qkv.weight"),
                &[HIDDEN as u64, conv_dim() as u64],
                FixtureQuant::PQ2_0,
                next_seed(),
            )?;
            builder.tensor(
                &blk("attn_gate.weight"),
                &[HIDDEN as u64, inner() as u64],
                FixtureQuant::PQ2_0,
                next_seed(),
            )?;
            builder.tensor(
                &blk("ssm_out.weight"),
                &[inner() as u64, HIDDEN as u64],
                FixtureQuant::PQ2_0,
                next_seed(),
            )?;
            builder.tensor_raw(
                &blk("ssm_alpha.weight"),
                &[HIDDEN as u64, N_V_HEADS as u64],
                TensorType::BF16,
                bf16_bytes(&ramp(HIDDEN * N_V_HEADS, next_seed())),
            );
            builder.tensor_raw(
                &blk("ssm_beta.weight"),
                &[HIDDEN as u64, N_V_HEADS as u64],
                TensorType::BF16,
                bf16_bytes(&ramp(HIDDEN * N_V_HEADS, next_seed())),
            );
            builder.tensor_raw(
                &blk("ssm_conv1d.weight"),
                &[CONV_KERNEL as u64, conv_dim() as u64],
                TensorType::F32,
                f32_bytes(&ramp(CONV_KERNEL * conv_dim(), next_seed())),
            );
            // `A = -exp(A_log)` must be negative on every element — never
            // routed through the generic `[-1, 1)` random weights.
            let ssm_a: Vec<f32> = (0..N_V_HEADS).map(|h| -0.25 - (h as f32) * 0.5).collect();
            builder.tensor_raw(
                &blk("ssm_a"),
                &[N_V_HEADS as u64],
                TensorType::F32,
                f32_bytes(&ssm_a),
            );
            let dt_bias: Vec<f32> = (0..N_V_HEADS).map(|h| -2.0 + (h as f32) * 0.125).collect();
            builder.tensor_raw(
                &blk("ssm_dt.bias"),
                &[N_V_HEADS as u64],
                TensorType::F32,
                f32_bytes(&dt_bias),
            );
            builder.tensor_raw(
                &blk("ssm_norm.weight"),
                &[STATE as u64],
                TensorType::F32,
                f32_bytes(&vec![1.0f32; STATE]),
            );
        }
    }

    builder.build()
}

/// A complete synthetic `qwen35` GGUF: PQ2_0 weights, Hadamard-folded,
/// grouped v-heads, `tokenizer.ggml.eos_token_id = 3` (see this module's doc
/// comment for the full contract and the public shape constants).
///
/// The infallible, spec'd entry point: a thin wrapper over
/// [`try_synthetic_qwen35_gguf`] so every caller of this deterministic,
/// zero-input fixture builder does not have to handle a `Result` an
/// out-of-tree caller can never meaningfully recover from anyway (there is
/// no smaller unit of input to retry with — the whole "input" is this
/// module's own constants).
///
/// # Panics
///
/// Never on a normal call — see [`try_synthetic_qwen35_gguf`]'s `# Errors`
/// section for the invariant this relies on (every shape constant in this
/// module is a multiple of `QK_PQ2_0`, and the layer loop never repeats a
/// tensor name). A panic here means this module's own constants were edited
/// into an inconsistent state, not that a caller passed bad input.
#[must_use]
pub fn synthetic_qwen35_gguf() -> Vec<u8> {
    match try_synthetic_qwen35_gguf() {
        Ok(bytes) => bytes,
        Err(e) => unreachable!(
            "oxibonsai_testkit::qwen35_fixture's own fixed shape constants must satisfy \
             try_synthetic_qwen35_gguf's documented invariant, but building it failed: {e}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::reader::GgufFile;

    #[test]
    fn synthetic_qwen35_gguf_is_deterministic() {
        assert_eq!(synthetic_qwen35_gguf(), synthetic_qwen35_gguf());
    }

    /// The infallible wrapper must be exactly its fallible half, unwrapped —
    /// no divergent behaviour, no silently-swallowed error.
    #[test]
    fn try_synthetic_qwen35_gguf_is_ok_and_byte_equal_to_the_infallible_wrapper() {
        let via_try =
            try_synthetic_qwen35_gguf().expect("must succeed for this module's fixed shapes");
        assert_eq!(via_try, synthetic_qwen35_gguf());
    }

    #[test]
    fn synthetic_qwen35_gguf_parses_and_carries_the_full_contract() {
        let bytes = synthetic_qwen35_gguf();
        assert!(bytes.starts_with(b"GGUF"));
        let file = GgufFile::parse(&bytes).expect("the real reader must parse this fixture");

        assert_eq!(
            file.metadata
                .get_string("general.architecture")
                .expect("architecture key"),
            "qwen35"
        );
        assert_eq!(
            file.metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(oxibonsai_core::MetadataValue::as_u32),
            Some(3)
        );
        assert_eq!(
            file.metadata
                .get("qwen35.full_attention_interval")
                .and_then(oxibonsai_core::MetadataValue::as_u32),
            Some(4)
        );
        assert_eq!(
            file.metadata
                .get("prism.hadamard.gdn_v_grouped")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        let weight_names = file
            .metadata
            .get("prism.hadamard.weight_names")
            .and_then(|v| v.as_array())
            .expect("prism.hadamard.weight_names");
        assert!(!weight_names.is_empty());

        // Globals.
        for name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
            assert!(
                file.tensors.get(name).is_some(),
                "missing global tensor {name}"
            );
        }

        // Every layer's tensor set matches its kind (full-attention on every
        // 4th layer, Gated-DeltaNet otherwise).
        for layer in 0..N_LAYERS {
            let blk = |suffix: &str| format!("blk.{layer}.{suffix}");
            for common in ["attn_norm.weight", "post_attention_norm.weight"] {
                assert!(
                    file.tensors.get(&blk(common)).is_some(),
                    "layer {layer}: missing {common}"
                );
            }
            if is_full(layer) {
                for name in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                ] {
                    assert!(
                        file.tensors.get(&blk(name)).is_some(),
                        "full-attention layer {layer}: missing {name}"
                    );
                }
                assert!(file.tensors.get(&blk("attn_qkv.weight")).is_none());
            } else {
                for name in [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_out.weight",
                    "ssm_alpha.weight",
                    "ssm_beta.weight",
                    "ssm_conv1d.weight",
                    "ssm_a",
                    "ssm_dt.bias",
                    "ssm_norm.weight",
                ] {
                    assert!(
                        file.tensors.get(&blk(name)).is_some(),
                        "linear-attention layer {layer}: missing {name}"
                    );
                }
                assert!(file.tensors.get(&blk("attn_q.weight")).is_none());
            }
        }
    }

    /// The public shape constants are exactly what the file declares, so a
    /// test asserting geometry against them asserts it against the file.
    #[test]
    fn the_public_shape_constants_describe_the_file() {
        let bytes = synthetic_qwen35_gguf();
        let file = GgufFile::parse(&bytes).expect("parse");
        assert_eq!(
            file.metadata
                .get_string("general.name")
                .expect("general.name"),
            MODEL_NAME
        );
        let u32_of = |key: &str| {
            file.metadata
                .get(key)
                .and_then(oxibonsai_core::MetadataValue::as_u32)
        };
        for (key, value) in [
            ("qwen35.embedding_length", HIDDEN),
            ("qwen35.feed_forward_length", INTERMEDIATE),
            ("qwen35.block_count", N_LAYERS),
            ("qwen35.attention.head_count", N_HEADS),
            ("qwen35.attention.head_count_kv", N_KV_HEADS),
            ("qwen35.attention.key_length", HEAD_DIM),
            ("qwen35.vocab_size", VOCAB),
            ("qwen35.context_length", CONTEXT_LENGTH),
            ("qwen35.ssm.state_size", STATE),
            ("qwen35.ssm.group_count", N_K_HEADS),
            ("qwen35.ssm.time_step_rank", N_V_HEADS),
            ("qwen35.ssm.conv_kernel", CONV_KERNEL),
            ("qwen35.ssm.inner_size", inner()),
            ("prism.hadamard.block_size", HADAMARD_BLOCK),
        ] {
            assert_eq!(u32_of(key), u32::try_from(value).ok(), "{key}");
        }
        assert_eq!(u32_of("tokenizer.ggml.eos_token_id"), Some(EOS_TOKEN_ID));
        assert_eq!(full_layer_count(), 2);
        assert_eq!(
            (0..N_LAYERS).filter(|layer| is_full(*layer)).count(),
            full_layer_count()
        );
        assert_eq!(conv_dim(), 2 * STATE * N_K_HEADS + inner());
        assert_eq!(heads_width(), N_HEADS * HEAD_DIM);
    }

    #[test]
    fn ssm_a_is_negative_on_every_linear_layer_element() {
        let bytes = synthetic_qwen35_gguf();
        let file = GgufFile::parse(&bytes).expect("parse");
        for layer in 0..N_LAYERS {
            if is_full(layer) {
                continue;
            }
            let name = format!("blk.{layer}.ssm_a");
            let data = file.tensor_data(&name).expect("ssm_a tensor data");
            assert_eq!(data.len(), N_V_HEADS * 4, "ssm_a: {name} byte length");
            for chunk in data.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                assert!(v < 0.0, "{name}: element {v} is not negative");
            }
        }
    }

    #[test]
    fn f32_to_bf16_bits_matches_the_upper_16_bits_for_exactly_representable_values() {
        // 1.0f32 is exactly representable in bf16 (no rounding needed): its
        // upper 16 bits alone must round-trip.
        assert_eq!(f32_to_bf16_bits(1.0), (1.0f32.to_bits() >> 16) as u16);
        assert_eq!(f32_to_bf16_bits(0.0), 0);
        assert_eq!(f32_to_bf16_bits(-2.0), ((-2.0f32).to_bits() >> 16) as u16);
    }

    #[test]
    fn ramp_is_deterministic_and_bounded() {
        let a = ramp(64, 42);
        let b = ramp(64, 42);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| (-1.0..1.0).contains(v)));
    }
}
