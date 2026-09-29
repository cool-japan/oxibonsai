//! Integration tests for the synthetic hybrid GGUF fixture generator and
//! its embedded f64 scalar reference model (`bonsai2-design.md`
//! §7.1/§7.2/§8.2, findings T-06/T-07).
//!
//! See `fixtures/hybrid_gguf.rs`'s module doc for exactly what "matches the
//! f64 reference model" means here. The real hybrid loader
//! (`oxibonsai_model::hybrid::model::HybridModel`) lives in this same
//! crate and is exercised against this fixture in
//! `hybrid_forward_parity_tests.rs`, which is why the checks in *this*
//! file stay at the generator/reference level: byte fidelity of the
//! written GGUF, the f64 reference's own internal self-consistency, and —
//! below — the f64 primitives' agreement with the real, SIMD `oxibonsai_kernels`
//! implementations they model.
//!
//! Every test name below contains `hybrid_fixture` so the package gate's
//! substring filter (`cargo test -p oxibonsai-model --all-features
//! hybrid_fixture`) actually selects it.

#[path = "fixtures/hybrid_gguf.rs"]
mod hybrid_gguf;

use std::collections::BTreeSet;

use half::f16;
use oxibonsai_core::gguf::quant_resolve::{
    compute_extents, resolve_type_42_with_sample, LegacyVersionTag, OrderEvidence,
};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::gguf::writer::TensorType;
use oxibonsai_core::quant_ternary::{
    sniff_sample_byte_cap, sniff_two_bit_layout, TwoBitLayout, SNIFF_DEFAULT_BLOCKS,
};
use oxibonsai_core::{
    count_plus_two_codes, BlockPQ2_0, BlockPTQ1_0, BlockQ1_0G128, BlockQ2_0G64, BlockTQ2_0_g128,
};
use oxibonsai_kernels::gated_delta_net::{gdn_step_with, GdnDims, GdnGates, GdnHeadOrder, GdnPath};
use oxibonsai_kernels::hadamard::fwht_forward_signed;
use oxibonsai_kernels::norms::{l2_norm_simd, rms_norm_gated_simd};
use oxibonsai_kernels::rope_mrope::rope_partial_splithalf_simd;
use oxibonsai_kernels::ssm_ops::causal_conv1d_k4_decode;

use hybrid_gguf::{
    all_variant_specs, apply_partial_rope_f64, build, build_invalid_ungrouped_fixture,
    causal_conv1d_step_f64, fwht_forward_signed_f64, fwht_round_trip_check,
    gated_rms_norm_head_f64, gdn_cross_check, gdn_step_fused_f64, l2_norm_f64, partial_rope_check,
    HybridFixtureSpec, Xorshift64Star, ALL_QUANT_TYPES, FFN, HADAMARD_BLOCK_SIZE,
    HADAMARD_BLOCK_SIZE_WIDE, HEAD_DIM, HIDDEN, ROPE_DIM, T_TOKENS, VOCAB,
};

const BASE_SEED: u64 = 0xB0A5_1234_5678_9ABC;

fn assert_all_finite(values: &[f64], context: &str) {
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite(), "{context}[{i}] = {v} is not finite");
    }
}

/// Slice tensor `name`'s exact raw bytes out of the whole file, using the
/// *known* writer-side `quant`/`shape` to compute the byte length ourselves
/// rather than trusting `GgufFile::tensor_data`'s length.
///
/// `GgufFile::tensor_data` derives its length from `info.data_size()`,
/// which is computed from the tensor's *parsed* type — and ggml id 42 is
/// inherently ambiguous (`GgufTensorType::from_id` always resolves it to
/// the legacy `TQ2_0_g128` reading, design §1.2). A `Q2_0_g64`-encoded
/// tensor also carries wire id 42, so the plain parser reports the wrong
/// (34-byte, not 18-byte) block geometry for it and `tensor_data` returns
/// the wrong slice length. Resolving that ambiguity correctly is
/// `oxibonsai_core::gguf::quant_resolve`'s job (already exercised by this
/// file's `..._bytes_sniff_as_..` tests via the real `sniff_two_bit_layout`);
/// this helper only needs the length *this test itself* already knows it
/// asked the writer for.
fn raw_tensor_bytes<'a>(
    file_bytes: &'a [u8],
    parsed: &GgufFile<'_>,
    name: &str,
    quant: TensorType,
    shape: &[u64],
) -> &'a [u8] {
    let info = parsed.tensors.require(name).expect("tensor present");
    let expected_len = quant.row_bytes(shape) as usize;
    let start = parsed.data_offset + info.offset as usize;
    &file_bytes[start..start + expected_len]
}

// ═════════════════════════════════════════════════════════════════════════
// Generator: all 24 variants build, parse and self-report consistently.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_all_24_canonical_variants_build_and_parse() {
    let specs = all_variant_specs(BASE_SEED);
    assert_eq!(
        specs.len(),
        24,
        "6 quant formats x 2 hadamard x 2 grouped must be the full cross product"
    );

    // Every spec's (quant, hadamard, grouped) triple must be distinct.
    let mut seen = BTreeSet::new();
    for spec in &specs {
        let key = (
            format!("{:?}", spec.quant),
            spec.hadamard,
            spec.gdn_v_grouped,
        );
        assert!(seen.insert(key.clone()), "duplicate variant {key:?}");
    }

    for spec in &specs {
        let fixture = build(spec).unwrap_or_else(|e| {
            panic!(
                "build() failed for {:?} hadamard={} grouped={}: {e}",
                spec.quant, spec.hadamard, spec.gdn_v_grouped
            )
        });
        assert!(fixture.path.exists(), "fixture file must exist on disk");

        // Real, already-merged core config parsing.
        assert_eq!(fixture.cfg.base.architecture, "qwen35");
        assert_eq!(fixture.cfg.full_attention_interval, 4);
        let full: Vec<usize> = (0..8)
            .filter(|&i| fixture.cfg.is_full_attention(i))
            .collect();
        assert_eq!(
            full,
            vec![3, 7],
            "full-attention layers must be exactly {{3, 7}}"
        );
        assert_eq!(fixture.cfg.num_full_layers(), 2);
        assert_eq!(fixture.cfg.num_linear_layers(), 6);
        assert_eq!(
            fixture.dims.v_per_k(),
            if spec.gdn_v_grouped { 2 } else { 1 }
        );

        assert_eq!(
            fixture.hadamard.is_some(),
            spec.hadamard,
            "hadamard presence must match the spec"
        );
        if let Some(had) = &fixture.hadamard {
            assert_eq!(had.block_size, HADAMARD_BLOCK_SIZE);
            assert_eq!(had.gdn_v_grouped, spec.gdn_v_grouped);
            assert!(had.is_inverse("token_embd.weight"));
            assert!(had.is_folded("output.weight"));
            assert!(
                had.is_folded("blk.3.attn_q.weight"),
                "full layer 3 attn_q must be folded"
            );
            assert!(
                had.is_folded("blk.0.attn_qkv.weight"),
                "linear layer 0 attn_qkv must be folded"
            );
            assert!(
                !had.is_folded("blk.0.attn_norm.weight"),
                "norms are never folded"
            );
            assert!(
                !had.is_folded("blk.0.ssm_alpha.weight"),
                "ssm_alpha is never folded"
            );
        }

        // Embedded f64 reference forward: sane and complete.
        assert_eq!(fixture.reference.token_ids.len(), T_TOKENS);
        assert_eq!(fixture.reference.logits.len(), T_TOKENS);
        for (t, row) in fixture.reference.logits.iter().enumerate() {
            assert_eq!(row.len(), VOCAB, "token {t} logits width");
            assert_all_finite(row, &format!("logits[{t}]"));
        }
        for (t, row) in fixture.reference.final_hidden.iter().enumerate() {
            assert_eq!(row.len(), HIDDEN, "token {t} final_hidden width");
            assert_all_finite(row, &format!("final_hidden[{t}]"));
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Byte fidelity: dequantizing what was written reproduces the planned
// values exactly, via the real, already-merged block codecs.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_dequant_reproduces_planned_values_for_every_quant_format() {
    for &quant in &ALL_QUANT_TYPES {
        let spec = HybridFixtureSpec {
            quant,
            hadamard: true,
            gdn_v_grouped: true,
            seed: BASE_SEED ^ 0xAAAA,
            hidden: HIDDEN,
            hadamard_block: HADAMARD_BLOCK_SIZE,
        };
        let fixture = build(&spec).expect("fixture should build");
        let bytes = std::fs::read(&fixture.path).expect("read fixture file");
        let parsed = GgufFile::parse(&bytes).expect("parse fixture file");

        // Representative sample: one global tensor, one full-layer tensor,
        // one linear-layer tensor.
        for name in [
            "output.weight",
            "blk.3.attn_q.weight",
            "blk.0.attn_qkv.weight",
        ] {
            let shape = fixture
                .planned_shape(name)
                .unwrap_or_else(|| panic!("{name} planned shape present"))
                .to_vec();
            let raw = raw_tensor_bytes(&bytes, &parsed, name, quant, &shape);
            let expected = fixture
                .planned_values(name)
                .unwrap_or_else(|| panic!("{name} planned values present"));
            let mut got = vec![0.0f32; expected.len()];
            match quant {
                TensorType::F32 => {
                    for (i, chunk) in raw.chunks_exact(4).enumerate() {
                        got[i] = f32::from_le_bytes(chunk.try_into().expect("4 bytes"));
                    }
                }
                TensorType::PQ2_0 => {
                    let blocks = BlockPQ2_0::slice_from_bytes(raw).expect("cast PQ2_0");
                    BlockPQ2_0::dequant(blocks, &mut got).expect("dequant PQ2_0");
                    for b in blocks {
                        assert_eq!(
                            b.count_plus_two(),
                            0,
                            "{name}: ternary source must never emit 0b11"
                        );
                    }
                }
                TensorType::PTQ1_0 => {
                    let blocks = BlockPTQ1_0::slice_from_bytes(raw).expect("cast PTQ1_0");
                    BlockPTQ1_0::dequant(blocks, &mut got).expect("dequant PTQ1_0");
                }
                TensorType::Q2_0G64 => {
                    let blocks = BlockQ2_0G64::slice_from_bytes(raw).expect("cast Q2_0G64");
                    BlockQ2_0G64::dequant(blocks, &mut got).expect("dequant Q2_0G64");
                    for b in blocks {
                        assert_eq!(
                            b.count_plus_two(),
                            0,
                            "{name}: ternary source must never emit 0b11"
                        );
                    }
                }
                TensorType::TQ2_0_g128 => {
                    let blocks = BlockTQ2_0_g128::slice_from_bytes(raw).expect("cast TQ2_0_g128");
                    BlockTQ2_0_g128::dequant(blocks, &mut got).expect("dequant TQ2_0_g128");
                    for b in blocks {
                        // Only `b.qs` is 2-bit-coded data (`b.d` is an f16
                        // scale, not codes) — scanning it in isolation is
                        // what makes this assertion meaningful rather than
                        // contaminated by the trailing scale bytes.
                        assert_eq!(
                            count_plus_two_codes(&b.qs),
                            0,
                            "{name}: ternary source must never emit 0b11"
                        );
                    }
                }
                TensorType::Q1_0G128 => {
                    let blocks = BlockQ1_0G128::slice_from_bytes(raw).expect("cast Q1_0_g128");
                    for (i, block) in blocks.iter().enumerate() {
                        for j in 0..128 {
                            got[i * 128 + j] = block.weight(j);
                        }
                    }
                }
                other => panic!("unexpected quant choice in this fixture: {other:?}"),
            }
            assert_eq!(
                got, expected,
                "{name} ({quant:?}): write -> parse -> dequant must reproduce the generator's values exactly"
            );
        }
    }
}

/// Plain-F32 tensors (norms, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`) are
/// always `F32` regardless of the variant's quant choice.
#[test]
fn hybrid_fixture_plain_f32_tensors_round_trip_regardless_of_quant_choice() {
    let spec = HybridFixtureSpec {
        quant: TensorType::PTQ1_0,
        hadamard: false,
        gdn_v_grouped: false,
        seed: BASE_SEED ^ 0xBEEF,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("fixture should build");
    let bytes = std::fs::read(&fixture.path).expect("read fixture file");
    let parsed = GgufFile::parse(&bytes).expect("parse fixture file");

    for name in [
        "output_norm.weight",
        "blk.3.attn_norm.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_conv1d.weight",
    ] {
        let raw = parsed
            .tensor_data(name)
            .unwrap_or_else(|_| panic!("{name} present"));
        let expected = fixture.planned_values(name).expect("planned values");
        let got: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        assert_eq!(got, expected, "{name}: plain F32 round trip must be exact");
    }
    // ssm_a must be negative (A = -exp(A_log), consumed as-is).
    for &v in fixture.planned_values("blk.0.ssm_a").expect("ssm_a") {
        assert!(v < 0.0, "ssm_a values must be negative, got {v}");
    }
}

/// `ssm_alpha`/`ssm_beta` are always `BF16` and never folded, regardless of
/// the variant's quant choice.
#[test]
fn hybrid_fixture_bf16_tensors_round_trip_and_are_never_folded() {
    let spec = HybridFixtureSpec {
        quant: TensorType::Q2_0G64,
        hadamard: true,
        gdn_v_grouped: true,
        seed: BASE_SEED ^ 0xC0DE,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("fixture should build");
    let bytes = std::fs::read(&fixture.path).expect("read fixture file");
    let parsed = GgufFile::parse(&bytes).expect("parse fixture file");

    for name in ["blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"] {
        let raw = parsed
            .tensor_data(name)
            .unwrap_or_else(|_| panic!("{name} present"));
        let expected = fixture.planned_values(name).expect("planned values");
        let got: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| oxibonsai_core::bf16::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(got, expected, "{name}: bf16 round trip must be exact");
        assert!(
            !fixture
                .hadamard
                .as_ref()
                .expect("hadamard on")
                .is_folded(name),
            "{name} must never be in the folded set"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════
// id-42 disambiguation (T-06 item 3): the two group geometries id 42 can
// carry are structurally distinguishable straight from the bytes this
// generator writes.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_tq2_0_g128_bytes_sniff_as_qs_first() {
    let spec = HybridFixtureSpec {
        quant: TensorType::TQ2_0_g128,
        hadamard: false,
        gdn_v_grouped: false,
        seed: BASE_SEED ^ 1,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("fixture should build");
    let bytes = std::fs::read(&fixture.path).expect("read fixture file");
    let parsed = GgufFile::parse(&bytes).expect("parse fixture file");
    let shape = fixture
        .planned_shape("output.weight")
        .expect("shape")
        .to_vec();
    let raw = raw_tensor_bytes(&bytes, &parsed, "output.weight", spec.quant, &shape);
    let verdict = sniff_two_bit_layout(raw, SNIFF_DEFAULT_BLOCKS);
    assert_eq!(verdict, TwoBitLayout::QsFirst34);
}

#[test]
fn hybrid_fixture_q2_0_g64_bytes_sniff_as_d_first_18() {
    let spec = HybridFixtureSpec {
        quant: TensorType::Q2_0G64,
        hadamard: false,
        gdn_v_grouped: false,
        seed: BASE_SEED ^ 2,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("fixture should build");
    let bytes = std::fs::read(&fixture.path).expect("read fixture file");
    let parsed = GgufFile::parse(&bytes).expect("parse fixture file");
    let shape = fixture
        .planned_shape("output.weight")
        .expect("shape")
        .to_vec();
    let raw = raw_tensor_bytes(&bytes, &parsed, "output.weight", spec.quant, &shape);
    let verdict = sniff_two_bit_layout(raw, SNIFF_DEFAULT_BLOCKS);
    assert_eq!(verdict, TwoBitLayout::DFirst18);
}

/// `PQ2_0` (ggml id 142) is never ambiguous at the wire-id level the way id
/// 42 is, but its 34-byte block struct is `d`-first — the exact byte-swapped
/// opposite of `TQ2_0_g128`'s qs-first 34-byte struct pinned above (T-06
/// item 2's required negative test: "our existing `BlockTQ2_0_g128`
/// qs-first layout is not accidentally reused — the two are byte-swapped").
/// This pins the third of the three structurally-distinguishable 34-byte
/// readings `sniff_two_bit_layout` must tell apart.
#[test]
fn hybrid_fixture_pq2_0_bytes_sniff_as_d_first_34() {
    let spec = HybridFixtureSpec {
        quant: TensorType::PQ2_0,
        hadamard: false,
        gdn_v_grouped: false,
        seed: BASE_SEED ^ 3,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("fixture should build");
    let bytes = std::fs::read(&fixture.path).expect("read fixture file");
    let parsed = GgufFile::parse(&bytes).expect("parse fixture file");
    let shape = fixture
        .planned_shape("output.weight")
        .expect("shape")
        .to_vec();
    let raw = raw_tensor_bytes(&bytes, &parsed, "output.weight", spec.quant, &shape);
    let verdict = sniff_two_bit_layout(raw, SNIFF_DEFAULT_BLOCKS);
    assert_eq!(verdict, TwoBitLayout::DFirst34);
}

// ═════════════════════════════════════════════════════════════════════════
// id-42 resolution via the real entry point (design §7.2 "id-42 resolution"
// row): `general.quantization_version` plus
// `resolve_type_42_with_sample` — not just the lower-level
// `sniff_two_bit_layout` probe above — must resolve each wire-id-42 fixture
// variant to the right `GgufTensorType`.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_tq2_0_g128_resolves_via_resolve_type_42_with_sample() {
    let spec = HybridFixtureSpec {
        quant: TensorType::TQ2_0_g128,
        hadamard: false,
        gdn_v_grouped: false,
        seed: BASE_SEED ^ 4,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("fixture should build");
    let bytes = std::fs::read(&fixture.path).expect("read fixture file");
    let parsed = GgufFile::parse(&bytes).expect("parse fixture file");

    let qver = parsed.metadata.get("general.quantization_version");
    assert_eq!(
        LegacyVersionTag::from_value(qver),
        LegacyVersionTag::Tq2_0G128,
        "write_gguf must spell general.quantization_version as the legacy string for TQ2_0_g128"
    );

    let infos: Vec<_> = parsed
        .tensors
        .sorted_by_offset()
        .into_iter()
        .cloned()
        .collect();
    let data_len = (parsed.data.len() as u64).saturating_sub(parsed.data_offset as u64);
    let extents = compute_extents(&infos, data_len).expect("extents");
    let alignment = parsed
        .metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as u64;

    let shape = fixture
        .planned_shape("output.weight")
        .expect("shape")
        .to_vec();
    let sample = raw_tensor_bytes(&bytes, &parsed, "output.weight", spec.quant, &shape);
    let sample = &sample[..sample
        .len()
        .min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];

    let resolved = resolve_type_42_with_sample(&infos, alignment, qver, Some(&extents), sample)
        .unwrap_or_else(|e| panic!("resolve_type_42_with_sample failed: {e}"));
    assert_eq!(resolved.tensor_type, GgufTensorType::TQ2_0_g128);
    // `OrderEvidence::LegacyVersionString` is only the fallback used when the
    // structural sniff itself is ambiguous (an all-zero/tiny sample); a real
    // generated fixture's ternary weights sniff unambiguously as
    // `QsFirst34`, so the data itself — not the legacy string tag — is what
    // settles it here, even though the string tag (asserted above) agrees.
    assert_eq!(resolved.order_evidence, OrderEvidence::DataSniff);
}

#[test]
fn hybrid_fixture_q2_0_g64_resolves_via_resolve_type_42_with_sample() {
    let spec = HybridFixtureSpec {
        quant: TensorType::Q2_0G64,
        hadamard: false,
        gdn_v_grouped: false,
        seed: BASE_SEED ^ 5,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("fixture should build");
    let bytes = std::fs::read(&fixture.path).expect("read fixture file");
    let parsed = GgufFile::parse(&bytes).expect("parse fixture file");

    let qver = parsed.metadata.get("general.quantization_version");
    assert_eq!(
        LegacyVersionTag::from_value(qver),
        LegacyVersionTag::Numeric,
        "write_gguf must spell general.quantization_version as numeric u32 2 for Q2_0_g64"
    );

    let infos: Vec<_> = parsed
        .tensors
        .sorted_by_offset()
        .into_iter()
        .cloned()
        .collect();
    let data_len = (parsed.data.len() as u64).saturating_sub(parsed.data_offset as u64);
    let extents = compute_extents(&infos, data_len).expect("extents");
    let alignment = parsed
        .metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as u64;

    let shape = fixture
        .planned_shape("output.weight")
        .expect("shape")
        .to_vec();
    let sample = raw_tensor_bytes(&bytes, &parsed, "output.weight", spec.quant, &shape);
    let sample = &sample[..sample
        .len()
        .min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];

    let resolved = resolve_type_42_with_sample(&infos, alignment, qver, Some(&extents), sample)
        .unwrap_or_else(|e| panic!("resolve_type_42_with_sample failed: {e}"));
    assert_eq!(resolved.tensor_type, GgufTensorType::Q2_0G64);
    assert_eq!(resolved.order_evidence, OrderEvidence::NotApplicable);
}

// ═════════════════════════════════════════════════════════════════════════
// Determinism and cleanup.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_same_seed_is_byte_identical_and_reference_identical() {
    let spec = HybridFixtureSpec {
        quant: TensorType::PQ2_0,
        hadamard: true,
        gdn_v_grouped: true,
        seed: 424242,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let a = build(&spec).expect("first build");
    let b = build(&spec).expect("second build");
    assert_ne!(
        a.path, b.path,
        "each build must land at a distinct temp path"
    );
    let bytes_a = std::fs::read(&a.path).expect("read a");
    let bytes_b = std::fs::read(&b.path).expect("read b");
    assert_eq!(
        bytes_a, bytes_b,
        "same seed must produce byte-identical files"
    );
    assert_eq!(
        a.reference.logits, b.reference.logits,
        "same seed must produce identical f64 reference logits"
    );
    assert_eq!(a.reference.token_ids, b.reference.token_ids);
}

#[test]
fn hybrid_fixture_different_seeds_produce_different_bytes_and_reference() {
    let spec_a = HybridFixtureSpec {
        quant: TensorType::PQ2_0,
        hadamard: true,
        gdn_v_grouped: true,
        seed: 1,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let spec_b = HybridFixtureSpec { seed: 2, ..spec_a };
    let a = build(&spec_a).expect("build a");
    let b = build(&spec_b).expect("build b");
    let bytes_a = std::fs::read(&a.path).expect("read a");
    let bytes_b = std::fs::read(&b.path).expect("read b");
    assert_ne!(
        bytes_a, bytes_b,
        "different seeds must produce different files"
    );
    assert_ne!(
        a.reference.logits, b.reference.logits,
        "different seeds must produce different f64 reference logits"
    );
}

#[test]
fn hybrid_fixture_temp_file_is_removed_on_drop() {
    let spec = HybridFixtureSpec {
        quant: TensorType::F32,
        hadamard: false,
        gdn_v_grouped: false,
        seed: 7,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("build");
    let path = fixture.path.clone();
    assert!(path.exists());
    // The path must land in the platform temp directory, never a
    // hardcoded absolute path.
    assert!(path.starts_with(std::env::temp_dir()));
    drop(fixture);
    assert!(!path.exists(), "fixture file must be removed once dropped");
}

#[test]
fn hybrid_fixture_invalid_ungrouped_negative_fixture_builds_and_is_flagged_unsupported() {
    let fixture = build_invalid_ungrouped_fixture(99, TensorType::PQ2_0).expect(
        "the negative fixture itself must still build (it is a valid GGUF file; only a future loader is expected to refuse it)",
    );
    assert!(fixture.path.exists());
    assert_eq!(
        fixture.v_per_k, 2,
        "the negative fixture must have v_per_k > 1"
    );

    let bytes = std::fs::read(&fixture.path).expect("read");
    let parsed = GgufFile::parse(&bytes).expect("parse");
    let hadamard = oxibonsai_core::hadamard_config::HadamardConfig::from_metadata(&parsed.metadata)
        .expect("hadamard metadata should parse")
        .expect("hadamard must be present for this fixture");
    assert!(
        !hadamard.gdn_v_grouped,
        "the negative fixture's metadata must claim ungrouped while v_per_k > 1 \
         (the combination design §3.3 says a hybrid loader must refuse)"
    );

    let path = fixture.path.clone();
    drop(fixture);
    assert!(!path.exists());
}

// ═════════════════════════════════════════════════════════════════════════
// Internal self-checks: independently re-derived math agrees with itself.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_fwht_forward_then_inverse_is_the_identity() {
    for &(width, block) in &[
        (HADAMARD_BLOCK_SIZE, HADAMARD_BLOCK_SIZE),
        (HIDDEN, HADAMARD_BLOCK_SIZE),
        (FFN, HADAMARD_BLOCK_SIZE),
    ] {
        let (x, recovered) = fwht_round_trip_check(width as u64 * 7 + 3, width, block);
        assert_eq!(x.len(), recovered.len());
        for (i, (&xi, &ri)) in x.iter().zip(&recovered).enumerate() {
            assert!(
                (xi - ri).abs() < 1e-9,
                "width={width} block={block} index {i}: {xi} vs {ri}"
            );
        }
    }
}

#[test]
fn hybrid_fixture_gdn_fused_naive_and_quadratic_forms_agree() {
    let (fused, naive, quadratic) = gdn_cross_check(0x5EED, HEAD_DIM, HEAD_DIM);
    assert_eq!(fused.len(), 2);
    assert_eq!(naive.len(), 2);
    assert_eq!(quadratic.len(), 2);
    for t in 0..2 {
        for j in 0..HEAD_DIM {
            let f = fused[t][j];
            let n = naive[t][j];
            let q = quadratic[t][j];
            assert!(f.is_finite() && n.is_finite() && q.is_finite());
            assert!(
                (f - n).abs() < 1e-9,
                "fused vs naive-3-pass diverge at t={t} j={j}: {f} vs {n}"
            );
            assert!(
                (f - q).abs() < 1e-6,
                "fused vs O(T^2) closed form diverge at t={t} j={j}: {f} vs {q}"
            );
        }
    }
}

#[test]
fn hybrid_fixture_partial_rope_leaves_the_tail_untouched_and_rotates_the_head() {
    let head_dim = HEAD_DIM;
    let (before, after) = partial_rope_check(0x1234, 5, ROPE_DIM, head_dim, 10_000.0);
    assert_eq!(before.len(), head_dim);
    assert_eq!(after.len(), head_dim);
    // The untouched tail must be bit-identical.
    assert_eq!(
        &before[ROPE_DIM..],
        &after[ROPE_DIM..],
        "dims past n_rot must be byte-identical after RoPE"
    );
    // The rotated head must actually have changed (a no-op rotation would
    // hide a wrong pairing).
    let changed = before[..ROPE_DIM]
        .iter()
        .zip(&after[..ROPE_DIM])
        .any(|(&b, &a)| (b - a).abs() > 1e-9);
    assert!(
        changed,
        "RoPE at a nonzero position must change the rotated dims"
    );

    // Position 0 is the fixed point of every RoPE angle (theta * 0 == 0),
    // so it must be a true no-op — a useful complementary check that the
    // position argument is actually threaded through.
    let (before0, after0) = partial_rope_check(0x1234, 0, ROPE_DIM, head_dim, 10_000.0);
    assert_eq!(before0, after0, "position 0 must be an exact RoPE no-op");
}

// ═════════════════════════════════════════════════════════════════════════
// Wiring sanity: turning Hadamard / grouping on actually changes the
// forward output (a silently-skipped rotation or permutation would not).
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_hadamard_on_changes_the_forward_output_vs_off() {
    let base = HybridFixtureSpec {
        quant: TensorType::PTQ1_0,
        hadamard: true,
        gdn_v_grouped: true,
        seed: 555,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let with_had = build(&base).expect("build with hadamard");
    let without_had = build(&HybridFixtureSpec {
        hadamard: false,
        ..base
    })
    .expect("build without hadamard");
    assert_ne!(
        with_had.reference.logits, without_had.reference.logits,
        "toggling Hadamard must change the reference forward output"
    );
}

#[test]
fn hybrid_fixture_v_head_grouping_changes_v_per_k_and_the_forward_output() {
    let base = HybridFixtureSpec {
        quant: TensorType::PQ2_0,
        hadamard: true,
        gdn_v_grouped: true,
        seed: 777,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let grouped = build(&base).expect("build grouped");
    let trivial = build(&HybridFixtureSpec {
        gdn_v_grouped: false,
        ..base
    })
    .expect("build trivial");
    assert_eq!(grouped.dims.v_per_k(), 2);
    assert_eq!(trivial.dims.v_per_k(), 1);
    assert_ne!(
        grouped.reference.logits, trivial.reference.logits,
        "a different V-head grouping (different n_k_heads/state shape) must change the output"
    );
}

// ═════════════════════════════════════════════════════════════════════════
// Determinism of the RNG itself and basic tensor-name coverage.
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn hybrid_fixture_tensor_set_covers_every_binding_the_hybrid_loader_needs() {
    let spec = HybridFixtureSpec {
        quant: TensorType::PQ2_0,
        hadamard: true,
        gdn_v_grouped: true,
        seed: 42,
        hidden: HIDDEN,
        hadamard_block: HADAMARD_BLOCK_SIZE,
    };
    let fixture = build(&spec).expect("build");
    let names: BTreeSet<&str> = fixture.tensor_names().collect();

    for global in ["token_embd.weight", "output_norm.weight", "output.weight"] {
        assert!(names.contains(global), "missing global tensor {global}");
    }
    // Full-attention layers (3, 7).
    for layer in [3, 7] {
        for suffix in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let name = format!("blk.{layer}.{suffix}");
            assert!(names.contains(name.as_str()), "missing {name}");
        }
    }
    // Linear-attention layers (0, 1, 2, 4, 5, 6).
    for layer in [0, 1, 2, 4, 5, 6] {
        for suffix in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_alpha.weight",
            "ssm_beta.weight",
            "ssm_conv1d.weight",
            "ssm_a",
            "ssm_dt.bias",
            "ssm_norm.weight",
            "ssm_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let name = format!("blk.{layer}.{suffix}");
            assert!(names.contains(name.as_str()), "missing {name}");
        }
    }
    // No full-attention tensor on a linear layer, and vice versa.
    assert!(!names.contains("blk.0.attn_q.weight"));
    assert!(!names.contains("blk.3.attn_qkv.weight"));
}

/// `d = 1.0` (exact in `f16`) is what makes the hand-packed `Q1_0_g128`
/// encoder lossless; pin that assumption directly against the `half` crate
/// rather than only indirectly through the dequant-matches-planned test.
#[test]
fn hybrid_fixture_q1_0_g128_scale_is_exactly_representable_in_f16() {
    assert_eq!(f16::from_f32(1.0).to_f32(), 1.0);
}

// ═════════════════════════════════════════════════════════════════════════
// Kernel cross-checks: every f64 primitive `fixtures/hybrid_gguf.rs`'s
// reference forward is built from is checked above only against itself, in
// a different algebraic form (`fwht_round_trip_check`/`gdn_cross_check`/
// `partial_rope_check`). The six tests below instead compare each f64
// primitive directly against the real, SIMD `oxibonsai_kernels` function it
// models, on deterministic random inputs neither side has seen before — an
// `f32` kernel vs an `f64` scalar reference, so agreement is checked to a
// relative tolerance (`REL_TOL`) rather than bit-exactly.
const REL_TOL: f32 = 1e-4;

fn assert_allclose_f32_f64(actual: &[f32], reference: &[f64], context: &str) {
    assert_eq!(actual.len(), reference.len(), "{context}: length mismatch");
    for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        let b32 = b as f32;
        let diff = (a - b32).abs();
        let bound = REL_TOL * b32.abs().max(1.0);
        assert!(
            diff <= bound,
            "{context}[{i}]: kernel={a} f64_reference={b} diff={diff:.3e} > bound={bound:.3e}"
        );
    }
}

/// `fwht_forward_signed` (kernel) vs `fwht_forward_signed_f64` (reference),
/// at both the narrow fixtures' block size and the real 27B's own 1024 —
/// the width the 24 canonical fixture variants never exercise.
#[test]
fn hybrid_fixture_fwht_forward_signed_matches_the_kernel_at_block_128_and_1024() {
    for block in [HADAMARD_BLOCK_SIZE, HADAMARD_BLOCK_SIZE_WIDE] {
        let mut rng = Xorshift64Star::new(0xF00D_0000 ^ block as u64);
        let n_blocks = 3;
        let width = block * n_blocks;
        let x64: Vec<f64> = (0..width).map(|_| rng.next_range_f64(-2.0, 2.0)).collect();
        let signs64: Vec<f64> = (0..width)
            .map(|_| {
                if rng.next_range_f64(-1.0, 1.0) < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect();
        let reference = fwht_forward_signed_f64(&x64, &signs64, block);

        let mut x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let signs32: Vec<f32> = signs64.iter().map(|&v| v as f32).collect();
        fwht_forward_signed(&mut x32, &signs32, block).expect("fwht_forward_signed");

        assert_allclose_f32_f64(&x32, &reference, &format!("fwht block={block}"));
    }
}

/// `l2_norm_simd` (kernel) vs `l2_norm_f64` (reference).
#[test]
fn hybrid_fixture_l2_norm_simd_matches_the_f64_reference() {
    let mut rng = Xorshift64Star::new(0x1207_0000);
    let n = HEAD_DIM;
    let eps = 1e-6;
    let x64: Vec<f64> = (0..n).map(|_| rng.next_range_f64(-3.0, 3.0)).collect();
    let reference = l2_norm_f64(&x64, eps);

    let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    let mut out32 = vec![0.0f32; n];
    l2_norm_simd(&x32, &mut out32, eps as f32).expect("l2_norm_simd");

    assert_allclose_f32_f64(&out32, &reference, "l2_norm");
}

/// `rms_norm_gated_simd` (kernel) vs `gated_rms_norm_head_f64` (reference).
#[test]
fn hybrid_fixture_rms_norm_gated_simd_matches_the_f64_reference() {
    let mut rng = Xorshift64Star::new(0x6A7E_D000);
    let n = HEAD_DIM;
    let eps = 1e-6;
    let o64: Vec<f64> = (0..n).map(|_| rng.next_range_f64(-2.0, 2.0)).collect();
    let z64: Vec<f64> = (0..n).map(|_| rng.next_range_f64(-2.0, 2.0)).collect();
    let w64: Vec<f64> = (0..n).map(|_| rng.next_range_f64(0.5, 1.5)).collect();
    let reference = gated_rms_norm_head_f64(&o64, &z64, &w64, eps);

    let o32: Vec<f32> = o64.iter().map(|&v| v as f32).collect();
    let z32: Vec<f32> = z64.iter().map(|&v| v as f32).collect();
    let w32: Vec<f32> = w64.iter().map(|&v| v as f32).collect();
    let mut out32 = vec![0.0f32; n];
    rms_norm_gated_simd(&o32, &w32, &z32, &mut out32, eps as f32).expect("rms_norm_gated_simd");

    assert_allclose_f32_f64(&out32, &reference, "rms_norm_gated");
}

/// `rope_partial_splithalf_simd` (kernel, precomputed cos/sin) vs
/// `apply_partial_rope_f64` (reference, computes `theta` internally) — the
/// same `theta = pos * freq_base^(-2*ic/n_rot)` formula, fed to the kernel
/// as the `cos`/`sin` tables it expects.
#[test]
fn hybrid_fixture_rope_partial_splithalf_simd_matches_the_f64_reference() {
    let mut rng = Xorshift64Star::new(0x40E5_0000);
    let head_dim = HEAD_DIM;
    let n_rot = ROPE_DIM;
    let pos = 17usize;
    let freq_base = 10_000.0f64;
    let x64: Vec<f64> = (0..head_dim)
        .map(|_| rng.next_range_f64(-2.0, 2.0))
        .collect();
    let mut reference = x64.clone();
    apply_partial_rope_f64(&mut reference, pos, n_rot, freq_base);

    let half = n_rot / 2;
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for (ic, (c, s)) in cos.iter_mut().zip(sin.iter_mut()).enumerate() {
        let theta = pos as f64 * freq_base.powf(-2.0 * ic as f64 / n_rot as f64);
        *c = theta.cos() as f32;
        *s = theta.sin() as f32;
    }
    let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    let mut out32 = vec![0.0f32; head_dim];
    rope_partial_splithalf_simd(&x32, &mut out32, head_dim, n_rot, &cos, &sin)
        .expect("rope_partial_splithalf_simd");

    assert_allclose_f32_f64(&out32, &reference, "rope_partial_splithalf");
}

/// `causal_conv1d_k4_decode` (kernel) vs `causal_conv1d_step_f64`
/// (reference) — same channel-major `[channels][KC-1]` state layout and
/// `[channels][KC]` weight layout, so this cross-check translates only the
/// flat-vs-nested `Vec` representation, not the semantics.
#[test]
fn hybrid_fixture_causal_conv1d_k4_decode_matches_the_f64_reference() {
    const KC: usize = 4;
    let channels = HEAD_DIM;
    let mut rng = Xorshift64Star::new(0xC04F_0000);

    let state_nested: Vec<Vec<f64>> = (0..channels)
        .map(|_| (0..KC - 1).map(|_| rng.next_range_f64(-1.0, 1.0)).collect())
        .collect();
    let w_nested: Vec<Vec<f64>> = (0..channels)
        .map(|_| (0..KC).map(|_| rng.next_range_f64(-1.0, 1.0)).collect())
        .collect();
    let x64: Vec<f64> = (0..channels)
        .map(|_| rng.next_range_f64(-1.0, 1.0))
        .collect();

    let mut state_ref = state_nested.clone();
    let reference = causal_conv1d_step_f64(&mut state_ref, &x64, &w_nested, channels, KC);

    let mut state_flat: Vec<f32> = state_nested
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    let w_flat: Vec<f32> = w_nested
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    let mut out32 = vec![0.0f32; channels];
    causal_conv1d_k4_decode(&mut state_flat, &x32, &w_flat, &mut out32)
        .expect("causal_conv1d_k4_decode");

    assert_allclose_f32_f64(&out32, &reference, "causal_conv1d_k4_decode output");

    // The post-step state must agree too — the same shift-and-append the
    // reference performs on `state_ref` in place.
    let state_ref_flat: Vec<f64> = state_ref.into_iter().flatten().collect();
    assert_allclose_f32_f64(
        &state_flat,
        &state_ref_flat,
        "causal_conv1d_k4_decode state",
    );
}

/// `gdn_step_with` (kernel) vs `gdn_step_fused_f64` (reference), at a
/// single head (`n_k_heads = n_v_heads = 1`) so the cross-check is of the
/// per-head recurrence math itself, not the multi-head grouping/tiling this
/// crate's own `gdn_cross_check`/`hybrid_fixture_v_head_grouping_...` tests
/// already cover separately.
#[test]
fn hybrid_fixture_gdn_step_with_matches_the_f64_reference() {
    let mut rng = Xorshift64Star::new(0x6D17_0000);
    let head_k_dim = HEAD_DIM;
    let head_v_dim = HEAD_DIM;
    let state_len = head_k_dim * head_v_dim;

    let state64: Vec<f64> = (0..state_len)
        .map(|_| rng.next_range_f64(-1.0, 1.0))
        .collect();
    let q64: Vec<f64> = (0..head_k_dim)
        .map(|_| rng.next_range_f64(-1.0, 1.0))
        .collect();
    let k64: Vec<f64> = (0..head_k_dim)
        .map(|_| rng.next_range_f64(-1.0, 1.0))
        .collect();
    let v64: Vec<f64> = (0..head_v_dim)
        .map(|_| rng.next_range_f64(-1.0, 1.0))
        .collect();
    let alpha_raw = rng.next_range_f64(-2.0, 2.0);
    let beta_raw = rng.next_range_f64(-2.0, 2.0);
    let dt_bias = rng.next_range_f64(-1.0, 1.0);
    // `a_neg` (`ssm_a`, `A = -exp(A_log)`) is always negative (design §7.0).
    let a_neg = -rng.next_range_f64(0.1, 2.0);

    let mut state_ref = state64.clone();
    let (out_ref, _decay, _delta) = gdn_step_fused_f64(
        &mut state_ref,
        &q64,
        &k64,
        &v64,
        alpha_raw,
        beta_raw,
        dt_bias,
        a_neg,
        head_k_dim,
        head_v_dim,
    );

    let mut state32: Vec<f32> = state64.iter().map(|&v| v as f32).collect();
    let q32: Vec<f32> = q64.iter().map(|&v| v as f32).collect();
    let k32: Vec<f32> = k64.iter().map(|&v| v as f32).collect();
    let v32: Vec<f32> = v64.iter().map(|&v| v as f32).collect();
    let alpha_raw32 = [alpha_raw as f32];
    let beta_raw32 = [beta_raw as f32];
    let dt_bias32 = [dt_bias as f32];
    let a_neg32 = [a_neg as f32];
    let gates = GdnGates::bonsai2(&alpha_raw32, &beta_raw32, &dt_bias32, &a_neg32);
    let dims = GdnDims::new(1, 1, head_k_dim, head_v_dim);
    let mut out32 = vec![0.0f32; head_v_dim];
    gdn_step_with(
        &mut state32,
        &q32,
        &k32,
        &v32,
        &gates,
        &mut out32,
        &dims,
        GdnHeadOrder::default(),
        GdnPath::default(),
    )
    .expect("gdn_step_with");

    assert_allclose_f32_f64(&out32, &out_ref, "gdn_step_with output");
    assert_allclose_f32_f64(&state32, &state_ref, "gdn_step_with state");
}
