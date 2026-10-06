//! Bitwise golden-block parity for the PrismML Bonsai 2 quantization formats,
//! plus the id-42 resolution acceptance checks.
//!
//! The reference implementations below are **hand transliterations** of
//! `ggml-quants.c` from the PrismML llama.cpp fork (`dequantize_row_pq2_0`
//! `:494-512`, `dequantize_row_q2_0` `:474-492`, `dequantize_row_ptq1_0`
//! `:2255-2285`, `quantize_row_ptq1_0_ref` `:2205-2253`). They deliberately
//! keep the C loop shape — including the inert first stage, the `n`-outer /
//! `m`-inner interleave and the `uint8_t` wrap — so that a refactor of the
//! production code that "tidies" any of those away is caught here rather than
//! by a wrong answer on a 7 GB model.
//!
//! No binary fixtures: every golden block is generated in-code from a fixed
//! seed, so the suite is reproducible and the repo stays clean.

use std::path::PathBuf;

use half::f16;
use oxibonsai_core::error::BonsaiError;
use oxibonsai_core::gguf::quant_resolve::{
    compute_extents, resolve_type_42, resolve_type_42_with_sample, LegacyVersionTag, OrderEvidence,
    AMBIGUOUS_TYPE_ID,
};
use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
use oxibonsai_core::quant_prism::{
    transcode_ptq1_0_to_tq2, BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, BLOCK_PQ2_0_BYTES,
    BLOCK_PTQ1_0_BYTES, BLOCK_Q2_0_G64_BYTES, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64,
};
use oxibonsai_core::quant_ternary::{
    sniff_sample_byte_cap, sniff_two_bit_layout, BlockTQ2_0_g128, TwoBitLayout,
    SNIFF_DEFAULT_BLOCKS,
};

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic pseudo-random source (no dev-dependency, no binary fixture)
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x2545_F491_4F6C_DD1D)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }

    /// A positive f16 scale in roughly the range real PrismML files use.
    fn next_scale(&mut self) -> f16 {
        let v = 0.005 + (self.next_u32() % 4096) as f32 * 1.0e-5;
        f16::from_f32(v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference transliterations
// ─────────────────────────────────────────────────────────────────────────────

/// `dequantize_row_pq2_0` / `dequantize_row_q2_0`, verbatim:
///
/// ```c
/// const uint8_t q = (x[i].qs[j/4] >> ((j%4)*2)) & 0x03;
/// y[i*qk + j] = ((int)q - 1) * d;   // 00=-1 01=0 10=+1 11=+2
/// ```
fn ref_dequant_two_bit(qs: &[u8], d: f16, qk: usize, out: &mut Vec<f32>) {
    let d = d.to_f32();
    for j in 0..qk {
        let byte_index = j / 4;
        let bit_offset = (j % 4) * 2;
        let q = (qs[byte_index] >> bit_offset) & 0x03;
        out.push((q as i32 - 1) as f32 * d);
    }
}

/// `dequantize_row_ptq1_0`, verbatim (`ptq1_0_stages = {32, 16, 8}`,
/// `sizeof(qs) == 24`, `sizeof(qh) == 2`).
// A deliberate 1:1 transliteration of C: the loop shape IS the specification.
// An idiomatic rewrite would lose the inert first stage, the `n`-outer /
// `m`-inner interleave and the explicit ceiling division that this suite
// exists to pin down, so the style lints that would "tidy" them away are
// switched off for this function only.
#[allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
fn ref_dequant_ptq1_0(qs: &[u8; 24], qh: &[u8; 2], d: f16, out: &mut Vec<f32>) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    const STAGES: [usize; 3] = [32, 16, 8];
    let d = d.to_f32();

    let mut j = 0usize;
    for s in 0..3usize {
        let c = STAGES[s];
        while j + c <= qs.len() {
            for n in 0..5usize {
                for m in 0..c {
                    // `uint8_t q = x[i].qs[j + m] * pow3[n];`
                    let q: u8 = ((qs[j + m] as u32 * POW3[n] as u32) & 0xFF) as u8;
                    let xi: i16 = (((q as u16) * 3) >> 8) as i16;
                    out.push((xi - 1) as f32 * d);
                }
            }
            j += c;
        }
    }
    for n in 0..4usize {
        for h in 0..qh.len() {
            let q: u8 = ((qh[h] as u32 * POW3[n] as u32) & 0xFF) as u8;
            let xi: i16 = (((q as u16) * 3) >> 8) as i16;
            out.push((xi - 1) as f32 * d);
        }
    }
}

/// `quantize_row_ptq1_0_ref`, verbatim, including the moving source pointer.
// A deliberate 1:1 transliteration of C: the loop shape IS the specification.
// An idiomatic rewrite would lose the inert first stage, the `n`-outer /
// `m`-inner interleave and the explicit ceiling division that this suite
// exists to pin down, so the style lints that would "tidy" them away are
// switched off for this function only.
#[allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
fn ref_quantize_ptq1_0(x: &[f32]) -> ([u8; 24], [u8; 2], f16) {
    let mut amax = 0.0f32;
    for &v in x.iter().take(QK_PTQ1_0) {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let d = amax;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    const STAGES: [usize; 3] = [32, 16, 8];
    let mut qs = [0u8; 24];
    let mut base = 0usize; // stands in for the C `x` pointer
    let mut j = 0usize;
    for s in 0..3usize {
        let c = STAGES[s];
        while j + c <= qs.len() {
            for m in 0..c {
                let mut q: u32 = 0;
                for n in 0..5usize {
                    let xi = (x[base + m + n * c] * id).round() as i32 + 1;
                    q = (q * 3 + xi as u32) & 0xFF;
                }
                qs[j + m] = ((q * 256 + 242) / 243) as u8;
            }
            base += 5 * c;
            j += c;
        }
    }

    let mut qh = [0u8; 2];
    for h in 0..2usize {
        let mut q: u32 = 0;
        for m in 0..4usize {
            let xi = (x[base + h + m * 2] * id).round() as i32 + 1;
            q = (q * 3 + xi as u32) & 0xFF;
        }
        q = (q * 3) & 0xFF;
        qh[h] = ((q * 256 + 242) / 243) as u8;
    }

    (qs, qh, f16::from_f32(amax))
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitwise dequant parity — §7.2 rows 1-4
// ─────────────────────────────────────────────────────────────────────────────

const GOLDEN_BLOCKS: usize = 256;

#[test]
fn pq2_0_dequant_matches_the_ggml_reference_bitwise() {
    let mut rng = Lcg::new(0xB0_5A_17);
    let mut blocks = Vec::with_capacity(GOLDEN_BLOCKS);
    let mut expected: Vec<f32> = Vec::with_capacity(GOLDEN_BLOCKS * QK_PQ2_0);
    for _ in 0..GOLDEN_BLOCKS {
        let mut qs = [0u8; 32];
        for b in qs.iter_mut() {
            *b = rng.next_u8();
        }
        let d = rng.next_scale();
        ref_dequant_two_bit(&qs, d, QK_PQ2_0, &mut expected);
        blocks.push(BlockPQ2_0 { d, qs });
    }

    let mut got = vec![0.0f32; GOLDEN_BLOCKS * QK_PQ2_0];
    BlockPQ2_0::dequant(&blocks, &mut got).expect("dequant");
    assert_bitwise(&expected, &got, "PQ2_0");
}

#[test]
fn q2_0_g64_dequant_matches_the_ggml_reference_bitwise() {
    let mut rng = Lcg::new(0x0064_6400);
    let mut blocks = Vec::with_capacity(GOLDEN_BLOCKS);
    let mut expected: Vec<f32> = Vec::with_capacity(GOLDEN_BLOCKS * QK_Q2_0_G64);
    for _ in 0..GOLDEN_BLOCKS {
        let mut qs = [0u8; 16];
        for b in qs.iter_mut() {
            *b = rng.next_u8();
        }
        let d = rng.next_scale();
        ref_dequant_two_bit(&qs, d, QK_Q2_0_G64, &mut expected);
        blocks.push(BlockQ2_0G64 { d, qs });
    }

    let mut got = vec![0.0f32; GOLDEN_BLOCKS * QK_Q2_0_G64];
    BlockQ2_0G64::dequant(&blocks, &mut got).expect("dequant");
    assert_bitwise(&expected, &got, "Q2_0_g64");
}

#[test]
fn ptq1_0_dequant_matches_the_ggml_reference_bitwise() {
    let mut rng = Lcg::new(0x17_43_00);
    let mut blocks = Vec::with_capacity(GOLDEN_BLOCKS);
    let mut expected: Vec<f32> = Vec::with_capacity(GOLDEN_BLOCKS * QK_PTQ1_0);
    for _ in 0..GOLDEN_BLOCKS {
        let mut qs = [0u8; 24];
        for b in qs.iter_mut() {
            *b = rng.next_u8();
        }
        let qh = [rng.next_u8(), rng.next_u8()];
        let d = rng.next_scale();
        ref_dequant_ptq1_0(&qs, &qh, d, &mut expected);
        blocks.push(BlockPTQ1_0 { qs, qh, d });
    }

    let mut got = vec![0.0f32; GOLDEN_BLOCKS * QK_PTQ1_0];
    BlockPTQ1_0::dequant(&blocks, &mut got).expect("dequant");
    assert_bitwise(&expected, &got, "PTQ1_0");
}

#[test]
fn ptq1_0_quantize_matches_the_ggml_reference_bitwise() {
    let mut rng = Lcg::new(0x5E_ED_42);
    for _ in 0..GOLDEN_BLOCKS {
        let scale = 0.01 + (rng.next_u32() % 100) as f32 * 0.01;
        let input: Vec<f32> = (0..QK_PTQ1_0)
            .map(|_| match rng.next_u32() % 3 {
                0 => -scale,
                1 => 0.0,
                _ => scale,
            })
            .collect();
        let (qs, qh, d) = ref_quantize_ptq1_0(&input);
        let got = BlockPTQ1_0::quantize(&input).expect("quantize");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].qs, qs, "qs must match quantize_row_ptq1_0_ref");
        assert_eq!(got[0].qh, qh, "qh must match quantize_row_ptq1_0_ref");
        assert_eq!(
            got[0].d.to_bits(),
            d.to_bits(),
            "scale must match bit-for-bit"
        );
    }
}

/// A hand-computed block, checked against the base-3 arithmetic by hand so
/// the whole chain does not rest on one transliteration.
#[test]
fn ptq1_0_hand_computed_block() {
    // Element order: e[n*16 + m] is trit n of qs[m].
    // Choose qs[0] to hold trits (2, 0, 1, 1, 1) reading most-significant
    // first: word = 2*81 + 0*27 + 1*9 + 1*3 + 1 = 175, byte = ceil(175*256/243) = 185.
    let word: u32 = 2 * 81 + 9 + 3 + 1;
    assert_eq!(word, 175);
    let byte = (word * 256).div_ceil(243) as u8;
    assert_eq!(byte, 185);

    let mut block = BlockPTQ1_0::zeroed();
    block.qs[0] = byte;
    block.d = f16::from_f32(1.0);

    let mut codes = [0u8; 128];
    block.decode_codes(&mut codes);
    // trit 0 -> index 0, trit 1 -> index 16, trit 2 -> 32, trit 3 -> 48, trit 4 -> 64
    assert_eq!(codes[0], 2, "trit 0");
    assert_eq!(codes[16], 0, "trit 1");
    assert_eq!(codes[32], 1, "trit 2");
    assert_eq!(codes[48], 1, "trit 3");
    assert_eq!(codes[64], 1, "trit 4");

    let mut out = vec![0.0f32; 128];
    BlockPTQ1_0::dequant(&[block], &mut out).expect("dequant");
    assert_eq!(out[0], 1.0);
    assert_eq!(out[16], -1.0);
    assert_eq!(out[32], 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Transcode — §7.2 row 2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn transcode_is_bitwise_lossless_for_both_targets() {
    let mut rng = Lcg::new(0x7A_A0_C0);
    let mut inputs = Vec::with_capacity(GOLDEN_BLOCKS * QK_PTQ1_0);
    for _ in 0..GOLDEN_BLOCKS * QK_PTQ1_0 {
        inputs.push(match rng.next_u32() % 3 {
            0 => -0.375f32,
            1 => 0.0,
            _ => 0.375,
        });
    }
    let ptq = BlockPTQ1_0::quantize(&inputs).expect("quantize");
    assert_eq!(ptq.len(), GOLDEN_BLOCKS);

    let mut base = vec![0.0f32; inputs.len()];
    BlockPTQ1_0::dequant(&ptq, &mut base).expect("dequant");

    let mut pq = vec![BlockPQ2_0::zeroed(); ptq.len()];
    BlockPTQ1_0::transcode_to_pq2(&ptq, &mut pq).expect("transcode pq2");
    let mut via_pq = vec![0.0f32; inputs.len()];
    BlockPQ2_0::dequant(&pq, &mut via_pq).expect("dequant pq2");
    assert_bitwise(&base, &via_pq, "PTQ1_0 -> PQ2_0");

    let tq = transcode_ptq1_0_to_tq2(&ptq);
    let mut via_tq = vec![0.0f32; inputs.len()];
    BlockTQ2_0_g128::dequant(&tq, &mut via_tq).expect("dequant tq2");
    assert_bitwise(&base, &via_tq, "PTQ1_0 -> TQ2_0_g128");

    // The scale must survive bit-for-bit, not through an f32 round-trip.
    for (i, b) in ptq.iter().enumerate() {
        assert_eq!(pq[i].d.to_bits(), b.d.to_bits());
        assert_eq!(tq[i].d.to_bits(), b.d.to_bits());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-size guards — K-13
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn claimed_block_count_guards_reject_a_mis_typed_tensor() {
    // 18 blocks of 34 bytes == 34 blocks of 18 bytes == 612 bytes, so length
    // divisibility alone cannot separate the two readings — the *claimed*
    // block count must.
    let buf = vec![0u8; 612];
    assert_eq!(612 % BLOCK_PQ2_0_BYTES, 0);
    assert_eq!(612 % BLOCK_Q2_0_G64_BYTES, 0);

    assert!(BlockPQ2_0::slice_from_bytes_exact(&buf, 18).is_ok());
    assert!(BlockQ2_0G64::slice_from_bytes_exact(&buf, 34).is_ok());
    // A tensor of 18 g128 blocks read as 18 g64 blocks must fail loudly.
    match BlockQ2_0G64::slice_from_bytes_exact(&buf, 18) {
        Err(BonsaiError::InvalidQuantBlockSize {
            format,
            expected,
            actual,
        }) => {
            assert_eq!(format, "Q2_0_g64");
            assert_eq!(expected, 18 * BLOCK_Q2_0_G64_BYTES);
            assert_eq!(actual, 612);
        }
        other => panic!("expected InvalidQuantBlockSize, got {other:?}"),
    }
    assert!(BlockPTQ1_0::slice_from_bytes_exact(&buf, 18).is_err());
    assert_eq!(BLOCK_PTQ1_0_BYTES * 18, 504);
}

// ─────────────────────────────────────────────────────────────────────────────
// id-42 resolution on the real models (skipped when absent)
// ─────────────────────────────────────────────────────────────────────────────

fn models_dir() -> PathBuf {
    oxibonsai_testkit::workspace::models_dir()
}

/// What each real file must resolve to, from the block-layout probe (§0.2).
///
/// Only files present in the checkout are exercised; the repo does not ship
/// model weights, so a fresh clone (and any worktree) skips all of these.
const REAL_FILES: &[(&str, Option<GgufTensorType>, LegacyVersionTag)] = &[
    (
        "Ternary-Bonsai-1.7B.gguf",
        Some(GgufTensorType::TQ2_0_g128),
        LegacyVersionTag::Tq2_0G128,
    ),
    (
        "Ternary-Bonsai-8B.gguf",
        Some(GgufTensorType::TQ2_0_g128),
        LegacyVersionTag::Tq2_0G128,
    ),
    (
        "Ternary-Bonsai-27B-Q2_0.gguf",
        Some(GgufTensorType::Q2_0G128DFirst),
        LegacyVersionTag::Numeric,
    ),
    (
        "Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf",
        Some(GgufTensorType::Q2_0G64),
        LegacyVersionTag::Numeric,
    ),
    // No type-42 tensor: id 142/143 are unambiguous on their own.
    (
        "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        None,
        LegacyVersionTag::Numeric,
    ),
    (
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        None,
        LegacyVersionTag::Numeric,
    ),
];

#[test]
fn id_42_resolves_correctly_on_every_real_model_present() {
    let dir = models_dir();
    let mut checked = 0usize;
    for (file, expect, expect_tag) in REAL_FILES {
        let path = dir.join(file);
        if !path.exists() {
            continue;
        }
        let mmap = mmap_gguf_file(&path).expect("mmap");
        let gguf = GgufFile::parse(&mmap).expect("parse");
        let infos: Vec<_> = gguf
            .tensors
            .sorted_by_offset()
            .into_iter()
            .cloned()
            .collect();
        let qver = gguf.metadata.get("general.quantization_version");
        assert_eq!(
            LegacyVersionTag::from_value(qver),
            *expect_tag,
            "{file}: quantization_version spelling"
        );

        let has_42 = infos
            .iter()
            .any(|i| i.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID);
        match expect {
            None => {
                assert!(
                    !has_42,
                    "{file} was not expected to contain type-42 tensors"
                );
                checked += 1;
                continue;
            }
            Some(expected_type) => {
                assert!(has_42, "{file} must contain type-42 tensors");
                let data_len = (gguf.data.len() as u64).saturating_sub(gguf.data_offset as u64);
                let extents = compute_extents(&infos, data_len).expect("extents");
                let alignment = gguf
                    .metadata
                    .get("general.alignment")
                    .and_then(|v| v.as_u32())
                    .unwrap_or(32) as u64;
                let sample_name = infos
                    .iter()
                    .find(|i| i.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID)
                    .map(|i| i.name.clone())
                    .expect("type-42 tensor");
                let full = gguf.tensor_data(&sample_name).expect("tensor data");
                let sample = &full[..full.len().min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];

                let resolved =
                    resolve_type_42_with_sample(&infos, alignment, qver, Some(&extents), sample)
                        .unwrap_or_else(|e| panic!("{file}: resolve failed: {e}"));
                assert_eq!(
                    resolved.tensor_type, *expected_type,
                    "{file}: resolved the wrong id-42 reading"
                );
                assert_eq!(resolved.tensor_type.wire_id(), 42);
                assert_eq!(
                    resolved.block_bytes,
                    expected_type.block_bytes(),
                    "{file}: block bytes"
                );
                // Group-64 has only one byte order, so the sniff never
                // settles it — `resolve_type_42_with_sample` reports
                // `NotApplicable` there, not `DataSniff` (quant_resolve.rs).
                let expected_evidence = if resolved.block_size == 64 {
                    OrderEvidence::NotApplicable
                } else {
                    OrderEvidence::DataSniff
                };
                assert_eq!(
                    resolved.order_evidence, expected_evidence,
                    "{file}: order evidence"
                );
                checked += 1;
            }
        }
    }
    if checked == 0 {
        eprintln!(
            "skipping: no model GGUF present under {} (weights are not in the repo)",
            dir.display()
        );
    }
}

/// Resolution must never be attempted without evidence, even for a real file.
#[test]
fn a_no_extent_call_errors_on_a_real_model() {
    let path = models_dir().join("Ternary-Bonsai-1.7B.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let mmap = mmap_gguf_file(&path).expect("mmap");
    let gguf = GgufFile::parse(&mmap).expect("parse");
    let infos: Vec<_> = gguf
        .tensors
        .sorted_by_offset()
        .into_iter()
        .cloned()
        .collect();
    let qver = gguf.metadata.get("general.quantization_version");
    assert!(matches!(
        resolve_type_42(&infos, 32, qver, None),
        Err(BonsaiError::AmbiguousQuantType { type_id: 42, .. })
    ));
}

/// The structural sniff must agree with the probe on the real bytes.
#[test]
fn sniff_verdicts_match_the_probe_on_real_models() {
    let dir = models_dir();
    let cases: &[(&str, TwoBitLayout)] = &[
        ("Ternary-Bonsai-1.7B.gguf", TwoBitLayout::QsFirst34),
        ("Ternary-Bonsai-2-27B-PQ2_0.gguf", TwoBitLayout::DFirst34),
        ("Ternary-Bonsai-27B-Q2_0.gguf", TwoBitLayout::DFirst34),
    ];
    let mut checked = 0usize;
    for (file, expect) in cases {
        let path = dir.join(file);
        if !path.exists() {
            continue;
        }
        let mmap = mmap_gguf_file(&path).expect("mmap");
        let gguf = GgufFile::parse(&mmap).expect("parse");
        let name = gguf
            .tensors
            .sorted_by_offset()
            .into_iter()
            .find(|i| i.tensor_type.is_two_bit_packed())
            .map(|i| i.name.clone())
            .expect("a 2-bit tensor");
        let full = gguf.tensor_data(&name).expect("tensor data");
        let sample = &full[..full.len().min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];
        assert_eq!(
            sniff_two_bit_layout(sample, SNIFF_DEFAULT_BLOCKS),
            *expect,
            "{file}: sniff verdict on {name}"
        );
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipping: no model GGUF present under {}", dir.display());
    }
}

/// Legacy regression guard (§7.4, load-path half): the real 1.7B must keep
/// resolving to the qs-first legacy reading, its offsets must keep satisfying
/// the padded invariant, and a sample of its blocks must keep dequantizing to
/// the same values. Any change to `GgufTensorType`, the block structs or the
/// sniff that moved the 1.7B would land here.
#[test]
fn legacy_1_7b_decode_path_is_unchanged() {
    let path = models_dir().join("Ternary-Bonsai-1.7B.gguf");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }
    let mmap = mmap_gguf_file(&path).expect("mmap");
    let gguf = GgufFile::parse(&mmap).expect("parse");

    // Every type-42 tensor is the legacy 34-byte qs-first block.
    for (name, info) in gguf.tensors.iter() {
        if info.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID {
            assert_eq!(info.tensor_type.block_bytes(), 34, "{name}");
            assert_eq!(info.tensor_type.block_size(), 128, "{name}");
        }
    }

    // Offsets satisfy `offset[i] == Σ GGML_PAD(nbytes, 32)`.
    let mut running = 0u64;
    for info in gguf.tensors.sorted_by_offset() {
        assert_eq!(
            info.offset, running,
            "tensor '{}' breaks the running padded-sum invariant",
            info.name
        );
        running += info
            .padded_extent(32)
            .expect("padded extent must not overflow");
    }

    // The decoded values of a known tensor are stable.
    let name = gguf
        .tensors
        .sorted_by_offset()
        .into_iter()
        .find(|i| i.tensor_type.is_two_bit_packed())
        .map(|i| i.name.clone())
        .expect("a 2-bit tensor");
    let raw = gguf.tensor_data(&name).expect("tensor data");
    let n_blocks = 64usize;
    let blocks = BlockTQ2_0_g128::slice_from_bytes(&raw[..n_blocks * 34]).expect("cast");
    let mut out = vec![0.0f32; n_blocks * 128];
    BlockTQ2_0_g128::dequant(blocks, &mut out).expect("dequant");
    // Every decoded value is one of {-d, 0, +d} for its block's scale.
    for (b, chunk) in out.chunks_exact(128).enumerate() {
        let d = blocks[b].d.to_f32();
        assert!(d.is_finite() && d >= 0.0, "block {b} scale {d}");
        for &v in chunk {
            assert!(
                v == 0.0 || v == d || v == -d,
                "legacy decode emitted {v} outside {{-{d}, 0, {d}}}"
            );
        }
    }
    let checksum = out.iter().fold(0u64, |acc, &v| {
        acc.wrapping_mul(1_000_003).wrapping_add(v.to_bits() as u64)
    });
    eprintln!("legacy 1.7B first-{n_blocks}-block decode checksum: {checksum:#x}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic fixtures for the id-42 layouts that have no in-repo model
// ─────────────────────────────────────────────────────────────────────────────

/// Build a one-tensor GGUF carrying `blocks` blocks of `block_bytes` bytes
/// under ggml id 42, laid out with the scale where `d_first` says.
///
/// `qs` is filled with `0b10_01_00_01`, which is legal ternary under the
/// intended reading but decodes to a **negative** f16 (`0x9191`) under every
/// other one, so the structural sniff has real separation to find.
fn synth_id42_gguf(block_bytes: usize, d_first: bool, rows: u64, legacy_tag: bool) -> Vec<u8> {
    let (ty, block_size) = if block_bytes == 34 {
        (TensorType::TQ2_0_g128, 128u64)
    } else {
        (TensorType::Q2_0G64, 64u64)
    };
    let qs_len = block_bytes - 2;
    let blocks_per_row = 128 / block_size;
    let n_blocks = (rows * blocks_per_row) as usize;

    let mut data = vec![0u8; n_blocks * block_bytes];
    let d = f16::from_f32(0.0415).to_le_bytes();
    for i in 0..n_blocks {
        let base = i * block_bytes;
        let (s, c) = if d_first {
            (base, base + 2)
        } else {
            (base + qs_len, base)
        };
        data[s] = d[0];
        data[s + 1] = d[1];
        for k in 0..qs_len {
            data[c + k] = 0b10_01_00_01;
        }
    }

    let mut w = GgufWriter::new();
    w.add_metadata(
        "general.architecture",
        MetadataWriteValue::Str("qwen3".into()),
    );
    w.add_metadata(
        "general.quantization_version",
        if legacy_tag {
            MetadataWriteValue::Str("TQ2_0_G128".into())
        } else {
            MetadataWriteValue::U32(2)
        },
    );
    w.add_tensor(TensorEntry {
        name: "blk.0.ffn_up.weight".to_string(),
        shape: vec![128, rows],
        tensor_type: ty,
        data,
    });
    // A second tensor so the offset replay has more than one constraint.
    w.add_tensor(TensorEntry {
        name: "output_norm.weight".to_string(),
        shape: vec![128],
        tensor_type: TensorType::F32,
        data: vec![0u8; 512],
    });
    w.to_bytes().expect("write synthetic gguf")
}

fn resolve_synthetic(bytes: &[u8]) -> GgufTensorType {
    let gguf = GgufFile::parse(bytes).expect("parse synthetic gguf");
    let infos: Vec<_> = gguf
        .tensors
        .sorted_by_offset()
        .into_iter()
        .cloned()
        .collect();
    let data_len = (gguf.data.len() as u64).saturating_sub(gguf.data_offset as u64);
    let extents = compute_extents(&infos, data_len).expect("extents");
    let sample = gguf
        .tensor_data("blk.0.ffn_up.weight")
        .expect("sample tensor");
    resolve_type_42_with_sample(
        &infos,
        32,
        gguf.metadata.get("general.quantization_version"),
        Some(&extents),
        sample,
    )
    .expect("resolve")
    .tensor_type
}

/// All three readings of ggml id 42, end to end through the writer, the
/// reader, the offset replay and the structural sniff. The gen-1 `d`-first
/// g128 file and the mainline g64 file are not in the repo (only their 64 MB
/// header captures exist), so these fixtures are what pins those two paths.
#[test]
fn all_three_id_42_readings_resolve_from_a_synthetic_file() {
    // Legacy OxiBonsai: 34 B, qs first, string quantization_version.
    assert_eq!(
        resolve_synthetic(&synth_id42_gguf(34, false, 512, true)),
        GgufTensorType::TQ2_0_g128,
        "qs-first g128 must resolve to the legacy reading"
    );
    // PrismML gen-1: 34 B, d first, numeric quantization_version.
    assert_eq!(
        resolve_synthetic(&synth_id42_gguf(34, true, 512, false)),
        GgufTensorType::Q2_0G128DFirst,
        "d-first g128 must resolve to the gen-1 reading"
    );
    // Mainline block_q2_0: 18 B, d first, group 64.
    assert_eq!(
        resolve_synthetic(&synth_id42_gguf(18, true, 512, false)),
        GgufTensorType::Q2_0G64,
        "18-byte blocks must resolve to the mainline group-64 reading"
    );
    // The g64 reading still writes and reads id 42 on the wire.
    assert_eq!(GgufTensorType::Q2_0G64.wire_id(), 42);
}

/// Row counts that make the two geometries' padded sums coincide are
/// genuinely undecidable, and must be reported rather than guessed.
#[test]
fn synthetic_sweep_of_row_counts_never_resolves_incorrectly() {
    for rows in [64u64, 100, 128, 256, 333, 512, 777] {
        for (block_bytes, d_first, legacy, expect) in [
            (34usize, false, true, GgufTensorType::TQ2_0_g128),
            (34, true, false, GgufTensorType::Q2_0G128DFirst),
            (18, true, false, GgufTensorType::Q2_0G64),
        ] {
            let bytes = synth_id42_gguf(block_bytes, d_first, rows, legacy);
            let got = resolve_synthetic(&bytes);
            assert_eq!(
                got, expect,
                "rows={rows} block_bytes={block_bytes} d_first={d_first}"
            );
        }
    }
}

/// End-to-end regression for the truncation bias that made the group-64
/// sniff a no-op above the sample cap (design §1.3: "MANDATORY, not a
/// backstop").
///
/// Neither 27B `Q2_0` GGUF large enough to exercise this naturally is
/// available locally (see the wave's acceptance-gap note), so this in-code
/// buffer is what pins the mechanism: a group-64 tensor comfortably larger
/// than the sample cap must still sniff as `DFirst18`, not silently degrade
/// to `Ambiguous` because the caller's truncation length happened to split
/// its last 18-byte block.
#[test]
fn a_large_group_64_tensor_still_sniffs_clean_after_the_default_truncation() {
    // `GgufFile::tensor_data` itself slices by `TensorInfo::data_size()`,
    // which — like every parse-time read of a wire-id-42 tensor — assumes
    // the legacy group-128/34-byte geometry regardless of what the bytes
    // actually are (harmless for every *real* tensor here, which is always
    // megabytes, dwarfing the byte counts below; only a synthetic fixture
    // this close to the sample cap ever notices). So `full.len()` below is
    // `rows * 34`, not the `rows * 2 * 18` actually written, and it is
    // `rows * 34` that must clear the sample cap: 2200 rows * 34 = 74_800 B,
    // comfortably over the 68_544 B cap so truncation still actually
    // happens on top of that.
    let bytes = synth_id42_gguf(18, true, 2200, false);
    let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
    let infos: Vec<_> = gguf
        .tensors
        .sorted_by_offset()
        .into_iter()
        .cloned()
        .collect();
    let data_len = (gguf.data.len() as u64).saturating_sub(gguf.data_offset as u64);
    let extents = compute_extents(&infos, data_len).expect("extents");
    let full = gguf
        .tensor_data("blk.0.ffn_up.weight")
        .expect("sample tensor");
    let cap = sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS);
    assert!(full.len() > cap, "fixture must actually exceed the cap");
    let sample = &full[..full.len().min(cap)];

    // The discriminating assertion: the mandatory sniff must actually SETTLE
    // on `DFirst18`, not merely fail to raise an error — the group-64 resolve
    // arm below tolerates an `Ambiguous` verdict too, so `Ok(..)` alone would
    // not catch a regression of the truncation bug.
    assert_eq!(
        sniff_two_bit_layout(sample, SNIFF_DEFAULT_BLOCKS),
        TwoBitLayout::DFirst18,
        "the capped sample must still sniff cleanly as group-64"
    );

    let resolved = resolve_type_42_with_sample(
        &infos,
        32,
        gguf.metadata.get("general.quantization_version"),
        Some(&extents),
        sample,
    )
    .expect("a large, structurally clean group-64 sample must resolve");
    assert_eq!(resolved.tensor_type, GgufTensorType::Q2_0G64);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn assert_bitwise(expected: &[f32], got: &[f32], what: &str) {
    assert_eq!(expected.len(), got.len(), "{what}: length");
    for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            e.to_bits(),
            g.to_bits(),
            "{what}: element {i} differs ({e} vs {g})"
        );
    }
}
