//! Real-GGUF write -> parse -> dequant round-trip tests for the K-quant
//! formats, through the exact encoder `oxibonsai quantize` uses
//! (`encode_quantized_tensor` in `quantize.rs`).
//!
//! Split into its own sibling file purely for size (`quantize.rs` crossed
//! the 2000-line policy limit once these tests plus the FIX3-GGUF-WRITE
//! item 1(c) encoder arms were added) -- mirrors the
//! `model/weight_loaders.rs` + `model/weight_loaders/tests.rs` split
//! already in this crate. A child module of `quantize` (declared
//! `#[cfg(test)] mod kquant_roundtrip_tests;` there), so `use super::*;`
//! below resolves exactly as it did when this was an inline
//! `mod kquant_writer_roundtrip { .. }` block nested inside
//! `quantize::scale_rule_tests` -- no caller-visible change, no test
//! renamed or altered.
//!
//! FIX3-GGUF-WRITE item 1(d), "the point of the exercise": before this
//! package, nothing in the workspace could WRITE a `Q2_K`/`Q3_K`/`Q8_K`
//! tensor at all (`encode_quantized_tensor` had no arm for them), so the
//! wave-2 K-quant dequant rewrite (core-gguf-K0) had no real-GGUF
//! round-trip test for any of the six K-quant formats -- only an in-memory
//! `BlockQxK::quantize`/`dequant` check
//! (`oxibonsai-core/tests/quant_k_tests.rs`) and a writer geometry table
//! (`gguf::writer::tests::new_tensor_types_have_the_ggml_geometry`). These
//! tests close that gap for all six formats in one harness: encode through
//! the exact function `oxibonsai quantize` uses (`encode_quantized_tensor`),
//! write a REAL GGUF file to disk via `GgufWriter::write` (not the
//! in-memory-only `to_bytes()`), read it straight back with
//! `GgufFile::parse`, dequantize through the production `BlockQxK::dequant`,
//! and assert the format's own quantization-step error bound -- not merely
//! that the bytes survived.

use std::io::Write as _;

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry};
use oxibonsai_core::quant_k::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ8K, QK_K};
use oxibonsai_core::quant_k_ext::{BlockQ5K, BlockQ6K};

use super::*;

/// A smooth, non-constant probe signal (period-256 sine plus a slow
/// cosine, ~[-1, 1]) so every 16/32/64-element K-quant sub-block
/// exercises a real local dynamic range. A constant or all-zero
/// input would make every sub-scale trivially zero and pass under
/// any decoder, buggy or not — the exact trap
/// `oxibonsai-core/tests/kquant_ggml_golden.rs`'s own `smooth_signal`
/// (same construction) documents guarding against.
fn probe_signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            0.85 * (t * std::f32::consts::TAU / 256.0).sin()
                + 0.15 * (t * std::f32::consts::TAU / 512.0).cos()
        })
        .collect()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// An 8-byte-aligned owned copy of a byte slice, sized at compile
/// time by `N`.
///
/// `Vec<u8>`/`std::fs::read` only guarantee 1-byte alignment, but
/// `BlockQ8K::slice_from_bytes` (like every K-quant block's) hard-
/// rejects a misaligned pointer (`BlockQ8K`'s `d: f32` needs 4-byte
/// alignment). Copying into this fixed, over-aligned buffer before
/// casting is the same pattern
/// `kquant_ggml_golden.rs`'s `Aligned<N>` uses, so the test cannot
/// flake on an allocator that happens not to 4-byte-align a `Vec<u8>`
/// base pointer plus a 32-aligned GGUF tensor-data offset.
#[repr(C, align(8))]
struct Aligned<const N: usize>([u8; N]);

fn aligned_copy<const N: usize>(bytes: &[u8]) -> Aligned<N> {
    assert_eq!(
        bytes.len(),
        N,
        "tensor byte length must match the expected block count"
    );
    let mut buf = [0u8; N];
    buf.copy_from_slice(bytes);
    Aligned(buf)
}

/// Encode `data` through the real `encode_quantized_tensor` writer
/// path, write it as the sole tensor of a real on-disk GGUF file
/// (`GgufWriter::write` into a `std::fs::File`, not `to_bytes()`),
/// then read that file back from disk and return the tensor's raw
/// on-disk bytes — the write -> parse cycle a real loader goes
/// through, not just the in-memory encoder/decoder pair.
fn write_and_reread(tensor_type: TensorType, data: &[f32], probe_tag: &str) -> Vec<u8> {
    let encoded = encode_quantized_tensor(data, data.len(), tensor_type, ScaleRule::default())
        .unwrap_or_else(|e| panic!("encode {tensor_type:?} failed: {e}"));

    let mut writer = GgufWriter::new();
    writer.add_tensor(TensorEntry {
        name: "probe.weight".to_string(),
        shape: vec![data.len() as u64],
        tensor_type,
        data: encoded,
    });

    let dir = std::env::temp_dir().join(format!(
        "oxibonsai_kquant_writer_roundtrip_{probe_tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));
    let path = dir.join("probe.gguf");
    {
        let file = std::fs::File::create(&path)
            .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
        let mut out = std::io::BufWriter::new(file);
        writer
            .write(&mut out)
            .unwrap_or_else(|e| panic!("write real gguf file: {e}"));
        out.flush().unwrap_or_else(|e| panic!("flush: {e}"));
    }

    let file_bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("read back {}: {e}", path.display()));
    let _ = std::fs::remove_dir_all(&dir);

    let parsed =
        GgufFile::parse(&file_bytes).unwrap_or_else(|e| panic!("parse real gguf file: {e}"));
    parsed
        .tensor_data("probe.weight")
        .unwrap_or_else(|e| panic!("tensor_data: {e}"))
        .to_vec()
}

#[test]
fn q2_k_real_gguf_roundtrip_within_error_bound() {
    let x = probe_signal(QK_K * 2);
    let bytes = write_and_reread(TensorType::Q2_K, &x, "q2k");
    let aligned = aligned_copy::<168>(&bytes); // 2 * BLOCK_Q2_K_BYTES(84)
    let blocks = BlockQ2K::slice_from_bytes(&aligned.0).expect("slice Q2_K blocks");
    let mut got = vec![0.0f32; x.len()];
    BlockQ2K::dequant(blocks, &mut got).expect("dequant Q2_K");

    let err = max_abs_err(&x, &got);
    // Matches the 0.02-0.21 self-consistent band `kquant_ggml_golden.rs`
    // established for this exact ggml-exact codec on a smooth signal.
    // Measured ~0.127 on this probe signal; 0.25 leaves ~2x headroom
    // while staying far tighter than the pre-fix core-gguf-K0
    // breakage (1.20-5.64 on the same class of signal).
    assert!(
        err < 0.25,
        "Q2_K real-GGUF round trip max error {err} exceeds tolerance"
    );
}

#[test]
fn q3_k_real_gguf_roundtrip_within_error_bound() {
    let x = probe_signal(QK_K * 2);
    let bytes = write_and_reread(TensorType::Q3_K, &x, "q3k");
    let aligned = aligned_copy::<220>(&bytes); // 2 * BLOCK_Q3K_BYTES(110)
    let blocks = BlockQ3K::slice_from_bytes(&aligned.0).expect("slice Q3_K blocks");
    let mut got = vec![0.0f32; x.len()];
    BlockQ3K::dequant(blocks, &mut got).expect("dequant Q3_K");

    let err = max_abs_err(&x, &got);
    // Measured ~0.107 on this probe signal (pre-fix core-gguf-K0
    // scored 5.64 on the analogous check).
    assert!(
        err < 0.25,
        "Q3_K real-GGUF round trip max error {err} exceeds tolerance"
    );
}

#[test]
fn q4_k_real_gguf_roundtrip_within_error_bound() {
    let x = probe_signal(QK_K * 2);
    let bytes = write_and_reread(TensorType::Q4_K, &x, "q4k");
    let aligned = aligned_copy::<288>(&bytes); // 2 * BLOCK_Q4_K_BYTES(144)
    let blocks = BlockQ4K::slice_from_bytes(&aligned.0).expect("slice Q4_K blocks");
    let mut got = vec![0.0f32; x.len()];
    BlockQ4K::dequant(blocks, &mut got).expect("dequant Q4_K");

    let err = max_abs_err(&x, &got);
    // Measured ~0.030 on this probe signal.
    assert!(
        err < 0.15,
        "Q4_K real-GGUF round trip max error {err} exceeds tolerance"
    );
}

#[test]
fn q5_k_real_gguf_roundtrip_within_error_bound() {
    let x = probe_signal(QK_K * 2);
    let bytes = write_and_reread(TensorType::Q5_K, &x, "q5k");
    let aligned = aligned_copy::<352>(&bytes); // 2 * 176
    let blocks = BlockQ5K::slice_from_bytes(&aligned.0).expect("slice Q5_K blocks");
    let mut got = vec![0.0f32; x.len()];
    BlockQ5K::dequant(blocks, &mut got).expect("dequant Q5_K");

    let err = max_abs_err(&x, &got);
    // Measured ~0.015 on this probe signal.
    assert!(
        err < 0.1,
        "Q5_K real-GGUF round trip max error {err} exceeds tolerance"
    );
}

#[test]
fn q6_k_real_gguf_roundtrip_within_error_bound() {
    let x = probe_signal(QK_K * 2);
    let bytes = write_and_reread(TensorType::Q6_K, &x, "q6k");
    let aligned = aligned_copy::<420>(&bytes); // 2 * 210
    let blocks = BlockQ6K::slice_from_bytes(&aligned.0).expect("slice Q6_K blocks");
    let mut got = vec![0.0f32; x.len()];
    BlockQ6K::dequant(blocks, &mut got).expect("dequant Q6_K");

    let err = max_abs_err(&x, &got);
    // Measured ~0.015 on this probe signal.
    assert!(
        err < 0.05,
        "Q6_K real-GGUF round trip max error {err} exceeds tolerance"
    );
}

#[test]
fn q8_k_real_gguf_roundtrip_reproduces_the_encoder_arithmetic_exactly() {
    let x = probe_signal(QK_K * 2);
    let bytes = write_and_reread(TensorType::Q8_K, &x, "q8k");
    let aligned = aligned_copy::<584>(&bytes); // 2 * BLOCK_Q8K_BYTES(292)
    let blocks = BlockQ8K::slice_from_bytes(&aligned.0).expect("slice Q8_K blocks");
    let mut got = vec![0.0f32; x.len()];
    BlockQ8K::dequant(blocks, &mut got).expect("dequant Q8_K");

    // Q8_K's dequant is a single per-super-block scale `d * qs[i]` with no
    // sub-block interleave (unlike Q2_K..Q6_K), so the round trip is
    // exactly reproducible per element, not merely boundable: re-derive
    // each super-block's own `d` (independently per 256-element chunk,
    // matching `BlockQ8K::quantize`'s per-block scale exactly) and each
    // element's rounded code, and require bit-for-bit agreement.
    //
    // An error-bound assertion alone (`|err| <= d/2`) would be a tautology
    // of the encoder's own arithmetic — true for *any* implementation that
    // shares that arithmetic, correct or not, including one with the block
    // layout wrong — and could not catch a real byte-layout regression
    // (e.g. a wrong `qs` offset shifting which element gets which code).
    for (block_idx, chunk) in x.chunks(QK_K).enumerate() {
        let max_abs = chunk
            .iter()
            .copied()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (i, &w) in chunk.iter().enumerate() {
            // Route through the same `as i8` truncation `BlockQ8K::quantize`
            // does (not just `.round().clamp(..)` kept as `f32`): casting to
            // `i8` collapses a rounded `-0.0` to the signless integer `0`,
            // which a bare `f32` comparison would otherwise flag as a
            // spurious mismatch against the real encoder's always-`+0.0`
            // dequant of an exact-zero code.
            let code = (w * inv_d).round().clamp(-127.0, 127.0) as i8;
            let want = d * (code as f32);
            let got_v = got[block_idx * QK_K + i];
            assert_eq!(
                got_v.to_bits(),
                want.to_bits(),
                "Q8_K block {block_idx} element {i}: round trip must reproduce \
                 d * round(w/d) exactly (got {got_v}, want {want})"
            );
        }
    }
}

/// A degenerate but real edge case: the exact all-zero tensor,
/// through the full write -> parse -> dequant cycle for every
/// format, must come back on-disk-correctly-sized AND (elementwise)
/// zero, not merely "small error" — every format's `d`/`dmin`
/// (or Q8_K's single `d`) collapses to `0.0` for an all-zero input,
/// so `d * q - dmin * m` is exactly `0.0` regardless of the
/// (irrelevant, since scaled by a zero `d`) quantized codes.
#[test]
fn all_k_quant_formats_roundtrip_a_zero_tensor_through_a_real_gguf() {
    let zero = vec![0.0f32; QK_K];
    for (tensor_type, block_bytes) in [
        (TensorType::Q2_K, 84usize),
        (TensorType::Q3_K, 110),
        (TensorType::Q4_K, 144),
        (TensorType::Q5_K, 176),
        (TensorType::Q6_K, 210),
        (TensorType::Q8_K, 292),
    ] {
        let bytes = write_and_reread(tensor_type, &zero, &format!("{tensor_type:?}"));
        assert_eq!(
            bytes.len(),
            block_bytes,
            "{tensor_type:?} on-disk block size"
        );

        let mut got = vec![0.0f32; QK_K];
        match tensor_type {
            TensorType::Q2_K => {
                let aligned = aligned_copy::<84>(&bytes);
                let blocks = BlockQ2K::slice_from_bytes(&aligned.0).expect("slice");
                BlockQ2K::dequant(blocks, &mut got).expect("dequant");
            }
            TensorType::Q3_K => {
                let aligned = aligned_copy::<110>(&bytes);
                let blocks = BlockQ3K::slice_from_bytes(&aligned.0).expect("slice");
                BlockQ3K::dequant(blocks, &mut got).expect("dequant");
            }
            TensorType::Q4_K => {
                let aligned = aligned_copy::<144>(&bytes);
                let blocks = BlockQ4K::slice_from_bytes(&aligned.0).expect("slice");
                BlockQ4K::dequant(blocks, &mut got).expect("dequant");
            }
            TensorType::Q5_K => {
                let aligned = aligned_copy::<176>(&bytes);
                let blocks = BlockQ5K::slice_from_bytes(&aligned.0).expect("slice");
                BlockQ5K::dequant(blocks, &mut got).expect("dequant");
            }
            TensorType::Q6_K => {
                let aligned = aligned_copy::<210>(&bytes);
                let blocks = BlockQ6K::slice_from_bytes(&aligned.0).expect("slice");
                BlockQ6K::dequant(blocks, &mut got).expect("dequant");
            }
            TensorType::Q8_K => {
                let aligned = aligned_copy::<292>(&bytes);
                let blocks = BlockQ8K::slice_from_bytes(&aligned.0).expect("slice");
                BlockQ8K::dequant(blocks, &mut got).expect("dequant");
            }
            other => panic!("unexpected tensor type in this test's own table: {other:?}"),
        }

        assert_eq!(
            got, zero,
            "{tensor_type:?}: an all-zero tensor must dequantize back to exactly zero"
        );
    }
}

/// FIX3-GGUF-WRITE item 1: `known_quant_types()` is generated from
/// `GgufTensorType::ALL.iter().filter(is_executable)`, not a hand
/// maintained list, so it must already report the three new K-quant
/// wire ids without any further edit — this pins that down rather
/// than trusting it silently.
#[test]
fn known_quant_types_lists_the_three_new_k_quants_automatically() {
    let known = crate::gguf_loader::known_quant_types();
    for (id, name) in [(10u32, "Q2_K"), (11, "Q3_K"), (15, "Q8_K")] {
        assert!(
            known.contains(&(id, name)),
            "known_quant_types() must list {name} (id {id}); got {known:?}"
        );
    }
}
