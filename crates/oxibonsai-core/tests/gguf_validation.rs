//! Cross-parser parity and end-to-end validation tests for the GGUF reader.
//!
//! Covers findings that specifically require driving the **batch**
//! (`GgufFile::parse` / `TensorStore::parse`) and **streaming**
//! (`GgufStreamParser`) parsers over the *same* bytes and comparing their
//! outcomes (core-gguf-16, core-gguf-15), plus end-to-end coverage of
//! [`GgufFile::diagnose`] and the tensor-layout validation pass
//! (core-gguf-03 / sec-13) that unit tests inside `reader.rs` itself cannot
//! exercise as a black box the way an external integration test can.

use oxibonsai_core::error::BonsaiError;
use oxibonsai_core::gguf::reader::{GgufDiagnosis, GgufFile};
use oxibonsai_core::gguf::streaming::GgufStreamParser;
use oxibonsai_core::gguf::types::GgufValueType;

const GGUF_MAGIC: u32 = 0x4655_4747;

fn gguf_string(s: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
    b
}

fn header(tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&tensor_count.to_le_bytes());
    b.extend_from_slice(&metadata_kv_count.to_le_bytes());
    b
}

fn kv_u32(key: &str, value: u32) -> Vec<u8> {
    let mut b = gguf_string(key);
    b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
    b.extend_from_slice(&value.to_le_bytes());
    b
}

fn kv_u64(key: &str, value: u64) -> Vec<u8> {
    let mut b = gguf_string(key);
    b.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
    b.extend_from_slice(&value.to_le_bytes());
    b
}

fn kv_i32(key: &str, value: i32) -> Vec<u8> {
    let mut b = gguf_string(key);
    b.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
    b.extend_from_slice(&value.to_le_bytes());
    b
}

fn tensor_info_bytes(name: &str, shape: &[u64], type_id: u32, offset: u64) -> Vec<u8> {
    let mut b = gguf_string(name);
    b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &d in shape {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&type_id.to_le_bytes());
    b.extend_from_slice(&offset.to_le_bytes());
    b
}

/// Assemble a full GGUF byte buffer: header + metadata entries + tensor-info
/// entries + alignment padding + `tensor_data_len` zero bytes.
fn assemble_gguf(
    metadata: &[Vec<u8>],
    tensor_infos: &[Vec<u8>],
    tensor_data_len: usize,
) -> Vec<u8> {
    let mut data = header(tensor_infos.len() as u64, metadata.len() as u64);
    for kv in metadata {
        data.extend_from_slice(kv);
    }
    for info in tensor_infos {
        data.extend_from_slice(info);
    }
    while !data.len().is_multiple_of(32) {
        data.push(0);
    }
    data.extend(vec![0u8; tensor_data_len]);
    data
}

// ─────────────────────────────────────────────────────────────────────────
// core-gguf-16 (wave-1 addendum #4): 5-D tensor parity between the batch
// and streaming parsers over the identical bytes.
// ─────────────────────────────────────────────────────────────────────────

/// A tensor declaring 5 dimensions is invalid input (`GGML_MAX_DIMS` is 4);
/// both parsers must reject it with the same error *and the exact same
/// message text* over the exact same bytes. This drives `GgufFile::parse`
/// (which owns `TensorStore` via `tensor_info.rs`) and `GgufStreamParser`
/// (`streaming.rs`) side by side — neither of those two files is owned by
/// this package, but the parity test itself belongs here per the wave-1
/// addendum.
///
/// The two parsers' reason strings were unified to the identical wording
/// ("tensor has {n} dimensions; GGML_MAX_DIMS is {max}") by the wave-2.5
/// addendum (item 4); this test now asserts that stronger message-text
/// parity directly, rather than only the weaker error *kind* it used to
/// check.
#[test]
fn five_dimensional_tensor_is_rejected_with_the_same_error_kind_by_both_parsers() {
    let tensor = tensor_info_bytes("five_d.weight", &[2, 2, 2, 2, 2], 0, 0);
    let data = assemble_gguf(&[], &[tensor], 0);

    let batch_result = GgufFile::parse(&data);
    let (batch_key, batch_reason) = match batch_result {
        Err(BonsaiError::InvalidMetadata { key, reason }) => (key, reason),
        other => panic!("expected InvalidMetadata from the batch parser, got: {other:?}"),
    };

    let mut parser = GgufStreamParser::new();
    let stream_result = parser.feed(&data);
    let (stream_key, stream_reason) = match stream_result {
        Err(BonsaiError::InvalidMetadata { key, reason }) => (key, reason),
        other => panic!("expected InvalidMetadata from the streaming parser, got: {other:?}"),
    };

    assert_eq!(
        batch_key, stream_key,
        "both parsers must name the same tensor in the error"
    );
    assert_eq!(
        batch_reason, stream_reason,
        "both parsers must reject a 5-D tensor with the exact same message text, not just the \
         same error kind"
    );
    assert_eq!(
        batch_reason, "tensor has 5 dimensions; GGML_MAX_DIMS is 4",
        "wording must match the unified check in both tensor_info.rs and streaming.rs exactly"
    );
}

/// The boundary case (exactly 4 dimensions) must be accepted by both
/// parsers over the same bytes.
#[test]
fn four_dimensional_tensor_is_accepted_by_both_parsers() {
    let tensor = tensor_info_bytes("four_d.weight", &[2, 2, 2, 2], 0, 0);
    let data = assemble_gguf(&[], &[tensor], 64);

    GgufFile::parse(&data).expect("batch parser must accept exactly 4 dimensions");

    let mut parser = GgufStreamParser::new();
    parser
        .feed(&data)
        .expect("streaming parser must accept exactly 4 dimensions");
    assert!(parser.is_complete());
}

// ─────────────────────────────────────────────────────────────────────────
// core-gguf-15: `general.alignment` must be spec-typed UINT32.
// ─────────────────────────────────────────────────────────────────────────

/// For the valid (`Uint32`) spelling, the batch parser's `data_offset` and
/// the streaming parser's `data_offset` must agree exactly over the same
/// bytes — the invariant `finalize()`'s own doc comment in `streaming.rs`
/// exists to preserve.
#[test]
fn data_offset_agrees_between_batch_and_streaming_parsers_for_the_uint32_spelling() {
    let metadata = vec![kv_u32("general.alignment", 64)];
    let data = assemble_gguf(&metadata, &[], 0);

    let file = GgufFile::parse(&data).expect("Uint32 alignment must parse");

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("streaming parse must succeed");
    let streamed = parser.finish().expect("streaming finish must succeed");

    assert_eq!(
        file.data_offset as u64, streamed.data_offset,
        "batch and streaming data_offset must agree for the spec-valid Uint32 spelling"
    );
    assert_eq!(file.data_offset % 64, 0);
}

/// A `general.alignment` stored as `Uint64` is spec-invalid (llama.cpp
/// rejects any non-`UINT32` spelling outright — `ggml/src/gguf.cpp:613-618`).
/// The batch parser (this package's `reader.rs`) now rejects it explicitly
/// instead of silently widening through `MetadataValue::as_u32()`.
#[test]
fn batch_parser_rejects_a_non_uint32_alignment_spelling() {
    for metadata in [
        vec![kv_u64("general.alignment", 64)],
        vec![kv_i32("general.alignment", 64)],
    ] {
        let data = assemble_gguf(&metadata, &[], 0);
        match GgufFile::parse(&data) {
            Err(BonsaiError::InvalidMetadata { key, .. }) => {
                assert_eq!(key, "general.alignment");
            }
            other => panic!(
                "expected InvalidMetadata for a non-Uint32 general.alignment, got: {other:?}"
            ),
        }
    }
}

/// The wave-1 asymmetry this test used to pin (`GgufStreamParser::finalize()`
/// silently defaulting a non-`Uint32` `general.alignment` instead of
/// erroring, unlike the batch parser) is fixed (wave-2.5 addendum item 3):
/// `finalize()` now matches `GgufFile::parse`'s shape exactly — an absent
/// key defaults, `Some(Uint32(v))` uses `v`, and any other type is a hard
/// `InvalidMetadata` error. This test now asserts that parity directly
/// instead of documenting the old divergence.
#[test]
fn streaming_parser_rejects_a_non_uint32_alignment_spelling_like_the_batch_parser() {
    let metadata = vec![kv_u64("general.alignment", 64)];
    let data = assemble_gguf(&metadata, &[], 0);

    // The batch parser rejects this file outright (previous test).
    assert!(GgufFile::parse(&data).is_err());

    // The streaming parser must now reject it identically, not silently
    // fall back to the 32-byte default.
    let mut parser = GgufStreamParser::new();
    match parser.feed(&data) {
        Err(BonsaiError::InvalidMetadata { key, .. }) => assert_eq!(key, "general.alignment"),
        other => {
            panic!("expected InvalidMetadata for a non-Uint32 general.alignment, got: {other:?}")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// core-gguf-03 / sec-13: end-to-end tensor-layout validation.
// ─────────────────────────────────────────────────────────────────────────

/// A GGUF file whose second tensor offset is subtly wrong (neither zero
/// nor a full extra padding unit off) previously loaded silently —
/// `cmd_validate.rs` would print "Validation: OK" for it regardless. This
/// pins the end-to-end outcome down: whatever specific check catches it
/// (alignment or the offset chain — a non-zero sub-alignment-unit error
/// generically fails both, since a correctly *padded* extent is always
/// alignment-respecting, so a small perturbation away from it usually
/// isn't), `GgufFile::parse` must reject the file, not silently accept it.
#[test]
fn a_file_with_a_subtly_wrong_second_tensor_offset_is_rejected() {
    let tensor_a = tensor_info_bytes("a", &[8], 0, 0); // F32, 32 bytes exact.
    let tensor_b = tensor_info_bytes("b", &[8], 0, 63); // should be 32.
    let data = assemble_gguf(&[], &[tensor_a, tensor_b], 128);
    match GgufFile::parse(&data) {
        Err(BonsaiError::TensorLayout { .. }) => {}
        other => {
            panic!("a file with a subtly wrong tensor offset must be rejected, got: {other:?}")
        }
    }
}

/// The exact discriminator core-gguf-01 needs: a **full extra padding
/// unit** of slack between two *alignment-respecting* offsets. The old,
/// weaker `offset[i] + size(i) <= offset[i+1]` inequality would silently
/// accept this file (32 + 32 <= 64 holds); only exact equality against the
/// declared next offset (32 == 64 does not) catches it — and both offsets
/// here are individually 32-aligned, so this isolates the offset-chain
/// check from the independent alignment check.
#[test]
fn a_full_padding_unit_gap_between_alignment_respecting_offsets_is_rejected() {
    let tensor_a = tensor_info_bytes("a", &[8], 0, 0); // F32, 32 bytes, ends at 32.
    let tensor_b = tensor_info_bytes("b", &[8], 0, 64); // should start at 32.
    let data = assemble_gguf(&[], &[tensor_a, tensor_b], 96);
    match GgufFile::parse(&data) {
        Err(BonsaiError::TensorLayout { name, reason }) => {
            assert_eq!(name, "a");
            assert!(reason.contains("gap or overlap"), "reason: {reason}");
        }
        other => panic!("expected a gap/overlap TensorLayout error, got: {other:?}"),
    }
}

/// [`GgufFile::diagnose`] degrades to the tolerant [`GgufDiagnosis::Degraded`]
/// report when strict parsing fails on a genuinely unrecognised
/// quantization type, instead of only handing back a bare parse error
/// (core-gguf-05's "more valuable half" — the diagnostic report already
/// existed via `probe_compat`, but nothing wired it in).
#[test]
fn diagnose_degrades_gracefully_on_an_unrecognised_quant_type() {
    let tensor = tensor_info_bytes("t", &[1], 99_999, 0);
    let data = assemble_gguf(&[], &[tensor], 0);

    assert!(
        GgufFile::parse(&data).is_err(),
        "strict parse must still reject this file"
    );

    match GgufFile::diagnose(&data).expect("diagnose must recover a real report") {
        GgufDiagnosis::Degraded(report) => {
            assert!(!report.is_loadable);
            assert!(report.unknown_quant_types.contains(&99_999));
        }
        GgufDiagnosis::Parsed(_) => panic!("expected a degraded diagnosis, not a full parse"),
    }
}

/// [`GgufFile::diagnose`] returns [`GgufDiagnosis::Parsed`] for a file that
/// strict parsing already handles, with no loss of information.
#[test]
fn diagnose_returns_the_full_parse_when_strict_parsing_succeeds() {
    let tensor = tensor_info_bytes("a", &[8], 0, 0);
    let data = assemble_gguf(&[], &[tensor], 32);

    match GgufFile::diagnose(&data).expect("diagnose must succeed") {
        GgufDiagnosis::Parsed(file) => {
            assert_eq!(file.tensors.len(), 1);
            assert!(file.compat.is_loadable);
        }
        GgufDiagnosis::Degraded(_) => panic!("expected a full parse, not a degraded diagnosis"),
    }
}

/// A file that is not recoverably GGUF at all (bad magic) must still fail
/// `diagnose`, not silently succeed via the tolerant fallback.
#[test]
fn diagnose_fails_on_a_file_that_is_not_gguf_at_all() {
    let mut data = Vec::new();
    data.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    assert!(GgufFile::diagnose(&data).is_err());
}
