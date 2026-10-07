//! Integration tests for the GGUF file writer.
//!
//! Each test produces a complete GGUF byte payload with [`GgufWriter`] and
//! then parses it back with the existing [`GgufFile`] reader to verify
//! round-trip correctness.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::writer::{
    GgufWriter, MetadataWriteValue, TensorEntry, TensorType, WriteError,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Write and immediately re-parse a GGUF buffer.  Panics on any error.
fn roundtrip(writer: &GgufWriter) -> Vec<u8> {
    writer.to_bytes().expect("GgufWriter::to_bytes failed")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// The first four bytes must be the GGUF magic, and version must be 3.
#[test]
fn test_gguf_writer_magic_and_version() {
    let writer = GgufWriter::new();
    let bytes = roundtrip(&writer);

    assert!(bytes.len() >= 8, "output must be at least 8 bytes");

    // The reader defines GGUF_MAGIC = 0x4655_4747, stored as LE u32.
    let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("slice length mismatch"));
    assert_eq!(magic, 0x4655_4747u32, "magic must match GGUF_MAGIC");

    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("slice length mismatch"));
    assert_eq!(version, 3, "version must be 3");
}

/// Metadata KV pairs can be round-tripped through the reader.
#[test]
fn test_gguf_writer_metadata() {
    let mut writer = GgufWriter::new();
    writer
        .add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen3".to_string()),
        )
        .add_metadata("llm.block_count", MetadataWriteValue::U32(28))
        .add_metadata("llm.rope.freq_base", MetadataWriteValue::F32(500_000.0));

    let bytes = roundtrip(&writer);
    let file = GgufFile::parse(&bytes).expect("GgufFile::parse failed");

    assert_eq!(file.header.metadata_kv_count, 3);

    let arch = file
        .metadata
        .get_string("general.architecture")
        .expect("general.architecture missing");
    assert_eq!(arch, "qwen3");

    let block_count = file
        .metadata
        .get_u32("llm.block_count")
        .expect("llm.block_count missing");
    assert_eq!(block_count, 28);

    let rope_freq = file
        .metadata
        .get_f32("llm.rope.freq_base")
        .expect("llm.rope.freq_base missing");
    assert!(
        (rope_freq - 500_000.0).abs() < 1.0,
        "rope_freq mismatch: {rope_freq}"
    );
}

/// Write a small F32 tensor, parse it back, and verify the data bytes.
#[test]
fn test_gguf_writer_tensor_roundtrip() {
    // 4 f32 weights = 16 raw bytes.
    let weights: Vec<f32> = vec![1.0, -2.0, 3.5, -0.5];
    let raw_bytes: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();

    let mut writer = GgufWriter::new();
    writer.add_tensor(TensorEntry {
        name: "token_embd.weight".to_string(),
        shape: vec![4],
        tensor_type: TensorType::F32,
        data: raw_bytes.clone(),
    });

    let bytes = roundtrip(&writer);
    let file = GgufFile::parse(&bytes).expect("GgufFile::parse failed");

    assert_eq!(file.header.tensor_count, 1);

    let tensor_data = file
        .tensor_data("token_embd.weight")
        .expect("tensor_data lookup failed");

    assert_eq!(
        tensor_data,
        raw_bytes.as_slice(),
        "tensor data must round-trip exactly"
    );

    // Re-interpret the bytes as f32 and verify values.
    let recovered: Vec<f32> = tensor_data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(recovered, weights);
}

/// The tensor data section must start at an offset that is a multiple of
/// the configured alignment.
#[test]
fn test_gguf_writer_alignment_padding() {
    for alignment in [1_usize, 8, 16, 32, 64, 128] {
        let mut writer = GgufWriter::new();
        writer.set_alignment(alignment);

        // Add a short metadata entry to make the pre-padding offset non-trivial.
        writer.add_metadata("x", MetadataWriteValue::U32(1));

        // Add a single 1-byte tensor so there is a data section.
        writer.add_tensor(TensorEntry {
            name: "t".to_string(),
            shape: vec![1],
            tensor_type: TensorType::F32,
            data: vec![0u8; 4], // 1 × f32
        });

        let bytes = roundtrip(&writer);
        let file = GgufFile::parse(&bytes).expect("parse failed");

        assert_eq!(
            file.data_offset % alignment,
            0,
            "data_offset {} must be aligned to {alignment}",
            file.data_offset
        );
    }
}

/// A writer with no tensors and no metadata should still produce a valid,
/// parseable GGUF file.
#[test]
fn test_gguf_writer_empty_file() {
    let writer = GgufWriter::new();
    let bytes = roundtrip(&writer);

    let file = GgufFile::parse(&bytes).expect("empty GGUF should be parseable");
    assert_eq!(file.header.tensor_count, 0);
    assert_eq!(file.header.metadata_kv_count, 0);
    assert!(file.tensors.is_empty());
    assert!(file.metadata.is_empty());
}

/// Writing a tensor whose `data` length doesn't match the expected size for
/// its type and shape must return a `DataSizeMismatch` error.
#[test]
fn test_gguf_writer_data_size_mismatch_error() {
    let mut writer = GgufWriter::new();
    writer.add_tensor(TensorEntry {
        name: "bad".to_string(),
        shape: vec![4],
        tensor_type: TensorType::F32,
        data: vec![0u8; 3], // wrong: 4 × f32 = 16 bytes expected
    });

    let result = writer.to_bytes();
    assert!(
        matches!(result, Err(WriteError::DataSizeMismatch { .. })),
        "expected DataSizeMismatch, got: {result:?}"
    );
}
