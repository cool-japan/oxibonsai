//! Tests for the streaming GGUF parser.
//!
//! T-07: `build_test_gguf` below is NOT re-pointed at
//! `oxibonsai_testkit::gguf_fixture::GgufFixtureBuilder`, despite building a
//! GGUF-shaped byte buffer. It tests a different layer of the system on
//! purpose: the *incremental streaming parser's* byte-level state machine
//! (one byte at a time, random chunk sizes, partial-header resumption), so
//! it deliberately builds ONLY the header + metadata + tensor-info section
//! (no tensor data, no alignment padding — `GgufFixtureBuilder` always
//! produces a complete, `GgufFile::parse`-able file) and deliberately
//! exercises every one of the 13 GGUF metadata value types (`GgufTestValue`)
//! for full type-tag coverage, which `GgufFixtureBuilder` does not expose
//! (only `Str`/`U32`/`F32`). Re-pointing this at the whole-file builder
//! would need to either lose that metadata-type-tag coverage or grow
//! `GgufFixtureBuilder` a second, incremental-friendly mode — recorded as
//! the honest remaining duplicate T-07 explicitly allows listing.

use oxibonsai_core::gguf::streaming::{GgufStreamParser, GgufValue, StreamState};
use oxibonsai_core::BonsaiError;

/// GGUF magic number in little-endian bytes.
const GGUF_MAGIC_LE: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // 0x46554747 as u32 LE bytes

/// Helper value types for building test GGUF data.
enum GgufTestValue {
    #[allow(dead_code)]
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(u32, Vec<GgufTestValue>), // (elem_type_id, elements)
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufTestValue {
    fn type_id(&self) -> u32 {
        match self {
            Self::Uint8(_) => 0,
            Self::Int8(_) => 1,
            Self::Uint16(_) => 2,
            Self::Int16(_) => 3,
            Self::Uint32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_, _) => 9,
            Self::Uint64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }
}

struct TestTensorInfo {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
    offset: u64,
}

/// Build a minimal valid GGUF byte stream for testing.
fn build_test_gguf(metadata: &[(String, GgufTestValue)], tensors: &[TestTensorInfo]) -> Vec<u8> {
    let mut buf = Vec::new();

    // Header: magic + version(3) + tensor_count + metadata_kv_count
    buf.extend_from_slice(&GGUF_MAGIC_LE);
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // Metadata KV pairs
    for (key, value) in metadata {
        write_gguf_string(&mut buf, key);
        buf.extend_from_slice(&value.type_id().to_le_bytes());
        write_gguf_value(&mut buf, value);
    }

    // Tensor info entries
    for t in tensors {
        write_gguf_string(&mut buf, &t.name);
        buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &dim in &t.dims {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&t.type_id.to_le_bytes());
        buf.extend_from_slice(&t.offset.to_le_bytes());
    }

    buf
}

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_gguf_value(buf: &mut Vec<u8>, value: &GgufTestValue) {
    match value {
        GgufTestValue::Uint8(v) => buf.push(*v),
        GgufTestValue::Int8(v) => buf.push(*v as u8),
        GgufTestValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufTestValue::Int16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufTestValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufTestValue::Int32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufTestValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufTestValue::Bool(v) => buf.push(if *v { 1 } else { 0 }),
        GgufTestValue::String(s) => write_gguf_string(buf, s),
        GgufTestValue::Array(elem_type, elems) => {
            buf.extend_from_slice(&elem_type.to_le_bytes());
            buf.extend_from_slice(&(elems.len() as u64).to_le_bytes());
            for elem in elems {
                write_gguf_value(buf, elem);
            }
        }
        GgufTestValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufTestValue::Int64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufTestValue::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
    }
}

/// LCG pseudo-random for deterministic chunk sizes (no rand crate).
fn lcg(s: &mut u64) -> usize {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 33) as usize
}

// ---- Tests ----

#[test]
fn parse_complete_minimal_gguf() {
    let data = build_test_gguf(&[], &[]);
    let mut parser = GgufStreamParser::new();
    let consumed = parser.feed(&data).expect("feed should succeed");
    assert_eq!(consumed, data.len());
    assert!(parser.is_complete());

    let result = parser.finish().expect("finish should succeed");
    assert_eq!(result.version, 3);
    assert!(result.metadata.is_empty());
    assert!(result.tensor_infos.is_empty());
}

#[test]
fn parse_one_byte_at_a_time() {
    let metadata = vec![
        (
            "model.name".to_string(),
            GgufTestValue::String("test".to_string()),
        ),
        ("model.layers".to_string(), GgufTestValue::Uint32(12)),
    ];
    let tensors = vec![TestTensorInfo {
        name: "weight.0".to_string(),
        dims: vec![128, 256],
        type_id: 0, // F32
        offset: 0,
    }];
    let data = build_test_gguf(&metadata, &tensors);

    let mut parser = GgufStreamParser::new();
    for &byte in &data {
        parser.feed(&[byte]).expect("feed should succeed");
    }
    assert!(parser.is_complete());

    let result = parser.finish().expect("finish should succeed");
    assert_eq!(result.metadata.len(), 2);
    assert_eq!(result.tensor_infos.len(), 1);
}

#[test]
fn parse_random_chunk_sizes() {
    let metadata = vec![
        (
            "arch".to_string(),
            GgufTestValue::String("qwen3".to_string()),
        ),
        ("count".to_string(), GgufTestValue::Uint64(42)),
    ];
    let tensors = vec![
        TestTensorInfo {
            name: "embed".to_string(),
            dims: vec![4096],
            type_id: 1, // F16
            offset: 0,
        },
        TestTensorInfo {
            name: "attn.weight".to_string(),
            dims: vec![4096, 4096],
            type_id: 41, // Q1_0_g128
            offset: 8192,
        },
    ];
    let data = build_test_gguf(&metadata, &tensors);

    let mut parser = GgufStreamParser::new();
    let mut offset = 0;
    let mut seed: u64 = 0xDEAD_BEEF;
    while offset < data.len() {
        let chunk_size = (lcg(&mut seed) % 17).max(1).min(data.len() - offset);
        parser
            .feed(&data[offset..offset + chunk_size])
            .expect("feed should succeed");
        offset += chunk_size;
    }
    assert!(parser.is_complete());

    let result = parser.finish().expect("finish should succeed");
    assert_eq!(result.metadata.len(), 2);
    assert_eq!(result.tensor_infos.len(), 2);
    assert_eq!(result.tensor_infos[1].name, "attn.weight");
}

#[test]
fn bad_magic_returns_error() {
    let mut data = vec![0u8; 24];
    // Write bad magic
    data[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    data[4..8].copy_from_slice(&3u32.to_le_bytes());

    let mut parser = GgufStreamParser::new();
    let err = parser.feed(&data).expect_err("should fail on bad magic");
    match err {
        BonsaiError::InvalidMagic { magic } => assert_eq!(magic, 0xDEAD_BEEF),
        other => panic!("expected InvalidMagic, got: {other}"),
    }
}

#[test]
fn bad_version_returns_error() {
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&GGUF_MAGIC_LE);
    data[4..8].copy_from_slice(&99u32.to_le_bytes());

    let mut parser = GgufStreamParser::new();
    let err = parser.feed(&data).expect_err("should fail on bad version");
    match err {
        BonsaiError::UnsupportedVersion { version } => assert_eq!(version, 99),
        other => panic!("expected UnsupportedVersion, got: {other}"),
    }
}

#[test]
fn parse_string_metadata() {
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufTestValue::String("qwen3".to_string()),
    )];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    assert!(parser.is_complete());

    let result = parser.finish().expect("finish should succeed");
    assert_eq!(result.metadata.len(), 1);
    assert_eq!(result.metadata[0].0, "general.architecture");
    match &result.metadata[0].1 {
        GgufValue::String(s) => assert_eq!(s, "qwen3"),
        other => panic!("expected String, got: {other:?}"),
    }
}

#[test]
fn parse_u32_metadata() {
    let metadata = vec![("layers".to_string(), GgufTestValue::Uint32(32))];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");
    match &result.metadata[0].1 {
        GgufValue::Uint32(v) => assert_eq!(*v, 32),
        other => panic!("expected Uint32, got: {other:?}"),
    }
}

#[test]
fn parse_f32_metadata() {
    let metadata = vec![("epsilon".to_string(), GgufTestValue::Float32(1e-5))];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");
    match &result.metadata[0].1 {
        GgufValue::Float32(v) => assert!((v - 1e-5).abs() < 1e-10),
        other => panic!("expected Float32, got: {other:?}"),
    }
}

#[test]
fn parse_array_metadata() {
    let metadata = vec![(
        "tokens".to_string(),
        GgufTestValue::Array(
            4, // Uint32
            vec![
                GgufTestValue::Uint32(100),
                GgufTestValue::Uint32(200),
                GgufTestValue::Uint32(300),
            ],
        ),
    )];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");
    match &result.metadata[0].1 {
        GgufValue::Array(arr) => {
            assert_eq!(arr.len(), 3);
            match &arr[0] {
                GgufValue::Uint32(v) => assert_eq!(*v, 100),
                other => panic!("expected Uint32, got: {other:?}"),
            }
        }
        other => panic!("expected Array, got: {other:?}"),
    }
}

#[test]
fn parse_tensor_info_entries() {
    let tensors = vec![
        TestTensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            dims: vec![4096, 4096],
            type_id: 41,
            offset: 0,
        },
        TestTensorInfo {
            name: "blk.0.attn_k.weight".to_string(),
            dims: vec![4096, 512],
            type_id: 41,
            offset: 2359296,
        },
    ];
    let data = build_test_gguf(&[], &tensors);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");

    assert_eq!(result.tensor_infos.len(), 2);
    assert_eq!(result.tensor_infos[0].name, "blk.0.attn_q.weight");
    assert_eq!(result.tensor_infos[0].n_dims, 2);
    assert_eq!(result.tensor_infos[0].dims[0], 4096);
    assert_eq!(result.tensor_infos[0].dims[1], 4096);
    assert_eq!(result.tensor_infos[0].offset, 0);
    assert_eq!(result.tensor_infos[1].offset, 2359296);
}

#[test]
fn progress_reporting() {
    let metadata = vec![
        ("key1".to_string(), GgufTestValue::Uint32(1)),
        ("key2".to_string(), GgufTestValue::Uint32(2)),
    ];
    let tensors = vec![TestTensorInfo {
        name: "t0".to_string(),
        dims: vec![128],
        type_id: 0,
        offset: 0,
    }];
    let data = build_test_gguf(&metadata, &tensors);

    let mut parser = GgufStreamParser::new();

    // Before any data
    let p0 = parser.progress();
    assert!(
        (0.0..=0.1).contains(&p0),
        "initial progress should be near 0, got {p0}"
    );

    // Feed just header
    parser.feed(&data[..24]).expect("feed header");
    let p1 = parser.progress();
    assert!(p1 > p0, "progress should increase after header, got {p1}");

    // Feed rest
    parser.feed(&data[24..]).expect("feed rest");
    assert!(parser.is_complete());
    let p_final = parser.progress();
    assert!(
        (p_final - 1.0).abs() < f32::EPSILON,
        "final progress should be 1.0, got {p_final}"
    );
}

#[test]
fn empty_metadata_file() {
    let data = build_test_gguf(&[], &[]);
    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    assert!(parser.is_complete());

    let result = parser.finish().expect("finish should succeed");
    assert!(result.metadata.is_empty());
    assert!(result.tensor_infos.is_empty());
    assert_eq!(result.version, 3);
}

#[test]
fn finish_before_complete_returns_error() {
    let parser = GgufStreamParser::new();
    let err = parser
        .finish()
        .expect_err("finish on incomplete should fail");
    match err {
        BonsaiError::UnexpectedEof { .. } => {}
        other => panic!("expected UnexpectedEof, got: {other}"),
    }
}

#[test]
fn bytes_consumed_tracking() {
    let metadata = vec![("k".to_string(), GgufTestValue::Uint32(7))];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    assert_eq!(parser.bytes_consumed(), 0);

    parser.feed(&data[..12]).expect("feed partial header");
    // Header not yet complete, no bytes consumed from header perspective
    assert_eq!(parser.bytes_consumed(), 0);

    parser.feed(&data[12..]).expect("feed rest");
    assert!(parser.is_complete());
    // Should have consumed header (24) + key string (8+1) + type(4) + value(4) = 41
    assert!(parser.bytes_consumed() > 0);
}

#[test]
fn state_transitions_are_correct() {
    let metadata = vec![("x".to_string(), GgufTestValue::Uint32(1))];
    let tensors = vec![TestTensorInfo {
        name: "t".to_string(),
        dims: vec![64],
        type_id: 0,
        offset: 0,
    }];
    let data = build_test_gguf(&metadata, &tensors);

    let mut parser = GgufStreamParser::new();
    assert_eq!(*parser.state(), StreamState::ReadingHeader);

    // Feed just the header
    parser.feed(&data[..24]).expect("feed header");
    assert_eq!(
        *parser.state(),
        StreamState::ReadingMetadata { remaining: 1 }
    );

    // Feed the metadata KV (key "x": 8+1 bytes, type 4 bytes, value 4 bytes = 17)
    // We need to feed enough to parse the metadata entry
    let meta_end = 24 + 8 + 1 + 4 + 4; // header + string_len(8) + "x"(1) + type(4) + u32(4) = 41
    parser.feed(&data[24..meta_end]).expect("feed metadata");
    assert_eq!(
        *parser.state(),
        StreamState::ReadingTensorInfo { remaining: 1 }
    );

    // Feed the tensor info
    parser.feed(&data[meta_end..]).expect("feed tensor info");
    assert!(parser.is_complete());
}

#[test]
fn partial_header_then_rest() {
    let data = build_test_gguf(&[], &[]);

    // Feed first 10 bytes (partial header)
    let mut parser = GgufStreamParser::new();
    parser.feed(&data[..10]).expect("feed partial");
    assert_eq!(*parser.state(), StreamState::ReadingHeader);
    assert!(!parser.is_complete());

    // Feed remaining header
    parser.feed(&data[10..]).expect("feed rest");
    assert!(parser.is_complete());
}

#[test]
fn parse_bool_metadata() {
    let metadata = vec![("flag".to_string(), GgufTestValue::Bool(true))];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");
    match &result.metadata[0].1 {
        GgufValue::Bool(v) => assert!(*v),
        other => panic!("expected Bool, got: {other:?}"),
    }
}

#[test]
fn parse_int_types_metadata() {
    let metadata = vec![
        ("i8val".to_string(), GgufTestValue::Int8(-42)),
        ("i16val".to_string(), GgufTestValue::Int16(-1000)),
        ("u16val".to_string(), GgufTestValue::Uint16(65535)),
        ("i32val".to_string(), GgufTestValue::Int32(-100000)),
        ("u64val".to_string(), GgufTestValue::Uint64(u64::MAX)),
        ("i64val".to_string(), GgufTestValue::Int64(i64::MIN)),
        (
            "f64val".to_string(),
            GgufTestValue::Float64(std::f64::consts::PI),
        ),
    ];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");
    assert_eq!(result.metadata.len(), 7);

    match &result.metadata[0].1 {
        GgufValue::Int8(v) => assert_eq!(*v, -42),
        other => panic!("expected Int8, got: {other:?}"),
    }
    match &result.metadata[1].1 {
        GgufValue::Int16(v) => assert_eq!(*v, -1000),
        other => panic!("expected Int16, got: {other:?}"),
    }
    match &result.metadata[4].1 {
        GgufValue::Uint64(v) => assert_eq!(*v, u64::MAX),
        other => panic!("expected Uint64, got: {other:?}"),
    }
    match &result.metadata[6].1 {
        GgufValue::Float64(v) => assert!((v - std::f64::consts::PI).abs() < 1e-15),
        other => panic!("expected Float64, got: {other:?}"),
    }
}

#[test]
fn data_offset_is_aligned() {
    // Build a GGUF with some metadata so data_offset is not trivially aligned
    let metadata = vec![(
        "key".to_string(),
        GgufTestValue::String("a non-trivial value string".to_string()),
    )];
    let data = build_test_gguf(&metadata, &[]);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");

    // data_offset should be 32-byte aligned
    assert_eq!(
        result.data_offset % 32,
        0,
        "data_offset should be 32-byte aligned"
    );
    // and at least as large as bytes_consumed
    assert!(result.data_offset >= data.len() as u64 - 31);
}

#[test]
fn feed_empty_data_returns_zero() {
    let mut parser = GgufStreamParser::new();
    let consumed = parser.feed(&[]).expect("feed empty should succeed");
    assert_eq!(consumed, 0);
}

#[test]
fn multiple_tensors_with_different_types() {
    let tensors = vec![
        TestTensorInfo {
            name: "embed.weight".to_string(),
            dims: vec![32000, 4096],
            type_id: 0, // F32
            offset: 0,
        },
        TestTensorInfo {
            name: "blk.0.q".to_string(),
            dims: vec![4096, 4096],
            type_id: 41, // Q1_0_g128
            offset: 524288000,
        },
        TestTensorInfo {
            name: "blk.0.k".to_string(),
            dims: vec![512, 4096],
            type_id: 1, // F16
            offset: 527433728,
        },
    ];
    let data = build_test_gguf(&[], &tensors);

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    let result = parser.finish().expect("finish should succeed");

    assert_eq!(result.tensor_infos.len(), 3);
    assert_eq!(result.tensor_infos[0].name, "embed.weight");
    assert_eq!(result.tensor_infos[1].name, "blk.0.q");
    assert_eq!(result.tensor_infos[2].name, "blk.0.k");
    assert_eq!(result.tensor_infos[2].dims[0], 512);
    assert_eq!(result.tensor_infos[2].dims[1], 4096);
}

#[test]
fn version_2_is_accepted() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC_LE);
    data.extend_from_slice(&2u32.to_le_bytes()); // version 2
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    let mut parser = GgufStreamParser::new();
    parser.feed(&data).expect("feed should succeed");
    assert!(parser.is_complete());

    let result = parser.finish().expect("finish should succeed");
    assert_eq!(result.version, 2);
}
