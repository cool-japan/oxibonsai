//! GGUF parsing edge case tests.
//!
//! Tests malformed headers, invalid metadata, tensor info edge cases,
//! and alignment calculations.

use oxibonsai_core::error::BonsaiError;
use oxibonsai_core::gguf::header::GgufHeader;
use oxibonsai_core::gguf::metadata::{MetadataStore, MetadataValue};
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::TensorStore;
use oxibonsai_core::gguf::types::{GgufTensorType, GgufValueType};

// ──────────────────────────────────────────────────────────────
// Helper: build raw GGUF bytes
// ──────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x4655_4747;

fn gguf_header_bytes(magic: u32, version: u32, tensors: u64, metadata: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&magic.to_le_bytes());
    data.extend_from_slice(&version.to_le_bytes());
    data.extend_from_slice(&tensors.to_le_bytes());
    data.extend_from_slice(&metadata.to_le_bytes());
    data
}

fn make_gguf_string(s: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
    b
}

fn make_kv_u32(key: &str, value: u32) -> Vec<u8> {
    let mut b = make_gguf_string(key);
    b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
    b.extend_from_slice(&value.to_le_bytes());
    b
}

fn make_kv_string(key: &str, value: &str) -> Vec<u8> {
    let mut b = make_gguf_string(key);
    b.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    b.extend_from_slice(&make_gguf_string(value));
    b
}

fn make_kv_bool(key: &str, value: bool) -> Vec<u8> {
    let mut b = make_gguf_string(key);
    b.extend_from_slice(&(GgufValueType::Bool as u32).to_le_bytes());
    b.push(if value { 1 } else { 0 });
    b
}

fn make_kv_f32(key: &str, value: f32) -> Vec<u8> {
    let mut b = make_gguf_string(key);
    b.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
    b.extend_from_slice(&value.to_le_bytes());
    b
}

fn make_kv_array_u32(key: &str, values: &[u32]) -> Vec<u8> {
    let mut b = make_gguf_string(key);
    b.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
    // array element type
    b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
    // array count
    b.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &v in values {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn make_tensor_info_bytes(name: &str, shape: &[u64], type_id: u32, offset: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        b.extend_from_slice(&dim.to_le_bytes());
    }
    b.extend_from_slice(&type_id.to_le_bytes());
    b.extend_from_slice(&offset.to_le_bytes());
    b
}

// ──────────────────────────────────────────────────────────────
// Header edge cases
// ──────────────────────────────────────────────────────────────

#[test]
fn malformed_magic_number_all_zeros() {
    let data = gguf_header_bytes(0x00000000, 3, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err());
    match result.expect_err("should fail on zero magic") {
        BonsaiError::InvalidMagic { magic } => assert_eq!(magic, 0),
        other => panic!("expected InvalidMagic, got: {other}"),
    }
}

#[test]
fn malformed_magic_number_reversed_bytes() {
    // "FUGF" instead of "GGUF"
    let data = gguf_header_bytes(0x47554647, 3, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err());
    match result.expect_err("should fail on reversed magic") {
        BonsaiError::InvalidMagic { magic } => assert_eq!(magic, 0x47554647),
        other => panic!("expected InvalidMagic, got: {other}"),
    }
}

#[test]
fn truncated_header_empty_input() {
    let data: &[u8] = &[];
    let result = GgufHeader::parse(data, 0);
    assert!(result.is_err());
    match result.expect_err("should fail on empty input") {
        BonsaiError::UnexpectedEof { .. } => {}
        other => panic!("expected UnexpectedEof, got: {other}"),
    }
}

#[test]
fn truncated_header_only_magic() {
    let data = GGUF_MAGIC.to_le_bytes();
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err());
    match result.expect_err("should fail on truncated after magic") {
        BonsaiError::UnexpectedEof { .. } => {}
        other => panic!("expected UnexpectedEof, got: {other}"),
    }
}

#[test]
fn truncated_header_missing_metadata_count() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes());
    // Missing metadata_kv_count
    let result = GgufHeader::parse(&data, 0);
    assert!(result.is_err());
}

#[test]
fn unsupported_gguf_version_0() {
    let data = gguf_header_bytes(GGUF_MAGIC, 0, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    match result.expect_err("version 0 should fail") {
        BonsaiError::UnsupportedVersion { version } => assert_eq!(version, 0),
        other => panic!("expected UnsupportedVersion, got: {other}"),
    }
}

#[test]
fn unsupported_gguf_version_1() {
    let data = gguf_header_bytes(GGUF_MAGIC, 1, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    match result.expect_err("version 1 should fail") {
        BonsaiError::UnsupportedVersion { version } => assert_eq!(version, 1),
        other => panic!("expected UnsupportedVersion, got: {other}"),
    }
}

#[test]
fn unsupported_gguf_version_4() {
    let data = gguf_header_bytes(GGUF_MAGIC, 4, 0, 0);
    let result = GgufHeader::parse(&data, 0);
    match result.expect_err("version 4 should fail") {
        BonsaiError::UnsupportedVersion { version } => assert_eq!(version, 4),
        other => panic!("expected UnsupportedVersion, got: {other}"),
    }
}

#[test]
fn supported_gguf_version_2() {
    let data = gguf_header_bytes(GGUF_MAGIC, 2, 5, 3);
    let (header, offset) = GgufHeader::parse(&data, 0).expect("v2 should be supported");
    assert_eq!(header.version, 2);
    assert_eq!(header.tensor_count, 5);
    assert_eq!(header.metadata_kv_count, 3);
    assert_eq!(offset, 24);
}

#[test]
fn supported_gguf_version_3() {
    let data = gguf_header_bytes(GGUF_MAGIC, 3, 291, 25);
    let (header, _) = GgufHeader::parse(&data, 0).expect("v3 should be supported");
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 291);
}

#[test]
fn header_read_from_reader() {
    let data = gguf_header_bytes(GGUF_MAGIC, 3, 10, 5);
    let mut cursor = std::io::Cursor::new(&data);
    let header = GgufHeader::read_from(&mut cursor).expect("should read from cursor");
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 10);
    assert_eq!(header.metadata_kv_count, 5);
}

// ──────────────────────────────────────────────────────────────
// Invalid metadata value type IDs
// ──────────────────────────────────────────────────────────────

#[test]
fn invalid_metadata_value_type_13() {
    let result = GgufValueType::from_id(13);
    assert!(result.is_err());
}

#[test]
fn invalid_metadata_value_type_255() {
    let result = GgufValueType::from_id(255);
    assert!(result.is_err());
}

#[test]
fn valid_metadata_value_type_roundtrip() {
    for id in 0..=12 {
        let result = GgufValueType::from_id(id);
        assert!(result.is_ok(), "type id {id} should be valid");
    }
}

// ──────────────────────────────────────────────────────────────
// Metadata edge cases
// ──────────────────────────────────────────────────────────────

#[test]
fn empty_metadata_store() {
    let store = MetadataStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    assert!(store.get("anything").is_none());
    assert!(store.get_u32("key").is_err());
    assert!(store.get_string("key").is_err());
    assert!(store.get_f32("key").is_err());
    assert!(store.get_u64("key").is_err());
}

#[test]
fn metadata_parse_zero_count() {
    let data: Vec<u8> = vec![0; 64];
    let (store, _) = MetadataStore::parse(&data, 0, 0).expect("zero count should succeed");
    assert!(store.is_empty());
}

#[test]
fn metadata_with_long_string_key() {
    let long_key = "a".repeat(10_000);
    let bytes = make_kv_u32(&long_key, 42);
    let (store, _) = MetadataStore::parse(&bytes, 0, 1).expect("long key should parse");
    assert_eq!(store.get_u32(&long_key).expect("should find long key"), 42);
}

#[test]
fn metadata_with_long_string_value() {
    let long_value = "b".repeat(50_000);
    let bytes = make_kv_string("test.key", &long_value);
    let (store, _) = MetadataStore::parse(&bytes, 0, 1).expect("long value should parse");
    assert_eq!(
        store.get_string("test.key").expect("should find key"),
        long_value
    );
}

#[test]
fn metadata_value_type_conversions() {
    let v_u32 = MetadataValue::Uint32(100);
    assert_eq!(v_u32.as_u32(), Some(100));
    assert_eq!(v_u32.as_u64(), Some(100));
    assert!(v_u32.as_str().is_none());
    assert!(v_u32.as_bool().is_none());
    assert!(v_u32.as_f32().is_none());

    let v_str = MetadataValue::String("hello".to_string());
    assert_eq!(v_str.as_str(), Some("hello"));
    assert!(v_str.as_u32().is_none());

    let v_bool = MetadataValue::Bool(true);
    assert_eq!(v_bool.as_bool(), Some(true));

    let v_f32 = MetadataValue::Float32(std::f32::consts::PI);
    assert!(v_f32.as_f32().is_some());

    let v_u64 = MetadataValue::Uint64(u64::MAX);
    assert_eq!(v_u64.as_u64(), Some(u64::MAX));
    // u64::MAX can't fit in u32
    assert!(v_u64.as_u32().is_none());

    let v_i32 = MetadataValue::Int32(-1);
    // Negative i32 can't convert to u32
    assert!(v_i32.as_u32().is_none());

    let v_i64 = MetadataValue::Int64(-5);
    assert!(v_i64.as_u64().is_none());

    let v_f64 = MetadataValue::Float64(std::f64::consts::E);
    assert!(v_f64.as_f32().is_some());
}

#[test]
fn metadata_get_with_default() {
    let store = MetadataStore::new();
    assert_eq!(store.get_u32_or("missing", 99), 99);
    assert!((store.get_f32_or("missing", 1.5) - 1.5).abs() < f32::EPSILON);
}

#[test]
fn metadata_multiple_entries() {
    let mut data = Vec::new();
    data.extend_from_slice(&make_kv_u32("key1", 10));
    data.extend_from_slice(&make_kv_u32("key2", 20));
    data.extend_from_slice(&make_kv_string("key3", "hello"));
    let (store, _) = MetadataStore::parse(&data, 0, 3).expect("should parse 3 entries");
    assert_eq!(store.len(), 3);
    assert_eq!(store.get_u32("key1").expect("key1"), 10);
    assert_eq!(store.get_u32("key2").expect("key2"), 20);
    assert_eq!(store.get_string("key3").expect("key3"), "hello");
}

#[test]
fn metadata_duplicate_keys_last_wins() {
    let mut data = Vec::new();
    data.extend_from_slice(&make_kv_u32("dup", 1));
    data.extend_from_slice(&make_kv_u32("dup", 2));
    let (store, _) = MetadataStore::parse(&data, 0, 2).expect("should parse duplicates");
    assert_eq!(store.len(), 1); // HashMap deduplicates
    assert_eq!(store.get_u32("dup").expect("dup key"), 2);
}

#[test]
fn metadata_bool_entry() {
    let bytes = make_kv_bool("flag", true);
    let (store, _) = MetadataStore::parse(&bytes, 0, 1).expect("bool should parse");
    let val = store.get("flag").expect("should find flag");
    assert_eq!(val.as_bool(), Some(true));
}

#[test]
fn metadata_f32_entry() {
    let bytes = make_kv_f32("epsilon", 1e-6);
    let (store, _) = MetadataStore::parse(&bytes, 0, 1).expect("f32 should parse");
    let val = store.get_f32("epsilon").expect("should find epsilon");
    assert!((val - 1e-6).abs() < 1e-10);
}

#[test]
fn metadata_array_u32() {
    let bytes = make_kv_array_u32("arr", &[1, 2, 3, 4, 5]);
    let (store, _) = MetadataStore::parse(&bytes, 0, 1).expect("array should parse");
    let val = store.get("arr").expect("should find arr");
    match val {
        MetadataValue::Array(arr) => {
            assert_eq!(arr.len(), 5);
            assert_eq!(arr[0].as_u32(), Some(1));
            assert_eq!(arr[4].as_u32(), Some(5));
        }
        other => panic!("expected Array, got: {other:?}"),
    }
}

// ──────────────────────────────────────────────────────────────
// Tensor info edge cases
// ──────────────────────────────────────────────────────────────

#[test]
fn tensor_info_zero_dimensions() {
    let data = make_tensor_info_bytes("scalar", &[], 0, 0);
    let (store, _) = TensorStore::parse(&data, 0, 1).expect("zero-dim should parse");
    let info = store.require("scalar").expect("should find scalar");
    assert_eq!(info.n_dims(), 0);
    // Product of empty shape = 1
    assert_eq!(info.element_count(), 1);
}

#[test]
fn tensor_info_single_dimension() {
    let data = make_tensor_info_bytes("vec", &[256], 41, 0);
    let (store, _) = TensorStore::parse(&data, 0, 1).expect("1d should parse");
    let info = store.require("vec").expect("should find vec");
    assert_eq!(info.n_dims(), 1);
    assert_eq!(info.element_count(), 256);
    // 256/128 = 2 blocks * 18 bytes = 36
    assert_eq!(info.data_size(), 36);
}

#[test]
fn tensor_info_large_shape() {
    let data = make_tensor_info_bytes("big", &[4096, 4096], 41, 0);
    let (store, _) = TensorStore::parse(&data, 0, 1).expect("large should parse");
    let info = store.require("big").expect("should find big");
    assert_eq!(info.element_count(), 4096 * 4096);
    let blocks = 4096u64 * 4096 / 128;
    assert_eq!(info.data_size(), blocks * 18);
}

#[test]
fn tensor_store_missing_tensor() {
    let store = TensorStore::new();
    assert!(store.require("nonexistent").is_err());
    assert!(store.get("nonexistent").is_none());
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

#[test]
fn tensor_store_sorted_names() {
    let mut data = Vec::new();
    data.extend_from_slice(&make_tensor_info_bytes("zzz", &[128], 0, 0));
    data.extend_from_slice(&make_tensor_info_bytes("aaa", &[128], 0, 0));
    data.extend_from_slice(&make_tensor_info_bytes("mmm", &[128], 0, 0));
    let (store, _) = TensorStore::parse(&data, 0, 3).expect("should parse 3 tensors");
    let names = store.sorted_names();
    assert_eq!(names, vec!["aaa", "mmm", "zzz"]);
}

#[test]
fn tensor_store_count_by_type() {
    let mut data = Vec::new();
    data.extend_from_slice(&make_tensor_info_bytes("t1", &[128], 41, 0));
    data.extend_from_slice(&make_tensor_info_bytes("t2", &[128], 41, 0));
    data.extend_from_slice(&make_tensor_info_bytes("t3", &[128], 0, 0));
    let (store, _) = TensorStore::parse(&data, 0, 3).expect("should parse");
    let counts = store.count_by_type();
    assert_eq!(counts.get(&GgufTensorType::Q1_0_g128), Some(&2));
    assert_eq!(counts.get(&GgufTensorType::F32), Some(&1));
}

#[test]
fn tensor_type_properties() {
    let q1 = GgufTensorType::Q1_0_g128;
    assert!(q1.is_one_bit());
    assert_eq!(q1.block_size(), 128);
    assert_eq!(q1.block_bytes(), 18);
    assert_eq!(q1.name(), "Q1_0_g128");
    assert_eq!(format!("{q1}"), "Q1_0_g128");

    let f32_t = GgufTensorType::F32;
    assert!(!f32_t.is_one_bit());
    assert_eq!(f32_t.block_size(), 1);
    assert_eq!(f32_t.block_bytes(), 4);
}

#[test]
fn tensor_type_unknown_id_returns_error() {
    // Note: 35=TQ2_0, 41=Q1_0_g128, 42=TQ2_0_g128, 43=F8_E4M3, 44=F8_E5M2 are valid; excluded here.
    for bad_id in [4, 5, 16, 20, 29, 31, 40, 45, 100, u32::MAX] {
        let result = GgufTensorType::from_id(bad_id);
        assert!(result.is_err(), "type id {bad_id} should be unsupported");
    }
}

// ──────────────────────────────────────────────────────────────
// Alignment calculations
// ──────────────────────────────────────────────────────────────

#[test]
fn alignment_to_32_byte_boundary() {
    // align_offset is private, but we can test via GgufFile parse behavior.
    // We test the formula: (offset + alignment - 1) & !(alignment - 1)
    fn align(offset: usize, alignment: usize) -> usize {
        (offset + alignment - 1) & !(alignment - 1)
    }
    assert_eq!(align(0, 32), 0);
    assert_eq!(align(1, 32), 32);
    assert_eq!(align(31, 32), 32);
    assert_eq!(align(32, 32), 32);
    assert_eq!(align(33, 32), 64);
    assert_eq!(align(63, 32), 64);
    assert_eq!(align(64, 32), 64);
    assert_eq!(align(100, 32), 128);
    // Various alignments
    assert_eq!(align(1, 64), 64);
    assert_eq!(align(65, 64), 128);
    assert_eq!(align(0, 1), 0);
    assert_eq!(align(5, 1), 5);
}

// ──────────────────────────────────────────────────────────────
// Full GgufFile parse edge cases
// ──────────────────────────────────────────────────────────────

#[test]
fn gguf_file_parse_minimal_valid() {
    // Build a minimal GGUF file: header + 0 metadata + 0 tensors
    let mut data = gguf_header_bytes(GGUF_MAGIC, 3, 0, 0);
    // Pad to 32-byte alignment for data section
    while data.len() < 32 {
        data.push(0);
    }
    let file = GgufFile::parse(&data).expect("minimal GGUF should parse");
    assert_eq!(file.header.version, 3);
    assert!(file.metadata.is_empty());
    assert!(file.tensors.is_empty());
    assert_eq!(file.data_offset, 32); // aligned to 32
}

#[test]
fn gguf_file_parse_with_one_metadata_entry() {
    let mut data = gguf_header_bytes(GGUF_MAGIC, 3, 0, 1);
    data.extend_from_slice(&make_kv_u32("test.key", 42));
    // Pad for alignment
    while data.len() < 64 {
        data.push(0);
    }
    let file = GgufFile::parse(&data).expect("should parse with metadata");
    assert_eq!(file.metadata.len(), 1);
    assert_eq!(
        file.metadata.get_u32("test.key").expect("should find key"),
        42
    );
}

#[test]
fn gguf_file_parse_bad_magic_fails() {
    let data = gguf_header_bytes(0xDEADBEEF, 3, 0, 0);
    let result = GgufFile::parse(&data);
    assert!(result.is_err());
}

#[test]
fn metadata_store_iter() {
    let mut data = Vec::new();
    data.extend_from_slice(&make_kv_u32("x", 1));
    data.extend_from_slice(&make_kv_u32("y", 2));
    let (store, _) = MetadataStore::parse(&data, 0, 2).expect("should parse");
    let entries: Vec<_> = store.iter().collect();
    assert_eq!(entries.len(), 2);
}

#[test]
fn metadata_store_default_trait() {
    let store = MetadataStore::default();
    assert!(store.is_empty());
}

#[test]
fn tensor_store_default_trait() {
    let store = TensorStore::default();
    assert!(store.is_empty());
}
