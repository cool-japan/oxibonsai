//! GGUF metadata key-value store.
//!
//! The GGUF metadata section stores typed key-value pairs that describe
//! the model architecture, tokenizer configuration, and other properties.

use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::types::GgufValueType;

/// A typed metadata value from the GGUF key-value store.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    /// Try to extract a u32 value.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint64(v) => u32::try_from(*v).ok(),
            Self::Int32(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract a u64 value.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(u64::from(*v)),
            Self::Int64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract a f32 value.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to extract a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    /// Try to extract a bool value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Key-value metadata store from GGUF file.
#[derive(Debug, Clone)]
pub struct MetadataStore {
    entries: HashMap<String, MetadataValue>,
}

impl MetadataStore {
    /// Create an empty metadata store.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Read metadata entries from a byte-slice cursor.
    pub fn parse(data: &[u8], offset: usize, count: u64) -> BonsaiResult<(Self, usize)> {
        let mut cursor = std::io::Cursor::new(data);
        cursor.set_position(offset as u64);

        let mut store = Self::new();
        for _ in 0..count {
            let (key, value) = read_kv_pair(&mut cursor)?;
            // A duplicate key silently overwrites the earlier entry via plain
            // `HashMap::insert`, with no error, warning, or trace of the
            // discarded value anywhere — reject it as a hard parse error
            // instead, since a corrupted/adversarial file with duplicate keys
            // must not silently load as if nothing were wrong.
            if store.entries.contains_key(&key) {
                return Err(BonsaiError::InvalidMetadata {
                    key: key.clone(),
                    reason: "duplicate metadata key".to_string(),
                });
            }
            store.entries.insert(key, value);
        }

        Ok((store, cursor.position() as usize))
    }

    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Get a required string value, returning an error if missing.
    pub fn get_string(&self, key: &str) -> BonsaiResult<&str> {
        self.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get a required u32 value, returning an error if missing.
    pub fn get_u32(&self, key: &str) -> BonsaiResult<u32> {
        self.get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get a required u64 value, returning an error if missing.
    pub fn get_u64(&self, key: &str) -> BonsaiResult<u64> {
        self.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get a required f32 value.
    pub fn get_f32(&self, key: &str) -> BonsaiResult<f32> {
        self.get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get an optional u32 value with a default.
    pub fn get_u32_or(&self, key: &str, default: u32) -> u32 {
        self.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
    }

    /// Get an optional f32 value with a default.
    pub fn get_f32_or(&self, key: &str, default: f32) -> f32 {
        self.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum string length we accept from GGUF metadata (256 MB).
const MAX_STRING_LEN: u64 = 256 * 1024 * 1024;

/// Maximum array element count we accept from GGUF metadata (16 M entries).
const MAX_ARRAY_COUNT: u64 = 16 * 1024 * 1024;

/// Maximum nesting depth for `Array`-of-`Array` metadata values.
///
/// Each nesting level costs only 12 bytes on disk (`elem_type: u32` +
/// `count: u64`), so without a depth cap a small, deliberately crafted file
/// can drive `read_value` tens of thousands of stack frames deep and blow
/// the stack (an unrecoverable process abort, not a catchable `Result`
/// error). No legitimate GGUF file nests arrays anywhere close to this
/// deep, so 32 comfortably covers real usage while bounding stack growth.
const MAX_ARRAY_NESTING_DEPTH: u32 = 32;

/// Upper bound on a single eager read chunk (and the initial `Vec`
/// reservation) while reading a declared-length GGUF string.
///
/// The declared `len` prefix is attacker-controlled and only capped against
/// [`MAX_STRING_LEN`] (256 MiB), so allocating `len` bytes up front before
/// confirming the reader actually has that much data left lets a tiny file
/// force a large allocation. Reading (and growing the buffer) in bounded
/// chunks instead means peak allocation tracks bytes actually confirmed
/// present, not the attacker-declared length.
const STRING_READ_CHUNK: usize = 64 * 1024;

/// Bound on the eager `Vec` capacity reservation for a metadata `Array`.
///
/// The declared element `count` is attacker-controlled and only capped
/// against [`MAX_ARRAY_COUNT`] (16 M), so reserving `count` elements of
/// capacity up front — before a single element has been read — lets a tiny
/// file force a large allocation (and, via [`MAX_ARRAY_NESTING_DEPTH`]
/// nesting, several such allocations simultaneously live on the stack).
/// Reserving only a small bounded amount up front and letting the `Vec`
/// grow amortized as elements are actually parsed keeps peak allocation
/// proportional to real progress instead of a declared-but-undelivered
/// count.
const ARRAY_EAGER_RESERVE_CAP: usize = 4096;

/// Read a GGUF string: [u64 length] [utf8 bytes].
///
/// Reads in bounded [`STRING_READ_CHUNK`]-sized pieces rather than
/// allocating the full declared `len` up front, so a small file with a
/// large declared string length fails fast (on the first `read_exact` that
/// finds fewer bytes than requested) instead of first committing a
/// multi-hundred-MB buffer.
fn read_gguf_string<R: Read>(reader: &mut R) -> BonsaiResult<String> {
    let len = reader
        .read_u64::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if len > MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString { offset: 0 });
    }
    let mut buf = Vec::with_capacity((len as usize).min(STRING_READ_CHUNK));
    let mut remaining = len;
    let mut chunk = [0u8; STRING_READ_CHUNK];
    while remaining > 0 {
        let take = (remaining as usize).min(STRING_READ_CHUNK);
        reader
            .read_exact(&mut chunk[..take])
            .map_err(BonsaiError::MmapError)?;
        buf.extend_from_slice(&chunk[..take]);
        remaining -= take as u64;
    }
    String::from_utf8(buf).map_err(|_| BonsaiError::InvalidString { offset: 0 })
}

/// Read a single typed value from the reader.
fn read_value<R: Read>(reader: &mut R, value_type: GgufValueType) -> BonsaiResult<MetadataValue> {
    read_value_at_depth(reader, value_type, 0)
}

/// Read a single typed value from the reader, tracking `Array` nesting depth.
///
/// `depth` counts how many `Array` values enclose this call; it is only
/// incremented when recursing into an `Array` element type, and is bounded
/// by [`MAX_ARRAY_NESTING_DEPTH`] so a maliciously nested `Array`-of-`Array`
/// chain fails cleanly with a `Result` error instead of overflowing the
/// stack.
fn read_value_at_depth<R: Read>(
    reader: &mut R,
    value_type: GgufValueType,
    depth: u32,
) -> BonsaiResult<MetadataValue> {
    match value_type {
        GgufValueType::Uint8 => {
            let v = reader.read_u8().map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint8(v))
        }
        GgufValueType::Int8 => {
            let v = reader.read_i8().map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int8(v))
        }
        GgufValueType::Uint16 => {
            let v = reader
                .read_u16::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint16(v))
        }
        GgufValueType::Int16 => {
            let v = reader
                .read_i16::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int16(v))
        }
        GgufValueType::Uint32 => {
            let v = reader
                .read_u32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint32(v))
        }
        GgufValueType::Int32 => {
            let v = reader
                .read_i32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int32(v))
        }
        GgufValueType::Float32 => {
            let v = reader
                .read_f32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Float32(v))
        }
        GgufValueType::Bool => {
            let v = reader.read_u8().map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Bool(v != 0))
        }
        GgufValueType::String => {
            let s = read_gguf_string(reader)?;
            Ok(MetadataValue::String(s))
        }
        GgufValueType::Array => {
            let next_depth = depth + 1;
            if next_depth > MAX_ARRAY_NESTING_DEPTH {
                return Err(BonsaiError::InvalidMetadata {
                    key: String::new(),
                    reason: format!(
                        "array nesting depth {next_depth} exceeds maximum of {MAX_ARRAY_NESTING_DEPTH}"
                    ),
                });
            }
            let elem_type_id = reader
                .read_u32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            let elem_type = GgufValueType::from_id(elem_type_id)?;
            let count = reader
                .read_u64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            if count > MAX_ARRAY_COUNT {
                return Err(BonsaiError::InvalidString { offset: 0 });
            }
            let mut values = Vec::with_capacity((count as usize).min(ARRAY_EAGER_RESERVE_CAP));
            for _ in 0..count {
                values.push(read_value_at_depth(reader, elem_type, next_depth)?);
            }
            Ok(MetadataValue::Array(values))
        }
        GgufValueType::Uint64 => {
            let v = reader
                .read_u64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint64(v))
        }
        GgufValueType::Int64 => {
            let v = reader
                .read_i64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int64(v))
        }
        GgufValueType::Float64 => {
            let v = reader
                .read_f64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Float64(v))
        }
    }
}

/// Read a key-value pair from the reader.
fn read_kv_pair<R: Read>(reader: &mut R) -> BonsaiResult<(String, MetadataValue)> {
    let key = read_gguf_string(reader)?;
    let value_type_id = reader
        .read_u32::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    let value_type = GgufValueType::from_id(value_type_id)?;
    let value = read_value(reader, value_type)?;
    Ok((key, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_string_bytes(s: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
        bytes
    }

    fn make_kv_u32(key: &str, value: u32) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes
    }

    #[test]
    fn parse_single_u32_metadata() {
        let data = make_kv_u32("test.key", 42);
        let (store, _) = MetadataStore::parse(&data, 0, 1).expect("metadata parse should succeed");
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.get_u32("test.key").expect("test.key should exist"),
            42
        );
    }

    #[test]
    fn missing_key_returns_error() {
        let store = MetadataStore::new();
        assert!(store.get_u32("nonexistent").is_err());
    }

    /// A duplicate metadata key must be a hard parse error, not a silent
    /// last-wins overwrite via `HashMap::insert`.
    #[test]
    fn duplicate_metadata_key_is_hard_error() {
        let mut data = make_kv_u32("dup", 1);
        data.extend_from_slice(&make_kv_u32("dup", 2));
        let result = MetadataStore::parse(&data, 0, 2);
        match result {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "dup");
                assert!(reason.contains("duplicate"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata for duplicate key, got: {other:?}"),
        }
    }

    /// A very long string (well under `MAX_STRING_LEN` but larger than a
    /// single `STRING_READ_CHUNK`) must still round-trip exactly through
    /// the chunked reader.
    #[test]
    fn long_string_spanning_multiple_read_chunks_round_trips() {
        let long_value = "x".repeat(200 * 1024); // > STRING_READ_CHUNK (64 KiB)
        let mut bytes = make_string_bytes("long.key");
        bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        bytes.extend_from_slice(&make_string_bytes(&long_value));

        let (store, _) = MetadataStore::parse(&bytes, 0, 1).expect("long string should parse");
        assert_eq!(
            store
                .get("long.key")
                .and_then(|v| v.as_str())
                .expect("long.key should exist"),
            long_value
        );
    }

    /// A declared string length larger than what's actually present must
    /// fail cleanly (on the first short chunk read) rather than allocate
    /// the full declared length up front.
    #[test]
    fn string_declared_longer_than_available_data_fails_cleanly() {
        let mut bytes = make_string_bytes("short.key");
        bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        // Declare a 10 MiB string but supply none of the bytes.
        bytes.extend_from_slice(&(10u64 * 1024 * 1024).to_le_bytes());

        let result = MetadataStore::parse(&bytes, 0, 1);
        assert!(
            result.is_err(),
            "truncated string data must fail cleanly, not allocate the full declared length"
        );
    }

    /// A chain of nested `[elem_type=Array(9)][count=1]` headers (12 bytes
    /// per level) must return a clean `Err` once the nesting exceeds
    /// [`MAX_ARRAY_NESTING_DEPTH`], never overflow the stack.
    #[test]
    fn deeply_nested_array_bomb_returns_error_not_stack_overflow() {
        let mut bytes = make_string_bytes("bomb.key");
        // Value type of the key itself: Array.
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());

        // Nest far beyond MAX_ARRAY_NESTING_DEPTH: each level is
        // [elem_type: u32 = Array][count: u64 = 1].
        let levels = (MAX_ARRAY_NESTING_DEPTH as usize) * 4;
        for _ in 0..levels {
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
        }
        // Innermost element type + a plausible value so a shallower parse
        // that ignored the depth cap would otherwise succeed.
        bytes.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
        bytes.push(7);

        let result = MetadataStore::parse(&bytes, 0, 1);
        assert!(
            result.is_err(),
            "nested array bomb should be rejected, not silently parsed or crash"
        );
        match result.expect_err("nested array bomb must error") {
            BonsaiError::InvalidMetadata { reason, .. } => {
                assert!(reason.contains("nesting depth"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata, got: {other}"),
        }
    }

    /// A chain of `MAX_ARRAY_NESTING_DEPTH` nested `Array` headers, each
    /// declaring `count = MAX_ARRAY_COUNT` (16 M) elements, from a file only
    /// a few hundred bytes long. Before the eager-allocation cap this forced
    /// up to `MAX_ARRAY_NESTING_DEPTH` simultaneous `Vec::with_capacity(16M)`
    /// reservations (an aggregate on the order of several GiB) purely from
    /// the declared counts, before a single element was confirmed to exist.
    /// This must still fail cleanly and quickly (bounded memory, no OOM
    /// abort) rather than hang or crash the process.
    #[test]
    fn nested_max_count_arrays_on_tiny_file_bounded_memory_clean_error() {
        let mut bytes = make_string_bytes("bomb.key");
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());

        // MAX_ARRAY_NESTING_DEPTH levels of Array-of-Array, each declaring
        // the maximum accepted element count, with no actual element data
        // following any of them.
        for _ in 0..MAX_ARRAY_NESTING_DEPTH {
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&MAX_ARRAY_COUNT.to_le_bytes());
        }

        // Well under 1 KB total, regardless of the declared counts above.
        assert!(bytes.len() < 1024, "crafted input must stay tiny");

        let result = MetadataStore::parse(&bytes, 0, 1);
        assert!(
            result.is_err(),
            "nested max-count array headers on a tiny file must fail cleanly, not hang/OOM"
        );
    }

    #[test]
    fn array_nesting_at_exactly_the_limit_is_accepted() {
        let mut bytes = make_string_bytes("ok.key");
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());

        // MAX_ARRAY_NESTING_DEPTH - 1 levels of Array-of-Array, terminated by
        // a final Array-of-Uint8 level so the total nesting depth reaches
        // exactly MAX_ARRAY_NESTING_DEPTH without exceeding it.
        for _ in 0..MAX_ARRAY_NESTING_DEPTH - 1 {
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
        }
        bytes.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.push(9);

        let (store, _) = MetadataStore::parse(&bytes, 0, 1)
            .expect("nesting exactly at the configured maximum should still parse");
        assert_eq!(store.len(), 1);
    }
}
