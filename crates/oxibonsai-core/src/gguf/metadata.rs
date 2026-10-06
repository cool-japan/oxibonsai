//! GGUF metadata key-value store.
//!
//! The GGUF metadata section stores typed key-value pairs that describe
//! the model architecture, tokenizer configuration, and other properties.

use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::tensor_info::read_string_body_chunked;
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
    ///
    /// Accepts any GGUF integer scalar type that can represent the value
    /// exactly or via a non-negative widening/narrowing conversion:
    /// `Uint64`/`Int32` (pre-existing), and `Uint8`/`Int8`/`Uint16`/`Int16`
    /// — the four small-integer scalar types that, before this, had no
    /// reader anywhere (`general.sampling.top_k`-style `Int32` values
    /// already worked via `u32::try_from`).
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint64(v) => u32::try_from(*v).ok(),
            Self::Int32(v) => u32::try_from(*v).ok(),
            Self::Uint8(v) => Some(u32::from(*v)),
            Self::Int8(v) => u32::try_from(*v).ok(),
            Self::Uint16(v) => Some(u32::from(*v)),
            Self::Int16(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract a u64 value.
    ///
    /// See [`as_u32`](Self::as_u32) for why `Uint8`/`Int8`/`Uint16`/`Int16`
    /// are accepted alongside the pre-existing `Uint32`/`Int64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(u64::from(*v)),
            Self::Int64(v) => u64::try_from(*v).ok(),
            Self::Uint8(v) => Some(u64::from(*v)),
            Self::Int8(v) => u64::try_from(*v).ok(),
            Self::Uint16(v) => Some(u64::from(*v)),
            Self::Int16(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract a f32 value.
    ///
    /// See [`as_u32`](Self::as_u32) for why `Uint8`/`Int8`/`Uint16`/`Int16`
    /// are accepted alongside the pre-existing `Float32`/`Float64`; all
    /// four integer widths convert to `f32` exactly (lossless).
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            Self::Uint8(v) => Some(f32::from(*v)),
            Self::Int8(v) => Some(f32::from(*v)),
            Self::Uint16(v) => Some(f32::from(*v)),
            Self::Int16(v) => Some(f32::from(*v)),
            _ => None,
        }
    }

    /// Try to extract a f64 value.
    ///
    /// Mirrors [`as_f32`](Self::as_f32)'s widening so the two float
    /// accessors stay symmetric: `Uint8`/`Int8`/`Uint16`/`Int16` all
    /// convert to `f64` exactly (lossless), alongside `Float32`/`Float64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => Some(*v),
            Self::Float32(v) => Some(f64::from(*v)),
            Self::Uint8(v) => Some(f64::from(*v)),
            Self::Int8(v) => Some(f64::from(*v)),
            Self::Uint16(v) => Some(f64::from(*v)),
            Self::Int16(v) => Some(f64::from(*v)),
            _ => None,
        }
    }

    /// Try to extract an i32 value, preserving sign.
    ///
    /// Accepts any GGUF integer scalar type whose value fits in `i32`:
    /// `Uint8`/`Int8`/`Uint16`/`Int16` convert exactly (always in range);
    /// `Uint32`/`Int64`/`Uint64` convert via a range-checked
    /// `i32::try_from` and yield `None` when the value does not fit
    /// (e.g. a `Uint32` above `i32::MAX`), rather than silently wrapping
    /// or dropping the sign.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int32(v) => Some(*v),
            Self::Int64(v) => i32::try_from(*v).ok(),
            Self::Uint64(v) => i32::try_from(*v).ok(),
            Self::Uint32(v) => i32::try_from(*v).ok(),
            Self::Int16(v) => Some(i32::from(*v)),
            Self::Uint16(v) => Some(i32::from(*v)),
            Self::Int8(v) => Some(i32::from(*v)),
            Self::Uint8(v) => Some(i32::from(*v)),
            _ => None,
        }
    }

    /// Try to extract an i64 value, preserving sign.
    ///
    /// See [`as_i32`](Self::as_i32); the only value that can fail to fit is
    /// a `Uint64` above `i64::MAX`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int64(v) => Some(*v),
            Self::Uint64(v) => i64::try_from(*v).ok(),
            Self::Int32(v) => Some(i64::from(*v)),
            Self::Uint32(v) => Some(i64::from(*v)),
            Self::Int16(v) => Some(i64::from(*v)),
            Self::Uint16(v) => Some(i64::from(*v)),
            Self::Int8(v) => Some(i64::from(*v)),
            Self::Uint8(v) => Some(i64::from(*v)),
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

    /// Try to extract an array's elements as a slice.
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// A short, stable name for this value's underlying GGUF type.
    ///
    /// Used to build precise type-mismatch error messages (naming both the
    /// offending key and the actual type found) in [`MetadataStore`]'s
    /// `get_*` accessors, and available to downstream crates building their
    /// own such messages (e.g. a Hadamard/config accessor reporting why a
    /// key could not be read as the type it expected).
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Uint8(_) => "Uint8",
            Self::Int8(_) => "Int8",
            Self::Uint16(_) => "Uint16",
            Self::Int16(_) => "Int16",
            Self::Uint32(_) => "Uint32",
            Self::Int32(_) => "Int32",
            Self::Float32(_) => "Float32",
            Self::Bool(_) => "Bool",
            Self::String(_) => "String",
            Self::Array(_) => "Array",
            Self::Uint64(_) => "Uint64",
            Self::Int64(_) => "Int64",
            Self::Float64(_) => "Float64",
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

    /// Get a required bool value, returning an error if missing or not a
    /// bool.
    pub fn get_bool(&self, key: &str) -> BonsaiResult<bool> {
        match self.get(key) {
            None => Err(BonsaiError::MissingConfigKey {
                key: key.to_string(),
            }),
            Some(value) => value.as_bool().ok_or_else(|| BonsaiError::InvalidMetadata {
                key: key.to_string(),
                reason: format!("expected bool, found {}", value.type_name()),
            }),
        }
    }

    /// Get a required array value, returning an error if the key is
    /// missing or its value is not an array. On a type mismatch the error
    /// names both the key and the value's actual GGUF type.
    pub fn get_array(&self, key: &str) -> BonsaiResult<&[MetadataValue]> {
        match self.get(key) {
            None => Err(BonsaiError::MissingConfigKey {
                key: key.to_string(),
            }),
            Some(value) => value
                .as_array()
                .ok_or_else(|| BonsaiError::InvalidMetadata {
                    key: key.to_string(),
                    reason: format!("expected an array, found {}", value.type_name()),
                }),
        }
    }

    /// Get a required array of i32 values.
    ///
    /// Every element must be representable as `i32` (see
    /// [`MetadataValue::as_i32`]); an element that is not an integer, or an
    /// integer that does not fit, is a hard error naming the array's key
    /// and that element's actual GGUF type — never a silent truncation or
    /// sign flip (e.g. smuggling `-1` through an unsigned accessor and
    /// getting back `4294967295`).
    pub fn get_i32_array(&self, key: &str) -> BonsaiResult<Vec<i32>> {
        self.get_array(key)?
            .iter()
            .map(|v| {
                v.as_i32().ok_or_else(|| BonsaiError::InvalidMetadata {
                    key: key.to_string(),
                    reason: format!(
                        "array element is not representable as i32, found {}",
                        v.type_name()
                    ),
                })
            })
            .collect()
    }

    /// Get a required array of string values.
    ///
    /// Every element must be a `String`; a non-string element is a hard
    /// error naming the array's key and that element's actual GGUF type.
    pub fn get_string_array(&self, key: &str) -> BonsaiResult<Vec<String>> {
        self.get_array(key)?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| BonsaiError::InvalidMetadata {
                        key: key.to_string(),
                        reason: format!("array element is not a string, found {}", v.type_name()),
                    })
            })
            .collect()
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
/// The declared `len` prefix is attacker-controlled and only capped against
/// [`MAX_STRING_LEN`] (256 MiB); the actual bounded-chunk read loop lives in
/// [`read_string_body_chunked`] (core-gguf-20 / sec-10) — this used to be an independent copy of
/// that exact loop, one of three in the crate.
fn read_gguf_string<R: Read>(reader: &mut R) -> BonsaiResult<String> {
    let len = reader
        .read_u64::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if len > MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString { offset: 0 });
    }
    read_string_body_chunked(reader, len)
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

    fn make_kv_scalar_u8(key: &str, value: u8) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Uint8 as u32).to_le_bytes());
        bytes.push(value);
        bytes
    }

    fn make_kv_scalar_i8(key: &str, value: i8) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Int8 as u32).to_le_bytes());
        bytes.push(value.to_le_bytes()[0]);
        bytes
    }

    fn make_kv_scalar_u16(key: &str, value: u16) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Uint16 as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes
    }

    fn make_kv_scalar_i16(key: &str, value: i16) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Int16 as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes
    }

    fn make_kv_i32_array(key: &str, values: &[i32]) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        bytes.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    fn make_kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            bytes.extend_from_slice(&make_string_bytes(v));
        }
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

    /// Reproduces the exact `sign_values` prefix from the real Bonsai 2 27B
    /// `prism.hadamard.sign_values` metadata array (`[-1,-1,-1,1,-1,1]`, per
    /// `core-gguf-08`'s evidence), plus a companion string array and every
    /// small-integer scalar type that previously had no accessor at all
    /// (`Uint8`/`Int8`/`Uint16`/`Int16`). Every value must read back with
    /// the correct sign, not just a plausible-looking one.
    #[test]
    fn reads_i32_array_string_array_and_small_int_scalars_with_correct_sign() {
        let mut data = Vec::new();
        data.extend_from_slice(&make_kv_i32_array("sign_values", &[-1, -1, -1, 1, -1, 1]));
        data.extend_from_slice(&make_kv_string_array("names", &["alpha", "beta"]));
        data.extend_from_slice(&make_kv_scalar_u8("u8_val", 200));
        data.extend_from_slice(&make_kv_scalar_i8("i8_val", -100));
        data.extend_from_slice(&make_kv_scalar_u16("u16_val", 60_000));
        data.extend_from_slice(&make_kv_scalar_i16("i16_val", -30_000));

        let (store, _) =
            MetadataStore::parse(&data, 0, 6).expect("well-formed metadata block should parse");
        assert_eq!(store.len(), 6);

        assert_eq!(
            store
                .get_i32_array("sign_values")
                .expect("sign_values should read as an i32 array"),
            vec![-1, -1, -1, 1, -1, 1]
        );
        assert_eq!(
            store
                .get_string_array("names")
                .expect("names should read as a string array"),
            vec!["alpha".to_string(), "beta".to_string()]
        );

        // Small-integer scalars: as_u32/as_u64/as_f32/as_f64 widened,
        // as_i32/as_i64 preserve sign.
        let u8_val = store.get("u8_val").expect("u8_val should exist");
        assert_eq!(u8_val.as_u32(), Some(200));
        assert_eq!(u8_val.as_i32(), Some(200));
        assert_eq!(u8_val.as_f32(), Some(200.0));
        assert_eq!(u8_val.as_f64(), Some(200.0));

        let i8_val = store.get("i8_val").expect("i8_val should exist");
        assert_eq!(i8_val.as_i32(), Some(-100), "sign must be preserved");
        assert_eq!(i8_val.as_i64(), Some(-100), "sign must be preserved");
        assert_eq!(
            i8_val.as_u32(),
            None,
            "a negative value must not silently reinterpret as a large u32"
        );
        assert_eq!(i8_val.as_f32(), Some(-100.0));

        let u16_val = store.get("u16_val").expect("u16_val should exist");
        assert_eq!(u16_val.as_u32(), Some(60_000));
        assert_eq!(u16_val.as_i32(), Some(60_000));

        let i16_val = store.get("i16_val").expect("i16_val should exist");
        assert_eq!(i16_val.as_i32(), Some(-30_000), "sign must be preserved");
        assert_eq!(
            i16_val.as_u32(),
            None,
            "a negative value must not silently reinterpret as a large u32"
        );
        assert_eq!(i16_val.as_u64(), None);
        assert_eq!(i16_val.as_f64(), Some(-30_000.0));
    }

    /// Type-mismatch errors from the array/bool accessors must name both
    /// the offending key and the value's actual GGUF type, never just
    /// "wrong type".
    #[test]
    fn get_array_accessors_name_the_key_and_actual_type_on_mismatch() {
        let mut data = Vec::new();
        data.extend_from_slice(&make_kv_i32_array("ints", &[1, 2, 3]));
        data.extend_from_slice(&make_kv_string_array("strings", &["a"]));
        data.extend_from_slice(&make_kv_u32("scalar", 7));
        let (store, _) = MetadataStore::parse(&data, 0, 3).expect("metadata block should parse");

        match store.get_array("missing") {
            Err(BonsaiError::MissingConfigKey { key }) => assert_eq!(key, "missing"),
            other => panic!("expected MissingConfigKey, got: {other:?}"),
        }

        match store.get_array("scalar") {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "scalar");
                assert!(reason.contains("Uint32"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata, got: {other:?}"),
        }

        match store.get_string_array("ints") {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "ints");
                assert!(reason.contains("Int32"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata, got: {other:?}"),
        }

        match store.get_i32_array("strings") {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "strings");
                assert!(reason.contains("String"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata, got: {other:?}"),
        }

        match store.get_bool("scalar") {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "scalar");
                assert!(reason.contains("Uint32"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata, got: {other:?}"),
        }
    }

    #[test]
    fn get_bool_reads_true_and_false() {
        let mut kv_true = make_string_bytes("flag_true");
        kv_true.extend_from_slice(&(GgufValueType::Bool as u32).to_le_bytes());
        kv_true.push(1);
        let mut kv_false = make_string_bytes("flag_false");
        kv_false.extend_from_slice(&(GgufValueType::Bool as u32).to_le_bytes());
        kv_false.push(0);

        let mut data = Vec::new();
        data.extend_from_slice(&kv_true);
        data.extend_from_slice(&kv_false);

        let (store, _) = MetadataStore::parse(&data, 0, 2).expect("metadata block should parse");
        assert!(store.get_bool("flag_true").expect("flag_true should read"));
        assert!(!store
            .get_bool("flag_false")
            .expect("flag_false should read"));
    }
}
