//! Model checkpoint format for saving and restoring training state.
//!
//! # Binary Format (version 1)
//!
//! ## Header
//! ```text
//! magic:        b"OXCK"   (4 bytes)
//! version:      u32 LE    (= 1)
//! flags:        u64 LE    (reserved, must be 0 on write; ignored on read)
//! num_tensors:  u64 LE
//! metadata_len: u32 LE
//! metadata:     UTF-8 JSON string (metadata_len bytes)
//! ```
//!
//! ## Per tensor
//! ```text
//! name_len:  u32 LE
//! name:      UTF-8 (name_len bytes)
//! ndim:      u32 LE
//! shape:     [u64 LE; ndim]
//! data_len:  u64 LE  (number of f32 elements)
//! data:      [f32 LE; data_len]
//! ```
//!
//! Metadata is serialised as a simple `{"key":"val",...}` JSON object
//! without nesting; keys and values must not contain `"` or `\`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint metadata key-value pairs.
pub type CheckpointMetadata = HashMap<String, String>;

/// A serialized model checkpoint containing metadata and named tensors.
#[derive(Debug)]
pub struct Checkpoint {
    /// Format version (always 1 for new checkpoints).
    pub version: u32,
    /// Arbitrary key-value metadata (e.g. step, loss, lr).
    pub metadata: CheckpointMetadata,
    /// Ordered list of tensor entries.
    pub tensors: Vec<CheckpointTensor>,
}

/// A single tensor entry in the checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointTensor {
    /// Unique tensor name within the checkpoint (e.g. `"layer.0.weight"`).
    pub name: String,
    /// N-dimensional shape; product must equal `data.len()`.
    pub shape: Vec<u64>,
    /// Raw `f32` data in row-major order.
    pub data: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointTensor
// ─────────────────────────────────────────────────────────────────────────────

impl CheckpointTensor {
    /// Construct a checkpoint tensor.
    pub fn new(name: impl Into<String>, data: Vec<f32>, shape: Vec<u64>) -> Self {
        Self {
            name: name.into(),
            shape,
            data,
        }
    }

    /// Total number of scalar elements: product of all shape dimensions.
    pub fn element_count(&self) -> u64 {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape.iter().product()
    }

    /// Size of the tensor data in bytes (`element_count * 4`).
    pub fn size_bytes(&self) -> usize {
        self.element_count() as usize * 4
    }

    /// Convert from a [`crate::model_merge::WeightTensor`].
    ///
    /// The `usize` shape dimensions are widened to `u64`.
    pub fn from_weight_tensor(wt: &crate::model_merge::WeightTensor) -> Self {
        Self {
            name: wt.name.clone(),
            shape: wt.shape.iter().map(|&d| d as u64).collect(),
            data: wt.data.clone(),
        }
    }

    /// Convert back to a [`crate::model_merge::WeightTensor`].
    ///
    /// The `u64` shape dimensions are narrowed to `usize`; values that do not
    /// fit in `usize` are clamped to `usize::MAX` (a safeguard — real models
    /// never have dimensions that large).
    pub fn to_weight_tensor(&self) -> crate::model_merge::WeightTensor {
        let shape: Vec<usize> = self
            .shape
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
            .collect();
        crate::model_merge::WeightTensor::new(self.name.clone(), self.data.clone(), shape)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint
// ─────────────────────────────────────────────────────────────────────────────

impl Checkpoint {
    /// Create an empty checkpoint (version 1, no metadata, no tensors).
    pub fn new() -> Self {
        Self {
            version: 1,
            metadata: CheckpointMetadata::new(),
            tensors: Vec::new(),
        }
    }

    /// Append a tensor to the checkpoint.
    pub fn add_tensor(&mut self, tensor: CheckpointTensor) {
        self.tensors.push(tensor);
    }

    /// Insert or replace a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Look up a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Find a tensor by name (linear scan; checkpoints are small).
    pub fn get_tensor(&self, name: &str) -> Option<&CheckpointTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total bytes occupied by all tensor data (`sum of size_bytes()`).
    pub fn total_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }

    /// Total number of `f32` parameters across all tensors.
    pub fn num_params(&self) -> u64 {
        self.tensors.iter().map(|t| t.element_count()).sum()
    }

    // ── file I/O ──────────────────────────────────────────────────────────────

    /// Save the checkpoint to `path`, creating or truncating the file.
    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }

    /// Load a checkpoint from `path`.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }

    // ── streaming I/O ─────────────────────────────────────────────────────────

    /// Serialise the checkpoint into `writer`.
    ///
    /// The writer is NOT flushed; callers that need it (e.g. `BufWriter`) must
    /// flush themselves, or use [`save`](Self::save) which wraps a `BufWriter`
    /// and flushes on drop.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), CheckpointError> {
        // ── header ──
        writer.write_all(b"OXCK")?;
        write_u32_le(writer, 1u32)?; // version
        write_u64_le(writer, 0u64)?; // flags (reserved)
        write_u64_le(writer, self.tensors.len() as u64)?;

        // metadata
        let meta_str = serialize_metadata(&self.metadata);
        let meta_bytes = meta_str.as_bytes();
        write_u32_le(writer, meta_bytes.len() as u32)?;
        writer.write_all(meta_bytes)?;

        // ── tensors ──
        for tensor in &self.tensors {
            let name_bytes = tensor.name.as_bytes();
            if name_bytes.len() > 65535 {
                return Err(CheckpointError::NameTooLong(name_bytes.len()));
            }
            write_u32_le(writer, name_bytes.len() as u32)?;
            writer.write_all(name_bytes)?;

            write_u32_le(writer, tensor.shape.len() as u32)?;
            for &dim in &tensor.shape {
                write_u64_le(writer, dim)?;
            }

            write_u64_le(writer, tensor.data.len() as u64)?;
            for &f in &tensor.data {
                writer.write_all(&f.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Deserialise a checkpoint from `reader`.
    ///
    /// Length-prefixed fields (`num_tensors`, `metadata_len`, `name_len`,
    /// `ndim`, `data_len`) are read from the untrusted stream but are never
    /// used to eagerly pre-allocate more than `MAX_EAGER_CAPACITY`
    /// elements/bytes of capacity up front — see `read_bytes_bounded` and
    /// the bounded loops below. This turns a truncated or maliciously large
    /// length field into an ordinary [`CheckpointError::TruncatedData`] once
    /// the stream actually runs out, instead of an immediate multi-gigabyte
    /// allocation attempt driven entirely by attacker-controlled input.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, CheckpointError> {
        // ── magic ──
        let mut magic = [0u8; 4];
        read_exact(reader, &mut magic)?;
        if &magic != b"OXCK" {
            return Err(CheckpointError::InvalidMagic(magic.to_vec()));
        }

        // ── version ──
        let version = read_u32_le(reader)?;
        if version == 0 || version > 1 {
            return Err(CheckpointError::UnsupportedVersion(version));
        }

        // ── flags (reserved) ──
        let _flags = read_u64_le(reader)?;

        // ── tensor count ──
        let num_tensors = read_u64_le(reader)? as usize;

        // ── metadata ──
        let meta_len = read_u32_le(reader)? as usize;
        let meta_bytes = read_bytes_bounded(reader, meta_len)?;
        let meta_str = std::str::from_utf8(&meta_bytes)
            .map_err(|e| CheckpointError::MetadataParse(e.to_string()))?;
        let metadata = deserialize_metadata(meta_str)?;

        // ── tensors ──
        let mut tensors = Vec::with_capacity(num_tensors.min(MAX_EAGER_CAPACITY));
        for _ in 0..num_tensors {
            // name
            let name_len = read_u32_le(reader)? as usize;
            let name_bytes = read_bytes_bounded(reader, name_len)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| CheckpointError::MetadataParse(e.to_string()))?;

            // shape
            let ndim = read_u32_le(reader)? as usize;
            let mut shape = Vec::with_capacity(ndim.min(MAX_EAGER_CAPACITY));
            for _ in 0..ndim {
                shape.push(read_u64_le(reader)?);
            }

            // data
            let data_len = read_u64_le(reader)? as usize;
            let mut data = Vec::with_capacity(data_len.min(MAX_EAGER_CAPACITY));
            for _ in 0..data_len {
                let mut buf = [0u8; 4];
                read_exact(reader, &mut buf)?;
                data.push(f32::from_le_bytes(buf));
            }

            // Cross-validate the shape/data invariant documented on
            // `CheckpointTensor::shape` ("product must equal data.len()")
            // *before* the tensor is accepted. `shape` and `data` are read as
            // two independent length-prefixed fields above, so a malformed or
            // adversarial checkpoint can otherwise construct a tensor whose
            // shape disagrees with its data length; that inconsistent tensor
            // would then flow — unvalidated — into `to_weight_tensor()` /
            // `WeightTensor::new` and on into shape-driven indexing such as
            // structured pruning, which would panic instead of surfacing a
            // typed error. Reject it here instead.
            // Mirror `CheckpointTensor::element_count()`'s convention: an
            // empty shape (no dimensions recorded) denotes zero elements,
            // not the empty product (1).
            let shape_product = if shape.is_empty() {
                Some(0u64)
            } else {
                shape
                    .iter()
                    .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            };
            let shape_matches_data = matches!(
                shape_product,
                Some(product) if product == data.len() as u64
            );
            if !shape_matches_data {
                return Err(CheckpointError::ShapeDataMismatch {
                    name,
                    shape,
                    data_len: data.len(),
                });
            }

            tensors.push(CheckpointTensor { name, shape, data });
        }

        Ok(Self {
            version,
            metadata,
            tensors,
        })
    }
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata serialization (no serde)
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize `metadata` as `{"key1":"val1","key2":"val2"}`.
///
/// Keys and values must not contain `"` or `\`; if they do, those characters
/// are escaped with `\` so the round-trip is still correct for typical
/// training metadata (step numbers, loss strings, etc.).
fn serialize_metadata(meta: &CheckpointMetadata) -> String {
    // Deterministic order for reproducibility.
    let mut pairs: Vec<(&String, &String)> = meta.iter().collect();
    pairs.sort_by_key(|(k, _)| k.as_str());

    let mut out = String::from('{');
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        push_escaped(&mut out, k);
        out.push_str("\":\"");
        push_escaped(&mut out, v);
        out.push('"');
    }
    out.push('}');
    out
}

/// Escape `"` → `\"` and `\` → `\\` within a JSON string value.
fn push_escaped(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            other => out.push(other),
        }
    }
}

/// Deserialize a simple `{"key":"val",...}` JSON object.
///
/// This is a purposely minimal state machine — it does not handle nested
/// objects or arrays.  Its sole purpose is to decode the metadata written by
/// [`serialize_metadata`].
fn deserialize_metadata(s: &str) -> Result<CheckpointMetadata, CheckpointError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(CheckpointMetadata::new());
    }

    // Allow both `{}` and plain empty strings as "no metadata".
    if s == "{}" {
        return Ok(CheckpointMetadata::new());
    }

    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') || bytes.last() != Some(&b'}') {
        return Err(CheckpointError::MetadataParse(format!(
            "expected JSON object, got: {s}"
        )));
    }

    // Strip outer braces.
    let inner = &s[1..s.len() - 1];
    let mut map = CheckpointMetadata::new();

    if inner.trim().is_empty() {
        return Ok(map);
    }

    // Parse "key":"value" pairs separated by commas.
    // We use a simple char-by-char scanner that handles `\"` escapes.
    let chars: Vec<char> = inner.chars().collect();
    let mut pos = 0usize;

    loop {
        // Skip optional whitespace and commas between pairs.
        while pos < chars.len() && (chars[pos] == ',' || chars[pos].is_whitespace()) {
            pos += 1;
        }
        if pos >= chars.len() {
            break;
        }

        // Expect opening `"` of key.
        if chars[pos] != '"' {
            return Err(CheckpointError::MetadataParse(format!(
                "expected '\"' at position {pos}, got '{}'",
                chars[pos]
            )));
        }
        pos += 1;

        let (key, new_pos) = parse_json_string(&chars, pos)?;
        pos = new_pos;

        // Expect `:`
        skip_ws(&chars, &mut pos);
        if pos >= chars.len() || chars[pos] != ':' {
            return Err(CheckpointError::MetadataParse(format!(
                "expected ':' after key '{key}'"
            )));
        }
        pos += 1;
        skip_ws(&chars, &mut pos);

        // Expect opening `"` of value.
        if pos >= chars.len() || chars[pos] != '"' {
            return Err(CheckpointError::MetadataParse(format!(
                "expected '\"' for value of key '{key}'"
            )));
        }
        pos += 1;

        let (value, new_pos) = parse_json_string(&chars, pos)?;
        pos = new_pos;

        map.insert(key, value);
    }

    Ok(map)
}

/// Parse a JSON string body starting at `pos` (after the opening `"`).
///
/// Returns `(string, position_after_closing_quote)`.
fn parse_json_string(chars: &[char], mut pos: usize) -> Result<(String, usize), CheckpointError> {
    let mut s = String::new();
    while pos < chars.len() {
        match chars[pos] {
            '"' => {
                pos += 1; // consume closing quote
                return Ok((s, pos));
            }
            '\\' => {
                pos += 1;
                if pos >= chars.len() {
                    return Err(CheckpointError::MetadataParse(
                        "unexpected end after backslash".into(),
                    ));
                }
                match chars[pos] {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    other => {
                        return Err(CheckpointError::MetadataParse(format!(
                            "unknown escape '\\{other}'"
                        )))
                    }
                }
                pos += 1;
            }
            ch => {
                s.push(ch);
                pos += 1;
            }
        }
    }
    Err(CheckpointError::MetadataParse("unterminated string".into()))
}

/// Advance `pos` past ASCII whitespace.
fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level I/O helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Upper bound on how much capacity [`Checkpoint::read_from`] eagerly
/// pre-allocates for any single length-prefixed field, regardless of the
/// value the untrusted stream claims. Chosen generously (64 Ki
/// elements/bytes — a few hundred KiB at most per field) so that ordinary
/// checkpoints never grow their `Vec`s via the slower incremental path,
/// while a corrupted or adversarial header cannot force a large allocation
/// before the corresponding bytes have actually been observed in the
/// stream.
const MAX_EAGER_CAPACITY: usize = 1 << 16;

/// Read exactly `len` bytes from `reader` into a freshly allocated `Vec<u8>`,
/// without ever pre-reserving more than [`MAX_EAGER_CAPACITY`] bytes of
/// capacity ahead of what has already been confirmed present in the stream.
///
/// Bytes are consumed in bounded chunks so that a `len` far larger than the
/// remaining input (e.g. a corrupted or adversarial length field) fails via
/// the normal [`CheckpointError::TruncatedData`] path — from [`read_exact`]
/// hitting EOF — rather than triggering an immediate attacker-controlled
/// allocation of `len` bytes.
fn read_bytes_bounded<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, CheckpointError> {
    const CHUNK_LEN: usize = 8192;
    let mut out = Vec::with_capacity(len.min(MAX_EAGER_CAPACITY));
    let mut remaining = len;
    let mut chunk = [0u8; CHUNK_LEN];
    while remaining > 0 {
        let take = remaining.min(CHUNK_LEN);
        read_exact(reader, &mut chunk[..take])?;
        out.extend_from_slice(&chunk[..take]);
        remaining -= take;
    }
    Ok(out)
}

fn write_u32_le<W: Write>(w: &mut W, v: u32) -> Result<(), CheckpointError> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn write_u64_le<W: Write>(w: &mut W, v: u64) -> Result<(), CheckpointError> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), CheckpointError> {
    let expected = buf.len();
    let mut total_read = 0usize;
    while total_read < expected {
        match r.read(&mut buf[total_read..]) {
            Ok(0) => {
                return Err(CheckpointError::TruncatedData {
                    expected,
                    got: total_read,
                })
            }
            Ok(n) => total_read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(CheckpointError::Io(e)),
        }
    }
    Ok(())
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, CheckpointError> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, CheckpointError> {
    let mut buf = [0u8; 8];
    read_exact(r, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during checkpoint I/O.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// Wraps any [`std::io::Error`] from the underlying reader/writer.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not begin with the expected `b"OXCK"` magic bytes.
    #[error("invalid magic bytes: expected OXCK, got {0:?}")]
    InvalidMagic(Vec<u8>),

    /// The checkpoint was written with a version this library cannot read.
    #[error("unsupported checkpoint version: {0}")]
    UnsupportedVersion(u32),

    /// The metadata block could not be parsed as a key-value JSON object.
    #[error("metadata parse error: {0}")]
    MetadataParse(String),

    /// The byte stream ended before the expected number of bytes were read.
    #[error("truncated data: expected {expected} bytes, got {got}")]
    TruncatedData { expected: usize, got: usize },

    /// A tensor name exceeds 65 535 bytes (the 16-bit length field limit).
    #[error("tensor name too long: {0} bytes (max 65535)")]
    NameTooLong(usize),

    /// A tensor's declared `shape` and its actual `data` length disagree
    /// (`shape.iter().product() != data.len()`), or the shape's element
    /// product overflows `u64`. This indicates a corrupted or adversarial
    /// checkpoint file — accepting the tensor anyway would let a downstream
    /// shape-driven consumer (e.g. structured pruning) index past the end of
    /// `data` and panic.
    #[error(
        "tensor '{name}' shape {shape:?} does not match data length {data_len} \
         (product of shape must equal data.len())"
    )]
    ShapeDataMismatch {
        /// Name of the offending tensor.
        name: String,
        /// The declared shape that produced the mismatch.
        shape: Vec<u64>,
        /// The actual number of `f32` elements read for this tensor.
        data_len: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── read_from resource-limit regression tests ──────────────────────────
    //
    // `Checkpoint::read_from` used to allocate directly from untrusted
    // length fields (`Vec::with_capacity(data_len)` etc.) before validating
    // that many bytes were actually present in the stream, so a tiny
    // corrupted/adversarial header claiming an enormous `data_len` (or
    // `meta_len` / `ndim` / `num_tensors`) could force a multi-gigabyte
    // allocation attempt. These tests build short headers that claim huge
    // lengths and confirm read_from fails fast with `TruncatedData` instead
    // of attempting to honor the claimed size.

    /// Build a minimal valid `OXCK` header (magic + version + flags) as bytes.
    fn header_bytes() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"OXCK");
        b.extend_from_slice(&1u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // flags
        b
    }

    #[test]
    fn read_from_huge_meta_len_fails_fast_not_oom() {
        let mut bytes = header_bytes();
        bytes.extend_from_slice(&0u64.to_le_bytes()); // num_tensors = 0
                                                      // meta_len claims ~4 GiB but the stream has no more bytes.
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        // No metadata bytes actually follow.

        let result = Checkpoint::read_from(&mut bytes.as_slice());
        assert!(
            matches!(result, Err(CheckpointError::TruncatedData { .. })),
            "expected TruncatedData for a huge meta_len with no backing bytes, got {result:?}"
        );
    }

    #[test]
    fn read_from_huge_data_len_fails_fast_not_oom() {
        let mut bytes = header_bytes();
        bytes.extend_from_slice(&1u64.to_le_bytes()); // num_tensors = 1
        bytes.extend_from_slice(&0u32.to_le_bytes()); // meta_len = 0 (no metadata)

        // Tensor 0: short valid name, no shape dims, but an enormous data_len.
        let name = b"x";
        bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // ndim = 0
                                                      // data_len claims ~4.6e18 f32 elements (~18 exabytes) but nothing follows.
        bytes.extend_from_slice(&(u64::MAX / 4).to_le_bytes());

        let result = Checkpoint::read_from(&mut bytes.as_slice());
        assert!(
            matches!(result, Err(CheckpointError::TruncatedData { .. })),
            "expected TruncatedData for a huge data_len with no backing bytes, got {result:?}"
        );
    }

    #[test]
    fn read_from_huge_ndim_fails_fast_not_oom() {
        let mut bytes = header_bytes();
        bytes.extend_from_slice(&1u64.to_le_bytes()); // num_tensors = 1
        bytes.extend_from_slice(&0u32.to_le_bytes()); // meta_len = 0

        let name = b"x";
        bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name);
        // ndim claims ~4 billion dimensions but no shape data follows.
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());

        let result = Checkpoint::read_from(&mut bytes.as_slice());
        assert!(
            matches!(result, Err(CheckpointError::TruncatedData { .. })),
            "expected TruncatedData for a huge ndim with no backing bytes, got {result:?}"
        );
    }

    #[test]
    fn read_from_huge_num_tensors_fails_fast_not_oom() {
        let mut bytes = header_bytes();
        // num_tensors claims ~1.8e19 tensors but the stream ends immediately after.
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // meta_len = 0

        let result = Checkpoint::read_from(&mut bytes.as_slice());
        assert!(
            matches!(result, Err(CheckpointError::TruncatedData { .. })),
            "expected TruncatedData for a huge num_tensors with no backing bytes, got {result:?}"
        );
    }

    #[test]
    fn read_bytes_bounded_reads_exact_content() {
        let payload = b"hello checkpoint world".to_vec();
        let mut reader = payload.as_slice();
        let out = read_bytes_bounded(&mut reader, payload.len()).expect("read_bytes_bounded");
        assert_eq!(out, payload);
    }

    #[test]
    fn read_bytes_bounded_zero_length_is_empty() {
        let mut reader: &[u8] = &[];
        let out = read_bytes_bounded(&mut reader, 0).expect("read_bytes_bounded");
        assert!(out.is_empty());
    }
}
