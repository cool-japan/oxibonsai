//! Streaming GGUF reader for progressive parsing.
//!
//! This module provides a state-machine-based parser that can consume GGUF data
//! incrementally as it arrives (e.g., from a network download), without requiring
//! the full file to be present in memory.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxibonsai_core::gguf::streaming::GgufStreamParser;
//!
//! let mut parser = GgufStreamParser::new();
//! // Feed data as it arrives:
//! // let consumed = parser.feed(&chunk)?;
//! // Check completion:
//! // if parser.is_complete() { let result = parser.finish()?; }
//! ```

use crate::error::BonsaiError;
use crate::gguf::types::{GgufTensorType, GgufValueType};

/// GGUF magic number: "GGUF" in little-endian = 0x46554747.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF header size: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes.
const HEADER_SIZE: usize = 24;

/// Maximum string length accepted (256 MB).
const MAX_STRING_LEN: u64 = 256 * 1024 * 1024;

/// Maximum array element count accepted (16M entries).
const MAX_ARRAY_COUNT: u64 = 16 * 1024 * 1024;

/// Maximum tensor dimensions.
///
/// `GGML_MAX_DIMS` is 4 (matches `StreamedTensorInfo::dims: [u64; 4]`), so a
/// GGUF declaring more than 4 dimensions is invalid input, not a real model
/// shape a `[u64; 4]` could hold anyway. Reject it explicitly (see
/// `try_parse_one_tensor_info`) instead of the previous behaviour: keeping
/// the too-large `n_dims` (validated only against a stale cap of 1024)
/// while silently dropping dimensions `4..n_dims` from `dims`.
const MAX_TENSOR_DIMS: u32 = 4;

/// Maximum nesting depth for `Array`-of-`Array` metadata values.
///
/// Mirrors the cap in `metadata.rs`: each nesting level costs only 12 bytes
/// on disk, so without a limit a small crafted stream can drive
/// `try_read_value` tens of thousands of stack frames deep and abort the
/// process. No legitimate GGUF file nests arrays anywhere close to this
/// deep.
const MAX_ARRAY_NESTING_DEPTH: u32 = 32;

/// Bound on the eager `Vec` capacity reservation for an `Array` value.
///
/// Mirrors the identical cap in `metadata.rs`: the declared element `count`
/// is attacker-controlled and only bounded against [`MAX_ARRAY_COUNT`] (16
/// M), so reserving `count` elements of capacity up front — before a single
/// element has actually arrived over the wire — defeats this module's own
/// purpose of not requiring the full data to be present in memory.
/// Reserving only a small bounded amount and letting the `Vec` grow
/// amortized as elements actually arrive keeps peak allocation proportional
/// to real progress instead of a declared-but-undelivered count.
const ARRAY_EAGER_RESERVE_CAP: usize = 4096;

/// Default alignment for tensor data in GGUF files (32 bytes).
const DEFAULT_ALIGNMENT: usize = 32;

/// State machine for progressive GGUF parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamState {
    /// Waiting for the 24-byte header.
    ReadingHeader,
    /// Parsing metadata key-value pairs; `remaining` entries left.
    ReadingMetadata { remaining: u64 },
    /// Parsing tensor info entries; `remaining` entries left.
    ReadingTensorInfo { remaining: u64 },
    /// All metadata and tensor info parsed; tensor data follows.
    ReadingTensorData,
    /// Parsing is fully complete.
    Complete,
}

/// A metadata value from the streaming parser.
///
/// This mirrors `MetadataValue` but is self-contained so the streaming module
/// does not depend on the cursor-based metadata parser.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// A short, stable name for `value`'s underlying GGUF type.
///
/// Mirrors `MetadataValue::type_name` (`metadata.rs`) for the streaming
/// module's self-contained [`GgufValue`], used to build a precise
/// type-mismatch message for [`GgufStreamParser::finish`]'s `finalize` step.
fn gguf_value_type_name(value: &GgufValue) -> &'static str {
    match value {
        GgufValue::Uint8(_) => "Uint8",
        GgufValue::Int8(_) => "Int8",
        GgufValue::Uint16(_) => "Uint16",
        GgufValue::Int16(_) => "Int16",
        GgufValue::Uint32(_) => "Uint32",
        GgufValue::Int32(_) => "Int32",
        GgufValue::Float32(_) => "Float32",
        GgufValue::Bool(_) => "Bool",
        GgufValue::String(_) => "String",
        GgufValue::Array(_) => "Array",
        GgufValue::Uint64(_) => "Uint64",
        GgufValue::Int64(_) => "Int64",
        GgufValue::Float64(_) => "Float64",
    }
}

/// Accumulated parse result from streaming.
#[derive(Debug, Clone)]
pub struct StreamedGguf {
    /// GGUF format version.
    pub version: u32,
    /// Parsed metadata key-value pairs (in order).
    pub metadata: Vec<(String, GgufValue)>,
    /// Parsed tensor info entries (in order).
    pub tensor_infos: Vec<StreamedTensorInfo>,
    /// Byte offset where tensor data begins (aligned).
    pub data_offset: u64,
}

/// Tensor information from the streaming parser.
#[derive(Debug, Clone)]
pub struct StreamedTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Dimensions (up to 4; unused dims are 0).
    pub dims: [u64; 4],
    /// Quantization / data type.
    pub tensor_type: GgufTensorType,
    /// Byte offset within the tensor data section.
    pub offset: u64,
}

/// In-progress state for a top-level metadata `Array` value that spans
/// multiple [`feed`](GgufStreamParser::feed) calls.
///
/// Without this, a metadata value's `Array` branch had to be re-parsed
/// from its own start (buffer offset 0) on every `feed()` call until the
/// whole array finally fit in the buffer at once — `O(elements)` calls
/// each redoing up to `O(elements)` of work, i.e. quadratic in the element
/// count (measured: a real 248320-entry `tokenizer.ggml.tokens` array
/// parses in ~159 ms in one shot; re-parsing it from scratch on every
/// chunk when fed incrementally was estimated at ~1.6 s total — roughly an
/// order of magnitude worse for the identical bytes, purely from the
/// restart-at-offset-0 behaviour). Stashing the already-parsed `values`
/// and the remaining element count here lets
/// [`GgufStreamParser::resume_partial_array`] treat buffer offset 0 as
/// always being the start of the *next* unparsed element, so each byte of
/// the array is parsed exactly once across however many `feed()` calls it
/// takes to arrive.
#[derive(Debug)]
struct PartialArray {
    /// The metadata key this array is the value of.
    key: String,
    /// Element type shared by every element in the array.
    elem_type: GgufValueType,
    /// Elements not yet parsed.
    remaining: u64,
    /// Elements parsed so far, in order.
    values: Vec<GgufValue>,
}

/// Streaming GGUF parser.
///
/// Feed bytes progressively via [`feed`](Self::feed). The parser buffers
/// incomplete data internally and advances through [`StreamState`] stages
/// as enough bytes accumulate.
#[derive(Debug)]
pub struct GgufStreamParser {
    state: StreamState,
    buffer: Vec<u8>,
    result: StreamedGguf,
    bytes_consumed: u64,
    // Cached header counts for progress estimation
    total_metadata: u64,
    total_tensors: u64,
    // Track keys/names seen so far so a duplicate can be rejected as a hard
    // parse error instead of silently accumulating twice in `result`.
    seen_metadata_keys: std::collections::HashSet<String>,
    seen_tensor_names: std::collections::HashSet<String>,
    // An in-progress top-level metadata `Array` value, when the parser is
    // mid-way through one across multiple `feed()` calls.
    partial_array: Option<PartialArray>,
}

impl GgufStreamParser {
    /// Create a new streaming parser in the initial state.
    pub fn new() -> Self {
        Self {
            state: StreamState::ReadingHeader,
            buffer: Vec::with_capacity(4096),
            result: StreamedGguf {
                version: 0,
                metadata: Vec::new(),
                tensor_infos: Vec::new(),
                data_offset: 0,
            },
            bytes_consumed: 0,
            total_metadata: 0,
            total_tensors: 0,
            seen_metadata_keys: std::collections::HashSet::new(),
            seen_tensor_names: std::collections::HashSet::new(),
            partial_array: None,
        }
    }

    /// Feed bytes into the parser. Returns the number of bytes consumed from `data`.
    ///
    /// `data` is always appended to the parser's internal buffer before any
    /// parsing is attempted, so bytes are never lost regardless of the
    /// return value — the return value reports whether *this call* made
    /// forward progress, not how much of `data` is still owed back to the
    /// caller.
    ///
    /// If nothing newly available in the buffer was enough to complete
    /// another header, metadata entry, tensor-info entry, or array element,
    /// this returns `Ok(0)`; call again with more data when that happens.
    /// Otherwise it returns `Ok(data.len())`.
    pub fn feed(&mut self, data: &[u8]) -> Result<usize, BonsaiError> {
        if data.is_empty() {
            return Ok(0);
        }

        // Append new data to internal buffer
        self.buffer.extend_from_slice(data);
        let input_len = data.len();

        // Whether this call advanced the parser at all. Every branch below
        // that changes parser state (a completed parse, a partial-array
        // element parsed, or a state transition) sets this; a chunk that
        // cannot even complete one more step of any kind leaves it `false`,
        // matching the `Ok(0)` contract documented above.
        let mut made_progress = false;

        // Process as much as possible from the buffer
        loop {
            match &self.state {
                StreamState::ReadingHeader => {
                    if self.try_parse_header()? {
                        made_progress = true;
                    } else {
                        break;
                    }
                }
                StreamState::ReadingMetadata { remaining } => {
                    if *remaining == 0 {
                        self.transition_to_tensor_info();
                        made_progress = true;
                        continue;
                    }
                    if self.try_parse_one_metadata()? {
                        made_progress = true;
                    } else {
                        break;
                    }
                }
                StreamState::ReadingTensorInfo { remaining } => {
                    if *remaining == 0 {
                        self.finalize()?;
                        made_progress = true;
                        break;
                    }
                    if self.try_parse_one_tensor_info()? {
                        made_progress = true;
                    } else {
                        break;
                    }
                }
                StreamState::ReadingTensorData | StreamState::Complete => {
                    break;
                }
            }
        }

        Ok(if made_progress { input_len } else { 0 })
    }

    /// Check if parsing is complete (all metadata + tensor info parsed).
    pub fn is_complete(&self) -> bool {
        matches!(
            self.state,
            StreamState::ReadingTensorData | StreamState::Complete
        )
    }

    /// Get current parse state.
    pub fn state(&self) -> &StreamState {
        &self.state
    }

    /// Get total bytes consumed so far.
    pub fn bytes_consumed(&self) -> u64 {
        self.bytes_consumed
    }

    /// Take the final result. Only valid after [`is_complete`](Self::is_complete) returns true.
    pub fn finish(self) -> Result<StreamedGguf, BonsaiError> {
        if !self.is_complete() {
            return Err(BonsaiError::UnexpectedEof {
                offset: self.bytes_consumed,
            });
        }
        Ok(self.result)
    }

    /// Estimated progress as a fraction in `[0.0, 1.0]`.
    ///
    /// Before the header is parsed, progress is based on bytes towards the 24-byte header.
    /// After the header, progress is based on how many metadata + tensor info entries
    /// have been parsed out of the total expected.
    pub fn progress(&self) -> f32 {
        match &self.state {
            StreamState::ReadingHeader => {
                // Progress towards header completion
                let have = self.buffer.len().min(HEADER_SIZE) as f32;
                (have / HEADER_SIZE as f32) * 0.1 // header is ~10% of progress
            }
            StreamState::ReadingMetadata { remaining } => {
                let total = self.total_metadata + self.total_tensors;
                if total == 0 {
                    return 0.5;
                }
                let done = self.total_metadata - remaining;
                0.1 + (done as f32 / total as f32) * 0.9
            }
            StreamState::ReadingTensorInfo { remaining } => {
                let total = self.total_metadata + self.total_tensors;
                if total == 0 {
                    return 0.9;
                }
                let done = self.total_metadata + (self.total_tensors - remaining);
                0.1 + (done as f32 / total as f32) * 0.9
            }
            StreamState::ReadingTensorData | StreamState::Complete => 1.0,
        }
    }

    // ---- Internal parsing methods ----

    /// Try to parse the 24-byte header from the buffer.
    /// Returns true if successful (state advanced), false if not enough data.
    fn try_parse_header(&mut self) -> Result<bool, BonsaiError> {
        if self.buffer.len() < HEADER_SIZE {
            return Ok(false);
        }

        let magic = read_u32_le(&self.buffer, 0);
        if magic != GGUF_MAGIC {
            return Err(BonsaiError::InvalidMagic { magic });
        }

        let version = read_u32_le(&self.buffer, 4);
        if version != 2 && version != 3 {
            return Err(BonsaiError::UnsupportedVersion { version });
        }

        let tensor_count = read_u64_le(&self.buffer, 8);
        let metadata_kv_count = read_u64_le(&self.buffer, 16);

        self.result.version = version;
        self.total_metadata = metadata_kv_count;
        self.total_tensors = tensor_count;
        self.bytes_consumed += HEADER_SIZE as u64;

        // Remove consumed header bytes from buffer
        self.buffer.drain(..HEADER_SIZE);

        self.state = StreamState::ReadingMetadata {
            remaining: metadata_kv_count,
        };
        Ok(true)
    }

    /// Try to parse one metadata KV entry from the buffer.
    ///
    /// Returns `Ok(true)` if this call advanced the parser at all — either
    /// a whole KV entry completed, or (for a top-level `Array` value spread
    /// across multiple `feed()` calls) at least one more element of an
    /// already-started array was parsed. Returns `Ok(false)` only when
    /// nothing in the buffer was enough to make any further progress.
    fn try_parse_one_metadata(&mut self) -> Result<bool, BonsaiError> {
        // Resume an in-progress top-level `Array` value first: buffer
        // offset 0 is always the start of its next unparsed element (see
        // `resume_partial_array`), so this never re-scans a byte an
        // earlier `feed()` call already consumed.
        if let Some(partial) = self.partial_array.take() {
            return self.resume_partial_array(partial);
        }

        let mut pos = 0;

        // Parse key string
        let key = match try_read_gguf_string(&self.buffer, pos)? {
            Some((s, new_pos)) => {
                pos = new_pos;
                s
            }
            None => return Ok(false),
        };

        // Parse value type
        if pos + 4 > self.buffer.len() {
            return Ok(false);
        }
        let value_type_id = read_u32_le(&self.buffer, pos);
        let value_type = GgufValueType::from_id(value_type_id)?;
        pos += 4;

        if value_type != GgufValueType::Array {
            // Scalar/string value: bounded size (at most one chunked
            // string read), parsed and drained in one shot exactly as
            // before — only a top-level `Array` needs the resumable path.
            let (value, new_pos) = match try_read_value(&self.buffer, pos, value_type, 0)? {
                Some(v) => v,
                None => return Ok(false),
            };
            pos = new_pos;

            // A duplicate key would otherwise silently accumulate a second
            // entry in `result.metadata` with no error or warning — reject
            // it as a hard parse error instead (mirrors the batch
            // `MetadataStore` parser in `metadata.rs`).
            if !self.seen_metadata_keys.insert(key.clone()) {
                return Err(BonsaiError::InvalidMetadata {
                    key,
                    reason: "duplicate metadata key".to_string(),
                });
            }

            self.bytes_consumed += pos as u64;
            self.buffer.drain(..pos);
            self.result.metadata.push((key, value));

            if let StreamState::ReadingMetadata { remaining } = &mut self.state {
                *remaining -= 1;
            }

            return Ok(true);
        }

        // Top-level `Array` value: read just its 12-byte header
        // (`elem_type: u32`, `count: u64`) here, then hand off to the
        // resumable element-by-element parser so a huge array (e.g. the
        // real 248320-entry `tokenizer.ggml.tokens`) is never re-scanned
        // from its own start on every `feed()` call.
        if pos + 12 > self.buffer.len() {
            return Ok(false);
        }
        let elem_type_id = read_u32_le(&self.buffer, pos);
        let elem_type = GgufValueType::from_id(elem_type_id)?;
        let count = read_u64_le(&self.buffer, pos + 4);
        if count > MAX_ARRAY_COUNT {
            return Err(BonsaiError::InvalidMetadata {
                key,
                reason: format!("array count too large: {count}"),
            });
        }
        pos += 12;
        self.bytes_consumed += pos as u64;
        self.buffer.drain(..pos);

        let partial = PartialArray {
            key,
            elem_type,
            remaining: count,
            values: Vec::with_capacity((count as usize).min(ARRAY_EAGER_RESERVE_CAP)),
        };
        self.resume_partial_array(partial)
    }

    /// Resume parsing a top-level metadata `Array` value's elements.
    ///
    /// Buffer offset 0 is always the start of the next unparsed element:
    /// this walks forward through as many complete elements as are
    /// currently buffered with a single moving cursor (no byte is ever
    /// re-read), then issues one `drain` for everything this call
    /// consumed — one shift of the buffer's tail per call, not one per
    /// element. This is what makes an array spanning many `feed()` calls
    /// cost `O(elements)` total instead of the previous restart-at-offset-0
    /// behaviour's `O(elements)` calls each redoing up to `O(elements)` of
    /// work.
    ///
    /// Returns `Ok(true)` if at least one more element was parsed this
    /// call (whether or not the array finished), `Ok(false)` if the buffer
    /// did not contain even one more complete element.
    fn resume_partial_array(&mut self, mut partial: PartialArray) -> Result<bool, BonsaiError> {
        let mut pos = 0usize;

        while partial.remaining > 0 {
            match try_read_value(&self.buffer, pos, partial.elem_type, 1)? {
                Some((v, new_pos)) => {
                    partial.values.push(v);
                    partial.remaining -= 1;
                    pos = new_pos;
                }
                None => break,
            }
        }

        let made_progress = pos > 0;
        if made_progress {
            self.bytes_consumed += pos as u64;
            self.buffer.drain(..pos);
        }

        if partial.remaining > 0 {
            // Buffer exhausted before the array finished: stash progress
            // and wait for the next `feed()` call to resume from here.
            self.partial_array = Some(partial);
            return Ok(made_progress);
        }

        // All elements parsed: finish the metadata KV entry exactly as the
        // scalar/string path does (duplicate-key check, push, decrement).
        let PartialArray { key, values, .. } = partial;
        if !self.seen_metadata_keys.insert(key.clone()) {
            return Err(BonsaiError::InvalidMetadata {
                key,
                reason: "duplicate metadata key".to_string(),
            });
        }
        self.result.metadata.push((key, GgufValue::Array(values)));
        if let StreamState::ReadingMetadata { remaining } = &mut self.state {
            *remaining -= 1;
        }
        Ok(true)
    }

    /// Transition from metadata to tensor info reading.
    fn transition_to_tensor_info(&mut self) {
        self.state = StreamState::ReadingTensorInfo {
            remaining: self.total_tensors,
        };
    }

    /// Try to parse one tensor info entry from the buffer.
    /// Returns true if successful, false if not enough data.
    fn try_parse_one_tensor_info(&mut self) -> Result<bool, BonsaiError> {
        let mut pos = 0;

        // Parse name
        let name = match try_read_gguf_string(&self.buffer, pos)? {
            Some((s, new_pos)) => {
                pos = new_pos;
                s
            }
            None => return Ok(false),
        };

        // Parse n_dims (u32)
        if pos + 4 > self.buffer.len() {
            return Ok(false);
        }
        let n_dims = read_u32_le(&self.buffer, pos);
        pos += 4;

        if n_dims > MAX_TENSOR_DIMS {
            return Err(BonsaiError::InvalidMetadata {
                key: name,
                // Wording aligned with `tensor_info.rs`'s identical check
                // (core-gguf-16 / wave-2.5 addendum item 4): the two parsers
                // used to disagree on the reason string for the exact same
                // invalid input, which meant a cross-parser parity test
                // could only assert the error *kind*, not its message.
                reason: format!(
                    "tensor has {n_dims} dimensions; GGML_MAX_DIMS is {MAX_TENSOR_DIMS}"
                ),
            });
        }

        // Parse dims (n_dims * u64). `n_dims` is guaranteed <= MAX_TENSOR_DIMS
        // (4) by the check above, so every declared dimension has a slot in
        // `dims` — this no longer truncates anything (the `n_dims > 4` case
        // that used to be silently dropped by a `.min(4)` here is now
        // rejected outright before reaching this line).
        let dims_bytes = n_dims as usize * 8;
        if pos + dims_bytes > self.buffer.len() {
            return Ok(false);
        }
        let mut dims = [0u64; 4];
        for (i, dim) in dims.iter_mut().enumerate().take(n_dims as usize) {
            *dim = read_u64_le(&self.buffer, pos + i * 8);
        }
        pos += dims_bytes;

        // Parse tensor type (u32)
        if pos + 4 > self.buffer.len() {
            return Ok(false);
        }
        let type_id = read_u32_le(&self.buffer, pos);
        let tensor_type = GgufTensorType::from_id(type_id)?;
        pos += 4;

        // Parse offset (u64)
        if pos + 8 > self.buffer.len() {
            return Ok(false);
        }
        let offset = read_u64_le(&self.buffer, pos);
        pos += 8;

        // A duplicate tensor name would otherwise silently accumulate a
        // second `StreamedTensorInfo` entry with no error or warning —
        // reject it as a hard parse error instead (mirrors the batch
        // `TensorStore` parser in `tensor_info.rs`).
        if !self.seen_tensor_names.insert(name.clone()) {
            return Err(BonsaiError::InvalidMetadata {
                key: name,
                reason: "duplicate tensor name".to_string(),
            });
        }

        self.bytes_consumed += pos as u64;
        self.buffer.drain(..pos);

        self.result.tensor_infos.push(StreamedTensorInfo {
            name,
            n_dims,
            dims,
            tensor_type,
            offset,
        });

        // Decrement remaining
        if let StreamState::ReadingTensorInfo { remaining } = &mut self.state {
            *remaining -= 1;
        }

        Ok(true)
    }

    /// Finalize parsing: compute data offset with alignment and transition to complete state.
    ///
    /// Returns an error if `general.alignment` was present but is zero or not
    /// a power of two, rather than silently mis-locating the tensor data
    /// section (see [`AlignmentError`](BonsaiError::AlignmentError)).
    ///
    /// A `general.alignment` present with a type *other than* `Uint32` is a
    /// hard [`BonsaiError::InvalidMetadata`], matching `GgufFile::parse`
    /// (`reader.rs`) exactly (core-gguf-15 / wave-2.5 integration addendum,
    /// item 3): this used to silently substitute the 32-byte default for a
    /// spec-invalid (e.g. `Uint64`-spelled) alignment instead of rejecting
    /// the file, which meant the batch and streaming parsers could disagree
    /// on `data_offset` for the identical bytes.
    fn finalize(&mut self) -> Result<(), BonsaiError> {
        // Check for alignment override in metadata. Only a genuinely
        // *absent* key defaults; a present-but-wrongly-typed one is a hard
        // error, never a silent fallback to the default.
        let alignment_entry = self
            .result
            .metadata
            .iter()
            .find(|(k, _)| k == "general.alignment");
        let alignment = match alignment_entry {
            None => DEFAULT_ALIGNMENT,
            Some((_, GgufValue::Uint32(n))) => *n as usize,
            Some((_, other)) => {
                return Err(BonsaiError::InvalidMetadata {
                    key: "general.alignment".to_string(),
                    reason: format!(
                        "general.alignment must be stored as UINT32 per the GGUF spec; found {}",
                        gguf_value_type_name(other)
                    ),
                });
            }
        };

        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(BonsaiError::AlignmentError {
                expected: DEFAULT_ALIGNMENT,
                offset: self.bytes_consumed,
            });
        }

        let offset = self.bytes_consumed as usize;
        let aligned = (offset + alignment - 1) & !(alignment - 1);
        self.result.data_offset = aligned as u64;

        self.state = StreamState::ReadingTensorData;
        Ok(())
    }
}

impl Default for GgufStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Low-level buffer readers (no std::io dependency) ----

/// Read a little-endian u32 from a byte slice at the given offset.
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    let b = &buf[offset..offset + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a little-endian u64 from a byte slice at the given offset.
fn read_u64_le(buf: &[u8], offset: usize) -> u64 {
    let b = &buf[offset..offset + 8];
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Read a little-endian i8 from a byte slice at the given offset.
fn read_i8_le(buf: &[u8], offset: usize) -> i8 {
    buf[offset] as i8
}

/// Read a little-endian i16 from a byte slice at the given offset.
fn read_i16_le(buf: &[u8], offset: usize) -> i16 {
    let b = &buf[offset..offset + 2];
    i16::from_le_bytes([b[0], b[1]])
}

/// Read a little-endian u16 from a byte slice at the given offset.
fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    let b = &buf[offset..offset + 2];
    u16::from_le_bytes([b[0], b[1]])
}

/// Read a little-endian i32 from a byte slice at the given offset.
fn read_i32_le(buf: &[u8], offset: usize) -> i32 {
    let b = &buf[offset..offset + 4];
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a little-endian i64 from a byte slice at the given offset.
fn read_i64_le(buf: &[u8], offset: usize) -> i64 {
    let b = &buf[offset..offset + 8];
    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Read a little-endian f32 from a byte slice at the given offset.
fn read_f32_le(buf: &[u8], offset: usize) -> f32 {
    let b = &buf[offset..offset + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a little-endian f64 from a byte slice at the given offset.
fn read_f64_le(buf: &[u8], offset: usize) -> f64 {
    let b = &buf[offset..offset + 8];
    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Try to read a GGUF string from the buffer at `offset`.
/// Returns `Some((string, new_offset))` if enough data, `None` otherwise.
fn try_read_gguf_string(buf: &[u8], offset: usize) -> Result<Option<(String, usize)>, BonsaiError> {
    if offset + 8 > buf.len() {
        return Ok(None);
    }
    let len = read_u64_le(buf, offset);
    if len > MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString {
            offset: offset as u64,
        });
    }
    let str_end = offset + 8 + len as usize;
    if str_end > buf.len() {
        return Ok(None);
    }
    let s =
        std::str::from_utf8(&buf[offset + 8..str_end]).map_err(|_| BonsaiError::InvalidString {
            offset: offset as u64,
        })?;
    Ok(Some((s.to_string(), str_end)))
}

/// Try to read a typed GGUF value from the buffer at `offset`.
/// Returns `Some((value, new_offset))` if enough data, `None` otherwise.
///
/// `depth` tracks how many `Array` values enclose this call and is bounded
/// by [`MAX_ARRAY_NESTING_DEPTH`] so a maliciously nested `Array`-of-`Array`
/// chain fails cleanly instead of overflowing the stack.
fn try_read_value(
    buf: &[u8],
    offset: usize,
    value_type: GgufValueType,
    depth: u32,
) -> Result<Option<(GgufValue, usize)>, BonsaiError> {
    match value_type {
        GgufValueType::Uint8 => {
            if offset + 1 > buf.len() {
                return Ok(None);
            }
            Ok(Some((GgufValue::Uint8(buf[offset]), offset + 1)))
        }
        GgufValueType::Int8 => {
            if offset + 1 > buf.len() {
                return Ok(None);
            }
            Ok(Some((GgufValue::Int8(read_i8_le(buf, offset)), offset + 1)))
        }
        GgufValueType::Uint16 => {
            if offset + 2 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Uint16(read_u16_le(buf, offset)),
                offset + 2,
            )))
        }
        GgufValueType::Int16 => {
            if offset + 2 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Int16(read_i16_le(buf, offset)),
                offset + 2,
            )))
        }
        GgufValueType::Uint32 => {
            if offset + 4 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Uint32(read_u32_le(buf, offset)),
                offset + 4,
            )))
        }
        GgufValueType::Int32 => {
            if offset + 4 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Int32(read_i32_le(buf, offset)),
                offset + 4,
            )))
        }
        GgufValueType::Float32 => {
            if offset + 4 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Float32(read_f32_le(buf, offset)),
                offset + 4,
            )))
        }
        GgufValueType::Bool => {
            if offset + 1 > buf.len() {
                return Ok(None);
            }
            Ok(Some((GgufValue::Bool(buf[offset] != 0), offset + 1)))
        }
        GgufValueType::String => match try_read_gguf_string(buf, offset)? {
            Some((s, new_pos)) => Ok(Some((GgufValue::String(s), new_pos))),
            None => Ok(None),
        },
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
            // Need element type (u32) + count (u64) = 12 bytes minimum
            if offset + 12 > buf.len() {
                return Ok(None);
            }
            let elem_type_id = read_u32_le(buf, offset);
            let elem_type = GgufValueType::from_id(elem_type_id)?;
            let count = read_u64_le(buf, offset + 4);
            if count > MAX_ARRAY_COUNT {
                return Err(BonsaiError::InvalidMetadata {
                    key: String::new(),
                    reason: format!("array count too large: {count}"),
                });
            }

            let mut pos = offset + 12;
            let mut values = Vec::with_capacity((count as usize).min(ARRAY_EAGER_RESERVE_CAP));
            for _ in 0..count {
                match try_read_value(buf, pos, elem_type, next_depth)? {
                    Some((v, new_pos)) => {
                        values.push(v);
                        pos = new_pos;
                    }
                    None => return Ok(None),
                }
            }
            Ok(Some((GgufValue::Array(values), pos)))
        }
        GgufValueType::Uint64 => {
            if offset + 8 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Uint64(read_u64_le(buf, offset)),
                offset + 8,
            )))
        }
        GgufValueType::Int64 => {
            if offset + 8 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Int64(read_i64_le(buf, offset)),
                offset + 8,
            )))
        }
        GgufValueType::Float64 => {
            if offset + 8 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Float64(read_f64_le(buf, offset)),
                offset + 8,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_gguf_string(bytes: &mut Vec<u8>, s: &str) {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal GGUF byte stream with a single `general.alignment`
    /// (Uint32) metadata entry and zero tensors.
    fn gguf_bytes_with_alignment(alignment: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        push_gguf_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        bytes.extend_from_slice(&alignment.to_le_bytes());
        bytes
    }

    #[test]
    fn zero_alignment_returns_error_not_silent_zero_offset() {
        let mut parser = GgufStreamParser::new();
        let data = gguf_bytes_with_alignment(0);
        let result = parser.feed(&data);
        match result {
            Err(BonsaiError::AlignmentError { .. }) => {}
            other => panic!("expected AlignmentError for zero alignment, got: {other:?}"),
        }
    }

    #[test]
    fn non_power_of_two_alignment_returns_error() {
        let mut parser = GgufStreamParser::new();
        let data = gguf_bytes_with_alignment(3);
        let result = parser.feed(&data);
        match result {
            Err(BonsaiError::AlignmentError { .. }) => {}
            other => {
                panic!("expected AlignmentError for non-power-of-two alignment, got: {other:?}")
            }
        }
    }

    #[test]
    fn valid_power_of_two_alignment_completes_successfully() {
        let mut parser = GgufStreamParser::new();
        let data = gguf_bytes_with_alignment(64);
        parser.feed(&data).expect("valid alignment should parse");
        assert!(parser.is_complete());
        let result = parser.finish().expect("finish should succeed");
        assert_eq!(result.data_offset % 64, 0);
    }

    /// Build a minimal GGUF byte stream with a single `general.alignment`
    /// entry stored as `Uint64` — spec-invalid (llama.cpp rejects any
    /// non-`UINT32` spelling outright, `ggml/src/gguf.cpp:613-618`) — and
    /// zero tensors.
    fn gguf_bytes_with_non_uint32_alignment(alignment: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        push_gguf_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
        bytes.extend_from_slice(&alignment.to_le_bytes());
        bytes
    }

    /// core-gguf-15 / wave-2.5 integration addendum, item 3: a
    /// `general.alignment` present but spelled as anything other than
    /// `Uint32` must be a hard `InvalidMetadata`, matching `GgufFile::parse`
    /// (`reader.rs::batch_parser_rejects_a_non_uint32_alignment_spelling`)
    /// exactly, instead of the previous behaviour of silently falling back
    /// to the 32-byte default and accepting the file. The cross-parser
    /// parity test in `crates/oxibonsai-core/tests/gguf_validation.rs`
    /// (`streaming_parser_rejects_a_non_uint32_alignment_spelling_like_the_batch_parser`)
    /// now asserts that parity directly instead of pinning the old,
    /// obsolete silent-default behaviour it used to document.
    #[test]
    fn non_uint32_alignment_is_a_hard_error_not_a_silent_default() {
        let mut parser = GgufStreamParser::new();
        let data = gguf_bytes_with_non_uint32_alignment(64);
        match parser.feed(&data) {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "general.alignment");
                assert!(reason.contains("UINT32"), "reason: {reason}");
                assert!(reason.contains("Uint64"), "reason: {reason}");
            }
            other => panic!(
                "expected InvalidMetadata for a non-Uint32 general.alignment, got: {other:?}"
            ),
        }
    }

    /// A present-but-absent-of-override alignment (the key is missing
    /// entirely) must still default to 32, unaffected by the type-check
    /// added above — only a *present-but-wrongly-typed* key is now an
    /// error, never a genuinely absent one.
    #[test]
    fn absent_alignment_key_still_defaults_to_32() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0

        let mut parser = GgufStreamParser::new();
        parser
            .feed(&bytes)
            .expect("no general.alignment key must still parse");
        let result = parser.finish().expect("finish should succeed");
        assert_eq!(result.data_offset % 32, 0);
    }

    #[test]
    fn default_creates_new_parser() {
        let parser = GgufStreamParser::default();
        assert_eq!(*parser.state(), StreamState::ReadingHeader);
        assert_eq!(parser.bytes_consumed(), 0);
        assert!(!parser.is_complete());
    }

    /// Build a minimal GGUF byte stream with two metadata entries sharing
    /// the same key.
    fn gguf_bytes_with_duplicate_metadata_key() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count

        for value in [1u32, 2u32] {
            push_gguf_string(&mut bytes, "dup.key");
            bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn duplicate_metadata_key_returns_error_not_silent_last_wins() {
        let mut parser = GgufStreamParser::new();
        let data = gguf_bytes_with_duplicate_metadata_key();
        match parser.feed(&data) {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "dup.key");
                assert!(reason.contains("duplicate"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata for duplicate key, got: {other:?}"),
        }
    }

    /// Build a minimal GGUF byte stream with two tensor-info entries
    /// sharing the same name.
    fn gguf_bytes_with_duplicate_tensor_name() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        for offset in [0u64, 16u64] {
            push_gguf_string(&mut bytes, "dup.weight");
            bytes.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            bytes.extend_from_slice(&4u64.to_le_bytes()); // dims[0]
            bytes.extend_from_slice(&0u32.to_le_bytes()); // tensor type (F32)
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn duplicate_tensor_name_returns_error_not_silent_last_wins() {
        let mut parser = GgufStreamParser::new();
        let data = gguf_bytes_with_duplicate_tensor_name();
        match parser.feed(&data) {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "dup.weight");
                assert!(reason.contains("duplicate"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata for duplicate tensor name, got: {other:?}"),
        }
    }

    /// A chain of nested `Array` headers well beyond
    /// `MAX_ARRAY_NESTING_DEPTH`, each declaring `count = MAX_ARRAY_COUNT`
    /// (16 M) elements, from a stream prefix only a few hundred bytes long.
    /// Before the eager-allocation cap, recursing down through each level
    /// forced a `Vec::with_capacity(16M)` reservation purely from the
    /// declared count — before a single element arrived over the wire,
    /// defeating this module's purpose of not requiring the full data up
    /// front. Walking `MAX_ARRAY_NESTING_DEPTH` such levels (each now
    /// capped) must still complete and fail cleanly via the depth check
    /// (bounded memory, no OOM abort, no hang), not just eventually stall
    /// waiting for more data.
    #[test]
    fn nested_max_count_arrays_on_tiny_prefix_bounded_memory_clean_error() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        push_gguf_string(&mut bytes, "bomb.key");
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());

        // Well beyond MAX_ARRAY_NESTING_DEPTH so the depth check fires
        // deterministically (it runs before the insufficient-data check),
        // while still exercising a `Vec::with_capacity(MAX_ARRAY_COUNT)`
        // attempt at every level along the way down.
        let levels = (MAX_ARRAY_NESTING_DEPTH as usize) * 2;
        for _ in 0..levels {
            bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            bytes.extend_from_slice(&MAX_ARRAY_COUNT.to_le_bytes());
        }

        assert!(bytes.len() < 1024, "crafted input must stay tiny");

        let mut parser = GgufStreamParser::new();
        let result = parser.feed(&bytes);
        match result {
            Err(BonsaiError::InvalidMetadata { reason, .. }) => {
                assert!(reason.contains("nesting depth"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata (nesting depth), got: {other:?}"),
        }
    }

    /// A `feed()` call that supplies fewer than the 24 header bytes cannot
    /// make any progress: it must report `Ok(0)`, not claim the whole
    /// chunk was "consumed" as the pre-fix code unconditionally did.
    #[test]
    fn feed_partial_header_returns_zero_progress() {
        let mut parser = GgufStreamParser::new();
        let mut header = Vec::new();
        header.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        header.extend_from_slice(&3u32.to_le_bytes());
        // Only 8 of the 24 header bytes.
        let consumed = parser
            .feed(&header)
            .expect("partial header feed should not error");
        assert_eq!(
            consumed, 0,
            "a chunk that cannot complete the header must report Ok(0)"
        );
        assert_eq!(*parser.state(), StreamState::ReadingHeader);
    }

    /// A `feed()` call that lands entirely inside an in-progress array
    /// element's 8-byte string-length prefix — after an earlier element of
    /// the same array has already been parsed — must report `Ok(0)`: no
    /// element completed this call, even though the array as a whole is
    /// mid-flight. Exercises `resume_partial_array` returning
    /// `made_progress == false` with its state correctly stashed back into
    /// `partial_array` for the next call.
    #[test]
    fn feed_returns_zero_when_mid_array_chunk_cannot_complete_next_element() {
        let mut header = Vec::new();
        header.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        header.extend_from_slice(&3u32.to_le_bytes()); // version
        header.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        header.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        let mut array_header = Vec::new();
        push_gguf_string(&mut array_header, "k");
        array_header.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        array_header.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        array_header.extend_from_slice(&2u64.to_le_bytes()); // count = 2

        let mut element0 = Vec::new();
        push_gguf_string(&mut element0, "ab");

        let mut element1 = Vec::new();
        push_gguf_string(&mut element1, "cdefgh");

        let mut parser = GgufStreamParser::new();

        // Call 1: header + array header + element0, plus the first 3 of
        // element1's 8-byte length prefix. Real progress happens (header,
        // array header, and one whole element all parse), so this must
        // report the full chunk as consumed, not `Ok(0)`.
        let mut first = Vec::new();
        first.extend_from_slice(&header);
        first.extend_from_slice(&array_header);
        first.extend_from_slice(&element0);
        first.extend_from_slice(&element1[..3]);
        let consumed_first = parser.feed(&first).expect("first feed should not error");
        assert_eq!(
            consumed_first,
            first.len(),
            "real progress was made, so the whole chunk counts as consumed"
        );

        // Call 2: 2 more bytes of the same length prefix (5 of 8 now
        // buffered) — still not enough to complete even one more element.
        let consumed_second = parser
            .feed(&element1[3..5])
            .expect("second feed should not error");
        assert_eq!(
            consumed_second, 0,
            "a chunk landing inside an incomplete length prefix must report Ok(0)"
        );
        assert!(!parser.is_complete());

        // Call 3: the rest of element1 completes the array and the file.
        parser
            .feed(&element1[5..])
            .expect("third feed should not error");
        assert!(parser.is_complete());

        let result = parser.finish().expect("finish should succeed");
        match &result.metadata[0].1 {
            GgufValue::Array(arr) => {
                assert_eq!(arr.len(), 2);
                match (&arr[0], &arr[1]) {
                    (GgufValue::String(a), GgufValue::String(b)) => {
                        assert_eq!(a, "ab");
                        assert_eq!(b, "cdefgh");
                    }
                    other => panic!("expected two String elements, got: {other:?}"),
                }
            }
            other => panic!("expected Array, got: {other:?}"),
        }
    }

    /// A string array fed across many small, arbitrary-boundary `feed()`
    /// calls must parse to exactly the same result as a one-shot feed,
    /// exercising `resume_partial_array` directly at a scale a human can
    /// read (the O(n)-vs-O(n^2) property at real scale is checked by the
    /// integration test in `tests/metadata_accessors.rs`).
    #[test]
    fn string_array_spanning_many_small_feed_calls_parses_exactly() {
        let elements: Vec<String> = (0..40).map(|i| format!("tok_{i:04}")).collect();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        push_gguf_string(&mut bytes, "tokens");
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        bytes.extend_from_slice(&(elements.len() as u64).to_le_bytes());
        for e in &elements {
            push_gguf_string(&mut bytes, e);
        }

        let mut parser = GgufStreamParser::new();
        // Deliberately tiny, odd chunk size so chunk boundaries land inside
        // both a length prefix and a payload at various points.
        for chunk in bytes.chunks(3) {
            parser.feed(chunk).expect("chunked feed should not error");
        }
        assert!(parser.is_complete());

        let result = parser.finish().expect("finish should succeed");
        assert_eq!(result.metadata.len(), 1);
        match &result.metadata[0].1 {
            GgufValue::Array(arr) => {
                assert_eq!(arr.len(), elements.len());
                for (got, want) in arr.iter().zip(elements.iter()) {
                    match got {
                        GgufValue::String(s) => assert_eq!(s, want),
                        other => panic!("expected String element, got: {other:?}"),
                    }
                }
            }
            other => panic!("expected Array, got: {other:?}"),
        }
    }

    /// `GGML_MAX_DIMS` is 4; a tensor declaring 5 dimensions is invalid
    /// input and must be rejected outright as `InvalidMetadata`, not
    /// silently accepted with dimensions `4..n_dims` dropped from a
    /// `[u64; 4]` (see also the batch-parser half of this same fix, owned
    /// by a different package — noted in this package's deviations).
    #[test]
    fn tensor_with_more_than_four_dims_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        push_gguf_string(&mut bytes, "five_d.weight");
        bytes.extend_from_slice(&5u32.to_le_bytes()); // n_dims = 5
        for dim in [2u64, 3, 4, 5, 6] {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes()); // tensor type (F32)
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut parser = GgufStreamParser::new();
        match parser.feed(&bytes) {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "five_d.weight");
                assert!(reason.contains("GGML_MAX_DIMS"), "reason: {reason}");
                assert_eq!(
                    reason, "tensor has 5 dimensions; GGML_MAX_DIMS is 4",
                    "wording must match tensor_info.rs's identical check exactly"
                );
            }
            other => panic!("expected InvalidMetadata for a 5-D tensor, got: {other:?}"),
        }
    }

    /// Exactly 4 dimensions must still be accepted (the boundary the
    /// previous test rejects just above).
    #[test]
    fn tensor_with_exactly_four_dims_is_accepted() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        push_gguf_string(&mut bytes, "four_d.weight");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // n_dims = 4
        for dim in [2u64, 3, 4, 5] {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes()); // tensor type (F32)
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut parser = GgufStreamParser::new();
        parser.feed(&bytes).expect("a 4-D tensor should parse");
        let result = parser.finish().expect("finish should succeed");
        assert_eq!(result.tensor_infos[0].n_dims, 4);
        assert_eq!(result.tensor_infos[0].dims, [2, 3, 4, 5]);
    }

    /// An `Array` value declaring `count = 0` must still complete the
    /// metadata entry (as `GgufValue::Array(vec![])`) and decrement the
    /// outer `ReadingMetadata` remaining count, not stall forever. The
    /// `count == 0` case skips `resume_partial_array`'s `while` loop
    /// entirely and falls straight through to the completion branch — this
    /// pins that down so a future change to the loop's exit condition
    /// cannot silently turn it into a permanent stall (parser stuck at
    /// `is_complete() == false` with no error surfaced anywhere, only
    /// `finish()` eventually reporting `UnexpectedEof`).
    #[test]
    fn empty_array_metadata_completes_immediately_not_a_stall() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        push_gguf_string(&mut bytes, "empty");
        bytes.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        bytes.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // count = 0

        let mut parser = GgufStreamParser::new();
        let consumed = parser
            .feed(&bytes)
            .expect("empty array feed should not error");
        assert_eq!(
            consumed,
            bytes.len(),
            "a whole-file feed that makes real progress must not report Ok(0)"
        );
        assert!(
            parser.is_complete(),
            "a zero-element array must not stall the parser"
        );

        let result = parser.finish().expect("finish should succeed");
        assert_eq!(result.metadata.len(), 1);
        assert_eq!(result.metadata[0].0, "empty");
        match &result.metadata[0].1 {
            GgufValue::Array(arr) => assert!(arr.is_empty()),
            other => panic!("expected an empty Array, got: {other:?}"),
        }
    }
}
