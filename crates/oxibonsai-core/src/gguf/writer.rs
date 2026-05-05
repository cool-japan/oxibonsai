//! GGUF v3 file writer.
//!
//! Produces a well-formed GGUF binary file from metadata key-value pairs
//! and tensor data, following the little-endian GGUF v3 specification.
//!
//! # Format summary
//!
//! ```text
//! [magic: 4 bytes]          — "GGUF" = 0x47 0x47 0x55 0x46
//! [version: u32]            — 3
//! [tensor_count: u64]
//! [metadata_kv_count: u64]
//! [metadata KV pairs]       — key (string), type (u32), value
//! [tensor info entries]     — name, n_dims (u32), shape (u64×n), type (u32), offset (u64)
//! [padding to alignment]    — zero bytes to reach next alignment boundary
//! [tensor data]             — raw bytes for each tensor, laid out sequentially
//! ```

use std::io::Write;

// ─── Metadata value type codes ──────────────────────────────────────────────

/// GGUF metadata value type codes — matches the GGUF spec exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

// ─── Metadata value ──────────────────────────────────────────────────────────

/// A typed metadata value to be written into a GGUF file.
#[derive(Debug, Clone)]
pub enum MetadataWriteValue {
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    U64(u64),
    Bool(bool),
    Str(String),
    ArrayStr(Vec<String>),
    ArrayF32(Vec<f32>),
    ArrayU32(Vec<u32>),
}

// Keep the public name requested in the task spec as an alias.
pub use MetadataWriteValue as MetadataValue;

// ─── Tensor type ─────────────────────────────────────────────────────────────

/// Tensor quantization type codes used by OxiBonsai in GGUF files.
///
/// Note: `Q1_0G128` maps to type ID **41** (the PrismML extension ID used
/// throughout the existing OxiBonsai reader).  `TQ2_0_g128` maps to type
/// ID **42** (PrismML ternary extension) and `TQ2_0` maps to type ID **35**
/// (llama.cpp upstream ternary quantization).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    /// llama.cpp ternary quantization: 256 sign-2 bits + FP16 group scale (upstream ID 35).
    TQ2_0 = 35,
    /// 1-bit, 128-element groups (OxiBonsai custom; type ID 41).
    Q1_0G128 = 41,
    /// PrismML ternary quantization: 128 sign-2 bits + FP16 group scale (type ID 42).
    TQ2_0_g128 = 42,
    /// PrismML FP8 E4M3FN quantization (type ID 43).
    F8_E4M3 = 43,
    /// PrismML FP8 E5M2 quantization (type ID 44).
    F8_E5M2 = 44,
}

impl TensorType {
    /// Block size in elements for this quantisation type.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 => 32,
            Self::Q1_0G128 => 128,
            Self::TQ2_0_g128 => 128,
            Self::TQ2_0 => 256,
            Self::F8_E4M3 | Self::F8_E5M2 => 32,
        }
    }

    /// Block size in bytes for this quantisation type.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q1_0G128 => 18,   // 2 (FP16 scale) + 16 (128 sign bits)
            Self::TQ2_0_g128 => 34, // 2 (FP16 scale) + 32 (128 ternary-2bit packed)
            Self::TQ2_0 => 66,      // 2 (FP16 scale) + 64 (256 ternary-2bit packed)
            Self::F8_E4M3 | Self::F8_E5M2 => 34, // 32 bytes qs + 2 bytes FP16 scale
        }
    }

    /// Expected byte count for a tensor with `element_count` elements.
    pub fn expected_bytes(self, element_count: u64) -> u64 {
        let block_size = self.block_size() as u64;
        let block_bytes = self.block_bytes() as u64;
        let num_blocks = element_count.div_ceil(block_size);
        num_blocks * block_bytes
    }
}

// ─── Tensor entry ─────────────────────────────────────────────────────────────

/// A tensor to be written to a GGUF file.
pub struct TensorEntry {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape dimensions — outermost dimension last, matching GGUF convention.
    pub shape: Vec<u64>,
    /// Quantisation type.
    pub tensor_type: TensorType,
    /// Raw serialised bytes. The caller is responsible for correct layout.
    pub data: Vec<u8>,
}

// ─── Writer ───────────────────────────────────────────────────────────────────

/// Builds and serialises a complete GGUF v3 file.
///
/// # Example
/// ```ignore
/// use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorEntry, TensorType};
///
/// let mut writer = GgufWriter::new();
/// writer.add_metadata("general.name", MetadataWriteValue::Str("my-model".to_string()));
///
/// let data: Vec<u8> = 1.0_f32.to_le_bytes().to_vec();
/// writer.add_tensor(TensorEntry {
///     name: "token_embd.weight".to_string(),
///     shape: vec![1],
///     tensor_type: TensorType::F32,
///     data,
/// });
///
/// let bytes = writer.to_bytes().expect("write failed");
/// ```
pub struct GgufWriter {
    metadata: Vec<(String, MetadataWriteValue)>,
    tensors: Vec<TensorEntry>,
    /// Alignment boundary for the tensor data section (default 32).
    alignment: usize,
}

impl GgufWriter {
    /// Create a new writer with default alignment of 32 bytes.
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
        }
    }

    /// Append a metadata key-value pair.
    pub fn add_metadata(&mut self, key: &str, value: MetadataWriteValue) -> &mut Self {
        self.metadata.push((key.to_string(), value));
        self
    }

    /// Append a tensor entry.
    pub fn add_tensor(&mut self, entry: TensorEntry) -> &mut Self {
        self.tensors.push(entry);
        self
    }

    /// Override the alignment boundary (default: 32).
    pub fn set_alignment(&mut self, alignment: usize) -> &mut Self {
        self.alignment = alignment;
        self
    }

    /// Serialise the GGUF file into `out`, returning the total number of bytes
    /// written on success.
    pub fn write<W: Write>(&self, out: &mut W) -> Result<usize, WriteError> {
        let mut pos: usize = 0;

        // ── Build effective metadata list ───────────────────────────────────
        // When a non-default alignment is requested, inject `general.alignment`
        // so that the reader can reconstruct the correct data_offset.
        // For the default alignment (32) the reader already defaults to 32,
        // so no extra entry is needed.
        const DEFAULT_ALIGNMENT: usize = 32;
        let has_alignment = self.metadata.iter().any(|(k, _)| k == "general.alignment");
        let alignment_entry: Option<(String, MetadataWriteValue)> =
            if !has_alignment && self.alignment != DEFAULT_ALIGNMENT {
                Some((
                    "general.alignment".to_string(),
                    MetadataWriteValue::U32(self.alignment as u32),
                ))
            } else {
                None
            };

        let effective_kv_count =
            self.metadata.len() + if alignment_entry.is_some() { 1 } else { 0 };

        // ── 1. Header ───────────────────────────────────────────────────────
        // Magic: GGUF_MAGIC = 0x46554747 stored as little-endian u32.
        // This matches what the reader expects (header.rs: const GGUF_MAGIC: u32 = 0x4655_4747).
        const GGUF_MAGIC: u32 = 0x4655_4747;
        Self::write_le_u32(out, GGUF_MAGIC)?;
        pos += 4;

        // Version: 3
        Self::write_le_u32(out, 3)?;
        pos += 4;

        // tensor_count
        Self::write_le_u64(out, self.tensors.len() as u64)?;
        pos += 8;

        // metadata_kv_count (including injected alignment key if needed)
        Self::write_le_u64(out, effective_kv_count as u64)?;
        pos += 8;

        // ── 2. Metadata KV pairs ────────────────────────────────────────────
        // Write injected alignment entry first (if any) so reader can find it.
        if let Some((ref key, ref value)) = alignment_entry {
            pos += Self::write_string(out, key)?;
            pos += Self::write_metadata_value(out, value)?;
        }
        for (key, value) in &self.metadata {
            pos += Self::write_string(out, key)?;
            pos += Self::write_metadata_value(out, value)?;
        }

        // ── 3. Tensor info entries ──────────────────────────────────────────
        // We need each tensor's offset into the data section. Compute cumulative
        // data offsets now (the data section starts after alignment padding).
        let mut data_offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        let mut running_offset: u64 = 0;
        for entry in &self.tensors {
            data_offsets.push(running_offset);
            let element_count: u64 = entry.shape.iter().product();
            let expected = entry.tensor_type.expected_bytes(element_count);
            running_offset += expected;
        }

        for (idx, entry) in self.tensors.iter().enumerate() {
            // Validate data size
            let element_count: u64 = entry.shape.iter().product();
            let expected = entry.tensor_type.expected_bytes(element_count) as usize;
            if entry.data.len() != expected {
                return Err(WriteError::DataSizeMismatch {
                    name: entry.name.clone(),
                    expected,
                    got: entry.data.len(),
                });
            }

            pos += Self::write_string(out, &entry.name)?;

            // n_dims (u32)
            let n_dims = entry.shape.len() as u32;
            Self::write_le_u32(out, n_dims)?;
            pos += 4;

            // shape (u64 per dimension)
            for &dim in &entry.shape {
                Self::write_le_u64(out, dim)?;
                pos += 8;
            }

            // tensor type (u32)
            Self::write_le_u32(out, entry.tensor_type as u32)?;
            pos += 4;

            // offset into data section (u64)
            Self::write_le_u64(out, data_offsets[idx])?;
            pos += 8;
        }

        // ── 4. Alignment padding ────────────────────────────────────────────
        let pad = Self::pad_to_alignment(out, pos, self.alignment)?;
        pos += pad;

        // ── 5. Tensor data ──────────────────────────────────────────────────
        for entry in &self.tensors {
            out.write_all(&entry.data)
                .map_err(|e| WriteError::Io(e.to_string()))?;
            pos += entry.data.len();
        }

        Ok(pos)
    }

    /// Convenience wrapper: serialise the complete GGUF file into a `Vec<u8>`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WriteError> {
        let mut buf: Vec<u8> = Vec::new();
        self.write(&mut buf)?;
        Ok(buf)
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Write a GGUF string: `[u64 length][utf-8 bytes]` (no null terminator).
    ///
    /// Returns the number of bytes written.
    fn write_string<W: Write>(out: &mut W, s: &str) -> Result<usize, WriteError> {
        let bytes = s.as_bytes();
        Self::write_le_u64(out, bytes.len() as u64)?;
        out.write_all(bytes)
            .map_err(|e| WriteError::Io(e.to_string()))?;
        Ok(8 + bytes.len())
    }

    /// Write a typed metadata value preceded by its 4-byte type tag.
    ///
    /// Returns the total number of bytes written (type tag + value).
    fn write_metadata_value<W: Write>(
        out: &mut W,
        val: &MetadataWriteValue,
    ) -> Result<usize, WriteError> {
        let mut n: usize = 0;

        match val {
            MetadataWriteValue::U32(v) => {
                Self::write_le_u32(out, GgufType::Uint32 as u32)?;
                Self::write_le_u32(out, *v)?;
                n += 8;
            }
            MetadataWriteValue::I32(v) => {
                Self::write_le_u32(out, GgufType::Int32 as u32)?;
                out.write_all(&v.to_le_bytes())
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += 8;
            }
            MetadataWriteValue::F32(v) => {
                Self::write_le_u32(out, GgufType::Float32 as u32)?;
                Self::write_le_f32(out, *v)?;
                n += 8;
            }
            MetadataWriteValue::F64(v) => {
                Self::write_le_u32(out, GgufType::Float64 as u32)?;
                out.write_all(&v.to_le_bytes())
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += 12;
            }
            MetadataWriteValue::U64(v) => {
                Self::write_le_u32(out, GgufType::Uint64 as u32)?;
                Self::write_le_u64(out, *v)?;
                n += 12;
            }
            MetadataWriteValue::Bool(v) => {
                Self::write_le_u32(out, GgufType::Bool as u32)?;
                out.write_all(&[if *v { 1u8 } else { 0u8 }])
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += 5;
            }
            MetadataWriteValue::Str(s) => {
                Self::write_le_u32(out, GgufType::String as u32)?;
                n += 4;
                n += Self::write_string(out, s)?;
            }
            MetadataWriteValue::ArrayStr(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                // element type
                Self::write_le_u32(out, GgufType::String as u32)?;
                // count
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for s in items {
                    n += Self::write_string(out, s)?;
                }
            }
            MetadataWriteValue::ArrayF32(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Float32 as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for &v in items {
                    Self::write_le_f32(out, v)?;
                    n += 4;
                }
            }
            MetadataWriteValue::ArrayU32(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Uint32 as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for &v in items {
                    Self::write_le_u32(out, v)?;
                    n += 4;
                }
            }
        }

        Ok(n)
    }

    fn write_le_u32<W: Write>(out: &mut W, v: u32) -> Result<(), WriteError> {
        out.write_all(&v.to_le_bytes())
            .map_err(|e| WriteError::Io(e.to_string()))
    }

    fn write_le_u64<W: Write>(out: &mut W, v: u64) -> Result<(), WriteError> {
        out.write_all(&v.to_le_bytes())
            .map_err(|e| WriteError::Io(e.to_string()))
    }

    fn write_le_f32<W: Write>(out: &mut W, v: f32) -> Result<(), WriteError> {
        out.write_all(&v.to_le_bytes())
            .map_err(|e| WriteError::Io(e.to_string()))
    }

    /// Write zero-byte padding so that the stream position reaches the next
    /// alignment boundary.  Returns the number of padding bytes emitted.
    fn pad_to_alignment<W: Write>(
        out: &mut W,
        pos: usize,
        alignment: usize,
    ) -> Result<usize, WriteError> {
        if alignment == 0 {
            return Ok(0);
        }
        let remainder = pos % alignment;
        if remainder == 0 {
            return Ok(0);
        }
        let pad = alignment - remainder;
        let zeros = vec![0u8; pad];
        out.write_all(&zeros)
            .map_err(|e| WriteError::Io(e.to_string()))?;
        Ok(pad)
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur while writing a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    /// An underlying I/O error.
    #[error("I/O error: {0}")]
    Io(String),

    /// The provided tensor data has the wrong byte length.
    #[error("Tensor data size mismatch for {name}: expected {expected}, got {got}")]
    DataSizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_alignment_is_32() {
        let w = GgufWriter::new();
        assert_eq!(w.alignment, 32);
    }

    #[test]
    fn set_alignment_changes_value() {
        let mut w = GgufWriter::new();
        w.set_alignment(64);
        assert_eq!(w.alignment, 64);
    }

    #[test]
    fn empty_file_has_correct_header() {
        let w = GgufWriter::new();
        let bytes = w.to_bytes().expect("write failed");

        // magic: GGUF_MAGIC = 0x46554747 as LE u32
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().expect("slice")),
            0x4655_4747
        );
        // version = 3
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().expect("slice")),
            3
        );
        // tensor_count = 0
        assert_eq!(
            u64::from_le_bytes(bytes[8..16].try_into().expect("slice")),
            0
        );
        // metadata_kv_count = 0
        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().expect("slice")),
            0
        );
    }

    #[test]
    fn data_size_mismatch_returns_error() {
        let mut w = GgufWriter::new();
        w.add_tensor(TensorEntry {
            name: "bad".to_string(),
            shape: vec![4],
            tensor_type: TensorType::F32,
            data: vec![0u8; 8], // wrong: should be 16 bytes for 4×f32
        });
        assert!(matches!(
            w.to_bytes(),
            Err(WriteError::DataSizeMismatch { .. })
        ));
    }
}
