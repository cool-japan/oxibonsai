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
///
/// Covers every GGUF scalar and array type. The signed-array and small-scalar
/// variants are what `prism.hadamard.sign_values` / `sign_widths`,
/// `qwen35.rope.dimension_sections` and `tokenizer.ggml.token_type` need —
/// all `arr[i32]` in the real 27B files. Smuggling `sign_values` through
/// [`MetadataWriteValue::ArrayU32`] does not work, because `-1` comes back as
/// `4294967295` (core-gguf-09).
#[derive(Debug, Clone)]
pub enum MetadataWriteValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    U64(u64),
    /// Signed 64-bit scalar (`GgufType::Int64`).
    ///
    /// Without this variant, a source `MetadataValue::Int64` had nowhere to
    /// round-trip to: `ExportConfig::with_source_metadata` mapped it to
    /// [`MetadataWriteValue::U64`], which turns a negative source value into
    /// a huge positive one (`v as u64`) instead of writing it back as the
    /// signed type the GGUF spec requires.
    I64(i64),
    Bool(bool),
    Str(String),
    ArrayStr(Vec<String>),
    ArrayF32(Vec<f32>),
    ArrayF64(Vec<f64>),
    ArrayU8(Vec<u8>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayI64(Vec<i64>),
    ArrayU64(Vec<u64>),
    ArrayBool(Vec<bool>),
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
///
/// Standard GGML/GGUF type IDs for K-quant formats follow the upstream spec:
/// Q8_0 = 8, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    /// 4-bit quantization, 32 weights per block, FP16 scale (GGML type 2, 18 bytes/block).
    Q4_0 = 2,
    /// IEEE 754 bfloat16: 1 element per "block", 2 bytes (GGML/GGUF type 30).
    ///
    /// Used to store tensors that the source model keeps in bfloat16 at full
    /// fidelity (e.g. FLUX.2 DiT skip-pattern tensors). The reader already
    /// recognises type ID 30 generically, so such tensors round-trip exactly.
    BF16 = 30,
    /// 8-bit quantization, 32 weights per block, FP16 scale (GGML type 8, 34 bytes/block).
    Q8_0 = 8,
    /// 2-bit K-quant, 256 weights per super-block, 4-bit packed scale/min pairs
    /// (GGML type 10, 84 bytes/block).
    Q2_K = 10,
    /// 3-bit K-quant, 256 weights per super-block, 6-bit packed sub-scales
    /// (GGML type 11, 110 bytes/block).
    Q3_K = 11,
    /// 4-bit K-quant, 256 weights per super-block, 6-bit sub-scales (GGML type 12, 144 bytes/block).
    Q4_K = 12,
    /// 5-bit K-quant, 256 weights per super-block, 6-bit sub-scales (GGML type 13, 176 bytes/block).
    Q5_K = 13,
    /// 6-bit K-quant, 256 weights per super-block, int8 sub-scales (GGML type 14, 210 bytes/block).
    Q6_K = 14,
    /// 8-bit K-quant, 256 weights per super-block, FP32 super-block scale
    /// (GGML type 15, 292 bytes/block). Unlike every other K-quant format the
    /// super-block scale is `f32`, not `f16`.
    Q8_K = 15,
    /// llama.cpp ternary quantization: 256 sign-2 bits + FP16 group scale (upstream ID 35).
    TQ2_0 = 35,
    /// 1-bit, 128-element groups (OxiBonsai custom; type ID 41).
    Q1_0G128 = 41,
    /// **Legacy OxiBonsai** ternary: 128 sign-2 bits + FP16 group scale, `qs`
    /// first / `d` last (type ID 42).
    TQ2_0_g128 = 42,
    /// PrismML FP8 E4M3FN quantization (type ID 43).
    F8_E4M3 = 43,
    /// PrismML FP8 E5M2 quantization (type ID 44).
    F8_E5M2 = 44,
    /// llama.cpp ternary, base-3 trit packing, 256-element groups (upstream ID 34).
    TQ1_0 = 34,
    /// Microscaling FP4: 32-element blocks, E8M0 byte scale (upstream ID 39).
    MXFP4 = 39,
    /// NVIDIA FP4: 64-element blocks, four E4M3 sub-block scales (upstream ID 40).
    NVFP4 = 40,
    /// PrismML `PQ2_0`: 128 weights in 34 bytes, `d` FIRST (type ID 142).
    PQ2_0 = 142,
    /// PrismML `PTQ1_0`: 128 weights in 28 bytes, `d` LAST (type ID 143).
    PTQ1_0 = 143,
    /// Mainline `block_q2_0`: 64 weights in 18 bytes, `d` FIRST.
    ///
    /// Stored on disk under ggml id **42**, which [`TensorType::TQ2_0_g128`]
    /// already owns, so this carries a private sentinel discriminant.
    /// Serialisation must go through [`TensorType::wire_id`] — writing
    /// `self as u32` would emit `0x4000_002A` verbatim and surface much later
    /// as an unreadable file.
    Q2_0G64 = 0x4000_002A,
}

impl TensorType {
    /// The ggml type id this variant is *stored* under on disk.
    ///
    /// The only value that may ever reach the tensor-info record.
    pub fn wire_id(self) -> u32 {
        match self {
            Self::Q2_0G64 => 42,
            other => other as u32,
        }
    }

    /// Block size in elements for this quantisation type.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q8_0 | Self::MXFP4 => 32,
            Self::NVFP4 | Self::Q2_0G64 => 64,
            Self::Q1_0G128 => 128,
            Self::TQ2_0_g128 | Self::PQ2_0 | Self::PTQ1_0 => 128,
            Self::TQ2_0
            | Self::TQ1_0
            | Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K => 256,
            Self::F8_E4M3 | Self::F8_E5M2 => 32,
        }
    }

    /// Block size in bytes for this quantisation type.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,                    // 2 (FP16 scale) + 32 (i8 weights)
            Self::Q2_K => 84, // 16 (packed 4-bit scale/min) + 64 (2-bit qs) + 2+2 (FP16 d, dmin)
            Self::Q3_K => 110, // 32 (hmask) + 64 (2-bit qs) + 12 (packed 6-bit scales) + 2 (FP16 d)
            Self::Q4_K => 144, // 2+2+12+128 (FP16 d+dmin, packed 6-bit scales, 4-bit nibbles)
            Self::Q5_K => 176, // 2+2+12+32+128 (FP16 d+dmin, scales, qh high bits, qs nibbles)
            Self::Q6_K => 210, // 128+64+16+2 (ql low nibbles, qh high bits, i8 scales, FP16 d)
            Self::Q8_K => 292, // 4 (f32 d) + 256 (i8 qs) + 32 (i16 bsums)
            Self::Q1_0G128 => 18, // 2 (FP16 scale) + 16 (128 sign bits)
            Self::TQ2_0_g128 => 34, // 2 (FP16 scale) + 32 (128 ternary-2bit packed)
            Self::TQ2_0 => 66, // 2 (FP16 scale) + 64 (256 ternary-2bit packed)
            Self::TQ1_0 => 54, // 2 + 256/64 + (256 - 4*256/64)/5
            Self::MXFP4 => 17, // 1 (E8M0) + 32/2
            Self::NVFP4 => 36, // 64/16 (E4M3 sub-scales) + 64/2
            Self::PQ2_0 => 34, // 2 (FP16 scale) + 32 (128 × 2-bit)
            Self::PTQ1_0 => 28, // 24 (qs) + 2 (qh) + 2 (FP16 scale)
            Self::Q2_0G64 => 18, // 2 (FP16 scale) + 16 (64 × 2-bit)
            Self::F8_E4M3 | Self::F8_E5M2 => 34, // 32 bytes qs + 2 bytes FP16 scale
        }
    }

    /// Expected byte count for a tensor with `element_count` elements,
    /// computed from the **flattened** element count.
    ///
    /// Correct only when the tensor's first dimension is a whole number of
    /// blocks, which [`GgufWriter::write`] now enforces; prefer
    /// [`TensorType::row_bytes`], which is ggml's real formula.
    ///
    /// `block_bytes` is a small fixed constant from a closed set of known
    /// formats, but `num_blocks` can in principle already be huge for a
    /// pathological (if arithmetically valid) `element_count`, so the final
    /// multiplication uses `saturating_mul` — mirroring
    /// `TensorInfo::data_size` on the reader side — rather than risking a
    /// silent wrap to a small, plausible-looking byte count.
    pub fn expected_bytes(self, element_count: u64) -> u64 {
        let block_size = self.block_size() as u64;
        let block_bytes = self.block_bytes() as u64;
        let num_blocks = element_count.div_ceil(block_size);
        num_blocks.saturating_mul(block_bytes)
    }

    /// ggml's **per-row** byte size: `nrows * ceil(ne0/blck) * type_size`.
    ///
    /// Rows are quantized independently, so a tensor whose `ne0` is not a
    /// block multiple needs one extra block per row — the flattened form
    /// under-counts (core-gguf-N1).
    pub fn row_bytes(self, shape: &[u64]) -> u64 {
        let block_size = self.block_size() as u64;
        let block_bytes = self.block_bytes() as u64;
        let ne0 = shape.first().copied().unwrap_or(1);
        let nrows = shape
            .iter()
            .skip(1)
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .unwrap_or(u64::MAX);
        nrows
            .saturating_mul(ne0.div_ceil(block_size))
            .saturating_mul(block_bytes)
    }
}

/// Total element count for `shape`, computed with checked multiplication.
///
/// Unlike the reader's `TensorInfo::element_count` (which saturates to
/// `u64::MAX` on overflow, because it must still return *some* value for a
/// file it cannot refuse to have already read), the writer controls
/// whether to proceed at all: a `TensorEntry.shape` whose product overflows
/// `u64` is malformed input from the caller, so this returns `None` and lets
/// `GgufWriter::write` fail cleanly with `WriteError::ShapeOverflow` instead
/// of silently wrapping to a small, plausible-looking element count in a
/// release build (`overflow-checks = false` is the Cargo default).
fn checked_element_count(shape: &[u64]) -> Option<u64> {
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
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

/// A producer that writes one tensor's bytes into a sink on demand.
///
/// Returns the number of bytes written, which must equal the length declared
/// alongside it in [`TensorSource::Callback`].
pub type TensorProducer<'a> = Box<dyn FnMut(&mut dyn Write) -> std::io::Result<u64> + 'a>;

/// Where a tensor's bytes come from.
///
/// A 27 B model is 6–7.2 GB of tensor data; holding every tensor as an owned
/// `Vec<u8>` before `write()` is a hard blocker on a 24 GB machine
/// (core-gguf-18). [`TensorSource::Callback`] lets a converter quantise and
/// stream one row at a time instead, and [`TensorSource::Borrowed`] lets it
/// hand over an mmap slice with no copy at all.
pub enum TensorSource<'a> {
    /// Bytes the writer owns.
    Owned(Vec<u8>),
    /// Bytes borrowed from a caller-owned buffer (typically an mmap).
    Borrowed(&'a [u8]),
    /// A producer that writes the tensor's bytes on demand, plus the exact
    /// number of bytes it promises to write.
    ///
    /// Only [`GgufWriter::write_streaming`] can drive it — it needs `&mut`.
    Callback(TensorProducer<'a>, u64),
}

impl std::fmt::Debug for TensorSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
            Self::Borrowed(s) => write!(f, "Borrowed({} bytes)", s.len()),
            Self::Callback(_, n) => write!(f, "Callback({n} bytes)"),
        }
    }
}

impl TensorSource<'_> {
    /// The number of bytes this source promises to produce.
    pub fn declared_len(&self) -> u64 {
        match self {
            Self::Owned(v) => v.len() as u64,
            Self::Borrowed(s) => s.len() as u64,
            Self::Callback(_, n) => *n,
        }
    }

    /// Whether this source needs `&mut` access (i.e. is a callback).
    pub fn needs_mut(&self) -> bool {
        matches!(self, Self::Callback(..))
    }
}

/// A tensor whose bytes come from a [`TensorSource`].
pub struct TensorStream<'a> {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape dimensions — outermost dimension last, matching GGUF convention.
    pub shape: Vec<u64>,
    /// Quantisation type.
    pub tensor_type: TensorType,
    /// Where the bytes come from.
    pub source: TensorSource<'a>,
}

/// Round `offset` up to the next multiple of `alignment`.
///
/// Re-exported from [`crate::gguf::tensor_info`] so the writer's padding and
/// the reader's offset assertion are provably the same function
/// (core-gguf-10).
pub use crate::gguf::tensor_info::align_up;

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
pub struct GgufWriter<'a> {
    metadata: Vec<(String, MetadataWriteValue)>,
    tensors: Vec<TensorStream<'a>>,
    /// Alignment boundary for the tensor data section (default 32).
    alignment: usize,
}

impl<'a> GgufWriter<'a> {
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

    /// Append a tensor entry whose bytes the writer takes ownership of.
    pub fn add_tensor(&mut self, entry: TensorEntry) -> &mut Self {
        self.tensors.push(TensorStream {
            name: entry.name,
            shape: entry.shape,
            tensor_type: entry.tensor_type,
            source: TensorSource::Owned(entry.data),
        });
        self
    }

    /// Append a tensor whose bytes come from a [`TensorSource`].
    ///
    /// Use this to borrow an mmap slice, or to quantise-and-stream a tensor
    /// row by row instead of materialising it (core-gguf-18). A
    /// [`TensorSource::Callback`] source requires
    /// [`GgufWriter::write_streaming`].
    pub fn add_tensor_stream(&mut self, stream: TensorStream<'a>) -> &mut Self {
        self.tensors.push(stream);
        self
    }

    /// Number of tensors queued.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Whether any queued tensor needs `&mut` access to serialise.
    pub fn needs_streaming_write(&self) -> bool {
        self.tensors.iter().any(|t| t.source.needs_mut())
    }

    /// Override the alignment boundary (default: 32).
    pub fn set_alignment(&mut self, alignment: usize) -> &mut Self {
        self.alignment = alignment;
        self
    }

    /// Serialise the GGUF file into `out`, returning the total number of bytes
    /// written on success.
    ///
    /// A [`TensorSource::Callback`] tensor cannot be driven from `&self`; use
    /// [`GgufWriter::write_streaming`] instead.
    pub fn write<W: Write>(&self, out: &mut W) -> Result<usize, WriteError> {
        if let Some(t) = self.tensors.iter().find(|t| t.source.needs_mut()) {
            return Err(WriteError::StreamingSourceRequiresWriteStreaming {
                name: t.name.clone(),
            });
        }
        let mut pos = self.write_header_and_infos(out)?;
        for entry in &self.tensors {
            let written = match &entry.source {
                TensorSource::Owned(v) => {
                    out.write_all(v)
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    v.len()
                }
                TensorSource::Borrowed(s) => {
                    out.write_all(s)
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    s.len()
                }
                // Excluded above; kept exhaustive rather than `unreachable!`.
                TensorSource::Callback(_, _) => {
                    return Err(WriteError::StreamingSourceRequiresWriteStreaming {
                        name: entry.name.clone(),
                    })
                }
            };
            pos += written;
            pos += Self::pad_to_alignment(out, written, self.alignment)?;
        }
        Ok(pos)
    }

    /// Serialise the GGUF file into `out`, driving
    /// [`TensorSource::Callback`] producers as it goes.
    ///
    /// Peak memory is one tensor's worth of whatever the callback buffers
    /// internally, rather than the whole model.
    pub fn write_streaming<W: Write>(&mut self, out: &mut W) -> Result<usize, WriteError> {
        let mut pos = self.write_header_and_infos(out)?;
        let alignment = self.alignment;
        for entry in &mut self.tensors {
            let declared = entry.source.declared_len();
            let written = match &mut entry.source {
                TensorSource::Owned(v) => {
                    out.write_all(v)
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    v.len() as u64
                }
                TensorSource::Borrowed(s) => {
                    out.write_all(s)
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    s.len() as u64
                }
                TensorSource::Callback(produce, _) => {
                    produce(out as &mut dyn Write).map_err(|e| WriteError::Io(e.to_string()))?
                }
            };
            if written != declared {
                return Err(WriteError::DataSizeMismatch {
                    name: entry.name.clone(),
                    expected: declared as usize,
                    got: written as usize,
                });
            }
            let written = usize::try_from(written).map_err(|_| WriteError::ShapeOverflow {
                name: entry.name.clone(),
            })?;
            pos += written;
            pos += Self::pad_to_alignment(out, written, alignment)?;
        }
        Ok(pos)
    }

    /// Write the header, metadata and tensor-info directory plus the padding
    /// that precedes the data section; returns the stream position reached.
    fn write_header_and_infos<W: Write>(&self, out: &mut W) -> Result<usize, WriteError> {
        // The reader (`GgufFile::parse` / `GgufStreamParser::finalize`)
        // unconditionally rejects `alignment == 0` or a non-power-of-two
        // alignment with `AlignmentError`. Validate here rather than
        // silently emitting a syntactically-valid file the project's own
        // reader will always refuse to load.
        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(WriteError::InvalidAlignment {
                alignment: self.alignment,
            });
        }

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
        let alignment_u64 = self.alignment as u64;
        let mut data_offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        let mut running_offset: u64 = 0;
        for entry in &self.tensors {
            data_offsets.push(running_offset);
            let expected = Self::validated_tensor_bytes(entry)?;
            // ggml lays every tensor at `GGML_PAD(running, alignment)`
            // (`ggml/src/gguf.cpp:780-793`); laying them back-to-back
            // produces offsets llama.cpp hard-rejects and that our own
            // `slice_from_bytes` alignment guards refuse (core-gguf-10).
            let advanced =
                running_offset
                    .checked_add(expected)
                    .ok_or_else(|| WriteError::ShapeOverflow {
                        name: entry.name.clone(),
                    })?;
            running_offset =
                align_up(advanced, alignment_u64).ok_or_else(|| WriteError::ShapeOverflow {
                    name: entry.name.clone(),
                })?;
        }

        for (idx, entry) in self.tensors.iter().enumerate() {
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

            // tensor type (u32) — ALWAYS the wire id, never `as u32`: the
            // `Q2_0G64` sentinel discriminant must never reach the file.
            Self::write_le_u32(out, entry.tensor_type.wire_id())?;
            pos += 4;

            // offset into data section (u64)
            Self::write_le_u64(out, data_offsets[idx])?;
            pos += 8;
        }

        // ── 4. Alignment padding ────────────────────────────────────────────
        let pad = Self::pad_to_alignment(out, pos, self.alignment)?;
        pos += pad;

        Ok(pos)
    }

    /// Validate one queued tensor and return its exact byte size.
    ///
    /// Rejects a shape whose product overflows `u64`, a quantized tensor
    /// whose first dimension is not a whole number of blocks (ggml rejects
    /// those at `gguf.cpp:721-727`), and a source whose declared length does
    /// not match the shape.
    fn validated_tensor_bytes(entry: &TensorStream<'_>) -> Result<u64, WriteError> {
        checked_element_count(&entry.shape).ok_or_else(|| WriteError::ShapeOverflow {
            name: entry.name.clone(),
        })?;
        let block_size = entry.tensor_type.block_size() as u64;
        if block_size > 1 {
            let ne0 = entry.shape.first().copied().unwrap_or(1);
            if !ne0.is_multiple_of(block_size) {
                return Err(WriteError::TensorLayout {
                    name: entry.name.clone(),
                    reason: format!(
                        "first dimension {ne0} is not a multiple of the {:?} block size \
                         {block_size}; ggml quantizes each row independently",
                        entry.tensor_type
                    ),
                });
            }
        }
        let expected = entry.tensor_type.row_bytes(&entry.shape);
        let declared = entry.source.declared_len();
        if declared != expected {
            let expected = usize::try_from(expected).unwrap_or(usize::MAX);
            let got = usize::try_from(declared).unwrap_or(usize::MAX);
            return Err(WriteError::DataSizeMismatch {
                name: entry.name.clone(),
                expected,
                got,
            });
        }
        Ok(expected)
    }

    /// Convenience wrapper: serialise the complete GGUF file into a `Vec<u8>`.
    ///
    /// # Memory cost
    ///
    /// This buffers the **entire file** in RAM on top of whatever the queued
    /// tensors already hold, so it is for tests and small metadata-only files
    /// only. Production converters must call [`GgufWriter::write`] (or
    /// [`GgufWriter::write_streaming`]) against a `File` — a 27 B model is
    /// 6–7.2 GB of tensor data (core-gguf-18).
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
            MetadataWriteValue::U8(v) => {
                Self::write_le_u32(out, GgufType::Uint8 as u32)?;
                out.write_all(&[*v])
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += 5;
            }
            MetadataWriteValue::I8(v) => {
                Self::write_le_u32(out, GgufType::Int8 as u32)?;
                out.write_all(&v.to_le_bytes())
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += 5;
            }
            MetadataWriteValue::U16(v) => {
                Self::write_le_u32(out, GgufType::Uint16 as u32)?;
                out.write_all(&v.to_le_bytes())
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += 6;
            }
            MetadataWriteValue::I16(v) => {
                Self::write_le_u32(out, GgufType::Int16 as u32)?;
                out.write_all(&v.to_le_bytes())
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += 6;
            }
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
            MetadataWriteValue::I64(v) => {
                Self::write_le_u32(out, GgufType::Int64 as u32)?;
                out.write_all(&v.to_le_bytes())
                    .map_err(|e| WriteError::Io(e.to_string()))?;
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
            MetadataWriteValue::ArrayI32(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Int32 as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for &v in items {
                    out.write_all(&v.to_le_bytes())
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    n += 4;
                }
            }
            MetadataWriteValue::ArrayI64(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Int64 as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for &v in items {
                    out.write_all(&v.to_le_bytes())
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    n += 8;
                }
            }
            MetadataWriteValue::ArrayU64(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Uint64 as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for &v in items {
                    Self::write_le_u64(out, v)?;
                    n += 8;
                }
            }
            MetadataWriteValue::ArrayF64(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Float64 as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for &v in items {
                    out.write_all(&v.to_le_bytes())
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    n += 8;
                }
            }
            MetadataWriteValue::ArrayU8(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Uint8 as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                out.write_all(items)
                    .map_err(|e| WriteError::Io(e.to_string()))?;
                n += items.len();
            }
            MetadataWriteValue::ArrayBool(items) => {
                Self::write_le_u32(out, GgufType::Array as u32)?;
                Self::write_le_u32(out, GgufType::Bool as u32)?;
                Self::write_le_u64(out, items.len() as u64)?;
                n += 16;
                for &v in items {
                    out.write_all(&[if v { 1u8 } else { 0u8 }])
                        .map_err(|e| WriteError::Io(e.to_string()))?;
                    n += 1;
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

impl Default for GgufWriter<'_> {
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

    /// The configured alignment is zero or not a power of two.
    ///
    /// The reader (`GgufFile::parse` / `GgufStreamParser::finalize`)
    /// unconditionally rejects such an alignment, so refusing to write one
    /// avoids silently producing a file no reader in this project can load.
    #[error("invalid alignment {alignment}: must be a nonzero power of two")]
    InvalidAlignment { alignment: usize },

    /// A tensor's `shape` produced an element count, or its contribution to
    /// the cumulative tensor-data offset, that overflows `u64`.
    #[error("tensor '{name}' shape/offset arithmetic overflows u64")]
    ShapeOverflow { name: String },

    /// A quantized tensor's first dimension is not a whole number of blocks.
    ///
    /// ggml quantizes each row independently and hard-rejects such a tensor
    /// (`ggml/src/gguf.cpp:721-727`); emitting one produces a file whose
    /// declared shape and byte size disagree with every other ggml consumer.
    #[error("tensor '{name}' has an invalid layout: {reason}")]
    TensorLayout { name: String, reason: String },

    /// A [`TensorSource::Callback`] tensor was queued but `write` was called.
    #[error(
        "tensor '{name}' has a streaming (callback) source; call \
         GgufWriter::write_streaming instead of write/to_bytes"
    )]
    StreamingSourceRequiresWriteStreaming { name: String },
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

    /// `set_alignment(0)` used to silently produce a syntactically-valid
    /// GGUF file that `GgufFile::parse` unconditionally refuses to load
    /// (`AlignmentError`) — `write`/`to_bytes` must now reject it up front
    /// instead.
    #[test]
    fn zero_alignment_is_rejected_at_write_time() {
        let mut w = GgufWriter::new();
        w.set_alignment(0);
        assert!(matches!(
            w.to_bytes(),
            Err(WriteError::InvalidAlignment { alignment: 0 })
        ));
    }

    /// Same as `zero_alignment_is_rejected_at_write_time` but for a
    /// non-power-of-two alignment (3), which the reader also unconditionally
    /// rejects.
    #[test]
    fn non_power_of_two_alignment_is_rejected_at_write_time() {
        let mut w = GgufWriter::new();
        w.set_alignment(3);
        assert!(matches!(
            w.to_bytes(),
            Err(WriteError::InvalidAlignment { alignment: 3 })
        ));
    }

    /// A shape whose element-count product overflows `u64` must fail
    /// cleanly with `ShapeOverflow` rather than silently wrapping to a
    /// small, plausible-looking element count (the writer's own arithmetic
    /// previously used a plain unchecked `.iter().product()`, unlike the
    /// reader's checked/saturating `TensorInfo::element_count`).
    #[test]
    fn shape_product_overflow_is_rejected_not_silently_wrapped() {
        let mut w = GgufWriter::new();
        w.add_tensor(TensorEntry {
            name: "overflow".to_string(),
            shape: vec![u64::MAX, 2],
            tensor_type: TensorType::F32,
            data: vec![0u8; 8],
        });
        assert!(matches!(
            w.to_bytes(),
            Err(WriteError::ShapeOverflow { name }) if name == "overflow"
        ));
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

    #[test]
    fn bf16_tensor_type_block_geometry() {
        // BF16 is 1 element per "block" of 2 bytes (GGUF type ID 30).
        assert_eq!(TensorType::BF16 as u32, 30);
        assert_eq!(TensorType::BF16.block_size(), 1);
        assert_eq!(TensorType::BF16.block_bytes(), 2);
        assert_eq!(TensorType::BF16.expected_bytes(6), 12);
    }

    #[test]
    fn bf16_tensor_roundtrips_through_reader() {
        use crate::gguf::reader::GgufFile;
        use crate::gguf::types::GgufTensorType;

        // Six bf16 bit patterns (values 1.0, -1.0, 0.0, 0.5, -0.0625, 2000.0).
        let bits: [u16; 6] = [
            half::bf16::from_f32(1.0).to_bits(),
            half::bf16::from_f32(-1.0).to_bits(),
            half::bf16::from_f32(0.0).to_bits(),
            half::bf16::from_f32(0.5).to_bits(),
            half::bf16::from_f32(-0.0625).to_bits(),
            half::bf16::from_f32(2000.0).to_bits(),
        ];
        let mut data = Vec::new();
        for b in bits {
            data.extend_from_slice(&b.to_le_bytes());
        }

        let mut w = GgufWriter::new();
        w.add_tensor(TensorEntry {
            name: "norm_out.weight".to_string(),
            shape: vec![2, 3], // 6 elements
            tensor_type: TensorType::BF16,
            data: data.clone(),
        });
        let file_bytes = w.to_bytes().expect("write bf16 gguf");

        let parsed = GgufFile::parse(&file_bytes).expect("parse bf16 gguf");
        let info = parsed
            .tensors
            .require("norm_out.weight")
            .expect("tensor present");
        assert_eq!(info.tensor_type, GgufTensorType::BF16);
        assert_eq!(info.shape, vec![2, 3]);

        let read_back = parsed.tensor_data("norm_out.weight").expect("data");
        assert_eq!(
            read_back,
            data.as_slice(),
            "bf16 bytes must round-trip exactly"
        );

        // And decode to f32 to confirm the values survive.
        let decoded: Vec<f32> = read_back
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_le_bytes(*c).to_f32())
            .collect();
        assert_eq!(decoded, vec![1.0, -1.0, 0.0, 0.5, -0.0625, 2000.0]);
    }

    // ── MetadataWriteValue::I64 ─────────────────────────────────────────

    /// A negative `i64` is the case the missing variant actually broke:
    /// before this, `ExportConfig::with_source_metadata` had nowhere to
    /// route a source `MetadataValue::Int64` except
    /// [`MetadataWriteValue::U64`], which turns `-1i64` into
    /// `18446744073709551615u64` instead of round-tripping the sign.
    #[test]
    fn i64_metadata_value_roundtrips_a_negative_number() {
        use crate::gguf::reader::GgufFile;
        use crate::gguf::types::GgufValueType;

        let mut w = GgufWriter::new();
        w.add_metadata("test.negative", MetadataWriteValue::I64(-1));
        w.add_metadata("test.min", MetadataWriteValue::I64(i64::MIN));
        let bytes = w.to_bytes().expect("write i64 metadata");
        let parsed = GgufFile::parse(&bytes).expect("parse i64 metadata");

        let negative = parsed.metadata.get("test.negative").expect("key present");
        assert_eq!(negative.type_name(), "Int64");
        assert_eq!(negative.as_i64(), Some(-1));

        let min = parsed.metadata.get("test.min").expect("key present");
        assert_eq!(min.as_i64(), Some(i64::MIN));

        // The type tag on the wire must be GgufType::Int64 (11), not the
        // Uint64 tag the old U64-only mapping would have used.
        assert_eq!(GgufType::Int64 as u32, GgufValueType::Int64 as u32);
    }

    // ── New tensor types + wire ids ───────────────────────────────────────

    #[test]
    fn new_tensor_types_have_the_ggml_geometry() {
        for (ty, id, blk, bytes) in [
            (TensorType::PQ2_0, 142u32, 128usize, 34usize),
            (TensorType::PTQ1_0, 143, 128, 28),
            (TensorType::Q2_0G64, 42, 64, 18),
            (TensorType::TQ1_0, 34, 256, 54),
            (TensorType::MXFP4, 39, 32, 17),
            (TensorType::NVFP4, 40, 64, 36),
            // The three K-quant formats this
            // module has writer support for. Byte sizes match
            // `oxibonsai_core::quant_k::{BLOCK_Q2_K_BYTES, BLOCK_Q3K_BYTES,
            // BLOCK_Q8K_BYTES}` and the reader-side
            // `GgufTensorType::block_bytes`, which this must never drift
            // from (a mismatch would make the writer emit a tensor the
            // reader immediately rejects as truncated or overlapping).
            (TensorType::Q2_K, 10, 256, 84),
            (TensorType::Q3_K, 11, 256, 110),
            (TensorType::Q8_K, 15, 256, 292),
        ] {
            assert_eq!(ty.wire_id(), id, "{ty:?} wire id");
            assert_eq!(ty.block_size(), blk, "{ty:?} block size");
            assert_eq!(ty.block_bytes(), bytes, "{ty:?} block bytes");
        }
    }

    /// The `Q2_0G64` sentinel discriminant must never reach a file; only
    /// `wire_id()` may be serialised.
    #[test]
    fn q2_0_g64_sentinel_is_not_its_wire_id() {
        assert_eq!(TensorType::Q2_0G64 as u32, 0x4000_002A);
        assert_eq!(TensorType::Q2_0G64.wire_id(), 42);
        assert_eq!(TensorType::TQ2_0_g128.wire_id(), 42);
    }

    // ── Per-row size + ne0 validation ─────────────────────────────────────

    #[test]
    fn row_bytes_uses_the_per_row_formula() {
        assert_eq!(TensorType::TQ2_0_g128.row_bytes(&[200, 3]), 204);
        assert_eq!(TensorType::TQ2_0_g128.row_bytes(&[256, 4]), 4 * 2 * 34);
        assert_eq!(TensorType::F32.row_bytes(&[7, 3]), 84);
        assert_eq!(TensorType::F32.row_bytes(&[]), 4);
    }

    /// A `[100, 2]` `TQ2_0_g128` entry must be refused: 100 is not a multiple
    /// of the 128-wide block, so ggml cannot describe it.
    #[test]
    fn writer_refuses_ne0_that_is_not_a_block_multiple() {
        let mut w = GgufWriter::new();
        w.add_tensor(TensorEntry {
            name: "ragged.weight".to_string(),
            shape: vec![100, 2],
            tensor_type: TensorType::TQ2_0_g128,
            data: vec![0u8; 68],
        });
        match w.to_bytes() {
            Err(WriteError::TensorLayout { name, reason }) => {
                assert_eq!(name, "ragged.weight");
                assert!(reason.contains("100"), "reason: {reason}");
                assert!(reason.contains("128"), "reason: {reason}");
            }
            other => panic!("expected TensorLayout, got {other:?}"),
        }
    }

    #[test]
    fn writer_accepts_block_aligned_shapes() {
        let mut w = GgufWriter::new();
        w.add_tensor(TensorEntry {
            name: "ok.weight".to_string(),
            shape: vec![128, 2],
            tensor_type: TensorType::TQ2_0_g128,
            data: vec![0u8; 68],
        });
        assert!(w.to_bytes().is_ok());
    }

    // ── Streaming sources (core-gguf-18) ──────────────────────────────────

    #[test]
    fn borrowed_source_avoids_the_copy_and_round_trips() {
        let backing = vec![7u8; 68];
        let mut w = GgufWriter::new();
        w.add_tensor_stream(TensorStream {
            name: "b.weight".to_string(),
            shape: vec![128, 2],
            tensor_type: TensorType::TQ2_0_g128,
            source: TensorSource::Borrowed(&backing),
        });
        let bytes = w.to_bytes().expect("write");
        let parsed = crate::gguf::reader::GgufFile::parse(&bytes).expect("parse");
        assert_eq!(parsed.tensor_data("b.weight").expect("data"), &backing[..]);
    }

    #[test]
    fn callback_source_streams_and_round_trips() {
        let mut w = GgufWriter::new();
        w.add_tensor_stream(TensorStream {
            name: "c.weight".to_string(),
            shape: vec![128, 2],
            tensor_type: TensorType::TQ2_0_g128,
            source: TensorSource::Callback(
                Box::new(|out| {
                    // Emit the tensor one 34-byte block at a time, so peak
                    // memory is one block rather than the whole tensor.
                    let mut produced = 0u64;
                    for _ in 0..2 {
                        out.write_all(&[3u8; 34])?;
                        produced += 34;
                    }
                    Ok(produced)
                }),
                68,
            ),
        });
        assert!(w.needs_streaming_write());
        assert!(matches!(
            w.to_bytes(),
            Err(WriteError::StreamingSourceRequiresWriteStreaming { .. })
        ));
        let mut bytes: Vec<u8> = Vec::new();
        w.write_streaming(&mut bytes).expect("streaming write");
        let parsed = crate::gguf::reader::GgufFile::parse(&bytes).expect("parse");
        assert_eq!(parsed.tensor_data("c.weight").expect("data"), [3u8; 68]);
    }

    #[test]
    fn callback_that_writes_the_wrong_length_is_rejected() {
        let mut w = GgufWriter::new();
        w.add_tensor_stream(TensorStream {
            name: "short.weight".to_string(),
            shape: vec![128, 2],
            tensor_type: TensorType::TQ2_0_g128,
            source: TensorSource::Callback(
                Box::new(|out| {
                    out.write_all(&[0u8; 34])?;
                    Ok(34)
                }),
                68,
            ),
        });
        let mut bytes: Vec<u8> = Vec::new();
        assert!(matches!(
            w.write_streaming(&mut bytes),
            Err(WriteError::DataSizeMismatch { .. })
        ));
    }

    #[test]
    fn tensor_source_reports_its_declared_length() {
        assert_eq!(TensorSource::Owned(vec![0u8; 5]).declared_len(), 5);
        let buf = [0u8; 7];
        assert_eq!(TensorSource::Borrowed(&buf).declared_len(), 7);
        let cb = TensorSource::Callback(Box::new(|_| Ok(0)), 11);
        assert_eq!(cb.declared_len(), 11);
        assert!(cb.needs_mut());
        assert!(format!("{cb:?}").contains("Callback"));
    }
}
