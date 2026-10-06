//! GGUF tensor information parsing.
//!
//! Each tensor in a GGUF file is described by:
//! - Name (GGUF string)
//! - Number of dimensions (u32)
//! - Shape (array of u64, one per dimension)
//! - Quantization type (u32 → GgufTensorType)
//! - Offset into the tensor data section (u64)

use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::types::GgufTensorType;

/// Round `offset` up to the next multiple of `alignment`.
///
/// This is `GGML_PAD` (`ggml.h`). It is the **one** shared helper for the
/// writer's tensor-data padding, the reader's offset assertion and the
/// id-42 resolver's running-sum replay, so a file the writer produces can
/// never be rejected by the reader's check (core-gguf-10).
///
/// Returns `None` when `alignment` is zero or not a power of two, or when
/// rounding up would overflow `u64`.
pub fn align_up(offset: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }
    let rem = offset % alignment;
    if rem == 0 {
        return Some(offset);
    }
    offset.checked_add(alignment - rem)
}

/// `GGML_PAD(ggml_nbytes(t), alignment)` — the byte extent a tensor of
/// `size` bytes occupies in the data section, padding included.
pub fn padded_size(size: u64, alignment: u64) -> Option<u64> {
    align_up(size, alignment)
}

/// ggml's per-row byte size for a tensor of `shape` stored as `tensor_type`.
///
/// `nrows * ceil(ne0 / block_size) * block_bytes`, where `ne0 = shape[0]` and
/// `nrows` is the product of the remaining dimensions (`1` for a 0-D or 1-D
/// tensor). ggml never packs across a row boundary, so the flattened
/// `ceil(total_elements / block_size)` form under-counts by one block per row
/// whenever `ne0` is not a block multiple (core-gguf-N1).
///
/// Overflow saturates to `u64::MAX` rather than wrapping, matching
/// [`TensorInfo::element_count`].
pub fn row_size_bytes(tensor_type: GgufTensorType, shape: &[u64]) -> u64 {
    let block_size = tensor_type.block_size() as u64;
    let block_bytes = tensor_type.block_bytes() as u64;
    // A 0-D tensor is a single scalar: one row of one element.
    let ne0 = shape.first().copied().unwrap_or(1);
    let nrows = shape
        .iter()
        .skip(1)
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
        .unwrap_or(u64::MAX);
    // `block_size` comes from a closed set of known formats and is never
    // zero (asserted by `types::tests::all_block_geometry_positive`).
    let blocks_per_row = ne0.div_ceil(block_size);
    nrows
        .saturating_mul(blocks_per_row)
        .saturating_mul(block_bytes)
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight").
    pub name: String,
    /// Shape dimensions (e.g., [4096, 4096] for a 2D weight matrix).
    pub shape: Vec<u64>,
    /// Quantization type.
    pub tensor_type: GgufTensorType,
    /// Byte offset into the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    ///
    /// `shape` dimensions are read directly from an untrusted GGUF file with
    /// no upper bound on individual dimension values, so the product is
    /// computed with `checked_mul` and saturates to `u64::MAX` on overflow
    /// rather than silently wrapping to a small, plausible-looking value.
    /// Callers that turn this into a byte range (see
    /// [`GgufFile::tensor_data`](crate::gguf::reader::GgufFile::tensor_data))
    /// must still bounds-check against the actual file length; `u64::MAX`
    /// only guarantees that check will fail rather than silently pass.
    pub fn element_count(&self) -> u64 {
        self.shape
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .unwrap_or(u64::MAX)
    }

    /// Total number of bytes this tensor occupies in the data section.
    ///
    /// Uses ggml's **per-row** formula (see [`row_size_bytes`]); the padding
    /// a following tensor's offset implies is *not* included — use
    /// [`TensorInfo::padded_extent`] for that.
    ///
    /// Overflow-safe for the same reason as [`element_count`](Self::element_count):
    /// a malformed shape or (in principle) a huge block-byte count saturates
    /// to `u64::MAX` instead of wrapping.
    ///
    /// # Provisional for wire id 42
    ///
    /// `self.tensor_type` for a wire-id-42 tensor is whatever
    /// [`GgufTensorType::from_id`] guessed at parse time — always the legacy
    /// qs-first group-128 reading — never the file's actual resolved
    /// geometry. That guess is byte-identical to the `d`-first group-128
    /// reading but disagrees with a genuine group-64 file. Call sites that
    /// need the *resolved* size for a possibly-ambiguous tensor must
    /// recompute it from the settled
    /// [`crate::gguf::quant_resolve::Resolved42::tensor_type`] (via
    /// [`row_size_bytes`]) instead of trusting this value as-is.
    pub fn data_size(&self) -> u64 {
        row_size_bytes(self.tensor_type, &self.shape)
    }

    /// The byte extent this tensor occupies including ggml's inter-tensor
    /// alignment padding: `GGML_PAD(data_size(), alignment)`.
    ///
    /// This is the quantity llama.cpp accumulates when it validates
    /// `offset[i] == Σ_{j<i} GGML_PAD(nbytes_j, alignment)`
    /// (`ggml/src/gguf.cpp:780-793`). Exported for the id-42 resolver and for
    /// the reader-side layout assertion.
    ///
    /// Returns `None` for an invalid `alignment` or on overflow.
    pub fn padded_extent(&self, alignment: u64) -> Option<u64> {
        padded_size(self.data_size(), alignment)
    }

    /// Number of dimensions.
    pub fn n_dims(&self) -> usize {
        self.shape.len()
    }

    /// The first (fastest-varying) dimension, `ne0`. `1` for a 0-D tensor.
    pub fn ne0(&self) -> u64 {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Validate that this tensor's `ne0` is a whole number of blocks.
    ///
    /// ggml hard-rejects any quantized tensor whose first dimension is not a
    /// multiple of the block size (`ggml/src/gguf.cpp:721-727`), because rows
    /// are quantized independently.
    pub fn validate_row_blocking(&self) -> BonsaiResult<()> {
        let block_size = self.tensor_type.block_size() as u64;
        if block_size <= 1 {
            return Ok(());
        }
        let ne0 = self.ne0();
        if !ne0.is_multiple_of(block_size) {
            return Err(BonsaiError::tensor_layout(
                self.name.clone(),
                format!(
                    "first dimension {ne0} is not a multiple of the {} block size {block_size}",
                    self.tensor_type.name()
                ),
            ));
        }
        Ok(())
    }
}

/// Collection of tensor metadata from a GGUF file.
#[derive(Debug, Clone)]
pub struct TensorStore {
    tensors: HashMap<String, TensorInfo>,
}

impl TensorStore {
    /// Create an empty tensor store.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Parse tensor info entries from a byte slice.
    pub fn parse(data: &[u8], offset: usize, count: u64) -> BonsaiResult<(Self, usize)> {
        let mut cursor = std::io::Cursor::new(data);
        cursor.set_position(offset as u64);

        let mut store = Self::new();
        for _ in 0..count {
            let info = read_tensor_info(&mut cursor)?;
            // ggml refuses a quantized tensor whose first dimension is not a
            // whole number of blocks; so must we, or `data_size()` silently
            // describes a layout the file does not have and the id-42
            // resolver matches the wrong candidate (core-gguf-N1).
            info.validate_row_blocking()?;
            // A duplicate tensor name silently overwrites the earlier entry
            // (and its weight data reference) via plain `HashMap::insert`,
            // with no error, warning, or trace of the discarded tensor
            // anywhere — reject it as a hard parse error instead, since a
            // corrupted/adversarial file that collapses two distinct tensors
            // onto one name must not silently "load successfully" while
            // computing with the wrong weights for that name.
            if store.tensors.contains_key(&info.name) {
                return Err(BonsaiError::InvalidMetadata {
                    key: info.name.clone(),
                    reason: "duplicate tensor name".to_string(),
                });
            }
            store.tensors.insert(info.name.clone(), info);
        }

        Ok((store, cursor.position() as usize))
    }

    /// Get tensor info by name.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get tensor info by name, returning an error if not found.
    pub fn require(&self, name: &str) -> BonsaiResult<&TensorInfo> {
        self.get(name).ok_or_else(|| BonsaiError::TensorNotFound {
            name: name.to_string(),
        })
    }

    /// Number of tensors in the store.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterate over all (name, tensor_info) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorInfo)> {
        self.tensors.iter()
    }

    /// All tensor infos sorted by their data-section offset, ascending.
    ///
    /// This is the order llama.cpp walks when it validates the running padded
    /// sum, and the order the id-42 resolver replays.
    pub fn sorted_by_offset(&self) -> Vec<&TensorInfo> {
        let mut infos: Vec<&TensorInfo> = self.tensors.values().collect();
        infos.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
        infos
    }

    /// Count tensors by quantization type.
    pub fn count_by_type(&self) -> HashMap<GgufTensorType, usize> {
        let mut counts = HashMap::new();
        for info in self.tensors.values() {
            *counts.entry(info.tensor_type).or_insert(0) += 1;
        }
        counts
    }

    /// Get all tensor names sorted alphabetically.
    pub fn sorted_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum string length we accept from GGUF tensor names (256 MB).
const MAX_STRING_LEN: u64 = 256 * 1024 * 1024;

/// Maximum tensor dimensions.
///
/// `GGML_MAX_DIMS` is 4 and llama.cpp rejects `n_dims > 4`, so a 5-D GGUF
/// tensor is invalid input, not a real model shape. The previous 1024 cap let
/// the batch parser accept shapes the streaming parser silently truncated to
/// four dimensions (core-gguf-16).
pub const MAX_TENSOR_DIMS: u32 = 4;

/// Upper bound on a single eager read chunk (and the initial `Vec`
/// reservation) while reading a declared-length GGUF string.
///
/// Shared with `metadata.rs` via [`read_string_body_chunked`] (core-gguf-20
/// / sec-10): the declared `len`
/// prefix is attacker-controlled and only bounded against
/// [`MAX_STRING_LEN`] (256 MiB), so allocating `len` bytes up front —
/// before confirming the reader actually has that much data left — lets a
/// tiny file force a large allocation. Reading in bounded chunks instead
/// keeps peak allocation proportional to bytes actually confirmed present.
pub(crate) const STRING_READ_CHUNK: usize = 64 * 1024;

/// Read exactly `len` bytes from `reader` in bounded [`STRING_READ_CHUNK`]
/// pieces and decode them as UTF-8, instead of allocating a `len`-byte
/// buffer before confirming that many bytes are actually available.
///
/// This is the ONE canonical body for what used to be three independently
/// hardened copies of the identical bounded-chunk read loop — this
/// function (formerly inlined here), `metadata.rs`'s former copy (now a
/// thin wrapper calling this), and `reader.rs`'s `read_gguf_string_chunked`
/// (which keeps its own call site). `pub(crate)` rather than `pub`: this is an internal
/// implementation detail, not part of this crate's public API.
pub(crate) fn read_string_body_chunked<R: std::io::Read>(
    reader: &mut R,
    len: u64,
) -> BonsaiResult<String> {
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

/// Read a GGUF string from a reader: [u64 length] [bytes].
///
/// Reads in bounded [`STRING_READ_CHUNK`]-sized pieces rather than
/// allocating the full declared `len` up front, so a small file with a
/// large declared tensor-name length fails fast instead of first
/// committing a multi-hundred-MB buffer.
fn read_gguf_string<R: std::io::Read>(reader: &mut R) -> BonsaiResult<String> {
    let len = reader
        .read_u64::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if len > MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString { offset: 0 });
    }
    read_string_body_chunked(reader, len)
}

/// Read a single tensor info entry from the reader.
fn read_tensor_info<R: std::io::Read>(reader: &mut R) -> BonsaiResult<TensorInfo> {
    let name = read_gguf_string(reader)?;

    let n_dims = reader
        .read_u32::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if n_dims > MAX_TENSOR_DIMS {
        return Err(BonsaiError::InvalidMetadata {
            key: name,
            reason: format!("tensor has {n_dims} dimensions; GGML_MAX_DIMS is {MAX_TENSOR_DIMS}"),
        });
    }

    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        let dim = reader
            .read_u64::<LittleEndian>()
            .map_err(BonsaiError::MmapError)?;
        shape.push(dim);
    }

    let type_id = reader
        .read_u32::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    let tensor_type = GgufTensorType::from_id(type_id)?;

    let offset = reader
        .read_u64::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;

    Ok(TensorInfo {
        name,
        shape,
        tensor_type,
        offset,
    })
}

/// Well-known GGUF metadata keys for Bonsai/Qwen3 models.
pub mod keys {
    pub const GENERAL_ARCHITECTURE: &str = "general.architecture";
    pub const GENERAL_NAME: &str = "general.name";
    pub const GENERAL_FILE_TYPE: &str = "general.file_type";
    pub const GENERAL_ALIGNMENT: &str = "general.alignment";
    pub const GENERAL_QUANTIZATION_VERSION: &str = "general.quantization_version";

    pub const LLM_CONTEXT_LENGTH: &str = "llm.context_length";
    pub const LLM_EMBEDDING_LENGTH: &str = "llm.embedding_length";
    pub const LLM_BLOCK_COUNT: &str = "llm.block_count";
    pub const LLM_FEED_FORWARD_LENGTH: &str = "llm.feed_forward_length";
    pub const LLM_ATTENTION_HEAD_COUNT: &str = "llm.attention.head_count";
    pub const LLM_ATTENTION_HEAD_COUNT_KV: &str = "llm.attention.head_count_kv";
    pub const LLM_ATTENTION_KEY_LENGTH: &str = "llm.attention.key_length";
    pub const LLM_ATTENTION_LAYER_NORM_RMS_EPSILON: &str = "llm.attention.layer_norm_rms_epsilon";
    pub const LLM_ROPE_FREQ_BASE: &str = "llm.rope.freq_base";
    pub const LLM_VOCAB_SIZE: &str = "llm.vocab_size";

    pub const TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
    pub const TOKENIZER_TOKENS: &str = "tokenizer.ggml.tokens";
    pub const TOKENIZER_BOS_TOKEN_ID: &str = "tokenizer.ggml.bos_token_id";
    pub const TOKENIZER_EOS_TOKEN_ID: &str = "tokenizer.ggml.eos_token_id";
    /// The "end of turn" token id, for models that distinguish it from
    /// [`TOKENIZER_EOS_TOKEN_ID`] (e.g. Bonsai 2's chat contract).
    pub const TOKENIZER_EOT_TOKEN_ID: &str = "tokenizer.ggml.eot_token_id";
}

/// Standard GGUF tensor name patterns for Qwen3/Bonsai models.
pub mod tensor_names {
    pub const TOKEN_EMBD: &str = "token_embd.weight";
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    pub const OUTPUT: &str = "output.weight";

    /// Generate block-scoped tensor names.
    pub fn block_tensor(layer: usize, suffix: &str) -> String {
        format!("blk.{layer}.{suffix}")
    }

    pub const ATTN_Q: &str = "attn_q.weight";
    pub const ATTN_K: &str = "attn_k.weight";
    pub const ATTN_V: &str = "attn_v.weight";
    pub const ATTN_OUTPUT: &str = "attn_output.weight";
    pub const ATTN_NORM: &str = "attn_norm.weight";
    pub const FFN_GATE: &str = "ffn_gate.weight";
    pub const FFN_UP: &str = "ffn_up.weight";
    pub const FFN_DOWN: &str = "ffn_down.weight";
    pub const FFN_NORM: &str = "ffn_norm.weight";
    pub const ATTN_Q_NORM: &str = "attn_q_norm.weight";
    pub const ATTN_K_NORM: &str = "attn_k_norm.weight";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor_info_bytes(name: &str, shape: &[u64], type_id: u32, offset: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&type_id.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes
    }

    #[test]
    fn parse_single_tensor_info() {
        let data = make_tensor_info_bytes("blk.0.attn_q.weight", &[4096, 4096], 41, 0);
        let (store, _) = TensorStore::parse(&data, 0, 1).expect("tensor info parse should succeed");

        let info = store
            .require("blk.0.attn_q.weight")
            .expect("tensor should exist");
        assert_eq!(info.tensor_type, GgufTensorType::Q1_0_g128);
        assert_eq!(info.shape, vec![4096, 4096]);
        assert_eq!(info.element_count(), 4096 * 4096);
    }

    #[test]
    fn q1_0_g128_data_size() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![128],
            tensor_type: GgufTensorType::Q1_0_g128,
            offset: 0,
        };
        // 128 elements / 128 per block = 1 block * 18 bytes
        assert_eq!(info.data_size(), 18);
    }

    #[test]
    fn missing_tensor_returns_error() {
        let store = TensorStore::new();
        assert!(store.require("nonexistent").is_err());
    }

    /// A duplicate tensor name must be a hard parse error, not a silent
    /// last-wins overwrite via `HashMap::insert` that discards the earlier
    /// tensor's weight-data reference with no error or trace.
    #[test]
    fn duplicate_tensor_name_is_hard_error() {
        let mut data = make_tensor_info_bytes("dup.weight", &[4], 0, 0);
        data.extend_from_slice(&make_tensor_info_bytes("dup.weight", &[8], 0, 16));
        let result = TensorStore::parse(&data, 0, 2);
        match result {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "dup.weight");
                assert!(reason.contains("duplicate"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata for duplicate tensor name, got: {other:?}"),
        }
    }

    /// A declared tensor-name length larger than what's actually present
    /// must fail cleanly rather than allocate the full declared length.
    #[test]
    fn tensor_name_declared_longer_than_available_data_fails_cleanly() {
        let mut bytes = Vec::new();
        // Declare a 10 MiB name but supply none of the bytes.
        bytes.extend_from_slice(&(10u64 * 1024 * 1024).to_le_bytes());
        let result = TensorStore::parse(&bytes, 0, 1);
        assert!(
            result.is_err(),
            "truncated tensor name data must fail cleanly, not allocate the full declared length"
        );
    }

    /// A malformed shape whose element-count product overflows `u64` must
    /// saturate to `u64::MAX` (a value that will always fail downstream
    /// bounds checks) rather than silently wrapping to a small, plausible,
    /// but wrong element count.
    #[test]
    fn element_count_overflow_saturates_instead_of_wrapping() {
        let info = TensorInfo {
            name: "overflow".to_string(),
            shape: vec![u64::MAX, 2],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        assert_eq!(info.element_count(), u64::MAX);
    }

    #[test]
    fn element_count_overflow_from_many_large_dims_saturates() {
        // 2^32 * 2^32 already overflows u64.
        let info = TensorInfo {
            name: "overflow2".to_string(),
            shape: vec![1u64 << 32, 1u64 << 32],
            tensor_type: GgufTensorType::Q1_0_g128,
            offset: 0,
        };
        assert_eq!(info.element_count(), u64::MAX);
        // data_size() must not panic, and must stay a huge (implausible for
        // any real file) value derived from the saturated element count —
        // dividing/multiplying it down by the block size/bytes need not
        // land exactly on u64::MAX, but it must never wrap around to a
        // small, in-bounds-looking value that could pass a downstream
        // bounds check.
        let size = info.data_size();
        assert!(
            size > 1_000_000_000_000_000,
            "expected an implausibly large data_size for an overflowing shape, got {size}"
        );
    }

    #[test]
    fn data_size_does_not_overflow_for_realistic_large_tensor() {
        // A large but entirely realistic tensor (e.g. a 32k x 32k weight
        // matrix) must still compute an exact, non-saturated data_size.
        let info = TensorInfo {
            name: "large".to_string(),
            shape: vec![32_768, 32_768],
            tensor_type: GgufTensorType::Q1_0_g128,
            offset: 0,
        };
        let elements = 32_768u64 * 32_768u64;
        let expected_blocks = elements.div_ceil(128);
        assert_eq!(info.data_size(), expected_blocks * 18);
        assert_ne!(info.data_size(), u64::MAX);
    }

    // ── Per-row size formula (core-gguf-N1) ───────────────────────────────

    /// ggml sizes a quantized tensor **per row**. For `[200, 3]` with
    /// `TQ2_0_g128` that is `3 * ceil(200/128) * 34 = 204`; the flattened
    /// `ceil(600/128) * 34` form returns 170 — a 34-byte, one-block-per-row
    /// shortfall that grows linearly with the row count.
    #[test]
    fn data_size_uses_the_ggml_per_row_formula() {
        let info = TensorInfo {
            name: "ragged".to_string(),
            shape: vec![200, 3],
            tensor_type: GgufTensorType::TQ2_0_g128,
            offset: 0,
        };
        assert_eq!(info.data_size(), 204);
        assert_ne!(info.data_size(), 170, "flattened formula must not be used");
    }

    #[test]
    fn data_size_per_row_for_every_quant_family() {
        // (shape, type, expected)
        let cases: &[(&[u64], GgufTensorType, u64)] = &[
            // 1-D: one row.
            (&[256], GgufTensorType::Q1_0_g128, 2 * 18),
            // Block-aligned 2-D: per-row and flattened agree.
            (&[256, 4], GgufTensorType::TQ2_0_g128, 4 * 2 * 34),
            // Ragged ne0 with a 64-wide group.
            (&[100, 2], GgufTensorType::Q2_0G64, 2 * 2 * 18),
            // PTQ1_0, 128-wide group, 28 bytes.
            (&[384, 5], GgufTensorType::PTQ1_0, 5 * 3 * 28),
            // Unquantized types are one element per block.
            (&[7, 3], GgufTensorType::F32, 21 * 4),
            // 0-D scalar.
            (&[], GgufTensorType::F32, 4),
        ];
        for (shape, ty, expected) in cases {
            let info = TensorInfo {
                name: "t".to_string(),
                shape: shape.to_vec(),
                tensor_type: *ty,
                offset: 0,
            };
            assert_eq!(info.data_size(), *expected, "{ty} {shape:?}");
        }
    }

    /// The real 27B `output.weight [5120, 248320]` must size exactly as the
    /// file does: 5120/128 = 40 blocks per row × 34 B × 248320 rows.
    #[test]
    fn data_size_matches_the_real_27b_output_weight() {
        let info = TensorInfo {
            name: "output.weight".to_string(),
            shape: vec![5120, 248_320],
            tensor_type: GgufTensorType::PQ2_0,
            offset: 0,
        };
        assert_eq!(info.data_size(), 248_320 * 40 * 34);
    }

    // ── ne0 % block_size validation ───────────────────────────────────────

    #[test]
    fn parse_rejects_ne0_that_is_not_a_block_multiple() {
        let data = make_tensor_info_bytes("bad.weight", &[100, 2], 42, 0);
        match TensorStore::parse(&data, 0, 1) {
            Err(BonsaiError::TensorLayout { name, reason }) => {
                assert_eq!(name, "bad.weight");
                assert!(reason.contains("100"), "reason: {reason}");
                assert!(reason.contains("128"), "reason: {reason}");
            }
            other => panic!("expected TensorLayout, got {other:?}"),
        }
    }

    #[test]
    fn parse_accepts_ne0_that_is_a_block_multiple() {
        let data = make_tensor_info_bytes("good.weight", &[128, 2], 42, 0);
        let (store, _) = TensorStore::parse(&data, 0, 1).expect("block-aligned ne0 must parse");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn parse_allows_any_ne0_for_unquantized_types() {
        for type_id in [0u32, 1, 30] {
            let data = make_tensor_info_bytes("norm.weight", &[7], type_id, 0);
            TensorStore::parse(&data, 0, 1).expect("block_size == 1 imposes no rule");
        }
    }

    #[test]
    fn zero_dim_tensor_parses_and_sizes_as_a_scalar() {
        let data = make_tensor_info_bytes("scalar", &[], 0, 0);
        let (store, _) = TensorStore::parse(&data, 0, 1).expect("zero-dim must parse");
        let info = store.require("scalar").expect("present");
        assert_eq!(info.n_dims(), 0);
        assert_eq!(info.element_count(), 1);
        assert_eq!(info.ne0(), 1);
        assert_eq!(info.data_size(), 4);
    }

    // ── Dimension cap (core-gguf-16) ──────────────────────────────────────

    #[test]
    fn five_dimensional_tensor_is_rejected() {
        let data = make_tensor_info_bytes("too.many.dims", &[2, 2, 2, 2, 2], 0, 0);
        match TensorStore::parse(&data, 0, 1) {
            Err(BonsaiError::InvalidMetadata { key, reason }) => {
                assert_eq!(key, "too.many.dims");
                assert!(reason.contains("GGML_MAX_DIMS"), "reason: {reason}");
            }
            other => panic!("expected InvalidMetadata for n_dims > 4, got {other:?}"),
        }
    }

    #[test]
    fn four_dimensional_tensor_is_accepted() {
        let data = make_tensor_info_bytes("four.dims", &[2, 2, 2, 2], 0, 0);
        let (store, _) = TensorStore::parse(&data, 0, 1).expect("4-D must parse");
        assert_eq!(store.require("four.dims").expect("present").n_dims(), 4);
    }

    // ── Padding helpers (shared with the writer and the resolver) ─────────

    #[test]
    fn align_up_matches_ggml_pad() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(170, 32), Some(192));
        // Invalid alignments.
        assert_eq!(align_up(4, 0), None);
        assert_eq!(align_up(4, 3), None);
        // Overflow.
        assert_eq!(align_up(u64::MAX, 32), None);
    }

    #[test]
    fn padded_extent_rounds_the_row_size() {
        let info = TensorInfo {
            name: "ragged".to_string(),
            shape: vec![200, 3],
            tensor_type: GgufTensorType::TQ2_0_g128,
            offset: 0,
        };
        assert_eq!(info.data_size(), 204);
        assert_eq!(info.padded_extent(32), Some(224));
        assert_eq!(info.padded_extent(1), Some(204));
        assert_eq!(info.padded_extent(0), None);
    }

    #[test]
    fn sorted_by_offset_is_stable_and_ascending() {
        let mut store = TensorStore::new();
        for (name, offset) in [("c", 64u64), ("a", 0), ("b", 32)] {
            store.tensors.insert(
                name.to_string(),
                TensorInfo {
                    name: name.to_string(),
                    shape: vec![8],
                    tensor_type: GgufTensorType::F32,
                    offset,
                },
            );
        }
        let names: Vec<&str> = store
            .sorted_by_offset()
            .iter()
            .map(|i| i.name.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }
}
