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
    /// Overflow-safe for the same reason as [`element_count`](Self::element_count):
    /// a malformed shape or (in principle) a huge block-byte count saturates
    /// to `u64::MAX` instead of wrapping.
    pub fn data_size(&self) -> u64 {
        let elements = self.element_count();
        let block_size = self.tensor_type.block_size() as u64;
        let block_bytes = self.tensor_type.block_bytes() as u64;
        // Number of blocks = ceil(elements / block_size). `block_size` is a
        // small fixed constant from a closed set of known formats (never
        // zero), so this division cannot fail; `u64::div_ceil` itself cannot
        // overflow internally.
        let num_blocks = elements.div_ceil(block_size);
        num_blocks.saturating_mul(block_bytes)
    }

    /// Number of dimensions.
    pub fn n_dims(&self) -> usize {
        self.shape.len()
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

/// Maximum tensor dimensions (realistically never > 8).
const MAX_TENSOR_DIMS: u32 = 1024;

/// Upper bound on a single eager read chunk (and the initial `Vec`
/// reservation) while reading a declared-length GGUF string.
///
/// Mirrors the identical cap in `metadata.rs`: the declared `len` prefix is
/// attacker-controlled and only bounded against [`MAX_STRING_LEN`] (256
/// MiB), so allocating `len` bytes up front — before confirming the reader
/// actually has that much data left — lets a tiny file force a large
/// allocation. Reading in bounded chunks instead keeps peak allocation
/// proportional to bytes actually confirmed present.
const STRING_READ_CHUNK: usize = 64 * 1024;

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

/// Read a single tensor info entry from the reader.
fn read_tensor_info<R: std::io::Read>(reader: &mut R) -> BonsaiResult<TensorInfo> {
    let name = read_gguf_string(reader)?;

    let n_dims = reader
        .read_u32::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if n_dims > MAX_TENSOR_DIMS {
        return Err(BonsaiError::InvalidString { offset: 0 });
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
}
