//! Flat, typed weight registry for a FLUX.2 DiT (`bonsai-image`) GGUF file.
//!
//! [`DitWeights`] owns the file bytes (memory-mapped from disk, or an in-memory
//! buffer) and the parsed GGUF metadata/tensor directory, and exposes every
//! tensor with typed access. It deliberately does **not** build the nested
//! transformer-block hierarchy or any forward pass — those belong to a later
//! phase, which can construct block structs on top of this registry.
//!
//! # Two storage conventions
//!
//! The converter writes tensors under two conventions, which the lookups here
//! honour transparently:
//!
//! 1. **Quantized linears** are stored as GGUF type `TQ2_0_g128` under their
//!    *base* module name with the `.weight` suffix stripped, e.g.
//!    `transformer_blocks.0.attn.to_q`. Use [`DitWeights::quantized_linear`].
//! 2. **Plain tensors** are stored as GGUF type `BF16` under their *full*
//!    name, e.g. `x_embedder.weight`. Use [`DitWeights::bf16_tensor`].
//!
//! # Reversed shapes
//!
//! Every tensor is stored with its logical shape reversed (outermost dimension
//! last), so GGUF `ne[0]` is the contraction dimension. The accessors recover
//! the logical shape by reversing `ne`; for a 2-D quantized linear this yields
//! `(out, in) = (ne[1], ne[0])`, exposed as [`QuantizedLinear::out_features`] /
//! [`QuantizedLinear::in_features`].
//!
//! # BF16 exposure
//!
//! BF16 tensors are kept as their raw little-endian bytes (a borrowed
//! [`Bf16Tensor::bytes`] slice, the only true zero-copy view). On top of that,
//! [`Bf16Tensor`] offers two decode-on-demand accessors that each allocate an
//! owned buffer: [`Bf16Tensor::bits`] (the `u16` bit patterns) and
//! [`Bf16Tensor::to_f32_vec`] (decoded `f32` values). Nothing is decoded until
//! a caller asks, so opening the file is cheap.

use std::path::Path;

use half::bf16;

use oxibonsai_core::gguf::metadata::MetadataStore;
use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::{TensorInfo, TensorStore};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::quant_ternary::{BlockTQ2_0_g128, QK_TQ2_0_G128};

use crate::config::DitConfig;
use crate::error::{DitError, DitResult};

/// Suffix used by diffusers `*.weight` linear names; stripped to obtain the
/// base name under which a quantized linear is stored.
const WEIGHT_SUFFIX: &str = ".weight";

/// Backing storage for the GGUF file bytes.
///
/// Owning the bytes (rather than borrowing) lets [`DitWeights`] hold the parsed
/// metadata/tensor directory alongside the data without a self-referential
/// borrow. The `Owned` variant supports in-memory construction (tests, or
/// callers that already have the bytes).
enum Backing {
    /// Memory-mapped file.
    Mmap(memmap2::Mmap),
    /// In-memory byte buffer.
    Owned(Vec<u8>),
}

impl Backing {
    /// Borrow the backing bytes.
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => &m[..],
            Self::Owned(v) => v.as_slice(),
        }
    }
}

/// A quantized (`TQ2_0_g128`) linear weight, exposed as ternary blocks plus its
/// recovered logical `(out, in)` dimensions.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedLinear<'a> {
    /// Out-major ternary blocks (`out * (in / 128)` of them).
    pub blocks: &'a [BlockTQ2_0_g128],
    /// Logical output feature count (rows).
    pub out_features: u64,
    /// Logical input feature count (columns, 128-divisible).
    pub in_features: u64,
}

impl QuantizedLinear<'_> {
    /// Number of ternary blocks expected for this linear: `out * (in / 128)`.
    pub fn expected_block_count(&self) -> u64 {
        self.out_features * (self.in_features / QK_TQ2_0_G128 as u64)
    }
}

/// A plain BF16 tensor, exposed as raw bytes with typed views and a logical
/// (reversed-`ne`) shape.
#[derive(Debug, Clone, Copy)]
pub struct Bf16Tensor<'a> {
    /// Raw little-endian BF16 bytes (`2 * element_count`).
    pub bytes: &'a [u8],
    /// Logical shape (GGUF `ne` reversed, outermost dimension first).
    shape_rev: &'a [u64],
}

impl<'a> Bf16Tensor<'a> {
    /// Logical shape, outermost dimension first (GGUF `ne` reversed).
    pub fn shape(&self) -> Vec<u64> {
        let mut s: Vec<u64> = self.shape_rev.to_vec();
        s.reverse();
        s
    }

    /// Total element count.
    pub fn element_count(&self) -> u64 {
        self.shape_rev.iter().product()
    }

    /// Decoded copy of the raw `u16` BF16 bit patterns (allocates a `Vec`).
    ///
    /// This is not a borrowed view: the bytes are re-read little-endian into an
    /// owned `Vec<u16>`, sidestepping the 2-byte alignment a `&[u16]` cast would
    /// require on memory-mapped data. Returns `None` if the byte length is odd
    /// (never the case for a well-formed BF16 tensor).
    pub fn bits(&self) -> Option<Vec<u16>> {
        if self.bytes.len() % 2 != 0 {
            return None;
        }
        Some(
            self.bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
        )
    }

    /// Decode the tensor to an owned `Vec<f32>` (row-major, logical order).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }
}

/// A flat, typed registry of every tensor in a `bonsai-image` DiT GGUF file,
/// plus the parsed [`DitConfig`].
pub struct DitWeights {
    backing: Backing,
    /// Byte offset where the tensor data section begins.
    data_offset: usize,
    /// Parsed GGUF metadata key-value store (owned).
    metadata: MetadataStore,
    /// Parsed GGUF tensor directory (owned).
    tensors: TensorStore,
    /// Parsed DiT configuration.
    config: DitConfig,
}

impl DitWeights {
    /// Open a `bonsai-image` DiT GGUF file from disk (memory-mapped).
    ///
    /// # Errors
    ///
    /// Returns [`DitError::Io`] if the file cannot be opened/mapped,
    /// [`DitError::Gguf`] on a parse failure, and a config error if the
    /// metadata is not a valid `bonsai-image` architecture.
    pub fn open(path: &Path) -> DitResult<Self> {
        let file = std::fs::File::open(path).map_err(|source| DitError::Io {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: read-only mapping; the file must not be mutated while mapped.
        // This is the standard model-loading pattern used across OxiBonsai.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| DitError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_backing(Backing::Mmap(mmap))
    }

    /// Construct from an in-memory GGUF byte buffer (no temp file needed).
    ///
    /// # Errors
    ///
    /// As [`DitWeights::open`], minus the I/O variant.
    pub fn from_bytes(bytes: Vec<u8>) -> DitResult<Self> {
        Self::from_backing(Backing::Owned(bytes))
    }

    /// Parse a backing buffer into a registry, dropping the transient borrow.
    fn from_backing(backing: Backing) -> DitResult<Self> {
        // Parse against a transient borrow, then move the owned metadata/tensor
        // stores out and drop the borrow so `Self` is not self-referential.
        let (metadata, tensors, data_offset) = {
            let file = GgufFile::parse(backing.as_bytes())?;
            (file.metadata, file.tensors, file.data_offset)
        };
        let config = DitConfig::from_metadata(&metadata)?;
        Ok(Self {
            backing,
            data_offset,
            metadata,
            tensors,
            config,
        })
    }

    /// The parsed DiT configuration.
    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    /// The parsed GGUF metadata store (for keys outside [`DitConfig`]).
    pub fn metadata(&self) -> &MetadataStore {
        &self.metadata
    }

    /// The parsed GGUF tensor directory.
    pub fn tensors(&self) -> &TensorStore {
        &self.tensors
    }

    /// Number of tensors in the file.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Names of all tensors stored as GGUF type `TQ2_0_g128` (quantized),
    /// sorted.
    pub fn quantized_names(&self) -> Vec<&str> {
        self.names_of_type(GgufTensorType::TQ2_0_g128)
    }

    /// Names of all tensors stored as GGUF type `BF16` (plain), sorted.
    pub fn bf16_names(&self) -> Vec<&str> {
        self.names_of_type(GgufTensorType::BF16)
    }

    /// Sorted tensor names whose stored GGUF type equals `ty`.
    fn names_of_type(&self, ty: GgufTensorType) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .tensors
            .iter()
            .filter(|(_, info)| info.tensor_type == ty)
            .map(|(name, _)| name.as_str())
            .collect();
        names.sort_unstable();
        names
    }

    /// Raw bytes of the named tensor in the data section.
    ///
    /// Mirrors `GgufFile::tensor_data`, but against the owned backing.
    fn raw_bytes(&self, info: &TensorInfo) -> DitResult<&[u8]> {
        let bytes = self.backing.as_bytes();
        let start = self.data_offset + info.offset as usize;
        let size = info.data_size() as usize;
        let end = start
            .checked_add(size)
            .ok_or_else(|| DitError::InvalidMetadata {
                key: info.name.clone(),
                reason: "tensor extent overflows usize".to_string(),
            })?;
        if end > bytes.len() {
            return Err(DitError::Gguf(
                oxibonsai_core::error::BonsaiError::UnexpectedEof { offset: end as u64 },
            ));
        }
        Ok(&bytes[start..end])
    }

    /// Look up a quantized (`TQ2_0_g128`) linear by its diffusers logical name.
    ///
    /// Accepts either the base module name (`transformer_blocks.0.attn.to_q`,
    /// the storage convention) or a `.weight`-suffixed name, which is stripped
    /// and retried. Returns the ternary blocks plus the recovered logical
    /// `(out, in)` dimensions.
    ///
    /// # Errors
    ///
    /// [`DitError::Gguf`] (`TensorNotFound`) if no such tensor exists,
    /// [`DitError::WrongTensorType`] if it is not `TQ2_0_g128`,
    /// [`DitError::WrongRank`] if it is not 2-D, or a slice-validation error
    /// from the core ternary block reader.
    pub fn quantized_linear(&self, name: &str) -> DitResult<QuantizedLinear<'_>> {
        let info = self.lookup_quantized_info(name)?;

        if info.tensor_type != GgufTensorType::TQ2_0_g128 {
            return Err(DitError::WrongTensorType {
                name: info.name.clone(),
                found: info.tensor_type.to_string(),
                expected: GgufTensorType::TQ2_0_g128.to_string(),
            });
        }
        if info.shape.len() != 2 {
            return Err(DitError::WrongRank {
                name: info.name.clone(),
                found: info.shape.len(),
                expected: 2,
            });
        }
        // GGUF ne = [in, out]; logical (out, in) = (ne[1], ne[0]).
        let in_features = info.shape[0];
        let out_features = info.shape[1];

        let bytes = self.raw_bytes(info)?;
        let blocks = BlockTQ2_0_g128::slice_from_bytes(bytes)?;

        Ok(QuantizedLinear {
            blocks,
            out_features,
            in_features,
        })
    }

    /// Resolve the [`TensorInfo`] for a quantized linear, honouring the
    /// base-name convention (strip a trailing `.weight` on miss).
    fn lookup_quantized_info(&self, name: &str) -> DitResult<&TensorInfo> {
        if let Some(info) = self.tensors.get(name) {
            return Ok(info);
        }
        if let Some(base) = name.strip_suffix(WEIGHT_SUFFIX) {
            if let Some(info) = self.tensors.get(base) {
                return Ok(info);
            }
        }
        Err(DitError::Gguf(
            oxibonsai_core::error::BonsaiError::TensorNotFound {
                name: name.to_string(),
            },
        ))
    }

    /// Look up a plain BF16 tensor by its full name.
    ///
    /// # Errors
    ///
    /// [`DitError::Gguf`] (`TensorNotFound`) if absent, or
    /// [`DitError::WrongTensorType`] if it is not stored as `BF16`.
    pub fn bf16_tensor(&self, name: &str) -> DitResult<Bf16Tensor<'_>> {
        let info = self.tensors.require(name)?;
        if info.tensor_type != GgufTensorType::BF16 {
            return Err(DitError::WrongTensorType {
                name: info.name.clone(),
                found: info.tensor_type.to_string(),
                expected: GgufTensorType::BF16.to_string(),
            });
        }
        let bytes = self.raw_bytes(info)?;
        Ok(Bf16Tensor {
            bytes,
            shape_rev: &info.shape,
        })
    }
}
