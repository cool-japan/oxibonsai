//! Production-quality GGUF model loader with validation, streaming, and memory budgeting.
//!
//! This module provides high-level utilities for loading GGUF model files with:
//! - Configurable memory budgets and validation strictness
//! - Lazy tensor metadata loading (no weight data read upfront)
//! - Streaming chunk iterators for progressive loading
//! - Memory footprint estimation before committing to a full load
//!
//! # One parser, not two
//!
//! Every function here delegates to [`oxibonsai_core::gguf::reader::GgufFile`]
//! rather than re-implementing the GGUF binary format. The module used to
//! carry its own byte-level parser whose `*pos + n` bounds checks could wrap
//! and whose unknown-type fallback silently assumed one byte per element
//! (sec-04); two independently-hardened parsers is the root cause, and it
//! would have drifted again the moment ids 142/143 landed. The core reader
//! also mmaps instead of `read_to_end`, so opening a 7.2 GB model to read its
//! header no longer materialises the whole file.

use std::path::Path;

use oxibonsai_core::error::BonsaiError;
use oxibonsai_core::gguf::quant_resolve::{
    compute_extents, resolve_type_42_with_sample, AMBIGUOUS_TYPE_ID,
};
use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_core::gguf::tensor_info::{row_size_bytes, tensor_names, TensorInfo};
use oxibonsai_core::gguf::types::GgufTensorType;
use oxibonsai_core::quant_ternary::{sniff_sample_byte_cap, SNIFF_DEFAULT_BLOCKS};

// ─────────────────────────────────────────────────────────────────────────────
// LoadError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during GGUF model loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// An underlying I/O error (e.g., file not found, permission denied).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The GGUF file could not be parsed (malformed binary).
    #[error("GGUF parse error: {0}")]
    Parse(String),

    /// Loading this file would exceed the configured memory budget.
    #[error("memory budget exceeded: need {need} bytes, budget {budget} bytes")]
    MemoryBudgetExceeded { need: u64, budget: u64 },

    /// The GGUF version in the file header is not supported.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// A required structural invariant was violated.
    #[error("validation failed: {0}")]
    ValidationFailed(String),
}

/// Map a core error onto a [`LoadError`], preserving the I/O variant so
/// callers can still distinguish "file missing" from "file malformed".
fn core_err(err: BonsaiError) -> LoadError {
    match err {
        BonsaiError::MmapError(io) => LoadError::Io(io),
        BonsaiError::UnsupportedVersion { version } => LoadError::UnsupportedVersion(version),
        other => LoadError::Parse(other.to_string()),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration governing how a GGUF model is loaded.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Maximum memory (in bytes) the loader is allowed to consume; `None` = unlimited.
    pub max_memory_bytes: Option<usize>,
    /// Whether to validate file-level checksums (currently advisory).
    pub validate_checksums: bool,
    /// If `true`, tensors with quantisation types this build cannot execute
    /// are reported as warnings; if `false`, they are a hard validation error.
    pub allow_unknown_quant_types: bool,
    /// Size of each streaming chunk in bytes when using [`TensorChunkIter`].
    pub streaming_chunk_size: usize,
    /// If `true`, reject GGUF files that declare an unsupported version.
    pub strict_version: bool,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: None,
            validate_checksums: false,
            allow_unknown_quant_types: true,
            streaming_chunk_size: 4 * 1024 * 1024, // 4 MiB
            strict_version: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics gathered during a model loading operation.
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Number of tensors successfully loaded.
    pub tensors_loaded: usize,
    /// Total bytes of tensor weight data loaded.
    pub bytes_loaded: u64,
    /// Tensors skipped because their quantisation type was unrecognised.
    pub skipped_tensors: usize,
    /// Wall-clock time for the entire load operation in milliseconds.
    pub load_time_ms: u64,
    /// Approximate peak memory usage (bytes) during loading.
    pub peak_memory_bytes: usize,
    /// Non-fatal issues found during validation (empty = clean).
    pub validation_warnings: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorEntry
// ─────────────────────────────────────────────────────────────────────────────

/// Quantisation type IDs this build can decode, and their display names.
///
/// Derived from [`GgufTensorType`] rather than re-listed, so it can never
/// drift from the decode table. `142` (`PQ2_0`), `143` (`PTQ1_0`) and the
/// group-64 reading of `42` are included because
/// [`GgufTensorType::is_executable`] reports them.
pub fn known_quant_types() -> Vec<(u32, &'static str)> {
    let mut out: Vec<(u32, &'static str)> = GgufTensorType::ALL
        .iter()
        .filter(|ty| ty.is_executable())
        .map(|ty| (ty.wire_id(), ty.name()))
        .collect();
    out.sort_by_key(|&(id, name)| (id, name));
    out.dedup_by_key(|&mut (id, _)| id);
    out
}

/// A loaded tensor entry — contains metadata only; no weight bytes are held here.
/// Use [`load_tensor_metadata`] to obtain a collection of these, then open the
/// file and seek to `offset` to read the actual data.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Tensor name as stored in the GGUF file (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape dimensions (e.g. `[4096, 4096]`).
    pub shape: Vec<u64>,
    /// Raw GGUF quantisation type ID.
    pub quant_type_id: u32,
    /// Byte offset of this tensor's data from the start of the tensor data section.
    pub offset: u64,
    /// Number of bytes occupied by this tensor in the data section.
    pub size_bytes: u64,
}

impl TensorEntry {
    /// Total number of elements across all dimensions.
    ///
    /// Shape dimensions come from an untrusted file, so the product is
    /// computed with `checked_mul` and saturates to `u64::MAX` rather than
    /// wrapping to a small, plausible-looking count in a release build
    /// (`overflow-checks` is off by default).
    pub fn element_count(&self) -> u64 {
        self.shape
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .unwrap_or(u64::MAX)
    }

    /// Human-readable name for the quantisation type, or `"UNKNOWN"`.
    pub fn quant_name(&self) -> &'static str {
        match GgufTensorType::from_id(self.quant_type_id) {
            Ok(ty) => ty.name(),
            Err(_) => "UNKNOWN",
        }
    }

    /// Returns `true` when the quantisation type ID is one OxiBonsai can decode.
    pub fn is_known_quant(&self) -> bool {
        GgufTensorType::from_id(self.quant_type_id)
            .map(|ty| ty.is_executable())
            .unwrap_or(false)
    }

    /// The parsed tensor type, or `None` when the id is not a ggml type.
    pub fn tensor_type(&self) -> Option<GgufTensorType> {
        GgufTensorType::from_id(self.quant_type_id).ok()
    }
}

/// A tensor paired with the quantisation variant that was actually **resolved**
/// for it, which is what decode selection must key on.
///
/// For ggml id 42 the raw id does not determine a layout at all — see
/// [`oxibonsai_core::gguf::quant_resolve`].
#[derive(Debug, Clone)]
pub struct ResolvedTensor {
    /// Name, shape, raw id, offset and byte size.
    pub entry: TensorEntry,
    /// The resolved variant. Its `wire_id()` is still the raw on-disk id.
    pub tensor_type: GgufTensorType,
}

// ─────────────────────────────────────────────────────────────────────────────
// Size computation
// ─────────────────────────────────────────────────────────────────────────────

/// ggml's per-row byte size for a tensor of `shape` stored as `tensor_type`.
///
/// `nrows * ceil(ne0 / block_size) * block_bytes`. Rows are quantized
/// independently, so the flattened `ceil(total / block_size)` form
/// under-counts whenever `ne0` is not a block multiple.
pub fn tensor_size_bytes(tensor_type: GgufTensorType, shape: &[u64]) -> u64 {
    row_size_bytes(tensor_type, shape)
}

/// Compute the byte size of a tensor given its shape and raw quant type ID.
///
/// Unknown ids return `None` instead of the old one-byte-per-element guess,
/// which silently under-reported every real quantized tensor it was applied
/// to (sec-04).
pub fn compute_tensor_size_bytes(shape: &[u64], quant_type_id: u32) -> Option<u64> {
    GgufTensorType::from_id(quant_type_id)
        .ok()
        .map(|ty| row_size_bytes(ty, shape))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Read a GGUF file's header and tensor directory through the core reader,
/// handing the parsed view to `f`.
fn with_gguf<T, F>(path: &Path, f: F) -> Result<T, LoadError>
where
    F: FnOnce(&GgufFile<'_>) -> Result<T, LoadError>,
{
    let mmap = mmap_gguf_file(path).map_err(core_err)?;
    let file = GgufFile::parse(&mmap).map_err(core_err)?;
    f(&file)
}

/// Project one parsed tensor info onto the loader's public entry type.
fn entry_from(info: &TensorInfo) -> TensorEntry {
    TensorEntry {
        name: info.name.clone(),
        shape: info.shape.clone(),
        quant_type_id: info.tensor_type.wire_id(),
        offset: info.offset,
        size_bytes: info.data_size(),
    }
}

/// Collect the tensor directory in declaration-independent (offset) order.
fn entries_from(file: &GgufFile<'_>) -> Vec<TensorEntry> {
    file.tensors
        .sorted_by_offset()
        .into_iter()
        .map(entry_from)
        .collect()
}

/// Validates a GGUF file at `path`, checking:
/// - File exists and is readable
/// - Magic bytes are correct
/// - Version is in the supported set
/// - Tensor count and metadata are self-consistent
///
/// Returns a (possibly empty) list of advisory warning strings.
/// An empty list means the file passed all checks.
pub fn validate_gguf_file(path: &Path) -> Result<Vec<String>, LoadError> {
    validate_gguf_file_with(path, &LoadConfig::default())
}

/// [`validate_gguf_file`] with an explicit configuration.
///
/// When `config.allow_unknown_quant_types` is `false`, a tensor whose type
/// this build cannot execute is a hard [`LoadError::ValidationFailed`]
/// instead of a warning.
pub fn validate_gguf_file_with(path: &Path, config: &LoadConfig) -> Result<Vec<String>, LoadError> {
    with_gguf(path, |file| {
        let mut warnings = Vec::new();

        const SUPPORTED_VERSIONS: &[u32] = &[2, 3];
        if !SUPPORTED_VERSIONS.contains(&file.header.version) {
            if config.strict_version {
                return Err(LoadError::UnsupportedVersion(file.header.version));
            }
            warnings.push(format!(
                "GGUF version {} is not in the officially supported set {:?}",
                file.header.version, SUPPORTED_VERSIONS
            ));
        }

        if file.tensors.is_empty() {
            warnings.push("file contains zero tensors".to_string());
        }

        for (name, info) in file.tensors.iter() {
            if !info.tensor_type.is_executable() {
                let message = format!(
                    "tensor '{name}' uses quantisation type {} (id {}), which this build can \
                     parse but not execute",
                    info.tensor_type.name(),
                    info.tensor_type.wire_id()
                );
                if config.allow_unknown_quant_types {
                    warnings.push(message);
                } else {
                    return Err(LoadError::ValidationFailed(message));
                }
            }
            if info.shape.is_empty() {
                warnings.push(format!("tensor '{name}' has zero-dimensional shape"));
            }
        }

        warnings.sort();
        Ok(warnings)
    })
}

/// Loads tensor metadata (names, shapes, types, offsets) from a GGUF file.
///
/// This is intentionally fast — the file is mmapped and only its header and
/// tensor directory are touched; no weight bytes are read.
pub fn load_tensor_metadata(path: &Path) -> Result<Vec<TensorEntry>, LoadError> {
    with_gguf(path, |file| Ok(entries_from(file)))
}

/// Loads tensor metadata **and** resolves each tensor's quantisation variant.
///
/// ggml id 42 has three incompatible on-disk layouts, so decode selection
/// must key on the resolved variant, never the raw id. The group size comes
/// from replaying llama.cpp's offset invariant and the byte order from a
/// structural sniff of the first type-42 tensor's real bytes.
pub fn load_tensor_metadata_resolved(path: &Path) -> Result<Vec<ResolvedTensor>, LoadError> {
    with_gguf(path, |file| {
        // Sort ONCE. Deriving the entry list and the info list from two
        // independent `sorted_by_offset()` calls would leave two vectors that
        // nothing forces to stay index-aligned, and a divergence would hand
        // every tensor some *other* tensor's resolved type — silently.
        let infos = file.tensors.sorted_by_offset();
        let has_42 = infos
            .iter()
            .any(|i| i.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID);

        let resolved_42 = if has_42 {
            let owned: Vec<_> = infos.iter().map(|i| (*i).clone()).collect();
            let data_len = (file.data.len() as u64).saturating_sub(file.data_offset as u64);
            let extents = compute_extents(&owned, data_len).map_err(core_err)?;
            let alignment = file
                .metadata
                .get("general.alignment")
                .and_then(|v| v.as_u32())
                .unwrap_or(32) as u64;
            let sample_name = owned
                .iter()
                .find(|i| i.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID)
                .map(|i| i.name.clone())
                .unwrap_or_default();
            let sample = file.tensor_data(&sample_name).map_err(core_err)?;
            // 2000 blocks of the widest candidate is plenty; the real files
            // separate by hundreds of illegal codes well before that. The cap
            // itself must stay a multiple of every candidate's block size
            // (34 AND 18 bytes), or truncation alone can disqualify a clean
            // group-64 reading via `LayoutScore::length_compatible` — see
            // `sniff_sample_byte_cap`.
            let sample = &sample[..sample
                .len()
                .min(sniff_sample_byte_cap(SNIFF_DEFAULT_BLOCKS))];
            Some(
                resolve_type_42_with_sample(
                    &owned,
                    alignment,
                    file.metadata.get("general.quantization_version"),
                    Some(&extents),
                    sample,
                )
                .map_err(core_err)?,
            )
        } else {
            None
        };

        let mut out = Vec::with_capacity(infos.len());
        for info in &infos {
            let tensor_type = if info.tensor_type.wire_id() == AMBIGUOUS_TYPE_ID {
                match resolved_42 {
                    Some(r) => r.tensor_type,
                    None => info.tensor_type,
                }
            } else {
                info.tensor_type
            };
            // `entry_from` sizes from `info.tensor_type` — the PARSE-TIME
            // guess, which for wire id 42 is always the legacy
            // `TQ2_0_g128` (128 elements / 34 bytes) reading regardless of
            // what was just resolved above. That guess is byte-identical to
            // `Q2_0G128DFirst` (also 128/34), but disagrees with
            // `Q2_0G64` (64 elements / 18 bytes): overwrite with the
            // RESOLVED type's own per-row size so a group-64 tensor is never
            // handed a group-128 byte count.
            let mut entry = entry_from(info);
            entry.size_bytes = row_size_bytes(tensor_type, &entry.shape);
            out.push(ResolvedTensor { entry, tensor_type });
        }
        Ok(out)
    })
}

/// Quantisation types [`EmbeddingTable::from_gguf`][embd]
/// (`oxibonsai-model/src/model/types/embedding.rs`, not owned by this
/// package) can decode **row-wise**, keeping `token_embd.weight` resident in
/// its on-disk quantized form (only the looked-up row is ever dequantized).
/// Every other type falls back to eager, whole-table dequantization to
/// `f32` — see [`token_embd_resident_bytes`].
///
/// Mirrors that file's own doc table by hand (it is not owned by this
/// package, so there is no shared predicate to call instead): `Q1_0_g128`,
/// `TQ2_0`, `TQ2_0_g128`, the `d`-first group-128/-64 readings of ggml id
/// 42, and PrismML's `PQ2_0`/`PTQ1_0`. Keep in sync if that file's row-wise
/// set ever changes — a real risk this comment names explicitly rather than
/// hiding.
///
/// [embd]: https://docs.rs/oxibonsai-model (module `model::types::embedding`)
fn token_embd_is_row_wise_decodable(ty: GgufTensorType) -> bool {
    matches!(
        ty,
        GgufTensorType::Q1_0_g128
            | GgufTensorType::TQ2_0
            | GgufTensorType::TQ2_0_g128
            | GgufTensorType::Q2_0G64
            | GgufTensorType::Q2_0G128DFirst
            | GgufTensorType::PQ2_0
            | GgufTensorType::PTQ1_0
    )
}

/// Honest resident-memory estimate for `token_embd.weight` alone: its
/// on-disk size when a row-wise decoder exists for its layout (the common,
/// memory-efficient case), or its full `element_count * 4`-byte
/// f32-dequantized size when it does not (the eager-dequant fallback M-02's
/// `EmbeddingTable::from_gguf` takes — this is the "eager-embedding" term
/// the M-07 review says [`estimate_memory_bytes`] was missing).
///
/// `None` when `token_embd.weight` is absent from `entries`, its type id is
/// not a recognised [`GgufTensorType`], or its f32-equivalent size would
/// overflow `u64` — each left to the caller to fall back to the naive sum
/// rather than silently reporting a wrong-but-plausible number.
fn token_embd_resident_bytes(entries: &[TensorEntry]) -> Option<u64> {
    let embd = entries
        .iter()
        .find(|e| e.name == tensor_names::TOKEN_EMBD)?;
    let ty = embd.tensor_type()?;
    if token_embd_is_row_wise_decodable(ty) {
        Some(embd.size_bytes)
    } else {
        embd.element_count().checked_mul(4)
    }
}

/// Computes the expected memory footprint (in bytes) for fully loading all
/// tensor weight data from the given GGUF file.
///
/// This reads only the file header and tensor metadata — no weight bytes.
///
/// # Honesty (M-07 / sec-17)
///
/// The naive sum of every [`TensorEntry::size_bytes`] understates the real
/// figure by exactly one term this build's own loader already knows about:
/// `token_embd.weight`'s **resident** cost (see
/// `token_embd_resident_bytes`) whenever no row-wise decoder exists for
/// its layout and it is dequantized eagerly to `f32` instead of staying
/// quantized-resident. This function adds that difference on top of the
/// naive sum so a caller building the RAM-derived context guard
/// (`oxibonsai_runtime::config::max_context_for_budget`'s `weight_bytes`
/// input) is not fed a number this build already knows to be wrong for that
/// case. For Bonsai 2 specifically the correction is a no-op — its
/// `token_embd.weight` is `PQ2_0`/`PTQ1_0`, both row-wise-decodable — but
/// this function is not Bonsai-2-specific.
///
/// This estimate still does **not** model the KV cache or recurrent-state
/// memory a running model additionally allocates — those scale with the
/// requested context length and architecture respectively, which is exactly
/// what `max_context_for_budget` solves for; folding a KV estimate in here
/// would double-count against that formula's own `weight_bytes` term.
///
/// # ggml id 42 and this estimate
///
/// This sums [`TensorEntry::size_bytes`] from the **unresolved**
/// [`load_tensor_metadata`], so any wire-id-42 tensor is sized from
/// [`GgufTensorType::from_id`]'s parse-time guess (the legacy qs-first
/// group-128 reading) rather than the file's actual resolved geometry. That
/// guess is byte-identical to the `d`-first group-128 reading — every real
/// GGUF this build has seen — but would under-count a genuine group-64 file
/// (18 bytes/64 weights vs the assumed 34/128). Resolving id 42 needs a byte
/// sample the way [`load_tensor_metadata_resolved`] takes one internally,
/// which this metadata-only, sample-free function cannot provide without
/// changing its contract (and calling [`load_tensor_metadata_resolved`]'s
/// sample-free sibling here would newly hard-error on every numeric-`qver`
/// PrismML file, trading an under-estimate for an outage). A caller that
/// must budget accurately for a file that might be group-64 should sum
/// [`ResolvedTensor::entry`]'s `size_bytes` from
/// [`load_tensor_metadata_resolved`] instead — its sizes are correct for
/// every resolved variant.
pub fn estimate_memory_bytes(path: &Path) -> Result<u64, LoadError> {
    let entries = load_tensor_metadata(path)?;
    let mut total = 0u64;
    for entry in &entries {
        total = total.saturating_add(entry.size_bytes);
    }
    if let Some(resident) = token_embd_resident_bytes(&entries) {
        let on_disk = entries
            .iter()
            .find(|e| e.name == tensor_names::TOKEN_EMBD)
            .map(|e| e.size_bytes)
            .unwrap_or(0);
        total = total.saturating_add(resident.saturating_sub(on_disk));
    }
    Ok(total)
}

/// Returns `true` when the GGUF file at `path` fits within `budget_bytes`.
///
/// Identical to calling [`estimate_memory_bytes`] and comparing.
pub fn fits_in_budget(path: &Path, budget_bytes: u64) -> Result<bool, LoadError> {
    let need = estimate_memory_bytes(path)?;
    Ok(need <= budget_bytes)
}

/// [`fits_in_budget`] as a hard check, returning the budget error.
pub fn require_budget(path: &Path, budget_bytes: u64) -> Result<u64, LoadError> {
    let need = estimate_memory_bytes(path)?;
    if need > budget_bytes {
        return Err(LoadError::MemoryBudgetExceeded {
            need,
            budget: budget_bytes,
        });
    }
    Ok(need)
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming iterator
// ─────────────────────────────────────────────────────────────────────────────

/// An iterator that yields successive fixed-size byte chunks from a tensor's
/// raw data buffer.  Use this for progressive / streaming loading.
///
/// ```
/// # use oxibonsai_model::gguf_loader::TensorChunkIter;
/// let data = vec![0u8; 100];
/// let mut iter = TensorChunkIter::new(data, 32);
/// assert_eq!(iter.total_chunks(), 4); // ceil(100/32)
/// ```
pub struct TensorChunkIter {
    data: Vec<u8>,
    chunk_size: usize,
    pos: usize,
}

impl TensorChunkIter {
    /// Create a new chunk iterator over `data` with the given `chunk_size`.
    ///
    /// # Panics
    ///
    /// Panics when `chunk_size == 0`, which would never terminate. Use
    /// [`TensorChunkIter::try_new`] to get an error instead.
    pub fn new(data: Vec<u8>, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        Self {
            data,
            chunk_size,
            pos: 0,
        }
    }

    /// Fallible [`TensorChunkIter::new`].
    pub fn try_new(data: Vec<u8>, chunk_size: usize) -> Result<Self, LoadError> {
        if chunk_size == 0 {
            return Err(LoadError::ValidationFailed(
                "streaming chunk_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            data,
            chunk_size,
            pos: 0,
        })
    }

    /// Total number of chunks (rounded up for any partial final chunk).
    pub fn total_chunks(&self) -> usize {
        if self.data.is_empty() {
            return 0;
        }
        self.data.len().div_ceil(self.chunk_size)
    }

    /// Remaining bytes not yet yielded by the iterator.
    pub fn bytes_remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

impl Iterator for TensorChunkIter {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let end = self
            .pos
            .saturating_add(self.chunk_size)
            .min(self.data.len());
        let chunk = self.data.get(self.pos..end)?.to_vec();
        self.pos = end;
        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_quant_types_include_the_prism_ids() {
        let known = known_quant_types();
        for id in [0u32, 1, 30, 41, 42, 142, 143] {
            assert!(
                known.iter().any(|&(k, _)| k == id),
                "quant type id {id} must be known"
            );
        }
        // The group-64 reading of 42 shares its wire id with TQ2_0_g128.
        assert_eq!(
            known.iter().filter(|&&(k, _)| k == 42).count(),
            1,
            "id 42 must appear once"
        );
        // Types the parser knows but cannot execute must not be listed.
        for id in [16u32, 23, 34, 39, 40] {
            assert!(
                !known.iter().any(|&(k, _)| k == id),
                "non-executable type {id} must not be listed as known"
            );
        }
    }

    #[test]
    fn size_uses_the_per_row_formula_for_the_new_types() {
        assert_eq!(
            compute_tensor_size_bytes(&[200, 3], 42),
            Some(204),
            "TQ2_0_g128 [200,3] must be 3 rows × 2 blocks × 34 B"
        );
        assert_eq!(compute_tensor_size_bytes(&[128, 4], 142), Some(4 * 34));
        assert_eq!(compute_tensor_size_bytes(&[128, 4], 143), Some(4 * 28));
        assert_eq!(
            tensor_size_bytes(GgufTensorType::Q2_0G64, &[128, 4]),
            4 * 2 * 18
        );
    }

    /// The old fallback assumed one byte per element for an unknown id, which
    /// silently under-reported every real tensor it touched.
    #[test]
    fn unknown_quant_id_has_no_size_instead_of_a_wrong_guess() {
        assert_eq!(compute_tensor_size_bytes(&[128, 4], 9999), None);
    }

    #[test]
    fn element_count_saturates_instead_of_wrapping() {
        let entry = TensorEntry {
            name: "overflow".to_string(),
            shape: vec![u64::MAX, 2],
            quant_type_id: 0,
            offset: 0,
            size_bytes: 0,
        };
        assert_eq!(entry.element_count(), u64::MAX);
    }

    #[test]
    fn try_new_rejects_zero_chunk_size() {
        assert!(TensorChunkIter::try_new(vec![0u8; 4], 0).is_err());
        assert!(TensorChunkIter::try_new(vec![0u8; 4], 4).is_ok());
    }

    #[test]
    fn chunk_iter_does_not_overflow_on_a_huge_chunk_size() {
        let mut iter = TensorChunkIter::new(vec![1u8, 2, 3], usize::MAX);
        assert_eq!(iter.next(), Some(vec![1, 2, 3]));
        assert_eq!(iter.next(), None);
    }

    /// Write a synthetic GGUF whose id-42 tensors carry `d`-first 34-byte
    /// blocks, interleaved with f32 tensors, and check that every entry gets
    /// **its own** resolved type.
    ///
    /// This is the regression guard for feeding decode selection from two
    /// separately-sorted lists: a divergence would hand each tensor some
    /// other tensor's variant, which no size or shape assertion could catch.
    #[test]
    fn resolved_types_are_paired_with_the_right_tensor() {
        use oxibonsai_core::gguf::writer::{
            GgufWriter, MetadataWriteValue, TensorEntry as WriteEntry, TensorType,
        };

        // `d` first, then 32 code bytes that are legal ternary under that
        // reading but a negative f16 under any other.
        let block = |out: &mut Vec<u8>| {
            let d = half::f16::from_f32(0.0415).to_le_bytes();
            out.push(d[0]);
            out.push(d[1]);
            out.extend(std::iter::repeat_n(0b10_01_00_01u8, 32));
        };
        let quant_data = |rows: usize| {
            let mut v = Vec::with_capacity(rows * 34);
            for _ in 0..rows {
                block(&mut v);
            }
            v
        };

        let mut w = GgufWriter::new();
        w.add_metadata("general.quantization_version", MetadataWriteValue::U32(2));
        // Deliberately interleaved and declared out of alphabetical order.
        w.add_tensor(WriteEntry {
            name: "blk.1.ffn_up.weight".to_string(),
            shape: vec![128, 512],
            tensor_type: TensorType::TQ2_0_g128,
            data: quant_data(512),
        });
        w.add_tensor(WriteEntry {
            name: "blk.0.attn_norm.weight".to_string(),
            shape: vec![128],
            tensor_type: TensorType::F32,
            data: vec![0u8; 512],
        });
        w.add_tensor(WriteEntry {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![128, 256],
            tensor_type: TensorType::TQ2_0_g128,
            data: quant_data(256),
        });
        w.add_tensor(WriteEntry {
            name: "output_norm.weight".to_string(),
            shape: vec![128],
            tensor_type: TensorType::F32,
            data: vec![0u8; 512],
        });
        let bytes = w.to_bytes().expect("write synthetic gguf");

        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_gguf_loader_resolved_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("synthetic.gguf");
        std::fs::write(&path, &bytes).expect("write temp gguf");

        let resolved = load_tensor_metadata_resolved(&path).expect("resolve");
        assert_eq!(resolved.len(), 4);
        for r in &resolved {
            let expected = match r.entry.name.as_str() {
                "blk.0.attn_norm.weight" | "output_norm.weight" => GgufTensorType::F32,
                _ => GgufTensorType::Q2_0G128DFirst,
            };
            assert_eq!(
                r.tensor_type, expected,
                "tensor '{}' got the wrong resolved type",
                r.entry.name
            );
            // The resolved variant must also describe the entry's own size.
            assert_eq!(
                r.entry.size_bytes,
                tensor_size_bytes(r.tensor_type, &r.entry.shape),
                "tensor '{}' size disagrees with its resolved type",
                r.entry.name
            );
        }
        // …and the raw id stays 42 on the wire for the quantized ones.
        for r in resolved
            .iter()
            .filter(|r| r.entry.name.ends_with(".weight"))
        {
            if r.tensor_type == GgufTensorType::Q2_0G128DFirst {
                assert_eq!(r.entry.quant_type_id, 42);
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Sibling of [`resolved_types_are_paired_with_the_right_tensor`] for the
    /// mainline group-64 reading.
    ///
    /// This is its own file rather than a case added to that test's writer:
    /// `resolve_type_42`'s group size is settled once per FILE by replaying
    /// the offset table over every id-42 tensor at once, so mixing a
    /// genuinely 34-byte tensor and a genuinely 18-byte tensor under the
    /// same id 42 in one file would make BOTH candidate geometries fail the
    /// replay (`settle_group` returns zero survivors), which is not the
    /// scenario this test needs.
    ///
    /// Pins the exact numbers the size-mismatch bug produced on a real
    /// group-64 tensor: `[128, 256]` must be 9216 bytes (256 rows × 2 blocks
    /// × 18 B), not 8704 (the group-128 parse-time guess's 256 × 1 × 34);
    /// `[128, 8192]` must be 294912, not 278528.
    #[test]
    fn resolved_group_64_types_get_their_own_size_not_the_parse_time_guess() {
        use oxibonsai_core::gguf::writer::{
            GgufWriter, MetadataWriteValue, TensorEntry as WriteEntry, TensorType,
        };

        // `d` first, then 16 code bytes that are legal ternary under the
        // group-64 reading (the only byte order a group-64 file has).
        let block = |out: &mut Vec<u8>| {
            let d = half::f16::from_f32(0.0415).to_le_bytes();
            out.push(d[0]);
            out.push(d[1]);
            out.extend(std::iter::repeat_n(0b10_01_00_01u8, 16));
        };
        // shape `[128, rows]`: ne0 = 128 is 2 blocks of 64, so `rows` rows
        // need `rows * 2` blocks of 18 bytes each.
        let quant_data = |rows: usize| {
            let mut v = Vec::with_capacity(rows * 2 * 18);
            for _ in 0..rows * 2 {
                block(&mut v);
            }
            v
        };

        let mut w = GgufWriter::new();
        w.add_metadata("general.quantization_version", MetadataWriteValue::U32(2));
        w.add_tensor(WriteEntry {
            name: "blk.0.attn_q.weight".to_string(),
            shape: vec![128, 256],
            tensor_type: TensorType::Q2_0G64,
            data: quant_data(256),
        });
        w.add_tensor(WriteEntry {
            name: "blk.0.attn_norm.weight".to_string(),
            shape: vec![128],
            tensor_type: TensorType::F32,
            data: vec![0u8; 512],
        });
        w.add_tensor(WriteEntry {
            name: "blk.1.ffn_up.weight".to_string(),
            shape: vec![128, 8192],
            tensor_type: TensorType::Q2_0G64,
            data: quant_data(8192),
        });
        let bytes = w.to_bytes().expect("write synthetic gguf");

        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_gguf_loader_resolved_g64_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("synthetic_g64.gguf");
        std::fs::write(&path, &bytes).expect("write temp gguf");

        let resolved = load_tensor_metadata_resolved(&path).expect("resolve");
        assert_eq!(resolved.len(), 3);
        for r in &resolved {
            let expected = if r.entry.name == "blk.0.attn_norm.weight" {
                GgufTensorType::F32
            } else {
                GgufTensorType::Q2_0G64
            };
            assert_eq!(
                r.tensor_type, expected,
                "tensor '{}' got the wrong resolved type",
                r.entry.name
            );
            // The resolved variant must also describe the entry's own size —
            // NOT the parse-time (group-128) guess `entry_from` starts from.
            assert_eq!(
                r.entry.size_bytes,
                tensor_size_bytes(r.tensor_type, &r.entry.shape),
                "tensor '{}' size disagrees with its resolved type",
                r.entry.name
            );
        }
        let attn_q = resolved
            .iter()
            .find(|r| r.entry.name == "blk.0.attn_q.weight")
            .expect("attn_q present");
        assert_eq!(
            attn_q.entry.size_bytes, 9216,
            "256 rows * 2 blocks * 18 B, not the group-128 guess's 8704"
        );
        let ffn_up = resolved
            .iter()
            .find(|r| r.entry.name == "blk.1.ffn_up.weight")
            .expect("ffn_up present");
        assert_eq!(
            ffn_up.entry.size_bytes, 294912,
            "8192 rows * 2 blocks * 18 B, not the group-128 guess's 278528"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The production resolved path, end to end on a real model when one is
    /// present. Model weights are not in the repo, so this skips on a fresh
    /// clone; it ran live this session against `Ternary-Bonsai-1.7B.gguf`.
    #[test]
    fn real_legacy_model_resolves_to_the_qs_first_reading() {
        let Some(path) = oxibonsai_testkit::workspace::find_model("Ternary-Bonsai-1.7B.gguf")
        else {
            eprintln!(
                "skipping: Ternary-Bonsai-1.7B.gguf not present under {}",
                oxibonsai_testkit::workspace::models_dir().display()
            );
            return;
        };
        let resolved = load_tensor_metadata_resolved(&path).expect("resolve real model");
        assert!(!resolved.is_empty());
        let quantized: Vec<_> = resolved
            .iter()
            .filter(|r| r.entry.quant_type_id == 42)
            .collect();
        assert!(!quantized.is_empty(), "the 1.7B is a type-42 model");
        for r in &quantized {
            assert_eq!(
                r.tensor_type,
                GgufTensorType::TQ2_0_g128,
                "tensor '{}' must keep the legacy qs-first reading",
                r.entry.name
            );
            assert_eq!(
                r.entry.size_bytes,
                tensor_size_bytes(r.tensor_type, &r.entry.shape)
            );
        }
        // Unquantized tensors must not be touched by the id-42 resolution.
        for r in resolved.iter().filter(|r| r.entry.quant_type_id == 0) {
            assert_eq!(r.tensor_type, GgufTensorType::F32);
        }
    }

    // ── estimate_memory_bytes honesty (M-07 / sec-17) ──

    /// Write a minimal synthetic GGUF whose `token_embd.weight` uses
    /// `tensor_type`, plus one small unrelated F32 tensor so the estimate
    /// isn't trivially just the embedding, and return the path (caller must
    /// remove the parent directory when done).
    fn write_synthetic_embd_gguf(
        tensor_type: oxibonsai_core::gguf::writer::TensorType,
        embd_data: Vec<u8>,
        embd_shape: Vec<u64>,
        tag: &str,
    ) -> std::path::PathBuf {
        use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry as WriteEntry};

        let mut w = GgufWriter::new();
        w.add_tensor(WriteEntry {
            name: "token_embd.weight".to_string(),
            shape: embd_shape,
            tensor_type,
            data: embd_data,
        });
        w.add_tensor(WriteEntry {
            name: "output_norm.weight".to_string(),
            shape: vec![8],
            tensor_type: oxibonsai_core::gguf::writer::TensorType::F32,
            data: vec![0u8; 32],
        });
        let bytes = w.to_bytes().expect("write synthetic gguf");

        let dir = std::env::temp_dir().join(format!(
            "oxibonsai_gguf_loader_estimate_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("synthetic.gguf");
        std::fs::write(&path, &bytes).expect("write temp gguf");
        path
    }

    #[test]
    fn estimate_memory_bytes_adds_no_eager_blowup_for_a_row_wise_decodable_embedding() {
        // TQ2_0_g128 (128 elements / 34 bytes) IS row-wise decodable, so the
        // estimate must equal the naive on-disk sum exactly — no correction.
        let d = half::f16::from_f32(0.05).to_le_bytes();
        let mut block = vec![d[0], d[1]];
        block.extend(std::iter::repeat_n(0b10_01_00_01u8, 32));
        // shape [128, 2]: ne0=128 is exactly one TQ2_0_g128 block; 2 rows.
        let data: Vec<u8> = block.iter().copied().chain(block.iter().copied()).collect();
        let path = write_synthetic_embd_gguf(
            oxibonsai_core::gguf::writer::TensorType::TQ2_0_g128,
            data,
            vec![128, 2],
            "rowwise",
        );

        let naive: u64 = load_tensor_metadata(&path)
            .expect("load metadata")
            .iter()
            .map(|e| e.size_bytes)
            .sum();
        let estimate = estimate_memory_bytes(&path).expect("estimate");
        assert_eq!(
            estimate, naive,
            "a row-wise-decodable embedding must not get an eager-dequant correction"
        );

        let _ = std::fs::remove_dir_all(path.parent().expect("has parent"));
    }

    #[test]
    fn estimate_memory_bytes_adds_the_eager_blowup_for_a_non_row_wise_embedding() {
        // Q8_0 (32 elements / 34 bytes) has NO row-wise decoder in
        // embedding.rs, so `EmbeddingTable::from_gguf` dequantizes the whole
        // table eagerly to f32: resident cost is `element_count * 4`, not
        // the compact on-disk Q8_0 size.
        //
        // shape [64, 4]: ne0=64 is two Q8_0 blocks (32 elements each) -> 2
        // blocks/row x 4 rows x 34 bytes/block = 272 bytes on disk.
        // element_count = 64 * 4 = 256 -> eager f32 resident = 1024 bytes.
        let mut block = [0u8; 34]; // 2-byte f16 scale + 32 int8 codes
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let one_row: Vec<u8> = block.iter().copied().chain(block.iter().copied()).collect();
        let data: Vec<u8> = (0..4).flat_map(|_| one_row.clone()).collect();
        assert_eq!(
            data.len(),
            272,
            "test fixture sanity: 4 rows x 2 blocks x 34 B"
        );

        let path = write_synthetic_embd_gguf(
            oxibonsai_core::gguf::writer::TensorType::Q8_0,
            data,
            vec![64, 4],
            "eager",
        );

        let naive: u64 = load_tensor_metadata(&path)
            .expect("load metadata")
            .iter()
            .map(|e| e.size_bytes)
            .sum();
        assert_eq!(naive, 272 + 32, "272 (embd Q8_0) + 32 (output_norm F32)");

        let estimate = estimate_memory_bytes(&path).expect("estimate");
        let expected_resident_embd = 64u64 * 4 * 4; // element_count * 4 bytes (f32)
        assert_eq!(expected_resident_embd, 1024);
        let expected_total = expected_resident_embd + 32; // + output_norm, unaffected
        assert_eq!(
            estimate, expected_total,
            "must add the eager-dequant delta (1024 - 272 = 752) on top of the naive sum"
        );
        assert!(
            estimate > naive,
            "the honest estimate must be strictly larger than the naive on-disk sum here"
        );

        let _ = std::fs::remove_dir_all(path.parent().expect("has parent"));
    }

    #[test]
    fn token_embd_resident_bytes_is_none_when_embedding_is_absent() {
        let entries = vec![TensorEntry {
            name: "some_other_tensor".to_string(),
            shape: vec![4],
            quant_type_id: 0,
            offset: 0,
            size_bytes: 16,
        }];
        assert_eq!(token_embd_resident_bytes(&entries), None);
    }

    #[test]
    fn token_embd_is_row_wise_decodable_matches_embedding_rs_doc_table() {
        // Positive cases: exactly the set embedding.rs's own module doc
        // names as row-wise decodable.
        for ty in [
            GgufTensorType::Q1_0_g128,
            GgufTensorType::TQ2_0,
            GgufTensorType::TQ2_0_g128,
            GgufTensorType::Q2_0G64,
            GgufTensorType::Q2_0G128DFirst,
            GgufTensorType::PQ2_0,
            GgufTensorType::PTQ1_0,
        ] {
            assert!(
                token_embd_is_row_wise_decodable(ty),
                "{ty:?} must be row-wise decodable"
            );
        }
        // Negative cases: embedding.rs's doc explicitly lists these as
        // falling back to eager dequantization.
        for ty in [
            GgufTensorType::Q4_0,
            GgufTensorType::Q8_0,
            GgufTensorType::Q4_K,
            GgufTensorType::F8_E4M3,
            GgufTensorType::F8_E5M2,
            GgufTensorType::TQ1_0,
            GgufTensorType::F32,
        ] {
            assert!(
                !token_embd_is_row_wise_decodable(ty),
                "{ty:?} must NOT be row-wise decodable"
            );
        }
    }
}
