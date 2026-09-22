//! GGUF file reader — orchestrates header, metadata, and tensor parsing.
//!
//! Provides a high-level [`GgufFile`] that parses a complete GGUF file
//! from either a byte slice or a memory-mapped file.

use byteorder::{LittleEndian, ReadBytesExt};

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::compat::{build_compat_report, check_gguf_header, CompatError, GgufCompatReport};
use crate::gguf::header::GgufHeader;
use crate::gguf::metadata::{MetadataStore, MetadataValue};
use crate::gguf::quant_resolve::AMBIGUOUS_TYPE_ID;
use crate::gguf::tensor_info::TensorStore;

/// Default alignment for tensor data in GGUF files (32 bytes).
const DEFAULT_ALIGNMENT: usize = 32;

/// Byte size of the fixed GGUF header (magic + version + tensor_count +
/// metadata_kv_count).
const HEADER_LEN: usize = 24;

/// Maximum string length accepted while tolerantly probing tensor names
/// (256 MB), matching the limit enforced by the strict tensor-info parser
/// in `tensor_info.rs`.
const PROBE_MAX_STRING_LEN: u64 = 256 * 1024 * 1024;

/// Maximum tensor dimensions accepted while tolerantly probing.
///
/// `GGML_MAX_DIMS` is 4 and llama.cpp rejects `n_dims > 4`; both strict
/// parsers (`tensor_info.rs::MAX_TENSOR_DIMS`, `streaming.rs`) already cap
/// at 4 (core-gguf-16). This tolerant probe path was left at a stale 1024
/// until this fix (wave-1 addendum #3).
const PROBE_MAX_TENSOR_DIMS: u32 = 4;

/// Translate a [`CompatError`] (raised by the shared forward-compat header
/// checker) into the equivalent [`BonsaiError`] variant already used
/// throughout the strict GGUF parsing path, so callers see one consistent
/// error vocabulary regardless of which validator caught the problem.
fn compat_error_to_bonsai(err: CompatError) -> BonsaiError {
    match err {
        CompatError::InvalidMagic(bytes) => {
            let magic = if bytes.len() >= 4 {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                0
            };
            BonsaiError::InvalidMagic { magic }
        }
        CompatError::UnsupportedVersion(version) => BonsaiError::UnsupportedVersion { version },
        CompatError::TruncatedHeader { got, .. } => {
            BonsaiError::UnexpectedEof { offset: got as u64 }
        }
    }
}

/// A parsed GGUF file, containing header, metadata, tensor info, and a
/// reference to the raw tensor data region.
#[derive(Debug)]
pub struct GgufFile<'a> {
    /// Parsed header.
    pub header: GgufHeader,
    /// Key-value metadata store.
    pub metadata: MetadataStore,
    /// Tensor metadata (names, shapes, types, offsets).
    pub tensors: TensorStore,
    /// Byte offset where tensor data begins.
    pub data_offset: usize,
    /// Raw file data (for tensor loading).
    pub data: &'a [u8],
    /// Forward-compatibility diagnostic report computed during parsing (see
    /// [`crate::gguf::compat`]). For a successfully-parsed file this always
    /// has `is_loadable = true` and empty `unknown_quant_types`: every
    /// tensor's quantization type already passed `TensorStore::parse`
    /// (which hard-fails on the first unrecognised type id), and
    /// [`ExtendedQuantType::from_u32`](crate::gguf::compat::ExtendedQuantType::from_u32)
    /// is derived from the exact same
    /// [`GgufTensorType::from_id`](crate::gguf::types::GgufTensorType::from_id)
    /// table `TensorStore::parse` uses (core-gguf-05) — so the two can no
    /// longer disagree about which ids this build recognises. Use
    /// [`GgufFile::probe_compat`] (or [`GgufFile::diagnose`]) to inspect a
    /// file that `parse` would reject.
    pub compat: GgufCompatReport,
}

impl<'a> GgufFile<'a> {
    /// Parse a GGUF file from a byte slice.
    pub fn parse(data: &'a [u8]) -> BonsaiResult<Self> {
        // 0. Validate magic + version via the shared forward-compat header
        // checker first, so `check_gguf_header` is genuinely exercised by
        // the production load path (previously it was only invoked from its
        // own unit tests) and any divergence between it and the header
        // parser below is caught immediately rather than silently ignored.
        check_gguf_header(data).map_err(compat_error_to_bonsai)?;

        // 1. Parse header
        let (header, offset) = GgufHeader::parse(data, 0)?;

        tracing::debug!(
            version = header.version,
            tensors = header.tensor_count,
            metadata = header.metadata_kv_count,
            "parsed GGUF header"
        );

        // 2. Parse metadata
        let (metadata, offset) = MetadataStore::parse(data, offset, header.metadata_kv_count)?;

        tracing::debug!(entries = metadata.len(), "parsed metadata");

        // 3. Parse tensor info
        let (tensors, offset) = TensorStore::parse(data, offset, header.tensor_count)?;

        tracing::debug!(count = tensors.len(), "parsed tensor info");

        // 4. Compute data offset (aligned to `general.alignment`, defaulting
        // to DEFAULT_ALIGNMENT). Rejects a malicious/malformed alignment
        // (zero or not a power of two) instead of silently mis-locating the
        // tensor data section.
        //
        // `general.alignment` must be spec-typed `UINT32` — llama.cpp itself
        // rejects any other GGUF value type for this key
        // (`ggml/src/gguf.cpp:613-618`) — so this reads the raw
        // `MetadataValue` and matches only `Uint32` directly, rather than
        // `MetadataValue::as_u32()` (which also *widens* `Uint64`/`Int32`/
        // the small-integer types). Before this, a spec-invalid file
        // spelling the key as `Uint64` parsed here with a silently
        // different `data_offset` than
        // [`GgufStreamParser`](crate::gguf::streaming::GgufStreamParser)
        // would compute for the identical bytes (core-gguf-15) — the
        // streaming parser already only accepts `Uint32` here, so this
        // brings the strict parser in line with it instead of the other
        // way around.
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT as u32,
            Some(MetadataValue::Uint32(v)) => *v,
            Some(other) => {
                return Err(BonsaiError::InvalidMetadata {
                    key: "general.alignment".to_string(),
                    reason: format!(
                        "general.alignment must be stored as UINT32 per the GGUF spec; found {}",
                        other.type_name()
                    ),
                });
            }
        } as usize;

        let data_offset = align_offset(offset, alignment)?;

        // 4b. Validate the tensor table against the data region
        // (core-gguf-03 / sec-13): sorted-by-offset tensors must start at
        // 0, land on `alignment`-aligned boundaries, and each end exactly
        // where `GGML_PAD(size, alignment)` says the next one begins — the
        // same invariant llama.cpp enforces (`ggml/src/gguf.cpp:780-793`)
        // and the cheapest available detector for a file whose declared
        // group size silently disagrees with its actual byte layout.
        validate_tensor_layout(
            &tensors,
            alignment as u64,
            data_offset as u64,
            data.len() as u64,
        )?;

        // 5. Build a compat-report for logging/diagnostics. Every tensor's
        // quantization type is already known-valid at this point (the
        // strict `TensorStore::parse` above would have failed on any
        // unrecognised type id), so `unknown_quant_types` reflects reality
        // (always empty here) rather than being fabricated; the report
        // still records version/tensor/metadata counts and surfaces any
        // future version-compat warnings if `GgufHeader::parse` is ever
        // relaxed to accept versions beyond {2, 3}.
        let type_ids: Vec<u32> = tensors
            .iter()
            .map(|(_, info)| info.tensor_type.wire_id())
            .collect();
        let compat = build_compat_report(
            header.version,
            header.tensor_count,
            header.metadata_kv_count,
            &type_ids,
        );
        if !compat.warnings.is_empty() {
            tracing::warn!(summary = %compat.summary(), "GGUF forward-compat warnings");
        }

        Ok(GgufFile {
            header,
            metadata,
            tensors,
            data_offset,
            data,
            compat,
        })
    }

    /// Perform a tolerant, best-effort compatibility scan of `data` without
    /// requiring every tensor quantization type or format version to be one
    /// this build can execute.
    ///
    /// Unlike [`GgufFile::parse`], which hard-fails the moment it hits an
    /// unrecognised format version or tensor quantization type, `probe_compat`
    /// tolerates both and folds them into the returned [`GgufCompatReport`]
    /// so a caller can present "this file cannot be loaded, and here is why"
    /// (unsupported version, N tensors with unrecognised quant types, ...)
    /// instead of a bare parse error. Intended as a pre-flight diagnostic —
    /// e.g. before attempting [`GgufFile::parse`] on a file obtained from an
    /// untrusted source, or for `model info`-style tooling.
    ///
    /// Caveat: [`GgufCompatReport::is_loadable`] is currently derived solely
    /// from `unknown_quant_types` (see [`build_compat_report`]); it does not
    /// additionally check that the version is one [`GgufFile::parse`] itself
    /// accepts (`GgufHeader::parse` currently hard-requires version 2 or 3,
    /// while [`check_gguf_header`] also tolerates version 1). A version-1
    /// file with only known quant types can therefore be reported as
    /// `is_loadable = true` here while still being rejected by `parse`.
    /// Callers that need a precise "will `parse` accept this file" answer
    /// should also check `report.version` explicitly.
    pub fn probe_compat(data: &[u8]) -> BonsaiResult<GgufCompatReport> {
        let version = check_gguf_header(data).map_err(compat_error_to_bonsai)?;

        if data.len() < HEADER_LEN {
            return Err(BonsaiError::UnexpectedEof {
                offset: data.len() as u64,
            });
        }
        let tensor_count = read_u64_field(data, 8);
        let metadata_kv_count = read_u64_field(data, 16);

        let (metadata, offset) = MetadataStore::parse(data, HEADER_LEN, metadata_kv_count)?;
        let (type_ids, _end_offset) = scan_tensor_type_ids(data, offset, tensor_count)?;

        Ok(build_compat_report(
            version.to_u32(),
            tensor_count,
            metadata.len() as u64,
            &type_ids,
        ))
    }

    /// Get raw tensor data bytes for a named tensor.
    pub fn tensor_data(&self, name: &str) -> BonsaiResult<&'a [u8]> {
        let info = self.tensors.require(name)?;

        // Validate the byte range entirely in `u64` using checked
        // arithmetic before ever casting to `usize`. `info.offset` and the
        // shape-derived `data_size()` are both attacker-controlled (read
        // directly off the file with no range validation), so a naive
        // `usize` addition can wrap silently in a release build (no
        // `overflow-checks`), letting a crafted `start > end` slip past a
        // guard that only checks `end`. Rejecting on overflow/out-of-range
        // here, before the slice index, turns that into a clean `Err`
        // instead of either a hard panic (`start..end` with `start > end`)
        // or silently-wrong bytes from the wrong file region.
        let data_len = self.data.len() as u64;
        let data_offset = self.data_offset as u64;
        let size = info.data_size();

        let start = data_offset
            .checked_add(info.offset)
            .ok_or(BonsaiError::UnexpectedEof { offset: u64::MAX })?;
        let end = start
            .checked_add(size)
            .ok_or(BonsaiError::UnexpectedEof { offset: u64::MAX })?;

        if end > data_len {
            return Err(BonsaiError::UnexpectedEof { offset: end });
        }

        // `end <= data_len` and `data_len == self.data.len()` (which is
        // already a valid `usize`), so both `start` and `end` are known to
        // fit in `usize` here.
        Ok(&self.data[start as usize..end as usize])
    }

    /// Diagnose `data` as a GGUF file, degrading gracefully instead of
    /// returning a bare parse error.
    ///
    /// Tries the strict [`Self::parse`] first. If that fails — e.g. because
    /// `data` uses a quantization type this build's *parser* does not
    /// recognise at all, an unsupported format version, or a malformed
    /// tensor layout — falls back to the tolerant [`Self::probe_compat`],
    /// which succeeds on far more inputs and reports *why* the file cannot
    /// be loaded (unsupported version, N unrecognised quant types, ...)
    /// instead of nothing.
    ///
    /// This is the "more valuable half" of core-gguf-05: previously nothing
    /// in the codebase called `probe_compat` at all, so `oxibonsai info` /
    /// `oxibonsai validate` on a file `parse` rejects died with an opaque
    /// [`BonsaiError`] instead of the diagnostic report that already
    /// existed. Intended as the fallback path for exactly that kind of
    /// tooling; see [`GgufDiagnosis`] for the two possible outcomes.
    ///
    /// # Errors
    /// Only when *both* the strict and the tolerant path fail — i.e. `data`
    /// is not recoverably a GGUF file at all (bad magic, truncated header,
    /// or malformed metadata neither parser can make sense of). Returns the
    /// strict [`Self::parse`] error in that case.
    pub fn diagnose(data: &'a [u8]) -> BonsaiResult<GgufDiagnosis<'a>> {
        match Self::parse(data) {
            Ok(file) => Ok(GgufDiagnosis::Parsed(file)),
            Err(parse_err) => match Self::probe_compat(data) {
                Ok(report) => Ok(GgufDiagnosis::Degraded(report)),
                Err(_probe_err) => Err(parse_err),
            },
        }
    }
}

/// Outcome of [`GgufFile::diagnose`].
#[derive(Debug)]
pub enum GgufDiagnosis<'a> {
    /// Strict parsing succeeded: every [`GgufFile`] field (config
    /// extraction, tensor loading, ...) is available.
    Parsed(GgufFile<'a>),
    /// Strict parsing failed; this is the best-effort tolerant scan
    /// instead, describing *why* the file cannot be loaded (unsupported
    /// version, unrecognised quant types, ...) rather than nothing.
    Degraded(GgufCompatReport),
}

impl<'a> GgufDiagnosis<'a> {
    /// The fully-parsed file, if strict parsing succeeded.
    pub fn parsed(&self) -> Option<&GgufFile<'a>> {
        match self {
            Self::Parsed(file) => Some(file),
            Self::Degraded(_) => None,
        }
    }

    /// The compatibility report either way: the one embedded in a
    /// successfully-parsed file, or the tolerant fallback report.
    pub fn compat(&self) -> &GgufCompatReport {
        match self {
            Self::Parsed(file) => &file.compat,
            Self::Degraded(report) => report,
        }
    }
}

/// Align an offset to the given alignment boundary.
///
/// `alignment` must be a nonzero power of two. GGUF files that set
/// `general.alignment` to zero or a non-power-of-two value are rejected as
/// malformed rather than silently mis-locating the tensor data section
/// (with `alignment = 0` the naive `alignment - 1` computation underflows,
/// which in a release build with default `overflow-checks = false` wraps to
/// `usize::MAX` and collapses every aligned offset to `0`).
fn align_offset(offset: usize, alignment: usize) -> BonsaiResult<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(BonsaiError::AlignmentError {
            expected: DEFAULT_ALIGNMENT,
            offset: offset as u64,
        });
    }
    Ok((offset + alignment - 1) & !(alignment - 1))
}

/// Validate the tensor table against the data region (core-gguf-03 /
/// sec-13).
///
/// `GgufFile::parse` previously computed `data_offset` and stopped: nothing
/// checked that tensor byte ranges covered the data section exactly, did
/// not overlap, or landed on `alignment`-aligned boundaries —
/// `tensor_data()` only bounds-checks one tensor, lazily, on first access.
/// Walking the tensors in offset order and replaying llama.cpp's own
/// invariant (`ggml/src/gguf.cpp:780-793`) is both the cheapest available
/// check and the one with a real discriminator: exact equality against the
/// *padded* size catches a file whose declared group size is wrong by less
/// than one padding unit, which the weaker `offset[i] + size(i) <=
/// offset[i+1]` inequality would silently accept.
///
/// Checks, per tensor `i` in ascending-offset order:
/// 1. `offset[0] == 0`.
/// 2. `offset[i] % alignment == 0`.
/// 3. `offset[i] + GGML_PAD(size(i), alignment) == offset[i+1]` — exact
///    equality, with one documented exception (see below).
/// 4. `ne0 % block_size == 0` — re-checked here as a cheap defence-in-depth
///    guard; `TensorStore::parse` already enforces this via
///    `TensorInfo::validate_row_blocking`, so this can only fire for a
///    `TensorStore` assembled by some future path other than `parse`.
/// 5. For the last tensor: `offset[last] + GGML_PAD(size(last), alignment)
///    <= file_len - data_offset` — skipped when `file_len <= data_offset`
///    (the caller supplied no tensor-payload bytes at all, e.g. a
///    deliberate metadata-only/bounded-prefix read of just the header and
///    tensor table — the usage sec-17's fix contemplates for
///    `oxibonsai-model`'s memory-estimation helpers). Once any payload
///    bytes are present the check runs normally.
///
/// # The wire-id-42 exception
///
/// A tensor whose [`GgufTensorType::wire_id`](crate::gguf::types::GgufTensorType::wire_id)
/// is [`AMBIGUOUS_TYPE_ID`] (42) has an `info.tensor_type` that is only
/// [`GgufTensorType::from_id`](crate::gguf::types::GgufTensorType::from_id)'s
/// historical qs-first-group-128 *guess* — never the file's actual resolved
/// geometry (see [`crate::gguf::tensor_info::TensorInfo::data_size`]'s own
/// doc, and [`crate::gguf::quant_resolve`], which exists precisely because
/// id 42 has three incompatible on-disk readings). For a genuine group-64
/// file the guessed per-row size always *undershoots* the true size (a
/// group-64 row needs two 2-byte scales per 128 elements against one for
/// group-128, so the true size is always >= the guess), so check 3 uses the
/// weaker `<=` for these tensors instead of rejecting a legitimately
/// ambiguous file before [`crate::gguf::quant_resolve::resolve_type_42`]
/// ever gets a chance to settle it. This is a defence-in-depth complement
/// to the direct cast-site alignment guard in `tensor.rs`
/// (core-gguf-07/sec-02), never a substitute for it: this function is not
/// the only way a byte slice reaches a block-cast.
fn validate_tensor_layout(
    tensors: &TensorStore,
    alignment: u64,
    data_offset: u64,
    file_len: u64,
) -> BonsaiResult<()> {
    let sorted = tensors.sorted_by_offset();
    let Some(first) = sorted.first() else {
        return Ok(());
    };

    // Pass 1 — per-tensor checks (1, 2, 4 above), independent of any other
    // tensor. Run to completion (in offset order) before pass 2, so a
    // tensor's own offset/blocking defect is always reported as such
    // instead of being masked by an earlier tensor's *chain* mismatch that
    // would otherwise be reached first in a single combined pass.
    if first.offset != 0 {
        return Err(BonsaiError::tensor_layout(
            first.name.clone(),
            format!(
                "first tensor in the data section must start at offset 0, got {}",
                first.offset
            ),
        ));
    }
    for info in &sorted {
        if !info.offset.is_multiple_of(alignment) {
            return Err(BonsaiError::tensor_layout(
                info.name.clone(),
                format!(
                    "offset {} is not a multiple of general.alignment ({alignment})",
                    info.offset
                ),
            ));
        }
        // Defence in depth: `TensorStore::parse` already enforces this via
        // `validate_row_blocking`, so a `TensorStore` reaching this point
        // via `parse` can never fail it — kept for a `TensorStore` built by
        // any other path.
        info.validate_row_blocking()?;
    }

    // Pass 2 — the offset chain and the data-section tail (3 and 5 above).
    for (idx, info) in sorted.iter().enumerate() {
        let padded = info.padded_extent(alignment).ok_or_else(|| {
            BonsaiError::tensor_layout(
                info.name.clone(),
                "tensor byte extent overflows u64 once padded to general.alignment".to_string(),
            )
        })?;
        let end = info.offset.checked_add(padded).ok_or_else(|| {
            BonsaiError::tensor_layout(
                info.name.clone(),
                "offset + padded extent overflows u64".to_string(),
            )
        })?;

        match sorted.get(idx + 1) {
            Some(next) => {
                let exact_required = info.tensor_type.wire_id() != AMBIGUOUS_TYPE_ID;
                let ok = if exact_required {
                    end == next.offset
                } else {
                    end <= next.offset
                };
                if !ok {
                    return Err(BonsaiError::tensor_layout(
                        info.name.clone(),
                        format!(
                            "tensor occupies [{}, {end}) but the next tensor '{}' starts at {} \
                             (expected {})",
                            info.offset,
                            next.name,
                            next.offset,
                            if exact_required {
                                "exact equality — a gap or overlap in the data section"
                            } else {
                                "end <= next start (ambiguous ggml id 42 tensor)"
                            },
                        ),
                    ));
                }
            }
            None => {
                if file_len > data_offset {
                    let available = file_len - data_offset;
                    if end > available {
                        return Err(BonsaiError::tensor_layout(
                            info.name.clone(),
                            format!(
                                "tensor extends to byte {end} of the data section, but only \
                                 {available} bytes are available ({file_len} total file bytes, \
                                 data starts at {data_offset})"
                            ),
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Read a little-endian `u64` field from `data` at `offset` without
/// requiring a full cursor. Callers must have already validated that
/// `offset + 8 <= data.len()`.
fn read_u64_field(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

/// Read a GGUF string `[u64 length][utf8 bytes]` from a cursor without
/// validating the resulting quantization type — used by
/// [`scan_tensor_type_ids`] to tolerantly probe tensor names.
///
/// Reads the string body via the one shared, already-hardened
/// [`crate::gguf::tensor_info::read_string_body_chunked`] (core-gguf-20 /
/// sec-10 / wave-2.5 integration addendum, item 5) in bounded 64 KiB
/// pieces, instead of this file's own former independent copy
/// (`read_gguf_string_chunked` / a private `STRING_READ_CHUNK`) that
/// allocated `vec![0u8; len]` up front — bounded only by
/// [`PROBE_MAX_STRING_LEN`] (256 MiB) — before confirming the reader
/// actually had that much data left. `metadata.rs` and `tensor_info.rs`
/// itself already call the same shared function; this was the one
/// remaining independent copy DRY was meant to remove.
fn probe_read_gguf_string(cursor: &mut std::io::Cursor<&[u8]>) -> BonsaiResult<String> {
    let len = cursor
        .read_u64::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if len > PROBE_MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString {
            offset: cursor.position(),
        });
    }
    crate::gguf::tensor_info::read_string_body_chunked(cursor, len).map_err(|_| {
        BonsaiError::InvalidString {
            offset: cursor.position(),
        }
    })
}

/// Scan raw tensor-info entries starting at `offset`, collecting each
/// tensor's raw quantization type ID without requiring it to be one this
/// build recognises or can execute.
///
/// Used by [`GgufFile::probe_compat`] to build a forward-compatibility
/// report for files containing tensor types this version of OxiBonsai does
/// not (yet) know about, which the strict [`crate::gguf::tensor_info::TensorStore::parse`]
/// would otherwise hard-reject before a report could ever be produced.
fn scan_tensor_type_ids(data: &[u8], offset: usize, count: u64) -> BonsaiResult<(Vec<u32>, usize)> {
    let mut cursor = std::io::Cursor::new(data);
    cursor.set_position(offset as u64);

    let mut type_ids = Vec::new();
    for _ in 0..count {
        let name = probe_read_gguf_string(&mut cursor)?;

        let n_dims = cursor
            .read_u32::<LittleEndian>()
            .map_err(BonsaiError::MmapError)?;
        if n_dims > PROBE_MAX_TENSOR_DIMS {
            return Err(BonsaiError::InvalidMetadata {
                key: name,
                reason: format!("tensor has too many dimensions: {n_dims}"),
            });
        }
        for _ in 0..n_dims {
            cursor
                .read_u64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
        }

        let type_id = cursor
            .read_u32::<LittleEndian>()
            .map_err(BonsaiError::MmapError)?;
        // Tensor byte offset; not validated here, this is a tolerant probe.
        let _tensor_offset = cursor
            .read_u64::<LittleEndian>()
            .map_err(BonsaiError::MmapError)?;

        type_ids.push(type_id);
    }
    Ok((type_ids, cursor.position() as usize))
}

/// Load a GGUF file from disk using memory-mapping (if the `mmap` feature is enabled).
#[cfg(feature = "mmap")]
pub fn mmap_gguf_file(path: &std::path::Path) -> BonsaiResult<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: We treat the mapped memory as read-only and the file should not be
    // modified while we hold the mapping. This is the standard usage pattern
    // for memory-mapped model files.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(mmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_offset_works() {
        assert_eq!(align_offset(0, 32).expect("valid alignment"), 0);
        assert_eq!(align_offset(1, 32).expect("valid alignment"), 32);
        assert_eq!(align_offset(31, 32).expect("valid alignment"), 32);
        assert_eq!(align_offset(32, 32).expect("valid alignment"), 32);
        assert_eq!(align_offset(33, 32).expect("valid alignment"), 64);
    }

    #[test]
    fn align_offset_rejects_zero_alignment() {
        let result = align_offset(100, 0);
        match result {
            Err(BonsaiError::AlignmentError { .. }) => {}
            other => panic!("expected AlignmentError for zero alignment, got: {other:?}"),
        }
    }

    #[test]
    fn align_offset_rejects_non_power_of_two_alignment() {
        for bad in [3usize, 5, 6, 7, 9, 33, 100] {
            let result = align_offset(100, bad);
            match result {
                Err(BonsaiError::AlignmentError { .. }) => {}
                other => panic!(
                    "expected AlignmentError for non-power-of-two alignment {bad}, got: {other:?}"
                ),
            }
        }
    }

    #[test]
    fn align_offset_accepts_powers_of_two() {
        for good in [1usize, 2, 4, 8, 16, 32, 64, 128, 1024] {
            align_offset(100, good).unwrap_or_else(|e| {
                panic!("expected alignment {good} to be accepted, got error: {e}")
            });
        }
    }

    fn gguf_header_bytes(
        magic: u32,
        version: u32,
        tensor_count: u64,
        metadata_kv_count: u64,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&magic.to_le_bytes());
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&tensor_count.to_le_bytes());
        bytes.extend_from_slice(&metadata_kv_count.to_le_bytes());
        bytes
    }

    const GGUF_MAGIC_TEST: u32 = 0x4655_4747;

    /// A file whose sole tensor has a raw quant type id this build does not
    /// recognise must be rejected outright by the strict `parse()` ...
    #[test]
    fn parse_rejects_unknown_quant_type() {
        let mut data = gguf_header_bytes(GGUF_MAGIC_TEST, 3, 1, 0);
        // One tensor: name "t", 1 dim [1], type id 9999 (unrecognised), offset 0.
        data.extend_from_slice(&1u64.to_le_bytes());
        data.push(b't');
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&1u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&9999u32.to_le_bytes()); // unknown type id
        data.extend_from_slice(&0u64.to_le_bytes()); // offset
        assert!(GgufFile::parse(&data).is_err());
    }

    /// ... but `probe_compat` on the exact same bytes must succeed and
    /// report the type as unknown/unloadable, giving a real diagnostic
    /// instead of a bare parse error.
    #[test]
    fn probe_compat_reports_unknown_quant_type_instead_of_hard_failing() {
        let mut data = gguf_header_bytes(GGUF_MAGIC_TEST, 3, 1, 0);
        data.extend_from_slice(&1u64.to_le_bytes());
        data.push(b't');
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&9999u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let report = GgufFile::probe_compat(&data)
            .expect("probe_compat should tolerate unknown quant types");
        assert!(!report.is_loadable);
        assert_eq!(report.unknown_quant_types.len(), 1);
        assert!(report.unknown_quant_types.contains(&9999));
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn probe_compat_accepts_fully_known_file() {
        let mut data = gguf_header_bytes(GGUF_MAGIC_TEST, 3, 1, 0);
        data.extend_from_slice(&1u64.to_le_bytes());
        data.push(b't');
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // F32, a known type
        data.extend_from_slice(&0u64.to_le_bytes());

        let report = GgufFile::probe_compat(&data).expect("probe_compat should succeed");
        assert!(report.is_loadable);
        assert!(report.unknown_quant_types.is_empty());
    }

    #[test]
    fn probe_compat_rejects_bad_magic_via_shared_checker() {
        let data = gguf_header_bytes(0xDEAD_BEEF, 3, 0, 0);
        assert!(GgufFile::probe_compat(&data).is_err());
    }

    // ── core-gguf-03 / sec-13: validate_tensor_layout via GgufFile::parse ──

    fn tensor_info_bytes(name: &str, shape: &[u64], type_id: u32, offset: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&type_id.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b
    }

    /// Assemble a full GGUF byte buffer: header (no metadata) + the given
    /// tensor-info entries + alignment padding + `tensor_data_len` zero
    /// bytes of tensor data.
    fn assemble_gguf(tensor_infos: &[Vec<u8>], tensor_data_len: usize) -> Vec<u8> {
        let mut data = gguf_header_bytes(GGUF_MAGIC_TEST, 3, tensor_infos.len() as u64, 0);
        for info in tensor_infos {
            data.extend_from_slice(info);
        }
        while !data.len().is_multiple_of(32) {
            data.push(0);
        }
        data.extend(vec![0u8; tensor_data_len]);
        data
    }

    #[test]
    fn parse_accepts_a_well_formed_two_tensor_layout() {
        // "a": 8 F32 elements = 32 bytes (already 32-aligned). "b": 4 F32
        // elements = 16 bytes, padded to 32. Total data section: 64 bytes.
        let infos = vec![
            tensor_info_bytes("a", &[8], 0, 0),
            tensor_info_bytes("b", &[4], 0, 32),
        ];
        let data = assemble_gguf(&infos, 64);
        GgufFile::parse(&data).expect("well-formed two-tensor layout must parse");
    }

    #[test]
    fn parse_rejects_first_tensor_not_at_offset_zero() {
        let infos = vec![tensor_info_bytes("a", &[8], 0, 32)];
        let data = assemble_gguf(&infos, 64);
        match GgufFile::parse(&data) {
            Err(BonsaiError::TensorLayout { name, reason }) => {
                assert_eq!(name, "a");
                assert!(reason.contains('0'), "reason: {reason}");
            }
            other => panic!("expected TensorLayout, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_a_gap_between_tensors() {
        // "a" needs exactly 32 bytes (8 F32 elements); declaring "b" at 64
        // instead of 32 leaves an undeclared 32-byte gap.
        let infos = vec![
            tensor_info_bytes("a", &[8], 0, 0),
            tensor_info_bytes("b", &[4], 0, 64),
        ];
        let data = assemble_gguf(&infos, 96);
        match GgufFile::parse(&data) {
            Err(BonsaiError::TensorLayout { name, reason }) => {
                assert_eq!(name, "a");
                assert!(reason.contains("gap or overlap"), "reason: {reason}");
            }
            other => panic!("expected TensorLayout for a gap, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_overlapping_tensors() {
        // "a" needs 64 bytes (16 F32 elements, already 32-aligned);
        // declaring "b" at the alignment-valid offset 32 places it inside
        // "a"'s [0, 64) span — an alignment-respecting overlap, so this
        // exercises the offset-chain check specifically, not check (iv).
        let infos = vec![
            tensor_info_bytes("a", &[16], 0, 0),
            tensor_info_bytes("b", &[4], 0, 32),
        ];
        let data = assemble_gguf(&infos, 64);
        match GgufFile::parse(&data) {
            Err(BonsaiError::TensorLayout { name, reason }) => {
                assert_eq!(name, "a");
                assert!(reason.contains("gap or overlap"), "reason: {reason}");
            }
            other => panic!("expected TensorLayout for an overlap, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_an_offset_not_a_multiple_of_alignment() {
        let infos = vec![
            tensor_info_bytes("a", &[8], 0, 0),
            tensor_info_bytes("b", &[4], 0, 17), // not a multiple of 32
        ];
        let data = assemble_gguf(&infos, 64);
        match GgufFile::parse(&data) {
            Err(BonsaiError::TensorLayout { name, reason }) => {
                assert_eq!(name, "b");
                assert!(reason.contains("alignment"), "reason: {reason}");
            }
            other => panic!("expected TensorLayout for a misaligned offset, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_the_last_tensor_exceeding_the_file_length() {
        let infos = vec![tensor_info_bytes("a", &[8], 0, 0)];
        // "a" needs 32 bytes but only 16 are actually supplied.
        let data = assemble_gguf(&infos, 16);
        match GgufFile::parse(&data) {
            Err(BonsaiError::TensorLayout { name, reason }) => {
                assert_eq!(name, "a");
                assert!(reason.contains("available"), "reason: {reason}");
            }
            other => panic!("expected TensorLayout for a truncated data section, got {other:?}"),
        }
    }

    #[test]
    fn parse_accepts_a_metadata_only_read_with_no_tensor_payload_bytes() {
        // Exactly the header + tensor-info + alignment padding, and not one
        // byte more — a deliberate bounded-prefix / "just the structure"
        // read (sec-17's contemplated usage). Check (iii) must not reject
        // this: there is no payload to bounds-check against.
        let infos = vec![tensor_info_bytes("a", &[8], 0, 0)];
        let data = assemble_gguf(&infos, 0);
        GgufFile::parse(&data).expect("a metadata-only read must still parse");
    }

    /// The one documented exception: a wire-id-42 tensor's `data_size()` is
    /// only `GgufTensorType::from_id`'s historical qs-first-g128 *guess*.
    /// For a genuine group-64 file the guess always undershoots the true
    /// size (two 2-byte scales per 128 elements vs. one), so a real file
    /// like this must still parse even though the guessed padded extent
    /// (544) does not exactly equal the next tensor's true offset (576).
    #[test]
    fn parse_tolerates_the_wire_id_42_size_guess_undershoot() {
        let infos = vec![
            tensor_info_bytes("a", &[128, 16], 42, 0),
            tensor_info_bytes("b", &[1], 0, 576),
        ];
        let data = assemble_gguf(&infos, 608);
        let file = GgufFile::parse(&data)
            .expect("a genuine group-64 id-42 tensor must not be rejected by the g128 guess");
        assert_eq!(
            file.tensors.require("a").expect("present").tensor_type,
            crate::gguf::types::GgufTensorType::TQ2_0_g128,
            "the guessed type for id 42 is always the legacy qs-first g128 reading"
        );
    }

    /// Without the wire-id-42 relaxation, the same bytes would fail exact
    /// equality (544 != 576) — pin the fixture's numbers down directly so a
    /// future change to either formula cannot silently make this test
    /// vacuous.
    #[test]
    fn wire_id_42_fixture_actually_exercises_the_undershoot() {
        use crate::gguf::tensor_info::{padded_size, row_size_bytes};
        use crate::gguf::types::GgufTensorType;

        let guessed = row_size_bytes(GgufTensorType::TQ2_0_g128, &[128, 16]);
        let guessed_padded = padded_size(guessed, 32).expect("valid alignment");
        assert_eq!(guessed_padded, 544);
        assert_ne!(
            guessed_padded, 576,
            "fixture must actually diverge from the true group-64 offset"
        );
    }

    // ── core-gguf-05: wire_id() cast in the compat type-id collection ──────

    /// `type_ids` collection in `parse()` must go through `wire_id()`, not
    /// a raw `as u32` cast — otherwise the `Q2_0G64`/`Q2_0G128DFirst`
    /// sentinel discriminants (`0x4000_002A`/`0x4001_002A`) would leak into
    /// the compat report instead of the wire id 42 they both serialise as.
    /// Every real tensor `TensorStore::parse` produces uses
    /// `GgufTensorType::from_id`, which never returns those two sentinel
    /// variants (only `from_id_marked` can) — so this test instead pins the
    /// invariant directly: a successfully-`parse`d file's compat report
    /// must be `is_loadable` with zero unknown quant types, which would not
    /// hold if the sentinel value ever leaked into `type_ids`.
    #[test]
    fn parse_compat_report_never_leaks_a_sentinel_discriminant() {
        let infos = vec![tensor_info_bytes("a", &[128], 42, 0)];
        let data = assemble_gguf(&infos, 64);
        let file = GgufFile::parse(&data).expect("parse");
        assert!(file.compat.is_loadable);
        assert!(file.compat.unknown_quant_types.is_empty());
    }
}
