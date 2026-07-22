//! GGUF file reader — orchestrates header, metadata, and tensor parsing.
//!
//! Provides a high-level [`GgufFile`] that parses a complete GGUF file
//! from either a byte slice or a memory-mapped file.

use byteorder::{LittleEndian, ReadBytesExt};

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::compat::{build_compat_report, check_gguf_header, CompatError, GgufCompatReport};
use crate::gguf::header::GgufHeader;
use crate::gguf::metadata::MetadataStore;
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

/// Maximum tensor dimensions accepted while tolerantly probing, matching
/// the limit enforced by the strict tensor-info parser in `tensor_info.rs`.
const PROBE_MAX_TENSOR_DIMS: u32 = 1024;

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
    /// has `is_loadable = true` and empty `unknown_quant_types` (any unknown
    /// quant type would have hard-failed [`GgufFile::parse`] before this
    /// point) — use [`GgufFile::probe_compat`] to inspect a file that
    /// `parse` would reject.
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
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(DEFAULT_ALIGNMENT as u32) as usize;

        let data_offset = align_offset(offset, alignment)?;

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
            .map(|(_, info)| info.tensor_type as u32)
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
fn probe_read_gguf_string(cursor: &mut std::io::Cursor<&[u8]>) -> BonsaiResult<String> {
    let len = cursor
        .read_u64::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if len > PROBE_MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString {
            offset: cursor.position(),
        });
    }
    let mut buf = vec![0u8; len as usize];
    std::io::Read::read_exact(cursor, &mut buf).map_err(BonsaiError::MmapError)?;
    String::from_utf8(buf).map_err(|_| BonsaiError::InvalidString {
        offset: cursor.position(),
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
        assert_eq!(report.unknown_quant_types, vec![9999]);
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
}
