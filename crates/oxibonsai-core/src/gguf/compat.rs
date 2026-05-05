//! Forward-compatibility layer for GGUF v2/v3 and future GGMLv4 formats.
//!
//! This module provides:
//! - [`GgufVersion`] — typed version enum with capability queries
//! - [`ExtendedQuantType`] — extended quantization type awareness (including unknown IDs)
//! - [`GgufCompatReport`] — compatibility report for a parsed GGUF file
//! - [`check_gguf_header`] — validate a raw GGUF magic+version header
//! - [`build_compat_report`] — construct a report from parsed file metadata
//! - [`CompatError`] — error type for compatibility operations

use thiserror::Error;

// ── GGUF magic constant ──────────────────────────────────────────────────────

/// GGUF magic bytes: ASCII "GGUF".
const GGUF_MAGIC_BYTES: &[u8; 4] = b"GGUF";

/// Minimum header size required to read magic (4 bytes) + version (4 bytes).
const GGUF_MIN_HEADER_BYTES: usize = 8;

// ── GgufVersion ─────────────────────────────────────────────────────────────

/// Supported GGUF file format versions.
///
/// The ordering reflects the chronological/capability ordering: V1 < V2 < V3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GgufVersion {
    /// GGUF version 1 — original format (rare in the wild).
    V1 = 1,
    /// GGUF version 2 — adds F16 KV cache metadata support.
    V2 = 2,
    /// GGUF version 3 — adds 32-byte aligned tensor data sections.
    V3 = 3,
}

impl GgufVersion {
    /// Parse a [`GgufVersion`] from a raw `u32` version field.
    ///
    /// Returns `None` for unrecognised version numbers.
    ///
    /// # Examples
    /// ```
    /// use oxibonsai_core::GgufVersion;
    /// assert_eq!(GgufVersion::from_u32(2), Some(GgufVersion::V2));
    /// assert_eq!(GgufVersion::from_u32(99), None);
    /// ```
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            3 => Some(Self::V3),
            _ => None,
        }
    }

    /// Convert back to the raw `u32` version field used in GGUF files.
    pub fn to_u32(self) -> u32 {
        self as u32
    }

    /// Whether this version supports F16 key-value cache metadata.
    ///
    /// Introduced in GGUF v2.
    pub fn supports_f16_kv(&self) -> bool {
        *self >= Self::V2
    }

    /// Whether this version mandates 32-byte-aligned tensor data sections.
    ///
    /// Introduced in GGUF v3.
    pub fn supports_aligned_tensors(&self) -> bool {
        *self >= Self::V3
    }
}

impl std::fmt::Display for GgufVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.to_u32())
    }
}

// ── ExtendedQuantType ────────────────────────────────────────────────────────

/// Extended quantization type awareness, including types beyond what
/// OxiBonsai can execute but which should be recognised for forward-compat
/// reporting.
///
/// Unknown type IDs (e.g. from future GGMLv4 formats) are preserved as
/// [`Unknown(u32)`](ExtendedQuantType::Unknown) rather than causing a hard
/// parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum ExtendedQuantType {
    /// 32-bit IEEE 754 float (type id 0).
    F32,
    /// 16-bit IEEE 754 half-float (type id 1).
    F16,
    /// 4-bit quantization, variant 0 (type id 2).
    Q4_0,
    /// 4-bit quantization, variant 1 (type id 3).
    Q4_1,
    /// 5-bit quantization, variant 0 (type id 6).
    Q5_0,
    /// 5-bit quantization, variant 1 (type id 7).
    Q5_1,
    /// 8-bit quantization, variant 0 (type id 8).
    Q8_0,
    /// 8-bit quantization, variant 1 (type id 9).
    Q8_1,
    /// 2-bit K-quant (type id 10).
    Q2_K,
    /// 3-bit K-quant (type id 11).
    Q3_K,
    /// 4-bit K-quant (type id 12) — corresponds to Q4_K_M family.
    Q4_K,
    /// 5-bit K-quant (type id 13) — corresponds to Q5_K_M family.
    Q5_K,
    /// 6-bit K-quant (type id 14).
    Q6_K,
    /// 8-bit K-quant (type id 15).
    Q8_K,
    /// OxiBonsai native 1-bit format: 128 sign-bits + FP16 group scale (type id 41).
    Q1_0_G128,
    /// PrismML FP8 E4M3FN quantization (type id 43).
    F8_E4M3,
    /// PrismML FP8 E5M2 quantization (type id 44).
    F8_E5M2,
    /// An unrecognised quantization type ID encountered in a GGUF file.
    ///
    /// Carrying the raw ID rather than failing allows forward-compatible
    /// inspection of future GGMLv4 files.
    Unknown(u32),
}

impl ExtendedQuantType {
    /// Construct from a raw GGUF tensor type ID.
    ///
    /// Any ID that does not map to a known variant becomes
    /// [`Unknown(id)`](ExtendedQuantType::Unknown).
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            41 => Self::Q1_0_G128,
            43 => Self::F8_E4M3,
            44 => Self::F8_E5M2,
            other => Self::Unknown(other),
        }
    }

    /// Return the raw GGUF tensor type ID for this quantization type.
    pub fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::Q1_0_G128 => 41,
            Self::F8_E4M3 => 43,
            Self::F8_E5M2 => 44,
            Self::Unknown(id) => id,
        }
    }

    /// Approximate bits-per-weight for this quantization format.
    ///
    /// For K-quants the figure accounts for per-block scale/min overhead
    /// amortised over the block size (256 weights). For unknown types,
    /// `0.0` is returned.
    pub fn bits_per_weight(self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 => 16.0,
            // Q4_0: 32 weights, 4 bits each + 16-bit scale
            // bytes = 2 + 16 = 18; bits_per_w = 18*8/32 = 4.5
            Self::Q4_0 => 4.5,
            // Q4_1: 32 weights + 16-bit scale + 16-bit min
            // bytes = 2 + 2 + 16 = 20; bits_per_w = 20*8/32 = 5.0
            Self::Q4_1 => 5.0,
            // Q5_0: 32 weights at 5 bits + 16-bit scale
            // bytes = 2 + 4 + 16 = 22; bits_per_w = 22*8/32 = 5.5
            Self::Q5_0 => 5.5,
            // Q5_1: 32 weights at 5 bits + 16-bit scale + 16-bit min
            // bytes = 2 + 2 + 4 + 16 = 24; bits_per_w = 24*8/32 = 6.0
            Self::Q5_1 => 6.0,
            // Q8_0: 32 weights at 8 bits + 16-bit scale
            // bytes = 2 + 32 = 34; bits_per_w = 34*8/32 = 8.5
            Self::Q8_0 => 8.5,
            // Q8_1: 32 weights at 8 bits + 16-bit scale + 16-bit delta
            // bytes = 4 + 4 + 32 = 40; bits_per_w = 40*8/32 = 10.0
            // (using the more accurate 8.5 commonly reported)
            Self::Q8_1 => 8.5,
            // Q2_K: 256 weights; block = 84 bytes; bits_per_w = 84*8/256 ≈ 2.625
            Self::Q2_K => 2.625,
            // Q3_K: 256 weights; block = 110 bytes; bits_per_w = 110*8/256 ≈ 3.4375
            Self::Q3_K => 3.4375,
            // Q4_K (Q4_K_M): 256 weights; block = 144 bytes; bits_per_w = 144*8/256 = 4.5
            Self::Q4_K => 4.5,
            // Q5_K (Q5_K_M): 256 weights; block = 176 bytes; bits_per_w = 176*8/256 = 5.5
            Self::Q5_K => 5.5,
            // Q6_K: 256 weights; block = 210 bytes; bits_per_w = 210*8/256 ≈ 6.5625
            Self::Q6_K => 6.5625,
            // Q8_K: 256 weights; block = 292 bytes; bits_per_w = 292*8/256 ≈ 9.125
            Self::Q8_K => 9.125,
            // Q1_0_G128: 128 weights, 1 bit each + 16-bit scale
            // block = 2 + 16 = 18 bytes; bits_per_w = 18*8/128 = 1.125
            Self::Q1_0_G128 => 1.125,
            // F8_E4M3 / F8_E5M2: 32 weights × 1 byte + 16-bit scale
            // block = 32 + 2 = 34 bytes; bits_per_w = 34*8/32 = 8.5
            Self::F8_E4M3 => 8.5,
            Self::F8_E5M2 => 8.5,
            Self::Unknown(_) => 0.0,
        }
    }

    /// Returns `true` if this is a known (recognised) quantization type.
    pub fn is_known(self) -> bool {
        !matches!(self, Self::Unknown(_))
    }

    /// Human-readable name for this quantization type.
    ///
    /// For [`Unknown`](ExtendedQuantType::Unknown) variants the string is
    /// the static string `"Unknown"`. Callers that need the raw ID can use
    /// [`to_u32`](ExtendedQuantType::to_u32).
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::Q1_0_G128 => "Q1_0_G128",
            Self::F8_E4M3 => "F8_E4M3",
            Self::F8_E5M2 => "F8_E5M2",
            // Cannot return dynamic &'static str, so return the generic label.
            Self::Unknown(_) => "Unknown",
        }
    }
}

impl std::fmt::Display for ExtendedQuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "Unknown({})", id),
            other => write!(f, "{}", other.name()),
        }
    }
}

// ── GgufCompatReport ─────────────────────────────────────────────────────────

/// Compatibility report produced after inspecting a GGUF file.
///
/// Describes whether the file can be loaded by this version of OxiBonsai,
/// and lists any forward-compatibility caveats encountered during parsing.
#[derive(Debug, Clone)]
pub struct GgufCompatReport {
    /// The GGUF format version detected in the file.
    pub version: GgufVersion,
    /// Number of tensors present in the file.
    pub tensor_count: u64,
    /// Number of metadata key-value pairs present in the file.
    pub metadata_count: u64,
    /// Raw quantization type IDs that were not recognised.
    pub unknown_quant_types: Vec<u32>,
    /// Human-readable warnings accumulated during compatibility checking.
    pub warnings: Vec<String>,
    /// Whether OxiBonsai believes it can load this file.
    ///
    /// Set by [`finalize`](GgufCompatReport::finalize); initially `true`.
    pub is_loadable: bool,
}

impl GgufCompatReport {
    /// Create a new report for a file at the given version with the given
    /// tensor and metadata counts.
    ///
    /// [`warnings`](GgufCompatReport::warnings) is empty and
    /// [`is_loadable`](GgufCompatReport::is_loadable) is `true` until
    /// [`finalize`](GgufCompatReport::finalize) is called.
    pub fn new(version: GgufVersion, tensor_count: u64, metadata_count: u64) -> Self {
        Self {
            version,
            tensor_count,
            metadata_count,
            unknown_quant_types: Vec::new(),
            warnings: Vec::new(),
            is_loadable: true,
        }
    }

    /// Append a human-readable warning to the report.
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Record an unrecognised quantization type ID.
    ///
    /// Duplicates are stored as-is; call sites should de-duplicate if needed.
    pub fn add_unknown_quant(&mut self, quant_id: u32) {
        self.unknown_quant_types.push(quant_id);
    }

    /// Finalise the report.
    ///
    /// Warnings are informational and do not prevent loading. However, if any
    /// tensors use unknown quant types OxiBonsai cannot decode them, so
    /// `is_loadable` is set to `false` and a synthesised warning is added.
    pub fn finalize(&mut self) {
        if !self.unknown_quant_types.is_empty() {
            self.is_loadable = false;
            self.add_warning(format!(
                "file contains {} tensor(s) with unrecognised quantization type(s): {:?}",
                self.unknown_quant_types.len(),
                self.unknown_quant_types,
            ));
        }
    }

    /// Return a single-line human-readable summary of the report.
    pub fn summary(&self) -> String {
        format!(
            "GGUF {} | tensors={} metadata={} | unknown_quants={} | warnings={} | loadable={}",
            self.version,
            self.tensor_count,
            self.metadata_count,
            self.unknown_quant_types.len(),
            self.warnings.len(),
            self.is_loadable,
        )
    }
}

// ── check_gguf_header ────────────────────────────────────────────────────────

/// Validate and parse the GGUF magic number and version from a raw byte slice.
///
/// Expects at least 8 bytes:
/// ```text
/// bytes[0..4]  — ASCII "GGUF"
/// bytes[4..8]  — version as little-endian u32
/// ```
///
/// # Errors
///
/// - [`CompatError::TruncatedHeader`] if `bytes.len() < 8`
/// - [`CompatError::InvalidMagic`] if the first four bytes are not `b"GGUF"`
/// - [`CompatError::UnsupportedVersion`] if the version field is not 1, 2, or 3
pub fn check_gguf_header(bytes: &[u8]) -> Result<GgufVersion, CompatError> {
    if bytes.len() < GGUF_MIN_HEADER_BYTES {
        return Err(CompatError::TruncatedHeader {
            need: GGUF_MIN_HEADER_BYTES,
            got: bytes.len(),
        });
    }

    let magic = &bytes[0..4];
    if magic != GGUF_MAGIC_BYTES {
        return Err(CompatError::InvalidMagic(magic.to_vec()));
    }

    // Version is stored as a little-endian u32 starting at byte 4.
    let version_bytes: [u8; 4] =
        bytes[4..8]
            .try_into()
            .map_err(|_| CompatError::TruncatedHeader {
                need: GGUF_MIN_HEADER_BYTES,
                got: bytes.len(),
            })?;
    let version_u32 = u32::from_le_bytes(version_bytes);

    GgufVersion::from_u32(version_u32).ok_or(CompatError::UnsupportedVersion(version_u32))
}

// ── build_compat_report ──────────────────────────────────────────────────────

/// Build a [`GgufCompatReport`] from information already extracted by a GGUF
/// reader.
///
/// `tensor_quant_type_ids` should contain the raw quantization type ID for
/// every tensor in the file. Any IDs not recognised by [`ExtendedQuantType`]
/// are recorded as unknown and will cause [`GgufCompatReport::is_loadable`]
/// to be set to `false` after [`finalize`](GgufCompatReport::finalize).
///
/// If `version_u32` is not a supported GGUF version, the report falls back to
/// [`GgufVersion::V3`] and adds an informational warning.
pub fn build_compat_report(
    version_u32: u32,
    tensor_count: u64,
    metadata_count: u64,
    tensor_quant_type_ids: &[u32],
) -> GgufCompatReport {
    let (version, unknown_ver) = match GgufVersion::from_u32(version_u32) {
        Some(v) => (v, false),
        None => (GgufVersion::V3, true),
    };

    let mut report = GgufCompatReport::new(version, tensor_count, metadata_count);

    if unknown_ver {
        report.add_warning(format!(
            "GGUF version {} is not explicitly supported; treating as v3 for structural parsing",
            version_u32
        ));
    }

    // Inspect each tensor's quantization type.
    for &quant_id in tensor_quant_type_ids {
        let qt = ExtendedQuantType::from_u32(quant_id);
        if !qt.is_known() {
            report.add_unknown_quant(quant_id);
        }
    }

    report.finalize();
    report
}

// ── CompatError ──────────────────────────────────────────────────────────────

/// Errors produced by the forward-compatibility layer.
#[derive(Debug, Error)]
pub enum CompatError {
    /// The first four bytes did not spell "GGUF".
    #[error("invalid GGUF magic: expected GGUF, got {0:?}")]
    InvalidMagic(Vec<u8>),

    /// The version field holds a value not in {1, 2, 3}.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// The byte slice was too short to contain the full header.
    #[error("truncated header: need at least {need} bytes, got {got}")]
    TruncatedHeader { need: usize, got: usize },
}

// ── Inline unit tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_version_round_trips_to_u32() {
        assert_eq!(GgufVersion::V1.to_u32(), 1);
        assert_eq!(GgufVersion::V2.to_u32(), 2);
        assert_eq!(GgufVersion::V3.to_u32(), 3);
    }

    #[test]
    fn gguf_version_display_format() {
        assert_eq!(GgufVersion::V1.to_string(), "v1");
        assert_eq!(GgufVersion::V2.to_string(), "v2");
        assert_eq!(GgufVersion::V3.to_string(), "v3");
    }

    #[test]
    fn extended_quant_unknown_display_includes_id() {
        let qt = ExtendedQuantType::Unknown(999);
        assert!(qt.to_string().contains("999"));
    }

    #[test]
    fn extended_quant_roundtrip_to_u32() {
        assert_eq!(ExtendedQuantType::F32.to_u32(), 0);
        assert_eq!(ExtendedQuantType::F16.to_u32(), 1);
        assert_eq!(ExtendedQuantType::Q1_0_G128.to_u32(), 41);
        assert_eq!(ExtendedQuantType::Unknown(99).to_u32(), 99);
    }
}
