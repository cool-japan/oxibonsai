//! Forward-compatibility layer for GGUF v2/v3 and future GGMLv4 formats.
//!
//! This module provides:
//! - [`GgufVersion`] — typed version enum with capability queries
//! - [`ExtendedQuantType`] — extended quantization type awareness (including unknown IDs)
//! - [`GgufCompatReport`] — compatibility report for a parsed GGUF file
//! - [`check_gguf_header`] — validate a raw GGUF magic+version header
//! - [`build_compat_report`] — construct a report from parsed file metadata
//! - [`CompatError`] — error type for compatibility operations

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::gguf::types::GgufTensorType;

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
    /// Any other type this build's parser recognises
    /// ([`GgufTensorType::from_id`] succeeds) that predates this enum's
    /// fixed variant set above — e.g. BF16 (30), TQ2_0 (35), the legacy
    /// `TQ2_0_g128` reading of id 42, the upstream `IQ*` family, or the
    /// Bonsai 2 27B formats PQ2_0 (142) / PTQ1_0 (143).
    ///
    /// Carries the full [`GgufTensorType`] (rather than collapsing to
    /// [`Unknown`](Self::Unknown)) so [`Self::to_u32`], [`Self::name`] and
    /// [`Self::bits_per_weight`] stay exact instead of losing information
    /// (core-gguf-05: this is what makes [`Self::from_u32`] a true superset
    /// of `GgufTensorType::from_id` rather than a second, independently
    /// maintained id table that silently drifts out of sync with it).
    Other(GgufTensorType),
    /// An unrecognised quantization type ID encountered in a GGUF file —
    /// one [`GgufTensorType::from_id`] itself does not recognise either.
    ///
    /// Carrying the raw ID rather than failing allows forward-compatible
    /// inspection of future GGMLv4 files.
    Unknown(u32),
}

impl From<GgufTensorType> for ExtendedQuantType {
    /// Map every [`GgufTensorType`] the parser recognises onto an
    /// [`ExtendedQuantType`], preferring one of the pre-existing named
    /// variants when there is an exact match (source/behaviour
    /// compatibility for callers written against the original, smaller
    /// table) and falling back to [`ExtendedQuantType::Other`] for
    /// everything else.
    fn from(ty: GgufTensorType) -> Self {
        match ty {
            GgufTensorType::F32 => Self::F32,
            GgufTensorType::F16 => Self::F16,
            GgufTensorType::Q4_0 => Self::Q4_0,
            GgufTensorType::Q4_1 => Self::Q4_1,
            GgufTensorType::Q5_0 => Self::Q5_0,
            GgufTensorType::Q5_1 => Self::Q5_1,
            GgufTensorType::Q8_0 => Self::Q8_0,
            GgufTensorType::Q8_1 => Self::Q8_1,
            GgufTensorType::Q2_K => Self::Q2_K,
            GgufTensorType::Q3_K => Self::Q3_K,
            GgufTensorType::Q4_K => Self::Q4_K,
            GgufTensorType::Q5_K => Self::Q5_K,
            GgufTensorType::Q6_K => Self::Q6_K,
            GgufTensorType::Q8_K => Self::Q8_K,
            GgufTensorType::Q1_0_g128 => Self::Q1_0_G128,
            GgufTensorType::F8_E4M3 => Self::F8_E4M3,
            GgufTensorType::F8_E5M2 => Self::F8_E5M2,
            other => Self::Other(other),
        }
    }
}

impl ExtendedQuantType {
    /// Construct from a raw GGUF tensor type ID.
    ///
    /// Derived from [`GgufTensorType::from_id`] (core-gguf-05) rather than
    /// an independently maintained id list: any id the parser recognises is
    /// "known" here too, by construction, so the two tables can never again
    /// silently disagree about which ids this build can parse. An id the
    /// parser does *not* recognise becomes
    /// [`Unknown(id)`](ExtendedQuantType::Unknown).
    pub fn from_u32(v: u32) -> Self {
        match GgufTensorType::from_id(v) {
            Ok(ty) => Self::from(ty),
            Err(_) => Self::Unknown(v),
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
            // `wire_id()`, never `as u32`: `GgufTensorType` carries sentinel
            // discriminants (`Q2_0G64`/`Q2_0G128DFirst`) for the two other
            // readings of ambiguous id 42 that a raw cast would not map
            // back to 42.
            Self::Other(ty) => ty.wire_id(),
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
            // Q8_1: 32 weights at 8 bits + `ggml_half2 ds` (2×f16 scale/min)
            // bytes = 2*2 + 32 = 36 (REQUIRED #3: ggml-common.h:297, not the
            // 40-byte figure an earlier version of this file used);
            // bits_per_w = 36*8/32 = 9.0
            Self::Q8_1 => 9.0,
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
            // Generic formula for everything else this build's parser
            // knows: `block_bytes * 8 / block_size`, computed from
            // `GgufTensorType` directly rather than a second hand-copied
            // per-variant table (which is exactly how the Q8_1 40-vs-36
            // byte drift this file used to have happened in the first
            // place).
            Self::Other(ty) => (ty.block_bytes() as f32) * 8.0 / (ty.block_size() as f32),
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
            // `GgufTensorType::name()` is itself `&'static str`, so this
            // stays exact instead of collapsing to a generic label.
            Self::Other(ty) => ty.name(),
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
    /// Raw quantization type IDs that were not recognised, deduplicated and
    /// sorted (core-gguf-17).
    ///
    /// Before this fix, one entry was pushed **per tensor**, so a real 27B
    /// PQ2_0 file (402 tensors of id 142 + 96 of id 30) produced a 498-entry
    /// `Vec` here — and [`finalize`](Self::finalize) formatted the whole
    /// thing into a single warning string, which is what actually flooded a
    /// terminal (not this field). A `BTreeSet` is both the fix (dedup) and
    /// the "sorted unique ids" accessor shape the finding asked for, kept
    /// as a **field** rather than a same-named method: this exact
    /// `is_empty()`/`len()`/`contains(&id)` field-access shape is what
    /// `tests/gguf_compat_tests.rs` (outside this package's owned files)
    /// already exercises, and a field and a method cannot share one name.
    /// See [`Self::unknown_quant_type_counts`] for how many tensors use
    /// each id.
    pub unknown_quant_types: BTreeSet<u32>,
    /// How many tensors use each unrecognised quantization type ID — the
    /// `BTreeMap<u32, u64>` core-gguf-17 asked for. `unknown_quant_types`
    /// is this map's key set (sorted, deduplicated by construction).
    pub unknown_quant_type_counts: BTreeMap<u32, u64>,
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
            unknown_quant_types: BTreeSet::new(),
            unknown_quant_type_counts: BTreeMap::new(),
            warnings: Vec::new(),
            is_loadable: true,
        }
    }

    /// Record one tensor's use of an unrecognised quantization type ID.
    ///
    /// Deduplicated into [`unknown_quant_types`](Self::unknown_quant_types);
    /// every occurrence (one per affected tensor) still increments
    /// [`unknown_quant_type_counts`](Self::unknown_quant_type_counts).
    pub fn add_unknown_quant(&mut self, quant_id: u32) {
        self.unknown_quant_types.insert(quant_id);
        *self.unknown_quant_type_counts.entry(quant_id).or_insert(0) += 1;
    }

    /// Append a human-readable warning to the report.
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Finalise the report.
    ///
    /// Warnings are informational and do not prevent loading. However, if any
    /// tensors use unknown quant types OxiBonsai cannot decode them, so
    /// `is_loadable` is set to `false` and a synthesised warning is added.
    ///
    /// The synthesised warning names the deduplicated set of unrecognised
    /// ids and the total affected tensor count separately (core-gguf-17):
    /// formatting the un-deduplicated per-tensor list into one string —
    /// e.g. 498 repeats of `142`/`30` for a real 27B PQ2_0 file — was what
    /// actually flooded a terminal, not the field it was drawn from.
    pub fn finalize(&mut self) {
        if !self.unknown_quant_types.is_empty() {
            self.is_loadable = false;
            let total_tensors: u64 = self.unknown_quant_type_counts.values().sum();
            self.add_warning(format!(
                "file contains {total_tensors} tensor(s) across {} unrecognised quantization \
                 type(s): {:?}",
                self.unknown_quant_types.len(),
                self.unknown_quant_types,
            ));
        }
    }

    /// Return a single-line human-readable summary of the report.
    pub fn summary(&self) -> String {
        format!(
            "GGUF {} | tensors={} metadata={} | unknown_quants={} types ({} tensors) | \
             warnings={} | loadable={}",
            self.version,
            self.tensor_count,
            self.metadata_count,
            self.unknown_quant_types.len(),
            self.unknown_quant_type_counts.values().sum::<u64>(),
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

    // ── core-gguf-05: unified quant-compat table ────────────────────────────

    /// The three ids the finding specifically named as missing from the old
    /// table (30 = BF16, 35 = TQ2_0, 42 = the legacy `TQ2_0_g128` reading)
    /// must now be `is_known()`, not `Unknown`.
    #[test]
    fn from_u32_knows_the_previously_missing_ids() {
        for (id, name) in [(30u32, "BF16"), (35, "TQ2_0"), (42, "TQ2_0_g128")] {
            let qt = ExtendedQuantType::from_u32(id);
            assert!(qt.is_known(), "id {id} ({name}) must be known: {qt:?}");
            assert_eq!(qt.name(), name, "id {id}");
            assert_eq!(qt.to_u32(), id, "round trip for id {id}");
        }
    }

    /// The two Bonsai 2 27B formats (PQ2_0 = 142, PTQ1_0 = 143) must also be
    /// known — this is the exact reproduction from the finding's evidence
    /// ("on the 27B PQ2_0 header, `unknown = [30, 142]`").
    #[test]
    fn from_u32_knows_the_bonsai_2_27b_formats() {
        for (id, name) in [(142u32, "PQ2_0"), (143, "PTQ1_0")] {
            let qt = ExtendedQuantType::from_u32(id);
            assert!(qt.is_known(), "id {id} ({name}) must be known: {qt:?}");
            assert_eq!(qt.name(), name);
        }
    }

    /// Ids `GgufTensorType::from_id` genuinely does not recognise must stay
    /// `Unknown` — the fix must not make everything "known".
    #[test]
    fn from_u32_still_reports_genuinely_unknown_ids() {
        for id in [4u32, 5, 31, 100, 999_999, u32::MAX] {
            assert_eq!(
                ExtendedQuantType::from_u32(id),
                ExtendedQuantType::Unknown(id)
            );
        }
    }

    /// `Other`'s `bits_per_weight` must be computed from the real
    /// `GgufTensorType` geometry, not a stale hand-copied constant — this is
    /// what the Q8_1 40-vs-36-byte drift (REQUIRED #3) would have shown up
    /// as if `Other` had instead hard-coded a value.
    #[test]
    fn other_variant_bits_per_weight_matches_block_geometry() {
        let bf16 = ExtendedQuantType::from_u32(30);
        assert_eq!(bf16, ExtendedQuantType::Other(GgufTensorType::BF16));
        assert_eq!(bf16.bits_per_weight(), 16.0);

        let tq2_0 = ExtendedQuantType::from_u32(35);
        // 66 bytes / 256 weights * 8 = 2.0625 bits/weight.
        assert!((tq2_0.bits_per_weight() - 2.0625).abs() < 1e-4);
    }

    #[test]
    fn q8_1_bits_per_weight_reflects_the_corrected_36_byte_block() {
        assert_eq!(ExtendedQuantType::Q8_1.bits_per_weight(), 9.0);
    }

    // ── core-gguf-17: deduplicated unknown_quant_types ──────────────────────

    /// Reproduces the finding's exact numbers (402 tensors of one
    /// unrecognised id, 96 of another — the real 27B PQ2_0 file's
    /// `402×142 + 96×30` shape) and asserts they dedupe to a two-entry set,
    /// with the per-id counts preserved separately. Uses stand-in ids
    /// `8888`/`9999` rather than the real `142`/`30`, because both of those
    /// are themselves now known after the core-gguf-05 fix and would not
    /// exercise the unknown-id path at all.
    #[test]
    fn build_compat_report_deduplicates_the_27b_pattern() {
        let mut ids = vec![8888u32; 402];
        ids.extend(vec![9999u32; 96]);
        let report = build_compat_report(3, ids.len() as u64, 0, &ids);

        assert_eq!(report.unknown_quant_types.len(), 2, "must dedupe to 2 ids");
        assert!(report.unknown_quant_types.contains(&8888));
        assert!(report.unknown_quant_types.contains(&9999));
        assert_eq!(report.unknown_quant_type_counts.get(&8888), Some(&402));
        assert_eq!(report.unknown_quant_type_counts.get(&9999), Some(&96));
        assert_eq!(
            report.unknown_quant_type_counts.values().sum::<u64>(),
            498,
            "total affected tensor count must still be the un-deduplicated 498"
        );
        assert!(!report.is_loadable);
    }

    /// `finalize()`'s synthesised warning must name the deduplicated set,
    /// not format hundreds of repeated ids into one string.
    #[test]
    fn finalize_warning_is_not_flooded_by_duplicate_ids() {
        let mut report = GgufCompatReport::new(GgufVersion::V3, 498, 0);
        for _ in 0..402 {
            report.add_unknown_quant(8888);
        }
        for _ in 0..96 {
            report.add_unknown_quant(9999);
        }
        report.finalize();
        assert_eq!(report.warnings.len(), 1);
        let warning = &report.warnings[0];
        // The dedup set has 2 entries; the un-deduplicated 498 must appear
        // as a count, not as 498 repeated occurrences of "8888"/"9999".
        assert!(warning.contains("498"), "warning: {warning}");
        assert!(warning.contains('2'), "warning: {warning}");
        assert!(
            warning.len() < 200,
            "a flooded warning would be far longer than this: {warning}"
        );
    }

    #[test]
    fn summary_includes_both_the_type_count_and_the_tensor_count() {
        let mut report = GgufCompatReport::new(GgufVersion::V3, 100, 10);
        report.add_unknown_quant(8888);
        report.add_unknown_quant(8888);
        report.add_unknown_quant(9999);
        report.finalize();
        let summary = report.summary();
        assert!(summary.contains("unknown_quants=2 types"), "{summary}");
        assert!(summary.contains("3 tensors"), "{summary}");
    }
}
