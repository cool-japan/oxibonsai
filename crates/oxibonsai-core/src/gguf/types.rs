//! GGUF data type enumerations.
//!
//! Defines the tensor quantization types and metadata value types
//! used in the GGUF file format.

use crate::error::{BonsaiError, BonsaiResult};

/// GGUF tensor quantization types.
///
/// The table mirrors upstream `enum ggml_type` (ids 0..=40) plus the
/// OxiBonsai/PrismML extensions, so a GGUF using a type this build cannot
/// *execute* still parses cleanly for `info`/`validate` (core-gguf-12).
/// Use [`GgufTensorType::is_executable`] to decide whether a kernel exists;
/// the loader — not the parser — refuses non-executable types.
///
/// # ggml id 42 is ambiguous
///
/// Three different on-disk layouts ship under ggml id 42:
///
/// | file family | group | bytes | byte order | variant |
/// |---|---|---|---|---|
/// | `Ternary-Bonsai-{1.7B,8B}` (OxiBonsai's own writer) | 128 | 34 | `qs` first, `d` last | [`GgufTensorType::TQ2_0_g128`] |
/// | `Ternary-Bonsai-27B-Q2_0` (PrismML gen-1) | 128 | 34 | `d` first, `qs` last | [`GgufTensorType::Q2_0G128DFirst`] |
/// | `Ternary-Bonsai-2-27B-Q2_0` (mainline `block_q2_0`) | 64 | 18 | `d` first, `qs` last | [`GgufTensorType::Q2_0G64`] |
///
/// [`GgufTensorType::from_id`] keeps the historical behaviour and maps 42 to
/// `TQ2_0_g128`; [`GgufTensorType::from_id_marked`] returns
/// [`TypeIdResolution::Ambiguous42`] instead, which
/// [`crate::gguf::quant_resolve`] settles from the file's offset table and a
/// structural sniff of the real bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    /// llama.cpp ternary quantization, base-3 trit packing, 256-element groups
    /// (upstream ID 34).
    TQ1_0 = 34,
    /// llama.cpp ternary quantization: 256 sign-2 bits + FP16 group scale (upstream ID 35).
    TQ2_0 = 35,
    /// Microscaling FP4, 32-element blocks with an E8M0 byte scale (upstream ID 39).
    MXFP4 = 39,
    /// NVIDIA FP4, 64-element blocks with four E4M3 sub-block scales (upstream ID 40).
    NVFP4 = 40,
    /// PrismML 1-bit quantization: 128 sign bits + FP16 group scale.
    Q1_0_g128 = 41,
    /// **Legacy OxiBonsai** ternary, ggml id 42, group 128, 34 B, `qs` FIRST / `d` LAST.
    ///
    /// This is the layout `models/Ternary-Bonsai-{1.7B,8B}.gguf` actually use;
    /// it matches neither PrismML spelling of id 42 and must not be changed.
    TQ2_0_g128 = 42,
    /// PrismML FP8 E4M3FN: 32 weights × 1 byte + FP16 scale (type ID 43).
    F8_E4M3 = 43,
    /// PrismML FP8 E5M2: 32 weights × 1 byte + FP16 scale (type ID 44).
    F8_E5M2 = 44,
    /// **Mainline `block_q2_0`** — ggml id 42 read as group 64 / 18 B, `d` FIRST.
    ///
    /// Carries a private sentinel discriminant because [`Self::TQ2_0_g128`]
    /// already owns `42` and `#[repr(u32)]` forbids duplicates. Serialisation
    /// must always go through [`Self::wire_id`], never `as u32`.
    Q2_0G64 = 0x4000_002A,
    /// **PrismML gen-1** ternary — ggml id 42 read as group 128 / 34 B, `d` FIRST.
    ///
    /// Wire-identical to [`Self::PQ2_0`] but stored under id 42. Sentinel
    /// discriminant for the same reason as [`Self::Q2_0G64`].
    Q2_0G128DFirst = 0x4001_002A,
    /// PrismML PQ2_0 (ggml id 142): group 128, 34 B, `d` FIRST then `qs[32]`.
    PQ2_0 = 142,
    /// PrismML PTQ1_0 (ggml id 143): group 128, 28 B, `qs[24]`, `qh[2]`, `d` LAST.
    PTQ1_0 = 143,
}

/// Outcome of mapping a raw on-disk ggml type id to a [`GgufTensorType`].
///
/// ggml id 42 is the only id that does not determine a layout on its own, so
/// it gets its own marker rather than a silently-chosen default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeIdResolution {
    /// The id maps to exactly one variant.
    Unique(GgufTensorType),
    /// ggml id 42: one of `TQ2_0_g128` / `Q2_0G128DFirst` / `Q2_0G64`.
    ///
    /// Settle it with [`crate::gguf::quant_resolve::resolve_type_42`].
    Ambiguous42,
}

impl GgufTensorType {
    /// Every variant, in declaration order.
    pub const ALL: &'static [GgufTensorType] = &[
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
        Self::IQ2_XXS,
        Self::IQ2_XS,
        Self::IQ3_XXS,
        Self::IQ1_S,
        Self::IQ4_NL,
        Self::IQ3_S,
        Self::IQ2_S,
        Self::IQ4_XS,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::IQ1_M,
        Self::BF16,
        Self::TQ1_0,
        Self::TQ2_0,
        Self::MXFP4,
        Self::NVFP4,
        Self::Q1_0_g128,
        Self::TQ2_0_g128,
        Self::F8_E4M3,
        Self::F8_E5M2,
        Self::Q2_0G64,
        Self::Q2_0G128DFirst,
        Self::PQ2_0,
        Self::PTQ1_0,
    ];

    /// Parse a tensor type from its numeric GGUF type ID.
    ///
    /// Historical behaviour is preserved for ggml id 42, which resolves to the
    /// legacy [`Self::TQ2_0_g128`] (qs-first) reading so the 1.7B/8B load path
    /// does not move. Callers that must not guess — the resolver and any new
    /// loader — use [`Self::from_id_marked`] instead.
    pub fn from_id(id: u32) -> BonsaiResult<Self> {
        match Self::from_id_marked(id)? {
            TypeIdResolution::Unique(ty) => Ok(ty),
            TypeIdResolution::Ambiguous42 => Ok(Self::TQ2_0_g128),
        }
    }

    /// Parse a tensor type from its numeric GGUF type ID, marking ggml id 42
    /// as ambiguous instead of guessing a layout.
    ///
    /// Accepts every id in the ggml type table, including ones this build can
    /// parse but not execute (see [`Self::is_executable`]); only ids that are
    /// not ggml types at all are rejected.
    pub fn from_id_marked(id: u32) -> BonsaiResult<TypeIdResolution> {
        let ty = match id {
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
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            40 => Self::NVFP4,
            41 => Self::Q1_0_g128,
            42 => return Ok(TypeIdResolution::Ambiguous42),
            43 => Self::F8_E4M3,
            44 => Self::F8_E5M2,
            142 => Self::PQ2_0,
            143 => Self::PTQ1_0,
            _ => return Err(BonsaiError::UnsupportedQuantType { type_id: id }),
        };
        Ok(TypeIdResolution::Unique(ty))
    }

    /// The ggml type id this variant is *stored* under on disk.
    ///
    /// Always use this for serialisation. The `Q2_0G64` / `Q2_0G128DFirst`
    /// sentinel discriminants (`0x4000_002A`, `0x4001_002A`) exist only so the
    /// three readings of id 42 can coexist in one `#[repr(u32)]` enum; writing
    /// `self as u32` would emit the sentinel verbatim.
    pub fn wire_id(self) -> u32 {
        match self {
            Self::Q2_0G64 | Self::Q2_0G128DFirst => 42,
            other => other as u32,
        }
    }

    /// Number of elements per quantized block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::NVFP4 | Self::Q2_0G64 => 64,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => 256,
            Self::Q1_0_g128
            | Self::TQ2_0_g128
            | Self::Q2_0G128DFirst
            | Self::PQ2_0
            | Self::PTQ1_0 => 128,
            Self::F8_E4M3 | Self::F8_E5M2 => 32,
        }
    }

    /// Number of bytes per quantized block.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 => 18,                                            // 2 + 16
            Self::Q4_1 => 20,                                            // 2 + 2 + 16
            Self::Q5_0 => 22,                                            // 2 + 4 + 16
            Self::Q5_1 => 24,                                            // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,                                            // 2 + 32
            Self::Q8_1 => 36,    // 2*f16 (ggml_half2 `ds`) + 32 (ggml-common.h:297)
            Self::Q2_K => 84,    // 256/4 + 256/16 + 2+2
            Self::Q3_K => 110,   // 256/4 + 256/8 + 12+2
            Self::Q4_K => 144,   // 2+2+12+4*32
            Self::Q5_K => 176,   // 2+2+12+4*32+256/8
            Self::Q6_K => 210,   // 256/2+256/4+256/16+2
            Self::Q8_K => 292,   // 4+256+256/16
            Self::IQ2_XXS => 66, // 2 + 256/8*2
            Self::IQ2_XS => 74,  // 2 + 256/8*2 + 256/32
            Self::IQ3_XXS => 98, // 2 + 3*(256/8)
            Self::IQ1_S => 50,   // 2 + 256/8 + 256/16
            Self::IQ4_NL => 18,  // 2 + 32/2
            Self::IQ3_S => 110,  // 2 + 13*(256/32) + 256/64
            Self::IQ2_S => 82,   // 2 + 256/4 + 256/16
            Self::IQ4_XS => 136, // 2 + 2 + 256/64 + 256/2
            Self::IQ1_M => 56,   // 256/8 + 256/16 + 256/32
            Self::TQ1_0 => 54,   // 2 + 256/64 + (256 - 4*256/64)/5
            Self::TQ2_0 => 66,   // 2 (FP16 scale) + 64 (256 ternary-2bit packed)
            Self::MXFP4 => 17,   // 1 (E8M0) + 32/2
            Self::NVFP4 => 36,   // 64/16 (E4M3 sub-scales) + 64/2
            Self::Q1_0_g128 => 18, // 2 (FP16 scale) + 16 (128 sign bits)
            Self::TQ2_0_g128 | Self::Q2_0G128DFirst | Self::PQ2_0 => 34, // 2 + 32
            Self::PTQ1_0 => 28,  // 24 (qs) + 2 (qh) + 2 (FP16 scale)
            Self::Q2_0G64 => 18, // 2 (FP16 scale) + 16 (64 × 2-bit)
            Self::F8_E4M3 | Self::F8_E5M2 => 34, // 32 bytes qs + 2 bytes FP16 scale
        }
    }

    /// Returns true if this is the Q1\_0\_g128 1-bit quantization type.
    pub fn is_one_bit(&self) -> bool {
        matches!(self, Self::Q1_0_g128)
    }

    /// Returns true if this is a ternary ({-1, 0, +1}) quantization type.
    ///
    /// The 2-bit Q2_0 family (`Q2_0G64`, `Q2_0G128DFirst`, `PQ2_0`) is a
    /// general 4-level codec whose fourth code means `+2`; PrismML only ever
    /// emits the three ternary levels, which is exactly the invariant the
    /// layout sniff exploits, so they are counted here.
    pub fn is_ternary(&self) -> bool {
        matches!(
            self,
            Self::TQ1_0
                | Self::TQ2_0
                | Self::TQ2_0_g128
                | Self::Q2_0G64
                | Self::Q2_0G128DFirst
                | Self::PQ2_0
                | Self::PTQ1_0
        )
    }

    /// Returns true if this variant is stored as 2-bit lanes, 4 per byte.
    ///
    /// These are the types the `0b11`-count layout sniff applies to.
    pub fn is_two_bit_packed(self) -> bool {
        matches!(
            self,
            Self::TQ2_0 | Self::TQ2_0_g128 | Self::Q2_0G64 | Self::Q2_0G128DFirst | Self::PQ2_0
        )
    }

    /// Returns true if this is an FP8 quantization type.
    pub fn is_fp8(self) -> bool {
        matches!(self, Self::F8_E4M3 | Self::F8_E5M2)
    }

    /// Returns true when the on-disk block places the FP16 scale FIRST.
    ///
    /// `None` for types that have no single FP16 block scale.
    pub fn scale_is_first(self) -> Option<bool> {
        match self {
            Self::Q2_0G64 | Self::Q2_0G128DFirst | Self::PQ2_0 | Self::Q1_0_g128 => Some(true),
            Self::TQ2_0 | Self::TQ2_0_g128 | Self::PTQ1_0 => Some(false),
            _ => None,
        }
    }

    /// Returns true when a weight of this type participates in the PrismML
    /// Hadamard rotation (`prism.hadamard.*`).
    pub fn is_prism_rotatable(self) -> bool {
        matches!(
            self,
            Self::PQ2_0
                | Self::PTQ1_0
                | Self::Q2_0G64
                | Self::Q2_0G128DFirst
                | Self::TQ2_0_g128
                | Self::Q1_0_g128
        )
    }

    /// Returns true when this build has a decoder/kernel for this type.
    ///
    /// The *parser* accepts every ggml type so headers stay readable; the
    /// *loader* refuses anything for which this returns `false`
    /// (`BonsaiError::NonExecutableQuantType`).
    pub fn is_executable(&self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::Q4_0
                | Self::Q8_0
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::Q8_K
                | Self::TQ2_0
                | Self::Q1_0_g128
                | Self::TQ2_0_g128
                | Self::F8_E4M3
                | Self::F8_E5M2
                | Self::Q2_0G64
                | Self::Q2_0G128DFirst
                | Self::PQ2_0
                | Self::PTQ1_0
        )
    }

    /// Display name for this quantization type.
    pub fn name(&self) -> &'static str {
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
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
            Self::NVFP4 => "NVFP4",
            Self::Q1_0_g128 => "Q1_0_g128",
            Self::TQ2_0_g128 => "TQ2_0_g128",
            Self::F8_E4M3 => "F8_E4M3",
            Self::F8_E5M2 => "F8_E5M2",
            Self::Q2_0G64 => "Q2_0_g64",
            Self::Q2_0G128DFirst => "Q2_0_g128_d_first",
            Self::PQ2_0 => "PQ2_0",
            Self::PTQ1_0 => "PTQ1_0",
        }
    }
}

impl std::fmt::Display for GgufTensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// GGUF metadata value types (for the key-value store).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
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

impl GgufValueType {
    /// Parse a value type from its numeric GGUF type ID.
    pub fn from_id(id: u32) -> BonsaiResult<Self> {
        match id {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(BonsaiError::InvalidMetadata {
                key: String::new(),
                reason: format!("unknown value type id: {id}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q1_0_g128_properties() {
        let ty = GgufTensorType::Q1_0_g128;
        assert_eq!(ty.block_size(), 128);
        assert_eq!(ty.block_bytes(), 18);
        assert!(ty.is_one_bit());
        assert_eq!(ty.name(), "Q1_0_g128");
        assert_eq!(ty as u32, 41);
    }

    #[test]
    fn parse_known_type_ids() {
        assert_eq!(
            GgufTensorType::from_id(0).expect("type id 0 is valid"),
            GgufTensorType::F32
        );
        assert_eq!(
            GgufTensorType::from_id(1).expect("type id 1 is valid"),
            GgufTensorType::F16
        );
        assert_eq!(
            GgufTensorType::from_id(41).expect("type id 41 is valid"),
            GgufTensorType::Q1_0_g128
        );
    }

    #[test]
    fn reject_unknown_type_id() {
        // 42 is now TQ2_0_g128; use a truly unregistered ID
        assert!(GgufTensorType::from_id(50).is_err());
        assert!(GgufTensorType::from_id(100).is_err());
    }

    #[test]
    fn tq2_0_g128_ternary_properties() {
        let ty = GgufTensorType::TQ2_0_g128;
        assert_eq!(ty.block_size(), 128);
        assert_eq!(ty.block_bytes(), 34);
        assert!(ty.is_ternary());
        assert!(!ty.is_one_bit());
        assert_eq!(ty.name(), "TQ2_0_g128");
        assert_eq!(ty as u32, 42);
    }

    #[test]
    fn tq2_0_ternary_properties() {
        let ty = GgufTensorType::TQ2_0;
        assert_eq!(ty.block_size(), 256);
        assert_eq!(ty.block_bytes(), 66);
        assert!(ty.is_ternary());
        assert!(!ty.is_one_bit());
        assert_eq!(ty.name(), "TQ2_0");
        assert_eq!(ty as u32, 35);
    }

    #[test]
    fn parse_ternary_type_ids() {
        assert_eq!(
            GgufTensorType::from_id(42).expect("42 valid"),
            GgufTensorType::TQ2_0_g128
        );
        let tq2_id = GgufTensorType::TQ2_0 as u32;
        assert_eq!(
            GgufTensorType::from_id(tq2_id).expect("TQ2_0 id valid"),
            GgufTensorType::TQ2_0
        );
    }

    #[test]
    fn one_bit_is_not_ternary() {
        assert!(!GgufTensorType::Q1_0_g128.is_ternary());
        assert!(GgufTensorType::Q1_0_g128.is_one_bit());
    }

    #[test]
    fn f8_e4m3_properties() {
        let ty = GgufTensorType::F8_E4M3;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 34);
        assert!(ty.is_fp8());
        assert!(!ty.is_ternary());
        assert!(!ty.is_one_bit());
        assert_eq!(ty.name(), "F8_E4M3");
        assert_eq!(ty as u32, 43);
    }

    #[test]
    fn f8_e5m2_properties() {
        let ty = GgufTensorType::F8_E5M2;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 34);
        assert!(ty.is_fp8());
        assert!(!ty.is_ternary());
        assert!(!ty.is_one_bit());
        assert_eq!(ty.name(), "F8_E5M2");
        assert_eq!(ty as u32, 44);
    }

    #[test]
    fn parse_fp8_type_ids() {
        assert_eq!(
            GgufTensorType::from_id(43).expect("43 valid"),
            GgufTensorType::F8_E4M3
        );
        assert_eq!(
            GgufTensorType::from_id(44).expect("44 valid"),
            GgufTensorType::F8_E5M2
        );
    }

    // ── New: the full upstream table (core-gguf-12) ────────────────────────

    /// Every id the upstream ggml type table defines must parse, with the
    /// block geometry ggml itself declares. Before this, `from_id` rejected
    /// 16..=29, 34, 39 and 40 outright, so `oxibonsai info` on an IQ4_XS /
    /// MXFP4 / TQ1_0 GGUF failed instead of printing the header.
    #[test]
    fn upstream_type_table_matches_ggml() {
        // (id, block_size, block_bytes) transcribed from ggml-common.h.
        let table: &[(u32, usize, usize)] = &[
            (0, 1, 4),
            (1, 1, 2),
            (2, 32, 18),
            (3, 32, 20),
            (6, 32, 22),
            (7, 32, 24),
            (8, 32, 34),
            (9, 32, 36),
            (10, 256, 84),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (15, 256, 292),
            (16, 256, 66),
            (17, 256, 74),
            (18, 256, 98),
            (19, 256, 50),
            (20, 32, 18),
            (21, 256, 110),
            (22, 256, 82),
            (23, 256, 136),
            (24, 1, 1),
            (25, 1, 2),
            (26, 1, 4),
            (27, 1, 8),
            (28, 1, 8),
            (29, 256, 56),
            (30, 1, 2),
            (34, 256, 54),
            (35, 256, 66),
            (39, 32, 17),
            (40, 64, 36),
            (41, 128, 18),
            (43, 32, 34),
            (44, 32, 34),
            (142, 128, 34),
            (143, 128, 28),
        ];
        for &(id, blk, bytes) in table {
            let ty = GgufTensorType::from_id(id)
                .unwrap_or_else(|e| panic!("ggml type id {id} must parse: {e}"));
            assert_eq!(ty.wire_id(), id, "wire_id round-trip for {ty}");
            assert_eq!(ty.block_size(), blk, "block_size for {ty} (id {id})");
            assert_eq!(ty.block_bytes(), bytes, "block_bytes for {ty} (id {id})");
        }
    }

    #[test]
    fn removed_and_unassigned_ids_are_still_rejected() {
        // 4/5 (Q4_2/Q4_3) and 31..33/36..38 were removed upstream; 45..141
        // are unassigned; 200 is beyond the table entirely.
        for bad in [
            4u32,
            5,
            31,
            32,
            33,
            36,
            37,
            38,
            45,
            100,
            141,
            144,
            200,
            u32::MAX,
        ] {
            assert!(
                GgufTensorType::from_id(bad).is_err(),
                "ggml type id {bad} must stay unsupported"
            );
        }
    }

    #[test]
    fn non_executable_types_still_parse() {
        for id in [16u32, 17, 18, 19, 20, 21, 22, 23, 29, 34, 39, 40] {
            let ty = GgufTensorType::from_id(id).expect("parses");
            assert!(
                !ty.is_executable(),
                "{ty} has no kernel in this build and must report so"
            );
        }
        for id in [0u32, 1, 30, 41, 42, 142, 143] {
            let ty = GgufTensorType::from_id(id).expect("parses");
            assert!(ty.is_executable(), "{ty} must be executable");
        }
    }

    // ── New: the id-42 split and the sentinel discriminants ────────────────

    #[test]
    fn id_42_is_marked_ambiguous_not_guessed() {
        assert_eq!(
            GgufTensorType::from_id_marked(42).expect("42 parses"),
            TypeIdResolution::Ambiguous42
        );
        // …while the legacy constructor keeps the historical reading.
        assert_eq!(
            GgufTensorType::from_id(42).expect("42 parses"),
            GgufTensorType::TQ2_0_g128
        );
    }

    #[test]
    fn from_id_marked_is_unique_for_every_other_id() {
        for ty in GgufTensorType::ALL {
            if ty.wire_id() == 42 {
                continue;
            }
            assert_eq!(
                GgufTensorType::from_id_marked(ty.wire_id()).expect("parses"),
                TypeIdResolution::Unique(*ty),
                "id {} must map uniquely",
                ty.wire_id()
            );
        }
    }

    /// The sentinel discriminants must never reach a file. `wire_id()` maps
    /// both id-42 readings back to 42; a raw `as u32` would emit
    /// `0x4000_002A` / `0x4001_002A`.
    #[test]
    fn id_42_variants_share_wire_id_42_but_differ_as_values() {
        assert_eq!(GgufTensorType::Q2_0G64.wire_id(), 42);
        assert_eq!(GgufTensorType::Q2_0G128DFirst.wire_id(), 42);
        assert_eq!(GgufTensorType::TQ2_0_g128.wire_id(), 42);
        assert_eq!(GgufTensorType::Q2_0G64 as u32, 0x4000_002A);
        assert_eq!(GgufTensorType::Q2_0G128DFirst as u32, 0x4001_002A);
        assert_ne!(GgufTensorType::Q2_0G64, GgufTensorType::TQ2_0_g128);
        assert_ne!(GgufTensorType::Q2_0G128DFirst, GgufTensorType::TQ2_0_g128);
    }

    #[test]
    fn id_42_readings_have_the_probed_geometry() {
        // §0.2 block-layout probe.
        assert_eq!(GgufTensorType::TQ2_0_g128.block_size(), 128);
        assert_eq!(GgufTensorType::TQ2_0_g128.block_bytes(), 34);
        assert_eq!(GgufTensorType::TQ2_0_g128.scale_is_first(), Some(false));

        assert_eq!(GgufTensorType::Q2_0G128DFirst.block_size(), 128);
        assert_eq!(GgufTensorType::Q2_0G128DFirst.block_bytes(), 34);
        assert_eq!(GgufTensorType::Q2_0G128DFirst.scale_is_first(), Some(true));

        assert_eq!(GgufTensorType::Q2_0G64.block_size(), 64);
        assert_eq!(GgufTensorType::Q2_0G64.block_bytes(), 18);
        assert_eq!(GgufTensorType::Q2_0G64.scale_is_first(), Some(true));

        assert_eq!(GgufTensorType::PQ2_0.scale_is_first(), Some(true));
        assert_eq!(GgufTensorType::PTQ1_0.scale_is_first(), Some(false));
    }

    #[test]
    fn prism_types_are_ternary_and_rotatable() {
        for ty in [
            GgufTensorType::PQ2_0,
            GgufTensorType::PTQ1_0,
            GgufTensorType::Q2_0G64,
            GgufTensorType::Q2_0G128DFirst,
            GgufTensorType::TQ2_0_g128,
        ] {
            assert!(ty.is_ternary(), "{ty} must be ternary");
            assert!(ty.is_prism_rotatable(), "{ty} must be rotatable");
        }
        assert!(GgufTensorType::Q1_0_g128.is_prism_rotatable());
        assert!(!GgufTensorType::F32.is_prism_rotatable());
        assert!(!GgufTensorType::PTQ1_0.is_two_bit_packed());
        assert!(GgufTensorType::PQ2_0.is_two_bit_packed());
    }

    /// Every variant must have a non-empty, unique display name.
    #[test]
    fn all_names_unique_and_non_empty() {
        let mut names: Vec<&str> = GgufTensorType::ALL.iter().map(|t| t.name()).collect();
        assert!(names.iter().all(|n| !n.is_empty()));
        names.sort_unstable();
        let total = names.len();
        names.dedup();
        assert_eq!(names.len(), total, "display names must be unique");
    }

    /// Every variant must have a strictly positive block geometry, or the
    /// per-row size formula divides by zero.
    #[test]
    fn all_block_geometry_positive() {
        for ty in GgufTensorType::ALL {
            assert!(ty.block_size() > 0, "block_size for {ty}");
            assert!(ty.block_bytes() > 0, "block_bytes for {ty}");
        }
    }

    /// ggml's real `block_q8_1` is `ggml_half2 ds` (2×f16 = 4 bytes) plus
    /// `qs[32]`, i.e. 36 bytes (`ggml-common.h:297`) — not 40. A wrong
    /// `block_bytes()` here would mis-size every Q8_1 tensor's offset
    /// replay (REQUIRED #3, gatekeeper wave-1+1.5 review).
    #[test]
    fn q8_1_block_bytes_matches_ggml_common_h() {
        assert_eq!(GgufTensorType::Q8_1.block_bytes(), 36);
    }

    // ── core-gguf-05: unified quant-compat table ────────────────────────────

    /// `GgufTensorType::from_id` (the parser's own table) and
    /// `ExtendedQuantType::from_u32` (the forward-compat report's table,
    /// `crate::gguf::compat`) must agree on every id in `0..=143` — the
    /// full range this crate has ever assigned meaning to, plus the two
    /// Bonsai 2 27B ids (142, 143). Before the fix, `ExtendedQuantType` had
    /// its own independent, stale id list (missing 30/35/42, among others),
    /// so a tensor that `GgufTensorType::from_id` (and hence
    /// `TensorStore::parse`/`GgufFile::parse`) accepted and decoded fine
    /// was still reported as an "unrecognised quantization type" by the
    /// compat report — the false `loadable=false` WARN every ternary run
    /// printed (cli-15).
    #[test]
    fn extended_quant_type_agrees_with_gguf_tensor_type_for_every_id() {
        use crate::gguf::compat::ExtendedQuantType;

        let mut mismatches = Vec::new();
        for id in 0..=143u32 {
            let parser_knows = GgufTensorType::from_id(id).is_ok();
            let compat_knows = ExtendedQuantType::from_u32(id).is_known();
            if parser_knows != compat_knows {
                mismatches.push(format!(
                    "id {id}: GgufTensorType::from_id.is_ok()={parser_knows} but \
                     ExtendedQuantType::from_u32(..).is_known()={compat_knows}"
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "quant-compat tables disagree:\n{}",
            mismatches.join("\n")
        );
    }
}
