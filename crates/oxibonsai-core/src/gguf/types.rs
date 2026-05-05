//! GGUF data type enumerations.
//!
//! Defines the tensor quantization types and metadata value types
//! used in the GGUF file format.

use crate::error::{BonsaiError, BonsaiResult};

/// GGUF tensor quantization types.
///
/// OxiBonsai focuses on Q1\_0\_g128 (type ID 41, PrismML extension),
/// TQ2\_0\_g128 (type ID 42, PrismML ternary extension), and TQ2\_0
/// (type ID 35, llama.cpp upstream ternary), but recognizes other types
/// for metadata-only access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    BF16 = 30,
    /// llama.cpp ternary quantization: 256 sign-2 bits + FP16 group scale (upstream ID 35).
    TQ2_0 = 35,
    /// PrismML 1-bit quantization: 128 sign bits + FP16 group scale.
    Q1_0_g128 = 41,
    /// PrismML ternary quantization: 128 sign-2 bits + FP16 group scale.
    TQ2_0_g128 = 42,
    /// PrismML FP8 E4M3FN: 32 weights × 1 byte + FP16 scale (type ID 43).
    F8_E4M3 = 43,
    /// PrismML FP8 E5M2: 32 weights × 1 byte + FP16 scale (type ID 44).
    F8_E5M2 = 44,
}

impl GgufTensorType {
    /// Parse a tensor type from its numeric GGUF type ID.
    pub fn from_id(id: u32) -> BonsaiResult<Self> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            30 => Ok(Self::BF16),
            35 => Ok(Self::TQ2_0),
            41 => Ok(Self::Q1_0_g128),
            42 => Ok(Self::TQ2_0_g128),
            43 => Ok(Self::F8_E4M3),
            44 => Ok(Self::F8_E5M2),
            _ => Err(BonsaiError::UnsupportedQuantType { type_id: id }),
        }
    }

    /// Number of elements per quantized block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::Q1_0_g128 => 128,
            Self::TQ2_0_g128 => 128,
            Self::TQ2_0 => 256,
            Self::F8_E4M3 | Self::F8_E5M2 => 32,
        }
    }

    /// Number of bytes per quantized block.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,       // 2 + 16
            Self::Q4_1 => 20,       // 2 + 2 + 16
            Self::Q5_0 => 22,       // 2 + 4 + 16
            Self::Q5_1 => 24,       // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,       // 2 + 32
            Self::Q8_1 => 40,       // 4 + 4 + 32
            Self::Q2_K => 84,       // 256/4 + 256/16 + 2+2
            Self::Q3_K => 110,      // 256/4 + 256/8 + 12+2
            Self::Q4_K => 144,      // 2+2+12+4*32
            Self::Q5_K => 176,      // 2+2+12+4*32+256/8
            Self::Q6_K => 210,      // 256/2+256/4+256/16+2
            Self::Q8_K => 292,      // 4+256+256/16
            Self::Q1_0_g128 => 18,  // 2 (FP16 scale) + 16 (128 sign bits)
            Self::TQ2_0_g128 => 34, // 2 (FP16 scale) + 32 (128 ternary-2bit packed)
            Self::TQ2_0 => 66,      // 2 (FP16 scale) + 64 (256 ternary-2bit packed)
            Self::F8_E4M3 | Self::F8_E5M2 => 34, // 32 bytes qs + 2 bytes FP16 scale
        }
    }

    /// Returns true if this is the Q1\_0\_g128 1-bit quantization type.
    pub fn is_one_bit(&self) -> bool {
        matches!(self, Self::Q1_0_g128)
    }

    /// Returns true if this is a ternary ({-1, 0, +1}) quantization type.
    pub fn is_ternary(&self) -> bool {
        matches!(self, Self::TQ2_0 | Self::TQ2_0_g128)
    }

    /// Returns true if this is an FP8 quantization type.
    pub fn is_fp8(self) -> bool {
        matches!(self, Self::F8_E4M3 | Self::F8_E5M2)
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
            Self::BF16 => "BF16",
            Self::TQ2_0 => "TQ2_0",
            Self::Q1_0_g128 => "Q1_0_g128",
            Self::TQ2_0_g128 => "TQ2_0_g128",
            Self::F8_E4M3 => "F8_E4M3",
            Self::F8_E5M2 => "F8_E5M2",
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
}
