//! Error types for OxiBonsai core operations.

use thiserror::Error;

use crate::gguf::types::GgufTensorType;

/// Result type alias for OxiBonsai core operations.
pub type BonsaiResult<T> = Result<T, BonsaiError>;

/// Render the list of quantization types this build can actually execute.
///
/// Generated from [`GgufTensorType::is_executable`] rather than hard-coded, so
/// the message can never drift from the decode table again (core-gguf-19: the
/// old message advertised only `Q1_0_g128=41, TQ2_0_g128=42, TQ2_0=35` while
/// the loader also executed F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/FP8).
pub fn executable_quant_type_list() -> String {
    let mut names: Vec<String> = GgufTensorType::ALL
        .iter()
        .filter(|ty| ty.is_executable())
        .map(|ty| format!("{}={}", ty.name(), ty.wire_id()))
        .collect();
    names.sort();
    names.join(", ")
}

/// Message body for [`BonsaiError::UnsupportedQuantType`].
fn unsupported_quant_type_message(type_id: u32) -> String {
    format!(
        "unsupported quantization type id {type_id}: executable types are {}",
        executable_quant_type_list()
    )
}

/// Errors that can occur during GGUF parsing, tensor loading, and configuration.
#[derive(Error, Debug)]
pub enum BonsaiError {
    /// Invalid GGUF magic number in file header.
    #[error("invalid GGUF magic number: expected 0x46554747, got 0x{magic:08X}")]
    InvalidMagic { magic: u32 },

    /// Unsupported GGUF format version.
    #[error("unsupported GGUF version: {version} (supported: 2, 3)")]
    UnsupportedVersion { version: u32 },

    /// Invalid or missing metadata entry.
    #[error("invalid metadata for key '{key}': {reason}")]
    InvalidMetadata { key: String, reason: String },

    /// A required tensor was not found in the model file.
    #[error("tensor not found: '{name}'")]
    TensorNotFound { name: String },

    /// Unsupported quantization type encountered.
    ///
    /// The message is generated from [`GgufTensorType::is_executable`].
    #[error("{}", unsupported_quant_type_message(*type_id))]
    UnsupportedQuantType { type_id: u32 },

    /// A ggml type id this build's *parser* does not recognise at all.
    ///
    /// Distinct from [`BonsaiError::NonExecutableQuantType`], which is raised
    /// by the *loader* for a type the parser understands (block geometry is
    /// known, so headers and `info` still work) but no kernel can execute.
    #[error("unknown ggml tensor type id {type_id} (not in the ggml type table)")]
    UnknownQuantType { type_id: u32 },

    /// A quantization type the parser understands but no kernel can execute.
    #[error(
        "tensor '{name}' uses quantization type {type_name} (id {type_id}), which this build can \
         parse but not execute; executable types are {executable}"
    )]
    NonExecutableQuantType {
        type_id: u32,
        name: String,
        type_name: String,
        executable: String,
    },

    /// ggml type id 42 has three distinct on-disk layouts and the available
    /// evidence does not settle which one this file uses.
    #[error("ambiguous quantization type id {type_id}: {hint}")]
    AmbiguousQuantType { type_id: u32, hint: String },

    /// The assumed 2-bit block byte order does not match the data.
    ///
    /// Raised by the load-time structural sniff (design §1.3) so a file that
    /// mainline llama.cpp would read as gibberish fails loudly instead.
    #[error(
        "tensor '{tensor}': 2-bit block layout mismatch — assumed {assumed}, but the data says \
         {suggestion}"
    )]
    QuantLayoutMismatch {
        tensor: String,
        assumed: String,
        suggestion: String,
    },

    /// Memory mapping failed.
    #[error("memory mapping failed: {0}")]
    MmapError(#[from] std::io::Error),

    /// Unexpected end of file during parsing.
    #[error("unexpected end of file at offset {offset}")]
    UnexpectedEof { offset: u64 },

    /// Data alignment error.
    #[error("alignment error: expected {expected}-byte alignment at offset {offset}")]
    AlignmentError { expected: usize, offset: u64 },

    /// Invalid string encoding in GGUF data.
    #[error("invalid UTF-8 string at offset {offset}")]
    InvalidString { offset: u64 },

    /// A required configuration key is missing from model metadata.
    #[error("missing config key '{key}' in model metadata")]
    MissingConfigKey { key: String },

    /// The GGUF declares an architecture this build has no forward pass for.
    #[error("unsupported model architecture '{arch}'")]
    UnsupportedArchitecture { arch: String },

    /// A `prism.hadamard.*` metadata contract was violated.
    #[error("Hadamard contract violation: {reason}")]
    HadamardContract { reason: String },

    /// A tensor's declared shape/offset layout is not a valid ggml layout.
    ///
    /// Covers `ne0 % block_size != 0`, an offset that does not match the
    /// running padded sum, and an overlapping or out-of-range byte range.
    #[error("tensor '{name}' has an invalid layout: {reason}")]
    TensorLayout { name: String, reason: String },

    /// Dimension mismatch between expected and actual tensor shape.
    #[error("tensor '{name}' shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },

    /// Block size validation failed for a named quantization format.
    ///
    /// The parameterised replacement for `BonsaiError::InvalidBlockSize`
    /// (core-gguf-19): a mis-typed 34-byte/28-byte tensor now reports which
    /// format it was read as and how many bytes that format needs.
    #[error("invalid {format} block data: expected {expected} bytes, got {actual}")]
    InvalidQuantBlockSize {
        format: &'static str,
        expected: usize,
        actual: usize,
    },

    /// K-quant quantization or dequantization error.
    #[error("k-quant error: {reason}")]
    KQuantError { reason: String },
}

impl BonsaiError {
    /// Build a [`BonsaiError::NonExecutableQuantType`] for a tensor.
    pub fn non_executable_quant_type(tensor_name: impl Into<String>, ty: GgufTensorType) -> Self {
        Self::NonExecutableQuantType {
            type_id: ty.wire_id(),
            name: tensor_name.into(),
            type_name: ty.name().to_string(),
            executable: executable_quant_type_list(),
        }
    }

    /// Build a [`BonsaiError::TensorLayout`] for a tensor.
    pub fn tensor_layout(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::TensorLayout {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Return a short, stable error code string for monitoring and alerting.
    pub fn error_code(&self) -> &str {
        match self {
            Self::InvalidMagic { .. } => "INVALID_MAGIC",
            Self::UnsupportedVersion { .. } => "UNSUPPORTED_VERSION",
            Self::InvalidMetadata { .. } => "INVALID_METADATA",
            Self::TensorNotFound { .. } => "TENSOR_NOT_FOUND",
            Self::UnsupportedQuantType { .. } => "UNSUPPORTED_QUANT_TYPE",
            Self::UnknownQuantType { .. } => "UNKNOWN_QUANT_TYPE",
            Self::NonExecutableQuantType { .. } => "NON_EXECUTABLE_QUANT_TYPE",
            Self::AmbiguousQuantType { .. } => "AMBIGUOUS_QUANT_TYPE",
            Self::QuantLayoutMismatch { .. } => "QUANT_LAYOUT_MISMATCH",
            Self::MmapError(_) => "MMAP_ERROR",
            Self::UnexpectedEof { .. } => "UNEXPECTED_EOF",
            Self::AlignmentError { .. } => "ALIGNMENT_ERROR",
            Self::InvalidString { .. } => "INVALID_STRING",
            Self::MissingConfigKey { .. } => "MISSING_CONFIG_KEY",
            Self::UnsupportedArchitecture { .. } => "UNSUPPORTED_ARCHITECTURE",
            Self::HadamardContract { .. } => "HADAMARD_CONTRACT",
            Self::TensorLayout { .. } => "TENSOR_LAYOUT",
            Self::ShapeMismatch { .. } => "SHAPE_MISMATCH",
            Self::InvalidQuantBlockSize { .. } => "INVALID_QUANT_BLOCK_SIZE",
            Self::KQuantError { .. } => "K_QUANT_ERROR",
        }
    }

    /// Whether this error is potentially recoverable by retrying.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::MmapError(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_list_names_the_real_decode_set() {
        let list = executable_quant_type_list();
        for needle in [
            "F32=0",
            "F16=1",
            "BF16=30",
            "Q4_0=2",
            "Q8_0=8",
            "Q4_K=12",
            "Q5_K=13",
            "Q6_K=14",
            "Q1_0_g128=41",
            "TQ2_0_g128=42",
            "PQ2_0=142",
            "PTQ1_0=143",
        ] {
            assert!(
                list.contains(needle),
                "executable list missing {needle}: {list}"
            );
        }
        // Types the parser knows but no kernel executes must NOT be listed.
        assert!(
            !list.contains("IQ2_XXS"),
            "non-executable type leaked: {list}"
        );
        assert!(
            !list.contains("MXFP4"),
            "non-executable type leaked: {list}"
        );
    }

    #[test]
    fn unsupported_quant_type_message_is_generated_not_hardcoded() {
        let err = BonsaiError::UnsupportedQuantType { type_id: 999 };
        let msg = err.to_string();
        assert!(msg.contains("999"));
        assert!(
            msg.contains("PQ2_0=142"),
            "message must be generated from is_executable(): {msg}"
        );
    }

    #[test]
    fn non_executable_quant_type_names_tensor_and_type() {
        let err =
            BonsaiError::non_executable_quant_type("blk.0.ffn_up.weight", GgufTensorType::IQ4_XS);
        let msg = err.to_string();
        assert!(msg.contains("blk.0.ffn_up.weight"), "{msg}");
        assert!(msg.contains("IQ4_XS"), "{msg}");
        assert!(msg.contains("23"), "{msg}");
        assert_eq!(err.error_code(), "NON_EXECUTABLE_QUANT_TYPE");
    }

    #[test]
    fn parameterised_block_size_error_names_the_format() {
        let err = BonsaiError::InvalidQuantBlockSize {
            format: "PTQ1_0",
            expected: 28,
            actual: 34,
        };
        let msg = err.to_string();
        assert!(msg.contains("PTQ1_0"), "{msg}");
        assert!(msg.contains("28"), "{msg}");
        assert!(msg.contains("34"), "{msg}");
        assert_eq!(err.error_code(), "INVALID_QUANT_BLOCK_SIZE");
    }

    #[test]
    fn new_variants_all_have_distinct_error_codes() {
        let errors = [
            BonsaiError::UnknownQuantType { type_id: 7 },
            BonsaiError::AmbiguousQuantType {
                type_id: 42,
                hint: String::new(),
            },
            BonsaiError::QuantLayoutMismatch {
                tensor: String::new(),
                assumed: String::new(),
                suggestion: String::new(),
            },
            BonsaiError::UnsupportedArchitecture {
                arch: String::new(),
            },
            BonsaiError::HadamardContract {
                reason: String::new(),
            },
            BonsaiError::tensor_layout("t", "r"),
            BonsaiError::non_executable_quant_type("t", GgufTensorType::MXFP4),
            BonsaiError::InvalidQuantBlockSize {
                format: "PQ2_0",
                expected: 34,
                actual: 18,
            },
        ];
        let mut codes: Vec<&str> = errors.iter().map(|e| e.error_code()).collect();
        let total = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), total, "error codes must be distinct");
    }
}
