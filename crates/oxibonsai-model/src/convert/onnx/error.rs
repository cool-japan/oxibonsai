//! Error types for the ONNX → GGUF conversion pipeline.

use std::path::PathBuf;

use thiserror::Error;

/// Errors raised by the MatMulNBits dequantizer.
#[derive(Debug, Error)]
pub enum DequantError {
    /// An unsupported `bits`/`block_size` combination was requested.
    #[error("unsupported MatMulNBits configuration: {0}")]
    Unsupported(String),

    /// A buffer (packed weights, scales, zero-points) had an unexpected length.
    #[error("length mismatch for {what}: expected {expected} bytes, got {got}")]
    LengthMismatch {
        /// Label of the mismatched buffer (e.g. `"packed B"`).
        what: &'static str,
        /// Required length derived from shape attributes.
        expected: usize,
        /// Actual length of the provided slice.
        got: usize,
    },

    /// A nibble in a 4-bit-packed zero-point buffer exceeded the maximum
    /// value allowed for a 2-bit code (3).
    #[error(
        "4-bit ZP nibble value {value} at index {index} exceeds 2-bit maximum (3); \
         GatherBlockQuantized 'bits=4' attribute appears inconsistent with the \
         actual packed data"
    )]
    NibbleOutOfRange {
        /// Flat nibble index (0-based) where the out-of-range value was found.
        index: usize,
        /// Observed nibble value (always > 3 when this error is raised).
        value: u8,
    },
}

/// Top-level error for the ONNX importer.
#[derive(Debug, Error)]
pub enum OnnxImportError {
    /// I/O failure while reading the `.onnx` file or its sidecar.
    #[error("I/O error for {path:?}: {source}")]
    Io {
        /// Path that was being accessed when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The `.onnx` protobuf file could not be parsed.
    #[error("failed to parse ONNX file {path:?}: {msg}")]
    Parse {
        /// Path to the ONNX file.
        path: PathBuf,
        /// Human-readable parser message (from `oxionnx-proto`).
        msg: String,
    },

    /// Required external-data metadata was missing from an initializer.
    #[error("missing external_data entry '{key}' for initializer '{tensor}'")]
    MissingExternalEntry {
        /// Tensor name in the ONNX graph.
        tensor: String,
        /// Missing key (`"location"`, `"offset"`, `"length"`).
        key: &'static str,
    },

    /// An initializer's `data_type` is not one of the supported dtypes.
    #[error("unsupported initializer dtype {dtype} for tensor '{tensor}'")]
    UnsupportedDtype {
        /// Tensor name.
        tensor: String,
        /// ONNX dtype code (1=float32, 10=float16, …).
        dtype: i32,
    },

    /// Could not locate `config.json` beside the ONNX file.
    #[error("config.json not found near {onnx_path:?}")]
    ConfigJsonMissing {
        /// Path of the ONNX file whose parent directories were searched.
        onnx_path: PathBuf,
    },

    /// JSON parse failure for `config.json`.
    #[error("failed to parse {path:?}: {source}")]
    ConfigJsonInvalid {
        /// Path to the invalid JSON file.
        path: PathBuf,
        /// Underlying serde_json error.
        #[source]
        source: serde_json::Error,
    },

    /// A MatMulNBits node referenced an input that is not an initializer.
    #[error("MatMulNBits node '{node}' input[{index}] ('{name}') is not an initializer")]
    MissingInitializer {
        /// Node name (`"<anon>"` if the ONNX `name` field was empty).
        node: String,
        /// Input index of the missing tensor.
        index: usize,
        /// Tensor name referenced by the node.
        name: String,
    },

    /// A required MatMulNBits attribute was missing or malformed.
    #[error("MatMulNBits node '{node}' missing attribute '{attr}'")]
    MissingAttribute {
        /// Node name.
        node: String,
        /// Attribute name (`"bits"`, `"block_size"`, `"N"`, `"K"`).
        attr: &'static str,
    },

    /// The ONNX graph does not expose an initializer we believe must be present
    /// (for example, `model.norm.weight`).
    #[error("expected initializer '{name}' not found in graph")]
    MissingNamedInitializer {
        /// HF-style initializer name that was searched for.
        name: String,
    },

    /// Dequantization of a MatMulNBits node failed.
    #[error("dequantization failed for node '{node}': {source}")]
    Dequant {
        /// Node name where the failure occurred.
        node: String,
        /// Underlying dequant error.
        #[source]
        source: DequantError,
    },

    /// Re-quantization to the requested output format (TQ2_0_g128 or
    /// Q1_0_g128) failed.
    #[error("quantization failed for tensor '{tensor}': {msg}")]
    Requantize {
        /// Target GGUF tensor name.
        tensor: String,
        /// Human-readable message from the underlying quantizer
        /// (`BlockTQ2_0_g128::quantize` or `quantize_q1_0_g128`).
        msg: String,
    },

    /// The underlying GGUF writer failed.
    #[error("GGUF writer error: {0}")]
    GgufWrite(String),

    /// A catch-all for miscellaneous conversion errors (e.g., ambiguous
    /// HF-name mapping).
    #[error("{0}")]
    Other(String),
}
