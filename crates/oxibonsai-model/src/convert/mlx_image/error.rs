//! Error types for the MLX (FLUX.2 DiT) → OxiBonsai GGUF conversion pipeline.

use std::path::PathBuf;

use thiserror::Error;

/// Errors raised while packing a single MLX-quantized linear module into
/// `BlockTQ2_0_g128` blocks.
///
/// These are the *parity guards* described in the converter design: any
/// violation means the MLX tensor does not match the validated ternary
/// assumptions, so we refuse to produce a silently-wrong GGUF file.
#[derive(Debug, Error)]
pub enum PackError {
    /// The packed-weight column count does not equal `in_features / 16`.
    #[error(
        "module '{module}': weight has {got} columns, expected in/16 = {expected} \
         (in_features = {in_features})"
    )]
    WeightColumnsMismatch {
        /// Diffusers module name.
        module: String,
        /// Observed `weight` column count.
        got: usize,
        /// Expected column count (`in_features / 16`).
        expected: usize,
        /// Logical input-feature dimension.
        in_features: usize,
    },

    /// The scales/biases column count does not equal `in_features / 128`.
    #[error(
        "module '{module}': {which} has {got} columns, expected in/128 = {expected} \
         (in_features = {in_features})"
    )]
    GroupColumnsMismatch {
        /// Diffusers module name.
        module: String,
        /// Which sub-tensor (`"scales"` or `"biases"`).
        which: &'static str,
        /// Observed column count.
        got: usize,
        /// Expected column count (`in_features / 128`).
        expected: usize,
        /// Logical input-feature dimension.
        in_features: usize,
    },

    /// A sub-tensor buffer had an unexpected element count for the stated shape.
    #[error("module '{module}': {which} has {got} elements, expected {expected}")]
    BufferLengthMismatch {
        /// Diffusers module name.
        module: String,
        /// Which sub-tensor (`"weight"`, `"scales"`, `"biases"`).
        which: &'static str,
        /// Observed element count.
        got: usize,
        /// Expected element count from `out × cols`.
        expected: usize,
    },

    /// `in_features` is not a positive multiple of 128 (the TQ2_0_g128 group size).
    #[error("module '{module}': in_features = {in_features} is not a positive multiple of 128")]
    InFeaturesNotAligned {
        /// Diffusers module name.
        module: String,
        /// Logical input-feature dimension.
        in_features: usize,
    },

    /// A 2-bit MLX code exceeded 2 (i.e. a reserved `q=3` was found), which is
    /// inconsistent with the validated ternary assumption (`q ∈ {0, 1, 2}`).
    #[error(
        "module '{module}' [row {row}, group {group}]: 2-bit code value {value} > 2 \
         (reserved q=3 found; tensor is not ternary)"
    )]
    CodeOutOfRange {
        /// Diffusers module name.
        module: String,
        /// Output-feature row index.
        row: usize,
        /// 128-element group index along the input dimension.
        group: usize,
        /// Observed code value (always > 2 when this error is raised).
        value: u8,
    },

    /// The affine bias was not exactly `-scale`, breaking the symmetric-ternary
    /// assumption (`w = scale·(q-1)`).
    #[error(
        "module '{module}' [row {row}, group {group}]: bias ({bias}) != -scale (-{scale}); \
         affine quantization is not symmetric ternary"
    )]
    AsymmetricBias {
        /// Diffusers module name.
        module: String,
        /// Output-feature row index.
        row: usize,
        /// 128-element group index along the input dimension.
        group: usize,
        /// Decoded bias value (f32).
        bias: f32,
        /// Decoded scale value (f32).
        scale: f32,
    },
}

/// Top-level error for the MLX FLUX.2 DiT importer.
#[derive(Debug, Error)]
pub enum MlxImageImportError {
    /// I/O failure while opening or memory-mapping the safetensors input.
    #[error("I/O error for {path:?}: {source}")]
    Io {
        /// Path that was being accessed when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The safetensors container could not be parsed.
    #[error("failed to parse safetensors file {path:?}: {msg}")]
    Parse {
        /// Path to the safetensors file.
        path: PathBuf,
        /// Human-readable parser message.
        msg: String,
    },

    /// A sub-tensor of a quantized module was missing or had the wrong dtype.
    #[error("module '{module}': {reason}")]
    BadModule {
        /// Diffusers module name.
        module: String,
        /// Human-readable explanation.
        reason: String,
    },

    /// Packing a quantized module into ternary blocks failed a parity guard.
    #[error("packing failed: {0}")]
    Pack(#[from] PackError),

    /// The underlying GGUF writer failed.
    #[error("GGUF writer error: {0}")]
    GgufWrite(String),

    /// An unsupported quantisation format string was requested.
    #[error("unsupported quantisation format '{0}'; only 'tq2_0_g128' is supported")]
    UnsupportedQuant(String),
}
