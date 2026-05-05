//! # oxibonsai-core
//!
//! GGUF Q1\_0\_g128 format parser, tensor types, and model configuration
//! for OxiBonsai — the Pure Rust 1-bit LLM inference engine.
//!
//! This crate provides the foundational data types and parsing logic used
//! by the rest of the OxiBonsai stack:
//!
//! - **GGUF v3 binary format parsing** — header, metadata key-value store,
//!   and tensor info directory (see [`gguf`]).
//! - **Q1\_0\_g128 block type** — the 18-byte packed representation used for
//!   1-bit weights (see [`tensor::BlockQ1_0G128`]).
//! - **Memory-mapped tensor loading** — zero-copy access to weight data
//!   from disk via `memmap2`.
//! - **Model configuration** — [`config::Qwen3Config`] extracted from GGUF
//!   metadata or constructed for known Bonsai variants (8B, 4B, 1.7B).
//!
//! ## GGUF Q1\_0\_g128 Format
//!
//! Each block is 18 bytes: 2-byte FP16 scale + 16 bytes (128 sign bits).
//! Weight = bit ? +scale : -scale. Effective 1.125 bits per weight.
//!
//! ## Crate Organisation
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`config`] | `Qwen3Config` with named constructors for each variant |
//! | [`gguf`] | Low-level GGUF v3 reader (header, metadata, tensors) |
//! | [`quant_ternary`] | `BlockTQ2_0_g128`, `BlockTQ2_0`, `TernaryCode` — ternary block types |
//! | [`tensor`] | `BlockQ1_0G128` and `OneBitTensor` types |
//! | [`error`] | `BonsaiError` / `BonsaiResult` |

pub mod config;
pub mod error;
pub mod gguf;
pub mod quant_fp8;
pub mod quant_k;
pub mod quant_ternary;
pub mod tensor;

pub use config::Qwen3Config;
pub use error::{BonsaiError, BonsaiResult};
pub use gguf::compat::{
    build_compat_report, check_gguf_header, CompatError, ExtendedQuantType, GgufCompatReport,
    GgufVersion,
};
pub use gguf::header::GgufHeader;
pub use gguf::metadata::{MetadataStore, MetadataValue};
pub use gguf::model_card::keys as model_card_keys;
pub use gguf::model_card::{extract_known_fields, extract_model_card, ModelCard};
pub use gguf::streaming::{
    GgufStreamParser, GgufValue, StreamState, StreamedGguf, StreamedTensorInfo,
};
pub use gguf::tensor_info::{TensorInfo, TensorStore};
pub use gguf::types::{GgufTensorType, GgufValueType};
pub use gguf::writer::MetadataWriteValue;
pub use gguf::writer::{GgufWriter, TensorEntry, TensorType, WriteError};
pub use quant_fp8::{
    fp8_e4m3_decode, fp8_e4m3_encode, fp8_e5m2_decode, fp8_e5m2_encode, BlockFP8E4M3, BlockFP8E5M2,
    BLOCK_FP8_BYTES, FP8_E4M3_MAX, FP8_E5M2_MAX, QK_FP8,
};
pub use quant_k::{BlockQ2K, BlockQ4K};
pub use quant_ternary::{
    BlockTQ2_0, BlockTQ2_0_g128, TernaryCode, BLOCK_TQ2_0_BYTES, BLOCK_TQ2_0_G128_BYTES, QK_TQ2_0,
    QK_TQ2_0_G128,
};
pub use tensor::{BlockQ1_0G128, OneBitTensor};
