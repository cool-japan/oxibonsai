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
//! | [`quant_prism`] | `BlockPQ2_0`, `BlockPTQ1_0`, `BlockQ2_0G64` — PrismML Bonsai 2 block types |
//! | [`quant_ternary`] | `BlockTQ2_0_g128`, `BlockTQ2_0`, `TernaryCode` — ternary block types |
//! | [`tensor`] | `BlockQ1_0G128` and `OneBitTensor` types |
//! | [`error`] | `BonsaiError` / `BonsaiResult` |

pub mod bf16;
pub mod config;
pub mod config_hybrid;
pub mod error;
pub mod gguf;
pub mod hadamard_config;
pub mod quant_fp8;
pub mod quant_k;
pub mod quant_k_ext;
pub mod quant_prism;
pub mod quant_std;
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
pub use gguf::quant_resolve::{
    compute_extents, resolve_type_42, resolve_type_42_with_sample, LegacyVersionTag, OrderEvidence,
    Resolved42, SizeModel, AMBIGUOUS_TYPE_ID,
};
pub use gguf::streaming::{
    GgufStreamParser, GgufValue, StreamState, StreamedGguf, StreamedTensorInfo,
};
pub use gguf::tensor_info::{
    align_up, padded_size, row_size_bytes, TensorInfo, TensorStore, MAX_TENSOR_DIMS,
};
pub use gguf::types::{GgufTensorType, GgufValueType, TypeIdResolution};
pub use gguf::writer::MetadataWriteValue;
pub use gguf::writer::{
    GgufWriter, TensorEntry, TensorProducer, TensorSource, TensorStream, TensorType, WriteError,
};
pub use quant_fp8::{
    fp8_e4m3_decode, fp8_e4m3_encode, fp8_e5m2_decode, fp8_e5m2_encode, BlockFP8E4M3, BlockFP8E5M2,
    BLOCK_FP8_BYTES, FP8_E4M3_MAX, FP8_E5M2_MAX, QK_FP8,
};
pub use quant_k::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ8K, BLOCK_Q2_K_BYTES, BLOCK_Q3K_BYTES, BLOCK_Q4_K_BYTES,
    BLOCK_Q8K_BYTES,
};
pub use quant_k_ext::{BlockQ5K, BlockQ6K, BLOCK_Q5K_BYTES, BLOCK_Q6K_BYTES};
pub use quant_prism::{
    count_plus_two_codes, q2_0_code_to_i32, transcode_ptq1_0_to_tq2, two_bit_code, BlockPQ2_0,
    BlockPTQ1_0, BlockQ2_0G64, BLOCK_PQ2_0_BYTES, BLOCK_PTQ1_0_BYTES, BLOCK_Q2_0_G64_BYTES, POW3,
    PTQ1_0_STAGES, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64,
};
pub use quant_std::{BlockQ4_0, BlockQ8_0, BLOCK_Q4_0_BYTES, BLOCK_Q8_0_BYTES, QK_Q4_0, QK_Q8_0};
pub use quant_ternary::{
    sniff_two_bit_layout, sniff_two_bit_layout_scores, ternary_code_to_i8, BlockTQ2_0,
    BlockTQ2_0_g128, LayoutScore, TernaryCode, TwoBitLayout, BLOCK_TQ2_0_BYTES,
    BLOCK_TQ2_0_G128_BYTES, QK_TQ2_0, QK_TQ2_0_G128, SNIFF_DEFAULT_BLOCKS,
    TWO_BIT_LAYOUT_CANDIDATES,
};
pub use tensor::{BlockQ1_0G128, OneBitTensor};
