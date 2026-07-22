//! # OxiBonsai
//!
//! **Pure Rust 1-bit LLM inference engine for PrismML Bonsai models.**
//!
//! OxiBonsai is a high-performance inference engine designed for 1-bit quantized
//! large language models in GGUF format. It provides a complete pipeline from model
//! loading through token generation, with optional RAG, tokenization, evaluation,
//! and HTTP serving capabilities.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxibonsai::core::GgufStreamParser;
//!
//! // Parse a GGUF model file via the streaming parser
//! let _parser = GgufStreamParser::new();
//! ```
//!
//! ## Crate Organization
//!
//! | Crate | Description |
//! |-------|-------------|
//! | [`oxibonsai-core`](https://crates.io/crates/oxibonsai-core) | GGUF loader, tensor types, quantization, configuration |
//! | [`oxibonsai-kernels`](https://crates.io/crates/oxibonsai-kernels) | Optimized compute kernels (SIMD, matmul, softmax) |
//! | [`oxibonsai-model`](https://crates.io/crates/oxibonsai-model) | Transformer model definitions, KV cache, attention |
//! | [`oxibonsai-runtime`](https://crates.io/crates/oxibonsai-runtime) | Inference engine, sampling, speculative decoding (`SpeculativeDecoder::generate_verified`, two real engines) |
//! | [`oxibonsai-tokenizer`](https://crates.io/crates/oxibonsai-tokenizer) | HuggingFace tokenizer integration |
//! | [`oxibonsai-rag`](https://crates.io/crates/oxibonsai-rag) | Retrieval-augmented generation pipeline |
//! | [`oxibonsai-eval`](https://crates.io/crates/oxibonsai-eval) | Model evaluation and benchmarking |
//! | [`oxibonsai-serve`](https://crates.io/crates/oxibonsai-serve) | OpenAI-compatible HTTP server |
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `server` | HTTP server support via `oxibonsai-serve` |
//! | `rag` | Retrieval-augmented generation |
//! | `native-tokenizer` | HuggingFace tokenizer support |
//! | `eval` | Model evaluation framework |
//! | `full` | Enable all optional features |
//! | `simd-avx2` | AVX2 SIMD kernels (x86_64) |
//! | `simd-avx512` | AVX-512 SIMD kernels (x86_64) |
//! | `simd-neon` | NEON SIMD kernels (AArch64) |
//! | `wasm` | WebAssembly target support |
//!
//! ## License
//!
//! Apache-2.0 — COOLJAPAN OU

/// Core GGUF loading, tensor types, quantization, and configuration.
pub use oxibonsai_core as core;

/// Optimized compute kernels (SIMD matmul, softmax, RMS norm, RoPE).
pub use oxibonsai_kernels as kernels;

/// Transformer model definitions, KV cache, paged attention.
pub use oxibonsai_model as model;

/// Inference engine, sampling strategies, speculative decoding. The production
/// speculative-decoding entry point is
/// `runtime::speculative::SpeculativeDecoder::generate_verified`, which drafts
/// against a delta-KV draft engine and verifies against a real, separate
/// target [`InferenceEngine`](oxibonsai_runtime::engine::InferenceEngine) —
/// its accepted output is token-identical to plain greedy decoding of the
/// target model.
pub use oxibonsai_runtime as runtime;

/// Retrieval-augmented generation pipeline.
#[cfg(feature = "rag")]
pub use oxibonsai_rag as rag;

/// HuggingFace tokenizer integration.
#[cfg(feature = "native-tokenizer")]
pub use oxibonsai_tokenizer as tokenizer;

/// Model evaluation and benchmarking framework.
#[cfg(feature = "eval")]
pub use oxibonsai_eval as eval;

/// OpenAI-compatible HTTP server.
#[cfg(feature = "server")]
pub use oxibonsai_serve as serve;
