//! # OxiBonsai
//!
//! **Pure Rust sub-2-bit LLM inference engine for PrismML Bonsai models.**
//!
//! OxiBonsai is a high-performance inference engine for PrismML's sub-2-bit
//! Bonsai family in GGUF format: the 1-bit line, the ternary line, and the
//! Bonsai 2 27B hybrid model, plus the Bonsai-Image text-to-image model. It
//! provides a complete pipeline from model loading through token generation,
//! with optional RAG, tokenization, evaluation, image generation, and HTTP
//! serving capabilities.
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
//! | [`oxibonsai-tokenizer`](https://crates.io/crates/oxibonsai-tokenizer) | Pure-Rust native BPE tokenizer |
//! | [`oxibonsai-rag`](https://crates.io/crates/oxibonsai-rag) | Retrieval-augmented generation pipeline |
//! | [`oxibonsai-eval`](https://crates.io/crates/oxibonsai-eval) | Model evaluation and benchmarking |
//! | [`oxibonsai-serve`](https://crates.io/crates/oxibonsai-serve) | OpenAI-compatible HTTP server |
//! | [`oxibonsai-image`](https://crates.io/crates/oxibonsai-image) | Text-to-image (Bonsai-Image / FLUX.2 Klein DiT): GGUF weight loading, forward pass, and pipeline |
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `server` | HTTP server support via `oxibonsai-serve` |
//! | `rag` | Retrieval-augmented generation |
//! | `native-tokenizer` | Pure-Rust native BPE tokenizer (`oxibonsai-tokenizer`), no C deps |
//! | `hf-tokenizer` | HuggingFace `tokenizers`-backed tokenizer backend |
//! | `eval` | Model evaluation framework |
//! | `image` | Text-to-image (`oxibonsai-image`: Bonsai-Image / FLUX.2 Klein DiT) |
//! | `gpu` | GPU-backend plumbing shared by `metal`/`native-cuda` (rarely enabled directly) |
//! | `metal` | Metal GPU acceleration for kernels/model/runtime, and for `image` when it is also enabled |
//! | `cuda` | Compile-only CUDA stub forward (no dispatch change; see `native-cuda`) |
//! | `native-cuda` | Real CUDA GPU acceleration (`cudarc`) for kernels/model/runtime, and for `image` when it is also enabled |
//! | `full` | Enable every optional crate above (`server`, `rag`, both tokenizer backends, `eval`, `image`); GPU features stay opt-in even under `full` |
//! | `simd-avx2` | AVX2 SIMD kernels (x86_64) |
//! | `simd-avx512` | AVX-512 SIMD kernels (x86_64) |
//! | `simd-neon` | NEON SIMD kernels (AArch64) |
//! | `wasm` | WebAssembly target support |
//!
//! `metal`/`native-cuda` forward to `oxibonsai-image` with a *weak*
//! dependency feature (`oxibonsai-image?/metal`, `oxibonsai-image?/native-cuda`):
//! enabling `metal` alone never pulls the imaging crate in on its own. Combine
//! it with `image` (`--features image,metal`) to get a GPU-accelerated
//! text-to-image build.
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

/// Pure-Rust native BPE tokenizer.
#[cfg(feature = "native-tokenizer")]
pub use oxibonsai_tokenizer as tokenizer;

/// Model evaluation and benchmarking framework.
#[cfg(feature = "eval")]
pub use oxibonsai_eval as eval;

/// OpenAI-compatible HTTP server.
#[cfg(feature = "server")]
pub use oxibonsai_serve as serve;

/// Text-to-image (Bonsai-Image / FLUX.2 Klein DiT).
#[cfg(feature = "image")]
pub use oxibonsai_image as image;
