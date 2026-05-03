# oxibonsai

Pure Rust 1-bit LLM inference engine for PrismML Bonsai models — umbrella crate.

**Status:** Stable (thin re-export facade) · **Version:** 0.1.3 · **Updated:** 2026-05-03

Re-exports all OxiBonsai subcrates for convenience. Add this single dependency
to get access to the entire OxiBonsai ecosystem:

```toml
[dependencies]
oxibonsai = "0.1.3"

# Enable optional subsystems:
oxibonsai = { version = "0.1.3", features = ["full"] }
```

## Subcrates

| Crate | Description |
|-------|-------------|
| `oxibonsai-core` | GGUF loader, tensor types, configuration |
| `oxibonsai-kernels` | 1-bit compute kernels (dequant, GEMV, GEMM, SIMD) |
| `oxibonsai-model` | Qwen3 transformer family (1.7B/4B/8B), KV cache, attention |
| `oxibonsai-runtime` | Inference engine, sampling, OpenAI-compatible server |
| `oxibonsai-tokenizer` | Pure Rust BPE tokenizer (optional) |
| `oxibonsai-rag` | Retrieval-augmented generation pipeline (optional) |
| `oxibonsai-eval` | Model evaluation framework (optional) |
| `oxibonsai-serve` | Standalone OpenAI-compatible server (optional) |

## Feature Flags

| Flag | Description |
|------|-------------|
| `server` | HTTP server (axum) |
| `rag` | RAG pipeline |
| `native-tokenizer` | Pure Rust BPE tokenizer |
| `eval` | Evaluation harness |
| `simd-avx2` | AVX2+FMA SIMD kernels |
| `simd-avx512` | AVX-512 SIMD kernels |
| `simd-neon` | NEON SIMD kernels |
| `wasm` | WASM-safe build |
| `full` | All optional features |

## License

Apache-2.0 — COOLJAPAN OU
