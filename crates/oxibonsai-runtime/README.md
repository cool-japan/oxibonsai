# oxibonsai-runtime

Inference runtime, sampling, and OpenAI-compatible server for OxiBonsai.

Ties together core, kernels, model, and tokenizer into a production-ready
inference stack with advanced sampling, SSE streaming, OpenAI API compatibility,
Prometheus metrics, circuit breaker, and comprehensive configuration.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Status

**Stable** — 1,044 tests passing (`cargo nextest run -p oxibonsai-runtime --all-features`), version 0.1.3.

## Features

- `Engine` / `InferenceEngine` — prefill + autoregressive decode loop
- `EngineBuilder` / `ConfigBuilder` / `SamplerBuilder` — ergonomic builder API
- Sampling: greedy, top-k, top-p, temperature, repetition penalty, `LcgRng`
- Sampling presets: Greedy, Balanced, Creative, Code
- Advanced samplers: Mirostat v1/v2, Locally Typical, Eta, Min-P, adaptive
- `SamplerChain` — composable sampling pipeline
- Speculative decoding with draft/verify loop
- Beam search with configurable width, length penalty, n-gram blocking
- Token healing, constrained decoding, JSON schema guidance
- Context window management and token budget tracking
- Continuous batching, prefix cache engine, semantic cache
- `InferencePipeline` — high-level generation API with stop reasons
- Streaming generation (`generate_streaming`) with SSE delivery
- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
- RAG endpoints (`/v1/rag/*`) and admin API (`/admin/*`)
- Rate limiting, circuit breaker, CORS, tower middleware
- Prometheus metrics (`/metrics`): tokens/s, latency, request counts
- Health endpoint (`/health`) with readiness probes
- Memory profiler (RSS via Mach on macOS / statm on Linux)
- Quality metrics, auto-tuner, hot reload, model cache, multi-model
- TOML configuration with layered loading (defaults → file → CLI)

## Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `server` | Axum HTTP server | ✅ enabled |
| `rag` | RAG server endpoints | disabled |
| `wasm` | WASM-safe build | disabled |
| `metal` | Metal GPU backend | disabled |
| `native-cuda` | Native CUDA backend | disabled |

## Usage

```toml
[dependencies]
oxibonsai-runtime = "0.1.3"
```

```rust
use oxibonsai_runtime::{EngineBuilder, SamplingPreset};

let engine = EngineBuilder::new()
    .model_path("models/Bonsai-8B.gguf")
    .preset(SamplingPreset::Balanced)
    .max_seq_len(4096)
    .build()?;
```

## License

Apache-2.0 — COOLJAPAN OU
