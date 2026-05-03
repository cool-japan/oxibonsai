# OxiBonsai
# (オキシ盆栽)

**Pure Rust Sub-2-Bit LLM Inference Engine for PrismML Bonsai Models**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org)

OxiBonsai is a zero-FFI, zero-C/C++ inference engine for PrismML's sub-2-bit Bonsai family — both the **1-bit** line (Q1\_0\_g128) and the **ternary** line (TQ2\_0\_g128). It runs on CPU (SIMD), Apple Silicon (Metal), and NVIDIA (CUDA) without depending on llama.cpp, BLAS, or any C/Fortran runtime. Built entirely on the COOLJAPAN ecosystem — SciRS2, OxiBLAS, OxiFFT — it delivers sovereign AI inference in Pure Rust.

## Status

**Version 0.1.4** — released 2026-05-03 · **3,650+ tests passing** · ~152k lines of Rust · Pure Rust

| Crate | Status | Tests |
|-------|--------|-------|
| oxibonsai-core     | Stable | 243   |
| oxibonsai-kernels  | Stable | 291   |
| oxibonsai-model    | Stable | 1,048 |
| oxibonsai-runtime  | Stable | 1,094 |
| oxibonsai-tokenizer| Stable | 268   |
| oxibonsai-rag      | Stable | 206   |
| oxibonsai-eval     | Stable | 151   |
| oxibonsai-serve    | Stable | 161   |
| oxibonsai (facade) | Stable | —     |

## Features

### Sub-2-Bit Native Inference

Two native quantization families, each with dedicated dequant / GEMV / full-forward kernels:

| Family | Encoding | Bits/weight | Block size | Example models |
|--------|----------|-------------|------------|----------------|
| **1-bit**  | Q1\_0\_g128  | 1.0   | 128 weights, FP16 group scale   | Bonsai-8B |
| **Ternary** | TQ2\_0\_g128 | ≈1.585 | 128 weights / 34 B, FP16 scale  | Ternary-Bonsai-8B / 4B / 1.7B |

- Full Qwen3 architecture: multi-layer decoder, GQA, SwiGLU, RoPE, RMSNorm
- `{-1, 0, +1}` ternary encoding: `0b00→−1, 0b01→0, 0b10→+1, 0b11→0`
- Correctness gate: at `--temperature 0 --seed 42`, CPU and Metal produce byte-identical output

### Acceleration Tiers

| Tier | Target | Width / Device | Feature Flag |
|------|--------|----------------|--------------|
| Reference | All platforms | Scalar | (default) |
| AVX2 + FMA | x86-64 | 256-bit | `simd-avx2` |
| AVX-512 | x86-64 | 512-bit | `simd-avx512` |
| NEON | AArch64 | 128-bit | `simd-neon` |
| **Metal** | Apple Silicon | GPU, fused full-forward | `metal` |
| **CUDA (native)** | NVIDIA GPU | GPU, NVRTC kernels | `native-cuda` |
| **CUDA (scirs2)** | NVIDIA GPU | GPU via scirs2-core | `cuda` |

Auto-detection via `KernelDispatcher::auto_detect()` selects the best CPU tier at runtime. GPU backends are opt-in at build time.

### Fused GPU Full-Forward Path

Both the 1-bit and ternary forward passes are encoded into a **single GPU command buffer** rather than one submission per GEMV. Per-layer dispatch sequence:

1. Pre-attn RMSNorm
2. Fused QKV GEMV (Q ‖ K ‖ V concatenated in weight SoA)
3. Fused QK-norm + RoPE
4. Fused KV-store
5. Batched attention: scores V2 → softmax → weighted-sum
6. Attn output GEMV + residual add
7. FFN RMSNorm
8. Gate + Up GEMV (gate ‖ up concatenated)
9. Batched SwiGLU
10. Down GEMV + residual add

= 14 dispatches/layer × N layers per command buffer. This is what unlocks the Metal and CUDA throughput numbers below.

### Observability

- Structured logging via `tracing` with env-filter and JSON output
- Inference metrics: tokens/sec, prefill/decode latency, request counts
- Health endpoint (`/health`) with readiness checks
- Circuit breaker for overload protection
- **Per-request tracing IDs** via `RequestId` (RFC 4122 UUIDv4, no `uuid` crate dependency)
- **Per-request rate metrics** via `RequestRateTracker` — TBT p50/p95, EWMA tokens/sec, queue-wait
- **Workload aggregator** — `RequestRateAggregator` rolls per-request snapshots into `oxibonsai_request_tokens_per_second`, `oxibonsai_inter_token_latency_p50/p95_seconds`, and `oxibonsai_queue_wait_seconds` Prometheus gauges

### Runtime Controllers (0.1.4)

Two adaptive controllers shipped in 0.1.4 let the runtime self-tune as the workload changes:

```rust
use oxibonsai_runtime::{KvCachePolicy, AdaptiveLookahead, AdaptiveLookaheadConfig};

// KV cache policy: FP16 ↔ Q8 ↔ Q4 driven by EWMA pressure with hysteresis.
let kv = KvCachePolicy::default();
let level = kv.observe(0.92);  // → escalates to Q8 once smoothed pressure crosses 0.80

// Speculative-decoding draft length: continuously updated from acceptance EWMA.
let mut k = AdaptiveLookahead::new(AdaptiveLookaheadConfig::default());
k.observe_step(5, 4);  // proposed=5, accepted=4 → k drifts toward 5
```

A worked end-to-end example lives in `examples/runtime_controllers.rs`:

```bash
cargo run --example runtime_controllers
```

### OpenAI-Compatible API

- `/v1/chat/completions` endpoint (POST)
- Streaming SSE support for real-time token output
- `/v1/models` endpoint
- CORS and tower middleware

### Builder Pattern API

```rust
use oxibonsai_runtime::{EngineBuilder, SamplingPreset};

let engine = EngineBuilder::new()
    .model_path("models/Ternary-Bonsai-1.7B.gguf")
    .preset(SamplingPreset::Balanced)
    .max_seq_len(4096)
    .build()?;
```

### Sampling Presets

| Preset | Temperature | Top-K | Top-P | Use Case |
|--------|-------------|-------|-------|----------|
| Greedy | 0.0 | 1 | 1.0 | Deterministic |
| Balanced | 0.7 | 40 | 0.9 | General |
| Creative | 1.0 | 100 | 0.95 | Creative writing |
| Code | 0.2 | 10 | 0.8 | Code generation |

## Bonsai Model Family

OxiBonsai supports PrismML's full Bonsai lineup across both quantization families:

| Model | Arch | Params | Format | Size | Context |
|-------|------|--------|--------|------|---------|
| Bonsai-8B                | Qwen3-8B  | 8.19 B | Q1\_0\_g128  | 1.15 GB | 65,536 |
| Ternary-Bonsai-8B        | Qwen3-8B  | 8.19 B | TQ2\_0\_g128 | ~1.75 GB | 65,536 |
| Ternary-Bonsai-4B        | Qwen3-4B  | ~4 B   | TQ2\_0\_g128 | ~900 MB  | 65,536 |
| Ternary-Bonsai-1.7B      | Qwen3-1.7B| ~1.7 B | TQ2\_0\_g128 | ~390 MB  | 65,536 |

Ternary weights trade roughly +600 MB (at 8B scale) for ~5 additional benchmark points over the 1-bit line. All models share the same Qwen3 architecture (GQA, SwiGLU, RoPE, RMSNorm), so the runtime, tokenizer, and server are identical across the family.

> **Note:** PrismML publishes Ternary Bonsai as unpacked safetensors. Use `scripts/download_ternary.sh` (or `oxibonsai convert --quant tq2_0_g128`) to fetch and repack as GGUF before loading. An `onnx-community` ONNX release (MatMulNBits bits=2) is also supported via `oxibonsai convert --onnx`.

## Installation

### CLI (recommended for end users)

```bash
cargo install oxibonsai-cli
```

This installs the `oxibonsai` binary. Rust 1.86+ required.

### Library (for Rust projects)

```toml
[dependencies]
oxibonsai = "0.1.3"
```

### Build from source (for development)

```bash
git clone https://github.com/cool-japan/oxibonsai
cd oxibonsai
cargo build --release
# binary at: target/release/oxibonsai
```

## Quick Start

> **If you installed via `cargo install oxibonsai-cli`**, start from Step 2.
> The `oxibonsai` binary is already on your PATH.

### Step 1 — (source builds only) Build

```bash
cargo build --release
export PATH="$PWD/target/release:$PATH"
```

### Step 2 — Get a model

Pick **one** of the two families (or grab both):

```bash
# ── Option A: 1-bit Bonsai-8B (1.16 GB pre-quantized GGUF — single curl) ─
mkdir -p models
curl -L -o models/Bonsai-8B.gguf \
  https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf

# ── Option B: Ternary Bonsai (download safetensors + convert to GGUF) ────
# Fetches unpacked safetensors from HF and runs `oxibonsai convert`
# to produce models/Ternary-Bonsai-<size>.gguf + models/tokenizer.json.
./scripts/download_ternary.sh 1.7b    # also: 4b | 8b
```

> **Ternary prerequisite:** `scripts/download_ternary.sh` uses the
> HuggingFace `hf` CLI — install with `pip install huggingface_hub`.

### Step 3 — Get the tokenizer

A tokenizer is required for all inference commands.
Option B above already downloads it automatically.
For **Option A** (or `cargo install` users):

```bash
oxibonsai tokenizer download          # saves to models/tokenizer.json
```

The tokenizer is pulled from `Qwen/Qwen3-8B` on HuggingFace (~2.7 MB).
Use `--output` to save elsewhere, `--repo` to use a different HF repo.

### Step 4 — Run inference

```bash
# 1-bit Bonsai-8B
oxibonsai run --model models/Bonsai-8B.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 512 --temperature 0.7 --top-p 0.9

# Ternary Bonsai (same CLI, different file)
oxibonsai run --model models/Ternary-Bonsai-1.7B.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 512 --temperature 0.7 --top-p 0.9

# Interactive chat, model info, server — all model-agnostic:
oxibonsai chat   --model models/Bonsai-8B.gguf
oxibonsai info   --model models/Ternary-Bonsai-1.7B.gguf
oxibonsai serve  --model models/Ternary-Bonsai-1.7B.gguf \
                 --host 127.0.0.1 --port 8080

# Convert safetensors → GGUF (HuggingFace unpacked safetensors dir)
oxibonsai convert \
  --from <unpacked-safetensors-dir> \
  --to models/my-model.gguf \
  --quant tq2_0_g128        # or q1_0_g128

# Convert ONNX → GGUF (MatMulNBits bits=2, e.g. onnx-community/Ternary-Bonsai-1.7B-ONNX)
oxibonsai convert --onnx \
  --from path/to/model.onnx \
  --to models/my-model.gguf
```

## CLI Smoke & Benchmark Scripts

Two parallel smoke tests — one per quantization family — plus a throughput benchmark and the ternary downloader.

| Script | Target model | Prerequisite | Purpose |
|--------|--------------|--------------|---------|
| `scripts/cli.sh [metal\|cuda]`                        | `models/Bonsai-8B.gguf`            | curl one-liner in [Quick Start](#quick-start)       | Build + end-to-end CLI test on **1-bit Bonsai-8B** |
| `scripts/cli_ternary.sh [metal\|cuda\|cuda-scirs]`    | `models/Ternary-Bonsai-1.7B.gguf` (default; `--model` to override) | **run `scripts/download_ternary.sh` first** | Build + end-to-end CLI test on **Ternary Bonsai** with a tok/s summary line |
| `scripts/bench_ternary.sh`                            | `models/Ternary-Bonsai-1.7B.gguf`  | `scripts/download_ternary.sh`                        | CPU vs Metal throughput benchmark (averaged over N runs) |
| `scripts/download_ternary.sh [8b\|4b\|1.7b]`          | —                                  | `pip install huggingface_hub`                        | Download Ternary Bonsai safetensors from HF and convert to GGUF |

Each CLI script:
1. Builds a `--release` binary with the requested feature flags
2. Runs inference (`oxibonsai run`)
3. Prints model info (`oxibonsai info`) and validates the GGUF (`oxibonsai validate`)
4. Reports the measured tok/s

```bash
# 1-bit flow (Bonsai-8B)
./scripts/cli.sh                 # CPU SIMD
./scripts/cli.sh metal           # Metal GPU (macOS)
./scripts/cli.sh cuda            # CUDA GPU  (Linux/Windows)

# Ternary flow — fetch + convert once, then run as many times as you like
./scripts/download_ternary.sh 1.7b
./scripts/cli_ternary.sh         # CPU SIMD
./scripts/cli_ternary.sh metal   # Metal GPU — fused TQ2 full-forward path
./scripts/cli_ternary.sh cuda    # native CUDA backend
./scripts/bench_ternary.sh       # CPU vs Metal, 3-run average + best
```

## Measured Throughput

End-to-end decode, averaged over 3 runs. "fused full-forward" = single GPU command buffer per token.

| Model | Backend | Hardware | tok/s |
|-------|---------|----------|------:|
| Ternary-Bonsai-1.7B | **Metal** (fused TQ2)  | Apple Silicon (M-series) | **~50** (best ~57) |
| Ternary-Bonsai-1.7B | **CUDA** (fused TQ2)   | NVIDIA GPU               | **~21.9** |
| Ternary-Bonsai-1.7B | CPU SIMD (NEON)        | Apple Silicon            | ~7–8 |
| Bonsai-8B           | Metal (fused Q1)       | Apple Silicon (M-series) | ~14.6 |

Numbers come from `scripts/bench_ternary.sh` / `scripts/cli_ternary.sh`. CPU baseline varies with thermal and background load; GPU numbers are the steady-state figures.

## Configuration

OxiBonsai supports TOML configuration files with `--config`:

```toml
[model]
path = "models/Ternary-Bonsai-1.7B.gguf"
max_seq_len = 4096

[sampling]
temperature = 0.7
top_k = 40
top_p = 0.9
repetition_penalty = 1.1

[server]
host = "127.0.0.1"
port = 8080

[observability]
log_level = "info"
json_logs = false
```

## Crate Structure

```
oxibonsai/
├── crates/
│   ├── oxibonsai-core/        GGUF loader, tensor types, config, error types
│   ├── oxibonsai-kernels/     Q1 + TQ2 kernels (dequant, GEMV, GEMM, SIMD tiers,
│   │                          tiled, parallel) + GPU backends:
│   │                            gpu_backend/metal_*       (Metal graph + fused
│   │                                                       full-forward, Q1 & TQ2)
│   │                            gpu_backend/cuda_*        (native NVRTC kernels)
│   │                            gpu_backend/scirs2_backend (scirs2-core CUDA/Metal)
│   ├── oxibonsai-tokenizer/   Pure Rust BPE tokenizer, vocabulary, ChatTemplate
│   ├── oxibonsai-model/       Qwen3 Transformer (GQA, SwiGLU, RoPE, RMSNorm,
│   │                          paged KV-cache, Q1 + TQ2 weight loaders)
│   ├── oxibonsai-rag/         RAG pipeline (chunking, embedders, vector store)
│   ├── oxibonsai-runtime/     Inference engine, sampling, OpenAI-compatible server,
│   │                          SSE streaming, metrics, health, circuit breaker
│   ├── oxibonsai-eval/        Evaluation harness (ROUGE, perplexity, MMLU)
│   └── oxibonsai-serve/       Standalone server binary
├── src/main.rs                CLI entry point (run, chat, serve, info, benchmark,
│                              convert, quantize, validate)
├── benches/                   Criterion kernel benchmarks
├── examples/                  Usage examples
├── tests/                     Integration + feature flag tests
└── scripts/                   Publish, CLI smoke tests, ternary benchmarks
```

## Examples

See the `examples/` directory:

- `basic_inference.rs` — Load a model and run single-shot inference
- `streaming.rs` — Server-sent event streaming
- `custom_sampling.rs` — Custom sampling parameters and presets

```bash
# 1-bit
cargo run --example basic_inference -- --model models/Bonsai-8B.gguf

# Ternary
cargo run --example basic_inference -- --model models/Ternary-Bonsai-1.7B.gguf
```

## COOLJAPAN Ecosystem

```
OxiBonsai (Pure Rust sub-2-bit LLM inference — Q1 + TQ2, CPU + Metal + CUDA)
  ├── SciRS2 v0.4.x     (tensor primitives, activation functions)
  ├── OxiBLAS v0.2.x    (GEMM/GEMV + 1-bit/ternary compute kernels)
  ├── OxiFFT v0.2.x     (optional RoPE acceleration)
  └── NumRS2 v0.3.x     (N-dimensional array backend)
```

All default-feature dependencies are Pure Rust — zero C/C++/Fortran, zero FFI. GPU backends (`metal`, `native-cuda`, `cuda`) are opt-in features that bring in vendor drivers.

## Development Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Foundation (workspace, GGUF loader, metadata) | ✅ |
| Phase 1 | 1-Bit Kernels (dequant, GEMV, GEMM) | ✅ |
| Phase 2 | Transformer Engine (Qwen3-8B forward pass) | ✅ |
| Phase 3 | Inference Runtime (KV cache, sampling, CLI) | ✅ |
| Phase 4 | Production Hardening (SIMD, parallel, tests, observability) | ✅ |
| Phase 5 | Ecosystem Integration (SSE streaming, WASM, API, Bonsai family) | ✅ |
| Phase 6 | Advanced Infrastructure (Multi-GPU, CUDA/Metal, PagedAttention) | ✅ |
| Phase 7 | Production Features (model merging, flash decoding, RAG, eval) | ✅ |
| Phase 8 | Final Polish (K-quant, streaming GGUF, kernel tuning, tests) | ✅ |
| Phase 9 | Ternary Bonsai (TQ2\_0\_g128 kernels, model variants, GGUF surface, export) | ✅ |
| Phase 10 | Ternary CPU SIMD tiers (AVX2 / AVX-512 / NEON TQ2 GEMV) | ✅ |
| Phase 11 | Metal TQ2 GEMV + per-kernel dispatch | ✅ |
| Phase 12 | Native CUDA backend (NVRTC, fused Q1 + TQ2 full-forward) | ✅ |
| Phase 13.x | Fused Metal TQ2 full-forward (single command buffer, ~13× speedup on 1.7B) | ✅ |
| Phase 13.y | Ternary LM head on GPU — closes all 7 `OutputWeight::Ternary` guard sites (4 Metal + 3 CUDA); +5 tok/s on Metal | ✅ |

## Sponsorship

OxiBonsai is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

The COOLJAPAN Ecosystem represents one of the largest Pure Rust scientific computing efforts in existence — spanning 40+ projects, 500+ crates, and millions of lines of Rust code across scientific computing, machine learning, quantum computing, geospatial analysis, legal technology, multimedia processing, and more. Every line is written and maintained by a small dedicated team committed to a C/Fortran-free future for scientific software.

If you find OxiBonsai or any COOLJAPAN project useful, please consider sponsoring to support continued development.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and expand the COOLJAPAN ecosystem (40+ projects, 500+ crates)
- Keep the entire stack 100% Pure Rust — no C/Fortran/system library dependencies
- Develop production-grade alternatives to OpenCV, FFmpeg, SciPy, NumPy, scikit-learn, PyTorch, TensorFlow, GDAL, and more
- Provide long-term support, security updates, and documentation
- Fund research into novel Rust-native algorithms and optimizations

## License

Apache License, Version 2.0

Copyright 2026 COOLJAPAN OU (Team KitaSan)
