# OxiBonsai
# (オキシ盆栽)

**Pure Rust Sub-2-Bit LLM Inference Engine for PrismML Bonsai Models**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org)

OxiBonsai is a zero-FFI, zero-C/C++ inference engine for PrismML's sub-2-bit Bonsai family — both the **1-bit** line (Q1\_0\_g128) and the **ternary** line (TQ2\_0\_g128). It runs on CPU (SIMD), Apple Silicon (Metal), and NVIDIA (CUDA) without depending on llama.cpp, BLAS, or any C/Fortran runtime. Built entirely on the COOLJAPAN ecosystem — SciRS2, OxiBLAS, OxiFFT — it delivers sovereign AI inference in Pure Rust.

To our knowledge, OxiBonsai is the first pure-Rust — C/C++/Fortran-free, zero-FFI — inference engine for the Bonsai 1-bit/ternary model family, and the first to bring its FLUX.2-Klein text-to-image (Bonsai-Image) to pure Rust, built entirely on the COOLJAPAN ecosystem.

## Documentation

- [CLI reference](docs/CLI.md) — every `oxibonsai` and `oxibonsai-serve` subcommand, flag, and environment variable.
- [Deployment guide](docs/DEPLOYMENT.md) — which binary to run in production, reverse-proxy/TLS setup, bearer auth, admission limits, and Prometheus scraping.
- [Image-generation guide](docs/IMAGEN.md) — end-to-end Bonsai-Image (FLUX.2-Klein) text-to-image walkthrough.
- [Known Limitations](#known-limitations) — honest current-reality notes on GPU-tier coverage, standalone (not-yet-wired) building blocks, and a mitigated CUDA/FP8 batch-prefill note (disabled-by-default, no correctness impact).

## Status

**Version 0.2.3** — 2026-07-21 · **5,158 tests passing** · ~197k lines of Rust · Pure Rust

| Crate | Status | Tests |
|-------|--------|-------|
| oxibonsai-core     | Stable | 465   |
| oxibonsai-kernels  | Stable | 508   |
| oxibonsai-model    | Stable | 1,209 |
| oxibonsai-runtime  | Stable | 1,667 |
| oxibonsai-tokenizer| Stable | 375   |
| oxibonsai-rag      | Stable | 212   |
| oxibonsai-eval     | Stable | 274   |
| oxibonsai-serve    | Stable | 178   |
| oxibonsai-image    | Stable | 86    |
| oxibonsai (facade) | Stable | 1     |
| oxibonsai-cli      | Stable | 183   |

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
| **CUDA (scirs2)** | NVIDIA GPU | CPU SIMD fallback † | `cuda` |

> † **The `cuda` (scirs2-core) tier currently runs on CPU, not GPU.** scirs2-core retired its cudarc-based CUDA backend in 0.6.x, so `is_accelerated()` always returns `false` for this tier and the dispatcher transparently falls back to CPU SIMD — output is correct, just not GPU-accelerated. Use `native-cuda` for real NVIDIA GPU acceleration. See [Known Limitations](#known-limitations).

Auto-detection via `KernelDispatcher::auto_detect()` selects the best CPU tier at runtime. GPU backends are opt-in at build time.

> **Note on CPU tiers:** The CPU tier is chosen *entirely at runtime* via `is_x86_feature_detected!` — the dispatcher picks AVX-512 only when AVX-512F+BW+VL are all present, otherwise AVX2+FMA, otherwise the scalar reference path. Each SIMD function carries a per-function `#[target_feature(...)]` attribute, so a single x86-64 binary is safe on every x86-64 CPU and automatically falls back (AVX-512 → AVX-2 → scalar) with no SIGILL. The `simd-avx2` / `simd-avx512` / `simd-neon` Feature Flags above are accepted for compatibility but do **not** gate tier selection — all tiers are always compiled in and chosen at runtime.
>
> AVX-512 has been absent from Intel *consumer* CPUs since Alder Lake (Raptor Lake, Meteor Lake, Arrow Lake and Lunar Lake have none); it mainly benefits Xeon / HEDT and AMD Zen 4+. On consumer hardware the AVX-2 tier is selected automatically.
>
> There is currently no INT8 dot-product tier (AVX-VNNI `vpdpbusd` / NEON-UDOT `vdotq_s32`): the 1-bit and ternary kernels expand weights to ±scale and accumulate in FP32 FMA. An INT8 dot-product tier — which would require quantizing activations to INT8 — is a possible future enhancement.

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

// KV cache policy: computes a target FP16 / Q8 / Q4 level from EWMA pressure
// with hysteresis. NOTE: this is advisory/telemetry only today — `observe()`
// drives the `/admin/*` JSON endpoint and a Prometheus gauge, it does not
// itself change the model's actual KV-cache storage precision (see
// "Known Limitations").
let kv = KvCachePolicy::default();
let level = kv.observe(0.92);  // → target escalates to Q8 once smoothed pressure crosses 0.80

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
oxibonsai = "0.2.3"
```

### Build from source (for development)

```bash
git clone https://github.com/cool-japan/oxibonsai
cd oxibonsai
cargo build --release
# binary at: target/release/oxibonsai
```

## Configuration (`.env`)

The CLI auto-loads a `.env` file from the current directory (or any parent), so you can
omit the model/path flags. Precedence: `--flag` > shell env var > `.env` > built-in default.

```bash
# Fetch the template from GitHub …
curl -fsSL https://raw.githubusercontent.com/cool-japan/oxibonsai/master/.env.example -o .env
# … or, in a source checkout:  cp .env.example .env

# Edit .env to point at your model files
$EDITOR .env
```

Keys:

| Key | Used by | Purpose |
|-----|---------|---------|
| `OXI_MODEL` | `run` / `chat` / `serve` / `info` | GGUF model path (omit `--model`) |
| `OXI_TOKENIZER` | `run` / `chat` / `serve` | tokenizer.json/dir (optional) |
| `OXI_DIT_GGUF` | `image` | FLUX.2 Klein ternary DiT GGUF |
| `OXI_VAE_WEIGHTS` | `image` | VAE decoder weights dir |
| `OXI_TE_4BIT` | `image` | 2.1 GB 4-bit MLX text-encoder `model.safetensors` |
| `OXI_TE_TOKENIZER_DIR` | `image` | text-encoder tokenizer dir |
| `OXI_DIT_ATTN_GPU`     | `image` / `repl` | Enable Metal/CUDA DiT flash-attention (default: on for Metal) |
| `OXI_VAE_GPU`          | `image` / `repl` | Enable Metal/CUDA VAE decode (default: on for Metal) |
| `OXI_TE_GPU`           | `image` / `repl` | Enable GPU text-encoder (experimental; default off) |

With `.env` in place, the flags become optional:

```bash
oxibonsai run   --prompt "Explain ternary quantization in one sentence."
oxibonsai image --prompt "a tiny bonsai tree in a ceramic pot" --out bonsai.png
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

> **Tip:** set `OXI_MODEL` (and optionally `OXI_TOKENIZER`) in `.env`
> (see [Configuration](#configuration-env)) to omit `--model`.

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

# Interactive image REPL — loads DiT/VAE/TE once, renders many prompts
oxibonsai repl   --seed 42 --steps 4 --width 512 --height 512

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
│                              convert, quantize, validate, image, repl)
├── src/cli/
│   ├── repl.rs                `oxibonsai repl` — resident `ImageSession` (loads
│   │                          DiT/VAE/TE once, renders many prompts); Kitty
│   │                          graphics protocol inline display (Ghostty detection)
│   └── term.rs                Terminal detection helpers (Ghostty / Kitty protocol)
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
| Phase 6 | Advanced Infrastructure (Multi-GPU, CUDA/Metal, PagedAttention) | ✅ * |
| Phase 7 | Production Features (model merging, flash decoding, RAG, eval) | ✅ |
| Phase 8 | Final Polish (K-quant, streaming GGUF, kernel tuning, tests) | ✅ |
| Phase 9 | Ternary Bonsai (TQ2\_0\_g128 kernels, model variants, GGUF surface, export) | ✅ |
| Phase 10 | Ternary CPU SIMD tiers (AVX2 / AVX-512 / NEON TQ2 GEMV) | ✅ |
| Phase 11 | Metal TQ2 GEMV + per-kernel dispatch | ✅ |
| Phase 12 | Native CUDA backend (NVRTC, fused Q1 + TQ2 full-forward) | ✅ |
| Phase 13.x | Fused Metal TQ2 full-forward (single command buffer, ~13× speedup on 1.7B) | ✅ |
| Phase 13.y | Ternary LM head on GPU — closes all 7 `OutputWeight::Ternary` guard sites (4 Metal + 3 CUDA); +5 tok/s on Metal | ✅ |

\* Phase 6 items marked complete are implemented and tested, but several are standalone building blocks not yet wired into the default inference path, or are honestly-labeled simulations rather than the literal capability their name suggests — see [Known Limitations](#known-limitations) immediately below for the itemized, current-reality breakdown.

## Known Limitations

Everything below is implemented, tested, and either self-documented in its own module doc comments or tracked in [`TODO.md`](TODO.md)'s Production-Release Audit section — nothing here is a secret, but several roadmap/README lines elsewhere in this document are easy to over-read as "fully wired into inference." This section is the honest, consolidated version.

**Acceleration & scaling**

- **`cuda` (scirs2-core) tier is a CPU fallback, not GPU acceleration.** `is_accelerated()` always returns `false` for CUDA because scirs2-core retired its cudarc-based backend in 0.6.x; the dispatcher silently falls back to CPU SIMD. Output is correct — there is no garbage-token risk — it is simply not GPU-accelerated. Use `native-cuda` for real NVIDIA GPU throughput.
- **"True multi-GPU inference" is a rayon CPU simulation of NCCL-style collectives**, not real inter-GPU communication. `multi_gpu.rs`'s own doc comment says as much ("a real GPU backend would swap in NCCL/cuBLAS calls"). `DeviceMesh`/`multi_gpu.rs` is not wired into any actual multi-device dispatch path.
- **"Distributed serving" is an in-memory routing topology, not a networked serving layer.** `distributed.rs` implements consistent-hash-ring request routing and a node registry entirely in-process; its own doc comment states "no actual TCP connections are made." There is no cross-node request forwarding anywhere in this codebase.

**GPU platform coverage**

- **Metal now has full Q4_0/Q8_0/K-quant (Q2_K–Q8_K) and FP8 batch-prefill coverage (closed 2026-07-21).** Two gaps that shipped as documented limitations through 0.2.3's first hardening pass are now resolved: (1) Metal GEMV kernels + host dispatch exist for all 8 standard/K-quant formats, and the corresponding model-layer `Linear*::forward()` implementations now try Metal first (mirroring the pre-existing CUDA short-circuit) before falling back to CPU on any error — macOS users loading Q4_0/K-quant GGUFs now get real GPU acceleration, not silent CPU-only execution. (2) Metal FP8 batch prefill is implemented as a hybrid design: the heavy linear projections (fused QKV, attn-output, fused gate/up SwiGLU, FFN down) run as batched FP8 GEMMs on the GPU, while q/k-norm, RoPE, attention, and the K/V store stay on the CPU against the same cache per-token decode reads — this avoids the split-KV-cache bug class described below *by construction* (contrast the CUDA FP8 note immediately below, which took the split-cache route and stays disabled). Both are parity-validated on Apple-Silicon M3 (GEMV ≤1e-4 rel / 5e-3 abs; FP8 prefill logit cos=1.000000 + identical greedy continuation).
- **FP8 CUDA batch prefill is intentionally disabled-by-default (KV-cache-handoff bug fixed 2026-07-20).** The FP8 batch-prefill path wrote the prompt K/V into its own private GPU KV cache, separate from the CPU cache per-token CUDA decode reads — the same bug class found and fixed for Q1/ternary and disabled-by-default for Q4_0/Q8_0/K-quant. It now carries the identical split-cache guard: FP8 CUDA batch prefill returns early so `forward_prefill` falls back to the bit-correct sequential per-token path (which populates the CPU KV cache decode reads), plus a context-length guard that turns an over-long prompt into a clean `SequenceTooLong` instead of a panic. **Practical impact: none for correctness** — FP8 E4M3/E5M2 CUDA models now decode correctly at any prompt length (verified by a CPU↔CUDA synthetic-model prefill→decode parity gate). The one cost is that FP8 CUDA prompt prefill runs the sequential per-token path rather than the fused batch GEMM; setting `OXIBONSAI_FORCE_CUDA_SPLIT_PREFILL=1` re-enables the fused (decode-incorrect) path for throughput microbenchmarks only. Note the Metal FP8 path above deliberately avoided this bug class rather than mitigating it after the fact — it never writes a GPU-private cache. Tracked in `TODO.md`.

**Building blocks implemented and tested, but not wired into the default forward path**

`BonsaiModel::forward`/`forward_prefill` run a single dense GQA-attention + dense-SwiGLU-FFN Qwen3 layer stack. The following are complete, unit-tested library primitives usable by external consumers of the crates, but none of them execute inside that default forward path today:

- **PagedAttention** (`PagedKvCache`/`BlockPool`/`BlockTable`/`KvPage`, `oxibonsai-model/src/paged_kv_cache.rs`) — the shipping engine uses the contiguous `KvCache` instead.
- **KV-cache quantization** (`QuantizedKvCache`, `Fp8KvCache`, `kv_cache_quant.rs`) — never constructed by the model or runtime. Separately, `oxibonsai-runtime`'s `KvCachePolicy` computes an adaptive FP16/Q8/Q4 *target level* from memory pressure, but only feeds a Prometheus gauge and an admin endpoint — it does not reconfigure the actual cache. Its `Fp8` tier is additionally unreachable (no threshold selects it).
- **Attention-variant layers** — attention sink / StreamingLLM, sparse attention (local/strided/BigBird/Longformer/dilated), flash decoding's alternate path, cross-attention, Mixture-of-Depths, and the MoE router/expert modules all have zero call sites in `TransformerBlock::forward` outside their own test suites.
- **Sliding-window attention** (`TransformerBlock::forward_with_sliding_window`, `block/types/forward_sw.rs`) — a complete, unit-tested ~330-line implementation (RoPE, QK-norm, windowed KV lookup, parallel/sequential attention, fused FFN) with the same integration gap as the attention variants above: `BonsaiModel::forward`/`forward_prefill` call only the dense/full-attention `block.forward(...)`, and `Qwen3Config` has no `sliding_window` field at all, so no shipped GGUF metadata can currently select it.
- **RoPE-scaling variants** — YaRN (`yarn_rope.rs`) and the linear/DynamicNTK/LLaMA-3.1/LongRoPE strategies (`rope_scaling.rs`) are fully implemented, but the production `RopeTable` only ever builds plain unscaled RoPE; no shipped native Bonsai GGUF metadata currently selects a scaling strategy.

**Chat prompt template**

- **The OpenAI-compatible server always assembles a hardcoded ChatML prompt** (`<|im_start|>role\ncontent<|im_end|>`, in `build_prompt` in `oxibonsai-runtime/src/server.rs` and `build_extended_prompt` in `oxibonsai-runtime/src/api_extensions.rs`), regardless of which model is loaded. `oxibonsai-tokenizer` separately implements a real multi-family `ChatTemplateKind` (ChatML, Llama-3, Mistral, Gemma, Qwen) with a `render`/`render_with_generation_prompt` API, but the server never calls into it, and no code path reads a per-model `chat_template` field from `tokenizer.json` to auto-select the right family. All shipped Bonsai models use Qwen3 (ChatML-compatible), so this is not a correctness bug for the models this project ships — but serving a non-ChatML fine-tune through the OpenAI-compatible API today gets the wrong prompt format with no warning.

**Speculative decoding**

- The production two-engine path is `SpeculativeDecoder::generate_verified(&mut target_engine, ...)` — it drafts against the delta-KV path and verifies against a real, separate target `InferenceEngine`, and its accepted output is token-identical to plain greedy decoding of the target model.
- The originally-shipped `generate_speculative`/`verify` API is retained only as a `#[doc(hidden)]` synthetic test harness (mock draft probability, synthesized target logits) — it is not exposed as a production entry point.

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
