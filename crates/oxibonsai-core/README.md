# oxibonsai-core

[![Version](https://img.shields.io/badge/version-0.1.3-blue.svg)](https://crates.io/crates/oxibonsai-core)
[![Tests](https://img.shields.io/badge/tests-243%20passing-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()

GGUF parser, quant block types, tensor types, and model configuration for OxiBonsai.

Provides the foundational data types and I/O layer: GGUF file loading (v1/v2/v3),
Q1_0_g128 / TQ2_0_g128 (ternary) / Q2_K / Q4_K block deserialization, Qwen3 model
configuration, streaming GGUF parser, GGUF writer, model card generation, and all
shared error types.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Features

- GGUF v1/v2/v3 reader with forward-compatibility layer (`gguf::compat`)
- `GgufStreamParser` — state-machine streaming parser for network-loaded models
- `GgufWriter` — produce valid GGUF byte streams with metadata and tensors
- `Qwen3Config` — model configuration for Bonsai-8B, 4B, and 1.7B variants
- `BlockQ1_0G128` / `OneBitTensor` — Q1_0_g128 block tensor types
- `BlockTQ2_0` / `BlockTQ2_0_g128` / `TernaryCode` — ternary block types
- K-quant formats: `BlockQ2K`, `BlockQ4K`
- `ModelCard` — structured model card (author, license, tags) embedded in GGUF
- `mmap` feature for zero-copy model file access
- 243 tests passing (unit, integration, fuzz, property)

## Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `mmap` | Memory-mapped file access via `memmap2` | enabled |
| `wasm` | WASM-safe builds (no `memmap2`) | disabled |

## Usage

```toml
[dependencies]
oxibonsai-core = "0.1.3"
```

## License

Apache-2.0 — COOLJAPAN OU
