# oxibonsai TODO

> v0.2.0 — 2026-06-02
> **STABLE** — umbrella facade crate, re-exports all oxibonsai-* subcrates.

## Status

This is the top-level facade crate. It has no independent logic — it re-exports
`oxibonsai-core`, `oxibonsai-kernels`, `oxibonsai-model`, `oxibonsai-runtime`,
and optionally `oxibonsai-rag`, `oxibonsai-eval`, `oxibonsai-tokenizer`, and
`oxibonsai-serve` via feature flags.

All substantive work lives in the subcrates. See the workspace-level `TODO.md`
for the full phase history and `/crates/*/TODO.md` for per-crate status.

## Features

| Feature              | Status       | Notes                                  |
|----------------------|--------------|----------------------------------------|
| `default`            | Stable       | Core inference (no server)             |
| `server`             | Stable       | OpenAI-compatible HTTP server          |
| `rag`                | Stable       | Retrieval-augmented generation         |
| `native-tokenizer`   | Stable       | oxibonsai-tokenizer integration        |
| `eval`               | Stable       | ARC/GSM8K evaluators                   |
| `full`               | Stable       | All of the above                       |
| `wasm`               | Stable       | WASM32 target (no GPU)                 |
| `simd-avx2/avx512`   | Stable       | x86_64 SIMD tiers                      |
| `simd-neon`          | Stable       | AArch64 NEON (default-on Apple/ARM)    |

## Deferred

- [ ] Python bindings (PyO3) — pending user request
- [ ] npm/WASM package auto-publish integration
