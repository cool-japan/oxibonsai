# oxibonsai-core TODO

> GGUF loader, quant block types, tensor types, model config, error types
> 16 files, ~5,215 lines, 243 tests
> Version: 0.1.3 — Last updated: 2026-05-03

## Status: ✅ All Features Complete

All Phase 0–1 functionality implemented and tested, including fuzz/property tests, K-quant formats, ternary block types, and streaming GGUF reader.

## Done

- [x] GGUF v3 header parsing
- [x] Metadata key-value extraction (string, u32, u64, f32, array)
- [x] Tensor info parsing with offset calculation
- [x] `BlockQ1_0G128` type (18 bytes: f16 scale + 16×u8 bits)
- [x] mmap reader with zero-copy tensor access
- [x] `ModelConfig` from GGUF metadata (Qwen3 architecture)
- [x] Error types (`CoreError`)
- [x] **Property tests** — GGUF roundtrip parsing, tensor block alignment assertions (`tensor_property.rs`)
- [x] **Fuzz testing** — Malformed GGUF headers, truncated files, invalid tensor offsets (`fuzz_gguf.rs`, `gguf_edge_cases.rs`)
- [x] **Additional quant formats** — Q2_K, Q4_K support implemented in `quant_k.rs` with BlockQ2K/BlockQ4K structs, dequant/quantize, 21 tests
- [x] **Streaming GGUF reader** — `gguf/streaming.rs` with GgufStreamParser state machine, progressive parsing, 22 tests

## Ternary Bonsai

- [x] Ternary block types (`BlockTQ2_0_g128`, `BlockTQ2_0`, `TernaryCode`) in `quant_ternary.rs`
- [x] `GgufTensorType::TQ2_0` / `TQ2_0_g128` registered in `gguf/types.rs` and `gguf/writer.rs`
- [x] Core re-export: `pub mod quant_ternary` + public re-exports from `lib.rs`
