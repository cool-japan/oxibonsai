# oxibonsai-tokenizer TODO

> Pure Rust BPE tokenizer: encode, decode, training, serialization
> 12 src files + 5 integration test files, ~6,700 lines, 260+ tests (all passing)
> Version: 0.1.3 · Last updated: 2026-05-03

## Status: Stable — HuggingFace-Compatible BPE

Full BPE tokenizer with HuggingFace `tokenizer.json` support, chat templates for five model families, UTF-8-safe streaming decoder, training, encoding/decoding, batch operations, special token handling, and JSON serialization.

## Done

- [x] Alpha → Stable uplift for `oxibonsai-tokenizer` — all source edits complete, 5 new test files added, clippy clean (-D warnings), doctests + bench compile green
- [x] `OxiTokenizer` struct — encode, decode, batch encode/decode
- [x] BPE algorithm — `BpeMerges` table, `bpe_encode`, GPT-2 style pre-tokenization
- [x] Byte fallback tokens (`<0xHH>`) for unknown bytes
- [x] Special token handling (BOS, EOS, PAD, custom tokens)
- [x] Char-level tokenizer (`char_level_stub`) for testing without trained vocab
- [x] `BpeTrainer` — learn merges from corpus with configurable vocab size
- [x] `TrainerConfig` — merge frequency thresholds, training statistics
- [x] `Vocabulary` — bidirectional token↔ID mapping
- [x] `ChatTemplateKind` — canned templates for ChatML, Llama-3, Mistral, Gemma, Qwen
- [x] `BatchEncoder` — padding (`PaddingStrategy`) and truncation (`TruncationSide`)
- [x] `from_json(vocab_json, merges_json, config)` tokenizer loader
- [x] `HfTokenizerJson` — full HuggingFace `tokenizer.json` parser (Qwen3/Llama-3/Mistral/Gemma), both merge shapes, ByteLevel detection, 256-entry bytes↔unicode map
- [x] `OxiTokenizer::from_json_file` / `from_hf_tokenizer_json` — load HF files directly
- [x] `StreamingDecoder` — UTF-8-safe streaming decode with strict/lossy finish
- [x] `TokenizerState::save` / `load` — base64 serialization format (FORMAT_MAGIC)
- [x] WASM-safe implementation (no filesystem dependency in core)
- [x] `#[non_exhaustive]` on public config + error enums for forward compatibility
- [x] No-unwrap compliance in production code (policy)
- [x] Comprehensive tests — 130+ in-module unit tests + 130+ integration tests spread across `hf_format_tests`, `chat_template_tests`, `streaming_tests`, `unicode_edge_tests`, `property_tests` (proptest), `serialization_tests`, `trainer_tests`
