# oxibonsai-tokenizer

**Version:** 0.1.3 · **Status:** Stable · **Tests:** 268 passing

Pure Rust BPE tokenizer for OxiBonsai — WASM-safe, zero FFI.

Implements a byte-pair encoding (BPE) tokenizer with vocabulary management,
BPE merge rules, ChatTemplate formatting (chatml), byte-fallback encoding,
JSON serialization, a BPE trainer for building new vocabularies, and a
streaming decoder for incremental token output.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Features

- `OxiTokenizer` — encode, decode, batch encode/decode
- `Vocabulary` — bidirectional token <-> id mapping, special token support
- `BpeMerges` — merge rule table with priority lookup
- `ChatTemplate` — chatml-style prompt formatting
- `ChatTemplateRegistry` — named registry of chat prompt templates
- `HfTokenizerJson` — HuggingFace tokenizer format parser
- `StreamingDecoder` — incremental token-by-token decoding
- Byte-fallback encoding for out-of-vocabulary bytes
- `TokenizerSerializer` — `save_json` / `load_json` roundtrip
- `BpeTrainer` / `TrainerConfig` — build vocabularies from text corpora
- Benchmark suite and extended Unicode edge-case tests
- WASM-safe: no C/FFI dependencies

## Usage

```toml
[dependencies]
oxibonsai-tokenizer = "0.1.3"
```

```rust
use oxibonsai_tokenizer::OxiTokenizer;

let tokenizer = OxiTokenizer::load("tokenizer.json")?;
let ids = tokenizer.encode("Hello, world!")?;
let text = tokenizer.decode(&ids)?;
```

## License

Apache-2.0 — COOLJAPAN OU
