# oxibonsai-rag

[![Version](https://img.shields.io/badge/version-0.1.3-blue)](https://crates.io/crates/oxibonsai-rag)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)](https://github.com/cool-japan/oxibonsai)
[![Tests](https://img.shields.io/badge/tests-206_passing-brightgreen)](https://github.com/cool-japan/oxibonsai)

Pure Rust Retrieval-Augmented Generation (RAG) pipeline for OxiBonsai.

Self-contained RAG stack: document chunking (character, sentence, paragraph,
semantic, hierarchical, sliding window, markdown), pure Rust embedders
(identity, TF-IDF), in-memory vector store with cosine similarity, top-k
retrieval, and end-to-end prompt-building pipeline.

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Status

**Stable** — version 0.1.3, 206 tests passing (`cargo nextest run -p oxibonsai-rag`). Uplifted from Alpha in 0.1.2.

## Features

- `RagPipeline` — end-to-end index + query pipeline
- `VectorStore` — in-memory L2-normalized cosine similarity search
- `Retriever` — document indexing and top-k chunk retrieval
- `Embedder` trait — pluggable embedding backends
- `IdentityEmbedder` — hash-based embedder for testing
- `TfIdfEmbedder` — bag-of-words TF-IDF embedding
- Chunking strategies: character window, sentence, paragraph, recursive,
  sliding window, markdown, semantic (cosine boundary), hierarchical
- `ChunkerRegistry` — dynamic dispatch for pluggable chunking backends
- Zero external API calls — fully self-contained

## Usage

```toml
[dependencies]
oxibonsai-rag = "0.1.3"
```

```rust
use oxibonsai_rag::RagPipeline;

let mut pipeline = RagPipeline::default();
pipeline.index_document("Rust is a systems programming language.")?;
let prompt = pipeline.build_prompt("What is Rust?")?;
```

## License

Apache-2.0 — COOLJAPAN OU
