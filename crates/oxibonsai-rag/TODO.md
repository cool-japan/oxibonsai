# oxibonsai-rag TODO

> Pure Rust RAG pipeline: document chunking, embedding, vector store, retrieval
> 9 files, ~2,450 lines, 58 tests passing

**Version:** 0.1.3
**Status:** Alpha — all core features complete
**Last updated:** 2026-05-03

Full retrieval-augmented generation pipeline implemented with multiple chunking strategies, pure Rust embedders, and in-memory vector search.

## Done

- [x] Basic chunking — `chunk_document`, `chunk_by_sentences`, `chunk_by_paragraphs`
- [x] Advanced chunking strategies — `SentenceChunker`, `RecursiveCharSplitter`, `SlidingWindowChunker`, `MarkdownChunker`
- [x] `ChunkerRegistry` — dynamic dispatch for pluggable chunking backends
- [x] `Embedder` trait with pure Rust implementations (no external API calls)
- [x] `IdentityEmbedder` — hash-based embedding for testing
- [x] `TfIdfEmbedder` — bag-of-words TF-IDF embedding
- [x] `VectorStore` — in-memory flat store with L2-normalized cosine similarity
- [x] `Retriever` — document indexing + top-k chunk retrieval
- [x] `RagPipeline` — end-to-end pipeline with `RagConfig`
- [x] Error types (`RagError`) — `EmptyDocument`, `EmptyQuery`, `NoDocumentsIndexed`, `DimensionMismatch`
- [x] Integration tests for all chunking strategies (`advanced_chunker_tests.rs`)
- [x] Alpha → Stable uplift for `oxibonsai-rag` — distance metrics, metadata filtering, JSON persistence, semantic & code chunkers, property-based tests, criterion benches
