//! Comprehensive tests for the oxibonsai-rag crate.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::chunker::{chunk_by_paragraphs, chunk_by_sentences, chunk_document, ChunkConfig};
    use crate::embedding::{l2_normalize, Embedder, IdentityEmbedder, TfIdfEmbedder};
    use crate::error::RagError;
    use crate::pipeline::{RagConfig, RagPipeline};
    use crate::retriever::{Retriever, RetrieverConfig};
    use crate::vector_store::{cosine_similarity, VectorStore};

    // ─────────────────────────────────────────────────────────────────────────
    // IdentityEmbedder tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_identity_embedder_produces_correct_dim() {
        for dim in [8, 16, 32, 64, 128] {
            let emb = IdentityEmbedder::new(dim).expect("valid dim");
            assert_eq!(emb.embedding_dim(), dim);
            let v = emb.embed("hello world").expect("embed should succeed");
            assert_eq!(v.len(), dim, "wrong dim for size {dim}");
        }
    }

    #[test]
    fn test_identity_embedder_output_is_unit_vector() {
        let emb = IdentityEmbedder::new(32).expect("valid dim");
        let v = emb.embed("unit test text").expect("embed should succeed");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}, expected ~1.0");
    }

    #[test]
    fn test_identity_embedder_deterministic() {
        let emb = IdentityEmbedder::new(32).expect("valid dim");
        let v1 = emb.embed("determinism check").expect("embed");
        let v2 = emb.embed("determinism check").expect("embed");
        assert_eq!(v1, v2, "identical inputs must produce identical outputs");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TfIdfEmbedder tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_tfidf_embedder_fit_and_embed() {
        let docs = [
            "the quick brown fox jumps",
            "the lazy dog rests",
            "rust programming language",
        ];
        let emb = TfIdfEmbedder::fit(&docs, 50);
        assert!(emb.vocab_size() > 0, "vocab should not be empty");
        let v = emb.embed("fox jumps").expect("embed");
        assert_eq!(v.len(), emb.embedding_dim());
    }

    #[test]
    fn test_tfidf_embedder_bow_sums_to_one() {
        let docs = ["apple banana cherry", "cherry date elderberry"];
        let emb = TfIdfEmbedder::fit(&docs, 20);
        let bow = emb.embed_bow("apple cherry cherry");
        // sum of term frequencies should be 1.0 (we divide by token count)
        let total: f32 = bow.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "TF sum = {total}");
    }

    #[test]
    fn test_tfidf_embedder_unknown_word_has_zero_weight() {
        let docs = ["cat sat mat", "bat rat hat"];
        let emb = TfIdfEmbedder::fit(&docs, 20);
        // "zzz" is not in vocab; embedding should still succeed and be all zeros
        // (or at least not panic)
        let v = emb.embed("zzz").expect("embed should succeed even for OOV");
        assert_eq!(v.len(), emb.embedding_dim());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Chunker tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_chunk_document_basic() {
        let text = "a".repeat(600);
        let config = ChunkConfig::default(); // chunk_size=512, overlap=64
        let chunks = chunk_document(&text, 0, &config);
        assert!(!chunks.is_empty(), "should produce at least one chunk");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.doc_id, 0);
            assert_eq!(chunk.chunk_idx, i);
        }
    }

    #[test]
    fn test_chunk_document_overlap() {
        let text: String = (b'a'..=b'z').cycle().take(800).map(|c| c as char).collect();
        let config = ChunkConfig {
            chunk_size: 100,
            overlap: 20,
            min_chunk_size: 10,
        };
        let chunks = chunk_document(&text, 0, &config);
        assert!(
            chunks.len() >= 2,
            "overlap config should produce multiple chunks"
        );
        // Verify that consecutive chunks share content (overlap)
        let c0_end: String = chunks[0]
            .text
            .chars()
            .rev()
            .take(20)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        let c1_start: String = chunks[1].text.chars().take(20).collect();
        assert_eq!(c0_end, c1_start, "chunks should overlap by ~20 chars");
    }

    #[test]
    fn test_chunk_document_short_text_discarded() {
        let text = "hi"; // 2 chars < min_chunk_size (32)
        let config = ChunkConfig::default();
        let chunks = chunk_document(text, 0, &config);
        assert!(chunks.is_empty(), "short text should produce no chunks");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ChunkConfig::validate() tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_chunk_config_validate_accepts_default() {
        assert!(ChunkConfig::default().validate().is_ok());
    }

    #[test]
    fn test_chunk_config_validate_rejects_zero_chunk_size() {
        let config = ChunkConfig {
            chunk_size: 0,
            overlap: 0,
            min_chunk_size: 0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_chunk_config_validate_rejects_overlap_ge_chunk_size() {
        let config = ChunkConfig {
            chunk_size: 32,
            overlap: 32,
            min_chunk_size: 8,
        };
        assert!(config.validate().is_err());
    }

    /// Regression: `min_chunk_size > chunk_size` guarantees every produced
    /// window is discarded by `chunk_document`'s min-length filter, so
    /// `validate()` must reject it rather than let ingestion silently yield
    /// zero chunks forever.
    #[test]
    fn test_chunk_config_validate_rejects_min_chunk_size_gt_chunk_size() {
        let config = ChunkConfig {
            chunk_size: 16,
            overlap: 4,
            min_chunk_size: 32,
        };
        let err = config.validate().expect_err("must reject min > chunk_size");
        assert!(
            err.contains("min_chunk_size"),
            "error message should mention min_chunk_size: {err}"
        );

        // Cross-check the invariant `validate()` is protecting against: with
        // this config every non-empty document yields zero chunks.
        let text = "x".repeat(1000);
        assert!(
            chunk_document(&text, 0, &config).is_empty(),
            "min_chunk_size > chunk_size must always discard every window"
        );
    }

    #[test]
    fn test_chunk_config_validate_accepts_min_chunk_size_eq_chunk_size() {
        let config = ChunkConfig {
            chunk_size: 32,
            overlap: 4,
            min_chunk_size: 32,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_chunk_by_sentences() {
        let text = "Hello world. This is a test. Another sentence here.";
        let chunks = chunk_by_sentences(text, 0, 2);
        assert!(!chunks.is_empty(), "should produce chunks");
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_chunk_by_sentences_single_max() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunk_by_sentences(text, 0, 1);
        // Each sentence should be its own chunk
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_chunk_by_paragraphs() {
        let text = "First paragraph.\nStill first.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_by_paragraphs(text, 0);
        assert_eq!(chunks.len(), 3, "should produce 3 paragraph chunks");
        assert!(chunks[0].text.contains("First paragraph"));
        assert!(chunks[1].text.contains("Second paragraph"));
        assert!(chunks[2].text.contains("Third paragraph"));
    }

    #[test]
    fn test_chunk_by_paragraphs_single() {
        let text = "Only one paragraph with no blank lines.";
        let chunks = chunk_by_paragraphs(text, 0);
        assert_eq!(chunks.len(), 1);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // VectorStore tests
    // ─────────────────────────────────────────────────────────────────────────

    fn make_chunk(text: &str) -> crate::chunker::Chunk {
        crate::chunker::Chunk::new(text.to_string(), 0, 0, 0)
    }

    #[test]
    fn test_vector_store_insert_and_search() {
        let mut store = VectorStore::new(4);
        let v1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0, 0.0];
        store
            .insert(v1.clone(), make_chunk("chunk one"))
            .expect("insert v1");
        store
            .insert(v2.clone(), make_chunk("chunk two"))
            .expect("insert v2");
        assert_eq!(store.len(), 2);

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // First result should be the one aligned with the query
        assert_eq!(results[0].chunk.text, "chunk one");
    }

    #[test]
    fn test_vector_store_cosine_similarity() {
        let mut store = VectorStore::new(3);
        store
            .insert(vec![1.0, 0.0, 0.0], make_chunk("x-axis"))
            .expect("insert");
        store
            .insert(vec![0.0, 1.0, 0.0], make_chunk("y-axis"))
            .expect("insert");

        // Query along x; x-axis chunk should score ~1.0, y-axis ~0.0
        let results = store.search(&[1.0, 0.0, 0.0], 2);
        assert!(results[0].score > 0.99, "score = {}", results[0].score);
        assert!(
            results[1].score.abs() < 0.01,
            "score = {}",
            results[1].score
        );
    }

    #[test]
    fn test_vector_store_search_threshold() {
        let mut store = VectorStore::new(2);
        store
            .insert(vec![1.0, 0.0], make_chunk("positive"))
            .expect("insert");
        store
            .insert(vec![-1.0, 0.0], make_chunk("negative"))
            .expect("insert");

        // Only the positive one should pass a 0.5 threshold
        let results = store.search_with_threshold(&[1.0, 0.0], 10, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "positive");
    }

    #[test]
    fn test_vector_store_memory_usage() {
        let mut store = VectorStore::new(16);
        assert_eq!(store.memory_usage_bytes(), 0);
        store
            .insert(vec![0.0f32; 16], make_chunk("hello"))
            .expect("insert");
        let mem = store.memory_usage_bytes();
        assert!(mem > 0, "memory_usage_bytes should be > 0 after insertion");
    }

    #[test]
    fn test_vector_store_dimension_mismatch() {
        let mut store = VectorStore::new(4);
        let result = store.insert(vec![1.0, 2.0], make_chunk("wrong dim"));
        assert!(matches!(
            result,
            Err(RagError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
    }

    #[test]
    fn test_vector_store_clear() {
        let mut store = VectorStore::new(2);
        store
            .insert(vec![1.0, 0.0], make_chunk("a"))
            .expect("insert");
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cosine / normalize unit tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical_is_one() {
        let mut v = vec![3.0f32, 4.0, 0.0];
        l2_normalize(&mut v);
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "sim = {sim}");
    }

    #[test]
    fn test_l2_normalize_produces_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm after normalise = {norm}");
    }

    #[test]
    fn test_l2_normalize_zero_vector_unchanged() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v);
        // Should not produce NaN
        assert!(v.iter().all(|x| !x.is_nan()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Retriever tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_retriever_add_and_retrieve() {
        let emb = IdentityEmbedder::new(32).expect("valid dim");
        let config = RetrieverConfig {
            top_k: 3,
            ..Default::default()
        };
        let mut ret = Retriever::new(emb, config);

        let chunk_cfg = ChunkConfig {
            chunk_size: 128,
            overlap: 16,
            min_chunk_size: 10,
        };
        let n = ret
            .add_document(
                "Rust is a systems programming language focused on safety and performance.",
                &chunk_cfg,
            )
            .expect("add_document");
        assert!(n > 0, "should index at least one chunk");

        let results = ret.retrieve("Rust programming").expect("retrieve");
        assert!(!results.is_empty(), "should retrieve at least one result");
    }

    #[test]
    fn test_retriever_empty_document_error() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let result = ret.add_document("   ", &ChunkConfig::default());
        assert!(matches!(result, Err(RagError::EmptyDocument)));
    }

    #[test]
    fn test_retriever_empty_query_error() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        ret.add_document(
            "some content here for testing",
            &ChunkConfig {
                chunk_size: 50,
                overlap: 5,
                min_chunk_size: 5,
            },
        )
        .expect("add");
        let result = ret.retrieve("  ");
        assert!(matches!(result, Err(RagError::EmptyQuery)));
    }

    #[test]
    fn test_retriever_no_documents_error() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let ret = Retriever::new(emb, RetrieverConfig::default());
        let result = ret.retrieve("anything");
        assert!(matches!(result, Err(RagError::NoDocumentsIndexed)));
    }

    /// Regression: a `ChunkConfig` with `min_chunk_size > chunk_size` must be
    /// rejected up front by `add_document` with an explicit error, rather
    /// than silently returning `Ok(0)` for every document forever.
    #[test]
    fn test_retriever_add_document_rejects_invalid_chunk_config() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let mut ret = Retriever::new(emb, RetrieverConfig::default());
        let bad_config = ChunkConfig {
            chunk_size: 16,
            overlap: 4,
            min_chunk_size: 32,
        };
        let result = ret.add_document("plenty of content to chunk here", &bad_config);
        assert!(
            matches!(result, Err(RagError::InvalidChunkConfig(_))),
            "expected InvalidChunkConfig, got {result:?}"
        );
        // No document should have been counted as indexed on error.
        assert_eq!(ret.document_count(), 0);
    }

    #[test]
    fn test_retriever_multiple_documents() {
        let emb = IdentityEmbedder::new(64).expect("valid dim");
        let config = RetrieverConfig {
            top_k: 5,
            ..Default::default()
        };
        let mut ret = Retriever::new(emb, config);
        let chunk_cfg = ChunkConfig {
            chunk_size: 100,
            overlap: 10,
            min_chunk_size: 10,
        };
        let docs = [
            "Rust memory safety without garbage collection.",
            "Python is widely used for data science and machine learning.",
            "Go is designed for concurrent network services.",
        ];
        let counts = ret.add_documents(&docs, &chunk_cfg).expect("add_documents");
        assert_eq!(counts.len(), 3);
        assert_eq!(ret.document_count(), 3);

        let texts = ret.retrieve_text("memory safety").expect("retrieve_text");
        assert!(!texts.is_empty());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pipeline tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_build_prompt() {
        let emb = IdentityEmbedder::new(64).expect("valid dim");
        let mut pipeline = RagPipeline::new(emb, RagConfig::default());
        pipeline
            .index_document("Rust is a safe systems language.")
            .expect("index");
        let prompt = pipeline
            .build_prompt("What is Rust?")
            .expect("build_prompt");
        assert!(prompt.contains("Question: What is Rust?"));
        assert!(prompt.contains("Answer:"));
    }

    #[test]
    fn test_pipeline_build_prompt_empty_query() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let pipeline = RagPipeline::new(emb, RagConfig::default());
        let result = pipeline.build_prompt("");
        assert!(matches!(result, Err(RagError::EmptyQuery)));
    }

    #[test]
    fn test_pipeline_retrieve_context() {
        let emb = IdentityEmbedder::new(64).expect("valid dim");
        let cfg = RagConfig {
            chunk_config: ChunkConfig {
                chunk_size: 80,
                overlap: 10,
                min_chunk_size: 10,
            },
            // Accept any cosine similarity (identity embedder may produce
            // negative scores for semantically unrelated hash vectors)
            retriever_config: RetrieverConfig {
                top_k: 5,
                min_score: -1.0,
                rerank: false,
            },
            ..RagConfig::default()
        };
        let mut pipeline = RagPipeline::new(emb, cfg);
        pipeline
            .index_document("The speed of light is approximately 299,792,458 metres per second.")
            .expect("index");
        let ctx = pipeline
            .retrieve_context("speed of light")
            .expect("context");
        assert!(!ctx.is_empty(), "context should not be empty");
    }

    #[test]
    fn test_pipeline_retrieve_context_no_docs() {
        let emb = IdentityEmbedder::new(16).expect("valid dim");
        let pipeline = RagPipeline::new(emb, RagConfig::default());
        // With no docs, build_prompt should still succeed (empty context)
        let prompt = pipeline.build_prompt("hello").expect("build_prompt");
        assert!(prompt.contains("Question: hello"));
    }

    #[test]
    fn test_pipeline_stats() {
        let emb = IdentityEmbedder::new(32).expect("valid dim");
        let mut pipeline = RagPipeline::new(emb, RagConfig::default());
        let s0 = pipeline.stats();
        assert_eq!(s0.documents_indexed, 0);
        assert_eq!(s0.chunks_indexed, 0);
        assert_eq!(s0.embedding_dim, 32);

        pipeline
            .index_document(
                "Some reasonably long document text that should generate at least one chunk \
                 when passed through the default chunker configuration settings here.",
            )
            .expect("index");
        let s1 = pipeline.stats();
        assert_eq!(s1.documents_indexed, 1);
        assert!(s1.chunks_indexed > 0);
        assert!(s1.store_memory_bytes > 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // End-to-end test
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_rag_end_to_end() {
        let corpus = [
            "The Eiffel Tower is located in Paris, France. It was constructed in 1889.",
            "The Great Wall of China stretches over 21,000 kilometres. \
             It was built during the Ming dynasty.",
            "Mount Everest is the highest mountain above sea level, \
             standing at 8,848.86 metres in the Himalayas.",
            "The Amazon River in South America is the largest river by discharge volume. \
             It flows through Brazil.",
            "The Sahara Desert is the largest hot desert on Earth, \
             covering much of North Africa.",
        ];

        let emb = IdentityEmbedder::new(128).expect("valid dim");
        let cfg = RagConfig {
            chunk_config: ChunkConfig {
                chunk_size: 200,
                overlap: 20,
                min_chunk_size: 20,
            },
            retriever_config: RetrieverConfig {
                top_k: 3,
                min_score: -1.0,
                rerank: false,
            },
            max_context_chars: 2048,
            context_separator: "\n---\n".to_string(),
            prompt_template: "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:".to_string(),
        };

        let mut pipeline = RagPipeline::new(emb, cfg);
        pipeline.index_documents(&corpus).expect("index all docs");

        let stats = pipeline.stats();
        assert_eq!(stats.documents_indexed, 5);
        assert!(stats.chunks_indexed > 0);

        let prompt = pipeline
            .build_prompt("Where is the Eiffel Tower?")
            .expect("build_prompt");
        assert!(prompt.contains("Question: Where is the Eiffel Tower?"));
        assert!(prompt.contains("Answer:"));
        assert!(prompt.contains("Context:"));
    }
}
