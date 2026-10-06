//! Retrieval-augmented question answering with `oxibonsai-rag`, end to end.
//!
//! The example indexes a handful of documents with the built-in TF-IDF
//! embedder (no model file, no network), retrieves the passages that answer
//! each question together with their similarity scores, and assembles the
//! grounded prompt a generation backend would receive. Every step is the
//! real library API a production pipeline uses:
//!
//! 1. fit an [`TfIdfEmbedder`] vocabulary on the corpus,
//! 2. build a [`RagPipeline`] around it and index the documents (they are
//!    chunked and embedded on the way in),
//! 3. [`RagPipeline::retriever`] -> `retrieve` for scored passages, and
//!    [`RagPipeline::build_prompt`] for the final prompt.
//!
//! ```bash
//! cargo run -p oxibonsai-rag --example rag_qa
//! cargo run -p oxibonsai-rag --example rag_qa -- "Which kernel tier runs on Apple GPUs?"
//! ```
//!
//! To answer the printed prompt with a real model, hand it to
//! `oxibonsai run --model <model.gguf> --prompt "<the prompt>"`, or serve the
//! same pipeline over HTTP with `oxibonsai serve --rag`.

use oxibonsai_rag::embedding::TfIdfEmbedder;
use oxibonsai_rag::pipeline::{RagConfig, RagPipeline};
use oxibonsai_rag::{RagError, RetrieverConfig};

/// The corpus: short, self-contained statements about the OxiBonsai stack.
const CORPUS: [&str; 6] = [
    "The Metal kernel tier runs the fused full-forward pass on Apple Silicon GPUs in a single command buffer per token.",
    "The NEON kernel tier accelerates ternary and one-bit matrix-vector products on AArch64 CPUs without any GPU.",
    "Ternary Bonsai models store every weight as minus one, zero or plus one in blocks of 128 weights with an FP16 scale.",
    "The OpenAI-compatible server exposes chat completions, text completions and embeddings over HTTP with bearer authentication.",
    "Bonsai 2 27B is a hybrid model: 48 linear-attention layers use a gated delta rule and 16 layers use full attention.",
    "The retrieval-augmented generation pipeline splits documents into chunks, embeds them and ranks chunks by cosine similarity.",
];

/// Questions asked when none is given on the command line.
const DEFAULT_QUESTIONS: [&str; 3] = [
    "Which kernel tier runs on Apple Silicon GPUs?",
    "How many layers of Bonsai 2 27B use full attention?",
    "How are chunks ranked in the retrieval pipeline?",
];

/// Characters of a passage printed in the ranked list.
const PREVIEW_CHARS: usize = 72;

fn preview(text: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= PREVIEW_CHARS {
        flat
    } else {
        let head: String = flat.chars().take(PREVIEW_CHARS).collect();
        format!("{head}...")
    }
}

fn main() -> Result<(), RagError> {
    // 1. Fit the TF-IDF vocabulary on the corpus (at most 512 terms).
    let embedder = TfIdfEmbedder::fit(&CORPUS, 512);

    // 2. Build the pipeline and index the documents.
    let config = RagConfig::default()
        .with_retriever_config(RetrieverConfig::default().with_top_k(2))
        .with_max_context_chars(1_200);
    let mut pipeline = RagPipeline::new(embedder, config);
    let chunk_counts = pipeline.index_documents(&CORPUS)?;
    let stats = pipeline.stats();
    println!(
        "indexed {} documents as {} chunks ({} chunks per document)\n",
        stats.documents_indexed,
        stats.chunks_indexed,
        chunk_counts
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("/"),
    );

    // 3. Retrieve and build a grounded prompt for each question.
    let mut questions: Vec<String> = std::env::args().skip(1).collect();
    if questions.is_empty() {
        questions = DEFAULT_QUESTIONS.iter().map(|q| (*q).to_string()).collect();
    }
    for question in &questions {
        println!("Question: {question}");
        for (rank, hit) in pipeline.retriever().retrieve(question)?.iter().enumerate() {
            println!(
                "  #{} score {:+.3}  {}",
                rank + 1,
                hit.score,
                preview(&hit.chunk.text)
            );
        }
        println!("--- prompt ---\n{}\n", pipeline.build_prompt(question)?);
    }
    Ok(())
}
