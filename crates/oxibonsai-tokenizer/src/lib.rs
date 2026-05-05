//! # oxibonsai-tokenizer
//!
//! Pure Rust BPE tokenizer for OxiBonsai — MeCrab-compatible, WASM-safe.
//!
//! This crate is a **production-ready** BPE implementation that can load
//! HuggingFace `tokenizer.json` files (Qwen3, Llama-3, Mistral, Gemma, ...)
//! directly without pulling in the `tokenizers` crate.  Features:
//!
//! - [`OxiTokenizer`] — high-level encode/decode API
//! - [`Vocabulary`] — bidirectional token ↔ ID mapping with special-token support
//! - [`BpeMerges`] — ordered BPE merge table
//! - [`bpe_encode`] / [`pretokenize`] — core BPE primitives
//! - [`byte_fallback_id`] — `<0xHH>` byte-fallback helper
//! - [`TokenizerError`] / [`TokenizerResult`] — error types
//! - [`hf_format::HfTokenizerJson`] — HuggingFace `tokenizer.json` parser
//! - [`streaming::StreamingDecoder`] — UTF-8-safe streaming decoder
//! - [`chat_templates::ChatTemplateKind`] — canned templates for ChatML,
//!   Llama-3, Mistral, Gemma and Qwen
//!
//! ## Quick start (character-level mode — no trained vocab required)
//!
//! ```rust
//! use oxibonsai_tokenizer::OxiTokenizer;
//!
//! let tok = OxiTokenizer::char_level_stub(256);
//! let ids = tok.encode("Hello!").expect("encode should succeed");
//! assert!(!ids.is_empty());
//! ```
//!
//! ## Loading from JSON vocab + merges
//!
//! ```rust
//! use oxibonsai_tokenizer::{OxiTokenizer, TokenizerConfig};
//!
//! let vocab_json = r#"{"a":10,"b":11,"ab":20,"<unk>":0,"<bos>":1,"<eos>":2,"<pad>":3}"#;
//! let merges_json = r#"[["a","b"]]"#;
//! let tok = OxiTokenizer::from_json(vocab_json, merges_json, TokenizerConfig::default())
//!     .expect("loading should succeed");
//! assert_eq!(tok.vocab_size(), 7);
//! ```
//!
//! ## Loading from a HuggingFace `tokenizer.json`
//!
//! ```no_run
//! use oxibonsai_tokenizer::OxiTokenizer;
//!
//! let tok = OxiTokenizer::from_json_file("tokenizer.json")
//!     .expect("HF tokenizer should load");
//! let ids = tok.encode("Hello!").expect("encode");
//! let text = tok.decode(&ids).expect("decode");
//! assert_eq!(text, "Hello!");
//! ```

pub mod bpe;
pub mod chat_templates;
pub mod error;
pub mod hf_format;
pub mod serialization;
pub mod streaming;
pub mod tests;
pub mod tokenizer;
pub mod trainer;
pub mod unigram;
pub mod utils;
pub mod vocab;

// Re-export the most commonly used types at the crate root.
pub use bpe::{bpe_encode, byte_fallback_id, pretokenize, BpeMerges};
pub use chat_templates::{ChatMessage, ChatTemplateKind};
pub use error::{TokenizerError, TokenizerResult};
pub use hf_format::{
    byte_to_unicode, bytes_to_unicode_map, unicode_to_byte, HfModelType, HfTokenizerJson,
};
pub use serialization::{
    base64_decode, base64_encode, SerializationError, TokenizerState, FORMAT_MAGIC,
};
pub use streaming::StreamingDecoder;
pub use tokenizer::{OxiTokenizer, TokenizerConfig};
pub use trainer::{
    BpeTrainer, MergeRule, SymbolPair, TrainedTokenizer, TrainerConfig, TrainerError, TrainingStats,
};
pub use unigram::{UnigramError, UnigramVocab};
pub use vocab::Vocabulary;
