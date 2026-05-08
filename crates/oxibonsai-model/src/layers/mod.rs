//! Transformer layer implementations.
//!
//! Reusable building blocks for the Qwen3-8B architecture:
//! - RMSNorm
//! - RoPE (Rotary Position Embeddings)
//! - SwiGLU activation
//! - 1-bit Linear layer
//! - Grouped Query Attention

pub mod alibi;
pub mod attention;
pub mod attention_config;
pub mod attention_fused;
pub mod attention_sink;
pub mod cross_attention;
pub mod flash_decode;
pub mod linear;
pub mod linear_kquant_ext;
pub mod linear_kquant_full;
pub mod linear_standard;
pub mod mixture_of_depths;
pub mod moe_expert;
pub mod moe_router;
pub mod rms_norm;
pub mod rope;
pub mod rope_scaling;
pub mod sliding_window;
pub mod sparse_attention;
pub mod swiglu;
pub mod yarn_rope;
