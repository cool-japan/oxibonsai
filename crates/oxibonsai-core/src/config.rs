//! Qwen3 model configuration extracted from GGUF metadata.
//!
//! The Bonsai-8B model uses the Qwen3-8B architecture. Configuration
//! values are read from GGUF metadata keys.

use crate::error::BonsaiResult;
use crate::gguf::metadata::MetadataStore;
use crate::gguf::tensor_info::keys;

/// Qwen3-8B model configuration.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    /// Hidden size (embedding dimension). Default: 4096.
    pub hidden_size: usize,
    /// Intermediate size for SwiGLU MLP. Default: 14336.
    pub intermediate_size: usize,
    /// Number of Transformer layers. Default: 36.
    pub num_layers: usize,
    /// Number of attention heads (query). Default: 32.
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA). Default: 8.
    pub num_kv_heads: usize,
    /// Head dimension (hidden_size / num_attention_heads). Default: 128.
    pub head_dim: usize,
    /// Vocabulary size. Default: 151936.
    pub vocab_size: usize,
    /// Maximum context length. Default: 65536.
    pub max_context_length: usize,
    /// RMSNorm epsilon. Default: 1e-6.
    pub rms_norm_eps: f32,
    /// RoPE frequency base. Default: 1000000.0.
    pub rope_freq_base: f32,
    /// Architecture name from GGUF metadata.
    pub architecture: String,
    /// Model name from GGUF metadata.
    pub model_name: String,
}

impl Qwen3Config {
    /// Extract configuration from GGUF metadata.
    pub fn from_metadata(metadata: &MetadataStore) -> BonsaiResult<Self> {
        let architecture = metadata
            .get_string(keys::GENERAL_ARCHITECTURE)
            .unwrap_or("qwen3")
            .to_string();

        let model_name = metadata
            .get_string(keys::GENERAL_NAME)
            .unwrap_or("Bonsai-8B")
            .to_string();

        // Architecture-scoped keys (e.g., "qwen3.embedding_length")
        let arch_prefix = &architecture;

        let hidden_size = metadata
            .get_u32(&format!("{arch_prefix}.embedding_length"))
            .or_else(|_| metadata.get_u32(keys::LLM_EMBEDDING_LENGTH))
            .unwrap_or(4096) as usize;

        let num_layers = metadata
            .get_u32(&format!("{arch_prefix}.block_count"))
            .or_else(|_| metadata.get_u32(keys::LLM_BLOCK_COUNT))
            .unwrap_or(36) as usize;

        let num_attention_heads = metadata
            .get_u32(&format!("{arch_prefix}.attention.head_count"))
            .or_else(|_| metadata.get_u32(keys::LLM_ATTENTION_HEAD_COUNT))
            .unwrap_or(32) as usize;

        let num_kv_heads = metadata
            .get_u32(&format!("{arch_prefix}.attention.head_count_kv"))
            .or_else(|_| metadata.get_u32(keys::LLM_ATTENTION_HEAD_COUNT_KV))
            .unwrap_or(8) as usize;

        let intermediate_size = metadata
            .get_u32(&format!("{arch_prefix}.feed_forward_length"))
            .or_else(|_| metadata.get_u32(keys::LLM_FEED_FORWARD_LENGTH))
            .unwrap_or(14336) as usize;

        let vocab_size = metadata
            .get_u32(&format!("{arch_prefix}.vocab_size"))
            .or_else(|_| metadata.get_u32(keys::LLM_VOCAB_SIZE))
            .unwrap_or(151936) as usize;

        let max_context_length = metadata
            .get_u32(&format!("{arch_prefix}.context_length"))
            .or_else(|_| metadata.get_u32(keys::LLM_CONTEXT_LENGTH))
            .unwrap_or(65536) as usize;

        let rms_norm_eps = metadata
            .get_f32(&format!("{arch_prefix}.attention.layer_norm_rms_epsilon"))
            .or_else(|_| metadata.get_f32(keys::LLM_ATTENTION_LAYER_NORM_RMS_EPSILON))
            .unwrap_or(1e-6);

        let rope_freq_base = metadata
            .get_f32(&format!("{arch_prefix}.rope.freq_base"))
            .or_else(|_| metadata.get_f32(keys::LLM_ROPE_FREQ_BASE))
            .unwrap_or(1_000_000.0);

        // head_dim is read explicitly from metadata when present, since for some
        // Qwen3 sizes (notably Qwen3-4B: hidden=2560, heads=32, head_dim=128)
        // the value is not hidden_size / num_attention_heads. Older GGUFs that
        // omit the key fall back to the derivation, which still holds for the
        // 8B and 1.7B variants in this project.
        let head_dim = metadata
            .get_u32(&format!("{arch_prefix}.attention.key_length"))
            .or_else(|_| metadata.get_u32(keys::LLM_ATTENTION_KEY_LENGTH))
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        Ok(Qwen3Config {
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_context_length,
            rms_norm_eps,
            rope_freq_base,
            architecture,
            model_name,
        })
    }

    /// Create a tiny configuration suitable for fast unit tests.
    ///
    /// Uses minimal dimensions so that model forward passes complete
    /// in milliseconds instead of tens of seconds.
    pub fn tiny_test() -> Self {
        Qwen3Config {
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 151936, // must match real vocab for token IDs
            max_context_length: 512,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10_000.0,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-Tiny-Test".to_string(),
        }
    }

    /// Create a Bonsai-4B configuration.
    ///
    /// 24 layers, hidden=2560, intermediate=6912, heads=20, kv_heads=4.
    pub fn bonsai_4b() -> Self {
        Qwen3Config {
            hidden_size: 2560,
            intermediate_size: 6912,
            num_layers: 24,
            num_attention_heads: 20,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 151936,
            max_context_length: 65536,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-4B".to_string(),
        }
    }

    /// Create a Bonsai-1.7B configuration.
    ///
    /// 16 layers, hidden=1536, intermediate=4096, heads=12, kv_heads=2.
    pub fn bonsai_1_7b() -> Self {
        Qwen3Config {
            hidden_size: 1536,
            intermediate_size: 4096,
            num_layers: 16,
            num_attention_heads: 12,
            num_kv_heads: 2,
            head_dim: 128,
            vocab_size: 151936,
            max_context_length: 65536,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-1.7B".to_string(),
        }
    }

    /// Create a default Qwen3-8B / Bonsai-8B configuration.
    pub fn bonsai_8b() -> Self {
        Qwen3Config {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 36,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_context_length: 65536,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            architecture: "qwen3".to_string(),
            model_name: "Bonsai-8B".to_string(),
        }
    }

    /// Create a Ternary-Bonsai-8B configuration.
    ///
    /// Same architecture as Bonsai-8B but with ternary ({-1,0,+1}) weights.
    pub fn ternary_bonsai_8b() -> Self {
        let mut cfg = Self::bonsai_8b();
        cfg.model_name = "Ternary-Bonsai-8B".to_string();
        cfg
    }

    /// Create a Ternary-Bonsai-4B configuration.
    pub fn ternary_bonsai_4b() -> Self {
        let mut cfg = Self::bonsai_4b();
        cfg.model_name = "Ternary-Bonsai-4B".to_string();
        cfg
    }

    /// Create a Ternary-Bonsai-1.7B configuration.
    pub fn ternary_bonsai_1_7b() -> Self {
        let mut cfg = Self::bonsai_1_7b();
        cfg.model_name = "Ternary-Bonsai-1.7B".to_string();
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bonsai_8b_config() {
        let config = Qwen3Config::bonsai_8b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.num_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.max_context_length, 65536);
    }

    #[test]
    fn bonsai_4b_config() {
        let config = Qwen3Config::bonsai_4b();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.intermediate_size, 6912);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_attention_heads, 20);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn bonsai_1_7b_config() {
        let config = Qwen3Config::bonsai_1_7b();
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.intermediate_size, 4096);
        assert_eq!(config.num_layers, 16);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn from_empty_metadata_uses_defaults() {
        let metadata = MetadataStore::new();
        let config = Qwen3Config::from_metadata(&metadata)
            .expect("config from empty metadata should use defaults");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 36);
    }

    #[test]
    fn ternary_bonsai_8b_matches_spec() {
        let cfg = Qwen3Config::ternary_bonsai_8b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.max_context_length, 65536);
        assert_eq!(cfg.model_name, "Ternary-Bonsai-8B");
        assert_eq!(cfg.architecture, "qwen3");
    }

    #[test]
    fn ternary_bonsai_name_distinct() {
        assert_ne!(
            Qwen3Config::bonsai_8b().model_name,
            Qwen3Config::ternary_bonsai_8b().model_name
        );
        assert_ne!(
            Qwen3Config::bonsai_4b().model_name,
            Qwen3Config::ternary_bonsai_4b().model_name
        );
        assert_ne!(
            Qwen3Config::bonsai_1_7b().model_name,
            Qwen3Config::ternary_bonsai_1_7b().model_name
        );
    }

    #[test]
    fn ternary_bonsai_4b_matches_spec() {
        let cfg = Qwen3Config::ternary_bonsai_4b();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.model_name, "Ternary-Bonsai-4B");
    }

    #[test]
    fn ternary_bonsai_1_7b_matches_spec() {
        let cfg = Qwen3Config::ternary_bonsai_1_7b();
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_layers, 16);
        assert_eq!(cfg.model_name, "Ternary-Bonsai-1.7B");
    }
}
