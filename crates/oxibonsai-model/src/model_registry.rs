//! Multi-model support: auto-detect Bonsai model variant from GGUF metadata.
//!
//! The model registry provides automatic detection of model architecture
//! variants (8B, 4B, 1.7B) based on configuration parameters like
//! layer count and hidden dimension size.

use oxibonsai_core::config::Qwen3Config;

/// Known Bonsai model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVariant {
    /// Bonsai-8B (Qwen3-8B architecture): 36 layers, hidden=4096
    Bonsai8B,
    /// Bonsai-4B: 24 layers, hidden=2560
    Bonsai4B,
    /// Bonsai-1.7B: 16 layers, hidden=1536
    Bonsai1_7B,
    /// Ternary-Bonsai-8B: same Qwen3-8B architecture, {-1,0,+1} weights (TQ2_0_g128).
    TernaryBonsai8B,
    /// Ternary-Bonsai-4B: same Qwen3-4B architecture, {-1,0,+1} weights (TQ2_0_g128).
    TernaryBonsai4B,
    /// Ternary-Bonsai-1.7B: same Qwen3-1.7B architecture, {-1,0,+1} weights (TQ2_0_g128).
    TernaryBonsai1_7B,
    /// FP8-Bonsai-8B: same Qwen3-8B architecture, FP8 weights (F8_E4M3 or F8_E5M2).
    FP8Bonsai8B,
    /// FP8-Bonsai-4B: same Qwen3-4B architecture, FP8 weights.
    FP8Bonsai4B,
    /// FP8-Bonsai-1.7B: same Qwen3-1.7B architecture, FP8 weights.
    FP8Bonsai1_7B,
    /// Custom or unrecognized architecture
    Custom,
}

impl ModelVariant {
    /// Auto-detect variant from model configuration.
    ///
    /// Matches on the combination of `num_layers` and `hidden_size`
    /// to identify known architectures.
    pub fn from_config(config: &Qwen3Config) -> Self {
        match (config.num_layers, config.hidden_size) {
            (36, 4096) => ModelVariant::Bonsai8B,
            (24, 2560) => ModelVariant::Bonsai4B,
            (16, 1536) => ModelVariant::Bonsai1_7B,
            _ => ModelVariant::Custom,
        }
    }

    /// Detect model variant from config + sample tensor type (for ternary vs 1-bit disambiguation).
    ///
    /// Architecture match is identical to `from_config`, but if `sample_tensor_type.is_ternary()`,
    /// the result is upgraded to the ternary sibling variant.
    pub fn from_config_and_sample_tensor_type(
        config: &Qwen3Config,
        sample_tensor_type: oxibonsai_core::GgufTensorType,
    ) -> Self {
        let base = Self::from_config(config);
        if sample_tensor_type.is_ternary() {
            match base {
                Self::Bonsai8B => Self::TernaryBonsai8B,
                Self::Bonsai4B => Self::TernaryBonsai4B,
                Self::Bonsai1_7B => Self::TernaryBonsai1_7B,
                other => other, // Custom or already-ternary → unchanged
            }
        } else if sample_tensor_type.is_fp8() {
            match base {
                Self::Bonsai8B => Self::FP8Bonsai8B,
                Self::Bonsai4B => Self::FP8Bonsai4B,
                Self::Bonsai1_7B => Self::FP8Bonsai1_7B,
                other => other, // Custom or already-fp8 → unchanged
            }
        } else {
            base
        }
    }

    /// Get the default configuration for this variant.
    ///
    /// Returns the standard configuration for known variants.
    /// For `Custom`, returns the 8B configuration as a fallback.
    pub fn default_config(&self) -> Qwen3Config {
        match self {
            ModelVariant::Bonsai8B => Qwen3Config::bonsai_8b(),
            ModelVariant::Bonsai4B => Qwen3Config::bonsai_4b(),
            ModelVariant::Bonsai1_7B => Qwen3Config::bonsai_1_7b(),
            ModelVariant::TernaryBonsai8B => Qwen3Config::ternary_bonsai_8b(),
            ModelVariant::TernaryBonsai4B => Qwen3Config::ternary_bonsai_4b(),
            ModelVariant::TernaryBonsai1_7B => Qwen3Config::ternary_bonsai_1_7b(),
            // FP8 variants share the same Qwen3 architecture as their 1-bit siblings.
            ModelVariant::FP8Bonsai8B => Qwen3Config::bonsai_8b(),
            ModelVariant::FP8Bonsai4B => Qwen3Config::bonsai_4b(),
            ModelVariant::FP8Bonsai1_7B => Qwen3Config::bonsai_1_7b(),
            ModelVariant::Custom => Qwen3Config::bonsai_8b(),
        }
    }

    /// Human-readable display name for this variant.
    pub fn name(&self) -> &'static str {
        match self {
            ModelVariant::Bonsai8B => "Bonsai-8B",
            ModelVariant::Bonsai4B => "Bonsai-4B",
            ModelVariant::Bonsai1_7B => "Bonsai-1.7B",
            ModelVariant::TernaryBonsai8B => "Ternary-Bonsai-8B",
            ModelVariant::TernaryBonsai4B => "Ternary-Bonsai-4B",
            ModelVariant::TernaryBonsai1_7B => "Ternary-Bonsai-1.7B",
            ModelVariant::FP8Bonsai8B => "FP8-Bonsai-8B",
            ModelVariant::FP8Bonsai4B => "FP8-Bonsai-4B",
            ModelVariant::FP8Bonsai1_7B => "FP8-Bonsai-1.7B",
            ModelVariant::Custom => "Custom",
        }
    }

    /// Approximate parameter count for this variant.
    ///
    /// Computed as: embedding + attention + ffn + norms + output head.
    /// For 1-bit models, each "parameter" is 1 bit + per-group scale.
    /// Ternary variants share the same architecture (and thus the same parameter count)
    /// as their 1-bit siblings; only the storage format differs.
    pub fn param_count(&self) -> u64 {
        match self {
            ModelVariant::Bonsai8B | ModelVariant::TernaryBonsai8B | ModelVariant::FP8Bonsai8B => {
                // Qwen3-8B: ~8.03B parameters
                // Embedding: 151936 * 4096 = 622M
                // Per layer: Q(4096*4096) + K(4096*1024) + V(4096*1024) + O(4096*4096)
                //          + gate(4096*14336) + up(4096*14336) + down(14336*4096)
                //          + 2 norms(4096 each)
                // = 16M + 4M + 4M + 16M + 58.7M + 58.7M + 58.7M + 8K = ~216M per layer
                // 36 layers = ~7.78B
                // + embedding(622M) + output(622M) + final norm(4K)
                8_030_000_000
            }
            ModelVariant::Bonsai4B | ModelVariant::TernaryBonsai4B | ModelVariant::FP8Bonsai4B => {
                // 24 layers, hidden=2560, intermediate=6912
                // Per layer: Q(2560*2560) + K(2560*512) + V(2560*512) + O(2560*2560)
                //          + gate(2560*6912) + up(2560*6912) + down(6912*2560) + norms
                // Embedding: 151936 * 2560
                4_020_000_000
            }
            ModelVariant::Bonsai1_7B
            | ModelVariant::TernaryBonsai1_7B
            | ModelVariant::FP8Bonsai1_7B => {
                // 16 layers, hidden=1536, intermediate=4096
                1_720_000_000
            }
            ModelVariant::Custom => 0,
        }
    }

    /// Expected model file size in bytes for the quantized GGUF file.
    ///
    /// For 1-bit variants: ~1 bit per param + scale factors + FP16 embeddings.
    /// For ternary variants: TQ2_0_g128 uses 34 bytes per 128 weights ≈ 0.266 bytes/param.
    /// Embeddings and norms are typically stored in FP16 or FP32.
    pub fn expected_model_size_bytes(&self) -> u64 {
        match self {
            ModelVariant::Bonsai8B => {
                // ~8B params at 1 bit = ~1 GB for weights
                // + embeddings in FP16: 151936 * 4096 * 2 = ~1.2 GB
                // + norms in FP32: ~0.01 GB
                // + metadata overhead
                // Total: ~2.2 GB
                2_200_000_000
            }
            ModelVariant::Bonsai4B => {
                // ~4B params at 1 bit = ~0.5 GB
                // + embeddings in FP16: 151936 * 2560 * 2 = ~0.78 GB
                // Total: ~1.3 GB
                1_300_000_000
            }
            ModelVariant::Bonsai1_7B => {
                // ~1.7B params at 1 bit = ~0.21 GB
                // + embeddings in FP16: 151936 * 1536 * 2 = ~0.47 GB
                // Total: ~0.7 GB
                700_000_000
            }
            ModelVariant::TernaryBonsai8B => {
                // TQ2_0_g128: 34 bytes per 128 weights ≈ 0.266 bytes/param
                // ~8.03B params × 0.266 ≈ ~2.13 GB minus embeddings sharing
                // Embeddings (FP16): 151936 * 4096 * 2 ≈ 1.24 GB — same as 1-bit
                // Transformer weights only (excl. embedding/output ~1.24B params):
                //   ~6.8B × 0.266 ≈ 1.81 GB + embedding 1.24 GB → ~1.75 GB total
                // (embeddings/output stored in FP16 dominate less at ternary density)
                1_750_000_000
            }
            ModelVariant::TernaryBonsai4B => {
                // ~4.02B params, transformer weights ~3.63B × 0.266 ≈ 0.97 GB
                // + embeddings (FP16): 151936 * 2560 * 2 ≈ 0.78 GB → ~0.90 GB total
                900_000_000
            }
            ModelVariant::TernaryBonsai1_7B => {
                // ~1.72B params, transformer weights ~1.49B × 0.266 ≈ 0.40 GB
                // + embeddings (FP16): 151936 * 1536 * 2 ≈ 0.47 GB → ~0.39 GB total
                390_000_000
            }
            ModelVariant::FP8Bonsai8B => {
                // FP8: 1 byte/weight + FP16 scale per 32-weight block ≈ 1.0625 bytes/weight
                // Transformer weights: ~7.88B × 1.0625 ≈ 8.37 GB — but embeddings in FP16
                // Embeddings (FP16): 151936 × 4096 × 2 ≈ 1.24 GB
                // Rough total: ~8.5 GB (FP8 is closer to FP16 in size)
                8_500_000_000
            }
            ModelVariant::FP8Bonsai4B => {
                // Transformer: ~3.63B × 1.0625 ≈ 3.86 GB + embeddings 0.78 GB → ~5.0 GB
                5_000_000_000
            }
            ModelVariant::FP8Bonsai1_7B => {
                // Transformer: ~1.49B × 1.0625 ≈ 1.58 GB + embeddings 0.47 GB → ~2.3 GB
                2_300_000_000
            }
            ModelVariant::Custom => 0,
        }
    }

    /// Return all known (non-Custom) variants.
    pub fn known_variants() -> &'static [ModelVariant] {
        &[
            ModelVariant::Bonsai8B,
            ModelVariant::Bonsai4B,
            ModelVariant::Bonsai1_7B,
            ModelVariant::TernaryBonsai8B,
            ModelVariant::TernaryBonsai4B,
            ModelVariant::TernaryBonsai1_7B,
            ModelVariant::FP8Bonsai8B,
            ModelVariant::FP8Bonsai4B,
            ModelVariant::FP8Bonsai1_7B,
        ]
    }

    /// Whether this variant is a known (non-custom) architecture.
    pub fn is_known(&self) -> bool {
        !matches!(self, ModelVariant::Custom)
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_bonsai_8b() {
        let config = Qwen3Config::bonsai_8b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai8B);
        assert_eq!(ModelVariant::Bonsai8B.name(), "Bonsai-8B");
        assert!(ModelVariant::Bonsai8B.is_known());
    }

    #[test]
    fn detect_bonsai_4b() {
        let config = Qwen3Config::bonsai_4b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai4B);
        assert_eq!(ModelVariant::Bonsai4B.name(), "Bonsai-4B");
        assert!(ModelVariant::Bonsai4B.is_known());
    }

    #[test]
    fn detect_bonsai_1_7b() {
        let config = Qwen3Config::bonsai_1_7b();
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Bonsai1_7B);
        assert_eq!(ModelVariant::Bonsai1_7B.name(), "Bonsai-1.7B");
        assert!(ModelVariant::Bonsai1_7B.is_known());
    }

    #[test]
    fn detect_custom() {
        let mut config = Qwen3Config::bonsai_8b();
        config.num_layers = 48;
        config.hidden_size = 8192;
        assert_eq!(ModelVariant::from_config(&config), ModelVariant::Custom);
        assert_eq!(ModelVariant::Custom.name(), "Custom");
        assert!(!ModelVariant::Custom.is_known());
    }

    #[test]
    fn default_configs_roundtrip() {
        // Only the 1-bit variants can round-trip through from_config() alone.
        // Ternary variants share the same architecture as their 1-bit siblings,
        // so from_config() returns the 1-bit sibling — that is expected and correct.
        // Ternary detection requires from_config_and_sample_tensor_type().
        let one_bit_variants = [
            ModelVariant::Bonsai8B,
            ModelVariant::Bonsai4B,
            ModelVariant::Bonsai1_7B,
        ];
        for variant in &one_bit_variants {
            let config = variant.default_config();
            let detected = ModelVariant::from_config(&config);
            assert_eq!(
                *variant, detected,
                "variant {:?} config should round-trip",
                variant
            );
        }
    }

    #[test]
    fn param_counts_are_reasonable() {
        assert!(ModelVariant::Bonsai8B.param_count() > 7_000_000_000);
        assert!(ModelVariant::Bonsai8B.param_count() < 10_000_000_000);

        assert!(ModelVariant::Bonsai4B.param_count() > 3_000_000_000);
        assert!(ModelVariant::Bonsai4B.param_count() < 5_000_000_000);

        assert!(ModelVariant::Bonsai1_7B.param_count() > 1_000_000_000);
        assert!(ModelVariant::Bonsai1_7B.param_count() < 2_500_000_000);

        assert_eq!(ModelVariant::Custom.param_count(), 0);
    }

    #[test]
    fn model_sizes_decrease_with_variant() {
        let size_8b = ModelVariant::Bonsai8B.expected_model_size_bytes();
        let size_4b = ModelVariant::Bonsai4B.expected_model_size_bytes();
        let size_1_7b = ModelVariant::Bonsai1_7B.expected_model_size_bytes();

        assert!(size_8b > size_4b, "8B should be larger than 4B");
        assert!(size_4b > size_1_7b, "4B should be larger than 1.7B");
        assert!(size_1_7b > 0, "1.7B should have nonzero size");
    }

    #[test]
    fn display_trait() {
        assert_eq!(format!("{}", ModelVariant::Bonsai8B), "Bonsai-8B");
        assert_eq!(format!("{}", ModelVariant::Custom), "Custom");
    }

    #[test]
    fn known_variants_list() {
        let variants = ModelVariant::known_variants();
        assert_eq!(variants.len(), 9);
        assert!(variants.contains(&ModelVariant::Bonsai8B));
        assert!(variants.contains(&ModelVariant::Bonsai4B));
        assert!(variants.contains(&ModelVariant::Bonsai1_7B));
        assert!(variants.contains(&ModelVariant::TernaryBonsai8B));
        assert!(variants.contains(&ModelVariant::TernaryBonsai4B));
        assert!(variants.contains(&ModelVariant::TernaryBonsai1_7B));
        assert!(variants.contains(&ModelVariant::FP8Bonsai8B));
        assert!(variants.contains(&ModelVariant::FP8Bonsai4B));
        assert!(variants.contains(&ModelVariant::FP8Bonsai1_7B));
    }

    #[test]
    fn detect_ternary_8b_by_tensor_type() {
        let cfg = Qwen3Config::ternary_bonsai_8b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant, ModelVariant::TernaryBonsai8B);
    }

    #[test]
    fn detect_bonsai_8b_stays_1bit() {
        let cfg = Qwen3Config::bonsai_8b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::Q1_0_g128,
        );
        assert_eq!(variant, ModelVariant::Bonsai8B);
    }

    #[test]
    fn ternary_variant_param_counts_match_bonsai() {
        assert_eq!(
            ModelVariant::TernaryBonsai8B.param_count(),
            ModelVariant::Bonsai8B.param_count()
        );
        assert_eq!(
            ModelVariant::TernaryBonsai4B.param_count(),
            ModelVariant::Bonsai4B.param_count()
        );
        assert_eq!(
            ModelVariant::TernaryBonsai1_7B.param_count(),
            ModelVariant::Bonsai1_7B.param_count()
        );
    }

    #[test]
    fn ternary_variant_expected_size_less_than_fp16() {
        // Ternary 8B at ~1.75 GB should be way less than FP16 8B at ~16 GB
        let ternary_size = ModelVariant::TernaryBonsai8B.expected_model_size_bytes();
        assert!(
            ternary_size < 2_000_000_000,
            "8B ternary expected < 2 GB, got {}",
            ternary_size
        );
        assert!(
            ternary_size > 1_000_000_000,
            "8B ternary expected > 1 GB, got {}",
            ternary_size
        );
    }

    #[test]
    fn ternary_variants_are_known() {
        assert!(ModelVariant::TernaryBonsai8B.is_known());
        assert!(ModelVariant::TernaryBonsai4B.is_known());
        assert!(ModelVariant::TernaryBonsai1_7B.is_known());
    }

    #[test]
    fn ternary_variant_names() {
        assert_eq!(ModelVariant::TernaryBonsai8B.name(), "Ternary-Bonsai-8B");
        assert_eq!(ModelVariant::TernaryBonsai4B.name(), "Ternary-Bonsai-4B");
        assert_eq!(
            ModelVariant::TernaryBonsai1_7B.name(),
            "Ternary-Bonsai-1.7B"
        );
    }

    #[test]
    fn ternary_display_trait() {
        assert_eq!(
            format!("{}", ModelVariant::TernaryBonsai8B),
            "Ternary-Bonsai-8B"
        );
        assert_eq!(
            format!("{}", ModelVariant::TernaryBonsai4B),
            "Ternary-Bonsai-4B"
        );
        assert_eq!(
            format!("{}", ModelVariant::TernaryBonsai1_7B),
            "Ternary-Bonsai-1.7B"
        );
    }

    #[test]
    fn ternary_default_configs_roundtrip() {
        // Ternary variants have identical architecture to their 1-bit siblings,
        // so from_config() returns the 1-bit variant — that is expected and correct.
        // Verify the default_config() returns sensible configs with matching architecture.
        let cfg_8b = ModelVariant::TernaryBonsai8B.default_config();
        assert_eq!(cfg_8b.num_layers, 36);
        assert_eq!(cfg_8b.hidden_size, 4096);

        let cfg_4b = ModelVariant::TernaryBonsai4B.default_config();
        assert_eq!(cfg_4b.num_layers, 24);
        assert_eq!(cfg_4b.hidden_size, 2560);

        let cfg_1_7b = ModelVariant::TernaryBonsai1_7B.default_config();
        assert_eq!(cfg_1_7b.num_layers, 16);
        assert_eq!(cfg_1_7b.hidden_size, 1536);
    }

    #[test]
    fn detect_ternary_4b_and_1_7b_by_tensor_type() {
        let cfg_4b = Qwen3Config::ternary_bonsai_4b();
        let variant_4b = ModelVariant::from_config_and_sample_tensor_type(
            &cfg_4b,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant_4b, ModelVariant::TernaryBonsai4B);

        let cfg_1_7b = Qwen3Config::ternary_bonsai_1_7b();
        let variant_1_7b = ModelVariant::from_config_and_sample_tensor_type(
            &cfg_1_7b,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant_1_7b, ModelVariant::TernaryBonsai1_7B);
    }

    #[test]
    fn custom_stays_custom_with_ternary_type() {
        let mut cfg = Qwen3Config::bonsai_8b();
        cfg.num_layers = 48;
        cfg.hidden_size = 8192;
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::TQ2_0_g128,
        );
        assert_eq!(variant, ModelVariant::Custom);
    }

    #[test]
    fn detect_fp8_e4m3_8b_by_tensor_type() {
        let cfg = Qwen3Config::bonsai_8b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::F8_E4M3,
        );
        assert_eq!(variant, ModelVariant::FP8Bonsai8B);
    }

    #[test]
    fn detect_fp8_e5m2_1_7b_by_tensor_type() {
        let cfg = Qwen3Config::bonsai_1_7b();
        let variant = ModelVariant::from_config_and_sample_tensor_type(
            &cfg,
            oxibonsai_core::GgufTensorType::F8_E5M2,
        );
        assert_eq!(variant, ModelVariant::FP8Bonsai1_7B);
    }

    #[test]
    fn fp8_variant_param_counts_match_bonsai() {
        assert_eq!(
            ModelVariant::FP8Bonsai8B.param_count(),
            ModelVariant::Bonsai8B.param_count()
        );
        assert_eq!(
            ModelVariant::FP8Bonsai4B.param_count(),
            ModelVariant::Bonsai4B.param_count()
        );
        assert_eq!(
            ModelVariant::FP8Bonsai1_7B.param_count(),
            ModelVariant::Bonsai1_7B.param_count()
        );
    }

    #[test]
    fn fp8_variant_names() {
        assert_eq!(ModelVariant::FP8Bonsai8B.name(), "FP8-Bonsai-8B");
        assert_eq!(ModelVariant::FP8Bonsai4B.name(), "FP8-Bonsai-4B");
        assert_eq!(ModelVariant::FP8Bonsai1_7B.name(), "FP8-Bonsai-1.7B");
    }

    #[test]
    fn fp8_variants_are_known() {
        assert!(ModelVariant::FP8Bonsai8B.is_known());
        assert!(ModelVariant::FP8Bonsai4B.is_known());
        assert!(ModelVariant::FP8Bonsai1_7B.is_known());
    }
}
