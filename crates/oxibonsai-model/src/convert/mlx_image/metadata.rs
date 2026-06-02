//! GGUF metadata for the FLUX.2 DiT (`bonsai-image`) architecture.
//!
//! The MLX safetensors input has no `config.json`, so (unlike the Qwen3
//! text-model path in [`crate::convert::common`]) the architecture dimensions
//! are emitted from the validated converter design constants below. All keys
//! are namespaced under `bonsai-image.*` to avoid colliding with the `llm.*`
//! keys used by the text models.

use oxibonsai_core::gguf::tensor_info::keys;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

/// FLUX.2 DiT architecture constants (from the converter design doc).
#[derive(Debug, Clone, Copy)]
pub struct DitArch {
    /// Number of dual-stream (`transformer_blocks`) layers.
    pub num_layers: u32,
    /// Number of single-stream (`single_transformer_blocks`) layers.
    pub num_single_layers: u32,
    /// Number of attention heads.
    pub num_attention_heads: u32,
    /// Per-head attention dimension.
    pub attention_head_dim: u32,
    /// Joint (text+image) attention feature width feeding `context_embedder`.
    pub joint_attention_dim: u32,
    /// Latent channel count (patch-embedding input width).
    pub in_channels: u32,
    /// Feed-forward expansion ratio.
    pub mlp_ratio: f32,
    /// Per-axis RoPE dimensions.
    pub axes_dims_rope: [u32; 4],
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether the model consumes a guidance-embedding input.
    pub guidance_embeds: bool,
}

impl Default for DitArch {
    /// The validated `bonsai-image-ternary-4B` DiT configuration.
    fn default() -> Self {
        Self {
            num_layers: 5,
            num_single_layers: 20,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 7680,
            in_channels: 128,
            mlp_ratio: 3.0,
            axes_dims_rope: [32, 32, 32, 32],
            rope_theta: 2000.0,
            guidance_embeds: false,
        }
    }
}

/// GGUF metadata key namespace for the FLUX.2 DiT.
pub mod arch_keys {
    /// Architecture identifier value written to `general.architecture`.
    pub const ARCHITECTURE: &str = "bonsai-image";

    /// Dual-stream layer count.
    pub const NUM_LAYERS: &str = "bonsai-image.num_layers";
    /// Single-stream layer count.
    pub const NUM_SINGLE_LAYERS: &str = "bonsai-image.num_single_layers";
    /// Attention head count.
    pub const ATTENTION_HEAD_COUNT: &str = "bonsai-image.attention.head_count";
    /// Per-head attention dimension.
    pub const ATTENTION_HEAD_DIM: &str = "bonsai-image.attention.head_dim";
    /// Joint attention feature width.
    pub const JOINT_ATTENTION_DIM: &str = "bonsai-image.joint_attention_dim";
    /// Latent channel count.
    pub const IN_CHANNELS: &str = "bonsai-image.in_channels";
    /// Feed-forward expansion ratio.
    pub const MLP_RATIO: &str = "bonsai-image.mlp_ratio";
    /// Per-axis RoPE dimensions.
    pub const AXES_DIMS_ROPE: &str = "bonsai-image.rope.axes_dims";
    /// RoPE base frequency.
    pub const ROPE_THETA: &str = "bonsai-image.rope.theta";
    /// Guidance-embedding flag.
    pub const GUIDANCE_EMBEDS: &str = "bonsai-image.guidance_embeds";
}

/// Write `bonsai-image` architecture metadata into a GGUF writer.
pub fn write_dit_metadata(writer: &mut GgufWriter, arch: &DitArch, model_name: &str) {
    writer.add_metadata(
        keys::GENERAL_ARCHITECTURE,
        MetadataWriteValue::Str(arch_keys::ARCHITECTURE.to_string()),
    );
    writer.add_metadata(
        keys::GENERAL_NAME,
        MetadataWriteValue::Str(model_name.to_string()),
    );
    writer.add_metadata(
        "general.quantization_version",
        MetadataWriteValue::Str("TQ2_0_G128".to_string()),
    );

    writer.add_metadata(
        arch_keys::NUM_LAYERS,
        MetadataWriteValue::U32(arch.num_layers),
    );
    writer.add_metadata(
        arch_keys::NUM_SINGLE_LAYERS,
        MetadataWriteValue::U32(arch.num_single_layers),
    );
    writer.add_metadata(
        arch_keys::ATTENTION_HEAD_COUNT,
        MetadataWriteValue::U32(arch.num_attention_heads),
    );
    writer.add_metadata(
        arch_keys::ATTENTION_HEAD_DIM,
        MetadataWriteValue::U32(arch.attention_head_dim),
    );
    writer.add_metadata(
        arch_keys::JOINT_ATTENTION_DIM,
        MetadataWriteValue::U32(arch.joint_attention_dim),
    );
    writer.add_metadata(
        arch_keys::IN_CHANNELS,
        MetadataWriteValue::U32(arch.in_channels),
    );
    writer.add_metadata(
        arch_keys::MLP_RATIO,
        MetadataWriteValue::F32(arch.mlp_ratio),
    );
    writer.add_metadata(
        arch_keys::AXES_DIMS_ROPE,
        MetadataWriteValue::ArrayU32(arch.axes_dims_rope.to_vec()),
    );
    writer.add_metadata(
        arch_keys::ROPE_THETA,
        MetadataWriteValue::F32(arch.rope_theta),
    );
    writer.add_metadata(
        arch_keys::GUIDANCE_EMBEDS,
        MetadataWriteValue::Bool(arch.guidance_embeds),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_arch_matches_design_doc() {
        let a = DitArch::default();
        assert_eq!(a.num_layers, 5);
        assert_eq!(a.num_single_layers, 20);
        assert_eq!(a.num_attention_heads, 24);
        assert_eq!(a.attention_head_dim, 128);
        assert_eq!(a.joint_attention_dim, 7680);
        assert_eq!(a.in_channels, 128);
        assert_eq!(a.mlp_ratio, 3.0);
        assert_eq!(a.axes_dims_rope, [32, 32, 32, 32]);
        assert_eq!(a.rope_theta, 2000.0);
        assert!(!a.guidance_embeds);
    }

    #[test]
    fn metadata_writes_without_panicking() {
        let mut w = GgufWriter::new();
        write_dit_metadata(&mut w, &DitArch::default(), "bonsai-image-4B");
        let bytes = w.to_bytes().expect("serialise metadata-only file");
        // Magic + version present.
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().expect("slice")),
            0x4655_4747
        );
    }
}
