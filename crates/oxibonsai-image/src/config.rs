//! FLUX.2 DiT (`bonsai-image`) configuration parsed from GGUF metadata.
//!
//! All architectural dimensions are read from the `bonsai-image.*` metadata
//! key namespace written by the `oxibonsai-model` MLX→GGUF converter
//! (`crates/oxibonsai-model/src/convert/mlx_image/metadata.rs`). Derived
//! quantities (e.g. [`DitConfig::hidden_size`]) are computed, never stored
//! redundantly in the file.
//!
//! # `eps` is NOT read from the GGUF
//!
//! The current converter does **not** write the layer-norm epsilon into the
//! GGUF metadata, so [`DitConfig::eps`] is filled from the documented default
//! [`DEFAULT_EPS`] (`1e-6`). This is recorded as a converter follow-up rather
//! than silently hardcoded inside an opaque struct; once the converter emits
//! the key, [`DitConfig::from_metadata`] should prefer the file value.

use oxibonsai_core::gguf::metadata::{MetadataStore, MetadataValue};

use crate::error::{DitError, DitResult};

/// Architecture identifier expected in `general.architecture`.
pub const ARCHITECTURE: &str = "bonsai-image";

/// Default layer-norm epsilon for the FLUX.2 DiT.
///
/// NOTE: the converter does not currently write an `eps` metadata key, so this
/// default is used. See the module docs.
pub const DEFAULT_EPS: f32 = 1e-6;

/// `general.architecture` metadata key.
const KEY_ARCHITECTURE: &str = "general.architecture";
/// Dual-stream layer count.
const KEY_NUM_LAYERS: &str = "bonsai-image.num_layers";
/// Single-stream layer count.
const KEY_NUM_SINGLE_LAYERS: &str = "bonsai-image.num_single_layers";
/// Attention head count.
const KEY_HEAD_COUNT: &str = "bonsai-image.attention.head_count";
/// Per-head attention dimension.
const KEY_HEAD_DIM: &str = "bonsai-image.attention.head_dim";
/// Joint (text+image) attention feature width.
const KEY_JOINT_ATTENTION_DIM: &str = "bonsai-image.joint_attention_dim";
/// Latent channel count.
const KEY_IN_CHANNELS: &str = "bonsai-image.in_channels";
/// Feed-forward expansion ratio.
const KEY_MLP_RATIO: &str = "bonsai-image.mlp_ratio";
/// Per-axis RoPE dimensions (array).
const KEY_AXES_DIMS_ROPE: &str = "bonsai-image.rope.axes_dims";
/// RoPE base frequency.
const KEY_ROPE_THETA: &str = "bonsai-image.rope.theta";
/// Guidance-embedding flag.
const KEY_GUIDANCE_EMBEDS: &str = "bonsai-image.guidance_embeds";

/// Parsed FLUX.2 DiT architecture configuration.
///
/// Fields mirror the diffusers `Flux*Transformer` config. Construct via
/// [`DitConfig::from_metadata`] against a parsed GGUF metadata store.
#[derive(Debug, Clone, PartialEq)]
pub struct DitConfig {
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
    pub axes_dims_rope: Vec<u32>,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether the model consumes a guidance-embedding input.
    pub guidance_embeds: bool,
    /// Layer-norm epsilon. Filled from [`DEFAULT_EPS`] (see module docs);
    /// the converter does not currently persist this value.
    pub eps: f32,
}

impl DitConfig {
    /// Parse a [`DitConfig`] from GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns [`DitError::WrongArchitecture`] if `general.architecture` is not
    /// `"bonsai-image"`, [`DitError::MissingMetadata`] if a required
    /// `bonsai-image.*` key is absent, and [`DitError::InvalidMetadata`] if a
    /// value has the wrong type/shape.
    pub fn from_metadata(meta: &MetadataStore) -> DitResult<Self> {
        let arch = meta
            .get(KEY_ARCHITECTURE)
            .and_then(|v| v.as_str())
            .ok_or_else(|| DitError::MissingMetadata {
                key: KEY_ARCHITECTURE.to_string(),
            })?;
        if arch != ARCHITECTURE {
            return Err(DitError::WrongArchitecture {
                found: arch.to_string(),
                expected: ARCHITECTURE.to_string(),
            });
        }

        let num_layers = require_u32(meta, KEY_NUM_LAYERS)?;
        let num_single_layers = require_u32(meta, KEY_NUM_SINGLE_LAYERS)?;
        let num_attention_heads = require_u32(meta, KEY_HEAD_COUNT)?;
        let attention_head_dim = require_u32(meta, KEY_HEAD_DIM)?;
        let joint_attention_dim = require_u32(meta, KEY_JOINT_ATTENTION_DIM)?;
        let in_channels = require_u32(meta, KEY_IN_CHANNELS)?;
        let mlp_ratio = require_f32(meta, KEY_MLP_RATIO)?;
        let axes_dims_rope = require_u32_array(meta, KEY_AXES_DIMS_ROPE)?;
        let rope_theta = require_f32(meta, KEY_ROPE_THETA)?;
        let guidance_embeds = require_bool(meta, KEY_GUIDANCE_EMBEDS)?;

        Ok(Self {
            num_layers,
            num_single_layers,
            num_attention_heads,
            attention_head_dim,
            joint_attention_dim,
            in_channels,
            mlp_ratio,
            axes_dims_rope,
            rope_theta,
            guidance_embeds,
            eps: DEFAULT_EPS,
        })
    }

    /// Hidden size = `num_attention_heads * attention_head_dim`.
    pub fn hidden_size(&self) -> u32 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Feed-forward inner width = `round(hidden_size * mlp_ratio)`.
    pub fn ffn_inner_size(&self) -> u32 {
        ((self.hidden_size() as f32) * self.mlp_ratio).round() as u32
    }

    /// Total RoPE dimension across all axes (sum of [`Self::axes_dims_rope`]).
    pub fn rope_dim(&self) -> u32 {
        self.axes_dims_rope.iter().sum()
    }
}

/// Read a required `u32` metadata value.
fn require_u32(meta: &MetadataStore, key: &str) -> DitResult<u32> {
    meta.get(key)
        .and_then(|v| v.as_u32())
        .ok_or_else(|| missing_or_invalid(meta, key, "expected u32"))
}

/// Read a required `f32` metadata value.
fn require_f32(meta: &MetadataStore, key: &str) -> DitResult<f32> {
    meta.get(key)
        .and_then(|v| v.as_f32())
        .ok_or_else(|| missing_or_invalid(meta, key, "expected f32"))
}

/// Read a required `bool` metadata value.
fn require_bool(meta: &MetadataStore, key: &str) -> DitResult<bool> {
    meta.get(key)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| missing_or_invalid(meta, key, "expected bool"))
}

/// Read a required array-of-`u32` metadata value.
fn require_u32_array(meta: &MetadataStore, key: &str) -> DitResult<Vec<u32>> {
    let value = meta.get(key).ok_or_else(|| DitError::MissingMetadata {
        key: key.to_string(),
    })?;
    let MetadataValue::Array(items) = value else {
        return Err(DitError::InvalidMetadata {
            key: key.to_string(),
            reason: "expected array".to_string(),
        });
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let n = item.as_u32().ok_or_else(|| DitError::InvalidMetadata {
            key: key.to_string(),
            reason: "array element is not a u32".to_string(),
        })?;
        out.push(n);
    }
    Ok(out)
}

/// Build the right error for a key that is either absent or wrongly typed.
fn missing_or_invalid(meta: &MetadataStore, key: &str, reason: &str) -> DitError {
    if meta.get(key).is_some() {
        DitError::InvalidMetadata {
            key: key.to_string(),
            reason: reason.to_string(),
        }
    } else {
        DitError::MissingMetadata {
            key: key.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_size_and_ffn_derivation() {
        let cfg = DitConfig {
            num_layers: 5,
            num_single_layers: 20,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 7680,
            in_channels: 128,
            mlp_ratio: 3.0,
            axes_dims_rope: vec![32, 32, 32, 32],
            rope_theta: 2000.0,
            guidance_embeds: false,
            eps: DEFAULT_EPS,
        };
        assert_eq!(cfg.hidden_size(), 3072);
        assert_eq!(cfg.ffn_inner_size(), 9216);
        assert_eq!(cfg.rope_dim(), 128);
        assert_eq!(cfg.eps, 1e-6);
    }
}
