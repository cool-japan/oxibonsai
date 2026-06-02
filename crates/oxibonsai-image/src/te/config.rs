//! Configuration for the Qwen3-4B text encoder.
//!
//! The architecture is fixed (Qwen3-4B as used by Bonsai-Image / FLUX.2 Klein),
//! so [`TeConfig::default`] carries the full configuration. If the exported
//! weights directory contains a `weights_manifest.json`, its scalar fields are
//! used to override the defaults (best-effort, via a tiny field extractor — no
//! extra JSON dependency is pulled into this crate for a handful of scalars).

use std::path::Path;

/// The hidden-state layer indices (1-indexed into `hidden_states_list`, i.e. the
/// outputs of decoder layers 8/17/26) that are stacked into the conditioning.
pub const STACK_LAYERS: [usize; 3] = [9, 18, 27];

/// Qwen3-4B text-encoder configuration.
#[derive(Debug, Clone)]
pub struct TeConfig {
    /// Token embedding table size.
    pub vocab_size: usize,
    /// Residual-stream width.
    pub hidden_size: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension (note `num_attention_heads * head_dim != hidden_size`).
    pub head_dim: usize,
    /// SwiGLU MLP intermediate width.
    pub intermediate_size: usize,
    /// RoPE base (`theta`).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl Default for TeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            num_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }
}

impl TeConfig {
    /// Number of query heads that share each key/value head (GQA group size).
    pub fn kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Total query-projection width (`num_attention_heads * head_dim`).
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Total key/value-projection width (`num_key_value_heads * head_dim`).
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Best-effort override of the defaults from a `weights_manifest.json` in
    /// `dir`. Returns `None` if the file is absent or unreadable; otherwise a
    /// config with any present scalar fields applied over the defaults.
    pub fn from_manifest_dir(dir: &Path) -> Option<Self> {
        let path = dir.join("weights_manifest.json");
        let text = std::fs::read_to_string(path).ok()?;
        let mut cfg = Self::default();
        if let Some(v) = extract_usize(&text, "num_layers") {
            cfg.num_layers = v;
        }
        if let Some(v) = extract_usize(&text, "hidden_size") {
            cfg.hidden_size = v;
        }
        if let Some(v) = extract_usize(&text, "num_attention_heads") {
            cfg.num_attention_heads = v;
        }
        if let Some(v) = extract_usize(&text, "num_key_value_heads") {
            cfg.num_key_value_heads = v;
        }
        if let Some(v) = extract_usize(&text, "head_dim") {
            cfg.head_dim = v;
        }
        if let Some(v) = extract_usize(&text, "intermediate_size") {
            cfg.intermediate_size = v;
        }
        if let Some(v) = extract_usize(&text, "vocab_size") {
            cfg.vocab_size = v;
        }
        if let Some(v) = extract_f32(&text, "rope_theta") {
            cfg.rope_theta = v;
        }
        if let Some(v) = extract_f32(&text, "rms_norm_eps") {
            cfg.rms_norm_eps = v;
        }
        Some(cfg)
    }
}

/// Extract the numeric token following `"key"` in a flat JSON string. Returns
/// the raw substring up to the next `,`, `}` or newline (trimmed).
fn extract_raw<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let k = text.find(&needle)?;
    let after = &text[k + needle.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let end = rest.find([',', '}', '\n']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

/// Extract an integer field from a flat JSON string.
fn extract_usize(text: &str, key: &str) -> Option<usize> {
    extract_raw(text, key)?.parse::<usize>().ok()
}

/// Extract a float field from a flat JSON string.
fn extract_f32(text: &str, key: &str) -> Option<f32> {
    extract_raw(text, key)?.parse::<f32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_qwen3_4b() {
        let c = TeConfig::default();
        assert_eq!(c.hidden_size, 2560);
        assert_eq!(c.num_layers, 36);
        assert_eq!(c.q_dim(), 4096);
        assert_eq!(c.kv_dim(), 1024);
        assert_eq!(c.kv_group(), 4);
    }

    #[test]
    fn extract_scalar_fields() {
        let json = r#"{ "num_layers": 36, "rope_theta": 1000000.0, "rms_norm_eps": 1e-06 }"#;
        assert_eq!(extract_usize(json, "num_layers"), Some(36));
        assert_eq!(extract_f32(json, "rope_theta"), Some(1_000_000.0));
        let eps = extract_f32(json, "rms_norm_eps").expect("eps");
        assert!((eps - 1e-6).abs() < 1e-12);
    }
}
