//! Configuration for the Qwen3-4B text encoder.
//!
//! The architecture is fixed (Qwen3-4B as used by Bonsai-Image / FLUX.2 Klein),
//! so [`TeConfig::default`] carries the full configuration. If the exported
//! weights directory contains a `weights_manifest.json`, its scalar fields are
//! used to override the defaults, parsed with `serde_json` into the typed
//! `ManifestOverrides` shape (not a hand-rolled text scanner — `serde_json` is
//! already a crate dependency).

use std::path::{Path, PathBuf};

use serde::Deserialize;

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
    /// `dir`.
    ///
    /// Returns `Ok(None)` if the file does not exist (the normal case — not
    /// every weights directory ships a manifest); the caller should then use
    /// [`TeConfig::default`]. Returns `Err` if the file **is present** but is
    /// not valid JSON matching `ManifestOverrides`'s shape: a present but
    /// malformed manifest is a real misconfiguration and must surface to the
    /// caller rather than being silently swallowed into (possibly wrong)
    /// defaults.
    ///
    /// Every recognised field is optional and unrecognised JSON fields are
    /// ignored, so a partial or extended manifest still parses. Some exporters
    /// nest the language-model's own scalar config under a `"text_config"`
    /// object (sibling to `vision_config` / `quantization_config` for a
    /// multimodal checkpoint); when a field is absent at the top level, this
    /// deliberately falls back to `text_config`'s same-named field — but a
    /// top-level value always wins, and no *other* sibling object is ever
    /// consulted. This replaces an older implementation that scanned the raw
    /// file text for the first textual occurrence of a key, which was silently
    /// wrong whenever a nested object happened to be written before the real
    /// top-level key (see the regression test
    /// `top_level_scalar_wins_over_an_earlier_nested_sibling_object`).
    ///
    /// # Errors
    /// [`ManifestConfigError::InvalidJson`] if the file exists but cannot be
    /// parsed as the expected JSON shape.
    pub fn from_manifest_dir(dir: &Path) -> Result<Option<Self>, ManifestConfigError> {
        let path = dir.join("weights_manifest.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let overrides: ManifestOverrides =
            serde_json::from_str(&text).map_err(|source| ManifestConfigError::InvalidJson {
                path: path.clone(),
                source,
            })?;
        let mut cfg = Self::default();
        overrides.apply_to(&mut cfg);
        Ok(Some(cfg))
    }
}

/// Error parsing an existing `weights_manifest.json`.
///
/// Distinct from "the file does not exist" — that is a normal case and
/// [`TeConfig::from_manifest_dir`] returns `Ok(None)` for it, not an error.
/// This variant is only returned when the file **is present** but is not
/// valid JSON (or not an object matching `ManifestOverrides`).
#[derive(Debug, thiserror::Error)]
pub enum ManifestConfigError {
    /// The file exists but could not be parsed as the expected JSON shape.
    #[error("invalid weights_manifest.json at {path}: {source}")]
    InvalidJson {
        /// The manifest path that failed to parse.
        path: PathBuf,
        /// The underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
}

/// Typed, best-effort override of [`TeConfig`]'s scalar fields, parsed from a
/// `weights_manifest.json`.
///
/// Every field is optional (`#[serde(default)]`), so a manifest that specifies
/// only a few fields — or has extra, unrecognised ones — still parses cleanly;
/// unknown JSON fields are ignored (this struct does not use
/// `deny_unknown_fields`).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ManifestOverrides {
    num_layers: Option<usize>,
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    intermediate_size: Option<usize>,
    vocab_size: Option<usize>,
    rope_theta: Option<f32>,
    rms_norm_eps: Option<f32>,
    /// A nested language-model sub-config, when the exporter groups scalars
    /// this way (see [`TeConfig::from_manifest_dir`]'s docs). Only consulted
    /// as a fallback for a field that is absent at the top level.
    text_config: Option<Box<ManifestOverrides>>,
}

impl ManifestOverrides {
    /// Apply every present field onto `cfg`: the top-level value wins, then
    /// the same-named field inside a nested `text_config` (if any), otherwise
    /// `cfg`'s existing (default) value is left untouched.
    fn apply_to(&self, cfg: &mut TeConfig) {
        let nested = self.text_config.as_deref();
        if let Some(v) = self
            .num_layers
            .or_else(|| nested.and_then(|n| n.num_layers))
        {
            cfg.num_layers = v;
        }
        if let Some(v) = self
            .hidden_size
            .or_else(|| nested.and_then(|n| n.hidden_size))
        {
            cfg.hidden_size = v;
        }
        if let Some(v) = self
            .num_attention_heads
            .or_else(|| nested.and_then(|n| n.num_attention_heads))
        {
            cfg.num_attention_heads = v;
        }
        if let Some(v) = self
            .num_key_value_heads
            .or_else(|| nested.and_then(|n| n.num_key_value_heads))
        {
            cfg.num_key_value_heads = v;
        }
        if let Some(v) = self.head_dim.or_else(|| nested.and_then(|n| n.head_dim)) {
            cfg.head_dim = v;
        }
        if let Some(v) = self
            .intermediate_size
            .or_else(|| nested.and_then(|n| n.intermediate_size))
        {
            cfg.intermediate_size = v;
        }
        if let Some(v) = self
            .vocab_size
            .or_else(|| nested.and_then(|n| n.vocab_size))
        {
            cfg.vocab_size = v;
        }
        if let Some(v) = self
            .rope_theta
            .or_else(|| nested.and_then(|n| n.rope_theta))
        {
            cfg.rope_theta = v;
        }
        if let Some(v) = self
            .rms_norm_eps
            .or_else(|| nested.and_then(|n| n.rms_norm_eps))
        {
            cfg.rms_norm_eps = v;
        }
    }
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
    fn missing_manifest_returns_ok_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        // No weights_manifest.json written — this is the common case (most
        // weight directories do not ship one) and must not be an error.
        let result =
            TeConfig::from_manifest_dir(dir.path()).expect("a missing file is not an error");
        assert!(result.is_none());
    }

    #[test]
    fn malformed_manifest_is_a_typed_error_not_a_silent_default() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("weights_manifest.json"),
            b"this is not json {{{",
        )
        .expect("write manifest");
        let err = TeConfig::from_manifest_dir(dir.path())
            .expect_err("malformed JSON must surface as an error, not silently default");
        assert!(matches!(err, ManifestConfigError::InvalidJson { .. }));
    }

    #[test]
    fn manifest_applies_top_level_scalar_overrides() {
        let dir = tempfile::tempdir().expect("tempdir");
        let json = r#"{ "num_layers": 40, "rope_theta": 500000.0, "rms_norm_eps": 1e-05 }"#;
        std::fs::write(dir.path().join("weights_manifest.json"), json).expect("write manifest");
        let cfg = TeConfig::from_manifest_dir(dir.path())
            .expect("parse")
            .expect("file present");
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.rope_theta, 500_000.0);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-12);
        // A field the manifest did not mention keeps its Qwen3-4B default.
        assert_eq!(cfg.hidden_size, TeConfig::default().hidden_size);
    }

    /// Regression test for RAG-EVAL-IMG-27: the previous implementation scanned
    /// the raw file text for the first textual occurrence of `"rope_theta"`
    /// anywhere in the document, so a nested sibling object written *before*
    /// the real top-level key silently won. This manifest reproduces exactly
    /// that shape (a `vision_config` with its own `rope_theta` ahead of the
    /// real top-level one); the fix must read the true top-level value
    /// regardless of what precedes it in the file.
    #[test]
    fn top_level_scalar_wins_over_an_earlier_nested_sibling_object() {
        let dir = tempfile::tempdir().expect("tempdir");
        let json = r#"{
            "vision_config": { "rope_theta": 999.0 },
            "rope_theta": 1000000.0
        }"#;
        std::fs::write(dir.path().join("weights_manifest.json"), json).expect("write manifest");
        let cfg = TeConfig::from_manifest_dir(dir.path())
            .expect("parse")
            .expect("file present");
        assert_eq!(
            cfg.rope_theta, 1_000_000.0,
            "the top-level rope_theta must win over an unrelated nested sibling's same-named field"
        );
    }

    /// The deliberate `text_config` fallback: some HF multimodal manifests nest
    /// the language-model's own scalars there. Absent at the top level, it
    /// must still be honoured.
    #[test]
    fn falls_back_to_nested_text_config_when_top_level_field_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let json = r#"{ "text_config": { "num_layers": 48, "vocab_size": 200000 } }"#;
        std::fs::write(dir.path().join("weights_manifest.json"), json).expect("write manifest");
        let cfg = TeConfig::from_manifest_dir(dir.path())
            .expect("parse")
            .expect("file present");
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.vocab_size, 200_000);
    }

    /// A sibling object that is *not* named `text_config` (e.g. `vision_config`
    /// alone, with no usable top-level or `text_config` field) must never be
    /// treated as a fallback source — only the exact `text_config` key is.
    #[test]
    fn unrelated_sibling_object_is_never_used_as_a_fallback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let json = r#"{ "vision_config": { "num_layers": 999 } }"#;
        std::fs::write(dir.path().join("weights_manifest.json"), json).expect("write manifest");
        let cfg = TeConfig::from_manifest_dir(dir.path())
            .expect("parse")
            .expect("file present");
        assert_eq!(
            cfg.num_layers,
            TeConfig::default().num_layers,
            "vision_config must not be mistaken for text_config"
        );
    }
}
