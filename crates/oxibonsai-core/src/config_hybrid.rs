//! `HybridConfig` — `qwen35` (Qwen3.5 / PrismML Bonsai 2) hyper-parameters.
//!
//! See `bonsai2-design.md` §1.6. A `qwen35` model interleaves full-attention
//! Transformer layers with recurrent Gated-DeltaNet ("linear attention")
//! layers; [`HybridConfig`] carries every hyperparameter needed to tell
//! which layer is which and to size the Gated-DeltaNet state, on top of the
//! ordinary dense-Qwen3 fields already covered by [`crate::config::Qwen3Config`]
//! (embedded as [`HybridConfig::base`]).

use crate::config::{arch_key, Qwen3Config};
use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::metadata::MetadataStore;

/// The single `general.architecture` value [`HybridConfig::from_metadata`]
/// accepts.
pub const HYBRID_ARCHITECTURE: &str = "qwen35";

/// Qwen3.5 / Bonsai 2 hybrid (full-attention + Gated-DeltaNet) configuration.
///
/// A superset of [`Qwen3Config`]; the base config is embedded so every
/// existing code path that only needs dense-Qwen3 fields keeps working
/// unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridConfig {
    /// `hidden_size`, `num_layers`, `head_count`/`head_count_kv`,
    /// `head_dim`/`value_length` (256 for the 27B), `vocab_size`,
    /// `context_length`, `rms_norm_eps`, `rope_freq_base`.
    pub base: Qwen3Config,
    /// `qwen35.full_attention_interval` (27B: 4). Layer `i` (0-indexed) is
    /// full attention iff `(i + 1) % full_attention_interval == 0`.
    pub full_attention_interval: usize,
    /// `qwen35.rope.dimension_count` — the number of rotated dimensions out
    /// of `base.head_dim` (27B: 64 of 256); a.k.a. `n_rot`.
    pub rope_dimension_count: usize,
    /// `qwen35.rope.dimension_sections` — M-RoPE section widths (27B:
    /// `[11, 11, 10, 0]`; text-only inference degenerates to standard RoPE
    /// over the first `rope_dimension_count` dimensions).
    pub rope_sections: [u32; 4],
    /// `qwen35.ssm.conv_kernel` — depthwise causal conv1d kernel width
    /// (27B: 4).
    pub ssm_conv_kernel: usize,
    /// `qwen35.ssm.state_size` — Gated-DeltaNet per-head state dimension,
    /// equal to both `head_k_dim` and `head_v_dim` (27B: 128).
    pub ssm_state_size: usize,
    /// `qwen35.ssm.group_count` — number of Gated-DeltaNet K-heads (`n_k`,
    /// 27B: 16).
    pub ssm_group_count: usize,
    /// `qwen35.ssm.time_step_rank` — number of Gated-DeltaNet V-heads
    /// (`n_v`, 27B: 48).
    pub ssm_time_step_rank: usize,
    /// `qwen35.ssm.inner_size` — Gated-DeltaNet value width, `n_v *
    /// head_v_dim` (27B: 6144).
    pub ssm_inner_size: usize,
    /// `qwen35.nextn_predict_layers` — MTP/NextN draft-head layer count.
    /// Absent on every real file seen so far (0 for the 27B: MTP is not
    /// executed), so this defaults to `0` rather than being required.
    pub nextn_predict_layers: usize,
    /// `general.sampling.top_k`, when declared.
    pub sampling_top_k: Option<u32>,
    /// `general.sampling.top_p`, when declared.
    pub sampling_top_p: Option<f32>,
    /// `general.sampling.temp`, when declared.
    pub sampling_temperature: Option<f32>,
}

impl HybridConfig {
    /// Parse a `qwen35` hybrid configuration from GGUF metadata.
    ///
    /// Returns [`BonsaiError::UnsupportedArchitecture`] immediately when
    /// `general.architecture` is not exactly [`HYBRID_ARCHITECTURE`] (rather
    /// than failing later with a confusing "missing qwen35.ssm.conv_kernel"
    /// on, say, a dense Qwen3-8B file). Every `qwen35`-specific hyperparameter
    /// is required — there is no dense-Qwen3 fallback for keys that only
    /// exist in the hybrid architecture.
    pub fn from_metadata(metadata: &MetadataStore) -> BonsaiResult<Self> {
        let base = Qwen3Config::from_metadata(metadata)?;
        if base.architecture != HYBRID_ARCHITECTURE {
            return Err(BonsaiError::UnsupportedArchitecture {
                arch: base.architecture,
            });
        }
        let arch = base.architecture.as_str();

        let full_attention_interval = required_usize(metadata, arch, "full_attention_interval")?;
        let rope_dimension_count = required_usize(metadata, arch, "rope.dimension_count")?;
        let rope_sections = required_rope_sections(metadata, arch)?;
        let ssm_conv_kernel = required_usize(metadata, arch, "ssm.conv_kernel")?;
        let ssm_state_size = required_usize(metadata, arch, "ssm.state_size")?;
        let ssm_group_count = required_usize(metadata, arch, "ssm.group_count")?;
        let ssm_time_step_rank = required_usize(metadata, arch, "ssm.time_step_rank")?;
        let ssm_inner_size = required_usize(metadata, arch, "ssm.inner_size")?;
        // Absent on every real 27B file staged for this session (MTP is not
        // executed there); treat as "no draft head" rather than requiring it.
        let nextn_predict_layers = metadata
            .get_u32(&arch_key(arch, "nextn_predict_layers"))
            .map(|v| v as usize)
            .unwrap_or(0);

        let sampling_top_k = metadata
            .get("general.sampling.top_k")
            .and_then(|v| v.as_u32());
        let sampling_top_p = metadata
            .get("general.sampling.top_p")
            .and_then(|v| v.as_f32());
        let sampling_temperature = metadata
            .get("general.sampling.temp")
            .and_then(|v| v.as_f32());

        let config = HybridConfig {
            base,
            full_attention_interval,
            rope_dimension_count,
            rope_sections,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
            nextn_predict_layers,
            sampling_top_k,
            sampling_top_p,
            sampling_temperature,
        };
        config.validate()?;
        Ok(config)
    }

    /// Whether layer `layer` (0-indexed) is full attention.
    ///
    /// `(i + 1) % full_attention_interval == 0` — for the 27B
    /// (`full_attention_interval == 4`, `block_count == 64`) this gives
    /// exactly `{3, 7, 11, ..., 63}` (16 layers); every other layer is
    /// Gated-DeltaNet ("linear attention").
    #[inline]
    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.full_attention_interval != 0
            && (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Number of Gated-DeltaNet K-heads (`n_k`).
    #[inline]
    pub fn n_k_heads(&self) -> usize {
        self.ssm_group_count
    }

    /// Number of Gated-DeltaNet V-heads (`n_v`).
    #[inline]
    pub fn n_v_heads(&self) -> usize {
        self.ssm_time_step_rank
    }

    /// Gated-DeltaNet per-K-head state dimension.
    #[inline]
    pub fn head_k_dim(&self) -> usize {
        self.ssm_state_size
    }

    /// Gated-DeltaNet per-V-head state dimension (`ssm_inner_size /
    /// n_v_heads`).
    #[inline]
    pub fn head_v_dim(&self) -> usize {
        self.ssm_inner_size / self.n_v_heads().max(1)
    }

    /// How many V-heads share each K-head (`n_v / n_k`; 27B: 3).
    #[inline]
    pub fn v_per_k(&self) -> usize {
        self.n_v_heads() / self.n_k_heads().max(1)
    }

    /// Width of the concatenated `[k | v]` (or `[q | k | v]`, depending on
    /// caller convention) activation the depthwise conv1d runs over for one
    /// linear-attention layer: `2 * head_k_dim * n_k_heads + ssm_inner_size`
    /// (27B: `2*128*16 + 6144 = 10240`, matching `attn_qkv.weight`'s output
    /// width).
    #[inline]
    pub fn conv_dim(&self) -> usize {
        2 * self.head_k_dim() * self.n_k_heads() + self.ssm_inner_size
    }

    /// Number of full-attention layers among `base.num_layers`.
    pub fn num_full_layers(&self) -> usize {
        (0..self.base.num_layers)
            .filter(|&i| self.is_full_attention(i))
            .count()
    }

    /// Number of Gated-DeltaNet ("linear attention") layers among
    /// `base.num_layers`.
    pub fn num_linear_layers(&self) -> usize {
        self.base.num_layers - self.num_full_layers()
    }

    /// Structural validation mirroring `runtime.py::load`: `n_v_heads` and
    /// `n_k_heads` must both be nonzero, `n_v_heads` must be a whole
    /// multiple of `n_k_heads` (so [`Self::v_per_k`] is exact), and
    /// `ssm_inner_size` must be a whole multiple of `n_v_heads` (so
    /// [`Self::head_v_dim`] is exact). Also rejects
    /// `full_attention_interval == 0`, which would otherwise silently make
    /// [`Self::is_full_attention`] always false and every layer count as
    /// linear, and `sum(rope_sections) > rope_dimension_count / 2`
    /// (design §1.6), which would otherwise let the M-RoPE section table
    /// address dimensions past `n_rot`'s rotated half and read/write
    /// out-of-range positions in `PartialRopeTable`'s `(i, i + n_rot/2)`
    /// pairing. The real 27B is exactly at the boundary
    /// (`11+11+10+0 == 32 == 64/2`), so the check is `<=`, not `<`.
    pub fn validate(&self) -> BonsaiResult<()> {
        if self.full_attention_interval == 0 {
            return Err(BonsaiError::InvalidMetadata {
                key: arch_key(&self.base.architecture, "full_attention_interval"),
                reason: "must be nonzero".to_string(),
            });
        }
        let nk = self.n_k_heads();
        let nv = self.n_v_heads();
        if nk == 0 || nv == 0 {
            return Err(BonsaiError::InvalidMetadata {
                key: arch_key(&self.base.architecture, "ssm.group_count/time_step_rank"),
                reason: format!("n_k_heads ({nk}) and n_v_heads ({nv}) must both be nonzero"),
            });
        }
        if !nv.is_multiple_of(nk) {
            return Err(BonsaiError::InvalidMetadata {
                key: arch_key(&self.base.architecture, "ssm.time_step_rank"),
                reason: format!("n_v_heads ({nv}) must be a whole multiple of n_k_heads ({nk})"),
            });
        }
        if !self.ssm_inner_size.is_multiple_of(nv) {
            return Err(BonsaiError::InvalidMetadata {
                key: arch_key(&self.base.architecture, "ssm.inner_size"),
                reason: format!(
                    "ssm_inner_size ({}) must be a whole multiple of n_v_heads ({nv})",
                    self.ssm_inner_size
                ),
            });
        }
        let sections_sum: usize = self.rope_sections.iter().map(|&s| s as usize).sum();
        let half_n_rot = self.rope_dimension_count / 2;
        if sections_sum > half_n_rot {
            return Err(BonsaiError::InvalidMetadata {
                key: arch_key(&self.base.architecture, "rope.dimension_sections"),
                reason: format!(
                    "sections sum ({sections_sum}) must not exceed rope_dimension_count/2 \
                     ({half_n_rot})"
                ),
            });
        }
        Ok(())
    }
}

/// Read a required `usize` hyperparameter under `<arch>.<suffix>` — no
/// legacy-`llm.*` fallback, since every `qwen35`-only key postdates that
/// (now-fixed) converter bug.
fn required_usize(metadata: &MetadataStore, arch: &str, suffix: &str) -> BonsaiResult<usize> {
    metadata
        .get_u32(&arch_key(arch, suffix))
        .map(|v| v as usize)
        .map_err(|_| BonsaiError::MissingConfigKey {
            key: arch_key(arch, suffix),
        })
}

/// Read `<arch>.rope.dimension_sections` as the fixed-size `[u32; 4]` the
/// M-RoPE section machinery expects.
fn required_rope_sections(metadata: &MetadataStore, arch: &str) -> BonsaiResult<[u32; 4]> {
    let key = arch_key(arch, "rope.dimension_sections");
    let raw = metadata
        .get_i32_array(&key)
        .map_err(|_| BonsaiError::MissingConfigKey { key: key.clone() })?;
    let mut sections = [0u32; 4];
    if raw.len() != sections.len() {
        return Err(BonsaiError::InvalidMetadata {
            key: key.clone(),
            reason: format!(
                "must have exactly {} elements, got {}",
                sections.len(),
                raw.len()
            ),
        });
    }
    for (slot, value) in sections.iter_mut().zip(raw) {
        *slot = u32::try_from(value).map_err(|_| BonsaiError::InvalidMetadata {
            key: key.clone(),
            reason: format!("element {value} is negative"),
        })?;
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::reader::GgufFile;
    use crate::gguf::writer::{GgufWriter, MetadataWriteValue};

    fn build_metadata_store(pairs: Vec<(&str, MetadataWriteValue)>) -> MetadataStore {
        let mut writer = GgufWriter::new();
        for (key, value) in pairs {
            writer.add_metadata(key, value);
        }
        let bytes = writer
            .to_bytes()
            .expect("synthetic metadata should serialise");
        GgufFile::parse(&bytes)
            .expect("synthetic metadata should parse back")
            .metadata
    }

    /// A complete, valid `qwen35` hybrid metadata set with the real 27B's
    /// SSM/rope shape numbers (block_count/embedding_length/etc. are kept
    /// tiny for test speed; only the qwen35-specific hyperparameters need to
    /// match the real ratios for `validate()`/`is_full_attention` to be
    /// meaningfully exercised).
    fn full_hybrid_pairs() -> Vec<(&'static str, MetadataWriteValue)> {
        vec![
            (
                "general.architecture",
                MetadataWriteValue::Str("qwen35".to_string()),
            ),
            ("qwen35.embedding_length", MetadataWriteValue::U32(64)),
            ("qwen35.block_count", MetadataWriteValue::U32(64)),
            ("qwen35.attention.head_count", MetadataWriteValue::U32(4)),
            ("qwen35.attention.head_count_kv", MetadataWriteValue::U32(2)),
            ("qwen35.feed_forward_length", MetadataWriteValue::U32(128)),
            ("qwen35.vocab_size", MetadataWriteValue::U32(32)),
            ("qwen35.context_length", MetadataWriteValue::U32(1024)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                MetadataWriteValue::F32(1e-6),
            ),
            (
                "qwen35.rope.freq_base",
                MetadataWriteValue::F32(10_000_000.0),
            ),
            ("qwen35.full_attention_interval", MetadataWriteValue::U32(4)),
            ("qwen35.rope.dimension_count", MetadataWriteValue::U32(64)),
            (
                "qwen35.rope.dimension_sections",
                MetadataWriteValue::ArrayI32(vec![11, 11, 10, 0]),
            ),
            ("qwen35.ssm.conv_kernel", MetadataWriteValue::U32(4)),
            ("qwen35.ssm.state_size", MetadataWriteValue::U32(128)),
            ("qwen35.ssm.group_count", MetadataWriteValue::U32(16)),
            ("qwen35.ssm.time_step_rank", MetadataWriteValue::U32(48)),
            ("qwen35.ssm.inner_size", MetadataWriteValue::U32(6144)),
        ]
    }

    #[test]
    fn parses_the_real_27b_shape_numbers() {
        let metadata = build_metadata_store(full_hybrid_pairs());
        let hybrid = HybridConfig::from_metadata(&metadata).expect("should parse");
        assert_eq!(hybrid.full_attention_interval, 4);
        assert_eq!(hybrid.rope_dimension_count, 64);
        assert_eq!(hybrid.rope_sections, [11, 11, 10, 0]);
        assert_eq!(hybrid.ssm_conv_kernel, 4);
        assert_eq!(hybrid.ssm_state_size, 128);
        assert_eq!(hybrid.ssm_group_count, 16);
        assert_eq!(hybrid.ssm_time_step_rank, 48);
        assert_eq!(hybrid.ssm_inner_size, 6144);
        assert_eq!(hybrid.nextn_predict_layers, 0);
        assert_eq!(hybrid.n_k_heads(), 16);
        assert_eq!(hybrid.n_v_heads(), 48);
        assert_eq!(hybrid.head_k_dim(), 128);
        assert_eq!(hybrid.head_v_dim(), 128);
        assert_eq!(hybrid.v_per_k(), 3);
        assert_eq!(hybrid.conv_dim(), 10240);
        assert!(hybrid.validate().is_ok());
    }

    #[test]
    fn is_full_attention_gives_exactly_the_real_layer_set() {
        let metadata = build_metadata_store(full_hybrid_pairs());
        let hybrid = HybridConfig::from_metadata(&metadata).expect("should parse");
        let full: Vec<usize> = (0..64).filter(|&i| hybrid.is_full_attention(i)).collect();
        let expected: Vec<usize> = (0..64).filter(|i| (i + 1) % 4 == 0).collect();
        assert_eq!(full, expected);
        assert_eq!(full.len(), 16);
        assert_eq!(full.first(), Some(&3));
        assert_eq!(full.last(), Some(&63));
        assert_eq!(hybrid.num_full_layers(), 16);
        assert_eq!(hybrid.num_linear_layers(), 48);
    }

    /// A complete, valid *dense* Qwen3 metadata set (no `qwen35.*`/`ssm.*`
    /// keys at all) — a genuine Bonsai-8B-style file, not merely a
    /// mislabeled `qwen35` one, so `Qwen3Config::from_metadata` itself
    /// succeeds and the rejection below is unambiguously about
    /// `HybridConfig`'s own architecture check, not a missing-key error
    /// bubbling up from `base`.
    fn full_dense_qwen3_pairs() -> Vec<(&'static str, MetadataWriteValue)> {
        vec![
            (
                "general.architecture",
                MetadataWriteValue::Str("qwen3".to_string()),
            ),
            ("qwen3.embedding_length", MetadataWriteValue::U32(64)),
            ("qwen3.block_count", MetadataWriteValue::U32(2)),
            ("qwen3.attention.head_count", MetadataWriteValue::U32(4)),
            ("qwen3.attention.head_count_kv", MetadataWriteValue::U32(2)),
            ("qwen3.feed_forward_length", MetadataWriteValue::U32(128)),
            ("qwen3.vocab_size", MetadataWriteValue::U32(32)),
            ("qwen3.context_length", MetadataWriteValue::U32(512)),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                MetadataWriteValue::F32(1e-6),
            ),
            ("qwen3.rope.freq_base", MetadataWriteValue::F32(10_000.0)),
        ]
    }

    #[test]
    fn rejects_non_qwen35_architecture() {
        let metadata = build_metadata_store(full_dense_qwen3_pairs());
        // Sanity: this metadata is a genuinely valid dense Qwen3 config on
        // its own terms, so the rejection below is really about
        // HybridConfig's architecture check, not a missing-key error from
        // `Qwen3Config::from_metadata` bubbling up first.
        assert!(Qwen3Config::from_metadata(&metadata).is_ok());

        let err = HybridConfig::from_metadata(&metadata)
            .expect_err("a dense qwen3 file must not parse as a hybrid config");
        match err {
            BonsaiError::UnsupportedArchitecture { arch } => assert_eq!(arch, "qwen3"),
            other => panic!("expected UnsupportedArchitecture, got {other:?}"),
        }
    }

    #[test]
    fn missing_ssm_key_is_hard_error() {
        let mut pairs = full_hybrid_pairs();
        pairs.retain(|(k, _)| *k != "qwen35.ssm.inner_size");
        let metadata = build_metadata_store(pairs);
        let err = HybridConfig::from_metadata(&metadata)
            .expect_err("a missing ssm.inner_size must be a hard error");
        match err {
            BonsaiError::MissingConfigKey { key } => assert_eq!(key, "qwen35.ssm.inner_size"),
            other => panic!("expected MissingConfigKey, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_zero_full_attention_interval() {
        let mut pairs = full_hybrid_pairs();
        for (k, v) in pairs.iter_mut() {
            if *k == "qwen35.full_attention_interval" {
                *v = MetadataWriteValue::U32(0);
            }
        }
        let metadata = build_metadata_store(pairs);
        let err = HybridConfig::from_metadata(&metadata)
            .expect_err("full_attention_interval == 0 must be rejected");
        assert!(
            matches!(err, BonsaiError::InvalidMetadata { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn validate_rejects_time_step_rank_not_a_multiple_of_group_count() {
        let mut pairs = full_hybrid_pairs();
        for (k, v) in pairs.iter_mut() {
            if *k == "qwen35.ssm.time_step_rank" {
                *v = MetadataWriteValue::U32(47); // not a multiple of group_count=16
            }
        }
        let metadata = build_metadata_store(pairs);
        let err =
            HybridConfig::from_metadata(&metadata).expect_err("nv % nk != 0 must be rejected");
        assert!(
            matches!(err, BonsaiError::InvalidMetadata { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn validate_rejects_inner_size_not_a_multiple_of_time_step_rank() {
        let mut pairs = full_hybrid_pairs();
        for (k, v) in pairs.iter_mut() {
            if *k == "qwen35.ssm.inner_size" {
                *v = MetadataWriteValue::U32(6145); // not a multiple of nv=48
            }
        }
        let metadata = build_metadata_store(pairs);
        let err =
            HybridConfig::from_metadata(&metadata).expect_err("inner % nv != 0 must be rejected");
        assert!(
            matches!(err, BonsaiError::InvalidMetadata { .. }),
            "{err:?}"
        );
    }

    /// Design §1.6's `sections sum <= n_rot/2` clause. The real 27B sits
    /// exactly at the boundary (`11+11+10+0 == 32 == 64/2`), so this test
    /// checks the boundary is accepted and one-past-it is rejected, rather
    /// than only a comfortably-valid or comfortably-invalid value.
    #[test]
    fn validate_accepts_sections_sum_exactly_at_half_n_rot() {
        let metadata = build_metadata_store(full_hybrid_pairs());
        let hybrid = HybridConfig::from_metadata(&metadata).expect("should parse");
        assert_eq!(hybrid.rope_sections.iter().sum::<u32>(), 32);
        assert_eq!(hybrid.rope_dimension_count, 64);
        assert!(hybrid.validate().is_ok());
    }

    #[test]
    fn validate_rejects_sections_sum_exceeding_half_n_rot() {
        let mut pairs = full_hybrid_pairs();
        for (k, v) in pairs.iter_mut() {
            if *k == "qwen35.rope.dimension_sections" {
                // sum 33 > 64/2 == 32.
                *v = MetadataWriteValue::ArrayI32(vec![11, 11, 11, 0]);
            }
        }
        let metadata = build_metadata_store(pairs);
        let err = HybridConfig::from_metadata(&metadata)
            .expect_err("sections sum exceeding rope_dimension_count/2 must be rejected");
        assert!(
            matches!(err, BonsaiError::InvalidMetadata { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn sampling_defaults_are_read_when_present_and_none_when_absent() {
        let metadata = build_metadata_store(full_hybrid_pairs());
        let hybrid = HybridConfig::from_metadata(&metadata).expect("should parse");
        assert_eq!(hybrid.sampling_top_k, None);
        assert_eq!(hybrid.sampling_top_p, None);
        assert_eq!(hybrid.sampling_temperature, None);

        let mut pairs = full_hybrid_pairs();
        pairs.push(("general.sampling.top_k", MetadataWriteValue::U32(20)));
        pairs.push(("general.sampling.top_p", MetadataWriteValue::F32(0.95)));
        pairs.push(("general.sampling.temp", MetadataWriteValue::F32(1.0)));
        let metadata = build_metadata_store(pairs);
        let hybrid = HybridConfig::from_metadata(&metadata).expect("should parse");
        assert_eq!(hybrid.sampling_top_k, Some(20));
        assert_eq!(hybrid.sampling_top_p, Some(0.95));
        assert_eq!(hybrid.sampling_temperature, Some(1.0));
    }
}
