//! GGUF `general.*` and `<arch>.*` metadata, shared by every writer.
//!
//! # Why this module exists (CQ-01 / core-gguf-02)
//!
//! Three separate writers used to invent their own metadata block, and all
//! three were wrong in the same two ways:
//!
//! * `general.quantization_version` was written as a GGUF **string**
//!   (`"TQ2_0_G128"`, `"Q1_0G128"`, …). The spec — and `llama.cpp` — type that
//!   key as `uint32`, so `MetadataValue::as_u32` returns `None` and even
//!   OxiBonsai's own `get_u32` errors on its own files. The correct value is
//!   the GGUF quantization-schema version, `2`; the human-readable label
//!   belongs in [`keys::GENERAL_FILE_TYPE`] plus the non-normative
//!   [`OXIBONSAI_QUANT_FORMAT`].
//! * Architecture hyper-parameters were written under an `llm.*` prefix. That
//!   namespace does not exist in GGUF: every real file (and every reader,
//!   including [`oxibonsai_core::Qwen3Config::from_metadata`]) looks them up
//!   under `<general.architecture>.*` — `qwen3.embedding_length`,
//!   `qwen35.ssm.state_size`, and so on. Files written with `llm.*` load only
//!   because the reader falls back to hard-coded 8 B defaults, which is how
//!   `oxibonsai info` came to print fabricated numbers.
//!
//! Everything that writes a GGUF in this crate now goes through
//! [`arch_key`], [`write_general_metadata`] and [`write_arch_metadata`], so
//! the two defects cannot be reintroduced in one writer and not the others.

use std::collections::BTreeSet;

use oxibonsai_core::gguf::tensor_info::keys;
use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue, TensorType};
use serde_json::Value;

/// GGUF quantization-schema version. `2` is what every current ggml writer
/// emits and what the Bonsai 2 27 B files carry.
pub const QUANTIZATION_VERSION: u32 = 2;

/// Non-normative key carrying the human-readable format label that used to be
/// (incorrectly) stuffed into `general.quantization_version`.
pub const OXIBONSAI_QUANT_FORMAT: &str = "oxibonsai.quant_format";

/// Non-normative key recording which [`crate::quantize::ScaleRule`] produced
/// the file, so a later re-quantization can reproduce it exactly.
pub const OXIBONSAI_SCALE_RULE: &str = "oxibonsai.quant_scale_rule";

/// Build an architecture-scoped metadata key: `arch_key("qwen35", "ssm.state_size")`
/// → `"qwen35.ssm.state_size"`.
///
/// This is the only way this crate composes an architecture key; the
/// `keys::LLM_*` constants name a namespace GGUF does not have and must not
/// be used for new writes.
#[inline]
pub fn arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

/// The `general.file_type` value (`llama_ftype`) that corresponds to a
/// dominant tensor type.
///
/// The values in the PrismML extension range are **observed**, not derived —
/// they are read straight out of the shipped 27 B headers
/// (`Bonsai-27B-Q1_0.gguf` → 40, `*-Q2_0.gguf` → 41, `*-PQ2_0.gguf` → 141,
/// `*-PTQ1_0.gguf` → 143), which is why they are not simply "tensor id minus
/// one": the fork's `llama_ftype` enum skips 142. The mainline values match
/// `llama.cpp`'s `enum llama_ftype`.
pub fn ggml_file_type(tensor_type: TensorType) -> u32 {
    match tensor_type {
        TensorType::F32 => 0,   // ALL_F32
        TensorType::F16 => 1,   // MOSTLY_F16
        TensorType::Q4_0 => 2,  // MOSTLY_Q4_0
        TensorType::Q8_0 => 7,  // MOSTLY_Q8_0
        TensorType::Q2_K => 10, // MOSTLY_Q2_K
        // `_M` (not `_S = 11`), matching this file's existing convention of
        // naming the "medium" upstream ftype for a K-quant type rather than
        // the small/large sibling (see `Q4_K`/`Q5_K` below).
        TensorType::Q3_K => 12,                             // MOSTLY_Q3_K_M
        TensorType::Q4_K => 15,                             // MOSTLY_Q4_K_M
        TensorType::Q5_K => 17,                             // MOSTLY_Q5_K_M
        TensorType::Q6_K => 18,                             // MOSTLY_Q6_K
        TensorType::BF16 => 32,                             // MOSTLY_BF16
        TensorType::TQ1_0 => 36,                            // MOSTLY_TQ1_0
        TensorType::TQ2_0 => 37,                            // MOSTLY_TQ2_0
        TensorType::Q1_0G128 => 40,                         // PrismML MOSTLY_Q1_0
        TensorType::TQ2_0_g128 | TensorType::Q2_0G64 => 41, // PrismML MOSTLY_Q2_0
        TensorType::PQ2_0 => 141,                           // PrismML MOSTLY_PQ2_0
        TensorType::PTQ1_0 => 143,                          // PrismML MOSTLY_PTQ1_0
        // No upstream `llama_ftype` exists for these OxiBonsai/PrismML
        // extensions; fall back to the tensor id so the value is at least
        // unambiguous rather than a lie about a standard format. `Q8_K` has
        // no upstream `llama_ftype` either (it is a llama.cpp *intermediate*
        // quantization-time format, never a file's dominant `general.file_type`
        // upstream), so it joins this fallback arm rather than inventing one.
        TensorType::F8_E4M3
        | TensorType::F8_E5M2
        | TensorType::MXFP4
        | TensorType::NVFP4
        | TensorType::Q8_K => tensor_type.wire_id(),
    }
}

// ─── general.* ────────────────────────────────────────────────────────────────

/// The `general.*` block of a GGUF file.
#[derive(Debug, Clone)]
pub struct GeneralMetadata {
    /// `general.architecture` — also the prefix of every `<arch>.*` key.
    pub architecture: String,
    /// `general.name`.
    pub name: String,
    /// `general.version`, when the source carries one.
    pub version: Option<String>,
    /// `general.description`, when the source carries one.
    pub description: Option<String>,
    /// `general.file_type` — see [`ggml_file_type`].
    pub file_type: u32,
    /// Human-readable format label for [`OXIBONSAI_QUANT_FORMAT`].
    pub quant_format: Option<String>,
    /// Scale rule label for [`OXIBONSAI_SCALE_RULE`].
    pub scale_rule: Option<String>,
}

impl GeneralMetadata {
    /// A minimal block: architecture, name and the file type implied by the
    /// dominant tensor type.
    pub fn new(architecture: &str, name: &str, dominant: TensorType) -> Self {
        Self {
            architecture: architecture.to_string(),
            name: name.to_string(),
            version: None,
            description: None,
            file_type: ggml_file_type(dominant),
            quant_format: Some(format!("{dominant:?}")),
            scale_rule: None,
        }
    }
}

/// Write the `general.*` block.
///
/// `general.quantization_version` is always [`QUANTIZATION_VERSION`] as a
/// `U32` — never a string (core-gguf-02). `general.alignment` is deliberately
/// **not** written here: [`GgufWriter`] injects it itself when it uses a
/// non-default alignment, and a second, possibly contradictory value would
/// make the reader compute the wrong data offset.
pub fn write_general_metadata(writer: &mut GgufWriter<'_>, general: &GeneralMetadata) {
    writer.add_metadata(
        keys::GENERAL_ARCHITECTURE,
        MetadataWriteValue::Str(general.architecture.clone()),
    );
    writer.add_metadata(
        keys::GENERAL_NAME,
        MetadataWriteValue::Str(general.name.clone()),
    );
    if let Some(ref v) = general.version {
        writer.add_metadata("general.version", MetadataWriteValue::Str(v.clone()));
    }
    if let Some(ref d) = general.description {
        writer.add_metadata("general.description", MetadataWriteValue::Str(d.clone()));
    }
    writer.add_metadata(
        keys::GENERAL_FILE_TYPE,
        MetadataWriteValue::U32(general.file_type),
    );
    writer.add_metadata(
        keys::GENERAL_QUANTIZATION_VERSION,
        MetadataWriteValue::U32(QUANTIZATION_VERSION),
    );
    if let Some(ref label) = general.quant_format {
        writer.add_metadata(
            OXIBONSAI_QUANT_FORMAT,
            MetadataWriteValue::Str(label.clone()),
        );
    }
    if let Some(ref rule) = general.scale_rule {
        writer.add_metadata(OXIBONSAI_SCALE_RULE, MetadataWriteValue::Str(rule.clone()));
    }
}

// ─── <arch>.* ─────────────────────────────────────────────────────────────────

/// The architecture hyper-parameters every language-model GGUF must carry.
///
/// These are exactly the keys [`oxibonsai_core::Qwen3Config::from_metadata`]
/// reads; omitting them is what made `oxibonsai quantize` produce files
/// OxiBonsai itself could not load with correct dimensions (CQ-01).
#[derive(Debug, Clone, PartialEq)]
pub struct ArchMetadata {
    /// `<arch>.block_count`.
    pub block_count: u32,
    /// `<arch>.embedding_length`.
    pub embedding_length: u32,
    /// `<arch>.feed_forward_length`.
    pub feed_forward_length: u32,
    /// `<arch>.attention.head_count`.
    pub head_count: u32,
    /// `<arch>.attention.head_count_kv`.
    pub head_count_kv: u32,
    /// `<arch>.attention.key_length` (head dim), when decoupled from
    /// `embedding_length / head_count`.
    pub key_length: Option<u32>,
    /// `<arch>.attention.value_length`, when it differs from `key_length`.
    pub value_length: Option<u32>,
    /// `<arch>.context_length`.
    pub context_length: u32,
    /// `<arch>.vocab_size`.
    pub vocab_size: u32,
    /// `<arch>.attention.layer_norm_rms_epsilon`.
    pub rms_norm_eps: f32,
    /// `<arch>.rope.freq_base`.
    pub rope_freq_base: f32,
    /// `<arch>.rope.dimension_count` (`n_rot`), when partial RoPE is used.
    pub rope_dimension_count: Option<u32>,
}

impl ArchMetadata {
    /// Read the architecture block out of a HuggingFace `config.json`.
    ///
    /// Every field the reader needs is **required**: silently defaulting a
    /// missing `hidden_size` is how a converted file ends up describing a
    /// different model than it contains. `head_dim`, `rope_theta` and the
    /// RoPE dimension count stay optional because real Qwen3 configs legitimately
    /// omit them.
    pub fn from_config_json(config: &Value) -> Result<Self, MetaError> {
        let req = |json_key: &str| -> Result<u32, MetaError> {
            config
                .get(json_key)
                .and_then(Value::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| MetaError::MissingConfigKey {
                    key: json_key.to_string(),
                })
        };

        let embedding_length = req("hidden_size")?;
        let head_count = req("num_attention_heads")?;

        Ok(Self {
            block_count: req("num_hidden_layers")?,
            embedding_length,
            feed_forward_length: req("intermediate_size")?,
            head_count,
            head_count_kv: req("num_key_value_heads")?,
            key_length: config
                .get("head_dim")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            value_length: None,
            context_length: req("max_position_embeddings")?,
            vocab_size: req("vocab_size")?,
            rms_norm_eps: config
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            rope_freq_base: resolve_rope_theta(config) as f32,
            rope_dimension_count: None,
        })
    }
}

/// Errors raised while assembling GGUF metadata.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MetaError {
    /// A required `config.json` field is absent or not an integer.
    #[error(
        "config.json is missing required field '{key}' — refusing to write a GGUF whose \
         architecture block would describe a different model than it contains"
    )]
    MissingConfigKey {
        /// The `config.json` key that was absent.
        key: String,
    },
}

/// Write the `<arch>.*` architecture block.
pub fn write_arch_metadata(writer: &mut GgufWriter<'_>, arch: &str, meta: &ArchMetadata) {
    let mut u32_key = |suffix: &str, value: u32| {
        writer.add_metadata(&arch_key(arch, suffix), MetadataWriteValue::U32(value));
    };
    u32_key("block_count", meta.block_count);
    u32_key("embedding_length", meta.embedding_length);
    u32_key("feed_forward_length", meta.feed_forward_length);
    u32_key("attention.head_count", meta.head_count);
    u32_key("attention.head_count_kv", meta.head_count_kv);
    u32_key("context_length", meta.context_length);
    u32_key("vocab_size", meta.vocab_size);
    if let Some(k) = meta.key_length {
        u32_key("attention.key_length", k);
    }
    if let Some(v) = meta.value_length {
        u32_key("attention.value_length", v);
    }
    if let Some(n) = meta.rope_dimension_count {
        u32_key("rope.dimension_count", n);
    }
    writer.add_metadata(
        &arch_key(arch, "attention.layer_norm_rms_epsilon"),
        MetadataWriteValue::F32(meta.rms_norm_eps),
    );
    writer.add_metadata(
        &arch_key(arch, "rope.freq_base"),
        MetadataWriteValue::F32(meta.rope_freq_base),
    );
}

/// The exact set of keys [`write_arch_metadata`] will write for `(arch, meta)`.
///
/// Kept in lock-step with `write_arch_metadata` by construction: every
/// suffix listed there is listed here. Used to de-duplicate metadata carried
/// from a source file against the keys this export's own `<arch>.*` block is
/// about to write a second time (see [`filter_carried_metadata`] /
/// CQ-01's duplicate-key regression).
///
/// This has to be an **exact** key set, not a namespace prefix: when
/// `arch == "qwen35"`, [`crate::convert::qwen35::write_qwen35_metadata`]
/// writes a *different* set of keys under that same `"qwen35."` prefix, so
/// dropping the whole prefix whenever only the hybrid writer runs would
/// delete this block's keys too, with nothing left to replace them.
pub fn arch_metadata_keys(arch: &str, meta: &ArchMetadata) -> BTreeSet<String> {
    let mut keys: BTreeSet<String> = [
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "context_length",
        "vocab_size",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
    ]
    .iter()
    .map(|suffix| arch_key(arch, suffix))
    .collect();
    if meta.key_length.is_some() {
        keys.insert(arch_key(arch, "attention.key_length"));
    }
    if meta.value_length.is_some() {
        keys.insert(arch_key(arch, "attention.value_length"));
    }
    if meta.rope_dimension_count.is_some() {
        keys.insert(arch_key(arch, "rope.dimension_count"));
    }
    keys
}

/// Drop every carried metadata entry that one of this export's explicit
/// blocks is about to write a second time.
///
/// `exact_exclude` names individual keys — used for `<arch>.*` / `qwen35.*`,
/// whose two writers can share one literal namespace (see
/// [`arch_metadata_keys`]). `prefix_exclude` names whole namespaces that
/// exactly one writer ever owns (`"tokenizer."`, `"prism.hadamard."`), so a
/// simple prefix match is safe there.
///
/// # Why this exists (CQ-01 duplicate-key regression)
///
/// [`crate::export::ExportConfig::with_source_metadata`] carries every key a
/// source GGUF has that the writer does not unconditionally own, which
/// includes the *entire* `<arch>.*`, `tokenizer.ggml.*`, `qwen35.*` and
/// `prism.hadamard.*` blocks. When the caller *also* attaches an explicit
/// block for one of those namespaces (`with_architecture`, `with_tokenizer`,
/// `with_qwen35`, `with_hadamard` — in any call order, since the builder
/// methods commute), the explicit block writes the same keys a second time,
/// and the GGUF reader rejects a duplicate key outright. Filtering has to
/// run once, after every builder call has run — which is what calling this
/// from the writer rather than from `with_source_metadata` itself achieves —
/// precisely because the call order is not fixed.
pub fn filter_carried_metadata(
    carried: &[(String, MetadataWriteValue)],
    exact_exclude: &BTreeSet<String>,
    prefix_exclude: &[&str],
) -> Vec<(String, MetadataWriteValue)> {
    carried
        .iter()
        .filter(|(key, _)| {
            !exact_exclude.contains(key) && !prefix_exclude.iter().any(|p| key.starts_with(p))
        })
        .cloned()
        .collect()
}

/// Resolve `rope_theta` from a HuggingFace `config.json`.
///
/// Looks in this order:
///   1. Top-level `rope_theta` (legacy Qwen2 layout).
///   2. Nested `rope_parameters.rope_theta` (Qwen3 ONNX / newer layout).
///   3. Fallback `10000.0` with a `tracing::warn!`.
pub fn resolve_rope_theta(config: &Value) -> f64 {
    if let Some(v) = config.get("rope_theta").and_then(Value::as_f64) {
        return v;
    }
    if let Some(v) = config
        .get("rope_parameters")
        .and_then(|rp| rp.get("rope_theta"))
        .and_then(Value::as_f64)
    {
        return v;
    }
    tracing::warn!(
        "config.json missing both `rope_theta` and `rope_parameters.rope_theta`; \
         falling back to default 10000.0"
    );
    10000.0
}

/// Read the model architecture name out of `config.json`.
///
/// Uses `model_type` (`"qwen3"`, `"qwen3_next"`, …) mapped to the GGUF
/// architecture spelling, falling back to `"qwen3"` for the Qwen3 family that
/// every existing converter target belongs to.
pub fn architecture_from_config_json(config: &Value) -> String {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match model_type {
        "qwen35" | "qwen3_5" | "qwen3_next" => "qwen35".to_string(),
        "" => "qwen3".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::reader::GgufFile;
    use serde_json::json;

    fn minimal_config() -> Value {
        json!({
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 512,
            "vocab_size": 1024,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
        })
    }

    #[test]
    fn arch_key_composes_namespaced_keys() {
        assert_eq!(arch_key("qwen3", "block_count"), "qwen3.block_count");
        assert_eq!(
            arch_key("qwen35", "ssm.state_size"),
            "qwen35.ssm.state_size"
        );
    }

    #[test]
    fn file_type_matches_shipped_27b_headers() {
        // Observed in scratchpad/gguf_headers_summary.txt.
        assert_eq!(ggml_file_type(TensorType::PTQ1_0), 143);
        assert_eq!(ggml_file_type(TensorType::PQ2_0), 141);
        assert_eq!(ggml_file_type(TensorType::TQ2_0_g128), 41);
        assert_eq!(ggml_file_type(TensorType::Q1_0G128), 40);
        assert_eq!(ggml_file_type(TensorType::Q8_0), 7);
        assert_eq!(ggml_file_type(TensorType::F32), 0);
    }

    /// FIX3-GGUF-WRITE item 1(b): `ggml_file_type` is an exhaustive match
    /// with no `_` arm, so adding `TensorType::Q2_K`/`Q3_K`/`Q8_K` (item
    /// 1(a)) must give every one of them a real value rather than leave the
    /// match non-exhaustive.
    #[test]
    fn file_type_covers_the_three_new_k_quants() {
        assert_eq!(ggml_file_type(TensorType::Q2_K), 10); // MOSTLY_Q2_K
        assert_eq!(ggml_file_type(TensorType::Q3_K), 12); // MOSTLY_Q3_K_M
                                                          // Q8_K has no upstream `llama_ftype`; falls back to the wire id (15).
        assert_eq!(ggml_file_type(TensorType::Q8_K), TensorType::Q8_K.wire_id());
    }

    /// FIX3-GGUF-WRITE item 3 / B2-16 handover 2: a negative `i64` metadata
    /// value must survive a **full export round trip**, not just the
    /// writer-level encode/decode `i64_metadata_value_roundtrips_a_negative_number`
    /// already covers in `gguf/writer.rs`.
    ///
    /// Before the fix, `convert_metadata_value` (`export.rs`) mapped a
    /// source `MetadataValue::Int64` to
    /// `MetadataWriteValue::U64(u64::try_from(*v).ok()?)`. For a negative
    /// value `u64::try_from` fails, so `.ok()?` short-circuits the whole
    /// match arm to `None` — `with_source_metadata`'s caller then treats
    /// that as "no writable representation" and silently **drops the key**
    /// (tracing::warn + omit), not the corrupted-huge-`u64` a naive read of
    /// the bug might expect. This test's regression is exactly that: the
    /// key must still be *present*, with its *sign* intact, after the fix.
    #[test]
    fn negative_i64_metadata_survives_a_full_export_round_trip() {
        use oxibonsai_core::gguf::reader::GgufFile;
        use oxibonsai_core::gguf::writer::GgufWriter;

        use crate::export::{export_to_gguf, ExportConfig, ExportFormat, WeightTensor};

        // A tiny source GGUF: just `general.architecture` (required by
        // `with_source_metadata`, else it refuses with `MissingArchitecture`)
        // plus one custom negative `i64` key standing in for a real one
        // (e.g. a signed hyperparameter a future architecture might carry).
        let mut source_writer = GgufWriter::new();
        source_writer.add_metadata(
            keys::GENERAL_ARCHITECTURE,
            MetadataWriteValue::Str("qwen3".to_string()),
        );
        source_writer.add_metadata("oxibonsai_test.negative_i64", MetadataWriteValue::I64(-5));
        source_writer.add_metadata(
            "oxibonsai_test.negative_i64_min",
            MetadataWriteValue::I64(i64::MIN),
        );
        let source_bytes = source_writer
            .to_bytes()
            .expect("write source metadata-only gguf");
        let source_gguf = GgufFile::parse(&source_bytes).expect("parse source gguf");

        // Carry the source metadata into a real export, exactly the path
        // `oxibonsai quantize` (`cmd_quantize::run`) drives.
        let tensors = vec![WeightTensor::new(
            "blk.0.attn_q.weight",
            vec![1.0; 4],
            vec![4],
        )];
        let config = ExportConfig::new(ExportFormat::Float32, "i64-export-regression")
            .with_source_metadata(&source_gguf.metadata)
            .expect("with_source_metadata must accept a source carrying general.architecture");
        let exported_bytes = export_to_gguf(&tensors, &config, &[]).expect("export_to_gguf");
        let exported_gguf = GgufFile::parse(&exported_bytes).expect("parse exported gguf");

        let value = exported_gguf
            .metadata
            .get("oxibonsai_test.negative_i64")
            .expect(
                "negative i64 key must survive export, not be silently dropped as \
                 \"no writable representation\"",
            );
        assert_eq!(value.type_name(), "Int64");
        assert_eq!(value.as_i64(), Some(-5));

        let min_value = exported_gguf
            .metadata
            .get("oxibonsai_test.negative_i64_min")
            .expect("i64::MIN key must also survive export");
        assert_eq!(min_value.as_i64(), Some(i64::MIN));
    }

    #[test]
    fn quantization_version_is_u32_not_string() {
        let mut w = GgufWriter::new();
        let general = GeneralMetadata::new("qwen3", "unit-test", TensorType::PQ2_0);
        write_general_metadata(&mut w, &general);
        let bytes = w.to_bytes().expect("serialise metadata-only file");
        let gguf = GgufFile::parse(&bytes).expect("parse");

        // The whole point of core-gguf-02: `get_u32` must succeed.
        assert_eq!(
            gguf.metadata
                .get_u32(keys::GENERAL_QUANTIZATION_VERSION)
                .expect("quantization_version must be a u32"),
            2
        );
        assert_eq!(
            gguf.metadata
                .get_u32(keys::GENERAL_FILE_TYPE)
                .expect("file_type must be a u32"),
            141
        );
        assert_eq!(
            gguf.metadata.get_string(keys::GENERAL_ARCHITECTURE).ok(),
            Some("qwen3")
        );
    }

    #[test]
    fn arch_block_uses_architecture_prefix_not_llm() {
        let cfg = minimal_config();
        let meta = ArchMetadata::from_config_json(&cfg).expect("arch metadata");
        let mut w = GgufWriter::new();
        write_general_metadata(
            &mut w,
            &GeneralMetadata::new("qwen3", "unit-test", TensorType::F32),
        );
        write_arch_metadata(&mut w, "qwen3", &meta);
        let bytes = w.to_bytes().expect("serialise");
        let gguf = GgufFile::parse(&bytes).expect("parse");

        assert_eq!(gguf.metadata.get_u32("qwen3.block_count").ok(), Some(2));
        assert_eq!(
            gguf.metadata.get_u32("qwen3.embedding_length").ok(),
            Some(128)
        );
        assert!(
            gguf.metadata.get_u32("llm.block_count").is_err(),
            "the non-existent llm.* namespace must not be written"
        );

        // And the whole point: the real config reader now sees real numbers.
        let cfg = oxibonsai_core::Qwen3Config::from_metadata(&gguf.metadata).expect("config");
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.hidden_size, 128);
        assert_eq!(cfg.intermediate_size, 256);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_kv_heads, 2);
        assert_eq!(cfg.vocab_size, 1024);
        assert_eq!(cfg.max_context_length, 512);
        assert!((cfg.rope_freq_base - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn missing_required_config_key_errors() {
        let mut cfg = minimal_config();
        cfg.as_object_mut()
            .expect("object")
            .remove("num_key_value_heads");
        let err = ArchMetadata::from_config_json(&cfg).expect_err("must refuse");
        assert_eq!(
            err,
            MetaError::MissingConfigKey {
                key: "num_key_value_heads".to_string()
            }
        );
    }

    #[test]
    fn rope_theta_resolution_order() {
        assert_eq!(
            resolve_rope_theta(&json!({"rope_theta": 500000.0})),
            500000.0
        );
        assert_eq!(
            resolve_rope_theta(&json!({"rope_parameters": {"rope_theta": 1e6}})),
            1e6
        );
        assert_eq!(
            resolve_rope_theta(
                &json!({"rope_theta": 250000.0, "rope_parameters": {"rope_theta": 1e6}})
            ),
            250000.0
        );
        assert_eq!(resolve_rope_theta(&json!({"hidden_size": 2048})), 10000.0);
    }

    #[test]
    fn architecture_detection() {
        assert_eq!(architecture_from_config_json(&json!({})), "qwen3");
        assert_eq!(
            architecture_from_config_json(&json!({"model_type": "qwen3"})),
            "qwen3"
        );
        assert_eq!(
            architecture_from_config_json(&json!({"model_type": "qwen3_next"})),
            "qwen35"
        );
    }

    #[test]
    fn arch_metadata_keys_includes_optional_fields_only_when_present() {
        let cfg = minimal_config();
        let meta = ArchMetadata::from_config_json(&cfg).expect("arch metadata");
        let keys = arch_metadata_keys("qwen3", &meta);
        assert!(keys.contains("qwen3.block_count"));
        assert!(keys.contains("qwen3.attention.layer_norm_rms_epsilon"));
        assert!(keys.contains("qwen3.rope.freq_base"));
        // `minimal_config` sets none of the optional fields.
        assert!(!keys.contains("qwen3.attention.key_length"));
        assert!(!keys.contains("qwen3.attention.value_length"));
        assert!(!keys.contains("qwen3.rope.dimension_count"));

        let mut with_optional = meta;
        with_optional.key_length = Some(64);
        let keys = arch_metadata_keys("qwen3", &with_optional);
        assert!(keys.contains("qwen3.attention.key_length"));
    }

    #[test]
    fn filter_carried_metadata_uses_exact_match_for_the_exclude_set() {
        let carried = vec![
            (
                "qwen35.block_count".to_string(),
                MetadataWriteValue::U32(64),
            ),
            (
                "qwen35.full_attention_interval".to_string(),
                MetadataWriteValue::U32(4),
            ),
            (
                "tokenizer.ggml.model".to_string(),
                MetadataWriteValue::Str("gpt2".to_string()),
            ),
            (
                "general.basename".to_string(),
                MetadataWriteValue::Str("folded".to_string()),
            ),
        ];
        let mut exact_exclude = BTreeSet::new();
        exact_exclude.insert("qwen35.full_attention_interval".to_string());
        let filtered = filter_carried_metadata(&carried, &exact_exclude, &["tokenizer."]);
        let kept: Vec<&str> = filtered.iter().map(|(k, _)| k.as_str()).collect();

        // Exact-excluded key is gone…
        assert!(!kept.contains(&"qwen35.full_attention_interval"));
        // …but a *different* key sharing the same "qwen35." prefix survives:
        // this is the whole point of exact matching over prefix matching.
        assert!(kept.contains(&"qwen35.block_count"));
        // Prefix-excluded namespace is gone.
        assert!(!kept.contains(&"tokenizer.ggml.model"));
        // Untouched key survives.
        assert!(kept.contains(&"general.basename"));
    }
}
