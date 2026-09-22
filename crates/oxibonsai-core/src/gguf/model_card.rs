//! GGUF model card: extract and render model information from GGUF metadata.
//!
//! This module provides structured extraction of well-known GGUF metadata fields
//! and renders them as a human-readable markdown model card or plain text summary.

use std::collections::HashMap;

use crate::gguf::metadata::{MetadataStore, MetadataValue};
use crate::gguf::tensor_info::TensorStore;

// ── Well-known GGUF metadata key names ──────────────────────────────────────

/// Well-known GGUF metadata key names.
///
/// # A note on the `llm.*`-prefixed constants (core-gguf-13)
///
/// `CONTEXT_LENGTH`, `EMBEDDING_LENGTH`, `NUM_LAYERS`, `NUM_HEADS`,
/// `NUM_KV_HEADS` and `ROPE_FREQ_BASE` below spell a literal `"llm."`
/// prefix. **No real GGUF file has ever used that prefix** — the GGUF spec
/// scopes these keys under the model's own `general.architecture` value
/// (`qwen35.context_length`, `llama.context_length`, ...), never a generic
/// `llm.` namespace. `VOCAB_SIZE` similarly names a key
/// (`tokenizer.ggml.tokens_count`) that appears in no real GGUF file either
/// — the vocabulary size is the length of the `tokenizer.ggml.tokens`
/// array. These constants are kept, unchanged, purely because
/// [`extract_model_card`] and `tests/model_card_tests.rs` (outside this
/// package's owned files) are built around this flat, pre-stringified
/// `HashMap<String, String>` API and its `llm.*`/`tokens_count` keys.
///
/// [`extract_model_card_from_gguf`] is the corrected extraction path: it
/// reads a real [`MetadataStore`] and re-keys on `general.*` plus the
/// file's actual `<architecture>.*` namespace (with an `llm.*` fallback
/// for any tooling that predates this convention), and derives
/// `vocab_size` from the tokens array's length instead of the
/// nonexistent `tokens_count` key.
pub mod keys {
    pub const MODEL_NAME: &str = "general.name";
    pub const ARCHITECTURE: &str = "general.architecture";
    pub const AUTHOR: &str = "general.author";
    pub const LICENSE: &str = "general.license";
    pub const DESCRIPTION: &str = "general.description";
    pub const CONTEXT_LENGTH: &str = "llm.context_length";
    pub const EMBEDDING_LENGTH: &str = "llm.embedding_length";
    pub const NUM_LAYERS: &str = "llm.block_count";
    pub const NUM_HEADS: &str = "llm.attention.head_count";
    pub const NUM_KV_HEADS: &str = "llm.attention.head_count_kv";
    pub const ROPE_FREQ_BASE: &str = "llm.rope.freq_base";
    pub const VOCAB_SIZE: &str = "tokenizer.ggml.tokens_count";
    pub const QUANTIZATION: &str = "general.quantization_version";
    pub const FILE_SIZE: &str = "general.file_size";
    pub const PARAMETER_COUNT: &str = "general.parameter_count";

    // ── Keys used by `extract_model_card_from_gguf` only ───────────────────

    /// `tokenizer.ggml.tokens` — the real vocabulary array; its length is
    /// the true vocab size (no real file has a `tokens_count` scalar key).
    pub const TOKENIZER_TOKENS: &str = "tokenizer.ggml.tokens";
    /// `token_embd.weight`'s tensor name — a secondary vocab-size fallback
    /// (its last shape dimension) for the rare file whose metadata omits
    /// the tokens array entirely.
    pub const TOKEN_EMBD_TENSOR: &str = "token_embd.weight";
    /// `prism.hadamard.version` — present on PrismML Bonsai 2-family models
    /// that fold a Hadamard rotation into their quantized weights.
    pub const HADAMARD_VERSION: &str = "prism.hadamard.version";
    /// `general.sampling.temp` — the model author's recommended sampling
    /// temperature.
    pub const SAMPLING_TEMP: &str = "general.sampling.temp";
    /// `general.sampling.top_p`.
    pub const SAMPLING_TOP_P: &str = "general.sampling.top_p";
    /// `general.sampling.top_k`.
    pub const SAMPLING_TOP_K: &str = "general.sampling.top_k";

    /// Architecture-scoped key suffixes: the real on-disk key is
    /// `"{architecture}.{suffix}"` (e.g. `qwen35.context_length`), where
    /// `architecture` is `general.architecture`'s own value — never a
    /// literal `"llm."` prefix.
    pub mod arch_suffix {
        pub const CONTEXT_LENGTH: &str = "context_length";
        pub const EMBEDDING_LENGTH: &str = "embedding_length";
        pub const BLOCK_COUNT: &str = "block_count";
        pub const HEAD_COUNT: &str = "attention.head_count";
        pub const HEAD_COUNT_KV: &str = "attention.head_count_kv";
        pub const ROPE_FREQ_BASE: &str = "rope.freq_base";
        /// Hybrid full-attention/recurrent architectures (e.g. `qwen35`):
        /// every Nth layer is full attention, the rest are linear/recurrent.
        pub const FULL_ATTENTION_INTERVAL: &str = "full_attention_interval";
    }

    /// The legacy generic prefix ([`super::keys`]'s own `CONTEXT_LENGTH`
    /// etc. above) — tried only when the architecture-scoped key is absent.
    pub const LEGACY_PREFIX: &str = "llm";
}

// ── Markdown rendering helpers ───────────────────────────────────────────────

/// Markdown rendering helpers.
mod render {
    /// Render a markdown heading at the given level (1–6).
    pub fn heading(level: u8, text: &str) -> String {
        let level = level.clamp(1, 6);
        let hashes = "#".repeat(level as usize);
        format!("{hashes} {text}\n")
    }

    /// Render a `**label**: value` field line.
    pub fn field(label: &str, value: &str) -> String {
        format!("- **{label}**: {value}\n")
    }

    /// Render a markdown table row with pipe-separated cells.
    pub fn table_row(cells: &[&str]) -> String {
        let inner = cells.join(" | ");
        format!("| {inner} |\n")
    }

    /// Wrap text in backticks for inline code.
    pub fn code(text: &str) -> String {
        format!("`{text}`")
    }

    /// Wrap text in double-asterisks for bold.
    #[allow(dead_code)]
    pub fn bold(text: &str) -> String {
        format!("**{text}**")
    }
}

// ── ModelCard ────────────────────────────────────────────────────────────────

/// Structured information extracted from GGUF metadata.
///
/// All fields are optional — a file may not include every piece of metadata.
/// Use [`extract_model_card`] to populate this from a raw metadata map.
#[derive(Debug, Clone, Default)]
pub struct ModelCard {
    /// Human-readable model name (e.g. `"Llama-3-8B"`).
    pub model_name: Option<String>,
    /// Architecture identifier (e.g. `"llama"`, `"qwen3"`).
    pub architecture: Option<String>,
    /// Author or organisation that produced the model.
    pub author: Option<String>,
    /// SPDX license identifier or URL (e.g. `"apache-2.0"`).
    pub license: Option<String>,
    /// Free-text description of the model.
    pub description: Option<String>,
    /// Maximum context window in tokens.
    pub context_length: Option<u64>,
    /// Hidden-state / embedding dimension.
    pub embedding_length: Option<u64>,
    /// Number of transformer blocks (layers).
    pub num_layers: Option<u64>,
    /// Number of attention heads.
    pub num_heads: Option<u64>,
    /// Number of key-value heads (GQA).
    pub num_kv_heads: Option<u64>,
    /// RoPE frequency base.
    pub rope_freq_base: Option<f64>,
    /// Vocabulary size.
    pub vocab_size: Option<u64>,
    /// Parameter count expressed in billions (e.g. `7.0` for a 7B model).
    pub parameter_count_billions: Option<f64>,
    /// Quantization scheme string (e.g. `"Q4_K_M"`).
    pub quantization: Option<String>,
    /// Size of the GGUF file on disk, in bytes.
    pub file_size_bytes: Option<u64>,
    /// `prism.hadamard.version`, when the file carries a PrismML Hadamard
    /// rotation contract (Bonsai 2 / `qwen35`-family models). Populated
    /// only by [`extract_model_card_from_gguf`].
    pub hadamard_version: Option<u64>,
    /// `<architecture>.full_attention_interval`, for a hybrid
    /// attention/recurrent architecture. Populated only by
    /// [`extract_model_card_from_gguf`].
    pub full_attention_interval: Option<u64>,
    /// `general.sampling.temp` — the model author's recommended sampling
    /// temperature. Populated only by [`extract_model_card_from_gguf`].
    pub sampling_temp: Option<f64>,
    /// `general.sampling.top_p`. Populated only by
    /// [`extract_model_card_from_gguf`].
    pub sampling_top_p: Option<f64>,
    /// `general.sampling.top_k`. Populated only by
    /// [`extract_model_card_from_gguf`].
    pub sampling_top_k: Option<u64>,
    /// Any metadata key-value pairs not covered by the typed fields above.
    pub extra_metadata: HashMap<String, String>,
}

impl ModelCard {
    /// Create an empty `ModelCard`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the card as a Markdown document.
    ///
    /// The output always begins with a level-1 heading so that the caller can
    /// detect a non-trivial render with `starts_with("# ")`.
    pub fn to_markdown(&self) -> String {
        let mut out = String::with_capacity(1024);

        // Title
        let title = self.model_name.as_deref().unwrap_or("Unknown Model");
        out.push_str(&render::heading(1, title));
        out.push('\n');

        // ── Model Information ──
        out.push_str(&render::heading(2, "Model Information"));
        if let Some(ref v) = self.architecture {
            out.push_str(&render::field("Architecture", v));
        }
        if let Some(ref v) = self.author {
            out.push_str(&render::field("Author", v));
        }
        if let Some(ref v) = self.license {
            out.push_str(&render::field("License", v));
        }
        if let Some(ref v) = self.description {
            out.push_str(&render::field("Description", v));
        }
        if let Some(ref v) = self.quantization {
            out.push_str(&render::field("Quantization", &render::code(v)));
        }
        if let Some(v) = self.file_size_bytes {
            let gb = v as f64 / (1024.0 * 1024.0 * 1024.0);
            out.push_str(&render::field(
                "File Size",
                &format!("{v} bytes ({gb:.2} GB)"),
            ));
        }
        out.push('\n');

        // ── Architecture Details ──
        out.push_str(&render::heading(2, "Architecture Details"));

        // Table header
        out.push_str(&render::table_row(&["Parameter", "Value"]));
        out.push_str(&render::table_row(&["---", "---"]));

        let param_count_str;
        let param_billions = self
            .parameter_count_billions
            .or_else(|| self.estimated_param_count());
        if let Some(b) = param_billions {
            param_count_str = format!("{b:.2}B");
            out.push_str(&render::table_row(&["Parameter Count", &param_count_str]));
        }

        let ctx_str;
        if let Some(v) = self.context_length {
            ctx_str = v.to_string();
            out.push_str(&render::table_row(&["Context Length", &ctx_str]));
        }

        let embed_str;
        if let Some(v) = self.embedding_length {
            embed_str = v.to_string();
            out.push_str(&render::table_row(&["Embedding Length", &embed_str]));
        }

        let layers_str;
        if let Some(v) = self.num_layers {
            layers_str = v.to_string();
            out.push_str(&render::table_row(&["Layers", &layers_str]));
        }

        let heads_str;
        if let Some(v) = self.num_heads {
            heads_str = v.to_string();
            out.push_str(&render::table_row(&["Attention Heads", &heads_str]));
        }

        let kv_heads_str;
        if let Some(v) = self.num_kv_heads {
            kv_heads_str = v.to_string();
            out.push_str(&render::table_row(&["KV Heads", &kv_heads_str]));
        }

        let rope_str;
        if let Some(v) = self.rope_freq_base {
            rope_str = format!("{v:.1}");
            out.push_str(&render::table_row(&["RoPE Freq Base", &rope_str]));
        }

        let vocab_str;
        if let Some(v) = self.vocab_size {
            vocab_str = v.to_string();
            out.push_str(&render::table_row(&["Vocab Size", &vocab_str]));
        }

        let interval_str;
        if let Some(v) = self.full_attention_interval {
            interval_str = v.to_string();
            out.push_str(&render::table_row(&[
                "Full Attention Interval",
                &interval_str,
            ]));
        }

        let hadamard_str;
        if let Some(v) = self.hadamard_version {
            hadamard_str = v.to_string();
            out.push_str(&render::table_row(&["Hadamard Version", &hadamard_str]));
        }

        out.push('\n');

        // ── Sampling Defaults ──
        if self.sampling_temp.is_some()
            || self.sampling_top_p.is_some()
            || self.sampling_top_k.is_some()
        {
            out.push_str(&render::heading(2, "Sampling Defaults"));
            if let Some(v) = self.sampling_temp {
                out.push_str(&render::field("Temperature", &format!("{v:.2}")));
            }
            if let Some(v) = self.sampling_top_p {
                out.push_str(&render::field("Top-p", &format!("{v:.2}")));
            }
            if let Some(v) = self.sampling_top_k {
                out.push_str(&render::field("Top-k", &v.to_string()));
            }
            out.push('\n');
        }

        // ── Extra Metadata ──
        if !self.extra_metadata.is_empty() {
            out.push_str(&render::heading(2, "Additional Metadata"));
            let mut sorted: Vec<(&String, &String)> = self.extra_metadata.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in sorted {
                out.push_str(&render::field(k, v));
            }
            out.push('\n');
        }

        out
    }

    /// Render a compact plain-text summary — one line per populated field.
    pub fn to_summary(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        macro_rules! push_opt_str {
            ($label:expr, $field:expr) => {
                if let Some(ref v) = $field {
                    lines.push(format!("{}: {}", $label, v));
                }
            };
        }
        macro_rules! push_opt_num {
            ($label:expr, $field:expr) => {
                if let Some(v) = $field {
                    lines.push(format!("{}: {}", $label, v));
                }
            };
        }

        push_opt_str!("Model", self.model_name);
        push_opt_str!("Architecture", self.architecture);
        push_opt_str!("Author", self.author);
        push_opt_str!("License", self.license);
        push_opt_str!("Description", self.description);
        push_opt_str!("Quantization", self.quantization);
        push_opt_num!("Context Length", self.context_length);
        push_opt_num!("Embedding Length", self.embedding_length);
        push_opt_num!("Layers", self.num_layers);
        push_opt_num!("Attention Heads", self.num_heads);
        push_opt_num!("KV Heads", self.num_kv_heads);

        if let Some(v) = self.rope_freq_base {
            lines.push(format!("RoPE Freq Base: {v:.1}"));
        }
        push_opt_num!("Vocab Size", self.vocab_size);

        let param_billions = self
            .parameter_count_billions
            .or_else(|| self.estimated_param_count());
        if let Some(b) = param_billions {
            lines.push(format!("Parameter Count: {b:.2}B"));
        }

        if let Some(v) = self.file_size_bytes {
            lines.push(format!("File Size: {v} bytes"));
        }

        push_opt_num!("Full Attention Interval", self.full_attention_interval);
        push_opt_num!("Hadamard Version", self.hadamard_version);
        push_opt_num!("Sampling Temp", self.sampling_temp);
        push_opt_num!("Sampling Top-p", self.sampling_top_p);
        push_opt_num!("Sampling Top-k", self.sampling_top_k);

        if lines.is_empty() {
            lines.push("(no metadata available)".to_owned());
        }

        lines.join("\n")
    }

    /// Returns `true` if no structured fields and no extra metadata are set.
    pub fn is_empty(&self) -> bool {
        self.model_name.is_none()
            && self.architecture.is_none()
            && self.author.is_none()
            && self.license.is_none()
            && self.description.is_none()
            && self.context_length.is_none()
            && self.embedding_length.is_none()
            && self.num_layers.is_none()
            && self.num_heads.is_none()
            && self.num_kv_heads.is_none()
            && self.rope_freq_base.is_none()
            && self.vocab_size.is_none()
            && self.parameter_count_billions.is_none()
            && self.quantization.is_none()
            && self.file_size_bytes.is_none()
            && self.hadamard_version.is_none()
            && self.full_attention_interval.is_none()
            && self.sampling_temp.is_none()
            && self.sampling_top_p.is_none()
            && self.sampling_top_k.is_none()
            && self.extra_metadata.is_empty()
    }

    /// Returns the number of typed (non-`extra_metadata`) fields that are `Some`.
    pub fn populated_count(&self) -> usize {
        let mut count = 0usize;
        if self.model_name.is_some() {
            count += 1;
        }
        if self.architecture.is_some() {
            count += 1;
        }
        if self.author.is_some() {
            count += 1;
        }
        if self.license.is_some() {
            count += 1;
        }
        if self.description.is_some() {
            count += 1;
        }
        if self.context_length.is_some() {
            count += 1;
        }
        if self.embedding_length.is_some() {
            count += 1;
        }
        if self.num_layers.is_some() {
            count += 1;
        }
        if self.num_heads.is_some() {
            count += 1;
        }
        if self.num_kv_heads.is_some() {
            count += 1;
        }
        if self.rope_freq_base.is_some() {
            count += 1;
        }
        if self.vocab_size.is_some() {
            count += 1;
        }
        if self.parameter_count_billions.is_some() {
            count += 1;
        }
        if self.quantization.is_some() {
            count += 1;
        }
        if self.file_size_bytes.is_some() {
            count += 1;
        }
        if self.hadamard_version.is_some() {
            count += 1;
        }
        if self.full_attention_interval.is_some() {
            count += 1;
        }
        if self.sampling_temp.is_some() {
            count += 1;
        }
        if self.sampling_top_p.is_some() {
            count += 1;
        }
        if self.sampling_top_k.is_some() {
            count += 1;
        }
        count
    }

    /// Estimate the parameter count (in billions) from known architecture dimensions
    /// when [`ModelCard::parameter_count_billions`] is not explicitly set.
    ///
    /// The approximation is based on the dominant transformer weight matrices:
    ///
    /// ```text
    /// Per-layer:
    ///   Q  projection: embed × (num_heads × head_dim) ≈ embed²
    ///   K  projection: embed × (kv_heads × head_dim)  ≈ embed × embed * kv_ratio
    ///   V  projection: same as K
    ///   O  projection: embed²
    ///   FFN (up + gate + down): 3 × embed × ffn_dim   ≈ 3 × embed × 2.67 × embed
    ///       (common ratio is ~8/3 ≈ 2.667)
    ///
    /// Embedding table: vocab_size × embed (shared with lm_head)
    /// ```
    ///
    /// Returns `None` when neither `embedding_length` nor `num_layers` is known.
    pub fn estimated_param_count(&self) -> Option<f64> {
        let embed = self.embedding_length? as f64;
        let layers = self.num_layers? as f64;

        // Attention parameters per layer.
        // Q: embed × embed (full)
        let q_params = embed * embed;
        // K+V: if kv_heads known, scale; otherwise assume MHA (same as Q)
        let kv_ratio = if let (Some(kv_h), Some(h)) = (self.num_kv_heads, self.num_heads) {
            if h > 0 {
                kv_h as f64 / h as f64
            } else {
                1.0
            }
        } else {
            1.0
        };
        let kv_params = 2.0 * embed * embed * kv_ratio;
        // O projection: embed × embed
        let o_params = embed * embed;

        // FFN parameters per layer (SwiGLU with 8/3 expansion).
        let ffn_dim = (embed * 8.0 / 3.0).ceil();
        let ffn_params = 3.0 * embed * ffn_dim;

        let per_layer = q_params + kv_params + o_params + ffn_params;
        let total_transformer = layers * per_layer;

        // Embedding table (vocab → embed); shared with lm_head → count once.
        let embed_table = self.vocab_size.unwrap_or(32_000) as f64 * embed;

        let total = total_transformer + embed_table;
        Some(total / 1e9)
    }
}

// ── Public extraction API ────────────────────────────────────────────────────

/// Extract a [`ModelCard`] from a flat `key → value` string metadata map.
///
/// Numeric fields are parsed from their string representations.  Unrecognised
/// keys that do not contain `"."` in the value are stored in
/// [`ModelCard::extra_metadata`] only when they are not already one of the
/// well-known keys processed into typed fields.
pub fn extract_model_card(metadata: &HashMap<String, String>) -> ModelCard {
    let mut card = ModelCard::new();

    // String fields
    card.model_name = metadata.get(keys::MODEL_NAME).cloned();
    card.architecture = metadata.get(keys::ARCHITECTURE).cloned();
    card.author = metadata.get(keys::AUTHOR).cloned();
    card.license = metadata.get(keys::LICENSE).cloned();
    card.description = metadata.get(keys::DESCRIPTION).cloned();
    card.quantization = metadata.get(keys::QUANTIZATION).cloned();

    // u64 fields
    card.context_length = parse_u64(metadata, keys::CONTEXT_LENGTH);
    card.embedding_length = parse_u64(metadata, keys::EMBEDDING_LENGTH);
    card.num_layers = parse_u64(metadata, keys::NUM_LAYERS);
    card.num_heads = parse_u64(metadata, keys::NUM_HEADS);
    card.num_kv_heads = parse_u64(metadata, keys::NUM_KV_HEADS);
    card.vocab_size = parse_u64(metadata, keys::VOCAB_SIZE);
    card.file_size_bytes = parse_u64(metadata, keys::FILE_SIZE);

    // f64 fields
    card.rope_freq_base = parse_f64(metadata, keys::ROPE_FREQ_BASE);

    // Parameter count may be stored as a raw integer (e.g. 7_000_000_000).
    if let Some(raw) = parse_u64(metadata, keys::PARAMETER_COUNT) {
        card.parameter_count_billions = Some(raw as f64 / 1e9);
    } else if let Some(raw) = parse_f64(metadata, keys::PARAMETER_COUNT) {
        // Already expressed as a decimal (rare, but possible).
        card.parameter_count_billions = Some(raw);
    }

    // Collect all unrecognised keys into extra_metadata.
    let known = known_key_set();
    for (k, v) in metadata {
        if !known.contains(k.as_str()) {
            card.extra_metadata.insert(k.clone(), v.clone());
        }
    }

    card
}

/// Return a map containing only the recognised GGUF metadata fields (those
/// whose keys appear in the [`keys`] module) that are present in `metadata`.
pub fn extract_known_fields(metadata: &HashMap<String, String>) -> HashMap<String, String> {
    let known = known_key_set();
    metadata
        .iter()
        .filter(|(k, _)| known.contains(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

// ── Public extraction API (real GGUF metadata) ───────────────────────────────

/// Extract a [`ModelCard`] directly from a parsed GGUF file's metadata and
/// tensor table (core-gguf-13).
///
/// This is the corrected extraction path. [`extract_model_card`] reads a
/// flat, already-stringified `key -> value` map keyed on a literal `"llm."`
/// prefix and a `tokenizer.ggml.tokens_count` scalar — neither of which any
/// real GGUF file has ever used (every architecture scopes its keys under
/// its own `general.architecture` value, and the vocabulary size is the
/// length of the `tokenizer.ggml.tokens` array, not a separate count key).
/// This function instead:
///
/// - re-keys on `general.*` plus the file's actual `<architecture>.*`
///   namespace, falling back to the legacy `llm.*` spelling only as a last
///   resort;
/// - derives `vocab_size` from `tokenizer.ggml.tokens`'s array length,
///   falling back to `token_embd.weight`'s last shape dimension for a file
///   whose metadata omits the tokens array;
/// - derives `quantization` from the tensor table's dominant quantization
///   type (e.g. `"PQ2_0"`), which is always present, rather than the
///   numeric `general.quantization_version` the old flat-map key actually
///   names;
/// - surfaces `prism.hadamard.version`, `<architecture>.full_attention_interval`
///   and `general.sampling.{temp,top_p,top_k}`, none of which the flat-map
///   API has fields for;
/// - populates `file_size_bytes` from `file_len` directly (there is no
///   real `general.file_size` metadata key to read it from).
///
/// `extra_metadata` collects every other **scalar** metadata entry
/// (arrays — `tokenizer.ggml.tokens`, `prism.hadamard.sign_values`, ... —
/// are deliberately excluded; a model card is not the place to dump a
/// quarter-million-entry token list).
pub fn extract_model_card_from_gguf(
    metadata: &MetadataStore,
    file_len: Option<u64>,
    tensors: &TensorStore,
) -> ModelCard {
    let mut card = ModelCard::new();

    card.model_name = metadata_str(metadata, keys::MODEL_NAME);
    let architecture = metadata_str(metadata, keys::ARCHITECTURE);
    card.author = metadata_str(metadata, keys::AUTHOR);
    card.license = metadata_str(metadata, keys::LICENSE);
    card.description = metadata_str(metadata, keys::DESCRIPTION);

    card.context_length = arch_or_legacy_u64(
        metadata,
        architecture.as_deref(),
        keys::arch_suffix::CONTEXT_LENGTH,
    );
    card.embedding_length = arch_or_legacy_u64(
        metadata,
        architecture.as_deref(),
        keys::arch_suffix::EMBEDDING_LENGTH,
    );
    card.num_layers = arch_or_legacy_u64(
        metadata,
        architecture.as_deref(),
        keys::arch_suffix::BLOCK_COUNT,
    );
    card.num_heads = arch_or_legacy_u64(
        metadata,
        architecture.as_deref(),
        keys::arch_suffix::HEAD_COUNT,
    );
    card.num_kv_heads = arch_or_legacy_u64(
        metadata,
        architecture.as_deref(),
        keys::arch_suffix::HEAD_COUNT_KV,
    );
    card.rope_freq_base = arch_or_legacy_f64(
        metadata,
        architecture.as_deref(),
        keys::arch_suffix::ROPE_FREQ_BASE,
    );
    card.full_attention_interval = arch_or_legacy_u64(
        metadata,
        architecture.as_deref(),
        keys::arch_suffix::FULL_ATTENTION_INTERVAL,
    );

    card.hadamard_version = metadata
        .get(keys::HADAMARD_VERSION)
        .and_then(MetadataValue::as_u64);
    card.sampling_temp = metadata
        .get(keys::SAMPLING_TEMP)
        .and_then(MetadataValue::as_f64);
    card.sampling_top_p = metadata
        .get(keys::SAMPLING_TOP_P)
        .and_then(MetadataValue::as_f64);
    card.sampling_top_k = metadata
        .get(keys::SAMPLING_TOP_K)
        .and_then(MetadataValue::as_u64);

    card.vocab_size = metadata
        .get_string_array(keys::TOKENIZER_TOKENS)
        .ok()
        .map(|tokens| tokens.len() as u64)
        .or_else(|| {
            tensors
                .get(keys::TOKEN_EMBD_TENSOR)
                .and_then(|info| info.shape.last().copied())
        });

    card.quantization = dominant_quant_type_name(tensors);
    card.file_size_bytes = file_len;
    card.architecture = architecture;

    let known = known_key_set_for_gguf(card.architecture.as_deref());
    for (k, v) in metadata.iter() {
        if known.contains(k.as_str()) {
            continue;
        }
        if let Some(s) = scalar_display_string(v) {
            card.extra_metadata.insert(k.clone(), s);
        }
    }

    card
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// The complete set of key strings defined in the [`keys`] module.
fn known_key_set() -> std::collections::HashSet<&'static str> {
    [
        keys::MODEL_NAME,
        keys::ARCHITECTURE,
        keys::AUTHOR,
        keys::LICENSE,
        keys::DESCRIPTION,
        keys::CONTEXT_LENGTH,
        keys::EMBEDDING_LENGTH,
        keys::NUM_LAYERS,
        keys::NUM_HEADS,
        keys::NUM_KV_HEADS,
        keys::ROPE_FREQ_BASE,
        keys::VOCAB_SIZE,
        keys::QUANTIZATION,
        keys::FILE_SIZE,
        keys::PARAMETER_COUNT,
    ]
    .into_iter()
    .collect()
}

/// Parse a `u64` from the given key in the metadata map.
fn parse_u64(metadata: &HashMap<String, String>, key: &str) -> Option<u64> {
    metadata.get(key)?.trim().parse::<u64>().ok()
}

/// Parse an `f64` from the given key in the metadata map.
fn parse_f64(metadata: &HashMap<String, String>, key: &str) -> Option<f64> {
    metadata.get(key)?.trim().parse::<f64>().ok()
}

// ── Internal helpers (`extract_model_card_from_gguf` only) ──────────────────

/// Build the real, on-disk key `"{architecture}.{suffix}"`.
fn arch_key(architecture: &str, suffix: &str) -> String {
    format!("{architecture}.{suffix}")
}

/// Build the legacy `"llm.{suffix}"` fallback key.
fn legacy_key(suffix: &str) -> String {
    format!("{}.{suffix}", keys::LEGACY_PREFIX)
}

/// Read a required-to-be-a-string metadata value.
fn metadata_str(metadata: &MetadataStore, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_str)
        .map(str::to_string)
}

/// Read a `u64` metadata value, trying `"{architecture}.{suffix}"` first
/// and falling back to the legacy `"llm.{suffix}"` spelling.
fn arch_or_legacy_u64(
    metadata: &MetadataStore,
    architecture: Option<&str>,
    suffix: &str,
) -> Option<u64> {
    if let Some(arch) = architecture {
        if let Some(v) = metadata
            .get(&arch_key(arch, suffix))
            .and_then(MetadataValue::as_u64)
        {
            return Some(v);
        }
    }
    metadata
        .get(&legacy_key(suffix))
        .and_then(MetadataValue::as_u64)
}

/// `f64` counterpart of [`arch_or_legacy_u64`].
fn arch_or_legacy_f64(
    metadata: &MetadataStore,
    architecture: Option<&str>,
    suffix: &str,
) -> Option<f64> {
    if let Some(arch) = architecture {
        if let Some(v) = metadata
            .get(&arch_key(arch, suffix))
            .and_then(MetadataValue::as_f64)
        {
            return Some(v);
        }
    }
    metadata
        .get(&legacy_key(suffix))
        .and_then(MetadataValue::as_f64)
}

/// The tensor table's most common quantization type, by tensor count — a
/// human-readable stand-in for "how is this model quantized" that is
/// always present (unlike `general.quantization_version`, a numeric enum
/// id the old flat-map `QUANTIZATION` key actually names, not a scheme
/// string like `"Q4_K_M"`/`"PQ2_0"`).
fn dominant_quant_type_name(tensors: &TensorStore) -> Option<String> {
    tensors
        .count_by_type()
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(ty, _)| ty.name().to_string())
}

/// Every metadata key `extract_model_card_from_gguf` already surfaces as a
/// typed field, so `extra_metadata` does not duplicate it. Unlike
/// [`known_key_set`] (the flat-map API's static set), this depends on the
/// file's actual architecture string.
fn known_key_set_for_gguf(architecture: Option<&str>) -> std::collections::HashSet<String> {
    let mut set: std::collections::HashSet<String> = [
        keys::MODEL_NAME,
        keys::ARCHITECTURE,
        keys::AUTHOR,
        keys::LICENSE,
        keys::DESCRIPTION,
        keys::SAMPLING_TEMP,
        keys::SAMPLING_TOP_P,
        keys::SAMPLING_TOP_K,
        keys::HADAMARD_VERSION,
        keys::TOKENIZER_TOKENS,
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    for suffix in [
        keys::arch_suffix::CONTEXT_LENGTH,
        keys::arch_suffix::EMBEDDING_LENGTH,
        keys::arch_suffix::BLOCK_COUNT,
        keys::arch_suffix::HEAD_COUNT,
        keys::arch_suffix::HEAD_COUNT_KV,
        keys::arch_suffix::ROPE_FREQ_BASE,
        keys::arch_suffix::FULL_ATTENTION_INTERVAL,
    ] {
        set.insert(legacy_key(suffix));
        if let Some(arch) = architecture {
            set.insert(arch_key(arch, suffix));
        }
    }
    set
}

/// Render a scalar [`MetadataValue`] as a display string for
/// `extra_metadata`. Returns `None` for `Array` values: dumping a
/// quarter-million-entry `tokenizer.ggml.tokens` or
/// `prism.hadamard.sign_values` list into a "model card" would defeat the
/// point of a card (a short, human-readable summary), not augment it.
fn scalar_display_string(v: &MetadataValue) -> Option<String> {
    match v {
        MetadataValue::String(s) => Some(s.clone()),
        MetadataValue::Bool(b) => Some(b.to_string()),
        MetadataValue::Uint8(n) => Some(n.to_string()),
        MetadataValue::Int8(n) => Some(n.to_string()),
        MetadataValue::Uint16(n) => Some(n.to_string()),
        MetadataValue::Int16(n) => Some(n.to_string()),
        MetadataValue::Uint32(n) => Some(n.to_string()),
        MetadataValue::Int32(n) => Some(n.to_string()),
        MetadataValue::Uint64(n) => Some(n.to_string()),
        MetadataValue::Int64(n) => Some(n.to_string()),
        MetadataValue::Float32(n) => Some(n.to_string()),
        MetadataValue::Float64(n) => Some(n.to_string()),
        MetadataValue::Array(_) => None,
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(keys::MODEL_NAME.to_owned(), "TestModel-7B".to_owned());
        m.insert(keys::ARCHITECTURE.to_owned(), "llama".to_owned());
        m.insert(keys::CONTEXT_LENGTH.to_owned(), "4096".to_owned());
        m.insert(keys::EMBEDDING_LENGTH.to_owned(), "4096".to_owned());
        m.insert(keys::NUM_LAYERS.to_owned(), "32".to_owned());
        m.insert(keys::NUM_HEADS.to_owned(), "32".to_owned());
        m.insert(keys::VOCAB_SIZE.to_owned(), "32000".to_owned());
        m
    }

    #[test]
    fn model_card_new_is_empty() {
        let card = ModelCard::new();
        assert!(card.is_empty());
    }

    #[test]
    fn populated_count_tracks_set_fields() {
        let mut card = ModelCard::new();
        assert_eq!(card.populated_count(), 0);
        card.model_name = Some("X".to_owned());
        assert_eq!(card.populated_count(), 1);
        card.architecture = Some("llama".to_owned());
        assert_eq!(card.populated_count(), 2);
        card.num_layers = Some(32);
        assert_eq!(card.populated_count(), 3);
    }

    #[test]
    fn markdown_empty_card_is_nonempty_string() {
        let card = ModelCard::new();
        let md = card.to_markdown();
        assert!(!md.is_empty());
    }

    #[test]
    fn markdown_contains_model_name() {
        let mut card = ModelCard::new();
        card.model_name = Some("Llama-3-8B".to_owned());
        let md = card.to_markdown();
        assert!(
            md.contains("Llama-3-8B"),
            "markdown must contain the model name"
        );
    }

    #[test]
    fn markdown_starts_with_heading() {
        let card = ModelCard::new();
        assert!(card.to_markdown().starts_with("# "));
    }

    #[test]
    fn summary_empty_card_is_nonempty() {
        let card = ModelCard::new();
        let s = card.to_summary();
        assert!(!s.is_empty());
    }

    #[test]
    fn summary_contains_known_fields() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        let s = card.to_summary();
        assert!(s.contains("TestModel-7B"));
        assert!(s.contains("llama"));
        assert!(s.contains("4096"));
    }

    #[test]
    fn extract_model_card_parses_name() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        assert_eq!(card.model_name.as_deref(), Some("TestModel-7B"));
    }

    #[test]
    fn extract_model_card_parses_architecture() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        assert_eq!(card.architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn extract_model_card_parses_context_length() {
        let metadata = sample_metadata();
        let card = extract_model_card(&metadata);
        assert_eq!(card.context_length, Some(4096));
    }

    #[test]
    fn extract_model_card_empty_metadata_gives_empty_card() {
        let card = extract_model_card(&HashMap::new());
        assert!(card.is_empty());
    }

    #[test]
    fn extract_known_fields_identifies_known() {
        let mut metadata = sample_metadata();
        metadata.insert("unknown.custom.key".to_owned(), "value".to_owned());
        let known = extract_known_fields(&metadata);
        assert!(known.contains_key(keys::MODEL_NAME));
        assert!(known.contains_key(keys::ARCHITECTURE));
        assert!(!known.contains_key("unknown.custom.key"));
    }

    #[test]
    fn estimated_param_count_returns_some_when_dims_known() {
        let mut card = ModelCard::new();
        card.embedding_length = Some(4096);
        card.num_layers = Some(32);
        card.num_heads = Some(32);
        card.num_kv_heads = Some(32);
        card.vocab_size = Some(32_000);
        let est = card.estimated_param_count();
        assert!(est.is_some(), "must return Some when dims are known");
        let b = est.expect("checked above");
        // A 4096/32-layer model is roughly 7B — allow a wide range.
        assert!(
            b > 1.0 && b < 50.0,
            "estimate {b:.2}B out of plausible range"
        );
    }

    // ── extract_model_card_from_gguf (core-gguf-13) ─────────────────────────

    use crate::gguf::types::GgufValueType;

    fn gguf_string(s: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
        b
    }

    fn kv_string(key: &str, value: &str) -> Vec<u8> {
        let mut b = gguf_string(key);
        b.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        b.extend_from_slice(&gguf_string(value));
        b
    }

    fn kv_u32(key: &str, value: u32) -> Vec<u8> {
        let mut b = gguf_string(key);
        b.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        b.extend_from_slice(&value.to_le_bytes());
        b
    }

    fn kv_f32(key: &str, value: f32) -> Vec<u8> {
        let mut b = gguf_string(key);
        b.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
        b.extend_from_slice(&value.to_le_bytes());
        b
    }

    fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
        let mut b = gguf_string(key);
        b.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
        b.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
        b.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            b.extend_from_slice(&gguf_string(v));
        }
        b
    }

    fn tensor_info_bytes(name: &str, shape: &[u64], type_id: u32, offset: u64) -> Vec<u8> {
        let mut b = gguf_string(name);
        b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&type_id.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b
    }

    /// A `qwen35`-flavoured metadata block modelled on the real Bonsai 2 27B
    /// GGUF header (`gguf_headers_summary.txt`): arch-scoped keys (no
    /// `llm.*`/`tokens_count` at all), plus Hadamard + sampling metadata.
    fn qwen35_metadata_bytes() -> (Vec<u8>, u64) {
        let entries: Vec<Vec<u8>> = vec![
            kv_string(keys::ARCHITECTURE, "qwen35"),
            kv_string(keys::MODEL_NAME, "Hf"),
            kv_u32("qwen35.context_length", 262_144),
            kv_u32("qwen35.attention.head_count", 24),
            kv_u32("qwen35.attention.head_count_kv", 4),
            kv_u32("qwen35.full_attention_interval", 4),
            kv_u32(keys::HADAMARD_VERSION, 1),
            kv_f32(keys::SAMPLING_TEMP, 1.0),
            kv_f32(keys::SAMPLING_TOP_P, 0.95),
            kv_u32(keys::SAMPLING_TOP_K, 20),
            kv_string_array(keys::TOKENIZER_TOKENS, &["a", "b", "c", "d", "e"]),
        ];
        let count = entries.len() as u64;
        (entries.concat(), count)
    }

    fn parse_metadata(bytes: &[u8], count: u64) -> MetadataStore {
        MetadataStore::parse(bytes, 0, count)
            .expect("synthetic metadata should parse")
            .0
    }

    fn parse_tensors(bytes: &[u8], count: u64) -> TensorStore {
        TensorStore::parse(bytes, 0, count)
            .expect("synthetic tensor table should parse")
            .0
    }

    #[test]
    fn extract_from_gguf_reads_arch_scoped_keys_not_llm_prefixed_ones() {
        let (bytes, count) = qwen35_metadata_bytes();
        let metadata = parse_metadata(&bytes, count);
        let tensors = TensorStore::new();

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert_eq!(card.architecture.as_deref(), Some("qwen35"));
        assert_eq!(card.model_name.as_deref(), Some("Hf"));
        assert_eq!(card.context_length, Some(262_144));
        assert_eq!(card.num_heads, Some(24));
        assert_eq!(card.num_kv_heads, Some(4));
        assert_eq!(card.full_attention_interval, Some(4));
    }

    #[test]
    fn extract_from_gguf_surfaces_hadamard_and_sampling_defaults() {
        let (bytes, count) = qwen35_metadata_bytes();
        let metadata = parse_metadata(&bytes, count);
        let tensors = TensorStore::new();

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert_eq!(card.hadamard_version, Some(1));
        assert!((card.sampling_temp.expect("temp set") - 1.0).abs() < 1e-6);
        assert!((card.sampling_top_p.expect("top_p set") - 0.95).abs() < 1e-3);
        assert_eq!(card.sampling_top_k, Some(20));
    }

    #[test]
    fn extract_from_gguf_derives_vocab_size_from_the_tokens_array_length() {
        let (bytes, count) = qwen35_metadata_bytes();
        let metadata = parse_metadata(&bytes, count);
        let tensors = TensorStore::new();

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        // The real key (`tokenizer.ggml.tokens_count`) does not exist in
        // this fixture, by design — only the true `tokenizer.ggml.tokens`
        // array does, matching every real GGUF file.
        assert_eq!(card.vocab_size, Some(5));
    }

    #[test]
    fn extract_from_gguf_falls_back_to_token_embd_shape_when_tokens_array_absent() {
        let entries: Vec<Vec<u8>> = vec![kv_string(keys::ARCHITECTURE, "qwen35")];
        let metadata = parse_metadata(&entries.concat(), entries.len() as u64);

        let tensor_bytes = tensor_info_bytes(keys::TOKEN_EMBD_TENSOR, &[5120, 248_320], 0, 0);
        let tensors = parse_tensors(&tensor_bytes, 1);

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert_eq!(
            card.vocab_size,
            Some(248_320),
            "must fall back to token_embd.weight's last shape dimension"
        );
    }

    #[test]
    fn extract_from_gguf_derives_quantization_from_the_dominant_tensor_type() {
        let entries: Vec<Vec<u8>> = vec![kv_string(keys::ARCHITECTURE, "qwen35")];
        let metadata = parse_metadata(&entries.concat(), entries.len() as u64);

        // 2 PQ2_0 (142) tensors + 1 F32 (0) norm — PQ2_0 must win.
        let mut tensor_bytes = Vec::new();
        tensor_bytes.extend_from_slice(&tensor_info_bytes("blk.0.ffn_up.weight", &[128], 142, 0));
        tensor_bytes.extend_from_slice(&tensor_info_bytes("blk.0.ffn_down.weight", &[128], 142, 0));
        tensor_bytes.extend_from_slice(&tensor_info_bytes("blk.0.attn_norm.weight", &[128], 0, 0));
        let tensors = parse_tensors(&tensor_bytes, 3);

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert_eq!(card.quantization.as_deref(), Some("PQ2_0"));
    }

    #[test]
    fn extract_from_gguf_populates_file_size_from_the_parameter_not_metadata() {
        let (bytes, count) = qwen35_metadata_bytes();
        let metadata = parse_metadata(&bytes, count);
        let tensors = TensorStore::new();

        let card = extract_model_card_from_gguf(&metadata, Some(7_211_000_000), &tensors);
        assert_eq!(card.file_size_bytes, Some(7_211_000_000));

        let no_len = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert_eq!(no_len.file_size_bytes, None);
    }

    #[test]
    fn extract_from_gguf_falls_back_to_legacy_llm_prefix_when_arch_key_absent() {
        // No `qwen35.context_length`; only the legacy generic spelling.
        let entries: Vec<Vec<u8>> = vec![
            kv_string(keys::ARCHITECTURE, "qwen35"),
            kv_u32("llm.context_length", 4096),
        ];
        let metadata = parse_metadata(&entries.concat(), entries.len() as u64);
        let tensors = TensorStore::new();

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert_eq!(card.context_length, Some(4096));
    }

    #[test]
    fn extract_from_gguf_excludes_bulk_arrays_from_extra_metadata() {
        let (bytes, count) = qwen35_metadata_bytes();
        let metadata = parse_metadata(&bytes, count);
        let tensors = TensorStore::new();

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert!(
            !card.extra_metadata.contains_key(keys::TOKENIZER_TOKENS),
            "the tokens array must never be dumped into extra_metadata"
        );
    }

    #[test]
    fn extract_from_gguf_puts_unrecognised_scalars_in_extra_metadata() {
        let entries: Vec<Vec<u8>> = vec![
            kv_string(keys::ARCHITECTURE, "qwen35"),
            kv_string("general.size_label", "27B"),
        ];
        let metadata = parse_metadata(&entries.concat(), entries.len() as u64);
        let tensors = TensorStore::new();

        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert_eq!(
            card.extra_metadata
                .get("general.size_label")
                .map(String::as_str),
            Some("27B")
        );
    }

    #[test]
    fn extract_from_gguf_empty_metadata_gives_an_empty_card() {
        let metadata = MetadataStore::new();
        let tensors = TensorStore::new();
        let card = extract_model_card_from_gguf(&metadata, None, &tensors);
        assert!(card.is_empty());
    }

    #[test]
    fn extract_from_gguf_renders_the_new_fields_in_markdown_and_summary() {
        let (bytes, count) = qwen35_metadata_bytes();
        let metadata = parse_metadata(&bytes, count);
        let tensors = TensorStore::new();
        let card = extract_model_card_from_gguf(&metadata, None, &tensors);

        let md = card.to_markdown();
        assert!(md.contains("Full Attention Interval"), "{md}");
        assert!(md.contains("Hadamard Version"), "{md}");
        assert!(md.contains("Sampling Defaults"), "{md}");

        let summary = card.to_summary();
        assert!(summary.contains("Hadamard Version: 1"), "{summary}");
    }
}
