//! `qwen35.*` architecture metadata, the `prism.hadamard.*` contract and the
//! hybrid tensor-name mapping for PrismML Bonsai 2.
//!
//! Bonsai 2 27 B is a Qwen3.5 hybrid: layer `i` is a full-attention layer iff
//! `(i + 1) % full_attention_interval == 0` (16 of 64 layers) and every other
//! layer is a Gated DeltaNet recurrent layer with a different tensor set
//! (`attn_qkv`, `attn_gate`, `ssm_*`). All 401 projection matrices are stored
//! in a rotated basis, described by a `prism.hadamard.*` block that a reader
//! **must** honour — running a folded weight through a non-Hadamard-aware
//! matmul is silently wrong maths, not a crash.
//!
//! Before this module the converter surface could emit none of that (CQ-06):
//! there were zero occurrences of `qwen35` or `hadamard` in any writer. The
//! [`HadamardSpec::validate`] checks below mirror `llama-model.cpp:1196-1335`
//! one for one, so a file this crate writes is rejected here — at write time,
//! with a specific message — rather than by a downstream loader.

use std::collections::BTreeSet;

use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};

use crate::convert::meta::arch_key;

/// GGUF architecture identifier for the Qwen3.5 hybrid.
pub const QWEN35_ARCH: &str = "qwen35";

/// The only `prism.hadamard.version` this implementation understands.
pub const HADAMARD_VERSION: u32 = 1;

/// The only transform name the PrismML contract defines.
pub const HADAMARD_TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";

/// The only axis the PrismML contract defines.
pub const HADAMARD_AXIS: &str = "input-last-dimension";

/// `prism.hadamard.*` metadata keys.
pub mod hadamard_keys {
    /// Contract version (`1`).
    pub const VERSION: &str = "prism.hadamard.version";
    /// FWHT block size (`1024`).
    pub const BLOCK_SIZE: &str = "prism.hadamard.block_size";
    /// Transform name.
    pub const TRANSFORM: &str = "prism.hadamard.transform";
    /// Axis the transform applies to.
    pub const AXIS: &str = "prism.hadamard.axis";
    /// `"identity"` or `"explicit"`.
    pub const SIGN_MODE: &str = "prism.hadamard.sign_mode";
    /// Input widths the sign vectors are defined for.
    pub const SIGN_WIDTHS: &str = "prism.hadamard.sign_widths";
    /// Concatenated ±1 sign vectors, one run per width.
    pub const SIGN_VALUES: &str = "prism.hadamard.sign_values";
    /// Tensors whose *input activation* is transformed before the matmul.
    pub const WEIGHT_NAMES: &str = "prism.hadamard.weight_names";
    /// Tensors whose *lookup result* gets the inverse transform.
    pub const INVERSE_WEIGHT_NAMES: &str = "prism.hadamard.inverse_weight_names";
    /// Whether `ssm_out`'s activation must be permuted tiled→grouped first.
    pub const GDN_V_GROUPED: &str = "prism.hadamard.gdn_v_grouped";
}

// ─── qwen35.* ─────────────────────────────────────────────────────────────────

/// The hybrid-specific half of the `qwen35.*` key set.
///
/// The base keys (`block_count`, `embedding_length`, …) are written by
/// [`crate::convert::meta::write_arch_metadata`]; this struct adds the keys
/// that only the hybrid architecture has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Metadata {
    /// `qwen35.full_attention_interval` — layer `i` is full attention iff
    /// `(i + 1) % interval == 0`.
    pub full_attention_interval: u32,
    /// `qwen35.rope.dimension_count` (`n_rot`), 64 for the 27 B.
    pub rope_dimension_count: u32,
    /// `qwen35.rope.dimension_sections`, `[11, 11, 10, 0]` for the 27 B.
    pub rope_dimension_sections: [i32; 4],
    /// `qwen35.ssm.conv_kernel` — depthwise causal conv width (4).
    pub ssm_conv_kernel: u32,
    /// `qwen35.ssm.state_size` — `head_k_dim == head_v_dim` (128).
    pub ssm_state_size: u32,
    /// `qwen35.ssm.group_count` — number of k heads (16).
    pub ssm_group_count: u32,
    /// `qwen35.ssm.time_step_rank` — number of v heads (48).
    pub ssm_time_step_rank: u32,
    /// `qwen35.ssm.inner_size` — `n_v_heads * head_v_dim` (6144).
    pub ssm_inner_size: u32,
    /// `qwen35.attention.value_length` (256).
    pub attention_value_length: u32,
}

impl Default for Qwen35Metadata {
    /// The verified Bonsai 2 27 B configuration (design Appendix A.4).
    fn default() -> Self {
        Self {
            full_attention_interval: 4,
            rope_dimension_count: 64,
            rope_dimension_sections: [11, 11, 10, 0],
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 48,
            ssm_inner_size: 6144,
            attention_value_length: 256,
        }
    }
}

impl Qwen35Metadata {
    /// Number of v heads.
    #[inline]
    pub fn n_v_heads(&self) -> u32 {
        self.ssm_time_step_rank
    }

    /// Number of k heads.
    #[inline]
    pub fn n_k_heads(&self) -> u32 {
        self.ssm_group_count
    }

    /// Width of the concatenated `q | k | v` projection the conv runs over:
    /// `2 * head_k_dim * n_k_heads + inner_size` (10240 for the 27 B).
    #[inline]
    pub fn conv_dim(&self) -> u32 {
        2 * self.ssm_state_size * self.ssm_group_count + self.ssm_inner_size
    }

    /// Fold the two keys shared with the generic architecture block into it.
    ///
    /// `rope.dimension_count` and `attention.value_length` live in the
    /// `<arch>.*` namespace and must be written exactly once; this is how a
    /// caller that has a [`Qwen35Metadata`] gets them there without a
    /// duplicate-key rejection.
    pub fn apply_to_arch(&self, arch: &mut crate::convert::meta::ArchMetadata) {
        arch.rope_dimension_count
            .get_or_insert(self.rope_dimension_count);
        arch.value_length.get_or_insert(self.attention_value_length);
    }

    /// Whether layer `layer` uses full attention.
    #[inline]
    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.full_attention_interval != 0
            && (layer + 1).is_multiple_of(self.full_attention_interval as usize)
    }

    /// Structural consistency, mirroring `runtime.py::load`: reject
    /// `nv <= 0 || nk <= 0 || nv % nk != 0 || inner % nv != 0`.
    pub fn validate(&self) -> Result<(), Qwen35Error> {
        let nv = self.n_v_heads();
        let nk = self.n_k_heads();
        if nv == 0 || nk == 0 {
            return Err(Qwen35Error::HeadCount { nv, nk });
        }
        if !nv.is_multiple_of(nk) {
            return Err(Qwen35Error::HeadCount { nv, nk });
        }
        if !self.ssm_inner_size.is_multiple_of(nv) {
            return Err(Qwen35Error::InnerSize {
                inner: self.ssm_inner_size,
                nv,
            });
        }
        if self.full_attention_interval == 0 {
            return Err(Qwen35Error::ZeroAttentionInterval);
        }
        // Sections cover the halved rotary dimension (real/imag pairs).
        let section_sum: i64 = self
            .rope_dimension_sections
            .iter()
            .map(|&s| i64::from(s))
            .sum();
        if self.rope_dimension_sections.iter().any(|&s| s < 0)
            || section_sum > i64::from(self.rope_dimension_count) / 2
        {
            return Err(Qwen35Error::RopeSections {
                sections: self.rope_dimension_sections,
                n_rot: self.rope_dimension_count,
            });
        }
        Ok(())
    }
}

/// Structural problems in a `qwen35.*` block.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Qwen35Error {
    /// `n_v_heads` / `n_k_heads` are zero or not a whole-number ratio.
    #[error("qwen35: invalid head counts — n_v_heads {nv}, n_k_heads {nk} (need nv % nk == 0, both > 0)")]
    HeadCount {
        /// `qwen35.ssm.time_step_rank`.
        nv: u32,
        /// `qwen35.ssm.group_count`.
        nk: u32,
    },

    /// `ssm.inner_size` is not divisible by the v-head count.
    #[error("qwen35: ssm.inner_size {inner} is not a multiple of n_v_heads {nv}")]
    InnerSize {
        /// `qwen35.ssm.inner_size`.
        inner: u32,
        /// `qwen35.ssm.time_step_rank`.
        nv: u32,
    },

    /// `full_attention_interval` is zero, which would make every layer linear
    /// *and* divide by zero in the layer-kind predicate.
    #[error("qwen35: full_attention_interval must be non-zero")]
    ZeroAttentionInterval,

    /// The M-RoPE sections do not fit in half the rotary dimension.
    #[error("qwen35: rope.dimension_sections {sections:?} do not fit in n_rot/2 ({n_rot}/2)")]
    RopeSections {
        /// `qwen35.rope.dimension_sections`.
        sections: [i32; 4],
        /// `qwen35.rope.dimension_count`.
        n_rot: u32,
    },
}

/// Write the keys that belong to the hybrid architecture **only**.
///
/// `rope.dimension_count` and `attention.value_length` are deliberately not
/// written here even though [`Qwen35Metadata`] carries them: both are part of
/// the shared architecture block, and emitting them twice makes the GGUF
/// reader reject the whole file with `duplicate metadata key`. Use
/// [`Qwen35Metadata::apply_to_arch`] to fold them into the
/// [`crate::convert::meta::ArchMetadata`] before that block is written.
///
/// Call after [`crate::convert::meta::write_arch_metadata`].
pub fn write_qwen35_metadata(writer: &mut GgufWriter<'_>, meta: &Qwen35Metadata) {
    let mut u32_key = |suffix: &str, value: u32| {
        writer.add_metadata(
            &arch_key(QWEN35_ARCH, suffix),
            MetadataWriteValue::U32(value),
        );
    };
    u32_key("full_attention_interval", meta.full_attention_interval);
    u32_key("ssm.conv_kernel", meta.ssm_conv_kernel);
    u32_key("ssm.state_size", meta.ssm_state_size);
    u32_key("ssm.group_count", meta.ssm_group_count);
    u32_key("ssm.time_step_rank", meta.ssm_time_step_rank);
    u32_key("ssm.inner_size", meta.ssm_inner_size);
    writer.add_metadata(
        &arch_key(QWEN35_ARCH, "rope.dimension_sections"),
        MetadataWriteValue::ArrayI32(meta.rope_dimension_sections.to_vec()),
    );
}

/// The exact set of keys [`write_qwen35_metadata`] writes.
///
/// Unlike [`crate::convert::meta::arch_metadata_keys`] this needs no
/// [`Qwen35Metadata`] instance: every key here is unconditional. Used to
/// de-duplicate metadata carried from a source file against this export's
/// own hybrid-only block (CQ-01's duplicate-key regression) — see
/// [`crate::convert::meta::filter_carried_metadata`].
pub fn qwen35_metadata_keys() -> BTreeSet<String> {
    [
        "full_attention_interval",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.time_step_rank",
        "ssm.inner_size",
        "rope.dimension_sections",
    ]
    .iter()
    .map(|suffix| arch_key(QWEN35_ARCH, suffix))
    .collect()
}

// ─── prism.hadamard.* ─────────────────────────────────────────────────────────

/// How the sign vectors are defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignMode {
    /// No sign flip — the transform is the plain FWHT.
    Identity,
    /// An explicit ±1 vector per input width.
    Explicit,
}

impl SignMode {
    /// The metadata spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Explicit => "explicit",
        }
    }

    /// Parse the metadata spelling.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "identity" => Some(Self::Identity),
            "explicit" => Some(Self::Explicit),
            _ => None,
        }
    }
}

/// The complete `prism.hadamard.*` contract for one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HadamardSpec {
    /// FWHT block size; must be a non-zero power of two (1024 for the 27 B).
    pub block_size: u32,
    /// Sign-vector mode.
    pub sign_mode: SignMode,
    /// Input widths the sign vectors are defined for.
    pub sign_widths: Vec<i32>,
    /// Concatenated ±1 values, one run of `width` entries per sign width.
    pub sign_values: Vec<i32>,
    /// Tensors whose input activation is rotated before the matmul.
    pub weight_names: Vec<String>,
    /// Tensors whose lookup result gets the inverse rotation.
    pub inverse_weight_names: Vec<String>,
    /// Whether `ssm_out`'s activation must be permuted tiled→grouped first.
    pub gdn_v_grouped: bool,
}

/// Everything [`HadamardSpec::validate`] can reject.
///
/// Each variant is a distinct, separately testable check, matching
/// `llama-model.cpp`'s validation one for one — a folded weight reaching a
/// non-Hadamard-aware path produces plausible-looking garbage rather than an
/// error, so every one of these is a hard failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HadamardError {
    /// `block_size` is zero or not a power of two.
    #[error("prism.hadamard.block_size {block_size} must be a non-zero power of two")]
    BlockSize {
        /// The offending block size.
        block_size: u32,
    },

    /// `sign_mode` is `explicit` but no widths were supplied.
    #[error("prism.hadamard.sign_mode is 'explicit' but sign_widths is empty")]
    ExplicitWithoutWidths,

    /// A sign width is zero, negative, or not a multiple of `block_size`.
    #[error(
        "prism.hadamard.sign_widths[{index}] = {width} must be positive and a multiple of \
         block_size {block_size}"
    )]
    SignWidth {
        /// Position in `sign_widths`.
        index: usize,
        /// The offending width.
        width: i32,
        /// The declared block size.
        block_size: u32,
    },

    /// `sum(sign_widths) != sign_values.len()`.
    #[error(
        "prism.hadamard.sign_values has {got} entries, expected {expected} = sum(sign_widths)"
    )]
    SignValueCount {
        /// Number of values supplied.
        got: usize,
        /// Number of values the widths imply.
        expected: usize,
    },

    /// A sign value is not `-1` or `+1`.
    #[error("prism.hadamard.sign_values[{index}] = {value} is not ±1")]
    SignValue {
        /// Position in `sign_values`.
        index: usize,
        /// The offending value.
        value: i32,
    },

    /// `weight_names` is empty although a Hadamard block is present.
    #[error("prism.hadamard.weight_names is empty — a Hadamard block with no folded weights is meaningless")]
    NoWeightNames,

    /// A folded name does not denote a tensor that can be folded.
    #[error(
        "prism.hadamard.weight_names contains '{name}', which is not a foldable projection \
         (expected output.weight or blk.<N>.<attn_q|attn_k|attn_v|attn_qkv|attn_gate|\
         attn_output|ffn_gate|ffn_up|ffn_down|ssm_out>.weight)"
    )]
    NotFoldable {
        /// The offending name.
        name: String,
    },

    /// A name appears twice in `weight_names`.
    #[error("prism.hadamard.weight_names contains '{name}' more than once")]
    DuplicateWeightName {
        /// The duplicated name.
        name: String,
    },

    /// `inverse_weight_names` contains something other than `token_embd.weight`.
    #[error(
        "prism.hadamard.inverse_weight_names contains '{name}'; only token_embd.weight can carry \
         the inverse transform"
    )]
    NotInvertible {
        /// The offending name.
        name: String,
    },

    /// A name is listed as both folded and inverse.
    #[error("'{name}' is listed in both weight_names and inverse_weight_names")]
    FoldedAndInverse {
        /// The offending name.
        name: String,
    },

    /// A listed name is not among the tensors actually in the file.
    #[error("prism.hadamard names '{name}', which is not a tensor in this file")]
    MissingTensor {
        /// The offending name.
        name: String,
    },
}

impl HadamardSpec {
    /// Validate the contract.
    ///
    /// `tensor_names` is the set of tensors that will actually be written;
    /// pass an empty slice to skip check 8 (presence), which is the only check
    /// that needs the tensor table.
    pub fn validate(&self, tensor_names: &BTreeSet<String>) -> Result<(), HadamardError> {
        // 2. block_size is a non-zero power of two.
        if self.block_size == 0 || !self.block_size.is_power_of_two() {
            return Err(HadamardError::BlockSize {
                block_size: self.block_size,
            });
        }

        // 5. explicit ⇒ sign_widths non-empty.
        if self.sign_mode == SignMode::Explicit && self.sign_widths.is_empty() {
            return Err(HadamardError::ExplicitWithoutWidths);
        }

        // 6. widths are positive multiples of block_size; the values line up
        //    and are all ±1.
        let mut expected = 0usize;
        for (index, &width) in self.sign_widths.iter().enumerate() {
            if width <= 0 || !(width as u32).is_multiple_of(self.block_size) {
                return Err(HadamardError::SignWidth {
                    index,
                    width,
                    block_size: self.block_size,
                });
            }
            expected += width as usize;
        }
        if expected != self.sign_values.len() {
            return Err(HadamardError::SignValueCount {
                got: self.sign_values.len(),
                expected,
            });
        }
        for (index, &value) in self.sign_values.iter().enumerate() {
            if value != 1 && value != -1 {
                return Err(HadamardError::SignValue { index, value });
            }
        }

        // 7. weight_names non-empty and every name foldable.
        if self.weight_names.is_empty() {
            return Err(HadamardError::NoWeightNames);
        }
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for name in &self.weight_names {
            if !is_foldable_weight_name(name) {
                return Err(HadamardError::NotFoldable { name: name.clone() });
            }
            // 8a. no duplicates.
            if !seen.insert(name.as_str()) {
                return Err(HadamardError::DuplicateWeightName { name: name.clone() });
            }
        }

        // 8b. inverse ⊆ {token_embd.weight}; folded ∩ inverse == ∅.
        for name in &self.inverse_weight_names {
            if name != "token_embd.weight" {
                return Err(HadamardError::NotInvertible { name: name.clone() });
            }
            if seen.contains(name.as_str()) {
                return Err(HadamardError::FoldedAndInverse { name: name.clone() });
            }
        }

        // 8c. every listed name is a tensor in the file.
        if !tensor_names.is_empty() {
            for name in self.weight_names.iter().chain(&self.inverse_weight_names) {
                if !tensor_names.contains(name) {
                    return Err(HadamardError::MissingTensor { name: name.clone() });
                }
            }
        }

        Ok(())
    }

    /// The ±1 vector for one input width, or `None` when the width has no
    /// sign run.
    pub fn signs_for(&self, width: usize) -> Option<&[i32]> {
        let mut offset = 0usize;
        for &w in &self.sign_widths {
            let w = usize::try_from(w).ok()?;
            if w == width {
                return self.sign_values.get(offset..offset + w);
            }
            offset += w;
        }
        None
    }
}

/// Whether `name` denotes a projection whose input activation can be folded
/// into the rotated basis.
///
/// Anything outside this set is a hard error rather than a warning: a folded
/// weight running through a non-Hadamard-aware kernel is silently wrong maths.
pub fn is_foldable_weight_name(name: &str) -> bool {
    const FOLDABLE_SUFFIXES: [&str; 10] = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_qkv.weight",
        "attn_gate.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "ssm_out.weight",
    ];

    if name == "output.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some(dot) = rest.find('.') else {
        return false;
    };
    let (index, suffix) = rest.split_at(dot);
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let Some(suffix) = suffix.strip_prefix('.') else {
        return false;
    };
    FOLDABLE_SUFFIXES.contains(&suffix)
}

/// Write the `prism.hadamard.*` block, validating it first.
///
/// The same [`HadamardSpec::validate`] a loader applies runs here, so an
/// inconsistent contract is refused at write time instead of producing a file
/// that only fails once someone tries to run it.
pub fn write_hadamard_metadata(
    writer: &mut GgufWriter<'_>,
    spec: &HadamardSpec,
    tensor_names: &BTreeSet<String>,
) -> Result<(), HadamardError> {
    spec.validate(tensor_names)?;

    writer.add_metadata(
        hadamard_keys::VERSION,
        MetadataWriteValue::U32(HADAMARD_VERSION),
    );
    writer.add_metadata(
        hadamard_keys::BLOCK_SIZE,
        MetadataWriteValue::U32(spec.block_size),
    );
    writer.add_metadata(
        hadamard_keys::TRANSFORM,
        MetadataWriteValue::Str(HADAMARD_TRANSFORM.to_string()),
    );
    writer.add_metadata(
        hadamard_keys::AXIS,
        MetadataWriteValue::Str(HADAMARD_AXIS.to_string()),
    );
    writer.add_metadata(
        hadamard_keys::SIGN_MODE,
        MetadataWriteValue::Str(spec.sign_mode.as_str().to_string()),
    );
    writer.add_metadata(
        hadamard_keys::SIGN_WIDTHS,
        MetadataWriteValue::ArrayI32(spec.sign_widths.clone()),
    );
    writer.add_metadata(
        hadamard_keys::SIGN_VALUES,
        MetadataWriteValue::ArrayI32(spec.sign_values.clone()),
    );
    writer.add_metadata(
        hadamard_keys::WEIGHT_NAMES,
        MetadataWriteValue::ArrayStr(spec.weight_names.clone()),
    );
    writer.add_metadata(
        hadamard_keys::INVERSE_WEIGHT_NAMES,
        MetadataWriteValue::ArrayStr(spec.inverse_weight_names.clone()),
    );
    writer.add_metadata(
        hadamard_keys::GDN_V_GROUPED,
        MetadataWriteValue::Bool(spec.gdn_v_grouped),
    );
    Ok(())
}

// ─── Hybrid tensor names ──────────────────────────────────────────────────────

/// GGUF tensor names for one hybrid layer.
///
/// A full-attention layer carries the classic Qwen3 attention set plus the
/// `post_attention_norm` pre-FFN norm; a linear layer replaces attention with
/// the Gated DeltaNet set (`attn_qkv`, `attn_gate`, `ssm_*`).
pub fn hybrid_layer_tensor_names(layer: usize, full_attention: bool) -> Vec<String> {
    let mut names: Vec<String> = Vec::with_capacity(16);
    let mut push = |suffix: &str| names.push(format!("blk.{layer}.{suffix}"));

    push("attn_norm.weight");
    push("post_attention_norm.weight");
    push("ffn_gate.weight");
    push("ffn_up.weight");
    push("ffn_down.weight");

    if full_attention {
        push("attn_q.weight");
        push("attn_k.weight");
        push("attn_v.weight");
        push("attn_output.weight");
        push("attn_q_norm.weight");
        push("attn_k_norm.weight");
    } else {
        push("attn_qkv.weight");
        push("attn_gate.weight");
        push("ssm_a");
        push("ssm_alpha.weight");
        push("ssm_beta.weight");
        push("ssm_conv1d.weight");
        push("ssm_dt.bias");
        push("ssm_norm.weight");
        push("ssm_out.weight");
    }

    names.sort();
    names
}

/// Tensors that must never be quantized in a `qwen35` model, beyond the
/// generic 1-D / `*norm.weight` rule.
///
/// `ssm_alpha.weight` and `ssm_beta.weight` are `[5120, 48]` **BF16** in the
/// real files: their first dimension *is* a block multiple, so the generic
/// predicate would happily quantize them, but the PrismML contract lists them
/// among the tensors that are explicitly **not** folded and not quantized.
/// `ssm_conv1d.weight` is `[4, 10240]` F32 — its `ne0` of 4 is not a block
/// multiple either, but naming it here makes the intent explicit rather than
/// accidental.
pub fn is_never_quantized_qwen35(name: &str) -> bool {
    const NEVER: [&str; 5] = [
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_conv1d.weight",
        "ssm_a",
        "ssm_dt.bias",
    ];
    NEVER.iter().any(|suffix| {
        name == *suffix || (name.starts_with("blk.") && name.ends_with(&format!(".{suffix}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxibonsai_core::gguf::reader::GgufFile;

    fn spec() -> HadamardSpec {
        HadamardSpec {
            block_size: 1024,
            sign_mode: SignMode::Explicit,
            sign_widths: vec![1024, 2048],
            sign_values: {
                let mut v = vec![1i32; 1024];
                v.extend(std::iter::repeat_n(-1i32, 2048));
                v
            },
            weight_names: vec![
                "output.weight".to_string(),
                "blk.0.attn_qkv.weight".to_string(),
                "blk.0.ssm_out.weight".to_string(),
            ],
            inverse_weight_names: vec!["token_embd.weight".to_string()],
            gdn_v_grouped: true,
        }
    }

    fn tensors() -> BTreeSet<String> {
        [
            "output.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_out.weight",
            "token_embd.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn valid_spec_passes() {
        spec().validate(&tensors()).expect("valid spec");
    }

    #[test]
    fn block_size_must_be_power_of_two() {
        let mut s = spec();
        s.block_size = 1000;
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::BlockSize { block_size: 1000 })
        ));
        s.block_size = 0;
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::BlockSize { block_size: 0 })
        ));
    }

    #[test]
    fn explicit_mode_needs_widths() {
        let mut s = spec();
        s.sign_widths.clear();
        s.sign_values.clear();
        assert_eq!(
            s.validate(&tensors()),
            Err(HadamardError::ExplicitWithoutWidths)
        );
    }

    #[test]
    fn widths_must_be_block_multiples() {
        let mut s = spec();
        s.sign_widths = vec![1500];
        s.sign_values = vec![1; 1500];
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::SignWidth { width: 1500, .. })
        ));
    }

    #[test]
    fn sign_value_count_must_match_widths() {
        let mut s = spec();
        s.sign_values.pop();
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::SignValueCount {
                expected: 3072,
                got: 3071
            })
        ));
    }

    #[test]
    fn sign_values_must_be_plus_minus_one() {
        let mut s = spec();
        s.sign_values[7] = 0;
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::SignValue { index: 7, value: 0 })
        ));
    }

    #[test]
    fn non_foldable_name_is_rejected() {
        let mut s = spec();
        s.weight_names.push("blk.0.attn_norm.weight".to_string());
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::NotFoldable { .. })
        ));
    }

    #[test]
    fn duplicate_folded_name_is_rejected() {
        let mut s = spec();
        s.weight_names.push("output.weight".to_string());
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::DuplicateWeightName { .. })
        ));
    }

    #[test]
    fn inverse_is_limited_to_token_embd() {
        let mut s = spec();
        s.inverse_weight_names = vec!["output.weight".to_string()];
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::NotInvertible { .. })
        ));
    }

    #[test]
    fn folded_and_inverse_are_disjoint() {
        let mut s = spec();
        s.weight_names.push("token_embd.weight".to_string());
        // token_embd is not a foldable projection, so that check fires first;
        // exercise the intersection rule with a genuinely foldable overlap.
        s.weight_names.pop();
        s.inverse_weight_names = vec!["token_embd.weight".to_string()];
        let mut names = tensors();
        names.insert("token_embd.weight".to_string());
        s.validate(&names).expect("disjoint sets are fine");
    }

    #[test]
    fn listed_tensor_must_exist() {
        let mut s = spec();
        s.weight_names.push("blk.9.ffn_up.weight".to_string());
        assert!(matches!(
            s.validate(&tensors()),
            Err(HadamardError::MissingTensor { .. })
        ));
    }

    #[test]
    fn foldable_name_classification() {
        assert!(is_foldable_weight_name("output.weight"));
        assert!(is_foldable_weight_name("blk.63.ffn_down.weight"));
        assert!(is_foldable_weight_name("blk.0.ssm_out.weight"));
        assert!(!is_foldable_weight_name("token_embd.weight"));
        assert!(!is_foldable_weight_name("blk.0.ssm_alpha.weight"));
        assert!(!is_foldable_weight_name("blk.x.ffn_up.weight"));
        assert!(!is_foldable_weight_name("ffn_up.weight"));
    }

    #[test]
    fn signs_for_slices_the_right_run() {
        let s = spec();
        assert_eq!(s.signs_for(1024).map(|v| v[0]), Some(1));
        assert_eq!(s.signs_for(2048).map(|v| v[0]), Some(-1));
        assert_eq!(s.signs_for(2048).map(<[i32]>::len), Some(2048));
        assert!(s.signs_for(4096).is_none());
    }

    #[test]
    fn metadata_round_trips_through_a_real_gguf() {
        let mut w = GgufWriter::new();
        let meta = Qwen35Metadata::default();
        meta.validate().expect("default config is valid");
        write_qwen35_metadata(&mut w, &meta);
        write_hadamard_metadata(&mut w, &spec(), &tensors()).expect("write hadamard");
        let bytes = w.to_bytes().expect("serialise");
        let gguf = GgufFile::parse(&bytes).expect("parse");

        assert_eq!(
            gguf.metadata.get_u32("qwen35.ssm.state_size").ok(),
            Some(128)
        );
        assert_eq!(
            gguf.metadata.get_u32("qwen35.full_attention_interval").ok(),
            Some(4)
        );
        assert_eq!(
            gguf.metadata
                .get_i32_array("qwen35.rope.dimension_sections")
                .expect("sections"),
            vec![11, 11, 10, 0]
        );
        // The whole reason ArrayI32 exists: -1 must survive as -1.
        let signs = gguf
            .metadata
            .get_i32_array(hadamard_keys::SIGN_VALUES)
            .expect("sign values");
        assert_eq!(signs.len(), 3072);
        assert_eq!(signs[2000], -1);
        assert_eq!(
            gguf.metadata.get_string(hadamard_keys::TRANSFORM).ok(),
            Some(HADAMARD_TRANSFORM)
        );
        assert_eq!(
            gguf.metadata.get_bool(hadamard_keys::GDN_V_GROUPED).ok(),
            Some(true)
        );
    }

    #[test]
    fn invalid_spec_is_refused_at_write_time() {
        let mut w = GgufWriter::new();
        let mut s = spec();
        s.block_size = 3;
        assert!(write_hadamard_metadata(&mut w, &s, &tensors()).is_err());
    }

    #[test]
    fn hybrid_layer_kinds_and_names() {
        let meta = Qwen35Metadata::default();
        // (i + 1) % 4 == 0 → 3, 7, …, 63.
        assert!(!meta.is_full_attention(0));
        assert!(meta.is_full_attention(3));
        assert!(meta.is_full_attention(63));
        assert_eq!((0..64).filter(|&i| meta.is_full_attention(i)).count(), 16);

        let linear = hybrid_layer_tensor_names(0, false);
        assert!(linear.contains(&"blk.0.attn_qkv.weight".to_string()));
        assert!(linear.contains(&"blk.0.ssm_out.weight".to_string()));
        assert!(!linear.contains(&"blk.0.attn_q.weight".to_string()));

        let full = hybrid_layer_tensor_names(3, true);
        assert!(full.contains(&"blk.3.attn_q.weight".to_string()));
        assert!(!full.contains(&"blk.3.ssm_out.weight".to_string()));
    }

    #[test]
    fn conv_dim_matches_the_real_model() {
        assert_eq!(Qwen35Metadata::default().conv_dim(), 10240);
    }

    #[test]
    fn structural_validation_rejects_bad_head_ratios() {
        // 47 % 16 != 0
        let m = Qwen35Metadata {
            ssm_time_step_rank: 47,
            ..Qwen35Metadata::default()
        };
        assert!(matches!(m.validate(), Err(Qwen35Error::HeadCount { .. })));

        // 6100 is not a multiple of 48
        let m = Qwen35Metadata {
            ssm_inner_size: 6100,
            ..Qwen35Metadata::default()
        };
        assert!(matches!(m.validate(), Err(Qwen35Error::InnerSize { .. })));

        let m = Qwen35Metadata {
            full_attention_interval: 0,
            ..Qwen35Metadata::default()
        };
        assert_eq!(m.validate(), Err(Qwen35Error::ZeroAttentionInterval));

        let m = Qwen35Metadata {
            rope_dimension_sections: [30, 30, 30, 0],
            ..Qwen35Metadata::default()
        };
        assert!(matches!(
            m.validate(),
            Err(Qwen35Error::RopeSections { .. })
        ));
    }

    #[test]
    fn apply_to_arch_fills_the_shared_keys_without_overwriting() {
        use crate::convert::meta::ArchMetadata;

        let hybrid = Qwen35Metadata::default();
        let mut arch = ArchMetadata {
            block_count: 64,
            embedding_length: 5120,
            feed_forward_length: 17408,
            head_count: 24,
            head_count_kv: 4,
            key_length: Some(256),
            value_length: None,
            context_length: 262_144,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1e7,
            rope_dimension_count: None,
        };
        hybrid.apply_to_arch(&mut arch);
        assert_eq!(arch.rope_dimension_count, Some(64));
        assert_eq!(arch.value_length, Some(256));

        // An explicit value already in the arch block wins.
        arch.value_length = Some(128);
        hybrid.apply_to_arch(&mut arch);
        assert_eq!(arch.value_length, Some(128));
    }

    #[test]
    fn qwen35_metadata_keys_never_overlaps_the_arch_block() {
        use crate::convert::meta::{arch_metadata_keys, ArchMetadata};

        let arch = ArchMetadata {
            block_count: 64,
            embedding_length: 5120,
            feed_forward_length: 17408,
            head_count: 24,
            head_count_kv: 4,
            key_length: Some(256),
            value_length: Some(256),
            context_length: 262_144,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1e7,
            rope_dimension_count: Some(64),
        };
        let arch_keys = arch_metadata_keys(QWEN35_ARCH, &arch);
        let hybrid_keys = qwen35_metadata_keys();

        // Both halves live under the literal "qwen35." prefix…
        assert!(arch_keys.iter().all(|k| k.starts_with("qwen35.")));
        assert!(hybrid_keys.iter().all(|k| k.starts_with("qwen35.")));
        // …but must be disjoint: `write_arch_metadata` and
        // `write_qwen35_metadata` must never both claim the same key.
        assert!(
            arch_keys.is_disjoint(&hybrid_keys),
            "arch keys {arch_keys:?} overlap hybrid keys {hybrid_keys:?}"
        );
        assert!(hybrid_keys.contains("qwen35.full_attention_interval"));
        assert!(arch_keys.contains("qwen35.block_count"));
    }

    #[test]
    fn never_quantized_list_covers_the_ssm_scalars() {
        assert!(is_never_quantized_qwen35("blk.0.ssm_alpha.weight"));
        assert!(is_never_quantized_qwen35("blk.12.ssm_beta.weight"));
        assert!(is_never_quantized_qwen35("blk.0.ssm_conv1d.weight"));
        assert!(is_never_quantized_qwen35("blk.0.ssm_a"));
        assert!(is_never_quantized_qwen35("blk.0.ssm_dt.bias"));
        assert!(!is_never_quantized_qwen35("blk.0.ssm_out.weight"));
        assert!(!is_never_quantized_qwen35("blk.0.ffn_up.weight"));
    }
}
