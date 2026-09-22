//! `HadamardConfig` — parse and validate `prism.hadamard.*` GGUF metadata.
//!
//! See `bonsai2-design.md` §1.7. PrismML's "folded" Bonsai 2 GGUFs
//! (`PQ2_0`/`PTQ1_0`) pre-rotate the *input activation* of most weight
//! matrices with a blockwise Hadamard transform, and record exactly which
//! tensors were folded (and with which sign pattern) in a `prism.hadamard.*`
//! metadata contract. [`HadamardConfig::from_metadata`] validates that
//! contract exactly as PrismML's own `fork/src/llama-model.cpp` does — every
//! one of its eight rules is a distinct, hard `Err`, never a warning,
//! because running the wrong (or no) rotation on a folded weight is silent
//! wrong math, not a cosmetic mismatch.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::metadata::MetadataStore;
use crate::gguf::tensor_info::{tensor_names, TensorStore};

const KEY_VERSION: &str = "prism.hadamard.version";
const KEY_BLOCK_SIZE: &str = "prism.hadamard.block_size";
const KEY_TRANSFORM: &str = "prism.hadamard.transform";
const KEY_AXIS: &str = "prism.hadamard.axis";
const KEY_SIGN_MODE: &str = "prism.hadamard.sign_mode";
const KEY_SIGN_WIDTHS: &str = "prism.hadamard.sign_widths";
const KEY_SIGN_VALUES: &str = "prism.hadamard.sign_values";
const KEY_WEIGHT_NAMES: &str = "prism.hadamard.weight_names";
const KEY_INVERSE_WEIGHT_NAMES: &str = "prism.hadamard.inverse_weight_names";
const KEY_GDN_V_GROUPED: &str = "prism.hadamard.gdn_v_grouped";

const EXPECTED_TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
const EXPECTED_AXIS: &str = "input-last-dimension";

/// `blk.<N>.<suffix>.weight` suffixes the fork accepts as foldable tensor
/// kinds (rule 7). `output.weight` is the one foldable name with no `blk.N.`
/// prefix and is checked separately.
const FOLDABLE_BLOCK_SUFFIXES: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_gate",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ssm_out",
];

/// The blockwise-Hadamard "fold" contract for one GGUF file: which tensors
/// were pre-rotated, with which per-width sign pattern, and how to invert
/// the rotation for the embedding lookup.
#[derive(Debug, Clone)]
pub struct HadamardConfig {
    /// FWHT block width (27B: 1024), always a power of two.
    pub block_size: usize,
    /// `width -> ±1.0 vector of exactly `width` entries`, sliced out of
    /// `prism.hadamard.sign_values` in `sign_widths` order.
    pub signs: HashMap<usize, Arc<[f32]>>,
    /// Tensor names whose *input activation* must be rotated (sign-flip then
    /// FWHT) before the matmul that consumes them.
    pub folded: HashSet<String>,
    /// Tensor names whose *lookup result* needs the inverse transform
    /// (FWHT then sign-flip) — in every real file, exactly
    /// `{"token_embd.weight"}`.
    pub inverse: HashSet<String>,
    /// `true` -> the Gated-DeltaNet `ssm_out`/value activation must be
    /// permuted tiled -> grouped before consumption (see
    /// `bonsai2-design.md` §3.3's V-head grouping).
    pub gdn_v_grouped: bool,
}

/// Alias kept for the finder-facing name used while this contract was
/// designed (`core-gguf.md`'s `HadamardSpec::from_metadata`); both names
/// resolve to the same type so downstream code can use either.
pub type HadamardSpec = HadamardConfig;

impl HadamardConfig {
    /// Parse and validate `prism.hadamard.*` from `metadata`.
    ///
    /// Returns `Ok(None)` when [`KEY_VERSION`] is absent — a non-folded
    /// model (e.g. the pre-Hadamard `Bonsai-27B-Q1_0.gguf` /
    /// `Ternary-Bonsai-27B-{PQ2_0,Q2_0}.gguf` generation). Once that key is
    /// present, every one of the eight rules below is a hard
    /// [`BonsaiError::HadamardContract`] — there is no partial/best-effort
    /// mode, because a wrongly-applied (or silently skipped) rotation is
    /// wrong math, not a display glitch.
    ///
    /// This function only validates what is derivable from `metadata`
    /// alone; the one remaining clause of rule 8 — every folded/inverse
    /// name must actually exist as a tensor in this file — needs the
    /// [`TensorStore`] and is [`HadamardConfig::validate_against_tensors`].
    pub fn from_metadata(metadata: &MetadataStore) -> BonsaiResult<Option<Self>> {
        if metadata.get(KEY_VERSION).is_none() {
            return Ok(None);
        }

        // Rule 1: version == 1.
        let version = require_u32(metadata, KEY_VERSION)?;
        if version != 1 {
            return Err(contract_err(format!(
                "unsupported {KEY_VERSION} {version}; only version 1 is implemented"
            )));
        }

        // Rule 2: block_size is a nonzero power of two.
        let block_size = require_u32(metadata, KEY_BLOCK_SIZE)? as usize;
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(contract_err(format!(
                "{KEY_BLOCK_SIZE} ({block_size}) must be a nonzero power of two"
            )));
        }

        // Rule 3: transform.
        let transform = require_str(metadata, KEY_TRANSFORM)?;
        if transform != EXPECTED_TRANSFORM {
            return Err(contract_err(format!(
                "unsupported {KEY_TRANSFORM} '{transform}'; expected '{EXPECTED_TRANSFORM}'"
            )));
        }

        // Rule 4: axis.
        let axis = require_str(metadata, KEY_AXIS)?;
        if axis != EXPECTED_AXIS {
            return Err(contract_err(format!(
                "unsupported {KEY_AXIS} '{axis}'; expected '{EXPECTED_AXIS}'"
            )));
        }

        // Rule 5: sign_mode ∈ {identity, explicit}; explicit ⇒ sign_widths
        // non-empty.
        let sign_mode = require_str(metadata, KEY_SIGN_MODE)?.to_string();
        if sign_mode != "identity" && sign_mode != "explicit" {
            return Err(contract_err(format!(
                "unsupported {KEY_SIGN_MODE} '{sign_mode}'; expected 'identity' or 'explicit'"
            )));
        }
        let sign_widths_present = metadata.get(KEY_SIGN_WIDTHS).is_some();
        if sign_mode == "explicit" && !sign_widths_present {
            return Err(contract_err(format!(
                "{KEY_SIGN_MODE} is 'explicit' but {KEY_SIGN_WIDTHS} is absent"
            )));
        }

        // Rule 6: every width > 0 and a multiple of block_size;
        // sum(widths) == sign_values.len(); every value ∈ {-1, +1}.
        let signs = if sign_widths_present {
            parse_signs(metadata, block_size)?
        } else {
            HashMap::new()
        };

        // Rule 7: weight_names non-empty; every name a foldable kind.
        let weight_names = require_str_array(metadata, KEY_WEIGHT_NAMES)?;
        if weight_names.is_empty() {
            return Err(contract_err(format!(
                "{KEY_WEIGHT_NAMES} must be non-empty"
            )));
        }
        let mut folded: HashSet<String> = HashSet::with_capacity(weight_names.len());
        for name in weight_names {
            if !is_foldable_kind(&name) {
                return Err(contract_err(format!(
                    "{KEY_WEIGHT_NAMES} contains '{name}', which is not a foldable tensor kind"
                )));
            }
            // Rule 8a: no duplicate names in weight_names.
            if !folded.insert(name.clone()) {
                return Err(contract_err(format!(
                    "{KEY_WEIGHT_NAMES} contains duplicate name '{name}'"
                )));
            }
        }

        // Rule 8b/8c: inverse_weight_names ⊆ {"token_embd.weight"};
        // folded ∩ inverse == ∅. Optional, defaulting to empty exactly like
        // the fork (`ml.get_arr("prism.hadamard.inverse_weight_names",
        // inverse_names, false)` at src_llama-model.cpp:1323 — the `false`
        // is llama.cpp's "not required" flag, and the C++ vector's default
        // construction is empty): a folded model with no inverse-transform
        // table (e.g. one that does not fold `token_embd.weight`'s lookup)
        // is a legitimate, if unseen-so-far, configuration, not a contract
        // violation. A key that IS present but malformed is still a hard
        // error, same as every other rule here.
        let inverse_weight_names = match metadata.get(KEY_INVERSE_WEIGHT_NAMES) {
            None => Vec::new(),
            Some(_) => require_str_array(metadata, KEY_INVERSE_WEIGHT_NAMES)?,
        };
        let mut inverse: HashSet<String> = HashSet::with_capacity(inverse_weight_names.len());
        for name in inverse_weight_names {
            if name != tensor_names::TOKEN_EMBD {
                return Err(contract_err(format!(
                    "{KEY_INVERSE_WEIGHT_NAMES} contains '{name}'; only '{}' is supported",
                    tensor_names::TOKEN_EMBD
                )));
            }
            if !inverse.insert(name.clone()) {
                return Err(contract_err(format!(
                    "{KEY_INVERSE_WEIGHT_NAMES} contains duplicate name '{name}'"
                )));
            }
            if folded.contains(&name) {
                return Err(contract_err(format!(
                    "'{name}' appears in both {KEY_WEIGHT_NAMES} and {KEY_INVERSE_WEIGHT_NAMES}"
                )));
            }
        }

        // Optional, defaulting to `false` exactly like the fork
        // (`ml.get_key("prism.hadamard.gdn_v_grouped", hadamard_gdn_v_grouped,
        // false)` at src_llama-model.cpp:1259, with the C++ member itself
        // default-initialised `= false` in llama-model.h): a folded model
        // with no Gated-DeltaNet V-head grouping to apply is legitimate,
        // not a contract violation. A key that IS present but not a bool is
        // still a hard error.
        let gdn_v_grouped = match metadata.get(KEY_GDN_V_GROUPED) {
            None => false,
            Some(value) => value
                .as_bool()
                .ok_or_else(|| contract_err(format!("'{KEY_GDN_V_GROUPED}' must be a bool")))?,
        };

        Ok(Some(Self {
            block_size,
            signs,
            folded,
            inverse,
            gdn_v_grouped,
        }))
    }

    /// Rule 8's remaining clause: every folded or inverse-transformed name
    /// must actually be a tensor present in this file. Separate from
    /// [`HadamardConfig::from_metadata`] because it needs the
    /// [`TensorStore`], which a metadata-only parse does not have.
    pub fn validate_against_tensors(&self, tensors: &TensorStore) -> BonsaiResult<()> {
        for name in self.folded.iter().chain(self.inverse.iter()) {
            if tensors.get(name).is_none() {
                return Err(contract_err(format!(
                    "'{name}' is listed in the Hadamard fold contract but is not a tensor in \
                     this file"
                )));
            }
        }
        Ok(())
    }

    /// The `±1.0` sign vector for `width`, or
    /// [`BonsaiError::HadamardContract`] naming the width when none is
    /// configured (e.g. `sign_mode == "identity"` with no `sign_widths`
    /// declared for it).
    #[inline]
    pub fn signs_for(&self, width: usize) -> BonsaiResult<&[f32]> {
        self.signs
            .get(&width)
            .map(|s| s.as_ref())
            .ok_or_else(|| contract_err(format!("no sign vector configured for width {width}")))
    }

    /// Whether `name`'s input activation must be rotated before the matmul
    /// that consumes it.
    #[inline]
    pub fn is_folded(&self, name: &str) -> bool {
        self.folded.contains(name)
    }

    /// Whether `name`'s lookup result needs the inverse transform.
    #[inline]
    pub fn is_inverse(&self, name: &str) -> bool {
        self.inverse.contains(name)
    }
}

fn contract_err(reason: impl Into<String>) -> BonsaiError {
    BonsaiError::HadamardContract {
        reason: reason.into(),
    }
}

fn require_str<'a>(metadata: &'a MetadataStore, key: &str) -> BonsaiResult<&'a str> {
    metadata
        .get(key)
        .ok_or_else(|| contract_err(format!("missing required key '{key}'")))?
        .as_str()
        .ok_or_else(|| contract_err(format!("'{key}' must be a string")))
}

fn require_u32(metadata: &MetadataStore, key: &str) -> BonsaiResult<u32> {
    metadata
        .get(key)
        .ok_or_else(|| contract_err(format!("missing required key '{key}'")))?
        .as_u32()
        .ok_or_else(|| contract_err(format!("'{key}' must be an unsigned integer")))
}

fn require_i32_array(metadata: &MetadataStore, key: &str) -> BonsaiResult<Vec<i32>> {
    metadata
        .get_i32_array(key)
        .map_err(|e| contract_err(format!("'{key}': {e}")))
}

fn require_str_array(metadata: &MetadataStore, key: &str) -> BonsaiResult<Vec<String>> {
    metadata
        .get_string_array(key)
        .map_err(|e| contract_err(format!("'{key}': {e}")))
}

/// Rule 6: parse `sign_widths`/`sign_values` into the per-width sign map.
fn parse_signs(
    metadata: &MetadataStore,
    block_size: usize,
) -> BonsaiResult<HashMap<usize, Arc<[f32]>>> {
    let widths_i32 = require_i32_array(metadata, KEY_SIGN_WIDTHS)?;
    let mut widths = Vec::with_capacity(widths_i32.len());
    for w in widths_i32 {
        let w = usize::try_from(w)
            .map_err(|_| contract_err(format!("{KEY_SIGN_WIDTHS} contains negative width {w}")))?;
        if w == 0 || !w.is_multiple_of(block_size) {
            return Err(contract_err(format!(
                "{KEY_SIGN_WIDTHS} entry {w} must be a nonzero multiple of block_size \
                 ({block_size})"
            )));
        }
        widths.push(w);
    }

    let expected_total: usize = widths.iter().sum();
    let values_i32 = require_i32_array(metadata, KEY_SIGN_VALUES)?;
    // HARD error (not a warning): the fork treats a length mismatch as a
    // fatal contract violation, since it means the concatenated sign array
    // cannot be sliced into the declared per-width chunks at all.
    if values_i32.len() != expected_total {
        return Err(contract_err(format!(
            "{KEY_SIGN_VALUES}.len() ({}) != sum({KEY_SIGN_WIDTHS}) ({expected_total})",
            values_i32.len()
        )));
    }

    let mut values = Vec::with_capacity(values_i32.len());
    for v in values_i32 {
        match v {
            -1 => values.push(-1.0f32),
            1 => values.push(1.0f32),
            other => {
                return Err(contract_err(format!(
                    "{KEY_SIGN_VALUES} contains {other}, which is not \u{00b1}1"
                )))
            }
        }
    }

    let mut signs = HashMap::with_capacity(widths.len());
    let mut cursor = 0usize;
    for w in widths {
        let slice: Arc<[f32]> = Arc::from(&values[cursor..cursor + w]);
        signs.insert(w, slice);
        cursor += w;
    }
    Ok(signs)
}

/// Rule 7: `name` is `"output.weight"` or `"blk.<N>.<suffix>.weight"` where
/// `<N>` is a decimal integer and `<suffix>` is one of
/// [`FOLDABLE_BLOCK_SUFFIXES`].
fn is_foldable_kind(name: &str) -> bool {
    if name == tensor_names::OUTPUT {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some(dot) = rest.find('.') else {
        return false;
    };
    let (layer_str, rest) = rest.split_at(dot);
    if layer_str.is_empty() || !layer_str.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let Some(suffix_and_weight) = rest.strip_prefix('.') else {
        return false;
    };
    let Some(suffix) = suffix_and_weight.strip_suffix(".weight") else {
        return false;
    };
    FOLDABLE_BLOCK_SUFFIXES.contains(&suffix)
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

    /// A small but complete, valid Hadamard contract: two widths (64, 128),
    /// three foldable weight names (one `output.weight`, two `blk.N.*`),
    /// one inverse name. Kept tiny (block_size 64, not the real 1024) for
    /// fast unit tests; the real numbers are asserted separately against
    /// staged real headers in `tests/config_real_headers.rs`.
    fn valid_hadamard_pairs() -> Vec<(&'static str, MetadataWriteValue)> {
        let mut values = vec![-1i32; 64];
        values.extend(vec![1i32; 128]);
        vec![
            (KEY_VERSION, MetadataWriteValue::U32(1)),
            (KEY_BLOCK_SIZE, MetadataWriteValue::U32(64)),
            (
                KEY_TRANSFORM,
                MetadataWriteValue::Str(EXPECTED_TRANSFORM.to_string()),
            ),
            (KEY_AXIS, MetadataWriteValue::Str(EXPECTED_AXIS.to_string())),
            (
                KEY_SIGN_MODE,
                MetadataWriteValue::Str("explicit".to_string()),
            ),
            (KEY_SIGN_WIDTHS, MetadataWriteValue::ArrayI32(vec![64, 128])),
            (KEY_SIGN_VALUES, MetadataWriteValue::ArrayI32(values)),
            (
                KEY_WEIGHT_NAMES,
                MetadataWriteValue::ArrayStr(vec![
                    "output.weight".to_string(),
                    "blk.0.attn_qkv.weight".to_string(),
                    "blk.1.ffn_down.weight".to_string(),
                ]),
            ),
            (
                KEY_INVERSE_WEIGHT_NAMES,
                MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
            ),
            (KEY_GDN_V_GROUPED, MetadataWriteValue::Bool(true)),
        ]
    }

    fn overridden(
        mut pairs: Vec<(&'static str, MetadataWriteValue)>,
        key: &'static str,
        value: MetadataWriteValue,
    ) -> Vec<(&'static str, MetadataWriteValue)> {
        if let Some(entry) = pairs.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = value;
        } else {
            pairs.push((key, value));
        }
        pairs
    }

    fn without(
        mut pairs: Vec<(&'static str, MetadataWriteValue)>,
        key: &'static str,
    ) -> Vec<(&'static str, MetadataWriteValue)> {
        pairs.retain(|(k, _)| *k != key);
        pairs
    }

    // ── Ok(None) when absent ────────────────────────────────────────────────

    #[test]
    fn absent_version_key_is_ok_none() {
        let metadata = build_metadata_store(vec![]);
        assert!(HadamardConfig::from_metadata(&metadata)
            .expect("absent version must not error")
            .is_none());
    }

    // ── Happy path ──────────────────────────────────────────────────────────

    #[test]
    fn valid_contract_parses() {
        let metadata = build_metadata_store(valid_hadamard_pairs());
        let hadamard = HadamardConfig::from_metadata(&metadata)
            .expect("valid contract should parse")
            .expect("version is present, so this must be Some");
        assert_eq!(hadamard.block_size, 64);
        assert_eq!(hadamard.folded.len(), 3);
        assert!(hadamard.is_folded("output.weight"));
        assert!(hadamard.is_folded("blk.0.attn_qkv.weight"));
        assert!(hadamard.is_folded("blk.1.ffn_down.weight"));
        assert!(!hadamard.is_folded("blk.0.attn_norm.weight"));
        assert!(hadamard.is_inverse("token_embd.weight"));
        assert!(hadamard.gdn_v_grouped);

        let w64 = hadamard.signs_for(64).expect("width 64 configured");
        assert_eq!(w64.len(), 64);
        assert!(w64.iter().all(|&v| v == -1.0));
        let w128 = hadamard.signs_for(128).expect("width 128 configured");
        assert_eq!(w128.len(), 128);
        assert!(w128.iter().all(|&v| v == 1.0));

        assert!(hadamard.signs_for(999).is_err());
    }

    #[test]
    fn validate_against_tensors_accepts_when_all_present() {
        use crate::gguf::writer::{TensorEntry, TensorType};
        let metadata = build_metadata_store(valid_hadamard_pairs());
        let hadamard = HadamardConfig::from_metadata(&metadata)
            .expect("should parse")
            .expect("Some");

        let mut writer = GgufWriter::new();
        for name in [
            "output.weight",
            "blk.0.attn_qkv.weight",
            "blk.1.ffn_down.weight",
            "token_embd.weight",
        ] {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                shape: vec![4],
                tensor_type: TensorType::F32,
                data: vec![0u8; 16],
            });
        }
        let bytes = writer.to_bytes().expect("serialise");
        let file = GgufFile::parse(&bytes).expect("parse");
        assert!(hadamard.validate_against_tensors(&file.tensors).is_ok());
    }

    #[test]
    fn validate_against_tensors_rejects_a_missing_folded_tensor() {
        use crate::gguf::writer::{TensorEntry, TensorType};
        let metadata = build_metadata_store(valid_hadamard_pairs());
        let hadamard = HadamardConfig::from_metadata(&metadata)
            .expect("should parse")
            .expect("Some");

        // Only two of the three folded names plus the inverse name.
        let mut writer = GgufWriter::new();
        for name in ["output.weight", "token_embd.weight"] {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                shape: vec![4],
                tensor_type: TensorType::F32,
                data: vec![0u8; 16],
            });
        }
        let bytes = writer.to_bytes().expect("serialise");
        let file = GgufFile::parse(&bytes).expect("parse");
        let err = hadamard
            .validate_against_tensors(&file.tensors)
            .expect_err("a folded name absent from the tensor list must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── Rule 1: version ─────────────────────────────────────────────────────

    #[test]
    fn rule1_wrong_version_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_VERSION,
            MetadataWriteValue::U32(2),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata).expect_err("version 2 must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── Rule 2: block_size ──────────────────────────────────────────────────

    #[test]
    fn rule2_non_power_of_two_block_size_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_BLOCK_SIZE,
            MetadataWriteValue::U32(100),
        );
        let metadata = build_metadata_store(pairs);
        let err =
            HadamardConfig::from_metadata(&metadata).expect_err("block_size 100 must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule2_zero_block_size_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_BLOCK_SIZE,
            MetadataWriteValue::U32(0),
        );
        let metadata = build_metadata_store(pairs);
        let err =
            HadamardConfig::from_metadata(&metadata).expect_err("block_size 0 must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── Rule 3: transform ───────────────────────────────────────────────────

    #[test]
    fn rule3_wrong_transform_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_TRANSFORM,
            MetadataWriteValue::Str("dct".to_string()),
        );
        let metadata = build_metadata_store(pairs);
        let err =
            HadamardConfig::from_metadata(&metadata).expect_err("wrong transform must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── Rule 4: axis ────────────────────────────────────────────────────────

    #[test]
    fn rule4_wrong_axis_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_AXIS,
            MetadataWriteValue::Str("output-last-dimension".to_string()),
        );
        let metadata = build_metadata_store(pairs);
        let err =
            HadamardConfig::from_metadata(&metadata).expect_err("wrong axis must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── Rule 5: sign_mode ───────────────────────────────────────────────────

    #[test]
    fn rule5_invalid_sign_mode_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_SIGN_MODE,
            MetadataWriteValue::Str("random".to_string()),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("invalid sign_mode must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule5_explicit_without_sign_widths_is_rejected() {
        let pairs = without(valid_hadamard_pairs(), KEY_SIGN_WIDTHS);
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("sign_mode=explicit with no sign_widths must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule5_identity_without_sign_widths_is_accepted_with_empty_signs() {
        let mut pairs = without(valid_hadamard_pairs(), KEY_SIGN_WIDTHS);
        pairs = without(pairs, KEY_SIGN_VALUES);
        pairs = overridden(
            pairs,
            KEY_SIGN_MODE,
            MetadataWriteValue::Str("identity".to_string()),
        );
        let metadata = build_metadata_store(pairs);
        let hadamard = HadamardConfig::from_metadata(&metadata)
            .expect("identity mode with no sign_widths should parse")
            .expect("Some");
        assert!(hadamard.signs.is_empty());
    }

    // ── Rule 6: widths / values ─────────────────────────────────────────────

    #[test]
    fn rule6_width_not_a_multiple_of_block_size_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_SIGN_WIDTHS,
            MetadataWriteValue::ArrayI32(vec![65, 128]), // 65 not a multiple of 64
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("a width not a multiple of block_size must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule6_zero_width_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_SIGN_WIDTHS,
            MetadataWriteValue::ArrayI32(vec![0, 128]),
        );
        let metadata = build_metadata_store(pairs);
        let err =
            HadamardConfig::from_metadata(&metadata).expect_err("a zero width must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule6_sign_values_length_mismatch_is_a_hard_error() {
        // sum(widths) == 192, but only 100 values supplied.
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_SIGN_VALUES,
            MetadataWriteValue::ArrayI32(vec![-1i32; 100]),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("sign_values.len() != sum(sign_widths) must be a HARD error");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule6_sign_value_not_plus_minus_one_is_rejected() {
        let mut values = vec![-1i32; 64];
        values.extend(vec![1i32; 127]);
        values.push(2); // not ±1
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_SIGN_VALUES,
            MetadataWriteValue::ArrayI32(values),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("a sign value of 2 must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── Rule 7: weight_names ────────────────────────────────────────────────

    #[test]
    fn rule7_empty_weight_names_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_WEIGHT_NAMES,
            MetadataWriteValue::ArrayStr(vec![]),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("empty weight_names must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule7_non_foldable_name_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_WEIGHT_NAMES,
            MetadataWriteValue::ArrayStr(vec!["blk.0.attn_norm.weight".to_string()]),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("attn_norm.weight is not a foldable kind and must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn is_foldable_kind_accepts_every_documented_suffix_and_rejects_junk() {
        assert!(is_foldable_kind("output.weight"));
        for suffix in FOLDABLE_BLOCK_SUFFIXES {
            assert!(is_foldable_kind(&format!("blk.0.{suffix}.weight")));
            assert!(is_foldable_kind(&format!("blk.63.{suffix}.weight")));
        }
        assert!(!is_foldable_kind("blk.0.attn_norm.weight"));
        assert!(!is_foldable_kind("blk.0.ssm_alpha.weight"));
        assert!(!is_foldable_kind("blk.x.attn_q.weight")); // non-numeric layer
        assert!(!is_foldable_kind("blk.0.attn_q.bias")); // wrong suffix
        assert!(!is_foldable_kind("attn_q.weight")); // missing blk. prefix
        assert!(!is_foldable_kind("output_norm.weight"));
        assert!(!is_foldable_kind(""));
    }

    // ── Rule 8: duplicates / inverse / disjointness ────────────────────────

    #[test]
    fn rule8_duplicate_weight_name_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_WEIGHT_NAMES,
            MetadataWriteValue::ArrayStr(vec![
                "output.weight".to_string(),
                "output.weight".to_string(),
            ]),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("a duplicate weight_names entry must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule8_inverse_name_other_than_token_embd_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_INVERSE_WEIGHT_NAMES,
            MetadataWriteValue::ArrayStr(vec!["output.weight".to_string()]),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("an inverse name other than token_embd.weight must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn rule8_duplicate_inverse_name_is_rejected() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_INVERSE_WEIGHT_NAMES,
            MetadataWriteValue::ArrayStr(vec![
                "token_embd.weight".to_string(),
                "token_embd.weight".to_string(),
            ]),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("a duplicate inverse_weight_names entry must be rejected");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    /// Rule 8's `folded ∩ inverse == ∅` check is defensive: rule 7 already
    /// restricts `weight_names` to foldable kinds (`output.weight` /
    /// `blk.N.*.weight`) and rule 8's own inverse-name check restricts
    /// `inverse_weight_names` to exactly `"token_embd.weight"`, which is
    /// never itself a foldable kind — so the two sets can never overlap
    /// through valid-shaped input today. This test documents that
    /// composition directly (attempting to fold `token_embd.weight` is
    /// rejected by rule 7, before the disjointness check is ever reached),
    /// and keeps the disjointness check itself as forward-compatible
    /// defense in depth should either allowlist ever be relaxed.
    #[test]
    fn rule8_folding_the_embedding_itself_is_rejected_by_rule7_before_disjointness_applies() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_WEIGHT_NAMES,
            MetadataWriteValue::ArrayStr(vec!["token_embd.weight".to_string()]),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("token_embd.weight is not a foldable kind");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── gdn_v_grouped ───────────────────────────────────────────────────────

    /// Corrected by this package's own verifier review: `gdn_v_grouped` is
    /// not one of design §1.7's eight numbered validation rules, and the
    /// fork reads it optionally (`ml.get_key("prism.hadamard.gdn_v_grouped",
    /// hadamard_gdn_v_grouped, false)` at src_llama-model.cpp:1259, with the
    /// C++ member default-initialised `= false` in llama-model.h) — a
    /// folded model that never grouped its Gated-DeltaNet V-heads is a
    /// legitimate configuration, not a contract violation. This replaces
    /// the previous `missing_gdn_v_grouped_is_hard_error`, which enshrined
    /// a stricter-than-the-fork defect.
    #[test]
    fn missing_gdn_v_grouped_defaults_to_false() {
        let pairs = without(valid_hadamard_pairs(), KEY_GDN_V_GROUPED);
        let metadata = build_metadata_store(pairs);
        let hadamard = HadamardConfig::from_metadata(&metadata)
            .expect("a missing gdn_v_grouped must default, not error")
            .expect("version is present, so this must be Some");
        assert!(!hadamard.gdn_v_grouped);
    }

    /// The default-on-absence above must not weaken the present-but-wrong-type
    /// case: `gdn_v_grouped` written as a non-bool is still a hard error.
    #[test]
    fn gdn_v_grouped_present_but_wrong_type_is_still_a_hard_error() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_GDN_V_GROUPED,
            MetadataWriteValue::U32(1),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("a present-but-malformed gdn_v_grouped must still be a hard error");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── inverse_weight_names (optional, like the fork) ─────────────────────

    /// Mirrors `missing_gdn_v_grouped_defaults_to_false`: the fork reads
    /// this array optionally too (`ml.get_arr(
    /// "prism.hadamard.inverse_weight_names", inverse_names, false)` at
    /// src_llama-model.cpp:1323, default-constructing an empty vector) — a
    /// folded model that folds weights but declares no inverse-lookup table
    /// is legitimate, not a contract violation.
    #[test]
    fn missing_inverse_weight_names_defaults_to_empty() {
        let pairs = without(valid_hadamard_pairs(), KEY_INVERSE_WEIGHT_NAMES);
        let metadata = build_metadata_store(pairs);
        let hadamard = HadamardConfig::from_metadata(&metadata)
            .expect("a missing inverse_weight_names must default, not error")
            .expect("version is present, so this must be Some");
        assert!(hadamard.inverse.is_empty());
        assert!(!hadamard.is_inverse("token_embd.weight"));
    }

    /// The default-on-absence above must not weaken the present-but-wrong-type
    /// case: `inverse_weight_names` written as a non-array is still a hard
    /// error.
    #[test]
    fn inverse_weight_names_present_but_wrong_type_is_still_a_hard_error() {
        let pairs = overridden(
            valid_hadamard_pairs(),
            KEY_INVERSE_WEIGHT_NAMES,
            MetadataWriteValue::U32(1),
        );
        let metadata = build_metadata_store(pairs);
        let err = HadamardConfig::from_metadata(&metadata)
            .expect_err("a present-but-malformed inverse_weight_names must still be a hard error");
        assert!(
            matches!(err, BonsaiError::HadamardContract { .. }),
            "{err:?}"
        );
    }

    // ── HadamardSpec alias ──────────────────────────────────────────────────

    #[test]
    fn hadamard_spec_alias_resolves_to_the_same_type() {
        let metadata = build_metadata_store(valid_hadamard_pairs());
        let via_spec: Option<HadamardSpec> =
            HadamardSpec::from_metadata(&metadata).expect("alias should parse identically");
        assert!(via_spec.is_some());
    }
}
