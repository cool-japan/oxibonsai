//! Shared helpers for HuggingFace / ONNX → GGUF conversion pipelines.
//!
//! Both the safetensors path (`convert::convert_hf_to_gguf`) and the ONNX path
//! (`convert::onnx::convert_onnx_to_gguf`) emit the same GGUF layout and share:
//!
//! * Parsing of the sibling `config.json`.
//! * Writing the `general.*`, `<arch>.*` and `tokenizer.ggml.*` blocks (via
//!   [`crate::convert::meta`] and [`crate::convert::tokenizer_meta`]).
//! * Deciding which GGUF tensor type each tensor gets, including the FP32
//!   carve-outs and the per-row block-alignment rule.
//! * A single [`ConvertStats`] result struct so callers can report progress
//!   uniformly.
//! * [`UnmappedReport`], which turns silently-dropped tensors into a hard
//!   error (CQ-04).

use std::path::Path;

use anyhow::Context;
use serde_json::Value;

use oxibonsai_core::gguf::writer::{GgufWriter, TensorType};
use oxibonsai_core::quant_ternary::{BlockTQ2_0_g128, BLOCK_TQ2_0_G128_BYTES};
use serde_json::Value as JsonValue;

use crate::convert::meta::{
    architecture_from_config_json, ggml_file_type, resolve_rope_theta, write_arch_metadata,
    write_general_metadata, ArchMetadata, GeneralMetadata,
};
use crate::convert::qwen35::is_never_quantized_qwen35;
use crate::convert::tokenizer_meta::{pre_tokenizer_for_arch, TokenizerMetadata};
use crate::export::keep_fp32_by_kind;
use crate::quantize::{row_is_block_aligned, SourceDtype};

/// Statistics returned after a successful conversion.
///
/// Shared between the safetensors (`convert_hf_to_gguf`) and ONNX
/// (`convert_onnx_to_gguf`) pipelines so CLI callers can report a single
/// struct regardless of source format.
#[derive(Debug, Clone, Default)]
pub struct ConvertStats {
    /// Total number of tensors written to the GGUF file.
    pub n_tensors: usize,
    /// Number of tensors quantized to the requested format.
    pub n_ternary: usize,
    /// Number of tensors stored as FP32.
    pub n_fp32: usize,
    /// Number of tensors stored verbatim as BF16 (e.g. MLX skip-pattern tensors).
    pub n_bf16: usize,
    /// Number of tensors stored verbatim as F16.
    pub n_f16: usize,
    /// Number of source tensors that had no mapping and were skipped.
    ///
    /// Non-zero only when the caller explicitly allowed unmapped tensors;
    /// otherwise the conversion fails instead (CQ-04).
    pub n_unmapped: usize,
    /// Total size of the output GGUF file in bytes.
    pub output_bytes: usize,
}

// ─── Completeness ─────────────────────────────────────────────────────────────

/// Collects source tensors the converter could not place in the output.
///
/// Both converters used to `continue` past an unrecognised tensor name at
/// `debug`/`warn` level and then report success, so a model with a renamed
/// layer silently produced a GGUF missing whole projections and exited 0
/// (CQ-04). There are two distinct ways that happened — an unmapped *name*
/// and an unsupported source *dtype* — and this type records both.
#[derive(Debug, Clone, Default)]
pub struct UnmappedReport {
    /// Source tensor names with no GGUF counterpart.
    pub unmapped_names: Vec<String>,
    /// Source tensors whose element type this crate cannot decode,
    /// as `(name, dtype label)`.
    pub unsupported_dtypes: Vec<(String, String)>,
    /// GGUF names that two different source tensors both mapped to.
    ///
    /// The pending-tensor tables are keyed by GGUF name, so a two-to-one
    /// mapping overwrites rather than skips — which no count-based check
    /// would ever notice.
    pub collisions: Vec<(String, String, String)>,
}

impl UnmappedReport {
    /// Record a source tensor with no GGUF counterpart.
    pub fn push_unmapped(&mut self, name: &str) {
        self.unmapped_names.push(name.to_string());
    }

    /// Record a source tensor whose dtype cannot be decoded.
    pub fn push_unsupported_dtype(&mut self, name: &str, dtype: &str) {
        self.unsupported_dtypes
            .push((name.to_string(), dtype.to_string()));
    }

    /// Record two source tensors competing for one GGUF name.
    pub fn push_collision(&mut self, gguf_name: &str, first: &str, second: &str) {
        self.collisions
            .push((gguf_name.to_string(), first.to_string(), second.to_string()));
    }

    /// Total number of source tensors that did not reach the output.
    pub fn count(&self) -> usize {
        self.unmapped_names.len() + self.unsupported_dtypes.len() + self.collisions.len()
    }

    /// Whether every source tensor was accounted for.
    pub fn is_complete(&self) -> bool {
        self.count() == 0
    }

    /// Fail unless the conversion was complete or the caller opted out.
    ///
    /// `allow_unmapped` corresponds to the CLI's `--allow-unmapped`.
    pub fn check(&self, allow_unmapped: bool) -> anyhow::Result<()> {
        if self.is_complete() || allow_unmapped {
            return Ok(());
        }
        anyhow::bail!("{}", self.describe());
    }

    /// A full, human-readable listing — never a truncated summary, because
    /// the whole point is to name every tensor that went missing.
    pub fn describe(&self) -> String {
        let mut out = format!(
            "conversion is incomplete: {} source tensor(s) did not reach the output \
             (pass --allow-unmapped to convert anyway)",
            self.count()
        );
        if !self.unmapped_names.is_empty() {
            out.push_str("\n  unmapped tensor names:");
            for name in &self.unmapped_names {
                out.push_str("\n    - ");
                out.push_str(name);
            }
        }
        if !self.unsupported_dtypes.is_empty() {
            out.push_str("\n  unsupported source dtypes:");
            for (name, dtype) in &self.unsupported_dtypes {
                out.push_str(&format!("\n    - {name} ({dtype})"));
            }
        }
        if !self.collisions.is_empty() {
            out.push_str("\n  two source tensors mapped to one GGUF name:");
            for (gguf, first, second) in &self.collisions {
                out.push_str(&format!("\n    - {gguf} <- {first} and {second}"));
            }
        }
        out
    }
}

// ─── config.json ──────────────────────────────────────────────────────────────

/// Read and parse `config.json` at the given path.
///
/// Callers are responsible for locating the file (the safetensors path uses
/// `from_dir.join("config.json")`; the ONNX path may need to search the ONNX
/// parent and grandparent directories).
pub fn read_config_json(config_path: &Path) -> anyhow::Result<Value> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {:?}", config_path))?;
    let value: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {:?}", config_path))?;
    Ok(value)
}

/// Reject a `config.json` whose `architectures` entry names a model family
/// this converter does not implement.
///
/// No converter previously looked at `architectures` at all, so pointing
/// `convert` at, say, a Llama checkpoint produced a file labelled `qwen3`
/// with Qwen3 tensor names and silently wrong dimensions.
pub fn check_supported_architecture(config: &Value) -> anyhow::Result<()> {
    const SUPPORTED: [&str; 6] = ["qwen2", "qwen3", "qwen35", "qwen3_next", "bonsai", "prism"];

    let listed: Vec<String> = config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    if listed.is_empty() {
        // `model_type` alone is enough for the older exports that omit
        // `architectures`; absence is not evidence of a wrong model.
        return Ok(());
    }
    let ok = listed.iter().any(|a| {
        let lower = a.to_ascii_lowercase();
        SUPPORTED.iter().any(|s| lower.contains(s))
    });
    if ok {
        return Ok(());
    }
    anyhow::bail!(
        "config.json declares architectures {:?}, none of which this converter implements \
         (supported families: {:?}); converting would produce a file labelled qwen3 with \
         Qwen3 tensor names and the wrong dimensions",
        listed,
        SUPPORTED
    )
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

/// Write the complete metadata block for a converted model.
///
/// Emits `general.*` (with `quantization_version` as a **U32**, never a
/// string), the `<arch>.*` hyper-parameters under the real architecture
/// prefix (never the non-existent `llm.*`), and — when a tokenizer is found
/// next to the source — the whole `tokenizer.ggml.*` vocabulary (CQ-08).
///
/// `source_dir` is where the tokenizer is looked for; pass the model
/// directory. `quant` is the target format string (`"pq2_0"`, `"q1_0_g128"`,
/// …), recorded as a non-normative label and reflected in
/// `general.file_type`.
pub fn write_metadata(
    writer: &mut GgufWriter<'_>,
    config: &Value,
    model_name: &str,
    quant: &str,
    source_dir: Option<&Path>,
) -> anyhow::Result<()> {
    let tensor_type = quant_format_tensor_type(quant)
        .ok_or_else(|| anyhow::anyhow!("unsupported quantisation format '{quant}'"))?;
    let architecture = architecture_from_config_json(config);

    let general = GeneralMetadata {
        architecture: architecture.clone(),
        name: model_name.to_string(),
        version: None,
        description: None,
        file_type: ggml_file_type(tensor_type),
        quant_format: Some(quant.to_ascii_uppercase()),
        scale_rule: Some("AbsMean".to_string()),
    };
    write_general_metadata(writer, &general);

    let arch_meta = ArchMetadata::from_config_json(config)?;
    write_arch_metadata(writer, &architecture, &arch_meta);

    // If the nested `rope_parameters` block indicates YaRN scaling, note it.
    if let Some(rp) = config.get("rope_parameters").and_then(Value::as_object) {
        let rope_type = rp.get("rope_type").and_then(Value::as_str).unwrap_or("");
        if rope_type.eq_ignore_ascii_case("yarn") {
            // Fully qualified: inside `tracing::info!`, a bare `Value` would
            // resolve to `tracing::Value`, the trait.
            let factor = rp.get("factor").and_then(serde_json::Value::as_f64);
            let original_max_pos = rp
                .get("original_max_position_embeddings")
                .and_then(serde_json::Value::as_u64);
            tracing::info!(
                ?factor,
                ?original_max_pos,
                "YaRN rope_parameters detected; relying on architecture defaults"
            );
        }
    }

    if let Some(dir) = source_dir {
        match TokenizerMetadata::discover(dir, pre_tokenizer_for_arch(&architecture)) {
            Ok(Some(tokenizer)) => {
                tracing::info!(
                    vocab_size = tokenizer.vocab_size(),
                    merges = tokenizer.merges.len(),
                    "embedding tokenizer into the GGUF"
                );
                tokenizer.write(writer);
            }
            Ok(None) => tracing::warn!(
                dir = ?dir,
                "no tokenizer.json found next to the source model; the converted GGUF will \
                 carry no tokenizer and will need an external one"
            ),
            Err(e) => return Err(anyhow::anyhow!("reading source tokenizer: {e}")),
        }
    }

    Ok(())
}

/// Map a converter `quant` string to its GGUF tensor type.
///
/// The two group-128 ternary spellings deliberately resolve to **different**
/// wire types. Their payload is bit-identical — the same `AbsMean` scale,
/// the same `0b00 / 0b01 / 0b10 → -1 / 0 / +1` codes, 34 bytes per 128
/// weights — and only the block byte order and the ggml id differ, so a
/// `tq2_0_g128` file is exactly a `pq2_0` file with every block rotated left
/// by two bytes, tensor type 142 → 42 and `general.file_type` 141 → 41:
///
/// * `"tq2_0_g128"` → [`TensorType::TQ2_0_g128`] — ggml id **42**, `qs`
///   first and `d` last, `general.file_type` 41. This is the **native**
///   layout the 0.2.4 runtime executes for a uniform `qwen3` model: the
///   per-layer and LM-head `LinearTernary` wrappers, their SIMD CPU kernels
///   and the fused Metal / CUDA full-layer ternary paths all consume
///   `BlockTQ2_0_g128`. It is what `scripts/download_ternary.sh` converts
///   the gen-1 Ternary-Bonsai 1.7B / 4B / 8B checkpoints to.
/// * `"pq2_0"` → [`TensorType::PQ2_0`] — ggml id **142**, `d` first,
///   `general.file_type` 141. This is the **llama.cpp-readable** choice
///   (core-gguf-02): `llama.cpp` validates each tensor offset against the
///   running padded sum and hard-errors on the 34-byte-at-id-42 files, and
///   the Prism fork prints a dedicated hint naming exactly them. The 0.2.4
///   runtime decodes `PQ2_0` and runs it per layer, but has no `PQ2_0`
///   output-projection (LM-head) wrapper and no fused GPU path for it, so a
///   `qwen3` model converted this way does not load in `run` / `chat` /
///   `serve`.
///
/// `"tq2_0_g128"` briefly resolved to `PQ2_0` as well (introduced after
/// v0.2.3), which made every freshly converted Ternary-Bonsai GGUF fail at
/// load time with "no output-projection Linear* kernel wrapper"; the
/// `tq2_0_g128_converts_to_native_id_42_and_loads` integration test pins
/// the native mapping.
pub fn quant_format_tensor_type(quant: &str) -> Option<TensorType> {
    match quant {
        "tq2_0_g128" => Some(TensorType::TQ2_0_g128),
        "pq2_0" => Some(TensorType::PQ2_0),
        "q1_0_g128" => Some(TensorType::Q1_0G128),
        "ptq1_0" => Some(TensorType::PTQ1_0),
        "q2_0_g64" => Some(TensorType::Q2_0G64),
        "f32" => Some(TensorType::F32),
        _ => None,
    }
}

/// Every `quant` string the converters accept, for help text and errors.
pub const SUPPORTED_QUANT_FORMATS: [&str; 6] = [
    "tq2_0_g128",
    "pq2_0",
    "ptq1_0",
    "q2_0_g64",
    "q1_0_g128",
    "f32",
];

// ─── Tensor typing ────────────────────────────────────────────────────────────

/// Decide the GGUF tensor type for one converted tensor.
///
/// Four rules, in order:
///
/// 1. A norm gain or any 1-D tensor stays in a float type — quantizing an
///    RMSNorm weight measured up to 910 % relative error (CQ-02). It is
///    written back in the dtype it arrived in, so a bf16 norm stays bf16
///    instead of being inflated 2× to f32 (CQ-M2).
/// 2. A `qwen35` hybrid scalar/conv weight that policy excludes from
///    quantization even though its shape passes rule 1 — e.g. the real
///    27B's `ssm_alpha.weight` / `ssm_beta.weight` are `[5120, 48]` BF16,
///    two-dimensional with a block-aligned `ne0` (CQ-06 residue). Applied
///    unconditionally rather than gated on the source architecture: these
///    exact tensor names never occur in any other supported architecture.
///    Also written back in its source dtype, like rule 1.
/// 3. A tensor whose first (fastest-varying) dimension is not a whole number
///    of blocks has no valid encoding in the target format, so it stays float
///    too — exactly what `llama.cpp`'s quantizer does. The converter used to
///    zero-pad the *flattened* tensor instead, which made every row after the
///    first straddle a group boundary (CQ-14).
/// 4. Otherwise, the requested quantized type.
pub fn tensor_type_for(
    name: &str,
    gguf_shape: &[u64],
    source_dtype: SourceDtype,
    target: TensorType,
    is_norm: bool,
) -> TensorType {
    let shape_usize: Vec<usize> = gguf_shape.iter().map(|&d| d as usize).collect();
    if is_norm || keep_fp32_by_kind(name, &shape_usize) || is_never_quantized_qwen35(name) {
        return unquantized_type(source_dtype, is_norm);
    }
    let ne0 = gguf_shape.first().copied().unwrap_or(0) as usize;
    if !row_is_block_aligned(ne0, target) {
        tracing::info!(
            tensor = name,
            ne0,
            block_size = target.block_size(),
            "first dimension is not a block multiple — keeping this tensor unquantized"
        );
        return unquantized_type(source_dtype, is_norm);
    }
    target
}

/// The float type an unquantized tensor is written as.
///
/// Norm gains are always widened to F32: they are tiny, and every consumer in
/// this workspace reads them as f32. Anything else keeps its source dtype.
fn unquantized_type(source_dtype: SourceDtype, is_norm: bool) -> TensorType {
    if is_norm {
        return TensorType::F32;
    }
    match source_dtype {
        SourceDtype::F32 => TensorType::F32,
        SourceDtype::F16 => TensorType::F16,
        SourceDtype::BF16 => TensorType::BF16,
    }
}

/// Count one written tensor into [`ConvertStats`].
pub fn record_tensor(stats: &mut ConvertStats, tensor_type: TensorType) {
    match tensor_type {
        TensorType::F32 => stats.n_fp32 += 1,
        TensorType::F16 => stats.n_f16 += 1,
        TensorType::BF16 => stats.n_bf16 += 1,
        _ => stats.n_ternary += 1,
    }
    stats.n_tensors += 1;
}

/// Serialise a slice of `BlockTQ2_0_g128` blocks into raw bytes.
///
/// Each block is 34 bytes: 32 bytes of packed `qs` + 2 bytes of FP16 `d`.
///
/// **This is the native qs-first (ggml id 42) layout and must never be used
/// for `PQ2_0`** — that type is `d`-first, so reusing this cast would emit
/// byte-swapped blocks which still pass every length check. Go through
/// [`crate::quantize::encode_quantized_tensor`] instead, which picks the
/// right block struct per tensor type.
///
/// # Safety
///
/// `BlockTQ2_0_g128` is `#[repr(C)]` with a compile-time size assertion of
/// exactly 34 bytes. The cast is safe because we size the output slice using
/// `blocks.len() * BLOCK_TQ2_0_G128_BYTES`.
pub fn blocks_to_bytes(blocks: &[BlockTQ2_0_g128]) -> Vec<u8> {
    let total = blocks.len() * BLOCK_TQ2_0_G128_BYTES;
    // SAFETY: repr(C) layout with compile-time size check; byte length verified.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, total) };
    bytes.to_vec()
}

/// Re-export so the ONNX path can resolve `rope_theta` the same way.
pub fn config_rope_theta(config: &JsonValue) -> f64 {
    resolve_rope_theta(config)
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

    /// `tq2_0_g128` is the runtime-native qs-first id 42; only `pq2_0` is
    /// the llama.cpp-readable d-first id 142. Mapping both spellings to 142
    /// produced Ternary-Bonsai GGUFs the 0.2.4 runtime refused to load.
    #[test]
    fn tq2_0_g128_resolves_to_native_id_42_and_pq2_0_to_142() {
        assert_eq!(
            quant_format_tensor_type("tq2_0_g128"),
            Some(TensorType::TQ2_0_g128)
        );
        assert_eq!(TensorType::TQ2_0_g128.wire_id(), 42);
        assert_eq!(TensorType::TQ2_0_g128.block_size(), 128);
        assert_eq!(TensorType::TQ2_0_g128.block_bytes(), 34);
        assert_eq!(quant_format_tensor_type("pq2_0"), Some(TensorType::PQ2_0));
        assert_eq!(TensorType::PQ2_0.wire_id(), 142);
        assert_eq!(quant_format_tensor_type("ptq1_0"), Some(TensorType::PTQ1_0));
        assert_eq!(
            quant_format_tensor_type("q2_0_g64"),
            Some(TensorType::Q2_0G64)
        );
        assert_eq!(TensorType::Q2_0G64.wire_id(), 42);
        assert_eq!(
            quant_format_tensor_type("q1_0_g128"),
            Some(TensorType::Q1_0G128)
        );
        assert_eq!(quant_format_tensor_type("f32"), Some(TensorType::F32));
        assert!(quant_format_tensor_type("nope").is_none());
        // Every advertised spelling must resolve, or the CLI help lies.
        for quant in SUPPORTED_QUANT_FORMATS {
            assert!(
                quant_format_tensor_type(quant).is_some(),
                "advertised quant format {quant:?} does not resolve"
            );
        }
    }

    /// The two group-128 ternary writers must agree on everything except the
    /// block byte order: same `AbsMean` scale, same codes. A `tq2_0_g128`
    /// tensor is therefore exactly the `pq2_0` tensor with each 34-byte block
    /// rotated left by two bytes (`[d][qs]` → `[qs][d]`).
    #[test]
    fn tq2_0_g128_and_pq2_0_payloads_are_a_two_byte_block_rotation() {
        use crate::quantize::{encode_quantized_tensor, ScaleRule};

        const GROUP: usize = 128;
        let ne0 = 4 * GROUP;
        let rows = 6;
        let mut data: Vec<f32> = (0..ne0 * rows)
            .map(|i| {
                let x = i as f32 * 0.137 + 0.25;
                x.sin() * (0.2 + (i % 11) as f32 * 0.05)
            })
            .collect();
        // An all-zero group (encoded as `d = 0`, codes all `0b01`).
        data[GROUP..2 * GROUP].fill(0.0);
        // An already-ternary group, which the AbsMean canonicaliser keeps
        // byte-for-byte (`{-0.5, 0, +0.5}`).
        for (j, w) in data[2 * GROUP..3 * GROUP].iter_mut().enumerate() {
            *w = [-0.5f32, 0.0, 0.5][j % 3];
        }
        // A group whose values sit exactly on the absmax/2 boundary.
        for (j, w) in data[5 * GROUP..6 * GROUP].iter_mut().enumerate() {
            *w = if j == 0 {
                1.0
            } else {
                0.5 * if j % 2 == 0 { 1.0 } else { -1.0 }
            };
        }

        let tq2 = encode_quantized_tensor(&data, ne0, TensorType::TQ2_0_g128, ScaleRule::AbsMean)
            .expect("encode TQ2_0_g128");
        let pq2 = encode_quantized_tensor(&data, ne0, TensorType::PQ2_0, ScaleRule::AbsMean)
            .expect("encode PQ2_0");
        assert_eq!(tq2.len(), pq2.len());
        assert_eq!(tq2.len(), data.len() / GROUP * 34);

        let (tq2_blocks, tq2_tail) = tq2.as_chunks::<34>();
        let (pq2_blocks, pq2_tail) = pq2.as_chunks::<34>();
        assert!(tq2_tail.is_empty() && pq2_tail.is_empty());
        for (block, (t, p)) in tq2_blocks.iter().zip(pq2_blocks).enumerate() {
            assert_eq!(&t[..32], &p[2..], "block {block}: codes differ");
            assert_eq!(&t[32..], &p[..2], "block {block}: scale differs");
            assert!(
                t[..32]
                    .iter()
                    .all(|b| (0..4u32).all(|lane| (b >> (2 * lane)) & 0x03 != 0x03)),
                "block {block}: the reserved 0b11 code must never be emitted"
            );
        }
    }

    #[test]
    fn metadata_is_spec_typed_and_arch_prefixed() {
        let mut w = GgufWriter::new();
        write_metadata(&mut w, &minimal_config(), "unit", "tq2_0_g128", None)
            .expect("write metadata");
        let bytes = w.to_bytes().expect("serialise");
        let gguf = GgufFile::parse(&bytes).expect("parse");

        assert_eq!(
            gguf.metadata.get_u32("general.quantization_version").ok(),
            Some(2)
        );
        // PrismML MOSTLY_Q2_0 — the dominant type is the native id-42 layout.
        assert_eq!(gguf.metadata.get_u32("general.file_type").ok(), Some(41));
        assert_eq!(
            gguf.metadata.get_string("oxibonsai.quant_format").ok(),
            Some("TQ2_0_G128")
        );
        assert_eq!(
            gguf.metadata.get_string("oxibonsai.quant_scale_rule").ok(),
            Some("AbsMean")
        );
        assert_eq!(gguf.metadata.get_u32("qwen3.block_count").ok(), Some(2));
        assert!(gguf.metadata.get_u32("llm.block_count").is_err());

        // `pq2_0` keeps the PrismML MOSTLY_PQ2_0 file type.
        let mut w = GgufWriter::new();
        write_metadata(&mut w, &minimal_config(), "unit", "pq2_0", None)
            .expect("write pq2_0 metadata");
        let bytes = w.to_bytes().expect("serialise pq2_0");
        let gguf = GgufFile::parse(&bytes).expect("parse pq2_0");
        assert_eq!(gguf.metadata.get_u32("general.file_type").ok(), Some(141));
    }

    #[test]
    fn norms_and_one_dimensional_tensors_stay_float() {
        assert_eq!(
            tensor_type_for(
                "blk.0.attn_norm.weight",
                &[128],
                SourceDtype::F32,
                TensorType::PQ2_0,
                true
            ),
            TensorType::F32
        );
        assert_eq!(
            tensor_type_for(
                "blk.0.ssm_a",
                &[48],
                SourceDtype::F32,
                TensorType::PQ2_0,
                false
            ),
            TensorType::F32,
            "a 1-D tensor is never quantized, whatever it is called"
        );
    }

    #[test]
    fn qwen35_ssm_scalars_stay_unquantized_even_when_block_aligned() {
        // ssm_alpha.weight / ssm_beta.weight are [5120, 48] BF16 in the real
        // 27B: ne0 = 5120 is a multiple of every block size in this crate
        // (5120 / 128 = 40), so without the qwen35 never-quantized policy
        // this would silently ternarize the gated-delta-net decay/gate
        // scalars.
        assert_eq!(
            tensor_type_for(
                "blk.0.ssm_alpha.weight",
                &[5120, 48],
                SourceDtype::BF16,
                TensorType::PQ2_0,
                false
            ),
            TensorType::BF16,
            "qwen35's never-quantized ssm scalars must keep their source dtype"
        );
        assert_eq!(
            tensor_type_for(
                "blk.3.ssm_beta.weight",
                &[5120, 48],
                SourceDtype::BF16,
                TensorType::PQ2_0,
                false
            ),
            TensorType::BF16
        );
        // A genuinely quantizable tensor of the same block-aligned shape is
        // unaffected.
        assert_eq!(
            tensor_type_for(
                "blk.0.ffn_gate.weight",
                &[5120, 48],
                SourceDtype::F32,
                TensorType::PQ2_0,
                false
            ),
            TensorType::PQ2_0
        );
    }

    #[test]
    fn bf16_source_keeps_bf16_when_unquantized() {
        assert_eq!(
            tensor_type_for(
                "blk.0.ssm_alpha.weight",
                &[48],
                SourceDtype::BF16,
                TensorType::PQ2_0,
                false
            ),
            TensorType::BF16,
            "a bf16 tensor must not be inflated 2x to f32 (CQ-M2)"
        );
    }

    #[test]
    fn non_block_aligned_rows_are_not_quantized() {
        // 130 is not a multiple of 128: padding the flattened tensor would
        // make rows 2..n straddle group boundaries (CQ-14).
        assert_eq!(
            tensor_type_for(
                "blk.0.ffn_up.weight",
                &[130, 4],
                SourceDtype::F32,
                TensorType::PQ2_0,
                false
            ),
            TensorType::F32
        );
        assert_eq!(
            tensor_type_for(
                "blk.0.ffn_up.weight",
                &[128, 4],
                SourceDtype::F32,
                TensorType::PQ2_0,
                false
            ),
            TensorType::PQ2_0
        );
    }

    #[test]
    fn unmapped_report_names_every_missing_tensor() {
        let mut r = UnmappedReport::default();
        assert!(r.is_complete());
        r.push_unmapped("model.layers.0.mystery.weight");
        r.push_unsupported_dtype("model.layers.0.packed.weight", "I8");
        r.push_collision("blk.0.attn_q.weight", "a.weight", "b.weight");
        assert_eq!(r.count(), 3);
        assert!(r.check(false).is_err());
        r.check(true).expect("--allow-unmapped converts anyway");

        let text = r.describe();
        assert!(text.contains("model.layers.0.mystery.weight"));
        assert!(text.contains("packed.weight"));
        assert!(text.contains("I8"));
        assert!(text.contains("blk.0.attn_q.weight <- a.weight and b.weight"));
    }

    #[test]
    fn architecture_gate_rejects_foreign_models() {
        let mut cfg = minimal_config();
        cfg["architectures"] = json!(["LlamaForCausalLM"]);
        assert!(check_supported_architecture(&cfg).is_err());

        cfg["architectures"] = json!(["Qwen3ForCausalLM"]);
        check_supported_architecture(&cfg).expect("qwen3 is supported");

        // Absent `architectures` is not evidence of a wrong model.
        let cfg = minimal_config();
        check_supported_architecture(&cfg).expect("absent key is allowed");
    }

    #[test]
    fn stats_count_by_written_type() {
        let mut stats = ConvertStats::default();
        record_tensor(&mut stats, TensorType::F32);
        record_tensor(&mut stats, TensorType::BF16);
        record_tensor(&mut stats, TensorType::PQ2_0);
        assert_eq!(stats.n_fp32, 1);
        assert_eq!(stats.n_bf16, 1);
        assert_eq!(stats.n_ternary, 1);
        assert_eq!(stats.n_tensors, 3);
    }

    #[test]
    fn rope_theta_resolution_is_shared_with_meta() {
        assert_eq!(
            config_rope_theta(&json!({"rope_theta": 500000.0})),
            500000.0
        );
        assert_eq!(config_rope_theta(&json!({})), 10000.0);
    }
}
