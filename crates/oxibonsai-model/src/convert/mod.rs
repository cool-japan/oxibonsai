//! HuggingFace safetensors → OxiBonsai GGUF conversion.
//!
//! Converts a HuggingFace model directory (containing `model.safetensors` or
//! sharded safetensors files and `config.json`) into an OxiBonsai GGUF file:
//! `TQ2_0_g128` (ggml id 42, `qs` first — the default group-128 ternary
//! format, the layout the runtime executes), PrismML `PQ2_0` (142, the same
//! ternary data `d` first, for llama.cpp interoperability), `PTQ1_0` (143),
//! mainline `Q2_0` (group 64) or `Q1_0_g128` for weight tensors, with norm
//! and 1-D tensors kept in a float type. See
//! [`common::quant_format_tensor_type`] for why the two group-128 ternary
//! spellings map to different ids.
//!
//! A sibling [`onnx`] module provides the same output format from HuggingFace
//! MatMulNBits-quantized ONNX models (e.g. `onnx-community/Ternary-Bonsai-1.7B-ONNX`).
//! Shared helpers live in [`common`]; the metadata writers live in
//! [`meta`] (`general.*` / `<arch>.*`), [`tokenizer_meta`]
//! (`tokenizer.ggml.*`) and [`qwen35`] (`qwen35.*` + `prism.hadamard.*`).
//!
//! A conversion is **complete or it fails**: a source tensor the converter
//! cannot place is reported by name rather than skipped at `debug!` level
//! behind a success message (CQ-04). Pass `allow_unmapped` to
//! [`convert_hf_to_gguf_with_options`] to convert anyway.
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use oxibonsai_model::convert::convert_hf_to_gguf;
//!
//! let stats = convert_hf_to_gguf(
//!     Path::new("/path/to/Ternary-Bonsai-1.7B-unpacked"),
//!     Path::new("/path/to/output.gguf"),
//!     "tq2_0_g128",
//! ).expect("conversion failed");
//!
//! println!("Converted {} tensors", stats.n_tensors);
//! ```

pub mod common;
pub mod meta;
pub mod mlx_image;
pub mod name_map;
pub mod onnx;
pub mod qwen35;
pub mod tokenizer_meta;

use std::collections::BTreeMap;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::Context;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry};

use crate::convert::common::{
    check_supported_architecture, read_config_json, write_metadata, UnmappedReport,
};
use crate::convert::name_map::hf_to_gguf_name;
use crate::quantize::{dequant_source_bytes, encode_quantized_tensor, ScaleRule, SourceDtype};

pub use crate::convert::common::ConvertStats;

/// Map a safetensors dtype to the shared [`SourceDtype`].
///
/// Returns `None` for the packed integer layouts (I8/U8/F8/U32) used by
/// MLX / AWQ / GPTQ exports, which this converter cannot decode. The caller
/// records the name in an [`UnmappedReport`] rather than warning and
/// continuing: the old `_ => vec![]` arm dropped every such tensor at warn
/// level and still exited 0 (CQ-04).
fn source_dtype_of(dtype: Dtype) -> Option<SourceDtype> {
    match dtype {
        Dtype::F32 => Some(SourceDtype::F32),
        Dtype::F16 => Some(SourceDtype::F16),
        Dtype::BF16 => Some(SourceDtype::BF16),
        _ => None,
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Convert a HuggingFace safetensors model directory to an OxiBonsai GGUF file.
///
/// # Arguments
///
/// * `from_dir` — Directory containing `model.safetensors` (or sharded files
///   plus `model.safetensors.index.json`) and `config.json`.
/// * `to_path` — Destination path for the GGUF file.
/// * `quant` — Quantisation format; see [`common::SUPPORTED_QUANT_FORMATS`].
///   `"tq2_0_g128"` emits the runtime-native `TQ2_0_g128` (ggml id 42,
///   `qs` first, `general.file_type` 41), which `oxibonsai run` / `chat` /
///   `serve` load and execute on every tier; `"pq2_0"` emits the same
///   ternary data as PrismML `PQ2_0` (ggml id 142, `d` first, file type
///   141), the layout llama.cpp can read but whose `qwen3` LM head the 0.2.4
///   runtime cannot execute (see [`common::quant_format_tensor_type`]).
///   Norm and 1-D tensors are kept unquantized regardless of format.
///
/// Equivalent to [`convert_hf_to_gguf_with_options`] with
/// `allow_unmapped = false`.
///
/// # Errors
///
/// Returns an error if the directory does not contain the expected files, if
/// `quant` names an unsupported format, if any source tensor could not be
/// placed in the output (see [`ConvertStats::n_unmapped`]), or if the output
/// file cannot be written.
pub fn convert_hf_to_gguf(
    from_dir: &Path,
    to_path: &Path,
    quant: &str,
) -> anyhow::Result<ConvertStats> {
    convert_hf_to_gguf_with_options(from_dir, to_path, quant, false)
}

/// Convert a HuggingFace safetensors model directory to an OxiBonsai GGUF
/// file, optionally tolerating source tensors that have no mapping.
///
/// `allow_unmapped` corresponds to the CLI's `--allow-unmapped`: without it a
/// source tensor the converter cannot place is a hard error naming every such
/// tensor, instead of a `debug!` line and a success message (CQ-04).
pub fn convert_hf_to_gguf_with_options(
    from_dir: &Path,
    to_path: &Path,
    quant: &str,
    allow_unmapped: bool,
) -> anyhow::Result<ConvertStats> {
    let target_type = common::quant_format_tensor_type(quant).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported quantisation format '{}'; supported formats are {:?}",
            quant,
            common::SUPPORTED_QUANT_FORMATS
        )
    })?;

    // ── 1. Read config.json ──────────────────────────────────────────────────
    let config = read_config_json(&from_dir.join("config.json"))?;
    check_supported_architecture(&config)?;

    // ── 2. Collect shard paths ───────────────────────────────────────────────
    let shard_paths = discover_shard_paths(from_dir)?;

    // ── 3. Memory-map shards (no copying into RAM) ───────────────────────────
    // Using mmap avoids loading the entire safetensors file (≥3 GB) into RAM.
    let shard_files: Vec<std::fs::File> = shard_paths
        .iter()
        .map(|p| std::fs::File::open(p).with_context(|| format!("opening shard {:?}", p)))
        .collect::<anyhow::Result<_>>()?;
    let shard_mmaps: Vec<Mmap> = shard_files
        .iter()
        .map(|f| unsafe { Mmap::map(f) }.with_context(|| "memory-mapping shard failed"))
        .collect::<anyhow::Result<_>>()?;

    // Parse SafeTensors views from mmap'd bytes.
    let parsed_shards: Vec<SafeTensors<'_>> = shard_mmaps
        .iter()
        .enumerate()
        .map(|(i, m)| {
            SafeTensors::deserialize(m.as_ref())
                .with_context(|| format!("parsing shard {:?}", shard_paths[i]))
        })
        .collect::<anyhow::Result<_>>()?;

    // Collect (hf_name → shard_index) for all tensors across shards, in a
    // deterministic order so the completeness report reads the same way twice.
    let mut name_to_shard: BTreeMap<&str, usize> = BTreeMap::new();
    for (shard_idx, shard) in parsed_shards.iter().enumerate() {
        for name in shard.names() {
            name_to_shard.insert(name, shard_idx);
        }
    }

    // ── 4. Build GGUF writer ─────────────────────────────────────────────────
    let mut writer = GgufWriter::new();

    // Model name derived from directory basename.
    let model_name = from_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    write_metadata(&mut writer, &config, model_name, quant, Some(from_dir))?;

    // ── 5. Determine tied-embedding flag ────────────────────────────────────
    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // ── 6. Collect tensor metadata only (no f32 data) ───────────────────────
    // Storing only names/shapes avoids accumulating gigabytes of f32 data.
    // Sort by GGUF name to get a canonical ordering (blk.0 before blk.1 etc.).
    let mut meta_entries: BTreeMap<String, TensorMetaOnly> = BTreeMap::new();
    let mut report = UnmappedReport::default();

    for (hf_name, &shard_idx) in &name_to_shard {
        let mapped = match hf_to_gguf_name(hf_name) {
            Some(m) => m,
            None => {
                report.push_unmapped(hf_name);
                continue;
            }
        };

        let shard = &parsed_shards[shard_idx];
        let view = shard
            .tensor(hf_name)
            .with_context(|| format!("tensor '{}' not found in shard", hf_name))?;

        let Some(source_dtype) = source_dtype_of(view.dtype()) else {
            // I8 / U8 / F8 / U32 — the packed MLX, AWQ and GPTQ layouts. The
            // old code returned an empty `Vec<f32>` here, warned, and carried
            // on to a successful exit (CQ-04).
            report.push_unsupported_dtype(hf_name, &format!("{:?}", view.dtype()));
            continue;
        };

        let shape_hf = view.shape();
        // GGUF shape = reversed HF shape (first dimension fastest-varying).
        let gguf_shape: Vec<u64> = shape_hf.iter().rev().map(|&d| d as u64).collect();

        if let Some(existing) = meta_entries.get(&mapped.gguf_name) {
            // Keyed by GGUF name, so a two-to-one mapping would overwrite
            // rather than skip — invisible to any count-based check.
            report.push_collision(&mapped.gguf_name, &existing.hf_name, hf_name);
            continue;
        }

        meta_entries.insert(
            mapped.gguf_name.clone(),
            TensorMetaOnly {
                gguf_name: mapped.gguf_name,
                is_norm: mapped.is_norm,
                gguf_shape,
                hf_name: hf_name.to_string(),
                shard_idx,
                source_dtype,
            },
        );
    }

    report.check(allow_unmapped)?;

    // ── 7. Handle tied embeddings (metadata only) ────────────────────────────
    // If tie_word_embeddings is true and output.weight is absent, duplicate
    // token_embd.weight as output.weight (the loader hard-requires it).
    if tie_word_embeddings && !meta_entries.contains_key("output.weight") {
        if let Some(embed_meta) = meta_entries.get("token_embd.weight") {
            let duplicated = TensorMetaOnly {
                gguf_name: "output.weight".to_string(),
                is_norm: false,
                gguf_shape: embed_meta.gguf_shape.clone(),
                hf_name: embed_meta.hf_name.clone(),
                shard_idx: embed_meta.shard_idx,
                source_dtype: embed_meta.source_dtype,
            };
            tracing::info!("tie_word_embeddings=true: duplicating token_embd as output.weight");
            meta_entries.insert("output.weight".to_string(), duplicated);
        }
    }

    // ── 8. Quantize one tensor at a time (no accumulation of f32 data) ───────
    // Each f32_data Vec is dropped at the end of its loop iteration, so peak
    // memory = (largest single tensor as f32) + accumulated quantised output.
    //
    // The source is continuous HuggingFace weights, so the ternary/1-bit
    // encoders use the error-minimising `AbsMean` scale rather than absmax,
    // which on dense weights inflates every layer output by ~3.5x (CQ-03).
    // Already-ternary groups are detected and keep the absmax encoding, so a
    // ternary checkpoint still round-trips bit-identically.
    let mut stats = ConvertStats {
        n_unmapped: report.count(),
        ..ConvertStats::default()
    };

    for meta in meta_entries.values() {
        let view = parsed_shards[meta.shard_idx]
            .tensor(&meta.hf_name)
            .with_context(|| format!("tensor '{}' not found in shard", meta.hf_name))?;

        let (_, f32_data) = decode_tensor(view.dtype(), view.data())
            .with_context(|| format!("decoding tensor '{}' ({:?})", meta.hf_name, view.dtype()))?;

        let tensor_type = common::tensor_type_for(
            &meta.gguf_name,
            &meta.gguf_shape,
            meta.source_dtype,
            target_type,
            meta.is_norm,
        );
        let ne0 = meta.gguf_shape.first().copied().unwrap_or(0) as usize;
        let raw_bytes = encode_quantized_tensor(&f32_data, ne0, tensor_type, ScaleRule::AbsMean)
            .map_err(|e| anyhow::anyhow!("{}", e.with_tensor(&meta.gguf_name)))?;
        // f32_data is dropped here (end of binding scope)
        drop(f32_data);

        println!(
            "  converting {} {:?} -> {:?}",
            meta.gguf_name, meta.gguf_shape, tensor_type
        );

        writer.add_tensor(TensorEntry {
            name: meta.gguf_name.clone(),
            shape: meta.gguf_shape.clone(),
            tensor_type,
            data: raw_bytes,
        });

        common::record_tensor(&mut stats, tensor_type);
    }

    // ── 9. Write GGUF file ───────────────────────────────────────────────────
    let out_file = std::fs::File::create(to_path)
        .with_context(|| format!("creating output file {:?}", to_path))?;
    let mut buf_writer = BufWriter::new(out_file);
    let bytes_written = writer
        .write(&mut buf_writer)
        .map_err(|e| anyhow::anyhow!("GGUF write error: {}", e))?;
    buf_writer
        .into_inner()
        .map_err(|e| anyhow::anyhow!("flushing output file: {}", e))?
        .sync_all()
        .with_context(|| format!("flushing {:?}", to_path))?;

    if stats.n_unmapped > 0 {
        println!(
            "  WARNING: {} source tensor(s) were skipped (--allow-unmapped):\n{}",
            stats.n_unmapped,
            report.describe()
        );
    }

    stats.output_bytes = bytes_written;
    Ok(stats)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Tensor metadata collected in pass 1 (no f32 data stored).
struct TensorMetaOnly {
    gguf_name: String,
    is_norm: bool,
    gguf_shape: Vec<u64>,
    /// Source HuggingFace tensor name (needed to retrieve data from the shard).
    hf_name: String,
    /// Index into `parsed_shards` where this tensor lives.
    shard_idx: usize,
    /// Element type the tensor had in the safetensors file (CQ-M2).
    source_dtype: SourceDtype,
}

/// Discover shard file paths from the model directory.
///
/// Prefers a single `model.safetensors`; falls back to the shards listed in
/// `model.safetensors.index.json`.
fn discover_shard_paths(from_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let single = from_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = from_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        anyhow::bail!(
            "neither model.safetensors nor model.safetensors.index.json found in {:?}",
            from_dir
        );
    }

    let raw = std::fs::read_to_string(&index_path)
        .with_context(|| format!("reading {:?}", index_path))?;
    let index: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {:?}", index_path))?;

    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .with_context(|| format!("missing 'weight_map' in {:?}", index_path))?;

    // Collect unique shard filenames, preserving first-seen order.
    let mut shard_names: Vec<String> = Vec::new();
    for file_name in weight_map.values() {
        if let Some(s) = file_name.as_str() {
            if !shard_names.contains(&s.to_string()) {
                shard_names.push(s.to_string());
            }
        }
    }
    shard_names.sort(); // canonical ordering

    let paths: Vec<PathBuf> = shard_names.iter().map(|name| from_dir.join(name)).collect();

    Ok(paths)
}

/// Convert raw safetensors bytes to `f32`, carrying the source dtype through.
///
/// Replaces the old `to_f32_vec`, which returned an **empty vec** for every
/// unsupported dtype and dropped the source type on the floor (CQ-M2/CQ-04):
/// the caller could neither tell an unsupported tensor from a genuinely empty
/// one, nor choose an output encoding that preserved a bf16 source. The
/// decode itself now lives in [`crate::quantize::dequant_source_bytes`],
/// which is also the crate's bf16 read path (CQ-M3).
fn decode_tensor(dtype: Dtype, data: &[u8]) -> Option<(SourceDtype, Vec<f32>)> {
    let source = source_dtype_of(dtype)?;
    let values = dequant_source_bytes(source, data).ok()?;
    Some((source, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_dtypes_round_trip_and_carry_their_type() {
        let f32_bytes: Vec<u8> = [1.0_f32, -2.5, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let (dtype, values) = decode_tensor(Dtype::F32, &f32_bytes).expect("f32 decodes");
        assert_eq!(dtype, SourceDtype::F32);
        assert_eq!(values, vec![1.0, -2.5, 0.0]);

        let bf16_bytes: Vec<u8> = [1.0_f32, -2.5]
            .iter()
            .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
            .collect();
        let (dtype, values) = decode_tensor(Dtype::BF16, &bf16_bytes).expect("bf16 decodes");
        assert_eq!(
            dtype,
            SourceDtype::BF16,
            "the bf16 read path exists (CQ-M3)"
        );
        assert_eq!(values, vec![1.0, -2.5]);
    }

    #[test]
    fn packed_integer_dtypes_are_reported_not_silently_dropped() {
        assert!(source_dtype_of(Dtype::I8).is_none());
        assert!(source_dtype_of(Dtype::U8).is_none());
        assert!(source_dtype_of(Dtype::I32).is_none());
        assert!(
            decode_tensor(Dtype::I8, &[0u8; 8]).is_none(),
            "an MLX/AWQ/GPTQ packed tensor must surface as unsupported, not as an empty vec"
        );
    }
}
