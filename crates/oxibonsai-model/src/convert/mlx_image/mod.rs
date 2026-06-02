//! MLX-packed FLUX.2 DiT transformer (safetensors) → OxiBonsai GGUF.
//!
//! Converts the **DiT (diffusion transformer) only** from a PrismML
//! `bonsai-image-ternary-*-mlx-2bit` checkpoint into a single OxiBonsai GGUF
//! file. The VAE and text-encoder converters are deliberately out of scope for
//! this phase; this module is structured so they can be added as siblings under
//! [`crate::convert`] later without disturbing the DiT path.
//!
//! # What it does
//!
//! The MLX checkpoint stores each quantized linear as three sub-tensors:
//!
//! ```text
//! <module>.weight   u32 [out, in/16]    16 two-bit codes per word, LE, LSB-first
//! <module>.scales   bf16[out, in/128]   per-group affine scale
//! <module>.biases   bf16[out, in/128]   per-group affine bias (== -scale, ternary)
//! ```
//!
//! For a ternary solver (`q ∈ {0,1,2}`, `bias == -scale`) the dequantized
//! weight is `w = scale·(q-1) ∈ {-s, 0, +s}`, which maps bit-identically to
//! OxiBonsai's [`BlockTQ2_0_g128`]. The actual packing (and its parity guards)
//! lives in [`pack::pack_quantized_module`]; this module handles safetensors
//! I/O, module classification, and GGUF emission.
//!
//! Every other tensor (the "skip-pattern" set: `proj_out`, `x_embedder`,
//! `context_embedder`, `time_text_embed`, `time_guidance_embed`, `norm_out`,
//! `*_modulation`, all `*.norm_q/norm_k/norm_added_q/norm_added_k`, and any
//! remaining bf16 tensor) is stored faithfully as **BF16** (GGUF type 30),
//! preserving the model's native precision exactly.
//!
//! # GGUF tensor names and shapes
//!
//! * Names follow the diffusers convention. A quantized module's GGUF tensor
//!   name is the base module path (the `.weight`/`.scales`/`.biases` suffix is
//!   stripped), e.g. `single_transformer_blocks.0.attn.to_qkv_mlp_proj`. Plain
//!   tensors keep their full safetensors name.
//! * Shapes use the OxiBonsai/GGUF convention of *reversed* safetensors shape
//!   (outermost dimension last). A quantized linear `[out, in]` is written as
//!   GGUF shape `[in, out]` so `ne[0] = in` is the (128-divisible) contraction
//!   dimension. The block byte stream is out-major (`row*(in/128)+group`),
//!   matching the GEMM kernel and independent of the shape label.
//!
//! # Usage
//!
//! ```no_run
//! use std::path::Path;
//! use oxibonsai_model::convert::mlx_image::convert_mlx_image_to_gguf;
//!
//! let stats = convert_mlx_image_to_gguf(
//!     Path::new("/path/to/diffusion_pytorch_model.safetensors"),
//!     Path::new("/path/to/bonsai-image-dit.gguf"),
//!     "tq2_0_g128",
//! ).expect("conversion failed");
//! println!("wrote {} tensors", stats.n_tensors);
//! ```

pub mod error;
pub mod metadata;
pub mod pack;

use std::collections::BTreeSet;
use std::io::BufWriter;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use oxibonsai_core::gguf::writer::{GgufWriter, TensorEntry, TensorType};

use crate::convert::common::{blocks_to_bytes, ConvertStats};

pub use self::error::{MlxImageImportError, PackError, PackError as MlxImagePackError};
use self::metadata::write_dit_metadata;
pub use self::metadata::DitArch;
use self::pack::pack_quantized_module;

/// Suffix marking the packed-weight sub-tensor of a quantized module.
const SUFFIX_WEIGHT: &str = ".weight";
/// Suffix marking the per-group scale sub-tensor of a quantized module.
const SUFFIX_SCALES: &str = ".scales";
/// Suffix marking the per-group bias sub-tensor of a quantized module.
const SUFFIX_BIASES: &str = ".biases";

/// Convert an MLX-packed FLUX.2 DiT safetensors file to an OxiBonsai GGUF file.
///
/// # Arguments
///
/// * `from_path` — the `diffusion_pytorch_model.safetensors` (DiT transformer).
/// * `to_path` — destination path for the GGUF file.
/// * `quant` — target quantisation format; only `"tq2_0_g128"` is supported.
///
/// # Errors
///
/// Returns [`MlxImageImportError`] on I/O, parse, shape, or parity-guard
/// failure (a quantized module whose codes or bias break the ternary
/// assumption aborts the conversion rather than producing a wrong file).
pub fn convert_mlx_image_to_gguf(
    from_path: &Path,
    to_path: &Path,
    quant: &str,
) -> Result<ConvertStats, MlxImageImportError> {
    convert_mlx_image_to_gguf_with_arch(from_path, to_path, quant, &DitArch::default())
}

/// Convert with an explicit [`DitArch`] (for non-default DiT configurations).
///
/// See [`convert_mlx_image_to_gguf`] for the common case.
pub fn convert_mlx_image_to_gguf_with_arch(
    from_path: &Path,
    to_path: &Path,
    quant: &str,
    arch: &DitArch,
) -> Result<ConvertStats, MlxImageImportError> {
    if quant != "tq2_0_g128" {
        return Err(MlxImageImportError::UnsupportedQuant(quant.to_string()));
    }

    // ── 1. Memory-map and parse the safetensors container ────────────────────
    let file = std::fs::File::open(from_path).map_err(|source| MlxImageImportError::Io {
        path: from_path.to_path_buf(),
        source,
    })?;
    // SAFETY: read-only mapping; the file is not modified while mapped (standard
    // model-loading pattern, mirroring the HF safetensors path).
    let mmap = unsafe { Mmap::map(&file) }.map_err(|source| MlxImageImportError::Io {
        path: from_path.to_path_buf(),
        source,
    })?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| MlxImageImportError::Parse {
        path: from_path.to_path_buf(),
        msg: e.to_string(),
    })?;

    // ── 2. Classify: quantized modules first, then leftover plain tensors ─────
    // A module is "quantized" iff it has all three of `.weight`(U32),
    // `.scales`(BF16), `.biases`(BF16). We record the base names and the exact
    // set of consumed sub-tensor names so plain tensors are *everything else*.
    let all_names: Vec<&str> = st.names();

    let mut quant_modules: BTreeSet<String> = BTreeSet::new();
    for name in &all_names {
        if let Some(base) = name.strip_suffix(SUFFIX_WEIGHT) {
            let scales = format!("{base}{SUFFIX_SCALES}");
            let biases = format!("{base}{SUFFIX_BIASES}");
            let has_scales = st.tensor(&scales).is_ok();
            let has_biases = st.tensor(&biases).is_ok();
            if has_scales && has_biases {
                // Confirm the weight is U32 — otherwise it's a plain `.weight`.
                let w_view = st.tensor(name).map_err(|e| MlxImageImportError::Parse {
                    path: from_path.to_path_buf(),
                    msg: e.to_string(),
                })?;
                if w_view.dtype() == Dtype::U32 {
                    quant_modules.insert(base.to_string());
                }
            }
        }
    }

    // Names consumed by quantized modules (so they are not re-emitted as plain).
    let mut consumed: BTreeSet<String> = BTreeSet::new();
    for base in &quant_modules {
        consumed.insert(format!("{base}{SUFFIX_WEIGHT}"));
        consumed.insert(format!("{base}{SUFFIX_SCALES}"));
        consumed.insert(format!("{base}{SUFFIX_BIASES}"));
    }

    // ── 3. Build the GGUF writer with architecture metadata ──────────────────
    let mut writer = GgufWriter::new();
    let model_name = from_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("bonsai-image-dit");
    write_dit_metadata(&mut writer, arch, model_name);

    let mut stats = ConvertStats::default();

    // ── 4. Emit quantized modules (sorted for canonical ordering) ────────────
    for base in &quant_modules {
        let entry = build_quantized_entry(&st, from_path, base)?;
        writer.add_tensor(entry);
        stats.n_ternary += 1;
        stats.n_tensors += 1;
    }

    // ── 5. Emit plain (BF16) tensors — everything not consumed above ─────────
    // Sorted for canonical ordering.
    let mut plain_names: Vec<&str> = all_names
        .iter()
        .copied()
        .filter(|n| !consumed.contains(*n))
        .collect();
    plain_names.sort_unstable();

    for name in plain_names {
        let entry = build_plain_entry(&st, from_path, name)?;
        match entry.tensor_type {
            TensorType::BF16 => stats.n_bf16 += 1,
            TensorType::F16 => stats.n_f16 += 1,
            _ => {}
        }
        writer.add_tensor(entry);
        stats.n_tensors += 1;
    }

    // ── 6. Write the GGUF file ───────────────────────────────────────────────
    let out_file = std::fs::File::create(to_path).map_err(|source| MlxImageImportError::Io {
        path: to_path.to_path_buf(),
        source,
    })?;
    let mut buf_writer = BufWriter::new(out_file);
    let bytes_written = writer
        .write(&mut buf_writer)
        .map_err(|e| MlxImageImportError::GgufWrite(e.to_string()))?;

    stats.output_bytes = bytes_written;
    Ok(stats)
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Build the GGUF tensor entry for one quantized module.
fn build_quantized_entry(
    st: &SafeTensors<'_>,
    from_path: &Path,
    base: &str,
) -> Result<TensorEntry, MlxImageImportError> {
    let weight_name = format!("{base}{SUFFIX_WEIGHT}");
    let scales_name = format!("{base}{SUFFIX_SCALES}");
    let biases_name = format!("{base}{SUFFIX_BIASES}");

    let w_view = st
        .tensor(&weight_name)
        .map_err(|e| parse_err(from_path, &e))?;
    let s_view = st
        .tensor(&scales_name)
        .map_err(|e| parse_err(from_path, &e))?;
    let b_view = st
        .tensor(&biases_name)
        .map_err(|e| parse_err(from_path, &e))?;

    // dtype checks (the validated MLX layout).
    if w_view.dtype() != Dtype::U32 {
        return Err(MlxImageImportError::BadModule {
            module: base.to_string(),
            reason: format!("weight dtype is {:?}, expected U32", w_view.dtype()),
        });
    }
    if s_view.dtype() != Dtype::BF16 {
        return Err(MlxImageImportError::BadModule {
            module: base.to_string(),
            reason: format!("scales dtype is {:?}, expected BF16", s_view.dtype()),
        });
    }
    if b_view.dtype() != Dtype::BF16 {
        return Err(MlxImageImportError::BadModule {
            module: base.to_string(),
            reason: format!("biases dtype is {:?}, expected BF16", b_view.dtype()),
        });
    }

    // Shapes: weight [out, in/16], scales/biases [out, in/128].
    let w_shape = w_view.shape();
    if w_shape.len() != 2 {
        return Err(MlxImageImportError::BadModule {
            module: base.to_string(),
            reason: format!("weight rank is {}, expected 2", w_shape.len()),
        });
    }
    let out_features = w_shape[0];
    let weight_cols = w_shape[1];
    let in_features = weight_cols * 16;

    // Decode raw LE bytes → typed buffers. The weight is read as raw u32 words;
    // routing it through the f32 path (which silently drops U32) would be wrong.
    let weight = u32_from_le_bytes(w_view.data());
    let scales = u16_from_le_bytes(s_view.data());
    let biases = u16_from_le_bytes(b_view.data());

    let blocks = pack_quantized_module(base, &weight, &scales, &biases, out_features, in_features)?;

    let data = blocks_to_bytes(&blocks);

    // GGUF shape = reversed safetensors logical shape [out, in] → [in, out].
    let gguf_shape = vec![in_features as u64, out_features as u64];

    Ok(TensorEntry {
        name: base.to_string(),
        shape: gguf_shape,
        tensor_type: TensorType::TQ2_0_g128,
        data,
    })
}

/// Build the GGUF tensor entry for one plain (passthrough) tensor.
///
/// BF16 tensors are stored verbatim as GGUF BF16 (full fidelity). F16 tensors
/// (should none exist in this checkpoint) are likewise stored verbatim. Any
/// other dtype is rejected — this converter targets the validated MLX layout
/// and refuses to silently lossy-convert an unexpected tensor.
fn build_plain_entry(
    st: &SafeTensors<'_>,
    from_path: &Path,
    name: &str,
) -> Result<TensorEntry, MlxImageImportError> {
    let view = st.tensor(name).map_err(|e| parse_err(from_path, &e))?;
    let tensor_type = match view.dtype() {
        Dtype::BF16 => TensorType::BF16,
        Dtype::F16 => TensorType::F16,
        other => {
            return Err(MlxImageImportError::BadModule {
                module: name.to_string(),
                reason: format!("unsupported plain-tensor dtype {other:?} (expected BF16 or F16)"),
            });
        }
    };

    // GGUF shape = reversed safetensors shape (outermost dimension last).
    let gguf_shape: Vec<u64> = view.shape().iter().rev().map(|&d| d as u64).collect();

    // Verbatim byte copy: BF16/F16 are 1 element per "block" of 2 bytes, so the
    // little-endian element bytes are already in GGUF layout.
    let data = view.data().to_vec();

    Ok(TensorEntry {
        name: name.to_string(),
        shape: gguf_shape,
        tensor_type,
        data,
    })
}

/// Map a safetensors error into the importer's parse error variant.
fn parse_err(from_path: &Path, e: &safetensors::SafeTensorError) -> MlxImageImportError {
    MlxImageImportError::Parse {
        path: from_path.to_path_buf(),
        msg: e.to_string(),
    }
}

/// Decode a little-endian `u32` buffer from raw bytes.
///
/// A trailing partial word (length not a multiple of 4) is ignored; callers
/// validate the resulting element count against the expected shape via
/// [`pack_quantized_module`].
fn u32_from_le_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decode a little-endian `u16` (bf16/f16 bit-pattern) buffer from raw bytes.
fn u16_from_le_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_decode_roundtrip() {
        let words = [0x0123_4567u32, 0x89ab_cdef, 0x0000_0001];
        let mut bytes = Vec::new();
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        assert_eq!(u32_from_le_bytes(&bytes), words);
    }

    #[test]
    fn u16_decode_roundtrip() {
        let vals = [0x0001u16, 0x8000, 0xffff, 0x3f80];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(u16_from_le_bytes(&bytes), vals);
    }

    // ── End-to-end: synthetic safetensors → GGUF → reader ───────────────────

    use self::pack::f32_to_bf16;
    use oxibonsai_core::gguf::reader::GgufFile;
    use oxibonsai_core::gguf::types::GgufTensorType;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    /// Build the 8 little-endian `u32` words for one 128-code group.
    fn group_words(codes: &[u8; 128]) -> [u32; 8] {
        let mut words = [0u32; 8];
        for (j, &q) in codes.iter().enumerate() {
            words[j / 16] |= (q as u32) << ((j % 16) * 2);
        }
        words
    }

    #[test]
    fn end_to_end_synthetic_dit_conversion() {
        // Synthetic DiT slice: one quantized linear [out=4, in=256] and one
        // plain BF16 tensor. Verify classification, naming, shape, packing, and
        // BF16 passthrough through the real file path.
        let out = 4usize;
        let in_features = 256usize;
        let group_cols = in_features / 128; // 2
        let weight_cols = in_features / 16; // 16

        let mut weight_words = vec![0u32; out * weight_cols];
        let mut scales_bits = vec![0u16; out * group_cols];
        let mut biases_bits = vec![0u16; out * group_cols];
        let mut expected_w = vec![0.0f32; out * in_features];

        for row in 0..out {
            for g in 0..group_cols {
                let scale = 1.0_f32 / ((1 << (row + g + 1)) as f32);
                scales_bits[row * group_cols + g] = f32_to_bf16(scale);
                biases_bits[row * group_cols + g] = f32_to_bf16(-scale);
                let mut codes = [0u8; 128];
                for (j, c) in codes.iter_mut().enumerate() {
                    let q = ((row + g + j) % 3) as u8;
                    *c = q;
                    expected_w[row * in_features + (g * 128 + j)] = scale * (q as f32 - 1.0);
                }
                let words = group_words(&codes);
                let base = row * weight_cols + g * 8;
                weight_words[base..base + 8].copy_from_slice(&words);
            }
        }

        // Raw little-endian byte buffers for safetensors.
        let weight_bytes: Vec<u8> = weight_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales_bytes: Vec<u8> = scales_bits.iter().flat_map(|s| s.to_le_bytes()).collect();
        let biases_bytes: Vec<u8> = biases_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        // Plain BF16 tensor [6, 8] = 48 elements.
        let plain_vals: Vec<f32> = (0..48).map(|i| (i as f32) * 0.03125).collect();
        let plain_bytes: Vec<u8> = plain_vals
            .iter()
            .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
            .collect();

        let module = "transformer_blocks.0.attn.to_q";
        let plain_name = "norm_out.linear.weight";

        let views = vec![
            (
                format!("{module}.weight"),
                TensorView::new(Dtype::U32, vec![out, weight_cols], &weight_bytes)
                    .expect("weight view"),
            ),
            (
                format!("{module}.scales"),
                TensorView::new(Dtype::BF16, vec![out, group_cols], &scales_bytes)
                    .expect("scales view"),
            ),
            (
                format!("{module}.biases"),
                TensorView::new(Dtype::BF16, vec![out, group_cols], &biases_bytes)
                    .expect("biases view"),
            ),
            (
                plain_name.to_string(),
                TensorView::new(Dtype::BF16, vec![6, 8], &plain_bytes).expect("plain view"),
            ),
        ];
        let st_bytes = safetensors::serialize(views, None).expect("serialize safetensors");

        // Write to a temp file and run the converter.
        let dir = std::env::temp_dir().join(format!("oxibonsai_mlx_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir temp");
        let st_path = dir.join("diffusion_pytorch_model.safetensors");
        let gguf_path = dir.join("dit.gguf");
        std::fs::write(&st_path, &st_bytes).expect("write safetensors");

        let stats = convert_mlx_image_to_gguf(&st_path, &gguf_path, "tq2_0_g128")
            .expect("conversion should succeed");
        assert_eq!(stats.n_ternary, 1, "one quantized module");
        assert_eq!(stats.n_bf16, 1, "one bf16 passthrough");
        assert_eq!(stats.n_tensors, 2, "no standalone scales/biases tensors");

        // Parse the produced GGUF and verify.
        let gguf_bytes = std::fs::read(&gguf_path).expect("read gguf");
        let parsed = GgufFile::parse(&gguf_bytes).expect("parse gguf");

        // Quantized tensor: base module name, type TQ2_0_g128, shape [in, out].
        let qinfo = parsed
            .tensors
            .require(module)
            .expect("quant tensor present");
        assert_eq!(qinfo.tensor_type, GgufTensorType::TQ2_0_g128);
        assert_eq!(qinfo.shape, vec![in_features as u64, out as u64]);

        // The suffixed sub-tensors must NOT be present.
        assert!(parsed.tensors.get(&format!("{module}.scales")).is_none());
        assert!(parsed.tensors.get(&format!("{module}.biases")).is_none());

        // Dequant the quantized tensor (out-major blocks) and compare to ref.
        let qbytes = parsed.tensor_data(module).expect("quant data");
        let blocks = oxibonsai_core::quant_ternary::BlockTQ2_0_g128::slice_from_bytes(qbytes)
            .expect("blocks");
        assert_eq!(blocks.len(), out * group_cols);
        let mut deq = vec![0.0f32; out * in_features];
        for row in 0..out {
            for g in 0..group_cols {
                let blk = &blocks[row * group_cols + g..row * group_cols + g + 1];
                let mut tmp = vec![0.0f32; 128];
                oxibonsai_core::quant_ternary::BlockTQ2_0_g128::dequant(blk, &mut tmp)
                    .expect("dequant");
                let base = row * in_features + g * 128;
                deq[base..base + 128].copy_from_slice(&tmp);
            }
        }
        assert_eq!(deq, expected_w, "dequantized weights must match reference");

        // Plain BF16 tensor: type BF16, reversed shape [8, 6], byte-identical.
        let pinfo = parsed
            .tensors
            .require(plain_name)
            .expect("plain tensor present");
        assert_eq!(pinfo.tensor_type, GgufTensorType::BF16);
        assert_eq!(pinfo.shape, vec![8, 6]);
        let pbytes = parsed.tensor_data(plain_name).expect("plain data");
        assert_eq!(
            pbytes,
            plain_bytes.as_slice(),
            "bf16 bytes preserved exactly"
        );

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }
}
