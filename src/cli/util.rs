//! Shared helpers used by several subcommands: stdin prompt reading,
//! tokenizer auto-detection, and GGUF tensor dequantization for `quantize`.

use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

pub(crate) fn read_prompt_stdin() -> String {
    let stdin = io::stdin();
    let mut lines = Vec::new();
    for line in stdin.lock().lines() {
        match line {
            Ok(l) => lines.push(l),
            Err(_) => break,
        }
    }
    lines.join("\n")
}

/// Result of attempting to locate a `tokenizer.json` for a given model.
///
/// `found` is `Some(path)` when either an explicit override was supplied
/// or auto-detection succeeded.  `searched` lists every candidate path
/// inspected during auto-detection so the user can see exactly where we
/// looked when nothing turned up.
pub(crate) struct TokenizerLookup {
    pub(crate) found: Option<String>,
    pub(crate) searched: Vec<PathBuf>,
}

/// Strip a trailing GGUF quantization suffix (e.g. `-Q2_0`, `-Q4_K_M`,
/// `-F16`, `-BF16`, `-F32`) from a model basename without pulling in the
/// `regex` crate.  Returns the basename unchanged when no recognized
/// suffix is present.
pub(crate) fn strip_quant_suffix(basename: &str) -> &str {
    // Locate the last '-' segment; only that segment is a quant suffix
    // candidate.
    let Some(dash_pos) = basename.rfind('-') else {
        return basename;
    };
    let suffix = &basename[dash_pos + 1..];
    if suffix.is_empty() {
        return basename;
    }

    let is_float = matches!(suffix, "F16" | "BF16" | "F32");
    let is_quant = {
        let mut chars = suffix.chars();
        match chars.next() {
            Some('Q') => {
                // Accept Q<digits>(_<alnum>+)*  e.g. Q1_0, Q2_K, Q4_K_M, Q8_0.
                let rest: String = chars.collect();
                if rest.is_empty() {
                    false
                } else {
                    let mut parts = rest.split('_');
                    let first = parts.next().unwrap_or("");
                    if first.is_empty() || !first.chars().all(|c| c.is_ascii_digit()) {
                        false
                    } else {
                        parts.all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric()))
                    }
                }
            }
            _ => false,
        }
    };

    if is_float || is_quant {
        &basename[..dash_pos]
    } else {
        basename
    }
}

/// Build the ordered list of candidate `tokenizer.json` paths to probe
/// for a given model file.  Duplicates (after canonical lexical form)
/// are removed so the warning message stays compact.
pub(crate) fn tokenizer_candidates(model_path: &Path) -> Vec<PathBuf> {
    let parent = model_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut out: Vec<PathBuf> = Vec::new();
    let push_unique = |p: PathBuf, out: &mut Vec<PathBuf>| {
        if !out.iter().any(|existing| existing == &p) {
            out.push(p);
        }
    };

    // 1. Same directory as the model.
    push_unique(parent.join("tokenizer.json"), &mut out);

    // 2. Parent of the model directory.
    push_unique(parent.join("..").join("tokenizer.json"), &mut out);

    // 3. Sibling directories derived from the model basename.
    if let Some(stem) = model_path.file_stem().and_then(|s| s.to_str()) {
        let base = strip_quant_suffix(stem);
        for variant in [
            base.to_string(),
            format!("{base}-unpacked"),
            format!("{base}-ONNX"),
        ] {
            push_unique(parent.join(&variant).join("tokenizer.json"), &mut out);
        }
    }

    // 4. Top-level `models/` directory if the model lives anywhere under it.
    for ancestor in model_path.ancestors().skip(1) {
        if ancestor.file_name().and_then(|n| n.to_str()) == Some("models") {
            push_unique(ancestor.join("tokenizer.json"), &mut out);
            break;
        }
    }

    out
}

/// Resolve the tokenizer path: use the explicit path if given, otherwise
/// auto-detect `tokenizer.json` in a small set of conventional locations
/// derived from the model path.
pub(crate) fn resolve_tokenizer(tokenizer: Option<&str>, model_path: &str) -> TokenizerLookup {
    if let Some(p) = tokenizer {
        return TokenizerLookup {
            found: Some(p.to_string()),
            searched: Vec::new(),
        };
    }

    let model = Path::new(model_path);
    let candidates = tokenizer_candidates(model);
    for candidate in &candidates {
        if candidate.exists() {
            tracing::info!(
                path = %candidate.display(),
                "auto-detected tokenizer alongside model"
            );
            return TokenizerLookup {
                found: Some(candidate.to_string_lossy().into_owned()),
                searched: candidates,
            };
        }
    }

    TokenizerLookup {
        found: None,
        searched: candidates,
    }
}

/// Build the multi-line "no tokenizer found" warning shown by the run /
/// chat / serve paths.  Centralized so all three surfaces stay in sync.
pub(crate) fn missing_tokenizer_warning(searched: &[PathBuf]) -> String {
    let mut msg = String::from("no tokenizer found. Searched:\n");
    if searched.is_empty() {
        msg.push_str("  (no candidate paths — model path was not provided)\n");
    } else {
        for path in searched {
            msg.push_str(&format!("  - {}\n", path.display()));
        }
    }
    msg.push_str("To fix:\n");
    msg.push_str("  - Pass --tokenizer <path/to/tokenizer.json>, OR\n");
    msg.push_str(
        "  - Run ./scripts/download_tokenizer.sh to fetch the Qwen3 tokenizer to models/tokenizer.json\n",
    );
    msg.push_str("Continuing with raw token IDs in output.");
    msg
}

/// Decode a raw IEEE 754 half-precision (binary16) bit pattern to `f32`.
///
/// A small local implementation rather than pulling in the `half` crate
/// as a production dependency of this binary (it is already used
/// pervasively elsewhere in the workspace, just not linked into the CLI
/// binary itself); handles zero, subnormals, normals, infinities and NaN.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as f32;

    let magnitude = if exponent == 0 {
        if mantissa == 0.0 {
            0.0f32
        } else {
            // Subnormal: value = mantissa / 1024 * 2^-14.
            mantissa * 2f32.powi(-24)
        }
    } else if exponent == 0x1F {
        if mantissa == 0.0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let exp = exponent as i32 - 15;
        (1.0 + mantissa / 1024.0) * 2f32.powi(exp)
    };

    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// Dequantize a single named tensor from a parsed GGUF file to `f32`.
///
/// Mirrors the internal tensor loader the inference engine uses
/// (`oxibonsai_model`'s private `load_f32_tensor`), re-implemented here
/// against the same public `oxibonsai_core` block types so the
/// `quantize` subcommand can re-encode a real model's weights through
/// [`oxibonsai_model::export::export_to_gguf`] instead of fabricating a
/// result. Returns an honest error (rather than silently misreading
/// bytes) for any tensor type this CLI does not yet know how to
/// dequantize.
pub(crate) fn dequantize_gguf_tensor(
    gguf: &oxibonsai_core::gguf::reader::GgufFile<'_>,
    name: &str,
) -> anyhow::Result<Vec<f32>> {
    use oxibonsai_core::GgufTensorType;

    let info = gguf.tensors.require(name)?;
    let data = gguf.tensor_data(name)?;

    let out = match info.tensor_type {
        GgufTensorType::F32 => {
            let count = data.len() / 4;
            let mut out = vec![0.0f32; count];
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        }
        GgufTensorType::F16 => {
            let count = data.len() / 2;
            let mut out = vec![0.0f32; count];
            for (i, chunk) in data.chunks_exact(2).enumerate() {
                out[i] = f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgufTensorType::Q1_0_g128 => {
            let blocks = oxibonsai_core::tensor::BlockQ1_0G128::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::tensor::QK1_0_G128;
            let mut out = vec![0.0f32; n];
            for (i, block) in blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let base = i * oxibonsai_core::tensor::QK1_0_G128;
                for j in 0..oxibonsai_core::tensor::QK1_0_G128 {
                    let byte_index = j / 8;
                    let bit_offset = j % 8;
                    let bit = (block.qs[byte_index] >> bit_offset) & 1;
                    out[base + j] = if bit != 0 { d } else { -d };
                }
            }
            out
        }
        GgufTensorType::TQ2_0_g128 => {
            let blocks = oxibonsai_core::BlockTQ2_0_g128::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_TQ2_0_G128;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockTQ2_0_g128::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q4_0 => {
            let blocks = oxibonsai_core::BlockQ4_0::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_Q4_0;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ4_0::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q8_0 => {
            let blocks = oxibonsai_core::BlockQ8_0::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_Q8_0;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ8_0::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q4_K => {
            let blocks = oxibonsai_core::BlockQ4K::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::quant_k::QK_K;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ4K::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q5_K => {
            let blocks = oxibonsai_core::BlockQ5K::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::quant_k::QK_K;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ5K::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::Q6_K => {
            let blocks = oxibonsai_core::BlockQ6K::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::quant_k::QK_K;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockQ6K::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::F8_E4M3 => {
            let blocks = oxibonsai_core::BlockFP8E4M3::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_FP8;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockFP8E4M3::dequant(blocks, &mut out)?;
            out
        }
        GgufTensorType::F8_E5M2 => {
            let blocks = oxibonsai_core::BlockFP8E5M2::slice_from_bytes(data)?;
            let n = blocks.len() * oxibonsai_core::QK_FP8;
            let mut out = vec![0.0f32; n];
            oxibonsai_core::BlockFP8E5M2::dequant(blocks, &mut out)?;
            out
        }
        other => anyhow::bail!(
            "tensor '{name}': cannot dequantize source type {other} — `quantize` only reads \
             F32, F16, Q1_0_g128, TQ2_0_g128, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, F8_E4M3, or \
             F8_E5M2 tensors"
        ),
    };
    Ok(out)
}

/// Map a CLI `--format` string to the [`oxibonsai_model::export::ExportFormat`]
/// it names, refusing (with an honest error) any format the export
/// pipeline cannot actually produce a loadable GGUF file for.
pub(crate) fn parse_quantize_format(
    format: &str,
) -> anyhow::Result<oxibonsai_model::export::ExportFormat> {
    use oxibonsai_model::export::ExportFormat;
    match format {
        "f32" => Ok(ExportFormat::Float32),
        "q1_0" | "q1_0_g128" => Ok(ExportFormat::Q1_0G128),
        "tq2_0_g128" | "ternary" => Ok(ExportFormat::TernaryG128),
        "fp8_e4m3" => Ok(ExportFormat::FP8E4M3),
        "fp8_e5m2" => Ok(ExportFormat::FP8E5M2),
        "q4_0" => Ok(ExportFormat::Q4_0),
        "q8_0" => Ok(ExportFormat::Q8_0),
        "q4_k" => Ok(ExportFormat::Q4K),
        "q5_k" => Ok(ExportFormat::Q5K),
        "q6_k" => Ok(ExportFormat::Q6K),
        other => anyhow::bail!(
            "unsupported quantization format '{other}' — the export pipeline can only \
             produce a loadable GGUF file for: f32, q1_0, tq2_0_g128, fp8_e4m3, fp8_e5m2, \
             q4_0, q8_0, q4_k, q5_k, q6_k (q2_k, q4_1, f16 output are not supported by the \
             export pipeline; q2_k/q4_1 have no writer, and f16 has no `ExportFormat` \
             arm — use f32 or a supported quantized format instead)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        missing_tokenizer_warning, resolve_tokenizer, strip_quant_suffix, tokenizer_candidates,
    };
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Helper: write an empty `tokenizer.json` at the given path, creating
    /// any missing parent directories.
    fn touch_tokenizer(path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create_dir_all");
        }
        fs::write(path, b"{}").expect("write tokenizer.json");
    }

    #[test]
    fn resolve_tokenizer_finds_in_same_dir() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Foo-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");
        touch_tokenizer(&model_dir.join("tokenizer.json"));

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        let found = lookup.found.as_deref().expect("expected to find tokenizer");
        assert_eq!(
            PathBuf::from(found),
            model_dir.join("tokenizer.json"),
            "should locate tokenizer in the same dir as the model"
        );
    }

    #[test]
    fn resolve_tokenizer_finds_in_parent_dir() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models").join("variant");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Foo-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");
        // Place tokenizer in the parent directory only.
        let parent_tokenizer = tmp.path().join("models").join("tokenizer.json");
        touch_tokenizer(&parent_tokenizer);

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        let found = lookup.found.as_deref().expect("expected to find tokenizer");
        // Either the literal `..` candidate or the canonicalized `models/tokenizer.json`
        // candidate is acceptable; both refer to the same file.
        let found_path = PathBuf::from(found);
        let canon_found = fs::canonicalize(&found_path).expect("canonicalize found");
        let canon_target = fs::canonicalize(&parent_tokenizer).expect("canonicalize target");
        assert_eq!(
            canon_found, canon_target,
            "should locate tokenizer in the model's parent directory"
        );
    }

    #[test]
    fn resolve_tokenizer_finds_via_unpacked_sibling() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");
        // Tokenizer only lives in the sibling unpacked directory.
        let unpacked = model_dir.join("Ternary-Bonsai-8B-unpacked");
        touch_tokenizer(&unpacked.join("tokenizer.json"));

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        let found = lookup.found.as_deref().expect("expected to find tokenizer");
        assert_eq!(
            PathBuf::from(found),
            unpacked.join("tokenizer.json"),
            "should locate tokenizer via <base>-unpacked sibling directory"
        );
    }

    #[test]
    fn resolve_tokenizer_strips_quant_suffix_for_sibling_lookup() {
        // Verifies the candidate list (without filesystem) for the
        // `Foo-Q2_0.gguf` case includes Foo/, Foo-unpacked/, Foo-ONNX/.
        let model_path = PathBuf::from("models/Foo-Q2_0.gguf");
        let candidates = tokenizer_candidates(&model_path);
        let candidate_strs: Vec<String> = candidates
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        let expected = [
            "models/Foo/tokenizer.json",
            "models/Foo-unpacked/tokenizer.json",
            "models/Foo-ONNX/tokenizer.json",
        ];
        for needle in expected {
            assert!(
                candidate_strs.iter().any(|c| c == needle),
                "missing expected candidate {needle}; got {candidate_strs:?}"
            );
        }
    }

    #[test]
    fn resolve_tokenizer_records_searched_paths_when_missing() {
        let tmp = TempDir::new().expect("tempdir");
        let model_dir = tmp.path().join("models");
        fs::create_dir_all(&model_dir).expect("create model_dir");
        let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
        fs::write(&model_path, b"").expect("touch model");

        let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
        assert!(
            lookup.found.is_none(),
            "should not find tokenizer in empty tree"
        );
        assert!(
            !lookup.searched.is_empty(),
            "searched list must be populated when nothing is found"
        );
        // Confirm at least the "same dir" candidate is recorded.
        assert!(
            lookup
                .searched
                .iter()
                .any(|p| p == &model_dir.join("tokenizer.json")),
            "searched list should include the same-directory candidate"
        );
        // Warning text must mention every searched path and both remedies.
        let warning = missing_tokenizer_warning(&lookup.searched);
        for path in &lookup.searched {
            assert!(
                warning.contains(&path.display().to_string()),
                "warning should list {}, got: {warning}",
                path.display()
            );
        }
        assert!(
            warning.contains("--tokenizer"),
            "warning must mention --tokenizer remedy"
        );
        assert!(
            warning.contains("download_tokenizer.sh"),
            "warning must mention download_tokenizer.sh remedy"
        );
    }

    #[test]
    fn resolve_tokenizer_explicit_override_skips_search() {
        let lookup = resolve_tokenizer(Some("/custom/path/tokenizer.json"), "models/foo.gguf");
        assert_eq!(
            lookup.found.as_deref(),
            Some("/custom/path/tokenizer.json"),
            "explicit override must be returned verbatim"
        );
        assert!(
            lookup.searched.is_empty(),
            "explicit override must not trigger a filesystem search"
        );
    }

    #[test]
    fn strip_quant_suffix_handles_known_formats() {
        assert_eq!(
            strip_quant_suffix("Ternary-Bonsai-8B-Q2_0"),
            "Ternary-Bonsai-8B"
        );
        assert_eq!(strip_quant_suffix("Foo-Q1_0"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-Q4_K_M"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-Q8_0"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-F16"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-BF16"), "Foo");
        assert_eq!(strip_quant_suffix("Foo-F32"), "Foo");
        // Non-quant suffix should be left alone.
        assert_eq!(strip_quant_suffix("Foo-bar"), "Foo-bar");
        assert_eq!(strip_quant_suffix("Foo"), "Foo");
    }

    #[test]
    fn tokenizer_candidates_includes_top_level_models_dir() {
        let model_path = PathBuf::from("models/sub/dir/Foo-Q2_0.gguf");
        let candidates = tokenizer_candidates(&model_path);
        let candidate_strs: Vec<String> = candidates
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert!(
            candidate_strs.iter().any(|c| c == "models/tokenizer.json"),
            "expected top-level models/tokenizer.json candidate, got {candidate_strs:?}"
        );
    }
}
