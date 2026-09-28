//! `oxibonsai validate` — validate that a GGUF file is well-formed, is a
//! recognized language-model architecture, and is actually runnable by
//! this build.
//!
//! cli-02 / gatekeeper REQUIRED #1: `Validation: OK` used to mean only
//! "`Qwen3Config::from_metadata` did not return `Err`" — which it never
//! does (every field defaults), so a CLIP vision-projector GGUF reported
//! `OK` with the 8B config's fabricated numbers. `OK` now requires:
//! `general.architecture` to be a recognized language-model architecture,
//! the required config keys to be genuinely present, minimal tensor
//! coverage (`token_embd.weight` + an output norm/head), and every tensor
//! type in the file to have a real load path in this build (not merely a
//! known id — see `model_desc`'s doc). A file that parses and is
//! architecturally sound but carries a tensor type this build cannot yet
//! execute (PQ2_0/PTQ1_0 ahead of B2-09) reports the honest interim
//! status "parses; not runnable by this build" instead of a blanket `OK`
//! or `FAILED`.
//!
//! REQUIRED #2 (wave-4b): loadability is a function of the constructor that
//! will actually run. A `qwen35` hybrid is `OK` only when a header-only dry
//! bind of `HybridModel::from_gguf` — exactly what `run` builds — succeeds
//! (the report then shows the layer split, ggml ids 142/143, the Hadamard
//! contract and the per-sequence state bytes); a bind failure is `PARSES
//! (not runnable by this build: <why>)` and a non-zero exit.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::keys;

use super::bonsai2;
use super::model_desc;

pub(crate) fn run(model: String) -> anyhow::Result<()> {
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;

    println!("Validating: {model}");

    let gguf = match GgufFile::parse(&mmap) {
        Ok(gguf) => gguf,
        Err(parse_err) => {
            // cli-02 / cli-15: fall back to the tolerant compatibility
            // scan so a file `parse` hard-rejects (e.g. an unrecognised
            // quant type on a forward-compatible file) prints a real
            // diagnostic report instead of an opaque parse error.
            return report_probe_compat_fallback(&mmap, &parse_err);
        }
    };

    println!("  GGUF version:     {}", gguf.header.version);
    println!("  Tensor count:     {}", gguf.header.tensor_count);
    println!("  Metadata entries: {}", gguf.header.metadata_kv_count);

    let arch = gguf
        .metadata
        .get_string(keys::GENERAL_ARCHITECTURE)
        .unwrap_or("-")
        .to_string();
    println!("  Architecture:     {arch}");

    let mut failures: Vec<String> = Vec::new();

    let known_arch = model_desc::is_known_language_model_architecture(&arch);
    if !known_arch {
        failures.push(format!(
            "general.architecture = '{arch}' is not a recognized language-model architecture \
             (known: {})",
            model_desc::KNOWN_LANGUAGE_MODEL_ARCHITECTURES.join(", ")
        ));
    }

    // Required keys (core-gguf-03-adjacent): the minimum a decoder-only
    // transformer config needs. Each is probed honestly (arch-scoped, then
    // the generic `llm.*` fallback) — a "-" here means genuinely absent,
    // never a substituted default.
    let required_numeric = [
        ("block_count", keys::LLM_BLOCK_COUNT, "num_layers"),
        (
            "embedding_length",
            keys::LLM_EMBEDDING_LENGTH,
            "hidden_size",
        ),
        (
            "attention.head_count",
            keys::LLM_ATTENTION_HEAD_COUNT,
            "num_attention_heads",
        ),
    ];
    for (arch_suffix, generic_key, label) in required_numeric {
        let value =
            model_desc::display_arch_scoped_u32(&gguf.metadata, &arch, arch_suffix, generic_key);
        if value == "-" {
            failures.push(format!(
                "required key missing: {arch}.{arch_suffix} (or {generic_key}) — needed for {label}"
            ));
        }
    }

    // Tensor coverage (core-gguf-03): the two tensors present in every
    // decoder-only transformer GGUF this project supports.
    if gguf.tensors.require("token_embd.weight").is_err() {
        failures.push("required tensor missing: token_embd.weight".to_string());
    }
    let has_output = gguf.tensors.require("output_norm.weight").is_ok()
        || gguf.tensors.require("output.weight").is_ok();
    if !has_output {
        failures.push("required tensor missing: output_norm.weight or output.weight".to_string());
    }

    // Quant executability: every tensor type in the file must have an
    // actual load path in this build's model loader (not merely a known
    // id — `GgufTensorType::is_executable()` alone is not enough, see
    // `model_desc`'s doc).
    let type_counts = gguf.tensors.count_by_type();
    let unsupported_types = model_desc::unsupported_tensor_types(&type_counts);

    // REQUIRED #2: a hybrid is judged by the dry bind of the constructor
    // `run` uses, not by the tensor-type allowlist alone.
    let hybrid = if known_arch && failures.is_empty() && bonsai2::is_qwen35_hybrid(&arch) {
        match model_desc::hybrid_report(&gguf) {
            Ok(report) => Some(report),
            Err(e) => {
                failures.push(format!("qwen35 hybrid metadata: {e}"));
                None
            }
        }
    } else {
        None
    };
    if let Some(report) = &hybrid {
        for line in report.lines(mmap.len() as u64) {
            println!("  {line}");
        }
    }

    println!();
    if !failures.is_empty() {
        println!("Validation: FAILED");
        for f in &failures {
            println!("  - {f}");
        }
        anyhow::bail!("GGUF validation failed: {}", failures.join("; "));
    }

    if !unsupported_types.is_empty() {
        // Interim wording until B2-09 lands the missing loaders (wave 3).
        println!("Validation: PARSES (not runnable by this build)");
        println!(
            "  - tensor type(s) with no load path in this build: {} (loader lands in a future \
             wave)",
            unsupported_types
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        anyhow::bail!(
            "GGUF parses and is architecturally sound, but is not runnable by this build: \
             unsupported tensor type(s) {}",
            unsupported_types
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    if let Some(report) = &hybrid {
        if let Err(bind_err) = &report.bind {
            println!("Validation: PARSES (not runnable by this build: {bind_err})");
            anyhow::bail!(
                "GGUF parses and is architecturally sound, but the qwen35 hybrid model cannot be \
                 bound by this build: {bind_err}"
            );
        }
    }

    println!("Validation: OK");
    Ok(())
}

/// `GgufFile::parse` failed; report `GgufFile::probe_compat`'s tolerant
/// scan instead of letting the opaque parse error stand alone (cli-02 /
/// cli-15). This is also what makes the previously-false
/// `loadable=false / unknown_quants=198` WARN disappear for a file whose
/// only issue is a quant type this exact build doesn't parse: the report
/// is now printed with full context (version, real unknown-id list,
/// warnings) rather than inferred from a bare parse failure.
fn report_probe_compat_fallback(
    mmap: &[u8],
    parse_err: &oxibonsai_core::BonsaiError,
) -> anyhow::Result<()> {
    println!();
    match GgufFile::probe_compat(mmap) {
        Ok(report) => {
            println!("  GGUF version:     {}", report.version);
            println!("  Tensor count:     {}", report.tensor_count);
            println!("  Metadata entries: {}", report.metadata_count);
            println!(
                "  Believed loadable: {} (structural probe only — this does not confirm this \
                 build can execute every tensor type)",
                report.is_loadable
            );
            if !report.unknown_quant_types.is_empty() {
                println!(
                    "  Unrecognized quantization type id(s): {:?}",
                    report.unknown_quant_types
                );
            }
            for warning in &report.warnings {
                println!("  - {warning}");
            }
            println!();
            println!("Validation: FAILED");
            println!("  - strict parse failed: {parse_err}");
            anyhow::bail!(
                "GGUF validation failed: strict parse rejected this file ({parse_err}); the \
                 tolerant compatibility probe above describes what OxiBonsai could still \
                 determine about it"
            );
        }
        Err(probe_err) => {
            println!("Validation: FAILED");
            println!("  - strict parse failed: {parse_err}");
            println!("  - compatibility probe also failed: {probe_err}");
            anyhow::bail!(
                "GGUF validation failed: this file is not a well-formed GGUF (parse error: \
                 {parse_err}; probe error: {probe_err})"
            );
        }
    }
}

#[cfg(test)]
#[path = "cmd_validate_tests.rs"]
mod tests;
