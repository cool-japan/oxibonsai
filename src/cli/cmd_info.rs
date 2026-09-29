//! `oxibonsai info` — display model info from a GGUF file.
//!
//! cli-02: prints only values actually present in
//! the file's metadata — never `Qwen3Config::from_metadata`'s fabricated
//! defaults standing in for a key a non-language-model GGUF (e.g. a CLIP
//! vision projector) never had in the first place. Every numeric field is
//! probed independently and rendered `"-"` (or `null` in `--json`) when
//! absent.
//!
//! A Bonsai 2 `qwen35` hybrid gets the full truthful report: the 16 full /
//! 48 Gated-DeltaNet layer split,
//! the resolved weight type with its ggml id (PQ2_0 = 142, PTQ1_0 = 143),
//! the `prism.hadamard.*` contract, vocabulary and context, the KV and
//! recurrent bytes per sequence, a dry bind of the hybrid model (the
//! constructor `run` actually uses), and the kernel tier the engine seam
//! really runs it on (the CPU tier — never the GPU tier `auto_detect`
//! would report for a dense model on this machine).

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_core::gguf::tensor_info::keys;

use super::bonsai2;
use super::model_desc;

pub(crate) fn run(model: Option<String>, json: bool) -> anyhow::Result<()> {
    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;

    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;
    let gguf = match GgufFile::parse(&mmap) {
        Ok(gguf) => gguf,
        Err(parse_err) => {
            // cli-02 / cli-15: a file the strict parser rejects (e.g. an
            // unrecognised quant id on an otherwise forward-compatible
            // file, such as the Bonsai 2 27B GGUFs before their loaders
            // land) used to surface only an opaque parse error here, with
            // no way to see anything about the file at all — unlike
            // `validate`, which already falls back to the tolerant
            // `probe_compat` scan. Do the same here.
            return report_probe_compat_fallback(&mmap, &model, json, &parse_err);
        }
    };

    let arch = gguf
        .metadata
        .get_string(keys::GENERAL_ARCHITECTURE)
        .unwrap_or("-")
        .to_string();
    let known_arch = model_desc::is_known_language_model_architecture(&arch);
    // Honest, present-only display of a couple of extra identifying
    // fields the old fixed field list never showed at all.
    let general_name = model_desc::display_metadata_value(&gguf.metadata, keys::GENERAL_NAME);
    let tokenizer_model = model_desc::display_metadata_value(&gguf.metadata, keys::TOKENIZER_MODEL);

    let type_counts = gguf.tensors.count_by_type();
    let unsupported_types = model_desc::unsupported_tensor_types(&type_counts);

    // Every numeric field is probed independently against the model's
    // real `general.architecture` (falling back to the generic `llm.*`
    // key) — never through `Qwen3Config::from_metadata`, whose
    // `.unwrap_or(<default>)` chain is exactly what fabricated the 8B
    // numbers for a non-Qwen3 file.
    let layers = model_desc::display_arch_scoped_u32(
        &gguf.metadata,
        &arch,
        "block_count",
        keys::LLM_BLOCK_COUNT,
    );
    let hidden_size = model_desc::display_arch_scoped_u32(
        &gguf.metadata,
        &arch,
        "embedding_length",
        keys::LLM_EMBEDDING_LENGTH,
    );
    let q_heads = model_desc::display_arch_scoped_u32(
        &gguf.metadata,
        &arch,
        "attention.head_count",
        keys::LLM_ATTENTION_HEAD_COUNT,
    );
    let kv_heads = model_desc::display_arch_scoped_u32(
        &gguf.metadata,
        &arch,
        "attention.head_count_kv",
        keys::LLM_ATTENTION_HEAD_COUNT_KV,
    );
    let head_dim = model_desc::display_arch_scoped_u32(
        &gguf.metadata,
        &arch,
        "attention.key_length",
        keys::LLM_ATTENTION_KEY_LENGTH,
    );
    let intermediate_size = model_desc::display_arch_scoped_u32(
        &gguf.metadata,
        &arch,
        "feed_forward_length",
        keys::LLM_FEED_FORWARD_LENGTH,
    );
    let max_context_length = model_desc::display_arch_scoped_u32(
        &gguf.metadata,
        &arch,
        "context_length",
        keys::LLM_CONTEXT_LENGTH,
    );
    // Vocab: the tokenizer's own token array length is authoritative when
    // present; otherwise fall back to the
    // arch-scoped `vocab_size` metadata key, honestly "-" if neither
    // exists.
    let vocab_size = gguf
        .metadata
        .get_array(keys::TOKENIZER_TOKENS)
        .ok()
        .map(|arr| arr.len().to_string())
        .unwrap_or_else(|| {
            model_desc::display_arch_scoped_u32(
                &gguf.metadata,
                &arch,
                "vocab_size",
                keys::LLM_VOCAB_SIZE,
            )
        });

    // The variant classifier needs a full `Qwen3Config` (num_layers +
    // hidden_size at minimum); only attempt it for a known language-model
    // architecture with both values genuinely present, so a CLIP file (or
    // any file missing those keys) never gets a variant guess built from
    // silently-substituted defaults.
    // `from_config_and_resolved_sample` (not the raw-parse-time
    // `from_config_and_sample_tensor_type`) so a
    // Bonsai 2 27B file reports its real variant name (e.g.
    // "Ternary-Bonsai-2-27B-PQ2_0") instead of the generic "Custom" a
    // ternary/2-bit tensor layout the older classifier does not
    // special-case would otherwise fall through to.
    let has_hadamard = gguf.metadata.get("prism.hadamard.version").is_some();
    let hybrid = bonsai2::is_qwen35_hybrid(&arch).then(|| model_desc::hybrid_report(&gguf));
    let bound_variant = match &hybrid {
        Some(Ok(report)) => report.bind.as_ref().ok().and_then(|b| b.variant.clone()),
        _ => None,
    };
    let variant_name = if bound_variant.is_some() {
        bound_variant
    } else if known_arch && layers != "-" && hidden_size != "-" {
        oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata)
            .ok()
            .map(|config| {
                let dominant_type = type_counts
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(ty, _)| *ty)
                    .unwrap_or(oxibonsai_core::GgufTensorType::Q1_0_g128);
                oxibonsai_model::ModelVariant::from_config_and_resolved_sample(
                    &config,
                    dominant_type,
                    has_hadamard,
                )
                .name()
                .to_string()
            })
    } else {
        None
    };

    // The EFFECTIVE kernel tier `run` would dispatch to: the CPU tier for a
    // hybrid (the engine seam pins it there — no hybrid GPU encoder exists
    // yet), the auto-detected tier for a dense model.
    let (kernel_tier, kernel_tier_reason) = match &hybrid {
        Some(_) => (
            oxibonsai_kernels::cpu_kernel_tier().to_string(),
            "hybrid qwen35 model: no hybrid GPU encoder exists yet, so it runs on the best CPU \
             tier under --backend auto/cpu (--backend metal is refused)"
                .to_string(),
        ),
        None => model_desc::dense_auto_tier(),
    };
    let weight_bytes = mmap.len() as u64;

    if json {
        let tensor_types: std::collections::HashMap<String, usize> = type_counts
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        let info = serde_json::json!({
            "model": model,
            "kernel_tier": kernel_tier,
            "kernel_tier_reason": kernel_tier_reason,
            "gguf_version": gguf.header.version,
            "tensor_count": gguf.header.tensor_count,
            "metadata_entries": gguf.header.metadata_kv_count,
            "architecture": arch,
            "known_language_model_architecture": known_arch,
            "general_name": null_if_dash(&general_name),
            "tokenizer_model": null_if_dash(&tokenizer_model),
            "variant": variant_name,
            "num_layers": null_if_dash(&layers),
            "hidden_size": null_if_dash(&hidden_size),
            "num_attention_heads": null_if_dash(&q_heads),
            "num_kv_heads": null_if_dash(&kv_heads),
            "head_dim": null_if_dash(&head_dim),
            "vocab_size": null_if_dash(&vocab_size),
            "max_context_length": null_if_dash(&max_context_length),
            "intermediate_size": null_if_dash(&intermediate_size),
            "tensor_types": tensor_types,
            "unsupported_tensor_types": unsupported_types.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "hybrid": match &hybrid {
                Some(Ok(report)) => report.to_json(weight_bytes),
                Some(Err(e)) => serde_json::json!({ "error": e.to_string() }),
                None => serde_json::Value::Null,
            },
        });
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("Model: {model}");
        println!("GGUF version: {}", gguf.header.version);
        println!("Tensor count: {}", gguf.header.tensor_count);
        println!("Metadata entries: {}", gguf.header.metadata_kv_count);
        println!();

        println!(
            "Architecture: {arch}{}",
            if known_arch {
                String::new()
            } else {
                " (not a recognized language-model architecture)".to_string()
            }
        );
        if let Some(variant) = &variant_name {
            println!("  Variant:      {variant}");
        }
        // A hybrid report's own lines (below) already carry a single
        // "Kernel tier:" line with the hybrid-specific reason
        // (`HybridReport::lines`) — print this header one only when there
        // is no such report to double up with (a dense model, or a hybrid
        // whose report failed to bind, in which case the header's own
        // reason is the only "Kernel tier:" line printed at all).
        if !matches!(&hybrid, Some(Ok(_))) {
            println!("  Kernel tier:  {kernel_tier} ({kernel_tier_reason})");
        }
        println!("  Name:         {general_name}");
        println!("  Tokenizer:    {tokenizer_model}");
        println!("  Layers:       {layers}");
        println!("  Hidden size:  {hidden_size}");
        println!("  Q heads:      {q_heads}");
        println!("  KV heads:     {kv_heads}");
        println!("  Head dim:     {head_dim}");
        println!("  Vocab:        {vocab_size}");
        println!("  Max context:  {max_context_length}");
        println!("  Intermediate: {intermediate_size}");
        match &hybrid {
            Some(Ok(report)) => {
                for line in report.lines(weight_bytes) {
                    println!("  {line}");
                }
            }
            Some(Err(e)) => println!("  Hybrid report: FAILED ({e})"),
            None => {}
        }
        println!();

        println!("Tensor types:");
        for (tensor_type, count) in &type_counts {
            println!("  {tensor_type}: {count}");
        }
        if !unsupported_types.is_empty() {
            println!();
            println!(
                "NOTE: this build's model loader has no execution path for: {} \
                 (present in the file, but `run`/`chat`/`serve` cannot load it yet).",
                unsupported_types
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    Ok(())
}

/// Convert the honest `"-"` display sentinel into JSON `null` for
/// `--json` output, so a machine consumer can test for absence with
/// `!= null` instead of string-comparing against `"-"`.
fn null_if_dash(value: &str) -> Option<&str> {
    if value == "-" {
        None
    } else {
        Some(value)
    }
}

/// `GgufFile::parse` rejected this file: report `GgufFile::probe_compat`'s
/// tolerant scan instead of letting the opaque parse error stand alone
/// (mirrors `cmd_validate::report_probe_compat_fallback`). Unlike
/// `validate`, `info` is a pure describe-what-you-can command (it already
/// returns `Ok` for a well-formed-but-unsupported-architecture file, such
/// as a CLIP vision projector), so a file the strict parser rejects but
/// the tolerant probe can still describe is reported and returns `Ok` too
/// — only a file neither pass can make any sense of is a hard error.
fn report_probe_compat_fallback(
    mmap: &[u8],
    model: &str,
    json: bool,
    parse_err: &oxibonsai_core::BonsaiError,
) -> anyhow::Result<()> {
    let report = GgufFile::probe_compat(mmap).map_err(|probe_err| {
        anyhow::anyhow!(
            "failed to parse model '{model}': {parse_err}; the tolerant compatibility probe \
             also failed: {probe_err}"
        )
    })?;

    if json {
        let info = serde_json::json!({
            "model": model,
            "strict_parse_error": parse_err.to_string(),
            "gguf_version": report.version.to_string(),
            "tensor_count": report.tensor_count,
            "metadata_entries": report.metadata_count,
            "believed_loadable": report.is_loadable,
            "unknown_quant_type_ids": report.unknown_quant_types,
            "warnings": report.warnings,
        });
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("Model: {model}");
        println!("  Strict parse failed: {parse_err}");
        println!();
        println!("Compatibility probe (structural scan only):");
        println!("  GGUF version:      {}", report.version);
        println!("  Tensor count:      {}", report.tensor_count);
        println!("  Metadata entries:  {}", report.metadata_count);
        println!(
            "  Believed loadable: {} (does not confirm this build can execute every tensor \
             type)",
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
    }
    Ok(())
}
