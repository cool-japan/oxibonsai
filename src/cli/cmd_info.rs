//! `oxibonsai info` — display model info from a GGUF file.

pub(crate) fn run(model: Option<String>, json: bool) -> anyhow::Result<()> {
    let model = model
        .or_else(|| std::env::var("OXI_MODEL").ok().filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!("no model: pass --model <gguf> or set OXI_MODEL (e.g. in .env)")
        })?;

    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;
    let config = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata)?;
    let type_counts = gguf.tensors.count_by_type();

    // Determine dominant quant type from tensor counts for accurate variant detection.
    let dominant_type = type_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(ty, _)| *ty)
        .unwrap_or(oxibonsai_core::GgufTensorType::Q1_0_g128);

    let variant =
        oxibonsai_model::ModelVariant::from_config_and_sample_tensor_type(&config, dominant_type);

    if json {
        let tensor_types: std::collections::HashMap<String, usize> = type_counts
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        let info = serde_json::json!({
            "model": model,
            "gguf_version": gguf.header.version,
            "tensor_count": gguf.header.tensor_count,
            "metadata_entries": gguf.header.metadata_kv_count,
            "architecture": format!("Qwen3 ({})", variant.name()),
            "variant": variant.name(),
            "num_layers": config.num_layers,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
            "max_context_length": config.max_context_length,
            "intermediate_size": config.intermediate_size,
            "tensor_types": tensor_types,
        });
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("Model: {model}");
        println!("GGUF version: {}", gguf.header.version);
        println!("Tensor count: {}", gguf.header.tensor_count);
        println!("Metadata entries: {}", gguf.header.metadata_kv_count);
        println!();

        println!("Architecture: Qwen3 ({})", variant.name());
        println!("  Layers:       {}", config.num_layers);
        println!("  Hidden size:  {}", config.hidden_size);
        println!("  Q heads:      {}", config.num_attention_heads);
        println!("  KV heads:     {}", config.num_kv_heads);
        println!("  Head dim:     {}", config.head_dim);
        println!("  Vocab:        {}", config.vocab_size);
        println!("  Max context:  {}", config.max_context_length);
        println!("  Intermediate: {}", config.intermediate_size);
        println!();

        println!("Tensor types:");
        for (tensor_type, count) in &type_counts {
            println!("  {tensor_type}: {count}");
        }
    }

    Ok(())
}
