//! `oxibonsai validate` — validate that a GGUF file is well-formed and
//! display a metadata summary.

pub(crate) fn run(model: String) -> anyhow::Result<()> {
    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(std::path::Path::new(&model))
        .map_err(|e| anyhow::anyhow!("failed to open model '{model}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

    // Attempt to parse model config — validates metadata consistency.
    let config_result = oxibonsai_core::config::Qwen3Config::from_metadata(&gguf.metadata);

    println!("Validating: {model}");
    println!("  GGUF version:     {}", gguf.header.version);
    println!("  Tensor count:     {}", gguf.header.tensor_count);
    println!("  Metadata entries: {}", gguf.header.metadata_kv_count);

    match config_result {
        Ok(config) => {
            println!("  Architecture:     Qwen3");
            println!("  Layers:           {}", config.num_layers);
            println!("  Hidden size:      {}", config.hidden_size);
            println!("  Vocab size:       {}", config.vocab_size);
            println!();
            println!("Validation: OK");
        }
        Err(e) => {
            println!();
            println!("Validation: FAILED");
            println!("  Error: {e}");
            return Err(anyhow::anyhow!("GGUF validation failed: {e}"));
        }
    }

    Ok(())
}
