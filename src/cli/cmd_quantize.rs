//! `oxibonsai quantize` — quantize a GGUF model to a lower-precision format.

use super::util::{dequantize_gguf_tensor, parse_quantize_format};

pub(crate) fn run(input: String, output: String, format: String) -> anyhow::Result<()> {
    use std::path::Path;

    let input_path = Path::new(&input);
    if !input_path.exists() {
        anyhow::bail!("input file does not exist: {input}");
    }

    // Resolve the requested format up front — refuse unsupported
    // combinations before doing any work rather than after
    // reading and dequantizing the whole model.
    let target_format = parse_quantize_format(&format)?;

    // Read the original on-disk file size for the reported
    // compression ratio.
    let original_bytes = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
    let original_mb = original_bytes as f64 / (1024.0 * 1024.0);

    println!("Quantizing {input} -> {output} (format: {format})...");

    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(input_path)
        .map_err(|e| anyhow::anyhow!("failed to open model '{input}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

    // Dequantize every source tensor to f32 in a deterministic
    // (name-sorted) order, then re-encode through the real
    // export pipeline — the same one `oxibonsai_model::export`
    // ships and is round-trip-tested against — instead of
    // fabricating a byte count.
    let mut names: Vec<&str> = gguf.tensors.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();

    let mut tensors = Vec::with_capacity(names.len());
    for name in names {
        let info = gguf.tensors.require(name)?;
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        let data = dequantize_gguf_tensor(&gguf, name)?;
        tensors.push(oxibonsai_model::export::WeightTensor::new(
            name, data, shape,
        ));
    }

    let model_name = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let export_config = oxibonsai_model::export::ExportConfig::new(target_format, model_name)
        .with_fp32_layers(oxibonsai_model::export::ExportConfig::default_fp32_exceptions());

    let stats = oxibonsai_model::export::export_stats(&tensors, &export_config);
    let bytes = oxibonsai_model::export::export_to_gguf(&tensors, &export_config, &[])?;

    std::fs::write(&output, &bytes)?;

    let quantized_mb = bytes.len() as f64 / (1024.0 * 1024.0);
    let compression_ratio = if bytes.is_empty() {
        1.0
    } else {
        original_bytes as f64 / bytes.len() as f64
    };

    println!(
        "Quantizing... Original: {original_mb:.1} MB \
         → Quantized: {quantized_mb:.1} MB \
         ({compression_ratio:.1}:1 compression)"
    );
    println!(
        "Output: {output} ({} tensors: {} quantized, {} kept fp32)",
        stats.num_tensors, stats.quantized_tensors, stats.fp32_tensors
    );

    Ok(())
}
