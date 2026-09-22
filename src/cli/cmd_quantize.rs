//! `oxibonsai quantize` — quantize a GGUF model to a lower-precision format.

use super::util::{check_quantize_memory_budget, dequantize_gguf_tensor, parse_quantize_format};

pub(crate) fn run(
    input: String,
    output: String,
    format: String,
    force: bool,
) -> anyhow::Result<()> {
    use std::io::Write as _;
    use std::path::Path;

    let input_path = Path::new(&input);
    if !input_path.exists() {
        anyhow::bail!("input file does not exist: {input}");
    }

    // Resolve the requested format up front — refuse unsupported
    // combinations before doing any work rather than after
    // reading and dequantizing anything.
    let target_format = parse_quantize_format(&format)?;

    // Read the original on-disk file size for the reported compression
    // ratio.
    let original_bytes = std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
    let original_mb = original_bytes as f64 / (1024.0 * 1024.0);

    println!("Quantizing {input} -> {output} (format: {format})...");

    let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(input_path)
        .map_err(|e| anyhow::anyhow!("failed to open model '{input}': {e}"))?;
    let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap)?;

    // cli-14: stream tensor-by-tensor through CONVERT-EXPORT's
    // `export_to_gguf_streaming` instead of dequantizing every source
    // tensor into one `Vec<WeightTensor>` and re-encoding the whole
    // output into one `Vec<u8>` — the old approach needed roughly the
    // model's full f32 size *twice* live in RAM, guaranteed OOM on the
    // 27B. The writer emits the header and the complete tensor-info
    // directory from a shape-only `TensorPlan` up front (no tensor data
    // touched yet), then pulls, encodes, writes and drops one tensor at a
    // time via the `load` callback below. Peak additional memory is now
    // one tensor, which is what `check_quantize_memory_budget` guards.
    check_quantize_memory_budget(&gguf, force)?;

    let mut names: Vec<&str> = gguf.tensors.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();

    let plan: Vec<oxibonsai_model::export::TensorPlan> = names
        .iter()
        .map(
            |&name| -> anyhow::Result<oxibonsai_model::export::TensorPlan> {
                let info = gguf.tensors.require(name)?;
                let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
                Ok(oxibonsai_model::export::TensorPlan::new(name, shape))
            },
        )
        .collect::<anyhow::Result<Vec<_>>>()?;

    let model_name = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    // Carry the source's architecture/tokenizer/provenance metadata into
    // the output when present, so a re-quantized real model stays
    // loadable instead of coming out with no `general.architecture` block
    // at all. A source with no architecture (e.g. a synthetic test
    // fixture, or a non-language-model file) falls back to the
    // architecture-less config exactly as before, rather than failing the
    // whole command over an optional enrichment.
    let base_config = oxibonsai_model::export::ExportConfig::new(target_format, model_name)
        .with_fp32_layers(oxibonsai_model::export::ExportConfig::default_fp32_exceptions());
    let export_config = match base_config.clone().with_source_metadata(&gguf.metadata) {
        Ok(cfg) => cfg,
        Err(oxibonsai_model::export::ExportError::MissingArchitecture) => base_config,
        Err(e) => return Err(e.into()),
    };

    let output_path = Path::new(&output);
    let file = std::fs::File::create(output_path)
        .map_err(|e| anyhow::anyhow!("failed to create output file '{output}': {e}"))?;
    let mut writer = std::io::BufWriter::new(file);

    let stats = oxibonsai_model::export::export_to_gguf_streaming(
        &plan,
        |plan_entry| {
            dequantize_gguf_tensor(&gguf, &plan_entry.name).map_err(|e| {
                oxibonsai_model::export::ExportError::QuantizeError {
                    name: plan_entry.name.clone(),
                    reason: e.to_string(),
                }
            })
        },
        &export_config,
        &[],
        &mut writer,
    )?;
    writer.flush()?;
    drop(writer);

    let quantized_bytes = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    let quantized_mb = quantized_bytes as f64 / (1024.0 * 1024.0);
    let compression_ratio = if quantized_bytes == 0 {
        1.0
    } else {
        original_bytes as f64 / quantized_bytes as f64
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
