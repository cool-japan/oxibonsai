//! `oxibonsai convert` — convert a HuggingFace safetensors model to GGUF format.

pub(crate) fn run(from: String, to: String, quant: String, onnx: bool) -> anyhow::Result<()> {
    use std::path::Path;
    let from_path = Path::new(&from);
    let to_path = Path::new(&to);
    if !from_path.exists() {
        anyhow::bail!("input path not found: {from}");
    }
    let format = if onnx { "onnx" } else { "hf" };
    println!("Converting {from} -> {to} (quant: {quant}, format: {format})");
    let stats = if onnx {
        oxibonsai_model::convert_onnx_to_gguf(from_path, to_path, &quant)?
    } else {
        oxibonsai_model::convert::convert_hf_to_gguf(from_path, to_path, &quant)?
    };
    println!(
        "Done: {} tensors ({} ternary + {} fp32), output: {:.1} MB",
        stats.n_tensors,
        stats.n_ternary,
        stats.n_fp32,
        stats.output_bytes as f64 / (1024.0 * 1024.0),
    );

    Ok(())
}
