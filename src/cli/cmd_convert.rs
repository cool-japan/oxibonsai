//! `oxibonsai convert` — convert a HuggingFace safetensors model (or an ONNX
//! MatMulNBits model) to GGUF format.
//!
//! CQ-04: `convert_hf_to_gguf_with_options` refuses a checkpoint with a
//! tensor it cannot map onto a GGUF name (it used to be dropped silently);
//! `--allow-unmapped` is the explicit escape hatch that drops such tensors
//! with a warning each.

pub(crate) fn run(
    from: String,
    to: String,
    quant: String,
    onnx: bool,
    allow_unmapped: bool,
) -> anyhow::Result<()> {
    use std::path::Path;
    let from_path = Path::new(&from);
    let to_path = Path::new(&to);
    if !from_path.exists() {
        anyhow::bail!("input path not found: {from}");
    }
    if onnx && allow_unmapped {
        anyhow::bail!(
            "--allow-unmapped applies to HuggingFace safetensors conversion only; the ONNX \
             converter maps every MatMulNBits initializer it reads (drop --allow-unmapped or \
             --onnx)"
        );
    }
    let format = if onnx { "onnx" } else { "hf" };
    println!(
        "Converting {from} -> {to} (quant: {quant}, format: {format}{})",
        if allow_unmapped {
            ", unmapped tensors dropped"
        } else {
            ""
        }
    );
    let stats = if onnx {
        oxibonsai_model::convert_onnx_to_gguf(from_path, to_path, &quant)?
    } else {
        oxibonsai_model::convert::convert_hf_to_gguf_with_options(
            from_path,
            to_path,
            &quant,
            allow_unmapped,
        )?
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
