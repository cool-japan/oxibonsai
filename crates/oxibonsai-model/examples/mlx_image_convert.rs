//! Convert an MLX-packed FLUX.2 DiT transformer (safetensors) to OxiBonsai GGUF.
//!
//! Reads the `diffusion_pytorch_model.safetensors` of a PrismML
//! `bonsai-image-ternary-*-mlx-2bit` checkpoint and emits a single OxiBonsai
//! GGUF file: quantized linears as TQ2_0_g128 and the skip-pattern tensors as
//! BF16 (full fidelity).
//!
//! Usage:
//!
//! ```text
//! cargo run -p oxibonsai-model --example mlx_image_convert --release -- \
//!     <path/to/diffusion_pytorch_model.safetensors> <output.gguf> [quant]
//! ```
//!
//! The optional third argument is the target quantisation format; it defaults
//! to `tq2_0_g128` (the only one supported today).

use std::path::Path;
use std::time::Instant;

use oxibonsai_model::convert_mlx_image_to_gguf;

fn main() {
    if let Err(e) = run() {
        eprintln!("mlx_image_convert error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let from_path = args.get(1).ok_or_else(|| {
        "usage: mlx_image_convert <model.safetensors> <output.gguf> [quant]".to_string()
    })?;
    let out_path = args.get(2).ok_or_else(|| {
        "usage: mlx_image_convert <model.safetensors> <output.gguf> [quant]".to_string()
    })?;
    let quant = args.get(3).map(String::as_str).unwrap_or("tq2_0_g128");

    let from_path = Path::new(from_path);
    let out_path = Path::new(out_path);

    println!("── OxiBonsai MLX (FLUX.2 DiT) → GGUF converter ──");
    println!("  source : {}", from_path.display());
    println!("  target : {}", out_path.display());
    println!("  quant  : {quant}");

    let t0 = Instant::now();
    let stats =
        convert_mlx_image_to_gguf(from_path, out_path, quant).map_err(|e| format!("{e}"))?;
    let elapsed = t0.elapsed();

    println!();
    println!("── Conversion complete ──");
    println!("  total tensors       : {}", stats.n_tensors);
    println!("  TQ2_0_g128 weights  : {}", stats.n_ternary);
    println!("  BF16 passthrough    : {}", stats.n_bf16);
    println!("  F16 passthrough     : {}", stats.n_f16);
    println!(
        "  output file size    : {} bytes ({:.2} MiB)",
        stats.output_bytes,
        (stats.output_bytes as f64) / (1024.0 * 1024.0)
    );
    println!("  elapsed             : {elapsed:.2?}");

    Ok(())
}
