//! Text-to-image with Bonsai-Image (FLUX.2-Klein DiT), end to end through the
//! library API: text encoder -> DiT sampler -> VAE decoder -> PNG.
//!
//! This is the library equivalent of `oxibonsai image`. The model assets are
//! not shipped with the crate; point the example at them with environment
//! variables (see `docs/IMAGEN.md` for how to obtain and convert them):
//!
//! | Variable | Asset |
//! |----------|-------|
//! | `OXI_DIT_GGUF` | the ternary DiT GGUF |
//! | `OXI_VAE_WEIGHTS` | the VAE decoder (`.safetensors` file or legacy `.npy` directory) |
//! | `OXI_TE_4BIT` | the 4-bit MLX text-encoder `model.safetensors` |
//! | `OXI_TE_TOKENIZER_DIR` | directory holding `tokenizer.json` (default: the text encoder's directory) |
//!
//! ```bash
//! OXI_DIT_GGUF=./bonsai-dit.gguf \
//! OXI_VAE_WEIGHTS=./bonsai-vae/vae/diffusion_pytorch_model.safetensors \
//! OXI_TE_4BIT=./bonsai-te/text_encoder-mlx-4bit/model.safetensors \
//!   cargo run --release -p oxibonsai-image --features metal --example text_to_image -- \
//!   "a tiny bonsai tree in a ceramic pot" bonsai.png
//! ```
//!
//! The first argument is the prompt, the second the output path (default: a
//! file in the system temporary directory). With any asset variable missing
//! the example prints what is needed and exits with status 2.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use oxibonsai_image::{text_to_image, TeSource, TextToImageCfg};

/// A non-empty environment variable as a path.
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "a tiny bonsai tree in a ceramic pot".to_string());
    let out_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("oxibonsai_text_to_image.png"));

    let (Some(dit_gguf), Some(vae_weights_dir), Some(te_weights)) = (
        env_path("OXI_DIT_GGUF"),
        env_path("OXI_VAE_WEIGHTS"),
        env_path("OXI_TE_4BIT"),
    ) else {
        eprintln!(
            "text_to_image needs the Bonsai-Image assets: set OXI_DIT_GGUF (DiT GGUF), \
             OXI_VAE_WEIGHTS (VAE decoder) and OXI_TE_4BIT (4-bit text-encoder \
             model.safetensors); optionally OXI_TE_TOKENIZER_DIR. See docs/IMAGEN.md."
        );
        return ExitCode::from(2);
    };
    let tokenizer_dir = env_path("OXI_TE_TOKENIZER_DIR")
        .or_else(|| te_weights.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));

    let cfg = TextToImageCfg {
        prompt,
        seed: 42,
        steps: 4,
        width: 512,
        height: 512,
        // Bonsai-Image is a distilled, CFG-free model: this reserved field
        // cannot change the output.
        guidance: 1.0,
        dit_gguf,
        vae_weights_dir,
        te_source: TeSource::Mlx4bit(te_weights),
        tokenizer_dir,
        golden_override: None,
    };

    match text_to_image(&cfg) {
        Ok(image) => match std::fs::write(&out_path, &image.png) {
            Ok(()) => {
                println!(
                    "wrote {}x{} PNG ({} bytes) to {}",
                    image.width,
                    image.height,
                    image.png.len(),
                    out_path.display()
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("could not write {}: {e}", out_path.display());
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("text_to_image failed: {e}");
            ExitCode::FAILURE
        }
    }
}
