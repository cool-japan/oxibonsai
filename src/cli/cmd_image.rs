//! `oxibonsai image` and `oxibonsai repl` — text-to-image generation
//! (single-shot and interactive REPL variants).

use std::path::{Path, PathBuf};

use super::repl;
use super::util::read_prompt_stdin;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_image(
    prompt: String,
    out: String,
    seed: u64,
    steps: usize,
    width: usize,
    height: usize,
    dit: Option<String>,
    vae: Option<String>,
    te: Option<String>,
    tokenizer: Option<String>,
) -> anyhow::Result<()> {
    use oxibonsai_image::pipeline::{text_to_image, TeSource, TextToImageCfg};

    let prompt_text = if prompt == "-" {
        read_prompt_stdin()
    } else {
        prompt
    };

    // Resolve paths: explicit arg → env → default.
    let resolve = |arg: Option<String>, env: &str, default: &str| -> String {
        arg.or_else(|| std::env::var(env).ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| default.to_string())
    };

    let dit_path = resolve(dit, "OXI_DIT_GGUF", "/tmp/parity.gguf");
    let vae_path = resolve(vae, "OXI_VAE_WEIGHTS", "/tmp/bonsai_golden/vae/weights");

    // TE source: a `.safetensors` path → 4-bit MLX loader; otherwise a
    // directory of f32 `.npy` dumps. Resolution order: --te → OXI_TE_4BIT
    // → OXI_TE_WEIGHTS → default npy dir.
    let te_path = te
        .or_else(|| std::env::var("OXI_TE_4BIT").ok().filter(|s| !s.is_empty()))
        .or_else(|| {
            std::env::var("OXI_TE_WEIGHTS")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "/tmp/bonsai_golden/te/weights".to_string());
    let te_source = if te_path.ends_with(".safetensors") {
        TeSource::Mlx4bit(PathBuf::from(&te_path))
    } else {
        TeSource::NpyDir(PathBuf::from(&te_path))
    };

    // Tokenizer dir: --tokenizer → OXI_TE_TOKENIZER_DIR → the TE dir
    // (its parent if the TE is a safetensors file).
    let tokenizer_dir = tokenizer
        .or_else(|| {
            std::env::var("OXI_TE_TOKENIZER_DIR")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let p = PathBuf::from(&te_path);
            if te_path.ends_with(".safetensors") {
                p.parent().map(Path::to_path_buf).unwrap_or(p)
            } else {
                p
            }
        });

    let cfg = TextToImageCfg {
        prompt: prompt_text,
        seed,
        steps,
        width,
        height,
        // Not a CLI-exposed knob: Bonsai-Image is a distilled,
        // CFG-free model (`guidance_embeds: false`), so this
        // reserved field can never change the output — see
        // `TextToImageCfg::guidance`'s doc comment.
        guidance: 1.0,
        dit_gguf: PathBuf::from(&dit_path),
        vae_weights_dir: PathBuf::from(&vae_path),
        te_source,
        tokenizer_dir,
        golden_override: None,
    };

    tracing::info!(
        seed,
        steps,
        width,
        height,
        dit = %dit_path,
        "starting text-to-image generation"
    );

    let start = std::time::Instant::now();
    let result =
        text_to_image(&cfg).map_err(|e| anyhow::anyhow!("text-to-image generation failed: {e}"))?;
    let elapsed = start.elapsed();

    std::fs::write(&out, &result.png).map_err(|e| anyhow::anyhow!("failed to write {out}: {e}"))?;

    println!(
        "Wrote {}x{} RGB PNG ({} bytes) to {out}",
        result.width,
        result.height,
        result.png.len()
    );
    println!(
        "  seed={seed} steps={steps} in {:.1}s",
        elapsed.as_secs_f64()
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_repl(
    seed: u64,
    steps: usize,
    width: usize,
    height: usize,
    cpu_te: bool,
    dit: Option<String>,
    vae: Option<String>,
    te: Option<String>,
    tokenizer: Option<String>,
) -> anyhow::Result<()> {
    use oxibonsai_image::pipeline::TeSource;
    use oxibonsai_image::RenderParams;

    // Same resolution as `image`: explicit arg → env → default.
    let resolve = |arg: Option<String>, env: &str, default: &str| -> String {
        arg.or_else(|| std::env::var(env).ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| default.to_string())
    };

    let dit_path = resolve(dit, "OXI_DIT_GGUF", "/tmp/parity.gguf");
    let vae_path = resolve(vae, "OXI_VAE_WEIGHTS", "/tmp/bonsai_golden/vae/weights");

    let te_path = te
        .or_else(|| std::env::var("OXI_TE_4BIT").ok().filter(|s| !s.is_empty()))
        .or_else(|| {
            std::env::var("OXI_TE_WEIGHTS")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "/tmp/bonsai_golden/te/weights".to_string());
    let te_source = if te_path.ends_with(".safetensors") {
        TeSource::Mlx4bit(PathBuf::from(&te_path))
    } else {
        TeSource::NpyDir(PathBuf::from(&te_path))
    };

    let tokenizer_dir = tokenizer
        .or_else(|| {
            std::env::var("OXI_TE_TOKENIZER_DIR")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let p = PathBuf::from(&te_path);
            if te_path.ends_with(".safetensors") {
                p.parent().map(Path::to_path_buf).unwrap_or(p)
            } else {
                p
            }
        });

    let params = RenderParams {
        prompt: String::new(),
        seed,
        steps,
        width,
        height,
    };
    let paths = repl::ReplPaths {
        dit: dit_path,
        vae: vae_path,
        te_source,
        tokenizer_dir,
    };
    repl::run(paths, params, !cpu_te)?;

    Ok(())
}
