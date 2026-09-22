//! `oxibonsai image` and `oxibonsai repl` — text-to-image generation
//! (single-shot and interactive REPL variants).
//!
//! deps-08 / RAG-EVAL-IMG-02: model paths never default to `/tmp` (a
//! world-writable directory — a shipped binary whose default is to load
//! model weights from a predictable world-writable path lets any local
//! user pre-plant a file there and have it silently loaded as weights).
//! Resolution is `--flag` → env var → a path under this process's own
//! data directory (if it already exists) → a hard error naming every
//! source checked.

use std::path::{Path, PathBuf};

use super::repl;
use super::util::{oxibonsai_data_dir, read_prompt_stdin};

/// Resolve a required model path: `--flag` → the first set env var → a
/// path under [`oxibonsai_data_dir`] IF it already exists on disk →
/// error naming every source that was checked. Never falls back to a
/// hardcoded absolute path, and never to `/tmp`.
fn resolve_required_path(
    arg: Option<String>,
    env_names: &[&str],
    default_relative: &str,
    flag_name: &str,
) -> anyhow::Result<String> {
    if let Some(v) = arg.filter(|s| !s.is_empty()) {
        return Ok(v);
    }
    for env_name in env_names {
        if let Ok(v) = std::env::var(env_name) {
            if !v.is_empty() {
                return Ok(v);
            }
        }
    }
    let default_path = oxibonsai_data_dir().join(default_relative);
    if default_path.exists() {
        return Ok(default_path.to_string_lossy().into_owned());
    }
    anyhow::bail!(
        "no path for {flag_name}: pass {flag_name} <path>, set env {} (never a hardcoded \
         default, and never /tmp — see deps-08), or place the weights at {}",
        env_names.join(" or "),
        default_path.display()
    );
}

/// Resolve the DiT / VAE / TE / tokenizer paths shared by `image` and
/// `repl`.
struct ResolvedImagePaths {
    dit_path: String,
    vae_path: String,
    te_source: oxibonsai_image::pipeline::TeSource,
    tokenizer_dir: PathBuf,
}

fn resolve_image_paths(
    dit: Option<String>,
    vae: Option<String>,
    te: Option<String>,
    tokenizer: Option<String>,
) -> anyhow::Result<ResolvedImagePaths> {
    use oxibonsai_image::pipeline::TeSource;

    let dit_path = resolve_required_path(dit, &["OXI_DIT_GGUF"], "models/dit.gguf", "--dit")?;
    let vae_path = resolve_required_path(vae, &["OXI_VAE_WEIGHTS"], "models/vae", "--vae")?;
    let te_path =
        resolve_required_path(te, &["OXI_TE_4BIT", "OXI_TE_WEIGHTS"], "models/te", "--te")?;
    let te_source = if te_path.ends_with(".safetensors") {
        TeSource::Mlx4bit(PathBuf::from(&te_path))
    } else {
        TeSource::NpyDir(PathBuf::from(&te_path))
    };

    // Tokenizer dir: --tokenizer → OXI_TE_TOKENIZER_DIR → the TE dir
    // (its parent if the TE is a safetensors file). Unlike the three
    // paths above, this one legitimately defaults relative to an
    // already-resolved, user-supplied path rather than a fixed location,
    // so it keeps its own (non-/tmp) fallback instead of
    // `resolve_required_path`.
    let tokenizer_dir = tokenizer
        .filter(|s| !s.is_empty())
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

    Ok(ResolvedImagePaths {
        dit_path,
        vae_path,
        te_source,
        tokenizer_dir,
    })
}

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
    use oxibonsai_image::pipeline::{text_to_image, TextToImageCfg};

    let prompt_text = if prompt == "-" {
        read_prompt_stdin()?
    } else {
        prompt
    };

    let paths = resolve_image_paths(dit, vae, te, tokenizer)?;

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
        dit_gguf: PathBuf::from(&paths.dit_path),
        vae_weights_dir: PathBuf::from(&paths.vae_path),
        te_source: paths.te_source,
        tokenizer_dir: paths.tokenizer_dir,
        golden_override: None,
    };

    tracing::info!(
        seed,
        steps,
        width,
        height,
        dit = %paths.dit_path,
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
    use oxibonsai_image::RenderParams;

    let paths = resolve_image_paths(dit, vae, te, tokenizer)?;

    let params = RenderParams {
        prompt: String::new(),
        seed,
        steps,
        width,
        height,
    };
    let repl_paths = repl::ReplPaths {
        dit: paths.dit_path,
        vae: paths.vae_path,
        te_source: paths.te_source,
        tokenizer_dir: paths.tokenizer_dir,
    };
    repl::run(repl_paths, params, !cpu_te)?;

    Ok(())
}
