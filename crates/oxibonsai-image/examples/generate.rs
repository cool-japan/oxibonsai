//! Pure-Rust end-to-end Bonsai-Image artifact: (text encoder →) DiT sampler →
//! VAE decoder → PNG.
//!
//! This is a thin wrapper over [`oxibonsai_image::pipeline::text_to_image`]: it
//! parses the example's args/env and builds a [`TextToImageCfg`], then writes
//! the returned PNG. The shared library function does the actual orchestration
//! (chaining the Qwen3-4B text encoder, the parity-validated FLUX.2 Klein DiT
//! Euler sampler, the FLUX.2 SMALL VAE decoder, and the Pure-Rust PNG encoder).
//!
//! Because this example is the parity/golden harness, it always runs in
//! golden-parity mode ([`GoldenOverride`]): the DiT init latent / position ids /
//! schedule come from the golden `.npy` dump (which is byte-identical to the
//! native `sample::*` scaffolding), so its output reproduces the earlier
//! golden-input render exactly. The native, seed-driven path (native noise +
//! native ids/schedule) is exercised by the `oxibonsai image` CLI subcommand
//! (which calls the same library function with `golden_override: None`).
//!
//! Usage:
//!
//! ```text
//! # Full Pure-Rust text→image from a PROMPT: tokenize → Qwen3 TE → 7680 cond →
//! # DiT sample → VAE → PNG. (~7-8 min for the DiT sample; the TE is seconds.)
//! cargo run --release -p oxibonsai-image --example generate -- \
//!     --prompt "a tiny bonsai tree in a ceramic pot" /tmp/bonsai_rust_out.png
//!
//! # Fast iteration: reuse the golden step-3 latent, skip the sampler.
//! OXI_USE_GOLDEN_LATENT=1 cargo run --release -p oxibonsai-image \
//!     --example generate -- /tmp/bonsai_rust_out.png
//!
//! # Golden-cond end-to-end (no prompt): run the DiT sampler then VAE + PNG.
//! cargo run --release -p oxibonsai-image --example generate -- /tmp/bonsai_rust_out.png
//! ```
//!
//! When `--prompt` is given, the conditioning is produced natively by the Rust
//! text encoder (no golden `cond.npy`); its cosine vs the golden cond is printed.
//!
//! Env:
//! - `OXI_USE_GOLDEN_LATENT=1` — use the golden `latent_after_step3` instead of
//!   sampling (fast path).
//! - `OXI_DIT_GGUF` — DiT GGUF path (default `/tmp/parity.gguf`).
//! - `OXI_GOLDEN_DIR` — bf16 golden dir (default `/tmp/bonsai_golden/bf16`).
//! - `OXI_VAE_WEIGHTS` — VAE weights dir (default `/tmp/bonsai_golden/vae/weights`).
//! - `OXI_VAE_GOLDEN` — VAE golden dir (default `/tmp/bonsai_golden/vae`).
//! - `OXI_TE_WEIGHTS` — TE f32 weights dir (default `/tmp/bonsai_golden/te/weights`).
//! - `OXI_TE_4BIT` — path to the native 4-bit `model.safetensors`. When set, the
//!   TE loads from this 2.1 GB file instead of the 15 GB f32 `.npy` dir.
//! - `OXI_TE_TOKENIZER_DIR` — dir with `tokenizer.json` (default: the 4-bit model dir).

use std::path::PathBuf;
use std::process::ExitCode;

use oxibonsai_image::pipeline::{text_to_image, GoldenOverride, TeSource, TextToImageCfg};

fn env_path(key: &str, default: &str) -> PathBuf {
    std::env::var(key)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

/// Parsed CLI arguments: an optional `--prompt`, `--seed`, and the output PNG
/// path (positional or `--out`).
struct Args {
    prompt: Option<String>,
    seed: u64,
    out_path: PathBuf,
}

/// Parse `[--prompt <text>] [--seed <n>] [--out <path> | <out.png>]`
/// (order-independent). In golden-parity mode the seed has no effect (the init
/// latent is loaded from the golden dump), but it is accepted so the documented
/// commands parse cleanly.
fn parse_args() -> Args {
    let mut prompt = None;
    let mut seed = 42u64;
    let mut out_path = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--prompt" | "-p" => prompt = it.next(),
            "--seed" => {
                if let Some(v) = it.next() {
                    if let Ok(n) = v.parse::<u64>() {
                        seed = n;
                    }
                }
            }
            "--out" | "-o" => {
                if let Some(v) = it.next() {
                    out_path = Some(PathBuf::from(v));
                }
            }
            other => {
                if out_path.is_none() {
                    out_path = Some(PathBuf::from(other));
                }
            }
        }
    }
    Args {
        prompt,
        seed,
        out_path: out_path.unwrap_or_else(|| PathBuf::from("/tmp/bonsai_rust_out.png")),
    }
}

fn run() -> Result<(), String> {
    let args = parse_args();
    let out_path = args.out_path;

    let gguf = env_path("OXI_DIT_GGUF", "/tmp/parity.gguf");
    let golden_dir = env_path("OXI_GOLDEN_DIR", "/tmp/bonsai_golden/bf16");
    let vae_weights_dir = env_path("OXI_VAE_WEIGHTS", "/tmp/bonsai_golden/vae/weights");
    let vae_golden_dir = env_path("OXI_VAE_GOLDEN", "/tmp/bonsai_golden/vae");
    let te_weights_dir = env_path("OXI_TE_WEIGHTS", "/tmp/bonsai_golden/te/weights");
    let te_tokenizer_dir = env_path("OXI_TE_TOKENIZER_DIR", "/path/to/text_encoder-mlx-4bit");
    let use_golden_latent = std::env::var("OXI_USE_GOLDEN_LATENT").is_ok();

    // TE source: the native 2.1 GB 4-bit safetensors when `OXI_TE_4BIT` is set,
    // else the f32 `.npy` dir.
    let te_source = match std::env::var("OXI_TE_4BIT").ok().filter(|p| !p.is_empty()) {
        Some(p) => TeSource::Mlx4bit(PathBuf::from(p)),
        None => TeSource::NpyDir(te_weights_dir),
    };

    println!(
        "Kernel tier: {:?}",
        oxibonsai_kernels::KernelDispatcher::auto_detect().tier()
    );
    match &te_source {
        TeSource::Mlx4bit(p) => println!("TE source: 4-bit safetensors {}", p.display()),
        TeSource::NpyDir(d) => println!("TE source: f32 npy {}", d.display()),
    }
    println!("DiT GGUF:   {}", gguf.display());
    println!("VAE weights: {}", vae_weights_dir.display());

    // This example is the golden/parity harness: always run in golden-parity
    // mode so the DiT init latent / ids / schedule come from the golden dump
    // (byte-identical to the native `sample::*` scaffolding). `--prompt`
    // produces the conditioning natively; otherwise the golden `cond.npy` is
    // used. `OXI_USE_GOLDEN_LATENT` skips the sampler.
    let has_prompt = args.prompt.is_some();
    let golden_override = Some(GoldenOverride {
        golden_dir: golden_dir.clone(),
        use_golden_cond: !has_prompt,
        use_golden_latent,
        vae_golden_dir: if vae_golden_dir.is_dir() {
            Some(vae_golden_dir)
        } else {
            None
        },
    });

    let cfg = TextToImageCfg {
        prompt: args.prompt.unwrap_or_default(),
        seed: args.seed,
        steps: 4,
        width: 512,
        height: 512,
        guidance: 1.0,
        dit_gguf: gguf,
        vae_weights_dir,
        te_source,
        tokenizer_dir: te_tokenizer_dir,
        golden_override,
    };

    if has_prompt {
        println!(
            "\n=== Text encoder (Qwen3-4B): prompt = {:?} ===",
            cfg.prompt
        );
    } else {
        println!("\n=== Using golden cond.npy (no --prompt) ===");
    }
    if use_golden_latent {
        println!("=== Using golden latent (OXI_USE_GOLDEN_LATENT=1) — skipping DiT sample ===");
    } else {
        println!("=== Running DiT Euler sampler + untiled VAE decode (a few minutes) ===");
    }

    let t = std::time::Instant::now();
    let out = text_to_image(&cfg).map_err(|e| e.to_string())?;
    println!("  (pipeline ran in {:.1}s)", t.elapsed().as_secs_f64());

    // Report the per-stage cosines the pipeline captured against the golden.
    for (label, cos) in &out.stage_cosines {
        match label.as_str() {
            "cond_vs_golden" => {
                println!("  TE cond vs golden cond.npy: cosine = {cos:.6} (expect >= 0.999)")
            }
            "latent_vs_golden" => {
                println!("  sampled-latent vs latent_after_step3: cosine = {cos:.6}")
            }
            "vae_vs_golden" => {
                println!("  decoded vs vae_decoded: cosine = {cos:.6} (expect >= 0.999)")
            }
            other => println!("  {other}: cosine = {cos:.6}"),
        }
    }

    std::fs::write(&out_path, &out.png)
        .map_err(|e| format!("write {}: {e}", out_path.display()))?;
    println!(
        "\n  Wrote PNG: {} ({} bytes, {}x{} RGB)",
        out_path.display(),
        out.png.len(),
        out.width,
        out.height
    );

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => {
            println!("\nDONE");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}
