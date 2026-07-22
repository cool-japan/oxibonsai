//! Interactive image REPL (`oxibonsai repl`).
//!
//! Loads the text-to-image pipeline once into a resident
//! [`oxibonsai_image::ImageSession`] and then reads prompts from stdin, rendering
//! each without re-loading or re-dequantising weights. On a kitty-graphics
//! terminal (Ghostty) the result is shown inline; elsewhere it is written to a
//! file. `:`-prefixed lines are commands (see [`print_help`]).

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use oxibonsai_image::pipeline::TeSource;
use oxibonsai_image::{ImageSession, RenderParams};

use super::term;

/// Resolved model asset paths for the session.
pub struct ReplPaths {
    /// DiT GGUF path.
    pub dit: String,
    /// VAE weights path (file or dir).
    pub vae: String,
    /// Text-encoder source.
    pub te_source: TeSource,
    /// Tokenizer directory.
    pub tokenizer_dir: PathBuf,
}

/// Whether a command loop iteration should keep going or exit.
enum Control {
    Continue,
    Quit,
}

/// Load the session and run the read-render loop until EOF or `:quit`.
///
/// `gpu_te` opts the text encoder onto the Metal GEMM path (`OXI_TE_GPU=1`),
/// set before the first forward so the one-shot env latch takes effect.
pub fn run(paths: ReplPaths, mut params: RenderParams, gpu_te: bool) -> anyhow::Result<()> {
    if gpu_te && std::env::var_os("OXI_TE_GPU").is_none() {
        // Safe on edition 2021; set before any TE forward so the OnceCell latch
        // in the GPU dispatch reads it.
        std::env::set_var("OXI_TE_GPU", "1");
    }

    println!("Loading models (resident)…");
    let t = std::time::Instant::now();
    let session = ImageSession::load(
        Path::new(&paths.dit),
        Path::new(&paths.vae),
        &paths.te_source,
        &paths.tokenizer_dir,
    )?;
    println!("  loaded in {:.1}s", t.elapsed().as_secs_f64());

    print!("Warming text encoder…");
    io::stdout().flush().ok();
    match session.warm() {
        Ok(d) => println!(" {:.1}s", d.as_secs_f64()),
        Err(e) => println!(" skipped ({e})"),
    }

    let inline = term::kitty_supported();
    let mut open_fallback = !inline;
    let mut out_template: Option<String> = None;
    let mut counter = 0usize;

    banner(inline, gpu_te);

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("\noxibonsai> ");
        io::stdout().flush().ok();

        let Some(line) = lines.next() else {
            break; // EOF (Ctrl-D)
        };
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(cmd) = trimmed.strip_prefix(':') {
            match handle_command(cmd, &mut params, &mut out_template, &mut open_fallback) {
                Control::Continue => continue,
                Control::Quit => break,
            }
        }

        // Anything else is a prompt.
        params.prompt = trimmed.to_string();
        match session.render(&params) {
            Ok(outcome) => {
                counter += 1;
                let path = out_template
                    .clone()
                    .unwrap_or_else(|| format!("oxibonsai-repl-{counter:03}.png"));
                if let Err(e) = std::fs::write(&path, &outcome.image.png) {
                    eprintln!("  write {path}: {e}");
                }

                if inline {
                    let mut so = io::stdout().lock();
                    if term::display_png_kitty(&mut so, &outcome.image.png).is_err() {
                        println!("  (inline display failed; wrote {path})");
                    }
                } else {
                    println!("  wrote {path}");
                    if open_fallback {
                        let _ = std::process::Command::new("open").arg(&path).status();
                    }
                }

                let ti = outcome.timings;
                println!(
                    "  {}x{}  {:.1}s total  (TE {:.1}s · DiT {:.1}s · VAE {:.1}s)  seed={} steps={}",
                    outcome.image.width,
                    outcome.image.height,
                    ti.total.as_secs_f64(),
                    ti.te_encode.as_secs_f64(),
                    ti.dit_sample.as_secs_f64(),
                    ti.vae_decode.as_secs_f64(),
                    params.seed,
                    params.steps,
                );
            }
            Err(e) => eprintln!("  render failed: {e}"),
        }
    }

    println!("bye");
    Ok(())
}

/// Dispatch a `:`-command. Returns whether to continue or quit.
fn handle_command(
    cmd: &str,
    params: &mut RenderParams,
    out_template: &mut Option<String>,
    open_fallback: &mut bool,
) -> Control {
    let mut it = cmd.split_whitespace();
    let Some(name) = it.next() else {
        return Control::Continue;
    };
    let arg = it.next();

    match name {
        "q" | "quit" | "exit" => return Control::Quit,
        "h" | "help" | "?" => print_help(),
        "show" | "set" => print_settings(params, out_template, *open_fallback),
        "steps" => match arg.and_then(|a| a.parse().ok()) {
            Some(v) => {
                params.steps = v;
                println!("  steps = {v}");
            }
            None => eprintln!("  usage: :steps N"),
        },
        "seed" => match arg.and_then(|a| a.parse().ok()) {
            Some(v) => {
                params.seed = v;
                println!("  seed = {v}");
            }
            None => eprintln!("  usage: :seed N"),
        },
        "size" => set_size(arg, params),
        "fast" => {
            params.steps = 2;
            params.width = 384;
            params.height = 384;
            println!("  preset: fast (2 steps, 384x384)");
        }
        "hq" => {
            params.steps = 8;
            params.width = 512;
            params.height = 512;
            println!("  preset: hq (8 steps, 512x512)");
        }
        "out" => {
            *out_template = arg.map(String::from);
            match out_template.as_deref() {
                Some(p) => println!("  output → {p}"),
                None => println!("  output → auto (oxibonsai-repl-NNN.png)"),
            }
        }
        "open" => match arg {
            Some("on") => {
                *open_fallback = true;
                println!("  open in viewer: on");
            }
            Some("off") => {
                *open_fallback = false;
                println!("  open in viewer: off");
            }
            _ => eprintln!("  usage: :open on|off"),
        },
        other => eprintln!("  unknown command :{other}  (try :help)"),
    }
    Control::Continue
}

/// Parse `:size WxH` or `:size N` (square).
fn set_size(arg: Option<&str>, params: &mut RenderParams) {
    let Some(a) = arg else {
        eprintln!("  usage: :size WxH  or  :size N");
        return;
    };
    let parts: Vec<&str> = a.split(['x', 'X', '*']).collect();
    let (w, h) = match parts.as_slice() {
        [w, h] => match (w.parse::<usize>(), h.parse::<usize>()) {
            (Ok(w), Ok(h)) => (w, h),
            _ => {
                eprintln!("  bad size (try :size 512x512)");
                return;
            }
        },
        [n] => match n.parse::<usize>() {
            Ok(n) => (n, n),
            Err(_) => {
                eprintln!("  bad size (try :size 512)");
                return;
            }
        },
        _ => {
            eprintln!("  bad size (try :size 512x512)");
            return;
        }
    };
    params.width = w;
    params.height = h;
    println!("  size = {w}x{h}");
}

/// Opening banner once the session is ready.
fn banner(inline: bool, gpu_te: bool) {
    println!();
    println!("OxiBonsai image REPL — type a prompt and press enter.");
    println!(
        "  display: {}   text-encoder: {}",
        if inline { "inline (kitty)" } else { "file" },
        if gpu_te { "GPU" } else { "CPU" },
    );
    println!("  :help for commands, :quit (or Ctrl-D) to exit.");
}

/// Print the command reference.
fn print_help() {
    println!(
        "\
  <text>          render <text> as an image
  :fast           preset: 2 steps, 384x384 (snappy preview)
  :hq             preset: 8 steps, 512x512 (higher quality)
  :steps N        set sampler steps
  :seed N         set RNG seed
  :size WxH       set output size (or :size N for square)
  :out PATH       write to PATH (no arg → auto oxibonsai-repl-NNN.png)
  :open on|off    open each image in a viewer (non-inline terminals)
  :show           print current settings
  :help           this help
  :quit           exit (also Ctrl-D)"
    );
}

/// Print the current render settings.
fn print_settings(params: &RenderParams, out_template: &Option<String>, open_fallback: bool) {
    println!(
        "  steps={}  seed={}  size={}x{}  out={}  open={}",
        params.steps,
        params.seed,
        params.width,
        params.height,
        out_template.as_deref().unwrap_or("auto"),
        if open_fallback { "on" } else { "off" },
    );
}
