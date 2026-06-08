//! End-to-end Pure-Rust text-to-image orchestration.
//!
//! [`text_to_image`] chains the whole Bonsai-Image pipeline behind a single
//! library entry point so both the `generate` example and the `oxibonsai image`
//! CLI subcommand share one code path (DRY):
//!
//! 1. Load the DiT GGUF ([`DitWeights::open`]), the SMALL VAE weights
//!    ([`crate::vae::VaeWeights::open`]/[`crate::vae::VaeDecoder::from_weights`]), and the Qwen3-4B text
//!    encoder ([`crate::te::TeWeights`] + [`crate::te::TextEncoder`]).
//! 2. Tokenize the prompt ([`crate::te::Qwen3Tokenizer`]) → TE forward → `cond_7680()`
//!    `[seq_txt, joint_dim]` conditioning.
//! 3. Native sampling scaffolding from [`crate::sample`]: seeded initial noise
//!    (`sample::create_noise`), the `img_ids` / `txt_ids` RoPE grids, and the
//!    flow-match `(timesteps, sigmas)` schedule — or, in parity mode
//!    ([`GoldenOverride`]), load those from a golden `.npy` dump instead.
//! 4. Euler sample ([`DitForward::sample`]) → final latent `[seq_img,
//!    in_channels]`.
//! 5. Reshape to packed NCHW (`latent_seq_to_packed_nchw`) → VAE decode
//!    ([`crate::vae::VaeDecoder::decode_packed_latents`]) → `clip(x/2 + 0.5)` → `u8`
//!    (NCHW → HWC) → PNG ([`encode_rgb8`]).
//!
//! All failures are surfaced as [`PipelineError`] (no `unwrap`/`expect`/`panic`).
//! The native noise is byte-exact against the golden `init_latents.npy` (see the
//! [`crate::sample`] module docs), so the seed-42 native path reproduces the
//! golden-input render.

use std::path::PathBuf;

use crate::png::encode_rgb8;
use crate::sample;
use crate::te::{Qwen3Tokenizer, TeError, TeWeights, TextEncoder};
use crate::vae::{VaeDecoder, VaeError, VaeWeights};
use crate::{DitError, DitForward, DitWeights, PngError, StepTap};

/// Where the text-encoder weights come from.
#[derive(Debug, Clone)]
pub enum TeSource {
    /// The native 4-bit MLX `model.safetensors` (≈2.1 GB), loaded via
    /// [`TeWeights::open_mlx_4bit`].
    Mlx4bit(PathBuf),
    /// A directory of f32 `.npy` weight dumps (≈15 GB), loaded via
    /// [`TeWeights::open`].
    NpyDir(PathBuf),
}

/// Parity-mode overrides: load the DiT inputs (initial latent, position ids, and
/// the sampler schedule) — and optionally the whole conditioning / a pre-sampled
/// latent — from a golden `.npy` dump instead of generating them natively.
///
/// This preserves the `generate` example's `OXI_USE_GOLDEN_LATENT` /
/// golden-cond byte-for-byte parity behaviour. When `None`, everything is
/// generated natively from `(prompt, seed, width, height)`.
#[derive(Debug, Clone)]
pub struct GoldenOverride {
    /// Directory containing the bf16 golden tensors (`tf_in_hidden_states.npy`,
    /// `img_ids.npy`, `txt_ids.npy`, `timesteps.npy`, `sigmas.npy`,
    /// `cond.npy`, `latent_after_step3.npy`, …).
    pub golden_dir: PathBuf,
    /// Use `cond.npy` from `golden_dir` instead of the native TE conditioning.
    pub use_golden_cond: bool,
    /// Skip the DiT sampler and use `latent_after_step3.npy` directly.
    pub use_golden_latent: bool,
    /// Optional VAE golden dir (`vae_decoded.npy`) for the decode-cosine tap.
    pub vae_golden_dir: Option<PathBuf>,
}

/// Configuration for a single text-to-image generation.
#[derive(Debug, Clone)]
pub struct TextToImageCfg {
    /// The text prompt.
    pub prompt: String,
    /// RNG seed for the initial noise (drives generation natively).
    pub seed: u64,
    /// Number of Euler sampler steps.
    pub steps: usize,
    /// Target image width in pixels.
    pub width: usize,
    /// Target image height in pixels.
    pub height: usize,
    /// Guidance scale (reserved; the current DiT forward is unconditional —
    /// kept for API stability and surfaced in logs).
    pub guidance: f32,
    /// Path to the DiT GGUF file.
    pub dit_gguf: PathBuf,
    /// VAE weights source: either a `.safetensors` file (the native FLUX.2
    /// `AutoencoderKLFlux2` checkpoint) or a directory of exported `.npy`
    /// tensors — [`VaeWeights::open`] auto-detects which.
    pub vae_weights_dir: PathBuf,
    /// Where the text-encoder weights come from.
    pub te_source: TeSource,
    /// Directory containing `tokenizer.json`.
    pub tokenizer_dir: PathBuf,
    /// When `Some`, run in golden-parity mode (see [`GoldenOverride`]).
    pub golden_override: Option<GoldenOverride>,
}

/// The result of a generation.
pub struct TextToImageOut {
    /// The encoded PNG byte stream.
    pub png: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Per-stage cosine similarities vs the golden (`(label, cosine)`). Empty
    /// unless `golden_override` was set and the corresponding golden tensors
    /// were available.
    pub stage_cosines: Vec<(String, f32)>,
}

/// Errors that can occur while running [`text_to_image`].
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// A DiT load / forward error.
    #[error("DiT error: {0}")]
    Dit(#[from] DitError),
    /// A text-encoder load / forward / tokenizer error.
    #[error("text-encoder error: {0}")]
    Te(#[from] TeError),
    /// A VAE load / decode error.
    #[error("VAE error: {0}")]
    Vae(#[from] VaeError),
    /// A PNG-encoding error.
    #[error("PNG error: {0}")]
    Png(#[from] PngError),
    /// An I/O error (reading a golden tensor, a missing weights path, …).
    #[error("I/O error for {path}: {source}")]
    Io {
        /// The path that triggered the error.
        path: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// A shape / configuration invariant was violated.
    #[error("pipeline shape error: {0}")]
    Shape(String),
    /// A required input path was missing.
    #[error("missing input: {0}")]
    MissingInput(String),
}

/// A loaded `.npy` tensor (f32, C-order).
struct Npy {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Npy {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Minimal NumPy `.npy` reader: v1.0/2.0 header, `descr == '<f4'`, handling
/// `fortran_order == True` by reordering to C (row-major). Used only by the
/// golden-parity path.
fn read_npy(path: &std::path::Path) -> Result<Npy, PipelineError> {
    let bytes = std::fs::read(path).map_err(|e| PipelineError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(PipelineError::Shape(format!(
            "{}: bad npy magic",
            path.display()
        )));
    }
    let major = bytes[6];
    let (header_start, header_len) = if major >= 2 {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (12usize, len)
    } else {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (10usize, len)
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err(PipelineError::Shape(format!(
            "{}: truncated npy header",
            path.display()
        )));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| PipelineError::Shape(format!("{}: header utf8: {e}", path.display())))?;
    if !header.contains("'<f4'") {
        return Err(PipelineError::Shape(format!(
            "{}: descr is not '<f4': {header}",
            path.display()
        )));
    }
    let fortran = header.contains("'fortran_order': True");
    let s_idx = header
        .find("'shape':")
        .ok_or_else(|| PipelineError::Shape(format!("{}: no shape key", path.display())))?;
    let open = header[s_idx..]
        .find('(')
        .map(|o| s_idx + o + 1)
        .ok_or_else(|| PipelineError::Shape(format!("{}: no shape open paren", path.display())))?;
    let close = header[open..]
        .find(')')
        .map(|c| open + c)
        .ok_or_else(|| PipelineError::Shape(format!("{}: no shape close paren", path.display())))?;
    let shape: Vec<usize> = header[open..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| PipelineError::Shape(format!("{}: shape parse: {e}", path.display())))
        })
        .collect::<Result<_, _>>()?;
    let payload = &bytes[header_end..];
    if payload.len() % 4 != 0 {
        return Err(PipelineError::Shape(format!(
            "{}: payload not f32-aligned",
            path.display()
        )));
    }
    let raw: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let numel: usize = shape.iter().product();
    if raw.len() < numel {
        return Err(PipelineError::Shape(format!(
            "{}: payload short ({} < {numel})",
            path.display(),
            raw.len()
        )));
    }
    let data = if fortran && shape.len() > 1 {
        fortran_to_c(&raw[..numel], &shape)
    } else {
        raw
    };
    Ok(Npy { data, shape })
}

/// Reorder a Fortran-stored (column-major) buffer into C (row-major) order.
fn fortran_to_c(src: &[f32], shape: &[usize]) -> Vec<f32> {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    let mut f_stride = vec![1usize; ndim];
    for d in 1..ndim {
        f_stride[d] = f_stride[d - 1] * shape[d - 1];
    }
    let mut out = vec![0.0f32; numel];
    for (c_pos, slot) in out.iter_mut().enumerate() {
        let mut rem = c_pos;
        let mut f_off = 0usize;
        for d in 0..ndim {
            let stride_c: usize = shape[d + 1..].iter().product();
            let idx = rem / stride_c;
            rem %= stride_c;
            f_off += idx * f_stride[d];
        }
        *slot = src[f_off];
    }
    out
}

/// Load a named golden tensor from `dir` (`<dir>/<name>.npy`).
fn load_golden(dir: &std::path::Path, name: &str) -> Result<Npy, PipelineError> {
    read_npy(&dir.join(format!("{name}.npy")))
}

/// Cosine similarity between two equal-length slices (f64 accumulation).
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

/// Reshape the DiT's final latent `[seq_img, in_channels]` (row-major,
/// sequence-major) into the VAE's packed NCHW latent `[1, in_channels, ph, pw]`.
///
/// This mirrors `flux2_klein`'s `reshape(1, ph, pw, in_channels).transpose(0, 3,
/// 1, 2)`: the sequence index `s = hh*pw + ww` (h-major over a `pw`-wide grid)
/// becomes the spatial position and the channel becomes the leading axis.
///
/// # Errors
/// [`PipelineError::Shape`] if `seq_img != ph*pw` or the latent length is wrong.
pub fn latent_seq_to_packed_nchw(
    latent: &[f32],
    seq_img: usize,
    in_channels: usize,
    ph: usize,
    pw: usize,
) -> Result<Vec<f32>, PipelineError> {
    if seq_img != ph * pw {
        return Err(PipelineError::Shape(format!(
            "seq_img {seq_img} != ph*pw ({ph}*{pw})"
        )));
    }
    if latent.len() != seq_img * in_channels {
        return Err(PipelineError::Shape(format!(
            "latent len {} != seq_img*in_channels ({seq_img}*{in_channels})",
            latent.len()
        )));
    }
    let plane = ph * pw;
    let mut packed = vec![0.0f32; in_channels * plane];
    for hh in 0..ph {
        for ww in 0..pw {
            let s = hh * pw + ww;
            let src_base = s * in_channels;
            for ch in 0..in_channels {
                packed[ch * plane + hh * pw + ww] = latent[src_base + ch];
            }
        }
    }
    Ok(packed)
}

/// Compute the `[seq_txt, joint_dim]` conditioning natively from the prompt.
fn compute_cond(
    cfg: &TextToImageCfg,
    seq_txt: usize,
    joint_dim: usize,
) -> Result<Vec<f32>, PipelineError> {
    let tok = Qwen3Tokenizer::open(&cfg.tokenizer_dir)?;
    let toks = tok.tokenize(&cfg.prompt, seq_txt)?;

    let weights = match &cfg.te_source {
        TeSource::Mlx4bit(path) => TeWeights::open_mlx_4bit(path)?,
        TeSource::NpyDir(dir) => TeWeights::open(dir)?,
    };
    let encoder = TextEncoder::new(&weights);
    let out = encoder.forward(&toks.input_ids, &toks.attention_mask)?;
    let cond = out.cond_7680()?;
    if cond.len() != seq_txt * joint_dim {
        return Err(PipelineError::Shape(format!(
            "TE cond len {} != seq_txt*joint_dim ({seq_txt}*{joint_dim})",
            cond.len()
        )));
    }
    Ok(cond)
}

/// Whether a VAE weights `path` can be loaded by [`VaeWeights::open`].
///
/// [`VaeWeights::open`] accepts **either** a `.safetensors` file (the native
/// FLUX.2 `AutoencoderKLFlux2` checkpoint) **or** a directory of exported `.npy`
/// tensors, so the pipeline precheck must accept both (mirroring the loader,
/// rather than the stricter `is_dir` that rejected a documented file path —
/// see issue #9). Any existing path passes here; the loader then validates the
/// contents.
fn vae_path_present(path: &std::path::Path) -> bool {
    path.is_file() || path.is_dir()
}

/// Convert a VAE-decoded planar CHW f32 tensor into a row-major HWC u8 RGB
/// buffer: `px = clip(x / 2 + 0.5, 0, 1) * 255`.
///
/// Shared by [`text_to_image`] and [`crate::session::ImageSession`] so both
/// produce byte-identical pixels. Returns `(width, height, rgb)`.
///
/// # Errors
/// [`PipelineError::Shape`] if `c != 3`.
pub(crate) fn decoded_chw_to_rgb8(
    c: usize,
    h: usize,
    w: usize,
    data: &[f32],
) -> Result<(usize, usize, Vec<u8>), PipelineError> {
    if c != 3 {
        return Err(PipelineError::Shape(format!(
            "expected 3 output channels, got {c}"
        )));
    }
    let plane = h * w;
    let mut rgb = vec![0u8; data.len()];
    for y in 0..h {
        for x in 0..w {
            let hw = y * w + x;
            let dst = (y * w + x) * 3;
            for ch in 0..3 {
                let v = (data[ch * plane + hw] / 2.0 + 0.5).clamp(0.0, 1.0);
                rgb[dst + ch] = (v * 255.0).round() as u8;
            }
        }
    }
    Ok((w, h, rgb))
}

/// Run the whole text→image pipeline and return the encoded PNG.
///
/// See the [module docs](self) for the stage-by-stage flow.
///
/// # Errors
/// [`PipelineError`] wrapping whichever stage failed (DiT/TE/VAE/PNG/IO/shape).
pub fn text_to_image(cfg: &TextToImageCfg) -> Result<TextToImageOut, PipelineError> {
    let mut stage_cosines: Vec<(String, f32)> = Vec::new();
    let timing = std::env::var("OXI_IMAGE_TIMING").is_ok();

    // ── 1. Load DiT + VAE ──
    if !cfg.dit_gguf.exists() {
        return Err(PipelineError::MissingInput(format!(
            "DiT GGUF not found: {}",
            cfg.dit_gguf.display()
        )));
    }
    if !vae_path_present(&cfg.vae_weights_dir) {
        return Err(PipelineError::MissingInput(format!(
            "VAE weights not found: {}",
            cfg.vae_weights_dir.display()
        )));
    }
    let t_load_dit = std::time::Instant::now();
    let dit_weights = DitWeights::open(&cfg.dit_gguf)?;
    let dcfg = dit_weights.config();
    let in_channels = dcfg.in_channels as usize;
    let joint_dim = dcfg.joint_attention_dim as usize;
    let num_axes = dcfg.axes_dims_rope.len();
    if timing {
        eprintln!(
            "[timing] load DiT: {:.2}s",
            t_load_dit.elapsed().as_secs_f64()
        );
    }

    let t_load_vae = std::time::Instant::now();
    let vae_weights = VaeWeights::open(&cfg.vae_weights_dir)?;
    let vae = VaeDecoder::from_weights(&vae_weights)?;
    if timing {
        eprintln!(
            "[timing] load VAE: {:.2}s",
            t_load_vae.elapsed().as_secs_f64()
        );
    }

    // ── 2. Geometry from the requested resolution ──
    let (lat_h, lat_w) = sample::latent_grid(cfg.height, cfg.width);
    let seq_img = lat_h * lat_w;
    // The text sequence length is the tokenizer's fixed pad length.
    let seq_txt = 512usize;

    let golden = cfg.golden_override.as_ref();

    // ── 3. Conditioning: native TE or golden cond ──
    let cond_vec: Vec<f32> = match golden {
        Some(g) if g.use_golden_cond => {
            let c = load_golden(&g.golden_dir, "cond")?;
            if c.numel() < seq_txt * joint_dim {
                return Err(PipelineError::Shape(format!(
                    "golden cond numel {} < {seq_txt}*{joint_dim}",
                    c.numel()
                )));
            }
            c.data[..seq_txt * joint_dim].to_vec()
        }
        _ => {
            let t_te = std::time::Instant::now();
            let cond = compute_cond(cfg, seq_txt, joint_dim)?;
            if timing {
                eprintln!("[timing] TE encode: {:.2}s", t_te.elapsed().as_secs_f64());
            }
            // Optional cosine vs golden cond (sanity), if a golden dir is present.
            if let Some(g) = golden {
                if let Ok(gc) = load_golden(&g.golden_dir, "cond") {
                    if gc.numel() >= cond.len() {
                        let cos = cosine(&cond, &gc.data[..cond.len()]) as f32;
                        stage_cosines.push(("cond_vs_golden".to_string(), cos));
                    }
                }
            }
            cond
        }
    };

    // ── 4. Final latent [seq_img, in_channels] ──
    let final_latent: Vec<f32> = match golden {
        Some(g) if g.use_golden_latent => {
            let glat = load_golden(&g.golden_dir, "latent_after_step3")?;
            if glat.numel() != seq_img * in_channels {
                return Err(PipelineError::Shape(format!(
                    "latent_after_step3 numel {} != {seq_img}*{in_channels}",
                    glat.numel()
                )));
            }
            glat.data
        }
        _ => {
            let fwd = DitForward::new(&dit_weights);

            // Initial latent + position ids + schedule: golden (parity) or native.
            let (init, img_ids, txt_ids, timesteps, sigmas) = match golden {
                Some(g) => {
                    let init = load_golden(&g.golden_dir, "tf_in_hidden_states")?;
                    let img_ids = load_golden(&g.golden_dir, "img_ids")?;
                    let txt_ids = load_golden(&g.golden_dir, "txt_ids")?;
                    let timesteps = load_golden(&g.golden_dir, "timesteps")?;
                    let sigmas = load_golden(&g.golden_dir, "sigmas")?;
                    (
                        init.data[..seq_img * in_channels].to_vec(),
                        img_ids.data[..seq_img * num_axes].to_vec(),
                        txt_ids.data[..seq_txt * num_axes].to_vec(),
                        timesteps.data,
                        sigmas.data,
                    )
                }
                None => {
                    let init = sample::create_noise(cfg.seed, cfg.height, cfg.width);
                    if init.len() != seq_img * in_channels {
                        return Err(PipelineError::Shape(format!(
                            "native noise len {} != {seq_img}*{in_channels} \
                             (in_channels {in_channels} vs PACKED_CHANNELS {})",
                            init.len(),
                            sample::PACKED_CHANNELS
                        )));
                    }
                    let img_ids = sample::img_ids(lat_h, lat_w);
                    let txt_ids = sample::txt_ids(seq_txt);
                    let (timesteps, sigmas) = sample::flow_match_schedule(seq_img, cfg.steps);
                    (init, img_ids, txt_ids, timesteps, sigmas)
                }
            };

            let mut steps: Vec<StepTap> = Vec::new();
            let t_sample = std::time::Instant::now();
            let latent = fwd.sample(
                &init,
                &cond_vec,
                &img_ids,
                &txt_ids,
                seq_img,
                seq_txt,
                &timesteps,
                &sigmas,
                Some(&mut steps),
            )?;
            if timing {
                eprintln!(
                    "[timing] DiT sample ({} steps): {:.2}s",
                    timesteps.len(),
                    t_sample.elapsed().as_secs_f64()
                );
            }

            // Optional cosine vs golden step-3 latent.
            if let Some(g) = golden {
                if let Ok(glat) = load_golden(&g.golden_dir, "latent_after_step3") {
                    if glat.numel() == latent.len() {
                        let cos = cosine(&latent, &glat.data) as f32;
                        stage_cosines.push(("latent_vs_golden".to_string(), cos));
                    }
                }
            }
            latent
        }
    };

    // ── 5. Reshape → packed NCHW, VAE-decode ──
    let (ph, pw) = (lat_h, lat_w);
    let packed = latent_seq_to_packed_nchw(&final_latent, seq_img, in_channels, ph, pw)?;
    let t_vae = std::time::Instant::now();
    let decoded = vae.decode_packed_latents(&packed, ph, pw, None)?;
    if timing {
        eprintln!("[timing] VAE decode: {:.2}s", t_vae.elapsed().as_secs_f64());
    }

    if let Some(g) = golden {
        if let Some(vae_golden_dir) = &g.vae_golden_dir {
            if let Ok(gd) = load_golden(vae_golden_dir, "vae_decoded") {
                if gd.numel() == decoded.data.len() {
                    let cos = cosine(&decoded.data, &gd.data) as f32;
                    stage_cosines.push(("vae_vs_golden".to_string(), cos));
                }
            }
        }
    }

    // ── 6. px = clip(x/2 + 0.5, 0, 1); NCHW → HWC; u8 ──
    let (w, h, rgb) = decoded_chw_to_rgb8(decoded.c, decoded.h, decoded.w, &decoded.data)?;

    // ── 7. PNG-encode ──
    let t_png = std::time::Instant::now();
    let png = encode_rgb8(w, h, &rgb)?;
    if timing {
        eprintln!("[timing] PNG encode: {:.2}s", t_png.elapsed().as_secs_f64());
    }

    Ok(TextToImageOut {
        png,
        width: w,
        height: h,
        stage_cosines,
    })
}

#[cfg(test)]
mod tests {
    use super::vae_path_present;
    use std::path::PathBuf;

    /// Build a unique scratch path under the system temp dir (policy: tests must
    /// use `std::env::temp_dir()`), tagged by `label` and the test's process id
    /// + nanos so concurrent / repeated runs never collide.
    fn scratch(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "oxibonsai_issue9_{label}_{}_{nanos}",
            std::process::id()
        ))
    }

    /// Regression test for issue #9: passing `--vae` as a `.safetensors` FILE
    /// (exactly as docs/CLI.md and docs/IMAGEN.md instruct, and exactly what
    /// `VaeWeights::open` accepts) must be accepted by the pipeline precheck.
    ///
    /// Before the fix the precheck was `cfg.vae_weights_dir.is_dir()`, so a real
    /// `.safetensors` file returned `false` here → the pipeline aborted with
    /// "VAE weights dir not found" before any loading. This asserts the predicate
    /// now returns `true` for such a file.
    #[test]
    fn test_issue_9_vae_safetensors_file_accepted() -> std::io::Result<()> {
        let file = scratch("vae").with_extension("safetensors");
        // A real, existing file on disk (contents irrelevant to the precheck —
        // the precheck only gates existence; `VaeWeights::open` validates bytes).
        std::fs::write(&file, b"not-a-real-safetensors-but-a-real-file")?;

        let present = vae_path_present(&file);
        std::fs::remove_file(&file)?;

        assert!(
            present,
            "issue #9: a .safetensors FILE must pass the VAE precheck (got false), \
             path = {}",
            file.display()
        );
        Ok(())
    }

    /// Companion: a directory of exported `.npy` weights (the original dev-time
    /// source) must still be accepted — the fix widens the predicate, it must not
    /// regress the directory case.
    #[test]
    fn test_issue_9_vae_directory_still_accepted() -> std::io::Result<()> {
        let dir = scratch("vae_dir");
        std::fs::create_dir_all(&dir)?;

        let present = vae_path_present(&dir);
        std::fs::remove_dir_all(&dir)?;

        assert!(
            present,
            "a VAE weights directory must still pass the precheck, path = {}",
            dir.display()
        );
        Ok(())
    }

    /// Companion: a path that exists as neither a file nor a directory must still
    /// be rejected, so a genuine typo / missing input is still caught early.
    #[test]
    fn test_issue_9_vae_nonexistent_rejected() {
        let missing = scratch("vae_missing").with_extension("safetensors");
        // Deliberately never created.
        assert!(
            !vae_path_present(&missing),
            "a nonexistent path must be rejected by the precheck, path = {}",
            missing.display()
        );
    }
}
