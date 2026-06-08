//! Resident image-generation session.
//!
//! [`ImageSession`] loads the DiT, VAE, text encoder, and tokenizer **once** and
//! holds them in memory, so a long-running front-end (e.g. the `oxibonsai repl`
//! command) can render many prompts without re-paying the per-call load and
//! 4-bit dequant cost. The text encoder is opened in *resident* mode
//! ([`TeWeights::set_resident`]): its dequantised f32 weights (~16 GB) stay
//! cached across renders instead of being thrown away after each forward. That
//! trades RAM for speed and is intended for high-memory machines.
//!
//! The per-render math is identical to the native (non-golden) path of
//! [`crate::pipeline::text_to_image`] — both share
//! [`crate::pipeline::decoded_chw_to_rgb8`] and the same `sample`/`forward`
//! stages, so a session render and a one-shot render of the same
//! `(prompt, seed, steps, size)` produce byte-identical PNGs.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::pipeline::{
    decoded_chw_to_rgb8, latent_seq_to_packed_nchw, PipelineError, TeSource, TextToImageOut,
};
use crate::png::encode_rgb8;
use crate::sample;
use crate::te::{Qwen3Tokenizer, TeWeights, TextEncoder};
use crate::vae::{VaeDecoder, VaeWeights};
use crate::{DitForward, DitWeights};

/// The fixed text sequence length the DiT conditioning expects (tokenizer pad).
const SEQ_TXT: usize = 512;

/// Per-render knobs. Mirrors the relevant fields of
/// [`crate::pipeline::TextToImageCfg`], minus the model paths (those are fixed
/// for the life of the session) and golden-parity overrides.
#[derive(Debug, Clone)]
pub struct RenderParams {
    /// The text prompt.
    pub prompt: String,
    /// RNG seed for the initial noise.
    pub seed: u64,
    /// Number of Euler sampler steps.
    pub steps: usize,
    /// Target width in pixels.
    pub width: usize,
    /// Target height in pixels.
    pub height: usize,
    /// Guidance scale (surfaced for parity; the DiT forward is unconditional).
    pub guidance: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            seed: 42,
            steps: 4,
            width: 512,
            height: 512,
            guidance: 1.0,
        }
    }
}

/// Wall-clock split of a single [`ImageSession::render`], so a front-end can
/// show where the time went (the same stages the `OXI_IMAGE_TIMING` taps print).
#[derive(Debug, Clone, Copy)]
pub struct StageTimings {
    /// Tokenize + text-encoder forward.
    pub te_encode: Duration,
    /// DiT flow-matching sampler (all steps).
    pub dit_sample: Duration,
    /// VAE decode.
    pub vae_decode: Duration,
    /// CHW→RGB + PNG encode.
    pub png_encode: Duration,
    /// End-to-end render time.
    pub total: Duration,
}

/// The result of one render: the encoded image plus its stage timings.
pub struct RenderOutcome {
    /// The encoded PNG and its dimensions.
    pub image: TextToImageOut,
    /// Per-stage wall-clock split.
    pub timings: StageTimings,
}

/// A loaded, resident text-to-image pipeline. Build once with [`Self::load`],
/// then call [`Self::render`] per prompt.
pub struct ImageSession {
    dit_weights: DitWeights,
    vae: VaeDecoder,
    te_weights: TeWeights,
    tokenizer: Qwen3Tokenizer,
    in_channels: usize,
    joint_dim: usize,
}

impl ImageSession {
    /// Load every model asset and keep it resident.
    ///
    /// `te_source` selects the 4-bit MLX safetensors or the f32 `.npy` dir, same
    /// as [`crate::pipeline::TextToImageCfg`]. The text encoder is put in
    /// resident mode so its dequantised weights persist across renders.
    ///
    /// # Errors
    /// [`PipelineError`] wrapping whichever asset failed to load.
    pub fn load(
        dit_gguf: &Path,
        vae_weights: &Path,
        te_source: &TeSource,
        tokenizer_dir: &Path,
    ) -> Result<Self, PipelineError> {
        let dit_weights = DitWeights::open(dit_gguf)?;
        let dcfg = dit_weights.config();
        let in_channels = dcfg.in_channels as usize;
        let joint_dim = dcfg.joint_attention_dim as usize;

        // VaeDecoder owns its weights, so the loader's buffers can be released.
        let vae_loaded = VaeWeights::open(vae_weights)?;
        let vae = VaeDecoder::from_weights(&vae_loaded)?;
        drop(vae_loaded);

        let te_weights = match te_source {
            TeSource::Mlx4bit(p) => TeWeights::open_mlx_4bit(p)?,
            TeSource::NpyDir(d) => TeWeights::open(d)?,
        };
        te_weights.set_resident(true);

        let tokenizer = Qwen3Tokenizer::open(tokenizer_dir)?;

        Ok(Self {
            dit_weights,
            vae,
            te_weights,
            tokenizer,
            in_channels,
            joint_dim,
        })
    }

    /// Populate the resident text-encoder cache so the *first* real render runs
    /// at warm speed. A single short encode touches every layer's weights, so
    /// one pass dequantises the whole encoder. Returns how long warming took.
    ///
    /// DiT and VAE weights are memory-mapped and page in lazily; this only warms
    /// the encoder, which is the dominant amortizable (re-dequant) cost.
    ///
    /// # Errors
    /// [`PipelineError`] if the warm-up encode fails.
    pub fn warm(&self) -> Result<Duration, PipelineError> {
        let t = Instant::now();
        let _ = self.encode("warmup")?;
        Ok(t.elapsed())
    }

    /// Tokenize and text-encode `prompt` into the `[SEQ_TXT * joint_dim]`
    /// conditioning vector.
    fn encode(&self, prompt: &str) -> Result<Vec<f32>, PipelineError> {
        let toks = self.tokenizer.tokenize(prompt, SEQ_TXT)?;
        let encoder = TextEncoder::new(&self.te_weights);
        let out = encoder.forward(&toks.input_ids, &toks.attention_mask)?;
        let cond = out.cond_7680()?;
        let need = SEQ_TXT * self.joint_dim;
        if cond.len() != need {
            return Err(PipelineError::Shape(format!(
                "TE cond len {} != SEQ_TXT*joint_dim ({SEQ_TXT}*{})",
                cond.len(),
                self.joint_dim
            )));
        }
        Ok(cond)
    }

    /// Render one image. Reuses the resident weights; nothing is re-loaded.
    ///
    /// # Errors
    /// [`PipelineError`] wrapping whichever stage failed.
    pub fn render(&self, params: &RenderParams) -> Result<RenderOutcome, PipelineError> {
        let t_total = Instant::now();

        let (lat_h, lat_w) = sample::latent_grid(params.height, params.width);
        let seq_img = lat_h * lat_w;

        // ── 1. Conditioning ──
        let t_te = Instant::now();
        let cond_vec = self.encode(&params.prompt)?;
        let te_encode = t_te.elapsed();

        // ── 2. DiT sampling ──
        let t_dit = Instant::now();
        let fwd = DitForward::new(&self.dit_weights);
        let init = sample::create_noise(params.seed, params.height, params.width);
        if init.len() != seq_img * self.in_channels {
            return Err(PipelineError::Shape(format!(
                "native noise len {} != {seq_img}*{} (in_channels vs PACKED_CHANNELS {})",
                init.len(),
                self.in_channels,
                sample::PACKED_CHANNELS
            )));
        }
        let img_ids = sample::img_ids(lat_h, lat_w);
        let txt_ids = sample::txt_ids(SEQ_TXT);
        let (timesteps, sigmas) = sample::flow_match_schedule(seq_img, params.steps);
        let latent = fwd.sample(
            &init, &cond_vec, &img_ids, &txt_ids, seq_img, SEQ_TXT, &timesteps, &sigmas, None,
        )?;
        let dit_sample = t_dit.elapsed();

        // ── 3. VAE decode ──
        let t_vae = Instant::now();
        let packed = latent_seq_to_packed_nchw(&latent, seq_img, self.in_channels, lat_h, lat_w)?;
        let decoded = self
            .vae
            .decode_packed_latents(&packed, lat_h, lat_w, None)?;
        let vae_decode = t_vae.elapsed();

        // ── 4. Pixels + PNG ──
        let t_png = Instant::now();
        let (w, h, rgb) = decoded_chw_to_rgb8(decoded.c, decoded.h, decoded.w, &decoded.data)?;
        let png = encode_rgb8(w, h, &rgb)?;
        let png_encode = t_png.elapsed();

        Ok(RenderOutcome {
            image: TextToImageOut {
                png,
                width: w,
                height: h,
                stage_cosines: Vec::new(),
            },
            timings: StageTimings {
                te_encode,
                dit_sample,
                vae_decode,
                png_encode,
                total: t_total.elapsed(),
            },
        })
    }
}
