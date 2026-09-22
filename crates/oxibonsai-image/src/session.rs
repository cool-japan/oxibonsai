//! Resident image-generation session.
//!
//! [`ImageSession`] loads the DiT, VAE, text encoder, and tokenizer **once** and
//! holds them in memory, so a long-running front-end (e.g. the `oxibonsai repl`
//! command) can render many prompts without re-paying the per-call load and
//! 4-bit dequant cost.
//!
//! ## Text-encoder memory policy (RAG-EVAL-IMG-25)
//!
//! [`ImageSession::load`]'s text-encoder residency default is **source-aware**
//! rather than a single unconditional flag:
//!
//! | `te_source`             | on-disk size | `OXI_TE_RESIDENT` unset | `=0`  | `=1` |
//! |--------------------------|-------------|-------------------------|-------|------|
//! | [`TeSource::NpyDir`]     | ~15 GB (f32) | resident (default ON)   | transient | resident |
//! | [`TeSource::Mlx4bit`]    | ~2.1 GB (4-bit) | transient (default OFF) | transient | resident |
//!
//! `NpyDir` is *already* f32 on disk and its own [`crate::te::TeWeights::get`]
//! caches every tensor unconditionally regardless of this flag (see that
//! method's doc) — so declaring it "resident" costs nothing extra and simply
//! matches what already happens, and the REPL front-end wants the amortization
//! by default. `Mlx4bit` is the one case where turning residency on actually
//! *materialises* a new ~16 GB fully-dequantised f32 cache (its native ~2.1 GB
//! 4-bit footprint is the whole point of that source), so it opts IN rather
//! than out — a 16 GB pin must never be the silent default on a
//! memory-constrained machine. Either source's default is overridden the same
//! two ways, and an explicit override always wins over the source default:
//! env `OXI_TE_RESIDENT=0`/`=1` at [`ImageSession::load`] time, or the
//! [`ImageSession::with_te_resident`] builder afterward.
//!
//! [`ImageSession::warm`] is gated on the resulting residency for
//! [`TeSource::Mlx4bit`] (warming a non-resident encoder would just waste a
//! throwaway forward with no caching benefit); a [`TeSource::NpyDir`]
//! session always benefits from warming, since that source caches
//! unconditionally regardless of the residency flag — see
//! [`crate::te::TeWeights::caches_tensors`]. A session intended to render
//! exactly once can additionally call [`ImageSession::one_shot`] so
//! [`ImageSession::render`] frees the text-encoder cache immediately after
//! computing the conditioning vector (before the DiT/VAE stages), rather
//! than leaving it cached for a render that will never come.
//!
//! The per-render math is identical to the native (non-golden) path of
//! [`crate::pipeline::text_to_image`] — both share
//! `decoded_chw_to_rgb8` and the same `sample`/`forward`
//! stages, so a session render and a one-shot render of the same
//! `(prompt, seed, steps, size)` produce byte-identical PNGs.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::pipeline::{
    decoded_chw_to_rgb8, latent_seq_to_packed_nchw, validate_render_params, PipelineError,
    TeSource, TextToImageOut,
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
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            seed: 42,
            steps: 4,
            width: 512,
            height: 512,
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

/// Which compute backend actually executed this process's GPU-eligible ops
/// (DiT ternary matmuls / joint attention, the text encoder's matmuls, and the
/// VAE's conv/norm/silu/upsample) — see [`ImageSession::effective_backend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// No GPU-eligible op has succeeded in this process (no GPU backend
    /// compiled in, disabled via an `OXI_*_GPU=0` env toggle, or every attempt
    /// failed and fell back — see `crate::gpu`'s one-time warning for the
    /// reason in that last case). Plain backticks, not an intra-doc link:
    /// `crate::gpu` is `cfg`-gated to `metal` + macOS, so a default-feature
    /// doc build (this crate's rustdoc gate) has no such item to resolve to.
    Cpu,
    /// At least one GPU-eligible op has succeeded on the Metal or CUDA path.
    Gpu,
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
    /// When `true`, [`Self::render`] frees the text-encoder's cached weights
    /// immediately after computing the conditioning vector (before the
    /// DiT/VAE stages), instead of leaving them cached for a next render. Set
    /// via [`Self::one_shot`]. Off by default (the ordinary "render many
    /// prompts" session use case).
    free_te_after_render: bool,
}

/// Which [`TeSource`] variant a residency default is being computed for,
/// without carrying its `PathBuf` — lets [`te_resident_from_env_value`] be
/// exercised directly in tests with no path/disk involved. See the
/// [module docs](self)'s text-encoder memory policy table for why the two
/// sources default differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TeSourceKind {
    /// The native ~2.1 GB 4-bit MLX safetensors. Residency means caching the
    /// fully dequantised f32 encoder (~16 GB) — expensive enough that it must
    /// be opted into, not assumed.
    Mlx4bit,
    /// The ~15 GB f32 `.npy` dump. [`crate::te::TeWeights::get`] caches this
    /// source unconditionally regardless of the residency flag, so declaring
    /// it resident by default costs nothing extra and matches what already
    /// happens.
    NpyDir,
}

impl TeSourceKind {
    fn of(source: &TeSource) -> Self {
        match source {
            TeSource::Mlx4bit(_) => Self::Mlx4bit,
            TeSource::NpyDir(_) => Self::NpyDir,
        }
    }

    /// This source's residency default when `OXI_TE_RESIDENT` is unset.
    fn resident_by_default(self) -> bool {
        matches!(self, Self::NpyDir)
    }
}

/// Pure decision function behind [`te_resident_default`]: given the raw
/// `OXI_TE_RESIDENT` env value and which TE source is being loaded, should the
/// session start resident? Split out so both the production env-reading path
/// and its tests exercise identical logic without any test mutating the real
/// process environment via `set_var`/`remove_var` — see the
/// [module docs](self)'s T-Missed-1 reference for why that class of shared
/// mutable process state is worth avoiding under this crate's one-process
/// `cargo test` run.
///
/// `OXI_TE_RESIDENT=0` always means transient and any other *explicit* value
/// (`"1"`, `"yes"`, ...) always means resident, for **either** source — an
/// explicit env value overrides the source-specific default. Only when the
/// env var is unset does [`TeSourceKind::resident_by_default`] decide: see
/// the [module docs](self)'s table.
fn te_resident_from_env_value(v: Option<&str>, source: TeSourceKind) -> bool {
    match v {
        Some("0") => false,
        Some(_) => true,
        None => source.resident_by_default(),
    }
}

/// Default text-encoder residency policy for [`ImageSession::load`]:
/// source-aware (see the [module docs](self)'s table) unless overridden by env
/// `OXI_TE_RESIDENT=0`/`=1`, which wins over either source's default.
/// [`ImageSession::with_te_resident`] is the equivalent programmatic override.
fn te_resident_default(source: &TeSource) -> bool {
    te_resident_from_env_value(
        std::env::var("OXI_TE_RESIDENT").ok().as_deref(),
        TeSourceKind::of(source),
    )
}

impl ImageSession {
    /// Load every model asset.
    ///
    /// `te_source` selects the 4-bit MLX safetensors or the f32 `.npy` dir, same
    /// as [`crate::pipeline::TextToImageCfg`]. The text encoder's residency
    /// (whether its dequantised weights are cached across renders) defaults
    /// **on** for [`TeSource::NpyDir`] and **off** for [`TeSource::Mlx4bit`] —
    /// see the [module docs](self)'s table — unless env `OXI_TE_RESIDENT=0`
    /// (force off) or `OXI_TE_RESIDENT=1` (force on) is set; use
    /// [`Self::with_te_resident`] to override programmatically instead.
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
        // RAG-EVAL-IMG-25: source-aware default (see module docs' table) —
        // resident for `NpyDir` (free — that source ignores this flag
        // internally and stays inherently resident once read regardless),
        // transient for `Mlx4bit` (residency is what would materialise the
        // ~16 GB dequantised cache) — with `OXI_TE_RESIDENT=0`/`=1` or
        // `with_te_resident` overriding either source's default.
        te_weights.set_resident(te_resident_default(te_source));

        let tokenizer = Qwen3Tokenizer::open(tokenizer_dir)?;

        Ok(Self {
            dit_weights,
            vae,
            te_weights,
            tokenizer,
            in_channels,
            joint_dim,
            free_te_after_render: false,
        })
    }

    /// Override (in either direction) keeping the text-encoder's dequantised
    /// weights resident across [`Self::render`] calls. The default depends on
    /// the source (see the [module docs](self)'s table: on for
    /// [`TeSource::NpyDir`], off for [`TeSource::Mlx4bit`]); pass `true` on a
    /// high-memory machine to amortize a [`TeSource::Mlx4bit`] session's
    /// dequant cost across renders (materialising its ~16 GB f32 cache), or
    /// `false` to trade that RAM back for the source's original per-read
    /// dequant cost on a memory-constrained one.
    ///
    /// Consumes and returns `self` for use right after [`Self::load`]:
    /// ```no_run
    /// # use std::path::Path;
    /// # use oxibonsai_image::pipeline::TeSource;
    /// # use oxibonsai_image::ImageSession;
    /// # fn doc() -> Result<(), Box<dyn std::error::Error>> {
    /// let session = ImageSession::load(
    ///     Path::new("dit.gguf"),
    ///     Path::new("vae.safetensors"),
    ///     &TeSource::Mlx4bit(Path::new("te.safetensors").to_path_buf()),
    ///     Path::new("tokenizer_dir"),
    /// )?
    /// // Mlx4bit defaults to transient; opt this high-memory-machine session
    /// // into residency explicitly (equivalent to env `OXI_TE_RESIDENT=1`).
    /// .with_te_resident(true);
    /// # let _ = session;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// On a [`TeSource::NpyDir`] (f32) session this has no *lasting* effect
    /// either way: [`crate::te::TeWeights::get`] never consults this flag for
    /// that source and unconditionally re-caches on its next read, so the
    /// source is resident again as soon as anything reads through it.
    /// `with_te_resident(true)` on that source is then a pure no-op; calling
    /// `with_te_resident(false)` *does* still do one visible thing — it clears
    /// whatever is already cached right now (so the very next read pays a
    /// disk read again) — before that self-healing kicks back in. Call this
    /// right after [`Self::load`], before any render, to avoid relying on
    /// that transient-clear behaviour.
    #[must_use]
    pub fn with_te_resident(self, resident: bool) -> Self {
        self.te_weights.set_resident(resident);
        self
    }

    /// Mark this session as one-shot: [`Self::render`] frees the
    /// text-encoder's cached weights immediately after computing the
    /// conditioning vector (before the DiT/VAE stages run), instead of
    /// leaving them cached for a render that will never come. Use for a
    /// session built to render exactly once and then be dropped.
    ///
    /// On a [`TeSource::NpyDir`] (f32) session this still clears whatever is
    /// currently cached (see [`Self::with_te_resident`]'s doc for why that is
    /// only a *transient* effect for that source) — harmless for a session
    /// that, as the name promises, renders only once more, but calling
    /// [`Self::render`] again afterward on the *same* one-shot NpyDir session
    /// pays a fresh disk read for every text-encoder tensor.
    #[must_use]
    pub fn one_shot(mut self, one_shot: bool) -> Self {
        self.free_te_after_render = one_shot;
        self
    }

    /// Whether the text encoder's dequantised weights are currently held
    /// resident (see the [module docs](self)'s text-encoder memory policy
    /// section, and [`crate::te::TeWeights::is_resident`]).
    pub fn is_te_resident(&self) -> bool {
        self.te_weights.is_resident()
    }

    /// Report which backend actually executed this process's GPU-eligible ops
    /// at least once so far.
    ///
    /// # Scope caveat
    /// This is **process-global and monotonic**, not scoped to `self` or to
    /// the most recent [`Self::render`]: it reads the crate's `*_was_used()`
    /// "did the GPU ever succeed" flags (`crate::gpu::gpu_was_used` — plain
    /// backticks, not a link: that item is `cfg`-gated to `metal` + macOS, so
    /// a default-feature doc build has nothing to resolve it to — and its
    /// VAE/TE/CUDA siblings), which are set once and never cleared for the
    /// life of the process. Once any GPU op has succeeded anywhere —
    /// including in a different [`ImageSession`] or a plain
    /// [`crate::pipeline::text_to_image`] call earlier in the same process —
    /// this reports [`Backend::Gpu`] even if the *next* render happens to
    /// fall back to CPU. It answers the operability question RAG-EVAL-IMG-18
    /// asks for ("has the GPU path been engaged in this process at all"), not
    /// a per-render guarantee. A build with no GPU backend compiled in (no
    /// `metal`/`native-cuda` feature, or the wrong `target_os`) always reports
    /// [`Backend::Cpu`].
    pub fn effective_backend(&self) -> Backend {
        effective_backend_from_process_state()
    }

    /// Populate the resident text-encoder cache so the *first* real render runs
    /// at warm speed. A single short encode touches every layer's weights, so
    /// one pass dequantises the whole encoder. Returns how long warming took.
    ///
    /// DiT and VAE weights are memory-mapped and page in lazily; this only warms
    /// the encoder, which is the dominant amortizable (re-dequant) cost. A
    /// no-op (returns `Duration::ZERO` without encoding anything) only when
    /// the text encoder's source would not keep the warm-up's work anyway —
    /// a non-resident [`TeSource::Mlx4bit`] session (see
    /// [`crate::te::TeWeights::caches_tensors`] and the [module docs](self)):
    /// with nothing kept, a throwaway encode would only cost time with no
    /// benefit to the next real render. A [`TeSource::NpyDir`] session always
    /// warms for real, because that source caches unconditionally regardless
    /// of [`Self::is_te_resident`]. A caller that prints this duration to the
    /// user (e.g. "Warming text encoder… `{d}`s") should check
    /// [`Self::is_te_resident`] first (for a `Mlx4bit` session) and skip the
    /// message rather than report a misleadingly-fast zero.
    ///
    /// # Errors
    /// [`PipelineError`] if the warm-up encode fails.
    pub fn warm(&self) -> Result<Duration, PipelineError> {
        if !self.te_weights.caches_tensors() {
            return Ok(Duration::ZERO);
        }
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
    /// `params.width`/`params.height` must be multiples of 16 within
    /// `[MIN_DIMENSION, MAX_DIMENSION]` (see [`crate::pipeline`])
    /// and `params.steps >= 1` (mirrors [`crate::pipeline::text_to_image`]'s
    /// validation — the REPL that builds `RenderParams` bypasses
    /// [`crate::pipeline::TextToImageCfg`] entirely, so this session applies
    /// the same check itself rather than relying on the caller).
    ///
    /// # Errors
    /// [`PipelineError`] wrapping whichever stage failed, including
    /// [`PipelineError::InvalidParams`] from the geometry/step validation.
    pub fn render(&self, params: &RenderParams) -> Result<RenderOutcome, PipelineError> {
        validate_render_params(params.width, params.height, params.steps)?;

        let t_total = Instant::now();

        let (lat_h, lat_w) = sample::latent_grid(params.height, params.width);
        let seq_img = lat_h * lat_w;

        // ── 1. Conditioning ──
        let t_te = Instant::now();
        let cond_vec = self.encode(&params.prompt)?;
        let te_encode = t_te.elapsed();
        if self.free_te_after_render {
            // One-shot session: release the text-encoder cache now, before the
            // DiT/VAE stages run, rather than waiting for `self` to drop.
            self.te_weights.set_resident(false);
        }

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
        // Auto-tile when the output exceeds 512 px (same threshold as pipeline.rs).
        let decoded = if 16 * lat_h > 512 {
            self.vae.decode_packed_latents_tiled(
                &packed,
                lat_h,
                lat_w,
                crate::vae::tiling::TileConfig::default(),
            )?
        } else {
            self.vae
                .decode_packed_latents(&packed, lat_h, lat_w, None)?
        };
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

/// The `*_was_used()` latch computation behind
/// [`ImageSession::effective_backend`], factored out so it can be exercised
/// directly in a unit test without an asset-backed [`ImageSession`] (its
/// source of truth is process-global, not session state — see that method's
/// doc for the scope caveat).
fn effective_backend_from_process_state() -> Backend {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if crate::gpu::gpu_was_used()
            || crate::gpu::dit_attn_gpu_was_used()
            || crate::te::gpu::te_gpu_was_used()
            || crate::vae::gpu::vae_gpu_was_used()
        {
            return Backend::Gpu;
        }
    }
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        if crate::cuda_gpu::gpu_was_used()
            || crate::cuda_gpu::dit_attn_gpu_was_used()
            || crate::te::cuda_gpu::te_gpu_was_used()
            || crate::vae::cuda_gpu::vae_gpu_was_used()
        {
            return Backend::Gpu;
        }
    }
    Backend::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_variants_are_distinct() {
        assert_ne!(Backend::Cpu, Backend::Gpu);
        assert_eq!(Backend::Cpu, Backend::Cpu);
        assert_eq!(Backend::Gpu, Backend::Gpu);
    }

    #[test]
    fn effective_backend_from_process_state_matches_the_was_used_latches() {
        // Not a hardcoded `assert_eq!(.., Backend::Cpu)`: this crate's tests
        // run in one process (`cargo test`, not nextest), and at least one
        // sibling test (`vae::tiling`'s
        // `tiled_conv_out_matches_untiled_under_default_dispatch`, which does
        // not force the CPU conv path) can genuinely flip a `*_was_used()`
        // latch to `true` on real Metal hardware depending on test execution
        // order — the exact latch-poisoning class T-Missed-1 diagnosed
        // elsewhere in this crate. So this recomputes the same process-global
        // latches independently (structurally mirroring, but not calling,
        // the function under test) and asserts the mapping, which holds
        // regardless of what ran before it in this process.
        fn any_gpu_used() -> bool {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                if crate::gpu::gpu_was_used()
                    || crate::gpu::dit_attn_gpu_was_used()
                    || crate::te::gpu::te_gpu_was_used()
                    || crate::vae::gpu::vae_gpu_was_used()
                {
                    return true;
                }
            }
            #[cfg(all(
                feature = "native-cuda",
                any(target_os = "linux", target_os = "windows")
            ))]
            {
                if crate::cuda_gpu::gpu_was_used()
                    || crate::cuda_gpu::dit_attn_gpu_was_used()
                    || crate::te::cuda_gpu::te_gpu_was_used()
                    || crate::vae::cuda_gpu::vae_gpu_was_used()
                {
                    return true;
                }
            }
            false
        }
        let expected = if any_gpu_used() {
            Backend::Gpu
        } else {
            Backend::Cpu
        };
        assert_eq!(effective_backend_from_process_state(), expected);
    }

    // ── RAG-EVAL-IMG-25: source-aware residency default, all four (source x
    //    env) combinations, entirely through the pure helper so no test ever
    //    mutates the real process environment (`set_var`/`remove_var`) — see
    //    `te_resident_from_env_value`'s doc and the module docs' T-Missed-1
    //    reference for why that class of shared mutable process state is
    //    worth avoiding under this crate's one-process `cargo test` run. ──

    #[test]
    fn te_resident_from_env_value_npydir_defaults_resident_when_unset() {
        assert!(
            te_resident_from_env_value(None, TeSourceKind::NpyDir),
            "NpyDir must default resident: that source already caches every \
             tensor unconditionally, so declaring it resident costs nothing \
             extra and the REPL wants the amortization"
        );
    }

    #[test]
    fn te_resident_from_env_value_mlx4bit_defaults_transient_when_unset() {
        assert!(
            !te_resident_from_env_value(None, TeSourceKind::Mlx4bit),
            "Mlx4bit must default transient: turning residency on for this \
             source is what materialises the ~16 GB dequantised cache, so it \
             must be an opt-in, not a silent default"
        );
    }

    #[test]
    fn te_resident_from_env_value_explicit_zero_overrides_both_sources() {
        for source in [TeSourceKind::NpyDir, TeSourceKind::Mlx4bit] {
            assert!(
                !te_resident_from_env_value(Some("0"), source),
                "OXI_TE_RESIDENT=0 must force transient for {source:?} even \
                 though NpyDir's own default is resident"
            );
        }
    }

    #[test]
    fn te_resident_from_env_value_explicit_non_zero_overrides_both_sources() {
        for source in [TeSourceKind::NpyDir, TeSourceKind::Mlx4bit] {
            for v in ["1", "yes"] {
                assert!(
                    te_resident_from_env_value(Some(v), source),
                    "OXI_TE_RESIDENT={v} must force resident for {source:?} \
                     even though Mlx4bit's own default is transient"
                );
            }
        }
    }

    #[test]
    fn te_source_kind_of_matches_the_te_source_variant() {
        assert_eq!(
            TeSourceKind::of(&TeSource::Mlx4bit(std::path::PathBuf::new())),
            TeSourceKind::Mlx4bit
        );
        assert_eq!(
            TeSourceKind::of(&TeSource::NpyDir(std::path::PathBuf::new())),
            TeSourceKind::NpyDir
        );
    }

    #[test]
    fn te_resident_default_matches_pure_helper_for_current_env_both_sources() {
        // Guards the env-reading wiring itself (not just the decision logic
        // above) without ever mutating the process environment — pure reads
        // only, so this is safe regardless of what other tests in this
        // process do to `OXI_TE_RESIDENT` or the order they run in.
        let raw = std::env::var("OXI_TE_RESIDENT").ok();
        let mlx4bit = TeSource::Mlx4bit(std::path::PathBuf::new());
        let npydir = TeSource::NpyDir(std::path::PathBuf::new());
        assert_eq!(
            te_resident_default(&mlx4bit),
            te_resident_from_env_value(raw.as_deref(), TeSourceKind::Mlx4bit)
        );
        assert_eq!(
            te_resident_default(&npydir),
            te_resident_from_env_value(raw.as_deref(), TeSourceKind::NpyDir)
        );
    }
}
