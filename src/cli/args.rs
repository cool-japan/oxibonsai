//! `clap` CLI surface: the `oxibonsai` argument parser and subcommand table.
//!
//! Sampling-related numeric flags (`--temperature`, `--top-p`,
//! `--repetition-penalty`, `--max-tokens`, ...) are `Option<T>` with **no**
//! `default_value` baked into `clap`: this is what lets `--config`
//! (cli-04) apply a `[sampling]`/`[model]`/`[server]` TOML value only when
//! the corresponding flag was genuinely absent from argv, without widening
//! or narrowing a value the user actually typed. The real, hardcoded
//! fallback (documented in each flag's help text) is applied once, in
//! `mod.rs`'s config-merge step, after the TOML layer.

use super::tokenizer_backend::TokenizerBackendChoice;
use clap::{Parser, Subcommand};

// Each numeric flag below is a `parse_x` (used as clap's `value_parser`,
// `&str -> Result<T, String>`) built on top of a `validate_x` (`T ->
// Result<T, String>`). The split matters beyond `clap`: a value from
// `[sampling]`/`[model]` in `--config` (cli-04) is parsed by `mod.rs`'s own
// `toml_f32`/`toml_usize` — plain `str::parse`, with none of clap's
// `value_parser` range checks — so without also running `validate_x` on
// the *resolved* value there, a config file could set e.g.
// `temperature = -5.0` and bypass cli-12's validation entirely through a
// different door than the one it was closed on. `mod.rs` calls the same
// `validate_x` functions after resolving each flag from CLI-or-config, so
// every source is checked identically.

pub(crate) fn validate_temperature(v: f32) -> Result<f32, String> {
    if !v.is_finite() || v < 0.0 {
        return Err(format!("temperature must be >= 0.0 and finite, got {v}"));
    }
    Ok(v)
}

pub(crate) fn parse_temperature(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    validate_temperature(v)
}

pub(crate) fn validate_top_p(v: f32) -> Result<f32, String> {
    if !(v > 0.0 && v <= 1.0) {
        return Err(format!("top-p must be in the range (0.0, 1.0], got {v}"));
    }
    Ok(v)
}

pub(crate) fn parse_top_p(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    validate_top_p(v)
}

pub(crate) fn validate_repetition_penalty(v: f32) -> Result<f32, String> {
    if !(v.is_finite() && v > 0.0) {
        return Err(format!("repetition-penalty must be > 0.0, got {v}"));
    }
    Ok(v)
}

pub(crate) fn parse_repetition_penalty(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    validate_repetition_penalty(v)
}

pub(crate) fn validate_openai_penalty(v: f32) -> Result<f32, String> {
    if !(v.is_finite() && (-2.0..=2.0).contains(&v)) {
        return Err(format!(
            "penalty must be in the range [-2.0, 2.0] (OpenAI convention), got {v}"
        ));
    }
    Ok(v)
}

pub(crate) fn parse_openai_penalty(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    validate_openai_penalty(v)
}

pub(crate) fn validate_max_tokens(v: usize) -> Result<usize, String> {
    if v < 1 {
        return Err("max-tokens must be >= 1".to_string());
    }
    Ok(v)
}

pub(crate) fn parse_max_tokens(s: &str) -> Result<usize, String> {
    let v: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid non-negative integer"))?;
    validate_max_tokens(v)
}

pub(crate) fn validate_max_seq_len(v: usize) -> Result<usize, String> {
    if v < 1 {
        return Err("max-seq-len must be >= 1".to_string());
    }
    Ok(v)
}

pub(crate) fn parse_max_seq_len(s: &str) -> Result<usize, String> {
    let v: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid non-negative integer"))?;
    validate_max_seq_len(v)
}

/// Min-p (probabilistic nucleus) threshold (RT-23 CLI surface). Rejects a
/// non-finite value or one outside `[0.0, 1.0]` up front — the same
/// fail-fast convention as [`validate_top_p`] — rather than silently
/// accepting an out-of-range value that would simply never filter anything
/// at sample time.
pub(crate) fn validate_min_p(v: f32) -> Result<f32, String> {
    if !(0.0..=1.0).contains(&v) || !v.is_finite() {
        return Err(format!("min-p must be in the range [0.0, 1.0], got {v}"));
    }
    Ok(v)
}

pub(crate) fn parse_min_p(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid number"))?;
    validate_min_p(v)
}

/// `--backend auto|cpu|metal`. Parses
/// through [`oxibonsai_runtime::engine_seam::Backend::parse`] so this CLI's
/// accepted spellings can never drift from the engine's own, and the
/// resolved value is the real enum clap stores directly — no intermediate
/// `String` round trip through `args.rs`.
pub(crate) fn parse_backend(s: &str) -> Result<oxibonsai_runtime::engine_seam::Backend, String> {
    oxibonsai_runtime::engine_seam::Backend::parse(s)
        .ok_or_else(|| format!("invalid --backend value '{s}': expected auto, cpu, or metal"))
}

/// `--rope-scaling auto|on|off` (`auto` applies YaRN only when the GGUF's
/// own RoPE-scaling metadata calls for it, matching Bonsai-8B's shipped
/// config). A thin wrapper over [`oxibonsai_runtime::config::RopeScalingMode`]'s own
/// `FromStr` (that type has no `clap::ValueEnum` derive — adding one would
/// need a new `clap` dependency on `oxibonsai-runtime` — so a manual
/// `value_parser` function, not `#[arg(value_enum)]`, is how this flag
/// reaches it).
pub(crate) fn parse_rope_scaling(
    s: &str,
) -> Result<oxibonsai_runtime::config::RopeScalingMode, String> {
    s.parse()
}

/// `--reasoning-effort low|medium|xhigh` (cli-11 / Bonsai 2 chat contract,
/// design §5.7). A plain string (not a `clap::ValueEnum`) because the
/// resolved value must also be layerable through `--config`'s
/// `[sampling]`/`[model]` sections the same way every other cli-04 flag is
/// (`mod.rs`'s `util::resolve_str` + re-validation), which needs a
/// `parse_x`/`validate_x` pair operating on `&str`/`String` like this
/// file's other resolved flags, not a `clap`-only enum type.
pub(crate) fn validate_reasoning_effort(v: &str) -> Result<String, String> {
    match v {
        "low" | "medium" | "xhigh" => Ok(v.to_string()),
        other => Err(format!(
            "invalid --reasoning-effort value '{other}': expected one of low, medium, xhigh"
        )),
    }
}

pub(crate) fn parse_reasoning_effort(s: &str) -> Result<String, String> {
    validate_reasoning_effort(s)
}

/// `--prefill-chunk <N>` (design §5.7): the prompt-ingestion chunk size.
/// Must be at least 1; the per-model default (512 for a `qwen35` hybrid's
/// Gated-DeltaNet prefill, the dense stack's own chunk plan otherwise)
/// applies when the flag is absent.
pub(crate) fn validate_prefill_chunk(v: usize) -> Result<usize, String> {
    if v < 1 {
        return Err("prefill-chunk must be >= 1".to_string());
    }
    Ok(v)
}

pub(crate) fn parse_prefill_chunk(s: &str) -> Result<usize, String> {
    let v: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid non-negative integer"))?;
    validate_prefill_chunk(v)
}

/// `--image-max-tokens <N>` (design §5.7, vision phase 2): the per-image
/// token budget the downscale guard enforces. Must be at least 1.
pub(crate) fn validate_image_max_tokens(v: usize) -> Result<usize, String> {
    if v < 1 {
        return Err("image-max-tokens must be >= 1".to_string());
    }
    Ok(v)
}

pub(crate) fn parse_image_max_tokens(s: &str) -> Result<usize, String> {
    let v: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid non-negative integer"))?;
    validate_image_max_tokens(v)
}

/// `serve --max-output-tokens <N>`: a hard ceiling on a request's effective
/// `max_tokens`. `0` is rejected (it would 400 every request instead of
/// capping it), matching `oxibonsai-serve`'s own parser.
#[cfg(feature = "server")]
pub(crate) fn validate_max_output_tokens(v: usize) -> Result<usize, String> {
    if v < 1 {
        return Err(
            "max-output-tokens must be at least 1 (a zero ceiling would reject every request \
             outright instead of capping it)"
                .to_string(),
        );
    }
    Ok(v)
}

#[cfg(feature = "server")]
pub(crate) fn parse_max_output_tokens(s: &str) -> Result<usize, String> {
    let v: usize = s
        .parse()
        .map_err(|_| format!("'{s}' is not a valid non-negative integer"))?;
    validate_max_output_tokens(v)
}

/// Which backend `serve` answers `/v1/embeddings` from
/// (`--embedding-backend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum EmbeddingBackendChoice {
    /// Mean-pooled hidden states of the loaded model (dense models; a
    /// hybrid `qwen35` model has no embedder yet and answers the honest 501).
    #[default]
    Model,
    /// Serve no embeddings at all: `/v1/embeddings` always answers 501.
    None,
    /// A TF-IDF (lexical, non-semantic) embedder whose vocabulary and IDF
    /// weights are fitted once, at startup, on `--embedding-corpus` — never
    /// on client requests, so the vector space is fixed for the server's
    /// lifetime.
    Tfidf,
}

#[derive(Parser)]
#[command(
    name = "oxibonsai",
    version,
    about = "Pure Rust sub-2-bit LLM inference engine for PrismML Bonsai models"
)]
pub(crate) struct Cli {
    /// Path to an OxiBonsai TOML configuration file. Every section
    /// (`[model]`, `[sampling]`, `[server]`, `[observability]`) is applied;
    /// a value here is used only where the matching CLI flag was not
    /// passed. An unreadable or unparsable file is a hard error naming the
    /// path and the underlying parse error, and an unknown key anywhere in
    /// the file is also an error (a typo'd key is silently ignored
    /// otherwise).
    #[arg(long, global = true)]
    pub(crate) config: Option<String>,

    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Run inference on a GGUF model.
    Run {
        /// Path to the GGUF model file (default: env OXI_MODEL, or
        /// `[model].model_path` in --config).
        #[arg(short, long)]
        model: Option<String>,

        /// Prompt text. Use "-" for stdin.
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate (default: 256, or
        /// `[sampling].max_tokens` in --config). Must be >= 1.
        #[arg(long, value_parser = parse_max_tokens)]
        max_tokens: Option<usize>,

        /// Sampling temperature; 0.0 = greedy argmax on every backend.
        /// Precedence (RT-17): this flag, else `[sampling].temperature` in
        /// --config, else the model's own `general.sampling.temp` (Bonsai
        /// 2: 1.0), else 0.7. Must be >= 0.0.
        #[arg(long, value_parser = parse_temperature)]
        temperature: Option<f32>,

        /// Top-k sampling, 0 = disabled. Precedence: this flag, else
        /// `[sampling].top_k`, else the model's `general.sampling.top_k`
        /// (Bonsai 2: 20), else 40.
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p (nucleus) sampling. Precedence: this flag, else
        /// `[sampling].top_p`, else the model's `general.sampling.top_p`
        /// (Bonsai 2: 0.95), else 0.9. Must be in (0.0, 1.0].
        #[arg(long, value_parser = parse_top_p)]
        top_p: Option<f32>,

        /// Repetition penalty; 1.0 = disabled, applied on every backend
        /// (default: 1.0, or `[sampling].repetition_penalty` in
        /// --config). Must be > 0.0. No hidden non-1.0 value is ever
        /// applied (every default in this workspace is 1.0 now), so
        /// `--temperature 0` means exactly argmax unless a penalty is
        /// explicitly requested. A non-default value is a hard error when
        /// combined with --grammar or --stop (that decode loop cannot apply
        /// it).
        #[arg(long, value_parser = parse_repetition_penalty)]
        repetition_penalty: Option<f32>,

        /// OpenAI-style frequency penalty applied over the generated-token
        /// history (default: 0.0 = disabled). Range [-2.0, 2.0]. Applies
        /// on every backend (CPU, Metal, CUDA) EXCEPT when combined with
        /// --grammar or --stop, where it is a hard error rather than a
        /// silently-dropped flag (that decode loop cannot apply it).
        #[arg(long, value_parser = parse_openai_penalty, allow_negative_numbers = true)]
        frequency_penalty: Option<f32>,

        /// OpenAI-style presence penalty applied over the generated-token
        /// history (default: 0.0 = disabled). Range [-2.0, 2.0]. Applies
        /// on every backend (CPU, Metal, CUDA) EXCEPT when combined with
        /// --grammar or --stop, where it is a hard error rather than a
        /// silently-dropped flag (that decode loop cannot apply it).
        #[arg(long, value_parser = parse_openai_penalty, allow_negative_numbers = true)]
        presence_penalty: Option<f32>,

        /// Random seed. Two runs with the same seed, model, prompt and
        /// sampling flags produce byte-identical output (RT-12).
        /// Precedence: this flag, else `[sampling].seed` in --config, else
        /// 42.
        #[arg(long)]
        seed: Option<u64>,

        /// Maximum sequence length (prompt + generated); `--ctx` is an
        /// alias (default: 8192 for a Bonsai 2 `qwen35` model, 4096
        /// otherwise, or `[model].max_seq_len` in --config). Must be >= 1,
        /// and for a `qwen35` model it is refused above both the model's
        /// declared context and the RAM-derived bound.
        #[arg(long, visible_alias = "ctx", value_parser = parse_max_seq_len)]
        max_seq_len: Option<usize>,

        /// Path to tokenizer.json file (default: auto-detected, or
        /// `[model].tokenizer_path` in --config).
        #[arg(long)]
        tokenizer: Option<String>,

        /// Which tokenizer backend to use.
        #[arg(long, value_enum, default_value_t = TokenizerBackendChoice::Auto)]
        tokenizer_backend: TokenizerBackendChoice,

        /// Render `--prompt` as a single user turn through the model's own
        /// chat template (the GGUF's `tokenizer.chat_template`, or the
        /// built-in ChatML/Qwen3 fallback when the file ships none) instead
        /// of feeding it to the model raw. Required by --think/--no-think,
        /// --reasoning-effort, --tools and --show-reasoning/--hide-reasoning,
        /// which only mean something inside the chat contract.
        #[arg(long, default_value_t = false)]
        chat: bool,

        /// Constrain generation to a grammar: a `.gbnf` file, or any other
        /// extension is parsed as a JSON Schema. Applies on every backend
        /// (a dedicated token-by-token decode loop, not the streaming
        /// fast paths).
        #[arg(long)]
        grammar: Option<String>,

        /// Stop generation as soon as any of these strings appears in the
        /// decoded output (may be passed multiple times).
        #[arg(long)]
        stop: Vec<String>,

        /// Min-p (probabilistic nucleus) sampling; 0.0 = disabled. Applied
        /// after top-k, before top-p (RT-23). Precedence: this flag, else
        /// `[sampling].min_p`, else the model's `general.sampling.min_p`,
        /// else 0.0. Only affects sampled decoding (`--temperature` > 0).
        #[arg(long, value_parser = parse_min_p)]
        min_p: Option<f32>,

        /// Which compute backend runs the model: `auto` (best available —
        /// GPU when accelerated, CPU for a hybrid `qwen35` model such as
        /// Bonsai 2, since no hybrid GPU encoder exists yet), `cpu` (best
        /// CPU SIMD tier, regardless of GPU availability — including the
        /// temperature-0 path, which never takes a GPU route under `cpu`),
        /// or `metal` (the Metal GPU, or a typed, non-zero-exit error —
        /// always, never a silent CPU fallback — when unavailable on this
        /// build/host or when the model is a hybrid).
        #[arg(long, value_parser = parse_backend)]
        backend: Option<oxibonsai_runtime::engine_seam::Backend>,

        /// RoPE scaling strategy: `auto` (default) honours whatever
        /// `<arch>.rope.scaling.*` the GGUF declares, exactly like
        /// llama.cpp; `off` forces plain RoPE even when the file declares
        /// scaling (reproduces OxiBonsai <= 0.2.4 behaviour on models such
        /// as Bonsai-8B, which declares YaRN factor 4); `on` requires the
        /// file to declare scaling and errors when it does not.
        #[arg(long, value_parser = parse_rope_scaling)]
        rope_scaling: Option<oxibonsai_runtime::config::RopeScalingMode>,

        /// Enable the chat contract's `<think>` reasoning block
        /// (`enable_thinking = true` in the chat template). Neither flag
        /// passed = the template's own default (the Bonsai 2 template
        /// thinks by default). Requires --chat.
        #[arg(long, conflicts_with = "no_think")]
        think: bool,

        /// Disable the `<think>` reasoning block (`enable_thinking =
        /// false`). Requires --chat.
        #[arg(long)]
        no_think: bool,

        /// Reasoning effort passed to the chat template's
        /// `reasoning_effort` variable (Bonsai 2 chat contract): low,
        /// medium or xhigh. Requires --chat.
        #[arg(long, value_parser = parse_reasoning_effort)]
        reasoning_effort: Option<String>,

        /// Path to a JSON file containing an OpenAI-style `tools` array,
        /// passed to the chat template verbatim (raw JSON text, so
        /// key order and number formatting are preserved byte-for-byte
        /// rather than round-tripped through a Rust value — the
        /// tool-call contract requires this for byte-identical template
        /// output). Requires --chat.
        #[arg(long)]
        tools: Option<String>,

        /// Print the model's `<think>` reasoning to stderr while the answer
        /// streams to stdout (the default in --chat mode). Requires --chat.
        #[arg(long, conflicts_with = "hide_reasoning")]
        show_reasoning: bool,

        /// Drop the model's `<think>` reasoning and print only the answer.
        /// Requires --chat.
        #[arg(long)]
        hide_reasoning: bool,

        /// Transcode every PTQ1_0 (1.75-bit) weight matrix to the lossless
        /// 2-bit PQ2_0 layout at load time (design §2.1) instead of running
        /// the native PTQ1_0 kernels. Materialises the transcoded weights
        /// in anonymous RAM (~7.2 GB for the 27B) instead of mmapping them;
        /// the native path is the default. A no-op (with a log line) for a
        /// file with no PTQ1_0 tensors.
        #[arg(long, default_value_t = false)]
        ptq1_transcode: bool,

        /// Prompt-ingestion chunk size in tokens (design §5.7): the
        /// Gated-DeltaNet prefill chunk for a `qwen35` hybrid (default
        /// 512), the chunked-prefill plan for a dense model (default: that
        /// model's own plan). Must be >= 1.
        #[arg(long, value_parser = parse_prefill_chunk)]
        prefill_chunk: Option<usize>,

        /// Vision projector GGUF (`clip` architecture, e.g.
        /// `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`). Parsed and validated
        /// now; the vision tower isn't wired up yet, so passing it is a
        /// typed `NOT_YET_SUPPORTED` error, never a silently ignored flag.
        #[arg(long)]
        mmproj: Option<String>,

        /// Image input (path or http(s) URL; repeatable). Parsed and
        /// validated now; the vision tower isn't wired up yet, so
        /// passing it is a typed `NOT_YET_SUPPORTED` error, never a
        /// silently ignored flag.
        #[arg(long)]
        image: Vec<String>,

        /// Per-image token budget for the vision downscale guard (default
        /// 1024, matching the reference demo). Must be >= 1. The vision
        /// tower isn't wired up yet: passing it explicitly is a typed
        /// `NOT_YET_SUPPORTED` error.
        #[arg(long, value_parser = parse_image_max_tokens)]
        image_max_tokens: Option<usize>,

        /// Proceed even when the resolved tokenizer's vocabulary is
        /// SMALLER than the model's (TOK-08). A smaller vocabulary means a
        /// different BPE, not a subset, and will silently mis-tokenize
        /// every prompt — only pass this if you have independently
        /// verified the tokenizer is correct for this model. A tokenizer
        /// with a LARGER vocabulary than the model is always a hard error
        /// (it can emit token ids the embedding table cannot index) and
        /// this flag does not override that.
        #[arg(long, default_value_t = false)]
        allow_vocab_mismatch: bool,

        /// Wait for the full completion and print it once at the end
        /// instead of streaming tokens as they are generated. Streaming is
        /// now the default on every backend, including native CUDA
        /// builds (F14); pass this explicitly for non-interactive
        /// benchmarking where per-token print overhead is unwanted.
        #[arg(long, default_value_t = false)]
        no_stream: bool,
    },

    /// Generate an image from a text prompt (Bonsai-Image: TE → DiT → VAE → PNG).
    Image {
        /// Text prompt. Use "-" for stdin.
        #[arg(short, long)]
        prompt: String,

        /// Output PNG path. A relative path is placed under
        /// `[imagen].output_dir` when --config sets one.
        #[arg(short, long)]
        out: String,

        /// RNG seed for the initial noise (default: 42, or `[imagen].seed`
        /// in --config).
        #[arg(long)]
        seed: Option<u64>,

        /// Number of Euler sampler steps (default: 4, or `[imagen].steps`).
        #[arg(long)]
        steps: Option<usize>,

        /// Image width in pixels (default: 512, or `[imagen].width`).
        #[arg(long)]
        width: Option<usize>,

        /// Image height in pixels (default: 512, or `[imagen].height`).
        #[arg(long)]
        height: Option<usize>,

        /// DiT GGUF path. Required: pass this flag, set `[imagen].model_path`
        /// in --config, or set env OXI_DIT_GGUF (there is no default path —
        /// never a world-writable directory).
        #[arg(long)]
        dit: Option<String>,

        /// VAE weights dir. Required: pass this flag or set env
        /// OXI_VAE_WEIGHTS (there is no default path).
        #[arg(long)]
        vae: Option<String>,

        /// Text-encoder weights: a 4-bit model.safetensors file or an f32
        /// .npy dir. Required: pass this flag or set env OXI_TE_4BIT /
        /// OXI_TE_WEIGHTS (there is no default path).
        #[arg(long)]
        te: Option<String>,

        /// Tokenizer dir containing tokenizer.json
        /// (default: env OXI_TE_TOKENIZER_DIR, else the TE dir).
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Interactive image REPL: load the pipeline once, render many prompts.
    ///
    /// Keeps the DiT, VAE, and (resident) text encoder in memory so each
    /// prompt skips the load/dequant cost. On Ghostty the image is shown
    /// inline; elsewhere it is written to a file. Model paths resolve the
    /// same way as `image` (flag → env → error; no default path).
    Repl {
        /// Initial RNG seed, changeable at runtime with :seed (default: 42,
        /// or `[imagen].seed` in --config).
        #[arg(long)]
        seed: Option<u64>,

        /// Initial sampler steps, changeable with :steps / :fast / :hq
        /// (default: 4, or `[imagen].steps`).
        #[arg(long)]
        steps: Option<usize>,

        /// Initial image width in pixels (default: 512, or `[imagen].width`).
        #[arg(long)]
        width: Option<usize>,

        /// Initial image height in pixels (default: 512, or
        /// `[imagen].height`).
        #[arg(long)]
        height: Option<usize>,

        /// Run the text-encoder GEMM on the CPU instead of the Metal GPU.
        #[arg(long)]
        cpu_te: bool,

        /// DiT GGUF path. Required: pass this flag or set env OXI_DIT_GGUF
        /// (there is no default path).
        #[arg(long)]
        dit: Option<String>,

        /// VAE weights path. Required: pass this flag or set env
        /// OXI_VAE_WEIGHTS (there is no default path).
        #[arg(long)]
        vae: Option<String>,

        /// Text-encoder weights: a 4-bit model.safetensors file or an f32
        /// .npy dir. Required: pass this flag or set env OXI_TE_4BIT /
        /// OXI_TE_WEIGHTS (there is no default path).
        #[arg(long)]
        te: Option<String>,

        /// Tokenizer dir containing tokenizer.json
        /// (default: env OXI_TE_TOKENIZER_DIR, else the TE dir).
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Interactive multi-turn conversation.
    Chat {
        /// Path to the GGUF model file (default: env OXI_MODEL, or
        /// `[model].model_path` in --config).
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum number of tokens to generate per turn (default: 512,
        /// or `[sampling].max_tokens` in --config). Must be >= 1.
        #[arg(long, value_parser = parse_max_tokens)]
        max_tokens: Option<usize>,

        /// Sampling temperature; 0.0 = greedy argmax on every backend.
        /// Precedence (RT-17): this flag, else `[sampling].temperature`,
        /// else the model's `general.sampling.temp`, else 0.7. Must be
        /// >= 0.0.
        #[arg(long, value_parser = parse_temperature)]
        temperature: Option<f32>,

        /// Top-k sampling, 0 = disabled. Precedence: this flag, else
        /// `[sampling].top_k`, else the model's `general.sampling.top_k`,
        /// else 40.
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p (nucleus) sampling. Precedence: this flag, else
        /// `[sampling].top_p`, else the model's `general.sampling.top_p`,
        /// else 0.9. Must be in (0.0, 1.0].
        #[arg(long, value_parser = parse_top_p)]
        top_p: Option<f32>,

        /// Repetition penalty; 1.0 = disabled, applied on every backend
        /// (default: 1.0, or `[sampling].repetition_penalty` in
        /// --config). Must be > 0.0. A non-default value is a hard error
        /// when combined with --grammar or --stop (that decode loop
        /// cannot apply it).
        #[arg(long, value_parser = parse_repetition_penalty)]
        repetition_penalty: Option<f32>,

        /// OpenAI-style frequency penalty (default: 0.0 = disabled).
        /// Range [-2.0, 2.0]. Applies on every backend EXCEPT when
        /// combined with --grammar or --stop, where it is a hard error
        /// rather than a silently-dropped flag (that decode loop cannot
        /// apply it).
        #[arg(long, value_parser = parse_openai_penalty, allow_negative_numbers = true)]
        frequency_penalty: Option<f32>,

        /// OpenAI-style presence penalty (default: 0.0 = disabled).
        /// Range [-2.0, 2.0]. Applies on every backend EXCEPT when
        /// combined with --grammar or --stop, where it is a hard error
        /// rather than a silently-dropped flag (that decode loop cannot
        /// apply it).
        #[arg(long, value_parser = parse_openai_penalty, allow_negative_numbers = true)]
        presence_penalty: Option<f32>,

        /// Random seed (RT-12: the same seed reproduces the same session).
        /// Precedence: this flag, else `[sampling].seed` in --config, else
        /// 42.
        #[arg(long)]
        seed: Option<u64>,

        /// Maximum sequence length; `--ctx` is an alias (default: 8192 for
        /// a Bonsai 2 `qwen35` model, 4096 otherwise, or
        /// `[model].max_seq_len` in --config). Must be >= 1. The whole
        /// conversation is re-rendered through the chat template every
        /// turn, so the oldest turns are dropped once it no longer fits.
        #[arg(long, visible_alias = "ctx", value_parser = parse_max_seq_len)]
        max_seq_len: Option<usize>,

        /// Path to tokenizer.json file (default: auto-detected, or
        /// `[model].tokenizer_path` in --config).
        #[arg(long)]
        tokenizer: Option<String>,

        /// Which tokenizer backend to use.
        #[arg(long, value_enum, default_value_t = TokenizerBackendChoice::Auto)]
        tokenizer_backend: TokenizerBackendChoice,

        /// Constrain generation to a grammar: a `.gbnf` file, or any other
        /// extension is parsed as a JSON Schema.
        #[arg(long)]
        grammar: Option<String>,

        /// Stop generation as soon as any of these strings appears in the
        /// decoded output (may be passed multiple times).
        #[arg(long)]
        stop: Vec<String>,

        /// Min-p (probabilistic nucleus) sampling; 0.0 = disabled. See
        /// `run --help` for the precedence.
        #[arg(long, value_parser = parse_min_p)]
        min_p: Option<f32>,

        /// Which compute backend runs the model. See `run --help`.
        #[arg(long, value_parser = parse_backend)]
        backend: Option<oxibonsai_runtime::engine_seam::Backend>,

        /// RoPE scaling strategy. See `run --help`.
        #[arg(long, value_parser = parse_rope_scaling)]
        rope_scaling: Option<oxibonsai_runtime::config::RopeScalingMode>,

        /// Enable the `<think>` reasoning block for every turn
        /// (`enable_thinking = true`). Neither flag = the template's own
        /// default.
        #[arg(long, conflicts_with = "no_think")]
        think: bool,

        /// Disable the `<think>` reasoning block for every turn.
        #[arg(long)]
        no_think: bool,

        /// Reasoning effort for every turn: low, medium or xhigh.
        #[arg(long, value_parser = parse_reasoning_effort)]
        reasoning_effort: Option<String>,

        /// Path to a JSON file containing an OpenAI-style `tools` array,
        /// used for the whole session (raw JSON text, key order preserved).
        #[arg(long)]
        tools: Option<String>,

        /// Print each turn's `<think>` reasoning to stderr (the default).
        #[arg(long, conflicts_with = "hide_reasoning")]
        show_reasoning: bool,

        /// Drop each turn's `<think>` reasoning and print only the answer.
        #[arg(long)]
        hide_reasoning: bool,

        /// Transcode PTQ1_0 weights to PQ2_0 at load. See `run --help`.
        #[arg(long, default_value_t = false)]
        ptq1_transcode: bool,

        /// Prompt-ingestion chunk size. See `run --help`.
        #[arg(long, value_parser = parse_prefill_chunk)]
        prefill_chunk: Option<usize>,

        /// Vision projector GGUF. See `run --help` — a typed
        /// `NOT_YET_SUPPORTED` error until the vision tower lands.
        #[arg(long)]
        mmproj: Option<String>,

        /// Image input (path or URL; repeatable). See `run --help` — a
        /// typed `NOT_YET_SUPPORTED` error until the vision tower lands.
        #[arg(long)]
        image: Vec<String>,

        /// Per-image token budget (default 1024). See `run --help`.
        #[arg(long, value_parser = parse_image_max_tokens)]
        image_max_tokens: Option<usize>,

        /// Proceed even when the resolved tokenizer's vocabulary is
        /// SMALLER than the model's (TOK-08); see `run --help` for the
        /// full explanation. A larger tokenizer vocabulary is always a
        /// hard error regardless of this flag.
        #[arg(long, default_value_t = false)]
        allow_vocab_mismatch: bool,
    },

    /// Start an OpenAI-compatible API server.
    ///
    /// Environment: before loading, the model file's SHA-256 is checked
    /// against the checksum manifest when that manifest lists the file (a
    /// mismatch refuses to start). The manifest path defaults to
    /// `scripts/checksums.sha256`, resolved RELATIVE TO THE CURRENT WORKING
    /// DIRECTORY (so it is only found when `oxibonsai serve` is started
    /// from the repository root); set `OXIBONSAI_CHECKSUMS_FILE=<path>` to
    /// point at a manifest anywhere else. No manifest, or a manifest that
    /// does not list this file, means no hash check. Other environment
    /// knobs: `OXI_MODEL`, `OXI_TOKENIZER`, `OXIBONSAI_BEARER_TOKEN`,
    /// `OXIBONSAI_BEARER_TOKEN_FILE`, `OXIBONSAI_RATE_LIMIT_RPM`,
    /// `OXIBONSAI_RATE_LIMIT_BURST`, `OXIBONSAI_CORS_ORIGIN`,
    /// `OXIBONSAI_CORS_ALLOW_CREDENTIALS`, `OXIBONSAI_MAX_BODY_BYTES`,
    /// `OXIBONSAI_ENGINE_POOL_SIZE`, `OXIBONSAI_SEED`,
    /// `OXIBONSAI_INSECURE_NO_AUTH`, `OXIBONSAI_ADMIN_TOKEN` /
    /// `OXI_ADMIN_TOKEN`, `OXIBONSAI_CUDA_DEVICE` — each flag below wins
    /// over its environment variable.
    #[cfg(feature = "server")]
    Serve {
        /// Path to the GGUF model file (default: env OXI_MODEL, or
        /// `[model].model_path` in --config).
        #[arg(short, long)]
        model: Option<String>,

        /// Host to bind to (default: 127.0.0.1, or `[server].host` in
        /// --config). Applying a config value here never happens when
        /// `--host` was passed explicitly, so a TOML default can never
        /// silently widen the bind address.
        #[arg(long)]
        host: Option<String>,

        /// Port to listen on (default: 8080, or `[server].port` in
        /// --config).
        #[arg(long)]
        port: Option<u16>,

        /// Maximum sequence length per request (prompt + generated);
        /// `--ctx` is an alias (default: 8192 for a Bonsai 2 `qwen35`
        /// model, 4096 otherwise, or `[model].max_seq_len` in --config).
        /// Must be >= 1; for a `qwen35` model it is refused above both the
        /// model's declared context and the RAM-derived bound.
        #[arg(long, visible_alias = "ctx", value_parser = parse_max_seq_len)]
        max_seq_len: Option<usize>,

        /// Path to tokenizer.json file (default: auto-detected, or
        /// `[model].tokenizer_path` in --config).
        #[arg(long)]
        tokenizer: Option<String>,

        /// Number of engine replicas (default: min(4, CPU cores);
        /// auto-clamped to 1 on GPU/Metal). Replicas share one token-embedding
        /// table, so each adds only a KV cache.
        #[arg(long)]
        pool_size: Option<usize>,

        /// Bearer token required on every OpenAI-compatible endpoint
        /// except `/health` and `/metrics` (default: env
        /// `OXIBONSAI_BEARER_TOKEN`). When unset, those endpoints are
        /// unauthenticated — only safe behind `--host 127.0.0.1` or
        /// another trusted network boundary. This flag does NOT gate the
        /// separate `/admin/*` surface: that is always authenticated by
        /// its own `OXI_ADMIN_TOKEN` environment variable, and every
        /// `/admin/*` request is refused with 403 while that variable is
        /// unset, independent of `--bearer-token`.
        #[arg(long)]
        bearer_token: Option<String>,

        /// Maximum number of requests admitted concurrently; requests
        /// beyond this bound are rejected with 503 instead of queuing
        /// unbounded.
        #[arg(long, default_value_t = 32)]
        max_concurrent_requests: usize,

        /// Per-request timeout in milliseconds before a request is
        /// aborted with 408.
        #[arg(long, default_value_t = 60_000)]
        request_timeout_ms: u64,

        /// Also mount the RAG HTTP API (`/rag/index`, `/rag/query`,
        /// `/rag/stats`) alongside the OpenAI-compatible endpoints.
        /// Requires the `rag` build feature.
        #[cfg(feature = "rag")]
        #[arg(long, default_value_t = false)]
        rag: bool,

        /// Which compute backend runs the model. See `oxibonsai run --help`.
        #[arg(long, value_parser = parse_backend)]
        backend: Option<oxibonsai_runtime::engine_seam::Backend>,

        /// RoPE scaling strategy. See `oxibonsai run --help`.
        #[arg(long, value_parser = parse_rope_scaling)]
        rope_scaling: Option<oxibonsai_runtime::config::RopeScalingMode>,

        /// Server-wide default `enable_thinking = true`, applied to a chat
        /// request that carries neither `chat_template_kwargs.enable_thinking`
        /// nor a top-level `enable_thinking` (a request always overrides
        /// this default).
        #[arg(long, conflicts_with = "no_think")]
        think: bool,

        /// Server-wide default `enable_thinking = false` (see `--think`).
        #[arg(long)]
        no_think: bool,

        /// Server-wide default reasoning effort (low, medium, xhigh),
        /// applied to a chat request that carries no `reasoning_effort` of
        /// its own.
        #[arg(long, value_parser = parse_reasoning_effort)]
        reasoning_effort: Option<String>,

        /// Path to a JSON file containing an OpenAI-style `tools` array,
        /// applied (as raw JSON text, key order preserved) to a chat
        /// request that carries no `tools` of its own.
        #[arg(long)]
        tools: Option<String>,

        /// CUDA device ordinal to bind (maps onto `OXIBONSAI_CUDA_DEVICE`,
        /// applied before the async runtime starts; only takes effect on a
        /// native-CUDA build with more than one visible device). Also
        /// `[server].cuda_device` in --config.
        #[arg(long)]
        cuda_device: Option<u32>,

        /// Path to a file whose contents are the bearer token required on
        /// every OpenAI-compatible endpoint (default: env
        /// `OXIBONSAI_BEARER_TOKEN_FILE`). Mutually exclusive with
        /// `--bearer-token`: pass at most one.
        #[arg(long, conflicts_with = "bearer_token")]
        bearer_token_file: Option<String>,

        /// Requests admitted per minute per client IP before a `429` is
        /// returned (default: env `OXIBONSAI_RATE_LIMIT_RPM`, unset =
        /// disabled).
        #[arg(long)]
        rate_limit_rpm: Option<u32>,

        /// Burst allowance on top of `--rate-limit-rpm` (default: env
        /// `OXIBONSAI_RATE_LIMIT_BURST`, or a small multiple of the RPM
        /// when unset).
        #[arg(long)]
        rate_limit_burst: Option<u32>,

        /// `Access-Control-Allow-Origin` value for every response (default:
        /// env `OXIBONSAI_CORS_ORIGIN`, unset = no CORS headers added).
        #[arg(long)]
        cors_origin: Option<String>,

        /// Send `Access-Control-Allow-Credentials: true` (default: env
        /// `OXIBONSAI_CORS_ALLOW_CREDENTIALS`). Requires `--cors-origin` to
        /// be a specific origin, never `*` (the CORS spec forbids
        /// combining a wildcard origin with credentials).
        #[arg(long, default_value_t = false)]
        cors_allow_credentials: bool,

        /// Maximum accepted request body size in bytes before a `413` is
        /// returned (default: env `OXIBONSAI_MAX_BODY_BYTES`, or a
        /// conservative built-in default when unset).
        #[arg(long)]
        max_body_bytes: Option<u64>,

        /// Mount the bundled minimal chat UI at `GET /ui` (SV-26; off by
        /// default — the page is unauthenticated whenever the server is).
        /// Also `[server].enable_ui` in --config.
        #[arg(long, default_value_t = false)]
        enable_ui: bool,

        /// Hard ceiling on a request's effective `max_tokens` /
        /// `max_completion_tokens` (SV-28): a request asking for more is
        /// rejected with 400, naming the ceiling (default: the server's
        /// compiled-in 8192). Must be >= 1. Also
        /// `[server].max_output_tokens` in --config.
        #[arg(long, value_parser = parse_max_output_tokens)]
        max_output_tokens: Option<usize>,

        /// Transcode PTQ1_0 weights to PQ2_0 at load. See `oxibonsai run
        /// --help`.
        #[arg(long, default_value_t = false)]
        ptq1_transcode: bool,

        /// Prompt-ingestion chunk size for every replica. See `oxibonsai
        /// run --help`.
        #[arg(long, value_parser = parse_prefill_chunk)]
        prefill_chunk: Option<usize>,

        /// Vision projector GGUF. A typed `NOT_YET_SUPPORTED` error until
        /// the vision tower lands (see `oxibonsai run --help`).
        #[arg(long)]
        mmproj: Option<String>,

        /// Image input for the vision tower (path or URL; repeatable). A
        /// typed `NOT_YET_SUPPORTED` error until the vision tower lands.
        #[arg(long)]
        image: Vec<String>,

        /// Per-image token budget (default 1024). A typed
        /// `NOT_YET_SUPPORTED` error when passed, until the vision tower lands.
        #[arg(long, value_parser = parse_image_max_tokens)]
        image_max_tokens: Option<usize>,

        /// Which backend answers `/v1/embeddings`: `model` (default: the
        /// loaded model's mean-pooled hidden states; a hybrid `qwen35`
        /// model has none yet and answers 501), `none` (always 501), or
        /// `tfidf` (lexical TF-IDF vectors over the vocabulary fitted, once
        /// at startup, on `--embedding-corpus`).
        #[arg(long, value_enum, default_value_t = EmbeddingBackendChoice::Model)]
        embedding_backend: EmbeddingBackendChoice,

        /// The corpus `--embedding-backend tfidf` fits its vocabulary and
        /// IDF weights on: a UTF-8 text file, one document per line (blank
        /// lines ignored). Required by, and only valid with, `tfidf`.
        #[arg(long)]
        embedding_corpus: Option<String>,
    },

    /// Display model info from a GGUF file.
    Info {
        /// Path to the GGUF model file (default: env OXI_MODEL).
        #[arg(short, long)]
        model: Option<String>,

        /// Emit info as JSON instead of human-readable text.
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Print what this `oxibonsai` binary was built with: enabled Cargo
    /// features, compiled-in kernel tiers, the kernel tier actually
    /// detected at runtime (with the reason), and a best-effort git
    /// commit hash. Takes no model — unlike `info`, it never requires one.
    BuildInfo,

    /// Run a throughput benchmark.
    ///
    /// With `--model`, benchmarks the real loaded model end to end
    /// (tokenizer, prefill, decode). Without `--model`, `--synthetic` must
    /// be passed explicitly to run the untrained 2-layer toy benchmark —
    /// its numbers are not representative of any real model and must never
    /// be the silent default.
    Benchmark {
        /// Path to a real GGUF model to benchmark (default: env
        /// OXI_MODEL). When given, `--synthetic` is ignored.
        #[arg(short, long)]
        model: Option<String>,

        /// Run the synthetic, untrained 2-layer toy benchmark instead of a
        /// real model. Only meaningful when `--model` is not given.
        #[arg(long, default_value_t = false)]
        synthetic: bool,

        /// Path to tokenizer.json file for `--model` (default:
        /// auto-detected).
        #[arg(long)]
        tokenizer: Option<String>,

        /// Which tokenizer backend loads the on-disk tokenizer for
        /// `--model` (same choices as `run --tokenizer-backend`).
        #[arg(long, value_enum, default_value_t = TokenizerBackendChoice::Auto)]
        tokenizer_backend: TokenizerBackendChoice,

        /// Total tokens to generate during the timed benchmark pass.
        #[arg(long, default_value_t = 100)]
        tokens: usize,

        /// Number of warmup tokens generated before timing begins.
        #[arg(long, default_value_t = 10)]
        warmup: usize,

        /// Sampling temperature.
        #[arg(long, value_parser = parse_temperature, default_value_t = 0.7)]
        temperature: f32,

        /// Random seed. Precedence: this flag, else `[sampling].seed` in
        /// --config, else 42.
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Quantize a GGUF model to a lower-precision format.
    ///
    /// Streams tensor by tensor (CQ-17): each source tensor is dequantized
    /// to f32 and re-encoded through the real `oxibonsai_model::export`
    /// pipeline, so only one tensor is resident at a time, and the source
    /// file's architecture/tokenizer metadata is carried over into the
    /// GGUF written at `--output`.
    Quantize {
        /// Path to the input GGUF model file.
        #[arg(long)]
        input: String,

        /// Destination path for the quantized file.
        #[arg(long)]
        output: String,

        /// Target quantization format: f32, q1_0 (Q1_0_g128), tq2_0_g128
        /// (ternary), fp8_e4m3, fp8_e5m2, q4_0, q8_0, q2_k, q3_k, q4_k,
        /// q5_k, q6_k, q8_k.
        #[arg(long, default_value = "q1_0")]
        format: String,

        /// Skip the up-front memory-estimate guard and proceed even when
        /// dequantizing the single largest tensor to f32 is estimated to
        /// need more RAM than this machine reports available (or, when
        /// that cannot be determined, more than a conservative 8 GiB). The
        /// guard exists to fail fast with a clear message instead of the
        /// OS OOM killer, not to prevent every large quantize.
        #[arg(long, default_value_t = false)]
        force: bool,
    },

    /// Validate that a GGUF file is well-formed and display a metadata summary.
    Validate {
        /// Path to the GGUF model file to validate.
        #[arg(short, long)]
        model: String,
    },

    /// Convert a HuggingFace safetensors model to GGUF format.
    Convert {
        /// Input directory containing model.safetensors (or shards) and config.json.
        #[arg(long)]
        from: String,

        /// Output GGUF file path.
        #[arg(long)]
        to: String,

        /// Quantization format: "tq2_0_g128" (default, ternary
        /// {-1,0,+1}, group 128), "q1_0_g128" (1-bit sign + FP16 group
        /// scale), "pq2_0" (PrismML 2-bit ternary, ggml id 142),
        /// "ptq1_0" (PrismML 1.75-bit ternary, ggml id 143) or "q2_0_g64"
        /// (mainline group-64 Q2_0). Any other value is rejected with an
        /// error before any work is done.
        #[arg(long, default_value = "tq2_0_g128")]
        quant: String,

        /// Treat --from as an ONNX model file (MatMulNBits, bits=2) and use the ONNX→GGUF converter.
        #[arg(long, default_value_t = false)]
        onnx: bool,

        /// Proceed (with a warning per tensor) when the source checkpoint
        /// contains tensors the converter cannot map onto a GGUF name,
        /// instead of failing loudly (CQ-04). Such tensors are dropped
        /// from the output.
        #[arg(long, default_value_t = false)]
        allow_unmapped: bool,
    },

    /// Evaluate a model against a real evaluation harness task, wired to
    /// the loaded engine end to end. Requires the `eval` build feature.
    ///
    /// `--dataset`'s expected JSONL shape depends on `--task`; see each
    /// `EvalTask` variant's own doc for the exact fields.
    #[cfg(feature = "eval")]
    Eval {
        /// Path to the GGUF model file (default: env OXI_MODEL).
        #[arg(short, long)]
        model: Option<String>,

        /// Which evaluation task to run.
        #[arg(long, value_enum, default_value_t = EvalTask::Rouge)]
        task: EvalTask,

        /// JSONL dataset path; shape depends on `--task` (see `EvalTask`).
        #[arg(long)]
        dataset: String,

        /// Only evaluate the first N examples (default: all).
        #[arg(long)]
        limit: Option<usize>,

        /// Maximum tokens generated per example (generation-based tasks only:
        /// rouge, bleu, chrf, meteor, qa, gsm8k, exact-match).
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        /// Maximum sequence length (prompt + generated) (default: 4096, or
        /// `[model].max_seq_len` in --config). Must be >= 1.
        #[arg(long, value_parser = parse_max_seq_len)]
        max_seq_len: Option<usize>,

        /// Path to tokenizer.json file.
        #[arg(long)]
        tokenizer: Option<String>,

        /// Which tokenizer backend to use.
        #[arg(long, value_enum, default_value_t = TokenizerBackendChoice::Auto)]
        tokenizer_backend: TokenizerBackendChoice,

        /// Also write the report as JSON to this path.
        #[arg(long)]
        report_json: Option<String>,

        /// Also write the report as Markdown to this path.
        #[arg(long)]
        report_markdown: Option<String>,

        /// Proceed even when the resolved tokenizer's vocabulary is
        /// SMALLER than the model's (TOK-08); see `run --help` for the
        /// full explanation. A larger tokenizer vocabulary is always a
        /// hard error regardless of this flag.
        #[arg(long, default_value_t = false)]
        allow_vocab_mismatch: bool,
    },

    /// Manage the Qwen3 tokenizer (download / inspect).
    Tokenizer {
        #[command(subcommand)]
        cmd: TokenizerCmd,
    },

    /// Download a Bonsai 2 / Bonsai model artifact (cli-05).
    ///
    /// `<MODEL_OR_URL>` is either a known name (`bonsai2-27b`,
    /// `bonsai2-27b-ptq1_0`, `bonsai2-27b-pq2_0`, `bonsai2-27b-mmproj`,
    /// `bonsai2-27b-q2_0`, `bonsai-27b-q1_0`, `ternary-bonsai-27b-pq2_0`,
    /// `ternary-bonsai-27b-q2_0`, `bonsai-8b`) or a full `http(s)://` URL.
    ///
    /// A named entry is verified FAIL-CLOSED against the SHA-256 and byte
    /// size compiled into this binary (never a CWD-relative file), after a
    /// structural check (GGUF magic, the expected `general.architecture`,
    /// and — for a Bonsai 2 language GGUF — `prism.hadamard.version ==
    /// 1`). `scripts/checksums.sha256` / `OXIBONSAI_CHECKSUMS_FILE`, when
    /// readable, is an additional cross-check, and the only hash authority
    /// for a bare URL. Downloads stream to `<file>.part` with a live
    /// progress line and resume from it; never auto-downloads at inference
    /// time. `OXIBONSAI_HF_BASE_URL` points at a mirror;
    /// `OXI_BONSAI2_REPO` / `OXI_BONSAI2_DEV_REPO` override the two Bonsai
    /// 2 repositories.
    Pull {
        /// Model name or a full http(s) URL.
        model_or_url: String,

        /// Output directory (default: `models`).
        #[arg(long, default_value = "models")]
        out: String,

        /// For `bonsai2-27b`: which quantization band to fetch (`pq2`, the
        /// default, or `ptq1`).
        #[arg(long, default_value = "pq2")]
        band: String,

        /// For `bonsai2-27b`: also fetch the mmproj vision projector.
        #[arg(long, default_value_t = false)]
        vision: bool,

        /// Overwrite an existing file without prompting.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
}

/// Which evaluator `oxibonsai eval --task` should run.
///
/// This is the whole point of `oxibonsai-eval`, made reachable from the
/// CLI (RAG-EVAL-IMG-16): the harness previously wired only `Rouge`
/// end-to-end, leaving every other evaluator unreachable from a real
/// model.
#[cfg(feature = "eval")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum EvalTask {
    /// ROUGE-1/2/L. Generation-based. Dataset: `{"input", "expected_output"}`
    /// JSONL (`EvalDataset`).
    Rouge,
    /// Corpus BLEU. Generation-based. Same dataset shape as `rouge`.
    Bleu,
    /// chrF (character n-gram F-score), averaged per example. Generation-based.
    /// Same dataset shape as `rouge`.
    Chrf,
    /// METEOR (lexical), averaged per example. Generation-based. Same
    /// dataset shape as `rouge`.
    Meteor,
    /// SQuAD-style Exact Match + token F1. Generation-based. Same dataset
    /// shape as `rouge`.
    Qa,
    /// GSM8K grade-school math (`"#### N"` answer extraction).
    /// Generation-based. Same dataset shape as `rouge`.
    Gsm8k,
    /// Exact-match accuracy of the raw generated string. Generation-based.
    /// Same dataset shape as `rouge`.
    ExactMatch,
    /// Perplexity of the model over each example's `input` text
    /// (teacher-forced, no generation). Dataset: `{"input"}` JSONL
    /// (`EvalDataset`; `expected_output` is ignored).
    Perplexity,
    /// MMLU-style multiple choice, with a per-subject breakdown when
    /// `subject` is present. Logit-based (no generation). Dataset:
    /// `{"id", "question", "choices": [...], "correct_answer", "subject"?}`
    /// JSONL (`McDataset`).
    Mmlu,
    /// ARC-Easy. Logit-based. Same dataset shape as `mmlu`.
    ArcEasy,
    /// ARC-Challenge. Logit-based. Same dataset shape as `mmlu`.
    ArcChallenge,
    /// HellaSwag 4-way sentence completion. Logit-based. Dataset:
    /// `{"ind", "activity_label", "ctx", "endings": [4 strings], "label"}`
    /// JSONL (`HellaSwagDataset`).
    Hellaswag,
    /// WinoGrande 2-way fill-in-the-blank. Logit-based. Dataset:
    /// `{"sentence", "option1", "option2", "answer": 1|2}` JSONL
    /// (`WinoGrandeDataset`).
    Winogrande,
    /// BoolQ yes/no reading comprehension. Logit-based (scores the "yes"
    /// vs "no" continuation). Dataset: `{"passage", "question", "answer":
    /// true|false}` JSONL.
    Boolq,
    /// TruthfulQA MC1 (single correct answer, argmax). Logit-based.
    /// Dataset: `{"question", "mc1_targets": {"choices", "labels"},
    /// "mc2_targets": {"choices", "labels"}}` JSONL (`TruthfulQaDataset`).
    TruthfulqaMc1,
    /// TruthfulQA MC2 (probabilistic: mean softmax mass on every correct
    /// answer). Logit-based. Same dataset shape as `truthfulqa-mc1`.
    TruthfulqaMc2,
    /// ECE + Brier + NLL calibration metrics computed from the same
    /// per-choice log-probabilities as `mmlu`. Logit-based. Same dataset
    /// shape as `mmlu`.
    Calibration,
}

#[derive(Subcommand)]
pub(crate) enum TokenizerCmd {
    /// Download tokenizer.json from HuggingFace and save it next to the model.
    ///
    /// Example:
    ///   oxibonsai tokenizer download --output models/tokenizer.json
    Download {
        /// Destination path (default: models/tokenizer.json in the current directory).
        #[arg(long, default_value = "models/tokenizer.json")]
        output: String,

        /// HuggingFace repo to download from (must contain tokenizer.json).
        #[arg(long, default_value = "Qwen/Qwen3-8B")]
        repo: String,

        /// Overwrite an existing tokenizer.json without prompting.
        #[arg(long, default_value_t = false)]
        force: bool,
    },

    /// Show the vocabulary size and model type stored in tokenizer.json.
    Info {
        /// Path to tokenizer.json.
        #[arg(long, default_value = "models/tokenizer.json")]
        path: String,
    },
}

#[cfg(test)]
#[path = "args_tests.rs"]
mod tests;
