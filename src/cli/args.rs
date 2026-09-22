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

        /// Sampling temperature; 0.0 = greedy argmax on every backend
        /// (default: 0.7, or `[sampling].temperature` in --config). Must
        /// be >= 0.0.
        #[arg(long, value_parser = parse_temperature)]
        temperature: Option<f32>,

        /// Top-k sampling, 0 = disabled (default: 40, or
        /// `[sampling].top_k` in --config).
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p (nucleus) sampling (default: 0.9, or `[sampling].top_p`
        /// in --config). Must be in (0.0, 1.0].
        #[arg(long, value_parser = parse_top_p)]
        top_p: Option<f32>,

        /// Repetition penalty; 1.0 = disabled, applied on every backend
        /// (default: 1.0, or `[sampling].repetition_penalty` in
        /// --config). Must be > 0.0. Unlike `oxibonsai_runtime`'s
        /// internal `SamplingParams::default()`, this CLI never applies a
        /// hidden non-1.0 value, so `--temperature 0` means exactly
        /// argmax unless a penalty is explicitly requested. A non-default
        /// value is a hard error when combined with --grammar or --stop
        /// (that decode loop cannot apply it).
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

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Maximum sequence length (prompt + generated) (default: 4096,
        /// or `[model].max_seq_len` in --config). Must be >= 1.
        #[arg(long, value_parser = parse_max_seq_len)]
        max_seq_len: Option<usize>,

        /// Path to tokenizer.json file (default: auto-detected, or
        /// `[model].tokenizer_path` in --config).
        #[arg(long)]
        tokenizer: Option<String>,

        /// Which tokenizer backend to use.
        #[arg(long, value_enum, default_value_t = TokenizerBackendChoice::Auto)]
        tokenizer_backend: TokenizerBackendChoice,

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

        /// Output PNG path.
        #[arg(short, long)]
        out: String,

        /// RNG seed for the initial noise.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Number of Euler sampler steps.
        #[arg(long, default_value_t = 4)]
        steps: usize,

        /// Image width in pixels.
        #[arg(long, default_value_t = 512)]
        width: usize,

        /// Image height in pixels.
        #[arg(long, default_value_t = 512)]
        height: usize,

        /// DiT GGUF path. Required: pass this flag or set env OXI_DIT_GGUF
        /// (there is no default path — never a world-writable directory).
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
        /// Initial RNG seed (changeable at runtime with :seed).
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Initial sampler steps (changeable with :steps / :fast / :hq).
        #[arg(long, default_value_t = 4)]
        steps: usize,

        /// Initial image width in pixels.
        #[arg(long, default_value_t = 512)]
        width: usize,

        /// Initial image height in pixels.
        #[arg(long, default_value_t = 512)]
        height: usize,

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

        /// Sampling temperature; 0.0 = greedy argmax on every backend
        /// (default: 0.7, or `[sampling].temperature` in --config). Must
        /// be >= 0.0.
        #[arg(long, value_parser = parse_temperature)]
        temperature: Option<f32>,

        /// Top-k sampling, 0 = disabled (default: 40, or
        /// `[sampling].top_k` in --config).
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p (nucleus) sampling (default: 0.9, or `[sampling].top_p`
        /// in --config). Must be in (0.0, 1.0].
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

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Maximum sequence length (default: 4096, or
        /// `[model].max_seq_len` in --config). Must be >= 1.
        #[arg(long, value_parser = parse_max_seq_len)]
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

        /// Proceed even when the resolved tokenizer's vocabulary is
        /// SMALLER than the model's (TOK-08); see `run --help` for the
        /// full explanation. A larger tokenizer vocabulary is always a
        /// hard error regardless of this flag.
        #[arg(long, default_value_t = false)]
        allow_vocab_mismatch: bool,
    },

    /// Start an OpenAI-compatible API server.
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

        /// Maximum sequence length (default: 4096, or
        /// `[model].max_seq_len` in --config). Must be >= 1.
        #[arg(long, value_parser = parse_max_seq_len)]
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

        /// Total tokens to generate during the timed benchmark pass.
        #[arg(long, default_value_t = 100)]
        tokens: usize,

        /// Number of warmup tokens generated before timing begins.
        #[arg(long, default_value_t = 10)]
        warmup: usize,

        /// Sampling temperature.
        #[arg(long, value_parser = parse_temperature, default_value_t = 0.7)]
        temperature: f32,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Quantize a GGUF model to a lower-precision format.
    ///
    /// Dequantizes every source tensor to f32 and re-encodes it through
    /// the real `oxibonsai_model::export` pipeline, writing an actual
    /// GGUF file at `--output`.
    Quantize {
        /// Path to the input GGUF model file.
        #[arg(long)]
        input: String,

        /// Destination path for the quantized file.
        #[arg(long)]
        output: String,

        /// Target quantization format: f32, q1_0 (Q1_0_g128), tq2_0_g128
        /// (ternary), fp8_e4m3, fp8_e5m2, q4_0, q8_0, q4_k, q5_k, q6_k.
        #[arg(long, default_value = "q1_0")]
        format: String,

        /// Skip the up-front memory-estimate guard and proceed even when
        /// dequantizing every tensor to f32 is estimated to need more RAM
        /// than this machine reports available (or, when that cannot be
        /// determined, more than a conservative 8 GiB). This command
        /// still materialises the whole model in RAM (cli-14); the guard
        /// exists to fail fast with a clear message instead of the OS OOM
        /// killer, not to prevent every large quantize.
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
        /// {-1,0,+1}) or "q1_0_g128" (1-bit sign + FP16 group scale).
        /// Any other value is rejected with an error before any work is
        /// done.
        #[arg(long, default_value = "tq2_0_g128")]
        quant: String,

        /// Treat --from as an ONNX model file (MatMulNBits, bits=2) and use the ONNX→GGUF converter.
        #[arg(long, default_value_t = false)]
        onnx: bool,
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
mod tests {
    use super::*;

    #[test]
    fn parse_temperature_rejects_negative() {
        assert!(parse_temperature("-1").is_err());
        assert!(parse_temperature("-0.0001").is_err());
    }

    #[test]
    fn parse_temperature_accepts_zero_and_positive() {
        assert_eq!(parse_temperature("0").unwrap(), 0.0);
        assert_eq!(parse_temperature("0.7").unwrap(), 0.7);
    }

    #[test]
    fn parse_temperature_rejects_nan_and_infinity() {
        assert!(parse_temperature("nan").is_err());
        assert!(parse_temperature("inf").is_err());
    }

    #[test]
    fn parse_top_p_rejects_out_of_range() {
        assert!(
            parse_top_p("0").is_err(),
            "0.0 excluded (empty distribution)"
        );
        assert!(parse_top_p("1.5").is_err());
        assert!(parse_top_p("-0.1").is_err());
    }

    #[test]
    fn parse_top_p_accepts_valid_range() {
        assert_eq!(parse_top_p("1.0").unwrap(), 1.0);
        assert_eq!(parse_top_p("0.9").unwrap(), 0.9);
        assert!(parse_top_p("0.0001").is_ok());
    }

    #[test]
    fn parse_repetition_penalty_rejects_non_positive() {
        assert!(parse_repetition_penalty("0").is_err());
        assert!(parse_repetition_penalty("-1.1").is_err());
    }

    #[test]
    fn parse_repetition_penalty_accepts_positive() {
        assert_eq!(parse_repetition_penalty("1.0").unwrap(), 1.0);
        assert_eq!(parse_repetition_penalty("1.1").unwrap(), 1.1);
    }

    #[test]
    fn parse_openai_penalty_accepts_negative_within_range() {
        assert_eq!(parse_openai_penalty("-2.0").unwrap(), -2.0);
        assert_eq!(parse_openai_penalty("0.0").unwrap(), 0.0);
        assert_eq!(parse_openai_penalty("2.0").unwrap(), 2.0);
    }

    #[test]
    fn parse_openai_penalty_rejects_out_of_range() {
        assert!(parse_openai_penalty("-2.1").is_err());
        assert!(parse_openai_penalty("2.1").is_err());
    }

    #[test]
    fn parse_max_tokens_rejects_zero() {
        assert!(parse_max_tokens("0").is_err());
    }

    #[test]
    fn parse_max_tokens_accepts_positive() {
        assert_eq!(parse_max_tokens("1").unwrap(), 1);
        assert_eq!(parse_max_tokens("256").unwrap(), 256);
    }

    #[test]
    fn parse_max_seq_len_rejects_zero() {
        assert!(parse_max_seq_len("0").is_err());
    }

    #[test]
    fn cli_parses_run_with_new_penalty_flags() {
        let cli = Cli::try_parse_from([
            "oxibonsai",
            "run",
            "--prompt",
            "hi",
            "--repetition-penalty",
            "1.2",
            "--frequency-penalty",
            "-0.5",
            "--presence-penalty",
            "0.5",
        ])
        .expect("should parse");
        match cli.command {
            Commands::Run {
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                ..
            } => {
                assert_eq!(repetition_penalty, Some(1.2));
                assert_eq!(frequency_penalty, Some(-0.5));
                assert_eq!(presence_penalty, Some(0.5));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn cli_rejects_negative_temperature() {
        let result =
            Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi", "--temperature", "-1"]);
        assert!(result.is_err(), "negative temperature must be rejected");
    }

    #[test]
    fn cli_rejects_top_p_above_one() {
        let result = Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi", "--top-p", "5.0"]);
        assert!(result.is_err(), "top-p > 1.0 must be rejected");
    }

    #[test]
    fn cli_rejects_zero_max_tokens() {
        let result =
            Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi", "--max-tokens", "0"]);
        assert!(result.is_err(), "max-tokens == 0 must be rejected");
    }

    #[test]
    fn cli_defaults_are_none_for_config_layering() {
        let cli = Cli::try_parse_from(["oxibonsai", "run", "--prompt", "hi"]).expect("parse");
        match cli.command {
            Commands::Run {
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_tokens,
                max_seq_len,
                ..
            } => {
                assert_eq!(temperature, None);
                assert_eq!(top_k, None);
                assert_eq!(top_p, None);
                assert_eq!(repetition_penalty, None);
                assert_eq!(max_tokens, None);
                assert_eq!(max_seq_len, None);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn benchmark_requires_explicit_synthetic_or_model() {
        let cli = Cli::try_parse_from(["oxibonsai", "benchmark"]).expect("parse");
        match cli.command {
            Commands::Benchmark {
                model, synthetic, ..
            } => {
                assert!(model.is_none());
                assert!(
                    !synthetic,
                    "synthetic must default to false, not silently on"
                );
            }
            _ => panic!("expected Benchmark"),
        }
    }

    #[test]
    fn validate_temperature_rejects_a_value_that_never_went_through_clap() {
        // The scenario a config-file-sourced value hits: parsed by
        // `mod.rs`'s own `toml_f32` (plain `str::parse`, no clap
        // `value_parser`), then re-validated by calling this directly.
        assert!(validate_temperature(-5.0).is_err());
        assert!(validate_temperature(0.0).is_ok());
    }

    #[test]
    fn validate_top_p_rejects_a_value_that_never_went_through_clap() {
        assert!(validate_top_p(5.0).is_err());
        assert!(validate_top_p(0.5).is_ok());
    }

    #[test]
    fn validate_repetition_penalty_rejects_a_value_that_never_went_through_clap() {
        assert!(validate_repetition_penalty(-1.0).is_err());
        assert!(validate_repetition_penalty(1.0).is_ok());
    }

    #[test]
    fn validate_max_tokens_rejects_a_value_that_never_went_through_clap() {
        assert!(validate_max_tokens(0).is_err());
        assert!(validate_max_tokens(1).is_ok());
    }

    #[test]
    fn quantize_force_defaults_to_false() {
        let cli = Cli::try_parse_from([
            "oxibonsai",
            "quantize",
            "--input",
            "a.gguf",
            "--output",
            "b.gguf",
        ])
        .expect("parse");
        match cli.command {
            Commands::Quantize { force, .. } => assert!(!force),
            _ => panic!("expected Quantize"),
        }
    }
}
