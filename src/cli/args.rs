//! `clap` CLI surface: the `oxibonsai` argument parser and subcommand table.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "oxibonsai",
    version,
    about = "1-bit LLM inference engine for Bonsai-8B"
)]
pub(crate) struct Cli {
    /// Path to an OxiBonsai TOML configuration file.
    #[arg(long, global = true)]
    pub(crate) config: Option<String>,

    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Run inference on a GGUF model.
    Run {
        /// Path to the GGUF model file (default: env OXI_MODEL).
        #[arg(short, long)]
        model: Option<String>,

        /// Prompt text. Use "-" for stdin.
        #[arg(short, long)]
        prompt: String,

        /// Maximum number of tokens to generate.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy).
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Top-k sampling (0 = disabled).
        #[arg(long, default_value_t = 40)]
        top_k: usize,

        /// Top-p (nucleus) sampling.
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Maximum sequence length (prompt + generated).
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,

        /// Path to tokenizer.json file.
        #[arg(long)]
        tokenizer: Option<String>,
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

        /// DiT GGUF path (default: env OXI_DIT_GGUF or /tmp/parity.gguf).
        #[arg(long)]
        dit: Option<String>,

        /// VAE weights dir (default: env OXI_VAE_WEIGHTS or /tmp/bonsai_golden/vae/weights).
        #[arg(long)]
        vae: Option<String>,

        /// Text-encoder weights: a 4-bit model.safetensors file or an f32 .npy dir
        /// (default: env OXI_TE_4BIT, else env OXI_TE_WEIGHTS, else /tmp/bonsai_golden/te/weights).
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
    /// same way as `image` (flag → env → default).
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

        /// DiT GGUF path (default: env OXI_DIT_GGUF or /tmp/parity.gguf).
        #[arg(long)]
        dit: Option<String>,

        /// VAE weights path (default: env OXI_VAE_WEIGHTS).
        #[arg(long)]
        vae: Option<String>,

        /// Text-encoder weights: a 4-bit model.safetensors file or an f32
        /// .npy dir (default: env OXI_TE_4BIT, else OXI_TE_WEIGHTS).
        #[arg(long)]
        te: Option<String>,

        /// Tokenizer dir containing tokenizer.json
        /// (default: env OXI_TE_TOKENIZER_DIR, else the TE dir).
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Interactive multi-turn conversation.
    Chat {
        /// Path to the GGUF model file (default: env OXI_MODEL).
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum number of tokens to generate per turn.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy).
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,

        /// Top-k sampling (0 = disabled).
        #[arg(long, default_value_t = 40)]
        top_k: usize,

        /// Top-p (nucleus) sampling.
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Maximum sequence length.
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,

        /// Path to tokenizer.json file.
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Start an OpenAI-compatible API server.
    #[cfg(feature = "server")]
    Serve {
        /// Path to the GGUF model file (default: env OXI_MODEL).
        #[arg(short, long)]
        model: Option<String>,

        /// Host to bind to.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,

        /// Maximum sequence length.
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,

        /// Path to tokenizer.json file.
        #[arg(long)]
        tokenizer: Option<String>,

        /// Number of engine replicas (default: min(4, CPU cores);
        /// auto-clamped to 1 on GPU/Metal). Replicas share one token-embedding
        /// table, so each adds only a KV cache.
        #[arg(long)]
        pool_size: Option<usize>,

        /// Bearer token required on every endpoint except `/health` and
        /// `/metrics` (default: env `OXIBONSAI_BEARER_TOKEN`). When
        /// unset, the server is unauthenticated — only safe behind
        /// `--host 127.0.0.1` or another trusted network boundary, since
        /// this also gates the `/admin/*` endpoints (including the
        /// mutating `POST /admin/reset-metrics`).
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

    /// Run a quick throughput benchmark (no real model weights required).
    Benchmark {
        /// Total tokens to generate during the benchmark pass.
        #[arg(long, default_value_t = 100)]
        tokens: usize,

        /// Number of warmup tokens generated before timing begins.
        #[arg(long, default_value_t = 10)]
        warmup: usize,

        /// Sampling temperature.
        #[arg(long, default_value_t = 0.7)]
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

    /// Evaluate a model's generation quality (ROUGE-1/2/L) against a
    /// JSONL dataset of `{"input": <prompt>, "expected_output": <reference>}`
    /// examples. Requires the `eval` build feature.
    #[cfg(feature = "eval")]
    Eval {
        /// Path to the GGUF model file (default: env OXI_MODEL).
        #[arg(short, long)]
        model: Option<String>,

        /// JSONL dataset path: one `{"input": "...", "expected_output": "..."}`
        /// object per line (see `oxibonsai_eval::EvalDataset::from_jsonl`).
        #[arg(long)]
        dataset: String,

        /// Only evaluate the first N examples (default: all).
        #[arg(long)]
        limit: Option<usize>,

        /// Maximum tokens generated per example.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,

        /// Maximum sequence length (prompt + generated).
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,

        /// Path to tokenizer.json file.
        #[arg(long)]
        tokenizer: Option<String>,

        /// Also write the report as JSON to this path.
        #[arg(long)]
        report_json: Option<String>,

        /// Also write the report as Markdown to this path.
        #[arg(long)]
        report_markdown: Option<String>,
    },

    /// Manage the Qwen3 tokenizer (download / inspect).
    Tokenizer {
        #[command(subcommand)]
        cmd: TokenizerCmd,
    },
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
