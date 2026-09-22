//! OxiBonsai — Pure Rust sub-2-bit LLM inference engine for PrismML Bonsai
//! models.
//!
//! This binary is not functional on WASM targets; the WASM entry points
//! are exposed via [`oxibonsai_runtime`] library APIs instead.

/// WASM stub: this binary is a no-op on wasm32 targets.
/// Consumers should use the `oxibonsai_runtime` library crate APIs directly.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod cli;

/// Native (non-WASM) entry point.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    // Auto-load `.env` (cwd or any parent dir) as the very first thing, before
    // the tokio runtime, tracing, or any `std::env::var` read. `dotenvy::dotenv`
    // does NOT override already-set real env vars, so precedence stays
    // explicit --flag > shell env > .env file > built-in default. A missing
    // `.env` is a silent no-op via `.ok()`, so behavior is unchanged without one.
    match dotenvy::dotenv() {
        Ok(path) => tracing::debug!(path = %path.display(), "loaded .env"),
        Err(_) => { /* no .env found (or unreadable): silently continue */ }
    }

    // Parse argv before building the tokio runtime. cli-24: this is what
    // lets `apply_pre_runtime_env` set the `OXI_TE_GPU` env-var latch
    // while this is still the only thread in the process — see that
    // function's doc for why that matters.
    use clap::Parser;
    let cli = cli::Cli::parse();
    cli::apply_pre_runtime_env(&cli);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build tokio runtime: {e}"))?
        .block_on(cli::run_with(cli))
}
