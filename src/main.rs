//! OxiBonsai — 1-bit LLM inference engine for Bonsai-8B.
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

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build tokio runtime: {e}"))?
        .block_on(cli::run())
}
