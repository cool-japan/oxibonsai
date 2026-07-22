//! Regression test for the advertised `wasm32-unknown-unknown` build target.
//!
//! `TODO.md` and the crate README advertise that `oxibonsai-runtime` compiles
//! for `wasm32-unknown-unknown` (see `wasm_api.rs`). That claim silently rotted
//! because `pub mod engine_pool;` in `lib.rs` was not `cfg`-gated the same way
//! `pub mod async_engine;` is, and `engine_pool.rs` unconditionally
//! `use`s `tokio::sync::{OwnedSemaphorePermit, Semaphore}` while `tokio` itself
//! is confined to `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` in
//! `Cargo.toml`. The crate therefore failed to compile for `wasm32-unknown-unknown`
//! with `error[E0433]: cannot find module or crate 'tokio'`, and nothing caught
//! it because the repository has no CI workflow.
//!
//! This test shells out to `cargo check` for the exact command the acceptance
//! criteria specify, reusing the workspace's shared `target/` directory (so it
//! is fast once dependencies are warm) and skips gracefully — rather than
//! failing — on machines that do not have the `wasm32-unknown-unknown` target
//! installed, since that is an environment precondition, not a crate defect.

use std::path::Path;
use std::process::Command;

/// True if the active `rustc` toolchain has the `wasm32-unknown-unknown`
/// standard library installed (i.e. `rustup target add wasm32-unknown-unknown`
/// has been run), without depending on `rustup` being on `PATH`.
fn wasm32_target_installed() -> bool {
    let sysroot_output = match Command::new("rustc").arg("--print").arg("sysroot").output() {
        Ok(output) if output.status.success() => output,
        _ => return false,
    };
    let sysroot = String::from_utf8_lossy(&sysroot_output.stdout)
        .trim()
        .to_string();
    if sysroot.is_empty() {
        return false;
    }
    Path::new(&sysroot)
        .join("lib")
        .join("rustlib")
        .join("wasm32-unknown-unknown")
        .is_dir()
}

/// `oxibonsai-runtime` must compile for `wasm32-unknown-unknown` with the
/// default features disabled (the combination the README/TODO advertise as
/// the WASM build: no `server`/`rag` since those pull in `axum`/`tower`,
/// which are not wasm32-gated and are irrelevant to a WASM-embedding host
/// that talks to `wasm_api` directly).
#[test]
fn runtime_compiles_for_wasm32_unknown_unknown_no_default_features() {
    if !wasm32_target_installed() {
        eprintln!(
            "skipping runtime_compiles_for_wasm32_unknown_unknown_no_default_features: \
             wasm32-unknown-unknown target is not installed (run `rustup target add \
             wasm32-unknown-unknown` to enable this check)"
        );
        return;
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    let output = Command::new(env!("CARGO"))
        .args([
            "check",
            "-p",
            "oxibonsai-runtime",
            "--target",
            "wasm32-unknown-unknown",
            "--no-default-features",
        ])
        .current_dir(manifest_dir)
        .output()
        .expect("failed to spawn `cargo check` for wasm32-unknown-unknown");

    assert!(
        output.status.success(),
        "cargo check -p oxibonsai-runtime --target wasm32-unknown-unknown \
         --no-default-features failed (advertised WASM build is broken):\n\
         --- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
