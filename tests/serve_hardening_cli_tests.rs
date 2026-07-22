//! Regression test for finding security-02: the documented `oxibonsai serve`
//! entry point shipped with none of the auth/admission hardening the audit
//! added to the standalone `oxibonsai-serve` binary (no `--bearer-token`,
//! no concurrency/timeout admission stack — `axum::serve(listener,
//! router).await?` was called directly on the runtime's bare router).
//!
//! `Commands::Serve` now mounts the same building blocks (`cli::admission`):
//! optional constant-time bearer auth wrapping the whole router (so it also
//! protects `/admin/*`) plus a bounded-concurrency + per-request-timeout
//! admission stack. This test proves the CLI surface is real (`--help`
//! advertises the new flags); `src/cli/admission.rs`'s own unit tests cover
//! the middleware behavior itself (auth accept/reject, timeout, exemptions).

#![cfg(feature = "server")]

use std::process::Command;

#[test]
fn serve_help_advertises_hardening_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args(["serve", "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        output.status.success(),
        "`serve --help` should succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--bearer-token",
        "--max-concurrent-requests",
        "--request-timeout-ms",
    ] {
        assert!(
            stdout.contains(flag),
            "`serve --help` must advertise {flag}; got:\n{stdout}"
        );
    }
}
